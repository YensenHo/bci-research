#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
edge_eval_pruned.py — 剪枝版端侧评估：参数量 / MACs / 前向耗时（MPS 参考）对比

对比对象：
  基线 22ch（eeg_tcnet_4class_lif_hbc_aug_s42.pth，板载实测 50.5ms @RV1106 朴素 C 循环）
  剪枝 16ch / 12ch（本次训练产物）

测量：
  1. 参数量（state_dict 实际统计）
  2. MACs（按 eeg_tcnet.py 的 macs_estimate 公式，brief 配置 KE=63/pool4/8/KT=12/T=500）
  3. MPS 单试次前向耗时（100 次重复取中位数，仅作主机参考，非板载）
  4. 板载耗时估算：按 MACs 比例缩放 50.5ms（conv1+depthwise 循环随 NCH 线性收缩，
     而 separable/TCN/LIF 与通道数无关，诚实标注为「估算」）

输出：channel_prune/edge_eval.json + 控制台对比表
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

ALGO_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ALGO_DIR))

from eeg_tcnet_lif import EEGTCNetLIF   # noqa: E402

BOARD_MS_22CH = 50.5  # 板载实测（RV1106，朴素 C 循环，来自 HBC噪声增强版板载实测说明.md）


def _macs_brief(n_channels: int) -> int:
    """按 eeg_tcnet.py macs_estimate 公式计算 brief 配置（T=500, KE=63, pool1=4, pool2=8,
    KT=12, F1=8, D=2, F2=16, FT=12, readout=gap）的 MACs。"""
    T, C = 500, n_channels
    F1, D, F2, FT = 8, 2, 16, 12
    KE, KT = 63, 12
    macs = 0
    macs += KE * F1 * C * T                       # conv1
    macs += C * F1 * D * T                        # depthwise
    t1 = T // 4
    macs += (16 + F2) * (F1 * D) * t1             # separable (sep_k=16)
    t2 = t1 // 8
    macs += KT * F2 * FT * t2                     # TCN block1 conv_a
    macs += KT * FT * FT * t2                     # TCN block1 conv_b
    macs += F2 * FT * t2                          # TCN block1 skip
    macs += KT * FT * FT * t2                     # TCN block2 conv_a
    macs += KT * FT * FT * t2                     # TCN block2 conv_b
    macs += FT * FT * t2                          # TCN block2 skip
    macs += FT * 4                                # fc
    return macs


def _load_model(path: Path, device: torch.device):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = EEGTCNetLIF(**ckpt["model_args"], **ckpt.get("lif_hparams", {}))
    model.load_state_dict(ckpt["state_dict"], strict=True)
    return model.to(device), ckpt


def _latency_mps(model, device, n_channels: int, reps: int = 60) -> float:
    """单试次前向耗时中位数（ms）。MPS 参考值，非板载。"""
    model.eval()
    x = torch.randn(1, 1, n_channels, 500, device=device)
    with torch.no_grad():
        for _ in range(5):          # warmup
            model(x)
        times = []
        for _ in range(reps):
            t0 = time.perf_counter()
            model(x)
            times.append((time.perf_counter() - t0) * 1000)
    return statistics.median(times)


def main() -> int:
    ap = argparse.ArgumentParser(description="剪枝版端侧评估")
    ap.add_argument("--ckpts", type=str, default=(
        "eeg_tcnet_4class_lif_hbc_aug_s42.pth,"
        "eeg_tcnet_4class_lif_hbc_aug_pr16.pth,"
        "eeg_tcnet_4class_lif_hbc_aug_pr12.pth"), help="逗号分隔权重路径")
    ap.add_argument("--names", type=str, default="基线22ch,剪枝16ch,剪枝12ch")
    args = ap.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    paths = [Path(p.strip()) for p in args.ckpts.split(",") if p.strip()]
    paths = [p if p.is_absolute() else ALGO_DIR / p for p in paths]
    names = [s.strip() for s in args.names.split(",") if s.strip()]

    print(f"设备 : {device} | 板载基线耗时参考 : {BOARD_MS_22CH} ms（RV1106 朴素 C 循环）\n")
    print(f"{'配置':<10}{'参数':>10}{'MACs(M)':>10}{'MACs降幅':>10}"
          f"{'MPS耗时(ms)':>14}{'板载估算(ms)':>14}")
    rows = []
    for name, path in zip(names, paths):
        if not path.exists():
            print(f"[跳过] {path} 不存在（训练未完成？）")
            continue
        model, ckpt = _load_model(path, device)
        nc = ckpt["model_args"]["n_channels"]
        n_params = sum(p.numel() for p in model.parameters())
        macs = _macs_brief(nc)
        lat = _latency_mps(model, device, nc)
        board_est = BOARD_MS_22CH * macs / _macs_brief(22)
        rows.append({"name": name, "n_channels": nc, "n_params": n_params,
                     "macs": macs, "mps_latency_ms": round(lat, 3),
                     "board_latency_est_ms": round(board_est, 2),
                     "final_test_clean": ckpt.get("final_test")})
        print(f"{name:<10}{n_params:>10,}{macs/1e6:>10.2f}{macs/_macs_brief(22)-1:>+9.1%}"
              f"{lat:>13.2f}{board_est:>13.2f}")

    if rows:
        out_dir = ALGO_DIR / "channel_prune"
        out_dir.mkdir(exist_ok=True)
        out = out_dir / "edge_eval.json"
        out.write_text(json.dumps({
            "board_22ch_reference_ms": BOARD_MS_22CH,
            "note": "板载估算 = 50.5ms × MACs比例（conv1+depthwise 随 NCH 线性收缩；"
                    "separable/TCN/LIF 与通道数无关）。MPS 耗时仅主机参考，非板载。",
            "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n✓ 结果已写入 {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
