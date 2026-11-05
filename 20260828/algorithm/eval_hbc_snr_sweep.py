#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_hbc_snr_sweep.py — HBC SNR 扫描评估：增强 vs 无增强 的鲁棒性对比

目的
----
对训练好的 LIF-EEG-TCNet 权重（无增强对照 / HBC 噪声增强），在 eval 集
（cross-session，E 会话）上做两个维度的真实评估：
  a. clean 四分类精度（增强是否掉 clean）；
  b. 低 SNR 鲁棒性：SNR 扫描（clean/20/16/12/8/4 dB）在 eval 上加噪测试，
     看增强模型在低 SNR 是否更稳（退化更缓、崩塌拐点后移）。

噪声协议与鲁棒性深化结果.md（EEGNet 已验证）一致：
  - 测试噪声 = 5bit 量化（qrange=4.0）+ AWGN，σ/H = 10^(-SNR/20)
  - 测试期噪声固定种子 TEST_NOISE_SEED=20240828：所有模型看到同一份噪声，公平对比
  - 附加 8dB+突发 测试域：验证「训练带突发增强」是否对突发测试域更鲁棒

用法
----
  env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin \\
      /opt/miniconda3/envs/openmmlab/bin/python eval_hbc_snr_sweep.py \\
      --ckpts eeg_tcnet_4class_lif_hbc_noaug.pth,eeg_tcnet_4class_lif_hbc_aug.pth \\
      --names 无增强,HBC增强

输出：hbc_aug_results/snr_sweep.json（全部原始数字）+ 控制台对比表
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ALGO_DIR = Path(__file__).resolve().parent
PROJECT = Path("/Users/heyixuan/BCI/luckfox-deploy")
DATA_DIR = PROJECT / "data"
DATA_4CLASS = ALGO_DIR / "data_4class"

sys.path.insert(0, str(ALGO_DIR))
sys.path.insert(0, str(DATA_DIR))

from dataset import EEGDataset          # noqa: E402
from eeg_tcnet_lif import EEGTCNetLIF   # noqa: E402
from hbc_noise_advanced import hbc_noise, snr_db_to_sigma_ratio  # noqa: E402

TEST_NOISE_SEED = 20240828  # 测试期噪声固定种子：所有模型看到同一份噪声实现


def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _build_model(ckpt: dict) -> EEGTCNetLIF:
    """按 ckpt 元信息重建模型（model_args + lif_hparams）。"""
    return EEGTCNetLIF(**ckpt["model_args"], **ckpt.get("lif_hparams", {}))


def _eval_noise(model, loader, device, noise_spec: dict | None) -> float:
    """在 eval 集上评估。noise_spec=None 表示干净；否则施加 hbc_noise。

    每次调用前重设 torch.manual_seed(TEST_NOISE_SEED)，保证不同模型
    看到同一份噪声实现（公平对比，与 robust_ablation.py 一致）。
    """
    model.eval()
    torch.manual_seed(TEST_NOISE_SEED)
    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            if noise_spec is not None:
                xb = hbc_noise(xb, **noise_spec)
            correct += (model(xb).argmax(1) == yb).sum().item()
            total += yb.size(0)
    return correct / total


# ---- SNR 扫描档位：clean / 20 / 16 / 12 / 8 / 4 dB（σ/H = 10^(-SNR/20)）----
# 附加 8dB+突发 测试域（训练带突发增强的针对性检验）
def _build_sweep() -> list[tuple[str, dict | None]]:
    points = [("clean(∞dB)", None)]
    for snr_db in (20.0, 16.0, 12.0, 8.0, 4.0):
        sigma = snr_db_to_sigma_ratio(snr_db)
        points.append((f"{snr_db:g}dB", dict(qbits=5, qrange=4.0, sigma_ratio=sigma)))
    # 附加：8dB AWGN + 突发（电极移动瞬态），与训练增强的突发分量同参数
    points.append(("8dB+突发", dict(qbits=5, qrange=4.0,
                                    sigma_ratio=snr_db_to_sigma_ratio(8.0),
                                    burst_sigma_ratio=0.3, burst_len=16, burst_prob=0.005)))
    return points


def main() -> int:
    ap = argparse.ArgumentParser(description="HBC SNR 扫描评估（增强 vs 无增强）")
    ap.add_argument("--ckpts", type=str, required=True,
                    help="逗号分隔的权重路径（相对 algorithm/ 或绝对路径）")
    ap.add_argument("--names", type=str, default="",
                    help="逗号分隔的展示名（缺省用文件名）")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--out-dir", type=str, default=str(ALGO_DIR / "hbc_aug_results"))
    args = ap.parse_args()

    ckpt_paths = [Path(p.strip()) for p in args.ckpts.split(",") if p.strip()]
    ckpt_paths = [p if p.is_absolute() else ALGO_DIR / p for p in ckpt_paths]
    names = [s.strip() for s in args.names.split(",") if s.strip()] if args.names else []
    names = names + [p.stem for p in ckpt_paths[len(names):]]
    assert len(names) == len(ckpt_paths), "names 与 ckpts 数量不匹配"

    device = _pick_device()
    eval_ds = EEGDataset(DATA_4CLASS / "eval.npz")
    loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False)
    sweep = _build_sweep()
    print(f"设备       : {device}")
    print(f"评估集     : eval.npz（{len(eval_ds)} 试次，cross-session，E 会话）")
    print(f"测试噪声   : 5bit 量化 + AWGN，σ/H=10^(-SNR/20)；固定种子 {TEST_NOISE_SEED}\n")

    # ---- 逐模型逐档评估 ----
    per_model = {}
    for name, path in zip(names, ckpt_paths):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        model = _build_model(ckpt)
        missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=True)
        assert not missing and not unexpected, f"权重加载失败: {missing} / {unexpected}"
        model = model.to(device)
        model.eval()
        row = {"ckpt": str(path), "final_test_clean": round(ckpt.get("final_test", -1.0), 4)}
        for label, spec in sweep:
            acc = _eval_noise(model, loader, device, spec)
            row[label] = round(acc, 4)
        per_model[name] = row
        print(f"[{name}] 已评估（clean ckpt 记录 final_test={row['final_test_clean']*100:.2f}%）")

    # ---- 对比表（控制台）----
    print("\n" + "=" * 76)
    print("SNR 扫描对比表（final test @ eval，cross-session，2592 试次）")
    print("=" * 76)
    header = f"{'测试域':<12}" + "".join(f"{n:>14}" for n in names)
    print(header)
    for label, _ in sweep:
        cells = "".join(f"{per_model[n][label]*100:>13.2f}%" for n in names)
        print(f"{label:<12}{cells}")
    # Δ 行（最后一个模型 − 第一个模型）
    if len(names) == 2:
        print("-" * 76)
        for label, _ in sweep:
            delta = (per_model[names[1]][label] - per_model[names[0]][label]) * 100
            print(f"{label:<12}{'':>14}{delta:>+13.2f}pp")

    # ---- 落盘 JSON ----
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {
        "meta": {
            "model": "LIF-EEG-TCNet (brief, thr=1.0/tau=2.0/T=8, 9512 参数)",
            "data": "data_4class/eval.npz (22ch 四分类, 2592 试次, cross-session)",
            "test_noise_seed": TEST_NOISE_SEED,
            "noise_spec": "5bit 量化(qrange=4.0) + AWGN σ/H=10^(-SNR/20)；8dB+突发 附加域",
            "device": str(device),
        },
        "sweep_points": [label for label, _ in sweep],
        "per_model": per_model,
    }
    (out_dir / "snr_sweep.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n✓ 结果已落盘：{out_dir / 'snr_sweep.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
