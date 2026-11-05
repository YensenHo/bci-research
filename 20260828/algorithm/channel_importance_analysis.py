#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
channel_importance_analysis.py — 通道重要性分析（借鉴 PlugSelect「即插即用通道选择」思想）

基线：LIF-EEG-TCNet + HBC 增强（eeg_tcnet_4class_lif_hbc_aug_s42.pth，clean 63.58% / 8dB 54.67%@MPS）

3 种方法（全部真实运行）：
  a. depthwise 权重范数  : depthwise.weight (F1*D, 1, 22, 1) 每通道核范数（跨滤波器 L2）。
                           每个滤波器对 22 个电极各有一个 (1,1) 权重，范数小 = 该电极贡献小。
  b. 逐通道消融          : eval 集上把第 c 通道整体置零，测精度下降（22 次全量推理，batch=128）。
                           下降越多 = 通道越重要。
  c. 脑区先验分组        : 额区(Fz,FCx) / 中央运动区(Cx,CPx) / 顶枕区(Px,POz)，
                           测「仅中央区」「仅外周」子集精度 + 各区域平均消融下降。

综合：score = 0.4·norm + 0.4·abl_drop + 0.2·region_prior（各指标 min-max 归一化到 [0,1]）。

输出：channel_importance.json + 控制台排序表
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ALGO_DIR = Path(__file__).resolve().parent
PROJECT = Path("/Users/heyixuan/BCI/luckfox-deploy")
DATA_DIR = PROJECT / "data"
DATA_4CLASS = ALGO_DIR / "data_4class"
BASELINE_CKPT = ALGO_DIR / "eeg_tcnet_4class_lif_hbc_aug_s42.pth"

sys.path.insert(0, str(ALGO_DIR))
sys.path.insert(0, str(DATA_DIR))

from dataset import EEGDataset          # noqa: E402
from eeg_tcnet_lif import EEGTCNetLIF   # noqa: E402

# ---- BCIC 2a 22 通道顺序（与 preprocess_4class.py 一致）----
EEG_22 = ["Fz", "FC3", "FC1", "FCz", "FC2", "FC4",
          "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
          "CP3", "CP1", "CPz", "CP2", "CP4",
          "P1", "Pz", "P2", "POz"]

# ---- 脑区先验分组 ----
REGIONS = {
    "额区(前额/Fz+FCx)": list(range(0, 6)),        # Fz, FC3, FC1, FCz, FC2, FC4
    "中央运动区(Cx+CPx)": list(range(6, 18)),       # C5..C6, CP3..CP4
    "顶枕区(Px+POz)": list(range(18, 22)),          # P1, Pz, P2, POz
}


def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _evaluate(model, loader, device, zero_ch=None, keep_ch=None) -> float:
    """评估。zero_ch: 置零的通道集合；keep_ch: 仅保留的通道集合（其余置零）。"""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            if zero_ch is not None:
                xb = xb.clone()
                xb[:, :, list(zero_ch), :] = 0.0
            elif keep_ch is not None:
                xb = xb.clone()
                xb[:, :, [c for c in range(22) if c not in keep_ch], :] = 0.0
            correct += (model(xb).argmax(1) == yb).sum().item()
            total += yb.size(0)
    return correct / total


def main() -> int:
    device = _pick_device()
    eval_ds = EEGDataset(DATA_4CLASS / "eval.npz")
    loader = DataLoader(eval_ds, batch_size=128, shuffle=False)
    print(f"设备 : {device} | eval 试次 : {len(eval_ds)} | 基线 : {BASELINE_CKPT.name}", flush=True)

    ckpt = torch.load(BASELINE_CKPT, map_location="cpu", weights_only=False)
    model = EEGTCNetLIF(**ckpt["model_args"], **ckpt.get("lif_hparams", {}))
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model = model.to(device)
    model.eval()
    print(f"模型参数 : {model.num_params():,} | 基线 final_test(clean)={ckpt['final_test']*100:.2f}%", flush=True)

    result = {"baseline_ckpt": str(BASELINE_CKPT), "channels": EEG_22,
              "regions": {k: [EEG_22[i] for i in v] for k, v in REGIONS.items()},
              "device": str(device)}

    # ---- 0) clean 参考 ----
    t0 = time.time()
    acc_clean = _evaluate(model, loader, device)
    result["acc_clean"] = round(acc_clean, 4)
    print(f"\n[参考] 全通道 clean 精度 : {acc_clean*100:.2f}%  ({time.time()-t0:.0f}s)", flush=True)

    # ---- a) depthwise 权重范数 ----
    dw = model.depthwise.weight.detach().cpu()          # (16, 1, 22, 1)
    norm_per_ch = dw.squeeze().pow(2).sum(0).sqrt().numpy()   # (22,)
    norm_rank = np.argsort(-norm_per_ch)                # 从大到小
    result["depthwise_norm"] = {EEG_22[c]: float(norm_per_ch[c]) for c in range(22)}
    print("\n[a] depthwise 核范数（跨 16 滤波器 L2）排序（大=重要）:", flush=True)
    for r, c in enumerate(norm_rank):
        print(f"    #{r+1:2d} {EEG_22[c]:>4s}  norm={norm_per_ch[c]:.4f}", flush=True)

    # ---- b) 逐通道消融（22 次全量推理）----
    abl_drop = {}
    print("\n[b] 逐通道消融（置零该通道，测 clean 精度下降）:", flush=True)
    for c in range(22):
        t0 = time.time()
        acc = _evaluate(model, loader, device, zero_ch={c})
        drop = acc_clean - acc
        abl_drop[c] = drop
        print(f"    置零 {EEG_22[c]:>4s}(idx {c:2d}) → {acc*100:5.2f}%  Δ={drop*100:+5.2f}pp  ({time.time()-t0:.0f}s)", flush=True)
    abl_rank = np.argsort(-np.array([abl_drop[c] for c in range(22)]))
    result["ablation_drop"] = {EEG_22[c]: round(float(abl_drop[c]), 4) for c in range(22)}

    # ---- c) 脑区先验分组 ----
    print("\n[c] 脑区先验分组（中央区 vs 额/顶枕区）:", flush=True)
    region_info = {}
    for rname, chs in REGIONS.items():
        t0 = time.time()
        acc_zero = _evaluate(model, loader, device, zero_ch=set(chs))   # 该区域整体置零
        drop_r = acc_clean - acc_zero
        region_info[rname] = {"channels": [EEG_22[i] for i in chs],
                              "acc_region_zeroed": round(float(acc_zero), 4),
                              "drop_pp": round(float(drop_r * 100), 2)}
        print(f"    {rname:<14} 置零后 {acc_zero*100:5.2f}%  Δ={drop_r*100:+5.2f}pp  ({time.time()-t0:.0f}s)", flush=True)
    # 仅中央区 / 仅外周 子集
    central = set(REGIONS["中央运动区(Cx+CPx)"])
    peripheral = set(range(22)) - central
    t0 = time.time()
    acc_central_only = _evaluate(model, loader, device, keep_ch=central)
    print(f"    仅中央运动区(12ch)   → {acc_central_only*100:5.2f}%  ({time.time()-t0:.0f}s)", flush=True)
    t0 = time.time()
    acc_periph_only = _evaluate(model, loader, device, keep_ch=peripheral)
    print(f"    仅外周(额+顶枕,10ch) → {acc_periph_only*100:5.2f}%  ({time.time()-t0:.0f}s)", flush=True)
    region_info["仅中央运动区12ch"] = {"acc": round(float(acc_central_only), 4)}
    region_info["仅外周10ch"] = {"acc": round(float(acc_periph_only), 4)}
    result["region_analysis"] = region_info

    # ---- 综合排序 ----
    def _minmax(vals):
        lo, hi = min(vals), max(vals)
        return [(v - lo) / (hi - lo) if hi > lo else 0.5 for v in vals]

    norm_mm = _minmax([norm_per_ch[c] for c in range(22)])
    abl_mm = _minmax([abl_drop[c] for c in range(22)])
    # region prior：该通道所属区域的平均消融下降（数据驱动，非手工设定）
    region_mean_drop = {}
    for rname, chs in REGIONS.items():
        region_mean_drop[rname] = float(np.mean([abl_drop[c] for c in chs]))
    reg_mm_vals = [region_mean_drop[r] for r in REGIONS]
    reg_lo, reg_hi = min(reg_mm_vals), max(reg_mm_vals)
    reg_prior = {}
    for c in range(22):
        for rname, chs in REGIONS.items():
            if c in chs:
                v = region_mean_drop[rname]
                reg_prior[c] = (v - reg_lo) / (reg_hi - reg_lo) if reg_hi > reg_lo else 0.5
                break

    final = {}
    for c in range(22):
        final[c] = 0.4 * norm_mm[c] + 0.4 * abl_mm[c] + 0.2 * reg_prior[c]
    order = sorted(range(22), key=lambda c: -final[c])

    print("\n" + "=" * 88)
    print("综合通道重要性排序（0.4·norm + 0.4·消融 + 0.2·区域先验，越大越重要）")
    print("=" * 88)
    print(f"{'排名':>4} {'通道':>5} {'区域':<14} {'norm':>8} {'消融Δpp':>8} {'区域先验':>8} {'综合分':>7}")
    table = []
    for r, c in enumerate(order):
        region = next(rn for rn, chs in REGIONS.items() if c in chs)
        print(f"{r+1:>4} {EEG_22[c]:>5} {region:<14} {norm_per_ch[c]:>8.4f} "
              f"{abl_drop[c]*100:>+8.2f} {reg_prior[c]:>8.3f} {final[c]:>7.3f}")
        table.append({"rank": r + 1, "idx": c, "channel": EEG_22[c], "region": region,
                      "norm": round(float(norm_per_ch[c]), 4),
                      "abl_drop_pp": round(float(abl_drop[c] * 100), 2),
                      "region_prior": round(float(reg_prior[c]), 3),
                      "score": round(float(final[c]), 4)})
    result["importance_ranking"] = table

    # ---- 剪枝建议：保 16ch（剪 6，去掉最不重要的 6 个）/ 保 12ch（剪 10）----
    # order 按重要性降序 → order[:K] 保留最重要的 K 个，order[K:] 为剪掉的低贡献通道
    keep16 = [c for c in order[:16]]          # 保留前 16（重要性最高）
    drop6 = [EEG_22[c] for c in order[16:]]   # 剪掉最低 6 个
    keep12 = [c for c in order[:12]]          # 保留前 12
    drop10 = [EEG_22[c] for c in order[12:]]  # 剪掉最低 10 个
    result["prune_keep16"] = {"keep_idx": keep16, "keep_ch": [EEG_22[c] for c in keep16],
                              "drop_ch": drop6}
    result["prune_keep12"] = {"keep_idx": keep12, "keep_ch": [EEG_22[c] for c in keep12],
                              "drop_ch": drop10}
    print("\n[剪枝方案]")
    print(f"  保 16ch：剪掉 {drop6} → 保留 {[EEG_22[c] for c in keep16]}")
    print(f"  保 12ch：剪掉 {drop10} → 保留 {[EEG_22[c] for c in keep12]}")

    out = ALGO_DIR / "channel_importance.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n✓ 结果已写入 {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
