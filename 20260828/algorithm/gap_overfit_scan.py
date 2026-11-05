#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""gap_overfit_scan.py — EEGMMIDB 4类上扫描教师容量，记录每档「过拟合 gap」，
与已有 30% 红利数据组合，标定「过拟合 gap 判负阈值」（事前判据的量化）。

用法：
  env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin \
    /opt/miniconda3/envs/openmmlab/bin/python gap_overfit_scan.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ALGO_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ALGO_DIR))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from eeg_tcnet import EEGTCNet  # noqa: E402

BRIEF = dict(kernel_length=63, tcn_kernel_size=12, pool1=4, pool2=8, readout_mode="gap")

# 教师容量配置（对齐 gap_scan.py）
TEACHER_CFGS = [(8, 16, 12), (8, 16, 16), (12, 24, 18), (16, 32, 24), (24, 48, 32), (32, 64, 48)]

# 已有 30% 红利（双 seed 均值，来自 gap_scan_compare.json）
KNOWN_GAIN = {1.00: 0.09, 1.53: -0.56, 2.14: -0.93, 3.70: -2.78, 6.77: -1.95, 14.21: -1.67}


def _pick_device():
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _eval(model, X, y, device, batch):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.from_numpy(X[i:i + batch]).to(device)
            yb = torch.from_numpy(y[i:i + batch]).to(device)
            correct += (model(xb).argmax(1) == yb).sum().item()
            total += yb.size(0)
    return correct / total


def main():
    device = _pick_device()
    tr = np.load(ALGO_DIR / "eegmmidb/data" / "train.npz")
    ev = np.load(ALGO_DIR / "eegmmidb/data" / "eval.npz")
    X_tr = tr["X"].astype(np.float32)
    y_tr = tr["y"].astype(np.int64)
    X_ev = ev["X"].astype(np.float32)
    y_ev = ev["y"].astype(np.int64)

    print(f"{'F1/F2/FT':>12} {'gap×':>6} {'train':>8} {'val':>8} {'过拟合gap':>10} {'30%红利(已知)':>14}")
    rows = []
    for f1, f2, ft in TEACHER_CFGS:
        torch.manual_seed(42)
        teacher = EEGTCNet(n_channels=21, n_classes=4, input_length=320,
                           F1=f1, D=2, F2=f2, FT=ft, dropout=0.25,
                           activation="elu", **BRIEF).to(device)
        n_params = sum(p.numel() for p in teacher.parameters() if p.requires_grad)
        opt = torch.optim.Adam(teacher.parameters(), lr=1e-3)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100)
        ds = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
        loader = DataLoader(ds, batch_size=16, shuffle=True)
        t0 = time.time()
        for _ in range(100):
            teacher.train()
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                F.cross_entropy(teacher(xb), yb).backward()
                opt.step()
            sched.step()
        train_acc = _eval(teacher, X_tr, y_tr, device, 16)
        val_acc = _eval(teacher, X_ev, y_ev, device, 16)
        gap = (train_acc - val_acc) * 100
        ratio = n_params / 9496
        gain = KNOWN_GAIN.get(round(ratio, 2), float("nan"))
        rows.append((ratio, gap, gain, val_acc))
        print(f"{f'{f1}/{f2}/{ft}':>12} {ratio:>6.2f} {train_acc*100:>7.2f}% {val_acc*100:>7.2f}% "
              f"{gap:>+9.2f}pp {gain:>+13.2f}pp（{time.time()-t0:.0f}s）")

    print("\n=== 过拟合 gap → 红利 关系（标定判负阈值）===")
    rows_sorted = sorted(rows, key=lambda r: r[1])  # 按过拟合 gap 排序
    for ratio, gap, gain, val in rows_sorted:
        print(f"  过拟合gap {gap:+7.2f}pp → 30%红利 {gain:+6.2f}pp (gap×{ratio:.2f}, teacher val {val*100:.1f}%)")

    # 保存
    import json
    out = [{"f1f2ft": f"{f1}/{f2}/{ft}", "gap_ratio": r[0], "overfit_gap_pp": r[1],
            "gain_30pct_pp": r[2], "teacher_val": r[3]}
           for (f1, f2, ft), r in zip(TEACHER_CFGS, rows)]
    (ALGO_DIR / "gap_overfit_scan.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n✓ 已保存 gap_overfit_scan.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
