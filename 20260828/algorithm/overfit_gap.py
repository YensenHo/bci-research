#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""overfit_gap.py — 验证「教师过拟合 gap（train_acc − val_acc）」能否作为蒸馏红利符号的事前判据。

v4 命题的判据是「sign(teacher − student 满标签)」，但这是后验（需先训练 student）。
本脚本验证更便宜的事前 proxy：训练 teacher 时就能得到的 train_acc − val_acc 过拟合 gap。

假设：
  · EEGMMIDB 4类（35K 教师，2160 样本，teacher 弱）→ 过拟合，train−val gap 大
  · BCIC IV 2a cross（35K 教师，4608 样本，teacher 强）→ 不过拟合，train−val gap 小

用法：
  env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin \
    /opt/miniconda3/envs/openmmlab/bin/python overfit_gap.py
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


def _pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
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


def train_teacher(data_dir, n_channels, input_length, seed=42, epochs=100, batch=16):
    device = _pick_device()
    torch.manual_seed(seed)
    np.random.seed(seed)
    tr = np.load(ALGO_DIR / data_dir / "train.npz")
    ev = np.load(ALGO_DIR / data_dir / "eval.npz")
    X_tr, y_tr = tr["X"].astype(np.float32), tr["y"].astype(np.int64)
    X_ev, y_ev = ev["X"].astype(np.float32), ev["y"].astype(np.int64)

    teacher = EEGTCNet(n_channels=n_channels, n_classes=4, input_length=input_length,
                       F1=16, D=2, F2=32, FT=24, dropout=0.25, activation="elu",
                       **BRIEF).to(device)
    opt = torch.optim.Adam(teacher.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    ds = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
    loader = DataLoader(ds, batch_size=batch, shuffle=True)
    t0 = time.time()
    for ep in range(epochs):
        teacher.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            F.cross_entropy(teacher(xb), yb).backward()
            opt.step()
        sched.step()
    train_acc = _eval(teacher, X_tr, y_tr, device, batch)
    val_acc = _eval(teacher, X_ev, y_ev, device, batch)
    n_params = sum(p.numel() for p in teacher.parameters() if p.requires_grad)
    print(f"[{data_dir}] 教师 {n_params:,} 参数 | train {train_acc*100:.2f}% | "
          f"val {val_acc*100:.2f}% | 过拟合gap {(train_acc-val_acc)*100:+.2f}pp "
          f"({time.time()-t0:.0f}s)")
    return train_acc, val_acc, (train_acc - val_acc) * 100, n_params


def main():
    cases = [
        ("eegmmidb/data", 21, 320),       # EEGMMIDB 4类，teacher 弱（负红利）
        ("data_4class_cross", 22, 500),   # BCIC IV 2a cross，teacher 强（正红利）
        ("data_4class", 22, 500),         # BCIC IV 2a within，teacher 强（正红利）
    ]
    print(f"{'数据/划分':<24} {'train':>8} {'val':>8} {'过拟合gap':>10}  预期红利符号")
    for data_dir, nch, nlen in cases:
        _, val, gap, np_ = train_teacher(data_dir, nch, nlen)
        expected = "负" if data_dir == "eegmmidb/data" else "正"
        print(f"{data_dir:<24} {val*100:>7.2f}% {'':>7} {gap:>+9.2f}pp  预期 {expected}（teacher "
              f"{'弱' if data_dir == 'eegmmidb/data' else '强'}）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
