#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""digit_distill.py — sklearn digits（8×8 图像，非脑电）上验证蒸馏符号判据。

目标：验证 sign(gain)=sign(teacher−student) 是否在非脑电图像上成立（通用规律 vs 脑电特有）。
用 MLP（teacher 大 + student 小），扫样本量（满/减样本）构造 teacher 强/弱场景。

用法：
  env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin PYTHONPATH= \
    /opt/miniconda3/envs/openmmlab/bin/python digit_distill.py --n-train 1437 --seed 42
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.datasets import load_digits

ALGO = Path(__file__).resolve().parent
DEV = "mps" if torch.backends.mps.is_available() else "cpu"


def load(n_train: int, seed: int):
    d = load_digits()
    X = d.data.astype(np.float32) / 16.0          # 归一化到 [0,1]
    y = d.target.astype(np.int64)
    n = len(y)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    # 固定 eval 最后 360，train 取前 n_train
    tr_idx = idx[:n_train]
    ev_idx = idx[n_train:n_train + 360]
    return X[tr_idx], y[tr_idx], X[ev_idx], y[ev_idx]


class MLP(nn.Module):
    def __init__(self, h1: int, h2: int = 0):
        super().__init__()
        layers = [nn.Linear(64, h1), nn.ReLU()]
        if h2:
            layers += [nn.Linear(h1, h2), nn.ReLU()]
            layers += [nn.Linear(h2, 10)]
        else:
            layers += [nn.Linear(h1, 10)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train(model, X, y, epochs=100, lr=1e-2, seed=42):
    torch.manual_seed(seed)
    model.to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    Xt = torch.tensor(X, device=DEV)
    yt = torch.tensor(y, device=DEV)
    for _ in range(epochs):
        opt.zero_grad()
        out = model(Xt)
        loss = F.cross_entropy(out, yt)
        loss.backward()
        opt.step()
    return model


def acc(model, X, y):
    model.eval()
    with torch.no_grad():
        out = model(torch.tensor(X, device=DEV))
        pred = out.argmax(1).cpu().numpy()
    return 100.0 * (pred == y).mean()


def distill_student(student, teacher, X_train, y_train,
                    alpha=0.5, T=4.0, epochs=100, seed=42):
    """用 teacher 软标签蒸馏 student（软标签在 X_train 上计算，与硬标签对齐）。"""
    torch.manual_seed(seed)
    teacher.eval()
    with torch.no_grad():
        soft = F.softmax(teacher(torch.tensor(X_train, device=DEV)) / T, dim=1)
    student.to(DEV)
    opt = torch.optim.Adam(student.parameters(), lr=1e-2)
    Xs = torch.tensor(X_train, device=DEV)
    ys = torch.tensor(y_train, device=DEV)
    sft = soft.to(DEV)
    for _ in range(epochs):
        opt.zero_grad()
        out = student(Xs)
        loss = alpha * F.cross_entropy(out, ys) + (1 - alpha) * F.kl_div(
            F.log_softmax(out / T, 1), sft, reduction="batchmean") * (T * T)
        loss.backward()
        opt.step()
    return student


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=1437)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--label-frac", type=float, default=0.3)
    args = ap.parse_args()

    X_tr, y_tr, X_ev, y_ev = load(args.n_train, args.seed)
    n_tr = len(y_tr)
    n_lab = int(n_tr * args.label_frac)

    # teacher 满标签（大 MLP 256×256）
    teacher = train(MLP(256, 256), X_tr, y_tr, seed=args.seed)
    tv = acc(teacher, X_ev, y_ev)
    # student 满标签（小 MLP 32）
    student_full = train(MLP(32), X_tr, y_tr, seed=args.seed)
    sv = acc(student_full, X_ev, y_ev)

    # hard 30%（小 MLP，只用 n_lab 标签）
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    lab_idx = rng.choice(n_tr, size=n_lab, replace=False)
    student_hard = train(MLP(32), X_tr[lab_idx], y_tr[lab_idx], seed=args.seed)
    hard = acc(student_hard, X_ev, y_ev)

    # distilled 30%（小 MLP，hard + teacher 软标签）
    student_dist = distill_student(MLP(32), teacher,
                                   X_tr[lab_idx], y_tr[lab_idx], seed=args.seed)
    dist = acc(student_dist, X_ev, y_ev)

    ts = tv - sv
    gain = dist - hard
    print(f"digits n_train={n_tr} seed={args.seed}: "
          f"teacher {tv:.1f}% student {sv:.1f}% (t−s {ts:+.2f}) | "
          f"hard {hard:.1f}% → dist {dist:.1f}% | 红利 {gain:+.2f}")
    print(f"  符号一致性: sign(gain)={'pos' if gain > 0 else 'neg'} vs sign(t−s)={'pos' if ts > 0 else 'neg'} "
          f"→ {'✓ 一致' if (ts > 0) == (gain > 0) else '✗ 不一致'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
