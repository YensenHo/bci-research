#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""calibrate_probe_per_subject_ft.py — 诊断探针：逐被试 few-shot 微调（真实产品场景）

主矩阵（calibrate_fewshot.py）协议 = 校准 A01 前 N 试次、测全部 9 人剩余；
该协议下微调无增益（向 A01 域漂移、伤及其他 8 人）。但产品真实场景是
**一人一设备**：新用户录 N 试次校准，之后只用自己的设备（自己的脑信号）。
本探针检验该真实场景：每个被试（9 人）用自己的前 N 试次校准（seq）、
微调预训练模型（全参数 lr=3e-4），只测自己剩余试次。N∈{10,20}，双 seed。
结果仅作报告补充（产品化意义），不进主矩阵。
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ALGO_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ALGO_DIR))

from calibrate_fewshot import (  # noqa: E402
    N_SUBJECTS, TRIALS_PER_SUBJECT, evaluate, load_model, _pick_device,
)

LR = 3e-4
EPOCHS = 20
FIXED_EP = 10
SEEDS = [42, 123]
N_LIST = [10, 20]


def run_ft(model, Xc, yc, Xt, yt, device, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    m = copy.deepcopy(model)
    m.train()
    ds = TensorDataset(torch.from_numpy(Xc), torch.from_numpy(yc))
    loader = DataLoader(ds, batch_size=min(16, len(ds)), shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    opt = torch.optim.Adam(m.parameters(), lr=LR)
    crit = nn.CrossEntropyLoss()
    curve = []
    for _ in range(EPOCHS):
        m.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(m(xb), yb)
            loss.backward()
            opt.step()
        curve.append(evaluate(m, Xt, yt, device)["acc"])
    return curve[FIXED_EP - 1], max(curve), int(np.argmax(curve)) + 1


def main():
    device = _pick_device(False)
    model = load_model(device)
    with np.load(ALGO_DIR / "data_4class/eval.npz", allow_pickle=False) as d:
        X = np.asarray(d["X"], dtype=np.float32)
        y = np.asarray(d["y"], dtype=np.int64)
    for N in N_LIST:
        rows = []
        for s in range(N_SUBJECTS):
            sl = slice(s * TRIALS_PER_SUBJECT, (s + 1) * TRIALS_PER_SUBJECT)
            Xs, ys = X[sl], y[sl]
            cal = np.arange(N)
            test = np.arange(N, len(ys))
            base = evaluate(model, Xs[test], ys[test], device)["acc"]
            fixes, bests, beps = [], [], []
            for sd in SEEDS:
                f, b, bep = run_ft(model, Xs[cal], ys[cal], Xs[test], ys[test],
                                   device, sd)
                fixes.append(f); bests.append(b); beps.append(bep)
            rows.append((s, base, fixes, bests, beps))
            print(f"  subj{s+1:02d} N={N}: 基线 {base*100:5.2f}% | "
                  f"微调 fixed{EPOCHS}ep {np.mean(fixes)*100:5.2f}% "
                  f"({(np.mean(fixes)-base)*100:+5.2f}pp) | "
                  f"best {np.mean(bests)*100:5.2f}% "
                  f"({(np.mean(bests)-base)*100:+5.2f}pp, ep "
                  f"{beps[0]}/{beps[1]})")
        bases = [r[1] for r in rows]
        fixes = [np.mean(r[2]) for r in rows]
        bests = [np.mean(r[3]) for r in rows]
        print(f"[逐被试微调 N={N}] 9 人均值：基线 {np.mean(bases)*100:.2f}% → "
              f"fixed10ep {np.mean(fixes)*100:.2f}% "
              f"(Δ { (np.mean(fixes)-np.mean(bases))*100:+.2f}pp) | "
              f"best {np.mean(bests)*100:.2f}% "
              f"(Δ {(np.mean(bests)-np.mean(bases))*100:+.2f}pp)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
