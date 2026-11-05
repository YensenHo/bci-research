#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""calibrate_probe_headonly.py — 诊断探针（非主矩阵）：微调方式是否影响 few-shot 结论

主实验（calibrate_fewshot.py）显示全参数微调在 N≤40 时测试精度从第 1 个 epoch 起
单调下滑。为排除"协议问题"（全参微调过猛 / lr 过高），本探针在 N=20 seq 上对比：
  - head-only：只微调 fc + 全部 BN（冻结卷积），lr=3e-4
  - full 低 lr：全参数微调，lr=1e-4
双 seed，固定 10 epoch 与回溯 best 两个口径。结果仅作诊断脚注，不进主矩阵。
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
    build_calibration, evaluate, load_model, _pick_device,
)
from eeg_tcnet_lif import EEGTCNetLIF  # noqa: E402

N = 20
LR = 3e-4
EPOCHS = 20
FIXED_EP = 10
SEEDS = [42, 123]


def freeze_params(model: nn.Module, mode: str) -> None:
    """mode='head'：冻结除 fc/BN 外全部；mode='full'：全部可训练。"""
    for name, p in model.named_parameters():
        if mode == "head":
            p.requires_grad = name.startswith("fc") or ".bn" in name
        else:
            p.requires_grad = True


def run(model, Xc, yc, Xt, yt, device, mode, lr, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    m = copy.deepcopy(model)
    freeze_params(m, mode)
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    m.train()
    ds = TensorDataset(torch.from_numpy(Xc), torch.from_numpy(yc))
    loader = DataLoader(ds, batch_size=len(ds), shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=lr)
    crit = nn.CrossEntropyLoss()
    curve = []
    for _ in range(1, EPOCHS + 1):
        m.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(m(xb), yb)
            loss.backward()
            opt.step()
        curve.append(evaluate(m, Xt, yt, device)["acc"])
    best = max(curve)
    return trainable, curve, best, int(np.argmax(curve)) + 1


def main():
    device = _pick_device(False)
    model = load_model(device)
    with np.load(ALGO_DIR / "data_4class/eval.npz", allow_pickle=False) as d:
        X = np.asarray(d["X"], dtype=np.float32)
        y = np.asarray(d["y"], dtype=np.int64)
    cal = build_calibration(len(y), N, "seq")
    test = np.setdiff1d(np.arange(len(y)), cal)
    Xc, yc, Xt, yt = X[cal], y[cal], X[test], y[test]
    base = evaluate(model, Xt, yt, device)["acc"]
    print(f"N={N} seq | 基线(无校准) {base*100:.2f}% | 校准类别分布 "
          f"{ {int(c): int((yc == c).sum()) for c in range(4)} }")
    for mode, lr in (("head", 3e-4), ("full", 1e-4)):
        res = []
        for sd in SEEDS:
            ntr, curve, best, bep = run(model, Xc, yc, Xt, yt, device, mode, lr, sd)
            res.append((curve, best, bep))
            print(f"[{mode} lr={lr}] seed{sd}: 可训练参数 {ntr:,} | "
                  f"fixed{EPOCHS}ep={curve[-1]*100:.2f}% "
                  f"fixed{FIXED_EP}ep={curve[FIXED_EP-1]*100:.2f}% "
                  f"best(ep{bep})={best*100:.2f}% | "
                  f"曲线ep1..{EPOCHS}={[round(v*100,1) for v in curve]}")
        fix = [r[0][FIXED_EP-1] for r in res]
        bs = [r[1] for r in res]
        print(f"  → 均值: fixed{FIXED_EP}ep={np.mean(fix)*100:.2f}% "
              f"(Δ { (np.mean(fix)-base)*100:+.2f}pp) | "
              f"best={np.mean(bs)*100:.2f}% (Δ {(np.mean(bs)-base)*100:+.2f}pp)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
