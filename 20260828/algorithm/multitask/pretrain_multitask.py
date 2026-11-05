#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""pretrain_multitask.py — 7 任务有监督多任务预训练通道无关骨干

目标：回答「多范式系统整合」是否成立——把 7 个脑电范式（运动想象/癫痫/睡眠/情绪/P300 等）
的 train 集联合训练一个共享通道无关时间骨干，看它学到的通用时间特征能否迁移到下游单任务。

有监督多任务（不是自监督——自监督/STDP 三线已定案端侧无红利）：每步均匀抽一个任务，取一批，
过共享骨干 + 该任务头，CE loss（类别权重校正不均衡：睡眠 W70%、CHB 2:1、DEAP 61%）。

用法：
  env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin \
    /opt/miniconda3/envs/openmmlab/bin/python pretrain_multitask.py \
    --device mps --epochs 30 --out multitask_backbone_s42.pth
  # 冒烟：
  .../python pretrain_multitask.py --device mps --smoke --epochs 2 --steps-per-epoch 10
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from itertools import cycle
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from multitask_data import TASKS, load_all          # noqa: E402
from multitask_model import MultiTaskModel, build_backbone  # noqa: E402


def _pick_device(name: str) -> torch.device:
    if name in ("cuda", "auto") and torch.cuda.is_available():
        return torch.device("cuda")
    if name in ("mps", "auto") and getattr(torch.backends, "mps", None) \
            and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _class_weights(y: np.ndarray, ncls: int) -> torch.Tensor:
    c = np.bincount(y, minlength=ncls).astype(np.float32)
    c = np.where(c == 0, 1.0, c)
    return torch.tensor(len(y) / (ncls * c), dtype=torch.float32)


def main() -> int:
    ap = argparse.ArgumentParser(description="多任务预训练通道无关骨干")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--steps-per-epoch", type=int, default=0)  # 0=自动取最大任务 batch 数
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--backbone", default="v1", choices=["v1", "v2"])
    ap.add_argument("--H", type=int, default=8)
    ap.add_argument("--kernel", type=int, default=16)
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--H-per-scale", type=int, default=8)   # v2 多尺度每尺度特征数
    ap.add_argument("--n-heads", type=int, default=4)        # v2 空间注意力头数
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--tasks", default="")
    ap.add_argument("--out", default="multitask_backbone.pth")
    args = ap.parse_args()

    ALGO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # algorithm/
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))              # algorithm/multitask/
    cache_dir = os.path.join(SCRIPT_DIR, "multitask_cache")
    tasks = [t for t in (args.tasks.split(",") if args.tasks else []) if t] or list(TASKS)
    device = _pick_device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data = load_all(ALGO, cache_dir, tasks=tasks, smoke=args.smoke)
    specs = {t: (data[t]["ch"], data[t]["ncls"]) for t in tasks}
    kw = dict(H=args.H, kernel=args.kernel, n_layers=args.n_layers,
              H_per_scale=args.H_per_scale, n_heads=args.n_heads)
    bb, H = build_backbone(args.backbone, **kw)
    model = MultiTaskModel(bb, H, specs).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    loaders, cws = {}, {}
    for t in tasks:
        ds = TensorDataset(torch.from_numpy(data[t]["Xtr"]),
                           torch.from_numpy(data[t]["ytr"]))
        loaders[t] = cycle(DataLoader(ds, batch_size=args.batch, shuffle=True,
                                      drop_last=False))
        cws[t] = _class_weights(data[t]["ytr"], data[t]["ncls"]).to(device)

    steps = args.steps_per_epoch or max(
        (len(data[t]["Xtr"]) + args.batch - 1) // args.batch for t in tasks)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[pretrain] backbone={args.backbone} tasks={tasks} device={device} "
          f"steps/epoch={steps} epochs={args.epochs} lr={args.lr} params={n_params}")

    for ep in range(args.epochs):
        model.train()
        t0 = time.time()
        tot = 0.0
        rng = np.random.default_rng(args.seed * 1000 + ep)
        for _ in range(steps):
            t = tasks[int(rng.integers(len(tasks)))]
            xb, yb = next(loaders[t])
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = F.cross_entropy(model(t, xb), yb, weight=cws[t])
            loss.backward()
            opt.step()
            tot += loss.item()
        print(f"  epoch {ep + 1}/{args.epochs}  avg_loss={tot / steps:.4f}  "
              f"({time.time() - t0:.0f}s)", flush=True)

    out = os.path.join(SCRIPT_DIR, args.out)
    torch.save({
        "backbone": model.backbone.state_dict(),
        "backbone_name": args.backbone, "H": H,
        "H_per_scale": args.H_per_scale, "kernel": args.kernel,
        "n_layers": args.n_layers, "n_heads": args.n_heads,
        "tasks": tasks, "seed": args.seed, "smoke": args.smoke,
    }, out)
    print(f"✓ 骨干已保存: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
