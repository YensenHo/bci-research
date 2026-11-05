#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""downstream_transfer.py — 下游迁移评估：随机初始化 vs 冻结骨干 vs 微调骨干

核心对照（唯一变量 = 初始化）：
  1. baseline  ：随机初始化骨干，从头训（骨干 + 头）。
  2. frozen    ：加载预训练骨干并冻结，只训任务头（测「预训练时间特征是否本身有用」）。
  3. finetune  ：加载预训练骨干并微调（测「预训练是否是好的初始化」）。
三档同 epochs / lr / batch / seed，只差初始化来源。eval 是各任务 held-out 的 eval.npz，
预训练/下游头训练都不碰 eval → 无泄漏。fixed-epoch 无早停（可部署口径，非 best 回溯）。

输出：控制台对照表 + multitask_transfer_results.json。

用法：
  .../python downstream_transfer.py --ckpt multitask_backbone_s42.pth --device mps --epochs 30
  # 冒烟：
  .../python downstream_transfer.py --ckpt multitask_backbone_s42.pth --device mps --smoke --epochs 2
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from multitask_data import TASKS, load_all          # noqa: E402
from multitask_model import TaskHead, build_backbone  # noqa: E402


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


def _bacc(logits: torch.Tensor, y: torch.Tensor) -> float:
    pred = logits.argmax(1).cpu().numpy()
    y = y.cpu().numpy()
    recs = []
    for c in range(logits.shape[1]):
        m = y == c
        if m.sum() > 0:
            recs.append(float((pred[m] == c).mean()))
    return float(np.mean(recs))


def train_eval(task: str, d: dict, backbone: TemporalBackbone, device,
               epochs: int, batch: int, lr: float, freeze: bool):
    """在 task 的 train 上训（骨干[+头]），在 eval 上测。返回 (acc, bacc)。"""
    Xtr, ytr, Xev, yev = d["Xtr"], d["ytr"], d["Xev"], d["yev"]
    C, ncls = d["ch"], d["ncls"]
    H = backbone.H
    head = TaskHead(C * H, ncls).to(device)

    for p in backbone.parameters():
        p.requires_grad_(not freeze)
    params = list(head.parameters()) + ([] if freeze else list(backbone.parameters()))
    opt = torch.optim.Adam(params, lr=lr)

    ds = TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr))
    loader = DataLoader(ds, batch_size=batch, shuffle=True, drop_last=False)
    cw = _class_weights(ytr, ncls).to(device)

    for _ in range(epochs):
        backbone.train()
        head.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = head(backbone(xb))
            loss = F.cross_entropy(logits, yb, weight=cw)
            loss.backward()
            opt.step()

    backbone.eval()
    head.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(Xev), batch):
            xb = torch.from_numpy(Xev[i:i + batch]).to(device)
            outs.append(head(backbone(xb)).cpu())
    logits = torch.cat(outs)
    yt = torch.from_numpy(yev)
    acc = float((logits.argmax(1) == yt).float().mean())
    bacc = _bacc(logits, yt)
    return acc, bacc


def main() -> int:
    ap = argparse.ArgumentParser(description="下游迁移评估（baseline/frozen/finetune）")
    ap.add_argument("--ckpt", default="multitask_backbone.pth")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--label-frac", type=float, default=1.0)  # 下游 train 标注子采样比例
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--tasks", default="")
    args = ap.parse_args()

    ALGO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # algorithm/
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))              # algorithm/multitask/
    cache_dir = os.path.join(SCRIPT_DIR, "multitask_cache")
    device = _pick_device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ckpt = torch.load(os.path.join(SCRIPT_DIR, args.ckpt), map_location="cpu", weights_only=False)
    bname = ckpt.get("backbone_name", "v1")
    H = ckpt["H"]
    kw = dict(H=ckpt.get("H", 8), kernel=ckpt.get("kernel", 16),
              n_layers=ckpt.get("n_layers", 2),
              H_per_scale=ckpt.get("H_per_scale", 8), n_heads=ckpt.get("n_heads", 4))

    def _fresh():
        bb, _ = build_backbone(bname, **kw)
        return bb

    pretrained = _fresh()
    pretrained.load_state_dict(ckpt["backbone"])

    tasks = [t for t in (args.tasks.split(",") if args.tasks else []) if t] or list(TASKS)
    data = load_all(ALGO, cache_dir, tasks=tasks, smoke=args.smoke)

    if args.label_frac < 1.0:
        rng = np.random.default_rng(args.seed)
        for t in tasks:
            n = int(round(len(data[t]["Xtr"]) * args.label_frac))
            idx = rng.permutation(len(data[t]["Xtr"]))[:n]
            data[t]["Xtr"] = data[t]["Xtr"][idx]
            data[t]["ytr"] = data[t]["ytr"][idx]
        print(f"[downstream] label-frac={args.label_frac:g}（各任务 train 子采样）")

    print(f"[downstream] ckpt={args.ckpt} backbone={bname} H={H} "
          f"epochs={args.epochs} seed={args.seed}")
    results = {}
    for t in tasks:
        d = data[t]

        rb = _fresh().to(device)
        acc_b, bacc_b = train_eval(t, d, rb, device, args.epochs, args.batch,
                                   args.lr, freeze=False)

        pf = _fresh().to(device)
        pf.load_state_dict(pretrained.state_dict())
        acc_f, bacc_f = train_eval(t, d, pf, device, args.epochs, args.batch,
                                   args.lr, freeze=True)

        pt = _fresh().to(device)
        pt.load_state_dict(pretrained.state_dict())
        acc_ft, bacc_ft = train_eval(t, d, pt, device, args.epochs, args.batch,
                                     args.lr, freeze=False)

        results[t] = dict(
            baseline=dict(acc=acc_b, bacc=bacc_b),
            frozen=dict(acc=acc_f, bacc=bacc_f),
            finetune=dict(acc=acc_ft, bacc=bacc_ft),
        )
        print(f"[{t:8s}] base {acc_b*100:5.2f}% | "
              f"frozen {acc_f*100:5.2f}% (Δ{acc_f - acc_b:+5.2f}) | "
              f"ft {acc_ft*100:5.2f}% (Δ{acc_ft - acc_b:+5.2f})", flush=True)

    lf = f"_lf{args.label_frac:g}" if args.label_frac < 1.0 else ""
    suffix = "_smoke" if args.smoke else ""
    out = os.path.join(SCRIPT_DIR,
                       f"multitask_transfer_results_{bname}_s{args.seed}{lf}{suffix}.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"✓ 结果已保存: {out}")

    # 汇总：冻结/微调相对 baseline 的平均 Δacc
    d_f = np.mean([results[t]["frozen"]["acc"] - results[t]["baseline"]["acc"] for t in tasks])
    d_t = np.mean([results[t]["finetune"]["acc"] - results[t]["baseline"]["acc"] for t in tasks])
    print(f"[汇总] 平均 Δacc  frozen={d_f*100:+.2f}pp  finetune={d_t*100:+.2f}pp")
    return 0


if __name__ == "__main__":
    sys.exit(main())
