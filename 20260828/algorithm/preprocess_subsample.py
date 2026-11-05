#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""preprocess_subsample.py — 从已有 train.npz 子采样训练集（保持类别均衡）。

用途：把 P300 等弱 regime 数据集减训练样本，逼出 teacher 过拟合 → 负红利强 regime 点。

用法：
  env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin \
    /opt/miniconda3/envs/openmmlab/bin/python preprocess_subsample.py \
    --src p300/data --out p300/data_small --n-train 1400 --seed 42
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ALGO_DIR = Path(__file__).resolve().parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=str, required=True, help="源目录（含 train.npz/eval.npz）")
    ap.add_argument("--out", type=str, required=True, help="输出目录")
    ap.add_argument("--n-train", type=int, required=True, help="子采样后的训练样本数")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    src = ALGO_DIR / args.src
    tr = np.load(src / "train.npz")
    ev = np.load(src / "eval.npz")
    X_tr = tr["X"].astype(np.float32)
    y_tr = tr["y"].astype(np.int64)
    X_ev = ev["X"].astype(np.float32)
    y_ev = ev["y"].astype(np.int64)

    # 按类别均衡子采样
    classes = np.unique(y_tr)
    per_cls = args.n_train // len(classes)
    rng = np.random.default_rng(args.seed)
    idxs = []
    for c in classes:
        ci = np.where(y_tr == c)[0]
        sel = rng.choice(ci, size=min(per_cls, len(ci)), replace=False)
        idxs.append(sel)
    idxs = np.sort(np.concatenate(idxs))
    X_sub = X_tr[idxs]
    y_sub = y_tr[idxs]

    out = ALGO_DIR / args.out
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "train.npz", X=X_sub, y=y_sub)
    np.savez(out / "eval.npz", X=X_ev, y=y_ev)
    print(f"子采样 {len(y_tr)} → {len(y_sub)}（每类 {per_cls}），类别分布 {np.bincount(y_sub).tolist()}")
    print(f"✓ 已保存 {out}/train.npz + eval.npz")
    return 0


if __name__ == "__main__":
    sys.exit(main())
