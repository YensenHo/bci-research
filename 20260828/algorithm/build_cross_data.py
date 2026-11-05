#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""build_cross_data.py — 从 data_4class_by_subject 构造 BCIC IV 2a 的 cross-subject 划分。

目的：消掉 v3 结论的混淆变量。当前对照用了两个不同数据集（EEGMMIDB 21ch/320 vs
BCIC IV 2a 22ch/500），天花板差异与数据集差异混在一起。本脚本用同一数据集
（BCIC IV 2a，22ch/500）构造两种划分：
  - within（已有 data_4class）：train = 9 被试 T 会话，eval = 9 被试 E 会话 → 高天花板
  - cross（本脚本 data_4class_cross）：train = 8 被试（T+E），eval = 1 被试（T+E）→ 低天花板
唯一变量 = 划分方式（which 决定天花板），通道/长度/样本结构一致。

用法：
  env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin \
    /opt/miniconda3/envs/openmmlab/bin/python build_cross_data.py \
    --train-subjects A01 A02 A03 A04 A05 A06 A07 A08 --eval-subjects A09
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ALGO_DIR = Path(__file__).resolve().parent
BY_SUBJECT = ALGO_DIR / "data_4class_by_subject"
OUT = ALGO_DIR / "data_4class_cross"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-subjects", type=str, nargs="+",
                    default=[f"A{i:02d}" for i in range(1, 9)])
    ap.add_argument("--eval-subjects", type=str, nargs="+", default=["A09"])
    args = ap.parse_args()

    train_X, train_y, eval_X, eval_y = [], [], [], []
    for subj in args.train_subjects:
        for sess in ("train", "eval"):   # T+E 会话都进 train
            p = BY_SUBJECT / subj / f"{sess}.npz"
            if not p.exists():
                raise SystemExit(f"✗ 缺失 {p}")
            d = np.load(p)
            train_X.append(d["X"]); train_y.append(d["y"])
    for subj in args.eval_subjects:
        for sess in ("train", "eval"):
            p = BY_SUBJECT / subj / f"{sess}.npz"
            if not p.exists():
                raise SystemExit(f"✗ 缺失 {p}")
            d = np.load(p)
            eval_X.append(d["X"]); eval_y.append(d["y"])

    OUT.mkdir(exist_ok=True)
    X_tr = np.concatenate(train_X, axis=0)
    y_tr = np.concatenate(train_y, axis=0)
    X_ev = np.concatenate(eval_X, axis=0)
    y_ev = np.concatenate(eval_y, axis=0)
    np.savez_compressed(OUT / "train.npz", X=X_tr.astype(np.float32), y=y_tr.astype(np.int64))
    np.savez_compressed(OUT / "eval.npz", X=X_ev.astype(np.float32), y=y_ev.astype(np.int64))
    print(f"✓ cross-subject 数据已生成：{OUT}")
    print(f"  train: X={X_tr.shape} y={y_tr.shape} 类别分布 {np.bincount(y_tr, minlength=4).tolist()}")
    print(f"  eval : X={X_ev.shape} y={y_ev.shape} 类别分布 {np.bincount(y_ev, minlength=4).tolist()}")
    print(f"  划分: train={args.train_subjects}, eval={args.eval_subjects}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
