#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""preprocess_ninapro.py — 把 Ninapro 肌电数据切成跨被试 train/eval（与 EEG 数据同格式）。

Ninapro DB 系列：X (N, 1, 16, 256)、y (N,)、subject (N, 1..10)。
跨被试划分：前 8 被试训练，后 2 被试评估（对齐 EEG 的 cross-subject 设定）。

输出：myo_cross/train.npz + eval.npz（X float32、y int64），供 sign_check.py / kd_decomposition.py 直接读。

用法：
  env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin \
    /opt/miniconda3/envs/openmmlab/bin/python preprocess_ninapro.py --db ninapro12
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ALGO_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ALGO_DIR))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=str, default="ninapro12",
                    choices=["ninapro12", "ninapro17", "ninapro52"])
    ap.add_argument("--n-train-subj", type=int, default=8)
    args = ap.parse_args()

    src = np.load(ALGO_DIR / "myo_text" / f"{args.db}.npz", allow_pickle=True)
    X = src["X"].astype(np.float32)
    y = src["y"].astype(np.int64)
    subj = src["subject"].astype(np.int64)

    subj_ids = np.unique(subj)
    train_subj = subj_ids[: args.n_train_subj]
    eval_subj = subj_ids[args.n_train_subj:]

    tr_mask = np.isin(subj, train_subj)
    ev_mask = np.isin(subj, eval_subj)
    X_tr, y_tr = X[tr_mask], y[tr_mask]
    X_ev, y_ev = X[ev_mask], y[ev_mask]

    out_dir = ALGO_DIR / ("myo_cross" if args.db == "ninapro12" else f"myo_cross_{args.db[-2:]}")
    out_dir.mkdir(exist_ok=True)
    np.savez(out_dir / "train.npz", X=X_tr, y=y_tr)
    np.savez(out_dir / "eval.npz", X=X_ev, y=y_ev)

    print(f"{args.db}: 训练被试 {train_subj}（{len(X_tr)} 样本），评估被试 {eval_subj}（{len(X_ev)} 样本）")
    print(f"  类别数 {len(np.unique(y))}，通道 {X.shape[2]}，时间 {X.shape[3]}")
    print(f"✓ 已保存 {out_dir}/train.npz + eval.npz")
    return 0


if __name__ == "__main__":
    sys.exit(main())
