#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""preprocess_eegmmidb_subset.py — EEGMMIDB 4类 减训练被试预处理（构造负红利第4个）。

复用 preprocess_eegmmidb.py 的 process_subject，但训练被试可配（默认前 12 个 S001-S012，
样本减半 ~1080 → teacher 35K 更易过拟合弱于 student）。评估被试固定 S025-S030。

用法：
  env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin \
    /opt/miniconda3/envs/openmmlab/bin/python preprocess_eegmmidb_subset.py \
    --train-n 12 --out data12
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

ALGO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ALGO)
EEG_DIR = os.path.join(ALGO, "eegmmidb")
sys.path.insert(0, EEG_DIR)  # 让 from preprocess_eegmmidb import 能找到
os.chdir(EEG_DIR)  # process_subject 用相对路径 RAW="data/raw"，需 cwd=eegmmidb
from preprocess_eegmmidb import process_subject  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-n", type=int, default=12, help="前 N 个被试训练（默认 12）")
    ap.add_argument("--out", type=str, default="data12")
    args = ap.parse_args()

    train_subs = list(range(1, args.train_n + 1))
    eval_subs = list(range(25, 31))
    subjs = train_subs + eval_subs

    train_X, train_y, eval_X, eval_y = [], [], [], []
    for subj in subjs:
        X, y = process_subject(subj)
        if X is None:
            print(f"S{subj:03d}: 无数据")
            continue
        print(f"S{subj:03d}: {len(y)} 试次  类别 {np.bincount(y, minlength=4).tolist()}")
        if subj in train_subs:
            train_X.append(X); train_y.append(y)
        else:
            eval_X.append(X); eval_y.append(y)

    out_dir = os.path.join(EEG_DIR, args.out)
    os.makedirs(out_dir, exist_ok=True)
    X_tr = np.concatenate(train_X, axis=0)
    y_tr = np.concatenate(train_y, axis=0)
    X_ev = np.concatenate(eval_X, axis=0)
    y_ev = np.concatenate(eval_y, axis=0)
    np.savez_compressed(os.path.join(out_dir, "train.npz"), X=X_tr, y=y_tr)
    np.savez_compressed(os.path.join(out_dir, "eval.npz"), X=X_ev, y=y_ev)
    meta = {"train_subs": train_subs, "eval_subs": eval_subs,
            "n_train": len(y_tr), "n_eval": len(y_ev),
            "train_dist": np.bincount(y_tr, minlength=4).tolist(),
            "eval_dist": np.bincount(y_ev, minlength=4).tolist()}
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    print(f"TRAIN: {meta['n_train']} {meta['train_dist']}")
    print(f"EVAL : {meta['n_eval']} {meta['eval_dist']}")
    print(f"DONE → {out_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
