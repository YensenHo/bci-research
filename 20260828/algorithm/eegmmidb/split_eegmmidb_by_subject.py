#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
split_eegmmidb_by_subject.py — EEGMMIDB 按被试切分（跨 subject LOSO 用）

复用 preprocess_eegmmidb.py 的 process_subject（21ch/160Hz/4类，cue 后 [0.5,2.5]s=320样本），
但按 subject 逐个保存 npz（而非合并 train/eval），供 LOSO 跨 subject 评估。

输出：eegmmidb/data_4class_by_subject/S{nnn}/data.npz（X (n,1,21,320), y (n,)）

用法：
  cd eegmmidb && env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin \
      /opt/miniconda3/envs/openmmlab/bin/python split_eegmmidb_by_subject.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

ALGO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ALGO)
os.chdir(ALGO)  # process_subject 用相对路径 RAW="data/raw"

from preprocess_eegmmidb import process_subject, SUBJECTS  # noqa: E402

OUT = os.path.join(ALGO, "data_4class_by_subject")


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    manifest = {}
    for subj in SUBJECTS:
        X, y = process_subject(subj)
        if X is None:
            print(f"S{subj:03d}: 无数据，跳过")
            continue
        subj_dir = os.path.join(OUT, f"S{subj:03d}")
        os.makedirs(subj_dir, exist_ok=True)
        np.savez_compressed(os.path.join(subj_dir, "data.npz"),
                            X=X.astype(np.float32), y=y.astype(np.int64))
        counts = np.bincount(y, minlength=4).tolist()
        manifest[f"S{subj:03d}"] = {"n_trials": int(len(y)), "class_counts": counts,
                                    "X_shape": list(X.shape)}
        print(f"S{subj:03d}: {len(y)} 试次 分布{counts} X{X.shape}")

    with open(os.path.join(OUT, "manifest.json"), "w") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    print(f"\n✓ 已写入 {OUT}（{len(manifest)} 个 subject）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
