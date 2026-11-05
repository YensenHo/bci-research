#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""preprocess_deap_small.py — DEAP 情感数据降采样，构造「低天花板 + 中等样本」负红利候选。

原始 deap/data：train (10560, 1, 32, 640) 2类（valence），32ch/640@128Hz。
降采样：时间 640→320（stride 2），样本子采样到 --n-samples（默认 3000）。
输出 deap_small/train.npz + eval.npz，供 sign_check.py 直接读。

用法：
  env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin \
    /opt/miniconda3/envs/openmmlab/bin/python preprocess_deap_small.py --n-samples 3000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ALGO_DIR = Path(__file__).resolve().parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-samples", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    tr = np.load(ALGO_DIR / "deap" / "data" / "train.npz")
    ev = np.load(ALGO_DIR / "deap" / "data" / "eval.npz")
    X_tr, y_tr = tr["X"].astype(np.float32), tr["y"].astype(np.int64)
    X_ev, y_ev = ev["X"].astype(np.float32), ev["y"].astype(np.int64)

    # ① 时间降采样 640 → 320（stride 2）
    X_tr = X_tr[:, :, :, ::2]
    X_ev = X_ev[:, :, :, ::2]

    # ② 训练样本子采样（保持类别比例）
    rng = np.random.default_rng(args.seed)
    n = min(args.n_samples, len(y_tr))
    idx = rng.permutation(len(y_tr))[:n]
    X_tr, y_tr = X_tr[idx], y_tr[idx]

    out = ALGO_DIR / "deap_small"
    out.mkdir(exist_ok=True)
    np.savez(out / "train.npz", X=X_tr, y=y_tr)
    np.savez(out / "eval.npz", X=X_ev, y=y_ev)
    print(f"DEAP 降采样：train {X_tr.shape}（{n} 样本，时间 640→320），eval {X_ev.shape}")
    print(f"  类别分布 train {np.bincount(y_tr)}，eval {np.bincount(y_ev)}")
    print(f"✓ 已保存 deap_small/train.npz + eval.npz")
    return 0


if __name__ == "__main__":
    sys.exit(main())
