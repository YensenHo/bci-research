#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""preprocess_nclasses.py — 从多类数据集选前 N 类，构造难度梯度。

用于价值 3 的「难度 × 信噪比」相图：同一数据集（EMG 16ch）不同类别数（2/4/8/17），
在相同 σ 下测抗噪红利，画「难度 → 红利」曲线。
"""
import argparse
import numpy as np
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="myo_cross_17")
    ap.add_argument("--n-classes", type=int, required=True)
    ap.add_argument("--dst", default=None)
    args = ap.parse_args()
    dst = args.dst or f"{args.src}_c{args.n_classes}"

    out = Path(dst)
    out.mkdir(exist_ok=True)
    for split in ["train", "eval"]:
        d = np.load(f"{args.src}/{split}.npz")
        X, y = d["X"], d["y"]
        mask = y < args.n_classes
        X_sub, y_sub = X[mask], y[mask]
        np.savez(f"{dst}/{split}.npz", X=X_sub, y=y_sub)
        print(f"{split}: {len(X_sub)} 样本，{len(np.unique(y_sub))} 类（从 {len(X)} 降采样）")


if __name__ == "__main__":
    main()
