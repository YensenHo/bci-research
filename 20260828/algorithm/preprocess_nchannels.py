#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""preprocess_nchannels.py — 从多通道脑电数据取前 N 通道，构造「信息量梯度」。

用于普适验证「teacher 满标签崩到 < chance」：假设由「通道数（信息量）× 加噪」决定，
单通道信息量极低 → teacher 崩；多通道 → 不崩。通道数梯度是检验这个假设的关键。
"""
import argparse
import numpy as np
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="eegmmidb/data")
    ap.add_argument("--n-channels", type=int, required=True)
    ap.add_argument("--dst", default=None)
    args = ap.parse_args()
    dst = args.dst or f"{args.src}_ch{args.n_channels}"

    Path(dst).mkdir(parents=True, exist_ok=True)
    for split in ["train", "eval"]:
        d = np.load(f"{args.src}/{split}.npz")
        X, y = d["X"], d["y"]
        X_sub = X[:, :, :args.n_channels, :]  # 取前 N 通道
        np.savez(f"{dst}/{split}.npz", X=X_sub, y=y)
        print(f"{split}: {X_sub.shape}（取前 {args.n_channels} 通道）")


if __name__ == "__main__":
    main()
