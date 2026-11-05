#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""preprocess_add_noise.py — 给训练集加 AWGN，构造受控的「信噪比渐变」数据集。

目标：验证「信噪比 → teacher 强弱 → 红利符号」的因果链（回应 Yuan 2020 冲突 + 相关性质疑）。
在 data_4class_cross（高信噪比，teacher 强 +4.68 → 红利 +5.56）的 train 上加噪声，
eval 保持干净。预期：噪声 ↑ → 信噪比 ↓ → teacher 过拟合 → teacher−student ↓ → 红利 ↓/转负。

用法：
  env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin \
    /opt/miniconda3/envs/openmmlab/bin/python preprocess_add_noise.py --sigma 0.5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ALGO_DIR = Path(__file__).resolve().parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sigma", type=float, required=True, help="AWGN 标准差（加到 train）")
    ap.add_argument("--src", type=str, default="data_4class_cross")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    src = ALGO_DIR / args.src
    tr = np.load(src / "train.npz")
    ev = np.load(src / "eval.npz")
    X_tr = tr["X"].astype(np.float32)
    y_tr = tr["y"].astype(np.int64)
    X_ev = ev["X"].astype(np.float32)
    y_ev = ev["y"].astype(np.int64)

    rng = np.random.default_rng(args.seed)
    noise = rng.normal(0, args.sigma, size=X_tr.shape).astype(np.float32)
    X_tr_noisy = X_tr + noise

    out = ALGO_DIR / f"{args.src}_n{args.sigma}"
    out.mkdir(exist_ok=True)
    np.savez(out / "train.npz", X=X_tr_noisy, y=y_tr)
    np.savez(out / "eval.npz", X=X_ev, y=y_ev)

    # 信噪比：信号方差（z-score 后 ≈1）vs 噪声方差
    snr = 10 * np.log10(1.0 / (args.sigma ** 2))
    print(f"加噪 σ={args.sigma}：train {X_tr_noisy.shape}，SNR ≈ {snr:.1f} dB（信号方差≈1）")
    print(f"✓ 已保存 {out}/train.npz + eval.npz")
    return 0


if __name__ == "__main__":
    sys.exit(main())
