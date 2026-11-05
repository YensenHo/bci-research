#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""scan_delta_k.py — 扫描 delta 调制阈值 k，找 clean 事件密度 25~30% 的甜区。

背景：输入脉冲编码报告.md 结论——clean 下密度 13.9%（k=0.5）信息不足（-22pp 主因），
      「全脉冲化可行边界 ≈ 输入事件密度 ≥25~30%」。本脚本扫更小的 k（θ=k·std 更细），
      统计 clean 密度 + 重构 SNR + 相关系数，定位密度甜区对应的 k 值。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ALGO = Path(__file__).resolve().parent
sys.path.insert(0, str(ALGO))
from delta_encode import compute_theta, delta_encode, delta_reconstruct  # noqa: E402


def main() -> int:
    npz = ALGO / "data_4class" / "train.npz"
    if not npz.exists():
        print(f"✗ 数据集缺失：{npz}")
        return 1
    with np.load(npz) as d:
        X = np.asarray(d["X"], dtype=np.float32)  # (n,1,22,500)

    print(f"数据: {X.shape}  (train 集 clean，无 HBC 增强)")
    print("=" * 72)
    print(f"{'k':>6} | {'clean密度':>9} | {'重构SNR':>8} | {'相关系数':>8} | 备注")
    print("-" * 72)

    for k in (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5):
        theta = compute_theta(X, k)
        pulses = delta_encode(torch.from_numpy(X), torch.from_numpy(theta),
                              bits=1, mode="signed").numpy()
        p = pulses[:, 0]  # (n, C, T)
        density = float((p != 0).mean())
        rec = delta_reconstruct(p, theta, X[:, 0, :, 0])
        mse = float(((rec - X[:, 0, :, :]) ** 2).mean())
        var = float(X[:, 0, :, :].var())
        snr = 10.0 * np.log10(var / mse) if mse > 0 else float("inf")
        corr = float(np.corrcoef(rec.ravel(), X[:, 0, :, :].ravel())[0, 1])
        mark = ""
        if 0.20 <= density <= 0.35:
            mark = "  ← 甜区 20~35%"
        print(f"{k:>6} | {density*100:8.2f}% | {snr:7.2f}dB | {corr:8.4f} |{mark}")

    print("=" * 72)
    print("参考：k=0.5 clean密度 13.9%（报告已知，-22pp）；目标 ≥25~30%。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
