#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""compare_encoding_level.py — 对比「原始波形事件化」vs「能量包络事件化」的信息保留。

背景：delta 在原始波形层面事件化，相关系数仅 ~0.31（R²≈10%），丢了 ERD 需要的
     功率/能量信息。本脚本验证「带通滤波(mu+beta) → 功率包络 → 事件化」能否在
     「事件驱动传感器」语义下保留更多判别信息。

    若包络事件化相关系数显著高于原始波形，说明「特征层面事件化」是可行的
     事件驱动 EEG 前端方向，值得进一步训练分类器验证精度。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ALGO = Path(__file__).resolve().parent
sys.path.insert(0, str(ALGO))
from delta_encode import compute_theta, delta_encode, delta_reconstruct  # noqa: E402

FS = 250          # BCIC IV 2a 采样率
BAND = (8.0, 30.0)  # mu(8-12) + beta(13-30) 频段


def _bandpass(x: np.ndarray) -> np.ndarray:
    """FFT 带通滤波（mu+beta），x: (n, C, T) → (n, C, T)。"""
    T = x.shape[2]
    freqs = np.fft.rfftfreq(T, d=1.0 / FS)
    mask = (freqs >= BAND[0]) & (freqs <= BAND[1])
    Xf = np.fft.rfft(x, axis=2)
    Xf[:, :, ~mask] = 0
    return np.fft.irfft(Xf, n=T, axis=2)


def _envelope(x_band: np.ndarray, win: int = 50) -> np.ndarray:
    """滑动 RMS 功率包络（win=50 样本 ≈ 200ms）。"""
    sq = x_band ** 2
    cum = np.cumsum(np.pad(sq, ((0, 0), (0, 0), (win // 2, win - win // 2 - 1)),
                          mode="edge"), axis=2)
    denom = win
    env = np.sqrt((cum[:, :, win:] - cum[:, :, :-win]) / denom)
    return env


def _delta_corr(signal: np.ndarray, k: float) -> float:
    """对信号做 delta 事件化，重构后算相关系数（信息保留量）。signal: (n, C, T)。"""
    s4 = signal[:, None, :, :]  # (n,1,C,T)
    theta = compute_theta(s4, k)
    pulses = delta_encode(torch.from_numpy(s4), torch.from_numpy(theta),
                          bits=1, mode="signed").numpy()
    p = pulses[:, 0]
    rec = delta_reconstruct(p, theta, signal[:, :, 0])
    return float(np.corrcoef(rec.ravel(), signal.ravel())[0, 1])


def main() -> int:
    npz = ALGO / "data_4class" / "train.npz"
    if not npz.exists():
        print(f"✗ 数据集缺失：{npz}")
        return 1
    with np.load(npz) as d:
        X = np.asarray(d["X"], dtype=np.float32)  # (n,1,22,500)

    x = X[:, 0]  # (n, 22, 500)
    print(f"数据: {x.shape}  采样率 {FS}Hz")

    print("\n===== 原始波形层面事件化（baseline，已知 -22pp）=====")
    for k in (0.2, 0.3, 0.5):
        corr = _delta_corr(x, k)
        print(f"  k={k:<4} 重构相关系数 = {corr:.4f}")

    print("\n===== 能量包络层面事件化（带通 mu+beta → RMS 包络 → delta）=====")
    x_band = _bandpass(x)
    for win in (25, 50, 100):
        env = _envelope(x_band, win)
        print(f"  [窗口 {win} 样本 ≈ {win/FS*1000:.0f}ms] 包络均值={env.mean():.4f}")
        for k in (0.2, 0.3, 0.5):
            corr = _delta_corr(env, k)
            print(f"    k={k:<4} 包络重构相关系数 = {corr:.4f}")

    print("\n注：相关系数 > 0.31（原始波形 delta）即说明「特征层面事件化」保留更多信息。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
