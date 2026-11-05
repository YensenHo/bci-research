#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""preprocess_hbc_noise.py — 用「真实 HBC 体表信道噪声」替代纯高斯 AWGN，生成
「可穿戴脑电端侧蒸馏部署边界」实验数据集。

背景
----
现有 preprocess_add_noise.py 只加纯高斯 AWGN，是理想化噪声。真实可穿戴脑电的
端侧链路（电极贴头皮/体表，随人体活动）并非纯高斯，而是：

  1. 5bit ADC 量化     —— 端侧芯片的量化退化（qbits=5, qrange=±4，z-score 后信号 std=1）
  2. AWGN              —— 背景热噪声 / 链路噪声（σ/H，H=ref_amplitude=1.0）
  3. 突发 burst        —— 电极移动 / 肌肉活动的瞬态干扰（持续数十 ms，通道独立）
  4. 脉冲 impulse      —— 电极接触毛刺 / 静电放电的单点尖峰（稀疏、非高斯拖尾）

这四类分量复用 hbc_noise_advanced.py 的 hbc_noise() 接口（量化→AWGN→burst→impulse，
逐项独立、精确加性）。本脚本把它们注入 train（eval 保持干净，对齐现有加噪协议），
用来回答核心科学问题：**真实人体信道噪声下，蒸馏红利符号规律 sign(gain)=sign(t−s)
是否稳健？K/R 分解里 R 的翻转边界是否随非高斯分量移动？**

场景（--scenario）：
  awgn    : 纯 AWGN（基线，对齐 preprocess_add_noise 的高斯实验）
  quant   : 5bit 量化 + AWGN（端侧 ADC 量化退化）
  burst   : AWGN + 突发（电极移动瞬态）
  impulse : AWGN + 脉冲（接触毛刺/静电放电）
  full    : 量化 + AWGN + burst + impulse（真实端到端退化）

用法：
  env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin \
    /opt/miniconda3/envs/openmmlab/bin/python preprocess_hbc_noise.py \
    --sigma 1.0 --scenario full
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ALGO_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ALGO_DIR))

from hbc_noise_advanced import hbc_noise  # noqa: E402

# HBC 信道退化分量参数（对齐 hbc_noise_advanced 默认，中等强度保证有区分度）
QBITS = 5
QRANGE = 4.0
BURST_SIGMA_RATIO = 0.5   # 突发段内噪声 σ/H（= 信号幅度一半）
BURST_LEN = 16            # @250Hz ≈ 64ms 电极移动瞬态
BURST_PROB = 0.005        # 每个采样点作为突发起点的概率（期望覆盖 ≈ 8%）
IMPULSE_AMP = 3.0         # 冲激幅度（≈ 3 倍信号 std）
IMPULSE_PROB = 0.002      # 每个采样点出现冲激的概率（稀疏 0.2%）


def make_noise(x: torch.Tensor, scenario: str, sigma_ratio: float) -> torch.Tensor:
    """按场景注入 HBC 信道噪声，返回与 x 同形的加噪张量。"""
    if scenario == "awgn":
        return hbc_noise(x, qbits=0, sigma_ratio=sigma_ratio)
    if scenario == "quant":
        return hbc_noise(x, qbits=QBITS, qrange=QRANGE, sigma_ratio=sigma_ratio)
    if scenario == "burst":
        return hbc_noise(x, qbits=0, sigma_ratio=sigma_ratio,
                         burst_sigma_ratio=BURST_SIGMA_RATIO, burst_len=BURST_LEN,
                         burst_prob=BURST_PROB)
    if scenario == "impulse":
        return hbc_noise(x, qbits=0, sigma_ratio=sigma_ratio,
                         impulse_amp=IMPULSE_AMP, impulse_prob=IMPULSE_PROB)
    if scenario == "full":
        return hbc_noise(x, qbits=QBITS, qrange=QRANGE, sigma_ratio=sigma_ratio,
                         burst_sigma_ratio=BURST_SIGMA_RATIO, burst_len=BURST_LEN,
                         burst_prob=BURST_PROB, impulse_amp=IMPULSE_AMP,
                         impulse_prob=IMPULSE_PROB)
    raise ValueError(f"未知场景 {scenario}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sigma", type=float, required=True,
                    help="AWGN 的 σ/H（H=1.0，z-score 后信号幅度），对应 SNR=20·log10(H/σ)")
    ap.add_argument("--scenario", type=str, default="full",
                    choices=["awgn", "quant", "burst", "impulse", "full"])
    ap.add_argument("--src", type=str, default="data_4class_cross")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    src = ALGO_DIR / args.src
    tr = np.load(src / "train.npz")
    ev = np.load(src / "eval.npz")
    X_tr = tr["X"].astype(np.float32)   # (N, 1, C, T)
    y_tr = tr["y"].astype(np.int64)
    X_ev = ev["X"].astype(np.float32)
    y_ev = ev["y"].astype(np.int64)

    torch.manual_seed(args.seed)
    X_tr_t = torch.from_numpy(X_tr)
    X_tr_noisy = make_noise(X_tr_t, args.scenario, args.sigma)

    out = ALGO_DIR / f"{args.src}_hbc_{args.scenario}{args.sigma}"
    out.mkdir(exist_ok=True)
    np.savez(out / "train.npz", X=X_tr_noisy.numpy().astype(np.float32), y=y_tr)
    np.savez(out / "eval.npz", X=X_ev, y=y_ev)

    snr = 20.0 * np.log10(1.0 / args.sigma) if args.sigma > 0 else float("inf")
    print(f"场景 {args.scenario:8s} | σ/H={args.sigma} ({snr:.1f} dB) | "
          f"train {X_tr_noisy.shape} | 原始 std=1.0000 → 加噪后 std={X_tr_noisy.float().std():.4f}")
    print(f"✓ 已保存 {out}/train.npz + eval.npz")
    return 0


if __name__ == "__main__":
    sys.exit(main())
