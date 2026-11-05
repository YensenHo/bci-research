#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hbc_noise_advanced.py — 更真实的 HBC 体表信道噪声模型（约束驱动鲁棒解码的噪声前端）

背景
----
现有 noise.py（~/BCI/hbc-bci-sim/sim/noise.py）只建模了 AWGN + 均匀量化 + 热噪声底，
即「纯高斯」假设。但真实 HBC 体表信道（电极贴头皮/体表，随人体活动）并非纯高斯：
  - 电极移动 / 肌肉活动 → 瞬态「突发」干扰（某段时间内噪声幅度骤然抬升，持续数十毫秒）；
  - 电极接触毛刺 / 静电放电 → 单点「脉冲」尖峰（幅度大、稀疏、非高斯拖尾）。

本模块在 AWGN 之上叠加两类非高斯分量，提供可组合的噪声注入接口，并与 noise.py 的
5bit 量化函数配合（量化→反量化→叠加噪声，模拟「HBC 链路量化 + 信道噪声」的端到端退化）。

噪声分量（均为可叠加的加性分量，逐项独立注入）：
  - AWGN          : 加性高斯白噪声，σ = sigma_ratio × ref_amplitude（背景热噪声/链路噪声）
  - 突发噪声 burst : 以 burst_prob 概率在每个采样点作为突发起点，突发持续 burst_len 个采样点，
                     段内叠加 σ = burst_sigma_ratio × ref_amplitude 的高斯噪声（电极移动瞬态）
  - 脉冲噪声 impulse: 以 impulse_prob 概率在每个采样点注入 ±impulse_amp 的冲激（接触毛刺）

SNR 约定（与 task 对齐）：σ/H = sigma_ratio，H = ref_amplitude（归一化参考幅度 = 1.0），
  SNR_dB = 20·log10(H/σ)。故 sigma_ratio ∈ {0.05, 0.1, 0.2, 0.4} ≈ {26, 20, 14, 8} dB。

接口设计
--------
  - 函数式 API（与 noise.py 同风格）：awgn / burst_noise / impulse_noise / hbc_noise
  - 类封装 HBCNoiseModel（保存参数，可复用、可 describe 打印配置）
  - 输入任意 (..., T) 形状的 torch 张量（EEG 为 (B,1,C,T)，时间在最后一维）

自检
----
  env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin \\
      /opt/miniconda3/envs/openmmlab/bin/python hbc_noise_advanced.py
  验证：awgn 方差 / burst 覆盖与段内方差 / impulse 稀疏性与幅度 / 5bit 量化与 noise.py 一致 /
  分量可叠加（awgn+burst+impulse 的精确加性）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# 0. 复用 noise.py 的量化函数（numpy 参考实现，用于交叉校验 5bit 量化一致性）
# ---------------------------------------------------------------------------
_SIM_DIR = Path.home() / "BCI/hbc-bci-sim/sim"
if _SIM_DIR.exists():
    sys.path.insert(0, str(_SIM_DIR))

try:
    from noise import quantize_dequantize as _np_quantize_dequantize  # numpy 参考
    _HAS_NOISE_REF = True
except Exception:  # noqa: BLE001
    _np_quantize_dequantize = None
    _HAS_NOISE_REF = False


# ---------------------------------------------------------------------------
# 1. SNR ↔ σ/H 换算
# ---------------------------------------------------------------------------
def sigma_ratio_to_snr_db(sigma_ratio: float, ref_amplitude: float = 1.0) -> float:
    """σ/H → SNR(dB)。SNR = 20·log10(H/σ)，H 为参考幅度。"""
    sigma = abs(sigma_ratio) * ref_amplitude
    if sigma <= 0.0:
        return float("inf")
    return 20.0 * float(np.log10(ref_amplitude / sigma))


def snr_db_to_sigma_ratio(snr_db: float, ref_amplitude: float = 1.0) -> float:
    """SNR(dB) → σ/H。σ = H / 10^(SNR/20)。"""
    return ref_amplitude / (10.0 ** (snr_db / 20.0))


# ---------------------------------------------------------------------------
# 2. 5bit 均匀量化（torch 版，数学上严格对齐 noise.py 的 quantize_dequantize）
# ---------------------------------------------------------------------------
def quantize_dequantize(x: torch.Tensor, bits: int, qrange: float) -> torch.Tensor:
    """均匀量化 → 反量化。bits 位、动态范围 ±qrange（对齐 noise.py 的 clip→rint→dequant）。"""
    if bits <= 0:
        return x  # 0 表示不量化
    levels = 1 << bits
    xc = x.clamp(-qrange, qrange)                       # 先 clip（与 noise.py 一致）
    xq = ((xc + qrange) / (2.0 * qrange) * (levels - 1)).round()
    return xq / (levels - 1) * 2.0 * qrange - qrange


# ---------------------------------------------------------------------------
# 3. 各噪声分量（加性、可叠加）
# ---------------------------------------------------------------------------
def awgn(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """加性高斯白噪声。返回噪声分量（x + N(0, σ²)），σ 为绝对标准差。"""
    if sigma <= 0.0:
        return torch.zeros_like(x)
    return torch.randn_like(x) * sigma


def _burst_mask(starts: torch.Tensor, burst_len: int) -> torch.Tensor:
    """把突发起点 (B,T) 扩展为持续 burst_len 个采样点的突发段 mask (B,T)。

    语义：mask[t]=1 ⇔ 存在突发起点 s ∈ [t-burst_len+1, t]（即 t 落在某个突发段内）。
    实现：starts 左补 burst_len-1 个零后做 kernel=burst_len 的 max-pooling（滑窗取 OR）。
    """
    B, T = starts.shape
    if burst_len <= 1:
        return starts.float()
    padded = F.pad(starts.float().unsqueeze(1), (burst_len - 1, 0))  # (B,1,T+burst_len-1)
    mask = F.max_pool1d(padded, kernel_size=burst_len, stride=1).squeeze(1)  # (B,T)
    return mask


def burst_noise(x: torch.Tensor, sigma: float, burst_len: int, burst_prob: float) -> torch.Tensor:
    """突发噪声分量：以 burst_prob 概率开启持续 burst_len 采样点的突发段，段内叠加 σ 高斯噪声。

    x : 任意 (..., T) 张量，时间在最后一维（突发是时间维的瞬态事件，对所有通道同时生效）。
    返回 : 与 x 同形的加性突发噪声分量（段外恒为 0）。
    """
    if sigma <= 0.0 or burst_prob <= 0.0:
        return torch.zeros_like(x)
    T = x.shape[-1]
    B = x.numel() // T
    starts = torch.rand(B, T, device=x.device) < burst_prob       # (B,T) 突发起点
    mask = _burst_mask(starts, burst_len).view(x.shape)           # 扩展成段，广播回原形
    noise = torch.randn_like(x) * sigma
    return noise * mask


def impulse_noise(x: torch.Tensor, amp: float, prob: float) -> torch.Tensor:
    """脉冲噪声分量：以 prob 概率在每个采样点注入 ±amp 的冲激（符号随机）。

    返回 : 与 x 同形的稀疏冲激分量（非冲击点恒为 0）。
    """
    if amp <= 0.0 or prob <= 0.0:
        return torch.zeros_like(x)
    spikes = (torch.rand_like(x) < prob).float()
    sign = torch.where(torch.rand_like(x) < 0.5, -1.0, 1.0)
    return spikes * sign * amp


# ---------------------------------------------------------------------------
# 4. 组合噪声注入接口（核心）
# ---------------------------------------------------------------------------
def hbc_noise(
    x: torch.Tensor,
    qbits: int = 5,
    qrange: float = 4.0,
    ref_amplitude: float = 1.0,
    sigma_ratio: float = 0.0,
    burst_sigma_ratio: float = 0.0,
    burst_len: int = 16,
    burst_prob: float = 0.0,
    impulse_amp: float = 0.0,
    impulse_prob: float = 0.0,
) -> torch.Tensor:
    """HBC 体表信道端到端退化：5bit 量化 → AWGN + 突发 + 脉冲（可叠加）。

    参数
    ----
    qbits             : 量化位宽（0 = 不量化；板载 HBC 链路为 5bit）
    qrange            : 量化动态范围 ±qrange（对齐 train.py 的 4.0）
    ref_amplitude     : 参考信号幅度 H（SNR 归一化基准，默认 1.0）
    sigma_ratio       : AWGN 的 σ/H（σ = sigma_ratio × H），0 = 无 AWGN
    burst_sigma_ratio : 突发段噪声 σ_burst/H，0 = 无突发
    burst_len         : 突发持续采样点数（@250Hz，16 点 ≈ 64ms）
    burst_prob        : 每个采样点作为突发起点的概率（≈ 突发频率 × burst_len）
    impulse_amp       : 冲激幅度（绝对），0 = 无脉冲
    impulse_prob      : 每个采样点出现冲激的概率
    """
    xq = quantize_dequantize(x, qbits, qrange)
    out = xq + awgn(xq, sigma_ratio * ref_amplitude)
    if burst_sigma_ratio > 0.0 and burst_prob > 0.0:
        out = out + burst_noise(xq, burst_sigma_ratio * ref_amplitude, burst_len, burst_prob)
    if impulse_amp > 0.0 and impulse_prob > 0.0:
        out = out + impulse_noise(xq, impulse_amp, impulse_prob)
    return out


class HBCNoiseModel:
    """可复用、可打印配置的 HBC 噪声注入器（封装 hbc_noise 的全部参数）。"""

    def __init__(self, **kwargs):
        self.params = dict(
            qbits=5, qrange=4.0, ref_amplitude=1.0,
            sigma_ratio=0.0, burst_sigma_ratio=0.0, burst_len=16, burst_prob=0.0,
            impulse_amp=0.0, impulse_prob=0.0,
        )
        self.params.update(kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return hbc_noise(x, **self.params)

    def describe(self) -> str:
        p = self.params
        snr = sigma_ratio_to_snr_db(p["sigma_ratio"], p["ref_amplitude"])
        parts = [f"qbits={p['qbits']}"]
        if p["sigma_ratio"] > 0:
            parts.append(f"AWGN σ/H={p['sigma_ratio']:.2f} ({snr:.1f}dB)")
        if p["burst_prob"] > 0 and p["burst_sigma_ratio"] > 0:
            parts.append(f"burst σ/H={p['burst_sigma_ratio']:.2f} len={p['burst_len']} "
                         f"p={p['burst_prob']:.4f}")
        if p["impulse_prob"] > 0 and p["impulse_amp"] > 0:
            parts.append(f"impulse amp={p['impulse_amp']:.2f} p={p['impulse_prob']:.4f}")
        return "HBCNoise(" + ", ".join(parts) + ")"

    def __repr__(self) -> str:
        return self.describe()


# ---------------------------------------------------------------------------
# 5. 自检
# ---------------------------------------------------------------------------
def _selfcheck() -> int:
    torch.manual_seed(0)
    print("=" * 72)
    print("HBC 体表信道噪声模型自检（hbc_noise_advanced.py）")
    print("=" * 72)

    # ---- 5.0 SNR 映射表（对齐 task：σ/H ∈ {0.05,0.1,0.2,0.4} → 26/20/14/8 dB）----
    print("\n[SNR 映射]  σ/H → SNR(dB)  (H=1.0)")
    for r in (0.05, 0.1, 0.2, 0.4):
        print(f"    σ/H={r:.2f}  →  SNR = {sigma_ratio_to_snr_db(r):5.2f} dB")
    assert abs(sigma_ratio_to_snr_db(0.05) - 26.02) < 0.05
    assert abs(sigma_ratio_to_snr_db(0.1) - 20.0) < 0.05
    assert abs(sigma_ratio_to_snr_db(0.2) - 13.98) < 0.05
    assert abs(sigma_ratio_to_snr_db(0.4) - 7.96) < 0.05
    print("    ✓ SNR 映射正确")

    # ---- 5.1 5bit 量化与 noise.py numpy 参考实现一致 ----
    if _HAS_NOISE_REF:
        torch.manual_seed(0)
        x0 = torch.randn(1000) * 1.5
        mine = quantize_dequantize(x0, 5, 4.0).numpy()
        ref = _np_quantize_dequantize(x0.numpy(), 5, 4.0)[0]
        maxdiff = float(np.max(np.abs(mine - ref)))
        assert maxdiff < 1e-6, f"5bit 量化与 noise.py 不一致，max_diff={maxdiff}"
        print(f"\n[量化] 5bit 与 noise.py numpy 参考 max_diff={maxdiff:.2e}  ✓")
    else:
        print("\n[量化] ⚠ 未找到 noise.py，跳过交叉校验")

    # ---- 5.2 AWGN 方差校验 ----
    torch.manual_seed(1)
    N = 50000
    z = torch.zeros(N)
    n = awgn(z, 0.1)
    est_std = float(n.std())
    print(f"\n[AWGN] 目标 σ=0.1000，估计 σ={est_std:.4f}（N={N}）")
    assert abs(est_std - 0.1) / 0.1 < 0.05, "AWGN 标准差偏差过大"
    print("    ✓ AWGN 标准差正确")

    # ---- 5.3 突发噪声：覆盖比例 + 段内方差 + 段外恒零 ----
    torch.manual_seed(2)
    B, T, blen, bprob, sigma = 8, 2000, 16, 0.005, 0.3
    z = torch.zeros(B, 1, 1, T)
    b = burst_noise(z, sigma, blen, bprob)
    nonzero = (b != 0)
    coverage = float(nonzero.float().mean())
    expected_cov = min(1.0, bprob * blen)  # 无重叠时的期望覆盖
    inside_std = float(b[nonzero].std())
    print(f"\n[突发] burst_len={blen} prob={bprob} σ={sigma}")
    print(f"    期望覆盖 ≈ {expected_cov*100:.2f}%，实测 {coverage*100:.2f}%")
    print(f"    段内噪声 σ ≈ {inside_std:.4f}（目标 {sigma:.4f}）")
    assert coverage > 0 and abs(coverage - expected_cov) / expected_cov < 0.35, "突发覆盖偏差过大"
    assert abs(inside_std - sigma) / sigma < 0.15, "突发段内方差偏差过大"
    print("    ✓ 突发噪声覆盖与段内方差正确")

    # ---- 5.4 脉冲噪声：稀疏性 + 幅度 ----
    torch.manual_seed(3)
    N = 20000
    z = torch.zeros(N)
    amp, prob = 2.0, 0.01
    i = impulse_noise(z, amp, prob)
    nnz = (i != 0)
    cnt = int(nnz.sum())
    amps = torch.unique(i[nnz].abs())
    print(f"\n[脉冲] amp={amp} prob={prob}，期望点数 ≈ {prob*N:.0f}，实测 {cnt}")
    print(f"    非零幅度集合 |·| = {sorted(amps.tolist())}")
    assert abs(cnt / N - prob) / prob < 0.25, "脉冲稀疏性偏差过大"
    assert amps.numel() == 1 and abs(float(amps[0]) - amp) < 1e-6, "脉冲幅度错误"
    print("    ✓ 脉冲噪声稀疏性与幅度正确")

    # ---- 5.5 叠加正确性：awgn + burst + impulse 精确加性 ----
    torch.manual_seed(4)
    x = torch.randn(4, 1, 8, 500)
    spec = dict(qbits=5, qrange=4.0, ref_amplitude=1.0, sigma_ratio=0.1,
                burst_sigma_ratio=0.3, burst_len=16, burst_prob=0.005,
                impulse_amp=1.0, impulse_prob=0.002)
    torch.manual_seed(99)
    combined = hbc_noise(x, **spec)
    torch.manual_seed(99)
    xq = quantize_dequantize(x, 5, 4.0)
    a = awgn(xq, 0.1)
    b = burst_noise(xq, 0.3, 16, 0.005)
    p = impulse_noise(xq, 1.0, 0.002)
    manual = xq + a + b + p
    maxdiff = float((combined - manual).abs().max())
    print(f"\n[叠加] hbc_noise == 量化 + awgn + burst + impulse（精确加性）max_diff={maxdiff:.2e}")
    assert torch.allclose(combined, manual, atol=1e-6), "噪声分量叠加不满足加性"
    print("    ✓ 噪声分量可叠加（加性正确）")

    # ---- 5.6 类封装 smoke test ----
    m = HBCNoiseModel(sigma_ratio=0.1, burst_sigma_ratio=0.3, burst_len=16, burst_prob=0.005)
    torch.manual_seed(5)
    out = m.forward(torch.randn(2, 1, 8, 500))
    print(f"\n[类] {m.describe()}")
    print(f"     输出 shape = {tuple(out.shape)}（应 (2,1,8,500)）")
    assert tuple(out.shape) == (2, 1, 8, 500)
    print("    ✓ HBCNoiseModel 封装可用")

    print("\n" + "=" * 72)
    print("✓ 全部自检通过")
    return 0


if __name__ == "__main__":
    sys.exit(_selfcheck())
