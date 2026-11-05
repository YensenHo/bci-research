#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
delta_encode.py — 输入脉冲编码：delta 调制（增量调制，DM）

背景（延续 SNN 稀疏化结论）
----------------------------
前端 φ1（conv1+depthwise）处理连续 EEG 占 96.9% MACs、不可稀疏化；LIF 脉冲域仅 3.1%。
本模块把**连续 EEG 输入本身脉冲化**（delta 调制），让 φ1 的输入也进入稀疏脉冲域，
是「稀疏脑信号 → 稀疏类脑计算 → 稀疏人体传输」纲领在输入端的落点：
delta 脉冲 = 差分阈值事件，正是事件驱动传感器（DVS 相机同理）的输出形态。

编码语义
--------
  dx[t] = x[t] - x[t-1]（时间差分，因果：脉冲只依赖过去）
  1bit : p[t] = +1 if dx[t] >  θ，-1 if dx[t] < -θ，否则 0   （θ = k·std_通道）
  2bit : p[t] = clip(round(dx[t]/θ), -2, +2)                 （2bit 幅度 {0,1,2} + 符号）
  输出  : mode="signed" → (B,1,C,T)，值域 {-1,0,1}（2bit 为 {-2..2}）
          mode="posneg" → (B,2,C,T)，正/负两个 0/1 脉冲通道（纯 spike 形态，仅 1bit）

阈值 θ：自适应——按**训练集**逐通道 std 的 k 倍（θ_c = k·std_c），k 可扫（0.5/1.0）。
固定 θ 而非逐 batch 自适应：保证 train/eval 编码一致，且 eval 加噪后 θ 不随噪声漂移。

重构（自检用，标准 delta 调制重构）：
  x̂[0] = x[0]，x̂[t] = x̂[t-1] + p[t]·θ_c  → 重构 SNR = 10·log10(var(x)/MSE)
  重构 SNR 反映「脉冲编码的信息损失」：k 越小（θ 越细）脉冲越多、损失越小。

用法
----
  自检（在 data_4class/train.npz 上统计密度/重构 SNR/正负平衡，验证编码实现正确）：
    env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin \\
        /opt/miniconda3/envs/openmmlab/bin/python delta_encode.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ALGO_DIR = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# 1. 阈值计算：逐通道 θ = k · std（在训练集上一次性算好，训练/评估共用）
# ---------------------------------------------------------------------------
def compute_theta(x: np.ndarray | torch.Tensor, k: float, eps: float = 1e-6) -> np.ndarray:
    """逐通道标准差 → 阈值 θ_c = k·std_c。

    x : (n, 1, C, T) 或 (n, C, T) 的连续 EEG。std 在 (n, T) 维上统计（每通道一个标量）。
    返回 : (C,) ndarray。退化通道（std≈0，如坏电极）给下限 eps，避免 θ=0 导致全发。
    """
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    x = x.squeeze(1) if x.ndim == 4 else x          # (n, C, T)
    std = x.std(axis=(0, 2))                        # 逐通道 std
    theta = k * np.maximum(std, eps)
    return theta.astype(np.float32)


# ---------------------------------------------------------------------------
# 2. delta 调制编码（torch 实现，纯函数；无参数、无梯度需求——输入是数据）
# ---------------------------------------------------------------------------
def delta_encode(
    x: torch.Tensor,
    theta: torch.Tensor | float,
    bits: int = 1,
    mode: str = "signed",
) -> torch.Tensor:
    """连续 EEG → delta 脉冲序列。

    参数
    ----
    x     : (B, 1, C, T) 或 (B, C, T) 连续 EEG（float）
    theta : 阈值。标量（全局）或 (C,) 张量（逐通道 θ_c = k·std_c）
    bits  : 1 = 1bit delta（±1 脉冲）；2 = 2bit delta（{-2,-1,0,1,2}，幅度 2bit）
    mode  : "signed" → (B, 1, C, T) 带符号脉冲（2bit 仅支持此模式）
            "posneg" → (B, 2, C, T) 正/负双 0/1 脉冲通道（仅 1bit）
    返回  : 脉冲张量（float32，0/±1 或 0/±2；posneg 为 0/1 双通道）
    """
    assert mode in ("signed", "posneg"), f"mode={mode!r} 无效"
    if bits == 2:
        assert mode == "signed", "2bit delta 仅支持 signed 模式（值域 {-2..2}）"
    x = x.squeeze(1) if x.ndim == 4 else x          # (B, C, T)
    theta_t = torch.as_tensor(theta, dtype=x.dtype, device=x.device)
    # 广播对齐：dx 为 (B,C,T-1)，逐通道 θ (C,) 需 reshape 成 (1,C,1)；标量 θ 直接广播
    if theta_t.numel() == 1:
        theta_b = theta_t.reshape(1)
    else:
        theta_b = theta_t.reshape(1, -1, 1)
    # 时间差分（因果：p[t] 编码 x[t]-x[t-1]），首点补零 → 输出长度 = T
    dx = x[..., 1:] - x[..., :-1]                   # (B, C, T-1)
    if bits == 1:
        p = torch.zeros_like(x)
        p[..., 1:] = torch.where(dx > theta_b, 1.0,
                     torch.where(dx < -theta_b, -1.0, torch.zeros_like(dx)))
    else:  # 2bit：幅度按 θ 均匀量化（round），clip 到 2bit 幅度 {0,1,2}，带符号
        q = torch.round(dx / theta_b).clamp(-2.0, 2.0)
        p = torch.zeros_like(x)
        p[..., 1:] = q
    if mode == "signed":
        return p.unsqueeze(1)                       # (B, 1, C, T)
    # posneg：正/负双通道 0/1 脉冲（值域仅 0/1，纯 spike 形态）
    pos = (p > 0).float()
    neg = (p < 0).float()
    return torch.stack([pos, neg], dim=1)           # (B, 2, C, T)


# ---------------------------------------------------------------------------
# 3. 重构（自检用）：x̂[t] = x̂[t-1] + p[t]·θ_c，返回重构信号与指标
# ---------------------------------------------------------------------------
def delta_reconstruct(pulses: np.ndarray, theta: np.ndarray, x0: np.ndarray) -> np.ndarray:
    """脉冲序列积分重构（delta 调制标准重构）。

    pulses : (B, C, T) 带符号脉冲（±1/±2 等，乘 θ 为步长）；首点恒 0
    theta  : (C,) 逐通道步长
    x0     : (B, C) 或标量，初始值 x[0]
    返回   : (B, C, T) 重构信号
    """
    steps = pulses * theta[np.newaxis, :, np.newaxis]   # 每脉冲对应 ±θ（或 2θ）步长
    rec = np.cumsum(steps, axis=2)                      # x̂[t] = x̂[0] + Σ p·θ
    rec = rec + x0[..., np.newaxis] if x0.ndim == 2 else rec + x0
    rec[..., 0] = x0                                    # 首点 = 初值（p[0]=0 已保证）
    return rec


# ---------------------------------------------------------------------------
# 4. 可嵌入模型的编码层（无参数；theta 注册为 buffer 随权重保存，保证可复现）
# ---------------------------------------------------------------------------
class DeltaEncode(nn.Module):
    """delta 调制输入编码层（纯前处理，无可训练参数）。

    把连续 EEG (B,1,C,T) 编码为脉冲 (B,1,C,T) signed 或 (B,2,C,T) posneg，
    之后网络全部工作在脉冲域。theta 为固定 buffer（训练集逐通道 k·std），
    随 state_dict 保存（warm-start 时按 missing-key 处理，见 delta_tcnet.py）。
    """

    def __init__(self, theta: torch.Tensor | None = None, k: float = 1.0,
                 bits: int = 1, mode: str = "signed"):
        super().__init__()
        self.k = float(k)
        self.bits = int(bits)
        self.mode = mode
        if theta is not None:
            self.register_buffer("theta", torch.as_tensor(theta, dtype=torch.float32).reshape(-1))
        else:
            self.register_buffer("theta", torch.zeros(1))  # 占位；训练脚本会从训练集算好覆盖
        self.input_density: float = 0.0   # 最近一次 forward 的输入脉冲密度（诊断用）

    def set_theta(self, theta: torch.Tensor | np.ndarray):
        """从训练集统计覆盖 θ（训练脚本调用，保证 train/eval 编码一致）。"""
        self.theta.data = torch.as_tensor(theta, dtype=torch.float32).reshape(-1).to(self.theta.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B,1,C,T) 连续 EEG → 脉冲 (B,1,C,T) signed 或 (B,2,C,T) posneg。"""
        assert x.shape[1] == 1, f"DeltaEncode 输入应为 (B,1,C,T)，实际 {tuple(x.shape)}"
        pulses = delta_encode(x, self.theta, bits=self.bits, mode=self.mode)
        # 输入脉冲密度（稀疏度指标）：非零元素占比（posneg 时按两通道合并口径）
        if self.mode == "posneg":
            self.input_density = float(((pulses.sum(1)) != 0).float().mean().item())
        else:
            self.input_density = float((pulses != 0).float().mean().item())
        return pulses


# ---------------------------------------------------------------------------
# 5. 自检：在 train.npz 上验证编码正确性（密度 / 重构 SNR / 正负平衡）
# ---------------------------------------------------------------------------
def _self_check() -> int:
    npz = ALGO_DIR / "data_4class" / "train.npz"
    if not npz.exists():
        print(f"✗ 数据集缺失：{npz}（先跑 preprocess_4class.py）")
        return 1
    with np.load(npz) as d:
        X = np.asarray(d["X"], dtype=np.float32)   # (n,1,22,500)
    n, _, C, T = X.shape
    print(f"数据 : {npz.name}  {X.shape}  （{n} 试次, {C} 通道, {T} 时间点）")
    print(f"全局幅度: mean={X.mean():.4f} std={X.std():.4f} min={X.min():.3f} max={X.max():.3f}")

    print("\n===== 逐通道 θ = k·std 与脉冲统计（train 集全量）=====")
    std_c = X.std(axis=(0, 2))
    print(f"逐通道 std: min={std_c.min():.4f} median={np.median(std_c):.4f} "
          f"max={std_c.max():.4f}")

    # 批次采样（全量 2592 试次统计即可，内存可控：2592×22×500×4B ≈ 114MB）
    rows = []
    for k in (0.5, 1.0):
        theta = compute_theta(X, k)
        for bits in (1, 2):
            pulses = delta_encode(torch.from_numpy(X), torch.from_numpy(theta),
                                  bits=bits, mode="signed").numpy()  # (n,1,C,T)
            p = pulses[:, 0]                                        # (n,C,T)
            density = float((p != 0).mean())
            pos_frac = float((p > 0).mean()) / max(density, 1e-9)
            # 重构 SNR（delta 调制标准重构）
            rec = delta_reconstruct(p, theta, X[:, 0, :, 0])
            mse = float(((rec - X[:, 0, :, :]) ** 2).mean())
            var = float(X[:, 0, :, :].var())
            snr = 10.0 * np.log10(var / mse) if mse > 0 else float("inf")
            corr = float(np.corrcoef(rec.ravel(), X[:, 0, :, :].ravel())[0, 1])
            rows.append((k, bits, density, snr, corr, pos_frac))
            print(f"  k={k:<4} bits={bits}bit : 脉冲密度={density*100:5.2f}%  "
                  f"重构SNR={snr:6.2f}dB  相关系数={corr:.4f}  +脉冲占比={pos_frac*100:5.1f}%")
    # posneg 模式密度（双通道合并口径）
    k = 0.5
    theta = compute_theta(X, k)
    pn = delta_encode(torch.from_numpy(X), torch.from_numpy(theta), bits=1, mode="posneg").numpy()
    dens_pn = float(((pn.sum(1)) != 0).mean())
    print(f"  k={k} posneg 双通道 : 合并脉冲密度={dens_pn*100:5.2f}%  "
          f"（signed 同参数应一致，验证实现等价）")

    # ---- 单元校验（编码语义正确性，与统计无关）----
    unit_ok = True
    # ① 常数信号 → 零脉冲
    xc = torch.ones(2, 1, 3, 50)
    pc = delta_encode(xc, torch.tensor([0.5, 0.5, 0.5]), bits=1, mode="signed")
    if pc.abs().sum().item() != 0:
        print("✗ 常数信号应零脉冲"); unit_ok = False
    # ② 单点冲激 → 上升沿 +1、下降沿 -1（delta 编码对阶跃的正确响应）
    xs = torch.zeros(1, 1, 1, 10); xs[0, 0, 0, 3] = 1.0   # dx@t3 = +1.0 > θ=0.4；dx@t4 = -1.0
    ps = delta_encode(xs, torch.tensor([0.4]), bits=1, mode="signed")
    if ps[0, 0, 0, 3] != 1.0 or ps[0, 0, 0, 4] != -1.0:
        print(f"✗ 阶跃脉冲位置/符号错误：{ps[0,0,0].tolist()}"); unit_ok = False
    # ③ posneg 与 signed 等价：pos−neg == signed
    pn3 = delta_encode(xs, torch.tensor([0.4]), bits=1, mode="posneg")
    if not torch.allclose(pn3[:, 0] - pn3[:, 1], ps[:, 0]):
        print("✗ posneg 与 signed 不等价"); unit_ok = False
    # ④ 2bit：dx=3θ → clip 到 +2
    p4 = delta_encode(xs, torch.tensor([0.2]), bits=2, mode="signed")  # dx=1.0 = 5θ → +2
    if p4[0, 0, 0, 3] != 2.0:
        print(f"✗ 2bit clip 错误：{p4[0,0,0,3]}"); unit_ok = False
    if unit_ok:
        print("单元校验 : 常数零脉冲 / 阶跃+1 / posneg≡signed / 2bit clip 全部通过 ✓")

    # ---- 统计断言（稀疏度目标 + 单调性）----
    # rows = [(k=0.5,1bit), (k=0.5,2bit), (k=1.0,1bit), (k=1.0,2bit)]
    d05_1, d05_2 = rows[0][2], rows[1][2]
    d10_1, d10_2 = rows[2][2], rows[3][2]
    ok = unit_ok
    if not (d10_1 < d05_1 and d10_2 < d05_2):
        print("✗ k 越大密度应越低（阈值越大脉冲越少）")
        ok = False
    if not (d05_2 > d05_1 and d10_2 > d10_1):
        print("✗ 同 k 下 2bit 密度应 ≥ 1bit（幅度细分产生更多事件）")
        ok = False
    if not (d05_1 < 0.30 and d10_1 < 0.30):
        print(f"✗ 1bit 输入密度应 <30%（稀疏目标；k=0.5={d05_1*100:.1f}%, k=1.0={d10_1*100:.1f}%）")
        ok = False
    # 相关性（信息保留量）：同 k 下 2bit ≥ 1bit（幅度信息更多 → 重构更贴合）
    if not (rows[1][4] >= rows[0][4] and rows[3][4] >= rows[2][4]):
        print("✗ 同 k 下 2bit 重构相关系数应 ≥ 1bit")
        ok = False
    if abs(dens_pn - d05_1) > 1e-3:
        print(f"✗ posneg 与 signed 密度不一致：{dens_pn:.4f} vs {d05_1:.4f}")
        ok = False
    # 正负平衡：EEG 差分近似对称，+占比应在 40%~60% 之间
    for _, _, _, _, _, pf in rows:
        if not (0.4 < pf < 0.6):
            print(f"✗ 正负脉冲不平衡：+占比 {pf*100:.1f}%（异常，编码可能出错）")
            ok = False
    print("\n" + ("✓ 自检通过：delta 编码实现正确" if ok else "✗ 自检未通过，请检查"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(_self_check())
