#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ann2snn.py — ANN-to-SNN 转换脚本骨架（BN 折叠 + LIF 替换 + 阈值标定）

背景（对齐 01_LIF-EEGNet演进路线.md §2.1 / §3 S3）：
  训练好 ReLU-EEGNet 代理后，本脚本把它转换成脉冲神经网络（SNN），
  使「约束驱动鲁棒解码」的成果可平滑迁移到神经形态硬件 / 片上 SNN 推理。

转换三步（本骨架实现全部三步，且各自可独立运行）：
  1) BN 折叠：推理时 BatchNorm 是逐通道仿射 y = γ·(x-μ)/√(σ²+ε) + β，
     可折叠进前一层卷积权重/偏置（SNN 里不能留 BN）。
     复用 luckfox-deploy/model/export_c_weights.py 的折叠公式。
  2) LIF 替换：ReLU 激活位点 → LIF 脉冲神经元（integrate-and-fire，可带泄漏）。
     脉冲本身是 0/1 二值，天然规避 RV1106 NPU INT8 量化噪声（类脑换轨解法）。
  3) 阈值标定（threshold balancing）：在标定样本上记录每层激活前的最大正值，
     作为该层 LIF 的发放阈值，使「最大激活 ≈ 满发放率」，尽量复现 ANN 的激活幅值。

  转换后的 SNN 采用**恒定电流输入**（rate-based）：输入 x 在 timesteps 个时间步内
  作为恒定电流反复馈入首层，各层 LIF 积分发放，输出层取 timesteps 步的均值作为
  分类得分（线性读出，不脉冲化）。

⚠️ 诚实边界：本脚本是「可运行的骨架」，只验证转换流程在 dummy 输入上跑通、
  折叠数学正确、形状正确；**不报告任何 SNN 精度**（无训练权重 + 无真实数据），
  精度评估属 S3 后续工作，标注「未验证」。

用法
----
  骨架自检（dummy 输入，验证折叠正确性 + 转换流程跑通）：
    env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin \\
        /opt/miniconda3/envs/openmmlab/bin/python ann2snn.py
"""

from __future__ import annotations

import sys
from typing import Iterable, Optional

import torch
import torch.nn as nn

# 复用 lif_eegnet.py 的 LIF-EEGNet 作为转换源 ANN（ReLU 模式）
from lif_eegnet import LIFEEGNet


# ---------------------------------------------------------------------------
# 1. BN 折叠
# ---------------------------------------------------------------------------
def fold_bn_layer(
    conv: nn.Conv2d,
    bn: nn.BatchNorm2d,
    eps: float = 1e-5,
) -> nn.Conv2d:
    """把 BatchNorm2d 折叠进前一层 Conv2d，返回新的 Conv2d（bias=True）。

    公式（逐输出通道仿射）：
        scale = γ / √(σ² + ε)
        W'    = W · scale                      （逐输出通道缩放卷积核）
        b'    = b·scale + (β - μ·scale)        （原 bias 缩放后再加 BN 平移）
    conv1/depthwise/pointwise 在 EEGNet 中均为 bias=False，故 b=0；为通用性仍处理 b。
    """
    w = conv.weight.data
    b = conv.bias.data if conv.bias is not None else \
        torch.zeros(conv.out_channels, device=conv.weight.device)
    scale = bn.weight.data / torch.sqrt(bn.running_var + eps)
    shift = bn.bias.data - bn.running_mean * scale

    w_folded = w * scale.view(-1, 1, 1, 1)
    b_folded = b * scale + shift

    new_conv = nn.Conv2d(
        conv.in_channels, conv.out_channels, conv.kernel_size,
        stride=conv.stride, padding=conv.padding,
        groups=conv.groups, bias=True,
    )
    new_conv.weight.data = w_folded
    new_conv.bias.data = b_folded
    return new_conv


def clone_conv(conv: nn.Conv2d) -> nn.Conv2d:
    """复制一个 Conv2d（保持 bias 与否一致）。用于无 BN 的 sep_depthwise 与 fc。"""
    new_conv = nn.Conv2d(
        conv.in_channels, conv.out_channels, conv.kernel_size,
        stride=conv.stride, padding=conv.padding,
        groups=conv.groups, bias=(conv.bias is not None),
    )
    new_conv.weight.data = conv.weight.data.clone()
    if conv.bias is not None:
        new_conv.bias.data = conv.bias.data.clone()
    return new_conv


# ---------------------------------------------------------------------------
# 2. LIF 脉冲神经元（推理用，无 surrogate，纯阶跃发放）
# ---------------------------------------------------------------------------
def lif_step(
    current: torch.Tensor,
    mem: Optional[torch.Tensor],
    threshold: float,
    decay: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """单时间步 LIF（integrate-and-fire，可带泄漏）。

    膜电位更新：mem = decay·mem + current
    发放：       spk = H(mem - threshold)
    硬重置：     发放后 mem 清零（mem = mem·(1-spk)）

    返回 (spk, mem')。decay=1.0 退化为无泄漏 IF（rate-exact，转换首选）；
    decay<1 为 LIF（生物合理，但引入发放率误差）。
    """
    if mem is None:
        mem = torch.zeros_like(current)
    mem = decay * mem + current
    spk = (mem >= threshold).to(current.dtype)
    mem = mem * (1.0 - spk)   # 硬重置
    return spk, mem


# ---------------------------------------------------------------------------
# 3. SNN 版 EEGNet（折叠后 + LIF 激活）
# ---------------------------------------------------------------------------
class SNNEEGNet(nn.Module):
    """转换后的脉冲 EEGNet。结构与 ANN 一致，但：BN 已折叠进卷积、ReLU 换 LIF。

    阈值 thr1/thr2 在标定后写入；forward 内部跑 timesteps 个时间步，
    返回输出层（线性读出）的时间均值作为分类得分。
    """

    def __init__(self, ann: LIFEEGNet):
        super().__init__()
        # ---- 折叠 BN 进卷积（SNN 不留 BN）----
        self.conv1 = fold_bn_layer(ann.conv1, ann.bn1)
        self.depthwise = fold_bn_layer(ann.depthwise, ann.bn2)
        self.sep_depthwise = clone_conv(ann.sep_depthwise)   # 无 BN，原样复制
        self.pointwise = fold_bn_layer(ann.pointwise, ann.bn3)
        self.fc = clone_conv(ann.fc)                          # 线性读出

        # ---- 结构参数与池化（AvgPool 在脉冲上 = 滑窗脉冲计数）----
        self.pool1 = nn.AvgPool2d((1, 4))
        self.pool2 = nn.AvgPool2d((1, 8))
        self.flatten = nn.Flatten()
        self.pad1_amt = ann.pad1_amt
        self.pad2_amt = ann.pad2_amt
        self.F2 = ann.F2
        self.n_classes = ann.n_classes
        self.temporal_out = ann.temporal_out

        # ---- 阈值（标定后由 calibrate_thresholds 写入）----
        self.thr1 = 1.0
        self.thr2 = 1.0

    def forward(
        self,
        x: torch.Tensor,
        timesteps: int = 16,
        decay: float = 1.0,
    ) -> torch.Tensor:
        """x: (B,1,8,500) 恒定电流输入 → 输出 (B, n_classes) 时间均值分类得分。

        返回 (out_sum/timesteps, s1_rate, s2_rate) 由 snn_infer 封装，
        这里为便于记录各层发放率，直接返回均值分类得分。
        """
        mem1 = mem2 = None
        out_sum = torch.zeros(x.shape[0], self.n_classes, device=x.device)
        for _ in range(timesteps):
            # Block 1（手动补零 + 折叠卷积 + LIF）
            B, C, H, T = x.shape
            pad1 = x.new_zeros(B, C, H, self.pad1_amt)
            xp = torch.cat([pad1, x, pad1], dim=3)
            a = self.depthwise(self.conv1(xp))      # (B, D*F1, 1, T)
            s1, mem1 = lif_step(a, mem1, self.thr1, decay)
            p1 = self.pool1(s1)                     # (B, D*F1, 1, T//4)

            # Block 2
            B, C, H, T = p1.shape
            pad2 = p1.new_zeros(B, C, H, self.pad2_amt)
            p1p = torch.cat([pad2, p1, pad2], dim=3)
            b = self.pointwise(self.sep_depthwise(p1p))   # (B, F2, 1, T//4)
            s2, mem2 = lif_step(b, mem2, self.thr2, decay)
            p2 = self.pool2(s2)                     # (B, F2, 1, T//32)

            # 输出层（线性读出，不脉冲化）
            flat = self.flatten(p2)                 # (B, F2*T//32)
            out = self.fc(flat.view(B, self.F2 * self.temporal_out, 1, 1))
            out_sum = out_sum + out.view(B, self.n_classes)
        return out_sum / timesteps


# ---------------------------------------------------------------------------
# 4. 转换主流程：构建 SNN + 阈值标定
# ---------------------------------------------------------------------------
def build_snn(ann: LIFEEGNet) -> SNNEEGNet:
    """从 ReLU-EEGNet 构建 SNN（BN 折叠 + LIF 替换）。阈值默认 1.0，待标定。"""
    assert ann.activation in ("relu", "elu"), \
        "转换源应为 ReLU-EEGNet（activation='relu'），lif 模式无需转换"
    return SNNEEGNet(ann)


def calibrate_thresholds(
    ann: LIFEEGNet,
    samples: Iterable[torch.Tensor],
    eps: float = 1e-3,
) -> tuple[float, float]:
    """逐层阈值标定（threshold balancing）。

    在标定样本上记录每层激活前的**最大正值**，作为该层 LIF 的发放阈值，
    使「最大激活 ≈ 满发放率」，尽量复现 ANN 激活幅值。

    返回 (thr1, thr2)。注意：只取正值（ReLU 负半轴本就不发放）。
    """
    max_act: dict[str, float] = {"act1": 0.0, "act2": 0.0}

    def make_hook(key: str):
        def hook(_m, _inp, out: torch.Tensor) -> None:
            v = float(out.detach().clamp(min=0).max())
            max_act[key] = max(max_act[key], v)
        return hook

    h1 = ann.bn2.register_forward_hook(make_hook("act1"))   # act1 激活前
    h2 = ann.bn3.register_forward_hook(make_hook("act2"))   # act2 激活前
    ann.eval()
    with torch.no_grad():
        for s in samples:
            ann(s)
    h1.remove()
    h2.remove()

    thr1 = max(max_act["act1"], eps)
    thr2 = max(max_act["act2"], eps)
    return thr1, thr2


def convert_and_calibrate(
    ann: LIFEEGNet,
    samples: Iterable[torch.Tensor],
) -> SNNEEGNet:
    """一步到位：构建 SNN 并完成阈值标定。"""
    snn = build_snn(ann)
    thr1, thr2 = calibrate_thresholds(ann, samples)
    snn.thr1 = thr1
    snn.thr2 = thr2
    return snn


def snn_infer(snn: SNNEEGNet, x: torch.Tensor, timesteps: int = 16) -> torch.Tensor:
    """SNN 推理：返回 (B, n_classes) 时间均值分类得分。"""
    snn.eval()
    with torch.no_grad():
        return snn(x, timesteps=timesteps)


# ---------------------------------------------------------------------------
# 5. 骨架自检
# ---------------------------------------------------------------------------
def _verify_fold(ann: LIFEEGNet, snn: SNNEEGNet) -> None:
    """验证 BN 折叠数学正确：折叠后卷积 == 原 BN∘Conv（eval/running-stats 下）。"""
    torch.manual_seed(1)
    x = torch.randn(2, 1, 8, 500)
    ann.eval()
    with torch.no_grad():
        # Block1 折叠验证（含手动补零）
        B, C, H, T = x.shape
        pad1 = x.new_zeros(B, C, H, ann.pad1_amt)
        xp = torch.cat([pad1, x, pad1], dim=3)
        ref1 = ann.bn1(ann.conv1(xp))
        got1 = snn.conv1(xp)
        ref2 = ann.bn2(ann.depthwise(ref1))
        got2 = snn.depthwise(got1)

        # Block2 折叠验证（pointwise + bn3）
        # 用 act1 之后的 pool 结果作为输入，简化：直接对随机张量验证 pointwise 折叠
        y = torch.randn(2, ann.D * ann.F1, 1, 125)
        ref3 = ann.bn3(ann.pointwise(y))
        got3 = snn.pointwise(y)

    for name, r, g in (("conv1+bn1", ref1, got1),
                       ("depthwise+bn2", ref2, got2),
                       ("pointwise+bn3", ref3, got3)):
        d = float((r - g).abs().max())
        assert torch.allclose(r, g, atol=1e-5), f"{name} 折叠不一致，max_diff={d:.3e}"
        print(f"  [折叠验证] {name}: max_diff={d:.3e}  ✓")


def _selfcheck() -> int:
    torch.manual_seed(0)
    print("=" * 72)
    print("ANN-to-SNN 转换骨架自检（dummy 输入）")
    print("=" * 72)

    # ---- 5.1 随机 ReLU-EEGNet 作为转换源 ----
    ann = LIFEEGNet(activation="relu")
    ann.eval()
    print(f"[转换源] ReLU-EEGNet 参数量={ann.num_params():,}")

    # ---- 5.2 BN 折叠验证 ----
    snn = build_snn(ann)
    print("[1/3] BN 折叠验证：")
    _verify_fold(ann, snn)

    # ---- 5.3 阈值标定 ----
    calib = [torch.randn(1, 1, 8, 500) for _ in range(8)]
    thr1, thr2 = calibrate_thresholds(ann, calib)
    snn.thr1, snn.thr2 = thr1, thr2
    print(f"[2/3] 阈值标定：thr1={thr1:.4f}, thr2={thr2:.4f}")

    # ---- 5.4 SNN 推理 ----
    x = torch.randn(1, 1, 8, 500)
    out = snn_infer(snn, x, timesteps=16)
    print(f"[3/3] SNN 推理：输入 {tuple(x.shape)} → 输出 {tuple(out.shape)}")
    print(f"      输出分类得分={out.detach().cpu().numpy().ravel()}")
    assert tuple(out.shape) == (1, 2), "SNN 输出形状错误"

    # ---- 5.5 各层发放率统计（透明度，非精度）----
    with torch.no_grad():
        mem1 = mem2 = None
        s1_sum = s2_sum = 0.0
        for _ in range(16):
            B, C, H, T = x.shape
            pad1 = x.new_zeros(B, C, H, snn.pad1_amt)
            xp = torch.cat([pad1, x, pad1], dim=3)
            a = snn.depthwise(snn.conv1(xp))
            s1, mem1 = lif_step(a, mem1, snn.thr1, 1.0)
            p1 = snn.pool1(s1)
            B, C, H, T = p1.shape
            pad2 = p1.new_zeros(B, C, H, snn.pad2_amt)
            p1p = torch.cat([pad2, p1, pad2], dim=3)
            b = snn.pointwise(snn.sep_depthwise(p1p))
            s2, mem2 = lif_step(b, mem2, snn.thr2, 1.0)
            s1_sum += float(s1.mean()); s2_sum += float(s2.mean())
    print(f"      act1 平均发放率={s1_sum/16:.4f}, act2 平均发放率={s2_sum/16:.4f}")
    print()
    print("✓ 转换骨架自检通过（折叠数学正确、阈值标定完成、SNN 推理跑通）")
    print("  ⚠️ SNN 精度未验证（本骨架无训练权重/真实数据，属 S3 后续工作）")
    return 0


if __name__ == "__main__":
    sys.exit(_selfcheck())
