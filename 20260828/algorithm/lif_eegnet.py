#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lif_eegnet.py — LIF-EEGNet：EEGNet 激活层抽象为可配置接口（elu / relu / lif）

设计目标（对齐 01_LIF-EEGNet演进路线.md §4 代码接口约束）：
  - 激活层不写死 ELU，统一走 `_make_activation(activation)` 工厂，返回 nn.Module。
  - activation="elu" / "relu"：结构与 ~/BCI/luckfox-deploy/model/eegnet.py 完全一致，
    输出、参数量、state_dict 键全部对齐（可直接 load 现有权重，零回归）。
  - activation="lif"：用可导 LIF 层（surrogate gradient），forward 返回**发放率聚合**
    （训练友好），作为「备选：surrogate gradient 直接训练 SNN」的落点。

LIF 层语义（发放率版 / rate-based）：
  - 把连续激活值当作「恒定输入电流」，在 `timesteps` 个时间步内做泄漏积分发放。
  - 膜电位更新：mem = decay * mem + x （decay = exp(-dt/tau)，dt=1）
  - 发放：spk = H(mem - threshold) （H 为阶跃，反向用 sigmoid 近似梯度，可导）
  - 硬重置：发放后 mem 清零 mem = mem * (1 - spk)
  - 输出：spk 在时间维求平均 = 发放率 ∈ [0, 1]

  说明：负半轴输入（mem 恒为负）永不发放 → 发放率=0，等效 ReLU 的「负半轴截断」。
  这与演进路线 §2.1 的「负半轴损失」警告一致，是 surrogate 训练路径的已知代价；
  缓解手段（双通道 signed 编码等）留待 S4。

用法
----
  前向自检（打印三种激活模式的前向 shape / 参数量，并与参考 eegnet.py 对齐）：
    env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin \\
        /opt/miniconda3/envs/openmmlab/bin/python lif_eegnet.py
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# 1. Surrogate Gradient：阶跃函数的可导近似
# ---------------------------------------------------------------------------
class SurrogateStep(torch.autograd.Function):
    """可导阶跃函数（surrogate gradient）。

    forward  : H(x) = (x > 0)   —— 真正的发放与否（不可导阶跃）
    backward : 用 sigmoid 近似的导数替代：d/dx sigmoid(αx) = α·σ(αx)·(1-σ(αx))
               参数 α 控制梯度宽度（α 越大越接近理想阶跃，梯度越集中）。
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float):
        ctx.save_for_backward(x)
        ctx.alpha = alpha
        return (x > 0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (x,) = ctx.saved_tensors
        alpha = ctx.alpha
        s = torch.sigmoid(alpha * x)
        grad = grad_output * (alpha * s * (1.0 - s))
        return grad, None


# ---------------------------------------------------------------------------
# 2. 可导 LIF 激活层
# ---------------------------------------------------------------------------
class LIFActivation(nn.Module):
    """可导 LIF 发放率激活层（训练友好，输出发放率 ∈ [0,1]）。

    输入任意形状 (B, ...) 的连续张量，逐元素做 LIF 积分发放，
    输出同形状的发放率张量（spk 在时间维求平均）。

    参数
    ----
    threshold : 发放阈值（固定标量，不参与训练）。ann2snn.py 的阈值标定
                可随时改写 model.act1.threshold 等属性。
    tau       : 膜时间常数（越大泄漏越慢，decay = exp(-1/tau) 越接近 1）。
    timesteps : 内部仿真时间步数（越大发放率估计越精细，计算量线性增加）。
    alpha     : surrogate sigmoid 的斜率（梯度宽度），仅影响反向。
    """

    def __init__(
        self,
        threshold: float = 1.0,
        tau: float = 2.0,
        timesteps: int = 8,
        alpha: float = 4.0,
    ):
        super().__init__()
        self.threshold = float(threshold)   # 发放阈值（标定接口，可外部改写）
        self.tau = float(tau)
        self.timesteps = int(timesteps)
        self.alpha = float(alpha)
        self.decay = math.exp(-1.0 / self.tau)  # dt=1 的泄漏系数

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, ...) 恒定输入电流 → 输出同形状发放率 ∈ [0,1]。"""
        mem = x.new_zeros(x.shape)  # 膜电位初值 0
        thr = self.threshold
        spike_sum = x.new_zeros(x.shape)
        for _ in range(self.timesteps):
            mem = self.decay * mem + x              # 泄漏积分（x 恒定电流）
            spk = SurrogateStep.apply(mem - thr, self.alpha)  # 可导发放
            mem = mem * (1.0 - spk)                 # 硬重置
            spike_sum = spike_sum + spk
        out = spike_sum / self.timesteps            # 发放率聚合 [0,1]
        self.last_firing_rate = float(out.mean())   # 记录平均发放率（诊断用，纯附加）
        return out

    def extra_repr(self) -> str:
        return (f"threshold={self.threshold}, tau={self.tau}, "
                f"timesteps={self.timesteps}, alpha={self.alpha}, "
                f"decay={self.decay:.4f}")


# ---------------------------------------------------------------------------
# 2b. 可导 Signed LIF 激活层（正负脉冲分离，补回负半轴）
# ---------------------------------------------------------------------------
class SignedLIFActivation(nn.Module):
    """可导 signed LIF 发放率激活层（正负脉冲分离，输出 ∈ [-1,1]）。

    动机
    ----
    普通 LIFActivation 只发正脉冲：负半轴输入（mem 恒为负）永不发放，发放率恒 0，
    等效 ReLU 的「负半轴截断」，丢弃了 ELU 保留的负半轴信息。二分类 8ch 下该信息
    不重要（零代价），四分类 22ch 下类别多、特征要求高，代价显现（约 -1.4pp）。

    做法（双通道 signed 编码）
    -------------------------
    正通道：mem_p = decay*mem_p + relu(x)   → spk_p = H(mem_p - thr) → 硬重置
    负通道：mem_n = decay*mem_n + relu(-x)  → spk_n = H(mem_n - thr) → 硬重置
    输出   = (sum(spk_p) - sum(spk_n)) / timesteps  ∈ [-1, 1]

    稳态下这是 relu(x) - relu(-x) = x 的脉冲近似，保留了 x 的正负符号信息。
    两个通道都走 SurrogateStep（sigmoid 近似梯度），反向均可导。

    参数（与 LIFActivation 一致，两通道共享同一 thr/tau）
    ----------------------------------------------------
    threshold : 发放阈值（固定标量，不参与训练）。
    tau       : 膜时间常数（decay = exp(-1/tau)）。
    timesteps : 内部仿真时间步数。
    alpha     : surrogate sigmoid 斜率（仅影响反向）。
    """

    def __init__(
        self,
        threshold: float = 1.0,
        tau: float = 2.0,
        timesteps: int = 8,
        alpha: float = 4.0,
    ):
        super().__init__()
        self.threshold = float(threshold)
        self.tau = float(tau)
        self.timesteps = int(timesteps)
        self.alpha = float(alpha)
        self.decay = math.exp(-1.0 / self.tau)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, ...) 恒定输入电流 → 输出同形状净发放率 ∈ [-1,1]。

        正通道吃 relu(x) 发正脉冲，负通道吃 relu(-x) 发负脉冲，净发放率相减。
        """
        thr = self.threshold
        # 正/负通道的恒定输入电流（把 x 的正负半轴拆成两路非负信号）
        xp = torch.relu(x)
        xn = torch.relu(-x)
        mem_p = x.new_zeros(x.shape)          # 正通道膜电位
        mem_n = x.new_zeros(x.shape)          # 负通道膜电位
        spike_sum_p = x.new_zeros(x.shape)    # 正脉冲累计
        spike_sum_n = x.new_zeros(x.shape)    # 负脉冲累计
        for _ in range(self.timesteps):
            # ---- 正通道 ----
            mem_p = self.decay * mem_p + xp
            spk_p = SurrogateStep.apply(mem_p - thr, self.alpha)  # 可导发放
            mem_p = mem_p * (1.0 - spk_p)                         # 硬重置
            spike_sum_p = spike_sum_p + spk_p
            # ---- 负通道 ----
            mem_n = self.decay * mem_n + xn
            spk_n = SurrogateStep.apply(mem_n - thr, self.alpha)
            mem_n = mem_n * (1.0 - spk_n)
            spike_sum_n = spike_sum_n + spk_n
        pos_rate = spike_sum_p / self.timesteps   # 正发放率 ∈ [0,1]
        neg_rate = spike_sum_n / self.timesteps   # 负发放率 ∈ [0,1]
        out = pos_rate - neg_rate                 # 净发放率 ∈ [-1,1]
        # 诊断量（纯附加，不参与计算图，用于 report）
        self.last_firing_rate = float(out.mean())       # 净发放率（可正可负）
        self.last_pos_rate = float(pos_rate.mean())     # 正通道发放率
        self.last_neg_rate = float(neg_rate.mean())     # 负通道发放率
        return out

    def extra_repr(self) -> str:
        return (f"threshold={self.threshold}, tau={self.tau}, "
                f"timesteps={self.timesteps}, alpha={self.alpha}, "
                f"decay={self.decay:.4f}")


# ---------------------------------------------------------------------------
# 3. 激活层工厂：统一可配置接口
# ---------------------------------------------------------------------------
def _make_activation(activation: str, **lif_kwargs) -> nn.Module:
    """按名字构造激活层。elu / relu 无参数；lif 返回可导 LIF 层；
    signed_lif 返回可导 signed LIF 层（正负脉冲分离）。"""
    if activation == "elu":
        return nn.ELU()
    if activation == "relu":
        return nn.ReLU()
    if activation == "lif":
        return LIFActivation(**lif_kwargs)
    if activation == "signed_lif":
        return SignedLIFActivation(**lif_kwargs)
    raise ValueError(
        f"不支持的 activation={activation!r}，可选：'elu' / 'relu' / 'lif' / 'signed_lif'"
    )


# ---------------------------------------------------------------------------
# 4. LIF-EEGNet 主模型（结构 100% 对齐 eegnet.py，仅激活层走可配置接口）
# ---------------------------------------------------------------------------
class LIFEEGNet(nn.Module):
    """EEGNet（8ch 二分类运动想象）的可配置激活版。

    结构、命名、手动补零方式均与 luckfox-deploy/model/eegnet.py 的 EEGNet 完全一致，
    区别仅在于：两个激活位点分别用 self.act1 / self.act2（由 _make_activation 构造），
    为后续 SNN 替换预留每层独立阈值标定的空间。
    """

    def __init__(
        self,
        n_channels: int = 8,
        n_classes: int = 2,
        input_length: int = 500,
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
        kernel_length: int = 63,
        dropout: float = 0.5,
        activation: str = "elu",
        **lif_kwargs,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.input_length = input_length
        self.F1, self.D, self.F2 = F1, D, F2
        self.kernel_length = kernel_length
        self.activation = activation

        # ---- Block 1：时间卷积 + 空间 depthwise 卷积 ----
        # 与 eegnet.py 一致：奇数核 + forward 手动补零（valid conv），
        # 规避 RV1106 NPU 不支持 Pad 算子的问题。
        self.conv1 = nn.Conv2d(1, F1, (1, kernel_length), padding=0, bias=False)
        self.pad1_amt = kernel_length // 2
        self.bn1 = nn.BatchNorm2d(F1)
        self.depthwise = nn.Conv2d(
            F1, D * F1, (n_channels, 1), groups=F1, padding=0, bias=False
        )
        self.bn2 = nn.BatchNorm2d(D * F1)
        # 激活层（可配置）：elu / relu / lif
        self.act1 = _make_activation(activation, **lif_kwargs)
        self.pool1 = nn.AvgPool2d((1, 4))
        self.dropout1 = nn.Dropout(dropout)

        # ---- Block 2：separable 卷积（depthwise (1,15) + pointwise (1,1)）----
        self.sep_depthwise = nn.Conv2d(
            D * F1, D * F1, (1, 15), groups=D * F1, padding=0, bias=False
        )
        self.pad2_amt = 7
        self.pointwise = nn.Conv2d(D * F1, F2, (1, 1), padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(F2)
        # 激活层（可配置）
        self.act2 = _make_activation(activation, **lif_kwargs)
        self.pool2 = nn.AvgPool2d((1, 8))
        self.dropout2 = nn.Dropout(dropout)

        # ---- 分类头：T → T//4 → T//32 ----
        # 用 Conv2d(1x1) 代替 Linear（对齐 eegnet.py，规避 rknn Gemm 量化 bug）。
        self.temporal_out = input_length // 32
        self.flatten = nn.Flatten()
        self.fc = nn.Conv2d(F2 * self.temporal_out, n_classes, (1, 1))

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, C, T) → 输出 (B, F2*T//32) 特征（flatten 后、fc 之前）。

        供 NPU 只算到特征层使用；与 eegnet.py.forward_features 逐行等价
        （仅把 self.elu 换成 self.act1 / self.act2）。
        """
        # Block 1
        B, C, H, T = x.shape
        pad1 = x.new_zeros(B, C, H, self.pad1_amt)
        x = torch.cat([pad1, x, pad1], dim=3)   # (B, C, H, T + 2*pad1)
        x = self.conv1(x)          # valid conv → (B, F1, C, T)
        x = self.bn1(x)
        x = self.depthwise(x)      # (B, D*F1, 1, T)
        x = self.bn2(x)
        x = self.act1(x)
        x = self.pool1(x)          # (B, D*F1, 1, T//4)
        x = self.dropout1(x)

        # Block 2
        B, C, H, T = x.shape
        pad2 = x.new_zeros(B, C, H, self.pad2_amt)
        x = torch.cat([pad2, x, pad2], dim=3)   # (B, C, H, T//4 + 2*pad2)
        x = self.sep_depthwise(x)  # valid conv → (B, D*F1, 1, T//4)
        x = self.pointwise(x)      # (B, F2, 1, T//4)
        x = self.bn3(x)
        x = self.act2(x)
        x = self.pool2(x)          # (B, F2, 1, T//32)
        x = self.dropout2(x)
        x = self.flatten(x)        # (B, F2 * T//32)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, C, T) → 输出 (B, n_classes) 未归一化 logits（或发放率）。"""
        x = self.forward_features(x)
        x = x.view(x.shape[0], self.F2 * self.temporal_out, 1, 1)
        x = self.fc(x)             # (B, n_classes, 1, 1)
        x = x.view(x.shape[0], self.n_classes)
        return x

    def num_params(self) -> int:
        """可训练参数量。"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# 5. 前向自检
# ---------------------------------------------------------------------------
def _selfcheck() -> int:
    torch.manual_seed(0)
    dummy = torch.randn(1, 1, 8, 500, dtype=torch.float32)
    print("=" * 72)
    print("LIF-EEGNet 前向自检  input=(1,1,8,500) → 期望输出 (1,2)")
    print("=" * 72)

    # ---- 5.1 各激活模式前向 shape + 参数量 ----
    for act in ("elu", "relu", "lif", "signed_lif"):
        m = LIFEEGNet(activation=act)
        m.eval()
        with torch.no_grad():
            out = m(dummy)
        print(f"[{act:>10}] 输出 shape={tuple(out.shape)}  参数量={m.num_params():,}")
        assert tuple(out.shape) == (1, 2), f"{act} 输出形状错误"
    print()

    # ---- 5.1b LIF 层本身输出应落在 [0,1]（发放率语义）----
    lif_layer = LIFActivation()
    with torch.no_grad():
        r = lif_layer(torch.randn(4, 8, 1, 64))
    assert float(r.min()) >= 0.0 and float(r.max()) <= 1.0, \
        f"LIF 层发放率应 ∈ [0,1]，实际 [{float(r.min()):.3f},{float(r.max()):.3f}]"
    print(f"[lif] LIF 层发放率范围检查：∈ [{float(r.min()):.3f}, {float(r.max()):.3f}]  ✓")
    print()

    # ---- 5.2 与参考 eegnet.py 对齐（elu / relu 模式必须零回归）----
    ref_dir = Path(
        os.environ.get("EEGNET_REF_DIR", str(Path.home() / "BCI/luckfox-deploy/model"))
    )
    aligned = False
    if ref_dir.exists():
        sys.path.insert(0, str(ref_dir))
        try:
            from eegnet import EEGNet as RefEEGNet  # 只读引用，不改动原代码
        except Exception as exc:  # noqa: BLE001
            print(f"⚠️  无法导入参考 eegnet.py：{exc}")
            RefEEGNet = None
        if RefEEGNet is not None:
            for act in ("elu", "relu"):
                ref = RefEEGNet(activation=act)
                mine = LIFEEGNet(activation=act)
                # state_dict 键必须完全一致（激活层 elu/relu 无参数，键可对齐）
                missing, unexpected = mine.load_state_dict(
                    ref.state_dict(), strict=True
                )
                assert not missing and not unexpected, "state_dict 键不一致"
                ref.eval(); mine.eval()
                with torch.no_grad():
                    ref_out = ref(dummy)
                    my_out = mine(dummy)
                max_diff = float((ref_out - my_out).abs().max())
                assert torch.allclose(ref_out, my_out, atol=1e-6), \
                    f"{act} 模式输出与参考 eegnet.py 不一致，max_diff={max_diff}"
                print(f"[对齐] {act} 模式：参数量 ref={ref.num_params():,} "
                      f"mine={mine.num_params():,}，输出 max_diff={max_diff:.3e}  ✓")
            aligned = True
    if not aligned:
        print("⚠️  未找到参考 eegnet.py，跳过对齐检查（不影响 shape/参数自检）。")
    print()

    # ---- 5.3 lif 模式反向传播自检（surrogate gradient 可导性）----
    m = LIFEEGNet(activation="lif")
    m.train()
    out = m(dummy)
    loss = out.sum()
    loss.backward()
    grads = [p.grad is not None and p.grad.abs().sum().item() > 0
             for p in m.parameters() if p.requires_grad]
    n_grad = sum(grads)
    print(f"[lif] 反向传播自检：{n_grad}/{len(grads)} 个参数有非零梯度 "
          f"（surrogate gradient 生效）")
    assert n_grad > 0, "lif 模式反向传播失败：无参数收到梯度"
    assert n_grad == len(grads), "lif 模式部分参数梯度为零，检查网络连通性"
    print()

    # ---- 5.4 signed_lif 前向自检（正负脉冲分离）----
    sl = SignedLIFActivation()
    # (a) 混合输入：输出 ∈ [-1,1]，且正、负脉冲都能发得出
    x_mix = torch.randn(4, 8, 1, 64)
    with torch.no_grad():
        r_mix = sl(x_mix)
    assert float(r_mix.min()) >= -1.0 and float(r_mix.max()) <= 1.0, \
        f"signed LIF 输出应 ∈ [-1,1]，实际 [{float(r_mix.min()):.3f},{float(r_mix.max()):.3f}]"
    assert sl.last_pos_rate > 0.0 and sl.last_neg_rate > 0.0, \
        "混合输入下正、负通道都应发放"
    print(f"[signed] 混合输入：输出 ∈ [{float(r_mix.min()):.3f}, {float(r_mix.max()):.3f}]，"
          f"正发放率={sl.last_pos_rate:.3f} 负发放率={sl.last_neg_rate:.3f}  ✓")
    # (b) 纯正输入：只发正脉冲，输出 ≥ 0
    x_pos = torch.rand(4, 8, 1, 64) * 2.0
    with torch.no_grad():
        r_pos = sl(x_pos)
    assert float(r_pos.min()) >= 0.0, f"纯正输入输出应 ≥ 0，实际 min={float(r_pos.min()):.3f}"
    assert sl.last_pos_rate > 0.0 and sl.last_neg_rate == 0.0, "纯正输入负通道不应发放"
    print(f"[signed] 纯正输入：输出 min={float(r_pos.min()):.3f}（≥0），"
          f"正发放率={sl.last_pos_rate:.3f} 负发放率={sl.last_neg_rate:.3f}  ✓")
    # (c) 纯负输入：只发负脉冲，输出 ≤ 0（普通 LIF 永远输出 0 的点）
    x_neg = -torch.rand(4, 8, 1, 64) * 2.0
    with torch.no_grad():
        r_neg = sl(x_neg)
    assert float(r_neg.max()) <= 0.0, f"纯负输入输出应 ≤ 0，实际 max={float(r_neg.max()):.3f}"
    assert sl.last_neg_rate > 0.0 and sl.last_pos_rate == 0.0, "纯负输入正通道不应发放"
    print(f"[signed] 纯负输入：输出 max={float(r_neg.max()):.3f}（≤0），"
          f"正发放率={sl.last_pos_rate:.3f} 负发放率={sl.last_neg_rate:.3f}  ✓")
    print()

    # ---- 5.5 signed_lif 模式反向传播自检（双通道都可导）----
    m = LIFEEGNet(activation="signed_lif")
    m.train()
    out = m(dummy)
    assert tuple(out.shape) == (1, 2), "signed_lif 输出形状错误"
    loss = out.sum()
    loss.backward()
    grads = [p.grad is not None and p.grad.abs().sum().item() > 0
             for p in m.parameters() if p.requires_grad]
    n_grad = sum(grads)
    print(f"[signed] 反向传播自检：{n_grad}/{len(grads)} 个参数有非零梯度 "
          f"（正负通道 surrogate gradient 生效）")
    assert n_grad == len(grads), "signed_lif 模式部分参数梯度为零，检查网络连通性"
    print()

    print("✓ 全部自检通过（含 elu/relu 与参考 eegnet.py 的对齐，以及 signed_lif 正负脉冲）")
    return 0


if __name__ == "__main__":
    sys.exit(_selfcheck())
