# -*- coding: utf-8 -*-
"""
finger_cnn.py —— 轻量全卷积 CNN：32ch sEMG → 8 指分类（类 EEGNet 结构）

复现对象：MyoText (arXiv:2601.03098) 第一级「手指分类」，但架构不同。
论文用 CNN-BiLSTM-Attention（85.45%），这里为「类脑化」改用**轻量全卷积 CNN**
（无循环、无注意力），结构对齐 EEGNet（Lawhern et al. 2018）：

  Block1: Conv2d(1,F1,(1,63)) → BN → DepthwiseConv2d(F1,D*F1,(32,1),groups=F1)
          → BN → 激活 → AvgPool(1,4) → Dropout
  Block2: SeparableConv2d(D*F1,F2,(1,15)) [depthwise + pointwise]
          → BN → 激活 → AvgPool(1,8) → Dropout
  分类头: Flatten → Conv2d(1x1) → 8

  - 时间卷积 (1,63) 抓跨时间局部特征，depthwise (32,1) 把 32 个 sEMG 通道压成 1，
    与 EEGNet 把 22 个 EEG 电极压成 1 同理。
  - 全卷积、无全连接（用 Conv2d(1x1) 代替 Linear），利于端侧量化 / 转 ONNX→RKNN。
  - 激活层走可配置工厂：elu / relu / lif（lif 复用 algorithm/lif_eegnet.py 的
    LIFActivation，surrogate gradient 可导，输出发放率 ∈ [0,1]）。

参数：n_channels=32、n_classes=8、input_length=4000（2s @ 2kHz）均可配置；
      F1=8, D=2, F2=16, kernel_length=63 对齐 EEGNet 默认（可改）。
参数量约 1.7 万（含 1x1 分类头 16k），远轻于 MyoText 的 BiLSTM-Attention。

用法
----
  前向自检（不依赖数据）：
    env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin \\
        /opt/miniconda3/envs/openmmlab/bin/python finger_cnn.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

# ---- 复用 algorithm/lif_eegnet.py 的 LIFActivation（surrogate gradient）----
_ALGO_DIR = Path(__file__).resolve().parent.parent
if str(_ALGO_DIR) not in sys.path:
    sys.path.insert(0, str(_ALGO_DIR))
try:
    from lif_eegnet import LIFActivation  # 只读引用，不改动原代码
    _HAS_LIF = True
except Exception as _e:  # noqa: BLE001  （若 import 失败则 lif 模式不可用）
    LIFActivation = None
    _HAS_LIF = False
    _LIF_IMPORT_ERR = _e


def _make_activation(activation: str, **lif_kwargs) -> nn.Module:
    """激活工厂：elu / relu / lif（lif 复用 lif_eegnet.LIFActivation）。"""
    activation = (activation or "elu").lower()
    if activation == "elu":
        return nn.ELU()
    if activation == "relu":
        return nn.ReLU()
    if activation == "lif":
        if not _HAS_LIF:
            raise RuntimeError(
                f"lif 模式不可用：无法 import lif_eegnet.LIFActivation（{_LIF_IMPORT_ERR}）"
            )
        return LIFActivation(**lif_kwargs)
    raise ValueError(f"不支持的 activation={activation!r}，可选：'elu' / 'relu' / 'lif'")


class FingerCNN(nn.Module):
    """轻量全卷积 CNN：32ch sEMG → 8 指分类（类 EEGNet，无循环无注意力）。"""

    def __init__(
        self,
        n_channels: int = 32,
        n_classes: int = 8,
        input_length: int = 4000,
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
        # 奇数核 + forward 手动补零（valid conv），与 eegnet.py 一致：
        # 规避 RV1106 NPU 不支持 Pad 算子的问题（同项目约束）。
        self.conv1 = nn.Conv2d(1, F1, (1, kernel_length), padding=0, bias=False)
        self.pad1_amt = kernel_length // 2
        self.bn1 = nn.BatchNorm2d(F1)
        self.depthwise = nn.Conv2d(
            F1, D * F1, (n_channels, 1), groups=F1, padding=0, bias=False
        )
        self.bn2 = nn.BatchNorm2d(D * F1)
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
        self.act2 = _make_activation(activation, **lif_kwargs)
        self.pool2 = nn.AvgPool2d((1, 8))
        self.dropout2 = nn.Dropout(dropout)

        # ---- 分类头：T → T//4 → T//32，用 Conv2d(1x1) 代替 Linear ----
        self.temporal_out = input_length // 32
        self.flatten = nn.Flatten()
        self.fc = nn.Conv2d(F2 * self.temporal_out, n_classes, (1, 1))

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, C, T) → (B, F2 * T//32) 特征（flatten 后、fc 前）。"""
        B, _, H, T = x.shape
        # Block 1
        pad1 = x.new_zeros(B, 1, H, self.pad1_amt)
        x = torch.cat([pad1, x, pad1], dim=3)       # (B,1,C,T+2*pad1)
        x = self.conv1(x)                            # (B,F1,C,T)
        x = self.bn1(x)
        x = self.depthwise(x)                        # (B,D*F1,1,T)
        x = self.bn2(x)
        x = self.act1(x)
        x = self.pool1(x)                            # (B,D*F1,1,T//4)
        x = self.dropout1(x)
        # Block 2
        B, _, H, T = x.shape
        pad2 = x.new_zeros(B, _, H, self.pad2_amt)
        x = torch.cat([pad2, x, pad2], dim=3)        # (B,C,H,T//4+2*pad2)
        x = self.sep_depthwise(x)                    # (B,D*F1,1,T//4)
        x = self.pointwise(x)                        # (B,F2,1,T//4)
        x = self.bn3(x)
        x = self.act2(x)
        x = self.pool2(x)                            # (B,F2,1,T//32)
        x = self.dropout2(x)
        x = self.flatten(x)                          # (B,F2*T//32)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, C, T) → (B, n_classes) logits（或 lif 发放率输出）。"""
        x = self.forward_features(x)
        x = x.view(x.shape[0], self.F2 * self.temporal_out, 1, 1)
        x = self.fc(x)                               # (B,n_classes,1,1)
        x = x.view(x.shape[0], self.n_classes)
        return x

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def _selfcheck() -> int:
    torch.manual_seed(0)
    dummy = torch.randn(2, 1, 32, 4000, dtype=torch.float32)
    print("=" * 72)
    print("FingerCNN 前向自检  input=(2,1,32,4000) → 期望输出 (2,8)")
    print("=" * 72)

    for act in ("elu", "relu"):
        m = FingerCNN(n_channels=32, n_classes=8, input_length=4000, activation=act)
        m.eval()
        with torch.no_grad():
            out = m(dummy)
        print(f"[{act:>4}] 输出 shape={tuple(out.shape)}  参数量={m.num_params():,}")
        assert tuple(out.shape) == (2, 8), f"{act} 输出形状错误 {tuple(out.shape)}"

    if _HAS_LIF:
        m = FingerCNN(n_channels=32, n_classes=8, input_length=4000, activation="lif",
                      threshold=1.0, tau=2.0, timesteps=8)
        m.eval()
        with torch.no_grad():
            out = m(dummy)
        print(f"[lif ] 输出 shape={tuple(out.shape)}  参数量={m.num_params():,}")
        assert tuple(out.shape) == (2, 8)
        # 反向传播自检（surrogate gradient 可导）
        m.train()
        out = m(dummy)
        out.sum().backward()
        n_grad = sum(1 for p in m.parameters()
                     if p.requires_grad and p.grad is not None and p.grad.abs().sum() > 0)
        n_all = sum(1 for p in m.parameters() if p.requires_grad)
        print(f"[lif ] 反向传播自检：{n_grad}/{n_all} 参数有非零梯度（surrogate 生效）")
        assert n_grad == n_all, "lif 模式部分参数梯度为零"
    else:
        print("⚠️  lif_eegnet.LIFActivation 未导入成功，跳过 lif 模式自检。")

    print("✓ 全部自检通过")
    return 0


if __name__ == "__main__":
    sys.exit(_selfcheck())
