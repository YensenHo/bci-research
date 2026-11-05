#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""multitask_model.py — 通道无关（channel-agnostic）骨干 + 任务专属头

统一接口：所有骨干 `forward(x) -> (B, F)`（2D 特征，F = C*H_per_ch），头 = TaskHead(F → ncls)。
这样 v1/v2 骨干可互换，downstream/pretrain 用 `--backbone v1|v2` 切换。

设计核心（对应「多范式系统整合」的「per-channel 共享 temporal 滤波、空间层下游学」）：
  - TemporalBackbone (v1)：纯时间卷积。reshape (B,C,T)→(B*C,1,T) 过共享 Conv1d（权重只依赖
    H/kernel/n_layers，与通道数 C 无关），再全局时间池化 → (B,C*H)。
  - TemporalBackboneV2 (v2)：多尺度时间卷积（短核抓 beta/gamma、长核抓 delta/alpha）+ 通道自注意力
    空间层（表达 ERD 的「C3/C4 相对功率」侧化）。两者都保持通道无关：多尺度卷积靠 reshape B*C 共享、
    自注意力是 permutation-equivariant + 变长（通道数各异天然支持），不加 positional encoding
    （通道布局 2D、无序列顺序）。
  - TaskHead：任务专属 MLP（Linear(C*H→hidden)→Linear(hidden→ncls)），每个任务自己的通道数/类别数。

注意：不用 BatchNorm（num_features 依赖 C，破坏跨任务共享）；数据已逐试次 z-score，靠 GeLU 稳定。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalBackbone(nn.Module):
    """v1 通道无关纯时间编码器。输入 (B,C,T)，输出 (B, C*H)。"""

    def __init__(self, H: int = 8, kernel: int = 16, n_layers: int = 2, act: str = "gelu"):
        super().__init__()
        self.H = H
        convs = []
        for i in range(n_layers):
            in_ch = 1 if i == 0 else H
            # padding="same" 保证输出长度精确 = T（偶核 kernel16 下 padding=kernel//2 会 +1/层）
            convs.append(nn.Conv1d(in_ch, H, kernel, padding="same"))
        self.convs = nn.ModuleList(convs)
        self._act = F.gelu if act == "gelu" else F.relu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T = x.shape
        x = x.reshape(B * C, 1, T)
        for conv in self.convs:
            x = self._act(conv(x))
        x = x.reshape(B, C * self.H, x.shape[-1])
        return x.mean(dim=-1)  # (B, C*H) 全局时间池化


class TemporalBackboneV2(nn.Module):
    """v2 多尺度时间卷积 + 通道自注意力空间层，通道无关。输入 (B,C,T)，输出 (B, C*H)。"""

    def __init__(self, H_per_scale: int = 8, kernels=(8, 32, 128), n_layers: int = 2,
                 n_heads: int = 4, dim_ff=None, act: str = "gelu", dropout: float = 0.1):
        super().__init__()
        self.H = H_per_scale * len(kernels)
        self.kernels = tuple(kernels)
        # 多尺度时间分支（每尺度独立 Conv 栈，通道无关 reshape B*C）
        self.branches = nn.ModuleList()
        for k in kernels:
            layers = []
            for i in range(n_layers):
                in_ch = 1 if i == 0 else H_per_scale
                layers.append(nn.Conv1d(in_ch, H_per_scale, k, padding="same"))
            self.branches.append(nn.ModuleList(layers))
        # 通道自注意力空间层（cross-channel attention，permutation-equivariant + 变长）
        assert self.H % n_heads == 0, f"H={self.H} 不能被 n_heads={n_heads} 整除"
        dim_ff = dim_ff or (self.H * 2)
        self.attn = nn.TransformerEncoderLayer(
            d_model=self.H, nhead=n_heads, batch_first=True,
            dim_feedforward=dim_ff, dropout=dropout)
        self._act = F.gelu if act == "gelu" else F.relu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T = x.shape
        outs = []
        for layers in self.branches:
            y = x.reshape(B * C, 1, T)
            for conv in layers:
                y = self._act(conv(y))
            # (B*C, H_per_scale, T) -> (B, C, H_per_scale, T)
            outs.append(y.reshape(B, C, self.H // len(self.kernels), y.shape[-1]))
        y = torch.cat(outs, dim=2)      # (B, C, H, T)
        y = y.mean(dim=-1)              # (B, C, H) 时间池化
        y = self.attn(y)                # (B, C, H) 通道自注意力
        return y.reshape(B, C * self.H)  # (B, C*H)


class TaskHead(nn.Module):
    """任务专属头：空间混合（隐式）+ 分类。输入 (B, C*H)，输出 (B, n_classes)。"""

    def __init__(self, in_dim: int, n_classes: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_backbone(name: str, **kw) -> tuple[nn.Module, int]:
    """按名构建骨干，返回 (backbone, H_per_ch)。"""
    if name == "v1":
        bb = TemporalBackbone(H=kw.get("H", 8), kernel=kw.get("kernel", 16),
                              n_layers=kw.get("n_layers", 2))
    elif name == "v2":
        bb = TemporalBackboneV2(H_per_scale=kw.get("H_per_scale", 8),
                                kernels=kw.get("kernels", (8, 32, 128)),
                                n_layers=kw.get("n_layers", 2),
                                n_heads=kw.get("n_heads", 4),
                                dim_ff=kw.get("dim_ff"))
    else:
        raise ValueError(f"未知骨干: {name}")
    return bb, bb.H


class MultiTaskModel(nn.Module):
    """共享骨干 + 每任务一个头。task_specs: dict[task] = (n_channels, n_classes)。"""

    def __init__(self, backbone: nn.Module, H: int, task_specs: dict):
        super().__init__()
        self.backbone = backbone
        self.heads = nn.ModuleDict({
            t: TaskHead(C * H, nc) for t, (C, nc) in task_specs.items()
        })
        self.H = H

    def forward(self, task: str, x: torch.Tensor) -> torch.Tensor:
        return self.heads[task](self.backbone(x))  # (B, C*H) -> (B, n_classes)
