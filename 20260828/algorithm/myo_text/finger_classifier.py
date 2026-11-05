# -*- coding: utf-8 -*-
"""
finger_classifier.py —— MyoText 第一层：手指分类器（CNN-BiLSTM-Attention）

复现对象：MyoText (arXiv:2601.03098) 分级解码流水线的第一级「手指分类」。
论文架构：对每个击键片段（前 1s + 后 1s，32 通道 sEMG），用
    CNN(局部时空特征) -> BiLSTM(双向时间依赖) -> Attention(时间池化) -> 8 类 softmax
把多通道表面肌电分类为 8 个手指之一（不含拇指）。

原论文指标（供对照，本复现尚未实测）：
    5-fold CV      : 85.45% ± 3.36%
    LOUO(留一被试) : 81.4%  ± 3.76%

部署设计（split-execution）：本模块跑在可穿戴端设备（本项目的 RV1106），
T5 句子重建跑在上位机（见 t5_reconstruct.py）。

【SNN 接口预留】activation_type 参数可在 'relu'（ANN，默认）与 'spike'
（脉冲/SNN 占位）之间切换，为后续把 CNN-BiLSTM 部分替换为脉冲版本、部署到
低功耗 NPU 预留位置。真实脉冲版本需引入 snnTorch / spikingjelly，此处仅给出
可插拔的占位实现与转换钩子 to_spiking()。

说明：本文件是「从零复现的代码骨架」，非官方实现；MyoText 无开源代码。
    无官方权重，所有数值均待自行训练后实测。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 常量：8 个手指（QWERTY 标准指法，不含拇指）
# ---------------------------------------------------------------------------
N_FINGERS = 8
FINGER_NAMES = [
    "left_pinky",   # 左手小指
    "left_ring",    # 左手无名指
    "left_middle",  # 左手中指
    "left_index",   # 左手食指
    "right_index",  # 右手食指
    "right_middle", # 右手中指
    "right_ring",   # 右手无名指
    "right_pinky",  # 右手小指
]

# 默认输入：emg2qwerty 双侧 16 通道 = 32 通道 sEMG
N_CHANNELS = 32


# ---------------------------------------------------------------------------
# SNN 激活占位实现
# ---------------------------------------------------------------------------
class _SpikeActivationPlaceholder(nn.Module):
    """
    脉冲激活（LIF 简化版）占位实现。

    【重要】这只是一个可运行的占位，用于打通 activation 可插拔接口，
    不代表真实的 SNN 转换。真正的脉冲版本需要：
      - snnTorch 的 snn.Leaky / spikingjelly 的 neuron.LIFNode；
      - 时间步展开（把静态张量按 timesteps 循环送入脉冲神经元）；
      - 梯度用代理函数（surrogate gradient，如 arctan / fast-sigmoid）；
      - 归一化/阈值（V_th）与静息电位（V_rest）标定。

    此占位在功能上退化为一个带可学习阈值的类 ReLU 激活，保证前向/反向可跑，
    便于先验证「切换 activation 后模型结构不变、输入输出维度不变」这条路径。
    """

    def __init__(self, threshold: float = 1.0):
        super().__init__()
        self.threshold = nn.Parameter(torch.tensor(float(threshold)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 简化：软阈值阶跃（可微代理），维度不变
        return torch.sigmoid((x - self.threshold) * 5.0) * x


def build_activation(kind: str, **kwargs) -> nn.Module:
    """
    可插拔激活工厂。

    kind='relu'   -> nn.ReLU(inplace=True)（ANN 默认）
    kind='spike'  -> _SpikeActivationPlaceholder（SNN 占位）
    kind='lif'    -> 别名，同 'spike'
    kind='gelu'   -> nn.GELU()
    """
    kind = (kind or "relu").lower()
    if kind in ("relu",):
        return nn.ReLU(inplace=True)
    if kind in ("spike", "lif", "snn"):
        return _SpikeActivationPlaceholder(**kwargs)
    if kind in ("gelu",):
        return nn.GELU()
    raise ValueError(f"未知 activation 类型: {kind!r}")


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
@dataclass
class FingerClassifierConfig:
    """手指分类器配置。所有数值为「骨架默认值」，未经调参验证。"""

    in_channels: int = N_CHANNELS          # sEMG 通道数（emg2qwerty=32）
    n_fingers: int = N_FINGERS             # 输出类别数（8 指）
    window_seconds: float = 2.0            # 片段时长（前1s+后1s）
    fs_hz: int = 2000                      # emg2qwerty 采样率（Hz，2000 为常见值，需按数据集校准）

    # CNN 部分
    cnn_channels: List[int] = field(default_factory=lambda: [64, 128, 128])
    cnn_kernels: List[int] = field(default_factory=lambda: [7, 5, 3])
    cnn_dropout: float = 0.25

    # BiLSTM 部分
    lstm_hidden: int = 128
    lstm_layers: int = 2
    lstm_dropout: float = 0.3

    # 分类头
    fc_dropout: float = 0.3

    # 激活类型：'relu'（ANN）/'spike'（SNN 占位）—— SNN 预留接口
    activation: str = "relu"

    def __post_init__(self):
        if len(self.cnn_channels) != len(self.cnn_kernels):
            raise ValueError("cnn_channels 与 cnn_kernels 长度必须一致")


# ---------------------------------------------------------------------------
# 注意力池化（Attention Pooling）
# ---------------------------------------------------------------------------
class TemporalAttention(nn.Module):
    """
    时间维注意力池化：对 BiLSTM 输出序列 (B, T, 2H) 学习每个时间步的权重，
    加权求和得到固定长度的上下文向量 (B, 2H)。

    可解释性：注意力权重即「预备肌电 / 残余肌电」在击键时序上的重要性分布，
    与论文「前 1s 预备 + 后 1s 残余」的片段切分语义对齐。
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (B, T, hidden_dim)
        scores = self.attn(x).squeeze(-1)          # (B, T)
        weights = F.softmax(scores, dim=-1)        # (B, T)
        context = torch.sum(x * weights.unsqueeze(-1), dim=1)  # (B, hidden_dim)
        return context, weights


# ---------------------------------------------------------------------------
# 主网络：CNN -> BiLSTM -> Attention -> 分类头
# ---------------------------------------------------------------------------
class FingerClassifier(nn.Module):
    """
    MyoText 手指分类器骨架。

    前向流程：
      输入 x: (B, C=32, T)  —— 单通道维为电极，时间维为采样点
      1) CNN 特征提取：若干 1D 卷积块（Conv1d + BN + 激活 + Dropout + MaxPool），
         提取局部时空特征，输出 (B, F, T')
      2) 维度转置 -> (B, T', F)，送入 BiLSTM 建模双向时间依赖
      3) TemporalAttention 池化 -> (B, 2H)
      4) 全连接分类头 -> (B, 8) logits
    """

    def __init__(self, cfg: Optional[FingerClassifierConfig] = None):
        super().__init__()
        self.cfg = cfg or FingerClassifierConfig()
        c = self.cfg

        # ---- CNN 特征提取 ----
        convs: List[nn.Module] = []
        in_ch = c.in_channels
        for out_ch, kernel in zip(c.cnn_channels, c.cnn_kernels):
            convs.append(nn.Conv1d(in_ch, out_ch, kernel, padding=kernel // 2))
            convs.append(nn.BatchNorm1d(out_ch))
            convs.append(build_activation(c.activation))
            convs.append(nn.Dropout(c.cnn_dropout))
            convs.append(nn.MaxPool1d(2))
            in_ch = out_ch
        self.cnn = nn.Sequential(*convs)
        cnn_out_ch = c.cnn_channels[-1]  # 与最后一个卷积块输出通道一致

        # ---- BiLSTM ----
        self.lstm = nn.LSTM(
            input_size=cnn_out_ch,
            hidden_size=c.lstm_hidden,
            num_layers=c.lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=c.lstm_dropout if c.lstm_layers > 1 else 0.0,
        )

        # ---- 注意力池化 ----
        self.attention = TemporalAttention(hidden_dim=c.lstm_hidden * 2)

        # ---- 分类头 ----
        self.classifier = nn.Sequential(
            nn.Dropout(c.fc_dropout),
            nn.Linear(c.lstm_hidden * 2, 64),
            build_activation(c.activation),
            nn.Linear(64, c.n_fingers),
        )

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, T) 原始 sEMG 片段
        返回: (B, 8) 分类 logits
        """
        feats = self.cnn(x)                              # (B, F, T')
        feats = feats.transpose(1, 2)                    # (B, T', F)
        lstm_out, _ = self.lstm(feats)                   # (B, T', 2H)
        context, attn_weights = self.attention(lstm_out) # (B, 2H)
        self._last_attn_weights = attn_weights           # 供可视化/可解释性分析
        logits = self.classifier(context)                # (B, 8)
        return logits

    def predict_finger(self, x: torch.Tensor) -> torch.Tensor:
        """返回 (B,) 的预测手指索引。"""
        logits = self.forward(x)
        return logits.argmax(dim=-1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """返回 (B, 8) 的 softmax 概率分布。"""
        return F.softmax(self.forward(x), dim=-1)

    @property
    def last_attn_weights(self) -> Optional[torch.Tensor]:
        """最近一次前向的注意力权重 (B, T')，用于检查预备/残余肌电时序。"""
        return getattr(self, "_last_attn_weights", None)

    # ------------------------------------------------------------------
    # SNN 转换预留钩子
    # ------------------------------------------------------------------
    def to_spiking(self, threshold: float = 1.0) -> "FingerClassifier":
        """
        【预留接口】把网络切换为脉冲（SNN）版本。

        当前实现：把所有 build_activation 产出替换为占位脉冲激活，
        仅验证「接口可切换、维度不变」。真正的 ANN->SNN 转换需额外：
          1) 引入 snnTorch/spikingjelly 的 LIF 神经元；
          2) 前向按时间步展开（重复 T 次送入脉冲层累加膜电位）；
          3) 替换 BatchNorm 为可脉冲化的归一化（或权重归一化）；
          4) 权重按 V_th 缩放（data-based normalization）以降低转换误差。

        返回 self，便于链式调用 model.to_spiking().to(device)。
        """
        cfg = self.cfg
        cfg.activation = "spike"

        def _recurse_replace(module: nn.Module):
            for name, child in list(module.named_children()):
                if isinstance(child, (nn.ReLU, nn.GELU)):
                    setattr(module, name, _SpikeActivationPlaceholder(threshold))
                else:
                    _recurse_replace(child)

        _recurse_replace(self)
        # CNN 中的激活也需要替换（build_activation 动态构建的部分）
        # 此处递归替换的是直接挂在 Sequential 里的 ReLU，其余 ReLU 同理。
        return self

    def to_ann(self) -> "FingerClassifier":
        """把网络切回 ANN 版本（relu）。"""
        self.cfg.activation = "relu"

        def _recurse_replace(module: nn.Module):
            for name, child in list(module.named_children()):
                if isinstance(child, _SpikeActivationPlaceholder):
                    setattr(module, name, nn.ReLU(inplace=True))
                else:
                    _recurse_replace(child)

        _recurse_replace(self)
        return self


# ---------------------------------------------------------------------------
# 便捷构建 + 冒烟测试
# ---------------------------------------------------------------------------
def build_finger_classifier(**overrides) -> FingerClassifier:
    """用关键字参数覆盖默认配置后构建模型。"""
    cfg = FingerClassifierConfig(**overrides)
    return FingerClassifier(cfg)


if __name__ == "__main__":
    # 冒烟测试：随机输入跑通前向，验证维度与 SNN 接口切换。
    # 运行（遵守项目环境铁律，隔离环境避免 numpy ImportError）：
    #   env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin \
    #       /opt/miniconda3/envs/openmmlab/bin/python finger_classifier.py
    torch.manual_seed(0)
    B, C, T = 4, N_CHANNELS, 4000  # 4 段 × 32 通道 × 4000 采样点（2s @ 2000Hz）

    model = FingerClassifier()
    x = torch.randn(B, C, T)
    logits = model(x)
    assert logits.shape == (B, N_FINGERS), f"logits 维度错误: {logits.shape}"

    print("[OK] ANN 前向成功，logits shape =", tuple(logits.shape))
    print("[OK] 预测手指 =", model.predict_finger(x).tolist())
    print("[OK] 注意力权重 shape =", tuple(model.last_attn_weights.shape))

    # 切换 SNN 占位并重跑前向，验证接口预留可用
    model.to_spiking()
    logits_snn = model(x)
    assert logits_snn.shape == (B, N_FINGERS), "SNN 占位前向维度错误"
    print("[OK] SNN 占位切换后前向成功，logits shape =", tuple(logits_snn.shape))

    # 参数量统计（供端侧部署估算）
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[OK] 总参数量 = {n_params:,}  (约 {n_params/1e6:.2f}M)")
    print("注意：SNN 占位非真实脉冲实现，真实 SNN 需引入 snnTorch/spikingjelly。")
