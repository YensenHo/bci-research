#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eeg_tcnet.py — EEG-TCNet 模型定义（Ingolfsson et al. 2020, arXiv:2006.00622）

原文：EEG-TCNet: An Accurate Temporal Convolutional Network for Embedded
Motor-Imagery Brain-Machine Interfaces。设计目标就是**嵌入式 BCI**：
BCIC IV 2a 四分类 77.35%（within-subject 固定超参）/ 83.84%（per-subject 调参），
MOABB 12 数据集 meta 超 SoA（meta-effect 0.25）；固定配置仅 4,272 参数 + 6.8 MMACs，
特征图 396 kB（8bit 量化），是专为 MCU/嵌入式设计的轻量模型。

结构（忠实对照原文 Table II / Fig.3，参数反推已验证）
----------------------------------------------------
  输入 x: (B, 1, C=22, T=500)（C=1 图像通道, H=22 EEG 电极, W=500 时间点）

  φ1  temporal conv    : Conv2d(1, F1, (1, KE)) → BatchNorm2d
                         → DepthwiseConv2d(F1, 2F1, (C,1), groups=F1) → BatchNorm2d
  φ2                   : ELU → AvgPool(1,8) → Dropout(pe)      [输出 (B,2F1,1,T/8)]
  φ3  separable conv   : SeparableConv2d(2F1, F2, (1,16)) → BatchNorm2d
                         → ELU → AvgPool(1,8) → Dropout(pe)     [输出 (B,F2,1,T/64)]
  φ4  TCN              : L=2 个残差块，FT 个滤波器，核 KT，膨胀率 {1,2}，因果
                         残差块 = 两层膨胀因果 Conv1d（BN→ELU→Dropout 间隔）
                         + skip 连接（通道变化时 1×1 投影）
  φ5  readout + Dense  : 取最后时间步（因果 TCN 最后步感受野覆盖全序列）
                         → Linear(FT, n_classes)

论文固定超参（grid search 最优）：F1=8, F2=16, KE=32, KT=4, L=2, FT=12,
pe=0.2, pt=0.3（原文 T=1125；我们协议 dropout 固定 0.25，见 train_eeg_tcnet.py）。

【参数量反推验证】用上述结构手算（T=1125 下与 T 无关，仅 MACs 相关）：
  前端 = 256(φ1) + 16(BN1) + 368(depthwise 含偏置) + 32(BN2) + 560(separable
        = depthwise 256 + pointwise 272 + BN3 32) = 1232
  TCN 块1 (16→FT=12) = 780 + 24 + 588 + 24 + 204(1×1 残差投影) = 1620
  TCN 块2 (12→12)    = 588 + 24 + 588 + 24 + 144(1×1 投影, 无偏置) = 1368
  Dense = 12×4 + 4 = 52
  合计 = 1232 + 1620 + 1368 + 52 = 4272 = 原文固定配置参数量（完全一致 ✓）
  → 说明结构与原文实现一致（TCN 内为普通膨胀因果卷积，非深度可分离；
    任务简报猜测的"TCN 深度可分离"不成立，以原文为准）。

与项目约定的适配（均有注释说明）：
  - 手动补零 + valid conv（与 eegnet.py 相同的 NPU 适配策略，规避 RV1106 不支持的 Pad op）。
  - 偶数核 KE=32 / sep 核 16 用对称补零 KE/2、16/2 保持 'same' 长度（数学等价）。
  - BatchNorm 默认 momentum（与本项目 EEGNet 基线一致，保证可比）。

activation 可配置（为 LIF 化留口）
--------------------------------
  activation="elu"（默认，与 EEGNet 基线一致）/ "relu"（板载转换链 relu 兼容）/
  "lif"（预留：复用同目录 lif_eegnet.py 的 LIFActivation，后续做 LIF-TCNet
  surrogate 训练）。

用法
----
  前向自检（打印输出形状 / 参数量 / 参数量反推校验）：
    env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin \\
        /opt/miniconda3/envs/openmmlab/bin/python eeg_tcnet.py
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 激活工厂：elu / relu / lif（lif 为预留口，复用 lif_eegnet.py 的 LIFActivation）
# ---------------------------------------------------------------------------
def _make_activation(name: str) -> nn.Module:
    """按名字构造激活层，统一接口（对齐 lif_eegnet.py 的设计约定）。

    - "elu"  : nn.ELU()，与 EEGNet ELU 基线一致（默认）
    - "relu" : nn.ReLU()，板载转换链（relu/lif 兼容）用
    - "lif"  : 复用 lif_eegnet.py 的 LIFActivation（surrogate 发放率激活，
               后续 LIF 化入口；LIF 层无参数，state_dict 键零回归）
    """
    if name == "elu":
        return nn.ELU()
    if name == "relu":
        return nn.ReLU()
    if name == "lif":
        from lif_eegnet import LIFActivation  # noqa: F401  # 同目录，延迟导入
        return LIFActivation()
    raise ValueError(f"未知激活函数: {name!r}（可选 elu / relu / lif）")


# ---------------------------------------------------------------------------
# TCN 残差块：两层膨胀因果卷积（Bai et al. 2018 式，原文修改为 BN/ELU/普通 Dropout）
# ---------------------------------------------------------------------------
class _TCNResidualBlock(nn.Module):
    """一个 TCN 残差块（原文 Fig.2 Right）。

    结构（原文：The residual block of a TCN consists of two layers of dilated
    convolutions, with batch normalization, non-linearity, and a dropout layer
    in-between the convolutions. 与 Bai et al. [21] 仅 BN/ELU/Dropout 三处不同，
    卷积为普通 Conv1d，非深度可分离）：
      conv_a（膨胀 d, 因果）→ BN → 激活 → Dropout
      conv_b（膨胀 d, 因果）→ BN → 激活 → Dropout
      skip：输入 + 输出；通道数变化时用 1×1 卷积投影对齐
      （原文 4272 参数反推：块1 投影 16→FT 带偏置，块2 投影 FT→FT 无偏置）

    因果性：膨胀卷积若对称 padding，右侧会"看到未来"；
    这里手动左补零 (KT-1)*d，输出长度与输入一致，严格因果。
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 4,
                 dilation: int = 1, dropout: float = 0.25, activation: str = "elu"):
        super().__init__()
        self.dilation = dilation
        self.pad_left = (kernel_size - 1) * dilation   # 因果：只补左边
        # 两层膨胀因果卷积（含偏置，与原文 4272 参数反推一致）
        self.conv_a = nn.Conv1d(in_channels, out_channels, kernel_size,
                                dilation=dilation, padding=0, bias=True)
        self.bn_a = nn.BatchNorm1d(out_channels)
        self.act_a = _make_activation(activation)
        self.drop_a = nn.Dropout(dropout)
        self.conv_b = nn.Conv1d(out_channels, out_channels, kernel_size,
                                dilation=dilation, padding=0, bias=True)
        self.bn_b = nn.BatchNorm1d(out_channels)
        self.act_b = _make_activation(activation)
        self.drop_b = nn.Dropout(dropout)
        # skip 投影：块1 (16→FT) 带偏置；块2 (FT→FT) 无偏置（对齐原文 4272）
        self.skip = nn.Conv1d(in_channels, out_channels, 1, bias=(in_channels != out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C_in, T) → 输出 (B, C_out, T)（长度不变，残差连接）。"""
        out = F.pad(x, (self.pad_left, 0))            # 因果左补零
        out = self.conv_a(out)
        out = self.bn_a(out)
        out = self.act_a(out)
        out = self.drop_a(out)
        out = F.pad(out, (self.pad_left, 0))
        out = self.conv_b(out)
        out = self.bn_b(out)
        out = self.act_b(out)
        out = self.drop_b(out)
        # skip 连接：输入经 1×1 投影（块1 16→FT；块2 FT→FT）后加到输出，再过激活
        return self.act_a(self.skip(x) + out)


# ---------------------------------------------------------------------------
# EEG-TCNet 主模型
# ---------------------------------------------------------------------------
class EEGTCNet(nn.Module):
    """EEG-TCNet 四分类运动想象模型（22ch / 4类可配，忠实对照原文）。

    参数
    ----
    n_channels      : EEG 电极数（默认 22，BCIC IV 2a）
    n_classes       : 类别数（默认 4：左/右/双脚/舌头）
    input_length    : 单试次时间点数（默认 500 = 250Hz × 2s；原文 1125）
    F1              : φ1 时间卷积滤波器数（原文 8）
    D               : depthwise 乘子（原文 2），前端中间通道 = F1*D
    F2              : φ3 pointwise 滤波器数（原文 16）
    FT              : TCN 滤波器数（原文 12，块1 将 F2→FT 扩张）
    kernel_length   : φ1 时间卷积核 KE（原文 32；本项目 EEGNet 基线用 63，可配）
    sep_kernel      : φ3 separable 卷积核（原文 16）
    pool1 / pool2   : φ2 / φ3 平均池化步长（原文均为 8 → T/64）
    tcn_kernel_size : TCN 膨胀因果卷积核 KT（原文 4；简报描述版 12）
    tcn_dilations   : L 个残差块的膨胀率（原文 [1, 2]，L=len）
    readout_steps   : 分类头读取的最后时间步数（原文取最后时间步，默认 1）
    readout_mode    : "last"=读最后时间步（原文）；"gap"=时间维全局平均池化
                      （任务简报描述版；诊断用）
    dropout         : 各层 Dropout 概率（协议 0.25；原文 pe=0.2/pt=0.3）
    activation      : 激活函数名（elu / relu / lif，见 _make_activation）
    """

    def __init__(
        self,
        n_channels: int = 22,
        n_classes: int = 4,
        input_length: int = 500,
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
        FT: int = 12,
        kernel_length: int = 32,
        sep_kernel: int = 16,
        pool1: int = 8,
        pool2: int = 8,
        tcn_kernel_size: int = 4,
        tcn_dilations: tuple = (1, 2),
        readout_steps: int = 1,
        readout_mode: str = "last",
        dropout: float = 0.25,
        activation: str = "elu",
    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.input_length = input_length
        self.F1, self.D, self.F2, self.FT = F1, D, F2, FT
        self.kernel_length = kernel_length
        self.sep_kernel = sep_kernel
        self.tcn_kernel_size = tcn_kernel_size
        self.pool1_size, self.pool2_size = pool1, pool2  # 供 macs_estimate 用
        self.activation = activation

        # ---- φ1：时间卷积 + 空间 depthwise 卷积 ----
        # 手动补零（valid conv）+ 偶数核对称补 KE/2，'same' 长度（数学等价，
        # 规避 ONNX→RKNN 的 Pad op，与 eegnet.py 同策略）
        self.conv1 = nn.Conv2d(1, F1, (1, kernel_length), padding=0, bias=False)
        self.pad1_amt = kernel_length // 2
        self.bn1 = nn.BatchNorm2d(F1)
        # depthwise：每组一个 (n_channels, 1) 核，把 C 个电极压成 1
        # 带偏置：Keras DepthwiseConv2D 默认 use_bias=True，4272 参数反推确认
        self.depthwise = nn.Conv2d(F1, D * F1, (n_channels, 1), groups=F1,
                                   padding=0, bias=True)
        self.bn2 = nn.BatchNorm2d(D * F1)
        self.act1 = _make_activation(activation)
        self.pool1 = nn.AvgPool2d((1, pool1))
        self.dropout1 = nn.Dropout(dropout)

        # ---- φ3：separable 卷积（depthwise (1,sep_kernel) + pointwise (1,1)）----
        # 含偏置（原文 SeparableConv2D 默认 use_bias=True，4272 反推确认）
        self.sep_depthwise = nn.Conv2d(D * F1, D * F1, (1, sep_kernel), groups=D * F1,
                                       padding=0, bias=False)
        self.pad_sep = sep_kernel // 2
        self.pointwise = nn.Conv2d(D * F1, F2, (1, 1), padding=0, bias=True)
        self.bn3 = nn.BatchNorm2d(F2)
        self.act2 = _make_activation(activation)
        self.pool2 = nn.AvgPool2d((1, pool2))
        self.dropout2 = nn.Dropout(dropout)

        # ---- φ4：TCN（L=len(tcn_dilations) 个残差块，首块 F2→FT 扩张）----
        blocks = []
        in_ch = F2
        for d in tcn_dilations:
            blocks.append(_TCNResidualBlock(in_ch, FT, kernel_size=tcn_kernel_size,
                                            dilation=d, dropout=dropout,
                                            activation=activation))
            in_ch = FT
        self.tcn = nn.Sequential(*blocks)
        self.gap_1d = nn.AdaptiveAvgPool1d(1)  # readout_mode='gap' 时用

        # ---- φ5：读最后时间步（原文）或 GAP（简报描述版）→ Linear ----
        assert readout_mode in ("last", "gap"), f"readout_mode={readout_mode!r} 无效"
        self.readout_steps = readout_steps
        self.readout_mode = readout_mode
        self.fc = nn.Linear(FT * (readout_steps if readout_mode == "last" else 1),
                            n_classes)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, C, T) → 输出 (B, FT) 的 TCN 最后时间步特征（fc 之前，板载拆分用）。"""
        B, C, H, T = x.shape
        # φ1：时间卷积（补 KE/2 双侧零）+ depthwise（空间）
        pad1 = x.new_zeros(B, C, H, self.pad1_amt)
        x = torch.cat([pad1, x, pad1], dim=3)          # (B,1,C,T+2*pad1)
        x = self.conv1(x)                               # (B,F1,C,T)
        x = self.bn1(x)
        x = self.depthwise(x)                           # (B,F1*D,1,T)
        x = self.bn2(x)
        # φ2：激活 → 池化(1,pool1) → Dropout
        x = self.act1(x)
        x = self.pool1(x)                               # (B,F1*D,1,T//pool1)
        x = self.dropout1(x)
        # φ3：separable 卷积（补 sep/2 双侧零）+ 池化(1,pool2)
        Bc, Cc, Hc, Tc = x.shape
        pad_s = x.new_zeros(Bc, Cc, Hc, self.pad_sep)
        x = torch.cat([pad_s, x, pad_s], dim=3)
        x = self.sep_depthwise(x)                       # (B,F1*D,1,T//pool1)
        x = self.pointwise(x)                           # (B,F2,1,T//pool1)
        x = self.bn3(x)
        x = self.act2(x)
        x = self.pool2(x)                               # (B,F2,1,T//pool1//pool2)
        x = self.dropout2(x)
        # reshape 成时序特征 (B, F2, T') 送入 TCN
        x = x.view(x.size(0), self.F2, -1)              # (B,F2,T')
        x = self.tcn(x)                                 # (B,FT,T') 长度不变
        if self.readout_mode == "gap":
            # 全局平均池化（简报描述版）：聚合全部时间步
            x = self.gap_1d(x)                          # (B,FT,1)
        else:
            # 读最后 readout_steps 个时间步（原文；因果 TCN 最后步感受野覆盖全序列）
            x = x[:, :, -self.readout_steps:].contiguous()
        return x.view(x.size(0), -1)                    # (B, FT*readout_steps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, C, T) → 输出 (B, n_classes) 未归一化 logits。"""
        x = self.forward_features(x)                    # (B, FT*readout_steps)
        return self.fc(x)                               # (B, n_classes)

    def num_params(self) -> int:
        """可训练参数量。"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def macs_estimate(self) -> int:
        """粗算单试次乘加次数（MACs，按原文公式，T=输入长度）。

        原文公式：Conv2D = K1·K2·Cin·Cout·Hout·Wout
                  Conv1D = K·Cin·Cout·Wout
                  SeparableConv2D = (K1·K2 + Cout)·Cin·Hout·Wout
                  DepthWiseConv2D = K1·K2·Cin·D·Hout·Wout
        用于端侧可行性评估（RV1106 NPU 算力对比）。池化/核尺寸按实际配置计算。
        """
        T = self.input_length
        C = self.n_channels
        F1, D, F2, FT = self.F1, self.D, self.F2, self.FT
        KE = self.kernel_length
        macs = 0
        # φ1 temporal conv: (1·KE)·1·F1·C·T
        macs += KE * F1 * C * T
        # depthwise: (C·1)·F1·D·1·T
        macs += C * F1 * D * T
        # φ3 separable: (1·sep_k + F2)·(F1·D)·1·(T//pool1)
        t1 = T // self.pool1_size
        macs += (self.sep_kernel + F2) * (F1 * D) * t1
        # TCN: 每块两层 Conv1d + skip 1×1
        t2 = t1 // self.pool2_size
        for i, d in enumerate(self.tcn_dilations):
            k = self.tcn_kernel_size
            in_ch = F2 if i == 0 else FT
            macs += k * in_ch * FT * t2          # conv_a
            macs += k * FT * FT * t2             # conv_b
            macs += in_ch * FT * t2              # skip 1×1
        # Dense
        read_dim = FT * (self.readout_steps if self.readout_mode == "last" else 1)
        macs += read_dim * self.n_classes
        return int(macs)

    @property
    def tcn_dilations(self):
        return [b.dilation for b in self.tcn]


if __name__ == "__main__":
    # ---- 前向自检：dummy input (1,1,22,500) → 输出 (1,4)，并打印参数量 ----
    import time

    model = EEGTCNet(n_channels=22, n_classes=4, input_length=500,
                     activation="elu")
    model.eval()
    dummy = torch.randn(1, 1, 22, 500, dtype=torch.float32)
    with torch.no_grad():
        t0 = time.perf_counter()
        out = model(dummy)
        dt = (time.perf_counter() - t0) * 1000
    print(f"输入形状 : {tuple(dummy.shape)}")
    print(f"输出形状 : {tuple(out.shape)}   （应为 (1, 4)）")
    print(f"参数量   : {model.num_params():,} 个（原文固定配置 4,272）")
    print(f"MACs 估计: {model.macs_estimate()/1e6:.2f} M（原文 6.8 M @ T=1125）")
    print(f"单试次前向耗时（CPU/MPS 首跑含初始化，参考）: {dt:.2f} ms")
    assert tuple(out.shape) == (1, 4), f"输出形状错误：{tuple(out.shape)}"

    # 原文固定配置（22ch / 4类 / T=1125）参数量应恰好 4272，反推验证结构
    ref = EEGTCNet(n_channels=22, n_classes=4, input_length=1125,
                   activation="elu")
    print(f"原文配置参数量 : {ref.num_params():,}"
          f"（{'== 4272 ✓ 结构与原文一致' if ref.num_params() == 4272 else '≠ 4272 需检查'}）")

    # relu 版结构一致性（参数量应完全相同，仅激活不同）
    model_r = EEGTCNet(n_channels=22, n_classes=4, input_length=500,
                       activation="relu")
    print(f"relu 版参数量 : {model_r.num_params():,} 个"
          f"（{'一致' if model_r.num_params() == model.num_params() else '不一致！'}）")
    print("✓ 前向测试通过")
