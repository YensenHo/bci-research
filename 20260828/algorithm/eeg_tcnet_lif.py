#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eeg_tcnet_lif.py — LIF 化 EEG-TCNet 模型定义（surrogate 训练版）

本文件是 eeg_tcnet.py 的**拷贝改造**（原文件只读不动），唯一差异：
  1. 激活工厂 `_make_activation` 支持透传 LIF 超参（threshold/tau/timesteps/alpha），
     不再写死 LIFActivation() 默认参数；
  2. TCN 残差块的残差输出激活独立为 act_res（原来复用 act_a），便于逐位点
     统计发放率（LIF 层无参数，state_dict 键零回归，可 strict=True 热启动
     relu/elu 版权重）；
  3. forward 结束后收集各 LIF 位点的平均发放率到 self.firing_rates（诊断用，
     纯附加，不影响计算图）。

LIF 化位置（诚实标注，对应任务清单第 4 点）
----------------------------------------------------
  - act1 / act2 ：前端（EEGNet 部分）两个激活位点（pool 之前），与 LIF-EEGNet 相同；
  - TCN 块内：每个残差块 3 个 LIF 位点 = 两层**膨胀因果卷积后的激活**
    （act_a: conv_a→BN 后；act_b: conv_b→BN 后）+ 残差输出的激活（act_res）。
    TCN 膨胀因果卷积的 LIF 化是类脑化的难点所在：TCN 特征通道少（FT=12）、
    时间步短（T'=15），LIF 率编码 + surrogate 梯度在此处容易发放率塌缩或梯度消失。
  - 读出头 GAP 直接作用在 LIF 发放率输出（∈[0,1]）上，无额外激活。

LIF 层语义（复用 lif_eegnet.LIFActivation，rate-based）：
  mem = decay*mem + x（decay=exp(-1/tau)），spk = H(mem - thr)，硬重置，
  输出 = spk 在 timesteps 上的均值（发放率 ∈ [0,1]），反向用 sigmoid surrogate 梯度。

用法（前向自检）：
  env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin \\
      /opt/miniconda3/envs/openmmlab/bin/python eeg_tcnet_lif.py
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 激活工厂：elu / relu / lif（lif 透传 thr/tau/T/alpha）
# ---------------------------------------------------------------------------
def _make_activation(name: str, **lif_kwargs) -> nn.Module:
    """按名字构造激活层。

    - "elu"  : nn.ELU()
    - "relu" : nn.ReLU()
    - "lif"  : lif_eegnet.LIFActivation(**lif_kwargs)，surrogate 发放率激活
               （threshold/tau/timesteps/alpha 可配；LIF 层无参数，state_dict 零回归）
    """
    if name == "elu":
        return nn.ELU()
    if name == "relu":
        return nn.ReLU()
    if name == "lif":
        from lif_eegnet import LIFActivation  # 同目录，延迟导入
        return LIFActivation(**lif_kwargs)
    raise ValueError(f"未知激活函数: {name!r}（可选 elu / relu / lif）")


# ---------------------------------------------------------------------------
# TCN 残差块：两层膨胀因果卷积（Bai et al. 2018 式），LIF 化版
# ---------------------------------------------------------------------------
class _TCNResidualBlockLIF(nn.Module):
    """一个 TCN 残差块（原文 Fig.2 Right），激活可配 LIF。

    结构（与 eeg_tcnet.py 的 _TCNResidualBlock 完全一致，参数逐层对齐）：
      conv_a（膨胀 d, 因果）→ BN → 激活 → Dropout
      conv_b（膨胀 d, 因果）→ BN → 激活 → Dropout
      skip：输入 + 输出；通道数变化时 1×1 卷积投影对齐
    差异：残差输出激活独立为 act_res（原版复用 act_a），只影响诊断统计，
    数学语义与 ELU/ReLU 版完全等价（同类型激活）。

    因果性：手动左补零 (KT-1)*d，输出长度与输入一致，严格因果。
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 4,
                 dilation: int = 1, dropout: float = 0.25, activation: str = "elu",
                 **lif_kwargs):
        super().__init__()
        self.dilation = dilation
        self.pad_left = (kernel_size - 1) * dilation   # 因果：只补左边
        # 两层膨胀因果卷积（含偏置，与原文 4272 参数反推一致）
        self.conv_a = nn.Conv1d(in_channels, out_channels, kernel_size,
                                dilation=dilation, padding=0, bias=True)
        self.bn_a = nn.BatchNorm1d(out_channels)
        self.act_a = _make_activation(activation, **lif_kwargs)
        self.drop_a = nn.Dropout(dropout)
        self.conv_b = nn.Conv1d(out_channels, out_channels, kernel_size,
                                dilation=dilation, padding=0, bias=True)
        self.bn_b = nn.BatchNorm1d(out_channels)
        self.act_b = _make_activation(activation, **lif_kwargs)
        self.drop_b = nn.Dropout(dropout)
        # 残差输出的独立激活（LIF 化诊断用；无参数，不影响 state_dict）
        self.act_res = _make_activation(activation, **lif_kwargs)
        # skip 投影：块1 (16→FT) 带偏置；块2 (FT→FT) 无偏置（对齐原文 4272）
        self.skip = nn.Conv1d(in_channels, out_channels, 1, bias=(in_channels != out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C_in, T) → 输出 (B, C_out, T)（长度不变，残差连接）。"""
        out = F.pad(x, (self.pad_left, 0))            # 因果左补零
        out = self.conv_a(out)
        out = self.bn_a(out)
        out = self.act_a(out)                          # LIF 位点：conv_a 后
        out = self.drop_a(out)
        out = F.pad(out, (self.pad_left, 0))
        out = self.conv_b(out)
        out = self.bn_b(out)
        out = self.act_b(out)                          # LIF 位点：conv_b 后
        out = self.drop_b(out)
        # skip 连接：输入经 1×1 投影后加到输出，再过激活（LIF 位点：残差输出）
        return self.act_res(self.skip(x) + out)


# ---------------------------------------------------------------------------
# EEG-TCNet 主模型（LIF 化拷贝版）
# ---------------------------------------------------------------------------
class EEGTCNetLIF(nn.Module):
    """EEG-TCNet 四分类运动想象模型（22ch / 4类），activation 可配 elu/relu/lif。

    结构、参数、命名与 eeg_tcnet.EEGTCNet 完全一致（参数量 9,512 @ brief 配置，
    激活层无参数），仅：
      - 激活位点全部走 _make_activation 并透传 lif_kwargs；
      - forward 后收集各 LIF 位点平均发放率（self.firing_rates）。
    因此可 strict=True 热启动自 eeg_tcnet_4class_relu_brief.pth（relu 版权重）。

    默认参数 = 简报最优配置（KE=63/KT=12/pool4/8/GAP），与 relu/ELU 版对比基线一致。
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
        kernel_length: int = 63,
        sep_kernel: int = 16,
        pool1: int = 4,
        pool2: int = 8,
        tcn_kernel_size: int = 12,
        tcn_dilations: tuple = (1, 2),
        readout_steps: int = 1,
        readout_mode: str = "gap",
        dropout: float = 0.25,
        activation: str = "lif",
        **lif_kwargs,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.input_length = input_length
        self.F1, self.D, self.F2, self.FT = F1, D, F2, FT
        self.kernel_length = kernel_length
        self.sep_kernel = sep_kernel
        self.tcn_kernel_size = tcn_kernel_size
        self.pool1_size, self.pool2_size = pool1, pool2
        self.activation = activation
        self.lif_kwargs = lif_kwargs
        self.firing_rates: dict = {}   # 最近一次 forward 的各 LIF 位点平均发放率

        # ---- φ1：时间卷积 + 空间 depthwise 卷积（同 eeg_tcnet.py）----
        self.conv1 = nn.Conv2d(1, F1, (1, kernel_length), padding=0, bias=False)
        self.pad1_amt = kernel_length // 2
        self.bn1 = nn.BatchNorm2d(F1)
        self.depthwise = nn.Conv2d(F1, D * F1, (n_channels, 1), groups=F1,
                                   padding=0, bias=True)
        self.bn2 = nn.BatchNorm2d(D * F1)
        self.act1 = _make_activation(activation, **lif_kwargs)
        self.pool1 = nn.AvgPool2d((1, pool1))
        self.dropout1 = nn.Dropout(dropout)

        # ---- φ3：separable 卷积 ----
        self.sep_depthwise = nn.Conv2d(D * F1, D * F1, (1, sep_kernel), groups=D * F1,
                                       padding=0, bias=False)
        self.pad_sep = sep_kernel // 2
        self.pointwise = nn.Conv2d(D * F1, F2, (1, 1), padding=0, bias=True)
        self.bn3 = nn.BatchNorm2d(F2)
        self.act2 = _make_activation(activation, **lif_kwargs)
        self.pool2 = nn.AvgPool2d((1, pool2))
        self.dropout2 = nn.Dropout(dropout)

        # ---- φ4：TCN（LIF 化残差块）----
        blocks = []
        in_ch = F2
        for d in tcn_dilations:
            blocks.append(_TCNResidualBlockLIF(in_ch, FT, kernel_size=tcn_kernel_size,
                                               dilation=d, dropout=dropout,
                                               activation=activation, **lif_kwargs))
            in_ch = FT
        self.tcn = nn.Sequential(*blocks)
        self.gap_1d = nn.AdaptiveAvgPool1d(1)

        # ---- φ5：读最后时间步 或 GAP → Linear ----
        assert readout_mode in ("last", "gap"), f"readout_mode={readout_mode!r} 无效"
        self.readout_steps = readout_steps
        self.readout_mode = readout_mode
        self.fc = nn.Linear(FT * (readout_steps if readout_mode == "last" else 1),
                            n_classes)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, C, T) → 输出 (B, FT) 特征（fc 之前）。"""
        B, C, H, T = x.shape
        # φ1：时间卷积（补 KE/2 双侧零）+ depthwise（空间）
        pad1 = x.new_zeros(B, C, H, self.pad1_amt)
        x = torch.cat([pad1, x, pad1], dim=3)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.depthwise(x)
        x = self.bn2(x)
        x = self.act1(x)            # LIF 位点 1：前端 φ2（pool 前）
        x = self.pool1(x)
        x = self.dropout1(x)
        # φ3：separable 卷积 + 池化
        Bc, Cc, Hc, Tc = x.shape
        pad_s = x.new_zeros(Bc, Cc, Hc, self.pad_sep)
        x = torch.cat([pad_s, x, pad_s], dim=3)
        x = self.sep_depthwise(x)
        x = self.pointwise(x)
        x = self.bn3(x)
        x = self.act2(x)            # LIF 位点 2：前端 φ3（pool 前）
        x = self.pool2(x)
        x = self.dropout2(x)
        x = x.view(x.size(0), self.F2, -1)
        x = self.tcn(x)             # LIF 位点 3..8：两个残差块 × (act_a/act_b/act_res)
        if self.readout_mode == "gap":
            x = self.gap_1d(x)
        else:
            x = x[:, :, -self.readout_steps:].contiguous()
        # 收集各 LIF 位点平均发放率（诊断用；纯附加，不参与计算图）
        self.firing_rates = {}
        from lif_eegnet import LIFActivation  # noqa: PLC0415
        for name, m in self.named_modules():
            if isinstance(m, LIFActivation):
                self.firing_rates[name] = float(m.last_firing_rate)
        return x.view(x.size(0), -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, C, T) → 输出 (B, n_classes) 未归一化 logits。"""
        x = self.forward_features(x)
        return self.fc(x)

    def num_params(self) -> int:
        """可训练参数量。"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def macs_estimate(self) -> int:
        """粗算单试次乘加次数（MACs，同 eeg_tcnet.py 公式，T=输入长度）。

        公式：Conv2D = K1·K2·Cin·Cout·Hout·Wout
              Conv1D = K·Cin·Cout·Wout
              SeparableConv2D = (K1·K2 + Cout)·Cin·Hout·Wout
              DepthWiseConv2D = K1·K2·Cin·D·Hout·Wout
        与激活无关（LIF 位点只增加加法/比较开销，卷积 MACs 相同）；
        端侧可行性以 eval 脚本 hook 实测值为准。
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
    # ---- 前向自检：lif 模式 forward/backward + 发放率收集 + relu 权重热启动 ----
    import sys
    from pathlib import Path

    torch.manual_seed(42)
    ALGO = Path(__file__).resolve().parent
    sys.path.insert(0, str(ALGO))

    model = EEGTCNetLIF(activation="lif", threshold=1.0, tau=2.0, timesteps=8)
    print(f"参数量 : {model.num_params():,}（brief 配置应为 9,512）")
    dummy = torch.randn(2, 1, 22, 500)
    out = model(dummy)
    print(f"前向   : {tuple(dummy.shape)} → {tuple(out.shape)}")
    print(f"发放率 : { {k: round(v, 4) for k, v in model.firing_rates.items()} }")
    loss = out.sum()
    loss.backward()
    n_grad = sum(1 for p in model.parameters() if p.requires_grad and p.grad is not None)
    print(f"反向   : {n_grad}/{sum(1 for p in model.parameters() if p.requires_grad)} 参数有梯度（surrogate 生效）")

    # 热启动自检：relu brief 权重 strict 载入（激活层无参数，键应零回归）
    ws = ALGO / "eeg_tcnet_4class_relu_brief.pth"
    if ws.exists():
        ck = torch.load(ws, map_location="cpu", weights_only=False)
        missing, unexpected = model.load_state_dict(ck["state_dict"], strict=True)
        assert not missing and not unexpected, f"热启动键不匹配: {missing} / {unexpected}"
        print(f"热启动 : {ws.name} strict=True ✓（relu→LIF，conv/BN 权重复用）")
    print("✓ LIF 化 EEG-TCNet 自检通过")
