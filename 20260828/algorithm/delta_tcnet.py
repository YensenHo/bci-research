#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
delta_tcnet.py — 全脉冲 LIF-EEG-TCNet：输入脉冲编码（delta 调制）→ 全脉冲网络

定位（延续「类脑 + BCI + HBC」纲领的稀疏性主线）
------------------------------------------------
既有结论：前端 φ1（conv1+depthwise）处理**连续 EEG** 占 96.9% MACs、不可稀疏化；
LIF 脉冲域仅 3.1%。本文件把输入换成 **delta 调制脉冲序列**（delta_encode.DeltaEncode），
使 φ1 的输入也进入稀疏脉冲域 → 前端 φ1 获得事件驱动潜力（输入稀疏化后 MACs 理论减少）。

「全脉冲」的含义（诚实标注，不夸大）
------------------------------------
  - 输入    ：连续 EEG (B,1,C,T) → DeltaEncode → 脉冲 (B,1,C,T) signed 或 (B,2,C,T) posneg
              （值域 {-1,0,1} / {0,1}，稀疏事件码；网络自身不再接收连续输入）
  - 全部激活：act1/act2 + TCN 内 6 个位点全部为 LIF 发放率（surrogate 训练，同 eeg_tcnet_lif）
  - 读出头  ：GAP 聚合发放率 → fc(4)（输出即「发放率 → 分类」）
  - 中间 BN ：卷积后 BatchNorm 为线性归一化层，保持连续（与基线架构一致）；
              conv1 的**输入**是脉冲，因此 conv1 是事件驱动可稀疏化的第一层
              （事件驱动 ACs ≈ 密集 MACs × 输入脉冲密度）

与 eeg_tcnet_lif.py 的关系（拷贝改造，原文件只读不动）
----------------------------------------------------
  - 结构/参数/命名完全一致（brief 配置 9,512 参数；activation="lif"），仅新增：
    1. DeltaEncode 输入编码层（无参数；theta 为 buffer，随权重保存）；
    2. conv1 输入通道可配 input_channels=1（signed 脉冲）或 2（posneg 双通道）；
    3. forward 收集输入脉冲密度 self.input_density（诊断用）。
  - 热启动：relu brief 权重 strict 兼容。posneg 模式用「[w, -w] 拆分」初始化
    conv1 双通道核（数学上等价于 signed 单通道：w·pos + (−w)·neg = w·(pos−neg)）。

用法（前向自检）
  env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin \\
      /opt/miniconda3/envs/openmmlab/bin/python delta_tcnet.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ALGO_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ALGO_DIR))

from delta_encode import DeltaEncode  # noqa: E402


# ---------------------------------------------------------------------------
# 激活工厂（同 eeg_tcnet_lif.py：elu / relu / lif，lif 透传超参）
# ---------------------------------------------------------------------------
def _make_activation(name: str, **lif_kwargs) -> nn.Module:
    if name == "elu":
        return nn.ELU()
    if name == "relu":
        return nn.ReLU()
    if name == "lif":
        from lif_eegnet import LIFActivation  # 同目录，延迟导入
        return LIFActivation(**lif_kwargs)
    raise ValueError(f"未知激活函数: {name!r}（可选 elu / relu / lif）")


# ---------------------------------------------------------------------------
# TCN 残差块（LIF 化，同 eeg_tcnet_lif._TCNResidualBlockLIF）
# ---------------------------------------------------------------------------
class _TCNResidualBlockLIF(nn.Module):
    """一个 TCN 残差块（原文 Fig.2 Right），激活可配 LIF。

    两层膨胀因果卷积（BN→激活→Dropout 间隔）+ skip 残差（1×1 投影对齐通道），
    残差输出激活独立为 act_res（诊断用）。因果性靠手动左补零 (KT-1)·d。
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 4,
                 dilation: int = 1, dropout: float = 0.25, activation: str = "elu",
                 **lif_kwargs):
        super().__init__()
        self.dilation = dilation
        self.pad_left = (kernel_size - 1) * dilation   # 因果：只补左边
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
        self.act_res = _make_activation(activation, **lif_kwargs)
        self.skip = nn.Conv1d(in_channels, out_channels, 1, bias=(in_channels != out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.pad(x, (self.pad_left, 0))
        out = self.conv_a(out)
        out = self.bn_a(out)
        out = self.act_a(out)                          # LIF 位点：conv_a 后
        out = self.drop_a(out)
        out = F.pad(out, (self.pad_left, 0))
        out = self.conv_b(out)
        out = self.bn_b(out)
        out = self.act_b(out)                          # LIF 位点：conv_b 后
        out = self.drop_b(out)
        return self.act_res(self.skip(x) + out)        # LIF 位点：残差输出


# ---------------------------------------------------------------------------
# 全脉冲 EEG-TCNet 主模型
# ---------------------------------------------------------------------------
class DeltaTCNet(nn.Module):
    """输入脉冲编码 + LIF-EEG-TCNet 全脉冲网络（22ch / 4类）。

    参数
    ----
    除 eeg_tcnet_lif.EEGTCNetLIF 全部结构参数外，新增：
    input_channels : conv1 输入通道数。1 = signed 脉冲（值域 {-1,0,1}，与基线架构
                     完全一致可 strict 热启动）；2 = posneg 正负双 0/1 脉冲通道。
    delta_k        : 阈值系数 θ = k·std_通道（编码层固定，训练前从训练集统计）
    delta_bits     : 1 = 1bit delta；2 = 2bit delta（幅度 {0,1,2}+符号，仅 signed）
    delta_mode     : "signed" / "posneg"（见 delta_encode.delta_encode）
    theta          : (C,) 逐通道阈值（训练集 k·std）；None 时占位，训练脚本会覆盖
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
        input_channels: int = 1,
        delta_k: float = 1.0,
        delta_bits: int = 1,
        delta_mode: str = "signed",
        theta: torch.Tensor | None = None,
        **lif_kwargs,
    ):
        super().__init__()
        assert input_channels in (1, 2), "input_channels 仅支持 1（signed）/ 2（posneg）"
        assert delta_mode in ("signed", "posneg")
        if delta_mode == "posneg":
            assert input_channels == 2, "posneg 模式 conv1 输入通道必须为 2"
        if delta_bits == 2:
            assert delta_mode == "signed", "2bit delta 仅支持 signed 模式"

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
        self.input_channels = input_channels
        self.firing_rates: dict = {}   # 最近一次 forward 的各 LIF 位点平均发放率
        self.input_density: float = 0.0  # 最近一次 forward 的输入脉冲密度

        # ---- 输入脉冲编码（delta 调制；无参数，theta 为 buffer）----
        self.delta = DeltaEncode(theta=theta, k=delta_k, bits=delta_bits, mode=delta_mode)

        # ---- φ1：时间卷积（处理脉冲序列）+ 空间 depthwise 卷积 ----
        self.conv1 = nn.Conv2d(input_channels, F1, (1, kernel_length), padding=0, bias=False)
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

        # ---- φ5：GAP（发放率聚合）或末步 → Linear ----
        assert readout_mode in ("last", "gap"), f"readout_mode={readout_mode!r} 无效"
        self.readout_steps = readout_steps
        self.readout_mode = readout_mode
        self.fc = nn.Linear(FT * (readout_steps if readout_mode == "last" else 1),
                            n_classes)

    # ------------------------------------------------------------------
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B,1,C,T) 连续 EEG → 脉冲编码 → LIF 特征 (B, FT)。"""
        # ---- 输入脉冲编码：连续 EEG → 稀疏脉冲序列（全脉冲网络的输入）----
        x = self.delta(x)                       # (B,1,C,T) signed 或 (B,2,C,T) posneg
        self.input_density = self.delta.input_density
        B, Cin, H, T = x.shape
        # φ1：时间卷积（补 KE/2 双侧零）+ depthwise（空间）
        pad1 = x.new_zeros(B, Cin, H, self.pad1_amt)
        x = torch.cat([pad1, x, pad1], dim=3)
        x = self.conv1(x)                       # (B,F1,C,T)  输入稀疏 → 事件驱动可稀疏化
        x = self.bn1(x)
        x = self.depthwise(x)                   # (B,F1*D,1,T)
        x = self.bn2(x)
        x = self.act1(x)                        # LIF 位点 1：前端 φ2（pool 前）
        x = self.pool1(x)
        x = self.dropout1(x)
        # φ3：separable 卷积 + 池化
        Bc, Cc, Hc, Tc = x.shape
        pad_s = x.new_zeros(Bc, Cc, Hc, self.pad_sep)
        x = torch.cat([pad_s, x, pad_s], dim=3)
        x = self.sep_depthwise(x)
        x = self.pointwise(x)
        x = self.bn3(x)
        x = self.act2(x)                        # LIF 位点 2：前端 φ3（pool 前）
        x = self.pool2(x)
        x = self.dropout2(x)
        x = x.view(x.size(0), self.F2, -1)
        x = self.tcn(x)                         # LIF 位点 3..8
        if self.readout_mode == "gap":
            x = self.gap_1d(x)                  # 发放率聚合（GAP）
        else:
            x = x[:, :, -self.readout_steps:].contiguous()
        # 收集各 LIF 位点平均发放率（诊断用；纯附加）
        self.firing_rates = {}
        from lif_eegnet import LIFActivation  # noqa: PLC0415
        for name, m in self.named_modules():
            if isinstance(m, LIFActivation):
                self.firing_rates[name] = float(m.last_firing_rate)
        return x.view(x.size(0), -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B,1,C,T) 连续 EEG → 输出 (B, n_classes) 未归一化 logits。"""
        x = self.forward_features(x)
        return self.fc(x)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def tcn_dilations(self):
        return [b.dilation for b in self.tcn]


# ---------------------------------------------------------------------------
# 热启动：relu brief 权重 → DeltaTCNet（含 posneg 的 conv1 双通道拆分）
# ---------------------------------------------------------------------------
def load_warm_start(model: DeltaTCNet, ckpt: dict) -> list[str]:
    """加载 relu/elu 版权重（conv/BN/fc），兼容两种输入形态。

    - signed（input_channels=1）：conv1 形状与基线一致，直接加载；
    - posneg（input_channels=2）：conv1 核 [w, -w] 拆分（数学等价 signed 单通道）；
    - LIF 层无参数、DeltaEncode 无参数 → 键零回归；theta buffer 不在旧权重中，
      属于预期 missing，由训练脚本在加载后 set_theta 覆盖。
    返回 : 预期 missing 的键列表（正常应为 ["delta.theta"]）。
    """
    sd = dict(ckpt["state_dict"])
    if model.input_channels == 2 and "conv1.weight" in sd:
        w = sd["conv1.weight"]                       # (F1, 1, 1, KE)
        assert w.shape[1] == 1, f"基线 conv1 应为单输入通道，实际 {tuple(w.shape)}"
        sd["conv1.weight"] = torch.cat([w, -w], dim=1)   # (F1, 2, 1, KE)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    extra = [k for k in missing if k != "delta.theta"]
    assert not extra, f"热启动缺少非预期键: {extra}"
    assert not unexpected, f"热启动出现多余键: {unexpected}"
    return list(missing)


if __name__ == "__main__":
    # ---- 前向自检：signed / posneg 两种输入形态 + 热启动 + 输入密度 ----
    torch.manual_seed(42)
    theta = torch.full((22,), 0.5)   # 占位阈值（真实训练从训练集统计）

    for mode, inch in (("signed", 1), ("posneg", 2)):
        model = DeltaTCNet(input_channels=inch, delta_k=0.5, delta_bits=1,
                           delta_mode=mode, theta=theta,
                           threshold=1.0, tau=2.0, timesteps=8)
        dummy = torch.randn(2, 1, 22, 500)
        out = model(dummy)
        print(f"[{mode}] 参数量={model.num_params():,}  "
              f"输入 {tuple(dummy.shape)} → 脉冲 {tuple(model.delta(dummy).shape)} → {tuple(out.shape)}  "
              f"输入密度={model.input_density*100:.1f}%  "
              f"平均发放率={sum(model.firing_rates.values())/max(len(model.firing_rates),1)*100:.1f}%")
        loss = out.sum(); loss.backward()
        n_param_tensors = sum(1 for p in model.parameters() if p.requires_grad)
        n_grad = sum(1 for p in model.parameters() if p.requires_grad and p.grad is not None)
        assert n_grad == n_param_tensors, "梯度未覆盖全部参数张量"
        # 热启动自检（relu brief）
        ws = ALGO_DIR / "eeg_tcnet_4class_relu_brief.pth"
        if ws.exists():
            ck = torch.load(ws, map_location="cpu", weights_only=False)
            missing = load_warm_start(model, ck)
            assert missing == ["delta.theta"], f"热启动 missing 异常: {missing}"
            print(f"    ✓ 热启动 {ws.name} 成功（missing={missing}）")
    print("✓ 全脉冲 DeltaTCNet 自检通过")
