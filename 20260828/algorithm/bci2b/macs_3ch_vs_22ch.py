#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
macs_3ch_vs_22ch.py — EEG-TCNet(LIF) 端侧计算量对比：3ch 二分类 vs 22ch 四分类

解析法统计（hook 按层输出形状 + 核形状计算 MACs，不依赖 thop，MPS 上可靠）：
  Conv2d   : out_numel × (in_ch/groups) × kH × kW
  Conv1d   : out_numel × (in_ch/groups) × k
  Linear   : out_numel × in_features
  LIF 激活 : 逐元素膜电位运算（mul+add+compare ≈ 3 次/元素/timestep），T=8 时
             附加元素级运算 ≈ 8×3×numel，非 MACs（卷积/全连接 MACs 两者相同，
             因为 LIF 层无参数）。报告里单列。

用法：
  env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin \
      /opt/miniconda3/envs/openmmlab/bin/python macs_3ch_vs_22ch.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

ALGO_DIR = Path("/Users/heyixuan/BCI/20260828/algorithm")
sys.path.insert(0, str(ALGO_DIR))

from eeg_tcnet_lif import EEGTCNetLIF  # noqa: E402


class MACsCounter:
    """按层 hook 统计 MACs（累乘计数，不含加法）。"""

    def __init__(self):
        self.macs: dict[str, int] = {}
        self.hooks = []

    def _mk_hook(self, name: str):
        def hook(module, inp, out):
            out_t = out if torch.is_tensor(out) else out[0]
            o = out_t.numel()  # 含 batch
            if isinstance(module, nn.Conv2d):
                per_out = (module.in_channels // module.groups) * module.kernel_size[0] * module.kernel_size[1]
            elif isinstance(module, nn.Conv1d):
                per_out = (module.in_channels // module.groups) * module.kernel_size[0]
            elif isinstance(module, nn.Linear):
                per_out = module.in_features
            else:
                per_out = 0
            self.macs[name] = o * per_out
        return hook

    def attach(self, model: nn.Module):
        for name, m in model.named_modules():
            if isinstance(m, (nn.Conv2d, nn.Conv1d, nn.Linear)):
                self.hooks.append(m.register_forward_hook(self._mk_hook(name)))
        return self

    def detach(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()

    def total(self) -> int:
        return sum(self.macs.values())


def _lif_elementwise_ops(model: nn.Module, x: torch.Tensor, T: int = 8) -> int:
    """LIF 激活附加元素级运算估计：每个 LIF 位点输出元素 × T × 3（乘/加/比较）。"""
    ops = 0
    model.eval()
    with torch.no_grad():
        model(x)
    from lif_eegnet import LIFActivation
    for name, m in model.named_modules():
        if isinstance(m, LIFActivation):
            # last_firing_rate 只给标量；用注册 buffer 的输入形状更稳，这里用最近一次前向的统计
            pass
    return ops  # 由调用方按激活位点输出形状单独估算


def analyze(tag: str, n_ch: int, n_cls: int, input_length: int = 500, T: int = 8):
    model = EEGTCNetLIF(n_channels=n_ch, n_classes=n_cls, input_length=input_length,
                        activation="lif", threshold=1.0, tau=2.0, timesteps=T, alpha=4.0)
    n_params = model.num_params()
    counter = MACsCounter().attach(model)
    dummy = torch.randn(1, 1, n_ch, input_length)
    model.eval()
    with torch.no_grad():
        model(dummy)
    macs = counter.total()
    counter.detach()

    # LIF 激活位点逐元素运算量：重跑一次，取各 LIF 模块输出张量 numel
    lif_els = 0
    lif_outs: dict[str, int] = {}
    def _hook_lif(name):
        def h(module, inp, out):
            lif_outs[name] = out.numel()
        return h
    hooks = []
    for name, m in model.named_modules():
        if type(m).__name__ == "LIFActivation":
            hooks.append(m.register_forward_hook(_hook_lif(name)))
    with torch.no_grad():
        model(dummy)
    for h in hooks:
        h.remove()
    lif_els = sum(lif_outs.values())
    lif_ops = lif_els * T * 3

    print(f"{tag}: n_channels={n_ch} n_classes={n_cls}")
    print(f"  参数量        : {n_params:,}")
    print(f"  卷积/FC MACs   : {macs:,}  ({macs/1e6:.3f} M)")
    print(f"  LIF 激活元素级: {lif_els:,} 元素 × T={T} × 3 ops = {lif_ops:,} ops ({lif_ops/1e6:.3f} M)")
    print(f"  单次前向总 MACs: {macs:,}（LIF 版与 relu 版相同，激活层无参数）")
    print()
    return dict(tag=tag, n_ch=n_ch, n_cls=n_cls, params=n_params, macs=macs,
                lif_els=lif_els, lif_ops=lif_ops)


if __name__ == "__main__":
    print("EEG-TCNet(LIF) 端侧计算量（brief 配置：F1=8 D=2 F2=16 FT=12 KE=63 KT=12 pool 4/8 GAP）\n")
    r3 = analyze("3ch 二分类（2b）", 3, 2)
    r22_4 = analyze("22ch 四分类（2a）", 22, 4)
    r22_2 = analyze("22ch 二分类（对照）", 22, 2)
    print("=" * 60)
    print(f"MACs 比值 : 3ch vs 22ch四分类 = 1 : {r22_4['macs']/r3['macs']:.2f}（3ch 为其 {100*r3['macs']/r22_4['macs']:.1f}%）")
    print(f"参数量比值: 3ch {r3['params']:,} vs 22ch {r22_4['params']:,}（3ch 为其 {100*r3['params']/r22_4['params']:.1f}%）")
