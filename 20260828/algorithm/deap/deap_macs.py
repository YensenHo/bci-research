#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""deap_macs.py — DEAP 32ch 二分类 5s 窗 端侧计算量统计（与 macs_3ch_vs_22ch.py 同法）"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

ALGO_DIR = Path("/Users/heyixuan/BCI/20260828/algorithm")
sys.path.insert(0, str(ALGO_DIR))

from eeg_tcnet_lif import EEGTCNetLIF  # noqa: E402


class MACsCounter:
    def __init__(self):
        self.macs: dict[str, int] = {}
        self.hooks = []

    def _mk_hook(self, name: str):
        def hook(module, inp, out):
            out_t = out if torch.is_tensor(out) else out[0]
            o = out_t.numel()
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


def main() -> int:
    model = EEGTCNetLIF(n_channels=32, n_classes=2, input_length=640,
                        activation="lif", threshold=1.0, tau=2.0,
                        timesteps=8, alpha=4.0)
    n_params = model.num_params()
    counter = MACsCounter().attach(model)
    dummy = torch.randn(1, 1, 32, 640)
    model.eval()
    with torch.no_grad():
        model(dummy)
    macs = counter.total()
    counter.detach()

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
    lif_ops = lif_els * 8 * 3

    print(f"DEAP 32ch 二分类 5s 窗（640 样本）:")
    print(f"  参数量        : {n_params:,}")
    print(f"  卷积/FC MACs   : {macs:,}  ({macs/1e6:.3f} M)")
    print(f"  LIF 激活元素级: {lif_els:,} 元素 × T=8 × 3 ops = {lif_ops:,} ops ({lif_ops/1e6:.3f} M)")
    print(f"  单次前向总 MACs: {macs:,}（LIF 版与 relu 版相同，激活层无参数）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
