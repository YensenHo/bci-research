#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_warmstart.py — 计算「ELU→LIF 激活交换」的即时代价（不微调）

目的：定位四分类下 LIF-EEGNet 与 EEGNet 差距的来源。
  - 载入 EEGNet ELU 权重（eegnet_4class.pth）→ eval 干净精度（应复现基线）。
  - 把同一份权重直接塞进 LIFEEGNet(activation='lif', thr/tau/T)，不微调，
    直接测 eval 精度 → 即「激活交换」本身的信息损失。
  - 再对比 surrogate 微调后的最终精度（各 thr/tau/T 报告），看微调能补回多少。
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ALGO_DIR = Path(__file__).resolve().parent
PROJECT = Path("/Users/heyixuan/BCI/luckfox-deploy")
sys.path.insert(0, str(ALGO_DIR))
sys.path.insert(0, str(PROJECT / "data"))
sys.path.insert(0, str(PROJECT / "model"))

from dataset import EEGDataset  # noqa: E402
from eegnet import EEGNet       # noqa: E402
from lif_eegnet import LIFEEGNet  # noqa: E402


def acc_on(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            correct += (model(xb).argmax(1) == yb).sum().item()
            total += xb.size(0)
    return correct / total


def main() -> int:
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    eval_ds = EEGDataset(ALGO_DIR / "data_4class" / "eval.npz")
    loader = DataLoader(eval_ds, batch_size=16, shuffle=False)

    ck = torch.load(ALGO_DIR / "eegnet_4class.pth", map_location="cpu", weights_only=False)
    sd = ck["state_dict"]

    # 1. EEGNet ELU（复现基线）
    ann = EEGNet(n_channels=22, n_classes=4, activation="elu").to(device)
    ann.load_state_dict(sd, strict=True)
    print(f"EEGNet ELU (eegnet_4class.pth)            : {acc_on(ann, loader, device)*100:.2f}%")

    # 2. 同权重直接换 LIF（不微调）—— 激活交换即时代价
    for thr, tau, T in [(1.0, 2.0, 8), (0.25, 2.0, 8), (0.5, 2.0, 16)]:
        lif = LIFEEGNet(n_channels=22, n_classes=4, activation="lif",
                        threshold=thr, tau=tau, timesteps=T, alpha=4.0).to(device)
        lif.load_state_dict(sd, strict=True)
        print(f"LIF 直接换激活(不微调) thr={thr} tau={tau} T={T} : "
              f"{acc_on(lif, loader, device)*100:.2f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
