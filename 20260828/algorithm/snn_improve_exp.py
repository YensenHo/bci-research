#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
snn_improve_exp.py — 一次性实验：ANN→SNN 精度恢复（数据驱动归一化 + timesteps，MPS 加速）

验证 Rueckauer 2017 式「数据驱动逐层归一化」能否把 SNN 精度拉回接近 ANN。
结果如实记录。关键：输入 x 恒定 → conv1/depthwise 每 timestep 相同，只算一次。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ALGO_DIR = Path(__file__).resolve().parent
DATA = Path("/Users/heyixuan/BCI/luckfox-deploy/data/processed")
sys.path.insert(0, str(ALGO_DIR))

from lif_eegnet import LIFEEGNet                      # noqa: E402
from ann2snn import fold_bn_layer, lif_step          # noqa: E402


def _fold(ann):
    conv1 = fold_bn_layer(ann.conv1, ann.bn1)
    dw = fold_bn_layer(ann.depthwise, ann.bn2)
    pw = fold_bn_layer(ann.pointwise, ann.bn3)
    sdw = ann.sep_depthwise
    fc = ann.fc
    return conv1, dw, sdw, pw, fc


def snn_logits(x, conv1, dw, sdw, pw, fc, thr1, thr2, T, pad1_amt, pad2_amt):
    """批量 SNN 前向（decay=1 纯 IF，硬重置），a1 恒定只算一次。"""
    B, C, H, W = x.shape
    n_classes = fc.weight.shape[0]
    pad1 = x.new_zeros(B, C, H, pad1_amt)
    xp = torch.cat([pad1, x, pad1], dim=3)
    a1 = dw(conv1(xp))
    F2 = a1.shape[1]
    mem1 = torch.zeros_like(a1)
    mem2 = None
    out = torch.zeros(B, n_classes, device=x.device)
    for _ in range(T):
        s1, mem1 = lif_step(a1, mem1, thr1, 1.0)
        p1 = F.avg_pool2d(s1, (1, 4))
        B2, C2, H2, T2 = p1.shape
        pad2 = p1.new_zeros(B2, C2, H2, pad2_amt)
        b = pw(sdw(torch.cat([pad2, p1, pad2], dim=3)))
        s2, mem2 = lif_step(b, mem2, thr2, 1.0)
        p2 = F.avg_pool2d(s2, (1, 8))
        out = out + fc(p2.view(B, -1, 1, 1)).view(B, n_classes)
    return out / T


def max_pre(x_calib, conv1, dw, sdw, pw, pad1_amt, pad2_amt):
    """当前权重下 act1/act2 前激活的最大正值（用 relu 近似层间连续传递）。"""
    m1 = m2 = 0.0
    for i in range(0, len(x_calib), 64):
        x = x_calib[i:i + 64]
        B, C, H, W = x.shape
        pad1 = x.new_zeros(B, C, H, pad1_amt)
        a1 = dw(conv1(torch.cat([pad1, x, pad1], dim=3)))
        m1 = max(m1, float(a1.clamp(min=0).max()))
        p1 = F.avg_pool2d(torch.relu(a1), (1, 4))
        B2, C2, H2, T2 = p1.shape
        pad2 = p1.new_zeros(B2, C2, H2, pad2_amt)
        b = pw(sdw(torch.cat([pad2, p1, pad2], dim=3)))
        m2 = max(m2, float(b.clamp(min=0).max()))
    return m1, m2


def acc(logits, ys):
    return (logits.argmax(1) == ys).float().mean().item()


def main() -> int:
    dev = torch.device("cpu")  # 小模型 + 一次性实验，CPU 足够，规避 MPS 折叠权重设备不一致
    print(f"device = {dev}")

    ann = LIFEEGNet(n_channels=8, n_classes=2, activation="relu")
    ck = torch.load(ALGO_DIR / "relu_proxy.pth", map_location="cpu", weights_only=False)
    ann.load_state_dict(ck["state_dict"])
    ann.eval()

    d = np.load(DATA / "eval.npz")
    N = 300
    X = torch.from_numpy(d["X"][:N].astype(np.float32)).to(dev)
    y = torch.from_numpy(d["y"][:N].astype(np.int64)).to(dev)
    calib = X[:200]
    pad1_amt, pad2_amt = ann.pad1_amt, ann.pad2_amt

    with torch.no_grad():
        ann_logits = ann(X)
    print(f"ANN(ReLU 代理) acc@{N}: {acc(ann_logits, y)*100:.2f}%")

    # ---- naive（thr = 全局最大，无归一化）----
    conv1, dw, sdw, pw, fc = _fold(ann)
    m1, m2 = max_pre(calib, conv1, dw, sdw, pw, pad1_amt, pad2_amt)
    print(f"naive thr1={m1:.4f}, thr2={m2:.4f}")
    for T in (16, 64, 128):
        t0 = time.time()
        with torch.no_grad():
            lg = snn_logits(X, conv1, dw, sdw, pw, fc, m1, m2, T, pad1_amt, pad2_amt)
        print(f"[naive ] T={T:3d}: {acc(lg, y)*100:.2f}%  ({time.time()-t0:.1f}s)")

    # ---- 数据驱动归一化（thr=1.0）----
    conv1, dw, sdw, pw, fc = _fold(ann)
    m1, _ = max_pre(calib, conv1, dw, sdw, pw, pad1_amt, pad2_amt)
    with torch.no_grad():
        dw.weight.data /= m1
        dw.bias.data /= m1
    m2 = None
    _, m2 = max_pre(calib, conv1, dw, sdw, pw, pad1_amt, pad2_amt)
    with torch.no_grad():
        pw.weight.data /= m2
        pw.bias.data /= m2
    print(f"归一化 λ1={m1:.4f}, λ2={m2:.4f} → thr1=thr2=1.0")
    for T in (16, 64, 128, 256):
        t0 = time.time()
        with torch.no_grad():
            lg = snn_logits(X, conv1, dw, sdw, pw, fc, 1.0, 1.0, T, pad1_amt, pad2_amt)
        print(f"[归一化] T={T:3d}: {acc(lg, y)*100:.2f}%  ({time.time()-t0:.1f}s)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
