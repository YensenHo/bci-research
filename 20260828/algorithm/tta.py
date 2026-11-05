#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tta.py — Spike-based Test-Time Adaptation（发放率目标追踪）

方向核心贡献：把 SNN 的「发放率」升格为「测试时适应的物理信号」。

设计（v2，修正 v1 的「区间保护」缺陷）：
  v1 的区间保护（发放率偏离 [r_min, r_max] 才调阈值）是「被动保护」，训练后的
  signed_lif 发放率已在健康区间，从不触发 → TTA 恒为 no-op（+0.00pp，实测证伪）。
  v2 改为「发放率目标追踪」：训练后记录每层平均总发放率 r_target，测试时逐样本
  让当前发放率收敛到 r_target（threshold += η·(r - r_target)）。这是「主动适应」，
  能捕获跨 subject 的温和幅度漂移（电极阻抗/信号强弱 → 发放率整体偏移）。

信号选择：用「总发放率」（signed_lif 的 pos_rate + neg_rate ∈ [0,2]）而非净发放率，
因为幅度漂移会让正负通道同比例变化，净发放率（pos-neg）可能抵消，总发放率才敏感。

判据：TTA 相对 frozen 的跨 subject 精度提升 ≥2pp，且消融证明发放率追踪是提升来源。
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import torch

ALGO_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ALGO_DIR))
sys.path.insert(0, str(Path("/Users/heyixuan/BCI/luckfox-deploy/model")))

from lif_eegnet import LIFEEGNet, LIFActivation, SignedLIFActivation  # noqa: E402

LAYER_NAMES = ("act1", "act2")


def _layer_total_rate(layer) -> float:
    """总发放率（对幅度漂移敏感）。signed_lif 用 pos+neg ∈ [0,2]，lif 用 firing_rate ∈ [0,1]。"""
    if isinstance(layer, SignedLIFActivation):
        p = getattr(layer, "last_pos_rate", 0.0)
        n = getattr(layer, "last_neg_rate", 0.0)
        return float(p + n)
    r = getattr(layer, "last_firing_rate", 0.5)
    return float(r)


def record_rates(model, X, device, batch=64) -> dict:
    """训练后记录每层平均总发放率（作为 TTA 追踪目标 r_target）。"""
    model.eval()
    sums = {n: 0.0 for n in LAYER_NAMES}
    total = 0
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.from_numpy(X[i:i + batch]).to(device)
            model(xb)
            for name in LAYER_NAMES:
                layer = getattr(model, name, None)
                if isinstance(layer, (LIFActivation, SignedLIFActivation)):
                    sums[name] += _layer_total_rate(layer) * xb.size(0)
            total += xb.size(0)
    return {n: (sums[n] / total if total else 0.0) for n in LAYER_NAMES}


def predict_tta(model, X, device, r_target: dict, eta=0.02, thr_floor=0.05) -> np.ndarray:
    """发放率目标追踪 TTA：逐样本前向，让发放率收敛到 r_target。返回 (n,) int64 预测。"""
    model.eval()
    preds = np.empty(len(X), dtype=np.int64)
    with torch.no_grad():
        for i in range(len(X)):
            xb = torch.from_numpy(X[i:i + 1]).to(device)
            logits = model(xb)
            preds[i] = int(logits.argmax(1).item())
            for name in LAYER_NAMES:
                layer = getattr(model, name, None)
                if not isinstance(layer, (LIFActivation, SignedLIFActivation)):
                    continue
                r = _layer_total_rate(layer)
                tgt = r_target.get(name, 0.2)
                layer.threshold = max(thr_floor, layer.threshold + eta * (r - tgt))
    return preds


def predict_frozen(model, X, device, batch=64) -> np.ndarray:
    """frozen 前向（无 TTA），返回 (n,) int64 预测标签。"""
    model.eval()
    preds = np.empty(len(X), dtype=np.int64)
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.from_numpy(X[i:i + batch]).to(device)
            preds[i:i + batch] = model(xb).argmax(1).cpu().numpy()
    return preds


def accuracy(preds: np.ndarray, y: np.ndarray) -> float:
    return float((preds == y).mean())


if __name__ == "__main__":
    # 冒烟：分布漂移场景下，验证目标追踪阈值方向正确
    # 训练分布 = randn（幅度 1），测试分布 = randn*5（幅度放大 5 倍，发放率更高）
    # 目标追踪应提阈值（抑制过度发放）
    model = LIFEEGNet(activation="signed_lif", n_channels=22, n_classes=4,
                      input_length=500, F1=8, D=2, F2=16, kernel_length=63, dropout=0.5)
    model.eval()
    dev = torch.device("cpu")
    r_target = record_rates(model, torch.randn(128, 1, 22, 500).numpy(), dev)
    thr_before = (model.act1.threshold, model.act2.threshold)
    # 幅度放大 5 倍的测试分布 → 发放率更高 → 应提阈值
    _ = predict_tta(model, (torch.randn(32, 1, 22, 500) * 5.0).numpy(), dev, r_target, eta=0.02)
    thr_after = (model.act1.threshold, model.act2.threshold)
    print(f"冒烟：r_target act1={r_target['act1']:.3f} act2={r_target['act2']:.3f}")
    print(f"  阈值 act1 {thr_before[0]:.3f}→{thr_after[0]:.3f}  act2 {thr_before[1]:.3f}→{thr_after[1]:.3f}")
    assert thr_after[0] > thr_before[0], "幅度放大应提阈值（抑制过度发放）"
    print("✓ tta.py v2 自检通过（分布漂移下目标追踪方向正确：提阈值）")
