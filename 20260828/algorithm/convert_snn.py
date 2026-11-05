#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
convert_snn.py — ANN→SNN 转换 + 阈值标定 + 精度评估 + 权重导出（S1/S2/S5 步）

流程
----
  1. 加载 ReLU-EEGNet 代理（train_relu_proxy.py 产物 relu_proxy.pth）；
  2. build_snn（BN 折叠 + LIF 替换）+ calibrate_thresholds（在标定子集上逐层取最大正激活）；
  3. snn_infer 在 eval 全量上推理，报 SNN 准确率，与 ReLU 代理对比 → ANN→SNN 精度损失；
  4. 把折叠后 SNN 权重 + thr1/thr2 + 元信息存成 snn_weights.pth；
  5. 调 export_snn_weights.export 生成 C 头文件 board_demo/snn_weights.h。

诚实边界：SNN 精度**如实报**——若 ANN→SNN 转换损失大（>5pp），如实写，不编「成功」。

用法
----
  env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin \
      /opt/miniconda3/envs/openmmlab/bin/python convert_snn.py \
      --proxy relu_proxy.pth --timesteps 16 --n-classes 2

架构硬要求：n_classes 可配置（默认 2），不写死。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ALGO_DIR = Path(__file__).resolve().parent
BOARD_DEMO_DIR = ALGO_DIR.parent / "board_demo"
PROJECT = Path("/Users/heyixuan/BCI/luckfox-deploy")
PROCESSED = PROJECT / "data" / "processed"
DATA_DIR = PROJECT / "data"

sys.path.insert(0, str(ALGO_DIR))
sys.path.insert(0, str(DATA_DIR))

from lif_eegnet import LIFEEGNet          # noqa: E402
from ann2snn import (build_snn, calibrate_thresholds,   # noqa: E402
                     convert_and_calibrate, snn_infer)
from export_snn_weights import export, export_from_state  # noqa: E402


def _load_proxy(path: Path, n_classes: int) -> LIFEEGNet:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    ma = ck.get("model_args", {})
    n_classes = n_classes or ma.get("n_classes", 2)
    model = LIFEEGNet(n_channels=ma.get("n_channels", 8),
                      n_classes=n_classes,
                      input_length=ma.get("input_length", 500),
                      F1=ma.get("F1", 8), D=ma.get("D", 2), F2=ma.get("F2", 16),
                      kernel_length=ma.get("kernel_length", 63),
                      dropout=ma.get("dropout", 0.5),
                      activation="relu")
    model.load_state_dict(ck["state_dict"], strict=True)
    model.eval()
    return model


def _eval_ann(ann: LIFEEGNet, loader) -> float:
    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb in loader:
            logits = ann(xb)
            correct += (logits.argmax(1) == yb).sum().item()
            total += xb.size(0)
    return correct / total


def _eval_snn(snn, loader, timesteps: int):
    """SNN 全量推理，返回 (acc, s1_rate, s2_rate)。"""
    from ann2snn import lif_step
    correct, total = 0, 0
    s1_sum = s2_sum = 0.0
    n_spk_samples = 0
    snn.eval()
    with torch.no_grad():
        for xb, yb in loader:
            logits = snn_infer(snn, xb, timesteps=timesteps)
            correct += (logits.argmax(1) == yb).sum().item()
            total += xb.size(0)
            # 发放率统计（透明度）
            B, C, H, T = xb.shape
            mem1 = mem2 = None
            for _ in range(timesteps):
                pad1 = xb.new_zeros(B, C, H, snn.pad1_amt)
                xp = torch.cat([pad1, xb, pad1], dim=3)
                a = snn.depthwise(snn.conv1(xp))
                s1, mem1 = lif_step(a, mem1, snn.thr1, 1.0)
                p1 = snn.pool1(s1)
                B2, C2, H2, T2 = p1.shape
                pad2 = p1.new_zeros(B2, C2, H2, snn.pad2_amt)
                p1p = torch.cat([pad2, p1, pad2], dim=3)
                b = snn.pointwise(snn.sep_depthwise(p1p))
                s2, mem2 = lif_step(b, mem2, snn.thr2, 1.0)
                s1_sum += float(s1.mean())
                s2_sum += float(s2.mean())
            n_spk_samples += 1
    n_steps = n_spk_samples * timesteps
    return correct / total, s1_sum / n_steps, s2_sum / n_steps


def main() -> int:
    parser = argparse.ArgumentParser(description="ANN→SNN 转换 + 阈值标定 + 精度评估")
    parser.add_argument("--proxy", type=str, default=str(ALGO_DIR / "relu_proxy.pth"))
    parser.add_argument("--timesteps", type=int, default=16)
    parser.add_argument("--n-classes", type=int, default=2)
    parser.add_argument("--calib-samples", type=int, default=200,
                        help="阈值标定用的样本数（eval 前 N 个）")
    parser.add_argument("--out", type=str, default=str(ALGO_DIR / "snn_weights.pth"))
    parser.add_argument("--header", type=str,
                        default=str(BOARD_DEMO_DIR / "snn_weights.h"))
    args = parser.parse_args()

    from dataset import EEGDataset  # noqa: E402

    eval_ds = EEGDataset(PROCESSED / "eval.npz")
    n_classes = args.n_classes or eval_ds.n_classes
    eval_loader = DataLoader(eval_ds, batch_size=32, shuffle=False)

    # ---- 1. 加载 ReLU 代理 + 报 ANN 基线 ----
    ann = _load_proxy(Path(args.proxy), n_classes)
    ann_acc = _eval_ann(ann, eval_loader)
    print("=" * 72)
    print("ANN→SNN 转换 + 阈值标定 + 精度评估")
    print("=" * 72)
    print(f"[ANN] ReLU-EEGNet 代理（eval 全量）准确率 : {ann_acc*100:.2f}%")

    # ---- 2. 阈值标定（取 eval 前 calib-samples 个）----
    calib_x = torch.from_numpy(eval_ds.X[:args.calib_samples].astype(np.float32))
    thr1, thr2 = calibrate_thresholds(ann, [calib_x[i:i+1] for i in range(len(calib_x))])
    snn = build_snn(ann)
    snn.thr1, snn.thr2 = thr1, thr2
    print(f"[标定] thr1={thr1:.4f}, thr2={thr2:.4f}（标定样本 {args.calib_samples} 个）")

    # ---- 3. SNN 全量推理 + 精度 + 发放率 ----
    snn_acc, s1_rate, s2_rate = _eval_snn(snn, eval_loader, args.timesteps)
    loss_pp = (ann_acc - snn_acc) * 100
    print(f"[SNN] 转换后准确率（timesteps={args.timesteps}） : {snn_acc*100:.2f}%")
    print(f"[损失] ANN→SNN 精度损失 : {loss_pp:+.2f} pp")
    print(f"[发放率] act1 平均 {s1_rate*100:.2f}% | act2 平均 {s2_rate*100:.2f}%")
    if loss_pp > 5.0:
        print("⚠️  转换损失 >5pp，如实记录（不视为「成功」，见报告）。")

    # ---- 4. 保存折叠权重 + 元信息 ----
    sd = snn.state_dict()
    state = {
        "conv1_weight": sd["conv1.weight"], "conv1_bias": sd["conv1.bias"],
        "depthwise_weight": sd["depthwise.weight"], "depthwise_bias": sd["depthwise.bias"],
        "sep_depthwise_weight": sd["sep_depthwise.weight"],
        "pointwise_weight": sd["pointwise.weight"], "pointwise_bias": sd["pointwise.bias"],
        "fc_weight": sd["fc.weight"], "fc_bias": sd["fc.bias"],
        "thr1": float(thr1), "thr2": float(thr2),
        "n_classes": int(n_classes), "timesteps": int(args.timesteps),
        "n_channels": 8, "input_length": 500, "F1": 8, "D": 2, "F2": 16,
        "kernel_length": 63,
        "ann_acc": float(ann_acc), "snn_acc": float(snn_acc),
    }
    torch.save(state, Path(args.out))
    print(f"[保存] 折叠后 SNN 权重 → {args.out}")

    # ---- 5. 导出 C 头文件 ----
    export_from_state(state, Path(args.header))
    print(f"[导出] C 头文件 → {args.header}")

    print("\n✓ 转换完成。下一步：zig 编译 lif_eegnet_infer.c → 板上验证 C-SNN vs PyTorch-SNN。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
