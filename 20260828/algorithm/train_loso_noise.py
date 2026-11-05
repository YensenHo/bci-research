#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_loso_noise.py — LOSO 跨 subject 零校准下的噪声鲁棒性对比（SNN vs ANN）

方向核心追问：TTA（发放率稳态）证伪后，SNN「真能赢的轴」是什么？
本脚本验证最有希望的候选：**噪声/电极脱落鲁棒性**——
  SNN 的稀疏 + 膜电位时序积分（低通）理论上对噪声更鲁棒。

协议（复用 LOSO）：
  9 折，train = 8 subject T 会话，test = 1 subject E 会话（零校准）。
  测试时注入高斯噪声 σ ∈ {0, 0.1, 0.2, 0.5, 1.0}（模拟电极噪声/伪迹/放大器噪声），
  对比 EEGNet（ANN）与 signed_lif（SNN）的精度下降曲线。

判据：SNN 在高噪声下的精度下降 < ANN（SNN 更鲁棒），即「噪声鲁棒性」是 SNN 真能赢的轴。

用法：
  env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin \
      /opt/miniconda3/envs/openmmlab/bin/python train_loso_noise.py --folds 3 --epochs 30
  env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin \
      /opt/miniconda3/envs/openmmlab/bin/python train_loso_noise.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ALGO_DIR = Path(__file__).resolve().parent
BY_SUBJECT = ALGO_DIR / "data_4class_by_subject"
MODEL_DIR = Path("/Users/heyixuan/BCI/luckfox-deploy/model")
SUBJECTS = [f"A{i:02d}" for i in range(1, 10)]

sys.path.insert(0, str(ALGO_DIR))
sys.path.insert(0, str(MODEL_DIR))

from eegnet import EEGNet            # noqa: E402
from lif_eegnet import LIFEEGNet     # noqa: E402
from tta import predict_frozen, accuracy  # noqa: E402
from train_loso import _pick_device, _build, _train, _load_subject  # noqa: E402

SIGMAS = [0.0, 0.1, 0.2, 0.5, 1.0]


def main() -> int:
    ap = argparse.ArgumentParser(description="LOSO 跨 subject 噪声鲁棒性（SNN vs ANN）")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--folds", type=int, default=9)
    ap.add_argument("--no-cuda", action="store_true")
    args = ap.parse_args()

    device = _pick_device(args.no_cuda)
    data = {s: _load_subject(s) for s in SUBJECTS}
    print(f"设备: {device} | LOSO {args.folds} 折 | epochs={args.epochs} | σ={SIGMAS}")

    # 累积：{model: {sigma: [accs]}}
    accs = {m: {s: [] for s in SIGMAS} for m in ("eegnet", "signed_lif")}
    for test_idx, test_subj in enumerate(SUBJECTS):
        if test_idx >= args.folds:
            break
        Xtr_list, ytr_list = [], []
        for s in SUBJECTS:
            if s == test_subj:
                continue
            Xtr_list.append(data[s][0]); ytr_list.append(data[s][1])
        Xtr = np.concatenate(Xtr_list); ytr = np.concatenate(ytr_list)
        Xte, yte = data[test_subj][2], data[test_subj][3]

        # 先训 eegnet（一次，作为 ANN 基线 + SNN 热启动源，公平对比）
        eeg = _build("eegnet").to(device)
        eeg_state, _ = _train(eeg, Xtr, ytr, device, args.epochs, args.lr, args.batch, args.seed)
        eeg.load_state_dict(eeg_state)
        snn = _build("signed_lif").to(device)
        snn.load_state_dict(eeg_state, strict=True)
        snn_state, _ = _train(snn, Xtr, ytr, device, args.epochs, args.lr, args.batch, args.seed)
        snn.load_state_dict(snn_state)

        for model_name, m in (("eegnet", eeg), ("signed_lif", snn)):
            for sigma in SIGMAS:
                if sigma == 0.0:
                    Xte_n = Xte
                else:
                    rng = np.random.RandomState(args.seed * 1000 + test_idx)
                    Xte_n = Xte + (sigma * rng.randn(*Xte.shape)).astype(np.float32)
                preds = predict_frozen(m, Xte_n, device)
                accs[model_name][sigma].append(accuracy(preds, yte))
            print(f"  [{test_subj}] {model_name:>10}: "
                  f"{' | '.join(f'σ={s}:{accs[model_name][s][-1]*100:.1f}%' for s in SIGMAS)}")

    # 报告：每个模型每个 σ 的平均精度 + 相对 σ=0 的下降
    report = {}
    lines = ["LOSO 跨 subject 噪声鲁棒性（SNN vs ANN）", "=" * 60,
             f"σ ∈ {SIGMAS} | {args.folds} 折 | 高斯噪声注入测试", "-" * 60,
             f"{'σ':>6} | {'EEGNet(ANN)':>16} | {'signed_lif(SNN)':>16} | {'SNN-ANN':>8}"]
    clean = {}
    for sigma in SIGMAS:
        row = {}
        for m in ("eegnet", "signed_lif"):
            a = np.array(accs[m][sigma]) * 100
            row[m] = round(float(a.mean()), 2)
            if sigma == 0.0:
                clean[m] = float(a.mean())
        gap = row["signed_lif"] - row["eegnet"]
        report[str(sigma)] = dict(eegnet=row["eegnet"], signed_lif=row["signed_lif"], gap=round(gap, 2))
        lines.append(f"{sigma:>6} | {row['eegnet']:>15.2f}% | {row['signed_lif']:>15.2f}% | {gap:>+8.2f}")
    # 高噪声下的下降（σ=1.0 vs σ=0）
    drop_ann = clean["eegnet"] - float(np.array(accs["eegnet"][1.0]).mean() * 100)
    drop_snn = clean["signed_lif"] - float(np.array(accs["signed_lif"][1.0]).mean() * 100)
    lines += ["-" * 60,
              f"σ=1.0 精度下降：ANN {drop_ann:.2f}pp vs SNN {drop_snn:.2f}pp "
              f"→ {'SNN 更鲁棒 ✓' if drop_snn < drop_ann else 'SNN 未更鲁棒 ✗'}"]
    (ALGO_DIR / "loso_noise_results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    (ALGO_DIR / "loso_noise_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines))
    print(f"\n✓ 报告已保存：loso_noise_results.json / loso_noise_report.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
