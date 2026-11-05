#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_loso_dropout.py — LOSO 跨 subject 零校准下的电极脱落鲁棒性（SNN vs ANN）

方向核心追问（比高斯噪声更贴近康复场景）：电极脱落/接触不良时，SNN 是否更鲁棒？
康复/可穿戴场景下电极易脱落（出汗、运动、佩戴移位），这是「零校准 + 端侧」的核心痛点。

协议：LOSO 9 折，test 时随机把 k 个通道置零（模拟电极脱落），
  k ∈ {0, 1, 3, 5, 8}（22 通道里脱落 k 个），对比 EEGNet(ANN) vs signed_lif(SNN) 精度下降。

判据：SNN 在电极脱落下的精度下降 < ANN（SNN 更鲁棒）。

用法：
  env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin \
      /opt/miniconda3/envs/openmmlab/bin/python train_loso_dropout.py --folds 3 --epochs 30
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ALGO_DIR = Path(__file__).resolve().parent
BY_SUBJECT = ALGO_DIR / "data_4class_by_subject"
MODEL_DIR = Path("/Users/heyixuan/BCI/luckfox-deploy/model")
SUBJECTS = [f"A{i:02d}" for i in range(1, 10)]
N_CH = 22

sys.path.insert(0, str(ALGO_DIR))
sys.path.insert(0, str(MODEL_DIR))

from tta import predict_frozen, accuracy  # noqa: E402
from train_loso import _pick_device, _build, _train, _load_subject  # noqa: E402

DROPS = [0, 1, 3, 5, 8]


def drop_channels(X: np.ndarray, k: int, seed: int) -> np.ndarray:
    """随机把 k 个通道置零（模拟电极脱落）。X: (n,1,22,500)。"""
    if k <= 0:
        return X
    rng = np.random.RandomState(seed)
    ch = rng.choice(N_CH, size=k, replace=False)
    X = X.copy()
    X[:, :, ch, :] = 0.0
    return X


def main() -> int:
    ap = argparse.ArgumentParser(description="LOSO 电极脱落鲁棒性（SNN vs ANN）")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--folds", type=int, default=9)
    ap.add_argument("--no-cuda", action="store_true")
    args = ap.parse_args()

    device = _pick_device(args.no_cuda)
    data = {s: _load_subject(s) for s in SUBJECTS}
    print(f"设备: {device} | LOSO {args.folds} 折 | epochs={args.epochs} | 脱落通道数={DROPS}")

    accs = {m: {k: [] for k in DROPS} for m in ("eegnet", "signed_lif")}
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

        eeg = _build("eegnet").to(device)
        eeg_state, _ = _train(eeg, Xtr, ytr, device, args.epochs, args.lr, args.batch, args.seed)
        eeg.load_state_dict(eeg_state)
        snn = _build("signed_lif").to(device)
        snn.load_state_dict(eeg_state, strict=True)
        snn_state, _ = _train(snn, Xtr, ytr, device, args.epochs, args.lr, args.batch, args.seed)
        snn.load_state_dict(snn_state)

        for model_name, m in (("eegnet", eeg), ("signed_lif", snn)):
            for k in DROPS:
                Xte_d = drop_channels(Xte, k, args.seed * 100 + test_idx * 10 + k)
                preds = predict_frozen(m, Xte_d, device)
                accs[model_name][k].append(accuracy(preds, yte))
            print(f"  [{test_subj}] {model_name:>10}: "
                  f"{' | '.join(f'脱落{k}:{accs[model_name][k][-1]*100:.1f}%' for k in DROPS)}")

    report = {}
    lines = ["LOSO 跨 subject 电极脱落鲁棒性（SNN vs ANN）", "=" * 60,
             f"脱落通道数 ∈ {DROPS} | {args.folds} 折 | 通道置零模拟", "-" * 60,
             f"{'脱落':>6} | {'EEGNet(ANN)':>15} | {'signed_lif(SNN)':>15} | {'SNN-ANN':>8}"]
    clean = {}
    for k in DROPS:
        row = {}
        for m in ("eegnet", "signed_lif"):
            a = np.array(accs[m][k]) * 100
            row[m] = round(float(a.mean()), 2)
            if k == 0:
                clean[m] = float(a.mean())
        gap = row["signed_lif"] - row["eegnet"]
        report[str(k)] = dict(eegnet=row["eegnet"], signed_lif=row["signed_lif"], gap=round(gap, 2))
        lines.append(f"{k:>6} | {row['eegnet']:>14.2f}% | {row['signed_lif']:>14.2f}% | {gap:>+8.2f}")
    drop_ann = clean["eegnet"] - float(np.array(accs["eegnet"][8]).mean() * 100)
    drop_snn = clean["signed_lif"] - float(np.array(accs["signed_lif"][8]).mean() * 100)
    lines += ["-" * 60,
              f"脱落 8 通道精度下降：ANN {drop_ann:.2f}pp vs SNN {drop_snn:.2f}pp "
              f"→ {'SNN 更鲁棒 ✓' if drop_snn < drop_ann else 'SNN 未更鲁棒 ✗'}"]
    (ALGO_DIR / "loso_dropout_results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    (ALGO_DIR / "loso_dropout_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines))
    print(f"\n✓ 报告已保存：loso_dropout_results.json / loso_dropout_report.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
