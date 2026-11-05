#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_loso_eegmmidb.py — EEGMMIDB 跨 subject 零校准 LOSO 评估（SNN vs ANN）

第二个数据集验证：EEGMMIDB（30 subject × 90 试次，21ch/160Hz/4类，320 样本）。
协议与 BCIC 2a 的 train_loso.py 一致：LOSO，train = 29 subject，test = 1 subject（零校准）。

用法：
  env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin \
      /opt/miniconda3/envs/openmmlab/bin/python train_loso_eegmmidb.py --folds 3 --epochs 30
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ALGO_DIR = Path(__file__).resolve().parent
MODEL_DIR = Path("/Users/heyixuan/BCI/luckfox-deploy/model")
DATA_DIR = ALGO_DIR / "eegmmidb" / "data_4class_by_subject"

sys.path.insert(0, str(ALGO_DIR))
sys.path.insert(0, str(MODEL_DIR))

from eegnet import EEGNet            # noqa: E402
from lif_eegnet import LIFEEGNet     # noqa: E402
from train_loso import _pick_device, _train  # noqa: E402

N_CH, N_CLS, IN_LEN = 21, 4, 320
SUBJECTS = [f"S{i:03d}" for i in range(1, 31)]


def _build(model: str) -> nn.Module:
    common = dict(n_channels=N_CH, n_classes=N_CLS, input_length=IN_LEN,
                  F1=8, D=2, F2=16, kernel_length=40, dropout=0.5)  # kernel=fs/4=40
    if model == "eegnet":
        return EEGNet(activation="elu", **common)
    return LIFEEGNet(activation="signed_lif", **common)


def _load(subj: str) -> tuple[np.ndarray, np.ndarray]:
    d = np.load(DATA_DIR / subj / "data.npz")
    return d["X"].astype(np.float32), d["y"].astype(np.int64)


def main() -> int:
    ap = argparse.ArgumentParser(description="EEGMMIDB 跨 subject 零校准 LOSO")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--folds", type=int, default=30)
    ap.add_argument("--no-cuda", action="store_true")
    args = ap.parse_args()

    device = _pick_device(args.no_cuda)
    data = {s: _load(s) for s in SUBJECTS}
    print(f"设备: {device} | EEGMMIDB LOSO {args.folds} 折 | 21ch/320/4类 | epochs={args.epochs}")

    accs = {"eegnet": [], "signed_lif": []}
    per_subject = {m: {} for m in accs}
    for test_idx, test_subj in enumerate(SUBJECTS):
        if test_idx >= args.folds:
            break
        Xtr_list, ytr_list = [], []
        for s in SUBJECTS:
            if s == test_subj:
                continue
            Xtr_list.append(data[s][0]); ytr_list.append(data[s][1])
        Xtr = np.concatenate(Xtr_list); ytr = np.concatenate(ytr_list)
        Xte, yte = data[test_subj]

        eeg = _build("eegnet").to(device)
        eeg_state, _ = _train(eeg, Xtr, ytr, device, args.epochs, args.lr, args.batch, args.seed)
        eeg.load_state_dict(eeg_state)
        snn = _build("signed_lif").to(device)
        snn.load_state_dict(eeg_state, strict=True)
        snn_state, _ = _train(snn, Xtr, ytr, device, args.epochs, args.lr, args.batch, args.seed)
        snn.load_state_dict(snn_state)

        for model_name, m in (("eegnet", eeg), ("signed_lif", snn)):
            m.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for i in range(0, len(Xte), args.batch):
                    xb = torch.from_numpy(Xte[i:i + args.batch]).to(device)
                    yb = torch.from_numpy(yte[i:i + args.batch]).to(device)
                    correct += (m(xb).argmax(1) == yb).sum().item()
                    total += yb.size(0)
            acc = correct / total
            accs[model_name].append(acc)
            per_subject[model_name][test_subj] = round(acc * 100, 2)
        print(f"  [{test_subj}] eegnet={accs['eegnet'][-1]*100:5.2f}% "
              f"signed_lif={accs['signed_lif'][-1]*100:5.2f}%")

    report = {}
    lines = ["EEGMMIDB 跨 subject 零校准 LOSO（SNN vs ANN）", "=" * 56,
             f"21ch/320/4类 | {args.folds} 折 | epochs={args.epochs}", "-" * 56]
    for m in ("eegnet", "signed_lif"):
        arr = np.array(accs[m]) * 100
        report[m] = dict(mean=round(float(arr.mean()), 2), std=round(float(arr.std()), 2))
        lines.append(f"{m:>10}: 平均 {arr.mean():.2f}% ± {arr.std():.2f}%")
    gap = report["signed_lif"]["mean"] - report["eegnet"]["mean"]
    lines.append(f"SNN-ANN 差距: {gap:+.2f}pp")
    report["gap_pp"] = round(gap, 2)
    report["per_subject"] = per_subject
    (ALGO_DIR / "loso_eegmmidb_results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    (ALGO_DIR / "loso_eegmmidb_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines))
    print(f"\n✓ 报告已保存：loso_eegmmidb_results.json / loso_eegmmidb_report.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
