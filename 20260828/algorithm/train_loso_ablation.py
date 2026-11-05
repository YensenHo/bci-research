#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_loso_ablation.py — 消融：负半轴保留 + 时序积分对「零代价复现」的贡献

方向机制解释：SNN 为何能零代价复现 ANN？
  - 负半轴消融：lif（丢负半轴，等效 ReLU 截断）vs signed_lif（补负半轴）vs eegnet（ELU）
    预期 lif 比 ELU 低（负半轴代价），signed_lif 追平（补回负半轴 = 零代价关键）
  - 时序积分消融：signed_lif 的 timesteps T ∈ {1, 4, 8, 16}（时序积分的必要性）

协议：BCIC 2a LOSO 9 折，train = 8 subject T，test = 1 subject E（零校准）。

用法：
  env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin \
      /opt/miniconda3/envs/openmmlab/bin/python train_loso_ablation.py --folds 3 --epochs 30
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ALGO_DIR = Path(__file__).resolve().parent
MODEL_DIR = Path("/Users/heyixuan/BCI/luckfox-deploy/model")
SUBJECTS = [f"A{i:02d}" for i in range(1, 10)]

sys.path.insert(0, str(ALGO_DIR))
sys.path.insert(0, str(MODEL_DIR))

from eegnet import EEGNet            # noqa: E402
from lif_eegnet import LIFEEGNet     # noqa: E402
from train_loso import _pick_device, _train, _load_subject  # noqa: E402

COMMON = dict(n_channels=22, n_classes=4, input_length=500,
              F1=8, D=2, F2=16, kernel_length=63, dropout=0.5)


def _build(model: str, timesteps: int = 8) -> torch.nn.Module:
    if model == "eegnet":
        return EEGNet(activation="elu", **COMMON)
    if model == "lif":
        return LIFEEGNet(activation="lif", timesteps=timesteps, **COMMON)
    return LIFEEGNet(activation="signed_lif", timesteps=timesteps, **COMMON)


def main() -> int:
    ap = argparse.ArgumentParser(description="消融：负半轴 + 时序积分")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--folds", type=int, default=9)
    ap.add_argument("--no-cuda", action="store_true")
    args = ap.parse_args()

    device = _pick_device(args.no_cuda)
    data = {s: _load_subject(s) for s in SUBJECTS}
    models = ["eegnet", "lif", "signed_lif"]
    ts_list = [1, 4, 8, 16]
    print(f"设备: {device} | LOSO {args.folds} 折 | 消融模型 {models} + signed_lif T∈{ts_list}")

    accs = {m: [] for m in models}
    t_accs = {t: [] for t in ts_list}
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

        # 训 eegnet（热启动源）
        eeg = _build("eegnet").to(device)
        eeg_state, _ = _train(eeg, Xtr, ytr, device, args.epochs, args.lr, args.batch, args.seed)
        eeg.load_state_dict(eeg_state)

        def eval_model(m):
            m.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for i in range(0, len(Xte), args.batch):
                    xb = torch.from_numpy(Xte[i:i + args.batch]).to(device)
                    yb = torch.from_numpy(yte[i:i + args.batch]).to(device)
                    correct += (m(xb).argmax(1) == yb).sum().item()
                    total += yb.size(0)
            return correct / total

        accs["eegnet"].append(eval_model(eeg))
        for model in ("lif", "signed_lif"):
            m = _build(model).to(device)
            m.load_state_dict(eeg_state, strict=True)
            st, _ = _train(m, Xtr, ytr, device, args.epochs, args.lr, args.batch, args.seed)
            m.load_state_dict(st)
            accs[model].append(eval_model(m))
        for t in ts_list:
            m = _build("signed_lif", timesteps=t).to(device)
            m.load_state_dict(eeg_state, strict=True)
            st, _ = _train(m, Xtr, ytr, device, args.epochs, args.lr, args.batch, args.seed)
            m.load_state_dict(st)
            t_accs[t].append(eval_model(m))
        print(f"  [{test_subj}] eegnet={accs['eegnet'][-1]*100:.1f}% "
              f"lif={accs['lif'][-1]*100:.1f}% signed_lif={accs['signed_lif'][-1]*100:.1f}% "
              + " | ".join(f"T{t}:{t_accs[t][-1]*100:.1f}%" for t in ts_list))

    lines = ["消融：负半轴保留 + 时序积分（BCIC 2a LOSO）", "=" * 56]
    report = {}
    for m in models:
        arr = np.array(accs[m]) * 100
        report[m] = dict(mean=round(float(arr.mean()), 2), std=round(float(arr.std()), 2))
        lines.append(f"{m:>10}: {arr.mean():.2f}% ± {arr.std():.2f}%")
    for t in ts_list:
        arr = np.array(t_accs[t]) * 100
        report[f"signed_lif_T{t}"] = dict(mean=round(float(arr.mean()), 2), std=round(float(arr.std()), 2))
        lines.append(f"signed_lif T={t:>2}: {arr.mean():.2f}% ± {arr.std():.2f}%")
    (ALGO_DIR / "loso_ablation_results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    (ALGO_DIR / "loso_ablation_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines))
    print("\n✓ 报告已保存：loso_ablation_results.json / loso_ablation_report.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
