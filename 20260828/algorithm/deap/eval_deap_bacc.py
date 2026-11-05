#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""eval_deap_bacc.py — DEAP 已训练 checkpoint 的详细评估（每类准确率 + 平衡准确率）

用法：
  env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin \\
      /opt/miniconda3/envs/openmmlab/bin/python eval_deap_bacc.py deap_2class_*.pth
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ALGO_DIR = Path("/Users/heyixuan/BCI/20260828/algorithm")
BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(ALGO_DIR))
from eeg_tcnet_lif import EEGTCNetLIF  # noqa: E402


def main() -> int:
    with np.load(BASE / "data" / "eval.npz") as d:
        X = np.asarray(d["X"], np.float32)
        y = np.asarray(d["y"], np.int64)
    print(f"eval 集: X{X.shape}, 高={int((y==1).sum())} 低={int((y==0).sum())}")
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    ck_paths = sorted(BASE.glob("deap_2class_*.pth"))
    for ck_path in ck_paths:
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        act = ck.get("activation", "relu")
        lif = ck.get("lif_hparams", {}) if act == "lif" else {}
        model = EEGTCNetLIF(n_channels=32, n_classes=2, input_length=640,
                            activation=act, **lif)
        model.load_state_dict(ck["state_dict"], strict=True)
        model.to(dev).eval()
        with torch.no_grad():
            logits = torch.empty(0, 2)
            for i in range(0, len(X), 256):
                xb = torch.from_numpy(X[i:i + 256]).to(dev)
                logits = torch.cat([logits.cpu(), model(xb).cpu()])
        pred = logits.argmax(1).numpy()
        acc = (pred == y).mean()
        hi_acc = (pred[y == 1] == 1).mean() if (y == 1).any() else float("nan")
        lo_acc = (pred[y == 0] == 0).mean() if (y == 0).any() else float("nan")
        bacc = (hi_acc + lo_acc) / 2
        print(f"\n{ck_path.name}  [{act}]")
        print(f"  总体 acc : {acc*100:6.2f}%  | 高类 recall {hi_acc*100:6.2f}% 低类 recall {lo_acc*100:6.2f}%")
        print(f"  平衡 acc : {bacc*100:6.2f}%（chance 50%）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
