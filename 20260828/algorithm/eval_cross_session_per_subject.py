#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_cross_session_per_subject.py — cross-session 基线模型逐被试精度

用 eeg_tcnet_4class_lif_hbc_aug_s42.pth（cross-session 63.58% 基线）对
data_4class_by_subject/A{nn}/eval.npz 逐被试推理，得到「开箱即用」模式下
每人 E 会话的精度（288 试次/人），供 within-subject 对标表使用。

用法：
  env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin \\
      /opt/miniconda3/envs/openmmlab/bin/python eval_cross_session_per_subject.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ALGO_DIR = Path(__file__).resolve().parent
DATA_DIR = Path("/Users/heyixuan/BCI/luckfox-deploy/data")
SUBJ_DATA = ALGO_DIR / "data_4class_by_subject"
OUT_DIR = ALGO_DIR / "within_subject"

sys.path.insert(0, str(ALGO_DIR))
sys.path.insert(0, str(DATA_DIR))

from dataset import EEGDataset          # noqa: E402
from eeg_tcnet_lif import EEGTCNetLIF   # noqa: E402

CKPT = ALGO_DIR / "eeg_tcnet_4class_lif_hbc_aug_s42.pth"


def main() -> int:
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    model_args = ck["model_args"]
    lif_hp = ck.get("lif_hparams", {})
    model = EEGTCNetLIF(**model_args, **lif_hp)
    model.load_state_dict(ck["state_dict"], strict=True)
    model.eval()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)

    results = []
    for subj in range(1, 10):
        ds = EEGDataset(SUBJ_DATA / f"A{subj:02d}" / "eval.npz")
        loader = DataLoader(ds, batch_size=32, shuffle=False)
        correct = total = 0
        with torch.no_grad():
            for xb, yb in loader:
                logits = model(xb.to(device))
                correct += (logits.argmax(1) == yb.to(device)).sum().item()
                total += xb.size(0)
        acc = correct / total
        results.append({"subject": f"A{subj:02d}", "acc": round(acc * 100, 2),
                        "correct": correct, "total": total})
        print(f"  A{subj:02d}: {acc*100:.2f}%  ({correct}/{total})")

    accs = [r["acc"] for r in results]
    print(f"\n平均: {np.mean(accs):.2f}% ± {np.std(accs):.2f}%  "
          f"(min {min(accs):.2f} / max {max(accs):.2f})")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "cross_session_per_subject.json").write_text(
        json.dumps({"model": CKPT.name, "results": results,
                    "mean": round(float(np.mean(accs)), 2),
                    "std": round(float(np.std(accs)), 2)}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(f"✓ 已写入 within_subject/cross_session_per_subject.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
