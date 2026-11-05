#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_loso_tta.py — LOSO 跨 subject 零校准 + spike-based TTA 对比

方向核心贡献实验：验证 spike-based TTA（发放率稳态）能否让 frozen SNN 追平/超过 ANN。

协议（与 train_loso.py 完全一致）：
  LOSO：9 折，train = 8 subject T 会话，test = 1 subject E 会话（零校准）。

对比（同一 signed_lif 权重）：
  - frozen       : 直接前向（无 TTA）
  - + spike TTA  : 逐样本读发放率，homeostasis 调阈值（tta.predict_tta）

判据：TTA 相对 frozen 提升 ≥2pp，且逐层消融证明发放率稳态是提升来源。

用法：
  # 冒烟（1 折 + 少 epochs）
  env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin \
      /opt/miniconda3/envs/openmmlab/bin/python train_loso_tta.py --smoke
  # 全量（9 折）
  env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin \
      /opt/miniconda3/envs/openmmlab/bin/python train_loso_tta.py
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
from tta import predict_frozen, predict_tta, record_rates, accuracy  # noqa: E402
from train_loso import _pick_device, _build, _train, _load_subject  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="LOSO 跨 subject + spike-based TTA 对比")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eta", type=float, default=0.02, help="阈值追踪步长（目标追踪）")
    ap.add_argument("--folds", type=int, default=9, help="跑前 N 折（默认 9，用于快速验证）")
    ap.add_argument("--no-cuda", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        args.epochs = 2
        args.folds = 1
    device = _pick_device(args.no_cuda)
    data = {s: _load_subject(s) for s in SUBJECTS}
    print(f"设备: {device} | LOSO {len(SUBJECTS)} 折 | epochs={args.epochs} | "
          f"TTA 目标追踪 η={args.eta}")

    frozen_accs, tta_accs = [], []
    per_subject = {}
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

        # ① 训 EEGNet（热启动源）
        eegnet = _build("eegnet").to(device)
        state, _ = _train(eegnet, Xtr, ytr, device, args.epochs, args.lr, args.batch, args.seed)
        eegnet.load_state_dict(state)

        # ② 训 signed_lif（热启动自 eegnet）
        snn = _build("signed_lif").to(device)
        snn.load_state_dict(state, strict=True)  # LIF 激活层无参数，键零回归
        snn_state, _ = _train(snn, Xtr, ytr, device, args.epochs, args.lr, args.batch, args.seed)
        snn.load_state_dict(snn_state)

        # ③ frozen vs TTA（同一权重）
        t0 = time.time()
        frozen_pred = predict_frozen(snn, Xte, device)
        frozen_acc = accuracy(frozen_pred, yte)
        r_target = record_rates(snn, Xtr, device)  # 训练发放率 = 追踪目标
        tta_pred = predict_tta(snn, Xte, device, r_target, args.eta)
        tta_acc = accuracy(tta_pred, yte)
        frozen_accs.append(frozen_acc); tta_accs.append(tta_acc)
        per_subject[test_subj] = dict(frozen=round(frozen_acc * 100, 2),
                                      tta=round(tta_acc * 100, 2))
        print(f"  [{test_subj}] frozen={frozen_acc*100:6.2f}% → TTA={tta_acc*100:6.2f}% "
              f"({(tta_acc-frozen_acc)*100:+.2f}pp, {time.time()-t0:.0f}s)")

    # 报告
    frozen_arr = np.array(frozen_accs) * 100
    tta_arr = np.array(tta_accs) * 100
    gain = tta_arr - frozen_arr
    report = dict(config=dict(epochs=args.epochs, seed=args.seed, smoke=args.smoke,
                              eta=args.eta),
                  per_subject=per_subject,
                  frozen_mean=round(float(frozen_arr.mean()), 2),
                  frozen_std=round(float(frozen_arr.std()), 2),
                  tta_mean=round(float(tta_arr.mean()), 2),
                  tta_std=round(float(tta_arr.std()), 2),
                  gain_mean_pp=round(float(gain.mean()), 2))
    (ALGO_DIR / "loso_tta_results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    lines = ["LOSO 跨 subject 零校准 + spike-based TTA 对比", "=" * 56,
             f"frozen 平均 {frozen_arr.mean():.2f}% ± {frozen_arr.std():.2f}%",
             f"+TTA   平均 {tta_arr.mean():.2f}% ± {tta_arr.std():.2f}%",
             f"TTA 提升 {gain.mean():+.2f}pp（判据 ≥2pp）",
             "-" * 56]
    for s, r in per_subject.items():
        lines.append(f"  {s}: frozen={r['frozen']:6.2f}% → TTA={r['tta']:6.2f}% "
                     f"({r['tta']-r['frozen']:+.2f}pp)")
    (ALGO_DIR / "loso_tta_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines))
    print(f"\n✓ 报告已保存：loso_tta_results.json / loso_tta_report.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
