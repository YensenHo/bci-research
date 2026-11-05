#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""sign_check.py — 单点符号规律验证：在任意脑电数据点输出「符号规律 + 25pp 阈值」的全部指标。

对一个数据点，输出：
  teacher 满标签 eval / 过拟合 gap（train−val）/ student 满标签 / teacher−student 符号
  / 30% 标签 hard / 30% 标签 distilled / 30% 红利

用于扩充符号规律和判负阈值的跨数据集验证（3 数据点 → 5+ 数据点）。

用法：
  env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin \
    /opt/miniconda3/envs/openmmlab/bin/python sign_check.py --data-dir eegmmidb/data2c \
    --n-channels 21 --input-length 640 --n-classes 2
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ALGO_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ALGO_DIR))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from distill_4class import (  # noqa: E402
    _build_teacher, _build_student, _train_hard, _train_distill, _eval, _pick_device,
)

LABEL_FRAC = 0.3


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=str, default="eegmmidb/data")
    ap.add_argument("--n-channels", type=int, default=21)
    ap.add_argument("--input-length", type=int, default=320)
    ap.add_argument("--n-classes", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = _pick_device("auto")
    seed = args.seed
    data_dir = ALGO_DIR / args.data_dir
    n_channels, n_classes, input_length = args.n_channels, args.n_classes, args.input_length
    torch.manual_seed(seed)
    np.random.seed(seed)

    tr = np.load(data_dir / "train.npz")
    ev = np.load(data_dir / "eval.npz")
    X_tr = tr["X"].astype(np.float32)
    y_tr = tr["y"].astype(np.int64)
    X_ev = ev["X"].astype(np.float32)
    y_ev = ev["y"].astype(np.int64)
    print(f"设备 {device} | {args.data_dir} | {n_classes}类 | train {X_tr.shape} | eval {X_ev.shape}")

    # ① teacher 满标签（35K）+ 过拟合 gap
    teacher = _build_teacher(n_channels, n_classes, input_length).to(device)
    t0 = time.time()
    _train_hard(teacher, X_tr, y_tr, device, 100, 1e-3, 16, seed, True, X_ev, y_ev)
    teacher.eval()
    t_train = _eval(teacher, X_tr, y_tr, device, 16)
    t_val = _eval(teacher, X_ev, y_ev, device, 16)
    overfit_gap = (t_train - t_val) * 100

    # ② student 满标签
    student = _build_student("relu", None, n_channels, n_classes, input_length).to(device)
    _train_hard(student, X_tr, y_tr, device, 100, 1e-3, 16, seed, True, X_ev, y_ev)
    s_full = _eval(student, X_ev, y_ev, device, 16)

    # ③ teacher 软标签
    soft_all = []
    with torch.no_grad():
        for i in range(0, len(X_tr), 16):
            xb = torch.from_numpy(X_tr[i:i + 16]).to(device)
            soft_all.append(F.softmax(teacher(xb) / 4.0, dim=1).cpu().numpy())
    soft_all = np.concatenate(soft_all)

    # ④ 30% 标签子集
    rng = np.random.default_rng(seed)
    n_lab = int(round(len(y_tr) * LABEL_FRAC))
    idx = rng.permutation(len(y_tr))[:n_lab]
    X_sub, y_sub, soft_sub = X_tr[idx], y_tr[idx], soft_all[idx]

    # ⑤ hard vs distilled（30%）
    s_hard = _build_student("relu", None, n_channels, n_classes, input_length).to(device)
    acc_hard = _train_hard(s_hard, X_sub, y_sub, device, 100, 1e-3, 16, seed, True, X_ev, y_ev)
    s_dist = _build_student("relu", None, n_channels, n_classes, input_length).to(device)
    acc_dist = _train_distill(s_dist, X_sub, y_sub, soft_sub, device, 100, 1e-3, 16, seed, True,
                              4.0, 0.5, X_ev, y_ev)

    gain = (acc_dist - acc_hard) * 100
    t_minus_s = (t_val - s_full) * 100
    print("=" * 70)
    print(f"  teacher 满标签 : {t_val*100:.2f}%  (train {t_train*100:.2f}%)")
    print(f"  student 满标签 : {s_full*100:.2f}%")
    print(f"  teacher−student: {t_minus_s:+.2f}pp  过拟合 gap: {overfit_gap:+.2f}pp")
    print(f"  30% hard {acc_hard*100:.2f}% → distilled {acc_dist*100:.2f}% | 红利 {gain:+.2f}pp")
    print(f"  阈值判定: gap {overfit_gap:+.1f}pp {'>' if overfit_gap > 25 else '<'} 25pp "
          f"→ 预测{'负' if overfit_gap > 25 else '正'}红利，实际{'负' if gain < 0 else '正'}红利 "
          f"{'✓ 一致' if (overfit_gap > 25) == (gain < 0) else '✗ 不一致'}")

    import json
    out = {
        "data": args.data_dir, "n_classes": n_classes, "seed": seed,
        "teacher_val": round(t_val * 100, 2), "teacher_train": round(t_train * 100, 2),
        "overfit_gap_pp": round(overfit_gap, 2),
        "student_full": round(s_full * 100, 2), "teacher_minus_student_pp": round(t_minus_s, 2),
        "hard_30pct": round(acc_hard * 100, 2), "distilled_30pct": round(acc_dist * 100, 2),
        "gain_30pct_pp": round(gain, 2),
        "threshold_predict": "neg" if overfit_gap > 25 else "pos",
        "actual": "neg" if gain < 0 else "pos",
        "consistent": (overfit_gap > 25) == (gain < 0),
    }
    suffix = "_" + args.data_dir.replace("/", "_").replace("\\", "_")
    if args.seed != 42:
        suffix += f"_s{args.seed}"
    out_path = ALGO_DIR / f"sign_check{suffix}.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n✓ 已保存 {out_path.name}（{time.time()-t0:.0f}s）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
