#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""r_flip.py — 验证 R（label smoothing 正则化）随标签预算的翻转曲线。

动机：Yuan 2020 在满标签下发现「KD ≈ label smoothing，有益」（R 正）；
本工作发现低标签（30%）下 label smoothing 有害（R 负）。中间的翻转点是
Yuan 2020 之后的增量。本脚本扫描标签预算，画出 R 随标签预算的翻转曲线，
标定 R 从负转正的翻转阈值。

R 定义：label smoothing 红利 = (label smoothing 训练) − (hard-only 训练)，不含 teacher 知识。
每档标签预算只训练 2 个学生（hard / label smoothing），无需 teacher。

用法：
  env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin \
    /opt/miniconda3/envs/openmmlab/bin/python r_flip.py --data-dir eegmmidb/data \
    --n-channels 21 --input-length 320 --n-classes 4
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
    _build_student, _train_hard, _train_distill, _pick_device,
)

EPS = 0.1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=str, default="eegmmidb/data")
    ap.add_argument("--n-channels", type=int, default=21)
    ap.add_argument("--input-length", type=int, default=320)
    ap.add_argument("--n-classes", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fracs", type=str, default="0.1,0.15,0.2,0.25,0.3,0.5,1.0")
    args = ap.parse_args()

    device = _pick_device("auto")
    seed = args.seed
    fracs = [float(f) for f in args.fracs.split(",")]
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
    print(f"设备 {device} | {args.data_dir} | {n_classes}类 | train {X_tr.shape}")

    print(f"\n{'标签预算':>8} {'标签数':>6} {'hard':>8} {'label-smooth':>12} {'R(红利)':>8}")
    rows = []
    for frac in fracs:
        rng = np.random.default_rng(seed)
        n_lab = int(round(len(y_tr) * frac))
        idx = rng.permutation(len(y_tr))[:n_lab]
        X_sub, y_sub = X_tr[idx], y_tr[idx]

        # hard-only
        s_hard = _build_student("relu", None, n_channels, n_classes, input_length).to(device)
        acc_hard = _train_hard(s_hard, X_sub, y_sub, device, 100, 1e-3, 16, seed, True, X_ev, y_ev)

        # label smoothing（均匀软标签）
        onehot = np.eye(n_classes, dtype=np.float32)[y_sub]
        soft_ls = (1.0 - EPS) * onehot + EPS * (np.ones(n_classes, dtype=np.float32) / n_classes)
        s_ls = _build_student("relu", None, n_channels, n_classes, input_length).to(device)
        acc_ls = _train_distill(s_ls, X_sub, y_sub, soft_ls, device, 100, 1e-3, 16, seed, True,
                                4.0, 0.5, X_ev, y_ev)

        R = (acc_ls - acc_hard) * 100
        rows.append((frac, n_lab, acc_hard * 100, acc_ls * 100, R))
        print(f"{frac:>8.0%} {n_lab:>6d} {acc_hard*100:>7.2f}% {acc_ls*100:>11.2f}% {R:>+7.2f}pp")

    # 翻转点
    print("\n=== R 翻转曲线 ===")
    signs = [1 if r[4] > 0 else (-1 if r[4] < 0 else 0) for r in rows]
    flip = None
    for i in range(1, len(rows)):
        if signs[i] != signs[i - 1] and signs[i - 1] != 0:
            flip = (rows[i - 1][0], rows[i][0])
            break
    print("  R 随标签预算:", " → ".join(f"{r[0]:.0%}:{r[4]:+.1f}pp" for r in rows))
    print(f"  翻转点: {flip[0]:.0%}→{flip[1]:.0%}" if flip else "  无翻转（全部同号）")

    import json
    out = {
        "data": args.data_dir, "n_classes": n_classes, "seed": seed, "eps": EPS,
        "points": [{"frac": r[0], "n_labels": r[1], "hard": round(r[2], 2),
                    "label_smoothing": round(r[3], 2), "R_pp": round(r[4], 2)} for r in rows],
        "flip_between": flip,
    }
    suffix = "_" + args.data_dir.replace("/", "_").replace("\\", "_")
    if args.seed != 42:
        suffix += f"_s{args.seed}"
    out_path = ALGO_DIR / f"r_flip{suffix}.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n✓ 已保存 {out_path.name}（{time.time():.0f}s）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
