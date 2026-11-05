#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""sample_scan.py — 样本量扫描：操作化「teacher 弱 = 容量 vs 数据量不匹配」机制。

对固定数据集，扫描训练样本量 N ∈ {若干档}，每档记录：
  teacher 满标签 / student 满标签 / teacher−student / 30% 标签蒸馏红利。

核心假说（v4 机制）：当样本量 N 不足以支撑教师容量时，teacher 过拟合弱于 student
  → 红利符号翻负。本实验给出「N → teacher−student → 红利」的完整曲线，
  是「过拟合根因 = 容量 vs 数据量」的直接实证，也是主动构造负红利样本的手段。

用法：
  env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin \
    /opt/miniconda3/envs/openmmlab/bin/python sample_scan.py \
    --data-dir eegmmidb/data --n-channels 21 --input-length 320 --n-classes 4 \
    --sample-sizes 270,540,1080,1620,2160
"""
from __future__ import annotations

import argparse
import json
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


def run_one_size(data_dir, n_channels, n_classes, input_length, seed, N, X_full, y_full,
                 X_ev, y_ev, device):
    """子采样 N 个训练样本，跑完整 sign_check 核心逻辑，返回指标 dict。"""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y_full))[:N]
    X_tr, y_tr = X_full[idx], y_full[idx]

    torch.manual_seed(seed)
    np.random.seed(seed)

    # ① teacher 满标签
    teacher = _build_teacher(n_channels, n_classes, input_length).to(device)
    _train_hard(teacher, X_tr, y_tr, device, 100, 1e-3, 16, seed, True, X_ev, y_ev)
    teacher.eval()
    t_train = _eval(teacher, X_tr, y_tr, device, 16)
    t_val = _eval(teacher, X_ev, y_ev, device, 16)

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

    # ④ 30% 标签子集 → hard vs distilled
    n_lab = int(round(len(y_tr) * LABEL_FRAC))
    idx2 = rng.permutation(len(y_tr))[:n_lab]
    X_sub, y_sub, soft_sub = X_tr[idx2], y_tr[idx2], soft_all[idx2]

    s_hard = _build_student("relu", None, n_channels, n_classes, input_length).to(device)
    acc_hard = _train_hard(s_hard, X_sub, y_sub, device, 100, 1e-3, 16, seed, True, X_ev, y_ev)
    s_dist = _build_student("relu", None, n_channels, n_classes, input_length).to(device)
    acc_dist = _train_distill(s_dist, X_sub, y_sub, soft_sub, device, 100, 1e-3, 16, seed,
                              True, 4.0, 0.5, X_ev, y_ev)

    return {
        "N": int(N),
        "teacher_train": round(t_train * 100, 2),
        "teacher_val": round(t_val * 100, 2),
        "overfit_gap_pp": round((t_train - t_val) * 100, 2),
        "student_full": round(s_full * 100, 2),
        "teacher_minus_student_pp": round((t_val - s_full) * 100, 2),
        "hard_30pct": round(acc_hard * 100, 2),
        "distilled_30pct": round(acc_dist * 100, 2),
        "gain_30pct_pp": round((acc_dist - acc_hard) * 100, 2),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=str, default="eegmmidb/data")
    ap.add_argument("--n-channels", type=int, default=21)
    ap.add_argument("--input-length", type=int, default=320)
    ap.add_argument("--n-classes", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sample-sizes", type=str, default="270,540,1080,1620,2160")
    args = ap.parse_args()

    device = _pick_device("auto")
    data_dir = ALGO_DIR / args.data_dir
    tr = np.load(data_dir / "train.npz")
    ev = np.load(data_dir / "eval.npz")
    X_full = tr["X"].astype(np.float32)
    y_full = tr["y"].astype(np.int64)
    X_ev = ev["X"].astype(np.float32)
    y_ev = ev["y"].astype(np.int64)

    sizes = [int(s) for s in args.sample_sizes.split(",")]
    suffix = "_" + args.data_dir.replace("/", "_").replace("\\", "_")
    out_path = ALGO_DIR / f"sample_scan{suffix}.json"
    print(f"设备 {device} | {args.data_dir} | 样本量扫描 {sizes} | train 总量 {len(y_full)}")

    rows = []
    for N in sizes:
        t0 = time.time()
        row = run_one_size(data_dir, args.n_channels, args.n_classes, args.input_length,
                           args.seed, N, X_full, y_full, X_ev, y_ev, device)
        rows.append(row)
        print(f"  N={N:>5}: teacher {row['teacher_val']:>6.2f}% vs student {row['student_full']:>6.2f}% "
              f"(t−s {row['teacher_minus_student_pp']:+.2f}pp) | 30% 红利 {row['gain_30pct_pp']:+.2f}pp "
              f"({time.time()-t0:.0f}s)", flush=True)
        # 每档完成即增量落盘，中断不丢进度
        out = {"data": args.data_dir, "n_classes": args.n_classes, "seed": args.seed,
               "sample_sizes": sizes, "rows": rows, "done": [r["N"] for r in rows]}
        out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n✓ 已保存 {out_path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
