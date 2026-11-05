#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""kd_decomposition.py — 验证 K + R 分解机制。

v4 理论：蒸馏红利 = 知识传递项 K（teacher 软标签携带的知识）+ 正则化项 R（软标签的 label smoothing 效应）。
本实验在 teacher 弱（过拟合）场景下，用三种训练方式分离 K 和 R：

  - hard-only      : 30% 硬标签（基线，无软标签）
  - label smoothing: 30% 硬标签 + 均匀软标签正则化（只含 R，无 teacher 知识）
  - distilled      : 30% 硬标签 + teacher 软标签（含 K + R）

分解：
  R     = label_smoothing − hard       （软标签的正则化红利）
  K + R = distilled − hard             （总蒸馏红利）
  K     = distilled − label_smoothing  （teacher 知识 vs 均匀软标签的差异）

预期（EEGMMIDB 4类，teacher 弱）：R > 0（正则化有益），K < 0（teacher 弱→有害知识），K+R < 0（K 负主导）。

用法：
  env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin \
    /opt/miniconda3/envs/openmmlab/bin/python kd_decomposition.py
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
    _build_teacher, _build_student, _train_hard, _train_distill, _eval, _pick_device, BRIEF,
)

LABEL_FRAC = 0.3
EPS = 0.1  # label smoothing 的均匀化强度


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=str, default="eegmmidb/data")
    ap.add_argument("--n-channels", type=int, default=21)
    ap.add_argument("--input-length", type=int, default=320)
    ap.add_argument("--n-classes", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--teacher-f1", type=int, default=16)
    ap.add_argument("--teacher-f2", type=int, default=32)
    ap.add_argument("--teacher-ft", type=int, default=24)
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
    print(f"设备 {device} | {args.data_dir} | train {X_tr.shape} | eval {X_ev.shape}")

    # ① 教师（容量可配，满标签）
    teacher = _build_teacher(n_channels, n_classes, input_length,
                             args.teacher_f1, args.teacher_f2, args.teacher_ft).to(device)
    t0 = time.time()
    t_acc = _train_hard(teacher, X_tr, y_tr, device, 100, 1e-3, 16, seed, True, X_ev, y_ev)
    print(f"teacher eval {t_acc*100:.2f}%（{time.time()-t0:.0f}s）")

    # ② teacher 软标签（T=4）
    teacher.eval()
    soft_all = []
    with torch.no_grad():
        for i in range(0, len(X_tr), 16):
            xb = torch.from_numpy(X_tr[i:i + 16]).to(device)
            soft_all.append(F.softmax(teacher(xb) / 4.0, dim=1).cpu().numpy())
    soft_t = np.concatenate(soft_all)

    # ③ 30% 标签子集
    rng = np.random.default_rng(seed)
    n_lab = int(round(len(y_tr) * LABEL_FRAC))
    idx = rng.permutation(len(y_tr))[:n_lab]
    X_sub, y_sub = X_tr[idx], y_tr[idx]
    soft_t_sub = soft_t[idx]

    # ④ 构造均匀软标签（label smoothing，无 teacher 知识）
    onehot = np.eye(n_classes, dtype=np.float32)[y_sub]
    soft_ls = (1.0 - EPS) * onehot + EPS * (np.ones(n_classes, dtype=np.float32) / n_classes)

    # ⑤ 三种方式训练学生
    def _run(label, mode):
        s = _build_student("relu", None, n_channels, n_classes, input_length).to(device)
        if mode == "hard":
            acc = _train_hard(s, X_sub, y_sub, device, 100, 1e-3, 16, seed, True, X_ev, y_ev)
        else:
            soft = soft_ls if mode == "ls" else soft_t_sub
            acc = _train_distill(s, X_sub, y_sub, soft, device, 100, 1e-3, 16, seed, True, 4.0, 0.5, X_ev, y_ev)
        print(f"  {label:<16} eval {acc*100:.2f}%")
        return acc * 100

    print("=" * 60)
    hard = _run("hard-only", "hard")
    ls = _run("label-smoothing", "ls")
    dist = _run("distilled", "dist")

    R = ls - hard
    K_plus_R = dist - hard
    K = dist - ls
    print("=" * 60)
    print(f"  基线 hard-only       : {hard:.2f}%")
    print(f"  label smoothing (R) : {ls:.2f}%   → R     = {R:+.2f}pp")
    print(f"  distilled (K+R)     : {dist:.2f}%   → K+R   = {K_plus_R:+.2f}pp")
    print(f"  知识传递项 K         :           → K     = {K:+.2f}pp")
    print(f"\n  验证：K 符号 {('负' if K < 0 else '正')}，R 符号 {('负' if R < 0 else '正')}，"
          f"总红利 {('负' if K_plus_R < 0 else '正')}")

    import json
    out = {
        "data": args.data_dir,
        "label_frac": LABEL_FRAC, "eps": EPS, "seed": seed,
        "teacher_eval": round(t_acc * 100, 2),
        "hard": round(hard, 2), "label_smoothing": round(ls, 2), "distilled": round(dist, 2),
        "R_pp": round(R, 2), "K_pp": round(K, 2), "total_gain_pp": round(K_plus_R, 2),
    }
    suffix = "_" + args.data_dir.replace("/", "_").replace("\\", "_")
    if args.seed != 42:
        suffix += f"_s{args.seed}"
    if (args.teacher_f1, args.teacher_f2, args.teacher_ft) != (16, 32, 24):
        suffix += f"_tf{args.teacher_f1}-{args.teacher_f2}-{args.teacher_ft}"
    out_path = ALGO_DIR / f"kd_decomposition{suffix}.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n✓ 已保存 {out_path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
