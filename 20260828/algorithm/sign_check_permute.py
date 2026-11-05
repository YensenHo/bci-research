#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""sign_check_permute.py — 标签置换 teacher：制造「自信地错」的 below-chance teacher。

把 teacher 的训练标签按固定映射置换（y -> (y+shift) mod C，或 shuffle 打乱），
teacher 学到「错误映射」，在真实 eval 上满标签精度 < chance。其余（student/hard/
distilled）用真实标签。用于跨数据集复现「teacher 低于 chance → 蒸馏有害」。

用法：
  python sign_check_permute.py --data-dir eegmmidb/data --n-channels 21 \
    --input-length 320 --n-classes 4 --permute shift=1 --seed 42
  --permute none       : 真实标签（对照，等价 sign_check.py）
  --permute shift=K    : y -> (y+K) mod C（固定错位，制造 below-chance teacher）
  --permute shuffle    : 完全打乱标签（teacher 学随机映射，≈chance）
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


def permute_labels(y: np.ndarray, mode: str, n_classes: int, seed: int) -> np.ndarray:
    """返回置换后的标签。mode: 'none' / 'shift=K' / 'shuffle'。"""
    if mode == "none":
        return y.copy()
    if mode == "shuffle":
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n_classes)
        return perm[y].astype(np.int64)
    if mode.startswith("shift="):
        k = int(mode.split("=")[1]) % n_classes
        return ((y + k) % n_classes).astype(np.int64)
    raise ValueError(f"未知 permute 模式: {mode}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=str, default="eegmmidb/data")
    ap.add_argument("--n-channels", type=int, default=21)
    ap.add_argument("--input-length", type=int, default=320)
    ap.add_argument("--n-classes", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--permute", type=str, default="shift=1",
                    help="none / shift=K / shuffle")
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

    # teacher 用置换标签训，其余用真实标签
    y_tr_teacher = permute_labels(y_tr, args.permute, n_classes, seed)

    # ① teacher（置换标签）+ 过拟合 gap
    teacher = _build_teacher(n_channels, n_classes, input_length).to(device)
    t0 = time.time()
    _train_hard(teacher, X_tr, y_tr_teacher, device, 100, 1e-3, 16, seed, True, X_ev, y_ev)
    teacher.eval()
    t_val = _eval(teacher, X_ev, y_ev, device, 16)  # 真实标签 eval

    # ② student 满标签（真实标签）
    student = _build_student("relu", None, n_channels, n_classes, input_length).to(device)
    _train_hard(student, X_tr, y_tr, device, 100, 1e-3, 16, seed, True, X_ev, y_ev)
    s_full = _eval(student, X_ev, y_ev, device, 16)

    # ③ teacher 软标签（来自置换 teacher）
    soft_all = []
    with torch.no_grad():
        for i in range(0, len(X_tr), 16):
            xb = torch.from_numpy(X_tr[i:i + 16]).to(device)
            soft_all.append(F.softmax(teacher(xb) / 4.0, dim=1).cpu().numpy())
    soft_all = np.concatenate(soft_all)

    # ④ 30% 标签子集（真实标签）
    rng = np.random.default_rng(seed)
    n_lab = int(round(len(y_tr) * LABEL_FRAC))
    idx = rng.permutation(len(y_tr))[:n_lab]
    X_sub, y_sub, soft_sub = X_tr[idx], y_tr[idx], soft_all[idx]

    # ⑤ hard vs distilled（30%，真实标签）
    s_hard = _build_student("relu", None, n_channels, n_classes, input_length).to(device)
    acc_hard = _train_hard(s_hard, X_sub, y_sub, device, 100, 1e-3, 16, seed, True, X_ev, y_ev)
    s_dist = _build_student("relu", None, n_channels, n_classes, input_length).to(device)
    acc_dist = _train_distill(s_dist, X_sub, y_sub, soft_sub, device, 100, 1e-3, 16, seed, True,
                              4.0, 0.5, X_ev, y_ev)

    gain = (acc_dist - acc_hard) * 100
    chance = 100.0 / n_classes
    print("=" * 70)
    print(f"  [{args.data_dir}] permute={args.permute} {n_classes}类 chance={chance:.1f}%")
    print(f"  teacher 满标签 : {t_val*100:.2f}%  {'< chance 崩' if t_val*100 < chance else ''}")
    print(f"  student 满标签 : {s_full*100:.2f}%")
    print(f"  30% hard {acc_hard*100:.2f}% → distilled {acc_dist*100:.2f}% | 红利 {gain:+.2f}pp")

    out = {
        "data": args.data_dir, "permute": args.permute, "n_classes": n_classes,
        "seed": seed, "chance": round(chance, 2),
        "teacher_val": round(t_val * 100, 2), "student_full": round(s_full * 100, 2),
        "hard_30pct": round(acc_hard * 100, 2), "distilled_30pct": round(acc_dist * 100, 2),
        "gain_30pct_pp": round(gain, 2),
    }
    name = args.data_dir.replace("/", "_").replace("\\", "_")
    suffix = f"_{name}_perm{args.permute.replace('=','')}"
    if args.seed != 42:
        suffix += f"_s{args.seed}"
    out_path = ALGO_DIR / f"sign_check_permute{suffix}.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n✓ 已保存 {out_path.name}（{time.time()-t0:.0f}s）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
