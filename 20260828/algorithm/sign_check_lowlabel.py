#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""sign_check_lowlabel.py — 低标签代理判据验证：sign(红利) = sign(teacher_30% − student_30%)？

关键重构：现实蒸馏场景（标签稀缺）里 teacher 本来就用 30% 标签训。
之前 sign_check.py 用「满标签 teacher」偏离现实。本脚本同时测两套框架：
  A. 满标签框架（旧）：teacher 满标签 → 软标签 → 蒸馏；判据 sign(teacher满 − student满)
  B. 低标签框架（新）：teacher 30% 标签 → 软标签 → 蒸馏；判据 sign(teacher_30% − student_30%)

若 B 判据在更多范式上强 regime 命中，则一次解决三个硬伤：
  ① 判据用蒸馏前就有的 30% 标签可算（根治决策价值漏洞）
  ② 低标签放大 teacher 过拟合 → 更多范式落强 regime（根治证据偏斜）
  ③ 这是「事前低标签代理判据」，非 Cho/Yuan 的「事后满标签 capacity mismatch」（根治 novelty）

用法：
  env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin PYTHONPATH= \
    /opt/miniconda3/envs/openmmlab/bin/python sign_check_lowlabel.py \
    --data-dir eegmmidb/data --n-channels 21 --input-length 320 --n-classes 4 --seed 42
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


def soft_labels(model, X, device, T=4.0, bs=16):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            xb = torch.from_numpy(X[i:i + bs]).to(device)
            out.append(F.softmax(model(xb) / T, dim=1).cpu().numpy())
    return np.concatenate(out)


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
    nc, ncl, il = args.n_channels, args.n_classes, args.input_length
    torch.manual_seed(seed)
    np.random.seed(seed)

    tr = np.load(data_dir / "train.npz")
    ev = np.load(data_dir / "eval.npz")
    X_tr = tr["X"].astype(np.float32)
    y_tr = tr["y"].astype(np.int64)
    X_ev = ev["X"].astype(np.float32)
    y_ev = ev["y"].astype(np.int64)

    # 30% 标签子集
    rng = np.random.default_rng(seed)
    n_lab = int(round(len(y_tr) * LABEL_FRAC))
    idx = rng.permutation(len(y_tr))[:n_lab]
    X_sub, y_sub = X_tr[idx], y_tr[idx]

    # ① teacher 满标签
    tf = _build_teacher(nc, ncl, il).to(device)
    _train_hard(tf, X_tr, y_tr, device, 100, 1e-3, 16, seed, True, X_ev, y_ev)
    t_full = _eval(tf, X_ev, y_ev, device, 16)

    # ② student 满标签
    sf = _build_student("relu", None, nc, ncl, il).to(device)
    _train_hard(sf, X_tr, y_tr, device, 100, 1e-3, 16, seed, True, X_ev, y_ev)
    s_full = _eval(sf, X_ev, y_ev, device, 16)

    # ③ teacher 30% 标签（低标签 teacher）
    tl = _build_teacher(nc, ncl, il).to(device)
    _train_hard(tl, X_sub, y_sub, device, 100, 1e-3, 16, seed, True, X_ev, y_ev)
    t_low = _eval(tl, X_ev, y_ev, device, 16)

    # ④ student 30% 标签（= hard 基线）
    sh = _build_student("relu", None, nc, ncl, il).to(device)
    hard = _train_hard(sh, X_sub, y_sub, device, 100, 1e-3, 16, seed, True, X_ev, y_ev)

    # ⑤ 软标签（满标签 teacher / 低标签 teacher）
    soft_full = soft_labels(tf, X_sub, device)
    soft_low = soft_labels(tl, X_sub, device)

    # ⑥ distilled（满标签 teacher 软标签）
    sd_full = _build_student("relu", None, nc, ncl, il).to(device)
    dist_full = _train_distill(sd_full, X_sub, y_sub, soft_full, device, 100, 1e-3, 16,
                               seed, True, 4.0, 0.5, X_ev, y_ev)

    # ⑦ distilled（低标签 teacher 软标签）
    sd_low = _build_student("relu", None, nc, ncl, il).to(device)
    dist_low = _train_distill(sd_low, X_sub, y_sub, soft_low, device, 100, 1e-3, 16,
                              seed, True, 4.0, 0.5, X_ev, y_ev)

    # 两套框架
    ts_full = (t_full - s_full) * 100
    gain_full = (dist_full - hard) * 100
    ts_low = (t_low - hard) * 100
    gain_low = (dist_low - hard) * 100

    def sign(x):
        return "pos" if x > 0 else ("neg" if x < 0 else "zero")

    print("=" * 72)
    print(f"{args.data_dir} seed={seed}  (teacher满 {t_full*100:.1f}% | teacher30% {t_low*100:.1f}% | "
          f"student满 {s_full*100:.1f}% | hard30% {hard*100:.1f}%)")
    print(f"  [A 满标签框架] t−s满 {ts_full:+.2f} → 红利满 {gain_full:+.2f} | "
          f"{'✓' if sign(ts_full) == sign(gain_full) else '✗'}")
    print(f"  [B 低标签框架] t−s30% {ts_low:+.2f} → 红利30% {gain_low:+.2f} | "
          f"{'✓' if sign(ts_low) == sign(gain_low) else '✗'}")

    out = {
        "data": args.data_dir, "n_classes": ncl, "seed": seed,
        "teacher_full": round(t_full * 100, 2), "teacher_30pct": round(t_low * 100, 2),
        "student_full": round(s_full * 100, 2), "hard_30pct": round(hard * 100, 2),
        "ts_full_pp": round(ts_full, 2), "gain_full_pp": round(gain_full, 2),
        "ts_low_pp": round(ts_low, 2), "gain_low_pp": round(gain_low, 2),
        "A_consistent": sign(ts_full) == sign(gain_full),
        "B_consistent": sign(ts_low) == sign(gain_low),
    }
    suffix = "_" + args.data_dir.replace("/", "_").replace("\\", "_")
    if args.seed != 42:
        suffix += f"_s{args.seed}"
    out_path = ALGO_DIR / f"lowlabel{suffix}.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✓ 已保存 {out_path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
