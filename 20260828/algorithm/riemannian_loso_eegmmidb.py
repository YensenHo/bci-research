#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
riemannian_loso_eegmmidb.py — 黎曼 LOSO 跨 subject 零校准基线（EEGMMIDB）

复用 riemannian_baseline 的黎曼工具（对数欧氏均值 + 切线投影 + LR/LDA），
数据源换成 EEGMMIDB（30 subject × 90 试次，21ch/320/4类），LOSO 30 折。

作为第二数据集的「经典 BCI 方法」强对照。

用法：
  env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin \
      /opt/miniconda3/envs/openmmlab/bin/python riemannian_loso_eegmmidb.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ALGO_DIR = Path(__file__).resolve().parent
DATA_DIR = ALGO_DIR / "eegmmidb" / "data_4class_by_subject"
SUBJECTS = [f"S{i:03d}" for i in range(1, 31)]
N_CH = 21

sys.path.insert(0, str(ALGO_DIR))
from riemannian_baseline import (  # noqa: E402
    batch_logm_sym, batch_expm_sym, batch_inv_sqrtm, fit_predict_eval,
)


def compute_cov(X: np.ndarray) -> np.ndarray:
    """X (n,1,21,320) -> 协方差 (n,21,21)。"""
    x = X[:, 0]
    xc = x - x.mean(axis=2, keepdims=True)
    C = xc @ xc.swapaxes(-1, -2) / (x.shape[2] - 1)
    return 0.5 * (C + C.swapaxes(-1, -2))


def tangent_proj(C: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """切线空间投影（动态 N_CH=21）。"""
    P_inv_sqrt = batch_inv_sqrtm(ref[None])[0]
    S = P_inv_sqrt @ C @ P_inv_sqrt
    T = batch_logm_sym(S)
    iu = np.triu_indices(N_CH)
    vec = T[:, iu[0], iu[1]].copy()
    off = iu[0] != iu[1]
    vec[:, off] *= np.sqrt(2.0)
    return vec


def main() -> int:
    ap = argparse.ArgumentParser(description="黎曼 LOSO 跨 subject（EEGMMIDB）")
    ap.add_argument("--clf", type=str, default="LR", choices=["LR", "LDA"])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    data = {s: np.load(DATA_DIR / s / "data.npz") for s in SUBJECTS}
    print(f"黎曼 LOSO EEGMMIDB | {args.clf} | {len(SUBJECTS)} 折 | 21ch/320")

    accs = []
    per_subject = {}
    for test_idx, test_subj in enumerate(SUBJECTS):
        Xtr_list, ytr_list = [], []
        for s in SUBJECTS:
            if s == test_subj:
                continue
            Xtr_list.append(data[s]["X"]); ytr_list.append(data[s]["y"])
        Xtr = np.concatenate(Xtr_list); ytr = np.concatenate(ytr_list)
        Xte, yte = data[test_subj]["X"], data[test_subj]["y"]

        t0 = time.time()
        C_tr = compute_cov(Xtr); C_te = compute_cov(Xte)
        ref = batch_expm_sym(batch_logm_sym(C_tr).mean(axis=0))
        F_tr = tangent_proj(C_tr, ref); F_te = tangent_proj(C_te, ref)
        r = fit_predict_eval(F_tr, ytr, F_te, yte, args.clf, args.seed)
        accs.append(r["acc_te"])
        per_subject[test_subj] = round(r["acc_te"] * 100, 2)
        print(f"  [{test_subj}] {args.clf}: test={r['acc_te']*100:5.2f}% "
              f"(train={r['acc_tr']*100:.2f}%, {time.time()-t0:.0f}s)")

    arr = np.array(accs) * 100
    report = dict(clf=args.clf, per_subject=per_subject,
                  mean=round(float(arr.mean()), 2), std=round(float(arr.std()), 2))
    (ALGO_DIR / "riemannian_loso_eegmmidb_results.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2))
    lines = ["黎曼 LOSO EEGMMIDB 跨 subject", "=" * 40,
             f"{args.clf} | {len(accs)} 折 | 平均 {arr.mean():.2f}% ± {arr.std():.2f}%"]
    (ALGO_DIR / "riemannian_loso_eegmmidb_report.txt").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))
    print("✓ 报告已保存")
    return 0


if __name__ == "__main__":
    sys.exit(main())
