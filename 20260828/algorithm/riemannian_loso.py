#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
riemannian_loso.py — 黎曼几何分类器 LOSO 跨 subject 零校准基线

复用 riemannian_baseline.py 的黎曼工具（协方差 → 黎曼均值 → 切线投影 → LR/LDA），
把协议从 cross-session（T→E 混合）改成 LOSO（leave-one-subject-out）：
  9 折，train = 8 subject T 会话，test = 1 subject E 会话（零校准）。

作为「经典 BCI 方法」的强对照，证明「跨 subject 零校准」是公认难题。

配置（沿用 cross-session 网格最优）：shrinkage α=0.0 + LR（multinomial）。
参考点 = 训练集协方差的黎曼（Karcher）均值。

用法：
  env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin \
      /opt/miniconda3/envs/openmmlab/bin/python riemannian_loso.py --smoke
  env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin \
      /opt/miniconda3/envs/openmmlab/bin/python riemannian_loso.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ALGO_DIR = Path(__file__).resolve().parent
BY_SUBJECT = ALGO_DIR / "data_4class_by_subject"
SUBJECTS = [f"A{i:02d}" for i in range(1, 10)]

sys.path.insert(0, str(ALGO_DIR))
from riemannian_baseline import (  # noqa: E402
    compute_covariances, shrink_covariances,
    batch_logm_sym, batch_expm_sym,
    tangent_projection, fit_predict_eval,
)


def main() -> int:
    ap = argparse.ArgumentParser(description="黎曼 LOSO 跨 subject 零校准基线")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--alpha", type=float, default=0.0, help="协方差 shrinkage（cross-session 最优 0.0）")
    ap.add_argument("--clf", type=str, default="LR", choices=["LR", "LDA"])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    data = {s: (np.load(BY_SUBJECT / s / "train.npz"), np.load(BY_SUBJECT / s / "eval.npz"))
            for s in SUBJECTS}
    print(f"黎曼 LOSO 跨 subject 零校准 | α={args.alpha} {args.clf} | {len(SUBJECTS)} 折")

    accs = []
    per_subject = {}
    for test_idx, test_subj in enumerate(SUBJECTS):
        if args.smoke and test_idx > 0:
            break
        Xtr_list, ytr_list = [], []
        for s in SUBJECTS:
            if s == test_subj:
                continue
            Xtr_list.append(data[s][0]["X"]); ytr_list.append(data[s][0]["y"])
        Xtr = np.concatenate(Xtr_list); ytr = np.concatenate(ytr_list)
        Xte, yte = data[test_subj][1]["X"], data[test_subj][1]["y"]

        t0 = time.time()
        C_tr = compute_covariances(Xtr)
        C_te = compute_covariances(Xte)
        Ctr_r = shrink_covariances(C_tr, args.alpha)
        Cte_r = shrink_covariances(C_te, args.alpha)
        # 参考点 = 对数欧氏均值（向量化，比 Karcher 迭代快 ~100 倍）
        ref = batch_expm_sym(batch_logm_sym(Ctr_r).mean(axis=0))
        F_tr = tangent_projection(Ctr_r, ref)
        F_te = tangent_projection(Cte_r, ref)
        r = fit_predict_eval(F_tr, ytr, F_te, yte, args.clf, args.seed)
        accs.append(r["acc_te"])
        per_subject[test_subj] = round(r["acc_te"] * 100, 2)
        print(f"  [{test_subj}] {args.clf}: test={r['acc_te']*100:6.2f}% "
              f"(train={r['acc_tr']*100:.2f}%, {time.time()-t0:.0f}s)")

    arr = np.array(accs) * 100
    report = dict(config=dict(alpha=args.alpha, clf=args.clf, seed=args.seed, smoke=args.smoke),
                  per_subject=per_subject,
                  mean=round(float(arr.mean()), 2), std=round(float(arr.std()), 2))
    (ALGO_DIR / "riemannian_loso_results.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2))
    lines = ["黎曼 LOSO 跨 subject 零校准基线", "=" * 48,
             f"α={args.alpha} {args.clf} | {len(accs)} 折",
             f"平均 {arr.mean():.2f}% ± {arr.std():.2f}%",
             "-" * 48]
    for s, a in per_subject.items():
        lines.append(f"  {s}: {a:.2f}%")
    (ALGO_DIR / "riemannian_loso_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines))
    print(f"\n✓ 报告已保存：riemannian_loso_results.json / riemannian_loso_report.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
