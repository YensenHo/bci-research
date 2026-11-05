#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
statistical_test.py — SNN vs ANN 跨 subject 精度的统计显著性检验

对 LOSO 结果做两类检验，证明「SNN ≈ ANN（零代价复现）」的统计显著性：
1. 配对 t 检验：9 折逐 subject 的 SNN-ANN 精度差，检验均值差是否显著偏离 0
2. Bootstrap 95% CI：对逐 subject 精度差重采样，得到均值差的置信区间

若 95% CI 包含 0 且 p > 0.05，则「SNN 与 ANN 无显著差异」= 零代价复现的统计证据。

用法：
  env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin \
      /opt/miniconda3/envs/openmmlab/bin/python statistical_test.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ALGO_DIR = Path(__file__).resolve().parent


def paired_test(results: dict, name: str, n_boot: int = 10000) -> dict:
    """对某数据集的 LOSO 结果做配对 t 检验 + bootstrap CI。"""
    ann = np.array([results["per_subject"]["eegnet"][s] for s in results["per_subject"]["eegnet"]], float)
    snn = np.array([results["per_subject"]["signed_lif"][s] for s in results["per_subject"]["signed_lif"]], float)
    diff = snn - ann  # 逐 subject 的 SNN-ANN 差
    t_stat, p_val = stats.ttest_rel(snn, ann)
    rng = np.random.default_rng(42)
    boot_means = [rng.choice(diff, size=len(diff), replace=True).mean() for _ in range(n_boot)]
    ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])
    out = dict(mean_diff_pp=round(float(diff.mean()), 2),
               t_stat=round(float(t_stat), 3), p_value=round(float(p_val), 4),
               ci95=[round(float(ci_lo), 2), round(float(ci_hi), 2)],
               significant=bool(p_val < 0.05))
    print(f"[{name}] 9 折配对：SNN-ANN 均值差 {out['mean_diff_pp']:+.2f}pp")
    print(f"  配对 t 检验: t={out['t_stat']}, p={out['p_value']} "
          f"({'显著差异' if out['significant'] else '无显著差异'})")
    print(f"  bootstrap 95% CI: [{out['ci95'][0]:+.2f}, {out['ci95'][1]:+.2f}]pp")
    ci_contains_zero = ci_lo <= 0 <= ci_hi
    print(f"  CI 是否含 0: {'是 → 零代价复现成立' if ci_contains_zero else '否 → 有显著差异'}")
    return out


def main() -> int:
    loso = json.load(open(ALGO_DIR / "loso_results.json"))
    print("=" * 60)
    print("BCIC 2a LOSO 跨 subject：SNN vs ANN 统计检验")
    print("=" * 60)
    res = paired_test(loso, "BCIC 2a")
    (ALGO_DIR / "statistical_test_results.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2))
    print("\n✓ 结果已保存：statistical_test_results.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
