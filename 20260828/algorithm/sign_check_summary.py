#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""sign_check_summary.py — 汇总所有 sign_check_*.json，输出符号规律命中率 + 相图。

读 ALGO_DIR 下所有 sign_check_*.json（新命名 sign_check_<data_dir>.json），
汇总每个数据点：teacher−student 符号 vs 红利符号，统计：
  1) 符号规律命中率（sign(gain) == sign(teacher−student)）
  2) 强 regime（|t−s| > 2pp）命中率
  3) 过拟合 gap 25pp 阈值命中率（对照，预期跨模态失效）
  4) 相图：横轴 teacher−student(pp)，纵轴 30% 红利(pp)

用法：
  env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin \
    /opt/miniconda3/envs/openmmlab/bin/python sign_check_summary.py
"""
from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

ALGO_DIR = Path(__file__).resolve().parent

# 旧命名（dirname 级别，未迁移的）也一并纳入
def load_all():
    rows = []
    for p in sorted(glob.glob(str(ALGO_DIR / "sign_check_*.json"))):
        name = Path(p).stem  # sign_check_xxx
        try:
            d = json.loads(Path(p).read_text(encoding="utf-8"))
        except Exception as e:
            print(f"✗ 跳过 {name}: {e}")
            continue
        # 排除空/无效
        if not isinstance(d, dict) or "gain_30pct_pp" not in d:
            continue
        rows.append((name, d))
    return rows


def main():
    rows = load_all()
    if not rows:
        print("无 sign_check_*.json 数据")
        return 1

    print("=" * 100)
    print(f"{'数据点':<42} {'类别':>3} {'t−s(pp)':>8} {'过拟合gap':>9} {'红利(pp)':>9} "
          f"{'符号':>5} {'gap阈值':>7}")
    print("-" * 100)

    sign_hit = 0
    sign_hit_strong = 0
    n_strong = 0
    gap_thresh_hit = 0
    total = 0
    points = []
    for name, d in rows:
        tms = d.get("teacher_minus_student_pp", 0.0)
        gain = d.get("gain_30pct_pp", 0.0)
        gap = d.get("overfit_gap_pp", 0.0)
        nc = d.get("n_classes", 0)
        total += 1
        sign_ok = (tms > 0) == (gain > 0)  # 严格同号（不含 0）
        strong = abs(tms) > 2.0
        if sign_ok:
            sign_hit += 1
        if strong:
            n_strong += 1
            if sign_ok:
                sign_hit_strong += 1
        gap_ok = (gap > 25) == (gain < 0)
        if gap_ok:
            gap_thresh_hit += 1
        points.append((name, nc, tms, gap, gain, sign_ok, strong, gap_ok))
        flag = "✓" if sign_ok else "✗"
        gflag = "✓" if gap_ok else "✗"
        print(f"{name:<42} {nc:>3} {tms:>+8.2f} {gap:>+9.2f} {gain:>+9.2f} "
              f"{flag:>5} {gflag:>7}")

    print("-" * 100)
    print(f"数据点总数 : {total}")
    print(f"符号规律命中（sign(gain)=sign(t−s)）        : {sign_hit}/{total} = {sign_hit/total*100:.0f}%")
    print(f"  其中强 regime（|t−s|>2pp）                : {n_strong} 个，命中 {sign_hit_strong} = "
          f"{(sign_hit_strong/n_strong*100 if n_strong else float('nan')):.0f}%")
    print(f"过拟合 gap 25pp 阈值命中（对照）            : {gap_thresh_hit}/{total} = {gap_thresh_hit/total*100:.0f}%")

    # 相图（matplotlib 可选）
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 7), dpi=110)
        for name, nc, tms, gap, gain, sign_ok, strong, gap_ok in points:
            color = "tab:green" if sign_ok else "tab:red"
            marker = "o" if strong else "s"  # 强 regime 实心圆，弱 regime 方
            ax.scatter(tms, gain, c=color, marker=marker, s=70, zorder=3, edgecolors="k", linewidths=0.5)
            ax.annotate(name.replace("sign_check_", ""), (tms, gain),
                        textcoords="offset points", xytext=(6, 4), fontsize=6, alpha=0.85)
        ax.axhline(0, color="gray", lw=0.8)
        ax.axvline(0, color="gray", lw=0.8)
        # 二三象限填充（同号区）
        ax.axvspan(0, max(p[2] for p in points) * 1.1 + 1, ymin=0.5, color="tab:green", alpha=0.05)
        ax.axvspan(min(p[2] for p in points) * 1.1 - 1, 0, ymin=0, ymax=0.5, color="tab:green", alpha=0.05)
        ax.set_xlabel("teacher − student (pp)")
        ax.set_ylabel("30% label distillation gain (pp)")
        ax.set_title("Sign criterion: sign(gain) = sign(teacher − student)")
        ax.grid(True, alpha=0.25)
        out = ALGO_DIR / "sign_check_phase_diagram.png"
        fig.savefig(out, bbox_inches="tight")
        print(f"\n✓ 相图已保存：{out}")
    except Exception as e:
        print(f"\n(matplotlib 相图跳过：{e})")

    # 导出汇总 JSON
    summary = {
        "n_points": total,
        "sign_hit": sign_hit, "sign_hit_strong": sign_hit_strong, "n_strong": n_strong,
        "gap_thresh_hit": gap_thresh_hit,
        "points": [{"name": n, "n_classes": nc, "t_minus_s_pp": tms, "overfit_gap_pp": gap,
                    "gain_pp": g, "sign_ok": so, "strong": st, "gap_ok": go}
                   for n, nc, tms, gap, g, so, st, go in points],
    }
    (ALGO_DIR / "sign_check_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✓ 汇总已保存：sign_check_summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
