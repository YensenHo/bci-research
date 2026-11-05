#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""finalize_sign_criterion.py — 证据定稿：整合 19 数据点（双 seed 均值），生成相图 + 汇总。

产出：
  1. sign_criterion_phase_diagram.png — 相图（teacher−student × 30% 红利）
  2. sign_criterion_final.json — 19 点双 seed 均值汇总（论文直接引用）

用法：
  env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin \
    /opt/miniconda3/envs/openmmlab/bin/python finalize_sign_criterion.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ALGO_DIR = Path(__file__).resolve().parent


def mean(f42: str, f123: str) -> tuple[float, float]:
    d42 = json.loads((ALGO_DIR / f42).read_text(encoding="utf-8"))
    d123 = json.loads((ALGO_DIR / f123).read_text(encoding="utf-8"))
    return ((d42["teacher_minus_student_pp"] + d123["teacher_minus_student_pp"]) / 2,
            (d42["gain_30pct_pp"] + d123["gain_30pct_pp"]) / 2)


def load_points() -> list[dict]:
    pts: list[dict] = []
    def add(name, task, t, g):
        pts.append({"name": name, "task": task, "t_minus_s": round(t, 2), "gain": round(g, 2)})

    # 核心 3 点（历史双 seed 均值，来自 gap_scan / 论文骨架）
    add("EEGMMIDB-4c", "MI 4-class cross-subject", -2.2, -2.78)
    add("BCIC-2a-cross", "MI 4-class cross-subject", 4.68, 5.56)
    # BCIC within（data_4class，双 seed）
    t, g = mean("sign_check_data_4class.json", "sign_check_data_4class_s123.json")
    add("BCIC-2a-within", "MI 4-class within-subject", t, g)
    # 9 被试 within（双 seed）
    for s in ["A01", "A02", "A03", "A04", "A05", "A06", "A07", "A08", "A09"]:
        t, g = mean(f"sign_check_data_4class_by_subject_{s}.json",
                    f"sign_check_data_4class_by_subject_{s}_s123.json")
        add(f"S-{s}", "MI 4-class within-subject", t, g)
    # CHB-MIT 癫痫（双 seed）
    t, g = mean("sign_check_chbmit_data.json", "sign_check_chbmit_data_s123.json")
    add("CHB-MIT", "seizure 2-class cross-patient", t, g)
    # Ninapro EMG（双 seed）
    t, g = mean("sign_check_myo_cross_17.json", "sign_check_myo_cross_17_s123.json")
    add("NinaPro-17", "EMG 17-class cross-subject", t, g)
    t, g = mean("sign_check_myo_cross.json", "sign_check_myo_cross_s123.json")
    add("NinaPro-12", "EMG 12-class cross-subject", t, g)
    # bci2b / data2c / data2c9 / p300（双 seed）
    t, g = mean("sign_check_data.json", "sign_check_bci2b_data_s123.json")
    add("BCI-2b", "MI 2-class cross-subject", t, g)
    t, g = mean("sign_check_data2c.json", "sign_check_eegmmidb_data2c_s123.json")
    add("EEGMMIDB-2c", "MI 2-class cross-subject", t, g)
    t, g = mean("sign_check_data2c9.json", "sign_check_eegmmidb_data2c9_s123.json")
    add("EEGMMIDB-2c9", "MI 2-class cross-subject (small)", t, g)
    t, g = mean("sign_check_p300_data.json", "sign_check_p300_data_s123.json")
    add("P300", "P300 2-class cross-subject", t, g)
    return pts


def main() -> int:
    pts = load_points()
    strong = [p for p in pts if abs(p["t_minus_s"]) > 2]
    weak = [p for p in pts if abs(p["t_minus_s"]) <= 2]
    strong_hit = sum(1 for p in strong if (p["t_minus_s"] > 0) == (p["gain"] > 0))
    weak_fail = [p["name"] for p in weak if not ((p["t_minus_s"] > 0) == (p["gain"] > 0))]
    neg = [p["name"] for p in pts if p["gain"] < 0]

    summary = {
        "n_points": len(pts),
        "n_strong": len(strong), "strong_hit": strong_hit,
        "strong_acc": strong_hit / len(strong),
        "n_weak": len(weak), "weak_fail": weak_fail,
        "negative_gain_points": neg,
        "points": pts,
    }
    (ALGO_DIR / "sign_criterion_final.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 70)
    print(f"符号判据定稿：{len(pts)} 数据点（全双 seed 均值）")
    print(f"  强 regime (|t−s|>2pp): {len(strong)} 个，命中 {strong_hit} = {strong_hit/len(strong)*100:.0f}%")
    print(f"  弱 regime: {len(weak)} 个，反例 {weak_fail}")
    print(f"  负红利: {neg}")
    print("=" * 70)

    # 相图
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 7.5), dpi=120)
    xs = [p["t_minus_s"] for p in pts]
    ys = [p["gain"] for p in pts]
    xmax = max(abs(min(xs)), abs(max(xs))) * 1.15 + 0.5
    ymax = max(abs(min(ys)), abs(max(ys))) * 1.15 + 0.5

    # 二三象限（同号区）高亮
    ax.axvspan(0, xmax, ymin=0.5, ymax=1, color="tab:green", alpha=0.06)
    ax.axvspan(-xmax, 0, ymin=0, ymax=0.5, color="tab:green", alpha=0.06)
    ax.axhline(0, color="gray", lw=0.9)
    ax.axvline(0, color="gray", lw=0.9)

    for p in pts:
        is_strong = abs(p["t_minus_s"]) > 2
        ok = (p["t_minus_s"] > 0) == (p["gain"] > 0)
        color = "tab:green" if ok else "tab:red"
        marker = "o" if is_strong else "s"
        fill = "full" if is_strong else "none"
        ax.scatter(p["t_minus_s"], p["gain"], c=color if is_strong else "none",
                   marker=marker, s=90, zorder=3, edgecolors=color,
                   linewidths=1.3, facecolors=color if is_strong else "none")
        ax.annotate(p["name"], (p["t_minus_s"], p["gain"]),
                    textcoords="offset points", xytext=(7, 4), fontsize=7)

    ax.set_xlim(-xmax, xmax)
    ax.set_ylim(-ymax, ymax)
    ax.set_xlabel("teacher − student full-label accuracy gap (pp)")
    ax.set_ylabel("30% label distillation gain (pp)")
    ax.set_title("Distillation gain sign = sign(teacher − student)  (strong regime |t−s|>2pp: "
                 f"{strong_hit}/{len(strong)})")
    ax.grid(True, alpha=0.22)
    # 图例
    from matplotlib.lines import Line2D
    legend = [Line2D([0], [0], marker="o", color="w", markerfacecolor="tab:green",
                     markeredgecolor="tab:green", markersize=9, label="strong regime (sign match)"),
              Line2D([0], [0], marker="s", color="w", markerfacecolor="none",
                     markeredgecolor="tab:green", markersize=9, label="weak regime (sign match)"),
              Line2D([0], [0], marker="s", color="w", markerfacecolor="none",
                     markeredgecolor="tab:red", markersize=9, label="weak regime (sign mismatch)")]
    ax.legend(handles=legend, loc="upper left", fontsize=8, framealpha=0.9)
    out = ALGO_DIR / "sign_criterion_phase_diagram.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    print(f"\n✓ 相图已保存：{out}")
    print(f"✓ 汇总已保存：sign_criterion_final.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
