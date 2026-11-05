#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""plot_hbc_tcnet_snr_curve.py — HBC 噪声增强 × LIF-EEG-TCNet 的 SNR 退化曲线（暗色主题）。"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
res = json.loads((HERE / "hbc_aug_results" / "snr_sweep.json").read_text(encoding="utf-8"))
pm = res["per_model"]
pts = res["sweep_points"]
base_pts = [p for p in pts if p != "8dB+突发"]
burst_pts = ["8dB+突发"]

names_b = ["无增强_s42", "无增强_s123"]
names_a = ["增强_s42", "增强_s123"]


def agg(keys, pts):
    return {p: (float(np.mean([pm[k][p] for k in keys])) * 100,
                float(np.std([pm[k][p] for k in keys])) * 100) for p in pts}


base = agg(names_b, base_pts)
aug = agg(names_a, base_pts)
base_b = agg(names_b, burst_pts)["8dB+突发"]
aug_b = agg(names_a, burst_pts)["8dB+突发"]

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
plt.rcParams["font.sans-serif"] = ["Hiragino Sans GB", "STHeiti", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

labels = ["∞", "20", "16", "12", "8", "4"]
x = list(range(len(base_pts)))

fig, ax = plt.subplots(figsize=(7.5, 5), dpi=150, facecolor="#1B2A4A")
ax.set_facecolor("#243555")
for sp in ax.spines.values():
    sp.set_color("#8FA3BF")
ax.tick_params(colors="white", labelsize=11)
ax.grid(True, color="#8FA3BF", alpha=0.25, lw=0.8)

ax.errorbar(x, [base[p][0] for p in base_pts], yerr=[base[p][1] for p in base_pts],
            color="#E76F51", lw=2.2, marker="o", ms=6, capsize=4, label="无增强（baseline）")
ax.errorbar(x, [aug[p][0] for p in base_pts], yerr=[aug[p][1] for p in base_pts],
            color="#00B4D8", lw=2.2, marker="s", ms=6, capsize=4,
            label="HBC增强 (SNR∈[10,20]dB, AWGN+突发)")
# 附加域 8dB+突发（菱形）
ax.scatter([x[4]], [base_b[0]], color="#E76F51", marker="D", s=70, zorder=5)
ax.scatter([x[4]], [aug_b[0]], color="#00B4D8", marker="D", s=70, zorder=5)
ax.annotate("8dB+突发\n(附加域)", xy=(x[4], aug_b[0]), xytext=(x[4] - 0.35, aug_b[0] + 7),
            color="white", fontsize=9, ha="center",
            arrowprops=dict(arrowstyle="->", color="#8FA3BF", lw=0.8))

ax.axhline(50, color="#8FA3BF", lw=1, ls="--", alpha=0.6)
ax.text(-0.2, 50.8, "50%", color="#8FA3BF", fontsize=9)
ax.axhline(25, color="#8FA3BF", lw=1, ls=":", alpha=0.5)
ax.text(-0.2, 25.8, "chance 25%", color="#8FA3BF", fontsize=9)

ax.set_xticks(x)
ax.set_xticklabels(labels, color="white")
ax.set_xlabel("测试 SNR (dB)，HBC 噪声（5bit 量化 + AWGN σ/H=10^(-SNR/20)）", color="white", fontsize=12)
ax.set_ylabel("final test 准确率 (%)", color="white", fontsize=12)
ax.set_title("HBC 噪声增强 × LIF-EEG-TCNet：SNR 退化曲线（2 种子 mean±std）",
             color="white", fontsize=14, weight="bold", pad=12)
ax.legend(facecolor="#243555", edgecolor="#8FA3BF", labelcolor="white", loc="lower left")
ax.set_ylim(20, 75)
fig.tight_layout(pad=2.0)
out = HERE / "hbc_aug_results" / "snr_curve_tcnet.png"
fig.savefig(out, facecolor="#1B2A4A", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"✓ 已保存：{out}")
