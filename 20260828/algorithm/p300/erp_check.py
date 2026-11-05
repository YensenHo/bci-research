#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ERP sanity 检查：目标 vs 非目标闪烁的平均波形差异（P300 应在 ~300ms 出正波）。

也用于报告插图：algorithm/p300/reports/erp_check.png
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Hiragino Sans GB", "STHeiti", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

BASE = Path("/Users/heyixuan/BCI/20260828/algorithm/p300")
OUT = BASE / "reports"
OUT.mkdir(parents=True, exist_ok=True)

with np.load(BASE / "data" / "train.npz") as d:
    X = np.asarray(d["X"], np.float32)     # (n,1,8,205)
    y = np.asarray(d["y"])
meta = json.loads((BASE / "data" / "meta.json").read_text(encoding="utf-8"))
ch = meta["channels"]
FS = meta["fs"]
WIN = X.shape[-1]

print(f"X={X.shape} 均值={X.mean():.4f} 标准差={X.std():.4f} NaN={np.isnan(X).sum()}")
print(f"每 epoch 方差均值 = {X.var(axis=(1,2,3)).mean():.4f}（z-score 后应≈1）")

Xt = X[y == 1][:, 0]    # (n_t,8,205) 目标
Xn = X[y == 0][:, 0]
print(f"目标 {len(Xt)} / 非目标 {len(Xn)}")
# 每通道目标-非目标平均
erp_diff = Xt.mean(axis=0) - Xn.mean(axis=0)   # (8,205)
t_axis = np.arange(WIN) / FS * 1000
# 报告 Pz (idx 2) 与 Oz (idx 3) 在 250-450ms 的平均差
for ci, name in enumerate(ch):
    if name in ("Pz", "Oz", "Cz", "Fz"):
        mask = (t_axis >= 250) & (t_axis <= 450)
        print(f"{name}: 目标-非目标 均值差 [250-450ms] = {erp_diff[ci, mask].mean():+.4f} (z 单位)")
# P300 窗口显著通道排序
mask = (t_axis >= 250) & (t_axis <= 450)
peak_ch = [(ch[i], erp_diff[i, mask].mean()) for i in range(len(ch))]
peak_ch.sort(key=lambda t: -t[1])
print("窗口内正差排序(前4):", [(c, round(v, 4)) for c, v in peak_ch[:4]])

fig, axes = plt.subplots(2, 2, figsize=(11, 7))
for ax, cname in zip(axes.ravel(), ["Pz", "Oz", "Cz", "Fz"]):
    ci = ch.index(cname)
    ax.plot(t_axis, Xt[:, ci].mean(axis=0), label="目标 Target", color="#c0392b")
    ax.plot(t_axis, Xn[:, ci].mean(axis=0), label="非目标 NonTarget", color="#2980b9")
    ax.axvspan(250, 450, color="gray", alpha=0.15)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_title(f"{cname}（z-score 单位）")
    ax.set_xlabel("ms (闪烁 onset 后)")
    ax.legend(fontsize=8)
fig.suptitle("BNCI 008-2014 P300：目标 vs 非目标平均波形（train=A01-A04 均衡集，逐 epoch z-score）")
fig.tight_layout()
fig.savefig(OUT / "erp_check.png", dpi=130)
print(f"✓ {OUT / 'erp_check.png'}")
