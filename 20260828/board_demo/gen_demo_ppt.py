#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成 PPT 用的板载四分类闭环演示图：暗色、间隔充足、字号加大，适配 ~4.57×3.05 英寸放置。

相比旧版：去掉 suptitle（与 PPT 下方说明重复）、wspace 0.3→0.55、字号整体上调、
手动 subplots_adjust 控制边距，解决「文字间隔不足」的问题。
"""
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Hiragino Sans GB", "STHeiti", "Arial Unicode MS", "PingFang SC"]
plt.rcParams["axes.unicode_minus"] = False

BG = "#0D1B2A"; CARD = "#142133"; TXT = "#E8ECF1"; BODY = "#B0BEC5"
BLUE = "#58A6FF"; GREEN = "#00C853"; MUTED = "#5A6E82"; GRID = "#1E2D3F"
LABEL_CN = ["左手", "右手", "双脚", "舌头"]

LINE_RE = re.compile(r"\[\s*(\d+)\]\s*真值=(\S+)\s+直连=(\S+)\s+经HBC=(\S+)\s+[✓✗]\s*\|\s*([\d.]+)ms")

truth, direct, hbc = [], [], []
with open("lif_4class_stream.log", encoding="utf-8", errors="replace") as f:
    for line in f:
        m = LINE_RE.search(line)
        if not m:
            continue
        truth.append(LABEL_CN.index(m.group(2)))
        direct.append(LABEL_CN.index(m.group(3)))
        hbc.append(LABEL_CN.index(m.group(4)))

truth = np.array(truth); direct = np.array(direct); hbc = np.array(hbc)
n = len(truth)
direct_ok = direct == truth
hbc_ok = hbc == truth

xs = np.arange(1, n + 1)
da = np.cumsum(direct_ok) / xs * 100
ha = np.cumsum(hbc_ok) / xs * 100

def per_class_acc(pred):
    out = []
    for c in range(4):
        m = truth == c
        out.append(100.0 * (pred[m] == c).mean() if m.sum() else 0.0)
    return out

da_cls = per_class_acc(direct)
ha_cls = per_class_acc(hbc)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(4.6, 3.1), dpi=220,
                               gridspec_kw={"width_ratios": [1.4, 1]})
fig.patch.set_facecolor(BG)
for ax in (ax1, ax2):
    ax.set_facecolor(CARD)
    for sp in ax.spines.values():
        sp.set_color(GRID)
    ax.tick_params(colors=BODY, labelsize=9.5)
    ax.grid(True, color=GRID, lw=0.5, alpha=0.7)

# 左：累计准确率曲线（直连 vs 经 HBC）
ax1.plot(xs, da, color=BLUE, lw=1.8, label="直连解码")
ax1.plot(xs, ha, color=GREEN, lw=1.8, label="经 HBC 回传")
ax1.axhline(25, color=MUTED, ls="--", lw=1)
ax1.text(n, 27, "随机基线 25%", color=MUTED, fontsize=8.5, ha="right")
ax1.set_ylim(0, 100)
ax1.set_xlabel("窗口序号", color=BODY, fontsize=10.5)
ax1.set_ylabel("累计准确率 %", color=BODY, fontsize=10.5)
ax1.set_title("直连 vs 经 HBC：几乎无损", color=TXT, fontsize=12.5, pad=9)
ax1.legend(fontsize=9.5, frameon=False, loc="lower right", labelcolor=BODY)

# 右：四分类逐类准确率
xpos = np.arange(4); w = 0.34
ax2.bar(xpos - w / 2, da_cls, w, color=BLUE, label="直连")
ax2.bar(xpos + w / 2, ha_cls, w, color=GREEN, label="经 HBC")
ax2.set_xticks(xpos)
ax2.set_xticklabels(LABEL_CN, color=BODY, fontsize=9.5)
ax2.set_ylim(0, 100)
ax2.set_title("四分类逐类", color=TXT, fontsize=12.5, pad=9)
ax2.legend(fontsize=9.5, frameon=False, loc="upper right", labelcolor=BODY)

# 手动控制边距：左右留 ylabel/刻度空间，上下留 title/xlabel 空间，wspace 给两图之间留足间隔
fig.subplots_adjust(left=0.13, right=0.97, top=0.86, bottom=0.15, wspace=0.55)

fig.savefig("demo_ppt.png", dpi=220, facecolor=BG)
print("saved demo_ppt.png  size:", fig.get_size_inches(), " dpi=220 ->", 
      int(fig.get_size_inches()[0]*220), "x", int(fig.get_size_inches()[1]*220), "px")
