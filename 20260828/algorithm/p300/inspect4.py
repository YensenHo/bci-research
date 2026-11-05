#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""诊断 trial 边界归属：逐 trial 统计区间内闪烁数与首闪烁相对 marker 偏移。"""
from pathlib import Path
import scipy.io as sio
import numpy as np

RAW = Path("/Users/heyixuan/BCI/20260828/algorithm/p300/data/raw")
m = sio.loadmat(RAW / "A01.mat", squeeze_me=True, struct_as_record=False,
                chars_as_strings=True)
d = m["data"]
ys = np.asarray(d.y_stim)
y = np.asarray(d.y)
tr = np.asarray(d.trial)
N = len(ys)

idx = np.flatnonzero(ys != 0)
splits = np.flatnonzero(np.diff(idx) > 1)
runs = np.split(idx, splits + 1)
onsets = np.array([r[0] for r in runs])
print(f"总 onset={len(onsets)} trial markers={len(tr)}")
# 每个 marker 前 64 样本内的 onsets 归属试验：检查 marker 两侧的 onset 计数
edges = np.append(tr, N + 1)
for k in range(len(tr)):
    lo, hi = tr[k], edges[k + 1]
    n_in = int(((onsets >= lo) & (onsets < hi)).sum())
    # 落在 marker 之前 64 样本内的 onset
    n_pre = int(((onsets >= lo - 64) & (onsets < lo)).sum())
    first = onsets[(onsets >= lo - 64) & (onsets < hi)][:2]
    if n_in != 120 or n_pre != 1:
        print(f"trial {k}: marker={lo} 区间内={n_in} 前64内={n_pre} first={first}")
print("\n前 3 trial 的 onset 序列（相对各自 marker 的偏移）:")
for k in range(3):
    sel = onsets[(onsets >= tr[k] - 64) & (onsets < (tr[k + 1] if k + 1 < 35 else N))]
    print(f"trial {k}: marker={tr[k]} 相对偏移(前3)={sel[:3] - tr[k]}, 段数={len(sel)}")
