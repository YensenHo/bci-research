#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""快速核对 A03-A06 结构与 A01 一致（形状/取值计数/trial 对齐）。"""
from pathlib import Path
import scipy.io as sio
import numpy as np

RAW = Path("/Users/heyixuan/BCI/20260828/algorithm/p300/data/raw")
for fn in ["A03.mat", "A04.mat", "A05.mat", "A06.mat"]:
    m = sio.loadmat(RAW / fn, squeeze_me=True, struct_as_record=False, chars_as_strings=True)
    d = m["data"]
    X = np.asarray(d.X)
    y = np.asarray(d.y)
    ys = np.asarray(d.y_stim)
    tr = np.asarray(d.trial)
    ch = [str(c) for c in np.asarray(d.channels, dtype=object)]
    yc = dict(zip(*np.unique(y, return_counts=True)))
    ysc = dict(zip(*np.unique(ys, return_counts=True)))
    print(f"{fn}: X={X.shape} ch={ch} y={yc}")
    print(f"   y_stim(非0)={ {k:v for k,v in ysc.items() if k>0} } trial={len(tr)}"
          f" trial_diff均={np.diff(tr).mean():.1f} NaN={np.isnan(X).sum()}")
    # 核对 y==2 段与 y_stim 目标段是否同 onset：对 y==2 找 run，onset 处 y_stim 码
    t2 = np.flatnonzero(y == 2)
    if len(t2):
        splits = np.flatnonzero(np.diff(t2) > 1)
        runs = np.split(t2, splits + 1)
        onsets = np.array([r[0] for r in runs])
        codes_at_onset = ys[onsets]
        same = (codes_at_onset != 0)
        print(f"   y==2 run 数={len(runs)}，onset 处 y_stim 非零占比={same.mean():.3f}"
              f"，码值集={np.unique(codes_at_onset[same])}")
