#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""探查 BNCI 008-2014 .mat：值分布 / 闪烁 run 结构 / trial 分段 / fs 推断。"""
from pathlib import Path
import scipy.io as sio
import numpy as np

RAW = Path("/Users/heyixuan/BCI/20260828/algorithm/p300/data/raw")

for fn in ["A01.mat", "A02.mat"]:
    f = RAW / fn
    m = sio.loadmat(f, squeeze_me=True, struct_as_record=False, chars_as_strings=True)
    d = m["data"]
    X = np.asarray(d.X, np.float64)
    y = np.asarray(d.y)
    ys = np.asarray(d.y_stim)
    tr = np.asarray(d.trial)
    ch = [str(c) for c in np.asarray(d.channels, dtype=object)]
    print(f"\n{'='*70}\n{fn} bytes={f.stat().st_size}")
    print(f"X={X.shape} {X.dtype} | ch={ch}")
    print(f"y={y.shape} {y.dtype} | y_stim={ys.shape} {ys.dtype} | trial={tr.shape}")
    print(f"X 全局: min={X.min():.1f} max={X.max():.1f} mean={X.mean():.2f} std={X.std():.2f} NaN={np.isnan(X).sum()}")
    print(f"y 取值: {np.unique(y, return_counts=True)}")
    print(f"y_stim 取值计数: {dict(zip(*np.unique(ys, return_counts=True)))}")
    print(f"trial = {tr}")
    # 闪烁 run 结构（y_stim != 0 的连续段）
    nz = ys != 0
    # 段边界
    idx = np.flatnonzero(nz)
    if len(idx):
        splits = np.flatnonzero(np.diff(idx) > 1)
        runs = np.split(idx, splits + 1)
        run_lens = np.array([len(r) for r in runs])
        run_vals = np.array([ys[r[0]] for r in runs])
        gaps = np.diff(np.array([r[0] for r in runs]))
        print(f"闪烁段数={len(runs)} len: min={run_lens.min()} max={run_lens.max()} "
              f"mean={run_lens.mean():.1f} | 各段码值分布: {dict(zip(*np.unique(run_vals, return_counts=True)))}")
        print(f"onset 间隔(样本): min={gaps.min()} max={gaps.max()} mean={gaps.mean():.1f}")
        # 目标段（y==1）的码值分布 —— 每字符一个目标码？
        tgt_idx = np.flatnonzero(y == 1)
        if len(tgt_idx):
            tgt_runs_split = np.flatnonzero(np.diff(tgt_idx) > 1)
            tgt_runs = np.split(tgt_idx, tgt_runs_split + 1)
            print(f"目标段数={len(tgt_runs)} 段长 min/max/mean = "
                  f"{min(len(r) for r in tgt_runs)}/{max(len(r) for r in tgt_runs)}/"
                  f"{np.mean([len(r) for r in tgt_runs]):.1f}")
            # 目标段的码是否单一（同一 row/col 重复闪烁 = 目标重复刺激）
            codes = [np.unique(ys[r]) for r in tgt_runs]
            single = sum(1 for c in codes if len(c) == 1)
            print(f"目标段码值单一占比: {single}/{len(codes)}")
            sample_codes = np.unique([int(c[0]) for c in codes if len(c) == 1])
            print(f"目标段出现过的刺激码: {sample_codes}")
    # 相邻闪烁 onset 间隔模式：前 24 个 onset 间隔
    onsets = np.array([r[0] for r in runs])[:30] if len(idx) else np.array([])
    print(f"前 30 个闪烁 onset 样本索引: {onsets}")
    # 若 175ms 闪烁 / 125ms 间隔等已知协议——先不猜，把统计留给人工判读
