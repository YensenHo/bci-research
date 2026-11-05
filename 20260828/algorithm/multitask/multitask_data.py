#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""multitask_data.py — 7 任务数据加载 + 统一重采样到 128Hz

硬约束（已实测）：7 任务通道 1~32、采样率 100~256Hz、窗长 205~3000、样本 564~11076。
「通道无关」设计靠模型（reshape B*C 共享滤波器）解决通道数差异；采样率差异靠统一重采样到
128Hz 解决；窗长差异靠全局时间池化解决。

重采样：scipy.signal.resample（FFT）沿时间轴，目标长度 = round(T * 128 / fs)。
重采样后按源语义「逐试次整体标量 z-score」重新归一（每试次跨通道×时间 μ/σ），保证各任务
输入尺度一致（源数据本身就是逐试次 z-score，此处只是重采样后校正尺度漂移）。

产物缓存到 multitask_cache/{task}.npz，二次运行直接读缓存（smoke 不写缓存）。
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from scipy.signal import resample

TARGET_FS = 128

# path 相对 algorithm/ 目录；fs/ch/ncls 为源数据事实（已核对 meta + npz 形状）
TASKS: dict[str, dict] = {
    "2a":       dict(path="data_4class",      fs=250, ch=22, ncls=4),
    "2b":       dict(path="bci2b/data",       fs=250, ch=3,  ncls=2),
    "chbmit":   dict(path="chbmit/data",      fs=256, ch=23, ncls=2),
    "sleep":    dict(path="sleep/data",       fs=100, ch=1,  ncls=5),
    "deap":     dict(path="deap/data",        fs=128, ch=32, ncls=2),
    "eegmmidb": dict(path="eegmmidb/data",    fs=160, ch=21, ncls=4),
    "p300":     dict(path="p300/data",        fs=256, ch=8,  ncls=2),
}


def _resample(X: np.ndarray, fs: float) -> np.ndarray:
    """(N,1,C,T) -> (N,C,T_new)，FFT 重采样到 128Hz 后逐试次整体标量 z-score。"""
    N, _, C, T = X.shape
    T_new = int(round(T * TARGET_FS / fs))
    Xr = resample(X[:, 0], T_new, axis=-1)          # (N,C,T_new)
    mu = Xr.mean(axis=(1, 2), keepdims=True)
    sd = Xr.std(axis=(1, 2), keepdims=True) + 1e-8
    Xr = (Xr - mu) / sd
    return np.ascontiguousarray(Xr, dtype=np.float32)


def load_all(algo_dir: str, cache_dir: str, tasks=None,
             smoke: bool = False, smoke_n: int = 300, smoke_seed: int = 42) -> dict:
    """加载（并缓存）指定任务。返回 dict[task] = dict(Xtr,ytr,Xev,yev,fs,ch,ncls)。"""
    tasks = list(TASKS) if tasks is None else list(tasks)
    os.makedirs(cache_dir, exist_ok=True)
    out: dict = {}
    for name in tasks:
        cfg = TASKS[name]
        cp = os.path.join(cache_dir, f"{name}.npz")
        if os.path.exists(cp) and not smoke:
            z = np.load(cp)
            Xtr, ytr = z["Xtr"], z["ytr"]
            Xev, yev = z["Xev"], z["yev"]
        else:
            base = os.path.join(algo_dir, cfg["path"])
            tr = np.load(os.path.join(base, "train.npz"))
            ev = np.load(os.path.join(base, "eval.npz"))
            Xtr = _resample(tr["X"].astype(np.float32), cfg["fs"])
            ytr = tr["y"].astype(np.int64)
            Xev = _resample(ev["X"].astype(np.float32), cfg["fs"])
            yev = ev["y"].astype(np.int64)
            if not smoke:
                np.savez(cp, Xtr=Xtr, ytr=ytr, Xev=Xev, yev=yev)
        if smoke:
            rng = np.random.RandomState(smoke_seed)
            idx = rng.permutation(len(Xtr))[:min(smoke_n, len(Xtr))]
            idv = rng.permutation(len(Xev))[:min(smoke_n // 2, len(Xev))]
            Xtr, ytr = Xtr[idx], ytr[idx]
            Xev, yev = Xev[idv], yev[idv]
        out[name] = dict(Xtr=Xtr, ytr=ytr, Xev=Xev, yev=yev,
                         fs=cfg["fs"], ch=cfg["ch"], ncls=cfg["ncls"])
    return out


if __name__ == "__main__":
    # 冒烟：打印重采样后各任务形状，验证管线
    import sys
    algo = str(Path(__file__).resolve().parent.parent)   # algorithm/
    cache = str(Path(__file__).resolve().parent / "multitask_cache")
    d = load_all(algo, cache, smoke=True, smoke_n=32)
    for t, v in d.items():
        print(f"{t:9s} ch={v['ch']:2d} ncls={v['ncls']} "
              f"train{v['Xtr'].shape} eval{v['Xev'].shape} "
              f"|Xtr mean={v['Xtr'].mean():+.4f} std={v['Xtr'].std():.4f}")
    sys.exit(0)
