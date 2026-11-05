#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
split_by_subject.py — BCIC IV 2a 按被试拆分（within-subject 对标专用）

背景
----
data_4class/train.npz 与 eval.npz 只有 X/y 两个字段，**没有被试索引**
（X (2592,1,22,500)，2592 = 9 人 × 288 试次）。本脚本从原始 .mat 重建
逐被试数据：每人 T 会话 → train、E 会话 → eval，各 288 试次（4 类 × 72）。

预处理与 preprocess_4class.py **逐行一致**（保证与 data_4class 完全可比）：
  - 滤波：0.5–50 Hz 4 阶 Butterworth 带通 + 50 Hz 陷波（iirnotch Q=30），
    对整段连续数据 filtfilt 后再分段；
  - 分段：cue 后 [0.5, 2.5]s = [trial+625, trial+1125]，500 样本；
  - 通道：25 通道取前 22 个 EEG（丢 3 个 EOG）；
  - 四分类映射 {1,2,3,4} → {0,1,2,3}；
  - **每段 z-score：按该试次自身统计量（跨 22 通道 × 500 时间点整体）**，
    即与 data_4class 完全相同的「统一/全局」方式 —— 不按被试统计量归一。
    说明：data_4class 的 z-score 是逐试次的（每个 trial 独立标准化），
    因此逐被试重建沿用同一逐试次方式，而非「每人自己的统计量」。

输出（每被试一个目录）
----
data_4class_by_subject/A01/train.npz  X (288,1,22,500) float32, y (288,) int64
data_4class_by_subject/A01/eval.npz   同上（E 会话）
... A02 .. A09
data_4class_by_subject/manifest.json  每人每会话试次数与类别分布

验证
----
1. 每会话试次数 == 288，四类各 72；
2. 逐被试 train 按 A01..A09 顺序拼接后与 data_4class/train.npz **逐位一致**
   （float32 位级相等），eval 同理 —— 证明重建与既有全局数据完全同源。

用法
----
  env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin \\
      /opt/miniconda3/envs/openmmlab/bin/python split_by_subject.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ALGO_DIR = Path(__file__).resolve().parent
RAW_DIR = Path("/Users/heyixuan/BCI/luckfox-deploy/data/raw")
OUT_DIR = ALGO_DIR / "data_4class_by_subject"
GLOBAL_DIR = ALGO_DIR / "data_4class"

FS = 250
T_CUE_S = 2.0
WIN = (0.5, 2.5)
START = int((T_CUE_S + WIN[0]) * FS)   # 625
STOP = int((T_CUE_S + WIN[1]) * FS)    # 1125
N_SAMPLES = STOP - START               # 500
N_CHANNELS = 22
KEEP_LABELS = {1: 0, 2: 1, 3: 2, 4: 3}
N_CLASSES = 4
EEG_IDX = list(range(22))
SUBJECTS = list(range(1, 10))


def _process_session(mat_path: Path, loadmat, butter, filtfilt, iirnotch
                     ) -> tuple[np.ndarray, np.ndarray] | None:
    """与 preprocess_4class._process_session 完全一致（逐行复刻）。"""
    m = loadmat(str(mat_path), struct_as_record=False, squeeze_me=True)
    data = m["data"]

    b_band, a_band = butter(4, [0.5, 50.0], btype="band", fs=FS)
    b_notch, a_notch = iirnotch(50.0, 30.0, fs=FS)

    X_list, y_list = [], []
    for run in data:
        trial = np.asarray(run.trial).ravel()
        if trial.size == 0:
            continue
        X = np.asarray(run.X, dtype=np.float64)
        y = np.asarray(run.y).ravel()
        X = filtfilt(b_band, a_band, X, axis=0)
        X = filtfilt(b_notch, a_notch, X, axis=0)

        for onset, label in zip(trial, y):
            label = int(label)
            if label not in KEEP_LABELS:
                continue
            start = int(onset) + START
            stop = int(onset) + STOP
            if stop > X.shape[0]:
                continue
            seg = X[start:stop, EEG_IDX]
            X_list.append(seg.T)
            y_list.append(KEEP_LABELS[label])

    if not X_list:
        return None
    X = np.stack(X_list).astype(np.float32)
    y = np.asarray(y_list, dtype=np.int64)

    # 每段 z-score（跨通道+时间整体）——与 data_4class 完全一致
    mean = X.mean(axis=(1, 2), keepdims=True)
    std = X.std(axis=(1, 2), keepdims=True)
    X = (X - mean) / (std + 1e-8)
    X = X[:, np.newaxis, :, :]
    return X, y


def main() -> int:
    from scipy.io import loadmat
    from scipy.signal import butter, filtfilt, iirnotch

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {}

    for subj in SUBJECTS:
        subj_dir = OUT_DIR / f"A{subj:02d}"
        subj_dir.mkdir(parents=True, exist_ok=True)
        manifest[f"A{subj:02d}"] = {}
        for sess, tag in (("T", "train"), ("E", "eval")):
            p = RAW_DIR / f"A{subj:02d}{sess}.mat"
            res = _process_session(p, loadmat, butter, filtfilt, iirnotch)
            if res is None:
                raise SystemExit(f"✗ {p.name} 无有效试次")
            X, y = res
            counts = np.bincount(y, minlength=N_CLASSES).tolist()
            manifest[f"A{subj:02d}"][tag] = {
                "n_trials": int(len(y)),
                "class_counts": counts,
                "X_shape": list(X.shape),
            }
            np.savez_compressed(subj_dir / f"{tag}.npz",
                                X=X.astype(np.float32), y=y.astype(np.int64))
            print(f"  ✓ A{subj:02d} {sess}->{tag}: X={X.shape} "
                  f"类别分布{counts}（共 {len(y)} 试次）")

    # ---- 验证 1：每会话 288 试次、四类各 72 ----
    print("\n[验证 1] 每会话 288 试次、四类均衡 72×4：")
    ok1 = True
    for subj in SUBJECTS:
        for tag in ("train", "eval"):
            d = np.load(OUT_DIR / f"A{subj:02d}" / f"{tag}.npz")
            counts = np.bincount(d["y"], minlength=4).tolist()
            good = len(d["y"]) == 288 and counts == [72, 72, 72, 72]
            ok1 &= good
            print(f"  A{subj:02d} {tag}: n={len(d['y'])} 分布={counts} "
                  f"{'✓' if good else '✗'}")

    # ---- 验证 2：与全局 data_4class 逐位一致（拼接序 A01..A09）----
    print("\n[验证 2] 与 data_4class/train.npz, eval.npz 逐位一致（float32 位级）：")
    ok2 = True
    for tag in ("train", "eval"):
        parts = [np.load(OUT_DIR / f"A{s:02d}" / f"{tag}.npz")
                 for s in SUBJECTS]
        X = np.concatenate([p["X"] for p in parts], axis=0)
        y = np.concatenate([p["y"] for p in parts], axis=0)
        g = np.load(GLOBAL_DIR / f"{tag}.npz")
        same_x = np.array_equal(X, g["X"])
        same_y = np.array_equal(y, g["y"])
        ok2 &= same_x and same_y
        print(f"  {tag}.npz: 重建 X={X.shape} vs 全局 {g['X'].shape} | "
              f"X 位级一致={same_x} y 一致={same_y}")

    (OUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nmanifest 已写入：{OUT_DIR / 'manifest.json'}")
    print(f"{'✓ 全部验证通过' if (ok1 and ok2) else '✗ 存在验证失败'}")
    return 0 if (ok1 and ok2) else 1


if __name__ == "__main__":
    sys.exit(main())
