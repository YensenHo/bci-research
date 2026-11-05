#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
preprocess_2b.py — BCIC IV 2b 预处理（.mat 版）：3 电极二分类运动想象

数据源
------
  B{s}T.mat（s=01..09）：内含 session 1-3（带标签）
  B{s}E.mat：内含 session 4-5（**官方无标签**，仅测试用）
  每个 run 结构（scipy loadmat, struct_as_record=False, squeeze_me=True）：
    X (n_samples, 6) float64 —— 列顺序按 MOABB 文档 = [C3, Cz, C4, EOG1, EOG2, EOG3]
    trial (n_trials,) —— cue 起点样本索引；y (n_trials,) ∈ {1,2}（左手/右手）
    fs = 250；run.classes = ['left_hand','right_hand']（通道名不内嵌，按文档顺序取前 3 列）

预处理逻辑（与 2a 的 preprocess_4class.py 完全一致，保证跨数据集可比）：
  - 滤波：0.5–50 Hz 带通 + 50 Hz 陷波（scipy，4 阶 Butterworth，整段滤波再分段）
  - 分段：cue 后 [0.5, 2.5]s = [trial+625, trial+1125]，共 500 样本（@250Hz，与 2a 同窗）
  - 通道：取前 3 列（C3/Cz/C4），丢弃 EOG（与 2a 丢弃 EOG 一致）
  - 每段 z-score 标准化（跨通道+时间整体）
  - 标签：{1,2} → {0,1}（二分类，chance=50%）

划分（与 2a「T 会话训练 → E 会话评估」完全同构，且是 2b 文献标准协议）：
  - train.npz = 全部 9 名被试的 B{s}T.mat（session 1-3，官方训练集，池化）
  - eval.npz  = 全部 9 名被试的 B{s}E.mat（session 4-5，官方测试集，池化）
  - 【实测 2026-08-31】官方 Graz 源分发的 E 文件**内含真实标签**（y∈{1,2}，
    与 2a 不同——2a 的 E 文件官方无标签；2b 的测试标签在赛后公开，文件内已带），
    因此可以直接做官方跨 session 评估，无需退化为 session-CV

输出（bci2b/data/）：
  train.npz   X (n,1,3,500) float32, y (n,) int64∈{0,1}
  eval.npz    同上
  meta.npz    subject/session 逐试次元数据（报告里做被试级分解用）

用法：
  env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin \\
      /opt/miniconda3/envs/openmmlab/bin/python preprocess_2b.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parent / "data"
RAW_DIR = DATA_DIR / "raw"

FS = 250
T_CUE_S = 2.0                  # cue 相对 trial 起点的秒数（2b 与 2a 相同：cue 前 2s 基线）
WIN = (0.5, 2.5)               # cue 后 [0.5, 2.5]s（与 2a 完全同窗）
START = int((T_CUE_S + WIN[0]) * FS)   # 625
STOP = int((T_CUE_S + WIN[1]) * FS)    # 1125
N_SAMPLES = STOP - START               # 500
N_CHANNELS = 3                 # C3, Cz, C4（前 3 列，MOABB 文档顺序）
KEEP_LABELS = {1: 0, 2: 1}     # 左手=1→0, 右手=2→1


def _load_scipy():
    from scipy.io import loadmat  # noqa: PLC0415
    from scipy.signal import butter, filtfilt, iirnotch  # noqa: PLC0415
    return loadmat, butter, filtfilt, iirnotch


def _process_session(mat_path: Path, loadmat, butter, filtfilt, iirnotch
                     ) -> list[tuple[np.ndarray, np.ndarray, int]]:
    """读取一个 .mat（含 3 个 session），返回 [(X, y, session_idx), ...]。"""
    print(f"  处理 {mat_path.name} ...")
    m = loadmat(str(mat_path), struct_as_record=False, squeeze_me=True)
    data = m["data"]

    b_band, a_band = butter(4, [0.5, 50.0], btype="band", fs=FS)
    b_notch, a_notch = iirnotch(50.0, 30.0, fs=FS)

    out = []
    for sess_idx, run in enumerate(data, start=1):
        trial = np.asarray(run.trial).ravel()
        if trial.size == 0:
            continue
        X = np.asarray(run.X, dtype=np.float64)
        y = np.asarray(run.y).ravel()
        # 滤波（全 6 通道滤波，分段时只取前 3 EEG 列；顺序无关）
        X = filtfilt(b_band, a_band, X, axis=0)
        X = filtfilt(b_notch, a_notch, X, axis=0)

        X_list, y_list = [], []
        for onset, label in zip(trial, y):
            label = int(label)
            if label not in KEEP_LABELS:
                continue
            start = int(onset) + START
            stop = int(onset) + STOP
            if stop > X.shape[0]:
                continue
            seg = X[start:stop, :N_CHANNELS]      # (500, 3) 只取 C3/Cz/C4
            X_list.append(seg.T)                  # (3, 500)
            y_list.append(KEEP_LABELS[label])

        if X_list:
            Xs = np.stack(X_list).astype(np.float32)   # (n, 3, 500)
            ys = np.asarray(y_list, dtype=np.int64)
            # 每段 z-score（跨通道+时间整体，与 2a 一致）
            mean = Xs.mean(axis=(1, 2), keepdims=True)
            std = Xs.std(axis=(1, 2), keepdims=True)
            Xs = (Xs - mean) / (std + 1e-8)
            Xs = Xs[:, np.newaxis, :, :]               # (n, 1, 3, 500)
            out.append((Xs, ys, sess_idx))
            print(f"    session {sess_idx}: {len(ys)} 试次"
                  f"（{np.bincount(ys, minlength=2).tolist()} 左/右）")
        else:
            print(f"    session {sess_idx}: 无有效试次")
    return out


def main() -> int:
    loadmat, butter, filtfilt, iirnotch = _load_scipy()

    train_X, train_y, train_meta = [], [], []
    eval_X, eval_y, eval_meta = [], [], []
    for subj in range(1, 10):
        p_t = RAW_DIR / f"B{subj:02d}T.mat"
        p_e = RAW_DIR / f"B{subj:02d}E.mat"
        if not p_t.exists():
            print(f"  ⚠ 缺失 {p_t.name}，跳过")
            continue
        for Xs, ys, sess in _process_session(p_t, loadmat, butter, filtfilt, iirnotch):
            train_X.append(Xs); train_y.append(ys)
            train_meta.append(np.full(len(ys), subj, dtype=np.int64))
        if not p_e.exists():
            print(f"  ⚠ 缺失 {p_e.name}，跳过")
            continue
        for Xs, ys, sess in _process_session(p_e, loadmat, butter, filtfilt, iirnotch):
            eval_X.append(Xs); eval_y.append(ys)
            eval_meta.append(np.full(len(ys), subj, dtype=np.int64))

    if not train_X and not eval_X:
        raise SystemExit(f"✗ 无数据。请确认 RAW_DIR 存在且已运行 download_2b.py：{RAW_DIR}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    def _save(name, Xs, ys, meta):
        X = np.concatenate(Xs, axis=0).astype(np.float32)
        y = np.concatenate(ys, axis=0).astype(np.int64)
        m = np.concatenate(meta, axis=0).astype(np.int64)
        np.savez_compressed(DATA_DIR / name, X=X, y=y)
        print(f"  ✓ {name}: X={X.shape}, y={y.shape}（左/右 = "
              f"{np.bincount(y, minlength=2).tolist()}，共 {len(y)} 试次）")
        return m

    print("\n保存预处理结果 ...")
    train_meta = _save("train.npz", train_X, train_y, train_meta)
    eval_meta = _save("eval.npz", eval_X, eval_y, eval_meta)
    np.savez_compressed(DATA_DIR / "meta.npz",
                        train_subject=train_meta, eval_subject=eval_meta)
    print(f"  ✓ meta.npz（train/eval 逐试次被试编号，报告被试级分解用）")
    print(f"\n✓ 完成，输出目录：{DATA_DIR}")
    print("  下一步：train_eeg_tcnet_hbc_aug_2b.py（relu → LIF 热启动）")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
