#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
preprocess_4class.py — BCIC IV 2a 预处理（.mat 版）：22 通道四分类运动想象

与 luckfox-deploy/data/preprocess.py 的区别（本文件为升级版，不改原文件）：
  - 通道：取全部 22 个 EEG 通道（丢弃 3 个 EOG），即 CH_25 的前 22 项。
    原 preprocess.py 只取 8 个运动区电极（MOTOR_8）。
  - 类别：四分类 {1=左手, 2=右手, 3=双脚, 4=舌头} → 映射为 {0,1,2,3}。
    原 preprocess.py 只保留 {1,2} 二分类。

复用逻辑（与原 preprocess.py 完全一致，保证可比性）：
  - 滤波：0.5–50 Hz 带通 + 50 Hz 陷波（scipy，4 阶 Butterworth）
  - 分段：cue 后 [0.5, 2.5]s = [trial+625, trial+1125]，共 500 样本
  - 每段 z-score 标准化（跨通道+时间整体）

数据源：
  .mat 结构（struct_as_record=False, squeeze_me=True）：
    m['data'] 9 元素数组：data[3..8] 是 6 个 MI run，每个含
    X(96735,25)、trial(48,)、y(48,)、fs=250；y=1~4。
  25 通道 = 22 EEG + 3 EOG（CH_25 顺序见下）。

输出（写入本目录下 data_4class/）：
  data_4class/train.npz  训练集（A{nn}T 会话），X (n,1,22,500) float32, y (n,) int64∈{0,1,2,3}
  data_4class/eval.npz   评估集（A{nn}E 会话）

用法：
  env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin \\
      /opt/miniconda3/envs/openmmlab/bin/python preprocess_4class.py --subjects 1 2 3 4 5 6 7 8 9
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

# ---- 路径：数据源读 luckfox-deploy/data/raw，输出写本目录 data_4class/ ----
ALGO_DIR = Path(__file__).resolve().parent
RAW_DIR = Path("/Users/heyixuan/BCI/luckfox-deploy/data/raw")
PROCESSED_DIR = ALGO_DIR / "data_4class"

FS = 250
T_CUE_S = 2.0                  # cue 相对 trial 起点的秒数
WIN = (0.5, 2.5)               # cue 后 [0.5, 2.5]s
START = int((T_CUE_S + WIN[0]) * FS)   # 625
STOP = int((T_CUE_S + WIN[1]) * FS)    # 1125
N_SAMPLES = STOP - START               # 500
N_CHANNELS = 22

# 25 通道顺序（22 EEG + 3 EOG）——与原 preprocess.py 第 57-63 行一致
CH_25 = ["Fz", "FC3", "FC1", "FCz", "FC2", "FC4",
         "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
         "CP3", "CP1", "CPz", "CP2", "CP4",
         "P1", "Pz", "P2", "POz",
         "EOG1", "EOG2", "EOG3"]
# 22 个 EEG 通道 = 前 22 项（丢弃最后 3 个 EOG）
EEG_22 = CH_25[:22]
EEG_IDX = list(range(22))

# 四分类映射：1=左手, 2=右手, 3=双脚, 4=舌头 → 0,1,2,3
KEEP_LABELS = {1: 0, 2: 1, 3: 2, 4: 3}
N_CLASSES = 4


def _load_scipy():
    try:
        from scipy.io import loadmat  # noqa: PLC0415
        from scipy.signal import butter, filtfilt, iirnotch  # noqa: PLC0415
    except ImportError as exc:
        raise SystemExit(
            "✗ 未安装 scipy。请执行：\n"
            "    /opt/miniconda3/envs/openmmlab/bin/pip install scipy\n"
            f"  原始错误：{exc}"
        ) from exc
    return loadmat, butter, filtfilt, iirnotch


def _process_session(mat_path: Path, loadmat, butter, filtfilt, iirnotch
                     ) -> tuple[np.ndarray, np.ndarray] | None:
    """读取一个会话 .mat，返回 (X, y) 或 None（22 EEG 通道 + 四分类）。"""
    print(f"  处理 {mat_path.name} ...")
    m = loadmat(str(mat_path), struct_as_record=False, squeeze_me=True)
    data = m["data"]

    # 带通 0.5–50 Hz + 50 Hz 陷波（对整段连续数据滤波，再分段，避免边界效应）
    b_band, a_band = butter(4, [0.5, 50.0], btype="band", fs=FS)
    b_notch, a_notch = iirnotch(50.0, 30.0, fs=FS)

    X_list, y_list = [], []
    for run in data:
        trial = np.asarray(run.trial).ravel()
        if trial.size == 0:
            continue                      # 跳过无试次的基线 run
        X = np.asarray(run.X, dtype=np.float64)   # (n_samples, 25)
        y = np.asarray(run.y).ravel()

        # 滤波（全 25 通道，EOG 丢弃前先滤，顺序无关）
        X = filtfilt(b_band, a_band, X, axis=0)
        X = filtfilt(b_notch, a_notch, X, axis=0)

        for onset, label in zip(trial, y):
            label = int(label)
            if label not in KEEP_LABELS:
                continue                  # 只保留四类 MI
            start = int(onset) + START
            stop = int(onset) + STOP
            if stop > X.shape[0]:
                continue                  # 越界段丢弃
            seg = X[start:stop, EEG_IDX]          # (500, 22) 只取 22 EEG 通道
            X_list.append(seg.T)                  # (22, 500)
            y_list.append(KEEP_LABELS[label])

    if not X_list:
        return None
    X = np.stack(X_list).astype(np.float32)       # (n, 22, 500)
    y = np.asarray(y_list, dtype=np.int64)

    # 每段 z-score（跨通道+时间整体）
    mean = X.mean(axis=(1, 2), keepdims=True)
    std = X.std(axis=(1, 2), keepdims=True)
    X = (X - mean) / (std + 1e-8)
    X = X[:, np.newaxis, :, :]                    # (n, 1, 22, 500)
    return X, y


def main() -> int:
    parser = argparse.ArgumentParser(description="BCIC IV 2a 预处理（.mat → 22ch 四分类 npz）")
    parser.add_argument("--subjects", type=int, nargs="+", default=list(range(1, 10)))
    args = parser.parse_args()

    loadmat, butter, filtfilt, iirnotch = _load_scipy()
    subjects = sorted(set(args.subjects))

    train_X, train_y, eval_X, eval_y = [], [], [], []
    for subj in subjects:
        for sess, buckets in (("T", (train_X, train_y)), ("E", (eval_X, eval_y))):
            p = RAW_DIR / f"A{subj:02d}{sess}.mat"
            if not p.exists():
                print(f"  ⚠ 缺失 {p.name}，跳过")
                continue
            res = _process_session(p, loadmat, butter, filtfilt, iirnotch)
            if res is not None:
                buckets[0].append(res[0]); buckets[1].append(res[1])

    if not train_X and not eval_X:
        raise SystemExit(f"✗ 无数据。请确认 RAW_DIR 存在：{RAW_DIR}")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    def _save(name, Xs, ys):
        if not Xs:
            print(f"  ⚠ {name}: 无数据，跳过")
            return
        X = np.concatenate(Xs, axis=0)
        y = np.concatenate(ys, axis=0)
        np.savez_compressed(PROCESSED_DIR / name, X=X.astype(np.float32), y=y.astype(np.int64))
        counts = np.bincount(y, minlength=N_CLASSES)
        print(f"  ✓ {name}: X={X.shape}, y={y.shape}")
        print(f"      类别分布[左/右/双脚/舌头] = {counts.tolist()}  "
              f"(共 {counts.sum()} 试次)")

    print("\n保存预处理结果 ...")
    _save("train.npz", train_X, train_y)
    _save("eval.npz", eval_X, eval_y)
    print(f"\n✓ 完成，输出目录：{PROCESSED_DIR}")
    print("  下一步：train_eegnet_4class.py / train_lif_4class.py")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
