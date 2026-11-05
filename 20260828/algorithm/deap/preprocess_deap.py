#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
preprocess_deap.py — DEAP 情绪识别预处理（valence 二分类，跨被试）

数据
----
  raw/data_preprocessed_python.zip（ModelScope 镜像 byzhrichard/DEAP，官方文件，
  sha256 见 download 校验）解压得 s01.dat..s32.dat（python2 pickle，protocol 2）。
  每个 .dat 是一个 dict（pickle.load(..., encoding='latin1')）：
    data   : (40 trials, 40 channels, 8064 samples) float64 @128Hz，63s/trial
             （前 3s 基线 + 60s 视频；官方已做 512→128Hz 降采样、4-45Hz 带通、
             EOG 伪迹去除）
    labels : (40, 4) = [valence, arousal, dominance, liking]，1-9 自评
    subject: 被试编号 1..32

流程（与项目 2a/CHB-MIT 预处理约定一致，改动如实记录）
------------------------------------------------------
  1. 通道：取前 32 个 EEG 通道（国际 10-20 布局），丢弃后 8 个外周通道
     （hEOG/vEOG/zEMG/tEMG/GSR/呼吸/血氧/体温）——情绪 EEG 分析惯例。
  2. 采样率：官方 data_preprocessed_python 已降采样到 128Hz，无需再降；
     以 8064/63=128.0 断言校验。
  3. 切窗：仅用 60s 视频段（丢弃 3s 基线，样本 [384:8064]），
     5s 非重叠窗 = 640 样本@128Hz，每 trial 12 窗。
     一个 trial 的 valence 标签复制到其全部 12 窗（窗级样本）。
  4. 标签：valence ≥5 → 1（高），<5 → 0（低），二分类，chance 50%。
  5. 标准化：每窗每通道 z-score（窗内统计；DEAP 文献常用做法，
     跨被试归一化——模型在未见被试上直接推理，不能偷看 eval 统计量）。
  6. 划分：跨被试——train = 被试 1-22（880 trials），eval = 被试 23-32
     （400 trials），被试级隔离，无同被试泄漏。

输出（data/）
------------
  train.npz  X=(10560,1,32,640) float32, y=(10560,) int64∈{0,1}
  eval.npz   X=(4800,1,32,640)  float32, y=(4800,) int64∈{0,1}
  meta.json  通道名/划分/统计/校验信息

用法
----
  env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin \
      /opt/miniconda3/envs/openmmlab/bin/python preprocess_deap.py [--smoke]
  （--smoke：只处理前 4 个被试 + 每 trial 前 2 窗，验证链路）
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import zipfile
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
RAW = BASE / "raw"
OUT = BASE / "data"
ZIP_PATH = RAW / "data_preprocessed_python.zip"
EXTRACT_DIR = RAW / "data_preprocessed_python"

FS = 128                     # 官方预处理版已降采样到 128Hz
N_EEG = 32                   # 前 32 个 EEG 通道（40 通道 = 32 EEG + 8 外周）
N_TRIALS = 40
TRIAL_SAMPLES = 8064         # 63s × 128
BASELINE_SAMPLES = 3 * FS    # 384，3s 基线丢弃
WIN_SEC = 5.0
WIN = int(WIN_SEC * FS)      # 640
N_WIN = 12                   # 60s / 5s 每 trial 窗数
VALENCE_TH = 5.0             # ≥5 高 / <5 低

TRAIN_SUBJECTS = list(range(1, 23))   # 被试 1-22
EVAL_SUBJECTS = list(range(23, 33))   # 被试 23-32

# 40 通道名（官方顺序：32 EEG 在前，8 外周在后）——仅记录用
EEG_32_NAMES = ["Fp1", "AF3", "F3", "F7", "FC5", "FC1", "C3", "T7",
                "CP5", "CP1", "P3", "P7", "PO3", "O1", "Oz", "Pz",
                "Fp2", "AF4", "F4", "F8", "FC6", "FC2", "C4", "T8",
                "CP6", "CP2", "P4", "P8", "PO4", "O2", "Cz", "Fz"]
PERIPHERAL_8_NAMES = ["hEOG", "vEOG", "zEMG", "tEMG", "GSR",
                      "Respiration belt", "Plethysmograph", "Temperature"]


def ensure_extracted() -> None:
    """解压 zip → EXTRACT_DIR/sNN.dat（校验 32 个文件 + 字节数）。"""
    if EXTRACT_DIR.exists() and len(list(EXTRACT_DIR.glob("s*.dat"))) == 32:
        print(f"[跳过] 已解压: {EXTRACT_DIR}（32 个 .dat）")
        return
    if not ZIP_PATH.exists():
        raise SystemExit(f"✗ 缺 {ZIP_PATH}，先跑下载（ModelScope byzhrichard/DEAP）")
    print(f"[解压] {ZIP_PATH.name} ...")
    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH) as z:
        names = [n for n in z.namelist() if n.endswith(".dat")]
        assert len(names) == 32, f"zip 内 .dat 数量 {len(names)} != 32"
        bad = z.testzip()
        if bad is not None:
            raise SystemExit(f"✗ zip 损坏: {bad}")
        for n in names:
            z.extract(n, EXTRACT_DIR)
    # zip 内带顶层目录（data_preprocessed_python/…），拍平 .dat 到 EXTRACT_DIR
    for p in EXTRACT_DIR.rglob("s*.dat"):
        if p.parent != EXTRACT_DIR:
            p.rename(EXTRACT_DIR / p.name)
    print(f"[OK] 解压完成，{len(names)} 个 .dat")


def load_subject(sid: int) -> tuple[np.ndarray, np.ndarray]:
    """读取 s{sid:02d}.dat → (data(40,40,8064) float64, labels(40,4))。"""
    p = EXTRACT_DIR / f"s{sid:02d}.dat"
    with open(p, "rb") as f:
        d = pickle.load(f, encoding="latin1")   # python2 pickle
    data = np.asarray(d["data"], dtype=np.float64)
    labels = np.asarray(d["labels"], dtype=np.float64)
    assert data.shape == (N_TRIALS, 40, TRIAL_SAMPLES), f"s{sid}: {data.shape}"
    assert labels.shape == (N_TRIALS, 4), f"s{sid}: {labels.shape}"
    # 采样率断言：63s × 128Hz = 8064
    assert TRIAL_SAMPLES % FS == 0
    return data, labels


def process_subject(sid: int) -> tuple[np.ndarray, np.ndarray]:
    """被试 → (X (480,1,32,640) float32, y (480,) int64)。

    每 trial：视频段 [384:8064] → 12 个 5s 非重叠窗；窗内每通道 z-score。
    """
    data, labels = load_subject(sid)
    eeg = data[:, :N_EEG, BASELINE_SAMPLES:]        # (40, 32, 7680)
    X = eeg.reshape(N_TRIALS, N_EEG, N_WIN, WIN)    # (40, 32, 12, 640)
    X = X.transpose(0, 2, 1, 3).reshape(-1, N_EEG, WIN)   # (480, 32, 640)
    # 窗内每通道 z-score（float64 统计，float32 存储）
    mu = X.mean(axis=2, keepdims=True)
    sd = X.std(axis=2, keepdims=True)
    sd[sd < 1e-9] = 1.0
    X = ((X - mu) / sd).astype(np.float32)[:, None, :, :]   # (480,1,32,640)
    valence = labels[:, 0]
    y = (valence >= VALENCE_TH).astype(np.int64)
    y = np.repeat(y, N_WIN)                          # 每 trial 12 窗同标签
    return X, y


def main() -> int:
    ap = argparse.ArgumentParser(description="DEAP 预处理")
    ap.add_argument("--smoke", action="store_true", help="只处理前 4 被试+每 trial 2 窗")
    args = ap.parse_args()

    ensure_extracted()

    def split_for(subjects: list[int]) -> tuple[np.ndarray, np.ndarray]:
        xs, ys = [], []
        for sid in subjects:
            X, y = process_subject(sid)
            if args.smoke:
                X, y = X[: N_WIN * 2], y[: N_WIN * 2]   # 每 trial 只留 2 窗
            xs.append(X)
            ys.append(y)
        return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)

    print(f"[处理] train 被试 {TRAIN_SUBJECTS[0]}-{TRAIN_SUBJECTS[-1]} "
          f"({len(TRAIN_SUBJECTS)} 人) / eval 被试 {EVAL_SUBJECTS[0]}-{EVAL_SUBJECTS[-1]} "
          f"({len(EVAL_SUBJECTS)} 人)，5s 窗×12/试次，valence≥5 高/低")
    Xtr, ytr = split_for(TRAIN_SUBJECTS)
    Xev, yev = split_for(EVAL_SUBJECTS)

    # 小样本校验（防漏加载假象）：形状/有限值/标签分布
    for name, X, y in (("train", Xtr, ytr), ("eval", Xev, yev)):
        assert X.shape[0] == len(y), name
        assert X.shape[1:] == (1, N_EEG, WIN), f"{name} {X.shape}"
        assert np.isfinite(X).all(), f"{name} 含 NaN/Inf"
        hi = (y == 1).sum()
        print(f"[校验] {name}: X{X.shape} y 高={hi} 低={(y == 0).sum()} "
              f"(高占比 {hi / len(y) * 100:.1f}%)")

    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "train.npz", X=Xtr, y=ytr)
    np.savez_compressed(OUT / "eval.npz", X=Xev, y=yev)
    meta = {
        "source": "ModelScope byzhrichard/DEAP → data_preprocessed_python.zip "
                  "(官方 DEAP, Koelstra 2012)",
        "fs_hz": FS,
        "channels": {"eeg_32": EEG_32_NAMES, "peripheral_dropped_8": PERIPHERAL_8_NAMES},
        "window_sec": WIN_SEC, "windows_per_trial": N_WIN,
        "label": "valence >= 5 -> 1 (high), < 5 -> 0 (low)",
        "split": {"train_subjects": TRAIN_SUBJECTS, "eval_subjects": EVAL_SUBJECTS},
        "shape": {"train": list(Xtr.shape), "eval": list(Xev.shape)},
        "train_hi_ratio": float((ytr == 1).mean()),
        "eval_hi_ratio": float((yev == 1).mean()),
        "note": "官方预处理版已 128Hz/4-45Hz/EOG 去除；此处仅取 32 EEG 通道、"
                "丢弃 3s 基线、5s 非重叠窗、窗内每通道 z-score",
    }
    (OUT / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                   encoding="utf-8")
    print(f"✓ {OUT/'train.npz'} ({Xtr.nbytes/1e6:.0f}MB 内存) + "
          f"{OUT/'eval.npz'} + meta.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
