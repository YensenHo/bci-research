#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
preprocess_sleep.py — Sleep-EDF Expanded（sleep-cassette SC 系列）睡眠分期预处理

数据（PhysioNet sleep-edfx 1.0.0，sleep-cassette/）
----------------------------------------------------
  SC4xxxE0-PSG.edf        整夜 PSG：EEG Fpz-Cz + EEG Pz-Oz 双导（100 Hz），另有
                          EOG/EMG/呼吸/体温/事件通道
  SC4xxxEC-Hypnogram.edf  标注文件：EDF 注释通道，每条注释 = 一个 30s epoch 的
                          睡眠分期（Rechtschaffen & Kales 旧分期：W/1/2/3/4/R，
                          AASM 映射：3+4→N3）

本脚本只取 **EEG Fpz-Cz 单通道**（TinySleepNet 等文献标准做法），
30s epoch = 3000 样本（100 Hz），R&K → AASM 5 分类：
  W→0（清醒）, 1→1（N1 浅睡）, 2→2（N2）, 3+4→3（N3 深睡）, R→4（REM）
异常标注（Movement time / Sleep stage ? 等）剔除，如实记录。

划分（跨被试）：
  train = subjects 0-3（SC4001, SC4011, SC4021, SC4031，4 例/4 夜）
  eval  = subjects 4-5（SC4041, SC4051，2 例/2 夜，只测一次）
  注：每例取第 1 夜（E0）——HF 镜像 zip 只有 E0 夜，PhysioNet 直连限速 ~7KB/s
  无法整包下载；zip 条目字节数与 PhysioNet 列表页逐字节一致（见 download_manifest.json）。

预处理：0.5-35 Hz 带通（butter 4 阶 filtfilt）+ 逐 epoch z-score（TinySleepNet 式）。

输出 data/train.npz, data/eval.npz（X=(n,1,1,3000) float32, y=(n,) int64）
     + data/meta.json（每文件统计 / 类别分布 / 划分记录）

用法
----
  env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin \\
      /opt/miniconda3/envs/openmmlab/bin/python preprocess_sleep.py [--smoke]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pyedflib
from scipy.signal import butter, filtfilt

BASE = Path(__file__).resolve().parent
RAW = BASE / "raw"
OUT = BASE / "data"
SFREQ = 100.0
EPOCH_SEC = 30.0
EPOCH = int(SFREQ * EPOCH_SEC)          # 3000
CHANNEL = "Fpz-Cz"                       # 单通道选择（TinySleepNet 做法）
BAND = (0.5, 35.0)                       # 带通（睡眠分期文献常用）

# 6 例 = subjects 0-5，各 1 夜（E0）；train 0-3（4 例），eval 4-5（2 例）——跨被试
SUBJECTS = {
    "subj0": ["SC4001"],
    "subj1": ["SC4011"],
    "subj2": ["SC4021"],
    "subj3": ["SC4031"],
    "subj4": ["SC4041"],
    "subj5": ["SC4051"],
}
TRAIN_SUBJECTS = ["subj0", "subj1", "subj2", "subj3"]
EVAL_SUBJECTS = ["subj4", "subj5"]

# R&K 标注 → AASM 5 分类；None = 剔除
STAGE_MAP = {
    "sleep stage w": 0,     # Wake
    "sleep stage 1": 1,     # N1
    "sleep stage 2": 2,     # N2
    "sleep stage 3": 3,     # N3（3+4 合并）
    "sleep stage 4": 3,     # N3
    "sleep stage r": 4,     # REM
}
DROP_STAGES = {"movement time", "sleep stage ?", "unknown", ""}


def find_psg(rec: str) -> Path:
    """找 PSG 文件（rec 如 SC4001 → SC4001E0-PSG.edf）。"""
    for p in sorted(RAW.glob(f"{rec}*-PSG.edf")):
        return p
    raise FileNotFoundError(f"✗ 缺 PSG: {rec}（先跑 download_sleep.sh）")


def find_hyp(rec: str) -> Path:
    for p in sorted(RAW.glob(f"{rec}*-Hypnogram.edf")):
        return p
    raise FileNotFoundError(f"✗ 缺 Hypnogram: {rec}")


def read_psg_channel(path: Path, want: str = CHANNEL) -> tuple[np.ndarray, float, list[str]]:
    """读 PSG EDF 指定通道物理信号。返回 (signal float64 µV, sfreq, all_labels)。"""
    f = pyedflib.EdfReader(str(path))
    try:
        labels = [l.strip().upper() for l in f.getSignalLabels()]
        nch = f.signals_in_file
        sfreqs = [f.getSampleFrequency(i) for i in range(nch)]
        idx = next((i for i, lab in enumerate(labels) if want.upper() in lab), None)
        if idx is None:
            raise ValueError(f"{path.name}: 找不到通道含 {want!r}，实际: {labels}")
        sig = f.readSignal(idx)
        sfreq = float(sfreqs[idx])
        nsamp = f.getNSamples()[idx]
    finally:
        f.close()
    assert len(sig) == nsamp
    return np.asarray(sig, np.float64), sfreq, labels


def read_hypnogram(path: Path) -> list[tuple[float, float, str]]:
    """读 Hypnogram EDF 注释通道 → [(onset_s, duration_s, stage_str), ...]。

    Sleep-EDF 的 Hypnogram 覆盖从记录开始起的 24h 窗口，每个注释 = 连续
    同分期的 30s epoch 合并（duration = 30×k）；首条注释通常是覆盖熄灯前
    全程的 "Sleep stage W"。onsets 均为 30 的整数倍（已与 MNE 对照验证）。
    """
    f = pyedflib.EdfReader(str(path))
    try:
        starts, durations, descriptions = f.readAnnotations()
    finally:
        f.close()
    anns = []
    for t0, dur, desc in zip(starts, durations, descriptions):
        anns.append((float(t0), float(dur), str(desc).strip().lower()))
    return anns


def expand_epoch_labels(anns: list[tuple[float, float, str]], n_epoch: int
                        ) -> tuple[np.ndarray, dict]:
    """把合并注释展开为逐 30s epoch 标签（epoch i ↔ 记录开始后 [30i, 30(i+1))s）。

    返回 (y (n_epoch,), -1=无有效分期剔除, stats)。首条长 W 注释也如实保留
    （整夜记录含睡前/睡后清醒，W 类占比高是数据事实，用类别权重处理）。
    """
    y = np.full(n_epoch, -1, np.int64)
    dropped_epochs = 0
    for t0, dur, desc in anns:
        cls = STAGE_MAP.get(desc)
        e0 = int(round(t0 / EPOCH_SEC))
        e1 = int(round((t0 + dur) / EPOCH_SEC))
        e0 = max(e0, 0)
        e1 = min(e1, n_epoch)
        if e1 <= e0:
            continue
        if cls is None:
            dropped_epochs += e1 - e0      # Movement time / ? / unknown 剔除
            y[e0:e1] = -1
        else:
            y[e0:e1] = cls
    stats = dict(dropped_epochs=int(dropped_epochs))
    return y, stats


def preprocess_epochs(sig: np.ndarray, anns: list[tuple[float, float, str]]
                      ) -> tuple[np.ndarray, np.ndarray, dict]:
    """切片 30s epoch + 标注展开 + 预处理。返回 (X (n,1,1,3000) float32, y (n,) int64, stats)。"""
    n_epoch = len(sig) // EPOCH
    stats = dict(n_signal_samples=len(sig), n_annotations=len(anns), n_epoch_signal=n_epoch)
    y_all, st = expand_epoch_labels(anns, n_epoch)
    stats.update(st)
    # 带通
    b, a = butter(4, BAND, btype="bandpass", fs=SFREQ)
    sig_f = filtfilt(b, a, sig)
    keep = np.where(y_all >= 0)[0]
    X = np.zeros((len(keep), 1, 1, EPOCH), np.float32)
    y = y_all[keep]
    for j, i in enumerate(keep):
        ep = sig_f[i * EPOCH:(i + 1) * EPOCH]
        mu, sd = ep.mean(), ep.std()
        if sd < 1e-9:
            sd = 1.0
        X[j, 0, 0, :] = ((ep - mu) / sd).astype(np.float32)   # 逐 epoch z-score
    stats["n_valid"] = len(keep)
    return X, y, stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="冒烟：每被试只取第 1 夜")
    args = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    per_file = {}
    split_data: dict[str, list] = {"train": [], "eval": []}

    for subj in TRAIN_SUBJECTS + EVAL_SUBJECTS:
        split = "train" if subj in TRAIN_SUBJECTS else "eval"
        recs = SUBJECTS[subj][:1] if args.smoke else SUBJECTS[subj]
        for rec in recs:
            psg_p, hyp_p = find_psg(rec), find_hyp(rec)
            sig, sfreq, labels = read_psg_channel(psg_p)
            if abs(sfreq - SFREQ) > 1e-6:
                print(f"  [警告] {psg_p.name}: sfreq={sfreq} ≠ 100，跳过")
                continue
            anns = read_hypnogram(hyp_p)
            X, y, st = preprocess_epochs(sig, anns)
            st.update(psg_bytes=psg_p.stat().st_size, hyp_bytes=hyp_p.stat().st_size,
                      channel_labels=labels)
            per_file[f"{subj}/{rec}"] = st
            split_data[split].append((X, y, subj, rec))
            print(f"  {rec:8s} [{subj}] 有效 {len(y):5d} epoch | 信号 {st['n_signal_samples']} "
                  f"标注 {st['n_annotations']} 丢弃 {st['dropped_epochs']} | {st['psg_bytes']/1e6:.1f}MB")

    meta = {"sfreq": SFREQ, "epoch_sec": EPOCH_SEC, "channel": CHANNEL,
            "bandpass": list(BAND), "stage_map": {k: v for k, v in STAGE_MAP.items()},
            "drop_stages": sorted(DROP_STAGES), "per_file": per_file}
    for split in ("train", "eval"):
        parts = split_data[split]
        X = np.concatenate([p[0] for p in parts], axis=0)
        y = np.concatenate([p[1] for p in parts], axis=0)
        dist = Counter(y.tolist())
        meta[f"{split}_subjects"] = [p[2] for p in parts]
        meta[f"{split}_recordings"] = [p[3] for p in parts]
        meta[f"{split}_n"] = int(len(y))
        meta[f"{split}_dist"] = {str(k): int(v) for k, v in sorted(dist.items())}
        np.savez_compressed(OUT / f"{split}.npz", X=X, y=y)
        tot = len(y)
        print(f"\n[{split}] n={tot}  shape={X.shape}  类别分布: " +
              ", ".join(f"{k}:{v} ({v/tot*100:.1f}%)" for k, v in sorted(dist.items())))

    # 类别权重（train，逆频率，供加权损失）
    dist = Counter()
    for X, y, _, _ in split_data["train"]:
        dist.update(y.tolist())
    n = sum(dist.values())
    w = np.array([n / dist.get(c, 1) for c in range(5)], np.float64)
    w = w / w.mean()
    meta["class_weights"] = {str(k): float(v) for k, v in enumerate(w)}
    meta["smoke"] = bool(args.smoke)
    (OUT / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                   encoding="utf-8")
    print(f"\n✓ {OUT/'train.npz'} {OUT/'eval.npz'} {OUT/'meta.json'}")
    print(f"  类别权重(逆频率归一化): {np.round(w, 3).tolist()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
