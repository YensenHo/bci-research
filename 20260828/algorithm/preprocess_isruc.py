#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""preprocess_isruc.py — ISRUC-Sleep 预处理：抽取 C3-A2 单通道 → 30s epoch → 5 类。

输出（与 sleep/data_1k 格式一致，可直接复用 sign_check.py / kd_decomposition.py）：
  isruc/data/train.npz : 每类 200 的平衡子采样（1000，复现「少样本」条件）
  isruc/data/eval.npz  : 剩余全量

通道：C3-A2（200Hz → 降采样 100Hz）。标注 value 0=W/1=N1/2=N2/3=N3/5=REM → 0-4，过滤 6=U。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import mne

RAW = Path(__file__).resolve().parent / "isruc" / "raw"
OUT = Path(__file__).resolve().parent / "isruc" / "data"
OUT.mkdir(parents=True, exist_ok=True)

LABEL_MAP = {0: 0, 1: 1, 2: 2, 3: 3, 5: 4}  # W/N1/N2/N3/REM -> 0-4
SR = 200
SR_TARGET = 100
EPOCH_LEN = 30 * SR_TARGET  # 3000
PER_CLASS = 200  # 平衡子采样每类数


def read_events(events_path: Path) -> list[int]:
    values = []
    with open(events_path, encoding="utf-8-sig") as f:
        for i, line in enumerate(f):
            if i == 0:  # header
                continue
            parts = line.strip().split("\t")
            if len(parts) >= 4:
                values.append(int(parts[3]))  # value 列
    return values


def main() -> int:
    X_all, y_all = [], []
    n_subjects = 0
    for edf_path in sorted(RAW.glob("sub-I*_eeg.edf")):
        sub = edf_path.name.split("_")[0]
        events_path = RAW / f"{sub}_events.tsv"
        if not events_path.exists() or events_path.stat().st_size < 1000:
            print(f"跳过 {sub}（无标注）")
            continue
        try:
            raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
        except Exception as e:
            print(f"跳过 {sub}（EDF 读取失败: {e}）")
            continue
        if "C3-A2" not in raw.ch_names:
            print(f"跳过 {sub}（无 C3-A2 通道）")
            continue
        data = raw.get_data()[raw.ch_names.index("C3-A2")]  # (time,)
        data = data[:: (SR // SR_TARGET)]  # 200 -> 100Hz
        # 修复 NEMAR 错误缩放：C3-A2 被 physical range 错误缩放到 ~1e-6，除以 std 归一化到 1（与 sleep 一致）
        sd = data.std()
        if sd > 0:
            data = data / sd
        values = read_events(events_path)

        n_epoch = min(len(data) // EPOCH_LEN, len(values))
        n_kept = 0
        for i in range(n_epoch):
            v = values[i]
            if v not in LABEL_MAP:
                continue
            seg = data[i * EPOCH_LEN:(i + 1) * EPOCH_LEN]
            X_all.append(seg)
            y_all.append(LABEL_MAP[v])
            n_kept += 1
        n_subjects += 1
        print(f"{sub}: {n_epoch} epoch → 保留 {n_kept}（去 U）")

    if not X_all:
        print("无有效数据，退出")
        return 1

    X = np.asarray(X_all, dtype=np.float32).reshape(-1, 1, 1, EPOCH_LEN)
    y = np.asarray(y_all, dtype=np.int64)
    print(f"\n合并 {n_subjects} 受试者，共 {len(y)} 个 epoch")
    print(f"类别分布: {dict(zip(*np.unique(y, return_counts=True)))}")

    # 平衡子采样 train（每类 PER_CLASS），剩余 eval
    rng = np.random.default_rng(42)
    train_idx = []
    for c in range(5):
        idx_c = np.where(y == c)[0]
        n_take = min(PER_CLASS, len(idx_c))
        train_idx.append(rng.choice(idx_c, n_take, replace=False))
    train_idx = np.concatenate(train_idx)
    eval_idx = np.setdiff1d(np.arange(len(y)), train_idx)

    np.savez(OUT / "train.npz", X=X[train_idx], y=y[train_idx])
    np.savez(OUT / "eval.npz", X=X[eval_idx], y=y[eval_idx])
    print(f"\ntrain: {len(train_idx)} 样本（每类 {PER_CLASS}）")
    print(f"eval : {len(eval_idx)} 样本")
    print(f"已保存 {OUT}/train.npz + eval.npz")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
