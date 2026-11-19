#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""preprocess_isruc_subjects.py — ISRUC-Sleep 按被试预处理（供 12 折留一被试 LOOCV）。

从 12 个受试者 sub-I001..sub-I012 的原始 .edf（C3-A2 单通道，200Hz）+ events.tsv，
生成带 subject 标签的数据，让评估能做 leave-one-subject-out。

处理：C3-A2 单通道 → 200→100Hz 降采样 → 0.5-35Hz 带通 → 30s epoch（3000 样本）
      → 5 类映射（W/N1/N2/N3/REM，过滤 U）→ 逐 epoch z-score。

EDF 边界审计：
    各通道每个 EDF 数据记录可能包含不同数量的样本，必须按 header 中每通道 n_samps 计算记录字节数。
    不能用「通道数 × 单样本字节数」估计记录长度。本脚本检查完整记录数并记录核验结果；
    此检查不等同于与上游原始文件逐字节校验。

输出：
  {out}/data_subjects.npz       X=(n,1,1,3000) float32, y=(n,) int64, subject=(n,) int64
                                subject ∈ 0..11 ↔ sub-I001..sub-I012
  {out}/data_subjects_meta.json 每被试 epoch 数 / 类别分布 / 有效时长 / 截断标记

依赖：numpy scipy mne
      安装：pip install mne   （numpy/scipy 通常已有）

LOOCV 用法（评估脚本里）：
    d = np.load('data_subjects.npz'); X, y, subj = d['X'], d['y'], d['subject']
    for s in range(12):
        tr = subj != s; ev = subj == s
        model.fit(X[tr], y[tr]);  acc = score(model, X[ev], y[ev])

用法：
    python preprocess_isruc_subjects.py --raw-dir isruc_raw [--out-dir isruc/data_subjects]

raw-dir 默认依次尝试 isruc_raw、isruc/raw（Mac 旧数据目录）。
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import mne
from scipy.signal import butter, filtfilt, resample_poly

SFREQ_TARGET = 100.0
EPOCH_SEC = 30.0
EPOCH = int(SFREQ_TARGET * EPOCH_SEC)          # 3000
BAND = (0.5, 35.0)                              # 带通（对齐 sleep 预处理）
LABEL_MAP = {0: 0, 1: 1, 2: 2, 3: 3, 5: 4}     # W/N1/N2/N3/REM -> 0-4（过滤 6=U 等）


def read_events_values(tsv_path: Path) -> list[int]:
    """读 events.tsv 的 value 列（第 4 列，0 基 index 3）。"""
    values = []
    with open(tsv_path, encoding="utf-8-sig") as f:
        for i, line in enumerate(f):
            if i == 0:  # header
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 4:
                values.append(int(parts[3]))
    return values


def find_c3_reference(labels: list[str]):
    """优先取 C3-A2；I011 只有 C3-M2，使用并记录该近似乳突参考通道。"""
    for i, lab in enumerate(labels):
        if lab.upper().replace(" ", "") == "C3-A2":
            return i, lab
    for i, lab in enumerate(labels):
        if lab.upper().replace(" ", "") == "C3-M2":
            return i, lab
    return None, None


def process_subject(edf_path: Path, events_path: Path, subj_id: int):
    """读一个受试者 → (X, y, stats) 或 (None, None, stats)。"""
    filesize = edf_path.stat().st_size
    raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)
    labels = list(raw.ch_names)
    n_ch = len(labels)
    idx, channel_used = find_c3_reference(labels)
    if idx is None:
        return None, None, dict(err="no C3-A2 channel", labels=labels)
    sfreq = float(raw.info["sfreq"])
    declared_n = raw.n_times  # header 声明的样本数

    # EDF 各通道每个 data record 的采样数可能不同，按完整 record 算有效长度。
    # 旧算法用 n_channels*2 会把有低采样率辅助通道的文件时长低估。
    extras = raw._raw_extras[0]
    data_offset = int(extras["data_offset"])
    samples_per_record = np.asarray(extras["n_samps"], dtype=np.int64)
    bytes_per_record = int(samples_per_record.sum() * int(extras["dtype_byte"]))
    declared_records = int(extras["n_records"])
    available_records = max(0, (filesize - data_offset) // bytes_per_record)
    valid_records = min(declared_records, available_records)
    valid_n = valid_records * int(samples_per_record[idx])
    truncated = available_records < declared_records
    data = raw.get_data()[idx].astype(np.float64)
    if valid_n < len(data):
        data = data[:valid_n]  # 丢弃 mne 读超的 buffer 残留

    # 200 → 100Hz 抗混叠重采样
    ratio = sfreq / SFREQ_TARGET
    if abs(ratio - 2.0) < 0.01:
        data = resample_poly(data, up=1, down=2)
    elif abs(ratio - 1.0) < 0.01:
        pass
    else:
        return None, None, dict(err=f"unexpected sfreq {sfreq}", n_signal=valid_n)

    # 带通 0.5-35Hz
    b, a = butter(4, BAND, btype="bandpass", fs=SFREQ_TARGET)
    sig_f = filtfilt(b, a, data)

    values = read_events_values(events_path)
    n_epoch = min(len(sig_f) // EPOCH, len(values))
    X_list, y_list = [], []
    for i in range(n_epoch):
        v = values[i]
        if v not in LABEL_MAP:
            continue
        ep = sig_f[i * EPOCH:(i + 1) * EPOCH]
        mu, sd = ep.mean(), ep.std()
        if sd < 1e-9:
            sd = 1.0
        X_list.append(((ep - mu) / sd).astype(np.float32))
        y_list.append(LABEL_MAP[v])
    X = np.stack(X_list).reshape(-1, 1, 1, EPOCH) if X_list else np.zeros((0, 1, 1, EPOCH), np.float32)
    y = np.asarray(y_list, np.int64)
    stats = dict(n_epoch_signal=n_epoch, n_valid=int(len(y)),
                 sfreq=sfreq, n_ch=n_ch,
                 declared_n_samples=int(declared_n),
                 valid_n_samples=int(valid_n),
                 declared_records=declared_records, valid_records=valid_records,
                 valid_seconds=round(valid_n / sfreq, 1),
                 truncated=bool(truncated), channel_used=channel_used,
                 edf_bytes=filesize, events_bytes=events_path.stat().st_size)
    return X, y, stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", type=Path, default=None,
                    help="edf+events.tsv 所在目录（默认依次找 isruc_raw、isruc/raw）")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="输出目录（默认 {raw-dir 同级}/data_subjects）")
    args = ap.parse_args()

    base = Path(__file__).resolve().parent
    raw = args.raw_dir or next((p for p in (base / "isruc_raw", base / "isruc" / "raw") if p.exists()), None)
    if raw is None or not raw.exists():
        print("✗ 找不到 raw 目录，请用 --raw-dir 指定", file=sys.stderr)
        return 1
    out = args.out_dir or (raw.parent / "data_subjects")
    out.mkdir(parents=True, exist_ok=True)

    edfs = sorted(raw.glob("sub-I*_eeg.edf"))
    if not edfs:
        print(f"✗ {raw} 下没有 sub-I*_eeg.edf，先跑 download_isruc_full.py", file=sys.stderr)
        return 1
    print(f"找到 {len(edfs)} 个 edf，输出到 {out}", flush=True)

    Xs, ys, subs = [], [], []
    per_subject = {}
    for edf_path in edfs:
        sub = edf_path.name.split("_")[0]          # sub-I001
        subj_id = int(sub[-3:]) - 1                 # stable IDs even when a subject is skipped
        events_path = raw / f"{sub}_events.tsv"
        if not events_path.exists():
            print(f"  {sub}: 缺 events.tsv，跳过", flush=True)
            per_subject[sub] = dict(err="missing events.tsv")
            continue
        X, y, stats = process_subject(edf_path, events_path, subj_id)
        if X is None:
            print(f"  {sub}: 失败 {stats.get('err')}", flush=True)
            per_subject[sub] = stats
            continue
        Xs.append(X); ys.append(y); subs.append(np.full(len(y), subj_id, np.int64))
        dist = Counter(y.tolist())
        per_subject[sub] = dict(subject_id=subj_id, dist={str(k): int(v) for k, v in sorted(dist.items())}, **stats)
        print(f"  {sub} [id={subj_id}] 有效 {len(y):5d} epoch | 有效时长 {stats['valid_seconds']}s"
              f"{' (截断)' if stats['truncated'] else ''} | 类别 {dict(sorted(dist.items()))}",
              flush=True)

    if not Xs:
        print("✗ 无有效数据", file=sys.stderr)
        return 1
    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0)
    subj = np.concatenate(subs, axis=0)

    np.savez_compressed(out / "data_subjects.npz", X=X, y=y, subject=subj)
    valid_subjects = sorted(per_subject[k]["subject_id"] for k in per_subject
                            if "subject_id" in per_subject[k])
    meta = dict(sfreq_target=SFREQ_TARGET, epoch_sec=EPOCH_SEC,
                channel="C3-A2 (I011 fallback: C3-M2)",
                bandpass=list(BAND), label_map={str(k): v for k, v in LABEL_MAP.items()},
                n_subjects=len(valid_subjects), subject_ids=valid_subjects, n_total=int(len(y)),
                note="按各通道每个 EDF data record 的样本数计算完整记录；该检查不等同于与上游原始文件逐字节校验。",
                per_subject=per_subject,
                loocv_note="subject 数组取值 0..11 对应 sub-I001..sub-I012；LOOCV: train=subject!=s, eval=subject==s")
    (out / "data_subjects_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n[OK] 合并 {len(edfs)} 受试者，共 {len(y)} epoch，shape={X.shape}", flush=True)
    print(f"  类别分布: {dict(sorted(Counter(y.tolist()).items()))}", flush=True)
    print(f"  每被试: { {s: int((subj == s).sum()) for s in range(len(edfs))} }", flush=True)
    print(f"[OK] 已保存 {out}/data_subjects.npz + data_subjects_meta.json", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
