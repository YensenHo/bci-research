#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
preprocess_chbmit.py — CHB-MIT 癫痫检测预处理（发作/非发作二分类）

数据
----
  raw/chbNN/chbNN_MM.edf        23 导双极 EEG，256 Hz（EDF 物理单位 µV）
  raw/chbNN/chbNN-summary.txt   发作标注（Seizure Start/End Time，秒）
  raw/annotations.json          由 parse_chbmit_summaries.py 解析的标注（每文件发作区间列表）

流程
----
  1. pyedflib 读取 EDF（23 通道全用，断言 nch==23, sfreq==256）
  2. 预处理：0.5-50Hz 带通（butter 4 阶, filtfilt）→ 每文件每通道 z-score
     （CHB-MIT 文献标准做法；z-score 消除跨患者幅值差异）
  3. 滑窗：2s（512 样本），非重叠（步长=512）。窗口与任一发作区间重叠 ≥1.0s → 正样本
  4. 类别平衡：发作远少于非发作——
       train（chb01-04）：正样本全取，负样本欠采样至 2× 正样本（rng seed=42）
       eval （chb05 跨患者留出）：正样本全取，负样本 1:1 均衡（rng seed=42，chance=50%）
  5. 输出 data/train.npz, data/eval.npz（X=(n,1,23,512) float32, y=(n,) int64）+ data/meta.json

用法
----
  env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin \
      /opt/miniconda3/envs/openmmlab/bin/python preprocess_chbmit.py [--smoke]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import scipy.signal
from scipy.signal import butter, filtfilt

import pyedflib

BASE = Path(__file__).resolve().parent
RAW = BASE / "raw"
OUT = BASE / "data"
SFREQ = 256
WIN_SEC = 2.0
WIN = int(SFREQ * WIN_SEC)          # 512
OVERLAP_POS_SEC = 1.0               # 窗口与发作重叠 ≥1s → 正
TRAIN_PATIENTS = ["chb01", "chb02", "chb03", "chb04"]
EVAL_PATIENTS = ["chb05"]
NEG_POS_RATIO_TRAIN = 2.0
NEG_POS_RATIO_EVAL = 1.0
SEED = 42

BAND = (0.5, 50.0)


def load_annotations() -> dict:
    """读取 raw/annotations.json：{patient: {file: [[start_s, end_s], ...]}}"""
    p = RAW / "annotations.json"
    if not p.exists():
        raise SystemExit("✗ 缺 raw/annotations.json，先跑 parse_chbmit_summaries.py")
    return json.loads(p.read_text(encoding="utf-8"))


def read_edf(path: Path) -> tuple[np.ndarray, list[str]]:
    """读取 EDF：返回 (23, N) 物理单位信号 + 通道名。
    兼容 23 导标准双极 EEG；chb04 部分文件为 24 导（第 24 导为 ECG），
    按标签剔除 ECG，统一为 23 导。断言 256Hz。"""
    f = pyedflib.EdfReader(str(path))
    try:
        nch = f.signals_in_file
        sfreq = f.getSampleFrequency(0)
        assert nch in (23, 24), f"{path.name}: nch={nch} != 23/24"
        assert abs(sfreq - 256) < 1e-6, f"{path.name}: sfreq={sfreq} != 256"
        labels = f.getSignalLabels()
        sigs = [f.readSignal(i) for i in range(nch)]
    finally:
        f.close()
    X = np.stack(sigs, axis=0).astype(np.float64)   # (nch, N)
    if nch == 24:
        keep = [i for i, lab in enumerate(labels)
                if lab.strip().upper() not in ("ECG", "EKG", "ECG1")]
        assert len(keep) == 23, f"{path.name}: 24 导但 ECG 剔除后剩 {len(keep)} 导: {labels}"
        X = X[keep]
        labels = [labels[i] for i in keep]
    assert X.shape[0] == 23, X.shape
    return X, labels


def preprocess_file(X: np.ndarray) -> np.ndarray:
    """带通 0.5-50Hz + 每通道 z-score（按整文件统计）。返回 float32 (23, N)。"""
    X = X.astype(np.float64)
    b, a = butter(4, BAND, btype="bandpass", fs=SFREQ)
    X = filtfilt(b, a, X, axis=1)
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True)
    sd[sd < 1e-9] = 1.0          # 平直通道防护
    X = (X - mu) / sd
    return X.astype(np.float32)


def windowize(X: np.ndarray, seizures: list[list[float]]) -> tuple[np.ndarray, np.ndarray]:
    """2s 非重叠滑窗 → (windows (M,23,512), labels (M,))。
    正样本：窗口与任一发作区间重叠 ≥ OVERLAP_POS_SEC。"""
    n = X.shape[1]
    starts = np.arange(0, n - WIN + 1, WIN)          # 非重叠
    if len(starts) == 0:
        return np.zeros((0, 23, WIN), np.float32), np.zeros(0, np.int64)
    win_idx = starts[:, None] + np.arange(WIN)[None, :]          # (M, 512)
    win_t0 = starts / SFREQ
    win_t1 = (starts + WIN) / SFREQ
    labels = np.zeros(len(starts), np.int64)
    for s, e in seizures:
        ov = np.minimum(win_t1, e) - np.maximum(win_t0, s)
        labels[ov >= OVERLAP_POS_SEC] = 1
    # 只取有标签区分度的窗口（全部窗口都要，正负都留）
    Xw = X[:, win_idx].transpose(1, 0, 2).astype(np.float32)     # (M, 23, 512)
    return Xw, labels


def collect(patients: list[str], ann: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    """处理一组患者的所有（已下载的）文件，返回窗口池 + 统计。"""
    Xs, ys, info = [], [], {}
    n_files = 0
    for pat in patients:
        files = sorted(ann[pat].keys())
        for fn in files:
            p = RAW / pat / fn
            if not p.exists() or p.stat().st_size < 8_000_000:
                print(f"  [跳过未下载] {pat}/{fn}")
                continue
            X, labels = read_edf(p)
            X = preprocess_file(X)
            seiz = ann[pat][fn]
            Xw, y = windowize(X, seiz)
            info[f"{pat}/{fn}"] = {"n_windows": int(len(y)),
                                   "n_pos": int(y.sum()),
                                   "n_neg": int(len(y) - y.sum()),
                                   "seizures": seiz}
            Xs.append(Xw); ys.append(y)
            n_files += 1
    if not Xs:
        raise SystemExit("✗ 没有可用的 EDF 文件（下载未完成？）")
    return np.concatenate(Xs), np.concatenate(ys), info


def balance(X: np.ndarray, y: np.ndarray, neg_ratio: float, seed: int):
    """负样本欠采样到 neg_ratio × 正样本数（rng 确定性）。返回 (Xb, yb, 采样索引)。"""
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    rng = np.random.default_rng(seed)
    n_neg = int(round(len(pos_idx) * neg_ratio))
    sel_neg = rng.choice(neg_idx, size=min(n_neg, len(neg_idx)), replace=False)
    keep = np.sort(np.concatenate([pos_idx, sel_neg]))
    return X[keep], y[keep], keep


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="只处理每个患者第一个可用文件，打印验证统计")
    args = ap.parse_args()

    ann = load_annotations()
    OUT.mkdir(exist_ok=True)
    print(f"预处理：窗 {WIN_SEC}s（{WIN} 样本）非重叠 | 正样本判据：重叠 ≥{OVERLAP_POS_SEC}s "
          f"| 带通 {BAND[0]}-{BAND[1]}Hz + z-score")
    print(f"train 患者 {TRAIN_PATIENTS}（负:正={NEG_POS_RATIO_TRAIN:g}）| eval 患者 "
          f"{EVAL_PATIENTS}（负:正={NEG_POS_RATIO_EVAL:g}）")

    if args.smoke:
        pat = TRAIN_PATIENTS[0]
        fn = [f for f in ann[pat]
              if (RAW / pat / f).exists() and (RAW / pat / f).stat().st_size >= 8_000_000][0]
        X, labels = read_edf(RAW / pat / fn)
        Xp = preprocess_file(X)
        Xw, y = windowize(Xp, ann[pat][fn])
        print(f"[smoke] {pat}/{fn}: nch={X.shape[0]} len={X.shape[1]/SFREQ:.0f}s "
              f"窗口={len(y)} 正={y.sum()} 负={len(y)-y.sum()}")
        print(f"[smoke] 窗口 X 形状 {Xw.shape} dtype {Xw.dtype} | NaN={np.isnan(Xw).any()} "
              f"| 通道均值/标准差: {Xw.mean():.4f}/{Xw.std():.4f}")
        print("[smoke] 通道名:", labels[:3], "...")
        return 0

    # ---- 训练集 chb01-04 ----
    print("\n[1/3] 读取并预处理 chb01-04 ...")
    Xtr, ytr, info_tr = collect(TRAIN_PATIENTS, ann)
    print(f"      原始窗口池: {len(ytr)}（正 {ytr.sum()} / 负 {len(ytr)-ytr.sum()}）")
    Xtr_b, ytr_b, _ = balance(Xtr, ytr, NEG_POS_RATIO_TRAIN, SEED)
    print(f"      欠采样后  : {len(ytr_b)}（正 {ytr_b.sum()} / 负 {len(ytr_b)-ytr_b.sum()}）")

    # ---- 评估集 chb05 ----
    print("[2/3] 读取并预处理 chb05（跨患者留出）...")
    Xev, yev, info_ev = collect(EVAL_PATIENTS, ann)
    print(f"      原始窗口池: {len(yev)}（正 {yev.sum()} / 负 {len(yev)-yev.sum()}）")
    Xev_b, yev_b, _ = balance(Xev, yev, NEG_POS_RATIO_EVAL, SEED)
    print(f"      欠采样后  : {len(yev_b)}（正 {yev_b.sum()} / 负 {len(yev_b)-yev_b.sum()}）")

    # ---- 验证 ----
    for name, X, y in [("train", Xtr_b, ytr_b), ("eval", Xev_b, yev_b)]:
        assert X.shape[1:] == (1, 23, WIN) or X.shape[1:] == (23, WIN), X.shape
        assert not np.isnan(X).any() and not np.isinf(X).any(), f"{name} 含 NaN/Inf"
        assert X.dtype == np.float32 and y.dtype == np.int64
        print(f"[验证] {name}: X={X.shape} {X.dtype} y={y.shape} "
              f"正={y.sum()} 负={len(y)-y.sum()} | 无 NaN ✓")

    # ---- 保存 ----
    np.savez_compressed(OUT / "train.npz", X=Xtr_b[:, None], y=ytr_b)
    np.savez_compressed(OUT / "eval.npz", X=Xev_b[:, None], y=yev_b)
    meta = {
        "dataset": "CHB-MIT subset (chb01-05)",
        "sfreq": SFREQ, "window_sec": WIN_SEC, "window_samples": WIN,
        "step_sec": WIN_SEC, "channels": 23,
        "pos_criterion": f"overlap>={OVERLAP_POS_SEC}s",
        "bandpass_hz": list(BAND), "normalization": "per-file per-channel z-score",
        "train_patients": TRAIN_PATIENTS, "eval_patients": EVAL_PATIENTS,
        "train_neg_pos_ratio": NEG_POS_RATIO_TRAIN, "eval_neg_pos_ratio": NEG_POS_RATIO_EVAL,
        "balance_seed": SEED,
        "train": {"n": int(len(ytr_b)), "pos": int(ytr_b.sum()),
                  "neg": int(len(ytr_b) - ytr_b.sum()),
                  "raw_pool": {"n": int(len(ytr)), "pos": int(ytr.sum())}},
        "eval": {"n": int(len(yev_b)), "pos": int(yev_b.sum()),
                 "neg": int(len(yev_b) - yev_b.sum()),
                 "raw_pool": {"n": int(len(yev)), "pos": int(yev.sum())}},
        "per_file": {**info_tr, **info_ev},
    }
    (OUT / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1),
                                   encoding="utf-8")
    print(f"\n✓ 已保存 {OUT/'train.npz'} {OUT/'eval.npz'} {OUT/'meta.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
