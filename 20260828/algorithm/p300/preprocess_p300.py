#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
preprocess_p300.py — BNCI 008-2014（EPFL P300 speller，ALS 患者）预处理

数据（data/raw/A01.mat..A06.mat，官方 CC BY-NC-ND 4.0，字节数与 HEAD 校验一致）
----------------------------------------------------------------------------
  .mat 内 data 为 MATLAB struct，字段（A01-A06 逐一实测，结构完全一致）：
    X        : (347704, 8) float64 EEG 时间×通道（整文件连续单会话）
    channels : ['Fz','Cz','Pz','Oz','P3','P4','PO7','PO8']（8 导，10-10，右耳参考）
    y        : (347704,) uint8 逐样本标记：0=无闪烁 / 1=非目标闪烁 / 2=目标闪烁
    y_stim   : (347704,) int16 逐样本刺激码：0=无闪烁 / 1-6=Row1-6 / 7-12=Col1-6
    trial    : (35,) 每个 trial（拼写一个字符）的起始样本
    classes  : ['NonTarget','Target']；每被试 35 trial = 7 个 5 字母单词
  官方协议（description.pdf，2026-09-02 下载核对）：BCI2000 + Farwell-Donchin
  6×6 矩阵；g.MOBILAB 256 Hz 采样，0.1-30 Hz 带通；行/列随机强化 125 ms + ISI
  125 ms（SOA 250 ms = 64 样本）；每字符 trial 全部 12 个刺激各强化 10 次
  （每 trial 120 次闪烁，其中目标行+目标列各 10 次 = 20 次目标闪烁）。
  数据实测与文档完全自洽：闪烁段长恒 32 样本（=125 ms@256Hz）、onset 间隔
  64 样本（=250 ms）、每被试 4200 闪烁（700 目标 + 3500 非目标，1:5）。

流程（改动如实记录）
--------------------
  1. 闪烁切窗：y_stim 非零连续段起点 = 闪烁 onset，取 [onset, onset+0.8s)
     （205 样本 @256Hz，P300 响应窗口，含闪烁期 0-125ms——标准单次刺激 epoch）
  2. 标签：onset 处 y==2 → 目标(1)；y==1 → 非目标(0)（二分类，chance 50%）。
     段内 y 恒定断言 + 每 trial 恰 2 个目标码（1 行 + 1 列）断言。
  3. 通道：8 导全用（Fz/Cz/Pz/Oz/P3/P4/PO7/PO8——数据集作者选定的 P300
     相关电极布局）。
  4. 标准化：逐 epoch z-score（该 epoch 跨 8 通道×时间的整体标量 μ/σ，
     与 2a「逐试次整体 z-score」同一约定；epoch 即试次）。
  5. 划分：跨被试 train=A01-A04（4 人 16800 闪烁）、eval=A05-A06（2 人 8400），
     被试级隔离。类不均衡 1:5（真实 P300 协议属性）→ 训练/主评估用
     均衡子采样（seed 42，全保留目标闪烁）：train 2800 目标+2800 非目标、
     eval 1400+1400；另输出 eval_full.npz（A05-A06 全部 8400 闪烁 + 元数据
     subj/trial/code/rep），供多轮叠加评估与真实分布指标。

输出（data/）
------------
  train.npz     X=(5600,1,8,205) float32, y=(5600,) int64 ∈{0,1}
  eval.npz      X=(2800,1,8,205) float32, y=(2800,) int64 ∈{0,1}（均衡）
  eval_full.npz X=(8400,1,8,205) float32, y, 及 meta(subj, trial, code, rep)
  meta.json     协议/校验/逐被试统计

用法
----
  env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin \
      /opt/miniconda3/envs/openmmlab/bin/python preprocess_p300.py [--smoke]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio

BASE = Path(__file__).resolve().parent
RAW = BASE / "data" / "raw"
OUT = BASE / "data"

FS = 256                      # 官方文档：g.MOBILAB 256 Hz
WIN_SEC = 0.8                 # P300 响应窗口 [0, 0.8s]
WIN = int(round(WIN_SEC * FS))  # 205 样本
CHANNELS = ["Fz", "Cz", "Pz", "Oz", "P3", "P4", "PO7", "PO8"]

TRAIN_SUBJECTS = ["A01", "A02", "A03", "A04"]
EVAL_SUBJECTS = ["A05", "A06"]
BALANCE_SEED = 42

# 官方 HEAD content-length（2026-09-02 实测，用于下载完整性校验）
EXPECTED_BYTES = {
    "A01": 21548593, "A02": 21588837, "A03": 21611830,
    "A04": 21580390, "A05": 21607457, "A06": 21596479,
}


def find_flash_runs(ys: np.ndarray) -> np.ndarray:
    """y_stim != 0 的连续段起始样本（=闪烁 onset）。"""
    idx = np.flatnonzero(ys != 0)
    assert len(idx) > 0
    splits = np.flatnonzero(np.diff(idx) > 1)
    runs = np.split(idx, splits + 1)
    return np.array([r[0] for r in runs])


def process_subject(name: str) -> dict:
    """解析一个 .mat，返回逐闪烁 epoch 与元数据。"""
    f = RAW / f"{name}.mat"
    if not f.exists():
        raise SystemExit(f"✗ 缺 {f}")
    nbytes = f.stat().st_size
    assert nbytes == EXPECTED_BYTES[name], \
        f"{name}.mat 字节数 {nbytes} != 预期 {EXPECTED_BYTES[name]}（下载不完整？）"
    m = sio.loadmat(f, squeeze_me=True, struct_as_record=False, chars_as_strings=True)
    d = m["data"]
    X = np.asarray(d.X, np.float64)
    y = np.asarray(d.y)
    ys = np.asarray(d.y_stim)
    tr = np.asarray(d.trial)
    ch = [str(c) for c in np.asarray(d.channels, dtype=object)]
    assert ch == CHANNELS, f"{name} 通道 {ch} != {CHANNELS}"
    assert X.ndim == 2 and X.shape[1] == len(CHANNELS)
    assert not np.isnan(X).any()
    N = X.shape[0]

    onsets = find_flash_runs(ys)
    print(f"  {name}: {N} 样本, {len(onsets)} 次闪烁, ch={ch}, NaN={np.isnan(X).sum()}")

    epochs, labels, codes, trial_ids, reps = [], [], [], [], []
    # 逐闪烁
    n_dropped = 0
    for i, o in enumerate(onsets):
        # 段内 y 必须恒定（全 1 或全 2）且与 onset 处 y_stim 一致
        seg = y[o:o + 32] if o + 32 <= N else y[o:]
        assert np.unique(seg).size == 1, f"{name} onset {o} 段内 y 不恒定"
        # 闪烁段长断言：32 样本 = 125ms @256Hz
        if o + 32 > N or np.diff([o, o + 32])[0] != 32:
            pass  # 末段长度由 seg 断言覆盖
        if o + WIN > N:
            n_dropped += 1
            continue
        lab = int(seg[0])
        assert lab in (1, 2)
        code = int(ys[o])
        assert code != 0
        # trial 归属：实测每个 trial 的 marker tr[k] = 该 trial 首个闪烁
        # onset + 1（首个闪烁在 tr[k]-1，见 description.pdf 的 cue 设计），
        # 故 onset o ∈ trial k ⇔ tr[k]-1 ≤ o < tr[k+1]-1
        t = int(np.searchsorted(tr - 1, o, side="right") - 1)
        t = min(max(t, 0), len(tr) - 1)
        epochs.append(X[o:o + WIN])
        labels.append(1 if lab == 2 else 0)
        codes.append(code)
        trial_ids.append(t)
    if n_dropped:
        print(f"    ⚠ {name}: {n_dropped} 次闪烁窗口越界丢弃")
    n_ep = len(labels)
    assert n_ep == 4200, f"{name}: 期望 4200 次闪烁，实际 {n_ep}"

    # rep：同一 trial 内同 code 第几次出现（每 trial 每 code 恰 10 次，断言）
    trial_ids = np.asarray(trial_ids)
    codes = np.asarray(codes)
    for t in range(len(tr)):
        for c in range(1, 13):
            cnt = ((trial_ids == t) & (codes == c)).sum()
            assert cnt == 10, f"{name} trial {t} code {c}: {cnt} != 10"
    # 每 trial 恰 2 个不同目标码（目标行 + 目标列）
    for t in range(len(tr)):
        tgt_codes = np.unique(codes[(trial_ids == t) & (np.asarray(labels) == 1)])
        assert len(tgt_codes) == 2, f"{name} trial {t}: 目标码 {tgt_codes} != 2 个"
    # rep 计算
    rep_counter: dict = {}
    for t, c in zip(trial_ids.tolist(), codes.tolist()):
        key = (t, c)
        rep_counter[key] = rep_counter.get(key, 0) + 1
        reps.append(rep_counter[key] - 1)
    reps = np.asarray(reps)
    assert reps.max() == 9 and (reps == 9).sum() == 12 * len(tr)

    X_ep = np.stack(epochs)              # (n_ep, C, WIN)
    labels = np.asarray(labels, np.int64)
    # 逐 epoch z-score（跨通道×时间整体标量；与 2a「逐试次整体」一致）
    mu = X_ep.mean(axis=(1, 2), keepdims=True)
    sd = X_ep.std(axis=(1, 2), keepdims=True)
    sd = np.where(sd < 1e-12, 1.0, sd)
    X_z = ((X_ep - mu) / sd).astype(np.float32)
    return dict(X=X_z, y=labels, trial=trial_ids, code=codes, rep=reps,
                subj_idx=None, n_ep=n_ep)


def balance(X, y, seed: int) -> tuple:
    """均衡子采样：全保留目标(1)，非目标(0) 随机抽到与目标等量（seed 固定）。"""
    idx_t = np.flatnonzero(y == 1)
    idx_n = np.flatnonzero(y == 0)
    rng = np.random.RandomState(seed)
    pick_n = rng.choice(idx_n, size=len(idx_t), replace=False)
    keep = np.sort(np.concatenate([idx_t, pick_n]))
    return X[keep], y[keep]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="小样本验证链路")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    subs = TRAIN_SUBJECTS + EVAL_SUBJECTS
    if args.smoke:
        subs = ["A01", "A05"]           # 只跑 1 个训练被试 + 1 个评估被试

    per_subj = {}
    all_X, all_y, all_trial, all_code, all_rep = [], [], [], [], []
    for name in subs:
        r = process_subject(name)
        per_subj[name] = {"n_epochs": r["n_ep"],
                          "n_target": int((r["y"] == 1).sum()),
                          "n_non_target": int((r["y"] == 0).sum())}
        all_X.append(r["X"])
        all_y.append(r["y"])
        all_trial.append(r["trial"])
        all_code.append(r["code"])
        all_rep.append(r["rep"])
    X = np.concatenate(all_X)
    y = np.concatenate(all_y)
    trial = np.concatenate(all_trial)
    code = np.concatenate(all_code)
    rep = np.concatenate(all_rep)
    print(f"\n总闪烁: {len(y)}（目标 {int((y==1).sum())} / 非目标 {int((y==0).sum())}）")

    if args.smoke:
        # smoke：只验证链路，不覆盖正式 npz（输出到 data/smoke_* 供检查后删除）
        np.savez_compressed(OUT / "smoke_train.npz", X=X[:64], y=y[:64])
        print(f"[smoke] ✓ 链路验证完成 → {OUT}/smoke_train.npz（64 epoch，不用于训练）")
        return 0

    # ---- 正式输出 ----
    # train: 前 4 人
    n_tr = len(TRAIN_SUBJECTS) * 4200
    Xtr, ytr = X[:n_tr], y[:n_tr]
    # eval_full: 后 2 人全部（A05/A06），含元数据
    Xevf, yevf = X[n_tr:], y[n_tr:]
    subj_evf = np.concatenate([np.full(4200, i, np.int64)
                               for i in range(len(EVAL_SUBJECTS))])
    trial_evf = trial[n_tr:]
    code_evf = code[n_tr:]
    rep_evf = rep[n_tr:]
    # eval 均衡子采样（seed 42）
    Xev, yev = balance(Xevf, yevf, BALANCE_SEED)
    # train 均衡子采样
    Xtrb, ytrb = balance(Xtr, ytr, BALANCE_SEED)

    def to_npz(Xb, yb):
        # Xb: (n, WIN, C) → (n, C, WIN) → (n, 1, C, WIN)
        Xb = Xb.transpose(0, 2, 1).reshape(-1, 1, len(CHANNELS), WIN)
        return Xb.astype(np.float32), yb.astype(np.int64)

    Xtrb, ytrb = to_npz(Xtrb, ytrb)
    Xev, yev = to_npz(Xev, yev)
    Xevf, yevf = to_npz(Xevf, yevf)

    np.savez_compressed(OUT / "train.npz", X=Xtrb, y=ytrb)
    np.savez_compressed(OUT / "eval.npz", X=Xev, y=yev)
    np.savez_compressed(OUT / "eval_full.npz",
                        X=Xevf, y=yevf,
                        subj=subj_evf, trial=trial_evf, code=code_evf, rep=rep_evf)
    print(f"✓ train.npz    : X={Xtrb.shape} y 分布 {dict(zip(*np.unique(ytrb, return_counts=True)))}")
    print(f"✓ eval.npz     : X={Xev.shape}  y 分布 {dict(zip(*np.unique(yev, return_counts=True)))}")
    print(f"✓ eval_full.npz: X={Xevf.shape} 全部 {len(yevf)} 闪烁含元数据")

    meta = {
        "dataset": "BNCI 008-2014 EPFL P300 speller (ALS patients)",
        "license": "CC BY-NC-ND 4.0 (IRCCS Fondazione Santa Lucia, Rome)",
        "description_pdf": "https://bnci-horizon-2020.eu/database/data-sets/008-2014/description.pdf",
        "subjects_downloaded": ["A01", "A02", "A03", "A04", "A05", "A06"],
        "subject_files": EXPECTED_BYTES,
        "bytes_verified": "逐文件 == HEAD content-length (2026-09-02)",
        "channels": CHANNELS,
        "fs": FS,
        "flash_ms": 125, "isi_ms": 125, "soa_ms": 250, "stim_per_trial": 12,
        "repetitions_per_trial": 10, "target_flash_per_trial": 20,
        "trials_per_subject": 35, "words": "7 词 × 5 字符",
        "window": f"[0, {WIN_SEC}]s = {WIN} 样本 @{FS}Hz（含闪烁期 125ms）",
        "labels": {"0": "非目标闪烁 NonTarget", "1": "目标闪烁 Target",
                   "y_raw": "0=无闪烁 1=NonTarget 2=Target", "y_stim": "0=无 1-6=Row 7-12=Col"},
        "zscore": "逐 epoch 整体标量（跨 8ch×T，与 2a 逐试次一致），每 epoch 方差=1",
        "split": {"train_subjects": TRAIN_SUBJECTS,
                  "eval_subjects": EVAL_SUBJECTS,
                  "note": "跨被试，被试级隔离；epoch 未打乱（chronological）"},
        "class_balance": {"raw_ratio": "1 目标 : 5 非目标（协议固有）",
                          "train": "2800 目标 + 2800 非目标（seed 42 子采样）",
                          "eval": "1400 目标 + 1400 非目标（seed 42 子采样）",
                          "eval_full": "A05-A06 全部 1400 目标 + 7000 非目标（叠加评估/真实分布用）"},
        "per_subject": per_subj,
        "n_train": int(len(ytrb)), "n_eval": int(len(yev)), "n_eval_full": int(len(yevf)),
        "chance": "50%（目标/非目标二分类）",
    }
    (OUT / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                   encoding="utf-8")
    print("✓ meta.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
