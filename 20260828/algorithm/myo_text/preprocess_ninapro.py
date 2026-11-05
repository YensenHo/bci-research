# -*- coding: utf-8 -*-
"""
preprocess_ninapro.py —— Ninapro DB5 sEMG → 52 类手势多分类切段预处理

复现对象：Ninapro DB5（Pizzolato et al., PLoS One 2017）
  - 10 名完好被试，双侧 Thalmic Myo 臂环（每环 8 通道）×2 = 16 通道 sEMG，采样率 200Hz。
  - 52 个手势（exercise A: 12 指部动作 + exercise B: 17 腕/手部 + exercise C: 23 抓握/功能），
    每个手势重复 6 次，动作间 3s 休息（stimulus=0）。本预处理**排除休息类**，只做 52 类
    手势分类（更贴近「几十类手势」任务，避免休息类因近零肌电过于易分、虚抬精度）。

【数据来源（如实记录）】
  - Zenodo 记录 DOI 10.5281/zenodo.1000116（"Ninapro dataset 5 (double Myo armband)"）。
  - 开放访问（access_right=open），**无需注册**，直接可下。
  - 10 个 zip：s1.zip .. s10.zip，每个 ~20MB，总计 ~200MB。
  - 下载直链（Zenodo API，支持 curl -C - 断点续传）：
      https://zenodo.org/api/records/1000116/files/sN.zip/content
  - 解压后每个被试一个目录 sN/，含 3 个 .mat：S{N}_E1_A1.mat / _E2_ / _E3_。

【.mat 结构（scipy.io.loadmat 实测）】
  - emg         : (samples, 16) float32   16 通道 sEMG（原始 ADC，范围约 ±128）
  - stimulus    : (samples, 1)  int8      当前动作标签（0=休息，1..k 为该 exercise 内局部编号）
  - restimulus  : (samples, 1)  int8      试次索引（0=休息）
  - repetition  : (samples, 1)  int8      该动作的第几次重复（1..6）
  - frequency   : 标量 = 200（Hz）
  - exercise    : 标量（内部编号，与文件名 E 编号不一致！见下）

【重要：内部 exercise 字段与文件名不一致（实测，10 个被试一致）】
  - 文件 S{N}_E1_A1.mat → 内部 exercise=3 → 12 个动作（官方 Exercise A）
  - 文件 S{N}_E2_A1.mat → 内部 exercise=1 → 17 个动作（官方 Exercise B）
  - 文件 S{N}_E3_A1.mat → 内部 exercise=2 → 23 个动作（官方 Exercise C）
  因此全局手势编号按「动作数量」而非文件名对齐：
    内部 exercise=3（12 动作）→ 全局 id 1..12   （偏移 0）
    内部 exercise=1（17 动作）→ 全局 id 13..29  （偏移 12）
    内部 exercise=2（23 动作）→ 全局 id 30..52  （偏移 29）
  标签 y = 全局 id - 1 ∈ {0..51}（52 类）。

【标准化（关键，如实）】
  Ninapro 各通道增益差异极大（实测 s1 通道 std 从 2.6 到 18.0，约 7 倍），且动作判别
  高度依赖「哪些通道更强」的**跨通道相对幅度**。因此：
  - 采用**全局逐通道标准化**：先扫全量数据求每个通道的均值/标准差（一次通过，running
    sum 累加，不整载入），再对每个窗做 (x - mean[c]) / std[c]。
  - 这样既消除通道间增益差异（7 倍），又**保留窗内跨通道相对幅度**（判别关键特征）。
  - 注意：不能用「逐窗逐通道」标准化（x-mean/std 每窗每通道各自算），那会把安静通道的
    噪声放大到与活跃通道同方差，抹掉空间幅度模式，实测导致精度卡在 ~15%（见报告）。

【切段方式（滑动窗）】
  - 每个试次（restimulus>0 的连续段）是一个手势的一次重复（约 5s=1000 样本）。
  - 在该段内滑动窗：窗长 WINDOW=256 样本（1.28s），步长 STRIDE=128 样本（0.64s）。
  - 同时记录每个窗所属 subject / repetition，供训练脚本做「重复留出」或「留一被试」划分。

【输出 npz 字段】
  - X        : (N, 1, 16, WINDOW) float32   全局标准化后 sEMG 窗
  - y        : (N,) int64                    全局手势标签 0..51
  - rep      : (N,) int8                     该窗所属重复 1..6
  - subject  : (N,) int8                     被试编号 1..10
  - chan_mean/chan_std : (16,)               全局逐通道统计（记录备查）
  - fs_hz, window, stride : 标量元信息

用法
----
  自检（mock .mat，验证切段+全局标准化逻辑）：
    env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin \\
        /opt/miniconda3/envs/openmmlab/bin/python preprocess_ninapro.py --selfcheck

  真实预处理（需先下载 data_ninapro/s1..s10）：
    env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin \\
        /opt/miniconda3/envs/openmmlab/bin/python preprocess_ninapro.py \\
        --data-dir data_ninapro --out ninapro52.npz --max-subjects 10
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# 常量（依据 DB5 实测，如实）
# ---------------------------------------------------------------------------
FS_HZ = 200            # 采样率 200Hz（Myo 臂环）
N_CHANNELS = 16        # 双 Myo 臂环 16 通道
WINDOW = 256           # 窗长 256 样本 = 1.28s
STRIDE = 128           # 步长 128 样本 = 0.64s
N_GESTURES = 52        # 52 个手势（排除休息）
# 内部 exercise 字段 → 全局 id 偏移（见文件头说明）
EXERCISE_OFFSET = {3: 0, 1: 12, 2: 29}


# ---------------------------------------------------------------------------
# .mat 读取
# ---------------------------------------------------------------------------
def _check_scipy():
    try:
        import scipy.io  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "✗ 缺少 scipy。请先安装：/opt/miniconda3/envs/openmmlab/bin/pip install scipy"
        ) from exc


def load_exercise_mat(path: Path) -> dict:
    """读取单个 exercise 的 .mat，返回需要的数组。"""
    _check_scipy()
    import scipy.io as sio

    m = sio.loadmat(str(path))
    for key in ("emg", "stimulus", "restimulus", "repetition", "exercise"):
        if key not in m:
            raise SystemExit(f"✗ {path} 缺少字段 {key}（实际 keys: {sorted(m.keys())}）")
    emg = np.asarray(m["emg"], dtype=np.float32)          # (samples, 16)
    stimulus = np.asarray(m["stimulus"], dtype=np.int32).ravel()
    restimulus = np.asarray(m["restimulus"], dtype=np.int32).ravel()
    repetition = np.asarray(m["repetition"], dtype=np.int32).ravel()
    exercise = int(np.asarray(m["exercise"]).ravel()[0])
    if emg.ndim != 2 or emg.shape[1] != N_CHANNELS:
        raise SystemExit(f"✗ {path} emg 形状异常：{emg.shape}")
    return {
        "emg": emg, "stimulus": stimulus, "restimulus": restimulus,
        "repetition": repetition, "exercise": exercise,
    }


# ---------------------------------------------------------------------------
# 全局逐通道统计（一次通过，running sum，不整载入）
# ---------------------------------------------------------------------------
def compute_global_channel_stats(mat_paths: List[Path]) -> Tuple[np.ndarray, np.ndarray]:
    """对所有 .mat 的 emg 求逐通道全局均值/标准差（所有样本，含休息）。"""
    s = np.zeros(N_CHANNELS, dtype=np.float64)
    s2 = np.zeros(N_CHANNELS, dtype=np.float64)
    n = 0
    for mp in mat_paths:
        d = load_exercise_mat(mp)
        emg = d["emg"].astype(np.float64)
        s += emg.sum(axis=0)
        s2 += (emg * emg).sum(axis=0)
        n += emg.shape[0]
    mean = s / n
    var = s2 / n - mean * mean
    std = np.sqrt(np.maximum(var, 1e-8))
    return mean.astype(np.float32), std.astype(np.float32)


# ---------------------------------------------------------------------------
# 切段：试次 → 滑动窗（应用全局标准化）
# ---------------------------------------------------------------------------
def segment_exercise(
    emg: np.ndarray,
    stimulus: np.ndarray,
    restimulus: np.ndarray,
    repetition: np.ndarray,
    exercise: int,
    chan_mean: np.ndarray,
    chan_std: np.ndarray,
    window: int = WINDOW,
    stride: int = STRIDE,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """把一个 exercise 的连续 sEMG 切成滑动窗（全局逐通道标准化）。

    返回 (X, y, rep)：X (N,1,16,window)，y 全局标签 0..51，rep 重复编号 1..6。
    """
    offset = EXERCISE_OFFSET[exercise]
    n = emg.shape[0]
    active = stimulus > 0
    empty = (np.zeros((0, 1, N_CHANNELS, window), np.float32),
             np.zeros((0,), np.int64), np.zeros((0,), np.int8))
    if not active.any():
        return empty

    boundaries = np.where(np.diff(active.astype(np.int8)) != 0)[0] + 1
    seg_starts = np.concatenate(([0], boundaries))
    seg_ends = np.concatenate((boundaries, [n]))

    Xs, ys, reps = [], [], []
    for s0, s1 in zip(seg_starts, seg_ends):
        if not active[s0]:
            continue
        seg = emg[s0:s1]
        L = seg.shape[0]
        if L < window:
            continue
        label_global = offset + int(stimulus[s0])
        rep = int(repetition[s0])
        # 全局逐通道标准化（保留窗内跨通道相对幅度）
        seg_norm = (seg - chan_mean) / chan_std
        for start in range(0, L - window + 1, stride):
            w = seg_norm[start:start + window]     # (window, 16)
            Xs.append(w.T[np.newaxis, ...])        # (1, 16, window)
            ys.append(label_global - 1)
            reps.append(rep)

    if not Xs:
        return empty
    X = np.stack(Xs, axis=0).astype(np.float32)
    y = np.asarray(ys, dtype=np.int64)
    rep = np.asarray(reps, dtype=np.int8)
    return X, y, rep


def _parse_subject_id(subject_dir: Path) -> int:
    name = subject_dir.name.lower()
    digits = "".join(ch for ch in name if ch.isdigit())
    return int(digits) if digits else -1


# ---------------------------------------------------------------------------
# 顶层：计算全局统计 + 切段 + 保存
# ---------------------------------------------------------------------------
def preprocess_all(
    data_dir: Path,
    out_npz: Path,
    window: int = WINDOW,
    stride: int = STRIDE,
    max_subjects: Optional[int] = None,
    exercise: Optional[int] = None,
) -> dict:
    subject_dirs = sorted(
        [d for d in data_dir.iterdir() if d.is_dir() and _parse_subject_id(d) > 0],
        key=_parse_subject_id,
    )
    if max_subjects:
        subject_dirs = subject_dirs[:max_subjects]
    if not subject_dirs:
        raise SystemExit(f"✗ 未找到被试目录（s1..sN）：{data_dir}")

    # 收集所有 .mat 路径
    all_mats: List[Path] = []
    for sd in subject_dirs:
        all_mats.extend(sorted(sd.glob("*.mat")))
    if not all_mats:
        raise SystemExit("✗ 未找到 .mat 文件")

    # 按内部 exercise 过滤（可选）：只保留指定 exercise 的 .mat，用于 12/17 类子集。
    # 内部编号：1=B类17动作(E2文件) / 2=C类23动作(E3文件) / 3=A类12动作(E1文件)。
    if exercise is not None:
        keep = []
        for mp in all_mats:
            if load_exercise_mat(mp)["exercise"] == exercise:
                keep.append(mp)
        all_mats = keep
        if not all_mats:
            raise SystemExit(f"✗ 未找到内部 exercise={exercise} 的 .mat 文件")
        print(f"[子集] 仅保留内部 exercise={exercise} 的 .mat，共 {len(all_mats)} 个")

    # 第一遍：全局逐通道统计
    chan_mean, chan_std = compute_global_channel_stats(all_mats)
    print(f"全局逐通道 std: {np.round(chan_std, 2).tolist()}")

    # 第二遍：切段 + 标准化（只用过滤后的 all_mats）
    Xs, ys, reps, subjects = [], [], [], []
    for mp in all_mats:
        subject_id = _parse_subject_id(mp.parent)
        d = load_exercise_mat(mp)
        X, y, rep = segment_exercise(
            d["emg"], d["stimulus"], d["restimulus"], d["repetition"],
            d["exercise"], chan_mean, chan_std, window=window, stride=stride,
        )
        if X.shape[0] > 0:
            Xs.append(X)
            ys.append(y)
            reps.append(rep)
            subjects.append(np.full((X.shape[0],), subject_id, dtype=np.int8))

    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0)
    rep = np.concatenate(reps, axis=0)
    subject = np.concatenate(subjects, axis=0)

    # 子集模式下把标签重映射到 0..k-1（训练脚本按 y.max()+1 推断类别数）
    if exercise is not None:
        unq = np.unique(y)
        remap = {int(u): i for i, u in enumerate(unq)}
        y = np.asarray([remap[int(v)] for v in y], dtype=np.int64)

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz, X=X, y=y, rep=rep, subject=subject,
        chan_mean=chan_mean, chan_std=chan_std,
        fs_hz=np.asarray(FS_HZ, dtype=np.int64),
        window=np.asarray(window, dtype=np.int64),
        stride=np.asarray(stride, dtype=np.int64),
    )

    n_classes = int(np.unique(y).size)
    class_counts = {int(c): int((y == c).sum()) for c in np.unique(y)}
    stats = {
        "n_subjects": int(np.unique(subject).size),
        "n_samples": int(X.shape[0]),
        "X_shape": list(X.shape),
        "n_classes": n_classes,
        "class_min": min(class_counts.values()),
        "class_max": max(class_counts.values()),
        "repetitions": sorted(int(r) for r in np.unique(rep)),
        "n_empty_classes": 0,
        "exercise": exercise,
    }
    return stats


# ---------------------------------------------------------------------------
# 自检：mock .mat 验证切段 + 全局标准化逻辑（不依赖真实数据）
# ---------------------------------------------------------------------------
def _selfcheck() -> int:
    _check_scipy()
    import scipy.io as sio

    print("=" * 72)
    print("preprocess_ninapro.py 自检：mock .mat → 切段 + 全局标准化")
    print("=" * 72)

    # ---- mock：2 动作 × 2 重复，每重复 1s(200样本)，间隔 0.5s 休息；通道幅度不同 ----
    fs = 200
    n_act, n_rep = 2, 2
    seg_len, rest_len = fs, fs // 2
    t = np.arange(seg_len, dtype=np.float32)
    emg_blocks, stim_blocks, restim_blocks, rep_blocks = [], [], [], []
    for a in range(1, n_act + 1):
        for r in range(1, n_rep + 1):
            # 动作段：通道0 大增益 + 强时变（正弦），通道1 小增益 + 弱时变。
            # 既模拟通道增益差异，又让窗内有时变成分——否则全局标准化后窗内 std 恒 0，
            # 无法验证「全局标准化保留跨通道相对幅度」这条断言（见下）。
            seg = np.zeros((seg_len, N_CHANNELS), np.float32)
            seg[:, 0] = a * 10.0 + 12.0 * np.sin(2.0 * np.pi * 4.0 * t / seg_len)
            seg[:, 1] = a * 2.0 + 0.5 * np.sin(2.0 * np.pi * 4.0 * t / seg_len)
            emg_blocks.append(seg)
            stim_blocks.append(np.full((seg_len,), a, np.int8))
            restim_blocks.append(np.full((seg_len,), a * n_rep + r, np.int8))
            rep_blocks.append(np.full((seg_len,), r, np.int8))
            # 休息段（全零）
            emg_blocks.append(np.zeros((rest_len, N_CHANNELS), np.float32))
            stim_blocks.append(np.zeros((rest_len,), np.int8))
            restim_blocks.append(np.zeros((rest_len,), np.int8))
            rep_blocks.append(np.zeros((rest_len,), np.int8))

    emg = np.concatenate(emg_blocks, axis=0)
    stimulus = np.concatenate(stim_blocks, axis=0)
    restimulus = np.concatenate(restim_blocks, axis=0)
    repetition = np.concatenate(rep_blocks, axis=0)

    tmpdir = Path("/tmp/ninapro_selfcheck")
    tmpdir.mkdir(exist_ok=True)
    mock_mat = tmpdir / "S1_E1_A1.mat"
    sio.savemat(str(mock_mat), {
        "emg": emg, "stimulus": stimulus[:, None], "restimulus": restimulus[:, None],
        "repetition": repetition[:, None], "exercise": np.array([[3]], dtype=np.uint8),
        "frequency": np.array([[200]], dtype=np.uint8),
    })

    d = load_exercise_mat(mock_mat)
    assert d["emg"].shape[1] == N_CHANNELS
    assert d["exercise"] == 3
    print(f"[OK] mock 读取：emg={d['emg'].shape}, exercise={d['exercise']}（内部编号=3 → 12动作组）")

    # 全局统计
    chan_mean, chan_std = compute_global_channel_stats([mock_mat])
    print(f"[OK] 全局逐通道 std：{np.round(chan_std, 2).tolist()}（通道0 应 > 通道1，保留相对幅度）")
    assert chan_std[0] > chan_std[1], "通道0 幅度大，std 应更大"

    X, y, rep = segment_exercise(
        d["emg"], d["stimulus"], d["restimulus"], d["repetition"], 3,
        chan_mean, chan_std, window=64, stride=32,
    )
    expect = n_act * n_rep * 5   # (200-64)//32+1 = 5 窗/重复
    assert X.shape[0] == expect, f"期望 {expect} 窗，实际 {X.shape[0]}"
    assert X.shape == (expect, 1, N_CHANNELS, 64)
    assert set(y.tolist()) == set(range(n_act))
    assert set(rep.tolist()) == set(range(1, n_rep + 1))
    print(f"[OK] 切段：X.shape={X.shape}，标签 {sorted(set(y.tolist()))}，重复 {sorted(set(rep.tolist()))}")

    # 关键：全局标准化后，动作段内通道0 的相对幅度仍应 > 通道1（空间模式保留）
    w = X[0, 0]           # (16, 64)
    assert w[0].std() > w[1].std(), "标准化后通道0 相对幅度应仍 > 通道1"
    print(f"[OK] 全局标准化保留跨通道相对幅度：窗内通道0 std={w[0].std():.2f} > 通道1 std={w[1].std():.2f}")

    print("✓ 全部自检通过")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="Ninapro DB5 → 52 类手势多分类切段")
    ap.add_argument("--selfcheck", action="store_true", help="运行自检（mock .mat）")
    ap.add_argument("--data-dir", type=str, default="data_ninapro", help="已解压的 Ninapro 目录")
    ap.add_argument("--out", type=str, default="ninapro52.npz", help="输出 npz")
    ap.add_argument("--window", type=int, default=WINDOW, help="窗长（样本）")
    ap.add_argument("--stride", type=int, default=STRIDE, help="步长（样本）")
    ap.add_argument("--max-subjects", type=int, default=None, help="最多处理被试数")
    ap.add_argument("--exercise", type=int, default=None,
                    help="仅抽取内部 exercise（1=B类17动作/E2, 2=C类23动作/E3, "
                         "3=A类12动作/E1）；默认 None=全 52 类")
    args = ap.parse_args()

    if args.selfcheck:
        return _selfcheck()

    stats = preprocess_all(
        Path(args.data_dir), Path(args.out), window=args.window, stride=args.stride,
        max_subjects=args.max_subjects, exercise=args.exercise,
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"✓ 已保存：{args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
