# -*- coding: utf-8 -*-
"""
preprocess_emg.py —— emg2qwerty 击键切段预处理：HDF5 会话 → 8 指 sEMG 片段 npz

复现对象：MyoText (arXiv:2601.03098) 第一级「手指分类」的数据准备。
论文做法：对每个击键事件，切出「击键前 1s + 击键后 1s」的 32 通道 sEMG 片段
（含预备肌电 + 残余肌电），按标准 QWERTY 指法（不含拇指）打 8 指标签。

【数据集事实（依据官方仓库 facebookresearch/emg2qwerty 的 data.py，如实记录）】
  - 每个 session 是单个 HDF5 文件，顶层 group 名 "emg2qwerty"。
  - "emg2qwerty/timeseries" 是 compound dataset，含三个字段：
        emg_left  : (T, 16)  左手 16 通道 sEMG（已高通滤波）
        emg_right : (T, 16)  右手 16 通道 sEMG（已高通滤波）
        time      : (T,)     绝对 unix 时间戳（秒）
    采样率 2kHz，双侧 16ch×2 = 32ch。
  - group attrs（元数据，很小，直接载入内存）：
        session_name / user / condition（"on_keyboard"）/ duration_mins
        keystrokes : JSON 字符串 → list[{"key": <pynput 键名>, "start": 按下时间, "end": 抬起时间}]
        prompts    : JSON 字符串 → list[{"payload": {"text": ...}, "start": ..., "end": ...}]
  - 击键 key 是 pynput 表示：字母/数字/标点直接是字符（如 'a', 'A', '1', ','），
    修饰键为 "Key.space"/"Key.shift"/"Key.enter"/"Key.backspace" 等。

【8 指映射】复用同目录 qwerty_mapping.LETTER_TO_FINGER（标准 QWERTY 指法，不含拇指）。
  拇指键（Key.space）与修饰键（shift/enter/backspace/tab 等）、数字/未覆盖符号
  在手指分类任务里不参与 → 直接跳过（返回 None）。

用法
----
  环境铁律（隔离 env，避免 numpy ImportError）：
    env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin \\
        /opt/miniconda3/envs/openmmlab/bin/python preprocess_emg.py --selfcheck

  自检（mock 会话切段逻辑验证，不依赖真实数据）：
    .../python preprocess_emg.py --selfcheck

  生成合成数据（供 train_lif_finger.py 管线冒烟，不依赖真实数据）：
    .../python preprocess_emg.py --mock-out mock_finger.npz --n-per-class 40

  真实数据（需先下载 emg2qwerty，见复现说明.md §1 数据探查结论）：
    .../python preprocess_emg.py --hdf5 /path/to/sessions --out finger.npz
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np

# ---- 复用同目录 QWERTY 指法映射（只读引用，不改动）----
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
from qwerty_mapping import FINGER_NAMES, LETTER_TO_FINGER  # noqa: E402

# ---------------------------------------------------------------------------
# 常量（依据官方 data.py，如实）
# ---------------------------------------------------------------------------
FS_HZ = 2000            # 采样率 2kHz（官方 data.py 明确）
N_CH_PER_SIDE = 16      # 每侧 16 通道
N_CHANNELS = N_CH_PER_SIDE * 2   # 双侧 32 通道
PRE_S = 1.0             # 击键前 1s（预备肌电）
POST_S = 1.0            # 击键后 1s（残余肌电）
WINDOW_T = int((PRE_S + POST_S) * FS_HZ)   # 2s @ 2kHz = 4000 采样点

# HDF5 字段名（官方 data.py 的 ClassVar 常量，逐字记录）
HDF5_GROUP = "emg2qwerty"
TIMESERIES = "timeseries"
EMG_LEFT = "emg_left"
EMG_RIGHT = "emg_right"
TIME = "time"
KEYSTROKES = "keystrokes"
PROMPTS = "prompts"
CONDITION = "condition"


# ---------------------------------------------------------------------------
# 键名归一化：pynput 键名 → 8 指索引（None = 跳过）
# ---------------------------------------------------------------------------
def normalize_key(key: Any) -> Optional[int]:
    """把一个 pynput 键名映射到 8 指索引；非手指键返回 None（跳过）。

    - "Key.*"：space（拇指）/ shift / enter / backspace / tab / ctrl 等 → 跳过。
    - 大写字母：shift 单独记录，字母键本身仍按字母处理（lowercase 后查表）。
    - 数字 / 未在 qwerty_mapping 覆盖的符号 → 跳过（触摸打字文本里占比极小）。
    """
    k = str(key)
    if k.startswith("Key."):
        return None  # 拇指/修饰/控制键，不属于 8 指分类目标
    ch = k.lower()
    if len(ch) == 1 and ch in LETTER_TO_FINGER:
        return LETTER_TO_FINGER[ch]
    return None


# ---------------------------------------------------------------------------
# HDF5 会话读取
# ---------------------------------------------------------------------------
def read_session(hdf5_path: Path) -> Tuple[np.ndarray, np.ndarray, List[dict]]:
    """读取一个 session 文件，返回 (emg32 (T,32), timestamps (T,), keystrokes list)。

    只读打开，元数据（keystrokes 等）很小直接载入；EMG 按需切片。
    """
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "✗ 缺少 h5py。请先安装：/opt/miniconda3/envs/openmmlab/bin/pip install h5py"
        ) from exc

    with h5py.File(str(hdf5_path), "r") as f:
        grp = f[HDF5_GROUP]
        ts = grp[TIMESERIES]
        # compound dataset 的字段
        assert EMG_LEFT in ts.dtype.fields, f"{hdf5_path} 缺少字段 {EMG_LEFT}"
        assert EMG_RIGHT in ts.dtype.fields, f"{hdf5_path} 缺少字段 {EMG_RIGHT}"
        assert TIME in ts.dtype.fields, f"{hdf5_path} 缺少字段 {TIME}"
        left = ts[EMG_LEFT][:]      # (T, 16)
        right = ts[EMG_RIGHT][:]     # (T, 16)
        timestamps = ts[TIME][:]     # (T,)
        emg32 = np.concatenate([left, right], axis=1).astype(np.float32)  # (T, 32)
        # 元数据：keystrokes 存的是 JSON 字符串
        keystrokes_raw = grp.attrs.get(KEYSTROKES, "[]")
        keystrokes = json.loads(keystrokes_raw) if isinstance(keystrokes_raw, str) \
            else list(keystrokes_raw)
    return emg32, timestamps.astype(np.float64), keystrokes


# ---------------------------------------------------------------------------
# 击键切段：前 1s + 后 1s → 固定长度片段
# ---------------------------------------------------------------------------
def segment_keystrokes(
    emg32: np.ndarray,
    timestamps: np.ndarray,
    keystrokes: List[dict],
    pre_s: float = PRE_S,
    post_s: float = POST_S,
    fs_hz: int = FS_HZ,
    standardize: bool = False,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """把会话切成一串击键片段。

    返回 (X (N,1,32,T), y (N,), keys (N, 原始键名))。
    - 每个击键取 [press - pre_s, press + post_s]，用 searchsorted 定位采样点。
    - 片段长度固定为 T = (pre_s+post_s)*fs；越界（会话首尾）用零补齐。
    - 跳过非手指键（normalize_key 返回 None）。
    """
    T = int((pre_s + post_s) * fs_hz)
    segs: List[np.ndarray] = []
    labels: List[int] = []
    keys: List[str] = []

    n = len(timestamps)
    for k in keystrokes:
        press = float(k["start"])
        finger = normalize_key(k.get("key"))
        if finger is None:
            continue
        # 定位窗口起止采样点
        lo = int(np.searchsorted(timestamps, press - pre_s, side="left"))
        hi = int(np.searchsorted(timestamps, press + post_s, side="right"))
        lo = max(0, lo)
        hi = min(n, hi)
        seg = emg32[lo:hi]                      # (L, 32)
        # 补齐/截断到固定长度 T
        if seg.shape[0] < T:
            pad = np.zeros((T - seg.shape[0], N_CHANNELS), dtype=np.float32)
            seg = np.concatenate([seg, pad], axis=0)
        elif seg.shape[0] > T:
            seg = seg[:T]
        if standardize:
            # 逐通道标准化（按该片段自身均值/方差，均值漂移不敏感）
            mu = seg.mean(axis=0, keepdims=True)
            sd = seg.std(axis=0, keepdims=True) + 1e-6
            seg = (seg - mu) / sd
        # (32, T) → (1, 32, T)，对齐 EEGNet 输入 (B, 1, C, T)
        segs.append(seg.T[np.newaxis, ...].astype(np.float32))
        labels.append(finger)
        keys.append(str(k.get("key")))

    if not segs:
        return (
            np.zeros((0, 1, N_CHANNELS, T), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
            [],
        )
    # 每个 seg 是 (1, 32, T)；用 stack 在 axis0 加批次维 → (N, 1, 32, T)
    X = np.stack(segs, axis=0)                # (N, 1, 32, T)
    y = np.asarray(labels, dtype=np.int64)    # (N,)
    return X, y, keys


def preprocess_sessions(
    hdf5_paths: List[Path],
    out_npz: Path,
    pre_s: float = PRE_S,
    post_s: float = POST_S,
    standardize: bool = False,
    max_sessions: Optional[int] = None,
) -> dict:
    """处理多个会话，合并并保存 npz。返回统计 dict（如实记录，不虚报）。"""
    Xs, ys, ks = [], [], []
    n_sessions = 0
    for p in hdf5_paths[:max_sessions] if max_sessions else hdf5_paths:
        emg32, timestamps, keystrokes = read_session(p)
        X, y, keys = segment_keystrokes(
            emg32, timestamps, keystrokes, pre_s=pre_s, post_s=post_s, standardize=standardize
        )
        if X.shape[0] > 0:
            Xs.append(X)
            ys.append(y)
            ks.extend(keys)
        n_sessions += 1
    if not Xs:
        raise SystemExit("✗ 无有效击键片段（可能所有击键均为非手指键或窗口越界）")
    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz, X=X, y=y,
        keys=np.asarray(ks, dtype=object),
        finger_names=np.asarray(FINGER_NAMES, dtype=object),
        fs_hz=np.asarray(FS_HZ, dtype=np.int64),
        window_T=np.asarray(WINDOW_T, dtype=np.int64),
    )
    stats = {
        "n_sessions": n_sessions,
        "n_samples": int(X.shape[0]),
        "X_shape": list(X.shape),
        "n_classes": int(np.unique(y).size),
        "class_counts": {FINGER_NAMES[i]: int((y == i).sum()) for i in range(8)},
    }
    return stats


# ---------------------------------------------------------------------------
# 合成数据生成（供管线冒烟测试，不依赖真实数据；非真实数据）
# ---------------------------------------------------------------------------
def make_mock_npz(
    out_npz: Path,
    n_per_class: int = 40,
    T: int = 2000,
    seed: int = 0,
    noise: float = 0.35,
) -> dict:
    """生成「可学习但含噪」的合成 8 指 sEMG 数据集，用于跑通训练管线。

    每个手指 f 在特定通道子集上放一个频率特征的正弦突发（模拟该手指运动时
    相关肌肉通道的激活），叠加高斯噪声。这样 CNN 能学到 >12.5% 的随机基线，
    用于验证「预处理 → CNN → LIF 化」整条管线可跑通。**这只是冒烟测试数据，
    不构成任何真实分类精度**。
    """
    rng = np.random.default_rng(seed)
    # 每个手指对应的通道子集：左手(0-3)用前 16 通道，右手(4-7)用后 16 通道
    chan = {
        f: np.arange(f * 2, f * 2 + 2) if f < 4 else np.arange(16 + (f - 4) * 2, 16 + (f - 4) * 2 + 2)
        for f in range(8)
    }
    t = np.arange(T) / FS_HZ
    Xs, ys = [], []
    for f in range(8):
        # 该手指的特征频率（5Hz ~ 19Hz，逐手指不同）
        freq = 5.0 + 2.0 * f
        for _ in range(n_per_class):
            seg = rng.normal(0.0, noise, size=(N_CHANNELS, T)).astype(np.float32)
            amp = rng.uniform(0.8, 1.4)
            # 在指定通道叠加频率突发（窗内衰减包络）
            env = np.sin(np.pi * np.arange(T) / T)  # 两端平滑包络
            for c in chan[f]:
                seg[c] += (amp * np.sin(2 * np.pi * freq * t + rng.uniform(0, np.pi)) * env)
            Xs.append(seg[np.newaxis, ...])  # (1, 32, T)
            ys.append(f)
    X = np.stack(Xs, axis=0).astype(np.float32)   # (N, 1, 32, T)
    y = np.asarray(ys, dtype=np.int64)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz, X=X, y=y,
        finger_names=np.asarray(FINGER_NAMES, dtype=object),
        fs_hz=np.asarray(FS_HZ, dtype=np.int64),
        window_T=np.asarray(T, dtype=np.int64),
        is_mock=np.asarray(True),
    )
    return {"n_samples": int(X.shape[0]), "X_shape": list(X.shape), "n_classes": 8,
            "is_mock": True}


# ---------------------------------------------------------------------------
# 自检：用 mock 会话验证切段逻辑（不依赖真实数据）
# ---------------------------------------------------------------------------
def _selfcheck() -> int:
    print("=" * 72)
    print("preprocess_emg.py 自检：mock 会话 → 击键切段逻辑")
    print("=" * 72)

    # ---- 1. 键名归一化正确性 ----
    assert normalize_key("a") == 0, "a → 左手小指(0)"
    assert normalize_key("f") == 3, "f → 左手食指(3)"
    assert normalize_key("F") == 3, "大写 F 也应 → 左手食指(3)"
    assert normalize_key("j") == 4, "j → 右手食指(4)"
    assert normalize_key("Key.space") is None, "空格(拇指)应跳过"
    assert normalize_key("Key.shift") is None, "shift 应跳过"
    assert normalize_key("Key.enter") is None, "enter 应跳过"
    assert normalize_key("1") is None, "数字(未覆盖)应跳过"
    print("[OK] 键名归一化：8 指 / 拇指 / 修饰键 / 数字 判定正确")

    # ---- 2. 构造 mock 会话（2kHz，10s = 20000 采样点）----
    rng = np.random.default_rng(0)
    duration = 10.0
    T = int(duration * FS_HZ)
    timestamps = np.arange(T, dtype=np.float64) / FS_HZ + 1_700_000_000.0  # 绝对时间戳
    emg32 = rng.normal(0, 1, size=(T, N_CHANNELS)).astype(np.float32)
    # 8 个击键事件（间隔 1s，从 1.0s 开始），覆盖全部 8 指
    home_keys = ["a", "s", "d", "f", "j", "k", "l", ";"]
    keystrokes = [
        {"key": k, "start": 1.0 + i * 1.0, "end": 1.0 + i * 1.0 + 0.08}
        for i, k in enumerate(home_keys)
    ]
    # 追加一个应跳过的键（空格/thumb）
    keystrokes.append({"key": "Key.space", "start": 9.0, "end": 9.1})

    X, y, keys = segment_keystrokes(emg32, timestamps, keystrokes)
    print(f"[OK] 切段结果：X.shape={X.shape}（期望 (8, 1, 32, 4000)）")
    assert X.shape == (8, 1, N_CHANNELS, WINDOW_T), f"X 形状错误 {X.shape}"
    assert y.shape == (8,), f"y 形状错误 {y.shape}"
    assert set(y.tolist()) == set(range(8)), f"8 指标签不完整：{sorted(set(y.tolist()))}"
    assert keys == home_keys, f"跳过的空格键不应出现在 keys 里：{keys}"
    print(f"[OK] 8 指标签完整：{sorted(y.tolist())}，空格键被正确跳过")

    # ---- 3. 越界补齐：击键时间贴近会话首部 ----
    keystrokes_edge = [{"key": "a", "start": 0.2, "end": 0.3}]  # 前 1s 不足，需零补齐
    Xe, ye, _ = segment_keystrokes(emg32, timestamps, keystrokes_edge)
    assert Xe.shape == (1, 1, N_CHANNELS, WINDOW_T), f"越界补齐后形状错误 {Xe.shape}"
    print("[OK] 越界窗口零补齐后长度固定为 4000")

    # ---- 4. 合成数据生成（管线冒烟用）----
    mock_path = _HERE / "mock_finger.npz"
    stats = make_mock_npz(mock_path, n_per_class=5, T=1000)
    print(f"[OK] 合成数据已生成：{mock_path}  {stats}")
    dat = np.load(mock_path)
    assert dat["X"].shape == (8 * 5, 1, N_CHANNELS, 1000)
    assert dat["y"].shape == (40,)
    print("✓ 全部自检通过")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="emg2qwerty 击键切段预处理")
    ap.add_argument("--selfcheck", action="store_true", help="运行自检（mock 会话）")
    ap.add_argument("--mock-out", type=str, default="", help="生成合成 npz 的路径")
    ap.add_argument("--n-per-class", type=int, default=40, help="mock 每类样本数")
    ap.add_argument("--mock-T", type=int, default=2000, help="mock 时间点数")
    ap.add_argument("--hdf5", type=str, default="", help="真实 HDF5 会话目录（含 .h5）")
    ap.add_argument("--out", type=str, default="finger.npz", help="输出 npz 路径")
    ap.add_argument("--pre-s", type=float, default=PRE_S)
    ap.add_argument("--post-s", type=float, default=POST_S)
    ap.add_argument("--standardize", action="store_true", help="逐通道标准化")
    ap.add_argument("--max-sessions", type=int, default=None, help="最多处理会话数")
    args = ap.parse_args()

    if args.selfcheck:
        return _selfcheck()

    if args.mock_out:
        stats = make_mock_npz(Path(args.mock_out), n_per_class=args.n_per_class, T=args.mock_T)
        print(json.dumps(stats, ensure_ascii=False, indent=2))
        return 0

    if args.hdf5:
        h5dir = Path(args.hdf5)
        if not h5dir.exists():
            raise SystemExit(f"✗ 目录不存在：{h5dir}")
        h5files = sorted(h5dir.glob("*.h5")) + sorted(h5dir.glob("*.hdf5"))
        if not h5files:
            raise SystemExit(f"✗ 目录下未找到 .h5/.hdf5 会话文件：{h5dir}")
        print(f"发现 {len(h5files)} 个会话文件")
        stats = preprocess_sessions(
            h5files, Path(args.out), pre_s=args.pre_s, post_s=args.post_s,
            standardize=args.standardize, max_sessions=args.max_sessions,
        )
        print(json.dumps(stats, ensure_ascii=False, indent=2))
        print(f"✓ 已保存：{args.out}")
        return 0

    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
