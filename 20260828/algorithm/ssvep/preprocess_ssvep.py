#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SSVEP Benchmark 2015 预处理：枕区 6 通道子集 + 多窗口切分
数据格式: S#.mat 的 'data' 为 (64 通道, 1500 采样点(6s@250Hz), 40 目标, 6 blocks)
刺激 onset 位于 trial 起点后 0.5s (第 125 个采样点), 刺激持续 5s。
枕区通道: O1/Oz/O2/PO3/POz/PO4 (按 64-channels.loc 匹配行号)
输出: npz, 每个窗口一个数组 (nsub, 6blocks, 40targets, 6ch, nsamp)
"""
import argparse
import re
import os
import numpy as np
import scipy.io as sio

FS = 250
ONSET_S = 0.5  # 刺激 onset 在 trial 开始后 0.5s
WINDOWS = [0.5, 1.0, 2.0, 3.0]
OCCIPITAL = ["O1", "OZ", "O2", "PO3", "POZ", "PO4"]


def parse_loc(path):
    """64-channels.loc -> {label: 0-based 行号}"""
    idx = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = re.split(r"[\s\t]+", line)
            if len(parts) < 4:
                continue
            chan_idx = int(parts[0])
            label = parts[3].strip().upper()
            idx[label] = chan_idx - 1
    return idx


def parse_subjects(spec):
    if "-" in spec:
        a, b = spec.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in spec.split(",")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", default="raw")
    ap.add_argument("--out", default="ssvep_occipital.npz")
    ap.add_argument("--subjects", default="1-10")
    ap.add_argument("--loc", default="raw/64-channels.loc")
    ap.add_argument("--onset", type=float, default=0.5,
                    help="窗口起点(相对 trial 起点, 秒)。数据集文档: 刺激 onset 在 0.5s(0.5s cue 后)")
    ap.add_argument("--save_channels", action="store_true",
                    help="同时保存所选通道名")
    args = ap.parse_args()

    subjects = parse_subjects(args.subjects)
    loc = parse_loc(args.loc)
    missing = [c for c in OCCIPITAL if c not in loc]
    if missing:
        raise RuntimeError("loc 中缺少通道: %s" % missing)
    rows = [loc[c] for c in OCCIPITAL]
    print("枕区通道行号(0-based):", {c: loc[c] for c in OCCIPITAL})

    onset = int(args.onset * FS)
    if onset + int(max(WINDOWS) * FS) > 1500:
        raise RuntimeError("窗口超出 trial 长度: onset=%gs, max win=%gs" % (
            args.onset, max(WINDOWS)))
    out = {}
    for w in WINDOWS:
        nsamp = int(w * FS)
        out["eeg_%g" % w] = np.empty((len(subjects), 6, 40, 6, nsamp),
                                      dtype=np.float64)

    for si, s in enumerate(subjects):
        mat_path = os.path.join(args.raw_dir, "S%d.mat" % s)
        m = sio.loadmat(mat_path)
        data = m["data"]  # (64, 1500, 40, 6)
        assert data.shape == (64, 1500, 40, 6), data.shape
        occ = data[rows]  # (6, 1500, 40, 6)
        # -> (6 blocks, 40 targets, 6 ch, time)
        occ = occ.transpose(3, 2, 0, 1)
        for w in WINDOWS:
            nsamp = int(w * FS)
            out["eeg_%g" % w][si] = occ[:, :, :, onset:onset + nsamp]
        print("S%d 处理完成, 每窗口形状 %s" % (s, out["eeg_%g" % WINDOWS[0]][si].shape))

    np.savez_compressed(args.out, **out)
    print("已保存 %s" % args.out)
    for w in WINDOWS:
        print("  eeg_%g: %s (nsub,block,target,ch,samp)" % (
            w, out["eeg_%g" % w].shape))


if __name__ == "__main__":
    main()
