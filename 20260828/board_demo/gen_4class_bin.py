#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gen_4class_bin.py — 生成 22ch 四分类板载测试二进制（2592 试次全量 eval）

格式（与 lif_surrogate_infer_4class.c 的 main 读取完全一致）：
    int32 N
    每个试次：float32 × (22*500=11000)（通道优先，逐 500 样本）+ int32 label

数据源：algorithm/data_4class/eval.npz（X=(2592,1,22,500) float32，y∈{0,1,2,3} int64，
        四类均衡 [648,648,648,648]）。

输出：board_demo/lif_4class_test_full.bin

用法：
  env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin \
      /opt/miniconda3/envs/openmmlab/bin/python gen_4class_bin.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

ALGO_DIR = Path(__file__).resolve().parent.parent / "algorithm"
BOARD_DEMO_DIR = Path(__file__).resolve().parent
N_CH = 22
N_SAMP = 500
NSAMP = N_CH * N_SAMP  # 11000


def main() -> int:
    d = np.load(ALGO_DIR / "data_4class" / "eval.npz")
    X = d["X"].astype(np.float32)          # (2592, 1, 22, 500)
    y = d["y"].astype(np.int32)            # (2592,)
    N = X.shape[0]
    assert X.shape == (N, 1, N_CH, N_SAMP), f"X shape 异常: {X.shape}"

    out = BOARD_DEMO_DIR / "lif_4class_test_full.bin"
    with open(out, "wb") as f:
        f.write(np.int32(N).tobytes())
        for i in range(N):
            # X[i, 0] 形状 (22, 500)，ravel() 行优先 = [ch0 500][ch1 500]...[ch21 500]
            f.write(X[i, 0].ravel().tobytes())
            f.write(np.int32(y[i]).tobytes())

    nbytes = out.stat().st_size
    expected = 4 + N * (4 * NSAMP + 4)
    assert nbytes == expected, f"大小异常: {nbytes} != {expected}"
    print(f"✓ 已生成板载测试二进制：{out}")
    print(f"  试次数 N={N}，每试次 {NSAMP} float32 + 1 int32 label")
    print(f"  类别分布 = {np.bincount(y, minlength=4).tolist()}（四类均衡）")
    print(f"  文件大小 = {nbytes / 1e6:.2f} MB")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
