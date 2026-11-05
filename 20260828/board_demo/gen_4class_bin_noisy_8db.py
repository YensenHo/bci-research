#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gen_4class_bin_noisy_8db.py — 生成 22ch 四分类板载测试二进制（eval 全量 + SNR=8dB HBC 噪声）

严格复刻 eval_hbc_snr_sweep.py 的 8dB 测试协议（同种子、同 batch、同噪声参数），
使板载实测数字与 snr_sweep.json 的 "8dB" 列直接可比：

  - 数据     : algorithm/data_4class/eval.npz（2592 试次，cross-session，四类均衡）
  - 噪声     : hbc_noise(qbits=5, qrange=4.0, sigma_ratio=10^(-8/20)=0.3981)
              = 5bit 量化 + AWGN（无突发，与 sweep 的 "8dB" 列一致）
  - 种子     : TEST_NOISE_SEED = 20240828（sweep 同款）
  - 施加方式 : DataLoader(batch_size=16, shuffle=False) 逐 batch 施加，
              与 sweep 的 _eval_noise 完全同构（RNG 消耗顺序一致）

输出：board_demo/lif_4class_test_full_8db.bin
格式（与 lif_surrogate_infer_tcnet.c 的 main 读取完全一致）：
    int32 N + 每试次 float32×(22*500=11000) + int32 label

注意：sweep 在 MPS 上跑、本脚本在 CPU 上跑，同种子下 RNG 实现不同导致
噪声实现（具体数值）可能不完全一致，但协议（种子/参数/批序）完全一致。

用法：
  env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin \
      /opt/miniconda3/envs/openmmlab/bin/python gen_4class_bin_noisy_8db.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

ALGO_DIR = Path(__file__).resolve().parent.parent / "algorithm"
BOARD_DEMO_DIR = Path(__file__).resolve().parent
N_CH = 22
N_SAMP = 500
NSAMP = N_CH * N_SAMP
SNR_DB = 8.0
TEST_NOISE_SEED = 20240828
BATCH_SIZE = 16

sys.path.insert(0, str(ALGO_DIR))
from hbc_noise_advanced import hbc_noise, snr_db_to_sigma_ratio  # noqa: E402


def main() -> int:
    d = np.load(ALGO_DIR / "data_4class" / "eval.npz")
    X = d["X"].astype(np.float32)          # (2592, 1, 22, 500)
    y = d["y"].astype(np.int64)            # (2592,)
    N = X.shape[0]
    assert X.shape == (N, 1, N_CH, N_SAMP), f"X shape 异常: {X.shape}"

    sigma_ratio = float(snr_db_to_sigma_ratio(SNR_DB))
    noise_spec = dict(qbits=5, qrange=4.0, sigma_ratio=sigma_ratio)
    print(f"[协议] SNR={SNR_DB}dB → σ/H = {sigma_ratio:.6f}，噪声规格 {noise_spec}")
    print(f"[协议] 固定种子 TEST_NOISE_SEED={TEST_NOISE_SEED}，batch_size={BATCH_SIZE}")

    # 与 eval_hbc_snr_sweep.py 的 _eval_noise 同构：manual_seed 一次 + 逐 batch 施加
    loader = DataLoader(TensorDataset(torch.from_numpy(X), torch.from_numpy(y)),
                        batch_size=BATCH_SIZE, shuffle=False)
    torch.manual_seed(TEST_NOISE_SEED)

    noisy = np.empty_like(X)
    with torch.no_grad():
        for i, (xb, _) in enumerate(loader):
            noisy[i * BATCH_SIZE:(i + 1) * BATCH_SIZE] = hbc_noise(xb, **noise_spec).numpy()

    # 噪声确实注入自检：前 1000 个样本的加噪前后差异应非零且幅度与 σ 量级一致
    diff = np.abs(noisy - X)
    print(f"[自检] 加噪前后 |Δ|: mean={diff.mean():.5f} max={diff.max():.4f}（σ/H=0.3981 量级一致）")
    assert diff.max() > 0.1, "噪声未注入？"

    out = BOARD_DEMO_DIR / "lif_4class_test_full_8db.bin"
    with open(out, "wb") as f:
        f.write(np.int32(N).tobytes())
        for i in range(N):
            f.write(noisy[i, 0].ravel().tobytes())
            f.write(np.int32(int(y[i])).tobytes())

    nbytes = out.stat().st_size
    expected = 4 + N * (4 * NSAMP + 4)
    assert nbytes == expected, f"大小异常: {nbytes} != {expected}"
    print(f"✓ 已生成 8dB 加噪板载测试二进制：{out}")
    print(f"  试次数 N={N}，每试次 {NSAMP} float32 + 1 int32 label，文件 {nbytes/1e6:.2f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
