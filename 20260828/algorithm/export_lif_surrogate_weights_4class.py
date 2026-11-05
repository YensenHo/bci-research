#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
export_lif_surrogate_weights_4class.py — 导出 22ch 四分类 surrogate LIF-EEGNet 权重为 C 头

与 export_lif_surrogate_weights.py（二分类 8ch）的唯一区别：
  - 加载 **lif_4class_t1.0_tau2_T8.pth**（22 通道、四分类、thr=1.0/tau=2.0/T=8，
    final test 56.87%），而非 lif_F_seed123.pth。
  - 四分类权重的 final_test 是 float（不是二分类的 dict['干净']/['HBC噪声']），
    打印逻辑据此适配。
  - 其余（BN 折叠公式 / LIF 发放率语义 / C 头格式）完全复用
    export_lif_surrogate_weights.py 的 build_folded_snn / export_header，保证
    与二分类部署链同源、逐位一致。

导出的 C 头 lif_surrogate_weights_4class.h 结构（22ch 四分类）：
    #define EEGNET_N_CH 22 / EEGNET_T 500 / EEGNET_F1 8 / EEGNET_D 2 / EEGNET_F2 16 / EEGNET_T_OUT 15
    #define SNN_N_CLASSES 4
    #define SNN_TIMESTEPS 8
    #define SNN_THR1 1.0f / SNN_THR2 1.0f / SNN_DECAY 0.606530666f
    conv1_w[8][1][1][63] + b、depthwise_w[16][1][22][1] + b、sep_dw_w[16][1][1][15]、
    pointwise_w[16][16][1][1] + b、fc_w[4][240] + b

用法
----
  env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin \
      /opt/miniconda3/envs/openmmlab/bin/python export_lif_surrogate_weights_4class.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ALGO_DIR = Path(__file__).resolve().parent
DEFAULT_CKPT = ALGO_DIR / "lif_4class_t1.0_tau2_T8.pth"
BOARD_DEMO_DIR = ALGO_DIR.parent / "board_demo"

sys.path.insert(0, str(ALGO_DIR))
# 复用二分类 export 脚本的折叠 + 导出逻辑（只读引用，不改原文件）
from export_lif_surrogate_weights import build_folded_snn, export_header  # noqa: E402


def main() -> int:
    ckpt = torch.load(DEFAULT_CKPT, map_location="cpu", weights_only=False)
    assert {"state_dict", "model_args", "lif_hparams"} <= set(ckpt), \
        f"✗ {DEFAULT_CKPT.name} 不是 surrogate LIF-EEGNet 权重"

    hp = ckpt["lif_hparams"]
    print(f"[加载] {DEFAULT_CKPT.name}")
    # 四分类 final_test 是 float（train_4class.py 保存的标量），非二分类的 dict
    ft = ckpt["final_test"]
    if isinstance(ft, dict):
        print(f"  final_test: 干净 {ft['干净']*100:.2f}% / HBC噪声 {ft['HBC噪声']*100:.2f}%")
    else:
        print(f"  final_test(eval): {ft*100:.2f}%（cross-session 四分类）")
    print(f"  model_args: {ckpt['model_args']}")
    print(f"  lif_hparams: {hp}")

    folded, meta = build_folded_snn(ckpt)
    # 产出两个副本：algorithm/（源）与 board_demo/（板侧部署目录）
    out_algo = ALGO_DIR / "lif_surrogate_weights_4class.h"
    out_board = BOARD_DEMO_DIR / "lif_surrogate_weights_4class.h"
    export_header(folded, meta, out_algo)
    export_header(folded, meta, out_board)
    return 0


if __name__ == "__main__":
    sys.exit(main())
