#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
export_snn_weights.py — 把折叠后的 SNN 权重导出为 C 头文件（供 lif_eegnet_infer.c 使用）

与 luckfox-deploy/model/export_c_weights.py 的格式对齐，差异：
  - 权重是 **已折叠** 的（conv1+bn1、depthwise+bn2、pointwise+bn3 折叠；sep_depthwise/fc 无 BN）；
  - 额外导出 SNN 专属超参：thr1 / thr2（发放阈值）、timesteps、n_classes。

导出的 C 头文件 board_demo/snn_weights.h 结构：
    #define EEGNET_N_CH 8 / EEGNET_T 500 / EEGNET_F1 8 / EEGNET_D 2 / EEGNET_F2 16 / EEGNET_T_OUT 15
    #define SNN_N_CLASSES 2
    #define SNN_TIMESTEPS 16
    #define SNN_THR1 1.2345f
    #define SNN_THR2 2.3456f
    conv1_w[8][1][1][63], conv1_b[8]
    depthwise_w[16][1][8][1], depthwise_b[16]
    sep_dw_w[16][1][1][15]
    pointwise_w[16][16][1][1], pointwise_b[16]
    fc_w[n_classes][240], fc_b[n_classes]

用法
----
  # 由 convert_snn.py 生成 snn_weights.pth 后：
  env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin \
      /opt/miniconda3/envs/openmmlab/bin/python export_snn_weights.py \
      --state snn_weights.pth --out ../board_demo/snn_weights.h

架构硬要求：n_classes 可配置（默认 2），不写死（fc_w 行数 = n_classes）。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ALGO_DIR = Path(__file__).resolve().parent
DEFAULT_HEADER = ALGO_DIR.parent / "board_demo" / "snn_weights.h"


def export_from_state(state: dict, out_path: Path) -> None:
    """从 snn_weights.pth 的 state dict 导出 C 头文件。"""
    n_classes = int(state["n_classes"])
    n_ch = int(state.get("n_channels", 8))
    T = int(state.get("input_length", 500))
    F1 = int(state.get("F1", 8))
    D = int(state.get("D", 2))
    F2 = int(state.get("F2", 16))
    kernel_length = int(state.get("kernel_length", 63))
    timesteps = int(state["timesteps"])
    thr1 = float(state["thr1"])
    thr2 = float(state["thr2"])

    def to_np(k):
        v = state[k]
        return v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else np.asarray(v, dtype=np.float32)

    conv1_w = to_np("conv1_weight")            # (F1,1,1,63)
    conv1_b = to_np("conv1_bias")              # (F1,)
    dw_w = to_np("depthwise_weight")           # (D*F1,1,n_ch,1)
    dw_b = to_np("depthwise_bias")             # (D*F1,)
    sdw_w = to_np("sep_depthwise_weight")      # (D*F1,1,1,15)
    pw_w = to_np("pointwise_weight")           # (F2,D*F1,1,1)
    pw_b = to_np("pointwise_bias")             # (F2,)
    fc_w = to_np("fc_weight").reshape(n_classes, -1)   # (n_classes, F2*T_OUT)
    fc_b = to_np("fc_bias")                    # (n_classes,)

    t_out = T // 32

    def fmt_array(name: str, arr: np.ndarray) -> str:
        flat = arr.ravel()
        vals = ", ".join(f"{v:.8e}f" for v in flat)
        return f"static const float {name}[{len(flat)}] = {{ {vals} }};\n"

    lines = [
        "/* 自动生成 —— 勿手改。来源: algorithm/export_snn_weights.py */",
        "/* LIF-EEGNet（SNN）折叠权重：conv1+bn1、depthwise+bn2、pointwise+bn3 已折叠；",
        "   sep_depthwise / fc 无 BN。thr1/thr2 为发放阈值（ann2snn 标定）。 */",
        "#ifndef SNN_WEIGHTS_H",
        "#define SNN_WEIGHTS_H",
        "",
        f"#define EEGNET_N_CH {n_ch}",
        f"#define EEGNET_T {T}",
        f"#define EEGNET_F1 {F1}",
        f"#define EEGNET_D {D}",
        f"#define EEGNET_F2 {F2}",
        f"#define EEGNET_T_OUT {t_out}",
        "",
        f"#define SNN_N_CLASSES {n_classes}",
        f"#define SNN_TIMESTEPS {timesteps}",
        f"#define SNN_THR1 {thr1:.8e}f",
        f"#define SNN_THR2 {thr2:.8e}f",
        "",
        fmt_array("conv1_w", conv1_w),
        fmt_array("conv1_b", conv1_b),
        fmt_array("depthwise_w", dw_w),
        fmt_array("depthwise_b", dw_b),
        fmt_array("sep_dw_w", sdw_w),
        fmt_array("pointwise_w", pw_w),
        fmt_array("pointwise_b", pw_b),
        fmt_array("fc_w", fc_w),
        fmt_array("fc_b", fc_b),
        "",
        "#endif /* SNN_WEIGHTS_H */",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    total = sum(a.size for a in (conv1_w, conv1_b, dw_w, dw_b, sdw_w, pw_w, pw_b, fc_w, fc_b))
    print(f"✓ 已导出 SNN C 头文件：{out_path}")
    print(f"  conv1_w {conv1_w.shape}+b {conv1_b.shape} | depthwise_w {dw_w.shape}+b {dw_b.shape}")
    print(f"  sep_dw_w {sdw_w.shape} | pointwise_w {pw_w.shape}+b {pw_b.shape} | fc_w {fc_w.shape}+b {fc_b.shape}")
    print(f"  thr1={thr1:.6f}, thr2={thr2:.6f}, timesteps={timesteps}, n_classes={n_classes}")
    print(f"  总参数量: {total}")


def export(snn, out_path: Path, n_classes: int, timesteps: int,
           thr1: float, thr2: float) -> None:
    """从 SNNEEGNet 实例直接导出（备用入口，给 convert_snn 或外部脚本调用）。"""
    sd = snn.state_dict()
    state = {
        "conv1_weight": sd["conv1.weight"], "conv1_bias": sd["conv1.bias"],
        "depthwise_weight": sd["depthwise.weight"], "depthwise_bias": sd["depthwise.bias"],
        "sep_depthwise_weight": sd["sep_depthwise.weight"],
        "pointwise_weight": sd["pointwise.weight"], "pointwise_bias": sd["pointwise.bias"],
        "fc_weight": sd["fc.weight"], "fc_bias": sd["fc.bias"],
        "thr1": float(thr1), "thr2": float(thr2),
        "n_classes": int(n_classes), "timesteps": int(timesteps),
        "n_channels": snn.pad1_amt * 2 + 1 if hasattr(snn, "pad1_amt") else 8,
        "input_length": 500, "F1": 8, "D": 2, "F2": snn.F2,
        "kernel_length": 63,
    }
    export_from_state(state, out_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="导出折叠后 SNN 权重为 C 头文件")
    parser.add_argument("--state", type=str, default=str(ALGO_DIR / "snn_weights.pth"))
    parser.add_argument("--out", type=str, default=str(DEFAULT_HEADER))
    args = parser.parse_args()
    state = torch.load(Path(args.state), map_location="cpu", weights_only=False)
    export_from_state(state, Path(args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
