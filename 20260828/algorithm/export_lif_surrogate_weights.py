#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
export_lif_surrogate_weights.py — 把 surrogate 训练的最优 LIF-EEGNet 权重导出为 C 头文件

与 export_snn_weights.py 的区别（关键语义，务必对齐）：
  - 加载 **lif_F_seed123.pth**（surrogate 直接训练的 LIF-EEGNet，activation='lif'），
    不是 ann2snn 转换后的 snn_weights.pth。
  - BN 折叠进卷积：conv1+bn1、depthwise+bn2、pointwise+bn3；sep_depthwise / fc 无 BN。
  - LIF 语义是 **发放率版**（rate-based）：LIF 在 pool **之前**、decay=exp(-1/tau)=0.6065、
    thr=1.0、timesteps=8、输出 = sum(spk)/timesteps（连续值 0~1）。
    这与 ann2snn 版（pool 后脉冲、thr 标定、decay=1.0）**不同**。

导出的 C 头文件 lif_surrogate_weights.h 结构：
    #define EEGNET_N_CH 8 / EEGNET_T 500 / EEGNET_F1 8 / EEGNET_D 2 / EEGNET_F2 16 / EEGNET_T_OUT 15
    #define SNN_N_CLASSES 2
    #define SNN_TIMESTEPS 8
    #define SNN_THR1 1.0f  / SNN_THR2 1.0f  / SNN_DECAY 0.6065306663513184f
    conv1_w[8][1][1][63] + b、depthwise_w[16][1][8][1] + b、sep_dw_w[16][1][1][15]、
    pointwise_w[16][16][1][1] + b、fc_w[2][240] + b

用法
----
  env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin \
      /opt/miniconda3/envs/openmmlab/bin/python export_lif_surrogate_weights.py \
      --checkpoint lif_F_seed123.pth --out ../board_demo/lif_surrogate_weights.h
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch

ALGO_DIR = Path(__file__).resolve().parent
DEFAULT_CKPT = ALGO_DIR / "lif_F_seed123.pth"
DEFAULT_HEADER = ALGO_DIR.parent / "board_demo" / "lif_surrogate_weights.h"

# 复用 lif_eegnet.py 的 LIF-EEGNet 与 ann2snn.py 的 BN 折叠公式（只读引用）
sys.path.insert(0, str(ALGO_DIR))
from lif_eegnet import LIFEEGNet  # noqa: E402
from ann2snn import fold_bn_layer, clone_conv  # noqa: E402


def build_folded_snn(ckpt: dict) -> tuple[dict, dict]:
    """加载 LIF-EEGNet 权重并折叠 BN，返回 (folded_state_dict, 元信息)。"""
    model_args = ckpt["model_args"]
    lif_hp = ckpt["lif_hparams"]
    threshold = float(lif_hp.get("threshold", 1.0))
    tau = float(lif_hp.get("tau", 2.0))
    timesteps = int(lif_hp.get("timesteps", 8))
    alpha = float(lif_hp.get("alpha", 4.0))

    ann = LIFEEGNet(**model_args, threshold=threshold, tau=tau,
                    timesteps=timesteps, alpha=alpha)
    ann.load_state_dict(ckpt["state_dict"], strict=True)
    ann.eval()

    # ---- BN 折叠（与 ann2snn.fold_bn_layer 完全一致，保证与验证参考逐位一致）----
    conv1 = fold_bn_layer(ann.conv1, ann.bn1)
    depthwise = fold_bn_layer(ann.depthwise, ann.bn2)
    sep_depthwise = clone_conv(ann.sep_depthwise)   # 无 BN，原样复制
    pointwise = fold_bn_layer(ann.pointwise, ann.bn3)
    fc = clone_conv(ann.fc)                          # 线性读出，无 BN

    folded = {
        "conv1_weight": conv1.weight.data, "conv1_bias": conv1.bias.data,
        "depthwise_weight": depthwise.weight.data, "depthwise_bias": depthwise.bias.data,
        "sep_depthwise_weight": sep_depthwise.weight.data,
        "pointwise_weight": pointwise.weight.data, "pointwise_bias": pointwise.bias.data,
        "fc_weight": fc.weight.data, "fc_bias": fc.bias.data,
    }
    decay = math.exp(-1.0 / tau)
    meta = {
        "n_classes": int(model_args["n_classes"]),
        "n_channels": int(model_args["n_channels"]),
        "input_length": int(model_args["input_length"]),
        "F1": int(model_args["F1"]), "D": int(model_args["D"]),
        "F2": int(model_args["F2"]), "kernel_length": int(model_args["kernel_length"]),
        "timesteps": timesteps, "threshold": threshold, "tau": tau, "decay": decay,
    }
    return folded, meta


def export_header(folded: dict, meta: dict, out_path: Path) -> None:
    """把折叠后的权重 + SNN 超参写入 C 头文件。"""
    n_classes = meta["n_classes"]
    n_ch = meta["n_channels"]
    T = meta["input_length"]
    F1 = meta["F1"]
    D = meta["D"]
    F2 = meta["F2"]
    timesteps = meta["timesteps"]
    thr = meta["threshold"]
    decay = meta["decay"]
    t_out = T // 32

    def to_np(k):
        v = folded[k]
        return v.detach().cpu().numpy().astype(np.float32) if isinstance(v, torch.Tensor) \
            else np.asarray(v, dtype=np.float32)

    conv1_w = to_np("conv1_weight")            # (F1,1,1,63)
    conv1_b = to_np("conv1_bias")              # (F1,)
    dw_w = to_np("depthwise_weight")           # (D*F1,1,n_ch,1)
    dw_b = to_np("depthwise_bias")             # (D*F1,)
    sdw_w = to_np("sep_depthwise_weight")      # (D*F1,1,1,15)
    pw_w = to_np("pointwise_weight")           # (F2,D*F1,1,1)
    pw_b = to_np("pointwise_bias")             # (F2,)
    fc_w = to_np("fc_weight").reshape(n_classes, -1)   # (n_classes, F2*T_OUT)
    fc_b = to_np("fc_bias")                    # (n_classes,)

    def fmt_array(name: str, arr: np.ndarray) -> str:
        # float32 精确往返需要 ≤9 位有效数字，%.8e 恰好 9 位
        flat = arr.ravel()
        vals = ", ".join(f"{float(v):.8e}f" for v in flat)
        return f"static const float {name}[{len(flat)}] = {{ {vals} }};\n"

    lines = [
        "/* 自动生成 —— 勿手改。来源: algorithm/export_lif_surrogate_weights.py */",
        "/* LIF-EEGNet（surrogate 训练，发放率版）：conv1+bn1、depthwise+bn2、pointwise+bn3 已折叠；",
        "   sep_depthwise / fc 无 BN。LIF 在 pool 前、decay=exp(-1/tau)、输出=发放率。 */",
        "#ifndef LIF_SURROGATE_WEIGHTS_H",
        "#define LIF_SURROGATE_WEIGHTS_H",
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
        f"#define SNN_THR1 {thr:.8e}f",
        f"#define SNN_THR2 {thr:.8e}f",
        f"#define SNN_DECAY {decay:.9e}f",
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
        "#endif /* LIF_SURROGATE_WEIGHTS_H */",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    total = sum(a.size for a in (conv1_w, conv1_b, dw_w, dw_b, sdw_w, pw_w, pw_b, fc_w, fc_b))
    print(f"✓ 已导出 LIF-SNN C 头文件：{out_path}")
    print(f"  conv1_w {conv1_w.shape}+b {conv1_b.shape} | depthwise_w {dw_w.shape}+b {dw_b.shape}")
    print(f"  sep_dw_w {sdw_w.shape} | pointwise_w {pw_w.shape}+b {pw_b.shape} | fc_w {fc_w.shape}+b {fc_b.shape}")
    print(f"  thr1=thr2={thr:.6f}, decay={decay:.9f}, timesteps={timesteps}, n_classes={n_classes}")
    print(f"  总参数量: {total}")


def main() -> int:
    parser = argparse.ArgumentParser(description="导出 surrogate LIF-EEGNet 折叠权重为 C 头文件")
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CKPT))
    parser.add_argument("--out", type=str, default=str(DEFAULT_HEADER))
    args = parser.parse_args()

    cp = Path(args.checkpoint)
    if not cp.exists():
        raise SystemExit(f"✗ 权重文件不存在：{cp}")
    ckpt = torch.load(cp, map_location="cpu", weights_only=False)
    assert "state_dict" in ckpt and "model_args" in ckpt and "lif_hparams" in ckpt, \
        f"✗ {cp.name} 不是 surrogate LIF-EEGNet 权重（缺 state_dict/model_args/lif_hparams）"

    print(f"[加载] {cp.name}")
    print(f"  final_test: 干净 {ckpt['final_test']['干净']*100:.2f}% / HBC噪声 {ckpt['final_test']['HBC噪声']*100:.2f}%")
    print(f"  lif_hparams: {ckpt['lif_hparams']}")
    folded, meta = build_folded_snn(ckpt)
    export_header(folded, meta, Path(args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
