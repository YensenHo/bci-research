#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
export_tcnet_snn_weights.py — 把 surrogate 训练的最优 LIF-EEG-TCNet 权重导出为 C 头文件

与 export_lif_surrogate_weights_4class.py（LIF-EEGNet 四分类版）的关键差异：
  - 模型：EEGTCNetLIF（eeg_tcnet_lif.py，brief 配置），不是 LIFEEGNet。
    前端与 EEGNet 相同的 φ1/φ2/φ3（conv1/depthwise/separable），
    但 φ4 换成 **TCN 两个残差块**（每块两层膨胀因果 Conv1d + 1×1 skip 投影），
    读出头是 **GAP（时间维全局平均池化）→ Linear**，不是 FC(240→4)。
  - BN 折叠位点比 EEGNet 版多 4 处：conv1+bn1、depthwise+bn2、pointwise+bn3
    （前端 3 处，同 EEGNet 版）+ **TCN 每块 conv_a+bn_a、conv_b+bn_b**
    （4 处 Conv1d 折叠）。sep_depthwise / TCN skip 卷积 / fc 无 BN，原样复制。
  - LIF 位点共 8 个：act1/act2（前端 pool 前）+ TCN 每残差块
    act_a/act_b/act_res（2 块 × 3），全部 thr=1.0、decay=exp(-1/tau)=0.6065、
    timesteps=8，输出 = 发放率 sum(spk)/T ∈ [0,1]（与 LIF-EEGNet 语义一致）。

导出的 C 头文件 lif_tcnet_snn_weights.h 结构：
    #define EEGNET_N_CH 22 / EEGNET_T 500 / EEGNET_F1 8 / EEGNET_D 2
    #define EEGNET_F2 16 / EEGNET_FT 12 / EEGNET_T4 125 / EEGNET_T15 15
    #define EEGNET_KT 12 / EEGNET_SEP_K 16
    #define SNN_N_CLASSES 4 / SNN_TIMESTEPS 8 / SNN_THR1..8 1.0f / SNN_DECAY 0.6065...
    conv1_w[8][63]+b、depthwise_w[16][22]+b、sep_dw_w[16][16]、
    pointwise_w[16][16]+b、tcn0_conv_a_w[12][16][12]+b、tcn0_conv_b_w[12][12][12]+b、
    tcn0_skip_w[12][16]+b、tcn1_conv_a_w[12][12][12]+b、tcn1_conv_b_w[12][12][12]+b、
    tcn1_skip_w[12][12]、fc_w[4][12]+b

每个数值用 %.8e（9 位有效数字，float32 精确往返），并做 round-trip 断言
（把格式化后的字符串解析回 float32 必须与原始值完全相等），保证 C 头与
PyTorch 权重逐位一致。

用法
----
  env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin \\
      /opt/miniconda3/envs/openmmlab/bin/python export_tcnet_snn_weights.py \\
      --checkpoint eeg_tcnet_4class_lif_t1.0_tau2.0_T8.pth \\
      --out ../board_demo/lif_tcnet_snn_weights.h
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ALGO_DIR = Path(__file__).resolve().parent
DEFAULT_CKPT = ALGO_DIR / "eeg_tcnet_4class_lif_t1.0_tau2.0_T8.pth"
DEFAULT_HEADER = ALGO_DIR.parent / "board_demo" / "lif_tcnet_snn_weights.h"

# 复用同目录的模型定义与 BN 折叠公式（只读引用）
sys.path.insert(0, str(ALGO_DIR))
from eeg_tcnet_lif import EEGTCNetLIF  # noqa: E402
from ann2snn import clone_conv         # noqa: E402  # Conv2d 克隆（前端 sep_depthwise）


def fold_bn_conv1d(conv: nn.Conv1d, bn: nn.BatchNorm1d, eps: float = 1e-5) -> nn.Conv1d:
    """把 BatchNorm1d 折叠进前一层 Conv1d（TCN 膨胀因果卷积用）。

    公式（逐输出通道仿射，与 ann2snn.fold_bn_layer 的 Conv2d 版同构）：
        scale = γ / √(σ² + ε)
        W'    = W · scale                     （逐输出通道缩放核）
        b'    = b·scale + (β - μ·scale)
    """
    w = conv.weight.data                              # (out, in, k)
    b = conv.bias.data if conv.bias is not None else torch.zeros(
        conv.out_channels, device=conv.weight.device)
    scale = bn.weight.data / torch.sqrt(bn.running_var + eps)
    shift = bn.bias.data - bn.running_mean * scale
    w_folded = w * scale.view(-1, 1, 1)
    b_folded = b * scale + shift
    new_conv = nn.Conv1d(conv.in_channels, conv.out_channels, conv.kernel_size,
                         stride=conv.stride, padding=conv.padding,
                         dilation=conv.dilation, groups=conv.groups, bias=True)
    new_conv.weight.data = w_folded
    new_conv.bias.data = b_folded
    return new_conv


def clone_conv1d(conv: nn.Conv1d) -> nn.Conv1d:
    """复制一个 Conv1d（保持 bias 与否一致）。用于 TCN 的 1×1 skip 投影卷积。"""
    new_conv = nn.Conv1d(conv.in_channels, conv.out_channels, conv.kernel_size,
                         stride=conv.stride, padding=conv.padding,
                         dilation=conv.dilation, groups=conv.groups,
                         bias=(conv.bias is not None))
    new_conv.weight.data = conv.weight.data.clone()
    if conv.bias is not None:
        new_conv.bias.data = conv.bias.data.clone()
    return new_conv


def build_folded_snn(ckpt: dict) -> tuple[dict, dict]:
    """加载 LIF-EEG-TCNet 权重并折叠全部 BN，返回 (folded_state_dict, 元信息)。"""
    model_args = ckpt["model_args"]
    lif_hp = ckpt["lif_hparams"]
    threshold = float(lif_hp.get("threshold", 1.0))
    tau = float(lif_hp.get("tau", 2.0))
    timesteps = int(lif_hp.get("timesteps", 8))
    alpha = float(lif_hp.get("alpha", 4.0))

    ann = EEGTCNetLIF(**model_args, threshold=threshold, tau=tau,
                      timesteps=timesteps, alpha=alpha)
    ann.load_state_dict(ckpt["state_dict"], strict=True)
    ann.eval()

    # ---- 前端 BN 折叠（同 LIF-EEGNet 版，Conv2d）----
    from ann2snn import fold_bn_layer  # noqa: PLC0415  # 延迟导入避免循环
    conv1 = fold_bn_layer(ann.conv1, ann.bn1)
    depthwise = fold_bn_layer(ann.depthwise, ann.bn2)
    sep_depthwise = clone_conv(ann.sep_depthwise)     # 无 BN，原样复制
    pointwise = fold_bn_layer(ann.pointwise, ann.bn3)

    # ---- TCN：两层膨胀因果卷积 + BN 折叠；skip 1×1 无 BN 原样复制 ----
    tcn = []
    for blk in ann.tcn:
        tcn.append({
            "conv_a": fold_bn_conv1d(blk.conv_a, blk.bn_a),
            "conv_b": fold_bn_conv1d(blk.conv_b, blk.bn_b),
            "skip": clone_conv1d(blk.skip),
        })

    # Linear 读出（GAP 后），无 BN：直接取 weight/bias（clone_conv 只支持 Conv2d）
    fc = ann.fc

    folded = {
        "conv1_weight": conv1.weight.data, "conv1_bias": conv1.bias.data,
        "depthwise_weight": depthwise.weight.data, "depthwise_bias": depthwise.bias.data,
        "sep_depthwise_weight": sep_depthwise.weight.data,
        "pointwise_weight": pointwise.weight.data, "pointwise_bias": pointwise.bias.data,
        "tcn0_conv_a_weight": tcn[0]["conv_a"].weight.data,
        "tcn0_conv_a_bias": tcn[0]["conv_a"].bias.data,
        "tcn0_conv_b_weight": tcn[0]["conv_b"].weight.data,
        "tcn0_conv_b_bias": tcn[0]["conv_b"].bias.data,
        "tcn0_skip_weight": tcn[0]["skip"].weight.data,
        "tcn0_skip_bias": tcn[0]["skip"].bias.data,
        "tcn1_conv_a_weight": tcn[1]["conv_a"].weight.data,
        "tcn1_conv_a_bias": tcn[1]["conv_a"].bias.data,
        "tcn1_conv_b_weight": tcn[1]["conv_b"].weight.data,
        "tcn1_conv_b_bias": tcn[1]["conv_b"].bias.data,
        "tcn1_skip_weight": tcn[1]["skip"].weight.data,
        "fc_weight": fc.weight.data, "fc_bias": fc.bias.data,
    }
    decay = math.exp(-1.0 / tau)
    meta = {
        "n_classes": int(model_args["n_classes"]),
        "n_channels": int(model_args["n_channels"]),
        "input_length": int(model_args["input_length"]),
        "F1": int(model_args["F1"]), "D": int(model_args["D"]),
        "F2": int(model_args["F2"]), "FT": int(model_args["FT"]),
        "kernel_length": int(model_args["kernel_length"]),
        "sep_kernel": int(ann.sep_kernel),
        "tcn_kernel_size": int(model_args["tcn_kernel_size"]),
        "pool1": int(model_args["pool1"]), "pool2": int(model_args["pool2"]),
        "readout_mode": model_args["readout_mode"],
        "timesteps": timesteps, "threshold": threshold, "tau": tau, "decay": decay,
    }
    return folded, meta


def export_header(folded: dict, meta: dict, out_path: Path) -> None:
    """把折叠后的权重 + SNN 超参写入 C 头文件（数值 %.8e + round-trip 断言）。"""
    n_classes = meta["n_classes"]
    n_ch = meta["n_channels"]
    T = meta["input_length"]
    F1, D, F2, FT = meta["F1"], meta["D"], meta["F2"], meta["FT"]
    KE = meta["kernel_length"]
    SEP_K = meta["sep_kernel"]
    KT = meta["tcn_kernel_size"]
    pool1, pool2 = meta["pool1"], meta["pool2"]
    t4 = T // pool1                 # 125
    t15 = t4 // pool2               # 15（TCN 输入时间步）
    timesteps = meta["timesteps"]
    thr = meta["threshold"]
    decay = meta["decay"]

    def to_np(k):
        v = folded[k]
        return v.detach().cpu().numpy().astype(np.float32) if isinstance(v, torch.Tensor) \
            else np.asarray(v, dtype=np.float32)

    conv1_w = to_np("conv1_weight")            # (F1,1,1,KE)
    conv1_b = to_np("conv1_bias")              # (F1,)
    dw_w = to_np("depthwise_weight")           # (D*F1,1,n_ch,1)
    dw_b = to_np("depthwise_bias")             # (D*F1,)
    sdw_w = to_np("sep_depthwise_weight")      # (D*F1,1,1,SEP_K)
    pw_w = to_np("pointwise_weight")           # (F2,D*F1,1,1)
    pw_b = to_np("pointwise_bias")             # (F2,)
    t0a_w = to_np("tcn0_conv_a_weight")        # (FT,F2,KT)
    t0a_b = to_np("tcn0_conv_a_bias")          # (FT,)
    t0b_w = to_np("tcn0_conv_b_weight")        # (FT,FT,KT)
    t0b_b = to_np("tcn0_conv_b_bias")          # (FT,)
    t0s_w = to_np("tcn0_skip_weight")          # (FT,F2,1)
    t0s_b = to_np("tcn0_skip_bias")            # (FT,)
    t1a_w = to_np("tcn1_conv_a_weight")        # (FT,FT,KT)
    t1a_b = to_np("tcn1_conv_a_bias")          # (FT,)
    t1b_w = to_np("tcn1_conv_b_weight")        # (FT,FT,KT)
    t1b_b = to_np("tcn1_conv_b_bias")          # (FT,)
    t1s_w = to_np("tcn1_skip_weight")          # (FT,FT,1)
    fc_w = to_np("fc_weight").reshape(n_classes, -1)   # (n_classes, FT)
    fc_b = to_np("fc_bias")                    # (n_classes,)

    # ---- 数值格式化：%.8e（9 位有效数字，float32 精确往返）+ round-trip 断言 ----
    def fmt_array(name: str, arr: np.ndarray) -> str:
        flat = arr.ravel()
        vals = []
        for v in flat:
            lit = f"{float(v):.8e}"
            back = np.float32(np.float32(lit))       # 字符串 → float32
            assert back == np.float32(v), \
                f"round-trip 失败: {name} 元素 {float(v):.9e} → '{lit}' → {float(back):.9e}"
            vals.append(lit + "f")
        return f"static const float {name}[{len(flat)}] = {{ {', '.join(vals)} }};\n"

    # decay 同样做 round-trip 断言（头文件宏必须是 float32 精确值）
    decay_lit = f"{float(decay):.8e}"
    assert np.float32(np.float32(decay_lit)) == np.float32(decay), \
        f"decay round-trip 失败: {decay_lit}"
    thr_lit = f"{float(thr):.8e}"
    assert np.float32(np.float32(thr_lit)) == np.float32(thr), \
        f"thr round-trip 失败: {thr_lit}"

    thr_macros = "".join(f"#define SNN_THR{i} {thr_lit}f\n" for i in range(1, 9))

    lines = [
        "/* 自动生成 —— 勿手改。来源: algorithm/export_tcnet_snn_weights.py */",
        "/* LIF-EEG-TCNet（surrogate 训练，发放率版，brief 配置 KE=63/KT=12/pool4/8/GAP）：",
        "   BN 折叠位点 = conv1+bn1 / depthwise+bn2 / pointwise+bn3 /",
        "   TCN 每块 conv_a+bn_a / conv_b+bn_b；sep_depthwise / TCN skip / fc 无 BN。",
        "   8 个 LIF 位点（act1/act2 + 2 残差块 × act_a/act_b/act_res）全部 thr=1.0，",
        "   decay=exp(-1/tau)=0.6065，timesteps=8，输出=发放率 sum(spk)/T。",
        "   读出头 = GAP（时间维均值）→ Linear(FT→4)。 */",
        "#ifndef LIF_TCNET_SNN_WEIGHTS_H",
        "#define LIF_TCNET_SNN_WEIGHTS_H",
        "",
        f"#define EEGNET_N_CH {n_ch}",
        f"#define EEGNET_T {T}",
        f"#define EEGNET_F1 {F1}",
        f"#define EEGNET_D {D}",
        f"#define EEGNET_F2 {F2}",
        f"#define EEGNET_FT {FT}",
        f"#define EEGNET_KE {KE}",
        f"#define EEGNET_SEP_K {SEP_K}",
        f"#define EEGNET_KT {KT}",
        f"#define EEGNET_POOL1 {pool1}",
        f"#define EEGNET_POOL2 {pool2}",
        f"#define EEGNET_T4 {t4}",
        f"#define EEGNET_T15 {t15}",
        "",
        f"#define SNN_N_CLASSES {n_classes}",
        f"#define SNN_TIMESTEPS {timesteps}",
        thr_macros,
        f"#define SNN_DECAY {decay_lit}f",
        "",
        fmt_array("conv1_w", conv1_w),
        fmt_array("conv1_b", conv1_b),
        fmt_array("depthwise_w", dw_w),
        fmt_array("depthwise_b", dw_b),
        fmt_array("sep_dw_w", sdw_w),
        fmt_array("pointwise_w", pw_w),
        fmt_array("pointwise_b", pw_b),
        fmt_array("tcn0_conv_a_w", t0a_w),
        fmt_array("tcn0_conv_a_b", t0a_b),
        fmt_array("tcn0_conv_b_w", t0b_w),
        fmt_array("tcn0_conv_b_b", t0b_b),
        fmt_array("tcn0_skip_w", t0s_w),
        fmt_array("tcn0_skip_b", t0s_b),
        fmt_array("tcn1_conv_a_w", t1a_w),
        fmt_array("tcn1_conv_a_b", t1a_b),
        fmt_array("tcn1_conv_b_w", t1b_w),
        fmt_array("tcn1_conv_b_b", t1b_b),
        fmt_array("tcn1_skip_w", t1s_w),
        fmt_array("fc_w", fc_w),
        fmt_array("fc_b", fc_b),
        "",
        "#endif /* LIF_TCNET_SNN_WEIGHTS_H */",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    total = sum(a.size for a in (conv1_w, conv1_b, dw_w, dw_b, sdw_w, pw_w, pw_b,
                                 t0a_w, t0a_b, t0b_w, t0b_b, t0s_w, t0s_b,
                                 t1a_w, t1a_b, t1b_w, t1b_b, t1s_w,
                                 fc_w, fc_b))
    print(f"✓ 已导出 LIF-EEG-TCNet SNN C 头文件：{out_path}")
    print(f"  conv1_w {conv1_w.shape}+b | depthwise_w {dw_w.shape}+b | sep_dw_w {sdw_w.shape}")
    print(f"  pointwise_w {pw_w.shape}+b | tcn0 conv_a {t0a_w.shape}+b conv_b {t0b_w.shape}+b "
          f"skip {t0s_w.shape}+b")
    print(f"  tcn1 conv_a {t1a_w.shape}+b conv_b {t1b_w.shape}+b skip {t1s_w.shape}（无偏置）")
    print(f"  fc_w {fc_w.shape}+b | thr1..8={thr_lit}f, decay={decay_lit}f, "
          f"timesteps={timesteps}, n_classes={n_classes}")
    print(f"  readout={meta['readout_mode']}（GAP 后 Linear），T4={t4}, T15={t15}")
    print(f"  总参数量: {total}")


def main() -> int:
    parser = argparse.ArgumentParser(description="导出 surrogate LIF-EEG-TCNet 折叠权重为 C 头文件")
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CKPT))
    parser.add_argument("--out", type=str, default=str(DEFAULT_HEADER))
    args = parser.parse_args()

    cp = Path(args.checkpoint)
    if not cp.exists():
        raise SystemExit(f"✗ 权重文件不存在：{cp}")
    ckpt = torch.load(cp, map_location="cpu", weights_only=False)
    assert "state_dict" in ckpt and "model_args" in ckpt and "lif_hparams" in ckpt, \
        f"✗ {cp.name} 不是 surrogate LIF-EEG-TCNet 权重（缺 state_dict/model_args/lif_hparams）"

    print(f"[加载] {cp.name}")
    ft = ckpt.get("final_test")
    if isinstance(ft, dict):
        print(f"  final_test: 干净 {ft.get('干净', float('nan'))*100:.2f}% / "
              f"HBC噪声 {ft.get('HBC噪声', float('nan'))*100:.2f}%")
    else:
        print(f"  final_test: {ft*100:.2f}%（标量）")
    print(f"  lif_hparams: {ckpt['lif_hparams']}")
    folded, meta = build_folded_snn(ckpt)
    export_header(folded, meta, Path(args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
