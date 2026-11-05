#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_tcnet_bit_exact.py — LIF-EEG-TCNet C 版 vs numpy(float32, C 顺序) 严格逐位一致验证

背景：PyTorch 的卷积用优化内核（累加顺序与 C 朴素循环不同），浮点非结合性会让
      pre-LIF 电流差 ~1e-6，极少数膜电位恰在阈值附近的元素会翻转发放决策。这是
      浮点舍入的固有现象，不是 bug。本脚本用 numpy float32 + 与 C 完全相同的累加
      顺序实现参考前向，对比 C 输出，应得到 max|diff| == 0（完全一致）。若 C 移植
      有结构性错误（如膨胀因果卷积的索引、残差 add、GAP、TCN skip 布局），这里会
      暴露。

覆盖的结构要点（LIF-EEG-TCNet 相对 LIF-EEGNet 的新增难点）：
  - TCN 膨胀因果 Conv1d：out[t] = Σ_j w[j]·in[t + j·d - (KT-1)·d]，块0 d=1 / 块1 d=2
  - 残差连接：1×1 skip 投影（块0 带偏置 / 块1 无偏置）+ 逐元素 add → act_res
  - GAP 读出头：TCN 输出 15 个时间步均值 → Linear(12→4)

用法：
  env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin \
      /opt/miniconda3/envs/openmmlab/bin/python verify_tcnet_bit_exact.py --n 20
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ALGO_DIR = Path(__file__).resolve().parent.parent / "algorithm"
BOARD_DEMO_DIR = Path(__file__).resolve().parent
CKPT = ALGO_DIR / "eeg_tcnet_4class_lif_t1.0_tau2.0_T8.pth"
DATA = ALGO_DIR / "data_4class" / "eval.npz"
HDR = BOARD_DEMO_DIR / "lif_tcnet_snn_weights.h"

sys.path.insert(0, str(ALGO_DIR))


def load_folded_np():
    """用与 export 脚本相同的折叠逻辑得到 numpy float32 权重（TCNet brief 配置）。"""
    import torch
    from eeg_tcnet_lif import EEGTCNetLIF
    from export_tcnet_snn_weights import fold_bn_conv1d, clone_conv1d
    from ann2snn import fold_bn_layer, clone_conv
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    ma, hp = ck["model_args"], ck["lif_hparams"]
    ann = EEGTCNetLIF(**ma, threshold=hp["threshold"], tau=hp["tau"],
                      timesteps=hp["timesteps"], alpha=hp["alpha"])
    ann.load_state_dict(ck["state_dict"], strict=True)
    ann.eval()
    c1 = fold_bn_layer(ann.conv1, ann.bn1)
    dw = fold_bn_layer(ann.depthwise, ann.bn2)
    sdw = clone_conv(ann.sep_depthwise)
    pw = fold_bn_layer(ann.pointwise, ann.bn3)
    t0a = fold_bn_conv1d(ann.tcn[0].conv_a, ann.tcn[0].bn_a)
    t0b = fold_bn_conv1d(ann.tcn[0].conv_b, ann.tcn[0].bn_b)
    t0s = clone_conv1d(ann.tcn[0].skip)
    t1a = fold_bn_conv1d(ann.tcn[1].conv_a, ann.tcn[1].bn_a)
    t1b = fold_bn_conv1d(ann.tcn[1].conv_b, ann.tcn[1].bn_b)
    t1s = clone_conv1d(ann.tcn[1].skip)
    # decay/thr 直接从 C 头文件解析，保证与 C 编译产物完全同源
    hdr_text = HDR.read_text(encoding="utf-8")
    decay = np.float32(float(re.search(r"#define SNN_DECAY ([\d.eE+-]+)f", hdr_text).group(1)))
    thr = np.float32(float(re.search(r"#define SNN_THR1 ([\d.eE+-]+)f", hdr_text).group(1)))
    # 断言与 float64 exp 计算一致（防止导出脚本与验证脚本口径漂移）
    assert decay == np.float32(np.exp(-1.0 / hp["tau"])), "decay 口径不一致"
    assert thr == np.float32(hp["threshold"]), "thr 口径不一致"
    n_classes = int(ma["n_classes"])
    return {
        "conv1_w": c1.weight.data.numpy().reshape(8, 63).astype(np.float32),
        "conv1_b": c1.bias.data.numpy().astype(np.float32),
        "dw_w": dw.weight.data.numpy().reshape(16, 22).astype(np.float32),
        "dw_b": dw.bias.data.numpy().astype(np.float32),
        "sdw_w": sdw.weight.data.numpy().reshape(16, 16).astype(np.float32),
        "pw_w": pw.weight.data.numpy().reshape(16, 16).astype(np.float32),
        "pw_b": pw.bias.data.numpy().astype(np.float32),
        "t0a_w": t0a.weight.data.numpy().reshape(12, 16, 12).astype(np.float32),
        "t0a_b": t0a.bias.data.numpy().astype(np.float32),
        "t0b_w": t0b.weight.data.numpy().reshape(12, 12, 12).astype(np.float32),
        "t0b_b": t0b.bias.data.numpy().astype(np.float32),
        "t0s_w": t0s.weight.data.numpy().reshape(12, 16).astype(np.float32),
        "t0s_b": t0s.bias.data.numpy().astype(np.float32),
        "t1a_w": t1a.weight.data.numpy().reshape(12, 12, 12).astype(np.float32),
        "t1a_b": t1a.bias.data.numpy().astype(np.float32),
        "t1b_w": t1b.weight.data.numpy().reshape(12, 12, 12).astype(np.float32),
        "t1b_b": t1b.bias.data.numpy().astype(np.float32),
        "t1s_w": t1s.weight.data.numpy().reshape(12, 12).astype(np.float32),
        "fc_w": ann.fc.weight.data.numpy().reshape(n_classes, 12).astype(np.float32),
        "fc_b": ann.fc.bias.data.numpy().astype(np.float32),
        "decay": decay, "thr": thr,
        "timesteps": int(hp["timesteps"]), "n_classes": n_classes,
    }


def c_order_forward(x: np.ndarray, w: dict) -> np.ndarray:
    """x: (22,500) float32。完全复刻 lif_surrogate_infer_tcnet.c 的累加顺序（float32 标量循环）。"""
    f32 = np.float32
    NCH, T = 22, 500
    F1, D, F2, FT = 8, 2, 16, 12
    KE, SEP_K, KT = 63, 16, 12
    T4, T15 = 125, 15
    conv1_w, conv1_b = w["conv1_w"], w["conv1_b"]
    dw_w, dw_b = w["dw_w"], w["dw_b"]
    sdw_w = w["sdw_w"]; pw_w, pw_b = w["pw_w"], w["pw_b"]
    t0a_w, t0a_b = w["t0a_w"], w["t0a_b"]
    t0b_w, t0b_b = w["t0b_w"], w["t0b_b"]
    t0s_w, t0s_b = w["t0s_w"], w["t0s_b"]
    t1a_w, t1a_b = w["t1a_w"], w["t1a_b"]
    t1b_w, t1b_b = w["t1b_w"], w["t1b_b"]
    t1s_w = w["t1s_w"]
    fc_w, fc_b = w["fc_w"], w["fc_b"]
    decay, thr = f32(w["decay"]), f32(w["thr"])
    TSTEPS = w["timesteps"]

    def lif_rate(I):
        mem = f32(0.0); n = 0
        for _ in range(TSTEPS):
            mem = f32(f32(decay * mem) + f32(I))
            if mem > thr:
                n += 1
                mem = f32(0.0)
        return f32(n) / f32(TSTEPS)

    # ---- Block 1: conv1（C 顺序，对称补零 31）----
    c1 = np.zeros((F1, NCH, T), dtype=np.float32)
    for f in range(F1):
        for c in range(NCH):
            for t in range(T):
                s = f32(conv1_b[f])
                for k in range(KE):
                    ii = t + k - KE // 2
                    if 0 <= ii < T:
                        s = f32(s + f32(f32(conv1_w[f, k]) * f32(x[c, ii])))
                c1[f, c, t] = s

    # ---- depthwise（22×1 核，groups=F1）----
    a1 = np.zeros((F2, T), dtype=np.float32)
    for o in range(F2):
        g = o // D
        for t in range(T):
            s = f32(dw_b[o])
            for c in range(NCH):
                s = f32(s + f32(f32(dw_w[o, c]) * f32(c1[g, c, t])))
            a1[o, t] = s

    # ---- LIF(act1) + pool1(4) ----（LIF 在 pool 前，发放率再池化）
    p1 = np.zeros((F2, T4), dtype=np.float32)
    for o in range(F2):
        for t in range(T4):
            acc = f32(0.0)
            for k in range(4):
                acc = f32(acc + lif_rate(a1[o, t * 4 + k]))
            p1[o, t] = f32(acc / f32(4.0))

    # ---- Block 2: sep_depthwise（对称补零 8）----
    sdw = np.zeros((F2, T4), dtype=np.float32)
    for c in range(F2):
        for t in range(T4):
            s = f32(0.0)
            for k in range(SEP_K):
                ii = t + k - SEP_K // 2
                if 0 <= ii < T4:
                    s = f32(s + f32(f32(sdw_w[c, k]) * f32(p1[c, ii])))
            sdw[c, t] = s

    # ---- pointwise ----
    a2 = np.zeros((F2, T4), dtype=np.float32)
    for o in range(F2):
        for t in range(T4):
            s = f32(pw_b[o])
            for c in range(F2):
                s = f32(s + f32(f32(pw_w[o, c]) * f32(sdw[c, t])))
            a2[o, t] = s

    # ---- LIF(act2) + pool2(8) → p2[F2][T15]（TCN 输入）----
    p2 = np.zeros((F2, T15), dtype=np.float32)
    for o in range(F2):
        for t in range(T15):
            acc = f32(0.0)
            for k in range(8):
                acc = f32(acc + lif_rate(a2[o, t * 8 + k]))
            p2[o, t] = f32(acc / f32(8.0))

    # ---- TCN 块 0（dilation=1，因果左补零 pad=(KT-1)*1=11）----
    pad0 = (KT - 1) * 1
    t0_ca = np.zeros((FT, T15), dtype=np.float32)
    for o in range(FT):
        for t in range(T15):
            s = f32(t0a_b[o])
            for c in range(F2):
                for j in range(KT):
                    ii = t + j - pad0
                    if ii >= 0:
                        s = f32(s + f32(f32(t0a_w[o, c, j]) * f32(p2[c, ii])))
            t0_ca[o, t] = s
    t0_ra = np.array([[lif_rate(t0_ca[o, t]) for t in range(T15)] for o in range(FT)],
                     dtype=np.float32)
    t0_cb = np.zeros((FT, T15), dtype=np.float32)
    for o in range(FT):
        for t in range(T15):
            s = f32(t0b_b[o])
            for c in range(FT):
                for j in range(KT):
                    ii = t + j - pad0
                    if ii >= 0:
                        s = f32(s + f32(f32(t0b_w[o, c, j]) * f32(t0_ra[c, ii])))
            t0_cb[o, t] = s
    t0_rb = np.array([[lif_rate(t0_cb[o, t]) for t in range(T15)] for o in range(FT)],
                     dtype=np.float32)
    t0_sk = np.zeros((FT, T15), dtype=np.float32)
    for o in range(FT):
        for t in range(T15):
            s = f32(t0s_b[o])
            for c in range(F2):
                s = f32(s + f32(f32(t0s_w[o, c]) * f32(p2[c, t])))
            t0_sk[o, t] = s
    t0_out = np.array([[lif_rate(f32(f32(t0_sk[o, t]) + f32(t0_rb[o, t])))
                       for t in range(T15)] for o in range(FT)], dtype=np.float32)

    # ---- TCN 块 1（dilation=2，因果左补零 pad=(KT-1)*2=22）----
    pad1 = (KT - 1) * 2
    t1_ca = np.zeros((FT, T15), dtype=np.float32)
    for o in range(FT):
        for t in range(T15):
            s = f32(t1a_b[o])
            for c in range(FT):
                for j in range(KT):
                    ii = t + 2 * j - pad1
                    if ii >= 0:
                        s = f32(s + f32(f32(t1a_w[o, c, j]) * f32(t0_out[c, ii])))
            t1_ca[o, t] = s
    t1_ra = np.array([[lif_rate(t1_ca[o, t]) for t in range(T15)] for o in range(FT)],
                     dtype=np.float32)
    t1_cb = np.zeros((FT, T15), dtype=np.float32)
    for o in range(FT):
        for t in range(T15):
            s = f32(t1b_b[o])
            for c in range(FT):
                for j in range(KT):
                    ii = t + 2 * j - pad1
                    if ii >= 0:
                        s = f32(s + f32(f32(t1b_w[o, c, j]) * f32(t1_ra[c, ii])))
            t1_cb[o, t] = s
    t1_rb = np.array([[lif_rate(t1_cb[o, t]) for t in range(T15)] for o in range(FT)],
                     dtype=np.float32)
    t1_sk = np.zeros((FT, T15), dtype=np.float32)
    for o in range(FT):
        for t in range(T15):
            s = f32(0.0)
            for c in range(FT):
                s = f32(s + f32(f32(t1s_w[o, c]) * f32(t0_out[c, t])))
            t1_sk[o, t] = s
    t1_out = np.array([[lif_rate(f32(f32(t1_sk[o, t]) + f32(t1_rb[o, t])))
                       for t in range(T15)] for o in range(FT)], dtype=np.float32)

    # ---- GAP：15 时间步均值 → Linear(12→4) ----
    g = np.zeros(FT, dtype=np.float32)
    for o in range(FT):
        acc = f32(0.0)
        for t in range(T15):
            acc = f32(acc + t1_out[o, t])
        g[o] = f32(acc / f32(T15))
    n_classes = w["n_classes"]
    logits = np.zeros(n_classes, dtype=np.float32)
    for o in range(n_classes):
        s = f32(fc_b[o])
        for c in range(FT):
            s = f32(s + f32(f32(fc_w[o, c]) * f32(g[c])))
        logits[o] = s
    return logits


def main() -> int:
    global CKPT
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--checkpoint", type=str, default=str(CKPT))
    args = parser.parse_args()
    CKPT = Path(args.checkpoint)
    N = args.n

    w = load_folded_np()
    n_classes = w["n_classes"]
    print(f"[权重] decay={w['decay']:.9f}, thr={w['thr']}, timesteps={w['timesteps']}, "
          f"n_classes={n_classes}（解析自 C 头文件，与 C 编译产物同源）")

    d = np.load(DATA)
    X = d["X"][:N].astype(np.float32)   # (N,1,22,500)
    print(f"[参考] 逐样本 numpy(float32, C 顺序) 前向（含 TCN 膨胀因果卷积/残差/GAP）...")
    logits_np = np.stack([c_order_forward(X[i, 0], w) for i in range(N)])

    # ---- C 推理（宿主机 cc，-ffp-contract=off 禁用 FMA 收缩保证两次舍入）----
    src = BOARD_DEMO_DIR / "lif_surrogate_infer_tcnet.c"
    hdr = BOARD_DEMO_DIR / "lif_tcnet_snn_weights.h"
    with tempfile.TemporaryDirectory() as tmpd:
        tmp = Path(tmpd)
        (tmp / "input.bin").write_bytes(X.tobytes())
        (tmp / "lif_tcnet_snn_weights.h").write_text(hdr.read_text(encoding="utf-8"))
        test_c = tmp / "test.c"
        test_c.write_text(
            '#include <stdio.h>\n'
            'extern int lif_surrogate_infer_tcnet(const float *x, float *logits);\n'
            'int main(int argc, char **argv) {\n'
            '    FILE *f = fopen(argv[1], "rb");\n'
            '    if (!f) return 2;\n'
            f'    for (int n = 0; n < {N}; n++) {{\n'
            '        float x[22*500]; float logits[' + str(n_classes) + '];\n'
            '        fread(x, sizeof(float), 22*500, f);\n'
            '        lif_surrogate_infer_tcnet(x, logits);\n'
            '        for (int o = 0; o < ' + str(n_classes) + '; o++) '
            'printf("%.9e%s", logits[o], o+1<' + str(n_classes) + ' ? " " : "");\n'
            '        printf("\\n");\n'
            '    }\n'
            '    fclose(f); return 0;\n'
            '}\n'
        )
        bin_path = tmp / "be_test"
        r = subprocess.run(["cc", "-O2", "-ffp-contract=off", str(src), str(test_c),
                            "-I", str(tmp), "-o", str(bin_path), "-lm"],
                           capture_output=True, text=True)
        if r.returncode != 0:
            print("✗ 编译失败："); print(r.stderr); return 1
        out = subprocess.run([str(bin_path), str(tmp / "input.bin")],
                             capture_output=True, text=True).stdout
        logits_c = np.array([[float(v) for v in line.split()]
                             for line in out.strip().split("\n") if line.strip()],
                            dtype=np.float32)

    max_diff = float(np.abs(logits_np.astype(np.float32) - logits_c).max())
    n_exact = int((logits_np.astype(np.float32) == logits_c).all(axis=1).sum())
    print(f"[对比] numpy(C 顺序) vs C  max|diff| = {max_diff:.3e}")
    print(f"[对比] 逐位完全一致的样本数   : {n_exact}/{N}")
    if max_diff == 0.0:
        print("✓ 严格逐位一致：C 移植（膨胀因果卷积/残差/GAP/FC）与 float32 C-顺序参考完全一致（0 误差）")
        return 0
    if max_diff < 1e-7:
        print("✓ 近似逐位一致（仅 float32 最后 1 ulp 量级差异）")
        return 0
    print("✗ 存在 >1e-7 差异，需排查 C 移植（重点检查 TCN 膨胀卷积索引与残差 add）")
    return 1


if __name__ == "__main__":
    sys.exit(main())
