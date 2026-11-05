#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_bit_exact_4class.py — 22ch 四分类 C 版 vs numpy(float32, C 顺序) 严格逐位一致验证

背景：PyTorch 的 conv2d 用优化内核（累加顺序与 C 朴素循环不同），浮点非结合性会让
      pre-LIF 电流差 ~1e-6，极少数膜电位恰在阈值附近的元素会翻转发放决策。这是浮点
      舍入的固有现象，不是 bug。本脚本用 numpy float32 + 与 C 完全相同的累加顺序实现
      参考前向，对比 C 输出，应得到 max|diff| == 0（完全一致）。若 C 移植有结构性错误
      （如 depthwise (22,1) 核索引、4 类 FC 布局），这里会暴露。

用法：
  env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin \
      /opt/miniconda3/envs/openmmlab/bin/python verify_bit_exact_4class.py --n 50
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ALGO_DIR = Path(__file__).resolve().parent.parent / "algorithm"
BOARD_DEMO_DIR = Path(__file__).resolve().parent
CKPT = ALGO_DIR / "lif_4class_t1.0_tau2_T8.pth"
DATA = ALGO_DIR / "data_4class" / "eval.npz"

sys.path.insert(0, str(ALGO_DIR))


def load_folded_np():
    """用与 export 脚本相同的折叠逻辑得到 numpy float32 权重（22ch 四分类）。"""
    import torch
    from lif_eegnet import LIFEEGNet
    from ann2snn import fold_bn_layer, clone_conv
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    ma, hp = ck["model_args"], ck["lif_hparams"]
    ann = LIFEEGNet(**ma, threshold=hp["threshold"], tau=hp["tau"],
                    timesteps=hp["timesteps"], alpha=hp["alpha"])
    ann.load_state_dict(ck["state_dict"], strict=True)
    ann.eval()
    c1 = fold_bn_layer(ann.conv1, ann.bn1)
    dw = fold_bn_layer(ann.depthwise, ann.bn2)
    sdw = clone_conv(ann.sep_depthwise)
    pw = fold_bn_layer(ann.pointwise, ann.bn3)
    fc = clone_conv(ann.fc)
    decay = float(np.float32(np.exp(-1.0 / hp["tau"])))
    thr = float(np.float32(hp["threshold"]))
    n_classes = int(ma["n_classes"])
    return {
        "conv1_w": c1.weight.data.numpy().astype(np.float32),   # (8,1,1,63)
        "conv1_b": c1.bias.data.numpy().astype(np.float32),      # (8,)
        "dw_w": dw.weight.data.numpy().astype(np.float32),       # (16,1,22,1)
        "dw_b": dw.bias.data.numpy().astype(np.float32),         # (16,)
        "sdw_w": sdw.weight.data.numpy().astype(np.float32),     # (16,1,1,15)
        "pw_w": pw.weight.data.numpy().astype(np.float32),       # (16,16,1,1)
        "pw_b": pw.bias.data.numpy().astype(np.float32),         # (16,)
        "fc_w": fc.weight.data.numpy().reshape(n_classes, -1).astype(np.float32),  # (4,240)
        "fc_b": fc.bias.data.numpy().astype(np.float32),         # (4,)
        "decay": decay, "thr": thr,
        "timesteps": int(hp["timesteps"]), "n_classes": n_classes,
    }


def c_order_forward(x: np.ndarray, w: dict) -> np.ndarray:
    """x: (22,500) float32。完全复刻 lif_surrogate_infer_4class.c 的累加顺序（float32 标量循环）。"""
    f32 = np.float32
    NCH, T = 22, 500
    F1, D, F2 = 8, 2, 16
    conv1_w = w["conv1_w"]; conv1_b = w["conv1_b"]
    dw_w = w["dw_w"]; dw_b = w["dw_b"]
    sdw_w = w["sdw_w"]; pw_w = w["pw_w"]; pw_b = w["pw_b"]
    fc_w = w["fc_w"]; fc_b = w["fc_b"]
    decay = f32(w["decay"]); thr = f32(w["thr"]); TSTEPS = w["timesteps"]

    # ---- Block 1: conv1（C 顺序）----
    c1 = np.zeros((F1, NCH, T), dtype=np.float32)
    for f in range(F1):
        for c in range(NCH):
            for t in range(T):
                s = f32(conv1_b[f])
                for k in range(63):
                    ii = t + k - 31
                    if 0 <= ii < T:
                        s = f32(s + f32(f32(conv1_w[f, 0, 0, k]) * f32(x[c, ii])))
                c1[f, c, t] = s

    # ---- depthwise（22×1 核）----
    a1 = np.zeros((F2, T), dtype=np.float32)
    for o in range(F2):
        g = o // D
        for t in range(T):
            s = f32(dw_b[o])
            for c in range(NCH):
                s = f32(s + f32(f32(dw_w[o, 0, c, 0]) * f32(c1[g, c, t])))
            a1[o, t] = s

    # ---- LIF(thr1) + pool1 ----
    def lif_rate(I):
        mem = f32(0.0); n = 0
        for _ in range(TSTEPS):
            mem = f32(f32(decay * mem) + f32(I))
            if mem > thr:
                n += 1
                mem = f32(0.0)
        return f32(n) / f32(TSTEPS)

    T4 = T // 4
    p1 = np.zeros((F2, T4), dtype=np.float32)
    for o in range(F2):
        for t in range(T4):
            acc = f32(0.0)
            for k in range(4):
                acc = f32(acc + lif_rate(a1[o, t * 4 + k]))
            p1[o, t] = f32(acc / f32(4.0))

    # ---- Block 2: sep_depthwise ----
    sdw_out = np.zeros((F2, T4), dtype=np.float32)
    for c in range(F2):
        for t in range(T4):
            s = f32(0.0)
            for k in range(15):
                ii = t + k - 7
                if 0 <= ii < T4:
                    s = f32(s + f32(f32(sdw_w[c, 0, 0, k]) * f32(p1[c, ii])))
            sdw_out[c, t] = s

    # ---- pointwise ----
    a2 = np.zeros((F2, T4), dtype=np.float32)
    for o in range(F2):
        for t in range(T4):
            s = f32(pw_b[o])
            for c in range(F2):
                s = f32(s + f32(f32(pw_w[o, c, 0, 0]) * f32(sdw_out[c, t])))
            a2[o, t] = s

    # ---- LIF(thr2) + pool2 ----
    T32 = T // 32
    p2 = np.zeros((F2, T32), dtype=np.float32)
    for o in range(F2):
        for t in range(T32):
            acc = f32(0.0)
            for k in range(8):
                acc = f32(acc + lif_rate(a2[o, t * 8 + k]))
            p2[o, t] = f32(acc / f32(8.0))

    # ---- FC ----
    n_classes = w["n_classes"]
    logits = np.zeros(n_classes, dtype=np.float32)
    for o in range(n_classes):
        s = f32(fc_b[o])
        for c in range(F2):
            for t in range(T32):
                s = f32(s + f32(f32(fc_w[o, c * T32 + t]) * f32(p2[c, t])))
        logits[o] = s
    return logits


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50)
    args = parser.parse_args()
    N = args.n

    w = load_folded_np()
    n_classes = w["n_classes"]
    print(f"[权重] decay={w['decay']:.9f}, thr={w['thr']}, timesteps={w['timesteps']}, n_classes={n_classes}")

    d = np.load(DATA)
    X = d["X"][:N].astype(np.float32)   # (N,1,22,500)
    print(f"[参考] 逐样本 numpy(float32, C 顺序) 前向 ...")
    logits_np = np.stack([c_order_forward(X[i, 0], w) for i in range(N)])

    # ---- C 推理 ----
    src = BOARD_DEMO_DIR / "lif_surrogate_infer_4class.c"
    hdr = BOARD_DEMO_DIR / "lif_surrogate_weights_4class.h"
    with tempfile.TemporaryDirectory() as tmpd:
        tmp = Path(tmpd)
        (tmp / "input.bin").write_bytes(X.tobytes())
        (tmp / "lif_surrogate_weights_4class.h").write_text(hdr.read_text(encoding="utf-8"))
        test_c = tmp / "test.c"
        test_c.write_text(
            '#include <stdio.h>\n'
            'extern int lif_surrogate_infer_4class(const float *x, float *logits);\n'
            'int main(int argc, char **argv) {\n'
            '    FILE *f = fopen(argv[1], "rb");\n'
            '    if (!f) return 2;\n'
            f'    for (int n = 0; n < {N}; n++) {{\n'
            '        float x[22*500]; float logits[' + str(n_classes) + '];\n'
            '        fread(x, sizeof(float), 22*500, f);\n'
            '        lif_surrogate_infer_4class(x, logits);\n'
            '        for (int o = 0; o < ' + str(n_classes) + '; o++) '
            'printf("%.9e%s", logits[o], o+1<' + str(n_classes) + ' ? " " : "");\n'
            '        printf("\\n");\n'
            '    }\n'
            '    fclose(f); return 0;\n'
            '}\n'
        )
        bin_path = tmp / "be_test"
        # -ffp-contract=off：禁用 FMA 收缩，保证 C 的 a*b+c 按两次舍入（与 numpy 参考一致）
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
        print("✓ 严格逐位一致：C 移植与 float32 C-顺序参考完全一致（0 误差）")
        return 0
    if max_diff < 1e-7:
        print("✓ 近似逐位一致（仅 float32 最后 1 ulp 量级差异）")
        return 0
    print("✗ 存在 >1e-7 差异，需排查 C 移植")
    return 1


if __name__ == "__main__":
    sys.exit(main())
