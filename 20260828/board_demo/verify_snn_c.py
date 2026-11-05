#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_snn_c.py — 验证 C 版 LIF-EEGNet（SNN）推理与 PyTorch-SNN 输出一致

方法：取 eval.npz 前 N 个真实脑电样本，分别跑 PyTorch-SNN（折叠权重 + thr1/thr2 + timesteps）
      和编译后的 C 版 lif_eegnet_infer.c，比较 logits 与预测类别。C 移植的正确性门槛。

两种模式：
  --mode host   用宿主机 cc 编译（快速自检，验证 C 逻辑正确）
  --mode board  用 zig 交叉编译 arm-linux-musleabihf → adb push → 板上运行 → 对比（真实闭环）

用法
----
  env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin \
      /opt/miniconda3/envs/openmmlab/bin/python verify_snn_c.py --n 50 --mode board
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ALGO_DIR = Path(__file__).resolve().parent.parent / "algorithm"
BOARD_DEMO_DIR = Path(__file__).resolve().parent
PROCESSED = Path("/Users/heyixuan/BCI/luckfox-deploy/data/processed")
ZIG = Path.home() / "luckfox" / "zig-0.13.0" / "zig"
ADB = Path.home() / "luckfox" / "platform-tools" / "adb"
BOARD_SERIAL = "5caaa04f9cdb6ba2"

sys.path.insert(0, str(ALGO_DIR))
from lif_eegnet import LIFEEGNet          # noqa: E402
from ann2snn import build_snn, snn_infer  # noqa: E402


def _load_pytorch_snn(state_path: Path):
    """从 snn_weights.pth 重建 PyTorch-SNN（折叠权重 + thr1/thr2 + timesteps）。"""
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    # 用 relu_proxy 重建结构，再覆盖折叠权重
    ann = LIFEEGNet(n_channels=state.get("n_channels", 8),
                    n_classes=state.get("n_classes", 2),
                    input_length=state.get("input_length", 500),
                    F1=state.get("F1", 8), D=state.get("D", 2), F2=state.get("F2", 16),
                    kernel_length=state.get("kernel_length", 63), activation="relu")
    snn = build_snn(ann)
    sd = snn.state_dict()
    sd["conv1.weight"] = state["conv1_weight"]
    sd["conv1.bias"] = state["conv1_bias"]
    sd["depthwise.weight"] = state["depthwise_weight"]
    sd["depthwise.bias"] = state["depthwise_bias"]
    sd["sep_depthwise.weight"] = state["sep_depthwise_weight"]
    sd["pointwise.weight"] = state["pointwise_weight"]
    sd["pointwise.bias"] = state["pointwise_bias"]
    sd["fc.weight"] = state["fc_weight"]
    sd["fc.bias"] = state["fc_bias"]
    snn.load_state_dict(sd, strict=True)
    snn.thr1 = float(state["thr1"])
    snn.thr2 = float(state["thr2"])
    snn.eval()
    return snn, int(state["n_classes"]), int(state["timesteps"]), float(state["thr1"]), float(state["thr2"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--mode", type=str, default="host", choices=["host", "board"])
    parser.add_argument("--state", type=str, default=str(ALGO_DIR / "snn_weights.pth"))
    args = parser.parse_args()
    N = args.n
    snn, n_classes, timesteps, thr1, thr2 = _load_pytorch_snn(Path(args.state))
    print(f"[PyTorch-SNN] n_classes={n_classes}, timesteps={timesteps}, thr1={thr1:.4f}, thr2={thr2:.4f}")

    # ---- PyTorch-SNN 推理 ----
    d = np.load(PROCESSED / "eval.npz")
    Xs = d["X"][:N].astype(np.float32)      # (N,1,8,500)
    with torch.no_grad():
        logits_py = snn_infer(snn, torch.from_numpy(Xs), timesteps=timesteps).numpy()
    pred_py = logits_py.argmax(1)

    # ---- C 推理 ----
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "input.bin").write_bytes(Xs.tobytes())
        test_c = tmp / "test.c"
        test_c.write_text(
            '#include <stdio.h>\n'
            '#include <stdlib.h>\n'
            'extern int lif_eegnet_infer(const float *x, float *logits);\n'
            'int main(int argc, char **argv) {\n'
            '    FILE *f = fopen(argc > 1 ? argv[1] : "input.bin", "rb");\n'
            '    if (!f) { fprintf(stderr, "cannot open input\\n"); return 2; }\n'
            f'    for (int n = 0; n < {N}; n++) {{\n'
            '        float x[8*500];\n'
            '        fread(x, sizeof(float), 8*500, f);\n'
            f'        float logits[{n_classes}];\n'
            '        lif_eegnet_infer(x, logits);\n'
            '        for (int o = 0; o < ' + str(n_classes) + '; o++) '
            'printf("%.6f%s", logits[o], o + 1 < ' + str(n_classes) + ' ? " " : "");\n'
            '        printf("\\n");\n'
            '    }\n'
            '    fclose(f);\n'
            '    return 0;\n'
            '}\n'
        )
        src = BOARD_DEMO_DIR / "lif_eegnet_infer.c"
        hdr = BOARD_DEMO_DIR / "snn_weights.h"

        if args.mode == "host":
            bin_path = tmp / "snn_test"
            r = subprocess.run(
                ["cc", "-O2", str(src), str(test_c), "-o", str(bin_path), "-lm"],
                capture_output=True, text=True,
            )
            if r.returncode != 0:
                print("✗ 宿主机 C 编译失败：")
                print(r.stderr)
                return 1
            out = subprocess.run([str(bin_path), str(tmp / "input.bin")],
                                 capture_output=True, text=True).stdout
        else:
            # ---- zig 交叉编译 → adb push → 板上运行 ----
            if not ZIG.exists():
                print(f"✗ 找不到 zig：{ZIG}")
                return 1
            bin_name = "lif_snn_test"
            bin_path = tmp / bin_name
            # zig 需要 header 在同一目录或 -I 指定；复制 snn_weights.h 到 tmp
            (tmp / "snn_weights.h").write_text(hdr.read_text(encoding="utf-8"))
            r = subprocess.run(
                [str(ZIG), "cc", "-target", "arm-linux-musleabihf", "-O2", "-static",
                 str(src), str(test_c), "-o", str(bin_path), "-lm"],
                capture_output=True, text=True,
            )
            if r.returncode != 0:
                print("✗ zig 交叉编译失败：")
                print(r.stderr)
                return 1
            # adb push
            adb = [str(ADB), "-s", BOARD_SERIAL]
            if subprocess.run(adb + ["shell", "true"], capture_output=True).returncode != 0:
                print("✗ adb 板子不通（序列 5caaa04f9cdb6ba2），标记「待板子」。")
                print("  （Python 侧与 C 代码已就绪，仅缺板上运行）")
                return 2
            subprocess.run(adb + ["push", str(bin_path), "/root/" + bin_name],
                           capture_output=True, text=True)
            subprocess.run(adb + ["push", str(tmp / "input.bin"), "/root/snn_input.bin"],
                           capture_output=True, text=True)
            subprocess.run(adb + ["shell", "chmod", "+x", "/root/" + bin_name],
                           capture_output=True, text=True)
            out = subprocess.run(
                adb + ["shell", f"/root/{bin_name} /root/snn_input.bin"],
                capture_output=True, text=True,
            ).stdout

        logits_c = np.array([[float(v) for v in line.split()]
                             for line in out.strip().split("\n") if line.strip()])
        pred_c = logits_c.argmax(1)

    # ---- 对比 ----
    if logits_c.shape != logits_py.shape:
        print(f"✗ 输出形状不一致：PyTorch {logits_py.shape} vs C {logits_c.shape}")
        return 1
    max_diff = float(np.abs(logits_py - logits_c).max())
    n_match = int((pred_py == pred_c).sum())
    print(f"logits 最大绝对误差 : {max_diff:.2e}")
    print(f"预测一致率         : {n_match}/{N} ({n_match/N*100:.1f}%)")
    if max_diff < 1e-3 and n_match == N:
        print("✓ C-SNN 与 PyTorch-SNN 逐位一致（LIF 脉冲 + 膜电位 + 阈值 移植正确）")
        return 0
    print("✗ C-SNN 与 PyTorch-SNN 不一致，需排查移植")
    return 1


if __name__ == "__main__":
    sys.exit(main())
