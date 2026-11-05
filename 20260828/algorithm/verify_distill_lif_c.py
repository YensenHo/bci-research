#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_distill_lif_c.py — 验证蒸馏×LIF 学生 C 推理与 PyTorch 逐位一致（host）

复用 verify_lif_surrogate_c.py 的 FoldedLIFRef（BN 折叠 + 发放率 LIF 语义）与
lif_surrogate_infer.c（同一份 C 推理），只把 checkpoint 换成蒸馏产物
（distill_lif_30pct_s42.pth）与对应 C 头（lif_surrogate_weights_distill.h）。
验证「蒸馏训出来的类脑模型能否直接走端侧部署链路」——门槛同 verify_lif_surrogate_c.py：
max|diff| < 1e-5 且预测一致率 100%。

用法
----
  env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin \
    /opt/miniconda3/envs/openmmlab/bin/python verify_distill_lif_c.py --n 100
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ALGO_DIR = Path(__file__).resolve().parent
BOARD_DEMO_DIR = ALGO_DIR.parent / "board_demo"
PROCESSED = Path("/Users/heyixuan/BCI/luckfox-deploy/data/processed")
CKPT = ALGO_DIR / "distill_lif_30pct_s42.pth"
HDR = BOARD_DEMO_DIR / "lif_surrogate_weights_distill.h"
SRC = BOARD_DEMO_DIR / "lif_surrogate_infer.c"

sys.path.insert(0, str(ALGO_DIR))
sys.path.insert(0, str(BOARD_DEMO_DIR))
from lif_eegnet import LIFEEGNet  # noqa: E402
from verify_lif_surrogate_c import FoldedLIFRef  # noqa: E402  （复用折叠参考）


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    args = ap.parse_args()
    N = args.n

    # ---- 加载蒸馏 checkpoint → 折叠参考 ----
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    ma = ckpt["model_args"]
    hp = ckpt["lif_hparams"]
    ann = LIFEEGNet(**ma, threshold=float(hp["threshold"]), tau=float(hp["tau"]),
                    timesteps=int(hp["timesteps"]), alpha=float(hp["alpha"]))
    ann.load_state_dict(ckpt["state_dict"], strict=True)
    ann.eval()
    folded = FoldedLIFRef(ann)
    folded.eval()
    n_classes = int(ma["n_classes"])
    print(f"[PyTorch] 蒸馏×LIF 学生 n_classes={n_classes} "
          f"thr={hp['threshold']} tau={hp['tau']} T={hp['timesteps']}")

    d = np.load(PROCESSED / "eval.npz")
    Xs = d["X"][:N].astype(np.float32)
    ys = d["y"][:N]

    with torch.no_grad():
        logits_py = folded(torch.from_numpy(Xs)).numpy()
        logits_ann = ann(torch.from_numpy(Xs)).numpy()
    fold_diff = float(np.abs(logits_py - logits_ann).max())
    print(f"[BN 折叠] 折叠参考 vs 原模型  logits max|diff| = {fold_diff:.2e}")
    pred_py = logits_py.argmax(1)

    # ---- C 推理（host cc 编译）----
    with tempfile.TemporaryDirectory() as tmpd:
        tmp = Path(tmpd)
        (tmp / "input.bin").write_bytes(Xs.tobytes())
        (tmp / "lif_surrogate_weights.h").write_text(HDR.read_text(encoding="utf-8"))
        (tmp / "lif_surrogate_infer.c").write_text(SRC.read_text(encoding="utf-8"))
        test_c = tmp / "test.c"
        test_c.write_text(
            '#include <stdio.h>\n'
            '#include <stdlib.h>\n'
            'extern int lif_surrogate_infer(const float *x, float *logits);\n'
            'int main(int argc, char **argv) {\n'
            '    FILE *f = fopen(argc > 1 ? argv[1] : "input.bin", "rb");\n'
            '    if (!f) return 2;\n'
            f'    for (int n = 0; n < {N}; n++) {{\n'
            '        float x[8*500];\n'
            '        fread(x, sizeof(float), 8*500, f);\n'
            f'        float logits[{n_classes}];\n'
            '        lif_surrogate_infer(x, logits);\n'
            '        for (int o = 0; o < ' + str(n_classes) + '; o++) '
            'printf("%.6f%s", logits[o], o + 1 < ' + str(n_classes) + ' ? " " : "");\n'
            '        printf("\\n");\n'
            '    }\n'
            '    fclose(f);\n'
            '    return 0;\n'
            '}\n'
        )
        bin_path = tmp / "snn_test"
        r = subprocess.run(["cc", "-O2", str(tmp / "lif_surrogate_infer.c"), str(test_c),
                            "-I", str(tmp), "-o", str(bin_path), "-lm"],
                           capture_output=True, text=True)
        if r.returncode != 0:
            print("✗ C 编译失败：")
            print(r.stderr)
            return 1
        out = subprocess.run([str(bin_path), str(tmp / "input.bin")],
                             capture_output=True, text=True).stdout

    logits_c = np.array([[float(v) for v in line.split()]
                         for line in out.strip().split("\n") if line.strip()])
    pred_c = logits_c.argmax(1)

    if logits_c.shape != logits_py.shape:
        print(f"✗ 输出形状不一致：PyTorch {logits_py.shape} vs C {logits_c.shape}")
        return 1
    max_diff = float(np.abs(logits_py - logits_c).max())
    n_match = int((pred_py == pred_c).sum())
    acc_py = float((pred_py == ys).mean())
    acc_c = float((pred_c == ys).mean())
    print(f"[对比] C vs PyTorch logits 最大绝对误差 : {max_diff:.2e}")
    print(f"[对比] 预测一致率          : {n_match}/{N} ({n_match/N*100:.1f}%)")
    print(f"[精度] PyTorch 折叠参考   : {acc_py*100:.2f}% | C : {acc_c*100:.2f}%")
    if max_diff < 1e-5 and n_match == N:
        print("✓ 蒸馏×LIF 学生 C 推理与 PyTorch 逐位一致（端侧部署链路可用）")
        return 0
    print("✗ C 与 PyTorch 不一致")
    return 1


if __name__ == "__main__":
    sys.exit(main())
