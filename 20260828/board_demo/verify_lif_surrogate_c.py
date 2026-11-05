#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_lif_surrogate_c.py — 验证 C 版 surrogate LIF-EEGNet 与 PyTorch 版逐位一致

方法：取 eval.npz 前 N 个真实脑电样本，分别跑
      (a) PyTorch 折叠参考（BN 折叠 + LIFActivation 发放率语义，CPU float32）
      (b) 编译后的 C 版 lif_surrogate_infer.c（同一份折叠权重头文件）
      比较 logits 与预测类别。C 移植的正确性门槛：max|diff| < 1e-5 且预测一致率 100%。

两种模式：
  --mode host   用宿主机 cc 编译（快速自检，验证 C 逻辑正确）
  --mode board  用 zig 交叉编译 arm-linux-musleabihf → adb push → 板上运行 → 对比（真实闭环）

用法
----
  env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin \
      /opt/miniconda3/envs/openmmlab/bin/python verify_lif_surrogate_c.py --n 100 --mode host
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ALGO_DIR = Path(__file__).resolve().parent.parent / "algorithm"
BOARD_DEMO_DIR = Path(__file__).resolve().parent
PROCESSED = Path("/Users/heyixuan/BCI/luckfox-deploy/data/processed")
ZIG = Path.home() / "luckfox" / "zig-0.13.0" / "zig"
ADB = Path.home() / "luckfox" / "platform-tools" / "adb"
BOARD_SERIAL = "5caaa04f9cdb6ba2"
CKPT = ALGO_DIR / "lif_F_seed123.pth"

sys.path.insert(0, str(ALGO_DIR))
from lif_eegnet import LIFEEGNet, LIFActivation  # noqa: E402
from ann2snn import fold_bn_layer, clone_conv     # noqa: E402


# ---------------------------------------------------------------------------
# 折叠参考模型：BN 折叠进卷积 + LIFActivation（发放率），结构与 LIFEEGNet.forward 一致
# ---------------------------------------------------------------------------
class FoldedLIFRef(nn.Module):
    """与 C 头文件完全同源的 PyTorch 参考（fold_bn_layer 折叠 + 发放率 LIF）。"""

    def __init__(self, ann: LIFEEGNet):
        super().__init__()
        self.conv1 = fold_bn_layer(ann.conv1, ann.bn1)
        self.depthwise = fold_bn_layer(ann.depthwise, ann.bn2)
        self.sep_depthwise = clone_conv(ann.sep_depthwise)
        self.pointwise = fold_bn_layer(ann.pointwise, ann.bn3)
        self.fc = clone_conv(ann.fc)
        # LIF 激活：阈值/时间常数/步数/斜率与训练时一致
        self.act1 = LIFActivation(threshold=ann.act1.threshold, tau=ann.act1.tau,
                                  timesteps=ann.act1.timesteps, alpha=ann.act1.alpha)
        self.act2 = LIFActivation(threshold=ann.act2.threshold, tau=ann.act2.tau,
                                  timesteps=ann.act2.timesteps, alpha=ann.act2.alpha)
        self.pool1 = nn.AvgPool2d((1, 4))
        self.pool2 = nn.AvgPool2d((1, 8))
        self.pad1_amt = ann.pad1_amt
        self.pad2_amt = ann.pad2_amt
        self.F2 = ann.F2
        self.n_classes = ann.n_classes
        self.temporal_out = ann.temporal_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, T = x.shape
        pad1 = x.new_zeros(B, C, H, self.pad1_amt)
        x = torch.cat([pad1, x, pad1], dim=3)
        x = self.conv1(x)
        x = self.depthwise(x)
        x = self.act1(x)                     # LIF 发放率（pool 前）
        x = self.pool1(x)
        B, C, H, T = x.shape
        pad2 = x.new_zeros(B, C, H, self.pad2_amt)
        x = torch.cat([pad2, x, pad2], dim=3)
        x = self.sep_depthwise(x)
        x = self.pointwise(x)
        x = self.act2(x)                     # LIF 发放率（pool 前）
        x = self.pool2(x)
        x = x.flatten(1)
        x = x.view(x.shape[0], self.F2 * self.temporal_out, 1, 1)
        x = self.fc(x)
        return x.view(x.shape[0], self.n_classes)


def _load_refs():
    """加载原模型 + 折叠参考模型。返回 (ann, folded_ref, meta)。"""
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    ma = ckpt["model_args"]
    hp = ckpt["lif_hparams"]
    ann = LIFEEGNet(**ma, threshold=float(hp["threshold"]), tau=float(hp["tau"]),
                    timesteps=int(hp["timesteps"]), alpha=float(hp["alpha"]))
    ann.load_state_dict(ckpt["state_dict"], strict=True)
    ann.eval()
    folded = FoldedLIFRef(ann)
    folded.eval()
    meta = dict(n_classes=int(ma["n_classes"]), timesteps=int(hp["timesteps"]),
                threshold=float(hp["threshold"]), tau=float(hp["tau"]))
    return ann, folded, meta


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--mode", type=str, default="host", choices=["host", "board"])
    args = parser.parse_args()
    N = args.n

    ann, folded, meta = _load_refs()
    n_classes, timesteps = meta["n_classes"], meta["timesteps"]
    print(f"[PyTorch] n_classes={n_classes}, timesteps={timesteps}, "
          f"thr={meta['threshold']}, tau={meta['tau']}")

    d = np.load(PROCESSED / "eval.npz")
    Xs = d["X"][:N].astype(np.float32)      # (N,1,8,500)
    ys = d["y"][:N]

    with torch.no_grad():
        # 折叠参考（与 C 同源）
        logits_py = folded(torch.from_numpy(Xs)).numpy()
        # 原模型（含 BN eval 模式）—— 验证 BN 折叠无损
        logits_ann = ann(torch.from_numpy(Xs)).numpy()
    fold_diff = float(np.abs(logits_py - logits_ann).max())
    print(f"[BN 折叠] 折叠参考 vs 原模型  logits max|diff| = {fold_diff:.2e}")
    pred_py = logits_py.argmax(1)

    # ---- C 推理 ----
    src = BOARD_DEMO_DIR / "lif_surrogate_infer.c"
    hdr = BOARD_DEMO_DIR / "lif_surrogate_weights.h"
    with tempfile.TemporaryDirectory() as tmpd:
        tmp = Path(tmpd)
        (tmp / "input.bin").write_bytes(Xs.tobytes())
        (tmp / "lif_surrogate_weights.h").write_text(hdr.read_text(encoding="utf-8"))
        test_c = tmp / "test.c"
        test_c.write_text(
            '#include <stdio.h>\n'
            '#include <stdlib.h>\n'
            'extern int lif_surrogate_infer(const float *x, float *logits);\n'
            'int main(int argc, char **argv) {\n'
            '    FILE *f = fopen(argc > 1 ? argv[1] : "input.bin", "rb");\n'
            '    if (!f) { fprintf(stderr, "cannot open input\\n"); return 2; }\n'
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

        if args.mode == "host":
            bin_path = tmp / "snn_test"
            r = subprocess.run(
                ["cc", "-O2", str(src), str(test_c), "-I", str(tmp),
                 "-o", str(bin_path), "-lm"],
                capture_output=True, text=True,
            )
            if r.returncode != 0:
                print("✗ 宿主机 C 编译失败：")
                print(r.stderr)
                return 1
            out = subprocess.run([str(bin_path), str(tmp / "input.bin")],
                                 capture_output=True, text=True).stdout
        else:
            if not ZIG.exists():
                print(f"✗ 找不到 zig：{ZIG}")
                return 1
            bin_name = "lif_snn_test"
            bin_path = tmp / bin_name
            r = subprocess.run(
                [str(ZIG), "cc", "-target", "arm-linux-musleabihf", "-O2", "-static",
                 str(src), str(test_c), "-o", str(bin_path), "-lm"],
                capture_output=True, text=True,
            )
            if r.returncode != 0:
                print("✗ zig 交叉编译失败：")
                print(r.stderr)
                return 1
            adb = [str(ADB), "-s", BOARD_SERIAL]
            if subprocess.run(adb + ["shell", "true"], capture_output=True).returncode != 0:
                print("✗ adb 板子不通（序列 5caaa04f9cdb6ba2），标记「待板子」。")
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
    acc_py = float((pred_py == ys).mean())
    acc_c = float((pred_c == ys).mean())
    print(f"[对比] C vs PyTorch logits 最大绝对误差 : {max_diff:.2e}")
    print(f"[对比] 预测一致率          : {n_match}/{N} ({n_match/N*100:.1f}%)")
    print(f"[精度] PyTorch 折叠参考   : {acc_py*100:.2f}% | C : {acc_c*100:.2f}%")
    if max_diff < 1e-5 and n_match == N:
        print("✓ C-SNN 与 PyTorch-SNN 逐位一致（LIF 在 pool 前 + decay=0.6065 + 发放率 移植正确）")
        return 0
    print("✗ C-SNN 与 PyTorch-SNN 不一致，需排查移植")
    # 打印前几个不一致样本的 logits 便于定位
    for i in range(N):
        if not np.isclose(logits_py[i], logits_c[i], atol=1e-5).all():
            print(f"  样本 {i}: PyTorch={logits_py[i].tolist()}  C={logits_c[i].tolist()}")
            if i >= 5:
                break
    return 1


if __name__ == "__main__":
    sys.exit(main())
