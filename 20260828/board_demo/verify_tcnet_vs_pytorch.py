#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_tcnet_vs_pytorch.py — 验证 22ch 四分类 C 版 surrogate LIF-EEG-TCNet 与 PyTorch 一致

方法：取 data_4class/eval.npz 前 N 个真实脑电样本（22ch 四分类），分别跑
      (a) PyTorch 折叠参考（BN 折叠 + LIFActivation 发放率语义，CPU float32，结构与
          eeg_tcnet_lif.forward 逐层一致：TCN 膨胀因果卷积/残差/GAP/Linear）
      (b) 宿主机 cc 编译的 C 版 lif_surrogate_infer_tcnet.c（同一份折叠权重头文件）
      比较 logits 与预测类别。C 移植的正确性门槛：预测一致率 100%（logits 会因浮点
      非结合性有 ~1e-2 量级差异，见 LIF-EEG-TCNet板载实测说明.md §验证，非 bug）。

模式：
  --mode host   用宿主机 cc 编译 + 前 N 样本对比（快速自检）+ 可选 --full 全量
  --mode board  zig 交叉编译 → adb push → 板上前 N 样本对比（真实闭环，板子在线时用）

用法：
  env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin \
      /opt/miniconda3/envs/openmmlab/bin/python verify_tcnet_vs_pytorch.py --n 200 --mode host
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ALGO_DIR = Path(__file__).resolve().parent.parent / "algorithm"
BOARD_DEMO_DIR = Path(__file__).resolve().parent
DATA_4CLASS = ALGO_DIR / "data_4class"
ZIG = Path.home() / "luckfox" / "zig-0.13.0" / "zig"
ADB = Path.home() / "luckfox" / "platform-tools" / "adb"
BOARD_SERIAL = "5caaa04f9cdb6ba2"
CKPT = ALGO_DIR / "eeg_tcnet_4class_lif_t1.0_tau2.0_T8.pth"
FULL_BIN = BOARD_DEMO_DIR / "lif_4class_test_full.bin"

sys.path.insert(0, str(ALGO_DIR))
from eeg_tcnet_lif import EEGTCNetLIF                       # noqa: E402
from lif_eegnet import LIFActivation                        # noqa: E402
from ann2snn import fold_bn_layer, clone_conv               # noqa: E402
from export_tcnet_snn_weights import fold_bn_conv1d, clone_conv1d  # noqa: E402


# ---------------------------------------------------------------------------
# 折叠参考模型：BN 折叠进卷积 + LIFActivation（发放率），结构与 eeg_tcnet_lif.forward 一致
# ---------------------------------------------------------------------------
class TCNBlockRef(nn.Module):
    """TCN 残差块参考：折叠 conv_a/conv_b + LIF + 1×1 skip + 残差 add + act_res。"""

    def __init__(self, blk, lif_kwargs):
        super().__init__()
        self.conv_a = fold_bn_conv1d(blk.conv_a, blk.bn_a)
        self.act_a = LIFActivation(**lif_kwargs)
        self.conv_b = fold_bn_conv1d(blk.conv_b, blk.bn_b)
        self.act_b = LIFActivation(**lif_kwargs)
        self.skip = clone_conv1d(blk.skip)
        self.act_res = LIFActivation(**lif_kwargs)
        self.pad_left = blk.pad_left          # (KT-1)*dilation，因果左补零

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.pad(x, (self.pad_left, 0))
        out = self.conv_a(out)
        out = self.act_a(out)
        out = F.pad(out, (self.pad_left, 0))
        out = self.conv_b(out)
        out = self.act_b(out)
        return self.act_res(self.skip(x) + out)


class FoldedTCNetRef(nn.Module):
    """与 C 头文件完全同源的 PyTorch 参考（BN 全部折叠 + 发放率 LIF）。"""

    def __init__(self, ann: EEGTCNetLIF, lif_kwargs: dict):
        super().__init__()
        self.conv1 = fold_bn_layer(ann.conv1, ann.bn1)
        self.pad1_amt = ann.pad1_amt
        self.depthwise = fold_bn_layer(ann.depthwise, ann.bn2)
        self.act1 = LIFActivation(**lif_kwargs)
        self.pool1 = ann.pool1
        self.sep_depthwise = clone_conv(ann.sep_depthwise)
        self.pad_sep = ann.pad_sep
        self.pointwise = fold_bn_layer(ann.pointwise, ann.bn3)
        self.act2 = LIFActivation(**lif_kwargs)
        self.pool2 = ann.pool2
        self.F2 = ann.F2
        self.tcn = nn.ModuleList([TCNBlockRef(b, lif_kwargs) for b in ann.tcn])
        self.gap_1d = nn.AdaptiveAvgPool1d(1)
        self.fc = ann.fc                       # Linear(FT, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B,1,22,500) → (B,4) logits（与 eeg_tcnet_lif.forward 逐层一致）。"""
        B, C, H, T = x.shape
        pad1 = x.new_zeros(B, C, H, self.pad1_amt)
        x = torch.cat([pad1, x, pad1], dim=3)
        x = self.conv1(x)
        x = self.depthwise(x)
        x = self.act1(x)
        x = self.pool1(x)
        Bc, Cc, Hc, Tc = x.shape
        pad_s = x.new_zeros(Bc, Cc, Hc, self.pad_sep)
        x = torch.cat([pad_s, x, pad_s], dim=3)
        x = self.sep_depthwise(x)
        x = self.pointwise(x)
        x = self.act2(x)
        x = self.pool2(x)
        x = x.view(x.size(0), self.F2, -1)
        for blk in self.tcn:
            x = blk(x)
        x = self.gap_1d(x)
        return self.fc(x.view(x.size(0), -1))


def _load_refs():
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    ma = ckpt["model_args"]
    hp = ckpt["lif_hparams"]
    lif_kwargs = dict(threshold=float(hp["threshold"]), tau=float(hp["tau"]),
                      timesteps=int(hp["timesteps"]), alpha=float(hp["alpha"]))
    ann = EEGTCNetLIF(**ma, **lif_kwargs)
    ann.load_state_dict(ckpt["state_dict"], strict=True)
    ann.eval()
    folded = FoldedTCNetRef(ann, lif_kwargs)
    folded.eval()
    return folded, lif_kwargs


def _compile_c(tmp: Path, mode: str) -> Path:
    """编译 C 推理（host: cc -ffp-contract=off / board: zig 静态交叉编译），返回二进制路径。"""
    src = BOARD_DEMO_DIR / "lif_surrogate_infer_tcnet.c"
    hdr = BOARD_DEMO_DIR / "lif_tcnet_snn_weights.h"
    (tmp / "lif_tcnet_snn_weights.h").write_text(hdr.read_text(encoding="utf-8"))
    test_c = tmp / "test.c"
    test_c.write_text(
        '#include <stdio.h>\n'
        '#include <stdlib.h>\n'
        'extern int lif_surrogate_infer_tcnet(const float *x, float *logits);\n'
        'int main(int argc, char **argv) {\n'
        '    FILE *f = fopen(argv[1], "rb");\n'
        '    if (!f) return 2;\n'
        '    int n = atoi(argv[2]);\n'
        '    float x[22*500]; float logits[4];\n'
        '    for (int k = 0; k < n; k++) {\n'
        '        fread(x, sizeof(float), 22*500, f);\n'
        '        int p = lif_surrogate_infer_tcnet(x, logits);\n'
        '        printf("%d", p);\n'
        '        for (int o = 0; o < 4; o++) printf(" %.9e", logits[o]);\n'
        '        printf("\\n");\n'
        '    }\n'
        '    fclose(f); return 0;\n'
        '}\n'
    )
    if mode == "host":
        bin_path = tmp / "tcnet_host"
        r = subprocess.run(["cc", "-O2", "-ffp-contract=off", str(src), str(test_c),
                            "-I", str(tmp), "-o", str(bin_path), "-lm"],
                           capture_output=True, text=True)
    else:
        bin_path = tmp / "tcnet_board"
        r = subprocess.run([str(ZIG), "cc", "-target", "arm-linux-musleabihf",
                            "-O2", "-static", str(src), str(test_c),
                            "-I", str(tmp), "-o", str(bin_path), "-lm"],
                           capture_output=True, text=True)
    if r.returncode != 0:
        print("✗ 编译失败："); print(r.stderr); raise SystemExit(1)
    return bin_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--mode", type=str, default="host", choices=["host", "board"])
    parser.add_argument("--full", action="store_true", help="宿主机全量 2592 试次精度（用 C 主程序跑完整 bin）")
    args = parser.parse_args()
    N = args.n

    folded, lif_kwargs = _load_refs()
    n_classes = int(folded.fc.out_features)
    print(f"[PyTorch] n_classes={n_classes}, lif_kwargs={lif_kwargs}")

    d = np.load(DATA_4CLASS / "eval.npz")
    Xs = d["X"][:N].astype(np.float32)      # (N,1,22,500)
    ys = d["y"][:N].astype(np.int64)
    with torch.no_grad():
        logits_pt = folded(torch.from_numpy(Xs)).numpy()
    pred_pt = logits_pt.argmax(1)

    with tempfile.TemporaryDirectory() as tmpd:
        tmp = Path(tmpd)
        (tmp / "input.bin").write_bytes(Xs.tobytes())
        bin_path = _compile_c(tmp, args.mode)

        if args.mode == "board":
            # 板载：adb push 二进制与输入，板子跑完拉回输出
            r = subprocess.run([str(ADB), "-s", BOARD_SERIAL, "push", str(bin_path), "/root/tcnet_verify"],
                               capture_output=True, text=True)
            if r.returncode != 0:
                print("✗ adb push 失败：", r.stderr); return 1
            r = subprocess.run([str(ADB), "-s", BOARD_SERIAL, "push", str(tmp / "input.bin"), "/root/tcnet_input.bin"],
                               capture_output=True, text=True)
            if r.returncode != 0:
                print("✗ adb push 失败：", r.stderr); return 1
            subprocess.run([str(ADB), "-s", BOARD_SERIAL, "shell", "chmod", "+x", "/root/tcnet_verify"],
                           capture_output=True, text=True)
            r = subprocess.run([str(ADB), "-s", BOARD_SERIAL, "shell",
                                f"/root/tcnet_verify /root/tcnet_input.bin {N}"],
                               capture_output=True, text=True, timeout=600)
            if r.returncode != 0:
                print("✗ 板载运行失败：", r.stderr); return 1
            out = r.stdout
        else:
            r = subprocess.run([str(bin_path), str(tmp / "input.bin"), str(N)],
                               capture_output=True, text=True)
            if r.returncode != 0:
                print("✗ C 运行失败：", r.stderr); return 1
            out = r.stdout

    rows = [ln.split() for ln in out.strip().split("\n") if ln.strip()]
    pred_c = np.array([int(rw[0]) for rw in rows], dtype=np.int64)
    logits_c = np.array([[float(v) for v in rw[1:5]] for rw in rows], dtype=np.float32)
    assert len(pred_c) == N, f"C 输出样本数 {len(pred_c)} != {N}"

    agree = int((pred_c == pred_pt).sum())
    max_diff = float(np.abs(logits_c - logits_pt).max())
    print(f"[对比] PyTorch(折叠参考) vs C({args.mode})  预测一致率 : {agree}/{N} = {100.0*agree/N:.2f}%")
    print(f"[对比] logits max|diff| : {max_diff:.3e}")
    if agree == N:
        print("✓ 预测一致率 100%（C 移植正确；logits 差异源于浮点非结合性，不改变 argmax）")
    else:
        print("✗ 存在预测不一致，需排查 C 移植")
        for i in range(N):
            if pred_c[i] != pred_pt[i]:
                print(f"  样本 {i}: C pred={pred_c[i]} vs PyTorch pred={pred_pt[i]} "
                      f"logits C={logits_c[i]} PT={logits_pt[i]}")
                if i > 10:
                    break
        return 1

    if args.mode == "host" and args.full:
        # ---- 宿主机全量 2592 试次：C 主程序直接读完整 bin（含 label）----
        src = BOARD_DEMO_DIR / "lif_surrogate_infer_tcnet.c"
        with tempfile.TemporaryDirectory() as tmpd2:
            tmp2 = Path(tmpd2)
            (tmp2 / "lif_tcnet_snn_weights.h").write_text(
                (BOARD_DEMO_DIR / "lif_tcnet_snn_weights.h").read_text(encoding="utf-8"))
            full_bin = tmp2 / "full.bin"
            full_bin.write_bytes(FULL_BIN.read_bytes())
            full_exe = tmp2 / "tcnet_full"
            r = subprocess.run(["cc", "-O2", "-ffp-contract=off",
                                "-DLIF_TCNET_INFER_MAIN", str(src),
                                "-I", str(tmp2), "-o", str(full_exe), "-lm"],
                               capture_output=True, text=True)
            if r.returncode != 0:
                print("✗ 全量编译失败：", r.stderr); return 1
            r = subprocess.run([str(full_exe), str(full_bin)], capture_output=True, text=True,
                               timeout=1200)
            print("---- 宿主机 C 全量 2592 试次输出 ----")
            print(r.stdout)
        # PyTorch 折叠参考全量 eval 精度（2592 试次，分片避免爆内存）
        d2 = np.load(DATA_4CLASS / "eval.npz")
        Xa, ya = d2["X"], d2["y"]
        preds = []
        with torch.no_grad():
            for i in range(0, len(Xa), 128):
                chunk = torch.from_numpy(Xa[i:i + 128].astype(np.float32))
                preds.append(folded(chunk).argmax(1).numpy())
        pred_all = np.concatenate(preds)
        acc_pt = float((pred_all == ya).mean())
        print(f"[PyTorch] 全量 eval（{len(ya)} 试次）精度 : {acc_pt*100:.4f}% "
              f"（checkpoint final_test=62.42%）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
