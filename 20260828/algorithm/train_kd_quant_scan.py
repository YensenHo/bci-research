#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_kd_quant_scan.py — 量化边界证伪实验：bit 扫描看 K/R 各自的断裂点

方向（Codex 第九轮候选一）：把 K/R 分解从「信噪比」轴平移到「量化比特数」轴，
回答「量化到底吃掉了蒸馏的哪一部分」。已有现象：5bit 量化把 K 从 15.97 压到 0.52，
R 几乎不动（7.47）——「量化吃知识、不吃正则化」。

本脚本做第一步可证伪判据：bit ∈ {0(不量化), 2, 3, 4, 5, 8} × 双 seed，
在 σ=1.0（纯高斯红利暴涨点）上，看 K、R 各自的断裂点：
  - 若「K 先崩、R 后崩」的分离区间存在且宽（≥3bit）→ 候选一成立
  - 若 4bit 以下 K/R 一起塌 → 证伪，退回候选二

协议（对齐 kd_decomposition.py）：teacher(F1=16/D=2/F2=32/FT=24, ELU) 满标签 →
软标签(T=4) → relu student(F1=8/D=2/F2=16/FT=12) 30% 标签，三种训练方式：
hard-only / label-smoothing(ε=0.1) / distilled(α=0.5)。
R = ls − hard, K = dist − ls, 总红利 = dist − hard。

用法：
  env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin \
    /opt/miniconda3/envs/openmmlab/bin/python train_kd_quant_scan.py \
    --bits 0,2,3,4,5,8 --sigma 1.0 --seed 42
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ALGO_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ALGO_DIR))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from distill_4class import (  # noqa: E402
    _build_teacher, _build_student, _train_hard, _train_distill, _pick_device,
)
from hbc_noise_advanced import quantize_dequantize, awgn  # noqa: E402

LABEL_FRAC = 0.3
EPS = 0.1
QRANGE = 4.0
EPOCHS = 100


def kd_decompose(X_tr, y_tr, X_ev, y_ev, n_channels, n_classes, input_length,
                 seed, device, epochs):
    """对给定数据做 K/R 分解，返回 (hard, ls, dist, R_pp, K_pp, total_pp)。"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    # ① teacher（满标签）
    teacher = _build_teacher(n_channels, n_classes, input_length, 16, 32, 24).to(device)
    _train_hard(teacher, X_tr, y_tr, device, epochs, 1e-3, 16, seed, False, X_ev, y_ev)
    teacher.eval()
    soft_all = []
    with torch.no_grad():
        for i in range(0, len(X_tr), 16):
            xb = torch.from_numpy(X_tr[i:i + 16]).to(device)
            soft_all.append(F.softmax(teacher(xb) / 4.0, dim=1).cpu().numpy())
    soft_t = np.concatenate(soft_all)
    # ② 30% 标签子集
    rng = np.random.default_rng(seed)
    n_lab = int(round(len(y_tr) * LABEL_FRAC))
    idx = rng.permutation(len(y_tr))[:n_lab]
    X_sub, y_sub = X_tr[idx], y_tr[idx]
    soft_t_sub = soft_t[idx]
    onehot = np.eye(n_classes, dtype=np.float32)[y_sub]
    soft_ls = (1.0 - EPS) * onehot + EPS * (np.ones(n_classes, dtype=np.float32) / n_classes)

    def _run(mode):
        s = _build_student("relu", None, n_channels, n_classes, input_length).to(device)
        if mode == "hard":
            return _train_hard(s, X_sub, y_sub, device, epochs, 1e-3, 16, seed, False, X_ev, y_ev)
        soft = soft_ls if mode == "ls" else soft_t_sub
        return _train_distill(s, X_sub, y_sub, soft, device, epochs, 1e-3, 16, seed, False,
                              4.0, 0.5, X_ev, y_ev)

    hard = _run("hard")
    ls = _run("ls")
    dist = _run("dist")
    return hard * 100, ls * 100, dist * 100, (ls - hard) * 100, (dist - ls) * 100, (dist - hard) * 100


def main() -> int:
    ap = argparse.ArgumentParser(description="量化边界 bit 扫描（K/R 断裂点）")
    ap.add_argument("--bits", type=str, default="0,2,3,4,5,8", help="逗号分隔的 bit 列表，0=不量化")
    ap.add_argument("--sigma", type=float, default=1.0, help="AWGN σ/H")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data-dir", type=str, default="data_4class_cross")
    ap.add_argument("--n-channels", type=int, default=22)
    ap.add_argument("--n-classes", type=int, default=4)
    ap.add_argument("--input-length", type=int, default=500)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    bits = [int(b) for b in args.bits.split(",")]
    if args.smoke:
        bits = bits[:1]
        args.epochs = 2

    device = _pick_device("auto")
    data_dir = ALGO_DIR / args.data_dir
    tr = np.load(data_dir / "train.npz")
    ev = np.load(data_dir / "eval.npz")
    X_tr0 = tr["X"].astype(np.float32)
    y_tr = tr["y"].astype(np.int64)
    X_ev = ev["X"].astype(np.float32)
    y_ev = ev["y"].astype(np.int64)
    print(f"设备 {device} | {args.data_dir} | train {X_tr0.shape} | σ/H={args.sigma} | bits={bits}")

    results = {}
    for bit in bits:
        t0 = time.time()
        # 量化 + AWGN 注入 train（eval 干净）
        X_tr_t = torch.from_numpy(X_tr0)
        if bit > 0:
            X_tr_t = quantize_dequantize(X_tr_t, bit, QRANGE)
        X_tr_noisy = (X_tr_t + awgn(X_tr_t, args.sigma)).numpy().astype(np.float32)
        hard, ls, dist, R, K, total = kd_decompose(
            X_tr_noisy, y_tr, X_ev, y_ev,
            args.n_channels, args.n_classes, args.input_length, args.seed, device, args.epochs)
        results[str(bit)] = dict(hard=round(hard, 2), ls=round(ls, 2), dist=round(dist, 2),
                                 R_pp=round(R, 2), K_pp=round(K, 2), total_pp=round(total, 2))
        print(f"  bit={bit:>2} | hard={hard:5.2f} ls={ls:5.2f} dist={dist:5.2f} "
              f"R={R:+.2f} K={K:+.2f} 总红利={total:+.2f} ({time.time()-t0:.0f}s)")

    # 找断裂点
    print("\n=== K/R 断裂点 ===")
    ks = [results[str(b)]["K_pp"] for b in bits]
    rs = [results[str(b)]["R_pp"] for b in bits]
    for b, k, r in zip(bits, ks, rs):
        print(f"  bit={b:>2}: K={k:+.2f}  R={r:+.2f}")

    out = dict(config=dict(bits=bits, sigma=args.sigma, seed=args.seed,
                           data=args.data_dir, smoke=args.smoke), results=results)
    out_path = ALGO_DIR / f"kd_quant_scan_s{args.seed}.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\n✓ 已保存 {out_path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
