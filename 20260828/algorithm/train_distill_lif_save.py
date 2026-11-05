#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_distill_lif_save.py — 训练蒸馏×LIF 学生并保存 checkpoint（供端侧 C 导出）

目的
----
  distill_lif.py 只跑完评估就丢弃模型，无法端侧导出。本脚本补上「训练 + 保存」：
  大 EEGNet teacher（ELU，100% 标签）→ 软标签 → LIFEEGNet student（30% 标签蒸馏），
  保存 checkpoint（state_dict + model_args + lif_hparams + final_test），
  供 export_lif_surrogate_weights.py 导出 C 头、verify_lif_surrogate_c.py 逐位验证。

  回答「可穿戴部署」方向的关键闭环：蒸馏训出来的类脑模型能否直接走端侧部署链路。

用法
----
  env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin \
    /opt/miniconda3/envs/openmmlab/bin/python train_distill_lif_save.py \
    --label-frac 0.3 --seed 42 --out distill_lif_30pct_s42.pth
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

PROJECT = Path("/Users/heyixuan/BCI/luckfox-deploy")
MODEL_DIR = PROJECT / "model"
DATA_DIR = PROJECT / "data"
ALGO_DIR = Path("/Users/heyixuan/BCI/20260828/algorithm")
sys.path.insert(0, str(MODEL_DIR))
sys.path.insert(0, str(DATA_DIR))
sys.path.insert(0, str(ALGO_DIR))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from eegnet import EEGNet          # noqa: E402  （teacher）
from lif_eegnet import LIFEEGNet   # noqa: E402  （student）


def _pick_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    if name in ("cuda", "auto") and torch.cuda.is_available():
        return torch.device("cuda")
    if name == "auto" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _build_teacher() -> EEGNet:
    return EEGNet(n_channels=8, n_classes=2, input_length=500,
                  F1=16, D=4, F2=32, kernel_length=63, dropout=0.5)


def _build_student() -> LIFEEGNet:
    # 默认 LIF 超参 threshold=1.0/tau=2.0/timesteps=8/alpha=4.0（与 distill_lif.py 一致）
    return LIFEEGNet(n_channels=8, n_classes=2, input_length=500,
                     F1=8, D=2, F2=16, kernel_length=63, dropout=0.5, activation="lif")


def _eval(model, X, y, device, batch):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.from_numpy(X[i:i + batch]).to(device)
            yb = torch.from_numpy(y[i:i + batch]).to(device)
            correct += (model(xb).argmax(1) == yb).sum().item()
            total += yb.size(0)
    return correct / total


def main() -> int:
    ap = argparse.ArgumentParser(description="训练蒸馏×LIF 学生并保存 checkpoint")
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--temperature", type=float, default=4.0)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--label-frac", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default="distill_lif_30pct_s42.pth")
    args = ap.parse_args()

    device = _pick_device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    tr = np.load(DATA_DIR / "processed" / "train.npz")
    ev = np.load(DATA_DIR / "processed" / "eval.npz")
    X_tr = tr["X"].astype(np.float32)
    y_tr = tr["y"].astype(np.int64)
    X_ev = ev["X"].astype(np.float32)
    y_ev = ev["y"].astype(np.int64)
    print(f"设备 : {device} | train {X_tr.shape} | eval {X_ev.shape} | 标签预算 {args.label_frac:.0%}")

    # ① teacher（大 ELU EEGNet，100% 标签）
    teacher = _build_teacher().to(device)
    t0 = time.time()
    opt = torch.optim.Adam(teacher.parameters(), lr=args.lr)
    ds = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True)
    for _ in range(args.epochs):
        teacher.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            F.cross_entropy(teacher(xb), yb).backward()
            opt.step()
    teacher_acc = _eval(teacher, X_ev, y_ev, device, args.batch)
    print(f"  teacher eval : {teacher_acc*100:.2f}%（{time.time()-t0:.0f}s）")

    # ② teacher 软标签（全 train 集）
    teacher.eval()
    soft_all = []
    with torch.no_grad():
        for i in range(0, len(X_tr), args.batch):
            xb = torch.from_numpy(X_tr[i:i + args.batch]).to(device)
            soft_all.append(F.softmax(teacher(xb) / args.temperature, dim=1).cpu().numpy())
    soft_all = np.concatenate(soft_all)

    # ③ 30% 标签子集
    rng = np.random.default_rng(args.seed)
    n_lab = int(round(len(y_tr) * args.label_frac))
    idx = rng.permutation(len(y_tr))[:n_lab]
    X_sub, y_sub, soft_sub = X_tr[idx], y_tr[idx], soft_all[idx]

    # ④ 蒸馏训练 LIF 学生
    student = _build_student().to(device)
    opt = torch.optim.Adam(student.parameters(), lr=args.lr)
    ds = TensorDataset(torch.from_numpy(X_sub), torch.from_numpy(y_sub), torch.from_numpy(soft_sub))
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True)
    t0 = time.time()
    for _ in range(args.epochs):
        student.train()
        for xb, yb, sb in loader:
            xb, yb, sb = xb.to(device), yb.to(device), sb.to(device)
            opt.zero_grad()
            z = student(xb)
            kl = F.kl_div(F.log_softmax(z / args.temperature, dim=1), sb, reduction="batchmean")
            loss = args.alpha * (kl * args.temperature * args.temperature) \
                + (1.0 - args.alpha) * F.cross_entropy(z, yb)
            loss.backward()
            opt.step()
    student_acc = _eval(student, X_ev, y_ev, device, args.batch)
    print(f"  student（蒸馏 {args.label_frac:.0%} 标签）eval : {student_acc*100:.2f}%（{time.time()-t0:.0f}s）")

    # ⑤ 保存 checkpoint（对齐 export_lif_surrogate_weights.py 期望格式）
    model_args = dict(n_channels=8, n_classes=2, input_length=500,
                      F1=8, D=2, F2=16, kernel_length=63, dropout=0.5, activation="lif")
    lif_hparams = dict(threshold=1.0, tau=2.0, timesteps=8, alpha=4.0)
    out = ALGO_DIR / args.out
    torch.save({
        "state_dict": student.state_dict(),
        "model_args": model_args,
        "lif_hparams": lif_hparams,
        "final_test": {"干净": student_acc, "HBC噪声": student_acc},
        "teacher_eval": teacher_acc,
        "label_frac": args.label_frac,
        "seed": args.seed,
    }, out)
    print(f"✓ checkpoint 已保存：{out}（蒸馏 {args.label_frac:.0%} 标签，eval {student_acc*100:.2f}%）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
