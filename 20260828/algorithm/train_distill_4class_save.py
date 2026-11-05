#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""train_distill_4class_save.py — 蒸馏 4 类 22ch LIF 学生并保存 checkpoint（供端侧 C 导出）

目的
----
  distill_4class.py 只跑完评估就丢弃模型（且只跑了 relu 学生，LIF 学生「可跳过」没跑），
  无法端侧导出。本脚本补上「训练蒸馏×LIF 学生 + 保存」：大 EEG-TCNet teacher（ELU，100% 标签）
  → 软标签 → EEGTCNetLIF student（30% 标签蒸馏），保存 checkpoint
  （state_dict + model_args + lif_hparams + final_test），供 export_tcnet_snn_weights.py
  导出 C 头、verify 脚本逐位验证。

  回答「可穿戴部署」方向的关键闭环：蒸馏训出来的类脑模型能否直接走端侧部署链路（4 类 22ch）。

  对齐协议（与 distill_4class.py 完全一致，唯一变量 = 是否用 teacher 软标签）：
  teacher F1=16/D=2/F2=32/FT=24；student brief F1=8/D=2/F2=16/FT=12；T=4、α=0.5；
  100 epoch 无早停、cosine 退火、batch 16、lr 1e-3、dropout 0.25。

用法
----
  env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin \
    /opt/miniconda3/envs/openmmlab/bin/python train_distill_4class_save.py \
    --label-frac 0.3 --seed 42 --out distill_4class_lif_30pct_s42.pth --smoke
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ALGO_DIR = Path(__file__).resolve().parent
DATA_4CLASS = ALGO_DIR / "data_4class"
sys.path.insert(0, str(ALGO_DIR))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from eeg_tcnet import EEGTCNet          # noqa: E402  （teacher，ELU）
from eeg_tcnet_lif import EEGTCNetLIF   # noqa: E402  （student，LIF）

# brief 结构（对齐 distill_4class.py BRIEF + train_eeg_tcnet_lif.py 默认）
BRIEF = dict(kernel_length=63, tcn_kernel_size=12, pool1=4, pool2=8, readout_mode="gap")


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


def _build_teacher(n_channels: int, n_classes: int, input_length: int) -> EEGTCNet:
    return EEGTCNet(n_channels=n_channels, n_classes=n_classes, input_length=input_length,
                    F1=16, D=2, F2=32, FT=24, dropout=0.25,
                    activation="elu", **BRIEF)


def _build_student(activation: str, lif_hp: dict, n_channels: int, n_classes: int,
                   input_length: int) -> EEGTCNetLIF:
    return EEGTCNetLIF(n_channels=n_channels, n_classes=n_classes, input_length=input_length,
                       F1=8, D=2, F2=16, FT=12, dropout=0.25,
                       activation=activation, **lif_hp, **BRIEF)


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
    ap = argparse.ArgumentParser(description="蒸馏 4 类 22ch LIF 学生并保存 checkpoint")
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--temperature", type=float, default=4.0)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--label-frac", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default="distill_4class_lif_30pct_s42.pth")
    ap.add_argument("--student", type=str, default="lif", choices=["relu", "lif"],
                    help="student 激活：relu（热启动源）/ lif（默认，可 --warm-start 热启动）")
    ap.add_argument("--warm-start", type=str, default="",
                    help="热启动 checkpoint（relu 学生权重 → LIF，strict 键零回归）")
    # LIF 超参（对齐 distill_4class.py 默认）
    ap.add_argument("--threshold", type=float, default=1.0)
    ap.add_argument("--tau", type=float, default=2.0)
    ap.add_argument("--timesteps", type=int, default=8)
    ap.add_argument("--alpha-lif", type=float, default=4.0)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--n-channels", type=int, default=22)
    ap.add_argument("--n-classes", type=int, default=4)
    ap.add_argument("--input-length", type=int, default=500)
    ap.add_argument("--data-dir", type=str, default="data_4class")
    args = ap.parse_args()

    if args.smoke:
        args.epochs = 2

    device = _pick_device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data_dir = ALGO_DIR / args.data_dir
    train_npz = data_dir / "train.npz"
    eval_npz = data_dir / "eval.npz"
    for p in (train_npz, eval_npz):
        if not p.exists():
            raise SystemExit(f"✗ 数据集缺失：{p}（--data-dir {args.data_dir}）")
    tr = np.load(train_npz)
    ev = np.load(eval_npz)
    X_tr = tr["X"].astype(np.float32)
    y_tr = tr["y"].astype(np.int64)
    X_ev = ev["X"].astype(np.float32)
    y_ev = ev["y"].astype(np.int64)
    lif_hp = dict(threshold=args.threshold, tau=args.tau,
                  timesteps=args.timesteps, alpha=args.alpha_lif)
    print(f"设备 : {device} | train {X_tr.shape} | eval {X_ev.shape} | "
          f"标签预算 {args.label_frac:.0%} | 学生 LIF thr={args.threshold}/tau={args.tau}/T={args.timesteps}")

    # ① teacher（大 ELU EEG-TCNet，100% 标签）
    teacher = _build_teacher(args.n_channels, args.n_classes, args.input_length).to(device)
    t0 = time.time()
    opt = torch.optim.Adam(teacher.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    ds = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True)
    for ep in range(args.epochs):
        teacher.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            F.cross_entropy(teacher(xb), yb).backward()
            opt.step()
        sched.step()
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

    # ③ 标签子集（30% 标签，蒸馏红利最大）
    rng = np.random.default_rng(args.seed)
    n_lab = int(round(len(y_tr) * args.label_frac))
    idx = rng.permutation(len(y_tr))[:n_lab]
    X_sub, y_sub, soft_sub = X_tr[idx], y_tr[idx], soft_all[idx]

    # ④ 蒸馏训练学生（LIF / relu）
    student = _build_student(args.student, lif_hp, args.n_channels, args.n_classes,
                             args.input_length).to(device)
    if args.warm_start:
        ws = torch.load(Path(args.warm_start), map_location="cpu", weights_only=False)
        student.load_state_dict(ws["state_dict"], strict=True)  # relu→lif 激活层无参数，strict 键零回归
        print(f"[热启动] 已加载 {args.warm_start} 的 conv/BN/fc 权重（{args.student} 激活位点无参数）")
    opt = torch.optim.Adam(student.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    ds = TensorDataset(torch.from_numpy(X_sub), torch.from_numpy(y_sub), torch.from_numpy(soft_sub))
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True)
    t0 = time.time()
    for ep in range(args.epochs):
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
        sched.step()
    student_acc = _eval(student, X_ev, y_ev, device, args.batch)
    print(f"  student（蒸馏 {args.label_frac:.0%} 标签）eval : {student_acc*100:.2f}%（{time.time()-t0:.0f}s）")

    # ⑤ 保存 checkpoint（对齐 export_tcnet_snn_weights.py 期望格式：
    #     state_dict + model_args（不含 lif 超参）+ lif_hparams + final_test 标量）
    model_args = dict(n_channels=args.n_channels, n_classes=args.n_classes,
                      input_length=args.input_length,
                      F1=8, D=2, F2=16, FT=12, kernel_length=BRIEF["kernel_length"],
                      tcn_kernel_size=BRIEF["tcn_kernel_size"],
                      pool1=BRIEF["pool1"], pool2=BRIEF["pool2"],
                      readout_mode=BRIEF["readout_mode"],
                      dropout=0.25, activation=args.student)
    out = ALGO_DIR / args.out
    torch.save({
        "state_dict": student.state_dict(),
        "model_args": model_args,
        "lif_hparams": lif_hp,
        "final_test": student_acc,
        "teacher_eval": teacher_acc,
        "label_frac": args.label_frac,
        "seed": args.seed,
        "smoke": args.smoke,
    }, out)
    print(f"✓ checkpoint 已保存：{out}（蒸馏 {args.label_frac:.0%} 标签，eval {student_acc*100:.2f}%）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
