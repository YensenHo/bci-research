#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_4class.py — BCIC IV 2a 22ch 四分类训练（EEGNet ELU 基线 + LIF-EEGNet surrogate）

目的
----
  在「22 EEG 通道 + 四分类」的原生 BCIC 2a 设置下，对比：
    - EEGNet（ELU 激活，ANN 基线）
    - LIF-EEGNet（surrogate gradient 直接训练，类脑化）
  验证「多分类 + 多通道」下，类脑化（surrogate 训练）是否仍零代价追平 ANN。

设计（复用已验证的 train_lif_surrogate.py 方法，但 n_channels=22 / n_classes=4）：
  - hold-out 早停：train 划 15% 验证，final test 用 eval（cross-session，只测一次）。
  - --model eegnet：构造 luckfox-deploy/model/eegnet.py 的 EEGNet（ELU），从零训练。
  - --model lif   ：构造 lif_eegnet.py 的 LIFEEGNet（activation='lif'），
                    可选 warm-start 从 EEGNet 4class ELU 权重热启动
                    （LIF 激活层无参数，state_dict 键零回归，strict=True 直接载入）。
                    可通过 --threshold/--tau/--timesteps/--alpha 调 LIF 超参，
                    不假设二分类的 thr=1.0/tau=2.0/T=8 仍最优。
  - 报告：本目录 eegnet_4class_report.txt / lif_4class_report_{tag}.txt，如实报精度。

用法
----
  EEGNet 基线：
    env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin \\
        /opt/miniconda3/envs/openmmlab/bin/python train_4class.py --model eegnet
  LIF-EEGNet（surrogate，热启动自 EEGNet 4class）：
    env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin \\
        /opt/miniconda3/envs/openmmlab/bin/python train_4class.py --model lif \\
        --threshold 1.0 --tau 2.0 --timesteps 8 --tag t1.0_tau2_T8
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ---- 路径 ----
ALGO_DIR = Path(__file__).resolve().parent
PROJECT = Path("/Users/heyixuan/BCI/luckfox-deploy")
DATA_DIR = PROJECT / "data"
MODEL_DIR = PROJECT / "model"
DATA_4CLASS = ALGO_DIR / "data_4class"

sys.path.insert(0, str(ALGO_DIR))       # from lif_eegnet import LIFEEGNet
sys.path.insert(0, str(DATA_DIR))       # from dataset import EEGDataset
sys.path.insert(0, str(MODEL_DIR))      # from eegnet import EEGNet

from dataset import EEGDataset  # noqa: E402
from eegnet import EEGNet       # noqa: E402
from lif_eegnet import LIFEEGNet  # noqa: E402


def _pick_device(no_cuda: bool):
    if no_cuda:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _build_model(args) -> nn.Module:
    """构造 EEGNet（elu）或 LIFEEGNet（lif）。两模型结构/参数量一致，仅激活层不同。"""
    common = dict(n_channels=args.n_channels, n_classes=args.n_classes,
                  input_length=500, F1=8, D=2, F2=16, kernel_length=63,
                  dropout=args.dropout)
    if args.model == "eegnet":
        return EEGNet(activation="elu", **common)
    return LIFEEGNet(activation="lif", threshold=args.threshold, tau=args.tau,
                     timesteps=args.timesteps, alpha=args.alpha, **common)


def _evaluate(model, loader, device) -> tuple[float, dict]:
    """返回 (acc, 发放率 dict)。lif 模型额外统计 act1/act2 平均发放率。"""
    model.eval()
    correct, total = 0, 0
    fr_sum = {"act1": 0.0, "act2": 0.0}
    is_lif = hasattr(model, "act1") and hasattr(model, "act2")
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            correct += (logits.argmax(1) == yb).sum().item()
            total += xb.size(0)
            if is_lif:
                fr_sum["act1"] += float(model.act1.last_firing_rate) * xb.size(0)
                fr_sum["act2"] += float(model.act2.last_firing_rate) * xb.size(0)
    fr = {k: v / total for k, v in fr_sum.items()} if total else {"act1": 0.0, "act2": 0.0}
    return correct / total, fr


def main() -> int:
    parser = argparse.ArgumentParser(description="BCIC 2a 22ch 四分类训练（EEGNet / LIF-EEGNet）")
    parser.add_argument("--model", type=str, default="eegnet", choices=["eegnet", "lif"])
    parser.add_argument("--n-channels", type=int, default=22)
    parser.add_argument("--n-classes", type=int, default=4)
    # LIF 超参（仅 --model lif 时生效）
    parser.add_argument("--threshold", type=float, default=1.0)
    parser.add_argument("--tau", type=float, default=2.0)
    parser.add_argument("--timesteps", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=4.0)
    parser.add_argument("--warm-start", type=str,
                        default=str(ALGO_DIR / "eegnet_4class.pth"),
                        help="LIF 热启动权重（EEGNet 4class ELU，state_dict 键零回归）")
    # 训练超参
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--cosine", action="store_true",
                        help="用 cosine 退火学习率（从 lr 线性退到 0），提升小模型收敛")
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--out", type=str, default="")
    parser.add_argument("--tag", type=str, default="")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = _pick_device(args.no_cuda)

    train_npz = DATA_4CLASS / "train.npz"
    eval_npz = DATA_4CLASS / "eval.npz"
    if not (train_npz.exists() and eval_npz.exists()):
        raise SystemExit(f"✗ 数据集缺失：{DATA_4CLASS}（先跑 preprocess_4class.py）")
    train_ds = EEGDataset(train_npz)
    eval_ds = EEGDataset(eval_npz)
    n_classes = args.n_classes
    print(f"设备     : {device}")
    print(f"模型     : {args.model}  ({'LIF-EEGNet surrogate' if args.model == 'lif' else 'EEGNet ELU'})")
    print(f"训练集   : {len(train_ds)} 试次 | 评估集 : {len(eval_ds)} 试次 | "
          f"n_channels={args.n_channels} n_classes={n_classes}")
    if args.model == "lif":
        print(f"LIF 超参 : threshold={args.threshold}, tau={args.tau}, "
              f"timesteps={args.timesteps}, alpha={args.alpha}")

    # ---- hold-out 划分：早停验证集与 final test（eval）分离 ----
    n_val = max(1, int(len(train_ds) * args.val_split))
    n_train = len(train_ds) - n_val
    g = torch.Generator().manual_seed(args.seed)
    train_ds, val_ds = torch.utils.data.random_split(train_ds, [n_train, n_val], generator=g)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False)
    print(f"hold-out : {n_train} 训练 / {n_val} 验证（早停用） | final test = eval（{len(eval_ds)} 试次）")

    model = _build_model(args)

    # ---- LIF 可选热启动：load EEGNet 4class ELU 权重 ----
    warm = False
    if args.model == "lif":
        ws = Path(args.warm_start)
        if ws.exists():
            ck = torch.load(ws, map_location="cpu", weights_only=False)
            model.load_state_dict(ck["state_dict"], strict=True)
            warm = True
            print(f"[热启动] 已加载 {ws.name} 的 conv/fc/BN 权重（ELU→LIF 激活位点无参数）")
        else:
            print(f"[警告] 未找到热启动权重 {ws}，LIF 从随机初始化训练")

    model = model.to(device)
    print(f"参数量   : {model.num_params():,}")

    # ---- 对照：热启动未微调，直接在 eval 上的准确率 ----
    if warm:
        with torch.no_grad():
            acc0, fr0 = _evaluate(model, eval_loader, device)
            print(f"\n[对照] EEGNet 权重直接换 LIF 激活（不微调）eval 准确率 : {acc0*100:.2f}%  "
                  f"(发放率 act1={fr0['act1']*100:.2f}% act2={fr0['act2']*100:.2f}%)")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
                 if args.cosine else None)
    history = []
    best_acc, best_epoch, best_state, patience_counter = 0.0, -1, None, 0
    print(f"\n开始训练 {args.model} ...")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        total_loss, total_correct, total_n = 0.0, 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
            total_correct += (logits.argmax(1) == yb).sum().item()
            total_n += xb.size(0)
        if scheduler is not None:
            scheduler.step()
        train_loss = total_loss / total_n
        train_acc = total_correct / total_n
        val_acc, _ = _evaluate(model, val_loader, device)
        dt = time.time() - t0
        history.append({"epoch": epoch, "train_loss": round(train_loss, 6),
                        "train_acc": round(train_acc, 4), "val_acc": round(val_acc, 4)})
        print(f"  Epoch {epoch:3d}/{args.epochs} | train_loss={train_loss:.4f} "
              f"train_acc={train_acc*100:5.2f}% val_acc={val_acc*100:5.2f}% | {dt:.1f}s")
        if val_acc > best_acc:
            best_acc, best_epoch = val_acc, epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\n早停：验证集连续 {args.patience} epoch 未提升。")
                break

    if best_state is None:
        raise SystemExit("✗ 训练未产生有效模型")
    model.load_state_dict(best_state)
    model.eval()

    # ---- final test：eval（cross-session，只测一次）----
    acc, fr = _evaluate(model, eval_loader, device)
    print(f"\n[final test] {args.model} (cross-session, E 会话, 只测一次) : {acc*100:.2f}%")
    if args.model == "lif":
        print(f"       发放率 act1={fr['act1']*100:.2f}% act2={fr['act2']*100:.2f}%")

    # ---- 保存 ----
    out = Path(args.out) if args.out else (
        ALGO_DIR / f"{args.model}_4class.pth")
    model_args = dict(n_channels=args.n_channels, n_classes=n_classes, input_length=500,
                      F1=8, D=2, F2=16, kernel_length=63, dropout=0.5,
                      activation="elu" if args.model == "eegnet" else "lif")
    torch.save({"state_dict": best_state, "epoch": best_epoch, "val_acc": best_acc,
                "final_test": acc, "model_args": model_args,
                "lif_hparams": (dict(threshold=args.threshold, tau=args.tau,
                                     timesteps=args.timesteps, alpha=args.alpha)
                                if args.model == "lif" else None)}, out)
    print(f"\n✓ 权重已保存：{out}（epoch {best_epoch}，val_acc={best_acc*100:.2f}%）")

    # ---- 报告 ----
    tag = args.tag or ("lif" if args.model == "lif" else "eegnet")
    report = ALGO_DIR / f"{args.model}_4class_report_{tag}.txt"
    chance = 100.0 / n_classes
    lines = [
        f"BCIC 2a 22ch 四分类训练报告 — {'LIF-EEGNet surrogate' if args.model == 'lif' else 'EEGNet ELU 基线'}",
        "=" * 64,
        f"tag             : {tag}",
        f"最佳验证准确率  : {best_acc*100:.2f}%（epoch {best_epoch}，hold-out 早停用）",
        f"final test(eval): {acc*100:.2f}%（cross-session，只测一次）",
        f"chance level    : {chance:.1f}%",
        f"n_channels      : {args.n_channels} | n_classes : {n_classes}",
        f"训练样本数      : {n_train} | 验证 {n_val} | final test {len(eval_ds)}",
        f"参数量          : {model.num_params():,}",
        f"设备            : {device}",
        f"超参数          : lr={args.lr}, batch={args.batch_size}, epochs={args.epochs}, "
        f"patience={args.patience}, seed={args.seed}, val_split={args.val_split}",
    ]
    if args.model == "lif":
        lines += [
            f"LIF 超参        : threshold={args.threshold}, tau={args.tau}, "
            f"timesteps={args.timesteps}, alpha={args.alpha}",
            f"热启动          : {warm}（{Path(args.warm_start).name}）",
            f"发放率          : act1={fr['act1']*100:.2f}% act2={fr['act2']*100:.2f}%",
        ]
    lines += ["-" * 64, "逐 epoch 记录 (epoch, train_loss, train_acc, val_acc):"]
    for h in history:
        lines.append(f"  {h['epoch']:3d}  {h['train_loss']:.6f}  "
                     f"{h['train_acc']*100:5.2f}%  {h['val_acc']*100:5.2f}%")
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  报告已写入：{report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
