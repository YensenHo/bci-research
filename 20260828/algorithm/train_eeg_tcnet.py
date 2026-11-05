#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_eeg_tcnet.py — EEG-TCNet 四分类训练（cross-session 协议，与 train_4class.py 一致）

目的
----
  在 BCIC IV 2a 22ch 四分类（cross-session, subject-independent）协议下训练
  世界主流轻量模型 EEG-TCNet（Ingolfsson 2020, arXiv:2006.00622），与基线对比：
    - EEGNet ELU 基线  : 58.29%（记录在 eegnet_4class_report_eegnet_v2.txt）
    - LIF-EEGNet 基线  : 56.87%（记录在 对比报告_四分类22ch_类脑化零代价验证.md）
  评估：①能否提升精度 ②参数量 / MPS 前向耗时 / RV1106 端侧可行性。

协议（与 train_4class.py / 对比报告完全一致，保证可比）
----------------------------------------------------
  - hold-out 早停：train 划 15% 验证（388 试次，seed=42），早停 patience=25
  - final test = eval（cross-session，E 会话，只测一次，消除选择偏差）
  - 优化器 Adam lr=0.001，batch=16，CrossEntropyLoss，cosine 退火
  - dropout=0.25，seed=42，epochs=150
  - 默认激活 ELU（与 EEGNet 基线一致）；--activation relu 验证 relu 版
    （为后续 LIF 化 + 板载转换准备，板载链 relu/lif 兼容）

用法
----
  ELU 版（主实验）：
    env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin \\
        /opt/miniconda3/envs/openmmlab/bin/python train_eeg_tcnet.py
  ReLU 版（验证，为 LIF 化/板载准备）：
    env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin \\
        /opt/miniconda3/envs/openmmlab/bin/python train_eeg_tcnet.py --activation relu

产物（均在 algorithm/ 目录）
--------------------------
  eeg_tcnet_4class{,_relu}.pth   权重（state_dict + 训练信息）
  eeg_tcnet_4class_report{,_relu}.txt  逐 epoch 训练报告
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
DATA_4CLASS = ALGO_DIR / "data_4class"

sys.path.insert(0, str(ALGO_DIR))   # from eeg_tcnet import EEGTCNet
sys.path.insert(0, str(DATA_DIR))   # from dataset import EEGDataset

from dataset import EEGDataset  # noqa: E402
from eeg_tcnet import EEGTCNet   # noqa: E402


def _pick_device(no_cuda: bool):
    if no_cuda:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _measure_latency(model, device, n_channels: int, input_length: int,
                     batch: int = 1, repeats: int = 30) -> float:
    """MPS/CPU 单次前向耗时（毫秒）。warmup 后取 repeats 次均值，供端侧可行性评估。"""
    model.eval()
    dummy = torch.randn(batch, 1, n_channels, input_length, device=device)
    with torch.no_grad():
        for _ in range(5):            # warmup（含 MPS 图编译/算子调度）
            model(dummy)
        if device.type == "mps":
            torch.mps.synchronize()
        t0 = time.perf_counter()
        for _ in range(repeats):
            model(dummy)
        if device.type == "mps":
            torch.mps.synchronize()
        dt = (time.perf_counter() - t0) / repeats * 1000
    return dt


def _evaluate(model, loader, device) -> float:
    """返回 top-1 准确率。"""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            correct += (logits.argmax(1) == yb).sum().item()
            total += xb.size(0)
    return correct / total


def main() -> int:
    parser = argparse.ArgumentParser(description="EEG-TCNet 22ch 四分类训练（cross-session）")
    parser.add_argument("--n-channels", type=int, default=22)
    parser.add_argument("--n-classes", type=int, default=4)
    parser.add_argument("--input-length", type=int, default=500)
    parser.add_argument("--activation", type=str, default="elu",
                        choices=["elu", "relu", "lif"],
                        help="激活函数（elu 默认与基线一致；relu 为板载转换链验证；lif 预留）")
    # 模型结构（默认 = 原文固定配置：F1=8, D=2, F2=16, FT=12, KE=32, KT=4, L=2）
    parser.add_argument("--F1", type=int, default=8)
    parser.add_argument("--D", type=int, default=2)
    parser.add_argument("--F2", type=int, default=16)
    parser.add_argument("--FT", type=int, default=12)
    parser.add_argument("--kernel-length", type=int, default=32)
    parser.add_argument("--tcn-kernel-size", type=int, default=4)
    parser.add_argument("--pool1", type=int, default=8,
                        help="φ2 平均池化步长（原文 8；诊断变体可设 4 → T/32）")
    parser.add_argument("--pool2", type=int, default=8)
    parser.add_argument("--readout-mode", type=str, default="last",
                        choices=["last", "gap"],
                        help="分类头读出：last=最后时间步（原文）；gap=全局平均池化（简报描述版）")
    # 训练超参（协议固定，与 train_4class.py / 对比报告一致）
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--dropout", type=float, default=0.25)
    # 协议固定 cosine 退火（与 train_4class.py --cosine 一致），无需开关
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--out", type=str, default="")
    parser.add_argument("--tag", type=str, default="",
                        help="产物后缀（默认=activation 名），变体实验用，避免覆盖")
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
    print(f"模型     : EEG-TCNet（activation={args.activation}）")
    print(f"训练集   : {len(train_ds)} 试次 | 评估集 : {len(eval_ds)} 试次 | "
          f"n_channels={args.n_channels} n_classes={n_classes}")

    # ---- hold-out 划分：早停验证集与 final test（eval）分离 ----
    n_val = max(1, int(len(train_ds) * args.val_split))
    n_train = len(train_ds) - n_val
    g = torch.Generator().manual_seed(args.seed)
    train_ds, val_ds = torch.utils.data.random_split(train_ds, [n_train, n_val], generator=g)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False)
    print(f"hold-out : {n_train} 训练 / {n_val} 验证（早停用） | final test = eval（{len(eval_ds)} 试次）")

    model = EEGTCNet(n_channels=args.n_channels, n_classes=n_classes,
                     input_length=args.input_length, F1=args.F1, D=args.D,
                     F2=args.F2, FT=args.FT, kernel_length=args.kernel_length,
                     tcn_kernel_size=args.tcn_kernel_size,
                     pool1=args.pool1, pool2=args.pool2,
                     readout_mode=args.readout_mode,
                     dropout=args.dropout, activation=args.activation)
    model = model.to(device)
    n_params = model.num_params()
    print(f"参数量   : {n_params:,}")

    # ---- MPS 前向耗时（单试次 / batch=16，端侧可行性评估用）----
    if device.type != "cpu":
        lat1 = _measure_latency(model, device, args.n_channels, args.input_length, batch=1)
        lat16 = _measure_latency(model, device, args.n_channels, args.input_length, batch=16)
        print(f"MPS 前向耗时 : 单试次 {lat1:.2f} ms | batch16 {lat16:.2f} ms（warmup 后 30 次均值）")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    history = []
    best_acc, best_epoch, best_state, patience_counter = 0.0, -1, None, 0
    print(f"\n开始训练 EEG-TCNet（activation={args.activation}）...")
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
        scheduler.step()  # cosine 退火（协议固定）
        train_loss = total_loss / total_n
        train_acc = total_correct / total_n
        val_acc = _evaluate(model, val_loader, device)
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
    acc = _evaluate(model, eval_loader, device)
    print(f"\n[final test] EEG-TCNet-{args.activation} (cross-session, E 会话, 只测一次) : {acc*100:.2f}%")

    # ---- 保存 ----
    tag = args.tag or args.activation
    out = Path(args.out) if args.out else (ALGO_DIR / f"eeg_tcnet_4class_{tag}.pth")
    model_args = dict(n_channels=args.n_channels, n_classes=n_classes,
                      input_length=args.input_length, F1=args.F1, D=args.D,
                      F2=args.F2, FT=args.FT, kernel_length=args.kernel_length,
                      tcn_kernel_size=args.tcn_kernel_size,
                      pool1=args.pool1, pool2=args.pool2,
                      readout_mode=args.readout_mode,
                      dropout=args.dropout, activation=args.activation)
    torch.save({"state_dict": best_state, "epoch": best_epoch, "val_acc": best_acc,
                "final_test": acc, "model_args": model_args, "n_params": n_params}, out)
    print(f"\n✓ 权重已保存：{out}（epoch {best_epoch}，val_acc={best_acc*100:.2f}%）")

    # ---- 报告 ----
    report = ALGO_DIR / f"eeg_tcnet_4class_report_{tag}.txt"
    chance = 100.0 / n_classes
    lines = [
        f"BCIC 2a 22ch 四分类训练报告 — EEG-TCNet（activation={args.activation}）",
        "=" * 64,
        f"tag             : {tag}",
        f"最佳验证准确率  : {best_acc*100:.2f}%（epoch {best_epoch}，hold-out 早停用）",
        f"final test(eval): {acc*100:.2f}%（cross-session，只测一次）",
        f"chance level    : {chance:.1f}%",
        f"n_channels      : {args.n_channels} | n_classes : {n_classes}",
        f"训练样本数      : {n_train} | 验证 {n_val} | final test {len(eval_ds)}",
        f"参数量          : {n_params:,}",
        f"设备            : {device}",
        f"超参数          : lr={args.lr}, batch={args.batch_size}, epochs={args.epochs}, "
        f"patience={args.patience}, seed={args.seed}, val_split={args.val_split}, "
        f"dropout={args.dropout}, cosine=True",
        f"结构            : F1={args.F1}, D={args.D}, F2={args.F2}, FT={args.FT}, "
        f"kernel={args.kernel_length}, tcn_kernel={args.tcn_kernel_size}, "
        f"dilations=(1,2), pool1={args.pool1}, pool2={args.pool2}, "
        f"readout={args.readout_mode}",
        "-" * 64, "逐 epoch 记录 (epoch, train_loss, train_acc, val_acc):",
    ]
    for h in history:
        lines.append(f"  {h['epoch']:3d}  {h['train_loss']:.6f}  "
                     f"{h['train_acc']*100:5.2f}%  {h['val_acc']*100:5.2f}%")
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  报告已写入：{report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
