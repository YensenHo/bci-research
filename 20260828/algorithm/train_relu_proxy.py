#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_relu_proxy.py — 训练 ReLU-EEGNet 代理权重（LIF-EEGNet 板载闭环 · S0 步）

目的
----
  ANN-to-SNN 转换要求源 ANN 的激活是 ReLU（非负、单调、可被 LIF 脉冲近似）。
  现有 eegnet.pth 是 ELU 版，本脚本训练一个 ReLU 代理：
    - 结构 = LIFEEGNet(activation='relu')，与 eegnet.py 零回归（可热启动）；
    - 热启动：先 load eegnet.pth 的 conv/fc/BN 权重（ELU→ReLU 仅激活位点不同）；
    - 微调：hold-out 早停（从 train 划 15% 做验证，final test 用 eval，消除选择偏差）；
    - 报告：干净 / HBC 噪声 两个维度的准确率，与 ELU 基线对比，如实记录
      ReLU 负半轴截断的精度代价。

用法
----
  env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin \
      /opt/miniconda3/envs/openmmlab/bin/python train_relu_proxy.py \
      --warm-start /Users/heyixuan/BCI/luckfox-deploy/model/eegnet.pth \
      --epochs 50 --n-classes 2

架构硬要求：n_classes 可配置（默认 2，为 BCIC 2a 四分类升级留接口），不写死。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ---- 路径：复用 luckfox-deploy 的 dataset，复用本目录的 lif_eegnet ----
ALGO_DIR = Path(__file__).resolve().parent
PROJECT = Path("/Users/heyixuan/BCI/luckfox-deploy")
PROCESSED = PROJECT / "data" / "processed"
DATA_DIR = PROJECT / "data"

sys.path.insert(0, str(ALGO_DIR))       # from lif_eegnet import LIFEEGNet
sys.path.insert(0, str(DATA_DIR))       # from dataset import EEGDataset

from lif_eegnet import LIFEEGNet  # noqa: E402


# ---------------------------------------------------------------------------
# 设备选择 / HBC 噪声增强（与 luckfox-deploy/model/train.py 语义一致）
# ---------------------------------------------------------------------------
def _pick_device(no_cuda: bool):
    if no_cuda:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _hbc_augment(x, qbits: int = 5, qrange: float = 4.0, noise_sigma: float = 0.1):
    """HBC 链路噪声增强：5bit 量化 → 反量化 → 信道 AWGN（对齐 train.py 的 _hbc_augment）。"""
    levels = 2 ** qbits
    xq = ((x + qrange) / (2.0 * qrange) * (levels - 1)).round().clamp(0, levels - 1)
    xq = xq / (levels - 1) * 2.0 * qrange - qrange
    return xq + noise_sigma * torch.randn_like(xq)


def _evaluate(model, loader, device, use_hbc=False):
    """返回 (acc)。HBC 噪声评估用固定种子保证可复现。"""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            if use_hbc:
                torch.manual_seed(1234)          # 固定噪声种子，可复现
                xb = _hbc_augment(xb)
            logits = model(xb)
            correct += (logits.argmax(1) == yb).sum().item()
            total += xb.size(0)
    return correct / total


def _save_curve(history, out_dir: Path, tag: str) -> None:
    """训练曲线存 PNG（真实产物，失败不影响）。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        epochs = [h["epoch"] for h in history]
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(8, 7), dpi=150, facecolor="#1B2A4A")
        for a in (a1, a2):
            a.set_facecolor("#243555")
            a.tick_params(colors="white")
            for sp in a.spines.values():
                sp.set_color("#8FA3BF")
            a.grid(True, color="#8FA3BF", alpha=0.25)
        a1.plot(epochs, [h["train_loss"] for h in history], color="#00B4D8", marker="o", ms=3)
        a1.set_ylabel("Loss", color="white")
        a1.set_title(f"ReLU-EEGNet 代理训练（{tag}）", color="white", fontsize=14, weight="bold")
        a2.plot(epochs, [h["train_acc"] * 100 for h in history], color="#2E86AB", marker="o", ms=3, label="train")
        a2.plot(epochs, [h["val_acc"] * 100 for h in history], color="#E76F51", marker="o", ms=3, label="val")
        a2.set_ylabel("Acc (%)", color="white"); a2.set_xlabel("Epoch", color="white")
        a2.legend(facecolor="#243555", edgecolor="#8FA3BF", labelcolor="white")
        fig.tight_layout()
        p = out_dir / f"relu_proxy_curve_{tag}.png"
        fig.savefig(p, facecolor="#1B2A4A", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  训练曲线已保存：{p}")
    except Exception as e:  # noqa: BLE001
        print(f"  ⚠ 曲线图保存失败（不影响训练）：{e}")


def main() -> int:
    parser = argparse.ArgumentParser(description="训练 ReLU-EEGNet 代理")
    parser.add_argument("--warm-start", type=str,
                        default=str(PROJECT / "model" / "eegnet.pth"),
                        help="ELU 权重热启动路径（eegnet.pth）")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--n-classes", type=int, default=2,
                        help="分类数（默认 2，为 BCIC 2a 四分类升级留接口）")
    parser.add_argument("--hbc-aug", action="store_true", help="训练时做 HBC 噪声增强")
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--out", type=str, default=str(ALGO_DIR / "relu_proxy.pth"))
    args = parser.parse_args()

    from dataset import EEGDataset  # noqa: E402

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = _pick_device(args.no_cuda)

    train_npz = PROCESSED / "train.npz"
    eval_npz = PROCESSED / "eval.npz"
    if not (train_npz.exists() and eval_npz.exists()):
        raise SystemExit(f"✗ 数据集缺失：{PROCESSED}")
    train_ds = EEGDataset(train_npz)
    eval_ds = EEGDataset(eval_npz)
    n_classes = args.n_classes if args.n_classes else train_ds.n_classes
    print(f"设备     : {device}")
    print(f"训练集   : {len(train_ds)} 试次 | 评估集 : {len(eval_ds)} 试次 | n_classes={n_classes}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False)

    # ---- hold-out 划分：早停验证集与 final test（eval）分离 ----
    n_val = max(1, int(len(train_ds) * args.val_split))
    n_train = len(train_ds) - n_val
    g = torch.Generator().manual_seed(args.seed)
    train_ds, val_ds = torch.utils.data.random_split(train_ds, [n_train, n_val], generator=g)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    print(f"hold-out : {n_train} 训练 / {n_val} 验证（早停用） | final test = eval（{len(eval_ds)} 试次）")

    # ---- 模型：ReLU-EEGNet（n_classes 可配置）----
    model = LIFEEGNet(n_channels=8, n_classes=n_classes, input_length=500,
                      F1=8, D=2, F2=16, kernel_length=63, dropout=0.5,
                      activation="relu")

    # ---- 热启动：load ELU eegnet.pth（激活位点无参数，state_dict 键零回归）----
    ws = Path(args.warm_start)
    if ws.exists():
        ck = torch.load(ws, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["state_dict"], strict=True)
        print(f"[热启动] 已加载 {ws.name} 的 conv/fc/BN 权重（ELU→ReLU 激活位点无参数）")
    else:
        print(f"[警告] 未找到热启动权重 {ws}，从随机初始化训练")

    model = model.to(device)
    print(f"参数量   : {model.num_params():,}")

    # ---- 基线对照：ELU 权重直接换 ReLU（不微调）在 eval 上的干净/HBC 准确率 ----
    print("\n[对照] ELU 权重直接换 ReLU 激活（不微调）—— 量化 ReLU 负半轴截断的即时代价：")
    with torch.no_grad():
        for tag, use_hbc in (("干净", False), ("HBC噪声", True)):
            acc = _evaluate(model, eval_loader, device, use_hbc=use_hbc)
            print(f"        ReLU(热启动未微调) {tag} : {acc*100:.2f}%")

    # ---- 微调 ----
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    history = []
    best_acc, best_epoch, best_state, patience_counter = 0.0, -1, None, 0
    print("\n开始微调 ReLU-EEGNet 代理 ...")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        total_loss, total_correct, total_n = 0.0, 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            if args.hbc_aug:
                xb = _hbc_augment(xb)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
            total_correct += (logits.argmax(1) == yb).sum().item()
            total_n += xb.size(0)
        train_loss = total_loss / total_n
        train_acc = total_correct / total_n
        model.eval()
        val_correct, val_n = 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                val_correct += (logits.argmax(1) == yb).sum().item()
                val_n += xb.size(0)
        val_acc = val_correct / val_n
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

    _save_curve(history, ALGO_DIR, "relu")
    if best_state is None:
        raise SystemExit("✗ 训练未产生有效模型")
    model.load_state_dict(best_state)
    model.eval()

    # ---- final test：clean + HBC 噪声（eval 只测一次）----
    print("\n[final test] ReLU-EEGNet 代理（eval，cross-session，只测一次）：")
    results = {}
    with torch.no_grad():
        for tag, use_hbc in (("干净", False), ("HBC噪声", True)):
            acc = _evaluate(model, eval_loader, device, use_hbc=use_hbc)
            results[tag] = acc
            print(f"        ReLU 代理 {tag} : {acc*100:.2f}%")

    # ---- 保存 ----
    out = Path(args.out)
    torch.save({"state_dict": best_state, "epoch": best_epoch, "val_acc": best_acc,
                "final_test": results,
                "model_args": dict(n_channels=8, n_classes=n_classes, input_length=500,
                                   F1=8, D=2, F2=16, kernel_length=63, dropout=0.5,
                                   activation="relu")}, out)
    print(f"\n✓ ReLU 代理已保存：{out}（epoch {best_epoch}，val_acc={best_acc*100:.2f}%）")

    # ---- 报告 ----
    report = ALGO_DIR / "relu_proxy_report.txt"
    lines = [
        "ReLU-EEGNet 代理训练报告（LIF-EEGNet 板载闭环 S0）",
        "=" * 56,
        f"最佳验证准确率 : {best_acc*100:.2f}%（epoch {best_epoch}，hold-out 早停用）",
        f"final test（eval）: 干净 {results['干净']*100:.2f}% / HBC噪声 {results['HBC噪声']*100:.2f}%",
        f"n_classes       : {n_classes}",
        f"训练样本数      : {n_train} | 验证 {n_val} | final test {len(eval_ds)}",
        f"参数量          : {model.num_params():,}",
        f"设备            : {device}",
        f"超参数          : lr={args.lr}, batch={args.batch_size}, epochs={args.epochs}, "
        f"patience={args.patience}, seed={args.seed}, hbc_aug={args.hbc_aug}",
        "-" * 56,
        "逐 epoch 记录 (epoch, train_loss, train_acc, val_acc):",
    ]
    for h in history:
        lines.append(f"  {h['epoch']:3d}  {h['train_loss']:.6f}  "
                     f"{h['train_acc']*100:5.2f}%  {h['val_acc']*100:5.2f}%")
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  报告已写入：{report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
