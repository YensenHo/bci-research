#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_signed_lif.py — Signed LIF-EEGNet 四分类训练（正负脉冲分离，补回负半轴）

目的
----
  验证「signed LIF 编码（正负脉冲分离）」能否补回普通 LIF 率编码丢弃的负半轴信息，
  把四分类 + 22ch 下类脑化的代价（普通 LIF 56.87% vs ELU 基线 58.29%，约 -1.4pp）
  降到接近 0（追平甚至超过 ELU）。

背景（已定位的问题）
-------------------
  普通 LIFActivation 只发正脉冲：负半轴输入永不发放 → 发放率恒 0，等效 ReLU 截断，
  丢弃 ELU 保留的负半轴信息。二分类 8ch 下负半轴不重要（零代价 77.7%），
  四分类 22ch 下类别多、特征要求高，代价显现（-1.4pp）。

做法
----
  用 lif_eegnet.SignedLIFActivation（双通道 signed 编码）：
    正通道 mem_p = decay*mem_p + relu(x)  → 发正脉冲
    负通道 mem_n = decay*mem_n + relu(-x) → 发负脉冲
    输出 = (sum spk_p - sum spk_n) / T ∈ [-1,1]
  两通道都走 surrogate gradient，反向可导。激活层无参数，state_dict 键与 EEGNet ELU
  完全一致，可直接 strict=True 热启动自 eegnet_4class.pth。

训练协议（与 train_4class.py 的 LIF baseline 完全一致，保证可比）
------------------------------------------------------------------
  - hold-out 早停：train 划 15% 验证，final test = eval（cross-session，只测一次）。
  - Adam lr=0.001, batch=16, CrossEntropyLoss, cosine 退火, dropout=0.25, seed=42。
  - thr/tau/T 可调，多组扫描找最优。

用法
----
  env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin \
      /opt/miniconda3/envs/openmmlab/bin/python train_signed_lif.py \
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

# ---- 路径（复用 train_4class.py 的依赖，不改 luckfox-deploy）----
ALGO_DIR = Path(__file__).resolve().parent
PROJECT = Path("/Users/heyixuan/BCI/luckfox-deploy")
DATA_DIR = PROJECT / "data"
MODEL_DIR = PROJECT / "model"
DATA_4CLASS = ALGO_DIR / "data_4class"

sys.path.insert(0, str(ALGO_DIR))       # from lif_eegnet import LIFEEGNet
sys.path.insert(0, str(DATA_DIR))       # from dataset import EEGDataset

from dataset import EEGDataset          # noqa: E402
from lif_eegnet import LIFEEGNet        # noqa: E402


def _pick_device(no_cuda: bool):
    if no_cuda:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _build_model(args) -> nn.Module:
    """构造 Signed LIF-EEGNet。结构/参数量与 EEGNet ELU、普通 LIF-EEGNet 完全一致，
    仅激活位点换成 SignedLIFActivation（正负脉冲分离）。"""
    common = dict(n_channels=args.n_channels, n_classes=args.n_classes,
                  input_length=500, F1=8, D=2, F2=16, kernel_length=63,
                  dropout=args.dropout)
    return LIFEEGNet(activation="signed_lif", threshold=args.threshold, tau=args.tau,
                     timesteps=args.timesteps, alpha=args.alpha, **common)


def _evaluate(model, loader, device) -> tuple[float, dict]:
    """返回 (acc, 发放率 dict)。

    发放率 dict 记录 act1/act2 的净发放率（可正可负）、正通道、负通道发放率，
    用于诊断「负通道到底发不发、发多少」（验证负半轴信息是否被真正编码进网络）。
    """
    model.eval()
    correct, total = 0, 0
    keys = ["net", "pos", "neg"]
    fr_sum = {f"act1_{k}": 0.0 for k in keys}
    fr_sum.update({f"act2_{k}": 0.0 for k in keys})
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            correct += (logits.argmax(1) == yb).sum().item()
            total += xb.size(0)
            for act in ("act1", "act2"):
                layer = getattr(model, act)
                fr_sum[f"{act}_net"] += float(layer.last_firing_rate) * xb.size(0)
                fr_sum[f"{act}_pos"] += float(layer.last_pos_rate) * xb.size(0)
                fr_sum[f"{act}_neg"] += float(layer.last_neg_rate) * xb.size(0)
    fr = {k: v / total for k, v in fr_sum.items()} if total else {k: 0.0 for k in fr_sum}
    return correct / total, fr


def main() -> int:
    parser = argparse.ArgumentParser(
        description="BCIC 2a 22ch 四分类训练（Signed LIF-EEGNet，正负脉冲分离）")
    parser.add_argument("--n-channels", type=int, default=22)
    parser.add_argument("--n-classes", type=int, default=4)
    # Signed LIF 超参
    parser.add_argument("--threshold", type=float, default=1.0)
    parser.add_argument("--tau", type=float, default=2.0)
    parser.add_argument("--timesteps", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=4.0)
    parser.add_argument("--warm-start", type=str,
                        default=str(ALGO_DIR / "eegnet_4class.pth"),
                        help="Signed LIF 热启动权重（EEGNet 4class ELU，state_dict 键零回归）")
    # 训练超参（默认对齐 LIF baseline：epochs=150/patience=25/dropout=0.25/cosine）
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--cosine", action="store_true", default=True,
                        help="用 cosine 退火学习率（默认开，对齐 LIF baseline 协议）")
    parser.add_argument("--no-cosine", action="store_true")
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
    print(f"模型     : Signed LIF-EEGNet（正负脉冲分离，输出发放率 ∈[-1,1]）")
    print(f"训练集   : {len(train_ds)} 试次 | 评估集 : {len(eval_ds)} 试次 | "
          f"n_channels={args.n_channels} n_classes={n_classes}")
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

    # ---- Signed LIF 热启动：load EEGNet 4class ELU 权重 ----
    warm = False
    ws = Path(args.warm_start)
    if ws.exists():
        ck = torch.load(ws, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["state_dict"], strict=True)
        warm = True
        print(f"[热启动] 已加载 {ws.name} 的 conv/fc/BN 权重（ELU→signed LIF 激活位点无参数）")
    else:
        print(f"[警告] 未找到热启动权重 {ws}，signed LIF 从随机初始化训练")

    model = model.to(device)
    print(f"参数量   : {model.num_params():,}")

    # ---- 对照：热启动未微调，直接在 eval 上的准确率 ----
    if warm:
        with torch.no_grad():
            acc0, fr0 = _evaluate(model, eval_loader, device)
            print(f"\n[对照] EEGNet 权重直接换 signed LIF 激活（不微调）eval 准确率 : {acc0*100:.2f}%")
            print(f"       净发放率 act1={fr0['act1_net']*100:+.2f}% (正{fr0['act1_pos']*100:.2f}%/负{fr0['act1_neg']*100:.2f}%) "
                  f"act2={fr0['act2_net']*100:+.2f}% (正{fr0['act2_pos']*100:.2f}%/负{fr0['act2_neg']*100:.2f}%)")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    use_cosine = args.cosine and not args.no_cosine
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
                 if use_cosine else None)
    history = []
    best_acc, best_epoch, best_state, patience_counter = 0.0, -1, None, 0
    print(f"\n开始训练 signed LIF ...")
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
    print(f"\n[final test] signed LIF (cross-session, E 会话, 只测一次) : {acc*100:.2f}%")
    print(f"       净发放率 act1={fr['act1_net']*100:+.2f}% (正{fr['act1_pos']*100:.2f}%/负{fr['act1_neg']*100:.2f}%) "
          f"act2={fr['act2_net']*100:+.2f}% (正{fr['act2_pos']*100:.2f}%/负{fr['act2_neg']*100:.2f}%)")

    # ---- 保存 ----
    out = Path(args.out) if args.out else ALGO_DIR / "signed_lif_4class.pth"
    model_args = dict(n_channels=args.n_channels, n_classes=n_classes, input_length=500,
                      F1=8, D=2, F2=16, kernel_length=63, dropout=args.dropout,
                      activation="signed_lif")
    torch.save({"state_dict": best_state, "epoch": best_epoch, "val_acc": best_acc,
                "final_test": acc, "model_args": model_args,
                "lif_hparams": dict(threshold=args.threshold, tau=args.tau,
                                    timesteps=args.timesteps, alpha=args.alpha)}, out)
    print(f"\n✓ 权重已保存：{out}（epoch {best_epoch}，val_acc={best_acc*100:.2f}%）")

    # ---- 报告 ----
    tag = args.tag or "signed"
    report = ALGO_DIR / f"signed_lif_4class_report_{tag}.txt"
    chance = 100.0 / n_classes
    lines = [
        f"BCIC 2a 22ch 四分类训练报告 — Signed LIF-EEGNet（正负脉冲分离）",
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
        f"patience={args.patience}, seed={args.seed}, val_split={args.val_split}, "
        f"dropout={args.dropout}, cosine={use_cosine}",
        f"LIF 超参        : threshold={args.threshold}, tau={args.tau}, "
        f"timesteps={args.timesteps}, alpha={args.alpha}",
        f"热启动          : {warm}（{Path(args.warm_start).name}）",
        f"净发放率        : act1={fr['act1_net']*100:+.2f}% (正{fr['act1_pos']*100:.2f}%/负{fr['act1_neg']*100:.2f}%) "
        f"act2={fr['act2_net']*100:+.2f}% (正{fr['act2_pos']*100:.2f}%/负{fr['act2_neg']*100:.2f}%)",
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
