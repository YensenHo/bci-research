#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
continue_delta_tcnet.py — 诊断用：从已训 delta 权重继续微调（低 LR，更长训练）

目的（诊断「39% 是信息损失还是训练动力学」）
--------------------------------------------
run1（delta1bit_k0.5）早停于 epoch 23（best val 43.04%，train≈val=46%，无过拟合），
说明模型**欠拟合**。两种可能：(a) delta 脉冲输入的信息瓶颈本身封顶；
(b) 稀疏输入 → conv1 梯度稀疏高方差 + relu 热启动权重对脉冲输入是错误先验 +
   早停太早（lr 仍高），优化没走完。

本脚本从已训 checkpoint 继续训练：低 LR（默认 3e-4）+ 更长训练（默认 100 epoch
+ patience 40），其余协议完全不变（HBC 增强、hold-out 划分、aug 同 seed）。
若 val/test 显著上升 → 训练动力学是主因；若仍 ≈43% → 信息瓶颈是硬顶。

用法
----
  env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin \\
      /opt/miniconda3/envs/openmmlab/bin/python continue_delta_tcnet.py \\
      --resume eeg_tcnet_4class_delta_delta1bit_k0.5.pth --lr 3e-4 \\
      --epochs 100 --patience 40 --tag cont_k0.5_lr3e4
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

ALGO_DIR = Path(__file__).resolve().parent
PROJECT = Path("/Users/heyixuan/BCI/luckfox-deploy")
DATA_DIR = PROJECT / "data"
DATA_4CLASS = ALGO_DIR / "data_4class"

sys.path.insert(0, str(ALGO_DIR))
sys.path.insert(0, str(DATA_DIR))

from dataset import EEGDataset            # noqa: E402
from delta_tcnet import DeltaTCNet        # noqa: E402
from hbc_noise_advanced import hbc_noise, snr_db_to_sigma_ratio  # noqa: E402


def _pick_device(no_cuda: bool):
    if no_cuda:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _evaluate(model, loader, device) -> tuple[float, dict, float]:
    model.eval()
    correct, total = 0, 0
    fr_sum: dict = {}
    dens_sum = 0.0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            correct += (logits.argmax(1) == yb).sum().item()
            total += xb.size(0)
            for name, fr in model.firing_rates.items():
                fr_sum[name] = fr_sum.get(name, 0.0) + fr * xb.size(0)
            dens_sum += model.input_density * xb.size(0)
    fr = {k: v / total for k, v in fr_sum.items()} if total else {}
    return correct / total, fr, dens_sum / total if total else 0.0


def main() -> int:
    ap = argparse.ArgumentParser(description="从已训 delta 权重继续微调（诊断训练动力学）")
    ap.add_argument("--resume", type=str, required=True, help="已训 delta checkpoint 路径")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--patience", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-cuda", action="store_true")
    ap.add_argument("--tag", type=str, default="cont", help="产物后缀")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = _pick_device(args.no_cuda)

    train_npz, eval_npz = DATA_4CLASS / "train.npz", DATA_4CLASS / "eval.npz"
    train_ds, eval_ds = EEGDataset(train_npz), EEGDataset(eval_npz)
    n_val = max(1, int(len(train_ds) * 0.15))
    n_train = len(train_ds) - n_val
    g = torch.Generator().manual_seed(args.seed)
    train_ds, val_ds = torch.utils.data.random_split(train_ds, [n_train, n_val], generator=g)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False)

    ck = torch.load(Path(args.resume), map_location="cpu", weights_only=False)
    dc = ck["delta_config"]
    model = DeltaTCNet(**ck["model_args"], **ck.get("lif_hparams", {}),
                       input_channels=dc["input_channels"], delta_k=dc["delta_k"],
                       delta_bits=dc["delta_bits"], delta_mode=dc["delta_mode"],
                       theta=ck.get("theta"))
    model.load_state_dict(ck["state_dict"], strict=True)
    model = model.to(device)
    print(f"设备     : {device} | 继续微调 {Path(args.resume).name}")
    print(f"起始状态 : final_test={ck.get('final_test', -1)*100:.2f}%  "
          f"val_acc={ck.get('val_acc', -1)*100:.2f}%（epoch {ck.get('epoch', -1)}）")
    print(f"微调配置 : lr={args.lr}, epochs={args.epochs}, patience={args.patience}, "
          f"aug=保留（SNR∈[10,20]dB HBC）")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    best_acc, best_epoch, best_state, patience_counter = 0.0, -1, None, 0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        total_loss, total_correct, total_n = 0.0, 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            snr_db = 10.0 + 10.0 * float(torch.rand(1).item())
            xb = hbc_noise(xb, qbits=5, qrange=4.0,
                           sigma_ratio=snr_db_to_sigma_ratio(snr_db),
                           burst_sigma_ratio=0.3, burst_len=16, burst_prob=0.005)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
            total_correct += (logits.argmax(1) == yb).sum().item()
            total_n += xb.size(0)
        scheduler.step()
        val_acc, _, _ = _evaluate(model, val_loader, device)
        print(f"  Epoch {epoch:3d}/{args.epochs} | train_loss={total_loss/total_n:.4f} "
              f"train_acc={total_correct/total_n*100:5.2f}% val_acc={val_acc*100:5.2f}% | "
              f"{time.time()-t0:.1f}s")
        if val_acc > best_acc:
            best_acc, best_epoch = val_acc, epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"早停：连续 {args.patience} epoch 未提升（epoch {epoch}）。")
                break
    if best_state is None:
        raise SystemExit("✗ 未产生有效模型")
    model.load_state_dict(best_state)
    model.eval()
    acc, fr, dens = _evaluate(model, eval_loader, device)
    print(f"\n[继续微调后] eval 准确率 : {acc*100:.2f}%（原 {ck.get('final_test', -1)*100:.2f}%）"
          f" | 输入密度 {dens*100:.2f}% | best val {best_acc*100:.2f}%@epoch {best_epoch}")
    out = ALGO_DIR / f"eeg_tcnet_4class_delta_{args.tag}.pth"
    torch.save({"state_dict": best_state, "epoch": best_epoch, "val_acc": best_acc,
                "final_test": acc, "model_args": ck["model_args"],
                "lif_hparams": ck.get("lif_hparams", {}),
                "delta_config": dc, "theta": ck.get("theta"),
                "aug_config": ck.get("aug_config"),
                "aug_desc": ck.get("aug_desc", "") + " [继续微调: lr=%g epochs=%d]" % (args.lr, args.epochs),
                "firing_rates": fr, "input_density": dens,
                "resumed_from": str(Path(args.resume).name)}, out)
    print(f"✓ 已保存：{out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
