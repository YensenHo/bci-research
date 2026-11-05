#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_sleep_lif.py — Sleep-EDF 睡眠分期（30s epoch 5 分类）LIF-EEG-TCNet 训练

协议（与 2a/2b/CHB-MIT 一致，保证跨任务可比）
--------------------------------------------
  - hold-out 早停：train 划 15% 验证（seed 可控），patience=25
  - final test = eval（跨被试留出 subjects 4-5，只测一次）
  - Adam lr=0.001, batch=16, cosine 退火, dropout=0.25
  - 损失：加权 CrossEntropy（类别权重 = train 逆频率，meta.json，N1 少）
  - 结构 = brief 最优配置（KE=63/KT=12/pool4/8/GAP），n_channels=1, n_classes=5,
    T=3000（30s @100Hz 原样，不做降采样——前端池化后 TCN 步长 93，规模可接受）
  - 无 HBC 增强：睡眠分期无 HBC 植入场景，干净训练（同 CHB-MIT，如实记录）

组别
----
  1) relu s42（基线）
  2) LIF s42（热启动 relu 权重，thr=1.0/tau=2.0/T=8/alpha=4.0）
  3) LIF s123（同上，双 seed 覆盖 MPS ±1.7pp 波动）

用法
----
  env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin \
      /opt/miniconda3/envs/openmmlab/bin/python train_sleep_lif.py \
      --activation relu --seed 42 --tag relu_s42
  ... --activation lif --seed 42  --warm sleep_5class_relu_s42.pth --tag lif_s42
  ... --activation lif --seed 123 --warm sleep_5class_relu_s42.pth --tag lif_s123
  （--smoke 先跑 2 epoch 验证链路）
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
from torch.utils.data import DataLoader, Dataset

ALGO_DIR = Path("/Users/heyixuan/BCI/20260828/algorithm")
BASE = Path(__file__).resolve().parent
DATA = BASE / "data"

sys.path.insert(0, str(ALGO_DIR))
from eeg_tcnet_lif import EEGTCNetLIF  # noqa: E402

CLASS_NAMES = ["W", "N1", "N2", "N3", "REM"]


class NpzDS(Dataset):
    def __init__(self, npz: Path):
        with np.load(npz) as d:
            self.X = np.asarray(d["X"], np.float32)
            self.y = np.asarray(d["y"], np.int64)
        assert not np.isnan(self.X).any(), f"{npz.name}: X 含 NaN"

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return torch.from_numpy(self.X[i]), torch.tensor(int(self.y[i]), dtype=torch.long)


def _pick_device(no_cuda: bool):
    if no_cuda:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _evaluate(model, loader, device) -> tuple[float, dict]:
    model.eval()
    correct = total = 0
    fr_sum: dict = {}
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            correct += (logits.argmax(1) == yb).sum().item()
            total += xb.size(0)
            for name, fr in model.firing_rates.items():
                fr_sum[name] = fr_sum.get(name, 0.0) + fr * xb.size(0)
    fr = {k: v / total for k, v in fr_sum.items()} if total else {}
    return correct / total, fr


def main() -> int:
    ap = argparse.ArgumentParser(description="Sleep-EDF 睡眠分期 LIF-EEG-TCNet 训练")
    ap.add_argument("--activation", choices=["relu", "lif"], default="relu")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--val-split", type=float, default=0.15)
    ap.add_argument("--dropout", type=float, default=0.25)
    ap.add_argument("--threshold", type=float, default=1.0)
    ap.add_argument("--tau", type=float, default=2.0)
    ap.add_argument("--timesteps", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=4.0)
    ap.add_argument("--warm", type=str, default="", help="relu 权重 pth（LIF 热启动）")
    ap.add_argument("--tag", type=str, required=True)
    ap.add_argument("--data-dir", type=str, default=str(DATA))
    ap.add_argument("--smoke", action="store_true", help="2 epoch 冒烟")
    ap.add_argument("--no-cuda", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = _pick_device(args.no_cuda)

    train_npz, eval_npz = Path(args.data_dir) / "train.npz", Path(args.data_dir) / "eval.npz"
    meta_p = Path(args.data_dir) / "meta.json"
    if not (train_npz.exists() and eval_npz.exists()):
        raise SystemExit(f"✗ 缺数据集 {DATA}（先跑 preprocess_sleep.py）")
    meta = json.loads(meta_p.read_text(encoding="utf-8"))
    train_ds, eval_ds = NpzDS(train_npz), NpzDS(eval_npz)

    n_val = max(1, int(len(train_ds) * args.val_split))
    n_train = len(train_ds) - n_val
    g = torch.Generator().manual_seed(args.seed)
    train_ds, val_ds = torch.utils.data.random_split(train_ds, [n_train, n_val], generator=g)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False)

    dist = meta["train_dist"]
    cw = torch.tensor([meta["class_weights"][str(i)] for i in range(5)], dtype=torch.float32)
    print(f"设备     : {device} | 激活 : {args.activation} | seed : {args.seed}")
    print(f"训练集   : {len(train_ds)} epoch（hold-out {n_train}/{n_val}）| eval : {len(eval_ds)}")
    print(f"数据形状 : X={train_ds.dataset.X.shape}（n,1,1,3000 = 30s@100Hz 单通道 Fpz-Cz）")
    print(f"类别分布(train): " + ", ".join(f"{CLASS_NAMES[int(k)]}:{v}" for k, v in sorted(dist.items())))
    print(f"加权损失 : 权重 {cw.numpy().round(3).tolist()}（逆频率归一化）| chance 20%")

    lif_kwargs = dict(threshold=args.threshold, tau=args.tau,
                      timesteps=args.timesteps, alpha=args.alpha)
    model = EEGTCNetLIF(n_channels=1, n_classes=5, input_length=3000,
                        activation=args.activation, **lif_kwargs)
    if args.activation == "lif" and args.warm:
        ck = torch.load(Path(args.warm), map_location="cpu", weights_only=False)
        model.load_state_dict(ck["state_dict"], strict=True)
        print(f"[热启动] {Path(args.warm).name} strict=True ✓（relu→LIF 权重复用）")
    elif args.activation == "lif":
        print("[警告] LIF 未热启动（从随机初始化训练）")
    model = model.to(device)
    print(f"参数量   : {model.num_params():,} | MACs/epoch(30s): {model.macs_estimate()/1e6:.2f} M")

    if args.activation == "lif":
        with torch.no_grad():
            acc0, fr0 = _evaluate(model, eval_loader, device)
            print(f"[对照] relu 权重直接换 LIF（未微调）eval : {acc0*100:.2f}%")

    epochs = 2 if args.smoke else args.epochs
    criterion = nn.CrossEntropyLoss(weight=cw.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    history = []
    best_acc, best_epoch, best_state, patience = 0.0, -1, None, 0
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        tl = tc = tn = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            tl += loss.item() * xb.size(0)
            tc += (logits.argmax(1) == yb).sum().item()
            tn += xb.size(0)
        scheduler.step()
        val_acc, _ = _evaluate(model, val_loader, device)
        dt = time.time() - t0
        history.append(dict(epoch=epoch, train_loss=tl / tn,
                            train_acc=tc / tn, val_acc=val_acc))
        print(f"  Epoch {epoch:3d}/{epochs} | loss={tl/tn:.4f} train={tc/tn*100:5.2f}% "
              f"val={val_acc*100:5.2f}% | {dt:.1f}s")
        if val_acc > best_acc:
            best_acc, best_epoch = val_acc, epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= args.patience and not args.smoke:
                print(f"早停：验证集连续 {args.patience} epoch 未提升。")
                break
    if best_state is None:
        raise SystemExit("✗ 训练未产生有效模型")
    model.load_state_dict(best_state)

    acc, fr = _evaluate(model, eval_loader, device)
    print(f"\n[final test] eval（跨被试 subjects 4-5，只测一次）: {acc*100:.2f}%  (chance 20%)")
    if args.activation == "lif":
        fr_str = "  ".join(f"{k.split('.')[-1]}={fr[k]*100:.1f}%" for k in sorted(fr))
        print(f"  发放率 : {fr_str}")

    out = BASE / f"sleep_5class_{args.tag}.pth"
    torch.save({"state_dict": best_state, "epoch": best_epoch, "val_acc": best_acc,
                "final_test": acc, "activation": args.activation,
                "n_params": model.num_params(), "seed": args.seed,
                "lif_hparams": lif_kwargs, "warm_start": bool(args.warm),
                "firing_rates": fr, "class_names": CLASS_NAMES}, out)

    report = BASE / f"sleep_5class_report_{args.tag}.txt"
    lines = [
        "Sleep-EDF 睡眠分期（30s epoch 5 分类）训练报告 — EEG-TCNet",
        "=" * 64,
        f"tag             : {args.tag}",
        f"激活            : {args.activation}（LIF: thr={args.threshold} tau={args.tau} "
        f"T={args.timesteps} alpha={args.alpha}）",
        f"最佳验证准确率  : {best_acc*100:.2f}%（epoch {best_epoch}，hold-out 早停）",
        f"final test(eval): {acc*100:.2f}%（跨被试 subjects 4-5，chance 20%）",
        f"训练 epoch 数   : {n_train} | 验证 {n_val} | eval {len(eval_ds)}",
        f"类别分布(train) : " + ", ".join(f"{CLASS_NAMES[int(k)]}:{v}" for k, v in sorted(dist.items())),
        f"加权损失权重    : {cw.numpy().round(3).tolist()}",
        f"参数量          : {model.num_params():,} | MACs/epoch {model.macs_estimate()/1e6:.2f} M",
        f"设备            : {device} | seed {args.seed}",
        f"超参数          : lr={args.lr}, batch={args.batch_size}, epochs={epochs}, "
        f"patience={args.patience}, val_split={args.val_split}, dropout={args.dropout}, cosine",
        "结构            : brief（F1=8 D=2 F2=16 FT=12 KE=63 KT=12 pool4/8 GAP, "
        "1ch Fpz-Cz 30s 窗 T=3000）",
        f"热启动          : {args.warm}",
        f"对照(未微调)    : " + (f"{acc0*100:.2f}%" if args.activation == "lif" and args.warm
                                else "N/A"),
        f"发放率(eval)    : " + (fr_str if args.activation == "lif" else "N/A"),
        "注              : 无 HBC 增强（睡眠分期无 HBC 场景，干净训练）；单 epoch 独立分类（无序列上下文）",
        "-" * 64, "逐 epoch (epoch, train_loss, train_acc, val_acc):",
    ]
    for h in history:
        lines.append(f"  {h['epoch']:3d}  {h['train_loss']:.6f}  "
                     f"{h['train_acc']*100:5.2f}%  {h['val_acc']*100:5.2f}%")
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"✓ {out.name} + {report.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
