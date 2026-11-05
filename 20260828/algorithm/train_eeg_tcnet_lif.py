#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_eeg_tcnet_lif.py — LIF 化 EEG-TCNet 四分类训练（surrogate gradient，热启动 relu 权重）

目的
----
  验证「类脑化（LIF 率编码 + surrogate 训练）」在 EEG-TCNet 架构上的精度代价：
  目标是对齐 relu 版 EEG-TCNet 61.61%（eeg_tcnet_4class_relu_brief.pth），
  为板载部署（RV1106，relu/lif 转换链）提供 LIF 版权重。延续
  「类脑计算 + BCI + HBC」纲领：LIF-EEGNet 在 22ch 四分类下 56.87%（EEGNet 架构，
  类脑化代价 -1.4pp vs ELU 58.29%），本实验看更强的 TCN 架构能否把代价压回 0。

做法（与既有 LIF 流程一致）
--------------------------
  - 模型：eeg_tcnet_lif.EEGTCNetLIF（= eeg_tcnet.py 拷贝改造，仅激活可配 LIF +
    发放率诊断；原 eeg_tcnet.py 只读不动）。
  - 热启动：strict=True 加载 relu brief 权重（conv/BN/fc 全部复用；LIF 激活层
    无参数，state_dict 键零回归）。LIF 位点：前端 act1/act2 + TCN 每残差块
    conv_a/conv_b/残差输出 3 处（共 8 个位点）。
  - thr/tau/T 可配（--threshold/--tau/--timesteps），多组扫描找最优。
  - 发放率诊断：train/eval 上统计 8 个 LIF 位点的平均发放率，监测塌缩
    （发放率→0 = 神经元死亡，梯度消失）或饱和（→1）。

训练协议（与 train_eeg_tcnet.py / train_4class.py 完全一致，保证可比）
--------------------------------------------------------------------
  - hold-out 早停：train 划 15% 验证（388 试次，seed=42），patience=25
  - final test = eval（cross-session，E 会话，只测一次，消除选择偏差）
  - Adam lr=0.001，batch=16，CrossEntropyLoss，cosine 退火，dropout=0.25
  - seed=42，epochs=150
  - 结构默认 = 简报最优（brief）配置：KE=63/KT=12/pool4/8/GAP（relu 61.61% 同款）

用法
----
  env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin \\
      /opt/miniconda3/envs/openmmlab/bin/python train_eeg_tcnet_lif.py \\
      --threshold 1.0 --tau 2.0 --timesteps 8 --tag t1.0_tau2.0_T8

产物（均在 algorithm/）
-----------------------
  eeg_tcnet_4class_lif_t{thr}_tau{tau}_T{T}.pth   权重（state_dict + 训练信息 + 发放率）
  eeg_tcnet_4class_report_lif_t{thr}_tau{tau}_T{T}.txt  逐 epoch 报告
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

sys.path.insert(0, str(ALGO_DIR))   # from eeg_tcnet_lif import EEGTCNetLIF
sys.path.insert(0, str(DATA_DIR))   # from dataset import EEGDataset

from dataset import EEGDataset          # noqa: E402
from eeg_tcnet_lif import EEGTCNetLIF   # noqa: E402


def _pick_device(no_cuda: bool):
    if no_cuda:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _evaluate(model, loader, device) -> tuple[float, dict]:
    """返回 (acc, 各 LIF 位点平均发放率 dict)。

    发放率 = 该位点 LIF 层输出张量的整体均值（rate-based 语义，∈[0,1]），
    按 batch 加权平均。用于诊断发放率塌缩（≈0）/ 饱和（≈1）。
    """
    model.eval()
    correct, total = 0, 0
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
    parser = argparse.ArgumentParser(
        description="BCIC 2a 22ch 四分类训练（LIF-EEG-TCNet，surrogate gradient，热启动 relu）")
    parser.add_argument("--n-channels", type=int, default=22)
    parser.add_argument("--n-classes", type=int, default=4)
    parser.add_argument("--input-length", type=int, default=500)
    # ---- 结构（默认 = 简报最优 brief 配置：KE=63/KT=12/pool4/8/GAP，relu 61.61% 同款）----
    parser.add_argument("--F1", type=int, default=8)
    parser.add_argument("--D", type=int, default=2)
    parser.add_argument("--F2", type=int, default=16)
    parser.add_argument("--FT", type=int, default=12)
    parser.add_argument("--kernel-length", type=int, default=63)
    parser.add_argument("--tcn-kernel-size", type=int, default=12)
    parser.add_argument("--pool1", type=int, default=4)
    parser.add_argument("--pool2", type=int, default=8)
    parser.add_argument("--readout-mode", type=str, default="gap", choices=["last", "gap"])
    # ---- LIF 超参（surrogate 扫描项）----
    parser.add_argument("--threshold", type=float, default=1.0, help="发放阈值 thr")
    parser.add_argument("--tau", type=float, default=2.0, help="膜时间常数（decay=exp(-1/tau)）")
    parser.add_argument("--timesteps", type=int, default=8, help="LIF 内部仿真时间步 T")
    parser.add_argument("--alpha", type=float, default=4.0, help="surrogate sigmoid 斜率")
    parser.add_argument("--warm-start", type=str,
                        default=str(ALGO_DIR / "eeg_tcnet_4class_relu_brief.pth"),
                        help="热启动权重（relu 版 brief，strict=True，conv/BN/fc 复用）")
    parser.add_argument("--no-warm", action="store_true", help="从随机初始化训练（对照）")
    # ---- 训练超参（协议固定）----
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--out", type=str, default="")
    parser.add_argument("--tag", type=str, default="", help="产物后缀，默认 lif_t{thr}_tau{tau}_T{T}")
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
    print(f"模型     : LIF-EEG-TCNet（brief 配置，LIF thr={args.threshold} "
          f"tau={args.tau} T={args.timesteps} alpha={args.alpha}）")
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

    # ---- 构造 LIF 模型 ----
    lif_kwargs = dict(threshold=args.threshold, tau=args.tau,
                      timesteps=args.timesteps, alpha=args.alpha)
    model = EEGTCNetLIF(n_channels=args.n_channels, n_classes=n_classes,
                        input_length=args.input_length, F1=args.F1, D=args.D,
                        F2=args.F2, FT=args.FT, kernel_length=args.kernel_length,
                        tcn_kernel_size=args.tcn_kernel_size,
                        pool1=args.pool1, pool2=args.pool2,
                        readout_mode=args.readout_mode,
                        dropout=args.dropout, activation="lif", **lif_kwargs)

    # ---- 热启动 relu 版权重（LIF 层无参数，strict=True 键零回归）----
    warm = False
    ws = Path(args.warm_start)
    if ws.exists() and not args.no_warm:
        ck = torch.load(ws, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["state_dict"], strict=True)
        warm = True
        print(f"[热启动] 已加载 {ws.name} 的 conv/BN/fc 权重"
              f"（relu→LIF 激活位点无参数，strict=True ✓）")
    else:
        print("[警告] 未热启动，LIF 从随机初始化训练（对照用）")

    model = model.to(device)
    n_params = model.num_params()
    print(f"参数量   : {n_params:,}")

    # ---- 对照：热启动未微调，直接换 LIF 激活在 eval 上的表现 + 发放率 ----
    if warm:
        with torch.no_grad():
            acc0, fr0 = _evaluate(model, eval_loader, device)
            print(f"\n[对照] relu 权重直接换 LIF 激活（不微调）eval 准确率 : {acc0*100:.2f}%")
            print("       发放率 : " + "  ".join(
                f"{k.split('.')[-1]}={fr0[k]*100:.1f}%" for k in sorted(fr0)))

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    history = []
    best_acc, best_epoch, best_state, patience_counter = 0.0, -1, None, 0
    print(f"\n开始训练 LIF-EEG-TCNet ...")
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

    # ---- final test：eval（cross-session，只测一次）+ 发放率诊断 ----
    acc, fr = _evaluate(model, eval_loader, device)
    print(f"\n[final test] LIF-EEG-TCNet (cross-session, E 会话, 只测一次) : {acc*100:.2f}%")
    fr_str = "  ".join(f"{k.split('.')[-1]}={fr[k]*100:.1f}%" for k in sorted(fr))
    print(f"       发放率 : {fr_str}")

    # ---- 保存 ----
    tag = args.tag or f"lif_t{args.threshold}_tau{args.tau}_T{args.timesteps}"
    out = Path(args.out) if args.out else ALGO_DIR / f"eeg_tcnet_4class_{tag}.pth"
    model_args = dict(n_channels=args.n_channels, n_classes=n_classes,
                      input_length=args.input_length, F1=args.F1, D=args.D,
                      F2=args.F2, FT=args.FT, kernel_length=args.kernel_length,
                      tcn_kernel_size=args.tcn_kernel_size,
                      pool1=args.pool1, pool2=args.pool2,
                      readout_mode=args.readout_mode,
                      dropout=args.dropout, activation="lif")
    torch.save({"state_dict": best_state, "epoch": best_epoch, "val_acc": best_acc,
                "final_test": acc, "model_args": model_args, "n_params": n_params,
                "lif_hparams": lif_kwargs, "warm_start": warm,
                "firing_rates": fr}, out)
    print(f"\n✓ 权重已保存：{out}（epoch {best_epoch}，val_acc={best_acc*100:.2f}%）")

    # ---- 报告 ----
    report = ALGO_DIR / f"eeg_tcnet_4class_report_{tag}.txt"
    chance = 100.0 / n_classes
    lines = [
        f"BCIC 2a 22ch 四分类训练报告 — LIF-EEG-TCNet（surrogate，热启动 relu）",
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
        f"LIF 超参        : threshold={args.threshold}, tau={args.tau}, "
        f"timesteps={args.timesteps}, alpha={args.alpha}",
        f"热启动          : {warm}（{Path(args.warm_start).name}）",
        f"对照(未微调)    : " + (f"{acc0*100:.2f}%（eval，relu 权重直接换 LIF）"
                                if warm else "N/A（无热启动）"),
        f"发放率(eval)    : {fr_str}",
        "LIF 位点        : act1/act2（前端 pool 前）+ TCN 每残差块 act_a/act_b/act_res "
        "（膨胀因果卷积后）",
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
