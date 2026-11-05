#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_eeg_tcnet_hbc_aug_2b.py — LIF-EEG-TCNet 3ch 二分类训练（BCIC IV 2b 跨数据集验证）

与 2a 的 train_eeg_tcnet_hbc_aug.py **协议完全一致**（保证跨数据集可比）：
  - hold-out 早停：train 划 15% 验证（seed 固定），patience=25
  - final test = eval（held-out session，只测一次）
  - Adam lr=0.001，batch=16，CrossEntropyLoss，cosine 退火，dropout=0.25
  - seed 可配（默认 42 / 123 双 seed），epochs=150
  - 结构 = 2a brief 最优配置：KE=63/KT=12/pool4/8/GAP
  - LIF 超参 = 2a 最优：thr=1.0 / tau=2.0 / T=8（surrogate，热启动 relu 权重）
  - HBC 增强保留：每 batch SNR ∈ [10,20] dB 随机（5bit 量化 + AWGN + 突发）

与 2a 的差异（跨数据集适配，诚实标注）：
  - 数据：bci2b/data/（train.npz = 官方训练集 session 1-3 池化，
    eval.npz = 官方测试集 session 4-5 池化——Graz 源 E 文件内含真实标签）
  - 通道：3（C3/Cz/C4），类别：2（左手/右手），chance=50%
  - 热启动权重：本数据集自己的 relu 版（bci2b/eeg_tcnet_2b_relu_s42.pth），
    不是 2a 的 22ch/4class 权重（通道/类别不同无法迁移，属预期）
  - relu 基线训练用 --activation relu --no-warm（从随机初始化）

用法（全部 env -i 隔离）：
  # ① relu 基线 + 热启动源（seed 42，HBC 增强）
  ... train_eeg_tcnet_hbc_aug_2b.py --activation relu --no-warm --tag relu_s42 --seed 42
  # ② LIF 热启动微调（seed 42）
  ... train_eeg_tcnet_hbc_aug_2b.py --activation lif --tag lif_s42 --seed 42
  # ③ LIF 热启动微调（seed 123，双 seed 验证 MPS 波动）
  ... train_eeg_tcnet_hbc_aug_2b.py --activation lif --tag lif_s123 --seed 123

产物（bci2b/）：
  eegmmidb_4class_{activation}_hbc_{tag}.pth        权重
  eegmmidb_4class_report_{activation}_hbc_{tag}.txt 逐 epoch 报告
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
ALGO_DIR = Path("/Users/heyixuan/BCI/20260828/algorithm")   # eeg_tcnet_lif / hbc_noise_advanced
EEGMMIDB_DIR = Path(__file__).resolve().parent
DATA_DIR = EEGMMIDB_DIR / "data"
DATA_2A = Path("/Users/heyixuan/BCI/luckfox-deploy/data")   # dataset.py（只读引用）

sys.path.insert(0, str(ALGO_DIR))
sys.path.insert(0, str(EEGMMIDB_DIR))
sys.path.insert(0, str(DATA_2A))

from dataset import EEGDataset          # noqa: E402  (luckfox-deploy/data)
from eeg_tcnet_lif import EEGTCNetLIF   # noqa: E402  (algorithm/)
from hbc_noise_advanced import hbc_noise, snr_db_to_sigma_ratio  # noqa: E402


def _pick_device(no_cuda: bool):
    if no_cuda:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _evaluate(model, loader, device) -> tuple[float, dict]:
    """返回 (acc, 各 LIF 位点平均发放率 dict)。与 2a 脚本一致。"""
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


def _make_aug_spec(args) -> dict | None:
    if args.no_aug:
        return None
    return dict(
        enabled=True,
        snr_min_db=args.snr_min,
        snr_max_db=args.snr_max,
        qbits=5,
        qrange=4.0,
        burst_sigma_ratio=None if args.no_burst else args.burst_sigma_ratio,
        burst_len=args.burst_len,
        burst_prob=0.0 if args.no_burst else args.burst_prob,
    )


def _apply_aug(xb: torch.Tensor, aug: dict) -> torch.Tensor:
    snr_db = aug["snr_min_db"] + (aug["snr_max_db"] - aug["snr_min_db"]) * float(torch.rand(1).item())
    sigma_ratio = snr_db_to_sigma_ratio(snr_db)
    kw = dict(qbits=aug["qbits"], qrange=aug["qrange"], sigma_ratio=sigma_ratio)
    if aug["burst_sigma_ratio"] is not None:
        kw.update(burst_sigma_ratio=aug["burst_sigma_ratio"],
                  burst_len=aug["burst_len"], burst_prob=aug["burst_prob"])
    return hbc_noise(xb, **kw)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="EEGMMIDB 21ch 四分类 MI 训练（LIF-EEG-TCNet + HBC 噪声增强，30 人大数据集跨被试）")
    parser.add_argument("--n-channels", type=int, default=21)
    parser.add_argument("--n-classes", type=int, default=4)
    parser.add_argument("--input-length", type=int, default=320)
    # ---- 结构（与 2a brief 最优一致）----
    parser.add_argument("--F1", type=int, default=8)
    parser.add_argument("--D", type=int, default=2)
    parser.add_argument("--F2", type=int, default=16)
    parser.add_argument("--FT", type=int, default=12)
    parser.add_argument("--kernel-length", type=int, default=63)
    parser.add_argument("--tcn-kernel-size", type=int, default=12)
    parser.add_argument("--pool1", type=int, default=4)
    parser.add_argument("--pool2", type=int, default=8)
    parser.add_argument("--readout-mode", type=str, default="gap", choices=["last", "gap"])
    # ---- 激活与 LIF 超参 ----
    parser.add_argument("--activation", type=str, default="lif", choices=["lif", "relu", "elu"])
    parser.add_argument("--threshold", type=float, default=1.0)
    parser.add_argument("--tau", type=float, default=2.0)
    parser.add_argument("--timesteps", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=4.0)
    parser.add_argument("--warm-start", type=str,
                        default=str(EEGMMIDB_DIR / "eegmmidb_4class_relu_s42.pth"),
                        help="热启动权重（2b 自己的 relu 版，strict=True）")
    parser.add_argument("--no-warm", action="store_true", help="从随机初始化训练（对照）")
    # ---- HBC 噪声增强（与 2a 协议一致，默认开，--no-aug 可关）----
    parser.add_argument("--no-aug", action="store_true", help="关闭 HBC 噪声增强（对照）")
    parser.add_argument("--snr-min", type=float, default=10.0)
    parser.add_argument("--snr-max", type=float, default=20.0)
    parser.add_argument("--burst-sigma-ratio", type=float, default=0.3)
    parser.add_argument("--burst-len", type=int, default=16)
    parser.add_argument("--burst-prob", type=float, default=0.005)
    parser.add_argument("--no-burst", action="store_true")
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
    parser.add_argument("--tag", type=str, default="")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = _pick_device(args.no_cuda)

    train_npz = DATA_DIR / "train.npz"
    eval_npz = DATA_DIR / "eval.npz"
    if not (train_npz.exists() and eval_npz.exists()):
        raise SystemExit(f"✗ 数据集缺失：{DATA_DIR}（先跑 preprocess_2b.py）")
    train_ds = EEGDataset(train_npz)
    eval_ds = EEGDataset(eval_npz)
    n_classes = args.n_classes

    aug = None if args.no_aug else _make_aug_spec(args)
    tag = args.tag or (f"{args.activation}_s{args.seed}")
    aug_desc = "无增强（对照）" if aug is None else \
        (f"HBC增强[5bit+AWGN σ/H=10^(-SNR/20), SNR∈[{args.snr_min},{args.snr_max}]dB"
         + (f" +burst σb/H={args.burst_sigma_ratio} len={args.burst_len} p={args.burst_prob}]"
            if not args.no_burst else "]（无突发）"))

    print(f"设备     : {device}")
    print(f"模型     : {args.activation.upper()}-EEG-TCNet（brief 配置，"
          + (f"LIF thr={args.threshold} tau={args.tau} T={args.timesteps} alpha={args.alpha}"
             if args.activation == "lif" else "无 LIF 超参"))
    print(f"训练增强 : {aug_desc}")
    print(f"训练集   : {len(train_ds)} 试次 | 评估集 : {len(eval_ds)} 试次 | "
          f"n_channels={args.n_channels} n_classes={n_classes}（chance=50%）")

    # ---- hold-out 划分：早停验证集与 final test（eval）分离 ----
    n_val = max(1, int(len(train_ds) * args.val_split))
    n_train = len(train_ds) - n_val
    g = torch.Generator().manual_seed(args.seed)
    train_ds, val_ds = torch.utils.data.random_split(train_ds, [n_train, n_val], generator=g)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False)
    print(f"hold-out : {n_train} 训练 / {n_val} 验证（早停用） | final test = eval（{len(eval_ds)} 试次）")

    # ---- 构造模型 ----
    lif_kwargs = dict(threshold=args.threshold, tau=args.tau,
                      timesteps=args.timesteps, alpha=args.alpha)
    model = EEGTCNetLIF(n_channels=args.n_channels, n_classes=n_classes,
                        input_length=args.input_length, F1=args.F1, D=args.D,
                        F2=args.F2, FT=args.FT, kernel_length=args.kernel_length,
                        tcn_kernel_size=args.tcn_kernel_size,
                        pool1=args.pool1, pool2=args.pool2,
                        readout_mode=args.readout_mode,
                        dropout=args.dropout, activation=args.activation, **lif_kwargs)

    # ---- 热启动 relu 版权重（LIF 层无参数，strict=True 键零回归）----
    warm = False
    ws = Path(args.warm_start)
    if ws.exists() and not args.no_warm:
        ck = torch.load(ws, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["state_dict"], strict=True)
        warm = True
        print(f"[热启动] 已加载 {ws.name} 的 conv/BN/fc 权重（relu→LIF 位点无参数，strict=True ✓）")
    else:
        print("[警告] 未热启动，从随机初始化训练（relu 基线 / 对照用）")

    model = model.to(device)
    n_params = model.num_params()
    print(f"参数量   : {n_params:,}")

    # ---- 对照：热启动未微调，直接换 LIF 激活在 eval 上的表现 ----
    if warm:
        with torch.no_grad():
            acc0, fr0 = _evaluate(model, eval_loader, device)
            print(f"\n[对照] relu 权重直接换 LIF 激活（不微调）eval 准确率 : {acc0*100:.2f}%")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    history = []
    best_acc, best_epoch, best_state, patience_counter = 0.0, -1, None, 0
    print(f"\n开始训练 {args.activation.upper()}-EEG-TCNet（{aug_desc}）...")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        total_loss, total_correct, total_n = 0.0, 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            if aug is not None:
                xb = _apply_aug(xb, aug)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
            total_correct += (logits.argmax(1) == yb).sum().item()
            total_n += xb.size(0)
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

    # ---- final test：eval（held-out session，只测一次）----
    acc, fr = _evaluate(model, eval_loader, device)
    print(f"\n[final test] {args.activation.upper()}-EEG-TCNet（{aug_desc}）eval 准确率 : {acc*100:.2f}%")

    # ---- 保存 ----
    out = Path(args.out) if args.out else EEGMMIDB_DIR / f"eegmmidb_4class_{args.activation}_hbc_{tag}.pth"
    model_args = dict(n_channels=args.n_channels, n_classes=n_classes,
                      input_length=args.input_length, F1=args.F1, D=args.D,
                      F2=args.F2, FT=args.FT, kernel_length=args.kernel_length,
                      tcn_kernel_size=args.tcn_kernel_size,
                      pool1=args.pool1, pool2=args.pool2,
                      readout_mode=args.readout_mode,
                      dropout=args.dropout, activation=args.activation)
    torch.save({"state_dict": best_state, "epoch": best_epoch, "val_acc": best_acc,
                "final_test": acc, "model_args": model_args, "n_params": n_params,
                "lif_hparams": lif_kwargs, "warm_start": warm,
                "aug_config": None if aug is None else {k: v for k, v in aug.items()},
                "aug_desc": aug_desc, "firing_rates": fr}, out)
    print(f"\n✓ 权重已保存：{out}（epoch {best_epoch}，val_acc={best_acc*100:.2f}%，"
          f"final_test={acc*100:.2f}%）")

    # ---- 报告 ----
    report = EEGMMIDB_DIR / f"eegmmidb_4class_report_{args.activation}_hbc_{tag}.txt"
    lines = [
        f"BCIC 2b 3ch 二分类训练报告 — {args.activation.upper()}-EEG-TCNet + HBC 噪声增强（跨数据集验证）",
        "=" * 64,
        f"tag             : {tag}",
        f"训练增强        : {aug_desc}",
        f"最佳验证准确率  : {best_acc*100:.2f}%（epoch {best_epoch}，hold-out 早停用）",
        f"final test(eval): {acc*100:.2f}%（held-out session，只测一次）",
        f"chance level    : 50.0%",
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
        f"发放率(eval)    : " + "  ".join(
            f"{k.split('.')[-1]}={fr[k]*100:.1f}%" for k in sorted(fr)),
        "噪声模型        : hbc_noise_advanced.hbc_noise（5bit 量化 + AWGN + 突发，"
        "与 luckfox-deploy/model/noise.py 严格一致）",
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
