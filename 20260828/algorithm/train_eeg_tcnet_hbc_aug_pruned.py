#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_eeg_tcnet_hbc_aug_pruned.py — 通道剪枝版 LIF-EEG-TCNet + HBC 增强训练

train_eeg_tcnet_hbc_aug.py 的拷贝改造（原文件只读不动），唯一差异：
  - 新增 --channels：保留的通道索引（EEG_22 顺序，0 基），数据/模型按此剪枝
  - 模型 n_channels = len(channels)，depthwise 核 (len(channels), 1)
  - 热启动：载入 relu brief 权重后，把 depthwise.weight 行切片到保留通道
    （其余 conv/BN/TCN/fc 与通道数无关，strict=True 零回归）
  - 其余协议与基线完全一致：hold-out 15% / patience 25 / cosine / dropout 0.25 /
    seed 42 / HBC 增强 SNR∈[10,20]dB / LIF thr=1.0 tau=2.0 T=8 / 热启动 relu

用法
----
  env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin \\
      /opt/miniconda3/envs/openmmlab/bin/python train_eeg_tcnet_hbc_aug_pruned.py \\
      --channels 7,8,9,10,11,13,14,15,16,17,1,2,3,4,5,6

产物（algorithm/）：
  eeg_tcnet_4class_lif_hbc_aug_pr{K}.pth         权重（含 channels 列表）
  eeg_tcnet_4class_report_lif_hbc_aug_pr{K}.txt  逐 epoch 报告
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# ---- 路径 ----
ALGO_DIR = Path(__file__).resolve().parent
PROJECT = Path("/Users/heyixuan/BCI/luckfox-deploy")
DATA_DIR = PROJECT / "data"
DATA_4CLASS = ALGO_DIR / "data_4class"

sys.path.insert(0, str(ALGO_DIR))
sys.path.insert(0, str(DATA_DIR))

from dataset import EEGDataset          # noqa: E402
from eeg_tcnet_lif import EEGTCNetLIF   # noqa: E402
from hbc_noise_advanced import hbc_noise, snr_db_to_sigma_ratio  # noqa: E402

EEG_22 = ["Fz", "FC3", "FC1", "FCz", "FC2", "FC4",
          "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
          "CP3", "CP1", "CPz", "CP2", "CP4",
          "P1", "Pz", "P2", "POz"]


class ChannelSliceDataset(Dataset):
    """把 EEGDataset 的 (1,22,T) 切片为 (1,K,T)，K=len(keep_idx)。"""

    def __init__(self, base: EEGDataset, keep_idx: list[int]):
        self.base = base
        self.keep_idx = keep_idx

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        x, y = self.base[idx]
        return x[:, self.keep_idx, :], y


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
    if not args.aug:
        return None
    return dict(enabled=True, snr_min_db=args.snr_min, snr_max_db=args.snr_max,
                qbits=5, qrange=4.0,
                burst_sigma_ratio=None if args.no_burst else args.burst_sigma_ratio,
                burst_len=args.burst_len,
                burst_prob=0.0 if args.no_burst else args.burst_prob)


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
        description="通道剪枝版 LIF-EEG-TCNet + HBC 增强训练（--channels 指定保留通道）")
    parser.add_argument("--channels", type=str, required=True,
                        help="保留通道索引（EEG_22 顺序 0 基，逗号分隔），如 7,8,9,10,11,13,14,15,16,17,1,2,3,4,5,6")
    parser.add_argument("--n-classes", type=int, default=4)
    parser.add_argument("--input-length", type=int, default=500)
    parser.add_argument("--F1", type=int, default=8)
    parser.add_argument("--D", type=int, default=2)
    parser.add_argument("--F2", type=int, default=16)
    parser.add_argument("--FT", type=int, default=12)
    parser.add_argument("--kernel-length", type=int, default=63)
    parser.add_argument("--tcn-kernel-size", type=int, default=12)
    parser.add_argument("--pool1", type=int, default=4)
    parser.add_argument("--pool2", type=int, default=8)
    parser.add_argument("--readout-mode", type=str, default="gap", choices=["last", "gap"])
    parser.add_argument("--threshold", type=float, default=1.0)
    parser.add_argument("--tau", type=float, default=2.0)
    parser.add_argument("--timesteps", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=4.0)
    parser.add_argument("--warm-start", type=str,
                        default=str(ALGO_DIR / "eeg_tcnet_4class_relu_brief.pth"))
    parser.add_argument("--no-warm", action="store_true")
    parser.add_argument("--aug", action="store_true")
    parser.add_argument("--snr-min", type=float, default=10.0)
    parser.add_argument("--snr-max", type=float, default=20.0)
    parser.add_argument("--burst-sigma-ratio", type=float, default=0.3)
    parser.add_argument("--burst-len", type=int, default=16)
    parser.add_argument("--burst-prob", type=float, default=0.005)
    parser.add_argument("--no-burst", action="store_true")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--out", type=str, default="")
    parser.add_argument("--tag", type=str, default="", help="产物后缀（默认 aug_pr{K}）")
    args = parser.parse_args()

    keep_idx = [int(s) for s in args.channels.split(",") if s.strip() != ""]
    assert len(keep_idx) == len(set(keep_idx)), f"通道索引重复：{keep_idx}"
    assert all(0 <= i < 22 for i in keep_idx), f"通道索引越界：{keep_idx}"
    assert len(keep_idx) < 22, "剪枝训练要求保留通道数 < 22（--channels 传保留子集）"
    keep_ch = [EEG_22[i] for i in keep_idx]
    n_channels = len(keep_idx)
    print(f"保留通道 ({n_channels}ch) : {list(zip(keep_idx, keep_ch))}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = _pick_device(args.no_cuda)

    train_npz = DATA_4CLASS / "train.npz"
    eval_npz = DATA_4CLASS / "eval.npz"
    if not (train_npz.exists() and eval_npz.exists()):
        raise SystemExit(f"✗ 数据集缺失：{DATA_4CLASS}")
    train_ds = ChannelSliceDataset(EEGDataset(train_npz), keep_idx)
    eval_ds = ChannelSliceDataset(EEGDataset(eval_npz), keep_idx)
    n_classes = args.n_classes

    aug = _make_aug_spec(args)
    tag = args.tag or (f"pr{n_channels}" + ("_aug" if aug is not None else "_noaug"))
    aug_desc = "无增强（对照）" if aug is None else \
        (f"HBC增强[5bit+AWGN σ/H=10^(-SNR/20), SNR∈[{args.snr_min},{args.snr_max}]dB"
         + (f" +burst σb/H={args.burst_sigma_ratio} len={args.burst_len} p={args.burst_prob}]"
            if not args.no_burst else "]（无突发）"))

    print(f"设备     : {device}")
    print(f"模型     : LIF-EEG-TCNet（brief 配置，剪枝 {n_channels}ch，"
          f"LIF thr={args.threshold} tau={args.tau} T={args.timesteps} alpha={args.alpha}）")
    print(f"训练增强 : {aug_desc}")
    print(f"训练集   : {len(train_ds)} 试次 | 评估集 : {len(eval_ds)} 试次")

    # ---- hold-out 划分（与基线同 seed，同数据集长度 → 同划分）----
    n_val = max(1, int(len(train_ds) * args.val_split))
    n_train = len(train_ds) - n_val
    g = torch.Generator().manual_seed(args.seed)
    train_ds, val_ds = torch.utils.data.random_split(train_ds, [n_train, n_val], generator=g)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False)
    print(f"hold-out : {n_train} 训练 / {n_val} 验证（早停用） | final test = eval（{len(eval_ds)} 试次）")

    # ---- 构造剪枝 LIF 模型 ----
    lif_kwargs = dict(threshold=args.threshold, tau=args.tau,
                      timesteps=args.timesteps, alpha=args.alpha)
    model = EEGTCNetLIF(n_channels=n_channels, n_classes=n_classes,
                        input_length=args.input_length, F1=args.F1, D=args.D,
                        F2=args.F2, FT=args.FT, kernel_length=args.kernel_length,
                        tcn_kernel_size=args.tcn_kernel_size,
                        pool1=args.pool1, pool2=args.pool2,
                        readout_mode=args.readout_mode,
                        dropout=args.dropout, activation="lif", **lif_kwargs)

    # ---- 热启动：relu brief 权重 + depthwise 行切片到保留通道 ----
    warm = False
    ws = Path(args.warm_start)
    if ws.exists() and not args.no_warm:
        ck = torch.load(ws, map_location="cpu", weights_only=False)
        sd = ck["state_dict"]
        assert sd["depthwise.weight"].shape[2] == 22, \
            f"热启动权重非 22ch：{sd['depthwise.weight'].shape}"
        sd = dict(sd)
        sd["depthwise.weight"] = sd["depthwise.weight"][:, :, keep_idx, :].contiguous()
        missing, unexpected = model.load_state_dict(sd, strict=True)
        assert not missing and not unexpected, f"热启动键不匹配: {missing} / {unexpected}"
        warm = True
        print(f"[热启动] {ws.name} 载入并切片 depthwise 到 {n_channels}ch（strict=True ✓）")
    else:
        print("[警告] 未热启动，从随机初始化训练（对照用）")

    model = model.to(device)
    n_params = model.num_params()
    print(f"参数量   : {n_params:,}")

    if warm:
        with torch.no_grad():
            acc0, fr0 = _evaluate(model, eval_loader, device)
            print(f"\n[对照] relu(剪枝) 权重直接换 LIF 激活（不微调）eval 准确率 : {acc0*100:.2f}%")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    history = []
    best_acc, best_epoch, best_state, patience_counter = 0.0, -1, None, 0
    print(f"\n开始训练 LIF-EEG-TCNet 剪枝版（{aug_desc}）...")
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
              f"train_acc={train_acc*100:5.2f}% val_acc={val_acc*100:5.2f}% | {dt:.1f}s", flush=True)
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

    acc, fr = _evaluate(model, eval_loader, device)
    print(f"\n[final test] LIF-EEG-TCNet 剪枝版（{aug_desc}）eval 准确率 : {acc*100:.2f}%")

    out = Path(args.out) if args.out else ALGO_DIR / f"eeg_tcnet_4class_lif_hbc_{tag}.pth"
    model_args = dict(n_channels=n_channels, n_classes=n_classes,
                      input_length=args.input_length, F1=args.F1, D=args.D,
                      F2=args.F2, FT=args.FT, kernel_length=args.kernel_length,
                      tcn_kernel_size=args.tcn_kernel_size,
                      pool1=args.pool1, pool2=args.pool2,
                      readout_mode=args.readout_mode,
                      dropout=args.dropout, activation="lif")
    torch.save({"state_dict": best_state, "epoch": best_epoch, "val_acc": best_acc,
                "final_test": acc, "model_args": model_args, "n_params": n_params,
                "channels": keep_idx, "channel_names": keep_ch,
                "lif_hparams": lif_kwargs, "warm_start": warm,
                "aug_config": None if aug is None else {k: v for k, v in aug.items()},
                "aug_desc": aug_desc, "firing_rates": fr}, out)
    print(f"\n✓ 权重已保存：{out}（epoch {best_epoch}，val_acc={best_acc*100:.2f}%，"
          f"final_test={acc*100:.2f}%）")

    report = ALGO_DIR / f"eeg_tcnet_4class_report_lif_hbc_{tag}.txt"
    lines = [
        f"BCIC 2a 四分类训练报告 — LIF-EEG-TCNet 通道剪枝版（{n_channels}ch）+ HBC 噪声增强",
        "=" * 64,
        f"tag             : {tag}",
        f"保留通道        : {list(zip(keep_idx, keep_ch))}（EEG_22 顺序索引）",
        f"训练增强        : {aug_desc}",
        f"最佳验证准确率  : {best_acc*100:.2f}%（epoch {best_epoch}，hold-out 早停用）",
        f"final test(eval): {acc*100:.2f}%（cross-session，只测一次）",
        f"chance level    : {100.0/n_classes:.1f}%",
        f"n_channels      : {n_channels}（基线 22） | n_classes : {n_classes}",
        f"训练样本数      : {n_train} | 验证 {n_val} | final test {len(eval_ds)}",
        f"参数量          : {n_params:,}（基线 9,512）",
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
        f"热启动          : {warm}（{Path(args.warm_start).name}，depthwise 行切片）",
        f"对照(未微调)    : " + (f"{acc0*100:.2f}%（eval，relu 剪枝权重直接换 LIF）"
                                if warm else "N/A"),
        f"发放率(eval)    : " + "  ".join(
            f"{k.split('.')[-1]}={fr[k]*100:.1f}%" for k in sorted(fr)),
        "噪声模型        : hbc_noise_advanced.hbc_noise（5bit 量化 + AWGN + 突发）",
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
