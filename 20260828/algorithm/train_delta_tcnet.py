#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_delta_tcnet.py — 全脉冲 LIF-EEG-TCNet（delta 输入脉冲编码）训练

目的
----
验证「输入脉冲编码（delta 调制）+ 全脉冲网络」的精度代价与稀疏收益：
  - 输入换成 delta 脉冲序列（θ=k·std，k 扫 0.5/1.0；可选 1bit/2bit；signed/posneg），
    网络从输入到输出全脉冲（输入脉冲 → conv1 → LIF → … → 发放率 → fc）；
  - 与基线 LIF-EEG-TCNet + HBC 增强（clean 63.58% @aug_s42 / 8dB 54.67%）同协议对比；
  - 报告输入脉冲密度（稀疏度目标 <30%）与前端 φ1 事件驱动潜力（评估脚本算 MACs）。

训练协议与 train_eeg_tcnet_hbc_aug.py **完全一致**（保证可比）：
  - hold-out 15% 验证（seed=42）/ patience=25 / cosine / dropout=0.25 / 热启动 relu brief
  - HBC 增强保留：每 batch SNR∈[10,20]dB 随机（5bit 量化 + AWGN + 突发 σb/H=0.3）
  - Adam lr=0.001, batch=16, epochs=150, seed=42；final test = eval（只测一次）

delta 编码细节
-------------
  - θ = k·std_通道，std 在**训练集**上逐通道统计（train.npz 全量），train/eval 共用，
    不随 batch 自适应、不随噪声漂移；θ 随权重保存（buffer），保证可复现。
  - 1bit signed : 脉冲 ∈ {-1,0,1}，conv1 单输入通道（与基线架构完全一致，strict 热启动）
  - 1bit posneg : 正/负双 0/1 脉冲通道，conv1 双输入通道（热启动 [w,-w] 拆分，等价 signed）
  - 2bit signed : 脉冲 ∈ {-2,-1,0,1,2}（幅度 2bit + 符号），conv1 单输入通道

用法
----
  env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin \\
      /opt/miniconda3/envs/openmmlab/bin/python train_delta_tcnet.py \\
      --delta-k 0.5 --delta-bits 1 --delta-mode signed --tag delta1bit_k0.5
  # posneg 双通道：--delta-mode posneg（自动 input_channels=2）
  # 2bit：--delta-bits 2（仅 signed）

产物（均在 algorithm/）
  eeg_tcnet_4class_delta_{tag}.pth         权重（含 theta/编码配置/训练信息）
  eeg_tcnet_4class_report_delta_{tag}.txt  逐 epoch 报告（含输入密度/发放率）
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

sys.path.insert(0, str(ALGO_DIR))   # from delta_tcnet import DeltaTCNet
sys.path.insert(0, str(DATA_DIR))   # from dataset import EEGDataset

from dataset import EEGDataset            # noqa: E402
from delta_encode import compute_theta    # noqa: E402
from delta_tcnet import DeltaTCNet, load_warm_start  # noqa: E402
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
    """返回 (acc, 各 LIF 位点平均发放率 dict, 输入脉冲密度)。与基线脚本一致。"""
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
    dens = dens_sum / total if total else 0.0
    return correct / total, fr, dens


def _make_aug_spec(args) -> dict | None:
    """构造训练期增强配置（与基线完全一致）。"""
    if not args.aug:
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
    """按增强配置给一个 batch 施加 HBC 噪声（每 batch 随机 SNR），同基线。"""
    snr_db = aug["snr_min_db"] + (aug["snr_max_db"] - aug["snr_min_db"]) * float(torch.rand(1).item())
    sigma_ratio = snr_db_to_sigma_ratio(snr_db)
    kw = dict(qbits=aug["qbits"], qrange=aug["qrange"], sigma_ratio=sigma_ratio)
    if aug["burst_sigma_ratio"] is not None:
        kw.update(burst_sigma_ratio=aug["burst_sigma_ratio"],
                  burst_len=aug["burst_len"], burst_prob=aug["burst_prob"])
    return hbc_noise(xb, **kw)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="全脉冲 LIF-EEG-TCNet（delta 输入脉冲编码）+ HBC 增强训练（surrogate，热启动 relu）")
    parser.add_argument("--n-channels", type=int, default=22)
    parser.add_argument("--n-classes", type=int, default=4)
    parser.add_argument("--input-length", type=int, default=500)
    # ---- 结构（默认 = 简报最优 brief 配置，与基线完全一致）----
    parser.add_argument("--F1", type=int, default=8)
    parser.add_argument("--D", type=int, default=2)
    parser.add_argument("--F2", type=int, default=16)
    parser.add_argument("--FT", type=int, default=12)
    parser.add_argument("--kernel-length", type=int, default=63)
    parser.add_argument("--tcn-kernel-size", type=int, default=12)
    parser.add_argument("--pool1", type=int, default=4)
    parser.add_argument("--pool2", type=int, default=8)
    parser.add_argument("--readout-mode", type=str, default="gap", choices=["last", "gap"])
    # ---- delta 输入脉冲编码（核心新增）----
    parser.add_argument("--delta-k", type=float, default=0.5,
                        help="阈值系数 θ=k·std_通道（训练集逐通道 std）")
    parser.add_argument("--delta-bits", type=int, default=1, choices=[1, 2],
                        help="1 = 1bit delta（±1）；2 = 2bit delta（{-2..2}，仅 signed）")
    parser.add_argument("--delta-mode", type=str, default="signed", choices=["signed", "posneg"],
                        help="signed=单通道带符号脉冲；posneg=正负双 0/1 脉冲通道")
    # ---- LIF 超参（默认 = 既有最优 thr=1.0/tau=2.0/T=8）----
    parser.add_argument("--threshold", type=float, default=1.0)
    parser.add_argument("--tau", type=float, default=2.0)
    parser.add_argument("--timesteps", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=4.0)
    parser.add_argument("--warm-start", type=str,
                        default=str(ALGO_DIR / "eeg_tcnet_4class_relu_brief.pth"),
                        help="热启动权重（relu brief；posneg 自动 [w,-w] 拆分 conv1）")
    parser.add_argument("--no-warm", action="store_true", help="从随机初始化训练（对照）")
    # ---- HBC 噪声增强（与基线一致）----
    parser.add_argument("--aug", action="store_true",
                        help="训练时对输入施加 HBC 噪声增强（5bit 量化 + AWGN + 突发）")
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
    parser.add_argument("--tag", type=str, default="",
                        help="产物后缀（默认 delta{bits}bit_k{k}[_posneg]）")
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

    # ---- delta 阈值：训练集逐通道 k·std（train/eval 共用，不随 batch/噪声漂移）----
    theta = compute_theta(train_ds.X, args.delta_k)          # (C,) float32
    input_channels = 2 if args.delta_mode == "posneg" else 1

    aug = _make_aug_spec(args)
    tag = args.tag or (f"delta{args.delta_bits}bit_k{args.delta_k:g}"
                       + ("_posneg" if args.delta_mode == "posneg" else ""))
    aug_desc = "无增强（对照）" if aug is None else \
        (f"HBC增强[5bit+AWGN σ/H=10^(-SNR/20), SNR∈[{args.snr_min},{args.snr_max}]dB"
         + (f" +burst σb/H={args.burst_sigma_ratio} len={args.burst_len} p={args.burst_prob}]"
            if not args.no_burst else "]（无突发）"))
    enc_desc = (f"delta{args.delta_bits}bit θ=k·std k={args.delta_k:g} mode={args.delta_mode}"
                f"（逐通道 θ ∈ [{theta.min():.3f}, {theta.max():.3f}]）")

    print(f"设备     : {device}")
    print(f"模型     : 全脉冲 DeltaTCNet（brief 配置，LIF thr={args.threshold} "
          f"tau={args.tau} T={args.timesteps} alpha={args.alpha}）")
    print(f"输入编码 : {enc_desc}（input_channels={input_channels}）")
    print(f"训练增强 : {aug_desc}")
    print(f"训练集   : {len(train_ds)} 试次 | 评估集 : {len(eval_ds)} 试次 | "
          f"n_channels={args.n_channels} n_classes={n_classes}")

    # ---- hold-out 划分：早停验证集与 final test（eval）分离（同基线）----
    n_val = max(1, int(len(train_ds) * args.val_split))
    n_train = len(train_ds) - n_val
    g = torch.Generator().manual_seed(args.seed)
    train_ds, val_ds = torch.utils.data.random_split(train_ds, [n_train, n_val], generator=g)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False)
    print(f"hold-out : {n_train} 训练 / {n_val} 验证（早停用） | final test = eval（{len(eval_ds)} 试次）")

    # ---- 构造全脉冲模型（delta 编码在模型内部 forward，输入到输出全脉冲）----
    lif_kwargs = dict(threshold=args.threshold, tau=args.tau,
                      timesteps=args.timesteps, alpha=args.alpha)
    model = DeltaTCNet(n_channels=args.n_channels, n_classes=n_classes,
                       input_length=args.input_length, F1=args.F1, D=args.D,
                       F2=args.F2, FT=args.FT, kernel_length=args.kernel_length,
                       tcn_kernel_size=args.tcn_kernel_size,
                       pool1=args.pool1, pool2=args.pool2,
                       readout_mode=args.readout_mode,
                       dropout=args.dropout, activation="lif",
                       input_channels=input_channels,
                       delta_k=args.delta_k, delta_bits=args.delta_bits,
                       delta_mode=args.delta_mode, theta=theta,
                       **lif_kwargs)

    # ---- 热启动 relu 版权重（LIF/编码层无参数；posneg 自动拆分 conv1）----
    warm = False
    ws = Path(args.warm_start)
    if ws.exists() and not args.no_warm:
        ck = torch.load(ws, map_location="cpu", weights_only=False)
        missing = load_warm_start(model, ck)
        warm = True
        print(f"[热启动] 已加载 {ws.name} 的 conv/BN/fc 权重（missing={missing}，"
              f"posneg conv1 按 [w,-w] 拆分）")
    else:
        print("[警告] 未热启动，从随机初始化训练（对照用）")

    model = model.to(device)
    n_params = model.num_params()
    print(f"参数量   : {n_params:,}（基线 LIF 版 9,512；posneg 双通道 +504）")

    # ---- 对照：热启动未微调，直接换 delta 输入 + LIF 激活在 eval 上的表现 ----
    if warm:
        with torch.no_grad():
            acc0, fr0, dens0 = _evaluate(model, eval_loader, device)
            print(f"\n[对照] relu 权重 + delta 输入直接评估（不微调）eval 准确率 : {acc0*100:.2f}%"
                  f"（输入密度 {dens0*100:.1f}%）")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    history = []
    best_acc, best_epoch, best_state, patience_counter = 0.0, -1, None, 0
    print(f"\n开始训练全脉冲 DeltaTCNet（{enc_desc}，{aug_desc}）...")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        total_loss, total_correct, total_n = 0.0, 0, 0
        dens_train = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            if aug is not None:
                xb = _apply_aug(xb, aug)          # HBC 增强保留（同基线协议）
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
            total_correct += (logits.argmax(1) == yb).sum().item()
            total_n += xb.size(0)
            dens_train += model.input_density * xb.size(0)
        scheduler.step()                          # cosine 退火（协议固定）
        train_loss = total_loss / total_n
        train_acc = total_correct / total_n
        val_acc, _, _ = _evaluate(model, val_loader, device)
        dt = time.time() - t0
        history.append({"epoch": epoch, "train_loss": round(train_loss, 6),
                        "train_acc": round(train_acc, 4), "val_acc": round(val_acc, 4),
                        "input_density": round(dens_train / total_n, 4)})
        print(f"  Epoch {epoch:3d}/{args.epochs} | train_loss={train_loss:.4f} "
              f"train_acc={train_acc*100:5.2f}% val_acc={val_acc*100:5.2f}% "
              f"输入密度={dens_train/total_n*100:4.1f}% | {dt:.1f}s")
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

    # ---- final test：eval（cross-session，只测一次）+ 输入稀疏度 ----
    acc, fr, dens = _evaluate(model, eval_loader, device)
    print(f"\n[final test] 全脉冲 DeltaTCNet（{enc_desc}，{aug_desc}）eval 准确率 : {acc*100:.2f}%"
          f" | 输入脉冲密度 : {dens*100:.2f}%")

    # ---- 保存 ----
    out = Path(args.out) if args.out else ALGO_DIR / f"eeg_tcnet_4class_delta_{tag}.pth"
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
                "delta_config": dict(delta_k=args.delta_k, delta_bits=args.delta_bits,
                                     delta_mode=args.delta_mode,
                                     input_channels=input_channels),
                "theta": theta,
                "aug_config": None if aug is None else {k: v for k, v in aug.items()},
                "aug_desc": aug_desc, "firing_rates": fr,
                "input_density": dens}, out)
    print(f"\n✓ 权重已保存：{out}（epoch {best_epoch}，val_acc={best_acc*100:.2f}%，"
          f"final_test={acc*100:.2f}%，输入密度={dens*100:.2f}%）")

    # ---- 报告 ----
    report = ALGO_DIR / f"eeg_tcnet_4class_report_delta_{tag}.txt"
    chance = 100.0 / n_classes
    lines = [
        f"BCIC 2a 22ch 四分类训练报告 — 全脉冲 DeltaTCNet（delta 输入脉冲编码 + HBC 增强，surrogate，热启动 relu）",
        "=" * 72,
        f"tag             : {tag}",
        f"输入编码        : {enc_desc}",
        f"训练增强        : {aug_desc}",
        f"最佳验证准确率  : {best_acc*100:.2f}%（epoch {best_epoch}，hold-out 早停用）",
        f"final test(eval): {acc*100:.2f}%（cross-session，只测一次）",
        f"输入脉冲密度    : {dens*100:.2f}%（稀疏目标 <30%；k=0.5 1bit 训练集 11.32%）",
        f"chance level    : {chance:.1f}%",
        f"n_channels      : {args.n_channels} | n_classes : {n_classes} | input_channels : {input_channels}",
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
        f"对照(未微调)    : " + (f"{acc0*100:.2f}%（eval，relu 权重 + delta 输入直接评估，"
                                f"输入密度 {dens0*100:.1f}%）" if warm else "N/A（无热启动）"),
        f"发放率(eval)    : " + "  ".join(
            f"{k.split('.')[-1]}={fr[k]*100:.1f}%" for k in sorted(fr)),
        "噪声模型        : hbc_noise_advanced.hbc_noise（5bit 量化 + AWGN + 突发，"
        "与 luckfox-deploy/model/noise.py 严格一致）",
        "编码层          : delta_encode.DeltaEncode（无参数；θ=k·std 逐通道，训练集统计，"
        "随权重保存；conv1 输入为脉冲 → 前端 φ1 事件驱动可稀疏化）",
        "-" * 72, "逐 epoch 记录 (epoch, train_loss, train_acc, val_acc, 输入密度):",
    ]
    for h in history:
        lines.append(f"  {h['epoch']:3d}  {h['train_loss']:.6f}  "
                     f"{h['train_acc']*100:5.2f}%  {h['val_acc']*100:5.2f}%  "
                     f"{h['input_density']*100:5.1f}%")
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  报告已写入：{report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
