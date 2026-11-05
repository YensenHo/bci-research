#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_within_subject.py — within-subject 对标：每人专用 LIF-EEG-TCNet 训练

场景
----
「专用设备」模式：出厂/交付前给用户采集训练专用模型（每人自己的 T 会话训练、
自己的 E 会话测试），回答产品化精度上限：开箱即用（cross-session 63.58%）
vs 每人专用模型。

协议与 train_eeg_tcnet_hbc_aug.py（cross-session 63.58% 基线）**完全一致**：
  - 结构：EEG-TCNet brief 配置（F1=8/D=2/F2=16/FT=12/KE=63/KT=12/pool4/8/GAP），
    9,512 参数；
  - 激活：LIF（thr=1.0 / tau=2.0 / T=8 / alpha=4.0，surrogate 训练），
    热启动 relu brief 权重（strict=True）；
  - HBC 增强：每 batch 随机 SNR ∈ [10,20] dB（5bit 量化 + AWGN + 突发
    σb/H=0.3 len=16 p=0.005）；
  - 优化：Adam lr=1e-3, batch=16, CrossEntropyLoss, cosine 退火 T_max=150,
    epochs=150, dropout=0.25, patience=25, hold-out 15%（seed=42，每人
    288×0.15=43 验证 / 245 训练）；
  - final test：**该被试自己的 E 会话**（cross-session 测的是 9 人混合 eval，
    这里严格一对一，泄漏红线不变：T 训练、E 测试，绝不相混）。

数据
----
data_4class_by_subject/A{nn}/{train,eval}.npz（split_by_subject.py 产出，
每人每会话 288 试次 = 4 类 × 72，与 data_4class 逐位一致）。

产物（within_subject/）
----
  weights/A{nn}_lif_hbc.pth        每人专用权重（state_dict + 训练信息）
  results_lif_hbc.json            全部被试精度（唯一事实来源）
  reports/A{nn}_lif_hbc.txt       逐 epoch 报告

用法
----
  # 全部 9 人（LIF + HBC 增强，seed 42）
  env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin \\
      /opt/miniconda3/envs/openmmlab/bin/python train_within_subject.py
  # smoke：1 人 3 epoch（验证管线，勿用于结论）
  ... python train_within_subject.py --subjects 1 --epochs 3 --tag smoke
  # relu 主体（LIF 太慢时的替代）：--activation relu
  # 抽查/重复：--seed 123 --subjects 3 --tag s123
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

ALGO_DIR = Path(__file__).resolve().parent
PROJECT = Path("/Users/heyixuan/BCI/luckfox-deploy")
DATA_DIR = PROJECT / "data"
SUBJ_DATA = ALGO_DIR / "data_4class_by_subject"
OUT_DIR = ALGO_DIR / "within_subject"

sys.path.insert(0, str(ALGO_DIR))
sys.path.insert(0, str(DATA_DIR))

from dataset import EEGDataset          # noqa: E402
from eeg_tcnet_lif import EEGTCNetLIF   # noqa: E402
from hbc_noise_advanced import hbc_noise, snr_db_to_sigma_ratio  # noqa: E402

WARM_START = ALGO_DIR / "eeg_tcnet_4class_relu_brief.pth"


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


def _apply_aug(xb: torch.Tensor, aug: dict) -> torch.Tensor:
    snr_db = aug["snr_min_db"] + (aug["snr_max_db"] - aug["snr_min_db"]) * float(torch.rand(1).item())
    sigma_ratio = snr_db_to_sigma_ratio(snr_db)
    kw = dict(qbits=aug["qbits"], qrange=aug["qrange"], sigma_ratio=sigma_ratio)
    if aug["burst_sigma_ratio"] is not None:
        kw.update(burst_sigma_ratio=aug["burst_sigma_ratio"],
                  burst_len=aug["burst_len"], burst_prob=aug["burst_prob"])
    return hbc_noise(xb, **kw)


def _make_aug_spec(args) -> dict | None:
    if not args.aug:
        return None
    return dict(enabled=True, snr_min_db=args.snr_min, snr_max_db=args.snr_max,
                qbits=5, qrange=4.0,
                burst_sigma_ratio=None if args.no_burst else args.burst_sigma_ratio,
                burst_len=args.burst_len,
                burst_prob=0.0 if args.no_burst else args.burst_prob)


def train_one(subj: int, args, device) -> dict:
    subj_dir = SUBJ_DATA / f"A{subj:02d}"
    train_npz, eval_npz = subj_dir / "train.npz", subj_dir / "eval.npz"
    if not (train_npz.exists() and eval_npz.exists()):
        raise SystemExit(f"✗ 缺少 {subj_dir} 下 train/eval.npz（先跑 split_by_subject.py）")

    train_ds = EEGDataset(train_npz)
    eval_ds = EEGDataset(eval_npz)
    assert len(train_ds) == 288 and len(eval_ds) == 288, "每人每会话应为 288 试次"

    aug = _make_aug_spec(args)

    # hold-out：与 cross-session 协议一致（15% 验证，seed=42）
    n_val = max(1, int(len(train_ds) * args.val_split))
    n_train = len(train_ds) - n_val
    g = torch.Generator().manual_seed(args.seed)
    train_ds, val_ds = torch.utils.data.random_split(train_ds, [n_train, n_val], generator=g)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False)

    lif_kwargs = dict(threshold=args.threshold, tau=args.tau,
                      timesteps=args.timesteps, alpha=args.alpha)
    model = EEGTCNetLIF(n_channels=args.n_channels, n_classes=args.n_classes,
                        input_length=args.input_length, F1=args.F1, D=args.D,
                        F2=args.F2, FT=args.FT, kernel_length=args.kernel_length,
                        tcn_kernel_size=args.tcn_kernel_size,
                        pool1=args.pool1, pool2=args.pool2,
                        readout_mode=args.readout_mode,
                        dropout=args.dropout, activation=args.activation, **lif_kwargs)

    warm = False
    ws = Path(args.warm_start)
    if ws.exists() and not args.no_warm:
        ck = torch.load(ws, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["state_dict"], strict=True)
        warm = True

    model = model.to(device)
    n_params = model.num_params()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc, best_epoch, best_state, patience_counter = 0.0, -1, None, 0
    history = []
    t_start = time.time()
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
                        "train_acc": round(train_acc, 4), "val_acc": round(val_acc, 4),
                        "sec": round(dt, 1)})
        print(f"  A{subj:02d} Epoch {epoch:3d}/{args.epochs} | train_loss={train_loss:.4f} "
              f"train_acc={train_acc*100:5.2f}% val_acc={val_acc*100:5.2f}% | {dt:.1f}s")
        if val_acc > best_acc:
            best_acc, best_epoch = val_acc, epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"  A{subj:02d} 早停：验证集连续 {args.patience} epoch 未提升。")
                break

    elapsed = time.time() - t_start
    if best_state is None:
        raise SystemExit("✗ 训练未产生有效模型")
    model.load_state_dict(best_state)
    model.eval()

    # final test：该被试自己的 E 会话（专用设备模式，只测一次）
    acc, fr = _evaluate(model, eval_loader, device)

    tag = args.tag or f"lif_hbc"
    out = OUT_DIR / "weights" / f"A{subj:02d}_{tag}.pth"
    out.parent.mkdir(parents=True, exist_ok=True)
    model_args = dict(n_channels=args.n_channels, n_classes=args.n_classes,
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
                "firing_rates": fr, "subject": f"A{subj:02d}",
                "train_n": n_train, "val_n": n_val, "test_n": len(eval_ds),
                "seed": args.seed, "elapsed_sec": round(elapsed, 1)}, out)

    # 逐 epoch 报告
    (OUT_DIR / "reports").mkdir(parents=True, exist_ok=True)
    rep = OUT_DIR / "reports" / f"A{subj:02d}_{tag}.txt"
    lines = [
        f"within-subject 专用模型训练报告 — A{subj:02d}（{args.activation} + HBC 增强）",
        "=" * 60,
        f"final test（本人 E 会话）: {acc*100:.2f}%  （288 试次，只测一次）",
        f"最佳验证准确率        : {best_acc*100:.2f}%（epoch {best_epoch}）",
        f"训练 {n_train} / 验证 {n_val}（hold-out 15%, seed={args.seed}）",
        f"训练时长              : {elapsed:.1f}s（{len(history)} epoch）",
        f"热启动                : {warm}（{Path(args.warm_start).name}）",
        f"LIF 超参              : thr={args.threshold} tau={args.tau} T={args.timesteps}",
        f"发放率(eval)          : " + "  ".join(
            f"{k.split('.')[-1]}={fr[k]*100:.1f}%" for k in sorted(fr)),
        "-" * 60, "逐 epoch (epoch, train_loss, train_acc, val_acc, sec):",
    ]
    for h in history:
        lines.append(f"  {h['epoch']:3d}  {h['train_loss']:.6f}  "
                     f"{h['train_acc']*100:5.2f}%  {h['val_acc']*100:5.2f}%  {h['sec']:.1f}s")
    rep.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"\n  ✓ A{subj:02d} final test（本人 E 会话）= {acc*100:.2f}% | "
          f"best val {best_acc*100:.2f}% @ ep{best_epoch} | {elapsed:.1f}s | 权重 {out.name}")
    return dict(subject=f"A{subj:02d}", acc=round(acc * 100, 2),
                best_val=round(best_acc * 100, 2), best_epoch=best_epoch,
                n_train=n_train, n_val=n_val, test_n=len(eval_ds),
                seed=args.seed, activation=args.activation,
                elapsed_sec=round(elapsed, 1), n_epochs=len(history),
                firing_rates={k: round(v * 100, 1) for k, v in sorted(fr.items())})


def main() -> int:
    parser = argparse.ArgumentParser(description="within-subject 专用模型训练（协议同 cross-session）")
    parser.add_argument("--subjects", type=int, nargs="+", default=list(range(1, 10)))
    parser.add_argument("--activation", type=str, default="lif", choices=["lif", "relu"])
    parser.add_argument("--tag", type=str, default="", help="产物后缀（默认 lif_hbc / relu_hbc）")
    parser.add_argument("--n-channels", type=int, default=22)
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
    parser.add_argument("--warm-start", type=str, default=str(WARM_START))
    parser.add_argument("--no-warm", action="store_true")
    parser.add_argument("--aug", action="store_true", default=True)
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
    parser.add_argument("--out-json", type=str, default="")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = _pick_device(args.no_cuda)
    print(f"设备: {device} | 激活: {args.activation} | seed: {args.seed} | "
          f"HBC 增强 SNR∈[{args.snr_min},{args.snr_max}]dB | 被试: {args.subjects}")

    tag = args.tag or (f"{args.activation}_hbc")
    results = []
    for subj in sorted(set(args.subjects)):
        results.append(train_one(subj, args, device))

    out_json = Path(args.out_json) if args.out_json else OUT_DIR / f"results_{tag}.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n✓ 结果已写入：{out_json}")
    for r in results:
        print(f"  {r['subject']}: {r['acc']:.2f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
