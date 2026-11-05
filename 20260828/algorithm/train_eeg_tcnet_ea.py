#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_eeg_tcnet_ea.py — 欧氏对齐（EA）叠加 EEG-TCNet / LIF-EEG-TCNet 四分类训练

目的
----
  在既有 cross-session 协议（train.npz = A{nn}T 会话，eval.npz = A{nn}E 会话）
  上叠加欧氏对齐（He & Wu 2019，实现见 euclidean_alignment.py）：
  用训练集（T 会话）算出参考协方差 R 与白化矩阵 R^(-1/2)，对训练 + 测试
  每个 trial 施加 X_align = R^(-1/2) @ X，再训练原模型。
  验证 EA 能否在以下基线上再提升：
    - EEG-TCNet ELU  : 64.31%（eeg_tcnet_4class_report_elu_brief.txt）
    - LIF-EEG-TCNet  : 62.42%（thr=1.0/tau=2.0/T=8，热启动 relu_brief）

  为公平起见，脚本同时支持 --ea none 重跑无 EA 基线（同协议、同 seed），
  保证「无 EA vs 有 EA」唯一变量 = 数据对齐。

训练协议（与 train_eeg_tcnet_lif.py / train_eeg_tcnet.py 完全一致，保证可比）
--------------------------------------------------------------------------
  - hold-out 早停：train 划 15% 验证（388 试次，seed=42），patience=25
  - final test = eval（cross-session，E 会话，只测一次，消除选择偏差）
  - Adam lr=0.001，batch=16，CrossEntropyLoss，cosine 退火，dropout=0.25
  - seed=42，epochs=150
  - 结构默认 = 简报最优（brief）配置：KE=63/KT=12/pool4/8/GAP（9,512 参数）

用法
----
  # ELU + EA（主实验 a）
  env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin \\
      /opt/miniconda3/envs/openmmlab/bin/python train_eeg_tcnet_ea.py \\
      --activation elu --ea train

  # LIF + EA（主实验 b，重点；热启动 relu_brief 权重，与基线同源）
  env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin \\
      /opt/miniconda3/envs/openmmlab/bin/python train_eeg_tcnet_ea.py \\
      --activation lif --ea train --threshold 1.0 --tau 2.0 --timesteps 8

  # 无 EA 对照（重跑基线，同协议）
  ... --activation elu --ea none
  ... --activation lif --ea none --threshold 1.0 --tau 2.0 --timesteps 8

产物（均在 algorithm/）
----------------------
  eeg_tcnet_4class_{tag}.pth            权重（state_dict + 训练信息）
  eeg_tcnet_4class_report_{tag}.txt     逐 epoch 报告
  tag 示例：elu_brief_ea / lif_t1.0_tau2.0_T8_ea / elu_brief_noea / ...
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ---- 路径 ----
ALGO_DIR = Path(__file__).resolve().parent
DATA_4CLASS = ALGO_DIR / "data_4class"

sys.path.insert(0, str(ALGO_DIR))   # from eeg_tcnet / eeg_tcnet_lif / euclidean_alignment

from euclidean_alignment import (      # noqa: E402
    align,
    alignment_self_check,
    compute_reference_covariance,
    compute_whitening_matrix,
)
from eeg_tcnet import EEGTCNet         # noqa: E402
from eeg_tcnet_lif import EEGTCNetLIF  # noqa: E402


def _pick_device(no_cuda: bool):
    if no_cuda:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _evaluate(model, loader, device) -> tuple[float, dict]:
    """返回 (acc, 各 LIF 位点平均发放率 dict)。非 LIF 激活时 fr 为空 dict。"""
    model.eval()
    correct, total = 0, 0
    fr_sum: dict = {}
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            correct += (logits.argmax(1) == yb).sum().item()
            total += xb.size(0)
            # 仅 LIF 模型有 firing_rates 属性；ELU 模型没有（getattr 兼容）
            for name, fr in getattr(model, "firing_rates", {}).items():
                fr_sum[name] = fr_sum.get(name, 0.0) + fr * xb.size(0)
    fr = {k: v / total for k, v in fr_sum.items()} if total else {}
    return correct / total, fr


def _load_and_align(train_npz: Path, eval_npz: Path, ea: str):
    """加载原始 npz，按 ea 模式做欧氏对齐，返回 (X_train, y_train, X_eval, y_eval, diag)。

    ea='train'：R 由训练集（T 会话）全量计算，训练+测试同一 R^(-1/2)（严格 cross-session，
                因为 train.npz 本身就是 T 会话、eval.npz 是 E 会话，二者等价）。
    ea='none' ：不做对齐（基线对照，保证唯一变量 = 对齐）。
    """
    with np.load(train_npz, allow_pickle=False) as d:
        X_train = np.asarray(d["X"], dtype=np.float32)   # (N,1,22,500)
        y_train = np.asarray(d["y"], dtype=np.int64)
    with np.load(eval_npz, allow_pickle=False) as d:
        X_eval = np.asarray(d["X"], dtype=np.float32)
        y_eval = np.asarray(d["y"], dtype=np.int64)

    diag = {"ea": ea}
    if ea == "train":
        Xt = X_train[:, 0]                               # (N,22,500)
        Xe = X_eval[:, 0]
        R = compute_reference_covariance(Xt)
        Rinv_sqrt = compute_whitening_matrix(R)
        X_train = align(Xt, Rinv_sqrt)[:, None, :, :]    # 恢复 (N,1,22,500)
        X_eval = align(Xe, Rinv_sqrt)[:, None, :, :]
        chk = alignment_self_check(X_train[:, 0])
        eigvals = np.linalg.eigvalsh(R)
        diag.update(
            R_cond=float(eigvals.max() / max(eigvals.min(), 1e-12)),
            align_frob_rel=chk["frob_rel"],
            align_max_abs=chk["max_abs_dev"],
        )
        print(f"[EA] R 由训练集（T 会话）{len(Xt)} 试次计算（无泄漏），"
              f"条件数 {diag['R_cond']:.1f}")
        print(f"[EA] 对齐自检：训练集平均协方差相对误差 "
              f"{chk['frob_rel']*100:.4f}%（应≈0），最大偏差 {chk['max_abs_dev']:.2e}")
    return X_train, y_train, X_eval, y_eval, diag


def main() -> int:
    parser = argparse.ArgumentParser(
        description="EEG-TCNet / LIF-EEG-TCNet 四分类训练（可叠加欧氏对齐 EA）")
    parser.add_argument("--activation", type=str, default="elu",
                        choices=["elu", "lif"], help="激活：elu=基线 / lif=类脑化")
    parser.add_argument("--ea", type=str, default="train", choices=["none", "train"],
                        help="train=用训练集算 R 并对齐（严格 cross-session）；"
                             "none=无 EA（基线对照）")
    parser.add_argument("--n-channels", type=int, default=22)
    parser.add_argument("--n-classes", type=int, default=4)
    parser.add_argument("--input-length", type=int, default=500)
    # ---- 结构（默认 = 简报最优 brief 配置：KE=63/KT=12/pool4/8/GAP）----
    parser.add_argument("--F1", type=int, default=8)
    parser.add_argument("--D", type=int, default=2)
    parser.add_argument("--F2", type=int, default=16)
    parser.add_argument("--FT", type=int, default=12)
    parser.add_argument("--kernel-length", type=int, default=63)
    parser.add_argument("--tcn-kernel-size", type=int, default=12)
    parser.add_argument("--pool1", type=int, default=4)
    parser.add_argument("--pool2", type=int, default=8)
    parser.add_argument("--readout-mode", type=str, default="gap",
                        choices=["last", "gap"])
    # ---- LIF 超参（仅 activation=lif 生效）----
    parser.add_argument("--threshold", type=float, default=1.0)
    parser.add_argument("--tau", type=float, default=2.0)
    parser.add_argument("--timesteps", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=4.0)
    parser.add_argument("--warm-start", type=str,
                        default=str(ALGO_DIR / "eeg_tcnet_4class_relu_brief.pth"),
                        help="LIF 热启动权重（relu 版 brief，与基线同源）")
    parser.add_argument("--no-warm", action="store_true", help="LIF 从随机初始化训练")
    # ---- 训练超参（协议固定，与基线一致）----
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

    train_npz = DATA_4CLASS / "train.npz"
    eval_npz = DATA_4CLASS / "eval.npz"
    if not (train_npz.exists() and eval_npz.exists()):
        raise SystemExit(f"✗ 数据集缺失：{DATA_4CLASS}（先跑 preprocess_4class.py）")

    # ---- 数据加载 + EA 对齐（R 只由训练集算，无泄漏）----
    X_train, y_train, X_eval, y_eval, ea_diag = _load_and_align(
        train_npz, eval_npz, args.ea)
    n_classes = args.n_classes
    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    eval_ds = TensorDataset(torch.from_numpy(X_eval), torch.from_numpy(y_eval))
    print(f"设备     : {device}")
    print(f"模型     : {'LIF-' if args.activation == 'lif' else ''}EEG-TCNet"
          f"（brief 配置，activation={args.activation}"
          + (f"，LIF thr={args.threshold} tau={args.tau} T={args.timesteps}"
             if args.activation == "lif" else "") + "）")
    print(f"EA 模式  : {args.ea}"
          + ("（R 由 T 会话训练集计算 → T/E 会话应用，严格 cross-session）"
             if args.ea == "train" else "（无对齐，基线对照）"))
    print(f"训练集   : {len(train_ds)} 试次 | 评估集 : {len(eval_ds)} 试次 | "
          f"n_channels={args.n_channels} n_classes={n_classes}")

    # ---- hold-out 划分：早停验证集与 final test（eval）分离 ----
    n_val = max(1, int(len(train_ds) * args.val_split))
    n_train = len(train_ds) - n_val
    g = torch.Generator().manual_seed(args.seed)
    train_ds, val_ds = torch.utils.data.random_split(train_ds, [n_train, n_val],
                                                     generator=g)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False)
    print(f"hold-out : {n_train} 训练 / {n_val} 验证（早停用） | "
          f"final test = eval（{len(eval_ds)} 试次，只测一次）")

    # ---- 构造模型（结构/协议与基线完全一致，唯一变量 = 数据对齐）----
    model_kwargs = dict(n_channels=args.n_channels, n_classes=n_classes,
                        input_length=args.input_length, F1=args.F1, D=args.D,
                        F2=args.F2, FT=args.FT, kernel_length=args.kernel_length,
                        tcn_kernel_size=args.tcn_kernel_size,
                        pool1=args.pool1, pool2=args.pool2,
                        readout_mode=args.readout_mode,
                        dropout=args.dropout)
    lif_kwargs = dict(threshold=args.threshold, tau=args.tau,
                      timesteps=args.timesteps, alpha=args.alpha)
    if args.activation == "lif":
        model = EEGTCNetLIF(**model_kwargs, activation="lif", **lif_kwargs)
    else:
        model = EEGTCNet(**model_kwargs, activation="elu")

    # ---- LIF 热启动 relu 版权重（与基线同源；ELU 从随机初始化，同基线）----
    warm = False
    if args.activation == "lif":
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

    # ---- 对照：热启动未微调，直接换 LIF 激活在 eval 上的表现 ----
    if warm:
        with torch.no_grad():
            acc0, fr0 = _evaluate(model, eval_loader, device)
            print(f"\n[对照] relu 权重直接换 LIF 激活（不微调）eval 准确率 : "
                  f"{acc0*100:.2f}%")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                           T_max=args.epochs)
    history = []
    best_acc, best_epoch, best_state, patience_counter = 0.0, -1, None, 0
    print(f"\n开始训练 {args.activation.upper()} + EA={args.ea} ...")
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
                        "train_acc": round(train_acc, 4),
                        "val_acc": round(val_acc, 4)})
        print(f"  Epoch {epoch:3d}/{args.epochs} | train_loss={train_loss:.4f} "
              f"train_acc={train_acc*100:5.2f}% val_acc={val_acc*100:5.2f}% "
              f"| {dt:.1f}s")
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
    print(f"\n[final test] {args.activation.upper()} + EA={args.ea} "
          f"(cross-session, E 会话, 只测一次) : {acc*100:.2f}%")
    if fr:
        fr_str = "  ".join(
            f"{k.split('.')[-1]}={fr[k]*100:.1f}%" for k in sorted(fr))
        print(f"       发放率 : {fr_str}")

    # ---- 保存 ----
    ea_suffix = {"none": "noea", "train": "ea"}[args.ea]
    tag = args.tag or (f"{args.activation}_brief_{ea_suffix}"
                       if args.activation == "elu"
                       else f"lif_t{args.threshold}_tau{args.tau}_"
                            f"T{args.timesteps}_{ea_suffix}")
    out = Path(args.out) if args.out else ALGO_DIR / f"eeg_tcnet_4class_{tag}.pth"
    model_args = dict(**model_kwargs, activation=args.activation)
    torch.save({"state_dict": best_state, "epoch": best_epoch,
                "val_acc": best_acc, "final_test": acc, "model_args": model_args,
                "n_params": n_params,
                "lif_hparams": lif_kwargs if args.activation == "lif" else None,
                "ea": ea_diag, "warm_start": warm}, out)
    print(f"\n✓ 权重已保存：{out}（epoch {best_epoch}，"
          f"val_acc={best_acc*100:.2f}%）")

    # ---- 报告 ----
    report = ALGO_DIR / f"eeg_tcnet_4class_report_{tag}.txt"
    chance = 100.0 / n_classes
    lines = [
        f"BCIC 2a 22ch 四分类训练报告 — "
        f"{'LIF-' if args.activation == 'lif' else ''}EEG-TCNet"
        f"（EA={args.ea}）",
        "=" * 64,
        f"tag             : {tag}",
        f"EA 模式         : {args.ea}"
        + ("（R 由 T 会话训练集计算 → T/E 应用，严格 cross-session，无泄漏）"
           if args.ea == "train" else "（无对齐，基线对照）"),
        f"EA 诊断         : R 条件数={ea_diag.get('R_cond', 'N/A'):.1f}，"
        f"对齐自检相对误差={ea_diag.get('align_frob_rel', 0)*100:.4f}%"
        if args.ea == "train" else "EA 诊断         : N/A（无对齐）",
        f"最佳验证准确率  : {best_acc*100:.2f}%（epoch {best_epoch}，hold-out 早停用）",
        f"final test(eval): {acc*100:.2f}%（cross-session，只测一次）",
        f"chance level    : {chance:.1f}%",
        f"n_channels      : {args.n_channels} | n_classes : {n_classes}",
        f"训练样本数      : {n_train} | 验证 {n_val} | final test {len(eval_ds)}",
        f"参数量          : {n_params:,}",
        f"设备            : {device}",
        f"超参数          : lr={args.lr}, batch={args.batch_size}, "
        f"epochs={args.epochs}, patience={args.patience}, seed={args.seed}, "
        f"val_split={args.val_split}, dropout={args.dropout}, cosine=True",
        f"结构            : F1={args.F1}, D={args.D}, F2={args.F2}, FT={args.FT}, "
        f"kernel={args.kernel_length}, tcn_kernel={args.tcn_kernel_size}, "
        f"dilations=(1,2), pool1={args.pool1}, pool2={args.pool2}, "
        f"readout={args.readout_mode}",
    ]
    if args.activation == "lif":
        lines += [
            f"LIF 超参        : threshold={args.threshold}, tau={args.tau}, "
            f"timesteps={args.timesteps}, alpha={args.alpha}",
            f"热启动          : {warm}（{Path(args.warm_start).name}）",
            f"对照(未微调)    : " + (f"{acc0*100:.2f}%（eval，relu 权重直接换 LIF）"
                                    if warm else "N/A（无热启动）"),
            f"发放率(eval)    : {fr_str}",
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
