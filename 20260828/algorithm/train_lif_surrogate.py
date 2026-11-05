#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_lif_surrogate.py — surrogate gradient 直接训练 LIF-EEGNet（S3 步）

目的
----
  ann2snn 转换路线卡在 ~55%（ReLU 代理 75.77% → 转换后 ~55%），根因是
  率编码 LIF 的粗量化 + 单标量 max 归一化。本脚本换一条路：不转换，而是
  用 surrogate gradient 直接端到端训练 LIFEEGNet(activation='lif')，让梯度
  下降自己去找「对发放率量化最友好的」权重 / 阈值 / 归一化，验证 SNN 精度
  上限能否绕开 55% 瓶颈。

关键设计
--------
  - 可导 LIF：复用 lif_eegnet.LIFActivation（sigmoid 近似 surrogate gradient，
    硬重置，输出发放率 ∈ [0,1]）。threshold/tau/timesteps/alpha 均作为超参传入。
  - 可选 learnable threshold：把 act1/act2 的 threshold 替换成 nn.Parameter，
    让梯度下降自动找逐层阈值（等效 Rueckauer 归一化的「可微版本」）。
    默认关闭（固定阈值），消融时显式开 --learnable-threshold。
  - 热启动：load relu_proxy.pth（75.77% ReLU 代理）的 conv/fc/BN 权重；
    LIF 激活层无参数，state_dict 键零回归，strict=True 直接载入。
  - hold-out 早停：train 划 15% 做验证，final test 用 eval（消除选择偏差）。
  - 报告：干净 / HBC 噪声 两个维度，与 ReLU 代理 75.77% 对比，如实记录。

用法
----
  env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin \\
      /opt/miniconda3/envs/openmmlab/bin/python train_lif_surrogate.py \\
      --warm-start relu_proxy.pth --threshold 1.0 --tau 2.0 --timesteps 8 \\
      --epochs 50 --n-classes 2 --out lif_surrogate.pth --tag t1.0_tau2_T8
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
# 设备选择 / HBC 噪声增强（与 train_relu_proxy.py 语义一致）
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
    """返回 (acc, 平均发放率 dict)。HBC 噪声评估用固定种子保证可复现。"""
    model.eval()
    correct, total = 0, 0
    fr_sum = {"act1": 0.0, "act2": 0.0}
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            if use_hbc:
                torch.manual_seed(1234)          # 固定噪声种子，可复现
                xb = _hbc_augment(xb)
            logits = model(xb)
            correct += (logits.argmax(1) == yb).sum().item()
            total += xb.size(0)
            # 发放率统计（评估阶段从 LIF 层读，用于诊断量化程度）
            for k, act in (("act1", model.act1), ("act2", model.act2)):
                fr_sum[k] += float(act.last_firing_rate) * xb.size(0)
    fr = {k: v / total for k, v in fr_sum.items()}
    return correct / total, fr


def _make_lif_model(n_classes: int, threshold: float, tau: float, timesteps: int,
                    alpha: float) -> LIFEEGNet:
    """构造 LIF-EEGNet（n_classes 可配置，LIF 超参统一传给 act1/act2）。"""
    return LIFEEGNet(n_channels=8, n_classes=n_classes, input_length=500,
                     F1=8, D=2, F2=16, kernel_length=63, dropout=0.5,
                     activation="lif", threshold=threshold, tau=tau,
                     timesteps=timesteps, alpha=alpha)


def _enable_learnable_threshold(model: LIFEEGNet, init: float) -> None:
    """把 act1/act2 的固定阈值替换成可学习 nn.Parameter（逐层独立，初值 init）。

    LIFActivation.forward 中 `mem - thr` 对 tensor 阈值同样可用（broadcast），
    因此无需改动 lif_eegnet.py，只需在构造 optimizer 之前替换属性即可。
    """
    for name in ("act1", "act2"):
        act = getattr(model, name)
        act.threshold = nn.Parameter(torch.tensor(float(init)))
        act.learnable_threshold = True


def main() -> int:
    parser = argparse.ArgumentParser(description="surrogate gradient 直接训练 LIF-EEGNet")
    parser.add_argument("--warm-start", type=str,
                        default=str(ALGO_DIR / "relu_proxy.pth"),
                        help="ReLU 代理权重热启动路径（conv/fc/BN）")
    parser.add_argument("--threshold", type=float, default=1.0, help="LIF 发放阈值（固定）")
    parser.add_argument("--tau", type=float, default=2.0, help="膜时间常数（decay=exp(-1/tau)）")
    parser.add_argument("--timesteps", type=int, default=8, help="仿真时间步数")
    parser.add_argument("--alpha", type=float, default=4.0, help="surrogate sigmoid 斜率")
    parser.add_argument("--learnable-threshold", action="store_true",
                        help="把逐层阈值设为可学习 nn.Parameter")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--n-classes", type=int, default=2)
    parser.add_argument("--hbc-aug", action="store_true", help="训练时做 HBC 噪声增强")
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--out", type=str, default=str(ALGO_DIR / "lif_surrogate.pth"))
    parser.add_argument("--tag", type=str, default="", help="报告/文件名后缀标识")
    parser.add_argument("--json-out", type=str, default="",
                        help="若给定，把 (tag→结果) 追加写入该 JSON 文件（供消融汇总）")
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
    print(f"LIF 超参 : threshold={args.threshold}, tau={args.tau}, "
          f"timesteps={args.timesteps}, alpha={args.alpha}, "
          f"learnable_threshold={args.learnable_threshold}")

    # ---- hold-out 划分：早停验证集与 final test（eval）分离 ----
    n_val = max(1, int(len(train_ds) * args.val_split))
    n_train = len(train_ds) - n_val
    g = torch.Generator().manual_seed(args.seed)
    train_ds, val_ds = torch.utils.data.random_split(train_ds, [n_train, n_val], generator=g)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False)
    print(f"hold-out : {n_train} 训练 / {n_val} 验证（早停用） | final test = eval（{len(eval_ds)} 试次）")

    # ---- 模型：LIF-EEGNet（surrogate gradient，n_classes 可配置）----
    model = _make_lif_model(n_classes, args.threshold, args.tau,
                            args.timesteps, args.alpha)

    # ---- 热启动：load ReLU 代理权重（LIF 激活层无参数，state_dict 键零回归）----
    ws = Path(args.warm_start)
    warm = False
    if ws.exists():
        ck = torch.load(ws, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["state_dict"], strict=True)
        warm = True
        print(f"[热启动] 已加载 {ws.name} 的 conv/fc/BN 权重（ReLU→LIF 激活位点无参数）")
    else:
        print(f"[警告] 未找到热启动权重 {ws}，从随机初始化训练")

    # ---- 可选：逐层阈值可学习（须在构造 optimizer 之前）----
    if args.learnable_threshold:
        _enable_learnable_threshold(model, args.threshold)
        print(f"[learnable] act1/act2 阈值已设为 nn.Parameter，初值 {args.threshold}")

    model = model.to(device)
    print(f"参数量   : {model.num_params():,}")

    # ---- 对照：热启动未微调，直接在 eval 上的干净/HBC 准确率 ----
    if warm:
        print("\n[对照] ReLU 权重直接换 LIF 激活（不微调）—— 量化 LIF 的即时代价：")
        with torch.no_grad():
            for tag, use_hbc in (("干净", False), ("HBC噪声", True)):
                acc, fr = _evaluate(model, eval_loader, device, use_hbc=use_hbc)
                print(f"        LIF(热启动未微调) {tag} : {acc*100:.2f}%  "
                      f"(发放率 act1={fr['act1']*100:.2f}% act2={fr['act2']*100:.2f}%)")

    # ---- 微调 ----
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    history = []
    best_acc, best_epoch, best_state, patience_counter = 0.0, -1, None, 0
    print("\n开始 surrogate gradient 微调 LIF-EEGNet ...")
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
        # 打印可学习阈值当前值（便于诊断）
        thr_str = ""
        if args.learnable_threshold:
            thr_str = (f" thr1={float(model.act1.threshold):.3f}"
                       f" thr2={float(model.act2.threshold):.3f}")
        print(f"  Epoch {epoch:3d}/{args.epochs} | train_loss={train_loss:.4f} "
              f"train_acc={train_acc*100:5.2f}% val_acc={val_acc*100:5.2f}%{thr_str} | {dt:.1f}s")
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

    # ---- final test：clean + HBC 噪声（eval 只测一次）----
    print("\n[final test] LIF-EEGNet（surrogate，eval，cross-session，只测一次）：")
    results = {}
    with torch.no_grad():
        for tag, use_hbc in (("干净", False), ("HBC噪声", True)):
            acc, fr = _evaluate(model, eval_loader, device, use_hbc=use_hbc)
            results[tag] = acc
            print(f"        LIF surrogate {tag} : {acc*100:.2f}%  "
                  f"(发放率 act1={fr['act1']*100:.2f}% act2={fr['act2']*100:.2f}%)")

    # ---- 保存 ----
    thr_final = {}
    if args.learnable_threshold:
        thr_final = {"act1": float(model.act1.threshold), "act2": float(model.act2.threshold)}
    out = Path(args.out)
    torch.save({"state_dict": best_state, "epoch": best_epoch, "val_acc": best_acc,
                "final_test": results, "firing_rate": fr,
                "lif_hparams": dict(threshold=args.threshold, tau=args.tau,
                                    timesteps=args.timesteps, alpha=args.alpha,
                                    learnable_threshold=args.learnable_threshold),
                "final_threshold": thr_final,
                "model_args": dict(n_channels=8, n_classes=n_classes, input_length=500,
                                   F1=8, D=2, F2=16, kernel_length=63, dropout=0.5,
                                   activation="lif")}, out)
    print(f"\n✓ LIF surrogate 权重已保存：{out}（epoch {best_epoch}，val_acc={best_acc*100:.2f}%）")

    # ---- 报告 ----
    tag = args.tag or f"thr{args.threshold}_tau{args.tau}_T{args.timesteps}"
    report = ALGO_DIR / f"lif_surrogate_report_{tag}.txt"
    lines = [
        "LIF-EEGNet surrogate gradient 直接训练报告（S3）",
        "=" * 60,
        f"tag             : {tag}",
        f"最佳验证准确率  : {best_acc*100:.2f}%（epoch {best_epoch}，hold-out 早停用）",
        f"final test(eval): 干净 {results['干净']*100:.2f}% / HBC噪声 {results['HBC噪声']*100:.2f}%",
        f"对照 ReLU 代理  : 干净 75.77% / HBC噪声 75.77%",
        f"n_classes       : {n_classes}",
        f"LIF 超参        : threshold={args.threshold}, tau={args.tau}, "
        f"timesteps={args.timesteps}, alpha={args.alpha}, "
        f"learnable_threshold={args.learnable_threshold}",
        f"最终逐层阈值    : {thr_final or '（固定）'}",
        f"发放率          : act1={fr['act1']*100:.2f}% act2={fr['act2']*100:.2f}%",
        f"训练样本数      : {n_train} | 验证 {n_val} | final test {len(eval_ds)}",
        f"参数量          : {model.num_params():,}",
        f"设备            : {device}",
        f"超参数          : lr={args.lr}, batch={args.batch_size}, epochs={args.epochs}, "
        f"patience={args.patience}, seed={args.seed}, hbc_aug={args.hbc_aug}",
        "-" * 60,
        "逐 epoch 记录 (epoch, train_loss, train_acc, val_acc):",
    ]
    for h in history:
        lines.append(f"  {h['epoch']:3d}  {h['train_loss']:.6f}  "
                     f"{h['train_acc']*100:5.2f}%  {h['val_acc']*100:5.2f}%")
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  报告已写入：{report}")

    # ---- 可选：追加 JSON 汇总（供消融脚本收集）----
    if args.json_out:
        rec = {
            "tag": tag,
            "threshold": args.threshold, "tau": args.tau, "timesteps": args.timesteps,
            "alpha": args.alpha, "learnable_threshold": args.learnable_threshold,
            "best_val_acc": best_acc, "final_clean": results["干净"],
            "final_hbc": results["HBC噪声"],
            "fr_act1": fr["act1"], "fr_act2": fr["act2"],
            "final_threshold": thr_final, "epoch": best_epoch,
        }
        jp = Path(args.json_out)
        allrecs = []
        if jp.exists():
            try:
                allrecs = json.loads(jp.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                allrecs = []
        allrecs.append(rec)
        jp.write_text(json.dumps(allrecs, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  结果已追加到：{jp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
