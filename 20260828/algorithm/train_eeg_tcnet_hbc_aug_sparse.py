#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_eeg_tcnet_hbc_aug_sparse.py — LIF-EEG-TCNet 稀疏正则训练（HBC 增强 + 发放率惩罚）

目的
----
把 LIF-EEG-TCNet 的脉冲发放压到真正稀疏（平均发放率 <15% / <10%），
评估「稀疏度 vs 精度 vs 推理潜力」的权衡，让「稀疏性纲领」从口号变成
芯片上的计算事实（事件驱动/稀疏推理潜力评估）。

与 train_eeg_tcnet_hbc_aug.py 的关系（协议 100% 一致，仅新增稀疏正则）：
  - HBC 增强照旧：每 batch SNR ∈ [10,20]dB 随机（5bit 量化 + AWGN + 突发），
    8dB 鲁棒性（54.17%+）是否保持是本实验的核心问题之一；
  - hold-out 15% / patience=25 / cosine / dropout=0.25 / 热启动 relu / seed=42；
  - 唯一新增：稀疏正则项 L_sparse，加进总损失。

稀疏正则（对 LIF 发放率加惩罚）
--------------------------------
对每个 LIF 位点（act1/act2 + TCN 两残差块 × act_a/act_b/act_res = 8 个），
用 forward hook 抓取该层输出的**可导发放率张量**（surrogate 梯度可回传），
逐层算惩罚后求和：

  mode="hinge"（默认，目标发放率正则）:
      L_sparse = λ · Σ_layers mean( relu( rate_l − target ) )
      → 发放率高于 target 才有梯度，低于 target 白送（不把网络往死里压）
  mode="l1"（L1 惩罚，等价 λ·mean(spk) 方案）:
      L_sparse = λ · Σ_layers mean( rate_l )
      → 无条件把发放率往 0 压（更激进，慎用）

  total = CE + L_sparse

λ 档位（本实验 3 档，保守/中/激进）：
  --sparse-lambda 0.3  --sparse-target 0.15   （保守：目标平均 <15%）
  --sparse-lambda 1.0  --sparse-target 0.10   （中：目标平均 <10%）
  --sparse-lambda 3.0  --sparse-target 0.05   （激进：尽量压低，看精度代价）

用法
----
  env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin \\
      /opt/miniconda3/envs/openmmlab/bin/python train_eeg_tcnet_hbc_aug_sparse.py \\
      --aug --snr-min 10 --snr-max 20 --sparse-lambda 1.0 --sparse-target 0.10 \\
      --tag sparse_l1.0_t0.10

产物（均在 algorithm/）
  eeg_tcnet_4class_lif_hbc_{tag}.pth        权重（state_dict + 训练信息 + 稀疏配置）
  eeg_tcnet_4class_report_lif_hbc_{tag}.txt 逐 epoch 报告（含稀疏损失曲线）
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
from hbc_noise_advanced import hbc_noise, snr_db_to_sigma_ratio  # noqa: E402
from lif_eegnet import LIFActivation    # noqa: E402


def _pick_device(no_cuda: bool):
    if no_cuda:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _evaluate(model, loader, device) -> tuple[float, dict]:
    """返回 (acc, 各 LIF 位点平均发放率 dict)。与 train_eeg_tcnet_lif.py 一致。"""
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
    """构造训练期增强配置（None = 不增强），与 train_eeg_tcnet_hbc_aug.py 完全一致。"""
    if not args.aug:
        return None
    spec = dict(
        enabled=True,
        snr_min_db=args.snr_min,
        snr_max_db=args.snr_max,
        qbits=5,
        qrange=4.0,
        burst_sigma_ratio=None if args.no_burst else args.burst_sigma_ratio,
        burst_len=args.burst_len,
        burst_prob=0.0 if args.no_burst else args.burst_prob,
    )
    return spec


def _apply_aug(xb: torch.Tensor, aug: dict) -> torch.Tensor:
    """按增强配置给一个 batch 施加 HBC 噪声（每 batch 随机 SNR）。"""
    snr_db = aug["snr_min_db"] + (aug["snr_max_db"] - aug["snr_min_db"]) * float(torch.rand(1).item())
    sigma_ratio = snr_db_to_sigma_ratio(snr_db)  # σ/H = 10^(-SNR/20)
    kw = dict(qbits=aug["qbits"], qrange=aug["qrange"], sigma_ratio=sigma_ratio)
    if aug["burst_sigma_ratio"] is not None:
        kw.update(burst_sigma_ratio=aug["burst_sigma_ratio"],
                  burst_len=aug["burst_len"], burst_prob=aug["burst_prob"])
    return hbc_noise(xb, **kw)


class _SparseRateHooks:
    """在 LIFActivation 上挂 forward hook，收集**可导**的逐层发放率。

    为什么不用 model.firing_rates：那个 dict 存的是 float（detached），
    没法回传梯度。这里 hook 直接抓 LIF 层输出张量（rate ∈ [0,1]），
    逐层 mean 后仍是可导标量，可进计算图做稀疏正则。
    """

    def __init__(self, model: nn.Module):
        self.rates: dict[str, torch.Tensor] = {}
        self._handles = []
        for name, m in model.named_modules():
            if isinstance(m, LIFActivation):
                self._handles.append(m.register_forward_hook(self._make_hook(name)))

    def _make_hook(self, name: str):
        def hook(_module, _inp, out):
            self.rates[name] = out.mean()  # 可导：整层发放率标量
        return hook

    def clear(self):
        self.rates.clear()

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()


def _sparse_loss(hooks: _SparseRateHooks, lam: float, target: float,
                 mode: str) -> tuple[torch.Tensor, dict]:
    """稀疏正则损失 = λ · Σ_layers penalty(rate_l)。

    hinge: penalty = mean(relu(rate − target))，只压高于目标的部分；
    l1   : penalty = mean(rate)，无条件压向 0。
    返回 (损失, 逐层 penalty dict 供报告)。
    """
    total = torch.zeros((), dtype=torch.float32, device=next(iter(hooks.rates.values())).device)
    per_layer = {}
    for name, rate in hooks.rates.items():
        if mode == "hinge":
            pen = torch.relu(rate - target)
        else:  # l1
            pen = rate
        per_layer[name] = float(pen.detach().item())
        total = total + pen
    return lam * total, per_layer


def main() -> int:
    parser = argparse.ArgumentParser(
        description="LIF-EEG-TCNet 稀疏正则训练（HBC 增强 + 发放率惩罚，surrogate，热启动 relu）")
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
    # ---- LIF 超参（默认 = 既有最优 thr=1.0/tau=2.0/T=8）----
    parser.add_argument("--threshold", type=float, default=1.0)
    parser.add_argument("--tau", type=float, default=2.0)
    parser.add_argument("--timesteps", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=4.0)
    parser.add_argument("--warm-start", type=str,
                        default=str(ALGO_DIR / "eeg_tcnet_4class_relu_brief.pth"),
                        help="热启动权重（relu brief，strict=True）")
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
    # ---- 稀疏正则（本脚本核心新增）----
    parser.add_argument("--sparse-lambda", type=float, default=0.0,
                        help="稀疏正则强度 λ（0 = 关闭，等价基线训练）")
    parser.add_argument("--sparse-target", type=float, default=0.10,
                        help="目标发放率（hinge 模式：高于 target 才受罚）")
    parser.add_argument("--sparse-mode", type=str, default="hinge", choices=["hinge", "l1"],
                        help="hinge=目标发放率正则（默认）；l1=λ·mean(spk) 无条件压 0")
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
    parser.add_argument("--tag", type=str, default="", help="产物后缀（默认 sparse_l{λ}_t{target}）")
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

    aug = _make_aug_spec(args)
    tag = args.tag or f"sparse_l{args.sparse_lambda:g}_t{args.sparse_target:g}"
    aug_desc = "无增强（对照）" if aug is None else \
        (f"HBC增强[5bit+AWGN σ/H=10^(-SNR/20), SNR∈[{args.snr_min},{args.snr_max}]dB"
         + (f" +burst σb/H={args.burst_sigma_ratio} len={args.burst_len} p={args.burst_prob}]"
            if not args.no_burst else "]（无突发）"))
    sparse_desc = ("关闭（等价基线）" if args.sparse_lambda <= 0.0 else
                   f"稀疏正则 λ={args.sparse_lambda:g} mode={args.sparse_mode} "
                   f"target={args.sparse_target:g}（L_sparse=λ·Σ mean(relu(rate−target))）"
                   if args.sparse_mode == "hinge" else
                   f"稀疏正则 λ={args.sparse_lambda:g} mode=l1（L_sparse=λ·Σ mean(rate)）")

    print(f"设备     : {device}")
    print(f"模型     : LIF-EEG-TCNet（brief 配置，LIF thr={args.threshold} "
          f"tau={args.tau} T={args.timesteps} alpha={args.alpha}）")
    print(f"训练增强 : {aug_desc}")
    print(f"稀疏正则 : {sparse_desc}")
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

    # ---- 构造 LIF 模型（同基线）----
    lif_kwargs = dict(threshold=args.threshold, tau=args.tau,
                      timesteps=args.timesteps, alpha=args.alpha)
    model = EEGTCNetLIF(n_channels=args.n_channels, n_classes=n_classes,
                        input_length=args.input_length, F1=args.F1, D=args.D,
                        F2=args.F2, FT=args.FT, kernel_length=args.kernel_length,
                        tcn_kernel_size=args.tcn_kernel_size,
                        pool1=args.pool1, pool2=args.pool2,
                        readout_mode=args.readout_mode,
                        dropout=args.dropout, activation="lif", **lif_kwargs)

    # ---- 热启动 relu 版权重（同基线）----
    warm = False
    ws = Path(args.warm_start)
    if ws.exists() and not args.no_warm:
        ck = torch.load(ws, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["state_dict"], strict=True)
        warm = True
        print(f"[热启动] 已加载 {ws.name} 的 conv/BN/fc 权重（relu→LIF 位点无参数，strict=True ✓）")
    else:
        print("[警告] 未热启动，LIF 从随机初始化训练（对照用）")

    model = model.to(device)
    n_params = model.num_params()
    print(f"参数量   : {n_params:,}")

    # ---- 对照：热启动未微调，直接换 LIF 激活在 eval 上的表现（同基线）----
    if warm:
        with torch.no_grad():
            acc0, fr0 = _evaluate(model, eval_loader, device)
            print(f"\n[对照] relu 权重直接换 LIF 激活（不微调）eval 准确率 : {acc0*100:.2f}%")
            print(f"       对照发放率 : " + "  ".join(
                f"{k.split('.')[-1]}={fr0[k]*100:.1f}%" for k in sorted(fr0)))

    # ---- 稀疏正则 hook（只影响训练；eval 时不用）----
    hooks = _SparseRateHooks(model) if args.sparse_lambda > 0.0 else None

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    history = []
    best_acc, best_epoch, best_state, patience_counter = 0.0, -1, None, 0
    print(f"\n开始训练 LIF-EEG-TCNet（{aug_desc}；{sparse_desc}）...")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        total_loss, total_ce, total_sp, total_correct, total_n = 0.0, 0.0, 0.0, 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            # ---- HBC 噪声增强：训练时对输入施加信道退化（val/eval 不加）----
            if aug is not None:
                xb = _apply_aug(xb, aug)
            optimizer.zero_grad()
            if hooks is not None:
                hooks.clear()
            logits = model(xb)
            loss = criterion(logits, yb)
            total_ce += loss.item() * xb.size(0)
            if hooks is not None:
                sp_loss, _ = _sparse_loss(hooks, args.sparse_lambda,
                                          args.sparse_target, args.sparse_mode)
                loss = loss + sp_loss
                total_sp += float(sp_loss.detach().item()) * xb.size(0)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
            total_correct += (logits.argmax(1) == yb).sum().item()
            total_n += xb.size(0)
        scheduler.step()  # cosine 退火（协议固定）
        train_loss = total_loss / total_n
        train_ce = total_ce / total_n
        train_sp = total_sp / total_n
        train_acc = total_correct / total_n
        val_acc, val_fr = _evaluate(model, val_loader, device)
        dt = time.time() - t0
        avg_fr = float(np.mean(list(val_fr.values()))) if val_fr else 0.0
        history.append({"epoch": epoch, "train_loss": round(train_loss, 6),
                        "train_ce": round(train_ce, 6), "train_sp": round(train_sp, 6),
                        "train_acc": round(train_acc, 4), "val_acc": round(val_acc, 4),
                        "avg_fr": round(avg_fr, 4)})
        print(f"  Epoch {epoch:3d}/{args.epochs} | ce={train_ce:.4f} sp={train_sp:.4f} "
              f"train_acc={train_acc*100:5.2f}% val_acc={val_acc*100:5.2f}% "
              f"avg_fr={avg_fr*100:5.1f}% | {dt:.1f}s")
        if val_acc > best_acc:
            best_acc, best_epoch = val_acc, epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\n早停：验证集连续 {args.patience} epoch 未提升。")
                break

    if hooks is not None:
        hooks.remove()
    if best_state is None:
        raise SystemExit("✗ 训练未产生有效模型")
    model.load_state_dict(best_state)
    model.eval()

    # ---- final test：eval（cross-session，只测一次）----
    acc, fr = _evaluate(model, eval_loader, device)
    print(f"\n[final test] LIF-EEG-TCNet（{aug_desc}；{sparse_desc}）eval 准确率 : {acc*100:.2f}%")
    print(f"             发放率(eval) : " + "  ".join(
        f"{k.split('.')[-1]}={fr[k]*100:.1f}%" for k in sorted(fr)))

    # ---- 保存 ----
    out = Path(args.out) if args.out else ALGO_DIR / f"eeg_tcnet_4class_lif_hbc_{tag}.pth"
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
                "aug_config": None if aug is None else {k: v for k, v in aug.items()},
                "aug_desc": aug_desc, "firing_rates": fr,
                "sparse_config": None if args.sparse_lambda <= 0.0 else {
                    "lambda": args.sparse_lambda, "target": args.sparse_target,
                    "mode": args.sparse_mode}},
               out)
    print(f"\n✓ 权重已保存：{out}（epoch {best_epoch}，val_acc={best_acc*100:.2f}%，"
          f"final_test={acc*100:.2f}%）")

    # ---- 报告 ----
    report = ALGO_DIR / f"eeg_tcnet_4class_report_lif_hbc_{tag}.txt"
    chance = 100.0 / n_classes
    lines = [
        f"BCIC 2a 22ch 四分类训练报告 — LIF-EEG-TCNet 稀疏正则 + HBC 噪声增强（surrogate，热启动 relu）",
        "=" * 64,
        f"tag             : {tag}",
        f"训练增强        : {aug_desc}",
        f"稀疏正则        : {sparse_desc}",
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
        f"发放率(eval)    : " + "  ".join(
            f"{k.split('.')[-1]}={fr[k]*100:.1f}%" for k in sorted(fr)),
        "噪声模型        : hbc_noise_advanced.hbc_noise（5bit 量化 + AWGN + 突发，"
        "与 luckfox-deploy/model/noise.py 严格一致）",
        "-" * 64, "逐 epoch 记录 (epoch, ce, sparse, train_acc, val_acc, avg_fr):",
    ]
    for h in history:
        lines.append(f"  {h['epoch']:3d}  {h['train_ce']:.6f}  {h['train_sp']:.6f}  "
                     f"{h['train_acc']*100:5.2f}%  {h['val_acc']*100:5.2f}%  {h['avg_fr']*100:5.1f}%")
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  报告已写入：{report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
