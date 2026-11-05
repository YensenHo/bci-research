#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
finetune_compare.py — 自监督预训练 vs 从零监督的对比微调（BCIC 2a，EEG-TCNet 9.5K）

核心问题：Masked EEG AE 自监督预训练（pretrain_masked_ae.py 的 ssl_ae_aug 权重）对
端侧小模型有无红利——按 数据效率（100/30/10% 标签）× 激活（relu/lif）× 鲁棒性（8dB）。

设计（公平性红线：**微调阶段所有组用同一协议，只变初始化**）：
  - 协议 = train_eeg_tcnet_hbc_aug.py 同款：HBC 增强（5bit+AWGN SNR∈[10,20]dB +突发，
    每 batch 随机 SNR）、hold-out 15% 早停（patience 25）、cosine、lr=1e-3、batch 16、
    dropout 0.25、seed 42、epochs 150。eval（E 会话）只测不训。
  - 初始化（唯一变量）：
      rand : 随机初始化（真·从零）
      warm : 热启动 eeg_tcnet_4class_relu_brief.pth（全量监督 relu 分类器权重）
             ——代码库「LIF 从零监督」标准管线（62.42/63.58 同源）；在 10/30% 标签时
             它本身携带全量标签信息，故在报告里充当「监督预训练对照」；
      ssl  : 热启动 pretrain_masked_ae.py 产出的 encoder 权重（无标签预训练；
             fc/gap 随机初始化），== 本实验的被测对象。
  - 标签子集：10%/30% 由固定 seed 从 train.npz 抽一次（同一子集给 rand/ssl/warm 共用），
    保存到 ssl_ae_results/label_subsets/ 供核查。
  - 对比矩阵（8 组，seed 42）：
      100%: relu_rand / relu_ssl / lif_ssl（lif_warm 引用既有 aug_s42=63.58，同协议）
      30% : relu_rand / lif_warm / lif_ssl
      10% : lif_warm / lif_ssl
  - 鲁棒性：测试时对 eval 施加 8dB（5bit+AWGN σ/H=10^(-8/20)）与 8dB+突发，
    噪声 seed=20240828（与 hbc_aug_results/snr_sweep.json 同约定）；每组测 clean/8dB/8dB+突发。

用法
----
  # 冒烟
  env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin \
      /opt/miniconda3/envs/openmmlab/bin/python finetune_compare.py \
      --act relu --init rand --frac 1.0 --smoke
  # 单组全量（每组长 <20min）
  ... finetune_compare.py --act relu --init rand --frac 1.0
  # 评估既有引用模型（同一 harness 复测 clean/8dB/8dB+突发）
  ... finetune_compare.py --eval-existing

产物（algorithm/ssl_ae_results/）
  finetune_{act}_{init}_{frac}_{seed}.pth / .json   每组权重 + 结果
  finetune_refs.json                                 既有模型同 harness 复测
  label_subsets/f{frac}_{seed}.json                  标签子集索引（泄漏可核查）
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
DATA_DIR = Path("/Users/heyixuan/BCI/luckfox-deploy/data")
DATA_4CLASS = ALGO_DIR / "data_4class"
OUT_DIR = ALGO_DIR / "ssl_ae_results"
SUB_DIR = OUT_DIR / "label_subsets"
RELU_BRIEF = ALGO_DIR / "eeg_tcnet_4class_relu_brief.pth"
SSL_CKPT = OUT_DIR / "ssl_ae_aug.pth"
NOISE_SEED = 20240828

sys.path.insert(0, str(ALGO_DIR))
sys.path.insert(0, str(DATA_DIR))

from dataset import EEGDataset                    # noqa: E402
from eeg_tcnet_lif import EEGTCNetLIF             # noqa: E402
from hbc_noise_advanced import hbc_noise, snr_db_to_sigma_ratio  # noqa: E402


def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _apply_aug(xb: torch.Tensor, aug: dict) -> torch.Tensor:
    """与 train_eeg_tcnet_hbc_aug.py 完全一致：每 batch 随机 SNR∈[10,20]dB + 突发。"""
    snr_db = aug["snr_min_db"] + (aug["snr_max_db"] - aug["snr_min_db"]) * float(torch.rand(1).item())
    sigma_ratio = snr_db_to_sigma_ratio(snr_db)
    kw = dict(qbits=aug["qbits"], qrange=aug["qrange"], sigma_ratio=sigma_ratio,
              burst_sigma_ratio=aug["burst_sigma_ratio"], burst_len=aug["burst_len"],
              burst_prob=aug["burst_prob"])
    return hbc_noise(xb, **kw)


AUG_SPEC = dict(snr_min_db=10.0, snr_max_db=20.0, qbits=5, qrange=4.0,
                burst_sigma_ratio=0.3, burst_len=16, burst_prob=0.005)


def _evaluate(model, loader, device) -> float:
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            correct += (logits.argmax(1) == yb).sum().item()
            total += xb.size(0)
    return correct / total


def _evaluate_snr(model, loader, device, snr_db: float | None, burst: bool) -> float:
    """测试期加噪评估：5bit(qrange=4.0) + AWGN σ/H=10^(-SNR/20)；burst=True 加突发。
    噪声固定 seed=NOISE_SEED（与 snr_sweep.json 同约定，逐 batch 顺序消费 → 可复现）。"""
    model.eval()
    torch.manual_seed(NOISE_SEED)
    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            if snr_db is not None:
                kw = dict(qbits=5, qrange=4.0,
                          sigma_ratio=snr_db_to_sigma_ratio(snr_db))
                if burst:
                    kw.update(burst_sigma_ratio=0.3, burst_len=16, burst_prob=0.005)
                xb = hbc_noise(xb, **kw)
            logits = model(xb)
            correct += (logits.argmax(1) == yb).sum().item()
            total += xb.size(0)
    return correct / total


def make_label_subset(n_all: int, frac: float, seed: int) -> list[int]:
    """固定 seed 抽取标签子集（rand/ssl/warm 共用同一子集，保证可比）。"""
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n_all)
    n_lab = int(round(n_all * frac))
    return sorted(perm[:n_lab].tolist())


def build_model(activation: str, lif_hparams: dict | None) -> EEGTCNetLIF:
    return EEGTCNetLIF(n_channels=22, n_classes=4, input_length=500,
                       F1=8, D=2, F2=16, FT=12, kernel_length=63,
                       tcn_kernel_size=12, pool1=4, pool2=8,
                       readout_mode="gap", dropout=0.25,
                       activation=activation, **(lif_hparams or {}))


def apply_init(model: EEGTCNetLIF, init: str, activation: str,
               lif_hparams: dict | None) -> str:
    """按 init 策略载入权重，返回描述字符串。"""
    if init == "rand":
        return "随机初始化（真·从零）"
    if init == "warm":
        ck = torch.load(RELU_BRIEF, map_location="cpu", weights_only=False)
        missing, unexpected = model.load_state_dict(ck["state_dict"], strict=True)
        assert not missing and not unexpected, f"warm strict 载入失败: {missing}/{unexpected}"
        return f"监督热启动（{RELU_BRIEF.name}，全量监督 relu 分类器，strict=True）"
    if init == "ssl":
        if not SSL_CKPT.exists():
            raise SystemExit(f"✗ 预训练权重缺失：{SSL_CKPT}（先跑 pretrain_masked_ae.py）")
        ck = torch.load(SSL_CKPT, map_location="cpu", weights_only=False)
        enc = ck["encoder_state_dict"]
        missing, unexpected = model.load_state_dict(enc, strict=False)
        assert not unexpected, f"ssl 载入出现 unexpected 键: {unexpected}"
        assert set(missing) <= {"fc.weight", "fc.bias"}, f"ssl 载入缺失键异常: {missing}"
        return (f"SSL 预训练热启动（{Path(SSL_CKPT).name}，{len(enc)} 键 conv/BN 权重；"
                f"fc 随机初始化 {sorted(missing)}）")
    raise ValueError(init)


def train_cell(args) -> dict:
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = _pick_device()

    train_ds = EEGDataset(DATA_4CLASS / "train.npz")
    eval_ds = EEGDataset(DATA_4CLASS / "eval.npz")
    n_all = len(train_ds)

    # ---- 标签子集（100% = 全量，与既有 63.58 运行同划分方式）----
    if args.frac >= 1.0:
        lab_ds = train_ds
        n_lab = n_all
        lab_idx = None
    else:
        lab_idx = make_label_subset(n_all, args.frac, seed)
        n_lab = len(lab_idx)
        lab_ds = torch.utils.data.Subset(train_ds, lab_idx)
        SUB_DIR.mkdir(parents=True, exist_ok=True)
        (SUB_DIR / f"f{int(round(args.frac * 100))}_seed{seed}.json").write_text(
            json.dumps({"frac": args.frac, "seed": seed, "n_lab": n_lab,
                        "indices": lab_idx}), encoding="utf-8")
    if args.smoke:
        lab_ds = torch.utils.data.Subset(lab_ds, list(range(min(n_lab, 240))))
        n_lab = min(n_lab, 240)

    n_val = max(1, int(round(n_lab * 0.15)))
    g = torch.Generator().manual_seed(seed)
    tr_sub, va_sub = torch.utils.data.random_split(
        lab_ds, [n_lab - n_val, n_val], generator=g)
    tr_loader = DataLoader(tr_sub, batch_size=args.batch_size, shuffle=True)
    va_loader = DataLoader(va_sub, batch_size=args.batch_size, shuffle=False)
    ev_loader = DataLoader(eval_ds, batch_size=32, shuffle=False)

    lif_hparams = None
    if args.act == "lif":
        lif_hparams = dict(threshold=1.0, tau=2.0, timesteps=8, alpha=4.0)
    model = build_model(args.act, lif_hparams)
    init_desc = apply_init(model, args.init, args.act, lif_hparams)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"设备     : {device} | 激活 {args.act} | init: {init_desc}")
    print(f"标签     : {args.frac:.0%}（n_lab={n_lab}，train {n_lab - n_val}/val {n_val}）"
          f" | final test = eval（{len(eval_ds)}，E 会话，只测不训）")

    epochs = 3 if args.smoke else args.epochs
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    best_acc, best_ep, best_state, patience = 0.0, -1, None, 0
    print(f"\n开始微调（{args.act}/{args.init}/{args.frac:.0%}，HBC 增强协议，{epochs} epochs）...")
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        tl, tc, tn = 0.0, 0, 0
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            xb = _apply_aug(xb, AUG_SPEC)          # 训练增强：与 63.58 运行同协议
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            tl += loss.item() * xb.size(0)
            tc += (logits.argmax(1) == yb).sum().item()
            tn += xb.size(0)
        scheduler.step()
        val_acc = _evaluate(model, va_loader, device)
        dt = time.time() - t0
        print(f"  Epoch {epoch:3d}/{epochs} | train_loss={tl / tn:.4f} "
              f"train_acc={tc / tn * 100:5.2f}% val_acc={val_acc * 100:5.2f}% | {dt:.1f}s")
        if val_acc > best_acc:
            best_acc, best_ep = val_acc, epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= args.patience:
                print(f"\n早停：val 连续 {args.patience} epoch 未提升（best ep {best_ep}）")
                break
    if best_state is None:
        raise SystemExit("✗ 训练未产生有效权重")
    model.load_state_dict(best_state)
    model.eval()

    # ---- final test：clean / 8dB / 8dB+突发（eval E 会话，只测一次口径）----
    acc_clean = _evaluate(model, ev_loader, device)
    acc_8 = _evaluate_snr(model, ev_loader, device, 8.0, burst=False)
    acc_8b = _evaluate_snr(model, ev_loader, device, 8.0, burst=True)
    print(f"\n[final test] clean={acc_clean * 100:.2f}% | 8dB={acc_8 * 100:.2f}% | "
          f"8dB+突发={acc_8b * 100:.2f}%")

    cell = f"{args.act}_{args.init}_{int(round(args.frac * 100))}_{seed}"
    torch.save({"state_dict": best_state, "model_args": dict(
        activation=args.act, **({"lif_hparams": lif_hparams} if lif_hparams else {})),
        "init": args.init, "init_desc": init_desc, "frac": args.frac,
        "best_epoch": best_ep, "best_val_acc": best_acc,
        "final_acc_clean": acc_clean, "final_acc_8db": acc_8,
        "final_acc_8db_burst": acc_8b, "n_params": n_params,
        "n_train": n_lab - n_val, "n_val": n_val, "seed": seed},
        OUT_DIR / f"finetune_{cell}.pth")
    res = dict(cell=cell, act=args.act, init=args.init, frac=args.frac, seed=seed,
               init_desc=init_desc, best_epoch=best_ep, best_val_acc=round(best_acc, 4),
               n_train=n_lab - n_val, n_val=n_val, n_params=n_params,
               acc_clean=round(acc_clean, 4), acc_8db=round(acc_8, 4),
               acc_8db_burst=round(acc_8b, 4),
               noise_seed=NOISE_SEED, aug_desc="HBC增强[5bit+AWGN SNR∈[10,20]dB +burst]")
    (OUT_DIR / f"finetune_{cell}.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"✓ 已保存：finetune_{cell}.pth / .json")
    return res


# ---------------------------------------------------------------------------
# 既有引用模型：同一 harness 复测（保证报告数字同源）
# ---------------------------------------------------------------------------
REF_CHECKPOINTS = {
    "lif_warm_100_aug(s42)": ALGO_DIR / "eeg_tcnet_4class_lif_hbc_aug_s42.pth",
    "lif_warm_100_noaug(s42)": ALGO_DIR / "eeg_tcnet_4class_lif_hbc_noaug_s42.pth",
    "lif_warm_100_aug(s123)": ALGO_DIR / "eeg_tcnet_4class_lif_hbc_aug_s123.pth",
    "relu_scratch_100_clean(s42)": ALGO_DIR / "eeg_tcnet_4class_relu_brief.pth",
}


def eval_existing() -> dict:
    device = _pick_device()
    ev_ds = EEGDataset(DATA_4CLASS / "eval.npz")
    ev_loader = DataLoader(ev_ds, batch_size=32, shuffle=False)
    out = {}
    for name, path in REF_CHECKPOINTS.items():
        if not path.exists():
            print(f"⚠ 跳过（不存在）：{path.name}")
            continue
        ck = torch.load(path, map_location="cpu", weights_only=False)
        ma = ck.get("model_args", {})
        lif_h = ck.get("lif_hparams")
        model = EEGTCNetLIF(**{k: v for k, v in ma.items() if k != "activation"},
                            activation="lif" if lif_h else "relu",
                            **(lif_h or {}))
        model.load_state_dict(ck["state_dict"], strict=True)
        model = model.to(device).eval()
        acc_clean = _evaluate(model, ev_loader, device)
        acc_8 = _evaluate_snr(model, ev_loader, device, 8.0, burst=False)
        acc_8b = _evaluate_snr(model, ev_loader, device, 8.0, burst=True)
        out[name] = dict(acc_clean=round(acc_clean, 4), acc_8db=round(acc_8, 4),
                         acc_8db_burst=round(acc_8b, 4),
                         orig_report=ck.get("final_test"))
        print(f"{name:32s} clean={acc_clean * 100:.2f}%  8dB={acc_8 * 100:.2f}%  "
              f"8dB+突发={acc_8b * 100:.2f}%（原报告 {ck.get('final_test', float('nan')) * 100:.2f}%）")
    (OUT_DIR / "finetune_refs.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"✓ 已保存：{OUT_DIR / 'finetune_refs.json'}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="自监督预训练 vs 从零监督 对比微调（2a）")
    ap.add_argument("--act", choices=["relu", "lif"], default="relu")
    ap.add_argument("--init", choices=["rand", "warm", "ssl"], default="rand")
    ap.add_argument("--frac", type=float, default=1.0, help="标签比例 1.0/0.3/0.1")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--eval-existing", action="store_true")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.eval_existing:
        eval_existing()
        return 0
    res = train_cell(args)
    print(f"\n✓ 结果：{json.dumps(res, ensure_ascii=False)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
