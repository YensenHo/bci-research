#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_loso.py — LOSO（leave-one-subject-out）跨 subject 零校准评估

方向：SNN 作为非平稳脑机接口的鲁棒性基元（首个跨 subject 零校准解码基准）
本脚本回答方向的第一道证伪判据：
    **SNN 跨 subject 零校准精度 是否 ≥ ANN 基线？**

协议（跨 subject 零校准，比「9 被试混合 cross-session」更严）：
  - LOSO：9 折，每折 train = 8 个 subject 的 T 会话（有标签），
    test = 剩下 1 个 subject 的 E 会话（零校准，直接预测，不碰目标 subject 任何数据）。
  - 这正是「康复患者是新被试、贴上去就要能用」的零校准场景。

对比模型：
  - eegnet     : EEGNet（ELU，ANN 基线）
  - signed_lif : LIF-EEGNet（signed_lif，正负脉冲分离版，四分类 22ch 补回负半轴）
  - lif        : LIF-EEGNet（lif，普通发放率版，丢负半轴）
  （LIF 版本热启动自同折的 EEGNet 权重，保证公平初始化。）

数据：data_4class_by_subject/A{nn}/train.npz（T 会话）+ eval.npz（E 会话），
      每人 288 试次（4 类 × 72），22ch × 500。

用法（环境铁律：env -i 隔离 + openmmlab python）：
  # 冒烟（1 折、少 epochs，验证脚本正确）
  env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin \
      /opt/miniconda3/envs/openmmlab/bin/python train_loso.py --smoke
  # 全量（9 折）
  env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin \
      /opt/miniconda3/envs/openmmlab/bin/python train_loso.py

产物：loso_results.json + loso_report.txt（9 折逐 subject 精度 + 平均 ± std）
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
from torch.utils.data import DataLoader, TensorDataset

ALGO_DIR = Path(__file__).resolve().parent
BY_SUBJECT = ALGO_DIR / "data_4class_by_subject"
MODEL_DIR = Path("/Users/heyixuan/BCI/luckfox-deploy/model")
SUBJECTS = [f"A{i:02d}" for i in range(1, 10)]

sys.path.insert(0, str(ALGO_DIR))
sys.path.insert(0, str(MODEL_DIR))

from eegnet import EEGNet            # noqa: E402
from lif_eegnet import LIFEEGNet     # noqa: E402

COMMON = dict(n_channels=22, n_classes=4, input_length=500,
              F1=8, D=2, F2=16, kernel_length=63, dropout=0.5)


def _pick_device(no_cuda: bool) -> torch.device:
    if no_cuda:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _build(model: str, **kw) -> nn.Module:
    if model == "eegnet":
        return EEGNet(activation="elu", **COMMON)
    if model == "signed_lif":
        return LIFEEGNet(activation="signed_lif", **COMMON)
    return LIFEEGNet(activation="lif", **COMMON)


def _train(model, X, y, device, epochs, lr, batch, seed, val_split=0.15):
    """hold-out 早停训练，返回 (best_state, val_acc)。"""
    torch.manual_seed(seed)
    n_val = max(1, int(len(X) * val_split))
    n_train = len(X) - n_val
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(X), generator=g).tolist()
    tr_idx, va_idx = idx[n_val:], idx[:n_val]
    Xtr, ytr = X[tr_idx], y[tr_idx]
    Xva, yva = X[va_idx], y[va_idx]
    train_loader = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr)),
                              batch_size=batch, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(Xva), torch.from_numpy(yva)),
                            batch_size=batch, shuffle=False)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss()
    best_acc, best_state, patience = 0.0, None, 0
    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
        sched.step()
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                correct += (model(xb).argmax(1) == yb).sum().item()
                total += yb.size(0)
        va = correct / total
        if va > best_acc:
            best_acc, best_state = va, {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= 10:
                break
    return best_state, best_acc


def _eval(model, X, y, device, batch):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.from_numpy(X[i:i + batch]).to(device)
            yb = torch.from_numpy(y[i:i + batch]).to(device)
            correct += (model(xb).argmax(1) == yb).sum().item()
            total += yb.size(0)
    return correct / total


def _load_subject(subj: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    tr = np.load(BY_SUBJECT / subj / "train.npz")
    ev = np.load(BY_SUBJECT / subj / "eval.npz")
    return (tr["X"].astype(np.float32), tr["y"].astype(np.int64),
            ev["X"].astype(np.float32), ev["y"].astype(np.int64))


def main() -> int:
    ap = argparse.ArgumentParser(description="LOSO 跨 subject 零校准评估（SNN vs ANN）")
    ap.add_argument("--smoke", action="store_true", help="只跑 1 折 + 少 epochs")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--models", type=str, default="eegnet,signed_lif,lif")
    ap.add_argument("--no-cuda", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        args.epochs = 2
    models = args.models.split(",")
    device = _pick_device(args.no_cuda)

    # 预加载全部 subject 数据
    data = {s: _load_subject(s) for s in SUBJECTS}
    print(f"设备: {device} | 模型: {models} | LOSO {len(SUBJECTS)} 折 | epochs={args.epochs}")

    results = {m: [] for m in models}
    per_subject = {m: {} for m in models}
    for test_idx, test_subj in enumerate(SUBJECTS):
        if args.smoke and test_idx > 0:
            break
        # train = 其他 8 subject 的 T 会话；test = 本 subject 的 E 会话（零校准）
        Xtr_list, ytr_list = [], []
        for s in SUBJECTS:
            if s == test_subj:
                continue
            Xtr_list.append(data[s][0]); ytr_list.append(data[s][1])
        Xtr = np.concatenate(Xtr_list); ytr = np.concatenate(ytr_list)
        Xte, yte = data[test_subj][2], data[test_subj][3]

        eegnet_state = None
        for model in models:
            t0 = time.time()
            m = _build(model).to(device)
            if model != "eegnet" and eegnet_state is not None:
                m.load_state_dict(eegnet_state, strict=True)  # LIF 激活层无参数，键零回归
            state, val_acc = _train(m, Xtr, ytr, device, args.epochs, args.lr,
                                    args.batch, args.seed)
            m.load_state_dict(state)
            acc = _eval(m, Xte, yte, device, args.batch)
            results[model].append(acc)
            per_subject[model][test_subj] = round(acc * 100, 2)
            if model == "eegnet":
                eegnet_state = state  # 供 LIF 热启动
            print(f"  [{test_subj}] {model:>10}: test={acc*100:6.2f}% "
                  f"(val={val_acc*100:5.2f}%, {time.time()-t0:.0f}s)")

    # 报告
    report = {"config": dict(models=models, epochs=args.epochs, lr=args.lr,
                             batch=args.batch, seed=args.seed, smoke=args.smoke),
              "per_subject": per_subject}
    lines = ["LOSO 跨 subject 零校准评估（BCIC 2a 22ch 4分类）", "=" * 60,
             f"协议: train=8 subject T 会话, test=1 subject E 会话（零校准）, {len(SUBJECTS)} 折",
             f"模型: {models} | epochs={args.epochs} | seed={args.seed}", "-" * 60,
             f"{'subject':>8} " + " ".join(f"{m:>11}" for m in models)]
    for s in SUBJECTS:
        if s not in per_subject[models[0]]:
            continue
        lines.append(f"{s:>8} " + " ".join(f"{per_subject[m][s]:>10.2f}%" for m in models))
    lines.append("-" * 60)
    for m in models:
        arr = np.array(results[m]) * 100
        report[f"{m}_mean"] = round(float(arr.mean()), 2)
        report[f"{m}_std"] = round(float(arr.std()), 2)
        lines.append(f"{m:>8}: 平均 {arr.mean():.2f}% ± {arr.std():.2f}%  (9折 {arr.round(1).tolist()})")

    (ALGO_DIR / "loso_results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    (ALGO_DIR / "loso_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines))
    print("\n✓ 报告已保存：loso_results.json / loso_report.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
