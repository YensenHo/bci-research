#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
distill_4class.py — 蒸馏 4 类 22ch：大 EEG-TCNet teacher → 小 EEG-TCNet student（relu / LIF）

动机（承接 distill.py / distill_lif.py 的 8ch 二分类正结果）
----------------------------------------------------------------
  8ch 二分类已证：蒸馏红利集中在低标签预算（relu 学生 30% +5.02pp、LIF 学生 30% +5.10pp，
  满标签中性，双 seed 复现）。本脚本回答下一个问题——**蒸馏红利是否跨任务成立**：
  从「8ch 二分类（76% 基线）」升到「22ch 四分类（62.42% LIF / 63.58% HBC 增强基线，
  已被榨干、更接近真实可穿戴多电极场景）」。

teacher：大容量 EEG-TCNet（ELU，F1=16/D=2/F2=32/FT=24，约 3.7× student 参数），100% 标签训练。
student：EEG-TCNet-LIF（brief 配置 F1=8/D=2/F2=16/FT=12，9,512 参数，
  62.42% LIF 基线），activation 可配 relu / lif（对比口径与 8ch 的 relu/LIF 学生一致）。
loss = α·T²·KL(softmax(z_s/T)‖softmax(z_t/T)) + (1−α)·CE，T=4、α=0.5。

对比（唯一变量 = 是否用 teacher 软标签）
  对每个 label_frac ∈ {1.0, 0.3, 0.1}：hard-only vs distilled，双 seed（42/123）取均值。
  协议固定：100 epoch 无早停、cosine 退火、batch 16、lr 1e-3、dropout 0.25
  （对齐 8ch distill.py 的「固定 epoch 只变软标签」clean 口径）。

用法
----
  # 冒烟（relu 学生，2 epoch）
  env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin \
    /opt/miniconda3/envs/openmmlab/bin/python distill_4class.py --smoke --student relu --device mps
  # 全量 relu 学生（先跑，快）
  ... distill_4class.py --student relu --device mps
  # 全量 LIF 学生（后跑，慢）
  ... distill_4class.py --student lif --device mps
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ALGO_DIR = Path(__file__).resolve().parent
DATA_4CLASS = ALGO_DIR / "data_4class"
sys.path.insert(0, str(ALGO_DIR))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from eeg_tcnet import EEGTCNet          # noqa: E402  （teacher，ELU）
from eeg_tcnet_lif import EEGTCNetLIF   # noqa: E402  （student，relu/lif）

# brief 结构（对齐 train_eeg_tcnet_lif.py 默认：KE=63/KT=12/pool4/8/GAP）
BRIEF = dict(kernel_length=63, tcn_kernel_size=12, pool1=4, pool2=8, readout_mode="gap")


def _pick_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    if name in ("cuda", "auto") and torch.cuda.is_available():
        return torch.device("cuda")
    if name == "auto" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _build_teacher(n_channels: int, n_classes: int, input_length: int,
                   f1: int = 16, f2: int = 32, ft: int = 24) -> EEGTCNet:
    return EEGTCNet(n_channels=n_channels, n_classes=n_classes, input_length=input_length,
                    F1=f1, D=2, F2=f2, FT=ft, dropout=0.25,
                    activation="elu", **BRIEF)


def _build_student(activation: str, lif_kwargs: dict | None, n_channels: int,
                   n_classes: int, input_length: int) -> EEGTCNetLIF:
    return EEGTCNetLIF(n_channels=n_channels, n_classes=n_classes, input_length=input_length,
                       F1=8, D=2, F2=16, FT=12, dropout=0.25,
                       activation=activation, **(lif_kwargs or {}), **BRIEF)


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


def _train_hard(model, X, y, device, epochs, lr, batch, seed, cosine, eval_X, eval_y):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs) if cosine else None
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    loader = DataLoader(ds, batch_size=batch, shuffle=True)
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            opt.step()
        if sched is not None:
            sched.step()
    return _eval(model, eval_X, eval_y, device, batch)


def _train_distill(model, X, y, soft, device, epochs, lr, batch, seed, cosine, T, alpha,
                   eval_X, eval_y):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs) if cosine else None
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y), torch.from_numpy(soft))
    loader = DataLoader(ds, batch_size=batch, shuffle=True)
    for _ in range(epochs):
        model.train()
        for xb, yb, sb in loader:
            xb, yb, sb = xb.to(device), yb.to(device), sb.to(device)
            opt.zero_grad()
            z = model(xb)
            kl = F.kl_div(F.log_softmax(z / T, dim=1), sb, reduction="batchmean")
            loss = alpha * (kl * T * T) + (1.0 - alpha) * F.cross_entropy(z, yb)
            loss.backward()
            opt.step()
        if sched is not None:
            sched.step()
    return _eval(model, eval_X, eval_y, device, batch)


def main() -> int:
    ap = argparse.ArgumentParser(description="蒸馏 4 类 22ch：大 EEG-TCNet → 小 EEG-TCNet（relu/LIF）")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--student", type=str, default="relu", choices=["relu", "lif", "elu"])
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--cosine", action="store_true", default=True)
    ap.add_argument("--no-cosine", action="store_true")
    ap.add_argument("--temperature", type=float, default=4.0)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--label-fracs", type=str, default="1.0,0.3,0.1")
    ap.add_argument("--seeds", type=str, default="42,123")
    ap.add_argument("--n-channels", type=int, default=22)
    ap.add_argument("--n-classes", type=int, default=4)
    ap.add_argument("--input-length", type=int, default=500)
    ap.add_argument("--data-dir", type=str, default="data_4class")
    ap.add_argument("--teacher-f1", type=int, default=16)
    ap.add_argument("--teacher-f2", type=int, default=32)
    ap.add_argument("--teacher-ft", type=int, default=24)
    # LIF 超参（仅 --student lif 时生效）
    ap.add_argument("--threshold", type=float, default=1.0)
    ap.add_argument("--tau", type=float, default=2.0)
    ap.add_argument("--timesteps", type=int, default=8)
    ap.add_argument("--alpha-lif", type=float, default=4.0)
    args = ap.parse_args()

    if args.smoke:
        args.epochs = 2
        args.batch = 16
    cosine = (not args.no_cosine) and args.cosine

    device = _pick_device(args.device)
    data_dir = ALGO_DIR / args.data_dir
    train_npz = data_dir / "train.npz"
    eval_npz = data_dir / "eval.npz"
    for p in (train_npz, eval_npz):
        if not p.exists():
            raise SystemExit(f"✗ 数据集缺失：{p}（先跑 preprocess_4class.py）")
    tr = np.load(train_npz)
    ev = np.load(eval_npz)
    X_tr = tr["X"].astype(np.float32)
    y_tr = tr["y"].astype(np.int64)
    X_ev = ev["X"].astype(np.float32)
    y_ev = ev["y"].astype(np.int64)
    lif_kwargs = dict(threshold=args.threshold, tau=args.tau,
                      timesteps=args.timesteps, alpha=args.alpha_lif)
    print(f"设备 : {device} | 学生激活={args.student} | train {X_tr.shape} | eval {X_ev.shape}")

    fracs = [float(f) for f in args.label_fracs.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]
    all_results = {str(s): {} for s in seeds}

    for seed in seeds:
        print(f"\n{'=' * 60}\n seed {seed}\n{'=' * 60}")
        torch.manual_seed(seed)
        np.random.seed(seed)

        # ① teacher（大 ELU EEG-TCNet，100% 标签）
        teacher = _build_teacher(args.n_channels, args.n_classes, args.input_length,
                                 args.teacher_f1, args.teacher_f2, args.teacher_ft).to(device)
        t0 = time.time()
        teacher_acc = _train_hard(teacher, X_tr, y_tr, device, args.epochs, args.lr,
                                  args.batch, seed, cosine, X_ev, y_ev)
        print(f"  teacher eval : {teacher_acc*100:.2f}%（{time.time()-t0:.0f}s）")

        # ② teacher 软标签（全 train 集）
        teacher.eval()
        soft_all = []
        with torch.no_grad():
            for i in range(0, len(X_tr), args.batch):
                xb = torch.from_numpy(X_tr[i:i + args.batch]).to(device)
                soft_all.append(F.softmax(teacher(xb) / args.temperature, dim=1).cpu().numpy())
        soft_all = np.concatenate(soft_all)

        # ③ 各标签预算：hard vs distilled
        for frac in fracs:
            rng = np.random.default_rng(seed)
            n_lab = int(round(len(y_tr) * frac))
            idx = rng.permutation(len(y_tr))[:n_lab]
            X_sub, y_sub, soft_sub = X_tr[idx], y_tr[idx], soft_all[idx]

            s_hard = _build_student(args.student, lif_kwargs, args.n_channels, args.n_classes,
                                    args.input_length).to(device)
            acc_hard = _train_hard(s_hard, X_sub, y_sub, device, args.epochs, args.lr,
                                   args.batch, seed, cosine, X_ev, y_ev)

            s_dist = _build_student(args.student, lif_kwargs, args.n_channels, args.n_classes,
                                    args.input_length).to(device)
            acc_dist = _train_distill(s_dist, X_sub, y_sub, soft_sub, device, args.epochs,
                                      args.lr, args.batch, seed, cosine, args.temperature,
                                      args.alpha, X_ev, y_ev)

            gain = (acc_dist - acc_hard) * 100
            all_results[str(seed)][f"{frac:.0%}"] = {
                "n_labels": n_lab, "hard": round(acc_hard * 100, 2),
                "distilled": round(acc_dist * 100, 2), "gain_pp": round(gain, 2),
            }
            print(f"  {frac:.0%} 标签 ({n_lab:4d}): hard {acc_hard*100:6.2f}% → "
                  f"distilled {acc_dist*100:6.2f}% | 增益 {gain:+.2f}pp")

    # ④ 双 seed 聚合
    agg = {}
    for frac in [f"{f:.0%}" for f in fracs]:
        gains = [all_results[str(s)][frac]["gain_pp"] for s in seeds]
        hards = [all_results[str(s)][frac]["hard"] for s in seeds]
        dists = [all_results[str(s)][frac]["distilled"] for s in seeds]
        agg[frac] = {
            "hard_mean": round(np.mean(hards), 2), "hard": hards,
            "distilled_mean": round(np.mean(dists), 2), "distilled": dists,
            "gain_mean_pp": round(np.mean(gains), 2), "gain_pp": gains,
        }

    n_teacher = sum(p.numel() for p in _build_teacher(args.n_channels, args.n_classes,
                                                       args.input_length, args.teacher_f1,
                                                       args.teacher_f2, args.teacher_ft).parameters()
                    if p.requires_grad)
    n_student = sum(p.numel() for p in _build_student(args.student, lif_kwargs, args.n_channels,
                                                       args.n_classes, args.input_length).parameters()
                    if p.requires_grad)
    report = {
        "config": {"student": args.student, "teacher_params": n_teacher,
                   "student_params": n_student, "temperature": args.temperature,
                   "alpha": args.alpha, "epochs": args.epochs, "cosine": cosine,
                   "seeds": seeds, "smoke": args.smoke},
        "per_seed": all_results, "aggregated": agg,
    }
    suffix = "" if args.data_dir == "data_4class" else f"_{Path(args.data_dir).name}"
    if args.teacher_f1 != 16 or args.teacher_f2 != 32 or args.teacher_ft != 24:
        suffix += f"_tf{args.teacher_f1}-{args.teacher_f2}-{args.teacher_ft}"
    out_json = ALGO_DIR / f"distill_4class_{args.student}_report{suffix}.json"
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        f"蒸馏 4 类 22ch 报告（ELU teacher → {args.student} student）",
        "=" * 56,
        f"teacher : ELU EEG-TCNet {n_teacher:,} 参数（F1=16/D=2/F2=32/FT=24）",
        f"student : {args.student} EEG-TCNet {n_student:,} 参数（F1=8/D=2/F2=16/FT=12）",
        f"温度 T={args.temperature}，α={args.alpha}，epochs={args.epochs}，seeds={seeds}",
        "-" * 56,
        "标签预算 | hard均值 | distilled均值 | 蒸馏增益(mean)",
    ]
    for frac, r in agg.items():
        lines.append(f"  {frac:<8} {r['hard_mean']:>8.2f}%  {r['distilled_mean']:>8.2f}%  "
                     f"{r['gain_mean_pp']:+.2f}pp  (seeds {r['gain_pp']})")
    (ALGO_DIR / f"distill_4class_{args.student}_report{suffix}.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")

    print(f"\n✓ 报告已保存：distill_4class_{args.student}_report.txt / .json")
    for frac, r in agg.items():
        print(f"  {frac:<8} hard {r['hard_mean']:6.2f}% → distilled {r['distilled_mean']:6.2f}% "
              f"| 增益 {r['gain_mean_pp']:+.2f}pp (seeds {r['gain_pp']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
