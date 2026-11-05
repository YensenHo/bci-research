# -*- coding: utf-8 -*-
"""
train_lif_ninapro.py —— Ninapro DB5 52 类 sEMG 手势分类：ANN 基线 + surrogate LIF 化

目的
----
  验证「类脑化 sEMG 多通道多分类」的可行性：在 Ninapro DB5（16ch / 200Hz / 52 手势）
  上，用轻量全卷积 CNN（复用 finger_cnn.FingerCNN，类 EEGNet），训练：
    1) ANN 基线（activation=relu）
    2) LIF 化（activation=lif，surrogate gradient，热启动 ANN 权重再微调）
  报两者真实测试精度，量化「类脑化损失」= ANN acc − LIF acc。

复用关系（只读，不改动原文件）
  - finger_cnn.FingerCNN（本目录，n_channels/n_classes/input_length 可配）
  - lif_eegnet.LIFActivation（finger_cnn 内部引用，surrogate 可导 LIF）

数据
----
  由 preprocess_ninapro.py 产出 npz：X (N,1,16,256)、y (N,)∈0..51、rep (N,)∈1..6、
  subject (N,)∈1..10。本脚本支持两种划分：
    --split rep      : 重复留出（test=第6次重复，train=1..5 次重复内 15% 留验证）。干净、类内。
    --split subject  : 留一被试（train=subject 1..8，test=subject 9..10）。跨被试、更硬。
  默认 rep（类内、被试用池化），跨被试作为附加消融可另跑。

用法
----
  # 先预处理（见 preprocess_ninapro.py）产出 ninapro52.npz
  # 1) ANN 基线
  env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin \
      /opt/miniconda3/envs/openmmlab/bin/python train_lif_ninapro.py \
      --data ninapro52.npz --activation relu --epochs 30 --out ann_ninapro.pth --tag ann
  # 2) LIF 化（热启动 ANN 权重）
  env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin \
      /opt/miniconda3/envs/openmmlab/bin/python train_lif_ninapro.py \
      --data ninapro52.npz --activation lif --warm-start ann_ninapro.pth \
      --threshold 1.0 --tau 2.0 --timesteps 8 --epochs 30 --out lif_ninapro.pth --tag lif
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
from torch.utils.data import DataLoader, Dataset

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from finger_cnn import FingerCNN  # noqa: E402


# ---------------------------------------------------------------------------
# 数据集：读 preprocess_ninapro.py 产出的 npz，暴露 rep/subject 供划分
# ---------------------------------------------------------------------------
class NinaproNpzDataset(Dataset):
    def __init__(self, npz_path: Path):
        dat = np.load(str(npz_path), allow_pickle=True)
        self.X = torch.from_numpy(dat["X"].astype(np.float32))
        self.y = torch.from_numpy(dat["y"].astype(np.int64))
        self.rep = np.asarray(dat["rep"]) if "rep" in dat else np.zeros(len(self.y))
        self.subject = np.asarray(dat["subject"]) if "subject" in dat else np.zeros(len(self.y))
        self.n_channels = self.X.shape[2]
        self.input_length = self.X.shape[-1]
        self.n_classes = int(self.y.max().item()) + 1

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]

    def indices_by(self, key: str) -> dict:
        """按 rep 或 subject 分组，返回 {取值: 索引数组}。"""
        col = self.rep if key == "rep" else self.subject
        out = {}
        for v in np.unique(col):
            out[int(v)] = np.where(col == v)[0]
        return out


def _pick_device(no_cuda: bool) -> torch.device:
    if no_cuda:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _evaluate(model: FingerCNN, loader: DataLoader, device: torch.device):
    model.eval()
    correct = total = 0
    fr = {"act1": 0.0, "act2": 0.0}
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            correct += (logits.argmax(1) == yb).sum().item()
            total += xb.size(0)
            for k, act in (("act1", model.act1), ("act2", model.act2)):
                fr[k] += float(getattr(act, "last_firing_rate", 0.0)) * xb.size(0)
    return correct / total, {k: v / total for k, v in fr.items()}


def _make_model(args, n_channels: int, n_classes: int, input_length: int) -> FingerCNN:
    lif_kwargs = dict(threshold=args.threshold, tau=args.tau,
                      timesteps=args.timesteps, alpha=args.alpha)
    return FingerCNN(n_channels=n_channels, n_classes=n_classes, input_length=input_length,
                     F1=args.f1, D=args.d, F2=args.f2, kernel_length=args.kernel,
                     dropout=args.dropout, activation=args.activation, **lif_kwargs)


def main() -> int:
    ap = argparse.ArgumentParser(description="Ninapro DB5 52 类 sEMG 分类：ANN + LIF 化")
    ap.add_argument("--data", type=str, default="ninapro52.npz", help="preprocess_ninapro.py 产出 npz")
    ap.add_argument("--split", type=str, default="rep", choices=["rep", "subject"],
                    help="划分方式：rep=重复留出 / subject=留一被试")
    ap.add_argument("--activation", type=str, default="relu",
                    help="relu/elu（ANN 基线）/ lif（surrogate LIF）")
    ap.add_argument("--warm-start", type=str, default="", help="热启动权重（ANN 基线 .pth）")
    ap.add_argument("--threshold", type=float, default=1.0, help="LIF 发放阈值")
    ap.add_argument("--tau", type=float, default=2.0, help="膜时间常数")
    ap.add_argument("--timesteps", type=int, default=8, help="LIF 仿真时间步数")
    ap.add_argument("--alpha", type=float, default=4.0, help="surrogate sigmoid 斜率")
    ap.add_argument("--f1", type=int, default=8)
    ap.add_argument("--d", type=int, default=2)
    ap.add_argument("--f2", type=int, default=16)
    ap.add_argument("--kernel", type=int, default=63)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-split", type=float, default=0.15)
    ap.add_argument("--no-cuda", action="store_true")
    ap.add_argument("--out", type=str, default="ninapro_model.pth")
    ap.add_argument("--tag", type=str, default="")
    ap.add_argument("--json-out", type=str, default="", help="追加结果到 JSON")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = _pick_device(args.no_cuda)

    data_path = Path(args.data)
    if not data_path.exists():
        raise SystemExit(f"✗ 数据缺失：{data_path}。请先跑 preprocess_ninapro.py")

    full_ds = NinaproNpzDataset(data_path)
    n_channels, n_classes, input_length = full_ds.n_channels, full_ds.n_classes, full_ds.input_length
    print(f"数据     : {data_path.name} | {len(full_ds)} 窗 | input (1,{n_channels},{input_length}) "
          f"| n_classes={n_classes}")
    print(f"设备     : {device} | 激活 {args.activation} | 划分 {args.split} | 随机基线 {100.0/n_classes:.2f}%")

    # ---- 划分 ----
    if args.split == "rep":
        groups = full_ds.indices_by("rep")          # {rep: idx}
        reps = sorted(groups.keys())
        test_idx = groups.get(6, np.array([], dtype=int))
        trainval_idx = np.concatenate([groups[r] for r in reps if r != 6]).astype(int)
        desc = f"重复留出：train/val=重复{{1..5}}，test=重复6"
    else:
        groups = full_ds.indices_by("subject")
        subs = sorted(groups.keys())
        test_idx = np.concatenate([groups[s] for s in subs if s > 8]).astype(int)
        trainval_idx = np.concatenate([groups[s] for s in subs if s <= 8]).astype(int)
        desc = f"留一被试：train/val=subject{{1..8}}，test=subject{{9..10}}"

    rng = np.random.default_rng(args.seed)
    rng.shuffle(trainval_idx)
    n_val = max(1, int(len(trainval_idx) * args.val_split))
    val_idx = trainval_idx[:n_val]
    train_idx = trainval_idx[n_val:]
    print(f"划分     : {desc} | train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

    train_ds = torch.utils.data.Subset(full_ds, train_idx)
    val_ds = torch.utils.data.Subset(full_ds, val_idx)
    test_ds = torch.utils.data.Subset(full_ds, test_idx)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    # ---- 模型 ----
    model = _make_model(args, n_channels, n_classes, input_length)
    warm = False
    ws = Path(args.warm_start) if args.warm_start else None
    if ws and ws.exists():
        ck = torch.load(ws, map_location="cpu", weights_only=False)
        sd = ck["state_dict"] if "state_dict" in ck else ck
        model.load_state_dict(sd, strict=False)
        # 校验：除激活层外其余权重是否全命中
        miss_conv = [k for k in sd if not k.startswith(("act1", "act2"))
                     and k not in model.state_dict()]
        if miss_conv:
            print(f"[警告] 热启动有 {len(miss_conv)} 个非激活层键未命中：{miss_conv[:5]}")
        else:
            warm = True
            print(f"[热启动] 已加载 {ws.name} 的 conv/BN/fc 权重")
    model = model.to(device)
    print(f"参数量   : {model.num_params():,}")

    # ---- 对照：热启动未微调，直接在测试集测一次（量化「直接转换」损失）----
    if warm and args.activation == "lif":
        acc, fr = _evaluate(model, test_loader, device)
        print(f"\n[对照] ANN 权重直接换 LIF 激活（不微调）test : {acc*100:.2f}% "
              f"(发放率 act1={fr['act1']*100:.2f}% act2={fr['act2']*100:.2f}%)")

    # ---- 训练 ----
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    history = []
    best_acc, best_epoch, best_state, patience_counter = 0.0, -1, None, 0
    print(f"\n开始训练（activation={args.activation}）...")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        tl = tc = tn = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            tl += loss.item() * xb.size(0)
            tc += (logits.argmax(1) == yb).sum().item()
            tn += xb.size(0)
        train_loss, train_acc = tl / tn, tc / tn
        model.eval()
        vc = vn = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                vc += (model(xb).argmax(1) == yb).sum().item()
                vn += xb.size(0)
        val_acc = vc / vn
        dt = time.time() - t0
        history.append({"epoch": epoch, "train_loss": round(train_loss, 6),
                        "train_acc": round(train_acc, 4), "val_acc": round(val_acc, 4)})
        print(f"  Epoch {epoch:3d}/{args.epochs} | loss={train_loss:.4f} "
              f"train={train_acc*100:5.2f}% val={val_acc*100:5.2f}% | {dt:.1f}s")
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

    test_acc, fr = _evaluate(model, test_loader, device)
    print(f"\n[final test] {args.activation} : {test_acc*100:.2f}% "
          f"(发放率 act1={fr['act1']*100:.2f}% act2={fr['act2']*100:.2f}%)")

    # ---- 保存 ----
    out = Path(args.out)
    torch.save({
        "state_dict": best_state, "epoch": best_epoch, "val_acc": best_acc,
        "test_acc": test_acc, "firing_rate": fr,
        "activation": args.activation, "split": args.split,
        "lif_hparams": dict(threshold=args.threshold, tau=args.tau,
                            timesteps=args.timesteps, alpha=args.alpha),
        "model_args": dict(n_channels=n_channels, n_classes=n_classes,
                           input_length=input_length, F1=args.f1, D=args.d, F2=args.f2,
                           kernel_length=args.kernel, dropout=args.dropout,
                           activation=args.activation),
    }, out)
    print(f"✓ 权重已保存：{out}（epoch {best_epoch}，val={best_acc*100:.2f}% test={test_acc*100:.2f}%）")

    # ---- 报告 ----
    tag = args.tag or f"ninapro_{args.activation}_{args.split}"
    report = _HERE / f"ninapro_report_{tag}.txt"
    lines = [
        f"Ninapro DB5 52 类 sEMG 分类报告（activation={args.activation}, split={args.split}）",
        "=" * 60,
        f"tag             : {tag}",
        f"最佳验证准确率  : {best_acc*100:.2f}%（epoch {best_epoch}）",
        f"final test      : {test_acc*100:.2f}%（随机基线 {100.0/n_classes:.2f}%）",
        f"n_classes       : {n_classes}",
        f"n_channels      : {n_channels}",
        f"input_length    : {input_length}",
        f"参数量          : {model.num_params():,}",
        f"设备            : {device}",
        f"数据            : {data_path.name}",
        f"划分            : {desc}",
    ]
    if args.activation == "lif":
        lines += [
            f"LIF 超参        : threshold={args.threshold}, tau={args.tau}, "
            f"timesteps={args.timesteps}, alpha={args.alpha}",
            f"发放率          : act1={fr['act1']*100:.2f}% act2={fr['act2']*100:.2f}%",
        ]
    lines += [
        f"超参数          : lr={args.lr}, batch={args.batch_size}, epochs={args.epochs}, "
        f"patience={args.patience}, seed={args.seed}, F1={args.f1}, F2={args.f2}",
        "-" * 60,
        "逐 epoch 记录 (epoch, train_loss, train_acc, val_acc):",
    ]
    for h in history:
        lines.append(f"  {h['epoch']:3d}  {h['train_loss']:.6f}  "
                     f"{h['train_acc']*100:5.2f}%  {h['val_acc']*100:5.2f}%")
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  报告已写入：{report}")

    if args.json_out:
        rec = {"tag": tag, "activation": args.activation, "split": args.split,
               "threshold": args.threshold, "tau": args.tau, "timesteps": args.timesteps,
               "best_val_acc": best_acc, "test_acc": test_acc,
               "fr_act1": fr["act1"], "fr_act2": fr["act2"], "epoch": best_epoch}
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
