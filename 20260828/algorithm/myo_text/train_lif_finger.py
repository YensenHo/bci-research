# -*- coding: utf-8 -*-
"""
train_lif_finger.py —— 训练 8 指 sEMG 分类 CNN：ANN 基线 + surrogate LIF 化

目的
----
  MyoText 手指分类用 CNN-BiLSTM-Attention（85.45%），类脑化难（循环+注意力）。
  本脚本训练**轻量全卷积 CNN**（finger_cnn.FingerCNN，类 EEGNet），并支持把
  激活层切换为 surrogate gradient 直接训练的 LIF（复用 lif_eegnet.LIFActivation），
  走「ANN 基线 → LIF 化」两步，服务类脑 + 神经接口 + HBC 纲领。

关键设计
--------
  - activation 可配：'relu' / 'elu' 产出 ANN 基线；'lif' 产出 LIF 化（surrogate）。
  - LIF 超参：threshold / tau（decay=exp(-1/tau)）/ timesteps / alpha 均作为命令行参数。
  - 热启动：--warm-start 加载 ANN 基线（relu/elu）的 conv/BN/fc 权重；
    LIF 激活层无参数，state_dict 键零回归，strict=True 直接载入。
  - hold-out 早停：训练集划 15% 验证，final 用独立测试集（eval）只测一次。
  - 报告发放率（act1/act2），诊断 LIF 的量化程度。

数据
----
  - 输入 npz 由 preprocess_emg.py 生成：X (N,1,32,T)、y (N,)（8 指）。
  - 无 npz 时可用 --mock 生成合成数据跑通管线（精度仅证明管线可跑，非真实精度）。

用法
----
  # 冒烟：合成数据上跑 ANN 基线 + LIF 化（各几个 epoch，验证管线）
  env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin \\
      /opt/miniconda3/envs/openmmlab/bin/python train_lif_finger.py \\
      --mock --activation relu --epochs 3 --out ann_baseline.pth --tag smoke_ann
  env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin \\
      /opt/miniconda3/envs/openmmlab/bin/python train_lif_finger.py \\
      --mock --activation lif --warm-start ann_baseline.pth --threshold 1.0 \\
      --tau 2.0 --timesteps 8 --epochs 3 --out lif_finger.pth --tag smoke_lif

  # 真实数据（需先下载 emg2qwerty + 预处理，见复现说明.md）
  env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin \\
      /opt/miniconda3/envs/openmmlab/bin/python train_lif_finger.py \\
      --data finger.npz --activation relu --epochs 50 --out ann_baseline.pth
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
from preprocess_emg import make_mock_npz  # noqa: E402


# ---------------------------------------------------------------------------
# 数据集：读 preprocess_emg.py 产出的 npz
# ---------------------------------------------------------------------------
class FingerNpzDataset(Dataset):
    """把 (N,1,32,T) npz 封装成 Dataset，返回 (X (1,32,T) float32, y int)。"""

    def __init__(self, npz_path: Path):
        # allow_pickle=True：npz 里存了 finger_names/keys 等 object 数组（字符串）
        dat = np.load(str(npz_path), allow_pickle=True)
        self.X = torch.from_numpy(dat["X"].astype(np.float32))
        self.y = torch.from_numpy(dat["y"].astype(np.int64))
        self.n_classes = int(self.y.max().item()) + 1
        self.finger_names = list(dat.get("finger_names", [str(i) for i in range(self.n_classes)]))

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


def _pick_device(no_cuda: bool) -> torch.device:
    if no_cuda:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _evaluate(model: FingerCNN, loader: DataLoader, device: torch.device):
    """返回 (acc, 平均发放率 dict)。lif 模式下从 act1/act2 读发放率。"""
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


def _make_model(args, n_classes: int, input_length: int) -> FingerCNN:
    lif_kwargs = dict(threshold=args.threshold, tau=args.tau,
                      timesteps=args.timesteps, alpha=args.alpha)
    return FingerCNN(n_channels=32, n_classes=n_classes, input_length=input_length,
                     F1=8, D=2, F2=16, kernel_length=63, dropout=0.5,
                     activation=args.activation, **lif_kwargs)


def main() -> int:
    ap = argparse.ArgumentParser(description="8 指 sEMG 分类：ANN 基线 + LIF 化")
    ap.add_argument("--data", type=str, default="", help="npz 路径（preprocess_emg.py 产出）")
    ap.add_argument("--mock", action="store_true", help="无数据时生成合成数据跑通管线")
    ap.add_argument("--activation", type=str, default="lif",
                    help="激活：relu / elu（ANN 基线）/ lif（surrogate LIF 化）")
    ap.add_argument("--warm-start", type=str, default="", help="热启动权重路径（ANN 基线 .pth）")
    ap.add_argument("--threshold", type=float, default=1.0, help="LIF 发放阈值")
    ap.add_argument("--tau", type=float, default=2.0, help="膜时间常数（decay=exp(-1/tau)）")
    ap.add_argument("--timesteps", type=int, default=8, help="LIF 仿真时间步数")
    ap.add_argument("--alpha", type=float, default=4.0, help="surrogate sigmoid 斜率")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-split", type=float, default=0.15)
    ap.add_argument("--no-cuda", action="store_true")
    ap.add_argument("--out", type=str, default="finger_model.pth")
    ap.add_argument("--tag", type=str, default="")
    ap.add_argument("--json-out", type=str, default="", help="追加结果到 JSON（消融汇总）")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = _pick_device(args.no_cuda)

    # ---- 数据准备 ----
    data_path = Path(args.data) if args.data else (_HERE / "mock_finger.npz")
    if args.mock or (not args.data and not data_path.exists()):
        data_path = _HERE / "mock_finger.npz"
        print("[mock] 生成合成数据（非真实数据，仅用于管线冒烟）")
        make_mock_npz(data_path, n_per_class=40, T=2000, seed=args.seed)
    if not data_path.exists():
        raise SystemExit(f"✗ 数据缺失：{data_path}。请先跑 preprocess_emg.py 或加 --mock")

    full_ds = FingerNpzDataset(data_path)
    n_classes = full_ds.n_classes
    input_length = full_ds.X.shape[-1]
    print(f"数据     : {data_path.name} | {len(full_ds)} 样本 | "
          f"input (1,32,{input_length}) | n_classes={n_classes}")
    print(f"设备     : {device} | 激活 {args.activation} | 随机基线 {100.0/n_classes:.1f}%")

    # ---- 划分：train(含早停验证) / test ----
    n_test = max(1, int(len(full_ds) * 0.15))
    n_rest = len(full_ds) - n_test
    g = torch.Generator().manual_seed(args.seed)
    rest_ds, test_ds = torch.utils.data.random_split(
        full_ds, [n_rest, n_test], generator=g)
    n_val = max(1, int(n_rest * args.val_split))
    n_train = n_rest - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        rest_ds, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed + 1))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
    print(f"划分     : {n_train} 训练 / {n_val} 验证（早停） / {n_test} 测试（只测一次）")

    # ---- 模型 ----
    model = _make_model(args, n_classes, input_length)
    warm = False
    ws = Path(args.warm_start) if args.warm_start else None
    if ws and ws.exists():
        ck = torch.load(ws, map_location="cpu", weights_only=False)
        sd = ck["state_dict"] if "state_dict" in ck else ck
        missing, unexpected = model.load_state_dict(sd, strict=False)
        # 热启动允许 activation 层参数不一致（lif 无参数 / relu 无参数），其余应全命中
        conv_keys = [k for k in sd if not k.startswith(("act1", "act2"))]
        miss_conv = [k for k in missing if not k.startswith(("act1", "act2"))]
        if miss_conv:
            print(f"[警告] 热启动有 {len(miss_conv)} 个非激活层键未命中，前几个：{miss_conv[:5]}")
        else:
            warm = True
            print(f"[热启动] 已加载 {ws.name} 的 conv/BN/fc 权重（激活层键可忽略）")
    model = model.to(device)
    print(f"参数量   : {model.num_params():,}")

    # ---- 对照：热启动未微调，直接在测试集上的准确率 ----
    if warm and args.activation == "lif":
        acc, fr = _evaluate(model, test_loader, device)
        print(f"\n[对照] ANN 权重直接换 LIF 激活（不微调）测试集 : {acc*100:.2f}% "
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

    # ---- final test（只测一次）----
    test_acc, fr = _evaluate(model, test_loader, device)
    print(f"\n[final test] {args.activation} : {test_acc*100:.2f}% "
          f"(发放率 act1={fr['act1']*100:.2f}% act2={fr['act2']*100:.2f}%)")

    # ---- 保存 ----
    out = Path(args.out)
    torch.save({
        "state_dict": best_state, "epoch": best_epoch, "val_acc": best_acc,
        "test_acc": test_acc, "firing_rate": fr,
        "activation": args.activation,
        "lif_hparams": dict(threshold=args.threshold, tau=args.tau,
                            timesteps=args.timesteps, alpha=args.alpha),
        "model_args": dict(n_channels=32, n_classes=n_classes, input_length=input_length,
                           F1=8, D=2, F2=16, kernel_length=63, dropout=0.5,
                           activation=args.activation),
    }, out)
    print(f"✓ 权重已保存：{out}（epoch {best_epoch}，val={best_acc*100:.2f}% test={test_acc*100:.2f}%）")

    # ---- 报告 ----
    tag = args.tag or f"{args.activation}_tau{args.tau}_T{args.timesteps}"
    report = _HERE / f"finger_report_{tag}.txt"
    lines = [
        f"8 指 sEMG 分类报告（activation={args.activation}）",
        "=" * 60,
        f"tag             : {tag}",
        f"最佳验证准确率  : {best_acc*100:.2f}%（epoch {best_epoch}）",
        f"final test      : {test_acc*100:.2f}%（随机基线 {100.0/n_classes:.1f}%）",
        f"n_classes       : {n_classes}",
        f"参数量          : {model.num_params():,}",
        f"设备            : {device}",
        f"数据            : {data_path.name}（{'mock 合成数据' if (args.mock or not args.data) else '真实数据'}）",
    ]
    if args.activation == "lif":
        lines += [
            f"LIF 超参        : threshold={args.threshold}, tau={args.tau}, "
            f"timesteps={args.timesteps}, alpha={args.alpha}",
            f"发放率          : act1={fr['act1']*100:.2f}% act2={fr['act2']*100:.2f}%",
        ]
    lines += [
        f"超参数          : lr={args.lr}, batch={args.batch_size}, epochs={args.epochs}, "
        f"patience={args.patience}, seed={args.seed}",
        "-" * 60,
        "逐 epoch 记录 (epoch, train_loss, train_acc, val_acc):",
    ]
    for h in history:
        lines.append(f"  {h['epoch']:3d}  {h['train_loss']:.6f}  "
                     f"{h['train_acc']*100:5.2f}%  {h['val_acc']*100:5.2f}%")
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  报告已写入：{report}")

    if args.json_out:
        rec = {"tag": tag, "activation": args.activation,
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
