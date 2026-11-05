#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_sleep_full.py — Sleep-EDF 睡眠分期全量评估（eval 集 subjects 4-5）

输出：
  - 总体准确率（chance 20% 对比）
  - 逐类 precision / recall / F1（W/N1/N2/N3/REM）
  - 混淆矩阵（行=真实，列=预测）
  - 端侧：参数量 + MACs（hook 解析法，同 CHB-MIT）
  - LIF：各激活位点平均发放率

用法
----
  env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin \\
      /opt/miniconda3/envs/openmmlab/bin/python eval_sleep_full.py \\
      --weights sleep_5class_lif_s42.pth [--activation lif] [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ALGO_DIR = Path("/Users/heyixuan/BCI/20260828/algorithm")
BASE = Path(__file__).resolve().parent
DATA = BASE / "data"

sys.path.insert(0, str(ALGO_DIR))
from eeg_tcnet_lif import EEGTCNetLIF  # noqa: E402

CLASS_NAMES = ["W", "N1", "N2", "N3", "REM"]


def _pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class MACsCounter:
    def __init__(self):
        self.macs: dict[str, int] = {}
        self.hooks = []

    def _mk_hook(self, name):
        def hook(module, inp, out):
            out_t = out if torch.is_tensor(out) else out[0]
            o = out_t.numel()
            if isinstance(module, nn.Conv2d):
                per = (module.in_channels // module.groups) * module.kernel_size[0] * module.kernel_size[1]
            elif isinstance(module, nn.Conv1d):
                per = (module.in_channels // module.groups) * module.kernel_size[0]
            elif isinstance(module, nn.Linear):
                per = module.in_features
            else:
                per = 0
            self.macs[name] = o * per
        return hook

    def attach(self, model: nn.Module):
        for name, m in model.named_modules():
            if isinstance(m, (nn.Conv2d, nn.Conv1d, nn.Linear)):
                self.hooks.append(m.register_forward_hook(self._mk_hook(name)))

    def detach(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=str, required=True)
    ap.add_argument("--activation", choices=["relu", "lif"], default="lif")
    ap.add_argument("--json", type=str, default="")
    ap.add_argument("--batch-size", type=int, default=512)
    args = ap.parse_args()

    device = _pick_device()
    ck = torch.load(BASE / args.weights, map_location="cpu", weights_only=False)
    model = EEGTCNetLIF(n_channels=1, n_classes=5, input_length=3000,
                        activation=args.activation)
    model.load_state_dict(ck["state_dict"], strict=True)
    model = model.to(device).eval()
    print(f"加载 {args.weights} | 激活 {args.activation} | 设备 {device}")
    print(f"参数量 : {model.num_params():,} | "
          f"MACs 估计(理论) : {model.macs_estimate()/1e6:.2f} M/epoch")

    # 端侧：hook 实测 MACs
    counter = MACsCounter()
    counter.attach(model)
    dummy = torch.zeros(1, 1, 1, 3000, device=device)
    with torch.no_grad():
        model(dummy)
    counter.detach()
    macs_hook = sum(counter.macs.values())
    print(f"MACs 实测(hook) : {macs_hook/1e6:.2f} M/epoch（30s 窗）")

    with np.load(DATA / "eval.npz") as d:
        X, y = d["X"], d["y"]
    fr_sum: dict = {}
    preds = np.zeros(len(y), np.int64)
    with torch.no_grad():
        for i in range(0, len(X), args.batch_size):
            xb = torch.from_numpy(X[i:i + args.batch_size]).to(device)
            logits = model(xb)
            preds[i:i + args.batch_size] = logits.argmax(1).cpu().numpy()
            for name, fr in model.firing_rates.items():
                fr_sum[name] = fr_sum.get(name, 0.0) + fr * xb.size(0)
    n = len(y)
    fr = {k: v / n for k, v in fr_sum.items()}

    acc = float((preds == y).mean())
    cm = np.zeros((5, 5), np.int64)
    for t, p in zip(y, preds):
        cm[t, p] += 1

    print(f"\n[final test] eval（跨被试 subjects 4-5, n={n}）: 准确率 {acc*100:.2f}% (chance 20%)")
    print(f"\n混淆矩阵（行=真实, 列=预测）:")
    print("        " + "  ".join(f"{c:>5s}" for c in CLASS_NAMES))
    for t in range(5):
        row = "  ".join(f"{v:5d}" for v in cm[t])
        print(f"  {CLASS_NAMES[t]:4s}  {row}  | {cm[t].sum():5d}")
    print(f"  合计  {sum(cm.sum(0)):5d}")
    print(f"\n逐类指标:")
    rows = []
    for c in range(5):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c].sum() - tp
        prec = tp / (tp + fp) if tp + fp else float("nan")
        rec = tp / (tp + fn) if tp + fn else float("nan")
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else float("nan")
        rows.append(dict(cls=CLASS_NAMES[c], n=cm[c].sum(), tp=int(tp),
                         precision=prec, recall=rec, f1=f1))
        print(f"  {CLASS_NAMES[c]:4s} n={cm[c].sum():5d}  prec={prec*100:5.2f}%  "
              f"rec={rec*100:5.2f}%  F1={f1*100:5.2f}%")
    bacc = float(np.mean([r["recall"] for r in rows]))
    kappa_num = acc - float((cm.sum(0) * cm.sum(1)).sum() / n**2)
    kappa_den = 1 - float((cm.sum(0) * cm.sum(1)).sum() / n**2)
    kappa = kappa_num / kappa_den if kappa_den else float("nan")
    print(f"\n平衡准确率(bacc) : {bacc*100:.2f}% | Cohen's κ : {kappa:.4f}")

    if args.activation == "lif" and fr:
        fr_str = "  ".join(f"{k.split('.')[-1]}={fr[k]*100:.1f}%" for k in sorted(fr))
        print(f"发放率(eval) : {fr_str}")

    if args.json:
        out = dict(weights=args.weights, activation=args.activation, n=n, acc=acc,
                   bacc=bacc, kappa=kappa, per_class=rows,
                   confusion_matrix=cm.tolist(), n_params=model.num_params(),
                   macs_theory=model.macs_estimate(), macs_hook=int(macs_hook),
                   firing_rates={k.split('.')[-1]: v for k, v in fr.items()},
                   seed=ck.get("seed"), epoch=ck.get("epoch"),
                   val_acc=ck.get("val_acc"), warm_start=ck.get("warm_start"))
        Path(args.json).write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                   encoding="utf-8")
        print(f"\n✓ {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
