#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_chbmit_full.py — CHB-MIT 全量评估（chb05 全部 39 个文件流式） + 端侧计算量

两部分：
  A. 全量评估（诚实临床视角）：chb05 全部窗口（含所有非发作窗口，不做均衡），
     输出 敏感度/特异度/精确率/F1/准确率/平衡准确率。流式处理，不落盘大数组。
  B. 端侧评估：参数量 + MACs（hook 解析法，同 macs_3ch_vs_22ch.py），
     23ch 二分类 @2s 窗 vs 22ch 四分类 @500（2a）vs 3ch 二分类 @500（2b）。

用法
----
  env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin \
      /opt/miniconda3/envs/openmmlab/bin/python eval_chbmit_full.py \
      --weights chbmit_2class_lif_s42.pth [--activation lif]
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
RAW = BASE / "raw"

sys.path.insert(0, str(ALGO_DIR))
from eeg_tcnet_lif import EEGTCNetLIF  # noqa: E402
sys.path.insert(0, str(BASE))
import preprocess_chbmit as pp  # noqa: E402  (read_edf / preprocess_file / windowize / load_annotations)


def _pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def full_eval(model, device, ann: dict, batch: int = 512) -> dict:
    """chb05 全部窗口流式评估。返回 sens/spec/ppv/f1/acc/bacc + 计数。"""
    model.eval()
    tp = fp = tn = fn = 0
    n_win = 0
    for fn_name in sorted(ann["chb05"].keys()):
        p = RAW / "chb05" / fn_name
        if not p.exists() or p.stat().st_size < 8_000_000:
            print(f"  [跳过] chb05/{fn_name}")
            continue
        X, _ = pp.read_edf(p)
        X = pp.preprocess_file(X)
        Xw, y = pp.windowize(X, ann["chb05"][fn_name])
        n_win += len(y)
        with torch.no_grad():
            for i in range(0, len(Xw), batch):
                xb = torch.from_numpy(Xw[i:i + batch])[:, None].to(device)
                logits = model(xb)
                pred = logits.argmax(1).cpu().numpy()
                yb = y[i:i + batch]
                tp += int(((pred == 1) & (yb == 1)).sum())
                fp += int(((pred == 1) & (yb == 0)).sum())
                tn += int(((pred == 0) & (yb == 0)).sum())
                fn += int(((pred == 0) & (yb == 1)).sum())
    sens = tp / (tp + fn) if tp + fn else float("nan")
    spec = tn / (tn + fp) if tn + fp else float("nan")
    ppv = tp / (tp + fp) if tp + fp else float("nan")
    acc = (tp + tn) / (tp + fp + tn + fn)
    bacc = (sens + spec) / 2
    f1 = 2 * ppv * sens / (ppv + sens) if ppv + sens else float("nan")
    return dict(n_windows=n_win, tp=tp, fp=fp, tn=tn, fn=fn,
                sensitivity=sens, specificity=spec, ppv=ppv, f1=f1,
                accuracy=acc, balanced_acc=bacc)


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

    def attach(self, model):
        for name, m in model.named_modules():
            if isinstance(m, (nn.Conv2d, nn.Conv1d, nn.Linear)):
                self.hooks.append(m.register_forward_hook(self._mk_hook(name)))
        return self

    def detach(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()

    def total(self):
        return sum(self.macs.values())


def edge_analysis():
    print("\n[端侧评估] EEG-TCNet（brief 配置）MACs / 参数量")
    rows = []
    for tag, n_ch, n_cls, T_in in [("CHB-MIT 23ch 二分类 @512", 23, 2, 512),
                                   ("2a 22ch 四分类 @500", 22, 4, 500),
                                   ("2b 3ch 二分类 @500", 3, 2, 500)]:
        m = EEGTCNetLIF(n_channels=n_ch, n_classes=n_cls, input_length=T_in,
                        activation="lif", threshold=1.0, tau=2.0, timesteps=8, alpha=4.0)
        counter = MACsCounter().attach(m)
        dummy = torch.randn(1, 1, n_ch, T_in)
        m.eval()
        with torch.no_grad():
            m(dummy)
        macs = counter.total()
        counter.detach()
        rows.append((tag, n_ch, n_cls, T_in, m.num_params(), macs))
        print(f"  {tag}: 参数={m.num_params():,} | 卷积/FC MACs={macs:,} ({macs/1e6:.3f}M)")
    r23, r22, r3 = rows
    print(f"\n  比值 vs 22ch 四分类: 参数 {r23[4]/r22[4]*100:.1f}% | "
          f"MACs {r23[5]/r22[5]*100:.1f}%（23ch 为其 {r22[5]/r23[5]:.2f} 分之 1 的倒数视角）")
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--activation", choices=["relu", "lif"], default="lif")
    ap.add_argument("--skip-full", action="store_true")
    args = ap.parse_args()

    device = _pick_device()
    ck = torch.load(Path(args.weights), map_location="cpu", weights_only=False)
    lif_kw = ck.get("lif_hparams", {})
    model = EEGTCNetLIF(n_channels=23, n_classes=2, input_length=512,
                        activation=args.activation, **lif_kw)
    model.load_state_dict(ck["state_dict"], strict=True)
    model.to(device).eval()
    print(f"模型 : {Path(args.weights).name}（{args.activation}，seed={ck.get('seed')}）| {device}")

    if not args.skip_full:
        ann = pp.load_annotations()
        print("\n[全量评估] chb05 全部窗口（真实类别分布，非均衡）...")
        r = full_eval(model, device, ann)
        print(f"  窗口数 : {r['n_windows']}（正 {r['tp']+r['fn']} / 负 {r['tn']+r['fp']}）")
        print(f"  TP={r['tp']} FP={r['fp']} TN={r['tn']} FN={r['fn']}")
        print(f"  敏感度(recall) = {r['sensitivity']*100:.2f}% | 特异度 = {r['specificity']*100:.2f}%")
        print(f"  精确率(PPV)    = {r['ppv']*100:.2f}% | F1 = {r['f1']*100:.2f}%")
        print(f"  准确率(全量)   = {r['accuracy']*100:.2f}% | 平衡准确率 = {r['balanced_acc']*100:.2f}%")
        (BASE / f"chbmit_full_eval_{Path(args.weights).stem}.json").write_text(
            json.dumps(r, indent=1, default=float), encoding="utf-8")

    edge_analysis()
    return 0


if __name__ == "__main__":
    sys.exit(main())
