#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_p300_avg.py — P300 叠加评估（单试次 vs 多轮叠加）与逐被试指标

目的
----
P300 拼写的核心机制是「目标闪烁诱发的小信号 + 多轮重复叠加提高 SNR」。
单试次（单次闪烁）分类弱是协议固有；本脚本用训练好的模型（relu/lif，
权重固定，eval 集只测一次的分析口径）量化「叠加轮数 → 精度」曲线：

  A. 单试次（逐闪烁）：主 eval.npz 均衡集（train_p300_lif.py 已报）之外，
     这里补逐被试（A05 / A06 各自均衡 700+700）口径。
  B. 多轮叠加（字符级）：每 trial（拼一个字符）= 12 个行/列码 × 10 轮重复。
     对同一 (trial, code) 的 n 轮 epoch 打分取平均 → 行得分(码1-6)与列得分
     (码7-12) → 行 argmax + 列 argmax = 预测字符。字符正确率 chance 1/36≈2.8%。
     曲线 n ∈ {1,2,3,5,10}。

数据
----
  data/eval_full.npz: X=(8400,1,8,205), y, subj(0/1=A05/A06), trial(0-34),
                      code(1-12), rep(0-9)（预处理期按时间序保存）

用法
----
  env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin \\
      /opt/miniconda3/envs/openmmlab/bin/python eval_p300_avg.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

ALGO_DIR = Path("/Users/heyixuan/BCI/20260828/algorithm")
BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
WOUT = BASE / "weights"
sys.path.insert(0, str(ALGO_DIR))
from eeg_tcnet_lif import EEGTCNetLIF  # noqa: E402

CHANCE_FLASH = 0.5
CHANCE_CHAR = 1.0 / 36.0      # 1/6 行 × 1/6 列


class NpzDS(Dataset):
    def __init__(self, npz: Path):
        with np.load(npz) as d:
            self.X = np.asarray(d["X"], np.float32)
            self.y = np.asarray(d["y"], np.int64)
            for k in ("subj", "trial", "code", "rep"):
                if k in d:
                    setattr(self, k, np.asarray(d[k]))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx])
        y = torch.tensor(int(self.y[idx]), dtype=torch.long)
        return x, y


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def predict_probs(model, ds, device, batch=256) -> np.ndarray:
    """返回 (n,) 目标类 softmax 概率。"""
    loader = DataLoader(ds, batch_size=batch, shuffle=False)
    model.eval()
    out = []
    with torch.no_grad():
        for xb, _ in loader:
            logits = model(xb.to(device))
            p = torch.softmax(logits, 1)[:, 1].cpu().numpy()
            out.append(p)
    return np.concatenate(out)


def main() -> int:
    device = pick_device()
    ds = NpzDS(DATA / "eval_full.npz")
    print(f"eval_full: X={ds.X.shape} y 分布 "
          f"{dict(zip(*np.unique(ds.y, return_counts=True)))} | device={device}")

    # 真值行/列：每 (subj, trial) 目标码（y==1）→ 行码(1-6)与列码(7-12)
    n_s = 2
    true_rowcol = {}      # (s, t) -> (row, col)
    for s in range(n_s):
        for t in range(35):
            codes = ds.code[(ds.subj == s) & (ds.trial == t) & (ds.y == 1)]
            rows = [int(c) for c in codes if c <= 6]
            cols = [int(c) for c in codes if c > 6]
            assert len(set(rows)) == 1 and len(set(cols)) == 1, (s, t, codes)
            true_rowcol[(s, t)] = (rows[0], cols[0])

    summary = {}
    for tag in ["relu_s42", "lif_s42", "lif_s123"]:
        pth = WOUT / f"p300_2class_{tag}.pth"
        if not pth.exists():
            print(f"[跳过] 缺 {pth.name}")
            continue
        ck = torch.load(pth, map_location="cpu", weights_only=False)
        ma = dict(ck["model_args"])
        lh = dict(ck.get("lif_hparams", {}))
        model = EEGTCNetLIF(**ma, **lh).to(device)
        model.load_state_dict(ck["state_dict"], strict=True)
        print(f"\n=== {tag}（n_params={ck['n_params']:,}）===")

        p = predict_probs(model, ds, device)

        # ---- A. 逐被试单试次均衡准确率 ----
        per_subj = {}
        for s in range(n_s):
            m = ds.subj == s
            yb = ds.y[m]
            pb = p[m]
            it = yb == 1
            inn = yb == 0
            # 均衡 700+700（所有目标 + 固定 seed 抽 700 非目标）
            rng = np.random.RandomState(42)
            pick = rng.choice(np.flatnonzero(inn), size=int(it.sum()), replace=False)
            sel = np.sort(np.concatenate([np.flatnonzero(it), pick]))
            acc = ((pb[sel] > 0.5).astype(int) == yb[sel]).mean()
            rec_t = (pb[it] > 0.5).mean()
            rec_n = (pb[inn] <= 0.5).mean()
            per_subj[f"A0{5 + s}"] = dict(acc=round(float(acc), 4),
                                          recall_target=round(float(rec_t), 4),
                                          recall_non_target=round(float(rec_n), 4),
                                          n=int(it.sum()))
            print(f"  被试 A0{5+s}: 单试次均衡 acc={acc*100:.2f}% "
                  f"(目标 recall {rec_t*100:.1f}% / 非目标 {rec_n*100:.1f}%)")

        # ---- B. 多轮叠加字符级 ----
        p_soft = p
        codes_all = sorted(range(1, 13))
        curve = {}
        for n_avg in [1, 2, 3, 5, 10]:
            n_char_ok = n_total = 0
            per_subj_char = {f"A0{5 + s}": [0, 0] for s in range(n_s)}
            for s in range(n_s):
                for t in range(35):
                    tr, tc = true_rowcol[(s, t)]
                    sc = {}
                    for c in codes_all:
                        sel = (ds.subj == s) & (ds.trial == t) & (ds.code == c) & (ds.rep < n_avg)
                        sc[c] = p_soft[sel].mean()
                    pred_row = max(range(1, 7), key=lambda c: sc[c])
                    pred_col = max(range(7, 13), key=lambda c: sc[c])
                    ok = (pred_row == tr and pred_col == tc)
                    n_char_ok += ok
                    n_total += 1
                    per_subj_char[f"A0{5 + s}"][0] += ok
                    per_subj_char[f"A0{5 + s}"][1] += 1
            curve[str(n_avg)] = dict(char_acc=round(n_char_ok / n_total, 4),
                                     n_chars=n_total,
                                     per_subj={k: round(v[0] / v[1], 4)
                                               for k, v in per_subj_char.items()})
            print(f"  叠加 {n_avg:2d} 轮: 字符级 acc = {n_char_ok}/{n_total} "
                  f"= {n_char_ok/n_total*100:.1f}% (chance {CHANCE_CHAR*100:.2f}%) "
                  f"| A05 {per_subj_char['A05'][0]}/35, A06 {per_subj_char['A06'][0]}/35")

        summary[tag] = dict(per_subj_single=per_subj, avg_curve=curve,
                            single_pooled=round(float(
                                ((p > 0.5).astype(int) == ds.y).mean()), 4))
    out = BASE / "reports" / "eval_avg_results.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n✓ {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
