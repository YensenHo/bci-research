#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stdp_eeg.py — EEG 脉冲流上的单层无监督 STDP 原型实验（BCIC IV 2a 四分类，T→E 跨会话）

问题（端侧无监督候选可行性）：
  连续 EEG 脉冲化后，单层 LIF + STDP 无监督学习能否学到类间判别特征？
  对比口径 = 同读出管线下的「STDP 学习权重」vs「随机权重」——STDP 特征若携带
  类间信息（准确率 > 随机权重基线且 > chance 25%），则无监督脉冲学习在 EEG 上
  有信号；否则如实记录为该原型的边界。

编码配置（双编码对照——率 vs 时序信息，都是 44 源 posneg 形态、~2 事件/步口径）：
  1) rate   : 幅度率编码。u_c(t)=clip(|x_c(t)|/θ_c,0,1) 泊松采样；x>0 → 正源、
              x<0 → 负源。θ_c = θ_k·std_c（train 集统计）。信息在「幅度/功率」。
  2) delta  : delta 调制。dx=x[t]-x[t-1]，|dx|>θ_c 发脉冲（正源 dx>θ、负源 dx<−θ）。
              θ_c = θ_k·std_c。信息在「变化事件/时序」。
  两者输出 (B, T, 44) 0/1：前 22 列 = ch0..21 正、后 22 列 = ch0..21 负。

学习与读出：
  - 无监督学习：train（T 会话）2592 试次按类均衡交错喂入单层 STDP
    （stdp_core.LIFSTDP：topk 类侧抑制 + adaptive threshold + L1 突触归一化，
    从合成实验继承的批量引擎参数族），epochs 可配。
  - 冻结 → 特征：每试次发放率向量 (n, N_out)。
  - 读出：LDA（sklearn，shrinkage）在 T 会话特征上训练 → E 会话（eval.npz）测试。
    4 类均衡 chance=25%。监督参考：ReLU 58.29% / delta 1bit 全脉冲 ~41-43%（见报告）。

运行（隔离环境）：
  env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin \
      /opt/miniconda3/envs/openmmlab/bin/python stdp_eeg.py          # 全量 4 配置
  同命令 + --smoke：1 配置、每类 60 学习试次 + 200 读出试次（快速冒烟）
结果 JSON → results/eeg_results.json（双 seed 取均值±std，含逐配置细节）
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stdp_core import LIFSTDP  # noqa: E402

ALGO = "/Users/heyixuan/BCI/20260828/algorithm"
DATA_TRAIN = os.path.join(ALGO, "data_4class", "train.npz")
DATA_EVAL = os.path.join(ALGO, "data_4class", "eval.npz")
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

N_CH = 22          # EEG 通道
N_SRC = 2 * N_CH   # posneg → 44 脉冲源
N_OUT = 128        # 学习层神经元数


# ---------------------------------------------------------------------------
# 脉冲编码（纯 numpy，posneg 44 源）
# ---------------------------------------------------------------------------
def rate_encode(x: np.ndarray, theta: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """幅度率编码：x (B,22,T) → (B,T,44)。u=|x|/θ 截断后泊松采样。"""
    B, C, T = x.shape
    u = np.clip(np.abs(x) / theta[:, None], 0.0, 1.0)   # (B,C,T)
    ev = rng.random(x.shape) < u
    out = np.zeros((B, T, 2 * C), dtype=np.float64)
    pos = ev & (x > 0)
    neg = ev & (x < 0)
    out[:, :, :C] = pos.transpose(0, 2, 1)
    out[:, :, C:] = neg.transpose(0, 2, 1)
    return out


def delta_encode(x: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """delta 调制：dx=x[t]-x[t-1] 过阈 → (B, T-1, 44) posneg。"""
    dx = np.diff(x, axis=2)                              # (B,C,T-1)
    pos = dx > theta[:, None]
    neg = dx < -theta[:, None]
    B, C, T1 = dx.shape
    out = np.zeros((B, T1, 2 * C), dtype=np.float64)
    out[:, :, :C] = pos.transpose(0, 2, 1)
    out[:, :, C:] = neg.transpose(0, 2, 1)
    return out


def encode_all(x_all: np.ndarray, enc: str, theta: np.ndarray,
               rng: np.random.Generator, batch: int = 256) -> np.ndarray:
    """分批编码全部 trial（内存友好）。x_all (n,22,T)。返回 (n, T', 44)。"""
    n = x_all.shape[0]
    out = []
    for b0 in range(0, n, batch):
        xb = x_all[b0:b0 + batch]
        if enc == "rate":
            out.append(rate_encode(xb, theta, rng))
        else:
            out.append(delta_encode(xb, theta))
    return np.concatenate(out, axis=0)


# ---------------------------------------------------------------------------
# 学习 / 前向 / 读出
# ---------------------------------------------------------------------------
def make_net(enc: str, seed: int, n_out: int = N_OUT) -> LIFSTDP:
    """批量引擎参数族（合成实验 r1 中权重稳定、不崩溃的配置；tau_minus 短窗防
    高频反因果抑制压死判别通道——合成 §3 的机制结论）。"""
    return LIFSTDP(N_SRC, n_out,
                   thr=1.0, tau_m=4.0, tau_plus=8.0, tau_minus=2.0,
                   a_plus=0.010, a_minus=0.004,
                   wmin=0.0, wmax=0.3, w_init=(0.0, 0.2), bias=0.0,
                   post_topk=1, thr_adapt=15.0,
                   norm_target=6.0, norm_eta=0.05,
                   rng=np.random.default_rng(seed))


def balanced_order(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """类均衡交错序：每类内部 shuffle 后按 0,1,2,3 循环排列（任一窗口类均衡）。"""
    classes = np.unique(y)
    perm = {c: rng.permutation(np.nonzero(y == c)[0]) for c in classes}
    n_per = min(len(v) for v in perm.values())
    order = []
    for i in range(n_per):
        for c in classes:
            order.append(perm[c][i])
    return np.asarray(order)


def firing_features(net: LIFSTDP, spikes: np.ndarray, batch: int = 128) -> np.ndarray:
    """冻结权重前向 → 每试次发放率 (n, N_out)。"""
    n = spikes.shape[0]
    feats = np.zeros((n, net.n_out))
    for b0 in range(0, n, batch):
        feats[b0:b0 + batch] = net.forward(spikes[b0:b0 + batch], batch=batch)
    return feats


def run_readout(feat_tr: np.ndarray, y_tr: np.ndarray,
                feat_te: np.ndarray, y_te: np.ndarray,
                seed: int) -> tuple[float, float, float]:
    """LDA 读出：T 训 → E 测。返回 (test_acc, train_acc, 每类均值acc最小值)。"""
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    clf.fit(feat_tr, y_tr)
    tr_acc = float((clf.predict(feat_tr) == y_tr).mean())
    te_pred = clf.predict(feat_te)
    te_acc = float((te_pred == y_te).mean())
    per_cls = [float(((te_pred == y_te) & (y_te == c)).mean()) for c in np.unique(y_te)]
    return te_acc, tr_acc, min(per_cls)


def run_config(enc: str, learn: bool, seed: int, x_tr: np.ndarray, y_tr: np.ndarray,
               x_te: np.ndarray, y_te: np.ndarray, theta: np.ndarray,
               epochs: int, smoke: bool) -> dict:
    rng = np.random.default_rng(1000 + seed)
    tag = f"{enc}_{'stdp' if learn else 'rand'}"
    t0 = time.time()

    # 无监督学习（learn=False → 随机权重基线，同分布不更新）
    net = make_net(enc, seed)
    if learn:
        order = balanced_order(y_tr, np.random.default_rng(seed + 7))
        for ep in range(epochs):
            for b0 in range(0, len(order), 128):
                idx = order[b0:b0 + 128]
                sp = encode_all(x_tr[idx], enc, theta,
                                np.random.default_rng(seed * 100 + ep * 31 + b0))
                net.train_trials(sp, learn=True, batch=128)
    else:
        # 同分布随机权重（未学习）——与 make_net 初始化一致的 uniform(0,0.2)
        net.set_weights(np.random.default_rng(seed + 99).uniform(0.0, 0.2, size=(N_OUT, N_SRC)))

    # 特征：T 会话（训练读出）+ E 会话（测试读出）
    feats_tr = np.zeros((len(y_tr), N_OUT))
    feats_te = np.zeros((len(y_te), N_OUT))
    for b0 in range(0, len(y_tr), 256):
        sp = encode_all(x_tr[b0:b0 + 256], enc, theta,
                        np.random.default_rng(seed + 555 + b0))
        feats_tr[b0:b0 + 256] = firing_features(net, sp)
    for b0 in range(0, len(y_te), 256):
        sp = encode_all(x_te[b0:b0 + 256], enc, theta,
                        np.random.default_rng(seed + 777 + b0))
        feats_te[b0:b0 + 256] = firing_features(net, sp)

    te_acc, tr_acc, min_cls = run_readout(feats_tr, y_tr, feats_te, y_te, seed)
    dt = time.time() - t0
    l1 = float(net.W.sum(axis=1).mean())
    fr = float(feats_tr.mean())
    print(f"[{tag}] seed={seed} | T→E acc={te_acc*100:.1f}%  train={tr_acc*100:.1f}%  "
          f"min_cls={min_cls*100:.1f}%  fr={fr:.2f}  L1={l1:.2f}  ({dt:.0f}s)", flush=True)
    return dict(tag=tag, enc=enc, learn=learn, seed=seed, te_acc=te_acc,
                tr_acc=tr_acc, min_cls=min_cls, train_s=round(dt, 1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seeds", type=str, default="1,2")
    ap.add_argument("--epochs", type=int, default=3)
    args = ap.parse_args()

    d_tr = np.load(DATA_TRAIN)
    d_te = np.load(DATA_EVAL)
    x_tr = d_tr["X"][:, 0].astype(np.float64)
    y_tr = d_tr["y"]
    x_te = d_te["X"][:, 0].astype(np.float64)
    y_te = d_te["y"]

    if args.smoke:
        keep_tr = np.concatenate([np.nonzero(y_tr == c)[0][:60] for c in range(4)])
        keep_te = np.concatenate([np.nonzero(y_te == c)[0][:50] for c in range(4)])
        x_tr, y_tr = x_tr[keep_tr], y_tr[keep_tr]
        x_te, y_te = x_te[keep_te], y_te[keep_te]

    # 逐通道阈值 θ_c = θ_k·std_c（train 集统计，train/eval 共用——与 delta_encode.py 一致）
    std_c = x_tr.std(axis=(0, 2))
    theta_rate = 8.0 * std_c       # 率编码：~0.8·std → 事件/步口径 ≈ delta 同量级
    theta_delta = 0.5 * std_c      # delta：k=0.5（输入脉冲编码报告：13.9% 密度口径）

    seeds = [int(s) for s in args.seeds.split(",")]
    os.makedirs(RESULTS, exist_ok=True)
    out = {"task": "EEG STDP 无监督 vs 随机（T→E）", "n_tr": len(y_tr),
           "n_te": len(y_te), "theta_rate_k": 8.0, "theta_delta_k": 0.5,
           "configs": []}

    # 密度统计（率编码随机种子影响小，固定 42）
    for enc, th in (("rate", theta_rate), ("delta", theta_delta)):
        sp = encode_all(x_tr[:256], enc, th, np.random.default_rng(42))
        out.setdefault("density", {})[enc] = round(float(sp.mean()), 4)

    # 4 配置：编码 × {STDP 学习, 随机权重}（均跑全部 seed）
    for enc in ("rate", "delta"):
        for learn in (True, False):
            entry = {"tag": f"{enc}_{'stdp' if learn else 'rand'}", "seeds": {}}
            for seed in seeds:
                r = run_config(enc, learn, seed, x_tr, y_tr, x_te, y_te,
                               theta_rate if enc == "rate" else theta_delta,
                               args.epochs, args.smoke)
                entry["seeds"][str(seed)] = r
            accs = [e["te_acc"] for e in entry["seeds"].values()]
            entry["te_acc_mean"] = float(np.mean(accs))
            entry["te_acc_std"] = float(np.std(accs))
            out["configs"].append(entry)

    with open(os.path.join(RESULTS, "eeg_results.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n== EEG 汇总（density: {out.get('density')}）==")
    for c in out["configs"]:
        print(f"  {c['tag']:12s} T→E = {c['te_acc_mean']*100:.1f}% ± {c['te_acc_std']*100:.1f}")


if __name__ == "__main__":
    main()
