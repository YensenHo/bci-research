#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dc_eeg_extra.py — 补充灵敏度检查（不占主表配置）：Diehl-Cook WTA 引擎能否在 EEG
脉冲流上学到判别特征？主表 stdp_eeg.py 用批量近似引擎（topk 非真 WTA），此脚本用
与合成收敛完全相同的 DC-WTA 配方（每步 winner 独占 + 共享抑制 + 发放后即时行归一化
+ adaptive threshold）在 EEG 率/delta 编码上重跑同一 LDA 读出协议，隔离
「STDP 机制不足 vs 批量引擎近似损失」。

输入事件密度对齐：EEG 率/delta 编码 ~2 事件/步(44 源) 位于合成临界带
（hi=0.15 的 1.5 事件/步收敛、hi=0.3 的 3 事件/步失败的中间），DC 引擎若也学不到
→ 事件密度/信息瓶颈本身是主因（与输入脉冲编码报告的 41-43% 封顶一致）。

学习子集每类 300 trial（T 会话），冻结后 LDA 在 T 子集训、E 会话全量测。
结果 → results/eeg_dc_extra.json
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stdp_eeg import DATA_TRAIN, DATA_EVAL, RESULTS, N_SRC, N_OUT, encode_all  # noqa: E402
from stdp_synthetic_dc import DiehlCookNet  # noqa: E402


def train_one_ext(net: DiehlCookNet, s: np.ndarray, rng: np.random.Generator) -> int:
    """DC 引擎在外部脉冲 s (T, 44) 上学习一个 trial（与合成 train_one 同逻辑）。"""
    T = s.shape[0]
    V = np.zeros(N_OUT)
    pre_trace = np.zeros(N_SRC)
    post_trace = np.zeros(N_OUT)
    n_fire = 0
    for t in range(T):
        sp = s[t]
        pre_trace *= net.decay_pre
        post_trace *= net.decay_post
        if sp.any():
            dec = net.a_minus * (post_trace[:, None] * sp[None, :])
            net.W -= dec
            np.clip(net.W, 0.0, net.wmax, out=net.W)
        pre_trace[sp > 0] = 1.0
        V = net.decay_mem * V + sp @ net.W.T
        fire = V >= (net.thr_base + net.theta)
        if fire.any():
            winner = int(np.argmax(V))
            net.W[winner] += net.a_plus * pre_trace
            np.clip(net.W, 0.0, net.wmax, out=net.W)
            if net.norm_target is not None:
                l1 = net.W[winner].sum()
                if l1 > 1e-9:
                    net.W[winner] *= net.norm_target / l1
                    np.clip(net.W, 0.0, net.wmax, out=net.W)
            net.theta[winner] += net.dth
            V[winner] = 0.0
            post_trace[winner] = 1.0
            n_fire += 1
            V[np.arange(N_OUT) != winner] -= net.inh
        net.theta *= net.decay_theta
    return n_fire


def dc_forward(net: DiehlCookNet, spikes: np.ndarray) -> np.ndarray:
    """DC 引擎冻结前向：返回每 trial 每神经元发放次数 (n, N)。"""
    n = spikes.shape[0]
    out = np.zeros((n, N_OUT))
    for i in range(n):
        s = spikes[i]
        T = s.shape[0]
        V = np.zeros(N_OUT)
        cnt = np.zeros(N_OUT)
        for t in range(T):
            V = net.decay_mem * V + s[t] @ net.W.T
            fire = V >= (net.thr_base + net.theta)
            if fire.any():
                w = int(np.argmax(V))
                cnt[w] += 1
                V[w] = 0.0
                V[np.arange(N_OUT) != w] -= net.inh
        out[i] = cnt
    return out


def main() -> None:
    d_tr = np.load(DATA_TRAIN)
    d_te = np.load(DATA_EVAL)
    x_tr = d_tr["X"][:, 0].astype(np.float64)
    y_tr = d_tr["y"]
    x_te = d_te["X"][:, 0].astype(np.float64)
    y_te = d_te["y"]
    std_c = x_tr.std(axis=(0, 2))
    thetas = {"rate": 8.0 * std_c, "delta": 0.5 * std_c}

    # 学习子集：每类 300（类均衡，固定索引）
    idx_tr = np.concatenate([np.random.default_rng(5).permutation(np.nonzero(y_tr == c)[0])[:300]
                             for c in range(4)])
    x_lrn = x_tr[idx_tr]
    y_lrn = y_tr[idx_tr]

    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    out = {"task": "DC-WTA 引擎 EEG 灵敏度检查（补充，非主表）", "configs": []}
    for enc in ("rate", "delta"):
        for seed in (1, 2):
            tag = f"dc_{enc}_s{seed}"
            # DC 合成收敛配方；W 重设为 EEG 44 源维度 (128, 44)
            net = DiehlCookNet(hi=0.15, seed=seed, a_plus=0.008, a_minus=0.002)
            net.W = np.random.default_rng(seed).uniform(0.0, 0.3, size=(N_OUT, N_SRC))
            net.theta = np.zeros(N_OUT)
            t0 = time.time()
            # 类均衡交错逐 trial 无监督学习（1 epoch）
            order = []
            for i in range(300):
                for c in range(4):
                    order.append(idx_tr[c * 300 + i])
            for k, trial_i in enumerate(order):
                s = encode_all(x_tr[[trial_i]], enc, thetas[enc],
                               np.random.default_rng(seed * 1000 + k))
                train_one_ext(net, s[0], np.random.default_rng(seed * 1000 + k))
            # 特征（冻结）：T 子集训 LDA → E 会话全量测
            feats_tr = dc_forward(net, encode_all(x_lrn, enc, thetas[enc],
                                                  np.random.default_rng(seed + 31)))
            feats_te = dc_forward(net, encode_all(x_te, enc, thetas[enc],
                                                  np.random.default_rng(seed + 77)))
            clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
            clf.fit(feats_tr, y_lrn)
            te_acc = float((clf.predict(feats_te) == y_te).mean())
            l1 = float(net.W.sum(1).mean())
            print(f"[{tag}] T→E={te_acc*100:.1f}%  L1={l1:.2f}  fr_tr={feats_tr.mean():.2f}  "
                  f"({time.time()-t0:.0f}s)", flush=True)
            out["configs"].append({"tag": tag, "enc": enc, "seed": seed,
                                   "te_acc": te_acc, "train_s": round(time.time() - t0, 1)})
    with open(os.path.join(RESULTS, "eeg_dc_extra.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("\n== DC 补充完成 ==")


if __name__ == "__main__":
    main()
