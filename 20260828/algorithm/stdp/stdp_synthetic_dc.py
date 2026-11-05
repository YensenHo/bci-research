#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stdp_synthetic_dc.py — Diehl-Cook 2015 WTA 兜底配方（合成收敛实验，独立实现）

为什么独立于 stdp_core.LIFSTDP：stdp_core 用「批量并行 + 批末一次性 norm」，
无法表达 Diehl-Cook 的「winner 发放当刻：其余神经元被共享抑制 + 该神经元权重行
立即归一化」。本文件逐 pattern 串行仿真，忠实复刻文献配方：

  ① LIF 积分（tau_m 指数衰减 + 硬重置）
  ② 每步至多 1 个 winner（V 最高且过阈者发放）；winner 当步把同 trial 其它神经元
     V 压到 -inh（共享抑制池语义 → WTA，阻止多头竞争）
  ③ STDP：post(winner) 发放 → 增强 pre_trace 高的输入（因果）；
     pre 发放且 post_trace 高 → 抑制（反因果）
  ④ adaptive threshold：每次发放 θ_i += dth，每步 θ_i *= decay_theta（发放多者
     让位 → 两模式都分到 winner，公平性）
  ⑤ 每次 winner 发放后其权重行立即乘性归一化到 L1=norm_target（总量受限 →
     神经元在通道间取舍）

输入口径：Diehl-Cook 的 MNIST 输入是稀疏泊松（像素发放率低、事件间隔 ≈ τ 量级）。
任务合成最初口径 hi=0.5（每步平均 5 事件/模式通道，间隔 0.2 步 ≪ τ）会让 STDP
时间窗被淹没——本脚本 hi 可配，验证「事件密度 vs 膜时间常数」匹配是 STDP 可学性
前提（hi=0.5 对照失败、hi=0.15 收敛，详见 STDP原型报告.md §合成验证）。

运行：
  env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin \
      /opt/miniconda3/envs/openmmlab/bin/python stdp_synthetic_dc.py
结果 JSON → results/synthetic_dc_results.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

C, N, T = 40, 64, 100
LO = 0.03
A_CH = np.arange(0, 10)
B_CH = np.arange(20, 30)
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


class DiehlCookNet:
    """单层 N 神经元 + 共享抑制池 WTA + STDP + adaptive threshold + 发放后即时归一化。"""

    def __init__(self, *, hi: float, thr: float = 1.0, tau_m: float = 20.0,
                 tau_plus: float = 20.0, tau_minus: float = 20.0,
                 a_plus: float = 0.004, a_minus: float = 0.003,
                 dth: float = 0.1, decay_theta: float = 0.995,
                 w_init: tuple[float, float] = (0.0, 0.3),
                 wmax: float = 1.0, norm_target: float | None = 6.0,
                 inh: float = 2.0, seed: int = 1):
        self.hi = hi
        self.thr_base = float(thr)
        self.decay_mem = float(np.exp(-1.0 / tau_m))
        self.decay_pre = float(np.exp(-1.0 / tau_plus))
        self.decay_post = float(np.exp(-1.0 / tau_minus))
        self.a_plus = float(a_plus)
        self.a_minus = float(a_minus)
        self.dth = float(dth)
        self.decay_theta = float(decay_theta)
        self.wmax = float(wmax)
        self.norm_target = norm_target
        self.inh = float(inh)
        rng = np.random.default_rng(seed)
        lo, hi_ = w_init
        self.W = rng.uniform(lo, hi_, size=(N, C))
        self.theta = np.zeros(N)
        self.rate_ema = np.zeros(N)

    def _spikes(self, mode: int, rng: np.random.Generator) -> np.ndarray:
        p = np.full(C, LO)
        p[A_CH if mode == 0 else B_CH] = self.hi
        return (rng.random((T, C)) < p).astype(np.float64)

    def _forward(self, s: np.ndarray) -> np.ndarray:
        """冻结权重前向（不学习）：返回每神经元发放次数。与训练同 WTA 语义。"""
        V = np.zeros(N)
        cnt = np.zeros(N)
        for t in range(T):
            V = self.decay_mem * V + s[t] @ self.W.T
            fire = V >= (self.thr_base + self.theta)
            if fire.any():
                w = int(np.argmax(V))
                cnt[w] += 1
                V[w] = 0.0
                others = np.arange(N) != w
                V[others] -= self.inh
        return cnt

    def train_one(self, mode: int, rng: np.random.Generator) -> int:
        s = self._spikes(mode, rng)
        V = np.zeros(N)
        pre_trace = np.zeros(C)
        post_trace = np.zeros(N)
        n_fire = 0
        for t in range(T):
            sp = s[t]
            pre_trace *= self.decay_pre
            post_trace *= self.decay_post
            if sp.any():                      # 反因果抑制
                dec = self.a_minus * (post_trace[:, None] * sp[None, :])
                self.W -= dec
                np.clip(self.W, 0.0, self.wmax, out=self.W)
            pre_trace[sp > 0] = 1.0
            V = self.decay_mem * V + sp @ self.W.T
            fire = V >= (self.thr_base + self.theta)
            if fire.any():
                winner = int(np.argmax(V))    # WTA：最高膜电位者发放
                self.W[winner] += self.a_plus * pre_trace
                np.clip(self.W, 0.0, self.wmax, out=self.W)
                if self.norm_target is not None:   # 发放后立即行归一化
                    l1 = self.W[winner].sum()
                    if l1 > 1e-9:
                        self.W[winner] *= self.norm_target / l1
                        np.clip(self.W, 0.0, self.wmax, out=self.W)
                self.theta[winner] += self.dth    # adaptive threshold：让位
                V[winner] = 0.0
                post_trace[winner] = 1.0
                n_fire += 1
                V[np.arange(N) != winner] -= self.inh   # 共享抑制
            self.theta *= self.decay_theta
        self.rate_ema = 0.99 * self.rate_ema + 0.01 * (n_fire / T)
        return n_fire

    def train(self, n_per: int, epochs: int, seed: int) -> None:
        rng = np.random.default_rng(seed + 500)
        for _ in range(epochs):
            for _ in range(n_per):
                self.train_one(0, rng)
                self.train_one(1, rng)

    def eval(self, seed: int, n_test: int = 120) -> dict:
        rng = np.random.default_rng(seed + 900)
        frA = np.zeros(N)
        frB = np.zeros(N)
        for _ in range(n_test):
            frA += self._forward(self._spikes(0, rng))
            frB += self._forward(self._spikes(1, rng))
        frA /= n_test * T
        frB /= n_test * T
        wA = self.W[:, A_CH].mean(1)
        wB = self.W[:, B_CH].mean(1)
        dW = wA - wB
        dF = frA - frB
        nA_w = int((dW > 0.05).sum())
        nB_w = int((dW < -0.05).sum())
        nA_f = int((dF > 2e-3).sum())
        nB_f = int((dF < -2e-3).sum())
        corr = float(np.corrcoef(frA, frB)[0, 1])
        a_only = int(((frA > 0.2) & (frB < 0.05)).sum())
        b_only = int(((frB > 0.2) & (frA < 0.05)).sum())
        both = int(((frA > 0.2) & (frB > 0.2)).sum())
        conv = (max(nA_w, nB_w) >= 0.25 * N and nA_w + nB_w >= 0.4 * N
                and max(nA_f, nB_f) >= 0.15 * N)
        return dict(nA_w=nA_w, nB_w=nB_w, nA_f=nA_f, nB_f=nB_f, corr=corr,
                    conv=conv, a_only=a_only, b_only=b_only, both=both,
                    wAch=float(wA.mean()), wBch=float(wB.mean()),
                    frA_mean=float(frA.mean()), frB_mean=float(frB.mean()))


def run_cfg(cfg: dict, seed: int, epochs: int = 3, n_per: int = 150) -> tuple[dict, float]:
    kw = {k: v for k, v in cfg.items() if k != "tag"}
    net = DiehlCookNet(seed=seed, **kw)
    t0 = time.time()
    net.train(n_per, epochs, seed)
    ev = net.eval(seed)
    ev["train_s"] = round(time.time() - t0, 1)
    return ev, net


CONFIGS = [
    # 第 2 轮（Diehl-Cook WTA 兜底）：hi 为输入口径（模式通道发放率），LO=0.03
    dict(tag="dc_hi0.15_sparse_ap8am2", hi=0.15, a_plus=0.008, a_minus=0.002),
    dict(tag="dc_hi0.15_sparse_ap4am3", hi=0.15, a_plus=0.004, a_minus=0.003),
    dict(tag="dc_hi0.15_sparse_ap8am2_dth05", hi=0.15, a_plus=0.008, a_minus=0.002, dth=0.05),
    dict(tag="dc_hi0.3_mid_ap8am2", hi=0.3, a_plus=0.008, a_minus=0.002),
    dict(tag="dc_hi0.5_hi_ap8am2", hi=0.5, a_plus=0.008, a_minus=0.002),
    dict(tag="dc_hi0.15_sparse_ap8am0", hi=0.15, a_plus=0.008, a_minus=0.0),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg-idx", type=int, default=None)
    ap.add_argument("--seeds", type=str, default="1,2,3")
    ap.add_argument("--epochs", type=int, default=3)
    args = ap.parse_args()

    os.makedirs(RESULTS, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(",")]
    cfgs = CONFIGS if args.cfg_idx is None else [CONFIGS[args.cfg_idx]]

    out = {"task": "synthetic A/B 分化 — Diehl-Cook WTA 兜底", "configs": []}
    for cfg in cfgs:
        entry = {"tag": cfg["tag"], "cfg": cfg, "seeds": {}}
        for seed in seeds:
            ev, _ = run_cfg(cfg, seed, epochs=args.epochs)
            entry["seeds"][str(seed)] = ev
            print(f"[{cfg['tag']}] seed={seed} | wA/B={ev['nA_w']}/{ev['nB_w']} "
                  f"fA/B={ev['nA_f']}/{ev['nB_f']} corr={ev['corr']:.3f} "
                  f"A-only/B-only={ev['a_only']}/{ev['b_only']} both={ev['both']} "
                  f"conv={ev['conv']} ({ev['train_s']}s)", flush=True)
        entry["conv_seeds"] = sum(1 for e in entry["seeds"].values() if e["conv"])
        entry["conv"] = entry["conv_seeds"] == len(seeds)
        out["configs"].append(entry)
    with open(os.path.join(RESULTS, "synthetic_dc_results.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    n_conv = sum(1 for c in out["configs"] if c["conv"])
    print(f"\n== DC-WTA 汇总：{n_conv}/{len(out['configs'])} 配置收敛（全 seed 判据）==")


if __name__ == "__main__":
    main()
