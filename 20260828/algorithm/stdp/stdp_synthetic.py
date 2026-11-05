#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stdp_synthetic.py — 合成收敛验证：两正交输入模式 A/B 下，单层 STDP 能否自发分化。

任务设计（可复现）：
  - C=40 输入通道；模式 A = ch0..9 高频泊松（p=0.5/步）；模式 B = ch20..29 高频（p=0.5/步）；
    其余通道低频背景 p=0.03/步。T=100 步/trial。A/B trial 等量交替（类平衡、先 shuffle）。
  - 学习层 N=64 个 LIF 神经元，全连接 W (N, C)。STDP + 竞争机制组合：
      topk（每步每 trial 最多发放神经元数，类侧抑制 / 廉价 WTA）
      L1 突触归一化（每神经元权重行总量 → norm_target，跨通道取舍）
      a+/a− 不平衡（增强/抑制幅度）
      adaptive threshold（thr_adapt·rate_ema 抬阈压 post 发放率）
  - 收敛判据（冻结权重，喂 120 个全新 A + 120 个全新 B trial，统计）：
      wA/wB  = 每神经元对 A 组/B 组通道的平均权重；dW = wA − wB
      frA/frB = 每神经元在 A/B trial 下的平均发放率；dF = frA − frB
      权重分化：dW>+0.05 的神经元数 nA_w、dW<−0.05 的 nB_w
      发放分化：dF>+2e-3 的 nA_f、dF<−2e-3 的 nB_f
      收敛（判定）⇔ max(nA_w,nB_w) ≥ 0.25·N 且 nA_w+nB_w ≥ 0.4·N 且
                     max(nA_f,nB_f) ≥ 0.15·N（发放层面确有类选择性响应）

运行（隔离环境，见 stdp_core.py 头部）：
  env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin \
      /opt/miniconda3/envs/openmmlab/bin/python stdp_synthetic.py            # 全配置
  同命令 + --smoke 只跑前 2 个配置、小规模（快速冒烟）
  同命令 + --cfg-idx i 只跑第 i 个配置（调参迭代用）
结果 JSON 写 results/synthetic_results.json（含每配置双 seed）。
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

C, N, T = 40, 64, 100
HI, LO = 0.5, 0.03
A_CH = np.arange(0, 10)
B_CH = np.arange(20, 30)
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def gen_trials(rng, n_per, label_seed=None):
    """n_per 个 A + n_per 个 B trial，类内 shuffle 后交替放置（类平衡）。"""
    rng2 = np.random.default_rng(label_seed) if label_seed is not None else rng
    nA = rng2.integers(0, 2, 2 * n_per)
    rasters = np.zeros((2 * n_per, T, C), dtype=np.float32)
    for i in range(2 * n_per):
        p = np.full(C, LO)
        p[A_CH if nA[i] == 0 else B_CH] = HI
        rasters[i] = rng2.random((T, C)) < p
    return rasters, nA


def evaluate(net, rng, n_test=120):
    """冻结权重，对全新 A/B trial 统计权重分化与发放分化。"""
    frA = np.zeros(N)
    frB = np.zeros(N)
    for i in range(n_test):
        p = np.full(C, LO)
        p[A_CH] = HI
        ta = (rng.random((T, C)) < p).astype(np.float32)[None]
        frA += net.forward(ta, batch=1)[0]
        p = np.full(C, LO)
        p[B_CH] = HI
        tb = (rng.random((T, C)) < p).astype(np.float32)[None]
        frB += net.forward(tb, batch=1)[0]
    frA /= n_test * T
    frB /= n_test * T
    wA = net.W[:, A_CH].mean(axis=1)
    wB = net.W[:, B_CH].mean(axis=1)
    dW = wA - wB
    dF = frA - frB
    nA_w = int((dW > 0.05).sum())
    nB_w = int((dW < -0.05).sum())
    nA_f = int((dF > 2e-3).sum())
    nB_f = int((dF < -2e-3).sum())
    converged = (
        max(nA_w, nB_w) >= 0.25 * N
        and nA_w + nB_w >= 0.4 * N
        and max(nA_f, nB_f) >= 0.15 * N
    )
    return dict(
        nA_w=nA_w, nB_w=nB_w, nA_f=nA_f, nB_f=nB_f,
        corr_f=np.corrcoef(frA, frB)[0, 1] if frA.std() > 0 and frB.std() > 0 else 1.0,
        l1_mean=float(net.W.sum(axis=1).mean()),
        l1_min=float(net.W.sum(axis=1).min()),
        l1_max=float(net.W.sum(axis=1).max()),
        frA_mean=float(frA.mean()), frB_mean=float(frB.mean()),
        frA_max=float(frA.max()), frB_max=float(frB.max()),
        conv=converged,
    )


def run_cfg(cfg, seed, n_per=None, epochs=None):
    if n_per is None:
        n_per = cfg.get("n_per", 200)
    if epochs is None:
        epochs = cfg.get("epochs", 3)
    rng = np.random.default_rng(seed)
    net = LIFSTDP(C, N,
                  thr=cfg["thr"], tau_m=cfg.get("tau_m", 4.0),
                  tau_plus=cfg.get("tau_p", 8.0), tau_minus=cfg.get("tau_m2", 8.0),
                  a_plus=cfg["a_p"], a_minus=cfg["a_m"],
                  wmin=cfg.get("wmin", 0.0), wmax=cfg.get("wmax", 0.3),
                  w_init=(0.0, cfg.get("w_hi", 0.3)),
                  bias=cfg.get("bias", 0.0),
                  homeo_target=cfg.get("ht"), homeo_scale=cfg.get("hs", 0.02),
                  homeo_cap=cfg.get("hc", 0.3), homeo_tau=cfg.get("htau", 20.0),
                  post_topk=cfg.get("topk"), thr_adapt=cfg.get("thrad"),
                  norm_target=cfg.get("norm"), norm_eta=cfg.get("nete", 0.05),
                  rng=rng)
    tr, lab = gen_trials(rng, n_per, label_seed=seed + 1000)
    t0 = time.time()
    for _ in range(epochs):
        net.train_trials(tr, batch=cfg.get("batch", 128))
    dt = time.time() - t0
    ev = evaluate(net, rng)
    ev["train_s"] = round(dt, 2)
    return ev


CONFIGS = [
    # ---- 第 1 轮（机制组合优先配方）：topk + L1 norm + adaptive thr + a±不平衡 ----
    # 全部在 stdp_core 批量引擎内组合；结果：权重分化 ≤5/64（多数神经元权重逃逸到
    # 低频背景通道），发放分化 7-13/64——未达收敛判据（详见 STDP原型报告.md）。
    dict(tag="r1_tm2_2_tk1_th15_ap10am4", thr=1.0, a_p=0.010, a_m=0.004,
         topk=1, thrad=15.0, norm=6.0, wmax=0.3, tau_m2=2.0),
    dict(tag="r1_tm2_2_tk1_th15_ap10am2", thr=1.0, a_p=0.010, a_m=0.002,
         topk=1, thrad=15.0, norm=6.0, wmax=0.3, tau_m2=2.0),
    dict(tag="r1_tm2_1_tk1_th15_ap10am4", thr=1.0, a_p=0.010, a_m=0.004,
         topk=1, thrad=15.0, norm=6.0, wmax=0.3, tau_m2=1.0),
    dict(tag="r1_tm2_2_tk1_th30_ap10am4", thr=1.0, a_p=0.010, a_m=0.004,
         topk=1, thrad=30.0, norm=6.0, wmax=0.3, tau_m2=2.0),
    dict(tag="r1_tm2_2_tk2_th15_ap10am4", thr=1.0, a_p=0.010, a_m=0.004,
         topk=2, thrad=15.0, norm=6.0, wmax=0.3, tau_m2=2.0),
    dict(tag="r1_tm2_2_tk1_th15_ap15am6", thr=1.2, a_p=0.015, a_m=0.006,
         topk=1, thrad=15.0, norm=6.0, wmax=0.3, tau_m2=2.0),
    dict(tag="r1_tm2_2_tk1_th15_ap10am4_n4", thr=1.0, a_p=0.010, a_m=0.004,
         topk=1, thrad=15.0, norm=4.0, wmax=0.2, tau_m2=2.0),
    dict(tag="r1_tm2_2_tk1_th15_ap10am4_tauP16", thr=1.0, a_p=0.010, a_m=0.004,
         topk=1, thrad=15.0, norm=6.0, wmax=0.3, tau_m2=2.0, tau_p=16.0),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="只跑前 2 个配置、小规模")
    ap.add_argument("--cfg-idx", type=int, default=None, help="只跑第 i 个配置（0-based）")
    ap.add_argument("--seeds", type=str, default="1,2", help="逗号分隔 seed 列表")
    args = ap.parse_args()

    os.makedirs(RESULTS, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(",")]
    cfgs = CONFIGS
    if args.smoke:
        cfgs = CONFIGS[:2]
    if args.cfg_idx is not None:
        cfgs = [CONFIGS[args.cfg_idx]]

    out = {"task": "synthetic A/B 分化", "configs": []}
    for cfg in cfgs:
        per_cfg = {"tag": cfg["tag"], "cfg": cfg, "seeds": {}}
        for seed in seeds:
            ev = run_cfg(cfg, seed)
            per_cfg["seeds"][str(seed)] = ev
            print(f"[{cfg['tag']}] seed={seed} | "
                  f"w-sel A/B={ev['nA_w']}/{ev['nB_w']}  f-sel A/B={ev['nA_f']}/{ev['nB_f']}  "
                  f"corr={ev['corr_f']:.3f}  L1={ev['l1_mean']:.2f}  "
                  f"frA={ev['frA_mean']:.4f}/{ev['frA_max']:.4f} frB={ev['frB_mean']:.4f}/{ev['frB_max']:.4f}  "
                  f"conv={ev['conv']}  ({ev['train_s']}s)", flush=True)
        conv_seeds = sum(1 for e in per_cfg["seeds"].values() if e["conv"])
        per_cfg["conv_seeds"] = conv_seeds
        per_cfg["conv"] = conv_seeds == len(seeds)
        out["configs"].append(per_cfg)
    with open(os.path.join(RESULTS, "synthetic_results.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    n_conv = sum(1 for c in out["configs"] if c["conv"])
    print(f"\n== 汇总：{n_conv}/{len(out['configs'])} 个配置收敛（全 seed 判据）==")
    for c in out["configs"]:
        print(f"  {c['tag']:35s} conv_seeds={c['conv_seeds']}/{len(seeds)}")


if __name__ == "__main__":
    main()
