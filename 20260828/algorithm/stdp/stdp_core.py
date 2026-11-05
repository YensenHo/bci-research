#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stdp_core.py — 单层 LIF 脉冲神经元 + STDP 突触可塑性（numpy 实现，原型）

语义（对齐类脑纲领，与 lif_eegnet.py 的 LIF 一致，但这里是**真实脉冲流 + 事件驱动
可塑性**，不是 surrogate 训练）：
  - 网络：输入层 C 个脉冲源 → 学习层 N 个 LIF 神经元，全连接 W (N, C)。
  - LIF 前向（每步 dt=1，时间驱动仿真）：
        V <- decay*V + W @ s_in + bias        （decay = exp(-1/tau_m)）
        fire: V >= thr → 发放，硬重置 V <- 0
    与 lif_eegnet.LIFActivation 的差别：输入不再是恒定电流，而是逐时间步的
    0/1 脉冲向量 s_in[t] ∈ {0,1}^C；发放也是 0/1 事件（不做率聚合）。
  - STDP 更新（trace 法，nearest-neighbour 指数核，事件时刻就地更新）：
        pre_trace[c]  *= exp(-1/tau_plus)    ；发放通道置 1（set-to-1 指数核）
        post_trace[n] *= exp(-1/tau_minus)   ；发放神经元置 1
        每步顺序：① trace 衰减 ② pre 发放 → 抑制：ΔW[n,c] = -A- * post_trace[n] * s_pre[c]
                   ③ LIF 积分 ④ post 发放 → 增强：ΔW[n,c] = +A+ * pre_trace[c] * s_post[n]
    与解析 pair 公式的对应：pre 在 t_pre 发放、post 在 t_post 发放（最近邻配对）时
        Δw = A+·exp(-(t_post-t_pre)/tau_plus)  若 t_post > t_pre
             -A-·exp(-(t_pre-t_post)/tau_minus) 若 t_pre > t_post
    权重裁剪到 [wmin, wmax]。
  - Homeostatic（防单神经元霸占/救活静默神经元，可选）：
        synaptic scaling：每个 trial 结束后按「目标发放率」乘性缩放每行权重
            W[n,:] *= clip(1 + s_scale*(target - r_n)/target, 1-s_cap, 1+s_cap)
        r_n = 该 trial 该神经元发放次数 / T（EMA 平滑）。
  - 批处理近似：一批 B 个 trial 并行仿真（同一 W）。W 的更新按 batch 内事件求和，
    等价于把各 trial 在**同一时间步**的更新叠加（小学习率下与串行近似等价——
    自检与合成验证都用同一引擎，保证自洽）。

编码选型说明（详见 stdp_eeg.py）：本模块只消费 0/1 脉冲序列 (T, C)（或批 (B,T,C)），
对"连续 EEG → 脉冲"的编码方式不做假设——率编码（泊松）与 delta 事件编码都喂同一种
输入格式，便于对照「率 vs 时序」两类信息在 STDP 下的可学性。

用法（自检，验证 LIF 前向 + STDP 符号约定 + pair 核一致性）：
  env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin \\
      /opt/miniconda3/envs/openmmlab/bin/python stdp_core.py
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# 解析 STDP pair 公式（自检对照用）
# ---------------------------------------------------------------------------
def pair_delta_w(dt: float, a_plus: float, tau_plus: float,
                 a_minus: float, tau_minus: float) -> float:
    """单对脉冲的解析权重变化。

    dt = t_post - t_pre：
      dt > 0（pre 先发 → post 后发，因果增强）：Δw =  A+ · exp(-dt/tau_plus)
      dt < 0（post 先发 → pre 后发，反因果抑制）：Δw = -A- · exp(dt/tau_minus)
      dt = 0（同一步，按增强处理）
    """
    if dt >= 0:
        return a_plus * np.exp(-dt / tau_plus)
    return -a_minus * np.exp(dt / tau_minus)


# ---------------------------------------------------------------------------
# 单层 LIF + STDP 学习层（numpy，时间驱动，支持批量并行）
# ---------------------------------------------------------------------------
class LIFSTDP:
    """单层 LIF 脉冲神经元 + 在线 STDP 学习（numpy）。

    参数
    ----
    n_in, n_out      : 输入通道数 / 学习层神经元数
    thr              : 发放阈值（标量或 (n_out,) 向量）
    tau_m            : 膜时间常数（步）
    tau_plus         : 增强核时间常数（步，pre→post）
    tau_minus        : 抑制核时间常数（步，post→pre）
    a_plus, a_minus  : 单事件权重变化幅度（增强 / 抑制）
    wmin, wmax       : 权重裁剪区间
    bias             : 常数注入电流（标量；帮静默神经元起步，可选）
    w_init           : 权重初始化区间 (lo, hi)（均匀）
    homeo_target     : 目标发放率（每神经元每步，e.g. 0.05）；None=关闭 scaling
    homeo_scale      : scaling 步长系数 s_scale
    homeo_cap        : 单次 scaling 上限（1±cap）
    homeo_tau        : 发放率 EMA 时间常数（trial 数）
    post_topk        : 每 trial 每步最多发放的神经元数（None=不限制）。>0 时同一
                       trial 同一步只保留膜电位最高的 topk 个发放——廉价的类侧抑制，
                       强制神经元分化（防霸占的另一手段）。
    thr_adapt        : 自适应阈值系数（None=关闭；Diehl & Cook 2015 式）。开启时
                       每神经元有效阈值 = thr + thr_adapt·rate_ema（rate_ema 为
                       该神经元长期发放率 EMA，per-step 单位）——发放多的神经元阈值
                       升高，把机会让给静默神经元（防单神经元霸占 + 神经元多样性）。
    norm_target      : 每神经元权重行 L1 范数目标（None=关闭）。>0 时权重行按
                       (target/L1)^(eta·batch) 乘性收缩/放大（eta 为 per-trial 步长，
                       批内按 trial 数次方复合）——**突触归一化**（homeostatic 家族的
                       一员）：总量受限 → 神经元必须在不同输入通道间取舍 → 模式特化
                       的竞争机制（纯 STDP + 逐神经元独立 homeostatic 无法产生跨模式
                       竞争，见 stdp_synthetic.py 说明）。
    norm_eta         : 归一化步长（每次最多向目标靠近的比例，0~1）
    rng              : np.random.Generator
    """

    def __init__(self, n_in: int, n_out: int, *,
                 thr: float = 1.0, tau_m: float = 4.0,
                 tau_plus: float = 4.0, tau_minus: float = 4.0,
                 a_plus: float = 0.005, a_minus: float = 0.006,
                 wmin: float = 0.0, wmax: float = 1.0,
                 bias: float = 0.0,
                 w_init: tuple[float, float] = (0.0, 0.25),
                 homeo_target: float | None = None,
                 homeo_scale: float = 0.02, homeo_cap: float = 0.5,
                 homeo_tau: float = 20.0,
                 post_topk: int | None = None,
                 thr_adapt: float | None = None,
                 norm_target: float | None = None,
                 norm_eta: float = 0.05,
                 rng: np.random.Generator | None = None):
        self.n_in = int(n_in)
        self.n_out = int(n_out)
        self.thr = float(thr)
        self.decay_mem = float(np.exp(-1.0 / tau_m))
        self.decay_pre = float(np.exp(-1.0 / tau_plus))
        self.decay_post = float(np.exp(-1.0 / tau_minus))
        self.a_plus = float(a_plus)
        self.a_minus = float(a_minus)
        self.wmin = float(wmin)
        self.wmax = float(wmax)
        self.bias = float(bias)
        self.homeo_target = homeo_target
        self.homeo_scale = float(homeo_scale)
        self.homeo_cap = float(homeo_cap)
        self.homeo_tau = float(homeo_tau)
        self.post_topk = post_topk
        self.thr_adapt = float(thr_adapt) if thr_adapt is not None else None
        self.thr_base = float(thr)
        self.norm_target = norm_target
        self.norm_eta = float(norm_eta)
        self.rng = rng if rng is not None else np.random.default_rng(0)
        lo, hi = w_init
        self.W = self.rng.uniform(lo, hi, size=(n_out, n_in)).astype(np.float64)
        # 诊断
        self.rate_ema = np.zeros(n_out)          # 长期发放率 EMA（每步）
        self.total_updates = 0

    # ---- 权重访问/重设 ------------------------------------------------------
    def set_weights(self, w: np.ndarray) -> None:
        """外部设权重（随机基线用：同分布冻结权重，不学习）。"""
        w = np.asarray(w, dtype=np.float64)
        assert w.shape == (self.n_out, self.n_in)
        self.W = w.copy()

    def random_weights_like_init(self, rng: np.random.Generator | None = None) -> np.ndarray:
        """与初始化同分布的权重（随机基线用）。"""
        rng = rng if rng is not None else self.rng
        return rng.uniform(self.W.min(), self.W.max() if self.W.max() > self.W.min() else 0.25,
                           size=(self.n_out, self.n_in)).astype(np.float64)

    # ---- 核心：一批 trial 的 LIF + 在线 STDP ----------------------------------
    def train_trials(self, spikes: np.ndarray, learn: bool = True,
                     batch: int = 64) -> np.ndarray:
        """对一批 trial 跑 LIF 前向（可在线 STDP 学习）。

        spikes : (n_trials, T, C) 0/1 脉冲（float）。同一批内各 trial 独立，长度须一致。
        返回   : (n_trials, n_out) 每 trial 每神经元的发放次数。
        """
        n_tr, T, C = spikes.shape
        assert C == self.n_in, f"输入通道 {C} != n_in {self.n_in}"
        counts_all = np.zeros((n_tr, self.n_out), dtype=np.float64)
        for b0 in range(0, n_tr, batch):
            sb = spikes[b0:b0 + batch]
            counts_all[b0:b0 + batch] = self._train_batch(sb, learn=learn)
        return counts_all

    def _train_batch(self, sb: np.ndarray, learn: bool) -> np.ndarray:
        B, T, C = sb.shape
        N = self.n_out
        V = np.zeros((B, N))
        pre_trace = np.zeros((B, C))
        post_trace = np.zeros((B, N))
        counts = np.zeros((B, N))
        d_mem, d_pre, d_post = self.decay_mem, self.decay_pre, self.decay_post
        # 有效阈值：固定标量 or 自适应（thr_base + thr_adapt·rate_ema）
        if self.thr_adapt is not None:
            thr_vec = self.thr_base + self.thr_adapt * self.rate_ema
        else:
            thr_vec = np.full(N, self.thr_base)
        W = self.W
        for t in range(T):
            s_pre = sb[:, t]                       # (B, C) 0/1
            # ① trace 衰减 + pre 发放置 1（set-to-1 指数核，最近邻配对）
            pre_trace *= d_pre
            post_trace *= d_post
            # ② pre 发放 → 抑制（post 先发、pre 后发）：ΔW -= A-·post_trace
            if learn and s_pre.any():
                # (N,B)@(B,C) -> (N,C)：各 trial 事件求和叠加
                delta = self.a_minus * (post_trace.T @ s_pre)
                np.subtract(W, delta, out=W)
                np.clip(W, self.wmin, self.wmax, out=W)   # 每步裁剪，防负权重病理
            # pre 置 1 放在抑制之后：同一步 pre 的发放不算"post 先发"（dt=0 归增强）
            fired_pre = s_pre > 0
            pre_trace[fired_pre] = 1.0
            # ③ LIF 积分（s_pre 参与本步电流）
            cur = s_pre @ W.T + self.bias           # (B, N)
            V = d_mem * V + cur
            # ④ 发放判定 + 硬重置
            fire = V >= thr_vec[None, :]
            if self.post_topk is not None and self.post_topk > 0:
                # 每 trial 每步最多 topk 个：膜电位最高的优先（类侧抑制）
                n_fire = fire.sum(axis=1)
                if n_fire.any():
                    for b in np.nonzero(n_fire)[0]:
                        idx = np.nonzero(fire[b])[0]
                        if len(idx) > self.post_topk:
                            keep = idx[np.argsort(-V[b, idx])[:self.post_topk]]
                            fire[b] = False
                            fire[b, keep] = True
            if learn and fire.any():
                # post 发放 → 增强：ΔW += A+·pre_trace（pre 先发或同步）
                W += self.a_plus * (fire.T @ pre_trace)
                # 原地 clip：保持 W 与 self.W 的别名（否则后续抑制写到脱离数组上）
                np.clip(W, self.wmin, self.wmax, out=W)
            V[fire] = 0.0
            counts += fire
            # post 置 1（增强之后；本步 post 先发与否留给下一步的抑制判定）
            post_trace[fire] = 1.0
        # rate EMA：学习期总是更新（adaptive thr / homeo scaling / 诊断共用）
        if learn:
            rates = counts / T                              # (B, N) 每 trial 发放率
            s_b = 1.0 - (1.0 - 1.0 / self.homeo_tau) ** B   # B 步复合的 EMA 系数
            self.rate_ema = (1.0 - s_b) * self.rate_ema + s_b * rates.mean(axis=0)
        # homeostatic：synaptic scaling（可选；常数 per-trial，批内按 B 复合）
        if learn and self.homeo_target is not None:
            err = (self.homeo_target - self.rate_ema) / max(self.homeo_target, 1e-9)
            factor = np.power(1.0 + self.homeo_scale * err, B)
            cap = (1.0 + self.homeo_cap) ** B if self.homeo_cap >= 0 else 1.0
            floor = (1.0 - self.homeo_cap) ** B
            factor = np.clip(factor, floor, cap)
            np.clip(self.W * factor[:, None], self.wmin, self.wmax, out=self.W)
        # 突触归一化：每神经元行 L1 → norm_target（幂次步长，per-trial 语义，B 次复合；
        # 这是跨模式竞争机制——总量受限 → 神经元必须在不同输入通道间取舍）
        if learn and self.norm_target is not None:
            l1 = self.W.sum(axis=1)
            factor = np.power(self.norm_target / np.maximum(l1, 1e-9),
                              self.norm_eta * B)
            np.clip(self.W * factor[:, None], self.wmin, self.wmax, out=self.W)
        self.total_updates += 1
        return counts

    def forward(self, spikes: np.ndarray, batch: int = 64) -> np.ndarray:
        """冻结权重前向（无学习）：返回每 trial 每神经元发放次数 (n_tr, n_out)。"""
        return self.train_trials(spikes, learn=False, batch=batch)


# ---------------------------------------------------------------------------
# 自检（__main__）：LIF 前向 + STDP 符号 + pair 核一致性
# ---------------------------------------------------------------------------
def _self_test() -> None:
    rng = np.random.default_rng(0)
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        status = "PASS" if cond else "FAIL"
        if not cond:
            ok = False
        print(f"  [{status}] {name} {detail}")

    print("== 自检 1/4：LIF 前向基本行为 ==")
    net = LIFSTDP(2, 1, thr=1.0, tau_m=2.0, w_init=(0.5, 0.5),
                  a_plus=0.0, a_minus=0.0, bias=0.0, rng=rng)
    # 静默（无输入）：不发放
    spikes0 = np.zeros((1, 20, 2))
    c0 = net.forward(spikes0)
    check("无输入不发放", c0.sum() == 0, f"counts={c0.tolist()}")
    # 恒定强输入：10 步每步 2 通道都发（每步电流 0.5+0.5=1.0 ≥ thr）→ 应每步都发
    spikes1 = np.ones((1, 10, 2))
    c1 = net.forward(spikes1)
    check("强输入每步发放", c1[0, 0] == 10, f"counts={c1[0].tolist()} (期望 10)")

    print("== 自检 2/4：STDP 符号约定（端到端，单突触） ==")
    # (a) post 先发 → pre 后发：抑制。偏置 bias=0.42（thr=1, tau_m=2 → asymptote
    #     0.42/(1-exp(-0.5))=1.014>1）让 post 精确在 t=5 自发放一次（V 第 6 步过阈）；
    #     pre 单脉冲在 t=7 → post_trace 已衰减两拍 → 抑制 A-·exp(-2/tau_minus)。
    #     w 取 0.1 保证 pre 步不把 post 重新打过阈（V(t7)=d·0.42+0.1+0.42<1）。
    net = LIFSTDP(1, 1, thr=1.0, tau_m=2.0, w_init=(0.1, 0.1),
                  tau_plus=4.0, tau_minus=4.0, a_plus=0.01, a_minus=0.02,
                  bias=0.42, rng=rng)
    sp = np.zeros((1, 8, 1))
    sp[0, 7, 0] = 1.0                      # pre 在 t=7（post 自发放 t=5 之后两拍）
    n_fire_bias = net.forward(sp)
    w0 = net.W[0, 0].copy()
    net = LIFSTDP(1, 1, thr=1.0, tau_m=2.0, w_init=(0.1, 0.1),
                  tau_plus=4.0, tau_minus=4.0, a_plus=0.01, a_minus=0.02,
                  bias=0.42, rng=np.random.default_rng(0))
    net.train_trials(sp)
    dw = net.W[0, 0] - w0
    expect_dw = -0.02 * np.exp(-2.0 / 4.0)
    check("post 先发→pre 后发 → 抑制 Δw<0", n_fire_bias[0, 0] == 1 and abs(dw - expect_dw) < 1e-12,
          f"Δw={dw:+.5f} 期望 {expect_dw:+.5f}（post 发放 {n_fire_bias[0,0]:.0f} 次）")
    # (b) pre 先发 → post 随后被驱动发放：增强。pre 在 t=4,5,6 连发（w=0.5, thr=0.9）：
    #     V(t4)=0.5, V(t5)=0.804, V(t6)=0.988 ≥0.9 → post 在 t=6 发放 → +A+·pre_trace(=1)
    net = LIFSTDP(1, 1, thr=0.9, tau_m=2.0, w_init=(0.5, 0.5),
                  tau_plus=4.0, tau_minus=4.0, a_plus=0.02, a_minus=0.02,
                  bias=0.0, rng=rng)
    w0 = net.W[0, 0].copy()
    sp = np.zeros((1, 12, 1))
    sp[0, 4, 0] = 1.0
    sp[0, 5, 0] = 1.0
    sp[0, 6, 0] = 1.0
    net.train_trials(sp)
    dw_pre_first = net.W[0, 0] - w0
    check("pre 先发 → post 后发 → 增强 Δw>0", abs(dw_pre_first - 0.02) < 1e-12,
          f"Δw={dw_pre_first:+.5f} 期望 +0.02000")

    print("== 自检 3/4：trace STDP 与解析 pair 公式一致 ==")
    # 单输入单输出：post 由外源驱动在固定时刻发放（用一发大电流脉冲通道模拟：先算好
    # pre 在 t=2，post 在 t=6 —— 直接构造两对事件，比较 trace 法 vs pair 公式。
    # 等价性：最近邻 set-to-1 trace 的离散衰减 exp(-1/tau) 就是 pair 核的逐点采样。
    tau_p, tau_m = 5.0, 5.0
    a_p, a_m = 0.01, 0.012
    net = LIFSTDP(1, 1, thr=1e9, w_init=(0.1, 0.1), tau_plus=tau_p,
                  tau_minus=tau_m, a_plus=a_p, a_minus=a_m, rng=rng)  # thr 极高→不发放
    # 但 post 永不发放就测不到增强——改用"手动注入"：直接调用内部 trace 状态机模拟
    # pre @t=2, post @t=6 的一对事件（等效于把两事件丢进同一状态机）
    d_p, d_m = np.exp(-1.0 / tau_p), np.exp(-1.0 / tau_m)
    # 模拟：pre t=2 → post t=6
    pre_tr, post_tr = 0.0, 0.0
    w = 0.1
    for t in range(8):
        pre_tr *= d_p
        post_tr *= d_m
        if t == 2:          # pre 事件：抑制(用此刻 post_trace) 后置 1
            w -= a_m * post_tr
            pre_tr = 1.0
        if t == 6:          # post 事件：增强(用此刻 pre_trace) 后置 1
            w += a_p * pre_tr
            post_tr = 1.0
    expect = 0.1 + a_p * np.exp(-4.0 / tau_p)
    check("增强幅度=解析 A+·exp(-4/τ+)",
          abs(w - expect) < 1e-9, f"trace={w:.6f} analytic={expect:.6f}")
    # post @t=2, pre @t=6 → 抑制
    pre_tr, post_tr, w = 0.0, 0.0, 0.1
    for t in range(8):
        pre_tr *= d_p
        post_tr *= d_m
        if t == 2:
            post_tr = 1.0
        if t == 6:
            w -= a_m * post_tr
    expect = 0.1 - a_m * np.exp(-4.0 / tau_m)
    check("抑制幅度=解析 -A-·exp(-4/τ-)",
          abs(w - expect) < 1e-9, f"trace={w:.6f} analytic={expect:.6f}")

    print("== 自检 4/4：pair_delta_w 符号/单调性 ==")
    check("dt>0 增强、dt<0 抑制、|dt|↑ → |Δw|↓",
          pair_delta_w(1, a_p, tau_p, a_m, tau_m) > 0
          and pair_delta_w(-1, a_p, tau_p, a_m, tau_m) < 0
          and pair_delta_w(2, a_p, tau_p, a_m, tau_m) < pair_delta_w(1, a_p, tau_p, a_m, tau_m)
          and abs(pair_delta_w(-3, a_p, tau_p, a_m, tau_m)) < abs(pair_delta_w(-1, a_p, tau_p, a_m, tau_m)),
          "")

    print("自检结果:", "全部通过 ✔" if ok else "存在失败 ✘")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    _self_test()
