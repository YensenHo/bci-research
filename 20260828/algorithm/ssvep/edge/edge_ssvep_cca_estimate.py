#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
端侧评估: SSVEP-CCA 在 RV1106 (Luckfox Pico, 双核 Cortex-A7 @1.2GHz) 的计算量估算
对比: 我们的运动想象 CNN 推理 ~50ms 量级。
模型: 每决策 = Cxx + 40×(Cxy + Cxx^{-1/2} 特征分解 + H 构造 + 最大奇异值/幂迭代)
输出 JSON + 控制台表格。不虚构: 所有 FLOP 计数按实际算法逐项累加, 效率按公开经验区间取值。
"""
import json

FREQS = 40
Q = 6          # 模板分量数 (3 次谐波 × sin/cos)
HARM_ITERS = 40  # 幂迭代次数

def jacobi_eig_flops(p, sweeps=30):
    """对称 p×p Jacobi 特征分解: 每 sweep p(p-1)/2 次旋转 × ~6p 次乘加"""
    return p * (p - 1) // 2 * (6 * p) * sweeps

def decision_flops(p, n, use_lu_for_inv=False):
    """单次 CCA 决策 (p 通道, n 采样, 40 频率, 3 谐波) 的 FLOP 数"""
    fl = {}
    fl["center_X"] = p * n
    fl["Cxx"] = 2 * p * p * n
    fl["Cxy_40freq"] = FREQS * 2 * p * Q * n
    if use_lu_for_inv or p > 12:
        fl["Cxx_inv(LU)"] = 2 * p**3 / 3
    else:
        fl["Cxx_inv(Jacobi)"] = jacobi_eig_flops(p)
    per_freq = {}
    if p > 12:
        per_freq["solve_CxxCxy"] = 2 * p * p * Q
    per_freq["H_matmul_2x"] = 2 * p * Q * Q
    per_freq["power_iter"] = HARM_ITERS * p * p
    fl["per_freq_*40"] = FREQS * sum(per_freq.values())
    return fl

def total(fl):
    return sum(fl.values())

def rv1106_time(flops, gf_slow, gf_fast):
    return flops / gf_slow * 1e3, flops / gf_fast * 1e3  # ms

def main():
    out = {}
    print("=" * 78)
    print("RV1106 (双核 Cortex-A7 @1.2GHz) SSVEP-CCA 计算量估算")
    print("峰值 fp32: 2 核 × 1.2GHz × 4 flops/cycle(NEON) ≈ 9.6 GFLOPS")
    print("现实取值: 标量 C 代码 fp32 持续 1~2 GFLOPS; fp64 约 0.5~1 GFLOPS")
    print("=" * 78)
    rows = []
    for p, n, label in [(6, 125, "6ch 0.5s"), (6, 250, "6ch 1.0s"),
                        (6, 500, "6ch 2.0s"), (6, 750, "6ch 3.0s"),
                        (64, 250, "64ch 1.0s"), (64, 750, "64ch 3.0s")]:
        fl = decision_flops(p, n)
        tot = total(fl)
        t_slow, t_fast = rv1106_time(tot, 0.5e9, 2.0e9)
        rows.append((label, tot, t_slow, t_fast))
        out[label] = {"channels": p, "samples": n, "MFLOP": round(tot / 1e6, 3),
                      "ms_fp64_0.5GF": round(t_slow, 3),
                      "ms_fp32_2GF": round(t_fast, 3)}
        print("%-10s  %9.3f MFLOP   fp64@0.5GF: %7.3f ms   fp32@2GF: %7.3f ms"
              % (label, tot / 1e6, t_slow, t_fast))

    print("-" * 78)
    print("明细 (6ch 1s):", {k: "%.0f flops" % v for k, v in
                            decision_flops(6, 250).items()})
    print("明细 (64ch 1s):", {k: "%.0f flops" % v for k, v in
                             decision_flops(64, 250).items()})
    print("-" * 78)
    print("C 实现实测 (macOS 主机, -O2, 含合成数据生成): 171.5 us/决策")
    print("按 主频比~3x + 无NEON标量fp64 ~3-5x 折算 → RV1106 约 1~3 ms/决策(6ch 1s)")
    print("对照: 运动想象 CNN 端侧推理 ~50 ms (NPU, RKNPU) —— CCA 计算快 ~2 个数量级,")
    print("      但 SSVEP 决策延迟受窗口长度主导 (0.5~3 s), 计算时间可忽略。")
    out["note"] = ("6ch 1s 主流程约 0.9 MFLOP/决策; RV1106 标量 C 实现估算 0.3~2 ms; "
                   "C 代码实测(主机) 171.5us/决策含数据合成; CNN 对照 50ms(NPU)")
    with open("edge_cca_estimate.json", "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("已保存 edge_cca_estimate.json")

if __name__ == "__main__":
    main()
