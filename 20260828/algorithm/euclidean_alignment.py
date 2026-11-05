#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
euclidean_alignment.py — 欧氏对齐（Euclidean Alignment, He & Wu 2019）

参考文献：He, H. & Wu, D. "Transfer Learning for Brain-Computer Interfaces:
A Euclidean Space Data Alignment Approach", IEEE TBE 2020 (arXiv:1908.05475)。

核心思想
--------
不同被试 / session 的 EEG 协方差分布差异显著，是跨 session 迁移精度掉落的
主要来源之一。EA 把每个 trial 线性变换到「训练集平均协方差 = 单位阵」的
欧氏空间后再分类，可显著降低域间协方差偏移：

  ① 每个 trial 协方差 : C_i = X_i X_i^T / T_samples        （22×22，T=500）
  ② 参考协方差       : R   = mean_i C_i                    （训练集算术平均）
  ③ 白化矩阵         : R^(-1/2) = U Λ^(-1/2) U^T           （对称矩阵特征分解）
  ④ 对齐变换         : X_align = R^(-1/2) @ X              （每个 trial 线性变换）

无监督性（关键）
----------------
R 只从训练集（= A{nn}T 会话，T 会话）计算，测试集（= A{nn}E 会话，E 会话）
复用同一个 R^(-1/2)，全程不触碰测试标签 / 测试统计 → 无泄漏。
本项目的 train.npz 本身就是 T 会话、eval.npz 是 E 会话，因此
「R 用 train.npz 全量算、eval.npz 应用」与「R 用 T 会话算、E 会话应用」
是同一件事，天然满足严格 cross-session 协议。

自检项
------
  对齐后训练集平均协方差 R^(-1/2) R R^(-1/2) ≈ I（单位阵），
  用 Frobenius 相对误差 ∥·∥_F / ∥I∥_F 与最大绝对偏差度量。
  测试集残差 ∥I − R^(-1/2) R_eval R^(-1/2)∥ 度量跨 session 协方差偏移
  在欧氏空间中未被消除的残余（R_eval = E 会话平均协方差）。

用法
----
  env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin \\
      /opt/miniconda3/envs/openmmlab/bin/python euclidean_alignment.py

产物
----
  data_4class_ea/Rinv_sqrt.npy    22×22 float32 白化矩阵（端侧化导出用）
  （stdout 输出：R 特征值分布 / 对齐自检 / 跨 session 残差诊断）
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# ---- 路径 ----
ALGO_DIR = Path(__file__).resolve().parent
DATA_4CLASS = ALGO_DIR / "data_4class"
EA_DIR = ALGO_DIR / "data_4class_ea"

N_CHANNELS = 22
N_SAMPLES = 500        # 单试次时间点数（250Hz × 2s）


def compute_reference_covariance(X: np.ndarray, n_samples: int = N_SAMPLES) -> np.ndarray:
    """计算参考协方差 R = mean_i (X_i X_i^T / T)。

    参数
    ----
    X : (N, C, T) float32 原始 trial 数据（channel × time）。
    n_samples : 协方差归一化除数 T（论文用 T_samples，即时间点数 500）。

    返回
    ----
    R : (C, C) 训练集平均协方差（对称正定，浮点下近似半正定）。
    """
    X = np.asarray(X, dtype=np.float64)
    N = X.shape[0]
    # 批量外积：einsum('nit,njt->ij') 对 N 个 trial 求和，再除 N*T
    R = np.einsum("nit,njt->ij", X, X) / (N * n_samples)
    # 强制对称（数值对称化），保证特征分解稳定
    R = 0.5 * (R + R.T)
    return R


def compute_whitening_matrix(R: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """由参考协方差 R 计算白化矩阵 R^(-1/2)（特征分解法，He & Wu 2019）。

    R = U Λ U^T（对称矩阵正交对角化，eigh 保证 U 正交、Λ 非负升序）
    R^(-1/2) = U Λ^(-1/2) U^T，其中 Λ^(-1/2) = diag(1/sqrt(max(λ_i, eps)))。

    eps 防除零：EEG 若存在近似零方差空间模式（如坏导联），λ≈0 会放大噪声，
    截断下限 eps 保持数值稳定（本项目数据 z-score 后 λ 均远离 0，见自检输出）。

    参数
    ----
    R   : (C, C) 对称半正定协方差矩阵。
    eps : 特征值下限（默认 1e-8）。

    返回
    ----
    Rinv_sqrt : (C, C) 白化矩阵，满足 Rinv_sqrt @ R @ Rinv_sqrt ≈ I。
    """
    eigvals, eigvecs = np.linalg.eigh(R)                 # 升序，U 正交
    eigvals_clamped = np.maximum(eigvals, eps)
    # U Λ^(-1/2) U^T：先对特征向量列缩放，再乘 U^T
    Rinv_sqrt = (eigvecs / np.sqrt(eigvals_clamped)) @ eigvecs.T
    Rinv_sqrt = 0.5 * (Rinv_sqrt + Rinv_sqrt.T)          # 数值对称化
    return Rinv_sqrt


def align(X: np.ndarray, Rinv_sqrt: np.ndarray) -> np.ndarray:
    """对每个 trial 施加对齐变换：X_align = R^(-1/2) @ X。

    参数
    ----
    X         : (N, C, T) 原始 trial 数据（训练/测试通用，同一个 Rinv_sqrt）。
    Rinv_sqrt : (C, C) 白化矩阵（只由训练集算出）。

    返回
    ----
    X_align : (N, C, T) 对齐后数据（float32，保持原始 dtype/数值尺度可训练）。
    """
    X = np.asarray(X)
    out = np.einsum("ij,njt->nit", Rinv_sqrt.astype(X.dtype), X)
    return out


def mean_covariance(X: np.ndarray, n_samples: int = N_SAMPLES) -> np.ndarray:
    """诊断辅助：计算一组 trial 的平均协方差（同 compute_reference_covariance）。"""
    return compute_reference_covariance(X, n_samples)


def alignment_self_check(X_aligned: np.ndarray, n_samples: int = N_SAMPLES
                         ) -> dict:
    """自检：对齐后平均协方差应 ≈ 单位阵。

    返回诊断字典：
      frob_err    : ∥mean_cov − I∥_F（绝对）
      frob_rel    : ∥mean_cov − I∥_F / ∥I∥_F（相对，∥I∥_F = sqrt(C)）
      max_abs_dev : max |mean_cov − I|（最大绝对偏差）
      diag_mean   : 对角均值（理想 = 1.0，反映各通道方差被归一化程度）
      offdiag_rms: 非对角均方根（理想 = 0，反映通道间相关被白化程度）
    """
    mc = mean_covariance(X_aligned, n_samples)           # (C, C)
    C = mc.shape[0]
    I = np.eye(C)
    diff = mc - I
    frob_err = float(np.linalg.norm(diff, "fro"))
    frob_rel = frob_err / float(np.linalg.norm(I, "fro"))
    max_abs_dev = float(np.abs(diff).max())
    offdiag = diff[~np.eye(C, dtype=bool)]
    return {
        "frob_err": frob_err,
        "frob_rel": frob_rel,
        "max_abs_dev": max_abs_dev,
        "diag_mean": float(np.mean(np.diag(mc))),
        "offdiag_rms": float(np.sqrt(np.mean(offdiag**2))),
    }


def main() -> int:
    train_npz = DATA_4CLASS / "train.npz"
    eval_npz = DATA_4CLASS / "eval.npz"
    if not (train_npz.exists() and eval_npz.exists()):
        raise SystemExit(f"✗ 数据集缺失：{DATA_4CLASS}（先跑 preprocess_4class.py）")

    with np.load(train_npz, allow_pickle=False) as d:
        X_train = np.asarray(d["X"], dtype=np.float32)   # (N,1,22,500)
        y_train = d["y"]
    with np.load(eval_npz, allow_pickle=False) as d:
        X_eval = np.asarray(d["X"], dtype=np.float32)
        y_eval = d["y"]
    assert X_train.shape[1] == 1 and X_train.shape[2] == N_CHANNELS
    # 去掉图像通道维 → (N, C, T)
    Xt = X_train[:, 0]
    Xe = X_eval[:, 0]
    print(f"训练集 : {Xt.shape}（T 会话，{len(y_train)} 试次）")
    print(f"测试集 : {Xe.shape}（E 会话，{len(y_eval)} 试次）")

    # ---- ① 参考协方差（只由训练集 = T 会话计算，无泄漏）----
    R_train = compute_reference_covariance(Xt)
    # ---- ② 白化矩阵 R^(-1/2) ----
    Rinv_sqrt = compute_whitening_matrix(R_train)

    eigvals = np.linalg.eigvalsh(R_train)
    print("\n[参考协方差 R（训练集 = T 会话）]")
    print(f"  特征值范围 : min={eigvals.min():.4f}  "
          f"median={np.median(eigvals):.4f}  max={eigvals.max():.4f}")
    print(f"  条件数     : {eigvals.max()/max(eigvals.min(), 1e-12):.1f}"
          "（≈1 表示通道方差尺度均匀，白化数值稳定）")

    # ---- ③ 对齐变换（训练 + 测试，同一个 Rinv_sqrt）----
    Xt_a = align(Xt, Rinv_sqrt)
    Xe_a = align(Xe, Rinv_sqrt)

    # ---- ④ 自检：训练集对齐后平均协方差 ≈ 单位阵 ----
    chk = alignment_self_check(Xt_a)
    print("\n[自检] 对齐后训练集平均协方差（应 ≈ I）")
    print(f"  Frobenius 绝对误差 : {chk['frob_err']:.4e}"
          f"（相对 {chk['frob_rel']*100:.4f}%）")
    print(f"  最大绝对偏差       : {chk['max_abs_dev']:.4e}")
    print(f"  对角均值           : {chk['diag_mean']:.6f}（理想 1.0）")
    print(f"  非对角 RMS         : {chk['offdiag_rms']:.4e}（理想 0）")
    assert chk["frob_rel"] < 1e-3, "✗ 对齐自检失败：训练集协方差未对齐到单位阵！"

    # ---- ⑤ 跨 session 诊断：E 会话对齐后的残余协方差偏移 ----
    R_eval = compute_reference_covariance(Xe)
    resid = np.linalg.norm(
        Rinv_sqrt @ R_eval @ Rinv_sqrt - np.eye(N_CHANNELS), "fro")
    resid_rel = resid / np.linalg.norm(np.eye(N_CHANNELS), "fro")
    # 对齐前 T/E 会话平均协方差的域间距离（Frobenius，归一化）
    dom_before = np.linalg.norm(R_train - R_eval, "fro") / np.linalg.norm(R_train, "fro")
    print("\n[跨 session 诊断]（T 会话算 R，E 会话应用，严格 cross-session）")
    print(f"  对齐前 T/E 平均协方差相对距离 : {dom_before*100:.2f}%")
    print(f"  对齐后 E 会话残差 ∥I−Rinv·R_E·Rinv∥_F : {resid:.4f}"
          f"（相对 {resid_rel*100:.2f}%；越小 = 跨 session 协方差域越一致）")

    # ---- ⑥ 导出 R^(-1/2)（端侧化用：22×22 float32，4 字节×484 ≈ 1.9 KB）----
    EA_DIR.mkdir(exist_ok=True)
    out_path = EA_DIR / "Rinv_sqrt.npy"
    np.save(out_path, Rinv_sqrt.astype(np.float32))
    print(f"\n✓ 白化矩阵已导出：{out_path}（{Rinv_sqrt.astype(np.float32).shape} "
          f"float32，{(Rinv_sqrt.astype(np.float32).nbytes/1024):.2f} KB）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
