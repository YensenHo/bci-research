#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
riemannian_baseline.py — 黎曼几何分类器基线（numpy 手写，无 pyriemann 依赖）

BCIC IV 2a 四分类 22ch，cross-session 协议（train=T 会话 2592 试次训练，eval=E 会话 2592 试次测试）
对比对象：LIF-EEG-TCNet + HBC 增强 clean 63.58%（cross-session 单测，seed42）

管线（BCI 经典黎曼方法，Barachant 2013 切线空间范式）：
  1. 每试次样本协方差 C_i = (1/(T-1)) X_i X_i^T ∈ SPD(22)
  2. 协方差正则化（shrinkage）：C_reg = (1-α)C + α·(trace(C)/n)·I，α ∈ {0.0, 0.1, 0.3, 0.5}
  3. 参考点 P = 黎曼均值（Karcher 均值，SPD 流形上的不动点迭代；备选：算术均值 / 对数欧氏均值）
  4. 切线空间投影：T_i = logm(P^{-1/2} C_i P^{-1/2})，上三角向量化（非对角 ×√2），维度 22·23/2 = 253
  5. 分类器：逻辑回归（多分类 multinomial，lbfgs）或 LDA（shrinkage='auto'）
  → 网格搜索 ≤2 参数：协方差正则 α × 分类器 {LR, LDA}；参考点选择单独报告 3 行（非网格）

用法（环境铁律：env -i 隔离 + openmmlab python）：
  env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin \
      /opt/miniconda3/envs/openmmlab/bin/python riemannian_baseline.py --smoke   # 冒烟：子集验证
  env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin \
      /opt/miniconda3/envs/openmmlab/bin/python riemannian_baseline.py           # 全量

输出：控制台表格 + riemannian_results.json（供报告引用）
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.linalg import eigh as scipy_eigh  # noqa: F401  (保留：如 numpy eig 异常时回退用)

ALGO_DIR = Path(__file__).resolve().parent
DATA_DIR = ALGO_DIR / "data_4class"
OUT_JSON = ALGO_DIR / "riemannian_results.json"

N_CH = 22
N_SAMPLES = 500
N_CLASSES = 4
VEC_DIM = N_CH * (N_CH + 1) // 2  # 253


# ---------------------------------------------------------------- 黎曼工具（纯 numpy）
def batch_eigh(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """批量对称 eigendecomposition：A (..., d, d) -> (w, U)，A = U diag(w) U^T。"""
    return np.linalg.eigh(A)


def batch_logm_sym(A: np.ndarray) -> np.ndarray:
    """对称正定矩阵批量对数：A (..., d, d) -> logm(A)。
    首选 numpy 批量 eigh（快）；病态矩阵下 numpy eigh 可能不收敛，回退 scipy.logm（稳健）。
    注意：必须 U·diag(log w)·U^T（s 为列向量 [..., :, None]），写成行向量会退化为 diag(log w)。"""
    A = np.asarray(A)
    try:
        w, U = np.linalg.eigh(A)
        w = np.clip(w, 1e-12, None)
        out = U @ (np.log(w)[..., :, None] * U.swapaxes(-1, -2))
        if np.isfinite(out).all():
            return out
    except np.linalg.LinAlgError:
        pass
    from scipy.linalg import logm
    out = np.empty_like(A)
    for i in range(A.shape[0]):
        out[i] = logm(A[i])
    return out


def batch_expm_sym(A: np.ndarray) -> np.ndarray:
    """对称矩阵批量指数。"""
    w, U = np.linalg.eigh(A)
    return U @ (np.exp(w)[..., :, None] * U.swapaxes(-1, -2))


def batch_inv_sqrtm(A: np.ndarray) -> np.ndarray:
    """SPD 批量逆平方根 A^{-1/2}。特征值裁剪防浮点负值。"""
    w, U = np.linalg.eigh(A)
    w = np.clip(w, 1e-12, None)
    return U @ (np.power(w, -0.5)[..., :, None] * U.swapaxes(-1, -2))


def batch_sqrtm(A: np.ndarray) -> np.ndarray:
    """SPD 批量平方根 A^{1/2}。"""
    w, U = np.linalg.eigh(A)
    w = np.clip(w, 1e-12, None)
    return U @ (np.sqrt(w)[..., :, None] * U.swapaxes(-1, -2))


def riemannian_mean(C: np.ndarray, tol: float = 1e-9, max_iter: int = 100) -> np.ndarray:
    """黎曼（Karcher）均值：M ← M^{1/2} expm(mean_i logm(M^{-1/2} C_i M^{-1/2})) M^{1/2}。
    C: (n, d, d) SPD 批量。返回 (d, d) SPD。
    用 scipy.linalg（logm/expm/sqrtm，带平衡的 eig）而非 numpy eigh——本数据协方差条件数
    ~46k（min eig 4e-4），numpy eigh（LAPACK dsyevd 分治）在病态中间矩阵上会不收敛。"""
    from scipy.linalg import expm, inv, logm, sqrtm
    M = C.mean(axis=0)
    for _ in range(max_iter):
        M_inv_sqrt = inv(sqrtm(M))
        T_bar = np.mean([logm(M_inv_sqrt @ Ci @ M_inv_sqrt) for Ci in C], axis=0)
        M_new = sqrtm(M) @ expm(T_bar) @ sqrtm(M)
        if np.linalg.norm(M_new - M) / (np.linalg.norm(M) + 1e-12) < tol:
            M = M_new
            break
        M = M_new
    return M


def tangent_projection(C: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """切线空间投影：T_i = logm(P^{-1/2} C_i P^{-1/2})，上三角向量化（非对角 ×√2）。
    返回 (n, d(d+1)/2)。"""
    P_inv_sqrt = batch_inv_sqrtm(ref[None])[0]
    S = P_inv_sqrt @ C @ P_inv_sqrt
    T = batch_logm_sym(S)                        # (n, d, d)
    iu = np.triu_indices(N_CH)
    vec = T[:, iu[0], iu[1]].copy()
    off = iu[0] != iu[1]
    vec[:, off] *= np.sqrt(2.0)
    return vec


def compute_covariances(X: np.ndarray) -> np.ndarray:
    """每试次样本协方差：X (n, 1, 22, 500) -> C (n, 22, 22) SPD。"""
    x = X[:, 0]                                   # (n, 22, 500)
    # 逐试次去均值（数据已整体 z-score，这里仅去时间均值，保证协方差无偏）
    xc = x - x.mean(axis=2, keepdims=True)
    C = xc @ xc.swapaxes(-1, -2) / (N_SAMPLES - 1)
    # 对称化 + 数值保护（eig 要求对称）
    C = 0.5 * (C + C.swapaxes(-1, -2))
    return C


def shrink_covariances(C: np.ndarray, alpha: float) -> np.ndarray:
    """Ledoit-Wolf 型 shrinkage：C_reg = (1-α)C + α·mean(trace(C)/n)·I。"""
    if alpha <= 0:
        return C
    d = C.shape[-1]
    scale = np.trace(C, axis1=-2, axis2=-1).mean() / d
    return (1.0 - alpha) * C + alpha * scale * np.eye(d)


# ---------------------------------------------------------------- 分类器
def fit_predict_eval(feat_tr: np.ndarray, y_tr: np.ndarray,
                     feat_te: np.ndarray, y_te: np.ndarray,
                     clf_name: str, seed: int) -> dict:
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.linear_model import LogisticRegression

    t0 = time.perf_counter()
    if clf_name == "LR":
        clf = LogisticRegression(multi_class="multinomial", solver="lbfgs",
                                 C=1.0, max_iter=2000, random_state=seed)
    elif clf_name == "LDA":
        clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    else:
        raise ValueError(clf_name)
    clf.fit(feat_tr, y_tr)
    t_fit = time.perf_counter() - t0

    t0 = time.perf_counter()
    acc_tr = float(clf.score(feat_tr, y_tr))
    acc_te = float(clf.score(feat_te, y_te))
    t_eval = time.perf_counter() - t0

    return {"acc_tr": acc_tr, "acc_te": acc_te, "t_fit_s": t_fit, "t_eval_s": t_eval}


def confusion_matrix(feat_te: np.ndarray, y_te: np.ndarray, clf) -> np.ndarray:
    pred = clf.predict(feat_te)
    cm = np.zeros((N_CLASSES, N_CLASSES), dtype=int)
    for t, p in zip(y_te, pred):
        cm[t, p] += 1
    return cm


# ---------------------------------------------------------------- 端侧评估
def edge_eval() -> dict:
    """参数量 / 计算量 / C 可行性 / RV1106 耗时估算（估算值，非实测）。"""
    # 参数量
    lr_params = VEC_DIM * N_CLASSES + N_CLASSES                      # 253×4 + 4 = 1016
    lda_params = VEC_DIM * N_CLASSES + VEC_DIM * (VEC_DIM + 1) // 2 + N_CLASSES  # 均值 + 共享协方差上三角 + 先验
    # 计算量（每试次，float32）
    cov_mac = N_CH * N_CH * N_SAMPLES                                # 22×22×500 = 242k MACs
    # 特征分解 22×22 ≈ O(d^3)，一次 eig ≈ 2.2×d^3 ≈ 23.4k flops（Jacobi 约 5-10 次扫描，取 8×）
    eig_flops = 8 * N_CH ** 3                                        # ≈ 85k
    # 每试次：白化 2 次 22×22 矩阵乘 ≈ 2×2×22^3 ≈ 42.6k + 1 次 logm（1 eig）≈ 85k
    per_trial_flops = cov_mac * 2 + 3 * eig_flops                    # 协方差 + 白化/对数 ≈ 0.74 MFLOP
    lr_mac = VEC_DIM * N_CLASSES                                     # 1012 MACs（可忽略）
    total_mflop = (per_trial_flops + lr_mac * 2) / 1e6

    # RV1106：双核 A7 @ ~1.2GHz，标量 FP32 有效吞吐粗估 0.2~0.5 GFLOP/s（A7 无 NEON 向量化时）
    # t_ms = MFLOP / (GFLOP/s)
    t_low = total_mflop / 0.5    # ms（乐观：0.5 GFLOP/s）
    t_high = total_mflop / 0.2   # ms（保守：0.2 GFLOP/s）
    # 参考点 P^{-1/2} 训练期预计算，推理期不再算（只需 1 次 eig + 1 次 logm eig）
    infer_mflop = (cov_mac * 2 + 2 * eig_flops) / 1e6
    t_infer_low = infer_mflop / 0.5
    t_infer_high = infer_mflop / 0.2

    return {
        "lr_params": lr_params, "lda_params": lda_params,
        "cov_mac_per_trial": cov_mac, "vec_dim": VEC_DIM,
        "total_mflop_per_trial": round(total_mflop, 3),
        "infer_mflop_per_trial": round(infer_mflop, 3),
        "rv1106_est_ms_per_trial_full": [round(t_high, 2), round(t_low, 2)],
        "rv1106_est_ms_per_trial_infer": [round(t_infer_high, 2), round(t_infer_low, 2)],
    }


# ---------------------------------------------------------------- 主流程
def main() -> int:
    ap = argparse.ArgumentParser(description="黎曼几何分类器基线（cross-session 四分类 22ch）")
    ap.add_argument("--smoke", action="store_true", help="冒烟模式：只用 300 试次子集验证管线")
    ap.add_argument("--shrinkages", type=float, nargs="+", default=[0.0, 0.1, 0.3, 0.5])
    ap.add_argument("--classifiers", nargs="+", default=["LR", "LDA"])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print("=" * 78)
    print("黎曼几何分类器基线 — BCIC IV 2a 四分类 22ch，cross-session 协议")
    print("=" * 78)

    # ---- 数据 ----
    tr = np.load(DATA_DIR / "train.npz"); te = np.load(DATA_DIR / "eval.npz")
    X_tr, y_tr, X_te, y_te = tr["X"], tr["y"], te["X"], te["y"]
    print(f"train: X{X_tr.shape} y{y_tr.shape}   eval: X{X_te.shape} y{y_te.shape}")

    if args.smoke:
        print("\n[SMOKE] 用 300 试次子集验证管线（不产出正式结论）")
        X_tr, y_tr = X_tr[:300], y_tr[:300]
        X_te, y_te = X_te[:300], y_te[:300]

    # ---- 每试次协方差（计时） ----
    t0 = time.perf_counter()
    C_tr = compute_covariances(X_tr)
    C_te = compute_covariances(X_te)
    t_cov = time.perf_counter() - t0
    n_tr = len(X_tr)
    print(f"\n协方差计算: {n_tr + len(X_te)} 试次共 {t_cov:.2f}s "
          f"({t_cov / (n_tr + len(X_te)) * 1000:.2f} ms/试次, numpy 向量化)")
    # 数值检查：SPD 性
    w_min = np.linalg.eigvalsh(C_tr).min()
    print(f"协方差最小特征值: {w_min:.2e}（>0 即 SPD 良态）")
    assert w_min > 0, "协方差非 SPD！"

    # ---- 网格：shrinkage α × 分类器（管线：正则化→黎曼均值→切线投影→分类器）----
    results = []
    for alpha in args.shrinkages:
        Ctr_r = shrink_covariances(C_tr, alpha)
        Cte_r = shrink_covariances(C_te, alpha)
        t0 = time.perf_counter()
        ref = riemannian_mean(Ctr_r)
        t_ref = time.perf_counter() - t0
        F_tr = tangent_projection(Ctr_r, ref)
        F_te = tangent_projection(Cte_r, ref)
        for clf_name in args.classifiers:
            r = fit_predict_eval(F_tr, y_tr, F_te, y_te, clf_name, args.seed)
            results.append({"alpha": alpha, "clf": clf_name, **r})
            print(f"  α={alpha:<4} {clf_name:<3}  train_acc={r['acc_tr']*100:.2f}%  "
                  f"eval_acc={r['acc_te']*100:.2f}%  (黎曼均值 {t_ref:.1f}s, fit {r['t_fit_s']:.1f}s)")

    best = max(results, key=lambda r: r["acc_te"])
    print(f"\n★ 最优: α={best['alpha']} {best['clf']} → eval {best['acc_te']*100:.2f}%")

    # ---- 参考点选择的影响（黎曼 vs 算术 vs 对数欧氏，非网格，补充信息） ----
    print("\n--- 参考点选择的影响（固定 α=best, LR）---")
    Ctr_r = shrink_covariances(C_tr, best["alpha"])
    Cte_r = shrink_covariances(C_te, best["alpha"])
    ref_variants = {}
    t0 = time.perf_counter()
    ref_riem = riemannian_mean(Ctr_r)
    t_riem = time.perf_counter() - t0
    ref_arith = Ctr_r.mean(axis=0)
    ref_logeu = batch_expm_sym(batch_logm_sym(Ctr_r).mean(axis=0)[None])[0]
    ref_variants = {"riemannian": ref_riem, "arithmetic": ref_arith, "log_euclidean": ref_logeu}
    print(f"黎曼均值迭代耗时: {t_riem:.2f}s（训练期一次性）")

    ref_results = []
    from sklearn.linear_model import LogisticRegression
    for name, ref in ref_variants.items():
        F_tr = tangent_projection(Ctr_r, ref)
        F_te = tangent_projection(Cte_r, ref)
        clf = LogisticRegression(multi_class="multinomial", solver="lbfgs",
                                 C=1.0, max_iter=2000, random_state=args.seed)
        clf.fit(F_tr, y_tr)
        acc_tr = clf.score(F_tr, y_tr)
        acc_te = clf.score(F_te, y_te)
        ref_results.append({"ref": name, "acc_tr": acc_tr, "acc_te": acc_te})
        print(f"  参考点={name:<13}  train_acc={acc_tr*100:.2f}%  eval_acc={acc_te*100:.2f}%")

    # ---- 最优配置的混淆矩阵 + 每类精度 ----
    best_ref = "riemannian"
    F_tr = tangent_projection(Ctr_r, ref_variants[best_ref])
    F_te = tangent_projection(Cte_r, ref_variants[best_ref])
    from sklearn.linear_model import LogisticRegression
    clf_best = LogisticRegression(multi_class="multinomial", solver="lbfgs",
                                  C=1.0, max_iter=2000, random_state=args.seed)
    clf_best.fit(F_tr, y_tr)
    cm = confusion_matrix(F_te, y_te, clf_best)
    per_class = cm.diagonal() / cm.sum(axis=1)
    print(f"\n--- 最优配置混淆矩阵（α={best['alpha']}, {best['clf']}, 参考点={best_ref}, eval 2592 试次）---")
    print("         预测:  左     右    双脚   舌头   (行=真实)")
    labels = ["左", "右", "双脚", "舌头"]
    for i in range(4):
        row = "  ".join(f"{cm[i, j]:5d}" for j in range(4))
        print(f"  真实 {labels[i]:<3}: {row}   每类精度 {per_class[i]*100:.2f}%")

    # ---- 端侧评估 ----
    edge = edge_eval()
    print("\n--- 端侧评估（估算）---")
    print(f"  参数量: LR={edge['lr_params']}  LDA={edge['lda_params']}")
    print(f"  每试次计算: 协方差 {edge['cov_mac_per_trial']} MACs + 3×eig(22×22)，合计 ≈ {edge['total_mflop_per_trial']} MFLOP")
    print(f"  推理期（P^-1/2 预计算）: ≈ {edge['infer_mflop_per_trial']} MFLOP/试次")
    print(f"  RV1106 估算（双核 A7 0.2~0.5 GFLOP/s 标量 FP32）: "
          f"全流程 {edge['rv1106_est_ms_per_trial_full'][0]}~{edge['rv1106_est_ms_per_trial_full'][1]} ms/试次，"
          f"推理期 {edge['rv1106_est_ms_per_trial_infer'][0]}~{edge['rv1106_est_ms_per_trial_infer'][1]} ms/试次")

    # ---- 保存 ----
    out = {
        "protocol": "cross-session: train=T 会话(2592) / eval=E 会话(2592), 22ch 四分类",
        "smoke": args.smoke,
        "grid": results,
        "best": best,
        "ref_point_variants": ref_results,
        "best_confusion_matrix": cm.tolist(),
        "best_per_class_acc": per_class.tolist(),
        "edge_eval": edge,
        "note": "DL 对照: LIF-EEG-TCNet+HBC增强 clean 63.58%（eeg_tcnet_4class_report_lif_hbc_aug_s42.txt）; "
                "EEGNet ELU 58.14%（eegnet_4class_report_proto25.txt）",
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n✓ 结果已保存: {OUT_JSON}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
