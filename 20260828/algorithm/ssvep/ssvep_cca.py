#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SSVEP 标准 CCA 检测: 信号 X(枕区 6 通道) vs 模板 Y=sin/cos(1~3 次谐波)
频率/相位来自 Freq_Phase.mat; max 典型相关 -> 预测标签。
输出: 各窗口长度 (0.5/1/2/3s) 的分类精度 + ITR (bits/min)。
ITR = 60/T * (log2(N) + p*log2(p) + (1-p)*log2((1-p)/(N-1))), T = 窗口 + 0.5s 注视转移
"""
import argparse
import json
import os
import numpy as np
import scipy.io as sio

FS = 250
N_HARMONICS = 3
N_TARGETS = 40
GAZE_S = 0.5  # 注视转移时间, 计入 ITR 的 T


def cca_rho_matrix(Xt, Y, Cyy_inv):
    """对一批 trial 计算与所有频率模板的典型相关。
    Xt: (ntrial, 6, n)  每 trial 6 通道 x n 采样
    Y:  (40, 6, n)      每频率 3 次谐波 sin/cos 模板
    Cyy_inv: (40, 6, 6) 预计算模板协方差逆
    返回 rho: (ntrial, 40)
    """
    ntrial, p, n = Xt.shape
    nf = Y.shape[0]
    Xc = Xt - Xt.mean(axis=2, keepdims=True)
    Yc = Y - Y.mean(axis=2, keepdims=True)
    # Cxx: (ntrial,6,6); Cxy: (ntrial,40,6,6)   [t: trial, c/d: X通道, s: 采样]
    Cxx = np.einsum("tcs,tds->tcd", Xc, Xc) / n
    Cxy = np.einsum("tcs,fds->tfcd", Xc, Yc) / n
    Cxx_inv = np.linalg.inv(Cxx + np.eye(p) * 1e-6)
    # rho[t,f] = sqrt(max eig( Cxx^-1 Cxy Cyy^-1 Cxy^T ))  (6x6 广义特征值)
    rho = np.zeros((ntrial, nf))
    for t in range(ntrial):
        Cxi = Cxx_inv[t]                        # (p,p)
        Cxy_t = Cxy[t]                          # (f,q,r)
        A = np.einsum("pq,fqr->fpr", Cxi, Cxy_t)     # Cxx^-1 @ Cxy   (f,p,r)
        B = np.einsum("frs,fqs->frq", Cyy_inv, Cxy_t)  # Cyy^-1 @ Cxy^T (f,r,q)
        Kf = np.einsum("fpr,frq->fpq", A, B)          # (f,p,q)
        ev = np.linalg.eigvals(Kf)  # (40, 6)
        rho[t] = np.sqrt(np.maximum(ev.real, 0).max(axis=1))
    return rho


def itr(p_acc, n_targets, t_sec):
    """p_acc: 0~1 的精度; 返回 bits/min"""
    p_acc = np.clip(p_acc, 1e-9, 1.0)
    if p_acc >= 1.0:
        return 60.0 / t_sec * np.log2(n_targets)
    term = (np.log2(n_targets) + p_acc * np.log2(p_acc)
            + (1 - p_acc) * np.log2((1 - p_acc) / (n_targets - 1)))
    return 60.0 / t_sec * term


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="ssvep_occipital.npz")
    ap.add_argument("--freq_phase", default="raw/Freq_Phase.mat")
    ap.add_argument("--out", default="cca_results.json")
    ap.add_argument("--windows", default="0.5,1,2,3")
    ap.add_argument("--smoke", action="store_true", help="仅前 2 被试")
    args = ap.parse_args()

    freqs = sio.loadmat(args.freq_phase)["freqs"].ravel()  # (40,)
    assert len(freqs) == N_TARGETS
    print("40 目标频率: %.1f ~ %.1f Hz" % (freqs.min(), freqs.max()))

    data = np.load(args.npz)
    windows = [float(w) for w in args.windows.split(",")]
    keys = ["eeg_%g" % w for w in windows]
    nsub = data[keys[0]].shape[0]
    if args.smoke:
        nsub = min(nsub, 2)
    print("被试数: %d (smoke=%s)" % (nsub, args.smoke))

    results = {}
    for w, key in zip(windows, keys):
        X = data[key][:nsub]  # (nsub, 6, 40, 6, nsamp)
        nsamp = X.shape[-1]
        t = np.arange(nsamp) / FS  # 窗口内时间, onset 对齐
        # 模板 Y: (40, 6, nsamp)
        Y = np.empty((N_TARGETS, 2 * N_HARMONICS, nsamp))
        for f in range(N_TARGETS):
            for h in range(1, N_HARMONICS + 1):
                Y[f, 2 * (h - 1)] = np.sin(2 * np.pi * h * freqs[f] * t)
                Y[f, 2 * (h - 1) + 1] = np.cos(2 * np.pi * h * freqs[f] * t)
        Yc = Y - Y.mean(axis=2, keepdims=True)
        Cyy = np.einsum("fpi,fqi->fpq", Yc, Yc) / nsamp
        Cyy_inv = np.linalg.inv(Cyy + np.eye(6) * 1e-6)

        acc_per_sub = []
        for s in range(nsub):
            Xs = X[s]  # (6 blocks, 40 targets, 6, nsamp)
            ntrial = 6 * 40
            Xt = Xs.reshape(ntrial, 6, nsamp)
            true_labels = np.tile(np.arange(40), 6)  # 每 block 内目标 0..39
            rho = cca_rho_matrix(Xt, Y, Cyy_inv)  # (240, 40)
            pred = rho.argmax(axis=1)
            acc = (pred == true_labels).mean()
            acc_per_sub.append(acc)
            print("  [%.1fs] S%d 精度: %.4f" % (w, s + 1, acc))
        mean_acc = float(np.mean(acc_per_sub))
        t_total = w + GAZE_S
        itr_val = itr(mean_acc, N_TARGETS, t_total)
        results["%.1fs" % w] = {
            "window_s": w,
            "mean_acc": mean_acc,
            "acc_per_sub": [round(a, 4) for a in acc_per_sub],
            "ITR_bits_per_min": round(itr_val, 2),
            "T_sec": t_total,
        }
        print("==> [%.1fs] 平均精度 %.4f, ITR %.2f bits/min (T=%.1fs)"
              % (w, mean_acc, itr_val, t_total))

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("已保存 %s" % args.out)


if __name__ == "__main__":
    main()
