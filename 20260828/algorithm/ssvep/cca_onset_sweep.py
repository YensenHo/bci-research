#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""onset 对齐细扫描: 窗口起点 0/0.5/1/1.5/2s × 窗口 1s/2s, 全 10 被试。
直接读原始 .mat, 复用 ssvep_cca.cca_rho_matrix。输出 JSON + 控制台表。
"""
import os, re, json
import numpy as np
import scipy.io as sio
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ssvep_cca import cca_rho_matrix, itr, FS, N_HARMONICS, N_TARGETS

RAW = "raw"
LOC = "raw/64-channels.loc"
OCCIPITAL = ["O1", "OZ", "O2", "PO3", "POZ", "PO4"]
ONSETS = [0.0, 0.5, 1.0, 1.5, 2.0]
WINDOWS = [1.0, 2.0]


def parse_loc(path):
    idx = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = re.split(r"[\s\t]+", line)
            if len(parts) < 4:
                continue
            idx[parts[3].strip().upper()] = int(parts[0]) - 1
    return idx


loc = parse_loc(LOC)
rows = [loc[c] for c in OCCIPITAL]
freqs = sio.loadmat("raw/Freq_Phase.mat")["freqs"].ravel()

# 预载所有被试枕区数据 (10, 6, 1500, 40, 6): blocks,targets,ch,time
subs = []
for s in range(1, 11):
    m = sio.loadmat(os.path.join(RAW, "S%d.mat" % s))
    d = m["data"][rows].transpose(3, 2, 0, 1)  # (6,40,6,1500)
    subs.append(d)
    print("S%d 枕区数据载入 %s" % (s, d.shape))
subs = np.stack(subs)
print("总形状 (nsub,block,target,ch,time):", subs.shape)

out = {}
for onset in ONSETS:
    o0 = int(onset * FS)
    for w in WINDOWS:
        nsamp = int(w * FS)
        X = subs[:, :, :, :, o0:o0 + nsamp]  # (10,6,40,6,nsamp)
        t = np.arange(nsamp) / FS
        Y = np.empty((N_TARGETS, 2 * N_HARMONICS, nsamp))
        for f in range(N_TARGETS):
            for h in range(1, N_HARMONICS + 1):
                Y[f, 2 * (h - 1)] = np.sin(2 * np.pi * h * freqs[f] * t)
                Y[f, 2 * (h - 1) + 1] = np.cos(2 * np.pi * h * freqs[f] * t)
        Yc = Y - Y.mean(axis=2, keepdims=True)
        Cyy = np.einsum("fpi,fqi->fpq", Yc, Yc) / nsamp
        Cyy_inv = np.linalg.inv(Cyy + np.eye(6) * 1e-6)
        accs = []
        for s in range(10):
            Xt = X[s].reshape(240, 6, nsamp)
            true_labels = np.tile(np.arange(40), 6)
            rho = cca_rho_matrix(Xt, Y, Cyy_inv)
            pred = rho.argmax(axis=1)
            accs.append((pred == true_labels).mean())
        mean_acc = float(np.mean(accs))
        key = "on%g_w%g" % (onset, w)
        out[key] = {
            "onset_s": onset, "window_s": w,
            "mean_acc": round(mean_acc, 4),
            "ITR": round(itr(mean_acc, N_TARGETS, w + 0.5), 2),
            "acc_per_sub": [round(a, 4) for a in accs],
        }
        print("onset=%.1fs win=%.1fs  acc=%.4f  ITR=%.2f"
              % (onset, w, mean_acc, itr(mean_acc, N_TARGETS, w + 0.5)))

with open("cca_onset_sweep.json", "w") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
print("已保存 cca_onset_sweep.json")
