#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EEGMMIDB 2 类（左右手，R04/08/12）预处理：窗口 [-0.5,3.5]s=4s=640样本，21ch，整体 z-score"""
import numpy as np, os, json
import pyedflib, wfdb

FS = 160.0
RUNS = [4, 8, 12]
N_CH = 21

def process(subj):
    Xs, ys = [], []
    for run in RUNS:
        base = f"data/raw/S{subj:03d}/S{subj:03d}R{run:02d}"
        if not os.path.exists(base + ".edf"):
            continue
        if not os.path.exists(base + ".event"):
            os.symlink(os.path.basename(base) + ".edf.event", base + ".event")
        ann = wfdb.rdann(base, "event")
        trials = [(int(s), 0 if a.split(" ")[0] == "T1" else 1)
                  for s, a in zip(ann.sample, ann.aux_note)
                  if a and a.split(" ")[0] in ("T1", "T2")]
        f = pyedflib.EdfReader(base + ".edf")
        for cue, lab in trials:
            s0 = int(cue - 0.5 * FS); s1 = int(cue + 3.5 * FS)
            if s1 > f.getNSamples()[0]:
                continue
            blk = np.stack([f.readSignal(c, s0, s1 - s0) for c in range(N_CH)])
            mu, sd = blk.mean(), blk.std() + 1e-9
            Xs.append(((blk - mu) / sd)[None])
            ys.append(lab)
        f._close()
    if not Xs:
        return None, None
    return np.stack(Xs, 0).astype(np.float32), np.array(ys, dtype=np.int64)

def main():
    trX, try_, evX, evy = [], [], [], []
    for s in range(1, 31):
        X, y = process(s)
        if X is None:
            continue
        print(f"S{s:03d}: {len(y)}", end="  ")
        if s <= 24:
            trX.append(X); try_.append(y)
        else:
            evX.append(X); evy.append(y)
    print()
    # 垂直拼接（n,1,21,640）
    Xt = np.concatenate(trX, axis=0); yt = np.concatenate(try_)
    Xe = np.concatenate(evX, axis=0); ye = np.concatenate(evy)
    print("train:", Xt.shape, "dist:", np.bincount(yt).tolist())
    print("eval:", Xe.shape, "dist:", np.bincount(ye).tolist())
    np.savez_compressed("data/train2c.npz", X=Xt, y=yt)
    np.savez_compressed("data/eval2c.npz", X=Xe, y=ye)
    json.dump({"n_train": len(yt), "n_eval": len(ye)}, open("data/meta2c.json", "w"))
    print("DONE")

if __name__ == "__main__":
    main()
