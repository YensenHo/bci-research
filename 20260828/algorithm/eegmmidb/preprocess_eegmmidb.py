#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EEGMMIDB 四分类 MI 预处理：
- 6 个 MI run（R04/06/08/10/12/14）每被试
- 21 运动区通道（Fc5..Cp6，索引 0-20），160Hz
- cue 后 [0.5,2.5]s = 320 样本，逐试次 z-score
- 标签：R04/08/12 单侧 run: T1=左手(0) T2=右手(1)；R06/10/14 双侧 run: T1=双拳(2) T2=双脚(3)
- 划分：S001-S024 训练、S025-S030 评估（跨被试 80/20）"""
import numpy as np, os, sys, json
import pyedflib, wfdb

RAW = "data/raw"
OUT = "data"
FS = 160.0
SINGLE_RUNS = {4, 8, 12}   # 单侧：T1=左手 T2=右手
DUAL_RUNS = {6, 10, 14}    # 双侧：T1=双拳 T2=双脚
N_CH = 21                   # Fc5..Cp6
WIN_START, WIN_END = 0.5, 2.5  # cue 后窗口（s）
SUBJECTS = list(range(1, 31))
TRAIN_SUBS = list(range(1, 25))
EVAL_SUBS = list(range(25, 31))

def run_label_map(run_no):
    if run_no in SINGLE_RUNS:
        return {"T1": 0, "T2": 1}
    if run_no in DUAL_RUNS:
        return {"T1": 2, "T2": 3}
    return None

def parse_event(subj, run_no):
    """返回 [(cue_sample, label)]"""
    base = os.path.join(RAW, f"S{subj:03d}", f"S{subj:03d}R{run_no:02d}")
    # 软链 .event（若没有）
    ev = base + ".event"
    if not os.path.exists(ev):
        os.symlink(os.path.basename(base) + ".edf.event", ev)
    ann = wfdb.rdann(base, "event")
    label_map = run_label_map(run_no)
    trials = []
    for s, aux in zip(ann.sample, ann.aux_note):
        if not aux:
            continue
        code = aux.split(" ")[0] if aux else ""
        if code in label_map:
            trials.append((int(s), label_map[code]))
    return trials

def process_subject(subj):
    """返回 X (n,21,320), y (n,)"""
    Xs, ys = [], []
    for run_no in sorted(SINGLE_RUNS | DUAL_RUNS):
        edf_path = os.path.join(RAW, f"S{subj:03d}", f"S{subj:03d}R{run_no:02d}.edf")
        if not os.path.exists(edf_path):
            print(f"  缺 {edf_path}"); continue
        trials = parse_event(subj, run_no)
        f = pyedflib.EdfReader(edf_path)
        n_samples = f.getNSamples()[0]
        for cue_sample, label in trials:
            s0 = int(cue_sample + WIN_START * FS)
            s1 = int(cue_sample + WIN_END * FS)
            if s1 > n_samples:
                continue
            # 读 21 通道
            block = np.zeros((N_CH, s1 - s0), dtype=np.float64)
            for c in range(N_CH):
                block[c] = f.readSignal(c, s0, s1 - s0)
            # 整体标量 z-score（跨通道+时间，与 2a/2b 协议一致）
            mu = block.mean()
            sd = block.std() + 1e-9
            block = (block - mu) / sd
            Xs.append(block[None])  # (1,21,320)，block 已整体归一
            ys.append(label)
        f._close()
    if not Xs:
        return None, None
    # np.stack 保留单例通道维 → (n,1,21,320)（np.concatenate 会吞掉）
    X = np.stack(Xs, axis=0).astype(np.float32)
    y = np.array(ys, dtype=np.int64)
    return X, y

def main():
    train_X, train_y, eval_X, eval_y = [], [], [], []
    for subj in SUBJECTS:
        X, y = process_subject(subj)
        if X is None:
            print(f"S{subj:03d}: 无数据"); continue
        print(f"S{subj:03d}: {len(y)} 试次  类别分布 {np.bincount(y, minlength=4).tolist()}")
        if subj in TRAIN_SUBS:
            train_X.append(X); train_y.append(y)
        else:
            eval_X.append(X); eval_y.append(y)
    os.makedirs(OUT, exist_ok=True)
    np.savez_compressed(os.path.join(OUT, "train.npz"),
                        X=np.concatenate(train_X, axis=0), y=np.concatenate(train_y, axis=0))
    np.savez_compressed(os.path.join(OUT, "eval.npz"),
                        X=np.concatenate(eval_X, axis=0), y=np.concatenate(eval_y, axis=0))
    meta = {"fs": FS, "n_ch": N_CH, "win": [WIN_START, WIN_END],
            "train_subs": TRAIN_SUBS, "eval_subs": EVAL_SUBS,
            "classes": ["left_hand", "right_hand", "both_fists", "both_feet"],
            "n_train": len(np.concatenate(train_X)), "n_eval": len(np.concatenate(eval_X)),
            "train_dist": np.bincount(np.concatenate(train_y), minlength=4).tolist(),
            "eval_dist": np.bincount(np.concatenate(eval_y), minlength=4).tolist()}
    with open(os.path.join(OUT, "meta.json"), "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    print("TRAIN:", meta["n_train"], meta["train_dist"])
    print("EVAL :", meta["n_eval"], meta["eval_dist"])
    print("DONE")

if __name__ == "__main__":
    main()
