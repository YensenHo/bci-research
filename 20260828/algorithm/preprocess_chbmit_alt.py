#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""preprocess_chbmit_alt.py — CHB-MIT 换患者划分（构造第二个癫痫强 regime 点）。

train=chb01-03（3 患者，更少），eval=chb04-05（2 患者，更多 + 含原训练患者）。
比原划分（chb01-04 / chb05）更难 → teacher 更可能过拟合弱于 student → 负红利。

含「跳过坏 EDF 文件」的 try/except（chb01_01.edf 等被 hf-mirror 截断）。

用法：
  env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin \
    /opt/miniconda3/envs/openmmlab/bin/python preprocess_chbmit_alt.py
"""
from __future__ import annotations

import os
import sys

ALGO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ALGO, "chbmit"))

import preprocess_chbmit as P  # noqa: E402

# monkey-patch：换患者划分 + 输出目录
P.TRAIN_PATIENTS = ["chb01", "chb02", "chb03"]
P.EVAL_PATIENTS = ["chb04", "chb05"]
P.OUT = P.BASE / "data_alt"


def _safe_read(path):
    try:
        return P.read_edf(path)
    except Exception:
        return None


def _safe_collect(patients, ann):
    """collect 的副本，跳过坏 EDF 文件（read_edf 抛异常则 continue）。"""
    Xs, ys, info = [], [], {}
    for pat in patients:
        for fn in sorted(ann[pat].keys()):
            p = P.RAW / pat / fn
            if not p.exists() or p.stat().st_size < 8_000_000:
                print(f"  [跳过未下载] {pat}/{fn}")
                continue
            X = _safe_read(p)
            if X is None:
                print(f"  [跳过坏文件] {pat}/{fn}")
                continue
            X, labels = X  # read_edf 返回 (X, labels) tuple
            X = P.preprocess_file(X)
            seiz = ann[pat][fn]
            Xw, y = P.windowize(X, seiz)
            info[f"{pat}/{fn}"] = {"n_windows": int(len(y)),
                                   "n_pos": int(y.sum()),
                                   "n_neg": int(len(y) - y.sum()),
                                   "seizures": seiz}
            Xs.append(Xw)
            ys.append(y)
    if not Xs:
        raise SystemExit("✗ 没有可用的 EDF 文件")
    return P.np.concatenate(Xs), P.np.concatenate(ys), info


P.collect = _safe_collect

if __name__ == "__main__":
    sys.exit(P.main())
