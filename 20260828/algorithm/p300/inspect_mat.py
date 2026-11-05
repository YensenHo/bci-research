#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""探查 BNCI 008-2014 .mat 结构（data 为 mat_struct，递归列出字段与形状）。"""
from pathlib import Path
import scipy.io as sio
import numpy as np

RAW = Path("/Users/heyixuan/BCI/20260828/algorithm/p300/data/raw")


def describe(name, v, depth=0):
    pad = "  " * depth
    if isinstance(v, sio.matlab.mio5_params.mat_struct):
        print(f"{pad}{name}: mat_struct with fields:")
        for fn in v._fieldnames:
            fv = getattr(v, fn)
            describe(fn, fv, depth + 1)
    elif isinstance(v, np.ndarray):
        if v.dtype.names:
            print(f"{pad}{name}: ndarray({v.shape}) with named fields {v.dtype.names}")
            for n in v.dtype.names:
                col = v[n]
                if col.size <= 40:
                    print(f"{pad}  {n}: {col.ravel()}")
                else:
                    print(f"{pad}  {n}: {col.ravel()[:20]} ... (len {col.size})")
        else:
            print(f"{pad}{name}: ndarray shape={v.shape} dtype={v.dtype}")
            if v.size <= 30 and v.size > 0:
                print(f"{pad}  values: {np.asarray(v).ravel()}")
    elif isinstance(v, (int, float, str)):
        print(f"{pad}{name}: scalar {type(v).__name__} = {v}")
    else:
        print(f"{pad}{name}: {type(v)}")


f = RAW / "A01.mat"
print(f"file={f.name} bytes={f.stat().st_size}")
m = sio.loadmat(f, squeeze_me=True, struct_as_record=False, chars_as_strings=True)
for k in m.keys():
    if k.startswith("__"):
        continue
    describe(k, m[k])
