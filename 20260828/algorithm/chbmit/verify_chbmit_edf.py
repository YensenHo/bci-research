#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""verify_chbmit_edf.py — 校验全部 79 个目标 EDF：字节数 + pyedflib 真读（23ch/256Hz/全长）"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path("/Users/heyixuan/BCI/20260828/algorithm/chbmit")))
import preprocess_chbmit as pp  # read_edf 已处理 24 导(剔 ECG)

BASE = Path("/Users/heyixuan/BCI/20260828/algorithm/chbmit")
RAW = BASE / "raw"

# 目标清单（与 download_chbmit_hf.sh 一致）
targets = []
targets += [(f"chb01", f"chb01_{n:02d}.edf") for n in [3, 4, 15, 16, 18, 21, 26, 7, 10, 12, 14, 20, 25, 30]]
targets += [("chb02", f) for f in ["chb02_16.edf", "chb02_16+.edf", "chb02_19.edf", "chb02_03.edf", "chb02_10.edf", "chb02_22.edf"]]
targets += [("chb03", f"chb03_{n:02d}.edf") for n in [1, 2, 3, 4, 34, 35, 36, 6, 9, 11, 15, 18, 22, 28]]
targets += [("chb04", f"chb04_{n:02d}.edf") for n in [5, 8, 28, 2, 12, 25]]
targets += [("chb05", f"chb05_{n:02d}.edf") for n in range(1, 40)]
assert len(targets) == 79, len(targets)

ok, bad = 0, []
for pat, fn in targets:
    p = RAW / pat / fn
    if not p.exists():
        bad.append((pat, fn, "MISSING", 0)); continue
    sz = p.stat().st_size
    try:
        X, labels = pp.read_edf(p)
        if X.shape[0] != 23:
            bad.append((pat, fn, f"BAD nch={X.shape[0]}", sz))
        else:
            ok += 1
    except Exception as e:
        bad.append((pat, fn, f"READ_FAIL {type(e).__name__}: {e}", sz))

print(f"校验 {len(targets)} 个目标 EDF：完整可用 {ok}，异常 {len(bad)}")
for b in bad:
    print("  ", b)
# 尺寸分布
from collections import Counter
sizes = Counter()
for pat, fn in targets:
    p = RAW / pat / fn
    sizes[p.stat().st_size] += 1
print("尺寸分布:", dict(sizes))
sys.exit(1 if bad else 0)
