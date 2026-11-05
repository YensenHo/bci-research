#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""parse_chbmit_summaries.py — 解析 chbNN-summary.txt，输出每例 (文件, 发作区间列表)"""
import re, json
from pathlib import Path

RAW = Path("/Users/heyixuan/BCI/20260828/algorithm/chbmit/raw")
out = {}
for sf in sorted(RAW.glob("chb0*/chb0*-summary.txt")):
    pat = sf.name.split("-")[0]  # chb01
    cur_file = None
    recs = {}
    for line in sf.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        m = re.match(r"File Name:\s*(\S+)", line)
        if m:
            cur_file = m.group(1)
            recs.setdefault(cur_file, [])
            continue
        m = re.match(r"Seizure\s+(?:\d+\s+)?Start Time:\s*([\d.]+)", line)
        if m and cur_file:
            recs[cur_file].append([float(m.group(1)), None])
            continue
        m = re.match(r"Seizure\s+(?:\d+\s+)?End Time:\s*([\d.]+)", line)
        if m and cur_file and recs[cur_file] and recs[cur_file][-1][1] is None:
            recs[cur_file][-1][1] = float(m.group(1))
    # 清洗：丢弃没配对的 start（END 缺失）
    for f, segs in recs.items():
        recs[f] = [s for s in segs if s[1] is not None]
    out[pat] = recs

# 统计
for pat, recs in sorted(out.items()):
    n_seiz_files = sum(1 for f, s in recs.items() if s)
    n_total = len(recs)
    tot_seiz_s = sum(e - st for s in recs.values() for st, e in s)
    print(f"{pat}: {n_total} files, {n_seiz_files} with seizures, total seizure {tot_seiz_s:.0f}s")
    seiz_files = sorted(f for f, s in recs.items() if s)
    print(f"   seizure files: {seiz_files}")
    for f in seiz_files[:3]:
        print(f"   {f}: {recs[f][:2]}")

json.dump({p: {f: s for f, s in recs.items()} for p, recs in out.items()},
          open(RAW / "annotations.json", "w"), indent=1)
print("saved -> raw/annotations.json")
