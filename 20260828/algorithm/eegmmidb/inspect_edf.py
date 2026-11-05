#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""inspect_edf.py — 解析单个 EEGMMIDB EDF：通道数/采样率/时长/注释通道"""
import sys
from pathlib import Path

import pyedflib
import numpy as np

p = Path(sys.argv[1])
print(f"== {p.name} ==")
f = pyedflib.EdfReader(str(p))
print(f"  signals={f.signals_in_file}  file_duration={f.file_duration:.1f}s")
print(f"  sample_frequency(0)={f.getSampleFrequency(0)}")
print(f"  n_records={f.datarecords_in_file}  record_duration={f.getFileDuration()/max(f.datarecords_in_file,1):.2f}s")
chans = f.getSignalLabels()
print(f"  通道列表: {chans}")
for i in range(f.signals_in_file):
    print(f"    [{i}] {chans[i]}: fs={f.getSampleFrequency(i)} n={f.getNSamples()[i]} "
          f"unit={f.getPhysicalDimension(i)}")
# 注释
annos = f.readAnnotations()
print(f"  注释数={len(annos[0])}")
for j in range(min(10, len(annos[0]))):
    print(f"    t={annos[0][j]:.2f}s dur={annos[1][j]:.2f} text={annos[2][j]!r}")
f.close()
