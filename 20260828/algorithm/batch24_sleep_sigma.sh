#!/bin/bash
# batch24：sleep σ 扫描 —— 定位「teacher 崩到 < chance」的 σ 阈值
# σ=1.0 已知 teacher 崩（15.3%/4.9%），扫 0.5/1.5/2.0 找边界
set -u
cd /Users/heyixuan/BCI/20260828/algorithm
PY="env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin PYTHONPATH= /opt/miniconda3/envs/openmmlab/bin/python"

for s in 0.5 1.5 2.0; do
  echo "▶ sleep σ=$s sign_check seed=42 $(date +%H:%M:%S)"
  $PY sign_check.py --data-dir sleep/data_1k_n${s} --n-channels 1 --input-length 3000 --n-classes 5 --seed 42 2>&1 | tail -6
done

echo "ALL DONE $(date +%H:%M:%S)"
