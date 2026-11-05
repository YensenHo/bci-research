#!/bin/bash
# batch29：σ=0.6/0.7 seed 123 —— 确认「临界放大」跳变（+27.75 → −7.16）是否双 seed 稳健
# 排除 hard 单 seed 波动假象，检验真临界相变
set -u
cd /Users/heyixuan/BCI/20260828/algorithm
PY="env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin PYTHONPATH= /opt/miniconda3/envs/openmmlab/bin/python"

for s in 0.6 0.7; do
  echo "▶ sleep σ=$s sign_check seed=123 $(date +%H:%M:%S)"
  $PY sign_check.py --data-dir sleep/data_1k_n${s} --n-channels 1 --input-length 3000 --n-classes 5 --seed 123 2>&1 | tail -6
done

echo "ALL DONE $(date +%H:%M:%S)"
