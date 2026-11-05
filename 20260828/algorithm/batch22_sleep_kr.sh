#!/bin/bash
# batch22：价值3 第四象限机制 —— sleep 5类 σ=1.0 K/R 分解（确认 teacher 崩 → K 严重负）
set -u
cd /Users/heyixuan/BCI/20260828/algorithm
PY="env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin PYTHONPATH= /opt/miniconda3/envs/openmmlab/bin/python"

for seed in 42 123; do
  echo "▶ sleep 5类 σ=1.0 kd_decomposition seed=$seed $(date +%H:%M:%S)"
  $PY kd_decomposition.py --data-dir sleep/data_1k_n1.0 --n-channels 1 --input-length 3000 --n-classes 5 --seed $seed 2>&1 | tail -9
done

echo "ALL DONE $(date +%H:%M:%S)"
