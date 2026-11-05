#!/bin/bash
# batch10：P300 减训练样本 → 逼出 teacher 过拟合 → 负红利强 regime 点（补 P300 范式）
set -u
cd /Users/heyixuan/BCI/20260828/algorithm
PY="env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin /opt/miniconda3/envs/openmmlab/bin/python"

for n in 1400 700; do
  for seed in 42 123; do
    echo "▶ P300 data_n$n seed=$seed $(date +%H:%M:%S)"
    $PY sign_check.py --data-dir "p300/data_n$n" --n-channels 8 --input-length 205 --n-classes 2 --seed $seed 2>&1 | tail -7
  done
done

echo "ALL DONE $(date +%H:%M:%S)"
