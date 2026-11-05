#!/bin/bash
# batch20：价值3 难度梯度相图 —— EMG 16ch 不同类别数（2/4/8 类）× σ=1.0 的抗噪红利
# 目的：把「难度 → 抗噪红利」从三档离散（2/4/17类）补成连续曲线，验证「中等难度窗口」
set -u
cd /Users/heyixuan/BCI/20260828/algorithm
PY="env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin PYTHONPATH= /opt/miniconda3/envs/openmmlab/bin/python"

for n in 2 4 8; do
  echo "▶ EMG ${n}类 σ=1.0 sign_check seed=42 $(date +%H:%M:%S)"
  $PY sign_check.py --data-dir myo_cross_17_c${n}_n1.0 --n-channels 16 --input-length 256 --n-classes $n --seed 42 2>&1 | tail -6
done

echo "ALL DONE $(date +%H:%M:%S)"
