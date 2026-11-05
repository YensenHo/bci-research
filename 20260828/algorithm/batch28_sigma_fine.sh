#!/bin/bash
# batch28：sleep σ 精细扫描（0.6/0.7/0.8/0.9）—— 检验「临界相变」是否非平凡跳变
# 关键：teacher 跨过 chance（σ≈0.7~0.8）那一刻，红利是平滑过渡（平凡）还是跳变（非平凡相变）
set -u
cd /Users/heyixuan/BCI/20260828/algorithm
PY="env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin PYTHONPATH= /opt/miniconda3/envs/openmmlab/bin/python"

for s in 0.6 0.7 0.8 0.9; do
  echo "▶ sleep σ=$s sign_check seed=42 $(date +%H:%M:%S)"
  $PY sign_check.py --data-dir sleep/data_1k_n${s} --n-channels 1 --input-length 3000 --n-classes 5 --seed 42 2>&1 | tail -6
done

echo "ALL DONE $(date +%H:%M:%S)"
