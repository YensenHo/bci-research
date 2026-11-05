#!/bin/bash
# batch25：sleep σ 扫描 seed 123 —— 确认「teacher 跨过 chance → 红利翻负」翻转点双 seed 稳健
set -u
cd /Users/heyixuan/BCI/20260828/algorithm
PY="env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin PYTHONPATH= /opt/miniconda3/envs/openmmlab/bin/python"

for s in 0.5 1.5 2.0; do
  echo "▶ sleep σ=$s sign_check seed=123 $(date +%H:%M:%S)"
  $PY sign_check.py --data-dir sleep/data_1k_n${s} --n-channels 1 --input-length 3000 --n-classes 5 --seed 123 2>&1 | tail -6
done

echo "ALL DONE $(date +%H:%M:%S)"
