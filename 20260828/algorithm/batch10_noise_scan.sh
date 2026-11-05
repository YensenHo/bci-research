#!/bin/bash
# batch10：信噪比受控扫描 σ=1.0/1.5/2.0/3.0，找 teacher 翻弱、红利转负的临界点
set -u
cd /Users/heyixuan/BCI/20260828/algorithm
PY="env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin /opt/miniconda3/envs/openmmlab/bin/python"

for s in 1.0 1.5 2.0 3.0; do
  echo "▶ 含噪 σ=$s sign_check seed=42 $(date +%H:%M:%S)"
  $PY sign_check.py --data-dir "data_4class_cross_n$s" --n-channels 22 --input-length 500 --n-classes 4 --seed 42 2>&1 | tail -8
  echo ""
done
echo "ALL DONE $(date +%H:%M:%S)"
