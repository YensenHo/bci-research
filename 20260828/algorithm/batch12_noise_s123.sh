#!/bin/bash
# batch12：σ=1.5/2.0/3.0 补 seed 123，确认负红利(σ=1.5)和异常(σ=2.0)的稳健性
set -u
cd /Users/heyixuan/BCI/20260828/algorithm
PY="env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin /opt/miniconda3/envs/openmmlab/bin/python"

for s in 1.5 2.0 3.0; do
  if [ -f "sign_check_data_4class_cross_n${s}_s123.json" ]; then
    echo "跳过 σ=$s s123（已存在）"
    continue
  fi
  echo "▶ 含噪 σ=$s sign_check seed=123 $(date +%H:%M:%S)"
  $PY sign_check.py --data-dir "data_4class_cross_n$s" --n-channels 22 --input-length 500 --n-classes 4 --seed 123 2>&1 | tail -6
  echo ""
done
echo "ALL DONE $(date +%H:%M:%S)"
