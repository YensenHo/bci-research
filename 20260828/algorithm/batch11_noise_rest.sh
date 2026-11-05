#!/bin/bash
# batch11：σ=1.5/2.0/3.0 补跑（σ=1.0 已完成），等 MPS 空闲后逐个完成
set -u
cd /Users/heyixuan/BCI/20260828/algorithm
PY="env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin /opt/miniconda3/envs/openmmlab/bin/python"

for s in 1.5 2.0 3.0; do
  # 若结果已存在则跳过
  if [ -f "sign_check_data_4class_cross_n${s}.json" ]; then
    echo "跳过 σ=$s（已存在）"
    continue
  fi
  echo "▶ 含噪 σ=$s sign_check seed=42 $(date +%H:%M:%S)"
  $PY sign_check.py --data-dir "data_4class_cross_n$s" --n-channels 22 --input-length 500 --n-classes 4 --seed 42 2>&1 | tail -6
  echo ""
done
echo "ALL DONE $(date +%H:%M:%S)"
