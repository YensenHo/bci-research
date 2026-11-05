#!/bin/bash
# batch14：K/R 在加噪数据上的分解 —— 解释信噪比相图的机制（K、R 各随 σ 怎么变）
set -u
cd /Users/heyixuan/BCI/20260828/algorithm
PY="env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin /opt/miniconda3/envs/openmmlab/bin/python"

for s in 1.0 1.5 2.0; do
  out="kd_decomposition_data_4class_cross_n${s}.json"
  if [ -f "$out" ]; then
    echo "跳过 σ=$s K/R（已存在）"
    continue
  fi
  echo "▶ K/R 分解 含噪 σ=$s seed=42 $(date +%H:%M:%S)"
  $PY kd_decomposition.py --data-dir "data_4class_cross_n$s" --n-channels 22 --input-length 500 --n-classes 4 --seed 42 2>&1 | tail -6
  echo ""
done
echo "ALL DONE $(date +%H:%M:%S)"
