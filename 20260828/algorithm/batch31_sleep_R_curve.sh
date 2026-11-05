#!/bin/bash
# batch31：sleep σ 完整 R 曲线 —— 坐实「R 中间 σ 窗口转负、两端中性」的剂量-效应
# 已有 σ=1.0 R=−15（双seed），补 0.5/1.5/2.0
set -u
cd /Users/heyixuan/BCI/20260828/algorithm
PY="env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin PYTHONPATH= /opt/miniconda3/envs/openmmlab/bin/python"

for s in 0.5 1.5 2.0; do
  echo "▶ sleep σ=$s kd_decomposition seed=42 $(date +%H:%M:%S)"
  $PY kd_decomposition.py --data-dir sleep/data_1k_n${s} --n-channels 1 --input-length 3000 --n-classes 5 --seed 42 2>&1 | tail -9
done

echo "ALL DONE $(date +%H:%M:%S)"
