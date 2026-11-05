#!/bin/bash
# batch17：HBC 机制双 seed 复核 —— HBC 四场景 kd_decomposition seed 123 + 纯高斯 σ=1.5/2.0 K/R seed 123
# 目的：① 量化打 K / 脉冲放大 K 两个机制需 seed 123 复核；② R 转负（σ=1.5）是否稳健需双 seed
set -u
cd /Users/heyixuan/BCI/20260828/algorithm
PY="env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin PYTHONPATH= /opt/miniconda3/envs/openmmlab/bin/python"

# HBC 四场景 kd_decomposition seed 123
for sc in quant burst impulse full; do
  d="data_4class_cross_hbc_${sc}1.0"
  out="kd_decomposition_${d}_s123.json"
  if [ -f "$out" ]; then echo "跳过 $out（已存在）"; continue; fi
  echo "▶ HBC[$sc] kd_decomposition seed=123 $(date +%H:%M:%S)"
  $PY kd_decomposition.py --data-dir "$d" --n-channels 22 --input-length 500 --n-classes 4 --seed 123 2>&1 | tail -8
done

# 纯高斯 σ=1.5/2.0 kd_decomposition seed 123（R 转负双 seed 复核）
for s in 1.5 2.0; do
  d="data_4class_cross_n${s}"
  out="kd_decomposition_${d}_s123.json"
  if [ -f "$out" ]; then echo "跳过 $out（已存在）"; continue; fi
  echo "▶ 纯高斯 σ=$s kd_decomposition seed=123 $(date +%H:%M:%S)"
  $PY kd_decomposition.py --data-dir "$d" --n-channels 22 --input-length 500 --n-classes 4 --seed 123 2>&1 | tail -8
done

echo "ALL DONE $(date +%H:%M:%S)"
