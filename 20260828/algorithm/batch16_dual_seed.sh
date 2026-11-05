#!/bin/bash
# batch16：双 seed 复核（seed 123）—— HBC 真实信道噪声 4 场景 + 纯高斯 R 转负 2 点
# 目的：把 seed 42 单 seed 的初步结论（红利恒正 / K,R 恒正 / 强 regime 符号规律）用 seed 123 复核。
#       同时补纯高斯 σ=1.5/2.0 的 K/R seed 123（R 转负的双 seed 验证）。
set -u
cd /Users/heyixuan/BCI/20260828/algorithm
PY="env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin /opt/miniconda3/envs/openmmlab/bin/python"

# --- ① HBC 4 场景 sign_check seed 123 ---
for sc in quant burst impulse full; do
  d="data_4class_cross_hbc_${sc}1.0"
  out="sign_check_${d}_s123.json"
  if [ -f "$out" ]; then echo "跳过 $d sign_check s123（已存在）"; continue; fi
  echo "▶ HBC[$sc] sign_check seed=123 $(date +%H:%M:%S)"
  $PY sign_check.py --data-dir "$d" --n-channels 22 --input-length 500 --n-classes 4 --seed 123 2>&1 | tail -6
  echo ""
done

# --- ② HBC 4 场景 kd_decomposition seed 123 ---
for sc in quant burst impulse full; do
  d="data_4class_cross_hbc_${sc}1.0"
  out="kd_decomposition_${d}_s123.json"
  if [ -f "$out" ]; then echo "跳过 $d kd s123（已存在）"; continue; fi
  echo "▶ HBC[$sc] kd_decomposition seed=123 $(date +%H:%M:%S)"
  $PY kd_decomposition.py --data-dir "$d" --n-channels 22 --input-length 500 --n-classes 4 --seed 123 2>&1 | tail -8
  echo ""
done

# --- ③ 纯高斯 σ=1.5/2.0 K/R seed 123（R 转负双 seed） ---
for s in 1.5 2.0; do
  d="data_4class_cross_n${s}"
  out="kd_decomposition_${d}_s123.json"
  if [ -f "$out" ]; then echo "跳过 $d kd s123（已存在）"; continue; fi
  echo "▶ 纯高斯 σ=$s kd_decomposition seed=123 $(date +%H:%M:%S)"
  $PY kd_decomposition.py --data-dir "$d" --n-channels 22 --input-length 500 --n-classes 4 --seed 123 2>&1 | tail -8
  echo ""
done

echo "ALL DONE $(date +%H:%M:%S)"
