#!/bin/bash
# batch18：新数据点 —— HBC full σ=1.5（验证 R 转负是否非高斯特有）+ EMG σ=1.0（第三范式抗噪鲁棒性）
set -u
cd /Users/heyixuan/BCI/20260828/algorithm
PY="env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin PYTHONPATH= /opt/miniconda3/envs/openmmlab/bin/python"

# HBC full σ=1.5
echo "▶ HBC full σ=1.5 sign_check seed=42 $(date +%H:%M:%S)"
$PY sign_check.py --data-dir data_4class_cross_hbc_full1.5 --n-channels 22 --input-length 500 --n-classes 4 --seed 42 2>&1 | tail -7
echo "▶ HBC full σ=1.5 kd_decomposition seed=42 $(date +%H:%M:%S)"
$PY kd_decomposition.py --data-dir data_4class_cross_hbc_full1.5 --n-channels 22 --input-length 500 --n-classes 4 --seed 42 2>&1 | tail -9

# EMG σ=1.0
echo "▶ EMG σ=1.0 sign_check seed=42 $(date +%H:%M:%S)"
$PY sign_check.py --data-dir myo_cross_17_n1.0 --n-channels 16 --input-length 256 --n-classes 17 --seed 42 2>&1 | tail -7
echo "▶ EMG σ=1.0 kd_decomposition seed=42 $(date +%H:%M:%S)"
$PY kd_decomposition.py --data-dir myo_cross_17_n1.0 --n-channels 16 --input-length 256 --n-classes 17 --seed 42 2>&1 | tail -9

echo "ALL DONE $(date +%H:%M:%S)"
