#!/bin/bash
# batch12：K+R 分解含噪数据 —— 看噪声让红利暴涨是 R 还是 K 主导
set -u
cd /Users/heyixuan/BCI/20260828/algorithm
PY="env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin /opt/miniconda3/envs/openmmlab/bin/python"

echo "▶ BCIC cross σ=1.0 K+R 分解（红利暴涨 +16.15 的机制）$(date +%H:%M:%S)"
$PY kd_decomposition.py --data-dir data_4class_cross_n1.0 --n-channels 22 --input-length 500 --n-classes 4 --seed 42 2>&1 | tail -9

echo "▶ EEGMMIDB σ=0.5 K+R 分解（负红利翻正的机制）$(date +%H:%M:%S)"
$PY kd_decomposition.py --data-dir eegmmidb/data_n0.5 --n-channels 21 --input-length 320 --n-classes 4 --seed 42 2>&1 | tail -9

echo "ALL DONE $(date +%H:%M:%S)"
