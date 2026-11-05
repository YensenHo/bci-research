#!/bin/bash
# batch21：脑电侧难度梯度 —— BCIC 2类（太易）+ sleep 5类（更难）σ=1.0
# 目的：验证弱信号（脑电）下「中等难度（4类）暴涨、太易/太难不暴涨」的钟形
set -u
cd /Users/heyixuan/BCI/20260828/algorithm
PY="env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin PYTHONPATH= /opt/miniconda3/envs/openmmlab/bin/python"

echo "▶ BCIC 2类 σ=1.0 sign_check seed=42 $(date +%H:%M:%S)"
$PY sign_check.py --data-dir data_2class_cross_n1.0 --n-channels 22 --input-length 500 --n-classes 2 --seed 42 2>&1 | tail -6

echo "▶ sleep 5类 σ=1.0 sign_check seed=42 $(date +%H:%M:%S)"
$PY sign_check.py --data-dir sleep/data_1k_n1.0 --n-channels 1 --input-length 3000 --n-classes 5 --seed 42 2>&1 | tail -6

echo "ALL DONE $(date +%H:%M:%S)"
