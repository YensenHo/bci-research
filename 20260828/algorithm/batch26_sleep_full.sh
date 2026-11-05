#!/bin/bash
# batch26：sleep 全量（11076）σ=1.0 —— 验证「少样本是 teacher 崩的关键条件」
# 若全量 teacher 不崩（> chance）而 data_1k（1000）崩（< chance），坐实「少样本 → teacher 过拟合 → 崩」
set -u
cd /Users/heyixuan/BCI/20260828/algorithm
PY="env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin PYTHONPATH= /opt/miniconda3/envs/openmmlab/bin/python"

echo "▶ sleep 全量 σ=1.0 sign_check seed=42 $(date +%H:%M:%S)"
$PY sign_check.py --data-dir sleep/data_n1.0 --n-channels 1 --input-length 3000 --n-classes 5 --seed 42 2>&1 | tail -6

echo "ALL DONE $(date +%H:%M:%S)"
