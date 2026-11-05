#!/bin/bash
# batch11：信噪比鲁棒性系统实验 —— 补 BCIC cross σ=1.5/2.0 + EEGMMIDB 4类 σ=0.5/1.0
# 目标：刻画「红利随噪声」完整曲线 + 跨数据集一致性（正红利暴涨 vs 负红利翻转）
set -u
cd /Users/heyixuan/BCI/20260828/algorithm
PY="env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin /opt/miniconda3/envs/openmmlab/bin/python"

# EEGMMIDB 4类（teacher 弱 → 负红利基线），快
for s in 0.5 1.0; do
  for seed in 42 123; do
    echo "▶ EEGMMIDB σ=$s seed=$seed $(date +%H:%M:%S)"
    $PY sign_check.py --data-dir "eegmmidb/data_n$s" --n-channels 21 --input-length 320 --n-classes 4 --seed $seed 2>&1 | tail -7
  done
done

# BCIC cross（teacher 强 → 正红利基线），慢
for s in 1.5 2.0; do
  for seed in 42 123; do
    echo "▶ BCIC cross σ=$s seed=$seed $(date +%H:%M:%S)"
    $PY sign_check.py --data-dir "data_4class_cross_n$s" --n-channels 22 --input-length 500 --n-classes 4 --seed $seed 2>&1 | tail -7
  done
done

echo "ALL DONE $(date +%H:%M:%S)"
