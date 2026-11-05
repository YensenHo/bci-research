#!/bin/bash
# batch19：价值3（抗噪鲁棒性）加码 —— ① 满标签 vs 低标签 teacher 软标签机制；② EMG 17类多 σ（难度边界）
set -u
cd /Users/heyixuan/BCI/20260828/algorithm
PY="env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin PYTHONPATH= /opt/miniconda3/envs/openmmlab/bin/python"

# ① 机制：BCIC cross σ=1.0（暴涨点）下，满标签 teacher 软标签 vs 低标签 teacher 软标签的红利差异
echo "▶ BCIC σ=1.0 lowlabel（满标签 vs 低标签 teacher 软标签）$(date +%H:%M:%S)"
$PY sign_check_lowlabel.py --data-dir data_4class_cross_n1.0 --n-channels 22 --input-length 500 --n-classes 4 --seed 42 2>&1 | tail -6

# ② EMG 17类多 σ（太难边界：任何 σ 都不暴涨）
for s in 0.5 1.5 2.0; do
  echo "▶ EMG σ=$s sign_check $(date +%H:%M:%S)"
  $PY sign_check.py --data-dir myo_cross_17_n${s} --n-channels 16 --input-length 256 --n-classes 17 --seed 42 2>&1 | tail -6
done

echo "ALL DONE $(date +%H:%M:%S)"
