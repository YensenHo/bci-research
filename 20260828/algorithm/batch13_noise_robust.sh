#!/bin/bash
# batch13：信噪比发现稳健性补强 —— CHB-MIT 第三数据集加噪声 + K/R 分解补 seed 123
set -u
cd /Users/heyixuan/BCI/20260828/algorithm
PY="env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin /opt/miniconda3/envs/openmmlab/bin/python"

# CHB-MIT 加噪声（teacher 弱基线，看负红利是否翻正）
for s in 0.5 1.0; do
  for seed in 42 123; do
    echo "▶ CHB-MIT σ=$s seed=$seed $(date +%H:%M:%S)"
    $PY sign_check.py --data-dir "chbmit/data_n$s" --n-channels 23 --input-length 512 --n-classes 2 --seed $seed 2>&1 | tail -7
  done
done

# K/R 分解补 seed 123
echo "▶ BCIC σ=1.0 K+R seed123 $(date +%H:%M:%S)"
$PY kd_decomposition.py --data-dir data_4class_cross_n1.0 --n-channels 22 --input-length 500 --n-classes 4 --seed 123 2>&1 | tail -9

echo "▶ EEGMMIDB σ=0.5 K+R seed123 $(date +%H:%M:%S)"
$PY kd_decomposition.py --data-dir eegmmidb/data_n0.5 --n-channels 21 --input-length 320 --n-classes 4 --seed 123 2>&1 | tail -9

echo "ALL DONE $(date +%H:%M:%S)"
