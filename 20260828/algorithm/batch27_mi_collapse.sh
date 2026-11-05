#!/bin/bash
# batch27：第二个数据集复现 teacher 崩 —— EEGMMIDB 1ch 强噪声 σ=2.0/3.0
# 假设：更强噪声让 MI 4类单通道也「不可解」，teacher 崩（三条件通用，非 sleep 特例）
set -u
cd /Users/heyixuan/BCI/20260828/algorithm
PY="env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin PYTHONPATH= /opt/miniconda3/envs/openmmlab/bin/python"

for s in 2.0 3.0; do
  echo "▶ EEGMMIDB 1ch σ=$s sign_check seed=42 $(date +%H:%M:%S)"
  $PY sign_check.py --data-dir eegmmidb/data_ch1_n${s} --n-channels 1 --input-length 320 --n-classes 4 --seed 42 2>&1 | tail -6
done

echo "ALL DONE $(date +%H:%M:%S)"
