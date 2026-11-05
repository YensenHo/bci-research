#!/bin/bash
# batch23：普适验证「teacher 崩到 < chance」—— 通道数梯度（EEGMMIDB 4类 1/2/4ch × σ=1.0）
# 假设：通道数（信息量）× 加噪决定 teacher 是否崩；单通道崩、多通道不崩
set -u
cd /Users/heyixuan/BCI/20260828/algorithm
PY="env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin PYTHONPATH= /opt/miniconda3/envs/openmmlab/bin/python"

for n in 1 2 4; do
  echo "▶ EEGMMIDB ${n}ch σ=1.0 sign_check seed=42 $(date +%H:%M:%S)"
  $PY sign_check.py --data-dir eegmmidb/data_ch${n}_n1.0 --n-channels $n --input-length 320 --n-classes 4 --seed 42 2>&1 | tail -6
done

echo "ALL DONE $(date +%H:%M:%S)"
