#!/bin/bash
# batch30：标签置换 teacher —— 验证「人为制造 below-chance teacher → 负红利」跨任务普适
# shift=1 固定错位（teacher 自信地错 → below chance）；shuffle 打乱（teacher ≈ chance，对照）
set -u
cd /Users/heyixuan/BCI/20260828/algorithm
PY="env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin PYTHONPATH= /opt/miniconda3/envs/openmmlab/bin/python"

for p in shift=1 shuffle; do
  echo "▶ EEGMMIDB 1ch permute=$p $(date +%H:%M:%S)"
  $PY sign_check_permute.py --data-dir eegmmidb/data_ch1 --n-channels 1 --input-length 320 --n-classes 4 --permute $p --seed 42 2>&1 | tail -6
done

echo "ALL DONE $(date +%H:%M:%S)"
