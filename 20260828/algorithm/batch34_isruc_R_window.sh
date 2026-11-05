#!/bin/bash
# batch34：ISRUC R 窗口 —— σ=0.5/1.5/2.0 kd_decomposition，验证「R 转负窗口」跨数据集
set -u
cd /Users/heyixuan/BCI/20260828/algorithm
for s in 0.5 1.5 2.0; do
  echo "▶ ISRUC σ=$s kd_decomposition $(date +%H:%M:%S)"
  env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin PYTHONPATH= \
    /opt/miniconda3/envs/openmmlab/bin/python kd_decomposition.py \
    --data-dir isruc/data_n${s} --n-channels 1 --input-length 3000 --n-classes 5 --seed 42 2>&1 | tail -7
done
echo "ALL DONE $(date +%H:%M:%S)"
