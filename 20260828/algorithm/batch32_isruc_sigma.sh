#!/bin/bash
# batch32：ISRUC σ=1.5/2.0 sign_check —— 找 ISRUC teacher 崩的噪声阈值
set -u
cd /Users/heyixuan/BCI/20260828/algorithm
PY="env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin PYTHONPATH= /opt/miniconda3/envs/openmmlab/bin/python"

for s in 1.5 2.0; do
  echo "▶ ISRUC σ=$s sign_check $(date +%H:%M:%S)"
  $PY sign_check.py --data-dir isruc/data_n${s} --n-channels 1 --input-length 3000 --n-classes 5 --seed 42 2>&1 | tail -7
done

echo "ALL DONE $(date +%H:%M:%S)"
