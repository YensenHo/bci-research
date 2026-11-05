#!/bin/bash
# batch9：信噪比受控实验 —— BCIC cross 加噪声，验证「信噪比→teacher弱→红利↓/转负」因果链
set -u
cd /Users/heyixuan/BCI/20260828/algorithm
PY="env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin /opt/miniconda3/envs/openmmlab/bin/python"

for s in 0.5 1.0; do
  echo "▶ 含噪 σ=$s sign_check seed=42 $(date +%H:%M:%S)"
  $PY sign_check.py --data-dir "data_4class_cross_n$s" --n-channels 22 --input-length 500 --n-classes 4 --seed 42 2>&1 | tail -8
done

echo "ALL DONE $(date +%H:%M:%S)"
