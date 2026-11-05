#!/bin/bash
# batch8：K+R 分解 seed 123（A03 + CHB-MIT），补 K 符号规律的双 seed 稳健性
set -u
cd /Users/heyixuan/BCI/20260828/algorithm
PY="env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin /opt/miniconda3/envs/openmmlab/bin/python"

echo "▶ A03 K+R seed=123 $(date +%H:%M:%S)"
$PY kd_decomposition.py --data-dir data_4class_by_subject/A03 --n-channels 22 --input-length 500 --n-classes 4 --seed 123 2>&1 | tail -10

echo "▶ CHB-MIT K+R seed=123 $(date +%H:%M:%S)"
$PY kd_decomposition.py --data-dir chbmit/data --n-channels 23 --input-length 512 --n-classes 2 --seed 123 2>&1 | tail -10

echo "ALL DONE $(date +%H:%M:%S)"
