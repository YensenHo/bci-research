#!/bin/bash
# batch14：低标签代理判据验证 —— 对比 A(满标签判据) vs B(低标签判据) 谁更准
set -u
cd /Users/heyixuan/BCI/20260828/algorithm
PY="env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin PYTHONPATH= /opt/miniconda3/envs/openmmlab/bin/python"

$PY sign_check_lowlabel.py --data-dir data_4class_by_subject/A03 --n-channels 22 --input-length 500 --n-classes 4 --seed 42 2>&1 | tail -5
$PY sign_check_lowlabel.py --data-dir chbmit/data --n-channels 23 --input-length 512 --n-classes 2 --seed 42 2>&1 | tail -5
$PY sign_check_lowlabel.py --data-dir eegmmidb/data --n-channels 21 --input-length 320 --n-classes 4 --seed 42 2>&1 | tail -5

echo "ALL DONE $(date +%H:%M:%S)"
