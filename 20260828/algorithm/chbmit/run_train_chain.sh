#!/bin/bash
# CHB-MIT 训练链：smoke → relu_s42 → lif_s42 → lif_s123
set -u
cd /Users/heyixuan/BCI/20260828/algorithm/chbmit
ISO="env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin"
PY="/opt/miniconda3/envs/openmmlab/bin/python"

echo "=== SMOKE $(date +%H:%M:%S) ==="
$ISO $PY train_chbmit_lif.py --activation relu --tag smoke --seed 42 --smoke 2>&1 | tail -6

echo "=== RUN1 relu_s42 $(date +%H:%M:%S) ==="
$ISO $PY train_chbmit_lif.py --activation relu --tag relu_s42 --seed 42 2>&1 | tail -8

echo "=== RUN2 lif_s42 $(date +%H:%M:%S) ==="
$ISO $PY train_chbmit_lif.py --activation lif --warm chbmit_2class_relu_s42.pth --tag lif_s42 --seed 42 2>&1 | tail -10

echo "=== RUN3 lif_s123 $(date +%H:%M:%S) ==="
$ISO $PY train_chbmit_lif.py --activation lif --warm chbmit_2class_relu_s42.pth --tag lif_s123 --seed 123 2>&1 | tail -10

echo "=== ALL DONE $(date +%H:%M:%S) ==="
