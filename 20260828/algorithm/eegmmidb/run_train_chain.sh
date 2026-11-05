#!/bin/bash
# EEGMMIDB 训练链：smoke(2ep) → relu_s42 → lif_s42 → lif_s123
set -u
cd /Users/heyixuan/BCI/20260828/algorithm/eegmmidb
ISO="env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin"
PY="/opt/miniconda3/envs/openmmlab/bin/python"

echo "=== SMOKE $(date +%H:%M:%S) ==="
$ISO $PY train_eegmmidb_lif.py --activation relu --no-warm --tag smoke --seed 42 --epochs 2 2>&1 | tail -5

echo "=== RUN1 relu_s42 $(date +%H:%M:%S) ==="
$ISO $PY train_eegmmidb_lif.py --activation relu --no-warm --tag relu_s42 --seed 42 2>&1 | tee train_relu_s42.log | tail -6

echo "=== RUN2 lif_s42 $(date +%H:%M:%S) ==="
$ISO $PY train_eegmmidb_lif.py --activation lif --tag lif_s42 --seed 42 2>&1 | tee train_lif_s42.log | tail -8

echo "=== RUN3 lif_s123 $(date +%H:%M:%S) ==="
$ISO $PY train_eegmmidb_lif.py --activation lif --tag lif_s123 --seed 123 2>&1 | tee train_lif_s123.log | tail -8

echo "=== ALL DONE $(date +%H:%M:%S) ==="
