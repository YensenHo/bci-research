#!/bin/bash
# P300 训练链：relu 基线 → LIF s42 → LIF s123（正式，各 ≤100 epoch / patience 25）
cd /Users/heyixuan/BCI/20260828/algorithm/p300
PY="env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin /opt/miniconda3/envs/openmmlab/bin/python"
echo "=== [1/3] relu_s42 基线 ==="
$PY train_p300_lif.py --activation relu --seed 42 --tag relu_s42 > reports/log_relu_s42.txt 2>&1
echo "relu exit=$?"
echo "=== [2/3] lif_s42 ==="
$PY train_p300_lif.py --activation lif --seed 42 --warm p300_2class_relu_s42.pth --tag lif_s42 > reports/log_lif_s42.txt 2>&1
echo "lif_s42 exit=$?"
echo "=== [3/3] lif_s123 ==="
$PY train_p300_lif.py --activation lif --seed 123 --warm p300_2class_relu_s42.pth --tag lif_s123 > reports/log_lif_s123.txt 2>&1
echo "lif_s123 exit=$?"
echo "=== ALL DONE ==="
