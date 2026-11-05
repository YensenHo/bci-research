#!/bin/bash
# run_ea_experiments.sh — 欧氏对齐叠加实验矩阵（顺序执行，避免 MPS 争抢）
# 实验组（≤6 组控制规模；每组 < 20 分钟）：
#   1. ELU no-EA（基线对照重跑）   2. ELU + EA
#   3. LIF no-EA（基线对照重跑）   4. LIF + EA（重点）
set -u
cd /Users/heyixuan/BCI/20260828/algorithm || exit 1
PY="env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin /opt/miniconda3/envs/openmmlab/bin/python"

echo "========== [1/4] ELU no-EA（基线对照） $(date '+%F %T') =========="
$PY train_eeg_tcnet_ea.py --activation elu --ea none || exit 1

echo "========== [2/4] ELU + EA $(date '+%F %T') =========="
$PY train_eeg_tcnet_ea.py --activation elu --ea train || exit 1

echo "========== [3/4] LIF no-EA（基线对照，thr=1.0/tau=2.0/T=8） $(date '+%F %T') =========="
$PY train_eeg_tcnet_ea.py --activation lif --ea none --threshold 1.0 --tau 2.0 --timesteps 8 || exit 1

echo "========== [4/4] LIF + EA（重点，thr=1.0/tau=2.0/T=8） $(date '+%F %T') =========="
$PY train_eeg_tcnet_ea.py --activation lif --ea train --threshold 1.0 --tau 2.0 --timesteps 8 || exit 1

echo "========== ALL DONE $(date '+%F %T') =========="
