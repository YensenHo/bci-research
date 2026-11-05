#!/bin/bash
# run_delta_eval.sh — 全部 delta 权重评估（clean/8dB/8dB+burst + 输入密度 + 事件驱动 ACs + 基线对照）
set -u
ALGO=/Users/heyixuan/BCI/20260828/algorithm
PY="/opt/miniconda3/envs/openmmlab/bin/python"
ISO="env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin"

cd "$ALGO"
$ISO "$PY" eval_delta_tcnet.py \
  --ckpts eeg_tcnet_4class_delta_delta1bit_k0.5.pth,eeg_tcnet_4class_delta_delta1bit_k1.0.pth,eeg_tcnet_4class_delta_delta2bit_k1.0.pth,eeg_tcnet_4class_delta_delta1bit_k0.5_posneg.pth,eeg_tcnet_4class_delta_cont_k0.5_lr3e4.pth \
  --names delta1bit_k0.5,delta1bit_k1.0,delta2bit_k1.0,delta1bit_k0.5_posneg,delta1bit_k0.5_继续微调lr3e4 \
  --include-baseline 2>&1 | tee "$ALGO/log_delta_eval_all.log"
