#!/bin/bash
# run_delta_remaining.sh — 剩余 delta 实验顺序执行（避免 MPS 争用）
# delta1bit_k1.0 已由上一子代理完成（29.24%），跳过。
# 每个 job 用 env -i 隔离（环境铁律），日志分别落盘，失败不中断后续 job。
set -u
ALGO=/Users/heyixuan/BCI/20260828/algorithm
PY="/opt/miniconda3/envs/openmmlab/bin/python"
ISO="env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin"

run_job() {
  local name="$1"; shift
  echo "===== [$(date '+%F %T')] START $name ====="
  ( cd "$ALGO" && $ISO "$PY" "$@" ) > "$ALGO/log_delta_${name}.log" 2>&1
  local rc=$?
  echo "===== [$(date '+%F %T')] END $name rc=$rc ====="
  tail -5 "$ALGO/log_delta_${name}.log"
  return $rc
}

# ① 剩余两组训练（协议与基线一致：--aug HBC 增强）
run_job delta2bit_k1.0     train_delta_tcnet.py --aug --delta-k 1.0 --delta-bits 2 --delta-mode signed --tag delta2bit_k1.0     || echo "JOB delta2bit_k1.0 FAILED"
run_job delta1bit_k0.5_posneg train_delta_tcnet.py --aug --delta-k 0.5 --delta-bits 1 --delta-mode posneg --tag delta1bit_k0.5_posneg || echo "JOB delta1bit_k0.5_posneg FAILED"

# ② 诊断：continue_delta_tcnet（信息瓶颈 vs 训练动力学）
run_job cont_k0.5_lr3e4   continue_delta_tcnet.py --resume eeg_tcnet_4class_delta_delta1bit_k0.5.pth --lr 3e-4 --epochs 100 --patience 40 --tag cont_k0.5_lr3e4 || echo "JOB cont_k0.5_lr3e4 FAILED"

echo "===== ALL DONE $(date '+%F %T') ====="
