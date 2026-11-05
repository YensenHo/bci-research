#!/bin/bash
# sign_check 批量扩展：把「红利符号判据」验证到全部可用数据集
# 目标：扩数据点 + 构造负红利样本（data_4class_by_subject 288样本/被试）
set -u
cd /Users/heyixuan/BCI/20260828/algorithm
PY="env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin /opt/miniconda3/envs/openmmlab/bin/python"

run() {
  local dir="$1" ch="$2" len="$3" cls="$4"
  echo "================================================================"
  echo "▶ $dir  (ch=$ch len=$len cls=$cls)  $(date +%H:%M:%S)"
  $PY sign_check.py --data-dir "$dir" --n-channels "$ch" --input-length "$len" --n-classes "$cls" 2>&1 | tail -9
  echo "--- exit=$? ---"
}

# ① 9 个被试的 within-subject 数据（每个 288 样本，35K 教师大概率过拟合 → 负红利样本来源）
for s in A01 A02 A03 A04 A05 A06 A07 A08 A09; do
  run "data_4class_by_subject/$s" 22 500 4
done

# ② CHB-MIT 癫痫（跨患者 2 类，低信噪比医疗数据）
run "chbmit/data" 23 512 2

# ③ EPFL P300 speller（跨被试 2 类，8 通道）
run "p300/data" 8 205 2

# ④ DEAP 情感（跨被试 2 类，32 通道，样本最多、最慢）
run "deap/data" 32 640 2

echo "ALL DONE $(date +%H:%M:%S)"
