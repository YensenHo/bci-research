#!/bin/bash
# 下一批实验：K+R 分解（负红利对称点）+ 双 seed 稳健性
set -u
cd /Users/heyixuan/BCI/20260828/algorithm
PY="env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin /opt/miniconda3/envs/openmmlab/bin/python"

run_kd() {
  local dir="$1" ch="$2" len="$3" cls="$4"
  echo "================================================================"
  echo "▶ K+R 分解 $dir  $(date +%H:%M:%S)"
  $PY kd_decomposition.py --data-dir "$dir" --n-channels "$ch" --input-length "$len" --n-classes "$cls" 2>&1 | tail -12
}

run_sign() {
  local dir="$1" ch="$2" len="$3" cls="$4" seed="$5"
  echo "================================================================"
  echo "▶ sign_check $dir seed=$seed  $(date +%H:%M:%S)"
  $PY sign_check.py --data-dir "$dir" --n-channels "$ch" --input-length "$len" --n-classes "$cls" --seed "$seed" 2>&1 | tail -9
}

# ① K+R 分解：A03（负红利第二点，teacher 弱 → 预期 K<0）
run_kd "data_4class_by_subject/A03" 22 500 4

# ② K+R 分解：CHB-MIT（teacher 弱 −9.04pp 但红利 0 → 预期 K≈0 天花板压缩）
run_kd "chbmit/data" 23 512 2

# ③ 双 seed：A03（关键负红利点）补 seed 123
run_sign "data_4class_by_subject/A03" 22 500 4 123

echo "ALL DONE $(date +%H:%M:%S)"
