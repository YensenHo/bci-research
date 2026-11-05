#!/bin/bash
# batch3：双 seed 稳健性 —— 9 被试 + CHB-MIT 补 seed 123
set -u
cd /Users/heyixuan/BCI/20260828/algorithm
PY="env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin /opt/miniconda3/envs/openmmlab/bin/python"

run_sign() {
  local dir="$1" ch="$2" len="$3" cls="$4" seed="$5"
  echo "▶ $dir seed=$seed $(date +%H:%M:%S)"
  $PY sign_check.py --data-dir "$dir" --n-channels "$ch" --input-length "$len" --n-classes "$cls" --seed "$seed" 2>&1 | tail -7
}

for s in A01 A02 A04 A05 A06 A07 A08 A09; do
  run_sign "data_4class_by_subject/$s" 22 500 4 123
done

run_sign "chbmit/data" 23 512 2 123

echo "ALL DONE $(date +%H:%M:%S)"
