#!/bin/bash
# batch5：补齐剩余单 seed 点的 seed 123，全部转双 seed
set -u
cd /Users/heyixuan/BCI/20260828/algorithm
PY="env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin /opt/miniconda3/envs/openmmlab/bin/python"

run() {
  local dir="$1" ch="$2" len="$3" cls="$4" seed="$5"
  echo "▶ $dir seed=$seed $(date +%H:%M:%S)"
  $PY sign_check.py --data-dir "$dir" --n-channels "$ch" --input-length "$len" --n-classes "$cls" --seed "$seed" 2>&1 | tail -7
}

# data_4class（BCIC within）之前无 sign_check，补 seed 42 + 123
run "data_4class" 22 500 4 42
run "data_4class" 22 500 4 123

# 单 seed 点补 seed 123
run "bci2b/data" 3 500 2 123
run "eegmmidb/data2c" 21 640 2 123
run "eegmmidb/data2c9" 21 640 2 123
run "myo_cross" 16 256 12 123
run "p300/data" 8 205 2 123

echo "ALL DONE $(date +%H:%M:%S)"
