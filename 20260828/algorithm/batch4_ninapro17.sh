#!/bin/bash
# batch4：ninapro17 极低天花板任务 sign_check（负红利富矿）
set -u
cd /Users/heyixuan/BCI/20260828/algorithm
PY="env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin /opt/miniconda3/envs/openmmlab/bin/python"

for seed in 42 123; do
  echo "▶ myo_cross_17 (17类) seed=$seed $(date +%H:%M:%S)"
  $PY sign_check.py --data-dir myo_cross_17 --n-channels 16 --input-length 256 --n-classes 17 --seed "$seed" 2>&1 | tail -8
done

echo "ALL DONE $(date +%H:%M:%S)"
