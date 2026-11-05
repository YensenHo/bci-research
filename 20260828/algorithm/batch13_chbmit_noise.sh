#!/bin/bash
# batch13：跨任务信噪比验证 —— CHB-MIT(癫痫) 加噪，验证「信噪比→红利翻转」非 MI 特例
set -u
cd /Users/heyixuan/BCI/20260828/algorithm
PY="env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin /opt/miniconda3/envs/openmmlab/bin/python"

for s in 1.0 2.0 3.0; do
  if [ -f "sign_check_chbmit_data_n${s}.json" ]; then
    echo "跳过 chbmit n$s（已存在）"
    continue
  fi
  echo "▶ CHB-MIT 含噪 σ=$s sign_check seed=42 $(date +%H:%M:%S)"
  $PY sign_check.py --data-dir "chbmit/data_n$s" --n-channels 23 --input-length 512 --n-classes 2 --seed 42 2>&1 | tail -6
  echo ""
done
echo "ALL DONE $(date +%H:%M:%S)"
