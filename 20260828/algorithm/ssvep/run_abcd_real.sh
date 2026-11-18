#!/bin/bash
# 真实数据 A/B/C/D：4类+40类 × 三种子(42/123/2024) + 聚合分析
set -uo pipefail
cd /Users/heyixuan/BCI/20260828/algorithm/ssvep
PY=/opt/miniconda3/envs/openmmlab/bin/python
OUT=/Users/heyixuan/BCI/gate1_ssvep_local
NPZ="$OUT/occipital_onset05.npz"
FREQ="raw/Freq_Phase.mat"
NPZ_SHA=e26a3d0b08a74bf90f0cd0a84a68ec822ca162a29a9534fe2f240235d7aefd9b
FREQ_SHA=6059f712688ec9e5df0beace9244dc0a4b03c418dacdef86bac50cf2b95f71b5

for task in 4 40; do
  for seed in 42 123 2024; do
    echo "=== task=$task seed=$seed start ==="
    if env -i HOME="$HOME" PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin PYTHONPATH= \
        "$PY" -u gate1_frequency_abcd.py --mode real --task "$task" --npz "$NPZ" --freq-phase "$FREQ" \
        --expected-npz-sha256 "$NPZ_SHA" --expected-freq-sha256 "$FREQ_SHA" \
        --seed "$seed" --device cpu --output "$OUT/abcd_${task}_${seed}.json"; then
      echo "=== task=$task seed=$seed OK ==="
    else
      echo "=== task=$task seed=$seed FAILED ==="
    fi
  done
  if env -i HOME="$HOME" PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin PYTHONPATH= \
      "$PY" analyse_frequency_abcd.py "$OUT/abcd_${task}_42.json" "$OUT/abcd_${task}_123.json" "$OUT/abcd_${task}_2024.json" \
      --output "$OUT/abcd_${task}_paired.json"; then
    echo "=== analyse task=$task OK ==="
  else
    echo "=== analyse task=$task FAILED ==="
  fi
done
echo "ALL DONE"
