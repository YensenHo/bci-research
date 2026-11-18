#!/bin/bash
# 修正窗口后的 3 种子 × 6 变体矩阵（MPS）。每种子内部顺序跑(relu→5个LIF变体)，种子间并行。
set -uo pipefail
cd /Users/heyixuan/BCI/20260828/algorithm

PY=/opt/miniconda3/envs/openmmlab/bin/python
DATA=data_4class_stca
RESULTS=outputs/bcic2a_stca_corrected_20261118
mkdir -p "$RESULTS"

run_train() {  # variant seed [warm-start]
  local v=$1 seed=$2 ws=${3:-}
  local out="$RESULTS/$v-$seed"
  if [ -f "$out/summary.json" ] && [ -f "$out/model.pth" ]; then
    echo "[$seed] skip $v (already done)"
    return 0
  fi
  mkdir -p "$out"
  local cmd=(env -i HOME="$HOME" PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin PYTHONPATH=
             "$PY" -u train_eeg_tcnet_stca.py --data-dir "$DATA" --output-dir "$out"
             --variant "$v" --seed "$seed" --device mps)
  if [ -n "$ws" ]; then cmd+=(--warm-start "$ws"); fi
  echo "[$seed] start $v"
  if "${cmd[@]}" >> "$out/run.log" 2>&1; then
    echo "[$seed] done $v"
  else
    echo "[$seed] FAILED $v (see $out/run.log)"
  fi
}

run_seed() {
  local seed=$1
  run_train relu "$seed"
  local ws="$RESULTS/relu-$seed/model.pth"
  run_train baseline "$seed" "$ws"
  run_train signed_fixed "$seed" "$ws"
  run_train adaptive_only "$seed" "$ws"
  run_train attention "$seed" "$ws"
  run_train attention_adaptive "$seed" "$ws"
}

for s in 42 123 2024; do
  run_seed "$s" > "$RESULTS/seed-$s.log" 2>&1 &
done
wait

echo "=== verify ==="
missing=0
for s in 42 123 2024; do
  for v in relu baseline signed_fixed adaptive_only attention attention_adaptive; do
    if [ ! -f "$RESULTS/$v-$s/summary.json" ] || [ ! -f "$RESULTS/$v-$s/model.pth" ]; then
      echo "MISSING: $v-$s"; missing=1
    fi
  done
done
if [ "$missing" = "1" ]; then echo "RESULT: INCOMPLETE"; exit 1; fi
echo "RESULT: ALL 18 RUNS COMPLETE"
