#!/bin/bash
# LIF-EEG-TCNet 阶段2/3：T 扫描（thr=1.0，tau=2.0）+ tau 扫描（可选）
# 阶段1结论：thr=0.5 与 1.0 打平（均 62.42% > relu 61.61%），thr=2.0 略降（61.54%）。
# 阶段2：thr=1.0 × T ∈ {4, 16}，定最优 T。
# 阶段3（注释开关）：最优 thr/T × tau ∈ {1.0, 4.0}，顺带扫 tau。
set -e
cd /Users/heyixuan/BCI/20260828/algorithm
PY=/opt/miniconda3/envs/openmmlab/bin/python
run() {
  echo "=== LIF-EEG-TCNet thr=$1 tau=$2 T=$3 ==="
  env -i HOME="$HOME" PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin "$PY" -u train_eeg_tcnet_lif.py \
    --epochs 150 --patience 25 --seed 42 --dropout 0.25 \
    --warm-start eeg_tcnet_4class_relu_brief.pth \
    --threshold "$1" --tau "$2" --timesteps "$3" --alpha 4.0 \
    --out "eeg_tcnet_4class_lif_t${1}_tau${2}_T${3}.pth" --tag "lif_t${1}_tau${2}_T${3}"
}
# 阶段2：T 扫描 @ thr=1.0, tau=2.0
run 1.0 2.0 4
run 1.0 2.0 16
echo "STAGE2_DONE"
