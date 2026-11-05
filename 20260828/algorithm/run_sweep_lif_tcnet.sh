#!/bin/bash
# LIF-EEG-TCNet 四分类 surrogate 超参扫描（thr × T，tau=2.0 固定）
# 训练协议与 relu/ELU brief 版完全一致：epochs=150, patience=25, seed=42,
# dropout=0.25, cosine, warm-start 自 eeg_tcnet_4class_relu_brief.pth（relu 61.61%）。
# 分阶段扫描（总组数 <8）：
#   阶段1: thr ∈ {0.5, 1.0, 2.0} × T=8        （先定最优 thr）
#   阶段2: 最优 thr × T ∈ {4, 16}              （再定最优 T）
#   阶段3: 最优 thr/T × tau ∈ {1.0, 4.0}       （顺带扫 tau，可选）
set -e
cd /Users/heyixuan/BCI/20260828/algorithm
PY=/opt/miniconda3/envs/openmmlab/bin/python
run() {
  echo "=== LIF-EEG-TCNet thr=$1 tau=$2 T=$3 ==="
  env -i HOME="$HOME" PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin "$PY" train_eeg_tcnet_lif.py \
    --epochs 150 --patience 25 --seed 42 --dropout 0.25 \
    --warm-start eeg_tcnet_4class_relu_brief.pth \
    --threshold "$1" --tau "$2" --timesteps "$3" --alpha 4.0 \
    --out "eeg_tcnet_4class_lif_t${1}_tau${2}_T${3}.pth" --tag "lif_t${1}_tau${2}_T${3}"
}
# 阶段1：thr 扫描 @ T=8, tau=2.0
run 0.5 2.0 8
run 1.0 2.0 8
run 2.0 2.0 8
echo "STAGE1_DONE"
