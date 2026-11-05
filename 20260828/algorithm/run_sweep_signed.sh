#!/bin/bash
# Signed LIF-EEGNet 四分类 surrogate 超参扫描（thr/tau/T）
# 训练协议与 LIF baseline 完全一致：epochs=150, patience=25, seed=42, dropout=0.25, cosine,
# warm-start 自 eegnet_4class.pth。仅激活层换成 SignedLIFActivation（正负脉冲分离）。
set -e
cd /Users/heyixuan/BCI/20260828/algorithm
PY=/opt/miniconda3/envs/openmmlab/bin/python
run() {
  echo "=== signed thr=$1 tau=$2 T=$3 ==="
  env -i HOME="$HOME" PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin "$PY" train_signed_lif.py \
    --epochs 150 --patience 25 --seed 42 --dropout 0.25 --cosine \
    --warm-start eegnet_4class.pth \
    --threshold "$1" --tau "$2" --timesteps "$3" --alpha 4.0 \
    --out "signed_lif_4class_t${1}_tau${2}_T${3}.pth" --tag "t${1}_tau${2}_T${3}"
}
run 1.0 2.0 8
run 0.5 2.0 8
run 0.25 2.0 8
run 1.0 4.0 8
run 2.0 2.0 8
run 0.5 2.0 16
run 0.5 2.0 4
echo "SWEEP_DONE"
