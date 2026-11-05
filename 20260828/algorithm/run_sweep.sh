#!/bin/bash
# LIF-EEGNet 四分类 surrogate 超参扫描（thr/tau/T）
set -e
cd /Users/heyixuan/BCI/20260828/algorithm
PY=/opt/miniconda3/envs/openmmlab/bin/python
run() {
  echo "=== thr=$1 tau=$2 T=$3 ==="
  env -i HOME="$HOME" PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin "$PY" train_4class.py \
    --model lif --epochs 150 --patience 25 --seed 42 --dropout 0.25 --cosine \
    --warm-start eegnet_4class.pth \
    --threshold "$1" --tau "$2" --timesteps "$3" --alpha 4.0 \
    --out "lif_4class_t${1}_tau${2}_T${3}.pth" --tag "t${1}_tau${2}_T${3}"
}
run 0.5 2.0 8
run 0.25 2.0 8
run 1.0 4.0 8
run 0.5 2.0 16
echo "SWEEP_DONE"
