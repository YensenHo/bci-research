#!/bin/bash
# batch7：R 翻转对称验证 —— BCIC cross（数据充足）+ EEGMMIDB 加密档（找精确翻转点）
set -u
cd /Users/heyixuan/BCI/20260828/algorithm
PY="env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin /opt/miniconda3/envs/openmmlab/bin/python"

echo "▶ BCIC cross R 翻转（数据充足，预期 R 恒正或翻转点不同）$(date +%H:%M:%S)"
$PY r_flip.py --data-dir data_4class_cross --n-channels 22 --input-length 500 --n-classes 4 \
  --fracs "0.1,0.2,0.3,0.5,1.0" --seed 42 2>&1 | tail -12

echo ""
echo "▶ EEGMMIDB 4类 R 翻转加密档（找精确翻转点）$(date +%H:%M:%S)"
$PY r_flip.py --data-dir eegmmidb/data --n-channels 21 --input-length 320 --n-classes 4 \
  --fracs "0.1,0.15,0.2,0.25,0.3,0.5,1.0" --seed 42 2>&1 | tail -14

echo "ALL DONE $(date +%H:%M:%S)"
