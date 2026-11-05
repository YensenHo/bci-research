#!/bin/bash
# batch16：方向2 —— 不同电极/皮肤阻抗场景 = full HBC 信道噪声扫多档 σ/H
# 背景：皮肤阻抗升高 → 信号幅度衰减 → 等效 σ/H 升高（信噪比下降）。
#       湿电极（低阻抗/高信噪比）σ/H 小，干电极（高阻抗/低信噪比）σ/H 大。
#       在 full 场景（量化+AWGN+突发+脉冲，最真实端到端退化）下扫 σ/H 档位，
#       标 R 翻转点，回答「真实人体信道噪声下 R 翻转点是否被非高斯分量移动」，
#       对照纯高斯 AWGN 的 R 翻转点 σ∈(1.0, 1.5]。
# 依赖：batch15（4 场景 × sign_check + kd_decomposition @ σ/H=1.0）全部落盘后
#       才启动，避免 MPS 并发争用（skill 红线）。
set -u
cd /Users/heyixuan/BCI/20260828/algorithm
PY="env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin /opt/miniconda3/envs/openmmlab/bin/python"

echo "batch16 启动 $(date +%H:%M:%S)，先等 batch15 完成..."

# --- 等 batch15 完成（4 场景 × sign_check + kd_decomposition = 8 个 JSON）---
deadline=$(( $(date +%s) + 14400 ))  # 最多等 4 小时
while [ $(date +%s) -lt $deadline ]; do
  missing=0
  for sc in quant burst impulse full; do
    for t in sign_check kd_decomposition; do
      f="${t}_data_4class_cross_hbc_${sc}1.0.json"
      [ -f "$f" ] || missing=$((missing+1))
    done
  done
  if [ $missing -eq 0 ]; then
    echo "✓ batch15 全部 8 个产物落盘 $(date +%H:%M:%S)"
    break
  fi
  sleep 60
done

# --- 方向2：full 场景扫 σ/H 档位（0.5/1.5/2.0/3.0；1.0 已在 batch15 里）---
for s in 0.5 1.5 2.0 3.0; do
  d="data_4class_cross_hbc_full${s}"
  out="sign_check_${d}.json"
  if [ -f "$out" ]; then
    echo "跳过 $d sign_check（已存在）"
  else
    echo "▶ full σ/H=$s sign_check seed=42 $(date +%H:%M:%S)"
    $PY sign_check.py --data-dir "$d" --n-channels 22 --input-length 500 --n-classes 4 --seed 42 2>&1 | tail -8
    echo ""
  fi
  out="kd_decomposition_${d}.json"
  if [ -f "$out" ]; then
    echo "跳过 $d kd（已存在）"
  else
    echo "▶ full σ/H=$s kd_decomposition seed=42 $(date +%H:%M:%S)"
    $PY kd_decomposition.py --data-dir "$d" --n-channels 22 --input-length 500 --n-classes 4 --seed 42 2>&1 | tail -8
    echo ""
  fi
done

echo "batch16 ALL DONE $(date +%H:%M:%S)"
