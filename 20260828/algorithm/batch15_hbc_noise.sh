#!/bin/bash
# batch15：HBC 真实体表信道噪声 —— 蒸馏红利符号判据 + K/R 分解（4 场景，σ/H=1.0 暴涨点）
# 背景：把「信噪比相图」从理想高斯升级到真实人体信道（量化+AWGN+突发+脉冲），
#       验证「可穿戴脑电端侧蒸馏部署边界」的核心问题：
#       (a) 红利符号 sign(gain)=sign(t−s) 在真实 HBC 噪声下是否稳健？
#       (b) K/R 分解里 R 的符号/幅度是否随非高斯分量（量化/突发/脉冲）移动？
# 对照基线 awgn（= data_4class_cross_n1.0，已有 sign_check + kd 结果，不重跑）。
set -u
cd /Users/heyixuan/BCI/20260828/algorithm
PY="env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin /opt/miniconda3/envs/openmmlab/bin/python"

# --- 第一轮：sign_check（红利符号判据） ---
for sc in quant burst impulse full; do
  d="data_4class_cross_hbc_${sc}1.0"
  out="sign_check_${d}.json"
  if [ -f "$out" ]; then
    echo "跳过 $d sign_check（已存在）"
    continue
  fi
  echo "▶ HBC[$sc] sign_check seed=42 $(date +%H:%M:%S)"
  $PY sign_check.py --data-dir "$d" --n-channels 22 --input-length 500 --n-classes 4 --seed 42 2>&1 | tail -8
  echo ""
done

# --- 第二轮：kd_decomposition（K/R 分解） ---
for sc in quant burst impulse full; do
  d="data_4class_cross_hbc_${sc}1.0"
  out="kd_decomposition_${d}.json"
  if [ -f "$out" ]; then
    echo "跳过 $d kd（已存在）"
    continue
  fi
  echo "▶ HBC[$sc] kd_decomposition seed=42 $(date +%H:%M:%S)"
  $PY kd_decomposition.py --data-dir "$d" --n-channels 22 --input-length 500 --n-classes 4 --seed 42 2>&1 | tail -8
  echo ""
done

echo "ALL DONE $(date +%H:%M:%S)"
