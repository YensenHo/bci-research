#!/bin/bash
# 监控 σ=2.0 K/R 分解 + batch15 HBC 噪声实验 全部完成后汇总结果
set -u
cd /Users/heyixuan/BCI/20260828/algorithm

FILES=(
  kd_decomposition_data_4class_cross_n2.0.json
  sign_check_data_4class_cross_hbc_quant1.0.json
  sign_check_data_4class_cross_hbc_burst1.0.json
  sign_check_data_4class_cross_hbc_impulse1.0.json
  sign_check_data_4class_cross_hbc_full1.0.json
  kd_decomposition_data_4class_cross_hbc_quant1.0.json
  kd_decomposition_data_4class_cross_hbc_burst1.0.json
  kd_decomposition_data_4class_cross_hbc_impulse1.0.json
  kd_decomposition_data_4class_cross_hbc_full1.0.json
)

deadline=$(( $(date +%s) + 10800 ))  # 最多等 3 小时
while [ $(date +%s) -lt $deadline ]; do
  missing=0
  for f in "${FILES[@]}"; do
    [ -f "$f" ] || missing=$((missing+1))
  done
  if [ $missing -eq 0 ]; then
    echo "✓ 全部 ${#FILES[@]} 个产物就绪 $(date +%H:%M:%S)"
    break
  fi
  sleep 30
done

echo ""
echo "========== σ=2.0 K/R 分解（MI 信噪比相图最后一块）=========="
cat kd_decomposition_data_4class_cross_n2.0.json 2>/dev/null || echo "缺失"

echo ""
echo "========== HBC 真实体表信道噪声：sign_check（红利符号判据）=========="
for sc in quant burst impulse full; do
  f="sign_check_data_4class_cross_hbc_${sc}1.0.json"
  echo "--- $sc ---"
  cat "$f" 2>/dev/null || echo "缺失"
done

echo ""
echo "========== HBC 真实体表信道噪声：K/R 分解 =========="
for sc in quant burst impulse full; do
  f="kd_decomposition_data_4class_cross_hbc_${sc}1.0.json"
  echo "--- $sc ---"
  cat "$f" 2>/dev/null || echo "缺失"
done

echo ""
echo "MONITOR DONE $(date +%H:%M:%S)"
