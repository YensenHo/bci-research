#!/usr/bin/env bash
# ============================================================
# reproduce.sh — HBC×BCI 端侧验证一键复现
# （2026-08-20 板载实测版，真实数字见 board_verification_log.md）
#
# 用法:
#   bash reproduce.sh            # 跑可离线阶段：verify + board（默认）
#   bash reproduce.sh all        # 全流程（含重下数据+训练，需联网+torch）
#   bash reproduce.sh data       # 重下 BCIC IV 2a + 预处理（~420MB，需联网）
#   bash reproduce.sh train      # 训练 EEGNet（需 torch + data）
#   bash reproduce.sh export     # 导出 ONNX + C 权重头文件
#   bash reproduce.sh verify     # C vs PyTorch 一致性 + 26 项自检
#   bash reproduce.sh board      # 板载真实推理（需 adb 连板）
#
# 关键结论（已板载实测复现）:
#   EEGNet 76.39% (1296试次/9被试, BCIC IV 2a) | C推理 18.50ms
#   HBC位深 5bit 保精度 | McNemar p=0.0028 | 峰值RSS 1.82MB
# ============================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # → BCI/
DEPLOY="$ROOT/luckfox-deploy"
PY="/opt/miniconda3/envs/openmmlab/bin/python"              # 含 torch/numpy/scipy
ADB="$HOME/luckfox/platform-tools/adb"
SERIAL="5caaa04f9cdb6ba2"

say() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }

stage_data() {
  say "[1/5] 重下 BCIC IV 2a 原始 .mat + 预处理（当前 data/raw 已清理，需重下）"
  cd "$DEPLOY"
  $PY data/download_bcic2a.py
  $PY data/preprocess.py
}

stage_train() {
  say "[2/5] 训练 EEGNet（8ch 运动区子集，76.39%）"
  cd "$DEPLOY"
  $PY model/train.py
}

stage_export() {
  say "[3/5] 导出 ONNX + C 权重头文件（eegnet_weights.h / fc_weights.h）"
  cd "$DEPLOY"
  $PY model/export_onnx.py
  $PY model/export_c_weights.py
}

stage_verify() {
  say "[4/5] C vs PyTorch 一致性（应 ~8e-07）+ 26 项自检"
  cd "$DEPLOY"
  $PY model/verify_c_infer.py --n 20
  $PY scripts/verify.py
}

stage_board() {
  say "[5/5] 板载真实推理（RV1106 CPU 版，二进制已在 /root）"
  # 单次前向（应 ~18.5ms）
  $ADB -s "$SERIAL" shell "cd /root && ./eegnet_infer"
  # McNemar 显著性（6480 配对，应 p≈0.0028）
  $ADB -s "$SERIAL" shell "cd /root && ./hbc_mcnemar /root/fusion_test_full.bin"
  # SNR 瀑布曲线（1296×5×7，约 30-40 分钟）
  $ADB -s "$SERIAL" shell "cd /root && ./hbc_snr_sweep /root/fusion_test_full.bin"
}

case "${1:-verify_board}" in
  all)          stage_data; stage_train; stage_export; stage_verify; stage_board ;;
  data)         stage_data ;;
  train)        stage_train ;;
  export)       stage_export ;;
  verify)       stage_verify ;;
  board)        stage_board ;;
  verify_board) stage_verify; stage_board ;;
  *) echo "未知 stage: $1（可用: all data train export verify board verify_board）"; exit 1 ;;
esac

echo ""
echo "复现完成。诚实边界：46.3dB 信道损耗 / BER / 闭环为「模型驱动」，"
echo "非真实体表传输测量；真实站得住的是 BCIC2a 真实脑电驱动的软件信号链在 RV1106 跑通。"
