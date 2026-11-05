#!/usr/bin/env bash
# ============================================================
# demo.sh — RV1106 板载部署验证 + 可演示 Demo 一键脚本
#
# 演示链路：端侧解码（板载 EEGNet）→ HBC 回传（板载信道仿真）
#          → 上位机可视化（demo_visualizer.py）
#
# 用法：
#   bash demo.sh check     板子在线检查 + 基线验证（eegnet_infer / hbc_realtime）
#   bash demo.sh live      完整实时链路（板载流式解码 → 上位机实时可视化）
#   bash demo.sh offline   离线回放（用本目录 mock_board_output.log，无需板子）
#   bash demo.sh mock      纯合成数据（无板子，仅演示可视化本身）
#   bash demo.sh           默认 = check
#
# 依赖：
#   - adb：~/luckfox/platform-tools/adb（可 export ADB 覆盖）
#   - Python：/opt/miniconda3/envs/openmmlab/bin/python（env -i 隔离运行）
# ============================================================
set -euo pipefail

# ---- 可覆盖配置 ----
ADB="${ADB:-$HOME/luckfox/platform-tools/adb}"
PY="/opt/miniconda3/envs/openmmlab/bin/python"
BOARD_SERIAL="${BOARD_SERIAL:-}"                     # 空 = 取当前唯一 device
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BOARD_BIN="/root/fusion_test.bin"                    # 100 试次小数据集（演示快）
BOARD_FULL="/root/fusion_test_full.bin"              # 1296 试次全量（统计基线）

# ---- 颜色 ----
G='\033[1;32m'; Y='\033[1;33m'; R='\033[1;31m'; C='\033[1;36m'; N='\033[0m'

info()  { echo -e "${C}[demo]${N} $*"; }
ok()    { echo -e "${G}[  ✓ ]${N} $*"; }
warn()  { echo -e "${Y}[  ! ]${N} $*"; }
err()   { echo -e "${R}[  ✗ ]${N} $*"; }

# ---- adb 封装（带序列号选择） ----
adb_shell() {
  if [ -n "$BOARD_SERIAL" ]; then
    "$ADB" -s "$BOARD_SERIAL" shell "$@"
  else
    "$ADB" shell "$@"
  fi
}

# ---- Python 隔离运行（env -i，规避 numpy ImportError） ----
run_py() {
  env -i HOME="$HOME" PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin \
    "$PY" "$@"
}

# ---- 检查板子在线 ----
check_device() {
  info "检查板子在线状态 ..."
  local devs
  devs="$("$ADB" devices | grep -v '^List' | grep -c 'device' || true)"
  if [ "$devs" -lt 1 ]; then
    err "未检测到板子（adb devices 无 device）。"
    err "请确认：板子 USB 连 Mac、$ADB 存在、'$ADB devices' 有 5caaa04f9cdb6ba2 device。"
    return 1
  fi
  "$ADB" devices | grep -v '^List'
  ok "板子在线。"
  return 0
}

# ---- 基线验证 ----
run_baseline() {
  info "板载基线：./eegnet_infer（EEGNet 单次前向自检）"
  adb_shell "cd /root && ./eegnet_infer" || { err "eegnet_infer 运行失败"; return 1; }
  echo
  info "板载基线：./hbc_realtime $BOARD_BIN 1（端侧解码 + HBC 回传，100 窗口流式）"
  adb_shell "cd /root && ./hbc_realtime $BOARD_BIN 1" | tail -8
  echo
  ok "基线验证完成。"
}

# ---- 完整实时链路 ----
run_live() {
  info "完整实时链路：板载 hbc_realtime → 上位机 demo_visualizer --live"
  info "（板子持续流式输出逐窗口解码结果，上位机实时可视化）"
  run_py "$DIR/demo_visualizer.py" --live --adb "$ADB" \
    --save "$DIR/demo_visualizer.png"
}

# ---- 离线回放 ----
run_offline() {
  local log="$DIR/mock_board_output.log"
  if [ ! -f "$log" ]; then
    err "找不到离线日志 $log"
    warn "可先执行 'bash demo.sh capture' 抓一段真实板载输出，或改用 'bash demo.sh mock'。"
    return 1
  fi
  info "离线回放：${log}（真实板载抓取输出）"
  run_py "$DIR/demo_visualizer.py" --file "$log" --save "$DIR/demo_visualizer.png"
}

# ---- 抓取真实板载输出到本地日志 ----
run_capture() {
  check_device || return 1
  local log="$DIR/mock_board_output.log"
  info "抓取板载 hbc_realtime 真实输出 → ${log}"
  adb_shell "cd /root && ./hbc_realtime $BOARD_BIN 1" > "$log" 2>&1
  ok "已抓取 $(wc -l < "$log" | tr -d ' ') 行。"
}

# ---- mock ----
run_mock() {
  info "纯合成数据（无板子，仅演示可视化，不冒充真实板载数字）"
  run_py "$DIR/demo_visualizer.py" --mock --windows 40 --save "$DIR/demo_visualizer.png"
}

# ---- 帮助 ----
usage() {
  cat <<'EOF'
demo.sh — RV1106 板载部署验证 + 可演示 Demo

用法：
  bash demo.sh check     板子在线检查 + 基线验证（eegnet_infer / hbc_realtime）
  bash demo.sh live      完整实时链路（板载流式解码 → 上位机实时可视化）
  bash demo.sh offline   离线回放 mock_board_output.log（无需板子）
  bash demo.sh capture   抓取一段真实板载输出到 mock_board_output.log
  bash demo.sh mock      纯合成数据（无板子，仅演示可视化）
  bash demo.sh           默认 = check

环境变量（可选）：
  ADB=<path>              adb 路径（默认 ~/luckfox/platform-tools/adb）
  BOARD_SERIAL=<serial>   板子序列号（默认自动取当前唯一 device）

详请见 Demo使用说明.md
EOF
}

# ---- 主入口 ----
MODE="${1:-check}"
case "$MODE" in
  check)
    check_device && run_baseline
    ;;
  live)
    check_device && run_live
    ;;
  offline)
    run_offline
    ;;
  capture)
    run_capture
    ;;
  mock)
    run_mock
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    err "未知模式：$MODE"
    usage
    exit 1
    ;;
esac
