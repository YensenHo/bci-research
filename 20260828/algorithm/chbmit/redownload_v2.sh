#!/bin/bash
# redownload_v2.sh — 补下 3 个文件：优先 hf-mirror(cache-buster) → 退回 PhysioNet
# 带 --speed-limit 快速失败，避免干等
set -u
RAW=/Users/heyixuan/BCI/20260828/algorithm/chbmit/raw
HF="https://hf-mirror.com/datasets/Yuchang-Zhao/CHB-MIT/resolve/main/rawData"
PN="https://physionet.org/files/chbmit/1.0.0"
MIN_SIZE=40000000
LOG=/Users/heyixuan/BCI/20260828/algorithm/chbmit/redownload_v2.log
BUSTER="cb=$(date +%s)"
: > "$LOG"
redl() {  # pat file
  local pat="$1" f="$2" url out size attempt src
  out="$RAW/$pat/$f"
  rm -f "$out"
  for src in "hf-mirror?$BUSTER" "physionet"; do
    case "$src" in
      hf-mirror*) url="$HF/$pat/$f?${BUSTER}" ;;
      *)          url="$PN/$pat/$f" ;;
    esac
    attempt=0
    while [ $attempt -lt 4 ]; do
      attempt=$((attempt+1))
      rm -f "$out"
      curl -sSL --max-time 300 --speed-limit 20000 --speed-time 20 \
           --retry 2 --retry-all-errors -o "$out" -w "%{http_code}" "$url" > /dev/null 2>&1
      size=$(stat -f%z "$out" 2>/dev/null || echo 0)
      if [ "$size" -ge $MIN_SIZE ]; then
        echo "OK  $pat/$f ($size B, src=$src attempt=$attempt)" >> "$LOG"
        return 0
      fi
      echo "retry $pat/$f src=$src attempt=$attempt size=$size" >> "$LOG"
      sleep 2
    done
  done
  echo "FAIL $pat/$f" >> "$LOG"
  return 1
}
redl chb01 chb01_20.edf
redl chb01 chb01_26.edf
redl chb02 chb02_16.edf
echo "=== 补下结束 ===" >> "$LOG"
cat "$LOG"
