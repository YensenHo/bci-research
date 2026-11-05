#!/bin/bash
# redownload_failed.sh — 补下 3 个失败文件（全量 GET，不用 -C - 绕开 CDN 416）
set -u
RAW=/Users/heyixuan/BCI/20260828/algorithm/chbmit/raw
BASE="https://hf-mirror.com/datasets/Yuchang-Zhao/CHB-MIT/resolve/main/rawData"
MIN_SIZE=40000000
LOG=/Users/heyixuan/BCI/20260828/algorithm/chbmit/redownload.log
: > "$LOG"
redl() {  # pat file
  local pat="$1" f="$2" url out size attempt code
  url="$BASE/$pat/$f"
  out="$RAW/$pat/$f"
  rm -f "$out"
  for attempt in 1 2 3 4 5 6 7 8; do
    code=$(curl -sSL --max-time 900 --retry 3 --retry-all-errors -o "$out" -w "%{http_code}" "$url" 2>/dev/null)
    size=$(stat -f%z "$out" 2>/dev/null || echo 0)
    if [ "$size" -ge $MIN_SIZE ]; then
      echo "OK  $pat/$f ($size B, attempt $attempt)" >> "$LOG"
      return 0
    fi
    echo "retry $pat/$f code=$code size=$size (attempt $attempt)" >> "$LOG"
    rm -f "$out"
    sleep 3
  done
  echo "FAIL $pat/$f" >> "$LOG"
  return 1
}
redl chb01 chb01_20.edf
redl chb01 chb01_26.edf
redl chb02 chb02_16.edf
echo "=== 补下结束 ===" >> "$LOG"
cat "$LOG"
