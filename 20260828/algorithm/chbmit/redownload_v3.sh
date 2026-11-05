#!/bin/bash
# redownload_v3.sh — 按 PhysioNet 精确字节数校验补下 3 个文件
# chb01_20.edf=31365632 chb01_26.edf=27385344 chb02_16.edf=11299328（PhysioNet HEAD 确认，均 <40MB 但完整）
set -u
RAW=/Users/heyixuan/BCI/20260828/algorithm/chbmit/raw
HF="https://hf-mirror.com/datasets/Yuchang-Zhao/CHB-MIT/resolve/main/rawData"
PN="https://physionet.org/files/chbmit/1.0.0"
LOG=/Users/heyixuan/BCI/20260828/algorithm/chbmit/redownload_v3.log
: > "$LOG"
redl() {  # pat file expect_size
  local pat="$1" f="$2" want="$3" out url size attempt src
  out="$RAW/$pat/$f"
  rm -f "$out"
  for src in hf physionet; do
    [ "$src" = hf ] && url="$HF/$pat/$f" || url="$PN/$pat/$f"
    for attempt in 1 2 3; do
      rm -f "$out"
      curl -sSL --max-time 300 --speed-limit 20000 --speed-time 20 \
           --retry 2 --retry-all-errors -o "$out" -w "%{http_code}" "$url" > /dev/null 2>&1
      size=$(stat -f%z "$out" 2>/dev/null || echo 0)
      if [ "$size" = "$want" ]; then
        echo "OK  $pat/$f ($size B, src=$src attempt=$attempt)" >> "$LOG"
        return 0
      fi
      echo "retry $pat/$f src=$src attempt=$attempt got=$size want=$want" >> "$LOG"
      sleep 2
    done
  done
  echo "FAIL $pat/$f (want $want)" >> "$LOG"
  return 1
}
redl chb01 chb01_20.edf 31365632
redl chb01 chb01_26.edf 27385344
redl chb02 chb02_16.edf 11299328
echo "=== 补下结束 ===" >> "$LOG"
cat "$LOG"
