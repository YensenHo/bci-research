#!/bin/bash
# download_sleep.sh — 下载 Sleep-EDF Expanded SC 系列 6 例（12 夜 PSG + 12 Hypnogram）
# 约定（MNE sleep_edfx 惯例）：SC4001/SC4002 = subject 0 两夜, SC4011/SC4012 = subject 1, ...
# 6 例 = SC4001..SC4052（subjects 0-5），train=0-3（8 夜）, eval=4-5（4 夜）——跨被试
# 期望字节数来自 PhysioNet 目录列表（https://physionet.org/files/sleep-edfx/1.0.0/sleep-cassette/）
set -u
BASE="https://physionet.org/files/sleep-edfx/1.0.0/sleep-cassette"
OUT="/Users/heyixuan/BCI/20260828/algorithm/sleep/raw"
mkdir -p "$OUT"

# 文件|期望字节数（来自服务器列表页）
FILES="
SC4001E0-PSG.edf|48338048
SC4001EC-Hypnogram.edf|4620
SC4002E0-PSG.edf|51621248
SC4002EC-Hypnogram.edf|4580
SC4011E0-PSG.edf|51110528
SC4011EH-Hypnogram.edf|3896
SC4012E0-PSG.edf|51986048
SC4012EC-Hypnogram.edf|5082
SC4021E0-PSG.edf|51147008
SC4021EH-Hypnogram.edf|4804
SC4022E0-PSG.edf|50271488
SC4022EJ-Hypnogram.edf|5292
SC4031E0-PSG.edf|51438848
SC4031EC-Hypnogram.edf|3700
SC4032E0-PSG.edf|49833728
SC4032EP-Hypnogram.edf|3800
SC4041E0-PSG.edf|46878848
SC4041EC-Hypnogram.edf|4830
SC4042E0-PSG.edf|50928128
SC4042EC-Hypnogram.edf|5276
SC4051E0-PSG.edf|49651328
SC4051EC-Hypnogram.edf|3976
SC4052E0-PSG.edf|51183488
SC4052EC-Hypnogram.edf|4232
"

FAIL=0
for line in $FILES; do
  f="${line%%|*}"; want="${line##*|}"
  if [ -f "$OUT/$f" ]; then
    got=$(stat -f%z "$OUT/$f")
    if [ "$got" = "$want" ]; then echo "SKIP(已存在且字节数一致) $f"; continue; fi
    echo "RE-DL $f (本地 $got ≠ 期望 $want)"; rm -f "$OUT/$f"
  fi
  curl -sS --fail --retry 3 --max-time 600 -o "$OUT/$f" "$BASE/$f"
  rc=$?
  got=$(stat -f%z "$OUT/$f" 2>/dev/null || echo 0)
  if [ $rc -ne 0 ] || [ "$got" != "$want" ]; then
    echo "FAIL $f (rc=$rc got=$got want=$want)"; FAIL=1
  else
    echo "OK   $f ($got bytes)"
  fi
done
echo "======================"
if [ $FAIL -eq 0 ]; then echo "ALL OK"; else echo "HAS FAILURES"; fi
