#!/bin/bash
# Download S1-S10: verify each against its own server content-length (HEAD)
cd /Users/heyixuan/BCI/20260828/algorithm/ssvep/raw || exit 1
BASE="https://bci.med.tsinghua.edu.cn/upload/yijun"
for i in 1 2 3 4 5 6 7 8 9 10; do
  f="S${i}.mat.7z"
  want=$(curl -sI --max-time 15 "$BASE/S${i}.mat.7z" | grep -i content-length | tr -d '\r' | awk '{print $2}')
  if [ -z "$want" ]; then echo "NO HEAD for $f"; exit 1; fi
  while true; do
    if [ -f "$f" ]; then have=$(stat -f%z "$f"); else have=0; fi
    if [ "$have" = "$want" ]; then echo "OK $f ($have bytes)"; break; fi
    echo "GET $f (have $have, want $want)"
    curl -sS --retry 5 -o "$f" "$BASE/S${i}.mat.7z"
    have=$(stat -f%z "$f")
    if [ "$have" != "$want" ]; then echo "WARN $f got $have, retrying"; sleep 2; fi
  done
done
echo "=== ALL DONE ==="
ls -la *.7z
