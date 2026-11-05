#!/bin/bash
# download_chbmit_hf.sh — 从 hf-mirror (Yuchang-Zhao/CHB-MIT 原始镜像) 下载 CHB-MIT 子集
# 范围：chb01-04 发作文件+等量对照文件（训练），chb05 全部 39 个文件（跨患者评估）
# 断点续传 + 重试 + 大小校验（EDF 全长 ~42.4MB）
set -u
RAW=/Users/heyixuan/BCI/20260828/algorithm/chbmit/raw
BASE="https://hf-mirror.com/datasets/Yuchang-Zhao/CHB-MIT/resolve/main/rawData"
MIN_SIZE=40000000
LOG=/Users/heyixuan/BCI/20260828/algorithm/chbmit/download_hf.log

download_one() {  # $1=chbNN, $2=file.edf
  local pat="$1" f="$2" url out code size attempt
  url="$BASE/$pat/$f"
  out="$RAW/$pat/$f"
  [ -f "$out" ] && [ "$(stat -f%z "$out")" -ge $MIN_SIZE ] && { echo "skip $pat/$f (already complete)"; return 0; }
  for attempt in 1 2 3 4 5; do
    code=$(curl -sSL -C - --max-time 600 -o "$out" -w "%{http_code}" "$url" 2>/dev/null)
    size=$(stat -f%z "$out" 2>/dev/null || echo 0)
    if [ "$size" -ge $MIN_SIZE ]; then
      echo "OK  $pat/$f ($size B, attempt $attempt)" >> "$LOG"
      return 0
    fi
    echo "retry $pat/$f code=$code size=$size (attempt $attempt)" >> "$LOG"
    # 416 = CDN 不支持 Range：删掉半截文件，下次全量 GET
    if [ "$code" = "416" ]; then rm -f "$out"; fi
    sleep 3
  done
  echo "FAIL $pat/$f" >> "$LOG"
  return 1
}

: > "$LOG"
# --- 训练集 chb01-04：发作文件 + 等量对照（固定 seed 随机选取，reproducible）---
# 发作文件（来自 summary 解析）
download_one chb01 chb01_03.edf; download_one chb01 chb01_04.edf; download_one chb01 chb01_15.edf
download_one chb01 chb01_16.edf; download_one chb01 chb01_18.edf; download_one chb01 chb01_21.edf
download_one chb01 chb01_26.edf
download_one chb02 chb02_16.edf; download_one chb02 "chb02_16+.edf"; download_one chb02 chb02_19.edf
download_one chb03 chb03_01.edf; download_one chb03 chb03_02.edf; download_one chb03 chb03_03.edf
download_one chb03 chb03_04.edf; download_one chb03 chb03_34.edf; download_one chb03 chb03_35.edf
download_one chb03 chb03_36.edf
download_one chb04 chb04_05.edf; download_one chb04 chb04_08.edf; download_one chb04 chb04_28.edf
# 对照文件（与发作文件等量，固定 seed 的确定性选择）
download_one chb01 chb01_07.edf; download_one chb01 chb01_10.edf; download_one chb01 chb01_12.edf
download_one chb01 chb01_14.edf; download_one chb01 chb01_20.edf; download_one chb01 chb01_25.edf
download_one chb01 chb01_30.edf
download_one chb02 chb02_03.edf; download_one chb02 chb02_10.edf; download_one chb02 chb02_22.edf
download_one chb03 chb03_06.edf; download_one chb03 chb03_09.edf; download_one chb03 chb03_11.edf
download_one chb03 chb03_15.edf; download_one chb03 chb03_18.edf; download_one chb03 chb03_22.edf
download_one chb03 chb03_28.edf
download_one chb04 chb04_02.edf; download_one chb04 chb04_12.edf; download_one chb04 chb04_25.edf
# --- 评估集 chb05：全部 39 个文件 ---
for M in $(seq -w 1 39); do download_one chb05 "chb05_${M}.edf"; done

echo "=== 下载结束 ==="
echo "已下载 EDF 总数: $(find "$RAW" -name '*.edf' -size +40M | wc -l | tr -d ' ')"
echo "失败列表: $(grep '^FAIL' "$LOG" | tr '\n' ' ')"
du -sh "$RAW"
