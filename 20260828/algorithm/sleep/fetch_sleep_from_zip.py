#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fetch_sleep_from_zip.py — 从 HuggingFace 镜像 zip 按 Range 提取 Sleep-EDF SC 系列 6 例

背景
----
  PhysioNet 直连被限速 ~7 KB/s（604MB 需 26 小时），改用 HF 镜像
  ronaldvandenbroek/physionet-sleep-data 的 physionet-sleep-data.zip（1.04GB，
  含 SC4xxx 系列第 1 夜 E0 的 30 例 PSG+Hypnogram）。
  该 zip 内条目未压缩字节数与 PhysioNet 服务器列表页完全一致
  （如 SC4001E0-PSG.edf usize=48338048 == physionet 48338048），
  因此按 Range 只下载所需条目的 deflate 流并解压，字节数与 PhysioNet 校验一致。

  只取 6 例（跨被试）：SC4001/SC4011/SC4021/SC4031 → train，SC4041/SC4051 → eval
  每例 = 1 夜（E0）。PSG 总下载量 ≈ 210MB（压缩流），远小于整包 1.04GB。

用法
----
  env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin \\
      /opt/miniconda3/envs/openmmlab/bin/python fetch_sleep_from_zip.py
"""
from __future__ import annotations

import json
import struct
import sys
import urllib.request
import zlib
from pathlib import Path

import subprocess

BASE = Path(__file__).resolve().parent
RAW = BASE / "raw"
ZIP_URL = ("https://hf-mirror.com/datasets/ronaldvandenbroek/"
           "physionet-sleep-data/resolve/main/physionet-sleep-data.zip")

# 需要的记录：名称前缀 → 期望字节数（PhysioNet 服务器列表页）
NEED = {
    "SC4001E0-PSG.edf": 48338048, "SC4001EC-Hypnogram.edf": 4620,
    "SC4011E0-PSG.edf": 51110528, "SC4011EH-Hypnogram.edf": 3896,
    "SC4021E0-PSG.edf": 51147008, "SC4021EH-Hypnogram.edf": 4804,
    "SC4031E0-PSG.edf": 51438848, "SC4031EC-Hypnogram.edf": 3700,
    "SC4041E0-PSG.edf": 46878848, "SC4041EC-Hypnogram.edf": 4830,
    "SC4051E0-PSG.edf": 49651328, "SC4051EC-Hypnogram.edf": 3976,
}


def curl_range(url: str, start: int, end: int, timeout: int = 180) -> bytes:
    """Range 下载 [start, end]（含），返回字节。用 curl（urllib TLS 握手在此环境超时）。"""
    out = subprocess.run(
        ["curl", "-sL", "--max-time", str(timeout),
         "-r", f"{start}-{end}", url],
        capture_output=True, check=True)
    return out.stdout


def main() -> int:
    RAW.mkdir(exist_ok=True)
    # 1) 总大小
    out = subprocess.run(["curl", "-sIL", "--max-time", "60", ZIP_URL],
                         capture_output=True, text=True, check=True)
    total = None
    for line in out.stdout.splitlines():
        if line.lower().startswith("content-length:"):
            total = int(line.split(":", 1)[1].strip())
    assert total, "无法获取 zip 大小"
    print(f"zip 总大小: {total:,} bytes")

    # 2) 尾部 256KB → EOCD + 中央目录
    tail = curl_range(ZIP_URL, total - 262144, total - 1)
    eocd = tail.rfind(b"PK\x05\x06")
    _, _, _, n_total, cd_size, cd_off = struct.unpack("<HHHHII", tail[eocd + 4:eocd + 20])
    assert n_total == 60 and cd_off >= total - len(tail)
    cd = tail[cd_off - (total - len(tail)): cd_off - (total - len(tail)) + cd_size]
    entries = []
    pos = 0
    while pos < len(cd) - 46:
        if cd[pos:pos + 4] != b"PK\x01\x02":
            pos += 1
            continue
        method = struct.unpack("<H", cd[pos + 10:pos + 12])[0]
        csize, usize = struct.unpack("<II", cd[pos + 20:pos + 28])
        fn_len, ex_len, cm_len = struct.unpack("<HHH", cd[pos + 28:pos + 34])
        lho = struct.unpack("<I", cd[pos + 42:pos + 46])[0]
        name = cd[pos + 46:pos + 46 + fn_len].decode("utf-8", "replace").split("/")[-1]
        entries.append(dict(name=name, method=method, csize=csize,
                            usize=usize, lho=lho))
        pos += 46 + fn_len + ex_len + cm_len
    print(f"中央目录条目: {len(entries)}")

    # 3) 逐文件提取
    ok = fail = 0
    for name, want in NEED.items():
        e = next((x for x in entries if x["name"] == name), None)
        assert e, f"zip 中找不到 {name}"
        assert e["usize"] == want, f"{name}: zip usize={e['usize']} ≠ PhysioNet {want}"
        dest = RAW / name
        if dest.exists() and dest.stat().st_size == want:
            print(f"SKIP {name}（已存在且字节一致）")
            ok += 1
            continue
        # 本地头（30 + fn + ex），读 fn/ex 长度
        head = curl_range(ZIP_URL, e["lho"], e["lho"] + 60)
        assert head[:4] == b"PK\x03\x04", f"{name}: 本地头签名错误"
        fn_l, ex_l = struct.unpack("<HH", head[26:30])
        data_start = e["lho"] + 30 + fn_l + ex_l
        data_end = data_start + e["csize"] - 1
        raw = curl_range(ZIP_URL, data_start, data_end)
        assert len(raw) == e["csize"], f"{name}: 压缩流长度 {len(raw)} ≠ {e['csize']}"
        if e["method"] == 8:
            content = zlib.decompress(raw, -15)
        elif e["method"] == 0:
            content = raw
        else:
            raise ValueError(f"{name}: 不支持的压缩方法 {e['method']}")
        if len(content) != want:
            fail += 1
            print(f"FAIL {name}: 解压 {len(content)} ≠ 期望 {want}")
            continue
        dest.write_bytes(content)
        print(f"OK   {name} ({len(content):,} bytes == PhysioNet {want:,})")
        ok += 1

    # 4) 汇总 + 清单
    manifest = {n: dict(bytes=NEED[n], ok=(RAW / n).exists() and (RAW / n).stat().st_size == NEED[n])
                for n in NEED}
    (BASE / "download_manifest.json").write_text(
        json.dumps(dict(source=ZIP_URL, zip_total_bytes=total, files=manifest),
                   ensure_ascii=False, indent=2), encoding="utf-8")
    total_dl = sum(e["csize"] for e in entries if e["name"] in NEED)
    print(f"\n{'✓ 全部下载成功' if fail == 0 and ok == len(NEED) else '✗ 有失败'}: "
          f"{ok}/{len(NEED)} 个文件，压缩流下载量 ≈ {total_dl/1e6:.1f} MB")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
