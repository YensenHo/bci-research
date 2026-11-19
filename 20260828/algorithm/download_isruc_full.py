#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""download_isruc_full.py — 从公开 S3 下载 ISRUC-Sleep Subgroup-I 12 人完整 .edf + events.tsv。

内嵌 URL（annex key 已解析，无需 /tmp/isruc_urls.txt），跨平台（Windows/macOS/Linux）。
下载后用 annex key 里的 size 校验完整性（之前 Mac 上有 4 个文件因断点续传截断，教训在此）。
"""
import concurrent.futures
import os
import sys
import urllib.request

# (subject, eeg_s3_url, expected_edf_size_bytes)
FILES = [
    ("sub-I001", "https://nemar.s3.us-east-2.amazonaws.com/nm000111/objects/SHA256E-s91778304--900755de373ff9045c9ad758236c6f304dd3bb8f39dbbade55b56b6be454968f.edf", 91778304),
    ("sub-I002", "https://nemar.s3.us-east-2.amazonaws.com/nm000111/objects/SHA256E-s58458112--114113de6519253f3099c356d19ca674f3c1be19af14bda1a9e16759f99fa35c.edf", 58458112),
    ("sub-I003", "https://nemar.s3.us-east-2.amazonaws.com/nm000111/objects/SHA256E-s154185620--92c2dc2efde9a811b4b5e458bc20f184493fc8af58077f509e1220407c1ddef7.edf", 154185620),
    ("sub-I004", "https://nemar.s3.us-east-2.amazonaws.com/nm000111/objects/SHA256E-s157455620--bc0c2b2ae719464a8cafe7cf4ad70639d14951a69dd310b63c55592bb99bbfce.edf", 157455620),
    ("sub-I005", "https://nemar.s3.us-east-2.amazonaws.com/nm000111/objects/SHA256E-s143067620--8c666ae94531f0dfad5301f09a3be87a13af813209166a703240375ec6803b2d.edf", 143067620),
    ("sub-I006", "https://nemar.s3.us-east-2.amazonaws.com/nm000111/objects/SHA256E-s146664620--4540ab0c2329e7b0742470c80ee97b8c39a566fab49b0481f193f7fca3913cd6.edf", 146664620),
    ("sub-I007", "https://nemar.s3.us-east-2.amazonaws.com/nm000111/objects/SHA256E-s152550620--e4fc74997855b07be59e5512bc8fbb8c3d73ed6d80e9ad46a0c612c46d216065.edf", 152550620),
    ("sub-I008", "https://nemar.s3.us-east-2.amazonaws.com/nm000111/objects/SHA256E-s120689632--7d4744746d62b7ed4cb99e152a8256e5299b1eadaeda7bfe19238945650dcb25.edf", 120689632),
    ("sub-I009", "https://nemar.s3.us-east-2.amazonaws.com/nm000111/objects/SHA256E-s158436620--7ccad29d0d095ad8b981fe8270920e4e3b870913b449b7d0f367125be80b7211.edf", 158436620),
    ("sub-I010", "https://nemar.s3.us-east-2.amazonaws.com/nm000111/objects/SHA256E-s137672120--318e58a7cdf06667686274d7c0e7454b87790ef534cfe2b3363f048c1e71577d.edf", 137672120),
    ("sub-I011", "https://nemar.s3.us-east-2.amazonaws.com/nm000111/objects/SHA256E-s198907900--ea67c90484c6411c0d265bf82571c926d0ddbf945a50ce4770bd64c045a045ff.edf", 198907900),
    ("sub-I012", "https://nemar.s3.us-east-2.amazonaws.com/nm000111/objects/SHA256E-s138980120--1e42dbbc8ba74d3801eb43ba1900b7e7b74fc936e058f5c0fe632863be768271.edf", 138980120),
]

DEST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "isruc_raw")
os.makedirs(DEST, exist_ok=True)


def download(url, out, timeout=180):
    """Stream a URL to disk and resume a partial file when the server supports Range."""
    existing = os.path.getsize(out) if os.path.exists(out) else 0
    headers = {"Range": f"bytes={existing}-"} if existing else {}
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as response:
        append = existing > 0 and response.status == 206
        mode = "ab" if append else "wb"
        with open(out, mode) as f:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)


def fetch_edf(sub, url, size):
    out = os.path.join(DEST, f"{sub}_eeg.edf")
    if os.path.exists(out) and os.path.getsize(out) == size:
        return f"{sub}: 已完整 ({size})"
    try:
        download(url, out, timeout=300)
        got = os.path.getsize(out) if os.path.exists(out) else 0
        if got != size:
            raise IOError(f"下载字节数不符：{got}/{size}；可重跑脚本续传")
        return f"{sub}: OK ({got} bytes)"
    except Exception as e:
        return f"{sub}: 失败：{e}"


def fetch_events(sub):
    url = f"https://raw.githubusercontent.com/nemarDatasets/nm000111/main/{sub}/eeg/{sub}_task-sleep_events.tsv"
    out = os.path.join(DEST, f"{sub}_events.tsv")
    if os.path.exists(out) and os.path.getsize(out) > 1000:
        return f"{sub}_events: 已存在"
    try:
        download(url, out, timeout=60)
        got = os.path.getsize(out) if os.path.exists(out) else 0
        if got < 1000:
            raise IOError(f"事件文件异常短：{got} bytes")
        return f"{sub}_events: OK ({got} bytes)"
    except Exception as e:
        return f"{sub}_events: 失败：{e}"


def main():
    failures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:
        futs = [ex.submit(fetch_edf, s, u, z) for s, u, z in FILES] + \
               [ex.submit(fetch_events, s) for s, _, _ in FILES]
        for f in concurrent.futures.as_completed(futs):
            try:
                result = f.result()
                print(result, flush=True)
                if "失败" in result:
                    failures.append(result)
            except Exception as e:
                print("ERR", e, flush=True)
                failures.append(str(e))
    print("ALL DONE — 全部文件校验通过" if not failures else
          f"下载未完成：{len(failures)} 个文件失败；可重跑续传", flush=True)
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
