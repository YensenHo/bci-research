#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""download_isruc.py — 从公开 S3 并行下载 ISRUC-Sleep Subgroup-I 受试者的 .edf + events.tsv。

S3 公开桶（encryption=none），URL = https://nemar.s3.us-east-2.amazonaws.com/nm000111/objects/<annex-key>
标注 events.tsv 是 GitHub 真实数据（raw）。
"""
import concurrent.futures
import os
import re
import subprocess
import sys

DEST = '/Users/heyixuan/BCI/20260828/algorithm/isruc/raw'
N_SUBJECTS = 12  # Subgroup-I 前 12 个
MAX_WORKERS = 8

os.makedirs(DEST, exist_ok=True)


def load_urls():
    urls = []
    with open('/tmp/isruc_urls.txt') as f:
        for line in f:
            sub, url = line.strip().split('\t')
            m = re.match(r'sub-I(\d+)$', sub)
            if m and 1 <= int(m.group(1)) <= N_SUBJECTS:
                urls.append((sub, url))
    return urls


def download_edf(item):
    sub, url = item
    out = f'{DEST}/{sub}_eeg.edf'
    if os.path.exists(out) and os.path.getsize(out) > 50_000_000:
        return f'{sub}: 已存在 ({os.path.getsize(out)} bytes)'
    subprocess.run(['curl', '-s', '-C', '-', '-o', out, url], timeout=3600)
    size = os.path.getsize(out) if os.path.exists(out) else 0
    return f'{sub}: {size} bytes'


def download_events(sub):
    url = f'https://raw.githubusercontent.com/nemarDatasets/nm000111/main/{sub}/eeg/{sub}_task-sleep_events.tsv'
    out = f'{DEST}/{sub}_events.tsv'
    if os.path.exists(out) and os.path.getsize(out) > 1000:
        return f'{sub}_events: 已存在'
    subprocess.run(['curl', '-s', '-o', out, url], timeout=120)
    size = os.path.getsize(out) if os.path.exists(out) else 0
    return f'{sub}_events: {size} bytes'


def main():
    urls = load_urls()
    print(f'准备下载 {len(urls)} 个受试者 .edf + events.tsv 到 {DEST}', flush=True)
    subs = [u[0] for u in urls]
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = [ex.submit(download_edf, u) for u in urls] + \
               [ex.submit(download_events, s) for s in subs]
        for fut in concurrent.futures.as_completed(futs):
            try:
                print(fut.result(), flush=True)
            except Exception as e:
                print(f'ERR: {e}', flush=True)
    print('ALL DOWNLOAD DONE', flush=True)


if __name__ == '__main__':
    sys.exit(main())
