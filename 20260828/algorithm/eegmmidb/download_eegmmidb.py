#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
download_eegmmidb.py — PhysioNet EEGMMIDB 并行下载 + SHA256 校验

- 每被试 14 个 EDF（S{NNN}R{01..14}.edf）+ 14 个 .edf.event 注释文件
- 校验：SHA256SUMS.txt（官方，3058 条）
- 32 线程 curl 并行（PhysioNet 单连接 ~8KB/s 被限速，靠并行凑吞吐）
- 断点续传：.part 中途文件 + 完成后校验

用法：
  env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin \
      /opt/miniconda3/envs/openmmlab/bin/python download_eegmmidb.py --subjects 1 2 ... 30
  （--all = S001..S109；--workers N 默认 32）
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import subprocess
import sys
import time
from pathlib import Path

BASE = "https://physionet.org/files/eegmmidb/1.0.0"
HERE = Path(__file__).resolve().parent
RAW = HERE / "data" / "raw"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def curl_download(url: str, dst: Path) -> bool:
    """下载单个文件（原子写 .part → rename）。返回是否成功。"""
    if dst.exists() and dst.stat().st_size > 0:
        return True
    part = dst.with_suffix(dst.suffix + ".part")
    for attempt in range(4):
        r = subprocess.run(
            ["curl", "-sS", "--retry", "2", "--max-time", "900",
             "-o", str(part), url],
            capture_output=True,
        )
        if r.returncode == 0 and part.exists() and part.stat().st_size > 0:
            part.rename(dst)
            return True
        # 失败重试
        part.unlink(missing_ok=True)
        time.sleep(2 + attempt * 3)
    print(f"    ✗ 多次失败: {url}")
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", type=int, nargs="+", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()

    if args.all:
        subjects = list(range(1, 110))
    elif args.subjects:
        subjects = sorted(set(args.subjects))
    else:
        ap.error("必须指定 --subjects 或 --all")

    r = subprocess.run(["curl", "-sS", f"{BASE}/SHA256SUMS.txt"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print("✗ 拉取 SHA256SUMS.txt 失败")
        return 1
    sums = {}
    for line in r.stdout.strip().splitlines():
        parts = line.split(None, 1)
        if len(parts) == 2:
            sums[parts[1].strip()] = parts[0].strip()
    print(f"✓ SHA256SUMS.txt: {len(sums)} 条记录, 被试数={len(subjects)}")

    tasks = []
    for s in subjects:
        subj = f"S{s:03d}"
        for rr in range(1, 15):
            fname = f"{subj}R{rr:02d}"
            for ext in (".edf", ".edf.event"):
                rel = f"{subj}/{fname}{ext}"
                if rel in sums:
                    tasks.append((rel, RAW / rel))

    t0 = time.time()
    n_ok = n_fail = 0
    with cf.ThreadPoolExecutor(args.workers) as ex:
        futs = {ex.submit(curl_download, f"{BASE}/{rel}", dst): rel
                for rel, dst in tasks}
        done = 0
        for fut in cf.as_completed(futs):
            rel = futs[fut]
            done += 1
            if fut.result():
                n_ok += 1
            else:
                n_fail += 1
                print(f"    FAIL {rel}")
            if done % 60 == 0 or done == len(futs):
                elapsed = time.time() - t0
                print(f"  进度 {done}/{len(futs)} OK={n_ok} FAIL={n_fail} "
                      f"elapsed={elapsed/60:.1f}min")

    total_bytes = sum(p.stat().st_size for _, p in tasks if p.exists())
    print(f"\n下载完成: OK={n_ok} FAIL={n_fail} 总字节={total_bytes/1e9:.3f} GB "
          f"耗时={(time.time()-t0)/60:.1f}min")

    print("\n校验 SHA256 ...")
    bad = []
    for rel, p in tasks:
        if not p.exists():
            bad.append(f"{rel}: 缺失")
            continue
        if sha256_of(p) != sums.get(rel, ""):
            bad.append(f"{rel}: sha256 不一致")
    if bad:
        print(f"✗ 校验失败 {len(bad)} 个文件:")
        for b in bad[:20]:
            print("   ", b)
        return 2
    print("✓ 全部文件 SHA256 与 PhysioNet 官方一致")
    return 0


if __name__ == "__main__":
    sys.exit(main())
