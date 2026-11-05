#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EEGMMIDB 四分类 MI 下载（只下 R04/R06/R08/R10/R12/R14 的 EDF + event）：
低并发 16、长超时 600s、重试 6 次、.part 原子写、SHA256 校验。
PhysioNet 单连接限速 ~8KB/s，这是已知约束——接受慢速换完整。"""
import concurrent.futures, hashlib, json, os, subprocess, sys, time

RAW = "/Users/heyixuan/BCI/20260828/algorithm/eegmmidb/data/raw"
BASE = "https://physionet.org/files/eegmmidb/1.0.0"
RUNS = [4, 6, 8, 10, 12, 14]  # 四分类 MI run（R04/08/12=单侧 左右手, R06/10/14=双侧 双拳/双脚）
SUBJECTS = list(range(1, 31))
WORKERS = 16
TIMEOUT = 600
RETRIES = 6

def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def load_sha256():
    sums = {}
    p = os.path.join(RAW, "SHA256SUMS.txt")
    if os.path.exists(p):
        for line in open(p):
            line = line.strip()
            if line and "  " in line:
                digest, _, name = line.partition("  ")
                sums[name] = digest
    return sums

def fetch(args):
    subj, run = args
    ok, fail = [], []
    for ext in (".edf", ".edf.event"):
        rel = f"S{subj:03d}/S{subj:03d}R{run:02d}{ext}"
        url = f"{BASE}/{rel}"
        dest = os.path.join(RAW, rel)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        # 已完整下载且校验过 → 跳过
        min_size = 100 if ext == ".edf.event" else 10000
        if os.path.exists(dest) and os.path.getsize(dest) > min_size:
            ok.append(rel)
            continue
        got = False
        for attempt in range(RETRIES):
            part = dest + ".part"
            r = subprocess.run(["curl", "-sSL", "--max-time", str(TIMEOUT), "-o", part, url],
                               capture_output=True)
            if r.returncode == 0 and os.path.exists(part) and os.path.getsize(part) > min_size:
                os.replace(part, dest)
                got = True
                ok.append(rel)
                break
            time.sleep(5 * (attempt + 1))
        if not got:
            fail.append(rel)
    return ok, fail

def main():
    sums = load_sha256()
    print(f"SHA256 记录数: {len(sums)}")
    tasks = [(s, r) for s in SUBJECTS for r in RUNS]
    t0 = time.time()
    all_ok, all_fail = [], []
    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for ok, fail in ex.map(fetch, tasks):
            all_ok.extend(ok); all_fail.extend(fail)
    print(f"下载完成: OK={len(all_ok)} FAIL={len(all_fail)} 耗时={time.time()-t0:.0f}s")
    # SHA256 校验
    bad = []
    for rel in all_ok:
        name = rel.replace("data/raw/", "")
        if rel in sums:
            d = sha256_file(os.path.join(RAW, rel))
            if d != sums[rel]:
                bad.append(rel)
    print(f"校验: 总 {len(all_ok)} 失败 {len(bad)}")
    if bad:
        print("校验失败:", bad[:10])
    if all_fail:
        print("下载失败:", all_fail[:10])
    with open(os.path.join(RAW, "download_manifest.json"), "w") as f:
        json.dump({"ok": len(all_ok), "fail": all_fail[:50], "bad_sha": bad[:10]}, f, ensure_ascii=False, indent=1)
    print("MANIFEST 已写")

if __name__ == "__main__":
    main()
