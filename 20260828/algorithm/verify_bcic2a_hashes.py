#!/usr/bin/env python3
"""Verify local raw .mat files match the git-blob hashes recorded by the mirror audit."""
import hashlib
import json
from pathlib import Path

RAW = Path("/Users/heyixuan/BCI/luckfox-deploy/data/raw")
JSON = Path("/Users/heyixuan/BCI/20260828/algorithm/bcic2a镜像文件哈希_20261118.json")


def git_blob_sha1(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def main() -> None:
    records = json.loads(JSON.read_text(encoding="utf8"))
    ok = bad = 0
    for rec in records:
        p = RAW / rec["file"]
        if not p.exists():
            print(f"MISSING  {rec['file']}")
            bad += 1
            continue
        actual = git_blob_sha1(p)
        expected = rec["git_blob"]
        if actual == expected:
            ok += 1
        else:
            bad += 1
            print(f"MISMATCH {rec['file']}\n  expect {expected}\n  actual {actual}")
    print(f"\nmatched={ok}  mismatched/missing={bad}  total={len(records)}")


if __name__ == "__main__":
    main()
