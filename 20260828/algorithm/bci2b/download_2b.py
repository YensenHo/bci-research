#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
download_2b.py — 下载 BCI Competition IV Dataset 2b（3 电极二分类运动想象）

数据集规格
----------
  - 3 个双极 EEG 通道（C3, Cz, C4）+ 3 个 EOG 通道（本预处理只用前 3 个 EEG 通道）
  - 250 Hz，二分类（左手=1 / 右手=2），9 名被试
  - 每名被试 5 个 session：B{s}T.mat 内含 session 1-3（带标签），
    B{s}E.mat 内含 session 4-5（官方无标签，仅用于测试）
  - 文件命名：B01T.mat ~ B09T.mat、B01E.mat ~ B09E.mat（共 18 个文件）

下载源（与 2a 的 download_bcic2a.py 同链路，逐级回退）：
  1. Graz 原始源 —— https://lampx.tugraz.at/~bci/database/004-2014/
  2. BNCI Horizon 2020 镜像 —— https://bnci-horizon-2020.eu/database/data-sets/004-2014/
     （该镜像 302 重定向到 Graz，两者等效）

【实测踩坑记录 2026-08-31】本机 Python urllib 访问该服务器稳定 ECONNREFUSED
（openmmlab 3.8 与系统 python3 均复现），而 curl 100% 成功——怀疑为服务器侧
socket/TLS 指纹过滤或链路中间设备所致。因此本脚本用 **curl 子进程**下载
（/usr/bin/curl，env -i 隔离下可用），这是本机验证过的唯一可靠路径。

用法：
  env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin \\
      /opt/miniconda3/envs/openmmlab/bin/python download_2b.py
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"
RAW_DIR = DATA_DIR / "raw"

SOURCES = [
    "https://lampx.tugraz.at/~bci/database/004-2014",
    "https://bnci-horizon-2020.eu/database/data-sets/004-2014",
]

CURL = "/usr/bin/curl"


def _download_one(url: str, dest: Path, retries: int = 5) -> bool:
    """单文件下载（curl 子进程），带重试。成功返回 True。"""
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  · {dest.name} 已存在，跳过")
        return True
    for attempt in range(1, retries + 1):
        tmp = dest.with_suffix(dest.suffix + ".part")
        # -f 失败返回非零码；-sS 静默但显示错误；--retry 由外层循环控制
        cmd = [CURL, "-f", "-sS", "--connect-timeout", "30",
               "--max-time", "1800", "-o", str(tmp), url]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=1900)
            if r.returncode == 0 and tmp.exists() and tmp.stat().st_size > 0:
                tmp.replace(dest)
                print(f"  ✓ {dest.name} 下载完成（{dest.stat().st_size // (1024*1024)} MB）")
                return True
            print(f"  ✗ 第 {attempt}/{retries} 次下载失败（curl rc={r.returncode} "
                  f"{r.stderr.strip()[:120]}）")
        except Exception as exc:  # noqa: BLE001
            print(f"  ✗ 第 {attempt}/{retries} 次下载异常（{exc}）")
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        if attempt == retries:
            return False
        time.sleep(3 * attempt)
    return False


def main() -> int:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    targets = []
    for subj in range(1, 10):
        for sess in ("T", "E"):
            targets.append(f"B{subj:02d}{sess}.mat")
    print(f"缓存目录：{RAW_DIR}")
    print(f"目标文件：{len(targets)} 个（{targets[0]} ... {targets[-1]}）")

    status: dict[str, bool] = {}
    for base in SOURCES:
        print(f"\n下载源：{base}")
        all_ok = True
        for fname in targets:
            ok = _download_one(f"{base}/{fname}", RAW_DIR / fname)
            status[fname] = ok
            all_ok = all_ok and ok
        if all_ok:
            break
        print("  · 该源存在失败项，尝试下一镜像...")

    missing = [f for f in targets if not (RAW_DIR / f).exists()]
    total_mb = sum(
        (RAW_DIR / f).stat().st_size for f in targets if (RAW_DIR / f).exists()
    ) // (1024 * 1024)
    if missing:
        print(f"\n⚠ {len(missing)} 个文件缺失：{missing}")
        return 1
    print(f"\n✓ 全部 {len(targets)} 个文件就绪，共约 {total_mb} MB。")
    print("  下一步：python3 preprocess_2b.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
