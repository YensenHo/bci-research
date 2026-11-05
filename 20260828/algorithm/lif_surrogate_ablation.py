#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lif_surrogate_ablation.py — LIF-EEGNet surrogate 训练关键消融驱动

消融设计（顺序寻优，避免全网格爆炸）：
  [A] threshold 扫描（fix tau=2.0, T=8）: {0.5, 1.0, 2.0, 4.0}
  [B] timesteps 扫描（fix best threshold, tau=2.0）: {4, 8, 16, 32}
  [C] tau 扫描（fix best threshold/timesteps）: {1.0, 2.0, 4.0}
  [D] learnable threshold（fix best T/tau）
  [E] 从随机初始化（无热启动）对照（best 配置）
  [F] final：best 配置 + 3 个 seed 求稳定性

每个 run 复用 train_lif_surrogate.py（env -i 隔离 + openmmlab python），
结果追加写入 ablation_lif.json，最后打印汇总表。

用法
----
  env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin \\
      /opt/miniconda3/envs/openmmlab/bin/python lif_surrogate_ablation.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ALGO_DIR = Path(__file__).resolve().parent
PY = "/opt/miniconda3/envs/openmmlab/bin/python"
ENV = {"HOME": str(Path.home()),
       "PATH": "/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin"}

TRAIN = str(ALGO_DIR / "train_lif_surrogate.py")
JSON = ALGO_DIR / "ablation_lif.json"

# 每 run 通用参数（早停快、够稳）
COMMON = ["--epochs", "40", "--patience", "8", "--batch-size", "16",
          "--lr", "0.001", "--seed", "42", "--n-classes", "2",
          "--json-out", str(JSON)]


def run(cfg: dict) -> dict:
    """跑一次训练，返回解析后的 JSON 记录。"""
    args = [PY, TRAIN] + COMMON + [
        "--threshold", str(cfg.get("threshold", 1.0)),
        "--tau", str(cfg.get("tau", 2.0)),
        "--timesteps", str(cfg.get("timesteps", 8)),
        "--alpha", str(cfg.get("alpha", 4.0)),
        "--out", str(ALGO_DIR / f"lif_{cfg['tag']}.pth"),
        "--tag", cfg["tag"],
    ]
    if cfg.get("learnable"):
        args.append("--learnable-threshold")
    if cfg.get("no_warm"):
        args += ["--warm-start", str(ALGO_DIR / "does_not_exist.pth")]
    if cfg.get("seed"):
        # seed 覆盖 COMMON 中的 42
        args[args.index("--seed") + 1] = str(cfg["seed"])
    if cfg.get("epochs"):
        args[args.index("--epochs") + 1] = str(cfg["epochs"])
        args[args.index("--patience") + 1] = str(cfg.get("patience", 8))

    print(f"\n{'='*70}\n▶ RUN {cfg['tag']}  {cfg}\n{'='*70}", flush=True)
    proc = subprocess.run(args, env=ENV, cwd=str(ALGO_DIR),
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True)
    # 只打印关键行，避免刷屏
    tail = proc.stdout.strip().splitlines()[-12:]
    print("\n".join(tail), flush=True)
    # 从 JSON 读取该 run 的记录
    if JSON.exists():
        data = json.loads(JSON.read_text(encoding="utf-8"))
        for rec in data:
            if rec.get("tag") == cfg["tag"]:
                return rec
    return {"tag": cfg["tag"], "error": proc.returncode}


def fmt(r: dict) -> str:
    if "error" in r and "final_clean" not in r:
        return f"{r['tag']:22s}  ERROR"
    lt = "✓" if r.get("learnable_threshold") else " "
    return (f"{r['tag']:22s} thr={r['threshold']:>5} tau={r['tau']:>4} T={r['timesteps']:>3} "
            f"lt={lt} | val={r['best_val_acc']*100:5.2f}% "
            f"clean={r['final_clean']*100:5.2f}% hbc={r['final_hbc']*100:5.2f}% "
            f"(fr1={r['fr_act1']*100:4.1f}% fr2={r['fr_act2']*100:4.1f}%)")


def main() -> int:
    results = []

    # ---- [A] threshold 扫描 ----
    print("\n### [A] threshold 扫描（tau=2.0, T=8）", flush=True)
    for thr in (0.5, 1.0, 2.0, 4.0):
        r = run({"tag": f"A_thr{thr}", "threshold": thr, "tau": 2.0, "timesteps": 8})
        results.append(r)
        print("   →", fmt(r), flush=True)
    best_thr = max(results, key=lambda x: x.get("final_clean", 0))["threshold"]

    # ---- [B] timesteps 扫描 ----
    print(f"\n### [B] timesteps 扫描（thr={best_thr}, tau=2.0）", flush=True)
    b_res = []
    for T in (4, 8, 16, 32):
        r = run({"tag": f"B_T{T}", "threshold": best_thr, "tau": 2.0, "timesteps": T})
        results.append(r); b_res.append(r)
        print("   →", fmt(r), flush=True)
    best_T = max(b_res, key=lambda x: x.get("final_clean", 0))["timesteps"]

    # ---- [C] tau 扫描 ----
    print(f"\n### [C] tau 扫描（thr={best_thr}, T={best_T}）", flush=True)
    c_res = []
    for tau in (1.0, 2.0, 4.0):
        r = run({"tag": f"C_tau{tau}", "threshold": best_thr, "tau": tau, "timesteps": best_T})
        results.append(r); c_res.append(r)
        print("   →", fmt(r), flush=True)
    best_tau = max(c_res, key=lambda x: x.get("final_clean", 0))["tau"]

    # ---- [D] learnable threshold ----
    print(f"\n### [D] learnable threshold（tau={best_tau}, T={best_T}）", flush=True)
    r = run({"tag": "D_learnthr", "threshold": best_thr, "tau": best_tau,
             "timesteps": best_T, "learnable": True})
    results.append(r)
    print("   →", fmt(r), flush=True)

    # ---- [E] 无热启动对照 ----
    print(f"\n### [E] 无热启动（随机初始化, thr={best_thr}, tau={best_tau}, T={best_T}）", flush=True)
    r = run({"tag": "E_scratch", "threshold": best_thr, "tau": best_tau,
             "timesteps": best_T, "no_warm": True})
    results.append(r)
    print("   →", fmt(r), flush=True)

    # ---- [F] final 稳定性（best 配置 × 3 seed）----
    print(f"\n### [F] final 稳定性（thr={best_thr}, tau={best_tau}, T={best_T}, seed=42/123/2024）", flush=True)
    for seed in (42, 123, 2024):
        r = run({"tag": f"F_seed{seed}", "threshold": best_thr, "tau": best_tau,
                 "timesteps": best_T, "seed": seed, "epochs": 50, "patience": 10})
        results.append(r)
        print("   →", fmt(r), flush=True)

    # ---- 汇总表 ----
    print("\n\n" + "=" * 90)
    print("消融汇总（ReLU 代理基线：clean 75.77% / HBC 75.77%）")
    print("=" * 90)
    for r in results:
        print(fmt(r))
    print(f"\n最优配置：threshold={best_thr}, tau={best_tau}, timesteps={best_T}")

    summary = ALGO_DIR / "lif_surrogate_ablation_summary.txt"
    with summary.open("w", encoding="utf-8") as f:
        f.write("LIF-EEGNet surrogate 消融汇总\n")
        f.write("ReLU 代理基线：clean 75.77% / HBC 75.77%\n")
        f.write("=" * 60 + "\n")
        for r in results:
            f.write(fmt(r) + "\n")
        f.write(f"\n最优配置：threshold={best_thr}, tau={best_tau}, timesteps={best_T}\n")
    print(f"\n汇总已写入：{summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
