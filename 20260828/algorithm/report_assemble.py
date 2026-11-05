#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""report_assemble.py — 汇总 ssl_ae_results/ 下的 finetune_*.json + finetune_refs.json
为报告用总表（ssl_ae_results/summary.json），并出对比图（ssl_ae_results/ssl_compare.png）。
只读既有 JSON，不产生任何数字。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "ssl_ae_results"


def main() -> int:
    cells = sorted(OUT.glob("finetune_*.json"))
    rows = [json.loads(p.read_text(encoding="utf-8")) for p in cells]
    refs = {}
    ref_path = OUT / "finetune_refs.json"
    if ref_path.exists():
        refs = json.loads(ref_path.read_text(encoding="utf-8"))

    summary = {"cells": rows, "refs": refs}
    # 核对：8 组 + 引用 ≥4
    print(f"微调结果组数: {len(rows)}（期望 8）")
    for r in rows:
        print(f"  {r['cell']:28s} clean={r['acc_clean']*100:6.2f}%  "
              f"8dB={r['acc_8db']*100:6.2f}%  best_ep={r['best_epoch']:3d}  "
              f"n_train={r['n_train']}")
    if refs:
        print(f"既有引用模型复测: {len(refs)}")
        for k, v in refs.items():
            print(f"  {k:32s} clean={v['acc_clean']*100:6.2f}%  8dB={v['acc_8db']*100:6.2f}%")

    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1),
                                      encoding="utf-8")
    print(f"✓ 已保存: {OUT / 'summary.json'}")

    # ---- 对比图（clean 与 8dB 两组，×标签比例）----
    # 与本次微调同协议（HBC 增强）的既有引用：lif warm 100%（aug_s42）并入序列
    merged = [dict(r) for r in rows]
    ref100 = refs.get("lif_warm_100_aug(s42)")
    if ref100:
        merged.append({"act": "lif", "init": "warm", "frac": 1.0, "from_ref": True,
                       "acc_clean": ref100["acc_clean"], "acc_8db": ref100["acc_8db"]})
    frac_ord = {100: 0, 30: 1, 10: 2}
    series_def = [
        ("relu 从零(rand)", "relu", "rand", "#B0B0B0"),
        ("relu 预训练(ssl)", "relu", "ssl", "#7FB3D8"),
        ("LIF 监督热启动(warm)", "lif", "warm", "#E76F51"),
        ("LIF 预训练(ssl)", "lif", "ssl", "#00B4D8"),
    ]
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams["font.sans-serif"] = ["Hiragino Sans GB", "STHeiti", "Arial Unicode MS"]
        plt.rcParams["axes.unicode_minus"] = False

        fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), dpi=150)
        for ax, key, title in ((axes[0], "acc_clean", "clean eval 准确率 (%)"),
                               (axes[1], "acc_8db", "8dB 鲁棒性 (%)")):
            series = []
            for name, act, init, color in series_def:
                pts = {}
                for r in merged:
                    if r["act"] == act and r["init"] == init:
                        f = int(round(r["frac"] * 100))
                        pts[f] = r[key] * 100
                if pts:
                    series.append((name, color, pts))
            for name, color, pts in series:
                xs = [frac_ord[f] for f in sorted(pts)]
                ys = [pts[f] for f in sorted(pts)]
                ax.plot(xs, ys, marker="o", lw=2, color=color, label=name)
            ax.set_xticks([0, 1, 2])
            ax.set_xticklabels(["100%", "30%", "10%"])
            ax.set_title(title)
            ax.set_ylim(0, 72)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
        fig.suptitle("自监督预训练 vs 从零监督（BCIC 2a，EEG-TCNet 9.5K，HBC 增强微调协议）",
                     fontsize=13, weight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        fig.savefig(OUT / "ssl_compare.png", bbox_inches="tight")
        plt.close(fig)
        print(f"✓ 对比图: {OUT / 'ssl_compare.png'}")
    except Exception as e:  # noqa: BLE001
        print(f"⚠ 绘图失败: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
