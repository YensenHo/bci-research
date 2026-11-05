#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""gap_scan.py — 在指定数据集上扫描教师容量（gap = teacher_params / student_params），
记录 30% 标签预算下的蒸馏红利，验证「gap × 天花板」关系、定位负红利阈值 gap*。

默认数据集：EEGMMIDB 4 类（跨被试，21ch/320，低天花板 ~38%，反例——负红利出现处）。
（注意：data_4class 是 BCIC IV 2a within-subject 22ch/500，天花板高 ~54%，正例对照。）

用法（对齐 distill_4class.py 的 BRIEF 配置与协议）：
  # ① dry-run：打印各教师配置的参数量与 gap 倍数
  env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin \
    /opt/miniconda3/envs/openmmlab/bin/python gap_scan.py --dry \
    --data-dir eegmmidb/data --n-channels 21 --input-length 320

  # ② 全量扫描
  env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin \
    /opt/miniconda3/envs/openmmlab/bin/python gap_scan.py \
    --data-dir eegmmidb/data --n-channels 21 --input-length 320 --seeds 42

设计：
  固定学生 F1=8/D=2/F2=16/FT=12，扫描教师 (F1,F2,FT)，gap 覆盖 1.0× → ~14×。
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ALGO_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ALGO_DIR))

# 与 distill_4class.py 的 BRIEF 完全一致
BRIEF = dict(kernel_length=63, tcn_kernel_size=12, pool1=4, pool2=8, readout_mode="gap")

# 学生配置（固定）
STUDENT = dict(F1=8, D=2, F2=16, FT=12)

# 教师配置扫描（gap 从 1× 到 ~14×）
TEACHER_CFGS = [
    ("1.0x", 8, 16, 12),    # 教师 = 学生容量
    ("1.5x", 8, 16, 16),    # 已有锚点
    ("2.2x", 12, 24, 18),
    ("3.7x", 16, 32, 24),   # 已有锚点
    ("6.8x", 24, 48, 32),
    ("14.2x", 32, 64, 48),
]

PY = "/opt/miniconda3/envs/openmmlab/bin/python"
ENV = dict(HOME=str(Path.home()),
           PATH="/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin")


def n_params(F1, D, F2, FT, n_channels, activation="elu"):
    from eeg_tcnet import EEGTCNet
    m = EEGTCNet(n_channels=n_channels, n_classes=4, input_length=500,
                 F1=F1, D=D, F2=F2, FT=FT, dropout=0.25,
                 activation=activation, **BRIEF)
    return m.num_params()


def report_suffix(f1, f2, ft, data_dir):
    """对齐 distill_4class.py 的 report 文件命名逻辑。"""
    suffix = "" if data_dir == "data_4class" else f"_{Path(data_dir).name}"
    if f1 != 16 or f2 != 32 or ft != 24:
        suffix += f"_tf{f1}-{f2}-{ft}"
    return suffix


def dry_run(n_channels):
    s = n_params(STUDENT["F1"], STUDENT["D"], STUDENT["F2"], STUDENT["FT"], n_channels, "relu")
    print(f"学生 relu : {s:,} 参数（gap 1.0×，F1={STUDENT['F1']}/F2={STUDENT['F2']}/FT={STUDENT['FT']}，{n_channels}ch）")
    print(f"{'标签':>6} {'F1/F2/FT':>14} {'teacher参数':>12} {'gap×':>7}")
    for tag, f1, f2, ft in TEACHER_CFGS:
        p = n_params(f1, STUDENT["D"], f2, ft, n_channels)
        print(f"{tag:>6} {f'{f1}/{f2}/{ft}':>14} {p:>12,} {p/s:>7.2f}")


def run_one(tag, f1, f2, ft, seed_list, data_dir, n_channels, n_classes,
            input_length, label_fracs="0.3"):
    """subprocess 调 distill_4class.py，返回 (teacher_params, 30% gain_mean_pp)。"""
    cmd = [PY, str(ALGO_DIR / "distill_4class.py"),
           "--student", "relu",
           "--teacher-f1", str(f1), "--teacher-f2", str(f2), "--teacher-ft", str(ft),
           "--label-fracs", label_fracs,
           "--seeds", ",".join(str(s) for s in seed_list),
           "--device", "mps",
           "--data-dir", data_dir,
           "--n-channels", str(n_channels),
           "--n-classes", str(n_classes),
           "--input-length", str(input_length)]
    print(f"\n{'='*64}\n[{tag}] 教师 F1={f1}/F2={f2}/FT={ft} | {data_dir} | "
          f"{n_channels}ch/{input_length} | 标签 {label_fracs} | seeds {seed_list}\n{'='*64}")
    t0 = time.time()
    r = subprocess.run(cmd, cwd=ALGO_DIR, env=ENV, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-2000:])
        print(r.stderr[-2000:])
        raise SystemExit(f"[{tag}] distill_4class.py 失败 rc={r.returncode}")
    report = ALGO_DIR / f"distill_4class_relu_report{report_suffix(f1, f2, ft, data_dir)}.json"
    data = json.loads(report.read_text(encoding="utf-8"))
    tp = data["config"]["teacher_params"]
    sp = data["config"]["student_params"]
    frac_key = f"{float(label_fracs.split(',')[0]):.0%}"
    gain = data["aggregated"][frac_key]["gain_mean_pp"]
    print(f"  → teacher {tp:,} / student {sp:,} = {tp/sp:.2f}× | 30% 红利 {gain:+.2f}pp（{time.time()-t0:.0f}s）")
    return tp, sp, gain


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true", help="只打印参数量与 gap 倍数")
    ap.add_argument("--seeds", type=str, default="42,123")
    ap.add_argument("--label-fracs", type=str, default="0.3")
    ap.add_argument("--data-dir", type=str, default="eegmmidb/data")
    ap.add_argument("--n-channels", type=int, default=21)
    ap.add_argument("--n-classes", type=int, default=4)
    ap.add_argument("--input-length", type=int, default=320)
    args = ap.parse_args()

    if args.dry:
        dry_run(args.n_channels)
        return 0

    seeds = [int(s) for s in args.seeds.split(",")]
    rows = []
    for tag, f1, f2, ft in TEACHER_CFGS:
        tp, sp, gain = run_one(tag, f1, f2, ft, seeds, args.data_dir,
                               args.n_channels, args.n_classes, args.input_length,
                               args.label_fracs)
        rows.append((tag, tp, sp, gain))

    print(f"\n\n{'='*64}\n gap 扫描汇总（30% 标签红利，{args.data_dir}）\n{'='*64}")
    print(f"{'tag':>6} {'gap×':>7} {'teacher参数':>12} {'30%红利':>9}")
    for tag, tp, sp, gain in rows:
        print(f"{tag:>6} {tp/sp:>7.2f} {tp:>12,} {gain:>+9.2f}pp")
    out = [{"tag": t, "gap": tp/sp, "teacher_params": tp, "student_params": sp,
            "gain_30pct_pp": g} for t, tp, sp, g in rows]
    (ALGO_DIR / "gap_scan_summary.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n✓ 汇总已保存：gap_scan_summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
