#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
demo_visualizer.py — 上位机可视化脚本（端侧解码 → HBC 回传 → 上位机可视化，四分类）

演示链路（完整闭环，四分类 左手/右手/双脚/舌头）：
  板载 RV1106 (端侧)                        上位机 (Mac)
  ─────────────────────                    ──────────────────
  lif_surrogate_fusion_4class              demo_visualizer.py
  ① LIF-EEGNet 直连解码（类脑，四分类）      读取 adb 流式输出
  ② 8bit 量化 → OOK → HBC 信道仿真         解析逐窗口结果
     → 包络解调 → 反量化（"回传"）         终端实时打印 + 暗色图
  ③ 收端二次解码（经HBC，四分类）           （可选）matplotlib 实时曲线

板载端复用现有二进制（不重新编译）：
  - ./lif_surrogate_fusion_4class <bin> stream → 逐窗口流式输出（演示主链路）
  - ./lif_surrogate_fusion_4class <bin>         → 全量统计（直连/经HBC/一致率/每类精度）

三种运行模式（--live / --file / --mock）：
  --live   默认：adb shell 启动板载 lif_surrogate_fusion_4class stream，实时流式可视化
  --file   离线：回放一段已抓取的板载输出日志（如 lif_4class_stream.log）
  --mock   纯离线：无板子时用合成数据演示可视化本身（不冒充真实板载数字）

依赖：numpy（必需）、matplotlib（可选，缺省则自动退化为纯终端模式）。
运行（遵循本项目 Python 环境铁律，env -i 隔离）：
  env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin \\
      /opt/miniconda3/envs/openmmlab/bin/python demo_visualizer.py --file lif_4class_stream.log
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

import numpy as np

# ---------------------------------------------------------------------------
# 0. 常量与标签映射
# ---------------------------------------------------------------------------
LABEL_CN = ["左手", "右手", "双脚", "舌头"]   # 四分类，与板载 lif_surrogate_fusion_4class.c 的 LABEL_CN[4] 一致
LABEL_EN = ["left hand", "right hand", "both feet", "tongue"]

# 板载流式输出行格式：
#   [  0] 真值=左手 直连=右手 经HBC=右手 ✗ | 60.2ms (实时余量 97%)
LINE_RE = re.compile(
    r"\[\s*(\d+)\]\s*"
    r"真值=(\S+)\s+直连=(\S+)\s+经HBC=(\S+)\s+([✓✗])\s*\|\s*"
    r"([\d.]+)ms\s*\(实时余量\s*([\d.]+)%\)"
)

# 汇总块行格式（lif_surrogate_fusion_4class.c 末尾；兼容二分类 hbc_bci_realtime.c 旧格式）：
#   直连解码准确率 : 56.8673% (1474/2592)
#   经HBC解码准确率: 57.1373% (1481/2592)
#   两者一致率     : 97.8395% (2536/2592)
SUMMARY_RE = re.compile(
    r"(直连解码准确率|直连准确率|经HBC解码准确率|经HBC准确率|两者一致率|平均单窗口)"
    r"\s*:\s*([\d.]+)\s*(%|ms)"
)

DEFAULT_ADB = str(Path.home() / "luckfox" / "platform-tools" / "adb")
BOARD_BIN = "lif_4class_test_full.bin"      # 四分类全量数据（2592 试次，stream 模式取前若干窗口）
BOARD_CMD = "cd /root && ./lif_surrogate_fusion_4class /root/lif_4class_test_full.bin stream"


# ---------------------------------------------------------------------------
# 1. 数据结构：单个窗口结果 + 汇总统计
# ---------------------------------------------------------------------------
@dataclass
class WindowResult:
    """解析自板载一行的单窗口结果。"""
    idx: int
    truth: int          # 真值类别 0~3（左手/右手/双脚/舌头）
    direct: int         # 直连解码预测 0~3
    hbc: int            # 经HBC回传后解码预测 0~3
    hbc_ok: bool        # 经HBC 是否命中真值
    direct_ok: bool     # 直连是否命中真值
    latency_ms: float   # 单窗口端到端处理耗时
    headroom_pct: float # 实时余量（1 - latency/2000ms）


@dataclass
class DemoStats:
    """滚动累计统计（含滑动窗口精度曲线）。"""
    results: list[WindowResult] = field(default_factory=list)
    summary: dict[str, str] = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.results)

    def direct_acc(self, last: Optional[int] = None) -> float:
        """直连累计/近 last 窗口准确率。"""
        r = self.results[-last:] if last else self.results
        return (sum(1 for w in r if w.direct_ok) / len(r) * 100.0) if r else 0.0

    def hbc_acc(self, last: Optional[int] = None) -> float:
        """经HBC累计/近 last 窗口准确率。"""
        r = self.results[-last:] if last else self.results
        return (sum(1 for w in r if w.hbc_ok) / len(r) * 100.0) if r else 0.0

    def consistent_rate(self) -> float:
        """直连与经HBC 预测一致率（HBC 回传是否改变解码结果）。"""
        r = self.results
        return (sum(1 for w in r if w.direct == w.hbc) / len(r) * 100.0) if r else 0.0

    def per_class_acc(self, which: str = "direct") -> list[float]:
        """每类准确率（四分类，0~3 各返回一个百分比；which ∈ {direct, hbc}）。"""
        acc = [0.0] * len(LABEL_CN)
        if not self.results:
            return acc
        for c in range(len(LABEL_CN)):
            cls = [w for w in self.results if w.truth == c]
            if cls:
                ok = (sum(1 for w in cls if w.direct_ok)
                      if which == "direct" else
                      sum(1 for w in cls if w.hbc_ok))
                acc[c] = 100.0 * ok / len(cls)
        return acc

    def latency_p50_p99(self) -> tuple[float, float]:
        a = np.array([w.latency_ms for w in self.results])
        return (float(np.percentile(a, 50)), float(np.percentile(a, 99))) if len(a) else (0.0, 0.0)


# ---------------------------------------------------------------------------
# 2. 解析
# ---------------------------------------------------------------------------
def parse_line(line: str) -> Optional[WindowResult]:
    """解析板载流式输出的一行；非数据行（表头/汇总）返回 None。"""
    line = line.strip().strip("\r\n")
    m = LINE_RE.search(line)
    if not m:
        return None
    idx, truth_cn, direct_cn, hbc_cn, mark, lat, hroom = m.groups()
    try:
        truth = LABEL_CN.index(truth_cn)
        direct = LABEL_CN.index(direct_cn)
        hbc = LABEL_CN.index(hbc_cn)
    except ValueError:
        return None
    return WindowResult(
        idx=int(idx),
        truth=truth,
        direct=direct,
        hbc=hbc,
        hbc_ok=(hbc == truth),
        direct_ok=(direct == truth),
        latency_ms=float(lat),
        headroom_pct=float(hroom),
    )


def parse_summary(line: str, stats: DemoStats) -> None:
    """提取末尾汇总块的行（如有）。"""
    m = SUMMARY_RE.search(line)
    if m:
        stats.summary[m.group(1)] = f"{m.group(2)}{m.group(3)}"


# ---------------------------------------------------------------------------
# 3. 数据源：live（adb）/ file（离线回放）/ mock（合成）
# ---------------------------------------------------------------------------
def iter_live(adb: str, timeout_s: float = 120.0) -> Iterator[str]:
    """通过 adb 启动板载 lif_surrogate_fusion_4class stream，逐行产出 stdout。

    板子若掉线/超时，抛异常，由上层捕获后提示可切离线模式。
    """
    cmd = [adb, "shell", BOARD_CMD]
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    assert proc.stdout is not None
    deadline = time.time() + timeout_s
    for raw in proc.stdout:
        yield raw
        if time.time() > deadline:
            proc.terminate()
            break
    proc.wait()


def iter_file(path: str) -> Iterator[str]:
    """离线回放：逐行读已抓取的板载日志。"""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            yield line


def iter_mock(n_windows: int = 60, seed: int = 0) -> Iterator[str]:
    """纯离线合成数据源（无板子时的可视化自检，不冒充真实板载数字）。

    生成与板载格式完全一致的四分类行；准确率约 57%（与板载四分类实测
    56.87%/57.14% 同量级），但明确标注为 MOCK。
    """
    rng = np.random.default_rng(seed)
    labels = LABEL_CN
    print(f"[MOCK] 生成 {n_windows} 个合成窗口（四分类，非板载真实数据，仅演示可视化）",
          file=sys.stderr)
    for i in range(n_windows):
        truth = i % len(labels)
        # 约 57% 正确率（四分类，与板载实测同量级）+ 直连/经HBC 高度一致
        direct_ok = rng.random() < 0.57
        hbc_ok = direct_ok and (rng.random() < 0.98)  # 经HBC 与直连基本一致
        others = [c for c in range(len(labels)) if c != truth]
        direct = truth if direct_ok else int(rng.choice(others))
        hbc = truth if hbc_ok else int(rng.choice([c for c in others if c != direct] or others))
        mark = "✓" if hbc_ok else "✗"
        lat = 133.0 + rng.random() * 5.0   # ~133-138ms，与板载四分类实测同量级
        hroom = 100.0 * (1.0 - lat / 2000.0)
        yield (f"[{i:3d}] 真值={labels[truth]} 直连={labels[direct]} "
               f"经HBC={labels[hbc]} {mark} | {lat:.1f}ms (实时余量 {hroom:.1f}%)")


# ---------------------------------------------------------------------------
# 4. 终端可视化
# ---------------------------------------------------------------------------
class TerminalView:
    """ANSI 终端实时渲染：进度条 + 累计指标表。"""

    def __init__(self) -> None:
        self._clear = "\033[2J\033[H"  # 清屏并回到左上角

    def render(self, stats: DemoStats, frame_no: int) -> None:
        n = stats.n
        if n == 0:
            return
        w = stats.results[-1]
        p50, p99 = stats.latency_p50_p99()
        da = stats.direct_acc()
        ha = stats.hbc_acc()
        cr = stats.consistent_rate()

        lines = [
            "=" * 72,
            "  RV1106 端侧 HBC×BCI 融合（四分类）—— 上位机实时可视化",
            "=" * 72,
            f"  窗口 #{n:<4d}  真值={LABEL_CN[w.truth]}  直连={LABEL_CN[w.direct]}  "
            f"经HBC={LABEL_CN[w.hbc]}  {'✓' if w.hbc_ok else '✗'}",
            f"  耗时 {w.latency_ms:6.1f} ms（2s 窗口，实时余量 {w.headroom_pct:.0f}%）",
            "-" * 72,
            f"  直连准确率   : {da:5.1f}%   {self._bar(da)}",
            f"  经HBC准确率  : {ha:5.1f}%   {self._bar(ha)}",
            f"  两者一致率   : {cr:5.1f}%   （HBC 回传是否改变解码结果）",
            f"  延迟 p50/p99 : {p50:5.1f} / {p99:5.1f} ms（p99 < 2000ms 即实时）",
        ]
        if stats.summary:
            lines.append("-" * 72)
            for k, v in stats.summary.items():
                lines.append(f"  [板载汇总] {k}: {v}")
        lines.append("=" * 72)
        sys.stdout.write(self._clear + "\n".join(lines) + "\n")
        sys.stdout.flush()

    @staticmethod
    def _bar(pct: float, width: int = 40) -> str:
        filled = int(round(pct / 100.0 * width))
        return "█" * filled + "░" * (width - filled)


# ---------------------------------------------------------------------------
# 5. matplotlib 暗色图（可选依赖）
# ---------------------------------------------------------------------------
class MatplotlibView:
    """暗色 2×2 面板：精度曲线 / 延迟 / 预测一致性 / ✓✗ 条带。"""

    def __init__(self, interactive: bool, save_path: Optional[str], delay: float = 0.01):
        import matplotlib
        # 交互模式用 TkAgg 实时刷新，否则显式 Agg（无显示环境也能存图）
        backend = "TkAgg" if interactive else "Agg"
        try:
            matplotlib.use(backend)
        except Exception:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        self.plt = plt
        self.interactive = interactive
        self.save_path = save_path
        self.delay = delay
        # 中文字体（macOS：PingFang SC 为 .ttc 集合，fontManager 不收录，用 Hiragino Sans GB）
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.sans-serif"] = ["Hiragino Sans GB", "STHeiti", "Arial Unicode MS"]
        plt.rcParams["axes.unicode_minus"] = False
        plt.rcParams.update({
            "figure.facecolor": "#0d1117",
            "axes.facecolor": "#161b22",
            "axes.edgecolor": "#30363d",
            "axes.labelcolor": "#c9d1d9",
            "text.color": "#c9d1d9",
            "xtick.color": "#8b949e",
            "ytick.color": "#8b949e",
            "grid.color": "#30363d",
            "axes.grid": True,
        })
        self.fig, self.ax = plt.subplots(2, 2, figsize=(12, 8))
        self.fig.suptitle("RV1106 端侧 HBC×BCI 融合（四分类）—— 上位机可视化",
                          color="#e6edf3", fontsize=14)
        if interactive:
            self.plt.ion()

    def update(self, stats: DemoStats) -> None:
        n = stats.n
        if n == 0:
            return
        xs = np.arange(1, n + 1)
        da = np.array([stats.direct_acc(last=k) for k in range(1, n + 1)])
        ha = np.array([stats.hbc_acc(last=k) for k in range(1, n + 1)])
        lat = np.array([w.latency_ms for w in stats.results])
        truth = np.array([w.truth for w in stats.results])
        direct = np.array([w.direct for w in stats.results])
        hbc = np.array([w.hbc for w in stats.results])
        ok = np.array([w.hbc_ok for w in stats.results], dtype=int)

        # 左上：累计准确率曲线（直连 vs 经HBC）
        ax = self.ax[0, 0]
        ax.clear()
        ax.plot(xs, da, color="#58a6ff", lw=1.8, label="直连解码")
        ax.plot(xs, ha, color="#3fb950", lw=1.8, label="经HBC回传")
        ax.axhline(25, color="#30363d", ls="--", lw=1)   # 四分类随机基线
        ax.set_ylim(0, 100)
        ax.set_title("累计准确率（四分类，随机=25%）", color="#c9d1d9")
        ax.set_xlabel("窗口序号")
        ax.set_ylabel("准确率 %")
        ax.legend(loc="lower right", facecolor="#161b22", edgecolor="#30363d",
                  labelcolor="#c9d1d9")

        # 右上：单窗口延迟（实时余量）
        ax = self.ax[0, 1]
        ax.clear()
        ax.plot(xs, lat, color="#d29922", lw=1.2, label="单窗口耗时")
        ax.axhline(2000, color="#f85149", ls="--", lw=1, label="2s 窗口边界")
        ax.set_title("单窗口端到端延迟（ms）")
        ax.set_xlabel("窗口序号")
        ax.set_ylabel("延迟 ms")
        ax.legend(loc="upper right", facecolor="#161b22", edgecolor="#30363d",
                  labelcolor="#c9d1d9")

        # 左下：直连 vs 经HBC 逐窗口预测散点（四分类，微抖动避免重叠）
        ax = self.ax[1, 0]
        ax.clear()
        jitter = (np.arange(n) % 5) * 0.06 - 0.12   # 确定性微抖动 [-0.12, 0.12)
        ax.scatter(xs, direct + jitter, s=14, color="#58a6ff", alpha=0.6, label="直连")
        ax.scatter(xs, hbc - jitter, s=24, marker="x", color="#3fb950", alpha=0.8, label="经HBC")
        ax.set_yticks(range(len(LABEL_CN)))
        ax.set_yticklabels(LABEL_CN)
        ax.set_ylim(-0.5, len(LABEL_CN) - 0.5)
        ax.set_title("逐窗口预测（直连 vs 经HBC，四分类）")
        ax.set_xlabel("窗口序号")
        ax.legend(loc="upper right", facecolor="#161b22", edgecolor="#30363d",
                  labelcolor="#c9d1d9")

        # 右下：✓/✗ 条带（经HBC 命中与否）
        ax = self.ax[1, 1]
        ax.clear()
        cmap = {1: "#3fb950", 0: "#f85149"}
        colors = [cmap[v] for v in ok]
        ax.bar(xs, np.ones(n), color=colors, width=1.0)
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_title("经HBC 解码命中条带（绿=命中 红=未命中）", color="#c9d1d9")
        ax.set_xlabel("窗口序号")

        self.fig.tight_layout()
        if self.interactive:
            self.plt.pause(self.delay)
        if self.save_path:
            self.fig.savefig(self.save_path, dpi=130, facecolor="#0d1117")

    def close(self) -> None:
        self.plt.close(self.fig)


# ---------------------------------------------------------------------------
# 6. 主流程
# ---------------------------------------------------------------------------
def run(source: Iterator[str], stats: DemoStats, tv: TerminalView,
        mv: Optional[MatplotlibView], save_path: Optional[str]) -> None:
    """消费数据源，逐窗口更新终端与图。返回后统计写入 stats。"""
    frame = 0
    for line in source:
        w = parse_line(line)
        if w is not None:
            stats.results.append(w)
            frame += 1
            tv.render(stats, frame)
            if mv is not None:
                mv.update(stats)
        else:
            parse_summary(line, stats)

    # 收尾：无窗口数据时提示
    if stats.n == 0:
        print("[!] 未解析到任何板载窗口数据。请检查板子/日志格式。", file=sys.stderr)

    if mv is not None:
        if not mv.interactive and save_path:
            mv.update(stats)          # Agg 下最后画一次并存图
        mv.close()

    # 最终摘要打印
    print("\n" + "=" * 72)
    print("  演示结束 —— 汇总")
    print("=" * 72)
    print(f"  窗口总数     : {stats.n}")
    if stats.n:
        p50, p99 = stats.latency_p50_p99()
        print(f"  直连准确率   : {stats.direct_acc():.1f}%")
        print(f"  经HBC准确率  : {stats.hbc_acc():.1f}%")
        print(f"  两者一致率   : {stats.consistent_rate():.1f}%")
        print(f"  延迟 p50/p99 : {p50:.1f} / {p99:.1f} ms")
        print(f"  实时余量     : {2000.0 / p99:.0f}×（2s 窗口 / p99）")
        da_cls, ha_cls = stats.per_class_acc("direct"), stats.per_class_acc("hbc")
        print("  每类准确率   : " + "  ".join(
            f"{LABEL_CN[c]} {da_cls[c]:.1f}/{ha_cls[c]:.1f}%" for c in range(len(LABEL_CN))))
        print("                （直连/经HBC，四分类 0=左手 1=右手 2=双脚 3=舌头）")
    if stats.summary:
        print("  [板载原始汇总]")
        for k, v in stats.summary.items():
            print(f"    {k}: {v}")
    if save_path and Path(save_path).exists():
        print(f"\n  图已保存     : {save_path}")
    print("=" * 72)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="RV1106 端侧 HBC×BCI 融合 —— 上位机可视化",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--live", action="store_true",
                    help="通过 adb 实时驱动板载 lif_surrogate_fusion_4class stream（默认，需板子在线）")
    ap.add_argument("--file", type=str, default=None,
                    help="离线回放板载输出日志（如 lif_4class_stream.log）")
    ap.add_argument("--mock", action="store_true",
                    help="纯离线合成数据（无板子，仅演示可视化，不冒充真实数字）")
    ap.add_argument("--adb", type=str, default=DEFAULT_ADB, help="adb 路径")
    ap.add_argument("--windows", type=int, default=60, help="mock 模式窗口数")
    ap.add_argument("--no-plot", action="store_true", help="关闭 matplotlib，仅终端")
    ap.add_argument("--save", type=str, default="demo_visualizer.png",
                    help="最终图保存路径（关闭用 --no-plot）")
    ap.add_argument("--timeout", type=float, default=180.0,
                    help="live 模式 adb 超时（秒）")
    ap.add_argument("--replay", action="store_true",
                    help="回放 --file 日志时弹实时窗口（录屏用，配合 --file）")
    ap.add_argument("--delay", type=float, default=0.25,
                    help="replay 每窗口间隔秒数（默认 0.25，越大越慢）")
    args = ap.parse_args()

    # ---- 选择数据源 ----
    stats = DemoStats()
    if args.file:
        print(f"[数据源] 离线回放日志：{args.file}")
        source = iter_file(args.file)
    elif args.live or (not args.mock):
        print(f"[数据源] adb 实时驱动板载（{args.adb}）")
        try:
            source = iter_live(args.adb, timeout_s=args.timeout)
        except FileNotFoundError:
            print(f"[!] 找不到 adb：{args.adb}，请用 --file 或 --mock 离线演示。",
                  file=sys.stderr)
            return 2
    else:
        source = iter_mock(args.windows)

    # ---- 视图 ----
    tv = TerminalView()
    mv = None
    if not args.no_plot:
        try:
            mv = MatplotlibView(interactive=(args.file is None and not args.mock) or args.replay,
                                save_path=args.save, delay=args.delay)
            print(f"[绘图] matplotlib 暗色面板已启用，最终保存 {args.save}")
        except Exception as exc:  # noqa: BLE001
            print(f"[!] matplotlib 不可用（{exc}），退化为纯终端模式。", file=sys.stderr)
            mv = None

    try:
        run(source, stats, tv, mv, args.save)
    except KeyboardInterrupt:
        print("\n[!] 用户中断。")
        return 130
    except Exception as exc:  # noqa: BLE001
        print(f"\n[!] 运行出错：{exc}", file=sys.stderr)
        print("    若板子掉线，可用 --file lif_4class_stream.log 离线演示。", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
