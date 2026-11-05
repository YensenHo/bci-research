#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""gen_html_78.py — 生成「知识蒸馏符号判据 · 第 7-8 周汇报」自包含暗色 HTML。

数据来源（真实实验，双 seed 均值）：
  - 符号判据 19+ 点：research/蒸馏符号判据_证据定稿_20261026.md + data12/sleep
  - K/R 分解：同上
  - 信噪比相图：research/信噪比相图_意外发现_20261027.md
  - HBC 真实信道：research/HBC真实信道噪声_蒸馏红利边界_20261027.md

产出：/Users/heyixuan/BCI/周报/组会展示_第78周_20261028.html
"""
from __future__ import annotations

from pathlib import Path

OUT = Path("/Users/heyixuan/BCI/周报/组会展示_第78周_20261028.html")

# ---------------- 数据（全部来自真实实验文档） ----------------
# (name, 任务, t-s, 红利, 强regime)
POINTS = [
    ("EEGMMIDB-4c", "MI 4类跨被试", -2.20, -2.78, True),
    ("BCIC-2a-cross", "MI 4类跨被试", 4.68, 5.56, True),
    ("BCIC-2a-within", "MI 4类 within", 3.59, 4.17, True),
    ("S-A01", "MI 4类 within", 3.47, 6.42, True),
    ("S-A02", "MI 4类 within", 2.08, 1.39, True),
    ("S-A03", "MI 4类 within", -3.64, -1.21, True),
    ("S-A04", "MI 4类 within", 10.07, 2.25, True),
    ("S-A05", "MI 4类 within", 5.03, 3.12, True),
    ("S-A06", "MI 4类 within", 7.46, 3.99, True),
    ("S-A07", "MI 4类 within", 7.12, 4.68, True),
    ("S-A08", "MI 4类 within", 3.47, 7.30, True),
    ("CHB-MIT", "癫痫 2类跨患者", -3.28, -0.89, True),
    ("NinaPro-17", "EMG 17类跨被试", 4.93, 3.74, True),
    ("NinaPro-12", "EMG 12类跨被试", 2.00, 1.62, False),
    ("BCI-2b", "MI 2类跨被试", 0.18, 1.89, False),
    ("EEGMMIDB-2c", "MI 2类跨被试", 1.48, 2.22, False),
    ("EEGMMIDB-2c9", "MI 2类(小样本)", 1.29, 0.37, False),
    ("P300", "P300 2类跨被试", -0.43, 1.12, False),
    ("S-A09", "MI 4类 within", -0.35, 3.12, False),
    ("data12", "MI 4类减被试", -2.23, -1.57, True),
    ("sleep", "睡眠 5类分期", -4.30, -0.60, True),
]

# K/R 分解（seed 42）
KR = [
    ("EEGMMIDB-4c", 36.30, -1.48, -2.59),
    ("BCIC-cross", 62.15, 4.51, 5.21),
    ("BCIC-within", 66.59, 1.89, 0.69),
    ("S-A03", 83.68, 0.35, -2.08),
    ("CHB-MIT", 77.48, -2.66, 2.66),
    ("EEGMMIDB-2c", 70.37, 0.00, 2.22),
    ("EEGMMIDB-2c9", 70.74, 0.74, -2.22),
]

# 信噪比相图（MI，双 seed 均值）
SNR = [(0, 5.56), (0.5, 1.39), (1.0, 16.15), (1.5, 2.34), (2.0, 9.46), (3.0, 11.20)]

# HBC 真实信道（seed 42）
HBC = [
    ("量化 quant", 7.99, 0.52, 7.47),
    ("突发 burst", 10.24, 3.12, 7.12),
    ("脉冲 impulse", 17.88, 13.89, 3.99),
    ("全退化 full", 15.80, 10.42, 5.38),
]


# ---------------- SVG 生成 ----------------
def svg_scatter(points, width=860, height=420):
    """t−s vs 红利散点图，对角线 y=x 表示 sign(gain)=sign(t−s)。"""
    L, R, T, B = 70, 40, 30, 55
    xmin, xmax = -6, 12
    ymin, ymax = -4, 8
    xs = lambda v: L + (v - xmin) / (xmax - xmin) * (width - L - R)
    ys = lambda v: T + (ymax - v) / (ymax - ymin) * (height - T - B)

    s = [f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;display:block">']
    # 网格 + 轴
    for gy in range(ymin, ymax + 1, 2):
        s.append(f'<line x1="{xs(xmin)}" y1="{ys(gy)}" x2="{xs(xmax)}" y2="{ys(gy)}" stroke="#1e2d3f" stroke-width="1"/>')
        s.append(f'<text x="{xs(xmin)-8}" y="{ys(gy)+4}" text-anchor="end" fill="#8b949e" font-size="12">{gy}</text>')
    for gx in range(xmin, xmax + 1, 2):
        s.append(f'<line x1="{xs(gx)}" y1="{ys(ymin)}" x2="{xs(gx)}" y2="{ys(ymax)}" stroke="#1e2d3f" stroke-width="1"/>')
        s.append(f'<text x="{xs(gx)}" y="{ys(ymin)+18}" text-anchor="middle" fill="#8b949e" font-size="12">{gx}</text>')
    # 零线
    s.append(f'<line x1="{xs(xmin)}" y1="{ys(0)}" x2="{xs(xmax)}" y2="{ys(0)}" stroke="#6e7681" stroke-width="1.4" stroke-dasharray="5,4"/>')
    s.append(f'<line x1="{xs(0)}" y1="{ys(ymin)}" x2="{xs(0)}" y2="{ys(ymax)}" stroke="#6e7681" stroke-width="1.4" stroke-dasharray="5,4"/>')
    # 对角线 y=x（符号判据）
    # 只在第一/三象限画（x,y 同号即命中）
    s.append(f'<line x1="{xs(0)}" y1="{ys(0)}" x2="{xs(min(xmax,ymax))}" y2="{ys(min(xmax,ymax))}" stroke="#d29922" stroke-width="1.6" stroke-dasharray="6,4"/>')
    # 点
    for name, task, ts, gain, strong in points:
        cx, cy = xs(ts), ys(gain)
        color = "#f85149" if gain < 0 else "#3fb950"
        r = 6 if strong else 4
        fill = color if strong else "none"
        stroke_w = 2
        s.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r}" fill="{fill}" stroke="{color}" stroke-width="{stroke_w}"/>')
    # 图例
    s.append(f'<circle cx="{width-260}" cy="{T+10}" r="6" fill="#3fb950"/><text x="{width-246}" y="{T+14}" fill="#8b949e" font-size="12">正红利（强regime）</text>')
    s.append(f'<circle cx="{width-260}" cy="{T+32}" r="6" fill="#f85149"/><text x="{width-246}" y="{T+36}" fill="#8b949e" font-size="12">负红利（强regime）</text>')
    s.append(f'<circle cx="{width-260}" cy="{T+54}" r="4" fill="none" stroke="#3fb950" stroke-width="2"/><text x="{width-246}" y="{T+58}" fill="#8b949e" font-size="12">弱regime（判据不适用）</text>')
    s.append(f'<line x1="{width-260}" y1="{T+76}" x2="{width-230}" y2="{T+76}" stroke="#d29922" stroke-width="1.6" stroke-dasharray="6,4"/><text x="{width-246}" y="{T+80}" fill="#8b949e" font-size="12">判据线 y=x</text>')
    # 轴标签
    s.append(f'<text x="{width/2}" y="{height-10}" text-anchor="middle" fill="#c9d1d9" font-size="14" font-weight="600">teacher − student 满标签精度 (pp)</text>')
    s.append(f'<text x="16" y="{height/2}" text-anchor="middle" fill="#c9d1d9" font-size="14" font-weight="600" transform="rotate(-90 16 {height/2})">30% 标签蒸馏红利 (pp)</text>')
    s.append('</svg>')
    return "".join(s)


def svg_snr(snr, width=860, height=360):
    L, R, T, B = 70, 40, 30, 55
    xmin, xmax = -0.3, 3.3
    ymin, ymax = 0, 18
    xs = lambda v: L + (v - xmin) / (xmax - xmin) * (width - L - R)
    ys = lambda v: T + (ymax - v) / (ymax - ymin) * (height - T - B)
    s = [f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;display:block">']
    for gy in range(0, 19, 2):
        s.append(f'<line x1="{xs(xmin)}" y1="{ys(gy)}" x2="{xs(xmax)}" y2="{ys(gy)}" stroke="#1e2d3f" stroke-width="1"/>')
        s.append(f'<text x="{xs(xmin)-8}" y="{ys(gy)+4}" text-anchor="end" fill="#8b949e" font-size="12">{gy}</text>')
    for gx in [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
        s.append(f'<line x1="{xs(gx)}" y1="{ys(ymin)}" x2="{xs(gx)}" y2="{ys(ymax)}" stroke="#1e2d3f" stroke-width="1"/>')
        s.append(f'<text x="{xs(gx)}" y="{ys(ymin)+18}" text-anchor="middle" fill="#8b949e" font-size="12">{gx}</text>')
    # 折线
    pts = " ".join(f"{xs(x):.1f},{ys(y):.1f}" for x, y in snr)
    s.append(f'<polyline points="{pts}" fill="none" stroke="#00B4D8" stroke-width="2.5"/>')
    for x, y in snr:
        s.append(f'<circle cx="{xs(x):.1f}" cy="{ys(y):.1f}" r="5" fill="#00B4D8"/>')
    # 暴涨点标注
    s.append(f'<text x="{xs(1.0)}" y="{ys(16.15)-10}" text-anchor="middle" fill="#d29922" font-size="13" font-weight="700">+16.15 暴涨</text>')
    s.append(f'<text x="{width/2}" y="{height-10}" text-anchor="middle" fill="#c9d1d9" font-size="14" font-weight="600">加噪强度 σ（信噪比下降）</text>')
    s.append(f'<text x="16" y="{height/2}" text-anchor="middle" fill="#c9d1d9" font-size="14" font-weight="600" transform="rotate(-90 16 {height/2})">蒸馏红利 (pp)</text>')
    s.append('</svg>')
    return "".join(s)


def svg_hbc(hbc, width=860, height=360):
    n = len(hbc)
    L, R, T, B = 70, 40, 30, 55
    ymin, ymax = 0, 20
    names = [h[0] for h in hbc]
    gains = [h[1] for h in hbc]
    Ks = [h[2] for h in hbc]
    Rs = [h[3] for h in hbc]
    bar_w = 70
    group_w = (width - L - R) / n
    s = [f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;display:block">']
    for gy in range(0, 21, 5):
        s.append(f'<line x1="{L}" y1="{ys(gy,ymin,ymax,T,B,height)}" x2="{width-R}" y2="{ys(gy,ymin,ymax,T,B,height)}" stroke="#1e2d3f" stroke-width="1"/>')
        s.append(f'<text x="{L-8}" y="{ys(gy,ymin,ymax,T,B,height)+4}" text-anchor="end" fill="#8b949e" font-size="12">{gy}</text>')
    for i in range(n):
        cx = L + group_w * i + group_w / 2
        # R 堆叠（底部）+ K 堆叠（顶部）
        r_bottom = 0
        r_h = Rs[i] / ymax * (height - T - B)
        k_h = Ks[i] / ymax * (height - T - B)
        y_base = ys(0, ymin, ymax, T, B, height)
        # R 柱（从 0 向上）
        s.append(f'<rect x="{cx-bar_w/2}" y="{y_base-r_h}" width="{bar_w}" height="{r_h}" fill="#E76F51"/>')
        # K 柱（在 R 之上）
        s.append(f'<rect x="{cx-bar_w/2}" y="{y_base-r_h-k_h}" width="{bar_w}" height="{k_h}" fill="#00B4D8"/>')
        # 总红利标注
        total = Rs[i] + Ks[i]
        s.append(f'<text x="{cx}" y="{y_base-r_h-k_h-6}" text-anchor="middle" fill="#e6edf3" font-size="13" font-weight="700">+{total:.2f}</text>')
        # 名称
        s.append(f'<text x="{cx}" y="{height-12}" text-anchor="middle" fill="#c9d1d9" font-size="13">{names[i]}</text>')
    # 图例
    s.append(f'<rect x="{width-250}" y="{T}" width="14" height="14" fill="#00B4D8"/><text x="{width-230}" y="{T+12}" fill="#8b949e" font-size="12">K（知识传递）</text>')
    s.append(f'<rect x="{width-250}" y="{T+22}" width="14" height="14" fill="#E76F51"/><text x="{width-230}" y="{T+34}" fill="#8b949e" font-size="12">R（正则化）</text>')
    s.append(f'<text x="16" y="{height/2}" text-anchor="middle" fill="#c9d1d9" font-size="14" font-weight="600" transform="rotate(-90 16 {height/2})">红利分解 (pp)</text>')
    s.append('</svg>')
    return "".join(s)


def ys(v, ymin, ymax, T, B, height):
    return T + (ymax - v) / (ymax - ymin) * (height - T - B)


# ---------------- HTML 模板 ----------------
CSS = """
*{box-sizing:border-box;margin:0;padding:0}
body{background:#0d1117;color:#e6edf3;font-family:-apple-system,'PingFang SC','Hiragino Sans GB','Microsoft YaHei',sans-serif;line-height:1.7;-webkit-font-smoothing:antialiased;padding:40px 20px 80px}
.wrap{max-width:1080px;margin:0 auto}
header{text-align:center;padding:20px 0 34px;border-bottom:1px solid #21262d;margin-bottom:40px}
header h1{font-size:30px;font-weight:700;letter-spacing:.5px}
header .sub{color:#8b949e;font-size:14px;margin-top:10px}
h2.sec{font-size:22px;font-weight:700;margin:52px 0 6px;display:flex;align-items:center;gap:12px}
h2.sec .n{display:inline-flex;align-items:center;justify-content:center;width:30px;height:30px;border-radius:8px;font-size:15px;flex:none}
h2.sec .n.b{background:#58a6ff22;color:#58a6ff;border:1px solid #58a6ff55}
h2.sec .n.g{background:#3fb95022;color:#3fb950;border:1px solid #3fb95055}
h2.sec .n.y{background:#d2992222;color:#d29922;border:1px solid #d2992255}
h2.sec .n.p{background:#bc8cff22;color:#bc8cff;border:1px solid #bc8cff55}
p.lead{color:#b0bec5;font-size:15px;margin:8px 0 18px}
h3.sub{font-size:16px;font-weight:700;color:#e6edf3;margin:26px 0 4px;padding-left:10px;border-left:3px solid #58a6ff}
.card{background:#161b22;border:1px solid #30363d;border-radius:12px;padding:20px;margin:14px 0}
.card .ct{font-size:15px;font-weight:700;color:#e6edf3;margin-bottom:10px}
.card .cd{color:#b0bec5;font-size:14px}
.svg-frame{background:#0d1117;border:1px solid #30363d;border-radius:12px;padding:14px;margin:16px 0;overflow:hidden}
.svg-frame svg{width:100%;height:auto;display:block}
table{width:100%;border-collapse:collapse;margin:12px 0;font-size:14px}
th,td{padding:10px 12px;text-align:left;border-bottom:1px solid #21262d}
th{background:#21262d;color:#e6edf3;font-weight:600;white-space:nowrap}
td{color:#c9d1d9}
tr:last-child td{border-bottom:none}
.pos{color:#3fb950;font-weight:600}
.neg{color:#f85149;font-weight:600}
.hl{background:#1c2128}
.callout{border-left:3px solid #3fb950;background:#3fb95011;padding:14px 18px;border-radius:0 10px 10px 0;margin:18px 0;font-size:14.5px;color:#e6edf3}
.warn{border-left:3px solid #d29922;background:#d2992211;padding:14px 18px;border-radius:0 10px 10px 0;margin:18px 0;font-size:14.5px;color:#e6edf3}
ul.plan{margin:8px 0 4px 22px;color:#c9d1d9;font-size:14.5px}
ul.plan li{margin:10px 0}
footer{text-align:center;color:#8b949e;font-size:14px;margin-top:60px;padding-top:24px;border-top:1px solid #21262d}
code{background:#21262d;padding:1px 6px;border-radius:4px;font-size:13px;color:#e6edf3}
.bignum{font-size:28px;font-weight:800}
.kpi{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:18px 0}
.kpi .k{background:#161b22;border:1px solid #30363d;border-radius:12px;padding:18px;text-align:center}
.kpi .k .v{font-size:26px;font-weight:800}
.kpi .k .t{color:#8b949e;font-size:13px;margin-top:4px}
@media(max-width:860px){.kpi{grid-template-columns:1fr 1fr}}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin:16px 0}
.grid2 .u{background:#161b22;border:1px solid #30363d;border-radius:12px;padding:18px}
.grid2 .u .ut{font-size:15px;font-weight:700;color:#bc8cff;margin-bottom:6px}
.grid2 .u .ud{color:#b0bec5;font-size:13.5px;line-height:1.65}
@media(max-width:860px){.grid2{grid-template-columns:1fr}}
"""


def fmt(v):
    return f"{v:+.2f}".replace("-", "−").replace("+", "+")


def main():
    # 统计
    strong = [p for p in POINTS if p[4]]
    neg = [p for p in POINTS if p[3] < 0]
    strong_hit = sum(1 for p in strong if (p[2] > 0) == (p[3] > 0))

    # 负红利表
    neg_rows = "".join(
        f'<tr><td>{p[0]}</td><td>{p[1]}</td><td class="neg">{fmt(p[2])}</td><td class="neg">{fmt(p[3])}</td></tr>'
        for p in neg
    )

    # KPI
    kpi = f"""
    <div class="kpi">
      <div class="k"><div class="v" style="color:#58a6ff">{len(POINTS)}</div><div class="t">数据点（双 seed）</div></div>
      <div class="k"><div class="v" style="color:#3fb950">{strong_hit}/{len(strong)}</div><div class="t">强 regime 命中</div></div>
      <div class="k"><div class="v" style="color:#f85149">{len(neg)}</div><div class="t">负红利（跨 3 范式）</div></div>
      <div class="k"><div class="v" style="color:#d29922">+16.15</div><div class="t">抗噪暴涨（σ=1.0）</div></div>
    </div>
    """

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>知识蒸馏符号判据 · 第7-8周（2026-10-28）</title>
<style>{CSS}</style>
</head>
<body>
<div class="wrap">

<header>
  <h1>知识蒸馏红利符号判据 · 第 7-8 周</h1>
  <div class="sub">何仡轩 · 26 级博士生 · 2026 年 10 月 18 日 → 10 月 28 日</div>
</header>

<!-- 总览 -->
<h2 class="sec"><span class="n b">总</span>两周一句话：从「3 个点」到「判据 + 机制 + 人体信道」</h2>
<p class="lead">两周前符号判据只有 3 个数据点；现在它有了 <strong>19+ 数据点、K/R 定量机制、信噪比抗噪发现、以及真实人体信道（量化/突发/脉冲）验证</strong>——蒸馏红利符号规律在脑电低信噪比测试床上站住了。</p>
{kpi}

<!-- 独特性 -->
<h2 class="sec"><span class="n p">★</span>独特性 · 创新性 · 突出性</h2>
<p class="lead">这项工作跟已有「蒸馏何时有害」研究的四个不同，也是它站得住的理由。</p>
<div class="grid2">
  <div class="u"><div class="ut">① 首个脑电测试床</div><div class="ud">公开可检索范围内，没人用脑电这种<b>低信噪比、高维、小样本</b>的生理信号研究「知识蒸馏失效边界」。2026 脑电 × 蒸馏的工作全是「用蒸馏服务脑电」（压缩/迁移/域泛化），我们是反过来「用脑电研究蒸馏」。</div></div>
  <div class="u"><div class="ut">② task-level 符号判据</div><div class="ud"><b>sign(红利) = sign(teacher − student)</b>，一条可操作的「该不该蒸」决策规则。2026「蒸馏何时有害」爆发前沿（LLM / 图像 / NLP）全是 per-sample、per-token、机制、seed 粒度，无人给出 task-level 的二值判据。</div></div>
  <div class="u"><div class="ut">③ 定量 K/R 分解</div><div class="ud">把红利<b>数值分离</b>成 K（知识传递）+ R（正则化）两项，符号归因到 K。Yuan 2020 等只做到定性「KD ≈ label smoothing」，没人做数值分离 + 符号归因。</div></div>
  <div class="u"><div class="ut">④ 蒸馏抗噪鲁棒性（突出性）</div><div class="ud">加噪声<b>不产生负红利</b>，σ=1.0 红利暴涨 <b>+16.15</b>；真实人体信道（量化/突发/脉冲）下蒸馏依然抗噪——「蒸馏在低信噪比下比纯 hard 训练更稳」的独立证据。</div></div>
</div>

<!-- 一、符号判据 -->
<h2 class="sec"><span class="n g">一</span>核心判据：红利符号 = sign(teacher − student)</h2>
<p class="lead">蒸馏收益不是理所当然的——teacher 强于 student 时蒸馏有益，teacher 弱于 student 时蒸馏帮倒忙（负红利）。这条规律在「差距显著（|t−s|&gt;2pp）」时 <strong>{strong_hit}/{len(strong)} 命中</strong>，跨运动想象 / 肌电 / 癫痫 / P300 / 睡眠。</p>
<div class="svg-frame">{svg_scatter(POINTS)}</div>
<div class="card">
  <div class="ct">负红利 {len(neg)} 个（跨 MI / 癫痫 / 睡眠 3 范式）</div>
  <table>
    <tr><th>数据点</th><th>任务</th><th>teacher−student (pp)</th><th>30% 红利 (pp)</th></tr>
    {neg_rows}
  </table>
  <div class="cd">负红利根因 = teacher 过拟合（容量 vs 数据量不匹配 / 跨患者分布漂移），不是「teacher 弱」本身。诚实边界：5 个里只有 EEGMMIDB-4c（−2.78）显著，其余 4 个接近噪声带。</div>
</div>

<!-- 二、K/R 机制 -->
<h2 class="sec"><span class="n b">二</span>机制：K/R 定量分解（超越 Yuan 2020 定性）</h2>
<p class="lead">把蒸馏红利拆成 <strong>K（知识传递）</strong>和 <strong>R（正则化）</strong>两个分量，数值分离 + 符号归因。</p>
<div class="card">
  <table>
    <tr><th>数据</th><th>teacher 满标签</th><th>K（知识传递）</th><th>R（正则化）</th><th>总红利</th></tr>
    {"".join(f'<tr><td>{d[0]}</td><td>{d[1]:.1f}%</td><td class="{"neg" if d[2]<0 else "pos"}">{fmt(d[2])}</td><td class="{"neg" if d[3]<0 else "pos"}">{fmt(d[3])}</td><td class="{"neg" if d[2]+d[3]<0 else "pos"}">{fmt(d[2]+d[3])}</td></tr>' for d in KR)}
  </table>
  <div class="cd"><b>关键发现</b>：负红利 = K 负（teacher 软标签是噪声）+ R 负（label smoothing 在低信噪比有害）叠加；CHB-MIT 的「红利 0」是 K 负（−2.66）+ R 正（+2.66）精确抵消。这是对 Yuan 2020「弱 teacher 仍帮助」表面冲突的化解钥匙。</div>
</div>

<!-- 三、信噪比 -->
<h2 class="sec"><span class="n y">三</span>信噪比相图：加噪不产生负红利，反而暴露「蒸馏抗噪鲁棒性」</h2>
<p class="lead">最初想「加噪 → teacher 弱 → 负红利」闭环因果，结果被证伪——<strong>加噪不产生负红利</strong>，反而在 σ=1.0 红利暴涨 <strong>+16.15（双 seed）</strong>。</p>
<div class="svg-frame">{svg_snr(SNR)}</div>
<div class="card">
  <div class="cd"><b>机制</b>：加噪后 30% 硬标签「学不动」（hard 崩到 20.31% &lt; chance），但 teacher 满标签软标签抗噪（救回 44.27%）。<b>「hard 崩」是红利暴涨的必要前提</b>——CHB-MIT 二分类（易）hard 不崩，无暴涨。K/R 分解：K 恒正、R 在低信噪比窗口转负。</div>
</div>

<!-- 四、HBC -->
<h2 class="sec"><span class="n p">四</span>真实人体信道</h2>
<p class="lead">把「信噪比相图」从理想高斯 AWGN 升级到 <strong>可穿戴脑电端侧链路的真实退化</strong>——量化 / 突发（电极移动）/ 脉冲（静电放电），正是可穿戴脑电采集链路真实退化的物理来源。</p>
<div class="svg-frame">{svg_hbc(HBC)}</div>
<div class="card">
  <div class="cd">四个非高斯场景<b>红利全为正（+7.99 ~ +17.88）</b>，蒸馏在真实人体信道下依然抗噪。两个机制发现：<b>① 量化打 K 不打 R</b>（5bit 量化把知识传递 K 从 +15.97 打到 +0.52，但正则化 R 稳 +7.47）；<b>② 脉冲放大 K</b>（静电放电把 hard 崩到 31.42%，teacher 软标签相对价值最大 K=+13.89）。</div>
</div>
<div class="callout"><b>落点</b>：蒸馏不是孤立 ML 理论，它是「可穿戴人体通信系统」在<b>解码端</b>的那块拼图。蒸馏在真实信道噪声下比纯 hard 训练更稳，是端侧部署手段。</div>

<!-- 五、诚实边界 -->
<h2 class="sec"><span class="n b">五</span>诚实边界</h2>
<div class="warn">
  <ul class="plan">
    <li><b>novelty 是切入角度级</b>，不是现象级：2026「蒸馏何时有害」已成红海（LLM/图像/NLP），我们是「脑电测试床 + task-level 符号判据 + 定量 K/R 分解」三个切入角度空白。</li>
    <li><b>证据分布偏斜</b>：强 regime 命中里 12/15 来自运动想象，癫痫/EMG/睡眠各 1 个——跨任务泛化初步、不强。</li>
    <li><b>负红利幅度小</b>：5 个里 4 个接近噪声带，只有 EEGMMIDB-4c（−2.78）显著。</li>
    <li><b>单 seed 不可信</b>：蒸馏红利结论必须双 seed 均值（多次复现单 seed 负红利双 seed 后消失）。</li>
    <li><b>理论是 heuristic</b>：两因素回归 gain = 0.349·(t−s) + 0.163·蒸馏空间（R²=0.60），非严格 PAC 定理。</li>
  </ul>
</div>

<footer>以上就是这两周汇报，请老师指正。</footer>

</div>
</body>
</html>
"""
    OUT.write_text(html, encoding="utf-8")
    print(f"✓ 已生成 {OUT}（{len(html)} 字符）")


if __name__ == "__main__":
    main()
