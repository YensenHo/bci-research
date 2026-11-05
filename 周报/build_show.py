#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成组会展示单页 HTML（第6周 2026-10-18）—— 暗色，自包含，精简版。
只讲知识蒸馏：框架化整理 + 可复用性验证（EEGMMIDB）。"""
import base64
from pathlib import Path

BASE = Path("/Users/heyixuan/BCI")
OUT = BASE / "周报" / "组会展示_第6周_20261018.html"


def data_uri(path, mime):
    return f"data:{mime};base64," + base64.b64encode(Path(path).read_bytes()).decode()


img_verify = data_uri(BASE / "luckfox-deploy/model/verify_consistency.png", "image/png")
framework_svg = (BASE / "20260828/algorithm/知识蒸馏_框架图.svg").read_text(encoding="utf-8")


def grouped_bars(cx_title, cx_subtitle, groups, series, vmin, vmax):
    W, H = 920, 350
    top, bottom = 52, 268
    left, right = 60, 906
    plot_w = right - left
    plot_h = bottom - top
    n = len(groups)
    group_w = plot_w / n
    nseries = len(series)
    bar_w = min(46, (group_w * 0.72) / nseries)
    gap = 6

    def y(v):
        return bottom - (v - vmin) / (vmax - vmin) * plot_h

    parts = []
    for t in [t for t in range(int(vmin) - 1, int(vmax) + 2) if vmin <= t <= vmax]:
        ty = y(t)
        parts.append(f'<line x1="{left}" y1="{ty:.1f}" x2="{right}" y2="{ty:.1f}" stroke="#1e2d3f" stroke-width="1"/>')
        parts.append(f'<text x="{left - 8}" y="{ty + 4:.1f}" text-anchor="end" fill="#8b949e" font-size="11">{t}</text>')
    zy = y(0)
    parts.append(f'<line x1="{left}" y1="{zy:.1f}" x2="{right}" y2="{zy:.1f}" stroke="#6e7681" stroke-width="1.4" stroke-dasharray="5,4"/>')

    for gi, gname in enumerate(groups):
        cx = left + group_w * (gi + 0.5)
        for si, (sname, color, vals) in enumerate(series):
            v = vals[gi]
            total_w = nseries * bar_w + (nseries - 1) * gap
            x0 = cx - total_w / 2 + si * (bar_w + gap)
            ytop = y(v)
            yy0, yy1 = (ytop, zy) if v >= 0 else (zy, ytop)
            parts.append(f'<rect x="{x0:.1f}" y="{min(yy0, yy1):.1f}" width="{bar_w:.1f}" height="{abs(yy0 - yy1):.1f}" rx="3" fill="{color}" opacity="0.92"/>')
            labely = ytop - 7 if v >= 0 else ytop + 13
            parts.append(f'<text x="{x0 + bar_w / 2:.1f}" y="{labely:.1f}" text-anchor="middle" fill="#e6edf3" font-size="11" font-weight="600">{v:+.2f}</text>')
        parts.append(f'<text x="{cx:.1f}" y="{bottom + 26}" text-anchor="middle" fill="#b0bec5" font-size="13">{gname}</text>')

    lx = left
    for sname, color, _ in series:
        parts.append(f'<rect x="{lx}" y="{H - 40}" width="13" height="13" rx="3" fill="{color}"/>')
        parts.append(f'<text x="{lx + 19}" y="{H - 29}" fill="#b0bec5" font-size="12">{sname}</text>')
        lx += 19 + 13 + 6 + len(sname) * 13 + 26
    parts.append(f'<text x="{W/2:.1f}" y="26" text-anchor="middle" fill="#e6edf3" font-size="16" font-weight="700">{cx_title}</text>')
    if cx_subtitle:
        parts.append(f'<text x="{W/2:.1f}" y="45" text-anchor="middle" fill="#8b949e" font-size="11.5">{cx_subtitle}</text>')
    return f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;display:block;">' + "".join(parts) + "</svg>"


svg_mmidb = grouped_bars(
    "EEGMMIDB 正例（2类）与反例（4类）分标签预算增益",
    "2类教师强 → 全程正红利；4类教师弱 → 30% 为负、仅 10% 正则化红利为正",
    ["100% 标签", "30% 标签", "10% 标签"],
    [("EEGMMIDB 2类（正例）", "#3fb950", [3.70, 2.22, 1.48]),
     ("EEGMMIDB 4类（反例）", "#f85149", [-0.18, -2.78, 2.50])],
    -4, 5,
)

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
.rule{border:1px solid #30363d;border-radius:10px;padding:15px 16px;background:#161b22}
.rule .rt{font-size:15px;font-weight:700;color:#58a6ff;margin-bottom:5px}
.rule .rd{font-size:13.5px;color:#c9d1d9}
.rules{display:grid;grid-template-columns:1fr 1fr 1fr;gap:14px;margin:16px 0}
.rules4{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin:16px 0}
@media(max-width:860px){.rules,.rules4{grid-template-columns:1fr}}
ul.plan{margin:8px 0 4px 22px;color:#c9d1d9;font-size:14.5px}
ul.plan li{margin:10px 0}
.callout{border-left:3px solid #3fb950;background:#3fb95011;padding:14px 18px;border-radius:0 10px 10px 0;margin:18px 0;font-size:14.5px;color:#e6edf3}
footer{text-align:center;color:#8b949e;font-size:14px;margin-top:60px;padding-top:24px;border-top:1px solid #21262d}
img.chart{width:100%;border-radius:8px;border:1px solid #30363d;display:block}
.img-cap{color:#8b949e;font-size:12.5px;margin-top:8px;text-align:center}
code{background:#21262d;padding:1px 6px;border-radius:4px;font-size:13px;color:#e6edf3}
.steps{display:flex;flex-wrap:wrap;gap:8px;margin:14px 0}
.steps span{background:#21262d;border:1px solid #30363d;padding:8px 14px;border-radius:8px;font-size:13.5px;color:#e6edf3}
.steps span b{color:#58a6ff}
"""

html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>知识蒸馏工作汇报 · 第6周（2026-10-18）</title>
<style>{CSS}</style>
</head>
<body>
<div class="wrap">

<header>
  <h1>知识蒸馏工作汇报 · 第 6 周</h1>
  <div class="sub">何仡轩 · 26 级博士生 · 2026 年 10 月 18 日</div>
</header>

<!-- 总览 -->
<h2 class="sec"><span class="n b">总</span>一张图看全本周工作</h2>
<p class="lead">本周两条线：<strong>框架化整理</strong> → <strong>可复用性验证（EEGMMIDB）</strong>，并据此挖出一个下一步的顶刊方向。</p>
<div class="svg-frame">{framework_svg}</div>

<!-- 一、框架化整理 -->
<h2 class="sec"><span class="n b">一</span>知识蒸馏框架化整理</h2>
<p class="lead">把之前的蒸馏工作整理成一套<strong>可复用框架</strong>：固定协议 + 三条规律 + 端侧闭环。</p>

<h3 class="sub">1.1 蒸馏协议（固定配方）</h3>
<div class="card">
  <div class="cd">软标签 <code>softmax(z/T)</code>；T=4.0、α=0.5、100 epoch、双 seed（42/123）。对比口径：<b>唯一变量 = 有无教师软标签</b>。</div>
</div>

<h3 class="sub">1.2 三条规律（双 seed 复现）</h3>
<div class="rules">
  <div class="rule"><div class="rt">① 红利集中低标签</div><div class="rd">满标注 ≈ 0；<span class="pos">30% 标注 +5~7pp</span>；10% 标注 +3~5pp。</div></div>
  <div class="rule"><div class="rt">② 红利随任务难度增大</div><div class="rd">22ch 四分类 30% 红利 <span class="pos">+7.00pp</span> &gt; 8ch 二分类的 <span class="pos">+5.02pp</span>。</div></div>
  <div class="rule"><div class="rt">③ 类脑化零代价</div><div class="rd">LIF 热启动 56.40% 追平 relu 57.25%（−0.85pp）。</div></div>
</div>

<h3 class="sub">1.3 端侧闭环</h3>
<div class="card">
  <div class="steps">
    <span><b>①</b> 蒸馏训练</span>
    <span><b>②</b> 保存 checkpoint</span>
    <span><b>③</b> BN 折叠</span>
    <span><b>④</b> 导出 C 头</span>
    <span><b>⑤</b> 端侧 C 推理逐位验证</span>
  </div>
  <div class="cd">8ch 二分类 1650 参数 max|diff|=1.91×10⁻⁶；22ch 四分类 9344 参数 <span class="pos">50/50 样本逐位一致、0 误差</span>。LIF 模型直接走既有端侧 C 链路，无需改结构。</div>
  <img class="chart" src="{img_verify}" alt="端侧逐位一致验证" style="margin-top:14px"/>
  <div class="img-cap">端侧 C 推理 vs PyTorch 一致性验证</div>
</div>

<!-- 二、可复用性验证 -->
<h2 class="sec"><span class="n g">二</span>框架可复用性验证 · EEGMMIDB（本周核心）</h2>
<p class="lead">参数化脚本换到第三个数据集，验证红利能否跨数据集复现。</p>

<h3 class="sub">2.1 数据集</h3>
<div class="card">
  <div class="cd"><b>EEGMMIDB</b>（PhysioNet 公开脑电运动想象库，跨被试），两个任务作正反对照：<b>2 类</b>（天花板高）与<b>4 类</b>跨被试（天花板低）。</div>
</div>

<h3 class="sub">2.2 核心结果：四行对照表</h3>
<div class="card">
  <div class="ct">dark knowledge 量决定红利，规律在「教师 − 学生」这一列</div>
  <table>
    <tr><th>任务</th><th>教师精度</th><th>学生满标注</th><th>教师 − 学生</th><th>30% 标签红利</th></tr>
    <tr><td>8ch 二分类</td><td>77.4%</td><td>69.4%</td><td class="pos">+8.0</td><td class="pos">+5.02pp</td></tr>
    <tr><td>22ch 四分类</td><td>65.7%</td><td>60.6%</td><td class="pos">+5.1</td><td class="pos">+7.00pp</td></tr>
    <tr><td>EEGMMIDB 2类</td><td>70.4%</td><td>68.5%</td><td class="pos">+1.9</td><td class="pos">+2.22pp</td></tr>
    <tr class="hl"><td>EEGMMIDB 4类</td><td>36.3%</td><td>38.5%</td><td class="neg">−2.2</td><td class="neg">−2.78pp</td></tr>
  </table>
  <div class="cd">前三个任务教师强于学生 → 红利为正；EEGMMIDB 4 类教师过拟合后反而弱于学生 → 红利为负。根因：<b>教师能强多少，受任务精度天花板制约</b>。</div>
</div>

<h3 class="sub">2.3 教师容量对照</h3>
<div class="card">
  <table>
    <tr><th>教师容量</th><th>教师精度</th><th>10% 红利</th><th>30% 红利</th></tr>
    <tr><td>大教师（35K 参数）</td><td>36.3%</td><td>+2.50pp</td><td class="neg">−2.78pp</td></tr>
    <tr><td>中等教师（14.5K 参数）</td><td>37.96%</td><td class="pos">+5.00pp</td><td>−0.56pp（中性）</td></tr>
  </table>
  <div class="cd">大教师在 2160 个样本上过拟合，教师容量要匹配数据量。</div>
</div>

<h3 class="sub">2.4 框架适用边界（四条）</h3>
<div class="rules4">
  <div class="rule"><div class="rt">边界 1</div><div class="rd"><b>红利来源</b> = teacher 显著强于 student，差距越大红利越大。</div></div>
  <div class="rule"><div class="rt">边界 2</div><div class="rd"><b>teacher 能强多少</b>取决于任务精度天花板。</div></div>
  <div class="rule"><div class="rt">边界 3</div><div class="rd"><b>teacher 容量匹配数据量</b>，过拟合拖累红利。</div></div>
  <div class="rule"><div class="rt">边界 4</div><div class="rd"><b>极低标签（10%）</b>正则化红利跨任务稳定（+1.5~5pp）。</div></div>
</div>

<div class="svg-frame">{svg_mmidb}</div>

<div class="callout"><b>本周最重要结论：</b>同一套协议，teacher 强 → 红利复现；teacher 弱 → 红利消失。框架从「在某个任务上成立」推进到「知道在什么条件下成立」。</div>

<!-- 三、下一步方向 -->
<h2 class="sec"><span class="n y">三</span>下一步：蒸馏红利的符号判据（sign(teacher−student)）</h2>
<p class="lead">红利符号由「teacher − student 满标签精度」决定，根因是教师过拟合（容量 vs 数据量），可用过拟合 gap 事前预测。三数据点 + 双 seed + 6 gap 扫描实证。</p>

<div class="card">
  <div class="ct">3.1 核心规律：红利符号 = sign(teacher − student 满标签)</div>
  <div class="cd">三数据点（30% 标签，3.7× gap）：<b>EEGMMIDB 4类</b>教师弱于学生（36.3% &lt; 38.5%，−2.2pp）→ 红利 <span class="neg">−2.78pp（负）</span>；<b>BCIC IV 2a cross</b>教师强（62.15% &gt; 57.47%，+4.68pp）→ <span class="pos">+5.56pp（正）</span>；<b>within</b>教师强（+5.1pp）→ <span class="pos">+2.58pp（正）</span>。符号由 teacher−student 决定，gap（1×→14×）只调节幅度、不改变符号。</div>
</div>

<div class="card">
  <div class="ct">3.2 根因：teacher 弱 = 过拟合（容量 vs 数据量）</div>
  <div class="cd">同样是跨被试 + 35K 教师，EEGMMIDB（2160 样本少）过拟合 → 教师弱；BCIC IV 2a cross（4608 样本够）不过拟合 → 教师强。形式化：teacher − student = Δbias − Δvariance，教师容量超过「数据量 × 任务固有难度」能支撑的容量时 Δvariance 主导 → 教师弱。gap 曲线呈倒 U：负红利在 3.7×（4倍）最深（−2.78pp），更大 gap 反而减弱。</div>
</div>

<div class="card">
  <div class="ct">3.3 两级判据（可操作）</div>
  <ul class="plan">
    <li><b>后验判据（精确）</b>：蒸馏前训练 teacher + student 满标签，teacher 强 → 蒸，teacher 弱 → 跳过。</li>
    <li><b>事前 proxy（便宜）</b>：只训练 teacher，看过拟合 gap（train−val）——EEGMMIDB 4类 +34.91pp → 负，cross +24.07pp → 正，within +22.84pp → 正，单调一致。gap 大 → 教师泛化弱 → 预期负红利。</li>
  </ul>
</div>

<div class="card">
  <div class="ct">3.4 切入角度：与现有工作的差异</div>
  <ul class="plan">
    <li>2026 年「蒸馏何时有害」兴起（Strong Teacher Not Needed / Why Robust Teachers Fail / When Does KD Hurt），但都从「教师容量 / 置信度 / 数据」切入，<b>没人给出「sign(teacher−student) 定符号」的精确判据 + 过拟合根因 + 脑电三数据点实证</b>。</li>
    <li>EEG × 蒸馏现有工作全是用蒸馏解决域泛化/轻量，没人用 EEG 研究「同模态容量蒸馏的符号边界」。</li>
    <li>适用范围诚实界定：负红利显著性依赖「高维 + 低信噪比 + 小数据」（脑电典型），合成数据上红利幅度极小、符号不稳。</li>
  </ul>
</div>

<div class="card">
  <div class="ct">3.5 已完成与下一步</div>
  <ul class="plan">
    <li><b>已完成</b>：三数据点 gap 扫描（1×→14×）+ 双 seed + 过拟合 gap 事前判据 + 同数据集 cross/within 对照（消混淆）。</li>
    <li><b>下一步</b>：真实图像（MNIST/CIFAR）验证通用性；偏差-方差 + 泛化界形式化，冲顶会。</li>
  </ul>
  <div class="cd" style="margin-top:10px"><b>可投</b>：应用实证版 IEEE JBHI / TNSRE（脑电蒸馏符号判据 + 三数据点对照）；理论版 NeurIPS / ICLR（sign(teacher−student) 的泛化界刻画）。证据已约 90%。</div>
</div>

<!-- 四、下周计划 -->
<h2 class="sec"><span class="n b">四</span>下周计划</h2>
<ul class="plan">
  <li><b>EEGMMIDB 4 类</b>上尝试更小容量教师或加正则化，验证中等标签下红利能否恢复。</li>
  <li>整理蒸馏框架与可复用性验证的<b>实验脚本和结果</b>，保证数字可复现。</li>
</ul>

<footer>以上就是本周汇报，请老师指正。</footer>

</div>
</body>
</html>
"""

OUT.write_text(html, encoding="utf-8")
print(f"OK 已生成：{OUT}")
print(f"大小：{OUT.stat().st_size/1024:.0f} KB")
