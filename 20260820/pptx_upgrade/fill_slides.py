#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""填充 5 个新页（slide25-29）的内容：标题/副标题/3×7 表格"""
import re, sys

SLIDES = {
    "slide25.xml": {
        "title": "20260828-0901 升级总览：从二分类到 5 任务通用脑电平台",
        "subtitle": "同一块 ¥338 RV1106：76.39% 二分类 → 63.58% 四分类 → 5 任务全部实测",
        "header": ["维度", "20260820 快照", "20260901 现状"],
        "rows": [
            ["解码模型", "EEGNet（二分类）", "LIF-EEG-TCNet（四分类）"],
            ["四分类精度", "无", "63.58%（板载实测）"],
            ["任务数", "1（MI 二分类）", "5（MI/癫痫/SSVEP/睡眠）"],
            ["类脑化", "无", "LIF 脉冲，4 任务零代价"],
            ["HBC 闭环", "二分类", "四分类（McNemar 不显著）"],
            ["鲁棒性", "无", "8dB 守住 54.17%（+14.85pp）"],
        ],
    },
    "slide26.xml": {
        "title": "模型升级：EEG-TCNet 移植 + 类脑化零代价",
        "subtitle": "世界主流轻量模型移植（同协议实测，非名声）+ 类脑脉冲全部板载",
        "header": ["模型", "四分类精度", "说明"],
        "rows": [
            ["EEGNet（ELU）", "58.14%", "原基线（四分类）"],
            ["EEG-TCNet（ELU）", "64.31%", "移植 +6.17pp"],
            ["LIF-EEG-TCNet", "62.42%", "类脑化零代价，上板"],
            ["+HBC 噪声增强", "63.58%", "clean 不降反升 +1.16pp"],
            ["8dB 信道降质", "54.17%", "+14.85pp vs 无增强"],
            ["类脑化零代价", "×4 任务", "2a/2b/CHB-MIT/睡眠"],
        ],
    },
    "slide27.xml": {
        "title": "三环闭环 + 鲁棒解码：类脑 × BCI × HBC",
        "subtitle": "四分类三环闭环（芯片上实测）+ 约束驱动鲁棒 + 真实 HBC 可行性",
        "header": ["环节", "实测值", "说明"],
        "rows": [
            ["三环闭环", "62.42→62.35%", "McNemar p≈0.80 不显著"],
            ["HBC 噪声增强", "8dB +14.85pp", "约束驱动鲁棒解码"],
            ["退化拐点后移", "≥4.5dB", "退化斜率 1/2.8"],
            ["真实 HBC 硬件", "PWM 100MHz", "ADC 18kSPS / GPIO 9.98MHz"],
            ["真实实验方案", "¥30 元件", "电极+运放，待动手实测"],
            ["SNN 稀疏化", "8~13% 甜区", "脉冲流量 −43~71%"],
        ],
    },
    "slide28.xml": {
        "title": "方法筛选：5 个诚实负结果（边界定位）",
        "subtitle": "同协议实测筛选：新任务出成果，修修补补全是负结果——边界即结论",
        "header": ["方法", "实测结果", "边界结论"],
        "rows": [
            ["欧氏对齐 EA", "0 增益", "T/E 差异仅 1.45%，无偏移可对齐"],
            ["通道剪枝", "−3.6~−5.9pp", "22ch 分布式编码无冗余"],
            ["输入脉冲编码", "−24~−34pp", "delta 量化信息瓶颈"],
            ["在线校准", "−1~−35pp", "10~40 试次微调过拟合"],
            ["within-subject", "= 63.50%", "开箱即用已是最优"],
            ["筛选原则", "实测为准", "名声≠适配（EEG-TCNet 上板/其余收口）"],
        ],
    },
    "slide29.xml": {
        "title": "5 任务全景：同一套架构，全部实测",
        "subtitle": "LIF-EEG-TCNet 只改通道数/类别数，类脑化零代价 ×4 任务",
        "header": ["任务", "精度", "chance"],
        "rows": [
            ["MI 四分类（2a, 22ch）", "63.58%", "25%"],
            ["MI 二分类（2b, 3ch）", "80.5%", "50%"],
            ["癫痫检测（23ch）", "82.8% bacc", "50%"],
            ["SSVEP 拼写（6ch）", "78.6% + 82.6bit/min", "2.5%"],
            ["睡眠分期（1ch）", "85.0% eval", "20%"],
            ["芯片最终状态", "¥338 RV1106", "类脑+BCI+HBC 通用平台"],
        ],
    },
}

def set_text(xml, idx, new_text):
    """替换第 idx 个 <a:t> 的文本内容"""
    count = [0]
    def repl(m):
        count[0] += 1
        if count[0] == idx:
            return "<a:t>" + new_text + "</a:t>"
        return m.group(0)
    return re.sub(r"<a:t>[^<]*</a:t>", repl, xml)

for fname, content in SLIDES.items():
    path = "ppt/slides/" + fname
    with open(path) as f:
        xml = f.read()
    # 1. 标题 = 第 1 个 a:t
    xml = set_text(xml, 1, content["title"])
    # 2. 副标题 = 第 2 个 a:t
    xml = set_text(xml, 2, content["subtitle"])
    # 3. 表格：第 3 个 a:t 起，3×7=21 个单元格
    cells = [content["header"]] + content["rows"]  # 7 rows × 3 cols
    flat = [c for row in cells for c in row]
    assert len(flat) == 21, f"{fname}: 需要 21 单元格，实际 {len(flat)}"
    for i, txt in enumerate(flat):
        xml = set_text(xml, 3 + i, txt)
    with open(path, "w") as f:
        f.write(xml)
    print(f"OK {fname}: title={content['title'][:30]}...")
print("ALL DONE")
