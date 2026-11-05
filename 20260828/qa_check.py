#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""像素级溢出/重叠 QA：检查卡片间隙是否被文字侵入（暗色主题，文字=亮像素 lum>150）。"""
from PIL import Image
import numpy as np
import os

DPI = 150.0  # pdftoppm -r 150
W_IN = 13.3333
PX = lambda inch: int(round(inch * DPI))
DIR = "/tmp/carbon_qa"

def bright_mask(im):
    a = np.asarray(im.convert("RGB")).astype(int)
    lum = a.mean(axis=2)
    return lum > 150

# 每个 slide 的要检查的「间隙禁区」：这里不该有亮像素（文字）
# 格式：(slide序号, [ (x0_in, x1_in, y0_in, y1_in, 说明), ... ])
GAPS = {
    3: [
        (4.35, 4.79, 1.8, 4.35, "支柱卡1-2间隙"),
        (8.54, 8.98, 1.8, 4.35, "支柱卡2-3间隙"),
    ],
    4: [
        (7.80, 8.00, 1.75, 4.95, "表格与右侧大图间隙"),
        (8.00, 12.73, 5.30, 5.95, "右侧图下方到部署条之间"),
    ],
    5: [
        (7.95, 8.15, 2.20, 3.12, "路由卡3与说明文字间隙"),
        (6.55, 6.78, 3.35, 6.67, "四要点左右列间隙"),
    ],
    6: [
        (7.95, 8.15, 1.72, 6.72, "左列与右列实物占位间隙"),
    ],
    7: [
        (4.45, 4.76, 3.45, 5.75, "三列卡1-2间隙"),
        (8.61, 8.92, 3.45, 5.75, "三列卡2-3间隙"),
    ],
    8: [
        (4.45, 4.76, 1.80, 4.80, "场景卡1-2间隙"),
        (8.61, 8.92, 1.80, 4.80, "场景卡2-3间隙"),
        (8.00, 8.20, 4.98, 6.70, "佩戴占位与右侧要点间隙"),
    ],
    9: [
        (4.45, 4.76, 3.75, 5.70, "三卡1-2间隙"),
        (8.61, 8.92, 3.75, 5.70, "三卡2-3间隙"),
    ],
    10: [
        (4.45, 4.76, 1.80, 4.25, "步骤卡1-2间隙"),
        (8.61, 8.92, 1.80, 4.25, "步骤卡2-3间隙"),
        (5.00, 5.25, 5.35, 6.70, "K200占位与金句间隙"),
    ],
}

# 全局边距：文字不应越界（排除 PDF 导出边缘 1px 白线伪影）
MARGIN = dict(l=0.55, r=12.90, t=0.35, b=7.42)

problems = 0
for slide in range(1, 11):
    p = f"{DIR}/slide-{slide:02d}.jpg"
    if not os.path.exists(p):
        print(f"[slide {slide}] 缺少渲染图，跳过")
        continue
    im = Image.open(p)
    w_px, h_px = im.size
    mask = bright_mask(im)

    # 全局边距检查
    L, R, T, B = [PX(v) for v in (MARGIN["l"], MARGIN["r"], MARGIN["t"], MARGIN["b"])]
    out_left = mask[:, :L].sum()
    out_right = mask[:, R:].sum()
    out_top = mask[:T, :].sum()
    out_bottom = mask[B:, :].sum()
    flags = []
    if out_left: flags.append(f"左越界 {int(out_left)}px")
    if out_right: flags.append(f"右越界 {int(out_right)}px")
    if out_top: flags.append(f"上越界 {int(out_top)}px")
    if out_bottom: flags.append(f"下越界 {int(out_bottom)}px")

    # 间隙禁区检查
    for (x0, x1, y0, y1, desc) in GAPS.get(slide, []):
        cnt = int(mask[PX(y0):PX(y1), PX(x0):PX(x1)].sum())
        if cnt:
            flags.append(f"间隙「{desc}」有 {cnt} 亮像素")

    if flags:
        problems += 1
        print(f"[slide {slide}] ⚠ " + " | ".join(flags))
    else:
        print(f"[slide {slide}] ✓ 无溢出/越界")

print(f"\n共 {problems} 页存在可疑溢出" if problems else "\n全部 10 页通过像素级检查")
