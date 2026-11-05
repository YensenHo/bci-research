#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""向 HBC-BCI PicoUltra deck 插入「对标范本·博睿康」6 页。纯标准库，直接操作 OOXML。"""
import re, shutil, os, zipfile

BASE = "/Users/heyixuan/BCI/PPT/教师商业化"
UNPACKED = os.path.join(BASE, "unpacked")
SRC = os.path.join(BASE, "HBC-BCI_PicoUltra_数据集模型商业化_20260812.pptx")
OUT = os.path.join(BASE, "HBC-BCI_PicoUltra_数据集模型商业化_20260812_博睿康对标.pptx")

def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

def rpr(sz, color, bold=False, alpha=None):
    fill = f'<a:solidFill><a:srgbClr val="{color}">'
    if alpha:
        fill += f'<a:alpha val="{alpha}"/>'
    fill += '</a:srgbClr></a:solidFill>'
    b = ' b="1"' if bold else ''
    return (f'<a:rPr lang="en-US" sz="{sz}"{b} dirty="0">{fill}'
            f'<a:latin typeface="Arial" panose="020B0604020202090204" pitchFamily="34" charset="0"/>'
            f'<a:ea typeface="Arial" panose="020B0604020202090204" pitchFamily="34" charset="-122"/>'
            f'<a:cs typeface="Arial" panose="020B0604020202090204" pitchFamily="34" charset="-120"/></a:rPr>')

def para(text, sz, color, bold=False, alpha=None, bullet=False, spc_aft="400"):
    if bullet:
        ppr = (f'<a:pPr marL="342900" indent="-342900"><a:spcAft><a:spcPts val="{spc_aft}"/></a:spcAft>'
               f'<a:buSzPct val="100000"/><a:buChar char="•"/></a:pPr>')
    else:
        ppr = '<a:pPr marL="0" indent="0"><a:buNone/></a:pPr>'
    return (f'<a:p>{ppr}<a:r>{rpr(sz, color, bold, alpha)}<a:t>{esc(text)}</a:t></a:r>'
            f'<a:endParaRPr lang="en-US" sz="{sz}" dirty="0"/></a:p>')

def shape(idx, name, x, y, cx, cy, paragraphs, fill=None):
    fillxml = f'<a:solidFill><a:srgbClr val="{fill}"/></a:solidFill>' if fill else '<a:noFill/>'
    return (f'<p:sp><p:nvSpPr><p:cNvPr id="{idx}" name="{name}"/><p:cNvSpPr/><p:nvPr/></p:nvSpPr>'
            f'<p:spPr><a:xfrm><a:off x="{x}" y="{y}"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm>'
            f'<a:prstGeom prst="rect"><a:avLst/></a:prstGeom>{fillxml}</p:spPr>'
            f'<p:txBody><a:bodyPr wrap="square" rtlCol="0" anchor="ctr"/><a:lstStyle/>'
            f'{"".join(paragraphs)}</p:txBody></p:sp>')

def slide_xml(bg, shapes):
    return ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
            '<p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
            'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
            'xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">'
            f'<p:cSld><p:bg><p:bgPr><a:solidFill><a:srgbClr val="{bg}"/></a:solidFill><a:effectLst/></p:bgPr></p:bg>'
            '<p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>'
            '<p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>'
            + "".join(shapes) +
            '</p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sld>')

def content_slide(title, bullets, title_sz=2400):
    shapes = []
    shapes.append(shape(2, "Text 0", 640080, 274320, 7863840, 457200,
                        [para(title, title_sz, "E8ECF1", bold=True)]))
    shapes.append(shape(3, "Shape 1", 640080, 731520, 640080, 27432, [], fill="00D4AA"))
    shapes.append(shape(4, "Text 2", 640080, 914400, 7863840, 3840480,
                        [para(b, 1200, "B0BEC5", bullet=True) for b in bullets]))
    return slide_xml("0D1B2A", shapes)

def header_slide(big, title, subtitle):
    shapes = []
    shapes.append(shape(2, "Text 0", 731520, 1097280, 3657600, 1371600,
                        [para(big, 9600, "00D4AA", bold=True, alpha=30000)]))
    shapes.append(shape(3, "Text 1", 731520, 2194560, 7863840, 731520,
                        [para(title, 3800, "E8ECF1", bold=True)]))
    shapes.append(shape(4, "Text 2", 731520, 2926080, 7315200, 457200,
                        [para(subtitle, 1400, "B0BEC5")]))
    shapes.append(shape(5, "Shape 3", 731520, 3566160, 1097280, 36576, [], fill="00D4AA"))
    return slide_xml("07111D", shapes)

# ---- 6 页内容 ----
slides = [
    header_slide("对标", "博睿康 · 实验室创业范本",
                 "清华神经工程实验室（高上凯 / 洪波）→ 科创板 IPO · 15 年产学研之路"),
    content_slide("起源：从实验室的「卡脖子」痛点出发", [
        "高端脑电采集设备长期依赖进口、价格高、供货受限 → 国产化诉求",
        "胥红来（清华生医工博士）＋ 黄肖山，同出清华神经工程实验室，师从高上凯",
        "首席科学家洪波：国内脑机接口早期开拓者，打通临床通道",
        "2011 年 11 月注册（常州）——研发靠实验室设备 / 课题经费，公司只做产业化落地",
        "导师理念「顶天立地」：科研既顶天（学术突破）又立地（产业转化）",
    ]),
    content_slide("融资阶梯：15 年循序渐进，不是一步到位", [
        "2015 天使轮（熔拓资本）· 2018 Pre-A 轮 —— 早期金额均未披露",
        "2021 B 轮 · 过亿元（红杉独家领投）",
        "C 轮 · 数亿元（松禾资本领投）→ D / E 轮 · 红杉、百度风投等",
        "估值 35–40 亿 · 2026.6 科创板 IPO 获受理，拟募资 25 亿",
        "关键：先小额天使 → 用产品 / 营收 / 临床数据说话 → 再拿大额",
    ]),
    content_slide("产品线：先易后难", [
        "早期：非侵入式脑电采集设备 → 科研市场，现金流基本盘",
        "现在：植入式脑机 NEO → 华山医院全国第三例 / 上海首例临床试验",
        "关键：先做能赚钱的养研发，再啃最难的植入式",
        "对标：先用宋勇实验室的 EEG 端侧部署 / 采集硬件做成科研套件，承接横向项目",
    ]),
    content_slide("导师的不可替代价值", [
        "高上凯：技术源头 + 学术背书",
        "洪波：首席科学家，打通临床通道（华山医院）",
        "三层价值：国家级项目立项背书 · 医院 / 监管人脉 · 投资人面前的学术信用",
        "宋勇院长 = 你的「高上凯 / 洪波」：院系科研渠道 + 医院人脉 + 政府科创项目申报 + 投资人背书",
    ]),
    content_slide("对标你的节奏", [
        "直博前中期 → 实验室打磨技术 + 横向项目现金流 + 专利（对应 2011–2015）",
        "博士中后期 → 产品定型 + 医院验证 + 天使融资（对应 2016–2021）",
        "毕业全职 → 产业化 + 取证 + 多轮融资（对应当下 IPO 阶段）",
        "结论：绑定实验室产学研体系、循序渐进验证技术，就不会被「缺钱」限制上限",
    ]),
]

# ---- 写 slide24~29 ----
new_files = []
for i, xml in enumerate(slides):
    fn = f"ppt/slides/slide{24+i}.xml"
    with open(os.path.join(UNPACKED, fn), "w", encoding="utf-8") as f:
        f.write(xml)
    new_files.append(fn)

# ---- 更新 [Content_Types].xml ----
ct_path = os.path.join(UNPACKED, "[Content_Types].xml")
ct = open(ct_path, encoding="utf-8").read()
add_ct = "".join(f'<Override PartName="/{fn}" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>'
                 for fn in new_files)
ct = ct.replace("</Types>", add_ct + "</Types>")
open(ct_path, "w", encoding="utf-8").write(ct)

# ---- 更新 presentation.xml.rels ----
rels_path = os.path.join(UNPACKED, "ppt/_rels/presentation.xml.rels")
rels = open(rels_path, encoding="utf-8").read()
add_rels = "".join(f'<Relationship Id="rId{38+i}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide{24+i}.xml"/>'
                   for i in range(6))
rels = rels.replace("</Relationships>", add_rels + "</Relationships>")
open(rels_path, "w", encoding="utf-8").write(rels)

# ---- 更新 presentation.xml（sldIdLst 插入到 rId36 即 slide22 之前）----
pres_path = os.path.join(UNPACKED, "ppt/presentation.xml")
pres = open(pres_path, encoding="utf-8").read()
# 新 sldId 用 id 287..292，rId38..43
new_ids = "".join(f'<p:sldId id="{287+i}" r:id="rId{38+i}"/>' for i in range(6))
anchor = '<p:sldId id="285" r:id="rId36"/>'
assert anchor in pres, "anchor sldId not found"
pres = pres.replace(anchor, new_ids + anchor, 1)
open(pres_path, "w", encoding="utf-8").write(pres)

print("DONE: wrote", new_files)
print("rels + content-types + presentation.xml updated")
