// 碳硅协同 · 检测解码系统 —— 10 页可编辑 PPT 构建脚本
// 运行：NODE_PATH=/opt/homebrew/lib/node_modules node build_carbon_silicon.js
const pptxgen = require("pptxgenjs");
const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.33" x 7.5"

const FONT = "Arial";
const IMG = "/Users/heyixuan/BCI/20260828/board_demo/demo_ppt.png";

// ── Deep Neural 暗色主题（已验证）──
const C = {
  bg: "0D1B2A", dark: "07111D", card: "142133",
  accent: "00D4AA", accent2: "6C5CE7", amber: "F0A500",
  coral: "FF6B6B", text: "E8ECF1", body: "B0BEC5",
  muted: "5A6E82", white: "FFFFFF", green: "00C853",
  red: "FF5252", blue: "58A6FF",
};
// 三大支柱语义色
const BCI = C.blue;      // 脑机接口
const SNN = C.accent2;   // 类脑计算
const HBC = C.accent;    // 人体通信
const MOL = C.amber;     // 分子通讯
const APP = C.green;     // 应用/意义

const TOTAL = 10;

// ── 通用组件 ──
function header(s, kickerText, kickerColor, titleText) {
  s.addText(kickerText, { x: 0.6, y: 0.42, w: 8, h: 0.32, fontSize: 12,
    fontFace: FONT, color: kickerColor, bold: true, charSpacing: 2 });
  s.addText(titleText, { x: 0.6, y: 0.74, w: 12.13, h: 0.72, fontSize: 29,
    fontFace: FONT, color: C.text, bold: true });
  s.addShape(pres.ShapeType.rect, { x: 0.62, y: 1.52, w: 0.95, h: 0.035,
    fill: { color: kickerColor } });
}
function footer(s, n) {
  s.addText("碳硅协同 · 检测解码系统", { x: 0.6, y: 7.05, w: 6, h: 0.3,
    fontSize: 9, fontFace: FONT, color: C.muted });
  s.addText(String(n).padStart(2, "0") + " / " + TOTAL, { x: 11.93, y: 7.05, w: 0.8, h: 0.3,
    fontSize: 9, fontFace: FONT, color: C.muted, align: "right" });
}
function card(s, x, y, w, h, fill, radius) {
  s.addShape(pres.ShapeType.roundRect, { x, y, w, h, fill: { color: fill || C.card },
    rectRadius: radius || 0.07, line: { color: "1E2D3F", width: 0.75 } });
}
// 图片占位框：虚线边框 + 说明文字，用户拿到后直接把图丢进来替换
function imgPlaceholder(s, x, y, w, h, label, note, color) {
  const c = color || "4A5A70";
  s.addShape(pres.ShapeType.roundRect, { x, y, w, h, fill: { color: "0B1520" },
    rectRadius: 0.05, line: { color: c, width: 1.2, dashType: "dash" } });
  const midY = y + h / 2;
  s.addText("＋ 待插入图片", { x, y: y + Math.max(0.1, h * 0.08), w, h: 0.3,
    fontSize: 9.5, fontFace: FONT, color: c, bold: true, align: "center" });
  s.addText(label, { x: x + 0.12, y: midY - 0.22, w: w - 0.24, h: 0.44,
    fontSize: 12.5, fontFace: FONT, color: C.text, bold: true, align: "center", valign: "middle" });
  if (note) s.addText(note, { x: x + 0.14, y: midY + 0.24, w: w - 0.28, h: h / 2 - 0.34,
    fontSize: 8.5, fontFace: FONT, color: C.muted, align: "center" });
}
// 支柱抽象图标：在彩色圆圈内画深色几何符号（x,y 为圆圈左上角，0.62×0.62）
function pillarIcon(s, x, y, type) {
  const D = C.dark;
  if (type === "bci") {
    // 脑电波形：四个竖条，中间高两边低
    [0.16, 0.3, 0.2, 0.28].forEach((bh, i) => {
      s.addShape(pres.ShapeType.rect, { x: x + 0.08 + i * 0.12, y: y + 0.31 - bh / 2,
        w: 0.055, h: bh, fill: { color: D } });
    });
  } else if (type === "snn") {
    // 脉冲序列：一条基线 + 两个放电尖
    s.addShape(pres.ShapeType.rect, { x: x + 0.06, y: y + 0.40, w: 0.50, h: 0.022, fill: { color: D } });
    s.addShape(pres.ShapeType.rect, { x: x + 0.16, y: y + 0.12, w: 0.04, h: 0.28, fill: { color: D } });
    s.addShape(pres.ShapeType.rect, { x: x + 0.36, y: y + 0.16, w: 0.04, h: 0.24, fill: { color: D } });
  } else if (type === "hbc") {
    // 信号扩散：中心点 + 四周环绕点
    s.addShape(pres.ShapeType.ellipse, { x: x + 0.27, y: y + 0.27, w: 0.08, h: 0.08, fill: { color: D } });
    s.addShape(pres.ShapeType.ellipse, { x: x + 0.18, y: y + 0.14, w: 0.05, h: 0.05, fill: { color: D } });
    s.addShape(pres.ShapeType.ellipse, { x: x + 0.39, y: y + 0.14, w: 0.05, h: 0.05, fill: { color: D } });
    s.addShape(pres.ShapeType.ellipse, { x: x + 0.18, y: y + 0.43, w: 0.05, h: 0.05, fill: { color: D } });
    s.addShape(pres.ShapeType.ellipse, { x: x + 0.39, y: y + 0.43, w: 0.05, h: 0.05, fill: { color: D } });
  }
}

// ════════════════════════════════════════════════
// Slide 1 · 封面
// ════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: C.dark };
  // 三支柱连接点（碳硅协同）
  const dots = [
    { x: 5.06, c: BCI, label: "脑机接口 · 读脑" },
    { x: 6.52, c: HBC, label: "人体通信 · 传身" },
    { x: 7.98, c: SNN, label: "类脑计算 · 仿脑" },
  ];
  s.addShape(pres.ShapeType.rect, { x: 5.36, y: 1.615, w: 2.78, h: 0.025, fill: { color: "2A3B4E" } });
  dots.forEach((d) => {
    s.addShape(pres.ShapeType.ellipse, { x: d.x, y: 1.48, w: 0.3, h: 0.3, fill: { color: d.c } });
  });
  s.addText("碳硅协同", { x: 1, y: 1.95, w: 11.33, h: 1.0, fontSize: 56,
    fontFace: FONT, color: C.white, bold: true, align: "center" });
  s.addText("脑机接口 × 人体通信 × 类脑计算", { x: 1, y: 3.0, w: 11.33, h: 0.5,
    fontSize: 20, fontFace: FONT, color: C.accent, bold: true, align: "center", charSpacing: 1 });
  // 三支柱彩色标签
  dots.forEach((d, i) => {
    const w = 2.7, x = 2.315 + i * 3.0;
    s.addShape(pres.ShapeType.roundRect, { x, y: 3.72, w, h: 0.52, fill: { color: C.card },
      rectRadius: 0.26, line: { color: d.c, width: 1 } });
    s.addText(d.label, { x, y: 3.72, w, h: 0.52, fontSize: 12.5, fontFace: FONT,
      color: d.c, bold: true, align: "center", valign: "middle" });
  });
  s.addText("一个「戴在身上」的脑电解码检测系统 —— 生物脑与硅基脑的无缝连接", {
    x: 1, y: 4.6, w: 11.33, h: 0.45, fontSize: 15, fontFace: FONT, color: C.body, align: "center" });
  s.addShape(pres.ShapeType.rect, { x: 5.92, y: 5.35, w: 1.5, h: 0.04, fill: { color: C.accent } });
  s.addText("何仡轩 · 宋勇老师课题组 · 2026.09", { x: 1, y: 5.6, w: 11.33, h: 0.4,
    fontSize: 12, fontFace: FONT, color: C.muted, align: "center" });
}

// ════════════════════════════════════════════════
// Slide 2 · 愿景
// ════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: C.bg };
  header(s, "愿景  VISION", C.accent, "让脑机接口走下病床，走进生活");
  s.addText("碳基大脑 × 硅基系统，无缝协作", { x: 0.6, y: 1.75, w: 12.13, h: 0.6,
    fontSize: 30, fontFace: FONT, color: C.white, bold: true, align: "center" });
  s.addText("把「读脑」从实验室搬到身上 —— 技术在日常里无声地工作，而不是躺在仪器台上", {
    x: 1.2, y: 2.42, w: 10.93, h: 0.4, fontSize: 14, fontFace: FONT, color: C.body, align: "center" });

  const boxes = [
    { t: "生物脑", sub: "碳基 · 大脑意图", c: C.amber },
    { t: "脑电采集", sub: "电极读取", c: C.body },
    { t: "硅基芯片", sub: "类脑解码", c: C.accent },
    { t: "人体通信", sub: "皮肤回传", c: SNN },
    { t: "设备动作", sub: "身边设备", c: C.green },
  ];
  const bw = 2.02, gap = 0.53, bx = 0.6, by = 3.3, bh = 1.15;
  boxes.forEach((b, i) => {
    const x = bx + i * (bw + gap);
    card(s, x, by, bw, bh, C.card);
    s.addText(b.t, { x, y: by + 0.18, w: bw, h: 0.42, fontSize: 17,
      fontFace: FONT, color: b.c, bold: true, align: "center" });
    s.addText(b.sub, { x, y: by + 0.62, w: bw, h: 0.34, fontSize: 11,
      fontFace: FONT, color: C.muted, align: "center" });
    if (i < boxes.length - 1) {
      s.addText("→", { x: x + bw + 0.04, y: by + 0.22, w: gap - 0.08, h: 0.5,
        fontSize: 20, fontFace: FONT, color: C.accent, align: "center", bold: true });
    }
  });
  s.addText("闭环：脑电 → 芯片解码 → 人体通信回传 → 设备动作，全程本地、无感、可穿戴", {
    x: 0.6, y: 4.72, w: 12.13, h: 0.44, fontSize: 13, fontFace: FONT, color: C.accent,
    align: "center", bold: true });
  // 闭环回传视觉条（从右回到左）
  s.addShape(pres.ShapeType.roundRect, { x: 0.6, y: 5.14, w: 12.13, h: 0.28,
    fill: { color: "0E1A2A" }, rectRadius: 0.14 });
  s.addText("← 反馈闭环：设备动作 → 体感反馈 → 大脑再调整", { x: 0.6, y: 5.14, w: 12.13, h: 0.28,
    fontSize: 10.5, fontFace: FONT, color: C.muted, align: "center", valign: "middle" });

  const kw = ["戴在身上", "日常连续", "无感陪伴"];
  kw.forEach((k, i) => {
    const w = 2.9, x = 2.22 + i * (w + 0.45);
    card(s, x, 5.5, w, 0.72, "0E1A2A");
    s.addText(k, { x, y: 5.5, w, h: 0.72, fontSize: 16, fontFace: FONT,
      color: C.text, align: "center", valign: "middle", bold: true });
  });
  footer(s, 2);
}

// ════════════════════════════════════════════════
// Slide 3 · 三大支柱总览
// ════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: C.bg };
  header(s, "系统架构  SYSTEM", C.accent, "三大技术支柱，共享同一个物理本质");

  const pillars = [
    { icon: "bci", name: "脑机接口", en: "BCI · 读脑", c: BCI,
      desc: "电极采集脑电，芯片实时读懂意图", res: "四分类 63.58%" },
    { icon: "snn", name: "类脑计算", en: "SNN · 仿脑", c: SNN,
      desc: "脉冲神经网络，像大脑一样稀疏地算", res: "零代价追平 ANN" },
    { icon: "hbc", name: "人体通信", en: "HBC · 传身", c: HBC,
      desc: "皮肤当导线，数据不出身体", res: "闭环零损失" },
  ];
  const pw = 3.75, gap2 = 0.44, px = 0.6, py = 1.8, ph = 2.55;
  pillars.forEach((p, i) => {
    const x = px + i * (pw + gap2);
    card(s, x, py, pw, ph);
    s.addShape(pres.ShapeType.ellipse, { x: x + 0.28, y: py + 0.26, w: 0.62, h: 0.62,
      fill: { color: p.c } });
    pillarIcon(s, x + 0.28, py + 0.26, p.icon);
    s.addText(p.name, { x: x + 1.02, y: py + 0.22, w: pw - 1.15, h: 0.45, fontSize: 19,
      fontFace: FONT, color: C.text, bold: true });
    s.addText(p.en, { x: x + 1.02, y: py + 0.68, w: pw - 1.15, h: 0.32, fontSize: 11,
      fontFace: FONT, color: p.c, bold: true });
    s.addText(p.desc, { x: x + 0.28, y: py + 1.15, w: pw - 0.56, h: 0.7, fontSize: 13,
      fontFace: FONT, color: C.body });
    s.addText(p.res, { x: x + 0.28, y: py + 1.95, w: pw - 0.56, h: 0.4, fontSize: 15,
      fontFace: FONT, color: p.c, bold: true });
  });

  card(s, 0.6, 4.7, 12.13, 0.82, "0E1A2A");
  s.addText("共享「稀疏」这一物理本质", { x: 0.85, y: 4.7, w: 3.3, h: 0.82,
    fontSize: 15, fontFace: FONT, color: C.accent, bold: true, valign: "middle" });
  s.addText("大脑信号稀疏  ·  脉冲计算稀疏  ·  人体传输载荷稀疏", { x: 4.15, y: 4.7,
    w: 8.4, h: 0.82, fontSize: 13, fontFace: FONT, color: C.body, valign: "middle" });

  card(s, 0.6, 5.75, 12.13, 0.78, C.card);
  s.addShape(pres.ShapeType.roundRect, { x: 0.82, y: 5.94, w: 1.5, h: 0.4,
    fill: { color: MOL }, rectRadius: 0.2 });
  s.addText("微观延伸", { x: 0.82, y: 5.94, w: 1.5, h: 0.4, fontSize: 11,
    fontFace: FONT, color: C.dark, bold: true, align: "center", valign: "middle" });
  s.addText("分子通讯：细胞之间的信号分子扩散，作为人体通信的细胞级延伸，与疾病诊疗密切相关",
    { x: 2.5, y: 5.75, w: 10.0, h: 0.78, fontSize: 13, fontFace: FONT, color: C.body, valign: "middle" });
  footer(s, 3);
}

// ════════════════════════════════════════════════
// Slide 4 · 脑机接口：芯片实测结果
// ════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: C.bg };
  header(s, "01 · 脑机接口  BCI", BCI, "一块 ¥338 芯片，读懂七种脑电任务");

  // 左侧表格
  const rows = [
    ["任务（公开数据集）", "精度", "说明"],
    ["运动想象 · 四分类", "63.58%", "基准 25%（BCIC 2a）"],
    ["运动想象 · 二分类", "80%", "左右手（BCIC 2b）"],
    ["跨被试 · 二分类", "71%", "30 人（EEGMMIDB）"],
    ["癫痫检测", "82.8%", "漏报 4.3%（CHB-MIT）"],
    ["SSVEP 打字", "78.6%", "82.6 bit/min"],
    ["睡眠分期", "85%", "五期（平衡 73.5%）"],
    ["P300 打字", "84%", "字符级"],
  ];
  const colW = [2.8, 1.3, 3.1];
  const colX = [0.6, 0.6 + colW[0], 0.6 + colW[0] + colW[1]];
  rows.forEach((row, ri) => {
    const y = 1.75 + ri * 0.52;
    row.forEach((cell, ci) => {
      const x = colX[ci];
      const isH = ri === 0;
      if (isH) {
        s.addShape(pres.ShapeType.rect, { x, y, w: colW[ci], h: 0.52, fill: { color: BCI } });
        s.addText(cell, { x: x + 0.1, y, w: colW[ci] - 0.2, h: 0.52, fontSize: 12,
          fontFace: FONT, color: C.dark, bold: true, align: "center", valign: "middle" });
      } else {
        if (ri % 2 === 0) s.addShape(pres.ShapeType.rect, { x, y, w: colW[ci], h: 0.52, fill: { color: "101F31" } });
        s.addText(cell, { x: x + 0.1, y, w: colW[ci] - 0.2, h: 0.52, fontSize: 11.5,
          fontFace: FONT, color: ci === 1 ? C.accent : C.body,
          bold: ci === 1, align: ci === 1 ? "center" : "left", valign: "middle" });
      }
    });
  });

  // 右侧：板载演示图（3:2 放大）+ 演示视频 16:9（缩小居中）
  s.addImage({ path: IMG, x: 8.27, y: 1.75, w: 4.2, h: 2.8 });
  // 视频占位 16:9
  s.addShape(pres.ShapeType.roundRect, { x: 8.77, y: 4.65, w: 3.2, h: 1.8,
    fill: { color: "0B1520" }, rectRadius: 0.05, line: { color: "4A5A70", width: 1.2, dashType: "dash" } });
  s.addText("▶", { x: 8.77, y: 4.8, w: 3.2, h: 0.6, fontSize: 26, fontFace: FONT,
    color: C.accent, align: "center", valign: "middle", bold: true });
  s.addText("演示视频（16:9 · 待插入）", { x: 8.77, y: 5.45, w: 3.2, h: 0.35, fontSize: 10.5,
    fontFace: FONT, color: C.text, align: "center", bold: true });
  s.addText("四分类闭环 · 直连 vs 经 HBC 回传", { x: 8.77, y: 5.82, w: 3.2, h: 0.3, fontSize: 8.5,
    fontFace: FONT, color: C.muted, align: "center" });

  // 底部：部署一致性（精简单行）
  card(s, 0.6, 6.58, 12.13, 0.42, "0E1A2A");
  s.addText("部署可信度：电脑端 → 芯片端一致率 100%（往返 / C / PyTorch 三方） · CPU 前向 50ms（40× 余量）", {
    x: 0.85, y: 6.58, w: 11.6, h: 0.42, fontSize: 11.5, fontFace: FONT, color: C.body, valign: "middle" });
  footer(s, 4);
}

// ════════════════════════════════════════════════
// Slide 5 · 类脑计算
// ════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: C.bg };
  header(s, "02 · 类脑计算  SNN", SNN, "零代价「仿脑」，绕开芯片的坑");

  s.addText("模型演进路线", { x: 0.6, y: 1.78, w: 3.0, h: 0.4, fontSize: 15,
    fontFace: FONT, color: C.text, bold: true });
  const route = [
    { t: "EEGNet", v: "58%", c: C.muted },
    { t: "EEG-TCNet", v: "64%", c: SNN },
    { t: "+ LIF 脉冲", v: "62.42%", c: SNN },
  ];
  let rx = 0.6;
  route.forEach((r, i) => {
    const w = 2.15;
    card(s, rx, 2.2, w, 0.92);
    s.addText(r.t, { x: rx, y: 2.32, w, h: 0.38, fontSize: 14, fontFace: FONT,
      color: C.text, bold: true, align: "center" });
    s.addText(r.v, { x: rx, y: 2.68, w, h: 0.38, fontSize: 16, fontFace: FONT,
      color: r.c, bold: true, align: "center" });
    if (i < route.length - 1) {
      s.addText("→", { x: rx + w, y: 2.4, w: 0.45, h: 0.5, fontSize: 18, fontFace: FONT,
        color: C.muted, align: "center", bold: true });
    }
    rx += w + 0.45;
  });
  s.addText("四分类：58% → 64% → 62.42%（脉冲化基本持平）", {
    x: 8.15, y: 2.4, w: 4.58, h: 0.5, fontSize: 12, fontFace: FONT, color: C.body, valign: "middle" });

  const pts = [
    { t: "零代价类脑化", c: SNN, d: "LIF 脉冲解码追平 ANN：二分类 77.7%、四分类 62.42%，仿脑不付精度代价" },
    { t: "混合架构 · 定案", c: SNN, d: "连续前端 + LIF 中间层：发放率压到 8~13%，但事件驱动红利被 φ1 密集前端稀释（仅省 2.9%）" },
    { t: "绕开 NPU 量化 bug", c: C.green, d: "INT8 量化板载崩坏（98.46%→42~58%），脉冲 0/1 天然免疫，属工具链问题" },
    { t: "脉冲化边界（诚实）", c: C.amber, d: "真脉冲三路线全负：脉冲输入 −24~34pp · 事件驱动传感器信息仅 16% · STDP≈随机" },
  ];
  const cw = 5.95, chh = 1.42, gapY = 0.16;
  pts.forEach((p, i) => {
    const col = i % 2, row = Math.floor(i / 2);
    const x = 0.6 + col * (cw + 0.23), y = 3.32 + row * (chh + gapY);
    card(s, x, y, cw, chh);
    s.addShape(pres.ShapeType.rect, { x: x, y, w: 0.07, h: chh, fill: { color: p.c } });
    s.addText(p.t, { x: x + 0.25, y: y + 0.12, w: cw - 0.4, h: 0.34, fontSize: 14.5,
      fontFace: FONT, color: p.c, bold: true });
    s.addText(p.d, { x: x + 0.25, y: y + 0.5, w: cw - 0.5, h: 0.85, fontSize: 11,
      fontFace: FONT, color: C.body });
  });

  card(s, 0.6, 6.48, 12.13, 0.5, "0E1A2A");
  s.addText("定案：连续前端 + LIF 中间层（混合架构）是 RV1106 上类脑化的最优解；真脉冲化三路线均为负结果，类脑能效红利待神经形态芯片（灵汐 KA200）兑现",
    { x: 0.85, y: 6.48, w: 11.7, h: 0.5, fontSize: 11.5, fontFace: FONT, color: C.accent, valign: "middle" });
  footer(s, 5);
}

// ════════════════════════════════════════════════
// Slide 6 · 人体通信
// ════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: C.bg };
  header(s, "03 · 人体通信  HBC", HBC, "皮肤当导线，数据不出身体");

  // 左列：两个结果卡
  const stats = [
    { num: "≈ 0", label: "闭环精度损失", d: "直连 vs 经 HBC 回传，McNemar 检验无显著差异", c: HBC },
    { num: "54.17%", label: "8dB 信道降质下", d: "鲁棒解码，比无噪声增强 +14.85pp", c: C.green },
  ];
  stats.forEach((st, i) => {
    const y = 1.72 + i * 1.2;
    card(s, 0.6, y, 7.35, 1.1);
    s.addText(st.num, { x: 0.9, y: y + 0.12, w: 2.1, h: 0.85, fontSize: 30,
      fontFace: FONT, color: st.c, bold: true, valign: "middle" });
    s.addText(st.label, { x: 3.05, y: y + 0.12, w: 4.7, h: 0.4, fontSize: 14,
      fontFace: FONT, color: C.text, bold: true });
    s.addText(st.d, { x: 3.05, y: y + 0.52, w: 4.7, h: 0.5, fontSize: 10.5,
      fontFace: FONT, color: C.body });
  });

  // 信道建模升级
  card(s, 0.6, 4.12, 7.35, 1.55);
  s.addText("信道建模升级", { x: 0.85, y: 4.22, w: 2.2, h: 0.4, fontSize: 14,
    fontFace: FONT, color: HBC, bold: true });
  s.addText("现状：高斯噪声简化模拟", { x: 0.85, y: 4.66, w: 6.9, h: 0.35, fontSize: 11,
    fontFace: FONT, color: C.muted });
  s.addText("下一步：", { x: 0.85, y: 5.04, w: 1.2, h: 0.35, fontSize: 11,
    fontFace: FONT, color: C.accent, bold: true });
  s.addText("电路传递函数法（基尔霍夫定律）  ·  有限元仿真法（皮肤/脂肪/肌肉分层）",
    { x: 2.0, y: 5.04, w: 5.8, h: 0.35, fontSize: 11, fontFace: FONT, color: C.body });

  // 三大优势（迷你卡）
  const adv = [
    { t: "隐私", d: "信号不出身体", c: HBC },
    { t: "低功耗", d: "不发射无线电", c: HBC },
    { t: "安全", d: "皮肤接触才通", c: HBC },
  ];
  adv.forEach((a, i) => {
    const x = 0.6 + i * 2.5;
    card(s, x, 5.85, 2.35, 0.9);
    s.addText(a.t, { x: x + 0.16, y: 5.9, w: 2.05, h: 0.34, fontSize: 13,
      fontFace: FONT, color: a.c, bold: true });
    s.addText(a.d, { x: x + 0.16, y: 6.24, w: 2.05, h: 0.34, fontSize: 10,
      fontFace: FONT, color: C.body });
  });

  // 右列：实物硬件（开发板 1:1 + 电极/收发板 1:1 并排）
  imgPlaceholder(s, 8.15, 1.72, 2.2, 2.2, "开发板实物照",
    "Luckfox Pico Ultra / RV1106（1:1）", BCI);
  imgPlaceholder(s, 10.53, 1.72, 2.2, 2.2, "电极 + 收发板",
    "皮肤导线收发两端（1:1）", HBC);
  // 下方：HBC 实测说明
  card(s, 8.15, 4.12, 4.58, 2.55, "0E1A2A");
  s.addText("真实人体信道实测", { x: 8.4, y: 4.32, w: 4.0, h: 0.4, fontSize: 13,
    fontFace: FONT, color: HBC, bold: true });
  s.addText("发射贴片 → 人体信道 → 接收贴片\n实测损耗回填信道模型，补全物理层短板", { x: 8.4, y: 4.78,
    w: 4.1, h: 1.7, fontSize: 11, fontFace: FONT, color: C.body });
  footer(s, 6);
}

// ════════════════════════════════════════════════
// Slide 7 · 分子通讯
// ════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: C.bg };
  header(s, "延伸 · 分子通讯  MOLECULAR", MOL, "宏观人体通信的细胞级延伸");

  card(s, 0.6, 1.75, 12.13, 1.5);
  s.addText("从「皮肤导线的电信号」到「细胞之间的分子信号」", { x: 0.9, y: 1.92, w: 7.4, h: 0.5,
    fontSize: 18, fontFace: FONT, color: C.white, bold: true });
  s.addText("人体通信解决「皮肤表面的电信号传输」，分子通讯解决「细胞层面的信号分子扩散」—— 两者是同一套人体信息网络的两个尺度",
    { x: 0.9, y: 2.46, w: 7.4, h: 0.7, fontSize: 12.5, fontFace: FONT, color: C.body });

  // 右侧：细胞分子信号示意图（形状绘制，可编辑）
  [[8.72, 2.14], [9.0, 2.44], [8.72, 2.74]].forEach(([mx, my]) => {
    s.addShape(pres.ShapeType.ellipse, { x: mx, y: my, w: 0.16, h: 0.16, fill: { color: MOL } });
  });
  s.addText("→", { x: 9.42, y: 2.22, w: 0.8, h: 0.5, fontSize: 18, fontFace: FONT,
    color: MOL, bold: true, align: "center", valign: "middle" });
  s.addShape(pres.ShapeType.ellipse, { x: 11.1, y: 2.0, w: 0.95, h: 0.95, fill: { color: "0B1520" },
    line: { color: MOL, width: 1.5 } });
  s.addShape(pres.ShapeType.ellipse, { x: 11.42, y: 2.32, w: 0.32, h: 0.32, fill: { color: SNN } });
  [2.26, 2.58].forEach((yy) => {
    s.addShape(pres.ShapeType.ellipse, { x: 10.98, y: yy, w: 0.14, h: 0.14, fill: { color: MOL } });
  });
  s.addText("信号分子扩散 → 受体结合", { x: 8.6, y: 3.06, w: 3.9, h: 0.28, fontSize: 9,
    fontFace: FONT, color: C.muted, align: "center" });

  const cols = [
    { t: "是什么", d: "细胞通过信号分子（神经递质、细胞因子等）的扩散与受体结合，完成信息传递", c: MOL },
    { t: "为什么相关", d: "与疾病诊疗密切相关：炎症信号通路、肿瘤微环境、神经退行性疾病的分子机制", c: C.coral },
    { t: "怎么纳入", d: "作为 HBC 宏观体系的微观延伸，与脑机接口「读脑」形成从分子到行为的完整链路", c: C.accent },
  ];
  cols.forEach((col, i) => {
    const x = 0.6 + i * 4.16;
    card(s, x, 3.45, 3.85, 2.3);
    s.addShape(pres.ShapeType.ellipse, { x: x + 0.25, y: 3.68, w: 0.5, h: 0.5, fill: { color: col.c } });
    s.addText(col.t, { x: x + 0.9, y: 3.66, w: 2.7, h: 0.5, fontSize: 16,
      fontFace: FONT, color: col.c, bold: true, valign: "middle" });
    s.addText(col.d, { x: x + 0.25, y: 4.35, w: 3.35, h: 1.25, fontSize: 11.5,
      fontFace: FONT, color: C.body });
  });

  card(s, 0.6, 6.0, 12.13, 0.85, "0E1A2A");
  s.addText("研究定位", { x: 0.85, y: 6.0, w: 2.0, h: 0.85, fontSize: 14,
    fontFace: FONT, color: MOL, bold: true, valign: "middle" });
  s.addText("导师建议的调研方向：先厘清细胞层面分子通讯的机制与建模方法，再判断它作为宏观人体通信「微观延伸」的可行性",
    { x: 2.9, y: 6.0, w: 9.6, h: 0.85, fontSize: 12.5, fontFace: FONT, color: C.body, valign: "middle" });
  footer(s, 7);
}

// ════════════════════════════════════════════════
// Slide 8 · 应用场景
// ════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: C.bg };
  header(s, "应用  SCENARIOS", APP, "落地场景：融入生活与工作");

  const scenes = [
    { t: "疲劳监测 · 重点", c: C.amber,
      d: "高负荷人群（核操作/交警/抢险救灾）；疲劳前兆预警，提醒休息换班（分钟级，待实测标定）" },
    { t: "老年健康 · 主动养老", c: APP,
      d: "24h 连续监测，居家健康管理；长期趋势预警，指标恶化时提醒就医（天/周级）" },
    { t: "家庭医疗 · 辅助", c: BCI,
      d: "居家康复与慢病管理，脑电解码能力下沉到家庭场景，降低对医院依赖" },
  ];
  scenes.forEach((sc, i) => {
    const y = 1.75 + i * 1.5;
    card(s, 0.6, y, 12.13, 1.34);
    s.addShape(pres.ShapeType.ellipse, { x: 0.98, y: y + 0.42, w: 0.5, h: 0.5, fill: { color: sc.c } });
    s.addText(sc.t, { x: 1.72, y: y + 0.22, w: 4.6, h: 0.44, fontSize: 17,
      fontFace: FONT, color: sc.c, bold: true });
    s.addText(sc.d, { x: 1.72, y: y + 0.74, w: 10.6, h: 0.5, fontSize: 12.5,
      fontFace: FONT, color: C.body });
  });

  // 三大硬指标（底部横向条）
  const strips = ["全程离线 · 不联网", "隐私 · 信号不出身体", "成本 · ¥338 芯片"];
  strips.forEach((st, i) => {
    const w = 3.4, x = 0.6 + i * (w + 0.56);
    card(s, x, 6.32, w, 0.56, "0E1A2A");
    s.addText(st, { x, y: 6.32, w, h: 0.56, fontSize: 13.5, fontFace: FONT,
      color: APP, bold: true, align: "center", valign: "middle" });
  });
  footer(s, 8);
}

// ════════════════════════════════════════════════
// Slide 9 · 应用层面的意义
// ════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: C.bg };
  header(s, "意义  SIGNIFICANCE", APP, "为什么是「在身上」：应用层面的意义");

  s.addText("不在医院实验室，就在身上", { x: 0.6, y: 2.0, w: 12.13, h: 0.95,
    fontSize: 40, fontFace: FONT, color: C.white, bold: true, align: "center" });
  s.addText("从「病人专属的医疗设备」到「人人可戴的日常系统」—— 脑机接口第一次可以连续地、无感地陪伴一个人",
    { x: 1.2, y: 3.0, w: 10.93, h: 0.45, fontSize: 14, fontFace: FONT, color: C.body, align: "center" });

  const vals = [
    { num: "01", t: "检测解码系统", c: APP, d: "完整的「采集→解码→通信→反馈」闭环，能独立工作、日常佩戴" },
    { num: "02", t: "生物脑 × 硅基脑", c: BCI, d: "碳基大脑负责「想」，硅基芯片负责「懂」，无缝连接、互为延伸" },
    { num: "03", t: "三个硬指标", c: HBC, d: "隐私（数据不出身体）· 便宜（¥338）· 实时（50ms）" },
  ];
  vals.forEach((v, i) => {
    const x = 0.6 + i * 4.16;
    // 顶部彩色短横线
    s.addShape(pres.ShapeType.rect, { x: x + 0.28, y: 3.55, w: 0.42, h: 0.05, fill: { color: v.c } });
    // 大序号
    s.addText(v.num, { x: x + 0.28, y: 3.72, w: 3.3, h: 0.62, fontSize: 30,
      fontFace: FONT, color: v.c, bold: true });
    // 标题
    s.addText(v.t, { x: x + 0.28, y: 4.42, w: 3.3, h: 0.42, fontSize: 17,
      fontFace: FONT, color: C.text, bold: true });
    // 描述
    s.addText(v.d, { x: x + 0.28, y: 4.9, w: 3.3, h: 1.05, fontSize: 12,
      fontFace: FONT, color: C.body });
  });

  card(s, 0.6, 6.05, 12.13, 0.82, "0E1A2A");
  s.addText("反哺学术", { x: 0.85, y: 6.05, w: 2.0, h: 0.82, fontSize: 14,
    fontFace: FONT, color: APP, bold: true, valign: "middle" });
  s.addText("「戴在身上」意味着采到实验室采不到的真实人体链路数据 —— 直接喂给论文实验章节，学术优先权与成果转化同步推进",
    { x: 2.9, y: 6.05, w: 9.6, h: 0.82, fontSize: 12.5, fontFace: FONT, color: C.body, valign: "middle" });
  footer(s, 9);
}

// ════════════════════════════════════════════════
// Slide 10 · 下一步 + 结尾
// ════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: C.dark };
  header(s, "计划  ROADMAP", C.accent, "下一步：补全物理层，走向真实");

  const steps = [
    { num: "1", t: "真实人体信道实测", c: HBC,
      d: "采购电极 + 收发板，把三环里最「纸面」的一环变成板上实测" },
    { num: "2", t: "灵汐 KA200 类脑芯片", c: SNN,
      d: "真类脑低功耗计算单元，补全「硅基脑」的硬件形态" },
    { num: "3", t: "可穿戴原型", c: APP,
      d: "把开发板 + 电脑演示，整合为一件戴得住的闭环原型" },
  ];
  // 时间轴横线（贯穿三个节点）
  s.addShape(pres.ShapeType.rect, { x: 1.2, y: 2.26, w: 10.93, h: 0.035, fill: { color: "2A3B4E" } });
  steps.forEach((st, i) => {
    const cx = 2.4 + i * 4.2;   // 节点圆心 x
    s.addShape(pres.ShapeType.ellipse, { x: cx - 0.3, y: 1.96, w: 0.6, h: 0.6,
      fill: { color: st.c }, line: { color: st.c, width: 2 } });
    s.addText(st.num, { x: cx - 0.3, y: 1.96, w: 0.6, h: 0.6, fontSize: 20,
      fontFace: FONT, color: C.dark, bold: true, align: "center", valign: "middle" });
    s.addText(st.t, { x: cx - 2.0, y: 2.78, w: 4.0, h: 0.45, fontSize: 15,
      fontFace: FONT, color: st.c, bold: true, align: "center" });
    s.addText(st.d, { x: cx - 2.0, y: 3.3, w: 4.0, h: 1.0, fontSize: 11.5,
      fontFace: FONT, color: C.body, align: "center" });
  });

  // 待办
  card(s, 0.6, 4.45, 12.13, 0.72, "0E1A2A");
  s.addText("本周待办", { x: 0.85, y: 4.45, w: 2.0, h: 0.72, fontSize: 14,
    fontFace: FONT, color: C.amber, bold: true, valign: "middle" });
  s.addText("采购电极与收发板（发票留存，周五下班前报销）· 本 PPT 周五前提交 · 周末修改完善形成实验室下一步研究方案",
    { x: 2.9, y: 4.45, w: 9.6, h: 0.72, fontSize: 12, fontFace: FONT, color: C.body, valign: "middle" });

  // 底部：结尾金句（居中占满）
  s.addText("让大脑的想法，变成身边设备的动作", { x: 0.6, y: 5.5, w: 12.13, h: 0.6,
    fontSize: 24, fontFace: FONT, color: C.white, bold: true, align: "center" });
  s.addText("戴在身上，而不是躺在实验室里。", { x: 0.6, y: 6.15, w: 12.13, h: 0.4,
    fontSize: 14, fontFace: FONT, color: C.accent, align: "center", bold: true });
}

// ── 写文件 ──
pres.writeFile({ fileName: "/Users/heyixuan/BCI/20260828/碳硅协同_检测解码系统_20260908.pptx" })
  .then((f) => console.log("DONE:", f))
  .catch((e) => console.error("ERR:", e));
