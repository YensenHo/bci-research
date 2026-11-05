const PptxGenJS = require("pptxgenjs");
const pptx = new PptxGenJS();
pptx.layout = "LAYOUT_WIDE"; // 13.33 x 7.5

const BG = "0B1120";
const PANEL = "111827";
const PANEL2 = "1E293B";
const ACCENT = "22D3EE";
const WHITE = "F8FAFC";
const MUTED = "94A3B8";
const GREEN = "34D399";
const RED = "F87171";
const AMBER = "FBBF24";
const FONT = "Microsoft YaHei";

// 辅助：标题
function title(slide, text) {
  slide.addText(text, {
    x: 0.55, y: 0.45, w: 12.2, h: 0.7,
    fontSize: 30, bold: true, color: WHITE, fontFace: FONT,
  });
}

// 辅助：圆角卡片
function card(slide, x, y, w, h, fill, line) {
  slide.addShape(pptx.ShapeType.roundRect, {
    x, y, w, h, fill: { color: fill }, line: line ? { color: line, width: 1 } : { color: fill, width: 0 },
    rectRadius: 0.08,
  });
}

// ============ 第 1 页 封面 ============
{
  const s = pptx.addSlide();
  s.background = { color: BG };
  s.addText("采集-去噪一体化前端模组 · 第一步", {
    x: 0.6, y: 1.2, w: 12, h: 0.4,
    fontSize: 15, color: ACCENT, fontFace: FONT, bold: true, charSpacing: 3,
  });
  s.addText("P0 单通道主动电极", {
    x: 0.55, y: 1.8, w: 12, h: 1.3,
    fontSize: 52, bold: true, color: WHITE, fontFace: FONT,
  });
  s.addText("验证实验方案", {
    x: 0.55, y: 3.1, w: 12, h: 0.8,
    fontSize: 30, color: MUTED, fontFace: FONT,
  });
  // 核心目标大卡
  card(s, 0.6, 4.6, 7.5, 1.8, PANEL, "22D3EE");
  s.addText("核心目标", {
    x: 0.9, y: 4.85, w: 6.8, h: 0.4,
    fontSize: 14, color: MUTED, fontFace: FONT,
  });
  s.addText("主动电极噪声底  ≤  湿电极 × 2", {
    x: 0.9, y: 5.25, w: 6.9, h: 0.8,
    fontSize: 30, bold: true, color: ACCENT, fontFace: FONT,
  });
  // 右侧三要点
  card(s, 8.35, 4.6, 4.4, 1.8, PANEL2);
  s.addText([
    { text: "只测一个变量：噪声底", options: { fontSize: 15, color: WHITE, fontFace: FONT, breakLine: true } },
    { text: "一个数字定成败", options: { fontSize: 15, color: WHITE, fontFace: FONT } },
  ], { x: 8.6, y: 4.85, w: 3.95, h: 1.3, valign: "middle" });
  s.addText("预算 ¥400–1000 · 周期约 2 周", {
    x: 0.6, y: 6.7, w: 12, h: 0.4,
    fontSize: 14, color: MUTED, fontFace: FONT,
  });
}

// ============ 第 2 页 P0 在验证什么 ============
{
  const s = pptx.addSlide();
  s.background = { color: BG };
  title(s, "P0 到底在验证什么");
  s.addText("普通采集板，电极直接接芯片。干/半干电极源阻抗高，直接接会把信号吃掉、噪声放大。", {
    x: 0.6, y: 1.35, w: 6.2, h: 0.9,
    fontSize: 16, color: WHITE, fontFace: FONT,
  });
  s.addText("主动电极 = 在电极头后面塞一个高输入阻抗的缓冲运放，让信号在离头皮最近的地方被低阻抗放大，再走线到 ADS1299。", {
    x: 0.6, y: 2.35, w: 6.2, h: 1.3,
    fontSize: 16, color: MUTED, fontFace: FONT,
  });
  // 右侧流程图
  card(s, 7.1, 1.35, 5.6, 2.9, PANEL);
  s.addText([
    { text: "电极头", options: { fontSize: 15, color: WHITE, fontFace: FONT, bold: true } },
    { text: "（高源阻抗）", options: { fontSize: 12, color: MUTED, fontFace: FONT } },
  ], { x: 7.4, y: 1.55, w: 5.0, h: 0.7, align: "center" });
  s.addText("↓", { x: 7.4, y: 2.15, w: 5.0, h: 0.4, align: "center", fontSize: 20, color: ACCENT });
  s.addText([
    { text: "缓冲运放（跟随器）", options: { fontSize: 15, color: ACCENT, fontFace: FONT, bold: true } },
    { text: "高阻进 · 低阻出", options: { fontSize: 12, color: MUTED, fontFace: FONT } },
  ], { x: 7.4, y: 2.5, w: 5.0, h: 0.7, align: "center" });
  s.addText("↓", { x: 7.4, y: 3.15, w: 5.0, h: 0.4, align: "center", fontSize: 20, color: ACCENT });
  s.addText("ADS1299", { x: 7.4, y: 3.5, w: 5.0, h: 0.5, align: "center", fontSize: 16, color: WHITE, fontFace: FONT, bold: true });
  // 底部三卡
  const cards = [
    ["只测噪声底", "阻抗监测 / 多通道 / HBC 一律不做，避免变量混在一起分不清谁的功劳"],
    ["金标准对照", "Ag/AgCl 湿电极 + 导电膏是基准，主动电极要逼近它"],
    ["可证伪", "噪声底达标进 P1，不达标诚实回退，不硬撑"],
  ];
  cards.forEach((c, i) => {
    const x = 0.6 + i * 4.1;
    card(s, x, 4.6, 3.85, 2.0, PANEL);
    s.addText(c[0], { x: x + 0.25, y: 4.85, w: 3.35, h: 0.5, fontSize: 18, bold: true, color: ACCENT, fontFace: FONT });
    s.addText(c[1], { x: x + 0.25, y: 5.35, w: 3.35, h: 1.1, fontSize: 13, color: MUTED, fontFace: FONT });
  });
}

// ============ 第 3 页 三个测量条件 ============
{
  const s = pptx.addSlide();
  s.background = { color: BG };
  title(s, "三个测量条件（缺一不可）");
  const cols = [
    ["A", "输入短接", "ADS1299 输入直接短接到地", "系统本底噪声", "最低极限", MUTED],
    ["B", "湿电极对照", "Ag/AgCl + 导电膏贴额头", "金标准基准", "要逼近的对象", WHITE],
    ["C", "主动电极", "干/半干电极 → 缓冲 → ADS1299", "被测对象", "判定主角", ACCENT],
  ];
  cols.forEach((c, i) => {
    const x = 0.6 + i * 4.1;
    card(s, x, 1.4, 3.85, 3.1, PANEL);
    s.addText(c[0], { x: x + 0.3, y: 1.6, w: 3.2, h: 0.6, fontSize: 40, bold: true, color: c[6], fontFace: FONT });
    s.addText(c[1], { x: x + 0.3, y: 2.25, w: 3.3, h: 0.5, fontSize: 18, bold: true, color: WHITE, fontFace: FONT });
    s.addText(c[2], { x: x + 0.3, y: 2.75, w: 3.3, h: 0.9, fontSize: 13, color: MUTED, fontFace: FONT });
    s.addText(c[3] + "  ·  " + c[4], { x: x + 0.3, y: 3.7, w: 3.3, h: 0.6, fontSize: 13, bold: true, color: c[6], fontFace: FONT });
  });
  // 判定条
  card(s, 0.6, 4.85, 12.1, 1.6, PANEL2);
  s.addText("判定标准", { x: 0.9, y: 5.05, w: 2.0, h: 0.4, fontSize: 14, color: MUTED, fontFace: FONT });
  s.addText([
    { text: "RMS(C) / RMS(B)  ≤  2.0", options: { fontSize: 26, bold: true, color: ACCENT, fontFace: FONT, breakLine: true } },
    { text: "A 是参考下限 —— C 不可能好过 A，越接近 A 说明缓冲做得越好", options: { fontSize: 13, color: MUTED, fontFace: FONT } },
  ], { x: 0.9, y: 5.45, w: 11.5, h: 0.9 });
}

// ============ 第 4 页 缓冲电路 ============
{
  const s = pptx.addSlide();
  s.background = { color: BG };
  title(s, "缓冲电路：电压跟随器");
  // 电路流程
  card(s, 0.6, 1.4, 12.1, 2.6, PANEL);
  const steps = [
    ["电极头", "干/半干"],
    ["100kΩ", "保护电阻"],
    ["运放 +", "同相端"],
    ["跟随器", "增益 = 1"],
    ["ADS1299", "通道输入"],
  ];
  steps.forEach((st, i) => {
    const x = 0.9 + i * 2.4;
    s.addShape(pptx.ShapeType.roundRect, { x, y: 2.1, w: 1.7, h: 1.2, fill: { color: PANEL2 }, rectRadius: 0.1 });
    s.addText(st[0], { x, y: 2.2, w: 1.7, h: 0.6, align: "center", fontSize: 14, bold: true, color: WHITE, fontFace: FONT });
    s.addText(st[1], { x, y: 2.75, w: 1.7, h: 0.4, align: "center", fontSize: 10, color: MUTED, fontFace: FONT });
    if (i < 4) s.addText("→", { x: x + 1.7, y: 2.35, w: 0.7, h: 0.5, align: "center", fontSize: 18, color: ACCENT });
  });
  s.addText("运放接成电压跟随器（增益 1），关键是输入阻抗高；运放要靠近电极头放（这就是「主动电极」的物理含义）。", {
    x: 0.9, y: 1.55, w: 11.5, h: 0.5, fontSize: 14, color: MUTED, fontFace: FONT,
  });
  // 选型策略
  s.addText("运放选型策略", { x: 0.6, y: 4.35, w: 4.0, h: 0.5, fontSize: 18, bold: true, color: WHITE, fontFace: FONT });
  const chips = [
    ["第一步", "TL072", "JFET 高阻 · 便宜", "先跑通电路", MUTED],
    ["第二步", "OPA376", "CMOS · 低噪声", "噪声不达标再换", ACCENT],
    ["第三步", "OPA140", "JFET · 超低噪声", "冲指标", AMBER],
  ];
  chips.forEach((c, i) => {
    const x = 0.6 + i * 4.1;
    card(s, x, 4.9, 3.85, 1.7, PANEL);
    s.addText(c[0], { x: x + 0.25, y: 5.05, w: 3.3, h: 0.4, fontSize: 12, color: MUTED, fontFace: FONT });
    s.addText(c[1], { x: x + 0.25, y: 5.4, w: 3.3, h: 0.5, fontSize: 20, bold: true, color: c[4], fontFace: FONT });
    s.addText(c[2] + "  ·  " + c[3], { x: x + 0.25, y: 5.95, w: 3.3, h: 0.5, fontSize: 12, color: MUTED, fontFace: FONT });
  });
}

// ============ 第 5 页 测量流程 ============
{
  const s = pptx.addSlide();
  s.background = { color: BG };
  title(s, "测量流程（5 步照做）");
  const steps = [
    ["1", "通 SPI", "读 ADS1299 ID 寄存器（应为 0x3E），确认通信正常"],
    ["2", "条件 A 短接", "输入短接到地，采 2 分钟，算 RMS 本底噪声"],
    ["3", "条件 B 湿电极", "Ag/AgCl + 导电膏贴额头，采 2–3 分钟静息脑电"],
    ["4", "条件 C 主动电极", "同位置换干/半干 → 缓冲，采 2–3 分钟"],
    ["5", "记录判定", "每个条件 RMS + 功率谱，算 C/B 比值"],
  ];
  steps.forEach((st, i) => {
    const x = 0.6 + i * 2.45;
    card(s, x, 1.5, 2.25, 3.4, PANEL);
    s.addText(st[0], { x: x + 0.15, y: 1.7, w: 0.7, h: 0.7, fontSize: 28, bold: true, color: ACCENT, fontFace: FONT });
    s.addText(st[1], { x: x + 0.15, y: 2.5, w: 1.95, h: 0.8, fontSize: 15, bold: true, color: WHITE, fontFace: FONT });
    s.addText(st[2], { x: x + 0.15, y: 3.3, w: 1.95, h: 1.5, fontSize: 12, color: MUTED, fontFace: FONT });
  });
  card(s, 0.6, 5.2, 12.1, 1.4, PANEL2);
  s.addText([
    { text: "判定：RMS(C) / RMS(B) ≤ 2.0 → 通过，进 P1", options: { fontSize: 18, bold: true, color: GREEN, fontFace: FONT, breakLine: true } },
    { text: "不达标 → 回到第 4 步换运放 / 换电极头，或退回「被动直连」方案", options: { fontSize: 14, color: MUTED, fontFace: FONT } },
  ], { x: 0.9, y: 5.4, w: 11.5, h: 1.0 });
}

// ============ 第 6 页 记录表 + 判定 ============
{
  const s = pptx.addSlide();
  s.background = { color: BG };
  title(s, "记录一张表，定成败");
  const rows = [
    [
      { text: "条件", options: { bold: true, color: WHITE, fill: { color: PANEL2 }, fontFace: FONT } },
      { text: "RMS 噪声底 (µV)", options: { bold: true, color: WHITE, fill: { color: PANEL2 }, fontFace: FONT } },
      { text: "备注", options: { bold: true, color: WHITE, fill: { color: PANEL2 }, fontFace: FONT } },
    ],
    [{ text: "A 短接", options: { color: MUTED, fontFace: FONT } }, { text: "", options: { fontFace: FONT } }, { text: "", options: { fontFace: FONT } }],
    [{ text: "B 湿电极", options: { color: WHITE, fontFace: FONT } }, { text: "", options: { fontFace: FONT } }, { text: "", options: { fontFace: FONT } }],
    [{ text: "C 主动电极", options: { color: ACCENT, bold: true, fontFace: FONT } }, { text: "", options: { fontFace: FONT } }, { text: "", options: { fontFace: FONT } }],
    [{ text: "比值 C/B", options: { color: AMBER, bold: true, fontFace: FONT } }, { text: "≤ 2.0 过 / > 2.0 回退", options: { color: AMBER, bold: true, fontFace: FONT } }, { text: "", options: { fontFace: FONT } }],
  ];
  s.addTable(rows, {
    x: 0.6, y: 1.5, w: 12.1,
    colW: [3.5, 5.0, 3.6],
    border: { type: "solid", color: "1E293B", pt: 1 },
    fontFace: FONT, fontSize: 15, valign: "middle", align: "left",
    rowH: 0.55,
  });
  s.addText("每个条件采完当场算 RMS（µV）+ 保存一张功率谱，看 1–50Hz 有没有异常隆起（尤其 50Hz 工频）。", {
    x: 0.6, y: 4.7, w: 12.1, h: 0.6, fontSize: 14, color: MUTED, fontFace: FONT,
  });
}

// ============ 第 7 页 诚实提醒 ============
{
  const s = pptx.addSlide();
  s.background = { color: BG };
  title(s, "三条诚实提醒");
  const items = [
    ["50Hz 工频", "会严重污染噪声底。若 RMS 被工频主导，先看频谱，再考虑电池供电 / 远离交流设备 / 软件陷波，但别用陷波掩盖真实噪声。"],
    ["干电极源阻抗", "先测一次电极-皮肤阻抗并记录。阻抗异常高（>几百 kΩ）说明接触不良，先解决接触再测噪声。"],
    ["拿真实数字", "P0 的目的不是冲指标，是拿一个真实数字判断路线走不走得通。达标进 P1，不达标诚实回退，不硬撑。"],
  ];
  items.forEach((it, i) => {
    const y = 1.6 + i * 1.75;
    card(s, 0.6, y, 12.1, 1.55, PANEL);
    s.addText(String(i + 1), { x: 0.9, y: y + 0.3, w: 0.9, h: 0.9, fontSize: 30, bold: true, color: ACCENT, fontFace: FONT });
    s.addText(it[0], { x: 1.95, y: y + 0.2, w: 10.5, h: 0.5, fontSize: 17, bold: true, color: WHITE, fontFace: FONT });
    s.addText(it[1], { x: 1.95, y: y + 0.7, w: 10.5, h: 0.75, fontSize: 13, color: MUTED, fontFace: FONT });
  });
}

pptx.writeFile({ fileName: "/Users/heyixuan/BCI/acquisition-frontend/ppt/P0验证实验方案.pptx" }).then((f) => {
  console.log("DONE:", f);
});
