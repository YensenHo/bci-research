// 完整报告版 PPT：HBC×BCI 板载实测 + 下一步任务 + 三线调研（整合全部 md 内容）
const pptxgen = require('pptxgenjs');
const p = new pptxgen();
p.layout = 'LAYOUT_WIDE';  // 13.3 x 7.5

// ---- 主题 ----
const BG    = '1B2A4A';
const CARD  = '24395E';
const CARD2 = '2D4A75';
const TXT   = 'FFFFFF';
const MUT   = 'A9BAD4';
const CYAN  = '4FC3F7';
const GREEN = '4ADE80';
const AMBER = 'FBBF24';
const RED   = 'F87171';
const M = 0.55;
const W = 12.2;   // 内容宽

let page = 0;
function header(s, title, sub) {
  page++;
  s.background = { color: BG };
  s.addText(title, { x: M, y: 0.4, w: W, h: 0.7, fontSize: 27, bold: true, color: TXT, fontFace: 'Arial' });
  if (sub) s.addText(sub, { x: M, y: 1.08, w: W, h: 0.38, fontSize: 13, color: CYAN, fontFace: 'Arial' });
  s.addShape('rect', { x: M, y: 1.56, w: W, h: 0.018, fill: { color: CARD2 } });
}
function footer(s) {
  s.addText('HBC×BCI 完整报告 · 2026-08-20', { x: M, y: 7.08, w: 4.5, h: 0.28, fontSize: 9, color: MUT, fontFace: 'Arial' });
  if (page > 0) s.addText(String(page + 1), { x: 11.6, y: 7.08, w: 1.1, h: 0.28, fontSize: 9, color: MUT, align: 'right', fontFace: 'Arial' });
}
function card(s, x, y, w, h, title, items, accent) {
  s.addShape('roundRect', { x, y, w, h, rectRadius: 0.05, fill: { color: CARD }, line: { color: CARD2, width: 0.5 } });
  s.addText(title, { x: x + 0.22, y: y + 0.14, w: w - 0.44, h: 0.34, fontSize: 13.5, bold: true, color: accent || CYAN, fontFace: 'Arial' });
  const list = Array.isArray(items) ? items : [items];
  const arr = list.map((it, i) => {
    const t = typeof it === 'string' ? it : it.text;
    const o = typeof it === 'string' ? {} : it.options;
    return { text: t, options: { bullet: false, breakLine: i < list.length - 1, paraSpaceAfter: 3, ...o } };
  });
  s.addText(arr, { x: x + 0.22, y: y + 0.52, w: w - 0.44, h: h - 0.62, fontSize: 11, color: TXT, fontFace: 'Arial', valign: 'top', lineSpacingMultiple: 1.12 });
}
// 表格：hdr=表头数组, data=二维数组(纯字符串), colW=列宽数组
function tbl(s, x, y, hdr, data, colW, opt) {
  opt = opt || {};
  const rows = [];
  rows.push(hdr.map(h => ({ text: h, options: { bold: true, color: CYAN, fill: { color: CARD2 } } })));
  for (const r of data) rows.push(r.map(c => ({ text: c, options: {} })));
  s.addTable(rows, {
    x, y, w: colW.reduce((a, b) => a + b, 0), colW,
    fontSize: opt.fontSize || 10, fontFace: 'Arial',
    border: { type: 'solid', color: CARD2, pt: 0.5 },
    fill: { color: CARD }, color: TXT,
    rowH: opt.rowH || 0.42, valign: 'middle', align: 'left', autoPage: false,
  });
}
function secBar(s, label) {
  s.addShape('roundRect', { x: M, y: 1.78, w: W, h: 0.5, rectRadius: 0.1, fill: { color: '1F3350' }, line: { color: CYAN, width: 0.75 } });
  s.addText(label, { x: M + 0.2, y: 1.82, w: W - 0.4, h: 0.42, fontSize: 14, bold: true, color: CYAN, fontFace: 'Arial', valign: 'middle' });
}

// ============ 1 封面 ============
let s = p.addSlide();
s.background = { color: BG };
s.addText('HBC × BCI 端侧部署', { x: M, y: 1.6, w: W, h: 1.1, fontSize: 46, bold: true, color: TXT, fontFace: 'Arial' });
s.addText('完整报告：板载实测 · 下一步任务 · 三线调研', { x: M, y: 2.75, w: W, h: 0.6, fontSize: 22, color: CYAN, fontFace: 'Arial' });
s.addText('国产 RV1106 端侧神经解码 × EQS-HBC 人体通信回传闭环', { x: M, y: 3.45, w: W, h: 0.5, fontSize: 15, color: MUT, fontFace: 'Arial' });
s.addText('2026-08-20 · 板子已插入，全部数字板载实测', { x: M, y: 4.15, w: W, h: 0.4, fontSize: 13, color: MUT, fontFace: 'Arial' });
s.addShape('rect', { x: M, y: 4.9, w: W, h: 0.03, fill: { color: CYAN } });
s.addText('诚实边界：46.3dB 信道损耗 / 闭环为「模型驱动」，非真实体表传输测量；真实站得住的是 BCIC IV 2a 真实脑电驱动的软件信号链在 RV1106 跑通。', { x: M, y: 5.1, w: W, h: 0.9, fontSize: 12, color: AMBER, fontFace: 'Arial', italic: true });
footer(s);

// ============ 2 目录 ============
s = p.addSlide();
header(s, '目录', '七大板块 · 32 页');
const toc = [
  ['一', '板载实测复现', '硬件信息 · 推理 · 延迟 · 位深 · McNemar · SNR 瀑布', '3-7'],
  ['二', '约束驱动鲁棒解码', '创新点定稿 · 三第一性约束 · 验证结果', '8-9'],
  ['三', '同平台模型对比', '三声明核实 · 六维对比表 · RV1106 首例', '10-13'],
  ['四', 'MyoText 精读 + 双模态', '框架 · 精度 · EEG/EMG 融合 · 可行性', '14-19'],
  ['五', '北京 BCI 公司 + 竞赛', '赛道地图 · 方向判断 · 竞赛清单', '20-23'],
  ['六', '以创促研 · 产学研路径', '数据瓶颈 · 数据放大器 · 优先权 · 成果转化 · 路线图 · 红线', '24-29'],
  ['七', '下一步任务 + 定位', 'P0-P4 · 分工 · 定位与诚实边界 · 出处', '30-32'],
];
tbl(s, M, 2.0, ['#', '板块', '内容', '页'], toc.map((r, i) => [r[0], r[1], r[2], r[3]]), [0.8, 3.2, 6.2, 2.0], { fontSize: 12, rowH: 0.66 });
footer(s);

// ============ 3 板载硬件 ============
s = p.addSlide();
header(s, '一 · 板载实测：硬件信息', '板子已插入 · adb 序列 5caaa04f9cdb6ba2');
tbl(s, M, 1.85, ['项目', '实测值', '来源'],
  [
    ['SoC', 'RV1106 · ARMv7 Cortex-A7 rev5', '/proc/cpuinfo'],
    ['内核', 'Linux 5.10.160 (armv7l)', 'uname -a'],
    ['系统 RAM', '256 MB（0x0fffffff）', '/proc/iomem'],
    ['内存', 'MemTotal 171MB + CMA 76MB(NPU/ISP预留)', '/proc/meminfo'],
    ['温度', '44.6 °C', 'thermal_zone0'],
    ['adb', '5caaa04f9cdb6ba2', 'adb devices'],
  ],
  [1.6, 7.0, 3.6], { fontSize: 11, rowH: 0.56 });
s.addShape('roundRect', { x: M, y: 5.6, w: W, h: 1.05, rectRadius: 0.06, fill: { color: '3A2A1F' }, line: { color: AMBER, width: 1 } });
s.addText('⚠️ 规格纠错：/proc/iomem 实测 System RAM = 256MB，config.yaml 与 README 原写「512MB / RV1106G3」与实测不符。已改 config.yaml（ram=256MB），PPT 里「512MB」卖点需核实（256MB 变体 or 内核只映射一半）。', { x: M + 0.2, y: 5.72, w: W - 0.4, h: 0.85, fontSize: 12, color: AMBER, fontFace: 'Arial', valign: 'middle', lineSpacingMultiple: 1.15 });
footer(s);

// ============ 4 推理+延迟 ============
s = p.addSlide();
header(s, '一 · 板载实测：EEGNet 推理 + 端到端延迟', '真实 adb 输出，与 board_verification_log.md 吻合；以下为旧权重实测');
card(s, M, 1.85, 5.9, 2.35, 'eegnet_infer（CPU 单次前向）', [
  { text: '预测类别: 1, logits=[-0.006, 0.051]', options: { fontFace: 'Courier New', fontSize: 10, breakLine: true } },
  { text: '单次前向耗时: 18.50 ms（旧权重；新权重 23.2ms）', options: { bold: true, color: GREEN, breakLine: true } },
  { text: '与 PyTorch 误差 8e-07（1296/1296 一致）', options: {} },
], CYAN);
card(s, 6.65, 1.85, 6.1, 2.35, 'hbc_benchmark（完整闭环）', [
  { text: '经HBC准确率 81.0% (81/100) · 位深 5bit', options: { breakLine: true } },
  { text: '延迟: 平均 44.85ms · p50 44.63ms · p99 65.00ms', options: { breakLine: true } },
  { text: '吞吐 22.3 窗口/s · 实时余量 31× · 峰值 RSS 1.82MB', options: { breakLine: true } },
  { text: 'RSS 1.82MB = 256MB 板子的 0.71%', options: { color: MUT, fontSize: 10 } },
], GREEN);
s.addShape('roundRect', { x: M, y: 4.45, w: W, h: 2.1, rectRadius: 0.05, fill: { color: CARD }, line: { color: CARD2, width: 0.5 } });
s.addText('关键结论', { x: M + 0.2, y: 4.58, w: 4, h: 0.35, fontSize: 13.5, bold: true, color: CYAN, fontFace: 'Arial' });
s.addText([
  { text: '• RV1106 纯 CPU 前向 18.5ms（旧权重），每个 2s 脑电 epoch 解码+传输端到端 p99 65ms，实时余量 31×', options: { bullet: false, breakLine: true } },
  { text: '• NPU 推理 1.2ms（15× 加速真实），但 INT8 量化精度不达标（argmax 一致率 45.83%，288 样本）→ 解码正确性走 CPU 版', options: { bullet: false, breakLine: true } },
  { text: '• 短板在 RV1106 量化工具链对深度可分离小模型的支持不足，不在模型本身', options: { bullet: false } },
], { x: M + 0.2, y: 4.95, w: W - 0.4, h: 1.5, fontSize: 12, color: TXT, fontFace: 'Arial', valign: 'top', lineSpacingMultiple: 1.3 });
footer(s);

// ============ 5 位深扫描 ============
s = p.addSlide();
header(s, '一 · 板载实测：HBC 位深扫描', '传输神经信号只需 5 bit/样本（比 8bit 省 37.5% 带宽）· 旧权重实测');
tbl(s, M, 1.85, ['QBITS', '量化MSE', 'BER', '直连', '经HBC', '下降'],
  [
    ['3', '0.109151', '0', '81.0%', '76.0%', '5.00pp'],
    ['4', '0.023689', '0', '81.0%', '75.0%', '6.00pp'],
    ['5', '0.005567', '5e-7', '81.0%', '81.0%', '0.00pp'],
    ['6', '0.001348', '0', '81.0%', '81.0%', '0.00pp'],
    ['7', '0.000335', '3.6e-7', '81.0%', '80.0%', '1.00pp'],
    ['8', '0.000086', '3.1e-7', '81.0%', '81.0%', '0.00pp'],
    ['9', '0.000024', '0', '81.0%', '81.0%', '0.00pp'],
    ['10', '0.000049', '2.5e-7', '81.0%', '81.0%', '0.00pp'],
    ['11', '0.000015', '6.8e-7', '81.0%', '81.0%', '0.00pp'],
    ['12', '0.000005', '2.1e-7', '81.0%', '81.0%', '0.00pp'],
  ],
  [1.6, 2.4, 2.2, 1.9, 2.0, 2.1], { fontSize: 11, rowH: 0.4 });
s.addShape('roundRect', { x: M, y: 6.15, w: W, h: 0.75, rectRadius: 0.06, fill: { color: '1F3350' }, line: { color: GREEN, width: 1 } });
s.addText('结论：5 bit 即保精度（量化 MSE 0.0056 不影响解码，下降 0.00pp）——HBC 回传链路只需 5bit/样本。', { x: M + 0.2, y: 6.25, w: W - 0.4, h: 0.55, fontSize: 13, color: TXT, fontFace: 'Arial', valign: 'middle' });
footer(s);

// ============ 6 McNemar ============
s = p.addSlide();
header(s, '一 · 板载实测：McNemar 显著性检验', '直连 vs 经HBC 配对预测（1296 试次 × 5 种子 = 6480 配对）· 旧权重实测');
tbl(s, M, 1.85, ['', '经HBC对', '经HBC错', '合计'],
  [
    ['直连对', '4845', '105', '4950'],
    ['直连错', '65', '1465', '1530'],
    ['合计', '4910', '1570', '6480'],
  ],
  [2.2, 2.6, 2.6, 2.6], { fontSize: 12, rowH: 0.55 });
card(s, M, 3.9, 6.0, 2.55, '显著性（b=105, c=65, n=170）', [
  { text: '直连准确率 76.39% (4950/6480)', options: { breakLine: true } },
  { text: '经HBC准确率 75.77% (4910/6480)  →  损失 0.62pp', options: { breakLine: true } },
  { text: 'χ²(无校正) = 9.4118  →  p = 0.00216', options: { breakLine: true } },
  { text: 'χ²(连续校正) = 8.9471  →  p = 0.00278', options: { breakLine: true } },
  { text: '精确二项(双侧)  →  p = 0.00267', options: {} },
], CYAN);
card(s, 6.75, 3.9, 6.0, 2.55, '结论', [
  { text: '经 HBC 引入的 0.62pp 损失（5bit 量化）统计显著（p≈0.003 < 0.01）但量级很小。', options: { breakLine: true } },
  { text: '与 board_verification_log.md §4 一致（旧权重实测）。', options: { breakLine: true } },
  { text: '说明 HBC 链路（量化+信道）对解码的影响是真实存在、但可忽略的小效应。', options: {} },
], GREEN);
footer(s);

// ============ 7 SNR 瀑布 ============
s = p.addSlide();
header(s, '一 · 板载实测：SNR 瀑布曲线', '片上 SNR→准确率/BER（1296 试次 × 5 种子，位深 5bit）· 旧权重实测');
tbl(s, M, 1.85, ['SNR(dB)', 'σ/H', '经HBC准确率(均值)', '跨种子范围', 'BER(均值)'],
  [
    ['26.0', '0.050', '75.77%', '75.77~75.77%', '0.00e+00'],
    ['20.0', '0.100', '75.77%', '75.77~75.77%', '3.32e-07'],
    ['14.0', '0.200', '72.44%', '71.30~73.15%', '6.22e-03'],
    ['8.0', '0.400', '56.76%', '55.94~57.64%', '1.06e-01'],
    ['1.9', '0.800', '52.22%', '51.62~53.40%', '2.66e-01'],
    ['-1.6', '1.200', '51.37%', '50.77~52.01%', '3.38e-01'],
    ['-6.0', '2.000', '50.63%', '49.69~51.93%', '4.01e-01'],
  ],
  [1.6, 1.6, 3.4, 3.0, 2.6], { fontSize: 11.5, rowH: 0.48 });
s.addShape('roundRect', { x: M, y: 5.6, w: W, h: 1.05, rectRadius: 0.06, fill: { color: '1F3350' }, line: { color: GREEN, width: 1 } });
s.addText('结论：拐点在 8~14dB 之间；≥20dB 完全保精度（BER≈0，准确率=直连 75.77%）；≤8dB 解码失效。片上实测拐点比 README 引用的「18dB 悬崖」更低——系统比宣称的更鲁棒。', { x: M + 0.2, y: 5.72, w: W - 0.4, h: 0.85, fontSize: 12.5, color: TXT, fontFace: 'Arial', valign: 'middle', lineSpacingMultiple: 1.2 });
footer(s);

// ============ 8 约束驱动鲁棒解码：创新点定稿 ============
s = p.addSlide();
header(s, '二 · 约束驱动鲁棒解码（P0 创新点·定稿）', '模型 = EEGNet 不动；创新点 = 把「HBC 噪声 + INT8 量化」作第一性约束纳入解码设计');
card(s, M, 1.85, 12.2, 1.25, '为什么原「自监督预训练」被毙（Hermes↔Codex 三方一致）', [
  { text: '数据量死穴：1296 试次 × ~2s ≈ 43 分钟 EEG；自监督增益来自 TUH 级数万小时语料，43 分钟 + 1.7K 参数无「表示容量」→ 增益趋近 0。', options: { breakLine: true } },
  { text: '红海（掩码重建 EEG 是 BENDR 2021 后标准操作，novelty=0）+ 成本（重下 420MB + GPU + 整条训练链，收益≈0）。', options: {} },
], RED);
s.addShape('roundRect', { x: M, y: 3.25, w: W, h: 1.35, rectRadius: 0.06, fill: { color: '1F3350' }, line: { color: GREEN, width: 1 } });
s.addText([
  { text: '定稿方向：现有 EEG 解码默认「理想无线链路 + 浮点算力」。本工作首次在 HBC 人体通信闭环设定下，把人体信道突发噪声 + 国产 NPU 整型量化作为第一性约束纳入解码设计，系统求解「精度–鲁棒性–功耗」三难，而非引入新解码架构。', options: { bullet: false, breakLine: true } },
  { text: '为什么站得住（过审稿人「换芯片」拷问）：把 RV1106 换成树莓派/Jetson/GAP9，结论与方法就变——「HBC 体信道噪声 + RKNPU INT8 量化」这两个约束只有这条链路才有。', options: { bullet: false } },
], { x: M + 0.2, y: 3.37, w: W - 0.4, h: 1.1, fontSize: 12, color: TXT, fontFace: 'Arial', valign: 'middle', lineSpacingMultiple: 1.18 });
tbl(s, M, 4.8, ['第一性约束', '实测事实', '对解码的含义'],
  [
    ['HBC 体信道噪声', '46.3dB 损耗 + 5bit 量化，SNR 拐点 8~14dB', '现有 EEG 论文默认理想链路，从不考虑链路退化'],
    ['NPU INT8 量化', 'argmax 一致率 45.83%（288 样本，近随机），已定位 RKNN depthwise 工具链', '端侧加速与精度不可兼得，需显式设计'],
    ['闭环实时性', 'CPU 23.2ms（新权重），端到端余量 31×', '预算内可加「鲁棒性开销」而不超时'],
  ],
  [3.0, 4.8, 4.4], { fontSize: 10.5, rowH: 0.62 });
footer(s);

// ============ 9 约束驱动鲁棒解码：验证结果 ============
s = p.addSlide();
header(s, '二 · 约束驱动鲁棒解码：验证结果（板载实测）', '三件事全落地 + 联合平坦化——全部真实数字，非预测');
card(s, M, 1.85, 12.2, 1.5, '三件事全落地（2026-08-20）', [
  { text: '① 闭环焊死：hbc_realtime_throttle 处理延迟 26.5ms（<250ms 目标，余量 ~9×），无漂移无溢出。', options: { breakLine: true } },
  { text: '② NPU 了断：+44*scale workaround 不全局成立（argmax 一致率 45.83%），NPU 线关闭，解码写死 CPU 23.2ms（新权重）。', options: { breakLine: true } },
  { text: '③ 鲁棒性增强：训练时注入 HBC 噪声（5bit 量化 + 信道 AWGN），噪声下几乎零损失（−0.08pp）。', options: {} },
], CYAN);
tbl(s, M, 3.5, ['训练方式', '干净', 'HBC噪声', 'INT8量化', '联合(叠加)'],
  [
    ['基线（无扰动）', '75.00%', '74.46%', '75.08%', '74.31%'],
    ['噪声增强（--hbc-aug）', '76.16%', '76.23%', '76.23%', '75.69%'],
    ['量化扰动（--quant-aug）', '75.23%', '74.61%', '75.39%', '74.31%'],
    ['联合平坦化（--hbc-aug --quant-aug）', '76.39%', '75.08%', '76.54%', '76.08%'],
  ],
  [3.7, 1.9, 1.9, 1.9, 2.8], { fontSize: 10, rowH: 0.52 });
s.addShape('roundRect', { x: M, y: 6.05, w: W, h: 0.85, rectRadius: 0.06, fill: { color: '1F3350' }, line: { color: GREEN, width: 1 } });
s.addText([
  { text: '核心结论：联合平坦化在 INT8量化（76.54%）+ 联合叠加（76.08%，+1.77pp）双维度最优；单域鲁棒不迁移（噪声增强在联合维度退化到 75.69%），证明量化与噪声是异质扰动，必须联合习得。', options: { bullet: false } },
], { x: M + 0.2, y: 6.15, w: W - 0.4, h: 0.65, fontSize: 11.5, color: TXT, fontFace: 'Arial', valign: 'middle', lineSpacingMultiple: 1.15 });
footer(s);

// ============ 10 三声明核实 ============
s = p.addSlide();
header(s, '三 · 同平台模型对比：会议三声明核实', '全部误记 / 查无出处 —— 真实画像实为 EEGNet 本身');
const claim = [
  ['DeepSleepNet「2500参数/量化<50KB/<0.5ms/72-75%」', '❌ 误记', '真实 DeepSleepNet = Supratak 2017 睡眠分期模型（CNN+BiLSTM，参数量 ~110万），与运动想象解码无关。该画像实际对应量化后的 EEGNet 类模型（本项目 ~1.7K 参数、INT8<50KB）。'],
  ['LeafNet「脉冲/<100参数/HBC带宽0.8%」', '❌ 查无出处', 'arXiv/GitHub/OpenAlex 均无脉冲 EEG LeafNet（仅植物叶片 SWP-LeafNET 与一个视觉语言模型）。SNN+EEG 端侧真实存在，但无 LeafNet、无 <100 参数记录。'],
  ['1gNet「准确率 72-75%」', '❌ 查无出处', '无此模型。72-75% 区间恰落在 EEGNet 的 3/4 类 MI 准确率（75.07%/65.07%）附近，疑为 EEGNet 误记。'],
];
tbl(s, M, 1.85, ['声明', '判定', '核实结论'], claim.map(c => [c[0], c[1], c[2]]), [3.6, 1.6, 7.0], { fontSize: 10, rowH: 1.05 });
s.addText('教训：会议纪要多处模型名/数值记串了；后续 PPT/论文引用模型数据必须以 arXiv/GitHub 原文为准，不再凭记忆。', { x: M, y: 5.3, w: W, h: 0.5, fontSize: 12, color: AMBER, fontFace: 'Arial', italic: true });
footer(s);

// ============ 11 六维对比表 (1/2) ============
s = p.addSlide();
header(s, '三 · 六维对比表（1/2）', '字段：模型 | 任务与准确率 | 参数量 | 量化体积 | 推理耗时 | 平台 | 出处');
tbl(s, M, 1.85, ['模型', '任务与准确率', '参数', '量化体积', '推理耗时', '平台', '出处'],
  [
    ['EEGNet(本项目)', '8ch MI 二分类 75.00%(基线)', '~1.7K', '<50KB/.rknn80KB', '23.2ms CPU(新权重); NPU已了断(45.83%)', 'RV1106', '板载实测'],
    ['EEGNet(标准)', 'Physionet 82.43/75.07/65.07%', '~2.5K', '—', '论文未给', 'GPU', '2004.00077'],
    ['EEGNet缩放(ETH PULP)', '损失0.31%/2.51%', '7.6~15×减', '内存-7.6~15×', 'M4F 101ms / M7 44ms', 'Cortex-M', '2004.00077'],
    ['Q-EEGNet', '4类MI, 8bit损失0.4%', '8bit', '内存-85%', '5.82ms·0.63mJ', 'Mr.Wolf', '2004.11690'],
    ['EEGNet在线学习', '+7.31% 适配', '—', '15.6KB', '14.9ms·0.76mJ', 'GAP9', '2409.00083'],
    ['MI-BMInet', '2类 82.51%', '8bit', '—', '2.95ms·30µJ', 'PULP MCU', '2203.14592'],
    ['EEG-TCNet', '4类 77.35%/83.84%', '极少参数', '—', '未找到', '嵌入式', '2006.00622'],
  ],
  [2.0, 2.7, 1.0, 1.4, 2.3, 1.3, 1.5], { fontSize: 9, rowH: 0.45 });
s.addText('注：「量化体积」多数论文只报相对缩减比或运行时内存足迹，不报绝对 KB；绝对体积能追溯的只有本项目(<50KB)与 FEMBA(~2MB)，其余如实标「—」。', { x: M, y: 6.0, w: W, h: 0.6, fontSize: 10, color: MUT, fontFace: 'Arial', italic: true });
footer(s);

// ============ 12 六维对比表 (2/2) ============
s = p.addSlide();
header(s, '三 · 六维对比表（2/2）', '续：Riemannian / SNN / Mamba / 生物识别 / 经典模型');
tbl(s, M, 1.85, ['模型', '任务与准确率', '参数', '量化体积', '推理耗时', '平台', '出处'],
  [
    ['Riemannian(非DL)', '4类 75.1%/76.4%', '混合精度', '—', '16.9ms·198µJ; 85µW', '低功耗MCU', '2112.10026'],
    ['癫痫SNN', 'ictal 93.3%/92.9%', 'SNN(LIF)', '—', '87.4µW+287.9µW', 'Xylo', '2410.16613'],
    ['癫痫DL', '96.7%/97.6%, <0.5%损失', '压缩+量化', '—', '通用MCU可运行', '通用MCU', '2012.00307'],
    ['FEMBA(Mamba)', 'TUAB AUROC 0.893, 2bit', '2bit', '~2MB(-74%)', '1.70s/5s; 27×少FLOPs', 'GAP9', '2603.26716'],
    ['GAPses', '生物识别 98.87%/99.86%', '—', '—', '121µJ/推理; 12.4mW', 'GAP9', '2406.07903'],
    ['ShallowConvNet', '2a 4类 67.48%', '~31.6K', '—', '未找到端侧', 'CPU/GPU', 'Schirrmeister17'],
    ['EEG-Conformer', '2a 78.66%/2b 84.63%', '~147K', '—', '无端侧部署', 'CPU/GPU', 'Song 2023'],
  ],
  [2.0, 2.7, 1.0, 1.4, 2.3, 1.3, 1.5], { fontSize: 9, rowH: 0.45 });
s.addText('出处均为 arXiv 编号（完整链接见「关键出处」页）；ShallowConvNet 出处 braindecode/Schirrmeister 2017 HBM；EEG-Conformer 出处 github.com/eeyhsong/EEG-Conformer (Song et al. TNSRE 2023)。', { x: M, y: 6.0, w: W, h: 0.6, fontSize: 10, color: MUT, fontFace: 'Arial', italic: true });
footer(s);

// ============ 13 RV1106 首例 + 结论 ============
s = p.addSlide();
header(s, '三 · RV1106 先例核查 + 一句话结论', '公开可检索范围：国产 NPU 跑 EEG 神经解码 = 首例');
card(s, M, 1.85, 6.0, 2.5, 'RV1106 / 国产 NPU 先例核查', [
  { text: 'GitHub 搜 rv1106 eeg / rockchip eeg / rknn bci / eeg rknn 均零结果', options: { breakLine: true } },
  { text: 'rknn_model_zoo 只收视觉模型(YOLO/MobileNet)，无生理信号', options: { breakLine: true } },
  { text: '国际 EEG 端侧集中在 Cortex-M4/M7 与 RISC-V(PULP/GAP9)', options: { breakLine: true } },
  { text: '→ 国产 RKNN NPU 路线 EEG 解码无公开先例', options: { color: GREEN } },
], CYAN);
card(s, 6.75, 1.85, 6.0, 2.5, '一句话结论', [
  { text: '本项目 75.00%@RV1106（基线，hold-out 修复后）与 ETH PULP/GAP9 国际一流端侧 EEGNet 同数量级', options: { breakLine: true } },
  { text: '是公开可检索范围内 RV1106 国产 NPU 上 EEG 神经解码首个案例', options: { breakLine: true } },
  { text: '但注意：① 75.00% 是 BCIC2a(9被试)二分类，不可与 Physionet(109被试)~82% 直接比；② NPU INT8 已了断（45.83%），短板在工具链', options: {} },
], GREEN);
card(s, M, 4.6, 12.2, 1.95, '诚实边界（reviewer 会问）', [
  { text: '• 精度绝对值不可跨数据集直接对比（被试规模、通道数、任务类数都不同）', options: { breakLine: true } },
  { text: '• NPU 1.2ms/15× 速度真实，但 INT8 量化精度不达标（argmax 45.83%），与 Q-EEGNet 8bit 仅损失 0.4% 反差鲜明 → 短板是 RV1106 量化工具链对深度可分离小模型支持不足，非模型本身', options: { breakLine: true } },
  { text: '• 唯一可主张「首例」的是：国产 Rockchip RV1106 NPU 跑 EEGNet 神经解码（GitHub/arXiv/OpenAlex 三源 0 命中）', options: {} },
], AMBER);
footer(s);

// ============ 14 MyoText 基本信息 ============
s = p.addSlide();
header(s, '四 · MyoText 精读：基本信息 + 信号来源', '普渡 Sen 组（Shreyas Sen = EQS-HBC Nature Elec 2023 通讯作者）');
tbl(s, M, 1.85, ['项目', '内容'],
  [
    ['标题', 'From Muscle to Text with MyoText: sEMG to Text via Finger Classification and Transformer-Based Decoding'],
    ['作者', 'Meghna Roy Chowdhury, Shreyas Sen, Yi Ding（均 Purdue）'],
    ['出处', 'arXiv 2601.03098（2026-01-06，投 ACM IMWUT/UbiComp）'],
    ['信号', '表面肌电 sEMG（非超声/MRI）—— 双前臂 16ch×2 = 32ch'],
    ['数据', 'Meta emg2qwerty 公开集（108人/1135会话/346h），评估用 30 人'],
  ],
  [1.6, 10.6], { fontSize: 10.5, rowH: 0.62 });
s.addShape('roundRect', { x: M, y: 5.45, w: W, h: 1.2, rectRadius: 0.05, fill: { color: '1F3350' }, line: { color: AMBER, width: 1 } });
s.addText('⚠️ 重要更正：任务背景「MyoText 很可能用超声或 MRI」的猜测是错的——论文标题/摘要/正文反复明确是 sEMG（皮肤贴片电极）。采集：按击键事件切前 1s+后 1s 时间对齐片段，按 QWERTY 指法标「哪根手指按的」（8 指，不含拇指）。', { x: M + 0.2, y: 5.55, w: W - 0.4, h: 1.0, fontSize: 12, color: TXT, fontFace: 'Arial', valign: 'middle', lineSpacingMultiple: 1.2 });
footer(s);

// ============ 15 MyoText 框架+精度 ============
s = p.addSlide();
header(s, '四 · MyoText 分级解码框架 + 精度', '生理学对齐的分级解码：运动 → 语言（三级）');
card(s, M, 1.85, 6.4, 4.15, '三级框架', [
  { text: '① 手指分类：CNN-BiLSTM-Attention 把多通道 sEMG 分 8 指', options: { breakLine: true } },
  { text: '② 字母池化：QWERTY 指法先验(8指→26字母) → 候选词构建', options: { breakLine: true } },
  { text: '③ 句子重建：微调 T5-base + 约束束搜索恢复完整句子', options: { breakLine: true } },
  { text: '部署：split-execution（手指分类器在可穿戴端，T5 在手机/边缘）——契合我们「端侧+回传」架构', options: { color: CYAN } },
], CYAN);
card(s, 7.15, 1.85, 5.6, 4.15, '精度 / 速度', [
  { text: '手指分类：5-fold 85.45%±3.36%；LOUO 81.4%±3.76%', options: { breakLine: true } },
  { text: '句子级：CER 5.4% / WER 6.5%', options: { bold: true, color: GREEN, breakLine: true } },
  { text: '对照 baseline(emg2qwerty)：被试内 6.9% CER，跨被试 51.78%', options: { breakLine: true } },
  { text: '→ MyoText 5.4% 优于端到端基线，贡献在「先分手指再靠语言模型」', options: { breakLine: true } },
  { text: '速度：论文未报 WPM/CPS（未报告）', options: { color: AMBER } },
], GREEN);
footer(s);

// ============ 16 MyoText 局限 ============
s = p.addSlide();
header(s, '四 · MyoText 核心局限', '论文自述 + 客观评价（5 点）');
tbl(s, M, 1.85, ['#', '局限', '说明'],
  [
    ['1', '跨被试泛化差', 'baseline 跨被试 CER 51.78%，MyoText LOUO 也掉到 81.4%；「无标定跨用户泛化是开放问题」'],
    ['2', '强依赖语言先验', '去语言模型后精度大幅退化(11.28%/55.38%)——信号本身信息量不足以独立支撑高精度打字'],
    ['3', 'sEMG 固有噪声/漂移', '电极位置、肌肉疲劳、个体解剖差异 → 跨会话/跨用户不稳定'],
    ['4', '数据依赖大', '仅 emg2qwerty 单数据集，跨数据集泛化(英文语料)仍是初步'],
    ['5', '非实时、无速度指标', '离线评估，未证明在线实时打字可行性'],
  ],
  [0.6, 2.6, 9.0], { fontSize: 10.5, rowH: 0.72 });
footer(s);

// ============ 17 双模态融合+数据集 ============
s = p.addSlide();
header(s, '四 · EEG+EMG 双模态：融合方式 + 数据集', '「双模态打字」作为成熟方向几乎不存在；融合文献集中在动作/语音分类');
s.addText('融合方式（6 类）', { x: M, y: 1.82, w: 4, h: 0.34, fontSize: 13, bold: true, color: CYAN, fontFace: 'Arial' });
tbl(s, M, 2.2, ['层级', '做法', '代表工作'],
  [
    ['早期/信号级', '原始信号或时频特征拼接喂单分类器', '传统默认'],
    ['特征级', '各模态分别提特征再拼接', 'Ben Said 2017'],
    ['中间/深度', '多流网络 + 注意力/跨注意力', 'FAConformer 2024 / CAT-Net 2025'],
    ['决策级', '独立分类器 + 投票/加权/贝叶斯', 'Leeb 2010 / Pfurtscheller 2010'],
    ['概率图模型', '层次图模型建模 EEG/EMG 联合概率', 'Ozdenizci 2018'],
    ['相干性特征', '皮层-肌肉相干作特征', 'Cisotto 2018'],
  ],
  [2.0, 4.2, 4.2], { fontSize: 10, rowH: 0.42 });
s.addText('公开数据集（同时含 EEG + EMG）', { x: M, y: 5.0, w: 5, h: 0.34, fontSize: 13, bold: true, color: CYAN, fontFace: 'Arial' });
tbl(s, M, 5.38, ['数据集', '出处', '规模', '模态', '任务'],
  [
    ['Way-EEG-GAL', 'Luciw 2014 SciData', '12人/3936', 'EEG32+EMG5', '抓握-抬起'],
    ['Jeong2020', 'GigaScience 2020', '25人/82500', 'EEG60+EMG7', '11上肢动作'],
    ['日本语音集', 'Sato 2026', '3人/1020h', 'EEG+面部EMG', '出声语音'],
    ['(对照)emg2qwerty', 'Sivakumar 2024', '108人/346h', '仅sEMG 16ch', 'QWERTY打字'],
  ],
  [2.0, 2.4, 1.6, 2.2, 2.2], { fontSize: 9, rowH: 0.5 });
footer(s);

// ============ 18 双模态代表论文 ============
s = p.addSlide();
header(s, '四 · EEG+EMG 代表论文 + baseline 精度', '注意：无任何公开数据集同时含 EEG+EMG 且任务为「打字」');
tbl(s, M, 1.85, ['论文', '出处/年份', '融合方式', '任务/数据', '精度'],
  [
    ['Leeb', 'EMBC 2010', '决策级', '混合BCI', '定性优于单模态'],
    ['Pfurtscheller', 'Front Neurosci 2010', '综述', 'EEG+EMG框架', '—'],
    ['Cisotto', 'arXiv 1809.03300', '相干特征', '抓握分类', '>0.8'],
    ['Ozdenizci', 'arXiv 1809.05635', '概率图模型', '5手势/双肢', '在线验证'],
    ['Ben Said', 'arXiv 1703.08970', '特征级(自编码)', '情绪标注', '联合压缩+分类'],
    ['FAConformer', 'arXiv 2409.18973', '深度融合', 'Jeong2020 11动作', 'SOTA'],
    ['CAT-Net', 'arXiv 2511.10935', '跨注意力+域对抗', '4声调 20EEG+5EMG', '87.83%/88.08%'],
    ['Wang & Suppiah', 'arXiv 2207.08650', '特征/深度', '上肢动作', '—'],
  ],
  [2.4, 2.6, 2.4, 2.6, 2.2], { fontSize: 9.5, rowH: 0.5 });
footer(s);

// ============ 19 双模态可行性 ============
s = p.addSlide();
header(s, '四 · 可行性判断（作为长期拓展）', '结论：方向成立、性价比存疑 → 建议作技术储备/发论文方向，非近期主线');
card(s, M, 1.85, 4.0, 4.5, '技术门槛：中高', [
  { text: '分级解码思路可迁移', options: { breakLine: true } },
  { text: 'split-execution 契合 RV1106+HBC', options: { breakLine: true } },
  { text: '难点：双套同步采集硬件 + T5 无法塞进 RV1106（须卸载上位机，MyoText 已验证可行）', options: { breakLine: true } },
  { text: 'CAT-Net 证明轻量通道(20EEG+5EMG)可到 87%', options: { color: GREEN } },
], CYAN);
card(s, 4.75, 1.85, 4.0, 4.5, '数据难度：高（最大瓶颈）', [
  { text: '无公开 EEG+EMG 打字数据集，需自采（双设备同步+标定+伦理）', options: { breakLine: true } },
  { text: '公开 EEG+EMG 集都是运动/语音，与打字有 domain gap', options: { breakLine: true } },
  { text: 'emg2qwerty 开源但无 EEG', options: { breakLine: true } },
  { text: '纯 sEMG 打字数据现成、路线清晰', options: { color: AMBER } },
], AMBER);
card(s, 8.95, 1.85, 3.8, 4.5, '复用度：中等', [
  { text: 'EEG 分支直接复用 EEGNet', options: { breakLine: true } },
  { text: 'EMG 分支同构浅层 CNN', options: { breakLine: true } },
  { text: 'HBC 回传层不动', options: { breakLine: true } },
  { text: 'T5 必须落上位机', options: { breakLine: true } },
  { text: '「双模态 vs 纯sEMG 增量收益」无人证明', options: { color: RED } },
], GREEN);
s.addShape('roundRect', { x: M, y: 6.5, w: W, h: 0.75, rectRadius: 0.06, fill: { color: '1F3350' }, line: { color: GREEN, width: 1 } });
s.addText('建议：近期沿 MyoText「分级解码 + RV1106 端侧浅层分类 + 上位机语言模型」用纯 sEMG(emg2qwerty) 先做通；EEG+EMG 双模态作后续差异化发论文方向，先解决「自采同步数据」与「跨被试泛化」，而非追「双模态」标签。', { x: M + 0.2, y: 6.55, w: W - 0.4, h: 0.65, fontSize: 11, color: TXT, fontFace: 'Arial', valign: 'middle', lineSpacingMultiple: 1.1 });
footer(s);

// ============ 20 公司赛道地图 ============
s = p.addSlide();
header(s, '五 · 北京/全国 BCI 公司赛道地图', '8 家全部核实到成立/总部/产品线/背景/技术路线');
tbl(s, M, 1.85, ['公司', '所在地', '产品线', '背景/地位', '技术路线'],
  [
    ['强脑科技 BrainCo', '杭州', '专注/睡眠头环+智能假肢', '行业头部，消费+医疗双线', '非侵入EEG'],
    ['脑陆科技 BrainUp', '北京', '脑健康筛查+脑开放云平台', '成长期，脑开放平台', '非侵入EEG+类脑'],
    ['博睿康 Neuracle', '常州', '科研/临床脑电设备+NEO植入式(2026.3.13全球首款获批)', '医疗级龙头，NEO全球首款', '非侵入EEG+微创植入'],
    ['念通智能', '上海', '脑控外骨骼康复', '上海交大孵化', '非侵入EEG'],
    ['柔灵科技', '杭州', '睡眠贴片+可穿戴', '成长期，柔性电极', '非侵入柔性电极'],
    ['中科搏锐', '北京', '无创脑氧监护', '中科院自动化所孵化', '近红外fNIRS(非EEG)'],
    ['臻泰智能', '西安', '脑控康复机器人', '西安交大孵化', '非侵入EEG'],
    ['云睿智能', '深圳', '消费EEG头环', '待核实', '非侵入EEG'],
  ],
  [2.0, 1.0, 3.4, 3.4, 2.4], { fontSize: 9, rowH: 0.48 });
s.addText('更正常见误区：念通智能实为上海交大孵化(非北京)；中科搏锐是近红外 fNIRS 脑氧监测(非 EEG)。云睿智能无权威企业词条，产品细节待核实。', { x: M, y: 6.55, w: W, h: 0.5, fontSize: 10, color: MUT, fontFace: 'Arial', italic: true });
footer(s);

// ============ 21 侵入式/类脑 + 方向差异化判断 ============
s = p.addSlide();
header(s, '五 · 补充玩家 + 方向差异化判断', '消费头环竞争充分 · 医疗康复门槛高 · 专用芯片门槛极高 → 差异化方向在「类脑×可穿戴边缘」');
card(s, M, 1.85, 5.9, 2.2, '侵入式/半侵入式（医疗级）', [
  { text: '博睿康(NEO 已获批)、脑虎科技(上海,中科院)、微灵医疗、阶梯医疗、应和脑科学(深圳)、NeuraMatrix(北京,清华)', options: { breakLine: true } },
  { text: '均为医疗级、注册周期长、研发投入大', options: {} },
], RED);
card(s, 6.65, 1.85, 6.1, 2.2, '类脑计算/类脑芯片', [
  { text: '灵汐科技(北京,清华类脑芯片)、时识科技 SynSense(深圳/瑞士)', options: { breakLine: true } },
  { text: '与 BCI 结合的「类脑+脑机」交叉赛道玩家极少', options: { color: GREEN } },
], CYAN);
tbl(s, M, 4.3, ['方向', '判断'],
  [
    ['消费级专注/睡眠头环', '竞争充分、同质化（强脑/柔灵/脑陆/回车已密集布局），差异化空间有限'],
    ['医疗康复外骨骼/假肢', '有价值但门槛高（需医疗器械注册证，周期长、临床投入大）'],
    ['脑机接口专用芯片', '相对空白但硬门槛极高（流片+资本，适合作中长期目标）'],
    ['类脑×非侵入可穿戴边缘解码', '最契合的差异化方向：成规模团队极少，与「BCI+类脑+可穿戴」定位对齐，形成「论文↔专利↔产品」闭环'],
  ],
  [3.4, 8.8], { fontSize: 10, rowH: 0.62 });
footer(s);

// ============ 22 竞赛清单 ============
s = p.addSlide();
header(s, '五 · 面向研究生的创业/科技竞赛清单', '报名/截止时间每年变动，备赛以官网当年通知为准');
tbl(s, M, 1.85, ['竞赛', '主办方', '周期', '研究生门槛', '亮点'],
  [
    ['中国国际大学生创新大赛(互联网+)', '教育部等12部委', '4-10月', '在校+毕业5年内', '全球最大双创,研究生组'],
    ['挑战杯·课外学术(大挑)', '共青团中央等', '奇数年', '本硕博', '黑科技展示对前沿硬科技友好'],
    ['挑战杯·创业计划(小挑)', '共青团中央等', '2026偶数年', '研究生可报', '创业BP/路演'],
    ['中国研究生创新实践系列大赛', '教育部学位司+学会', '各主题赛', '专门面向研究生', 'AI/电子设计商业计划书专项=最佳入口'],
    ['集成电路创新创业大赛(集创赛)', '工信部', '1月启动', '研究生/博士(B组)', '芯片/AFE方向最对口'],
    ['海创赛(3类同名需辨明)', '多主办', '3-4月', '海归', '硬科技(需海外背景)'],
    ['AI类大赛(补充)', '华为/百度/阿里/学会', '全年滚动', '多可报', '昇腾/飞桨/天池等'],
  ],
  [3.1, 2.3, 1.5, 1.9, 3.4], { fontSize: 9, rowH: 0.55 });
footer(s);

// ============ 23 竞赛优先级 + 来源 ============
s = p.addSlide();
header(s, '五 · 竞赛优先级 + 来源核实说明', '同一技术底座「类脑 BCI 可穿戴」反复包装、分层投稿');
s.addShape('roundRect', { x: M, y: 1.85, w: W, h: 1.15, rectRadius: 0.06, fill: { color: '1F3350' }, line: { color: GREEN, width: 1 } });
s.addText('优先级建议（结合「边读博边创业」）：研创赛（人工智能/电子设计商业计划书专项）→ 互联网+（研究生组）→ 挑战杯小挑(2026) → 集创赛（若走芯片方向）。四者都允许研究生参赛，可用同一套技术底座产出「论文/专利 + 竞赛奖项 + 创业 BP」三份成果——正是说服导师「以创促研」的抓手。', { x: M + 0.2, y: 1.95, w: W - 0.4, h: 0.95, fontSize: 12, color: TXT, fontFace: 'Arial', valign: 'middle', lineSpacingMultiple: 1.2 });
tbl(s, M, 3.3, ['核实状态', '内容'],
  [
    ['✅ 可核实', '强脑/脑陆/博睿康/念通/柔灵/中科搏锐/臻泰——均从维基百科/百度百科核实到成立/总部/创始人/产品线/融资'],
    ['✅ 竞赛来源', '百度百科「中国国际大学生创新大赛」「研创赛」「集创赛」「挑战杯」系列词条'],
    ['⚠️ 部分无法核实', '云睿智能无权威企业词条；念通/臻泰公开融资未核实；各竞赛2027届精确截止日以官网为准'],
    ['来源访问方式', 'Google/百度/搜狗均反爬，最终以「维基百科+百度百科」为权威来源，无法核实项已显式标注'],
  ],
  [2.2, 10.0], { fontSize: 10, rowH: 0.62 });
footer(s);

// ============ 23A 以创促研：实验室数据瓶颈 ============
s = p.addSlide();
header(s, '六 · 实验室的隐性瓶颈：真实脑电数据', '我的博士论文 = 端侧脑电解码 + HBC 闭环；它的瓶颈不是算法，是数据——实验室采不到真实场景数据');
card(s, M, 1.85, 6.0, 2.1, '自采数据的局限', '实验室范式：场景单一、受试者同质（学生/同事）、时长短，离真实可穿戴场景很远。', AMBER);
card(s, 6.75, 1.85, 6.0, 2.1, '公开数据集的困境', 'BCIC 2a 等公开集已被全球刷爆、无首发空间；且都是「理想链路」数据，不含真实体信道噪声。', AMBER);
card(s, M, 4.1, 12.2, 1.5, '清华/博睿康的根因：临床级真实数据通道', '博睿康能持续发顶刊，根因不是有钱，是手握 NMPA 三类证（2026.3.13 获批）——三类证 = 临床数据合法性 = 能合法采到真正能发论文的数据。这就是数据通道的价值。', GREEN);
s.addShape('roundRect', { x: M, y: 5.8, w: W, h: 0.85, rectRadius: 0.06, fill: { color: '1F3350' }, line: { color: CYAN, width: 1 } });
s.addText('宋老师，我的博士论文不缺算法，缺的是数据。而数据不在实验室里，在真实用户的头上。', { x: M + 0.2, y: 5.9, w: W - 0.4, h: 0.65, fontSize: 13, bold: true, color: CYAN, fontFace: 'Arial', valign: 'middle', align: 'center' });
footer(s);

// ============ 23A2 以创促研：数据放大器 ============
s = p.addSlide();
header(s, '六 · 以产品养数据：给实验室装一台数据放大器', '不是「卖产品赚钱」，是「用产品触达真实用户，把数据引回实验室」');
card(s, M, 1.85, 12.2, 1.4, '为什么是端侧 + HBC 闭环', 'RV1106（¥338 国产端侧）+ EQS-HBC 体信道回传，天然适配可穿戴/居家长时程场景——正好采到「真实体信道噪声 + 真实用户」数据，这是实验室范式采不到的。', CYAN);
s.addShape('roundRect', { x: M, y: 3.4, w: W, h: 2.0, rectRadius: 0.05, fill: { color: '0E1A2E' }, line: { color: CARD2, width: 0.5 } });
s.addText('反哺闭环', { x: M + 0.2, y: 3.52, w: 4, h: 0.34, fontSize: 13, bold: true, color: CYAN, fontFace: 'Arial' });
s.addText([
  { text: '产品采集（真实场景、长时程、体信道噪声） → 数据回流归档进实验室 → 直接喂论文实验章节（真实数据实证） → 论文更扎实 → 反哺产品迭代', options: { bullet: false, breakLine: true, paraSpaceAfter: 6 } },
  { text: '一个闭环，每一步都产出学术成果，而不是消耗实验室资源。', options: { bullet: false } },
], { x: M + 0.2, y: 3.9, w: W - 0.4, h: 1.4, fontSize: 13, color: TXT, fontFace: 'Arial', valign: 'top', lineSpacingMultiple: 1.3 });
s.addShape('roundRect', { x: M, y: 5.7, w: W, h: 0.85, rectRadius: 0.06, fill: { color: '1F3350' }, line: { color: GREEN, width: 1 } });
s.addText('这台设备对实验室的价值，不在于能卖多少钱，而在于能带回多少实验室里采不到的数据。', { x: M + 0.2, y: 5.8, w: W - 0.4, h: 0.65, fontSize: 13, bold: true, color: GREEN, fontFace: 'Arial', valign: 'middle', align: 'center' });
footer(s);

// ============ 23A3 以创促研：学术优先权 ============
s = p.addSlide();
header(s, '六 · 学术优先权锁定：先发论文，再谈转化', '所有创新点的首发权、专利、数据都锁在实验室');
card(s, M, 1.85, 6.0, 2.1, '首发权在实验室', '创新点由实验室锁定，优先投 IEEE TBME / JNE / Sensors / BSN。板载实测基线就是底气。', GREEN);
card(s, 6.75, 1.85, 6.0, 2.1, '专利是职务发明', '学校/实验室为权利人、学生为发明人——是「成果转化」前置，不是「创业」前置。', GREEN);
card(s, M, 4.1, 6.0, 2.1, '数据治理', '数据归档进实验室，作为公共科研资产。「数据进实验室、成果上期刊」——不垄断、不外流。', CYAN);
card(s, 6.75, 4.1, 6.0, 2.1, '已板载实测（首发权底气）', 'hold-out 基线 75.00% → 联合平坦化：干净 76.39% / INT8 76.54% / 联合叠加 76.08%；26.5ms、5bit、NPU 写死。', CYAN);
s.addShape('roundRect', { x: M, y: 6.35, w: W, h: 0.65, rectRadius: 0.06, fill: { color: '1F3350' }, line: { color: GREEN, width: 1 } });
s.addText('转化只是顺带，学术优先权永远是第一位。', { x: M + 0.2, y: 6.42, w: W - 0.4, h: 0.5, fontSize: 12.5, bold: true, color: GREEN, fontFace: 'Arial', valign: 'middle', align: 'center' });
footer(s);

// ============ 23B 以创促研：成果转化职称 ============
s = p.addSlide();
header(s, '六 · 成果转化：国家级项目与职称的硬加分项', '这件事的终点不是一家公司，是一批论文、专利、成果转化项目');
card(s, M, 1.85, 6.0, 2.1, '职称与学科评估认什么', '成果转化、产学研合作是双一流、学科评估、职称评审的显性指标——论文、授权专利、成果转化合同，才是这里认的东西。', GREEN);
card(s, 6.75, 1.85, 6.0, 2.1, '双轨落地', '纵向（国家重点研发/自然科学基金）+ 横向（企业合作 + 成果转化合同），都是「成果转化业绩」。', CYAN);
card(s, M, 4.1, 6.0, 2.1, '学生差异化 = 实验室加分', '产学研项目 = 技术发明奖/科技进步奖的前置积累 + 职称材料。学生的差异化与实验室加分是同一件事。', CYAN);
card(s, 6.75, 4.1, 6.0, 2.1, '角色定位', '在读期间产学研实践：导师是项目负责人、学生是执行人。不是「创业导师」，是课题负责人。', AMBER);
s.addShape('roundRect', { x: M, y: 6.35, w: W, h: 0.65, rectRadius: 0.06, fill: { color: '1F3350' }, line: { color: GREEN, width: 1 } });
s.addText('这些才是职称和学科评估里认的东西。', { x: M + 0.2, y: 6.42, w: W - 0.4, h: 0.5, fontSize: 12.5, bold: true, color: GREEN, fontFace: 'Arial', valign: 'middle', align: 'center' });
footer(s);

// ============ 23B2 以创促研：合作路线图 ============
s = p.addSlide();
header(s, '六 · 合作路线图：以学术里程碑为节点', '把「A 轮到 IPO」改成「第一篇论文到成果转化项目」');
tbl(s, M, 1.85, ['阶段', '博士论文 ↔ 产学研（同一件事的两面）', '产出物'],
  [
    ['Phase 1（已/进行中）', '论文核心方法「约束驱动鲁棒解码」：hold-out 基线 75.00%，联合平坦化 76.39%/76.54%/76.08% → 投稿 TBME/JNE/Sensors', '论文主体 + 第一篇'],
    ['Phase 2', '论文实验章节：端侧闭环 + 真实场景小样本采集 → 真实数据实证 → 论文第二变体', '第二篇 + 首批真实数据'],
    ['Phase 3', '论文结论延伸：专利布局 + 纵向申报 + 横向转化合同', '授权专利 + 成果转化业绩'],
  ],
  [2.6, 6.6, 3.0], { fontSize: 11, rowH: 1.0 });
s.addShape('roundRect', { x: M, y: 5.5, w: W, h: 0.85, rectRadius: 0.06, fill: { color: '1F3350' }, line: { color: CYAN, width: 1 } });
s.addText('论文和产学研是同一个课题的两个面向，相辅相成——您看哪个节点该动，我们就动哪个。产出物是论文、专利、项目，不是融资轮次。', { x: M + 0.2, y: 5.6, w: W - 0.4, h: 0.65, fontSize: 12, bold: true, color: CYAN, fontFace: 'Arial', valign: 'middle', align: 'center' });
footer(s);

// ============ 23B3 以创促研：边界红线 ============
s = p.addSlide();
header(s, '六 · 边界与红线：这是产学研合作，不是办公司', '您只需像指导任何科研课题一样指导这个课题');
card(s, M, 1.85, 6.0, 2.3, '明确不碰', '融资、估值、IPO、独角兽、公司化运作、辞职创业——这些一个字都不提，也不做。', RED);
card(s, 6.75, 1.85, 6.0, 2.3, '学术红线', '不追植入式（需临床资质与伦理）；数据只在实验室内部流动，不进任何商业系统。', RED);
card(s, M, 4.3, 12.2, 1.5, '您的角色', '项目负责人 / 学术把关人——不是「创业导师」，更不是「背书人」或「资源接口」。这不是读博之外的副业，就是我的博士论文课题本身。师生关系不变。', GREEN);
s.addShape('roundRect', { x: M, y: 6.0, w: W, h: 0.75, rectRadius: 0.06, fill: { color: '1F3350' }, line: { color: CYAN, width: 1 } });
s.addText('我只需要您做一件事——像指导任何一个科研课题一样指导这个课题。剩下的数据、论文、专利、项目，都由这个课题自然长出来。', { x: M + 0.2, y: 6.08, w: W - 0.4, h: 0.6, fontSize: 12.5, bold: true, color: CYAN, fontFace: 'Arial', valign: 'middle', align: 'center' });
footer(s);

// ============ 24 下一步任务 ============
s = p.addSlide();
header(s, '七 · 下一步任务（P0–P4）与分工', 'P0 闭环已基本打通（板载实测复现完成），P1-P3 调研已出，落地待推进');
tbl(s, M, 1.85, ['优先级', '任务', '状态', '负责人'],
  [
    ['P0', '闭环打通：数据对齐 + 一键复现 + 约束驱动鲁棒解码', '✅ 鲁棒解码已验证（联合平坦化），复现脚本就绪', '何仡轩'],
    ['P1', '同平台对比确立优势', '✅ 调研完成（RV1106 首例）', '何仡轩'],
    ['P2', 'MyoText 精读 + 双模态打字', '✅ 调研完成（建议先纯 sEMG）', '何仡轩'],
    ['P3', '北京 BCI 公司 + 竞赛调研', '✅ 调研完成（差异化方向已判）', '宋老师+何仡轩'],
    ['P4', '板子实物照片 + 尺寸对比图', '⏳ 待拍（需本人手机拍摄）', '何仡轩'],
  ],
  [1.2, 6.0, 3.4, 1.6], { fontSize: 11.5, rowH: 0.62 });
card(s, M, 5.35, 12.2, 1.35, '长期拓展（下一步之后的下一步）', [
  { text: '① 真实 EQS-HBC 体耦合测量（信号源+电极+示波器）→ 闭环数据自监督（数据回流实验室，支撑下一批论文）', options: { breakLine: true } },
  { text: '② 多模态(EEG+EMG)打字落地 ③ 全身分布式推理网络（多节点端侧芯片协同可穿戴）', options: {} },
], CYAN);
footer(s);

// ============ 25 定位声明 + 诚实边界 ============
s = p.addSlide();
header(s, '七 · 定位声明（诚实版）+ 诚实边界', '首例性红线：不踩「首创」，只主张可辩护的组合 · 投稿定位：IEEE TBME / JNE / Sensors / BSN（非 NeurIPS/ICLR）');
card(s, M, 1.85, 6.0, 2.3, '✅ 可主张（组合无先例）', '公开可检索范围内，首个在国产低成本 NPU（RV1106，¥338）上端侧完成 EEGNet 脑电解码、并以 EQS-HBC 人体信道作回传链路的融合闭环系统。', GREEN);
card(s, 6.75, 1.85, 6.0, 2.3, '❌ 不主张（已有先例）', '不称首创 EQS-HBC（Purdue Sen 组 2018 首创，BP-QBC 上 Nature Electronics 2023）；不称首创端侧 EEGNet（ETH PULP 已系统做过 ARM/RISC-V）。', RED);
card(s, M, 4.35, 6.0, 2.3, '⚠️ 模型 vs 测量', '46.3dB 信道损耗来自 IT\'IS 4-Cole-Cole 参数化模型 + 105pF 返回路径，非真实体表传输测量；真实体耦合测量（信号源+电极+示波器）是下一步。', AMBER);
card(s, 6.75, 4.35, 6.0, 2.3, '✅ 真实站得住的核心', '真实脑电（BCIC2a）驱动的软件信号链在 RV1106 跑通；EEGNet 精确移植纯 C（逐位一致）；量化(5bit)+信道损失小到不影响分类。', GREEN);
footer(s);

// ============ 26 关键出处 ============
s = p.addSlide();
header(s, '附 · 关键出处链接汇总', '全部 arXiv / DOI / GitHub，可逐一追溯');
const links = [
  ['EEGNet 标准/缩放 (ETH PULP)', 'arxiv.org/abs/2004.00077'],
  ['Q-EEGNet (Mr.Wolf)', 'arxiv.org/abs/2004.11690'],
  ['EEGNet on-device 学习 (GAP9)', 'arxiv.org/abs/2409.00083'],
  ['MI-BMInet', 'arxiv.org/abs/2203.14592'],
  ['EEG-TCNet', 'arxiv.org/abs/2006.00622'],
  ['Riemannian (Sub-100µW)', 'arxiv.org/abs/2112.10026'],
  ['癫痫 SNN @Xylo', 'arxiv.org/abs/2410.16613'],
  ['癫痫 DL @通用MCU', 'arxiv.org/abs/2012.00307'],
  ['FEMBA (Mamba 2bit, GAP9)', 'arxiv.org/abs/2603.26716'],
  ['GAPses (GAP9)', 'arxiv.org/abs/2406.07903'],
  ['DeepSleepNet (睡眠分期,澄清)', 'arxiv.org/abs/1703.04046'],
  ['DeepSleepNet-Lite', 'arxiv.org/abs/2108.10600'],
  ['EEG-Conformer', 'github.com/eeyhsong/EEG-Conformer'],
  ['MyoText', 'arxiv.org/abs/2601.03098'],
  ['emg2qwerty 数据集/基线', 'arxiv.org/abs/2410.20081'],
  ['Way-EEG-GAL', 'doi.org/10.1038/sdata.2014.47'],
  ['Jeong2020 数据集', 'doi.org/10.1093/gigascience/giaa098'],
  ['日本 EEG-EMG-audio 语音集', 'arxiv.org/abs/2606.01264'],
  ['FAConformer', 'arxiv.org/abs/2409.18973'],
  ['CAT-Net', 'arxiv.org/abs/2511.10935'],
  ['Leeb 2010 hybrid-BCI', 'doi.org/10.1109/iembs.2010.5626233'],
  ['Pfurtscheller 2010', 'doi.org/10.3389/fnpro.2010.00003'],
  ['Cisotto 2018', 'arxiv.org/abs/1809.03300'],
  ['Ozdenizci 2018', 'arxiv.org/abs/1809.05635'],
];
const half = links.slice(0, 12);
const half2 = links.slice(12);
tbl(s, M, 1.85, ['文献', '出处'], half, [2.6, 3.4], { fontSize: 8.5, rowH: 0.38 });
tbl(s, M + 6.35, 1.85, ['文献', '出处'], half2, [2.6, 3.4], { fontSize: 8.5, rowH: 0.38 });
footer(s);

p.writeFile({ fileName: '/Users/heyixuan/BCI/20260820/HBCxBCI_板载实测与下一步调研_20260820.pptx' }).then(f => console.log('OK ->', f));
