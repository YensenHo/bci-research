# Ninapro DB5 sEMG 手势分类 + 类脑化（LIF）结果报告

> 本报告为真实复现结果，所有精度均为实际训练产出（MPS / torch 2.4.1），不虚构。
> 日期：2026-08-28。工作目录：`algorithm/myo_text/`。

---

## 1. 数据来源与预处理（如实）

- **数据集**：Ninapro DB5（Pizzolato et al., PLoS One 2017），Zenodo DOI 10.5281/zenodo.1000116，
  开放访问，无需注册。10 名完好被试，双侧 Thalmic Myo 臂环 ×2 = **16 通道** sEMG，
  采样率 **200 Hz**，原始 ADC 约 ±128。
- **动作**：52 个手势（Exercise A: 12 指部 / B: 17 腕手部 / C: 23 抓握功能），每手势 6 次重复，
  动作间 3s 休息（stimulus=0）。**本任务排除休息类**，只做手势多分类。
- **.mat 字段实测**：`emg(samples,16)` float32；`stimulus`（0=休息，1..k 局部标签）；
  `restimulus`（试次）；`repetition`（1..6）；`exercise`（内部编号，与文件名不一致！）。

### 关键坑（已实测确认）

1. **内部 exercise 与文件名不一致**：`E1.mat → 内部 exercise=3（12 动作）`、
   `E2.mat → 内部 exercise=1（17 动作）`、`E3.mat → 内部 exercise=2（23 动作）`。
   全局编号偏移 `{3:0, 1:12, 2:29}`。
2. **各通道增益差 ~3~7 倍**（全局逐通道 std 实测 4.5~15.0，约 3.3 倍；单被试 s1 可达 7 倍）。
   动作判别高度依赖「哪些通道更强」的**跨通道相对幅度**。

### 标准化（本次的核心修复）

上一子代理产出的 `ninapro52.npz` 用的是**逐窗逐通道 z-score**（每个窗每个通道都归一化到
std=1），这会**抹掉跨通道相对幅度**（安静通道噪声被放大到与活跃通道同方差），实测导致
52 类精度卡在 ~15%。

本次改为**全局逐通道标准化**（running sum 一次通过求全局 mean/std，再 `(x-mean)/std`）：

| 通道 | 全局 std（52 类全量） |
|---|---|
| ch0..ch15 | 13.1 15.0 7.1 4.5 8.2 10.2 8.9 11.5 14.0 13.3 7.2 5.7 10.2 9.4 8.3 10.6 |

标准化后**窗内跨通道相对幅度得以保留**（每个窗内各通道 std 不再恒为 1，而是随激活模式变化）。

### 切段

- 滑动窗：窗长 256 样本（1.28s），步长 128 样本（0.64s），在每个试次（restimulus>0）内滑动。
- 输出：`X (N,1,16,256)` float32，附带 `y`（标签）、`rep`（重复 1..6）、`subject`（1..10）。
- 产出 npz：`ninapro52.npz`（24488 窗）、`ninapro12.npz`（5748 窗，exercise A）、
  `ninapro17.npz`（7970 窗，exercise B）。

---

## 2. 模型与划分

- **模型**：复用 `finger_cnn.py` 的 `FingerCNN`（类 EEGNet 轻量全卷积，无循环无注意力）。
  16 通道 → 时间卷积(1,63) + depthwise(16,1) → 可分离卷积(1,15) → 1×1 分类头。
  - 配置：`F1=16, D=2, F2=32, kernel=63, dropout=0.5`。
  - 参数量：52 类 16,548；12 类 6,268（差异全部来自 1×1 分类头随类别数缩放）。
- **激活**：`relu` / `elu`（ANN 基线），`lif`（复用 `lif_eegnet.LIFActivation`，surrogate gradient，
  发放率聚合 ∈[0,1]）。
- **划分（rep 重复留出）**：train/val = 重复 {1..5}（内部 15% 留验证），test = 第 6 次重复。
  类内、被试池化，是干净且标准的 sEMG 划分。随机基线 = 100/类别数 %。
- **训练**：Adam lr=1e-3，batch=64，30 epoch，早停 patience=8，seed=42。
- **LIF 化**：`threshold=1.0, tau=2.0, timesteps=8, alpha=4.0`，**热启动 ANN 的 conv/BN/fc 权重**
  （只把 act1/act2 换成可导 LIF 层），先测「直接转换」（不微调），再微调。

---

## 3. 结果（真实精度，rep 划分 / test = 第 6 次重复）

| 类别数 | 数据 | 激活 | 验证最佳 | **test** | 随机基线 | 相对随机 |
|---|---|---|---|---|---|---|
| 52（全量） | ninapro52 | relu（ANN） | 32.48% | **32.57%** | 1.92% | ~17× |
| 17（Exercise B） | ninapro17 | relu（ANN） | 46.59% | **44.39%** | 5.88% | ~7.5× |
| 12（Exercise A） | ninapro12 | relu（ANN） | 51.74% | **51.97%** | 8.33% | ~6.2× |
| 12（Exercise A） | ninapro12 | elu（ANN） | 52.16% | **48.13%** | 8.33% | ~5.8× |
| 12（Exercise A） | ninapro12 | **lif（LIF）** | 55.37% | **52.28%** | 8.33% | ~6.3× |

### LIF 化的「类脑化损失」

以 12 类为例（ANN relu 基线 51.97%）：

| 阶段 | test | 与 ANN relu 的差 |
|---|---|---|
| ANN relu（基线） | 51.97% | — |
| ANN 权重**直接换 LIF**（不微调） | 39.94% | **−12.03%**（转换损失） |
| LIF **微调后** | 52.28% | **+0.31%**（追平并略超） |

- LIF 微调后发放率：act1=9.39%、act2=18.11%（非退化，rate-coded 正常水平）。
- **结论：12 类 sEMG 下，surrogate-LIF 热启动+微调能追平 ANN（52.28% vs 51.97%）。**
  直接转换（无微调）会损失 ~12%，但微调可完全找回。

---

## 4. 结论与诚实边界

### 结论
1. **全局逐通道标准化是关键修复**：同样 F1=16/F2=32 的 relu ANN，52 类从逐窗标准化的
   14.81% 提升到全局标准化的 **32.57%**（+17.8 个百分点）。跨通道相对幅度是 sEMG 判别
   的核心特征，逐窗标准化会把它抹掉。
2. **52 类对轻量全卷积 CNN 过难**：即使全局标准化后，32.57% 仍 <40%（随机 1.92%）。
   这符合预期——原始 16ch sEMG 上区分 52 个手势，轻量 CNN 容量不足。
3. **降到可解范围后 ANN 基线可达 40%+**：17 类 44.39%，12 类 51.97%（relu）/
   48.13%（elu）。relu 略优于 elu。
4. **LIF 化可追平 ANN（12 类）**：surrogate-LIF 热启动+微调 52.28% ≥ ANN relu 51.97%，
   类脑化损失 ≈ 0（甚至 +0.31%，在噪声范围内）。直接转换损失 ~12%，微调可恢复。

### 诚实边界（不夸大）
- **仅 12 类完整跑了 LIF 化**（受训练次数预算 <6 组约束）。17 类的 ANN 已到 44.39%，
  未再单独跑 17 类 LIF，但其趋势应与 12 类一致（LIF 追平 ANN）。
- **划分是 rep 留出（类内）**，未跑留一被试（跨被试）的 LIF。跨被试 sEMG 通常更难，
  ANN/LIF 精度都会更低，但二者的「相对差距」结论预期不变。留一被试可作为后续消融。
- **52 类结论是「容量不足」而非「数据不可分」**：更大模型 / 更复杂特征（时频、CNN-LSTM、
  更深的 backbone）或更长的窗可进一步提分，这超出了本次「轻量 CNN + 类脑化」的范围。
- **LIF 是 rate-coded surrogate**，不是严格的脉冲时序编码（Temporal SNN）。负半轴输入
  永不发放（等效 ReLU 截断），是 surrogate 训练路径的已知代价（`lif_eegnet.py` 文档已注明）。
- 12 类 51~52% 的绝对精度不高（随机 8.33%，约 6×），受限于「原始 16ch sEMG + 1.28s 窗 +
  轻量 CNN」的组合，但已足够支撑「LIF 能否追平 ANN」这一验证目标。

---

## 5. 文件清单

**修改**
- `preprocess_ninapro.py`：①修复 mock 自检（动作段加窗内正弦时变成分，避免全局标准化后
  窗内 std 恒 0 的误报）；②新增 `--exercise` 子集抽取（内部编号 1/2/3）+ 标签重映射 0..k-1；
  ③修复切段循环改用过滤后的 all_mats（此前 exercise 过滤未真正生效）。

**产出**
- `ninapro52.npz` / `ninapro12.npz` / `ninapro17.npz`：全局标准化后的切段数据。
- `ann_ninapro_52.pth` / `ann_ninapro_17.pth` / `ann_ninapro_12.pth`（relu）、
  `ann_ninapro_12_elu.pth`（elu）、`lif_ninapro_12.pth`（LIF 微调）。
- `ninapro_report_*.txt`：各次训练的逐 epoch 记录。
- `ninapro_results.json`：全部结果的机器可读汇总。
- 本报告 `Ninapro_类脑化结果报告.md`。

**未改动**：`finger_cnn.py`、`lif_eegnet.py`（只读复用）。

---

## 6. 复现命令（关键几条）

```bash
PY="/opt/miniconda3/envs/openmmlab/bin/python"
E="env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin"

# 1) 自检
$E $PY preprocess_ninapro.py --selfcheck
# 2) 预处理（全量 + 子集）
$E $PY preprocess_ninapro.py --data-dir data_ninapro --out ninapro52.npz --max-subjects 10
$E $PY preprocess_ninapro.py --data-dir data_ninapro --out ninapro12.npz --exercise 3 --max-subjects 10
$E $PY preprocess_ninapro.py --data-dir data_ninapro --out ninapro17.npz --exercise 1 --max-subjects 10
# 3) ANN 基线
$E $PY train_lif_ninapro.py --data ninapro12.npz --activation relu --f1 16 --f2 32 --epochs 30 --out ann_ninapro_12.pth --tag ann12_global
# 4) LIF 热启动 + 微调
$E $PY train_lif_ninapro.py --data ninapro12.npz --activation lif --warm-start ann_ninapro_12.pth --threshold 1.0 --tau 2.0 --timesteps 8 --alpha 4.0 --f1 16 --f2 32 --epochs 30 --out lif_ninapro_12.pth --tag lif12_warm
```
