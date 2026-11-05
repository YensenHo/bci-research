# STDP 原型报告 — 单层 LIF + 脉冲时序可塑性在 EEG 上的边界定位

> 日期：2026-09-02 ｜ 引擎：`stdp_core.py`（numpy 单层 LIF + trace STDP）
> 实验：`stdp_synthetic.py` / `stdp_synthetic_dc.py`（合成收敛）、`stdp_eeg.py`（EEG 主表）、
> `dc_eeg_extra.py`（DC-WTA 引擎 EEG 补充灵敏度）
> 数据：合成双模式（A=ch0-9、B=ch20-29 高频泊松）；BCIC IV 2a data_4class
> （2592×22×500×4 类均衡，train=T 会话 / eval=E 会话，cross-session）
> 运行环境：/opt/miniconda3/envs/openmmlab（torch 2.4.1/MPS，本实验纯 numpy），env -i 隔离
> **全部数字为真实运行结果（双 seed），无虚构。** 原始数据：`results/*.json`。

---

## 0. 一句话结论

**单层 STDP 原型的定位边界 =「能学，但挑食」：它必须在「精细竞争机制
（Diehl-Cook WTA：winner 独占 + 共享抑制 + 发放后即时行归一化）+ 输入事件间隔与
膜/可塑性时间常数匹配（稀疏脉冲）」下才收敛；在 EEG 的两种实用脉冲编码
（幅度率编码 / delta 事件编码，~2 事件/步/44 源）上，合成式的条件不成立——
STDP 学到的权重经 LDA 读出在 T→E 跨会话上 ≈ chance（25.1-25.7%），**不超随机权重
基线（27.1-32.8%）**。结论：**当前单层 STDP 无监督方案不是 EEG 端侧无监督的可行
候选**；这不是实现 bug（自检 4/4、合成可收敛），而是「无监督 STDP 需要的时间结构
与 EEG 实用脉冲编码提供的信息结构」之间的本质错配 + 单层读出上限共同所致。**

---

## 1. 实现与自检（stdp_core.py，上一子代理完成，本报告复核）

单层 LIF 脉冲神经元 + trace STDP（numpy，时间驱动，批量并行）：

- **LIF 前向**：V ← decay·V + W@s_in + bias，过阈硬重置（decay=exp(-1/τ_m)），
  与项目既有 LIF 语义一致，但输入为逐时间步 0/1 脉冲。
- **STDP**（trace 法，set-to-1 指数核，最近邻配对）：
  pre 先发→post 后发 = 增强 +A+·exp(-Δt/τ+)；post 先发→pre 后发 = 抑制 −A−·exp(-Δt/τ−)。
  权重裁剪 [wmin, wmax]。
- **竞争/稳态机制（可组合开关）**：
  - `post_topk`：每 trial 每步最多发放 topk 个神经元（廉价的类侧抑制）
  - `thr_adapt`：Diehl-Cook 式自适应阈值 thr + thr_adapt·rate_ema（压高发放神经元）
  - `norm_target`：每神经元权重行 L1 突触归一化（总量受限→通道间取舍）
  - `homeo_target`：synaptic scaling（per-trial 语义）
- 自检 4/4（LIF 前向 / STDP 符号 / trace 与解析 pair 核一致 / pair 单调性）——复核通过：

```
== 自检 1/4：LIF 前向基本行为 ==   [PASS] 无输入不发放 / 强输入每步发放
== 自检 2/4：STDP 符号约定 ==       [PASS] post 先发→抑制 Δw<0 / pre 先发→增强 Δw>0
== 自检 3/4：trace vs 解析 pair ==  [PASS] 增强/抑制幅度与 A±·exp(-Δt/τ) 一致（<1e-9）
== 自检 4/4：pair 单调性 ==         [PASS]
自检结果: 全部通过 ✔
```

---

## 2. 合成验证：收敛条件与失败机制（如实记录）

任务：40 输入通道，模式 A=ch0-9、B=ch20-29 高频泊松（p=0.5/步/通道，其余 0.03 背景），
100 步/trial，A/B 类均衡交替；64 个学习神经元。收敛判据（冻结权重测 120 新 A + 120 新 B）：
权重分化（|w̄A−w̄B|>0.05 的神经元 ≥ 40%）且发放分化（A/B 选择性响应神经元 ≥ 15%）。

### 2.1 第 1 轮：机制组合（stdp_core 批量引擎内 topk/norm/a±/adaptive thr 全组合）

8 配置 × 2 seed = **0/8 收敛**（`results/synthetic_results.json`）：

| 配置族 | w 分化 (A/B sel) | 发放分化 (A/B sel) | 结果 |
|--------|-----------------|-------------------|------|
| topk=1~2, thr_adapt=15~30, norm L1=4~6, a+=0.010~0.015, a−=0.002~0.006, τ−=1~2 | 0~6 | 7~15 | **不收敛** |

诊断（`/tmp/diag_stdp.py` 轨迹）：少数神经元能完美特化（如 wA=0.153/wB=0、
frA=0.098/frB=0），但多数神经元权重**逃逸到低频背景通道**（背景通道均值权重
0.25 而 A/B 判别通道仅 0.03-0.12）。机制根因：

1. **反因果抑制的自噬**：输入事件间隔（0.2 步）≪ τ_minus 时，post 发放后高频
   判别通道的持续 pre 事件在长抑制窗内反复施加 −A−·exp(-Δt/τ−)，把刚增强的
   判别权重吃掉；低频背景通道事件稀疏、几乎不触发抑制 → 权重漂向背景。
2. **无跨模式竞争**：批量引擎的 topk 只限制「同一步」的发放数，神经元可以同时
   被 A 和 B 驱动增强 → 无共享额度 → 中性化（与上一子代理定位一致）。
3. norm 是**乘性**整行缩放，只约束总量、不产生通道选择性——它无法把漂到背景的
   权重推回判别通道。

### 2.2 第 2 轮兜底：Diehl-Cook WTA 配方（独立实现，逐 pattern 串行）

`stdp_synthetic_dc.py`（因为批量近似无法表达「winner 当刻独占 + 其余强抑制 +
发放后即时行归一化」，独立忠实实现，不影响 stdp_core 自检）：
每步最高膜电位且过阈者 = winner 独占发放/更新，winner 当步把同 trial 其它神经元
V 压到 −inh（共享抑制池）；每次发放后该行 L1 立即归一到 norm_target；adaptive
threshold 让发放多者让位（公平性）。**3/6 配置全 seed 收敛**（`results/synthetic_dc_results.json`）：

| 配置（输入口径 hi / a+ / a−） | 每步模式通道事件数 | w 分化 A/B | conv (3 seed) |
|------|------|------|------|
| hi=0.15 / 0.008 / 0.002 | ~1.5 | 29-35 / 27-31 | **✓ 3/3** |
| hi=0.15 / 0.008 / 0.002, dth=0.05 | ~1.5 | 29-32 / 29-32 | **✓ 3/3** |
| hi=0.15 / 0.008 / 0 | ~1.5 | 29-33 / 28-31 | **✓ 3/3** |
| hi=0.15 / 0.004 / 0.003 | ~1.5 | 0/0（发放分化 24-27 但权重幅度 <0.05 判据） | ✗（a+ 小、收敛慢；8 epochs 仍权重不足） |
| hi=0.3 / 0.008 / 0.002 | ~3 | 9-15 | ✗（部分分化不足判据） |
| hi=0.5 / 0.008 / 0.002 | ~5 | 1-2 | ✗（同第 1 轮失败机制） |

收敛配置细节（hi=0.15, seed1）：w 分化 29 A-sel + 31 B-sel / 64，发放互斥
（A-only 22 + B-only 20，双发仅 1），A/B trial 发放率相关 corr ≈ −0.2（负相关=
选择性响应）。5 个独立 seed 全部收敛。

**合成验证结论（诚实）**：
- 纯 STDP + 逐神经元独立稳态机制（第 1 轮全组合）**不能**在两正交模式上收敛——
  需要真正的**跨神经元共享竞争**（WTA/共享抑制）+ **事件级即时归一化**。
- 即使给了 Diehl-Cook WTA，收敛仍要求**输入事件间隔 ≈ τ 量级**（hi=0.15 →
  每步 ~1.5 事件/模式 ≈ τ/事件间隔 平衡）；任务最初口径 hi=0.5（每步 5 事件）
  在任何配方下都不收敛——事件密度淹没 STDP 时间窗。这与文献中 STDP 无监督
  演示普遍使用稀疏输入（MNIST 泊松发放率 ~0-63Hz、事件间隔 ≈ 20ms 量级）一致。

---

## 3. EEG 实验（stdp_eeg.py 主表 + dc_eeg_extra.py 补充）

### 3.1 协议

- **数据**：data_4class T 会话 train 2592（训无监督 + 训读出）、E 会话 eval 2592
  （测，cross-session；与项目监督基线同协议）。
- **编码**（双配置对照，44 源 = 22 通道 × 正/负，都是 0/1 脉冲序列）：
  - `rate`：幅度率编码 u=clip(|x|/θ_c,0,1) 泊松采样，θ_c=8·std_c → 密度 ~5.0%
  - `delta`：差分过阈 |dx|>θ_c，θ_c=0.5·std_c → 密度 ~4.3%
  - 两者事件率均 ≈ 2 事件/步/44 源——落在合成「收敛临界带」附近
    （合成 hi=0.15→1.5 事件/步收敛、hi=0.3→3 事件/步失败）。
- **学习**：类均衡交错喂入单层 STDP（主表 = stdp_core 批量引擎 topk=1 +
  thr_adapt=15 + norm L1=6 + τ−=2 短窗，3 epochs）；读出 = 冻结权重 → 每试次
  发放率向量 (2592, 128) → LDA（sklearn lsqr+shrinkage，T 训/E 测）。
- **对照**：同管线随机权重（同分布未学习）。判据：STDP > 随机即「学到类间信息」。
- 每格 2 seed。chance = 25%。监督参考：ReLU 58.29%（T→E）；delta 1bit 全脉冲
  监督上限 ~41-43%（输入脉冲编码报告）。

### 3.2 结果表（主表 4 配置，真实数字，`results/eeg_results.json`）

| 编码 | 权重 | T→E acc (2 seed) | train acc | min 类 acc | 每 trial 发放率 | W L1/行 |
|------|------|-----------------|-----------|-----------|----------------|---------|
| rate (5.0%) | **STDP** | **25.7% ± 0.1** | 33.2% | 5.0% | 3.75 | 12.3 |
| rate (5.0%) | 随机 | 27.1% ± 0.8 | 38.4% | 5.8% | 2.58 | 4.4 |
| delta (4.3%) | **STDP** | **24.8% ± 1.6** | 37.2% | 5.1% | 2.34 | 12.2 |
| delta (4.3%) | 随机 | 32.8% ± 0.9 | 42.4% | 6.6% | 1.68 | 4.4 |

### 3.3 补充灵敏度：DC-WTA 引擎（`results/eeg_dc_extra.json`）

为隔离「批量引擎 topk 近似损失 vs STDP 机制本身不足」，把合成收敛的 DC-WTA 引擎
（同一配方）直接用于 EEG 脉冲流（每类 300 学习 trial，冻结后同 LDA 协议）：

| 编码 | T→E acc (2 seed) | W L1/行 |
|------|-----------------|---------|
| rate | 25.1% / 25.1% | 5.9 |
| delta | 27.7% / 28.7% | 5.6 |

→ DC-WTA 引擎也没有学出超随机信息（rate ≈ chance；delta 略升但仍低于主表
delta-随机 32.8% 基线）。

### 3.4 EEG 读数解读（诚实）

1. **STDP 未超随机**：两编码下 STDP 权重 LDA 读出 ≈ chance（24.8-25.7%），
   **低于同管线的随机权重（27.1-32.8%）**。随机权重在 delta 编码下反而到 32.8%，
   说明 delta 脉冲流经随机投影 + 发放率聚合仍保留了少量可读的通道统计信息——
   STDP 学习没有把这些信息组织得更可读，反而（权重趋向饱和 L1≈12 vs 随机 4.4）
   损失了投影多样性。
2. **与合成结论自洽**：EEG 编码事件密度 ~2 事件/步/44 源 ≈ 合成临界带；且 EEG
   判别信息主要是频带功率（μ 节律 ERD），单层全连接 STDP 没有时间卷积/频带
   选择结构，只有 8-20 步的短时关联核——它能利用的时间结构（因果共发）
   与 EEG 的判别结构（节律功率的幅度调制）不匹配。
3. **监督基线对照**：同一数据监督 ReLU 58.29%（连续输入）、delta 1bit 全脉冲
   监督 ~41-43%（有标签 + 深度网络 + 卷积空间滤波）。无监督单层 STDP（无标签、
   全连接、发放率读出）25% ≈ chance，差距主要来自「无监督 + 无频带结构单层 +
   脉冲信息瓶颈」三者叠加。

---

## 4. 结论（诚实）

1. **STDP 引擎正确且合成可收敛**（自检 4/4；Diehl-Cook WTA + 稀疏输入下
   3/6 配置、多 seed 稳定收敛，神经元分化成 A/B 互斥 detector）。
2. **收敛条件苛刻且与 EEG 实用配置冲突**：需要跨神经元共享竞争（WTA）、事件级
   归一化、以及「事件间隔 ≈ 可塑性时间常数」的稀疏输入。批量近似引擎（topk 只限
   同步、norm 只限总量）无法产生跨模式竞争——第 1 轮 0/8、hi=0.5 口径 0/3 如实
   记录为不收敛。
3. **EEG 主结论：当前单层无监督 STDP 不是端侧无监督候选**——率/delta 双编码 ×
   批量引擎 × DC-WTA 引擎，T→E LDA 读出均 ≈ chance 或不超随机基线。STDP 学不到
   EEG 判别特征（学到的权重带类间信息 < 随机权重）。
4. **边界定位（不否定整个方向）**：本原型只验证了「单层全连接 STDP + 发放率读出 +
   幅度/delta 线性脉冲编码」。文献中 STDP 无监督在 EEG 上的正结果通常依赖：
   ① 卷积/频带选择前端（把 μ/β 节律功率先提取成稀疏事件）② 更精细的编码
   （如 NEST/阈值自适应，事件密度 25-30% 甜区，见输入脉冲编码报告 §6）
   ③ 多层 + 读出层联合。三者任一缺失，单层 STDP 即退化到 chance——这是本项目
   「端侧无监督可行性」的诚实原型证据，后续若推进应优先补 ①③，而非继续调单层参数。

---

## 5. 产物与复现

目录：`/Users/heyixuan/BCI/20260828/algorithm/stdp/`

| 文件 | 说明 |
|------|------|
| `stdp_core.py` | 引擎：单层 LIF + trace STDP + topk/thr_adapt/norm/homeo（自检 4/4） |
| `stdp_synthetic.py` | 第 1 轮机制组合矩阵（0/8 收敛，如实） |
| `stdp_synthetic_dc.py` | Diehl-Cook WTA 独立实现（3/6 配置收敛） |
| `stdp_eeg.py` | EEG 主表实验（4 配置 × 2 seed，LDA T→E） |
| `dc_eeg_extra.py` | DC-WTA 引擎 EEG 补充灵敏度 |
| `results/synthetic_results.json` | 第 1 轮全部原始数字 |
| `results/synthetic_dc_results.json` | DC-WTA 全部原始数字（含 conv 判据逐 seed） |
| `results/eeg_results.json` | EEG 主表全部原始数字 |
| `results/eeg_dc_extra.json` | EEG DC 补充原始数字 |

复现命令（必须 env -i 隔离）：
```
cd /Users/heyixuan/BCI/20260828/algorithm/stdp
env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin \
    /opt/miniconda3/envs/openmmlab/bin/python stdp_core.py              # 自检
env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin \
    /opt/miniconda3/envs/openmmlab/bin/python stdp_synthetic.py         # 第 1 轮
env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin \
    /opt/miniconda3/envs/openmmlab/bin/python stdp_synthetic_dc.py      # DC-WTA
env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin \
    /opt/miniconda3/envs/openmmlab/bin/python stdp_eeg.py --seeds 1,2   # EEG 主表
```
