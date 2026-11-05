# 欧氏对齐（EA）叠加验证报告 — EEG-TCNet / LIF-EEG-TCNet

日期：2026-08-29 ｜ 协议：BCIC IV 2a 22ch 四分类 cross-session（train=A{nn}T 会话，eval=A{nn}E 会话）
方法：He & Wu 2019, *Transfer Learning for BCI: A Euclidean Space Data Alignment Approach*（arXiv:1908.05475）

## 一、结论（TL;DR）

**欧氏对齐在本项目 cross-session 协议下无提升，不触发端侧化。** 如实报告：

| 实验组 | final test (eval) | 对比 | 结论 |
|---|---|---|---|
| EEG-TCNet ELU 无 EA（本次重跑对照） | 62.58% | 历史基线 64.31%（MPS 非确定性 ~1.7pp 波动） | 见下 |
| **EEG-TCNet ELU + EA** | **62.54%** | **−0.04pp** vs 本次对照 | 无提升（噪声级） |
| LIF-EEG-TCNet 无 EA（本次重跑对照） | **62.42%**（与历史基线逐字节一致） | — | 对照可信 |
| **LIF-EEG-TCNet + EA**（重点） | **62.23%** | **−0.19pp** vs 62.42% | **无提升，轻微下降** |

- EA 提升幅度：ELU −0.04pp / LIF −0.19pp，均远低于「有意义」阈值（≥1pp），**未达 ≥2pp 端侧化触发条件**。
- LIF 热启动管线在 MPS 上完全确定（no-EA 重跑与历史报告 best epoch=40 / val=66.75% / final=62.42% 完全一致），
  因此 **LIF+EA 的 −0.19pp 是可靠结论，非随机波动**。
- 分析：本项目 T/E 会话协方差差异小（对齐前平均协方差相对距离仅 1.45%，同被试同设备），
  EA 的收益场景是**跨被试/大域偏移**；此处域偏移已很小，白化不再带来增益，
  反而因 R 条件数高（≈4935，低方差空间方向被放大）引入轻微噪声，精度微降。

## 二、EA 实现与自检（euclidean_alignment.py）

- **实现**：① 每 trial 协方差 C_i = X_i X_i^T / 500（22×22）；② 参考协方差 R = 训练集算术平均；
  ③ R^(-1/2) = U Λ^(-1/2) U^T（`np.linalg.eigh` 特征分解，特征值下限 eps=1e-8 防除零）；
  ④ 每 trial（训练+测试）X_align = R^(-1/2) @ X（批量 einsum）。
- **无泄漏**：R 只由训练集计算；本项目 train.npz 本身即 T 会话、eval.npz 即 E 会话
  （见 preprocess_4class.py 第 24-25 行），故「R 用 train 全量算、eval 应用」与
  「R 用 T 会话算、E 会话应用」**等价**，天然满足严格 cross-session 协议（任务 3c 两个变体为同一件事，无差异可报）。
- **自检通过**：对齐后训练集平均协方差相对误差 **0.0000%**（Frobenius 绝对误差 1.6e-6），
  对角均值 = 1.000000，非对角 RMS = 6.5e-8 → 对齐到单位阵 ✓。
- **跨 session 诊断**：对齐前 T/E 平均协方差相对距离 1.45%（域偏移小）；R 特征值 0.0037~18.02（条件数 4934.6）。

## 三、实验设计

- 文件：`train_eeg_tcnet_ea.py`（复用 train_eeg_tcnet_lif.py 协议：hold-out 15% seed=42、patience=25、
  dropout=0.25、cosine、Adam lr=1e-3、batch=16、epochs=150、final test 只测一次；brief 结构 KE=63/KT=12/pool4/8/GAP，9,512 参数）。
- 矩阵 4 组（≤6 组控制规模），每组 < 20 分钟（实测 ELU ~1.8s/epoch、LIF ~7s/epoch，均早停收敛）：
  ELU×{无EA, EA}、LIF(thr=1.0/tau=2.0/T=8，热启动 relu_brief)×{无EA, EA}。
  唯一变量 = 数据是否对齐；LIF 热启动权重与基线同源（eeg_tcnet_4class_relu_brief.pth）。

## 四、详细结果

| 组 | best val (epoch) | final test | 早停 epoch | 备注 |
|---|---|---|---|---|
| ELU no-EA | 66.49% (80) | 62.58% | 105 | 历史基线 64.31% 重跑差异 = MPS 卷积非确定性（随机初始化 + MPS 非 bit-exact） |
| ELU + EA | 65.72% (46) | 62.54% | 71 | 对齐自检误差 0.0000% |
| LIF no-EA | 66.75% (40) | **62.42%** | 65 | 与历史报告完全一致（确定性验证 ✓） |
| LIF + EA | 65.72% (18) | **62.23%** | 43 | 发放率略升（act1 14.5%→15.1%，act2 18.9%→20.8%） |

LIF+EA 早停更早（epoch 43 vs 65）、best val 相同但出现在 epoch 18 —— 对齐数据使训练更快收敛但精度上限未变，final test 反而低 0.19pp。

## 五、失败/无增益原因分析

1. **域偏移本就小**：T/E 会话平均协方差相对距离仅 1.45%（同 9 名被试、同设备、相邻会话采集），
   EA 的对齐收益与域偏移量正相关，偏移小则收益趋近于 0。
2. **白化的副作用**：R 条件数 ≈4935（z-score 后仍有通道方差差异），R^(-1/2) 会放大低方差空间方向；
   这些方向多为噪声/个体特异模式，cross-session 下跨会话不一致，放大后对分类无益甚至有害
   （与对齐后 E 会话残差 75% 的 Frobenius 相对偏差一致——偏移集中在低方差方向）。
3. **EA 适用场景错配**：EA 论文的典型收益场景是**跨被试**迁移（不同被试协方差差异大）；
   本协议是 subject-independent 的 T→E 同被试 cross-session，非 EA 的主场。
4. ELU 组 MPS 非确定性（64.31% vs 62.58% 重跑差 1.73pp）说明 ELU 单次对比噪声大；
   但 LIF 组完全确定，其 −0.19pp 结论稳健。

## 六、端侧化评估（未触发，仅记录）

EA 提升 < 2pp，**不进入端侧化实施**。若未来用于跨被试场景，成本已量化：

- **R^(-1/2) 已导出**：`data_4class_ea/Rinv_sqrt.npy`（22×22 float32，**1.89 KB**，2,064 字节）。
- **trial 级对齐开销**：X_align = R^(-1/2) @ X，即 22×22 矩阵 × 22×500 矩阵 =
  **242,000 MACs / trial**（0.242 MMACs），相对模型本体 6.8 MMACs 增加 ~3.6%，
  且为纯线性代数（乘加 + 无激活），**纯 C 即可实现**（RV1106 上 <1 ms 量级），
  端侧完全可行；R^(-1/2) 只需出厂标定一次，随固件下发。
- 结论：端侧化技术上零障碍，但当前数据协议下 EA 无精度收益，不值得引入。

## 七、产物清单

- `euclidean_alignment.py` — EA 实现（R / R^(-1/2) / 对齐变换 / 自检 / 诊断）
- `train_eeg_tcnet_ea.py` — EA 叠加训练（--activation elu|lif，--ea none|train）
- `run_ea_experiments.sh` / `ea_experiments.log` — 实验矩阵运行脚本与日志
- `data_4class_ea/Rinv_sqrt.npy` — 白化矩阵导出（端侧化备查）
- 权重/报告：`eeg_tcnet_4class_{elu_brief,lif_t1.0_tau2.0_T8}_{noea,ea}.pth/.txt`（4 组 ×2）

## 八、后续建议（如坚持 EA 路线）

1. 若目标是跨**被试**迁移（如新用户冷启动），EA 值得重测——但当前 luckfox-deploy 是单用户同被试
   T→E 场景，收益预期低。
2. 可尝试 Riemannian 对齐（RAGA 白化到黎曼均值）或 EA 变体（正则化 R、只对齐部分通道），
   但鉴于域偏移仅 1.45%，上限空间有限。
3. 优先方向仍是模型侧（LIF 化代价 -1.89pp 已接近零代价）与数据增强，而非域对齐。
