# 竞争方法对比：PRED / SPDIM / ResMF / Gerster(FACED 混杂)

2026-11-17。交接文档第 2 条的交付物：逐篇记录「类别标签、自评、适配阶段、目标数据是否含标签、素材/被试划分、额外模态、计算预算」七项，并给出代码/数据可用性与复现可行性判断。本审查通过 arXiv API、OpenAlex、Crossref、Semantic Scholar、GitHub API 与 bioRxiv/IJCAI 页面逐篇核实，只标注「已核实」的内容；无法核实的一律写明，不伪称完成。

## 一、四篇论文定位与出处（已核实）

| 简称 | 全称 / 出处 | 作者 | 状态 |
|---|---|---|---|
| PRED | "When and How to Adapt: Subject Shifts Detection and Prototype-Guided Correction for Online EEG Decoding"，IJCAI 2026，DOI 10.24963/ijcai.2026/418，全文 PDF 公开 | Shaoqi Zhang, Xi Jin, Xiaojun Ning, Yilin Chen, Jing Wang | 会议论文（已发表，cited 0） |
| SPDIM | "SPDIM: Source-Free Unsupervised Conditional and Label Shift Adaptation in EEG"，ICLR 2025，arXiv 2411.07249 | Shanglin Li, Motoaki Kawanabe, Reinmar J. Kobler | 会议论文（arXiv cited 8） |
| ResMF | **未能定位**：arXiv（ti/全字段）、OpenAlex、Crossref、GitHub 检索均无匹配（返回膜过滤、奥斯曼史等无关结果）。需 GPT 提供全称、DOI 或链接，否则无法对比 | — | 待澄清 |
| Gerster | "Stimulus identity rather than emotion drives EEG classification on the FACED dataset"，bioRxiv 2026-06-16，DOI 10.64898/2026.06.12.731889，CC-BY 4.0 | Moritz Gerster 等（Human Augmented Brain System 员工） | 预印本 |

## 二、七项逐条对比（摘要/页面级已核实；标注「需全文」的未逐字核到）

| 维度 | PRED | SPDIM | Gerster（非方法，基准审计） |
|---|---|---|---|
| **类别标签** | 三数据集分类：FACED 九类情绪、PhysioNet-MI 运动想象、ISRUC 睡眠分期 | EEG BCI（运动想象）+ 睡眠分期，明确覆盖 label shift（类别不平衡） | FACED 九类情绪分类（LinearSVC + CLISA） |
| **自评** | 用 FACED 的**刺激指派情绪标签**（九类），非个人自评回归 | 无自评（BCI/睡眠为客观/半客观标签） | 核心就是对比「刺激指派标签」vs「个人自评」：换自评后准确率下降（发现 2） |
| **适配阶段** | 在线/流式：SSD（policy stability，k=20, δ=0）无监督检测被试偏移 → 软边界触发 PSC 校正 | 离线 source-free 无监督域适配（SFUDA），每目标域学一个 SPD 流形约束参数 | 无适配（有监督） |
| **目标数据是否含标签** | 无标签（UCDA，unsupervised continual DA） | 无标签目标域（且 source-free，源域数据也不可用） | 有标签（有监督） |
| **素材/被试划分** | 跨被试（inter-subject）；摘要未提刺激留出，疑为标准「同刺激跨被试」（需全文核实） | 跨被试 / 跨天域偏移 | 同时批评 intra-subject、cross-subject 与**视频内时间切分**三种设置 |
| **额外模态** | 仅 EEG | 仅 EEG | 仅 EEG |
| **计算预算** | 多粒度原型记忆 + 原型校正；摘要/页面未给 FLOPs（需全文） | 「parameter-efficient」：每目标域仅 1 个 SPD 流形参数，定性极轻 | LinearSVC（轻）+ CLISA（深）；无算力数字 |

## 三、代码与数据可用性（GitHub API 已核实）

| 论文 | 代码 | 数据 | 复现可行性 |
|---|---|---|---|
| PRED | GitHub 公开检索**未找到**官方仓库 | FACED / PhysioNet-MI / ISRUC 均公开 | 需自实现；全文 PDF 公开（ijcai.org/proceedings/2026/0418.pdf），可照方法重写 |
| SPDIM | GitHub 公开检索**未找到** | BCI/睡眠公开数据集 | 需自实现（arXiv 仅摘要+PDF） |
| Gerster | **有**：github.com/moritz-gerster/faced-stimulus-confound | FACED 公开 | **可直接复现**，最高优先级 |

## 四、与本文候选贡献的协议可比性

本文候选（据「第一篇论文_研究问题重新收束」）＝ 连续个人唤醒度自评（0–7）+ 短基线无标签校准（log 功率比）+ 已知内容/群体先验下测 EEG 增量 + 轻量可穿戴（10 通道 band-power + Ridge）。

- **PRED**：任务为**类别分类**且用刺激指派标签，与我们「连续个人自评回归」协议不同；但「无标签在线何时适配」这一核心主张已被它跨三任务覆盖，不能称首创。
- **SPDIM**：解决 label shift 的轻量黎曼修正，与我们「短基线校准」目标相邻但不重叠；「标签偏移下对齐有害 + 参数高效修正」已有先例。
- **Gerster**：直接支撑我们的阴性结果——它证明 FACED 用刺激指派标签做分类会高估，换成个人自评则难。我们项目恰好在用更难的个人自评做回归目标，且当前 EEG 未超常数均值，与 Gerster 的结论方向一致。**这篇最值得复现并作为本文 related work / 反证引用**。
- **ResMF**：无法评估，需先定位。

## 五、结论（诚实）

1. 「无标签何时适配」「标签偏移对齐有害+轻量修正」「FACED 刺激混杂」三个创新点均已被 PRED / SPDIM / Gerster 占据，不能靠术语拼接声称首创——与 GPT 交接结论一致。
2. 本文唯一可能的差异空间是**协议细节组合**：连续个人自评（非类别）+ 短基线无标签校准 + 已知先验下测增量。但拆开看，每个单点都有先例（2017 预测个人主观评分、EMAP 连续评分、静息 EEG 基线可重复性研究、PRISM 通道重要性+半监督适配）。
3. **下一步最该做的是复现 Gerster**（代码公开、数据公开），把它的「自评换标签」分析接到我们的连续唤醒度协议上，看是否一致；这比自实现 PRED/SPDIM 更省、且直接决定本文选题是否成立。
4. **ResMF 需 GPT 澄清**全称，补到本表后再定是否需要对比。
