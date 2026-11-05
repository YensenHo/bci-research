# MyoText 精读 + EEG+EMG 双模态打字调研报告

> 调研日期：2026-08-21。所有事实均来自可溯源链接（arXiv / Semantic Scholar / Europe PMC / Crossref），未查到的数值已明确标注。

---

## (a) MyoText 精读笔记

### 论文基本信息
- **标题**：From Muscle to Text with MyoText: sEMG to Text via Finger Classification and Transformer-Based Decoding
- **作者**：Meghna Roy Chowdhury, Shreyas Sen, Yi Ding（三人均属 **Purdue University / West Lafayette**）
  - 邮箱：mroycho@purdue.edu, shreyas@purdue.edu, yiding@purdue.edu（原文脚注）
- **出处**：arXiv **2601.03098**（2026-01-06 提交，v1，25 页/11 表/11 图）；原文标注投 **ACM IMWUT**（Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies，即 UbiComp 系列），目前 Volume 0（在投/待刊）。
- **链接**：https://arxiv.org/abs/2601.03098 （HTML 全文：https://arxiv.org/html/2601.03098v1）
- **与项目关联**：Shreyas Sen 即 EQS-HBC 领域 Nature Electronics 2023（BP-QBC）的通讯作者，SPARC Lab 负责人；Yi Ding 亦为普渡 HBC/BCI 相关研究者。确系会议纪要所指的"普渡 Sen 组肌肉打字论文"。

### 信号来源（重要更正）
- **MyoText 用的是表面肌电 sEMG（surface EMG），不是超声、也不是 MRI。** 任务背景中"MyoText 很可能用超声或 MRI"的猜测是**错的**——论文标题、摘要、正文反复明确是 sEMG（非侵入、皮肤贴片电极）。
- 数据来自 Meta Reality Labs 的 **emg2qwerty** 公开数据集：双前臂 sEMG（**每侧 16 通道，双侧共 32 通道**），带逐击键标注（keystroke 时间戳），在 QWERTY 键盘上触摸打字的同步记录。emg2qwerty 规模：**108 人 / 1135 会话 / 346 小时**（全开源，GitHub: facebookresearch/emg2qwerty）。
- **采集方式**：MyoText 本身不做采集，直接从 emg2qwerty 的 HDF5 日志按每个击键事件切出 **前 1s + 后 1s** 的时间对齐片段（含预备+残余肌电活动），32 通道保持原始分辨率；每个字母按标准 QWERTY 指法打上"哪个手指按的"标签（8 指，不含拇指）。预处理在 Google Colab 上跑了约 15 小时，最终取 **30 名被试**（按多样性与算力可行性筛选）。

### 解码框架（分级/层次化，非端到端）
MyoText 的核心卖点是"生理学对齐的分级解码"，把打字建模为"运动→语言"的层级，共三级：
1. **手指分类（Finger-Level Classification）**：对每个击键片段，用 **CNN–BiLSTM–Attention** 网络把多通道 sEMG 分类为 8 个手指之一（8 类 softmax；CNN 提取局部时空特征，BiLSTM 建模双向时间依赖——预备+残余激活，加 dropout + batch norm）。
2. **字母池化 + 候选词构建（Letter Pooling / Candidate Word Construction）**：利用标准 QWERTY 指法先验（home-row：左手 ASDF、右手 HJKL，8 指→26 字母的映射）把"手指预测"翻译成候选字母，再组合成候选词（缩小解码搜索空间）。
3. **句子重建（Sentence Prediction）**：用**微调的 T5-base 变压器语言模型** + **约束束搜索（constrained beam search，束宽 B）**把候选词序列恢复成完整句子。论文评估了三种 T5-base 变体（基线交叉熵损失等）。
- **部署设计**：采用 **split-execution（拆分执行）**——手指分类器跑在可穿戴端设备，T5 句子解码器跑在配对的手机/边缘处理器上（论文明确说这样能保护隐私、降延迟）。这跟我们"端侧解码 + 回传"的思路高度一致。

### 精度 / 速度
- **手指分类**：5 折交叉验证 **85.45% ± 3.36%**；留一被试交叉验证（LOUO，5 名 held-out）**81.4% ± 3.76%**（Model 1）。头条数字"85.4%"即 5-fold CV。
- **句子级**：**CER 5.4% / WER 6.5%**（字符/词错误率）。
- **对照 baseline（emg2qwerty 原文，Sivakumar et al. 2024）**：端到端 ASR 式模型（时间-深度可分离 CNN，1s 感受野 + 频带旋转不变滤波器）——**被试内个性化 6.9% CER**（+6-gram 语言模型）；**跨被试泛化 51.78% CER**；无语言模型时分别退化到 11.28% / 55.38%。MyoText 的 5.4% 优于该基线，且其贡献在于"先分手指再靠语言模型"的层次化路径。
- **速度**：论文未报告字符/词每分钟（WPM/CPS）的实时输入速度，只报精度。没有"打字速度"指标，需标注为"未报告"。

### 被试规模
- 评估用 **30 名被试**（来自 emg2qwerty 的 108 人子集）。

### 核心局限（论文自述 + 客观评价）
1. **跨被试泛化仍是最大瓶颈**：baseline 跨被试 CER 高达 51.78%，MyoText 的 LOUO 手指分类也掉到 81.4%；句子级跨被试表现未显著解决。论文自己承认"无标定的跨用户泛化是开放问题"。
2. **强依赖语言先验**：去掉语言模型后精度大幅退化（11.28%/55.38%），说明信号本身信息量不足以单独支撑高精度打字。
3. **sEMG 固有噪声/漂移**：电极位置、肌肉疲劳、个体解剖差异导致跨会话/跨用户不稳定。
4. **数据依赖大**：仅基于 emg2qwerty 单数据集，跨数据集泛化（英文语料）做了测试但仍是初步。
5. **非实时、无速度指标**：离线评估，未证明在线实时打字的可行性。

---

## (b) EEG+EMG 双模态文献综述

### 关键结论先行
- **"EEG+EMG 双模态打字"作为一个成熟方向目前几乎不存在**。打字（text entry）领域目前是**单模态各自独立**：sEMG 打字走 emg2qwerty → MyoText 这条线（有大规模公开数据）；EEG 打字走 SSVEP/P300 拼写器（speller）路线。**没有公开的"同时含 EEG+EMG 的打字专用数据集"**。
- EEG+EMG 融合的现有文献集中在**动作/手势分类、抓握力解码、语音/声调解码、康复控制**，尚未延伸到"打字"。

### 融合方式（现有技术路线分类）
| 融合层级 | 做法 | 代表工作 |
|---|---|---|
| 早期/信号级融合 | 原始信号或时频特征直接拼接后喂单分类器 | 传统做法（很多论文默认） |
| 特征级融合 | 各模态分别提特征再拼接 | Ben Said et al. 2017（自编码器联合压缩+分类） |
| 中间/深度融合 | 多流网络 + 注意力/跨注意力交互 | FAConformer 2024（频带注意力+多尺度融合）、CAT-Net 2025（cross-attention） |
| 决策级融合 | 各模态独立分类器 + 投票/加权/贝叶斯 | Leeb et al. 2010、Pfurtscheller 2010（hybrid BCI 经典） |
| 概率图模型融合 | 层次图模型建模 EEG/EMG 联合概率 | Ozdenizci et al. 2018 |
| 相干性特征 | 皮层-肌肉相干（cortico-muscular coherence）作为特征 | Cisotto et al. 2018 |

### 公开数据集（同时含 EEG + EMG）
| 数据集 | 出处 | 规模 | 模态 | 任务 |
|---|---|---|---|---|
| **Way-EEG-GAL** | Luciw et al. 2014, Scientific Data, DOI:10.1038/sdata.2014.47 | 12 人 / 3936 trials | EEG **32ch** + EMG **5 块手臂/手肌肉** + 3D 位置/力 | 抓握-抬起（重量/摩擦变化） |
| **Jeong2020** | Jeong et al. 2020, GigaScience, DOI:10.1093/gigascience/giaa098 | 25 人 / 82500 trials / 3 天 | EEG **60ch** + EMG **7ch** + EOG 4ch | 11 个上肢直觉动作（含运动想象） |
| **日本语音 EEG-EMG-audio** | Sato et al. 2026, arXiv:2606.01264 | 3 人 / 1020 小时 | EEG 62–128ch + 面部 EMG + 音频 | 开放词汇出声语音（面向语音解码） |
| （对照·仅 EMG）**emg2qwerty** | Sivakumar et al. 2024, arXiv:2410.20081 | 108 人 / 346h | sEMG 双侧 16ch（打字） | QWERTY 触摸打字 |

> 注：无任何公开数据集同时含 EEG+EMG 且任务为"打字"。打字数据只有 sEMG（emg2qwerty）；EEG+EMG 数据集都是运动/语音任务。

### 代表论文 + baseline 精度
| 论文 | 出处/年份 | 融合方式 | 任务/数据 | 精度 |
|---|---|---|---|---|
| Leeb et al., "Multimodal Fusion of Muscle and Brain Signals for a Hybrid-BCI" | EMBC 2010, DOI:10.1109/iembs.2010.5626233 | 决策级融合 | 混合 BCI 经典 | 定性优于单模态（未摘具体数） |
| Pfurtscheller, "The hybrid BCI" | Front. Neurosci. 2010, DOI:10.3389/fnpro.2010.00003 | 综述（EEG+EMG 混合框架） | — | — |
| Cisotto et al., "Classification of grasping tasks based on EEG-EMG coherence" | arXiv:1809.03300 (2018) | 皮层-肌肉相干特征 | 抓握分类 | **>0.8** 准确率 |
| Ozdenizci et al., "Hierarchical Graphical Models for Context-Aware Hybrid BMI" | arXiv:1809.05635 (2018) | 概率图模型融合 | 5 手势/双肢 | 会话内+跨会话在线验证 |
| Ben Said et al., "Multimodal deep learning for joint EEG-EMG compression and classification" | arXiv:1703.08970 (2017) | 特征级（自编码器） | 情绪标注信号 | 联合压缩+分类 |
| He et al., "EEG-EMG FAConformer" | arXiv:2409.18973 (2024) | 深度融合（频带注意力+多尺度融合+独立通道卷积） | Jeong2020 11 动作 | 优于现有方法（SOTA，具体数值见论文） |
| Zhuang et al., "CAT-Net: Cross-Attention Tone Network" | arXiv:2511.10935 (2025) | 跨注意力融合 + 域对抗训练 | 4 声调（出声/默声语音），10 人，20EEG+5EMG | **87.83%**（出声）/ **88.08%**（默声）；跨被试 **83.27% / 85.10%** |
| Wang & Suppiah, "Upper Limb Movement Recognition utilising EEG and EMG" | arXiv:2207.08650 (2022) | 特征/深度融合 | 上肢动作康复机器人 | — |
| Sato et al., "1000-hour EEG-EMG-audio dataset" | arXiv:2606.01264 (2026) | 数据集（非方法） | 日本语音 | — |

---

## (c) 可行性判断（作为本项目"长期拓展"）

**结论：方向成立、但性价比存疑，建议作为"技术储备/发论文方向"而非近期主线。** 三点拆解：

1. **技术门槛：中等偏高，但架构可抄作业。** 分级解码（手指→字母→句子）的思路直接可迁移，且 MyoText 的 split-execution（浅层分类在端侧、语言模型在上位机）恰好契合我们"RV1106 端侧解码 + HBC 回传闭环"的架构。难点在于 EEG+EMG 双流融合需要**两套同步采集硬件**（时间对齐、电极布置、跨模态噪声），以及 transformer 级句子解码（T5-base）无法塞进 RV1106 NPU——必须卸载到上位机/手机，这一点 MyoText 已经替我们验证过是可行的。CAT-Net 已证明**轻量通道（20EEG+5EMG）+ 轻量模块就能到 87%**，说明端侧有戏。

2. **数据获取难度：高，是最大瓶颈。** 没有公开的 EEG+EMG"打字"数据集，需**自采**（两套设备同步、标定、伦理），成本远高于单模态。可借用的公开 EEG+EMG 数据（Way-EEG-GAL、Jeong2020、日本语音集）都是运动/语音任务，与"打字"目标存在 domain gap。sEMG 打字的 emg2qwerty（108 人）虽开源但**无 EEG**。相较之下，纯 sEMG 打字方向数据现成、路线清晰。

3. **与现有 EEGNet+RV1106 技术栈的复用度：中等（信号/浅层分类部分高，语言模型部分低）。** EEG 分支可**直接复用已在 RV1106 跑通的 EEGNet**；EMG 分支可用结构类似的浅层 CNN（sEMG 特征与 EEG 时频结构相通，可复用同一套前端/量化部署管线）；HBC 回传层完全不动。但 T5 级语言模型必须落在上位机/云端，且"EEG+EMG 双模态相比纯 sEMG 的增量收益"目前没有论文证明——sEMG 打字已经做到 5.4% CER，比 EEG 拼写器（SSVEP/P300 通常 80–95% 字符准确但慢得多）的输入效率高一个量级，**加 EEG 的边际价值不明**，需要我们自己论证（可能的卖点：EEG 提供运动意图/起止时刻，EMG 提供精细手指动作，二者互补提升跨被试鲁棒性）。

**一句话建议**：近期先沿 MyoText 的"分级解码 + 端侧浅层分类 + 上位机语言模型"路径，用纯 sEMG（emg2qwerty）在 RV1106 上做通，验证架构与 HBC 回传；把 EEG+EMG 双模态融合作为后续的差异化发论文方向，并优先解决"自采同步数据"与"跨被试泛化"两个真问题，而不是追"双模态"这个标签本身。

---

### 附：关键链接汇总
- MyoText：https://arxiv.org/abs/2601.03098
- emg2qwerty 数据集/基线：https://arxiv.org/abs/2410.20081 ；https://github.com/facebookresearch/emg2qwerty
- Way-EEG-GAL：https://doi.org/10.1038/sdata.2014.47
- Jeong2020 数据集：https://doi.org/10.1093/gigascience/giaa098
- 日本 EEG-EMG-audio 语音集：https://arxiv.org/abs/2606.01264
- FAConformer：https://arxiv.org/abs/2409.18973
- CAT-Net：https://arxiv.org/abs/2511.10935
- Leeb 2010 hybrid-BCI：https://doi.org/10.1109/iembs.2010.5626233
- Pfurtscheller 2010：https://doi.org/10.3389/fnpro.2010.00003
- Cisotto 2018 相干分类：https://arxiv.org/abs/1809.03300
- Ozdenizci 2018 概率图模型：https://arxiv.org/abs/1809.05635
