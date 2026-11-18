# 类脑机制对照：CPU 完整矩阵与 Hermes 交接

## 数据和协议

修正 trial 一基索引后使用 BCIC IV 2a 的 9 名被试、每次会话各 288 试次；T 会话训练、E 会话测试，属于同被试跨会话。15% T 试次用于早停，训练与验证同人同会话且连续记录先双向滤波，早停分数并非独立泛化指标。此前已观察 E 会话，该分析不是全新独立验证。

本机 CPU 的修正窗口 train.npz SHA-256 `1ff83fd8454eb5f85258e572916b5d850b4c6ce1b3cd280ea4ef383c21c98eba`，eval.npz `f4bc9b854322d938915aa2f4eafe8a5f9b6665ccb82e3c84aaf570198372b8ba`。18 个源文件与镜像固定提交的 blob 一致；尚无官方文件哈希。每个种子从头训练一个 ReLU，各模型共用该种子权重；旧 62.42% 权重不可得。

## 测试准确率矩阵（%）

| 变体 | 42 | 123 | 2024 |
|---|---:|---:|---:|
| baseline | 59.18 | 60.07 | 58.60 |
| signed_fixed | 60.38 | 59.72 | 56.40 |
| adaptive_only | 57.06 | 57.99 | 60.15 |
| attention | 58.76 | 58.91 | 57.37 |
| attention_adaptive | 58.37 | 59.57 | 58.18 |

## 逐被试配对：被试内跨种子平均（n=9）

差值为前者减后者，单位百分点；标准差为 9 名被试的样本标准差。先对每名被试的 3 个种子取平均，再做 Wilcoxon 检验，未把种子当独立被试。p 值未经多重比较校正，仅供探索。

| 对照 | 平均差 | 标准差 | 升 / 降 / 平 | Wilcoxon p |
|---|---:|---:|---:|---:|
| signed_fixed_vs_baseline | -0.45 | 3.07 | 4 / 5 / 0 | 0.7891 |
| adaptive_only_vs_baseline | -0.89 | 2.79 | 4 / 5 / 0 | 0.3789 |
| attention_vs_baseline | -0.94 | 1.31 | 2 / 6 / 1 | 0.0547 |
| attention_adaptive_vs_baseline | -0.58 | 1.83 | 3 / 6 / 0 | 0.4961 |
| adaptive_only_vs_signed_fixed | -0.44 | 2.00 | 3 / 6 / 0 | 0.7344 |
| attention_adaptive_vs_adaptive_only | +0.31 | 1.63 | 6 / 3 / 0 | 0.7344 |

每个种子的逐被试差、正负数与检验保存在 `results/bcic2a_stca_cpu_corrected_20261118/paired.json`；各次 summary 包含数据哈希、同种子热启动哈希、样本数和 9 名匿名被试测试准确率。所有 15 组逐被试加权值均与总体结果一致。

## 结论与 Hermes 矩阵的边界

本机 attention_adaptive 相对 baseline 的三种子测试差分别为 −0.81、−0.50、−0.42 个百分点，旧窗口观察到的正向信号未在此独立重训中保持；不能宣称组合稳定有效。adaptive_only 相对 signed_fixed 的三种子差分别为 −3.32、−1.74、+3.74 个百分点，方向不一致，不能把改善归因于自适应阈值。样本数和种子数均少，不能证明任何机制无效。

Hermes 的 Apple MPS 有 15/18 个完整运行（含 3 个 ReLU），缺 attention-123、attention_adaptive-42、attention_adaptive-123。其数据文件 SHA-256 为 train `e808dac3304b4bb1d4a43d4ccc88b54bf268bb35f7aa285b752042680eb1e731`、eval `0a7711eec6760a8408870b93b8b89dc52b896785eee1c62e357757b78d5c3e42`，与本机 NPZ 不同；哈希不同本身不能说明解压后数组是否不同。相同种子的 ReLU 权重哈希也不同，因此不能把本机结果拼进其矩阵。

Hermes 的三个种子均有 adaptive_only 和 signed_fixed，可作不完整矩阵中的探索性比较：先在被试内跨种子平均，差为 +0.89±0.77 pp，7 人升、2 人降，未校正 Wilcoxon p=0.0195；该方向与本机跨种子均差 −0.44 pp 不一致。两者存在模型权重、数据文件和计算后端等差异，不能将差异归因于设备；多重检验、小样本及结果已被查看均禁止据此宣称显著或稳定。

要严格补齐 Hermes 三组，需要原 Mac 保存的相应种子 ReLU 权重和同一份 NPZ；若改在 CPU 训练，应明确标记混合后端而不是宣称 MPS 结果。在 Mac 原环境续跑更直接。不得将权重或 EEG 数据提交到仓库。
