# 多范式系统整合 · 通道无关跨任务基座（设计记录）

> 日期 2026-10-03。导师方向「脑机接口数据的 AI 模型设计 × 可穿戴硬件部署」下的新前沿。
> 本实验要回答的科学问题：**把 7 个脑电范式（运动想象/癫痫/睡眠/情绪/P300 等）的 train 集
> 联合训练一个共享的「通道无关时间骨干」，它学到的通用时间特征能否迁移到下游单任务、
> 超过随机初始化？**

## 为什么这是新机制（不是重开自监督）
- 端侧无监督三线已定案负结果：STDP、masked-AE ×2 架构（channel-collapsing EEGTCNet-AE /
  channel-preserving TemporalConv-AE）全部贴 chance 或负收益。结论「端侧受益于蒸馏，不是自监督」。
- 本实验是**有监督多任务**（用标签），机制与自监督不同：共享骨干从 7 个任务的判别监督里
  学「脑电时间波形的通用判别结构」，而非重建统计结构。这正是技能里「下一步走内部资产化
  （多范式系统整合）」的落地。

## 硬约束（实测盘清）
7 任务四维全不一致：通道 1~32、采样率 100~256Hz、窗长 205~3000 样本、样本量 564~11076。
- 通道数差异 → 靠模型「通道无关」（reshape B*C 共享滤波器，权重与 C 无关）。
- 采样率差异 → 统一 FFT 重采样到 128Hz。
- 窗长差异 → 全局时间池化（长度无关）。
- 类别/样本不均衡 → 逆频类别权重 + 均匀任务抽样。

## 架构
- **TemporalBackbone（共享，通道无关）**：`(B,C,T) → reshape (B*C,1,T) → n 层 Conv1d(1→H / H→H,
  kernel=16) + GeLU → reshape (B,C*H,T)`。权重只取决于 (H,kernel,n_layers)，与 C 无关。
  无 BatchNorm（BN 的 num_features 依赖 C 会破坏跨任务共享；数据已逐试次 z-score）。
- **TaskHead（任务专属）**：`Linear(C*H→128)→ReLU→Linear(128→ncls)`，空间混合 + 分类都在头里。
- **MultiTaskModel**：共享骨干 + 每任务一个头，多任务联合训练。

## 协议（诚实口径）
- 预训练：30 epoch，每 epoch `steps=max(ceil(n_t/batch))` 步、每步均匀抽任务取一批，
  CE loss（逆频类别权重）。
- 下游对照（唯一变量 = 初始化）：baseline（随机骨干从头训）/ frozen（冻结预训练骨干只训头）/
  finetune（微调预训练骨干）。三档同 epochs/lr/batch/seed。
- 泄漏控制：预训练与下游头训练都只用各任务 train 集；eval 是 held-out，全程不碰。
- fixed-epoch 无早停（可部署口径，非 best 回溯）。
- 双 seed（42/123）报均值与两次值。

## 预期与风险
- 预期（若成立）：低数据任务（chbmit eval 564、eegmmidb 540）从跨任务预训练吃到迁移红利，
  frozen ≥ baseline；finetune ≥ frozen。
- 风险（可能又是负结果）：① 骨干太弱（H=8/kernel=16/2 层，有效感受野 ~242ms，抓不到 delta
  慢节律）→ 需要加深/加宽；② 7 任务的判别结构可能根本不共享（各范式频带/空间模式差异大），
  「通用时间滤波器」假设不成立；③ 1296~11076 的样本量对「基座」仍偏小。

## 产物
- `multitask_model.py` / `multitask_data.py` / `pretrain_multitask.py` / `downstream_transfer.py`
- `multitask_cache/`（重采样缓存）、`multitask_backbone_s{s}.pth`、`multitask_transfer_results.json`
