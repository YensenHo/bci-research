# LIF-EEGNet 演进路线：类脑计算 × BCI × HBC 的算法接口

> 顶层方针「类脑计算 + BCI + HBC」在算法层的落地文档。
> 定位：不是「新模型 novelty」，而是给现有 EEGNet 解码链路预留**切换 SNN 脉冲类脑模型的路径**，
> 使「约束驱动鲁棒解码」的成果可平滑迁移到神经形态硬件。
> 现状：`board/lif_sim.py`（单神经元）+ `lif_net.c`（scope creep 历史代码），真正 LIF-EEGNet 尚未落地。

---

## 1. 为什么做 LIF-EEGNet（对齐顶层方针）

- **类脑计算**：LIF（Leaky Integrate-and-Fire）脉冲神经元是神经形态芯片（SynSense DYNAP-CNN、Intel Loihi、Rockchip 未来 NPU）的通用计算原语。EEGNet 若停留在 ReLU/ELU 连续激活，无法上神经形态硅。
- **BCI**：EEGNet 已验证是 8ch 小数据 MI 二分类的最优解码器（1650 参数、75% 基线），是 BCI 解码的「甜点模型」，值得作为 SNN 化的第一目标。
- **HBC**：脉冲编码（0/1）天然适合 HBC 低带宽链路——片上 SNN 推理后只传脉冲序列或分类标签，比传浮点特征更省带宽，与「5bit 保精度」的链路约束同源。
- **与 P0-2 NPU 量化的关系**：RV1106 NPU INT8 量化崩坏（42~58%）是「连续激活 + 深度可分离卷积」在低精度整数推理下的固有失配；SNN 的脉冲本来就是 0/1 二值，天然规避量化噪声——**LIF-EEGNet 是 NPU 量化问题的「换轨」解法，而非「修复」**。这是类脑方向对 NPU 困局的战略价值。

## 2. 技术路线：ANN-to-SNN 转换（首选）vs 直接代理训练（备选）

### 2.1 首选：训练代理 ANN → 转 SNN（成本最低，可复用现有权重）

1. **训练代理 ANN**：把 EEGNet 的 `ELU` 换成 `ReLU`（或 `LeakyReLU`），其余结构不变，训一个「ReLU-EEGNet」作为转换源。
   - ⚠️ **负半轴损失是核心难点**：EEG 信号负半轴含判别信息（mu/beta 节律的负向偏转），ELU 保留负值、ReLU 截断负值。直接换 ReLU 预计掉 1~3pp。缓解：①输入加**双通道编码**（正/负分离，正半轴→正脉冲、负半轴→负脉冲）；②或换 LeakyReLU（斜率 α 小）保留少量负值，转换时用带负向的 LIF。
2. **BN 折叠**：推理时把 BatchNorm 折叠进前一层 Conv 权重（现有 `export_c_weights.py` 已做 BN 折叠，复用其折叠逻辑），SNN 里不能留 BN。
3. **池化替换**：AvgPool → 脉冲计数（spike rate 求和）或 MaxPool → 脉冲时间编码。EEGNet 的 AvgPool(1,4)/(1,8) 在 SNN 里用「滑窗内脉冲计数」近似，需验证对时序信息的影响。
4. **阈值标定**：ANN 的 ReLU 激活值 → SNN 的发放率，需逐层标定阈值（threshold balancing），使脉冲发放率近似 ReLU 激活。这是 ANN-to-SNN 转换的标准步骤。
5. **评估**：转换后 SNN 在 {干净, HBC 噪声, 5bit 量化} 三域下的精度 vs 原 EEGNet，报精度损失 + 理论功耗收益（脉冲稀疏性带来的 MAC→累加）。

### 2.2 备选：surrogate gradient 直接训练 SNN（精度上限更高，成本高）

- 用可导 LIF（surrogate gradient，如 `dσ/dv` 的 sigmoid/arctan 近似）直接训练，省掉转换误差，但训练不稳定、需调参、周期长。作为转换方案精度不够时的备选。

## 3. 分阶段计划（先骨架，后上板）

| 阶段 | 产出 | 验收 |
|---|---|---|
| **S1（本次）** | `lif_eegnet.py` 骨架：EEGNet 激活层抽象为 `Activation` 接口（支持 elu/relu/lif），跑通前向 + 与现有 EEGNet 对齐基线 | 前向 shape 一致、参数量一致、elu/relu 模式下与 `eegnet.py` 输出一致 |
| **S2（本次）** | 训 ReLU-EEGNet 代理，报干净/HBC/量化三域精度 | 记录 ELU→ReLU 的精度代价 |
| **S3（本次）** | `ann2snn.py` 转换脚本：BN 折叠 + LIF 替换 + 阈值标定，SNN 推理仿真（`snnTorch` 或自写 LIF 层） | 转后 SNN 精度 vs ANN，报损失 |
| **S4（后续）** | 双通道 signed 脉冲编码，缓解负半轴损失 | 精度损失收窄到 <2pp |
| **S5（后续）** | 板载 C 版 SNN 推理（复用 `lif_net.c` 思路重写为 EEGNet-SNN 版） | RV1106 上 SNN 实时性 + 功耗对比 |

## 4. 代码接口约束（P0-1b 硬要求，所有算法代码遵守）

```python
# 激活层必须走可配置接口，不写死 ELU：
class EEGNet(nn.Module):
    def __init__(self, ..., activation: str = "elu"):  # elu / relu / lif
        ...
        self.act1 = _make_activation(activation)  # 返回 nn.Module
```

- `activation="lif"` 时用可导 LIF 层（surrogate gradient），forward 返回脉冲序列（训练用发放率聚合）。
- 任何新模型（含 MyoText 手指分类网络）都要预留同样的 `activation` 接口，为后续 SNN 替换留空间。

## 5. 诚实边界

- LIF-EEGNet **不声称**「首个 EEG 脉冲网络」——SNN+EEG 已有 2023 起探索（如 Spiking-EEGNet 类工作），本项目定位是「在 HBC 闭环 + 国产 RV1106 端侧这个交叉点，把脉冲类脑解码作为约束驱动鲁棒解码的自然延伸」，复用现有「约束驱动」叙事。
- S1~S3 为主机 CPU 仿真，S4~S5 的神经形态硅（DYNAP-CNN 等）与板载 SNN 是后续工作，不在本次承诺上板精度。
