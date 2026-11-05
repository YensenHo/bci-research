# NPU INT8 量化修复消融：PTQ / QAT / 混合量化 三方案对比

> 日期：2026-08-28　|　对象：RV1106 板载 EEGNet（8ch 二分类，1650 参数）脑电解码
> 配套脚本：`qat_train.py`（PyTorch fake-quant，本机可跑）、`hybrid_quant_convert.py`（RKNN 混合量化骨架，需 Linux x86）
> 原则：**只写实测与可推理的边界，不虚构「修复成功」。**

---

## 0. 结论先行（TL;DR）

1. **问题定性翻转**：NPU 上 INT8 的 argmax 一致率 42~58%（接近随机），但 PyTorch 侧
   对称 int8 fake-quant 仿真 **复现不出这个崩坏**——四种口径下 argmax 一致率都是 98~99%。
   这说明崩坏根因 **不是 INT8 精度极限**，而更可能是 **RKNN 工具链 bug**
   （已有已知线索：`rknn-toolkit2 2.3.2` 对 Gemm 量化产生 logits[0] 恒定偏移 −44）。

2. **QAT 在这个问题上的作用被证伪**：既然模型在「伪量化」层面本就鲁棒（≈98% 一致），
   QAT 就 **没有可救的东西**。实测 QAT 微调后一致率 94.8%（≈ FP32 上界，且与 PTQ 同量级），
   既不崩坏也无增益——因为崩坏不在模型侧，而在工具链侧。QAT 修不了工具链 bug。

3. **混合量化的价值重定位**：不再指望「混合精度救精度」，而是当作 **诊断工具**——
   把疑似 bug 的层（fc / depthwise / 含 ELU 负半轴的层）单独保持 FP32，看崩坏是否消失，
   从而 **定位** 具体问题算子。可行性受 rv1106「NPU 仅 INT8、FP32 层只能 CPU 回退」硬约束，
   需 Linux x86 + 真板实测。

---

## 1. 问题背景

| 事实 | 说明 |
|---|---|
| 板载模型 | EEGNet，8ch 二分类（左/右手），FP32 准确率 76.39%（eval 1296 试次） |
| 权重/ONNX | `eegnet.pth`、`eegnet.onnx` 已存在 |
| NPU 现状 | RKNN INT8 转换后 argmax 一致率仅 42~58%，接近随机 |
| 已证伪 | PTQ + `44*scale` workaround 无效 |
| 硬件硬约束 | rv1106 NPU 仅 INT8；w8a16 / int16 / FP16 均报 `not support` |
| 工具链硬约束 | rknn-toolkit2 仅 Linux x86_64，Mac 无法跑 RKNN 转换 |

---

## 2. 实验设定（qat_train.py 的仿真口径）

fake-quant 用「对称 int8 量化 + 直通估计（STE）」模拟 NPU INT8：
- **权重量化**：每个 Conv2d 的 weight 做对称 int8 量化（`torch.nn.utils.parametrize` 挂伪量化，
  forward 量化、backward STE 直通）。
- **激活量化**：两种口径——①「算子边界」：量化每个 Conv 的输入（等价于上一层 BN/ELU/pool 输出）；
  ②「逐算子输出」：每个 Conv/BN/ELU/Pool 的输出都量化一次（最贴近 RKNN 逐算子量化）。
- 三种模式：`baseline`（FP32 上界）/ `ptq`（插 fake-quant 不重训）/ `qat`（fake-quant 下微调 15 epoch）。

### 实测结果（真实数据，eval 1296 试次）

| 方案 | 权重口径 | 激活口径 | 准确率 | vs FP32 argmax 一致率 |
|---|---|---|---|---|
| **FP32 上界** | — | — | **76.39%** | 100.0%（基准） |
| PTQ（fake-quant 不重训） | per-tensor | 算子边界 | 76.70% | **98.46%** |
| PTQ | per-channel | 算子边界 | 76.85% | 99.07% |
| PTQ | per-tensor | 逐算子输出 | 76.39% | 98.61% |
| PTQ | per-channel | 逐算子输出 | 77.01% | 98.61% |
| QAT（微调 15 epoch, freeze-bn） | per-tensor | 算子边界 | 76.93% | 94.83% |

> 解读：**无论权重 per-tensor/per-channel、激活边界/逐算子，PTQ 一致率都 ≈98~99%**，
> 无一接近 NPU 的 42~58%。QAT 后 94.8% 与 PTQ 同量级（微调使模型在边界样本上与 FP32 参考
> 产生少量「合法分歧」，准确率本身 76.39%→76.93% 甚至略升），**无增益也无崩坏**。

---

## 3. 三方案对比总表

| 维度 | PTQ | QAT | 混合量化 |
|---|---|---|---|
| 原理 | 训练后直接对权重/激活统计 min/max 量化 | 训练时注入 fake-quant，让权重在量化噪声下收敛 | 选定的层保持 FP32，其余层 INT8 |
| 改模型？ | 否 | 否（权重微调） | 否（只改转换配置） |
| 是否需重训 | 否 | **是**（微调若干 epoch） | 否 |
| 执行环境 | Linux x86（RKNN） | **本机 PyTorch 可跑** | Linux x86（RKNN） |
| 针对的问题 | INT8 精度极限 | INT8 精度极限（模型侧） | 特定层/算子的量化 bug（工具链侧） |
| 本问题实测 | 复现不出崩坏（98%） | 无可救（94.8% ≈ 上界） | 未实测（Mac 跑不了，骨架已备） |
| 结论 | ❌ 无法解释 NPU 崩坏 | ❌ 修不了工具链 bug | ⚠ 定位工具链 bug 的诊断手段 |

---

## 4. 各方案详述

### 4.1 PTQ（Post-Training Quantization，后训练量化）

- **设计**：加载 FP32 权重 → 用校准集统计每层激活 min/max → 权重/激活量化到 int8 → 推理。
  现有 `convert_rknn.py` 即此方案（`build(do_quantization=True, dataset=calib)`）。
- **预期**：若崩坏是 INT8 精度极限，PTQ 仿真应复现 42~58%。
- **实测边界**：**没有复现**。对称 int8 下一致率 98~99%。因此「校准集大小、per-tensor/per-channel」
  这些常规 PTQ 旋钮 **不是** 崩坏主因（四种口径都试了，都不崩）。
- **可落地结论**：PTQ 在 PyTorch 口径下是可用的（≈98% 一致），NPU 崩坏是 RKNN 转换侧的问题。

### 4.2 QAT（Quantization-Aware Training，量化感知训练）

- **设计**：`qat_train.py`，STE fake-quant 挂在权重 + 激活上，从 FP32 权重出发微调，让模型
  「适应」量化噪声。
- **预期**：若崩坏是模型对 int8 敏感，QAT 应能把一致率从 ~50% 拉回 ≥95%。
- **实测边界**：因为 PTQ 仿真本就 98%（未崩坏），QAT **没有可救的精度**；微调后 94.8% 与上界
  同量级。**QAT 无法解决工具链 bug**——这是本次消融的核心结论。
- **可落地结论**：QAT 在本问题上**没有作用**；若仍想用，只有在「PTQ 仿真确实复现了崩坏」的前提
  下才有意义（结论模板见 §6）。

### 4.3 混合量化（Hybrid Quantization）

- **设计**：`hybrid_quant_convert.py`，`hybrid_quantization_step1` 生成每层量化误差表 → 人工挑出
  敏感/疑似 bug 的层 → `step2` 用 `custom_hybrid={层名: np.float32}` 保持这些层 FP32 构建。
- **预期**：把崩坏层定位到具体算子后，单独 FP32 可让崩坏消失。
- **边界（诚实声明）**：
  1. rv1106 NPU **仅 INT8**；被标 FP32 的层只能 **CPU 回退**，板载 rknn-lite2 是否支持、
     回退后延迟是否可接受，**均未实测**。
  2. 混合 dtype（w8a16/int16/FP16）已确认 `not support`，与本方案不同但同受 rv1106 限制。
  3. 层名与 API 签名随 rknn-toolkit2 版本变化，需以本机 docstring 为准。
  4. 需在 **Linux x86** 跑，本机（Mac）只能备骨架。
- **可落地结论**：作为**诊断工具**价值明确；作为**部署方案**可行性存疑（CPU 回退可能抵消 NPU 收益）。

---

## 5. 关键结论与建议

1. **NPU 崩坏的根因应从「INT8 精度」改判为「RKNN 工具链 bug」**。首要动作不是 QAT，
   而是在 Linux x86 上用 `convert_rknn.py --verify`（其 `_verify_accuracy` 已对比模拟器 vs ONNX）
   确认崩坏发生在「转换阶段」还是「板子 runtime」。
2. **头号嫌疑**：Gemm/输出层量化（已知 logits[0] 偏移 −44）。模型已把 `Linear` 换成 `Conv2d(1x1)`
   规避，但需在真 RKNN 上验证该规避是否生效。
3. **诊断路径**：`hybrid_quant_convert.py` 依次把 fc → depthwise → 含 ELU 的层 保持 FP32，
   看崩坏在哪一层消失，即锁定 bug 算子。
4. **若工具链 bug 无法规避**：如实写「工具链固有限制，NPU 线维持关闭，解码走 CPU FP32」——
   这与 `weight_precision_scan.py` / `activation_precision_scan.py` 的既有结论一致。

---

## 6. 可落地结论模板（供最终报告填空）

```text
【方案结论 · PTQ】
在 PyTorch 对称 int8 fake-quant 口径下，EEGNet 的 argmax 一致率 = ____%（实测 98~99%），
[复现 / 未复现] NPU 上 42~58% 的崩坏。
→ 结论：PTQ 常规旋钮（校准集/scale/通道粒度）[是 / 不是] 崩坏主因。

【方案结论 · QAT】
在 fake-quant 下微调 ____ epoch 后，一致率 = ____%（实测 94.8%），与 FP32 上界（76.39% 准确率）
相比 [有 / 无] 显著提升。
→ 结论：QAT [能 / 不能] 救回 INT8 精度。若 PTQ 仿真本就未崩坏，则 QAT 无对象可救，
   问题定位到工具链侧。

【方案结论 · 混合量化】
将 [层名1 / 层名2 / ...] 保持 FP32 后，板载 argmax 一致率 = ____%（需 Linux x86 + 真板实测）。
→ 结论：崩坏 [是否] 随某层 FP32 而消失。若消失，则锁定 bug 算子 = ____；
   若 rv1106 无法 CPU 回退 FP32 层，则混合量化不可落地，NPU 线维持关闭。
```

---

## 7. 交付物清单

| 文件 | 说明 | 可运行环境 |
|---|---|---|
| `qat_train.py` | fake-quant QAT 消融（baseline/ptq/qat 三模式，含 per-channel、逐算子口径） | 本机 PyTorch ✅ 已跑通 |
| `qat_results.json` | 本次消融的量化结果落盘 | 本机 ✅ |
| `hybrid_quant_convert.py` | RKNN 混合量化骨架（step1 误差表 + step2 custom_hybrid），含 rv1106 边界与 Linux x86 执行标注 | 仅 Linux x86（Mac 只读骨架） |
| 本文档 | 三方案消融对比 + 结论模板 | — |
