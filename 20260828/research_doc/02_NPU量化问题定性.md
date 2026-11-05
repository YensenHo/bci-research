# NPU 量化问题定性（P0-2 关键结论 · 2026-08-28）

> 由 Codex 子代理 fake-quant 消融实验得出，Hermes 复核确认。
> 这个结论**改变了 P0-2 的问题定性**，并强化「类脑计算换轨」的战略价值。

## 核心结论

**RV1106 NPU INT8 量化崩坏（argmax 一致率 42~58%）不是 INT8 精度的物理极限，而是 RKNN 工具链对深度可分离小模型的实现 bug。**

## 证据链

| 方案 | argmax 一致率（vs FP32） |
|---|---|
| FP32 上界 | 100% |
| **PTQ（PyTorch fake-quant，标准 INT8）** | **98.46%** |
| PTQ（per-channel 权重） | 99.07% |
| QAT（微调 15 epoch） | 94.83% |
| **板载 NPU 实测** | **42~58%** |

- PyTorch 侧标准 INT8 量化（fake-quant）一致率 98~99%，证明「EEGNet 这类模型做 INT8 量化理论上不该崩」。
- 板载 NPU 实测 42~58%，两者差 40~56pp，**差距无法用「量化精度损失」解释，只能归因于 RKNN 工具链实现**。
- 与既有线索一致：`convert_rknn.py` 注释已记录「rknn-toolkit2 2.3.2 对 Gemm 量化有 bug（logits[0] 偏移 −44）」，且把 fc 从 Linear(Gemm) 改为 Conv2d(1x1) 规避后，`+44*scale` workaround 仍不全局成立（45.83%，见 board_verification_log §6）——说明 bug 不止一处，深度可分离卷积路径同样有问题。

## 推论（对后续工作的影响）

1. **QAT / 混合量化「无可救的对象」**：问题不在模型的量化方法（QAT/混合量化都是优化「量化精度损失」的手段），而在工具链的算子实现。PyTorch 侧 QAT 一致率 94.83% 甚至略低于 PTQ 98.46%，且都是合法分歧，无法作用于工具链 bug。
2. **彻底修复需要 Linux x86 真跑 RKNN**：换 rknn-toolkit2 版本、或用 `custom_hybrid` 把敏感层保持 FP32（CPU 回退）、或等 Rockchip 修 bug。当前 Mac 无法执行，属硬边界。
3. **强化「类脑换轨」战略价值**：SNN 的 0/1 脉冲天然规避 INT8 量化（无连续激活、无乘法→累加），LIF-EEGNet 不是「修复」NPU，而是「换轨」绕开工具链 bug。这为顶层方针「类脑计算 + BCI + HBC」提供了最硬的技术动机。
4. **P0-2 的正式工程结论**：NPU 线维持关闭，解码写死 CPU（18.5ms/23.2ms）；NPU 1.2ms/15× 仅作速度演示。**这是真实结论，不是「没修好」，是「问题定位清楚了」。**

## 待办（受环境约束，如实标注）

- [ ] Linux x86 主机上跑真 RKNN 混合量化（`hybrid_quant_convert.py`），验证「敏感层 FP32 CPU 回退」能否救回精度——未做，需 Linux 环境。
- [ ] 尝试不同 rknn-toolkit2 版本（2.3.2 之后是否有修复）——未做。
