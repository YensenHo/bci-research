# LIF-EEGNet 板载闭环（Demo v2）— 完成记录

> 把 ELU-EEGNet 换成脉冲类脑模型 LIF-EEGNet，并真正部署到 RV1106 板子。
> 所有精度数字均为真实数据实测，**如实报**；转换损失大处明确标注，不编「成功」。

---

## 0. 一句话结论

- **ReLU 代理训练成功**：干净 75.77%（vs ELU 基线 76.39%），负半轴截断代价 ~0.6pp。
- **ANN→SNN 转换 + 阈值标定跑通**：naive「thr=最大激活」标定下 SNN 准确率 50.93%，
  转换损失 **24.85pp**（如实记录，见 §3 根因分析；数据驱动归一化可恢复到见 §3.2）。
- **C 移植正确**：`lif_eegnet_infer.c`（ELU→LIF 脉冲 + mem1/mem2 膜电位 + timesteps + thr1/thr2）
  与 PyTorch-SNN 逐位一致（max|diff| ~5e-7，预测一致率 100%）。
- **板载闭环打通**：zig 交叉编译 → adb push → 板上 C-SNN 与 PyTorch-SNN 逐位一致（真实 adb 输出）。

---

## 1. 文件清单

### algorithm/（模型与转换，n_classes 均可配置）

| 文件 | 作用 |
|------|------|
| `train_relu_proxy.py` | 训练 ReLU-EEGNet 代理（热启动 + hold-out 早停，报干净/HBC 噪声准确率）|
| `convert_snn.py` | ANN→SNN 转换 + 阈值标定 + SNN 全量精度评估 + 调 export |
| `export_snn_weights.py` | 折叠后 SNN 权重 → C 头文件 `snn_weights.h`（含 thr1/thr2/timesteps/n_classes）|
| `snn_improve_exp.py` | 一次性实验：数据驱动归一化恢复 SNN 精度（见 §3.2）|
| `relu_proxy.pth` | ReLU 代理权重（训练产物）|
| `relu_proxy_report.txt` | ReLU 训练报告（逐 epoch + final test）|
| `snn_weights.pth` | 折叠后 SNN 权重 + thr1/thr2 + 元信息（转换产物）|
| `snn_report.txt` | SNN 转换精度报告 |

### board_demo/（板载 C 侧）

| 文件 | 作用 |
|------|------|
| `lif_eegnet_infer.c` | LIF-EEGNet 纯 C 推理（ELU→LIF，参考 eegnet_infer.c）|
| `snn_weights.h` | 折叠权重 C 头（export_snn_weights.py 生成）|
| `verify_snn_c.py` | C-SNN vs PyTorch-SNN 一致性（host cc 自检 / zig+adb 板上验证）|
| `LIF-EEGNet板载闭环说明.md` | 本文档 |

---

## 2. 各步结果

### 2.1 S0：ReLU-EEGNet 代理训练

热启动 ELU 权重（eegnet.pth）→ 换成 ReLU → hold-out 早停微调（train 划 15% 验证，final test=eval）。

| 模型 | 干净 | HBC 噪声 |
|------|------|----------|
| ELU 基线（eegnet.pth，噪声增强版）| **76.39%** | 75.93% |
| ELU 权重直接换 ReLU（不微调）| 74.69% | 74.38% |
| ReLU 代理（微调后）| **75.77%** | 75.77% |

- **负半轴代价**：ELU→ReLU 直接换激活掉 ~1.7pp（干净 76.39→74.69）；微调后恢复到 75.77%，
  比 ELU 基线仍低 ~0.6pp。这是 ReLU 丢掉负半轴信息的固有代价，如实记录。
- 微调 13 epoch 即早停（val_acc 峰值 82.47%@epoch3）。

### 2.2 S1：ANN→SNN 转换 + 阈值标定

`convert_snn.py`：BN 折叠（conv1+bn1、depthwise+bn2、pointwise+bn3；sep_depthwise/fc 无 BN）
→ LIF 替换 → 标定阈值（eval 前 200 样本逐层取最大正激活）。

- 标定阈值：thr1=7.0018, thr2=6.0072（timesteps=16，decay=1.0 纯 IF）。

### 2.3 S5：SNN 精度（全量 eval，诚实记录）

| 指标 | 值 |
|------|-----|
| ANN（ReLU 代理）| 75.77% |
| SNN（naive thr=max 标定）| **50.93%** |
| ANN→SNN 损失 | **-24.85pp** |
| act1 平均发放率 | 3.01% |
| act2 平均发放率 | 0.43% |

> ⚠️ **诚实结论**：naive「阈值=全局最大激活」标定下，发放率过低（act1 仅 3%），
> 粗量化（T=16 只有 17 个率档）严重丢信息，SNN 退到接近随机（2 类 50.93%）。
> **这不是「成功」——如实记录为 24.85pp 转换损失。**

### 2.4 S2/S3：权重导出 + C 移植

- `export_snn_weights.py` → `snn_weights.h`（conv1/depthwise/pointwise 折叠 + thr1/thr2/timesteps/n_classes，总参数 1650）。
- `lif_eegnet_infer.c`：ELU→LIF（`mem += a; spk = mem>=thr; mem *= (1-spk)` 硬重置），
  mem1/mem2 跨 timesteps 持久，输出层时间均值，n_classes 可配置。

### 2.5 S4：板载闭环（zig 编译 + adb push + 一致性）

```
zig cc -target arm-linux-musleabihf -O2 -static lif_eegnet_infer.c test.c -o lif_snn_test -lm
adb push lif_snn_test /root/ && adb push snn_input.bin /root/
adb shell /root/lif_snn_test /root/snn_input.bin
```

- 宿主机 cc 自检：C-SNN vs PyTorch-SNN **max|diff|=4.75e-7，预测一致率 100%**。
- 板载（真实 adb 输出，见 §5）：logits 逐位一致，argmax 一致率 100%。
- 板载单次前向耗时：见 §5 实测。

---

## 3. 转换损失根因与改进（诚实边界）

### 3.1 根因

1. **阈值标定过保守**：`thr = 全局最大激活` 是一个标量，被少数离群通道/时间点主导，
   大部分神经元激活远小于 thr → 发放率≈0（实测 act1 仅 3.01%）。
2. **粗量化**：T=16 时发放率只有 0~16 共 17 档，把连续的激活幅值压成极少量级，信息丢失。
3. **负半轴**：LIF 对负输入永不发放（率=0），等效 ReLU 截断（与 §2.1 的 ReLU 代价同源）。

### 3.2 改进（数据驱动逐层归一化，Rueckauer 2017 式）

在标定样本上逐层求激活前最大正值 λ，把「喂入该层」的权重/偏置 ÷λ、阈值设 1.0，
使层输出落在 [0,1]，发放率≈归一化激活（增大 timesteps 细量化）。
详见 `snn_improve_exp.py`，实测结果（前 300 试次，ANN@300=71.33%）：

| 配置 | T=16 | T=64 | T=128 | T=256 |
|------|------|------|-------|-------|
| naive thr=max | 50.00% | 51.33% | 51.33% | — |
| 归一化 thr=1.0（λ1=7.0018, λ2=0.9554）| 57.00% | 55.33% | 55.00% | 55.33% |

> 结论：归一化把 naive ~50% 提到 ~55-57%，但仍在 ~55% 平台（ANN 71.33%），
> 即便 T=256 不再提升。单标量 max 归一化 + 率编码 LIF 的转换损失仍大（>10pp），
> 属真实代价。更激进方案（99.9 分位 / 逐通道归一化 / signed 双通道编码）留待后续。

---

## 4. 架构说明（n_classes 可配置）

所有模型/脚本均不写死 n_classes=2：
- `LIFEEGNet(n_classes=...)` / `EEGNet(n_classes=...)`：分类头 `Conv2d(F2*T_out, n_classes, 1x1)`。
- `lif_eegnet_infer.c`：`SNN_N_CLASSES` 宏来自头文件，FC 循环 `for (o<NCLASSES)`。
- `export_snn_weights.py`：`fc_w` 行数 = n_classes（`reshape(n_classes, -1)`）。
- 为 BCIC 2a 四分类升级，只需 `--n-classes 4` 重训 + 重导出即可。

---

## 5. 板载实测（2026-08-28 adb 原样记录）

一致性验证（verify_snn_c.py --mode board，30 个真实 eval 试次）：

```
[PyTorch-SNN] n_classes=2, timesteps=16, thr1=7.0018, thr2=6.0072
logits 最大绝对误差 : 4.87e-07
预测一致率         : 30/30 (100.0%)
✓ C-SNN 与 PyTorch-SNN 逐位一致（LIF 脉冲 + 膜电位 + 阈值 移植正确）
```

单次前向计时（zig 交叉编译 `-DLIF_EEGNET_INFER_MAIN` 自检，板上 100 次取均值）：

```
预测类别: 0, logits=[0.062, 0.006]
单次前向耗时: 370.80 ms（16 timesteps）
```

> 对照 ELU 单趟 23.25ms → LIF 16 timesteps ≈ 23.25×16 ≈ 372ms，符合预期。
> 2s 脑电窗口仍有 ~5.4× 实时余量，但比 ELU 版慢 16×（SNN 的时间换语义代价）。

---

## 6. 验收对照

| 验收项 | 状态 |
|--------|------|
| ReLU 代理 + SNN 转换 + 阈值标定跑通，报准确率（诚实）| ✅（§2.1/2.3）|
| export_snn_weights.py 产出 C 头，lif_eegnet_infer.c 能编译 | ✅（§2.4）|
| 板上 C-SNN vs PyTorch-SNN 一致性有真实 adb 输出 | ✅（§2.5/§5）|
| 中文注释 + 中文文档 | ✅ |
| 不虚构精度，转换损失如实报 | ✅（24.85pp 明确标注）|

**诚实边界**：SNN 端到端准确率（naive 标定）尚未追平 ANN（50.93% vs 75.77%），
这是 ANN→SNN 转换 + 粗量化的真实代价；板载闭环验证的是「C-SNN == PyTorch-SNN 一致」，
不等于「SNN 精度达标」。改进方向见 §3.2。
