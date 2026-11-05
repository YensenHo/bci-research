# Demo 替换 LIF-EEGNet 说明 — 把 EEGNet 换成脉冲类脑模型

> 本文档说明：当前 Demo（板载 EEGNet 解码）后续如何替换为 **LIF-EEGNet 脉冲类脑模型**。
> 引用 `../algorithm/lif_eegnet.py` 与 `../algorithm/ann2snn.py` 的接口。
> 诚实边界：SNN 替换的 Python 侧代码已写好并通过骨架自检，**板载 C 移植尚未完成**，本文给出可执行的替换路线。

---

## 1. 现状：当前 Demo 用的是 ELU-EEGNet

板载 `eegnet_infer.c`（纯 C）对应 `~/BCI/luckfox-deploy/model/eegnet.py`：

```
conv1(1→8,1×63) → depthwise(8→16,8×1) → ELU → AvgPool(1,4)
→ sep_depthwise(16→16,1×15) → pointwise(16→16) → ELU → AvgPool(1,8) → FC(240→2)
```

- 激活是 **ELU**（`eegnet_infer.c` 里内联 `elu(x) = x>0 ? x : expf(x)-1`，第 72/106 行两处）。
- BN 已在 `export_c_weights.py` 折叠进卷积，权重存 `eegnet_weights.h`。
- 当前权重 76.16%（噪声增强版），单次前向 23.2ms，与 PyTorch 误差 ~1e-6。

**替换目标**：把两个 ELU 激活位点换成 **LIF 脉冲神经元**，前向变成「恒定电流输入 + timesteps 个时间步积分发放 + 输出时间均值」，得到脉冲类脑解码器 LIF-EEGNet。

---

## 2. 接口一：`algorithm/lif_eegnet.py`（可配置激活的 EEGNet）

文件路径：`/Users/heyixuan/BCI/20260828/algorithm/lif_eegnet.py`

关键类/函数：

| 符号 | 说明 |
|------|------|
| `LIFEEGNet(activation='elu'/'relu'/'lif', **lif_kwargs)` | 主模型，结构 100% 对齐 `eegnet.py`，仅激活层走工厂 |
| `LIFActivation(threshold=1.0, tau=2.0, timesteps=8, alpha=4.0)` | 可导 LIF 发放率层（surrogate gradient，训练友好）|
| `_make_activation(activation, **lif_kwargs)` | 激活工厂：elu/relu 无参；lif 返回 LIFActivation |
| `LIFEEGNet.forward_features(x)` | 输出 (B, F2*T//32) 特征（flatten 后、fc 前）|
| `LIFEEGNet.num_params()` | 参数量 |
| `self.act1` / `self.act2` | 两个激活位点（Block1 后 / Block2 后），可被 ann2snn 改写阈值 |

**设计意图**：
- `activation='elu'/'relu'`：与 `eegnet.py` 零回归（state_dict 键、参数量、输出完全一致），可直接 load 现有权重。
- `activation='lif'`：把连续激活值当「恒定输入电流」，`timesteps` 步泄漏积分发放，输出发放率 ∈ [0,1]。
  这是「surrogate gradient 直接训练 SNN」的落点（可导，能反向传播）。
- LIF 层语义（发放率版）：`mem = decay*mem + x`（decay=exp(-1/tau)）→ `spk = H(mem-threshold)` → 硬重置 `mem *= (1-spk)` → 输出 `mean(spk)`。
- ⚠️ 已知代价：负半轴输入永不发放（发放率=0），等效 ReLU 截断，`lif` 训练路径的「负半轴损失」尚未缓解（文档标注 S4 后续）。

---

## 3. 接口二：`algorithm/ann2snn.py`（ANN→SNN 转换 + 阈值标定）

文件路径：`/Users/heyixuan/BCI/20260828/algorithm/ann2snn.py`

关键函数/类：

| 符号 | 说明 |
|------|------|
| `fold_bn_layer(conv, bn, eps=1e-5)` | BN 折叠进前一层卷积（`scale=γ/√(σ²+ε)`，`W'=W·scale`，`b'=b·scale+(β−μ·scale)`）|
| `clone_conv(conv)` | 复制无 BN 的卷积（sep_depthwise、fc）|
| `lif_step(current, mem, threshold, decay)` | 单时间步 LIF：`mem=decay·mem+current` → `spk=H(mem-threshold)` → 硬重置；返回 `(spk, mem')` |
| `SNNEEGNet(ann)` | 折叠 + LIF 替换后的脉冲 EEGNet；`self.thr1`/`self.thr2` 待标定 |
| `build_snn(ann)` | 从 ReLU/ELU-EEGNet 构建 SNN（断言 `ann.activation in ('relu','elu')`）|
| `calibrate_thresholds(ann, samples, eps=1e-3)` | 逐层阈值标定：记录 act1/act2 激活前最大正值作为发放阈值，返回 `(thr1, thr2)` |
| `convert_and_calibrate(ann, samples)` | 一步：build_snn + 标定，返回已设阈值的 SNN |
| `snn_infer(snn, x, timesteps=16)` | SNN 推理：返回 (B, n_classes) 时间均值分类得分 |

**转换三步（已实现，各自可独立运行）**：
1. **BN 折叠**：SNN 里不能留 BN，折叠进卷积（复用 `export_c_weights.py` 的公式）。
2. **LIF 替换**：ReLU 位点 → LIF 脉冲神经元（0/1 二值发放）。
3. **阈值标定**：标定样本上记录每层激活前最大正值作为发放阈值。

**SNN 前向语义**（rate-based，`SNNEEGNet.forward` 第 153-187 行）：
- 输入 x 作恒定电流，在 `timesteps` 步内反复馈入首层；
- Block1/Block2 各跑一次 LIF（`decay=1.0` 纯 IF 为默认，rate-exact）；
- 输出层线性读出（不脉冲化），`out_sum/timesteps` 作分类得分。

**诚实边界**（ann2snn.py docstring 原文）：这是「可运行骨架」，只验证 dummy 输入上跑通、折叠数学正确、形状正确；
**不报告任何 SNN 精度**（无训练权重 + 无真实数据），精度评估属 S3 后续工作，标注「未验证」。

---

## 4. 替换路线（板载 C 侧，尚未实施）

### 4.1 现有部署链（复用）

```
model/export_c_weights.py   →  board/eegnet_weights.h   →  zig 交叉编译  →  adb push
model/verify_c_infer.py     →  C vs PyTorch 一致性（max|diff|<1e-3 + 100% 预测一致）
```

### 4.2 替换步骤（分阶段，标注状态）

| 步骤 | 内容 | 状态 |
|------|------|------|
| S0 | 训练/加载 ReLU-EEGNet 代理权重（`LIFEEGNet(activation='relu')` load 现有 eegnet.pth）| 可做（relu 与 eegnet.py 零回归）|
| S1 | `ann2snn.convert_and_calibrate(ann, calib_samples)` 得折叠权重 + thr1/thr2 | Python 已实现，需接真实权重/标定样本 |
| S2 | 导出折叠后 SNN 权重到新 C 头（复用 export_c_weights.py 的折叠公式，注意 sep_depthwise 无 BN）| 需写 `export_snn_weights.py` |
| S3 | 改 `eegnet_infer.c`：ELU → LIF 神经元 + timesteps 循环 + 膜电位 mem1/mem2 + thr1/thr2 | 需写（核心改动）|
| S4 | zig 交叉编译 → 板上验证 C-SNN vs PyTorch-SNN 一致 | 需做 |
| S5 | 端到端精度评估（真实脑电全量）| 需做，当前「未验证」|

### 4.3 核心改动点（`eegnet_infer.c` → `lif_eegnet_infer.c`）

当前 `eegnet_infer.c` 的结构（第 72/106 行 ELU）：

```c
/* ELU + AvgPool(1,4) */
for (int o=0; o<F2; o++) for (int t=0; t<T4; t++) {
    float acc=0; for(int k=0;k<4;k++) acc += elu(dw[o][t*4+k]);
    p1[o][t]=acc/4.0f;
}
```

替换为 LIF（rate-based，需在整段 forward 外套 timesteps 循环）：

```c
/* LIF 神经元：mem = decay*mem + x; spk = (mem>=thr); mem*=(1-spk) */
static float mem1[F2][T4], mem2[F2][T32];   /* 跨 timesteps 持久膜电位 */
/* 每个 timestep：
   a1 = depthwise(conv1(x));  spk1, mem1 = lif(a1, mem1, thr1);
   p1 = avgpool(spk1);
   a2 = pointwise(sep_depthwise(p1));  spk2, mem2 = lif(a2, mem2, thr2);
   p2 = avgpool(spk2);  out += fc(flatten(p2));
   最终 logits = out / timesteps; */
```

关键差异对照：

| 项 | 当前 EEGNet (ELU) | 替换后 LIF-EEGNet (SNN) |
|----|-------------------|--------------------------|
| 激活 | `elu(x)` 逐元素 | `lif_step(a, mem, thr, decay)` 带膜电位记忆 |
| 时间维 | 单次前向 | `timesteps` 步积分发放，输出时间均值 |
| 权重 | ELU 权重（BN 已折叠）| SNN 折叠权重（conv1+bn1、depthwise+bn2、pointwise+bn3；sep_depthwise/fc 无 BN）|
| 额外参数 | 无 | `thr1`/`thr2`（标定）、`decay`、`timesteps` |

### 4.4 为什么值得换（类脑换轨论证，诚实表述）

- **0/1 二值发放**天然规避 RV1106 NPU 的 INT8 量化噪声（ann2snn.py 的「类脑换轨解法」；
  但 NPU 线已关闭，实际仍是 CPU 跑，此点只作论证不作卖点）。
- 发放率编码天然稀疏、事件驱动，是「端侧神经形态」的叙事主线。
- ⚠️ 不主张「全球首创/唯一」——SNN+EEG（如 arXiv 2304.11106、2307.08501）与 ANN-to-SNN（Diehl 2015 / Rueckauer 2017）均已有公开工作，本替换是「公开可检索范围内的差异化探索」。

---

## 5. 与 Demo 可视化链的衔接

替换完成后，板载二进制从 `eegnet_infer` / `hbc_realtime` 换成 LIF-EEGNet 版，但**上位机可视化链路不变**：

```
./lif_eegnet_realtime <bin>          demo_visualizer.py（无需改动）
  ① LIF-EEGNet 直连解码（SNN）        读取 adb 流式输出
  ② 5bit 量化 → OOK → HBC 信道 → 解调   逐窗口解析
  ③ 收端二次解码（经HBC）              终端 + 暗色图
```

- 只要板载流式输出保持 `[n] 真值=... 直连=... 经HBC=... ✓ | Xms (实时余量 Y%)` 格式，
  `demo_visualizer.py` 无需任何改动即可可视化 LIF-EEGNet 结果。
- 建议：新增的 `lif_eegnet_realtime` 复刻 `hbc_bci_realtime.c` 的主循环，仅把 `eegnet_infer()` 换成
  `lif_eegnet_infer()`（含 thr1/thr2/timesteps 参数），HBC 信道仿真代码原样复用。

---

## 6. 验收建议（替换完成后）

1. **折叠正确性**：`ann2snn._verify_fold` 已验 BN 折叠数学；导出到 C 后再用 `verify_c_infer.py` 的思路做 C-SNN vs PyTorch-SNN 逐位一致。
2. **精度诚实**：先报告「C-SNN == PyTorch-SNN」的一致率，再报告端到端准确率；未测就写「未验证」，不编数字。
3. **实时性**：LIF 多出 timesteps 循环，需重测单窗口耗时，确认仍满足 2s 窗口实时余量（当前 ELU 版 59ms，34× 余量，有充分预算）。
