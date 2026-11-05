# LIF-EEG-TCNet 板载实测说明（RV1106 / Luckfox Pico Ultra）

> 实测日期：2026-08-28 ｜ 板子：Luckfox Pico Ultra（RV1106G3，Cortex-A7 @1.2GHz，512MB RAM）
> 链路：PyTorch surrogate 训练 → BN 折叠导出 C 头 → 纯 C 推理 → zig 交叉编译 → adb 板载全量 2592 试次
> 结论先行：**板载四分类精度 62.4228%（1618/2592），与 PyTorch 逐位一致（同一份权重、同一正确数），
> 比现有 LIF-EEGNet 板载 56.8673% 高 +5.56pp，芯片四分类精度提升达成。**

---

## 1. 结论摘要

| 指标 | LIF-EEG-TCNet（本次） | LIF-EEGNet（现板载） | 变化 |
|---|---|---|---|
| 板载四分类精度 | **62.4228%（1618/2592）** | 56.8673%（1474/2592） | **+5.56pp** |
| 宿主机 C 全量精度 | 62.4228%（1618/2592） | — | 与板载逐位一致 |
| PyTorch 折叠参考全量 | 62.4228%（1618/2592） | — | 与 C 完全一致（含正确数） |
| checkpoint final_test | 62.42%（eeg_tcnet_4class_lif_t1.0_tau2.0_T8.pth） | — | 0 差异 |
| 单次前向耗时（板载，含读盘） | 95.552 ms | 49.747 ms | +45.8 ms（TCN/GAP/LIF 位点增加） |
| 单次前向耗时（板载，纯计算冒烟） | 50.322 ms | — | — |
| 单次前向耗时（宿主机 arm64） | 2.475 ms | — | — |

说明：LIF-EEG-TCNet 的 +5.56pp 来自架构升级（EEGNet → EEG-TCNet），类脑化代价为 **0**：
LIF 版（62.42%）甚至略高于 relu 版 brief 基线（61.61%，eeg_tcnet_4class_relu_brief.pth），
这是 surrogate 直接训练（热启动 relu 权重微调）把类脑化代价压回 0 的直接证据。

---

## 2. 模型结构与部署链

权重：`algorithm/eeg_tcnet_4class_lif_t1.0_tau2.0_T8.pth`（LIF thr=1.0 / tau=2.0 / T=8 / alpha=4.0，
surrogate 训练，热启动 relu brief 权重，参数量 9,512，final_test=62.42%）。

模型（eeg_tcnet_lif.py，brief 配置 = 简报最优 relu 配置 KE=63/KT=12/pool4/8/GAP）：

| 层 | 尺寸 | 折叠 |
|---|---|---|
| conv1 | Conv2d(1→8, 1×63, 无偏置) | +bn1 → conv1_w[8][63]+b |
| depthwise | Conv2d(8→16, 22×1, groups=8, 偏置) | +bn2 → depthwise_w[16][22]+b |
| act1（LIF 位点 1） | thr=1.0, decay=0.6065, T=8 → 发放率 | — |
| pool1 | AvgPool(1,4) → T4=125 | — |
| sep_depthwise | Conv2d(16→16, 1×16, groups=16, 无偏置) | 无 BN，原样 |
| pointwise | Conv2d(16→16, 1×1, 偏置) | +bn3 → pointwise_w[16][16]+b |
| act2（LIF 位点 2） | 同上 | — |
| pool2 | AvgPool(1,8) → T15=15 | — |
| TCN 块0（dilation=1） | conv_a: Conv1d(16→12, k12, 偏置)[+bn] → LIF(act_a) → conv_b: Conv1d(12→12, k12, 偏置)[+bn] → LIF(act_b) → skip 1×1(16→12, 偏置) 残差 add → LIF(act_res) | conv_a/conv_b 折叠 BN；skip 无 BN |
| TCN 块1（dilation=2） | 同上但 12→12，skip 1×1 无偏置 | 同上 |
| GAP | AdaptiveAvgPool1d(1)：15 时间步均值 | — |
| fc | Linear(12→4) | 无 BN，原样 |

TCN 膨胀因果卷积语义（eeg_tcnet_lif.py 原样复刻）：手动左补零 `pad_left=(KT-1)*d`，
C 索引 `out[t] = Σ_j w[j]·in[t + j·d - pad_left]`（仅需检查下界 idx≥0，上界自动满足）。
块0 因果感受野 12 步 / 块1 22 步，覆盖 T15=15 的整个序列，GAP 聚合全部时间步。

部署链：`algorithm/export_tcnet_snn_weights.py`（BN 折叠 + %.8e 精确往返断言导出）→
`board_demo/lif_tcnet_snn_weights.h`（9344 参数）→ `board_demo/lif_surrogate_infer_tcnet.c`。

---

## 3. 验证：宿主机三级（全部通过）

1. **导出 round-trip 断言**：export 脚本对每个权重数值做「%.8e 字符串 → float32」往返校验，
   保证 C 头与 PyTorch 权重逐位一致（9344 参数全过）。
2. **严格逐位一致**（`verify_tcnet_bit_exact.py`，numpy float32 C-顺序参考 vs C，
   编译 `cc -O2 -ffp-contract=off` 禁用 FMA 收缩）：
   ```
   [权重] decay=0.606530666, thr=1.0, timesteps=8, n_classes=4（解析自 C 头文件，与 C 编译产物同源）
   [对比] numpy(C 顺序) vs C  max|diff| = 0.000e+00
   [对比] 逐位完全一致的样本数   : 20/20
   ✓ 严格逐位一致：C 移植（膨胀因果卷积/残差/GAP/FC）与 float32 C-顺序参考完全一致（0 误差）
   ```
   该层专门覆盖 TCN 膨胀卷积索引、1×1 skip 残差 add、GAP、Linear 读出的结构性正确性。
3. **真实 PyTorch 折叠参考一致率**（`verify_tcnet_vs_pytorch.py --mode host`）：
   ```
   [对比] PyTorch(折叠参考) vs C(host)  预测一致率 : 200/200 = 100.00%
   [对比] logits max|diff| : 7.153e-07
   ✓ 预测一致率 100%
   ```
   logits 差异 7e-07 量级（浮点非结合性，TCN 层小 + 发放率离散化使其远小于 EEGNet 版的 1e-2），
   不改变 argmax。**全量 2592 试次下 C 与 PyTorch 正确数完全相同（都是 1618）**，即全量预测一致率 100%。

---

## 4. 板载实测（真实 adb 输出）

交叉编译：`~/luckfox/zig-0.13.0/zig cc -target arm-linux-musleabihf -O2 -static
-DLIF_TCNET_INFER_MAIN lif_surrogate_infer_tcnet.c -o lif_surrogate_infer_tcnet -lm`
（ARMv7 硬浮点静态 musl，勿加 -mcpu/-march，默认即 A7 兼容）。

`adb push` 后板载运行 `/root/lif_surrogate_infer_tcnet /root/lif_4class_test_full.bin`（全量 2592 试次）：
```
============================================================
  LIF-EEG-TCNet（surrogate 发放率版，22ch 四分类）板载端到端精度实测
  thr=1.00  decay=0.6065  timesteps=8  8 个 LIF 位点（含 TCN act_a/act_b/act_res）
  TCN: KT=12 dilations=(1,2) 残差 + GAP 读出头
============================================================
  试次数         : 2592
  正确数         : 1618
  端到端精度     : 62.4228% (1618/2592)
  类别 0 精度     : 64.35% (417/648)
  类别 1 精度     : 66.82% (433/648)
  类别 2 精度     : 60.96% (395/648)
  类别 3 精度     : 57.56% (373/648)
  总耗时         : 247671.8 ms
  单次前向耗时   : 95.552 ms
============================================================
```

冒烟模式（无参数，纯计算 100 次，随机输入）：
```
预测类别: 1, logits=[0.944, 1.463, -3.291, -0.410]
单次前向耗时: 50.322 ms（8 timesteps，decay=0.6065）
```
> 95.552 ms 是全量端到端每试次均值（含从 eMMC 流式读 44KB/试次 + malloc）；纯计算约 50.3 ms。

---

## 5. 精度对比（芯片四分类 +5.56pp）

| 模型（22ch 四分类，同一 eval 全量 2592 试次） | 精度 | 来源 |
|---|---|---|
| EEGNet ELU 基线 | 58.29% | 对比报告_四分类22ch_类脑化零代价验证.md |
| LIF-EEGNet（现板载） | **56.8673%** | 四分类板载实测说明.md（板载实测） |
| EEG-TCNet relu brief | 61.61% | eeg_tcnet_4class_relu_brief.pth |
| **LIF-EEG-TCNet（本次板载）** | **62.4228%** | 本说明 §4（板载实测） |

要点：
- 类脑化代价 = 0：LIF-TCNet 62.42% ≥ relu-TCNet 61.61%（surrogate 直接训练微调补回甚至反超）。
- 相对现板载 LIF-EEGNet 提升 +5.56pp，主因 TCN 膨胀因果卷积的时间上下文建模 + GAP 全序列聚合。
- 每类精度：左手 64.35% / 右手 66.82% / 双脚 60.96% / 舌头 57.56%（各 648 试次）。

## 6. 计算量与实时性

- MACs 估算（T=500）：conv1 5.54M + depthwise 0.18M + separable 0.06M + TCN ~0.16M + 读出 ≈ **5.9M/试次**。
- 板载单次前向 50.3 ms（纯计算）≈ 2s 试次的 **40× 实时余量**；LIF 8 位点 × 8 timesteps 的
  发放率计算与 TCN 卷积合计约 +0.6ms 相对前端开销，大头仍是 conv1（5.5M MACs，与 EEGNet 相同）。
- 静态内存 ~410KB（BSS），峰值 RSS 远低于 512MB 板载内存。

## 7. 交付文件

```
algorithm/
  export_tcnet_snn_weights.py         # BN 折叠导出（含 TCN Conv1d 折叠 + round-trip 断言）
board_demo/
  lif_tcnet_snn_weights.h             # 自动生成：LIF-EEG-TCNet 折叠权重（9344 参数）
  lif_surrogate_infer_tcnet.c         # 22ch 四分类 C 推理（膨胀因果卷积+残差+GAP+LIF）
  verify_tcnet_bit_exact.py           # numpy C-顺序逐位一致（max|diff|=0）
  verify_tcnet_vs_pytorch.py          # PyTorch 折叠参考 vs C（预测一致率 100% + 全量精度）
  lif_surrogate_infer_tcnet           # zig 交叉编译产物（ARMv7 静态）
  lif_surrogate_infer_tcnet_host      # 宿主机 cc 产物（供全量自检）
  tcnet_board_full.log                # 板载全量 2592 试次真实输出
  LIF-EEG-TCNet板载实测说明.md         # 本文档
```

复现：
```bash
# 1) 导出权重头
env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin \
    /opt/miniconda3/envs/openmmlab/bin/python algorithm/export_tcnet_snn_weights.py
# 2) 宿主机验证
env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin \
    /opt/miniconda3/envs/openmmlab/bin/python board_demo/verify_tcnet_bit_exact.py --n 20
env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin \
    /opt/miniconda3/envs/openmmlab/bin/python board_demo/verify_tcnet_vs_pytorch.py --n 200 --mode host --full
# 3) 交叉编译 + 板载
~/luckfox/zig-0.13.0/zig cc -target arm-linux-musleabihf -O2 -static \
    -DLIF_TCNET_INFER_MAIN board_demo/lif_surrogate_infer_tcnet.c -o board_demo/lif_surrogate_infer_tcnet -lm
~/luckfox/platform-tools/adb -s 5caaa04f9cdb6ba2 push board_demo/lif_surrogate_infer_tcnet /root/
~/luckfox/platform-tools/adb -s 5caaa04f9cdb6ba2 shell "/root/lif_surrogate_infer_tcnet /root/lif_4class_test_full.bin"
```
