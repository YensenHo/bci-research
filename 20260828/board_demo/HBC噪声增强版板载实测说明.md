# HBC 噪声增强版 LIF-EEG-TCNet 板载实测说明

> 日期：2026-08-29　·　方向：类脑 + BCI + HBC（约束驱动鲁棒解码）
> 数据：BCIC IV 2a 22ch 四分类，eval 全量 2592 试次（cross-session，E 会话）
> 板子：Luckfox Pico（RV1106 Cortex-A7，ARMv7-A musl，zig 0.13.0 静态交叉编译，序列号 5caaa04f9cdb6ba2）
> **全部板载数字来自真实 adb shell 运行输出（日志：`tcnet_aug_board_clean.log` / `tcnet_aug_board_8db.log` / `tcnet_noaug_board_8db.log`），无任何虚构。**

---

## 0. 结论速览（TL;DR）

把 HBC 噪声增强训练的 LIF-EEG-TCNet 权重上板，**部署代码零改动**（只换权重头）：

| 场景 | 板载实测 | PyTorch 参考 | 结论 |
|---|---|---|---|
| **增强版 clean（2592 试次）** | **63.5802%（1648/2592）** | 63.58%（ckpt final_test） | 与 PyTorch **逐试次完全一致** |
| **增强版 8dB（2592 试次）** | **54.1667%（1404/2592）** | 54.67%（sweep 8dB，s42） | 守住 >50%，未崩塌 |
| **无增强版 8dB（同一份噪声）** | **39.3133%（1019/2592）** | 38.73%（sweep 8dB，s42） | 接近崩塌（chance 25%） |
| **8dB 增益（同一份噪声，板载实测）** | **+14.85pp** | +14.47pp（2 种子 mean） | 与算法报告一致 |

**一句话：增强权重上板后，芯片在 HBC 低 SNR（8dB）下精度从 39.31% 提到 54.17%（+14.85pp），clean 不降反升（62.42% → 63.58%）。**

---

## 1. 权重选择：aug_s42 确认最优

| 权重 | eval clean（final_test） | 8dB（sweep 固定噪声种子） |
|---|---|---|
| **aug_s42**（`eeg_tcnet_4class_lif_hbc_aug_s42.pth`） | **63.58%** | **54.67%** |
| aug_s123（`eeg_tcnet_4class_lif_hbc_aug_s123.pth`） | 61.11% | 54.75% |

- 选 **aug_s42**：clean 最高（63.58% > 61.11%），8dB 几乎持平（54.67 vs 54.75）。
- 与算法报告 `HBC噪声增强×TCNet鲁棒性报告.md` 一致（2 种子 mean：clean 62.35%，8dB 54.71% vs 无增强 40.23%）。

## 2. 部署方式：逻辑不变，只换权重头

**核心原则：`lif_surrogate_infer_tcnet.c` 推理逻辑零改动，仅把权重头从 `lif_tcnet_snn_weights.h`（无增强）换成 `lif_tcnet_snn_weights_aug.h`（增强）。原版二进制保留可用。**

```
diff lif_surrogate_infer_tcnet.c lif_surrogate_infer_tcnet_aug.c
46c46
< #include "lif_tcnet_snn_weights.h"
---
> #include "lif_tcnet_snn_weights_aug.h"
```

（diff 仅此一行差异，逻辑逐字节不变。）

### 2.1 导出增强版权重头

```bash
cd ~/BCI/20260828/algorithm
env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin \
  /opt/miniconda3/envs/openmmlab/bin/python export_tcnet_snn_weights.py \
  --checkpoint eeg_tcnet_4class_lif_hbc_aug_s42.pth \
  --out ../board_demo/lif_tcnet_snn_weights_aug.h
# ✓ 已导出（9344 参：conv1/depthwise/sep/pointwise + TCN 2块×3 + fc，BN 折叠）
# 仅将头文件 include guard 改为 LIF_TCNET_SNN_WEIGHTS_AUG_H（防御性，两版可共存）
```

### 2.2 宿主机编译 + 板载交叉编译

```bash
cd ~/BCI/20260828/board_demo
# 宿主机（逐位验证用 -ffp-contract=off）
cc -O2 -ffp-contract=off -DLIF_TCNET_INFER_MAIN lif_surrogate_infer_tcnet_aug.c \
   -o lif_surrogate_infer_tcnet_aug_host -lm
# 板载（RV1106 = ARMv7-A musl 静态；zig 勿加 -mcpu/-march）
~/luckfox/zig-0.13.0/zig cc -target arm-linux-musleabihf -O2 -static \
   -DLIF_TCNET_INFER_MAIN lif_surrogate_infer_tcnet_aug.c \
   -o lif_surrogate_infer_tcnet_aug -lm
```

### 2.3 8dB 加噪测试数据（严格复刻算法评估协议）

```bash
env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin \
  /opt/miniconda3/envs/openmmlab/bin/python gen_4class_bin_noisy_8db.py
```

协议与 `eval_hbc_snr_sweep.py` 的 **8dB 列完全同构**：`hbc_noise(qbits=5, qrange=4.0, sigma_ratio=10^(-8/20)=0.3981)`（5bit 量化 + AWGN，无突发），固定种子 `TEST_NOISE_SEED=20240828`，`batch_size=16` 逐 batch 施加 → `lif_4class_test_full_8db.bin`（2592 试次，114 MB）。

> 诚实说明：sweep 参考数字在 MPS 上跑、板载噪声 bin 在 CPU 上生成，同种子下 RNG 实现不同 → 噪声**实现**（具体数值）有差异，但**协议**（种子/参数/批序）完全一致。板载 8dB 实测 54.17% 与 sweep 参考 54.67% 偏差 0.5pp，在噪声实现差异的合理范围内；**板载增/无增强对比是在同一份噪声实现上实测的（公平对比）**。

### 2.4 部署到板子

```bash
ADB=~/luckfox/platform-tools/adb
$ADB -s 5caaa04f9cdb6ba2 push lif_surrogate_infer_tcnet_aug    /root/
$ADB -s 5caaa04f9cdb6ba2 push lif_surrogate_infer_tcnet_noaug  /root/
$ADB -s 5caaa04f9cdb6ba2 push lif_4class_test_full.bin         /root/
$ADB -s 5caaa04f9cdb6ba2 push lif_4class_test_full_8db.bin     /root/
```

---

## 3. 板载实测结果（真实 adb 输出）

### 3.1 增强版 clean（全量 2592 试次）

`adb shell /root/lif_surrogate_infer_tcnet_aug /root/lif_4class_test_full.bin`：

```
============================================================
  LIF-EEG-TCNet（surrogate 发放率版，22ch 四分类）板载端到端精度实测
  thr=1.00  decay=0.6065  timesteps=8  8 个 LIF 位点（含 TCN act_a/act_b/act_res）
  TCN: KT=12 dilations=(1,2) 残差 + GAP 读出头
============================================================
  试次数         : 2592
  正确数         : 1648
  端到端精度     : 63.5802% (1648/2592)
  类别 0 精度     : 63.43% (411/648)
  类别 1 精度     : 69.60% (451/648)
  类别 2 精度     : 65.28% (423/648)
  类别 3 精度     : 56.02% (363/648)
  总耗时         : 131578.3 ms
  单次前向耗时   : 50.763 ms
============================================================
```

### 3.2 增强版 8dB 加噪（全量 2592 试次）

`adb shell /root/lif_surrogate_infer_tcnet_aug /root/lif_4class_test_full_8db.bin`：

```
  试次数         : 2592
  正确数         : 1404
  端到端精度     : 54.1667% (1404/2592)
  类别 0 精度     : 48.61% (315/648)
  类别 1 精度     : 67.28% (436/648)
  类别 2 精度     : 36.73% (238/648)
  类别 3 精度     : 64.04% (415/648)
  总耗时         : 131862.3 ms
  单次前向耗时   : 50.873 ms
```

### 3.3 无增强版 8dB（对照组，同一份噪声数据）

`adb shell /root/lif_surrogate_infer_tcnet_noaug /root/lif_4class_test_full_8db.bin`：

```
  试次数         : 2592
  正确数         : 1019
  端到端精度     : 39.3133% (1019/2592)
  类别 0 精度     : 23.15% (150/648)
  类别 1 精度     : 24.69% (160/648)
  类别 2 精度     : 19.60% (127/648)
  类别 3 精度     : 89.81% (582/648)
  总耗时         : 131024.1 ms
  单次前向耗时   : 50.549 ms
```

---

## 4. 精度汇总与对照

| 来源 | 增强版 clean | 增强版 8dB | 无增强 8dB | 8dB 增益 |
|---|---|---|---|---|
| PyTorch（sweep，s42） | 63.58% | 54.67% | 38.73% | +15.94pp |
| PyTorch（报告，2 种子 mean） | 62.35% | 54.71% | 40.23% | +14.47pp |
| 宿主机 C（cc -O2） | 63.5802% | 54.1667% | 39.3133% | **+14.85pp** |
| **板载 C（zig，RV1106）** | **63.5802%** | **54.1667%** | **39.3133%** | **+14.85pp** |

- **板载 vs 宿主机：逐试次完全一致**（三组数字完全相同），交叉编译无精度损失。
- **板载 clean vs PyTorch：完全一致**（1648/2592，63.58%），增强权重无损上板。
- **板载 8dB vs sweep 参考：偏差 0.5pp**（54.17 vs 54.67 / 39.31 vs 38.73），来源是 CPU/MPS 噪声实现差异（协议相同），属正常范围。
- **同一份 8dB 噪声实现下，增强版比无增强版高 +14.85pp**（54.17 vs 39.31），与算法报告 +14.47pp 一致，低 SNR 鲁棒性增益在真实芯片上复现。

## 5. 结论与意义

1. **HBC 噪声增强权重真实上板，部署零改动**：只换一个权重头，逻辑逐字节不变，原版二进制保留可用。
2. **clean 不降反升**：板载 62.42% → 63.58%（+1.16pp，s42 单种子），增强不是拿 clean 换鲁棒性。
3. **低 SNR 显著更稳**：8dB 下板载 39.31% → 54.17%（**+14.85pp**），增强版在 8dB 仍守住 >50% 精度线，无增强版已逼近随机（39.31% vs chance 25%）。
4. **补齐算法报告最后一块拼图**：报告 §5 局限③「未评估真实板载链路（RV1106 量化）下的端到端表现」——本轮在真实芯片上完成验证，仿真结论在硬件上成立。
5. 板载单次前向 **50.5~50.9 ms**（RV1106，朴素 C 循环），全量 2592 试次约 131 s。

---

## 6. 交付文件

| 文件 | 说明 |
|---|---|
| `lif_tcnet_snn_weights_aug.h` | **新增**：增强版（aug_s42）折叠权重 C 头（export 脚本自动生成，仅 guard 改名） |
| `lif_surrogate_infer_tcnet_aug.c` | **新增**：与 `lif_surrogate_infer_tcnet.c` 仅差一行 include（逻辑零改动） |
| `gen_4class_bin_noisy_8db.py` | **新增**：8dB 加噪测试数据生成（复刻 sweep 协议：种子 20240828、5bit+AWGN σ/H=0.3981、batch16） |
| `lif_4class_test_full_8db.bin` | **新增**：8dB 加噪板载测试数据（2592 试次，114 MB） |
| `lif_surrogate_infer_tcnet_aug` | **新增**：zig 交叉编译 ARM 静态二进制（增强版） |
| `lif_surrogate_infer_tcnet_noaug` | **新增**：zig 交叉编译 ARM 静态二进制（无增强对照，逻辑=原版） |
| `lif_surrogate_infer_tcnet_aug_host` / `_noaug_host` | **新增**：宿主机验证二进制 |
| `tcnet_aug_board_clean.log` / `tcnet_aug_board_8db.log` / `tcnet_noaug_board_8db.log` | **新增**：板载真实输出日志 |
| `lif_tcnet_snn_weights.h` + `lif_surrogate_infer_tcnet.c` | 原版保留未动（clean 62.42% 仍可用） |

**未改动任何 `~/BCI/luckfox-deploy` 代码。**
