# HBC 噪声增强 × LIF-EEG-TCNet 鲁棒性报告

> 日期：2026-08-29　·　方向：约束驱动鲁棒解码（世界主流模型 + HBC 信道噪声增强）
> 数据：BCIC IV 2a 22ch 四分类，train/eval 各 2592 试次（cross-session，E 会话）
> 模型：LIF-EEG-TCNet（brief 配置 KE=63/KT=12/pool4/8/GAP，9,512 参数，
>        thr=1.0/tau=2.0/T=8，surrogate 训练，热启动 relu 权重）
> 全部数字为真实训练/测试跑出（逐 epoch 记录在 report txt，未虚构）。

---

## 0. 结论速览

1. **clean 持平（略升）**：HBC 噪声增强训练后 clean 四分类精度 **61.42% → 62.35%**
   （2 种子 mean，+0.93pp），没有掉 clean。逐种子：seed42 62.42→63.58（+1.16pp）、
   seed123 60.42→61.11（+0.69pp），两个种子均不降。
2. **低 SNR 显著更稳（核心发现，比 EEGNet 更强）**：8 dB 处无增强崩塌到 **40.23%**（较 clean
   掉 21.2pp，接近随机），增强模型守住 **54.71%**（只掉 7.6pp），**+14.47pp**；
   4 dB 处 **+14.22pp**（27.82% vs 42.03%）。增强让退化斜率减为基线的 1/2.8。
3. **SNR 崩塌拐点后移 ≥ 4.5 dB**：以「精度 ≥50%」为判据，无增强拐点 ≈ **11.0 dB**，
   增强模型 **< 6.5 dB**——恰好覆盖并下推板载 SNR 瀑布拐点区间（8~14 dB）。
4. **最恶劣 SNR 下更稳**：8 dB 处种子 std 从 ±1.51pp 压到 ±0.04pp；训练带突发分量后，
   8dB+突发测试域同样守住（53.39% vs 41.57%，+11.82pp）。
5. 与 EEGNet（8ch 二分类，8 dB 处 +3.94pp）相比，TCNet 上收益更大（+14.47pp）：
   LIF 率编码前端在强噪声下更易发放饱和/塌缩，训练期噪声约束的收益因此被放大
   （详见 §5 讨论）。

---

## 1. 交付物与复现

| 文件 | 说明 |
|---|---|
| `train_eeg_tcnet_hbc_aug.py` | LIF-EEG-TCNet 训练 + HBC 噪声增强（训练时对输入施加 5bit 量化 + AWGN + 突发，每 batch SNR∈[10,20]dB 随机，复用 `hbc_noise_advanced.py`） |
| `eval_hbc_snr_sweep.py` | SNR 扫描评估（clean/20/16/12/8/4 dB + 8dB+突发），测试噪声固定种子 20240828，公平对比 |
| `plot_hbc_tcnet_snr_curve.py` | SNR 退化曲线（暗色主题，2 种子 mean±std） |
| `hbc_aug_results/snr_sweep.json` | 全部原始数字（逐模型逐档） |
| `hbc_aug_results/snr_curve_tcnet.png` | SNR 退化曲线图 |
| `eeg_tcnet_4class_lif_hbc_{noaug,aug}_s{42,123}.pth` | 4 组训练最优权重（含增强配置与训练信息） |
| `eeg_tcnet_4class_report_lif_hbc_{...}.txt` | 4 组逐 epoch 报告 |
| 本报告 | 全部数字为真实训练/测试跑出 |

复现命令（必须 `env -i` 隔离，否则 numpy ImportError）：

```bash
cd ~/BCI/20260828/algorithm
# ① 无增强对照（重跑 train_eeg_tcnet_lif.py 同协议，验证 ≈62.42%）
env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin \
  /opt/miniconda3/envs/openmmlab/bin/python train_eeg_tcnet_hbc_aug.py --tag noaug_s42 --seed 42
# ② HBC 噪声增强（AWGN+突发，每 batch SNR∈[10,20]dB 随机）
env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin \
  /opt/miniconda3/envs/openmmlab/bin/python train_eeg_tcnet_hbc_aug.py \
  --aug --snr-min 10 --snr-max 20 --tag aug_s42 --seed 42
# ③ SNR 扫描评估（对比增强 vs 无增强）
env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin \
  /opt/miniconda3/envs/openmmlab/bin/python eval_hbc_snr_sweep.py \
  --ckpts eeg_tcnet_4class_lif_hbc_noaug_s42.pth,eeg_tcnet_4class_lif_hbc_aug_s42.pth \
  --names 无增强,增强
```

---

## 2. 训练结果（clean，cross-session 四分类）

| 种子 | 无增强 | HBC 增强 | Δ |
|---|---|---|---|
| 42  | 62.42% | **63.58%** | +1.16 |
| 123 | 60.42% | **61.11%** | +0.69 |
| **mean ± std** | **61.42 ± 1.00** | **62.35 ± 1.24** | **+0.93** |

- 协议与既有最优完全一致：seed42 无增强 **62.42%**（最佳 val 66.75% @ epoch 40，
  早停 @65）与 `eeg_tcnet_4class_lif_t1.0_tau2.0_T8.pth` 逐位复现，协议无回归。
- 增强模型收敛略慢（噪声正则化）：seed42 最佳 val @ epoch 86（早停 @111），
  但最终精度不降反升。
- **结论：HBC 噪声增强在 clean 上持平（两个种子均不降，均值 +0.93pp）。**

---

## 3. SNR 扫描（低 SNR 鲁棒性，核心结果）

测试噪声 = 5bit 量化（qrange=4.0）+ AWGN，σ/H = 10^(-SNR/20)；测试噪声固定种子
20240828（所有模型看到同一份噪声实现）。**2 种子 mean ± std**：

| 测试 SNR | 无增强 (baseline) | HBC 增强 | Δ(增强−无增强) |
|---|---|---|---|
| ∞（clean） | 61.42 ± 1.00 | 62.35 ± 1.24 | **+0.93** |
| 20 dB | 60.47 ± 1.26 | 61.25 ± 0.29 | +0.78 |
| 16 dB | 59.24 ± 0.56 | 60.13 ± 0.45 | +0.89 |
| 12 dB | 53.18 ± 0.25 | 58.81 ± 0.02 | **+5.63** |
| **8 dB** | **40.23 ± 1.51** | **54.71 ± 0.04** | **+14.47** |
| 4 dB | 27.82 ± 0.93 | 42.03 ± 0.36 | **+14.22** |
| 8dB+突发（附加域） | 41.57 ± 1.76 | 53.39 ± 0.35 | **+11.82** |

单位：%（mean ± std，2 种子）；曲线见 `hbc_aug_results/snr_curve_tcnet.png`。

### 退化曲线与拐点后移

- **干净 → 8 dB 退化幅度**：无增强 **−21.19pp**（61.42 → 40.23），增强 **−7.64pp**
  （62.35 → 54.71）——增强让退化斜率减为 1/2.8。
- **拐点后移**（以「精度 ≥50%」为崩塌判据，线性插值）：
  - 无增强：12 dB（53.18%）→ 8 dB（40.23%）跨 50% 的拐点 ≈ **11.0 dB**；
  - 增强：8 dB 处仍 54.71%（≥50%），4 dB（42.03%）才跌破，拐点 ≈ **6.5 dB**。
  - 即 **SNR 崩塌拐点后移 ≈ 4.5 dB**，把解码失效边界整体下推，覆盖并越过板载
    SNR 瀑布拐点区间（8~14 dB）。
- **稳定性**：8 dB 处无增强种子 std = ±1.51pp，增强 std = ±0.04pp——增强在最恶劣
  SNR 下**既更准又更稳**（与 EEGNet 消融结论一致，且此处压得更狠）。

### 逐种子原始数字（诚实全量）

| 种子 | 配置 | clean | 20dB | 16dB | 12dB | 8dB | 4dB | 8dB+突发 |
|---|---|---|---|---|---|---|---|---|
| 42 | 无增强 | 62.42 | 61.73 | 59.80 | 52.93 | 38.73 | 26.89 | 39.81 |
| 42 | HBC增强 | 63.58 | 61.54 | 59.68 | 58.80 | 54.67 | 41.67 | 53.05 |
| 123 | 无增强 | 60.42 | 59.22 | 58.68 | 53.43 | 41.74 | 28.74 | 43.33 |
| 123 | HBC增强 | 61.11 | 60.96 | 60.57 | 58.83 | 54.75 | 42.40 | 53.74 |

两个种子在 8/4 dB 处方向完全一致（增强均 > 无增强 12~15pp），非单种子运气。

---

## 4. 结论

1. **增强 × TCNet 在低 SNR 显著更鲁棒**：8 dB 处 **+14.47pp**（40.23 vs 54.71），
   4 dB 处 **+14.22pp**；退化斜率减为基线 1/2.8，**SNR 崩塌拐点后移 ≈ 4.5 dB**
   （11.0 → 6.5 dB），且低 SNR 种子方差从 ±1.51 压到 ±0.04pp。
2. **clean 持平（略升）**：61.42% → 62.35%（+0.93pp），两个种子均不降——增强不是
   拿 clean 换鲁棒性。
3. **对「约束驱动鲁棒解码」纲领的印证 +1**：把 HBC 信道噪声（5bit 量化 + AWGN +
   突发）作为训练约束，收益落在低 SNR 瀑布区而非训练/干净条件——世界主流模型
   （EEG-TCNet）与类脑 LIF 化在此处行为一致，方法可迁移。

---

## 5. 讨论（诚实性声明与局限）

1. **为何 TCNet 上收益（+14.5pp）比 EEGNet（+3.94pp）大？** 两点解释：
   - LIF 率编码前端对输入噪声更敏感：强噪声下膜电位积分被噪声主导、发放率饱和/塌缩，
     特征严重退化；训练期噪声约束直接教 LIF 层在噪声下维持有效发放，因此收益被放大；
   - 本实验 8 dB 处基线已近崩塌（40.23%，离 chance 25% 很近），而 EEGNet 基线 8 dB
     仍 66.36%——基线越低，增强的「守住」空间越大。两者都验证了「增强把失效边界
     下推」这一共同机制。
2. **单次评估只测一次**：final test 为 cross-session eval（E 会话，不参与早停），
   2 个训练种子取 mean±std；测试噪声固定种子，所有模型看到同一份噪声。
3. **增强强度**：训练 SNR 每 batch 在 [10,20] dB 均匀采样（+ 突发 σb/H=0.3, len=16,
   p=0.005）。未做强度扫描；[10,20] 是任务建议区间，实测已足够（8 dB 泛化收益显著）。
4. **局限**：① 未做「匹配 SNR 训练」（每档 SNR 各自训练）对照；② 2 种子而非 EEGNet
   消融的 3 种子（训练预算 ≤4 组），8 dB 处结论方向在两个种子上完全一致，可信度足够；
   ③ 未评估真实板载链路（RV1106 量化）下的端到端表现，仅仿真链路噪声。
5. **未改动任何 `~/BCI/luckfox-deploy` 代码**：仅复用 `dataset.py`/`eeg_tcnet_lif.py`
   并读取 `data_4class/{train,eval}.npz`；噪声模型直接 `import hbc_noise_advanced.py`
   （其 5bit 量化与 luckfox-deploy/model/noise.py 交叉校验 max_diff≈2.3e-7）。
