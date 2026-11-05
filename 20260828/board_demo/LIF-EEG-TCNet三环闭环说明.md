# LIF-EEG-TCNet 三环闭环说明（类脑 + BCI + HBC，22ch 四分类，板载实测）

> 版本：2026-08-29 · 板卡：Luckfox Pico Ultra（RV1106G3，Cortex-A7 @1.2GHz，512MB RAM）
> 数据：BCIC IV 2a 四分类评估集全量 2592 试次（`lif_4class_test_full.bin`，22ch × 500 样本/试次，标签 0=左手 1=右手 2=双脚 3=舌头）
> 解码器：LIF-EEG-TCNet（surrogate 发放率版，thr=1.0 / tau=2.0 / T=8，含 TCN 膨胀因果残差块）

## 1. 三环闭环结构

本 demo 在端侧芯片 RV1106 上，把「类脑解码（环1）+ 脑机接口（环2）+ 人体信道（环3）」串成一条完整闭环：
真实脑电先被脉冲类脑模型解码，再把**脑电信号本身**（不是 2-bit 类别标签）经 HBC 人体信道传输，收端用同一类脑模型再解码一次，对比「直连 vs 经 HBC」两条路径的准确率。

```
┌──────────────────────────────────────────────────────────────────────┐
│ 环1·类脑：LIF-EEG-TCNet（surrogate 发放率版）                          │
│           conv1 → depthwise → LIF → pool → separable → pointwise      │
│           → LIF → pool → TCN块0(d=1) → TCN块1(d=2) → GAP → Linear(4)  │
│           （8 个 LIF 位点：act1/act2 + 每块 act_a/act_b/act_res）        │
│ 环2·BCI ：读真实脑电 (22ch × 500) 直连解码 → pred_d                     │
│ 环3·HBC ：量化 8bit → 逐 bit OOK 调制(1MHz EQS 载波) → 体表信道          │
│           (IT'IS 4-Cole-Cole 色散分层 + AWGN) → 包络解调 → 反量化重建   │
│ 收端解码：经 HBC 重建的脑电再过 LIF-EEG-TCNet → pred_h                  │
└──────────────────────────────────────────────────────────────────────┘
```

**可证伪指标**：`经HBC准确率 − 直连准确率 ≈ 0`，且 McNemar 配对检验不显著 —— 证明 HBC 人体信道传输
没有破坏类脑解码精度，三条环都成立且能闭环工作。只传 2-bit 类别标签是控制组（任何无线都能传 2 bit），
传脑电本身才能证明 HBC 信道真的能承载脑机接口信号。

## 2. 交付物清单（board_demo/）

| 文件 | 说明 |
|------|------|
| `lif_surrogate_fusion_tcnet.c` | 三环闭环主程序（本次新增，由 `lif_surrogate_fusion_4class.c` 复制改造） |
| `lif_surrogate_infer_tcnet.c` + `lif_tcnet_snn_weights.h` | LIF-EEG-TCNet 纯 C 推理（只读复用，含 `#ifdef LIF_TCNET_INFER_MAIN` 单测开关） |
| `lif_surrogate_fusion_tcnet` | zig 交叉编译 ARM 静态二进制（板载运行） |
| `lif_surrogate_fusion_tcnet_host` | 宿主机 cc 编译版（bit-exact 对账 / 预期值） |
| `lif_surrogate_fusion_tcnet_stream.log` | 板载真实输出日志（stream 模式，含逐窗口行 + 末尾全量汇总） |
| `demo_visualizer.py` / `demo.sh` | 上位机可视化（复用，见 §7） |
| `LIF-EEG-TCNet板载实测说明.md` | TCNet 解码器本身的板载精度实测（62.4228%，1618/2592） |

外部只读引用（不改不拷）：`~/BCI/luckfox-deploy/board/hbc_channel.{c,h}` + `cole_cole_params.h`
（IT'IS V4.2 4-Cole-Cole 色散分层信道模型，1MHz 体表增益 ~-49.5dB，颅骨为主衰减层）。

## 3. 改动说明（vs lif_surrogate_fusion_4class.c）

按任务要求「复制 + 换解码」，只动了 **2 处调用 + 1 处声明 + 文案**，HBC 闭环链一字未改：

1. 声明：`int lif_surrogate_infer_4class(...)` → `int lif_surrogate_infer_tcnet(...)`（签名
   `int (*)(const float *x, float *logits)` 完全一致，22ch 输入、4 维 logits、返回 argmax 类别）；
2. 直连解码调用：`pred_d = lif_surrogate_infer_4class(x, logits_d)` → `lif_surrogate_infer_tcnet`；
3. 收端解码调用：`pred_h = lif_surrogate_infer_4class(x_rec, logits_h)` → `lif_surrogate_infer_tcnet`；
4. 其余原样保留：数据读取（int32 N=2592 + float32×11000 + int32 label）、`LABEL_CN[4]`、
   量化 8bit(±4) → MSB-first OOK → 信道(LCG 种子 88172645463325252 + Box-Muller) → 解调 → 反量化、
   直连/经HBC/一致率/每类精度/McNemar b/c/耗时统计、`[stream]` 逐窗口输出。

`lif_surrogate_infer_tcnet.c` 的 `main` 被 `#ifdef LIF_TCNET_INFER_MAIN` 挡掉，链接进本程序时只贡献
`lif_surrogate_infer_tcnet` 符号，零改动复用。

## 4. 编译

```bash
# 宿主机（bit-exact 对账，-ffp-contract=off 禁 FMA 收缩）
cc -O2 -ffp-contract=off -I /Users/heyixuan/BCI/luckfox-deploy/board \
   lif_surrogate_fusion_tcnet.c lif_surrogate_infer_tcnet.c \
   /Users/heyixuan/BCI/luckfox-deploy/board/hbc_channel.c \
   -o lif_surrogate_fusion_tcnet_host -lm

# 交叉编译（RV1106 = ARMv7-A musl 静态；zig 勿加 -mcpu/-march，默认即 ARMv7 硬浮点）
~/luckfox/zig-0.13.0/zig cc -target arm-linux-musleabihf -O2 -static \
   -I /Users/heyixuan/BCI/luckfox-deploy/board \
   lif_surrogate_fusion_tcnet.c lif_surrogate_infer_tcnet.c \
   /Users/heyixuan/BCI/luckfox-deploy/board/hbc_channel.c \
   -o lif_surrogate_fusion_tcnet -lm
```

## 5. 板载实测（全量 2592 试次，真实 adb 输出）

```
~/luckfox/platform-tools/adb shell "cd /root && ./lif_surrogate_fusion_tcnet /root/lif_4class_test_full.bin stream"
```

完整日志见 `lif_surrogate_fusion_tcnet_stream.log`（2592 行逐窗口 + 末尾汇总），汇总块摘录：

```
============================================================
  类脑 + BCI + HBC 三环闭环验证 — RV1106 端侧（22ch 四分类，板载实测）
  类脑解码器    : LIF-EEG-TCNet（surrogate 发放率版，T=8，decay=0.6065，TCN 残差块，四分类）
  试次数        : 2592（四分类标签 0=左手 1=右手 2=双脚 3=舌头）
  体表信道损耗  : 46.3 dB @ 1 MHz（IT'IS Cole-Cole 色散，颅骨主衰减）
  量化          : 8 bit / 样本（范围 ±4）

  ── 三环闭环解码结果 ──
  直连解码准确率 : 62.4228% (1618/2592)
  经HBC解码准确率: 62.3457% (1616/2592)
  两者一致率     : 96.8750% (2511/2592)
  >> 准确率下降   : 0.0772 个百分点
  McNemar 分歧   : b(直连对/HBC错)=32, c(直连错/HBC对)=30

  ── 每类精度（直连 / 经HBC）──
  类别0(左手) : 直连 64.35% (417/648) | 经HBC 63.89% (414/648)
  类别1(右手) : 直连 66.82% (433/648) | 经HBC 66.67% (432/648)
  类别2(双脚) : 直连 60.96% (395/648) | 经HBC 61.57% (399/648)
  类别3(舌头) : 直连 57.56% (373/648) | 经HBC 57.25% (371/648)

  ── 传输质量 ──
  误码率 BER     : 3.11e-07 (71/228096000 bit)
  信号 MSE(量化) : 0.000256

  ── 性能 ──
  类脑解码总耗时 : 261274.7 ms（5184 次前向 = 直连 + 经HBC 各 2592）
  单次前向耗时   : 50.400 ms
  平均单窗口耗时 : 136.8 ms（端到端：直连解码 + HBC + 收端解码）
  总耗时         : 355429.7 ms（2592 试次 × 11000 样本 × 8 bit）
============================================================
```

### 5.1 结果解读

| 指标 | 数值 | 结论 |
|------|------|------|
| 直连精度 | **62.4228% (1618/2592)** | 与 `lif_surrogate_infer_tcnet` 单独板载实测（62.4228%, 1618/2592）**完全一致** → 融合闭环直连路径没有引入任何偏差 |
| 经HBC精度 | 62.3457% (1616/2592) | 只差 2 个试次，Δ = **−0.0772 pp** |
| 一致率 | 96.8750% (2511/2592) | 81 个试次解码结论被量化噪声翻转（32 直连对/经HBC错 + 30 直连错/经HBC对 + 19 双双错但不同类） |
| McNemar | b=32, c=30 → χ²=0.0645 (无校正) / 0.0161 (校正), **p≈0.80 / 0.90** | p 远大于 0.05，**不显著** → 经 HBC 传输不改变解码表现 |
| 单次前向 | 50.400 ms | 与 TCNet 解码器单独实测 50.3ms 一致（TCN 块仅 ~0.16M MACs，conv1 5.5M MACs 仍是绝对大头） |
| 端到端窗口 | 136.8 ms 平均 / p99 155.9 ms | 2s 窗口 → 实时余量 **13×**（p99 口径），符合预期 100–150ms |
| BER | 3.11e-07 (71/228096000 bit) | 46.3dB 信道 + 8bit 量化的误码率，噪声对解码影响可忽略 |

**「直连 vs 经HBC」结论**：经 HBC 的 0.0772pp 损失（2/2592 试次）落在采样噪声内
（McNemar p≈0.8~0.9 完全不显著），且方向与 LIF-EEGNet 二分类闭环（Δ−0.15pp）一致 ——
HBC 人体信道传输不破坏类脑解码精度，**三环闭环成立**。

**板载 vs 宿主机 bit-exact**：同一 LCG 种子 + double 噪声路径下，板载与 `lif_surrogate_fusion_tcnet_host`
输出逐位一致（直连 1618/2592、经HBC 1616/2592、b/c=32/30、BER 71/228096000、MSE 0.000256 全同）——
宿主机数字就是板载预期值，板载日志无虚构空间。

**与 LIF-EEGNet 四分类闭环对比**（同为 22ch/4类、同一条 HBC 链、同 2592 试次）：

| 指标 | LIF-EEGNet 4class | **LIF-EEG-TCNet（本次）** |
|------|------|------|
| 直连精度 | 56.8673% (1474/2592) | **62.4228% (1618/2592)（+5.56pp）** |
| 经HBC精度 | 57.1373% (1481/2592) | 62.3457% (1616/2592) |
| 一致率 | 97.84% | 96.88% |
| McNemar | b=17/c=24, p≈0.27 不显著 | b=32/c=30, p≈0.80 不显著 |
| 单次前向 | 49.8 ms | 50.4 ms |
| 平均单窗口 | 135.3 ms | 136.8 ms |

类脑结构（TCN 时域卷积残差）带来 +5.56pp 的实打实解码增益，HBC 链不受影响 ——
TCNet 版三环闭环在精度上全面优于 EEGNet 版。

## 6. 统计口径

- **一致率**：同一试次 `pred_h == pred_d` 的比例（量化噪声不翻转解码结论的比例）。
- **McNemar**：2×2 分歧列联表 `b`=直连对/经HBC错、`c`=直连错/经HBC对；
  χ²=(|b−c|−1)²/(b+c)（连续性校正），p = chi2.sf(χ², df=1)。
  p>0.05 → 经 HBC 与直连差异不显著 → 三环闭环成立。
- 板载与宿主机同一 LCG 种子、double 噪声路径 bit-exact，宿主机数字可直接当板载预期值。

## 7. 如何接 demo_visualizer.py

`demo_visualizer.py` 已支持四分类（`LABEL_CN=["左手","右手","双脚","舌头"]`），无需改代码，只需把
`BOARD_BIN` / `BOARD_CMD` 指向 TCNet 版：

```python
BOARD_BIN = "lif_4class_test_full.bin"   # 数据不变（四分类全量）
BOARD_CMD = "cd /root && ./lif_surrogate_fusion_tcnet /root/lif_4class_test_full.bin stream"
```

用法（与四分类版完全一致）：

```bash
# 实时流式（板子在线）
/opt/miniconda3/envs/openmmlab/bin/python demo_visualizer.py --live
# 离线回放已抓取的板载日志
/opt/miniconda3/envs/openmmlab/bin/python demo_visualizer.py --file lif_surrogate_fusion_tcnet_stream.log
# 纯终端汇总（全量 2592 行绘图是 O(n²) 会卡，离线验收用 --no-plot）
/opt/miniconda3/envs/openmmlab/bin/python demo_visualizer.py --file lif_surrogate_fusion_tcnet_stream.log --no-plot
```

要点：
- **stream 是位置参数**（C 里 `argc>=3` 判断），不是 flag；
- **子集 ≠ 全量**：`--file` 回放前 N 行时精度是子集值，全量权威数字以日志末尾汇总为准
  （演示时主动标注，避免观众拿子集数字发难）；
- 抓流式日志别用 `adb shell "cmd" | head -n N`（head 退出后远端进程跑完全部 2592 窗口，本地挂死），
  用 `terminal(background=true)` 重定向到文件后按行数轮询、再 `head -n N` 截断；
- 本 log 首窗口 137.7ms 与后续基本一致（本次未见启动尖峰；若观察首窗口偏高属 adb 启动开销，非真实耗时）。

## 8. 与 LIF-EEGNet 四分类闭环对比（实测）

| 指标 | LIF-EEGNet 4class（已实测） | LIF-EEG-TCNet（本次已实测） |
|------|------|------|
| 直连精度 | 56.8673% (1474/2592) | **62.4228% (1618/2592)**（+5.56pp） |
| 解码器结构 | EEGNet 变体 | + TCN 膨胀因果残差块 + GAP 读出头 |
| 单次前向 | 49.8 ms | 50.4 ms（TCN 仅 ~0.16M MACs，conv1 5.5M 仍是大头） |
| HBC 链 | 8bit OOK @1MHz 同一条链 | 同一条链（复用，不改） |
