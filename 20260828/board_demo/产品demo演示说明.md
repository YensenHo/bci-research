# 产品 Demo 演示说明 —— 板载四分类三环闭环 → 上位机可视化

> 一句话：**RV1106 端侧芯片上跑「类脑 LIF-EEGNet 解码 → 真实脑电 → HBC 人体信道回传 → 收端再解码」四分类三环闭环，上位机实时可视化，导师/投资人可现场直接看。**
>
> 本说明对应交付目录 `~/BCI/20260828/board_demo/`，所有数字均为板载实测，未虚构。

---

## 0. 演示链路总览

```
板载 RV1106（端侧）                         上位机 Mac
─────────────────────                     ──────────────────
lif_surrogate_fusion_4class               demo_visualizer.py
① 环1·类脑：LIF-EEGNet 直连解码（四分类）    adb 读取流式输出
② 环2·BCI ：读真实脑电 22ch×500 直连       解析逐窗口结果
③ 环3·HBC ：8bit 量化 → OOK → 体表信道     终端实时打印 + 暗色图
   （IT'IS Cole-Cole 色散 + AWGN）          （4 面板：累计精度 / 延迟 /
④ 收端二次解码（经HBC，四分类）              逐窗口预测 / 命中条带）
```

- **四分类标签**：`0=左手 1=右手 2=双脚 3=舌头`（与板载 `lif_surrogate_fusion_4class.c` 的 `LABEL_CN[4]`、`preprocess_4class.py` 完全一致）
- **数据**：`lif_4class_test_full.bin`，BCIC IV 2a 四分类评估集，2592 试次（每类 648，均衡）
- **板载闭环结论（全量 2592 试次实测）**：直连 **56.87%** / 经HBC **57.14%** / 一致率 **97.84%** / McNemar **p≈0.27 不显著** → 经人体信道传输后类脑解码准确率不下降，三环闭环成立。

### 文件清单

| 文件 | 说明 |
|---|---|
| `demo_visualizer.py` | **已升级四分类**的上位机可视化（--live / --file / --mock 三模式） |
| `lif_4class_stream.log` | **板载真实 stream 抓取**（100 窗口，本演示的主回放素材） |
| `lif_4class_demo.png` | `--file` 回放生成的暗色四分类可视化图（验收产物） |
| `lif_surrogate_fusion_4class` | 板载 ARM 二进制（zig 交叉编译，只读复用，**不重新编译**） |
| `lif_surrogate_fusion_4class.c` | 板载主程序源码（四分类，只读复用） |
| `lif_4class_test_full.bin` | 四分类全量测试数据（2592 试次） |
| `lif_surrogate_infer_4class.c` / `lif_surrogate_weights_4class.h` | 四分类类脑解码器与权重（只读复用） |
| `四分类三环闭环说明.md` | 板载全量实测技术文档（56.87%/57.14%/97.84% 出处） |

---

## 1. 前置准备（2 分钟）

```bash
# ① 板子在线检查（序列号 5caaa04f9cdb6ba2）
~/luckfox/platform-tools/adb devices
#    必须看到: 5caaa04f9cdb6ba2  device

# ② 板载文件确认（四分类二进制 + 数据已部署）
~/luckfox/platform-tools/adb -s 5caaa04f9cdb6ba2 shell \
  "ls -l /root/lif_surrogate_fusion_4class /root/lif_4class_test_full.bin"
```

**环境铁律（本项目约定）**：所有 Python 命令必须用 `/opt/miniconda3/envs/openmmlab/bin/python` 且 `env -i` 隔离运行，否则 numpy 会 ImportError：

```bash
env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin \
    /opt/miniconda3/envs/openmmlab/bin/python demo_visualizer.py --help
```

演示目录：`cd ~/BCI/20260828/board_demo`

---

## 2. 演示流程（推荐顺序：回放 → 实时 → 基线）

### ① 离线回放（主演示路径，3 秒出图，最稳）

```bash
cd ~/BCI/20260828/board_demo
env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin \
    /opt/miniconda3/envs/openmmlab/bin/python demo_visualizer.py \
    --file lif_4class_stream.log --save lif_4class_demo.png
```

**看什么**：
- 终端逐窗口刷新：`窗口 #n 真值=左手 直连=左手 经HBC=左手 ✓ | 134.9ms (实时余量 93%)`，四分类标签轮换出现；
- 四根累计精度条（直连 / 经HBC / 一致率 / 延迟）实时上涨；
- 结束汇总打印每类精度（直连/经HBC 各一列）；
- 生成暗色图 `lif_4class_demo.png`（可直接投屏）。

**预期输出（lif_4class_stream.log 100 窗口实测值，板载真实数据）**：

| 指标 | 实测值（100 窗口） |
|---|---|
| 窗口总数 | 100（idx 0–99，四类均衡 26/25/24/25） |
| 直连准确率 | **69.0%** |
| 经HBC准确率 | **69.0%** |
| 两者一致率 | **98.0%** |
| 延迟 p50/p99 | **134.9 / 154.5 ms**（2s 窗口，实时余量 ~93%，13× 余量） |
| 每类（直连/经HBC） | 左手 88.5/88.5% · 右手 64.0/64.0% · 双脚 70.8/70.8% · 舌头 52.0/52.0% |

> ⚠️ **如实说明（防被追问）**：前 100 窗口准确率（69%）高于全量（56.87%），是数据顺序效应——`lif_4class_test_full.bin` 前 100 试次恰好较易。**全量 2592 试次的权威数字是 直连 56.87% / 经HBC 57.14% / 一致率 97.84%**（见第 2 节 ③），演示时建议主动补一句：「这是前 100 窗口的实时片段，全量 2592 试次的稳定数字在下面这个全量闭环里」。

### ② 板载实时链路（--live，现场感最强）

```bash
cd ~/BCI/20260828/board_demo
env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin \
    /opt/miniconda3/envs/openmmlab/bin/python demo_visualizer.py \
    --live --adb ~/luckfox/platform-tools/adb --save lif_4class_demo.png
```

**这一步在做什么**：脚本通过 adb 在板子上启动
`./lif_surrogate_fusion_4class /root/lif_4class_test_full.bin stream`，
板子**边算边吐**逐窗口结果（每窗口 ~135ms 端到端：直连解码 + HBC 信道 + 收端解码），
上位机逐行解析、实时刷新终端与图。

**看什么**：
- 板载侧（另开终端 `adb shell` 手跑同命令）与上位机画面同步滚动；
- 演示「真·闭环」：真值 → 直连预测 → 经HBC预测，三列同屏对比；
- 每窗口 `✓/✗` 与红色/绿色命中条带实时变化。

**预期**：窗口以 ~7 个/秒推进；默认 180s 超时自动收尾（也可 Ctrl-C）。
如需固定时长展示，可在板载侧直接 `adb shell "/root/lif_surrogate_fusion_4class /root/lif_4class_test_full.bin stream" | head -n 100`。

### ③ 板载全量闭环基线（数字权威版，~5.8 分钟，可选）

```bash
~/luckfox/platform-tools/adb -s 5caaa04f9cdb6ba2 shell \
  "/root/lif_surrogate_fusion_4class /root/lif_4class_test_full.bin"
```

**预期输出（全量 2592 试次，板载实测，一字未改）**：

```
  直连解码准确率 : 56.8673% (1474/2592)
  经HBC解码准确率: 57.1373% (1481/2592)
  两者一致率     : 97.8395% (2536/2592)
  >> 准确率下降   : -0.2701 个百分点
  McNemar 分歧   : b(直连对/HBC错)=17, c(直连错/HBC对)=24   → p≈0.27 不显著
  类别0(左手) : 直连 64.35% | 经HBC 64.81%
  类别1(右手) : 直连 50.15% | 经HBC 50.62%
  类别2(双脚) : 直连 58.64% | 经HBC 58.80%
  类别3(舌头) : 直连 54.32% | 经HBC 54.32%
  误码率 BER     : 3.11e-07 (71/228096000 bit)
  平均单窗口耗时 : 135.3 ms（端到端）
```

**给观众的三句话**：
1. 端侧芯片上真实脑电四分类解码 **56.87%**（四分类随机基线 25%，高于随机 2.3 倍）；
2. 脑电经 **HBC 人体信道**回传后再解码 **57.14%** —— 传输几乎不损失准确率（McNemar p=0.27 不显著）；
3. 每窗口端到端 **135ms**，远小于 2s 窗口 → **实时性成立**，具备做真实闭环控制（光标/机械臂）的余量。

### ④ 兜底：纯离线合成（--mock，板子不在也能演示可视化）

```bash
env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin \
    /opt/miniconda3/envs/openmmlab/bin/python demo_visualizer.py \
    --mock --windows 60 --save lif_4class_demo.png
```

- 生成 60 个合成窗口，准确率 ~57%（与板载实测同量级），终端明确标注 `[MOCK]`，**不冒充真实板载数字**；
- 用途：演示可视化本身、调试、无板子场合。

---

## 3. 暗色图 lif_4class_demo.png 怎么看（四个面板）

| 面板 | 内容 | 观众看点 |
|---|---|---|
| 左上 累计准确率 | 直连（蓝）/ 经HBC（绿）两条累计曲线 + 25% 四分类随机基线 | 两条线几乎重合 = HBC 传输无损；全程高于 25% 虚线 = 有效解码 |
| 右上 延迟 | 每窗口端到端耗时（~135ms）vs 2000ms 窗口边界红线 | 一条平直线贴在底部 = 实时余量巨大（~93%） |
| 左下 逐窗口预测 | 四分类 y 轴（左手/右手/双脚/舌头），蓝点=直连、绿×=经HBC | 四类标签轮换出现；蓝绿几乎重叠 = 回传不改变解码结果 |
| 右下 命中条带 | 每窗口经HBC 是否命中（绿=✓ 红=✗） | 一眼看出实时命中/失误分布 |

---

## 4. 真实化扩展（接真实体耦合测量 / 自采数据后怎么改）

当前 demo 的 HBC 信道是**基于 IT'IS 4-Cole-Cole 色散模型的数值仿真**（FEM 求得的体表增益 H、AWGN 噪声），脑电数据是 **BCIC IV 2a 公开数据集**。向真实产品演进，改动点如下（均为**替换数据/参数**，不改架构）：

### 4.1 接真实体耦合测量（HBC 信道真实化）
- **现状**：`hbc_channel.c` 用 `channel_gain_cc()` 数值求解 1MHz 体表信道增益（~-49.5dB）+ 高斯噪声。
- **改法**：用矢量网络分析仪/示波器实测人体前臂-躯干 1MHz 插入损耗与噪声底，把 `main()` 里的 `H`（信道增益）、`noise_sigma`、`thresh` 三个常量替换为实测值（或实测信道冲激响应做卷积）。
- **不改**：量化→OOK→解调→反量化链路、解码器、上位机解析格式。**改动量：约 3 行常量。**

### 4.2 接自采脑电数据（数据真实化）
- **现状**：`lif_4class_test_full.bin` 来自 BCIC IV 2a 公开集。
- **改法**：自采流程 = 采集（湿电极/干电极 22ch，500Hz，2s 窗口）→ 预处理（带通 4–38Hz、坏道剔除、协方差归一化，对齐 `preprocess_4class.py`）→ 用 `train_4class.py` 重新训练/微调 LIF-EEGNet → 导出权重到 `lif_surrogate_weights_4class.h` → 生成同格式 bin（`gen_4class_bin.py` 的格式：`int32 N + float32×11000 + int32 label`）→ 交叉编译 + 推送板子。
- **不改**：三环闭环主程序、上位机。**改动量：数据与权重文件。**

### 4.3 上位机接真实数据流（传输通道真实化）
- **现状**：`--live` 通过 `adb shell` 读板载 stdout。
- **改法**：把 `iter_live()` 换成串口（板载 UART）/Wi-Fi UDP 数据源，**输出行格式不变**（`[n] 真值=… 直连=… 经HBC=…`），`parse_line()` 与绘图零改动。未来接真实脑电采集盒（如 OpenBCI 流），在 `iter_live` 里做「真值已知→真值未知」切换（演示模式用预标注流，真实模式去真值列）。
- **改动量：一个数据源函数。**

### 4.4 输出端真实化（闭环控制）
- 预测类别（0–3）目前仅可视化；真实化后映射为控制指令：左手→左转、右手→右转、双脚→前进、舌头→急停（光标/轮椅/机械臂），端侧直接输出指令码，上位机显示执行状态。

### 演进路线建议（给投资人的一句话）
> 芯片端解码与信道传输已闭环跑通（56.87% / 57.14% / p=0.27），
> 下一步是「信道实测参数 + 自采脑电 + 串口直连」三步替换，即可从演示系统变成可穿戴样机。

---

## 5. 故障排查

| 现象 | 处理 |
|---|---|
| `adb devices` 无 device | 重插 USB，`adb kill-server && adb start-server`；确认序列号 `5caaa04f9cdb6ba2` |
| `--live` 报错/卡住 | 板子掉线 → 改用 `--file lif_4class_stream.log` 离线演示（结果完全相同，素材即板载真实抓取） |
| numpy ImportError | 忘了 `env -i` 隔离，或没用 `/opt/miniconda3/envs/openmmlab/bin/python` |
| 图中文字是方块 | matplotlib 找不到中文字体；本机已有 Hiragino Sans GB，确认未删 macOS 字体 |
| 想自己抓一份新的 100 窗口 log | `adb shell "/root/lif_surrogate_fusion_4class /root/lif_4class_test_full.bin stream" \| head -n 100 > lif_4class_stream.log`（head 退出后板子进程会自行终止，无需等待 2592 跑完） |

---

*文档生成：2026-08-28 · 板载实测（序列号 5caaa04f9cdb6ba2）· 四分类三环闭环 · 数字出处见 `四分类三环闭环说明.md` 与 `lif_4class_stream.log`*
