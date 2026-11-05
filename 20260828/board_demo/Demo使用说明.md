# Demo 使用说明 — RV1106 板载部署验证 + 可演示 Demo

> 本文档讲清楚：怎么连板、怎么跑端侧解码、怎么起上位机可视化、每一步预期输出。
> 所有板载数字均为 **2026-08-28 adb 实测原样记录**，未做任何修饰；跑不了的地方如实标注。

---

## 0. 这个 Demo 是什么

一条**可照着操作跑通**的演示链路：

```
板载 RV1106（端侧）                        上位机（Mac）
─────────────────────                    ─────────────────────
./hbc_realtime <bin>                     demo_visualizer.py
  ① EEGNet 直连解码（CPU ~23ms/前向）      读取 adb 流式输出
  ② 5bit 量化 → OOK 调制 → HBC 信道仿真    逐窗口解析
     → 包络解调 → 反量化（"HBC 回传"）      终端实时打印 + 暗色图
  ③ 收端二次解码（经HBC）                  累计精度/延迟/一致率
```

- **端侧解码**：EEGNet（8 通道二分类运动想象）纯 C 移植，BN 已折叠，与 PyTorch 逐位一致（误差 ~1e-6）。
- **HBC 回传**：板载 `hbc_channel.c` 的 IT'IS Cole-Cole 分层信道模型（1MHz 载波，46.3 dB 损耗）+ OOK 调制解调，**信道是模型不是测量**（见 luckfox-deploy/RESULTS.md 诚实边界）。
- **上位机可视化**：`demo_visualizer.py` 实时解析板载输出，终端打印 + 暗色 matplotlib 四面板图。

---

## 1. 环境要求

| 依赖 | 路径 / 版本 | 说明 |
|------|-------------|------|
| adb | `~/luckfox/platform-tools/adb` | 板子序列 `5caaa04f9cdb6ba2` |
| Python | `/opt/miniconda3/envs/openmmlab/bin/python` | 必须 `env -i` 隔离运行（否则 numpy ImportError）|
| numpy | 1.24.3 | 必需 |
| matplotlib | 3.7.5 | 可选（缺省自动退化为纯终端模式）|

板载二进制（已上板，无需重新编译，本 Demo 直接复用）：
`/root/` 下的 `eegnet_infer`、`hbc_realtime`、`hbc_mcnemar`、`hbc_realtime_throttle`、
`hbc_snr_sweep`、`hbc_benchmark`、`hbc_bitdepth_scan`、`hbc_fusion`、`hbc_bci_loop` 等。

---

## 2. 文件清单（本目录 board_demo/）

| 文件 | 作用 |
|------|------|
| `demo.sh` | 一键脚本（check / live / offline / capture / mock）|
| `demo_visualizer.py` | 上位机可视化（实时解析 + 终端 + 暗色图）|
| `mock_board_output.log` | 真实板载输出抓取（100 窗口），供离线回放 |
| `demo_visualizer.png` | 可视化脚本自动产出的结果图 |
| `Demo使用说明.md` | 本文档 |
| `Demo替换LIF-EEGNet说明.md` | 后续把 EEGNet 换成 LIF-EEGNet 脉冲模型的路线 |

---

## 3. 板子现状验证记录（2026-08-28 adb 实测）

### 3.1 板子在线 + 系统信息

```bash
$ ~/luckfox/platform-tools/adb devices
List of devices attached
5caaa04f9cdb6ba2	device

$ ~/luckfox/platform-tools/adb shell uname -a
Linux luckfox 5.10.160 #12 Sat Jun 7 17:19:04 CST 2025 armv7l GNU/Linux
```

### 3.2 /root 二进制清单（实测存在）

```
eegnet_infer(844612)  hbc_mcnemar(928888)   hbc_snr_sweep(930764)
hbc_realtime(934936)  hbc_realtime_throttle(947296)  hbc_benchmark(948924)
hbc_bitdepth_scan(926832)  hbc_fusion(942216)  hbc_bci_loop(1321508)
eegnet_npu(20984)     npu_audit(12488)      eegnet.rknn(80660)
fusion_test.bin(1600404)  fusion_test_full.bin(20741188)
X.bin  y.bin  cpu_logits.bin  ...
```

### 3.3 基线一：EEGNet 单次前向自检（`./eegnet_infer`）

```bash
$ adb shell "cd /root && ./eegnet_infer"
预测类别: 0, logits=[0.130, -0.050]
单次前向耗时: 23.25 ms        # 两次实测 23.25 / 23.26 ms，与旧权重 18.5ms 不同是因为新权重
```

> 说明：当前板载是 **76.16% 噪声增强版新权重**，单次前向 23.2ms（旧权重 18.5ms）。
> `eegnet_infer` 是自检程序：用确定性合成输入跑 100 次计时，输出 logits + 单次耗时，不读外部数据。

### 3.4 基线二：端侧解码 + HBC 回传流式（`./hbc_realtime fusion_test.bin`）

```bash
$ adb shell "cd /root && ./hbc_realtime /root/fusion_test.bin 1"
=== 实时流式演示：RV1106 端侧 HBC×BCI 融合（100 窗口 × 1 轮）===
体表信道 46.3 dB @1MHz | 单窗口=2s 脑电

[  0] 真值=左手 直连=右手 经HBC=右手 ✗ | 60.2ms (实时余量 97%)
[  1] 真值=右手 直连=右手 经HBC=右手 ✓ | 59.4ms (实时余量 97%)
...
[ 99] 真值=右手 直连=右手 经HBC=右手 ✓ | 59.4ms (实时余量 97%)

=== 汇总（100 窗口）===
  直连准确率   : 85.0% (85/100)
  经HBC准确率  : 85.0% (85/100)
  两者一致率   : 100.0% (100/100)
  平均单窗口   : 59.4 ms（2s 窗口，实时余量 34×）
```

> ⚠️ **诚实说明**：`fusion_test.bin` 是 100 试次子集，该子集准确率恰好 85%；全量 1296 试次
> 是 76.16%（见 3.5）。演示用 100 试次图快（几秒跑完），不代表全量精度。

### 3.5 基线三：全量显著性检验（`./hbc_mcnemar fusion_test_full.bin`）

```bash
$ adb shell "cd /root && ./hbc_mcnemar /root/fusion_test_full.bin"
=== 直连 vs 经HBC 配对预测（1296 试次 × 5 种子）===
噪声 σ/H=0.1，位深 5bit，信道 46.3 dB @1MHz

直连准确率  : 76.16% (4935/6480)
经HBC准确率 : 76.17% (4936/6480)

配对 2×2 列联表（跨 5 种子汇总）：
                      经HBC对    经HBC错
  直连对                 4855        80
  直连错                   81      1464

  b(直对H错)=80  c(直错H对)=81  → McNemar 由 Python 算 p 值
```

McNemar 检验（Python 侧计算，本会话实测）：
- χ² = (b−c)²/(b+c) = (80−81)²/161 = **0.0062**
- p = chi2.sf(0.0062, 1) = **0.9372**（不显著）；精确二项检验 p = 1.0
- **结论**：新权重（76.16%）下，经 HBC 回传引入的差异**统计上不显著**（直连 76.16% vs 经HBC 76.17%，几乎一致）。

> 注：旧权重（76.39%）曾测出 b=105/c=65、p=0.0028（显著但量级小）。换新权重后 b/c 掉到 80/81，
> p 不显著——这是权重更新带来的真实变化，如实记录，不沿用旧数字。

---

## 4. 快速开始（三条路径）

### 4.0 准备

```bash
cd ~/BCI/20260828/board_demo
chmod +x demo.sh demo_visualizer.py
```

### 4.1 路径 A：完整实时链路（需要板子在线）★ 主演示

```bash
bash demo.sh check    # 先确认板子在线 + 基线可复现
bash demo.sh live     # 板载流式解码 → 上位机实时可视化
```

**预期输出（live）**：
1. 终端先打印 `[数据源] adb 实时驱动板载`。
2. 板载 `hbc_realtime` 逐窗口输出被实时解析，终端以清屏方式滚动显示：
   ```
   ========================================================================
     RV1106 端侧 HBC×BCI 融合 —— 上位机实时可视化
   ========================================================================
     窗口 #12   真值=右手  直连=右手  经HBC=右手  ✓
     耗时   59.3 ms（2s 窗口，实时余量 97%）
   ------------------------------------------------------------------------
     直连准确率   :  85.0%   ██████████████████████████████░░░░░░░░░░
     经HBC准确率  :  85.0%   ██████████████████████████████░░░░░░░░░░
     两者一致率   : 100.0%   （HBC 回传是否改变解码结果）
     延迟 p50/p99 :  59.4 /  59.6 ms（p99 < 2000ms 即实时）
   ========================================================================
   ```
3. 同时弹出暗色 matplotlib 四面板窗口（累计精度曲线 / 单窗口延迟 / 逐窗口预测 / ✓✗ 条带）。
4. 结束后自动保存 `demo_visualizer.png`，并打印最终汇总（与板载原汇总一致）。

### 4.2 路径 B：离线回放（无需板子）★ 验收用

```bash
bash demo.sh offline    # 回放 mock_board_output.log（真实板载抓取输出）
```

**预期输出**：与 4.1 相同的可视化与汇总，但数据来自已抓取日志；结果：
```
窗口总数 100 | 直连 85.0% | 经HBC 85.0% | 一致率 100.0% | p99 59.6ms | 实时余量 34×
```

### 4.3 路径 C：纯合成数据（无板子，仅演示可视化本身）

```bash
bash demo.sh mock       # 生成合成窗口，明确标注 MOCK，不冒充真实板载数字
```

---

## 5. demo.sh 全部子命令

```bash
bash demo.sh check      # 板子在线检查 + 基线验证（eegnet_infer / hbc_realtime）
bash demo.sh live       # 完整实时链路（板载流式解码 → 上位机实时可视化）
bash demo.sh offline    # 离线回放 mock_board_output.log
bash demo.sh capture    # 抓一段真实板载输出到 mock_board_output.log（刷新离线数据）
bash demo.sh mock       # 纯合成数据演示可视化
bash demo.sh            # 默认 = check
```

环境变量（可选）：
```bash
ADB=/path/to/adb              # 默认 ~/luckfox/platform-tools/adb
BOARD_SERIAL=5caaa04f9cdb6ba2  # 默认自动取当前唯一 device
```

---

## 6. demo_visualizer.py 参数说明

```bash
python demo_visualizer.py [选项]

  --live          通过 adb 实时驱动板载 hbc_realtime（默认，需板子在线）
  --file FILE     离线回放板载输出日志（如 mock_board_output.log）
  --mock          纯合成数据（无板子，仅演示可视化）
  --adb PATH      adb 路径（默认 ~/luckfox/platform-tools/adb）
  --windows N     mock 模式窗口数（默认 40）
  --no-plot       关闭 matplotlib，仅终端
  --save PNG      最终图保存路径（默认 demo_visualizer.png）
  --timeout SEC   live 模式 adb 超时（默认 180s）
```

直接运行（遵循本项目 Python 环境铁律，`env -i` 隔离）：

```bash
env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin \
    /opt/miniconda3/envs/openmmlab/bin/python demo_visualizer.py --file mock_board_output.log
```

---

## 7. 故障排查

| 症状 | 处理 |
|------|------|
| `adb devices` 无 device | 检查 USB 数据线（非纯充电线）、板子是否进 Maskrom（看 PID，见 rockchip-edge-boards 技能）|
| 找不到 adb | `ADB=/path/to/adb bash demo.sh ...` 覆盖 |
| `numpy ImportError` | 必须用 `env -i` 隔离跑（demo.sh 已内置），别裸跑 `python demo_visualizer.py` |
| 板子中途掉线 | 改用 `bash demo.sh offline` 离线演示，可视化脚本不卡住 |
| matplotlib 缺中文字形 | 已内置 Hiragino Sans GB 字体回退；若仍缺，`--no-plot` 走纯终端 |
| `hbc_mcnemar` 长时间无输出 | 全量 1296×5=6480 配对，板载需 ~6-7 分钟，属正常 |

---

## 8. 演示话术要点（诚实版）

- **"端侧芯片能不能跑 HBC+BCI 融合"** → 能，单窗口端到端 ~59ms，2s 脑电窗口有 34× 实时余量。
- **"HBC 回传是否损伤解码"** → 不显著：直连 76.16% vs 经HBC 76.17%，McNemar p≈0.94。
- **边界**：信道是 IT'IS 分层模型（非真实头骨测量）；NPU 线已关闭（INT8 量化失真），解码走 CPU。
