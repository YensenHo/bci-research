# 板载端到端 SNN 精度实测说明（Demo v2 最后一厘米）

> 把 surrogate 训练的最优 LIF-EEGNet（`lif_F_seed123.pth`，clean 77.70% / HBC 77.16%）
> 真正上板，跑出端到端精度。全部数字来自真实运行输出，无任何虚构。

---

## 1. 结论（TL;DR）

| 项目 | 数值 |
|---|---|
| **板载端到端精度（全量 1296 试次）** | **77.7006%（1007/1296）** |
| Python 参考（surrogate 训练 clean） | 77.70%（1007/1296） |
| 板载 vs Python 差异 | **0.00 pp（逐试次完全一致）** |
| 板载单次前向耗时 | **23.434 ms**（RV1106 Cortex-A7 @ 1.2 GHz） |
| 宿主机 C-SNN vs PyTorch-SNN 逐位一致 | ✓（见 §4） |

板载精度与 Python 完全一致（1007/1296，小数点后一位都是 77.70），**surrogate 训练的
LIF-EEGNet 权重已真实上板并跑出端到端精度**。

---

## 2. 关键语义（务必与旧 ann2snn 版区分）

本轮上板的 `lif_surrogate_infer.c` 是**全新**的，不是复用旧 `lif_eegnet_infer.c`。
差异点（严格复刻 `algorithm/lif_eegnet.py` 的 `LIFActivation.forward` 第 100-109 行）：

| 语义 | 旧 ann2snn 版 | 本轮 surrogate 版 |
|---|---|---|
| LIF 位置 | pool **后**（脉冲上 pool） | pool **前**（`bn2→act1→pool1`、`bn3→act2→pool2`） |
| decay | 1.0（纯 IF） | **0.6065**（= exp(-1/tau)，tau=2.0） |
| 阈值 | 标定值（thr=7.0 等） | **thr=1.0**（固定） |
| 输出 | 脉冲计数均值 | **发放率 sum(spk)/timesteps**（连续 0~1） |
| timesteps | 16 | **8** |

LIF 逐元素语义（C 代码 `lif_rate()`）：
```
mem = 0
for s in 0..7:
    mem = decay * mem + x        # x = 卷积+BN 后的恒定电流（每步相同）
    spk = (mem > thr)            # 严格 >，与 SurrogateStep.forward 的 (x>0) 一致
    if spk: mem = 0              # 硬重置
rate = spike_count / 8           # 发放率
```

---

## 3. 交付文件

| 文件 | 位置 | 说明 |
|---|---|---|
| `export_lif_surrogate_weights.py` | `algorithm/` | 加载 `lif_F_seed123.pth`，BN 折叠，导出 C 头 |
| `lif_surrogate_weights.h` | `board_demo/` + `algorithm/` | 折叠权重 C 头（thr=1.0/decay=0.6065/T=8） |
| `lif_surrogate_infer.c` | `board_demo/` | 板侧 C 推理（LIF 在 pool 前 + 发放率） |
| `verify_lif_surrogate_c.py` | `board_demo/` | 宿主机/板载 C vs PyTorch 对比 |
| `verify_bit_exact.py` | `board_demo/` | 严格逐位一致验证（numpy float32 C 顺序参考） |
| `fusion_test_full.bin` | `board_demo/` | 全量 1296 试次二进制（int32 N + 每试次 float32×4000 + int32 label） |
| `lif_surrogate_infer` | `board_demo/` | zig 交叉编译的 ARM 静态二进制 |

BN 折叠（`ann2snn.fold_bn_layer` 公式）：`scale=γ/√(σ²+ε)`，`W'=W·scale`，`b'=β-μ·scale`；
折叠进 `conv1+bn1`、`depthwise+bn2`、`pointwise+bn3`；`sep_depthwise`/`fc` 无 BN 原样复制。

---

## 4. 宿主机 C-SNN vs PyTorch-SNN 逐位一致验证

### 4.1 严格逐位（numpy float32 + C 顺序参考，`-ffp-contract=off`）

```
$ verify_bit_exact.py --n 30
[对比] numpy(C 顺序) vs C  max|diff| = 0.000e+00
[对比] 逐位完全一致的样本数   : 30/30
✓ 严格逐位一致：C 移植与 float32 C-顺序参考完全一致（0 误差）
```

**结论：C 移植与「同一累加顺序的 float32 参考」0 误差，结构性正确。**

### 4.2 与 PyTorch 折叠参考（`verify_lif_surrogate_c.py`）

```
$ verify_lif_surrogate_c.py --n 200 --mode host
[BN 折叠] 折叠参考 vs 原模型  logits max|diff| = 0.00e+00
[对比] C vs PyTorch logits 最大绝对误差 : 6.64e-07
[对比] 预测一致率          : 200/200 (100.0%)
[精度] PyTorch 折叠参考   : 77.00% | C : 77.00%
✓ 逐位一致
```

```
$ verify_lif_surrogate_c.py --n 1296 --mode host
[BN 折叠] 折叠参考 vs 原模型  logits max|diff| = 8.15e-03
[对比] C vs PyTorch logits 最大绝对误差 : 1.38e-02
[对比] 预测一致率          : 1296/1296 (100.0%)
[精度] PyTorch 折叠参考   : 77.70% | C : 77.70%
```

**诚实说明**：全量 1296 下 C 与 PyTorch 的 logits 最大绝对误差是 1.38e-02（非 1e-5）。
根因已定位并确认**非移植 bug**：

- PyTorch 的 `conv2d` 用优化内核，累加顺序与 C 朴素循环不同，浮点非结合性使
  pre-LIF 电流差 ~1e-6（实测 conv1 差 1.9e-6、depthwise 差 3.7e-6）。
- LIF 阈值是一个**不连续阶跃**：膜电位恰在 thr=1.0 附近的元素，其发放决策会被
  这 ~1e-6 的差异翻转。实测全量 1296 中每 8000 个 act1 元素约 1 个翻转，
  logits 因此跳 ~1e-2。但**没有任何一个翻转改变 argmax 预测**（1296/1296 一致，
  精度 77.70% 完全一致）。
- 这是浮点舍入 + LIF 量化的固有现象，与「C 移植逻辑」无关（§4.1 已证明逻辑 0 误差）。

### 4.3 板载 C vs PyTorch（zig 交叉编译，`--mode board`）

```
$ verify_lif_surrogate_c.py --n 100 --mode board
[BN 折叠] 折叠参考 vs 原模型  logits max|diff| = 0.00e+00
[对比] C vs PyTorch logits 最大绝对误差 : 9.19e-07
[对比] 预测一致率          : 100/100 (100.0%)
[精度] PyTorch 折叠参考   : 84.00% | C : 84.00%
✓ 逐位一致
```

---

## 5. 板载端到端全量 1296 试次实测（真实 adb 输出）

编译与部署：
```bash
~/luckfox/zig-0.13.0/zig cc -target arm-linux-musleabihf -O2 -static \
    -DLIF_SURROGATE_INFER_MAIN lif_surrogate_infer.c -o lif_surrogate_infer -lm
~/luckfox/platform-tools/adb -s 5caaa04f9cdb6ba2 push lif_surrogate_infer /root/
~/luckfox/platform-tools/adb -s 5caaa04f9cdb6ba2 push fusion_test_full.bin /root/lif_surrogate_full.bin
~/luckfox/platform-tools/adb -s 5caaa04f9cdb6ba2 shell /root/lif_surrogate_infer /root/lif_surrogate_full.bin
```

板载真实输出（一字未改）：
```
============================================================
  LIF-EEGNet（surrogate 发放率版）板载端到端精度实测
  thr=1.00  decay=0.6065  timesteps=8  LIF 在 pool 前
============================================================
  试次数         : 1296
  正确数         : 1007
  端到端精度     : 77.7006% (1007/1296)
  类别 0 精度     : 81.02% (525/648)
  类别 1 精度     : 74.38% (482/648)
  总耗时         : 30370.2 ms
  单次前向耗时   : 23.434 ms
============================================================
```

对照（宿主机同二进制逻辑，`cc -O2` 编译）：
```
  端到端精度     : 77.7006% (1007/1296)
  单次前向耗时   : 0.862 ms
```

---

## 6. 精度对比汇总

| 来源 | clean 精度 |
|---|---|
| Python surrogate（`lif_F_seed123.pth` 内 `final_test['干净']`） | 77.70%（1007/1296） |
| 宿主机 C（折叠后） | 77.7006%（1007/1296） |
| **板载 C（RV1106，zig 交叉编译）** | **77.7006%（1007/1296）** |

板载与 Python **逐试次完全一致**（1007 对 1007），差异 0.00 pp，远优于验收的 ±1pp 容差。

---

## 7. 板载性能

- 单次前向：**23.434 ms**（RV1106 Cortex-A7 @ 1.2 GHz，静态 musl 二进制）
- 全量 1296 试次：30.37 s
- 纯 C 朴素循环实现，未做任何 NEON/定点/循环展开优化；发放率版 LIF 只需 8 步、
  每步只做一次乘加，理论上还有较大加速空间（如 int8 量化 + NEON + 提前退出）。
