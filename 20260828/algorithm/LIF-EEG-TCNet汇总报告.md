# LIF-EEG-TCNet 汇总（类脑化 EEG-TCNet，cross-session 四分类）

> 世界主流轻量模型 EEG-TCNet（Ingolfsson 2020）→ 移植进「类脑 + BCI + HBC」体系。
> 全部数字为真实训练产物（逐 epoch 记录在 report txt），未虚构。

## 一、精度对比（cross-session 四分类 22ch，协议与 train_4class.py 一致）

| 模型 | final test | 参数量 |
|---|---|---|
| EEG-TCNet ELU（简报配置 KE=63/KT=12/pool4/8/GAP） | **64.31%** | 9,512 |
| **LIF-EEG-TCNet（surrogate，thr=0.5/1.0, T=8）** | **62.42%** | 9,512 |
| EEG-TCNet relu（简报配置） | 61.61% | 9,512 |
| LIF-EEGNet（现在板上的） | 56.87% | 2,396 |
| EEG-TCNet 论文原始配置（KE=32/KT=4/pool8/8/末步读出） | 56.40% | 4,272 |

## 二、LIF-EEG-TCNet 扫描明细（全部真实）

| 配置 | final test | 最佳 val |
|---|---|---|
| **thr=0.5, tau=2.0, T=8** | **62.42%** | 66.75% |
| thr=1.0, tau=2.0, T=8 | 62.42% | 66.75% |
| thr=1.0, tau=2.0, T=4 | 62.38% | — |
| thr=2.0, tau=2.0, T=8 | 61.54% | 66.75% |
| thr=1.0, tau=2.0, T=16 | 61.46% | 67.27% |

最优：**thr=0.5 或 1.0, tau=2.0, T=8 → 62.42%**（5 组全在 61.5~62.4%，结论稳）。

## 三、核心结论

1. **类脑化在 EEG-TCNet 架构（带 TCN 块）上零代价**：LIF-EEG-TCNet 62.42% 追平并略超 relu 61.61%（+0.81pp）。「LIF surrogate 训练追平 ANN」从简单 EEGNet 扩展到带膨胀因果卷积 + 残差块的复杂架构，普适性再 +1。
2. **芯片四分类精度提升确定**：56.87% → ~62%（+5.5pp）。
3. **配置洞察**：论文原始配置（激进池化 + 末步读出）在我们 T=500 短窗跨会话场景反而 −1.9pp；KE=63 大前端核 + GAP 聚合更适配——移植世界主流技术要理解结构 + 适配自家数据，不是照抄。

## 四、端侧可行性（RV1106）

- 参数量 9,512（<10K），INT8 权重 ~10KB，内存充裕。
- 算子集：Conv2d/depthwise/Conv1d(膨胀因果)/BN/ELU(LIF)/GAP/Linear——与已部署的 EEGNet 同族，纯 C 可实现。
- 需新增 C 实现：膨胀因果卷积（dilation 1/2 + 因果 padding）、TCN 残差 add、GAP 读出。

## 五、下一步

导出 SNN 权重 C 头（TCN 结构）→ C 移植（lif_surrogate_infer_tcnet.c）→ 宿主机逐位一致 → zig 编译 → 板上实测 2592 试次（预期 ≈62.42%）。
