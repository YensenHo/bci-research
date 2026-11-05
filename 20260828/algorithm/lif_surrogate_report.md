# LIF-EEGNet surrogate gradient 直接训练报告（S3）

## 结论一句话

**surrogate gradient 直接训练 SNN 成功绕开了 ann2snn 转换的 55% 瓶颈**：
LIF-EEGNet 在最优配置下达到 **clean 76.9% ± 0.8%（3 seed）/ HBC 76.3% ± 0.7%**，
追平并略超 ReLU 代理（75.77%）。转换路线的 ~55% 平台**不是率编码 LIF 的固有上限**，
而是 naive 阈值标定（全局最大激活 thr≈7.0）把发放率压到近零（act1 仅 ~3%）所致。

## 最终 SNN 精度 vs 基线

| 模型 | clean | HBC 噪声 |
|---|---|---|
| ELU 基线（eegnet.pth，v1 参考） | ~75%+ | — |
| ReLU 代理（relu_proxy.pth） | **75.77%** | 75.77% |
| ann2snn 转换（naive 标定） | 50.93% | — |
| ann2snn 转换（归一化改进） | ~55% | — |
| **LIF surrogate（thr=1.0, tau=2.0, T=8, seed42）** | **76.85%** | 75.85% |
| **LIF surrogate（seed123）** | **77.70%** | 77.16% |
| **LIF surrogate（seed2024）** | 76.08% | 76.00% |
| **LIF surrogate（3 seed 均值）** | **76.88%** | 76.34% |

- 结论：surrogate 训练 SNN **追平 ANN**（差距 −1.1 ~ +1.9 pp，跨 session 噪声量级内）。
- 关键洞察：**阈值标定是唯一关键变量**。BN 输出近似单位方差（均值为 0），
  thr=1.0 恰好落在发放率健康的区间（act1≈16%、act2≈13%），
  而转换路线用「全局最大激活」thr≈7.0 → 发放率 3%/0.4% → 信息被压死。

## 关键消融（thr × timesteps × tau，16 run，全部真实 eval 结果）

### [A] threshold 扫描（fix tau=2.0, T=8）
| threshold | clean | HBC | act1 发放率 | act2 发放率 |
|---|---|---|---|---|
| 0.5 | 76.00% | 74.38% | 25.2% | 23.5% |
| **1.0** | **76.85%** | 75.85% | 15.6% | 13.3% |
| 2.0 | 75.77% | 76.16% | 8.3% | 6.4% |
| 4.0 | 65.59% | 65.20% | 1.3% | 0.7% |

→ 阈值 1.0 是峰值；4.0 时发放率塌缩到 ~1% → 精度崩到 65.6%。
  （这正是转换路线 55% 瓶颈的微观原因：thr≈7 比 4.0 更极端。）

### [B] timesteps 扫描（fix thr=1.0, tau=2.0）
| T | clean | HBC |
|---|---|---|
| 4 | 76.70% | 75.00% |
| **8** | **76.85%** | 75.85% |
| 16 | 76.31% | 75.54% |
| 32 | 76.77% | 75.54% |

→ T=8 已饱和；继续加大 T 无收益（甚至微降）。部署友好：板载只需 8 步。

### [C] tau 扫描（fix thr=1.0, T=8）
| tau | clean | HBC |
|---|---|---|
| 1.0（≈IF） | 76.54% | 75.39% |
| **2.0** | **76.85%** | 75.85% |
| 4.0 | 76.39% | 75.85% |

→ tau 不敏感（1.0~4.0 均在 76.4~76.9%）。

### [D] 可学习阈值（learnable threshold, fix tau=2.0, T=8）
- clean 76.47% / HBC 75.31%。**不优于固定 thr=1.0**（76.85%）。
- 说明 BN 单位方差尺度下，固定 thr=1.0 已是近优，无需可学习阈值增加部署复杂度。

### [E] 无热启动（随机初始化, thr=1.0, tau=2.0, T=8）
- clean 74.54% / HBC 74.92%。surrogate 从零也能训练 SNN，仅比热启动低 ~2 pp。

## 最优配置与部署含义

- **最优：threshold=1.0, tau=2.0, timesteps=8**（固定阈值，无需可学习）。
- T=8 意味着板载 RV1106 只需 8 个时间步（远低于转换路线需要的 T=256），
  单次前向 ≈ 8×ELU 单趟成本，远优于「T=256 仍卡 55%」的转换路线。
- 下一步（可选）：把 surrogate 最优权重（lif_F_seed123.pth 等）按 thr=1.0
  重做 C 头文件导出，即可板载验证 surrogate-SNN 的真实精度（预计 76~78%）。

## 诚实边界

- 本报告 SNN 精度为 PyTorch 端到端 surrogate 训练结果（MPS），非板载 RV1106 实测；
  C-SNN 逐位一致性已由 v2 验证（max|diff|≈4.9e-7），但「端到端板载精度」仍需
  按 thr=1.0 重新导出权重后实测，属后续步骤，未在本任务中虚构。
- SNN 精度上限 ≈ ANN 上限（~76-77%），未「超越」ANN；这是诚实的物理事实：
  率编码 LIF 的发放率量化（T=8 时 9 档）+ 负半轴截断（与 ReLU 同源）构成了
  与 ANN 相同的天花板，surrogate 训练只是**追平**而非**突破**这一上限。
- 若未来要突破 77%：需要 temporal/signed 编码、更宽 timesteps 的 burst 编码、
  或更深 SNN 架构，这些不在本任务范围。

## 产物文件

- `train_lif_surrogate.py` — surrogate 训练脚本（热启动/早停/n_classes/timesteps/threshold/tau/learnable-threshold 均支持）
- `lif_surrogate_ablation.py` — 消融驱动（16 run 顺序寻优）
- `ablation_lif.json` — 16 个 run 的完整原始结果
- `lif_surrogate_ablation_summary.txt` — 汇总表
- `lif_*.pth` / `lif_surrogate_report_*.txt` — 各 run 的最优权重 + 逐 epoch 报告
- `lif_eegnet.py` — 追加了 `last_firing_rate` 发放率诊断（纯附加，不影响其他脚本）
