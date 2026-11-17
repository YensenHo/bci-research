# 情绪脑电：个体校准与可靠性研究

本目录记录截至 2026-11-17 的公开数据审计、候选研究问题和探索性对照；它们都不是正式论文结论。科研方向属于导师指定的 BCI AI 模型和可穿戴部署主轴，情绪自评是用于测试底层个体适配与信号可靠性的研究任务，不构成医疗产品或诊断用途。

## 请先读

1. 项目根目录的 `README.md`、`HARNESS.md` 和 `AGENTS.md`。
2. [研究问题与边界](outputs/第一篇论文_研究问题重新收束_20261117.md)。
3. [FACED 跨电影开发结果](outputs/FACED_跨电影混合采样率开发结果_20261117.md)和[训练侧双重留出结果](outputs/FACED_双重留出_训练侧交叉验证结果_20261117.md)。两项都属于开发，当前 EEG 模型没有稳定超过常数均值；不要称为论文成功或产品有效。
4. [竞争论文](outputs/EEG_个人校准竞争研究补充_20261117.md)。PRED (IJCAI 2026)、SPDIM (ICLR 2025) 等已涉及 EEG 无标签适配；需逐项比较假设、数据、标签、时序及成本，不能把自动选择适配当首创。
5. [DENS 数据审计](outputs/DENS_第二数据集可用性审计_20261117.md)与[其他数据集限制](outputs/第二数据集_EMAP_DENS_可用性与约束_20261117.md)。DENS 的事件时间、.fdt 长度和许可存在重要问题，不应直接当完整第二数据集。EMAP 禁止商业系统训练/改进，应遵守其许可。
6. [Hermes 协作交接](HERMES_HANDOFF.md)。

## 当前结果与可复现边界

FACED 的 16 人训练/6 人验证、112/42 个试次的开发实验中，训练均值、无校准 EEG、短基线校准 EEG 的 MAE 分别为 1.029、1.160、1.316（量表 0–7）。类别均值 MAE 0.992，但它额外使用刺激类别，不是同信息预算的模型排名。信号质量与事件时间仍需核实；个人情绪标签是观看后自报，不能代指动态脑电真值。

训练侧 16 人×7 部影片、逐人逐片双重留出下，均值、无校准及校准模型 MAE 分别为 1.422、2.308、1.694。这是 112 个重复交叉验证预测，不是 112 个独立人，也不是独立测试。每个影片对应一个类别，留出片也留出了该类。

脚本固定 FACED NEMAR v1.1.4 commit `b19270dcf75170fc7588ddba3b1e9129a01e9e43`、DENS NEMAR v1.0.0 commit `beb73da6fde5c043d42145637db9f5bce7e3fe12`。`.gitignore` 排除 `work/` 下的生成数据目录。本仓库不包含 EEG、个人评分、特征缓存或逐试次预测；使用者需要根据数据集条款独立获取。

## 复现顺序（从本目录运行）

需要 Python、NumPy、SciPy、pandas、scikit-learn、requests；需访问公开数据源，采集大量片段前先确认本地空间、网络、数据使用授权及预处理语义。

```text
python work/faced_audit_20261117.py
python work/faced_ratings_crosscheck_20261117.py
python work/faced_development_20261117.py
python work/faced_development_v2_20261117.py
python work/faced_crossed_cv_20261117.py
python work/baseline_noise_simulation_20261117.py
```

前两个 FACED 脚本请求公开的个人评分元数据，后续脚本请求 EEG 片段，会在本地生成不应提交的文件。FACED CV 的原始结果包含逐试次评分和预测，不在仓库发布。`dens_metadata_audit_20261117.py`、`dens_trial_manifest_20261117.py`、`dens_signal_probe_20261117.py`、`dens_baseline_probe_20261117.py`、`dens_baseline_batch_20261117.py` 仅供审计，DENS 文件长度和许可问题解决前不能用于产品训练或声称第二数据集验证。

合成高斯脚本仅验证假设模型中基线噪声可能使完全相减变差；它假设已知噪声方差，不是脑电实验或原创理论。
