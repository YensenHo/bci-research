#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_snn_sparsity.py — SNN 稀疏化评估：发放率 vs 精度 vs 推理潜力

对一组 LIF-EEG-TCNet 权重（基线 HBC 增强 + 各档稀疏正则）做三件事：
  a. 稀疏度   ：eval 集上逐 LIF 位点平均发放率（clean 与 8dB 噪声下都测）
  b. 精度     ：clean 四分类精度 + 8dB（5bit+AWGN，σ/H=10^(-8/20)）精度
                （+ 附加 8dB+突发 域，与 HBC 鲁棒性报告协议一致，固定噪声种子）
  c. 推理潜力 ：事件驱动 MACs 估算——理想事件驱动硬件上，每个卷积的代价
                正比于**输入脉冲密度**：ACs(layer) ≈ dense_MACs(layer) × f_in，
                f_in 为该层输入张量的实际发放率（实测）。φ1（conv1/depthwise）
                输入是连续 EEG，不可稀疏化（f=1.0）；LIF 之后的层按实测密度折算。
                汇总得「事件驱动计算量 / 密集计算量」的减少比例，即稀疏推理潜力。

密度→计算量映射（逐层）
-----------------------
  conv1 / depthwise      : 输入连续 EEG            → f = 1.0（密集，不可省）
  sep_depthwise/pointwise: 输入 = act1 输出        → f = f_act1
  block1.conv_a / skip   : 输入 = act2 输出(pool2 后, 脉冲计数不变密度) → f = f_act2
  block1.conv_b          : 输入 = act_a1 输出       → f = f_act_a1
  block2.conv_a / skip   : 输入 = act_res1 输出     → f = f_act_res1
  block2.conv_b          : 输入 = act_a2 输出       → f = f_act_a2
  fc                     : 输入 = GAP(act_res2)     → f = f_act_res2（量级可忽略）
池化/加法在事件驱动实现中为脉冲计数/位运算，成本远小于卷积，估为零。

注意：T=8 时间步下，密集版总 MACs = T × dense_MACs，事件驱动总 ACs = T × ΣACs，
比例与 T 无关，因此报告「单次前向」比例即可。这是**理想事件驱动硬件**（或稀疏
内核 CPU 实现）的理论潜力；RV1106 的 NPU 是密集 MAC 引擎，不能直接吃这个红利，
该数字的意义是：稀疏化后模型已具备事件驱动部署的形态（详见报告结论）。

用法
----
  env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin \\
      /opt/miniconda3/envs/openmmlab/bin/python eval_snn_sparsity.py \\
      --ckpts eeg_tcnet_4class_lif_hbc_aug_s42.pth,eeg_tcnet_4class_lif_hbc_sparse_l0.3_t0.15.pth,... \\
      --names 基线aug_s42,稀疏λ0.3_t0.15,...

输出：snn_sparsity_results/sparsity_eval.json（全部原始数字）+ 控制台对比表
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ALGO_DIR = Path(__file__).resolve().parent
PROJECT = Path("/Users/heyixuan/BCI/luckfox-deploy")
DATA_DIR = PROJECT / "data"
DATA_4CLASS = ALGO_DIR / "data_4class"

sys.path.insert(0, str(ALGO_DIR))
sys.path.insert(0, str(DATA_DIR))

from dataset import EEGDataset          # noqa: E402
from eeg_tcnet_lif import EEGTCNetLIF   # noqa: E402
from hbc_noise_advanced import hbc_noise, snr_db_to_sigma_ratio  # noqa: E402

TEST_NOISE_SEED = 20240828  # 测试期噪声固定种子：所有模型看到同一份噪声实现


def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _build_model(ckpt: dict) -> EEGTCNetLIF:
    """按 ckpt 元信息重建模型（model_args + lif_hparams）。"""
    return EEGTCNetLIF(**ckpt["model_args"], **ckpt.get("lif_hparams", {}))


def _eval_noise(model, loader, device, noise_spec: dict | None) -> tuple[float, dict]:
    """在 eval 集上评估，返回 (acc, 逐层发放率 dict)。

    noise_spec=None 表示干净；否则施加 hbc_noise。
    每次调用前重设 torch.manual_seed(TEST_NOISE_SEED)，保证不同模型
    看到同一份噪声实现（公平对比，与 eval_hbc_snr_sweep.py 一致）。
    """
    model.eval()
    torch.manual_seed(TEST_NOISE_SEED)
    correct, total = 0, 0
    fr_sum: dict = {}
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            if noise_spec is not None:
                xb = hbc_noise(xb, **noise_spec)
            logits = model(xb)
            correct += (logits.argmax(1) == yb).sum().item()
            total += xb.size(0)
            for name, fr in model.firing_rates.items():
                fr_sum[name] = fr_sum.get(name, 0.0) + fr * xb.size(0)
    fr = {k: v / total for k, v in fr_sum.items()} if total else {}
    return correct / total, fr


# ---------------------------------------------------------------------------
# 密集 MACs 精确计算（brief 配置：F1=8,D=2,F2=16,FT=12,KE=63,sep=16,KT=12,
# pool1=4,pool2=8,T=500,C=22,dilations=(1,2)；公式同 eeg_tcnet.macs_estimate）
# ---------------------------------------------------------------------------
def _dense_macs(model: EEGTCNetLIF) -> dict[str, float]:
    """逐层密集 MACs（单次前向，与 T=8 时间步无关的每步代价）。"""
    T = model.input_length          # 500
    C = model.n_channels            # 22
    F1, D, F2, FT = model.F1, model.D, model.F2, model.FT
    KE = model.kernel_length        # 63
    sep = model.sep_kernel          # 16
    KT = model.tcn_kernel_size      # 12
    p1, p2 = model.pool1_size, model.pool2_size   # 4, 8
    T1 = T // p1                    # 125
    T2 = T1 // p2                   # 15

    macs = {}
    # φ1：conv1 (1→F1, (1,KE)) 于 (1,C,T) → (F1,C,T)；depthwise (F1→D·F1, (C,1), groups=F1) → (D·F1,1,T)
    macs["conv1"] = 1.0 * KE * 1 * F1 * C * T
    macs["depthwise"] = C * 1 * 1 * (D * F1) * 1 * T
    # φ3：sep_depthwise ((D·F1)→(D·F1), (1,sep), groups) → (D·F1,1,T1)；pointwise → (F2,1,T1)
    macs["sep_depthwise"] = 1 * sep * 1 * (D * F1) * 1 * T1
    macs["pointwise"] = 1 * 1 * (D * F1) * F2 * 1 * T1
    # φ4：TCN 两残差块（conv_a: in→FT, KT；conv_b: FT→FT, KT；skip: in→FT, k=1）
    for i, blk in enumerate(model.tcn):
        in_ch = F2 if i == 0 else FT
        macs[f"tcn{i}.conv_a"] = KT * in_ch * FT * T2
        macs[f"tcn{i}.conv_b"] = KT * FT * FT * T2
        macs[f"tcn{i}.skip"] = 1 * in_ch * FT * T2
    # φ5：fc (FT→4)
    macs["fc"] = FT * model.n_classes
    return macs


def _input_density_map(model: EEGTCNetLIF, fr: dict) -> dict[str, float]:
    """每个 conv 层的**输入张量脉冲密度**（事件驱动折算系数）。

    映射逻辑见文件头 docstring；key 与 _dense_macs 对齐。
    fr 的 key 为完整模块名（act1 / act2 / tcn.0.act_a / tcn.1.act_b / ...）。
    """
    def rate_of(suffix: str, blk: int | None = None) -> float:
        if blk is None:
            return fr[f"act{suffix}"]                    # 前端 act1 / act2
        return fr[f"tcn.{blk}.act_{suffix}"]             # TCN 内 act_a / act_b / act_res

    f_act1 = rate_of("1")          # 前端 act1
    f_act2 = rate_of("2")          # 前端 act2
    f_aa0 = rate_of("a", 0); f_ab0 = rate_of("b", 0); f_res0 = rate_of("res", 0)
    f_aa1 = rate_of("a", 1); f_ab1 = rate_of("b", 1); f_res1 = rate_of("res", 1)

    return {
        "conv1": 1.0,                       # 连续 EEG，不可稀疏化
        "depthwise": 1.0,                   # 连续特征
        "sep_depthwise": f_act1,            # act1 输出（pool1 为脉冲计数）
        "pointwise": f_act1,
        "tcn0.conv_a": f_act2,              # act2 输出（pool2 后密度不变）
        "tcn0.skip": f_act2,
        "tcn0.conv_b": f_aa0,               # act_a 块0 输出
        "tcn1.conv_a": f_res0,              # act_res 块0 输出（残差和）
        "tcn1.skip": f_res0,
        "tcn1.conv_b": f_aa1,               # act_a 块1 输出
        "fc": f_res1,                       # GAP(act_res 块1)，量级可忽略
    }


def _spike_counts(model: EEGTCNetLIF, fr: dict) -> dict[str, float]:
    """各 LIF 位点每试次脉冲数 = 发放率 × 神经元数 × T 步（MCU 脉冲流量估算）。

    神经元数按结构精确推导（brief 配置）：
      act1      : (F1·D, 1, T=500)      = 16×500
      act2      : (F2, 1, T1=125)       = 16×125
      tcn.?.act_*: (FT=12, T2=15)       = 12×15
    """
    T = model.lif_kwargs.get("timesteps", 8)
    T1 = model.input_length // model.pool1_size
    T2 = T1 // model.pool2_size
    numel = {"act1": (model.F1 * model.D) * model.input_length,
             "act2": model.F2 * T1,
             "tcn.0.act_a": model.FT * T2, "tcn.0.act_b": model.FT * T2,
             "tcn.0.act_res": model.FT * T2,
             "tcn.1.act_a": model.FT * T2, "tcn.1.act_b": model.FT * T2,
             "tcn.1.act_res": model.FT * T2}
    out = {}
    for name, rate in fr.items():
        out[name] = rate * numel.get(name, 0) * T
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="SNN 稀疏化评估：发放率 vs 精度 vs 推理潜力")
    ap.add_argument("--ckpts", type=str, required=True,
                    help="逗号分隔的权重路径（相对 algorithm/ 或绝对路径）")
    ap.add_argument("--names", type=str, default="",
                    help="逗号分隔的展示名（缺省用文件名）")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--out-dir", type=str, default=str(ALGO_DIR / "snn_sparsity_results"))
    args = ap.parse_args()

    ckpt_paths = [Path(p.strip()) for p in args.ckpts.split(",") if p.strip()]
    ckpt_paths = [p if p.is_absolute() else ALGO_DIR / p for p in ckpt_paths]
    names = [s.strip() for s in args.names.split(",") if s.strip()] if args.names else []
    names = names + [p.stem for p in ckpt_paths[len(names):]]
    assert len(names) == len(ckpt_paths), "names 与 ckpts 数量不匹配"

    device = _pick_device()
    eval_ds = EEGDataset(DATA_4CLASS / "eval.npz")
    loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False)

    # 测试域：clean / 8dB / 8dB+突发（协议同 eval_hbc_snr_sweep.py）
    domains = [("clean", None),
               ("8dB", dict(qbits=5, qrange=4.0, sigma_ratio=snr_db_to_sigma_ratio(8.0))),
               ("8dB+burst", dict(qbits=5, qrange=4.0,
                                  sigma_ratio=snr_db_to_sigma_ratio(8.0),
                                  burst_sigma_ratio=0.3, burst_len=16, burst_prob=0.005))]
    print(f"设备       : {device}")
    print(f"评估集     : eval.npz（{len(eval_ds)} 试次，cross-session，E 会话）")
    print(f"测试噪声   : 5bit 量化 + AWGN，σ/H=10^(-SNR/20)；固定种子 {TEST_NOISE_SEED}\n")

    per_model = {}
    for name, path in zip(names, ckpt_paths):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        model = _build_model(ckpt)
        missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=True)
        assert not missing and not unexpected, f"权重加载失败: {missing} / {unexpected}"
        model = model.to(device)
        model.eval()

        row = {"ckpt": str(path), "final_test_clean_ckpt": round(ckpt.get("final_test", -1.0), 4)}
        for label, spec in domains:
            acc, fr = _eval_noise(model, loader, device, spec)
            row[f"{label}_acc"] = round(acc, 4)
            row[f"{label}_fr"] = {k: round(v, 4) for k, v in fr.items()}
            row[f"{label}_avg_fr"] = round(float(np.mean(list(fr.values()))), 4) if fr else -1.0

        # ---- 推理潜力：事件驱动 ACs 估算（用 clean 域发放率，标准参考）----
        fr_clean = row["clean_fr"]
        dense = _dense_macs(model)
        dens_map = _input_density_map(model, fr_clean)
        ev = {k: dense[k] * dens_map[k] for k in dense}
        total_dense = float(sum(dense.values()))
        total_ev = float(sum(ev.values()))
        row["dense_macs"] = round(total_dense, 1)
        row["event_driven_acs"] = round(total_ev, 1)
        row["macs_reduction_ratio"] = round(1.0 - total_ev / total_dense, 4)
        row["per_layer"] = {"dense_macs": {k: round(v, 1) for k, v in dense.items()},
                            "input_density": {k: round(v, 4) for k, v in dens_map.items()},
                            "event_acs": {k: round(v, 1) for k, v in ev.items()}}
        # 每试次脉冲流量（MCU 视角补充指标）
        spk = _spike_counts(model, fr_clean)
        row["spikes_per_trial"] = round(float(sum(spk.values())), 1)
        row["spikes_per_trial_per_layer"] = {k: round(v, 1) for k, v in spk.items()}
        per_model[name] = row
        print(f"[{name}] clean={row['clean_acc']*100:.2f}% 8dB={row['8dB_acc']*100:.2f}% "
              f"avg_fr={row['clean_avg_fr']*100:.1f}% "
              f"事件驱动ACs/密集MACs={total_ev/total_dense*100:.1f}% "
              f"（省 {row['macs_reduction_ratio']*100:.1f}%）")

    # ---- 对比表（控制台）----
    print("\n" + "=" * 92)
    print("SNN 稀疏化对比表（final test @ eval，cross-session，2592 试次）")
    print("=" * 92)
    hdr = f"{'模型':<18}{'clean':>9}{'8dB':>9}{'8dB+burst':>11}{'avg_fr':>9}{'事件驱动/密集':>14}"
    print(hdr)
    for n in names:
        r = per_model[n]
        print(f"{n:<18}{r['clean_acc']*100:>8.2f}%{r['8dB_acc']*100:>8.2f}%"
              f"{r['8dB+burst_acc']*100:>10.2f}%{r['clean_avg_fr']*100:>8.1f}%"
              f"{r['event_driven_acs']/r['dense_macs']*100:>13.1f}%")
    print("-" * 92)
    print("逐层发放率（clean / 8dB）：")
    for n in names:
        r = per_model[n]
        fc = r["clean_fr"]; f8 = r["8dB_fr"]
        parts = "  ".join(f"{k.split('.')[-1]}={fc[k]*100:.1f}/{f8[k]*100:.1f}%" for k in sorted(fc))
        print(f"  {n:<16}: {parts}")

    # ---- 落盘 JSON ----
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {
        "meta": {
            "model": "LIF-EEG-TCNet (brief, thr=1.0/tau=2.0/T=8, 9512 参数)",
            "data": "data_4class/eval.npz (22ch 四分类, 2592 试次, cross-session)",
            "test_noise_seed": TEST_NOISE_SEED,
            "noise_spec": "5bit 量化(qrange=4.0) + AWGN σ/H=10^(-SNR/20)；8dB+突发 附加域",
            "event_driven_assumption": "ACs(layer) ≈ dense_MACs(layer) × 实测输入脉冲密度；"
                                       "φ1(conv1/depthwise) 输入连续 EEG 不可稀疏化；"
                                       "池化/加法按脉冲计数估零；理想事件驱动硬件或稀疏内核实现的理论潜力",
            "device": str(device),
        },
        "domains": [l for l, _ in domains],
        "per_model": per_model,
    }
    (out_dir / "sparsity_eval.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n✓ 结果已落盘：{out_dir / 'sparsity_eval.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
