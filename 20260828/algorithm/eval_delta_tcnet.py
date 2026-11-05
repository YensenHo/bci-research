#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_delta_tcnet.py — 全脉冲 DeltaTCNet 评估：精度 vs 输入稀疏度 vs 前端事件驱动潜力

对一组 delta 输入脉冲编码权重（+ 基线 LIF-EEG-TCNet 对照）做三件事：
  a. 精度     ：eval 集上 clean / 8dB / 8dB+burst 三域准确率
                （噪声协议同 eval_hbc_snr_sweep.py：5bit 量化 + AWGN σ/H=10^(-SNR/20)，
                固定噪声种子 20240828 + batch 16，所有模型看到同一份噪声实现）
  b. 稀疏度   ：输入脉冲密度（<30% 目标）+ 逐 LIF 位点平均发放率（clean 与 8dB）
  c. 前端 φ1 事件驱动潜力：事件驱动 ACs ≈ Σ 密集MACs(layer) × 实测输入脉冲密度。
     ★ 与既有结论的关键差异：此前 conv1/depthwise 输入是**连续 EEG**（密度=1.0，
       不可稀疏化，占 96.9% MACs）；delta 编码后 conv1 的输入是**稀疏脉冲**，
       密度=f_delta → φ1 的 conv1（占全部 MACs 94%）首次可稀疏化。
       诚实口径：depthwise 的输入是 conv1+BN 的**连续输出**，仍按密度 1.0 计；
       conv1 之后各层按 LIF 实测发放率折算（同 eval_snn_sparsity.py）。

密度→计算量映射（逐层，key 同 eval_snn_sparsity.py，conv1 语义更新）：
  conv1                : 输入 = delta 脉冲（稀疏）        → f = 输入脉冲密度（实测）
  depthwise            : 输入 = BN(conv1) 连续输出         → f = 1.0（诚实：未稀疏化）
  sep_depthwise/pointwise : 输入 = act1 输出              → f = f_act1
  tcn0.conv_a / skip   : 输入 = act2 输出                 → f = f_act2
  tcn0.conv_b          : 输入 = act_a0 输出               → f = f_act_a0
  tcn1.conv_a / skip   : 输入 = act_res0 输出             → f = f_act_res0
  tcn1.conv_b          : 输入 = act_a1 输出               → f = f_act_a1
  fc                   : 输入 = GAP(act_res1)             → f = f_act_res1

用法
----
  env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin \\
      /opt/miniconda3/envs/openmmlab/bin/python eval_delta_tcnet.py \\
      --ckpts eeg_tcnet_4class_delta_delta1bit_k0.5.pth,... \\
      --names delta1bit_k0.5,... [--include-baseline]

输出：delta_sparsity_results/delta_eval.json（全部原始数字）+ 控制台对比表
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

from dataset import EEGDataset            # noqa: E402
from delta_tcnet import DeltaTCNet        # noqa: E402
from hbc_noise_advanced import hbc_noise, snr_db_to_sigma_ratio  # noqa: E402

TEST_NOISE_SEED = 20240828  # 测试期噪声固定种子（同 eval_hbc_snr_sweep.py / eval_snn_sparsity.py）

BASELINE_CKPT = ALGO_DIR / "eeg_tcnet_4class_lif_hbc_aug_s42.pth"


def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _build_model(ckpt: dict):
    """按 ckpt 元信息重建模型（delta 配置或基线 LIF 配置）。"""
    if "delta_config" in ckpt:
        dc = ckpt["delta_config"]
        return DeltaTCNet(**ckpt["model_args"], **ckpt.get("lif_hparams", {}),
                          input_channels=dc["input_channels"],
                          delta_k=dc["delta_k"], delta_bits=dc["delta_bits"],
                          delta_mode=dc["delta_mode"], theta=ckpt.get("theta"))
    from eeg_tcnet_lif import EEGTCNetLIF  # 基线（无 delta 编码）
    return EEGTCNetLIF(**ckpt["model_args"], **ckpt.get("lif_hparams", {}))


def _eval_noise(model, loader, device, noise_spec: dict | None) -> tuple[float, dict, float]:
    """在 eval 集上评估，返回 (acc, 逐层发放率 dict, 输入脉冲密度)。

    noise_spec=None 表示干净；否则施加 hbc_noise。每次调用前重设
    torch.manual_seed(TEST_NOISE_SEED)，保证不同模型看到同一份噪声实现。
    """
    model.eval()
    torch.manual_seed(TEST_NOISE_SEED)
    correct, total = 0, 0
    fr_sum: dict = {}
    dens_sum = 0.0
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
            dens_sum += getattr(model, "input_density", 1.0) * xb.size(0)
    fr = {k: v / total for k, v in fr_sum.items()} if total else {}
    dens = dens_sum / total if total else 1.0
    return correct / total, fr, dens


# ---------------------------------------------------------------------------
# 密集 MACs 精确计算（brief 配置；delta 版 conv1 输入通道 = input_channels）
# ---------------------------------------------------------------------------
def _dense_macs(model) -> dict[str, float]:
    T = model.input_length
    C = model.n_channels
    F1, D, F2, FT = model.F1, model.D, model.F2, model.FT
    KE = model.kernel_length
    sep = model.sep_kernel
    KT = model.tcn_kernel_size
    p1, p2 = model.pool1_size, model.pool2_size
    T1 = T // p1
    T2 = T1 // p2
    cin = getattr(model, "input_channels", 1)   # delta 版 conv1 输入通道（posneg=2）

    macs = {}
    macs["conv1"] = 1.0 * KE * cin * F1 * C * T
    macs["depthwise"] = C * 1 * 1 * (D * F1) * 1 * T
    macs["sep_depthwise"] = 1 * sep * 1 * (D * F1) * 1 * T1
    macs["pointwise"] = 1 * 1 * (D * F1) * F2 * 1 * T1
    for i, blk in enumerate(model.tcn):
        in_ch = F2 if i == 0 else FT
        macs[f"tcn{i}.conv_a"] = KT * in_ch * FT * T2
        macs[f"tcn{i}.conv_b"] = KT * FT * FT * T2
        macs[f"tcn{i}.skip"] = 1 * in_ch * FT * T2
    macs["fc"] = FT * model.n_classes
    return macs


def _input_density_map(model, fr: dict, f_delta: float) -> dict[str, float]:
    """每个 conv 层的**输入张量脉冲密度**（事件驱动折算系数）。"""
    def rate_of(suffix: str, blk: int | None = None) -> float:
        if blk is None:
            return fr[f"act{suffix}"]
        return fr[f"tcn.{blk}.act_{suffix}"]

    f_act1 = rate_of("1"); f_act2 = rate_of("2")
    f_aa0 = rate_of("a", 0); f_ab0 = rate_of("b", 0); f_res0 = rate_of("res", 0)
    f_aa1 = rate_of("a", 1); f_ab1 = rate_of("b", 1); f_res1 = rate_of("res", 1)

    return {
        "conv1": f_delta,               # ★ delta 脉冲输入：首次可稀疏化（此前恒 1.0）
        "depthwise": 1.0,               # 输入 = BN(conv1) 连续输出，诚实按密集计
        "sep_depthwise": f_act1,
        "pointwise": f_act1,
        "tcn0.conv_a": f_act2,
        "tcn0.skip": f_act2,
        "tcn0.conv_b": f_aa0,
        "tcn1.conv_a": f_res0,
        "tcn1.skip": f_res0,
        "tcn1.conv_b": f_aa1,
        "fc": f_res1,
    }


def _input_spikes_per_trial(model, f_delta: float) -> float:
    """每试次输入脉冲事件数（= 密度 × 通道 × 时间；HBC 传输载荷视角）。

    signed 单通道：事件数 = f_delta·C·T；posneg 双通道：每个事件在正或负通道
    各 1 bit（合并口径密度不变，事件数 = f_delta·C·T，只是表示成 2 通道）。
    """
    C, T = model.n_channels, model.input_length
    return f_delta * C * T


def main() -> int:
    ap = argparse.ArgumentParser(description="全脉冲 DeltaTCNet 评估：精度 vs 稀疏度 vs 事件驱动潜力")
    ap.add_argument("--ckpts", type=str, required=True, help="逗号分隔的权重路径")
    ap.add_argument("--names", type=str, default="", help="逗号分隔的展示名")
    ap.add_argument("--include-baseline", action="store_true",
                    help="附加基线 LIF-EEG-TCNet（aug_s42，63.58% clean）对照行")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--out-dir", type=str, default=str(ALGO_DIR / "delta_sparsity_results"))
    args = ap.parse_args()

    ckpt_paths = [Path(p.strip()) for p in args.ckpts.split(",") if p.strip()]
    ckpt_paths = [p if p.is_absolute() else ALGO_DIR / p for p in ckpt_paths]
    names = [s.strip() for s in args.names.split(",") if s.strip()] if args.names else []
    names = names + [p.stem.replace("eeg_tcnet_4class_delta_", "") for p in ckpt_paths[len(names):]]
    if args.include_baseline and BASELINE_CKPT.exists():
        ckpt_paths.append(BASELINE_CKPT)
        names.append("基线LIF-aug_s42(连续输入)")
    assert len(names) == len(ckpt_paths), "names 与 ckpts 数量不匹配"

    device = _pick_device()
    eval_ds = EEGDataset(DATA_4CLASS / "eval.npz")
    loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False)

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

        is_delta = "delta_config" in ckpt
        dc = ckpt.get("delta_config", {})
        row = {"ckpt": str(path), "final_test_clean_ckpt": round(ckpt.get("final_test", -1.0), 4),
               "is_delta": is_delta,
               "delta_desc": (f"{dc.get('delta_bits',1)}bit θ=k·std k={dc.get('delta_k',0.5):g} "
                              f"mode={dc.get('delta_mode','signed')}" if is_delta else "连续输入（基线）")}
        for label, spec in domains:
            acc, fr, dens = _eval_noise(model, loader, device, spec)
            row[f"{label}_acc"] = round(acc, 4)
            row[f"{label}_fr"] = {k: round(v, 4) for k, v in fr.items()}
            row[f"{label}_avg_fr"] = round(float(np.mean(list(fr.values()))), 4) if fr else -1.0
            row[f"{label}_input_density"] = round(dens, 4)

        # ---- 事件驱动 ACs（用 clean 域密度与发放率，标准参考）----
        f_delta = row["clean_input_density"] if is_delta else 1.0
        fr_clean = row["clean_fr"]
        dense = _dense_macs(model)
        dens_map = _input_density_map(model, fr_clean, f_delta)
        ev = {k: dense[k] * dens_map[k] for k in dense}
        total_dense = float(sum(dense.values()))
        total_ev = float(sum(ev.values()))
        row["dense_macs"] = round(total_dense, 1)
        row["event_driven_acs"] = round(total_ev, 1)
        row["macs_reduction_ratio"] = round(1.0 - total_ev / total_dense, 4)
        row["per_layer"] = {"dense_macs": {k: round(v, 1) for k, v in dense.items()},
                            "input_density": {k: round(v, 4) for k, v in dens_map.items()},
                            "event_acs": {k: round(v, 1) for k, v in ev.items()}}
        # 前端 φ1（conv1+depthwise）单独拆分
        phi1_dense = dense["conv1"] + dense["depthwise"]
        phi1_ev = ev["conv1"] + ev["depthwise"]
        row["phi1"] = {"dense_macs": round(phi1_dense, 1), "event_acs": round(phi1_ev, 1),
                       "reduction": round(1.0 - phi1_ev / phi1_dense, 4)}
        # 输入脉冲载荷（HBC 传输视角）
        row["input_spikes_per_trial"] = round(_input_spikes_per_trial(model, f_delta), 1)
        per_model[name] = row
        print(f"[{name}] clean={row['clean_acc']*100:.2f}% 8dB={row['8dB_acc']*100:.2f}% "
              f"输入密度={f_delta*100:.1f}% "
              f"事件驱动ACs/密集MACs={total_ev/total_dense*100:.1f}% "
              f"（省 {row['macs_reduction_ratio']*100:.1f}%）")

    # ---- 对比表（控制台）----
    print("\n" + "=" * 104)
    print("全脉冲 DeltaTCNet 对比表（final test @ eval，cross-session，2592 试次）")
    print("=" * 104)
    hdr = (f"{'模型':<22}{'clean':>9}{'8dB':>9}{'8dB+b':>9}{'输入密度':>10}"
           f"{'事件/密集':>11}{'φ1省':>9}")
    print(hdr)
    for n in names:
        r = per_model[n]
        print(f"{n:<22}{r['clean_acc']*100:>8.2f}%{r['8dB_acc']*100:>8.2f}%"
              f"{r['8dB+burst_acc']*100:>8.2f}%{r['clean_input_density']*100:>9.1f}%"
              f"{r['event_driven_acs']/r['dense_macs']*100:>10.1f}%"
              f"{r['phi1']['reduction']*100:>8.1f}%")
    print("-" * 104)
    print("逐层发放率（clean / 8dB）与输入密度：")
    for n in names:
        r = per_model[n]
        fc = r["clean_fr"]; f8 = r["8dB_fr"]
        parts = "  ".join(f"{k.split('.')[-1]}={fc[k]*100:.1f}/{f8[k]*100:.1f}%" for k in sorted(fc))
        print(f"  {n:<20}: {parts}")

    # ---- 落盘 JSON ----
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {
        "meta": {
            "model": "全脉冲 DeltaTCNet（delta 输入脉冲编码 + LIF-EEG-TCNet brief, 9512 参数）",
            "data": "data_4class/eval.npz (22ch 四分类, 2592 试次, cross-session)",
            "test_noise_seed": TEST_NOISE_SEED,
            "noise_spec": "5bit 量化(qrange=4.0) + AWGN σ/H=10^(-SNR/20)；8dB+突发 附加域",
            "baseline": "LIF-EEG-TCNet + HBC 增强 aug_s42（clean 63.58% / 8dB 54.67%，连续输入）",
            "event_driven_assumption": "ACs(layer) ≈ dense_MACs(layer) × 实测输入脉冲密度；"
                                       "conv1 输入 = delta 脉冲（密度实测）；"
                                       "depthwise 输入 = BN(conv1) 连续输出，诚实按 1.0 计；"
                                       "LIF 之后各层按实测发放率折算；"
                                       "池化/加法按脉冲计数估零；理想事件驱动硬件或稀疏内核实现的理论潜力",
            "device": str(device),
        },
        "domains": [l for l, _ in domains],
        "per_model": per_model,
    }
    (out_dir / "delta_eval.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n✓ 结果已落盘：{out_dir / 'delta_eval.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
