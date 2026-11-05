#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
robust_ablation.py — 约束驱动鲁棒解码的细粒度消融（HBC 噪声类型 × SNR 分档）

复用（不改动）：
  - 模型：~/BCI/luckfox-deploy/model/eegnet.py 的 EEGNet（8ch 二分类，1650 参数）
  - 数据：~/BCI/luckfox-deploy/data/processed/{train,eval}.npz（各 1296 试次，hold-out 早停）
  - 噪声：同目录 hbc_noise_advanced.py 的 hbc_noise()（5bit 量化 + AWGN/突发/脉冲可叠加）

两组消融
--------
  (a) 噪声类型消融（3 训练配置 × 4 测试域）：
        训练增强 ∈ {无增强, 纯AWGN增强(σ/H=0.1), AWGN+突发增强(σ/H=0.1+σb=0.3)}
        final test ∈ {干净, HBC噪声(5bit+AWGN), 突发噪声(5bit+AWGN+burst), 脉冲噪声(补充)}
      看「突发噪声增强」是否比「纯 AWGN 增强」更鲁棒。

  (b) SNR 细粒度分档：
        对 {无增强, 纯AWGN增强} 两个模型，在 HBC 噪声下扫 σ/H ∈ {0.05, 0.1, 0.2, 0.4}
        （≈ 26/20/14/8 dB，对齐板载 SNR 瀑布拐点 8~14dB），画鲁棒性退化曲线，
        判断增强训练是否把「精度崩塌拐点」推向更低 SNR。

hold-out 早停（对齐 train.py 的 --val-split）：训练集划 15% 作验证（早停），
final test 用 eval.npz（E 会话，cross-session，只测一次，不参与早停）。
划分方式**严格对齐 train.py**：torch.randperm(n, generator=torch.Generator().manual_seed(seed))，
取前 n_val 作验证集（等价 random_split）。

多种子：默认跑 3 个训练种子，每个 cell 报 mean ± std，消除 EEGNet 小模型单种子方差。

用法
----
  # 快速冒烟（1 epoch，验证管线）
  env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin \\
      /opt/miniconda3/envs/openmmlab/bin/python robust_ablation.py --quick
  # 完整消融（3 种子）
  env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin \\
      /opt/miniconda3/envs/openmmlab/bin/python robust_ablation.py

输出（默认 ablation_results/）
  results.json   全部准确率数字（raw 每种子 + mean/std 聚合）
  snr_curve.png  SNR 退化曲线（暗色主题，对齐 PPT #1B2A4A）
  ckpt_<种子>_<配置>.pth  各训练配置的最优权重（供复现/复核）
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------------------------
# 路径与导入
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
EEGNET_DIR = Path(os.environ.get("EEGNET_DIR", str(Path.home() / "BCI/luckfox-deploy/model")))
PROCESSED_DIR = Path(os.environ.get(
    "BCI_PROCESSED_DIR", str(Path.home() / "BCI/luckfox-deploy/data/processed")))
sys.path.insert(0, str(EEGNET_DIR))
sys.path.insert(0, str(HERE))

from eegnet import EEGNet                      # noqa: E402  只读复用，不改动原代码
from hbc_noise_advanced import hbc_noise       # noqa: E402  噪声注入接口

TEST_NOISE_SEED = 20240828  # 测试期噪声固定种子：所有模型看到同一份噪声实现，公平对比


# ---------------------------------------------------------------------------
# 设备选择
# ---------------------------------------------------------------------------
def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# 数据加载 + hold-out 划分（严格对齐 train.py 的 --val-split / random_split）
# ---------------------------------------------------------------------------
def load_data(val_split: float, seed: int):
    tr = np.load(PROCESSED_DIR / "train.npz", allow_pickle=False)
    ev = np.load(PROCESSED_DIR / "eval.npz", allow_pickle=False)
    X_tr = torch.from_numpy(np.asarray(tr["X"], dtype=np.float32))
    y_tr = torch.from_numpy(np.asarray(tr["y"], dtype=np.int64))
    X_ev = torch.from_numpy(np.asarray(ev["X"], dtype=np.float32))
    y_ev = torch.from_numpy(np.asarray(ev["y"], dtype=np.int64))

    # 与 train.py 的 random_split(train_ds, [n_train, n_val], generator=g) 等价：
    # random_split 内部 = randperm(sum(lengths), generator=g)，前 n_val 作第一子集（验证）。
    n = len(X_tr)
    n_val = max(1, int(n * val_split))
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=gen)
    val_idx = perm[:n_val]
    tr_idx = perm[n_val:]
    return (X_tr[tr_idx], y_tr[tr_idx], X_tr[val_idx], y_tr[val_idx], X_ev, y_ev)


# ---------------------------------------------------------------------------
# 训练单个配置（hold-out 早停，返回最优 state_dict 与验证历史）
# ---------------------------------------------------------------------------
def train_one(name, X_tr, y_tr, X_val, y_val, aug_kwargs, args, device, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_ds = TensorDataset(X_tr, y_tr)
    val_ds = TensorDataset(X_val, y_val)
    gen = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              generator=gen, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = EEGNet(n_channels=8, n_classes=2, input_length=500,
                   F1=8, D=2, F2=16, kernel_length=63, dropout=0.5, activation="elu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_acc, best_epoch, best_state, patience_counter = 0.0, -1, None, 0
    history = []
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        tot_loss, tot_corr, tot_n = 0.0, 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            if aug_kwargs is not None:
                xb = hbc_noise(xb, **aug_kwargs)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            tot_loss += loss.item() * xb.size(0)
            tot_corr += (logits.argmax(1) == yb).sum().item()
            tot_n += xb.size(0)

        model.eval()
        val_corr, val_n = 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                val_corr += (model(xb).argmax(1) == yb).sum().item()
                val_n += xb.size(0)
        val_acc = val_corr / val_n
        history.append({"epoch": epoch, "train_loss": round(tot_loss / tot_n, 6),
                        "train_acc": round(tot_corr / tot_n, 4),
                        "val_acc": round(val_acc, 4)})

        if val_acc > best_acc:
            best_acc, best_epoch = val_acc, epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                break

    dt = time.time() - t_start
    print(f"  ✓ {name}: best_val={best_acc*100:.2f}% @epoch {best_epoch} ({dt:.1f}s)")
    return best_state, history, best_epoch, best_acc


# ---------------------------------------------------------------------------
# 测试期评估（固定噪声种子，返回准确率）
# ---------------------------------------------------------------------------
def evaluate(state, X_ev, y_ev, noise_spec, batch_size, device):
    model = EEGNet(n_channels=8, n_classes=2, input_length=500,
                   F1=8, D=2, F2=16, kernel_length=63, dropout=0.5, activation="elu")
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()
    torch.manual_seed(TEST_NOISE_SEED)  # 固定噪声种子：所有模型同一份噪声
    correct, total = 0, 0
    with torch.no_grad():
        for i in range(0, len(X_ev), batch_size):
            xb = X_ev[i:i + batch_size].to(device)
            yb = y_ev[i:i + batch_size].to(device)
            if noise_spec is not None:
                xb = hbc_noise(xb, **noise_spec)
            correct += (model(xb).argmax(1) == yb).sum().item()
            total += yb.size(0)
    return correct / total


# ---------------------------------------------------------------------------
# 训练 / 测试域配置
# ---------------------------------------------------------------------------
# (a) 训练增强配置：None 表示不增强
TRAIN_CONFIGS = {
    "无增强": None,
    "纯AWGN增强": dict(qbits=5, qrange=4.0, sigma_ratio=0.1),
    "AWGN+突发增强": dict(qbits=5, qrange=4.0, sigma_ratio=0.1,
                          burst_sigma_ratio=0.3, burst_len=16, burst_prob=0.005),
}

# (a) 测试域（干净 / HBC噪声 / 突发噪声 / 脉冲噪声-补充）
TEST_DOMAINS = {
    "干净": dict(qbits=0, sigma_ratio=0.0),
    "HBC噪声": dict(qbits=5, qrange=4.0, sigma_ratio=0.1),
    "突发噪声": dict(qbits=5, qrange=4.0, sigma_ratio=0.1,
                     burst_sigma_ratio=0.3, burst_len=16, burst_prob=0.005),
    "脉冲噪声(补充)": dict(qbits=5, qrange=4.0, sigma_ratio=0.1,
                          impulse_amp=1.0, impulse_prob=0.003),
}

# (b) SNR 细粒度分档：σ/H → (标签, 测试噪声 spec)
SNR_SWEEP = [
    ("干净(∞dB)", dict(qbits=0, sigma_ratio=0.0)),
    ("σ/H=0.05 (26dB)", dict(qbits=5, qrange=4.0, sigma_ratio=0.05)),
    ("σ/H=0.1  (20dB)", dict(qbits=5, qrange=4.0, sigma_ratio=0.1)),
    ("σ/H=0.2  (14dB)", dict(qbits=5, qrange=4.0, sigma_ratio=0.2)),
    ("σ/H=0.4  (8dB)", dict(qbits=5, qrange=4.0, sigma_ratio=0.4)),
]

SWEEP_MODELS = ["无增强", "纯AWGN增强"]


def _run_one_seed(seed, args, device, out_dir, verbose=True):
    """单种子：训练 3 配置 + 两组消融评估，返回结果 dict。"""
    X_tr, y_tr, X_val, y_val, X_ev, y_ev = load_data(args.val_split, seed)

    states, train_meta = {}, {}
    for name, aug in TRAIN_CONFIGS.items():
        if verbose:
            print(f"  [seed={seed}] 训练 {name}" + ("" if aug is None else f"  ({aug})"))
        st, hist, be, ba = train_one(name, X_tr, y_tr, X_val, y_val, aug, args, device, seed)
        states[name] = st
        train_meta[name] = {"best_val_acc": round(ba, 4), "best_epoch": be}
        if not args.quick:
            torch.save(st, out_dir / f"ckpt_s{seed}_{name}.pth")

    ablation_a = {}
    for name in TRAIN_CONFIGS:
        ablation_a[name] = {}
        for dom, spec in TEST_DOMAINS.items():
            ablation_a[name][dom] = round(
                evaluate(states[name], X_ev, y_ev, spec, args.batch_size, device), 4)

    ablation_b = {}
    for label, spec in SNR_SWEEP:
        ablation_b[label] = {}
        for m in SWEEP_MODELS:
            ablation_b[label][m] = round(
                evaluate(states[m], X_ev, y_ev, spec, args.batch_size, device), 4)

    return {"train_meta": train_meta, "ablation_a": ablation_a, "ablation_b_snr": ablation_b}


def _aggregate(per_seed: list[dict], container_key: str, keys, nested_leaf_keys) -> dict:
    """把多种子结果聚合成 {key: {leaf: [mean, std]}}。per_seed 元素含 container_key 子 dict。"""
    agg = {}
    for k in keys:
        agg[k] = {}
        for leaf in nested_leaf_keys:
            vals = [s[container_key][k][leaf] for s in per_seed]
            if vals:
                agg[k][leaf] = [round(float(np.mean(vals)), 4),
                                round(float(np.std(vals)), 4)]
    return agg


def _save_snr_curve(agg_b: dict, out_dir: Path) -> None:
    """画 (b) 组 SNR 退化曲线（暗色主题，对齐 PPT #1B2A4A）。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams["font.sans-serif"] = ["Hiragino Sans GB", "STHeiti", "Arial Unicode MS"]
        plt.rcParams["axes.unicode_minus"] = False

        labels = [k for k in agg_b]
        base = [agg_b[k]["无增强"][0] * 100 for k in labels]
        base_e = [agg_b[k]["无增强"][1] * 100 for k in labels]
        awgn = [agg_b[k]["纯AWGN增强"][0] * 100 for k in labels]
        awgn_e = [agg_b[k]["纯AWGN增强"][1] * 100 for k in labels]

        fig, ax = plt.subplots(figsize=(7.5, 5), dpi=150, facecolor="#1B2A4A")
        ax.set_facecolor("#243555")
        for sp in ax.spines.values():
            sp.set_color("#8FA3BF")
        ax.tick_params(colors="white", labelsize=11)
        ax.grid(True, color="#8FA3BF", alpha=0.25, lw=0.8)

        xpos = list(range(len(labels)))
        ax.errorbar(xpos, base, yerr=base_e, color="#E76F51", lw=2.2, marker="o", ms=6,
                    capsize=4, label="无增强（baseline）")
        ax.errorbar(xpos, awgn, yerr=awgn_e, color="#00B4D8", lw=2.2, marker="s", ms=6,
                    capsize=4, label="纯AWGN增强 (σ/H=0.1)")
        ax.axhline(50, color="#8FA3BF", lw=1, ls="--", alpha=0.6)
        ax.set_xticks(xpos)
        ax.set_xticklabels(["∞", "26", "20", "14", "8"], color="white")
        ax.set_xlabel("测试 SNR (dB)，HBC 噪声 σ/H 递减", color="white", fontsize=12)
        ax.set_ylabel("final test 准确率 (%)", color="white", fontsize=12)
        ax.set_title("鲁棒性随 SNR 的退化曲线（多种子 mean±std）",
                     color="white", fontsize=14, weight="bold", pad=12)
        ax.legend(facecolor="#243555", edgecolor="#8FA3BF", labelcolor="white", loc="lower left")
        ax.set_ylim(40, 90)
        fig.tight_layout(pad=2.0)
        path = out_dir / "snr_curve.png"
        fig.savefig(path, facecolor="#1B2A4A", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  SNR 曲线已保存：{path}")
    except Exception as e:  # noqa: BLE001
        print(f"  ⚠ SNR 曲线图保存失败（不影响结果）：{e}")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="HBC 噪声鲁棒解码细粒度消融")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42, help="主种子（hold-out 划分基准）")
    ap.add_argument("--seeds", type=str, default="42,123,2024",
                    help="训练种子列表（逗号分隔），多种子取 mean±std")
    ap.add_argument("--val-split", type=float, default=0.15)
    ap.add_argument("--out-dir", type=str, default=str(HERE / "ablation_results"))
    ap.add_argument("--quick", action="store_true", help="冒烟测试：1 epoch、不画图、单种子")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.replace(" ", "").split(",") if s]
    if args.quick:
        args.epochs, args.patience = 1, 5
        seeds = seeds[:1]

    device = _pick_device()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("鲁棒解码细粒度消融（HBC 噪声类型 × SNR 分档，多种子）")
    print("=" * 72)
    print(f"设备       : {device}")
    print(f"数据       : {PROCESSED_DIR}")
    print(f"hold-out   : val_split={args.val_split}, 早停 patience={args.patience}")
    print(f"训练种子   : {seeds}")
    print(f"训练配置数 : {len(TRAIN_CONFIGS)}（epochs={args.epochs}, batch={args.batch_size}, lr={args.lr}）")
    print(f"hold-out 划分严格对齐 train.py 的 random_split(seed={args.seed}) 语义（每训练种子独立划分）\n")

    per_seed = []
    for seed in seeds:
        print(f"[seed={seed}]")
        res = _run_one_seed(seed, args, device, out_dir)
        per_seed.append({"seed": seed, **res})
        # 打印该种子 (a)/(b) 摘要
        for name in TRAIN_CONFIGS:
            row = "  ".join(f"{d}={res['ablation_a'][name][d]*100:.2f}%"
                            for d in TEST_DOMAINS)
            print(f"    (a) {name:<12} {row}")
        print()

    # ---- 聚合多种子 ----
    agg_a = _aggregate(per_seed, "ablation_a", list(TRAIN_CONFIGS), list(TEST_DOMAINS))
    agg_b = _aggregate(per_seed, "ablation_b_snr", [k for k, _ in SNR_SWEEP], SWEEP_MODELS)
    agg_train = _aggregate(per_seed, "train_meta", list(TRAIN_CONFIGS), ["best_val_acc"])

    print("=" * 72)
    print("(a) 噪声类型消融 —— 多种子 mean ± std（final test，E 会话）")
    print("=" * 72)
    for name in TRAIN_CONFIGS:
        row = "  ".join(f"{d}={agg_a[name][d][0]*100:.2f}±{agg_a[name][d][1]*100:.2f}%"
                        for d in TEST_DOMAINS)
        print(f"  {name:<12} {row}")

    print("\n" + "=" * 72)
    print("(b) SNR 细粒度分档 —— 多种子 mean ± std")
    print("=" * 72)
    for label, _ in SNR_SWEEP:
        b = agg_b[label]["无增强"]
        a = agg_b[label]["纯AWGN增强"]
        print(f"  {label:<18} 无增强={b[0]*100:.2f}±{b[1]*100:.2f}%   "
              f"纯AWGN增强={a[0]*100:.2f}±{a[1]*100:.2f}%")

    # ---- 落盘 ----
    results = {
        "meta": {
            "seeds": seeds, "val_split": args.val_split, "epochs": args.epochs,
            "batch_size": args.batch_size, "lr": args.lr, "patience": args.patience,
            "device": str(device), "test_noise_seed": TEST_NOISE_SEED,
            "model": "EEGNet(8ch,2类,1650参数)",
            "data": str(PROCESSED_DIR),
            "split_note": "hold-out 严格对齐 train.py random_split(seed) 语义：randperm 前 n_val 作验证",
            "train_configs": TRAIN_CONFIGS,
            "test_domains": TEST_DOMAINS,
            "snr_sweep": [{"label": l, "spec": s} for l, s in SNR_SWEEP],
        },
        "per_seed": per_seed,
        "aggregated": {
            "train_val_acc": agg_train,
            "ablation_a": agg_a,
            "ablation_b_snr": agg_b,
        },
    }
    (out_dir / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    if not args.quick:
        _save_snr_curve(agg_b, out_dir)

    print("\n✓ 结果已落盘：", out_dir / "results.json")
    print("  下一步：把数字填入 鲁棒性深化结果.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
