#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pretrain_masked_ae.py — Masked EEG Autoencoder 自监督预训练（BCIC 2a，EEG-TCNet 前端）

背景
----
20260820 方案（PPT slide8-9）落地：自监督预训练对端侧小模型（EEG-TCNet/LIF 版，
9.5K 参数）有无红利。本脚本 = 预训练阶段：
  - encoder：EEG-TCNet 卷积前端（复用 eeg_tcnet_lif.py 的 EEGTCNetLIF 的 conv body，
    activation=relu，**不含分类头**：输出 TCN 特征图 (B, FT=12, T'=15)，GAP/fc 不参与）。
    权重键与 EEGTCNetLIF 完全对齐（激活层无参数），下游 relu/lif 微调可 strict 级热启动；
  - mask：随机 mask ≈50% 时间片（连续块 50~100 样本，块内置 0；z-score 数据 0≈均值）；
  - 增强（与 HBC 增强协议同源）：5bit 量化(qrange=4.0) + AWGN σ/H=10^(-SNR/20)，
    SNR∈[10,20]dB 每 batch 随机 + 突发噪声（σb/H=0.3 len=16 p=0.005）——「自监督 × HBC」
    缝成一个增强：encoder 必须在 HBC 退化输入上学到能补全干净信号的表征；
  - decoder：ConvTranspose1d 镜像时间轴（15→60→480→interp 500）+ 通道展开 16→22，
    重建原始 22ch×500（空间混合的逆由 decoder 自学，故重建 MSE 有不可归零的下限）；
  - loss：MSE 只算 mask 区域（标准 MAE 思路：逼模型用上下文补全，防 decoder 抄可见区捷径）；
    val 上同时监控 mask 区域与全窗 MSE（诊断）。

泄漏红线：预训练只用 data_4class/train.npz（T 会话 2592 试次，无标签）；
eval.npz（E 会话）绝不进预训练（也不进任何验证）。内部 val = train 的 10%（T 会话内），
仅用于早停与画曲线。

用法
----
  # 冒烟（防漏加载假象）
  env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin \
      /opt/miniconda3/envs/openmmlab/bin/python pretrain_masked_ae.py --smoke
  # 全量（HBC 增强版，主产物；noaug 版做增强消融）
  ... python pretrain_masked_ae.py --tag ssl_ae_aug
  ... python pretrain_masked_ae.py --tag ssl_ae_noaug --no-aug

产物（algorithm/ssl_ae_results/）
  ssl_ae_{tag}/... 见 main() 末尾：{tag}.pth / {tag}_history.json / {tag}_curve.png / {tag}_recon_example.png
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# ---- 路径 ----
ALGO_DIR = Path(__file__).resolve().parent
DATA_DIR = Path("/Users/heyixuan/BCI/luckfox-deploy/data")
DATA_4CLASS = ALGO_DIR / "data_4class"
OUT_DIR = ALGO_DIR / "ssl_ae_results"

sys.path.insert(0, str(ALGO_DIR))   # eeg_tcnet_lif / hbc_noise_advanced
sys.path.insert(0, str(DATA_DIR))   # dataset.EEGDataset

from dataset import EEGDataset                    # noqa: E402
from eeg_tcnet_lif import EEGTCNetLIF             # noqa: E402
from hbc_noise_advanced import hbc_noise, snr_db_to_sigma_ratio  # noqa: E402


def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# 1. Masked EEG Autoencoder
# ---------------------------------------------------------------------------
class MaskedEEGAutoencoder(nn.Module):
    """encoder = EEG-TCNet conv body（EEGTCNetLIF，activation=relu，输出 pre-GAP TCN 特征）。

    decoder = ConvTranspose1d 镜像时间轴 + 通道展开：
      z (B,12,15) → ConvT1d(12→16, k4 s4) → (B,16,60)
                  → ConvT1d(16→16, k8 s8) → (B,16,480)
                  → Conv1d(16→22, 1×1)     → (B,22,480)   （空间 16→22 展开）
                  → F.interpolate(500)      → (B,22,500)   （pool 取整余量的线性补齐）
    encoder 时间压缩比 500→15 非整数（500→125→15 含 pool 取整），末段用线性插值补齐，
    loss 在原始 22ch×500 空间计算。
    """

    def __init__(
        self,
        n_channels: int = 22,
        input_length: int = 500,
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
        FT: int = 12,
        kernel_length: int = 63,
        tcn_kernel_size: int = 12,
        pool1: int = 4,
        pool2: int = 8,
        dropout: float = 0.25,
    ):
        super().__init__()
        # encoder：与下游 EEGTCNetLIF 同结构（activation=relu，激活层无参数，
        # 全部 conv/BN 权重键与 LIF 版一一对应 → 微调 strict 热启动）
        self.encoder = EEGTCNetLIF(
            n_channels=n_channels, n_classes=4, input_length=input_length,
            F1=F1, D=D, F2=F2, FT=FT, kernel_length=kernel_length,
            tcn_kernel_size=tcn_kernel_size, pool1=pool1, pool2=pool2,
            dropout=dropout, activation="relu",
        )
        self.FT = FT
        self.n_channels = n_channels
        # decoder
        self.up1 = nn.ConvTranspose1d(FT, F2, kernel_size=4, stride=4)   # 15→60
        self.up2 = nn.ConvTranspose1d(F2, F2, kernel_size=8, stride=8)   # 60→480
        self.expand = nn.Conv1d(F2, n_channels, kernel_size=1)           # 16→22 通道展开

    # ---- encoder：逐行复刻 eeg_tcnet_lif.forward_features，但停在 GAP 之前 ----
    def encode_features(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B,1,22,500) → (B, FT, 15) pre-GAP TCN 特征图。"""
        m = self.encoder
        B, C, H, T = x.shape
        pad1 = x.new_zeros(B, C, H, m.pad1_amt)
        x = torch.cat([pad1, x, pad1], dim=3)
        x = m.conv1(x)
        x = m.bn1(x)
        x = m.depthwise(x)
        x = m.bn2(x)
        x = m.act1(x)
        x = m.pool1(x)
        x = m.dropout1(x)
        Bc, Cc, Hc, Tc = x.shape
        pad_s = x.new_zeros(Bc, Cc, Hc, m.pad_sep)
        x = torch.cat([pad_s, x, pad_s], dim=3)
        x = m.sep_depthwise(x)
        x = m.pointwise(x)
        x = m.bn3(x)
        x = m.act2(x)
        x = m.pool2(x)
        x = m.dropout2(x)
        x = x.view(x.size(0), m.F2, -1)
        x = m.tcn(x)                      # (B, FT, 15)，含残差激活
        return x

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B,FT,15) → (B,22,500)。"""
        x = F.relu(self.up1(z))
        x = F.relu(self.up2(x))
        x = self.expand(x)                # (B,22,480)
        x = F.interpolate(x, size=self.encoder.input_length, mode="linear",
                          align_corners=False)   # 480→500
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode_features(x))

    def encoder_body_state_dict(self) -> dict:
        """下游可热启动的 encoder 权重（剔除分类头 fc —— 随机初始化，不参与预训练）。"""
        return {k: v.detach().cpu().clone() for k, v in self.encoder.state_dict().items()
                if not k.startswith("fc.")}


# ---------------------------------------------------------------------------
# 2. 时间片 mask 生成（连续块，≈ratio 覆盖率，块内置 0）
# ---------------------------------------------------------------------------
def make_time_masks(batch: int, length: int, ratio: float,
                    block_min: int, block_max: int,
                    rng: np.random.Generator) -> tuple[torch.Tensor, np.ndarray]:
    """返回 (mask, fracs)。mask: (B,1,1,length) float32，1=mask 区域；fracs: (B,) 实际覆盖率。"""
    masks = np.zeros((batch, length), dtype=np.float32)
    fracs = np.zeros(batch, dtype=np.float32)
    for i in range(batch):
        cov = 0.0
        attempts = 0
        while cov < ratio and attempts < 64:
            L = int(rng.integers(block_min, block_max + 1))
            start = int(rng.integers(0, length - L + 1))
            masks[i, start:start + L] = 1.0
            cov = float(masks[i].mean())
            attempts += 1
            if cov > ratio + 0.08:      # 硬上限（≈58%），避免超 mask
                masks[i, start:start + L] = 0.0
                cov = float(masks[i].mean())
                break
        fracs[i] = cov
    return torch.from_numpy(masks).unsqueeze(1).unsqueeze(1), fracs   # (B,1,1,T)


# ---------------------------------------------------------------------------
# 3. HBC 增强（与 train_eeg_tcnet_hbc_aug.py 同协议：5bit + AWGN SNR∈[min,max] + 突发）
# ---------------------------------------------------------------------------
def make_aug_spec(aug: bool, snr_min: float, snr_max: float) -> dict | None:
    if not aug:
        return None
    return dict(enabled=True, snr_min_db=snr_min, snr_max_db=snr_max,
                qbits=5, qrange=4.0, burst_sigma_ratio=0.3, burst_len=16,
                burst_prob=0.005)


def apply_aug(x: torch.Tensor, aug: dict) -> torch.Tensor:
    """x: (B,1,C,T)；每 batch 随机 SNR ∈ [snr_min, snr_max] dB（dB 均匀）。"""
    snr_db = aug["snr_min_db"] + (aug["snr_max_db"] - aug["snr_min_db"]) * float(torch.rand(1).item())
    sigma_ratio = snr_db_to_sigma_ratio(snr_db)
    return hbc_noise(x, qbits=aug["qbits"], qrange=aug["qrange"], sigma_ratio=sigma_ratio,
                     burst_sigma_ratio=aug["burst_sigma_ratio"], burst_len=aug["burst_len"],
                     burst_prob=aug["burst_prob"])


# ---------------------------------------------------------------------------
# 4. 训练
# ---------------------------------------------------------------------------
def masked_mse(x_hat: torch.Tensor, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """只算 mask 区域的 MSE。x_hat/x: (B,22,500)，mask: (B,1,1,500)→(B,22,500) 广播。"""
    diff = (x_hat - x) ** 2
    m = mask.reshape(mask.size(0), 1, mask.size(-1)).expand_as(diff)
    return (diff * m).sum() / (m.sum() + 1e-8)


def full_mse(x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return ((x_hat - x) ** 2).mean()


def main() -> int:
    ap = argparse.ArgumentParser(description="Masked EEG Autoencoder 自监督预训练（2a train 无标签）")
    ap.add_argument("--tag", type=str, default="", help="产物 tag（默认 ssl_ae_aug / ssl_ae_noaug）")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--val-frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    # mask 参数
    ap.add_argument("--mask-ratio", type=float, default=0.5)
    ap.add_argument("--block-min", type=int, default=50)
    ap.add_argument("--block-max", type=int, default=100)
    # HBC 增强
    ap.add_argument("--no-aug", action="store_true", help="关闭 HBC 增强（消融）")
    ap.add_argument("--snr-min", type=float, default=10.0)
    ap.add_argument("--snr-max", type=float, default=20.0)
    # 冒烟
    ap.add_argument("--smoke", action="store_true", help="小数据小 epoch 冒烟")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = _pick_device()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    aug = make_aug_spec(not args.no_aug, args.snr_min, args.snr_max)
    tag = args.tag or ("ssl_ae_aug" if aug else "ssl_ae_noaug")
    if args.smoke:
        tag = f"{tag}_smoke"
    aug_desc = "HBC增强[5bit+AWGN SNR∈[10,20]dB +burst σb/H=0.3 len=16 p=0.005]" if aug else "无增强（消融）"

    train_npz = DATA_4CLASS / "train.npz"
    if not train_npz.exists():
        raise SystemExit(f"✗ 数据集缺失：{train_npz}")
    ds = EEGDataset(train_npz)
    n_all = len(ds)
    n_val = int(n_all * args.val_frac)
    if args.smoke:
        n_all, n_val = 256, 32
        ds = torch.utils.data.Subset(ds, list(range(n_all)))   # 冒烟只验管线
    g = torch.Generator().manual_seed(args.seed)
    tr_ds, va_ds = torch.utils.data.random_split(ds, [n_all - n_val, n_val], generator=g)
    tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True)
    va_loader = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False)
    print(f"设备     : {device}")
    print(f"预训练集 : {len(tr_ds)}（train.npz T 会话内 hold-out，无标签）| val {len(va_ds)}")
    print(f"增强     : {aug_desc} | mask 比例 {args.mask_ratio}（块 {args.block_min}-{args.block_max} 样本）")
    print(f"泄漏红线 : eval.npz（E 会话）绝不进入预训练 ✓")

    model = MaskedEEGAutoencoder().to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_enc = sum(p.numel() for p in model.encoder.parameters() if p.requires_grad) - \
        sum(p.numel() for p in model.encoder.fc.parameters())
    print(f"参数量   : encoder body {n_enc:,}（= 下游可热启动权重，不含 fc）| 全 AE {n_params:,}")

    # 冒烟自检：前向 + mask 覆盖率 + 形状
    dummy = torch.randn(2, 1, 22, 500)
    rng0 = np.random.default_rng(0)
    mk0, fr0 = make_time_masks(2, 500, args.mask_ratio, args.block_min, args.block_max, rng0)
    assert abs(float(mk0.mean()) - args.mask_ratio) < 0.1, f"mask 覆盖率异常: {mk0.mean():.3f}"
    with torch.no_grad():
        z = model.encode_features(dummy.to(device))
        xh = model(dummy.to(device))
    print(f"自检     : enc → {tuple(z.shape)} | recon → {tuple(xh.shape)} | "
          f"mask 覆盖率 {float(mk0.mean()):.3f}（目标 {args.mask_ratio}）")
    assert tuple(xh.shape) == (2, 22, 500)

    # 固定 val mask（无噪声，纯补全诊断：跨 epoch 可比）
    rng_val = np.random.default_rng(1000 + args.seed)
    va_masks = []
    va_xs = []
    for xb, _ in va_loader:
        m, _ = make_time_masks(xb.size(0), 500, args.mask_ratio, args.block_min,
                               args.block_max, rng_val)
        va_masks.append(m)
        va_xs.append(xb)
    print(f"val 固定 mask 已生成（{len(va_xs)} batch，噪声关闭）")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    best_val, best_ep, best_state, patience = float("inf"), -1, None, 0
    history: list[dict] = []

    def _val_metrics() -> tuple[float, float]:
        model.eval()
        vm, vf, vn = 0.0, 0.0, 0
        with torch.no_grad():
            for (xb, _), m in zip(va_loader, va_masks):
                xb, m = xb.to(device), m.to(device)
                x_in = xb * (1.0 - m)
                xh = model(x_in)
                vm += masked_mse(xh, xb[:, 0], m).item() * xb.size(0)
                vf += full_mse(xh, xb[:, 0]).item() * xb.size(0)
                vn += xb.size(0)
        return vm / vn, vf / vn

    print(f"\n开始预训练（{tag}，{args.epochs} epochs）...")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        tm, tn, frac_sum, n_smp = 0.0, 0, 0.0, 0
        for xb, _ in tr_loader:
            xb = xb.to(device)
            rng = np.random.default_rng(args.seed * 100000 + epoch * 1000 + tn)
            mask, fracs = make_time_masks(xb.size(0), 500, args.mask_ratio,
                                          args.block_min, args.block_max, rng)
            mask = mask.to(device)
            x_in = xb.clone()
            if aug is not None:                       # 先 HBC 退化，再 mask 置零
                x_in = apply_aug(x_in, aug)
            x_in = x_in * (1.0 - mask)
            optimizer.zero_grad()
            xh = model(x_in)
            loss = masked_mse(xh, xb[:, 0], mask)
            loss.backward()
            optimizer.step()
            tm += loss.item() * xb.size(0)
            frac_sum += float(fracs.sum())
            n_smp += xb.size(0)
            tn += xb.size(0)
        scheduler.step()
        tr_mse = tm / n_smp
        vm, vf = _val_metrics()
        dt = time.time() - t0
        hist = {"epoch": epoch, "train_masked_mse": round(tr_mse, 6),
                "val_masked_mse": round(vm, 6), "val_full_mse": round(vf, 6),
                "mask_frac": round(float(frac_sum) / n_smp, 4), "sec": round(dt, 1)}
        history.append(hist)
        print(f"  Epoch {epoch:3d} | train_mask_mse={tr_mse:.5f} "
              f"val_mask_mse={vm:.5f} val_full_mse={vf:.5f} | mask={hist['mask_frac']:.2f} | {dt:.1f}s")
        if vm < best_val:
            best_val, best_ep = vm, epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= args.patience:
                print(f"\n早停：val masked MSE 连续 {args.patience} epoch 未下降（best ep {best_ep}）")
                break
    if best_state is None:
        raise SystemExit("✗ 预训练未产生有效权重")
    model.load_state_dict(best_state)
    model.eval()
    print(f"\n✓ 预训练完成：best val masked MSE = {best_val:.5f}（epoch {best_ep}）")

    # ---- 产物 ----
    ck = {
        "tag": tag, "aug": aug is not None, "aug_desc": aug_desc,
        "encoder_state_dict": model.encoder_body_state_dict(),
        "decoder_state_dict": {k: v for k, v in model.state_dict().items()
                               if k.startswith(("up1", "up2", "expand"))},
        "model_args": dict(n_channels=22, input_length=500, F1=8, D=2, F2=16, FT=12,
                           kernel_length=63, tcn_kernel_size=12, pool1=4, pool2=8,
                           dropout=0.25),
        "mask": dict(ratio=args.mask_ratio, block_min=args.block_min,
                     block_max=args.block_max),
        "best_val_masked_mse": best_val, "best_epoch": best_ep,
        "n_train": len(tr_ds), "n_val": len(va_ds), "n_params_encoder_body": n_enc,
        "epochs_run": len(history), "seed": args.seed,
        "data_leak": "只用 data_4class/train.npz（T 会话）；eval(E) 未进入",
    }
    torch.save(ck, OUT_DIR / f"{tag}.pth")
    (OUT_DIR / f"{tag}_history.json").write_text(
        json.dumps({"meta": {"tag": tag, "aug_desc": aug_desc,
                             "mask": ck["mask"], "seed": args.seed},
                    "history": history}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    print(f"✓ 产物：{OUT_DIR / (tag + '.pth')}")

    # ---- 曲线 + 重建示例图 ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams["font.sans-serif"] = ["Hiragino Sans GB", "STHeiti", "Arial Unicode MS"]
        plt.rcParams["axes.unicode_minus"] = False
        ep = [h["epoch"] for h in history]
        fig, ax = plt.subplots(1, 2, figsize=(11, 4), dpi=150)
        ax[0].plot(ep, [h["train_masked_mse"] for h in history], lw=1.6, label="train masked MSE")
        ax[0].plot(ep, [h["val_masked_mse"] for h in history], lw=1.6, label="val masked MSE")
        ax[0].set_title(f"Masked 重建 MSE 曲线（{aug_desc}）")
        ax[0].set_xlabel("epoch"); ax[0].legend(); ax[0].grid(alpha=0.3)
        ax[1].plot(ep, [h["val_full_mse"] for h in history], lw=1.6, color="#C44")
        ax[1].set_title("val 全窗 MSE（诊断）")
        ax[1].set_xlabel("epoch"); ax[1].grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"{tag}_curve.png", bbox_inches="tight")
        plt.close(fig)

        # 重建示例：val 第 0 个样本（通道 7 = Cz 附近）
        with torch.no_grad():
            xb0, _ = next(iter(va_loader))
            m0, _ = make_time_masks(xb0.size(0), 500, args.mask_ratio, args.block_min,
                                    args.block_max, np.random.default_rng(7))
            m0 = m0.to(device)
            x_in0 = xb0.to(device) * (1.0 - m0)
            xh0 = model(x_in0)
        fig2, ax2 = plt.subplots(figsize=(11, 3.2), dpi=150)
        ch = 7
        t = np.arange(500)
        ax2.plot(t, xb0[0, 0, ch].cpu().numpy(), lw=0.9, color="#333", label="原始（clean）")
        ax2.plot(t, xh0[0, ch].cpu().numpy(), lw=0.9, color="#E07", label="重建（mask 区域为补全）")
        mm = m0[0, 0, 0].cpu().numpy().astype(bool)
        ax2.fill_between(t, -4, 4, where=mm, color="#FA0", alpha=0.15, label="mask 区域")
        ax2.set_ylim(-3.5, 3.5); ax2.legend(loc="upper right", fontsize=8)
        ax2.set_title(f"重建示例（val 样本 ch{ch}，mask 覆盖率 {mm.mean():.0%}）")
        ax2.grid(alpha=0.3)
        fig2.tight_layout()
        fig2.savefig(OUT_DIR / f"{tag}_recon_example.png", bbox_inches="tight")
        plt.close(fig2)
        print(f"✓ 曲线/示例图已保存：{tag}_curve.png / {tag}_recon_example.png")
    except Exception as e:  # noqa: BLE001 —— 图失败不影响权重产物
        print(f"⚠ 绘图失败（不影响训练产物）：{e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
