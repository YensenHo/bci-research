#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ssl_probe.py — 预训练 encoder 的机制诊断（不进对比矩阵，只回答「特征里有没有东西」）
1. 冻结特征线性探针：AE encoder (ssl_ae_aug) 输出 GAP 后 12 维特征 →
   LogisticRegression（sklearn 若无则 torch 线性头），在 100%/10% 标签上训，eval 测；
   与「随机初始化 encoder 特征」同探针对照 → 分离度差异 = 预训练学的类信息下限。
2. 重建质量按区域量化：mask 区 / 非 mask 区 R²（每个 val 试次），佐证曲线数值。
用法同项目：env -i ... python ssl_probe.py"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ALGO_DIR = Path(__file__).resolve().parent
DATA_DIR = Path("/Users/heyixuan/BCI/luckfox-deploy/data")
OUT = ALGO_DIR / "ssl_ae_results"
sys.path.insert(0, str(ALGO_DIR))
sys.path.insert(0, str(DATA_DIR))

from dataset import EEGDataset          # noqa: E402
from eeg_tcnet_lif import EEGTCNetLIF   # noqa: E402
from pretrain_masked_ae import MaskedEEGAutoencoder, make_time_masks  # noqa: E402

DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


@torch.no_grad()
def encode_all(model: torch.nn.Module, npz_path: Path, bs: int = 64) -> tuple[np.ndarray, np.ndarray]:
    ds = EEGDataset(npz_path)
    feats, ys = [], []
    model.eval()
    for i in range(0, len(ds), bs):
        xb = torch.from_numpy(ds.X[i:i + bs]).to(DEV)
        z = model.encode_features(xb)          # (B,12,15)
        f = z.mean(dim=2).cpu().numpy()        # GAP
        feats.append(f)
        ys.append(ds.y[i:i + bs])
    return np.concatenate(feats), np.concatenate(ys)


def linear_probe_acc(feats_tr: np.ndarray, y_tr: np.ndarray,
                     feats_ev: np.ndarray, y_ev: np.ndarray,
                     frac: float, seed: int = 42) -> float:
    """sklearn LogisticRegression 线性探针（C 默认 1.0，max_iter 加大）。"""
    from sklearn.linear_model import LogisticRegression
    n = len(y_tr)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    nl = int(round(n * frac))
    idx = perm[:nl]
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(feats_tr[idx], y_tr[idx])
    return float(clf.score(feats_ev, y_ev))


def recon_region_r2() -> dict:
    """mask 区 / 非 mask 区重建 R²（val 子集 64 试次，固定 mask seed）。"""
    ds = EEGDataset(ALGO_DIR / "data_4class/train.npz")
    ae = MaskedEEGAutoencoder().to(DEV)
    ck = torch.load(OUT / "ssl_ae_aug.pth", map_location="cpu", weights_only=False)
    st = ae.state_dict()
    for k, v in ck["encoder_state_dict"].items():
        st["encoder." + k] = v
    for k, v in ck["decoder_state_dict"].items():
        st[k] = v
    ae.load_state_dict(st)
    ae.eval()
    rng = np.random.default_rng(7)
    sm, sf, sn = 0.0, 0.0, 0
    with torch.no_grad():
        for i in range(64):
            x = torch.from_numpy(ds.X[i]).to(DEV)          # (1,1,22,500)
            m, _ = make_time_masks(1, 500, 0.5, 50, 100, rng)
            m = m.to(DEV)
            xh = ae(x * (1 - m))[0]                        # (22,500)
            x0 = x[0, 0]
            mm = m[0, 0, 0].bool()
            r2 = lambda a, b: 1 - ((a - b) ** 2).sum() / ((b - b.mean()) ** 2).sum()  # noqa: E731
            sm += float(r2(xh[:, mm], x0[:, mm]))
            sf += float(r2(xh[:, ~mm], x0[:, ~mm]))
            sn += 1
    return {"masked_R2": sm / sn, "visible_R2": sf / sn, "n": sn}


def main() -> int:
    print("设备:", DEV)
    # 重建 R² 分区量化
    r2 = recon_region_r2()
    print(f"重建 R²（64 试次，固定 mask）：mask 区 {r2['masked_R2']:.4f} | "
          f"非 mask 区 {r2['visible_R2']:.4f}")
    # 线性探针：ssl 特征 vs 随机特征（同探针协议，同标签子集）
    ck = torch.load(OUT / "ssl_ae_aug.pth", map_location="cpu", weights_only=False)
    ssl_model = MaskedEEGAutoencoder().to(DEV)
    st = ssl_model.state_dict()
    for k, v in ck["encoder_state_dict"].items():
        st["encoder." + k] = v
    ssl_model.load_state_dict(st, strict=False)
    rand_model = MaskedEEGAutoencoder().to(DEV)   # 随机初始化对照

    ftr, ytr = encode_all(ssl_model, ALGO_DIR / "data_4class/train.npz")
    fev, yev = encode_all(ssl_model, ALGO_DIR / "data_4class/eval.npz")
    rtr, _ = encode_all(rand_model, ALGO_DIR / "data_4class/train.npz")
    rev, _ = encode_all(rand_model, ALGO_DIR / "data_4class/eval.npz")
    print(f"特征维度 {ftr.shape[1]}，train {ftr.shape[0]} / eval {fev.shape[0]} 试次")
    for frac in (1.0, 0.3, 0.1):
        a_ssl = linear_probe_acc(ftr, ytr, fev, yev, frac)
        a_rnd = linear_probe_acc(rtr, ytr, rev, yev, frac)
        print(f"线性探针 {frac:.0%} 标签：SSL 特征 {a_ssl * 100:5.2f}% | "
              f"随机特征 {a_rnd * 100:5.2f}% | Δ={ (a_ssl - a_rnd) * 100:+.2f}pp（chance 25%）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
