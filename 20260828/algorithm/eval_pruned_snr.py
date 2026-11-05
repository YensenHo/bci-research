#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_pruned_snr.py — 剪枝版 LIF-EEG-TCNet 的 SNR 扫描评估（clean/20/16/12/8/4 dB）

协议与 eval_hbc_snr_sweep.py 完全一致：eval 集 cross-session、测试噪声 5bit 量化 +
AWGN（σ/H=10^(-SNR/20)）、固定种子 TEST_NOISE_SEED=20240828（所有模型同一份噪声），
并附加 8dB+突发 域。差异：按 ckpt 里的 channels 列表对输入切片。

用法：
  env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin \\
      /opt/miniconda3/envs/openmmlab/bin/python eval_pruned_snr.py \\
      --ckpts 基线,剪16,剪12 --paths a.pth,b.pth,c.pth
输出：channel_prune/snr_sweep_pruned.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

ALGO_DIR = Path(__file__).resolve().parent
PROJECT = Path("/Users/heyixuan/BCI/luckfox-deploy")
DATA_DIR = PROJECT / "data"
DATA_4CLASS = ALGO_DIR / "data_4class"

sys.path.insert(0, str(ALGO_DIR))
sys.path.insert(0, str(DATA_DIR))

from dataset import EEGDataset          # noqa: E402
from eeg_tcnet_lif import EEGTCNetLIF   # noqa: E402
from hbc_noise_advanced import hbc_noise, snr_db_to_sigma_ratio  # noqa: E402

TEST_NOISE_SEED = 20240828


class ChannelSliceDataset(Dataset):
    def __init__(self, base: EEGDataset, keep_idx: list[int]):
        self.base = base
        self.keep_idx = keep_idx

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        x, y = self.base[idx]
        return x[:, self.keep_idx, :], y


def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _eval_noise(model, loader, device, noise_spec: dict | None) -> float:
    model.eval()
    torch.manual_seed(TEST_NOISE_SEED)
    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            if noise_spec is not None:
                xb = hbc_noise(xb, **noise_spec)
            correct += (model(xb).argmax(1) == yb).sum().item()
            total += yb.size(0)
    return correct / total


def _build_sweep() -> list[tuple[str, dict | None]]:
    points = [("clean", None)]
    for snr_db in (20.0, 16.0, 12.0, 8.0, 4.0):
        points.append((f"{snr_db:g}dB", dict(qbits=5, qrange=4.0,
                                             sigma_ratio=snr_db_to_sigma_ratio(snr_db))))
    points.append(("8dB+突发", dict(qbits=5, qrange=4.0,
                                   sigma_ratio=snr_db_to_sigma_ratio(8.0),
                                   burst_sigma_ratio=0.3, burst_len=16, burst_prob=0.005)))
    return points


def main() -> int:
    ap = argparse.ArgumentParser(description="剪枝版 LIF-EEG-TCNet SNR 扫描评估")
    ap.add_argument("--ckpts", type=str, required=True, help="逗号分隔权重路径")
    ap.add_argument("--names", type=str, default="", help="逗号分隔展示名")
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    ckpt_paths = [Path(p.strip()) for p in args.ckpts.split(",") if p.strip()]
    ckpt_paths = [p if p.is_absolute() else ALGO_DIR / p for p in ckpt_paths]
    names = [s.strip() for s in args.names.split(",") if s.strip()] if args.names else []
    names = names + [p.stem for p in ckpt_paths[len(names):]]

    device = _pick_device()
    base_eval = EEGDataset(DATA_4CLASS / "eval.npz")
    sweep = _build_sweep()
    print(f"设备 : {device} | eval : {len(base_eval)} 试次 | 噪声种子 : {TEST_NOISE_SEED}")

    per_model = {}
    for name, path in zip(names, ckpt_paths):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        keep_idx = ckpt.get("channels", list(range(ckpt["model_args"]["n_channels"])))
        model = EEGTCNetLIF(**ckpt["model_args"], **ckpt.get("lif_hparams", {}))
        model.load_state_dict(ckpt["state_dict"], strict=True)
        model = model.to(device)
        model.eval()
        loader = DataLoader(ChannelSliceDataset(base_eval, keep_idx),
                            batch_size=args.batch_size, shuffle=False)
        row = {"ckpt": str(path), "n_channels": len(keep_idx),
               "channels": keep_idx,
               "channel_names": ckpt.get("channel_names", []),
               "final_test_clean": round(ckpt.get("final_test", -1.0), 4),
               "n_params": ckpt.get("n_params", -1)}
        for label, spec in sweep:
            acc = _eval_noise(model, loader, device, spec)
            row[label] = round(acc, 4)
        per_model[name] = row
        print(f"[{name}] {len(keep_idx)}ch 评估完成（clean={row['clean']*100:.2f}% "
              f"8dB={row['8dB']*100:.2f}%）")

    print("\n" + "=" * 76)
    print("SNR 扫描对比表（final test @ eval，cross-session，2592 试次，同一份噪声）")
    print("=" * 76)
    header = f"{'测试域':<10}" + "".join(f"{n:>13}" for n in names)
    print(header)
    for label, _ in sweep:
        cells = "".join(f"{per_model[n][label]*100:>12.2f}%" for n in names)
        print(f"{label:<10}{cells}")

    out_dir = ALGO_DIR / "channel_prune"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / "snr_sweep_pruned.json"
    out.write_text(json.dumps({"meta": {"model": "LIF-EEG-TCNet 剪枝版",
                                        "test_noise_seed": TEST_NOISE_SEED},
                               "sweep_points": [l for l, _ in sweep],
                               "per_model": per_model},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n✓ 结果已写入 {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
