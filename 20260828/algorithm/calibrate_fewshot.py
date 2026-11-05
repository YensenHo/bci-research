#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
calibrate_fewshot.py — 在线校准（few-shot）实验：新用户前 N 试次校准，剩余试次测试

产品化场景
----------
预训练权重（T 会话 9 人训练，cross-session eval 63.58%）发给新用户后，
新用户（E 会话）开头录 N 个试次（有标签，屏幕提示动作）做校准，之后开始用。
本脚本模拟：校准集 = eval 前 N 试次（按时间顺序；可选按类别均衡抽样作对比），
测试集 = eval 剩余全部试次。**校准试次绝不进测试集**（索引不相交，断言校验）。

三种方法（对每个 N 全跑）
--------------------------
  a. 无校准基线   : 预训练权重直接测剩余试次（对照；全量 eval 应 ≈ 63.58%）
  b. 协方差对齐   : 校准试次算参考协方差 R 与白化矩阵 R^(-1/2)（EA 风格，
                    He & Wu 2019，实现复用 euclidean_alignment.py），
                    对测试集每个 trial 施加 X_align = R^(-1/2) @ X → 测。
                    （注意：模型训练于原始域，推理期单侧对齐是分布失配的，
                    结果如实记录，报告里解释。）
  c. few-shot 微调: 校准试次上 Adam lr=3e-4 微调全部参数，最多 20 epoch，
                    每 epoch 记录测试集精度。上报两个口径：
                      - acc@10ep  : 固定 10 epoch（先验固定，可部署协议）
                      - best(回溯) : 1..20 epoch 中测试精度最大值
                                    （不可部署的上界，标注清楚）
                    双 seed（42/123）报均值与两次值（MPS 波动带）。
  d. EA+微调(补充) : 方法 b 是"推理期单侧对齐"，对原始域训练的模型是结构性
                    分布失配（白化放大低方差通道，见报告诊断）——这不是 EA 的
                    标准用法。标准 few-shot EA 协议是：校准算 R、R^(-1/2) 同时
                    对齐校准集与测试集，再在对齐后的校准集上微调、测对齐后的
                    测试集。本方法直接检验"EA 在新用户校准场景是主场"的假设，
                    仅 seq 模式（真实场景），双 seed。

类别均衡（可选对比）
--------------------
  seq: 校准 = 纯时间顺序前 N 试次（不引导用户，真实但可能类别不均）
  bal: 校准 = 每类取 N//4（余数给前几类）抽样，类内保持时间顺序
       （引导式校准，如屏幕提示"每类录 10 个"；N=10 → 3/3/2/2）

补充诊断
--------
  - 全量 eval 参考精度（应复现 63.58%）
  - R 的条件数 / 对齐自检（校准集对齐后平均协方差 ≈ I）
  - 校准 R 与全量 eval R 的域间距离（校准 N 个试次能否代表 E 会话）
  - 逐被试 EA 补充：每个被试自己前 N 试次校准、自己剩余试次测试
    （纯"新用户各校各的"视角；仅矩阵运算，无训练）

用法
----
  # 冒烟（小规模）
  env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin \\
      /opt/miniconda3/envs/openmmlab/bin/python calibrate_fewshot.py \\
      --n-list 10 --epochs 3 --seeds 42
  # 正式
  env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin \\
      /opt/miniconda3/envs/openmmlab/bin/python calibrate_fewshot.py \\
      --n-list 10 20 40 --epochs 20 --seeds 42 123

产物（algorithm/）
------------------
  calibrate_fewshot_results.json   全部精度/诊断（报告数据源，唯一事实来源）
  在线校准报告.md                   报告（脚本生成表格 + 人工补叙述）
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ALGO_DIR = Path(__file__).resolve().parent
DATA_4CLASS = ALGO_DIR / "data_4class"
CKPT = ALGO_DIR / "eeg_tcnet_4class_lif_hbc_aug_s42.pth"
EVAL_NPZ = DATA_4CLASS / "eval.npz"

N_SUBJECTS = 9
TRIALS_PER_SUBJECT = 2592 // N_SUBJECTS   # 288
N_CLASSES = 4
CHANCE = 100.0 / N_CLASSES

sys.path.insert(0, str(ALGO_DIR))
from euclidean_alignment import (      # noqa: E402
    align,
    alignment_self_check,
    compute_reference_covariance,
    compute_whitening_matrix,
)
from eeg_tcnet_lif import EEGTCNetLIF   # noqa: E402


def _pick_device(no_cuda: bool) -> torch.device:
    if no_cuda:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(device: torch.device) -> nn.Module:
    """从最优权重重建 LIF-EEG-TCNet（strict=True，结构来自 ckpt 内 model_args）。"""
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    ma = ck["model_args"]
    lh = ck["lif_hparams"]
    model = EEGTCNetLIF(
        n_channels=ma["n_channels"], n_classes=ma["n_classes"],
        input_length=ma["input_length"], F1=ma["F1"], D=ma["D"],
        F2=ma["F2"], FT=ma["FT"], kernel_length=ma["kernel_length"],
        tcn_kernel_size=ma["tcn_kernel_size"], pool1=ma["pool1"],
        pool2=ma["pool2"], readout_mode=ma["readout_mode"],
        dropout=ma["dropout"], activation=ma["activation"],
        threshold=lh["threshold"], tau=lh["tau"],
        timesteps=lh["timesteps"], alpha=lh["alpha"],
    )
    model.load_state_dict(ck["state_dict"], strict=True)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def evaluate(model: nn.Module, X: np.ndarray, y: np.ndarray,
             device: torch.device, batch: int = 64) -> dict:
    """X: (N,1,22,500) → 返回 {acc, correct, total, per_class}。"""
    model.eval()
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    loader = DataLoader(ds, batch_size=batch, shuffle=False)
    correct, total = 0, 0
    per_correct = np.zeros(N_CLASSES, dtype=int)
    per_total = np.zeros(N_CLASSES, dtype=int)
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        pred = logits.argmax(1)
        correct += (pred == yb).sum().item()
        total += xb.size(0)
        for c in range(N_CLASSES):
            m = yb == c
            per_correct[c] += (pred[m] == yb[m]).sum().item()
            per_total[c] += m.sum().item()
    return {
        "acc": correct / total, "correct": int(correct), "total": int(total),
        "per_class": {int(c): (int(per_correct[c]), int(per_total[c]))
                      for c in range(N_CLASSES)},
    }


def build_calibration(n_total: int, N: int, mode: str) -> np.ndarray:
    """返回校准索引数组（int64），mode='seq' 前 N 个；mode='bal' 每类等量（类内保序）。"""
    if mode == "seq":
        return np.arange(N, dtype=np.int64)
    if mode == "bal":
        base = N // N_CLASSES
        rem = N % N_CLASSES
        # 余数分给前 rem 个类别（确定性，文档化；N=10 → 3/3/2/2）
        per = [base + (1 if c < rem else 0) for c in range(N_CLASSES)]
        idx = []
        for c, k in enumerate(per):
            pos = np.where(y_all == c)[0]
            assert len(pos) >= k, f"类别 {c} 只有 {len(pos)} 个试次，无法抽 {k} 个"
            idx.append(pos[:k])
        return np.sort(np.concatenate(idx)).astype(np.int64)
    raise ValueError(f"未知校准模式: {mode!r}")


def run_ea(X_calib: np.ndarray, X_test: np.ndarray,
           device: torch.device, model: nn.Module,
           y_test: np.ndarray) -> dict:
    """EA 风格：校准算 R、R^(-1/2)，对齐测试集 → 测。返回 {acc, diag}。"""
    Xc = X_calib[:, 0]                     # (N,22,500)
    Xt = X_test[:, 0]
    R = compute_reference_covariance(Xc)
    Rinv_sqrt = compute_whitening_matrix(R)
    chk = alignment_self_check(align(Xc, Rinv_sqrt))
    eigvals = np.linalg.eigvalsh(R)
    cond = float(eigvals.max() / max(eigvals.min(), 1e-12))
    # 域间距离：校准 R 与全量 eval R（诊断校准集代表性）
    R_full = compute_reference_covariance(X_full[:, 0])
    dom_dist = float(np.linalg.norm(R - R_full, "fro") / np.linalg.norm(R_full, "fro"))
    Xt_a = align(Xt, Rinv_sqrt)[:, None, :, :].astype(np.float32)
    ev = evaluate(model, Xt_a, y_test, device)
    return {
        "acc": ev["acc"],
        "diag": {
            "R_cond": round(cond, 1),
            "align_frob_rel": round(float(chk["frob_rel"]), 6),
            "align_max_abs": float(chk["max_abs_dev"]),
            "calib_full_R_dist": round(dom_dist, 4),
        },
    }


def run_finetune(model: nn.Module, X_calib: np.ndarray, y_calib: np.ndarray,
                 X_test: np.ndarray, y_test: np.ndarray,
                 device: torch.device, seed: int, lr: float,
                 epochs: int, batch: int, fixed_epoch: int = 10) -> dict:
    """few-shot 微调：校准集上低 LR 微调全部参数，逐 epoch 记测试精度。

    返回：{curve:[每 epoch acc], fixed10: acc@fixed_epoch, best_retro: 1..epochs 最大值}。
    无测试集参与模型选择：fixed10 是先验固定 epoch 的可部署口径；
    best_retro 是回溯上界（不可部署，仅标注参考）。
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    m = copy.deepcopy(model)
    m.train()
    ds = TensorDataset(torch.from_numpy(X_calib), torch.from_numpy(y_calib))
    loader = DataLoader(ds, batch_size=min(batch, len(ds)), shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    optimizer = torch.optim.Adam(m.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    curve, train_accs = [], []
    for ep in range(1, epochs + 1):
        m.train()
        tot_c, tot_n, tot_l = 0, 0, 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = m(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            tot_c += (logits.argmax(1) == yb).sum().item()
            tot_n += xb.size(0)
            tot_l += loss.item() * xb.size(0)
        train_accs.append(tot_c / tot_n)
        curve.append(evaluate(m, X_test, y_test, device)["acc"])
    acc_fixed = curve[fixed_epoch - 1] if fixed_epoch <= epochs else curve[-1]
    best_ep = int(np.argmax(curve)) + 1
    return {
        "curve": [round(v, 6) for v in curve],
        "train_acc_curve": [round(v, 6) for v in train_accs],
        "acc_fixed10": acc_fixed, "best_retro": curve[best_ep - 1],
        "best_epoch": best_ep,
    }


def run_ea_finetune(model: nn.Module, X_calib: np.ndarray, y_calib: np.ndarray,
                    X_test: np.ndarray, y_test: np.ndarray,
                    device: torch.device, seed: int, lr: float,
                    epochs: int, batch: int, fixed_epoch: int = 10) -> dict:
    """EA + few-shot 微调：R_calib^(-1/2) 同时对齐校准+测试，对齐校准上微调，测对齐测试。

    返回 {ea_diag, per_seed:{seed: {fixed10, best_retro, curve, train_acc_curve}}}。
    R 只由校准试次计算（无泄漏）；校准/测试索引与主流程一致（不相交）。
    """
    Xc = X_calib[:, 0]
    Xt = X_test[:, 0]
    R = compute_reference_covariance(Xc)
    Rinv_sqrt = compute_whitening_matrix(R)
    chk = alignment_self_check(align(Xc, Rinv_sqrt))
    eigvals = np.linalg.eigvalsh(R)
    cond = float(eigvals.max() / max(eigvals.min(), 1e-12))
    Xc_a = align(Xc, Rinv_sqrt)[:, None, :, :].astype(np.float32)
    Xt_a = align(Xt, Rinv_sqrt)[:, None, :, :].astype(np.float32)
    per_seed = {}
    for sd in seed if isinstance(seed, (list, tuple)) else [seed]:
        per_seed[sd] = run_finetune(model, Xc_a, y_calib, Xt_a, y_test,
                                    device, sd, lr, epochs, batch, fixed_epoch)
    return {"ea_diag": {"R_cond": round(cond, 1),
                        "align_frob_rel": round(float(chk["frob_rel"]), 6)},
            "per_seed": per_seed}


def run_per_subject_ea(model: nn.Module, X: np.ndarray, y: np.ndarray,
                       device: torch.device, N: int) -> dict:
    """逐被试 EA：每个被试自己前 N 试次校准（R），自己剩余试次测试。"""
    base_accs, ea_accs = [], []
    per_subj = {}
    for s in range(N_SUBJECTS):
        sl = slice(s * TRIALS_PER_SUBJECT, (s + 1) * TRIALS_PER_SUBJECT)
        Xs, ys = X[sl], y[sl]
        cal = np.arange(N)
        test = np.arange(N, len(ys))
        base = evaluate(model, Xs[test], ys[test], device)["acc"]
        r = run_ea(Xs[cal], Xs[test], device, model, ys[test])
        base_accs.append(base)
        ea_accs.append(r["acc"])
        per_subj[s] = {"base": base, "ea": r["acc"]}
    return {
        "per_subject": per_subj,
        "mean_base": float(np.mean(base_accs)),
        "mean_ea": float(np.mean(ea_accs)),
        "mean_gain_pp": float((np.mean(ea_accs) - np.mean(base_accs)) * 100),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="在线校准 few-shot 实验")
    ap.add_argument("--n-list", type=int, nargs="+", default=[10, 20, 40])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--fixed-epoch", type=int, default=10,
                    help="可部署固定微调 epoch 数（先验固定）")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 123])
    ap.add_argument("--no-cuda", action="store_true")
    args = ap.parse_args()

    global y_all, X_full
    with np.load(EVAL_NPZ, allow_pickle=False) as d:
        X_full = np.asarray(d["X"], dtype=np.float32)
        y_all = np.asarray(d["y"], dtype=np.int64)
    assert X_full.shape == (2592, 1, 22, 500) and len(y_all) == 2592
    device = _pick_device(args.no_cuda)
    model = load_model(device)
    print(f"设备      : {device}")
    print(f"预训练权重: {CKPT.name}（brief LIF 配置，9,512 参数）")
    print(f"eval 全集 : {X_full.shape}，四类均衡 "
          f"{np.bincount(y_all).tolist()}，逐被试 {TRIALS_PER_SUBJECT} 试次/人")

    # 参考：全量 eval（应复现 63.58%）
    ref = evaluate(model, X_full, y_all, device)
    print(f"[参考] 无校准全量 eval 精度 : {ref['acc']*100:.2f}%"
          f"（原报告 final_test=63.58%，MPS 复现应接近）")

    results = {"meta": {"ckpt": CKPT.name, "device": str(device),
                        "epochs": args.epochs, "lr": args.lr,
                        "fixed_epoch": args.fixed_epoch,
                        "seeds": args.seeds,
                        "full_eval_acc": ref["acc"],
                        "reference_6358": 0.6358024691358025},
               "N": {}}

    for N in args.n_list:
        print(f"\n{'='*72}\nN = {N}（校准试次数）")
        res_N = {}
        for mode in ("seq", "bal"):
            cal_idx = build_calibration(len(y_all), N, mode)
            test_idx = np.setdiff1d(np.arange(len(y_all)), cal_idx)
            # ── 泄漏红线断言 ──
            assert len(test_idx) == len(y_all) - N, "测试集大小错误"
            assert len(np.intersect1d(cal_idx, test_idx)) == 0, "校准/测试索引重叠！"
            Xc, yc = X_full[cal_idx], y_all[cal_idx]
            Xt, yt = X_full[test_idx], y_all[test_idx]
            dist = {int(c): int((yc == c).sum()) for c in range(N_CLASSES)}
            print(f"\n[{mode}] 校准 {N} 试次 类别分布={dist} | "
                  f"测试 {len(yt)} 试次（泄漏控制 ✓ 索引不相交）")

            # a) 无校准基线（测剩余）
            base = evaluate(model, Xt, yt, device)

            # b) EA 协方差对齐
            ea = run_ea(Xc, Xt, device, model, yt)

            # c) few-shot 微调（双 seed）
            ft = {"per_seed": {}}
            for sd in args.seeds:
                ft["per_seed"][sd] = run_finetune(
                    model, Xc, yc, Xt, yt, device, sd, args.lr,
                    args.epochs, args.batch, args.fixed_epoch)
            ft_accs = [ft["per_seed"][sd]["acc_fixed10"] for sd in args.seeds]
            ft_bests = [ft["per_seed"][sd]["best_retro"] for sd in args.seeds]
            ft["acc_fixed10_mean"] = float(np.mean(ft_accs))
            ft["acc_fixed10_seeds"] = ft_accs
            ft["best_retro_mean"] = float(np.mean(ft_bests))
            ft["best_retro_seeds"] = ft_bests

            gain_base = (ea["acc"] - base["acc"]) * 100
            gain_ft = (ft["acc_fixed10_mean"] - base["acc"]) * 100
            print(f"  基线(无校准)      : {base['acc']*100:.2f}%")
            print(f"  EA 协方差对齐     : {ea['acc']*100:.2f}%"
                  f"（Δ {gain_base:+.2f}pp | R cond={ea['diag']['R_cond']} "
                  f"对齐自检 {ea['diag']['align_frob_rel']*100:.4f}% | "
                  f"校准R-全集R距离 {ea['diag']['calib_full_R_dist']*100:.1f}%）")
            print(f"  微调(fixed{args.fixed_epoch}ep) : "
                  + " / ".join(f"{a*100:.2f}%" for a in ft_accs)
                  + f"（均值 {ft['acc_fixed10_mean']*100:.2f}%，Δ {gain_ft:+.2f}pp）")
            print(f"  微调(best 回溯)   : "
                  + " / ".join(f"{a*100:.2f}%" for a in ft_bests)
                  + f"（均值 {ft['best_retro_mean']*100:.2f}%，"
                  f"best_epoch 各 seed="
                  + "/".join(str(ft["per_seed"][sd]["best_epoch"]) for sd in args.seeds) + "）")

            res_N[mode] = {
                "calib_dist": dist, "n_test": int(len(yt)),
                "baseline": base, "ea": ea, "finetune": ft,
            }

        # d) EA + 微调（仅 seq：R_calib 双侧对齐 + 对齐校准上微调，直接检验 EA 校准主场假设）
        cal_idx = build_calibration(len(y_all), N, "seq")
        test_idx = np.setdiff1d(np.arange(len(y_all)), cal_idx)
        eaft = run_ea_finetune(model, X_full[cal_idx], y_all[cal_idx],
                               X_full[test_idx], y_all[test_idx],
                               device, args.seeds, args.lr, args.epochs,
                               args.batch, args.fixed_epoch)
        eft_accs = [eaft["per_seed"][sd]["acc_fixed10"] for sd in args.seeds]
        eft_bests = [eaft["per_seed"][sd]["best_retro"] for sd in args.seeds]
        print(f"  EA+微调(fixed{args.fixed_epoch}ep) : "
              + " / ".join(f"{a*100:.2f}%" for a in eft_accs)
              + f"（均值 {np.mean(eft_accs)*100:.2f}% | "
              f"R cond={eaft['ea_diag']['R_cond']}）")
        print(f"  EA+微调(best 回溯) : "
              + " / ".join(f"{a*100:.2f}%" for a in eft_bests)
              + f"（均值 {np.mean(eft_bests)*100:.2f}%）")
        res_N["ea_finetune_seq"] = {
            "per_seed": eaft["per_seed"], "ea_diag": eaft["ea_diag"],
            "acc_fixed10_mean": float(np.mean(eft_accs)),
            "acc_fixed10_seeds": eft_accs,
            "best_retro_mean": float(np.mean(eft_bests)),
            "best_retro_seeds": eft_bests,
        }

        # 逐被试 EA（补充，纯新用户视角）
        psea = run_per_subject_ea(model, X_full, y_all, device, N)
        print(f"\n[逐被试 EA 补充] 每人前 {N} 试次自校准、自测剩余："
              f"基线均值 {psea['mean_base']*100:.2f}% → "
              f"EA 均值 {psea['mean_ea']*100:.2f}%（Δ {psea['mean_gain_pp']:+.2f}pp）")
        res_N["per_subject_ea"] = psea
        results["N"][str(N)] = res_N

    out = ALGO_DIR / "calibrate_fewshot_results.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n✓ 结果已保存：{out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
