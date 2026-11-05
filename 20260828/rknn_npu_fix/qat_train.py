#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qat_train.py — QAT 量化感知训练消融实验（验证「QAT 能否救回 NPU INT8 精度」）

背景
----
RV1106 板载 EEGNet 用 RKNN INT8 量化后 argmax 一致率仅 42~58%（接近随机），
已证伪「PTQ + 44*scale workaround」。本脚本在 **PyTorch 侧**（不依赖 RKNN）用
fake-quant（伪量化 + 直通估计 STE）模拟 NPU 的 INT8 量化噪声，回答一个前置问题：

    如果连「训练时暴露在 INT8 量化噪声下」的 QAT 都救不回精度，
    那 NPU 线就没有继续调 RKNN 的必要 —— 问题在工具链/硬件的固有限制。

三种模式（--mode 控制）：
  baseline  FP32 浮点模型（无量化）→ 上界基准
  ptq      加载 FP32 权重后**直接**插入 fake-quant（不重训）→ 复现 NPU PTQ 崩坏
  qat      在 fake-quant 噪声下微调若干 epoch → QAT 的最终答案
  all      依次跑 baseline → ptq → qat（默认）

模拟口径（与 RKNN INT8 对齐）
---------------------------
  - 权重量化：每个 Conv2d 的 weight 做「对称 per-tensor / per-channel int8 量化」，
    通过 torch.nn.utils.parametrize 把 fake-quant 包在 weight 上（forward 量化、backward STE 直通）。
  - 激活量化：每个 Conv2d 的**输入**做「对称 per-tensor int8 量化」，通过 forward_pre_hook 注入。
    —— 等价于量化「上一层算子（BN/ELU/pool）的输出」，即 RKNN 逐算子边界量化激活的口径。
  - 量化区间：对称 int8，scale = max(|x|) / 127，zero_point = 0。
  - 默认 per-tensor 权重（对应「单-scale INT8 装不下」的叙事）；--per-channel-weight 可开
    per-channel 权重量化，用于诊断「逐通道 scale 是否能救回」（RKNN conv 常支持逐通道）。

诚实声明
--------
  本脚本只回答「PyTorch fake-quant 层面 QAT 能否救回精度」。它**不能**证明 RKNN 转出来一定好：
  - RKNN 的量化细节（per-tensor/per-channel 取舍、Conv+BN 折叠、ELU 负半轴处理）与这里的
    fake-quant 简化版不完全一致，只是**一阶近似**。
  - 最终结论仍需在 Linux x86 上用 rknn-toolkit2 实测（见 hybrid_quant_convert.py）。
  若 fake-quant 层面 QAT 也救不回，则如实写「工具链固有限制，NPU 线维持关闭」。

用法
----
  # 完整消融（真实数据，微调 15 epoch）
  env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin \
      /opt/miniconda3/envs/openmmlab/bin/python qat_train.py

  # 只跑 PTQ（不重训，最快复现崩坏）
  env -i ... python qat_train.py --mode ptq

  # 无数据时用随机 dummy 数据验证训练循环能跑
  env -i ... python qat_train.py --dummy --epochs 3

依赖：torch（>=2.0，需 torch.nn.utils.parametrize）、numpy
数据：~/BCI/luckfox-deploy/data/processed/{train,eval}.npz（可用 BCI_* 环境变量覆盖）
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
# 路径与导入（只读引用 luckfox-deploy，不改动原代码）
# ---------------------------------------------------------------------------
LUCKFOX = Path(os.environ.get("BCI_LUCKFOX", str(Path.home() / "BCI" / "luckfox-deploy")))
MODEL_DIR = LUCKFOX / "model"
DATA_DIR = Path(os.environ.get("BCI_DATA_PROCESSED", str(LUCKFOX / "data" / "processed")))

sys.path.insert(0, str(MODEL_DIR))          # 供 from eegnet import EEGNet
sys.path.insert(0, str(LUCKFOX / "data"))   # 供 from dataset import EEGDataset

from eegnet import EEGNet          # noqa: E402  （只读引用）
from dataset import EEGDataset     # noqa: E402  （只读引用）

OUT_DIR = Path(__file__).resolve().parent
RESULTS_JSON = OUT_DIR / "qat_results.json"

# 模型/数据固定规格（与 eegnet.py / preprocess.py 一致）
N_CHANNELS, N_CLASSES, INPUT_LENGTH = 8, 2, 500
MODEL_ARGS = dict(F1=8, D=2, F2=16, kernel_length=63, dropout=0.5)


# ===========================================================================
# 1. fake-quant 核心：对称 int8 量化 + 直通估计（STE）
# ===========================================================================
class _FakeQuantSTE(torch.autograd.Function):
    """对称 int8 伪量化：forward 量化→反量化，backward 直通（STE）。

    这是 torch.ao.quantization.FakeQuantize 的等价自实现，写成显式 autograd
    Function 是为了让「梯度直通」一目了然，并避免依赖 torch.round/clamp 的
    隐式梯度行为在不同 torch 版本上的差异。
    """

    @staticmethod
    def forward(ctx, x, qmin, qmax, per_channel):  # noqa: D102
        # scale：per_channel 时对「输出通道维（dim 0）之外的所有维」做 max，
        #        即每个卷积输出通道一个 scale；否则整个张量一个 scale。
        if per_channel:
            dims = tuple(range(1, x.dim()))
            scale = x.detach().abs().amax(dim=dims, keepdim=True).clamp(min=1e-8) / qmax
        else:
            scale = x.detach().abs().max().clamp(min=1e-8) / qmax
        q = torch.round(x / scale).clamp(qmin, qmax)
        return q * scale

    @staticmethod
    def backward(ctx, grad_output):  # noqa: D102
        # STE：梯度原样直通（scale/qmin/qmax/per_channel 均不需要梯度）。
        return grad_output, None, None, None


def _bits_range(bits: int) -> tuple[int, int]:
    """int 对称量化区间：bits=8 → (-128, 127)。"""
    return -2 ** (bits - 1), 2 ** (bits - 1) - 1


class _WeightFakeQuant(nn.Module):
    """权重伪量化模块，挂到 Conv2d.weight 上作 parametrization。

    forward 返回 fake-quant 后的 weight（形状不变），backward 走 STE 更新原始权重。
    """

    def __init__(self, bits: int = 8, per_channel: bool = False):
        super().__init__()
        self.qmin, self.qmax = _bits_range(bits)
        self.per_channel = per_channel

    def forward(self, w: torch.Tensor) -> torch.Tensor:  # noqa: D102
        return _FakeQuantSTE.apply(w, self.qmin, self.qmax, self.per_channel)


def _make_activation_quant(bits: int):
    """返回一个 forward_pre_hook：量化进入该卷积的输入激活（per-tensor 对称 int8）。"""
    qmin, qmax = _bits_range(bits)

    def hook(module, args):
        x = args[0]
        if x is None:
            return args
        # 注意：返回 tuple 以替换 forward 的输入（只动第一个输入张量）。
        return (_FakeQuantSTE.apply(x, qmin, qmax, False),)

    return hook


def _make_output_quant(bits: int):
    """返回一个 forward_hook：量化该模块的输出张量（per-tensor 对称 int8）。

    用于「量化每个算子的输出」的更激进口径（--quant-every-op），
    更贴近 RKNN 逐算子边界量化激活的行为（BN/ELU/pool 的输出都会各自量化）。
    """
    qmin, qmax = _bits_range(bits)

    def hook(module, inp, out):
        return _FakeQuantSTE.apply(out, qmin, qmax, False)

    return hook


def build_qat_model(base: EEGNet, bits: int = 8,
                    per_channel_weight: bool = False,
                    quant_every_op: bool = False) -> EEGNet:
    """把基础 EEGNet 改造成 QAT 版（原地）：所有 Conv2d 权重 + 激活都伪量化。

    注意：必须在加载 FP32 state_dict **之后**调用（parametrization 会把当前
    weight 值收编为「原始参数」，之后 forward 里 conv.weight 就是量化后的值）。

    两种激活量化口径：
      quant_every_op=False（默认「算子边界」口径）：量化每个 Conv 的**输入**，
          等价于量化「上一层算子（BN/ELU/pool）的输出」，覆盖 输入/BN 出/池化出 等关键边界。
      quant_every_op=True（「逐算子输出」口径）：额外量化每个 Conv/BN/ELU/Pool 的
          **输出**，即每个中间张量都单独量化一次，最贴近 RKNN 逐算子量化、误差逐层累积。
    """
    for name, module in base.named_modules():
        if isinstance(module, nn.Conv2d):
            # 1) 权重伪量化（parametrization，STE 更新原始权重）
            torch.nn.utils.parametrize.register_parametrization(
                module, "weight", _WeightFakeQuant(bits, per_channel_weight)
            )
            # 2) 输入激活伪量化（forward_pre_hook）
            if not quant_every_op:
                module.register_forward_pre_hook(_make_activation_quant(bits))

    if quant_every_op:
        # 逐算子输出量化口径：网络原始输入（conv1 的输入）+ 每个算子的输出都量化。
        base.conv1.register_forward_pre_hook(_make_activation_quant(bits))
        for module in base.modules():
            if isinstance(module, (nn.Conv2d, nn.BatchNorm2d, nn.ELU, nn.AvgPool2d)):
                module.register_forward_hook(_make_output_quant(bits))
    return base


def freeze_bn(model: nn.Module) -> None:
    """把 BN 固定为 eval 模式（用 running stats），模拟 RKNN 的 Conv+BN 折叠。

    RKNN optimization_level=3 会把 Conv+BN 折叠成单个量化 Conv（BN 用训练收敛的
    running stats）。这里冻结 BN 让 fake-quant 看到的 BN 输出与折叠口径一致。
    """
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()


# ===========================================================================
# 2. 数据加载（真实数据 / dummy 回退）
# ===========================================================================
def build_loaders(batch_size: int, dummy: bool):
    """返回 (train_loader, eval_loader)。dummy=True 时用随机高斯数据自检训练循环。"""
    if dummy:
        rng = np.random.default_rng(42)
        X_tr = rng.standard_normal((256, 1, N_CHANNELS, INPUT_LENGTH)).astype(np.float32)
        y_tr = rng.integers(0, 2, 256).astype(np.int64)
        X_ev = rng.standard_normal((128, 1, N_CHANNELS, INPUT_LENGTH)).astype(np.float32)
        y_ev = rng.integers(0, 2, 128).astype(np.int64)
        train_ds = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
        eval_ds = TensorDataset(torch.from_numpy(X_ev), torch.from_numpy(y_ev))
        print(f"[dummy] 训练 {len(train_ds)} 试次 / 评估 {len(eval_ds)} 试次（随机数据，仅自检）")
    else:
        train_npz = DATA_DIR / "train.npz"
        eval_npz = DATA_DIR / "eval.npz"
        for p in (train_npz, eval_npz):
            if not p.exists():
                raise SystemExit(
                    f"✗ 数据缺失：{p}\n"
                    "  请先跑：python3 data/download_bcic2a.py && python3 data/preprocess.py\n"
                    "  或加 --dummy 用随机数据自检训练循环。"
                )
        train_ds = EEGDataset(train_npz)
        eval_ds = EEGDataset(eval_npz)
        print(f"[真实数据] 训练 {len(train_ds)} 试次 / 评估 {len(eval_ds)} 试次")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    eval_loader = DataLoader(eval_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, eval_loader


# ===========================================================================
# 3. 训练 / 评估
# ===========================================================================
def predict(model: nn.Module, loader: DataLoader) -> np.ndarray:
    """返回全量预测 argmax（numpy int64 数组）。"""
    model.eval()
    preds = []
    with torch.no_grad():
        for xb, _yb in loader:
            out = model(xb)
            preds.append(out.argmax(dim=1).numpy())
    return np.concatenate(preds)


def eval_labels(loader: DataLoader) -> np.ndarray:
    """返回评估集真实标签。"""
    ys = [yb.numpy() for _xb, yb in loader]
    return np.concatenate(ys)


def run_qat_finetune(model: nn.Module, loader: DataLoader, epochs: int,
                     lr: float, freeze_bn_: bool, seed: int) -> None:
    """在 fake-quant 噪声下微调（模型已带 QAT parametrization + hooks）。"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if freeze_bn_:
        freeze_bn(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        total_loss, correct, n = 0.0, 0, 0
        for xb, yb in loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
            correct += (logits.argmax(1) == yb).sum().item()
            n += xb.size(0)
        dt = time.time() - t0
        print(f"  QAT epoch {epoch:3d}/{epochs} | "
              f"loss={total_loss / n:.4f} acc={100.0 * correct / n:5.2f}% | {dt:.1f}s")


def load_fp32_model() -> EEGNet:
    """加载 FP32 预训练权重（eegnet.pth），返回 base EEGNet。"""
    ckpt_path = MODEL_DIR / "eegnet.pth"
    if not ckpt_path.exists():
        raise SystemExit(f"✗ 权重缺失：{ckpt_path}\n  请先在 luckfox-deploy 侧训练出 eegnet.pth。")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    model = EEGNet(n_channels=N_CHANNELS, n_classes=N_CLASSES, input_length=INPUT_LENGTH,
                   **MODEL_ARGS)
    model.load_state_dict(state)
    model.eval()
    return model


def _metrics(preds: np.ndarray, y: np.ndarray, ref: np.ndarray | None):
    """返回 dict：准确率 + （相对参考的）argmax 一致率。"""
    acc = float((preds == y).mean())
    out = {"accuracy": round(acc * 100, 2)}
    if ref is not None:
        out["argmax_agreement_vs_ref"] = round(float((preds == ref).mean()) * 100, 2)
    return out


# ===========================================================================
# 4. 主流程
# ===========================================================================
def main() -> int:
    parser = argparse.ArgumentParser(description="QAT 消融：验证 fake-quant 能否救回 INT8 精度")
    parser.add_argument("--mode", choices=["baseline", "ptq", "qat", "all"], default="all")
    parser.add_argument("--bits", type=int, default=8, help="量化位深（默认 8）")
    parser.add_argument("--per-channel-weight", action="store_true",
                        help="权重量化用 per-channel（逐通道 scale）而非 per-tensor")
    parser.add_argument("--quant-every-op", action="store_true",
                        help="激活量化用「逐算子输出」口径（每个 Conv/BN/ELU/Pool 输出都量化，"
                             "最贴近 RKNN 逐算子量化、误差逐层累积）")
    parser.add_argument("--epochs", type=int, default=15, help="QAT 微调 epoch 数")
    parser.add_argument("--lr", type=float, default=1e-3, help="QAT 微调学习率")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--freeze-bn", action="store_true",
                        help="QAT 微调时冻结 BN（eval 模式，模拟 Conv+BN 折叠口径）")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dummy", action="store_true",
                        help="用随机 dummy 数据自检训练循环（无数据时用）")
    parser.add_argument("--no-cuda", action="store_true", help="强制 CPU（默认即 CPU）")
    args = parser.parse_args()

    # 设备：固定 CPU（MPS 对自定义 autograd.Function 的 STE 支持不稳，模型极小 CPU 足够快）
    device = torch.device("cpu")
    print(f"设备     : {device}")
    print(f"量化位深 : int{args.bits}（权重 {'per-channel' if args.per_channel_weight else 'per-tensor'}）")

    train_loader, eval_loader = build_loaders(args.batch_size, args.dummy)
    y_eval = eval_labels(eval_loader)

    results: dict = {"bits": args.bits,
                     "per_channel_weight": args.per_channel_weight,
                     "quant_every_op": args.quant_every_op,
                     "modes": {}}

    # ---- baseline：FP32 上界 ----
    if args.mode in ("baseline", "all"):
        model = load_fp32_model()
        ref = predict(model, eval_loader)
        results["modes"]["baseline"] = _metrics(ref, y_eval, None)
        results["baseline_acc"] = results["modes"]["baseline"]["accuracy"]
        print(f"\n[baseline FP32] 评估准确率 = {results['modes']['baseline']['accuracy']}%")
    else:
        # 其它模式也需要 FP32 参考预测来算 argmax 一致率
        model = load_fp32_model()
        ref = predict(model, eval_loader)
        print(f"\n[baseline FP32] 评估准确率 = {_metrics(ref, y_eval, None)['accuracy']}%（参考）")

    # ---- ptq：加载 FP32 → 插入 fake-quant（不重训）→ 评估 ----
    if args.mode in ("ptq", "all"):
        model = load_fp32_model()                      # 先加载 FP32 权重
        build_qat_model(model, args.bits, args.per_channel_weight,
                        args.quant_every_op)           # 再插 fake-quant
        model.eval()
        preds = predict(model, eval_loader)
        results["modes"]["ptq"] = _metrics(preds, y_eval, ref)
        print(f"[ptq    fake-quant 不重训] 准确率 = {results['modes']['ptq']['accuracy']}% | "
              f"vs FP32 argmax 一致率 = {results['modes']['ptq']['argmax_agreement_vs_ref']}%")

    # ---- qat：fake-quant 下微调 → 评估 ----
    if args.mode in ("qat", "all"):
        model = load_fp32_model()
        build_qat_model(model, args.bits, args.per_channel_weight,
                        args.quant_every_op)
        print(f"\n开始 QAT 微调（{args.epochs} epoch, lr={args.lr}, "
              f"freeze_bn={args.freeze_bn}）...")
        run_qat_finetune(model, train_loader, args.epochs, args.lr,
                         args.freeze_bn, args.seed)
        preds = predict(model, eval_loader)            # eval 模式（fake-quant 仍生效）
        results["modes"]["qat"] = _metrics(preds, y_eval, ref)
        print(f"[qat    fake-quant 微调后] 准确率 = {results['modes']['qat']['accuracy']}% | "
              f"vs FP32 argmax 一致率 = {results['modes']['qat']['argmax_agreement_vs_ref']}%")

    # ---- 汇总 + 落盘 ----
    print("\n" + "=" * 60)
    print("消融汇总（评估集）")
    for tag, m in results["modes"].items():
        line = f"  {tag:8s} | 准确率 {m['accuracy']:6.2f}%"
        if "argmax_agreement_vs_ref" in m:
            line += f" | vs FP32 argmax 一致率 {m['argmax_agreement_vs_ref']:6.2f}%"
        print(line)
    print("=" * 60)

    # 结论提示（不编造：只根据本次 fake-quant 数字给出方向性判断）
    if "ptq" in results["modes"]:
        ptq_agree = results["modes"]["ptq"].get("argmax_agreement_vs_ref", 0.0)
        if ptq_agree >= 95.0:
            print(f"→ 关键发现：PTQ fake-quant 的 argmax 一致率 = {ptq_agree:.1f}%，"
                  f"**没有复现** NPU 上 42~58% 的崩坏。")
            print("  这说明该模型在「对称 int8 伪量化」层面本就鲁棒（≈98% 一致），")
            print("  NPU 崩坏的根因不是 INT8 精度极限，而更可能是 RKNN 工具链 bug")
            print("  （如已知的 Gemm 量化 logits[0] 恒定偏移 -44），或量化方案/校准集失配。")
            print("  QAT 无法修复工具链 bug —— 结论应如实写：fake-quant 层面无需/无法通过 QAT")
            print("  救回，问题定位到工具链侧；需 Linux x86 上实测定位（见 hybrid_quant_convert.py）。")
        else:
            qat_agree = results["modes"].get("qat", {}).get("argmax_agreement_vs_ref", 0.0)
            print(f"→ PTQ fake-quant 复现了崩坏（一致率 {ptq_agree:.1f}%），"
                  f"QAT 后一致率 {qat_agree:.1f}%。")
            if qat_agree >= 95.0:
                print("  若 QAT 把一致率救回 ≥95%，建议进入 RKNN 实测阶段验证。")
            else:
                print("  若 QAT 仍显著低于 FP32 上界，说明该结构对 INT8 单-scale 量化不鲁棒，")
                print("  结论应如实写为「工具链/硬件固有限制，NPU 线维持关闭」——请勿宣称修复成功。")

    RESULTS_JSON.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")
    print(f"\n结果已落盘：{RESULTS_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
