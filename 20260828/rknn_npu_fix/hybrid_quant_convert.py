#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hybrid_quant_convert.py — RKNN 混合量化（hybrid quantization）转换脚本骨架

⚠ 平台硬边界（务必先读，与 convert_rknn.py 一致）
--------------------------------------------------
  rknn-toolkit2 **只能在 Linux x86_64 上运行**（官方只发 x86_64 Linux 轮子）：
    - macOS（本项目所在机器）—— 不支持，本脚本**无法在 Mac 上执行**，仅作为骨架。
    - Windows —— 请用 WSL2。
  本脚本的每一行真正执行都发生在 Linux x86 主机上；Mac 上只能读、不能跑。

  rv1106 的混合精度边界（已实测/已知）：
    - rv1106 NPU 计算单元**只支持 INT8**。混合 dtype（w8a16 / int16 / FP16）在 build 时
      均报 "not support"（已有结论，见 convert_rknn.py 注释）。
    - 混合量化（hybrid quantization）与混合 dtype 是**两回事**：混合量化是把**选定的层
      保持 FP32**（其余层 INT8）。FP32 层在 rv1106 上只能回退到 CPU 执行（NPU 跑不了 FP32）。
      因此混合量化在 rv1106 上「是否被 rknn-lite2 支持」「FP32 层 CPU 回退后延迟是否可接受」
      —— 都**尚未验证**，需要 Linux x86 + 真板子实测。不要预设它能成。

本脚本做什么
------------
  1. config(target_platform='rv1106') + load_onnx（同 convert_rknn.py）。
  2. hybrid_quantization_step1(dataset=calib) —— 生成「每层量化误差表」，据此判断哪些层
     对 INT8 敏感、值得保持 FP32。
  3. 人工挑选敏感层，填 HYBRID_LAYERS（层名 → np.float32）。
  4. hybrid_quantization_step2(..., custom_hybrid=HYBRID_LAYERS) —— 按混合精度构建并导出 .rknn。
  5. init_runtime 验证（rv1106 无 x86 模拟器，预期跳过）+ export_rknn。

为什么值得做（结合 qat_train.py 的结论）
----------------------------------------
  qat_train.py 已证明：在 PyTorch 对称 int8 fake-quant 层面，该 EEGNet 的 argmax 一致率
  约 98%，**没有复现** NPU 上 42~58% 的崩坏。这说明崩坏根因更可能是 RKNN 工具链对**某个
  特定算子/层**的量化 bug（例如已知的 Gemm 量化 logits[0] 恒定偏移 -44），而非 INT8 精度
  极限。混合量化的价值于是变成**诊断工具**：把疑似 bug 的层（输出层 fc / 含 ELU 负半轴的
  层 / depthwise 层）单独保持 FP32，看崩坏是否消失，从而**定位**问题算子。

⚠ API 版本差异（重要）
---------------------
  hybrid_quantization_step1/step2 的签名在不同 rknn-toolkit2 版本间有差异。跑之前请先：
      python3 -c "from rknn.api import RKNN; print(RKNN.hybrid_quantization_step1.__doc__)"
      python3 -c "from rknn.api import RKNN; print(RKNN.hybrid_quantization_step2.__doc__)"
  以本机安装版本的实际签名为准；本脚本对签名不符做了 try/except 兜底并打印 docstring。

用法（Linux x86 主机上）
------------------------
  python3 hybrid_quant_convert.py                 # 跑 step1 打印误差表 + 若 HYBRID_LAYERS 已填则跑 step2
  python3 hybrid_quant_convert.py --list-only     # 只跑 step1 打印误差表，不构建
  python3 hybrid_quant_convert.py --verify        # 构建后额外跑校准集对比精度

依赖：rknn-toolkit2（仅 Linux x86）、numpy
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# 假设本脚本与 eegnet.onnx / calib_data.txt 放在一起，或通过环境变量指定。
PROJECT_ROOT = Path(__file__).resolve().parent
ONNX_PATH = PROJECT_ROOT / "eegnet.onnx"
RKNN_PATH = PROJECT_ROOT / "eegnet_hybrid.rknn"
CALIB_TXT = PROJECT_ROOT / "calib_data.txt"

# 模型输入形状（8ch 二分类，与 eegnet.onnx 固定形状一致）
INPUT_SHAPE = (1, 1, 8, 500)

# ---------------------------------------------------------------------------
# ★★★ 待人工填写：混合量化要保持 FP32 的层名（来自 step1 打印的误差表）★★★
# ---------------------------------------------------------------------------
# 先跑一次 `python3 hybrid_quant_convert.py --list-only`，阅读 step1 打印的表格
# （每行一个算子，含其 int8 量化误差），把「int8 误差大 / 疑似 bug」的层名填到下面。
# 层名是 RKNN 内部算子名（一般继承 ONNX 节点名，如 conv1、fc、depthwise…）。
# 值用 np.float32 表示「该层保持 FP32，不量化」。
#
# 建议优先怀疑（据 qat_train.py 结论，这些是 NPU 崩坏的头号嫌疑）：
#   - fc（输出层 1x1 Conv，已知 Gemm 量化 bug 换 Conv 规避后仍需验证）
#   - 含 ELU 负半轴的激活层 / depthwise 卷积（逐通道 range 差异大）
#   - pointwise（1x1 升维，激活范围可能较大）
#
# 示例（层名仅示意，以 step1 表为准）：
HYBRID_LAYERS: dict = {
    # 'conv1': np.float32,
    # 'depthwise': np.float32,
    # 'fc': np.float32,
}
# 若留空，脚本只跑 step1 打印误差表并停在这里，不构建 .rknn。
# ---------------------------------------------------------------------------


def _import_rknn():
    """导入 rknn-toolkit2，未安装 / 平台不支持时给出清晰中文提示。"""
    try:
        from rknn.api import RKNN  # noqa: PLC0415
        return RKNN
    except ImportError as exc:
        raise SystemExit(
            "✗ 无法导入 rknn-toolkit2。\n"
            "  原因：rknn-toolkit2 仅在 Linux x86_64 上可用（Mac/Windows 需 WSL2）。\n"
            "  请在一台 Linux x86 主机（WSL2 或云服务器）上执行本脚本：\n"
            "    pip install rknn-toolkit2\n"
            "  版本对照见：https://github.com/airockchip/rknn-toolkit2\n"
            f"  原始错误：{exc}"
        ) from exc


def _run_step1(rknn, calib_txt: Path) -> int:
    """混合量化第一步：生成每层量化误差表（据此挑层）。返回码 0 = 成功。"""
    print("运行 hybrid_quantization_step1（生成每层量化误差表）...")
    try:
        # 常见签名：hybrid_quantization_step1(dataset=...) —— dataset 为校准集清单 txt
        ret = rknn.hybrid_quantization_step1(dataset=str(calib_txt))
    except TypeError as exc:
        # 版本差异兜底：不同版本可能要求 inputs=/model_input=/model_output= 等参数。
        print(f"  ⚠ 签名不符（{exc}）。请以本机实际签名为准：")
        try:
            print(rknn.hybrid_quantization_step1.__doc__)
        except AttributeError:
            pass
        raise
    if ret != 0:
        print(f"  ✗ hybrid_quantization_step1 返回码 {ret}")
    else:
        print("  ✓ step1 完成。请阅读上面打印的误差表，把敏感层填进 HYBRID_LAYERS。")
    return ret


def _run_step2(rknn, calib_txt: Path, custom_hybrid: dict) -> int:
    """混合量化第二步：按 custom_hybrid 构建混合精度模型并导出 .rknn。"""
    print(f"运行 hybrid_quantization_step2（保持 FP32 的层：{list(custom_hybrid.keys())}）...")
    try:
        # 常见签名：hybrid_quantization_step2(model_input, model_output, dataset, custom_hybrid)
        ret = rknn.hybrid_quantization_step2(
            model_input=str(ONNX_PATH),
            model_output=str(RKNN_PATH),
            dataset=str(calib_txt),
            custom_hybrid=custom_hybrid,
        )
    except TypeError as exc:
        print(f"  ⚠ 签名不符（{exc}）。请以本机实际签名为准：")
        try:
            print(rknn.hybrid_quantization_step2.__doc__)
        except AttributeError:
            pass
        raise
    if ret != 0:
        print(f"  ✗ hybrid_quantization_step2 返回码 {ret}")
    return ret


def _verify_accuracy(rknn) -> None:
    """用 rv1106 模拟器跑校准集对比 ONNX（rv1106 无 x86 模拟器，预期跳过）。"""
    if not CALIB_TXT.exists():
        print("  · 未找到 calib_data.txt，跳过精度验证。")
        return
    lines = [ln.strip() for ln in CALIB_TXT.read_text().splitlines() if ln.strip()]
    if not lines:
        print("  · calib_data.txt 为空，跳过。")
        return
    try:
        import onnxruntime as ort  # noqa: PLC0415
        sess = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
        in_name = sess.get_inputs()[0].name
    except Exception as exc:  # noqa: BLE001
        print(f"  · 无法加载 onnxruntime（{exc}），跳过 ONNX 对比。")
        sess = None

    correct = total = 0
    max_err = sum_err = 0.0
    for ln in lines[:100]:
        x = np.load(PROJECT_ROOT / ln).astype(np.float32)
        if x.shape != INPUT_SHAPE:
            x = x.reshape(INPUT_SHAPE)
        out = rknn.inference(inputs=[x])
        npu_logits = np.asarray(out[0]).ravel()
        total += 1
        if sess is not None:
            onnx_logits = sess.run(None, {in_name: x})[0].ravel()
            correct += int(np.argmax(npu_logits) == np.argmax(onnx_logits))
            e = float(np.abs(npu_logits - onnx_logits).max())
            max_err = max(max_err, e)
            sum_err += e
    if sess is not None:
        print(f"  ✓ 模拟器 vs ONNX 分类一致率：{correct}/{total} = {correct / total * 100:.1f}%")
        print(f"     logits 最大误差 {max_err:.4f}，平均误差 {sum_err / total:.4f}")
        print("  （若一致率≈100%，说明混合量化把崩坏救回来了，问题定位到被保持 FP32 的层；")
        print("   若仍≈50%，说明崩坏层不在你选的 HYBRID_LAYERS 里，需继续加层诊断。）")


def main() -> int:
    parser = argparse.ArgumentParser(description="RKNN 混合量化（hybrid quantization，rv1106）")
    parser.add_argument("--list-only", action="store_true",
                        help="只跑 step1 打印误差表，不构建（用于先挑层）")
    parser.add_argument("--verify", action="store_true", help="构建后跑校准集验证精度")
    args = parser.parse_args()

    RKNN = _import_rknn()

    # 前置文件校验（与 convert_rknn.py 相同口径）
    if not ONNX_PATH.exists():
        raise SystemExit(
            f"✗ ONNX 缺失：{ONNX_PATH}\n  请先在能跑 torch 的环境导出 eegnet.onnx 并拷到本机。"
        )
    if not CALIB_TXT.exists():
        raise SystemExit(
            f"✗ 校准集缺失：{CALIB_TXT}\n  请先生成：python3 model/calibration.py，"
            "并把 calib_data/ + calib_data.txt 拷到本机。"
        )

    rknn = RKNN(verbose=True)

    # ---- 1. 配置（同 convert_rknn.py：输入已 z-score，不做额外归一）----
    #   注意：混合量化里 optimization_level 不宜设 3 的激进融合，因为融合会把 Conv+BN
    #   拼成一个算子，导致无法按「层」单独保持 FP32。建议 optimization_level=0 或 1，
    #   保留算子边界，才能精准挑层。这里用 1（轻融合、保留逐层边界）。
    print("配置 RKNN（target=rv1106, optimization_level=1，保留逐层边界以支持混合量化）...")
    rknn.config(mean_values=[[0]], std_values=[[1]], target_platform="rv1106",
                optimization_level=1)

    # ---- 2. 加载 ONNX ----
    print(f"加载 ONNX：{ONNX_PATH}")
    ret = rknn.load_onnx(model=str(ONNX_PATH))
    if ret != 0:
        raise SystemExit(f"✗ load_onnx 失败，返回码 {ret}（可能是 opset 版本不兼容）")

    # ---- 3. 混合量化第一步：误差表 ----
    ret = _run_step1(rknn, CALIB_TXT)
    if ret != 0:
        raise SystemExit(f"✗ hybrid_quantization_step1 失败，返回码 {ret}")

    # ---- 4. 混合量化第二步：构建（需先填 HYBRID_LAYERS）----
    if args.list_only:
        print("\n--list-only：只打印误差表，不构建。填好 HYBRID_LAYERS 后再跑：")
        print("  python3 hybrid_quant_convert.py --verify")
        rknn.release()
        return 0

    if not HYBRID_LAYERS:
        print("\n⚠ HYBRID_LAYERS 为空 —— 未指定要保持 FP32 的层。")
        print("  请根据 step1 打印的误差表，把敏感层填进脚本顶部的 HYBRID_LAYERS，再重跑。")
        print("  （直接跑会得到纯 INT8 结果，等价于 convert_rknn.py，无法诊断崩坏层。）")
        rknn.release()
        return 0

    ret = _run_step2(rknn, CALIB_TXT, HYBRID_LAYERS)
    if ret != 0:
        raise SystemExit(f"✗ hybrid_quantization_step2 失败，返回码 {ret}")

    # ---- 5. 验证精度（rv1106 无 x86 模拟器，预期跳过；真板用 board/npu_audit.c 实测）----
    print("初始化 rv1106 runtime（x86 无模拟器，预期跳过）...")
    try:
        ret = rknn.init_runtime(target="rv1106")
        if ret != 0:
            print(f"  ⚠ init_runtime 返回非 0（{ret}），跳过精度验证")
        elif args.verify:
            _verify_accuracy(rknn)
    except Exception as exc:  # noqa: BLE001 —— rv1106 无 x86 模拟器
        print(f"  ⚠ init_runtime 异常（rv1106 无 x86 模拟器，跳过）：{exc}")

    # ---- 6. 导出 .rknn ----
    print(f"导出 RKNN：{RKNN_PATH}")
    ret = rknn.export_rknn(str(RKNN_PATH))
    if ret != 0:
        raise SystemExit(f"✗ export_rknn 失败，返回码 {ret}")
    print(f"\n✓ 混合量化模型已导出：{RKNN_PATH}（{RKNN_PATH.stat().st_size / 1024:.1f} KB）")
    print("⚠ 注意：rv1106 上 FP32 层需 CPU 回退；板载是否支持 + 延迟是否可接受需实测。")

    rknn.release()
    print("\n下一步：把 .rknn 拷到板子，用 board/infer_eegnet.py 实测 argmax 一致率。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
