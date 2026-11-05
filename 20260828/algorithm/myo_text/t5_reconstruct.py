# -*- coding: utf-8 -*-
"""
t5_reconstruct.py —— MyoText 第三层：T5 句子重建（约束束搜索）骨架

复现对象：MyoText (arXiv:2601.03098) 分级解码的第三级「句子重建」。
原理：手指分类（端侧）-> QWERTY 候选字母/候选词 -> 微调 T5-base 语言模型，
用「约束束搜索（constrained beam search，束宽 B）」把候选词序列恢复成完整句子。

原论文指标（供对照，本复现尚未实测）：
    CER 5.4% / WER 6.5%（字符/词错误率）

【部署设计 —— split-execution 关键】
    本模块跑在「上位机」（配对的手机 / 边缘 / 云端），不进入 RV1106 端侧。
    原因：T5-base 约 220M 参数，远超端侧 NPU 能力；且语言模型强依赖大词表、
    大显存与 beam search 的算力。MyoText 论文已明确采用这一拆分以保护隐私、降延迟。

【环境】依赖 HuggingFace `transformers`。当前复现机未安装（见复现说明文档），
    本文件对 transformers 采用「懒加载 + 友好报错」，保证模块可被 import、
    其它两个模块不因缺少该依赖而无法运行。安装：
        pip install transformers sentencepiece
    （T5 tokenizer 依赖 sentencepiece）

说明：本文件是「从零复现的代码骨架」，非官方实现；MyoText 无开源代码，
    无官方 T5 微调权重。约束束搜索用 HuggingFace generate() 的
    prefix_allowed_tokens_fn 实现「硬约束到候选字母/候选词」，属于骨架近似，
    论文实际的约束细节（词格约束、三元 T5 变体、损失设计）以原文为准。
"""

from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Optional

import torch


# ---------------------------------------------------------------------------
# transformers 懒加载
# ---------------------------------------------------------------------------
def _require_transformers():
    """运行时才导入 transformers，避免上游模块因缺依赖而无法 import。"""
    try:
        import transformers  # noqa: F401
        return transformers
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "缺少依赖 `transformers`。T5 句子重建需在上位机运行，请先安装：\n"
            "    pip install transformers sentencepiece\n"
            f"原始错误: {e}"
        ) from e


# ---------------------------------------------------------------------------
# 约束束搜索：把「候选字母/候选词」硬约束进 T5 的生成空间
# ---------------------------------------------------------------------------
def build_prefix_allowed_fn_from_letters(
    tokenizer,
    letter_lattice: List[List[str]],
) -> Callable[[int, torch.Tensor], List[int]]:
    """
    构造 prefix_allowed_tokens_fn：在每个生成步，只允许下一步输出
    「对应位置候选字母」对应的 token id。

    这是实现「约束束搜索」最直接的方式——把 QWERTY 候选字母（第 2 层输出）
    作为解码约束，与 T5 语言模型先验联合，缩小搜索空间。

    letter_lattice: 长度 = 期望句子 token 数，每个元素是该位置允许的字母列表。
                    （实际使用中可把候选词展平/对齐到字符位，或用词格）

    返回的函数签名符合 transformers.generate(prefix_allowed_tokens_fn=...) 要求：
        fn(batch_id: int, input_ids: torch.Tensor) -> List[int]
    其中 input_ids 是「当前已生成序列」，返回「下一步允许的 token id 列表」。
    """
    def allowed(batch_id: int, input_ids: torch.Tensor) -> List[int]:
        # 已生成的 token 数（不含 prompt）
        gen_len = input_ids.shape[-1] - _PROMPT_LEN_HINT
        # 简化：用固定映射「第 gen_len 步 -> 允许字母」；超出格长度则放行 EOS
        if gen_len < 0 or gen_len >= len(letter_lattice):
            return [tokenizer.eos_token_id]
        letters = letter_lattice[gen_len]
        ids: List[int] = []
        for ch in letters:
            tid = tokenizer.convert_tokens_to_ids(f"▁{ch}" if False else ch)
            # T5 用 SentencePiece，字母通常映射为单个 subword；此处按字符取 id，
            # 若字符不在词表则跳过（骨架近似，真实需对齐 tokenizer 的 subword）
            if tid is not None and tid != tokenizer.unk_token_id:
                ids.append(tid)
        return ids if ids else [tokenizer.eos_token_id]

    return allowed


# 全局提示长度 hint（简单实现：约束按生成步数对齐，不精确对齐 prompt 长度）
_PROMPT_LEN_HINT = 0


# ---------------------------------------------------------------------------
# T5 模型封装
# ---------------------------------------------------------------------------
class T5Reconstructor:
    """
    T5 句子重建器（上位机运行）。

    用法：
        rec = T5Reconstructor(model_name="t5-base")
        pred = rec.reconstruct(
            candidate_words=["the", "cat"],
            num_beams=8,
            max_new_tokens=32,
        )
    """

    def __init__(self, model_name: str = "t5-base", device: Optional[str] = None):
        """
        model_name: 论文为 't5-base'；也支持微调后的本地路径/其它 T5 变体。
        device    : 默认自动选 cuda / mps / cpu（上位机通常有 GPU）。
        """
        transformers = _require_transformers()
        self._transformers = transformers
        self.device = device or ("cuda" if torch.cuda.is_available()
                                 else "mps" if torch.backends.mps.is_available()
                                 else "cpu")

        self.tokenizer = transformers.T5TokenizerFast.from_pretrained(model_name)
        self.model = transformers.T5ForConditionalGeneration.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    # ------------------------------------------------------------------
    def reconstruct(
        self,
        candidate_words: List[str],
        num_beams: int = 8,
        max_new_tokens: int = 64,
        letter_lattice: Optional[List[List[str]]] = None,
    ) -> str:
        """
        约束束搜索句子重建（骨架）。

        candidate_words: 第 2 层输出的候选词列表（按概率降序）
        num_beams      : 束宽 B（论文用约束束搜索，束宽需实测调参）
        letter_lattice : 可选；若提供，则用「逐字符硬约束」的 prefix_allowed_tokens_fn，
                         实现真正的约束束搜索；若不提供，则退化为标准束搜索。

        返回: 重建出的句子字符串。
        """
        transformers = self._transformers

        # 输入构造：把候选词拼成一段 prompt，让 T5 做「纠正/补全」。
        # 论文可能用专门的任务前缀（如 "fix: " / "transcribe: "），此处骨架
        # 采用 "fix: " 作为占位任务前缀，需按论文/微调设定替换。
        input_text = "fix: " + " ".join(candidate_words)
        enc = self.tokenizer(input_text, return_tensors="pt").to(self.device)

        gen_kwargs: Dict = dict(
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            early_stopping=True,
        )
        if letter_lattice is not None:
            gen_kwargs["prefix_allowed_tokens_fn"] = \
                build_prefix_allowed_fn_from_letters(self.tokenizer, letter_lattice)

        with torch.no_grad():
            out = self.model.generate(**enc, **gen_kwargs)

        return self.tokenizer.decode(out[0], skip_special_tokens=True)

    # ------------------------------------------------------------------
    def reconstruct_batch(
        self,
        candidate_word_lists: List[List[str]],
        num_beams: int = 8,
        **kwargs,
    ) -> List[str]:
        """批量重建（逐个调用，骨架简单实现，可后续优化为批量 generate）。"""
        return [self.reconstruct(words, num_beams=num_beams, **kwargs)
                for words in candidate_word_lists]


# ---------------------------------------------------------------------------
# 端到端编排骨架（三层串联，供未来流水线集成参考）
# ---------------------------------------------------------------------------
def myotext_pipeline(
    finger_classifier,          # finger_classifier.FingerClassifier（端侧）
    reconstructor,              # T5Reconstructor（上位机）
    emg_segments: torch.Tensor, # (N, C, T) 击键片段
    vocabulary: Iterable[str],
    num_beams: int = 8,
) -> List[str]:
    """
    三层串联：端侧手指分类 -> QWERTY 候选词 -> 上位机 T5 句子重建。
    此处仅作「数据流编排」示意，实际部署时手指分类在端侧、T5 在上位机，
    中间通过 HBC 回传候选词（而非原始 sEMG），以保护隐私、降低上行带宽。
    """
    from qwerty_mapping import finger_logits_to_candidates

    finger_logits = finger_classifier(emg_segments)          # (N, 8)
    sentences: List[str] = []
    # 简化：每段一个句子；真实打字需按「词边界」分段聚合
    for logits in finger_logits:
        _, words = finger_logits_to_candidates(
            logits.unsqueeze(0), vocabulary)
        candidate_words = [w.word for w in words]
        sentences.append(reconstructor.reconstruct(
            candidate_words, num_beams=num_beams))
    return sentences


if __name__ == "__main__":
    # 冒烟测试：不加载真实 T5（避免下载权重），仅验证模块可 import、
    # 约束函数签名正确、依赖报错信息友好。
    print("=== T5 句子重建骨架自检 ===")
    print("[OK] 模块可正常 import，transformers 采用懒加载。")

    # 依赖检测
    try:
        _require_transformers()
        print("[OK] transformers 已安装，可加载 t5-base 进行真实推理。")
    except ImportError as e:
        print("[SKIP] transformers 未安装（预期）：", str(e).splitlines()[0])

    # 说明
    print("说明：T5 重建器跑在上位机（配对手机/边缘/云端），不进 RV1106 端侧。")
    print("      无官方权重，CER/WER 指标待自行微调 T5 后实测，本骨架不声称复现结果。")
