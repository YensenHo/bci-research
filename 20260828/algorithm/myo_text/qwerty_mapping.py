# -*- coding: utf-8 -*-
"""
qwerty_mapping.py —— MyoText 第二层：QWERTY 指法 -> 候选字母 / 候选词构建

复现对象：MyoText (arXiv:2601.03098) 分级解码的第二级「字母池化 + 候选词构建」。
原理：手指分类器只输出「哪个手指按的键」（8 指），而标准 QWERTY 触摸打字里
每个手指固定负责若干按键（home-row 先验：左手 ASDF、右手 HJKL）。
据此把「手指预测（含概率）」翻译成「候选字母（含概率）」，再把连续位置的
候选字母组合成候选词，用于缩小 T5 句子重建的搜索空间。

本模块是纯逻辑层（无神经网络），可跑在任意端侧/上位机，与 T5 解耦。

注意：这是从零复现的代码骨架，非官方实现；字母-手指映射遵循通用 QWERTY
十指打字标准指法，若论文另有自定义映射需以原文为准。
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence, Tuple

import torch


# ---------------------------------------------------------------------------
# 8 指 -> 26 字母 的 QWERTY 标准指法映射（home-row 先验）
# ---------------------------------------------------------------------------
# 手指索引与 finger_classifier.FINGER_NAMES 一致：
#   0=left_pinky  1=left_ring  2=left_middle  3=left_index
#   4=right_index 5=right_middle 6=right_ring 7=right_pinky
FINGER_NAMES: Tuple[str, ...] = (
    "left_pinky", "left_ring", "left_middle", "left_index",
    "right_index", "right_middle", "right_ring", "right_pinky",
)

FINGER_TO_LETTERS: Dict[int, Tuple[str, ...]] = {
    0: ("q", "a", "z"),                    # 左手小指
    1: ("w", "s", "x"),                    # 左手无名指
    2: ("e", "d", "c"),                    # 左手中指
    3: ("r", "t", "f", "g", "v", "b"),     # 左手食指
    4: ("y", "u", "h", "j", "n", "m"),     # 右手食指
    5: ("i", "k", ","),                    # 右手中指
    6: ("o", "l", "."),                    # 右手无名指
    7: ("p", ";", "/"),                    # 右手小指
}

# 反向映射：字母 -> 手指索引（用于把数据集击键标签转成手指标签）
LETTER_TO_FINGER: Dict[str, int] = {
    ch: finger for finger, letters in FINGER_TO_LETTERS.items() for ch in letters
}

# home-row 先验：每个手指的「基准键」（用于先验加权）
HOME_ROW: Dict[int, str] = {
    0: "a", 1: "s", 2: "d", 3: "f",     # 左手 ASDF
    4: "j", 5: "k", 6: "l", 7: ";",     # 右手 HJKL
}

# 归一化字母表（26 字母，供 T5 词表约束时对齐）
ALPHABET: str = "abcdefghijklmnopqrstuvwxyz"


@dataclass
class CandidateLetter:
    """单个候选字母：字母 + 归一化概率 + 是否 home-row 键。"""
    letter: str
    prob: float
    is_home_row: bool = False


def finger_to_candidates(
    finger_idx: int,
    finger_prob: float = 1.0,
    home_row_prior: float = 1.0,
) -> List[CandidateLetter]:
    """
    把一个手指（或其概率质量）展开成候选字母列表。

    finger_idx     : 0..7 手指索引
    finger_prob    : 该手指被预测的概率（来自手指分类器 softmax）
    home_row_prior : home-row 键的相对加权（>1 提升基准键概率，论文的 home-row 先验）

    简化做法：手指概率均匀分给其负责的字母；若开启 home-row 先验，
    则 home-row 键权重乘 home_row_prior 后再归一化。这是骨架实现，
    论文实际可能用「手指内字母的条件分布（基于语料统计）」。
    """
    letters = FINGER_TO_LETTERS.get(finger_idx)
    if letters is None:
        return []

    weights = []
    for ch in letters:
        w = home_row_prior if HOME_ROW[finger_idx] == ch else 1.0
        weights.append(w)

    total = sum(weights)
    return [
        CandidateLetter(
            letter=ch,
            prob=finger_prob * (w / total),
            is_home_row=(HOME_ROW[finger_idx] == ch),
        )
        for ch, w in zip(letters, weights)
    ]


def finger_probs_to_letter_probs(
    finger_probs: torch.Tensor,
    home_row_prior: float = 1.0,
) -> Dict[str, float]:
    """
    把 8 指 softmax 分布 (8,) 转成 26 字母（+ 少量符号）的字母概率分布。

    返回 dict: letter -> prob。用于后续候选词构建或直接 argmax 取字母。
    """
    letter_probs: Dict[str, float] = {}
    for finger_idx in range(finger_probs.numel()):
        p = float(finger_probs[finger_idx])
        if p <= 0.0:
            continue
        for cand in finger_to_candidates(finger_idx, p, home_row_prior):
            letter_probs[cand.letter] = letter_probs.get(cand.letter, 0.0) + cand.prob
    return letter_probs


def decode_finger_sequence_to_letters(
    finger_preds: Sequence[int],
    home_row_prior: float = 1.0,
) -> List[List[CandidateLetter]]:
    """
    把一列手指预测（长度 = 击键数）展开成「每个位置一列候选字母」的格状结构
    （letter lattice）。这是送入候选词构建 / T5 约束搜索的中间表示。

    finger_preds: 例如 [3, 2, 5, ...]（长度 = 击键数）
    返回: 每个位置的候选字母列表（按概率降序）
    """
    lattice: List[List[CandidateLetter]] = []
    for fp in finger_preds:
        cands = finger_to_candidates(fp, finger_prob=1.0, home_row_prior=home_row_prior)
        cands.sort(key=lambda c: c.prob, reverse=True)
        lattice.append(cands)
    return lattice


# ---------------------------------------------------------------------------
# 候选词构建（Letter Pooling / Candidate Word Construction）
# ---------------------------------------------------------------------------
@dataclass
class CandidateWord:
    """候选词：字符串 + 累积概率（由各位置候选字母概率连乘得到）。"""
    word: str
    prob: float = 0.0


def build_candidate_words(
    lattice: List[List[CandidateLetter]],
    vocabulary: Iterable[str],
    beam_width: int = 50,
    min_len: int = 1,
    max_len: int = 20,
) -> List[CandidateWord]:
    """
    把字母格状结构组合成「候选词」，用词表过滤 + 束剪枝。

    lattice    : 每个击键位置的候选字母列表
    vocabulary : 词表（英文词，小写）；不在词表中的组合被丢弃
    beam_width : 每个长度保留的最大候选数（束宽）
    min/max_len: 候选词长度范围

    骨架实现采用「按位置扩展开的字典树」思路，复杂度可控；论文实际
    可能用更精细的 prefix-trie + 语言模型打分。此处的词表过滤是
    从零复现的合理近似，不作为复现结果声称。
    """
    vocab_set = set(w.strip().lower() for w in vocabulary if w.strip())
    if not vocab_set:
        # 无词表时退化为全组合（不建议，仅兜底）
        vocab_set = None

    # 每个位置的概率 -> log 概率，避免连乘下溢
    def _key(cands: List[CandidateLetter]) -> List[Tuple[str, float]]:
        return [(c.letter, c.prob) for c in cands]

    # 用束搜索逐位置扩展：beam 存 (prefix, log_prob)
    beams: List[Tuple[str, float]] = [("", 0.0)]
    for pos_cands in lattice:
        new_beams: Dict[str, float] = {}
        for prefix, lp in beams:
            for c in pos_cands:
                nxt = prefix + c.letter
                nlp = lp + _log(c.prob)
                # 词表约束：只有当前缀可能是词表里某个词的前缀时才保留
                if vocab_set is not None and not _is_prefix_of_vocab(nxt, vocab_set):
                    continue
                if nlp > new_beams.get(nxt, float("-inf")):
                    new_beams[nxt] = nlp
        # 束剪枝
        beams = sorted(new_beams.items(), key=lambda kv: kv[1], reverse=True)[:beam_width]
        if not beams:
            break

    results: List[CandidateWord] = []
    for word, lp in beams:
        if min_len <= len(word) <= max_len and (vocab_set is None or word in vocab_set):
            results.append(CandidateWord(word=word, prob=math.exp(lp)))
    results.sort(key=lambda w: w.prob, reverse=True)
    return results[:beam_width]


def _log(p: float) -> float:
    return math.log(max(p, 1e-12))


def _is_prefix_of_vocab(prefix: str, vocab_set: set) -> bool:
    """前缀是否为词表中某个词的公共前缀（字典树剪枝的近似）。"""
    if prefix in vocab_set:
        return True
    # 近似：仅检查是否存在以 prefix 开头的词（对大词表可用 trie 加速）
    return any(w.startswith(prefix) for w in vocab_set)


# ---------------------------------------------------------------------------
# 顶层接口：手指概率序列 -> 候选字母序列 + 候选词
# ---------------------------------------------------------------------------
def finger_logits_to_candidates(
    finger_logits: torch.Tensor,  # (T, 8) 每击键位置的手指 logits
    vocabulary: Iterable[str],
    top_k_fingers: int = 1,
    home_row_prior: float = 1.0,
    beam_width: int = 50,
) -> Tuple[List[CandidateLetter], List[CandidateWord]]:
    """
    便捷入口：手指分类器输出 (T, 8) logits -> 每位置 top-k 手指 -> 字母格
    -> 候选词。返回 (每个位置最可能的候选字母, 候选词列表)。
    """
    finger_probs = torch.softmax(finger_logits, dim=-1)          # (T, 8)
    lattice: List[List[CandidateLetter]] = []
    best_letters: List[CandidateLetter] = []

    for t in range(finger_probs.shape[0]):
        probs = finger_probs[t]
        # 取 top-k 手指，合并其候选字母
        cands: List[CandidateLetter] = []
        topk = torch.topk(probs, min(top_k_fingers, probs.numel()))
        for idx, p in zip(topk.indices.tolist(), topk.values.tolist()):
            cands.extend(finger_to_candidates(int(idx), float(p), home_row_prior))
        cands.sort(key=lambda c: c.prob, reverse=True)
        lattice.append(cands)
        best_letters.append(cands[0] if cands else CandidateLetter("", 0.0))

    words = build_candidate_words(lattice, vocabulary, beam_width=beam_width)
    return best_letters, words


if __name__ == "__main__":
    print("=== QWERTY 指法映射冒烟测试 ===")
    print("8 指 -> 字母映射：")
    for f, letters in FINGER_TO_LETTERS.items():
        print(f"  {f} {FINGER_NAMES[f]:<13} -> {''.join(letters)}")

    # 反向映射正确性：26 字母都应能找到手指
    assert len(LETTER_TO_FINGER) >= 26, "字母-手指反向映射不完整"
    missing = [ch for ch in ALPHABET if ch not in LETTER_TO_FINGER]
    print(f"[OK] 反向映射覆盖 26 字母，缺失: {missing or '无'}")

    # 手指概率 -> 字母概率
    fp = torch.zeros(8)
    fp[3] = 0.8  # 左手食指
    fp[2] = 0.2  # 左手中指
    lp = finger_probs_to_letter_probs(fp)
    top = sorted(lp.items(), key=lambda kv: kv[1], reverse=True)[:3]
    print("[OK] 字母概率 top3 =", top)

    # 候选词构建（小词表）
    # 例：手指序列 [2, 0, 3] = 左中指(c) + 左小指(a) + 左食指(r/t) -> 命中 "car"/"cat"
    vocab = {"the", "and", "cat", "dog", "that", "this", "car"}
    finger_preds = [2, 0, 3]
    lattice = decode_finger_sequence_to_letters(finger_preds)
    words = build_candidate_words(lattice, vocab, beam_width=20)
    print("[OK] 候选词 =", [(w.word, round(w.prob, 4)) for w in words[:10]])
    print("注意：以上均为骨架演示，非复现结果。")
