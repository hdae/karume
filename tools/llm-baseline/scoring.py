"""先頭 token と窓の重複を除外する教師強制スコア。"""

from __future__ import annotations

import math
from collections.abc import Iterator

import torch
from torch.nn import functional


def windows(tokens: int, length: int, stride: int) -> Iterator[tuple[int, int, int]]:
    if tokens < 2 or length < 2 or not 1 <= stride < length:
        raise ValueError("tokens/length >= 2、1 <= stride < length が必要")
    previous = 1
    for start in range(0, tokens - 1, stride):
        end = min(start + length, tokens)
        score_start = max(1, previous - start)
        if score_start >= end - start:
            raise ValueError("予測対象のない窓")
        yield start, end, score_start
        previous = end
        if end == tokens:
            return


def token_nll(logits: torch.Tensor, ids: list[int], score_start: int) -> list[float]:
    if logits.ndim != 3 or logits.shape[0] != 1 or logits.shape[1] != len(ids):
        raise ValueError("全行 logits [1,T,V] が必要")
    if not 1 <= score_start < len(ids):
        raise ValueError("予測する継続がありません")
    scores = logits[0, score_start - 1 : -1].float()
    if not torch.isfinite(scores).all():
        raise ValueError("非有限の logits")
    targets = torch.tensor(ids[score_start:], device=scores.device, dtype=torch.int64)
    return functional.cross_entropy(scores, targets, reduction="none").double().tolist()


def accuracy(correct: int, count: int) -> dict:
    if not 0 <= correct <= count or count <= 0:
        raise ValueError("不正な正答数")
    p = correct / count
    z = 1.959963984540054
    denominator = 1 + z * z / count
    center = (p + z * z / (2 * count)) / denominator
    radius = z * math.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / denominator
    return {
        "correct": correct,
        "count": count,
        "accuracy": p,
        "wilson95": [center - radius, center + radius],
    }
