"""Teacher-forced scores that exclude the first token and window overlaps."""

from __future__ import annotations

import math
from collections.abc import Iterator

import torch
from torch.nn import functional


def windows(tokens: int, length: int, stride: int) -> Iterator[tuple[int, int, int]]:
    if tokens < 2 or length < 2 or not 1 <= stride < length:
        raise ValueError("tokens/length >= 2 and 1 <= stride < length are required")
    previous = 1
    for start in range(0, tokens - 1, stride):
        end = min(start + length, tokens)
        score_start = max(1, previous - start)
        if score_start >= end - start:
            raise ValueError("window with nothing to predict")
        yield start, end, score_start
        previous = end
        if end == tokens:
            return


def token_nll(logits: torch.Tensor, ids: list[int], score_start: int) -> list[float]:
    if logits.ndim != 3 or logits.shape[0] != 1 or logits.shape[1] != len(ids):
        raise ValueError("full-row logits [1,T,V] are required")
    if not 1 <= score_start < len(ids):
        raise ValueError("no continuation to predict")
    scores = logits[0, score_start - 1 : -1].float()
    if not torch.isfinite(scores).all():
        raise ValueError("non-finite logits")
    targets = torch.tensor(ids[score_start:], device=scores.device, dtype=torch.int64)
    return functional.cross_entropy(scores, targets, reduction="none").double().tolist()


def accuracy(correct: int, count: int) -> dict:
    if not 0 <= correct <= count or count <= 0:
        raise ValueError("invalid number of correct answers")
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
