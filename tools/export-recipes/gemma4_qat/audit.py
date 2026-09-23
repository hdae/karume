"""据えた容器の固定整数・scale を入力 bytes と突合する。"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path

import torch

from _shared.container_read import read_stored
from karume.emit import FixedQuantizedWeight


def _digest(tensor: torch.Tensor) -> bytes:
    return hashlib.sha256(memoryview(tensor.numpy())).digest()


def assert_fixed_bytes(path: Path, fixed: Mapping[str, FixedQuantizedWeight]) -> None:
    """容器から取り直した payload を入力 bytes と照合する。f32 展開はしない。

    piece に割れた席も供給計画が行の順に畳むので（{@link _shared.container_read.read_stored}）、
    突合の意味は分割の有無で変わらない。companion scale は同じ供給計画がぶら下げて持つ。
    """
    stored = read_stored(path)
    for key, value in fixed.items():
        entry = stored.get(key)
        if entry is None:
            raise ValueError(f"{path}: 固定重み '{key}' が容器に無い")
        if hashlib.sha256(entry.payload).digest() != _digest(value.packed):
            raise ValueError(f"{path}: 固定 payload {key} が入力 bytes と違う")
        if entry.scale is None:
            raise ValueError(f"{path}: 固定重み '{key}' に companion scale が無い")
        if hashlib.sha256(entry.scale).digest() != _digest(value.scale):
            raise ValueError(f"{path}: 固定 scale {key} が入力 bytes と違う")
