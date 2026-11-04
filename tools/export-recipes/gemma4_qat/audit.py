"""Compare the fixed integers and scales of a written container with the input bytes."""

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
    """Check the payloads read back from the container against the input bytes. Nothing is
    expanded to f32.

    The supply plan folds tensors split into pieces back in row order too
    ({@link _shared.container_read.read_stored}), so the comparison means the same with or
    without a split. The same supply plan carries the companion scale.
    """
    stored = read_stored(path)
    for key, value in fixed.items():
        entry = stored.get(key)
        if entry is None:
            raise ValueError(f"{path}: fixed weight '{key}' is not in the container")
        if hashlib.sha256(entry.payload).digest() != _digest(value.packed):
            raise ValueError(f"{path}: fixed payload {key} differs from the input bytes")
        if entry.scale is None:
            raise ValueError(f"{path}: fixed weight '{key}' has no companion scale")
        if hashlib.sha256(entry.scale).digest() != _digest(value.scale):
            raise ValueError(f"{path}: fixed scale {key} differs from the input bytes")
