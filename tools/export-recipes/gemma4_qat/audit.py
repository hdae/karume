"""分割後の固定整数・scale を入力 bytes と突合する。"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

from karume.emit import FixedQuantizedWeight
from karume.shards import parse_piece_key, resolve_shards


def assert_fixed_bytes(path: Path, fixed: Mapping[str, FixedQuantizedWeight]) -> None:
    """piece をファイル順に連結した digest を照合する。f32 展開はしない。"""
    expected = {key: value.packed for key, value in fixed.items()}
    expected.update({f"karume.scale.{key}": value.scale for key, value in fixed.items()})
    digests = {key: hashlib.sha256() for key in expected}
    for shard in resolve_shards(path):
        with shard.open("rb") as stream:
            length = int.from_bytes(stream.read(8), "little")
            header = json.loads(stream.read(length))
            for name, entry in header.items():
                if name == "__metadata__":
                    continue
                piece = parse_piece_key(name)
                key = name if piece is None else piece[0]
                if key not in digests:
                    continue
                begin, end = entry["data_offsets"]
                stream.seek(8 + length + begin)
                left = end - begin
                while left:
                    block = stream.read(min(left, 1 << 20))
                    if not block:
                        raise ValueError(f"{shard}: {name} が途中で終わる")
                    digests[key].update(block)
                    left -= len(block)
    for key, tensor in expected.items():
        if digests[key].digest() != hashlib.sha256(memoryview(tensor.numpy())).digest():
            raise ValueError(f"{path}: 固定 payload {key} が入力 bytes と違う")
