"""QAT PLE を固定 packed のまま token-major sidecar へ書く（ADR 0097 追記 4）。"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from safetensors.torch import save_file

from gemma4.export_product import (
    PLE_INDEX_FILE,
    PLE_PROBE_FILE,
    PROBE_INPUTS_KEY,
    PROBE_TOKENS_KEY,
    plan_ple_shards,
    ple_probe_tokens,
)
from karume.dist import safetensors_header
from karume.emit import ContainerEntry, container_order, write_container
from karume.shards import SHARD_DATA_CAPACITY, shard_name
from karume.verify import assert_reader_layout

if TYPE_CHECKING:
    from transformers.integrations.gemma_quant import QuantizedEmbedding


def _assert_bytes(path: Path, payloads: dict[str, torch.Tensor]) -> None:
    """書き出しを正規 reader で検査し、全 payload をブロック読みで元の固定 bytes と照合する。"""
    assert_reader_layout(path)
    header = safetensors_header(path)
    with path.open("rb") as stream:
        base = 8 + int.from_bytes(stream.read(8), "little")
        for key, tensor in payloads.items():
            begin, end = header[key]["data_offsets"]
            stream.seek(base + begin)
            digest = hashlib.sha256()
            left = end - begin
            while left:
                block = stream.read(min(left, 1 << 20))
                if not block:
                    raise ValueError(f"{path}: {key} の payload が途中で終わる")
                digest.update(block)
                left -= len(block)
            if digest.digest() != hashlib.sha256(memoryview(tensor.numpy())).digest():
                raise ValueError(f"{path}: {key} が上流の固定 bytes と違う")


def write_ple(
    module: QuantizedEmbedding,
    layers: int,
    dim: int,
    destination: Path,
    *,
    shard_capacity: int = SHARD_DATA_CAPACITY,
) -> dict[str, Any]:
    """固定表と上流モジュールによる散点参照を保存する。destination は staging 側が作る。"""
    bits, tokens = module.num_bits, module.num_embeddings
    if bits not in (2, 4) or layers <= 0 or dim <= 0 or dim % 16:
        raise ValueError("PLE は I2/I4、正の layers、16の倍数の dim が必要")
    factor = 8 // bits
    values, scales = module.embedding_quantized.detach(), module.embedding_scale.detach()
    embed_scale = float(module.scalar_embed_scale)
    if not math.isfinite(embed_scale) or embed_scale <= 0:
        raise ValueError("PLE embedScale は正の有限数が必要")
    if values.device.type != "cpu" or values.dtype != torch.uint8 or not values.is_contiguous():
        raise ValueError("PLE packed values は CPU U8 の連続配置が必要")
    if values.shape != (tokens, layers * dim // factor):
        raise ValueError("PLE packed values の形が tokens/layers/dim と違う")
    if (
        scales.device.type != "cpu"
        or scales.dtype != torch.float32
        or scales.shape != (tokens, layers)
    ):
        raise ValueError("PLE scales は CPU F32 [tokens,layers] が必要")
    if not torch.isfinite(scales).all() or not (scales > 0).all():
        raise ValueError("PLE scales は正の有限数が必要")
    scales = scales.contiguous()
    ranges = plan_ple_shards(tokens, layers * (dim // factor + 4), shard_capacity)
    probe = ple_probe_tokens(tokens, ranges)
    with torch.inference_mode():
        expected = module(torch.tensor([list(probe)], dtype=torch.int64)).reshape(
            1, len(probe), layers, dim
        )
    common = {
        "schema": 2,
        "storage": f"i{bits}",
        "tokens": tokens,
        "layers": layers,
        "dim": dim,
        "embedScale": embed_scale,
    }
    shards = []
    for number, (start, stop) in enumerate(ranges, 1):
        rows = stop - start
        file = shard_name("ple.safetensors", number, len(ranges))
        path = destination / file
        payloads = {"values": values[start:stop], "scales": scales[start:stop]}
        entries = container_order(
            [
                ContainerEntry(
                    "values", f"I{bits}", (rows, layers, dim), payloads["values"].numel()
                ),
                ContainerEntry("scales", "F32", (rows, layers), payloads["scales"].numel() * 4),
            ]
        )
        metadata = {**common, "start": start, "stop": stop}
        write_container(
            path,
            entries,
            {"karume_ple": json.dumps(metadata)},
            lambda entry, payloads=payloads: [memoryview(payloads[entry.name].numpy()).cast("B")],
        )
        _assert_bytes(path, payloads)
        shards.append({"file": file, "start": start, "stop": stop})
    index = {**common, "shards": shards}
    (destination / PLE_INDEX_FILE).write_text(json.dumps(index, indent=2) + "\n")
    save_file(
        {
            PROBE_TOKENS_KEY: torch.tensor(probe, dtype=torch.int32),
            PROBE_INPUTS_KEY: expected.contiguous(),
        },
        str(destination / PLE_PROBE_FILE),
    )
    return index
