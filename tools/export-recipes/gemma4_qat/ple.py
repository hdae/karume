"""Build the QAT PLE, still fixed and packed, into **the container's asset** (ADR 0097 addendum 4
/ ADR 0109 decision 4)."""

from __future__ import annotations

import math
from collections.abc import Buffer, Callable
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import torch
from safetensors.torch import save_file

from gemma4.export_product import (
    PLE_PROBE_FILE,
    PROBE_INPUTS_KEY,
    PROBE_TOKENS_KEY,
    ple_probe_tokens,
    spill_payload,
)
from karume.container import AssetInput
from karume.ple import ple_assets, ple_block_ranges

if TYPE_CHECKING:
    from transformers.integrations.gemma_quant import QuantizedEmbedding


class PleBuild(NamedTuple):
    """What {@link build_ple} returns — the assets passed to the container, and the facts the
    reference comparison needs."""

    #: The assets passed as they are to `publish_model(assets=…)` (the index + the block lists of
    #: `values` / `scales`).
    assets: dict[str, AssetInput]
    #: The scattered token ids used as the reference for bit-exact dequantization.
    probe: tuple[int, ...]
    #: The block count of `values` (what `reference.json` is compared against).
    blocks: int
    #: Temporary files in the working directory holding the token-major payload (not part of the
    #: distribution).
    spills: tuple[Path, ...]

    def discard(self) -> None:
        """Delete the temporary files (MUST be called before the working directory is swapped in
        — so they never end up in the distribution)."""
        for path in self.spills:
            path.unlink(missing_ok=True)


def _spill_reader(tensor: torch.Tensor, path: Path) -> Callable[[int, int], Buffer]:
    """Spill the row-major concatenated data to a temporary file in the working directory and
    return a range reader over it.

    Upstream already holds the QAT PLE token-major (`values[tokens, layers*dim//factor]` /
    `scales[tokens, layers]`), so no transpose is needed and the raw bytes are the payload as
    they are.

    MUST: **do not carry the data along** (the same rule as the product series'
    {@link gemma4.export_product._spill_tables}) — a reader holding the in-memory tensor would
    keep the packed data resident through `trace_qat` and the write. Once spilled, the caller can
    let go of the whole upstream module.
    """
    spill_payload(memoryview(tensor.numpy()).cast("B"), path)

    def read(begin: int, end: int) -> Buffer:
        with path.open("rb") as handle:
            handle.seek(begin)
            raw = handle.read(end - begin)
        if len(raw) != end - begin:
            raise AssertionError(f"{path}: the PLE range [{begin}, {end}) ran out partway")
        return raw

    return read


def build_ple(module: QuantizedEmbedding, layers: int, dim: int, destination: Path) -> PleBuild:
    """Build the fixed tables into the container's asset and write the scattered reference (probe)
    into the series.

    `destination` is a directory the staging side creates, and all this function writes there is
    the probe, **which is not part of the distribution** (the asset itself goes inside the
    container, so writing it is `publish_model`'s job).

    MUST: **do not requantize** the upstream fixed integers and scales — only the shape and the
    value range are checked, and the bytes become the asset's payload without a single bit
    rewritten.
    """
    bits, tokens = module.num_bits, module.num_embeddings
    if bits not in (2, 4) or layers <= 0 or dim <= 0 or dim % 16:
        raise ValueError("PLE needs I2/I4, positive layers and a dim that is a multiple of 16")
    factor = 8 // bits
    values, scales = module.embedding_quantized.detach(), module.embedding_scale.detach()
    embed_scale = float(module.scalar_embed_scale)
    if not math.isfinite(embed_scale) or embed_scale <= 0:
        raise ValueError("PLE embedScale must be a positive finite number")
    if values.device.type != "cpu" or values.dtype != torch.uint8 or not values.is_contiguous():
        raise ValueError("PLE packed values must be contiguous CPU U8")
    if values.shape != (tokens, layers * dim // factor):
        raise ValueError("the shape of the PLE packed values does not match tokens/layers/dim")
    if (
        scales.device.type != "cpu"
        or scales.dtype != torch.float32
        or scales.shape != (tokens, layers)
    ):
        raise ValueError("PLE scales must be CPU F32 [tokens,layers]")
    if not torch.isfinite(scales).all() or not (scales > 0).all():
        raise ValueError("PLE scales must be positive finite numbers")
    scales = scales.contiguous()
    storage = f"i{bits}"
    ranges = ple_block_ranges(storage=storage, tokens=tokens, layers=layers, dim=dim)
    probe = ple_probe_tokens(tokens, ranges["values"])
    with torch.inference_mode():
        expected = module(torch.tensor([list(probe)], dtype=torch.int64)).reshape(
            1, len(probe), layers, dim
        )
    save_file(
        {
            PROBE_TOKENS_KEY: torch.tensor(probe, dtype=torch.int32),
            PROBE_INPUTS_KEY: expected.contiguous(),
        },
        str(destination / PLE_PROBE_FILE),
    )
    spills = {key: destination / f".ple.{key}.spill" for key in ("values", "scales")}
    assets = ple_assets(
        storage=storage,
        tokens=tokens,
        layers=layers,
        dim=dim,
        embed_scale=embed_scale,
        read_values=_spill_reader(values, spills["values"]),
        read_scales=_spill_reader(scales, spills["scales"]),
    )
    return PleBuild(assets, probe, len(ranges["values"]), tuple(spills.values()))
