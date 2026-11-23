"""Script that writes the vision tower of EmbeddingGemma 2 (an image's patches → its soft tokens)
to an IR v2 container + golden io.

    uv run --with 'transformers==5.23.0' python -m embeddinggemma2.export_vision
    uv run --with 'transformers==5.23.0' python -m embeddinggemma2.export_vision --dtype bf16
    uv run --with 'transformers==5.23.0' python -m embeddinggemma2.export_vision --dtype i8

transformers is pinned to 5.23.0, as for the text tower ({@link embeddinggemma2.export}).

## Source

The pinned snapshot of `google/embeddinggemma-2` ({@link embeddinggemma2.sources.PIN}). Of
`model.safetensors` only the 210 `vision_tower.*` tensors and `embed_vision.embedding_projection`
are read, loaded strictly into upstream's `Gemma4VisionModel` (the class `vision_config` names,
`model_type` `gemma4_vision`) and `EmbeddingGemma2MultimodalEmbedder` ({@link load_tower}); the
weights are bf16 and upcast to f32 (exact). The goldens' images go through upstream's
`Gemma4ImageProcessor` as the snapshot's `processor_config.json` configures it.

## The graph and its io contract

    pixels      [1, 9n, 768] f32   the patch rows, rescaled as upstream's processor gives them
    position_x  [1, 9n]      i32   each patch's column in the resized image's patch grid
    position_y  [1, 9n]      i32   each patch's row
    → soft      [1, n, 512]  f32   the image's soft tokens, ready to be soft rows of the text graph

`n` is the image's soft-token count, the graph's one symbol; the patch axis is `9n` (3 × 3 patches
per soft token). A patch row is upstream's: the 16 × 16 × 3 values of one patch flattened in
(row, column, channel) order, each `u8 · f32(1/255)` (the processor's rescale; no normalisation —
the tower applies `2 · (x − 0.5)` itself).

**The host passes only real patches, ordered so that each soft token's 9 patches are adjacent.**
With the resized image cut into a `W × H` patch grid (`W` and `H` multiples of 3), soft token
`j = bx + (W / 3) · by` is the 3 × 3 block of patches at columns `3bx … 3bx + 2` and rows
`3by … 3by + 2`; the rows of the graph's inputs are the blocks in that order (row of blocks by row
of blocks, left to right), and inside a block its 9 patches in raster order (`dy` then `dx`):

    patch i of the graph = block i // 9, (dy, dx) = divmod(i % 9, 3)
    position_x = 3 · bx + dx,   position_y = 3 · by + dy

That order is part of the contract ({@link block_order} writes it): upstream pools each 3 × 3 block
by its positions (a one-hot matrix product), so here the pooling is a reshape `[1, n, 9, 768]` and a
mean over the 9 — rows in another order still run, and pool the wrong patches together. The soft
tokens come out in upstream's order (block raster), which is the order of the image's placeholder
rows in the text.

Upstream's processor pads every image to 2,520 patch rows (the 280-token budget × 9) with zero rows
at position `(−1, −1)`; the tower masks them as keys, zeroes them before pooling and strips their
pooled rows, so the real rows' values do not depend on them. The host strips them, and the graph has
no mask, no padding row and no data-dependent shape.

## What the wrapper changes ({@link VisionTowerWrapper})

Upstream's modules run as they are — the patch embedder's `input_proj`, the 16 encoder layers
(sandwich RMSNorms, q / k norms, the scale-free v norm, attention with `scaling = 1.0`, the gated
GELU-tanh MLP), the pooler's `× sqrt(768)` in f32 and `embed_vision` (scale-free RMSNorm → linear
768 → 512) — except three forms the IR has no op for, each rewritten without a new op:

- **The positions' tables.** Upstream reads `position_embedding_table[2, 10240, 768]` by its two
  axes; the wrapper holds the two halves as two tables read by `embedding` (the Gemma 4 12B
  embedder's precedent), added first and then to the projection, as upstream groups it. The two
  halves are views of upstream's table, and every other module of the wrapper is upstream's own
  object: the wrapper holds no copy of a weight, so a rounding of its weights is the rounding of
  the modules the goldens run.
- **The 2-D RoPE.** Upstream computes `cos` / `sin` of `position · inv_freq` at run time; `cos` is
  not a run-time op. The wrapper folds upstream's own rotary module at the positions
  `0 … P − 1` into two tables `[P, 32]` ({@link rope_tables} — the 32 values of one axis, the same
  16 frequencies twice, as upstream lays them out) and gathers them by `position_x` and
  `position_y` with `embedding`: `[cos(x), cos(x), cos(y), cos(y)]` is upstream's layout. `P` =
  {@link ROPE_POSITIONS} = 840, the longest side the resize rule gives at 280 soft tokens (13,440
  px = 840 patches) — a position past it is outside the graph's range. Preferred over Gemma 4's host
  RoPE inputs: the values are torch's own (the host would compute `cos` with another library and
  round differently), and the host binds no extra input.
- **The pooling** — the block order above.

The attention runs through the text tower's bidirectional hook
({@link embeddinggemma2.export.bidirectional_gqa_attention}) with no mask: every patch sees every
patch, as upstream's mask does once the padding rows are gone.

## Checks before anything is published

- {@link assert_rope_tables} — the gathered `cos` / `sin` equal upstream's rotary module at each
  golden image's positions, bit for bit.
- {@link compare_with_upstream} — on every golden image, the wrapper's soft tokens against
  upstream's forward on the processor's padded rows (the golden) and on the same rows the graph
  takes; both within {@link UPSTREAM_TOLERANCE} (max abs, soft tokens reach about 5).
- {@link assert_ir_form} — the three inputs and the one output with their shapes, one symbol, no
  state, 16 attention nodes `[1, 12, 9n, 64]` with scale 1 and no mask, the six `embedding`
  lookups (the two learned tables and the two RoPE tables by both axes), and no `where`.
- {@link round_weights_to_i8} (`--dtype i8`) — the rounding reached exactly the linears and the
  two learned position tables.
- {@link assert_storage_form} — `f32`: every initializer f32; `bf16`: every linear and the two
  learned position tables bf16 (the checkpoint's values, nothing rounded), the RoPE tables and the
  norms f32; `i8`: the same weight slots i8 and exactly the rounded weights, the rest f32.

## Storage

`--dtype f32` (the default, {@link VISION_SERIES}) and `--dtype bf16` (`<series>-bf16` — upstream's
own dtype for the weights): the linears and the learned position tables bf16, everything else f32.
The RoPE tables sit in an `embedding` weight slot but are not bf16 values, so they are kept f32
explicitly ({@link ROPE_TABLES}). The goldens are the same for both (the values are the same).
Activations are f32 throughout, as the text tower's.

`--dtype i8` (`<series>-i8` — the distribution's int8 preset, the text tower's i8 method): every
tensor the bf16 series stores as bf16 — the 114 linear weights and the two learned position tables
— is rounded to per-row symmetric int8 (`karume.quantize.fake_quant_int8`, one F32 scale per output
row, per position row for the tables) **before** the reference and the goldens are taken
({@link round_weights_to_i8}); the RoPE tables stay f32 as in bf16. The tables are bare parameters
read by `embedding`, so the rounding names them as that op's weights (`functional_weights`) and
stores each with its own scale. The graph and the images are the other series' own; the goldens
are upstream's forward on the rounded weights (the wrapper shares them), so the series' goldens
are the quantized tower's own outputs.

## Goldens

Upstream's processor and upstream's modules over the real weights in f32 (the rounded weights
under i8), padded rows included as upstream runs them, on the synthetic images of
{@link embeddinggemma2.images.IMAGES} (a square, a wide and a tall aspect, and the smallest
soft-token count the resize rule produces):
`io.<image>.safetensors` holds `input.pixels` / `input.position_x` / `input.position_y` (the rows
in the graph's order) and `output.0` (upstream's soft tokens `[1, n, 512]`); `reference.json`
records per image its size, the SHA-256 of its RGB8 bytes and of the processor's pixel rows, the
resized size, the soft-token count and the measured differences.

## Output layout

    outputs/series/embeddinggemma-2-vision/model-NNNNN-of-NNNNN.krm  the container (graph `vision`)
    outputs/series/embeddinggemma-2-vision/io.<image>.safetensors   the goldens
    outputs/series/embeddinggemma-2-vision/reference.json            provenance and measurements
    outputs/series/embeddinggemma-2-vision-bf16/…                    the same with `--dtype bf16`
    outputs/series/embeddinggemma-2-vision-i8/…                      `--dtype i8` (its own goldens)
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Collection, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch import nn
from torch.export import Dim
from torch.nn import functional

from _shared.paths import SERIES_ROOT
from embeddinggemma.export import INPUT_PREFIX, IO_PREFIX, IO_SUFFIX, MODEL_FILE, OUTPUT_PREFIX
from embeddinggemma2 import sources
from embeddinggemma2.export import (
    ATTENTION_NAME,
    LICENSE,
    WHERE_OP,
    EmbeddingGemma2ExportError,
    assert_transformers_version,
    register_attention,
)
from embeddinggemma2.images import IMAGES, SyntheticImage
from karume.artifacts import staged_publication
from karume.container import Provenance, container_parts
from karume.convert import PRESERVED_OP_PREFIXES_WITH_ATTENTION, normalize_boundary_tensor
from karume.emit import eligible_compressed_initializers
from karume.ir import IrGraph
from karume.ops import ATTENTION_OP
from karume.pipeline import export_module, publish_model
from karume.quantize import QUANT_MODULE_TYPES, Int8Report, fake_quant_int8, iter_quant_targets
from karume.shapes import declared_shape

#: The series this script writes.
VISION_SERIES = "embeddinggemma-2-vision"

#: The storage dtypes `--dtype` takes.
WEIGHT_DTYPES = ("f32", "bf16", "i8")

#: One series per storage (ADR 0019).
DEFAULT_OUT_DIRS: Mapping[str, Path] = {
    "f32": SERIES_ROOT / VISION_SERIES,
    "bf16": SERIES_ROOT / f"{VISION_SERIES}-bf16",
    "i8": SERIES_ROOT / f"{VISION_SERIES}-i8",
}

#: The graph name (the component name a distribution will give the tower).
GRAPH_NAME = "vision"

#: The graph's inputs, the symbol (the soft-token count) and the output's width.
PIXELS = "pixels"
POSITION_X = "position_x"
POSITION_Y = "position_y"
GRAPH_INPUTS = (PIXELS, POSITION_X, POSITION_Y)
SYMBOL = "n"
SOFT_WIDTH = 512

#: The snapshot files the script reads (each checked against the pin first).
READ_FILES: tuple[str, ...] = (
    "model.safetensors",
    "config.json",
    "processor_config.json",
    "preprocessor_config.json",
)

#: The checkpoint's key prefixes of the tower and of its projector.
TOWER_PREFIX = "vision_tower."
PROJECTION_PREFIX = "embed_vision."

#: The soft-token budget the processor runs at (`max_soft_tokens` of `processor_config.json`,
#: checked) — the upper bound of `n`.
SOFT_TOKEN_BUDGET = 280

#: The patches per soft token (`pooling_kernel_size²`, checked against the config).
POOLING = 3
PATCHES_PER_TOKEN = POOLING * POOLING

#: The positions the RoPE tables cover: the longest side the resize rule gives at the budget is
#: `(budget · 9 // 9) · 48` px = 13,440 px, 840 patches (module doc).
ROPE_POSITIONS = SOFT_TOKEN_BUDGET * POOLING

#: The smallest soft-token count the resize rule produces at the budget (the `stripes-edge` image —
#: a rounding edge of the rule, `embeddinggemma2.images`; pinned by the tests) — the lower bound of
#: `n` in the trace.
SOFT_TOKENS_MIN = 139

#: The tensor keys of the two RoPE tables (kept f32 under bf16 — module doc, "Storage").
ROPE_COS = "rope_cos"
ROPE_SIN = "rope_sin"
ROPE_TABLES = (ROPE_COS, ROPE_SIN)

#: The two learned position tables (the halves of `position_embedding_table`).
POSITION_TABLES = ("pos_x", "pos_y")

#: The attention implementation of upstream's own forward in {@link compare_with_upstream}.
UPSTREAM_ATTENTION = "sdpa"

#: The bound of "equal to f32 noise" between the wrapper and upstream (max abs of the soft tokens,
#: which reach 5.0 to 6.2). Measured on 2026-11-22 over the four golden images: the wrapper against
#: upstream on the same rows 1.19e-6 to 1.67e-6 (the pooling's summation order), upstream on the
#: graph's rows against upstream on the processor's padded rows 1.98e-5 to 3.85e-5 (upstream's own
#: matrix products over 2,520 rows instead of the real ones), so the wrapper against the golden
#: 1.86e-5 to 3.87e-5 — the bound is 2.6 times the worst. The rows in the processor's raster
#: order (patches pooled into the wrong tokens) or the two position axes swapped move them by 5.9
#: to 6.5 (measured on two goldens).
UPSTREAM_TOLERANCE = 1e-4

#: The trace's example soft-token count (inside the symbol's range, far from 0 / 1).
EXAMPLE_TOKENS = 200

#: The golden file names.
REFERENCE_FILE = "reference.json"


class VisionExportError(EmbeddingGemma2ExportError):
    """An input of the script is not what the tower graph is built for (fails before writing)."""


# ---- The upstream modules ---------------------------------------------------------------------


@dataclass(frozen=True)
class Tower:
    """Upstream's vision tower and its projector over the real weights in f32, and their config."""

    tower: nn.Module
    embedder: nn.Module
    config: Any


def load_config(directory: Path) -> Any:
    """The composite config (`EmbeddingGemma2Config`), its vision part checked against what the
    graph is built for."""
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(directory)
    vision = config.vision_config
    facts = {
        "model_type": (vision.model_type, "gemma4_vision"),
        "pooling_kernel_size": (int(vision.pooling_kernel_size), POOLING),
        "use_clipped_linears": (bool(vision.use_clipped_linears), False),
        "standardize": (bool(vision.standardize), False),
        "text hidden_size": (int(config.text_config.hidden_size), SOFT_WIDTH),
    }
    wrong = {key: found for key, (found, expected) in facts.items() if found != expected}
    if wrong:
        raise VisionExportError(f"the vision config is not the one the graph is built for: {wrong}")
    return config


def checkpoint_state(model_file: Path) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """`(tower state, projector state)` under upstream's module keys, upcast to f32 (exact)."""
    tower: dict[str, torch.Tensor] = {}
    projector: dict[str, torch.Tensor] = {}
    with safe_open(str(model_file), framework="pt") as handle:
        for key in handle.keys():  # noqa: SIM118 — safe_open is not a mapping
            for prefix, state in ((TOWER_PREFIX, tower), (PROJECTION_PREFIX, projector)):
                if key.startswith(prefix):
                    state[key[len(prefix) :]] = handle.get_tensor(key).to(torch.float32, copy=True)
    if not tower or not projector:
        raise VisionExportError(
            f"{model_file} lacks '{TOWER_PREFIX}*' or '{PROJECTION_PREFIX}*' tensors"
        )
    return tower, projector


def load_tower(directory: Path) -> Tower:
    """Upstream's `Gemma4VisionModel` and `EmbeddingGemma2MultimodalEmbedder` with the real weights
    in f32.

    MUST: strict loads — a key renamed upstream would otherwise leave a random tensor in the graph.
    """
    import transformers
    from transformers import AutoModel
    from transformers.models.embedding_gemma2.modeling_embedding_gemma2 import (
        EmbeddingGemma2MultimodalEmbedder,
    )

    assert_transformers_version(transformers.__version__)
    register_attention()
    config = load_config(directory)
    config.vision_config._attn_implementation = ATTENTION_NAME
    tower_state, projector_state = checkpoint_state(directory / "model.safetensors")
    tower = AutoModel.from_config(config.vision_config)
    if type(tower).__name__ != "Gemma4VisionModel":
        raise VisionExportError(f"vision_config builds {type(tower).__name__}")
    tower.load_state_dict(tower_state, strict=True)
    embedder = EmbeddingGemma2MultimodalEmbedder(config.vision_config, config.text_config)
    embedder.load_state_dict(projector_state, strict=True)
    tower = tower.to(torch.float32).eval().requires_grad_(False)
    embedder = embedder.to(torch.float32).eval().requires_grad_(False)
    return Tower(tower=tower, embedder=embedder, config=config)


@contextmanager
def upstream_attention(config: Any) -> Iterator[None]:
    """Run the tower on upstream's own attention implementation inside the block."""
    previous = config._attn_implementation
    config._attn_implementation = UPSTREAM_ATTENTION
    try:
        yield
    finally:
        config._attn_implementation = previous


def upstream_soft_tokens(
    tower: Tower, pixel_values: torch.Tensor, position_ids: torch.Tensor
) -> torch.Tensor:
    """Upstream's soft tokens `[n, 512]` of the rows given (`get_image_features` of the composite:
    the tower, then `embed_vision`), on upstream's attention implementation."""
    with torch.no_grad(), upstream_attention(tower.config.vision_config):
        hidden = tower.tower(pixel_values=pixel_values, pixel_position_ids=position_ids)
        return tower.embedder(inputs_embeds=hidden.last_hidden_state)


# ---- The host's rows --------------------------------------------------------------------------


def block_order(width: int, height: int) -> list[int]:
    """The raster indices of a `width × height` patch grid in the graph's order (module doc, "The
    graph and its io contract"): the 3 × 3 blocks in raster order, inside each its 9 patches in
    raster order."""
    if width % POOLING or height % POOLING or width <= 0 or height <= 0:
        raise VisionExportError(f"a {width} × {height} patch grid is not cut into 3 × 3 blocks")
    return [
        (POOLING * by + dy) * width + POOLING * bx + dx
        for by in range(height // POOLING)
        for bx in range(width // POOLING)
        for dy in range(POOLING)
        for dx in range(POOLING)
    ]


@dataclass(frozen=True)
class HostRows:
    """One image as the graph takes it: the real patch rows in the graph's order."""

    #: `[1, 9n, 768]` f32.
    pixels: torch.Tensor
    #: `[1, 9n]` int64 each.
    position_x: torch.Tensor
    position_y: torch.Tensor

    @property
    def soft_tokens(self) -> int:
        return int(self.pixels.shape[1]) // PATCHES_PER_TOKEN

    def position_ids(self) -> torch.Tensor:
        """`[1, 9n, 2]` (x, y) — upstream's `pixel_position_ids` of these rows."""
        return torch.stack((self.position_x, self.position_y), dim=-1)


def host_rows(pixel_values: torch.Tensor, position_ids: torch.Tensor) -> HostRows:
    """The processor's padded rows of one image (`[1, P, 768]`, `[1, P, 2]`) → the graph's rows.

    MUST: the real rows are exactly the raster patches of a grid whose sides are multiples of 3,
    then only padding `(−1, −1)` — rows in any other form are not what the order is defined on.
    """
    positions = position_ids[0]
    real = (positions != -1).all(dim=-1)
    count = int(real.sum())
    if not bool(real[:count].all()) or bool(real[count:].any()):
        raise VisionExportError("the processor's real rows are not the first rows")
    if bool((positions[count:] != -1).any()) or bool(pixel_values[0, count:].any()):
        raise VisionExportError("the processor's padding rows are not zeros at (-1, -1)")
    width = int(positions[:count, 0].max()) + 1
    height = int(positions[:count, 1].max()) + 1
    raster = torch.stack(
        torch.meshgrid(torch.arange(width), torch.arange(height), indexing="xy"), dim=-1
    ).reshape(-1, 2)
    if count != width * height or not torch.equal(positions[:count], raster):
        raise VisionExportError(
            f"the {count} real rows are not the raster of a {width} × {height} grid"
        )
    order = torch.tensor(block_order(width, height))
    chosen = positions[order]
    return HostRows(
        pixels=pixel_values[0, order].unsqueeze(0).contiguous(),
        position_x=chosen[:, 0].unsqueeze(0).contiguous(),
        position_y=chosen[:, 1].unsqueeze(0).contiguous(),
    )


# ---- The wrapper ------------------------------------------------------------------------------


def rope_tables(rotary: nn.Module, positions: int) -> tuple[torch.Tensor, torch.Tensor]:
    """`(cos, sin)` `[positions, 32]`: upstream's own rotary module at the positions
    `0 … positions − 1` on both axes, the half of one axis (module doc).

    MUST: the x half and the y half are the same values (one table serves both axes); checked.
    """
    index = torch.arange(positions)
    position_ids = torch.stack((index, index), dim=-1).unsqueeze(0)
    with torch.no_grad():
        cos, sin = rotary(torch.zeros(1, dtype=torch.float32), position_ids)
    half = int(cos.shape[-1]) // 2
    for name, table in (("cos", cos), ("sin", sin)):
        if not torch.equal(table[0, :, :half], table[0, :, half:]):
            raise VisionExportError(f"the rotary {name}'s x and y halves differ at equal positions")
    return cos[0, :, :half].contiguous(), sin[0, :, :half].contiguous()


class VisionTowerWrapper(nn.Module):
    """`(pixels [1, 9n, 768], position_x [1, 9n], position_y [1, 9n]) → soft [1, n, 512]`:
    upstream's tower and `embed_vision` over the host's rows (module doc).

    MUST: the graph inputs carry these names (the host binds by name).
    """

    def __init__(self, tower: Tower, positions: int = ROPE_POSITIONS) -> None:
        super().__init__()
        model = tower.tower
        patch = model.patch_embedder
        self.input_proj = patch.input_proj
        # MUST: views of upstream's table, not copies — every parameter of the wrapper is
        # upstream's own, so a rounding of the wrapper's weights (`--dtype i8`) is the rounding of
        # the modules the goldens run.
        table = patch.position_embedding_table.detach()
        self.pos_x = nn.Parameter(table[0], requires_grad=False)
        self.pos_y = nn.Parameter(table[1], requires_grad=False)
        self.layers = model.encoder.layers
        cos, sin = rope_tables(model.encoder.rotary_emb, positions)
        self.register_buffer(ROPE_COS, cos)
        self.register_buffer(ROPE_SIN, sin)
        self.root_hidden_size = float(model.pooler.root_hidden_size)
        self.embed_vision = tower.embedder

    def rope(
        self, position_x: torch.Tensor, position_y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """`(cos, sin)` `[1, 9n, 64]` in upstream's layout, gathered from the tables."""
        cos = torch.cat(
            (
                functional.embedding(position_x, self.rope_cos),
                functional.embedding(position_y, self.rope_cos),
            ),
            dim=-1,
        )
        sin = torch.cat(
            (
                functional.embedding(position_x, self.rope_sin),
                functional.embedding(position_y, self.rope_sin),
            ),
            dim=-1,
        )
        return cos, sin

    def forward(
        self, pixels: torch.Tensor, position_x: torch.Tensor, position_y: torch.Tensor
    ) -> torch.Tensor:
        # Upstream's patch embedder (no normalisation in the processor — the tower scales here).
        hidden = self.input_proj(2 * (pixels - 0.5))
        # MUST: the two lookups are added first, then to the projection (upstream's grouping).
        positions = functional.embedding(position_x, self.pos_x) + functional.embedding(
            position_y, self.pos_y
        )
        hidden = hidden + positions
        rope = self.rope(position_x, position_y)
        # Only its last axis is read (the number of RoPE axes, 2).
        position_ids = torch.stack((position_x, position_y), dim=-1)
        for layer in self.layers:
            hidden = layer(
                hidden, attention_mask=None, position_embeddings=rope, position_ids=position_ids
            )
        # The block order makes the 3 × 3 average a reshape and a mean (module doc), spelled as
        # the sum over the 9 divided by 9 (`mean.dim` is not an IR op; torch's mean is that sum
        # and division too).
        tokens = hidden.shape[1] // PATCHES_PER_TOKEN
        blocks = hidden.reshape(1, tokens, PATCHES_PER_TOKEN, hidden.shape[-1])
        pooled = blocks.sum(dim=2) / PATCHES_PER_TOKEN
        # Upstream's pooler: × sqrt(hidden) in f32.
        return self.embed_vision(pooled.float() * self.root_hidden_size)


# ---- The processor ----------------------------------------------------------------------------


def load_image_processor(directory: Path) -> Any:
    """Upstream's image processor as the snapshot configures it, its budget checked."""
    from transformers import AutoImageProcessor

    processor = AutoImageProcessor.from_pretrained(directory)
    facts = {
        "class": (type(processor).__name__, "Gemma4ImageProcessor"),
        "max_soft_tokens": (int(processor.max_soft_tokens), SOFT_TOKEN_BUDGET),
        "patch_size": (int(processor.patch_size), 16),
        "pooling_kernel_size": (int(processor.pooling_kernel_size), POOLING),
        "do_normalize": (bool(processor.do_normalize), False),
    }
    wrong = {key: found for key, (found, expected) in facts.items() if found != expected}
    if wrong:
        raise VisionExportError(
            f"the image processor is not the one the graph is built for: {wrong}"
        )
    return processor


@dataclass(frozen=True)
class ProcessedImage:
    """One synthetic image through upstream's processor."""

    image: SyntheticImage
    #: `[1, 2520, 768]` f32 and `[1, 2520, 2]` int64, padded as the processor gives them.
    pixel_values: torch.Tensor
    position_ids: torch.Tensor
    soft_tokens: int
    rows: HostRows

    @property
    def resized(self) -> tuple[int, int]:
        """`(width, height)` of the resized image in pixels."""
        patch = 16
        return (
            (int(self.rows.position_x.max()) + 1) * patch,
            (int(self.rows.position_y.max()) + 1) * patch,
        )


def process_image(processor: Any, image: SyntheticImage) -> ProcessedImage:
    """Upstream's processor on the image's RGB8 bytes (a PIL image, as upstream receives one)."""
    from PIL import Image

    pil = Image.fromarray(image.rgb8(), mode="RGB")
    features = processor(images=[pil], return_tensors="pt")
    pixel_values = features["pixel_values"]
    position_ids = features["image_position_ids"]
    if pixel_values.dtype != torch.float32:
        raise VisionExportError(f"{image.name}: the processor's pixels are {pixel_values.dtype}")
    soft_tokens = int(features["num_soft_tokens_per_image"][0])
    rows = host_rows(pixel_values, position_ids)
    if rows.soft_tokens != soft_tokens:
        raise VisionExportError(
            f"{image.name}: {rows.soft_tokens} soft tokens from the rows, the processor says"
            f" {soft_tokens}"
        )
    return ProcessedImage(image, pixel_values, position_ids, soft_tokens, rows)


# ---- Checks -----------------------------------------------------------------------------------


def assert_rope_tables(wrapper: VisionTowerWrapper, tower: Tower, rows: HostRows) -> None:
    """The gathered `cos` / `sin` equal upstream's rotary module at the rows' positions, bit for
    bit."""
    rotary = tower.tower.encoder.rotary_emb
    with torch.no_grad():
        expected = rotary(torch.zeros(1, dtype=torch.float32), rows.position_ids())
        gathered = wrapper.rope(rows.position_x, rows.position_y)
    for name, left, right in zip(("cos", "sin"), gathered, expected, strict=True):
        if not torch.equal(left, right):
            raise AssertionError(
                f"the gathered RoPE {name} differs from upstream's by"
                f" {float((left - right).abs().max()):.3e}"
            )


def _difference(left: torch.Tensor, right: torch.Tensor) -> dict[str, float]:
    left, right = left.reshape(-1).double(), right.reshape(-1).double()
    return {
        "max_abs": float((left - right).abs().max()),
        "cosine": float(torch.dot(left, right) / (left.norm() * right.norm())),
        "max_abs_value": float(right.abs().max()),
    }


def compare_with_upstream(
    wrapper: VisionTowerWrapper, tower: Tower, image: ProcessedImage
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Upstream's soft tokens of the processor's padded rows (the golden) and the measured
    differences: the wrapper against them and against upstream on the graph's own rows, and
    upstream's two forwards against each other (what the padding and the row order alone move).

    MUST: each difference within {@link UPSTREAM_TOLERANCE} — beyond it the wrapper is not
    upstream's tower.
    """
    golden = upstream_soft_tokens(tower, image.pixel_values, image.position_ids)
    same_rows = upstream_soft_tokens(tower, image.rows.pixels, image.rows.position_ids())
    with torch.no_grad():
        ours = wrapper(image.rows.pixels, image.rows.position_x, image.rows.position_y)[0]
    if list(golden.shape) != [image.soft_tokens, SOFT_WIDTH] or ours.shape != golden.shape:
        raise AssertionError(
            f"{image.image.name}: soft tokens {list(golden.shape)} / {list(ours.shape)}, not"
            f" [{image.soft_tokens}, {SOFT_WIDTH}]"
        )
    measured = {
        "wrapper_vs_upstream_padded": _difference(ours, golden),
        "wrapper_vs_upstream_same_rows": _difference(ours, same_rows),
        "upstream_same_rows_vs_padded": _difference(same_rows, golden),
    }
    worst = max(entry["max_abs"] for entry in measured.values())
    if worst > UPSTREAM_TOLERANCE:
        raise AssertionError(
            f"{image.image.name}: the wrapper differs from upstream's tower by {measured}"
            f" (> {UPSTREAM_TOLERANCE})"
        )
    return golden, measured


def trace(wrapper: VisionTowerWrapper) -> tuple[IrGraph, dict[str, torch.Tensor]]:
    """Export with the soft-token count symbolic in `[SOFT_TOKENS_MIN, SOFT_TOKEN_BUDGET]` and the
    patch axis `9n` (the graph does not depend on the example's values)."""
    tokens = Dim(SYMBOL, min=SOFT_TOKENS_MIN, max=SOFT_TOKEN_BUDGET)
    rows = EXAMPLE_TOKENS * PATCHES_PER_TOKEN
    width = int(wrapper.input_proj.in_features)
    patches = PATCHES_PER_TOKEN * tokens
    return export_module(
        wrapper,
        (
            torch.zeros(1, rows, width, dtype=torch.float32),
            torch.zeros(1, rows, dtype=torch.int64),
            torch.zeros(1, rows, dtype=torch.int64),
        ),
        dynamic_shapes=({1: patches}, {1: patches}, {1: patches}),
        symbol_names=(SYMBOL,),
        preserved=PRESERVED_OP_PREFIXES_WITH_ATTENTION,
    )


def assert_ir_form(graph: IrGraph, config: Any) -> dict[str, Any]:
    """The traced form (module doc, "Checks"). Each property breaks while the numbers can still
    agree: a mask left in the graph, a widened k / v, an attention scale other than upstream's 1,
    a lookup read by the wrong axis or a position table missing, a `where` of a padding mask."""
    vision = config.vision_config
    width = int(vision.patch_size) ** 2 * 3
    patches = f"{PATCHES_PER_TOKEN}{SYMBOL}"
    expected = [
        [PIXELS, "f32", [1, patches, width]],
        [POSITION_X, "i32", [1, patches]],
        [POSITION_Y, "i32", [1, patches]],
    ]
    found = [[spec.name, spec.dtype, list(spec.shape)] for spec in graph.inputs]
    if found != expected:
        raise AssertionError(f"graph inputs are {found}, not {expected}")
    if list(graph.symbols) != [SYMBOL] or graph.states:
        raise AssertionError(f"symbols {list(graph.symbols)} / states {sorted(graph.states)}")
    if len(graph.outputs) != 1:
        raise AssertionError(f"{len(graph.outputs)} outputs (the soft tokens are one)")
    output = graph.values[graph.outputs[0]]
    if output.dtype != "f32" or list(output.shape) != [1, SYMBOL, SOFT_WIDTH]:
        raise AssertionError(f"the output is {output.dtype} {list(output.shape)}")
    heads, head_dim = int(vision.num_attention_heads), int(vision.head_dim)
    attentions = [node for node in graph.nodes if node.op == ATTENTION_OP]
    if len(attentions) != int(vision.num_hidden_layers):
        raise AssertionError(
            f"{len(attentions)} attention nodes for {vision.num_hidden_layers} layers"
        )
    for index, node in enumerate(attentions):
        shapes = [declared_shape(graph, name) for name in node.ins]
        if shapes != [[1, heads, patches, head_dim]] * 3 or node.attrs.get("scale") != 1.0:
            raise AssertionError(
                f"attention[{index}] takes {shapes} with {node.attrs} — not three"
                f" [1, {heads}, {patches}, {head_dim}] with scale 1 and no mask"
            )
    lookups = sorted((node.ins[0], node.ins[1]) for node in graph.nodes if node.op == "embedding")
    tables = {node.ins[0] for node in graph.nodes if node.op == "embedding"}
    by_axis = sorted((graph.initializers[table].tensor, index) for table, index in lookups)
    expected_lookups = sorted(
        [
            ("pos_x", POSITION_X),
            ("pos_y", POSITION_Y),
            (ROPE_COS, POSITION_X),
            (ROPE_COS, POSITION_Y),
            (ROPE_SIN, POSITION_X),
            (ROPE_SIN, POSITION_Y),
        ]
    )
    if by_axis != expected_lookups:
        raise AssertionError(f"the embedding lookups are {by_axis}, not {expected_lookups}")
    rows = {graph.initializers[table].tensor: list(graph.values[table].shape) for table in tables}
    expected_rows = {
        "pos_x": [int(vision.position_embedding_size), int(vision.hidden_size)],
        "pos_y": [int(vision.position_embedding_size), int(vision.hidden_size)],
        ROPE_COS: [ROPE_POSITIONS, head_dim // 2],
        ROPE_SIN: [ROPE_POSITIONS, head_dim // 2],
    }
    if rows != expected_rows:
        raise AssertionError(f"the tables are {rows}, not {expected_rows}")
    wheres = [node for node in graph.nodes if node.op == WHERE_OP]
    if wheres:
        raise AssertionError(f"{len(wheres)} `{WHERE_OP}` nodes (no padding mask is in the graph)")
    return {
        "inputs": found,
        "attention_nodes": len(attentions),
        "ops": dict(Counter(node.op for node in graph.nodes).most_common()),
    }


def round_weights_to_i8(wrapper: VisionTowerWrapper) -> Int8Report:
    """Round, in place, the weights the bf16 series stores as bf16 — every `nn.Linear` weight and
    the two learned position tables — to per-row symmetric int8, and return the scale ledger
    (keyed by the wrapper's FQNs, the exported tensor keys).

    The tables are bare parameters read by `embedding`, rounded as that op's weights (per position
    row) through `functional_weights`. MUST: called before the reference and the goldens are taken
    (module doc, "Storage"); the wrapper shares every weight with upstream's modules, so the
    goldens are the rounded tower's. MUST: the rounding reaches exactly those weights — a weight
    left out ships its bf16 value under the int8 preset's name. The target set is decided before
    anything is rounded, so a refusal leaves the weights as they were.
    """
    linears = {
        f"{name}.weight"
        for name, module in wrapper.named_modules()
        if isinstance(module, nn.Linear)
    }
    quantizable = {fqn for fqn, _weight, _axis in iter_quant_targets(wrapper, QUANT_MODULE_TYPES)}
    if quantizable != linears:
        raise VisionExportError(
            f"the module weights of the tower are {len(quantizable)}, not its {len(linears)}"
            f" linears (other module types {sorted(quantizable - linears)[:3]})"
        )
    report = fake_quant_int8(
        wrapper,
        lambda name: f"{name}.weight" in linears,
        functional_weights={table: nn.Embedding for table in POSITION_TABLES},
    )
    expected = linears | set(POSITION_TABLES)
    if set(report.scales) != expected:
        missing = sorted(expected - set(report.scales))[:3]
        extra = sorted(set(report.scales) - expected)[:3]
        raise VisionExportError(
            f"the int8 rounding reached {len(report.scales)} weights, not the {len(expected)}"
            f" linears and position tables (missing {missing} / extra {extra})"
        )
    return report


def assert_storage_form(
    graph: IrGraph, dtype: str, rounded: Collection[str] = ()
) -> dict[str, int]:
    """The storage the writer committed (module doc, "Storage"): under `bf16` / `i8`, every weight
    slot but the RoPE tables in that dtype and the rest f32; under `f32`, every initializer f32 (or
    i32).

    `rounded` is the tensor keys the int8 rounding reached ({@link round_weights_to_i8}); `i8`
    needs it and the other storages take none. MUST: under `i8` the i8 weight slots are exactly
    those keys — a rounded weight outside a slot would ship its rounded values as f32.
    """
    if dtype not in WEIGHT_DTYPES:
        raise VisionExportError(f"storage {dtype!r} is not one of {WEIGHT_DTYPES}")
    if (dtype == "i8") != bool(rounded):
        raise VisionExportError(
            f"the {dtype} storage form takes {'the' if dtype == 'i8' else 'no'} rounded weights"
            f" ({len(rounded)} given)"
        )
    eligible = eligible_compressed_initializers(graph)
    counts: dict[str, int] = {}
    wrong: list[str] = []
    for name, initializer in graph.initializers.items():
        stored = initializer.storage.dtype
        counts[stored] = counts.get(stored, 0) + 1
        compressed = dtype != "f32" and name in eligible and initializer.tensor not in ROPE_TABLES
        expected = {dtype} if compressed else {"f32", "i32"}
        if stored not in expected:
            wrong.append(
                f"{initializer.tensor}: {stored} (expected {' / '.join(sorted(expected))})"
            )
    if wrong:
        raise AssertionError(f"the {dtype} storage form does not hold: {wrong[:4]}")
    stored_tables = {
        initializer.tensor: initializer.storage.dtype
        for initializer in graph.initializers.values()
        if initializer.tensor in (*POSITION_TABLES, *ROPE_TABLES)
    }
    expected_tables = {
        **{name: dtype for name in POSITION_TABLES},
        **{name: "f32" for name in ROPE_TABLES},
    }
    if stored_tables != expected_tables:
        raise AssertionError(f"the tables are stored {stored_tables}, not {expected_tables}")
    if dtype == "i8":
        slots = {
            initializer.tensor
            for initializer in graph.initializers.values()
            if initializer.storage.dtype == "i8"
        }
        if slots != set(rounded):
            raise AssertionError(
                f"the i8 weight slots are not the {len(rounded)} rounded weights (stored i8 but"
                f" not rounded {sorted(slots - set(rounded))[:3]} / rounded but not stored i8"
                f" {sorted(set(rounded) - slots)[:3]})"
            )
    return dict(sorted(counts.items()))


# ---- Writing ----------------------------------------------------------------------------------


def _sha256(array: np.ndarray | torch.Tensor) -> str:
    data = array.numpy() if isinstance(array, torch.Tensor) else array
    return hashlib.sha256(np.ascontiguousarray(data).tobytes()).hexdigest()


def write_io(graph: IrGraph, image: ProcessedImage, golden: torch.Tensor, out_dir: Path) -> str:
    """`io.<image>.safetensors`: the graph's three inputs and upstream's soft tokens
    `[1, n, 512]`."""
    args = {
        PIXELS: image.rows.pixels,
        POSITION_X: image.rows.position_x,
        POSITION_Y: image.rows.position_y,
    }
    tensors = {
        f"{INPUT_PREFIX}{spec.name}": normalize_boundary_tensor(
            args[spec.name], f"{image.image.name}: input '{spec.name}'"
        )
        for spec in graph.inputs
    }
    tensors[f"{OUTPUT_PREFIX}0"] = normalize_boundary_tensor(
        golden.unsqueeze(0).contiguous(), f"{image.image.name}: output 0"
    )
    path = out_dir / f"{IO_PREFIX}{image.image.name}{IO_SUFFIX}"
    save_file(tensors, str(path))
    return path.name


def image_record(image: ProcessedImage, measured: Mapping[str, Any]) -> dict[str, Any]:
    """What `reference.json` records of one golden image."""
    width, height = image.resized
    return {
        "name": image.image.name,
        "why": image.image.why,
        "width": image.image.width,
        "height": image.image.height,
        "rgb8Sha256": _sha256(image.image.rgb8()),
        "resized": {"width": width, "height": height},
        "softTokens": image.soft_tokens,
        "processorPixelsSha256": _sha256(
            image.pixel_values[0, : image.soft_tokens * PATCHES_PER_TOKEN]
        ),
        "measured": dict(measured),
    }


def export_series(
    directory: Path,
    out_dir: Path,
    *,
    dtype: str = "f32",
    images: Collection[str] = tuple(IMAGES),
) -> dict[str, Any]:
    """Write the tower's container and goldens, and return a summary (staged — nothing is placed
    unless every check passes)."""
    if dtype not in WEIGHT_DTYPES:
        raise VisionExportError(f"storage {dtype!r} is not one of {WEIGHT_DTYPES}")
    sources.assert_pinned(directory, READ_FILES)
    tower = load_tower(directory)
    processor = load_image_processor(directory)
    wrapper = VisionTowerWrapper(tower).eval()
    # MUST: i8 rounds before anything below reads the weights (the reference, the trace, the
    # goldens — module doc, "Storage").
    rounding = round_weights_to_i8(wrapper) if dtype == "i8" else None
    processed = [process_image(processor, IMAGES[name]) for name in images]
    counts = sorted(image.soft_tokens for image in processed)
    if counts[0] != SOFT_TOKENS_MIN:
        raise VisionExportError(f"no golden image runs at the smallest count {SOFT_TOKENS_MIN}")
    goldens: dict[str, tuple[torch.Tensor, dict[str, Any]]] = {}
    for image in processed:
        assert_rope_tables(wrapper, tower, image.rows)
        goldens[image.image.name] = compare_with_upstream(wrapper, tower, image)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    graph, tensors = trace(wrapper)
    form = assert_ir_form(graph, tower.config)
    with staged_publication(out_dir) as staged:
        staged.mkdir()
        graph = publish_model(
            staged / MODEL_FILE,
            graph,
            tensors,
            provenance=Provenance(license=LICENSE, upstream_revision=sources.PIN.revision),
            graph_name=GRAPH_NAME,
            weight_dtype=dtype,
            weight_scales=None if rounding is None else rounding.scales,
            weight_dtype_overrides={name: "f32" for name in ROPE_TABLES}
            if dtype != "f32"
            else None,
        )
        storage = assert_storage_form(
            graph, dtype, () if rounding is None else tuple(rounding.scales)
        )
        written = [
            write_io(graph, image, goldens[image.image.name][0], staged) for image in processed
        ]
        record = {
            "series": out_dir.name,
            "upstream": sources.PIN.label,
            "softTokenBudget": SOFT_TOKEN_BUDGET,
            "ropePositions": ROPE_POSITIONS,
            "upstreamTolerance": UPSTREAM_TOLERANCE,
            "images": [image_record(image, goldens[image.image.name][1]) for image in processed],
        }
        (staged / REFERENCE_FILE).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return {
        "dir": str(out_dir),
        "nodes": len(graph.nodes),
        "initializers": len(graph.initializers),
        "parts": len(container_parts(out_dir / MODEL_FILE)),
        "model_bytes": sum(p.stat().st_size for p in container_parts(out_dir / MODEL_FILE)),
        "ops": sorted(graph.required_ops),
        "storage": storage,
        **(
            {}
            if rounding is None
            else {"rounded": {"tensors": rounding.modules, "elements": rounding.elements}}
        ),
        "form": form,
        "io": written,
        "images": {name: measured for name, (_, measured) in goldens.items()},
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--snapshot-dir", type=Path, default=sources.SNAPSHOT_DIR)
    parser.add_argument("--dtype", choices=WEIGHT_DTYPES, default="f32")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    out_dir = args.out if args.out is not None else DEFAULT_OUT_DIRS[args.dtype]
    summary = export_series(args.snapshot_dir, out_dir, dtype=args.dtype)
    print(json.dumps({"dtype": args.dtype, **summary}, indent=1))


if __name__ == "__main__":
    main()
