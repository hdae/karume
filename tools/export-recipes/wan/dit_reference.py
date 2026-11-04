"""The layer-by-layer CPU reference of the Wan DiT (f64 / f32 — ADR 0121 decision 7, step 0) and
its bit-exact comparison with the existing 1.3B goldens.

    uv run --group wan --inexact python -m wan.dit_reference compare --dtype i8 --max-tokens 192
    uv run --group wan --inexact python -m wan.dit_reference compare --dtype f16
    uv run --group wan --inexact python -m wan.dit_reference compare --dtype i8 --source module

## Why layer by layer

The Wan2.1 f64 reference loaded the f32 model and made all of it f64 with `model.double()` (about
10.4 GB for 1.3B). For 5B the full f64 size is 37.25 GiB, which does not fit a 31 GiB machine (the
capacity table of ADR 0121). So the DiT is made f64 / f32 and run **one block at a time**
({@link LayerwiseDit}). What is held at once is only the part outside the blocks (patch embedding,
condition embedding, output projection — 0.67 GiB in f64 for 5B) and one block (1.22 GiB likewise).
The loop variable of upstream's block loop keeps the previous block while asking for the next, so
the previous block's weights are released before the next is loaded ({@link LayerwiseDit._opener}).

## Running upstream's forward without rewriting it

Upstream `WanTransformer3DModel` is built on meta without weights, and `blocks` is replaced by "a
sequence that loads and materializes one block's weights each time it is iterated"
({@link _LazyBlocks}). Upstream's forward (RoPE table, patch embedding, condition embedding, block
loop, output norm / projection, unpatchify) **runs as upstream's own code** — nothing is copied, so
no copying mistakes come in. The RoPE table is a buffer-only component, so it is built by running
upstream's initialization once more on CPU (it is not a weight — the values of `model.rope` depend
on the config alone).

The per-token timesteps of Wan2.2 TI2V are passed through the reference wrapper of ADR 0121
decision 4 (`condition=` — `dit_patch.reference_dit`): upstream's `condition_embedder` is run with
M = 1 for each timestep value, and the per-token table is passed to upstream's blocks. It runs in
both f32 and f64 (the timestep sinusoid keeps its batch-1 shape — the shape the f64 watch allows).
diffusers' M = S path, which passes a 2-D timestep as is, is for observation and f32 only (it
builds an `[S]` sinusoid outside what the f64 watch allows, so it is stopped at the entrance —
{@link LayerwiseDit.forward}).

The eager of the patched graph (`dit_patch.WanDitTokensTi2v`) can also run on the same
layer-by-layer weights (pass the body to {@link LayerwiseDit.run} — the way to check eager
equivalence without holding the full 18.63 GiB of 5B f32, `wan.ti2v_export_dit`). The per-block
hooks (`collect_blocks` / `on_block` / `on_output`), however, fire from the blocks' forward hooks,
so they do not reach the patched TI2V graph, which calls the block's components directly (passing
them fails loudly — {@link LayerwiseDit.run}).

## Weight sources ({@link WeightSource} — every value is f32)

- {@link ContainerDitWeights}: one tensor at a time from the series container (`krm`). i8 is the
  packed values times the per-row scale multiplied **in f32** (the same values as the export's
  fake-quant — dequantizing in f64 would split bitwise from the existing goldens, which are f32
  fake-quant values widened to f64 — ADR 0121 decision 7). f16 is f16 → f32.
- {@link ModuleWeights}: an upstream model that is already loaded and rounded (the 1.3B of
  `wan.export_dit` — a machine that holds the full f32 model).
- {@link CheckpointDitWeights}: reads one tensor at a time from the upstream checkpoint (F32) and
  applies the series rounding (f16 / i8 fake-quant) to each (a preview of 5B, which has no container
  yet — `wan.dit_probe`). The rounding is closed within one tensor (f16 per element, i8 with a
  per-row scale), so it is bit-identical to rounding the whole model.

The f64 reference widens the f32 values to f64 whichever the source (`.to(float64)` is exact).

## The f32 reference and weight addresses (measured 2026-10-04 — torch 2.13.0+cpu, MKL 2024.2, AVX2)

On this machine the f32 GEMM (MKL) of torch CPU changes the last bits depending on **the offset of
the weight's start address from a 64-byte boundary**. Weights with the same values, shape and stride
moved the output by up to 4.6e-6 at S = 128 on 1.3B when only the offset changed (i8 series
`accept-s00128-t0030` of `compare`). The f64 reference was bit-identical on the same case (the f64
weights are fresh allocations on both paths — 64-byte aligned).

The f32 weights of upstream's eager (a model loaded with `from_pretrained`) sit at addresses that
mmap the checkpoint's safetensors. The offset is `(8 + header length + data_offsets[0]) mod 64`
(it matched the measurement for all 825 tensors of 1.3B). So when the f32 reference is given
`alignment` (key → offset — {@link upstream_alignment}), it places the weights at that offset
before running ({@link _placed}). This makes even weights read from a container bit-identical to
"upstream's f32 eager" ("the layer-by-layer f32 is bit-identical to upstream's f32 eager" of
ADR 0121 decision 7).
NOTE: this is the behavior of the MKL implementation, not a guarantee of the spec (enabling MKL's
CNR (`MKL_CBWR`) removes the dependence on the offset, but then results split bitwise from the
existing f32 goldens).

## No floating-point value other than f64 on the f64 path

The f64 forward is held to the same rule as the f64 reference of `dit_patch`: upstream's `.float()`
casts pass through unchanged in f64, and it stops if a floating-point value that is not f64 is
created (the only exception is the shape of the timestep sinusoid — `dit_patch.float64_forward`).
Weight loading (including dequantization in f32), however, is outside the f64 path, and the watch
is paused only during it ({@link _PausableWatch}).

## Comparison with the existing 1.3B goldens (`compare` — the pass condition of step 0)

Re-takes the case table of `wan.export_dit` ({@link wan.export_dit.series_cases}) layer by layer.
It writes `reference.<case>.safetensors` to **a separate location** (default {@link DEFAULT_OUT})
and then compares each tensor bitwise with the series' existing `reference.<case>` (whether the
files' sha256 match is also recorded). It never writes to the existing goldens or containers (read
only). Finished cases are skipped by looking at the result rows (`results.jsonl`), so after a stop
the same `--out` continues where it left off ({@link _done} — a row carries the run mode (full / f64
only) and the comparison version ({@link COMPARE_VERSION}), and rows that do not match are not
counted as done). At the end a summary goes to `summary.json` (it passes when `differ` is empty).
Rows and the summary also carry the execution environment the f32 bit-exactness depends on (torch
thread count, CPU instruction set, MKL / OpenMP environment variables —
{@link numeric_environment}).

MUST: import diffusers inside functions (the `wan` group is not in the default sync —
`tests/test_optional_group_imports.py`).
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import struct
import sys
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Literal, Protocol, TypeVar

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch import nn

from _shared.paths import BENCH_ROOT
from karume.container import codec_entry
from karume.quantize import QUANT_MODULE_TYPES, channel_scale, iter_quant_targets, quantize_to_int8
from wan import dit_patch
from wan.pipeline_ref import assert_no_mps
from wan.umt5_export import Checkpoint
from wan.umt5_reference import ContainerWeights

#: The default output location of the comparison
#: (`outputs/bench/<distribution name>/<date>_<purpose>/` — `docs/assets-layout.md`; separate from
#: the existing series and goldens — running jobs read the series). The date is the day of the
#: start, so to continue across a date change pass the earlier location with `--out`.
DEFAULT_OUT = BENCH_ROOT / "karume-wan2.1" / f"{date.today().isoformat()}_dit-layerwise"

#: The comparison's result rows (one JSON line per case — the resume markers) and the summary
#: written at the end.
RESULTS_FILE = "results.jsonl"
SUMMARY_FILE = "summary.json"

#: The comparison version. Bump it when what is compared (keys, case inputs, how the written goldens
#: are assembled) or the row fields change — rows of another version are not counted as "done" on
#: resume ({@link _done}).
COMPARE_VERSION = 2

#: The run mode (full = f32 output, blocks and f64 / f64 only = `--f64-only`).
CompareMode = Literal["full", "f64-only"]

#: Run mode → the row modes counted as "done" (a full row also covers an f64-only run — not the
#: other way round).
_COVERING_MODES: dict[str, frozenset[str]] = {
    "full": frozenset({"full"}),
    "f64-only": frozenset({"full", "f64-only"}),
}

#: Prefixes of the recorded environment variables (MKL's CNR (`MKL_CBWR`), thread counts, OpenMP
#: affinity).
_NUMERIC_ENV_PREFIXES = ("MKL_", "OMP_", "KMP_")

#: A weight that is a Conv3d upstream, held by the container in its Linear form
#: (`[Cout, C·pt·ph·pw]`).
PATCH_EMBEDDING_WEIGHT = "patch_embedding.weight"

#: File names of a diffusers component checkpoint (sharded index, single file).
DIFFUSERS_INDEX = "diffusion_pytorch_model.safetensors.index.json"
DIFFUSERS_SINGLE = "diffusion_pytorch_model.safetensors"

#: The upstream component directory (under the snapshot).
TRANSFORMER_SUBFOLDER = "transformer"

#: Weight rounding (the series storage dtype — `none` keeps upstream's F32 without quantizing).
Rounding = Literal["f16", "i8", "none"]


class DitReferenceError(RuntimeError):
    """A weight, shape or dtype departs from the premises of the layer-by-layer reference."""


# ---------------------------------------------------------------------------
# Weight sources
# ---------------------------------------------------------------------------


class WeightSource(Protocol):
    """Upstream state-dict key → f32 value (the patch embedding may come in its Linear form)."""

    def tensor(self, key: str) -> torch.Tensor: ...


class ModuleWeights:
    """The weights of an upstream model that is already loaded and rounded (only holds references —
    no copies)."""

    def __init__(self, model: nn.Module) -> None:
        self._state = model.state_dict()

    def tensor(self, key: str) -> torch.Tensor:
        value = self._state.get(key)
        if value is None:
            raise DitReferenceError(f"the model has no '{key}'")
        return value.detach()


class ContainerDitWeights(ContainerWeights):
    """Reads weights one tensor at a time in f32 from the series container (`krm`) (f32 / f16 /
    per-row i8).

    f32 and i8 are read exactly as `umt5_reference.ContainerWeights` reads them (i8 is packed ×
    scale in f32 — the same values as fake-quant). f16 is f16 → f32 (exact). Each block's sha256 is
    checked against its declaration every time it is taken.
    MUST: any other storage fails loudly (the reference would silently become other weights).
    """

    def tensor(self, key: str) -> torch.Tensor:
        supply = self._supply(key)
        if codec_entry(supply.encoding.codec).layout != "f16":
            return super().tensor(key)
        raw = b"".join(self._payload(supply, index) for index in range(len(supply.blocks)))
        half = torch.frombuffer(bytearray(raw), dtype=torch.float16)
        return half.reshape(self._shape(key)).to(torch.float32)


class CheckpointDitWeights:
    """Reads one tensor at a time from the upstream checkpoint (F32) and applies the series rounding
    to each.

    - `f16`: every weight to f16-representable values (the same per-element rounding as
      `karume.quantize.round_weights_to_f16` — fails loudly if a finite value saturates to
      non-finite)
    - `i8`: only `quant_keys` (the linear and patch-embedding weights — {@link quant_keys}) to
      per-channel symmetric RTN (the same `quantize(w, s)·s` as `fake_quant_int8`, in 2-D with
      output channels as rows). The rest (bias, norm, `scale_shift_table`) stays upstream's f32
    - `none`: upstream's F32 as is (the reference without quantization)
    """

    def __init__(
        self, directory: Path, rounding: Rounding, quant_keys: frozenset[str] = frozenset()
    ) -> None:
        if rounding == "i8" and not quant_keys:
            raise DitReferenceError("i8 rounding got no quantization targets (quant_keys)")
        self._checkpoint = Checkpoint(directory, index=DIFFUSERS_INDEX, single=DIFFUSERS_SINGLE)
        self._rounding = rounding
        self._quant_keys = quant_keys

    def tensor(self, key: str) -> torch.Tensor:
        value = self._checkpoint.read(key)
        if self._rounding == "f16":
            rounded = value.to(torch.float16).to(torch.float32)
            if bool((torch.isfinite(value) & ~torch.isfinite(rounded)).any()):
                raise DitReferenceError(
                    f"'{key}': rounding to f16 saturated a finite value to non-finite"
                )
            return rounded
        if self._rounding == "i8" and key in self._quant_keys:
            rows = value.reshape(value.shape[0], -1)
            scale = channel_scale(rows, 0)
            return (quantize_to_int8(rows, scale).to(torch.float32) * scale).reshape(value.shape)
        return value


def quant_keys(config: Mapping[str, Any]) -> frozenset[str]:
    """The quantization targets of the i8 series (FQNs of the linear weights of the S-shaped
    wrapper — the same selection as the export).

    The wrapper's FQNs live in the same space as upstream's state-dict keys (the wrapper holds
    references to upstream's components). The patch embedding is a Linear in the wrapper and a
    Conv3d upstream, with the same key `patch_embedding.weight`.
    """
    wrapper = dit_patch.WanDitTokens(meta_model(config))
    return frozenset(fqn for fqn, _, _ in iter_quant_targets(wrapper, op_types=QUANT_MODULE_TYPES))


def load_config(directory: Path) -> dict[str, Any]:
    """Upstream's DiT config (the `transformer/` directory — no weights are read)."""
    from diffusers import WanTransformer3DModel

    return dict(WanTransformer3DModel.load_config(directory))


#: The boundary the weight-address offset is measured from (bytes — module doc "The f32 reference
#: and weight addresses").
ALIGNMENT_BYTES = 64


def upstream_alignment(directory: Path) -> dict[str, int]:
    """The offset of each weight's start address from an {@link ALIGNMENT_BYTES} boundary when the
    upstream checkpoint is loaded with `from_pretrained` (key → bytes).

    safetensors places the data after an 8-byte length + the JSON header, and the start of an mmap
    is page-aligned, so the offset is `(8 + header length + data_offsets[0]) mod 64`. Only the
    headers are read.
    """
    files = sorted(directory.glob("*.safetensors"))
    if not files:
        raise DitReferenceError(f"no safetensors in {directory}")
    alignment: dict[str, int] = {}
    for file in files:
        with file.open("rb") as stream:
            (length,) = struct.unpack("<Q", stream.read(8))
            header = json.loads(stream.read(length))
        header.pop("__metadata__", None)
        for key, entry in header.items():
            alignment[key] = (8 + length + int(entry["data_offsets"][0])) % ALIGNMENT_BYTES
    return alignment


def _placed(value: torch.Tensor, offset: int) -> torch.Tensor:
    """Returns a copy of `value` placed at an address whose start is `offset` bytes away from an
    {@link ALIGNMENT_BYTES} boundary (contiguous, same shape — the values do not change)."""
    size = value.element_size()
    if offset % size or not 0 <= offset < ALIGNMENT_BYTES:
        raise DitReferenceError(f"an offset of {offset} bytes cannot hold {value.dtype}")
    buffer = torch.empty(value.numel() + ALIGNMENT_BYTES // size, dtype=value.dtype)
    shift = ((offset - buffer.data_ptr() % ALIGNMENT_BYTES) % ALIGNMENT_BYTES) // size
    placed = buffer[shift : shift + value.numel()].view(value.shape)
    placed.copy_(value)
    if placed.data_ptr() % ALIGNMENT_BYTES != offset:
        raise DitReferenceError(f"could not place the tensor at an offset of {offset} bytes")
    return placed


# ---------------------------------------------------------------------------
# The layer-by-layer forward
# ---------------------------------------------------------------------------


def meta_model(config: Mapping[str, Any]) -> nn.Module:
    """Upstream `WanTransformer3DModel` built from the config alone (weights and buffers on meta,
    eval)."""
    from diffusers import WanTransformer3DModel

    with torch.device("meta"):
        return WanTransformer3DModel.from_config(dict(config)).eval()


class _PausableWatch(dit_patch._NarrowFloatWatch):
    """The watch of the f64 path (the same as in `dit_patch`) plus a way to pause it while weights
    load."""

    def __init__(self, allowed: frozenset[tuple[int, ...]]) -> None:
        super().__init__(allowed)
        self._paused = False

    def __torch_dispatch__(
        self,
        func: Any,
        types: Any,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        if self._paused:
            return func(*args, **(kwargs or {}))
        return super().__torch_dispatch__(func, types, args, kwargs)

    @contextmanager
    def paused(self) -> Iterator[None]:
        self._paused = True
        try:
            yield
        finally:
            self._paused = False


class _LazyBlocks(nn.Module):
    """A sequence that materializes and hands out one block each time upstream's
    `for block in self.blocks:` iterates.

    It registers no submodules (`parameters()` does not pick up the meta templates — the outer
    dtype check sees only the part outside the blocks). {@link LayerwiseDit} supplies the
    materialization through `opener`.
    """

    def __init__(self, count: int) -> None:
        super().__init__()
        self.count = count
        self.opener: Callable[[int], nn.Module] | None = None

    def __len__(self) -> int:
        return self.count

    def __iter__(self) -> Iterator[nn.Module]:
        opener = self.opener
        if opener is None:
            raise DitReferenceError(
                "the layer-by-layer sequence was iterated with no weight source"
            )
        for index in range(self.count):
            yield opener(index)


@dataclass(frozen=True)
class DitReference:
    """The result of one layer-by-layer forward (the output `[1,C,F,H,W]` stays in the dtype it ran
    in)."""

    output: torch.Tensor
    #: Each block's output `[1,S,dim]` (only with `collect_blocks` — in the dtype it ran in).
    blocks: list[torch.Tensor]
    seconds: float


#: The return type of `body` of {@link LayerwiseDit.run}.
_Result = TypeVar("_Result")


@dataclass(frozen=True)
class _BlockHooks:
    """The hooks applied to each layer-by-layer block ({@link LayerwiseDit._opener})."""

    sink: list[torch.Tensor] | None
    on_block: Callable[[int, float], None] | None
    on_output: Callable[[int, torch.Tensor], None] | None
    prepare: Callable[[nn.Module], None] | None
    #: Indices of the blocks whose forward hook fired (in firing order — for the {@link watched}
    #: check).
    fired: list[int]

    @property
    def watched(self) -> bool:
        """Whether there is a hook that fires from the blocks' forward hook (`sink` / `on_block` /
        `on_output`)."""
        return self.sink is not None or self.on_block is not None or self.on_output is not None


def _unreached(index: int) -> DitReferenceError:
    """A hook that fires from the forward hook was given to a `body` that does not go through the
    blocks' forward."""
    return DitReferenceError(
        f"block {index}'s forward was not called — collect_blocks / on_block / on_output fire"
        " from the blocks' forward hook. Do not combine them with a body that calls the block's"
        " components directly without going through its forward (`_ti2v_block` of"
        " `dit_patch.WanDitTokensTi2v`)"
    )


class LayerwiseDit:
    """A CPU reference that runs upstream `WanTransformer3DModel` loading one block's weights at a
    time (module doc).

    Given `alignment` (key → offset — {@link upstream_alignment}), the f32 forward places the
    weights at the same offsets as upstream's `from_pretrained` before running (module doc "The f32
    reference and weight addresses").
    Without it, the source's values are used as they are (`ModuleWeights` is the upstream model's
    own addresses).
    """

    def __init__(
        self, config: Mapping[str, Any], *, alignment: Mapping[str, int] | None = None
    ) -> None:
        assert_no_mps()
        self._alignment = None if alignment is None else dict(alignment)
        from diffusers.models.transformers.transformer_wan import WanRotaryPosEmbed

        template = meta_model(config)
        self.config = template.config
        self._blocks = list(template.blocks)
        template.blocks = _LazyBlocks(len(self._blocks))
        # The RoPE table is a buffer determined by the config alone — built by running upstream's
        # initialization on CPU (the f32 table dropped from float64 on a machine without MPS — the
        # same values as `model.rope` of `from_pretrained`).
        template.rope = WanRotaryPosEmbed(
            int(self.config.attention_head_dim),
            tuple(self.config.patch_size),
            int(self.config.rope_max_seq_len),
        )
        self._outer = template
        self.outer_shapes = {name: list(p.shape) for name, p in template.named_parameters()}
        self.block_shapes = {name: list(p.shape) for name, p in self._blocks[0].named_parameters()}
        stray = [name for name, _ in self._blocks[0].named_buffers()]
        stray += [name for name, _ in template.named_buffers() if not name.startswith("rope.")]
        if stray:
            raise DitReferenceError(f"non-weight buffers {stray} exist (not loaded layer by layer)")

    @property
    def patch_size(self) -> tuple[int, int, int]:
        patch_t, patch_h, patch_w = (int(size) for size in self.config.patch_size)
        return patch_t, patch_h, patch_w

    @property
    def layers(self) -> int:
        return len(self._blocks)

    @property
    def rope(self) -> nn.Module:
        """Upstream's RoPE table (buffers built on CPU from the config alone — where the host table
        and the `rope_base` asset come from)."""
        return self._outer.rope

    def timesteps_proj(self, timestep: torch.Tensor) -> torch.Tensor:
        """The host's timestep sinusoid (upstream's `condition_embedder.timesteps_proj` — it has no
        weights)."""
        return dit_patch.dit_timesteps_proj(self._outer, timestep)

    def weight_keys(self) -> list[str]:
        """Every weight key that is read (upstream's state-dict spelling — outside the blocks + all
        blocks)."""
        inner = [
            f"blocks.{index}.{name}" for index in range(self.layers) for name in self.block_shapes
        ]
        return [*self.outer_shapes, *inner]

    def forward(
        self,
        source: WeightSource,
        dtype: torch.dtype,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        *,
        condition: dit_patch.TimestepCondition | None = None,
        collect_blocks: bool = False,
        on_block: Callable[[int, float], None] | None = None,
        on_output: Callable[[int, torch.Tensor], None] | None = None,
        blocks: int | None = None,
    ) -> DitReference:
        """Runs upstream's plain forward in `dtype` (f32 / f64) (the f32 input values are widened
        to `dtype`).

        f32 is called the same way as `dit_patch.reference_dit` and f64 as
        `dit_patch.reference_dit_f64` (attention pinned to the CPU flash path); the only difference
        is that the weights are loaded one block at a time. `condition` is the per-token timestep
        of TI2V (the reference wrapper of ADR 0121 decision 4 — both f32 and f64). `on_output`
        receives the blocks' outputs one at a time (a way to compare or write them without holding
        them — the per-block record at full size). `blocks` runs only the first N blocks (a
        truncation for measuring how the error grows with depth — the output is the head applied
        after block N, not a reference).
        """
        if dtype == torch.float64 and timestep.ndim != 1:
            raise DitReferenceError(
                f"2 次元の timestep（形 {list(timestep.shape)} — diffusers の M = S の経路）は"
                " f64 の参照にしない（観測用・f32 だけ）。トークンごとの時刻は ADR 0121 段 1 の"
                "参照ラッパ"
                "（`condition=` — 決定 4）で渡す（モジュール doc）"
            )
        wide = (lambda tensor: tensor) if dtype == torch.float32 else torch.Tensor.double
        output, collected, seconds = self.run(
            source,
            dtype,
            lambda model: dit_patch.reference_dit(
                model, wide(latents), timestep, wide(encoder_hidden_states), condition
            ),
            collect_blocks=collect_blocks,
            on_block=on_block,
            on_output=on_output,
            blocks=blocks,
        )
        return DitReference(output=output, blocks=collected, seconds=seconds)

    def run(
        self,
        source: WeightSource,
        dtype: torch.dtype,
        body: Callable[[nn.Module], _Result],
        *,
        collect_blocks: bool = False,
        on_block: Callable[[int, float], None] | None = None,
        on_output: Callable[[int, torch.Tensor], None] | None = None,
        prepare_block: Callable[[nn.Module], None] | None = None,
        blocks: int | None = None,
    ) -> tuple[_Result, list[torch.Tensor], float]:
        """Hands `body` the upstream model with the part outside the blocks materialized in
        `dtype` (`blocks` is the layer-by-layer sequence) and runs it (returns `body`'s return
        value, the collected block outputs and the elapsed seconds).

        {@link forward} passes upstream's plain forward as `body`. The eager of the patched graph
        builds the S-shaped wrapper in `body` and runs it (the wrapper is built with
        `install_processors=False` and `dit_patch.install_real_pair_processors` is passed as
        `prepare_block` — iterating the sequence loads weights, so it is not iterated while
        building). f64 runs with `.float()` passing through and under the f64 watch.

        `collect_blocks` collects every block output (only when `body` iterates the sequence once —
        fails loudly if the count differs from the number of blocks run). `blocks` truncates the
        sequence to the first N blocks (the doc of {@link forward}).

        MUST: `collect_blocks` / `on_block` / `on_output` fire from the blocks' **forward hook**.
        They do not reach a `body` that calls the block's components directly without going through
        its forward (the patched TI2V graph — `_ti2v_block` of `dit_patch.WanDitTokensTi2v`), so
        passing them fails loudly (it stops when the next block is requested and the previous
        block's forward was not called, and checks the last block at the end). It never silently
        returns an empty record.
        """
        if dtype not in (torch.float32, torch.float64):
            raise DitReferenceError(f"the reference dtype must be f32 / f64 (got {dtype})")
        count = self.layers if blocks is None else blocks
        if not 0 < count <= self.layers:
            raise DitReferenceError(f"回すブロックの数 {count} が 1〜{self.layers} の外")
        started = time.perf_counter()
        collected: list[torch.Tensor] = []
        with torch.no_grad():
            model = copy.deepcopy(self._outer)
            model.blocks.count = count
            state = {
                key: self._read(source, key, shape, dtype)
                for key, shape in self.outer_shapes.items()
            }
            model.load_state_dict(state, strict=True, assign=True)
            del state
            model.rope = model.rope.to(dtype)
            narrow = sorted(
                {str(tensor.dtype) for tensor in (*model.parameters(), *model.buffers())}
                - {str(dtype)}
            )
            if narrow:
                raise DitReferenceError(f"{narrow} weights / buffers remain outside the blocks")
            hooks = _BlockHooks(
                sink=collected if collect_blocks else None,
                on_block=on_block,
                on_output=on_output,
                prepare=prepare_block,
                fired=[],
            )
            if dtype == torch.float32:
                model.blocks.opener = self._opener(source, dtype, hooks, None)
                result = body(model)
            else:
                watch = _PausableWatch(
                    dit_patch._timestep_sinusoid_shapes(int(self.config.freq_dim))
                )
                model.blocks.opener = self._opener(source, dtype, hooks, watch)
                with dit_patch._float_keeps_float64(), watch:
                    result = body(model)
                if watch.found:
                    raise DitReferenceError(
                        f"f64 の参照の中で f64 でない値が作られた: {watch.found[:5]}"
                    )
        if hooks.watched and hooks.fired[-1:] != [count - 1]:
            raise _unreached(count - 1)
        if collect_blocks and len(collected) != count:
            raise DitReferenceError(f"collected {len(collected)} block outputs (expected {count})")
        return result, collected, time.perf_counter() - started

    def _opener(
        self,
        source: WeightSource,
        dtype: torch.dtype,
        hooks: _BlockHooks,
        watch: _PausableWatch | None,
    ) -> Callable[[int], nn.Module]:
        """A function that materializes block `index` (weight loading is outside the f64 watch).

        Block outputs are copied only when `hooks.sink` is set (holding 30 of them at S = 32,760 is
        about 6 GB in f32), and are passed to `hooks.on_output` without a copy. `hooks.on_block`
        receives the time of one block including its loading (for a one-line progress report).
        `hooks.prepare` is applied once to the materialized block.

        MUST: release the previous block's weights before loading the next. The loop variable of
        upstream's `for block in self.blocks` keeps the previous block while asking for the next,
        so without releasing, two blocks are alive at once (1.22 → 2.44 GiB in f64 for 5B). The
        release happens after the previous block's forward has finished (when the next is
        requested), so the values do not change.
        """
        opened: list[nn.Module] = []

        def open_block(index: int) -> nn.Module:
            if hooks.watched and index > 0 and hooks.fired[-1:] != [index - 1]:
                raise _unreached(index - 1)
            started = time.perf_counter()
            with watch.paused() if watch is not None else nullcontext():
                while opened:
                    opened.pop().to_empty(device="meta")
                block = copy.deepcopy(self._blocks[index])
                state = {
                    name: self._read(source, f"blocks.{index}.{name}", shape, dtype)
                    for name, shape in self.block_shapes.items()
                }
                block.load_state_dict(state, strict=True, assign=True)
                del state
            wrong = sorted({str(p.dtype) for p in block.parameters()} - {str(dtype)})
            if wrong:
                raise DitReferenceError(f"block {index} still has {wrong} weights")
            if hooks.prepare is not None:
                hooks.prepare(block)

            def after(_module: nn.Module, _inputs: Any, output: torch.Tensor) -> None:
                hooks.fired.append(index)
                if hooks.sink is not None:
                    hooks.sink.append(output.detach().clone())
                if hooks.on_output is not None:
                    # The receiver's computation (comparison with the f32 record and so on) is
                    # outside the f64 path — pause the watch.
                    with watch.paused() if watch is not None else nullcontext():
                        hooks.on_output(index, output.detach())
                if hooks.on_block is not None:
                    hooks.on_block(index, time.perf_counter() - started)

            block.register_forward_hook(after)
            opened.append(block)
            return block

        return open_block

    def _read(
        self, source: WeightSource, key: str, shape: list[int], dtype: torch.dtype
    ) -> torch.Tensor:
        """The source's value in upstream's shape and `dtype` (sources return f32 only; only the
        patch embedding gets its Linear form reshaped back). In f32 with `alignment`, it is placed
        at upstream's offset."""
        value = source.tensor(key)
        if value.dtype != torch.float32:
            raise DitReferenceError(f"'{key}' is {value.dtype} (sources return f32)")
        if list(value.shape) != shape:
            if key != PATCH_EMBEDDING_WEIGHT or value.numel() != math.prod(shape):
                raise DitReferenceError(
                    f"'{key}': shape {list(value.shape)} differs from upstream's {shape}"
                )
            value = value.reshape(shape)
        if dtype != torch.float32:
            return value.to(dtype)
        if self._alignment is None:
            return value
        offset = self._alignment.get(key)
        if offset is None:
            raise DitReferenceError(f"'{key}' の上流のずれが無い（alignment の表の外）")
        return _placed(value, offset)


# ---------------------------------------------------------------------------
# Comparison with the existing 1.3B goldens
# ---------------------------------------------------------------------------


def _tensor_bits(tensor: torch.Tensor) -> torch.Tensor:
    """For comparing as bit strings (-0.0 vs 0.0 and NaN are not left to numeric equality)."""
    return tensor.detach().contiguous().flatten().view(torch.uint8)


def compare_tensors(
    actual: Mapping[str, torch.Tensor], expected_path: Path, keys: Sequence[str] | None = None
) -> dict[str, Any]:
    """Compares the written golden (`actual`) with the existing `reference.<case>` bitwise, tensor
    by tensor.

    With `keys` only those keys are compared (`--f64-only`). Names the keys that are missing from
    the existing file, differ in shape / dtype, or differ in bits.
    """
    compared = sorted(actual) if keys is None else sorted(keys)
    mismatched: dict[str, str] = {}
    with safe_open(str(expected_path), framework="pt") as handle:
        present = set(handle.keys())
        wanted = set(compared) if keys is not None else set(compared) | present
        for key in sorted(wanted):
            if key not in actual:
                mismatched[key] = "書いた側に無い"
                continue
            if key not in present:
                mismatched[key] = "既存に無い"
                continue
            mine, theirs = actual[key], handle.get_tensor(key)
            if mine.dtype != theirs.dtype or list(mine.shape) != list(theirs.shape):
                mismatched[key] = (
                    f"{mine.dtype} {list(mine.shape)} / existing {theirs.dtype}"
                    f" {list(theirs.shape)}"
                )
            elif not torch.equal(_tensor_bits(mine), _tensor_bits(theirs)):
                difference = (mine.double() - theirs.double()).abs().max()
                mismatched[key] = f"ビットが違う（最大絶対差 {float(difference):.3e}）"
    return {"keys": sorted(wanted), "mismatched": mismatched}


def file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def numeric_environment() -> dict[str, Any]:
    """The execution environment the f32 bit-exactness depends on (module doc "The f32 reference
    and weight addresses" — MKL behavior rather than a spec guarantee, so it is kept in the result
    rows for later reading)."""
    return {
        "torch": torch.__version__,
        "threads": torch.get_num_threads(),
        "cpu_capability": torch.backends.cpu.get_cpu_capability(),
        "env": {
            name: value
            for name, value in sorted(os.environ.items())
            if name.startswith(_NUMERIC_ENV_PREFIXES)
        },
    }


def _done(results: Path, mode: CompareMode) -> dict[str, dict[str, Any]]:
    """The result rows (resume markers) — case name → row for the cases this run counts as "done".

    Only rows whose version equals {@link COMPARE_VERSION} and whose mode covers this run
    ({@link _COVERING_MODES}) are counted. A half-written last row from a crash (no trailing line
    break — a row is written up to its line break in one write) is dropped and also cut from the
    file, and that case is run again. A row that is not the last one and cannot be read is not a
    half-written row, so it fails loudly.
    """
    if not results.is_file():
        return {}
    lines = results.read_text(encoding="utf-8").split("\n")
    partial = lines.pop()
    if partial:
        print(
            f"[compare] {results}: dropping the half-written last row ({len(partial)} characters)"
            " — that case runs again",
            flush=True,
        )
        results.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")
    done: dict[str, dict[str, Any]] = {}
    stale = 0
    for number, line in enumerate(lines, start=1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise DitReferenceError(f"{results}:{number} の行が読めない（末尾ではない）") from error
        if (
            row.get("compare_version") == COMPARE_VERSION
            and row.get("mode") in (_COVERING_MODES[mode])
        ):
            done[row["case"]] = row
        else:
            stale += 1
    if stale:
        print(
            f"[compare] {results}: rows whose comparison version ({COMPARE_VERSION}) or run mode"
            f" ({mode}) does not match are not counted as done ({stale} rows)",
            flush=True,
        )
    return done


def _append_row(results: Path, row: Mapping[str, Any]) -> None:
    """Writes a result row up to its line break in one write and flushes it to disk (after a crash
    only the last row can be partial)."""
    with results.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def compare_series(
    dtype: str,
    *,
    source_kind: Literal["container", "module"] = "container",
    names: Sequence[str] = (),
    max_tokens: int | None = None,
    f64_only: bool = False,
    out: Path = DEFAULT_OUT,
) -> dict[str, Any]:
    """Re-takes the existing goldens of a 1.3B series (`dtype` = f16 / i8) layer by layer and
    compares them bitwise.

    It writes only under `out/<series name>-<source_kind>/` (the existing series is only read).
    Cases come in the order of `wan.export_dit.series_cases(dtype)` and can be narrowed with
    `names` / `max_tokens`.
    """
    from wan import export_dit
    from wan.sources import DEFAULT_MODEL, local_snapshot

    existing = export_dit.series_dir(dtype) / export_dit.TARGET
    target = out / f"{existing.parent.name}-{source_kind}"
    target.mkdir(parents=True, exist_ok=True)
    results = target / RESULTS_FILE
    mode: CompareMode = "f64-only" if f64_only else "full"
    done = _done(results, mode)
    environment = numeric_environment()
    upstream = local_snapshot(DEFAULT_MODEL) / TRANSFORMER_SUBFOLDER
    # The container's weights are not at upstream's mmap addresses — run the f32 reference with the
    # same offsets as upstream's eager (module doc). The module source is the upstream model's own
    # addresses, so it is not re-placed.
    alignment = upstream_alignment(upstream) if source_kind == "container" else None
    writer = LayerwiseDit(load_config(upstream), alignment=alignment)
    known = {spec.name(writer.patch_size) for spec in export_dit.series_cases(dtype)}
    if set(names) - known:
        raise DitReferenceError(f"unknown case names: {sorted(set(names) - known)}")
    specs = [
        spec
        for spec in export_dit.series_cases(dtype)
        if (not names or spec.name(writer.patch_size) in names)
        and (max_tokens is None or _tokens(spec, writer.patch_size) <= max_tokens)
        # `--f64-only` covers only cases with an f64 reference (the small-S cases of the f16
        # series have only the f32 reference).
        and (not f64_only or export_dit.wants_float64(dtype, spec))
    ]
    source = _source(source_kind, dtype, existing)
    finished = sum(spec.name(writer.patch_size) in done for spec in specs)
    print(
        f"[compare] {existing}: {len(specs)} cases ({finished} done)"
        f" → {target} (source {source_kind}, mode {mode}, environment {json.dumps(environment)})",
        flush=True,
    )
    for position, spec in enumerate(specs, start=1):
        name = spec.name(writer.patch_size)
        if name in done:
            print(f"[{position}/{len(specs)}] {name}: done ({done[name]['status']})", flush=True)
            continue
        row = {
            **_compare_case(writer, source, dtype, spec, existing, target, f64_only=f64_only),
            "mode": mode,
            "compare_version": COMPARE_VERSION,
            "environment": environment,
        }
        _append_row(results, row)
        done[name] = row
        print(f"[{position}/{len(specs)}] {json.dumps(row, ensure_ascii=False)}", flush=True)
    chosen = [done[spec.name(writer.patch_size)] for spec in specs]
    environments = []
    for row in chosen:
        if row["environment"] not in environments:
            environments.append(row["environment"])
    summary = {
        "series": str(existing),
        "out": str(target),
        "mode": mode,
        "compare_version": COMPARE_VERSION,
        "cases": len(chosen),
        "equal": sum(row["status"] == "equal" for row in chosen),
        "differ": [row["case"] for row in chosen if row["status"] != "equal"],
        # The environments the rows were taken in (more than one across resumes — f32 bits depend
        # on the environment).
        "environments": environments,
    }
    (target / SUMMARY_FILE).write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    return summary


#: The minimum S that prints per-block progress rows (full-size cases — one forward takes minutes).
PROGRESS_TOKENS = 4096


def _block_progress(name: str, label: str, layers: int) -> Callable[[int, float], None]:
    def report(index: int, seconds: float) -> None:
        print(f"[block] {name} {label} {index + 1}/{layers} {seconds:.1f} s", flush=True)

    return report


def _tokens(spec: Any, patch_size: tuple[int, int, int]) -> int:
    frames, height, width = spec.latent_shape
    return (frames // patch_size[0]) * (height // patch_size[1]) * (width // patch_size[2])


def _source(kind: str, dtype: str, existing: Path) -> WeightSource:
    """The comparison's weight source (container = the same path as 5B, module = the 1.3B path of
    `export_dit`)."""
    from wan import export_dit

    if kind == "container":
        return ContainerDitWeights(existing / export_dit.MODEL_FILE)
    model = export_dit.load_transformer(export_dit.DEFAULT_MODEL)
    export_dit.fake_quant(dtype, model, dit_patch.WanDitTokens(model))
    return ModuleWeights(model)


def _compare_case(
    writer: LayerwiseDit,
    source: WeightSource,
    dtype: str,
    spec: Any,
    existing: Path,
    target: Path,
    *,
    f64_only: bool,
) -> dict[str, Any]:
    """Re-takes and writes one case, and returns the result row of the comparison with the
    existing golden."""
    from wan import export_dit

    name = spec.name(writer.patch_size)
    latents, timestep, encoder_hidden_states = export_dit.case_inputs(writer, spec)
    long_case = _tokens(spec, writer.patch_size) >= PROGRESS_TOKENS
    seconds: dict[str, float] = {}
    reference32 = None
    if not f64_only:
        reference32 = writer.forward(
            source,
            torch.float32,
            latents,
            timestep,
            encoder_hidden_states,
            collect_blocks=spec.blocks,
            on_block=_block_progress(name, "f32", writer.layers) if long_case else None,
        )
        seconds["f32"] = round(reference32.seconds, 1)
    reference64 = None
    if export_dit.wants_float64(dtype, spec):
        reference64 = writer.forward(
            source,
            torch.float64,
            latents,
            timestep,
            encoder_hidden_states,
            on_block=_block_progress(name, "f64", writer.layers) if long_case else None,
        )
        seconds["f64"] = round(reference64.seconds, 1)
    if reference32 is None and reference64 is None:
        raise DitReferenceError(f"{name}: --f64-only on a case without an f64 reference")
    tensors = export_dit.reference_tensors(
        name,
        latents=latents,
        timestep=timestep,
        output=None if reference32 is None else reference32.output,
        blocks=[] if reference32 is None else reference32.blocks,
        output_f64=None if reference64 is None else reference64.output,
    )
    filename = f"{export_dit.REFERENCE_PREFIX}{name}{export_dit.CASE_SUFFIX}"
    save_file(tensors, str(target / filename))
    keys = [export_dit.REFERENCE_F64_KEY] if f64_only else None
    comparison = compare_tensors(tensors, existing / filename, keys)
    return {
        "case": name,
        "status": "differ" if comparison["mismatched"] else "equal",
        "compared": len(comparison["keys"]),
        "mismatched": comparison["mismatched"],
        "file_sha256_equal": file_sha256(target / filename) == file_sha256(existing / filename),
        "seconds": seconds,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("command", choices=("compare",))
    parser.add_argument("--dtype", required=True, choices=("f16", "i8"), help="the 1.3B series")
    parser.add_argument(
        "--source",
        default="container",
        choices=("container", "module"),
        help="the weight source (container = the series container, module = the f32 model +"
        " fake-quant)",
    )
    parser.add_argument(
        "--case", action="append", default=[], help="narrow by case name (repeatable)"
    )
    parser.add_argument("--max-tokens", type=int, help="only cases with S up to this")
    parser.add_argument(
        "--f64-only",
        action="store_true",
        help="take and compare only the f64 reference (output.f64)",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    summary = compare_series(
        args.dtype,
        source_kind=args.source,
        names=args.case,
        max_tokens=args.max_tokens,
        f64_only=args.f64_only,
        out=args.out,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    return 0 if not summary["differ"] else 1


if __name__ == "__main__":
    sys.exit(main())
