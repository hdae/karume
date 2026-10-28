"""The per-layer CPU reference of the MOSS-Image DiT graphs — f64 and f32, weights streamed block by
block from the i8 series — and the goldens the TypeScript port checks read (ADR NNNN decision 11,
stage 2; the ADR 0121 decision 7 form of `wan.dit_reference`).

    uv run --group moss --inexact python -m moss.dit_reference write                # every case
    uv run --group moss --inexact python -m moss.dit_reference write --group small  # one group
    uv run --group moss --inexact python -m moss.dit_reference write --case <name>  # one case
    uv run --group moss --inexact python -m moss.dit_reference list

## What runs

Each case runs one graph's module of `moss.dit_patch` — `MossDit` (`dit`), `MossDit` without block
22 (`dit_skip22`) or `MossDitText` (`dit_text`) — twice on the CPU: in f32 (the denominator of the
normalised ratio r — ADR 0118 decision 8) and in f64 (activations f64 — the reference). These are
the modules step 1 exported, and their f32 eager form is bit-identical to upstream's
`MOSSImageTransformer.forward` (step 1's gate 2), so the reference is upstream's computation without
a transcription of it here. In f64, upstream's `.float()` casts keep f64 (they are identities in
the f32 model, there for bf16 runs) and any other non-f64 floating value fails loudly
({@link float64_forward}); the inputs (f32 values) are widened, which is exact.

## Weights — the dequantised values the GPU path sees

{@link ContainerWeights} reads the series containers (`dit`, `dit_text`): an i8 weight is packed ×
per-row scale **in f32** (what the GPU dequantizes and the fake-quant weight of step 1's gate 2,
whose container check proved the two bit-identical), an f32 tensor as stored. The f64 run widens
these f32 values (`.to(float64)` is exact). This follows `wan.dit_reference.ContainerDitWeights`
(ADR 0121 decision 7): dequantising in f64 would give other weights than the GPU's.
`dit_skip22` borrows every weight of `dit`, so its reference reads the `dit` container.

## Bounded RAM

The f64 DiT is 40.7 GB, so the 42 blocks are materialised one at a time ({@link _LazyLayers} —
the previous block's weights are released before the next is read; the loop variable of
`MossDit.forward` still holds the previous block when the next is asked for). Held at once: the
stored container bytes (4.73 GB of i8, read once per process), the modules outside the blocks
(0.69 GB in f64), one block (0.91 GB in f64) and the activations. `dit_text` (0.24 B parameters)
is materialised whole.

## Cases ({@link DIT_CASES}, {@link TEXT_CASES} — the TS side mirrors the table)

Bands follow ADR 0120 decision 3: per group, six band cases set the band (worst r × 5) and the
acceptance cases are judged against it, never the other way round.

- `small` (256², N = 256) and `full` (1024², N = 4,096): the positive branch (`dit`, prompt
  `fridge-note`, T = 63) and the negative branch (`dit_skip22`, the empty negative, T = 10), the
  band cases at the training-grid indices {@link BAND_INDICES} and the acceptance cases at
  {@link ACCEPT_INDICES}, for both branches. The two branches of one index share their latents
  (seed {@link SEED} + index — torch's CPU `randn`) and timestep, as the two forwards of a step do.
- `text` (`dit_text`): band cases at T = 2, 10, 31, 62, 63, 134 and acceptance cases at T = 1 (the
  MUST of ADR NNNN decision 3: the graph was traced with T ≥ 2, and no CPU path runs the IR at
  T = 1), 7, 45, 100 — the stage-0 embeddings and prefixes of them (a prefix of a causal
  encoder's hidden states is the encoder output of that token prefix).

The inputs of a DiT case are built by `moss.dit_patch`'s host functions from the series' own
assets: the RoPE table assembled from `rope_base`, the `timesteps_proj` row of the case's
training-grid index from `training_grid`, and as the text context the f32 `dit_text` output of the
case's prompt computed here (the same f32 value is fed to both sides, so a DiT case checks `dit`
alone; `dit_text` has its own cases).

## Goldens (`<case>.safetensors` in {@link REFERENCE_DIR})

Common: `output` (f32 run), the f64 run as two f32 tensors — `output.f64` (its f32 rounding) and
`output.f64.lo` (the f32 rounding of the rest, f64 − `output.f64`; {@link split_float64}) — and a
single JSON metadata key ({@link METADATA_KEY}: the case, the sha256 of each series container's
part 0 — `dit`, `dit_skip22` and `dit_text`; the TS checks refuse goldens of another series — the
environment, time and peak RSS).

WHY two parts: the f64 output rounded to f32 alone is off by up to half an f32 ULP. Where the f32
run is itself only 1–2 ULP off (`dit_text`, whose largest values sit in one outlier channel near
1,800), that rounding is as large as the denominator of r (on the `dit_text` cases it scaled the
denominator by 0.69–1.63). Summed in double, the two parts are the f64 value within 2⁻⁴⁸
relative.

- DiT: `latents [1,128,h,w]`, `text_context [1,T,3072]` (the refined text the inputs were built
  with), `grid_index` (I32), `timestep` (F32 — the training grid's value at that index), `output` /
  `output.f64` / `output.f64.lo` as upstream's sample `[1,128,h,w]` (`dit_patch.head_rows`) and
  `output.rows`, the f32 graph output `[1,S,128]` before the head rows are taken. The `small` cases
  also keep the six graph inputs `input.<name>` exactly as `dit_patch.dit_inputs` built them (the
  host functions' TS transcription is checked against them).
- `dit_text`: `text_embeds [1,T,4096]`, `rope_cos` / `rope_sin [1,1,T,128]` and `output` /
  `output.f64` / `output.f64.lo [1,T,3072]`.

A case whose golden exists with this {@link WRITER_VERSION} and the current containers is skipped
(the run resumes); every golden goes through a staging file. `summary.json` covers the cases of the
last `write` only (a `--group` or `--case` run lists just those); `results.jsonl` keeps one row per
golden ever written (the f32-vs-f64 ratio, time and peak RSS).

MUST: diffusers is imported inside the functions (`tests/test_optional_group_imports.py`).
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import load
from torch import nn
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_leaves

from _shared.container_read import StoredTensor, read_asset, read_stored
from _shared.paths import SERIES_ROOT
from karume.container import container_parts
from moss import dit_patch, export_dit, pipeline_ref, records, text_embeds
from moss.dit_patch import DIT, DIT_SKIP22, DIT_TEXT, SKIP_BLOCK
from moss.prompts import NEGATIVE, POSITIVE

#: Where the goldens go (next to the series they are taken from — docs/assets-layout.md; a
#: separate directory, so `export_dit write`, which replaces the series directory, keeps them).
REFERENCE_DIR = SERIES_ROOT / f"{export_dit.SERIES_NAME}-reference"

#: Bump when a golden's content or the case inputs change (a golden of another version is redone).
#: 2: `output.f64.lo` and the `dit_skip22` container's digest.
WRITER_VERSION = 2

#: The single metadata key of the goldens (`moss.records`).
METADATA_KEY = "karume.moss.dit_reference"

#: The summary of the last `write`, and the log of every golden written (one JSON row each).
SUMMARY_FILE = "summary.json"
RESULTS_FILE = "results.jsonl"

#: The seed of a DiT case's latents is `SEED + grid_index` (both branches of one index share them).
SEED = 20261027

#: The prompt of the positive branch (the model card's example; T = 63).
POSITIVE_PROMPT = "fridge-note"

#: The embedding the negative branch takes (the empty negative prompt; T = 10).
NEGATIVE_PROMPT = "negative"

#: The DiT groups: name → (height, width) in pixels.
DIT_GROUPS: Mapping[str, tuple[int, int]] = {"small": (256, 256), "full": (1024, 1024)}

#: Training-grid indices (0 = σ 1, timestep 1000; 999 = the smallest σ) of the band and acceptance
#: cases (disjoint — ADR 0120 decision 3).
BAND_INDICES: tuple[int, ...] = (0, 400, 850)
ACCEPT_INDICES: tuple[int, ...] = (200, 950)

#: The modules of `MossDit` outside the blocks (materialised once per run).
OUTER_MODULES = (
    "video_embedder",
    "time_embedding",
    "shared_block_modulation",
    "final_norm_video",
    "final_linear_video",
    "video_head_modulation",
)

BAND = "band"
ACCEPT = "accept"


class DitReferenceError(RuntimeError):
    """A weight, an input or a case left the form the reference accepts."""


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DitCase:
    """One forward of `dit` (positive) or `dit_skip22` (negative)."""

    group: str
    role: str
    branch: str
    grid_index: int

    @property
    def name(self) -> str:
        return f"{self.group}-{self.role}-{self.branch}-g{self.grid_index:04d}"

    @property
    def graph(self) -> str:
        return DIT if self.branch == POSITIVE else DIT_SKIP22

    @property
    def prompt(self) -> str:
        return POSITIVE_PROMPT if self.branch == POSITIVE else NEGATIVE_PROMPT

    @property
    def size(self) -> tuple[int, int]:
        return DIT_GROUPS[self.group]

    @property
    def seed(self) -> int:
        return SEED + self.grid_index


@dataclass(frozen=True)
class TextCase:
    """One forward of `dit_text` on the first `length` rows of a stage-0 embedding."""

    role: str
    prompt: str
    length: int

    group = "text"

    @property
    def name(self) -> str:
        return f"text-{self.role}-{self.prompt}-t{self.length:04d}"

    @property
    def graph(self) -> str:
        return DIT_TEXT


def _dit_cases(group: str) -> tuple[DitCase, ...]:
    return tuple(
        DitCase(group, role, branch, index)
        for role, indices in ((BAND, BAND_INDICES), (ACCEPT, ACCEPT_INDICES))
        for branch in (POSITIVE, NEGATIVE)
        for index in indices
    )


#: Six band and four acceptance cases per DiT group (module docstring).
DIT_CASES: tuple[DitCase, ...] = (*_dit_cases("small"), *_dit_cases("full"))

#: Six band and four acceptance cases of `dit_text` (module docstring).
TEXT_CASES: tuple[TextCase, ...] = (
    TextCase(BAND, "guide-whiteboard", 2),
    TextCase(BAND, "negative", 10),
    TextCase(BAND, "fridge-note", 31),
    TextCase(BAND, "guide-fridge-note", 62),
    TextCase(BAND, "fridge-note", 63),
    TextCase(BAND, "guide-whiteboard", 134),
    TextCase(ACCEPT, "fridge-note", 1),
    TextCase(ACCEPT, "negative", 7),
    TextCase(ACCEPT, "guide-fridge-note", 45),
    TextCase(ACCEPT, "guide-whiteboard", 100),
)

Case = DitCase | TextCase

#: Every case in run order (the cheap groups first).
CASES: tuple[Case, ...] = (
    *TEXT_CASES,
    *(case for case in DIT_CASES if case.group == "small"),
    *(case for case in DIT_CASES if case.group == "full"),
)

GROUPS = ("text", *DIT_GROUPS)


# ---------------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------------


def _floats(raw: bytes) -> torch.Tensor:
    return torch.frombuffer(bytearray(raw), dtype=torch.float32)


def part0_sha256(container: Path) -> str:
    """The sha256 of a container's part 0 (the two documents — they name every block's sha256, so
    this names the container's whole content; the TS checks match it to their container)."""
    with container_parts(container)[0].open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class ContainerWeights:
    """A series container's weights by tensor key, as f32 (module docstring "Weights").

    The stored bytes are read once (every block's sha256 is checked against the container's
    declaration on the way — `ReadContainer.block`) and decoded one tensor at a time. MUST: only
    f32 and per-row i8 (one scale per row of a rank-2 weight) are accepted — another storage would
    make the reference silently run other weights.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._stored: Mapping[str, StoredTensor] = read_stored(path)
        #: {@link part0_sha256} of the container.
        self.part0_sha256 = part0_sha256(path)

    def names(self) -> frozenset[str]:
        """The tensor keys the container stores."""
        return frozenset(self._stored)

    def tensor(self, key: str) -> torch.Tensor:
        stored = self._stored.get(key)
        if stored is None:
            raise DitReferenceError(f"{self.path}: no tensor '{key}'")
        shape = list(stored.shape)
        if stored.layout == "f32":
            return _floats(stored.payload).reshape(shape)
        if stored.layout != "i8":
            raise DitReferenceError(f"{self.path}: '{key}' is stored as {stored.layout}")
        if len(shape) != 2 or stored.scale is None or stored.group_size != shape[1]:
            raise DitReferenceError(
                f"{self.path}: '{key}' is not per-row i8 (shape {shape}, group size"
                f" {stored.group_size}, scale {'present' if stored.scale else 'missing'})"
            )
        scale = _floats(stored.scale)
        if scale.numel() != shape[0]:
            raise DitReferenceError(f"{self.path}: '{key}' has {scale.numel()} scales")
        packed = torch.frombuffer(bytearray(stored.payload), dtype=torch.int8).reshape(shape)
        return packed.to(torch.float32) * scale.reshape(-1, 1)


# ---------------------------------------------------------------------------
# f64 without narrowing
# ---------------------------------------------------------------------------


class _NarrowFloatWatch(TorchDispatchMode):
    """Records every floating value that is not f64 created inside an f64 run (paused while
    weights are read — their f32 decode is outside the f64 computation)."""

    def __init__(self) -> None:
        super().__init__()
        self.found: list[str] = []
        self._paused = False

    def __torch_dispatch__(
        self,
        func: Any,
        types: Any,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        result = func(*args, **(kwargs or {}))
        if self._paused:
            return result
        for leaf in tree_leaves(result):
            if (
                isinstance(leaf, torch.Tensor)
                and leaf.is_floating_point()
                and leaf.dtype != torch.float64
            ):
                self.found.append(f"{func}: {leaf.dtype} {tuple(leaf.shape)}")
        return result

    @contextmanager
    def paused(self) -> Iterator[None]:
        self._paused = True
        try:
            yield
        finally:
            self._paused = False


@contextmanager
def _float_keeps_float64() -> Iterator[None]:
    """`Tensor.float()` returns an f64 tensor unchanged for the duration (other dtypes as before).

    WHY: upstream casts with `.float()` in `_RMSNorm` and on the time embedding so that bf16 runs
    normalise in f32; in the f32 model these are identities, but in an f64 model they would drop
    the norms to f32 and the reference would not be "activations in f64".
    """
    original = vars(torch.Tensor).get("float")
    narrow = torch.Tensor.float

    def keep_float64(tensor: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        if tensor.dtype == torch.float64:
            return tensor
        return narrow(tensor, *args, **kwargs)

    torch.Tensor.float = keep_float64  # type: ignore[method-assign]
    try:
        yield
    finally:
        if original is None:
            del torch.Tensor.float
        else:
            torch.Tensor.float = original  # type: ignore[method-assign]


@contextmanager
def float64_forward() -> Iterator[_NarrowFloatWatch]:
    """The f64 run's guard (module docstring): `.float()` keeps f64, and a non-f64 floating value
    created inside fails loudly when the block exits."""
    watch = _NarrowFloatWatch()
    with _float_keeps_float64(), watch:
        yield watch
    if watch.found:
        raise DitReferenceError(f"a non-f64 value was created in the f64 run: {watch.found[:5]}")


# ---------------------------------------------------------------------------
# The layer-by-layer modules
# ---------------------------------------------------------------------------


class _LazyLayers(nn.Module):
    """`MossDit.layers` that materialises one block per step of `values()` (module docstring
    "Bounded RAM"). It registers no child modules (the meta templates stay outside).

    MUST: the previous block's weights are released before the next one is read — the caller's
    loop variable still references the previous block when the next is requested.
    """

    def __init__(
        self,
        opener: Callable[[int], nn.Module],
        indices: Sequence[int],
        outside: Callable[[], Any] = nullcontext,
    ) -> None:
        super().__init__()
        self._opener = opener
        #: The context in which weights are read and released (the f64 watch paused).
        self._outside = outside
        self.indices = tuple(indices)
        #: The indices opened, in order (the tests check the order and the skip).
        self.opened: list[int] = []
        #: The most blocks with real weights alive at once (the tests check it is 1).
        self.peak_live = 0
        self._live: list[nn.Module] = []

    def _release(self) -> None:
        with self._outside():
            while self._live:
                self._live.pop().to_empty(device="meta")

    def values(self) -> Iterator[nn.Module]:
        for index in self.indices:
            self._release()
            block = self._opener(index)
            self._live.append(block)
            self.opened.append(index)
            self.peak_live = max(self.peak_live, len(self._live))
            yield block
        self._release()


class LayerwiseMoss:
    """The graphs' modules on a weightless upstream model, filled from the series weights per run
    (module docstring)."""

    def __init__(
        self,
        module: ModuleType,
        config: Mapping[str, Any],
        dit_weights: ContainerWeights,
        text_weights: ContainerWeights,
    ) -> None:
        self.config = dict(config)
        self._template = export_dit.meta_model(module, config)
        self._dit_weights = dit_weights
        self._text_weights = text_weights
        #: The last run's `_LazyLayers` (the tests read what it opened).
        self.last_layers: _LazyLayers | None = None

    @staticmethod
    def _outside(watch: _NarrowFloatWatch | None) -> Any:
        return watch.paused() if watch is not None else nullcontext()

    def _load(
        self,
        target: nn.Module,
        prefix: str,
        weights: ContainerWeights,
        dtype: torch.dtype,
        watch: _NarrowFloatWatch | None,
    ) -> None:
        with self._outside(watch):
            state = {
                name: weights.tensor(f"{prefix}{name}").to(dtype)
                for name, _ in target.named_parameters()
            }
            target.load_state_dict(state, strict=True, assign=True)
        wrong = sorted({str(p.dtype) for p in target.parameters()} - {str(dtype)})
        if wrong:
            raise DitReferenceError(f"'{prefix}' has {wrong} parameters (expected {dtype})")

    def text(self, dtype: torch.dtype, inputs: Sequence[torch.Tensor]) -> torch.Tensor:
        """`dit_text` on its three inputs (`text_embeds`, `rope_cos`, `rope_sin`; f32 values)."""
        model = copy.deepcopy(self._template)
        graph = dit_patch.MossDitText(model)
        with torch.no_grad(), self._guard(dtype) as watch:
            self._load(graph.text_embedder, "text_embedder.", self._text_weights, dtype, watch)
            return graph(*(value.to(dtype) for value in inputs))

    def dit(
        self, dtype: torch.dtype, inputs: Sequence[torch.Tensor], *, skip: bool
    ) -> torch.Tensor:
        """`dit` (or `dit_skip22` with `skip`) on its six inputs (`dit_patch.DIT_INPUTS` order;
        float inputs are widened to `dtype`, `is_image` stays bool)."""
        model = copy.deepcopy(self._template)
        skip_layers = frozenset({SKIP_BLOCK}) if skip else frozenset()
        graph = dit_patch.MossDit(model, skip_layers=skip_layers)
        templates = {int(index): layer for index, layer in graph.layers.items()}
        with torch.no_grad(), self._guard(dtype) as watch:
            for name in OUTER_MODULES:
                self._load(getattr(graph, name), f"{name}.", self._dit_weights, dtype, watch)

            def open_block(index: int) -> nn.Module:
                with self._outside(watch):
                    block = copy.deepcopy(templates[index])
                self._load(block, f"layers.{index}.", self._dit_weights, dtype, watch)
                return block

            layers = _LazyLayers(open_block, sorted(templates), lambda: self._outside(watch))
            graph.layers = layers
            self.last_layers = layers
            widened = [value.to(dtype) if value.is_floating_point() else value for value in inputs]
            return graph(*widened)

    @contextmanager
    def _guard(self, dtype: torch.dtype) -> Iterator[_NarrowFloatWatch | None]:
        if dtype == torch.float32:
            yield None
        elif dtype == torch.float64:
            with float64_forward() as watch:
                yield watch
        else:
            raise DitReferenceError(f"the reference runs in f32 or f64 ({dtype})")


# ---------------------------------------------------------------------------
# Inputs and goldens
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SeriesAssets:
    """The `dit` container's assets (`rope_base`, `training_grid`)."""

    rope_base: Mapping[str, torch.Tensor]
    training_grid: Mapping[str, torch.Tensor]


def series_assets(dit_container: Path) -> SeriesAssets:
    return SeriesAssets(
        rope_base=load(read_asset(dit_container, export_dit.ROPE_BASE_ASSET)),
        training_grid=load(read_asset(dit_container, export_dit.TRAINING_GRID_ASSET)),
    )


def text_inputs(
    assets: SeriesAssets, embeds: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """`dit_text`'s inputs for `[T, text_dim]` embeddings (the text rows of the RoPE table — the
    same rows `moss.export_dit`'s gate 2 feeds it)."""
    length = int(embeds.shape[0])
    table = dit_patch.rope_table(assets.rope_base, 1, 1, length)[:, 1:]
    rope_cos, rope_sin = dit_patch.split_rope(table)
    return embeds[None].contiguous(), rope_cos, rope_sin


def dit_inputs(
    assets: SeriesAssets, latents: torch.Tensor, text_context: torch.Tensor, grid_index: int
) -> tuple[torch.Tensor, ...]:
    """`dit` / `dit_skip22`'s inputs (`dit_patch.dit_inputs`) from the series assets."""
    height, width = int(latents.shape[2]), int(latents.shape[3])
    rope = dit_patch.rope_table(assets.rope_base, height, width, int(text_context.shape[1]))
    sinusoid = assets.training_grid["time_sinusoid"][grid_index : grid_index + 1].contiguous()
    return dit_patch.dit_inputs(latents, text_context, rope, sinusoid)


def split_float64(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """An f64 tensor as two f32 tensors whose double sum is the value within 2⁻⁴⁸ relative (module
    docstring "Goldens"): the f32 rounding and the f32 rounding of the rest (the rest, f64 − the
    rounding, is exact in f64)."""
    if value.dtype != torch.float64:
        raise DitReferenceError(f"split_float64 takes f64 ({value.dtype})")
    high = value.to(torch.float32)
    low = (value - high.to(torch.float64)).to(torch.float32)
    return high.contiguous(), low.contiguous()


def ratio(actual: torch.Tensor, reference: torch.Tensor) -> float:
    """max |actual − reference| ÷ max |reference| (in f64)."""
    reference = reference.double()
    return float((actual.double() - reference).abs().max() / reference.abs().max())


def _environment() -> dict[str, Any]:
    environment = records.environment()
    return {key: environment[key] for key in ("cpu_capability", "torch_threads", "threading_env")}


@dataclass(frozen=True)
class Golden:
    tensors: dict[str, torch.Tensor]
    metadata: dict[str, Any]


def text_golden(
    layerwise: LayerwiseMoss, assets: SeriesAssets, embeds: torch.Tensor, case: TextCase
) -> Golden:
    """One `dit_text` case (`embeds` = the prompt's full `[T, text_dim]` embedding)."""
    if not 1 <= case.length <= int(embeds.shape[0]):
        raise DitReferenceError(f"{case.name}: T = {case.length} outside 1 … {embeds.shape[0]}")
    inputs = text_inputs(assets, embeds[: case.length])
    started = time.perf_counter()
    output = layerwise.text(torch.float32, inputs)
    output64 = layerwise.text(torch.float64, inputs)
    high, low = split_float64(output64)
    tensors = {
        "text_embeds": inputs[0],
        "rope_cos": inputs[1],
        "rope_sin": inputs[2],
        "output": output.contiguous(),
        "output.f64": high,
        "output.f64.lo": low,
    }
    return Golden(
        tensors,
        {"T": case.length, "f32_ratio": ratio(output, output64), "seconds": _since(started)},
    )


def dit_golden(
    layerwise: LayerwiseMoss,
    assets: SeriesAssets,
    latents: torch.Tensor,
    text_context: torch.Tensor,
    case: DitCase,
    *,
    keep_inputs: bool,
) -> Golden:
    """One DiT case (`latents` `[1, C, h, w]`; `text_context` = the f32 `dit_text` output of the
    case's prompt)."""
    inputs = dit_inputs(assets, latents, text_context, case.grid_index)
    cells = (int(latents.shape[2]), int(latents.shape[3]))
    skip = case.graph == DIT_SKIP22
    started = time.perf_counter()
    rows = layerwise.dit(torch.float32, inputs, skip=skip)
    seconds32 = _since(started)
    rows64 = layerwise.dit(torch.float64, inputs, skip=skip)
    sample = dit_patch.head_rows(rows, *cells).contiguous()
    sample64 = dit_patch.head_rows(rows64, *cells).contiguous()
    timestep = assets.training_grid["timesteps"][case.grid_index : case.grid_index + 1]
    high, low = split_float64(sample64)
    tensors = {
        "latents": latents,
        "text_context": text_context.contiguous(),
        "grid_index": torch.tensor([case.grid_index], dtype=torch.int32),
        "timestep": timestep.contiguous(),
        "output": sample,
        "output.f64": high,
        "output.f64.lo": low,
        "output.rows": rows.contiguous(),
    }
    if keep_inputs:
        tensors.update(
            {
                f"input.{name}": value.contiguous()
                for name, value in zip(dit_patch.DIT_INPUTS, inputs, strict=True)
            }
        )
    return Golden(
        tensors,
        {
            "N": cells[0] * cells[1],
            "T": int(text_context.shape[1]),
            "S": int(rows.shape[1]),
            "height": cells[0] * pipeline_ref.PIXELS_PER_CELL,
            "width": cells[1] * pipeline_ref.PIXELS_PER_CELL,
            "seed": case.seed,
            "grid_index": case.grid_index,
            "timestep": float(timestep),
            "f32_ratio": ratio(sample, sample64),
            "seconds": {"f32": seconds32, "total": _since(started)},
        },
    )


def _since(started: float) -> float:
    return round(time.perf_counter() - started, 1)


# ---------------------------------------------------------------------------
# The writer
# ---------------------------------------------------------------------------


def golden_path(directory: Path, case: Case) -> Path:
    return directory / f"{case.name}.safetensors"


def _current(path: Path, containers: Mapping[str, str]) -> bool:
    """A golden of this writer version, taken from these containers, exists."""
    if not path.is_file():
        return False
    with safe_open(str(path), framework="pt") as handle:
        raw = (handle.metadata() or {}).get(METADATA_KEY)
    if raw is None:
        return False
    metadata = json.loads(raw)
    return metadata.get("version") == WRITER_VERSION and metadata.get("containers") == containers


def case_latents(case: DitCase) -> torch.Tensor:
    """A DiT case's latents: torch's CPU `randn` from seed `SEED + grid_index` at the group's size
    (`moss.pipeline_ref.seeded_latents` — the stage-0 references' noise)."""
    return pipeline_ref.seeded_latents(case.seed, *case.size)


@dataclass
class Writer:
    """The cases' shared state: the layer-wise modules, the series assets, the stage-0 embeddings,
    the containers' part-0 sha256 and the refined text per prompt (computed once)."""

    layerwise: LayerwiseMoss
    assets: SeriesAssets
    embeds: Mapping[str, torch.Tensor]
    containers: dict[str, str]
    refined: dict[str, torch.Tensor]
    latents: Callable[[DitCase], torch.Tensor] = case_latents

    def text_context(self, prompt: str) -> torch.Tensor:
        """The f32 `dit_text` output of a prompt's whole embedding (`[1, T, dim]`)."""
        if prompt not in self.refined:
            inputs = text_inputs(self.assets, self.embeds[prompt])
            self.refined[prompt] = self.layerwise.text(torch.float32, inputs)
        return self.refined[prompt]

    def golden(self, case: Case) -> Golden:
        if isinstance(case, TextCase):
            return text_golden(self.layerwise, self.assets, self.embeds[case.prompt], case)
        return dit_golden(
            self.layerwise,
            self.assets,
            self.latents(case),
            self.text_context(case.prompt),
            case,
            keep_inputs=case.group == "small",
        )


def open_writer(
    series: Path, embeds: Mapping[str, torch.Tensor], module: ModuleType, config: Mapping[str, Any]
) -> Writer:
    """The writer over a series directory (`export_dit.container_paths`) and the embeddings."""
    paths = export_dit.container_paths(series)
    dit_weights = ContainerWeights(paths[DIT])
    text_weights = ContainerWeights(paths[DIT_TEXT])
    return Writer(
        layerwise=LayerwiseMoss(module, config, dit_weights, text_weights),
        assets=series_assets(paths[DIT]),
        embeds=embeds,
        containers={
            DIT: dit_weights.part0_sha256,
            DIT_SKIP22: part0_sha256(paths[DIT_SKIP22]),
            DIT_TEXT: text_weights.part0_sha256,
        },
        refined={},
    )


def select(cases: Sequence[Case], groups: Sequence[str], names: Sequence[str]) -> list[Case]:
    known = {case.name for case in cases}
    unknown = sorted(set(names) - known)
    if unknown:
        raise DitReferenceError(f"unknown cases {unknown}")
    return [
        case
        for case in cases
        if (not groups or case.group in groups) and (not names or case.name in names)
    ]


def real_writer() -> Writer:
    """The writer over the written series and the stage-0 embedding asset (pinned upstream)."""
    snapshot, module = pipeline_ref.upstream()
    embeds, _metadata = text_embeds.read_asset(
        SERIES_ROOT / text_embeds.SERIES_NAME / text_embeds.ASSET_NAME
    )
    return open_writer(export_dit.SERIES, embeds, module, export_dit.load_config(snapshot))


def write(cases: Sequence[Case], *, out: Path, writer: Writer) -> dict[str, Any]:
    """Write the goldens of `cases` that are not current (module docstring "Goldens")."""
    clock = records.Clock()
    out.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for position, case in enumerate(cases, start=1):
        path = golden_path(out, case)
        if _current(path, writer.containers):
            print(f"[{position}/{len(cases)}] {case.name}: current", flush=True)
            rows.append({"case": case.name, "status": "current"})
            continue
        golden = writer.golden(case)
        metadata = {
            "version": WRITER_VERSION,
            "case": case.name,
            "graph": case.graph,
            "group": case.group,
            "role": case.role,
            "prompt": case.prompt,
            "containers": writer.containers,
            "environment": _environment(),
            "peak_rss_gib": round(records.peak_rss_gib(), 2),
            **golden.metadata,
        }
        records.write_record(path, golden.tensors, metadata, key=METADATA_KEY)
        row = {
            "version": WRITER_VERSION,
            "case": case.name,
            "status": "written",
            "bytes": path.stat().st_size,
            "f32_ratio": golden.metadata["f32_ratio"],
            "seconds": golden.metadata["seconds"],
            "peak_rss_gib": metadata["peak_rss_gib"],
        }
        rows.append(row)
        with (out / RESULTS_FILE).open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row) + "\n")
        clock.lap(case.name)
        print(f"[{position}/{len(cases)}] {json.dumps(row)}", flush=True)
    summary = {
        "out": str(out),
        "containers": writer.containers,
        "cases": rows,
        "environment": records.environment(),
        "seconds": {**clock.phases, "total": clock.total()},
        "peak_rss_gib": round(records.peak_rss_gib(), 2),
        "versions": records.versions(),
    }
    (out / SUMMARY_FILE).write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("command", choices=("write", "list"))
    parser.add_argument("--group", action="append", default=[], choices=GROUPS)
    parser.add_argument("--case", action="append", default=[], help="a case name (repeatable)")
    parser.add_argument("--out", type=Path, default=REFERENCE_DIR)
    args = parser.parse_args(argv)
    cases = select(CASES, args.group, args.case)
    if args.command == "list":
        for case in cases:
            print(f"{case.name}\t{case.graph}\t{case.role}")
        return 0
    clock = records.Clock()
    writer = real_writer()
    clock.lap("open the series")
    print(json.dumps(write(cases, out=args.out, writer=writer), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
