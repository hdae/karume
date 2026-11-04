"""The script that writes the 2 chunk graphs of the Wan video VAE decoder out to IR v2 (step 4 of
ADR 0118 / 0121).

The series, latent tile and fixture cases of each model (`--model` — default `t2v-1.3b`) are set
all at once by the {@link VAE_SERIES} table (forgetting to pass something only writes the 2.1 series
with 2.1 values; there is no path that silently breaks another model's series):

| Model | Series (`outputs/series/` — decision 7) | Tile t | Latent z | Frame C |
|---|---|---|---|---|
| `t2v-1.3b` (Wan2.1) | `wan2.1-t2v-1.3b-f16-dyn` (ADR 0118) | 32 | 16 | 3 (pixels) |
| `ti2v-5b` (Wan2.2) | `wan2.2-ti2v-5b-f16-dyn` (ADR 0121) | 16 | 48 | 12 (patchify space) |

What it writes:

- `vae_decoder_first/model.krm` — for the first chunk. Inputs = latent `[z,1,t,t]` + 30 caches;
  outputs = frames `[C,1,8t,8t]` + the 30 updated caches (it does not have the 2 of `time_conv`).
- `vae_decoder_next/model.krm` — for the following chunks. Inputs = latent + 32 caches; outputs =
  frames `[C,4,8t,8t]` + the 32 updated caches.
- `vae_chunks.<case>.safetensors` (at the series root) — the fixtures for checking the chunk
  sequence. A fixed-seed random latent (de-normalized, `z·std + mean`) and the output of the chunk
  loop of upstream's non-tiled `_decode` **before the clamp** (for 2.2 also before unpatchify)
  ({@link wan.vae_patch.reference_decode_unclamped}). The GPU check of the chunk sequence (the
  acceptance of step 4) reads them. There are 3 cases (ADR 0118 addendum 2026-10-02 — band cases
  and acceptance cases are separate): `band`, which sets the band (9 chunks = 33 frames), and
  `accept` (5 chunks = 17 frames) and `long` (21 chunks = 81 frames — ADR 0118 step 8, long enough
  to carry the caches over 20 times), which judge acceptance with another latent and other chunk
  boundaries. The seeds differ per model.

The tile side `t` (latent) can be overridden by an argument (the default comes from the table — 2.1
is diffusers' `tile_sample_min` 256 px ÷ 8, 2.2 is 256 px ÷ the spatial compression 16). The host
does not hold the tile side as a literal; it derives it from the input shapes of the opened asset
(decision 2).

Storage defaults to the f16 preset (decision 7): the weights are rounded to f16-representable
values (fake-quant — ADR 0006) before the export and the fixtures' reference are taken. Taking the
reference before rounding would make the checked difference a mix of quantization error and
implementation error, and the band would lose its meaning.

`--dtype i8` (ADR 0133 decision 7 — only for models with an i8 series, currently Wan2.2) applies
`fake_quant_int8` after the f16 rounding (the default targets — the decoder's conv3d and conv2d and
the mid 1×1 convs get i8 with one scale per output channel) and writes the same 2 graphs to **the
model's i8 series root** (for Wan2.2 the same `wan2.2-ti2v-5b-i8-dyn` as the DiT's i8 container —
the component directories `vae_decoder_first/` / `vae_decoder_next/` do not overlap with
`transformer/`). An i8 run writes no fixtures (the fixtures are the e2e f16 reference — the i8
series is compared with the f16 decode on the GPU side).

`--shared` (ADR 0134 decision 1 — `--tile` is required) traces the same module again at the
`--tile` side and writes, to the same series root, a pair that holds no weight bytes at all,
`<target>_<tile>/model.krm` (e.g. `vae_decoder_first_32` / `vae_decoder_next_32` — graph name =
component directory name = the key under the manifest's weights). **Every** initializer is a
shared declaration (`IrShared` — the lender's tensor key and the storage of the lender's binding
table), and the lender is the default-tile container of the same series root and the same storage
(`<series root>/<target>/model.krm` — written at the table's tile side). No bytes are written, so
neither the weight rounding nor the i8 quantization is applied (the storage declarations are copied
from the lender — `--dtype i8` also just reads the lender of the i8 series root). No fixtures are
written (the reference of the chunk sequence is the default-tile pair's fixtures).

    uv run --group wan --inexact python -m wan.export_vae              # 2 graphs + fixtures
    uv run --group wan --inexact python -m wan.export_vae --verify     # eager equivalence (real)
    uv run --group wan --inexact python -m wan.export_vae --model ti2v-5b [--verify]   # Wan2.2
    uv run --group wan --inexact python -m wan.export_vae --model ti2v-5b --dtype i8   # i8 series
    uv run --group wan --inexact python -m wan.export_vae --model ti2v-5b --tile 32 --shared
    uv run --group wan --inexact python -m wan.export_vae --model ti2v-5b --dtype i8 \\
        --tile 32 --shared                                             # the 32 pair (i8 series)

MUST: import diffusers / huggingface_hub inside functions (the `wan` group is not in the default
sync — `tests/test_optional_group_imports.py`).
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from safetensors.torch import save_file

from _shared.paths import SERIES_ROOT
from _shared.upstream import snapshot_license
from karume.artifacts import staged_publication
from karume.container import Provenance, container_parts
from karume.convert import PRESERVED_OP_PREFIXES_WITH_ATTENTION
from karume.dist import NOTICE_FILENAME
from karume.emit import storage_breakdown
from karume.ir import IrGraph, IrInitializer, IrShared, IrStorage
from karume.pipeline import export_module, export_to_file, publish_model
from karume.quantize import fake_quant_int8, round_weights_to_f16
from karume.verify import ir_graph_from_container, verify_container
from wan import vae_patch
from wan.sources import DEFAULT_MODEL, SOURCES, local_snapshot

if TYPE_CHECKING:
    from diffusers import AutoencoderKLWan

#: The series name (decision 7 — the `-dyn` suffix is the ADR 0077 convention, matching the DiT's
#: S shape. The 2 VAE graphs are in the same series).
SERIES_NAME = "wan2.1-t2v-1.3b-f16-dyn"

#: Target name = component name = the container's graph name (decision 7).
TARGET_FIRST = "vae_decoder_first"
TARGET_NEXT = "vae_decoder_next"
TARGETS = (TARGET_FIRST, TARGET_NEXT)

MODEL_FILE = "model.krm"

#: The default latent tile side (decision 2 — diffusers' `tile_sample_min` 256 px ÷ 8).
DEFAULT_TILE = 32

#: The spatial ratio between the chunk graph's inputs and outputs (the frame side per latent 1 —
#: pixels of the patchify space for 2.2). Upstream config's `scale_factor_spatial` is this ×
#: `patch_size` (8 × 1 for 2.1, 8 × 2 = 16 for 2.2 — checked by {@link assert_series_config},
#: ADR 0121 decision 6).
SPATIAL_SCALE = 8

#: The number of frames a next chunk produces (2 temporal upsample stages = 4×). first produces 1.
NEXT_FRAMES = 4

#: The file name of the chunk-sequence fixtures (at the series root —
#: `vae_chunks.<case>.safetensors`).
FIXTURE_PREFIX = "vae_chunks."
FIXTURE_SUFFIX = ".safetensors"

#: The fixtures' metadata `reference` (the reference's provenance). One for the generation without
#: patchify (2.1) and one for the generation whose chunk-loop output is in the patchify space
#: before unpatchify (2.2).
FIXTURE_REFERENCE = "diffusers AutoencoderKLWan._decode chunk loop before clamp (CPU f32)"
FIXTURE_REFERENCE_BEFORE_UNPATCHIFY = (
    "diffusers AutoencoderKLWan._decode chunk loop before unpatchify and clamp (CPU f32)"
)

#: The spelling of the graph input (the argument names of `WanVaeChunkDecoder.forward` become the IR
#: input names as is).
LATENT_INPUT = "latent"

#: The set of ops allowed in the chunk graph's IR (anything else means a missed rewrite — fail
#: loudly). SiLU decomposes into `x · sigmoid(x)`; the mid attention is SDPA preserved (ADR 0023).
EXPECTED_OPS = frozenset(
    {
        "add",
        "attention",
        "cat",
        "clamp_min",
        "conv2d",
        "conv3d",
        "div",
        "expand",
        "mul",
        "permute",
        "reshape",
        "sigmoid",
        "slice",
        "sqrt",
        "sum",
    }
)

#: The upper bound on value rank (the limit of the strided-copy family — decision 2). Rank 5 is
#: allowed only for conv3d weights.
MAX_VALUE_RANK = 4


@dataclass(frozen=True)
class FixtureCase:
    """One chunk-sequence fixture (the latent's seed and the chunk count)."""

    name: str
    seed: int
    chunks: int
    #: `band` = a case that sets the band, `accept` = a case that judges acceptance (addendum
    #: 2026-10-02).
    role: str


#: Band and acceptance cases differ in both the latent (seed) and the chunk boundaries (chunk
#: count). `long` is the 81-frame acceptance (ADR 0118 step 8), and the band stays that of `band`
#: (the band is not re-derived from the 21-chunk result).
FIXTURE_CASES = (
    FixtureCase("band", seed=20261002, chunks=9, role="band"),
    FixtureCase("accept", seed=20261003, chunks=5, role="accept"),
    FixtureCase("long", seed=20261004, chunks=21, role="accept"),
)

#: The VAE series of Wan2.2 TI2V-5B (ADR 0121 decision 7 — a VAE-only series, no DiT).
TI2V_SERIES_NAME = "wan2.2-ti2v-5b-f16-dyn"

#: The i8 series of the Wan2.2 VAE decoder (ADR 0133 decision 7 — the same series root as the
#: DiT's i8 container; the writer's spelling is `wan.ti2v_export_dit.SERIES_NAME`).
TI2V_I8_SERIES_NAME = "wan2.2-ti2v-5b-i8-dyn"

#: The weight storages that can be written (= `weight_dtype` of `export_to_file`). i8 only for
#: models with an i8 series.
WEIGHT_DTYPES = ("f16", "i8")

#: The latent tile side of Wan2.2 (ADR 0121 decision 6 — output 256 px ÷ spatial compression 16).
TI2V_TILE = 16

#: The Wan2.2 cases (the same construction as 2.1, only the seeds differ — 1 band case + 2
#: acceptance cases).
TI2V_FIXTURE_CASES = (
    FixtureCase("band", seed=20261041, chunks=9, role="band"),
    FixtureCase("accept", seed=20261042, chunks=5, role="accept"),
    FixtureCase("long", seed=20261043, chunks=21, role="accept"),
)


@dataclass(frozen=True)
class VaeSeries:
    """The VAE series of one model (location, latent tile, patchify factor, fixture cases)."""

    series: str
    tile: int
    #: The expected `patch_size` of the upstream config (fail loudly otherwise —
    #: {@link assert_series_config}).
    patch_size: int | None
    cases: tuple[FixtureCase, ...]
    #: The series `--dtype i8` writes to (ADR 0133 decision 7 — a model with None cannot write i8).
    i8_series: str | None = None


#: Model name (a key of `wan.sources.SOURCES`) → series.
VAE_SERIES: dict[str, VaeSeries] = {
    "t2v-1.3b": VaeSeries(SERIES_NAME, DEFAULT_TILE, None, FIXTURE_CASES, None),
    "ti2v-5b": VaeSeries(TI2V_SERIES_NAME, TI2V_TILE, 2, TI2V_FIXTURE_CASES, TI2V_I8_SERIES_NAME),
}


class ChunkGraphError(AssertionError):
    """The exported chunk graph departs from the conventions of decision 2 (rank, ops, cache
    inputs / outputs), or the writer's premises (the source revision, upstream config's
    `patch_size` and spatial compression) do not match the series."""


def load_vae(model: str = DEFAULT_MODEL, *, round_f16: bool) -> AutoencoderKLWan:
    """Loads the VAE of the pinned revision on CPU f32 (with `round_f16`, rounds it to
    f16-representable values)."""
    from diffusers import AutoencoderKLWan

    vae = AutoencoderKLWan.from_pretrained(
        local_snapshot(model), subfolder="vae", torch_dtype=torch.float32
    ).eval()
    if round_f16:
        report = round_weights_to_f16(vae)
        print(
            f"[fake-quant] vae: rounded to f16-representable values — {report.describe()}",
            flush=True,
        )
    return vae


def fake_quant_i8(vae: AutoencoderKLWan) -> Mapping[str, torch.Tensor]:
    """Rounds the f16-rounded VAE weights to per-channel symmetric i8-representable values and
    returns the scale ledger (ADR 0133 decision 7 — the same quantizer as the DiT's `dit8`, default
    targets).

    The targets are only the weights of `fake_quant_int8`'s default types (including `nn.Conv3d`);
    biases and norms keep their f16-rounded values (stored as f32). The ledger's keys are the VAE's
    FQNs, and the chunk graph wrapper ({@link wan.vae_patch.WanVaeChunkDecoder}) holds `decoder` /
    `post_quant_conv` under the same names, so it can be passed to `export_to_file` as is (emit does
    not look up the encoder's entries).
    """
    report = fake_quant_int8(vae)
    print(f"[fake-quant] vae: rounded to i8 per-channel — {report.describe()}", flush=True)
    return report.scales


def provenance(model: str = DEFAULT_MODEL) -> Provenance:
    """The provenance baked into the container (the license from the local snapshot's README, the
    revision is the pinned value).

    The snapshot's directory name is the HF cache commit SHA itself, and `local_snapshot` resolves
    only the pinned revision. A mismatch fails loudly (never claim the provenance of another
    checkpoint).
    """
    snapshot = local_snapshot(model)
    revision = SOURCES[model].revision
    if snapshot.name != revision:
        raise ChunkGraphError(f"snapshot {snapshot} is not the pinned revision {revision}")
    return Provenance(
        license=snapshot_license(snapshot), notice=NOTICE_FILENAME, upstream_revision=revision
    )


def assert_series_config(vae: AutoencoderKLWan, series: VaeSeries) -> None:
    """Checks that the upstream config matches the series' premises (`patch_size` and spatial
    compression) (fails loudly otherwise).

    The spatial compression `scale_factor_spatial` must be the chunk graph's input / output spatial
    ratio {@link SPATIAL_SCALE} × `patch_size` — the host's overlap formula
    (`vae_tiling.min_overlap_latent`) and the frame side `8t` depend on this ratio.
    """
    config = vae.config
    if config.patch_size != series.patch_size:
        raise ChunkGraphError(
            f"upstream patch_size {config.patch_size} differs from what series {series.series}"
            f" expects ({series.patch_size})"
        )
    expected = SPATIAL_SCALE * (series.patch_size or 1)
    if config.scale_factor_spatial != expected:
        raise ChunkGraphError(
            f"upstream scale_factor_spatial {config.scale_factor_spatial} is not"
            f" {SPATIAL_SCALE} × patch_size = {expected}"
        )


def weight_summary(vae: AutoencoderKLWan) -> dict[str, Any]:
    """Element count, max absolute value and non-finite count of the weights the graphs read
    (decoder + post-quant) (for the summary record).

    The f16 rounding (`round_weights_to_f16`) fails loudly when a finite value saturates to
    non-finite, so the non-finite count after rounding should be 0 — the measurement is kept here
    (the acceptance of ADR 0121 step 4).
    """
    tensors = [*vae.decoder.parameters(), *vae.post_quant_conv.parameters()]
    with torch.no_grad():
        return {
            "elements": sum(tensor.numel() for tensor in tensors),
            "abs_max": max(float(tensor.abs().max()) for tensor in tensors),
            "nonfinite": sum(int((~torch.isfinite(tensor)).sum()) for tensor in tensors),
        }


def _peak_rss_gib() -> float:
    """The peak RSS of this process (GiB — Linux `ru_maxrss` is in KiB)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1 << 20)


def example_inputs(
    module: vae_patch.WanVaeChunkDecoder, tile: int
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """The export's example inputs (latent of 1 frame + zero caches — the values only affect the
    shapes)."""
    channels = int(module.post_quant_conv.in_channels)
    latent = torch.zeros(channels, 1, tile, tile)
    return latent, vae_patch.zero_caches(module.slots, tile, first=module.first)


def assert_chunk_graph(graph: IrGraph, module: vae_patch.WanVaeChunkDecoder, tile: int) -> None:
    """Checks that the chunk graph's IR meets the conventions of decision 2 (fails loudly
    otherwise).

    - ops only from {@link EXPECTED_OPS} (no pad — causal padding is expressed by the cache cat).
    - values of rank 4 or less. Rank 5 only for conv3d weights (initializer, second input) (a rank-6
      reshape also fails this check).
    - inputs are `latent` + caches (names, shapes and order as in the table); outputs are the frames
      + the updated caches (in the same order). The channel counts of the latent and the frames are
      read from the module (the input of post-quant, the output of `conv_out`).
    - output cache k is built from "a slice of the value with cache input k concatenated at the
      start in time" (a graph with a swapped order silently exchanges caches of matching shapes — a
      shape check cannot catch it).
    """
    ops = Counter(node.op for node in graph.nodes)
    unexpected = sorted(set(ops) - EXPECTED_OPS)
    if unexpected:
        raise ChunkGraphError(f"unexpected ops {unexpected} (a missed rewrite)")

    producer = {out: node for node in graph.nodes for out in node.outs}
    conv3d_weights = {node.ins[1] for node in graph.nodes if node.op == "conv3d"}
    for name, value in graph.values.items():
        rank = len(value.shape)
        if rank <= MAX_VALUE_RANK:
            continue
        if name not in graph.initializers or name not in conv3d_weights or rank != 5:
            raise ChunkGraphError(f"rank {rank} value '{name}' (rank 5 only for conv3d weights)")

    slots = module.cache_slots
    side = tile * SPATIAL_SCALE
    latent_channels = int(module.post_quant_conv.in_channels)
    frame_channels = int(module.decoder.conv_out.out_channels)
    expected_inputs = [(LATENT_INPUT, [latent_channels, 1, tile, tile])]
    expected_inputs += [(slot.name, list(slot.shape(tile))) for slot in slots]
    actual_inputs = [(entry.name, [int(dim) for dim in entry.shape]) for entry in graph.inputs]
    if actual_inputs != expected_inputs:
        raise ChunkGraphError(f"inputs {actual_inputs} differ from the table {expected_inputs}")

    if len(graph.outputs) != 1 + len(slots):
        raise ChunkGraphError(f"{len(graph.outputs)} outputs (expected 1 + {len(slots)})")
    frames = 1 if module.first else NEXT_FRAMES
    frame_shape = [int(dim) for dim in graph.values[graph.outputs[0]].shape]
    if frame_shape != [frame_channels, frames, side, side]:
        raise ChunkGraphError(
            f"frame shape {frame_shape} (expected [{frame_channels}, {frames}, {side}, {side}])"
        )
    for slot, output in zip(slots, graph.outputs[1:], strict=True):
        shape = [int(dim) for dim in graph.values[output].shape]
        if shape != list(slot.shape(tile)):
            raise ChunkGraphError(f"output '{output}' has shape {shape}, unlike {slot.name}")
        sliced = producer.get(output)
        joined = producer.get(sliced.ins[0]) if sliced is not None and sliced.ins else None
        if (
            sliced is None
            or sliced.op != "slice"
            or joined is None
            or joined.op != "cat"
            or joined.ins[0] != slot.name
        ):
            raise ChunkGraphError(
                f"output '{output}' is not made from a slice of cat({slot.name}, …) (swapped order)"
            )


@contextmanager
def _staged_set(finals: Sequence[Path]) -> Iterator[list[Path]]:
    """Hands out staging directories for several finals in the order of `finals`, and publishes all
    of them only when the `with` exits without an exception.

    If any write or check fails, every staging directory is removed and no final changes by a
    single byte ({@link karume.artifacts.staged_publication} stacked once per staging directory).
    Publishing is one rename per staging directory, so the only window where a crash can land is
    the final sequence of renames.
    """
    with ExitStack() as stack:
        yield [stack.enter_context(staged_publication(final)) for final in finals]


def _graph_summary(
    target: str, out_dir: Path, staged: Path, graph: IrGraph, tile: int, started: float
) -> dict[str, Any]:
    """The summary of one graph (the container's byte count comes from what is in the staging
    directory — publishing only renames).

    `peak_rss_gib` is the peak RSS of this process so far (the peak of the graph phase; the overall
    peak is the value at the summary root).
    """
    breakdown = storage_breakdown(graph)
    return {
        "target": target,
        "dir": str(out_dir),
        "tile": tile,
        "nodes": len(graph.nodes),
        "op_counts": dict(sorted(Counter(node.op for node in graph.nodes).items())),
        "inputs": [[entry.name, [int(dim) for dim in entry.shape]] for entry in graph.inputs],
        "frame_shape": [int(dim) for dim in graph.values[graph.outputs[0]].shape],
        "caches": len(graph.outputs) - 1,
        "cache_outputs": [
            [output, [int(dim) for dim in graph.values[output].shape]]
            for output in graph.outputs[1:]
        ],
        "initializers": len(graph.initializers),
        "compressed_tensors": breakdown.compressed_tensors,
        "compressed_bytes": breakdown.compressed_bytes,
        "plain_tensors": breakdown.plain_tensors,
        "plain_bytes": breakdown.plain_bytes,
        "model_bytes": sum(part.stat().st_size for part in container_parts(staged / MODEL_FILE)),
        "seconds": round(time.perf_counter() - started, 1),
        "peak_rss_gib": round(_peak_rss_gib(), 2),
    }


def fixture_path(out_root: Path, case: FixtureCase) -> Path:
    """Where a chunk-sequence fixture lives (`vae_chunks.<case>.safetensors` at the series
    root)."""
    return out_root / f"{FIXTURE_PREFIX}{case.name}{FIXTURE_SUFFIX}"


def emit_targets(
    targets: Sequence[str],
    vae: AutoencoderKLWan,
    tile: int,
    out_root: Path,
    source: Provenance,
    *,
    cases: Sequence[FixtureCase],
    patch_size: int | None,
    weight_dtype: str = "f16",
    weight_scales: Mapping[str, torch.Tensor] | None = None,
) -> dict[str, Any]:
    """Writes the graphs and the chunk-sequence fixtures (`cases` — none when empty) as one set into
    staging directories, and publishes them after every check has passed.

    `weight_dtype` is the weight storage ({@link WEIGHT_DTYPES}); i8 comes with `weight_scales` (the
    ledger of {@link fake_quant_i8}). i8 writes no fixtures (the metadata `weights` claims only the
    f16-rounded reference — a reference made with i8 weights is not published under that name).

    `patch_size` is the value of the series table ({@link VaeSeries}) and decides only the wording
    of the fixtures' metadata `reference` (2.1 has no unpatchify, so the wording stays that of the
    existing 2.1 fixtures).

    MUST: publish as one set. Publishing per graph or per file would let a run that failed midway
    leave a mixed old/new set in the series (only first at the new tile side, next and the fixtures
    still old). A mix of different tile sides is rejected by the distribution check
    (`wan.distribution.assert_vae_chunk_pair`) and by the TS `wanVaeChunkLayout`, but a mix of the
    same shapes where only the generation of the contents differs cannot be told apart by either
    (the provenance carries only the revision — only the GPU check of the chunk sequence can catch
    it).
    """
    unknown = sorted(set(targets) - set(TARGETS))
    if unknown or len(set(targets)) != len(targets):
        raise ChunkGraphError(
            f"targets {list(targets)} hold an unknown or duplicate entry"
            f" (known: {', '.join(TARGETS)})"
        )
    if weight_dtype not in WEIGHT_DTYPES:
        raise ChunkGraphError(
            f"cannot write storage {weight_dtype!r} ({' / '.join(WEIGHT_DTYPES)})"
        )
    if weight_dtype != "f16" and cases:
        raise ChunkGraphError(
            f"a {weight_dtype} run writes no fixtures (only the f16 reference does)"
        )
    reference = FIXTURE_REFERENCE if patch_size is None else FIXTURE_REFERENCE_BEFORE_UNPATCHIFY
    out_root.mkdir(parents=True, exist_ok=True)
    finals = [out_root / target for target in targets]
    finals += [fixture_path(out_root, case) for case in cases]
    graphs: list[dict[str, Any]] = []
    written: list[dict[str, Any]] = []
    with _staged_set(finals) as seats:
        for target, staged in zip(targets, seats[: len(targets)], strict=True):
            started = time.perf_counter()
            module = vae_patch.WanVaeChunkDecoder(vae, first=target == TARGET_FIRST).eval()
            staged.mkdir()
            with torch.no_grad():
                graph = export_to_file(
                    module,
                    example_inputs(module, tile),
                    staged / MODEL_FILE,
                    provenance=source,
                    graph_name=target,
                    weight_dtype=weight_dtype,
                    weight_scales=weight_scales,
                    preserved=PRESERVED_OP_PREFIXES_WITH_ATTENTION,
                )
            assert_chunk_graph(graph, module, tile)
            graphs.append(_graph_summary(target, out_root / target, staged, graph, tile, started))
        for case, staged in zip(cases, seats[len(targets) :], strict=True):
            started = time.perf_counter()
            latents = fixture_latents(vae, case, tile)
            with torch.no_grad():
                frames = vae_patch.reference_decode_unclamped(vae, latents)[0]
            save_file(
                {"latents": latents[0].contiguous(), "frames": frames.contiguous()},
                str(staged),
                metadata={
                    "seed": str(case.seed),
                    "chunks": str(case.chunks),
                    "tile": str(tile),
                    "role": case.role,
                    "weights": "f16-rounded",
                    "reference": reference,
                },
            )
            written.append(
                {
                    "case": case.name,
                    "role": case.role,
                    "latents": list(latents.shape[1:]),
                    "frames": list(frames.shape),
                    "abs_max": float(frames.abs().max()),
                    "seconds": round(time.perf_counter() - started, 1),
                    "path": str(fixture_path(out_root, case)),
                }
            )
    summary: dict[str, Any] = {"series": str(out_root), "dtype": weight_dtype, "graphs": graphs}
    if cases:
        summary["fixtures"] = written
    return summary


def shared_target(target: str, tile: int) -> str:
    """The name `<target>_<tile>` of a shared pair (ADR 0134 decision 1) (graph name = component
    directory name = the key under the manifest's weights)."""
    return f"{target}_{tile}"


#: The graph names of the shared pair the distribution names (ADR 0134 decision 5 — Wan2.2's 32
#: latent tile). `--shared` can write any tile, but these 2 are the ones that become keys under
#: weights (the spelling check `tests/test_graph_names.py` compares them with
#: `wan.distribution.WAN_WEIGHTS`).
SHARED_TILE = 32
SHARED_TARGETS = tuple(shared_target(target, SHARED_TILE) for target in TARGETS)


def read_lender(path: Path, target: str, lender_tile: int, weight_dtype: str) -> IrGraph:
    """Reads the lender's graph (the default-tile container) as IR with the binding table's storage
    filled in (no weights are read).

    Checks (fail loudly otherwise): the container exists, the graph name is `target`, the latent
    input is `[z, 1, t, t]` with `t = lender_tile`, and the only compressed storage is
    `weight_dtype` (an i8 container placed by mistake at an f16 series root would make the borrower
    name weights of another storage).
    """
    parts = container_parts(path)
    if not all(part.is_file() for part in parts):
        raise ChunkGraphError(
            f"lender container {path} is missing — write the default-tile pair first"
            " (without `--tile` and `--shared`)"
        )
    lender = ir_graph_from_container(verify_container(parts).read, target)
    latent = lender.inputs[0] if lender.inputs else None
    if latent is None or latent.name != LATENT_INPUT or list(latent.shape[2:]) != [lender_tile] * 2:
        shape = None if latent is None else [latent.name, list(latent.shape)]
        raise ChunkGraphError(
            f"lender {path}: latent input {shape} is not tile {lender_tile}'s '{LATENT_INPUT}'"
        )
    compressed = {init.storage.dtype for init in lender.initializers.values()} - {"f32"}
    if compressed != {weight_dtype}:
        raise ChunkGraphError(
            f"lender {path}: compressed storage {sorted(compressed)}"
            f" is not of the {weight_dtype} series"
        )
    return lender


def share_lender_weights(graph: IrGraph, lender: IrGraph, lender_path: Path) -> IrGraph:
    """Replaces **every** initializer of the traced graph with a shared declaration pointing at the
    same tensor key of the lender.

    The declaration's storage is the storage (layout) of the lender's binding table, and no scale
    is written (the lender's resident weights hold it — the contract of `IrInitializer`). Checks:
    each initializer's tensor key exists as a real tensor in the lender, and the value's dtype and
    shape equal the lender's (tracing at another tile side does not change the weight shapes — if
    they change, the module is a different one).
    """
    initializers: dict[str, IrInitializer] = {}
    for name, initializer in graph.initializers.items():
        key = initializer.tensor
        lent = lender.initializers.get(key) if key is not None else None
        if lent is None or lent.is_shared:
            raise ChunkGraphError(
                f"initializer '{name}' ({key}) has no real tensor in the lender {lender_path}"
            )
        mine, theirs = graph.values[name], lender.values[key]
        if (mine.dtype, list(mine.shape)) != (theirs.dtype, list(theirs.shape)):
            raise ChunkGraphError(
                f"initializer '{key}' is {mine.dtype} {list(mine.shape)} — the lender {lender_path}"
                f" has {theirs.dtype} {list(theirs.shape)}"
            )
        initializers[name] = IrInitializer(
            shared=IrShared(tensor=key), storage=IrStorage(dtype=lent.storage.dtype)
        )
    return IrGraph(
        symbols=list(graph.symbols),
        inputs=list(graph.inputs),
        outputs=list(graph.outputs),
        initializers=initializers,
        values=dict(graph.values),
        states=dict(graph.states),
        nodes=list(graph.nodes),
    )


def emit_shared_targets(
    targets: Sequence[str],
    vae: AutoencoderKLWan,
    tile: int,
    out_root: Path,
    source: Provenance,
    *,
    lender_tile: int,
    weight_dtype: str = "f16",
    cases: Sequence[FixtureCase] = (),
) -> dict[str, Any]:
    """Writes the pair `<target>_<tile>` that holds no weight bytes (ADR 0134 decision 1 —
    `--shared`).

    The lender is `<target>/model.krm` at the same series root (tile `lender_tile`, storage
    `weight_dtype`). Not a single byte of the module's weights is written, so neither the f16
    rounding nor the i8 quantization is needed (the storage declarations are copied from the
    lender's binding table). A non-empty `cases` is rejected — the reference of the chunk sequence
    is the lender pair's fixtures, and this pair writes none (the tile reference is ADR 0134
    decision 6).

    Checks (nothing is published otherwise): the lender's container exists ({@link read_lender}),
    every initializer of the traced graph names a real tensor of the lender with the same key, shape
    and dtype ({@link share_lender_weights}), every initializer of the written container is a shared
    declaration (none holds bytes), and {@link assert_chunk_graph} holds at `tile`. The 2 are
    published as one set, as in {@link emit_targets}.
    """
    unknown = sorted(set(targets) - set(TARGETS))
    if unknown or len(set(targets)) != len(targets):
        raise ChunkGraphError(
            f"targets {list(targets)} hold an unknown or duplicate entry"
            f" (known: {', '.join(TARGETS)})"
        )
    if weight_dtype not in WEIGHT_DTYPES:
        raise ChunkGraphError(
            f"cannot write storage {weight_dtype!r} ({' / '.join(WEIGHT_DTYPES)})"
        )
    if cases:
        raise ChunkGraphError(
            "a shared pair writes no fixtures (the chunk sequence's references are the lender"
            " pair's)"
        )
    if tile == lender_tile:
        raise ChunkGraphError(
            f"tile {tile} equals the lender's — a shared pair is only for another tile side"
        )
    lenders = {
        target: read_lender(out_root / target / MODEL_FILE, target, lender_tile, weight_dtype)
        for target in targets
    }
    names = [shared_target(target, tile) for target in targets]
    graphs: list[dict[str, Any]] = []
    with _staged_set([out_root / name for name in names]) as seats:
        for target, name, staged in zip(targets, names, seats, strict=True):
            started = time.perf_counter()
            lender_path = out_root / target / MODEL_FILE
            module = vae_patch.WanVaeChunkDecoder(vae, first=target == TARGET_FIRST).eval()
            staged.mkdir()
            with torch.no_grad():
                traced, _ = export_module(
                    module,
                    example_inputs(module, tile),
                    preserved=PRESERVED_OP_PREFIXES_WITH_ATTENTION,
                )
            graph = publish_model(
                staged / MODEL_FILE,
                share_lender_weights(traced, lenders[target], lender_path),
                {},
                provenance=source,
                graph_name=name,
            )
            owned = sorted(key for key, init in graph.initializers.items() if not init.is_shared)
            if owned:
                raise ChunkGraphError(f"{name}: initializers holding bytes remain {owned[:4]}")
            assert_chunk_graph(graph, module, tile)
            summary = _graph_summary(name, out_root / name, staged, graph, tile, started)
            summary["lender"] = str(lender_path)
            graphs.append(summary)
    return {
        "series": str(out_root),
        "dtype": weight_dtype,
        "tile": tile,
        "lender": {"tile": lender_tile, "targets": list(targets)},
        "graphs": graphs,
    }


def fixture_latents(vae: AutoencoderKLWan, case: FixtureCase, tile: int) -> torch.Tensor:
    """A fixed-seed random latent, de-normalized, `[1, z, F, t, t]` (the value right before decode
    in upstream `WanPipeline`; z is upstream config's `z_dim` — 16 for 2.1, 48 for 2.2).

    `WanPipeline` de-normalizes with `latents / (1/std) + mean` before passing to `vae.decode`. The
    graph input is the value after that, so the fixtures are built in the same value range as real
    use (the band's justification is tied to the value range of real use — the same judgment as the
    dacvae goldens).
    """
    generator = torch.Generator().manual_seed(case.seed)
    channels = int(vae.config.z_dim)
    mean = torch.tensor(vae.config.latents_mean, dtype=torch.float32).view(1, channels, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std, dtype=torch.float32).view(1, channels, 1, 1, 1)
    noise = torch.randn(1, channels, case.chunks, tile, tile, generator=generator)
    return noise * std + mean


def verify(model: str, tile: int, chunks: int, seed: int) -> dict[str, Any]:
    """Measures the patch's eager equivalence on the real weights (f32, no rounding).

    - `clamp(unpatchify(upstream's chunk loop before clamp)) == upstream _decode` (the reference's
      provenance — bit-exact. unpatchify applies only to the generation with patchify (2.2), using
      diffusers' function itself)
    - the form with only the cache normalization applied (upstream's code as is) == upstream
      (bit-exact — the measurement behind the derivation of decision 2)
    - the difference between the final rewritten form ({@link wan.vae_patch.chunk_decode}) and
      upstream (recorded)
    """
    from diffusers.models.autoencoders.autoencoder_kl_wan import unpatchify

    series = VAE_SERIES[model]
    vae = load_vae(model, round_f16=False)
    assert_series_config(vae, series)
    case = FixtureCase("verify", seed=seed, chunks=chunks, role="verify")
    latents = fixture_latents(vae, case, tile)
    with torch.no_grad():
        reference = vae_patch.reference_decode_unclamped(vae, latents)
        decoded = vae._decode(latents, return_dict=False)[0]
        normalized = vae_patch.normalized_cache_decode(vae, latents, tile)
        first = vae_patch.WanVaeChunkDecoder(vae, first=True).eval()
        following = vae_patch.WanVaeChunkDecoder(vae, first=False).eval()
        final = vae_patch.chunk_decode(first, following, latents[0])
    restored = reference if series.patch_size is None else unpatchify(reference, series.patch_size)
    difference = (final - reference[0]).abs()
    reference_max = float(reference.abs().max())
    per_frame = [float(difference[:, index].max()) for index in range(difference.shape[1])]
    return {
        "model": model,
        "tile": tile,
        "chunks": chunks,
        "seed": seed,
        "frames": list(reference.shape[1:]),
        "reference_abs_max": reference_max,
        "clamped_reference_equals_decode": torch.equal(restored.clamp(-1.0, 1.0), decoded),
        "normalized_cache_bit_exact": torch.equal(normalized, reference),
        "final_bit_exact": torch.equal(final, reference[0]),
        "final_max_abs": float(difference.max()),
        "final_ratio": float(difference.max()) / reference_max,
        "final_max_abs_first_frame": per_frame[0],
        "final_max_abs_last_frame": per_frame[-1],
        "peak_rss_gib": round(_peak_rss_gib(), 2),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--model",
        choices=sorted(VAE_SERIES),
        default=DEFAULT_MODEL,
        help=f"model (series, tile and cases come from the table — default {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--tile",
        type=int,
        default=None,
        help="latent tile side (default from the table — 32 for 2.1, 16 for 2.2)",
    )
    parser.add_argument(
        "--dtype",
        choices=WEIGHT_DTYPES,
        default="f16",
        help="weight storage (default f16. i8 only for models with an i8 series — ADR 0133"
        " decision 7. i8 writes no fixtures: the fixtures are the e2e f16 reference, and the i8"
        " series is compared with the f16 decode on the GPU side)",
    )
    parser.add_argument(
        "--out", type=Path, default=None, help="series root (default: the model's series — table)"
    )
    parser.add_argument(
        "--target",
        action="append",
        choices=TARGETS,
        default=None,
        help="graphs to write (default: both)",
    )
    parser.add_argument(
        "--no-fixtures", action="store_true", help="do not write the chunk-sequence fixtures"
    )
    parser.add_argument(
        "--shared",
        action="store_true",
        help="write the pair `<target>_<tile>` that holds no weight bytes (--tile required; the"
        " lender is the default-tile pair at the same series root; no fixtures — ADR 0134"
        " decision 1)",
    )
    parser.add_argument(
        "--verify", action="store_true", help="only measure the eager equivalence (writes nothing)"
    )
    parser.add_argument(
        "--chunks", type=int, default=9, help="chunk count for --verify (default 9)"
    )
    parser.add_argument("--seed", type=int, default=0, help="latent seed for --verify")
    args = parser.parse_args(argv)
    series = VAE_SERIES[args.model]
    tile = series.tile if args.tile is None else args.tile
    if tile <= 0:
        parser.error(f"--tile must be a positive integer (got {tile})")
    i8 = args.dtype == "i8"
    series_name = series.series
    if i8:
        if series.i8_series is None:
            parser.error(
                f"--dtype i8: model {args.model} has no VAE i8 series (ADR 0133 decision 7)"
            )
        series_name = series.i8_series
    if args.shared and args.tile is None:
        parser.error(
            "--shared requires --tile (a side other than the lender's default tile"
            " — ADR 0134 decision 1)"
        )
    if args.shared and args.verify:
        parser.error("--shared and --verify cannot be used together (--verify writes nothing)")

    if args.verify:
        summary = verify(args.model, tile, args.chunks, args.seed)
        print(json.dumps(summary, indent=1, ensure_ascii=False))
        return 0

    source = provenance(args.model)
    if args.shared:
        # No bytes are written, so no rounding (the storage declarations are copied from the
        # lender).
        vae = load_vae(args.model, round_f16=False)
        assert_series_config(vae, series)
        summary = emit_shared_targets(
            args.target or TARGETS,
            vae,
            tile,
            args.out or SERIES_ROOT / series_name,
            source,
            lender_tile=series.tile,
            weight_dtype=args.dtype,
        )
        summary["peak_rss_gib"] = round(_peak_rss_gib(), 2)
        print(json.dumps(summary, indent=1, ensure_ascii=False))
        return 0
    vae = load_vae(args.model, round_f16=True)
    assert_series_config(vae, series)
    scales = fake_quant_i8(vae) if i8 else None
    summary = emit_targets(
        args.target or TARGETS,
        vae,
        tile,
        args.out or SERIES_ROOT / series_name,
        source,
        cases=() if i8 or args.no_fixtures else series.cases,
        patch_size=series.patch_size,
        weight_dtype=args.dtype,
        weight_scales=scales,
    )
    summary["weights"] = weight_summary(vae)
    summary["peak_rss_gib"] = round(_peak_rss_gib(), 2)
    print(json.dumps(summary, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
