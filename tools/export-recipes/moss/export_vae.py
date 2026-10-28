"""Export of the MOSS-Image VAE decoder as an f32 series — the graphs `vae_decoder_pre`,
`vae_decoder_attn`, `vae_decoder_post`, the asset `latent_batch_norm` and the GPU goldens (ADR NNNN
decision 8, stage 3).

    uv run --group moss --inexact python -m moss.export_vae inspect  # traces + structure only
    uv run --group moss --inexact python -m moss.export_vae write    # the series, behind the gates
    uv run --group moss --inexact python -m moss.export_vae write --cap  # + the area-cap golden
    uv run --group moss --inexact python -m moss.export_vae eager    # the eager gate on the series

`write` reads the stage-0 VAE records (`moss.vae_reference`; default
`outputs/bench/karume-moss-image/<today>_stage0-reference/`, another directory by `--stage0`) for
gate 3. Each command prints a JSON summary and writes it to `--out` (default
`outputs/bench/karume-moss-image/<date>_vae-export/<command>.json`); a failed gate exits non-zero.

## The series (`outputs/series/moss-image-sft-f32-dyn/` — a VAE-only series)

The VAE has its own series, named by its storage (f32 — upstream decodes in f32 and the VAE
runs f32 in every preset, ADR NNNN decision 13), the Wan2.2 form (`wan2.2-ti2v-5b-f16-dyn` holds
only the VAE — ADR 0121 decision 7). Two reasons against the DiT's series: `moss.export_dit write`
replaces its series directory as a whole (the VAE parts would vanish on every DiT export), and the
VAE does not change with the DiT's storage (i8 now, bf16 later — ADR NNNN decision 13).

| Part               | Symbols | Weights (all f32)                          | Asset               |
| ------------------ | ------- | ------------------------------------------ | ------------------- |
| `vae_decoder_pre`  | h, w    | post_quant_conv, conv_in, mid resnet 0     | `latent_batch_norm` |
| `vae_decoder_attn` | S       | the mid attention                          | none                |
| `vae_decoder_post` | h, w    | mid resnet 1, up blocks, norm out, conv_out | none               |

and at the series root the goldens `vae_decoder.<case>.safetensors` ({@link CASES}): `latents`
(the DiT-space input `[1,128,h/2,w/2]`), `z` (the decode input), `decoded` (upstream's untiled f32
decode before the clamp) and `image` (clamped to [-1, 1]), one JSON metadata key.

The graphs are {@link moss.vae_patch}. The asset `latent_batch_norm` (role `latent-batch-norm`,
safetensors `mean` / `std` `[128]` f32) holds the two vectors of the pipeline's BatchNorm
restoration `z·std + mean` — `std = (running_var + eps).sqrt()` written as upstream's statement
in `moss.pipeline_ref.denormalize` computes it — so the host multiplies and adds and never takes a
square root (the ADR 0034 decision 3 rule for host tables).

## Gate 1 — structure ({@link inspect_structure})

- inputs, output and symbols as declared (`z [1,32,h,w]` → `[1,384,h,w]`; `tokens [1,384,S]` →
  `[1,384,S]`; `hidden [1,384,h,w]` → `[1,3,8h,8w]`);
- no symbol-dependent constant baked in: no initializer of rank ≥ 5 or with a dimension equal to a
  trace's symbol maximum, no `sym_prefix_slice` (the DiT gate's form);
- the census against the module structure ({@link expected_census}): one `conv2d` per Conv2d, one
  `group_norm` per GroupNorm (none decomposed), a `sigmoid` and a `mul` per SiLU, an `add` and a
  `div` per resnet (the residual and `output_scale_factor`), and per ×2 upsample the reshape /
  expand form (2 `expand` + 4 `reshape`) — the form the runtime's fusion rule `upsample2x` takes;
- storage: f32 on every initializer; the three graphs' tensor keys are disjoint and together the
  decoder's and `post_quant_conv`'s parameters.

## Gate 2 — eager equivalence, bit-identical (MUST)

One process, upstream's f32 weights: for every case, the three graph modules chained as the host
chains them ({@link moss.vae_patch.decode_split}) ≡ upstream's untiled `vae.decode`, and each
graph boundary ≡ the tensor upstream's decode had there (`moss.pipeline_ref.VaeRecorder` — mid
resnet 0's output, the attention's output). With the written series, every tensor of the three
containers ≡ the VAE parameter under the same key (bit for bit).

## Gate 3 — the host tables ({@link host_tables_gate})

The asset read back from the written `vae_decoder_pre`: `latents · std + mean` (torch f32) ≡
`moss.pipeline_ref.denormalize` and its unpatchify ≡ the decode input, on every case and on the
stage-0 VAE records (`denormalized`, `vae.z`); the golden of the stage-0 seed ≡ the record
(`latents`, `vae.z`, `vae.decoded`, `image`). The records are required.

## The cases ({@link CASES})

256² and 512×768 (height × width — non-square so H and W cannot be swapped unnoticed): per size
three band-decision latents and two acceptance latents (one is the stage-0 record's seed). The
latents are seeded DiT-space noise through upstream's BatchNorm restoration and unpatchify, as
`moss.vae_reference` builds them. `--cap` adds one acceptance case at the area cap of a 2 GiB
binding: 1696×1648 (2,795,008 px — 192·4 B per pixel = 2,146,566,144 B, the largest accepted area
on the RTX 3080 Ti), for the opt-in GPU comparison and the VRAM measurement.

MUST: diffusers is imported inside the functions (`tests/test_optional_group_imports.py`).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load, save
from torch import nn
from torch.export import Dim

from _shared.container_read import open_container, read_asset
from _shared.paths import BENCH_ROOT, SERIES_ROOT
from karume.artifacts import staged_publication
from karume.container import AssetInput, Provenance, codec_entry
from karume.convert import PRESERVED_OP_PREFIXES_WITH_ATTENTION
from karume.dist import NOTICE_FILENAME
from karume.ir import IrGraph
from karume.ops import SYM_PREFIX_SLICE_OP
from karume.pipeline import export_module, publish_model
from moss import pipeline_ref, records, sources, vae_patch, vae_reference
from moss.records import bits_equal
from moss.vae_patch import GRAPHS, INPUT_NAMES, VAE_DECODER_ATTN, VAE_DECODER_POST, VAE_DECODER_PRE

#: The VAE series (f32 storage, symbolic-length graphs — ADR 0077's `-dyn` convention).
SERIES_NAME = "moss-image-sft-f32-dyn"
SERIES = SERIES_ROOT / SERIES_NAME

MODEL_FILE = "model.krm"

#: The asset of `vae_decoder_pre` (module docstring).
BATCH_NORM_ASSET = "latent_batch_norm"
BATCH_NORM_ROLE = "latent-batch-norm"

#: The symbols (pre / post: the VAE latent height and width; attn: S = h·w).
LATENT_SYMBOLS = ("h", "w")
SEQUENCE_SYMBOL = "S"

#: torch.export's bounds of the symbols — they bound the trace only (IR symbols carry no bounds;
#: the area cap is the host's, per device). h / w: above the longest accepted side (65,536 px =
#: 8,192 VAE cells — ADR NNNN decision 9). S: above h·w at a 4 GiB binding (2³² / (192·4·64)
#: ≈ 87,381). MUST: no bound equals a static extent of the graphs (gate 1 reads a dimension equal
#: to the bound as a baked constant).
SYM_MIN = 2
LATENT_MAX = 8_192
SEQUENCE_MAX = 131_072

#: The trace's example latent (non-square, h ≠ w).
TRACE_LATENT = (24, 40)

#: The output names of the bench directory (docs/assets-layout.md).
BENCH_PURPOSE = "vae-export"

#: The golden file names (`vae_decoder.<case>.safetensors` at the series root).
GOLDEN_PREFIX = "vae_decoder."
GOLDEN_SUFFIX = ".safetensors"

#: The single metadata key of the goldens.
GOLDEN_KEY = "karume.moss.vae_golden"

#: The golden's reference (what `decoded` is).
GOLDEN_REFERENCE = (
    "diffusers AutoencoderKLFlux2.decode, untiled, f32 (CPU), on the BatchNorm-restored and"
    " unpatchified latents"
)

#: How the golden latents are drawn (`moss.pipeline_ref.seeded_latents`).
GOLDEN_NOISE = "torch.randn(generator=torch.Generator().manual_seed(seed), dtype=float32)"


class VaeExportError(RuntimeError):
    """A graph, a weight or a comparison left the form step 3 accepts."""


def default_out() -> Path:
    return BENCH_ROOT / records.BENCH_NAME / f"{date.today().isoformat()}_{BENCH_PURPOSE}"


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VaeCase:
    """One golden: image size (pixels), the seed of its DiT-space latents and its role."""

    height: int
    width: int
    seed: int
    #: `band` = band decision, `accept` = acceptance, `cap` = the opt-in area-cap case.
    role: str

    @property
    def name(self) -> str:
        return f"{self.height}x{self.width}-{self.role}-s{self.seed}"


#: The seeds of the band-decision and acceptance latents (the stage-0 seed is an acceptance case).
BAND_SEEDS = (20261101, 20261102, 20261103)
ACCEPT_SEEDS = (vae_reference.SEED, 20261104)

#: The sizes of the default cases (the stage-0 VAE sizes).
SIZES = vae_reference.SIZES

#: The area-cap case (module docstring): the largest accepted area at a 2 GiB binding.
CAP_SIZE = (1696, 1648)
CAP_SEED = 20261105


def _cases(*, cap: bool) -> tuple[VaeCase, ...]:
    cases = [
        VaeCase(height, width, seed, role)
        for height, width in SIZES
        for role, seeds in (("band", BAND_SEEDS), ("accept", ACCEPT_SEEDS))
        for seed in seeds
    ]
    if cap:
        cases.append(VaeCase(*CAP_SIZE, CAP_SEED, "cap"))
    return tuple(cases)


CASES = _cases(cap=False)
CASES_WITH_CAP = _cases(cap=True)


def golden_path(directory: Path, case: VaeCase) -> Path:
    return directory / f"{GOLDEN_PREFIX}{case.name}{GOLDEN_SUFFIX}"


# ---------------------------------------------------------------------------
# Trace
# ---------------------------------------------------------------------------


def symbols_of(name: str) -> tuple[str, ...]:
    return (SEQUENCE_SYMBOL,) if name == VAE_DECODER_ATTN else LATENT_SYMBOLS


def example_inputs(name: str, vae: Any) -> tuple[torch.Tensor]:
    """The trace's example input (values do not reach the trace)."""
    height, width = TRACE_LATENT
    hidden = vae_patch.hidden_channels(vae)
    if name == VAE_DECODER_PRE:
        return (torch.zeros(1, int(vae.config.latent_channels), height, width),)
    if name == VAE_DECODER_ATTN:
        return (torch.zeros(1, hidden, height * width),)
    if name == VAE_DECODER_POST:
        return (torch.zeros(1, hidden, height, width),)
    raise VaeExportError(f"unknown graph {name!r} ({', '.join(GRAPHS)})")


def dynamic_shapes(name: str, *, latent_max: int = LATENT_MAX, sequence_max: int = SEQUENCE_MAX):
    if name == VAE_DECODER_ATTN:
        return ({2: Dim(SEQUENCE_SYMBOL, min=SYM_MIN, max=sequence_max)},)
    height = Dim(LATENT_SYMBOLS[0], min=SYM_MIN, max=latent_max)
    width = Dim(LATENT_SYMBOLS[1], min=SYM_MIN, max=latent_max)
    return ({2: height, 3: width},)


def graph_modules(vae: Any) -> dict[str, nn.Module]:
    pre, attention, post = vae_patch.decoder_graphs(vae)
    return {VAE_DECODER_PRE: pre, VAE_DECODER_ATTN: attention, VAE_DECODER_POST: post}


@dataclass(frozen=True)
class Traced:
    graph: IrGraph
    tensors: Mapping[str, torch.Tensor]


def trace(name: str, module: nn.Module, vae: Any, **bounds: int) -> Traced:
    with torch.no_grad():
        graph, tensors = export_module(
            module,
            example_inputs(name, vae),
            dynamic_shapes=dynamic_shapes(name, **bounds),
            symbol_names=symbols_of(name),
            preserved=PRESERVED_OP_PREFIXES_WITH_ATTENTION,
        )
    return Traced(graph=graph, tensors=tensors)


# ---------------------------------------------------------------------------
# Gate 1 — structure
# ---------------------------------------------------------------------------


def census(graph: IrGraph) -> dict[str, int]:
    return dict(sorted(Counter(node.op for node in graph.nodes).items()))


def expected_census(name: str, module: nn.Module) -> dict[str, int]:
    """The node counts the module structure implies (module docstring "Gate 1")."""
    from diffusers.models.resnet import ResnetBlock2D
    from diffusers.models.upsampling import Upsample2D

    modules = list(module.modules())
    convs = sum(isinstance(m, nn.Conv2d) for m in modules)
    norms = sum(isinstance(m, nn.GroupNorm) for m in modules)
    silus = sum(isinstance(m, nn.SiLU) for m in modules)
    resnets = [m for m in modules if isinstance(m, ResnetBlock2D)]
    upsamplers = sum(isinstance(m, Upsample2D) for m in modules)
    # A resnet calls its SiLU twice (after norm1 and after norm2); the decoder's `conv_act` once.
    activations = 2 * len(resnets) + (silus - len({id(r.nonlinearity) for r in resnets}))
    counts = {
        "conv2d": convs,
        "group_norm": norms,
        "sigmoid": activations,
        "mul": activations,
        "add": len(resnets),
        "div": len(resnets),
    }
    if name == VAE_DECODER_ATTN:
        # q, k, v, out; one residual add and the rescale division; the head split / merge.
        counts |= {"linear": 4, "attention": 1, "add": 1, "div": 1, "permute": 6, "reshape": 4}
    if upsamplers:
        counts |= {"expand": 2 * upsamplers, "reshape": 4 * upsamplers}
    return {op: count for op, count in sorted(counts.items()) if count}


def declared_io(name: str, vae: Any) -> tuple[list[Any], list[Any]]:
    """`[input name, shape]` and the output shape as declared."""
    hidden = vae_patch.hidden_channels(vae)
    scale = vae_patch.spatial_scale(vae)
    latent = [1, hidden, "h", "w"]
    if name == VAE_DECODER_PRE:
        return [INPUT_NAMES[name], [1, int(vae.config.latent_channels), "h", "w"]], latent
    if name == VAE_DECODER_ATTN:
        return [INPUT_NAMES[name], [1, hidden, "S"]], [1, hidden, "S"]
    out = int(vae.config.out_channels)
    return [INPUT_NAMES[name], latent], [1, out, f"{scale}h", f"{scale}w"]


def symbol_constants(graph: IrGraph, bounds: Sequence[int]) -> list[str]:
    """What looks like a baked symbol-dependent constant (module docstring "Gate 1")."""
    found = [
        f"initializer {name} {graph.values[name].shape} (rank ≥ 5)"
        for name in graph.initializers
        if len(graph.values[name].shape) >= 5
    ]
    found += [
        f"initializer {name} {graph.values[name].shape} (a dimension equals a symbol bound)"
        for name in graph.initializers
        if any(bound in graph.values[name].shape for bound in bounds)
    ]
    found += [
        f"node {n.outs} is {SYM_PREFIX_SLICE_OP}"
        for n in graph.nodes
        if n.op == SYM_PREFIX_SLICE_OP
    ]
    return found


def inspect_structure(
    name: str, traced: Traced, module: nn.Module, vae: Any, *, bounds: Sequence[int]
) -> dict[str, Any]:
    """Gate 1's storage-independent items for one graph (`failures` empty = passed)."""
    graph = traced.graph
    ops = census(graph)
    inputs = [[entry.name, list(entry.shape)] for entry in graph.inputs]
    outputs = [list(graph.values[output].shape) for output in graph.outputs]
    expected_input, expected_output = declared_io(name, vae)
    failures: list[str] = []
    if list(graph.symbols) != list(symbols_of(name)):
        failures.append(f"symbols {graph.symbols} (expected {list(symbols_of(name))})")
    if inputs != [expected_input]:
        failures.append(f"inputs {inputs} (expected {[expected_input]})")
    if outputs != [expected_output]:
        failures.append(f"outputs {outputs} (expected {[expected_output]})")
    if any(entry.dtype != "f32" for entry in graph.inputs):
        failures.append("an input is not f32")
    failures += symbol_constants(graph, bounds)
    expected = expected_census(name, module)
    if ops != expected:
        failures.append(f"census {ops} (expected {expected})")
    return {
        "ops": ops,
        "symbols": list(graph.symbols),
        "inputs": inputs,
        "outputs": outputs,
        "initializers": len(graph.initializers),
        "failures": failures,
    }


def inspect_storage(graphs: Mapping[str, IrGraph], vae: Any) -> dict[str, Any]:
    """f32 on every initializer; the three graphs' tensor keys are disjoint and cover the decoder's
    and `post_quant_conv`'s parameters (a committed graph per name)."""
    parameters = _parameter_keys(vae)
    keys: dict[str, set[str]] = {}
    failures: list[str] = []
    for name, graph in graphs.items():
        own = {init.tensor for init in graph.initializers.values() if init.tensor is not None}
        keys[name] = own & parameters
        other = sorted(
            init.tensor
            for init in graph.initializers.values()
            if init.tensor is not None and init.storage.dtype != "f32"
        )
        if other:
            failures.append(f"{name}: storage other than f32 on {other[:5]}")
    union: set[str] = set()
    for name, own in keys.items():
        overlap = sorted(union & own)
        if overlap:
            failures.append(f"{name}: tensor keys shared with another graph {overlap[:5]}")
        union |= own
    missing = sorted(parameters - union)
    if missing:
        failures.append(f"decoder parameters in no graph: {missing[:5]}")
    return {
        "weights": {name: len(own) for name, own in keys.items()},
        "parameters": len(parameters),
        "failures": failures,
    }


def _parameter_keys(vae: Any) -> set[str]:
    """The decoder's and `post_quant_conv`'s state-dict keys (the graphs use upstream's names)."""
    return {key for key in vae.state_dict() if key.startswith(("decoder.", "post_quant_conv."))}


# ---------------------------------------------------------------------------
# Assets and cases
# ---------------------------------------------------------------------------


def batch_norm_tables(vae: Any) -> dict[str, torch.Tensor]:
    """The asset `latent_batch_norm` (module docstring): `std` is upstream's
    `(bn.running_var + eps).sqrt()` (the expression of `moss.pipeline_ref.denormalize`), `mean` its
    `bn.running_mean`, both f32 `[128]`."""
    bn = vae.bn
    eps = vae.config.batch_norm_eps
    std = (bn.running_var + eps).sqrt().to(torch.float32).contiguous()
    mean = bn.running_mean.to(torch.float32).contiguous()
    return {"mean": mean, "std": std}


def batch_norm_asset(vae: Any) -> AssetInput:
    payload = save(batch_norm_tables(vae))
    return AssetInput(BATCH_NORM_ROLE, len(payload), payload)


def host_denormalize(tables: Mapping[str, torch.Tensor], latents: torch.Tensor) -> torch.Tensor:
    """What the host computes from the asset: `latents · std + mean` per channel in f32."""
    return latents * tables["std"].view(1, -1, 1, 1) + tables["mean"].view(1, -1, 1, 1)


def case_tensors(vae: Any, case: VaeCase) -> dict[str, torch.Tensor]:
    """One golden's tensors (module docstring "The series"), with upstream's decode."""
    latents = pipeline_ref.seeded_latents(case.seed, case.height, case.width)
    z = pipeline_ref.unpatchify(pipeline_ref.denormalize(vae, latents))
    with torch.no_grad():
        decoded = vae_patch.decode_reference(vae, z)
    return {"latents": latents, "z": z, "decoded": decoded, "image": decoded.clamp(-1, 1)}


def provenance() -> Provenance:
    return Provenance(
        license=sources.SOURCE.license,
        notice=NOTICE_FILENAME,
        upstream_revision=sources.SOURCE.revision,
    )


# ---------------------------------------------------------------------------
# Gate 2 — eager equivalence
# ---------------------------------------------------------------------------


def _difference(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, Any]:
    return {
        "bit_identical": bits_equal(actual, expected),
        "differing_elements": int(actual.ne(expected).sum()),
        "max_abs_diff": float((actual - expected).abs().max()),
    }


@torch.no_grad()
def eager_case(vae: Any, graphs: Mapping[str, nn.Module], z: torch.Tensor) -> dict[str, Any]:
    """The split decode against upstream's decode on one input (`failures` empty = passed)."""
    with pipeline_ref.VaeRecorder(vae) as recorder:
        upstream = vae_patch.decode_reference(vae, z)
    seen = recorder.tensors
    hidden = graphs[VAE_DECODER_PRE](z)
    batch, channels, height, width = hidden.shape
    tokens = graphs[VAE_DECODER_ATTN](hidden.reshape(batch, channels, height * width))
    attended = tokens.reshape(batch, channels, height, width)
    decoded = graphs[VAE_DECODER_POST](attended)
    report = {
        "pre": _difference(hidden, seen["vae.pre_attention"]),
        "attn": _difference(attended, seen["vae.post_attention"]),
        "decode": _difference(decoded, upstream),
    }
    report["failures"] = [
        f"{key}: not bit-identical ({report[key]['differing_elements']} elements,"
        f" max {report[key]['max_abs_diff']:.3e})"
        for key in ("pre", "attn", "decode")
        if not report[key]["bit_identical"]
    ]
    return report


def container_weights(path: Path, vae: Any) -> dict[str, Any]:
    """Every tensor of a written container ≡ the VAE's state-dict entry of the same key (f32)."""
    verified = open_container(path)
    state = vae.state_dict()
    differing: list[str] = []
    counts: Counter[str] = Counter()
    for bound in verified.graphs.values():
        values = bound.declaration["values"]
        for key, supply in bound.supplies.items():
            if key not in state:
                counts["constants"] += 1
                continue
            layout = codec_entry(supply.encoding.codec).layout
            if layout != "f32":
                raise VaeExportError(f"{path}: '{key}' stored as {layout}")
            payload = b"".join(verified.read.block(b.id)[: b.payload_bytes] for b in supply.blocks)
            shape = tuple(values[key]["shape"])
            value = torch.frombuffer(bytearray(payload), dtype=torch.float32).reshape(shape)
            counts["f32"] += 1
            if not bits_equal(value, state[key].detach().to(torch.float32)):
                differing.append(key)
    return {"counts": dict(counts), "differing": differing}


# ---------------------------------------------------------------------------
# Gate 3 — host tables
# ---------------------------------------------------------------------------


def stage0_records(directory: Path) -> dict[tuple[int, int], dict[str, torch.Tensor]]:
    """The stage-0 VAE records (`moss.vae_reference`); missing records fail loudly."""
    found = {}
    for height, width in SIZES:
        path = directory / f"vae-{height}x{width}.safetensors"
        if not path.is_file():
            raise VaeExportError(
                f"no stage-0 VAE record at {path} — run `python -m moss.vae_reference` or pass"
                " `--stage0`"
            )
        found[(height, width)], _ = records.read_record(path, key=vae_reference.METADATA_KEY)
    return found


def host_tables_gate(
    tables: Mapping[str, torch.Tensor],
    vae: Any,
    goldens: Mapping[VaeCase, Mapping[str, torch.Tensor]],
    stage0: Mapping[tuple[int, int], Mapping[str, torch.Tensor]],
) -> dict[str, Any]:
    """Gate 3 (module docstring); `failures` empty = passed."""
    failures: list[str] = []
    if set(tables) != {"mean", "std"}:
        failures.append(f"{BATCH_NORM_ASSET} tables {sorted(tables)}")
    channels = int(vae.bn.running_mean.shape[0])
    for key in ("mean", "std"):
        if key in tables and (tables[key].dtype, list(tables[key].shape)) != (
            torch.float32,
            [channels],
        ):
            failures.append(
                f"{BATCH_NORM_ASSET} '{key}' is {tables[key].dtype} {list(tables[key].shape)}"
            )
    if failures:
        return {"failures": failures}
    checked = 0
    inputs = [(case.name, g["latents"], g["z"]) for case, g in goldens.items()]
    inputs += [
        (f"stage0 {h}x{w}", record["latents"], record["vae.z"]) for (h, w), record in stage0.items()
    ]
    for label, latents, z in inputs:
        upstream = pipeline_ref.denormalize(vae, latents)
        host = host_denormalize(tables, latents)
        if not bits_equal(host, upstream):
            failures.append(f"{label}: latents · std + mean differs from upstream's denormalize")
        if not bits_equal(pipeline_ref.unpatchify(host), z):
            failures.append(f"{label}: the unpatchified latents differ from the decode input")
        checked += 1
    for (height, width), record in stage0.items():
        if not bits_equal(record["denormalized"], pipeline_ref.denormalize(vae, record["latents"])):
            failures.append(f"stage0 {height}x{width}: the record's denormalized differs")
        twin = [
            g
            for case, g in goldens.items()
            if (case.height, case.width, case.seed) == (height, width, vae_reference.SEED)
        ]
        if len(twin) != 1:
            failures.append(f"stage0 {height}x{width}: {len(twin)} goldens of the stage-0 seed")
            continue
        for key, record_key in (
            ("latents", "latents"),
            ("z", "vae.z"),
            ("decoded", "vae.decoded"),
            ("image", "image"),
        ):
            if not bits_equal(twin[0][key], record[record_key]):
                failures.append(f"stage0 {height}x{width}: golden '{key}' ≠ record '{record_key}'")
    return {"checked": checked, "failures": failures}


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def container_paths(directory: Path) -> dict[str, Path]:
    return {name: directory / name / MODEL_FILE for name in GRAPHS}


def write_graphs(directory: Path, vae: Any, **bounds: int) -> dict[str, Any]:
    """Trace, check and write the three containers into `directory`; returns gate 1's report."""
    modules = graph_modules(vae)
    paths = container_paths(directory)
    limits = [bounds.get("latent_max", LATENT_MAX), bounds.get("sequence_max", SEQUENCE_MAX)]
    reports: dict[str, Any] = {}
    committed: dict[str, IrGraph] = {}
    for name in GRAPHS:
        traced = trace(name, modules[name], vae, **bounds)
        structure = inspect_structure(name, traced, modules[name], vae, bounds=limits)
        paths[name].parent.mkdir(parents=True)
        assets = {BATCH_NORM_ASSET: batch_norm_asset(vae)} if name == VAE_DECODER_PRE else {}
        tensors = {key: value.detach().to(torch.float32) for key, value in traced.tensors.items()}
        committed[name] = publish_model(
            paths[name],
            traced.graph,
            tensors,
            provenance=provenance(),
            graph_name=name,
            assets=assets,
        )
        reports[name] = structure
    storage = inspect_storage(committed, vae)
    failures = {n: r["failures"] for n, r in reports.items() if r["failures"]}
    if storage["failures"]:
        failures["storage"] = storage["failures"]
    return {"graphs": reports, "storage": storage, "failures": failures}


def eager_gate(
    vae: Any,
    cases: Sequence[VaeCase],
    goldens: Mapping[VaeCase, Mapping[str, torch.Tensor]] | None = None,
    containers: Mapping[str, Path] | None = None,
    *,
    clock: records.Clock,
) -> dict[str, Any]:
    """Gate 2 on the cases (module docstring); `failures` empty = passed."""
    graphs = graph_modules(vae)
    reports = []
    for case in cases:
        z = goldens[case]["z"] if goldens is not None else case_tensors(vae, case)["z"]
        report = {"case": case.name, **eager_case(vae, graphs, z)}
        reports.append(report)
        clock.lap(f"eager {case.name}")
    failures: dict[str, Any] = {r["case"]: r["failures"] for r in reports if r["failures"]}
    checked: dict[str, Any] = {}
    if containers is not None:
        for name, path in containers.items():
            checked[name] = container_weights(path, vae)
            if checked[name]["differing"]:
                failures[f"container {name}"] = checked[name]["differing"][:5]
    return {"cases": reports, "container_weights": checked, "failures": failures}


def load_checked_vae() -> Any:
    """The pinned VAE in f32, its file hash checked against the pin."""
    snapshot = sources.local_snapshot()
    actual = sources.sha256_of(snapshot / sources.VAE_FILE)
    pinned = sources.WEIGHT_SHA256[sources.VAE_FILE]
    if actual != pinned:
        raise VaeExportError(f"{sources.VAE_FILE}: sha256 {actual} ≠ the pin {pinned}")
    vae = pipeline_ref.load_vae(snapshot)
    vae_patch.assert_supported(vae)
    return vae


def write_series(stage0_dir: Path, *, cap: bool) -> dict[str, Any]:
    """Export into a staging directory, run the three gates, then move the series into place."""
    clock = records.Clock()
    stage0 = stage0_records(stage0_dir)
    vae = load_checked_vae()
    clock.lap("load vae (f32)")
    cases = CASES_WITH_CAP if cap else CASES
    summary: dict[str, Any] = {"series": str(SERIES), "cases": [c.name for c in cases]}
    with staged_publication(SERIES) as staged:
        staged.mkdir(parents=True)
        structure = write_graphs(staged, vae)
        summary["structure"] = structure
        if structure["failures"]:
            raise VaeExportError(f"gate 1 (structure) failed: {structure['failures']}")
        clock.lap("write graphs")
        goldens: dict[VaeCase, dict[str, torch.Tensor]] = {}
        for case in cases:
            goldens[case] = case_tensors(vae, case)
            metadata = {
                "case": case.name,
                "role": case.role,
                "size": [case.height, case.width],
                "seed": case.seed,
                "reference": GOLDEN_REFERENCE,
                "noise": GOLDEN_NOISE,
                "source": {"repo": sources.SOURCE.repo, "revision": sources.SOURCE.revision},
                "versions": records.versions(),
                "torch_threads": torch.get_num_threads(),
            }
            records.write_record(golden_path(staged, case), goldens[case], metadata, key=GOLDEN_KEY)
            clock.lap(f"golden {case.name}")
        tables = load(read_asset(container_paths(staged)[VAE_DECODER_PRE], BATCH_NORM_ASSET))
        summary["host_tables"] = host_tables_gate(tables, vae, goldens, stage0)
        if summary["host_tables"]["failures"]:
            raise VaeExportError(
                f"gate 3 (host tables) failed: {summary['host_tables']['failures']}"
            )
        gate = eager_gate(vae, cases, goldens, container_paths(staged), clock=clock)
        summary["eager"] = gate
        if gate["failures"]:
            raise VaeExportError(f"gate 2 (eager equivalence) failed: {gate['failures']}")
        summary["files"] = {
            str(path.relative_to(staged)): path.stat().st_size
            for path in sorted(staged.rglob("*"))
            if path.is_file()
        }
    summary["seconds"] = {**clock.phases, "total": clock.total()}
    summary["peak_rss_gib"] = round(records.peak_rss_gib(), 2)
    summary["environment"] = records.environment()
    summary["passed"] = True
    return summary


def inspect_only() -> dict[str, Any]:
    """Gate 1's storage-independent items on the traces (no series written)."""
    vae = load_checked_vae()
    modules = graph_modules(vae)
    graphs = {
        name: inspect_structure(
            name,
            trace(name, modules[name], vae),
            modules[name],
            vae,
            bounds=[LATENT_MAX, SEQUENCE_MAX],
        )
        for name in GRAPHS
    }
    failures = {name: g["failures"] for name, g in graphs.items() if g["failures"]}
    return {
        "describe": vae_patch.describe(vae),
        "graphs": graphs,
        "failures": failures,
        "passed": not failures,
    }


def eager_only(*, cap: bool) -> dict[str, Any]:
    """Gate 2 against the written series (its containers' weights included)."""
    clock = records.Clock()
    vae = load_checked_vae()
    paths = container_paths(SERIES)
    if not all(path.parent.is_dir() for path in paths.values()):
        raise VaeExportError(f"no written series at {SERIES} — run `write` first")
    gate = eager_gate(vae, CASES_WITH_CAP if cap else CASES, containers=paths, clock=clock)
    return {
        **gate,
        "seconds": {**clock.phases, "total": clock.total()},
        "peak_rss_gib": round(records.peak_rss_gib(), 2),
        "passed": not gate["failures"],
    }


COMMANDS: Mapping[str, Callable[[argparse.Namespace], dict[str, Any]]] = {
    "inspect": lambda _args: inspect_only(),
    "write": lambda args: write_series(
        args.stage0 or records.default_reference_dir(), cap=args.cap
    ),
    "eager": lambda args: eager_only(cap=args.cap),
}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("command", choices=sorted(COMMANDS))
    parser.add_argument("--out", type=Path, default=None, help="where the summary goes")
    parser.add_argument("--cap", action="store_true", help="write / eager: add the area-cap case")
    parser.add_argument("--stage0", type=Path, default=None, help="write: the stage-0 record dir")
    args = parser.parse_args(argv)
    if args.stage0 is not None and args.command != "write":
        parser.error("--stage0 applies to write only")
    if args.cap and args.command == "inspect":
        parser.error("--cap applies to write and eager only")
    summary = COMMANDS[args.command](args)
    document = {"command": args.command, "result": summary, "versions": records.versions()}
    text = json.dumps(document, ensure_ascii=False, indent=1)
    print(text)
    out = default_out() if args.out is None else args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.command}.json").write_text(text + "\n", encoding="utf-8")
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
