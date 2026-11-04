"""Measure the quality loss of Irodori-TTS v4 per quantization configuration on torch CPU (the
measurement script of wave 2).

A quality gate in the same spirit as `measure_quant_sbv2.py` / `measure_quant_anima.py` — zero
lines of GPU code; what it measures is **the quality of the quantization itself** (under ADR
0006's fake-quant methodology the E2E tests measure only implementation error, so quality has to
be measured on a separate axis). The question it wants to answer is one:

    **Does S (the frame count) move?** — putting duration in i8 can move the utterance length
    (the same axis as SBV2 w8's w_ceil 198→196, ADR 0029). If the latent check's "S / forwards
    match exactly" breaks, the distribution's i8 preset has to be mixed (keeping only duration
    unquantized) (ADR 0050 decision 6).

    cd tools/export-recipes
    uv run --with descript-audiotools --with einops --with 'transformers==5.14.1' \
        python -m irodori.measure_quant --config f32
    uv run ... python -m irodori.measure_quant --config i8-all
    uv run ... python -m irodori.measure_quant --config i4-all
    uv run ... python -m irodori.measure_quant --config gptq-rtn   # calibrated (wave J-2)
    uv run ... python -m irodori.measure_quant --scan   # w4 target scale and estimates only
    uv run ... python -m irodori.measure_quant          # no measurement; rebuild the report

**One run = one configuration** (`--config`). The full loop is a torch CPU run that does 240 DiT
forwards per case, so packing 9 configurations into one process makes an indivisible block of
several hours. Each configuration can be invoked independently: per-config JSON / WAV / z are
left under `--out`, and the last run compares them all and writes `report.json` and a Markdown
table (run the baseline `f32` first — the other configurations read its outputs).

## Configurations (the spellings of `--config`)

| Configuration    | Weights             | What it answers                                         |
| ---------------- | ------------------- | ------------------------------------------------------- |
| `f32`            | none (**baseline**) | no vacuous check (byte-identical to the golden's z)     |
| `f16`            | every role f16      | counterpart of the series of ADR 0050 wave 1            |
| `i8-all`         | every role i8       | the distribution's `i8` preset (25.2% of it)            |
| `i8-mixed`       | i8 but duration     | mixed-table candidate (ADR 0050 decision 6, option (i)) |
| `w8a8`           | every role i8 + a8  | the `i8-a8` preset (i8 activations: DiT Linear only)    |
| `i8-<role>-only` | that role in i8     | **5 orthogonal decompositions** (1:1 with `load_*`)     |

The 5 w4 configurations (ADR 0069 addendum 5) apply **the 4 methods that won the screening**
(g=32 fixed) to every role, widening the rounded ops to the same 5 kinds as i8 (an explicit
opt-in through `op_types` — linear / the conv family / embedding). **This is not a shippable
form** (the i4 execution path is linear-only = decision 5, and the conv family's scales cannot
structurally reach emit) — what is measured here is the upper bound of "how far quality recovers
when even non-linear ops are rounded"; not one line of the storage acceptance set or the runtime
is touched.

| Configuration       | Method          | What it answers                                   |
| ------------------- | --------------- | ------------------------------------------------- |
| `i4-all`            | `rtn-i4-g32`    | **does S move** in all-w4 (first check) + quality |
| `i4-mixed`          | `rtn-i4-g32`    | fallback on the axis of `i8-mixed` (duration f32) |
| `nf4-all`           | `nf4`           | how far a normal-quantile grid recovers           |
| `mxfp4-all`         | `mxfp4`         | what power-of-two scales (E8M0) buy — 4.25 bpw    |
| `kmeans-shared-all` | `kmeans:shared` | what a learned shared table buys (+16×f32 table)  |

`i4-mixed` exists only for RTN because whether S moves is decided by **the coarseness of the
rounding** (mixed versions of all 4 methods would only ask the same question 4 times; once RTN
settles the shape of the fallback, it carries over to the other methods).

`kmeans-shared-all` concatenates the normalized values of its targets into one before fitting
the table, so a full fit of the dit role (750 million elements in the rounded tree) exceeds
20 GB including the f64 work area. `--kmeans-fit-stride` is the core's escape hatch (**only the
table fit** uses an evenly spaced subsample; application is always to everything), and the value
used is recorded in the output.

## Calibrated rounding ({@link CALIB_CONFIGS}, wave J-2)

The 5 w4 configurations above round by looking at the weights alone (the RTN family).
**Calibrated rounding** (GPTQ — the core's `karume.quant_calib`) re-chooses rounding targets
**within the same grid** from "the activations that actually flow through the layer"; it adds 3
storage grids to the same measurement series:

| Configuration | Method / grid        | What it answers                                     |
| ------------- | -------------------- | --------------------------------------------------- |
| `gptq-rtn`    | `gptq/rtn`           | recovery on the shippable grid (the main candidate) |
| `gptq-nf4`    | `gptq/nf4`           | normal-quantile grid × calibration                  |
| `gptq-kmeans` | `gptq/kmeans_shared` | learned table (**scoped per layer**) × calibration  |

The targets are **limited to the `nn.Linear` of the DiT block sequence** ({@link CALIB_TARGET}),
**a different set** from `i4-linear` (the linears of every role) — the calibration driver
(`calibrate_stages`) is closed over the `nn.Linear` inside the stages, and backbone / projector /
speaker / duration / codec, as well as the DiT's `in_proj` / `out_proj` / `cond_module`, are
outside the stages.

The calibration input is **every step of the reference loops (f32, CFG included) of the 2
{@link irodori.pipeline_ref.PIPELINE_CASES}**; at each step the `(hidden, accompanying
arguments)` into the first DiT block is captured as one batch ({@link capture_case_batches}).
The activation distribution of a diffusion model moves with t, so unless it is captured **across
steps**, the distribution of the later steps escapes calibration. The accompanying arguments
(`cond_embed` / the normalized condition states / masks / the RoPE table) are built once
**before** the block loop and the same ones are passed to every block, so they can be invariant
across stages (no `advance_kwargs` needed). The condition states get their norm once per
generation in `dit-context` (`DitContextGraph`), and `cond_embed` / masks / the RoPE table are
built by `DitGraph.forward` — the stages recompute the condition-side K/V projections **inside the
block** (so that GPTQ sees the inputs of those 72 weights — even after ADR 0114 split the shipped
graph, a calibration stage is the whole block).

MUST: the capture happens only once, **before any rounding is applied** (pristine). That the
stage split is **bit-identical** to one forward of the two distribution wrappers is also measured
on the spot before rounding ({@link assert_stage_split_matches_graph}) — if the copy of the block
body drifted from upstream, the values would quietly become something else while the table still
came out.

**The linear-only form is not a configuration** — `--scan` counts and prints the target scale
(count, channels, elements, reduction ratio). If the model is conv-dominated, rounding only the
linears barely moves the distribution size and is not worth tens of minutes of full loop, so this
table decides whether to run it.

Weights whose quantization axis (the last dimension folded by `channel_rows`) is not divisible
by g32 are excluded from the targets with `include` (no partial groups = ADR 0069 decision 2).
**The exclusion list is written to the output every time** — excluding silently would leave only
the non-divisible layers in f32 behind a table that reads as "every role was rounded".

The size columns (effective bpw / projected MiB / formula) are **projections by formula**, not
measurements. The counts cover **only the weights in the shipped graphs**
(`irodori.export.load_dit` also holds the copy of the backbone that is not in the graph, so
counting the rounded tree would inflate it by weights that do not ship). The rounded tree itself
stays on the same role modules as i8 — if "what was rounded" disagrees with i8, the comparison
between methods does not hold.

`w8a8` adds, through module hooks, a per-token i8 fake-quant of the inputs of the DiT's
`nn.Linear` (`karume.act_quant.quantize_rows` — the numeric mirror of the runtime's
`quantize-rows.ts`) on top of **the same rounded weights** as `i8-all`. Hooks are safe because
the only things `patch.py` calls through `functional` are `rms_norm` / `silu`, so the Linears to
quantize remain real modules (in SBV2 the direct `functional.conv1d` calls made the hooks silently
miss them — the detection limit of ADR 0029). **The number applied is printed every time, and 0
fails loudly** (no shape in which "numbers for w8 were taken while meaning w8a8" goes unnoticed).
Activation quantization is limited to the DiT to match the distribution's preset (`session`
reaches only the `dit` Session — `pipeline.ts` on the models side).

The 5 orthogonal decompositions are `dit` / `backbone` / `speaker` / `duration` / `codec`. **The
sum of the 5 is not `i8-all`** — the 2 projectors (6.83 MB each) do not get an axis of their own
(two orders of magnitude smaller, with no motivation to quantize them alone — the sizeBreakdown
of the quantization recon). They are included in `i8-all` / `i8-mixed`.

## Fixing the time grid (a MUST that makes SNR / LSD meaningful quantities)

duration's output decides S, so if quantization flips a single banker's rounding, **the latent
length itself changes**, the waveform lengths disagree between configurations and SNR is
undefined. So every configuration but the baseline runs **with the `f32` S fixed from outside**
(`irodori.pipeline_ref.run_case(frames_override=…)` — the same choice as `measure_quant_sbv2.py`
fixing the time grid with the dump side's `w_ceil`).

**The S each configuration predicts itself** is kept separately as `predictedS`. This is **the
main result of this script**, and the quality metrics read as "a comparison on the same time
grid" — in production the utterance length moves too.

## Orthogonality checks (blocking vacuous checks)

How each role acts is structurally determined, so **"does not change" is a check as well as
"changes"** (with only one side, both a rounding that passes through and a leaking scope would be
green):

- `duration` **acts only on S** (`log_frames` → banker's rounding → S; no other stage reads it)
  ⇒ with the grid fixed, `i8-duration-only`'s z and waveform are correctly **bit-identical** to
  the baseline
- `codec` **acts only on the waveform** (after the latent) ⇒ z is bit-identical, the waveform
  changes
- `speaker` acts on z **only in cases with a reference** (without one, the host's zero-supply
  short-circuit means the encoder does not run — ADR 0048) ⇒ green means `i8-speaker-only`
  changes in full and is bit-identical in no-ref
- everything else (backbone / projector / dit) changes z
- `w8a8` **has a z different from the weight-only `i8-all` too** (detects the activation shim
  passing through — if it passed through, z would be bit-identical to `i8-all`, and looking only
  at the difference from the baseline would misread it as "it has an effect")

## Output

    <--out>/<config>.json               one configuration's measurements (S / forwards /
                                        seconds / quality metrics)
    <--out>/<config>.<case>.safetensors z and waveform (for f32, the baseline that later
                                        configurations read)
    <--out>/<config>.<case>.wav         for listening (16-bit PCM; the final ruling is by ear
                                        = the user)
    <--out>/report.json                 comparison of all configurations (the Markdown table
                                        goes to stdout)

NOTE: the tail trim (`find_flattening_point`) and the cut by seconds are **not applied** — if the
cut position moved per configuration, the waveform lengths would not line up and SNR would be
undefined. The production length is kept by `predictedSeconds`.

NOTE: the metric implementations (`snr_db` / `log_spectral_distance` / `rel_rms`) are
**duplicated** from `measure_quant_sbv2.py`. Turning them into a shared module belongs to the
exporter's io-sharing wave (just before Gemma); until then each script stays self-contained (the
D3 ruling of the exhaustive review of 2026-08-12).
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import time
import wave
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from types import MappingProxyType
from typing import Any, NamedTuple

import torch
from safetensors.torch import load_file, save_file
from torch import nn

from _shared.paths import BENCH_ROOT, SERIES_ROOT
from karume import act_quant
from karume.quant_calib import (
    CalibMethod,
    CalibReport,
    GridSpec,
    StageBatch,
    StageSpec,
    calibrate_stages,
)
from karume.quant_methods import (
    DEFAULT_CODEBOOK_LEVELS,
    fake_quant_kmeans,
    fake_quant_mxfp4,
    fake_quant_nf4,
)
from karume.quantize import (
    DEFAULT_GROUP_SIZE,
    QUANT_MODULE_TYPES,
    fake_quant_int4,
    iter_quant_targets,
)

from . import export as ex
from . import patch
from . import pipeline_ref as ip
from .dacvae import export as dv

#: Where the bench outputs go (kept apart from the assets — `rm -rf outputs/bench` does not take
#: the series with it).
#: NOTE: the date is fixed at process start. **One run = one configuration**, so when the runs to
#: be compared across configurations span a date change, pass `--out` explicitly to keep them in
#: the same place.
DEFAULT_OUT = BENCH_ROOT / "karume-irodori-v4-small" / f"{date.today().isoformat()}_quant-sim"

#: The comparison target that guards against a vacuous check (the full-loop golden
#: `irodori/pipeline_ref.py` writes to the f32 series).
DEFAULT_GOLDEN_DIR = SERIES_ROOT / "irodori-v4-small" / "pipeline"

#: The spelling of the baseline configuration (other configurations read its outputs to
#: compare).
BASE_CONFIG = "f32"

#: Roles = graph boundaries (1:1 with `irodori.export.load_*`) + the codec.
ROLE_CODEC = "codec"
ROLES: tuple[str, ...] = (
    ex.TARGET_BACKBONE,
    ex.TARGET_TEXT_PROJ,
    ex.TARGET_CAPTION_PROJ,
    ex.TARGET_SPEAKER,
    ex.TARGET_DURATION,
    ex.TARGET_DIT,
    ROLE_CODEC,
)

#: Roles that can move the latent (z). `duration` acts only on S and `codec` comes after the
#: latent, so neither moves z by a single bit in the grid-fixed comparison (the "orthogonality
#: checks" of the module docstring).
LATENT_ROLES: frozenset[str] = frozenset(ROLES) - {ex.TARGET_DURATION, ROLE_CODEC}

#: Roles that act on z **only when the case has a reference**. In cases without a reference
#: the speaker encoder does not run (the host's zero-supply short-circuit — ADR 0048), so z is
#: correctly unchanged when `speaker` is rounded (measured 2026-08-12: `i8-speaker-only`'s no-ref
#: is bit-identical to the baseline in both z and waveform).
REFERENCE_ONLY_ROLES: frozenset[str] = frozenset({ex.TARGET_SPEAKER})

#: Case name → whether it has a reference voice (the table that lets the orthogonality checks
#: split expectations by case condition).
CASE_HAS_REFERENCE: Mapping[str, bool] = MappingProxyType(
    {case.name: case.reference is not None for case in ip.PIPELINE_CASES}
)


# ---- w4 methods (the 4 winners of the screening) ------------------------------

#: The w4 group length = **fixed at 32** (the core's default = the value settled in ADR 0069
#: addendum 1). g is not varied together with the method because once a method difference and a
#: g difference mix, "which one had the effect" cannot be told (re-evaluating the g axis comes
#: after the method is settled — item 3 of ADR 0069 addendum 5).
W4_GROUP_SIZE = DEFAULT_GROUP_SIZE

#: The op kinds w4 rounds = the same 5 kinds as i8 (an explicit opt-in through `op_types` of
#: `fake_quant_int4` — item 1 of ADR 0069 addendum 5). **Not the shipping path** — the i4
#: execution path is linear-only (decision 5), and the conv family's scales become rank 2 from the
#: receptive-field flattening and cannot structurally reach emit. Widening here measures "how far
#: quality recovers when even non-linear ops are rounded" without touching the runtime.
W4_OP_TYPES: tuple[type[nn.Module], ...] = QUANT_MODULE_TYPES

#: Bits of one k-means table (16 centroids × f32).
CODEBOOK_BITS = DEFAULT_CODEBOOK_LEVELS * 32


class TargetCounts(NamedTuple):
    """Counts of the w4 target tensor set (the input of the size estimate).

    `channels` is the sum of the row counts of `karume.quantize.channel_rows`, `modules` the
    number of layers.
    """

    modules: int
    channels: int
    elements: int

    @property
    def groups(self) -> int:
        """Total number of g32 groups (used by the size estimates of methods that need one scale
        per group)."""
        return self.elements // W4_GROUP_SIZE


def total_counts(counts: Sequence[TargetCounts]) -> TargetCounts:
    """Add the per-role counts into one (size estimates read the whole model)."""
    return TargetCounts(
        modules=sum(item.modules for item in counts),
        channels=sum(item.channels for item in counts),
        elements=sum(item.elements for item in counts),
    )


class Excluded(NamedTuple):
    """A weight excluded from the targets because its quantization axis is not divisible by g32
    (the list always goes into the output)."""

    module: str
    fqn: str
    axis_length: int
    elements: int


def group_axis_length(weight: torch.Tensor, axis: int) -> int:
    """The quantization axis length when folded by `karume.quantize.channel_rows` (= elements /
    channels).

    `channel_rows` is not actually called because flattening a `ConvTranspose1d` is a **copy** via
    `reshape` (a count that only needs the shape should not allocate a weight-sized temporary).
    """
    return weight.numel() // int(weight.shape[axis])


class RoleScan(NamedTuple):
    """The w4 target scale of one tree (the side that fits g32 and the side excluded as not
    divisible)."""

    counts: TargetCounts
    excluded: tuple[Excluded, ...]


def scan_targets(model: nn.Module, op_types: tuple[type[nn.Module], ...]) -> RoleScan:
    """Split the targets of `iter_quant_targets` into "fits g32 / excluded as not divisible".

    Target selection goes through the core (the same `iter_quant_targets` as the storage path) —
    a separately copied implementation would silently split "the targets measured" from "the
    targets rounded". Making no partial group is the storage side's own constraint (ADR 0069
    decision 2), so non-divisible weights are not rounded but **listed**.
    """
    modules = channels = elements = 0
    excluded: list[Excluded] = []
    for fqn, weight, axis in iter_quant_targets(model, op_types):
        length = group_axis_length(weight, axis)
        if length % W4_GROUP_SIZE:
            excluded.append(Excluded(fqn.removesuffix(".weight"), fqn, length, weight.numel()))
            continue
        modules += 1
        channels += int(weight.shape[axis])
        elements += weight.numel()
    return RoleScan(TargetCounts(modules, channels, elements), tuple(excluded))


def aligned_include(excluded: Sequence[Excluded]) -> Callable[[str], bool]:
    """An `include` predicate that drops the modules not divisible by g32 (looked up by module
    FQN)."""
    dropped = {item.module for item in excluded}
    return lambda name: name not in dropped


@dataclass(frozen=True)
class W4Method:
    """One w4 rounding method — binds the name, how the rounding is applied and the size-estimate
    formula into one row.

    Scattering the three over separate tables would silently split "the method whose quality was
    measured" from "the method whose size was estimated" (adding a method would update only one
    of them). The rounding implementations are shared in the core (`karume.quantize` /
    `karume.quant_methods`); all this holds is the dispatch and the formula.
    """

    #: The same method name as in the screening (appears as is in the tables and JSON).
    name: str
    #: Applies the rounding to the model in place (`(model, op_types, include, fit_stride)` → a
    #: report with `describe()`). `op_types` is decided by the configuration
    #: ({@link Recipe.op_types}) — baking it in here could split "the targets counted" from "the
    #: targets rounded" per configuration. Only {@link W4_KMEANS_SHARED} reads `fit_stride`.
    round_model: Callable[[nn.Module, tuple[type[nn.Module], ...], Callable[[str], bool], int], Any]
    #: Projected bits of the target set (`(counts, number of tables)`).
    projected_bits: Callable[[TargetCounts, int], float]
    #: The formula of `projected_bits` (**written to the output as is** — so that the
    #: projection's assumptions can be traced from the table).
    formula: str
    #: Whether the table fit can use a subsample (whether `--kmeans-fit-stride` applies to the
    #: method).
    samples_fit: bool = False


#: RTN (= the storage form `i4`, ADR 0069 decision 3). **The baseline** of the w4 columns; the
#: other 3 methods read as differences from it.
W4_RTN = W4Method(
    "rtn-i4-g32",
    lambda model, op_types, include, _stride: fake_quant_int4(
        model, W4_GROUP_SIZE, include=include, op_types=op_types
    ),
    lambda counts, _tables: 4 * counts.elements + 32 * counts.groups,
    "4bit + g32 f32 scale = 5.0 bpw",
)

W4_NF4 = W4Method(
    "nf4",
    lambda model, op_types, include, _stride: fake_quant_nf4(
        model, W4_GROUP_SIZE, include=include, op_types=op_types
    ),
    lambda counts, _tables: 4 * counts.elements + 32 * counts.groups,
    "4bit + g32 f32 scale = 5.0 bpw",
)

W4_MXFP4 = W4Method(
    "mxfp4",
    lambda model, op_types, include, _stride: fake_quant_mxfp4(
        model, W4_GROUP_SIZE, include=include, op_types=op_types
    ),
    lambda counts, _tables: 4 * counts.elements + 8 * counts.groups,
    "4bit + g32 E8M0 scale = 4.25 bpw",
)

#: The k-means shared table. **One table per role** (`tables` = the number of roles with
#: targets) — the distribution ships each role as a separate IR artifact, so the storage side has
#: no place for one table shared across artifacts. Rounding is also applied closed per role
#: (`apply_weight_quant`), so the estimated and actual table counts agree.
#:
#: NOTE: `shared` concatenates the normalized values of its targets into one before fitting the
#: table, so for the dit role (750 million elements in the rounded tree) it exceeds 20 GB
#: including the f64 work area. `--kmeans-fit-stride` is the core's escape hatch (**only the
#: table fit** uses an evenly spaced subsample; application is always to everything).
W4_KMEANS_SHARED = W4Method(
    "kmeans:shared",
    lambda model, op_types, include, stride: fake_quant_kmeans(
        model, "shared", W4_GROUP_SIZE, include=include, op_types=op_types, fit_stride=stride
    ),
    lambda counts, tables: 4 * counts.elements + 32 * counts.groups + CODEBOOK_BITS * tables,
    "4bit + g32 f32 scale + one 16×f32 table per role",
    samples_fit=True,
)


# ---- calibrated rounding configurations (wave J-2) -------------------------------

#: The target name of the calibrated configurations (the spelling of "what was rounded", next to
#: `w4.opTypes` in the JSON). **A different set** from the w4 methods' `i4-linear` (the linears of
#: every role) — the calibration driver is closed over the `nn.Linear` inside the stages, so
#: neither the outside of the DiT blocks (`in_proj` / `out_proj` / `cond_module`) nor other roles
#: are included.
CALIB_TARGET = "dit:blocks-linear"

#: The **in-model FQN prefix** of the DiT block sequence (`.blocks` of the `TextToLatentRFDiT`
#: that `load_dit` builds). Passed to the stages to put the scale ledger's keys in the same FQN
#: space as `Int4Report` (the contract of the core's `StageSpec`).
DIT_BLOCK_PREFIX = "blocks"


@dataclass(frozen=True)
class CalibConfig:
    """The specification of one calibrated rounding (method × storage grid).

    `projected_bits` / `formula` are the same holders as in {@link W4Method}, and
    {@link project_size} accepts both — calibration **does not change the grid by a single byte**
    (it only re-chooses rounding targets within the same grid), so the storage-form formula can be
    shared with the methods.
    """

    #: Configuration name (the spelling of `--config`; appears as is in the tables and JSON).
    name: str
    #: The core's calibration method (`karume.quant_calib`).
    method: CalibMethod
    #: The storage grid of the rounding targets.
    grid: GridSpec
    #: Projected bits of the target set (`(counts, number of tables)`).
    projected_bits: Callable[[TargetCounts, int], float]
    #: The formula of `projected_bits` (**written to the output as is**).
    formula: str


def _calib_group_scaled_bits(counts: TargetCounts, _tables: int) -> float:
    """Total bits of a grid with one f32 scale per group (shared by `rtn` / `nf4`)."""
    return 4 * counts.elements + 32 * counts.groups


def _calib_codebook_bits(counts: TargetCounts, tables: int) -> float:
    """Total bits of `kmeans_shared` — **one table per layer** (`tables` = the number of target
    layers)."""
    return _calib_group_scaled_bits(counts, tables) + CODEBOOK_BITS * tables


#: The 3 calibrated configurations (listed in this order in the table). The rounding is shared in
#: the core's `karume.quant_calib`; all this holds is the dispatch and the projection formulas.
#: AWQ is not included because its equivalence factor `s` cannot be stored on its own (it needs a
#: fold or a companion), making a column with no way out (the MUST of the core's module
#: docstring).
CALIB_CONFIGS: tuple[CalibConfig, ...] = (
    CalibConfig(
        "gptq-rtn",
        "gptq",
        GridSpec(kind="rtn", group_size=W4_GROUP_SIZE),
        _calib_group_scaled_bits,
        "4bit + g32 f32 scale = 5.0 bpw (stored as RTN i4 itself — calibration changes only the"
        " rounding targets)",
    ),
    CalibConfig(
        "gptq-nf4",
        "gptq",
        GridSpec(kind="nf4", group_size=W4_GROUP_SIZE),
        _calib_group_scaled_bits,
        "4bit + g32 f32 scale = 5.0 bpw (the grid is NF4's fixed table)",
    ),
    CalibConfig(
        "gptq-kmeans",
        "gptq",
        GridSpec(kind="kmeans_shared", group_size=W4_GROUP_SIZE),
        _calib_codebook_bits,
        "4bit + g32 f32 scale + one 16×f32 table **per layer**"
        " (scoped per layer — a different formula from kmeans:shared with one table per role)",
    ),
)

CALIB_NAMES: tuple[str, ...] = tuple(config.name for config in CALIB_CONFIGS)


def check_fit_stride(recipe: Recipe, fit_stride: int) -> None:
    """Fail loudly when `--kmeans-fit-stride` is passed to a configuration it has no effect on
    (do not silently ignore it)."""
    if fit_stride == 1:
        return
    if recipe.calib is not None:
        raise SystemExit(
            f"--kmeans-fit-stride {fit_stride} has no effect on calibrated configurations"
            f" ({recipe.weight} fits its tables **per layer**, so even a full fit has a small work"
            " area)"
        )
    if recipe.method is None or not recipe.method.samples_fit:
        raise SystemExit(
            f"--kmeans-fit-stride {fit_stride} is only for methods that fit the table on a"
            f" subsample (this configuration rounds with {recipe.weight})"
        )


@dataclass(frozen=True)
class SizeProjection:
    """The **projected** size of method × target set (not measured — the formula is
    {@link W4Method.formula})."""

    counts: TargetCounts
    tables: int
    bits: float
    formula: str

    @property
    def bits_per_weight(self) -> float:
        return self.bits / self.counts.elements

    @property
    def projected_mib(self) -> float:
        return self.bits / 8 / 1024**2

    @property
    def f32_mib(self) -> float:
        """MiB of the same target set held as f32 (the projection's denominator — so the
        reduction ratio can be read from the table)."""
        return self.counts.elements * 4 / 1024**2


def project_size(
    method: W4Method | CalibConfig, counts: TargetCounts, tables: int
) -> SizeProjection:
    """Apply a method's formula to the counts of a target set (`tables` = the number of
    artifacts that carry a table).

    Calibrated configurations ({@link CalibConfig}) are accepted by the same holder — calibration
    does not change the grid by a single byte, so the storage-form formula can be shared with the
    methods.
    """
    return SizeProjection(counts, tables, method.projected_bits(counts, tables), method.formula)


@dataclass(frozen=True)
class Recipe:
    """The quantization recipe of one configuration (`weight` of `None` means no rounding)."""

    #: The spelling of the rounding (storage dtype `f16` / `i8`, or a w4 method name).
    weight: str | None
    roles: tuple[str, ...]
    #: Whether to apply a per-token i8 fake-quant to the inputs of the DiT's `nn.Linear` (the
    #: activation side of w8a8).
    act_quant: bool = False
    #: The w4 method (`None` = `weight` is a storage-dtype spelling and goes through
    #: `irodori.export.fake_quant`).
    method: W4Method | None = None
    #: The op kinds w4 rounds (meaningful only with `method`). The default is the 5 kinds widened
    #: for measurement; narrowed to `(nn.Linear,)` it is **today's distribution-compatible form**
    #: (the i4 execution path — ADR 0069 decision 5).
    op_types: tuple[type[nn.Module], ...] = W4_OP_TYPES
    #: Calibrated rounding (`None` = rounding by looking at the weights alone). **Exclusive** with
    #: `method` — calibration is driven stage by stage, so it does not go through the per-role
    #: batch path of `apply_weight_quant`.
    calib: CalibConfig | None = None


def w4_recipe(
    method: W4Method,
    roles: tuple[str, ...],
    op_types: tuple[type[nn.Module], ...] = W4_OP_TYPES,
) -> Recipe:
    """The recipe of one w4 method (the `weight` spelling is the method name — no duplicated
    bookkeeping)."""
    return Recipe(method.name, roles, method=method, op_types=op_types)


#: The comparison partner of the activation shim (the configuration with **the same weights and
#: plain activations**). Pass-through detection looks at the difference from this one.
WEIGHT_ONLY_BASE = "i8-all"

#: The roles of the mixed form (only duration stays f32 — option (i) of ADR 0050 decision 6). i8
#: and w4 use **the same axis** (written separately, the two mixes would silently diverge when
#: only one of them moves).
MIXED_ROLES: tuple[str, ...] = tuple(role for role in ROLES if role != ex.TARGET_DURATION)

#: The 5 main configurations (the ones in the listening comparison and the distribution
#: ruling).
CONFIGS: Mapping[str, Recipe] = MappingProxyType(
    {
        BASE_CONFIG: Recipe(None, ()),
        "f16": Recipe("f16", ROLES),
        WEIGHT_ONLY_BASE: Recipe("i8", ROLES),
        # Option (i) of ADR 0050 decision 6 — only duration stays f32.
        "i8-mixed": Recipe("i8", MIXED_ROLES),
        # The distribution's `i8-a8` preset — the weights are byte-identical to `i8-all`; only
        # the DiT's activations are i8.
        "w8a8": Recipe("i8", ROLES, act_quant=True),
    }
)

#: Orthogonal decomposition (one graph boundary at a time in i8 — attributes the degradation).
DECOMPOSED_ROLES: tuple[str, ...] = (
    ex.TARGET_DIT,
    ex.TARGET_BACKBONE,
    ex.TARGET_SPEAKER,
    ex.TARGET_DURATION,
    ROLE_CODEC,
)
DIAGNOSTICS: Mapping[str, Recipe] = MappingProxyType(
    {f"i8-{role}-only": Recipe("i8", (role,)) for role in DECOMPOSED_ROLES}
)

#: The 4 winners of the w4 screening × the every-role form + RTN's mix that keeps duration
#: unquantized.
#:
#: The first check is on **the same axis** as i8 — does S move (ADR 0050 decision 6). As the
#: fallback in case it moves, only RTN gets a mixed form (whether S moves is decided by the
#: coarseness of the rounding, so mixed versions of all 4 methods would only ask the same
#: question 4 times; once RTN settles the shape of the fallback, it carries over to the other
#: methods).
#:
#: The op kinds of the every-role form are {@link W4_OP_TYPES} (linear / the conv family /
#: embedding).
#:
#: `i4-linear` is **today's distribution-compatible form** (RTN i4 × linear only = the only form
#: that can actually ship — ADR 0069 decision 5). The `--scan` measurement showed linears cover
#: 81.8% of the target elements, so the premise "skip it if the model is conv-dominated" did not
#: hold, and it is kept as a quality configuration (a follow-up after the full review of
#: 2026-08-19). codec is left out of the roles because codec has not a single linear (no role with
#: zero targets is run — that is an identity, not an exclusion, so the configuration still means
#: "the linears of every role"). There is no mixed linear form (duration excluded) — the
#: every-role measurement established that the S drift comes from the deformation of upstream
#: features rather than duration's weights (i4-mixed's forced-grid quality is bit-identical to
#: i4-all), so mixing cannot save S.
W4_LINEAR_ROLES: tuple[str, ...] = tuple(role for role in ROLES if role != ROLE_CODEC)
W4_CONFIGS: Mapping[str, Recipe] = MappingProxyType(
    {
        "i4-all": w4_recipe(W4_RTN, ROLES),
        "i4-linear": w4_recipe(W4_RTN, W4_LINEAR_ROLES, op_types=(nn.Linear,)),
        "i4-mixed": w4_recipe(W4_RTN, MIXED_ROLES),
        "nf4-all": w4_recipe(W4_NF4, ROLES),
        "mxfp4-all": w4_recipe(W4_MXFP4, ROLES),
        "kmeans-shared-all": w4_recipe(W4_KMEANS_SHARED, ROLES),
    }
)

#: Calibrated configurations (wave J-2) — the only role is `dit` (what is rounded is the linears
#: of the DiT block sequence; the other roles do not move by a single bit). The `weight` spelling
#: is the configuration name itself (no duplicated bookkeeping — the same convention as
#: {@link w4_recipe}).
CALIB_RECIPES: Mapping[str, Recipe] = MappingProxyType(
    {
        config.name: Recipe(config.name, (ex.TARGET_DIT,), op_types=(nn.Linear,), calib=config)
        for config in CALIB_CONFIGS
    }
)

#: The full table of configuration name → recipe.
RECIPES: Mapping[str, Recipe] = MappingProxyType(
    {**CONFIGS, **DIAGNOSTICS, **W4_CONFIGS, **CALIB_RECIPES}
)

#: STFT settings of the LSD (an analysis grid at 48 kHz — hop 512 ≈ 10.7 ms).
STFT_N_FFT = 2048
STFT_HOP = 512
#: The amplitude floor (relative to the reference's maximum amplitude = −100 dB). A lower bound
#: only to keep the log of silent regions from falling to −inf, and **the same floor** is applied
#: to both series (applied to only one side, the difference would depend on the floor).
STFT_FLOOR = 1e-5

CASE_SUFFIX = ".safetensors"
REPORT_FILE = "report.json"


# ---- metrics (duplicated from `measure_quant_sbv2.py` — the NOTE of the module docstring) ----


def snr_db(value: torch.Tensor, reference: torch.Tensor) -> float:
    """`10·log10(Σref² / Σ(value−ref)²)` (the main metric of audio-quality comparison between
    configurations)."""
    err = float((value - reference).double().square().sum())
    ref = float(reference.double().square().sum())
    if err == 0.0:
        return math.inf
    if ref == 0.0:
        return -math.inf
    return 10.0 * math.log10(ref / err)


def log_spectral_distance(value: torch.Tensor, reference: torch.Tensor) -> float:
    """Log-spectral distance `sqrt(mean((20log10|X| − 20log10|Y|)²))` [dB].

    Waveform SNR collapses with even a small phase shift (once the flow-matching trajectory
    changes, the residual reaches the order of the signal even for audio that sounds the same).
    LSD looks only at the magnitude spectrum, so it is **insensitive to phase shifts**, and is
    reported alongside as the metric of "how timbre and bandwidth break". Neither one is judged
    alone.
    """
    window = torch.hann_window(STFT_N_FFT)
    spectra = [
        torch.stft(
            wave_data, n_fft=STFT_N_FFT, hop_length=STFT_HOP, window=window, return_complex=True
        ).abs()
        for wave_data in (value, reference)
    ]
    floor = float(spectra[1].max()) * STFT_FLOOR
    a, b = (20.0 * torch.log10(spectrum.clamp_min(floor)) for spectrum in spectra)
    return float((a - b).square().mean().sqrt())


def rel_rms(value: torch.Tensor, reference: torch.Tensor) -> float:
    denom = float(reference.double().square().sum())
    if denom == 0.0:
        return 0.0
    return math.sqrt(float((value - reference).double().square().sum()) / denom)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_wav(path: Path, audio: torch.Tensor, sample_rate: int) -> None:
    """Write mono 16-bit PCM (the inverse of `decodeWav` — writing with ×32767 is Karume's
    convention)."""
    clipped = audio.detach().reshape(-1).to(torch.float32).clamp(-1.0, 1.0)
    pcm = torch.floor(clipped * 32767.0 + 0.5).to(torch.int16)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.numpy().tobytes())


# ---- assembling the configurations ------------------------------------------------


def role_modules(modules: Mapping[str, nn.Module], roles: Sequence[str]) -> dict[str, nn.Module]:
    """Role → the modules that role rounds (the form passed to `irodori.export.fake_quant`).

    MUST: the 3 norms are bundled with **the role that consumes them** (`speaker_norm` inside the
    speaker graph, `text_norm` inside the duration graph, `caption_norm` inside the caption-proj
    graph — where they live is ADR 0048 decision 1). Selecting every role gives **the same set**
    as the 9 that `irodori.export.export_series` / `irodori.pipeline_ref.emit` round, which
    guarantees as a set that the f16 configuration is the counterpart of the distribution
    series.
    """
    grouped: dict[str, dict[str, nn.Module]] = {
        ex.TARGET_BACKBONE: {ex.TARGET_BACKBONE: modules[ex.TARGET_BACKBONE]},
        ex.TARGET_TEXT_PROJ: {ex.TARGET_TEXT_PROJ: modules[ex.TARGET_TEXT_PROJ]},
        ex.TARGET_CAPTION_PROJ: {
            ex.TARGET_CAPTION_PROJ: modules[ex.TARGET_CAPTION_PROJ],
            "caption_norm": modules["caption_norm"],
        },
        ex.TARGET_SPEAKER: {
            ex.TARGET_SPEAKER: modules[ex.TARGET_SPEAKER],
            "speaker_norm": modules["speaker_norm"],
        },
        ex.TARGET_DURATION: {
            ex.TARGET_DURATION: modules[ex.TARGET_DURATION],
            "text_norm": modules["text_norm"],
        },
        ex.TARGET_DIT: {ex.TARGET_DIT: modules[ex.TARGET_DIT]},
    }
    picked: dict[str, nn.Module] = {}
    for role in roles:
        picked.update(grouped.get(role, {}))
    return picked


def apply_weight_quant(
    recipe: Recipe, scoped: Mapping[str, nn.Module], fit_stride: int = 1
) -> dict[str, str]:
    """Apply a configuration's weight rounding to the role modules and return role → summary.

    f16 / i8 are delegated to `irodori.export.fake_quant` as before (the same function as the
    distribution path). w4 calls the core's rounding per method and drops with `include` only the
    weights whose quantization axis is not divisible by g32 (no partial groups — ADR 0069
    decision 2).

    MUST: even for w4 the rounding target is **the tree of the role module** (= the same set as
    i8). Rounding only the subtree in the shipped graph would make "what was rounded" disagree
    with the i8 measurements and the comparison between methods would not hold (the inner copies
    not in the graph get rounded too, but they never appear in the forward, so they do not affect
    quality). Only the counts of the size estimate are taken from the shipped graph side
    ({@link graph_scans}).

    MUST: when a role has w4 candidates but all of them drop out on g32, fail loudly — running
    that role alone in f32 would measure "a configuration believed to be rounded".
    """
    if recipe.method is None:
        return dict(ex.fake_quant(recipe.weight, scoped).reports)
    reports: dict[str, str] = {}
    for name, module in sorted(scoped.items()):
        scan = scan_targets(module, recipe.op_types)
        if scan.counts.modules == 0:
            if scan.excluded:
                raise SystemExit(
                    f"{name}: all {len(scan.excluded)} w4 targets have a quantization axis"
                    f" not divisible by g32, leaving 0 ({scan.excluded[0].fqn} with axis length"
                    f" {scan.excluded[0].axis_length} and others)"
                )
            reports[name] = "kept as f32 storage (no w4 target types)"
            continue
        rounded = recipe.method.round_model(
            module, recipe.op_types, aligned_include(scan.excluded), fit_stride
        )
        if rounded.modules != scan.counts.modules:
            raise SystemExit(
                f"{name}: the rounded count {rounded.modules} differs from the"
                f" {scan.counts.modules} targets (op_types or include diverged from the scan — the"
                " targets counted and the targets rounded must be the same set)"
            )
        skipped = f" / excluded {len(scan.excluded)} not aligned to g32" if scan.excluded else ""
        sampled = f" / table fit on a 1/{fit_stride} subsample" if fit_stride > 1 else ""
        reports[name] = f"rounded to {recipe.method.name} — {rounded.describe()}{skipped}{sampled}"
    for name, report in reports.items():
        print(f"[fake-quant] {name}: {report}", flush=True)
    return reports


def role_graphs(graphs: ip.HostGraphs) -> dict[str, nn.Module]:
    """Role → its **shipped graph** (the 7 wrappers `irodori.export.export_series` emits).

    The DiT role returns its two wrappers (`dit-context` / `dit` — ADR 0114) bundled into one
    tree — the rounding role (the whole `TextToLatentRFDiT`) and the shipped graphs became one to
    two, so the size-estimate counts are taken over the union of the two (the shared `k_norm` is
    counted once thanks to the deduplication of `named_modules`).
    """
    return {
        ex.TARGET_BACKBONE: graphs.backbone,
        ex.TARGET_TEXT_PROJ: graphs.text_proj,
        ex.TARGET_CAPTION_PROJ: graphs.caption_proj,
        ex.TARGET_SPEAKER: graphs.speaker,
        ex.TARGET_DURATION: graphs.duration,
        ex.TARGET_DIT: nn.ModuleList([graphs.dit_context, graphs.dit]),
    }


def graph_scans(
    graphs: Mapping[str, nn.Module],
    roles: Sequence[str],
    op_types: tuple[type[nn.Module], ...],
) -> dict[str, RoleScan]:
    """Role → the scale of the w4 targets in the shipped graph (the input of the size estimate
    and the exclusion list).

    The wrappers are counted instead of the rounded tree because the DiT that
    `irodori.export.load_dit` builds also holds **a copy of the backbone that is not in the
    graph** (the NOTE of `export.TARGET_SCALE_SOURCES`). Estimating on the rounded set would
    inflate the projected MiB by weights that do not ship. The op kinds are the same ones the
    rounding receives (if they moved separately, "the targets rounded" and "the targets estimated"
    would silently split).
    """
    return {
        role: scan_targets(graph, op_types) for role, graph in graphs.items() if role in set(roles)
    }


class Loaded(NamedTuple):
    """The return value of `load_modules` (the plain set before any rounding is applied)."""

    source: Any
    modules: dict[str, nn.Module]
    config: Any
    text_config: Mapping[str, Any]
    model_config: Mapping[str, Any]


def load_modules(model_dir: Path, source_dir: Path) -> Loaded:
    """Build the same 9 modules as `irodori.pipeline_ref.emit` (no rounding applied yet)."""
    source = ex.IrodoriSource(source_dir)
    text_config, model_config = ex.read_configs(model_dir)
    state = load_file(str(model_dir / ex.CHECKPOINT_FILE))
    backbone = ex.load_backbone(source, state, text_config)
    hidden_size = int(backbone.hidden_size)
    config = source.model_config(model_config)
    modules = {
        ex.TARGET_BACKBONE: backbone,
        ex.TARGET_TEXT_PROJ: ex.load_projector(
            source,
            state,
            model_config,
            ex.TEXT_PROJ_PREFIX,
            hidden_size,
            int(model_config["text_dim"]),
        ),
        ex.TARGET_CAPTION_PROJ: ex.load_projector(
            source,
            state,
            model_config,
            ex.CAPTION_PROJ_PREFIX,
            hidden_size,
            int(model_config["caption_dim"]),
        ),
        ex.TARGET_SPEAKER: ex.load_speaker_encoder(source, state, config),
        ex.TARGET_DURATION: ex.load_duration_predictor(source, state, config),
        ex.TARGET_DIT: ex.load_dit(source, state, config, text_config),
        "speaker_norm": ex.load_rms_norm(
            source, state, ex.SPEAKER_NORM_PREFIX, int(config.speaker_dim), float(config.norm_eps)
        ),
        "text_norm": ex.load_rms_norm(
            source, state, ex.TEXT_NORM_PREFIX, int(config.text_dim), float(config.norm_eps)
        ),
        "caption_norm": ex.load_rms_norm(
            source,
            state,
            ex.CAPTION_NORM_PREFIX,
            int(config.caption_dim_resolved),
            float(config.norm_eps),
        ),
    }
    return Loaded(source, modules, config, text_config, model_config)


def build_graphs(
    modules: Mapping[str, nn.Module], config: Any, model_config: Mapping[str, Any]
) -> ip.HostGraphs:
    """The same 7 graph wrappers as `irodori.pipeline_ref.emit`.

    MUST: they work correctly only **after the patches are applied** (they pass the real-valued
    RoPE table).
    """
    speaker_max = ex.speaker_sym_max(model_config)
    dit_graphs = ex.dit_wrappers(modules[ex.TARGET_DIT], ex.dit_sym_max(config))
    return ip.HostGraphs(
        backbone=ex.BackboneGraph(modules[ex.TARGET_BACKBONE]),
        text_proj=ex.ProjectorGraph(modules[ex.TARGET_TEXT_PROJ]),
        caption_proj=ex.CaptionProjectorGraph(
            modules[ex.TARGET_CAPTION_PROJ], modules["caption_norm"]
        ),
        speaker=ex.SpeakerGraph(modules[ex.TARGET_SPEAKER], modules["speaker_norm"], speaker_max),
        duration=ex.DurationGraph(modules[ex.TARGET_DURATION], modules["text_norm"]),
        dit_context=dit_graphs.context,
        dit=dit_graphs.dit,
    )


class DecoderStage(NamedTuple):
    """The return value of {@link build_decoder}. `scan` is filled only when a w4 configuration
    rounds the codec."""

    graph: nn.Module
    sample_rate: int
    report: str | None
    scan: RoleScan | None


def build_decoder(
    source_dir: Path, model_dir: Path, recipe: Recipe, fit_stride: int = 1
) -> DecoderStage:
    """Return the codec's decode path (`DecoderGraph`), the sampling rate and the rounding
    summary.

    MUST (ordering): rounding comes **after** `fold_weight_norm` / `lift_snake_alphas` (the ordering
    MUST of `_fake_quant` in `irodori.dacvae.export` — this runs the same preprocessing in the same
    order before applying the rounding). The truncated `in_proj` does not appear on the decode
    path, so it is not built.

    w4 does not go through `dv._fake_quant` (which accepts only storage-dtype spellings); it
    applies the method's rounding to **the whole codec model**. The counts and the exclusion list
    are taken from the decode path's wrapper (the same reason as {@link graph_scans} — the
    encoder is not distributed).

    NOTE: the `DecoderGraph` tree keeps the unused layers of the watermark branch
    (`bypass_watermark` only replaces the forward and does not remove the modules), so the count
    is **6,266,080 elements (about 24 MiB, 8.8% of the codec) more** than the set decode actually
    goes through. `dv.main_path` would split it exactly, but that is "a copy used only by the
    checks", so it is not brought into the counts (0.8% over for the whole model).
    """
    source = dv.DacvaeSource(source_dir)
    model = dv.load_codec(source, model_dir)
    dv.bypass_watermark(model.decoder)
    dv.fold_weight_norm(model)
    dv.lift_snake_alphas(source, model)
    quantize = ROLE_CODEC in recipe.roles and recipe.weight is not None
    if not quantize:
        report = dv._fake_quant("f32", model).report
    elif recipe.method is None:
        report = dv._fake_quant(recipe.weight, model).report
    else:
        scan = scan_targets(model, recipe.op_types)
        if scan.counts.modules == 0:
            raise SystemExit(
                f"codec: all {len(scan.excluded)} w4 targets have a quantization axis not"
                " divisible by g32, leaving 0"
            )
        rounded = recipe.method.round_model(
            model, recipe.op_types, aligned_include(scan.excluded), fit_stride
        )
        sampled = f" / table fit on a 1/{fit_stride} subsample" if fit_stride > 1 else ""
        report = f"rounded to {recipe.method.name} — {rounded.describe()}{sampled}"
        print(f"[fake-quant] codec: {report}", flush=True)
    graph = dv.DecoderGraph(model.quantizer.out_proj, model.decoder)
    return DecoderStage(
        graph,
        int(model.sample_rate),
        report,
        scan_targets(graph, recipe.op_types) if quantize and recipe.method is not None else None,
    )


# ---- driving calibrated rounding (stage split + Catcher) ---------------------------


class _CalibStepsReached(Exception):  # noqa: N818 — a signal to stop, not an error
    """The signal that the needed number of steps has been captured (a sentinel that winds the
    reference loop down early)."""


class DitBlockStage(nn.Module):
    """A stage wrapper that wraps one DiT block in the form "takes the hidden state as a
    positional argument".

    The block body is built in **the same order** as the 4 loop lines of
    `irodori.export.DitGraph.forward` — the blocks of `TextToLatentRFDiT` have
    `DiffusionBlock.forward`, but `DitGraph` does not call it and expands into the equivalent
    attention implementation ({@link irodori.export.DitGraph._attention}), so using the block as is
    for the stage would make **the measured path a different one**. The attention borrows that
    staticmethod directly instead of adding another copy. In the distribution the condition-side
    K/V is built once per generation by `dit-context` (ADR 0114), but the stage calls the same
    single implementation ({@link irodori.export.DitContextGraph.block_context}) **inside the
    block** — the 72 condition-side projections are block weights too, and GPTQ needs to see their
    inputs (the normalized condition states). The values are bit-identical to the distribution's
    path (the same modules called on the same inputs).

    MUST: that the copy has not drifted from upstream is measured bit-exactly by
    {@link assert_stage_split_matches_graph} before rounding (dropping `nn.Dropout(p=0.0)` is the
    same exact identity as in `DitGraph`).

    The child module is named with **the block index** so that adding {@link DIT_BLOCK_PREFIX} to
    a stage's local FQN (`0.attention.wq.weight`) is all it takes to get back the in-model FQN.
    """

    def __init__(self, index: int, block: nn.Module) -> None:
        super().__init__()
        self.child = str(index)
        self.add_module(self.child, block)

    def forward(
        self,
        x: torch.Tensor,
        cond_embed: torch.Tensor,
        text: torch.Tensor,
        speaker: torch.Tensor,
        caption: torch.Tensor,
        mask: torch.Tensor,
        freqs: torch.Tensor,
    ) -> torch.Tensor:
        block = getattr(self, self.child)
        h, attention_gate = block.attention_adaln(x, cond_embed)
        context_k, context_v = ex.DitContextGraph.block_context(
            block.attention, text, speaker, caption
        )
        x = x + attention_gate * ex.DitGraph._attention(
            block.attention, h, context_k, context_v, mask, freqs
        )
        h, mlp_gate = block.mlp_adaln(x, cond_embed)
        return x + mlp_gate * block.mlp(h)


def dit_stages(graph: nn.Module) -> tuple[StageSpec, ...]:
    """Return the DiT blocks in execution order as `(in-model FQN prefix, stage)`.

    A stage holds **the whole real block module** (`DitGraph.source_blocks`) — `DitGraph`'s
    `blocks` is a holder without the condition-side projections (the range the distribution owns),
    so wrapping that would let the 72 condition-side weights escape calibration.
    """
    return tuple(
        (DIT_BLOCK_PREFIX, DitBlockStage(index, block))
        for index, block in enumerate(graph.source_blocks)
    )


def calib_targets(stages: Sequence[StageSpec]) -> tuple[dict[str, torch.Tensor], TargetCounts]:
    """Return the calibration targets inside the stages as weights keyed by FQN plus counts (**the
    scan** = the basis of the exact-coverage check).

    Target selection shares the core's `iter_quant_targets` (same as {@link scan_targets}).

    MUST: a weight not aligned to g32 **fails loudly instead of being excluded** — calibration
    drives whole stages, so dropping a single weight in the middle with `include` would break the
    check "scanned count = rounded count" (the w4 methods are independent per layer, so printing
    an exclusion list is fine there).
    """
    weights: dict[str, torch.Tensor] = {}
    modules = channels = elements = 0
    for prefix, stage in stages:
        for local, weight, axis in iter_quant_targets(stage, (nn.Linear,)):
            fqn = f"{prefix}.{local}"
            length = group_axis_length(weight, axis)
            if length % W4_GROUP_SIZE:
                raise SystemExit(
                    f"{fqn}: quantization axis {length} is not divisible by g{W4_GROUP_SIZE}"
                    " (calibration is driven per stage, so there is no escape hatch that drops a"
                    " single weight)"
                )
            weights[fqn] = weight
            modules += 1
            channels += int(weight.shape[axis])
            elements += int(weight.numel())
    if not modules:
        raise SystemExit(
            f"the calibration target '{CALIB_TARGET}' has not a single quantizable weight"
        )
    return weights, TargetCounts(modules, channels, elements)


def capture_case_batches(
    context_graph: nn.Module,
    graph: nn.Module,
    run_reference: Callable[[], object],
    limit: int,
) -> tuple[list[StageBatch], tuple[torch.Tensor, ...]]:
    """Capture **one batch per step** from one case's reference loop (f32, CFG included).

    Returns `(batches, the DitGraph arguments of the first forward)`. The latter is used by
    {@link assert_stage_split_matches_graph} to measure the equivalence of the stage split.

    One step is one cond forward + n uncond forwards for CFG, and all of them receive **the same
    `x_t`** (`irodori.pipeline_ref._euler` builds only one `x_t` per step), so step boundaries are
    split by the **identity** of `x_t`, and only the one forward right after it changes = the cond
    side is taken.

    What is captured is the set built **before** the block loop — the normalized text and caption
    and the speaker state from `context_graph` (`dit-context`, once per generation — ADR 0114), and
    `cond_embed` / the concatenated mask / the RoPE table and the hidden state into the first block
    from the forward of `graph` (`dit`). They are not rebuilt here because the moment they are
    rebuilt they can silently diverge from the wrappers' spelling (the same discipline as EG's
    Catcher).

    Once `limit` is captured, the sentinel winds the reference loop down — `run_case` runs one
    more loop without CFG **after** the loop with CFG, so without winding down "another trajectory
    of the same step" would be mixed in.
    """
    batches: list[StageBatch] = []
    probe: list[tuple[torch.Tensor, ...]] = []
    state: dict[str, Any] = {"x_t": None, "take": False}
    pending: dict[str, torch.Tensor] = {}
    conditions: dict[str, torch.Tensor] = {}

    def on_context(_module: nn.Module, args: tuple[Any, ...]) -> None:
        conditions["speaker"] = args[1].detach()

    def on_text(_module: nn.Module, _args: tuple[Any, ...], output: torch.Tensor) -> None:
        conditions["text"] = output.detach()

    def on_caption(_module: nn.Module, _args: tuple[Any, ...], output: torch.Tensor) -> None:
        conditions["caption"] = output.detach()

    def on_graph(_module: nn.Module, args: tuple[Any, ...]) -> None:
        if args[0] is state["x_t"]:
            state["take"] = False
            return
        if len(batches) >= limit:
            raise _CalibStepsReached
        missing = sorted({"text", "speaker", "caption"} - set(conditions))
        if missing:
            raise SystemExit(
                f"dit ran before dit-context (conditions {missing} were not captured)"
                " — the reference loop's spelling disagrees with what this script assumes"
            )
        state["x_t"] = args[0]
        state["take"] = True
        pending["mask"] = args[2]
        if not probe:
            probe.append(tuple(args))

    def on_block(_module: nn.Module, args: tuple[Any, ...]) -> None:
        if not state["take"]:
            return
        hidden = args[0].detach()
        batches.append(
            (
                (hidden,),
                {
                    "cond_embed": args[1].detach(),
                    "text": conditions["text"],
                    "speaker": conditions["speaker"],
                    "caption": conditions["caption"],
                    "mask": pending["mask"],
                    "freqs": graph.rope_table[: int(hidden.shape[1])],
                },
            )
        )
        state["take"] = False

    handles = [
        context_graph.register_forward_pre_hook(on_context),
        context_graph.text_norm.register_forward_hook(on_text),
        context_graph.caption_norm.register_forward_hook(on_caption),
        graph.register_forward_pre_hook(on_graph),
        graph.blocks[0].attention_adaln.register_forward_pre_hook(on_block),
    ]
    try:
        run_reference()
    except _CalibStepsReached:
        pass
    finally:
        for handle in handles:
            handle.remove()
    if not batches or not probe:
        raise SystemExit(
            "could not capture calibration input for even 1 step (the DitGraph spelling disagrees"
            " with what this script assumes)"
        )
    return batches, probe[0]


def assert_stage_split_matches_graph(
    graph: nn.Module,
    probe: tuple[torch.Tensor, ...],
    batch: StageBatch,
    stages: Sequence[StageSpec],
) -> None:
    """The stage split + the tail (`out_norm` → `out_proj`) is **bit-identical** to one forward
    of `DitGraph`.

    `probe` holds that forward's `DitGraph` arguments, and `batch` the input to the first stage
    captured in **the same forward**.

    MUST: measured before rounding. {@link DitBlockStage} is a copy of the block loop, so it
    silently drifts when upstream's `DitGraph.forward` changes — rounding on the drifted side
    would measure "GPTQ along another path", and the numbers would still come out normally (it
    cannot be read off the table).
    """
    args, kwargs = batch
    with torch.no_grad():
        reference = graph(*probe)
        hidden = args[0]
        for _prefix, stage in stages:
            hidden = stage(hidden, **kwargs)
        rebuilt = graph.out_proj(graph.out_norm(hidden))
    if not torch.equal(rebuilt, reference):
        raise SystemExit(
            "the stage-split reconstruction is not bit-identical to the DitGraph output"
            f" (max abs difference {float((rebuilt - reference).abs().max()):.4e})"
            " — DitBlockStage's copy has drifted from irodori.export.DitGraph.forward"
        )


@dataclass(frozen=True)
class CalibRig:
    """The scaffolding the calibrated configurations share (the stage list, the scan, the input to
    the first stage).

    It is not rebuilt per configuration — the calibration input is taken only once, **before any
    rounding is applied** (the pristine discipline that `restore` stands for is guaranteed by
    "one run = one configuration").
    """

    stages: tuple[StageSpec, ...]
    scan: Mapping[str, torch.Tensor]
    counts: TargetCounts
    batches: tuple[StageBatch, ...]
    #: Case name → the number of steps captured (keeps a reduced run next to the numbers).
    steps: Mapping[str, int]
    #: Total tokens of the hidden states stacked for calibration (= Σ S over steps).
    tokens: int


def build_calib_rig(
    graphs: ip.HostGraphs, run_reference: Callable[[ip.PipelineCase], object], limit: int
) -> CalibRig:
    """Build the calibration scaffolding (stage split → scan → reference-loop capture →
    equivalence check)."""
    graph = graphs.dit
    stages = dit_stages(graph)
    scan, counts = calib_targets(stages)
    batches: list[StageBatch] = []
    steps: dict[str, int] = {}
    for case in ip.PIPELINE_CASES:
        caught, probe = capture_case_batches(
            graphs.dit_context, graph, lambda case=case: run_reference(case), limit
        )
        if not batches:
            # `probe` and `caught[0]` are **the same forward** (the cond side of the first step).
            assert_stage_split_matches_graph(graph, probe, caught[0], stages)
        steps[case.name] = len(caught)
        batches += caught
        print(f"[calib] {case.name}: captured {len(caught)} calibration steps", flush=True)
    tokens = sum(int(args[0].shape[1]) for args, _kwargs in batches)
    print(
        f"[calib] {len(stages)} stages / {counts.modules} target linears"
        f" / {len(batches)} batches / {tokens:,} hidden tokens in total"
        f" (the target is {CALIB_TARGET} — the outside of the DiT blocks and other roles are not"
        " included)",
        flush=True,
    )
    return CalibRig(
        stages=stages,
        scan=scan,
        counts=counts,
        batches=tuple(batches),
        steps=steps,
        tokens=tokens,
    )


def apply_calib(config: CalibConfig, rig: CalibRig) -> tuple[dict[str, str], CalibReport]:
    """Apply one calibrated rounding to the DiT in place (the stage-by-stage driving is on the
    core side)."""
    report = calibrate_stages(rig.stages, rig.batches, method=config.method, spec=config.grid)
    assert_calib_covers_scan(report, rig.scan, config.name)
    summary = f"rounded to {config.name} with calibration — {report.describe()}"
    print(f"[fake-quant] {ex.TARGET_DIT}: {summary}", flush=True)
    return {ex.TARGET_DIT: summary}, report


def assert_calib_covers_scan(
    report: CalibReport, scan: Mapping[str, torch.Tensor], name: str
) -> None:
    """Check that the layers calibration rounded match the stage scan **exactly**.

    MUST: fail loudly. Even if a change in the stage spelling or the target types leaves part of
    a block out of calibration, the table keeps its rows, and the missed rounding makes quality
    come out **better** (passing through cannot be read off the numbers).
    """
    rounded = {layer.fqn for layer in report.layers}
    missing = sorted(set(scan) - rounded)
    extra = sorted(rounded - set(scan))
    if missing or extra or report.modules != len(scan):
        raise SystemExit(
            f"[{name}] calibration rounded {report.modules}, which does not match the"
            f" {len(scan)} in the scan (missed {missing[:3]} / not in the scan {extra[:3]})"
        )


# ---- running one configuration ----------------------------------------------------


def golden_z(golden_dir: Path, case: str) -> torch.Tensor:
    """The final z of the full-loop golden (the only comparison target that keeps the `f32`
    configuration from being vacuous)."""
    path = golden_dir / f"{ip.CASE_PREFIX}{case}{ip.CASE_SUFFIX}"
    if not path.is_file():
        raise SystemExit(
            f"no full-loop golden: {path}"
            " (built with `uv run --with 'transformers==5.14.1' python -m irodori.pipeline_ref`)"
        )
    return load_file(str(path))["z"].to(torch.float32)


def base_payload(out_dir: Path, case: str) -> dict[str, torch.Tensor]:
    """The z and waveform of the baseline configuration (`f32`)."""
    path = out_dir / f"{BASE_CONFIG}.{case}{CASE_SUFFIX}"
    if not path.is_file():
        raise SystemExit(
            f"no outputs of the baseline configuration: {path}"
            f" (run `--config {BASE_CONFIG}` first — the comparison baseline is f32)"
        )
    return load_file(str(path))


def weight_only_path(out_dir: Path, case: str) -> Path:
    """The outputs of the activation shim's comparison partner ({@link WEIGHT_ONLY_BASE} — same
    weights, plain activations).

    The existence check is separate so that it can run **before** the full loop (failing with
    "no comparison partner" after running it through would throw away tens of minutes of
    compute).
    """
    path = out_dir / f"{WEIGHT_ONLY_BASE}.{case}{CASE_SUFFIX}"
    if not path.is_file():
        raise SystemExit(
            f"no comparison partner for the activation shim: {path}"
            f" (run `--config {WEIGHT_ONLY_BASE}` first — pass-through detection is the difference"
            " from the weight-only configuration)"
        )
    return path


def weight_only_payload(out_dir: Path, case: str) -> dict[str, torch.Tensor]:
    """The z and waveform of {@link weight_only_path}."""
    return load_file(str(weight_only_path(out_dir, case)))


class LatentResult(NamedTuple):
    """The return value of {@link latent_stage}. `act_quant_linears` is the number of
    `nn.Linear` the activation shim is attached to."""

    metas: dict[str, dict[str, Any]]
    latents: dict[str, torch.Tensor]
    reports: dict[str, str]
    act_quant_linears: int
    #: Role → the w4 target scale on the shipped graph side (w4 configurations only; empty
    #: otherwise).
    scans: dict[str, RoleScan]
    #: The scaffolding and rounding counts of a calibrated configuration (calibrated
    #: configurations only; `None` otherwise).
    calib: tuple[CalibRig, CalibReport] | None = None


def attach_dit_act_quant(
    recipe: Recipe, modules: Mapping[str, nn.Module]
) -> tuple[list[object], int]:
    """In the w8a8 configuration, apply a per-token i8 fake-quant to the inputs of the DiT's
    `nn.Linear`.

    MUST: **0 fails loudly** — running without the shim attached would put the same numbers as
    `i8-all` into the report under the name `w8a8`, with quality on the "better" side
    (activation quantization adds error, so a pass-through is always a lie leaning green). It is
    limited to the DiT to match the distribution's preset (`session` reaches only the `dit`
    Session).
    """
    if not recipe.act_quant:
        return [], 0
    handles, attached = act_quant.attach_act_quant(modules[ex.TARGET_DIT])
    if attached == 0:
        act_quant.detach_act_quant(handles)
        raise SystemExit(
            "the activation-quantization shim is attached to 0 nn.Linear"
            f" (the Linears of `{ex.TARGET_DIT}` do not meet the eligibility condition"
            f" k % {act_quant.PACK_ALIGN} == 0, or the module structure changed)"
        )
    return handles, attached


def latent_stage(
    name: str,
    recipe: Recipe,
    args: argparse.Namespace,
    base_frames: Mapping[str, int] | None,
) -> LatentResult:
    """Text → latent (apply the configuration's rounding and run the full loop)."""
    from tokenizers import Tokenizer

    source, modules, config, text_config, model_config = load_modules(
        args.model_dir, args.source_dir
    )
    # MUST: rounding happens right after load and before the golden is taken (the ordering MUST of
    # `irodori.export.fake_quant`). The targets are empty only for `i8-codec-only` (it includes no
    # latent-side role), and then the rounding is not called at all — `fake_quant`'s "0 fails
    # loudly" is the right discipline, so it is satisfied here by **not calling** rather than by
    # making an exception.
    #
    # Only the calibrated configurations have a different order (rounding after the graphs) — the
    # calibration input is taken from "the reference loop run in plain f32", so rounding is only
    # possible after the graphs are built and one pristine full loop has run on them. The MUST
    # **before rounding** is kept (up to this block no weight has moved by a single bit).
    scoped = role_modules(modules, recipe.roles)
    reports = (
        apply_weight_quant(recipe, scoped, args.kmeans_fit_stride)
        if recipe.weight is not None and recipe.calib is None and scoped
        else {}
    )
    patch.apply_patches()
    # The activation shim comes **after** the weight rounding (weights are fixed before execution,
    # activations at execution time — the two are independent, but ordering the report as
    # "weights → activations" lets a pass-through be isolated from a single output).
    handles, act_quant_linears = attach_dit_act_quant(recipe, modules)
    if act_quant_linears:
        print(
            f"[{name}] activation-quantization shim: {act_quant_linears} nn.Linear (dit)",
            flush=True,
        )
    graphs = build_graphs(modules, config, model_config)
    # The size-estimate counts are taken from **the shipped graphs** ({@link graph_scans}) — the
    # wrappers only hold the rounded modules, so counting here does not touch the values.
    scans = (
        graph_scans(role_graphs(graphs), recipe.roles, recipe.op_types)
        if recipe.method is not None
        else {}
    )
    speaker_max = ex.speaker_sym_max(model_config)
    caps = {
        "text": int(model_config["max_text_len"]),
        "speaker": speaker_max + 1,
        "caption": int(model_config["max_caption_len"]),
    }
    tokenizer = Tokenizer.from_file(str(args.model_dir / ex.TOKENIZER_FILE))

    # The graph set is **bound through default arguments** because the `del` below removes the
    # locals of the same name (a plain closure would leave the shape of a NameError when called
    # after they are gone).
    def run_case(
        case: ip.PipelineCase,
        graphs: ip.HostGraphs = graphs,
        source: Any = source,
        modules: Mapping[str, nn.Module] = modules,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        return ip.run_case(
            case,
            graphs,
            source,
            modules[ex.TARGET_DURATION],
            tokenizer,
            text_config,
            model_config,
            config,
            caps,
            frames_override=None if base_frames is None else base_frames[case.name],
        )

    calib: tuple[CalibRig, CalibReport] | None = None
    if recipe.calib is not None:
        rig = build_calib_rig(graphs, run_case, args.calib_steps)
        reports, calib_report = apply_calib(recipe.calib, rig)
        scans = {ex.TARGET_DIT: RoleScan(rig.counts, ())}
        calib = (rig, calib_report)

    metas: dict[str, dict[str, Any]] = {}
    latents: dict[str, torch.Tensor] = {}
    for case in ip.PIPELINE_CASES:
        started = time.perf_counter()
        tensors, meta = run_case(case)
        meta["elapsed"] = round(time.perf_counter() - started, 1)
        metas[case.name] = meta
        latents[case.name] = tensors["z"].detach().clone()
        print(
            f"[{name}/{case.name}] S={meta['S']} predicted={meta.get('predictedS', meta['S'])}"
            f" forwards={meta['forwards']} {meta['elapsed']:.0f}s",
            flush=True,
        )
    act_quant.detach_act_quant(handles)
    del graphs, modules, source
    gc.collect()
    return LatentResult(metas, latents, reports, act_quant_linears, scans, calib)


def w4_payload(recipe: Recipe, scans: Mapping[str, RoleScan], fit_stride: int) -> dict[str, Any]:
    """The target scale, exclusion list and size estimate of a w4 configuration (in the form that
    goes into the JSON as is).

    `tables` is the number of artifacts that carry a table = the number of roles with targets
    (`kmeans:shared` has one table per role — {@link W4_KMEANS_SHARED}). It has no effect on the
    formulas of the other 3 methods. **For calibrated configurations the table is scoped per
    layer**, so the number of target layers is passed instead ({@link _calib_codebook_bits}).
    """
    spec: W4Method | CalibConfig | None = recipe.method or recipe.calib
    if spec is None:
        raise SystemExit(
            "asked for an estimate of a recipe that is not a w4 configuration"
            f" (weight={recipe.weight})"
        )
    counts = {role: scan.counts for role, scan in scans.items()}
    total = total_counts(list(counts.values()))
    tables = (
        total.modules
        if recipe.calib is not None
        else sum(1 for item in counts.values() if item.modules)
    )
    size = project_size(spec, total, tables)
    return {
        "method": spec.name,
        "groupSize": W4_GROUP_SIZE,
        "opTypes": [cls.__name__ for cls in recipe.op_types],
        # MUST: a subsampled table and a full table can differ, so the fact of using one goes next
        # to the numbers (the MUST of `karume.quant_methods.fake_quant_kmeans`).
        "fitStride": fit_stride,
        "roles": {
            role: {"modules": item.modules, "channels": item.channels, "elements": item.elements}
            for role, item in sorted(counts.items())
        },
        "excluded": [
            {
                "role": role,
                "fqn": item.fqn,
                "axisLength": item.axis_length,
                "elements": item.elements,
            }
            for role, scan in sorted(scans.items())
            for item in scan.excluded
        ],
        "size": {
            "formula": size.formula,
            "tables": size.tables,
            "modules": size.counts.modules,
            "channels": size.counts.channels,
            "elements": size.counts.elements,
            "bits": size.bits,
            "bitsPerWeight": size.bits_per_weight,
            "projectedMiB": size.projected_mib,
            "f32MiB": size.f32_mib,
        },
    }


def run_config(name: str, args: argparse.Namespace) -> dict[str, Any]:
    """Run one configuration, write its per-config JSON and return it."""
    started = time.perf_counter()
    recipe = RECIPES[name]
    check_fit_stride(recipe, args.kmeans_fit_stride)
    base_frames = None
    goldens = None
    if name == BASE_CONFIG:
        # MUST: the comparison target is checked **before running** (failing with "no golden"
        # after running the full loop through would throw away tens of minutes of compute).
        goldens = {case.name: golden_z(args.golden_dir, case.name) for case in ip.PIPELINE_CASES}
    else:
        base_json = args.out / f"{BASE_CONFIG}.json"
        if not base_json.is_file():
            raise SystemExit(
                f"no outputs of the baseline configuration: {base_json}"
                f" (run `--config {BASE_CONFIG}` first — the comparison baseline is f32)"
            )
        base_meta = json.loads(base_json.read_text(encoding="utf-8"))
        base_frames = {case: entry["S"] for case, entry in base_meta["cases"].items()}
    if recipe.act_quant:
        # MUST: this too is checked **before running** (pass-through detection of the activation
        # shim can only come from the difference with the weight-only configuration, so without it
        # running this configuration gives no answer).
        for case in ip.PIPELINE_CASES:
            weight_only_path(args.out, case.name)

    metas, latents, reports, act_quant_linears, scans, calib = latent_stage(
        name, recipe, args, base_frames
    )

    stage = build_decoder(
        args.codec_source_dir, args.codec_model_dir, recipe, args.kmeans_fit_stride
    )
    decoder, sample_rate, codec_report = stage.graph, stage.sample_rate, stage.report
    if stage.scan is not None:
        scans[ROLE_CODEC] = stage.scan
    audios: dict[str, torch.Tensor] = {}
    for case, latent in latents.items():
        with torch.no_grad():
            audios[case] = decoder(latent).reshape(-1).detach().clone()
        print(f"[{name}/{case}] decoded {audios[case].shape[0]} samples", flush=True)
    del decoder, stage
    gc.collect()

    args.out.mkdir(parents=True, exist_ok=True)
    entries: dict[str, dict[str, Any]] = {}
    for case, meta in metas.items():
        audio, latent = audios[case], latents[case]
        path = args.out / f"{name}.{case}.wav"
        write_wav(path, audio, sample_rate)
        save_file(
            {"z": latent.contiguous(), "audio": audio.contiguous()},
            str(args.out / f"{name}.{case}{CASE_SUFFIX}"),
        )
        entry: dict[str, Any] = {
            "S": meta["S"],
            "predictedS": meta.get("predictedS", meta["S"]),
            "forwards": meta["forwards"],
            "predictedFrames": meta["duration"]["predictedFrames"],
            "predictedSeconds": round(meta.get("predictedS", meta["S"]) / ex.CODEC_FRAME_RATE, 3),
            "audioSeconds": round(int(audio.shape[0]) / sample_rate, 3),
            "samples": int(audio.shape[0]),
            "zAbsMax": meta["zAbsMax"],
            "cfgEffectMaxAbs": meta["cfgEffectMaxAbs"],
            "elapsed": meta["elapsed"],
            "wav": str(path),
            "wavSha256": sha256(path.read_bytes()),
        }
        if name != BASE_CONFIG:
            base = base_payload(args.out, case)
            base_z, base_audio = base["z"].to(torch.float32), base["audio"].to(torch.float32)
            if latent.shape != base_z.shape or audio.shape != base_audio.shape:
                raise SystemExit(
                    f"{case}: the shape differs from the baseline (z {tuple(latent.shape)} vs"
                    f" {tuple(base_z.shape)} / waveform {tuple(audio.shape)} vs"
                    f" {tuple(base_audio.shape)}) — if the baseline was rebaked, rerun every"
                    " configuration"
                )
            entry["vsBase"] = {
                "zBitEqual": bool(torch.equal(latent, base_z)),
                "zRelRms": rel_rms(latent, base_z),
                "zMaxAbsDiff": float((latent - base_z).abs().max()),
                "wavBitEqual": bool(torch.equal(audio, base_audio)),
                "wavSnrDb": snr_db(audio, base_audio),
                "wavLsdDb": log_spectral_distance(audio, base_audio),
                "wavRelRms": rel_rms(audio, base_audio),
                "wavMaxAbsDiff": float((audio - base_audio).abs().max()),
            }
        if recipe.act_quant:
            # Observation for pass-through detection (the verdict lives in {@link run_gates}) — it
            # looks at **the difference from the configuration with the same weights**, so 0 here
            # means not a single activation shim had an effect.
            weight_only = weight_only_payload(args.out, case)
            other_z = weight_only["z"].to(torch.float32)
            other_audio = weight_only["audio"].to(torch.float32)
            if latent.shape != other_z.shape or audio.shape != other_audio.shape:
                raise SystemExit(
                    f"{case}: the shape differs from {WEIGHT_ONLY_BASE} (z {tuple(latent.shape)}"
                    f" vs {tuple(other_z.shape)} / waveform {tuple(audio.shape)} vs"
                    f" {tuple(other_audio.shape)}) — fixing the time grid has no effect"
                )
            entry["vsWeightOnly"] = {
                "config": WEIGHT_ONLY_BASE,
                "zBitEqual": bool(torch.equal(latent, other_z)),
                "zRelRms": rel_rms(latent, other_z),
                "zMaxAbsDiff": float((latent - other_z).abs().max()),
                "wavSnrDb": snr_db(audio, other_audio),
                "wavLsdDb": log_spectral_distance(audio, other_audio),
            }
        entries[case] = entry

    payload = {
        "config": name,
        "weight": recipe.weight,
        "roles": list(recipe.roles),
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "torch": torch.__version__,
        "sampleRate": sample_rate,
        "frameRate": ex.CODEC_FRAME_RATE,
        "fakeQuant": reports,
        "codecFakeQuant": codec_report,
        "actQuantLinears": act_quant_linears,
        "cases": entries,
        "gates": run_gates(name, recipe, latents, entries, goldens),
        "elapsed": round(time.perf_counter() - started, 1),
    }
    if recipe.method is not None or recipe.calib is not None:
        payload["w4"] = w4_payload(recipe, scans, args.kmeans_fit_stride)
    if calib is not None:
        rig, calib_report = calib
        payload["calib"] = {
            "method": f"{calib_report.method}/{calib_report.grid}",
            "target": CALIB_TARGET,
            "stages": calib_report.stages,
            "roundedModules": calib_report.modules,
            "scanModules": rig.counts.modules,
            "steps": dict(rig.steps),
            "batches": len(rig.batches),
            "tokens": rig.tokens,
            "quantReport": calib_report.describe(),
        }
    (args.out / f"{name}.json").write_text(
        json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return payload


# ---- verification gates ----------------------------------------------------------


def run_gates(
    name: str,
    recipe: Recipe,
    latents: Mapping[str, torch.Tensor],
    entries: Mapping[str, Mapping[str, Any]],
    goldens: Mapping[str, torch.Tensor] | None,
) -> dict[str, Any]:
    """Detectors for vacuous checks and scope leaks (the reds are raised together at the end).

    ① the `f32` z is **byte-identical** to the full-loop golden (proof that this script's path is
       the same computation as the real one — if it fails, every measurement loses its meaning)
    ② each configuration's z / waveform **changes only where it should** (the orthogonality
       checks of the module docstring). Checking only "changes" catches only a rounding that
       passes through, and checking only "does not change" catches only a scope leak — both come
       from the expectation table
    ③ a configuration with the activation shim **has a z different from the weight-only
       configuration too** (pass-through detection). The difference from the baseline alone can
       be explained by the weight rounding, and would be green even with not a single shim in
       effect
    """
    gates: dict[str, Any] = {}
    failures: list[str] = []
    if goldens is not None:
        matched = {
            case: bool(torch.equal(latent, goldens[case])) for case, latent in latents.items()
        }
        gates["z_bit_exact_vs_golden"] = {"cases": matched}
        failures += [
            f"{case}: the f32 z is not byte-identical to the full-loop golden (the check that the"
            " path is not vacuous)"
            for case, ok in matched.items()
            if not ok
        ]
    else:
        observed = {
            case: {
                "zChanged": not entry["vsBase"]["zBitEqual"],
                "wavChanged": not entry["vsBase"]["wavBitEqual"],
            }
            for case, entry in entries.items()
        }
        expectations: dict[str, dict[str, bool]] = {}
        for case in observed:
            # Expectations are split **by case condition** — in cases without a reference the
            # speaker path does not run ({@link REFERENCE_ONLY_ROLES}), so those roles are taken
            # out of the expectation.
            effective = set(recipe.roles) & LATENT_ROLES
            if not CASE_HAS_REFERENCE[case]:
                effective -= REFERENCE_ONLY_ROLES
            expect_z = bool(effective)
            expectations[case] = {
                "zChanged": expect_z,
                "wavChanged": expect_z or ROLE_CODEC in recipe.roles,
            }
        gates["orthogonality"] = {"expected": expectations, "observed": observed}
        for case, seen in observed.items():
            expect_z = expectations[case]["zChanged"]
            expect_wav = expectations[case]["wavChanged"]
            if seen["zChanged"] != expect_z:
                failures.append(
                    f"{case}: z went against the expectation that it"
                    f" {'changes' if expect_z else 'does not change'}"
                    f" (roles {list(recipe.roles)} — a rounding passing through or a scope leak)"
                )
            if seen["wavChanged"] != expect_wav:
                failures.append(
                    f"{case}: the waveform went against the expectation that it"
                    f" {'changes' if expect_wav else 'does not change'}"
                    f" (roles {list(recipe.roles)})"
                )
        if recipe.act_quant:
            moved = {
                case: not entry["vsWeightOnly"]["zBitEqual"] for case, entry in entries.items()
            }
            gates["act_quant_changes_the_latent"] = {"versus": WEIGHT_ONLY_BASE, "cases": moved}
            failures += [
                f"{case}: z is bit-identical to {WEIGHT_ONLY_BASE} (the activation shim passes"
                " through — the same numbers as the weight-only configuration are measured under"
                " the name w8a8)"
                for case, ok in moved.items()
                if not ok
            ]
    if recipe.weight is not None:
        gates["quantized_roles"] = list(recipe.roles)
    gates["failures"] = failures
    return gates


# ---- report ----------------------------------------------------------------------


def collect(out_dir: Path) -> dict[str, dict[str, Any]]:
    """Collect the per-config JSON left in `--out`, in configuration order.

    MUST: the orthogonality checks are **re-evaluated here** (the saved JSON's `gates` are not
    replayed) — the code is the source of truth for the checks' expectations, and when a mistake
    on the expectation side is fixed after measuring, the saved verdicts would come back into the
    report with the old expectations (a real case: the case condition of speaker — no-ref is
    correctly unchanged because of the zero short-circuit). Only the golden bit identity of the
    baseline `f32` is a tensor comparison (possible only at measurement time), so its saved value
    is used. The observations (the bit equality of `vsBase` / `vsWeightOnly`) are facts of the
    measurement and, unlike verdicts, may be saved.
    """
    found: dict[str, dict[str, Any]] = {}
    for name in RECIPES:
        path = out_dir / f"{name}.json"
        if path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8"))
            if name != BASE_CONFIG:
                payload["gates"] = run_gates(name, RECIPES[name], {}, payload["cases"], None)
            found[name] = payload
    return found


def drift_table(collected: Mapping[str, Mapping[str, Any]]) -> str:
    """**The top table** — did S move (the ruling on the mixed table rests on this one table)."""
    header = ["config", "case", "predicted S", "base S", "S drift", "forwards", "utterance (s)"]
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    base = collected.get(BASE_CONFIG, {}).get("cases", {})
    for name, payload in collected.items():
        for case, entry in payload["cases"].items():
            reference = base.get(case, {}).get("predictedS")
            drift = "—" if reference is None else f"{entry['predictedS'] - reference:+d}"
            lines.append(
                "| "
                + " | ".join(
                    [
                        name,
                        case,
                        str(entry["predictedS"]),
                        "—" if reference is None else str(reference),
                        "match" if drift == "+0" else drift,
                        str(entry["forwards"]),
                        f"{entry['predictedSeconds']:.2f}",
                    ]
                )
                + " |"
            )
    return "\n".join(lines)


def quality_table(collected: Mapping[str, Mapping[str, Any]]) -> str:
    """Quality (a comparison **on the same time grid as the baseline**)."""
    header = ["config", "case", "latent relRMS", "latent max|Δ|", "SNR (dB)", "LSD (dB)", "s"]
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    for name, payload in collected.items():
        for case, entry in payload["cases"].items():
            versus = entry.get("vsBase")
            if versus is None:
                row = ["0 (base)", "0", "inf (base)", "0 (base)"]
            else:
                snr = versus["wavSnrDb"]
                row = [
                    f"{versus['zRelRms']:.4e}",
                    f"{versus['zMaxAbsDiff']:.4e}",
                    "inf" if math.isinf(snr) else f"{snr:.2f}",
                    f"{versus['wavLsdDb']:.2f}",
                ]
            lines.append("| " + " | ".join([name, case, *row, f"{entry['elapsed']:.0f}"]) + " |")
    return "\n".join(lines)


def size_table(collected: Mapping[str, Mapping[str, Any]]) -> str:
    """Size estimates of the w4 configurations (**projections by formula**, not measurements).

    Listing quality alone hides the cost of the tables (`kmeans:shared` needs a 16×f32 table per
    artifact), so the formulas go into the same report.
    """
    header = [
        "config",
        "method",
        "count",
        "elements",
        "formula",
        "bpw",
        "projected MiB",
        "f32 MiB",
        "ratio",
    ]
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    for name, payload in collected.items():
        size = payload.get("w4", {}).get("size")
        if size is None:
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    name,
                    payload["w4"]["method"],
                    str(size["modules"]),
                    f"{size['elements']:,}",
                    size["formula"],
                    f"{size['bitsPerWeight']:.3f}",
                    f"{size['projectedMiB']:.1f}",
                    f"{size['f32MiB']:.1f}",
                    f"{size['projectedMiB'] / size['f32MiB']:.3f}",
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def excluded_lines(collected: Mapping[str, Mapping[str, Any]]) -> list[str]:
    """The list of weights excluded from the targets as not aligned to g32 (per configuration;
    **always written to the output**).

    Excluding silently would leave only the non-divisible layers in f32 behind a table that reads
    as "every role was rounded".
    """
    lines: list[str] = []
    for name, payload in collected.items():
        excluded = payload.get("w4", {}).get("excluded")
        if excluded is None:
            continue
        if not excluded:
            lines.append(f"- {name}: no exclusions for g32 misalignment")
            continue
        lines.append(f"- {name}: excluded {len(excluded)} not aligned to g32")
        lines += [
            f"    - {item['role']} / {item['fqn']} axis length {item['axisLength']}"
            f" ({item['elements']:,} elements)"
            for item in excluded
        ]
    return lines


def build_report(collected: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "script": "tools/export-recipes/irodori/measure_quant.py",
        "torch": torch.__version__,
        "base": BASE_CONFIG,
        "method": {
            "time_grid": "every configuration but the baseline fixes the f32 S from outside (lines"
            " up the waveform lengths so SNR is defined). The configuration's own prediction is"
            " predictedS",
            "roles": list(ROLES),
            "latent_roles": sorted(LATENT_ROLES),
            "metrics": "SNR is the waveform residual (sensitive to phase shifts) / LSD is the"
            f" log-magnitude spectral distance (n_fft={STFT_N_FFT} hop={STFT_HOP}, floor −100 dB,"
            " insensitive to phase shifts). Read both",
            "trim": "no tail trim or cut by seconds (to keep the lengths equal across"
            " configurations)",
            "act_quant": "w8a8 applies a per-token i8 fake-quant to the inputs of the DiT's"
            " nn.Linear (karume.act_quant — the mirror of the runtime's quantize-rows.ts)."
            f" Pass-through detection is z bit equality with {WEIGHT_ONLY_BASE}",
            "w4": f"methods are compared with g={W4_GROUP_SIZE} fixed (item 3 of ADR 0069 addendum"
            " 5). The target ops are per configuration (the source of truth is w4.opTypes in each"
            " config JSON — the every-role form = the same 5 kinds as i8 / i4-linear = Linear"
            " only). The i4 execution path is linear-only (decision 5), so the every-role form is"
            " not shippable and measures the upper bound of quality; i4-linear is today's"
            " distribution-compatible form. The size columns are **projections by formula**, and"
            " the counts cover only the weights in the shipped graphs",
            "calib": f"the calibrated configurations ({' / '.join(CALIB_NAMES)}) round only the"
            f" linears of the DiT block sequence ({CALIB_TARGET}) stage by stage with the core's"
            " `karume.quant_calib`. The calibration input spans the steps of the reference loop"
            " (f32, CFG included), capturing the input to the first block at each step (the"
            " number of steps used is calib.steps in each config JSON). **The target set differs"
            " from i4-linear**, so even at the same bpw the quality rows are not directly"
            " comparable",
            "verdict": "the final ruling is by ear (the user) — the WAVs line up with the same"
            " text and the same seed",
        },
        "configs": dict(collected),
        "failures": {
            name: payload["gates"]["failures"]
            for name, payload in collected.items()
            if payload["gates"]["failures"]
        },
    }


# ---- previewing the target scale (`--scan`) -----------------------------------------

#: The target op sets the preview lists. `linear` is the form the i4 execution path can take as is
#: (ADR 0069 decision 5), `all` is {@link W4_OP_TYPES} widened for measurement. **The linear-only
#: form does not measure quality** — counting tells whether the model is conv-dominated, and if
#: it is, rounding only the linears does not move the distribution size, so running the full
#: loop for tens of minutes is not worth it (this table decides whether to run it).
SCAN_OP_SETS: Mapping[str, tuple[type[nn.Module], ...]] = MappingProxyType(
    {"linear": (nn.Linear,), "all": W4_OP_TYPES}
)

SCAN_FILE = "w4-scan.json"

#: The methods whose estimates the preview lists (the same order and the same objects as the 4
#: in the configuration table).
SCAN_METHODS: tuple[W4Method, ...] = (W4_RTN, W4_NF4, W4_MXFP4, W4_KMEANS_SHARED)


def scan_graphs(args: argparse.Namespace) -> dict[str, nn.Module]:
    """Build the 7 shipped graphs (6 roles + the codec's decode path) **without rounding**."""
    source, modules, config, _text_config, model_config = load_modules(
        args.model_dir, args.source_dir
    )
    patch.apply_patches()
    graphs = role_graphs(build_graphs(modules, config, model_config))
    stage = build_decoder(args.codec_source_dir, args.codec_model_dir, RECIPES[BASE_CONFIG])
    del source
    return {**graphs, ROLE_CODEC: stage.graph}


def scan_report(args: argparse.Namespace) -> dict[str, Any]:
    """The target scale and size estimates of role × op set (without running the full loop even
    once)."""
    graphs = scan_graphs(args)
    op_sets: dict[str, Any] = {}
    for label, op_types in SCAN_OP_SETS.items():
        scans = {role: scan_targets(graph, op_types) for role, graph in graphs.items()}
        counts = {role: scan.counts for role, scan in scans.items()}
        tables = sum(1 for item in counts.values() if item.modules)
        total = total_counts(list(counts.values()))
        op_sets[label] = {
            "opTypes": [cls.__name__ for cls in op_types],
            "roles": {
                role: {
                    "modules": item.modules,
                    "channels": item.channels,
                    "elements": item.elements,
                    "f32MiB": item.elements * 4 / 1024**2,
                }
                for role, item in counts.items()
            },
            "excluded": [
                {
                    "role": role,
                    "fqn": item.fqn,
                    "axisLength": item.axis_length,
                    "elements": item.elements,
                }
                for role, scan in scans.items()
                for item in scan.excluded
            ],
            "total": {
                "modules": total.modules,
                "channels": total.channels,
                "elements": total.elements,
                "f32MiB": total.elements * 4 / 1024**2,
            },
            "methods": {},
        }
        for method in SCAN_METHODS:
            size = project_size(method, total, tables)
            op_sets[label]["methods"][method.name] = {
                "formula": size.formula,
                "tables": size.tables,
                "bits": size.bits,
                "bitsPerWeight": size.bits_per_weight,
                "projectedMiB": size.projected_mib,
                "f32MiB": size.f32_mib,
                "ratio": size.projected_mib / size.f32_mib,
            }
    return {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "script": "tools/export-recipes/irodori/measure_quant.py --scan",
        "groupSize": W4_GROUP_SIZE,
        "note": "the counts cover only the weights in the shipped graphs (the rounded tree is the"
        " role modules = the same set as i8). The sizes are projections by formula, not"
        " measurements",
        "opSets": op_sets,
    }


def scan_scale_table(payload: Mapping[str, Any]) -> str:
    """The target scale of role × op set (**count, channels, elements**)."""
    header = ["op set", "role", "count", "channels", "elements", "f32 MiB"]
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    for label, entry in payload["opSets"].items():
        rows = [*entry["roles"].items(), ("total", entry["total"])]
        lines += [
            "| "
            + " | ".join(
                [
                    label,
                    role,
                    str(item["modules"]),
                    f"{item['channels']:,}",
                    f"{item['elements']:,}",
                    f"{item['f32MiB']:.2f}",
                ]
            )
            + " |"
            for role, item in rows
        ]
    return "\n".join(lines)


def scan_size_table(payload: Mapping[str, Any]) -> str:
    """Size estimates of op set × method (**projections by formula**, not measurements)."""
    header = ["op set", "method", "formula", "bpw", "projected MiB", "f32 MiB", "ratio"]
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    for label, entry in payload["opSets"].items():
        lines += [
            "| "
            + " | ".join(
                [
                    label,
                    name,
                    size["formula"],
                    f"{size['bitsPerWeight']:.3f}",
                    f"{size['projectedMiB']:.1f}",
                    f"{size['f32MiB']:.1f}",
                    f"{size['ratio']:.3f}",
                ]
            )
            + " |"
            for name, size in entry["methods"].items()
        ]
    return "\n".join(lines)


def scan_excluded_lines(payload: Mapping[str, Any]) -> list[str]:
    """The list of weights excluded from the targets as not aligned to g32 (per op set)."""
    lines: list[str] = []
    for label, entry in payload["opSets"].items():
        if not entry["excluded"]:
            lines.append(f"- {label}: no exclusions for g32 misalignment")
            continue
        lines.append(f"- {label}: {len(entry['excluded'])} not aligned to g32")
        lines += [
            f"    - {item['role']} / {item['fqn']} axis length {item['axisLength']}"
            f" ({item['elements']:,} elements)"
            for item in entry["excluded"]
        ]
    return lines


def run_scan(args: argparse.Namespace) -> None:
    """Run {@link scan_report}, write the JSON and print the markdown tables to stdout."""
    payload = scan_report(args)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / SCAN_FILE).write_text(
        json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print()
    print("w4 target scale (only the weights in the shipped graphs, unrounded):")
    print(scan_scale_table(payload))
    print()
    print("exclusions for g32 misalignment:")
    print("\n".join(scan_excluded_lines(payload)))
    print()
    print("size estimates (projections by formula, not measurements):")
    print(scan_size_table(payload))
    print()
    print(f"scan → {args.out / SCAN_FILE}")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model-dir", type=Path, default=ex.DEFAULT_MODEL_DIR)
    parser.add_argument("--source-dir", type=Path, default=ex.DEFAULT_SOURCE_DIR)
    parser.add_argument("--codec-model-dir", type=Path, default=dv.DEFAULT_MODEL_DIR)
    parser.add_argument("--codec-source-dir", type=Path, default=dv.DEFAULT_SOURCE_DIR)
    parser.add_argument("--golden-dir", type=Path, default=DEFAULT_GOLDEN_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--config",
        choices=tuple(RECIPES),
        default=None,
        help="the configuration to run (one run = one configuration). When omitted nothing is"
        f" measured; {REPORT_FILE} and the tables are only rebuilt from the outputs left in --out",
    )
    parser.add_argument(
        "--scan",
        action="store_true",
        help="print only the w4 target scale and size estimates and exit (no full loop)",
    )
    parser.add_argument(
        "--calib-steps",
        type=int,
        default=ip.NUM_STEPS,
        help="steps per case used by the calibrated configurations (default = all"
        f" {ip.NUM_STEPS} steps of the reference loop). The reference loop is wound down once they"
        " are captured, so lower this for a reduced smoke run. The value used is recorded in the"
        " output",
    )
    parser.add_argument(
        "--kmeans-fit-stride",
        type=int,
        default=1,
        help="use an evenly spaced subsample for **only the table fit** of kmeans:shared"
        " (application is always to everything). A full fit of the dit role exceeds 20 GB"
        " including the f64 work area, so raise this for kmeans-shared-all to fit the real memory."
        " The value used is recorded in the output",
    )
    args = parser.parse_args(argv)
    if not 1 <= args.calib_steps <= ip.NUM_STEPS:
        parser.error(f"--calib-steps must be 1 to {ip.NUM_STEPS} (the reference loop's step count)")

    if args.scan:
        run_scan(args)
        return

    if args.config is not None:
        payload = run_config(args.config, args)
        print(f"[{args.config}] {payload['elapsed']:.0f}s → {args.out / f'{args.config}.json'}")

    collected = collect(args.out)
    if not collected:
        raise SystemExit(f"{args.out} has no configuration outputs at all (run with --config)")
    report = build_report(collected)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / REPORT_FILE).write_text(
        json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print()
    print("S drift (the mixed-table ruling rests on this table):")
    print(drift_table(collected))
    print()
    print("quality (on the same time grid as the baseline):")
    print(quality_table(collected))
    exclusions = excluded_lines(collected)
    if exclusions:
        print()
        print("w4 size estimates (projections by formula, not measurements):")
        print(size_table(collected))
        print()
        print("exclusions for g32 misalignment:")
        print("\n".join(exclusions))
    print()
    print(f"report → {args.out / REPORT_FILE}")
    if report["failures"]:
        raise AssertionError(
            "verification gates are red: "
            + " / ".join(
                f"{name}: {'; '.join(items)}" for name, items in report["failures"].items()
            )
        )
    print("gates: all green")


if __name__ == "__main__":
    main()
