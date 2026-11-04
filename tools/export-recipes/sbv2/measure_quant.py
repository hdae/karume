"""Measure the audio degradation of SBV2's **generator network** (net_g) per quantization pattern on
torch CPU.

A quality gate in the same spirit as `measure_quant_anima.py` (perf-c Q0) — before building any
execution kernel, it answers first, with torch fake-quant only, "how far does this quantization
degrade the audio". Zero lines of GPU code; what it measures is **the quality of the quantization
itself** (under ADR 0006's fake-quant methodology the E2E tests measure only implementation error,
so quality has to be measured on a separate axis).

It runs 14 configurations on **the same utterance with the same random numbers** (the discrete
inputs and noise sequences of `outputs/examples/karume-sbv2-jvnv/sbv2-dump/dump.safetensors`). The
existing 5 configurations vary **only the generator network** (**BERT stays f32**):

    (1) f32    the baseline. The same path as `sbv2.demo reference` (bit-identical to the existing
               reference.wav)
    (2) f16    rounds the eligible weights to f16-representable values
               (`quantize.round_weights_to_f16`). Activations stay f32
               = the counterpart of the implemented **f16 storage series** (ADR 0027)
    (3) w8     per-channel symmetric i8 fake-quant of the weights (`quantize.fake_quant_int8`)
               = the counterpart of ADR 0019's w8a32 (i8 storage, f32 compute)
    (4) w8a16  (3) + **the input activations of the target ops** rounded to f16
    (5) w8a8   (3) + per-position symmetric i8 fake-quant of **the input activations of the target
               ops**

The 6 w4 configurations (the 4 winners of the method screening; the group size g is **32 by default
and variable with `--w4-group-size`** — ADR 0069 addendum 5 / the g axis of wave J-3). All rounding
is shared with core (`karume.quantize` / `karume.quant_methods`); the script has no local
implementation:

    (6) w4-rtn      **every role** of net_g (conv1d / conv_transpose1d / linear / embedding) as
                    RTN group symmetric i4 (the `op_types` opt-in of `fake_quant_int4`)
    (7) w4-nf4      the same roles with NF4's fixed table × group absmax scale
    (8) w4-mxfp4    the same roles with the FP4 e2m1 table × a power-of-two scale per group (OCP MX)
    (9) w4-kmeans   the same roles with a k-means codebook (a table shared across layers × group
                    normalization)
    (10) bert-w4-rtn  **only the linears of BERT (DeBERTa)** as RTN i4 (group size g); net_g stays
                      f32
    (11) bert-w4-nf4  the same with NF4

The 2 BERT-side configurations are **the main distribution candidates** — `shared/text_encoder`
takes a little under 30% of the karume-sbv2-jvnv distribution, and linear is exactly today's i4
execution path (ADR 0069 decision 5), so if they pass listening they become distribution candidates
as they are. The net_g-side w4, by contrast, **has no storage format** (the scale of the conv family
is a rank-2 tensor over the flattened receptive field and structurally cannot be handed to emit —
addendum 5), so it is a measurement series that answers, with zero lines of GPU code, "how much
quality does non-linear w4 keep". The **linear-only** form of net_g has no quality measurement —
it covers 6 tensors / under 1% of the eligible elements, and **the absence of any gain is itself the
reason for the decision**, so only the size estimate goes to the report.

Weights whose quantization axis (the in axis of each op — the receptive field `Cin·K` for conv) is
not divisible by the group size are **excluded** from the targets (i4 MUST NOT create a partial
group — core fails loudly per ADR 0069 decision 2). The exclusion list, the target size per role
and the size estimate are in the `w4` section of `report.json`.

## Calibrated rounding ({@link CALIB_CONFIGS}, wave J-2)

The w4 above rounds by looking at the weights only (the RTN family). **Calibrated rounding** (GPTQ
— core's `karume.quant_calib`) re-chooses the rounding targets **within the same grid** from "the
activations that actually flow into that layer", and adds 3 configurations on the BERT side (net_g
stays f32; the grid is byte-for-byte the same as the method grid above):

    (12) bert-gptq-rtn     BERT's linears with GPTQ × RTN i4 g32 (the calibrated `bert-w4-rtn`)
    (13) bert-gptq-nf4     the same with NF4's fixed table (the calibrated `bert-w4-nf4`)
    (14) bert-gptq-kmeans  the same with a k-means table (core's `kmeans_shared` = one **per
                           layer**)

The calibration inputs are the 48 sentences of {@link deberta.calib_texts.CALIB_TEXTS}, taken
through the existing DeBERTa tokenizer path; the call into the first encoder layer is captured and
fed into the stage sequence ({@link capture_stage_batches}). The stages go **only up to the layer
the features are taken from**, so the targets are a subset of the `bert:linear` census
({@link CALIB_TARGET}). `--calib-limit` allows a reduced run, and the fact that it was reduced is
recorded both in the table and in `report.json`.

    uv run --group sbv2 python -m sbv2.measure_quant

The output goes to `outputs/bench/karume-sbv2-jvnv/<date>_quant-sim/` (14 `<config>.wav` +
`report.json`). `--configs` narrows the main configurations by name (`f32` always runs because it
is the SNR baseline).
The w4 group size can be varied with
`--w4-group-size` (a power of two, at least 16, default 32) — the same g flows into eligibility,
rounding and the method name (the grid of the calibrated configurations stays 32 =
{@link CALIB_GROUP_SIZE}).

## Granularity and application point of activation quantization

The application point is **the input of the target op** (not its output). The targets are the ops
with a weight slot whose activations are continuous values, that is `conv1d` / `conv_transpose1d` /
`linear`. `embedding` is out of scope because its input is integer IDs (only its weights are
quantized).

**MUST: the replacement is per op** (it replaces the `torch.nn.functional` functions) — a module
`forward_pre_hook` is not enough. `sbv2.patch._patched_ffn_forward` calls
`functional.conv1d(x, self.conv_1.weight, …)` directly instead of `self.conv_1(x)`, so a hook on
`nn.Conv1d` never fires (the hole the first version of this script fell into — all 12 enc_p + 48
flow FFN convs passed through untouched, and it stayed silent in the direction of "the degradation
looks mild while we think we are quantizing"). On the Karume side the IR and the kernels also
consume activations per op, so per op is semantically correct too.

The granularity is ADR 0025's **per-token ±127** carried over to conv1d:

    linear           [..., k]     row = the last axis (ADR 0025 itself)
    conv1d           [B, Cin, T]  row = **the Cin axis per time step t** (amax with `dim=-2`)
    conv_transpose1d [B, Cin, T]  the same (the `[Cin,Cout,K]` transpose of the weights concerns
                                  the weights and does not affect the activations)

The numeric formula **reuses** `karume.act_quant.quantize_rows` (no copy) — the axis is transposed
to the last axis and the same function is applied. So `s = clamp(rowmax|x|/127, tiny)` /
`q = clamp(round(x/s), ±127)` / `x̂ = q·s` (round half to even; an all-zero row is exactly 0) share
one source of truth.

**Why "per time step"** (it has to be a granularity a future kernel can implement): building conv1d
as an implicit GEMM makes the reduction axis `Cin·K` (taps included), and one im2col column
gathers values from K time steps, so "one column = one scale" needs the amax over the whole window.
conv1d can also be decomposed into **a shift-and-add of per-tap 1x1 convs (= GEMMs)**, and in that
form the reduction axis of one tap is only `Cin`, so "one scale per time step" has exactly the same
shape as ADR 0025's per-token, and the integer reduction `(Σ q_a·q_w)·(s_a·s_w)` holds (the taps
are summed in f32). The granularity measured here corresponds to that decomposition. The window-amax
scheme (one scale per column) takes the max over the same set of time steps, so it is **always
coarser than this one**; read this measurement as **the optimistic upper bound** of w8a8 conv1d.

NOTE: alignment conditions of i8 packing such as `k % 4 == 0` are not emulated (an implementation
detail that padding satisfies, irrelevant to the quality question).

## Fixing the time grid (a MUST so that SNR is a meaningful quantity)

The front's output `logw` decides the durations, so if quantization flips even one `ceil`, **the
frame count itself changes**, the waveform lengths disagree between configurations, and SNR cannot
be defined. So every configuration expands with **the dump's `w_ceil`** (the same choice
`sbv2.demo.run_reference` makes because of the shape of zp_noise). How many frames each
configuration "would have produced with its own w_ceil" is kept as a diagnostic in the `w_ceil`
field of `report.json` — if these differ, read it as "the SNR compares on the same time grid, and in
real use the utterance length changes too".
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import re
import time
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from itertools import combinations
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol

import torch
from safetensors.torch import load_file
from torch import nn

from _shared.container_read import read_layouts
from _shared.paths import BENCH_ROOT, DIST_ROOT, EXAMPLES_ROOT
from deberta.calib_texts import CALIB_TEXTS
from karume.act_quant import quantize_rows
from karume.container import container_parts
from karume.dist import ir_graph
from karume.ir import MIN_GROUP_SIZE
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
    channel_rows,
    fake_quant_int4,
    fake_quant_int8,
    iter_quant_targets,
    round_weights_to_f16,
)

from . import demo, export, patch

#: The default dump input (the default output location of `examples/sbv2/dump.ts` — that side owns
#: the spelling).
DEFAULT_DUMP_DIR = EXAMPLES_ROOT / "karume-sbv2-jvnv" / "sbv2-dump"
DEFAULT_DUMP = DEFAULT_DUMP_DIR / "dump.safetensors"
#: What the f32 configuration is compared against so that it cannot pass vacuously (the WAV that
#: `sbv2.demo reference` wrote from the same dump).
DEFAULT_REFERENCE_WAV = DEFAULT_DUMP_DIR / "reference.wav"
#: Where the results go. These are benchmark products, so they are kept apart from both the assets
#: (`sbv2.demo.DEFAULT_DEMO_DIR`) and the series — so that a clean-up (`rm -rf outputs/bench`) does
#: not take those with it (docs/assets-layout.md).
DEFAULT_OUT = BENCH_ROOT / "karume-sbv2-jvnv" / f"{date.today().isoformat()}_quant-sim"
#: The distribution the size reduction is measured against (the output of `karume dist` — only its
#: real bytes are read, nothing is written).
DEFAULT_DIST_DIR = DIST_ROOT / "karume-sbv2-jvnv"


# ---- w4 vocabulary (methods, roles) — ADR 0069 addendum 5 --------------------------------

# The w4 group size g is varied with `--w4-group-size` (default = core's storage default
# `DEFAULT_GROUP_SIZE`) — the g axis of wave J-3. To look only at the differences between methods,
# run with the default (varying both axes at once mixes the method differences with the g
# differences — the framing of ADR 0069 addendum 5).
#
# MUST: g is **passed as an argument**, not held as a module-level constant. Eligibility
# (`census_w4_targets`), rounding (the `group_size` that `w4_methods` passes to `fake_quant_*`) and
# the method name (`rtn_method_name`) must share one value — moving only one of them silently
# produces "a g32 measurement labelled g16" (if only the rounding moves, core fails loudly on a
# partial group, but if only eligibility moves the numbers still come out with mismatched grids).
# The grid of the calibrated configurations is **not on** this axis (`CALIB_GROUP_SIZE`).

#: The bit count of one shared k-means table (16 levels × f32). Quality alone does not show **the
#: cost of the table**, so the size estimate puts the table cost into the formula explicitly.
CODEBOOK_BITS = DEFAULT_CODEBOOK_LEVELS * 32


@dataclass(frozen=True)
class TargetCounts:
    """Counts of a w4 target set (the input of the size estimate).

    The group size lives in the same container as the counts so that **"the g that was counted"
    and "the g that was projected" cannot silently diverge** — the bpw projection derives from the
    group count, so carrying g as a separate argument lets one of them go stale.
    """

    modules: int
    channels: int
    elements: int
    group_size: int

    def __add__(self, other: TargetCounts) -> TargetCounts:
        if self.group_size != other.group_size:
            raise AssertionError(
                f"group 長の違う計数は足せない（{self.group_size} と {other.group_size}）"
            )
        return TargetCounts(
            modules=self.modules + other.modules,
            channels=self.channels + other.channels,
            elements=self.elements + other.elements,
            group_size=self.group_size,
        )

    @property
    def groups(self) -> int:
        """The total number of groups (used in the estimate of methods that need one scale per
        group).

        `elements = Σ channel count × in axis`, and the targets are only those whose in axis is
        divisible by the group size (the indivisible weights are excluded by
        {@link census_w4_targets}), so there is no remainder.
        """
        return self.elements // self.group_size


class QuantReport(Protocol):
    """The common surface of the count reports core's `fake_quant_*` return (`Int4Report` /
    `MethodReport`)."""

    modules: int
    elements: int

    def describe(self) -> str: ...


@dataclass(frozen=True)
class W4Method:
    """One w4 rounding method — binds the name, how to apply it and the size-estimate formula in
    one row.

    Spreading the three over separate tables lets "the method whose quality was measured" and "the
    method whose size was estimated" silently diverge (adding a method would update only one of
    them). All rounding is shared with core and this only dispatches — writing rounding locally in
    the script would let the measured formula and the shipped formula start to move independently.
    """

    #: The method name shown in the table and the JSON (only RTN bakes g into it —
    #: {@link rtn_method_name}). The table's **key** is the method kind, which does not depend on
    #: g, so varying g does not move what the configuration table {@link CONFIGS} points at.
    name: str
    #: Applies the rounding to the model in place (`(model, op_types, include)`). The group size is
    #: closed over by {@link w4_methods}, so the grid is fixed to one even without the caller
    #: carrying g.
    apply: Callable[[nn.Module, tuple[type[nn.Module], ...], Callable[[str], bool]], QuantReport]
    #: The projected bit count of the quantization target set.
    projected_bits: Callable[[TargetCounts], float]
    #: The formula of `projected_bits` (**written to the output as is** — so the assumptions of
    #: the projection can be traced from the table).
    formula: str


#: The **method kind** of RTN (= the only storage format, `i4`) = a key of {@link w4_methods}. It
#: is **the baseline** of the method series; the other 3 kinds are read as differences from it.
RTN_KIND = "rtn"


def rtn_method_name(group_size: int) -> str:
    """The RTN method name (**g is baked into the name**).

    MUST: the default 32 gives the same `rtn-i4-g32` as before — past research records (the method
    screening of ADR 0069 addendum 5) keep numbers under this spelling, and the default run's
    comparison target must not move. The other 3 kinds carry no g in their names (their spelling is
    kept for the same reason), so g is read from `group_size` of the `w4` section and from each
    method's formula ({@link W4Method.formula}).
    """
    return f"rtn-i4-g{group_size}"


def w4_methods(group_size: int) -> Mapping[str, W4Method]:
    """The table of the 4 methods (the winners of the screening) bound to the group size
    `group_size`.

    The keys are **the method kinds** (independent of g) and only the values carry g — so that
    what the configuration table {@link CONFIGS} points at does not move with g. The table is
    **built** from g so that the `group_size` passed to the rounding, the formula shown in the
    table and the method name derive from the same single g (holding them separately makes "a g32
    measurement labelled g16" possible).

    Rejected and therefore absent: per_tensor / per_channel of FP4 and k-means (rejected by the
    measurements on 2 cheap families — ADR 0069 addendum 5).
    """
    return MappingProxyType(
        {
            RTN_KIND: W4Method(
                rtn_method_name(group_size),
                lambda model, op_types, include: fake_quant_int4(
                    model, group_size, include=include, op_types=op_types
                ),
                lambda counts: 4 * counts.elements + 32 * counts.groups,
                f"4bit + g{group_size} f32 scale = {4 + 32 / group_size} bpw",
            ),
            "nf4": W4Method(
                "nf4",
                lambda model, op_types, include: fake_quant_nf4(
                    model, group_size, include=include, op_types=op_types
                ),
                lambda counts: 4 * counts.elements + 32 * counts.groups,
                f"4bit + g{group_size} f32 scale = {4 + 32 / group_size} bpw"
                " (the level table is a fixed value, so it is not stored in the model)",
            ),
            "mxfp4": W4Method(
                "mxfp4",
                lambda model, op_types, include: fake_quant_mxfp4(
                    model, group_size, include=include, op_types=op_types
                ),
                lambda counts: 4 * counts.elements + 8 * counts.groups,
                f"4bit + g{group_size} E8M0 scale = {4 + 8 / group_size} bpw",
            ),
            "kmeans:shared": W4Method(
                "kmeans:shared",
                lambda model, op_types, include: fake_quant_kmeans(
                    model, "shared", group_size, include=include, op_types=op_types
                ),
                lambda counts: 4 * counts.elements + 32 * counts.groups + CODEBOOK_BITS,
                f"4bit + g{group_size} f32 scale + one {DEFAULT_CODEBOOK_LEVELS}×f32 table overall",
            ),
        }
    )


#: The w4 target **roles** → `op_types`. `all` passes the same 5-op-type table as i8
#: (`QUANT_MODULE_TYPES`) as is; net_g contains 4 of them (conv1d / conv_transpose1d / linear /
#: embedding) — it is not copied into a separate list (if they move independently, "the types an
#: axis can be taken from" and "the types that are targets" silently diverge). `linear` is
#: **today's distributable form** (the i4 execution path is linear only — ADR 0069 decision 5).
W4_ROLES: Mapping[str, tuple[type[nn.Module], ...]] = MappingProxyType(
    {"all": QUANT_MODULE_TYPES, "linear": (nn.Linear,)}
)


# ---- Calibrated rounding vocabulary (wave J-2) --------------------------------------------

#: The group size of calibrated rounding = **pinned to 32** (not on the g axis of
#: `--w4-group-size`). This compares methods in which calibration **does not change the grid by a
#: single byte** (it only re-chooses rounding targets within the same grid — the module docstring of
#: `karume.quant_calib`), so what should vary is the method, not g. It is an independent constant
#: instead of borrowing the method grid's g so that varying the g axis does not **silently move**
#: the calibration grid along with it — a mismatch is readable from `w4.group_size` and
#: `w4.calib.group_size` being reported separately.
CALIB_GROUP_SIZE = DEFAULT_GROUP_SIZE

#: The target name of the calibrated configurations (the target column of the `w4` section). It is
#: **a different set** from the `bert:linear` census — the calibration stages are built only up to
#: the layer the features are taken from ({@link bert_stages}), so the linears of later layers are
#: not included. In exchange it **matches the linear set of the distribution (the 22-layer
#: variant)**, so the reduction estimate of {@link project_distribution} is about this very set.
CALIB_TARGET = "bert:linear:calib"


def group_scale_bits(counts: TargetCounts) -> float:
    """4-bit payload + one f32 scale per group (exactly the storage of RTN / NF4)."""
    return 4 * counts.elements + 32 * counts.groups


def layer_table_bits(counts: TargetCounts) -> float:
    """The formula above plus **one k-means table per layer**.

    Core's `kmeans_shared` spans "one table over the value space normalized by the group absmax",
    but its reach is closed **within a layer** (driving stage by stage does not mesh with a table
    across all layers — the NOTE of {@link karume.quant_calib.GridSpec}). This is why the formula
    differs from the method grid's `kmeans:shared` (one table overall = {@link CODEBOOK_BITS} only
    once).
    """
    return group_scale_bits(counts) + CODEBOOK_BITS * counts.modules


@dataclass(frozen=True)
class CalibConfig:
    """The spec of one calibrated rounding (method × storage grid).

    The rounding implementation is shared with core (`karume.quant_calib`); this only holds the
    dispatch and the projection formula — for the same reason as {@link W4Method} (writing rounding
    locally in the script lets the measured formula and the shipped formula start to move
    independently).
    """

    name: str
    method: CalibMethod
    grid: GridSpec
    #: The projected bit count of the quantization target set (the same container as
    #: {@link W4Method.projected_bits}).
    projected_bits: Callable[[TargetCounts], float]
    #: The formula of `projected_bits` (**written to the output as is**).
    formula: str

    @property
    def label(self) -> str:
        """The method name shown in the table and the JSON (`<method>/<storage grid>`)."""
        return f"{self.method}/{self.grid.kind}"


#: 3 calibration grids = GPTQ × 3 storage grids (all of core's `GRID_KINDS`). The **calibrated
#: versions** of the existing `bert-w4-rtn` / `bert-w4-nf4`; the grid does not change by a byte.
#:
#: There is no AWQ family because its rounded weights take the form `W_eff = Q(W')/s`, and **they
#: cannot be stored on their own** until the equivalent scale `s` is folded into an adjacent op or
#: shipped as a companion tensor (the module docstring of `karume.quant_calib`). This is the slot
#: for **checking how the distribution candidates sound by listening to the WAVs**, so a series with
#: no prospect of storage would only add WAVs to listen to (the slot that measures the absence of a
#: storage format is the 4 net_g-side w4 configurations — the opposite position).
CALIB_CONFIGS: Mapping[str, CalibConfig] = MappingProxyType(
    {
        config.name: config
        for config in (
            CalibConfig(
                "gptq-rtn",
                "gptq",
                GridSpec(kind="rtn", group_size=CALIB_GROUP_SIZE),
                group_scale_bits,
                "4bit + g32 f32 scale = 5.0 bpw (stored exactly as RTN i4)",
            ),
            CalibConfig(
                "gptq-nf4",
                "gptq",
                GridSpec(kind="nf4", group_size=CALIB_GROUP_SIZE),
                group_scale_bits,
                "4bit + g32 f32 scale = 5.0 bpw (the grid is NF4's fixed table)",
            ),
            CalibConfig(
                "gptq-kmeans",
                "gptq",
                GridSpec(kind="kmeans_shared", group_size=CALIB_GROUP_SIZE),
                layer_table_bits,
                f"4bit + g32 f32 scale + one {DEFAULT_CODEBOOK_LEVELS}×f32 table **per layer**",
            ),
        )
    }
)


@dataclass(frozen=True)
class Recipe:
    """The quantization recipe of one configuration.

    `scope` is the **subgraph** the rounding is applied to (`front` = enc_p + dp + sdp / `voice` =
    flow + dec). Configurations that vary net_g apply it to both (= the whole generator network).
    The one-sided forms are for diagnostics (below), and configurations that vary only BERT have an
    empty scope (net_g stays f32).
    """

    weight: str | None
    act: str | None
    scope: tuple[str, ...] = ("front", "voice")
    #: The w4 method kind (meaningful only when `weight == "w4"` — a key of {@link w4_methods}).
    method: str | None = None
    #: The w4 target role (likewise — a key of {@link W4_ROLES}).
    roles: str = "all"
    #: The w4 method applied to BERT's (DeBERTa's) linears. **None = stays f32** (the contract of
    #: the existing 5 configurations).
    bert_method: str | None = None
    #: The **calibrated** rounding applied to BERT's linears (a key of {@link CALIB_CONFIGS}).
    #: Combining it with `bert_method` is a mistake in the configuration ({@link bert_variant}
    #: rejects it).
    bert_calib: str | None = None


#: The 14 main configurations (their WAVs go directly under `--out` and are what is listened to).
CONFIGS: Mapping[str, Recipe] = MappingProxyType(
    {
        "f32": Recipe(None, None),
        "f16": Recipe("f16", None),
        "w8": Recipe("i8", None),
        "w8a16": Recipe("i8", "f16"),
        "w8a8": Recipe("i8", "i8"),
        # w4 on every role of net_g (a measurement series with no storage format — module
        # docstring).
        "w4-rtn": Recipe("w4", None, method=RTN_KIND),
        "w4-nf4": Recipe("w4", None, method="nf4"),
        "w4-mxfp4": Recipe("w4", None, method="mxfp4"),
        "w4-kmeans": Recipe("w4", None, method="kmeans:shared"),
        # BERT's linears only (net_g stays f32 — a separate axis orthogonal to the existing 5).
        "bert-w4-rtn": Recipe(None, None, scope=(), bert_method=RTN_KIND),
        "bert-w4-nf4": Recipe(None, None, scope=(), bert_method="nf4"),
        # Calibrated versions of the 2 above (GPTQ, net_g stays f32, same grid — wave J-2).
        "bert-gptq-rtn": Recipe(None, None, scope=(), bert_calib="gptq-rtn"),
        "bert-gptq-nf4": Recipe(None, None, scope=(), bert_calib="gptq-nf4"),
        "bert-gptq-kmeans": Recipe(None, None, scope=(), bert_calib="gptq-kmeans"),
    }
)

#: Diagnostic configurations (written to `--out/diagnostics/`). They **decompose the degradation
#: orthogonally** into the front side and the voice side.
#:
#: The values recorded by the earlier experiment (`docs/research/2026-08-03-demo-w8-perf-recon.md`:
#: int8 per-channel voice SNR 12.8–24.2dB) measure **voice alone**, so they cannot be compared
#: directly with the numbers of the full configurations. The counterpart for matching magnitudes is
#: this `w8-voice-only`.
DIAGNOSTICS: Mapping[str, Recipe] = MappingProxyType(
    {
        "w8-front-only": Recipe("i8", None, ("front",)),
        "w8-voice-only": Recipe("i8", None, ("voice",)),
        "w8a8-front-only": Recipe("i8", "i8", ("front",)),
        "w8a8-voice-only": Recipe("i8", "i8", ("voice",)),
    }
)

#: The full table of configuration name → recipe (`run_config` looks it up here).
RECIPES: Mapping[str, Recipe] = MappingProxyType({**CONFIGS, **DIAGNOSTICS})

#: The **ops** whose activations are quantized (`torch.nn.functional` names) → the axis of the input
#: tensor's "rows".
#:
#: MUST: the application point is **the op, not the module**. Thinking a module `forward_pre_hook`
#: is enough is wrong: `sbv2.patch._patched_ffn_forward` calls
#: `functional.conv1d(x, self.conv_1.weight, …)` directly instead of `self.conv_1(x)`, so a hook on
#: `nn.Conv1d` **never fires** (measured: all 12 enc_p + 48 flow FFN convs passed through
#: untouched). On the Karume side the IR/kernels also consume activations per op, so per op is
#: semantically correct too.
#:
#: `embedding` is not listed because its input is integer IDs (only its weights are quantized).
ACT_ROW_AXIS: Mapping[str, int] = MappingProxyType(
    {
        "conv1d": -2,
        "conv_transpose1d": -2,
        "linear": -1,
    }
)

#: The op objects before replacement (the source of truth of the restoration check — captured once
#: at import).
PRISTINE_OPS: Mapping[str, Any] = MappingProxyType(
    {op: getattr(torch.nn.functional, op) for op in ACT_ROW_AXIS}
)

#: Collapses the layer indices to group layers by role (`flow.flows.0.enc.in_layers.3|conv1d` →
#: `flow.flows.*.enc.in_layers.*|conv1d`). Keys have the form `<module FQN>|<op>`.
_INDEX = re.compile(r"\.\d+(?=[.|]|$)")


def layer_group(name: str) -> str:
    return _INDEX.sub(".*", name)


# ---- Activation fake-quant hooks ------------------------------------------------


@dataclass
class ActStat:
    """The activation quantization error of one call site (module name × op), as running sums of
    squares."""

    op: str
    axis: int
    calls: int = 0
    err2: float = 0.0
    ref2: float = 0.0
    saturated: int = 0

    def rel_rms(self) -> float:
        return math.sqrt(self.err2 / self.ref2) if self.ref2 > 0 else 0.0


def round_to_f16(x: torch.Tensor) -> torch.Tensor:
    """Rounds activations to f16-representable values (computation stays f32).

    The weight-side `round_weights_to_f16` fails loudly on saturation, but activation values are
    only known at run time, so saturation is **counted and reported** (whether saturation occurs is
    itself the answer to whether w8a16 is viable, and stopping midway would throw away the
    measurements of the remaining 4 configurations, which costs more).
    """
    return x.to(torch.float16).to(torch.float32)


def quantize_activation(x: torch.Tensor, axis: int) -> torch.Tensor:
    """Per-row symmetric i8 fake-quant treating `axis` as the row.

    The numeric formula uses `act_quant.quantize_rows` (ADR 0025's source of truth, with the last
    axis as the row) **as is**, matching only the axis by a transpose — copying the formula would
    let one copy drift from the spec.
    """
    if axis in (-1, x.dim() - 1):
        return quantize_rows(x)
    return quantize_rows(x.transpose(axis, -1)).transpose(axis, -1).contiguous()


@dataclass
class ActQuant:
    """Rounds **the inputs** of the target ops and aggregates the error per call site.

    The mechanism has 2 parts:

    1. **Replaces** `torch.nn.functional.{conv1d, conv_transpose1d, linear}` (the rounding itself).
       Because it is per op, calls through a module and calls the patch layer makes directly are
       caught at the same single place (the MUST of {@link ACT_ROW_AXIS}).
    2. Puts push/pop hooks on **every module** under `roots` and keeps a name stack (no rounding —
       only to attribute the error to "the layer where it happened").

    An empty stack = outside `roots`, so no rounding. This is how `scope` is implemented: in the
    diagnostic configurations that quantize only the front, the voice calls pass through untouched
    (the only way to cut a scope while keeping the global replacement).

    `mode` is `"f16"` (w8a16) or `"i8"` (w8a8). `detach` always restores the replacement.
    """

    roots: tuple[tuple[str, nn.Module], ...]
    mode: str
    disable: bool = False
    stats: dict[str, ActStat] = field(default_factory=dict)
    handles: list[torch.utils.hooks.RemovableHandle] = field(default_factory=list)
    stack: list[str] = field(default_factory=list)
    patched: dict[str, Any] = field(default_factory=dict)
    skipped_calls: int = 0

    def attach(self) -> int:
        """Installs the name-stack hooks and replaces the target ops. Returns the number of
        replaced ops.

        MUST: return the count so the caller can detect 0 (the same convention as
        `act_quant.attach_act_quant` — running with 0 means "took the w8 numbers while meaning to
        measure w8a8" goes unnoticed).
        """
        for tag, root in self.roots:
            for name, module in root.named_modules():
                qualified = f"{tag}.{name}" if name else tag
                self.handles.append(module.register_forward_pre_hook(self._make_push(qualified)))
                self.handles.append(module.register_forward_hook(self._pop))
        if self.disable:
            # Fault injection (`--inject drop-act-quant`): build only the attribution scaffolding,
            # without rounding.
            return 0
        for op, axis in ACT_ROW_AXIS.items():
            original = getattr(torch.nn.functional, op)
            self.patched[op] = original
            setattr(torch.nn.functional, op, self._make_wrapper(op, axis, original))
        return len(self.patched)

    def detach(self) -> None:
        """Always restores the replacements and the hooks (no leaks between configurations)."""
        for op, original in self.patched.items():
            setattr(torch.nn.functional, op, original)
        self.patched.clear()
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.stack.clear()

    def _make_push(self, name: str):
        def push(_module: nn.Module, _args: tuple[Any, ...]) -> None:
            self.stack.append(name)

        return push

    def _pop(self, _module: nn.Module, _args: tuple[Any, ...], _output: Any) -> None:
        self.stack.pop()

    def _make_wrapper(self, op: str, axis: int, original: Any):
        def wrapper(*args: Any, **kwargs: Any):
            if not self.stack:
                # Outside roots (= out of scope). Pass through.
                return original(*args, **kwargs)
            if not args or not isinstance(args[0], torch.Tensor):
                # A call whose input comes as a keyword cannot be rounded. Count it to make it
                # visible.
                self.skipped_calls += 1
                return original(*args, **kwargs)
            x = args[0]
            key = f"{self.stack[-1]}|{op}"
            stat = self.stats.setdefault(key, ActStat(op=op, axis=axis))
            rounded = round_to_f16(x) if self.mode == "f16" else quantize_activation(x, axis)
            if self.mode == "f16":
                stat.saturated += int((torch.isfinite(x) & ~torch.isfinite(rounded)).sum())
            stat.calls += 1
            stat.err2 += float((rounded - x).square().sum())
            stat.ref2 += float(x.square().sum())
            return original(rounded, *args[1:], **kwargs)

        return wrapper


# ---- Metrics -------------------------------------------------------------------


def snr_db(value: torch.Tensor, reference: torch.Tensor) -> float:
    """`10·log10(Σref² / Σ(value−ref)²)` (the main metric for comparing audio quality between
    configurations)."""
    err = float((value - reference).double().square().sum())
    ref = float(reference.double().square().sum())
    if err == 0.0:
        return math.inf
    if ref == 0.0:
        return -math.inf
    return 10.0 * math.log10(ref / err)


#: The STFT settings of LSD. The hop matches the assets' `hopLength` (dec's total upsampling rate).
STFT_N_FFT = 2048
STFT_HOP = 512
#: The amplitude floor (relative to the reference's maximum amplitude = −100dB). A lower bound only
#: to keep the log of silent regions from falling to −inf; **the same floor** is applied to both
#: signals (applying it to one side only would make the difference depend on the floor).
STFT_FLOOR = 1e-5


def log_spectral_distance(value: torch.Tensor, reference: torch.Tensor) -> float:
    """The log-spectral distance `sqrt(mean((20log10|X| − 20log10|Y|)²))` [dB].

    Waveform SNR collapses with even a small phase shift (if the trajectory of HiFi-GAN's
    excitation changes, the residual is of the same order as the signal even when it sounds the
    same). In this measurement too, the order of the configurations swaps in the 13–16dB SNR band.
    LSD looks only at the magnitude spectrum, so it is **insensitive to phase shifts**, and it is
    reported alongside as the metric of "how the timbre and the bands break". Neither one alone
    decides.
    """
    window = torch.hann_window(STFT_N_FFT)
    spectra = [
        torch.stft(
            wave, n_fft=STFT_N_FFT, hop_length=STFT_HOP, window=window, return_complex=True
        ).abs()
        for wave in (value, reference)
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


# ---- w4 target set (census) --------------------------------------------------


@dataclass(frozen=True)
class W4Census:
    """The w4 target set of one root × one role — the eligible counts and the weights left out
    because they are not divisible by the group size.

    The **source of truth for target selection is core** (`quantize.iter_quant_targets`); this only
    adds the divisibility check by the group size. The check is folded into core's predicate by
    {@link include}, so "the targets counted" and "the targets rounded" stay one set to the end.
    """

    counts: TargetCounts
    #: The module FQNs judged eligible (used to match against the tensor names of the distribution
    #: — the distribution carries torch's FQNs as they are, so this set is the source of truth of
    #: "eligible for today's i4").
    eligible: tuple[str, ...]
    #: The excluded module FQNs (the quantization axis is not divisible by the group size =
    #: `counts.group_size`).
    excluded: tuple[str, ...]
    excluded_elements: int
    #: Module type name → eligible counts (to report the target size per role).
    by_role: Mapping[str, TargetCounts]

    def include(self) -> Callable[[str], bool]:
        """Core's `include` (a predicate on module FQNs) — the complement of the excluded set."""
        excluded = frozenset(self.excluded)
        return lambda name: name not in excluded


def census_w4_targets(
    root: nn.Module, op_types: tuple[type[nn.Module], ...], group_size: int
) -> W4Census:
    """Counts the w4 targets of `root` per role and leaves out the weights whose quantization axis
    is not divisible by `group_size`.

    MUST: indivisible weights are **left out of the targets rather than dropping the whole
    configuration** (i4 does not create partial groups — ADR 0069 decision 2. Core fails loudly when
    they are not divisible, so without leaving them out the measurement cannot run at all). In the
    measurement (FN4, default g32) the only ones left out are convs whose receptive field is not a
    multiple of 32 (the separable convs of sdp, the 1x1 with 1 input channel, the 16-channel
    resblocks at the end of dec), under 0.1% of the eligible elements — the list and the counts are
    in `w4.census` of `report.json` (so a reader can detect "the targets silently shrank").
    """
    kinds = {name: type(module).__name__ for name, module in root.named_modules()}
    totals: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    eligible: list[str] = []
    excluded: list[str] = []
    excluded_elements = 0
    for fqn, weight, axis in iter_quant_targets(root, op_types):
        name = fqn[: -len(".weight")]
        if channel_rows(weight, axis).shape[-1] % group_size:
            excluded.append(name)
            excluded_elements += weight.numel()
            continue
        eligible.append(name)
        row = totals[kinds[name]]
        row[0] += 1
        row[1] += int(weight.shape[axis])
        row[2] += weight.numel()
    by_role = {
        kind: TargetCounts(*row, group_size=group_size) for kind, row in sorted(totals.items())
    }
    total = TargetCounts(0, 0, 0, group_size=group_size)
    for counts in by_role.values():
        total = total + counts
    return W4Census(
        counts=total,
        eligible=tuple(eligible),
        excluded=tuple(excluded),
        excluded_elements=excluded_elements,
        by_role=MappingProxyType(by_role),
    )


def net_g_census(model_dir: Path, group_size: int) -> Mapping[tuple[str, str], W4Census]:
    """Loads net_g once and takes the target sets per role × subgraph (keys are `(role, tag)`).

    It does not recount per configuration so that `include` has a single origin —
    {@link run_config} passes the predicate of the set returned here to core as is. The FQN space
    is the same when reloaded, so it applies as is to the net_g reloaded per configuration (the
    pristine reload that keeps rounding from stacking).
    """
    net_g, _ = export.load_net_g(model_dir)
    patch.apply_all_patches()
    export.ensure_dec_plain(net_g)
    roots = {"front": patch.Sbv2Front(net_g), "voice": patch.Sbv2Voice(net_g)}
    census = {
        (roles, tag): census_w4_targets(root, op_types, group_size)
        for roles, op_types in W4_ROLES.items()
        for tag, root in roots.items()
    }
    del net_g, roots
    gc.collect()
    return MappingProxyType(census)


# ---- Size estimate (**a projection by formula, not a measurement**) ---------------------------


@dataclass(frozen=True)
class SizeProjection:
    """The projected size of method × target set (the formula is kept verbatim in
    {@link W4Method.formula})."""

    counts: TargetCounts
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
        """The MiB of the same target set held in f32 (the projection's denominator — so the
        compression ratio can be read from the table)."""
        return self.counts.elements * 4 / 1024**2


#: Target sets of the size estimate → description. `net_g:linear` is a slot without a quality
#: measurement (the targets are under 1% of the whole, and **the absence of any gain is itself the
#: reason for the decision**), so it appears only here.
PROJECTION_TARGETS: Mapping[str, str] = MappingProxyType(
    {
        "net_g:all": "every role of net_g (no storage format — the conv family's scale cannot be"
        " handed to emit)",
        "net_g:linear": "the linears of net_g only (today's distributable form)",
        "bert:linear": "the linears of BERT (DeBERTa) only (today's distributable form, the main"
        " distribution candidate)",
    }
)


def build_projections(
    counts: Mapping[str, TargetCounts], methods: Mapping[str, W4Method]
) -> list[dict[str, Any]]:
    """The size estimate of every combination of target set × method.

    Which rows get a quality measurement is **looked up** from {@link CONFIGS} (writing it in a
    separate table would update only one of them when a configuration is added). The matching is by
    **method kind** (= the table's key); the `method` shown in a row is the method name with g
    baked in.
    """
    measured: dict[tuple[str, str], str] = {}
    for name, recipe in CONFIGS.items():
        if recipe.weight == "w4" and recipe.method is not None:
            measured[(f"net_g:{recipe.roles}", recipe.method)] = name
        elif recipe.bert_method is not None:
            measured[("bert:linear", recipe.bert_method)] = name
    rows: list[dict[str, Any]] = []
    for target, description in PROJECTION_TARGETS.items():
        for kind, method in methods.items():
            projection = SizeProjection(
                counts=counts[target],
                bits=method.projected_bits(counts[target]),
                formula=method.formula,
            )
            rows.append(
                {
                    "target": target,
                    "target_description": description,
                    "method": method.name,
                    "measured_by": measured.get((target, kind)),
                    "modules": projection.counts.modules,
                    "elements": projection.counts.elements,
                    "bits_per_weight": projection.bits_per_weight,
                    "projected_mib": projection.projected_mib,
                    "f32_mib": projection.f32_mib,
                    "formula": projection.formula,
                }
            )
    return rows


# ---- Reduction estimate against the distribution ----------------------------------------------

#: The tensor-name prefix that only DeBERTa's distribution graph carries (the 22-layer variant —
#: its layer indices are a subset of the 24 layers on the measurement side, so stripping this puts
#: it in the same FQN space as the rounding target set).
DIST_BERT_PREFIX = "model."

#: The extension of the distribution's containers (container-v1 §1).
CONTAINER_SUFFIX = ".krm"


def project_distribution(
    dist_dir: Path, linear_fqns: frozenset[str], group_size: int
) -> dict[str, Any]:
    """The "i8 → i4 (group size `group_size`)" reduction estimate against the **real files** of the
    distribution.

    The targets are **today's distributable form = the linear weight slots only** (i4 for conv /
    embedding has neither a storage format nor an execution path — ADR 0069 decision 5 /
    addendum 5). The distribution carries torch's FQNs as tensor names, so they can be looked up by
    matching against the rounding target set (`linear_fqns` = the same FQNs the census counted).
    Looking up by the shape "rank-2 I8" alone would mix in embeddings (`enc_p.emb` /
    `word_embeddings`), so the count of unmatched rank-2 I8 tensors is returned for the reader to
    audit.

    The formula (per tensor `[O,I]`, verbatim):

        i8 … `O·I` bytes + `4·O` bytes of the scale `[O,1]` (both are **the distribution's real
             bytes**)
        i4 … `O·I/2` bytes (nibble packing) + `4·O·(I/g)` bytes of the scale `[O,I/g]`

    The denominator is the total real bytes of the files in the distribution directory. What is
    under `shared/` (the one DeBERTa shared by the speakers) and the per-speaker net_g are reported
    separately — the former counts once, the latter once per speaker.

    `group_size` MUST be **the same g** as eligibility (the census that built `linear_fqns`) —
    passing a different g makes a mismatch of the form "a weight the census left out is divisible
    here" appear only in the numbers.
    """
    files = sorted(path for path in dist_dir.rglob("*") if path.is_file())
    total_bytes = sum(path.stat().st_size for path in files)
    groups: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    unmatched = 0
    for path in files:
        # Count each container once (start from part 0 and skip the rest of the sequence).
        if path.suffix != CONTAINER_SUFFIX or path != container_parts(path)[0]:
            continue
        values = ir_graph(path).get("values") or {}
        bucket = "shared（DeBERTa text_encoder）" if "shared" in path.parts else "話者ごとの net_g"
        for key, layout in sorted(read_layouts(path).items()):
            shape = (values.get(key) or {}).get("shape") or []
            if layout != "i8" or len(shape) != 2:
                continue
            if key not in linear_fqns and key.removeprefix(DIST_BERT_PREFIX) not in linear_fqns:
                unmatched += 1
                continue
            out_channels, in_axis = int(shape[0]), int(shape[1])
            if in_axis % group_size:
                raise AssertionError(
                    f"{path}: '{key}' has in axis {in_axis}, not divisible by group {group_size}"
                )
            row = groups[bucket]
            row[0] += 1
            # i8 is "O·I bytes of values + 4·O bytes of per-channel scale" (exactly the container
            # payload).
            row[1] += out_channels * in_axis + 4 * out_channels
            row[2] += out_channels * in_axis // 2 + 4 * out_channels * (in_axis // group_size)
    delta = sum(row[1] - row[2] for row in groups.values())
    return {
        "root": str(dist_dir),
        "group_size": group_size,
        "files": len(files),
        "total_bytes": total_bytes,
        "groups": {
            name: {
                "tensors": row[0],
                "current_bytes": row[1],
                "projected_bytes": row[2],
                "delta_bytes": row[1] - row[2],
                "shrink_of_total": (row[1] - row[2]) / total_bytes,
            }
            for name, row in sorted(groups.items())
        },
        "rank2_i8_not_linear": unmatched,
        "delta_bytes": delta,
        "shrink_of_total": delta / total_bytes if total_bytes else 0.0,
        "formula": f"i8 = O·I + 4·O bytes / i4 = O·I/2 + 4·O·(I/{group_size}) bytes"
        " (the linear weight slots only; the denominator is the total real bytes of the"
        " distribution's files)",
    }


# ---- Running one configuration ------------------------------------------------------------


@dataclass
class ChainInputs:
    """The inputs shared by every configuration (built once, including the f32 BERT features)."""

    tensors: dict[str, torch.Tensor]
    meta: dict[str, Any]
    bert_feature: torch.Tensor
    #: BERT's w4 target set (linear only) — the source of truth for the size estimate and the
    #: rounding's `include`.
    bert_census: W4Census
    g: torch.Tensor
    style_vec: torch.Tensor


def load_bert() -> Any:
    """Loads the reference DeBERTa (**all 24 layers, not truncated**)."""
    from transformers import DebertaV2Model

    bert = DebertaV2Model.from_pretrained(
        demo.BERT_REPO, dtype=torch.float32, attn_implementation="eager"
    )
    bert.eval()
    return bert


def bert_feature_of(bert: nn.Module, tensors: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """Runs BERT and returns the features tiled out to the phonemes (the common path regardless of
    rounding)."""
    with torch.no_grad():
        hidden_states = bert(
            input_ids=tensors["input_ids"].to(torch.int64),
            attention_mask=tensors["attention_mask"].to(torch.int64),
            output_hidden_states=True,
        ).hidden_states
    # MUST: the reference is the **untruncated full 24-layer** model, so the position uses the
    # constant directly (`meta["bertHiddenFromEnd"]` is the position on the distribution graph's
    # side — how sbv2.demo uses its 2 constants).
    hidden = hidden_states[-demo.BERT_HIDDEN_FROM_END][0]
    word2ph = tensors["word2ph"].reshape(-1).tolist()
    return demo.tile_bert(hidden, word2ph).unsqueeze(0)


def prepare_inputs(dump_path: Path, assets_path: Path, group_size: int) -> ChainInputs:
    """Reads the dump / assets and runs **the f32 BERT** once to build the features.

    Sharing the f32 features between configurations is the contract itself (the existing 5
    configurations + net_g's w4 keep BERT=f32); it only folds a computation that gives the same
    numbers every time into one run. Configurations that vary BERT build theirs separately **from
    the plain weights** in {@link quantized_bert_feature}. The tokenization check is the same one as
    `sbv2.demo.run_reference` (if it disagrees with the Deno-side port, it fails before the
    waveform).
    """
    meta = demo.dump_metadata(dump_path)
    tensors = load_file(str(dump_path))
    assets = load_file(str(assets_path))

    tokenizer = demo.load_bert_tokenizer()
    expected_ids = tokenizer(meta["bertText"])["input_ids"]
    dumped_ids = tensors["input_ids"].reshape(-1).tolist()
    if expected_ids != dumped_ids:
        raise AssertionError(
            "DeBERTa tokenization disagrees with the dump"
            f" (python={expected_ids} / dump={dumped_ids})"
        )

    bert = load_bert()
    # The target set is taken from the f32 model (rounding does not move FQNs or shapes, so no
    # reload is needed).
    bert_census = census_w4_targets(bert, W4_ROLES["linear"], group_size)
    bert_feature = bert_feature_of(bert, tensors)
    del bert
    gc.collect()

    return ChainInputs(
        tensors=tensors,
        meta=meta,
        bert_feature=bert_feature,
        bert_census=bert_census,
        g=assets["g"].to(torch.float32),
        style_vec=assets["style_vec"].to(torch.float32),
    )


def quantized_bert_feature(
    kind: str, inputs: ChainInputs, methods: Mapping[str, W4Method]
) -> tuple[torch.Tensor, str]:
    """Rounds **only BERT's linear weights** with a w4 method, then takes the features
    (`(features, counts)`).

    MUST: the model is **reloaded from the plain weights** (methods do not stack) — the same
    pristine approach as net_g calling `load_net_g` again per configuration, with no mutable
    save/restore state. The targets are linear only because that is the only i4 execution path
    (ADR 0069 decision 5). BERT's embeddings (`word_embeddings` / `rel_embeddings`) and its one conv
    stay f32.
    """
    bert = load_bert()
    report = methods[kind].apply(bert, W4_ROLES["linear"], inputs.bert_census.include())
    feature = bert_feature_of(bert, inputs.tensors)
    del bert
    gc.collect()
    return feature, report.describe()


# ---- Driving calibrated rounding (Catcher + stage decomposition) -------------------------------

#: The in-model FQN prefix of the calibration stages (`DebertaV2Model.encoder` = `DebertaV2Encoder`
#: → `.layer`). {@link EncoderStage} names its child by the layer index so that the local FQN in a
#: stage becomes `<layer index>.attention.self.query_proj.weight`, so the prefix up to here is
#: enough (to put the scale ledger's keys in the same **in-model FQN** space as
#: {@link karume.quantize.Int4Report} — the contract of core's `StageSpec`).
BERT_STAGE_PREFIX = "encoder.layer"

#: The keyword name {@link EncoderStage.forward} receives for the ConvLayer. Apart from the 4-D
#: mask passed to the layers, `DebertaV2Encoder` passes **the plain 2-D mask** to the ConvLayer.
INPUT_MASK_KWARG = "input_mask"


class _FirstStageReached(Exception):  # noqa: N818 — a signal to stop, not a failure, so not "Error"
    """The signal that the first stage's inputs are complete (a sentinel that stops the calibration
    forward)."""


class EncoderStage(nn.Module):
    """A stage wrapper that wraps one encoder layer into a form that "takes hidden as a positional
    argument".

    There are 2 reasons to wrap, both coming from how `DebertaV2Encoder.forward` (transformers
    5.14.1) calls its layers:

    1. It passes the mask to the layer **positionally** (`layer_module(next_kv, attention_mask,
       …)`). `calibrate_stages` drives the next stage with "the chosen output as **the only
       positional argument**", so unless the mask is moved to a keyword it fails from the second
       stage on.
    2. **Only the first layer** has a ConvLayer on its output (`i == 0` in the loop — it mixes the
       stage's input hidden back in as a residual). Dropping this in the stage decomposition makes
       the later stages see a hidden different from the real one, and calibration chooses rounding
       targets from different activations.

    The layer kwargs (`relative_pos` / `rel_embeddings` / `query_states` / `output_attentions`) are
    built once **before** the loop and are **the same for every layer** (confirmed by measurement —
    there is no per-layer_type switch as in Gemma3), so they fit `calibrate_stages`' default
    (kwargs unchanged between stages) as is.

    The child module is named by **the layer index** so that the local FQN in a stage
    (`0.attention.self.query_proj.weight`) turns back into the in-model FQN just by adding
    {@link BERT_STAGE_PREFIX}. The ConvLayer is registered as a child too, but it holds no
    `nn.Linear` (`Conv1d` + `LayerNorm`), so the target set of the scan (`iter_quant_targets`
    restricted to `nn.Linear`) does not move.
    """

    def __init__(self, index: int, layer: nn.Module, conv: nn.Module | None) -> None:
        super().__init__()
        self.child = str(index)
        self.add_module(self.child, layer)
        self.conv = conv

    def forward(
        self, hidden: torch.Tensor, *, input_mask: torch.Tensor, **layer_kwargs: Any
    ) -> torch.Tensor:
        output = getattr(self, self.child)(hidden, **layer_kwargs)[0]
        if self.conv is None:
            return output
        return self.conv(hidden, output, input_mask)


def bert_stages(bert: nn.Module) -> tuple[StageSpec, ...]:
    """Returns the encoder layers in execution order as `(in-model FQN prefix, stage)`.

    **Builds only up to the layer the features are taken from** — the reference model has all 24
    layers untruncated, but the features are taken from `hidden_states[-demo.BERT_HIDDEN_FROM_END]`
    (= the output of layer 22), so rounding later layers does not affect the output by a single bit
    (the same boundary as the distribution graph being the 22-layer variant — ADR 0045 decisions 1/2
    "separate the position constants of the reference side and the distribution-graph side"; this is
    the reference-side position). So the calibration target set is **a subset** of the existing
    `bert-w4-*` census (all 24 layers) and **matches the linear set of the distribution**
    ({@link CALIB_TARGET}).
    """
    encoder = bert.encoder
    if encoder.conv is None:
        raise AssertionError(
            "the DeBERTa encoder has no ConvLayer (a model without the first layer's residual"
            " mixing — the model's structure differs from what the script assumes)"
        )
    layers = list(encoder.layer)
    count = len(layers) - (demo.BERT_HIDDEN_FROM_END - 1)
    if count < 1:
        raise AssertionError(
            f"the encoder has only {len(layers)} layers"
            f" (the features are taken from hidden number {demo.BERT_HIDDEN_FROM_END} from the end)"
        )
    return tuple(
        (BERT_STAGE_PREFIX, EncoderStage(index, layer, encoder.conv if index == 0 else None))
        for index, layer in enumerate(layers[:count])
    )


def calib_targets(stages: Sequence[StageSpec]) -> tuple[dict[str, torch.Tensor], TargetCounts]:
    """Returns the quantization targets inside the stages as weights keyed by fqn plus counts (the
    calibration's **scan** = the reference of the check).

    Target selection is shared with core's `iter_quant_targets` (the same as
    {@link census_w4_targets} — a copied implementation would let "the targets counted" and "the
    targets rounded" silently diverge). {@link assert_calib_covers_scan} matches the count taken
    here against the count calibration actually rounded.

    Unlike the net_g census, indivisible weights are **not left out but fail loudly** — DeBERTa's
    linears have only 2 quantization axis lengths, 1024 / 4096, and exclusion is only needed on the
    net_g side, which has convs with odd receptive fields. Adding a way to leave weights out here
    would let "the targets silently shrank" grow on the calibration side too.
    """
    weights: dict[str, torch.Tensor] = {}
    channels = 0
    elements = 0
    for prefix, stage in stages:
        for local, weight, axis in iter_quant_targets(stage, W4_ROLES["linear"]):
            fqn = f"{prefix}.{local}"
            span = int(channel_rows(weight, axis).shape[-1])
            if span % CALIB_GROUP_SIZE:
                raise AssertionError(
                    f"'{fqn}': 量子化軸 {span} が group {CALIB_GROUP_SIZE} で割り切れない"
                    "（i4 は端数 group を作らない — ADR 0069 決定 2）"
                )
            weights[fqn] = weight
            channels += int(weight.shape[axis])
            elements += int(weight.numel())
    if not weights:
        raise AssertionError(
            "the encoder stages contain no nn.Linear (the model's structure differs from what is"
            " assumed)"
        )
    return weights, TargetCounts(
        modules=len(weights), channels=channels, elements=elements, group_size=CALIB_GROUP_SIZE
    )


def calib_corpus(limit: int | None) -> tuple[str, ...]:
    """The sentences used for calibration (`limit` caps it to **the first N sentences** — `None` is
    all 48).

    They are taken from the start because {@link deberta.calib_texts.CALIB_TEXTS} interleaves
    read-aloud style / questions / number reading (a reduced run keeps the mix of roles).
    """
    if limit is not None and limit < 1:
        raise ValueError(f"校正文数の上限は 1 以上（実測 {limit}）")
    return CALIB_TEXTS if limit is None else CALIB_TEXTS[:limit]


def assert_calib_disjoint(texts: Sequence[str], evaluated: Sequence[str]) -> None:
    """Checks that the calibration corpus does not overlap the evaluated text, **not even as a
    substring**.

    MUST: fail loudly. With an overlap, SNR / LSD measure "whether it could reproduce a sentence it
    saw in calibration" — measuring leakage rather than the quality of calibration. The source of
    truth of the evaluated text is the dump's `meta`, so the corpus keeps no copy and the match is
    done here (a copy would leave one side stale when the dump is re-recorded).
    """
    hits = sorted(
        {text for text in texts for body in evaluated if body and (body in text or text in body)}
    )
    if hits:
        raise AssertionError(
            f"校正コーパスが評価文と重なっている: {hits[:3]}（評価文 {list(evaluated)}）"
        )


def calib_inputs(texts: Sequence[str]) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    """Turns the calibration corpus into `[1,T]` `(input_ids, attention_mask)` through **the
    existing tokenizer path** ({@link sbv2.demo.load_bert_tokenizer}).

    The tokenizer is not copied for the same reason as the dump check of `prepare_inputs` — taking
    another path would silently give "the activations seen in calibration" and "the activations
    flowing in evaluation" different tokenizations.
    """
    tokenizer = demo.load_bert_tokenizer()
    encoded = [tokenizer(text) for text in texts]
    return tuple(
        (
            torch.tensor([entry["input_ids"]], dtype=torch.int64),
            torch.tensor([entry["attention_mask"]], dtype=torch.int64),
        )
        for entry in encoded
    )


def capture_stage_batches(
    bert: nn.Module, inputs: Sequence[tuple[torch.Tensor, torch.Tensor]]
) -> tuple[StageBatch, ...]:
    """Captures the hidden state and the accompanying arguments going into the first stage with a
    forward_pre_hook (the Catcher).

    The calibration input is "the hidden state after the embeddings, plus the 4-D mask / relative
    positions / `rel_embeddings` the encoder built"; rebuilding them by hand silently diverges from
    transformers. Capturing **the call into the first layer itself** takes the same input without
    copying the earlier stages.

    The 2-D mask the ConvLayer needs is not passed to the layers, so one more is captured from
    **the ConvLayer's call** (for the same reason: copying the `attention_mask.dim() <= 2` branch
    would let one copy drift from the spec). The sentinel stops at the ConvLayer — up to that point
    only the first layer runs.

    MUST: if the forward runs to completion without everything captured, fail loudly — it signals
    that the stage spelling disagrees with the model's structure, and silently going on would only
    produce a "zero calibration inputs" diagnostic on core's side.
    """
    encoder = bert.encoder
    captured: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    masks: list[torch.Tensor] = []

    def catch_layer(_module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        captured.append((args, dict(kwargs)))

    def catch_conv(_module: nn.Module, args: tuple[Any, ...]) -> None:
        masks.append(args[2])
        raise _FirstStageReached

    handles = [
        encoder.layer[0].register_forward_pre_hook(catch_layer, with_kwargs=True),
        encoder.conv.register_forward_pre_hook(catch_conv),
    ]
    batches: list[StageBatch] = []
    try:
        for index, (ids, mask) in enumerate(inputs):
            captured.clear()
            masks.clear()
            try:
                with torch.no_grad():
                    bert(input_ids=ids, attention_mask=mask)
            except _FirstStageReached:
                pass
            if len(captured) != 1 or len(masks) != 1:
                raise AssertionError(
                    f"校正入力 {index} で先頭 stage の入力が揃わなかった"
                    f"（層 {len(captured)} 件 / ConvLayer {len(masks)} 件）"
                    "— stage の綴りが模型の構成と食い違っている"
                )
            args, kwargs = captured[0]
            if len(args) != 2:
                raise AssertionError(
                    f"the first layer was called with {len(args)} positional arguments (2, hidden"
                    " and mask, are expected — the way transformers calls it has changed)"
                )
            hidden, attention_mask = args
            batches.append(
                (
                    (hidden.detach(),),
                    {"attention_mask": attention_mask, INPUT_MASK_KWARG: masks[0], **kwargs},
                )
            )
    finally:
        for handle in handles:
            handle.remove()
    return tuple(batches)


@dataclass(frozen=True)
class CalibRig:
    """The scaffolding of one calibrated configuration (stage sequence, scan, inputs to the first
    stage).

    Unlike the minicpm5 / EG rigs, it is **rebuilt per configuration** — this script reloads BERT
    from the plain weights per configuration so that methods do not stack (the same pristine
    approach as {@link quantized_bert_feature}), so the stages and the first stage's inputs are tied
    to that model.
    """

    stages: tuple[StageSpec, ...]
    scan: Mapping[str, torch.Tensor]
    counts: TargetCounts
    batches: tuple[StageBatch, ...]


def build_calib_rig(bert: nn.Module, meta: Mapping[str, Any], limit: int | None) -> CalibRig:
    """Builds the calibration scaffolding (stage decomposition → scan → separation check against
    the evaluated text → tokenize → Catcher)."""
    stages = bert_stages(bert)
    scan, counts = calib_targets(stages)
    texts = calib_corpus(limit)
    assert_calib_disjoint(texts, (meta["text"], meta["bertText"]))
    return CalibRig(
        stages=stages,
        scan=scan,
        counts=counts,
        batches=capture_stage_batches(bert, calib_inputs(texts)),
    )


def apply_calib(config: CalibConfig, rig: CalibRig) -> CalibReport:
    """Applies one calibrated rounding to the model in place (core drives the stages in order)."""
    return calibrate_stages(rig.stages, rig.batches, method=config.method, spec=config.grid)


def assert_calib_covers_scan(
    report: CalibReport, scan: Mapping[str, torch.Tensor], name: str
) -> None:
    """Checks that the layers calibration rounded match the stage scan **exactly, with nothing
    missing or extra**.

    MUST: fail loudly. If the stage spelling or the target types change and part of the encoder is
    no longer calibrated, the row still stays in the table, and SNR / LSD come out **on the better
    side** by the amount left unrounded (a pass-through cannot be read from the numbers). It blocks
    vacuous passes in the same role as the existing 6 gates ({@link run_gates}).
    """
    rounded = {layer.fqn for layer in report.layers}
    missing = sorted(set(scan) - rounded)
    extra = sorted(rounded - set(scan))
    if missing or extra or report.modules != len(scan):
        raise AssertionError(
            f"[{name}] 校正が丸めた {report.modules} 本が走査の {len(scan)} 本と一致しない"
            f"（丸め漏れ {missing[:3]} / 走査に無い {extra[:3]}）"
        )


def calibrated_bert_feature(
    name: str, inputs: ChainInputs, limit: int | None
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Rounds BERT's **in-encoder linears** with calibrated rounding, then takes the features
    (`(features, report)`).

    MUST: the model is **reloaded from the plain weights** (methods do not stack — the same as
    {@link quantized_bert_feature}). The rounding itself is shared with core's `calibrate_stages`;
    the script holds only the stage decomposition and how the calibration inputs are taken.
    """
    config = CALIB_CONFIGS[name]
    bert = load_bert()
    rig = build_calib_rig(bert, inputs.meta, limit)
    report = apply_calib(config, rig)
    assert_calib_covers_scan(report, rig.scan, name)
    feature = bert_feature_of(bert, inputs.tensors)
    projection = SizeProjection(
        counts=rig.counts, bits=config.projected_bits(rig.counts), formula=config.formula
    )
    rows = [layer.tokens for layer in report.layers]
    detail = {
        "quant": report.describe(),
        "method": config.label,
        "target": CALIB_TARGET,
        "stages": report.stages,
        "calib_texts": len(rig.batches),
        # Records whether this was a reduced run with `--calib-limit` (`None` = all) — kept in
        # the same file as the metrics so that reduced numbers are not read as full ones.
        "calib_limit": limit,
        # The range of input **rows** each layer saw (the weight of `H = Σ XᵀX`). `query_proj` /
        # `key_proj` also consume share_att_key's relative position embeddings (512 rows per
        # forward), so they always come out above the plain token row count (= the lower side) —
        # writing one number would mislead the reader.
        "calib_rows": {"min": min(rows), "max": max(rows)},
        "size": {
            "modules": projection.counts.modules,
            "elements": projection.counts.elements,
            "bits_per_weight": projection.bits_per_weight,
            "projected_mib": projection.projected_mib,
            "f32_mib": projection.f32_mib,
            "formula": projection.formula,
        },
    }
    del bert, rig
    gc.collect()
    return feature, detail


def bert_variant(recipe: Recipe, methods: Mapping[str, W4Method]) -> str | None:
    """The cache key of the BERT features = **the method name shown in the table** (`None` = stays
    f32).

    The plain methods ({@link w4_methods}) and the calibrated ones ({@link CALIB_CONFIGS}) do not
    share a namespace, so one key is enough. A Recipe with both is a mistake in the configuration
    and is rejected.

    The key is **the name** rather than the kind because this key is written as is as the field
    name of `w4.bert_quant` and `gates.bert_quant_effective` — if 2 runs with different g had the
    same field name, the reader would confuse them.
    """
    if recipe.bert_method is not None and recipe.bert_calib is not None:
        raise AssertionError("BERT の丸めは素の方式か校正付きのどちらか一方")
    if recipe.bert_method is not None:
        return methods[recipe.bert_method].name
    return recipe.bert_calib


def bert_feature_for(
    recipe: Recipe, inputs: ChainInputs, calib_limit: int | None, methods: Mapping[str, W4Method]
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Builds the features matching the configuration's BERT contract and the report of their
    rounding (not called for configurations that keep f32)."""
    if recipe.bert_calib is not None:
        return calibrated_bert_feature(recipe.bert_calib, inputs, calib_limit)
    if recipe.bert_method is None:
        raise AssertionError("trying to build features for a configuration without BERT rounding")
    feature, described = quantized_bert_feature(recipe.bert_method, inputs, methods)
    return feature, {
        "quant": described,
        "method": methods[recipe.bert_method].name,
        "target": "bert:linear",
    }


def run_config(
    name: str,
    model_dir: Path,
    inputs: ChainInputs,
    *,
    bert_feature: torch.Tensor,
    census: Mapping[tuple[str, str], W4Census],
    methods: Mapping[str, W4Method],
    group_size: int,
    inject: str | None,
) -> dict[str, Any]:
    """Runs the chain of one configuration and returns the waveform and the diagnostics.

    net_g is reloaded per configuration (= the pristine approach that keeps rounding from
    stacking). `bert_feature` is passed in by the caller as the features matching the
    configuration's BERT contract (f32 or rounded with w4).

    MUST (order — the `quantize` module and ADR 0018/0019/0027): weight rounding is applied
    **after `apply_all_patches` and `ensure_dec_plain`**. Rounding before remove_weight_norm rounds
    `weight_g`/`weight_v`, and the effective weights built from them do not sit on the f16/i8 grid
    (for i8 the discarded elements affect the amax and shift the whole per-channel scale; for i4
    the same happens per group).
    """
    started = time.perf_counter()
    recipe = RECIPES[name]
    tensors = inputs.tensors
    knobs = inputs.meta["knobs"]

    net_g, hps = export.load_net_g(model_dir)
    patch.apply_all_patches()
    export.ensure_dec_plain(net_g)
    front = patch.Sbv2Front(net_g)
    voice = patch.Sbv2Voice(net_g)
    # front (enc_p / dp / sdp) and voice (flow / dec) are disjoint module sets.
    subgraphs = (("front", front), ("voice", voice))
    scoped = [(tag, module) for tag, module in subgraphs if tag in recipe.scope]

    # --- Weight rounding -----------------------------------------------------------
    weight_report: dict[str, str] = {}
    w4_targets: dict[str, dict[str, int]] = {}
    for tag, module in scoped:
        if recipe.weight == "f16":
            weight_report[tag] = round_weights_to_f16(module).describe()
        elif recipe.weight == "i8":
            weight_report[tag] = fake_quant_int8(module).describe()
        elif recipe.weight == "w4":
            if recipe.method is None:
                raise AssertionError(f"{name}: a w4 configuration without a method")
            scoped_census = census[(recipe.roles, tag)]
            report = methods[recipe.method].apply(
                module, W4_ROLES[recipe.roles], scoped_census.include()
            )
            weight_report[tag] = report.describe()
            w4_targets[tag] = {"modules": report.modules, "elements": report.elements}

    # --- Activation rounding (replaces the inputs of the target ops) ----------------------------
    quant: ActQuant | None = None
    attached = 0
    if recipe.act is not None:
        quant = ActQuant(roots=tuple(scoped), mode=recipe.act, disable=(inject == "drop-act-quant"))
        attached = quant.attach()

    # --- front ---------------------------------------------------------------
    x_mask = tensors["x_mask"].to(torch.float32)
    with torch.no_grad():
        logw_sdp, logw_dp, m_p, logs_p = front(
            tensors["x"].to(torch.int64),
            x_mask,
            tensors["tone"].to(torch.int64),
            tensors["language"].to(torch.int64),
            bert_feature,
            inputs.style_vec,
            inputs.g,
            tensors["z_noise"].to(torch.float32),
        )

    # --- Host glue (same formula as the demo main.ts / demo.run_reference) -----------
    logw = logw_sdp * knobs["sdpRatio"] + logw_dp * (1.0 - knobs["sdpRatio"])
    w = torch.exp(logw) * x_mask * knobs["lengthScale"]
    own_ceil = torch.ceil(w).to(torch.int64).reshape(-1)
    dumped_ceil = tensors["w_ceil"].reshape(-1).to(torch.int64)
    # MUST: the expansion uses **the dump's w_ceil**. Using each configuration's own w_ceil splits
    # the waveform lengths and SNR cannot be defined (zp_noise's shape is also fixed by the dump's
    # Ty).
    expand_idx = torch.repeat_interleave(torch.arange(dumped_ceil.shape[0]), dumped_ceil)
    total_frames = int(expand_idx.shape[0])
    zp_noise = tensors["zp_noise"].to(torch.float32)
    z_p = (
        m_p[:, :, expand_idx] + zp_noise * torch.exp(logs_p[:, :, expand_idx]) * knobs["noiseScale"]
    )
    y_mask = torch.ones(1, 1, total_frames)
    idx_k, valid = patch.build_relattn_tables(total_frames, export.EXPECTED_WINDOW_SIZE)

    # --- voice (flow + dec fused) --------------------------------------------
    with torch.no_grad():
        audio = voice(z_p, y_mask, inputs.g, idx_k, valid).reshape(-1)

    stats: dict[str, ActStat] = {}
    skipped = 0
    if quant is not None:
        quant.detach()
        stats = quant.stats
        skipped = quant.skipped_calls
    # Restoring the replacements and the hooks is directly about leaks between configurations (a
    # leftover op replacement silently quantizes the next configuration). `is` checks that the ops
    # are back to **the plain objects** — the only mutable state across configurations, so it is
    # checked every time, even for the configurations without rounding (f32 / f16 / w8).
    leaked = [op for op in ACT_ROW_AXIS if getattr(torch.nn.functional, op) is not PRISTINE_OPS[op]]
    if leaked:
        raise AssertionError(
            f"{name}: an activation-quantization op replacement is left over: {leaked}"
        )

    diagnostics = {
        "config": name,
        "weight_mode": recipe.weight,
        "act_mode": recipe.act,
        "scope": list(recipe.scope),
        "w4": {
            "method": methods[recipe.method].name if recipe.method is not None else None,
            "roles": recipe.roles if recipe.weight == "w4" else None,
            "group_size": group_size if recipe.weight == "w4" else None,
            "targets": w4_targets,
        },
        "bert_method": (
            methods[recipe.bert_method].name if recipe.bert_method is not None else None
        ),
        "bert_calib": recipe.bert_calib,
        "version": hps.version,
        "frames": total_frames,
        "samples": int(audio.shape[0]),
        "seconds": round(audio.shape[0] / inputs.meta["samplingRate"], 3),
        "weight_quant": weight_report,
        "act_quant": {
            "patched_ops": attached,
            "call_sites": len(stats),
            "calls": sum(stat.calls for stat in stats.values()),
            "calls_by_op": {
                op: sum(stat.calls for stat in stats.values() if stat.op == op)
                for op in ACT_ROW_AXIS
            },
            "skipped_calls": skipped,
            "f16_saturated": sum(stat.saturated for stat in stats.values()),
        },
        "w_ceil": {
            "exact_vs_dump": bool(torch.equal(own_ceil, dumped_ceil)),
            "own_frames": int(own_ceil.sum()),
            "dump_frames": int(dumped_ceil.sum()),
            "max_abs_diff": int((own_ceil - dumped_ceil).abs().max()),
        },
        "elapsed": round(time.perf_counter() - started, 1),
    }
    front_outputs = {
        "logw": logw.detach().clone(),
        "m_p": m_p.detach().clone(),
        "logs_p": logs_p.detach().clone(),
    }

    del net_g, front, voice, z_p, idx_k, valid
    gc.collect()
    return {
        "audio": audio.detach().clone(),
        "front": front_outputs,
        "stats": stats,
        "diagnostics": diagnostics,
    }


# ---- Report ---------------------------------------------------------------


def act_group_table(stats: dict[str, ActStat], limit: int) -> list[dict[str, Any]]:
    """The activation quantization error per layer group (only the top rows by descending
    relRMS)."""
    groups: dict[str, list[ActStat]] = defaultdict(list)
    for name, stat in stats.items():
        groups[layer_group(name)].append(stat)
    rows = [
        {
            "group": name,
            "op": members[0].op,
            "call_sites": len(members),
            "rel_rms_max": max(stat.rel_rms() for stat in members),
            "rel_rms_mean": sum(stat.rel_rms() for stat in members) / len(members),
        }
        for name, members in groups.items()
    ]
    rows.sort(key=lambda row: -row["rel_rms_max"])
    return rows[:limit]


def build_report(
    args: argparse.Namespace,
    inputs: ChainInputs,
    results: dict[str, dict[str, Any]],
    wavs: dict[str, Path],
    gates: dict[str, Any],
    w4: dict[str, Any],
) -> dict[str, Any]:
    base = results["f32"]["audio"]

    def entry_for(name: str) -> dict[str, Any]:
        result = results[name]
        audio = result["audio"]
        entry = dict(result["diagnostics"])
        entry["wav"] = str(wavs[name])
        entry["wav_sha256"] = sha256(wavs[name].read_bytes())
        entry["maxAbs"] = float(audio.abs().max())
        entry["snr_db_vs_f32"] = snr_db(audio, base)
        entry["lsd_db_vs_f32"] = log_spectral_distance(audio, base)
        entry["rel_rms_vs_f32"] = rel_rms(audio, base)
        entry["maxAbs_diff_vs_f32"] = float((audio - base).abs().max())
        entry["bit_equal_to_f32"] = bool(torch.equal(audio, base))
        entry["front_snr_db_vs_f32"] = {
            key: snr_db(result["front"][key], results["f32"]["front"][key])
            for key in ("logw", "m_p", "logs_p")
        }
        if result["stats"]:
            entry["act_groups"] = act_group_table(result["stats"], args.top)
        return entry

    configs = [entry_for(name) for name in CONFIGS if name in results]
    diagnostics = [entry_for(name) for name in DIAGNOSTICS if name in results]
    # MUST: BERT's rounding is derived from **the configurations that actually ran in this run**
    # (writing it as a constant leaves only the summary line stale on the day a configuration is
    # added — a later reader of a saved report.json would take it as "BERT was measured in f32
    # throughout"). `bert_quant` can also be empty when narrowed with `--only` and the like.
    bert_configs = sorted(w4.get("bert_quant") or {})
    bert_method = (
        "configurations that vary net_g keep BERT f32 / BERT was varied by"
        f" {', '.join(bert_configs)}"
        " (breakdown in `w4.bert_quant`)"
        if bert_configs
        else "BERT was not varied in this run (every configuration keeps f32 — only the generator"
        " network was varied)"
    )

    return {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "script": "tools/export-recipes/sbv2/measure_quant.py",
        "torch": torch.__version__,
        "dump": str(args.dump),
        "text": inputs.meta["text"],
        "knobs": inputs.meta["knobs"],
        "sampling_rate": inputs.meta["samplingRate"],
        "method": {
            "bert": bert_method,
            "act_target_ops": sorted(ACT_ROW_AXIS),
            "act_apply_point": "the inputs of the target ops (the torch.nn.functional ops"
            " themselves are replaced — a module forward_pre_hook misses the FFN patch's direct"
            " calls)",
            "act_row_axis": dict(ACT_ROW_AXIS),
            "act_i8_formula": "s = clamp(rowmax|x|/127, f32 tiny) / q = clamp(round(x/s), ±127)"
            " / x̂ = q·s (reuses act_quant.quantize_rows with an axis transpose)",
            "time_grid": "every configuration expands with the dump's w_ceil (aligns the waveform"
            " lengths so SNR can be defined)",
            "metrics": "SNR is the waveform residual (sensitive to phase shifts) / LSD is the"
            " distance of the log magnitude spectra"
            f" (n_fft={STFT_N_FFT} hop={STFT_HOP}, floor −100dB, insensitive to phase shifts)."
            " Read both",
            "diagnostics": "the front / voice orthogonal decomposition of the degradation (the"
            " counterpart of the earlier experiment's recorded voice SNR is `w8-voice-only`)",
            "w4": f"the methods use g={args.w4_group_size} (`--w4-group-size`, default"
            f" {DEFAULT_GROUP_SIZE} — the method axis of ADR 0069 addendum 5 / the g axis of wave"
            " J-3). net_g on every role, BERT on its linears only; the rounding is shared with core"
            " (karume.quantize / karume.quant_methods). Sizes are **a projection by formula**, not"
            " a measurement, and only RTN (`i4`) has a storage format",
            "bert_calib": "calibrated rounding (GPTQ) is shared with core (karume.quant_calib), and"
            " **the grid is byte-for-byte the same as the method grid** (what changes is which"
            " level each weight moves to within the same grid). The calibration inputs are the 48"
            " sentences of deberta/calib_texts.py taken through the existing DeBERTa tokenizer"
            " path, separated from the evaluated text down to substrings. The stages go **up to"
            " the layer of the encoder the features are taken from**, and"
            f" the target is `{CALIB_TARGET}` (a subset of the `bert:linear` census)",
        },
        "gates": gates,
        "w4": w4,
        "configs": configs,
        "diagnostics": diagnostics,
    }


def format_table(entries: list[dict[str, Any]]) -> str:
    header = [
        "config",
        "SNR vs f32 (dB)",
        "LSD (dB)",
        "relRMS",
        "maxAbs",
        "max|Δ|",
        "own w_ceil",
        "s",
    ]
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    for entry in entries:
        snr = entry["snr_db_vs_f32"]
        ceil = entry["w_ceil"]
        frames = f"{ceil['own_frames']} / {ceil['dump_frames']}"
        lines.append(
            "| "
            + " | ".join(
                [
                    entry["config"],
                    "inf (baseline)" if math.isinf(snr) else f"{snr:.2f}",
                    f"{entry['lsd_db_vs_f32']:.2f}",
                    f"{entry['rel_rms_vs_f32']:.4e}",
                    f"{entry['maxAbs']:.6f}",
                    f"{entry['maxAbs_diff_vs_f32']:.4e}",
                    ("match" if ceil["exact_vs_dump"] else "split") + f"（{frames} frames）",
                    f"{entry['elapsed']:.0f}",
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def format_projection_table(rows: list[dict[str, Any]]) -> str:
    header = [
        "target set",
        "method",
        "tensors",
        "elements",
        "bpw",
        "projected MiB",
        "f32 MiB",
        "quality measurement",
        "formula",
    ]
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["target"],
                    row["method"],
                    str(row["modules"]),
                    f"{row['elements']:,}",
                    f"{row['bits_per_weight']:.3f}",
                    f"{row['projected_mib']:.1f}",
                    f"{row['f32_mib']:.1f}",
                    row["measured_by"] or "—",
                    row["formula"],
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def format_distribution(projection: dict[str, Any]) -> str:
    mib = 1024**2
    lines = [
        f"Against the distribution {projection['root']} ({projection['files']} real files /"
        f" {projection['total_bytes'] / mib:.1f} MiB), the reduction estimate of"
        f" i8 → i4 g{projection['group_size']}:",
        f"  formula: {projection['formula']}",
    ]
    for name, group in projection["groups"].items():
        lines.append(
            f"  {name}: {group['tensors']} tensors"
            f" {group['current_bytes'] / mib:.1f} MiB → {group['projected_bytes'] / mib:.1f} MiB"
            f" (−{group['delta_bytes'] / mib:.1f} MiB = "
            f"−{group['shrink_of_total'] * 100:.2f}% of the whole distribution)"
        )
    lines.append(
        f"  total: −{projection['delta_bytes'] / mib:.1f} MiB ="
        f" −{projection['shrink_of_total'] * 100:.2f}% of the whole distribution"
        f" ({projection['rank2_i8_not_linear']} rank-2 I8 tensors that are not linear are out of"
        " scope)"
    )
    return "\n".join(lines)


# ---- Verification gates -------------------------------------------------------------


def run_gates(
    results: dict[str, dict[str, Any]],
    wavs: dict[str, Path],
    reference_wav: Path,
    census: Mapping[tuple[str, str], W4Census],
    bert_reports: Mapping[str, dict[str, Any]],
) -> dict[str, Any]:
    """Detectors of vacuous passes and configuration-building bugs (the red ones are raised
    together at the end).

    ① the f32 WAV is byte-identical to the existing `reference.wav` (the sim path computes the same
       as the real one)
    ② each configuration actually differs from f32 (the rounding is not passed through)
    ③ w8a16 / w8a8 differ from w8 (**the detector of a missed activation quantization** — when it
       is missed, the activation rounding disappears and the result is **bit-identical** to w8).
       Zero checks on the number of replaced ops and of call sites sit alongside (②③ look at the
       hole from the values, these from the mechanism)
    ④ the w4 rounded tensor count and element count match the census (if `include` is ignored the
       targets silently grow, and the counts drift before core fails loudly on the weights that
       were supposed to be excluded)
    ⑤ the w4 configurations differ from each other (**the detector of a missed method dispatch** —
       applying the same method twice makes them bit-identical)
    ⑥ BERT's rounding actually moves the features (if the rounding passes through, the features are
       bit-identical to f32, and ②'s waveform side alone cannot tell it from "the same because
       net_g is f32")

    When the configurations are narrowed with `--configs`, the checks involving configurations
    that did not run are **skipped** (making them red on the narrowed side would keep the smoke run
    always red). That they were skipped is readable from the fields being empty.
    """
    gates: dict[str, Any] = {}
    reference_ok = None
    if reference_wav.is_file():
        reference_ok = sha256(wavs["f32"].read_bytes()) == sha256(reference_wav.read_bytes())
    gates["f32_wav_bit_exact_vs_reference"] = {
        "reference": str(reference_wav),
        "checked": reference_ok is not None,
        "passed": reference_ok,
    }

    base = results["f32"]["audio"]
    differs = {
        name: not bool(torch.equal(results[name]["audio"], base))
        for name in CONFIGS
        if name != "f32" and name in results
    }
    gates["differs_from_f32"] = differs

    act_names = [name for name in ("w8a16", "w8a8") if name in results]
    act_effective: dict[str, bool] = {}
    if "w8" in results:
        w8 = results["w8"]["audio"]
        act_effective = {
            name: not bool(torch.equal(results[name]["audio"], w8)) for name in act_names
        }
    gates["act_quant_effective_vs_w8"] = act_effective

    patched_ops = {
        name: results[name]["diagnostics"]["act_quant"]["patched_ops"] for name in act_names
    }
    call_sites = {
        name: results[name]["diagnostics"]["act_quant"]["call_sites"] for name in act_names
    }
    gates["act_quant_patched_ops"] = patched_ops
    gates["act_quant_call_sites"] = call_sites

    w4_names = [name for name in CONFIGS if name in results and CONFIGS[name].weight == "w4"]
    w4_targets = {
        name: {
            tag: (
                results[name]["diagnostics"]["w4"]["targets"][tag]
                == {
                    "modules": census[(CONFIGS[name].roles, tag)].counts.modules,
                    "elements": census[(CONFIGS[name].roles, tag)].counts.elements,
                }
            )
            for tag in CONFIGS[name].scope
        }
        for name in w4_names
    }
    gates["w4_targets_match_census"] = w4_targets

    w4_distinct = {
        f"{left} vs {right}": not bool(torch.equal(results[left]["audio"], results[right]["audio"]))
        for left, right in combinations(w4_names, 2)
    }
    gates["w4_methods_distinct"] = w4_distinct

    bert_effective = {method: report["differs_from_f32"] for method, report in bert_reports.items()}
    gates["bert_quant_effective"] = bert_effective

    failures: list[str] = []
    if reference_ok is False:
        failures.append(
            "the f32 WAV is not byte-identical to reference.wav (the sim path's vacuous-pass check)"
        )
    failures += [f"{name} came out bit-identical to f32" for name, ok in differs.items() if not ok]
    failures += [
        f"{name} came out bit-identical to w8 (the activation quantization was not applied)"
        for name, ok in act_effective.items()
        if not ok
    ]
    failures += [
        f"{name} replaced 0 activation-quantization ops"
        for name, n in patched_ops.items()
        if n == 0
    ]
    failures += [
        f"{name} has 0 activation-quantization call sites"
        for name, n in call_sites.items()
        if n == 0
    ]
    failures += [
        f"the w4 targets of {name}/{tag} disagree with the census"
        for name, tags in w4_targets.items()
        for tag, ok in tags.items()
        if not ok
    ]
    failures += [
        f"{pair} came out bit-identical (the w4 method dispatch was missed)"
        for pair, ok in w4_distinct.items()
        if not ok
    ]
    failures += [
        f"BERT's {method} produced the same features as f32 (the rounding passed through)"
        for method, ok in bert_effective.items()
        if not ok
    ]
    gates["failures"] = failures
    return gates


def selected_configs(spec: str | None) -> tuple[str, ...]:
    """Resolves the `--configs` spec into a subset of the main configurations (`f32` is always
    included as the SNR baseline)."""
    if spec is None:
        return tuple(CONFIGS)
    names = [name for name in spec.split(",") if name]
    unknown = [name for name in names if name not in CONFIGS]
    if unknown:
        raise SystemExit(f"未知の構成 {unknown}（選べるのは {list(CONFIGS)}）")
    return tuple(name for name in CONFIGS if name == "f32" or name in names)


def build_w4_section(
    census: Mapping[tuple[str, str], W4Census],
    inputs: ChainInputs,
    bert_reports: Mapping[str, dict[str, Any]],
    dist_dir: Path,
    calib_limit: int | None,
    methods: Mapping[str, W4Method],
    group_size: int,
) -> dict[str, Any]:
    """The `w4` section of report.json — the census of the target sets, the size estimate and the
    projection onto the distribution.

    The slot without a quality measurement (net_g's linears only) also appears here — **the
    absence of any gain is itself the reason for the decision**, so its target count and size
    estimate must always be visible.
    """
    net_g_counts = {
        roles: census[(roles, "front")].counts + census[(roles, "voice")].counts
        for roles in W4_ROLES
    }
    counts = {
        "net_g:all": net_g_counts["all"],
        "net_g:linear": net_g_counts["linear"],
        "bert:linear": inputs.bert_census.counts,
    }
    # "Eligible for today's i4" matched against the distribution's tensor names = the FQN set of the
    # linear weight slots.
    linear_fqns = frozenset(
        f"{name}.weight"
        for source in (census[("linear", "front")], census[("linear", "voice")], inputs.bert_census)
        for name in source.eligible
    )
    section: dict[str, Any] = {
        "group_size": group_size,
        "methods": {method.name: method.formula for method in methods.values()},
        "census": {
            "net_g": {
                f"{roles}/{tag}": _census_entry(entry) for (roles, tag), entry in census.items()
            },
            "bert": _census_entry(inputs.bert_census),
        },
        "projections": build_projections(counts, methods),
        "bert_quant": dict(bert_reports),
        "calib": {
            "group_size": CALIB_GROUP_SIZE,
            "target": CALIB_TARGET,
            "target_description": "the in-encoder linears of BERT (DeBERTa) **up to the layer the"
            " features are taken from** — a subset of the `bert:linear` census (the last layers"
            " do not affect the output)",
            "texts": len(calib_corpus(calib_limit)),
            "limit": calib_limit,
            "methods": {name: config.formula for name, config in CALIB_CONFIGS.items()},
        },
    }
    if dist_dir.is_dir():
        section["distribution"] = project_distribution(dist_dir, linear_fqns, group_size)
    else:
        section["distribution"] = {"root": str(dist_dir), "checked": False}
    return section


def _census_entry(census: W4Census) -> dict[str, Any]:
    return {
        "modules": census.counts.modules,
        "channels": census.counts.channels,
        "elements": census.counts.elements,
        "groups": census.counts.groups,
        "by_role": {
            role: {"modules": c.modules, "channels": c.channels, "elements": c.elements}
            for role, c in census.by_role.items()
        },
        "excluded": list(census.excluded),
        "excluded_modules": len(census.excluded),
        "excluded_elements": census.excluded_elements,
    }


def parse_w4_group_size(raw: str) -> int:
    """Accepts `--w4-group-size` (**a power of two that is at least {@link MIN_GROUP_SIZE}**).

    The value range is core's storage rule itself (ADR 0069 decision 2 — the condition under which
    row boundaries and group boundaries are always byte-aligned), borrowed from `karume.ir`. It is
    rejected here because with an out-of-range g **the result cannot be shipped even if measured**
    — having emit / verify reject it after the measurement would waste a whole run.
    """
    value = int(raw)
    if value & (value - 1) or value < MIN_GROUP_SIZE:
        raise argparse.ArgumentTypeError(
            f"group 長 {value} が 2 冪かつ {MIN_GROUP_SIZE} 以上でない（ADR 0069 決定 2）"
        )
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model-dir", type=Path, default=export.DEFAULT_MODEL_DIR)
    parser.add_argument("--dump", type=Path, default=DEFAULT_DUMP)
    parser.add_argument("--assets", type=Path, default=demo.DEFAULT_DEMO_DIR / demo.STYLE_FILE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--reference-wav", type=Path, default=DEFAULT_REFERENCE_WAV)
    parser.add_argument("--dist-dir", type=Path, default=DEFAULT_DIST_DIR)
    parser.add_argument("--top", type=int, default=12, help="rows in the layer group table")
    parser.add_argument(
        "--configs",
        default=None,
        help="narrows the main configurations to run, comma-separated (default: all)."
        " f32 always runs because it is the SNR baseline",
    )
    parser.add_argument(
        "--w4-group-size",
        type=parse_w4_group_size,
        default=DEFAULT_GROUP_SIZE,
        help=f"the w4 group size (a power of two, at least {MIN_GROUP_SIZE}, default"
        f" {DEFAULT_GROUP_SIZE}). The same g flows into eligibility, rounding and the method"
        " name. The grid of the calibrated configurations stays 32",
    )
    parser.add_argument(
        "--calib-limit",
        type=int,
        default=None,
        help=f"uses only the first N sentences of the calibration corpus (default: all"
        f" {len(CALIB_TEXTS)}). A knob for reduced smoke runs; it affects only the calibrated"
        " configurations",
    )
    parser.add_argument(
        "--no-diagnostics",
        action="store_true",
        help="does not run the front / voice orthogonal-decomposition diagnostics (main"
        " configurations only)",
    )
    parser.add_argument(
        "--inject",
        choices=("drop-act-quant",),
        default=None,
        help="fault injection (demonstrates the detectors' power) — builds w8a16 / w8a8 without"
        " replacing the activation-quantization ops (reproduces a configuration-building bug)",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    configs = selected_configs(args.configs)
    # MUST: g is read here once and passed as an argument from then on (holding it at module level
    # makes a silent split of "eligibility at g16 / rounding at g32" possible — the MUST of
    # {@link rtn_method_name}).
    methods = w4_methods(args.w4_group_size)

    inputs = prepare_inputs(args.dump, args.assets, args.w4_group_size)
    print(
        f"[inputs] text={inputs.meta['text']!r} bert={list(inputs.bert_feature.shape)}"
        f" bert_linear={inputs.bert_census.counts.modules}",
        flush=True,
    )
    census = net_g_census(args.model_dir, args.w4_group_size)
    for roles in W4_ROLES:
        front, voice = census[(roles, "front")], census[(roles, "voice")]
        print(
            f"[census] net_g {roles}: {front.counts.modules + voice.counts.modules} eligible"
            f" / {front.counts.elements + voice.counts.elements:,} elements"
            f" ({len(front.excluded) + len(voice.excluded)} excluded"
            f" / {front.excluded_elements + voice.excluded_elements:,} elements)",
            flush=True,
        )

    plan: list[tuple[str, Path]] = [(name, args.out) for name in configs]
    if not args.no_diagnostics:
        diag_dir = args.out / "diagnostics"
        diag_dir.mkdir(parents=True, exist_ok=True)
        plan += [(name, diag_dir) for name in DIAGNOSTICS]

    # The features of the configurations that vary BERT (built once per rounding — they are
    # reloaded from the plain weights, so nothing stacks).
    features: dict[str | None, torch.Tensor] = {None: inputs.bert_feature}
    bert_reports: dict[str, dict[str, Any]] = {}

    results: dict[str, dict[str, Any]] = {}
    wavs: dict[str, Path] = {}
    for name, directory in plan:
        variant = bert_variant(RECIPES[name], methods)
        if variant is not None and variant not in features:
            feature, detail = bert_feature_for(RECIPES[name], inputs, args.calib_limit, methods)
            features[variant] = feature
            bert_reports[variant] = detail | {
                "differs_from_f32": not bool(torch.equal(feature, inputs.bert_feature)),
                "rel_rms_vs_f32": rel_rms(feature, inputs.bert_feature),
            }
            print(f"[bert:{variant}] {detail['quant']}", flush=True)
        results[name] = run_config(
            name,
            args.model_dir,
            inputs,
            bert_feature=features[variant],
            census=census,
            methods=methods,
            group_size=args.w4_group_size,
            inject=args.inject,
        )
        audio = results[name]["audio"]
        path = directory / f"{name}.wav"
        path.write_bytes(demo.wav_pcm16(audio.numpy(), inputs.meta["samplingRate"]))
        wavs[name] = path
        diagnostics = results[name]["diagnostics"]
        print(
            f"[{name}] {diagnostics['elapsed']:.0f}s frames={diagnostics['frames']}"
            f" act={diagnostics['act_quant']['calls']} calls → {path}",
            flush=True,
        )

    gates = run_gates(results, wavs, args.reference_wav, census, bert_reports)
    w4 = build_w4_section(
        census,
        inputs,
        bert_reports,
        args.dist_dir,
        args.calib_limit,
        methods,
        args.w4_group_size,
    )
    report = build_report(args, inputs, results, wavs, gates, w4)
    # Also keep the comparison with the dump's waveform (Karume's output on a real GPU) — a second
    # cross-check of the f32 path.
    dump_audio = inputs.tensors["audio"].reshape(-1).to(torch.float32)
    if dump_audio.shape == results["f32"]["audio"].shape:
        report["f32_vs_dump_audio"] = {
            "maxAbs": float((results["f32"]["audio"] - dump_audio).abs().max()),
            "snr_db": snr_db(dump_audio, results["f32"]["audio"]),
        }
    (args.out / "report.json").write_text(
        json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print()
    print(format_table(report["configs"]))
    if report["diagnostics"]:
        print()
        print("Diagnostics (front / voice orthogonal decomposition):")
        print(format_table(report["diagnostics"]))
    calib_notes = [
        f"> {variant}: {entry['quant']} / projected {entry['size']['bits_per_weight']:.3f} bpw"
        f" ({entry['size']['formula']})"
        for variant, entry in bert_reports.items()
        if variant in CALIB_CONFIGS
    ]
    if calib_notes:
        print()
        print(
            f"Breakdown of the calibrated configurations ({len(calib_corpus(args.calib_limit))}"
            f" calibration sentences / target {CALIB_TARGET}):"
        )
        print("\n".join(calib_notes))
        if args.calib_limit is not None:
            print(
                f"> calibration is a reduced run with only the first {args.calib_limit} sentences"
                f" (not all {len(CALIB_TEXTS)} — do not read the numbers as the full ones)"
            )
    print()
    print(
        "w4 size estimate (**a projection by formula, not a measurement** — only RTN has a"
        " storage format):"
    )
    print(format_projection_table(report["w4"]["projections"]))
    distribution = report["w4"]["distribution"]
    print()
    if distribution.get("checked") is False:
        print(f"no distribution at {distribution['root']}, so the reduction estimate is skipped")
    else:
        print(format_distribution(distribution))
    print()
    print(f"report → {args.out / 'report.json'}")
    if gates["failures"]:
        raise AssertionError("verification gates are red: " + " / ".join(gates["failures"]))
    print("gates: all green")


if __name__ == "__main__":
    main()
