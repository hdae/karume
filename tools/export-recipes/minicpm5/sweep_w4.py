"""Script for the w4 fake-quant sweep of MiniCPM5-1B
(ADR [0069](../../../docs/decisions/0069-packed-w4-storage.md) decision 6 = Phase 0) and the
**quantization-method screening**.

    uv run --with 'transformers==5.14.1' python -m minicpm5.sweep_w4

**Not one line of the runtime is touched**. All this does is round the weights on the torch side
(fake-quant) and measure the quality loss. The storage form (the container codec `int4-sym-g` —
`docs/container-v1.md` §6.3 / pack order / WGSL) was implemented after Phase 0, and this script
does not go through it. In terms of ADR 0058 decision 5's split it produces the numbers side of
"quality = human review + advisory numbers", and its only mechanical check is the reproducibility
of the baseline ({@link assert_baseline_reproduces}).

## What is measured

**Phase 0 grid** ({@link SWEEP_CONFIGS}) = group_size {32, 64, 128} × symmetric / asymmetric, 6
configurations + the baseline + one run excluding `lm_head`.

**Method grid** ({@link METHOD_CONFIGS}) = 7 methods ({@link QUANT_METHODS} — RTN i4 / FP4 / NF4 /
MXFP4 / k-means at 3 table granularities) × 2 target sets (linear only, 169 tensors / including
`embed_tokens`, 170 tensors). **group_size is fixed at 32** and the g axis is not swept — Phase 0
has answered the g question, and what is wanted here is only "the difference when the rounding
changes at the same g" (sweeping both axes at once would mix the method difference with the g
difference). The rounding implementation is shared with core (`karume.quant_methods` /
`karume.quantize`), and so is the target selection (`iter_quant_targets`) — a copied
implementation would let "the targets and formulas measured" and "the targets and formulas
shipped" silently diverge.

**Calibration grid** ({@link CALIB_CONFIGS}) = 5 calibrated roundings (GPTQ × 3 storage grids /
AWQ / AWQ+GPTQ). The rounding is shared with core's `karume.quant_calib`, and **the grid is the
same as the method grid's** (what changes is which level of the same grid a value goes to). The
target is **a single set: the linears inside the decoder** — neither a set including
`embed_tokens` nor `lm_head` is built — because the calibration driver (`calibrate_stages`) is
confined to the `nn.Linear` inside a stage, so the target set differs from the existing table of
"7 methods × 2 target sets" ({@link CalibConfig}). The calibration input is the 48 texts of
{@link minicpm5.calib_texts.CALIB_TEXTS}; the calls into the first decoder layer are captured and
fed to the stage sequence ({@link capture_stage_batches}). Before feeding, **the stage-split
check** ({@link assert_stage_split}) measures with bit identity that "the sequential forward of
the stage sequence = the model's final hidden" — a split that drifts still yields ordinary
numbers, so without the check the adoption of a method would be decided by a measurement of a
different path.

Four columns are taken per config (the first 3 are the same as Phase 0; the 4th is for the method
and calibration grids):

1. **Relative weight RMSE** (per family) — the size of the rounding itself. `‖w − fq‖₂ / ‖w‖₂`
   over all the layers of a family, to see which families break easily.
2. **Teacher-forced agreement / NLL** — the next token when the expected sequence is fed one token
   at a time. The sequence does not shift even after a divergence, so the loss at **the same
   positions** can be compared across configs.
3. **Divergence step of free-running greedy** — how long actual generation follows the expected
   sequence. The column closest to "how it breaks" as a human reads it.
4. **Size estimate** ({@link QuantMethod.formula}) — each method's effective bpw and the MiB
   projected over the quantized tensor set alone. Quality alone hides the cost of the tables
   (k-means per-channel has tables proportional to the channel count), so it sits in the same
   table as quality. **A projection by formula, not a measurement**; methods without a storage
   form (all 7 of this script except RTN) cannot even be written.

**The source of truth of the expected sequences is the greedy expected sequences
`minicpm5.export_decode` took with the margin check** (the 3 cases of
`outputs/series/minicpm5-1b-decode/greedy.<case>.safetensors`). They are not re-taken here because
what the sweep wants to measure is only "the quantization loss against the same expected
sequences". No tokenizer is involved (everything stays in token ids).

## The quantization form (ADR 0069 decision 3)

The group axis is **the last dimension of the weight = the in axis of the linear**. Symmetric is
`s = clamp(amax/7, f32 tiny)` / `q = clamp(round(w/s), ±7)`; the amax element is restored exactly,
so the fake-quant is idempotent (the 4-bit version of ADR 0019's ±127 argument). Asymmetric is a
**measurement column** that uses the zero-point as a continuous value, neither rounded nor stored
— a column that shows "the upper bound of the quality reachable by adding a zero-point", to
detect a quality shortfall of symmetric before implementing it (reservation 3 of decision 3).

## Output

Progress goes to stderr, flushed immediately (to follow a background run). At the end, 4 markdown
tables go to stdout (summary / quality / wRMSE per family / size estimate) — in a form to copy into
`docs/research/` as is. With `--json`, machine-readable JSON is written too (**rewritten after
every config** — if a run of tens of minutes dies halfway, the measurements so far remain).
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import torch
from safetensors.torch import load_file
from torch import nn

from _shared.decode_series import EXPECTED_KEY, GREEDY_PREFIX, GREEDY_SUFFIX, PROMPT_KEY
from karume.quant_calib import (
    GPTQ_DAMPING,
    CalibMethod,
    CalibReport,
    GridSpec,
    StageBatch,
    StageSpec,
    calibrate_stages,
    first_tensor_output,
)
from karume.quant_methods import (
    DEFAULT_CODEBOOK_LEVELS,
    fake_quant_fp4,
    fake_quant_kmeans,
    fake_quant_mxfp4,
    fake_quant_nf4,
)
from karume.quantize import (
    DEFAULT_GROUP_SIZE,
    QuantizeError,
    fake_quant_int4,
    iter_quant_targets,
)
from minicpm5 import export as one_shot
from minicpm5 import export_decode as decode
from minicpm5.calib_texts import CALIB_TEXTS

#: Where the real weights live (the same material as the one-shot / decode forms).
DEFAULT_MODEL_DIR = one_shot.DEFAULT_MODEL_DIR

#: Where the expected sequences live (the greedy golden `export_decode` took with the margin check).
DEFAULT_DECODE_DIR = decode.DEFAULT_OUT_DIR

#: Half-width of symmetric quantization (**−8 is not used**). Closing it at ±7 puts a group's amax
#: element on `q = ±7`, restored exactly by `q·s`, so the fake-quant is **idempotent** (ADR 0069
#: decision 3 — the 4-bit version of ADR 0019's ±127 argument). Allowing −8 causes
#: requantizations where only the scale moves.
INT4_MAX = 7

#: Value range of the asymmetric unsigned nibble (15 levels, **0 unused**). The same range that
#: symmetric `q ∈ [−7,7]` lands on with offset 8, so the pack format and pack order do not change
#: with or without a zero-point (reservation 2 of ADR 0069 decision 3).
UINT4_MIN = 1
UINT4_MAX = 15

#: Default when the zero-point is omitted (reservation 2 of ADR 0069 decision 3 — dequant is
#: `(u − 8)·scale`). Degenerate asymmetric groups fall back to this default
#: ({@link asymmetric_components}).
DEFAULT_ZERO_POINT = 8

#: The linear families to quantize (the last module name of the fqn). MiniCPM5-1B has 24 layers ×
#: q/k/v/o/gate/up/down + `lm_head` = 169 tensors, and `tie_word_embeddings: false` makes `lm_head`
#: an independent weight. Splitting into families separates "where it breaks".
LINEAR_FAMILIES: tuple[str, ...] = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "lm_head",
)

#: Family name of the vocabulary side (what the `--only`-like exclusion
#: {@link SweepConfig.include_lm_head} refers to).
LM_HEAD = "lm_head"

#: Family name of the input-side embedding (the one tensor the method grid's "including
#: non-linear" adds). `tie_word_embeddings: false`, so it is a weight independent of `lm_head`.
EMBED_TOKENS = "embed_tokens"

#: Columns of the per-family wRMSE table (the method grid can reach `embed_tokens`).
METHOD_FAMILIES: tuple[str, ...] = (*LINEAR_FAMILIES, EMBED_TOKENS)

#: Config name of the baseline ({@link select_configs} always puts it first).
BASELINE_NAME = "baseline"


@dataclass(frozen=True)
class SweepConfig:
    """The spec of one run.

    `group_size is None` is the unquantized baseline. `include_lm_head=False` is the column that
    leaves `lm_head` out of the targets, there to separate the contribution of the 200M vocabulary
    weights from the block side.
    """

    name: str
    group_size: int | None = None
    asymmetric: bool = False
    include_lm_head: bool = True


#: The sweep grid (runs in this order). group_size is a power of two and ≥ 16 (ADR 0069
#: decision 2).
SWEEP_CONFIGS: tuple[SweepConfig, ...] = (
    SweepConfig(BASELINE_NAME),
    SweepConfig("g32-sym", group_size=32),
    SweepConfig("g64-sym", group_size=64),
    SweepConfig("g128-sym", group_size=128),
    SweepConfig("g32-asym", group_size=32, asymmetric=True),
    SweepConfig("g64-asym", group_size=64, asymmetric=True),
    SweepConfig("g128-asym", group_size=128, asymmetric=True),
    SweepConfig("g128-sym-blocks", group_size=128, include_lm_head=False),
)


# ---- Method grid (quantization-method screening) ---------------------------

#: Group length of the method comparison = **fixed at 32** (core's default = the value Phase 0's
#: measurements settled). The g axis is not swept at the same time because once the method
#: difference and the g difference mix, "which one had the effect" can no longer be told.
METHOD_GROUP_SIZE = DEFAULT_GROUP_SIZE

#: Bits of one k-means table (16 centroids × f32).
CODEBOOK_BITS = DEFAULT_CODEBOOK_LEVELS * 32


@dataclass(frozen=True)
class TargetCounts:
    """Counts of the quantized tensor set (the input of the size estimate).

    `channels` is the sum of the row counts of {@link karume.quantize.channel_rows} = the number of
    units a per-channel table spans, and `modules` is the layer count (the number of per-tensor
    tables).
    """

    modules: int
    channels: int
    elements: int

    @property
    def groups(self) -> int:
        """Total number of g32 groups (used by the size estimate of methods that need one scale per
        group)."""
        return self.elements // METHOD_GROUP_SIZE


@dataclass(frozen=True)
class QuantMethod:
    """One rounding method — name, how the rounding is applied, and the size-estimate formula
    bundled in one row.

    Scattering the three across separate tables would let "the method whose quality was measured"
    and "the method whose size was estimated" silently diverge (adding a method would update only
    one of them).
    """

    #: Method name shown in the tables and the JSON.
    name: str
    #: Prefix of the config names (`<slug>-linear` / `<slug>-embed`).
    slug: str
    #: Applies the rounding to the model in place (`(model, op_types, group_size)`).
    round_weights: Callable[[nn.Module, tuple[type[nn.Module], ...], int], object]
    #: Projected bit count of the quantized set.
    projected_bits: Callable[[TargetCounts], float]
    #: The formula of `projected_bits` (**put into the output as is** — so the projection's
    #: assumptions can be traced from the table).
    formula: str


#: Method name of RTN (= the storage form `i4`). **The baseline** of the method columns — the
#: other 6 are read as differences from it.
RTN_METHOD = "rtn-g32-sym"

#: Prefix of the k-means method names (`kmeans:<granularity>` — the same spelling as core's
#: {@link karume.quant_methods.KMEANS_GRANULARITIES}).
KMEANS_PREFIX = "kmeans:"

#: The 7 methods (run in this order). Every rounding is core's shared implementation; this only
#: dispatches between them.
QUANT_METHODS: tuple[QuantMethod, ...] = (
    QuantMethod(
        RTN_METHOD,
        "rtn",
        lambda model, op_types, group: fake_quant_int4(model, group, op_types=op_types),
        lambda counts: 4 * counts.elements + 32 * counts.groups,
        "4bit + g32 f32 scale = 5.0 bpw",
    ),
    QuantMethod(
        "fp4",
        "fp4",
        lambda model, op_types, group: fake_quant_fp4(model, group, op_types=op_types),
        lambda counts: 4 * counts.elements + 32 * counts.groups,
        "4bit + g32 f32 scale = 5.0 bpw",
    ),
    QuantMethod(
        "nf4",
        "nf4",
        lambda model, op_types, group: fake_quant_nf4(model, group, op_types=op_types),
        lambda counts: 4 * counts.elements + 32 * counts.groups,
        "4bit + g32 f32 scale = 5.0 bpw",
    ),
    QuantMethod(
        "mxfp4",
        "mxfp4",
        lambda model, op_types, group: fake_quant_mxfp4(model, group, op_types=op_types),
        lambda counts: 4 * counts.elements + 8 * counts.groups,
        "4bit + g32 E8M0 scale = 4.25 bpw",
    ),
    QuantMethod(
        f"{KMEANS_PREFIX}per_tensor",
        "kmeans-tensor",
        lambda model, op_types, group: fake_quant_kmeans(
            model, "per_tensor", group, op_types=op_types
        ),
        lambda counts: 4 * counts.elements + CODEBOOK_BITS * counts.modules,
        "4bit + 16×f32 table per layer (no scale)",
    ),
    QuantMethod(
        f"{KMEANS_PREFIX}per_channel",
        "kmeans-channel",
        lambda model, op_types, group: fake_quant_kmeans(
            model, "per_channel", group, op_types=op_types
        ),
        lambda counts: 4 * counts.elements + CODEBOOK_BITS * counts.channels,
        "4bit + 16×f32 table per channel (no scale)",
    ),
    QuantMethod(
        f"{KMEANS_PREFIX}shared",
        "kmeans-shared",
        lambda model, op_types, group: fake_quant_kmeans(model, "shared", group, op_types=op_types),
        lambda counts: 4 * counts.elements + 32 * counts.groups + CODEBOOK_BITS,
        "4bit + g32 f32 scale + one 16×f32 table overall",
    ),
)


@dataclass(frozen=True)
class MethodConfig:
    """One run of the method grid (method × 2 target sets).

    `include_embedding` is the "including non-linear" column — it adds `embed_tokens` to the
    targets. ADR 0069 addendum 6 opened the i4-eligible ops to embedding too
    (`karume.emit.I4_WEIGHT_OPS`), so with RTN i4 this form can be stored as well. The quality and
    size share of rounding the vocabulary side informs which tensors to make i4 (methods other than
    RTN have no storage form and are measurement only).
    """

    name: str
    method: QuantMethod
    include_embedding: bool


#: Method grid = 7 methods × 2 target sets (linear only / including `embed_tokens`).
METHOD_CONFIGS: tuple[MethodConfig, ...] = tuple(
    MethodConfig(
        name=f"{method.slug}-{'embed' if include_embedding else 'linear'}",
        method=method,
        include_embedding=include_embedding,
    )
    for method in QUANT_METHODS
    for include_embedding in (False, True)
)


# ---- Calibration grid (calibrated rounding) ---------------------------------

#: Group length of calibrated rounding = fixed at 32, the same as the method grid. Calibration
#: **does not change a single byte of the grid** (it only re-chooses rounding targets within the
#: same grid — the module docstring of `karume.quant_calib`), so the reason for not sweeping the g
#: axis is the same as for {@link METHOD_GROUP_SIZE}.
CALIB_GROUP_SIZE = METHOD_GROUP_SIZE


def group_scale_bits(counts: TargetCounts) -> float:
    """4-bit payload + one f32 scale per group (exactly the storage of RTN / NF4)."""
    return 4 * counts.elements + 32 * counts.groups


def layer_table_bits(counts: TargetCounts) -> float:
    """The formula above plus **one k-means table per layer**.

    Core's `kmeans_shared` spans "one table over the value space normalized by the group absmax",
    but its reach is confined **within a layer** (the stage-sequential driver and a table across all
    layers do not fit together — the NOTE of {@link karume.quant_calib.GridSpec}). This is why the
    formula differs from the method grid's `kmeans:shared` (one table overall).
    """
    return group_scale_bits(counts) + CODEBOOK_BITS * counts.modules


#: Projection formula of a 4-bit grid with group-absmax scales (the wording shared by the methods
#: without tables).
GROUP_SCALE_FORMULA = "4bit + g32 f32 scale = 5.0 bpw"

#: The note attached to the projection formulas of the AWQ family — the equivalent scale `s` can
#: only be held folded into `W_eff`, and storing it needs a fold or a companion (the module
#: docstring of `karume.quant_calib`).
AWQ_FORMULA_NOTE = " (the fold / companion of the equivalent scale s is not counted)"


@dataclass(frozen=True)
class CalibConfig:
    """The spec of one calibrated rounding (method × storage grid).

    The targets are **limited to the `nn.Linear` inside the decoder**
    ({@link karume.quant_calib.calibrate_stages} looks only at the linears inside a stage).
    `lm_head` and `embed_tokens` are outside the stages and not included, so **the target set
    differs** from the method grid's "2 target sets" — the size estimate is also computed from the
    counts of the set the calibration actually rounded ({@link calib_targets}).
    """

    name: str
    method: CalibMethod
    grid: GridSpec
    #: Projected bit count of the quantized set (the same holder as
    #: {@link QuantMethod.projected_bits}).
    projected_bits: Callable[[TargetCounts], float]
    #: The formula of `projected_bits` (**put into the output as is**).
    formula: str
    #: GPTQ sweep axes (the defaults are bit-identical to core's defaults — the experiment axes of
    #: {@link karume.quant_calib}).
    static_groups: bool = False
    act_order: bool = False
    #: A swept value MUST be stated in the output (core-side docstring) — {@link label} does that.
    damping: float = GPTQ_DAMPING

    @property
    def label(self) -> str:
        """Method name shown in the tables and the JSON (`<method>/<storage grid>` + the non-default
        sweep axes)."""
        axes = ""
        if self.act_order:
            axes += "+act"
        elif self.static_groups:
            axes += "+static"
        if self.damping != GPTQ_DAMPING:
            axes += f"+damp{self.damping:g}"
        return f"{self.method}/{self.grid.kind}{axes}"


#: Calibration grid = 5 runs (in this order). Every rounding is core's shared implementation
#: (`karume.quant_calib`); this only holds the dispatch and the projection formulas. **The target is
#: a single set, linear only** — no set including `embed_tokens` is built (`calibrate_stages` is
#: limited to `nn.Linear`, and the vocabulary table is outside the stages).
CALIB_CONFIGS: tuple[CalibConfig, ...] = (
    CalibConfig(
        "gptq-rtn",
        "gptq",
        GridSpec(kind="rtn", group_size=CALIB_GROUP_SIZE),
        group_scale_bits,
        GROUP_SCALE_FORMULA,
    ),
    CalibConfig(
        "gptq-nf4",
        "gptq",
        GridSpec(kind="nf4", group_size=CALIB_GROUP_SIZE),
        group_scale_bits,
        f"{GROUP_SCALE_FORMULA} (the grid is NF4's fixed table)",
    ),
    CalibConfig(
        "gptq-kmeans",
        "gptq",
        GridSpec(kind="kmeans_shared", group_size=CALIB_GROUP_SIZE),
        layer_table_bits,
        "4bit + g32 f32 scale + one 16×f32 table per layer",
    ),
    CalibConfig(
        "awq-rtn",
        "awq",
        GridSpec(kind="rtn", group_size=CALIB_GROUP_SIZE),
        group_scale_bits,
        f"{GROUP_SCALE_FORMULA}{AWQ_FORMULA_NOTE}",
    ),
    CalibConfig(
        "awq-gptq-rtn",
        "awq+gptq",
        GridSpec(kind="rtn", group_size=CALIB_GROUP_SIZE),
        group_scale_bits,
        f"{GROUP_SCALE_FORMULA}{AWQ_FORMULA_NOTE}",
    ),
)

#: The type of config `--only` can choose (the 3 lines: Phase 0 / method / calibration).
AnyConfig = SweepConfig | MethodConfig | CalibConfig

#: Every config `--only` can choose (declaration order = run order).
ALL_CONFIGS: tuple[AnyConfig, ...] = (*SWEEP_CONFIGS, *METHOD_CONFIGS, *CALIB_CONFIGS)

#: Experiment points of the GPTQ sweep axes (filed 2026-08-31 — the port of upstream
#: --static-groups × --act-order and the damping sensitivity). **They run only when named with
#: `--only`** — not part of the default full sweep (no arguments) (one configuration ≈ the extra
#: cost of one full calibration, and the default sweep time is not silently doubled).
EXPERIMENT_CONFIGS: tuple[CalibConfig, ...] = (
    CalibConfig(
        "gptq-rtn-static",
        "gptq",
        GridSpec(kind="rtn", group_size=CALIB_GROUP_SIZE),
        group_scale_bits,
        GROUP_SCALE_FORMULA,
        static_groups=True,
    ),
    CalibConfig(
        "gptq-rtn-act",
        "gptq",
        GridSpec(kind="rtn", group_size=CALIB_GROUP_SIZE),
        group_scale_bits,
        GROUP_SCALE_FORMULA,
        static_groups=True,
        act_order=True,
    ),
    *(
        CalibConfig(
            f"gptq-rtn-damp{tag}",
            "gptq",
            GridSpec(kind="rtn", group_size=CALIB_GROUP_SIZE),
            group_scale_bits,
            GROUP_SCALE_FORMULA,
            damping=value,
        )
        for tag, value in (("001", 0.001), ("003", 0.003), ("03", 0.03), ("1", 0.1))
    ),
)

#: The population `--only` resolves names in (the default run = ALL_CONFIGS / experiment points
#: only by explicit selection).
SELECTABLE_CONFIGS: tuple[AnyConfig, ...] = (*ALL_CONFIGS, *EXPERIMENT_CONFIGS)


# ---- Quantization formulas (ADR 0069 decision 3) ----------------------------


def grouped_view(weight: torch.Tensor, group_size: int) -> torch.Tensor:
    """Split the quantization axis (**the last dimension = the in axis of the linear**) into
    `[…, groups, group_size]`.

    MUST: a shape that does not divide fails loudly. `i4` requires (MUST) that the in axis be
    divisible by `group_size` (ADR 0069 decision 2 — the constraint against partial groups
    guarantees byte alignment at row and group boundaries), so measuring the quality of a form that
    silently built partial groups would mean choosing the default on "numbers of a form that cannot
    be stored".
    """
    in_axis = int(weight.shape[-1])
    if in_axis % group_size:
        raise QuantizeError(
            f"the in axis {in_axis} is not divisible by group_size {group_size}"
            " (`i4` requires (MUST) the quantization axis to be divisible by group_size — ADR 0069"
            " decision 2)"
        )
    return weight.reshape(*weight.shape[:-1], in_axis // group_size, group_size)


def symmetric_components(
    weight: torch.Tensor, group_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return `(q, scale)` of grouped symmetric quantization in group form (`[…, groups, *]`).

    `scale = clamp(amax / 7, f32 tiny)` / `q = clamp(round(w / scale), ±7)`. The lower clamp is
    needed for an all-zero group (`amax == 0`) — dividing as is would give NaN. After the clamp it
    falls to `q = 0` and `q·scale = 0` restores the original value exactly (the 4-bit version of
    the same reasoning as `quantize.channel_scale`).

    The components are returned separately from the fq ({@link fake_quant_symmetric}) so the tests
    can directly constrain the value range (**no −8**) and that the scales are separate per group.
    """
    grouped = grouped_view(weight, group_size)
    amax = grouped.abs().amax(dim=-1, keepdim=True)
    scale = torch.clamp(amax / INT4_MAX, min=torch.finfo(torch.float32).tiny)
    return torch.round(grouped / scale).clamp_(-INT4_MAX, INT4_MAX), scale


def fake_quant_symmetric(weight: torch.Tensor, group_size: int) -> torch.Tensor:
    """Symmetric 15-level fake-quant (returns `q·scale` in the same shape as the weight).

    MUST: **idempotent** (bit-identical when reapplied). It follows from the amax element landing
    on `q = ±7` and being restored exactly; if it broke, "weights that should already be rounded
    move with every requantization" = the correspondence between the storage form and the reference
    breaks (ADR 0019 / 0069 decision 3). `tests/test_sweep_w4.py` pins it.
    """
    q, scale = symmetric_components(weight, group_size)
    return (q * scale).reshape(weight.shape)


def asymmetric_components(
    weight: torch.Tensor, group_size: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return `(u, scale, zero)` of asymmetric quantization (**continuous** zero-point) in group
    form — measurement only.

    `scale = clamp((max − min) / 14, f32 tiny)` / `zero = 1 − min/scale` /
    `u = clamp(round(w/scale + zero), 1, 15)`. The zero is neither rounded nor stored, so this is
    "the **upper bound** of the quality reachable by adding a zero-point field", not a storage form
    (reservation 3 of ADR 0069 decision 3 — a measurement column to detect a quality shortfall of
    symmetric before implementing it).

    MUST: a degenerate group (`max == min`) **falls back to the symmetric formula**
    (`scale = clamp(|c|/7, tiny)` / `zero = 8`). With the min-max formula `scale` sticks to the
    lower clamp, `w/scale` diverges, and a constant group is crushed to ~0 by the clamp (a silent
    wrong value where a non-zero constant group quietly becomes 0).
    """
    grouped = grouped_view(weight, group_size)
    tiny = torch.finfo(torch.float32).tiny
    upper = grouped.amax(dim=-1, keepdim=True)
    lower = grouped.amin(dim=-1, keepdim=True)
    degenerate = upper == lower
    span = torch.clamp((upper - lower) / (UINT4_MAX - UINT4_MIN), min=tiny)
    scale = torch.where(degenerate, torch.clamp(upper.abs() / INT4_MAX, min=tiny), span)
    zero = torch.where(
        degenerate,
        torch.full_like(span, float(DEFAULT_ZERO_POINT)),
        UINT4_MIN - lower / span,
    )
    u = torch.round(grouped / scale + zero).clamp_(UINT4_MIN, UINT4_MAX)
    return u, scale, zero


def fake_quant_asymmetric(weight: torch.Tensor, group_size: int) -> torch.Tensor:
    """Asymmetric 15-level fake-quant (returns `(u − zero)·scale` in the same shape as the weight).

    NOTE: unlike symmetric, **idempotence is not guaranteed** — restoring the min / max elements is
    looser by f32 rounding, and reapplying can move `scale` by 1ulp. It is a measurement column, so
    idempotence is not required (the storage form starts with symmetric only — ADR 0069
    decision 3).
    """
    u, scale, zero = asymmetric_components(weight, group_size)
    return ((u - zero) * scale).reshape(weight.shape)


# ---- Quantization targets ---------------------------------------------------


def family_of(fqn: str) -> str:
    """The fqn of a weight (`….q_proj.weight`) → the family name (the last module name)."""
    return fqn.split(".")[-2]


def linear_weights(model: nn.Module) -> dict[str, torch.Tensor]:
    """Collect the `weight` of every `nn.Linear` keyed by fqn (bias / norm / embedding are not
    touched).

    The fqn comes from `named_modules` (the same way as `quantize.fake_quant_int8` — not matched by
    `id(tensor)`. Parameter identity breaks easily under normalization, and when it breaks the
    targets just shrink silently).

    MUST: a linear whose family is not in {@link LINEAR_FAMILIES} fails loudly. If the transformers
    side's structure changed and the targets grew or shrank, measuring on silently would make the
    per-family table the numbers of another model.
    """
    weights: dict[str, torch.Tensor] = {}
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        fqn = f"{name}.weight"
        if family_of(fqn) not in LINEAR_FAMILIES:
            raise QuantizeError(
                f"'{fqn}': unknown linear family (the targets are {list(LINEAR_FAMILIES)})"
                " — the model's structure differs from what the sweep assumes"
            )
        weights[fqn] = module.weight
    if not weights:
        raise QuantizeError(
            "no nn.Linear at all (the model's structure differs from what the sweep assumes)"
        )
    return weights


def method_op_types(include_embedding: bool) -> tuple[type[nn.Module], ...]:
    """The method grid's 2 target sets → the `op_types` of `iter_quant_targets`.

    The "including non-linear" column adds only `nn.Embedding` because MiniCPM5-1B has no other
    quantizable type (conv family) at all — passing all 5 i8 types would give the same targets and
    only add the appearance of "passing a wide set of types".
    """
    return (nn.Linear, nn.Embedding) if include_embedding else (nn.Linear,)


def collect_targets(
    model: nn.Module, op_types: tuple[type[nn.Module], ...]
) -> tuple[dict[str, torch.Tensor], TargetCounts]:
    """Return the quantization targets as weights keyed by fqn plus counts (the target selection is
    shared with core).

    MUST: a target whose family is not in {@link METHOD_FAMILIES} fails loudly (the same reasoning
    as {@link linear_weights}). If the targets silently grew or shrank, the per-family table and
    the size estimate would be the numbers of another model.
    """
    weights: dict[str, torch.Tensor] = {}
    channels = 0
    elements = 0
    for fqn, weight, axis in iter_quant_targets(model, op_types):
        if family_of(fqn) not in METHOD_FAMILIES:
            raise QuantizeError(
                f"'{fqn}': unknown family (the targets are {list(METHOD_FAMILIES)})"
                " — the model's structure differs from what the sweep assumes"
            )
        weights[fqn] = weight
        channels += int(weight.shape[axis])
        elements += int(weight.numel())
    if not weights:
        raise QuantizeError(
            "no quantization targets at all (target types:"
            f" {', '.join(cls.__name__ for cls in op_types)})"
        )
    return weights, TargetCounts(modules=len(weights), channels=channels, elements=elements)


# ---- Size estimate ----------------------------------------------------------


@dataclass(frozen=True)
class SizeProjection:
    """The **projected** size of method × target set (not a measurement — the formula is
    {@link QuantMethod.formula})."""

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
        """MiB of the same target set held in f32 (the denominator of the projection — so the
        compression ratio can be read from the table)."""
        return self.counts.elements * 4 / 1024**2


def project_size(method: QuantMethod, counts: TargetCounts) -> SizeProjection:
    """Apply the method's formula to the counts of the target set."""
    return SizeProjection(counts=counts, bits=method.projected_bits(counts), formula=method.formula)


# ---- Applying a method ------------------------------------------------------


def apply_method(model: nn.Module, config: MethodConfig, shared_stride: int) -> None:
    """Apply the rounding of one method-grid run to the model in place.

    Only when `shared_stride > 1` does **the table fit** of `kmeans:shared` use an evenly spaced
    subsample (core's `fit_stride` — the application is always over everything, and it has no
    effect on the other methods). A full fit exceeds real memory at the 1B scale, so this is the
    escape hatch, and the fact that it was used goes into the JSON and the markdown footnote as is
    (the MUST of {@link karume.quant_methods.fake_quant_kmeans}).
    """
    op_types = method_op_types(config.include_embedding)
    if config.method.name == f"{KMEANS_PREFIX}shared" and shared_stride > 1:
        fake_quant_kmeans(
            model, "shared", METHOD_GROUP_SIZE, op_types=op_types, fit_stride=shared_stride
        )
        return
    config.method.round_weights(model, op_types, METHOD_GROUP_SIZE)


# ---- Driving calibrated rounding (Catcher + stage split) --------------------

#: The in-model FQN prefix of the decoder layers (`CausalLmWrapper.model` = `LlamaForCausalLM` →
#: `.model` = `LlamaModel` → `.layers`). The prefix is passed to `calibrate_stages` so that the
#: keys of the scale ledger live in the same **in-model FQN** space as
#: {@link karume.quantize.Int4Report} (core's `StageSpec` contract).
DECODER_PREFIX = "model.model.layers"


class _FirstStageReached(Exception):  # noqa: N818 — a stop signal, not an error, so not named Error
    """Signal that input reached the first stage (a sentinel to cut the calibration forward
    short)."""


def decoder_stages(wrapper: nn.Module) -> tuple[StageSpec, ...]:
    """Return the decoder layers in execution order as `(in-model FQN prefix, module)`.

    The final `norm` and `lm_head` are **not included** — the calibration targets only the linears
    inside the decoder, and adding `lm_head` as a stage would give the form "its input has not gone
    through the final norm". This is why the target set differs from the method grid's 169 tensors
    ({@link CalibConfig}).
    """
    layers = wrapper.model.model.layers
    return tuple((f"{DECODER_PREFIX}.{index}", layer) for index, layer in enumerate(layers))


def calib_targets(stages: Sequence[StageSpec]) -> tuple[dict[str, torch.Tensor], TargetCounts]:
    """Return the quantization targets inside the stages as weights keyed by fqn plus counts (the
    calibration's **scan** = the reference of the check).

    The target selection is shared with core's `iter_quant_targets` (the same reasoning as
    {@link collect_targets} — a copied implementation would let "the targets measured" and "the
    targets rounded" silently diverge). {@link assert_calib_covers_scan} compares the count taken
    here with the count the calibration actually rounded.
    """
    weights: dict[str, torch.Tensor] = {}
    channels = 0
    elements = 0
    for prefix, stage in stages:
        for local, weight, axis in iter_quant_targets(stage, (nn.Linear,)):
            fqn = f"{prefix}.{local}"
            if family_of(fqn) not in LINEAR_FAMILIES:
                raise QuantizeError(
                    f"'{fqn}': unknown linear family (the targets are {list(LINEAR_FAMILIES)})"
                    " — the model's structure differs from what the sweep assumes"
                )
            weights[fqn] = weight
            channels += int(weight.shape[axis])
            elements += int(weight.numel())
    if not weights:
        raise QuantizeError(
            "no nn.Linear at all in the decoder stages (the model's structure differs from what"
            " is assumed)"
        )
    return weights, TargetCounts(modules=len(weights), channels=channels, elements=elements)


def calib_inputs(model_dir: Path, limit: int | None) -> tuple[torch.Tensor, ...]:
    """Lower the calibration corpus to `[1,T]` id sequences through **the existing tokenizer path**
    ({@link minicpm5.export.load_tokenizer}).

    `limit` caps the number of leading texts (for a reduced smoke run — `None` means all 48 texts).
    They are taken from the start because {@link minicpm5.calib_texts.CALIB_TEXTS} is ordered with
    languages and styles mixed.
    """
    if limit is not None and limit < 1:
        raise ValueError(f"the cap on calibration texts must be 1 or more (got {limit})")
    texts = CALIB_TEXTS if limit is None else CALIB_TEXTS[:limit]
    tokenizer = one_shot.load_tokenizer(model_dir)
    return tuple(torch.tensor([tokenizer.encode(text).ids], dtype=torch.int64) for text in texts)


def capture_stage_batches(
    wrapper: nn.Module, first: nn.Module, inputs: Sequence[torch.Tensor]
) -> tuple[StageBatch, ...]:
    """Capture the `(args, kwargs)` into the first decoder layer with a forward_pre_hook
    (Catcher).

    The calibration input is "the hidden after embedding, the RoPE table and the additive causal
    mask, plus its accompanying arguments"; rebuilding it by hand would silently diverge from the
    transformers-side front end (and from the mask form the wrapper of `export.py` passes).
    Capturing **the call into the first stage itself** and cutting the forward short with a
    sentinel exception is the way to get the same input without copying the front end.

    The kwargs (mask / position_ids / position_embeddings …) are **identical for every layer** in
    Llama, so they ride on `calibrate_stages`'s default (carried as invariant across stages) as is.

    MUST: if the forward runs to completion without a capture, fail loudly — it is the sign that
    the stage spelling disagrees with the model's structure, and going on silently would only
    produce a "zero calibration inputs" diagnostic on the core side.
    """
    captured: list[StageBatch] = []

    def catcher(_module: nn.Module, args: tuple[object, ...], kwargs: dict[str, object]) -> None:
        captured.append((tuple(args), dict(kwargs)))
        raise _FirstStageReached

    handle = first.register_forward_pre_hook(catcher, with_kwargs=True)
    try:
        for index, ids in enumerate(inputs):
            try:
                with torch.no_grad():
                    wrapper(ids)
            except _FirstStageReached:
                continue
            raise AssertionError(
                f"with calibration input {index} the forward ran to completion without calling the"
                " first decoder layer (the stage spelling disagrees with the model's structure)"
            )
    finally:
        handle.remove()
    return tuple(captured)


@dataclass(frozen=True)
class CalibRig:
    """The scaffolding the calibrated configurations share (stage sequence, scan, inputs to the
    first stage).

    Not rebuilt per config — the input of the first stage is a hidden that has only gone through
    `embed_tokens`, and calibration rounds only the linears inside the decoder, so restoring the
    pristine weights in between does not move it (taken **once, before any rounding is applied**).
    """

    stages: tuple[StageSpec, ...]
    scan: Mapping[str, torch.Tensor]
    counts: TargetCounts
    batches: tuple[StageBatch, ...]


def assert_stage_split(
    wrapper: nn.Module,
    probe: torch.Tensor,
    batch: StageBatch,
    stages: Sequence[StageSpec],
) -> None:
    """Check that the sequential forward of the stage sequence is **bit-identical** to the model's
    final hidden.

    `probe` is the very id sequence used for the capture, and `batch` is the input to the first
    stage captured with **the same input**. The stage sequence does not include the final `norm`
    and `lm_head` ({@link decoder_stages}), so `norm` is applied to the sequential result before
    comparing with `LlamaModel`'s `last_hidden_state` (the wrapper's output is logits and cannot be
    compared directly). The stages are driven the same way as core (`_advance_batch`) — the only
    positional argument is the hidden, and the kwargs are invariant across stages.

    MUST: measured before rounding, and compared with `torch.equal`, not an approximate tolerance —
    with the same modules called in the same order, a mismatch means the split takes a path
    different from the real one. Calibration chooses rounding targets from "the activations that
    actually flow into the layer", so a drifting split would decide the adoption of a method by a
    measurement of a different path (the numbers come out normally, so the table does not show it).
    """
    inner = wrapper.model.model
    args, kwargs = batch
    with torch.no_grad():
        reference = inner(
            input_ids=probe,
            attention_mask=one_shot.additive_causal_mask(int(probe.shape[1])),
            use_cache=False,
        ).last_hidden_state
        hidden = args[0]
        for _prefix, stage in stages:
            hidden = first_tensor_output(stage(hidden, **kwargs))
        hidden = inner.norm(hidden)
    if not torch.equal(hidden, reference):
        raise AssertionError(
            "the final hidden of the stage split is not bit-identical to the model's"
            " last_hidden_state (max absolute difference"
            f" {float((hidden - reference).abs().max()):.4e})"
            " — the decoder stage sequence has drifted from the path of LlamaModel.forward"
        )


def build_calib_rig(wrapper: nn.Module, model_dir: Path, limit: int | None) -> CalibRig:
    """Build the calibration scaffolding (stage split → scan → tokenize → Catcher → split
    check)."""
    stages = decoder_stages(wrapper)
    scan, counts = calib_targets(stages)
    inputs = calib_inputs(model_dir, limit)
    batches = capture_stage_batches(wrapper, stages[0][1], inputs)
    assert_stage_split(wrapper, inputs[0], batches[0], stages)
    return CalibRig(stages=stages, scan=scan, counts=counts, batches=batches)


def apply_calib(config: CalibConfig, rig: CalibRig, shared_stride: int) -> CalibReport:
    """Apply one calibrated rounding to the model in place (the stage-sequential driving is on the
    core side).

    Only when `shared_stride > 1` does **the table fit** of `kmeans_shared` use an evenly spaced
    subsample (the same escape hatch and the same knob as {@link apply_method}). The fact that it
    was used is picked up by {@link karume.quant_calib.CalibReport.describe} and goes into the
    output.
    """
    spec = config.grid
    if spec.kind == "kmeans_shared" and shared_stride > 1:
        spec = replace(spec, fit_stride=shared_stride)
    return calibrate_stages(
        rig.stages,
        rig.batches,
        method=config.method,
        spec=spec,
        static_groups=config.static_groups,
        act_order=config.act_order,
        damping=config.damping,
    )


def assert_calib_covers_scan(
    report: CalibReport, scan: Mapping[str, torch.Tensor], name: str
) -> None:
    """Check that the layers the calibration rounded match the stage scan **exactly**.

    MUST: fail loudly. If the stage spelling or the target types change and part of the decoder
    drops out of the calibration, the table still has its row, and the quality even comes out on
    **the good side** by the unrounded share (a pass-through cannot be read from the numbers).
    """
    rounded = {layer.fqn for layer in report.layers}
    missing = sorted(set(scan) - rounded)
    extra = sorted(rounded - set(scan))
    if missing or extra or report.modules != len(scan):
        raise AssertionError(
            f"[{name}] the {report.modules} tensors the calibration rounded do not match the"
            f" {len(scan)} of the scan (not rounded {missing[:3]} / not in the scan {extra[:3]})"
        )


# ---- Expected sequences (the greedy assets of export_decode) -----------------


@dataclass(frozen=True)
class GreedyCase:
    """One case of `export_decode`'s greedy expected sequences (`prompt[1,T]` / `expected[K]` —
    both i64)."""

    name: str
    prompt: torch.Tensor
    expected: torch.Tensor

    @property
    def steps(self) -> int:
        return int(self.expected.numel())


def load_cases(decode_dir: Path) -> tuple[GreedyCase, ...]:
    """Read `greedy.<case>.safetensors` by glob (no tokenizer involved).

    Without the assets it fails **with the steps to produce them** — at the entrance of a
    60-minute-class run, a failure that can be fixed is not returned as a bare "file not found".
    """
    paths = sorted(decode_dir.glob(f"{GREEDY_PREFIX}*{GREEDY_SUFFIX}"))
    if not paths:
        raise FileNotFoundError(
            f"no greedy expected sequences at all in {decode_dir} — build the decode assets"
            " first:\n"
            "  cd tools/export-recipes && "
            "uv run --with 'transformers==5.14.1' python -m minicpm5.export_decode"
        )
    cases: list[GreedyCase] = []
    for path in paths:
        name = path.name[len(GREEDY_PREFIX) : -len(GREEDY_SUFFIX)]
        tensors = load_file(str(path))
        cases.append(
            GreedyCase(
                name=name,
                prompt=tensors[PROMPT_KEY].to(torch.int64).unsqueeze(0),
                expected=tensors[EXPECTED_KEY].to(torch.int64),
            )
        )
    return tuple(cases)


# ---- Measurement ------------------------------------------------------------


@dataclass(frozen=True)
class CaseResult:
    """The measurements of 1 config × 1 case."""

    steps: int
    matches: int
    nll: float
    prefix: int


@dataclass(frozen=True)
class ConfigResult:
    """The measurements of one config (the per-family wRMSE holds only the quantized families).

    `overall_rmse` is `‖w − fq‖₂ / ‖w‖₂` over the whole target set as one (not a family average —
    the element counts of the families differ 20-fold, so an average would weight them
    arbitrarily). Only the method and calibration grids have `size` (the Phase 0 configs have a
    single storage form, i4, so an estimate would list the same number).
    """

    name: str
    weight_rmse: Mapping[str, float]
    cases: Mapping[str, CaseResult]
    seconds: float = 0.0
    overall_rmse: float | None = None
    #: Method name shown in the tables and the JSON (`None` for Phase 0 configs, which have no
    #: method axis).
    method: str | None = None
    size: SizeProjection | None = None
    #: One-line summary of a calibrated configuration
    #: ({@link karume.quant_calib.CalibReport.describe} — the fact that `fit_stride` was used and
    #: the range of AWQ's α also appear here).
    calib: str | None = None


def teacher_forced(wrapper: nn.Module, case: GreedyCase) -> tuple[int, float]:
    """Run `cat(prompt, expected)` in one forward and return `(matches, mean NLL)`.

    The K rows read are positions `T−1 … T+K−2` — "the next token when expected is fed one at a
    time from the end of the prompt"; unlike free running, **the sequence does not shift after a
    divergence**, so the loss at the same positions can be compared across configs. The NLL is the
    mean of `−log softmax(logits)[expected]` over the same K rows and picks up the loss before the
    top-1 changes (the rank holds but the confidence thins).
    """
    ids = torch.cat([case.prompt, case.expected.unsqueeze(0)], dim=1)
    with torch.no_grad():
        logits = wrapper(ids)
    start = int(case.prompt.shape[1]) - 1
    rows = logits[0, start : start + case.steps]
    matches = int((rows.argmax(dim=-1) == case.expected).sum())
    log_probs = torch.log_softmax(rows.to(torch.float32), dim=-1)
    nll = float(-log_probs.gather(1, case.expected.unsqueeze(1)).mean())
    return matches, nll


def greedy_prefix(wrapper: nn.Module, case: GreedyCase) -> int:
    """The number of leading steps for which free-running greedy keeps matching expected (K when
    all match).

    MUST: full re-forward (every step computed from the start with standard causal — the same way
    as `export_decode .greedy_continuation`). Taking it through the KV cache path would mix the
    quantization loss with the effect of the cache implementation. It stops at the divergence — the
    sequence after that is "a different text" and meaningless to compare, and the more degraded a
    config, the sooner it finishes.
    """
    current = case.prompt
    for step in range(case.steps):
        with torch.no_grad():
            logits = wrapper(current)
        token = int(logits[0, -1].argmax())
        if token != int(case.expected[step]):
            return step
        current = torch.cat([current, torch.tensor([[token]], dtype=torch.int64)], dim=1)
    return case.steps


def restore_pristine(
    weights: Mapping[str, torch.Tensor], pristine: Mapping[str, torch.Tensor]
) -> None:
    """Restore every target to pristine (the plain f32).

    MUST: restore **for every config**. Applying without restoring would measure "NF4 on top of
    RTN", and the comparison between configs would lose its meaning (methods are not stacked). The
    restored range is **the union** of Phase 0 and the method grid (linear + `embed_tokens`) — so
    that weights one grid touched do not remain in the other's measurements.
    """
    with torch.no_grad():
        for fqn, weight in weights.items():
            weight.copy_(pristine[fqn])


def apply_config(weights: Mapping[str, torch.Tensor], config: SweepConfig) -> list[str]:
    """Apply Phase 0's fake-quant and return the quantized fqns (the caller restores)."""
    if config.group_size is None:
        return []
    quantize = fake_quant_asymmetric if config.asymmetric else fake_quant_symmetric
    touched: list[str] = []
    with torch.no_grad():
        for fqn, weight in weights.items():
            if family_of(fqn) == LM_HEAD and not config.include_lm_head:
                continue
            weight.copy_(quantize(weight, config.group_size))
            touched.append(fqn)
    return touched


def measure_rmse(
    weights: Mapping[str, torch.Tensor],
    pristine: Mapping[str, torch.Tensor],
    touched: Sequence[str],
) -> tuple[dict[str, float], float | None]:
    """Take the per-family + overall relative RMSE from the rounded weights and pristine.

    The relative RMSE is `‖w − fq‖₂ / ‖w‖₂` over all layers of a family (the element counts cancel,
    so it is not weighted by layer size). The sums are taken in f64 — some families add 200 million
    elements, and with f32 accumulation the cancellation would eat the meaning of the numbers.
    """
    errors: dict[str, float] = defaultdict(float)
    norms: dict[str, float] = defaultdict(float)
    with torch.no_grad():
        for fqn in touched:
            family = family_of(fqn)
            difference = weights[fqn] - pristine[fqn]
            errors[family] += float(difference.pow(2).sum(dtype=torch.float64))
            norms[family] += float(pristine[fqn].pow(2).sum(dtype=torch.float64))
    if not errors:
        return {}, None
    families = {family: math.sqrt(errors[family] / norms[family]) for family in errors}
    return families, math.sqrt(sum(errors.values()) / sum(norms.values()))


def assert_baseline_reproduces(
    cases: Sequence[GreedyCase], measured: Mapping[str, CaseResult]
) -> None:
    """Check that the baseline (still f32) reproduces the expected sequences **completely** under
    teacher forcing.

    MUST: fail loudly. A split here is not a quantization matter but the sign that **the harness is
    broken** (mixed-up weights, shifted positions, expected sequences that do not match the model),
    and the numbers of every later config lose their meaning. This check is why the baseline always
    runs first even in a partial rerun (`--only`).
    """
    wrong = {
        case.name: f"{measured[case.name].matches}/{case.steps}"
        for case in cases
        if measured[case.name].matches != case.steps
    }
    if wrong:
        raise AssertionError(
            f"the baseline's teacher-forced agreement is not complete {wrong}"
            " (a problem before quantization — suspect the correspondence of model, expected"
            " sequences and positions)"
        )


def select_configs(only: Sequence[str]) -> tuple[AnyConfig, ...]:
    """Return the configs chosen with `--only` in declaration order (**the baseline always runs
    first**).

    The baseline cannot be left out so that even a partial rerun always goes through the sanity
    check ({@link assert_baseline_reproduces}) — the more a 60-minute-class run is cut into pieces,
    the more it matters to leave no path that "keeps measuring with a broken model".
    """
    if not only:
        return ALL_CONFIGS
    chosen = {BASELINE_NAME, *only}
    return tuple(config for config in SELECTABLE_CONFIGS if config.name in chosen)


def run_sweep(
    model_dir: Path,
    configs: Sequence[AnyConfig],
    *,
    decode_dir: Path = DEFAULT_DECODE_DIR,
    kmeans_shared_stride: int = 0,
    calib_limit: int | None = None,
    json_path: Path | None = None,
) -> list[ConfigResult]:
    """Load the model once, then apply and measure the configs in order.

    A pristine clone (CPU, ~3.5GB for the 169 linears, ~4.3GB including `embed_tokens`) is taken
    first, so each config's rounding always starts from the original weights. The saved range is
    **always the union** so that a Phase 0 config cannot create a form that leaves the embedding
    unrestored.

    The scaffolding of the calibrated configurations ({@link CalibRig}) is built **only once, the
    first time it is requested** — the pristine restore has already run at the start of each
    config, so the first-stage input taken there belongs to the plain weights. If no calibration
    config is chosen, neither the tokenizer nor the Catcher runs.

    Progress is flushed to stderr after every case / config, and with `json_path` the JSON is
    rewritten after every config (if a k-means run of tens of minutes dies halfway, the
    measurements remain).
    """
    cases = load_cases(decode_dir)
    lengths = " ".join(f"{case.name}(T={case.prompt.shape[1]},K={case.steps})" for case in cases)
    print(f"[sweep] expected sequences, {len(cases)} cases: {lengths}", file=sys.stderr, flush=True)

    wrapper = one_shot.load_wrapper(model_dir)
    linear = linear_weights(wrapper)
    union, counts = collect_targets(wrapper, method_op_types(include_embedding=True))
    print(
        f"[sweep] saving pristine copies of {len(linear)} target linears / a union of"
        f" {counts.modules} tensors ({counts.elements:,} elements)",
        file=sys.stderr,
        flush=True,
    )
    pristine = {fqn: weight.detach().clone() for fqn, weight in union.items()}

    results: list[ConfigResult] = []
    rig: CalibRig | None = None
    for config in configs:
        # Release the previous config's scratch space (k-means works in f64 on the order of the
        # targets) before the next rounding. If GB-class tensors caught in reference cycles wait for
        # generational GC, the next config breaks through real memory (refcounting alone is not
        # enough in places, so it is explicit).
        gc.collect()
        started = time.perf_counter()
        restore_pristine(union, pristine)
        method: str | None = None
        size: SizeProjection | None = None
        calib: str | None = None
        if isinstance(config, SweepConfig):
            touched = apply_config(linear, config)
        elif isinstance(config, MethodConfig):
            method = config.method.name
            targets, target_counts = collect_targets(
                wrapper, method_op_types(config.include_embedding)
            )
            apply_method(wrapper, config, kmeans_shared_stride)
            touched = list(targets)
            size = project_size(config.method, target_counts)
        else:
            if rig is None:
                rig = build_calib_rig(wrapper, model_dir, calib_limit)
                print(
                    f"[sweep] calibration: {len(rig.batches)} texts / {len(rig.stages)} stages /"
                    f" {rig.counts.modules} target linears"
                    " (inside the decoder only — lm_head / embed_tokens not included)",
                    file=sys.stderr,
                    flush=True,
                )
            method = config.label
            report = apply_calib(config, rig, kmeans_shared_stride)
            assert_calib_covers_scan(report, rig.scan, config.name)
            touched = list(rig.scan)
            calib = report.describe()
            size = SizeProjection(
                counts=rig.counts,
                bits=config.projected_bits(rig.counts),
                formula=config.formula,
            )
            print(f"[{config.name}] {calib}", file=sys.stderr, flush=True)
        weight_rmse, overall = measure_rmse(union, pristine, touched)
        rounded = time.perf_counter() - started
        overall_text = "-" if overall is None else f"{overall:.4g}"
        print(
            f"[{config.name}] rounded {len(touched)} tensors {rounded:.1f}s wRMSE {overall_text}",
            file=sys.stderr,
            flush=True,
        )
        measured: dict[str, CaseResult] = {}
        for case in cases:
            matches, nll = teacher_forced(wrapper, case)
            prefix = greedy_prefix(wrapper, case)
            measured[case.name] = CaseResult(
                steps=case.steps, matches=matches, nll=nll, prefix=prefix
            )
            print(
                f"[{config.name}] {case.name} teacher {matches}/{case.steps}"
                f" nll {nll:.4g} greedy {prefix}/{case.steps}",
                file=sys.stderr,
                flush=True,
            )
        if config.name == BASELINE_NAME:
            assert_baseline_reproduces(cases, measured)
        results.append(
            ConfigResult(
                name=config.name,
                weight_rmse=weight_rmse,
                cases=measured,
                seconds=time.perf_counter() - started,
                overall_rmse=overall,
                method=method,
                size=size,
                calib=calib,
            )
        )
        print(
            f"[{config.name}] done {time.perf_counter() - started:.1f}s",
            file=sys.stderr,
            flush=True,
        )
        if json_path is not None:
            json_path.write_text(
                results_json(results, cases, kmeans_shared_stride, calib_limit), encoding="utf-8"
            )
    return results


# ---- Tables -----------------------------------------------------------------


def _markdown(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = [f"| {' | '.join(header)} |", f"| {' | '.join('---' for _ in header)} |"]
    lines += [f"| {' | '.join(row)} |" for row in rows]
    return "\n".join(lines)


def quality_table(results: Sequence[ConfigResult]) -> str:
    """Table of config × per-case [teacher agreement / NLL / divergence step]."""
    names = list(results[0].cases)
    header = ["config"]
    for name in names:
        header += [f"{name} match", f"{name} NLL", f"{name} diverge"]
    rows = []
    for result in results:
        row = [result.name]
        for name in names:
            case = result.cases[name]
            row += [
                f"{case.matches}/{case.steps}",
                f"{case.nll:.4g}",
                f"{case.prefix}/{case.steps}",
            ]
        rows.append(row)
    return _markdown(header, rows)


def rmse_table(results: Sequence[ConfigResult]) -> str:
    """Table of config × per-family relative weight RMSE (`-` for families not quantized)."""
    rows = [
        [result.name]
        + [
            f"{result.weight_rmse[family]:.4g}" if family in result.weight_rmse else "-"
            for family in METHOD_FAMILIES
        ]
        for result in results
    ]
    return _markdown(["config", *METHOD_FAMILIES], rows)


def summary_table(results: Sequence[ConfigResult]) -> str:
    """Summary table reading method × 2 target sets one config per row (cases summed).

    The sums are "the matches over 3 cases × 16 positions = 48 points", "the plain sum of NLL over
    the 3 cases" and "the sum over the 3 cases of the free-running prefix length" — the per-case
    breakdown stays in {@link quality_table}.
    """
    header = [
        "config",
        "method",
        "targets",
        "bpw",
        "projected MiB",
        "wRMSE",
        "teacher",
        "NLL sum",
        "greedy",
    ]
    rows = []
    for result in results:
        cases = list(result.cases.values())
        total = sum(case.steps for case in cases)
        rows.append(
            [
                result.name,
                result.method or "-",
                "-" if result.size is None else f"{result.size.counts.modules} tensors",
                "-" if result.size is None else f"{result.size.bits_per_weight:.4g}",
                "-" if result.size is None else f"{result.size.projected_mib:.1f}",
                "-" if result.overall_rmse is None else f"{result.overall_rmse:.4g}",
                f"{sum(case.matches for case in cases)}/{total}",
                f"{sum(case.nll for case in cases):.4g}",
                f"{sum(case.prefix for case in cases)}/{total}",
            ]
        )
    return _markdown(header, rows)


def size_table(results: Sequence[ConfigResult]) -> str:
    """Size estimate of method × target set (**a projection by formula**, not a measurement)."""
    header = [
        "config",
        "tensors",
        "channels",
        "elements",
        "formula",
        "bpw",
        "projected MiB",
        "f32 MiB",
    ]
    rows = [
        [
            result.name,
            f"{result.size.counts.modules}",
            f"{result.size.counts.channels:,}",
            f"{result.size.counts.elements:,}",
            result.size.formula,
            f"{result.size.bits_per_weight:.4g}",
            f"{result.size.projected_mib:.1f}",
            f"{result.size.f32_mib:.1f}",
        ]
        for result in results
        if result.size is not None
    ]
    return _markdown(header, rows)


def results_json(
    results: Sequence[ConfigResult],
    cases: Sequence[GreedyCase],
    kmeans_shared_stride: int,
    calib_limit: int | None = None,
) -> str:
    """Machine-readable measurements (every column of the markdown tables + counts the tables do
    not show).

    `calib_limit` records **whether the calibration was a reduced run** (`None` = full) — kept in
    the same file as the measurements so that reduced numbers are not read as full ones.
    """
    payload = {
        "group_size": METHOD_GROUP_SIZE,
        "kmeans_shared_stride": kmeans_shared_stride,
        "calib_limit": calib_limit,
        "cases": [
            {"name": case.name, "prompt_length": int(case.prompt.shape[1]), "steps": case.steps}
            for case in cases
        ],
        "configs": [
            {
                "name": result.name,
                "method": result.method,
                "calib": result.calib,
                "seconds": round(result.seconds, 3),
                "weight_rmse": dict(result.weight_rmse),
                "overall_rmse": result.overall_rmse,
                "cases": {
                    name: {
                        "steps": case.steps,
                        "matches": case.matches,
                        "nll": case.nll,
                        "prefix": case.prefix,
                    }
                    for name, case in result.cases.items()
                },
                "size": None
                if result.size is None
                else {
                    "modules": result.size.counts.modules,
                    "channels": result.size.counts.channels,
                    "elements": result.size.counts.elements,
                    "formula": result.size.formula,
                    "bits_per_weight": result.size.bits_per_weight,
                    "projected_mib": result.size.projected_mib,
                    "f32_mib": result.size.f32_mib,
                },
            }
            for result in results
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--decode-dir", type=Path, default=DEFAULT_DECODE_DIR)
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        choices=[config.name for config in SELECTABLE_CONFIGS],
        help="Run only this config (repeatable, for partial reruns). The baseline always runs"
        " first. The GPTQ sweep-axis experiment points (EXPERIMENT_CONFIGS) run only when named"
        " here.",
    )
    parser.add_argument(
        "--json", type=Path, default=None, help="Where to write the machine-readable measurements."
    )
    parser.add_argument(
        "--kmeans-shared-stride",
        type=int,
        default=0,
        help="Take only the table fit of kmeans:shared from an evenly strided subsample (0 / 1 ="
        " everything). An escape hatch for machines where the full set does not fit in real"
        " memory; the rounding is always applied to everything.",
    )
    parser.add_argument(
        "--calib-limit",
        type=int,
        default=None,
        help="Use only the first N texts of the calibration corpus (default: all"
        f" {len(CALIB_TEXTS)})."
        " A knob for reduced smoke runs; it affects only the calibrated configurations.",
    )
    args = parser.parse_args(argv)
    results = run_sweep(
        args.model_dir,
        select_configs(args.only),
        decode_dir=args.decode_dir,
        kmeans_shared_stride=args.kmeans_shared_stride,
        calib_limit=args.calib_limit,
        json_path=args.json,
    )
    if args.kmeans_shared_stride > 1:
        print(
            f"> the kmeans:shared table is fit on a 1/{args.kmeans_shared_stride} subsample"
            " (applied to everything)"
        )
        print()
    if args.calib_limit is not None:
        print(
            f"> calibration corpus: first {args.calib_limit} texts only (a reduced run of all"
            f" {len(CALIB_TEXTS)})"
        )
        print()
    print(summary_table(results))
    print()
    print(quality_table(results))
    print()
    print(rmse_table(results))
    print()
    print(size_table(results))
    notes = [f"> {result.name}: {result.calib}" for result in results if result.calib]
    if notes:
        print()
        print("Breakdown of the calibrated configurations (`CalibReport.describe`):")
        print("\n".join(notes))


if __name__ == "__main__":
    main()
