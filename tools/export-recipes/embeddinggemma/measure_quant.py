"""Measure the **quantization methods** for EmbeddingGemma-300m side by side on torch CPU (method
screening).

A quality axis with the same idea as `anima/measure_quant.py` / `irodori/measure_quant.py` — **zero
lines of GPU code**; what is measured here is the quality of the quantization itself (under the
fake-quant methodology of ADR 0006 the E2E measures only implementation error, so quality has to be
measured on a separate axis). Two questions are to be answered:

    ① **Does the way the 4-bit grid is laid out change the quality of the embedding** — RTN(i4) /
       FP4 / NF4 / MXFP4 / k-means codebook at 3 granularities, side by side with the same group
       (g=32 fixed) and the same targets.
    ② **Can the vocabulary table go down to 4 bits** — in EG `embed_tokens` takes 201M of 307M
       (about 66% of the distribution size), so the gain from quantization concentrates there. The
       targets are run in 2 forms, "linear only" and "linear + embedding", and the difference is
       read as the vocabulary table's contribution.

NOTE: i4 on the `linear+embedding` target is **a form that can be stored**. The ops eligible for i4
are the weight slots of linear / embedding / conv1d (ADR 0069 addenda 6 / 7,
`karume.emit.I4_WEIGHT_OPS`), stored with the codec `int4-sym-g` of `docs/container-v1.md` §6.3.
What is measured here, though, is a side-by-side comparison of method quality, not a claim about a
quant preset to ship (FP4 / NF4 / MXFP4 / k-means are measurement-only methods with no storage form
— the same standing as the measurement-only methods of `quant_methods`).

## Configs (8 = baseline + 7 methods, × 2 targets — the baseline is target-independent, so 1)

| method           | rounding                                                         |
| ---------------- | ---------------------------------------------------------------- |
| `f32`            | no rounding (**baseline** — every cosine is against this embedding) |
| `rtn-i4-g32`     | `karume.quantize.fake_quant_int4` (exactly the shipped storage)  |
| `fp4`            | fixed FP4 (e2m1) table × group absmax                            |
| `nf4`            | fixed NF4 table × group absmax                                   |
| `mxfp4`          | FP4 table × a **power-of-two** group scale (OCP MX)              |
| `kmeans:*`       | 16-centroid codebook (per_tensor / per_channel / shared)         |

group is **fixed at g=32** (to narrow the comparison between methods to the grid layout alone — the
group length axis is already covered by the sweep of ADR 0069 Phase 0).

## Calibrated rounding ({@link CALIB_CONFIGS})

The 7 methods above round by looking at the weights alone (RTN family). **Calibrated rounding**
(GPTQ / AWQ — `karume.quant_calib` in core) re-chooses the rounding targets within the same grid
from "the activations that actually flow into that layer"; 5 of them (GPTQ × 3 storage grids / AWQ /
AWQ+GPTQ) are added to the same measurement columns.

The target is **only 1 form, linear inside the decoder** (no combination including `embed_tokens`)
— the calibration driver (`calibrate_stages`) is closed over the `nn.Linear` inside a stage, and the
2 Dense stages are outside the decoder, so this is a separate set with 2 fewer than the `linear`
target above ({@link CALIB_TARGET}). The calibration inputs are the 48 texts of
{@link embeddinggemma.calib_texts.CALIB_TEXTS}; the call into the first decoder layer is captured
together with the mask / RoPE tables per layer_type and fed into the stage sequence
({@link capture_stage_batches}). Before feeding, the **stage-split agreement check**
({@link assert_stage_split}) measures bit-exactly that "the sequential forward of the stage
sequence = the body's final hidden" — a split that drifts still produces ordinary numbers, so
without the check the choice of method would be decided by a measurement on another path.

MUST: **do not stack methods** — every rounding is in place, so each config goes back to pristine
(the plain f32 weights) before being applied ({@link restore}). Applying without restoring measures
"MXFP4 on top of NF4", and the comparison between methods itself loses its meaning. The model is
loaded only once.

## Measurement columns

- **Per-case cosine** (the 5 golden cases of `export.py`) — min and mean of the cosine against the
  baseline f32 embedding. The embeddings are L2-normalised, so this is the dot product itself, but
  it is taken with normalisation.
- **Preservation of the semantic order** — whether the cosine of the near pair
  ({@link export.NEAR_PAIR}) beats the far pair ({@link export.FAR_PAIR}). The same pairs as
  `export._sanity`; a column that catches "the numbers move but it is broken as an embedding"
  independently of the cosine against the baseline (even at a cosine of 0.99, a broken order breaks
  retrieval).
- **Drift of the pairwise cosine matrix** — the max absolute difference of the cosines of all 10
  pairs of the 5 cases against f32. Embeddings are used for "comparing vectors with each other", so
  this is closer to real use than closeness to the baseline vectors (if every vector rotates the
  same way, retrieval ranking does not move).
- **Size projection** — effective bpw per method and projected MiB of the target set (the formula
  is {@link Method.formula}).

## Output

    uv run --with 'transformers==5.14.1' python -m embeddinggemma.measure_quant --out <dir>

All numbers go to `<out>/report.json`, and 2 Markdown tables (quality / size projection) to stdout.
Progress goes to stderr, flushed immediately (to follow a background run of tens of minutes).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path
from typing import Any

import torch
from torch import nn

from _shared.paths import BENCH_ROOT
from karume.quant_calib import (
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
from karume.quantize import channel_rows, fake_quant_int4, iter_quant_targets

from . import export as eg
from .calib_texts import CALIB_TEXTS

#: The default location of the artifacts (artifacts of demos and measurements are kept apart from
#: the series — the same as `irodori/measure_quant.py`).
DEFAULT_OUT = BENCH_ROOT / "embeddinggemma-300m" / f"{date.today().isoformat()}_quant-screen"

REPORT_FILE = "report.json"

#: The spelling of the baseline config (no rounding — the denominator of every comparison).
BASE_CONFIG = "f32"

#: The group length fixed for every method (fixed to narrow the comparison between methods to "the
#: grid layout" alone — module docstring). The group length axis itself belongs to the sweep of
#: ADR 0069 Phase 0.
GROUP_SIZE = 32

# ---- Bit widths of the size projection (verbatim from the storage rules) ------------------
#
# The sources are the codec `int4-sym-g` of `docs/container-v1.md` §6.3 (the scale is **F32**, one
# per group, groupSize a power of 2 and at least 16) and OCP Microscaling Formats v1.0 (the shared
# scale of MX is E8M0 = a 1-byte exponent).
# k-means is a measurement-only method with no storage form, so it is counted plainly **including
# the cost of the tables**.

#: The payload of the 4-bit grid (common to every method — what is compared is "the grid layout",
#: not the bit count).
PAYLOAD_BITS = 4.0
#: The bit width of a group scale (storing `int4-sym-g` MUST use an F32 group scale —
#: `docs/container-v1.md` §6.3).
F32_SCALE_BITS = 32.0
#: The shared scale of MXFP4 is E8M0 (a 1-byte exponent).
MX_SCALE_BITS = 8.0
#: The bit width of one codebook entry (the centroids are kept in F32).
CODEBOOK_ENTRY_BITS = 32.0

BITS_PER_BYTE = 8
MIB = 1024 * 1024
#: Bits per element of the baseline (no rounding).
F32_BITS = 32.0


@dataclass(frozen=True)
class TargetStats:
    """Properties of the target tensor set (the input of the size projection).

    `groups` is the total number of g=32 groups, and `channels` the total number of rows the cost of
    a per-channel table is proportional to (= the rows of {@link karume.quantize.channel_rows} =
    output channels).
    """

    modules: int
    elements: int
    groups: int
    channels: int


@dataclass(frozen=True)
class Target:
    """The quantization target range (switched with `op_types` — target selection's source of truth
    is in core)."""

    name: str
    op_types: tuple[type[nn.Module], ...]


#: The transformer's linear + the 2 SentenceTransformer Dense stages (the form that applies i4 only
#: to linear weight slots).
TARGET_LINEAR = Target("linear", (nn.Linear,))
#: The above plus `embed_tokens` (the dominant term of EG's distribution size — ② of the module
#: docstring).
TARGET_WITH_EMBEDDING = Target("linear+embedding", (nn.Linear, nn.Embedding))
TARGETS: tuple[Target, ...] = (TARGET_LINEAR, TARGET_WITH_EMBEDDING)

#: What the pristine backup and restore cover (taken in **the widest form** — configs with a
#: narrower target are restored from the same backup).
WIDEST_OP_TYPES = TARGET_WITH_EMBEDDING.op_types


def _group_scaled_bits(scale_bits: float) -> Callable[[TargetStats], float]:
    """Total bits of a method with group scales (`4·N + scale_bits·G`)."""

    def bits(stats: TargetStats) -> float:
        return PAYLOAD_BITS * stats.elements + scale_bits * stats.groups

    return bits


def _codebook_bits(
    tables: Callable[[TargetStats], int], scale_bits: float
) -> Callable[[TargetStats], float]:
    """Total bits of a codebook method (`4·N + scale_bits·G + 32·16·number of tables`)."""

    def bits(stats: TargetStats) -> float:
        table_bits = CODEBOOK_ENTRY_BITS * DEFAULT_CODEBOOK_LEVELS * tables(stats)
        return PAYLOAD_BITS * stats.elements + scale_bits * stats.groups + table_bits

    return bits


@dataclass(frozen=True)
class Method:
    """The recipe of one method.

    `apply` applies the rounding **to the plain f32 weights** and returns a one-line summary of the
    counts (the ban on stacking is guaranteed by the caller's {@link restore}). `bits` is the size
    projection, and `formula` is its formula verbatim, in a form the reader can verify.
    """

    name: str
    apply: Callable[[nn.Module, tuple[type[nn.Module], ...]], str]
    bits: Callable[[TargetStats], float]
    formula: str


METHODS: tuple[Method, ...] = (
    Method(
        "rtn-i4-g32",
        lambda model, op_types: fake_quant_int4(model, GROUP_SIZE, op_types=op_types).describe(),
        _group_scaled_bits(F32_SCALE_BITS),
        "4·N + 32·G (one F32 scale per group, G = N/32)",
    ),
    Method(
        "fp4",
        lambda model, op_types: fake_quant_fp4(model, GROUP_SIZE, op_types=op_types).describe(),
        _group_scaled_bits(F32_SCALE_BITS),
        "4·N + 32·G (the same group absmax scale as i4; only the grid is exponential)",
    ),
    Method(
        "nf4",
        lambda model, op_types: fake_quant_nf4(model, GROUP_SIZE, op_types=op_types).describe(),
        _group_scaled_bits(F32_SCALE_BITS),
        "4·N + 32·G (as above; the grid is the quantiles of a normal distribution)",
    ),
    Method(
        "mxfp4",
        lambda model, op_types: fake_quant_mxfp4(model, GROUP_SIZE, op_types=op_types).describe(),
        _group_scaled_bits(MX_SCALE_BITS),
        "4·N + 8·G (the shared scale is an E8M0 power of two — OCP MX v1.0)",
    ),
    Method(
        "kmeans:per_tensor",
        lambda model, op_types: fake_quant_kmeans(
            model, "per_tensor", GROUP_SIZE, op_types=op_types
        ).describe(),
        _codebook_bits(lambda stats: stats.modules, 0.0),
        "4·N + 32·16·(number of tensors) (no scale; one table per layer)",
    ),
    Method(
        "kmeans:per_channel",
        lambda model, op_types: fake_quant_kmeans(
            model, "per_channel", GROUP_SIZE, op_types=op_types
        ).describe(),
        _codebook_bits(lambda stats: stats.channels, 0.0),
        "4·N + 32·16·(number of output channels) (no scale; one table per row)",
    ),
    Method(
        "kmeans:shared",
        lambda model, op_types: fake_quant_kmeans(
            model, "shared", GROUP_SIZE, op_types=op_types
        ).describe(),
        _codebook_bits(lambda _stats: 1, F32_SCALE_BITS),
        "4·N + 32·G + 32·16 (one table for the whole model, with a group absmax scale)",
    ),
)

METHOD_NAMES = tuple(method.name for method in METHODS)

#: The order of the case names (the golden's declaration order — also the row/column order of the
#: pair matrix).
CASE_NAMES: tuple[str, ...] = tuple(name for name, _prompt, _body in eg.GOLDEN_CASES)


def config_name(method: Method, target: Target) -> str:
    return f"{method.name}/{target.name}"


# ---- Calibrated rounding (work item J-2) ---------------------------------------------

#: The target name of the calibrated configs (the "target" column of the table). **A different
#: target set** from the 2 `linear` forms — the calibration driver
#: ({@link karume.quant_calib.calibrate_stages}) is closed over the `nn.Linear` inside a stage, so
#: the 2 Dense stages outside the decoder (`2_Dense` / `3_Dense`) are not included.
CALIB_TARGET = "linear:decoder"


@dataclass(frozen=True)
class CalibConfig:
    """The specification of one calibrated rounding (method × storage grid).

    `bits` / `formula` are the same holders as in {@link Method} ({@link size_projection} accepts
    both) — calibration **does not change the grid by a single byte** (it only re-chooses rounding
    targets within the same grid), so the storage formula can be shared with the method side.
    """

    name: str
    method: CalibMethod
    grid: GridSpec
    bits: Callable[[TargetStats], float]
    formula: str

    @property
    def label(self) -> str:
        """The method name shown in the table and the JSON (`<method>/<storage grid>`)."""
        return f"{self.method}/{self.grid.kind}"


#: The note attached to the AWQ formulas — the equivalent scale factor `s` can only be held folded
#: into `W_eff`, and storing it needs a fold or a companion (the module docstring of
#: `karume.quant_calib`).
AWQ_FORMULA_NOTE = " (the fold / companion of the equivalent scale s is not counted)"

#: The calibration grid = 5 configs (run in this order). The rounding is shared through core's
#: `karume.quant_calib`; this file holds only the dispatch and the projection formulas. **The
#: target is only 1 form, linear only** — no combination including `embed_tokens` (calibration is
#: limited to `nn.Linear`, and the vocabulary table is outside the stages).
CALIB_CONFIGS: tuple[CalibConfig, ...] = (
    CalibConfig(
        "gptq-rtn",
        "gptq",
        GridSpec(kind="rtn", group_size=GROUP_SIZE),
        _group_scaled_bits(F32_SCALE_BITS),
        "4·N + 32·G (the storage is RTN i4 itself — calibration changes only the rounding targets)",
    ),
    CalibConfig(
        "gptq-nf4",
        "gptq",
        GridSpec(kind="nf4", group_size=GROUP_SIZE),
        _group_scaled_bits(F32_SCALE_BITS),
        "4·N + 32·G (the grid is the fixed NF4 table)",
    ),
    CalibConfig(
        "gptq-kmeans",
        "gptq",
        GridSpec(kind="kmeans_shared", group_size=GROUP_SIZE),
        _codebook_bits(lambda stats: stats.modules, F32_SCALE_BITS),
        "4·N + 32·G + 32·16·(number of layers) (a table's scope is **within a layer** — a"
        " different formula from kmeans:shared with one table overall)",
    ),
    CalibConfig(
        "awq-rtn",
        "awq",
        GridSpec(kind="rtn", group_size=GROUP_SIZE),
        _group_scaled_bits(F32_SCALE_BITS),
        f"4·N + 32·G{AWQ_FORMULA_NOTE}",
    ),
    CalibConfig(
        "awq-gptq-rtn",
        "awq+gptq",
        GridSpec(kind="rtn", group_size=GROUP_SIZE),
        _group_scaled_bits(F32_SCALE_BITS),
        f"4·N + 32·G{AWQ_FORMULA_NOTE}",
    ),
)

CALIB_NAMES = tuple(config.name for config in CALIB_CONFIGS)


def calib_config_name(config: CalibConfig) -> str:
    return f"{config.name}/{CALIB_TARGET}"


# ---- Metrics ------------------------------------------------------------------------


def cosine(value: torch.Tensor, reference: torch.Tensor) -> float:
    """`⟨v,r⟩ / (‖v‖·‖r‖)` taken in f64.

    The embeddings are L2-normalised in the last stage of the graph, so this should be the dot
    product itself, but it is taken with normalisation — so that when rounding moves the norm, the
    norm breakage does not leak into the cosine as "a dot product above / below 1" (norm breakage
    is a separate quantity).
    """
    x = value.reshape(-1).to(torch.float64)
    y = reference.reshape(-1).to(torch.float64)
    denom = float(x.norm() * y.norm())
    if denom == 0.0:
        raise AssertionError(
            "the cosine denominator is 0 (the embedding is all zeros — the path is broken before"
            " any rounding)"
        )
    return float(torch.dot(x, y) / denom)


def pair_label(pair: tuple[str, str]) -> str:
    return f"{pair[0]}×{pair[1]}"


def pair_cosines(vectors: Mapping[str, torch.Tensor]) -> dict[str, float]:
    """Cosines of all pairs of the 5 cases (the 10 pairs of the upper triangle)."""
    return {
        pair_label((left, right)): cosine(vectors[left], vectors[right])
        for index, left in enumerate(CASE_NAMES)
        for right in CASE_NAMES[index + 1 :]
    }


def l2_norms(vectors: Mapping[str, torch.Tensor]) -> dict[str, float]:
    return {name: float(vector.to(torch.float64).norm()) for name, vector in vectors.items()}


def measure(
    vectors: Mapping[str, torch.Tensor], base: Mapping[str, torch.Tensor]
) -> dict[str, Any]:
    """The measurement columns of one config (compared with the baseline f32 embeddings)."""
    per_case = {name: cosine(vectors[name], base[name]) for name in CASE_NAMES}
    pairs = pair_cosines(vectors)
    base_pairs = pair_cosines(base)
    drift = {label: pairs[label] - base_pairs[label] for label in pairs}
    worst = max(drift, key=lambda label: abs(drift[label]))
    near, far = pairs[pair_label(eg.NEAR_PAIR)], pairs[pair_label(eg.FAR_PAIR)]
    base_near = base_pairs[pair_label(eg.NEAR_PAIR)]
    base_far = base_pairs[pair_label(eg.FAR_PAIR)]
    values = list(per_case.values())
    return {
        "caseCosine": per_case,
        "caseCosineMin": min(values),
        "caseCosineMean": sum(values) / len(values),
        "l2Norms": l2_norms(vectors),
        "order": {
            "near": pair_label(eg.NEAR_PAIR),
            "far": pair_label(eg.FAR_PAIR),
            "nearCosine": near,
            "farCosine": far,
            "margin": near - far,
            "baseMargin": base_near - base_far,
            "holds": near > far,
        },
        "pairs": pairs,
        "pairDrift": drift,
        "pairDriftMaxAbs": abs(drift[worst]),
        "pairDriftWorst": worst,
    }


# ---- Target sets and size projection ------------------------------------------------


def count_targets(targets: Iterable[tuple[str, torch.Tensor, int]], where: str) -> TargetStats:
    """Count the modules, elements, groups and channels of the listed targets (the input of the size
    projection).

    The group count is taken from the flattened form of {@link karume.quantize.channel_rows} —
    unless it is counted on the same axis the rounding splits groups on (each op's "in axis"), the
    projection alone would be a number for a different shape. A target whose quantization axis is
    not divisible by `GROUP_SIZE` fails loudly (no bpw for a shape that cannot be stored — ADR 0069
    decision 2).
    """
    modules = elements = groups = channels = 0
    for fqn, weight, axis in targets:
        rows = channel_rows(weight, axis)
        span = int(rows.shape[1])
        if span % GROUP_SIZE:
            raise AssertionError(
                f"'{fqn}': quantization axis {span} is not divisible by group {GROUP_SIZE}"
                " (ADR 0069 decision 2 — no projection for a shape with a partial group)"
            )
        modules += 1
        elements += int(weight.numel())
        groups += int(rows.shape[0]) * (span // GROUP_SIZE)
        channels += int(rows.shape[0])
    if not modules:
        raise AssertionError(f"target '{where}' has no quantizable weight")
    return TargetStats(modules=modules, elements=elements, groups=groups, channels=channels)


def target_stats(model: nn.Module, target: Target) -> TargetStats:
    """Take the properties of the target set from the whole model (target selection's source of
    truth is core's `iter_quant_targets`)."""
    return count_targets(iter_quant_targets(model, target.op_types), target.name)


def size_projection(method: Method | CalibConfig, stats: TargetStats) -> dict[str, Any]:
    """Effective bpw and projected MiB of method × target set (the total over the target tensors
    only — not the whole model)."""
    bits = method.bits(stats)
    baseline_bits = F32_BITS * stats.elements
    return {
        "elements": stats.elements,
        "modules": stats.modules,
        "groups": stats.groups,
        "channels": stats.channels,
        "bitsPerWeight": bits / stats.elements,
        "projectedMiB": bits / BITS_PER_BYTE / MIB,
        "f32MiB": baseline_bits / BITS_PER_BYTE / MIB,
        "ratio": bits / baseline_bits,
        "formula": method.formula,
    }


# ---- Driving calibrated rounding (Catcher + stage split) ----------------------------

#: The in-model FQN prefix of the decoder layers (`EmbeddingWrapper.model` = `Gemma3TextModel` →
#: `.layers`). {@link LayerStage} names its child after the layer index so that the local FQN
#: inside a stage is `<layer index>.self_attn.q_proj.weight`, so the prefix only needs to reach this
#: far (to align the ledger keys with the in-model FQN space of `Int4Report` — core's `StageSpec`).
DECODER_PREFIX = "model.layers"

#: The keyword name under which {@link LayerStage.forward} receives the kwargs per layer_type.
LAYER_KWARGS = "layer_kwargs"


class _FirstStagesReached(Exception):  # noqa: N818 — a signal to stop, not an error
    """Signal that the kwargs of every layer_type are in (a sentinel to cut the calibration forward
    short)."""


class LayerStage(nn.Module):
    """Stage wrapper that wraps one decoder layer into a form "that takes hidden as a positional
    argument".

    Gemma3 gives each layer **its own mask and its own RoPE table** (sliding / full of
    `layer_types` switch both `create_causal_mask` and `rotary_emb` — `Gemma3TextModel.forward` of
    transformers 5.14.1). `calibrate_stages` carries the kwargs unchanged across stages, so the
    captured **set of kwargs per layer_type** is passed as a whole and the stage itself picks one.
    `advance_kwargs` (the hook that looks them up again by layer index) is not enough — both the
    mask and the RoPE table differ per calibration text, and the layer index alone cannot identify
    "which text's mask".

    The child module is named after **the layer index** so that adding {@link DECODER_PREFIX} to the
    local FQN inside the stage (`0.self_attn.q_proj.weight`) is enough to get back the in-model FQN.
    """

    def __init__(self, index: int, layer: nn.Module, layer_type: str) -> None:
        super().__init__()
        self.child = str(index)
        self.layer_type = layer_type
        self.add_module(self.child, layer)

    def forward(
        self, hidden: torch.Tensor, layer_kwargs: Mapping[str, Mapping[str, Any]]
    ) -> torch.Tensor:
        return getattr(self, self.child)(hidden, **layer_kwargs[self.layer_type])


def decoder_stages(wrapper: nn.Module) -> tuple[StageSpec, ...]:
    """Return the decoder layers in execution order as `(in-model FQN prefix, stage)`.

    The final `norm` and the 2 Dense stages (`2_Dense` / `3_Dense`) are **not included** — the
    calibration targets are only the linear inside the decoder, and the Dense stages are outside
    the decoder (after the pooling). This is why there are 2 fewer than the `linear` target of the
    method grid ({@link CALIB_TARGET}).
    """
    types = tuple(wrapper.model.config.layer_types)
    layers = list(wrapper.model.layers)
    if len(types) != len(layers):
        raise AssertionError(
            f"{len(types)} layer_types disagree with {len(layers)} decoder layers"
            " (the model layout differs from what the script expects)"
        )
    return tuple(
        (DECODER_PREFIX, LayerStage(index, layer, types[index]))
        for index, layer in enumerate(layers)
    )


def calib_targets(stages: Sequence[StageSpec]) -> tuple[dict[str, torch.Tensor], TargetStats]:
    """Return the quantization targets inside the stages as weights keyed by fqn plus their
    properties (the calibration **scan** = the reference of the check).

    Target selection is shared through core's `iter_quant_targets` (same as
    {@link target_stats}). {@link assert_calib_covers_scan} compares the count taken here with the
    count the calibration actually rounded.
    """
    weights: dict[str, torch.Tensor] = {}
    found: list[tuple[str, torch.Tensor, int]] = []
    for prefix, stage in stages:
        for local, weight, axis in iter_quant_targets(stage, (nn.Linear,)):
            fqn = f"{prefix}.{local}"
            weights[fqn] = weight
            found.append((fqn, weight, axis))
    return weights, count_targets(found, CALIB_TARGET)


def calib_inputs(model_dir: Path, limit: int | None) -> tuple[torch.Tensor, ...]:
    """Turn the calibration corpus into `[1,T]` id sequences through **the existing tokenizer path**
    (the same lookup as `export.build_cases`).

    The prompt prefixes are verbatim from `config_sentence_transformers.json` ({@link
    embeddinggemma.export.load_prompts}) — no copy is kept, the same as on the golden side.
    """
    from tokenizers import Tokenizer

    if limit is not None and limit < 1:
        raise ValueError(f"the calibration text limit must be at least 1 (got {limit})")
    entries = CALIB_TEXTS if limit is None else CALIB_TEXTS[:limit]
    tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    prompts = eg.load_prompts(model_dir)
    inputs: list[torch.Tensor] = []
    for prompt_key, body in entries:
        text = body if prompt_key is None else prompts[prompt_key] + body
        ids = torch.tensor([tokenizer.encode(text).ids], dtype=torch.int64)
        length = int(ids.shape[1])
        if not 2 <= length <= eg.SYM_MAX:
            raise AssertionError(
                f"calibration text T={length} is outside the symbolic dimension range"
                f" [2, {eg.SYM_MAX}]"
            )
        inputs.append(ids)
    return tuple(inputs)


def capture_stage_batches(
    wrapper: nn.Module, inputs: Sequence[torch.Tensor]
) -> tuple[StageBatch, ...]:
    """Capture the hidden into the first decoder layer and **the kwargs per layer_type** (Catcher).

    The calibration input is "the hidden after the embedding (including Gemma3's normalizer) and its
    accompanying arguments"; rebuilding the mask and the RoPE table ourselves would silently split
    from the transformers side. **The call into the first layer of each layer_type** is captured,
    and once every kind is in, a sentinel exception cuts the forward short. The mask / RoPE table
    are determined only by shape (T and layer_type), not by the layer's values, so kwargs captured
    at a later layer can be used for every layer of that layer_type (`Gemma3TextModel.forward`
    builds both kinds **before** the loop).

    MUST: fail loudly if the forward runs to completion without everything captured — it signals
    that the stage spelling disagrees with the model layout, and carrying on silently would only
    produce a "zero calibration inputs" diagnostic in core.
    """
    layers = list(wrapper.model.layers)
    types = tuple(wrapper.model.config.layer_types)
    firsts: dict[str, int] = {}
    for index, kind in enumerate(types):
        firsts.setdefault(kind, index)
    last = max(firsts.values())

    hidden: list[torch.Tensor] = []
    kwargs_by_type: dict[str, Mapping[str, Any]] = {}

    def make_hook(kind: str, index: int) -> Callable[..., None]:
        def pre(_module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
            kwargs_by_type[kind] = dict(kwargs)
            if index == 0:
                hidden.append(args[0].detach())
            if index == last:
                raise _FirstStagesReached

        return pre

    handles = [
        layers[index].register_forward_pre_hook(make_hook(kind, index), with_kwargs=True)
        for kind, index in firsts.items()
    ]
    batches: list[StageBatch] = []
    try:
        for position, ids in enumerate(inputs):
            hidden.clear()
            kwargs_by_type.clear()
            try:
                with torch.no_grad():
                    wrapper(ids, torch.ones_like(ids, dtype=torch.float32))
            except _FirstStagesReached:
                pass
            if len(hidden) != 1 or set(kwargs_by_type) != set(firsts):
                raise AssertionError(
                    f"calibration input {position}: the inputs of the first stages were not all"
                    f" captured (hidden {len(hidden)} / layer_type {sorted(kwargs_by_type)})"
                    " — the stage spelling disagrees with the model layout"
                )
            batches.append(((hidden[0],), {LAYER_KWARGS: dict(kwargs_by_type)}))
    finally:
        for handle in handles:
            handle.remove()
    return tuple(batches)


@dataclass(frozen=True)
class CalibRig:
    """The scaffolding shared by the calibrated configs (the stage sequence, the scan, the inputs to
    the first stage).

    Not rebuilt per config — the input to the first stage is the hidden that has only passed
    `embed_tokens`, and calibration rounds only the linear inside the decoder, so it does not move
    across a pristine restore (taken once, **before a single rounding is applied**).
    """

    stages: tuple[StageSpec, ...]
    scan: Mapping[str, torch.Tensor]
    stats: TargetStats
    batches: tuple[StageBatch, ...]


def assert_stage_split(
    wrapper: nn.Module,
    probe: torch.Tensor,
    batch: StageBatch,
    stages: Sequence[StageSpec],
) -> None:
    """Check that the sequential forward of the stage sequence is **bit-identical** to the body's
    final hidden.

    `probe` is the id sequence used for the capture itself, and `batch` the input to the first stage
    captured with **the same input**. The stage sequence does not include the final `norm`
    ({@link decoder_stages}), so `norm` is applied to the sequential result before comparing it with
    `last_hidden_state` of `Gemma3TextModel` (= the first line of `EmbeddingWrapper.forward`). The
    stages are driven in the same form as core (`_advance_batch`) — the only positional argument is
    the hidden, and the kwargs do not change across stages.

    MUST: measure before rounding, and compare with `torch.equal` rather than an approximate
    tolerance — the same modules are called in the same order, so a disagreement means the split
    goes down a path different from the real one. Calibration chooses rounding targets from "the
    activations that actually flow into that layer", so a drifted split decides the choice of method
    on a measurement of another path (and the numbers come out ordinary, so the table does not show
    it).
    """
    args, kwargs = batch
    with torch.no_grad():
        reference = wrapper.model(input_ids=probe, use_cache=False).last_hidden_state
        hidden = args[0]
        for _prefix, stage in stages:
            hidden = first_tensor_output(stage(hidden, **kwargs))
        hidden = wrapper.model.norm(hidden)
    if not torch.equal(hidden, reference):
        raise AssertionError(
            "the final hidden of the stage split is not bit-identical to the model's"
            " last_hidden_state"
            f" (max abs diff {float((hidden - reference).abs().max()):.4e})"
            " — the LayerStage sequence deviates from the path of Gemma3TextModel.forward"
        )


def build_calib_rig(wrapper: nn.Module, model_dir: Path, limit: int | None) -> CalibRig:
    """Build the calibration scaffolding (stage split → scan → tokenize → Catcher → split agreement
    check)."""
    stages = decoder_stages(wrapper)
    scan, stats = calib_targets(stages)
    inputs = calib_inputs(model_dir, limit)
    batches = capture_stage_batches(wrapper, inputs)
    assert_stage_split(wrapper, inputs[0], batches[0], stages)
    return CalibRig(stages=stages, scan=scan, stats=stats, batches=batches)


def apply_calib(config: CalibConfig, rig: CalibRig, shared_stride: int = 0) -> CalibReport:
    """Apply one calibrated rounding to the model in place (core drives the stages sequentially).

    Only when `shared_stride > 1` does the **table fit** of `kmeans_shared` use an evenly spaced
    subsample. The fact that it was used is picked up by
    {@link karume.quant_calib.CalibReport.describe} and appears in the output.
    """
    spec = config.grid
    if spec.kind == "kmeans_shared" and shared_stride > 1:
        spec = replace(spec, fit_stride=shared_stride)
    return calibrate_stages(rig.stages, rig.batches, method=config.method, spec=spec)


def assert_calib_covers_scan(
    report: CalibReport, scan: Mapping[str, torch.Tensor], name: str
) -> None:
    """Check that the layers the calibration rounded match the stage scan **exactly, with nothing
    missing or extra**.

    MUST: fail loudly. If the stage spelling or the target type changes and part of the decoder
    drops out of calibration, the row stays in the table, and the cosine even comes out on **the
    good side** by the amount left unrounded (the pass-through cannot be read from the numbers). It
    blocks vacuity, the same standing as the existing 3 gates ({@link run_gates}).
    """
    rounded = {layer.fqn for layer in report.layers}
    missing = sorted(set(scan) - rounded)
    extra = sorted(rounded - set(scan))
    if missing or extra or report.modules != len(scan):
        raise AssertionError(
            f"[{name}] the calibration rounded {report.modules} modules, which does not match the"
            f" {len(scan)} of the scan (not rounded {missing[:3]} / not in the scan {extra[:3]})"
        )


# ---- Run ----------------------------------------------------------------------------


def embed_cases(
    wrapper: nn.Module, cases: Sequence[tuple[str, torch.Tensor, torch.Tensor]]
) -> dict[str, torch.Tensor]:
    """Embeddings of the 5 golden cases (clones flattened to `[H]`)."""
    vectors: dict[str, torch.Tensor] = {}
    for name, ids, pool_mask in cases:
        with torch.no_grad():
            vectors[name] = wrapper(ids, pool_mask).detach().reshape(-1).clone()
    return vectors


def stacked(vectors: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """Stack the 5 cases into one tensor for the bit-equality decision (the order is fixed to
    {@link CASE_NAMES})."""
    return torch.stack([vectors[name] for name in CASE_NAMES])


def restore(weights: Mapping[str, torch.Tensor], pristine: Mapping[str, torch.Tensor]) -> None:
    """Restore every target to the pristine f32 weights (the body of the MUST **do not stack
    methods**)."""
    with torch.no_grad():
        for fqn, weight in weights.items():
            weight.copy_(pristine[fqn])


def select_methods(only: Sequence[str]) -> tuple[Method, ...]:
    """Return the methods chosen with `--only` in declaration order (the baseline f32 always runs —
    the denominator of the comparison)."""
    if not only:
        return METHODS
    chosen = set(only)
    return tuple(method for method in METHODS if method.name in chosen)


def select_calib(only: Sequence[str]) -> tuple[CalibConfig, ...]:
    """Return the calibrated configs chosen with `--only` in declaration order (the same way as
    {@link select_methods})."""
    if not only:
        return CALIB_CONFIGS
    chosen = set(only)
    return tuple(config for config in CALIB_CONFIGS if config.name in chosen)


def run(
    model_dir: Path,
    methods: Sequence[Method],
    calib_configs: Sequence[CalibConfig] = (),
    calib_limit: int | None = None,
) -> dict[str, Any]:
    """Load the model only once, apply the configs in turn and measure them.

    The calibrated configs run **after** the method grid (the target sets differ, so they do not
    mix in the table's order). The scaffolding ({@link CalibRig}) is built only once — the
    calibration inputs do not move across `restore`.
    """
    cases = eg.build_cases(model_dir, eg.SYM_MAX)
    lengths = " ".join(f"{name}(T={int(ids.shape[1])})" for name, ids, _ in cases)
    print(f"[cases] {len(cases)} cases: {lengths}", file=sys.stderr, flush=True)

    wrapper = eg.load_wrapper(model_dir)
    stats = {target.name: target_stats(wrapper, target) for target in TARGETS}
    for target in TARGETS:
        entry = stats[target.name]
        print(
            f"[target] {target.name}: {entry.modules} modules / {entry.elements:,} elements"
            f" / group {entry.groups:,} / channels {entry.channels:,}",
            file=sys.stderr,
            flush=True,
        )

    weights = {fqn: weight for fqn, weight, _axis in iter_quant_targets(wrapper, WIDEST_OP_TYPES)}
    pristine = {fqn: weight.detach().clone() for fqn, weight in weights.items()}

    base_vectors = embed_cases(wrapper, cases)
    # The check of the baseline itself — unit norm and semantic order (verbatim `export._sanity`).
    # If it breaks, every number after it loses its meaning, so fail before applying any method.
    base_sanity = eg._sanity(base_vectors)
    print(f"[{BASE_CONFIG}] sanity {json.dumps(base_sanity, ensure_ascii=False)}", file=sys.stderr)

    configs: dict[str, dict[str, Any]] = {
        BASE_CONFIG: {
            "method": BASE_CONFIG,
            "target": "—",
            "quantReport": "no rounding (baseline)",
            **measure(base_vectors, base_vectors),
        }
    }
    for method in methods:
        for target in TARGETS:
            name = config_name(method, target)
            started = time.perf_counter()
            restore(weights, pristine)
            report = method.apply(wrapper, target.op_types)
            vectors = embed_cases(wrapper, cases)
            entry: dict[str, Any] = {
                "method": method.name,
                "target": target.name,
                "quantReport": report,
                "moved": not torch.equal(stacked(vectors), stacked(base_vectors)),
                "size": size_projection(method, stats[target.name]),
                **measure(vectors, base_vectors),
                "elapsed": round(time.perf_counter() - started, 1),
            }
            configs[name] = entry
            print(
                f"[{name}] {report} cos min {entry['caseCosineMin']:.6f}"
                f" order {'held' if entry['order']['holds'] else 'collapsed'}"
                f" drift {entry['pairDriftMaxAbs']:.4e}"
                f" ({entry['elapsed']:.0f}s)",
                file=sys.stderr,
                flush=True,
            )

    rig: CalibRig | None = None
    for config in calib_configs:
        name = calib_config_name(config)
        started = time.perf_counter()
        restore(weights, pristine)
        if rig is None:
            rig = build_calib_rig(wrapper, model_dir, calib_limit)
            print(
                f"[calib] {len(rig.batches)} calibration texts / {len(rig.stages)} stages /"
                f" {rig.stats.modules} target linear modules"
                " (inside the decoder only — the 2 Dense stages and embed_tokens are excluded)",
                file=sys.stderr,
                flush=True,
            )
        calib_report = apply_calib(config, rig)
        assert_calib_covers_scan(calib_report, rig.scan, config.name)
        vectors = embed_cases(wrapper, cases)
        calib_entry: dict[str, Any] = {
            "method": config.label,
            "target": CALIB_TARGET,
            "quantReport": calib_report.describe(),
            "moved": not torch.equal(stacked(vectors), stacked(base_vectors)),
            "size": size_projection(config, rig.stats),
            "calibModules": calib_report.modules,
            **measure(vectors, base_vectors),
            "elapsed": round(time.perf_counter() - started, 1),
        }
        configs[name] = calib_entry
        print(
            f"[{name}] {calib_entry['quantReport']}"
            f" cos min {calib_entry['caseCosineMin']:.6f}"
            f" order {'held' if calib_entry['order']['holds'] else 'collapsed'}"
            f" drift {calib_entry['pairDriftMaxAbs']:.4e}"
            f" ({calib_entry['elapsed']:.0f}s)",
            file=sys.stderr,
            flush=True,
        )
    restore(weights, pristine)
    return {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "script": "tools/export-recipes/embeddinggemma/measure_quant.py",
        "torch": torch.__version__,
        "modelDir": str(model_dir),
        "groupSize": GROUP_SIZE,
        "codebookLevels": DEFAULT_CODEBOOK_LEVELS,
        "cases": {name: int(ids.shape[1]) for name, ids, _ in cases},
        "baseSanity": base_sanity,
        "calibTexts": len(rig.batches) if rig else None,
        "targets": {
            target.name: {
                "opTypes": [cls.__name__ for cls in target.op_types],
                **vars(stats[target.name]),
            }
            for target in TARGETS
        },
        "configs": configs,
        "gates": run_gates(configs, methods),
    }


def run_gates(
    configs: Mapping[str, Mapping[str, Any]], methods: Sequence[Method]
) -> dict[str, Any]:
    """Block vacuity (rounding passing through, and whether `op_types` takes effect).

    ① Each config's embeddings are **not bit-identical** to the baseline — they agree only when not
       a single rounding was applied, and since quality then comes out on the "perfect" side,
       staying silent invites nothing but misreading.
    ② `linear+embedding` rounds **exactly the embedding more** than `linear` (a difference of 1
       module). If `op_types` has no effect this is 0, and a table that reads as "the vocabulary
       table was measured too" comes out with the vocabulary table passed through.
    ③ The embeddings of the 2 forms of ② **differ from each other** — if only the count grows and
       the values stay the same, the rounding of the embedding does not reach the output (catches
       "the targets grew but the path is different").

    For the calibrated configs, "rounded count = scan count" fails loudly **on the spot** per config
    in {@link assert_calib_covers_scan} (instead of letting a 60-minute run finish before failing).
    This only records the result for the reader and does not hold the decision twice.
    """
    failures: list[str] = []
    for name, entry in configs.items():
        if name != BASE_CONFIG and not entry["moved"]:
            failures.append(
                f"{name}: the embeddings are bit-identical to the baseline (the rounding passed"
                " through)"
            )
    embedding_targets: list[str] = []
    for method in methods:
        linear = configs[config_name(method, TARGET_LINEAR)]
        widened = configs[config_name(method, TARGET_WITH_EMBEDDING)]
        delta = widened["size"]["modules"] - linear["size"]["modules"]
        embedding_targets.append(f"{method.name}: +{delta} modules")
        if delta != 1:
            failures.append(
                f"{method.name}: linear+embedding rounds {delta} more modules than linear"
                " (it should be 1, for embed_tokens — the op_types switch has no effect)"
            )
        if widened["caseCosine"] == linear["caseCosine"]:
            failures.append(
                f"{method.name}: linear+embedding and linear give identical embeddings"
                " (the rounding of the vocabulary table does not reach the output)"
            )
    return {
        "moved": {name: entry["moved"] for name, entry in configs.items() if name != BASE_CONFIG},
        "embeddingWidening": embedding_targets,
        "calibRounded": {
            name: f"{entry['calibModules']}/{entry['size']['modules']} modules"
            for name, entry in configs.items()
            if "calibModules" in entry
        },
        "failures": failures,
    }


# ---- Tables -------------------------------------------------------------------------


def _markdown(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = [f"| {' | '.join(header)} |", f"| {' | '.join('---' for _ in header)} |"]
    lines += [f"| {' | '.join(row)} |" for row in rows]
    return "\n".join(lines)


def quality_table(report: Mapping[str, Any]) -> str:
    """Method × target × measurement columns (the main result of this script)."""
    header = [
        "method",
        "target",
        "modules",
        "cos min",
        "cos mean",
        "near",
        "far",
        "semantic order",
        "max pair drift",
    ]
    rows: list[list[str]] = []
    for name, entry in report["configs"].items():
        order = entry["order"]
        modules = entry["size"]["modules"] if "size" in entry else "—"
        rows.append(
            [
                name if name == BASE_CONFIG else entry["method"],
                entry["target"],
                str(modules),
                f"{entry['caseCosineMin']:.6f}",
                f"{entry['caseCosineMean']:.6f}",
                f"{order['nearCosine']:.4f}",
                f"{order['farCosine']:.4f}",
                "held" if order["holds"] else "**collapsed**",
                f"{entry['pairDriftMaxAbs']:.4e}",
            ]
        )
    return _markdown(header, rows)


def size_table(report: Mapping[str, Any]) -> str:
    """Size projection of method × target (the total over the target tensor set only)."""
    header = ["method", "target", "elements", "bpw", "projected MiB", "f32 MiB", "ratio", "formula"]
    rows = [
        [
            entry["method"],
            entry["target"],
            f"{entry['size']['elements']:,}",
            f"{entry['size']['bitsPerWeight']:.3f}",
            f"{entry['size']['projectedMiB']:.1f}",
            f"{entry['size']['f32MiB']:.1f}",
            f"{entry['size']['ratio'] * 100:.1f}%",
            entry["size"]["formula"],
        ]
        for entry in report["configs"].values()
        if "size" in entry
    ]
    return _markdown(header, rows)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model-dir", type=Path, default=eg.DEFAULT_MODEL_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        choices=(*METHOD_NAMES, *CALIB_NAMES),
        help="run only this method / calibrated config (repeatable, for partial reruns). The"
        f" baseline f32 always runs. {REPORT_FILE} lists only the configs that ran.",
    )
    parser.add_argument(
        "--calib-limit",
        type=int,
        default=None,
        help="use only the first N texts of the calibration corpus"
        f" (default: all {len(CALIB_TEXTS)})."
        " A knob for a reduced smoke run; it affects only the calibrated configs.",
    )
    args = parser.parse_args(argv)
    report = run(
        args.model_dir,
        select_methods(args.only),
        select_calib(args.only),
        args.calib_limit,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / REPORT_FILE).write_text(
        json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print("Quality (compared with the baseline f32 embeddings):")
    print(quality_table(report))
    print()
    print(f"Size projection (target tensor set only; N = elements, G = N/{GROUP_SIZE}):")
    print(size_table(report))
    print()
    print(f"report → {args.out / REPORT_FILE}")
    failures = report["gates"]["failures"]
    if failures:
        raise AssertionError("verification gates are red: " + " / ".join(failures))
    print("gates: all green")


if __name__ == "__main__":
    main()
