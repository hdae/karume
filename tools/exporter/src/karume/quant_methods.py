"""**Measurement-only** weight rounding methods (FP4 / NF4 / MXFP4 / k-means codebook) — they have
no storage path.

Every form in `quantize.py` (f16 / i8 / i4) is a "**writable storage form**" with an emit and a
runtime implementation; the methods here cannot be written, and their aim is to answer "how far
does quality recover with this method" first, with zero lines of GPU code (the same idea as the
Phase 0 sweep of ADR 0069 decision 6 — in the division of ADR 0058 decision 5, the numbers side
of "quality = advisory numbers"). They live in a separate module so as not to break the reading
of `quantize.py` that "what is in it = the storage forms that can ship".

MUST: no hook into emit is built. The return value is counts only, with no scale / codebook
ledger (returning one would open a route for "measurement rounding" into the storage path —
eligibility is decided by the consuming op, so the emit side cannot tell right from wrong at the
point it receives it). The rounded weights themselves are rewritten in place, so degradation is
measured from them.

MUST: **same input → bit-identical output**. Even the k-means initialisation is deterministic
(quantiles + a fixed iteration count) because with a random seed "the difference between methods"
and "the difference between seeds" could not be separated.

MUST: rounding happens **before the reference and the goldens are captured** (same text as the
`quantize` module docstring).

Target selection is **shared** with `quantize.iter_quant_targets`; the default is linear only, as
for {@link quantize.fake_quant_int4}, and `op_types` can widen it up to every type of i8
(`QUANT_MODULE_TYPES`).
It is not a copied separate implementation, to keep the measured targets and the shipped targets
from silently diverging.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

from .quantize import (
    DEFAULT_GROUP_SIZE,
    QuantizeError,
    channel_rows,
    group_size_of,
    grouped_view,
    iter_quant_targets,
    restore_channel_rows,
)

#: The FP4 (e2m1) value set — 1 sign / 2 exponent / 1 mantissa bit (subnormals included, from
#: steps of 0.5 up to 6.0). Of the 16 codes ±0 are equal, so there are **15 distinct values** (as
#: i4's ±7 has 15 levels).
#: MUST: ascending (rounding is a binary search over the midpoints of adjacent levels —
#: {@link round_to_levels}).
FP4_E2M1_LEVELS: tuple[float, ...] = (
    -6.0,
    -4.0,
    -3.0,
    -2.0,
    -1.5,
    -1.0,
    -0.5,
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
)

#: The 16 levels of NF4 (4-bit NormalFloat). Source = NF4 in §3 of Dettmers et al., "QLoRA:
#: Efficient Finetuning of Quantized LLMs" (arXiv:2305.14314), and the table produced by
#: `create_normal_map()` of its implementation, bitsandbytes. An asymmetric grid of standard normal
#: quantiles normalised to ±1, which **contains 0.0 and ±1.0 exactly** (without 0 in the table
#: sparse weights cannot be restored, and without ±1 the group's amax element is not restored).
#: Unlike FP4, the steps differ between the positive and negative sides.
NF4_LEVELS: tuple[float, ...] = (
    -1.0,
    -0.6961928009986877,
    -0.5250730514526367,
    -0.39491748809814453,
    -0.28444138169288635,
    -0.18477343022823334,
    -0.09105003625154495,
    0.0,
    0.07958029955625534,
    0.16093020141124725,
    0.24611230194568634,
    0.33791524171829224,
    0.44070982933044434,
    0.5626170039176941,
    0.7229568362236023,
    1.0,
)

#: Maximum exponent of the MXFP4 element format (= FP4 e2m1). The conversion procedure of OCP
#: Microscaling Formats (MX) v1.0 takes the block's shared scale as the power of two
#: `2^(floor(log2(amax)) − emax_elem)`. The largest e2m1 value is 6.0 = 1.5×2², so
#: `emax_elem = 2`.
MXFP4_ELEMENT_EMAX = 2

#: Number of k-means codebook levels (4 bit = 16 centroids).
DEFAULT_CODEBOOK_LEVELS = 16

#: Fixed number of Lloyd iterations. **No convergence test stops it** — its threshold would be an
#: input-dependent arbitrary choice, and "not enough iterations" would mix with "the method is
#: weak". For 1-D with 16 centroids this count settles well enough.
DEFAULT_CODEBOOK_ITERATIONS = 16

#: Granularity of the k-means table. `shared` fits **one table across layers** in the value space
#: normalised by the group (g32) absmax (the table cost does not grow with the layer count, at the
#: price of needing a scale).
KMeansGranularity = Literal["per_tensor", "per_channel", "shared"]

KMEANS_GRANULARITIES: tuple[KMeansGranularity, ...] = ("per_tensor", "per_channel", "shared")


@dataclass(frozen=True)
class MethodReport:
    """The rounding method and the number of tensors and elements (a count so that "a method was
    requested but 0 tensors" is never silent).

    It holds no scale / codebook ledger because of the module docstring's MUST (no hook into
    emit). The table a layer received can be recovered from `torch.unique` of the rounded weights.
    """

    method: str
    modules: int
    elements: int

    def describe(self) -> str:
        return f"{self.method}: modules {self.modules} / {self.elements:,} elements"


def levels_tensor(levels: Sequence[float]) -> torch.Tensor:
    """Make a level table into an F32 tensor (with a check that it is ascending).

    If the table is not ascending, the binary search of {@link round_to_levels} silently returns
    another level (no exception; only the values stop being the nearest), so it is checked here,
    once, on the side that builds the table.
    """
    table = torch.tensor(levels, dtype=torch.float32)
    if table.dim() != 1 or table.numel() < 2:
        raise QuantizeError(
            f"the level table must be 1-D with 2 or more entries (actual {list(table.shape)})"
        )
    if not bool(torch.all(table[1:] > table[:-1])):
        raise QuantizeError(
            "the level table is not ascending (nearest rounding is a binary search, so ascending"
            " is a MUST)"
        )
    return table


def round_to_levels(values: torch.Tensor, levels: torch.Tensor) -> torch.Tensor:
    """Round each element of `values` to the **nearest** entry of the ascending level table
    `levels` (the shape is kept).

    An exact midpoint between adjacent levels falls to the **lower** side (toward −∞) (the default
    `right=False` of `torch.bucketize` = `boundaries[i-1] < x ≤ boundaries[i]`). It is neither
    round-half-to-even nor round-half-away-from-zero, but values exactly on a boundary are close to
    measure zero in real weights and do not matter for comparing methods — the direction is fixed
    here for determinism (observed: `amax/scale = 5.0` takes this branch in MXFP4).
    """
    midpoints = (levels[1:] + levels[:-1]) / 2
    return levels[torch.bucketize(values, midpoints)]


def round_rows_to_codebook(values: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
    """The version of {@link round_to_levels} with a separate table per row (`values[R,N]` ×
    `codebook[R,L]`).

    Applying per-channel tables in a Python loop would take thousands of round trips per layer, so
    the batched form of `searchsorted` (with per-row boundaries too) does it in one call.
    """
    midpoints = (codebook[:, 1:] + codebook[:, :-1]) / 2
    index = torch.searchsorted(midpoints.contiguous(), values.contiguous())
    return torch.gather(codebook, 1, index)


def group_absmax_scale(
    rows: torch.Tensor, group_size: int, max_level: float, where: str = "weight"
) -> torch.Tensor:
    """Return `scale = clamp(amax_group / max_level, f32 tiny)` in group shape
    `[channel, group count]`.

    The level-table version of `quantize.group_scale` (i4's `amax/7`); `max_level` is the table's
    largest magnitude (FP4 = 6.0 / NF4 = 1.0). The group's amax element lands on an end of the
    table, and a reachable scale `s = fl(amax/max_level)` is a fixed point of
    `fl(fl(max_level·s)/max_level)`, so rounding is idempotent for the same reason as on the
    `quantize` side (the `INT8_MAX` argument) — exact restoration of the amax element is not relied
    on. The lower clamp avoids division by zero for all-zero groups (`amax == 0`).
    """
    grouped = grouped_view(rows, group_size, where)
    amax = grouped.abs().amax(dim=-1)
    return torch.clamp(amax / max_level, min=torch.finfo(torch.float32).tiny)


def group_power_of_two_scale(
    rows: torch.Tensor, group_size: int, where: str = "weight"
) -> torch.Tensor:
    """OCP MX-style **power-of-two** group scale = `2^(floor(log2(amax)) − MXFP4_ELEMENT_EMAX)`.

    The exponent is taken from `frexp` (`x = m·2^e` with `m ∈ [0.5,1)`, so
    `floor(log2 x) = e − 1`) — `log2` can come out 1 ulp low at an exact power of two, but `frexp`
    returns the exponent exactly.

    Rounding down to a power of two puts `amax/scale` in `[4, 8)`, and {@link round_to_levels}
    drops values outside the table (6 < |x| < 8) to the end value 6.0 (= MX saturation). Unlike the
    absmax version the amax element itself is not restored, but in exchange the scale's
    multiplication and division are **exact** (rounding is truly idempotent). For an all-zero group
    `frexp` returns `e = 0`, so the scale falls to `2^-3` and the values stay 0.
    """
    grouped = grouped_view(rows, group_size, where)
    amax = grouped.abs().amax(dim=-1)
    _, exponent = torch.frexp(amax)
    shared = exponent - 1 - MXFP4_ELEMENT_EMAX
    return torch.ldexp(torch.ones_like(amax), shared)


def round_groups_to_levels(
    rows: torch.Tensor, levels: torch.Tensor, scale: torch.Tensor, where: str = "weight"
) -> torch.Tensor:
    """Normalise each group by `scale`, round to the nearest entry of `levels`, and multiply
    `scale` back.

    MUST: the group size comes **only from the shape of the scale passed in**
    (`quantize.group_size_of` — taking it as a separate argument would let "the groups used for
    rounding" and "the groups the scale was taken over" move independently; the same hole as ADR
    0069 decision 3).
    """
    grouped = grouped_view(rows, group_size_of(rows, scale), where)
    rounded = round_to_levels(grouped / scale.unsqueeze(-1), levels)
    return (rounded * scale.unsqueeze(-1)).reshape(rows.shape)


def _round_model_in_place(
    model: nn.Module,
    method: str,
    round_rows: Callable[[torch.Tensor, str], torch.Tensor],
    op_types: tuple[type[nn.Module], ...],
    include: Callable[[str], bool] | None,
) -> MethodReport:
    """Round every target weight with `round_rows(rows, where) -> rows` and write it back in place
    (the shared body).

    Writing back goes through {@link quantize.restore_channel_rows} because of `ConvTranspose1d` —
    its flat form is not a view of the weight (non-contiguous through `movedim` → `reshape`
    copies), so writing into the flat form would not change a single element of the original
    weight.

    "0 targets" fails loudly (the same permanent diagnostic as on the `quantize` side — so that "a
    method was requested but the weights effectively stayed f32" is never silent).
    """
    modules = 0
    elements = 0
    with torch.no_grad():
        for fqn, weight, axis in iter_quant_targets(model, op_types, include):
            rows = channel_rows(weight, axis)
            weight.copy_(restore_channel_rows(round_rows(rows, f"'{fqn}'"), weight, axis))
            modules += 1
            elements += weight.numel()
    if not modules:
        raise QuantizeError(
            f"{method}: there is no weight that can be rounded"
            f" (target types: {', '.join(cls.__name__ for cls in op_types)})"
        )
    return MethodReport(method=method, modules=modules, elements=elements)


def fake_quant_fp4(
    model: nn.Module,
    group_size: int = DEFAULT_GROUP_SIZE,
    include: Callable[[str], bool] | None = None,
    op_types: tuple[type[nn.Module], ...] = (nn.Linear,),
) -> MethodReport:
    """Round with the fixed FP4 (e2m1) table × a group (default g32) absmax scale.

    The only difference from i4 is **the density of the grid** (the same 15 levels and the same
    group scale; i4 is evenly spaced, FP4 is exponentially sparse). It is dense near 0, so this
    column measures the hypothesis that FP4 may gain the closer the weight distribution is to
    normal.
    """
    table = levels_tensor(FP4_E2M1_LEVELS)
    top = float(table.abs().max())

    def rounder(rows: torch.Tensor, where: str) -> torch.Tensor:
        return round_groups_to_levels(
            rows, table, group_absmax_scale(rows, group_size, top, where), where
        )

    return _round_model_in_place(model, "fp4", rounder, op_types, include)


def fake_quant_nf4(
    model: nn.Module,
    group_size: int = DEFAULT_GROUP_SIZE,
    include: Callable[[str], bool] | None = None,
    op_types: tuple[type[nn.Module], ...] = (nn.Linear,),
) -> MethodReport:
    """Round with the fixed NF4 table ({@link NF4_LEVELS}) × a group (default g32) absmax scale.

    The table is "quantiles of the normal distribution", so this column measures the claim that,
    when the weights within a group are close to normal, the levels are used evenly (= close to
    information-theoretically optimal). Only **the inner levels** are asymmetric; both ends contain
    ±1.0 exactly ({@link NF4_LEVELS}), so the group's amax element is restored on both signs.
    """
    table = levels_tensor(NF4_LEVELS)
    top = float(table.abs().max())

    def rounder(rows: torch.Tensor, where: str) -> torch.Tensor:
        return round_groups_to_levels(
            rows, table, group_absmax_scale(rows, group_size, top, where), where
        )

    return _round_model_in_place(model, "nf4", rounder, op_types, include)


def fake_quant_mxfp4(
    model: nn.Module,
    group_size: int = DEFAULT_GROUP_SIZE,
    include: Callable[[str], bool] | None = None,
    op_types: tuple[type[nn.Module], ...] = (nn.Linear,),
) -> MethodReport:
    """Round with MXFP4 (the FP4 table × a **power-of-two** scale per block) — the OCP Microscaling
    way.

    The only difference from the absmax version ({@link fake_quant_fp4}) is how the scale is
    represented: the scale fits in E8M0 (a 1-byte exponent), so storage is lighter, but in exchange
    the group's amax is rounded down to a power of two and up to one step's worth of headroom is
    thrown away. The default block length is this repository's default group (32), which matches
    the MX specification's block = 32.
    """
    table = levels_tensor(FP4_E2M1_LEVELS)

    def rounder(rows: torch.Tensor, where: str) -> torch.Tensor:
        return round_groups_to_levels(
            rows, table, group_power_of_two_scale(rows, group_size, where), where
        )

    return _round_model_in_place(model, "mxfp4", rounder, op_types, include)


def fit_codebook(values: torch.Tensor, levels: int, iterations: int) -> torch.Tensor:
    """Apply 1-D k-means (Lloyd's method) per row and return the ascending centroid table
    `[R, levels]`.

    MUST: **deterministic** (same input → bit-identical table). Initialisation is not random
    seeding but **quantiles** (the `(i+0.5)/levels` positions of the sorted values), and an empty
    cluster keeps its centroid (reseeding would make "where to seed" a new arbitrary choice). The
    iteration count is fixed ({@link DEFAULT_CODEBOOK_ITERATIONS}).

    In 1-D the nearest centroid is found exactly by "binary search over the midpoints" (no
    brute-force distance computation over centroids). The centroid update sums are accumulated in
    **f64** — sequential f32 addition loses precision in layers where one cluster holds millions of
    elements, and the mean starts to depend on the input order.
    """
    if values.dim() != 2:
        raise QuantizeError(f"k-means input must be 2-D `[R, N]` (actual {list(values.shape)})")
    if values.shape[1] < levels:
        raise QuantizeError(
            f"k-means has {levels} levels but there are only {values.shape[1]} elements"
            " (quantile initialisation would line up centroids with the same value)"
        )
    sorted_values, _ = torch.sort(values, dim=-1)
    count = sorted_values.shape[1]
    position = ((torch.arange(levels, dtype=torch.float64) + 0.5) / levels * count).long()
    centroids = sorted_values.index_select(1, position.clamp(max=count - 1))
    wide = values.to(torch.float64)
    ones = torch.ones_like(wide)
    for _ in range(iterations):
        index = torch.searchsorted(
            ((centroids[:, 1:] + centroids[:, :-1]) / 2).contiguous(), values.contiguous()
        )
        totals = torch.zeros(centroids.shape, dtype=torch.float64).scatter_add_(1, index, wide)
        counts = torch.zeros(centroids.shape, dtype=torch.float64).scatter_add_(1, index, ones)
        means = (totals / counts.clamp(min=1.0)).to(torch.float32)
        centroids = torch.where(counts.to(torch.float32) > 0, means, centroids)
        # Keeping an empty cluster in place can break the order (the kept value overtakes a
        # neighbour's mean). A broken table breaks the binary search's premise, so it is sorted
        # again on every iteration (cheap with 16 elements).
        centroids, _ = torch.sort(centroids, dim=-1)
    return centroids


def fake_quant_kmeans(
    model: nn.Module,
    granularity: KMeansGranularity,
    group_size: int = DEFAULT_GROUP_SIZE,
    include: Callable[[str], bool] | None = None,
    op_types: tuple[type[nn.Module], ...] = (nn.Linear,),
    levels: int = DEFAULT_CODEBOOK_LEVELS,
    iterations: int = DEFAULT_CODEBOOK_ITERATIONS,
    fit_stride: int = 1,
) -> MethodReport:
    """Round with a 16-centroid k-means codebook (3 table granularities).

    - `per_tensor` … one table per layer. The table costs `levels` entries per layer, and no
      scale is needed
    - `per_channel` … one table per output channel (a row of {@link quantize.channel_rows}). The
      table cost grows with the channel count, but the table absorbs the scale differences
      between channels
    - `shared` … **one table across layers**. It is fitted in the value space normalised by the
      group (default g32) absmax, so the scale carries the scale differences and the shared table
      carries the shape (= the table costs `levels` entries for the whole model)

    `group_size` matters only for `shared` (the other two have no scale).

    MUST: same input → bit-identical output ({@link fit_codebook} is deterministic and does not
    depend on a seed).

    NOTE: `shared` **concatenates the normalised values of every target into one tensor** before
    fitting the table, so it needs an f32 temporary of the same size as the target weights (+ the
    f64 working area of {@link fit_codebook}), which exceeds physical memory at the 1B class.
    `fit_stride` is the way out — **only the fit of the `shared` table** uses an evenly spaced
    subsample `flat[::fit_stride]` (per module) (**the table is always applied to everything**).
    No randomness is used, so the determinism MUST still holds. MUST: when it is used, the
    measuring side's output states it — a subsampled table and a full table can differ, so whoever
    reads the numbers must be able to tell them apart.
    """
    if granularity not in KMEANS_GRANULARITIES:
        raise QuantizeError(
            f"k-means granularity '{granularity}' is not supported"
            f" ({', '.join(KMEANS_GRANULARITIES)})"
        )
    if fit_stride < 1:
        raise QuantizeError(f"fit_stride must be 1 or more (actual {fit_stride})")
    if fit_stride != 1 and granularity != "shared":
        raise QuantizeError(
            f"fit_stride is for shared only (the fit of '{granularity}' does not build a"
            " concatenation the size of the targets, so there is no reason to subsample — and it"
            " is not silently ignored either)"
        )
    method = f"kmeans:{granularity}"
    if granularity != "shared":
        per_channel = granularity == "per_channel"

        def rounder(rows: torch.Tensor, _where: str) -> torch.Tensor:
            flat = rows if per_channel else rows.reshape(1, -1)
            codebook = fit_codebook(flat, levels, iterations)
            return round_rows_to_codebook(flat, codebook).reshape(rows.shape)

        return _round_model_in_place(model, method, rounder, op_types, include)

    # `shared` takes 2 passes — the first collects the normalised values of every target and fits
    # one table, the second applies it. Both passes select targets with the same
    # `iter_quant_targets` (so the first pass cannot end up wider).
    normalized: list[torch.Tensor] = []
    with torch.no_grad():
        for fqn, weight, axis in iter_quant_targets(model, op_types, include):
            rows = channel_rows(weight, axis)
            scale = group_absmax_scale(rows, group_size, 1.0, f"'{fqn}'")
            grouped = grouped_view(rows, group_size_of(rows, scale), f"'{fqn}'")
            flat = (grouped / scale.unsqueeze(-1)).reshape(-1)
            # When only the sample is kept, clone detaches it from its base (as a view the whole
            # tensor stays alive, and the reduction of temporary memory that fit_stride exists for
            # is silently lost).
            normalized.append(flat[::fit_stride].clone() if fit_stride > 1 else flat)
    if not normalized:
        raise QuantizeError(
            f"{method}: there is no weight that can be rounded"
            f" (target types: {', '.join(cls.__name__ for cls in op_types)})"
        )
    codebook = fit_codebook(torch.cat(normalized).reshape(1, -1), levels, iterations)
    table = codebook.reshape(-1)

    def apply_shared(rows: torch.Tensor, where: str) -> torch.Tensor:
        return round_groups_to_levels(
            rows, table, group_absmax_scale(rows, group_size, 1.0, where), where
        )

    return _round_model_in_place(model, method, apply_shared, op_types, include)
