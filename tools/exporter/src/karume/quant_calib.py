"""**Rounding that uses calibration data** (GPTQ / AWQ) — the second step on top of the storage
grids of `quantize`.

`quantize.fake_quant_int4` rounds each weight to the nearest value on its own (= without looking at
the input distribution). The methods here look at "the activations that actually flow into the
layer" through calibration data and re-choose the rounding target **within the same storage
grid**; the grid itself does not change by a single byte from the first step. The aim is "how far
quality can be recovered without changing the storage form".

- **GPTQ** (`method="gptq"`) … rounds the columns one at a time and **redistributes the rounding
  error onto the unprocessed columns** through the inverse of the Hessian `H = Σ XᵀX`. With the
  storage grid `rtn` it is **shippable** — the grid is exactly that of
  {@link quantize.quantize_to_int4} (`scale = group amax / 7`, `q ∈ [−7,+7]`, −8 unused), and the
  scale ledger can be passed to emit as is in the {@link quantize.Int4Report} container.
- **AWQ** (`method="awq"`) … a **fake-quant** that picks a per-in-channel equivalent factor `s`
  from the activation magnitudes, rounds `W' = W·diag(s)` and writes back `W_eff = Q(W')/s`. A
  column for measuring the upper bound of the ideal fold, and **it cannot be stored as is** (the
  MUST below).
- **Combined** (`method="awq+gptq"`) … picks `s`, then applies GPTQ to `W'`. The effective input
  becomes `x/s`, so the Hessian is transformed to `H' = diag(1/s)·H·diag(1/s)` too.

MUST: **only `method="gptq"` × `grid="rtn"` may flow into emit**. This is a deliberate design; the
other combinations carry no scale ledger in their report:

- The `nf4` / `kmeans_shared` grids return no ledger for the same reason as `quant_methods` (no
  "hook into emit" for methods without a storage path and a runtime implementation — the
  `quant_methods` module docstring).
- AWQ writes back `W_eff = Q(W')/s`, but what would need storing is the pair `Q(W')` and `s`;
  requantizing `W_eff` as is with group absmax **cancels `s` against the absmax scale as a
  constant within the group, and it returns to the equivalent of RTN** (the information `s`
  carried is lost). Shipping it needs an implementation that **folds** `s` into an adjacent
  operation or stores `s` as a **companion tensor**.

MUST: **same input → bit-identical output**. Not a single random number is used. Hessian
accumulation and the Cholesky solver are **f64** (the same reason as the f64 sums of
`quant_methods.fit_codebook` — a sequential sum of millions of elements starts to depend on the
order in f32). On the other hand, **choosing the grid scale and the rounding itself are done in
f32** — dividing in f64 and dropping to f32 can move `scale` by 1 ulp through double rounding, and
**bit identity** with {@link quantize.fake_quant_int4} (the oracle below) would no longer hold.

MUST: rounding happens **before the reference and the goldens are captured** (same text as the
`quantize` module docstring).

NOTE (oracle): when `H` is diagonal (e.g. `λI`) the compensation term `U[j, j+1:]` is exactly 0,
so `gptq` × `rtn` is **bit-identical** to {@link quantize.fake_quant_int4}. This is a mathematical
consequence, not a property of the implementation, and `tests/test_quant_calib.py` pins it as a
check.

NOTE (act-order / static-groups are **opt-in experimental axes**): the reference implementation of
GPTQ gains quality when it reorders the in axis by descending `diag(H)` before rounding (act-order
/ desc_act). Reordering breaks the storage's group alignment **only under dynamic groups (scales
taken from the compensated current values)**; combined with `static_groups=True` (all scales are
settled up front from the **original contiguous column blocks** before rounding), the reordering
affects only "the processing order of the compensation loop" — storage stays in the original
in-axis order with contiguous `group_size` columns = 1 group, so no companion permutation tensor
is needed and **the runtime changes not at all** (the same condition as `--static-groups` ×
`--act-order` of upstream `IST-DASLab/gptq`).

MUST: both **default to `False`**. The default path is **bit-identical** to the current dynamic
groups, and adding these flags does not move the reproducibility of the existing i4 distribution
assets and goldens by a single bit (the A/B cross-check in `tests/test_quant_calib.py` is the
check). `act_order=True` × `static_groups=False` fails loudly (scales taken from the current values
in the reordered processing order cannot be mapped back to the original contiguous groups).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

import torch
from torch import nn

from .quant_methods import (
    DEFAULT_CODEBOOK_ITERATIONS,
    DEFAULT_CODEBOOK_LEVELS,
    NF4_LEVELS,
    fit_codebook,
    group_absmax_scale,
    levels_tensor,
    round_groups_to_levels,
    round_to_levels,
)
from .quantize import (
    DEFAULT_GROUP_SIZE,
    INT4_MAX,
    Int4Report,
    QuantizeError,
    channel_rows,
    dequantize_int4,
    group_scale,
    grouped_view,
    iter_quant_targets,
    quantize_to_int4,
    restore_channel_rows,
)

#: GPTQ column block length (default). A block is only "the delay width over which compensation is
#: applied in bulk"; the mathematical result does not depend on the block length (only the order of
#: floating-point additions changes).
DEFAULT_GPTQ_BLOCK = 128

#: The **default** damping factor — `λ = damping × mean(diag(H))` is added to the diagonal.
#: Equivalent to upstream `fasterquant(percdamp=.01)`. Callers may vary it with `damping=` **only
#: for sensitivity sweeps**; as long as the default is not moved the existing path is
#: bit-identical (a varied value MUST be stated in the measuring side's output — numbers that mix
#: "the difference between methods" and "the difference in damping" cannot be read).
GPTQ_DAMPING = 0.01

#: Number of divisions of the AWQ α grid (17 points, `α ∈ {0, 1/16, …, 1}`).
AWQ_ALPHA_STEPS = 16

#: Lower bound of AWQ's `act_amax`. Leaving a zero channel (an input that never moved on the
#: calibration data) gives `s_j = 0` and breaks the normalisation to geometric mean 1 (`log 0`).
AWQ_ACT_EPS = 1e-5

#: Maximum number of input sample rows used by the AWQ objective (a deterministic thinning that
#: takes rows **from the start**).
DEFAULT_AWQ_SAMPLES = 1024

#: Kinds of storage grid. Only `rtn` has a shipping path (the MUST of the module docstring).
GridKind = Literal["rtn", "nf4", "kmeans_shared"]

GRID_KINDS: tuple[GridKind, ...] = ("rtn", "nf4", "kmeans_shared")

#: Methods of calibrated rounding.
CalibMethod = Literal["gptq", "awq", "awq+gptq"]

CALIB_METHODS: tuple[CalibMethod, ...] = ("gptq", "awq", "awq+gptq")

#: One batch passed to the first stage (a pair of positional and keyword arguments).
StageBatch = tuple[tuple[object, ...], Mapping[str, object]]

#: One stage in execution order (a pair of **the in-model FQN prefix** and the module). The prefix
#: is taken so that the scale ledger keys live in the same **in-model FQN** space as
#: {@link quantize.Int4Report} (walking `named_modules` on a stage alone yields only local names
#: such as `0.weight`).
StageSpec = tuple[str, nn.Module]


@dataclass(frozen=True)
class GridSpec:
    """The storage grid to round onto (a spec that points directly at the grids of `quantize` /
    `quant_methods`).

    - `rtn` … the grid of {@link quantize.quantize_to_int4} (group amax / 7, `q ∈ [−7,+7]`).
      **The only shippable grid**; it returns a scale ledger
    - `nf4` … the fixed table of {@link quant_methods.NF4_LEVELS} × a group absmax scale
    - `kmeans_shared` … a k-means table **per layer** × a group absmax scale. The table is fitted
      by applying {@link quant_methods.fit_codebook} to **the original weights** (before rounding)

    NOTE: the "shared" of `kmeans_shared` has the same semantics as the granularity of the same
    name in {@link quant_methods.fake_quant_kmeans} (one table fitted in the value space normalised
    by group absmax), but the table's **reach is closed within the layer** — GPTQ advances one
    stage at a time, so a table spanning every layer would need a separate pass over every target
    before quantization, which does not fit the core of the stage-sequential design (holding H for
    only one stage).

    `fit_stride` is the way out that fits the `kmeans_shared` table from the subsample
    `flat[::fit_stride]` (it is always applied to everything and uses no randomness, so the
    determinism MUST holds). MUST: when it is used, the measuring side's output states it —
    {@link CalibReport.describe} picks it up.
    """

    kind: GridKind = "rtn"
    group_size: int = DEFAULT_GROUP_SIZE
    levels: int = DEFAULT_CODEBOOK_LEVELS
    iterations: int = DEFAULT_CODEBOOK_ITERATIONS
    fit_stride: int = 1

    def __post_init__(self) -> None:
        if self.kind not in GRID_KINDS:
            raise QuantizeError(
                f"storage grid '{self.kind}' is not supported ({', '.join(GRID_KINDS)})"
            )
        if self.group_size < 1:
            raise QuantizeError(f"group_size must be 1 or more (actual {self.group_size})")
        if self.levels < 2:
            raise QuantizeError(f"levels must be 2 or more (actual {self.levels})")
        if self.iterations < 1:
            raise QuantizeError(f"iterations must be 1 or more (actual {self.iterations})")
        if self.fit_stride < 1:
            raise QuantizeError(f"fit_stride must be 1 or more (actual {self.fit_stride})")
        if self.fit_stride != 1 and self.kind != "kmeans_shared":
            raise QuantizeError(
                f"fit_stride is for kmeans_shared only ('{self.kind}' has no table to fit"
                " — and it is not silently ignored either)"
            )

    @property
    def shippable(self) -> bool:
        """Whether this grid's scale ledger may be passed to emit (the MUST of the module
        docstring)."""
        return self.kind == "rtn"


#: The default grid (the RTN grid of i4 g32 = the only shipping path). Held by name so that the
#: argument default is **not a call expression** (an equal frozen dataclass however often it is
#: built, so sharing it is fine).
DEFAULT_GRID = GridSpec()


@dataclass(frozen=True)
class _BoundGrid:
    """A grid bound to one layer (only the `kmeans_shared` table changes per layer).

    MUST: choosing the scale and rounding both stay in **f32** (module docstring — if double
    rounding through f64 moves `scale` by 1 ulp, bit identity with `fake_quant_int4` breaks).
    """

    spec: GridSpec
    where: str
    #: The level table (`rtn` has no table and uses the grid of {@link quantize.quantize_to_int4}
    #: directly).
    table: torch.Tensor | None
    #: The table's largest magnitude (the denominator of the group absmax normalisation).
    max_level: float

    def scale_of(self, block: torch.Tensor) -> torch.Tensor:
        """Choose the scale from the **current values** of one group's columns
        (`[channel, group_size]` F32).

        GPTQ rewrites the remaining columns as it advances, so the group scale is taken from "the
        values at the moment the group's columns are **reached**". Taken from the original weights
        before rounding, values moved by compensation could fall outside the range the scale
        covers.
        """
        if self.table is None:
            return group_scale(block, self.spec.group_size, self.where).squeeze(-1)
        return group_absmax_scale(block, self.spec.group_size, self.max_level, self.where).squeeze(
            -1
        )

    def round_column(self, values: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """Round one column (`[channel]` F32) onto the grid of `scale` (`[channel]` F32).

        `rtn` goes **as is** through {@link quantize.quantize_to_int4} /
        {@link quantize.dequantize_int4} (treating the column as 1-element groups `[channel, 1]`)
        — a detour so that the storage formula and the rounding formula are not separate
        implementations; the per-element operations are identical to the whole-layer version.
        """
        if self.table is None:
            column = values.unsqueeze(-1)
            group = scale.unsqueeze(-1)
            return dequantize_int4(quantize_to_int4(column, group), group).squeeze(-1)
        return round_to_levels(values / scale, self.table) * scale

    def round_rows(self, rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Round the whole layer plainly (fake-quant without GPTQ) — used by AWQ's α search.

        The `rtn` path is **bit-identical** to one layer of {@link quantize.fake_quant_int4}.
        """
        if self.table is None:
            scale = group_scale(rows, self.spec.group_size, self.where)
            return dequantize_int4(quantize_to_int4(rows, scale), scale), scale
        scale = group_absmax_scale(rows, self.spec.group_size, self.max_level, self.where)
        return round_groups_to_levels(rows, self.table, scale, self.where), None


def _bind_grid(spec: GridSpec, fit_source: torch.Tensor, where: str) -> _BoundGrid:
    """Bind a grid spec to one layer (`fit_source` is the weight the `kmeans_shared` table is taken
    from).

    GPTQ passes the weights **before rounding** as `fit_source` — re-taking the table from values
    moved by column compensation would make the table depend on how the blocks advance. AWQ's α
    search passes `W' = W·diag(s)` (what actually gets rounded is `W'`, so taking the table from it
    is the plain fake-quant).
    """
    if spec.kind == "rtn":
        return _BoundGrid(spec=spec, where=where, table=None, max_level=float(INT4_MAX))
    if spec.kind == "nf4":
        table = levels_tensor(NF4_LEVELS)
        return _BoundGrid(spec=spec, where=where, table=table, max_level=float(table.abs().max()))
    scale = group_absmax_scale(fit_source, spec.group_size, 1.0, where)
    grouped = grouped_view(fit_source, spec.group_size, where)
    flat = (grouped / scale.unsqueeze(-1)).reshape(1, -1)
    sampled = flat[:, :: spec.fit_stride] if spec.fit_stride > 1 else flat
    table = fit_codebook(sampled, spec.levels, spec.iterations).reshape(-1)
    return _BoundGrid(spec=spec, where=where, table=table, max_level=1.0)


# ---- GPTQ (per layer) ----------------------------------------------------------


def _inverse_cholesky(
    hessian: torch.Tensor, where: str, damping: float = GPTQ_DAMPING
) -> torch.Tensor:
    """Return the upper-triangular Cholesky factor `U` of the inverse of the damped `H`
    (`H⁻¹ = Uᵀ U`) in F64.

    `λ = damping × mean(diag(H))` is added to the diagonal, so positive definiteness holds even
    with input channels that never moved on the calibration data (diagonal 0). Only when **H is 0
    altogether** is `λ = 0` and H singular, so that fails loudly — so that "a layer outside the
    calibration forward was quantized" is never silent. An explicit `damping = 0` loses the
    positive-definiteness guarantee for the same reason, so it is not accepted here (a sweep's low
    end steps through small positive values, not 0).
    """
    if not damping > 0.0:
        raise QuantizeError(
            f"{where}: damping must be positive (actual {damping}) — with λ = 0 input channels with"
            " a 0 diagonal can remain and H can be singular"
        )
    wide = hessian.to(torch.float64)
    average = float(torch.diagonal(wide).mean())
    # Written as `not > 0` rather than `<= 0` so that NaN fails too (every comparison with NaN is
    # false).
    if not average > 0.0:
        raise QuantizeError(
            f"{where}: the Hessian's diagonal mean is {average:.4g} (not positive) — no calibration"
            " input row flowed, or it was all zero, and with damping λ = 0 H is singular"
        )
    damped = wide.clone()
    damped.diagonal().add_(damping * average)
    try:
        factor = torch.linalg.cholesky(damped)
        return torch.linalg.cholesky(torch.cholesky_inverse(factor), upper=True)
    except RuntimeError as error:  # torch's LinAlgError derives from RuntimeError
        raise QuantizeError(
            f"{where}: the Cholesky decomposition of the Hessian failed ({error})"
        ) from error


def _static_group_scales(
    grid: _BoundGrid, rows: torch.Tensor, group_size: int
) -> list[torch.Tensor]:
    """Settle the group scales up front per contiguous column block of the weights **before
    rounding** (static-groups).

    Dynamic groups (the docstring of {@link _BoundGrid.scale_of}) take them from "the current
    values at the moment the group is reached"; this takes them from the original `rows`, which
    have never seen compensation. Reordering the processing order then moves neither group
    membership nor scales, which is the premise of act-order.
    """
    source = rows.to(torch.float32)
    return [
        grid.scale_of(source[:, head : head + group_size])
        for head in range(0, int(rows.shape[1]), group_size)
    ]


def _activation_order(hessian: torch.Tensor) -> torch.Tensor:
    """Return the column **processing order** by descending `diag(H)` (act-order).

    Ties keep the original column order (`stable=True`) — the determinism MUST, with neither
    randomness nor an implementation-dependent tie-break.
    """
    return torch.argsort(torch.diagonal(hessian), descending=True, stable=True)


def gptq_quantize_rows(
    rows: torch.Tensor,
    hessian: torch.Tensor,
    spec: GridSpec = DEFAULT_GRID,
    where: str = "weight",
    block_size: int = DEFAULT_GPTQ_BLOCK,
    *,
    static_groups: bool = False,
    act_order: bool = False,
    damping: float = GPTQ_DAMPING,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Round a weight folded into `[channel, in]` with GPTQ and return `(rounded rows, scale
    ledger)`.

    `hessian` is the `[in, in]` `H = Σ XᵀX` (`X` = this layer's input `[tokens, in]`). The ledger is
    F32 `[channel, group count]` only when `spec.kind == "rtn"`, and `None` otherwise (the MUST of
    the module docstring).

    The procedure is the standard column-block method:

    1. Take `U = cholesky(H⁻¹, upper)` (`H` is damped — {@link _inverse_cholesky})
    2. Round column `j` onto the grid from its **current value** and form
       `err = (w_j − dq_j) / U[j,j]`
    3. Compensate the remaining columns in the block with `W[:, j+1:] −= err ⊗ U[j, j+1:]`
    4. At the end of the block, update all remaining columns outside the block at once with
       `Err @ U[block, rest]`

    MUST: the ledger returns **the scales used for quantization as is** (the same no-recomputation
    rule as `quantize_to_int8` / `quantize_to_int4`, but it carries more weight in GPTQ) — with RTN
    the group's amax element always lands on `|q| = 7`, so re-deriving gives the same scale, but
    GPTQ can push the amax element down through compensation. In a group pushed down `|q|` does not
    reach 7, and re-deriving from the rounded weights gives not a 1 ulp drift but **a different
    scale**, and the stored values are no longer restored (observed: it does not happen in every
    group, so the test builds its check on a model that contains a group meeting the condition —
    `tests/test_quant_calib.py`).

    NOTE: `block_size` is rounded to a multiple of `group_size` (if a group straddled two blocks,
    "the current values at the moment the group's columns are reached" could not be taken). The
    result does not depend on the block length mathematically.

    `static_groups` / `act_order` / `damping` are **opt-in axes for sweeps** (module docstring):

    - `static_groups=True` … the scales are settled up front from the contiguous column blocks of
      `rows` **before rounding** ({@link _static_group_scales}). The meaning of the ledger does not
      change (**the scales actually used**)
    - `act_order=True` … columns are processed by descending `diag(H)`. `H` and `W` are reordered
      by perm and solved, group membership is looked up from **the original column index**
      (`perm[position] // group_size`), and the rounded result is written straight back to the
      original column (equivalent to upstream's `Q = Q[:, invperm]`). **Requires
      `static_groups=True`**
    - `damping` … the `λ` factor of `_inverse_cholesky`. Left at the default, the path is
      bit-identical

    MUST: the default (both `False` × `damping=GPTQ_DAMPING`) is **bit-identical** to the current
    path.
    """
    if act_order and not static_groups:
        raise QuantizeError(
            f"{where}: act_order requires static_groups=True"
            " (dynamic groups would take scales from the current values in the reordered"
            " processing order, which cannot be mapped back to the original contiguous groups)"
        )
    if rows.dim() != 2:
        raise QuantizeError(
            f"{where}: GPTQ input must be 2-D `[channel, in]` (actual {list(rows.shape)})"
        )
    in_axis = int(rows.shape[1])
    if hessian.dim() != 2 or list(hessian.shape) != [in_axis, in_axis]:
        raise QuantizeError(
            f"{where}: the Hessian must be `[in, in]` = [{in_axis}, {in_axis}]"
            f" (actual {list(hessian.shape)})"
        )
    # Remainder groups fail loudly through the same single path as the storage side (ADR 0069
    # decision 2).
    groups = int(grouped_view(rows, spec.group_size, where).shape[-2])
    grid = _bind_grid(spec, rows, where)
    static = _static_group_scales(grid, rows, spec.group_size) if static_groups else None
    # Reordering applies the same perm to **both** `H` and `W` (the indices of `U` live in the
    # processing-order space too).
    order = _activation_order(hessian) if act_order else None
    upper = _inverse_cholesky(
        hessian if order is None else hessian[order][:, order], where, damping
    )
    block = spec.group_size * max(1, block_size // spec.group_size)

    work = (rows if order is None else rows[:, order]).to(torch.float64)
    # Position in processing order → original column index. `None` is the identity (the default
    # path makes neither a copy nor a tolist).
    columns = None if order is None else order.tolist()
    # The destination is created **on the same device as the input** (the default `torch.empty`
    # is always on the CPU, so the moment GPU weights are passed, the column writes hit a device
    # mismatch). All on CPU, it lives where it always did.
    rounded = torch.empty(rows.shape, dtype=torch.float32, device=rows.device)
    ledger = (
        torch.empty((int(rows.shape[0]), groups), dtype=torch.float32, device=rows.device)
        if spec.shippable
        else None
    )
    if ledger is not None and static is not None:
        for index, fixed in enumerate(static):
            ledger[:, index] = fixed
    # `scale` is always bound at `offset == 0` of each block (static takes it for every column,
    # dynamic at the head of a group, and the block length is a multiple of group_size, so offset 0
    # is always the head of a group).
    for start in range(0, in_axis, block):
        stop = min(start + block, in_axis)
        span = work[:, start:stop].clone()
        residuals = torch.zeros_like(span)
        for offset in range(stop - start):
            position = start + offset
            column = position if columns is None else columns[position]
            if static is not None:
                scale = static[column // spec.group_size]
            elif not offset % spec.group_size:
                # dynamic group — taken from the **current values** at the moment the group's first
                # column is reached.
                scale = grid.scale_of(span[:, offset : offset + spec.group_size].to(torch.float32))
                if ledger is not None:
                    ledger[:, position // spec.group_size] = scale
            quantized = grid.round_column(span[:, offset].to(torch.float32), scale)
            rounded[:, column] = quantized
            residual = (span[:, offset] - quantized.to(torch.float64)) / upper[position, position]
            residuals[:, offset] = residual
            tail = upper[position, position + 1 : stop]
            if tail.numel():
                span[:, offset + 1 :] -= residual.unsqueeze(1) * tail
        if stop < in_axis:
            work[:, stop:] -= residuals @ upper[start:stop, stop:]
    return rounded, ledger


# ---- AWQ (per layer) ----------------------------------------------------------


@dataclass(frozen=True)
class AwqSearch:
    """The result of AWQ's α search (per layer)."""

    #: The chosen α (from the 17 points `{0, 1/16, …, 1}`).
    alpha: float
    #: The objective value at that α, `‖X·W_effᵀ − X·Wᵀ‖²` (accumulated in F64).
    error: float
    #: The per-in-channel equivalent factor `s` (geometric mean 1, `[in]` F64).
    channel_scale: torch.Tensor


def awq_search_scale(
    rows: torch.Tensor,
    act_amax: torch.Tensor,
    samples: torch.Tensor,
    spec: GridSpec = DEFAULT_GRID,
    where: str = "weight",
) -> AwqSearch:
    """Search the per-in-channel equivalent factor `s` from the activation magnitudes (AWQ's α
    sweep).

    `s_j = clamp(act_amax_j, AWQ_ACT_EPS)^α` is normalised to **geometric mean 1**,
    `W' = W·diag(s)` is fake-quantized on the grid of `spec`, and `W_eff = Q(W')/s` is formed. The
    objective is `‖X·W_effᵀ − X·Wᵀ‖²` (F64) over the captured input samples `X`, and the α with the
    smallest value is taken. Ties go to **the smaller α** (the determinism MUST — a strict `<`
    comparison).

    The geometric-mean-1 normalisation is needed because a constant factor on the whole of `s` is
    absorbed by the group absmax scale and has no effect on the objective (without normalisation
    α would mean only "a direction").

    MUST: the returned `s` is **part of the storage**; storing `W_eff` alone does not restore it
    (module docstring — an `s` constant within a group cancels against the absmax scale and returns
    to the equivalent of RTN).
    """
    if rows.dim() != 2:
        raise QuantizeError(
            f"{where}: AWQ input must be 2-D `[channel, in]` (actual {list(rows.shape)})"
        )
    in_axis = int(rows.shape[1])
    if act_amax.dim() != 1 or int(act_amax.shape[0]) != in_axis:
        raise QuantizeError(
            f"{where}: act_amax must be `[in]` = [{in_axis}] (actual {list(act_amax.shape)})"
        )
    if samples.dim() != 2 or int(samples.shape[1]) != in_axis:
        raise QuantizeError(
            f"{where}: input samples must be `[tokens, in]`"
            f" (in = {in_axis}; actual {list(samples.shape)})"
        )
    if not int(samples.shape[0]):
        raise QuantizeError(
            f"{where}: input samples have 0 rows — the α objective would be 0 everywhere"
        )
    grouped_view(rows, spec.group_size, where)  # drop remainder groups before the α loop

    amax = torch.clamp(act_amax.to(torch.float64), min=AWQ_ACT_EPS)
    inputs = samples.to(torch.float64)
    wide = rows.to(torch.float64)
    reference = inputs @ wide.T
    best_alpha = 0.0
    best_error = math.inf
    best_channel = torch.ones(in_axis, dtype=torch.float64, device=rows.device)
    for step in range(AWQ_ALPHA_STEPS + 1):
        alpha = step / AWQ_ALPHA_STEPS
        channel = amax.pow(alpha)
        channel = channel / torch.exp(torch.log(channel).mean())
        scaled = (wide * channel).to(torch.float32)
        quantized, _ = _bind_grid(spec, scaled, where).round_rows(scaled)
        effective = (quantized.to(torch.float64) / channel).to(torch.float32)
        error = float((inputs @ effective.to(torch.float64).T - reference).pow(2).sum())
        if error < best_error:
            best_alpha, best_error, best_channel = alpha, error, channel
    # MUST: NaN is always false under `<`, so when every α is non-finite the initial values (α=0,
    # error=inf) silently remain as "the optimum". Rather than letting the diagnostic turn into a
    # number, fail naming the source of the non-finite values.
    if not math.isfinite(best_error):
        sources = [
            name
            for name, tensor in (
                ("weight", rows),
                ("act_amax", act_amax),
                ("input samples", samples),
            )
            if not bool(torch.isfinite(tensor).all())
        ]
        origin = (
            ", ".join(sources)
            if sources
            else "every input is finite — an intermediate computation overflowed"
        )
        raise QuantizeError(
            f"{where}: the AWQ objective is non-finite for every α (source: {origin})"
        )
    return AwqSearch(alpha=best_alpha, error=best_error, channel_scale=best_channel)


# ---- Stage-sequential driver -------------------------------------------------------


@dataclass(frozen=True)
class LayerCalibReport:
    """The calibration result of one layer (at a granularity that lets a reader of the numbers
    identify the layer)."""

    #: In-model FQN (`<module>.weight`).
    fqn: str
    #: Index of the stage that processed it (0-based).
    stage: int
    #: Number of input rows (tokens) accumulated into this layer's statistics.
    tokens: int
    #: The α AWQ chose (`None` for GPTQ alone).
    alpha: float | None
    #: AWQ's objective value (`None` for GPTQ alone).
    error: float | None


@dataclass(frozen=True)
class CalibReport:
    """Counts and ledger of calibrated rounding (a container in the same style as `Int8Report` /
    `Int4Report`)."""

    method: str
    grid: str
    group_size: int
    stages: int
    elements: int
    fit_stride: int
    layers: tuple[LayerCalibReport, ...]
    #: The shippable scale ledger — non-`None` only for `method="gptq"` × `grid="rtn"`.
    int4: Int4Report | None

    @property
    def modules(self) -> int:
        return len(self.layers)

    def describe(self) -> str:
        text = (
            f"{self.method}/{self.grid}: stages {self.stages} / modules {self.modules}"
            f" / {self.elements:,} elements / group {self.group_size}"
        )
        if self.fit_stride > 1:
            text += f" / fit_stride {self.fit_stride} (table fitted on a subsample)"
        alphas = [layer.alpha for layer in self.layers if layer.alpha is not None]
        if alphas:
            text += f" / AWQ α {min(alphas):.4g}〜{max(alphas):.4g}"
        text += (
            " / scale ledger present (can be passed to emit)"
            if self.int4 is not None
            else " / no scale ledger (storage needs a fold or a companion)"
        )
        return text


@dataclass
class _LayerStats:
    """Calibration statistics of one layer (H is held for only one stage at a time — the core of
    the memory design)."""

    in_features: int
    tokens: int = 0
    hessian: torch.Tensor | None = None
    act_amax: torch.Tensor | None = None
    samples: list[torch.Tensor] = field(default_factory=list)
    sampled: int = 0


def first_tensor_output(output: object) -> torch.Tensor:
    """The default choice of the next stage's input from a stage output (a Tensor as is; the first
    element of a tuple / list).

    Transformer blocks often return a tuple `(hidden, present_kv, …)`, so the default is the first
    element. A stage that uses another position swaps it in with `select_output`.
    """
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, tuple | list) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    raise QuantizeError(
        f"cannot choose the next stage's input from the stage output ({type(output).__name__})"
        " (pass how to choose it with `select_output`)"
    )


def _owner_name(fqn: str) -> str:
    """Recover the owning module name from an `iter_quant_targets` FQN (`<module>.weight`)."""
    return "" if fqn == "weight" else fqn[: -len(".weight")]


def _qualify(prefix: str, local: str) -> str:
    """Lift a local FQN within a stage to an in-model FQN."""
    return f"{prefix}.{local}" if prefix else local


def _stage_device(stage: nn.Module) -> torch.device:
    """The stage's compute device (where its first parameter lives).

    The device is **derived from the stage** rather than taken as an argument, to keep the place
    the weights to round live and the place the observation forward runs structurally the same
    (taken as an argument, a call that moved only one of them could write back "rounding targets
    chosen with GPU activations onto CPU weights").
    """
    parameter = next(stage.parameters(), None)
    if parameter is None:
        raise QuantizeError(
            "the stage has no parameters — cannot derive the calibration compute device"
            " (pass an nn.Module that holds the calibration targets as the stage)"
        )
    return parameter.device


def _moved(value: object, device: torch.device) -> object:
    """Move one element of a batch to `device` (only Tensors and the contents of **plain** tuples /
    lists).

    It looks inside tuples because real-model stages take **tuples** such as rope's `(cos, sin)` as
    keywords — passing them through untouched would leave part of a batch whose placement was
    supposed to be aligned off the compute device, and the stage would hit a device mismatch.
    Conversely, **subclasses** of tuple (namedtuple and the like) are passed through (rebuilding
    can break their set of fields, and torch fails loudly on a tuple that cannot be moved).
    """
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if type(value) is tuple:
        return tuple(_moved(item, device) for item in value)
    if type(value) is list:
        return [_moved(item, device) for item in value]
    return value


def _batch_on(batch: StageBatch, device: torch.device) -> StageBatch:
    """Return a copy of one batch with only its Tensors moved to the stage's device.

    The calibration batches stay **where the caller put them** and are moved only right before
    the stage runs (JIT moves). Keeping every batch resident on the compute device would make the
    weights and the calibration inputs compete for the same VRAM in GPU calibration. `Tensor.to`
    is a no-op returning self on the same device, so the all-CPU default path does not move by a
    single bit.
    """
    args, kwargs = batch
    return (
        tuple(_moved(value, device) for value in args),
        {name: _moved(value, device) for name, value in kwargs.items()},
    )


def _advance_batch(
    stage: nn.Module,
    batch: StageBatch,
    device: torch.device,
    select_output: Callable[[object], torch.Tensor],
) -> torch.Tensor:
    """Run one batch through the rounded stage and return the next stage's input **moved back to
    its original device**.

    It is moved back to keep the placement of the calibration batches unchanged — leaving the
    next stage's inputs on the compute device would keep the activations of every batch resident
    and make the JIT moves of {@link _batch_on} pointless. The round trip is once per batch,
    negligible against one stage's forward and GPTQ solve.
    """
    args = batch[0]
    if not args or not isinstance(args[0], torch.Tensor):
        raise QuantizeError(
            "the first positional argument of the calibration batch is not a Tensor"
            " (hand-off between stages is closed to the form: the only positional argument ="
            " a Tensor)"
        )
    moved_args, moved_kwargs = _batch_on(batch, device)
    return select_output(stage(*moved_args, **moved_kwargs)).detach().to(args[0].device)


def _make_pre_hook(
    stats: _LayerStats, needs_hessian: bool, needs_act: bool, sample_limit: int
) -> Callable[[nn.Module, tuple[object, ...]], None]:
    """A forward_pre_hook that passes the input through while accumulating `H` / activation
    statistics / input samples.

    Not rewriting the activations is the difference from `act_quant.attach_act_quant` — this one
    only observes, and the numbers the calibration forward produces do not differ from the
    original model by a single bit.
    """

    def pre(_module: nn.Module, args: tuple[object, ...]) -> None:
        if not args or not isinstance(args[0], torch.Tensor):
            raise QuantizeError(
                "the linear being calibrated was not called with a positional Tensor"
                " (the input cannot be observed, so H cannot be taken)"
            )
        given = args[0].detach()
        if int(given.shape[-1]) != stats.in_features:
            raise QuantizeError(
                f"the last dimension of the calibration input {int(given.shape[-1])} differs"
                f" from in_features {stats.in_features}"
            )
        plain = given.reshape(-1, stats.in_features)
        wide = plain.to(torch.float64)
        stats.tokens += int(plain.shape[0])
        if needs_hessian:
            gram = wide.T @ wide
            stats.hessian = gram if stats.hessian is None else stats.hessian + gram
        if needs_act:
            amax = wide.abs().amax(dim=0)
            stats.act_amax = amax if stats.act_amax is None else torch.maximum(stats.act_amax, amax)
            room = sample_limit - stats.sampled
            if room > 0:
                taken = plain[:room].to(torch.float32).clone()
                stats.samples.append(taken)
                stats.sampled += int(taken.shape[0])
        return None

    return pre


def calibrate_stages(
    stages: Sequence[StageSpec],
    batches: Sequence[StageBatch],
    method: CalibMethod = "gptq",
    spec: GridSpec = DEFAULT_GRID,
    include: Callable[[str], bool] | None = None,
    block_size: int = DEFAULT_GPTQ_BLOCK,
    samples: int = DEFAULT_AWQ_SAMPLES,
    select_output: Callable[[object], torch.Tensor] = first_tensor_output,
    advance_kwargs: Callable[[int, Mapping[str, object]], Mapping[str, object]] | None = None,
    *,
    static_groups: bool = False,
    act_order: bool = False,
    damping: float = GPTQ_DAMPING,
) -> CalibReport:
    """Apply calibrated rounding while advancing one stage at a time (a driver that **does not
    pass over the calibration data twice**).

    `stages` is a sequence of `(in-model FQN prefix, module)` in execution order, and `batches` a
    sequence of `(positional arguments, keyword arguments)` for **the first stage**. For stage
    `i`:

    1. Attach observation hooks to the target linears, forward every batch, and accumulate `H` /
       activation statistics
    2. Quantize that stage's targets (`H` is dropped here — only **one stage's worth** is held at
       a time)
    3. Forward the stage again **with the quantized weights** to build the next stage's inputs

    Step 3 is the standard GPTQ form with error propagation — later stages round while seeing the
    degradation of earlier ones.

    Hand-off between stages takes the form "the chosen output becomes the **only positional
    argument** of the next stage" (how to choose the output is `select_output`). Keyword arguments
    are treated as unchanged across stages by default; when they need to change, pass
    `advance_kwargs(next stage index, current kwargs) -> kwargs`.

    Targets are limited to `nn.Linear` (and its subclasses) — `H = Σ XᵀX` is closed to the linear
    form "the input's last dimension = the in axis" (conv would need im2col). Enumeration is
    shared with {@link quantize.iter_quant_targets}, and `include` is a predicate on the **module
    FQN** (the local name within the stage).

    When not a single weight could be rounded it fails loudly (the same permanent diagnostic as
    `quantize` / `quant_methods`).

    NOTE (memory): what lives at once is "every input batch of the current stage + every input
    batch of the next stage + the current stage's `H` (F64 `in²` per target linear)". For a
    linear with `in = 4096` one `H` is 128MiB, so the number of targets per stage matters.

    NOTE (device): the compute device is **derived from the stage itself (where its first
    parameter lives)** — it is not taken as an argument. Batches stay where the caller put them
    (the CPU in the default setup) and are moved to the stage's device only right before the
    observation and advance forwards ({@link _batch_on}). The advance outputs are moved back to
    the original device too ({@link _advance_batch}), so only "the weights + `H` + 1 batch" stay
    resident, and the batch count does not affect memory even in GPU calibration. On an all-CPU
    run `Tensor.to` is a no-op returning self, so neither the values nor the order of operations
    of the default path change by a single bit.

    `static_groups` / `act_order` / `damping` pass the sweep axes of {@link gptq_quantize_rows}
    through as is (they matter only for methods that go through GPTQ — the AWQ-only path has no
    `H`). The default is bit-identical to the current path.
    """
    if method not in CALIB_METHODS:
        raise QuantizeError(f"method '{method}' is not supported ({', '.join(CALIB_METHODS)})")
    if not stages:
        raise QuantizeError("there are no stages (pass the stage sequence in execution order)")
    if not batches:
        raise QuantizeError("there are no calibration batches")
    if samples < 1:
        raise QuantizeError(f"samples must be 1 or more (actual {samples})")
    needs_hessian = method in ("gptq", "awq+gptq")
    needs_act = method in ("awq", "awq+gptq")
    if not needs_hessian and (static_groups or act_order or damping != GPTQ_DAMPING):
        # The same style as `GridSpec.fit_stride` — an axis that has no effect is not silently
        # ignored when passed.
        raise QuantizeError(
            f"method '{method}' does not go through GPTQ, so static_groups / act_order / damping"
            " have no effect (and are not silently ignored either)"
        )

    current: list[StageBatch] = [(tuple(args), dict(kwargs)) for args, kwargs in batches]
    reports: list[LayerCalibReport] = []
    scales: dict[str, torch.Tensor] = {}
    elements = 0

    with torch.no_grad():
        for index, (prefix, stage) in enumerate(stages):
            targets = list(iter_quant_targets(stage, (nn.Linear,), include))
            stats = _observe_stage(stage, targets, current, needs_hessian, needs_act, samples)
            for local, weight, axis in targets:
                fqn = _qualify(prefix, local)
                report, ledger = _quantize_layer(
                    stats[local],
                    weight,
                    axis,
                    index,
                    fqn,
                    method,
                    spec,
                    block_size,
                    static_groups=static_groups,
                    act_order=act_order,
                    damping=damping,
                )
                reports.append(report)
                if ledger is not None:
                    scales[fqn] = ledger
                elements += weight.numel()
            stats.clear()
            if index + 1 < len(stages):
                device = _stage_device(stage)
                current = [
                    (
                        (_advance_batch(stage, (args, kwargs), device, select_output),),
                        dict(advance_kwargs(index + 1, kwargs) if advance_kwargs else kwargs),
                    )
                    for args, kwargs in current
                ]

    if not reports:
        raise QuantizeError(
            "calibrated rounding was requested but there is no target nn.Linear"
            " (check the spelling of the stages or include)"
        )
    return CalibReport(
        method=method,
        grid=spec.kind,
        group_size=spec.group_size,
        stages=len(stages),
        elements=elements,
        fit_stride=spec.fit_stride,
        layers=tuple(reports),
        int4=(
            Int4Report(
                scales=scales, group_size=spec.group_size, modules=len(scales), elements=elements
            )
            if scales
            else None
        ),
    )


def _observe_stage(
    stage: nn.Module,
    targets: Sequence[tuple[str, torch.Tensor, int]],
    batches: Sequence[StageBatch],
    needs_hessian: bool,
    needs_act: bool,
    sample_limit: int,
) -> dict[str, _LayerStats]:
    """Attach the hooks, pass over every batch once, take the calibration statistics per target,
    and **always detach** the hooks."""
    device = _stage_device(stage)
    lookup = dict(stage.named_modules())
    stats: dict[str, _LayerStats] = {}
    handles = []
    for local, weight, _axis in targets:
        layer = _LayerStats(in_features=int(weight.shape[-1]))
        stats[local] = layer
        handles.append(
            lookup[_owner_name(local)].register_forward_pre_hook(
                _make_pre_hook(layer, needs_hessian, needs_act, sample_limit), with_kwargs=False
            )
        )
    try:
        for batch in batches:
            moved_args, moved_kwargs = _batch_on(batch, device)
            stage(*moved_args, **moved_kwargs)
    finally:
        for handle in handles:
            handle.remove()
    return stats


def _quantize_layer(
    stats: _LayerStats,
    weight: torch.Tensor,
    axis: int,
    stage: int,
    fqn: str,
    method: CalibMethod,
    spec: GridSpec,
    block_size: int,
    *,
    static_groups: bool = False,
    act_order: bool = False,
    damping: float = GPTQ_DAMPING,
) -> tuple[LayerCalibReport, torch.Tensor | None]:
    """Round one layer, write it back in place, and return `(layer report, scale ledger or None)`.

    The AWQ family rounds `W' = W·diag(s)` and then writes back `W_eff = Q(W')/s`, so the returned
    ledger is always `None` (the group amax of `W_eff` does not correspond to the scale of
    `Q(W')` — the fold / companion of the module docstring). `awq+gptq` transforms the Hessian to
    `H' = diag(1/s)·H·diag(1/s)` too, since the effective input becomes `x/s`.
    """
    where = f"'{fqn}'"
    if not stats.tokens:
        raise QuantizeError(
            f"{where}: no calibration input row flowed (not in the forward of stage {stage})"
        )
    rows = channel_rows(weight, axis)
    alpha: float | None = None
    error: float | None = None
    ledger: torch.Tensor | None = None

    if method == "gptq":
        rounded, ledger = gptq_quantize_rows(
            rows,
            stats.hessian,
            spec,
            where,
            block_size,
            static_groups=static_groups,
            act_order=act_order,
            damping=damping,
        )
    else:
        search = awq_search_scale(rows, stats.act_amax, torch.cat(stats.samples), spec, where)
        alpha, error = search.alpha, search.error
        channel = search.channel_scale
        scaled = (rows.to(torch.float64) * channel).to(torch.float32)
        if method == "awq+gptq":
            inverse = 1.0 / channel
            transformed = stats.hessian * inverse.unsqueeze(0) * inverse.unsqueeze(1)
            quantized, _ = gptq_quantize_rows(
                scaled,
                transformed,
                spec,
                where,
                block_size,
                static_groups=static_groups,
                act_order=act_order,
                damping=damping,
            )
        else:
            quantized, _ = _bind_grid(spec, scaled, where).round_rows(scaled)
        rounded = (quantized.to(torch.float64) / channel).to(torch.float32)

    weight.copy_(restore_channel_rows(rounded, weight, axis))
    return (
        LayerCalibReport(fqn=fqn, stage=stage, tokens=stats.tokens, alpha=alpha, error=error),
        ledger,
    )
