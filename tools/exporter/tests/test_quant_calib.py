"""Calibrated rounding (GPTQ / AWQ) — the storage grid stays that of the first step; only the
rounding targets are chosen again.

Six points are checked: (1) same input → **bit-identical** (no random numbers); (2) rounded values
lie on the **storage grid** (`q·scale`, `|q| ≤ 7`); (3) with a diagonal `H` the result is
**bit-identical** to `quantize.fake_quant_int4` (the compensation term vanishes as a mathematical
consequence = an oracle that is not tautological); (4) with correlated inputs GPTQ's `H`-weighted
error is **smaller** than RTN's (the counterpart showing the method actually works); (5) the
**error propagation** of the stage-by-stage pass reaches later stages (fault injection: stopping
the propagation changes the result); (6) the sweep axes (`static_groups` / `act_order` /
`damping`) are **bit-identical to the existing path by default**, and when enabled move nothing of
the storage form (original column order, contiguous groups, group scales).
"""

from __future__ import annotations

import copy
import math

import pytest
import torch
from torch import nn
from torch.nn import functional

from karume.quant_calib import (
    AWQ_ALPHA_STEPS,
    DEFAULT_GRID,
    GPTQ_DAMPING,
    CalibReport,
    GridSpec,
    awq_search_scale,
    calibrate_stages,
    first_tensor_output,
    gptq_quantize_rows,
)
from karume.quant_methods import NF4_LEVELS, group_absmax_scale, levels_tensor
from karume.quantize import (
    INT4_MAX,
    QuantizeError,
    channel_rows,
    dequantize_int4,
    fake_quant_int4,
    group_scale,
    grouped_view,
    quantize_to_int4,
)

#: Group length of the calibration tests (short enough to divide the fixtures' in axes 32 / 128).
GROUP = 8

#: Default grid with `group_size=GROUP` (the shipping path = RTN).
GRID = GridSpec(group_size=GROUP)


def weights(rows: int, columns: int, seed: int) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(rows, columns) * 0.05


def correlated_inputs(tokens: int, columns: int, seed: int) -> torch.Tensor:
    """Calibration input with correlated input channels (`H` far from diagonal — the condition
    under which GPTQ helps)."""
    torch.manual_seed(seed)
    mixing = torch.randn(columns, columns) * 0.3 + torch.eye(columns)
    return torch.randn(tokens, columns) @ mixing


def gram(inputs: torch.Tensor) -> torch.Tensor:
    """`H = Σ XᵀX` (F64)."""
    wide = inputs.detach().to(torch.float64)
    return wide.T @ wide


def weighted_error(delta: torch.Tensor, hessian: torch.Tensor) -> float:
    """`tr(ΔW·H·ΔWᵀ)` — exactly the quantity GPTQ tries to minimize."""
    wide = delta.to(torch.float64)
    return float((wide @ hessian * wide).sum())


def round_to_nearest(rows: torch.Tensor, group_size: int) -> torch.Tensor:
    """RTN for comparison (the `quantize` grid; rounding that ignores calibration data)."""
    scale = group_scale(rows, group_size)
    return dequantize_int4(quantize_to_int4(rows, scale), scale)


class Chain(nn.Module):
    """A stage returning a `(hidden, state)` tuple (the return shape of a transformer block).

    The effect of kwargs is observed through `shift` (an **addition** to the proj input). Not a
    multiplication, because GPTQ is **invariant** to scaling `H` by a constant (with `H → cH`,
    damping and `U` move by the same ratio and the compensation cancels out) — with a factor, the
    result could not tell "did the kwargs arrive".
    """

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.proj = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor, shift: float = 0.0) -> tuple[torch.Tensor, str]:
        return self.proj(x + shift), "state"


class Roped(nn.Module):
    """A stage taking a **tuple** like rope's `(cos, sin)` by keyword (as a real DiT block does)."""

    def __init__(self, features: int) -> None:
        super().__init__()
        self.proj = nn.Linear(features, features)

    def forward(self, x: torch.Tensor, rotary: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        cos, sin = rotary
        return self.proj(x * cos + sin)


class Pair(nn.Module):
    """Two linears in one stage (to see the effect of `include`)."""

    def __init__(self, features: int) -> None:
        super().__init__()
        self.left = nn.Linear(features, features)
        self.right = nn.Linear(features, features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.left(x) + self.right(x)


class TestGridSpec:
    def test_only_rtn_is_shippable(self):
        """Only the RTN grid may be passed to emit (the MUST of the module docstring)."""
        assert GridSpec(kind="rtn").shippable
        assert not GridSpec(kind="nf4").shippable
        assert not GridSpec(kind="kmeans_shared").shippable

    def test_the_default_grid_is_i4_g32_rtn(self):
        assert DEFAULT_GRID.kind == "rtn"
        assert DEFAULT_GRID.group_size == 32

    def test_an_unknown_grid_fails_loudly(self):
        with pytest.raises(QuantizeError, match="is not supported"):
            GridSpec(kind="int3")  # type: ignore[arg-type]

    def test_a_stride_on_a_grid_without_a_fitted_table_fails_loudly(self):
        """`fit_stride` is only for granularities that fit a table (the same convention as
        `quant_methods`)."""
        with pytest.raises(QuantizeError, match="is for kmeans_shared only"):
            GridSpec(kind="rtn", fit_stride=2)

    def test_a_non_positive_stride_fails_loudly(self):
        with pytest.raises(QuantizeError, match="must be 1 or more"):
            GridSpec(kind="kmeans_shared", fit_stride=0)


class TestGptqRows:
    def test_a_diagonal_hessian_reproduces_the_storage_round(self):
        """**Oracle**: with a diagonal `H` the compensation vanishes, bit-identical to
        `fake_quant_int4`.

        `U = cholesky(H⁻¹, upper)` is diagonal, so `U[j, j+1:]` is exactly 0 and
        `W[:, j+1:] −= err ⊗ 0` moves not a single bit — a mathematical consequence, not a
        property of the implementation.
        """
        module = nn.Linear(32, 6, bias=False)
        twin = copy.deepcopy(module)
        fake_quant_int4(twin, group_size=GROUP)

        rounded, ledger = gptq_quantize_rows(
            channel_rows(module.weight.detach(), 0), torch.eye(32, dtype=torch.float64), GRID
        )

        assert torch.equal(rounded, twin.weight.detach())
        assert ledger is not None
        assert torch.equal(ledger, group_scale(channel_rows(twin.weight.detach(), 0), GROUP))

    def test_the_oracle_is_not_vacuous(self):
        """Counterpart of the above — a correlated `H` gives a result **different** from RTN (it
        moves when not diagonal)."""
        rows = weights(6, 32, seed=1)

        rounded, _ = gptq_quantize_rows(rows, gram(correlated_inputs(96, 32, seed=2)), GRID)

        assert not torch.equal(rounded, round_to_nearest(rows, GROUP))

    def test_every_value_lands_on_the_int4_grid(self):
        """Every rounded element lies on `q·scale` (`|q| ≤ 7`, −8 unused) = a shippable form."""
        rows = weights(6, 32, seed=3)

        rounded, ledger = gptq_quantize_rows(rows, gram(correlated_inputs(96, 32, seed=4)), GRID)

        assert ledger is not None
        quantized = quantize_to_int4(rounded, ledger)
        assert int(quantized.abs().amax()) <= INT4_MAX
        assert torch.equal(dequantize_int4(quantized, ledger), rounded)

    def test_the_ledger_is_the_scale_that_was_used_not_a_recomputed_one(self):
        """MUST: the ledger is **the very values used for quantization** (re-deriving from the
        rounded values gives something else).

        With RTN the amax element of a group always lands on `|q| = 7`, so re-deriving gives the
        same scale, but GPTQ's compensation can push the amax element down. In a group pushed
        down, `|q|` does not reach 7, and applying `group_scale` to the rounded values again gives
        **a different scale**, so the stored values are not recovered.

        This model **contains a group that meets that condition** (the `peak` assert below pins
        the condition itself, so a fixture change that removes it is noticed here).
        """
        rows = weights(6, 32, seed=5)

        rounded, ledger = gptq_quantize_rows(rows, gram(correlated_inputs(96, 32, seed=2)), GRID)

        assert ledger is not None
        peak = quantize_to_int4(rounded, ledger).abs().reshape(rows.shape[0], -1, GROUP).amax(-1)
        assert int((peak < INT4_MAX).sum()), "no group has its amax element pushed down"

        recomputed = group_scale(rounded, GROUP)
        assert not torch.equal(ledger, recomputed)
        # Recovery works only with the ledger (= confirms the ledger holds "the values used").
        assert torch.equal(dequantize_int4(quantize_to_int4(rounded, ledger), ledger), rounded)
        assert not torch.equal(
            dequantize_int4(quantize_to_int4(rounded, recomputed), recomputed), rounded
        )

    def test_correlated_inputs_beat_round_to_nearest(self):
        """With correlated inputs GPTQ's `H`-weighted error `tr(ΔW·H·ΔWᵀ)` is smaller than RTN's."""
        rows = weights(24, 128, seed=7)
        hessian = gram(correlated_inputs(512, 128, seed=8))

        rounded, _ = gptq_quantize_rows(rows, hessian, GridSpec(group_size=32))

        assert weighted_error(rounded - rows, hessian) < weighted_error(
            round_to_nearest(rows, 32) - rows, hessian
        )

    def test_the_same_input_gives_a_bit_identical_result(self):
        """MUST: no random numbers at all (varying the seed does not move the result)."""
        rows = weights(6, 32, seed=9)
        hessian = gram(correlated_inputs(96, 32, seed=10))

        first, _ = gptq_quantize_rows(rows, hessian, GRID)
        torch.manual_seed(999)
        second, _ = gptq_quantize_rows(rows, hessian, GRID)

        assert torch.equal(first, second)

    @pytest.mark.parametrize("kind", ["nf4", "kmeans_shared"])
    def test_a_grid_without_a_storage_path_returns_no_ledger(self, kind):
        """The same MUST as `quant_methods` — no emit route for a method without a storage path."""
        rows = weights(6, 32, seed=11)

        rounded, ledger = gptq_quantize_rows(
            rows, gram(correlated_inputs(96, 32, seed=12)), GridSpec(kind=kind, group_size=GROUP)
        )

        assert ledger is None
        assert rounded.shape == rows.shape

    def test_the_nf4_grid_lands_on_the_nf4_table(self):
        """Swapping the grid swaps the rounding targets (rules out a tautology where only `rtn`
        exists)."""
        rows = weights(6, 32, seed=13)
        table = levels_tensor(NF4_LEVELS)

        rounded, _ = gptq_quantize_rows(
            rows, torch.eye(32, dtype=torch.float64), GridSpec(kind="nf4", group_size=GROUP)
        )

        scale = group_absmax_scale(rows, GROUP, float(table.abs().max()))
        grouped = grouped_view(rounded, GROUP, "check")
        allowed = scale.unsqueeze(-1).unsqueeze(-1) * table
        assert bool((grouped.unsqueeze(-1) == allowed).any(dim=-1).all())

    def test_an_axis_the_group_size_does_not_divide_fails_loudly(self):
        """No partial groups (shared with `quantize.grouped_view` — ADR 0069 decision 2)."""
        with pytest.raises(QuantizeError, match="is not divisible by group_size"):
            gptq_quantize_rows(weights(3, 20, seed=14), torch.eye(20, dtype=torch.float64), GRID)

    def test_a_hessian_that_never_saw_an_input_fails_loudly(self):
        """An all-zero `H` makes damping 0 too and is singular — no silent approximation."""
        with pytest.raises(QuantizeError, match="diagonal mean"):
            gptq_quantize_rows(weights(3, 32, seed=15), torch.zeros(32, 32), GRID)

    def test_a_hessian_of_the_wrong_shape_fails_loudly(self):
        with pytest.raises(QuantizeError, match=r"\[in, in\]"):
            gptq_quantize_rows(weights(3, 32, seed=16), torch.eye(16, dtype=torch.float64), GRID)


class TestGptqOptions:
    """Sweep axes `static_groups` / `act_order` / `damping` (opt-in; the default is bit-identical to
    the current path).

    Three points are checked: (1) left at the default, not a single bit moves; (2) the scale of
    `static_groups` comes from **the original contiguous column block before rounding**;
    (3) `act_order` changes only the processing order and moves nothing of the storage form
    (original column order, contiguous groups, the group-scale ledger).
    """

    @staticmethod
    def fixture(seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        """A small model with `H` far from diagonal (= the compensation actually acts)."""
        return weights(6, 32, seed=seed), gram(correlated_inputs(96, 32, seed=seed + 1))

    @staticmethod
    def order_of(hessian: torch.Tensor) -> torch.Tensor:
        """Processing order by descending `diag(H)` (the same stable sort as the implementation)."""
        return torch.argsort(torch.diagonal(hessian), descending=True, stable=True)

    def test_the_options_off_are_bit_identical_to_the_call_without_them(self):
        """MUST: adding the sweep axes moves not a bit of the default path (reproducibility of the
        existing i4 distribution assets)."""
        rows, hessian = self.fixture(seed=101)

        bare, bare_ledger = gptq_quantize_rows(rows, hessian, GRID)
        off, off_ledger = gptq_quantize_rows(
            rows, hessian, GRID, static_groups=False, act_order=False, damping=GPTQ_DAMPING
        )

        assert torch.equal(bare, off)
        assert bare_ledger is not None
        assert off_ledger is not None
        assert torch.equal(bare_ledger, off_ledger)

    def test_static_groups_takes_the_scale_from_the_original_block(self):
        """The `static_groups=True` ledger = the group scales of the contiguous column blocks of
        `rows` **before rounding**.

        The counterpart (ruling out a tautology) is the dynamic side — it takes the compensated
        current values, so it is **something else**. On a model with no difference this check
        says nothing, so the difference itself is pinned by an assert.
        """
        rows, hessian = self.fixture(seed=103)

        static, static_ledger = gptq_quantize_rows(rows, hessian, GRID, static_groups=True)
        dynamic, dynamic_ledger = gptq_quantize_rows(rows, hessian, GRID)

        assert static_ledger is not None
        assert dynamic_ledger is not None
        assert torch.equal(static_ledger, group_scale(rows, GROUP))
        assert not torch.equal(dynamic_ledger, static_ledger), (
            "the model shows no difference from dynamic"
        )
        assert not torch.equal(static, dynamic)
        # The ledger stays "the very scale used" = the stored values are recovered (the MUST holds
        # with static groups too).
        assert torch.equal(
            dequantize_int4(quantize_to_int4(static, static_ledger), static_ledger), static
        )

    def test_a_diagonal_hessian_reproduces_the_storage_round_with_static_groups(self):
        """The oracle holds with static groups too — a diagonal `H` has zero compensation, so
        dynamic and static agree."""
        module = nn.Linear(32, 6, bias=False)
        twin = copy.deepcopy(module)
        fake_quant_int4(twin, group_size=GROUP)

        rounded, ledger = gptq_quantize_rows(
            channel_rows(module.weight.detach(), 0),
            torch.eye(32, dtype=torch.float64),
            GRID,
            static_groups=True,
        )

        assert torch.equal(rounded, twin.weight.detach())
        assert ledger is not None
        assert torch.equal(ledger, group_scale(channel_rows(twin.weight.detach(), 0), GROUP))

    def test_act_order_writes_the_columns_back_in_the_original_order(self):
        """**Oracle**: with a diagonal `H`, reordering is still bit-identical to `fake_quant_int4`.

        The compensation term is exactly 0, so the rounding does not depend on the processing
        order. Agreement therefore proves "the reordered columns go back **to their original
        columns**, and group membership stays the original contiguous blocks" (equivalent to the
        upstream `Q = Q[:, invperm]`). The diagonal values are non-uniform so the perm is **not
        the identity** — an identity perm would not check the reordering.
        """
        module = nn.Linear(32, 6, bias=False)
        twin = copy.deepcopy(module)
        fake_quant_int4(twin, group_size=GROUP)
        diagonal = torch.arange(32, 0, -1, dtype=torch.float64).roll(7)
        assert not torch.equal(self.order_of(torch.diag(diagonal)), torch.arange(32))

        rounded, ledger = gptq_quantize_rows(
            channel_rows(module.weight.detach(), 0),
            torch.diag(diagonal),
            GRID,
            static_groups=True,
            act_order=True,
        )

        assert torch.equal(rounded, twin.weight.detach())
        assert ledger is not None
        assert torch.equal(ledger, group_scale(channel_rows(module.weight.detach(), 0), GROUP))

    def test_act_order_changes_the_result_but_not_the_storage_shape(self):
        """Counterpart — with a correlated `H` the processing order really matters. Even so
        **nothing of the storage form moves**."""
        rows, hessian = self.fixture(seed=105)
        assert not torch.equal(self.order_of(hessian), torch.arange(32))

        ordered, ordered_ledger = gptq_quantize_rows(
            rows, hessian, GRID, static_groups=True, act_order=True
        )
        plain, plain_ledger = gptq_quantize_rows(rows, hessian, GRID, static_groups=True)

        assert not torch.equal(ordered, plain)
        assert ordered_ledger is not None
        assert plain_ledger is not None
        # Group alignment (original order, contiguous groups) is unchanged = zero runtime changes.
        assert torch.equal(ordered_ledger, plain_ledger)
        quantized = quantize_to_int4(ordered, ordered_ledger)
        assert int(quantized.abs().amax()) <= INT4_MAX
        assert torch.equal(dequantize_int4(quantized, ordered_ledger), ordered)

    def test_act_order_without_static_groups_fails_loudly(self):
        """Reordering dynamic groups cannot map group membership back — no silent approximation."""
        rows, hessian = self.fixture(seed=106)

        with pytest.raises(QuantizeError, match="requires static_groups=True"):
            gptq_quantize_rows(rows, hessian, GRID, act_order=True)

    def test_the_explicit_default_damping_matches_the_constant(self):
        rows, hessian = self.fixture(seed=107)

        explicit, _ = gptq_quantize_rows(rows, hessian, GRID, damping=GPTQ_DAMPING)
        default, _ = gptq_quantize_rows(rows, hessian, GRID)

        assert torch.equal(explicit, default)

    def test_damping_reaches_the_solver(self):
        """Counterpart — `damping` reaches `λ` of `_inverse_cholesky` and moves the result."""
        rows = weights(24, 128, seed=108)
        hessian = gram(correlated_inputs(512, 128, seed=109))
        grid = GridSpec(group_size=32)

        light, _ = gptq_quantize_rows(rows, hessian, grid, damping=0.001)
        heavy, _ = gptq_quantize_rows(rows, hessian, grid, damping=0.1)

        assert not torch.equal(light, heavy)

    @pytest.mark.parametrize("damping", [0.0, -0.01])
    def test_a_non_positive_damping_fails_loudly(self, damping):
        """With `λ = 0`, input channels with a zero diagonal remain and `H` can become singular."""
        rows, hessian = self.fixture(seed=110)

        with pytest.raises(QuantizeError, match="damping must be positive"):
            gptq_quantize_rows(rows, hessian, GRID, damping=damping)


class TestAwqSearch:
    def test_a_flat_activation_profile_keeps_alpha_zero(self):
        """With all channels the same size, `s` is 1 for any α — ties go to the smaller α
        (determinism)."""
        rows = weights(6, 32, seed=17)

        search = awq_search_scale(
            rows, torch.ones(32, dtype=torch.float64), correlated_inputs(64, 32, seed=18), GRID
        )

        assert search.alpha == 0.0
        assert torch.equal(search.channel_scale, torch.ones(32, dtype=torch.float64))

    def test_a_skewed_activation_profile_moves_alpha_off_zero(self):
        """Counterpart — with an outlier channel, α > 0 **actually** lowers the objective."""
        torch.manual_seed(19)
        act_amax = torch.rand(32, dtype=torch.float64) * 0.1 + 0.01
        act_amax[5] = 20.0
        act_amax[21] = 12.0

        search = awq_search_scale(
            weights(6, 32, seed=20), act_amax, correlated_inputs(64, 32, seed=21), GRID
        )

        assert search.alpha > 0.0
        assert search.alpha <= 1.0

    def test_the_channel_scale_has_geometric_mean_one(self):
        """A constant factor on all of `s` is absorbed by the absmax scale, so without normalization
        α loses its meaning."""
        torch.manual_seed(22)
        act_amax = torch.rand(32, dtype=torch.float64) * 5.0 + 0.01

        search = awq_search_scale(
            weights(6, 32, seed=23), act_amax, correlated_inputs(64, 32, seed=24), GRID
        )

        assert float(torch.log(search.channel_scale).mean()) == pytest.approx(0.0, abs=1e-10)

    def test_the_alpha_grid_has_seventeen_points(self):
        """α ∈ {0, 1/16, …, 1} (the chosen value is on the grid)."""
        torch.manual_seed(25)
        act_amax = torch.rand(32, dtype=torch.float64) * 5.0 + 0.01

        search = awq_search_scale(
            weights(6, 32, seed=26), act_amax, correlated_inputs(64, 32, seed=27), GRID
        )

        assert search.alpha in {step / AWQ_ALPHA_STEPS for step in range(AWQ_ALPHA_STEPS + 1)}

    def test_a_non_finite_act_amax_silently_collapses_alpha_to_zero(self):
        """A non-finite `act_amax` turns every α>0 objective into NaN, and α=0 silently remains.

        `NaN < best_error` is always false, so no α>0 point is ever chosen. Only for α=0 does
        `amax.pow(0)` return 1 even for non-finite elements, so it stays finite, and the return
        value cannot be told apart from "α=0 was optimal" (a finite error, an `s` with geometric
        mean 1). The counterpart is test_a_skewed_activation_profile_moves_alpha_off_zero, where
        α>0 wins with the same skew.
        """
        torch.manual_seed(19)
        act_amax = torch.rand(32, dtype=torch.float64) * 0.1 + 0.01
        act_amax[5] = 20.0
        act_amax[21] = 12.0
        control = awq_search_scale(
            weights(6, 32, seed=20), act_amax, correlated_inputs(64, 32, seed=21), GRID
        )
        act_amax[5] = float("inf")

        search = awq_search_scale(
            weights(6, 32, seed=20), act_amax, correlated_inputs(64, 32, seed=21), GRID
        )

        assert control.alpha > 0.0, (
            "counterpart — before mixing in non-finite values, α>0 lowers the objective"
        )
        assert search.alpha == 0.0
        assert math.isfinite(search.error), (
            "only the α=0 objective is finite (why the silence is invisible)"
        )
        assert torch.equal(search.channel_scale, torch.ones(32, dtype=torch.float64))
        assert search.channel_scale.device == act_amax.device, (
            "`s` comes out on the same device as the input"
        )

    def test_a_non_finite_weight_fails_loudly_instead_of_reporting_alpha_zero(self):
        """A NaN in the weight turns the objective of every α, α=0 included, into NaN — the initial
        value is not taken silently."""
        rows = weights(6, 32, seed=17)
        rows[0, 0] = float("nan")

        with pytest.raises(
            QuantizeError,
            match=r"blocks\.0\.proj\.weight: .*non-finite for every α \(source: weight\)",
        ):
            awq_search_scale(
                rows,
                torch.ones(32, dtype=torch.float64),
                correlated_inputs(64, 32, seed=18),
                GRID,
                "blocks.0.proj.weight",
            )

    def test_no_sample_fails_loudly(self):
        with pytest.raises(QuantizeError, match="have 0 rows"):
            awq_search_scale(
                weights(6, 32, seed=28), torch.ones(32, dtype=torch.float64), torch.zeros(0, 32)
            )

    def test_an_axis_the_group_size_does_not_divide_fails_loudly(self):
        with pytest.raises(QuantizeError, match="is not divisible by group_size"):
            awq_search_scale(
                weights(3, 20, seed=29),
                torch.ones(20, dtype=torch.float64),
                correlated_inputs(16, 20, seed=30),
                GRID,
            )


class TestCalibrateStages:
    @staticmethod
    def two_stages(seed: int = 31) -> list[tuple[str, nn.Module]]:
        torch.manual_seed(seed)
        return [("block.0", nn.Linear(32, 32)), ("block.1", nn.Linear(32, 16))]

    @staticmethod
    def batch(seed: int = 32) -> list[tuple[tuple[object, ...], dict[str, object]]]:
        return [((correlated_inputs(24, 32, seed=seed),), {})]

    def test_the_same_input_gives_a_bit_identical_result(self):
        """MUST: same input → bit-identical (no random numbers; varying the seed moves nothing)."""
        stages = self.two_stages()
        twin = [(prefix, copy.deepcopy(module)) for prefix, module in stages]
        batches = self.batch()

        calibrate_stages(stages, batches, spec=GRID)
        torch.manual_seed(999)
        calibrate_stages(twin, batches, spec=GRID)

        for (_, quantized), (_, other) in zip(stages, twin, strict=True):
            assert torch.equal(quantized.weight.detach(), other.weight.detach())

    def test_the_ledger_keys_are_model_wide_fqns(self):
        """Ledger keys are **model-wide FQNs**, not stage-local names (the same space as emit)."""
        report = calibrate_stages(self.two_stages(), self.batch(), spec=GRID)

        assert isinstance(report, CalibReport)
        assert report.int4 is not None
        assert sorted(report.int4.scales) == ["block.0.weight", "block.1.weight"]
        assert report.int4.group_size == GROUP
        assert [layer.fqn for layer in report.layers] == ["block.0.weight", "block.1.weight"]
        assert all(layer.tokens == 24 for layer in report.layers)
        assert report.modules == 2

    def test_stage_one_quantization_reaches_the_stage_two_hessian(self):
        """Error propagation: the later stage's `H` is taken from the earlier stage's output **after
        quantization**.

        The fault injected is "an `H` with propagation stopped" — rounding from an `H` taken on
        the original weights' output changes the result. If it did not, the re-forward of step 3
        would serve no purpose.
        """
        stages = self.two_stages()
        (_, first), (_, second) = stages
        original_first = first.weight.detach().clone()
        original_second = second.weight.detach().clone()
        bias = first.bias.detach().clone()
        inputs = correlated_inputs(24, 32, seed=32)

        calibrate_stages(stages, self.batch(), spec=GRID)

        expected_first, _ = gptq_quantize_rows(original_first, gram(inputs), GRID)
        assert torch.equal(first.weight.detach(), expected_first)

        propagated = functional.linear(inputs, first.weight.detach(), bias)
        expected_second, _ = gptq_quantize_rows(original_second, gram(propagated), GRID)
        assert torch.equal(second.weight.detach(), expected_second)

        stale = functional.linear(inputs, original_first, bias)
        without_propagation, _ = gptq_quantize_rows(original_second, gram(stale), GRID)
        assert not torch.equal(without_propagation, expected_second)

    def test_only_included_modules_are_touched(self):
        """`include` is a predicate on **module FQNs local to the stage** (the same meaning as in
        `quantize`)."""
        torch.manual_seed(46)
        stage = Pair(32)
        untouched = stage.right.weight.detach().clone()
        before = stage.left.weight.detach().clone()

        report = calibrate_stages(
            [("pair", stage)], self.batch(), spec=GRID, include=lambda fqn: fqn == "left"
        )

        assert [layer.fqn for layer in report.layers] == ["pair.left.weight"]
        assert torch.equal(stage.right.weight.detach(), untouched)
        assert not torch.equal(stage.left.weight.detach(), before)

    def test_a_stage_without_a_target_fails_loudly(self):
        """Does not silence "a method was given but matched 0 modules" (the same standing
        diagnostic as in `quantize`)."""
        with pytest.raises(QuantizeError, match=r"there is no target nn\.Linear"):
            calibrate_stages([("norm", nn.LayerNorm(32))], self.batch(), spec=GRID)
        with pytest.raises(QuantizeError, match=r"there is no target nn\.Linear"):
            calibrate_stages(self.two_stages(), self.batch(), spec=GRID, include=lambda _fqn: False)

    def test_no_stage_fails_loudly(self):
        with pytest.raises(QuantizeError, match="there are no stages"):
            calibrate_stages([], self.batch(), spec=GRID)

    def test_no_batch_fails_loudly(self):
        with pytest.raises(QuantizeError, match="there are no calibration batches"):
            calibrate_stages(self.two_stages(), [], spec=GRID)

    def test_an_unknown_method_fails_loudly(self):
        with pytest.raises(QuantizeError, match="is not supported"):
            calibrate_stages(self.two_stages(), self.batch(), method="smooth", spec=GRID)  # type: ignore[arg-type]

    def test_an_axis_the_group_size_does_not_divide_fails_loudly(self):
        torch.manual_seed(33)
        with pytest.raises(QuantizeError, match="is not divisible by group_size"):
            calibrate_stages(
                [("odd", nn.Linear(20, 4))],
                [((correlated_inputs(8, 20, seed=34),), {})],
                spec=GRID,
            )

    def test_a_layer_the_calibration_never_reaches_fails_loudly(self):
        """A linear not on the forward path (= no `H` can be taken) is not rounded silently."""

        class Detour(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.used = nn.Linear(32, 32)
                self.unused = nn.Linear(32, 32)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.used(x)

        torch.manual_seed(35)
        with pytest.raises(QuantizeError, match="no calibration input row flowed"):
            calibrate_stages([("d", Detour())], self.batch(), spec=GRID)


class TestCalibrateStagesDevice:
    """The compute device is **derived from the stage**, and calibration batches stay where the
    caller put them (just-in-time moves).

    Calibration batches can reach hundreds × tens of MB per stage, so a driver that moved them to
    the compute device on its own would compete with the weights for the same VRAM. This pins two
    points, "the batches do not move" and "computation runs on the stage's device"; that the
    all-CPU default path is a no-op is covered by the bit-identity tests above.
    """

    def test_the_batches_reach_the_stage_without_a_copy(self):
        """On the CPU default path `Tensor.to` returns self — not even a copy is made.

        Checked on the tensors the stage **actually received** (looking only at the caller's list
        would say only that the driver does not rewrite its input — a tautology).
        """
        stages = TestCalibrateStages.two_stages()
        batches = TestCalibrateStages.batch()
        given = batches[0][0][0]
        seen: list[torch.Tensor] = []
        stages[0][1].register_forward_pre_hook(lambda _module, args: seen.append(args[0]))

        calibrate_stages(stages, batches, spec=GRID)

        assert seen, "the first stage was never called"
        assert all(tensor is given for tensor in seen)

    def test_a_stage_without_parameters_fails_loudly(self):
        """A stage with nothing to derive a device from is not treated as CPU silently."""
        with pytest.raises(QuantizeError, match="the stage has no parameters"):
            calibrate_stages([("act", nn.ReLU())], TestCalibrateStages.batch(), spec=GRID)

    def test_a_tuple_keyword_reaches_the_stage_without_a_copy(self):
        """Tuple kwargs (rope's `(cos, sin)`) are not copied on the CPU path either."""
        torch.manual_seed(46)
        stages: list[tuple[str, nn.Module]] = [("block.0", Roped(32))]
        rotary = (torch.ones(32), torch.zeros(32))
        seen: list[object] = []
        stages[0][1].register_forward_pre_hook(
            lambda _module, _args, kwargs: seen.append(kwargs["rotary"]), with_kwargs=True
        )

        calibrate_stages(
            stages, [((correlated_inputs(24, 32, seed=47),), {"rotary": rotary})], spec=GRID
        )

        assert seen, "the stage was never called"
        assert all(tuple(given) == rotary for given in seen)
        assert all(given[0] is rotary[0] and given[1] is rotary[1] for given in seen)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="cannot be checked without CUDA")
    def test_a_cuda_stage_runs_from_cpu_batches(self):
        """A GPU stage × CPU batches (tuple kwargs included) runs to completion, and the batches
        stay on the CPU.

        The shared venv's torch is the CPU build, so this is skipped by default (running GPTQ
        calibration on CUDA is a path for sensitivity experiments; the distribution path stays on
        the CPU).
        """
        torch.manual_seed(44)
        stages: list[tuple[str, nn.Module]] = [
            ("block.0", Roped(64).cuda()),
            ("block.1", Roped(64).cuda()),
        ]
        rotary = (torch.ones(64), torch.zeros(64))
        batches = [((correlated_inputs(48, 64, seed=45),), {"rotary": rotary})]

        report = calibrate_stages(stages, batches, spec=GRID)

        assert report.modules == 2
        assert report.int4 is not None
        assert sorted(report.int4.scales) == ["block.0.proj.weight", "block.1.proj.weight"]
        assert all(scale.device.type == "cuda" for scale in report.int4.scales.values())
        assert all(module.proj.weight.device.type == "cuda" for _prefix, module in stages)
        assert batches[0][0][0].device.type == "cpu"
        assert all(part.device.type == "cpu" for part in rotary)


class TestCalibrateStagesMethods:
    @staticmethod
    def linear_stage(seed: int = 36) -> list[tuple[str, nn.Module]]:
        torch.manual_seed(seed)
        return [("only", nn.Linear(32, 8))]

    @staticmethod
    def sign_batch() -> list[tuple[tuple[object, ...], dict[str, object]]]:
        """Calibration input whose `|x|` max is exactly 1 on every channel (AWQ's `s` is 1 for any
        α)."""
        torch.manual_seed(37)
        return [((torch.randn(24, 32).sign(),), {})]

    def test_awq_with_a_flat_activation_profile_matches_round_to_nearest(self):
        """α = 0 means `s = 1`, so the weights written back match `fake_quant_int4` bit for bit."""
        stages = self.linear_stage()
        (_, module) = stages[0]
        twin = copy.deepcopy(module)
        fake_quant_int4(twin, group_size=GROUP)

        report = calibrate_stages(stages, self.sign_batch(), method="awq", spec=GRID)

        assert [layer.alpha for layer in report.layers] == [0.0]
        assert torch.equal(module.weight.detach(), twin.weight.detach())

    def test_awq_returns_no_ledger(self):
        """MUST: `W_eff = Q(W')/s` cannot be stored alone (it needs a fold or a companion)."""
        report = calibrate_stages(self.linear_stage(), self.sign_batch(), method="awq", spec=GRID)

        assert report.int4 is None
        assert "fold" in report.describe()

    def test_awq_and_gptq_together_report_both_alpha_and_no_ledger(self):
        report = calibrate_stages(
            self.linear_stage(),
            [((correlated_inputs(64, 32, seed=38),), {})],
            method="awq+gptq",
            spec=GRID,
        )

        assert report.method == "awq+gptq"
        assert report.int4 is None
        assert report.layers[0].alpha is not None
        assert report.layers[0].error is not None

    def test_a_fitted_table_stride_is_reported(self):
        """MUST: a fit on a subsample is stated in the measuring side's output (same wording as
        `quant_methods`)."""
        report = calibrate_stages(
            self.linear_stage(),
            [((correlated_inputs(64, 32, seed=39),), {})],
            spec=GridSpec(kind="kmeans_shared", group_size=GROUP, fit_stride=3),
        )

        assert report.fit_stride == 3
        assert "fit_stride 3" in report.describe()
        assert report.int4 is None


class TestCalibrateStagesOptions:
    """The sweep axes reach the layers from the driver **unchanged** (sweeps run through
    `calibrate_stages`)."""

    @staticmethod
    def stage_and_inputs() -> tuple[nn.Linear, torch.Tensor]:
        torch.manual_seed(112)
        return nn.Linear(32, 8), correlated_inputs(64, 32, seed=113)

    def test_the_options_reach_the_layer(self):
        stage, inputs = self.stage_and_inputs()
        original = stage.weight.detach().clone()

        calibrate_stages(
            [("only", stage)],
            [((inputs,), {})],
            spec=GRID,
            static_groups=True,
            act_order=True,
            damping=0.05,
        )

        expected, _ = gptq_quantize_rows(
            original, gram(inputs), GRID, static_groups=True, act_order=True, damping=0.05
        )
        assert torch.equal(stage.weight.detach(), expected)
        # Counterpart — differs from the default run (confirms the axis is passed through).
        default, _ = gptq_quantize_rows(original, gram(inputs), GRID)
        assert not torch.equal(expected, default)

    def test_the_defaults_leave_the_driver_bit_identical(self):
        stage, inputs = self.stage_and_inputs()
        original = stage.weight.detach().clone()

        calibrate_stages([("only", stage)], [((inputs,), {})], spec=GRID)

        expected, _ = gptq_quantize_rows(original, gram(inputs), GRID)
        assert torch.equal(stage.weight.detach(), expected)

    def test_gptq_options_on_a_method_without_a_hessian_fail_loudly(self):
        """The same convention as `GridSpec.fit_stride` — an axis that has no effect is not silently
        ignored."""
        stage, inputs = self.stage_and_inputs()

        with pytest.raises(QuantizeError, match="have no effect"):
            calibrate_stages(
                [("only", stage)],
                [((inputs,), {})],
                method="awq",
                spec=GRID,
                static_groups=True,
            )


class TestStageHandoff:
    def test_a_tuple_output_stage_hands_its_first_tensor_to_the_next(self):
        torch.manual_seed(40)
        stages: list[tuple[str, nn.Module]] = [("a", Chain(32, 32)), ("b", Chain(32, 16))]
        (_, first), (_, second) = stages
        original_second = second.proj.weight.detach().clone()
        inputs = correlated_inputs(24, 32, seed=41)

        calibrate_stages(stages, [((inputs,), {})], spec=GRID)

        propagated, _ = first(inputs)
        expected, _ = gptq_quantize_rows(original_second, gram(propagated), GRID)
        assert torch.equal(second.proj.weight.detach(), expected)

    @staticmethod
    def run_with(advance) -> list[torch.Tensor]:
        """Run the same model on the same calibration input, swapping only `advance_kwargs`."""
        torch.manual_seed(42)
        stages: list[tuple[str, nn.Module]] = [("a", Chain(32, 32)), ("b", Chain(32, 16))]
        inputs = correlated_inputs(24, 32, seed=43)
        calibrate_stages(stages, [((inputs,), {"shift": 3.0})], spec=GRID, advance_kwargs=advance)
        return [module.proj.weight.detach().clone() for _, module in stages]

    def test_the_keyword_arguments_carry_over_unchanged_by_default(self):
        """The default is "kwargs unchanged between stages" — **the same result** as an explicit
        identity override."""
        assert all(
            torch.equal(default, explicit)
            for default, explicit in zip(
                self.run_with(None), self.run_with(lambda _index, kwargs: kwargs), strict=True
            )
        )

    def test_advance_kwargs_overrides_the_next_stage_arguments(self):
        """Counterpart — the override really takes effect (changing the later `shift` changes `H`).

        The earlier stage runs with the `current` kwargs, so it **not changing** is correct (the
        override is for the next stage).
        """
        default = self.run_with(None)
        overridden = self.run_with(lambda _index, _kwargs: {"shift": 0.0})

        assert torch.equal(default[0], overridden[0])
        assert not torch.equal(default[1], overridden[1])

    def test_an_output_that_is_not_a_tensor_fails_loudly(self):
        with pytest.raises(QuantizeError, match="cannot choose"):
            first_tensor_output({"hidden": torch.zeros(2)})

    def test_a_tensor_output_passes_through(self):
        tensor = torch.zeros(2)

        assert first_tensor_output(tensor) is tensor
        assert first_tensor_output((tensor, "state")) is tensor
