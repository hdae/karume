"""Measurement-only rounding methods (FP4 / NF4 / MXFP4 / k-means codebook) — they have no
storage path.

Every method is checked on 3 points: ① the rounded weights lie on **the representable set of
table × scale** ② same input → **bit-identical output** (deterministic down to the k-means
initialization, independent of the seed) ③ target selection is **the same single path** as
`quantize` (include / op_types / zero targets / divisibility violation).
"""

from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from karume.quant_methods import (
    FP4_E2M1_LEVELS,
    KMEANS_GRANULARITIES,
    NF4_LEVELS,
    fake_quant_fp4,
    fake_quant_kmeans,
    fake_quant_mxfp4,
    fake_quant_nf4,
    fit_codebook,
    group_absmax_scale,
    group_power_of_two_scale,
    levels_tensor,
    round_to_levels,
)
from karume.quantize import (
    QUANT_MODULE_TYPES,
    QuantizeError,
    channel_rows,
    group_size_of,
    grouped_view,
    iter_quant_targets,
)

#: The group length of the method tests (short enough to divide the fixtures' in axes 16 / 32 —
#: the storage side's accepted set "a power of 2 and ≥ 16" is a rule of `verify.py`; the rounding
#: itself has no lower bound).
GROUP = 8


class Layers(nn.Module):
    """Every kind in `QUANT_MODULE_TYPES` + non-targets (norm / bare Parameter) — **every in axis
    is a multiple of 8 after flattening**."""

    def __init__(self) -> None:
        super().__init__()
        self.dense = nn.Linear(16, 3)
        self.wide = nn.Linear(32, 2)
        self.conv = nn.Conv1d(2, 3, kernel_size=8)
        self.image = nn.Conv2d(2, 2, kernel_size=(2, 4))
        self.volume = nn.Conv3d(2, 2, kernel_size=(2, 1, 4))
        self.up = nn.ConvTranspose1d(2, 3, kernel_size=8)
        self.table = nn.Embedding(5, 16)
        self.norm = nn.LayerNorm(3)
        self.gain = nn.Parameter(torch.randn(3))


def layers(seed: int = 0) -> Layers:
    torch.manual_seed(seed)
    return Layers()


def unit_group_layers(seed: int = 0) -> nn.Module:
    """A model whose every group has an absmax of **exactly 1.0** (a shape for looking at the
    `shared` table without a division).

    The scale of the absmax normalization is exactly 1.0, so the value set of the rounded weights
    **is the shared table as is** (dividing back by `w/s` would add a 1 ulp error and rule out an
    exact comparison).
    """
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(16, 3), nn.Linear(32, 2))
    with torch.no_grad():
        for module in model:
            grouped = grouped_view(module.weight, GROUP, "test")
            module.weight.copy_(
                (grouped / grouped.abs().amax(dim=-1, keepdim=True)).reshape(module.weight.shape)
            )
    return model


def targets(model: nn.Module) -> list[tuple[str, torch.Tensor, int]]:
    """The test side also takes its targets from the `quantize` enumeration (the target spelling is
    not copied into the tests)."""
    return list(iter_quant_targets(model, QUANT_MODULE_TYPES, None))


def representable(weight: torch.Tensor, levels: torch.Tensor, scale: torch.Tensor) -> bool:
    """Whether every element of `weight` lies on the per-group representable set
    (`levels × scale`)."""
    grouped = grouped_view(weight, group_size_of(weight, scale), "test")
    representable_values = scale.unsqueeze(-1).unsqueeze(-1) * levels
    return bool((grouped.unsqueeze(-1) == representable_values).any(dim=-1).all())


#: The 3 methods of fixed table × group scale (they differ only in how the table and the scale
#: are made).
FIXED_TABLE_METHODS = (
    pytest.param(fake_quant_fp4, FP4_E2M1_LEVELS, False, id="fp4"),
    pytest.param(fake_quant_nf4, NF4_LEVELS, False, id="nf4"),
    pytest.param(fake_quant_mxfp4, FP4_E2M1_LEVELS, True, id="mxfp4"),
)


class TestLevelTables:
    def test_fp4_is_the_e2m1_value_set(self):
        """The value set of 1 sign / 2 exponent / 1 mantissa bit. ±0 are equal, so **there are 15
        distinct values**."""
        table = levels_tensor(FP4_E2M1_LEVELS)

        assert table.numel() == 15
        assert float(table.abs().max()) == 6.0
        assert torch.equal(table, -table.flip(0))  # symmetric about 0

    def test_nf4_is_the_qlora_table(self):
        """The 16 quantiles of QLoRA — contains 0.0 and ±1.0 exactly, asymmetric between positive
        and negative."""
        table = levels_tensor(NF4_LEVELS)

        assert table.numel() == 16
        assert float(table[0]) == -1.0
        assert float(table[-1]) == 1.0
        assert 0.0 in set(NF4_LEVELS)
        assert not torch.equal(table, -table.flip(0))

    def test_a_table_that_is_not_ascending_fails_loudly(self):
        """With a table that is not ascending, the binary search silently returns another level (no
        longer the nearest, without an exception)."""
        with pytest.raises(QuantizeError, match="is not ascending"):
            levels_tensor((0.0, 1.0, 0.5))


class TestRoundToLevels:
    def test_values_go_to_the_nearest_level(self):
        table = levels_tensor((-1.0, 0.0, 2.0))

        rounded = round_to_levels(torch.tensor([-0.6, -0.4, 0.9, 1.1, 5.0]), table)

        assert torch.equal(rounded, torch.tensor([-1.0, 0.0, 0.0, 2.0, 2.0]))

    def test_a_value_on_the_midpoint_goes_down(self):
        """Ties go one fixed way (for determinism — not round-half-to-even)."""
        table = levels_tensor((0.0, 1.0))

        assert float(round_to_levels(torch.tensor([0.5]), table)) == 0.0

    def test_values_outside_the_table_saturate(self):
        """Values outside the table saturate to the ends (MXFP4's block saturation takes this path
        too)."""
        table = levels_tensor((-1.0, 0.0, 1.0))

        rounded = round_to_levels(torch.tensor([-9.0, 9.0]), table)

        assert torch.equal(rounded, torch.tensor([-1.0, 1.0]))


class TestFixedTableMethods:
    @pytest.mark.parametrize(("method", "levels", "power_of_two"), FIXED_TABLE_METHODS)
    def test_every_weight_lands_on_the_representable_set(self, method, levels, power_of_two):
        """Every element after rounding lies on "table × the scale of its group".

        The scale is derived again from **the weights before rounding** (deriving it from the
        rounded weights would presuppose that it equals the scale the implementation used, and
        the check would only confirm itself).
        """
        model = layers()
        table = levels_tensor(levels)
        before = {name: weight.detach().clone() for name, weight in model.named_parameters()}

        method(model, group_size=GROUP, op_types=QUANT_MODULE_TYPES)

        for fqn, _, axis in targets(model):
            rows = channel_rows(before[fqn], axis)
            scale = (
                group_power_of_two_scale(rows, GROUP)
                if power_of_two
                else group_absmax_scale(rows, GROUP, float(table.abs().max()))
            )
            assert representable(channel_rows(model.get_parameter(fqn), axis), table, scale), fqn

    @pytest.mark.parametrize(("method", "levels", "power_of_two"), FIXED_TABLE_METHODS)
    def test_the_same_input_gives_a_bit_identical_result(self, method, levels, power_of_two):
        """Same input → bit-identical (independent of the RNG state)."""
        model = layers()
        twin = copy.deepcopy(model)

        method(model, group_size=GROUP)
        torch.manual_seed(999)
        method(twin, group_size=GROUP)

        assert torch.equal(model.dense.weight, twin.dense.weight)
        assert torch.equal(model.wide.weight, twin.wide.weight)

    @pytest.mark.parametrize(("method", "levels", "power_of_two"), FIXED_TABLE_METHODS)
    def test_reapplying_the_round_is_bit_identical(self, method, levels, power_of_two):
        """Idempotent (the group's amax element lands on the table's end, so the scale is a fixed
        point — observed)."""
        model = layers()
        method(model, group_size=GROUP, op_types=QUANT_MODULE_TYPES)
        snapshot = {name: weight.detach().clone() for name, weight in model.named_parameters()}

        method(model, group_size=GROUP, op_types=QUANT_MODULE_TYPES)

        for name, before in snapshot.items():
            assert torch.equal(model.get_parameter(name), before), name

    @pytest.mark.parametrize(("method", "levels", "power_of_two"), FIXED_TABLE_METHODS)
    def test_only_included_modules_are_touched(self, method, levels, power_of_two):
        model = layers()
        untouched = model.wide.weight.detach().clone()

        report = method(model, group_size=GROUP, include=lambda fqn: fqn == "dense")

        assert report.modules == 1
        assert torch.equal(model.wide.weight, untouched)

    @pytest.mark.parametrize(("method", "levels", "power_of_two"), FIXED_TABLE_METHODS)
    def test_no_target_fails_loudly(self, method, levels, power_of_two):
        """Zero targets with a method specified is not silenced (the same permanent diagnostic as on
        the `quantize` side)."""
        with pytest.raises(QuantizeError, match="there is no weight"):
            method(nn.LayerNorm(3), group_size=GROUP)
        with pytest.raises(QuantizeError, match="there is no weight"):
            method(layers(), group_size=GROUP, include=lambda fqn: False)

    @pytest.mark.parametrize(("method", "levels", "power_of_two"), FIXED_TABLE_METHODS)
    def test_an_axis_the_group_size_does_not_divide_fails_loudly(
        self, method, levels, power_of_two
    ):
        """No remainder group (shared with `quantize.grouped_view` — the same rule as ADR 0069
        decision 2)."""
        with pytest.raises(QuantizeError, match="is not divisible"):
            method(nn.Linear(20, 3), group_size=GROUP)

    @pytest.mark.parametrize(("method", "levels", "power_of_two"), FIXED_TABLE_METHODS)
    def test_the_target_set_is_shared_with_the_storage_path(self, method, levels, power_of_two):
        """Target selection is shared with `quantize.iter_quant_targets` — linear only by default,
        widening to every kind."""
        assert method(layers(), group_size=GROUP).modules == 2
        assert method(layers(), group_size=GROUP, op_types=QUANT_MODULE_TYPES).modules == 7


class TestMxfp4Scale:
    def test_the_scale_is_a_power_of_two(self):
        """The shared scale of OCP MX is a power of 2 (a shape that fits E8M0)."""
        rows = channel_rows(layers().dense.weight, 0)

        scale = group_power_of_two_scale(rows, GROUP)

        mantissa, _ = torch.frexp(scale)
        assert torch.equal(mantissa, torch.full_like(mantissa, 0.5))

    def test_the_group_amax_is_not_restored_exactly(self):
        """The difference from the absmax version — amax is rounded down to a power of 2, so it is
        not restored exactly and **can be exceeded**.

        In exchange, multiplying and dividing by the scale are exact (the idempotence of
        {@link TestFixedTableMethods} truly holds).
        """
        model = nn.Linear(16, 1)
        with torch.no_grad():
            model.weight.copy_(torch.full((1, 16), 0.1))
            # Between 2² and 2³ = scale 1.0 after rounding down; a value off the grid.
            model.weight[0, 0] = 5.5

        fake_quant_mxfp4(model, group_size=16)

        assert float(model.weight[0, 0].detach()) == 6.0  # 5.5 → nearest is 6.0 (above amax)


class TestKMeansCodebook:
    def test_a_per_tensor_table_has_at_most_sixteen_values(self):
        model = layers()

        report = fake_quant_kmeans(model, "per_tensor", group_size=GROUP)

        assert report.method == "kmeans:per_tensor"
        assert int(torch.unique(model.dense.weight).numel()) <= 16
        assert int(torch.unique(model.wide.weight).numel()) <= 16

    def test_a_per_channel_table_is_fitted_row_by_row(self):
        """A separate table per row (= each row has at most 16 values; across rows there are more
        than 16)."""
        model = layers()

        fake_quant_kmeans(model, "per_channel", group_size=GROUP, op_types=QUANT_MODULE_TYPES)

        rows = channel_rows(model.dense.weight, 0)
        for index, row in enumerate(rows):
            assert int(torch.unique(row).numel()) <= 16, index
        assert int(torch.unique(rows).numel()) > 16

    def test_a_shared_table_spans_every_layer(self):
        """One table across layers (the model's normalization scale is exactly 1.0, so the value set
        is the table as is)."""
        model = unit_group_layers()

        fake_quant_kmeans(model, "shared", group_size=GROUP)

        values = torch.cat([module.weight.reshape(-1) for module in model])
        assert int(torch.unique(values).numel()) <= 16

    def test_a_per_tensor_table_is_not_shared_between_layers(self):
        """The counterpart of `shared` (with a table per layer on the same model, the value set
        exceeds 16)."""
        model = unit_group_layers()

        fake_quant_kmeans(model, "per_tensor", group_size=GROUP)

        values = torch.cat([module.weight.reshape(-1) for module in model])
        assert int(torch.unique(values).numel()) > 16

    @pytest.mark.parametrize("granularity", KMEANS_GRANULARITIES)
    def test_the_same_input_gives_a_bit_identical_result(self, granularity):
        """MUST: deterministic down to the initialization (quantiles + fixed iterations) — changing
        the seed does not move the result."""
        model = layers()
        twin = copy.deepcopy(model)

        fake_quant_kmeans(model, granularity, group_size=GROUP)
        torch.manual_seed(999)
        fake_quant_kmeans(twin, granularity, group_size=GROUP)

        assert torch.equal(model.dense.weight, twin.dense.weight)
        assert torch.equal(model.wide.weight, twin.wide.weight)

    @pytest.mark.parametrize("granularity", KMEANS_GRANULARITIES)
    def test_only_included_modules_are_touched(self, granularity):
        model = layers()
        untouched = model.wide.weight.detach().clone()

        report = fake_quant_kmeans(
            model, granularity, group_size=GROUP, include=lambda fqn: fqn == "dense"
        )

        assert report.modules == 1
        assert torch.equal(model.wide.weight, untouched)

    @pytest.mark.parametrize("granularity", KMEANS_GRANULARITIES)
    def test_no_target_fails_loudly(self, granularity):
        with pytest.raises(QuantizeError, match="there is no weight"):
            fake_quant_kmeans(nn.LayerNorm(3), granularity, group_size=GROUP)
        with pytest.raises(QuantizeError, match="there is no weight"):
            fake_quant_kmeans(layers(), granularity, group_size=GROUP, include=lambda fqn: False)

    @pytest.mark.parametrize("granularity", KMEANS_GRANULARITIES)
    def test_the_target_set_is_shared_with_the_storage_path(self, granularity):
        assert fake_quant_kmeans(layers(), granularity, group_size=GROUP).modules == 2
        assert (
            fake_quant_kmeans(
                layers(), granularity, group_size=GROUP, op_types=QUANT_MODULE_TYPES
            ).modules
            == 7
        )

    def test_an_axis_the_group_size_does_not_divide_fails_loudly_for_the_shared_table(self):
        """The group applies only to `shared` (the table's granularity sits on the group
        normalization)."""
        with pytest.raises(QuantizeError, match="is not divisible"):
            fake_quant_kmeans(nn.Linear(20, 3), "shared", group_size=GROUP)

    def test_a_strided_fit_is_deterministic(self):
        """`fit_stride` is evenly spaced sampling (no randomness) — changing the seed does not move
        the result."""
        model = layers()
        twin = copy.deepcopy(model)

        fake_quant_kmeans(model, "shared", group_size=GROUP, fit_stride=3)
        torch.manual_seed(999)
        fake_quant_kmeans(twin, "shared", group_size=GROUP, fit_stride=3)

        assert torch.equal(model.dense.weight, twin.dense.weight)
        assert torch.equal(model.wide.weight, twin.wide.weight)

    def test_the_strided_fit_sees_only_the_sampled_values(self):
        """Control (the sampling really takes effect) — with weights of +1 at even indices / −1 at
        odd ones, the stride=2 table sees only +1, so after applying it every element falls to +1
        (the full fit restores ±1 exactly).
        """

        def alternating() -> nn.Linear:
            module = nn.Linear(32, 1, bias=False)
            with torch.no_grad():
                values = torch.ones(32)
                values[1::2] = -1.0
                module.weight.copy_(values.reshape(1, 32))
            return module

        full = alternating()
        sampled = alternating()

        fake_quant_kmeans(full, "shared", group_size=GROUP)
        fake_quant_kmeans(sampled, "shared", group_size=GROUP, fit_stride=2)

        assert torch.equal(full.weight, alternating().weight)
        assert torch.equal(sampled.weight, torch.ones(1, 32))

    def test_a_stride_on_a_granularity_without_a_concat_fit_fails_loudly(self):
        """`fit_stride` is for shared only — it is never silently ignored on another granularity."""
        with pytest.raises(QuantizeError, match="is for shared only"):
            fake_quant_kmeans(layers(), "per_tensor", group_size=GROUP, fit_stride=2)

    def test_a_non_positive_stride_fails_loudly(self):
        with pytest.raises(QuantizeError, match="1 or more"):
            fake_quant_kmeans(layers(), "shared", group_size=GROUP, fit_stride=0)

    def test_an_unknown_granularity_fails_loudly(self):
        with pytest.raises(QuantizeError, match="is not supported"):
            fake_quant_kmeans(layers(), "per_group", group_size=GROUP)  # type: ignore[arg-type]


class TestFitCodebook:
    def test_the_centroids_are_ascending(self):
        """The table MUST be ascending (the nearest is found by a binary search over the
        midpoints)."""
        codebook = fit_codebook(torch.randn(2, 512), levels=16, iterations=8)

        assert codebook.shape == (2, 16)
        assert bool(torch.all(codebook[:, 1:] > codebook[:, :-1]))

    def test_clearly_separated_clusters_are_found(self):
        """The counterpart against a tautological test — quantile initialization + Lloyd really
        moves to the clusters (the table moves with the input)."""
        values = torch.cat([torch.full((1, 64), -3.0), torch.full((1, 64), 5.0)], dim=1)

        codebook = fit_codebook(values, levels=2, iterations=8)

        assert torch.equal(codebook, torch.tensor([[-3.0, 5.0]]))

    def test_fewer_elements_than_levels_fails_loudly(self):
        with pytest.raises(QuantizeError, match="elements"):
            fit_codebook(torch.randn(1, 8), levels=16, iterations=8)
