"""測定専用の丸め方式（FP4 / NF4 / MXFP4 / k-means codebook）— 格納経路は持たない。

どの方式も見るのは 3 点: ①丸めた重みが**表 × scale の表現可能集合**に載っていること
②同一入力 → **ビット同一出力**（k-means の初期化まで決定的・seed に依存しない）
③対象選択が `quantize` と**同じ 1 本**（include / op_types / 対象 0 本 / 整除違反）。
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

#: 方式テストの group 長（フィクスチャの in 軸 16 / 32 を割り切る短さ — 格納側の受理集合
#: 「2 冪かつ ≥ 16」は `verify.py` の規則で、丸めそのものに下限は無い）。
GROUP = 8


class Layers(nn.Module):
    """5 op 種 + 対象外（norm / 生 Parameter）— **平坦化後の in 軸が全部 8 の倍数**。"""

    def __init__(self) -> None:
        super().__init__()
        self.dense = nn.Linear(16, 3)
        self.wide = nn.Linear(32, 2)
        self.conv = nn.Conv1d(2, 3, kernel_size=8)
        self.image = nn.Conv2d(2, 2, kernel_size=(2, 4))
        self.up = nn.ConvTranspose1d(2, 3, kernel_size=8)
        self.table = nn.Embedding(5, 16)
        self.norm = nn.LayerNorm(3)
        self.gain = nn.Parameter(torch.randn(3))


def layers(seed: int = 0) -> Layers:
    torch.manual_seed(seed)
    return Layers()


def unit_group_layers(seed: int = 0) -> nn.Module:
    """全 group の absmax が**厳密に 1.0**の模型（`shared` の表を割り算なしで見るための形）。

    absmax 正規化の scale がちょうど 1.0 になるので、丸め済みの重みの値集合が**そのまま
    共有表**になる（`w/s` の割り戻しを挟むと 1ulp の誤差が入って厳密比較ができない）。
    """
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(16, 3), nn.Linear(32, 2))
    with torch.no_grad():
        for module in model:
            grouped = grouped_view(module.weight, GROUP, "検査")
            module.weight.copy_(
                (grouped / grouped.abs().amax(dim=-1, keepdim=True)).reshape(module.weight.shape)
            )
    return model


def targets(model: nn.Module) -> list[tuple[str, torch.Tensor, int]]:
    """検査側でも対象を `quantize` の列挙から引く（対象の綴りをテストへ写さない）。"""
    return list(iter_quant_targets(model, QUANT_MODULE_TYPES, None))


def representable(weight: torch.Tensor, levels: torch.Tensor, scale: torch.Tensor) -> bool:
    """`weight` の全要素が group ごとの表現可能集合（`levels × scale`）に載っているか。"""
    grouped = grouped_view(weight, group_size_of(weight, scale), "検査")
    representable_values = scale.unsqueeze(-1).unsqueeze(-1) * levels
    return bool((grouped.unsqueeze(-1) == representable_values).any(dim=-1).all())


#: 固定表 × group scale の 3 方式（表と scale の作り方だけが違う）。
FIXED_TABLE_METHODS = (
    pytest.param(fake_quant_fp4, FP4_E2M1_LEVELS, False, id="fp4"),
    pytest.param(fake_quant_nf4, NF4_LEVELS, False, id="nf4"),
    pytest.param(fake_quant_mxfp4, FP4_E2M1_LEVELS, True, id="mxfp4"),
)


class TestLevelTables:
    def test_fp4_is_the_e2m1_value_set(self):
        """符号 1 / 指数 2 / 仮数 1 bit の値集合。±0 が同値なので**相異なる値は 15**。"""
        table = levels_tensor(FP4_E2M1_LEVELS)

        assert table.numel() == 15
        assert float(table.abs().max()) == 6.0
        assert torch.equal(table, -table.flip(0))  # 0 対称

    def test_nf4_is_the_qlora_table(self):
        """QLoRA の 16 分位値 — 0.0 と ±1.0 を厳密に含み、正負で非対称。"""
        table = levels_tensor(NF4_LEVELS)

        assert table.numel() == 16
        assert float(table[0]) == -1.0
        assert float(table[-1]) == 1.0
        assert 0.0 in set(NF4_LEVELS)
        assert not torch.equal(table, -table.flip(0))

    def test_a_table_that_is_not_ascending_fails_loudly(self):
        """昇順でない表は二分探索が黙って別の準位を返す（例外なしで最近傍でなくなる）。"""
        with pytest.raises(QuantizeError, match="昇順"):
            levels_tensor((0.0, 1.0, 0.5))


class TestRoundToLevels:
    def test_values_go_to_the_nearest_level(self):
        table = levels_tensor((-1.0, 0.0, 2.0))

        rounded = round_to_levels(torch.tensor([-0.6, -0.4, 0.9, 1.1, 5.0]), table)

        assert torch.equal(rounded, torch.tensor([-1.0, 0.0, 0.0, 2.0, 2.0]))

    def test_a_value_on_the_midpoint_goes_down(self):
        """同点の向きを 1 つに決めておく（決定性のため — 偶数丸めではない）。"""
        table = levels_tensor((0.0, 1.0))

        assert float(round_to_levels(torch.tensor([0.5]), table)) == 0.0

    def test_values_outside_the_table_saturate(self):
        """表の外は両端へ飽和する（MXFP4 の block 飽和もこの経路）。"""
        table = levels_tensor((-1.0, 0.0, 1.0))

        rounded = round_to_levels(torch.tensor([-9.0, 9.0]), table)

        assert torch.equal(rounded, torch.tensor([-1.0, 1.0]))


class TestFixedTableMethods:
    @pytest.mark.parametrize(("method", "levels", "power_of_two"), FIXED_TABLE_METHODS)
    def test_every_weight_lands_on_the_representable_set(self, method, levels, power_of_two):
        """丸め後の全要素が「表 × その group の scale」に載る。

        scale は**丸める前の重み**から引き直す（丸め済みから引くと、実装が使った scale と
        同じ値になる保証を前提にしてしまい、検査が自分自身を確かめるだけになる）。
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
        """同一入力 → ビット同一（乱数状態に依存しない）。"""
        model = layers()
        twin = copy.deepcopy(model)

        method(model, group_size=GROUP)
        torch.manual_seed(999)
        method(twin, group_size=GROUP)

        assert torch.equal(model.dense.weight, twin.dense.weight)
        assert torch.equal(model.wide.weight, twin.wide.weight)

    @pytest.mark.parametrize(("method", "levels", "power_of_two"), FIXED_TABLE_METHODS)
    def test_reapplying_the_round_is_bit_identical(self, method, levels, power_of_two):
        """冪等（group の amax 要素が表の端へ乗るので scale が不動点になる — 実測）。"""
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
        """「方式を指定したのに 0 本」を沈黙させない（`quantize` 側と同じ常設診断）。"""
        with pytest.raises(QuantizeError, match="1 本も無い"):
            method(nn.LayerNorm(3), group_size=GROUP)
        with pytest.raises(QuantizeError, match="1 本も無い"):
            method(layers(), group_size=GROUP, include=lambda fqn: False)

    @pytest.mark.parametrize(("method", "levels", "power_of_two"), FIXED_TABLE_METHODS)
    def test_an_axis_the_group_size_does_not_divide_fails_loudly(
        self, method, levels, power_of_two
    ):
        """端数 group を作らない（`quantize.grouped_view` の共有 — ADR 0069 決定 2 と同じ流儀）。"""
        with pytest.raises(QuantizeError, match="割り切れない"):
            method(nn.Linear(20, 3), group_size=GROUP)

    @pytest.mark.parametrize(("method", "levels", "power_of_two"), FIXED_TABLE_METHODS)
    def test_the_target_set_is_shared_with_the_storage_path(self, method, levels, power_of_two):
        """対象選択は `quantize.iter_quant_targets` の共有 — 既定 linear のみ・5 種まで広がる。"""
        assert method(layers(), group_size=GROUP).modules == 2
        assert method(layers(), group_size=GROUP, op_types=QUANT_MODULE_TYPES).modules == 6


class TestMxfp4Scale:
    def test_the_scale_is_a_power_of_two(self):
        """OCP MX の共有 scale は 2 のべき（E8M0 に載る形）。"""
        rows = channel_rows(layers().dense.weight, 0)

        scale = group_power_of_two_scale(rows, GROUP)

        mantissa, _ = torch.frexp(scale)
        assert torch.equal(mantissa, torch.full_like(mantissa, 0.5))

    def test_the_group_amax_is_not_restored_exactly(self):
        """absmax 版との差 — amax は 2 のべきへ切り下げられるので厳密復元されず、**超えうる**。

        代わりに scale の乗除算が厳密になる（{@link TestFixedTableMethods} の冪等が真に成立）。
        """
        model = nn.Linear(16, 1)
        with torch.no_grad():
            model.weight.copy_(torch.full((1, 16), 0.1))
            model.weight[0, 0] = 5.5  # 2² と 2³ の間 = 切り下げで scale 1.0・格子の外へ出る値

        fake_quant_mxfp4(model, group_size=16)

        assert float(model.weight[0, 0].detach()) == 6.0  # 5.5 → 最近傍は 6.0（amax 超え）


class TestKMeansCodebook:
    def test_a_per_tensor_table_has_at_most_sixteen_values(self):
        model = layers()

        report = fake_quant_kmeans(model, "per_tensor", group_size=GROUP)

        assert report.method == "kmeans:per_tensor"
        assert int(torch.unique(model.dense.weight).numel()) <= 16
        assert int(torch.unique(model.wide.weight).numel()) <= 16

    def test_a_per_channel_table_is_fitted_row_by_row(self):
        """行ごとに別の表（= 行の値集合が 16 以下・行をまたぐと 16 を超える）。"""
        model = layers()

        fake_quant_kmeans(model, "per_channel", group_size=GROUP, op_types=QUANT_MODULE_TYPES)

        rows = channel_rows(model.dense.weight, 0)
        for index, row in enumerate(rows):
            assert int(torch.unique(row).numel()) <= 16, index
        assert int(torch.unique(rows).numel()) > 16

    def test_a_shared_table_spans_every_layer(self):
        """層をまたいで 1 枚（正規化 scale が厳密に 1.0 の模型なので値集合がそのまま表）。"""
        model = unit_group_layers()

        fake_quant_kmeans(model, "shared", group_size=GROUP)

        values = torch.cat([module.weight.reshape(-1) for module in model])
        assert int(torch.unique(values).numel()) <= 16

    def test_a_per_tensor_table_is_not_shared_between_layers(self):
        """`shared` の対（同じ模型で層ごとに表を張ると値集合が 16 を超える）。"""
        model = unit_group_layers()

        fake_quant_kmeans(model, "per_tensor", group_size=GROUP)

        values = torch.cat([module.weight.reshape(-1) for module in model])
        assert int(torch.unique(values).numel()) > 16

    @pytest.mark.parametrize("granularity", KMEANS_GRANULARITIES)
    def test_the_same_input_gives_a_bit_identical_result(self, granularity):
        """MUST: 初期化まで決定的（分位点 + 固定反復）— seed を振っても結果が動かない。"""
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
        with pytest.raises(QuantizeError, match="1 本も無い"):
            fake_quant_kmeans(nn.LayerNorm(3), granularity, group_size=GROUP)
        with pytest.raises(QuantizeError, match="1 本も無い"):
            fake_quant_kmeans(layers(), granularity, group_size=GROUP, include=lambda fqn: False)

    @pytest.mark.parametrize("granularity", KMEANS_GRANULARITIES)
    def test_the_target_set_is_shared_with_the_storage_path(self, granularity):
        assert fake_quant_kmeans(layers(), granularity, group_size=GROUP).modules == 2
        assert (
            fake_quant_kmeans(
                layers(), granularity, group_size=GROUP, op_types=QUANT_MODULE_TYPES
            ).modules
            == 6
        )

    def test_an_axis_the_group_size_does_not_divide_fails_loudly_for_the_shared_table(self):
        """group が効くのは `shared` だけ（表の粒度が group 正規化に乗る形）。"""
        with pytest.raises(QuantizeError, match="割り切れない"):
            fake_quant_kmeans(nn.Linear(20, 3), "shared", group_size=GROUP)

    def test_an_unknown_granularity_fails_loudly(self):
        with pytest.raises(QuantizeError, match="未対応"):
            fake_quant_kmeans(layers(), "per_group", group_size=GROUP)  # type: ignore[arg-type]


class TestFitCodebook:
    def test_the_centroids_are_ascending(self):
        """表は昇順 MUST（最近傍が中点の二分探索）。"""
        codebook = fit_codebook(torch.randn(2, 512), levels=16, iterations=8)

        assert codebook.shape == (2, 16)
        assert bool(torch.all(codebook[:, 1:] > codebook[:, :-1]))

    def test_clearly_separated_clusters_are_found(self):
        """恒真化の対 — 分位点初期化 + Lloyd が実際にクラスタへ寄る（表が入力に応じて動く）。"""
        values = torch.cat([torch.full((1, 64), -3.0), torch.full((1, 64), 5.0)], dim=1)

        codebook = fit_codebook(values, levels=2, iterations=8)

        assert torch.equal(codebook, torch.tensor([[-3.0, 5.0]]))

    def test_fewer_elements_than_levels_fails_loudly(self):
        with pytest.raises(QuantizeError, match="要素が"):
            fit_codebook(torch.randn(1, 8), levels=16, iterations=8)
