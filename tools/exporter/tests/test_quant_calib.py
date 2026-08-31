"""校正付き丸め（GPTQ / AWQ）— 格納グリッドは 1 段目のまま、丸め先だけを選び直す方式。

見るのは 5 点: ①同一入力 → **ビット同一**（乱数不使用）②丸め済みの値が**格納グリッド**に
載る（`q·scale`・`|q| ≤ 7`）③ `H` が対角なら `quantize.fake_quant_int4` と**ビット同一**
（補償項が消える数学的帰結 = 恒真化していないオラクル）④相関入力では GPTQ の `H` 加重誤差が
RTN より**小さい**（方式が実際に効く対照）⑤ stage 逐次の**誤差伝播**が後段へ届いている
（伝播を止めると結果が変わる故障注入）⑥掃引軸（`static_groups` / `act_order` / `damping`）は
**既定で既存経路とビット同一**、かつ有効時は格納の形（元列順・連続 group・group scale）を
1 つも動かさない。
"""

from __future__ import annotations

import copy

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

#: 校正テストの group 長（フィクスチャの in 軸 32 / 128 を割り切る短さ）。
GROUP = 8

#: `group_size=GROUP` の既定グリッド（出荷経路 = RTN）。
GRID = GridSpec(group_size=GROUP)


def weights(rows: int, columns: int, seed: int) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(rows, columns) * 0.05


def correlated_inputs(tokens: int, columns: int, seed: int) -> torch.Tensor:
    """入力チャネル間に相関を入れた校正入力（`H` が対角から遠い形 — GPTQ が効く条件）。"""
    torch.manual_seed(seed)
    mixing = torch.randn(columns, columns) * 0.3 + torch.eye(columns)
    return torch.randn(tokens, columns) @ mixing


def gram(inputs: torch.Tensor) -> torch.Tensor:
    """`H = Σ XᵀX`（F64）。"""
    wide = inputs.detach().to(torch.float64)
    return wide.T @ wide


def weighted_error(delta: torch.Tensor, hessian: torch.Tensor) -> float:
    """`tr(ΔW·H·ΔWᵀ)` — GPTQ が最小化しようとしている量そのもの。"""
    wide = delta.to(torch.float64)
    return float((wide @ hessian * wide).sum())


def round_to_nearest(rows: torch.Tensor, group_size: int) -> torch.Tensor:
    """比較対象の RTN（`quantize` の格子・校正データを見ない丸め）。"""
    scale = group_scale(rows, group_size)
    return dequantize_int4(quantize_to_int4(rows, scale), scale)


class Chain(nn.Module):
    """`(hidden, state)` の tuple を返す stage（transformer ブロックの返り値の形）。

    kwargs の効きは `shift`（proj の入力への**加算**）で観測する。乗算にしないのは GPTQ が
    `H` の定数倍に**不変**だから（`H → cH` で damping も `U` も同じ比で動き、補償項が
    相殺する）— 倍率では「kwargs が届いたか」を結果から判別できない。
    """

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.proj = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor, shift: float = 0.0) -> tuple[torch.Tensor, str]:
        return self.proj(x + shift), "state"


class Roped(nn.Module):
    """rope の `(cos, sin)` のような**組**を keyword で受ける stage（実 DiT block と同じ形）。"""

    def __init__(self, features: int) -> None:
        super().__init__()
        self.proj = nn.Linear(features, features)

    def forward(self, x: torch.Tensor, rotary: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        cos, sin = rotary
        return self.proj(x * cos + sin)


class Pair(nn.Module):
    """1 つの stage に 2 本の linear（`include` の効きを見るための形）。"""

    def __init__(self, features: int) -> None:
        super().__init__()
        self.left = nn.Linear(features, features)
        self.right = nn.Linear(features, features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.left(x) + self.right(x)


class TestGridSpec:
    def test_only_rtn_is_shippable(self):
        """emit へ渡せるのは RTN 格子だけ（モジュール docstring の MUST）。"""
        assert GridSpec(kind="rtn").shippable
        assert not GridSpec(kind="nf4").shippable
        assert not GridSpec(kind="kmeans_shared").shippable

    def test_the_default_grid_is_i4_g32_rtn(self):
        assert DEFAULT_GRID.kind == "rtn"
        assert DEFAULT_GRID.group_size == 32

    def test_an_unknown_grid_fails_loudly(self):
        with pytest.raises(QuantizeError, match="未対応"):
            GridSpec(kind="int3")  # type: ignore[arg-type]

    def test_a_stride_on_a_grid_without_a_fitted_table_fails_loudly(self):
        """`fit_stride` は表を fit する粒度専用（`quant_methods` と同じ流儀）。"""
        with pytest.raises(QuantizeError, match="kmeans_shared 専用"):
            GridSpec(kind="rtn", fit_stride=2)

    def test_a_non_positive_stride_fails_loudly(self):
        with pytest.raises(QuantizeError, match="1 以上"):
            GridSpec(kind="kmeans_shared", fit_stride=0)


class TestGptqRows:
    def test_a_diagonal_hessian_reproduces_the_storage_round(self):
        """**オラクル**: `H` が対角なら補償項が消えて `fake_quant_int4` とビット同一。

        `U = cholesky(H⁻¹, upper)` が対角になるので `U[j, j+1:]` が厳密に 0 で、
        `W[:, j+1:] −= err ⊗ 0` が 1bit も動かさない — 実装の性質ではなく数学的帰結。
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
        """上の対 — 相関のある `H` では RTN と**違う**結果になる（対角でなければ動く）。"""
        rows = weights(6, 32, seed=1)

        rounded, _ = gptq_quantize_rows(rows, gram(correlated_inputs(96, 32, seed=2)), GRID)

        assert not torch.equal(rounded, round_to_nearest(rows, GROUP))

    def test_every_value_lands_on_the_int4_grid(self):
        """丸め済みの全要素が `q·scale`（`|q| ≤ 7`・−8 不使用）に載る = 出荷可能な形。"""
        rows = weights(6, 32, seed=3)

        rounded, ledger = gptq_quantize_rows(rows, gram(correlated_inputs(96, 32, seed=4)), GRID)

        assert ledger is not None
        quantized = quantize_to_int4(rounded, ledger)
        assert int(quantized.abs().amax()) <= INT4_MAX
        assert torch.equal(dequantize_int4(quantized, ledger), rounded)

    def test_the_ledger_is_the_scale_that_was_used_not_a_recomputed_one(self):
        """MUST: 台帳は**量子化に使った値そのもの**（丸め済みから引き直すと別物になる）。

        RTN なら group の amax 要素が必ず `|q| = 7` に乗るので引き直しても同じ scale が出るが、
        GPTQ は補償で amax 要素を押し下げうる。押し下がった group では `|q|` が 7 に届かず、
        `group_scale` を丸め済みへ当て直すと**別の scale** になって格納値が復元されない。

        この模型は**その条件を満たす group を含む**（下の `peak` の assert が条件そのものを
        固定するので、フィクスチャが変わって条件が消えたらここで気づける）。
        """
        rows = weights(6, 32, seed=5)

        rounded, ledger = gptq_quantize_rows(rows, gram(correlated_inputs(96, 32, seed=2)), GRID)

        assert ledger is not None
        peak = quantize_to_int4(rounded, ledger).abs().reshape(rows.shape[0], -1, GROUP).amax(-1)
        assert int((peak < INT4_MAX).sum()), "amax 要素が押し下がった group が 1 つも無い"

        recomputed = group_scale(rounded, GROUP)
        assert not torch.equal(ledger, recomputed)
        # 台帳側でだけ復元が成立する（= 台帳が「使った値」であることの裏取り）。
        assert torch.equal(dequantize_int4(quantize_to_int4(rounded, ledger), ledger), rounded)
        assert not torch.equal(
            dequantize_int4(quantize_to_int4(rounded, recomputed), recomputed), rounded
        )

    def test_correlated_inputs_beat_round_to_nearest(self):
        """相関入力では GPTQ の `H` 加重誤差 `tr(ΔW·H·ΔWᵀ)` が RTN より小さい。"""
        rows = weights(24, 128, seed=7)
        hessian = gram(correlated_inputs(512, 128, seed=8))

        rounded, _ = gptq_quantize_rows(rows, hessian, GridSpec(group_size=32))

        assert weighted_error(rounded - rows, hessian) < weighted_error(
            round_to_nearest(rows, 32) - rows, hessian
        )

    def test_the_same_input_gives_a_bit_identical_result(self):
        """MUST: 乱数を 1 つも使わない（seed を振っても結果が動かない）。"""
        rows = weights(6, 32, seed=9)
        hessian = gram(correlated_inputs(96, 32, seed=10))

        first, _ = gptq_quantize_rows(rows, hessian, GRID)
        torch.manual_seed(999)
        second, _ = gptq_quantize_rows(rows, hessian, GRID)

        assert torch.equal(first, second)

    @pytest.mark.parametrize("kind", ["nf4", "kmeans_shared"])
    def test_a_grid_without_a_storage_path_returns_no_ledger(self, kind):
        """`quant_methods` と同じ MUST — 格納経路の無い方式へ emit の口を作らない。"""
        rows = weights(6, 32, seed=11)

        rounded, ledger = gptq_quantize_rows(
            rows, gram(correlated_inputs(96, 32, seed=12)), GridSpec(kind=kind, group_size=GROUP)
        )

        assert ledger is None
        assert rounded.shape == rows.shape

    def test_the_nf4_grid_lands_on_the_nf4_table(self):
        """グリッドを差し替えれば丸め先も差し替わる（`rtn` 一択の恒真化を排す）。"""
        rows = weights(6, 32, seed=13)
        table = levels_tensor(NF4_LEVELS)

        rounded, _ = gptq_quantize_rows(
            rows, torch.eye(32, dtype=torch.float64), GridSpec(kind="nf4", group_size=GROUP)
        )

        scale = group_absmax_scale(rows, GROUP, float(table.abs().max()))
        grouped = grouped_view(rounded, GROUP, "検査")
        allowed = scale.unsqueeze(-1).unsqueeze(-1) * table
        assert bool((grouped.unsqueeze(-1) == allowed).any(dim=-1).all())

    def test_an_axis_the_group_size_does_not_divide_fails_loudly(self):
        """端数 group を作らない（`quantize.grouped_view` の共有 — ADR 0069 決定 2）。"""
        with pytest.raises(QuantizeError, match="割り切れない"):
            gptq_quantize_rows(weights(3, 20, seed=14), torch.eye(20, dtype=torch.float64), GRID)

    def test_a_hessian_that_never_saw_an_input_fails_loudly(self):
        """`H` が丸ごと 0 だと damping も 0 になり特異 — 黙って近似しない。"""
        with pytest.raises(QuantizeError, match="対角平均"):
            gptq_quantize_rows(weights(3, 32, seed=15), torch.zeros(32, 32), GRID)

    def test_a_hessian_of_the_wrong_shape_fails_loudly(self):
        with pytest.raises(QuantizeError, match=r"\[in, in\]"):
            gptq_quantize_rows(weights(3, 32, seed=16), torch.eye(16, dtype=torch.float64), GRID)


class TestGptqOptions:
    """掃引軸 `static_groups` / `act_order` / `damping`（既定は現行経路とビット同一の opt-in）。

    見るのは 3 点: ①既定のままなら 1bit も動かない ② `static_groups` の scale が**丸める前の
    元の連続列ブロック**由来である ③ `act_order` は処理順しか変えず、格納の形（元列順・連続
    group・group scale の台帳）を動かさない。
    """

    @staticmethod
    def fixture(seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        """`H` が対角から遠い（= 補償が実際に効く）小さい模型。"""
        return weights(6, 32, seed=seed), gram(correlated_inputs(96, 32, seed=seed + 1))

    @staticmethod
    def order_of(hessian: torch.Tensor) -> torch.Tensor:
        """`diag(H)` 降順の処理順（実装と同じ安定ソート）。"""
        return torch.argsort(torch.diagonal(hessian), descending=True, stable=True)

    def test_the_options_off_are_bit_identical_to_the_call_without_them(self):
        """MUST: 掃引軸を足しても既定経路は 1bit も動かない（既存 i4 配布資産の再現性）。"""
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
        """`static_groups=True` の台帳 = **丸める前**の `rows` の連続列ブロックの group scale。

        対（恒真化の排除）は dynamic 側 — 補償済みの現在値から採るので**別物**になる。
        差が出ない模型ではこの門は何も言っていないので、差そのものを assert で固定する。
        """
        rows, hessian = self.fixture(seed=103)

        static, static_ledger = gptq_quantize_rows(rows, hessian, GRID, static_groups=True)
        dynamic, dynamic_ledger = gptq_quantize_rows(rows, hessian, GRID)

        assert static_ledger is not None
        assert dynamic_ledger is not None
        assert torch.equal(static_ledger, group_scale(rows, GROUP))
        assert not torch.equal(dynamic_ledger, static_ledger), "dynamic と差が出ない模型"
        assert not torch.equal(static, dynamic)
        # 台帳は「使った scale そのもの」のまま = 格納値が復元される（MUST は静的化でも不変）。
        assert torch.equal(
            dequantize_int4(quantize_to_int4(static, static_ledger), static_ledger), static
        )

    def test_a_diagonal_hessian_reproduces_the_storage_round_with_static_groups(self):
        """オラクルは静的化でも成立 — 対角 `H` では補償が 0 なので dynamic と static が一致する。"""
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
        """**オラクル**: 対角 `H` なら並べ替えても `fake_quant_int4` とビット同一。

        補償項が厳密に 0 なので丸めの中身は処理順に依らない。したがって一致することは
        「並べ替えた列が**元の列へ**戻り、group 帰属も元の連続ブロックのまま」の証明になる
        （upstream の `Q = Q[:, invperm]` と同値）。対角値は非一様にして perm を**恒等でない**
        ものにする — 恒等 perm では並べ替えの門にならない。
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
        """対 — 相関 `H` では処理順が実際に効く。それでも**格納の形は 1 つも動かない**。"""
        rows, hessian = self.fixture(seed=105)
        assert not torch.equal(self.order_of(hessian), torch.arange(32))

        ordered, ordered_ledger = gptq_quantize_rows(
            rows, hessian, GRID, static_groups=True, act_order=True
        )
        plain, plain_ledger = gptq_quantize_rows(rows, hessian, GRID, static_groups=True)

        assert not torch.equal(ordered, plain)
        assert ordered_ledger is not None
        assert plain_ledger is not None
        # group 整列（元順・連続 group）は不変 = ランタイム側の変更ゼロ。
        assert torch.equal(ordered_ledger, plain_ledger)
        quantized = quantize_to_int4(ordered, ordered_ledger)
        assert int(quantized.abs().amax()) <= INT4_MAX
        assert torch.equal(dequantize_int4(quantized, ordered_ledger), ordered)

    def test_act_order_without_static_groups_fails_loudly(self):
        """dynamic group で並べ替えると group 帰属が写し戻せない — 黙って近似しない。"""
        rows, hessian = self.fixture(seed=106)

        with pytest.raises(QuantizeError, match="static_groups=True が前提"):
            gptq_quantize_rows(rows, hessian, GRID, act_order=True)

    def test_the_explicit_default_damping_matches_the_constant(self):
        rows, hessian = self.fixture(seed=107)

        explicit, _ = gptq_quantize_rows(rows, hessian, GRID, damping=GPTQ_DAMPING)
        default, _ = gptq_quantize_rows(rows, hessian, GRID)

        assert torch.equal(explicit, default)

    def test_damping_reaches_the_solver(self):
        """対照 — `damping` は `_inverse_cholesky` の `λ` に届いて結果を動かす。"""
        rows = weights(24, 128, seed=108)
        hessian = gram(correlated_inputs(512, 128, seed=109))
        grid = GridSpec(group_size=32)

        light, _ = gptq_quantize_rows(rows, hessian, grid, damping=0.001)
        heavy, _ = gptq_quantize_rows(rows, hessian, grid, damping=0.1)

        assert not torch.equal(light, heavy)

    @pytest.mark.parametrize("damping", [0.0, -0.01])
    def test_a_non_positive_damping_fails_loudly(self, damping):
        """`λ = 0` では対角 0 の入力チャネルが残って `H` が特異になりうる。"""
        rows, hessian = self.fixture(seed=110)

        with pytest.raises(QuantizeError, match="damping は正の値"):
            gptq_quantize_rows(rows, hessian, GRID, damping=damping)


class TestAwqSearch:
    def test_a_flat_activation_profile_keeps_alpha_zero(self):
        """全チャネル同じ大きさなら `s` は α に依らず 1 — 同値は小さい α が勝つ（決定性）。"""
        rows = weights(6, 32, seed=17)

        search = awq_search_scale(
            rows, torch.ones(32, dtype=torch.float64), correlated_inputs(64, 32, seed=18), GRID
        )

        assert search.alpha == 0.0
        assert torch.equal(search.channel_scale, torch.ones(32, dtype=torch.float64))

    def test_a_skewed_activation_profile_moves_alpha_off_zero(self):
        """対照 — 外れチャネルがあると α > 0 が目的関数を**実際に**下げる。"""
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
        """`s` 全体の定数倍は absmax scale に吸収されるので、正規化しないと α が意味を失う。"""
        torch.manual_seed(22)
        act_amax = torch.rand(32, dtype=torch.float64) * 5.0 + 0.01

        search = awq_search_scale(
            weights(6, 32, seed=23), act_amax, correlated_inputs(64, 32, seed=24), GRID
        )

        assert float(torch.log(search.channel_scale).mean()) == pytest.approx(0.0, abs=1e-10)

    def test_the_alpha_grid_has_seventeen_points(self):
        """α ∈ {0, 1/16, …, 1}（採用値が格子上にある）。"""
        torch.manual_seed(25)
        act_amax = torch.rand(32, dtype=torch.float64) * 5.0 + 0.01

        search = awq_search_scale(
            weights(6, 32, seed=26), act_amax, correlated_inputs(64, 32, seed=27), GRID
        )

        assert search.alpha in {step / AWQ_ALPHA_STEPS for step in range(AWQ_ALPHA_STEPS + 1)}

    def test_no_sample_fails_loudly(self):
        with pytest.raises(QuantizeError, match="0 行"):
            awq_search_scale(
                weights(6, 32, seed=28), torch.ones(32, dtype=torch.float64), torch.zeros(0, 32)
            )

    def test_an_axis_the_group_size_does_not_divide_fails_loudly(self):
        with pytest.raises(QuantizeError, match="割り切れない"):
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
        """MUST: 同一入力 → ビット同一（乱数不使用・seed を振っても動かない）。"""
        stages = self.two_stages()
        twin = [(prefix, copy.deepcopy(module)) for prefix, module in stages]
        batches = self.batch()

        calibrate_stages(stages, batches, spec=GRID)
        torch.manual_seed(999)
        calibrate_stages(twin, batches, spec=GRID)

        for (_, quantized), (_, other) in zip(stages, twin, strict=True):
            assert torch.equal(quantized.weight.detach(), other.weight.detach())

    def test_the_ledger_keys_are_model_wide_fqns(self):
        """台帳のキーは stage 内の局所名ではなく**モデル内 FQN**（emit と同じ空間）。"""
        report = calibrate_stages(self.two_stages(), self.batch(), spec=GRID)

        assert isinstance(report, CalibReport)
        assert report.int4 is not None
        assert sorted(report.int4.scales) == ["block.0.weight", "block.1.weight"]
        assert report.int4.group_size == GROUP
        assert [layer.fqn for layer in report.layers] == ["block.0.weight", "block.1.weight"]
        assert all(layer.tokens == 24 for layer in report.layers)
        assert report.modules == 2

    def test_stage_one_quantization_reaches_the_stage_two_hessian(self):
        """誤差伝播: 後段の `H` は**量子化後の**前段の出力から採られる。

        故障注入は「伝播を止めた `H`」— 元の重みの出力で採った `H` から丸めると結果が変わる。
        変わらなければ 3 の再 forward は何の役にも立っていないことになる。
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
        """`include` は **stage 内の局所モジュール FQN** の述語（`quantize` と同じ意味）。"""
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
        """「方式を指定したのに 0 本」を沈黙させない（`quantize` 側と同じ常設診断）。"""
        with pytest.raises(QuantizeError, match="1 本も無い"):
            calibrate_stages([("norm", nn.LayerNorm(32))], self.batch(), spec=GRID)
        with pytest.raises(QuantizeError, match="1 本も無い"):
            calibrate_stages(self.two_stages(), self.batch(), spec=GRID, include=lambda _fqn: False)

    def test_no_stage_fails_loudly(self):
        with pytest.raises(QuantizeError, match="stage が 1 つも無い"):
            calibrate_stages([], self.batch(), spec=GRID)

    def test_no_batch_fails_loudly(self):
        with pytest.raises(QuantizeError, match="校正バッチが 1 つも無い"):
            calibrate_stages(self.two_stages(), [], spec=GRID)

    def test_an_unknown_method_fails_loudly(self):
        with pytest.raises(QuantizeError, match="未対応"):
            calibrate_stages(self.two_stages(), self.batch(), method="smooth", spec=GRID)  # type: ignore[arg-type]

    def test_an_axis_the_group_size_does_not_divide_fails_loudly(self):
        torch.manual_seed(33)
        with pytest.raises(QuantizeError, match="割り切れない"):
            calibrate_stages(
                [("odd", nn.Linear(20, 4))],
                [((correlated_inputs(8, 20, seed=34),), {})],
                spec=GRID,
            )

    def test_a_layer_the_calibration_never_reaches_fails_loudly(self):
        """forward に載っていない linear（= `H` が採れない）を黙って丸めない。"""

        class Detour(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.used = nn.Linear(32, 32)
                self.unused = nn.Linear(32, 32)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.used(x)

        torch.manual_seed(35)
        with pytest.raises(QuantizeError, match="1 行も流れていない"):
            calibrate_stages([("d", Detour())], self.batch(), spec=GRID)


class TestCalibrateStagesDevice:
    """計算デバイスは **stage から導き**、校正バッチは呼び出し側の置き場に留める（JIT 移動）。

    校正バッチは stage あたり数百本 × 数十 MB になりうるので、駆動が勝手に計算デバイスへ
    引き上げると重みと同じ VRAM を取り合う。ここが固定するのは「バッチの置き場は動かない」
    「stage が居るデバイスで計算が回る」の 2 点で、全 CPU の既定経路が no-op になることは
    上のビット同一テストが受け持つ。
    """

    def test_the_batches_reach_the_stage_without_a_copy(self):
        """CPU の既定経路では `Tensor.to` が self を返す — 写しすら作らない。

        stage が**実際に受け取った**テンソルで見る（呼び出し側のリストを見るだけでは、駆動が
        入力を書き換えないことしか言えず恒真化する）。
        """
        stages = TestCalibrateStages.two_stages()
        batches = TestCalibrateStages.batch()
        given = batches[0][0][0]
        seen: list[torch.Tensor] = []
        stages[0][1].register_forward_pre_hook(lambda _module, args: seen.append(args[0]))

        calibrate_stages(stages, batches, spec=GRID)

        assert seen, "先頭 stage が 1 度も呼ばれていない"
        assert all(tensor is given for tensor in seen)

    def test_a_stage_without_parameters_fails_loudly(self):
        """デバイスの導出元が無い stage を黙って CPU 扱いにしない。"""
        with pytest.raises(QuantizeError, match="パラメータを 1 つも持たない"):
            calibrate_stages([("act", nn.ReLU())], TestCalibrateStages.batch(), spec=GRID)

    def test_a_tuple_keyword_reaches_the_stage_without_a_copy(self):
        """組で渡る kwargs（rope の `(cos, sin)`）も CPU 経路では写しを作らない。"""
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

        assert seen, "stage が 1 度も呼ばれていない"
        assert all(tuple(given) == rotary for given in seen)
        assert all(given[0] is rotary[0] and given[1] is rotary[1] for given in seen)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA が無い環境では検証できない")
    def test_a_cuda_stage_runs_from_cpu_batches(self):
        """GPU の stage × CPU のバッチ（組の kwargs 込み）で完走し、バッチは CPU に置かれたまま。

        共有 venv の torch は CPU 版なので既定では skip される（GPTQ 校正を CUDA で回すのは
        感度実験用の経路で、配布経路は CPU のまま）。
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
        """全チャネルの `|x|` max が厳密に 1 の校正入力（AWQ の `s` が α に依らず 1 になる形）。"""
        torch.manual_seed(37)
        return [((torch.randn(24, 32).sign(),), {})]

    def test_awq_with_a_flat_activation_profile_matches_round_to_nearest(self):
        """α = 0 は `s = 1` なので、書き戻る重みは `fake_quant_int4` とビット同一。"""
        stages = self.linear_stage()
        (_, module) = stages[0]
        twin = copy.deepcopy(module)
        fake_quant_int4(twin, group_size=GROUP)

        report = calibrate_stages(stages, self.sign_batch(), method="awq", spec=GRID)

        assert [layer.alpha for layer in report.layers] == [0.0]
        assert torch.equal(module.weight.detach(), twin.weight.detach())

    def test_awq_returns_no_ledger(self):
        """MUST: `W_eff = Q(W')/s` は単独では格納できない（fold か companion が要る）。"""
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
        """MUST: 部分標本で fit したら測定側の出力へ明記する（`quant_methods` と同文）。"""
        report = calibrate_stages(
            self.linear_stage(),
            [((correlated_inputs(64, 32, seed=39),), {})],
            spec=GridSpec(kind="kmeans_shared", group_size=GROUP, fit_stride=3),
        )

        assert report.fit_stride == 3
        assert "fit_stride 3" in report.describe()
        assert report.int4 is None


class TestCalibrateStagesOptions:
    """掃引軸が駆動から層まで**そのまま**届く（sweep は `calibrate_stages` 越しに回す）。"""

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
        # 対 — 既定で回した結果とは違う（軸が素通しされていることの裏取り）。
        default, _ = gptq_quantize_rows(original, gram(inputs), GRID)
        assert not torch.equal(expected, default)

    def test_the_defaults_leave_the_driver_bit_identical(self):
        stage, inputs = self.stage_and_inputs()
        original = stage.weight.detach().clone()

        calibrate_stages([("only", stage)], [((inputs,), {})], spec=GRID)

        expected, _ = gptq_quantize_rows(original, gram(inputs), GRID)
        assert torch.equal(stage.weight.detach(), expected)

    def test_gptq_options_on_a_method_without_a_hessian_fail_loudly(self):
        """`GridSpec.fit_stride` と同じ流儀 — 効かない軸を渡されて黙って無視しない。"""
        stage, inputs = self.stage_and_inputs()

        with pytest.raises(QuantizeError, match="効かない"):
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
        """同じ模型・同じ校正入力で `advance_kwargs` だけを差し替えて回す。"""
        torch.manual_seed(42)
        stages: list[tuple[str, nn.Module]] = [("a", Chain(32, 32)), ("b", Chain(32, 16))]
        inputs = correlated_inputs(24, 32, seed=43)
        calibrate_stages(stages, [((inputs,), {"shift": 3.0})], spec=GRID, advance_kwargs=advance)
        return [module.proj.weight.detach().clone() for _, module in stages]

    def test_the_keyword_arguments_carry_over_unchanged_by_default(self):
        """既定は「stage 間で kwargs 不変」— 明示の恒等上書きと**同じ結果**になる。"""
        assert all(
            torch.equal(default, explicit)
            for default, explicit in zip(
                self.run_with(None), self.run_with(lambda _index, kwargs: kwargs), strict=True
            )
        )

    def test_advance_kwargs_overrides_the_next_stage_arguments(self):
        """対照 — 上書き口が実際に効く（後段の `shift` を変えると `H` が変わる）。

        前段は `current` の kwargs で回るので**変わらない**のが正しい（上書きは次 stage 用）。
        """
        default = self.run_with(None)
        overridden = self.run_with(lambda _index, _kwargs: {"shift": 0.0})

        assert torch.equal(default[0], overridden[0])
        assert not torch.equal(default[1], overridden[1])

    def test_an_output_that_is_not_a_tensor_fails_loudly(self):
        with pytest.raises(QuantizeError, match="選べない"):
            first_tensor_output({"hidden": torch.zeros(2)})

    def test_a_tensor_output_passes_through(self):
        tensor = torch.zeros(2)

        assert first_tensor_output(tensor) is tensor
        assert first_tensor_output((tensor, "state")) is tensor
