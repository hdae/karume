"""`embeddinggemma/measure_quant.py` の pure 部分（実重み不要分）。

実測そのものは手動（1.2GB の実重みを 15 構成ぶん丸めて回す）。ここで固定するのは、壊れると
**測定値が静かに嘘になる**側だけ:

- **方式を積み重ねない** — 構成ごとに pristine へ戻すこと（戻さないと「NF4 の上の MXFP4」を
  測ることになり、方式比較そのものが成立しない）
- サイズ試算の式 — g=32 の group scale 込みで i4 / FP4 / NF4 が 5.0 bpw・MXFP4 が 4.25 bpw・
  codebook が表のコスト込み（表に載る数字の出どころ）
- 指標 — cosine の境界値とスケール不変性、ペア行列が上三角 10 対、意味順序の崩壊の検出
- 対象集合の素性 — group 数を「丸めが group を割る軸」と同じ軸で数えること・整除違反が
  fail loudly
- 恒真化の遮断 — 丸めの素通りと `op_types` の効き目の門が**実際に落ちる**こと
"""

from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from embeddinggemma import export as eg
from embeddinggemma import measure_quant as mq
from karume.quantize import fake_quant_int4, iter_quant_targets


class TinyModel(nn.Module):
    """linear 1 本 + embedding 1 本（in 軸はどちらも group 32 で割り切れる）。"""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(32, 4, bias=False)
        self.embed = nn.Embedding(8, 32)


def _stats(**overrides: int) -> mq.TargetStats:
    """既定は「32 要素 = 1 group・1 テンソル・1 チャネル」の最小形。"""
    base = {"modules": 1, "elements": 32, "groups": 1, "channels": 1}
    return mq.TargetStats(**{**base, **overrides})


def _method(name: str) -> mq.Method:
    return next(method for method in mq.METHODS if method.name == name)


class TestSizeProjection:
    """表に載る bpw の出どころ（`docs/ir-v1.md` の i4 格納形と OCP MX の逐語）。"""

    @pytest.mark.parametrize("name", ["rtn-i4-g32", "fp4", "nf4"])
    def test_group_absmax_methods_land_on_five_bits(self, name):
        """4bit ペイロード + group 32 ごとの F32 scale = 4 + 32/32 = 5.0 bpw。"""
        projection = mq.size_projection(_method(name), _stats(elements=3200, groups=100))

        assert projection["bitsPerWeight"] == pytest.approx(5.0)

    def test_mxfp4_lands_on_four_and_a_quarter_bits(self):
        """共有 scale が E8M0（8bit）なので 4 + 8/32 = 4.25 bpw。"""
        projection = mq.size_projection(_method("mxfp4"), _stats(elements=3200, groups=100))

        assert projection["bitsPerWeight"] == pytest.approx(4.25)

    def test_the_per_tensor_codebook_costs_one_table_per_tensor(self):
        """scale が無い代わりに層ごとの表（16 centroid × F32）が乗る。"""
        stats = _stats(modules=10, elements=3200, groups=100, channels=40)

        projection = mq.size_projection(_method("kmeans:per_tensor"), stats)

        assert projection["bitsPerWeight"] == pytest.approx(4.0 + 32 * 16 * 10 / 3200)

    def test_the_per_channel_codebook_costs_one_table_per_channel(self):
        """表のコストが**チャネル数**に比例する（per_tensor との違いはここだけ）。"""
        stats = _stats(modules=10, elements=3200, groups=100, channels=40)

        projection = mq.size_projection(_method("kmeans:per_channel"), stats)

        assert projection["bitsPerWeight"] == pytest.approx(4.0 + 32 * 16 * 40 / 3200)

    def test_the_shared_codebook_costs_one_table_for_the_whole_model(self):
        """表は 1 枚だが group scale が要る（= i4 と同じ 5.0 bpw + 表 1 枚ぶん）。"""
        stats = _stats(modules=10, elements=3200, groups=100, channels=40)

        projection = mq.size_projection(_method("kmeans:shared"), stats)

        assert projection["bitsPerWeight"] == pytest.approx(5.0 + 32 * 16 / 3200)

    def test_the_ratio_is_the_bpw_against_f32(self):
        """比は「f32 格納に対する何 %」— MiB の 2 系列と同じ数から出ていること。"""
        projection = mq.size_projection(_method("nf4"), _stats(elements=3200, groups=100))

        assert projection["ratio"] == pytest.approx(5.0 / 32.0)
        assert projection["projectedMiB"] == pytest.approx(projection["f32MiB"] * 5.0 / 32.0)


class TestTargetStats:
    def test_it_counts_groups_on_the_axis_the_rounding_splits(self):
        """group 数は `channel_rows` の平坦形（= 丸めが group を割る軸）から数える。"""
        model = TinyModel()

        linear = mq.target_stats(model, mq.TARGET_LINEAR)
        widened = mq.target_stats(model, mq.TARGET_WITH_EMBEDDING)

        # linear `[4,32]` … 4 行 × (32/32) group / embedding `[8,32]` … 8 行 × 1 group。
        assert (linear.modules, linear.elements, linear.groups, linear.channels) == (1, 128, 4, 4)
        assert (widened.modules, widened.elements) == (2, 128 + 256)
        assert (widened.groups, widened.channels) == (4 + 8, 4 + 8)

    def test_an_indivisible_quantization_axis_fails_loudly(self):
        """MUST: 端数 group は格納できない形（ADR 0069 決定 2）— その bpw を出さない。"""

        class Odd(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.fc = nn.Linear(30, 2, bias=False)

        with pytest.raises(AssertionError, match="割り切れない"):
            mq.target_stats(Odd(), mq.TARGET_LINEAR)

    def test_an_empty_target_fails_loudly(self):
        with pytest.raises(AssertionError, match="1 本も無い"):
            mq.target_stats(nn.Module(), mq.TARGET_LINEAR)


class TestTargets:
    def test_the_narrow_target_leaves_the_vocabulary_table_alone(self):
        """対象 2 形の差は `embed_tokens` 1 本ぶん（門 ② の期待値そのもの）。"""
        model = TinyModel()

        narrow = [fqn for fqn, _w, _a in iter_quant_targets(model, mq.TARGET_LINEAR.op_types)]
        wide = [fqn for fqn, _w, _a in iter_quant_targets(model, mq.WIDEST_OP_TYPES)]

        assert narrow == ["fc.weight"]
        assert wide == ["fc.weight", "embed.weight"]


class TestMethodsAreNotStacked:
    """MUST: 構成ごとに pristine へ戻す（戻さないと方式間の比較が成立しない）。"""

    def _pristine(self, model: nn.Module) -> tuple[dict, dict]:
        weights = {fqn: w for fqn, w, _a in iter_quant_targets(model, mq.WIDEST_OP_TYPES)}
        return weights, {fqn: w.detach().clone() for fqn, w in weights.items()}

    def test_restore_brings_back_the_original_weights_bit_for_bit(self):
        torch.manual_seed(0)
        model = TinyModel()
        weights, pristine = self._pristine(model)

        _method("nf4").apply(model, mq.TARGET_WITH_EMBEDDING.op_types)
        mq.restore(weights, pristine)

        assert all(torch.equal(weights[fqn], pristine[fqn]) for fqn in pristine)

    def test_a_restored_run_matches_a_fresh_one(self):
        """別方式の後に戻して当てた i4 が、素の重みへ当てた i4 とビット一致する。"""
        torch.manual_seed(0)
        model = TinyModel()
        weights, pristine = self._pristine(model)

        _method("fp4").apply(model, mq.TARGET_LINEAR.op_types)
        mq.restore(weights, pristine)
        fake_quant_int4(model, mq.GROUP_SIZE, op_types=mq.TARGET_LINEAR.op_types)
        after_restore = weights["fc.weight"].detach().clone()

        mq.restore(weights, pristine)
        fake_quant_int4(model, mq.GROUP_SIZE, op_types=mq.TARGET_LINEAR.op_types)

        assert torch.equal(after_restore, weights["fc.weight"])

    def test_stacking_the_two_methods_would_have_changed_the_answer(self):
        """恒真化の防止 — 戻さずに重ねた場合は実際に別の値になる（上のテストが効く条件）。"""
        torch.manual_seed(0)
        model = TinyModel()
        weights, pristine = self._pristine(model)

        _method("fp4").apply(model, mq.TARGET_LINEAR.op_types)
        fake_quant_int4(model, mq.GROUP_SIZE, op_types=mq.TARGET_LINEAR.op_types)
        stacked = weights["fc.weight"].detach().clone()

        mq.restore(weights, pristine)
        fake_quant_int4(model, mq.GROUP_SIZE, op_types=mq.TARGET_LINEAR.op_types)

        assert not torch.equal(stacked, weights["fc.weight"])


class TestCosine:
    def test_an_identical_vector_scores_one(self):
        vector = torch.tensor([0.6, 0.8, 0.0])

        assert mq.cosine(vector, vector) == pytest.approx(1.0)

    def test_an_opposite_vector_scores_minus_one(self):
        vector = torch.tensor([0.6, 0.8])

        assert mq.cosine(-vector, vector) == pytest.approx(-1.0)

    def test_it_is_scale_invariant(self):
        """MUST: ノルムの崩れを cosine へ混ぜない（ノルムは別列で見る）。"""
        vector = torch.tensor([1.0, 2.0, 3.0])

        assert mq.cosine(vector * 7.0, vector) == pytest.approx(1.0)

    def test_a_zero_vector_fails_loudly(self):
        """全ゼロは丸めの話ではなく経路が壊れている合図 — NaN を返して測り続けない。"""
        with pytest.raises(AssertionError, match="全ゼロ"):
            mq.cosine(torch.zeros(3), torch.ones(3))


def _vectors(**overrides: torch.Tensor) -> dict[str, torch.Tensor]:
    """5 ケースぶんの合成埋め込み（既定は互いに直交する単位ベクトル）。"""
    base = {name: torch.eye(len(mq.CASE_NAMES))[index] for index, name in enumerate(mq.CASE_NAMES)}
    return {**base, **overrides}


class TestPairCosines:
    def test_it_covers_the_upper_triangle_only(self):
        pairs = mq.pair_cosines(_vectors())

        assert len(pairs) == len(mq.CASE_NAMES) * (len(mq.CASE_NAMES) - 1) // 2
        assert mq.pair_label(eg.NEAR_PAIR) in pairs
        assert mq.pair_label(eg.FAR_PAIR) in pairs

    def test_orthogonal_cases_score_zero(self):
        pairs = mq.pair_cosines(_vectors())

        assert all(value == pytest.approx(0.0) for value in pairs.values())


class TestMeasure:
    def _near_beats_far(self) -> dict[str, torch.Tensor]:
        """近い対（query-en × document-en）だけが揃った合成埋め込み。"""
        near_partner = (
            torch.eye(len(mq.CASE_NAMES))[0] * 0.8 + torch.eye(len(mq.CASE_NAMES))[1] * 0.6
        )
        return _vectors(**{eg.NEAR_PAIR[1]: near_partner})

    def test_the_base_against_itself_is_a_perfect_row(self):
        vectors = self._near_beats_far()

        result = mq.measure(vectors, vectors)

        assert result["caseCosineMin"] == pytest.approx(1.0)
        assert result["pairDriftMaxAbs"] == pytest.approx(0.0)

    def test_it_keeps_the_semantic_order_when_the_near_pair_leads(self):
        vectors = self._near_beats_far()

        result = mq.measure(vectors, vectors)["order"]

        assert result["holds"]
        assert result["nearCosine"] == pytest.approx(0.8)
        assert result["farCosine"] == pytest.approx(0.0)
        assert result["margin"] == pytest.approx(0.8)

    def test_it_reports_a_collapsed_order(self):
        """MUST: 基準比 cosine とは独立に見る — 全ベクトルが近ければ cos は高いまま崩れる。"""
        base = self._near_beats_far()
        collapsed = _vectors(**{eg.FAR_PAIR[1]: base[eg.NEAR_PAIR[0]]})

        result = mq.measure(collapsed, base)["order"]

        assert not result["holds"]
        assert result["farCosine"] == pytest.approx(1.0)

    def test_the_pair_drift_names_the_worst_pair(self):
        base = self._near_beats_far()
        moved = _vectors(**{eg.FAR_PAIR[1]: base[eg.NEAR_PAIR[0]]})

        result = mq.measure(moved, base)

        assert result["pairDriftWorst"] == mq.pair_label(eg.FAR_PAIR)
        assert result["pairDriftMaxAbs"] == pytest.approx(1.0)

    def test_the_case_cosine_ignores_a_pure_norm_change(self):
        """ノルムだけ動いた場合に cosine が 1 のままであること（`l2Norms` 側で見る量）。"""
        base = self._near_beats_far()
        scaled = {name: vector * 0.5 for name, vector in base.items()}

        result = mq.measure(scaled, base)

        assert result["caseCosineMin"] == pytest.approx(1.0)
        assert all(value == pytest.approx(0.5, abs=1e-6) for value in result["l2Norms"].values())


def _entry(*, modules: int, moved: bool = True, cosine: float = 0.99) -> dict[str, object]:
    return {
        "moved": moved,
        "size": {"modules": modules},
        "caseCosine": dict.fromkeys(mq.CASE_NAMES, cosine),
    }


def _configs(**overrides: dict[str, object]) -> dict[str, dict[str, object]]:
    method = _method("nf4")
    configs = {
        mq.BASE_CONFIG: {"caseCosine": dict.fromkeys(mq.CASE_NAMES, 1.0)},
        mq.config_name(method, mq.TARGET_LINEAR): _entry(modules=170),
        mq.config_name(method, mq.TARGET_WITH_EMBEDDING): _entry(modules=171, cosine=0.98),
    }
    return {**configs, **overrides}


def _failures(configs: dict[str, dict[str, object]]) -> list[str]:
    return mq.run_gates(configs, (_method("nf4"),))["failures"]


class TestGates:
    """MUST: 素通りは常に「品質が良い」側の嘘になる — 沈黙させない。"""

    def test_a_well_formed_pair_of_configs_is_green(self):
        assert _failures(_configs()) == []

    def test_a_config_bit_identical_to_the_base_is_red(self):
        configs = _configs()
        configs[mq.config_name(_method("nf4"), mq.TARGET_LINEAR)] = _entry(modules=170, moved=False)

        assert any("素通り" in message for message in _failures(configs))

    def test_a_widened_target_that_did_not_add_the_embedding_is_red(self):
        """`op_types` の切り替えが効かないと、語彙表を測ったことになっている表が出る。"""
        configs = _configs()
        configs[mq.config_name(_method("nf4"), mq.TARGET_WITH_EMBEDDING)] = _entry(
            modules=170, cosine=0.98
        )

        assert any("op_types" in message for message in _failures(configs))

    def test_a_widened_target_with_identical_embeddings_is_red(self):
        """本数だけ増えて値が同じなら、語彙表の丸めが出力へ届いていない。"""
        configs = _configs()
        configs[mq.config_name(_method("nf4"), mq.TARGET_WITH_EMBEDDING)] = _entry(modules=171)

        assert any("届いていない" in message for message in _failures(configs))


class TestMethodTable:
    def test_every_method_is_measured_on_both_targets(self):
        assert len(mq.METHODS) == 7
        assert [target.name for target in mq.TARGETS] == ["linear", "linear+embedding"]

    def test_the_group_size_is_pinned_for_every_method(self):
        """方式比較の軸は「格子の張り方」だけ — group 長は動かさない（ユーザー裁定）。"""
        assert mq.GROUP_SIZE == 32

    def test_selecting_a_subset_keeps_the_declaration_order(self):
        chosen = mq.select_methods(["nf4", "rtn-i4-g32"])

        assert [method.name for method in chosen] == ["rtn-i4-g32", "nf4"]

    def test_an_empty_selection_runs_everything(self):
        assert mq.select_methods([]) == mq.METHODS


class TestTables:
    def _report(self) -> dict[str, object]:
        entry = {
            "method": "nf4",
            "target": "linear",
            "caseCosineMin": 0.9987,
            "caseCosineMean": 0.9991,
            "order": {"nearCosine": 0.71, "farCosine": 0.32, "holds": True},
            "pairDriftMaxAbs": 1.5e-3,
            "size": {
                "modules": 170,
                "elements": 3200,
                "bitsPerWeight": 5.0,
                "projectedMiB": 1.9,
                "f32MiB": 12.2,
                "ratio": 0.15625,
                "formula": "4·N + 32·G",
            },
        }
        return {"configs": {"nf4/linear": entry}}

    def test_the_quality_table_carries_the_order_verdict(self):
        table = mq.quality_table(self._report())

        assert "| nf4 | linear | 170 | 0.998700 | 0.999100 | 0.7100 | 0.3200 | 保持 |" in table

    def test_a_collapsed_order_is_marked_up(self):
        report = self._report()
        report["configs"]["nf4/linear"]["order"]["holds"] = False

        assert "**崩壊**" in mq.quality_table(report)

    def test_the_size_table_shows_the_formula(self):
        table = mq.size_table(self._report())

        assert "| nf4 | linear | 3,200 | 5.000 | 1.9 | 12.2 | 15.6% | 4·N + 32·G |" in table


class TestMibConversion:
    def test_one_mebibyte_is_the_binary_one(self):
        """`MiB` 表記どおり 2^20（10^6 と混ぜると表の数字が 5% ずれる）。"""
        assert mq.MIB == 2**20
        assert math.log2(mq.MIB) == 20
