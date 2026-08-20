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


# ---- 校正付き丸め（波 J-2）--------------------------------------------------


class TinyAttention(nn.Module):
    """量子化対象の linear 2 本（in 軸は group 32 で割り切れる）。"""

    def __init__(self, features: int) -> None:
        super().__init__()
        self.q_proj = nn.Linear(features, features, bias=False)
        self.o_proj = nn.Linear(features, features, bias=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.o_proj(self.q_proj(hidden))


class TinyDecoderLayer(nn.Module):
    """decoder layer 相当。`bias` は「layer_type ごとに違う kwargs」の代役（加算で観測）。"""

    def __init__(self, features: int) -> None:
        super().__init__()
        self.self_attn = TinyAttention(features)

    def forward(self, hidden: torch.Tensor, bias: float = 0.0) -> torch.Tensor:
        return self.self_attn(hidden + bias)


class TinyConfig:
    """`Gemma3TextConfig` の代役（`layer_types` だけを持つ）。"""

    def __init__(self, layer_types: tuple[str, ...]) -> None:
        self.layer_types = layer_types


class TinyText(nn.Module):
    """`Gemma3TextModel` 相当（`.config.layer_types` / `.layers` / `.embed_tokens`）。"""

    def __init__(self, vocab: int, features: int, layer_types: tuple[str, ...]) -> None:
        super().__init__()
        self.config = TinyConfig(layer_types)
        self.embed_tokens = nn.Embedding(vocab, features)
        self.layers = nn.ModuleList(TinyDecoderLayer(features) for _ in layer_types)


#: layer_type ごとの kwargs（Gemma3 の sliding / full 2 系統の代役）。
TINY_BIAS = {"sliding_attention": 0.5, "full_attention": -0.25}

#: 先頭が sliding・途中に full が 1 枚（`capture_stage_batches` が 2 種類を捕まえる形）。
TINY_LAYER_TYPES = ("sliding_attention", "sliding_attention", "full_attention")


class TinyWrapper(nn.Module):
    """`EmbeddingWrapper` 相当 — decoder の**外**に Dense を 1 本持つ（校正対象外の席）。"""

    def __init__(self, vocab: int = 8, features: int = 32) -> None:
        super().__init__()
        self.model = TinyText(vocab, features, TINY_LAYER_TYPES)
        self.dense2 = nn.Linear(features, features, bias=False)

    def forward(self, input_ids: torch.Tensor, pool_mask: torch.Tensor) -> torch.Tensor:
        hidden = self.model.embed_tokens(input_ids)
        for kind, layer in zip(self.model.config.layer_types, self.model.layers, strict=True):
            hidden = layer(hidden, bias=TINY_BIAS[kind])
        pooled = torch.sum(hidden * pool_mask.unsqueeze(-1), dim=1)
        return self.dense2(pooled)


def tiny_inputs(count: int = 2) -> tuple[torch.Tensor, ...]:
    """長さの違う id 列（校正入力の代役 — tokenizer は通さない）。"""
    return tuple(
        torch.arange(2 + index, dtype=torch.int64).unsqueeze(0) % 8 for index in range(count)
    )


def tiny_rig(wrapper: TinyWrapper, inputs: tuple[torch.Tensor, ...]) -> mq.CalibRig:
    stages = mq.decoder_stages(wrapper)
    scan, stats = mq.calib_targets(stages)
    return mq.CalibRig(
        stages=stages,
        scan=scan,
        stats=stats,
        batches=mq.capture_stage_batches(wrapper, inputs),
    )


class TestCalibConfigs:
    def test_every_calib_config_is_selectable(self):
        assert tuple(config.name for config in mq.CALIB_CONFIGS) == mq.CALIB_NAMES

    def test_selecting_one_config_keeps_the_declaration_order(self):
        chosen = mq.select_calib(["awq-rtn", "gptq-rtn"])

        assert [config.name for config in chosen] == ["gptq-rtn", "awq-rtn"]

    def test_an_empty_selection_runs_every_calib_config(self):
        assert mq.select_calib([]) == mq.CALIB_CONFIGS

    def test_selecting_only_a_method_leaves_no_calib_config(self):
        """`--only nf4` は方式だけ — 校正付き構成を巻き込まない。"""
        assert mq.select_calib(["nf4"]) == ()

    def test_the_group_size_is_pinned_for_every_calib_config(self):
        assert all(config.grid.group_size == mq.GROUP_SIZE for config in mq.CALIB_CONFIGS)

    def test_no_calib_config_fits_its_table_on_a_subsample_by_default(self):
        assert all(config.grid.fit_stride == 1 for config in mq.CALIB_CONFIGS)

    def test_the_label_names_both_the_method_and_the_grid(self):
        labels = {config.name: config.label for config in mq.CALIB_CONFIGS}

        assert labels["gptq-nf4"] == "gptq/nf4"
        assert labels["awq-gptq-rtn"] == "awq+gptq/rtn"

    def test_the_size_projection_takes_a_calib_config(self):
        """校正は格子を変えない — RTN グリッドの bpw は方式側と同じ 5.0。"""
        config = next(c for c in mq.CALIB_CONFIGS if c.name == "gptq-rtn")

        projection = mq.size_projection(config, _stats(elements=3200, groups=100))

        assert projection["bitsPerWeight"] == pytest.approx(5.0)

    def test_the_calibrated_codebook_costs_one_table_per_layer(self):
        """core の `kmeans_shared` は**層内**の表（全体 1 枚の kmeans:shared とは式が違う）。"""
        config = next(c for c in mq.CALIB_CONFIGS if c.name == "gptq-kmeans")
        stats = _stats(modules=4, elements=3200, groups=100)

        projection = mq.size_projection(config, stats)

        tables = mq.CODEBOOK_ENTRY_BITS * mq.DEFAULT_CODEBOOK_LEVELS * stats.modules
        assert projection["bitsPerWeight"] == pytest.approx(5.0 + tables / stats.elements)


class TestCalibStages:
    def test_the_stage_fqns_are_model_wide(self):
        """台帳のキーを `Int4Report` と同じ FQN 空間へ揃える（core の `StageSpec` 契約）。"""
        scan, stats = mq.calib_targets(mq.decoder_stages(TinyWrapper()))

        assert sorted(scan)[:2] == [
            "model.layers.0.self_attn.o_proj.weight",
            "model.layers.0.self_attn.q_proj.weight",
        ]
        assert stats.modules == 2 * len(TINY_LAYER_TYPES)

    def test_the_dense_outside_the_decoder_is_not_in_the_scan(self):
        """Dense 2 段は decoder の外 — 校正の対象集合に入らない（表の対象名の由来）。"""
        scan, _stats = mq.calib_targets(mq.decoder_stages(TinyWrapper()))

        assert not any("dense" in fqn for fqn in scan)

    def test_each_stage_keeps_its_layer_type(self):
        stages = mq.decoder_stages(TinyWrapper())

        assert [stage.layer_type for _prefix, stage in stages] == list(TINY_LAYER_TYPES)

    def test_a_layer_type_table_that_does_not_match_the_layers_fails_loudly(self):
        wrapper = TinyWrapper()
        wrapper.model.config.layer_types = TINY_LAYER_TYPES[:-1]

        with pytest.raises(AssertionError, match="食い違う"):
            mq.decoder_stages(wrapper)

    def test_a_stage_applies_the_keyword_arguments_of_its_own_layer_type(self):
        """MUST: 層ごとに違う mask / RoPE 表を取り違えない（`LayerStage` の存在理由）。"""
        torch.manual_seed(0)
        wrapper = TinyWrapper()
        stages = mq.decoder_stages(wrapper)
        hidden = torch.randn(1, 3, 32)
        layer_kwargs = {kind: {"bias": bias} for kind, bias in TINY_BIAS.items()}

        with torch.no_grad():
            full = stages[2][1](hidden, layer_kwargs=layer_kwargs)
            direct = wrapper.model.layers[2](hidden, bias=TINY_BIAS["full_attention"])

        assert torch.equal(full, direct)


class TestCatcher:
    def test_it_captures_one_batch_per_calibration_text(self):
        wrapper = TinyWrapper()

        batches = mq.capture_stage_batches(wrapper, tiny_inputs(3))

        assert len(batches) == 3
        assert [int(args[0].shape[1]) for args, _kwargs in batches] == [2, 3, 4]

    def test_it_captures_the_keyword_arguments_of_every_layer_type(self):
        """片方の layer_type しか捕まえないと、その種類の層が再実行できない。"""
        wrapper = TinyWrapper()

        _args, kwargs = mq.capture_stage_batches(wrapper, tiny_inputs(1))[0]

        assert kwargs[mq.LAYER_KWARGS] == {
            "sliding_attention": {"bias": 0.5},
            "full_attention": {"bias": -0.25},
        }

    def test_the_hooks_are_removed_even_though_the_forward_was_aborted(self):
        wrapper = TinyWrapper()

        mq.capture_stage_batches(wrapper, tiny_inputs(1))

        assert not any(layer._forward_pre_hooks for layer in wrapper.model.layers)


class TestCalibrationRun:
    def test_it_rounds_every_scanned_linear(self):
        torch.manual_seed(1)
        wrapper = TinyWrapper()
        rig = tiny_rig(wrapper, tiny_inputs(2))
        before = {fqn: weight.detach().clone() for fqn, weight in rig.scan.items()}

        report = mq.apply_calib(mq.CALIB_CONFIGS[0], rig)
        mq.assert_calib_covers_scan(report, rig.scan, "gptq-rtn")

        assert report.modules == len(rig.scan)
        assert all(not torch.equal(rig.scan[fqn], value) for fqn, value in before.items())

    def test_the_dense_outside_the_decoder_is_left_alone(self):
        """校正は decoder 内だけ — 外の Dense は 1 ビットも動かない。"""
        torch.manual_seed(2)
        wrapper = TinyWrapper()
        rig = tiny_rig(wrapper, tiny_inputs(2))
        dense = wrapper.dense2.weight.detach().clone()

        mq.apply_calib(mq.CALIB_CONFIGS[0], rig)

        assert torch.equal(wrapper.dense2.weight, dense)

    def test_every_calib_config_runs_end_to_end(self):
        for config in mq.CALIB_CONFIGS:
            torch.manual_seed(3)
            wrapper = TinyWrapper()
            rig = tiny_rig(wrapper, tiny_inputs(2))

            report = mq.apply_calib(config, rig)
            mq.assert_calib_covers_scan(report, rig.scan, config.name)

            assert report.method == config.method
            assert report.grid == config.grid.kind

    def test_a_subsampled_table_is_reported(self):
        """MUST: `fit_stride` を使ったら出力へ明記（`describe` が拾う）。"""
        torch.manual_seed(4)
        wrapper = TinyWrapper()
        rig = tiny_rig(wrapper, tiny_inputs(2))
        kmeans = next(config for config in mq.CALIB_CONFIGS if config.name == "gptq-kmeans")

        report = mq.apply_calib(kmeans, rig, shared_stride=4)

        assert "fit_stride 4" in report.describe()

    def test_a_scan_the_calibration_missed_fails_loudly(self):
        torch.manual_seed(5)
        wrapper = TinyWrapper()
        rig = tiny_rig(wrapper, tiny_inputs(2))
        report = mq.apply_calib(mq.CALIB_CONFIGS[0], rig)
        widened = {**rig.scan, "model.layers.9.self_attn.q_proj.weight": torch.zeros(1)}

        with pytest.raises(AssertionError, match="一致しない"):
            mq.assert_calib_covers_scan(report, widened, "gptq-rtn")
