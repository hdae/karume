"""`sbv2.measure_quant` の w4 まわりの単体テスト（**資産の要らないロジックだけ**）。

実重み・dump を使う測定そのものは台本の検証ゲート（`measure_quant.run_gates`）が受け持つ。
ここで切り分けるのは、資産が無くても成り立つ 4 つの部品:

    ① 対象集合の census（group 長で割り切れない重みの除外と `include` への畳み込み）
    ② サイズ試算の式（方式ごとの実効 bpw）
    ③ 方式の呼び分け（4 種が互いに違う値へ丸め、除外した重みには 1 要素も触らない）
    ④ 配布形との突合（テンソル名で linear の重みスロットだけを拾う）

丸めの実装そのもの（i4 / NF4 / MXFP4 / k-means の数値）は core 側のテストが持つ — ここが
測るのは「台本が core をどう呼び分けているか」だけで、写経した式は 1 つも無い。
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from sbv2 import measure_quant as measure

#: 合成モデルの初期化 seed（方式の差を見るテストは値が退化していないことが前提）。
SEED = 20260819


class TinyNet(nn.Module):
    """w4 の対象になる 5 op 種のうち 4 種を、**適格と除外の両方**で持つ合成モデル。

    量子化軸（= 各 op の in 軸）が group 長 32 で割り切れるかどうかで対を作る:
    `linear` 64 / `odd_linear` 48 / `conv` 受容野 4·8 = 32 / `depthwise` 受容野 1·3 = 3。
    """

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(64, 8)
        self.odd_linear = nn.Linear(48, 4)
        self.conv = nn.Conv1d(4, 6, 8)
        self.depthwise = nn.Conv1d(6, 6, 3, groups=6)
        self.emb = nn.Embedding(10, 32)


@pytest.fixture
def tiny() -> TinyNet:
    torch.manual_seed(SEED)
    return TinyNet()


class TestW4Census:
    def test_it_excludes_weights_whose_quantization_axis_is_not_divisible(self, tiny) -> None:
        census = measure.census_w4_targets(tiny, measure.W4_ROLES["all"])
        assert sorted(census.excluded) == ["depthwise", "odd_linear"]
        assert sorted(census.eligible) == ["conv", "emb", "linear"]

    def test_it_counts_the_excluded_elements(self, tiny) -> None:
        census = measure.census_w4_targets(tiny, measure.W4_ROLES["all"])
        assert census.excluded_elements == 4 * 48 + 6 * 1 * 3

    def test_it_counts_the_eligible_targets_per_role(self, tiny) -> None:
        census = measure.census_w4_targets(tiny, measure.W4_ROLES["all"])
        by_role = {role: (c.modules, c.channels, c.elements) for role, c in census.by_role.items()}
        assert by_role == {
            "Conv1d": (1, 6, 6 * 4 * 8),
            "Embedding": (1, 10, 10 * 32),
            "Linear": (1, 8, 8 * 64),
        }
        assert census.counts.elements == 192 + 320 + 512
        assert census.counts.groups == (192 + 320 + 512) // measure.W4_GROUP_SIZE

    def test_it_narrows_the_targets_to_linear_for_the_shippable_role(self, tiny) -> None:
        census = measure.census_w4_targets(tiny, measure.W4_ROLES["linear"])
        assert census.eligible == ("linear",)
        assert census.excluded == ("odd_linear",)

    def test_it_folds_the_exclusion_into_the_core_include_predicate(self, tiny) -> None:
        include = measure.census_w4_targets(tiny, measure.W4_ROLES["all"]).include()
        assert [
            name for name in ("linear", "odd_linear", "conv", "depthwise") if include(name)
        ] == [
            "linear",
            "conv",
        ]


class TestW4SizeProjection:
    #: 65536 要素 = 2048 group（g32）— bpw が割り切れる大きさに採る。
    COUNTS = measure.TargetCounts(modules=2, channels=8, elements=65536)

    @pytest.mark.parametrize(
        ("method", "expected_bpw"),
        [("rtn-i4-g32", 5.0), ("nf4", 5.0), ("mxfp4", 4.25)],
    )
    def test_it_projects_the_effective_bits_per_weight(self, method, expected_bpw) -> None:
        projection = measure.SizeProjection(
            counts=self.COUNTS,
            bits=measure.W4_METHODS[method].projected_bits(self.COUNTS),
            formula=measure.W4_METHODS[method].formula,
        )
        assert projection.bits_per_weight == expected_bpw

    def test_it_charges_the_shared_codebook_to_kmeans(self) -> None:
        """表の代金は品質と同じ表に載る — k-means だけ全体で 1 枚ぶん重い。"""
        rtn = measure.W4_METHODS["rtn-i4-g32"].projected_bits(self.COUNTS)
        kmeans = measure.W4_METHODS["kmeans:shared"].projected_bits(self.COUNTS)
        assert kmeans - rtn == measure.CODEBOOK_BITS

    def test_it_reports_mib_against_the_f32_baseline(self) -> None:
        projection = measure.SizeProjection(
            counts=self.COUNTS,
            bits=measure.W4_METHODS["rtn-i4-g32"].projected_bits(self.COUNTS),
            formula="",
        )
        assert projection.f32_mib == 65536 * 4 / 1024**2
        assert projection.projected_mib == projection.f32_mib * 5.0 / 32


class TestW4ProjectionRows:
    def test_it_marks_which_rows_have_a_quality_measurement(self) -> None:
        """どの行に測定が付くかは CONFIGS から引く（別表に書くと片方だけ古くなる）。"""
        counts = dict.fromkeys(
            measure.PROJECTION_TARGETS, measure.TargetCounts(modules=1, channels=1, elements=32)
        )
        measured = {
            (row["target"], row["method"]): row["measured_by"]
            for row in measure.build_projections(counts)
        }
        assert measured[("net_g:all", "rtn-i4-g32")] == "w4-rtn"
        assert measured[("net_g:all", "kmeans:shared")] == "w4-kmeans"
        assert measured[("bert:linear", "nf4")] == "bert-w4-nf4"
        # net_g の linear 限定は**サイズだけの席**（品質は測らない — 対象が僅少）。
        assert all(measured[("net_g:linear", name)] is None for name in measure.W4_METHODS)
        # BERT は配布対応形の 2 方式だけを測る。
        assert measured[("bert:linear", "mxfp4")] is None


class TestW4Methods:
    def test_it_rounds_only_the_eligible_weights(self, tiny) -> None:
        census = measure.census_w4_targets(tiny, measure.W4_ROLES["all"])
        untouched = {name: getattr(tiny, name).weight.clone() for name in census.excluded}
        report = measure.W4_METHODS["rtn-i4-g32"].apply(
            tiny, measure.W4_ROLES["all"], census.include()
        )
        assert (report.modules, report.elements) == (
            census.counts.modules,
            census.counts.elements,
        )
        for name, before in untouched.items():
            assert torch.equal(getattr(tiny, name).weight, before)

    def test_it_moves_the_eligible_weights(self, tiny) -> None:
        census = measure.census_w4_targets(tiny, measure.W4_ROLES["all"])
        before = {name: getattr(tiny, name).weight.clone() for name in census.eligible}
        measure.W4_METHODS["rtn-i4-g32"].apply(tiny, measure.W4_ROLES["all"], census.include())
        for name, weight in before.items():
            assert not torch.equal(getattr(tiny, name).weight, weight), name

    def test_it_keeps_the_four_methods_distinct(self) -> None:
        """方式の呼び分け漏れ（同じ丸めを 2 度当てる）の検出 — 台本のゲートと同型。"""
        rounded: dict[str, torch.Tensor] = {}
        for name in measure.W4_METHODS:
            torch.manual_seed(SEED)
            model = TinyNet()
            census = measure.census_w4_targets(model, measure.W4_ROLES["all"])
            measure.W4_METHODS[name].apply(model, measure.W4_ROLES["all"], census.include())
            rounded[name] = model.linear.weight.clone()
        names = list(rounded)
        for index, left in enumerate(names):
            for right in names[index + 1 :]:
                assert not torch.equal(rounded[left], rounded[right]), f"{left} と {right}"


def _write_dist(root: Path) -> None:
    """配布形の最小形（linear の重み 1 本 + embedding 1 本 + それぞれの scale）を書く。"""
    root.mkdir(parents=True, exist_ok=True)
    save_file(
        {
            "enc_p.style_proj.weight": torch.ones(8, 64, dtype=torch.int8),
            measure.DIST_SCALE_PREFIX + "enc_p.style_proj.weight": torch.ones(8, 1),
            "enc_p.emb.weight": torch.ones(4, 32, dtype=torch.int8),
            measure.DIST_SCALE_PREFIX + "enc_p.emb.weight": torch.ones(4, 1),
        },
        str(root / "model.i8.safetensors"),
    )


class TestDistributionProjection:
    LINEAR = frozenset({"enc_p.style_proj.weight"})

    def test_it_projects_only_the_linear_weight_slots(self, tmp_path: Path) -> None:
        _write_dist(tmp_path / "F1" / "front")
        projection = measure.project_distribution(tmp_path, self.LINEAR)
        group = projection["groups"]["話者ごとの net_g"]
        assert group["tensors"] == 1
        # i8 = 8·64 + 4·8 バイト / i4 = 8·64/2 + 4·8·(64/32) バイト
        assert group["current_bytes"] == 8 * 64 + 4 * 8
        assert group["projected_bytes"] == 8 * 64 // 2 + 4 * 8 * (64 // 32)
        assert projection["delta_bytes"] == group["delta_bytes"]

    def test_it_reports_the_rank2_int8_tensors_it_did_not_match(self, tmp_path: Path) -> None:
        """rank 2 の I8 という形だけで引くと embedding が混ざる — 見送った本数を出す。"""
        _write_dist(tmp_path / "F1" / "front")
        assert measure.project_distribution(tmp_path, self.LINEAR)["rank2_i8_not_linear"] == 1

    def test_it_separates_the_shared_subtree_from_the_per_speaker_one(self, tmp_path: Path) -> None:
        _write_dist(tmp_path / "F1" / "front")
        _write_dist(tmp_path / "shared" / "text_encoder")
        projection = measure.project_distribution(tmp_path, self.LINEAR)
        assert sorted(projection["groups"]) == [
            "shared（DeBERTa text_encoder）",
            "話者ごとの net_g",
        ]
        assert projection["files"] == 2

    def test_it_measures_the_shrink_against_the_real_file_bytes(self, tmp_path: Path) -> None:
        _write_dist(tmp_path / "F1" / "front")
        projection = measure.project_distribution(tmp_path, self.LINEAR)
        total = sum(path.stat().st_size for path in tmp_path.rglob("*") if path.is_file())
        assert projection["total_bytes"] == total
        assert projection["shrink_of_total"] == projection["delta_bytes"] / total

    def test_it_fails_loudly_when_the_scale_tensor_is_missing(self, tmp_path: Path) -> None:
        target = tmp_path / "F1" / "front"
        target.mkdir(parents=True)
        save_file(
            {"enc_p.style_proj.weight": torch.ones(8, 64, dtype=torch.int8)},
            str(target / "model.i8.safetensors"),
        )
        with pytest.raises(AssertionError, match="scale テンソルが無い"):
            measure.project_distribution(tmp_path, self.LINEAR)


class TestConfigSelection:
    def test_it_runs_every_config_by_default(self) -> None:
        assert measure.selected_configs(None) == tuple(measure.CONFIGS)

    def test_it_always_keeps_f32_as_the_snr_baseline(self) -> None:
        assert measure.selected_configs("w4-rtn") == ("f32", "w4-rtn")

    def test_it_keeps_the_declaration_order(self) -> None:
        assert measure.selected_configs("w4-nf4,w8") == ("f32", "w8", "w4-nf4")

    def test_it_rejects_an_unknown_config(self) -> None:
        with pytest.raises(SystemExit, match="未知の構成"):
            measure.selected_configs("w4-int3")


# ---- 校正付き丸め（波 J-2）--------------------------------------------------


class TinyLayer(nn.Module):
    """`DebertaV2Layer` 相当の stage 本体（mask の効きは**乗算**で観測できる形にする）。

    返り値を `(hidden, None)` の tuple にするのは本物と同じ — {@link measure.EncoderStage} が
    先頭要素を選ぶことの検査になる。
    """

    def __init__(self, features: int) -> None:
        super().__init__()
        self.query_proj = nn.Linear(features, features, bias=False)
        self.dense = nn.Linear(features, features, bias=False)

    def forward(
        self,
        hidden: torch.Tensor,
        attention_mask: torch.Tensor,
        relative_pos: torch.Tensor | None = None,
        output_attentions: bool = False,
    ) -> tuple[torch.Tensor, None]:
        return (self.dense(torch.tanh(self.query_proj(hidden * attention_mask))), None)


class TinyConv(nn.Module):
    """`ConvLayer` 相当 — **`nn.Linear` を 1 本も持たない**（走査の対象集合に効かない）。"""

    def __init__(self, features: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(features, features, 3, padding=1)

    def forward(
        self, hidden: torch.Tensor, residual: torch.Tensor, input_mask: torch.Tensor
    ) -> torch.Tensor:
        out = self.conv(hidden.transpose(1, 2)).transpose(1, 2)
        return (residual + torch.tanh(out)) * input_mask.unsqueeze(-1)


class TinyEncoder(nn.Module):
    """`DebertaV2Encoder` 相当 — mask を**位置引数**で渡し、先頭層の出力にだけ conv を乗せる。"""

    def __init__(self, features: int, layers: int) -> None:
        super().__init__()
        self.layer = nn.ModuleList(TinyLayer(features) for _ in range(layers))
        self.conv = TinyConv(features)

    def forward(
        self, hidden: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        input_mask = attention_mask.to(hidden.dtype)
        expanded = input_mask.unsqueeze(-1)
        relative_pos = torch.arange(int(hidden.shape[1]), dtype=torch.int64)
        states = [hidden]
        for index, layer in enumerate(self.layer):
            output = layer(hidden, expanded, relative_pos=relative_pos, output_attentions=False)[0]
            if index == 0:
                output = self.conv(hidden, output, input_mask)
            hidden = output
            states.append(hidden)
        return tuple(states)


class TinyBert(nn.Module):
    """`DebertaV2Model` 相当 — `measure.bert_stages` が辿る `encoder.layer` / `encoder.conv`。

    forward は `output_hidden_states=True` の本物と同じく**全 hidden の列**を返す
    （stage 分解の等価性を層番号で突き合わせるため）。
    """

    def __init__(self, vocab: int = 16, features: int = 32, layers: int = 4) -> None:
        super().__init__()
        self.embeddings = nn.Embedding(vocab, features)
        self.encoder = TinyEncoder(features, layers)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        return self.encoder(self.embeddings(input_ids), attention_mask)


class SilentBert(TinyBert):
    """先頭 stage を**呼ばない** forward（Catcher の fail loudly を踏ませる形）。"""

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        return (self.embeddings(input_ids),)


def tiny_calib_inputs(count: int = 3) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    """長さの違う `(input_ids, attention_mask)`（校正入力の代役 — tokenizer は通さない）。"""
    return tuple(
        (
            torch.arange(3 + index, dtype=torch.int64).unsqueeze(0) % 16,
            torch.ones(1, 3 + index, dtype=torch.int64),
        )
        for index in range(count)
    )


def tiny_rig(bert: TinyBert, inputs) -> measure.CalibRig:
    """`build_calib_rig` の資産に依らない版（dump の meta が要る分離検査だけを抜く）。"""
    stages = measure.bert_stages(bert)
    scan, counts = measure.calib_targets(stages)
    return measure.CalibRig(
        stages=stages,
        scan=scan,
        counts=counts,
        batches=measure.capture_stage_batches(bert, inputs),
    )


class TestCalibConfigs:
    def test_the_calibrated_configs_are_selectable(self) -> None:
        assert measure.selected_configs("bert-gptq-rtn,bert-gptq-kmeans") == (
            "f32",
            "bert-gptq-rtn",
            "bert-gptq-kmeans",
        )

    def test_they_keep_net_g_in_f32(self) -> None:
        """BERT だけを振る構成 — 生成ネット側は 1 本も丸めない（既存 bert-w4-* と同じ契約）。"""
        for name in measure.CALIB_CONFIGS:
            recipe = measure.CONFIGS[f"bert-{name}"]
            assert (recipe.weight, recipe.act, recipe.scope) == (None, None, ())

    def test_the_group_size_is_pinned_for_every_calib_config(self) -> None:
        """校正は格子を変えない — g 軸を振らないのは方式グリッドと同文。"""
        assert all(
            config.grid.group_size == measure.CALIB_GROUP_SIZE
            for config in measure.CALIB_CONFIGS.values()
        )

    def test_no_calib_config_fits_its_table_on_a_subsample(self) -> None:
        """表は常に全量から fit（部分標本の逃げ道は net_g 側にも無い）。"""
        assert all(config.grid.fit_stride == 1 for config in measure.CALIB_CONFIGS.values())

    def test_the_label_names_both_the_method_and_the_grid(self) -> None:
        """`w4` 節の方式名は方式と格納グリッドの組（片方だけだと行が読めない）。"""
        labels = {name: config.label for name, config in measure.CALIB_CONFIGS.items()}

        assert labels == {
            "gptq-rtn": "gptq/rtn",
            "gptq-nf4": "gptq/nf4",
            "gptq-kmeans": "gptq/kmeans_shared",
        }

    def test_the_group_scale_grids_project_five_bits(self) -> None:
        counts = measure.TargetCounts(modules=2, channels=8, elements=65536)

        assert measure.group_scale_bits(counts) / counts.elements == 5.0

    def test_the_calibrated_codebook_costs_one_table_per_layer(self) -> None:
        """core の `kmeans_shared` は**層内**の表 — 全体 1 枚の `kmeans:shared` とは式が違う。"""
        counts = measure.TargetCounts(modules=2, channels=8, elements=65536)

        difference = measure.layer_table_bits(counts) - measure.group_scale_bits(counts)

        assert difference == measure.CODEBOOK_BITS * counts.modules


class TestBertVariant:
    def test_it_keys_the_plain_and_calibrated_roundings_in_one_space(self) -> None:
        assert measure.bert_variant(measure.CONFIGS["bert-w4-nf4"]) == "nf4"
        assert measure.bert_variant(measure.CONFIGS["bert-gptq-nf4"]) == "gptq-nf4"
        assert measure.bert_variant(measure.CONFIGS["f32"]) is None

    def test_the_two_name_spaces_do_not_collide(self) -> None:
        """交わると 1 本の鍵で引けなくなる（特徴のキャッシュが黙って混ざる）。"""
        assert not set(measure.W4_METHODS) & set(measure.CALIB_CONFIGS)

    def test_a_recipe_that_asks_for_both_is_rejected(self) -> None:
        recipe = measure.Recipe(None, None, scope=(), bert_method="nf4", bert_calib="gptq-nf4")

        with pytest.raises(AssertionError, match="どちらか一方"):
            measure.bert_variant(recipe)


class TestCalibCorpus:
    def test_it_takes_the_head_of_the_corpus(self) -> None:
        assert measure.calib_corpus(4) == measure.CALIB_TEXTS[:4]
        assert measure.calib_corpus(None) == measure.CALIB_TEXTS

    def test_it_rejects_a_non_positive_limit(self) -> None:
        with pytest.raises(ValueError, match="1 以上"):
            measure.calib_corpus(0)


class TestCalibDisjoint:
    """MUST: 校正と評価の分離（重なると「校正で見た文を出せたか」を測る数になる）。"""

    EVALUATED = ("こんにちは、これはテストです。", "こんにちは,これはテストです.")

    def test_the_shipped_corpus_is_separate_from_the_evaluation_text(self) -> None:
        measure.assert_calib_disjoint(measure.CALIB_TEXTS, self.EVALUATED)

    def test_it_catches_a_calib_text_that_quotes_the_evaluation_text(self) -> None:
        quoting = ("昨日、" + self.EVALUATED[0] + "と言われた。",)

        with pytest.raises(AssertionError, match="重なっている"):
            measure.assert_calib_disjoint(quoting, self.EVALUATED)

    def test_it_catches_a_calib_text_quoted_by_the_evaluation_text(self) -> None:
        """部分一致は片方向では見つからない（`in` を両向きに見る）。"""
        quoted = ("これはテストです。",)

        with pytest.raises(AssertionError, match="重なっている"):
            measure.assert_calib_disjoint(quoted, self.EVALUATED)


class TestBertStages:
    def test_it_stops_at_the_layer_the_feature_is_taken_from(self) -> None:
        """末尾の層は特徴に 1bit も効かないので stage に入れない（`bert_feature_of` の位置）。"""
        bert = TinyBert(layers=4)

        stages = measure.bert_stages(bert)

        assert len(stages) == 4 - (measure.demo.BERT_HIDDEN_FROM_END - 1)

    def test_only_the_first_stage_carries_the_conv(self) -> None:
        """先頭層だけ出力に ConvLayer が乗る（`DebertaV2Encoder` の `i == 0`）。"""
        stages = measure.bert_stages(TinyBert())

        assert [stage.conv is not None for _prefix, stage in stages] == [True, False]

    def test_the_scan_uses_model_wide_fqns(self) -> None:
        """台帳のキーを `Int4Report` と同じ FQN 空間へ揃えるための接頭辞（core の契約）。"""
        scan, counts = measure.calib_targets(measure.bert_stages(TinyBert()))

        assert sorted(scan) == [
            "encoder.layer.0.dense.weight",
            "encoder.layer.0.query_proj.weight",
            "encoder.layer.1.dense.weight",
            "encoder.layer.1.query_proj.weight",
        ]
        assert counts.modules == 4

    def test_the_conv_is_not_in_the_scan(self) -> None:
        """ConvLayer は子として登録されるが `nn.Linear` を持たないので対象集合を動かさない。"""
        scan, _counts = measure.calib_targets(measure.bert_stages(TinyBert()))

        assert not any("conv" in fqn for fqn in scan)

    def test_a_weight_whose_axis_is_not_divisible_fails_loudly(self) -> None:
        """net_g 側の census と違って**外さない** — 黙って痩せる道を校正側へ作らない。"""
        bert = TinyBert(features=32)
        bert.encoder.layer[0].dense = nn.Linear(48, 48, bias=False)

        with pytest.raises(AssertionError, match="割り切れない"):
            measure.calib_targets(measure.bert_stages(bert))


class TestCatcher:
    def test_it_captures_one_batch_per_calibration_text(self) -> None:
        bert = TinyBert()

        batches = measure.capture_stage_batches(bert, tiny_calib_inputs(3))

        assert len(batches) == 3
        assert [int(args[0].shape[1]) for args, _kwargs in batches] == [3, 4, 5]

    def test_it_moves_the_mask_from_positional_to_keyword(self) -> None:
        """`calibrate_stages` は次 stage へ位置引数を 1 つしか渡さない（core の駆動）。"""
        bert = TinyBert()

        (args, kwargs), *_ = measure.capture_stage_batches(bert, tiny_calib_inputs(1))

        assert len(args) == 1
        assert sorted(kwargs) == [
            "attention_mask",
            measure.INPUT_MASK_KWARG,
            "output_attentions",
            "relative_pos",
        ]

    def test_it_captures_the_two_dimensional_mask_the_conv_needs(self) -> None:
        """層へ渡る mask と ConvLayer へ渡る mask は別物（写すと片方だけ仕様から外れる）。"""
        bert = TinyBert()

        (_args, kwargs), *_ = measure.capture_stage_batches(bert, tiny_calib_inputs(1))

        assert kwargs[measure.INPUT_MASK_KWARG].dim() == 2
        assert kwargs["attention_mask"].dim() == 3

    def test_the_hooks_are_removed_even_though_the_forward_was_aborted(self) -> None:
        """番兵で打ち切っても後始末は済む（残ると以後の forward が全部落ちる）。"""
        bert = TinyBert()

        measure.capture_stage_batches(bert, tiny_calib_inputs(1))

        assert not bert.encoder.layer[0]._forward_pre_hooks
        assert not bert.encoder.conv._forward_pre_hooks
        with torch.no_grad():
            bert(*tiny_calib_inputs(1)[0])

    def test_a_forward_that_never_reaches_the_first_stage_fails_loudly(self) -> None:
        """MUST: 素通りを黙って通すと「校正入力ゼロ」の診断まで見えない。"""
        with pytest.raises(AssertionError, match="揃わなかった"):
            measure.capture_stage_batches(SilentBert(), tiny_calib_inputs(1))


class TestStageChain:
    def test_the_stage_chain_reproduces_the_encoder(self) -> None:
        """stage 分解が本物の encoder と**同じ hidden** を作る（先頭 conv の落としの検出器）。"""
        torch.manual_seed(SEED)
        bert = TinyBert()
        inputs = tiny_calib_inputs(1)
        with torch.no_grad():
            states = bert(*inputs[0])
        rig = tiny_rig(bert, inputs)
        args, kwargs = rig.batches[0]

        hidden = args[0]
        with torch.no_grad():
            for _prefix, stage in rig.stages:
                hidden = stage(hidden, **kwargs)

        assert torch.equal(hidden, states[len(rig.stages)])

    def test_dropping_the_conv_moves_the_first_stage(self) -> None:
        """上の検出器の故障注入 — conv を落とすと実際に別の hidden になる。"""
        torch.manual_seed(SEED)
        bert = TinyBert()
        inputs = tiny_calib_inputs(1)
        with torch.no_grad():
            states = bert(*inputs[0])
        args, kwargs = measure.capture_stage_batches(bert, inputs)[0]
        bare = measure.EncoderStage(0, bert.encoder.layer[0], None)

        with torch.no_grad():
            output = bare(args[0], **kwargs)

        assert not torch.equal(output, states[1])


class TestCalibrationRun:
    def test_it_rounds_every_scanned_linear(self) -> None:
        torch.manual_seed(SEED)
        bert = TinyBert()
        rig = tiny_rig(bert, tiny_calib_inputs(2))
        before = {fqn: weight.detach().clone() for fqn, weight in rig.scan.items()}

        report = measure.apply_calib(measure.CALIB_CONFIGS["gptq-rtn"], rig)
        measure.assert_calib_covers_scan(report, rig.scan, "gptq-rtn")

        assert report.modules == len(rig.scan)
        assert all(not torch.equal(rig.scan[fqn], value) for fqn, value in before.items())

    def test_it_leaves_the_layers_beyond_the_feature_untouched(self) -> None:
        """stage に入らない層は f32 のまま（対象集合が `bert:linear` の部分集合である証拠）。"""
        torch.manual_seed(SEED)
        bert = TinyBert(layers=4)
        beyond = bert.encoder.layer[3].query_proj.weight.detach().clone()
        rig = tiny_rig(bert, tiny_calib_inputs(2))

        measure.apply_calib(measure.CALIB_CONFIGS["gptq-rtn"], rig)

        assert torch.equal(bert.encoder.layer[3].query_proj.weight, beyond)

    def test_every_calib_config_runs_end_to_end(self) -> None:
        """3 本とも core の駆動へ通る（方式と格納グリッドの組が全部生きている）。"""
        for name, config in measure.CALIB_CONFIGS.items():
            torch.manual_seed(SEED)
            bert = TinyBert()
            rig = tiny_rig(bert, tiny_calib_inputs(2))

            report = measure.apply_calib(config, rig)
            measure.assert_calib_covers_scan(report, rig.scan, name)

            assert report.method == config.method
            assert report.grid == config.grid.kind

    def test_the_three_grids_round_to_different_values(self) -> None:
        """格納グリッドの呼び分け漏れの検出（同じ格子を 2 度当てるとビット一致する）。"""
        rounded: dict[str, torch.Tensor] = {}
        for name, config in measure.CALIB_CONFIGS.items():
            torch.manual_seed(SEED)
            bert = TinyBert()
            rig = tiny_rig(bert, tiny_calib_inputs(2))
            measure.apply_calib(config, rig)
            rounded[name] = bert.encoder.layer[0].query_proj.weight.clone()
        names = list(rounded)
        for index, left in enumerate(names):
            for right in names[index + 1 :]:
                assert not torch.equal(rounded[left], rounded[right]), f"{left} と {right}"

    def test_a_scan_the_calibration_missed_fails_loudly(self) -> None:
        """恒真化の遮断 — 丸め漏れは SNR / LSD の**良い側**へ出るので数字から読めない。"""
        torch.manual_seed(SEED)
        bert = TinyBert()
        rig = tiny_rig(bert, tiny_calib_inputs(2))
        report = measure.apply_calib(measure.CALIB_CONFIGS["gptq-rtn"], rig)
        widened = {**rig.scan, "encoder.layer.9.query_proj.weight": torch.zeros(1)}

        with pytest.raises(AssertionError, match="一致しない"):
            measure.assert_calib_covers_scan(report, widened, "gptq-rtn")
