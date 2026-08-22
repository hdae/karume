"""`sbv2.measure_quant` の w4 まわりの単体テスト（**資産の要らないロジックだけ**）。

実重み・dump を使う測定そのものは台本の検証ゲート（`measure_quant.run_gates`）が受け持つ。
ここで切り分けるのは、資産が無くても成り立つ 4 つの部品:

    ① 対象集合の census（group 長で割り切れない重みの除外と `include` への畳み込み）
    ② サイズ試算の式（方式ごとの実効 bpw）
    ③ 方式の呼び分け（4 種が互いに違う値へ丸め、除外した重みには 1 要素も触らない）
    ④ 配布形との突合（テンソル名で linear の重みスロットだけを拾う）
    ⑤ group 長 g の CLI 上書き（`--w4-group-size` が適格判定と丸めの**両方**へ届くこと）

丸めの実装そのもの（i4 / NF4 / MXFP4 / k-means の数値）は core 側のテストが持つ — ここが
測るのは「台本が core をどう呼び分けているか」だけで、写経した式は 1 つも無い。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from sbv2 import measure_quant as measure

#: 合成モデルの初期化 seed（方式の差を見るテストは値が退化していないことが前提）。
SEED = 20260819

#: 既定の group 長（`--w4-group-size` を指定しないときの g）。g を振るテストはここからの差で読む。
DEFAULT_G = measure.DEFAULT_GROUP_SIZE


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
        census = measure.census_w4_targets(tiny, measure.W4_ROLES["all"], DEFAULT_G)
        assert sorted(census.excluded) == ["depthwise", "odd_linear"]
        assert sorted(census.eligible) == ["conv", "emb", "linear"]

    def test_it_counts_the_excluded_elements(self, tiny) -> None:
        census = measure.census_w4_targets(tiny, measure.W4_ROLES["all"], DEFAULT_G)
        assert census.excluded_elements == 4 * 48 + 6 * 1 * 3

    def test_it_counts_the_eligible_targets_per_role(self, tiny) -> None:
        census = measure.census_w4_targets(tiny, measure.W4_ROLES["all"], DEFAULT_G)
        by_role = {role: (c.modules, c.channels, c.elements) for role, c in census.by_role.items()}
        assert by_role == {
            "Conv1d": (1, 6, 6 * 4 * 8),
            "Embedding": (1, 10, 10 * 32),
            "Linear": (1, 8, 8 * 64),
        }
        assert census.counts.elements == 192 + 320 + 512
        assert census.counts.groups == (192 + 320 + 512) // DEFAULT_G

    def test_it_narrows_the_targets_to_linear_for_the_shippable_role(self, tiny) -> None:
        census = measure.census_w4_targets(tiny, measure.W4_ROLES["linear"], DEFAULT_G)
        assert census.eligible == ("linear",)
        assert census.excluded == ("odd_linear",)

    def test_it_folds_the_exclusion_into_the_core_include_predicate(self, tiny) -> None:
        include = measure.census_w4_targets(tiny, measure.W4_ROLES["all"], DEFAULT_G).include()
        assert [
            name for name in ("linear", "odd_linear", "conv", "depthwise") if include(name)
        ] == [
            "linear",
            "conv",
        ]


class TestW4SizeProjection:
    #: 65536 要素 = 2048 group（g32）— bpw が割り切れる大きさに採る。
    COUNTS = measure.TargetCounts(modules=2, channels=8, elements=65536, group_size=DEFAULT_G)
    METHODS = measure.w4_methods(DEFAULT_G)

    @pytest.mark.parametrize(
        ("kind", "expected_bpw"),
        [(measure.RTN_KIND, 5.0), ("nf4", 5.0), ("mxfp4", 4.25)],
    )
    def test_it_projects_the_effective_bits_per_weight(self, kind, expected_bpw) -> None:
        projection = measure.SizeProjection(
            counts=self.COUNTS,
            bits=self.METHODS[kind].projected_bits(self.COUNTS),
            formula=self.METHODS[kind].formula,
        )
        assert projection.bits_per_weight == expected_bpw

    def test_it_charges_the_shared_codebook_to_kmeans(self) -> None:
        """表の代金は品質と同じ表に載る — k-means だけ全体で 1 枚ぶん重い。"""
        rtn = self.METHODS[measure.RTN_KIND].projected_bits(self.COUNTS)
        kmeans = self.METHODS["kmeans:shared"].projected_bits(self.COUNTS)
        assert kmeans - rtn == measure.CODEBOOK_BITS

    def test_it_reports_mib_against_the_f32_baseline(self) -> None:
        projection = measure.SizeProjection(
            counts=self.COUNTS,
            bits=self.METHODS[measure.RTN_KIND].projected_bits(self.COUNTS),
            formula="",
        )
        assert projection.f32_mib == 65536 * 4 / 1024**2
        assert projection.projected_mib == projection.f32_mib * 5.0 / 32


class TestW4ProjectionRows:
    def test_it_marks_which_rows_have_a_quality_measurement(self) -> None:
        """どの行に測定が付くかは CONFIGS から引く（別表に書くと片方だけ古くなる）。"""
        methods = measure.w4_methods(DEFAULT_G)
        counts = dict.fromkeys(
            measure.PROJECTION_TARGETS,
            measure.TargetCounts(modules=1, channels=1, elements=32, group_size=DEFAULT_G),
        )
        measured = {
            (row["target"], row["method"]): row["measured_by"]
            for row in measure.build_projections(counts, methods)
        }
        assert measured[("net_g:all", "rtn-i4-g32")] == "w4-rtn"
        assert measured[("net_g:all", "kmeans:shared")] == "w4-kmeans"
        assert measured[("bert:linear", "nf4")] == "bert-w4-nf4"
        # net_g の linear 限定は**サイズだけの席**（品質は測らない — 対象が僅少）。
        assert all(measured[("net_g:linear", method.name)] is None for method in methods.values())
        # BERT は配布対応形の 2 方式だけを測る。
        assert measured[("bert:linear", "mxfp4")] is None


class TestW4Methods:
    METHODS = measure.w4_methods(DEFAULT_G)

    def test_it_rounds_only_the_eligible_weights(self, tiny) -> None:
        census = measure.census_w4_targets(tiny, measure.W4_ROLES["all"], DEFAULT_G)
        untouched = {name: getattr(tiny, name).weight.clone() for name in census.excluded}
        report = self.METHODS[measure.RTN_KIND].apply(
            tiny, measure.W4_ROLES["all"], census.include()
        )
        assert (report.modules, report.elements) == (
            census.counts.modules,
            census.counts.elements,
        )
        for name, before in untouched.items():
            assert torch.equal(getattr(tiny, name).weight, before)

    def test_it_moves_the_eligible_weights(self, tiny) -> None:
        census = measure.census_w4_targets(tiny, measure.W4_ROLES["all"], DEFAULT_G)
        before = {name: getattr(tiny, name).weight.clone() for name in census.eligible}
        self.METHODS[measure.RTN_KIND].apply(tiny, measure.W4_ROLES["all"], census.include())
        for name, weight in before.items():
            assert not torch.equal(getattr(tiny, name).weight, weight), name

    def test_it_keeps_the_four_methods_distinct(self) -> None:
        """方式の呼び分け漏れ（同じ丸めを 2 度当てる）の検出 — 台本のゲートと同型。"""
        rounded: dict[str, torch.Tensor] = {}
        for name in self.METHODS:
            torch.manual_seed(SEED)
            model = TinyNet()
            census = measure.census_w4_targets(model, measure.W4_ROLES["all"], DEFAULT_G)
            self.METHODS[name].apply(model, measure.W4_ROLES["all"], census.include())
            rounded[name] = model.linear.weight.clone()
        names = list(rounded)
        for index, left in enumerate(names):
            for right in names[index + 1 :]:
                assert not torch.equal(rounded[left], rounded[right]), f"{left} と {right}"


class TestW4GroupSizeOption:
    """`--w4-group-size`（波 J-3 の g 軸）の受理と既定。"""

    def test_the_default_is_the_core_storage_default(self) -> None:
        """既定は 32 — 指定しない実行が過去の研究記録と同じ格子で走ること。"""
        assert DEFAULT_G == 32
        assert measure.build_parser().parse_args([]).w4_group_size == 32

    @pytest.mark.parametrize("value", [16, 32, 64, 128])
    def test_it_accepts_a_power_of_two_from_sixteen(self, value) -> None:
        parsed = measure.build_parser().parse_args(["--w4-group-size", str(value)])

        assert parsed.w4_group_size == value

    @pytest.mark.parametrize("raw", ["24", "8", "0", "-32", "g32"])
    def test_it_rejects_a_group_core_cannot_store(self, raw) -> None:
        """MUST: 受理集合の外は argparse で落とす（測り終えてから emit が撥ねると 1 本丸損）。"""
        with pytest.raises(SystemExit):
            measure.build_parser().parse_args(["--w4-group-size", raw])

    def test_the_rejection_names_the_storage_rule(self) -> None:
        with pytest.raises(argparse.ArgumentTypeError, match="2 冪かつ"):
            measure.parse_w4_group_size("24")


class TestW4MethodNames:
    def test_the_rtn_name_carries_the_group_size(self) -> None:
        """既定 32 では従来と同じ綴り（過去の研究記録との突合を壊さない）。"""
        assert measure.rtn_method_name(DEFAULT_G) == "rtn-i4-g32"
        assert measure.rtn_method_name(16) == "rtn-i4-g16"

    def test_the_table_keys_do_not_move_with_g(self) -> None:
        """鍵が g で動くと構成表（`w4-rtn` の指す先）が g ごとに引けなくなる。"""
        assert list(measure.w4_methods(64)) == list(measure.w4_methods(DEFAULT_G))
        assert measure.CONFIGS["w4-rtn"].method in measure.w4_methods(64)

    def test_the_formula_states_the_group_and_its_bits_per_weight(self) -> None:
        """式は投影表へそのまま載る — g を変えて bpw が据え置きだと表が読めない。"""
        assert (
            measure.w4_methods(DEFAULT_G)[measure.RTN_KIND].formula
            == "4bit + g32 f32 scale = 5.0 bpw"
        )
        assert measure.w4_methods(64)[measure.RTN_KIND].formula == "4bit + g64 f32 scale = 4.5 bpw"


class TestW4GroupSizeReach:
    """指定した g が**適格判定と丸めの両方**へ届くこと（片方だけ動けば黙って割れる）。"""

    def test_a_smaller_group_widens_the_eligible_set(self, tiny) -> None:
        """g16 では受容野 48 の linear も割り切れる（適格判定が g を見ている証拠）。"""
        census = measure.census_w4_targets(tiny, measure.W4_ROLES["all"], 16)

        assert sorted(census.eligible) == ["conv", "emb", "linear", "odd_linear"]
        assert census.excluded == ("depthwise",)

    def test_a_larger_group_narrows_it(self, tiny) -> None:
        census = measure.census_w4_targets(tiny, measure.W4_ROLES["all"], 64)

        assert census.eligible == ("linear",)
        assert sorted(census.excluded) == ["conv", "depthwise", "emb", "odd_linear"]

    def test_the_counts_carry_the_group_they_were_counted_with(self, tiny) -> None:
        """計数が g を担ぐので、bpw の投影が別の g で行われることが起きない。"""
        census = measure.census_w4_targets(tiny, measure.W4_ROLES["all"], 16)

        assert census.counts.group_size == 16
        assert census.counts.groups == census.counts.elements // 16

    def test_counts_from_different_groups_cannot_be_summed(self) -> None:
        """front + voice のような合算で g が混ざったら fail loudly（黙って片方の g になる穴）。"""
        left = measure.TargetCounts(modules=1, channels=1, elements=64, group_size=16)
        right = measure.TargetCounts(modules=1, channels=1, elements=64, group_size=32)

        with pytest.raises(AssertionError, match="group 長の違う計数"):
            _ = left + right

    @pytest.mark.parametrize("group_size", [16, 64])
    @pytest.mark.parametrize("kind", list(measure.w4_methods(measure.DEFAULT_GROUP_SIZE)))
    def test_the_rounding_covers_exactly_the_eligible_set(self, kind, group_size) -> None:
        """適格判定と丸めが**同じ g** を見る（4 方式とも）。

        方式側が 32 を握ったままだと、g16 で適格になった受容野 48 の linear を core が端数
        group で撥ねる — 値ではなく例外で出るので、この 1 本が渡し忘れの門になる。
        """
        torch.manual_seed(SEED)
        model = TinyNet()
        census = measure.census_w4_targets(model, measure.W4_ROLES["all"], group_size)

        report = measure.w4_methods(group_size)[kind].apply(
            model, measure.W4_ROLES["all"], census.include()
        )

        assert (report.modules, report.elements) == (
            census.counts.modules,
            census.counts.elements,
        )

    def test_the_rtn_report_states_the_group_it_used(self) -> None:
        """RTN の計数報告は g を逐語で出す（`[bert:…]` の行と `weight_quant` に載る文字列）。"""
        torch.manual_seed(SEED)
        model = TinyNet()
        census = measure.census_w4_targets(model, measure.W4_ROLES["all"], 16)

        report = measure.w4_methods(16)[measure.RTN_KIND].apply(
            model, measure.W4_ROLES["all"], census.include()
        )

        assert "group 16" in report.describe()

    @pytest.mark.parametrize("kind", list(measure.w4_methods(measure.DEFAULT_GROUP_SIZE)))
    def test_every_method_rounds_to_other_values_under_another_group(self, kind) -> None:
        """g は 4 方式**すべて**の `fake_quant_*` へ届く（渡し忘れると値が 1 本も動かない）。

        どの重みが動くかは方式で違う — MXFP4 の scale は 2 のべきなので、group を割っても
        amax の指数が変わらなければ同じ値に落ちる（実測で動くのは正規分布の埋め込みだけ）。
        したがって縛るのは「どこか 1 本は動く」まで。
        """
        rounded: dict[int, dict[str, torch.Tensor]] = {}
        for group_size in (16, DEFAULT_G):
            torch.manual_seed(SEED)
            model = TinyNet()
            census = measure.census_w4_targets(model, measure.W4_ROLES["all"], group_size)
            measure.w4_methods(group_size)[kind].apply(
                model, measure.W4_ROLES["all"], census.include()
            )
            rounded[group_size] = {
                name: getattr(model, name).weight.clone() for name in ("linear", "conv", "emb")
            }

        moved = [
            name
            for name, weight in rounded[16].items()
            if not torch.equal(weight, rounded[DEFAULT_G][name])
        ]
        assert moved, f"{kind} の丸めが g で動かない"


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
        projection = measure.project_distribution(tmp_path, self.LINEAR, DEFAULT_G)
        group = projection["groups"]["話者ごとの net_g"]
        assert group["tensors"] == 1
        # i8 = 8·64 + 4·8 バイト / i4 = 8·64/2 + 4·8·(64/32) バイト
        assert group["current_bytes"] == 8 * 64 + 4 * 8
        assert group["projected_bytes"] == 8 * 64 // 2 + 4 * 8 * (64 // 32)
        assert projection["delta_bytes"] == group["delta_bytes"]

    def test_a_larger_group_charges_fewer_scale_bytes(self, tmp_path: Path) -> None:
        """縮小試算も同じ g で引く（g を上げれば scale の代金がそのぶん減る）。"""
        _write_dist(tmp_path / "F1" / "front")

        projection = measure.project_distribution(tmp_path, self.LINEAR, 64)

        assert projection["group_size"] == 64
        assert projection["groups"]["話者ごとの net_g"]["projected_bytes"] == (
            8 * 64 // 2 + 4 * 8 * (64 // 64)
        )

    def test_it_reports_the_rank2_int8_tensors_it_did_not_match(self, tmp_path: Path) -> None:
        """rank 2 の I8 という形だけで引くと embedding が混ざる — 見送った本数を出す。"""
        _write_dist(tmp_path / "F1" / "front")
        projection = measure.project_distribution(tmp_path, self.LINEAR, DEFAULT_G)
        assert projection["rank2_i8_not_linear"] == 1

    def test_it_separates_the_shared_subtree_from_the_per_speaker_one(self, tmp_path: Path) -> None:
        _write_dist(tmp_path / "F1" / "front")
        _write_dist(tmp_path / "shared" / "text_encoder")
        projection = measure.project_distribution(tmp_path, self.LINEAR, DEFAULT_G)
        assert sorted(projection["groups"]) == [
            "shared（DeBERTa text_encoder）",
            "話者ごとの net_g",
        ]
        assert projection["files"] == 2

    def test_it_measures_the_shrink_against_the_real_file_bytes(self, tmp_path: Path) -> None:
        _write_dist(tmp_path / "F1" / "front")
        projection = measure.project_distribution(tmp_path, self.LINEAR, DEFAULT_G)
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
            measure.project_distribution(tmp_path, self.LINEAR, DEFAULT_G)


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
        """校正は格子を変えない — 振るのは方式であって g ではない。

        釘は既定と同じ 32 だが `--w4-group-size` には**乗らない**（方式グリッド側だけの軸）。
        値を共有していた頃の綴りに戻すと、g を振ったとき校正の格子まで黙って動く。
        """
        assert measure.CALIB_GROUP_SIZE == DEFAULT_G
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
        counts = measure.TargetCounts(
            modules=2, channels=8, elements=65536, group_size=measure.CALIB_GROUP_SIZE
        )

        assert measure.group_scale_bits(counts) / counts.elements == 5.0

    def test_the_calibrated_codebook_costs_one_table_per_layer(self) -> None:
        """core の `kmeans_shared` は**層内**の表 — 全体 1 枚の `kmeans:shared` とは式が違う。"""
        counts = measure.TargetCounts(
            modules=2, channels=8, elements=65536, group_size=measure.CALIB_GROUP_SIZE
        )

        difference = measure.layer_table_bits(counts) - measure.group_scale_bits(counts)

        assert difference == measure.CODEBOOK_BITS * counts.modules


class TestBertVariant:
    METHODS = measure.w4_methods(DEFAULT_G)

    def test_it_keys_the_plain_and_calibrated_roundings_in_one_space(self) -> None:
        assert measure.bert_variant(measure.CONFIGS["bert-w4-nf4"], self.METHODS) == "nf4"
        assert measure.bert_variant(measure.CONFIGS["bert-gptq-nf4"], self.METHODS) == "gptq-nf4"
        assert measure.bert_variant(measure.CONFIGS["f32"], self.METHODS) is None

    def test_it_keys_the_plain_rounding_by_the_name_that_carries_g(self) -> None:
        """欄名は g を焼いた方式名 — g を振った 2 回の実行で `w4.bert_quant` の欄が割れる。"""
        recipe = measure.CONFIGS["bert-w4-rtn"]

        assert measure.bert_variant(recipe, self.METHODS) == "rtn-i4-g32"
        assert measure.bert_variant(recipe, measure.w4_methods(16)) == "rtn-i4-g16"

    def test_the_two_name_spaces_do_not_collide(self) -> None:
        """交わると 1 本の鍵で引けなくなる（特徴のキャッシュが黙って混ざる）。"""
        names = {method.name for method in self.METHODS.values()}
        assert not names & set(measure.CALIB_CONFIGS)

    def test_a_recipe_that_asks_for_both_is_rejected(self) -> None:
        recipe = measure.Recipe(None, None, scope=(), bert_method="nf4", bert_calib="gptq-nf4")

        with pytest.raises(AssertionError, match="どちらか一方"):
            measure.bert_variant(recipe, self.METHODS)


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

    def test_the_scan_counts_carry_the_calibration_grid(self) -> None:
        """校正の計数が担ぐ g は**校正の格子**（方式軸の g ではない — 投影はここから出る）。"""
        _scan, counts = measure.calib_targets(measure.bert_stages(TinyBert()))

        assert counts.group_size == measure.CALIB_GROUP_SIZE

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
