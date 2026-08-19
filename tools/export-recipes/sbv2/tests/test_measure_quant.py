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
