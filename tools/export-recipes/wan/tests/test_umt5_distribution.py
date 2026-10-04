"""umT5-XXL encoder の配布 recipe（`wan.umt5_distribution`）— 配布リポ `karume-umt5-xxl` の
組み立て。

入力は実物と同じ綴り・同じ種類ごとの格納の合成容器（`umt5_fixture`）。門に落とされることを見る
ケースも同じ器で作り、**宣言だけを実物とずらす**。実物の系列（`outputs/series/umt5-xxl-i8-dyn`）が
ある機では、実物の束縛表の本数（i8 169 本・f32 73 本）まで固定する（無ければ SKIP）。
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import ir_fixtures
import pytest
from container_series import part_paths, write_component
from upstream_fixture import OTHER_REVISION, stamp_fixture_provenance

import dist
from _shared.licenses import APACHE_LICENSE_2_0_PATH
from _shared.paths import SERIES_ROOT
from karume.container import Provenance
from karume.dist import (
    MANIFEST_FILENAME,
    MODEL_CARD_FILENAME,
    NOTICE_FILENAME,
    DistError,
    assemble_family,
    resolve_card_renderer,
)
from wan.sources import UMT5_SOURCES
from wan.tests import umt5_fixture
from wan.umt5_distribution import (
    PIPELINE,
    UMT5_CONSUMER_REPO_NAME,
    UMT5_DEFAULT_MODEL,
    UMT5_INPUTS,
    UMT5_KIND_LAYOUTS,
    UMT5_MODEL_FILE,
    UMT5_OUTPUT_PATHS,
    UMT5_PIPELINE,
    UMT5_RELATIVE_BIAS,
    UMT5_REPO_NAME,
    UMT5_ROLE,
    UMT5_SERIES,
    UMT5_SUPPORTED_PIPELINE,
    UMT5_TOKENS,
    render_umt5_model_card,
    storage_kind,
    umt5_bindings,
    umt5_container,
    umt5_plan,
)

#: umT5 の上流の表の行（本家 `google/umt5-xxl` の pin — ADR 0122 決定 1）。
_UPSTREAM = UMT5_SOURCES[UMT5_DEFAULT_MODEL].source

#: 書き手（`wan.umt5_export.provenance`）が焼く出所の正常形 — umT5 の上流の pin した revision。
_PINNED = Provenance(
    license=_UPSTREAM.license,
    notice=NOTICE_FILENAME,
    upstream_revision=_UPSTREAM.revision,
)


@pytest.fixture(autouse=True)
def _pinned_provenance(monkeypatch: pytest.MonkeyPatch) -> None:
    """合成の容器（`ir_fixtures` を含む）に「台本が pin した revision から焼いた」出所を
    名乗らせる。"""
    stamp_fixture_provenance(monkeypatch, _PINNED)


def _series(root: Path, **overrides: Any) -> Path:
    """系列に容器 1 本を置き、代表 path を返す（`overrides` は合成の容器の故障注入の席 —
    出所は `ir_fixtures` が焼く値に揃え、テストの間だけ差し替わる）。"""
    container = umt5_container(root / "outputs" / "series")
    parts = umt5_fixture.umt5_container(provenance=ir_fixtures.FIXTURE_PROVENANCE, **overrides)
    write_component(container, parts)
    # 配布に入ってはいけない golden（系列には実際にこれが並んでいる）。
    (container.parent / "reference.band-l0008.safetensors").write_bytes(b"golden")
    return container


def _assemble(root: Path, container: Path) -> tuple[Path, dict[str, Any]]:
    out_dir = root / "models" / UMT5_REPO_NAME
    manifest = assemble_family(
        [umt5_plan(container)],
        out_dir,
        UMT5_DEFAULT_MODEL,
        render_card=lambda manifest, host_assets: PIPELINE.card_profiles["umt5"](
            manifest, repo=f"hdae/{UMT5_REPO_NAME}", host_assets=host_assets
        ),
        root_files=PIPELINE.root_files,
    )
    return out_dir, manifest


@pytest.fixture
def assembled(tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    return _assemble(tmp_path, _series(tmp_path))


def _model(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    return manifest["models"][UMT5_DEFAULT_MODEL]


def _present(out_dir: Path) -> list[str]:
    return sorted(str(path.relative_to(out_dir)) for path in out_dir.rglob("*") if path.is_file())


class TestLayout:
    def test_it_places_one_container_and_the_legal_texts(self, assembled) -> None:
        """容器 1 本（部品 `text_encoder`）と直下の 4 本だけ — 系列の golden は入らない。"""
        out_dir, _ = assembled
        expected = [
            f"{UMT5_DEFAULT_MODEL}/{rel}" for rel in part_paths(UMT5_OUTPUT_PATHS[UMT5_ROLE])
        ]
        assert _present(out_dir) == sorted(
            [*expected, MANIFEST_FILENAME, MODEL_CARD_FILENAME, "LICENSE.md", NOTICE_FILENAME]
        )
        assert UMT5_OUTPUT_PATHS[UMT5_ROLE] == "text_encoder/model.i8.krm"

    def test_the_manifest_declares_the_encoder_seat(self, assembled) -> None:
        """pipeline は部品の役（`umt5-encoder/1`）・席は i8 の 1 つ・資産なし・設定は空
        （ADR 0119 追記 D）。"""
        _, manifest = assembled
        model = _model(manifest)
        assert manifest["defaultModel"] == UMT5_DEFAULT_MODEL == "xxl"
        assert model["pipeline"] == UMT5_PIPELINE == "umt5-encoder/1"
        assert list(model["weights"]) == [UMT5_ROLE]
        assert list(model["weights"][UMT5_ROLE]) == ["i8"]
        assert model["assets"] == {}
        assert list(model["quants"]) == ["i8"]
        assert model["quants"]["i8"]["weights"] == {UMT5_ROLE: "i8"}
        assert model["quants"]["i8"]["session"] == {}
        assert model["defaultQuant"] == "i8"
        assert model["pipelineConfig"] == {}

    def test_the_repository_ships_the_apache_license_and_the_change_notice(self, assembled) -> None:
        out_dir, _ = assembled
        assert (out_dir / "LICENSE.md").read_bytes() == APACHE_LICENSE_2_0_PATH.read_bytes()
        prose = " ".join((out_dir / NOTICE_FILENAME).read_text(encoding="utf-8").split())
        assert "Apache License, Version 2.0" in prose
        assert "**int8 weights**" in prose
        assert "keep the source float32 values" in prose
        assert 'GELU(approximate="tanh")' in prose
        # 上流の格納は F32（bf16 の写しではない — 調査の実測）。
        assert "holds the encoder in float32" in prose
        assert "bfloat16" not in prose
        # 上流は本家の encoder で、Wan2.1 の text_encoder とのビット一致は確かめた（ADR 0122
        # 決定 1 — 「同一は未確認」から事実が変わった）。decoder と lm_head は含めない。
        assert "has not been checked" not in prose
        assert "the encoder of the umT5-XXL checkpoint `google/umt5-xxl`" in prose
        assert "that repository has no NOTICE file" in prose
        assert "bit-identical to the float32 `text_encoder` folder" in prose
        assert "the decoder and `lm_head` are not included" in prose


class TestTheBindingGate:
    """束縛表の種類ごとの格納（`assert_umt5_bindings`）。「i8 を含む」だけでは素通りする
    取り違え。"""

    def test_it_refuses_quantized_relative_position_tables(self, tmp_path: Path) -> None:
        container = _series(tmp_path, table_layout="i8")
        with pytest.raises(DistError, match=UMT5_RELATIVE_BIAS):
            umt5_plan(container)

    def test_it_refuses_linear_weights_left_in_f32(self, tmp_path: Path) -> None:
        container = _series(tmp_path, linear_layout="f32")
        with pytest.raises(DistError, match="'linear'"):
            umt5_plan(container)

    def test_it_refuses_an_encoder_without_relative_position_tables(self, tmp_path: Path) -> None:
        container = _series(tmp_path, tables=False)
        with pytest.raises(DistError, match="重みの種類"):
            umt5_plan(container)

    def test_the_bindings_are_counted_by_kind(self, tmp_path: Path) -> None:
        """対（非恒真）: 正常形は種類ごとに要求の格納だけを持つ。"""
        bindings = umt5_bindings(_series(tmp_path))
        layers = umt5_fixture.LAYERS
        assert bindings == {
            "constant": {"f32": 1},
            "embed_tokens": {"i8": 1},
            "linear": {"i8": layers},
            "norm": {"f32": layers + 1},
            UMT5_RELATIVE_BIAS: {"f32": layers},
        }
        assert all(set(layouts) == {UMT5_KIND_LAYOUTS[kind]} for kind, layouts in bindings.items())


class TestTheContract:
    """グラフの入出力（`umt5_context_width`）— 読み手は入力を名前で束ねる。"""

    @pytest.mark.parametrize(
        ("index", "replacement"),
        [
            (0, ("ids", "i32", [1, "L"])),
            (0, ("input_ids", "f32", [1, "L"])),
            (1, ("relative_position_buckets", "i32", ["L", "M"])),
        ],
        ids=["renamed", "dtype", "second-symbol"],
    )
    def test_it_refuses_other_inputs(
        self, tmp_path: Path, index: int, replacement: tuple[str, str, list[Any]]
    ) -> None:
        inputs = list(UMT5_INPUTS)
        inputs[index] = replacement
        container = _series(tmp_path, inputs=inputs)
        with pytest.raises(DistError, match="グラフ入力"):
            umt5_plan(container)

    def test_it_returns_the_output_width(self, tmp_path: Path) -> None:
        from wan.umt5_distribution import assert_umt5_encoder

        assert assert_umt5_encoder(_series(tmp_path, width=12), UMT5_DEFAULT_MODEL) == 12


class TestTheProvenance:
    def test_it_refuses_a_container_baked_from_another_revision(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        stamp_fixture_provenance(
            monkeypatch,
            Provenance(
                license=_PINNED.license, notice=NOTICE_FILENAME, upstream_revision=OTHER_REVISION
            ),
        )
        with pytest.raises(DistError, match="別の revision"):
            umt5_plan(_series(tmp_path))

    def test_it_refuses_a_container_that_names_another_license(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        stamp_fixture_provenance(
            monkeypatch,
            Provenance(
                license="cc-by-nc-4.0",
                notice=NOTICE_FILENAME,
                upstream_revision=_PINNED.upstream_revision,
            ),
        )
        with pytest.raises(DistError, match=r"provenance\.license が 'cc-by-nc-4.0'"):
            umt5_plan(_series(tmp_path))

    def test_it_refuses_a_model_missing_from_the_upstream_table(self, tmp_path: Path) -> None:
        with pytest.raises(DistError, match="知らない"):
            umt5_plan(_series(tmp_path), "base")

    def test_the_upstream_is_the_pinned_commit_of_google_umt5_xxl(self) -> None:
        """本家の main の commit を 40 桁で pin し、encoder に要る shard 3 本の sha256（64 桁）を
        持つ。"""
        row = UMT5_SOURCES[UMT5_DEFAULT_MODEL]
        assert row.source.repo == "google/umt5-xxl"
        assert row.source.revision == "66cb9e7e85526fe440a945569e42c72fb6cbc0ad"
        assert row.source.license == "apache-2.0"
        assert sorted(row.shards) == [f"pytorch_model-0000{n}-of-00006.bin" for n in (1, 2, 3)]
        assert all(
            len(digest) == 64 and set(digest) <= set("0123456789abcdef")
            for digest in row.shards.values()
        )


class TestTheDriver:
    def test_the_driver_carries_the_pipeline(self) -> None:
        assert dist.PIPELINES["umt5"] is PIPELINE

    def test_the_default_output_is_the_repository_name(self) -> None:
        assert dist.default_out_dir(PIPELINE, [UMT5_DEFAULT_MODEL]).name == UMT5_REPO_NAME

    def test_the_card_profile_needs_no_choice(self) -> None:
        assert list(PIPELINE.card_profiles) == ["umt5"]
        assert resolve_card_renderer(PIPELINE, None) is render_umt5_model_card


class TestTheModelCard:
    def test_it_refuses_a_pipeline_it_does_not_describe(self, assembled) -> None:
        _, manifest = assembled
        foreign = json.loads(json.dumps(manifest))
        _model(foreign)["pipeline"] = "wan/1"
        with pytest.raises(ValueError, match=UMT5_SUPPORTED_PIPELINE):
            render_umt5_model_card(foreign, "hdae/x")

    def test_it_attributes_the_encoder_of_google_umt5_xxl(self, assembled) -> None:
        """帰属は本家の commit の encoder（decoder と lm_head は含めない）・Relation は Wan の
        2 つの checkpoint との確かめた関係（ADR 0122 決定 1）。本家の encoder はリポ直下にあるので、
        帰属の文面に空の subfolder を流さない。"""
        out_dir, _ = assembled
        card = (out_dir / MODEL_CARD_FILENAME).read_text(encoding="utf-8")
        prose = " ".join(card.split())
        assert "base_model: google/umt5-xxl" in card
        assert "base_model_relation: quantized" in card
        assert f"license: {_UPSTREAM.license}" in card
        assert f"at commit `{_UPSTREAM.revision}`" in card
        assert "the encoder weights (`shared.weight` and `encoder.*`) of [google/umt5-xxl]" in prose
        assert "The decoder and `lm_head` are not included." in prose
        assert "folder of [" not in prose
        assert "the `` folder" not in prose
        assert "every tensor is bit-identical to the float32 `text_encoder` folder" in prose
        assert "round-to-nearest-even rounding (checked on 2026-10-04)" in prose
        assert "base_model: Wan-AI/" not in card
        assert "has not been checked" not in prose

    def test_the_usage_points_at_the_wan_distribution(self, assembled) -> None:
        """単体の公開クラスは無い — Usage は参照元の Wan のリポを名指しする（ADR 0119 追記 D）。"""
        out_dir, _ = assembled
        card = " ".join((out_dir / MODEL_CARD_FILENAME).read_text(encoding="utf-8").split())
        assert "There is no standalone pipeline class for this repository." in card
        assert f"[`hdae/{UMT5_CONSUMER_REPO_NAME}`]" in card
        assert ".fromPretrained({" not in card

    def test_the_inputs_and_the_token_range_are_named(self, assembled) -> None:
        out_dir, _ = assembled
        card = " ".join((out_dir / MODEL_CARD_FILENAME).read_text(encoding="utf-8").split())
        low, high = UMT5_TOKENS
        assert "`input_ids` `[1, L]`" in card
        assert "`relative_position_buckets` `[L, L]`" in card
        assert f"accepts {low} to {high}, the end token included" in card

    def test_the_device_limits_come_from_the_manifest(self, assembled) -> None:
        _, manifest = assembled
        assert "### Device limits" not in render_umt5_model_card(manifest, "hdae/x")
        changed = json.loads(json.dumps(manifest))
        _model(changed)["quants"]["i8"]["requiredLimits"] = {
            "maxBufferSize": 1_050_148_864,
            "maxStorageBufferBindingSize": 1_050_148_864,
        }
        card = render_umt5_model_card(changed, "hdae/x")
        assert "### Device limits" in card
        assert "`maxBufferSize` ≥ 1,050,148,864 bytes" in card


class TestTheWritersSpellTheSameNames:
    """配布 recipe は torch を読まないので綴りを自前で持つ — 書き手（torch を読む）と一致する。"""

    def test_the_series_and_the_part_name(self) -> None:
        from wan import umt5_export

        assert umt5_export.SERIES_NAME == UMT5_SERIES
        assert umt5_export.MODEL_FILE == UMT5_MODEL_FILE
        assert umt5_export.COMPONENT_DIR == UMT5_ROLE
        assert umt5_export.GRAPH_NAME == UMT5_ROLE

    def test_the_writer_bakes_the_row_the_gate_checks(self) -> None:
        """書き手が焼く出所と配布の門が突き合わせる出所は、同じ表の同じ行。"""
        from wan import umt5_export

        assert umt5_export.provenance(UMT5_DEFAULT_MODEL) == _PINNED

    def test_the_graph_inputs_and_the_table_attribute(self) -> None:
        from wan import umt5_patch

        assert tuple(name for name, _, _ in UMT5_INPUTS) == umt5_patch.INPUT_NAMES
        assert all(umt5_patch.SYMBOL in shape for _, _, shape in UMT5_INPUTS)
        assert umt5_patch.RELATIVE_BIAS_ATTRIBUTE == UMT5_RELATIVE_BIAS

    def test_the_writer_classifies_with_the_same_function(self) -> None:
        """種類の分類は 1 本（書き手の検収の表と組み立ての門が同じ関数を引く）。"""
        from wan import umt5_export

        assert umt5_export.storage_kind is storage_kind

    def test_the_token_range_is_the_tokenizer_range(self) -> None:
        from wan import umt5_tokenizer

        assert UMT5_TOKENS == (umt5_tokenizer.MIN_TOKENS, umt5_tokenizer.MAX_LENGTH)

    def test_the_consumer_is_the_wan_repository(self) -> None:
        from wan.distribution import WAN_REPO_NAME

        assert UMT5_CONSUMER_REPO_NAME == WAN_REPO_NAME


class TestStorageKind:
    @pytest.mark.parametrize(
        ("key", "kind"),
        [
            ("encoder.embed_tokens.weight", "embed_tokens"),
            ("encoder.block.3.layer.0.SelfAttention.q.weight", "linear"),
            ("encoder.block.3.layer.1.DenseReluDense.wi_0.weight", "linear"),
            ("encoder.block.3.layer.0.layer_norm.weight", "norm"),
            ("encoder.final_layer_norm.weight", "norm"),
            (
                "encoder.block.3.layer.0.SelfAttention.relative_attention_bias.weight",
                UMT5_RELATIVE_BIAS,
            ),
            ("const.e7554f9eb93a2e4e", "constant"),
        ],
    )
    def test_it_names_the_kind(self, key: str, kind: str) -> None:
        assert storage_kind(key) == kind


_REAL = umt5_container(SERIES_ROOT)


@pytest.mark.skipif(not _REAL.parent.is_dir(), reason=f"実物の系列が無い: {_REAL.parent}")
class TestTheRealSeries:
    """実物の系列（`wan.umt5_export` が書いた i8 の容器）で計画が組め、束縛表が決定 5 の形を
    している。

    読むのは容器の 2 文書と束縛表だけ（重みの payload は読まない）。
    """

    def test_the_plan_passes_every_gate(self) -> None:
        plan = umt5_plan(_REAL)
        assert plan.pipeline == UMT5_PIPELINE
        assert set(plan.artifacts) == {UMT5_ROLE}

    def test_the_binding_table_is_169_int8_and_73_float32_weights(self) -> None:
        """linear 168 + 語彙埋め込み 1 が i8、相対位置の表 24 + RMSNorm 49 が f32（24 層 —
        ADR 0119 追記 10b）。持ち上げ定数（`const.*`）は重みではないので数えない。"""
        bindings = umt5_bindings(_REAL)
        assert bindings == {
            "constant": {"f32": 2},
            "embed_tokens": {"i8": 1},
            "linear": {"i8": 168},
            "norm": {"f32": 49},
            UMT5_RELATIVE_BIAS: {"f32": 24},
        }
        weights = {kind: layouts for kind, layouts in bindings.items() if kind != "constant"}
        assert sum(layouts.get("i8", 0) for layouts in weights.values()) == 169
        assert sum(layouts.get("f32", 0) for layouts in weights.values()) == 73
