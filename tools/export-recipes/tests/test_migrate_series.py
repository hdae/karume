"""系列の移行ドライバ（`migrate_series`）— 合成の系列で不変条件を固定する。

被験体は合成の**正当な旧配布形**（`legacy_writer.legacy_shards` が旧規約で焼く shard 列）を
実在の family のレイアウトどおりに並べたもの。family を偽物にしないのは、このドライバの仕事が
まさに「どのディレクトリがどの部品名か」を family の表から引くことだから — 偽の family を
足して通すと、表そのものの綻びを見ない恒真な門になる。

見るのは 4 つ:

1. `krm` が据わり、容器の**グラフ名が weights のキー**である（ディレクトリ名ではない）。
2. golden（`io.*`）・旧 shard・`*.json` は **1 バイトも動かない**（移行は読むだけ）。
3. sidecar（`rope_base` / PLE）が**ビット同一**で容器の資産になる。
4. 分類できない系列・表に無い部品・畳み先の無い `.safetensors` は fail loudly。
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest
from legacy_writer import (
    ROPE_BASE_NAME,
    legacy_ple_sidecar,
    legacy_rope_base,
    legacy_shards,
    stage_shards,
)

import migrate_series
from _shared.container_read import read_asset, read_asset_declarations
from gemma4_qat.config import REFERENCE_SCHEMA
from karume.container import container_parts, read_container
from karume.migrate import MigrateError
from karume.ple import PLE_INDEX_ASSET

#: 旧 shard 列の代表ファイル名（どの family も同じ綴り）。
COMPONENT = f"{migrate_series.SHARD_STEM}{migrate_series.SHARD_SUFFIX}"

#: golden の 1 本（移行が触らないことを見るための同居ファイル）。
GOLDEN = "io.case0.safetensors"

#: 配布形に載らない部品名（golden 検証専用の単体グラフ — `sbv2/distribution.py` のモジュール
#: doc）。ドライバの表には在るが、どの `<family>/distribution.py` の weights にも居ない。
GOLDEN_ONLY_ROLES: frozenset[str] = frozenset({"dp", "flow", "dec"})


def place(directory: Path, *, mark: str, groups: int = 2) -> dict[str, bytes]:
    """旧 shard 列 + golden 1 本を置き、**置いた時点のバイト列**を返す（不変の突合用）。"""
    stage_shards(directory, COMPONENT, legacy_shards(mark=mark, groups=groups))
    (directory / GOLDEN).write_bytes(b"golden-" + mark.encode())
    return {path.name: path.read_bytes() for path in sorted(directory.iterdir())}


def run(root: Path, *extra: str) -> None:
    migrate_series.main(["--series-dir", str(root), *extra])


def graph_names(component: Path) -> list[str]:
    """据わった容器が名乗るグラフ名（読み直して引く — 計画の写しではない）。"""
    read = read_container(list(container_parts(component)))
    return sorted(read.graph.graphs)


def assert_untouched(directory: Path, before: dict[str, bytes]) -> None:
    """移行の前に在ったファイルが 1 バイトも動いていないこと。"""
    for name, payload in before.items():
        assert (directory / name).read_bytes() == payload, name


def _write_reference(series: Path, *, shards: int, schema: int = REFERENCE_SCHEMA - 1) -> None:
    """QAT 系列の旧世代 `reference.json`（`pleShards` を持ち `pleBlocks` を持たない）。"""
    (series / migrate_series.QAT_REFERENCE_FILE).write_text(
        json.dumps(
            {
                "schema": schema,
                "family": "gemma4-qat",
                "model": "e2b",
                "fixedWeights": 4,
                "weightFiles": 2,
                migrate_series.QAT_SHARDS_FIELD: shards,
                "upstreamUnused": {"vision": 1},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


class TestFlatSeries:
    """系列直下に 1 部品を置く family（部品名は系列名と綴りが違う）。"""

    def test_the_graph_is_named_after_the_weights_key(self, tmp_path: Path) -> None:
        series = tmp_path / "vowel-detector-fixture"
        before = place(series, mark="vowel")
        run(tmp_path)
        assert graph_names(series / "model.krm") == ["crnn"]
        assert_untouched(series, before)

    def test_the_old_shards_stay_next_to_the_container(self, tmp_path: Path) -> None:
        series = tmp_path / "vowel-detector-fixture"
        before = place(series, mark="vowel")
        run(tmp_path)
        placed = {path.name for path in series.iterdir()}
        assert set(before) <= placed
        assert any(name.endswith(".krm") for name in placed)

    def test_dry_run_writes_nothing(self, tmp_path: Path) -> None:
        series = tmp_path / "vowel-detector-fixture"
        before = place(series, mark="vowel")
        run(tmp_path, "--dry-run")
        assert {path.name for path in series.iterdir()} == set(before)


class TestNestedSeries:
    """サブディレクトリに部品が並ぶ family（ディレクトリ名 ≠ 部品名の実例）。"""

    def test_the_hyphenated_directory_becomes_the_underscored_key(self, tmp_path: Path) -> None:
        series = tmp_path / "irodori-fixture"
        before = place(series / "caption-proj", mark="caption")
        place(series / "dit", mark="dit")
        run(tmp_path)
        assert graph_names(series / "caption-proj" / "model.krm") == ["caption_proj"]
        assert graph_names(series / "dit" / "model.krm") == ["dit"]
        assert_untouched(series / "caption-proj", before)

    def test_the_codec_directory_becomes_the_prefixed_key(self, tmp_path: Path) -> None:
        series = tmp_path / "dacvae-fixture"
        place(series / "decoder", mark="decoder")
        run(tmp_path)
        assert graph_names(series / "decoder" / "model.krm") == ["codec_decoder"]

    def test_every_deberta_variant_carries_the_consuming_seat(self, tmp_path: Path) -> None:
        series = tmp_path / "deberta-fixture"
        place(series / "sbv2-22layer", mark="sbv2")
        place(series / "full-24layer", mark="full")
        run(tmp_path)
        for variant in ("sbv2-22layer", "full-24layer"):
            assert graph_names(series / variant / "model.krm") == ["text_encoder"]

    def test_the_speaker_name_and_the_rate_label_are_stripped(self, tmp_path: Path) -> None:
        """sbv2 は系列名が `sbv2-<話者>-<格納>` で、部品名はサブディレクトリ側にある。"""
        series = tmp_path / "sbv2-F1-i8"
        place(series / "front", mark="front")
        place(series / "voice", mark="voice")
        run(tmp_path)
        assert graph_names(series / "front" / "model.krm") == ["front"]
        assert graph_names(series / "voice" / "model.krm") == ["voice"]

    @pytest.mark.parametrize(
        ("series", "expected"), [("sbv2-F1", "F1"), ("sbv2-M2-i4", "M2"), ("sbv2-F2-f16", "F2")]
    )
    def test_the_provenance_is_looked_up_by_the_speaker_alone(
        self, series: str, expected: str
    ) -> None:
        """出所（ライセンスと上流の版）は話者ごとに違う — 剥がし方が唯一の引き当て。"""
        assert migrate_series.sbv2_model(series) == expected


class TestTheOrderOfTheFamilyTable:
    """判定の順序と接尾の分岐（表の中で唯一「条件が入る」3 か所）。"""

    def test_a_qat_series_is_not_eaten_by_the_plain_gemma_entry(self, tmp_path: Path) -> None:
        """`gemma4-qat-*` は `gemma4-` の接頭辞を共有する — 表の並びが逆だと qat が消える。"""
        series = tmp_path / "gemma4-qat-fixture-product"
        place(series, mark="qat")
        legacy_ple_sidecar(series)
        _write_reference(series, shards=1)

        run(tmp_path)

        assert migrate_series.classify(series.name).name == migrate_series.QAT_FAMILY
        assert graph_names(series / "model.krm") == ["model"]

    def test_the_drafter_suffix_names_the_other_seat(self, tmp_path: Path) -> None:
        """MTP の借り手だけが別の部品名（同じ family・同じ系列直下形）。"""
        drafter = tmp_path / "gemma4-fixture-drafter"
        place(drafter, mark="drafter")
        product = tmp_path / "gemma4-fixture-product"
        place(product, mark="product")

        run(tmp_path)

        assert graph_names(drafter / "model.krm") == ["drafter"]
        assert graph_names(product / "model.krm") == ["model"]


class TestSidecarsBecomeAssets:
    """同居する sidecar が**ビット同一**で容器の資産になる。"""

    def test_rope_base_is_folded_byte_for_byte(self, tmp_path: Path) -> None:
        component = tmp_path / "anima-fixture-dyn" / "transformer"
        place(component, mark="dit")
        payload = legacy_rope_base(component)
        run(tmp_path)
        container = component / "model.krm"
        assert read_asset(container, ROPE_BASE_NAME) == payload
        # 畳んでも**元のファイルは残る**（退避はディレクトリ単位で別に行う）。
        assert (component / f"{ROPE_BASE_NAME}.safetensors").read_bytes() == payload

    def test_the_ple_sidecar_becomes_the_index_and_the_blocks(self, tmp_path: Path) -> None:
        component = tmp_path / "gemma4-fixture-product"
        place(component, mark="gemma")
        ple = legacy_ple_sidecar(component)
        run(tmp_path)
        container = component / "model.krm"
        index = json.loads(read_asset(container, PLE_INDEX_ASSET))
        assert index["tokens"] == ple.tokens
        assert index["storage"] == ple.storage
        for key, payload in ple.payloads.items():
            joined = b"".join(
                read_asset(container, block["asset"]) for block in index[key]["blocks"]
            )
            assert joined == payload, key

    def test_the_probe_is_not_folded(self, tmp_path: Path) -> None:
        """`ple.probe.safetensors` は検収の参照であって資産ではない（畳むと配布物に混ざる）。"""
        component = tmp_path / "gemma4-fixture-product"
        place(component, mark="gemma")
        legacy_ple_sidecar(component)
        (component / "ple.probe.safetensors").write_bytes(b"probe")
        run(tmp_path)
        declared = read_asset_declarations(component / "model.krm")
        assert not [name for name in declared if "probe" in name]


class TestGatesThatFailLoudly:
    """移行は 1 度きりなので、決められない形は黙って通さない。"""

    def test_an_unknown_series_name_stops(self, tmp_path: Path) -> None:
        place(tmp_path / "unknown-family-fixture", mark="x")
        with pytest.raises(migrate_series.SeriesMigrationError, match="family"):
            run(tmp_path)

    def test_a_directory_outside_the_role_table_stops(self, tmp_path: Path) -> None:
        place(tmp_path / "irodori-fixture" / "not-a-target", mark="x")
        with pytest.raises(migrate_series.SeriesMigrationError, match="not-a-target"):
            run(tmp_path)

    def test_a_safetensors_with_nowhere_to_go_stops(self, tmp_path: Path) -> None:
        component = tmp_path / "vowel-detector-fixture"
        place(component, mark="vowel")
        (component / "mel_basis.safetensors").write_bytes(b"unknown")
        with pytest.raises(migrate_series.SeriesMigrationError, match="mel_basis"):
            run(tmp_path)

    def test_the_diagnostic_names_the_directory_once(self, tmp_path: Path) -> None:
        """系列直下形では `<系列>/<系列>` にならない（部品 path は系列からの相対）。"""
        component = tmp_path / "vowel-detector-fixture"
        place(component, mark="vowel")
        (component / "mel_basis.safetensors").write_bytes(b"unknown")

        with pytest.raises(migrate_series.SeriesMigrationError) as raised:
            run(tmp_path)

        assert str(raised.value).startswith("vowel-detector-fixture: 'mel_basis.safetensors'")

    def test_a_sidecar_only_subdirectory_is_examined_too(self, tmp_path: Path) -> None:
        """MUST: 門は**系列の全ディレクトリ**（旧 shard 列を持たない置き場も）。

        irodori の `pipeline/` や dacvae の `host/` は golden だけを持つので部品にならない。
        部品ディレクトリだけを見る形に戻すと、そこへ生えた新種の sidecar が黙って置き去りに
        なる（移行は 1 度きりなので、気づく機会が後から来ない）。
        """
        series = tmp_path / "irodori-fixture"
        place(series / "dit", mark="dit")
        (series / "pipeline").mkdir()
        (series / "pipeline" / "t-embed.safetensors").write_bytes(b"golden")
        (series / "pipeline" / "latents.safetensors").write_bytes(b"unknown")

        with pytest.raises(migrate_series.SeriesMigrationError) as raised:
            run(tmp_path)

        assert "irodori-fixture/pipeline: 'latents.safetensors'" in str(raised.value)

    def test_a_series_without_any_component_still_passes_the_gate(self, tmp_path: Path) -> None:
        """部品 0 本の系列（`anima-pipeline*` の形）でも取りこぼしは見る。"""
        series = tmp_path / "anima-fixture-pipeline"
        series.mkdir()
        (series / "pipeline.safetensors").write_bytes(b"golden")
        (series / "latents.safetensors").write_bytes(b"unknown")

        with pytest.raises(migrate_series.SeriesMigrationError, match=r"latents\.safetensors"):
            run(tmp_path)

    @pytest.mark.parametrize("name", ["case.full.safetensors", "trim.safetensors"])
    def test_the_host_goldens_are_allowed_everywhere(self, tmp_path: Path, name: str) -> None:
        """対（恒真でない）: 表に載っている golden は同じ置き場でも通る。"""
        series = tmp_path / "irodori-fixture"
        place(series / "dit", mark="dit")
        (series / "pipeline").mkdir()
        (series / "pipeline" / name).write_bytes(b"golden")

        run(tmp_path)

        assert graph_names(series / "dit" / "model.krm") == ["dit"]

    def test_a_ple_shard_the_index_does_not_declare_stops(self, tmp_path: Path) -> None:
        """MUST: PLE の勘定は**索引の宣言**から引く（現物の glob ではない）。

        glob で勘定すると、索引より多い現物（版を変えた再生成の残骸）が**畳まれないのに
        門も通る** — F-7 と同じ「資産が 1 つ足りない容器」を作る経路。
        """
        series = tmp_path / "gemma4-fixture-product"
        place(series, mark="gemma")
        legacy_ple_sidecar(series)
        (series / "ple-00002.safetensors").write_bytes(b"leftover")

        with pytest.raises(migrate_series.SeriesMigrationError, match=r"ple-00002\.safetensors"):
            run(tmp_path)

    def test_a_flat_shard_run_under_a_nested_family_stops(self, tmp_path: Path) -> None:
        place(tmp_path / "irodori-fixture", mark="x")
        with pytest.raises(migrate_series.SeriesMigrationError, match="サブディレクトリ"):
            run(tmp_path)

    def test_a_second_run_refuses_to_overwrite(self, tmp_path: Path) -> None:
        place(tmp_path / "vowel-detector-fixture", mark="vowel")
        run(tmp_path)
        with pytest.raises(MigrateError, match="残っている"):
            run(tmp_path)


class TestTheBlockSizeReachesBothPaths:
    """資産の刻みと容器の刻みが**同じ 1 つの値**から回ること。"""

    def test_a_smaller_block_cuts_the_ple_asset_too(self, tmp_path: Path) -> None:
        """MUST: `block_bytes` は資産の畳み込みと容器の両方へ同じ値が届く。

        別々に決まる形だと、寸法を差し込んだ実行で**資産だけが既定の刻みのまま**据わる
        （容器は 1 行ごとの block、資産は 1 本の block という食い違った器ができる）。
        """
        series = tmp_path / "gemma4-fixture-product"
        place(series, mark="gemma")
        # 行 512 バイト × 16 token の PLE（既定の刻みでは 1 block に収まる大きさ）。
        ple = legacy_ple_sidecar(series, tokens=16, layers=8, dim=64)
        plan = migrate_series.plan_series(series)
        row_bytes = len(ple.payloads["values"]) // ple.tokens
        rows_per_block = 4

        migrate_series.migrate_series(plan, block_bytes=row_bytes * rows_per_block)

        index = json.loads(read_asset(series / "model.krm", PLE_INDEX_ASSET))
        blocks = index["values"]["blocks"]
        assert len(blocks) == ple.tokens // rows_per_block
        assert {block["stop"] - block["start"] for block in blocks} == {rows_per_block}


class TestTheQatReferenceFollowsTheAssets:
    """QAT 系列の `reference.json` は「触らないもの」の唯一の例外（裁定 2）。"""

    def _migrate(self, tmp_path: Path, *, shards: int = 1) -> Path:
        series = tmp_path / "gemma4-qat-fixture-product"
        place(series, mark="qat")
        legacy_ple_sidecar(series)
        _write_reference(series, shards=shards)
        run(tmp_path)
        return series

    def test_the_shard_count_becomes_the_block_count(self, tmp_path: Path) -> None:
        """MUST: shard 本数は畳んだ後どこにも無い数（block の切り方は shard 境界を無視する）。"""
        series = self._migrate(tmp_path)

        record = json.loads(
            (series / migrate_series.QAT_REFERENCE_FILE).read_text(encoding="utf-8")
        )
        index = json.loads(read_asset(series / "model.krm", PLE_INDEX_ASSET))

        assert record["schema"] == REFERENCE_SCHEMA
        assert migrate_series.QAT_SHARDS_FIELD not in record
        assert record[migrate_series.QAT_BLOCKS_FIELD] == len(index["values"]["blocks"])

    def test_the_other_fields_and_their_order_stay_put(self, tmp_path: Path) -> None:
        """移行は値を 1 ビットも変えない — 現物と突き合わせる数はそのまま正しい。"""
        series = self._migrate(tmp_path)

        record = json.loads(
            (series / migrate_series.QAT_REFERENCE_FILE).read_text(encoding="utf-8")
        )

        assert record["fixedWeights"] == 4
        assert record["upstreamUnused"] == {"vision": 1}
        # `pleShards` の席へ `pleBlocks` が入る（再 export が書く記録と同じ並び）。
        assert list(record) == [
            "schema",
            "family",
            "model",
            "fixedWeights",
            "weightFiles",
            migrate_series.QAT_BLOCKS_FIELD,
            "upstreamUnused",
        ]

    def test_a_record_that_already_counts_blocks_stops(self, tmp_path: Path) -> None:
        """対（恒真でない）: 畳んだ後の記録をもう一度上げようとしたら止まる。"""
        series = tmp_path / "gemma4-qat-fixture-product"
        place(series, mark="qat")
        legacy_ple_sidecar(series)
        _write_reference(series, shards=1)
        record = json.loads(
            (series / migrate_series.QAT_REFERENCE_FILE).read_text(encoding="utf-8")
        )
        record[migrate_series.QAT_BLOCKS_FIELD] = record.pop(migrate_series.QAT_SHARDS_FIELD)
        (series / migrate_series.QAT_REFERENCE_FILE).write_text(json.dumps(record))

        with pytest.raises(migrate_series.SeriesMigrationError, match="畳む前の記録ではない"):
            run(tmp_path)

    def test_a_qat_series_without_the_record_stops(self, tmp_path: Path) -> None:
        series = tmp_path / "gemma4-qat-fixture-product"
        place(series, mark="qat")
        legacy_ple_sidecar(series)

        with pytest.raises(
            migrate_series.SeriesMigrationError, match=migrate_series.QAT_REFERENCE_FILE
        ):
            run(tmp_path)

    def test_only_the_record_of_a_qat_series_is_rewritten(self, tmp_path: Path) -> None:
        """対（恒真でない）: 通常 Gemma の系列に同じ記録を置いても触らない。"""
        series = tmp_path / "gemma4-fixture-product"
        place(series, mark="gemma")
        legacy_ple_sidecar(series)
        _write_reference(series, shards=1)
        before = (series / migrate_series.QAT_REFERENCE_FILE).read_bytes()

        run(tmp_path)

        assert (series / migrate_series.QAT_REFERENCE_FILE).read_bytes() == before


class TestWhatIsNotConverted:
    """参照されていない系列（実験 / 棄却の記録）と、旧 shard 列を持たない系列は移さない。"""

    @pytest.mark.parametrize("name", ["minicpm5-fixture-probe", "qwen3-fixture-rejected"])
    def test_experiment_records_are_skipped(self, tmp_path: Path, name: str) -> None:
        before = place(tmp_path / name, mark="x")
        run(tmp_path)
        assert {path.name for path in (tmp_path / name).iterdir()} == set(before)

    def test_a_series_without_shards_is_skipped(self, tmp_path: Path) -> None:
        series = tmp_path / "anima-fixture-tokenizer"
        (series / "text").mkdir(parents=True)
        (series / "text" / "tokenizer.json").write_text("{}")
        run(tmp_path)
        assert [path.name for path in (series / "text").iterdir()] == ["tokenizer.json"]


class TestTheFamilyTableMatchesTheDistributionDeclarations:
    """部品名の表が `<family>/distribution.py` の weights と食い違っていないこと。

    ドライバが名乗る綴りは配布形の部品名そのもの（container-v1 §12）なので、片方だけ動いた
    日に「移行した容器だけがランタイムから引けない」形になる。
    """

    @pytest.mark.parametrize(
        ("declared", "families"),
        [
            ("anima.distribution.ANIMA_WEIGHTS", ("anima",)),
            # コーデックの 2 席は別系列（`dacvae-*`）が持つ。
            ("irodori.distribution.IRODORI_WEIGHTS", ("irodori", "dacvae")),
            # `text_encoder` 席を焼くのは別 recipe（`deberta-*` 系列）。
            ("sbv2.distribution.SBV2_WEIGHTS", ("sbv2", "deberta")),
        ],
    )
    def test_the_table_and_the_declaration_name_the_same_seats(
        self, declared: str, families: tuple[str, ...]
    ) -> None:
        """**両方向**で見る（宣言 ⊆ 表 の片側だけだと、表の打ち間違いの席が素通りする）。"""
        module, name = declared.rsplit(".", 1)
        weights = getattr(importlib.import_module(module), name)
        roles = {role for family in families for role in _family(family).roles.values()}
        assert set(weights) == roles - GOLDEN_ONLY_ROLES

    def test_the_flat_families_name_a_single_key(self) -> None:
        for family in migrate_series.FAMILIES:
            assert bool(family.roles) != (family.flat_role is not None), family.name


def _family(name: str) -> migrate_series.Family:
    for family in migrate_series.FAMILIES:
        if family.name == name:
            return family
    raise AssertionError(f"family '{name}' が表に無い")
