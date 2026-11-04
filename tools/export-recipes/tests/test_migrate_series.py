"""The series migration driver (`migrate_series`) — its invariants pinned on synthetic series.

The subject is a synthetic **valid old distribution** (the shard list `legacy_writer.legacy_shards`
writes under the old convention), laid out like a real family. The families are not faked because
this driver's job is precisely to look up "which directory is which component name" in the family
table — passing a fake family would make a vacuous check that never sees a flaw in the table
itself.

4 things are checked:

1. A `krm` is placed, and the container's **graph name is the weights key** (not the directory
   name).
2. Goldens (`io.*`), old shards and `*.json` **do not change by a single byte** (the migration
   only reads them).
3. Sidecars (`rope_base` / PLE) become container assets **bit-identical**.
4. A series that cannot be classified, a component not in the table and a `.safetensors` with
   nowhere to fold into fail loudly.
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

#: The representative file name of an old shard list (the same spelling in every family).
COMPONENT = f"{migrate_series.SHARD_STEM}{migrate_series.SHARD_SUFFIX}"

#: One golden (a file placed alongside to check that the migration does not touch it).
GOLDEN = "io.case0.safetensors"

#: Component names that are not in the distribution (single graphs only for golden verification —
#: the module doc of `sbv2/distribution.py`). They are in the driver's table but in the weights of
#: no `<family>/distribution.py`.
GOLDEN_ONLY_ROLES: frozenset[str] = frozenset({"dp", "flow", "dec"})


def place(directory: Path, *, mark: str, groups: int = 2) -> dict[str, bytes]:
    """Place an old shard list + one golden and return **the bytes as placed** (for matching the
    invariance)."""
    stage_shards(directory, COMPONENT, legacy_shards(mark=mark, groups=groups))
    (directory / GOLDEN).write_bytes(b"golden-" + mark.encode())
    return {path.name: path.read_bytes() for path in sorted(directory.iterdir())}


def run(root: Path, *extra: str) -> None:
    migrate_series.main(["--series-dir", str(root), *extra])


def graph_names(component: Path) -> list[str]:
    """The graph names the placed container claims (read back — not a copy of the plan)."""
    read = read_container(list(container_parts(component)))
    return sorted(read.graph.graphs)


def assert_untouched(directory: Path, before: dict[str, bytes]) -> None:
    """The files that existed before the migration have not changed by a single byte."""
    for name, payload in before.items():
        assert (directory / name).read_bytes() == payload, name


def _write_reference(series: Path, *, shards: int, schema: int = REFERENCE_SCHEMA - 1) -> None:
    """The old-generation `reference.json` of a QAT series (it has `pleShards` and no
    `pleBlocks`)."""
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
    """Families that put one component directly under the series (the component name is spelled
    differently from the series name)."""

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
    """Families whose components sit in subdirectories (real cases of directory name ≠ component
    name)."""

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
        """sbv2's series name is `sbv2-<speaker>-<storage>`, and the component name is on the
        subdirectory side."""
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
        """The provenance (license and upstream revision) differs per speaker — how the name is
        stripped is the only lookup."""
        assert migrate_series.sbv2_model(series) == expected


class TestTheInputCheckpointOfAFlatSeries:
    """The provenance of a flat series is read from the checkpoint that built it (the same one as
    the script's)."""

    @pytest.mark.parametrize(
        ("series", "expected"),
        [
            ("depth-anything-v2-small-hf", "Depth-Anything-V2-Small-hf"),
            ("birefnet-hr-1024", "BiRefNet_HR"),
            ("minicpm5-1b-decode", "MiniCPM5-1B"),
        ],
    )
    def test_the_directory_name_spelled_as_the_series_prefix_is_found(
        self, tmp_path: Path, series: str, expected: str
    ) -> None:
        for name in ("Depth-Anything-V2-Small-hf", "BiRefNet_HR", "MiniCPM5-1B", "MiniCPM5-2B"):
            (tmp_path / name).mkdir()

        assert migrate_series.input_checkpoint(tmp_path, series) == tmp_path / expected

    def test_a_series_without_its_checkpoint_stops(self, tmp_path: Path) -> None:
        """A series whose provenance cannot be derived is not migrated (no container that claims
        another checkpoint's provenance is made)."""
        (tmp_path / "Depth-Anything-V2-Base-hf").mkdir()

        with pytest.raises(migrate_series.SeriesMigrationError, match="一意に決まらない"):
            migrate_series.input_checkpoint(tmp_path, "depth-anything-v2-small-hf")


class TestTheOrderOfTheFamilyTable:
    """The order of matching and the suffix branches (the only 3 places in the table that hold a
    condition)."""

    def test_a_qat_series_is_not_eaten_by_the_plain_gemma_entry(self, tmp_path: Path) -> None:
        """`gemma4-qat-*` shares the `gemma4-` prefix — with the table in the reverse order the qat
        entry would vanish."""
        series = tmp_path / "gemma4-qat-fixture-product"
        place(series, mark="qat")
        legacy_ple_sidecar(series)
        _write_reference(series, shards=1)

        run(tmp_path)

        assert migrate_series.classify(series.name).name == migrate_series.QAT_FAMILY
        assert graph_names(series / "model.krm") == ["model"]

    def test_the_drafter_suffix_names_the_other_seat(self, tmp_path: Path) -> None:
        """Only the MTP borrower has another component name (same family, same flat layout)."""
        drafter = tmp_path / "gemma4-fixture-drafter"
        place(drafter, mark="drafter")
        product = tmp_path / "gemma4-fixture-product"
        place(product, mark="product")

        run(tmp_path)

        assert graph_names(drafter / "model.krm") == ["drafter"]
        assert graph_names(product / "model.krm") == ["model"]


class TestSidecarsBecomeAssets:
    """The sidecars placed alongside become container assets **bit-identical**."""

    def test_rope_base_is_folded_byte_for_byte(self, tmp_path: Path) -> None:
        component = tmp_path / "anima-fixture-dyn" / "transformer"
        place(component, mark="dit")
        payload = legacy_rope_base(component)
        run(tmp_path)
        container = component / "model.krm"
        assert read_asset(container, ROPE_BASE_NAME) == payload
        # Folding **keeps the original file** (moving it aside is done separately, per
        # directory).
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
        """`ple.probe.safetensors` is a reference for the acceptance check, not an asset (folding
        it would mix it into the distribution)."""
        component = tmp_path / "gemma4-fixture-product"
        place(component, mark="gemma")
        legacy_ple_sidecar(component)
        (component / "ple.probe.safetensors").write_bytes(b"probe")
        run(tmp_path)
        declared = read_asset_declarations(component / "model.krm")
        assert not [name for name in declared if "probe" in name]


class TestGatesThatFailLoudly:
    """The migration runs only once, so a shape that cannot be decided is never let through
    silently."""

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
        """A flat series does not become `<series>/<series>` (the component path is relative to the
        series)."""
        component = tmp_path / "vowel-detector-fixture"
        place(component, mark="vowel")
        (component / "mel_basis.safetensors").write_bytes(b"unknown")

        with pytest.raises(migrate_series.SeriesMigrationError) as raised:
            run(tmp_path)

        assert str(raised.value).startswith("vowel-detector-fixture: 'mel_basis.safetensors'")

    def test_a_sidecar_only_subdirectory_is_examined_too(self, tmp_path: Path) -> None:
        """MUST: the check covers **every directory of the series** (including those without an
        old shard list).

        irodori's `pipeline/` and dacvae's `host/` hold only goldens, so they are not components.
        Going back to looking only at component directories would silently leave behind a new kind
        of sidecar that appears there (the migration runs only once, so there is no later chance to
        notice).
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
        """Leftovers are checked even in a series with no component (the `anima-pipeline*`
        shape)."""
        series = tmp_path / "anima-fixture-pipeline"
        series.mkdir()
        (series / "pipeline.safetensors").write_bytes(b"golden")
        (series / "latents.safetensors").write_bytes(b"unknown")

        with pytest.raises(migrate_series.SeriesMigrationError, match=r"latents\.safetensors"):
            run(tmp_path)

    @pytest.mark.parametrize("name", ["case.full.safetensors", "trim.safetensors"])
    def test_the_host_goldens_are_allowed_everywhere(self, tmp_path: Path, name: str) -> None:
        """Counterpart (not vacuous): a golden that is in the table passes in the same place."""
        series = tmp_path / "irodori-fixture"
        place(series / "dit", mark="dit")
        (series / "pipeline").mkdir()
        (series / "pipeline" / name).write_bytes(b"golden")

        run(tmp_path)

        assert graph_names(series / "dit" / "model.krm") == ["dit"]

    def test_a_ple_shard_the_index_does_not_declare_stops(self, tmp_path: Path) -> None:
        """MUST: the PLE count comes from **the index's declaration** (not a glob of the files).

        Counting with a glob lets files beyond the index (leftovers of a regeneration at another
        revision) **pass the check without being folded** — the same route as F-7 to "a container
        missing one asset".
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
        with pytest.raises(MigrateError, match="remain"):
            run(tmp_path)


class TestTheBlockSizeReachesBothPaths:
    """The asset's block size and the container's block size come from **one and the same
    value**."""

    def test_a_smaller_block_cuts_the_ple_asset_too(self, tmp_path: Path) -> None:
        """MUST: `block_bytes` reaches both the asset folding and the container with the same
        value.

        If they were decided separately, a run with an injected size would place **only the asset
        at the default block size** (a mismatched result: the container in per-row blocks, the
        asset in a single block).
        """
        series = tmp_path / "gemma4-fixture-product"
        place(series, mark="gemma")
        # A PLE of 512-byte rows × 16 tokens (small enough to fit in one block at the default
        # size).
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
    """The `reference.json` of a QAT series is the only exception to "left untouched" (ruling
    2)."""

    def _migrate(self, tmp_path: Path, *, shards: int = 1) -> Path:
        series = tmp_path / "gemma4-qat-fixture-product"
        place(series, mark="qat")
        legacy_ple_sidecar(series)
        _write_reference(series, shards=shards)
        run(tmp_path)
        return series

    def test_the_shard_count_becomes_the_block_count(self, tmp_path: Path) -> None:
        """MUST: the shard count is a number found nowhere after folding (the block split ignores
        the shard boundaries)."""
        series = self._migrate(tmp_path)

        record = json.loads(
            (series / migrate_series.QAT_REFERENCE_FILE).read_text(encoding="utf-8")
        )
        index = json.loads(read_asset(series / "model.krm", PLE_INDEX_ASSET))

        assert record["schema"] == REFERENCE_SCHEMA
        assert migrate_series.QAT_SHARDS_FIELD not in record
        assert record[migrate_series.QAT_BLOCKS_FIELD] == len(index["values"]["blocks"])

    def test_the_other_fields_and_their_order_stay_put(self, tmp_path: Path) -> None:
        """The migration changes no value by a single bit — the numbers matched against the files
        stay correct."""
        series = self._migrate(tmp_path)

        record = json.loads(
            (series / migrate_series.QAT_REFERENCE_FILE).read_text(encoding="utf-8")
        )

        assert record["fixedWeights"] == 4
        assert record["upstreamUnused"] == {"vision": 1}
        # `pleBlocks` takes the place of `pleShards` (the same order as the record a re-export
        # writes).
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
        """Counterpart (not vacuous): trying to upgrade an already-folded record again stops."""
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
        """Counterpart (not vacuous): the same record placed in a plain Gemma series is left
        untouched."""
        series = tmp_path / "gemma4-fixture-product"
        place(series, mark="gemma")
        legacy_ple_sidecar(series)
        _write_reference(series, shards=1)
        before = (series / migrate_series.QAT_REFERENCE_FILE).read_bytes()

        run(tmp_path)

        assert (series / migrate_series.QAT_REFERENCE_FILE).read_bytes() == before


class TestWhatIsNotConverted:
    """Unreferenced series (records of experiments / rejected candidates) and series without an
    old shard list are not migrated."""

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
    """The component-name table does not disagree with the weights of `<family>/distribution.py`.

    The spellings the driver uses are the distribution's component names themselves (container-v1
    §2.1), so on the day only one side moves, "only the migrated containers cannot be looked up by
    the runtime".
    """

    @pytest.mark.parametrize(
        ("declared", "families"),
        [
            ("anima.distribution.ANIMA_WEIGHTS", ("anima",)),
            # The 2 codec slots belong to a separate series (`dacvae-*`).
            ("irodori.distribution.IRODORI_WEIGHTS", ("irodori", "dacvae")),
            # The `text_encoder` slot is written by another recipe (the `deberta-*` series).
            ("sbv2.distribution.SBV2_WEIGHTS", ("sbv2", "deberta")),
        ],
    )
    def test_the_table_and_the_declaration_name_the_same_seats(
        self, declared: str, families: tuple[str, ...]
    ) -> None:
        """Checked in **both directions** (with only declaration ⊆ table, a mistyped table entry
        would pass)."""
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
    raise AssertionError(f"family '{name}' is not in the table")
