"""Migration from the old distribution to the container format (`karume.migrate`) —
docs/container-v1.md §12.

The subject is a synthetic **valid** old distribution (the shard sequence
{@link legacy_writer.legacy_shards} writes under the old convention — the writer no longer exists
in the product, so it lives only in the tests). The source of truth for the comparison is **the
raw bytes read directly from the input safetensors**, taken on a path separate from the migrator's
self-check (so that the tests fail even if the self-check is tautological).

The rules of the graph alone (renaming, canonical serialization, block packing) belong to
`test_container.py`. What this file checks is the migration itself — deriving the bindings, the
check of the scale shape, payload identity, the discipline of the swap, the CLI.
"""

from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import replace
from pathlib import Path

import pytest
from ir_fixtures import fixture_spec
from legacy_writer import (
    PLE_INDEX_FILE,
    PLE_SHARD_FILE,
    ROPE_BASE_NAME,
    Entry,
    legacy_piece_shards,
    legacy_ple_sidecar,
    legacy_rope_base,
    legacy_shards,
    order,
    piece_entries,
    stage_shards,
    write_safetensors,
)

from karume import migrate, publish
from karume.container import (
    PART_LENGTH_CHOICES,
    AssetInput,
    ContainerFormatError,
    Encoding,
    Provenance,
    numbered_path,
    read_container,
)
from karume.emit import stored_model
from karume.ir import (
    IrGraph,
    IrInitializer,
    IrInput,
    IrNode,
    IrShared,
    IrStorage,
    IrValue,
)
from karume.legacy import (
    IR_METADATA_KEY,
    LegacyFormatError,
    SourceTensor,
    StoredEntry,
    parse_piece_key,
    payload_chunks,
    read_component,
    resolve_shards,
)
from karume.migrate import EXTRA_ASSETS, FileRef, MigrateError, migrate_component
from karume.ple import PLE_INDEX_ASSET, ple_assets
from karume.publish import publish_container
from karume.verify import bind_graphs, parse_ir_graph

#: The file name of the component's representative path (with the storage tag — the same spelling
#: as the series).
COMPONENT = "model.i4.safetensors"

#: The container asset name and role the old `extras`' `rope_base` is folded into
#: (`karume.migrate.EXTRA_ASSETS`).
ROPE_BASE_ASSET, ROPE_BASE_ROLE = EXTRA_ASSETS[ROPE_BASE_NAME]

#: The block limit when cutting the PLE (the setting that splits the small synthetic table into
#: **several blocks** — an axis separate from the container's block limit, deciding the size of
#: one asset).
PLE_BLOCK_BYTES = 64

#: The provenance the caller of the migration passes (`--license` is required; `writer` appears
#: only when given explicitly).
PROVENANCE = Provenance(license="apache-2.0", writer="karume/test")

#: Sizes lowered so that the synthetic assets (a few KiB) exercise part crossing and piece
#: splitting.
SMALL_PART_BYTES = 512
SMALL_BLOCK_BYTES = 256

#: A part length that one block and its group of companion scales do not fit in (makes a run whose
#: placement cannot be decided).
UNPLACEABLE_PART_BYTES = 64

MIB = 1024 * 1024


def read_safetensors(blob: bytes) -> tuple[dict[str, str], dict[str, bytes]]:
    """Opens safetensors bytes into `(__metadata__, tensor key → raw bytes)`."""
    length = struct.unpack("<Q", blob[:8])[0]
    header = json.loads(blob[8 : 8 + length])
    start = 8 + length
    tensors = {
        name: blob[start + spec["data_offsets"][0] : start + spec["data_offsets"][1]]
        for name, spec in header.items()
        if name != "__metadata__"
    }
    return dict(header.get("__metadata__", {})), tensors


def source_material(shards: list[bytes]) -> tuple[dict[str, str], dict[str, bytes]]:
    """Folds a shard sequence into one material (the metadata is the first shard's)."""
    metadata, tensors = read_safetensors(shards[0])
    for blob in shards[1:]:
        tensors.update(read_safetensors(blob)[1])
    return metadata, tensors


def stage_series(directory: Path, shards: list[bytes]) -> Path:
    """Places the current form (a numbered shard sequence) and returns the component's
    representative path."""
    directory.mkdir(parents=True, exist_ok=True)
    for index, blob in enumerate(shards, start=1):
        numbered_path(directory / COMPONENT, index, len(shards)).write_bytes(blob)
    return directory / COMPONENT


def stage_legacy_single(directory: Path, shards: list[bytes]) -> Path:
    """The old single-file distribution (`karume_ir` and every tensor share one file)."""
    directory.mkdir(parents=True, exist_ok=True)
    metadata, tensors = source_material(shards)
    shapes = {}
    for blob in shards:
        length = struct.unpack("<Q", blob[:8])[0]
        for name, spec in json.loads(blob[8 : 8 + length]).items():
            if name != "__metadata__":
                shapes[name] = (spec["dtype"], tuple(spec["shape"]))
    entries = order(
        [
            Entry(name=name, dtype=shapes[name][0], shape=shapes[name][1], payload=raw)
            for name, raw in tensors.items()
        ]
    )
    target = directory / COMPONENT
    target.write_bytes(write_safetensors(entries, metadata))
    return target


def migrated_payloads(paths: list[Path]) -> dict[str, bytes]:
    """Reads **the payload of each initializer** back from the written container (the actual bytes
    without padding).

    The read goes through the merge layer's supply plan (`verify.bind_graphs`), so both the piece
    sequence and the scales are read from "where the declaration says they are" — reaching the same
    answer on a path separate from the migrator's self-check.
    """
    read = read_container(paths)
    bound = next(iter(bind_graphs(read.graph, read.model).values()))
    found: dict[str, bytes] = {}
    for name, supply in bound.supplies.items():
        found[name] = b"".join(
            read.block(block.id)[: block.payload_bytes] for block in supply.blocks
        )
        if supply.scale is not None:
            assert supply.encoding.scale_block is not None
            found[f"scale:{name}"] = read.block(supply.scale.id)[: supply.scale.payload_bytes]
    return found


def scale_keys(graph: IrGraph) -> dict[str, str]:
    """Initializer tensor key → scale tensor key (the counterpart of the comparison is looked up
    from the old declaration)."""
    return {
        init.tensor: init.storage.scale
        for init in graph.initializers.values()
        if init.tensor is not None and init.storage.scale is not None
    }


@pytest.fixture
def small_parts(monkeypatch: pytest.MonkeyPatch) -> None:
    """Adds **the size of the synthetic assets** to the writer's choice set (256 MiB and up).

    Synthetic assets of a few KiB do not split into parts at the set's part lengths, so the set is
    widened only for runs that exercise part crossing. The check of the set itself (values outside
    the set fail) is covered by `TestThePartLength` with the real set.
    """
    monkeypatch.setattr(
        migrate,
        "PART_LENGTH_CHOICES",
        (*PART_LENGTH_CHOICES, SMALL_PART_BYTES, UNPLACEABLE_PART_BYTES),
    )


@pytest.fixture
def written_part_bytes(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Collects the part lengths that reached the writer (`write_model_container`) in call order
    (the write itself passes through)."""
    seen: list[int] = []
    original = publish.write_model_container

    def spy(*args, **kwargs):
        seen.append(kwargs["part_bytes"])
        return original(*args, **kwargs)

    monkeypatch.setattr(publish, "write_model_container", spy)
    return seen


@pytest.fixture
def component(tmp_path: Path) -> tuple[Path, dict[str, str], dict[str, bytes]]:
    """Places a current distribution with mixed i4 (the same form as the real one, where i4 / i8 /
    f32 coexist)."""
    shards = legacy_shards(mark="mig", storage="i4")
    metadata, tensors = source_material(shards)
    return stage_series(tmp_path / "src", shards), metadata, tensors


class TestTheMigratedContainer:
    def test_every_payload_survives_byte_for_byte(
        self, component: tuple[Path, dict[str, str], dict[str, bytes]], tmp_path: Path
    ) -> None:
        """Invariant 1 of §12 — neither the data nor the scales change by a single byte (padding is
        written anew)."""
        path, metadata, tensors = component
        result = migrate_component(path, tmp_path / "out", provenance=PROVENANCE)
        graph = parse_ir_graph(metadata[IR_METADATA_KEY])
        scales = scale_keys(graph)

        found = migrated_payloads(list(result.parts))
        expected = {key: tensors[key] for key in scales}
        expected.update({f"scale:{key}": tensors[scale] for key, scale in scales.items()})
        expected.update(
            {
                init.tensor: tensors[init.tensor]
                for init in graph.initializers.values()
                if init.tensor is not None
            }
        )
        assert found == expected
        # The count the self-check made (data + scales) matches the count compared here.
        assert result.payloads == len(expected)
        assert result.initializers == len(graph.initializers)

    def test_the_old_distribution_is_read_only(
        self, component: tuple[Path, dict[str, str], dict[str, bytes]], tmp_path: Path
    ) -> None:
        """§12 — "the old input is kept" (the CLI neither deletes nor rewrites the input)."""
        path, _, _ = component
        before = {entry.name: entry.read_bytes() for entry in sorted(path.parent.iterdir())}
        migrate_component(path, tmp_path / "out", provenance=PROVENANCE)

        assert {entry.name: entry.read_bytes() for entry in sorted(path.parent.iterdir())} == before

    def test_the_same_input_produces_the_same_bytes(
        self, component: tuple[Path, dict[str, str], dict[str, bytes]], tmp_path: Path
    ) -> None:
        """Invariant 4 of §12 — deterministic (both the packing into parts and the JSON key order
        are decided by the input)."""
        path, _, _ = component
        first = migrate_component(path, tmp_path / "a", provenance=PROVENANCE)
        second = migrate_component(path, tmp_path / "b", provenance=PROVENANCE)

        assert [part.read_bytes() for part in first.parts] == [
            part.read_bytes() for part in second.parts
        ]

    def test_an_old_single_file_form_migrates_to_the_same_container(self, tmp_path: Path) -> None:
        """A distribution under the old rules (single file) is accepted too — the current checks are
        not applied to the input."""
        shards = legacy_shards(mark="mig", storage="i4")
        series = migrate_component(
            stage_series(tmp_path / "series", shards),
            tmp_path / "a",
            provenance=PROVENANCE,
            graph_name="depth",
        )
        single = migrate_component(
            stage_legacy_single(tmp_path / "legacy", shards),
            tmp_path / "b",
            provenance=PROVENANCE,
            graph_name="depth",
        )

        assert [part.read_bytes() for part in single.parts] == [
            part.read_bytes() for part in series.parts
        ]

    def test_a_numbered_input_path_is_folded_to_the_component(
        self, component: tuple[Path, dict[str, str], dict[str, bytes]], tmp_path: Path
    ) -> None:
        """Pointing at a local file (`-00001-of-00002`) does not migrate that one file alone."""
        path, _, _ = component
        first = sorted(path.parent.iterdir())[0]
        whole = migrate_component(path, tmp_path / "a", provenance=PROVENANCE)
        folded = migrate_component(first, tmp_path / "b", provenance=PROVENANCE)

        assert [part.read_bytes() for part in folded.parts] == [
            part.read_bytes() for part in whole.parts
        ]

    def test_the_graph_name_defaults_to_the_component_directory(
        self, component: tuple[Path, dict[str, str], dict[str, bytes]], tmp_path: Path
    ) -> None:
        path, _, _ = component
        result = migrate_component(path, tmp_path / "out", provenance=PROVENANCE)
        read = read_container(list(result.parts))

        assert list(read.graph.graphs) == [path.parent.name]

    def test_a_directory_name_outside_the_container_vocabulary_asks_for_an_explicit_name(
        self, tmp_path: Path
    ) -> None:
        """The default is implicit (the parent directory name), so a run where it does not fit stops
        and spells out **what to pass**."""
        shards = legacy_shards(mark="mig", storage="f32")
        path = stage_series(tmp_path / "text encoder", shards)

        with pytest.raises(MigrateError, match="pass it explicitly with `--graph-name`"):
            migrate_component(path, tmp_path / "out", provenance=PROVENANCE)

    def test_the_split_form_is_named_like_a_shard_series(
        self,
        component: tuple[Path, dict[str, str], dict[str, bytes]],
        tmp_path: Path,
        small_parts: None,
    ) -> None:
        path, _, _ = component
        result = migrate_component(
            path,
            tmp_path / "out",
            provenance=PROVENANCE,
            part_bytes=SMALL_PART_BYTES,
            _block_bytes=SMALL_BLOCK_BYTES,
        )
        total = len(result.parts)

        assert total >= 3
        assert [part.name for part in result.parts] == [
            f"model.i4-{index + 1:05d}-of-{total:05d}.krm" for index in range(total)
        ]

    def test_the_single_form_is_one_file(
        self, component: tuple[Path, dict[str, str], dict[str, bytes]], tmp_path: Path
    ) -> None:
        path, _, _ = component
        result = migrate_component(path, tmp_path / "out", provenance=PROVENANCE, single=True)

        assert [part.name for part in result.parts] == ["model.i4.krm"]
        assert read_container(list(result.parts)).header.kind == "model"

    def test_the_graph_container_carries_the_same_graph_description(
        self, component: tuple[Path, dict[str, str], dict[str, bytes]], tmp_path: Path
    ) -> None:
        """The `krg` is a byte-copy extraction from the `krm` (§9 — the graph descriptor is
        byte-identical)."""
        path, _, _ = component
        result = migrate_component(path, tmp_path / "out", provenance=PROVENANCE, write_graph=True)
        assert result.graph is not None
        model = read_container(list(result.parts))
        graph = read_container([result.graph])

        assert result.graph.name == "model.i4.krg"
        assert graph.header.kind == "graph"
        assert graph.model is None
        assert graph.graph_descriptor_bytes == model.graph_descriptor_bytes

    def test_the_container_is_never_staged_over_a_previous_output(
        self, component: tuple[Path, dict[str, str], dict[str, bytes]], tmp_path: Path
    ) -> None:
        """Leftovers at the destination **stop the run without being deleted** (which bytes to ship
        is not uniquely decided)."""
        path, _, _ = component
        migrate_component(path, tmp_path / "out", provenance=PROVENANCE)

        with pytest.raises(MigrateError, match="Artifacts of a previous run remain"):
            migrate_component(path, tmp_path / "out", provenance=PROVENANCE)

    def test_a_failed_migration_leaves_nothing_behind(
        self,
        component: tuple[Path, dict[str, str], dict[str, bytes]],
        tmp_path: Path,
        small_parts: None,
    ) -> None:
        """Put in place only after the checks pass — a failed run disappears together with its
        temporary files (the final name never appears)."""
        path, _, _ = component
        out = tmp_path / "out"
        out.mkdir()

        with pytest.raises(ContainerFormatError):
            # A part length that one block and its group of companion scales do not fit in (the
            # placement cannot be decided).
            migrate_component(path, out, provenance=PROVENANCE, part_bytes=UNPLACEABLE_PART_BYTES)

        assert sorted(out.iterdir()) == []


class TestTheDerivedBindings:
    """The binding (codec / rowAxis / groupSize) is decided by the old declaration and **the
    consuming op**."""

    @staticmethod
    def _graph(storage: IrStorage, shape: list[int], op: str = "linear") -> IrGraph:
        return IrGraph(
            inputs=[IrInput(name="x", dtype="f32", shape=[1, shape[-1]])],
            outputs=["y"],
            initializers={"p_w": IrInitializer(tensor="w", storage=storage)},
            values={"p_w": IrValue(dtype="f32", shape=shape), "y": IrValue(dtype="f32", shape=[1])},
            nodes=[IrNode(op=op, ins=["x", "p_w"], outs=["y"], attrs={})],
        )

    @pytest.mark.parametrize(
        ("dtype", "codec"),
        [("f32", "f32"), ("f16", "f16"), ("bf16", "bf16"), ("i32", "i32")],
    )
    def test_an_unquantized_storage_maps_to_its_codec(self, dtype: str, codec: str) -> None:
        bindings = migrate._bindings(self._graph(IrStorage(dtype=dtype), [4, 32]))

        assert bindings == {"w": Encoding(codec)}

    def test_a_per_channel_storage_declares_the_row_length_as_its_group_size(self) -> None:
        bindings = migrate._bindings(self._graph(IrStorage(dtype="i8", scale="w_scale"), [4, 32]))

        assert bindings == {
            "w": Encoding("int8-sym", group_size=32, row_axis=0, scale_key="w_scale")
        }

    def test_a_conv_transpose1d_weight_declares_row_axis_1(self) -> None:
        """Only the transposed layout `[Cin,Cout,K]` is axis 1 (the mirror of
        `emit.weight_channel_axes`)."""
        bindings = migrate._bindings(
            self._graph(IrStorage(dtype="i8", scale="w_scale"), [4, 3, 8], op="conv_transpose1d")
        )

        # Row length = numel / shape[1] = 96 / 3.
        assert bindings == {
            "w": Encoding("int8-sym", group_size=32, row_axis=1, scale_key="w_scale")
        }

    def test_a_group_storage_keeps_its_declared_group_size(self) -> None:
        bindings = migrate._bindings(
            self._graph(IrStorage(dtype="i4", scale="w_scale", group_size=16), [4, 32])
        )

        assert bindings == {
            "w": Encoding("int4-sym-g", group_size=16, row_axis=0, scale_key="w_scale")
        }

    def test_an_i2_storage_maps_to_int2_off_and_never_to_ternary(self) -> None:
        """The claim of being ternary is made by the quantizer (§6.3 — old i2 has codes outside that
        range)."""
        bindings = migrate._bindings(self._graph(IrStorage(dtype="i2", scale="w_scale"), [4, 32]))

        assert bindings["w"].codec == "int2-off"

    def test_a_shared_initializer_has_no_binding(self) -> None:
        graph = self._graph(IrStorage(dtype="f16"), [4, 32])
        graph.initializers["p_s"] = IrInitializer(
            shared=IrShared(tensor="lend.weight"), storage=IrStorage(dtype="f16")
        )
        graph.values["p_s"] = IrValue(dtype="f32", shape=[4, 32])

        assert set(migrate._bindings(graph)) == {"w"}

    def test_a_group_storage_consumed_on_axis_1_fails_loudly(self) -> None:
        """An old group scale was written with the leading dimension as rows — incompatible with
        consumption on axis 1."""
        graph = self._graph(
            IrStorage(dtype="i4", scale="w_scale", group_size=16), [4, 3, 8], "conv_transpose1d"
        )

        with pytest.raises(MigrateError, match="no shape to map to"):
            migrate._bindings(graph)

    def test_a_quantized_storage_without_a_scale_fails_loudly(self) -> None:
        with pytest.raises(MigrateError, match="no scale is declared"):
            migrate._bindings(self._graph(IrStorage(dtype="i8"), [4, 32]))


class TestTheScaleLayoutGate:
    """Checks before writing that the **shape** of an old scale matches the new declaration
    (`rowAxis` / `groupSize`)."""

    @staticmethod
    def _sources(weight: tuple[int, ...], scale: tuple[int, ...]) -> dict[str, SourceTensor]:
        return {
            "w": SourceTensor(
                entry=StoredEntry(name="w", dtype="I8", shape=weight, nbytes=64), segments=()
            ),
            "w_scale": SourceTensor(
                entry=StoredEntry(name="w_scale", dtype="F32", shape=scale, nbytes=4 * scale[0]),
                segments=(),
            ),
        }

    def _graph(self) -> IrGraph:
        return TestTheDerivedBindings._graph(IrStorage(dtype="i8", scale="w_scale"), [8, 8])

    def test_a_keepdim_scale_on_the_row_axis_passes(self) -> None:
        graph = self._graph()

        migrate._assert_scale_layouts(
            graph, migrate._bindings(graph), self._sources((8, 8), (8, 1))
        )

    def test_a_per_column_scale_is_refused_even_though_the_byte_count_agrees(self) -> None:
        """For a square weight the byte count happens to match — without checking the shape, the
        channels get swapped."""
        graph = self._graph()

        with pytest.raises(MigrateError, match=r"scale 'w_scale' has shape \[1, 8\]"):
            migrate._assert_scale_layouts(
                graph, migrate._bindings(graph), self._sources((8, 8), (1, 8))
            )


class TestTheCli:
    def test_it_writes_the_container_and_prints_one_line(
        self,
        component: tuple[Path, dict[str, str], dict[str, bytes]],
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        path, _, _ = component
        migrate.main([str(path), "--out", str(tmp_path / "out"), "--license", "apache-2.0"])
        printed = capsys.readouterr().out

        # Part 0 (the two documents) + part 1 (the const region) + one weight part.
        assert (tmp_path / "out" / "model.i4-00001-of-00003.krm").is_file()
        assert "parts 3" in printed and "payloads" in printed

    def test_the_writer_is_absent_unless_the_caller_names_one(
        self, component: tuple[Path, dict[str, str], dict[str, bytes]], tmp_path: Path
    ) -> None:
        """MUST: by default `writer` is **not written** (no generator tag is embedded).

        Embedding it would make a migrated container and a container a recipe exported directly
        (no `PROVENANCE` spells `writer`) different bytes forever — observed: part 0 of the 7
        mirror repositories differed from the rebuild by exactly the 25 bytes of
        `,"writer":"karume/0.12.0"`. The tool version is held in one place, the `generator` field
        of `karume.json`.
        """
        path, _, _ = component
        migrate.main([str(path), "--out", str(tmp_path / "out"), "--license", "mit"])
        read = read_container(sorted((tmp_path / "out").glob("*.krm")))

        assert read.model is not None
        assert read.model.provenance.license == "mit"
        assert read.model.provenance.writer is None
        assert "writer" not in read.model.provenance.to_document()

    def test_an_explicit_writer_is_still_written(
        self, component: tuple[Path, dict[str, str], dict[str, bytes]], tmp_path: Path
    ) -> None:
        """Counterpart (not tautological): given explicitly, it appears — what was dropped is "the
        default", not the field."""
        path, _, _ = component
        migrate.main(
            [str(path), "--out", str(tmp_path / "out"), "--license", "mit", "--writer", "acme/1"]
        )
        read = read_container(sorted((tmp_path / "out").glob("*.krm")))

        assert read.model is not None
        assert read.model.provenance.writer == "acme/1"

    @pytest.mark.parametrize("missing", [["--license", "mit"], ["--out", "/tmp/out"]])
    def test_it_requires_both_the_output_directory_and_the_license(
        self, missing: list[str]
    ) -> None:
        with pytest.raises(SystemExit) as raised:
            migrate.main(["a/model.safetensors", *missing])

        assert raised.value.code == 2

    def test_a_single_graph_name_cannot_cover_several_components(self, tmp_path: Path) -> None:
        with pytest.raises(MigrateError, match="can only be passed for one component"):
            migrate.main(
                [
                    "a/model.safetensors",
                    "b/model.safetensors",
                    "--out",
                    str(tmp_path),
                    "--license",
                    "mit",
                    "--graph-name",
                    "shared",
                ]
            )


class TestThePartLength:
    """The part length is one of the writer's choice set {256, 512, 768, 1024} MiB (container-v1
    §4.2)."""

    def test_the_default_is_256_mib(
        self,
        component: tuple[Path, dict[str, str], dict[str, bytes]],
        tmp_path: Path,
        written_part_bytes: list[int],
    ) -> None:
        path, _, _ = component
        migrate.main([str(path), "--out", str(tmp_path / "cli"), "--license", "mit"])
        migrate_component(path, tmp_path / "api", provenance=PROVENANCE)

        assert written_part_bytes == [256 * MIB, 256 * MIB]

    def test_the_cli_hands_the_chosen_mib_to_the_writer_as_bytes(
        self,
        component: tuple[Path, dict[str, str], dict[str, bytes]],
        tmp_path: Path,
        written_part_bytes: list[int],
    ) -> None:
        path, _, _ = component
        out = tmp_path / "out"
        migrate.main([str(path), "--out", str(out), "--license", "mit", "--part-bytes", "512"])

        assert written_part_bytes == [512 * MIB]
        assert read_container(sorted(out.glob("*.krm"))).header.kind == "model"

    @pytest.mark.parametrize("part_bytes", [300 * MIB, SMALL_PART_BYTES])
    def test_a_length_outside_the_choices_fails_loudly_before_writing(
        self,
        component: tuple[Path, dict[str, str], dict[str, bytes]],
        tmp_path: Path,
        part_bytes: int,
    ) -> None:
        """Even inside the ceiling (1024 MiB), a value outside the set stops the run — a check
        separate from the writer's ceiling check."""
        path, _, _ = component
        out = tmp_path / "out"

        with pytest.raises(
            MigrateError, match=r"outside the writer's choice set \{256, 512, 768, 1024\} MiB"
        ):
            migrate_component(path, out, provenance=PROVENANCE, part_bytes=part_bytes)

        assert not out.exists()

    def test_the_cli_refuses_a_length_outside_the_choices(self, tmp_path: Path) -> None:
        with pytest.raises(SystemExit) as raised:
            migrate.main(
                [
                    "a/model.safetensors",
                    "--out",
                    str(tmp_path),
                    "--license",
                    "mit",
                    "--part-bytes",
                    "300",
                ]
            )

        assert raised.value.code == 2


class TestTheSelfCheck:
    """The self-check is not tautological (fault injection) — written bytes that differ from the old
    ones are not put in place."""

    def test_a_flipped_byte_in_a_written_part_is_caught(
        self,
        component: tuple[Path, dict[str, str], dict[str, bytes]],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A run where one byte is corrupted right after writing is not put in place (the writer
        declares the block's sha256)."""
        path, _, _ = component
        original = publish.write_model_container

        def flip(*args, **kwargs):
            written = original(*args, **kwargs)
            raw = bytearray(written[-1].read_bytes())
            raw[-1] ^= 0xFF
            written[-1].write_bytes(bytes(raw))
            return written

        monkeypatch.setattr(publish, "write_model_container", flip)
        out = tmp_path / "out"

        with pytest.raises(
            ContainerFormatError, match=r"sha256 of .* differs from the declaration"
        ):
            migrate_component(path, out, provenance=PROVENANCE)

        assert not out.exists() or list(out.iterdir()) == []

    def test_the_payload_check_reads_the_old_shards(
        self, component: tuple[Path, dict[str, str], dict[str, bytes]]
    ) -> None:
        """The counterpart of the comparison is **the bytes read back from the old shards** (not a
        copy of the new container).

        If this became tautological, the migration would only "check what it wrote itself".
        """
        path, _, tensors = component
        _, stored = read_component(list(resolve_shards(path)))
        payloads = migrate._SourcePayloads(stored)

        assert {key: bytes(payloads[key]) for key in stored} == tensors


def test_the_payload_reader_reads_the_whole_payload(tmp_path: Path) -> None:
    """The base of the comparison (reading the old data) really streams every byte — missing even
    one byte gives a different value."""
    blob = bytes(range(256)) * 8
    target = tmp_path / "raw.bin"
    target.write_bytes(blob)
    source = SourceTensor(
        entry=StoredEntry(name="w", dtype="F32", shape=(len(blob) // 4,), nbytes=len(blob)),
        segments=((target, 0, len(blob)),),
    )

    digest = hashlib.sha256()
    for chunk in payload_chunks(source):
        digest.update(chunk)

    assert digest.hexdigest() == hashlib.sha256(blob).hexdigest()


class TestTheMigratedContainerMatchesADirectWrite:
    """**The identity check** — containers made from the same material are byte-identical even
    through different paths.

    ① A container written in the old shard form and then moved with `karume migrate`, and ② a
    container written directly with `stored_model` → `publish_container`, match down to the bytes
    of the part sequence. If this splits, "the migrated mirror" and "the re-exported series" become
    different assets despite the same weights, and the pin ↔ sha256 correspondence breaks at every
    step (the effective side of invariant 4 "deterministic" of container-v1 §12).

    The binding table, the placement and the JSON key order are all decided by the input, so the
    only room for a split is **where the derivation forks into two paths** — when it actually did
    (when a copy of `container_bindings` lived on the migration side), "declared i4 / actual i8"
    could be produced.
    """

    @staticmethod
    def _material(storage: str):
        return fixture_spec("ident", storage, (), ([1],), ("weight",), None)

    def _migrated(self, tmp_path: Path, storage: str) -> list[bytes]:
        """Writes the old shard form, then migrates it (①)."""
        shards = legacy_shards(mark="ident", storage=storage, groups=2)
        source = stage_shards(tmp_path / "old" / "enc", f"model.{storage}.safetensors", shards)
        out = tmp_path / "migrated"
        result = migrate_component(source, out, provenance=PROVENANCE, graph_name="enc")
        return [path.read_bytes() for path in result.parts]

    def _direct(self, tmp_path: Path, storage: str) -> list[bytes]:
        """Writes directly from the same material (②)."""
        graph, tensors, scales, overrides = self._material(storage)
        stored = stored_model(
            graph,
            tensors,
            weight_dtype=storage,
            weight_scales=scales,
            weight_dtype_overrides=overrides,
        )
        result = publish_container(
            tmp_path / "direct" / f"model.{storage}.krm",
            stored.graph,
            stored.tensors,
            stored.bindings,
            graph_name="enc",
            provenance=PROVENANCE,
        )
        return [path.read_bytes() for path in result.parts]

    @pytest.mark.parametrize("storage", ["f32", "f16", "i8", "i4"])
    def test_both_routes_write_the_same_parts(self, tmp_path: Path, storage: str) -> None:
        migrated = self._migrated(tmp_path, storage)
        direct = self._direct(tmp_path, storage)

        assert [len(part) for part in migrated] == [len(part) for part in direct]
        assert migrated == direct

    def _asset_material(self, tmp_path: Path, kind: str):
        """Places the old sidecar and returns the pair of ① the assets the migration builds and ②
        the assets the direct writer builds.

        ① is the asset path of the migration CLI (`migrate._extra_assets` / `migrate._ple_assets` —
        reading the old form and folding the index), ② is the entry point the writer actually uses
        (`AssetInput` / `karume.ple.ple_assets`). **Both build "a correct container"**, so this
        check is the only detector of a drift in the physical placement (dedicated parts, trailing
        padding, canonical serialization of the `assets` section).
        """
        repo = tmp_path / "old"
        where = "ident"
        if kind == "rope_base":
            payload = legacy_rope_base(repo)
            ref = FileRef(f"{ROPE_BASE_NAME}.safetensors", len(payload), "0" * 64)
            migrated = migrate._extra_assets(repo, ((ROPE_BASE_NAME, ref),), where)
            direct = {
                ROPE_BASE_ASSET: AssetInput(ROPE_BASE_ROLE, len(payload), payload),
            }
            return migrated, direct
        ple = legacy_ple_sidecar(repo)
        if kind == "ple-sidecar":
            # The path that migrates a series directory (the entry `migrate_series` goes through —
            # the index names the location).
            migrated = migrate.ple_sidecar_assets(
                repo / PLE_INDEX_FILE, where=where, block_bytes=PLE_BLOCK_BYTES
            )
        else:
            refs = tuple(
                (name, FileRef(name, (repo / name).stat().st_size, "0" * 64))
                for name in (PLE_INDEX_FILE, PLE_SHARD_FILE)
            )
            index = migrate.read_ple_index(repo / PLE_INDEX_FILE, where)
            fold = migrate._PleFold("enc", ((PLE_INDEX_ASSET, refs[0][1]), refs[1]), index)
            migrated = migrate._ple_assets(repo, fold, PLE_BLOCK_BYTES, where)
        direct = ple_assets(
            storage=ple.storage,
            tokens=ple.tokens,
            layers=ple.layers,
            dim=ple.dim,
            embed_scale=ple.embed_scale,
            read_values=lambda begin, end: ple.payloads["values"][begin:end],
            read_scales=lambda begin, end: ple.payloads["scales"][begin:end],
            block_bytes=PLE_BLOCK_BYTES,
        )
        return migrated, direct

    @pytest.mark.parametrize("kind", ["rope_base", "ple", "ple-sidecar"])
    def test_both_routes_place_the_same_assets(self, tmp_path: Path, kind: str) -> None:
        """A container with bundled assets is byte-identical on both paths too (the contract's
        "weights + scales + const + one asset").

        `rope_base` is an asset in an ordinary part, while the PLE is an asset with **1 block = 1
        part** (`dedicated_part`), with different placement rules — both are exercised. The PLE's
        location can be looked up in two ways (the old manifest's `assets` / the file name the
        index gives), so **both** are compared with the direct write — the block id of an asset is
        decided by the order of the dict the caller passes (`_plan_asset_parts` does not reorder),
        so an order that differs per path gives "a container with the same values and different
        bytes".
        """
        migrated_assets, direct_assets = self._asset_material(tmp_path, kind)
        graph, tensors, scales, _ = self._material("f32")
        stored = stored_model(graph, tensors, weight_dtype="f32", weight_scales=scales)
        left = publish_container(
            tmp_path / "left" / "model.f32.krm",
            stored.graph,
            stored.tensors,
            stored.bindings,
            graph_name="enc",
            provenance=PROVENANCE,
            assets=migrated_assets,
        )
        right = publish_container(
            tmp_path / "right" / "model.f32.krm",
            stored.graph,
            stored.tensors,
            stored.bindings,
            graph_name="enc",
            provenance=PROVENANCE,
            assets=direct_assets,
        )

        assert [path.read_bytes() for path in left.parts] == [
            path.read_bytes() for path in right.parts
        ]
        # The counterpart showing the assets are really there (not comparing two empty `assets`).
        assert sorted(migrated_assets) == sorted(direct_assets)
        assert migrated_assets

    def test_the_asset_gate_is_not_vacuous(self, tmp_path: Path) -> None:
        """The check against a tautological test — a one-byte difference in an asset payload moves
        the part sequence."""
        migrated_assets, _ = self._asset_material(tmp_path, "rope_base")
        graph, tensors, scales, _ = self._material("f32")
        stored = stored_model(graph, tensors, weight_dtype="f32", weight_scales=scales)
        original = publish_container(
            tmp_path / "a" / "model.f32.krm",
            stored.graph,
            stored.tensors,
            stored.bindings,
            graph_name="enc",
            provenance=PROVENANCE,
            assets=migrated_assets,
        )
        payload = bytearray(migrated_assets[ROPE_BASE_ASSET].payload())
        payload[-1] ^= 0xFF
        flipped = publish_container(
            tmp_path / "b" / "model.f32.krm",
            stored.graph,
            stored.tensors,
            stored.bindings,
            graph_name="enc",
            provenance=PROVENANCE,
            assets={ROPE_BASE_ASSET: AssetInput(ROPE_BASE_ROLE, len(payload), bytes(payload))},
        )

        assert [path.read_bytes() for path in original.parts] != [
            path.read_bytes() for path in flipped.parts
        ]

    def test_the_gate_is_not_vacuous(self, tmp_path: Path) -> None:
        """The check against a tautological test — a one-character difference in the provenance
        moves the bytes of part 0.

        The counterpart showing the comparison looks at **actual bytes**, not at "both paths called
        the same function".
        """
        graph, tensors, scales, _ = self._material("f32")
        stored = stored_model(graph, tensors, weight_dtype="f32", weight_scales=scales)
        other = publish_container(
            tmp_path / "other" / "model.f32.krm",
            stored.graph,
            stored.tensors,
            stored.bindings,
            graph_name="enc",
            provenance=Provenance(license="mit", writer=PROVENANCE.writer),
        )

        assert [path.read_bytes() for path in other.parts] != self._migrated(tmp_path, "f32")


class TestTheTensorPieces:
    """Folds the tensor pieces of old shard spec v3 (`<parent name>#NNNNN-of-NNNNN`) into one parent
    (ADR 0090).

    The migrator's self-check compares "folded results" with each other (the counterpart is the
    output of the same `read_component`), so a folding error cancels out there and stays invisible.
    The normal cases are pinned against **a directly written container**, the error cases by the
    `LegacyFormatError` message for each branch of old reader contract 5.
    """

    PARENT = "piece.weight"

    def test_a_piece_series_migrates_to_the_same_container_as_a_direct_write(
        self, tmp_path: Path
    ) -> None:
        shards = legacy_piece_shards(
            parent=self.PARENT, pieces=[(0, 1), (1, 3), (3, 4)], mark="piece"
        )
        source = stage_shards(tmp_path / "old" / "enc", "model.f32.safetensors", shards)
        migrated = migrate_component(source, tmp_path / "migrated", provenance=PROVENANCE)
        graph, tensors, scales, overrides = fixture_spec("piece", "f32")
        stored = stored_model(
            graph,
            tensors,
            weight_dtype="f32",
            weight_scales=scales,
            weight_dtype_overrides=overrides,
        )
        direct = publish_container(
            tmp_path / "direct" / "model.f32.krm",
            stored.graph,
            stored.tensors,
            stored.bindings,
            graph_name="enc",
            provenance=PROVENANCE,
        )

        # The raw bytes before splitting (read directly from the old form without pieces — a path
        # separate from the migrator).
        unsplit = source_material(legacy_shards(mark="piece"))[1][self.PARENT]
        assert migrated_payloads(list(migrated.parts))[self.PARENT] == unsplit
        assert [path.read_bytes() for path in migrated.parts] == [
            path.read_bytes() for path in direct.parts
        ]

    #: A parent of 4 rows × 32 bytes (rows are a multiple of 4 bytes).
    WHOLE = Entry("w", "F32", (4, 8), bytes(range(128)))

    @staticmethod
    def _read(tmp_path: Path, weight_shards: list[list[Entry]]):
        """Places the graph shard + `weight_shards` (the tensors of each shard) and runs them
        through the old reader."""
        shards = [
            legacy_shards(mark="piece")[0],
            *(write_safetensors(order(entries), {}) for entries in weight_shards),
        ]
        return read_component(
            resolve_shards(stage_shards(tmp_path, "model.f32.safetensors", shards))
        )

    def test_the_readers_join_is_the_unsplit_tensor(self, tmp_path: Path) -> None:
        """Counterpart — a valid placement folds into one parent (the error cases below are not
        "always fail")."""
        first, second = piece_entries(self.WHOLE, [(0, 1), (1, 4)])

        _, stored = self._read(tmp_path, [[first], [second]])

        joined = stored["w"]
        assert (joined.entry.dtype, joined.entry.shape, joined.entry.nbytes) == ("F32", (4, 8), 128)
        assert b"".join(payload_chunks(joined)) == self.WHOLE.payload

    def test_a_key_numbered_with_full_width_digits_is_not_a_piece(self) -> None:
        """The sequence numbers are ASCII digits — Python's `\\d` also matches U+FF11 (full-width
        one), and `int()` would read such a key as piece 1 of 2."""
        assert parse_piece_key("w#00001-of-00002") == ("w", 1, 2)
        assert parse_piece_key("w#0000１-of-00002") is None
        assert parse_piece_key("w#00001-of-0000２") is None

    def test_two_pieces_of_one_parent_in_one_shard_fail_loudly(self, tmp_path: Path) -> None:
        first, second = piece_entries(self.WHOLE, [(0, 2), (2, 4)])

        with pytest.raises(
            LegacyFormatError, match=r"shard\[1\] holds two pieces of the same parent"
        ):
            self._read(tmp_path, [[first, second]])

    def test_a_missing_piece_fails_loudly(self, tmp_path: Path) -> None:
        first, second, _ = piece_entries(self.WHOLE, [(0, 1), (1, 2), (2, 4)])

        with pytest.raises(LegacyFormatError, match="2 pieces do not match the declared total 3"):
            self._read(tmp_path, [[first], [second]])

    def test_pieces_that_disagree_on_the_total_fail_loudly(self, tmp_path: Path) -> None:
        first, second = piece_entries(self.WHOLE, [(0, 2), (2, 4)])
        second = replace(second, name="w#00002-of-00003")

        with pytest.raises(LegacyFormatError, match="the piece totals disagree, 2 and 3"):
            self._read(tmp_path, [[first], [second]])

    def test_pieces_out_of_shard_order_fail_loudly(self, tmp_path: Path) -> None:
        first, second = piece_entries(self.WHOLE, [(0, 2), (2, 4)])

        with pytest.raises(LegacyFormatError, match="piece 1 in shard order has index 2"):
            self._read(tmp_path, [[second], [first]])

    def test_pieces_in_non_consecutive_shards_fail_loudly(self, tmp_path: Path) -> None:
        first, second = piece_entries(self.WHOLE, [(0, 2), (2, 4)])
        filler = Entry("filler", "F32", (1,), bytes(4))

        with pytest.raises(
            LegacyFormatError, match=r"piece 2 is in shard\[3\], the previous piece in shard\[1\]"
        ):
            self._read(tmp_path, [[first], [filler], [second]])

    def test_a_piece_with_another_dtype_fails_loudly(self, tmp_path: Path) -> None:
        first, second = piece_entries(self.WHOLE, [(0, 2), (2, 4)])

        with pytest.raises(
            LegacyFormatError, match=r"the dtype of piece 2 is I32 \(piece 1 is F32\)"
        ):
            self._read(tmp_path, [[first], [replace(second, dtype="I32")]])

    def test_a_piece_with_other_trailing_dims_fails_loudly(self, tmp_path: Path) -> None:
        first, second = piece_entries(self.WHOLE, [(0, 2), (2, 4)])

        with pytest.raises(
            LegacyFormatError,
            match=r"piece 2 has remaining dimensions \[4\], which differ from piece 1's \[8\]",
        ):
            self._read(tmp_path, [[first], [replace(second, shape=(4, 4))]])

    def test_a_piece_without_rows_fails_loudly(self, tmp_path: Path) -> None:
        first, empty, last = piece_entries(self.WHOLE, [(0, 2), (2, 2), (2, 4)])

        with pytest.raises(LegacyFormatError, match="piece 2 has fewer than 1 row"):
            self._read(tmp_path, [[first], [empty], [last]])

    def test_a_non_final_piece_that_is_not_a_multiple_of_4_bytes_fails_loudly(
        self, tmp_path: Path
    ) -> None:
        first, second = piece_entries(Entry("v", "I8", (4, 3), bytes(12)), [(0, 1), (1, 4)])

        with pytest.raises(
            LegacyFormatError, match="non-final piece 1 is 3 bytes, not a multiple of 4"
        ):
            self._read(tmp_path, [[first], [second]])

    def test_a_parent_that_is_both_whole_and_split_fails_loudly(self, tmp_path: Path) -> None:
        first, second = piece_entries(self.WHOLE, [(0, 2), (2, 4)])

        with pytest.raises(LegacyFormatError, match="both whole and as pieces"):
            self._read(tmp_path, [[self.WHOLE, first], [second]])


def stage_doctored(
    directory: Path, shards: list[bytes], name: str, *, dtype: str | None = None, shape=None
) -> Path:
    """Places the old single-file form and rewrites only the declaration (dtype / shape) of the
    tensor `name`.

    Not a single byte changes — the old shard reader's checks (declared length match, alignment)
    pass, so only the comparison with the IR declaration can catch the mismatch.
    """
    directory.mkdir(parents=True, exist_ok=True)
    metadata, tensors = source_material(shards)
    specs: dict[str, tuple[str, tuple[int, ...]]] = {}
    for blob in shards:
        length = struct.unpack("<Q", blob[:8])[0]
        for key, spec in json.loads(blob[8 : 8 + length]).items():
            if key != "__metadata__":
                specs[key] = (spec["dtype"], tuple(spec["shape"]))
    assert name in specs, f"the subject has no '{name}'"
    old_dtype, old_shape = specs[name]
    specs[name] = (dtype or old_dtype, tuple(shape) if shape is not None else old_shape)
    entries = order(
        [
            Entry(name=key, dtype=specs[key][0], shape=specs[key][1], payload=raw)
            for key, raw in tensors.items()
        ]
    )
    target = directory / COMPONENT
    target.write_bytes(write_safetensors(entries, metadata))
    return target


class TestTheStoredDeclarationsMatchTheIr:
    """Checks the old safetensors' dtype / shape against the IR declaration (the old writer's rules)
    before writing (R7-003).

    The old writer mapped storage dtype → safetensors dtype (f32 → F32 / i8 → I8 / i4 → I4 …) one
    to one and wrote shapes in **logical form**. Without the check, a different type of the same
    byte length (an I32 1 read as f32 is about 1.4e-45) or a different shape with only the same
    element count would be silently written into the new container.
    """

    @pytest.mark.parametrize(
        ("name", "override", "message"),
        [
            ("x.karume_fixture_seed0", {"dtype": "I32"}, r"dtype I32, which differs from.*F32"),
            ("x.weight", {"shape": (8, 16)}, r"shape \[8, 16\], which differs from.*\[4, 32\]"),
            ("karume.scale.x.weight", {"dtype": "I32"}, r"dtype I32, which differs from.*F32"),
        ],
        ids=["same-bytes-other-dtype", "same-numel-other-shape", "integer-scale"],
    )
    def test_a_declaration_that_disagrees_with_the_ir_writes_nothing(
        self, tmp_path: Path, name: str, override: dict, message: str
    ) -> None:
        source = stage_doctored(
            tmp_path / "src", legacy_shards(mark="x", storage="i8"), name, **override
        )
        out = tmp_path / "out"

        with pytest.raises(MigrateError, match=message):
            migrate_component(source, out, provenance=PROVENANCE)
        assert not out.exists() or not any(out.iterdir())

    def test_a_storage_the_old_writer_never_wrote_fails_loudly(self) -> None:
        """The old writer could not write bf16 — even with matching declaration and data it is not
        accepted as an old distribution."""
        graph = TestTheDerivedBindings._graph(IrStorage(dtype="bf16"), [4, 32])
        sources = {"w": SourceTensor(StoredEntry("w", "BF16", (4, 32), 256), ())}

        with pytest.raises(MigrateError, match="a dtype the old writer did not write"):
            migrate._assert_stored_declarations(graph, sources)

    def test_the_undoctored_form_still_migrates(self, tmp_path: Path) -> None:
        """Counterpart — the rewriting scaffold itself does not break the migration (the failures
        above come from the injected mismatch)."""
        source = stage_doctored(tmp_path / "src", legacy_shards(mark="x", storage="i8"), "x.weight")

        result = migrate_component(source, tmp_path / "out", provenance=PROVENANCE)

        assert result.parts
