"""Writer and reader of the container format (`karume.container`) — docs/container-v1.md /
docs/ir-v2.md.

The source of truth for correctness is **the TS reader**, so this file checks "the rules that close
on the Python side alone": the spelling of canonical JSON, the renaming and ordering into IR v2,
how blocks / parts are packed, and the re-read of the self-check. "Can TS really open it" is covered
by the cross-language fixture ({@link TestTheCrossLanguageFixture} writes it,
`packages/runtime/tests/container_fixture_test.ts` opens it).

The ledger and the size constants **are held as the same table on both sides** (a cross-package
invariant, not a derivable fact), so their single comparison point also lives here
({@link TestTheSharedTables}).
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from itertools import count
from pathlib import Path
from typing import Any

import pytest
from conftest import RUNTIME_FIXTURES
from container_fixture import (
    FIXTURE_BLOCK_BYTES,
    FIXTURE_PART_BYTES,
    FIXTURE_PATH,
    GRAPH_NAME,
    SPLIT_FIXTURE_PARTS,
    SPLIT_SCALE_BLOCK_BYTES,
    SPLIT_SCALE_FIXTURE_PATH,
    SPLIT_SCALE_GRAPH_NAME,
    SPLIT_SCALE_PART_BYTES,
    fixture_write_requested,
    pattern_bytes,
    split_asset_payloads,
    split_fixture_paths,
    split_scale_bindings,
    split_scale_graph,
    split_scale_tensors,
    synthetic_bindings,
    synthetic_graph,
    synthetic_tensors,
    write_split_container,
    write_split_scale_container,
    write_synthetic_container,
)

import karume.container as container_module
from karume.container import (
    BLOCK_MAX_BYTES,
    BLOCK_START_ALIGN,
    BLOCK_TAIL_ALIGN,
    CODEC_FOR_STORAGE,
    CODEC_LEDGER,
    HEADER_BYTES,
    MAX_DESCRIPTOR_BYTES,
    MAX_JSON_DEPTH,
    MIN_GROUP_SIZE,
    PART_LENGTH_CHOICES,
    AssetInput,
    ContainerFormatError,
    Encoding,
    Provenance,
    align_up,
    base_path,
    canonical_json,
    ir_v2_document,
    joined_scale_id,
    read_container,
    read_descriptor_refs,
    read_header,
    resolve_sequence,
    sequence_siblings,
    serialize_graph_descriptor,
    serialize_model_descriptor,
    write_graph_container,
    write_header,
    write_model_container,
)
from karume.ir import IrGraph, IrInitializer, IrInput, IrNode, IrShared, IrStorage, IrValue
from karume.publish import publish_container
from karume.verify import STORAGE_DTYPES, verify_container

#: Location of the TS ledger and size constants (the other side of the comparison).
CODECS_TS = RUNTIME_FIXTURES.parents[1] / "src" / "format" / "container" / "codecs.ts"
LIMITS_TS = RUNTIME_FIXTURES.parents[1] / "src" / "format" / "container" / "limits.ts"


def write_synthetic(tmp_path: Path, *, single: bool) -> list[Path]:
    return write_synthetic_container(tmp_path / "synthetic.krm", single=single)


def rewrite_part0(part0: Path, *, graph: bytes | None = None, model: bytes | None = None) -> None:
    """Replace the documents of a split-form part 0 (header + two documents only) and rewrite the
    header lengths to match.

    With the lengths in line, a broken document does not turn into "a wrong part 0 length" and
    takes only the targeted check branch.
    """
    raw = part0.read_bytes()
    header = read_header(raw)
    graph_bytes = raw[HEADER_BYTES : HEADER_BYTES + header.graph_length] if graph is None else graph
    model_bytes = (
        raw[HEADER_BYTES + header.graph_length : header.part0_length] if model is None else model
    )
    part0.write_bytes(
        write_header(kind=header.kind, graph_length=len(graph_bytes), model_length=len(model_bytes))
        + graph_bytes
        + model_bytes
    )


def edit_model_document(written: list[Path], edit: Callable[[dict[str, Any]], None]) -> None:
    """Rewrite the split form's model descriptor as JSON (respelled as canonical JSON)."""
    document = json.loads(read_container(written).model_descriptor_bytes)
    edit(document)
    rewrite_part0(written[0], model=canonical_json(document).encode())


class TestCanonicalJson:
    """Numbers are spelled as ECMAScript `Number::toString` (docs/ir-v2.md "Canonical
    serialization")."""

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (0.000001, "0.000001"),
            (1e-7, "1e-7"),
            (10000.0, "10000"),
            (1e21, "1e+21"),
            (1e20, "100000000000000000000"),
            (-0.0, "0"),
            (0.5, "0.5"),
            (1e-12, "1e-12"),
            (0.2973017692565918, "0.2973017692565918"),
            (123456789012345680000, "123456789012345680000"),
            # Edges: the double maximum and the subnormal minimum (both spelled in exponent
            # notation).
            (1.7976931348623157e308, "1.7976931348623157e+308"),
            (5e-324, "5e-324"),
            # An integral float has no decimal point (`repr` returns `10000.0`).
            (-1024.0, "-1024"),
        ],
    )
    def test_a_number_is_spelled_like_ecmascript(self, value: float, expected: str) -> None:
        assert canonical_json(value) == expected

    def test_the_document_has_no_whitespace_and_keeps_the_declared_key_order(self) -> None:
        """Ordering is the assembler's job — the serializer spells the declared order as is."""
        assert canonical_json({"b": 1, "a": [True, False, None]}) == '{"b":1,"a":[true,false,null]}'

    def test_non_ascii_stays_raw_and_control_characters_are_escaped(self) -> None:
        assert canonical_json("\u3042\n\u0001") == '"\u3042\\n\\u0001"'

    @pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
    def test_a_non_finite_number_is_refused(self, value: float) -> None:
        with pytest.raises(ContainerFormatError, match="non-finite number"):
            canonical_json(value)

    def test_a_non_string_key_is_refused(self) -> None:
        with pytest.raises(ContainerFormatError, match="key is not a string"):
            canonical_json({1: 2})


def _minimal_graph(**overrides: object) -> IrGraph:
    """Minimal graph with one initializer and one node (subject of the rename and order checks)."""
    graph = IrGraph(
        symbols=["T"],
        inputs=[IrInput(name="x", dtype="f32", shape=["T", 4])],
        outputs=["p_w"],
        initializers={
            "p_w": IrInitializer(tensor="b.weight", storage=IrStorage(dtype="f32")),
            "p_a": IrInitializer(tensor="a.weight", storage=IrStorage(dtype="f32")),
        },
        values={
            "p_w": IrValue(dtype="f32", shape=[4, 4]),
            "p_a": IrValue(dtype="f32", shape=[4, 4]),
            "y": IrValue(dtype="f32", shape=["T", 4]),
        },
        nodes=[IrNode(op="matmul", ins=["x", "p_a"], outs=["y"], attrs={"b": 1, "a": 2})],
    )
    for key, value in overrides.items():
        setattr(graph, key, value)
    return graph


class TestIrV2Document:
    def test_an_initializer_is_renamed_to_its_tensor_key_everywhere(self) -> None:
        graph = _minimal_graph()
        document = ir_v2_document(graph)

        assert set(document["initializers"]) == {"a.weight", "b.weight"}
        assert set(document["values"]) == {"a.weight", "b.weight", "y"}
        assert document["nodes"][0]["ins"] == ["x", "a.weight"]
        assert document["outputs"] == ["b.weight"]

    def test_storage_is_dropped_and_shared_becomes_a_flag(self) -> None:
        graph = _minimal_graph()
        graph.initializers["p_s"] = IrInitializer(
            shared=IrShared(tensor="lm_head.weight"), storage=IrStorage(dtype="f16")
        )
        graph.values["p_s"] = IrValue(dtype="f32", shape=[4, 4])
        document = ir_v2_document(graph)

        assert document["initializers"]["a.weight"] == {}
        assert document["initializers"]["lm_head.weight"] == {"shared": True}

    def test_the_document_is_version_2_and_name_maps_are_in_code_point_order(self) -> None:
        document = ir_v2_document(_minimal_graph())
        text = canonical_json(document)

        assert document["version"] == 2
        assert list(document["initializers"]) == ["a.weight", "b.weight"]
        assert list(document["values"]) == ["a.weight", "b.weight", "y"]
        # attrs are in code point order recursively too (free-form values, so the assembler
        # canonicalizes them).
        assert '"attrs":{"a":2,"b":1}' in text
        # Arrays keep the declared order (fields where order has meaning).
        assert '"ins":["x","a.weight"]' in text

    def test_digit_only_names_follow_code_point_order_like_the_ts_writer(self) -> None:
        """Digit-only names are in code point order too ("10" < "2").

        Matches the fixture shared with the TS writer byte for byte.
        """
        emoji = "\U0001f600"
        tensors = ["10", "2", emoji, "0"]
        chain = ["x", "a1", "a2", "a3", "y"]
        # From the initializer names before renaming (p_*) to the real keys.
        # Nodes are added in the order "0" → "2" → "10" → emoji.
        order = ["0", "2", "10", emoji]
        graph = IrGraph(
            symbols=[],
            inputs=[IrInput(name="x", dtype="f32", shape=[4])],
            outputs=["y"],
            initializers={
                f"p_{index}": IrInitializer(tensor=tensor, storage=IrStorage(dtype="f32"))
                for index, tensor in enumerate(tensors)
            },
            values={
                **{f"p_{index}": IrValue(dtype="f32", shape=[4]) for index in range(len(tensors))},
                **{name: IrValue(dtype="f32", shape=[4]) for name in chain[1:]},
            },
            nodes=[
                IrNode(
                    op="add",
                    ins=[chain[step], f"p_{tensors.index(tensor)}"],
                    outs=[chain[step + 1]],
                    attrs={"10": 1, "2": 2, "a": {"10": True, "9": False}} if step == 0 else {},
                )
                for step, tensor in enumerate(order)
            ],
        )
        expected = (RUNTIME_FIXTURES / "ir" / "canonical-name-order.txt").read_text("utf-8")

        assert canonical_json(ir_v2_document(graph)) == expected

    def test_an_empty_states_section_is_not_written(self) -> None:
        document = ir_v2_document(_minimal_graph())

        assert "states" not in document
        assert "states" not in document["nodes"][0]

    def test_a_rename_that_collides_with_a_value_name_fails_loudly(self) -> None:
        graph = _minimal_graph()
        graph.initializers["p_w"] = IrInitializer(tensor="y", storage=IrStorage(dtype="f32"))

        with pytest.raises(ContainerFormatError, match="existing value name"):
            ir_v2_document(graph)

    def test_two_initializers_that_rename_to_the_same_key_fail_loudly(self) -> None:
        graph = _minimal_graph()
        graph.initializers["p_w"] = IrInitializer(tensor="a.weight", storage=IrStorage(dtype="f32"))

        with pytest.raises(ContainerFormatError, match="rename target"):
            ir_v2_document(graph)

    def test_an_initializer_without_a_tensor_key_fails_loudly(self) -> None:
        graph = _minimal_graph()
        graph.initializers["p_w"] = IrInitializer(storage=IrStorage(dtype="f32"))

        with pytest.raises(ContainerFormatError, match="invalid as IR v1"):
            ir_v2_document(graph)


def _parse_ts_ledger(source: str) -> dict[str, tuple[str, int, int, int, str, str | None]]:
    """Read the TS `CODEC_LEDGER` (registered name → layout + 3 packing values + scale + grouping).

    It is read with regular expressions because the other side is **immutable data in the
    repository** (a form importable at run time would make the Python tests require a TypeScript
    runtime). If the TS spelling moves, this fails.
    """
    raw_align = re.search(r"const RAW = \(.*?alignBytes: (\d+)", source, re.S)
    raw_elements = re.search(r"const RAW = \(.*?blockElements: (\d+)", source, re.S)
    assert raw_align is not None and raw_elements is not None, "cannot read the RAW helper"
    entries: dict[str, tuple[str, int, int, int, str, str | None]] = {}
    for name, layout, block_bytes in re.findall(
        r'\["([\w-]+)", RAW\("([\w-]+)", (\d+)\)\]', source
    ):
        entries[name] = (
            layout,
            int(raw_elements.group(1)),
            int(block_bytes),
            int(raw_align.group(1)),
            "forbidden",
            None,
        )
    # Oxfmt (2026-10-16) splits long entries over several lines "[ / name / { … } / ]", so
    # whitespace and newlines are allowed between the name and \`{\` and between \`}\` and \`]\`
    # (RAW lines stay on one line).
    for name, body in re.findall(r'\[\s*"([\w-]+)",\s*\{(.*?)\n\s*\},?\s*\]', source, re.S):
        numbers = {
            key: int(re.search(rf"{key}: (\d+)", body).group(1))  # type: ignore[union-attr]
            for key in ("blockElements", "blockBytes", "alignBytes")
        }
        layout_match = re.search(r'layout: "([\w-]+)"', body)
        scale = re.search(r'scale: "(\w+)"', body)
        grouping = re.search(r'grouping: "?(\w+)"?', body)
        assert layout_match is not None, f"{name}: cannot read layout"
        assert scale is not None and grouping is not None, f"{name}: cannot read scale / grouping"
        entries[name] = (
            layout_match.group(1),
            numbers["blockElements"],
            numbers["blockBytes"],
            numbers["alignBytes"],
            scale.group(1),
            None if grouping.group(1) == "undefined" else grouping.group(1),
        )
    return entries


class TestTheSharedTables:
    """Both sides hold **the same values** for the ledger and size constants (they are written into
    assets, so one side cannot move alone)."""

    def test_the_codec_ledger_matches_the_runtime_one(self) -> None:
        ours = {
            name: (
                entry.layout,
                entry.packing.block_elements,
                entry.packing.block_bytes,
                entry.packing.align_bytes,
                entry.scale,
                entry.grouping,
            )
            for name, entry in CODEC_LEDGER.items()
        }

        assert _parse_ts_ledger(CODECS_TS.read_text(encoding="utf-8")) == ours

    def test_every_legacy_storage_dtype_maps_to_a_registered_codec(self) -> None:
        """Every kind of old `storage.dtype` has a target (the migration CLI does not silently drop
        a slot)."""
        assert set(CODEC_FOR_STORAGE) == set(STORAGE_DTYPES)
        assert set(CODEC_FOR_STORAGE.values()) <= set(CODEC_LEDGER)
        # The claim of being ternary is the quantizer's to make — old i2 maps only to `int2-off`
        # (§6.3).
        assert "ternary" not in CODEC_FOR_STORAGE.values()

    def test_the_group_size_floor_matches(self) -> None:
        source = CODECS_TS.read_text(encoding="utf-8")

        assert f"MIN_GROUP_SIZE = {MIN_GROUP_SIZE};" in source

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("HEADER_BYTES", HEADER_BYTES),
            ("BLOCK_START_ALIGN", BLOCK_START_ALIGN),
            ("BLOCK_TAIL_ALIGN", BLOCK_TAIL_ALIGN),
        ],
    )
    def test_a_layout_constant_matches_the_runtime_one(self, name: str, expected: int) -> None:
        source = LIMITS_TS.read_text(encoding="utf-8")

        assert f"export const {name} = {expected};" in source

    def test_the_block_and_part_limits_match(self) -> None:
        source = LIMITS_TS.read_text(encoding="utf-8")
        mib = BLOCK_MAX_BYTES // (1024 * 1024)
        choices = ", ".join(f"{value // (1024 * 1024)} * MIB" for value in PART_LENGTH_CHOICES)

        assert f"export const BLOCK_MAX_BYTES = {mib} * MIB;" in source
        assert f"PART_LENGTH_CHOICES: readonly number[] = [{choices}]" in source


class TestWriteAndReadBack:
    """Re-read what was written (the same discipline as invariant 5 of §12 — never produce
    something that was written but cannot be read)."""

    def test_a_written_container_reads_back_and_reserializes_to_the_same_bytes(
        self, tmp_path: Path
    ) -> None:
        written = write_synthetic(tmp_path, single=True)
        read = read_container(written)

        assert read.header.kind == "model"
        assert serialize_graph_descriptor(read.graph) == read.graph_descriptor_bytes
        assert read.model is not None
        assert serialize_model_descriptor(read.model) == read.model_descriptor_bytes

    def test_every_block_and_part_matches_its_declared_sha256(self, tmp_path: Path) -> None:
        read = read_container(write_synthetic(tmp_path, single=True))

        assert sorted(read.verify_blocks()) == sorted(read.block_ids)
        read.verify_parts()

    def test_the_same_input_produces_the_same_bytes(self, tmp_path: Path) -> None:
        first = write_synthetic(tmp_path / "a", single=True)[0].read_bytes()
        second = write_synthetic(tmp_path / "b", single=True)[0].read_bytes()

        assert first == second

    def test_the_single_and_split_forms_share_the_descriptor_bytes(self, tmp_path: Path) -> None:
        single = read_container(write_synthetic(tmp_path / "single", single=True))
        split = read_container(write_synthetic(tmp_path / "split", single=False))

        assert split.graph_descriptor_bytes == single.graph_descriptor_bytes
        assert split.model_descriptor_bytes == single.model_descriptor_bytes
        assert [split.block(name) for name in split.block_ids] == [
            single.block(name) for name in single.block_ids
        ]

    def test_the_split_form_is_named_like_a_shard_series(self, tmp_path: Path) -> None:
        written = write_synthetic(tmp_path, single=False)

        assert [path.name for path in written] == [
            f"synthetic-{index + 1:05d}-of-{len(written):05d}.krm" for index in range(len(written))
        ]

    def test_the_first_part_holds_only_the_header_and_the_two_documents(
        self, tmp_path: Path
    ) -> None:
        written = write_synthetic(tmp_path, single=False)
        read = read_container(written)

        assert written[0].stat().st_size == (
            HEADER_BYTES + len(read.graph_descriptor_bytes) + len(read.model_descriptor_bytes)
        )

    def test_pieces_cover_the_rows_and_the_scale_stays_with_the_first_piece(
        self, tmp_path: Path
    ) -> None:
        read = read_container(write_synthetic(tmp_path, single=True))
        assert read.model is not None
        supply = read.model.binding[GRAPH_NAME]["big.weight"]
        blocks = {block.id: block for block in read.model.blocks}
        assert supply.pieces is not None

        assert [rows for _, rows in supply.pieces] == [(0, 8), (8, 16)]
        assert all(blocks[block].length == 256 for block, _ in supply.pieces)
        assert supply.encoding.scale_block is not None
        # Rule ③: the companion scale is in the same part as piece 1.
        assert blocks[supply.encoding.scale_block].part == blocks[supply.pieces[0][0]].part

    def test_a_tail_pad_is_baked_by_the_writer(self, tmp_path: Path) -> None:
        read = read_container(write_synthetic(tmp_path, single=True))
        assert read.model is not None
        supply = read.model.binding[GRAPH_NAME]["dec.weight"]
        block = next(record for record in read.model.blocks if record.id == supply.block)

        assert block.length == 44
        assert read.block(block.id)[42:] == b"\x00\x00"

    def test_a_row_axis_1_scale_is_one_value_per_channel(self, tmp_path: Path) -> None:
        read = read_container(write_synthetic(tmp_path, single=True))
        assert read.model is not None
        encoding = read.model.binding[GRAPH_NAME]["conv.weight"].encoding
        blocks = {block.id: block for block in read.model.blocks}
        assert encoding.scale_block is not None

        assert encoding.row_axis == 1
        assert encoding.group_size == 32
        assert blocks[encoding.scale_block].length == 3 * 4

    def test_a_flipped_byte_is_caught_by_the_block_sha256(self, tmp_path: Path) -> None:
        written = write_synthetic(tmp_path, single=True)
        read = read_container(written)
        target = next(record for record in read.model.blocks if record.role == "weight")  # type: ignore[union-attr]
        raw = bytearray(written[0].read_bytes())
        offset = raw.find(read.block(target.id))
        raw[offset] ^= 0xFF
        written[0].write_bytes(bytes(raw))

        with pytest.raises(
            ContainerFormatError, match=r"sha256 of block .* differs from the declaration"
        ):
            read_container(written).block(target.id)

    def test_a_truncated_part_is_caught_before_any_block_is_read(self, tmp_path: Path) -> None:
        written = write_synthetic(tmp_path, single=False)
        written[-1].write_bytes(written[-1].read_bytes()[:-4])

        with pytest.raises(ContainerFormatError, match="differs from the declared"):
            read_container(written)


class TestTheGraphContainer:
    """A `krg` is extracted from a `krm` by byte copy (§9 — the condition for judging identity by
    content hash)."""

    def test_the_extracted_krg_equals_a_directly_written_krg(self, tmp_path: Path) -> None:
        read = read_container(write_synthetic(tmp_path, single=True))
        bindings = synthetic_bindings()
        tensors = synthetic_tensors()
        direct = write_graph_container(
            tmp_path / "synthetic.krg",
            synthetic_graph(),
            tensors,
            {name: bindings[name] for name in bindings if name.startswith("const.")},
            graph_name=GRAPH_NAME,
            block_bytes=FIXTURE_BLOCK_BYTES,
        )

        assert read.extract_graph() == direct.read_bytes()

    def test_the_krg_reads_back_as_a_graph_container(self, tmp_path: Path) -> None:
        bindings = synthetic_bindings()
        path = write_graph_container(
            tmp_path / "synthetic.krg",
            synthetic_graph(),
            synthetic_tensors(),
            {name: bindings[name] for name in bindings if name.startswith("const.")},
            graph_name=GRAPH_NAME,
            block_bytes=FIXTURE_BLOCK_BYTES,
        )
        read = read_container([path])

        assert read.header.kind == "graph"
        assert read.model is None
        assert sorted(read.verify_blocks()) == sorted(read.block_ids)

    def test_a_weight_binding_is_refused_for_a_krg(self, tmp_path: Path) -> None:
        with pytest.raises(ContainerFormatError, match="surplus"):
            write_graph_container(
                tmp_path / "synthetic.krg",
                synthetic_graph(),
                synthetic_tensors(),
                synthetic_bindings(),
                graph_name=GRAPH_NAME,
                block_bytes=FIXTURE_BLOCK_BYTES,
            )

    def test_an_empty_const_region_still_extracts(self, tmp_path: Path) -> None:
        """No padding before a zero-length part (§3 — padding would shift the extraction result)."""
        graph = _minimal_graph()
        bindings = {"a.weight": Encoding("f32"), "b.weight": Encoding("f32")}
        tensors = {"a.weight": pattern_bytes(64, 5), "b.weight": pattern_bytes(64, 6)}
        written = write_model_container(
            tmp_path / "m.krm",
            graph,
            tensors,
            bindings,
            graph_name="minimal",
            provenance=Provenance(license="mit"),
            part_bytes=1024,
            single=True,
        )
        direct = write_graph_container(tmp_path / "m.krg", graph, {}, {}, graph_name="minimal")

        assert read_container(written).extract_graph() == direct.read_bytes()


class TestTheWriterRefusesWhatItCannotRepresent:
    def test_a_missing_binding_is_reported_with_the_name(self, tmp_path: Path) -> None:
        bindings = synthetic_bindings()
        del bindings["dec.weight"]

        with pytest.raises(ContainerFormatError, match=r"missing \[dec.weight\]"):
            write_model_container(
                tmp_path / "m.krm",
                synthetic_graph(),
                synthetic_tensors(),
                bindings,
                graph_name=GRAPH_NAME,
                provenance=Provenance(license="mit"),
            )

    def test_a_const_that_exceeds_the_block_limit_fails_loudly(self, tmp_path: Path) -> None:
        """The const region has no piece-splitting mechanism (§3 — the writer stops instead of
        opening an escape hatch)."""
        graph = _minimal_graph()
        graph.initializers["c"] = IrInitializer(
            tensor="const.0011223344556677", storage=IrStorage(dtype="f32")
        )
        graph.values["c"] = IrValue(dtype="f32", shape=[256])

        with pytest.raises(ContainerFormatError, match="must fit in one block"):
            write_model_container(
                tmp_path / "m.krm",
                graph,
                {
                    "a.weight": pattern_bytes(64, 5),
                    "b.weight": pattern_bytes(64, 6),
                    "const.0011223344556677": pattern_bytes(1024, 7),
                },
                {
                    "a.weight": Encoding("f32"),
                    "b.weight": Encoding("f32"),
                    "const.0011223344556677": Encoding("f32"),
                },
                graph_name="minimal",
                provenance=Provenance(license="mit"),
                block_bytes=512,
            )

    def test_a_row_axis_1_initializer_cannot_be_split(self, tmp_path: Path) -> None:
        with pytest.raises(ContainerFormatError, match="cannot be split into pieces"):
            write_model_container(
                tmp_path / "m.krm",
                synthetic_graph(),
                synthetic_tensors(),
                synthetic_bindings(),
                graph_name=GRAPH_NAME,
                provenance=Provenance(license="mit"),
                part_bytes=FIXTURE_PART_BYTES,
                # Lower the size until conv.weight (96 bytes, rowAxis 1) splits.
                block_bytes=64,
            )

    def test_a_payload_that_disagrees_with_the_declaration_fails_loudly(
        self, tmp_path: Path
    ) -> None:
        tensors = synthetic_tensors()
        tensors["dec.weight"] = pattern_bytes(40, 23)

        with pytest.raises(
            ContainerFormatError, match="differ from the 42 bytes decided by the declaration"
        ):
            write_model_container(
                tmp_path / "m.krm",
                synthetic_graph(),
                tensors,
                synthetic_bindings(),
                graph_name=GRAPH_NAME,
                provenance=Provenance(license="mit"),
                part_bytes=FIXTURE_PART_BYTES,
                block_bytes=FIXTURE_BLOCK_BYTES,
            )

    def test_a_group_size_that_is_not_a_power_of_two_fails_loudly(self, tmp_path: Path) -> None:
        bindings = synthetic_bindings()
        bindings["enc.weight"] = Encoding(
            "int4-sym-g", group_size=24, row_axis=0, scale_key="enc.weight_scale"
        )

        with pytest.raises(ContainerFormatError, match="power of 2"):
            write_model_container(
                tmp_path / "m.krm",
                synthetic_graph(),
                synthetic_tensors(),
                bindings,
                graph_name=GRAPH_NAME,
                provenance=Provenance(license="mit"),
            )


class TestTheProvenanceReader:
    """The three optional fields (notice / upstreamRevision / writer) also require a non-empty
    string, as in the TS reader.

    The publish re-read and `karume verify` go through this reader, so if it let them through, a
    container TS `openContainer` cannot open (`karume migrate --notice ""` and the like) would be
    installed.
    """

    def test_an_empty_notice_from_the_writer_is_refused_on_read_back(self, tmp_path: Path) -> None:
        written = write_model_container(
            tmp_path / "m.krm",
            synthetic_graph(),
            synthetic_tensors(),
            synthetic_bindings(),
            graph_name=GRAPH_NAME,
            provenance=Provenance(license="mit", notice=""),
            part_bytes=FIXTURE_PART_BYTES,
            block_bytes=FIXTURE_BLOCK_BYTES,
        )

        with pytest.raises(
            ContainerFormatError, match=r"provenance\.notice is not a non-empty string"
        ):
            read_container(written)

    @pytest.mark.parametrize(
        ("key", "value"),
        [("writer", 5), ("upstreamRevision", None), ("notice", None), ("writer", "")],
    )
    def test_an_optional_field_that_is_not_a_non_empty_string_is_refused(
        self, tmp_path: Path, key: str, value: object
    ) -> None:
        written = write_synthetic(tmp_path, single=False)
        edit_model_document(written, lambda document: document["provenance"].update({key: value}))

        with pytest.raises(
            ContainerFormatError, match=rf"provenance\.{key} is not a non-empty string"
        ):
            read_container(written)

    def test_all_three_fields_as_non_empty_strings_read_back(self, tmp_path: Path) -> None:
        present = {"notice": "NOTICE", "upstreamRevision": "0123abcd", "writer": "karume"}
        written = write_synthetic(tmp_path, single=False)
        edit_model_document(written, lambda document: document["provenance"].update(present))

        read = read_container(written)

        assert read.model is not None
        assert read.model.provenance == Provenance(
            license="apache-2.0", notice="NOTICE", upstream_revision="0123abcd", writer="karume"
        )


class TestTheAssetIntake:
    """Asset input (§2.2 / §4.2) — no piece splitting, no part shared with weights."""

    @staticmethod
    def _write(tmp_path: Path, assets: dict[str, AssetInput], **overrides: object) -> list[Path]:
        return write_model_container(
            tmp_path / "m.krm",
            _minimal_graph(),
            {"a.weight": pattern_bytes(64, 5), "b.weight": pattern_bytes(64, 6)},
            {"a.weight": Encoding("f32"), "b.weight": Encoding("f32")},
            graph_name="minimal",
            provenance=Provenance(license="mit"),
            assets=assets,
            part_bytes=1024,
            single=True,
            **overrides,  # type: ignore[arg-type]
        )

    def test_a_lazy_payload_writes_the_same_bytes_as_a_plain_one(self, tmp_path: Path) -> None:
        """A deferred call changes only "when it is pulled" (the bytes produced are the same)."""
        payload = pattern_bytes(37, 61)
        plain = self._write(tmp_path / "a", {"x": AssetInput("rope-base", 37, payload)})
        lazy = self._write(tmp_path / "b", {"x": AssetInput("rope-base", 37, lambda: payload)})

        assert lazy[0].read_bytes() == plain[0].read_bytes()

    def test_the_asset_blocks_never_share_a_part_with_weights(self, tmp_path: Path) -> None:
        read = read_container(self._write(tmp_path, {"x": AssetInput("rope-base", 8, b"\x01" * 8)}))
        assert read.model is not None
        blocks = {block.id: block for block in read.model.blocks}
        asset = blocks[read.model.assets["x"].block]

        assert asset.role == "asset"
        assert all(block.part != asset.part for block in blocks.values() if block.role != "asset")

    def test_the_declared_length_survives_into_the_model_descriptor(self, tmp_path: Path) -> None:
        """The logical length copies the declaration; the block length rounds it up to a multiple of
        4 (ADR 0109 decision 4)."""
        read = read_container(
            self._write(tmp_path, {"x": AssetInput("rope-base", 37, b"\x01" * 37)})
        )
        assert read.model is not None
        record = read.model.assets["x"]
        block = next(block for block in read.model.blocks if block.id == record.block)

        assert (record.role, record.length) == ("rope-base", 37)
        assert block.length == 40
        assert read.block(record.block) == b"\x01" * 37 + b"\x00" * 3

    def test_a_dedicated_asset_gets_a_part_of_its_own(self, tmp_path: Path) -> None:
        """An asset that needs range reads MUST sit alone in a dedicated part (container-v1 §4.2).

        A read-whole asset is passed **first** to also check that the dedicated part does not
        become "the continuation of the previous part" (looking only at the side that opens
        after it, this rule would pass green without checking anything).
        """
        read = read_container(
            self._write(
                tmp_path,
                {
                    "idx": AssetInput("ple-index", 8, b"\x03" * 8),
                    "v.0": AssetInput("ple-values", 8, b"\x01" * 8, dedicated_part=True),
                    "v.1": AssetInput("ple-values", 8, b"\x02" * 8, dedicated_part=True),
                    "tail": AssetInput("rope-base", 8, b"\x04" * 8),
                },
            )
        )
        assert read.model is not None
        blocks = {block.id: block for block in read.model.blocks}
        parts = {name: blocks[record.block].part for name, record in read.model.assets.items()}

        # The dedicated part holds exactly one block and is not shared with read-whole assets
        # either.
        assert len({parts["idx"], parts["v.0"], parts["v.1"], parts["tail"]}) == 4
        for name in ("v.0", "v.1"):
            assert [block.id for block in read.model.blocks if block.part == parts[name]] == [
                read.model.assets[name].block
            ]

    def test_the_physical_order_follows_the_call_order(self, tmp_path: Path) -> None:
        """Layout order is **the order passed in** (PLE arrives in token order) — not name order."""
        read = read_container(
            self._write(
                tmp_path,
                {
                    f"ple.values.{index}": AssetInput("ple-values", 8, bytes([index]) * 8)
                    for index in (0, 1, 2, 10, 11)
                },
            )
        )
        assert read.model is not None
        placed = sorted(
            (block.part, block.offset, block.id)
            for block in read.model.blocks
            if block.role == "asset"
        )

        assert [entry[2] for entry in placed] == ["a.0", "a.1", "a.2", "a.3", "a.4"]
        assert [read.model.assets[f"ple.values.{k}"].block for k in (0, 1, 2, 10, 11)] == [
            entry[2] for entry in placed
        ]

    def test_an_asset_that_exceeds_the_block_limit_fails_loudly(self, tmp_path: Path) -> None:
        """Assets have no piece-splitting mechanism (readers find rows through the role's index)."""
        with pytest.raises(ContainerFormatError, match="an asset must fit in one block"):
            self._write(
                tmp_path, {"x": AssetInput("ple-values", 300, b"\x00" * 300)}, block_bytes=256
            )

    def test_an_empty_asset_fails_loudly(self, tmp_path: Path) -> None:
        with pytest.raises(ContainerFormatError, match="a block of length 0 is not created"):
            self._write(tmp_path, {"x": AssetInput("ple-values", 0, b"")})

    def test_an_asset_name_that_collides_with_a_tensor_key_fails_loudly(
        self, tmp_path: Path
    ) -> None:
        """Allowing a collision would silently swap the block contents."""
        with pytest.raises(ContainerFormatError, match="collide with tensor keys"):
            self._write(tmp_path, {"a.weight": AssetInput("ple-values", 8, b"\x01" * 8)})

    def test_a_payload_that_differs_between_pulls_fails_loudly(self, tmp_path: Path) -> None:
        """MUST return the same bytes each time it is pulled (checked so that any number of pulls
        gives the same conclusion)."""
        pulls = count(8)

        with pytest.raises(
            ContainerFormatError, match="differ from the 8 bytes decided by the declaration"
        ):
            self._write(tmp_path, {"x": AssetInput("ple-values", 8, lambda: b"\x01" * next(pulls))})

    def test_a_payload_that_disagrees_with_the_declared_length_fails_loudly(
        self, tmp_path: Path
    ) -> None:
        """The declaration comes first (it decides the layout) — different contents fail."""
        with pytest.raises(
            ContainerFormatError, match="differ from the 8 bytes decided by the declaration"
        ):
            self._write(tmp_path, {"x": AssetInput("ple-values", 8, b"\x01" * 12)})

    def test_a_blank_role_fails_loudly(self, tmp_path: Path) -> None:
        with pytest.raises(ContainerFormatError, match="role is not a non-empty string"):
            self._write(tmp_path, {"x": AssetInput("", 8, b"\x01" * 8)})


class TestTheCrossLanguageFixture:
    """Keeps the asset the TS `container_fixture_test.ts` opens exactly as this writer writes it."""

    def test_the_committed_fixture_matches_the_generator(self, tmp_path: Path) -> None:
        rebuilt = write_synthetic(tmp_path, single=True)[0].read_bytes()
        if fixture_write_requested():
            FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
            FIXTURE_PATH.write_bytes(rebuilt)

        assert FIXTURE_PATH.read_bytes() == rebuilt, (
            "the fixture drifted from the writer"
            " (rebuild: KARUME_FIXTURE=write uv run pytest tests/test_container.py -k fixture)"
        )

    def test_the_fixture_exercises_the_readers_branches(self, tmp_path: Path) -> None:
        """The fixture "takes each branch once" (no silent fallback to a thinner asset)."""
        read = read_container([FIXTURE_PATH])
        assert read.model is not None
        supplies = read.model.binding[GRAPH_NAME]

        assert len(read.graph.const_blocks) == 2
        assert len(read.model.parts) >= 3
        assert sorted({supply.encoding.codec for supply in supplies.values()}) == [
            "f16",
            "int4-sym-g",
            "int8-sym",
        ]
        assert any(supply.pieces is not None for supply in supplies.values())
        assert any(supply.encoding.row_axis == 1 for supply in supplies.values())
        assert json.loads(read.graph_descriptor_bytes)["graphs"][GRAPH_NAME]["initializers"][
            "lm_head.weight"
        ] == {"shared": True}

    def test_the_committed_split_fixture_matches_the_generator(self, tmp_path: Path) -> None:
        rebuilt = [path.read_bytes() for path in write_split_container(tmp_path)]
        committed = split_fixture_paths()
        if fixture_write_requested():
            committed[0].parent.mkdir(parents=True, exist_ok=True)
            for path, raw in zip(committed, rebuilt, strict=True):
                path.write_bytes(raw)

        assert [path.read_bytes() for path in committed] == rebuilt, (
            "the split-form fixture drifted from the writer"
            " (rebuild: KARUME_FIXTURE=write uv run pytest tests/test_container.py -k fixture)"
        )

    def test_the_split_fixture_exercises_the_asset_and_empty_const_branches(self) -> None:
        """Assets + empty const + 2 weight parts (branches the single-form fixture cannot take)."""
        paths = split_fixture_paths()
        read = read_container(paths)
        assert read.model is not None

        assert len(paths) == SPLIT_FIXTURE_PARTS
        assert read.graph.const_blocks == ()
        assert read.model.parts[0].length == 0
        assert paths[1].stat().st_size == 0
        assert {name: record.role for name, record in read.model.assets.items()} == {
            "ple_index": "ple-index",
            "rope_base": "rope-base",
        }
        assert {name: record.length for name, record in read.model.assets.items()} == {
            "ple_index": 64,
            "rope_base": 37,
        }
        # Asset blocks do not share a part with weight blocks (§4.2).
        asset_blocks = {record.block for record in read.model.assets.values()}
        parts = {block.part for block in read.model.blocks if block.id in asset_blocks}
        assert parts.isdisjoint(
            {block.part for block in read.model.blocks if block.id not in asset_blocks}
        )
        # An odd-length payload is padded with trailing 0x00 to 4-byte alignment.
        payloads = split_asset_payloads()
        raw = read.block(read.model.assets["rope_base"].block)
        assert raw == payloads["rope_base"] + b"\x00" * 3

    def test_the_committed_split_scale_fixture_matches_the_generator(self, tmp_path: Path) -> None:
        rebuilt = write_split_scale_container(tmp_path / SPLIT_SCALE_FIXTURE_PATH.name)[
            0
        ].read_bytes()
        if fixture_write_requested():
            SPLIT_SCALE_FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
            SPLIT_SCALE_FIXTURE_PATH.write_bytes(rebuilt)

        assert SPLIT_SCALE_FIXTURE_PATH.read_bytes() == rebuilt, (
            "the split-scale fixture drifted from the writer"
            " (rebuild: KARUME_FIXTURE=write uv run pytest tests/test_container.py -k fixture)"
        )

    def test_the_split_scale_fixture_has_one_split_scale_and_one_whole_scale(self) -> None:
        """A split scale beside a piece-split weight whose scale fits (both spellings)."""
        read = read_container([SPLIT_SCALE_FIXTURE_PATH])
        assert read.model is not None
        supplies = read.model.binding[SPLIT_SCALE_GRAPH_NAME]

        split, whole = supplies["w.weight"], supplies["v.weight"]
        assert split.pieces is not None and split.encoding.scale_pieces is not None
        assert [rows for _, rows in split.encoding.scale_pieces] == [
            rows for _, rows in split.pieces
        ]
        assert len(split.encoding.scale_pieces) == 4
        assert whole.pieces is not None and len(whole.pieces) == 2
        assert whole.encoding.scale_block is not None and whole.encoding.scale_pieces is None


class TestTheDescriptorReadIsBounded:
    """`read_descriptor_refs` does **not load the whole of** part 0 into RAM (header + two
    documents only).

    The caller (`karume.dist._materialize_family`) goes through here **before** deciding "is it a
    single form". A whole read would make an assembly that put a single-form container in the
    weights slot read several GB before failing as a rule violation (the RAM-peak concern for
    which a harness was added at ADR 0108 step 2).
    """

    def test_it_never_reads_the_whole_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = write_synthetic_container(tmp_path / "model.krm", single=True)[0]
        # The implementation reads through `path.open("rb")`, so blocking `read_bytes` alone would
        # miss a regression to a whole read via `handle.read()` — count the bytes the handle of
        # the target path returned and bound them.
        reads: list[int] = []
        original_open = Path.open

        class CountingHandle:
            def __init__(self, handle: Any) -> None:
                self._handle = handle

            def __enter__(self) -> CountingHandle:
                return self

            def __exit__(self, *exc: object) -> None:
                self._handle.close()

            def read(self, size: int = -1) -> bytes:
                chunk = self._handle.read(size)
                reads.append(len(chunk))
                return chunk

            def __getattr__(self, name: str) -> Any:
                return getattr(self._handle, name)

        def spying_open(self: Path, *args: Any, **kwargs: Any) -> Any:
            handle = original_open(self, *args, **kwargs)
            return CountingHandle(handle) if self == path else handle

        def forbidden(self: Path) -> bytes:
            raise AssertionError(
                f"{self}: read the whole file (MUST be header + two documents only)"
            )

        monkeypatch.setattr(Path, "open", spying_open)
        monkeypatch.setattr(Path, "read_bytes", forbidden)
        graph, model = read_descriptor_refs(path)

        assert graph.length > 0
        assert model.length > 0
        descriptors = HEADER_BYTES + graph.length + model.length
        # Premise of the subject: a single form with const / weight data after part 0 (thinner, the
        # check would become tautological).
        assert path.stat().st_size > descriptors
        assert sum(reads) == descriptors

    def test_a_graph_container_is_refused(self, tmp_path: Path) -> None:
        """A `krg` lacks the two documents (no model descriptor) — not an entry for taking manifest
        expected values."""
        bindings = synthetic_bindings()
        path = write_graph_container(
            tmp_path / "synthetic.krg",
            synthetic_graph(),
            synthetic_tensors(),
            {name: bindings[name] for name in bindings if name.startswith("const.")},
            graph_name=GRAPH_NAME,
            block_bytes=FIXTURE_BLOCK_BYTES,
        )

        with pytest.raises(ContainerFormatError, match="not a krm"):
            read_descriptor_refs(path)


class TestTheReaderRefuses:
    """Negative cases of the reader (checks of the header and the two documents) — one case per
    branch pins `ContainerFormatError`.

    `karume verify` and the publish / migrate self-checks rely entirely on this reader, so any one
    check loosening (a swapped comparison operator and the like) must turn something red. The
    subject is a split-form part 0 (header + two documents only); cases that rewrite a document
    have {@link rewrite_part0} fix the header lengths too — so they do not turn into "wrong
    length" and take only the targeted branch.
    """

    @staticmethod
    def _patched_header(tmp_path: Path, start: int, raw: bytes) -> list[Path]:
        written = write_synthetic(tmp_path, single=False)
        part0 = bytearray(written[0].read_bytes())
        part0[start : start + len(raw)] = raw
        written[0].write_bytes(bytes(part0))
        return written

    def test_an_untouched_rewrite_still_reads(self, tmp_path: Path) -> None:
        """Counterpart — the rewriting scaffold itself does not break the container (the failures
        below come from the injected violations)."""
        written = write_synthetic(tmp_path, single=False)
        edit_model_document(written, lambda document: None)

        assert read_container(written).model is not None

    @pytest.mark.parametrize(
        ("start", "raw", "message"),
        [
            (0, b"XXXX", "Unknown magic"),
            (4, (2).to_bytes(4, "little"), "Unsupported container version 2"),
            (16, (0).to_bytes(8, "little"), "krm but the model descriptor length is 0"),
            (8, (0).to_bytes(8, "little"), "graph descriptor length is 0"),
            (
                8,
                (MAX_DESCRIPTOR_BYTES + 1).to_bytes(8, "little"),
                "graph descriptor length .* exceeds the limit",
            ),
            (0, b"KRGC", "krg but the model descriptor length is"),
        ],
        ids=["magic", "version", "model-length-0", "graph-length-0", "over-limit", "krg-model"],
    )
    def test_a_malformed_header_is_refused(
        self, tmp_path: Path, start: int, raw: bytes, message: str
    ) -> None:
        written = self._patched_header(tmp_path, start, raw)

        with pytest.raises(ContainerFormatError, match=message):
            read_container(written)

    def test_a_header_shorter_than_24_bytes_is_refused(self, tmp_path: Path) -> None:
        written = write_synthetic(tmp_path, single=False)
        written[0].write_bytes(written[0].read_bytes()[: HEADER_BYTES - 1])

        with pytest.raises(ContainerFormatError, match="header needs 24 bytes but has only 23"):
            read_container(written)

    def test_a_document_that_is_not_utf8_is_refused(self, tmp_path: Path) -> None:
        written = write_synthetic(tmp_path, single=False)
        graph = read_container(written).graph_descriptor_bytes
        rewrite_part0(written[0], graph=b"\xff" + graph[1:])

        with pytest.raises(ContainerFormatError, match="graphDescriptor is invalid as UTF-8"):
            read_container(written)

    def test_a_document_that_is_not_json_is_refused(self, tmp_path: Path) -> None:
        written = write_synthetic(tmp_path, single=False)
        graph = read_container(written).graph_descriptor_bytes
        rewrite_part0(written[0], graph=graph[:-1])

        with pytest.raises(ContainerFormatError, match="graphDescriptor is invalid as JSON"):
            read_container(written)

    def test_nesting_deeper_than_the_limit_is_refused(self, tmp_path: Path) -> None:
        written = write_synthetic(tmp_path, single=False)
        rewrite_part0(written[0], model=b"[" * (MAX_JSON_DEPTH + 2) + b"]" * (MAX_JSON_DEPTH + 2))

        with pytest.raises(
            ContainerFormatError, match=f"exceeded the depth limit {MAX_JSON_DEPTH}"
        ):
            read_container(written)

    def test_nesting_at_the_limit_passes_the_depth_check(self, tmp_path: Path) -> None:
        """Boundary — depth exactly at the limit does not fail on depth (it fails on the next "not
        an object")."""
        written = write_synthetic(tmp_path, single=False)
        rewrite_part0(written[0], model=b"[" * (MAX_JSON_DEPTH + 1) + b"]" * (MAX_JSON_DEPTH + 1))

        with pytest.raises(ContainerFormatError, match="modelDescriptor is not an object"):
            read_container(written)

    @pytest.mark.parametrize(
        ("field", "value", "message"),
        [
            ("offset", 63, r"blocks\[1\]: offset 63 is not a multiple of 64"),
            ("length", 0, r"blocks\[1\]: length is 0"),
            # blocks[0] is 0..256 of the same part — a start at 192 overlaps (a multiple of 64, so
            # alignment passes).
            ("offset", 192, "block 's0003' overlaps the preceding block"),
        ],
        ids=["unaligned-offset", "zero-length", "overlap"],
    )
    def test_a_malformed_block_record_is_refused(
        self, tmp_path: Path, field: str, value: int, message: str
    ) -> None:
        written = write_synthetic(tmp_path, single=False)
        edit_model_document(written, lambda document: document["blocks"][1].update({field: value}))

        with pytest.raises(ContainerFormatError, match=message):
            read_container(written)

    def test_an_unknown_key_is_refused(self, tmp_path: Path) -> None:
        written = write_synthetic(tmp_path, single=False)
        edit_model_document(written, lambda document: document.update({"extra": 1}))

        with pytest.raises(ContainerFormatError, match="modelDescriptor: unknown key 'extra'"):
            read_container(written)

    @pytest.mark.parametrize(
        ("requires", "message"),
        [
            ({}, r"requires\.ops is missing"),
            ({"ops": "matmul"}, r"requires\.ops is not an array"),
            ({"ops": ["matmul"], "features": []}, r"requires: unknown key 'features'"),
        ],
        ids=["missing-ops", "string-ops", "unknown-key"],
    )
    def test_a_graph_requires_outside_the_ts_acceptance_set_is_refused(
        self, tmp_path: Path, requires: dict[str, Any], message: str
    ) -> None:
        """`requires` takes only `{ops: string[]}`, like TS `parseIrDeclaration`.

        With plain indexing a missing `ops` becomes a `KeyError`, and a string is iterated one
        character at a time and turns into a violation elsewhere, "capabilities.ops does not
        match the union".
        """
        written = write_synthetic(tmp_path, single=False)
        document = json.loads(read_container(written).graph_descriptor_bytes)
        document["graphs"][GRAPH_NAME]["requires"] = requires
        rewrite_part0(written[0], graph=canonical_json(document).encode())

        with pytest.raises(ContainerFormatError, match=message):
            read_container(written)


def _raw_model_document(
    written: list[Path], edit: Callable[[dict[str, Any]], None], literals: dict[str, str]
) -> None:
    """Rewrite the model descriptor and splice in placeholder strings replaced by **raw number
    literals**.

    `canonical_json` folds `64.0` to `64`, so to deliver an integral float as spelled, the
    `json.dumps` output is string-replaced (the reader does not require the canonical form).
    """
    document = json.loads(read_container(written).model_descriptor_bytes)
    edit(document)
    text = json.dumps(document, ensure_ascii=False, separators=(",", ":"))
    for placeholder, literal in literals.items():
        assert f'"{placeholder}"' in text, f"{placeholder} was not spliced in (a no-op)"
        text = text.replace(f'"{placeholder}"', literal)
    rewrite_part0(written[0], model=text.encode())


class TestTheReaderAcceptsIntegralFloats:
    """JSON integer fields spelled `64.0` / `6.4e1` are accepted as integers, as in TS.

    The TS reader decides with `Number.isSafeInteger`, so it does not tell `64.0` from `64`. If
    Python refused floats outright, **only the exporter's checks could not open a container the
    runtime opens** (the same accepted set as `_assert_integer_attr` / `verify._parse_shape`).
    """

    @pytest.mark.parametrize("spelling", ["{}.0", "{}e0"], ids=["fraction", "exponent"])
    def test_an_integral_float_offset_and_length_read_as_int(
        self, tmp_path: Path, spelling: str
    ) -> None:
        written = write_synthetic(tmp_path, single=False)
        block = read_container(written).model.blocks[1]  # type: ignore[union-attr]

        _raw_model_document(
            written,
            lambda document: document["blocks"][1].update(
                {"offset": "__OFFSET__", "length": "__LENGTH__"}
            ),
            {
                "__OFFSET__": spelling.format(block.offset),
                "__LENGTH__": spelling.format(block.length),
            },
        )
        read = read_container(written).model

        assert read is not None
        assert read.blocks[1].offset == block.offset
        assert read.blocks[1].length == block.length
        # Held as int (holding a float would leave room for to_document to write a different
        # spelling in the round trip).
        assert type(read.blocks[1].offset) is int
        assert type(read.blocks[1].length) is int

    @pytest.mark.parametrize("literal", ["64.5", "true"], ids=["fraction", "bool"])
    def test_a_non_integral_float_or_a_bool_is_still_refused(
        self, tmp_path: Path, literal: str
    ) -> None:
        """Normalization does not widen the accepted set — non-integers and bool stay refused."""
        written = write_synthetic(tmp_path, single=False)

        _raw_model_document(
            written,
            lambda document: document["blocks"][1].update({"offset": "__OFFSET__"}),
            {"__OFFSET__": literal},
        )

        with pytest.raises(
            ContainerFormatError, match=r"blocks\[1\]\.offset is not a non-negative safe integer"
        ):
            read_container(written)


class TestTheReaderMatchesTheRuntimeRefusals:
    """Declarations the TS reader refuses are refused by the Python reader too (ADR 0108 addendum 1,
    item 13 / container-v1 §0).

    Python's bool derives from int and `True == 1` is true, so checking the version or packing with
    `==` or tuple comparison alone lets `true` pass as `1`. TS rejects it as different with
    `!== 1` — this closes the asymmetry where verifying an external `.krm` with `karume verify` is
    "green in Python, red in the browser".
    """

    @pytest.mark.parametrize("document", ["graph", "model"])
    def test_a_bool_version_is_refused(self, tmp_path: Path, document: str) -> None:
        written = write_synthetic(tmp_path, single=False)
        read = read_container(written)
        raw = read.graph_descriptor_bytes if document == "graph" else read.model_descriptor_bytes
        edited = json.loads(raw)
        edited["version"] = True
        rewrite_part0(written[0], **{document: canonical_json(edited).encode()})

        with pytest.raises(ContainerFormatError, match=r"Descriptor\.version"):
            read_container(written)

    def test_a_bool_packing_value_is_refused(self, tmp_path: Path) -> None:
        written = write_synthetic(tmp_path, single=False)

        def edit(document: dict[str, Any]) -> None:
            supplies = document["binding"][GRAPH_NAME]
            f16 = [s for s in supplies.values() if s["encoding"]["codec"] == "f16"]
            assert f16, "the subject has no codec with blockElements = 1 (a no-op)"
            f16[0]["encoding"]["packing"]["blockElements"] = True

        edit_model_document(written, edit)

        with pytest.raises(ContainerFormatError, match=r"packing\.blockElements"):
            read_container(written)

    def test_a_proto_key_is_refused_wherever_it_appears(self, tmp_path: Path) -> None:
        """Even in a map with free-form names (asset names here), `__proto__` fails at read time."""
        written = write_split_container(tmp_path)

        def edit(document: dict[str, Any]) -> None:
            document["assets"]["__proto__"] = document["assets"].pop("rope_base")

        _raw_model_document(written, edit, {})

        with pytest.raises(ContainerFormatError, match="key '__proto__' is not accepted"):
            read_container(written)

    def test_an_empty_asset_name_is_refused(self, tmp_path: Path) -> None:
        written = write_split_container(tmp_path)
        edit_model_document(
            written,
            lambda document: document["assets"].update({"": document["assets"].pop("rope_base")}),
        )

        with pytest.raises(
            ContainerFormatError, match=r"key of modelDescriptor\.assets is not a non-empty string"
        ):
            read_container(written)

    def test_too_many_model_blocks_are_refused(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        written = write_synthetic(tmp_path, single=False)
        model_blocks = len(read_container(written).model.blocks)  # type: ignore[union-attr]
        monkeypatch.setattr(container_module, "MAX_BLOCKS", model_blocks - 1)

        with pytest.raises(
            ContainerFormatError, match=r"modelDescriptor\.blocks count exceeds the limit"
        ):
            read_container(written)

    def test_too_many_blocks_across_both_documents_are_refused(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Even with each document within its limit, const plus weights over the limit fails (the
        same total limit as TS)."""
        written = write_synthetic(tmp_path, single=False)
        read = read_container(written)
        const_blocks = len(read.graph.const_blocks)
        model_blocks = len(read.model.blocks)  # type: ignore[union-attr]
        assert const_blocks > 0, (
            "the subject has no const block (the total limit would check nothing)"
        )
        monkeypatch.setattr(container_module, "MAX_BLOCKS", const_blocks + model_blocks - 1)

        with pytest.raises(
            ContainerFormatError, match=r"Block count \(const .* \+ weights .*\) exceeds the limit"
        ):
            read_container(written)


def _write_pieced_f16(tmp_path: Path) -> list[Path]:
    """Write one f16 weight without a scale split into 4 pieces (parts 2, 2, 3, 3).

    A codec without a scale takes only the part order of the pieces without turning into rule ③
    (the scale is in the same part as piece 1). It also yields consecutive pieces in the same part
    (the equal side of non-decreasing).
    """
    graph = IrGraph(
        symbols=["T"],
        inputs=[IrInput(name="x", dtype="f32", shape=["T", 32])],
        outputs=["y"],
        initializers={"p_w": IrInitializer(tensor="w.weight", storage=IrStorage(dtype="f16"))},
        values={
            "p_w": IrValue(dtype="f32", shape=[32, 16]),
            "y": IrValue(dtype="f32", shape=["T", 16]),
        },
        nodes=[IrNode(op="matmul", ins=["x", "p_w"], outs=["y"], attrs={})],
    )
    return write_model_container(
        tmp_path / "pieced.krm",
        graph,
        {"w.weight": pattern_bytes(1024, 3)},
        {"w.weight": Encoding("f16")},
        graph_name="pieced",
        provenance=Provenance(license="mit"),
        part_bytes=512,
        block_bytes=256,
        single=False,
    )


def _piece_parts(document: dict[str, Any]) -> list[int]:
    parts = {block["id"]: block["part"] for block in document["blocks"]}
    return [parts[piece["block"]] for piece in document["binding"]["pieced"]["w.weight"]["pieces"]]


class TestThePieceOrder:
    """Rule ④: piece parts are non-decreasing in index order (container-v1 §5; TS refuses in
    `bindGraphs`)."""

    def test_pieces_that_stay_in_the_same_part_are_accepted(self, tmp_path: Path) -> None:
        written = _write_pieced_f16(tmp_path)
        document = json.loads(read_container(written).model_descriptor_bytes)

        assert _piece_parts(document) == [2, 2, 3, 3]

    def test_pieces_whose_parts_go_backwards_are_refused(self, tmp_path: Path) -> None:
        """Swap the block refs of pieces 2 and 3 to get parts 2, 3, 2, 3 (the same length 256)."""
        written = _write_pieced_f16(tmp_path)

        def edit(document: dict[str, Any]) -> None:
            pieces = document["binding"]["pieced"]["w.weight"]["pieces"]
            pieces[1]["block"], pieces[2]["block"] = pieces[2]["block"], pieces[1]["block"]
            assert _piece_parts(document) == [2, 3, 2, 3]

        edit_model_document(written, edit)

        with pytest.raises(
            ContainerFormatError, match="part 2 comes before part 3 of the preceding piece"
        ):
            read_container(written)


class TestTheWriterRowAxes:
    """The writer also stops at each codec's accepted `rowAxis` set (no container that was written
    but cannot be read)."""

    @pytest.mark.parametrize(
        ("codec", "storage", "payload"),
        [("int4-sym-g", "i4", 128), ("int2-off", "i2", 64), ("ternary", "i2", 64)],
    )
    def test_row_axis_1_is_refused_for_a_codec_without_it(
        self, tmp_path: Path, codec: str, storage: str, payload: int
    ) -> None:
        graph = _minimal_graph()
        graph.initializers["p_w"] = IrInitializer(
            tensor="b.weight", storage=IrStorage(dtype=storage, scale="b.scale", group_size=16)
        )
        graph.values["p_w"] = IrValue(dtype="f32", shape=[16, 16])

        with pytest.raises(ContainerFormatError, match=f"rowAxis of codec '{codec}' must be 0"):
            write_model_container(
                tmp_path / "m.krm",
                graph,
                {
                    "a.weight": pattern_bytes(64, 5),
                    "b.weight": pattern_bytes(payload, 6),
                    "b.scale": pattern_bytes(64, 7),
                },
                {
                    "a.weight": Encoding("f32"),
                    "b.weight": Encoding(codec, group_size=16, row_axis=1, scale_key="b.scale"),
                },
                graph_name="minimal",
                provenance=Provenance(license="mit"),
            )


def _parse_ts_row_axes(source: str) -> dict[str, tuple[int, ...]]:
    """Read `rowAxes` of the TS `CODEC_LEDGER` (the RAW helper's default and each entry's value)."""
    raw = re.search(r"const RAW = \(.*?rowAxes: \[([\d, ]*)\]", source, re.S)
    assert raw is not None, "cannot read rowAxes of the RAW helper"

    def axes(text: str) -> tuple[int, ...]:
        return tuple(int(axis) for axis in re.findall(r"\d+", text))

    entries = {
        name: axes(raw.group(1))
        for name in re.findall(r'\["([\w-]+)", RAW\("[\w-]+", \d+\)\]', source)
    }
    # Oxfmt (2026-10-16) splits long entries over several lines "[ / name / { … } / ]", so
    # whitespace and newlines are allowed between the name and \`{\` and between \`}\` and \`]\`
    # (RAW lines stay on one line).
    for name, body in re.findall(r'\[\s*"([\w-]+)",\s*\{(.*?)\n\s*\},?\s*\]', source, re.S):
        found = re.search(r"rowAxes: \[([\d, ]*)\]", body)
        assert found is not None, f"{name}: cannot read rowAxes"
        entries[name] = axes(found.group(1))
    return entries


class TestTheSharedRowAxes:
    def test_the_codec_row_axes_match_the_runtime_ones(self) -> None:
        ours = {name: entry.row_axes for name, entry in CODEC_LEDGER.items()}

        assert _parse_ts_row_axes(CODECS_TS.read_text(encoding="utf-8")) == ours


class TestATrailingNewlineIsRefused:
    """The vocabulary checks refuse a trailing newline — the same accepted set as the TS reader
    (JS `$`).

    Python's `$` also matches just before a trailing newline, so `^…$` with `.match` would let a
    trailing newline through.
    """

    def test_a_graph_name_with_a_trailing_newline_is_refused_by_the_reader(
        self, tmp_path: Path
    ) -> None:
        written = write_synthetic(tmp_path, single=False)
        edited = json.loads(read_container(written).graph_descriptor_bytes)
        edited["graphs"] = {f"{name}\n": graph for name, graph in edited["graphs"].items()}
        rewrite_part0(written[0], graph=canonical_json(edited).encode())

        with pytest.raises(ContainerFormatError, match="outside the vocabulary"):
            read_container(written)

    def test_a_graph_name_with_a_trailing_newline_is_refused_by_the_writer(self) -> None:
        with pytest.raises(ContainerFormatError, match="outside the vocabulary"):
            container_module._assert_graph_name("synthetic\n")

    def test_a_block_id_with_a_trailing_newline_is_refused(self) -> None:
        with pytest.raises(ContainerFormatError, match="outside the vocabulary"):
            container_module._require_block_id("b0\n", "test")

    def test_a_sha256_with_a_trailing_newline_is_refused(self) -> None:
        with pytest.raises(ContainerFormatError, match="sha256"):
            container_module._require_sha256("a" * 64 + "\n", "test")


# ---------------------------------------------------------------------------
# Split scales (container-v1 §5 rule ③ — ADR 0108 addendum 15)
# ---------------------------------------------------------------------------


def _split_scale_supply(tmp_path: Path, **options: Any) -> tuple[Any, Any]:
    """Write the split-scale model and return `(reader, merged supply of w.weight)`."""
    verified = verify_container(write_split_scale_container(tmp_path / "model.krm", **options))
    return verified.read, verified.graphs[SPLIT_SCALE_GRAPH_NAME].supplies["w.weight"]


class TestASplitScale:
    """A scale over the block limit is split on the data's piece boundaries and read back whole."""

    def test_the_scale_is_split_on_the_rows_of_the_weights_pieces(self, tmp_path: Path) -> None:
        _, supply = _split_scale_supply(tmp_path)

        assert [block.rows for block in supply.blocks] == [(0, 4), (4, 8), (8, 12), (12, 16)]
        assert [piece.rows for piece in supply.scale.pieces] == [
            block.rows for block in supply.blocks
        ]
        assert [piece.payload_bytes for piece in supply.scale.pieces] == [64, 64, 64, 64]

    def test_every_scale_piece_sits_in_the_part_of_the_first_weight_piece(
        self, tmp_path: Path
    ) -> None:
        _, supply = _split_scale_supply(tmp_path)

        assert {piece.part for piece in supply.scale.pieces} == {supply.blocks[0].part}

    def test_the_joined_id_reads_the_whole_scale_and_names_no_block(self, tmp_path: Path) -> None:
        read, supply = _split_scale_supply(tmp_path)

        assert supply.scale.id == joined_scale_id([piece.id for piece in supply.scale.pieces])
        assert supply.scale.id not in read.block_ids
        assert supply.scale.payload_bytes == 256
        assert read.block(supply.scale.id) == split_scale_tensors()["w.weight_scale"]

    def test_the_publish_path_compares_the_joined_scale_with_the_passed_bytes(
        self, tmp_path: Path
    ) -> None:
        """The single export path re-reads the split scale through its joined id."""
        result = publish_container(
            tmp_path / "out" / "model.krm",
            split_scale_graph(),
            split_scale_tensors(),
            split_scale_bindings(),
            graph_name=SPLIT_SCALE_GRAPH_NAME,
            provenance=Provenance(license="apache-2.0"),
            part_bytes=SPLIT_SCALE_PART_BYTES,
            block_bytes=SPLIT_SCALE_BLOCK_BYTES,
        )

        # 2 bodies + 2 scales (the split one counted once, as one scale).
        assert result.payloads == 4

    def test_a_scale_that_fits_keeps_the_one_block_spelling(self, tmp_path: Path) -> None:
        read, _ = _split_scale_supply(tmp_path)
        document = json.loads(read.model_descriptor_bytes)

        assert list(
            document["binding"][SPLIT_SCALE_GRAPH_NAME]["v.weight"]["encoding"]["scale"]
        ) == [
            "block",
            "dtype",
        ]
        assert list(
            document["binding"][SPLIT_SCALE_GRAPH_NAME]["w.weight"]["encoding"]["scale"]
        ) == [
            "pieces",
            "dtype",
        ]

    def test_a_container_that_needs_no_split_is_the_bytes_the_writer_wrote_before(
        self, tmp_path: Path
    ) -> None:
        """At a 256 B limit the 256-byte scale fits exactly (the limit is inclusive), so nothing is
        split — the bytes equal the output of the writer before split scales (sha256 taken from
        that writer, `git show cd7a0398:tools/exporter/src/karume/container.py`, on the same
        input)."""
        written = write_split_scale_container(tmp_path / "model.krm", block_bytes=256)[0]

        assert hashlib.sha256(written.read_bytes()).hexdigest() == (
            "76e36cfa27d6699df35c6d03164220e65639ff5db8e8ebf58bc14784193e15e4"
        )


def _int8_column(
    rows: int, *, const: bool = False
) -> tuple[IrGraph, dict[str, bytes], dict[str, Encoding]]:
    """An int8 `[rows, 1]` tensor — its scale (4 B per row) is 4x its data (1 B per row), the one
    ledger shape whose scale outgrows its data."""
    key = "const.c" if const else "c.weight"
    graph = IrGraph(
        symbols=["T"],
        inputs=[IrInput(name="x", dtype="f32", shape=["T", 1])],
        outputs=["h"],
        initializers={
            "p_c": IrInitializer(tensor=key, storage=IrStorage(dtype="i8", scale=f"{key}_scale"))
        },
        values={
            "p_c": IrValue(dtype="f32", shape=[rows, 1]),
            "h": IrValue(dtype="f32", shape=["T", rows]),
        },
        nodes=[IrNode(op="linear", ins=["x", "p_c"], outs=["h"], attrs={})],
    )
    tensors = {key: pattern_bytes(rows, 71), f"{key}_scale": pattern_bytes(rows * 4, 72)}
    bindings = {key: Encoding("int8-sym", group_size=1, row_axis=0, scale_key=f"{key}_scale")}
    return graph, tensors, bindings


class TestTheSplitScaleFailsLoudlyAtPlanning:
    """A scale that no split can bring under the block limit fails **before a byte is written**."""

    def _write(
        self, tmp_path: Path, graph: IrGraph, tensors: Any, bindings: Any, **kw: Any
    ) -> None:
        write_model_container(
            tmp_path / "out" / "model.krm",
            graph,
            tensors,
            bindings,
            graph_name="g",
            provenance=Provenance(license="mit"),
            **{"part_bytes": 1024, "block_bytes": 128, **kw},
        )

    def test_a_scale_over_the_limit_beside_data_in_one_block(self, tmp_path: Path) -> None:
        # 64 B of data fits one block; its 256 B of scale does not, and there are no pieces.
        with pytest.raises(ContainerFormatError, match="there are no piece boundaries to split"):
            self._write(tmp_path, *_int8_column(64))
        assert not (tmp_path / "out").exists()

    def test_a_scale_piece_over_the_limit(self, tmp_path: Path) -> None:
        # 512 B of data → 4 pieces of 128 rows; each scale piece would be 512 B.
        with pytest.raises(
            ContainerFormatError, match=r"scale 0: rows \[0, 128\) of the scale are 512 bytes"
        ):
            self._write(tmp_path, *_int8_column(512))
        assert not (tmp_path / "out").exists()

    def test_a_const_scale_over_the_limit(self, tmp_path: Path) -> None:
        with pytest.raises(
            ContainerFormatError, match="the scale of a const must fit in one block"
        ):
            self._write(tmp_path, *_int8_column(64, const=True))
        assert not (tmp_path / "out").exists()

    def test_piece_1_and_the_split_scale_must_fit_one_part(self, tmp_path: Path) -> None:
        # Piece 1 (128 B) + 4 scale pieces (64 B each) = 384 B, over a 256 B part (rule ③).
        with pytest.raises(ContainerFormatError, match="must sit in the same part exceeds"):
            self._write(
                tmp_path,
                split_scale_graph(),
                split_scale_tensors(),
                split_scale_bindings(),
                part_bytes=256,
            )
        assert not (tmp_path / "out").exists()


def _split_scale_written(tmp_path: Path) -> list[Path]:
    """The split-scale model in split form (the reader-rejection tests rewrite its part 0)."""
    return write_model_container(
        tmp_path / "model.krm",
        split_scale_graph(),
        split_scale_tensors(),
        split_scale_bindings(),
        graph_name=SPLIT_SCALE_GRAPH_NAME,
        provenance=Provenance(license="apache-2.0"),
        part_bytes=SPLIT_SCALE_PART_BYTES,
        block_bytes=SPLIT_SCALE_BLOCK_BYTES,
    )


def _w_scale(document: dict[str, Any]) -> dict[str, Any]:
    return document["binding"][SPLIT_SCALE_GRAPH_NAME]["w.weight"]["encoding"]["scale"]


def _block_of(document: dict[str, Any], block_id: str) -> dict[str, Any]:
    return next(block for block in document["blocks"] if block["id"] == block_id)


class TestTheReaderRejectsABrokenSplitScale:
    """Each reader rule of the split scale, broken one at a time on a correct container."""

    def _rejects(self, tmp_path: Path, edit: Callable[[dict[str, Any]], None], match: str) -> None:
        written = _split_scale_written(tmp_path)
        edit_model_document(written, edit)
        with pytest.raises(ContainerFormatError, match=match):
            verify_container(written)

    def test_a_gap_between_pieces(self, tmp_path: Path) -> None:
        def edit(document: dict[str, Any]) -> None:
            _w_scale(document)["pieces"][1]["rows"] = [5, 8]

        self._rejects(tmp_path, edit, r"scale\.pieces\[1\]: does not continue from row 4")

    def test_overlapping_pieces(self, tmp_path: Path) -> None:
        def edit(document: dict[str, Any]) -> None:
            _w_scale(document)["pieces"][1]["rows"] = [3, 8]

        self._rejects(tmp_path, edit, r"scale\.pieces\[1\]: does not continue from row 4")

    def test_rows_that_are_not_the_weights_pieces(self, tmp_path: Path) -> None:
        def edit(document: dict[str, Any]) -> None:
            _w_scale(document)["pieces"][0]["rows"] = [0, 2]
            _w_scale(document)["pieces"][1]["rows"] = [2, 8]

        self._rejects(
            tmp_path, edit, r"rows \[0, 2\) differ from the rows \[0, 4\) of body piece 0"
        )

    def test_a_piece_over_the_block_limit(self, tmp_path: Path) -> None:
        def edit(document: dict[str, Any]) -> None:
            _block_of(document, _w_scale(document)["pieces"][3]["block"])["length"] = (
                BLOCK_MAX_BYTES + 4
            )

        self._rejects(tmp_path, edit, f"exceeds the block limit {BLOCK_MAX_BYTES}")

    def test_a_piece_list_of_one_entry(self, tmp_path: Path) -> None:
        def edit(document: dict[str, Any]) -> None:
            first = _w_scale(document)["pieces"][0]["block"]
            _w_scale(document)["pieces"] = [{"block": first, "rows": [0, 16]}]

        self._rejects(tmp_path, edit, r"scale\.pieces must have 2 or more entries")

    @pytest.mark.parametrize("spelling", ["both", "neither"])
    def test_both_or_neither_of_block_and_pieces(self, tmp_path: Path, spelling: str) -> None:
        def edit(document: dict[str, Any]) -> None:
            if spelling == "both":
                _w_scale(document)["block"] = _w_scale(document)["pieces"][0]["block"]
            else:
                del _w_scale(document)["pieces"]

        self._rejects(tmp_path, edit, "write exactly one of 'block' .* and 'pieces'")

    def test_a_piece_in_another_part_than_piece_1(self, tmp_path: Path) -> None:
        def edit(document: dict[str, Any]) -> None:
            # Only the declarations are checked here, so lengthening the last part keeps the moved
            # block inside it.
            block = _block_of(document, _w_scale(document)["pieces"][3]["block"])
            last = document["parts"][-1]
            block["part"] = last["index"]
            block["offset"] = align_up(last["length"], BLOCK_START_ALIGN)
            last["length"] = block["offset"] + block["length"]

        self._rejects(
            tmp_path,
            edit,
            r"encoding\.scale\.pieces\[3\]: scale is in part 3, the start of the body in part 2",
        )

    def test_a_piece_whose_block_is_not_a_scale_block(self, tmp_path: Path) -> None:
        def edit(document: dict[str, Any]) -> None:
            _block_of(document, _w_scale(document)["pieces"][2]["block"])["role"] = "weight"

        self._rejects(tmp_path, edit, "the role of block .* is 'weight'")

    def test_a_split_scale_on_a_const(self, tmp_path: Path) -> None:
        written = write_synthetic(tmp_path, single=False)
        document = json.loads(read_container(written).graph_descriptor_bytes)
        entry = document["const"]["constants"][0]
        entry["encoding"] = {
            "codec": "int8-sym",
            "packing": {"blockElements": 1, "blockBytes": 1, "alignBytes": 4},
            "rowAxis": 0,
            "groupSize": 1,
            "scale": {
                "pieces": [
                    {"block": entry["block"], "rows": [0, 4]},
                    {"block": entry["block"], "rows": [4, 8]},
                ],
                "dtype": "f32",
            },
        }
        rewrite_part0(written[0], graph=canonical_json(document).encode())

        with pytest.raises(ContainerFormatError, match="a const cannot have a split scale"):
            read_container(written)


class TestTheSequenceNumbersAreAsciiDigits:
    """The sequence numbers of numbered file names are ASCII digits only.

    Python's `\\d` also matches Unicode digits (U+FF11 = full-width one) and `int()` converts them,
    so a name `numbered_name` never writes would be folded to its stem or read as part 1.
    """

    def test_a_full_width_numbered_name_is_not_folded_to_its_stem(self) -> None:
        assert base_path(Path("model-00001-of-00003.krm")) == Path("model.krm")
        odd = Path("model-0000１-of-00003.krm")

        assert base_path(odd) == odd

    def test_a_full_width_numbered_file_is_not_read_as_a_part(self, tmp_path: Path) -> None:
        (tmp_path / "model-0000１-of-0000１.krm").write_bytes(b"part")
        path = tmp_path / "model.krm"

        assert resolve_sequence(path) == (path,)
        assert sequence_siblings(path) == ()
