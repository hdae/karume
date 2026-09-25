"""コンテナ形式の書き手と読み手（`karume.container`）— docs/container-v1.md / docs/ir-v2.md。

正解判定の正本は **TS 側の読み手**なので、ここが見るのは「Python 側だけで閉じる規則」である:
正準 JSON の綴り・IR v2 への改名と並び・block / part の詰め方・自己検査の読み直し。
「TS が本当に開けるか」は言語横断 fixture（{@link TestTheCrossLanguageFixture} が焼き、
`packages/runtime/tests/container_fixture_test.ts` が開く）が受け持つ。

台帳と寸法定数は**両側で同じ表を持つ**（導出できる事実ではなく cross-package の不変条件）ので、
その突合点も 1 本ここに置く（{@link TestTheSharedTables}）。
"""

from __future__ import annotations

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
    fixture_write_requested,
    pattern_bytes,
    split_asset_payloads,
    split_fixture_paths,
    synthetic_bindings,
    synthetic_graph,
    synthetic_tensors,
    write_split_container,
    write_synthetic_container,
)

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
    canonical_json,
    ir_v2_document,
    read_container,
    read_descriptor_refs,
    read_header,
    serialize_graph_descriptor,
    serialize_model_descriptor,
    write_graph_container,
    write_header,
    write_model_container,
)
from karume.ir import IrGraph, IrInitializer, IrInput, IrNode, IrShared, IrStorage, IrValue
from karume.verify import STORAGE_DTYPES

#: TS 側の台帳と寸法定数の置き場（突合の相手）。
CODECS_TS = RUNTIME_FIXTURES.parents[1] / "src" / "format" / "container" / "codecs.ts"
LIMITS_TS = RUNTIME_FIXTURES.parents[1] / "src" / "format" / "container" / "limits.ts"


def write_synthetic(tmp_path: Path, *, single: bool) -> list[Path]:
    return write_synthetic_container(tmp_path / "synthetic.krm", single=single)


def rewrite_part0(part0: Path, *, graph: bytes | None = None, model: bytes | None = None) -> None:
    """分割形の part 0（ヘッダ + 2 文書だけ）の文書を差し替え、ヘッダの長さも合わせて書き直す。

    長さを揃えるので、壊した文書が「part 0 の長さ違い」に化けず、狙った検査分岐だけを踏む。
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
    """分割形のモデル記述を JSON として書き換える（正準 JSON で綴り直す）。"""
    document = json.loads(read_container(written).model_descriptor_bytes)
    edit(document)
    rewrite_part0(written[0], model=canonical_json(document).encode())


class TestCanonicalJson:
    """数値の綴りは ECMAScript `Number::toString`（docs/ir-v2.md「正準直列化」）。"""

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
            # 端: 倍精度の最大と非正規化の最小（どちらも指数表記で綴る領域）。
            (1.7976931348623157e308, "1.7976931348623157e+308"),
            (5e-324, "5e-324"),
            # 整数値の float は小数点を持たない（`repr` は `10000.0` を返す）。
            (-1024.0, "-1024"),
        ],
    )
    def test_a_number_is_spelled_like_ecmascript(self, value: float, expected: str) -> None:
        assert canonical_json(value) == expected

    def test_the_document_has_no_whitespace_and_keeps_the_declared_key_order(self) -> None:
        """並べ替えは組み立て側の責任 — 直列化器は宣言順をそのまま綴る。"""
        assert canonical_json({"b": 1, "a": [True, False, None]}) == '{"b":1,"a":[true,false,null]}'

    def test_non_ascii_stays_raw_and_control_characters_are_escaped(self) -> None:
        assert canonical_json("\u3042\n\u0001") == '"\u3042\\n\\u0001"'

    @pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
    def test_a_non_finite_number_is_refused(self, value: float) -> None:
        with pytest.raises(ContainerFormatError, match="非有限数"):
            canonical_json(value)

    def test_a_non_string_key_is_refused(self) -> None:
        with pytest.raises(ContainerFormatError, match="キーが文字列でない"):
            canonical_json({1: 2})


def _minimal_graph(**overrides: object) -> IrGraph:
    """1 initializer・1 ノードの最小グラフ（改名と並びの検査の被験体）。"""
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
        # attrs も再帰的に code point 順（自由形の値なので組み立て側が正準化する）。
        assert '"attrs":{"a":2,"b":1}' in text
        # 配列は宣言順のまま（順序が意味を持つ欄）。
        assert '"ins":["x","a.weight"]' in text

    def test_an_empty_states_section_is_not_written(self) -> None:
        document = ir_v2_document(_minimal_graph())

        assert "states" not in document
        assert "states" not in document["nodes"][0]

    def test_a_rename_that_collides_with_a_value_name_fails_loudly(self) -> None:
        graph = _minimal_graph()
        graph.initializers["p_w"] = IrInitializer(tensor="y", storage=IrStorage(dtype="f32"))

        with pytest.raises(ContainerFormatError, match="既存の値名"):
            ir_v2_document(graph)

    def test_two_initializers_that_rename_to_the_same_key_fail_loudly(self) -> None:
        graph = _minimal_graph()
        graph.initializers["p_w"] = IrInitializer(tensor="a.weight", storage=IrStorage(dtype="f32"))

        with pytest.raises(ContainerFormatError, match="改名先"):
            ir_v2_document(graph)

    def test_an_initializer_without_a_tensor_key_fails_loudly(self) -> None:
        graph = _minimal_graph()
        graph.initializers["p_w"] = IrInitializer(storage=IrStorage(dtype="f32"))

        with pytest.raises(ContainerFormatError, match="IR v1 として不正"):
            ir_v2_document(graph)


def _parse_ts_ledger(source: str) -> dict[str, tuple[str, int, int, int, str, str | None]]:
    """TS の `CODEC_LEDGER` を読む（登録名 → layout + packing 3 値 + scale + grouping）。

    正規表現で読むのは、突合の相手が**リポ内の不変なデータ**だからである（実行時に import できる
    形にすると Python 側のテストが deno を要求する）。TS 側の綴りが動けばここが落ちる。
    """
    raw_align = re.search(r"const RAW = \(.*?alignBytes: (\d+)", source, re.S)
    raw_elements = re.search(r"const RAW = \(.*?blockElements: (\d+)", source, re.S)
    assert raw_align is not None and raw_elements is not None, "RAW ヘルパを読めない"
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
    for name, body in re.findall(r'\["([\w-]+)", \{(.*?)\n  \}\]', source, re.S):
        numbers = {
            key: int(re.search(rf"{key}: (\d+)", body).group(1))  # type: ignore[union-attr]
            for key in ("blockElements", "blockBytes", "alignBytes")
        }
        layout_match = re.search(r'layout: "([\w-]+)"', body)
        scale = re.search(r'scale: "(\w+)"', body)
        grouping = re.search(r'grouping: "?(\w+)"?', body)
        assert layout_match is not None, f"{name}: layout を読めない"
        assert scale is not None and grouping is not None, f"{name}: scale / grouping を読めない"
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
    """台帳と寸法定数は両側が**同じ値**を持つ（資産に焼かれる値なので片側だけ動かせない）。"""

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
        """旧 `storage.dtype` の全種に写し先がある（移行 CLI が黙って席を落とさない）。"""
        assert set(CODEC_FOR_STORAGE) == set(STORAGE_DTYPES)
        assert set(CODEC_FOR_STORAGE.values()) <= set(CODEC_LEDGER)
        # 三値であるという主張は量子化器の側がする — 旧 i2 は `int2-off` へだけ写る（§6.3）。
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
    """書いたものを読み直す（§12 の不変条件 5 と同じ流儀 — 書けたのに読めないものを作らない）。"""

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
        # 規則③: companion scale は piece 1 と同一 part。
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

        with pytest.raises(ContainerFormatError, match="sha256 が宣言と違う"):
            read_container(written).block(target.id)

    def test_a_truncated_part_is_caught_before_any_block_is_read(self, tmp_path: Path) -> None:
        written = write_synthetic(tmp_path, single=False)
        written[-1].write_bytes(written[-1].read_bytes()[:-4])

        with pytest.raises(ContainerFormatError, match="が宣言"):
            read_container(written)


class TestTheGraphContainer:
    """`krg` は `krm` からバイトコピーで抜ける（§9 — 同一性を内容ハッシュで判定する条件）。"""

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
        with pytest.raises(ContainerFormatError, match="余剰"):
            write_graph_container(
                tmp_path / "synthetic.krg",
                synthetic_graph(),
                synthetic_tensors(),
                synthetic_bindings(),
                graph_name=GRAPH_NAME,
                block_bytes=FIXTURE_BLOCK_BYTES,
            )

    def test_an_empty_const_region_still_extracts(self, tmp_path: Path) -> None:
        """長さ 0 の part の前に詰め物を挿まない（§3 — 挿むと抽出結果がずれる）。"""
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

        with pytest.raises(ContainerFormatError, match=r"不足 \[dec.weight\]"):
            write_model_container(
                tmp_path / "m.krm",
                synthetic_graph(),
                synthetic_tensors(),
                bindings,
                graph_name=GRAPH_NAME,
                provenance=Provenance(license="mit"),
            )

    def test_a_const_that_exceeds_the_block_limit_fails_loudly(self, tmp_path: Path) -> None:
        """const は piece 分割の機構を持たない（§3 — 逃げ道を作らず書き手が止まる）。"""
        graph = _minimal_graph()
        graph.initializers["c"] = IrInitializer(
            tensor="const.0011223344556677", storage=IrStorage(dtype="f32")
        )
        graph.values["c"] = IrValue(dtype="f32", shape=[256])

        with pytest.raises(ContainerFormatError, match="1 block に収める"):
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
        with pytest.raises(ContainerFormatError, match="piece 分割できない"):
            write_model_container(
                tmp_path / "m.krm",
                synthetic_graph(),
                synthetic_tensors(),
                synthetic_bindings(),
                graph_name=GRAPH_NAME,
                provenance=Provenance(license="mit"),
                part_bytes=FIXTURE_PART_BYTES,
                # conv.weight（96 バイト・rowAxis 1）が割れる寸法まで下げる。
                block_bytes=64,
            )

    def test_a_payload_that_disagrees_with_the_declaration_fails_loudly(
        self, tmp_path: Path
    ) -> None:
        tensors = synthetic_tensors()
        tensors["dec.weight"] = pattern_bytes(40, 23)

        with pytest.raises(ContainerFormatError, match="宣言から決まる 42 バイトと違う"):
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

        with pytest.raises(ContainerFormatError, match="2 冪"):
            write_model_container(
                tmp_path / "m.krm",
                synthetic_graph(),
                synthetic_tensors(),
                bindings,
                graph_name=GRAPH_NAME,
                provenance=Provenance(license="mit"),
            )


class TestTheProvenanceReader:
    """省略可の 3 欄（notice / upstreamRevision / writer）も TS の読み手と同じく非空文字列を要る。

    publish の読み直しと `karume verify` はこの読み手を通るので、ここが素通しだと TS の
    `openContainer` が開けない容器（`karume migrate --notice ""` 等）が据わる。
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

        with pytest.raises(ContainerFormatError, match=r"provenance\.notice が非空文字列でない"):
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

        with pytest.raises(ContainerFormatError, match=rf"provenance\.{key} が非空文字列でない"):
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
    """資産の受け口（§2.2 / §4.2）— piece 分割せず、重みと part を共有しない。"""

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
        """遅延の呼び出しは「いつ引くか」だけを変える（出るバイト列は同じ）。"""
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
        """論理長は宣言の写し・block 長はそれを 4 の倍数へ切り上げた値（ADR 0109 決定 4）。"""
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
        """区間読みを要する資産は専用 part に単独で置く MUST（container-v1 §4.2）。

        全量読みの資産を**先に**渡すのは、専用 part が「直前の part の続き」にならないことまで
        見るためである（後ろに開く側だけを見ると、この規則は空振りで緑になる）。
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

        # 専用 part は 1 block だけを持ち、全量読みの資産とも同居しない。
        assert len({parts["idx"], parts["v.0"], parts["v.1"], parts["tail"]}) == 4
        for name in ("v.0", "v.1"):
            assert [block.id for block in read.model.blocks if block.part == parts[name]] == [
                read.model.assets[name].block
            ]

    def test_the_physical_order_follows_the_call_order(self, tmp_path: Path) -> None:
        """配置の順は**渡した順**（PLE は token 順で渡る）— 名前の辞書順ではない。"""
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
        """資産は piece 分割の機構を持たない（読み手は役割の索引で行を引く）。"""
        with pytest.raises(ContainerFormatError, match="資産は 1 block に収める"):
            self._write(
                tmp_path, {"x": AssetInput("ple-values", 300, b"\x00" * 300)}, block_bytes=256
            )

    def test_an_empty_asset_fails_loudly(self, tmp_path: Path) -> None:
        with pytest.raises(ContainerFormatError, match="長さ 0 の block は作らない"):
            self._write(tmp_path, {"x": AssetInput("ple-values", 0, b"")})

    def test_an_asset_name_that_collides_with_a_tensor_key_fails_loudly(
        self, tmp_path: Path
    ) -> None:
        """衝突を許すと block の実体が黙って入れ替わる。"""
        with pytest.raises(ContainerFormatError, match="テンソルキーと衝突"):
            self._write(tmp_path, {"a.weight": AssetInput("ple-values", 8, b"\x01" * 8)})

    def test_a_payload_that_differs_between_pulls_fails_loudly(self, tmp_path: Path) -> None:
        """引かれるたびに同じバイト列を返す MUST（何度引く実装でも同じ結論になる形で見る）。"""
        pulls = count(8)

        with pytest.raises(ContainerFormatError, match="宣言から決まる 8 バイトと違う"):
            self._write(tmp_path, {"x": AssetInput("ple-values", 8, lambda: b"\x01" * next(pulls))})

    def test_a_payload_that_disagrees_with_the_declared_length_fails_loudly(
        self, tmp_path: Path
    ) -> None:
        """宣言が先（配置は宣言で決まる）— 実体が違えば落とす。"""
        with pytest.raises(ContainerFormatError, match="宣言から決まる 8 バイトと違う"):
            self._write(tmp_path, {"x": AssetInput("ple-values", 8, b"\x01" * 12)})

    def test_a_blank_role_fails_loudly(self, tmp_path: Path) -> None:
        with pytest.raises(ContainerFormatError, match="役割が非空文字列でない"):
            self._write(tmp_path, {"x": AssetInput("", 8, b"\x01" * 8)})


class TestTheCrossLanguageFixture:
    """TS 側 `container_fixture_test.ts` が開く資産を、この書き手が焼いたまま保つ。"""

    def test_the_committed_fixture_matches_the_generator(self, tmp_path: Path) -> None:
        rebuilt = write_synthetic(tmp_path, single=True)[0].read_bytes()
        if fixture_write_requested():
            FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
            FIXTURE_PATH.write_bytes(rebuilt)

        assert FIXTURE_PATH.read_bytes() == rebuilt, (
            "fixture が書き手とずれている"
            "（焼き直し: KARUME_FIXTURE=write uv run pytest tests/test_container.py -k fixture）"
        )

    def test_the_fixture_exercises_the_readers_branches(self, tmp_path: Path) -> None:
        """fixture が「分岐を 1 回ずつ踏む」形であること（痩せた資産に黙って戻らない）。"""
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
            "分割形 fixture が書き手とずれている"
            "（焼き直し: KARUME_FIXTURE=write uv run pytest tests/test_container.py -k fixture）"
        )

    def test_the_split_fixture_exercises_the_asset_and_empty_const_branches(self) -> None:
        """資産 + 空 const + 重み part 2 本（単一形の fixture では踏めない分岐）。"""
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
        # 資産の block は重み block と part を共有しない（§4.2）。
        asset_blocks = {record.block for record in read.model.assets.values()}
        parts = {block.part for block in read.model.blocks if block.id in asset_blocks}
        assert parts.isdisjoint(
            {block.part for block in read.model.blocks if block.id not in asset_blocks}
        )
        # 奇数長の payload は末尾 0x00 で 4 バイト整列へ詰められる。
        payloads = split_asset_payloads()
        raw = read.block(read.model.assets["rope_base"].block)
        assert raw == payloads["rope_base"] + b"\x00" * 3


class TestTheDescriptorReadIsBounded:
    """`read_descriptor_refs` は part 0 を**丸ごと RAM へ載せない**（ヘッダ + 2 文書だけ）。

    呼び手（`karume.dist._materialize_family`）は「単一形かどうか」を判定する**前に**ここを
    通る。全量読みにすると、単一形の容器を weights 席へ挿した組み立てが、規則違反として
    落ちる前に数 GB を読む（段 2 で harness を入れた RAM ピークの関心事）。
    """

    def test_it_never_reads_the_whole_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = write_synthetic_container(tmp_path / "model.krm", single=True)[0]
        # 実装は `path.open("rb")` 経由で読むので、`read_bytes` を塞ぐだけでは `handle.read()` の
        # 全量読みへの退行を見逃す — 対象 path のハンドルが返したバイト数を数えて縛る。
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
            raise AssertionError(f"{self}: 現物を丸ごと読んだ（ヘッダ + 2 文書だけ MUST）")

        monkeypatch.setattr(Path, "open", spying_open)
        monkeypatch.setattr(Path, "read_bytes", forbidden)
        graph, model = read_descriptor_refs(path)

        assert graph.length > 0
        assert model.length > 0
        descriptors = HEADER_BYTES + graph.length + model.length
        # 被験体の前提: part 0 の後ろに const / 重みの data が続く単一形（痩せると門が恒真になる）。
        assert path.stat().st_size > descriptors
        assert sum(reads) == descriptors

    def test_a_graph_container_is_refused(self, tmp_path: Path) -> None:
        """`krg` は 2 文書を持たない（モデル記述が無い）— manifest の期待値を採る口ではない。"""
        bindings = synthetic_bindings()
        path = write_graph_container(
            tmp_path / "synthetic.krg",
            synthetic_graph(),
            synthetic_tensors(),
            {name: bindings[name] for name in bindings if name.startswith("const.")},
            graph_name=GRAPH_NAME,
            block_bytes=FIXTURE_BLOCK_BYTES,
        )

        with pytest.raises(ContainerFormatError, match="krm でない"):
            read_descriptor_refs(path)


class TestTheReaderRefuses:
    """読み手（ヘッダと 2 文書の検査）の負例 — 1 分岐 1 ケースで `ContainerFormatError` を縛る。

    `karume verify` と publish / migrate の自己検査はこの読み手に全面的に頼るので、検査が 1 本
    緩んでも（比較演算子の取り違え等）どこかが赤になる形にしておく。被験体は分割形の part 0
    （ヘッダ + 2 文書だけ）で、文書を書き換えるケースは {@link rewrite_part0} がヘッダの長さも
    合わせる — 「長さ違い」に化けず、狙った分岐だけを踏む。
    """

    @staticmethod
    def _patched_header(tmp_path: Path, start: int, raw: bytes) -> list[Path]:
        written = write_synthetic(tmp_path, single=False)
        part0 = bytearray(written[0].read_bytes())
        part0[start : start + len(raw)] = raw
        written[0].write_bytes(bytes(part0))
        return written

    def test_an_untouched_rewrite_still_reads(self, tmp_path: Path) -> None:
        """対照 — 書き換えの足場そのものは容器を壊さない（下の赤は注入した違反のもの）。"""
        written = write_synthetic(tmp_path, single=False)
        edit_model_document(written, lambda document: None)

        assert read_container(written).model is not None

    @pytest.mark.parametrize(
        ("start", "raw", "message"),
        [
            (0, b"XXXX", "未知の magic"),
            (4, (2).to_bytes(4, "little"), "未対応のコンテナ版 2"),
            (16, (0).to_bytes(8, "little"), "krm なのにモデル記述の長さが 0"),
            (8, (0).to_bytes(8, "little"), "グラフ記述の長さが 0"),
            (8, (MAX_DESCRIPTOR_BYTES + 1).to_bytes(8, "little"), "グラフ記述の長さ .* が上限"),
            (0, b"KRGC", "krg なのにモデル記述の長さが"),
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

        with pytest.raises(ContainerFormatError, match="ヘッダに 24 バイト必要だが 23 バイト"):
            read_container(written)

    def test_a_document_that_is_not_utf8_is_refused(self, tmp_path: Path) -> None:
        written = write_synthetic(tmp_path, single=False)
        graph = read_container(written).graph_descriptor_bytes
        rewrite_part0(written[0], graph=b"\xff" + graph[1:])

        with pytest.raises(ContainerFormatError, match="graphDescriptor が UTF-8 として不正"):
            read_container(written)

    def test_a_document_that_is_not_json_is_refused(self, tmp_path: Path) -> None:
        written = write_synthetic(tmp_path, single=False)
        graph = read_container(written).graph_descriptor_bytes
        rewrite_part0(written[0], graph=graph[:-1])

        with pytest.raises(ContainerFormatError, match="graphDescriptor が JSON として不正"):
            read_container(written)

    def test_nesting_deeper_than_the_limit_is_refused(self, tmp_path: Path) -> None:
        written = write_synthetic(tmp_path, single=False)
        rewrite_part0(written[0], model=b"[" * (MAX_JSON_DEPTH + 2) + b"]" * (MAX_JSON_DEPTH + 2))

        with pytest.raises(ContainerFormatError, match=f"深さ上限 {MAX_JSON_DEPTH} を超えた"):
            read_container(written)

    def test_nesting_at_the_limit_passes_the_depth_check(self, tmp_path: Path) -> None:
        """境界 — 深さちょうど上限は深さでは落ちない（次の「オブジェクトでない」で落ちる）。"""
        written = write_synthetic(tmp_path, single=False)
        rewrite_part0(written[0], model=b"[" * (MAX_JSON_DEPTH + 1) + b"]" * (MAX_JSON_DEPTH + 1))

        with pytest.raises(ContainerFormatError, match="modelDescriptor がオブジェクトでない"):
            read_container(written)

    @pytest.mark.parametrize(
        ("field", "value", "message"),
        [
            ("offset", 63, r"blocks\[1\]: offset 63 が 64 の倍数でない"),
            ("length", 0, r"blocks\[1\]: length が 0"),
            # blocks[0] は同じ part の 0..256 — 192 始まりは重なる（64 の倍数なので整列は通る）。
            ("offset", 192, "block 's0003' が直前の block と重なる"),
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

        with pytest.raises(ContainerFormatError, match="modelDescriptor: 未知のキー 'extra'"):
            read_container(written)

    @pytest.mark.parametrize(
        ("requires", "message"),
        [
            ({}, r"requires\.ops が無い"),
            ({"ops": "matmul"}, r"requires\.ops が配列でない"),
            ({"ops": ["matmul"], "features": []}, r"requires: 未知のキー 'features'"),
        ],
        ids=["missing-ops", "string-ops", "unknown-key"],
    )
    def test_a_graph_requires_outside_the_ts_acceptance_set_is_refused(
        self, tmp_path: Path, requires: dict[str, Any], message: str
    ) -> None:
        """`requires` は TS の `parseIrDeclaration` と同じく `{ops: string[]}` だけを受ける。

        素の添字だと `ops` の欠落は `KeyError` になり、文字列は 1 文字ずつ反復されて
        「capabilities.ops が和集合と一致しない」という別の場所の違反に化ける。
        """
        written = write_synthetic(tmp_path, single=False)
        document = json.loads(read_container(written).graph_descriptor_bytes)
        document["graphs"][GRAPH_NAME]["requires"] = requires
        rewrite_part0(written[0], graph=canonical_json(document).encode())

        with pytest.raises(ContainerFormatError, match=message):
            read_container(written)
