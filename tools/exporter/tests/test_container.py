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
from pathlib import Path

import pytest
from conftest import RUNTIME_FIXTURES
from container_fixture import (
    FIXTURE_BLOCK_BYTES,
    FIXTURE_PART_BYTES,
    FIXTURE_PATH,
    GRAPH_NAME,
    deterministic_bytes,
    fixture_write_requested,
    synthetic_bindings,
    synthetic_graph,
    synthetic_tensors,
    write_synthetic_container,
)

from karume.container import (
    BLOCK_MAX_BYTES,
    BLOCK_START_ALIGN,
    BLOCK_TAIL_ALIGN,
    CODEC_FOR_STORAGE,
    CODEC_LEDGER,
    HEADER_BYTES,
    MIN_GROUP_SIZE,
    PART_LENGTH_CHOICES,
    ContainerFormatError,
    Encoding,
    Provenance,
    canonical_json,
    ir_v2_document,
    read_container,
    serialize_graph_descriptor,
    serialize_model_descriptor,
    write_graph_container,
    write_model_container,
)
from karume.ir import IrGraph, IrInitializer, IrInput, IrNode, IrShared, IrStorage, IrValue
from karume.verify import STORAGE_DTYPES

#: TS 側の台帳と寸法定数の置き場（突合の相手）。
CODECS_TS = RUNTIME_FIXTURES.parents[1] / "src" / "format" / "container" / "codecs.ts"
LIMITS_TS = RUNTIME_FIXTURES.parents[1] / "src" / "format" / "container" / "limits.ts"


def write_synthetic(tmp_path: Path, *, single: bool) -> list[Path]:
    return write_synthetic_container(tmp_path / "synthetic.krm", single=single)


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
        tensors = {"a.weight": deterministic_bytes(64, 5), "b.weight": deterministic_bytes(64, 6)}
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
                    "a.weight": deterministic_bytes(64, 5),
                    "b.weight": deterministic_bytes(64, 6),
                    "const.0011223344556677": deterministic_bytes(1024, 7),
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
        tensors["dec.weight"] = deterministic_bytes(40, 23)

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
