"""コンテナの合流層（`karume.verify.bind_graphs` / `verify_container`）— container-v1 §5 / §6。

TS 側 `packages/runtime/src/format/container/bind.ts` の鏡像なので、ここが固定するのは
**宣言 shape を要する規則**だけ（2 文書の構造と block の配置は読み手 = `test_container.py` の
担当）。故障注入は descriptor を手組みして掛ける — 正しいコンテナを壊すよりも「どの欄が規則を
破ったのか」が 1 対 1 で見えるため。
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from container_fixture import (
    FIXTURE_BLOCK_BYTES,
    FIXTURE_PATH,
    GRAPH_NAME,
    synthetic_bindings,
    synthetic_graph,
    synthetic_tensors,
)

from karume.container import (
    BlockEncoding,
    ContainerFormatError,
    DataBlockRecord,
    GraphDescriptor,
    ModelDescriptor,
    PartRecord,
    Provenance,
    WeightSupply,
    write_graph_container,
)
from karume.verify import InitializerSupply, bind_graphs, verify_container

#: 手組みの descriptor が持つ sha256 欄（合流層は実バイトを読まないので値は効かない）。
ZERO_SHA = "0" * 64

GRAPH = "g"
WEIGHT = "w"


def declaration(values: dict[str, tuple[str, list[int]]], shared: Sequence[str] = ()) -> dict:
    """IR v2 のグラフ宣言（合流層が読むのは `initializers` と `values` だけ）。"""
    return {
        "format": "karume-ir",
        "version": 2,
        "requires": {"ops": []},
        "symbols": [],
        "inputs": [],
        "outputs": [],
        "initializers": {name: ({"shared": True} if name in shared else {}) for name in values},
        "values": {
            name: {"dtype": dtype, "shape": shape} for name, (dtype, shape) in values.items()
        },
        "nodes": [],
    }


def bind(
    shape: list[Any],
    supply: WeightSupply,
    blocks: Sequence[tuple[str, int, str]],
    *,
    dtype: str = "f32",
) -> InitializerSupply:
    """1 グラフ・1 initializer の descriptor を組んで合流する（`blocks` は id / 長さ / role）。"""
    graph = GraphDescriptor(
        graphs={GRAPH: declaration({WEIGHT: (dtype, shape)})},
        const_length=0,
        const_blocks=(),
        constants=(),
    )
    model = ModelDescriptor(
        parts=(PartRecord(1, 0, ZERO_SHA), PartRecord(2, 1 << 20, ZERO_SHA)),
        blocks=tuple(
            DataBlockRecord(name, 2, 0, length, ZERO_SHA, role)  # type: ignore[arg-type]
            for name, length, role in blocks
        ),
        binding={GRAPH: {WEIGHT: supply}},
        assets={},
        provenance=Provenance(license="mit"),
    )
    return bind_graphs(graph, model)[GRAPH].supplies[WEIGHT]


def quantized(codec: str, group_size: int, row_axis: int = 0) -> BlockEncoding:
    return BlockEncoding(codec, row_axis, group_size, "s1")


class TestTheFixtureContainer:
    """言語横断 fixture（Python が書いた実物）が合流まで通る。"""

    def test_every_initializer_gets_a_supply(self) -> None:
        bound = verify_container([FIXTURE_PATH])[GRAPH_NAME]

        assert {name: supply.encoding.codec for name, supply in bound.supplies.items()} == {
            "big.weight": "int8-sym",
            "conv.weight": "int8-sym",
            "dec.weight": "f16",
            "enc.weight": "int4-sym-g",
            "const.a1b2c3d4e5f60718": "f32",
            "const.b1b2c3d4e5f6071a": "i32",
        }
        # shared 宣言は供給計画を持たない（貸し手の常駐重みが実体）。
        assert "lm_head.weight" not in bound.supplies

    def test_a_const_initializer_is_supplied_from_the_graph(self) -> None:
        bound = verify_container([FIXTURE_PATH])[GRAPH_NAME]
        supply = bound.supplies["const.a1b2c3d4e5f60718"]

        assert supply.origin == "const"
        assert [block.part for block in supply.blocks] == [1]

    def test_a_piece_series_covers_the_rows_and_carries_its_payload_length(self) -> None:
        supply = verify_container([FIXTURE_PATH])[GRAPH_NAME].supplies["big.weight"]

        assert [block.rows for block in supply.blocks] == [(0, 8), (8, 16)]
        assert [block.payload_bytes for block in supply.blocks] == [256, 256]

    def test_a_tail_pad_is_outside_the_payload(self) -> None:
        """f16 の 42 バイトは 44 バイトの block に入る（差 2 バイトが詰め物）。"""
        supply = verify_container([FIXTURE_PATH])[GRAPH_NAME].supplies["dec.weight"]

        assert [(block.length, block.payload_bytes) for block in supply.blocks] == [(44, 42)]

    def test_a_row_axis_1_scale_is_one_value_per_channel(self) -> None:
        supply = verify_container([FIXTURE_PATH])[GRAPH_NAME].supplies["conv.weight"]
        assert supply.scale is not None

        # conv_transpose1d 形 [4,3,8]: 行数は shape[1] = 3。
        assert supply.encoding.row_axis == 1
        assert supply.scale.rows == (0, 3)
        assert supply.scale.payload_bytes == 12

    def test_a_graph_container_supplies_only_its_const_region(self, tmp_path: Path) -> None:
        """`krg` は束縛表を持たない — 重みが要る initializer は供給無しのまま残る。"""
        bindings = synthetic_bindings()
        path = write_graph_container(
            tmp_path / "synthetic.krg",
            synthetic_graph(),
            synthetic_tensors(),
            {name: bindings[name] for name in bindings if name.startswith("const.")},
            graph_name=GRAPH_NAME,
            block_bytes=FIXTURE_BLOCK_BYTES,
        )
        bound = verify_container([path])[GRAPH_NAME]

        assert sorted(bound.supplies) == [
            "const.a1b2c3d4e5f60718",
            "const.b1b2c3d4e5f6071a",
        ]
        assert all(supply.origin == "const" for supply in bound.supplies.values())


class TestTheDeclarationRules:
    """宣言 shape × encoding × block 目次で決まる規則（bind.ts の受理集合）。"""

    def test_a_whole_block_carries_the_declared_payload(self) -> None:
        supply = bind(
            [4, 32], WeightSupply(BlockEncoding("f32"), block="w1"), [("w1", 512, "weight")]
        )

        assert supply.origin == "model"
        assert [(block.rows, block.payload_bytes) for block in supply.blocks] == [((0, 4), 512)]

    def test_a_codec_outside_the_semantic_dtype_is_refused(self) -> None:
        """`i32` 宣言は生の int32 だけ — f16 のビット列として読む形を作らない。"""
        with pytest.raises(ContainerFormatError, match="に codec 'f16' は組めない"):
            bind(
                [4, 32],
                WeightSupply(BlockEncoding("f16"), block="w1"),
                [("w1", 256, "weight")],
                dtype="i32",
            )

    def test_a_block_longer_than_the_payload_plus_its_pad_is_refused(self) -> None:
        with pytest.raises(ContainerFormatError, match="詰め物（4 未満）でない"):
            bind([4, 32], WeightSupply(BlockEncoding("f32"), block="w1"), [("w1", 516, "weight")])

    def test_a_symbolic_shape_cannot_hold_an_entity(self) -> None:
        with pytest.raises(ContainerFormatError, match="記号次元は使えない"):
            bind([4, "T"], WeightSupply(BlockEncoding("f32"), block="w1"), [("w1", 512, "weight")])

    def test_an_initializer_without_a_value_declaration_is_refused(self) -> None:
        graph = GraphDescriptor(
            graphs={GRAPH: declaration({WEIGHT: ("f32", [4])}) | {"values": {}}},
            const_length=0,
            const_blocks=(),
            constants=(),
        )
        model = ModelDescriptor(
            parts=(PartRecord(1, 0, ZERO_SHA),),
            blocks=(DataBlockRecord("w1", 2, 0, 16, ZERO_SHA, "weight"),),
            binding={GRAPH: {WEIGHT: WeightSupply(BlockEncoding("f32"), block="w1")}},
            assets={},
            provenance=Provenance(license="mit"),
        )

        with pytest.raises(ContainerFormatError, match="dtype / shape 宣言が無い"):
            bind_graphs(graph, model)

    def test_an_undeclared_block_is_refused(self) -> None:
        with pytest.raises(ContainerFormatError, match="未宣言の block 'w9'"):
            bind([4, 32], WeightSupply(BlockEncoding("f32"), block="w9"), [("w1", 512, "weight")])


class TestThePieceRules:
    def test_the_last_piece_must_reach_the_declared_rows(self) -> None:
        supply = WeightSupply(BlockEncoding("f32"), pieces=(("w1", (0, 2)), ("w2", (2, 3))))

        with pytest.raises(ContainerFormatError, match="宣言 shape の先頭次元 4 行と違う"):
            bind([4, 32], supply, [("w1", 256, "weight"), ("w2", 128, "weight")])

    def test_an_intermediate_piece_cannot_be_padded(self) -> None:
        """中間 piece の詰め物は次の piece の先頭を潰す（行範囲のバイト数ちょうど MUST）。"""
        supply = WeightSupply(BlockEncoding("f16"), pieces=(("w1", (0, 1)), ("w2", (1, 3))))

        with pytest.raises(ContainerFormatError, match="詰め物不可"):
            # 1 行 = 7 要素 = 14 バイト。中間 piece を 16 バイトに膨らませる。
            bind([3, 7], supply, [("w1", 16, "weight"), ("w2", 28, "weight")])

    def test_a_row_axis_1_initializer_cannot_be_split(self) -> None:
        supply = WeightSupply(
            quantized("int8-sym", 24, row_axis=1), pieces=(("w1", (0, 2)), ("w2", (2, 4)))
        )

        with pytest.raises(ContainerFormatError, match="piece 分割できない"):
            bind(
                [4, 3, 8],
                supply,
                [("w1", 48, "weight"), ("w2", 48, "weight"), ("s1", 12, "scale")],
            )


class TestTheQuantizedRules:
    def test_a_per_channel_group_size_is_the_row_length(self) -> None:
        with pytest.raises(ContainerFormatError, match="groupSize は行長 32 に等しい"):
            bind(
                [4, 32],
                WeightSupply(quantized("int8-sym", 16), block="w1"),
                [("w1", 128, "weight"), ("s1", 16, "scale")],
            )

    def test_a_group_size_must_be_a_power_of_two_at_or_above_the_floor(self) -> None:
        with pytest.raises(ContainerFormatError, match="2 冪かつ 16 以上でない"):
            bind(
                [4, 32],
                WeightSupply(quantized("int4-sym-g", 8), block="w1"),
                [("w1", 64, "weight"), ("s1", 64, "scale")],
            )

    def test_a_group_size_must_divide_the_row_length(self) -> None:
        with pytest.raises(ContainerFormatError, match="で割り切れない"):
            bind(
                [4, 24],
                WeightSupply(quantized("int4-sym-g", 32), block="w1"),
                [("w1", 48, "weight"), ("s1", 16, "scale")],
            )

    def test_the_scale_block_holds_one_value_per_group(self) -> None:
        supply = bind(
            [4, 32],
            WeightSupply(quantized("int4-sym-g", 16), block="w1"),
            [("w1", 64, "weight"), ("s1", 32, "scale")],
        )
        assert supply.scale is not None

        # 4 行 × (32 / 16) group × 4 バイト。
        assert supply.scale.payload_bytes == 32

    def test_a_scale_block_of_the_wrong_length_is_refused(self) -> None:
        with pytest.raises(ContainerFormatError, match="scale: block 's1'"):
            bind(
                [4, 32],
                WeightSupply(quantized("int4-sym-g", 16), block="w1"),
                [("w1", 64, "weight"), ("s1", 16, "scale")],
            )

    def test_an_i2_declaration_must_be_rank_2_with_a_16_multiple_row(self) -> None:
        with pytest.raises(ContainerFormatError, match="正の rank 2・行長 16 の倍数"):
            bind(
                [4, 3, 8],
                WeightSupply(quantized("int2-off", 24), block="w1"),
                [("w1", 24, "weight"), ("s1", 16, "scale")],
            )

    def test_a_quantized_encoding_without_its_three_fields_is_refused(self) -> None:
        with pytest.raises(ContainerFormatError, match="rowAxis / groupSize / scale が要る"):
            bind(
                [4, 32],
                WeightSupply(BlockEncoding("int8-sym"), block="w1"),
                [("w1", 128, "weight")],
            )
