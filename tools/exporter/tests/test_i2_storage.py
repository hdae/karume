"""INT2 の固定整数・宣言・容器への往復（ADR 0097）。"""

import json

import pytest
import torch

from karume.container import Provenance
from karume.emit import (
    EmitError,
    FixedQuantizedWeight,
    pack_int2,
    storage_breakdown,
    stored_model,
    unpack_int2,
)
from karume.ir import IrGraph, IrInitializer, IrInput, IrNode, IrStorage, IrValue
from karume.publish import publish_container
from karume.verify import IrError, parse_ir_graph, verify_container

#: 2 行 × 16 要素を [-2,1] で埋めた固定整数の packed バイト（1 行 4 バイト）。
PACKED = bytes([0xE4, 0x1B, 0x72, 0x8D] * 2)

PROVENANCE = Provenance(license="mit")


def graph(storage: IrStorage | None = None):
    """i2 を宣言したグラフ（IR 単体の規則を見る側が使う形）。"""
    declared = storage if storage is not None else IrStorage(dtype="i2", scale="scale")
    return IrGraph(
        symbols=[],
        inputs=[IrInput(name="ids", dtype="i32", shape=[2])],
        outputs=["y"],
        initializers={"w": IrInitializer(tensor="w", storage=declared)},
        values={"w": IrValue(dtype="f32", shape=[2, 16]), "y": IrValue(dtype="f32", shape=[2, 16])},
        nodes=[IrNode(op="embedding", ins=["w", "ids"], outs=["y"], attrs={"padding_idx": -1})],
    )


class TestFixedInt2:
    def test_all_signed_values_and_byte_positions(self):
        q = torch.tensor([-2, -1, 0, 1, 1, 0, -1, -2, 0, -2, 1, -1, -1, 1, -2, 0], dtype=torch.int8)
        packed = pack_int2(q)
        assert packed.tolist() == [0xE4, 0x1B, 0x72, 0x8D]
        assert torch.equal(unpack_int2(packed, [1, 16]), q.reshape(1, 16))

    @pytest.mark.parametrize("values", [[-3, 0, 0, 0], [2, 0, 0, 0], [-2, -1, 0]])
    def test_invalid_integer_stream_is_rejected(self, values):
        with pytest.raises(EmitError):
            pack_int2(torch.tensor(values, dtype=torch.int8))

    @pytest.mark.parametrize("shape", [[32], [2, 15], [2, 20], [0, 16], [2, 0], [1, 2, 16]])
    def test_ir_rejects_non_row_aligned_shape(self, shape):
        g = graph().to_dict()
        g["values"]["w"]["shape"] = shape
        with pytest.raises(IrError):
            parse_ir_graph(json.dumps(g))

    def test_group_size_is_not_accepted_for_row_scale(self):
        g = graph().to_dict()
        g["initializers"]["w"]["storage"]["group_size"] = 16
        with pytest.raises(IrError, match="group_size"):
            parse_ir_graph(json.dumps(g))


def _fixed(scale_shape: tuple[int, int]) -> tuple[IrGraph, dict, dict]:
    """i2 の固定重み 1 本（論理は f32/meta・実体は packed）。"""
    plain = graph(IrStorage(dtype="f32"))
    tensors = {"w": torch.empty(2, 16, dtype=torch.float32, device="meta")}
    fixed = {
        "w": FixedQuantizedWeight(
            dtype="i2",
            packed=torch.frombuffer(bytearray(PACKED), dtype=torch.uint8).reshape(2, 4),
            scale=torch.tensor([0.375, 1.25]).reshape(scale_shape),
        )
    }
    return plain, tensors, fixed


class TestFixedPackedBytesReachTheContainer:
    """固定 packed 値は**再量子化せずそのまま** block の payload になる。"""

    def test_a_row_scale_lands_in_the_container_byte_for_byte(self, tmp_path):
        plain, tensors, fixed = _fixed((2, 1))
        stored = stored_model(plain, tensors, fixed_weights=fixed)

        result = publish_container(
            tmp_path / "model.krm",
            stored.graph,
            stored.tensors,
            stored.bindings,
            graph_name="i2",
            provenance=PROVENANCE,
        )

        verified = verify_container(result.parts, blocks=True)
        supply = verified.graphs["i2"].supplies["w"]
        assert verified.read.block(supply.blocks[0].id)[: supply.blocks[0].payload_bytes] == PACKED
        assert supply.encoding.codec == "int2-off"
        counts = storage_breakdown(stored.graph)
        assert (counts.compressed_bytes, counts.scale_bytes) == (8, 8)

    def test_a_per_column_scale_is_refused_before_anything_is_written(self, tmp_path):
        """`[1, rows]` は行ごとの scale ではない — 通すとチャネルの値が黙って入れ替わる。"""
        plain, tensors, fixed = _fixed((1, 2))

        with pytest.raises(EmitError, match="scale"):
            stored_model(plain, tensors, fixed_weights=fixed)

        assert list(tmp_path.iterdir()) == []
