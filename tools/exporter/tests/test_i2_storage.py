"""INT2 の固定整数・宣言・低レベルwriter / reader の一致（ADR 0097）。"""

import json

import pytest
import torch

from karume.emit import (
    ContainerEntry,
    EmitError,
    container_order,
    pack_int2,
    storage_breakdown,
    unpack_int2,
    write_container,
)
from karume.ir import IR_METADATA_KEY, IrGraph, IrInitializer, IrInput, IrNode, IrStorage, IrValue
from karume.verify import ContainerError, IrError, parse_ir_graph, verify_shards


def graph():
    return IrGraph(
        symbols=[],
        inputs=[IrInput(name="ids", dtype="i32", shape=[2])],
        outputs=["y"],
        initializers={"w": IrInitializer(tensor="w", storage=IrStorage(dtype="i2", scale="scale"))},
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

    @pytest.mark.parametrize("scale_shape", [(2, 1), (1, 2)])
    def test_fixed_packed_bytes_roundtrip(self, tmp_path, scale_shape):
        g = graph()
        gpath = tmp_path / "model-00001-of-00002.safetensors"
        wpath = tmp_path / "model-00002-of-00002.safetensors"
        write_container(gpath, [], {IR_METADATA_KEY: g.to_json()}, lambda entry: ())
        payload = {
            "w": bytes([0xE4, 0x1B, 0x72, 0x8D] * 2),
            "scale": torch.tensor([0.375, 1.25]).numpy().tobytes(),
        }
        entries = container_order(
            [
                ContainerEntry(name="w", dtype="I2", shape=(2, 16), nbytes=8),
                ContainerEntry(name="scale", dtype="F32", shape=scale_shape, nbytes=8),
            ]
        )
        write_container(wpath, entries, {}, lambda entry: (payload[entry.name],))
        if scale_shape != (2, 1):
            with pytest.raises(ContainerError):
                verify_shards([gpath, wpath])
        else:
            read = verify_shards([gpath, wpath])
            assert read.to_dict() == g.to_dict()
            assert wpath.read_bytes()[-8:] == payload["w"]
            counts = storage_breakdown(read)
            assert (counts.compressed_bytes, counts.scale_bytes) == (8, 8)
