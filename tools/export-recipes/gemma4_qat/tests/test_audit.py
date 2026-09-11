"""固定 payload の保存監査が、分割先の値・scale の破損を検出する。"""

import json

import pytest
import torch

from gemma4_qat.audit import assert_fixed_bytes
from karume.emit import FixedQuantizedWeight, write_model
from karume.ir import IrGraph, IrInitializer, IrInput, IrNode, IrStorage, IrValue


@pytest.mark.parametrize("fault", [None, "weight", "scale"])
def test_audit_reads_all_pieces_and_rejects_changed_payload(tmp_path, fault):
    rows, width = 32, 64
    graph = IrGraph(
        symbols=[],
        inputs=[IrInput(name="ids", dtype="i32", shape=[1])],
        outputs=["y"],
        initializers={"w": IrInitializer(tensor="weight", storage=IrStorage(dtype="f32"))},
        values={
            "w": IrValue(dtype="f32", shape=[rows, width]),
            "y": IrValue(dtype="f32", shape=[1, width]),
        },
        nodes=[IrNode(op="embedding", ins=["w", "ids"], outs=["y"], attrs={"padding_idx": -1})],
    )
    packed = torch.arange(rows * width // 4).to(torch.uint8).reshape(rows, width // 4)
    fixed = {"weight": FixedQuantizedWeight("i2", packed, torch.full((rows, 1), 0.125))}
    path = tmp_path / "model.safetensors"
    paths = write_model(
        path,
        graph,
        {"weight": torch.empty(rows, width, device="meta")},
        fixed_weights=fixed,
        _shard_capacity=256,
    )
    assert len(paths) > 2
    assert_fixed_bytes(path, fixed)
    if fault is None:
        return
    changed = False
    for shard in paths:
        raw = bytearray(shard.read_bytes())
        base = 8 + int.from_bytes(raw[:8], "little")
        header = json.loads(raw[8:base])
        key = next(
            (
                name
                for name in header
                if name.startswith("karume.scale.") == (fault == "scale") and name != "__metadata__"
            ),
            None,
        )
        if key is not None:
            raw[base + header[key]["data_offsets"][0]] ^= 1
            shard.write_bytes(raw)
            changed = True
            break
    assert changed
    with pytest.raises(ValueError, match="入力 bytes と違う"):
        assert_fixed_bytes(path, fixed)
