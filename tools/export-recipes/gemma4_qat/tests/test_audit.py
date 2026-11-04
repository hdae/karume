"""The storage audit of fixed payloads detects corruption of values and scales split into
pieces."""

import pytest
import torch

from gemma4_qat.audit import assert_fixed_bytes
from gemma4_qat.tests.series_fixture import FIXTURE_PROVENANCE
from karume.container import ContainerFormatError, container_parts, read_container
from karume.emit import FixedQuantizedWeight, stored_model
from karume.ir import IrGraph, IrInitializer, IrInput, IrNode, IrStorage, IrValue
from karume.publish import publish_container


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
    path = tmp_path / "model.krm"
    stored = stored_model(
        graph, {"weight": torch.empty(rows, width, device="meta")}, fixed_weights=fixed
    )
    # Lower the block limit to 4 rows (64 bytes) so the 32-row weight is split into pieces.
    result = publish_container(
        path,
        stored.graph,
        stored.tensors,
        stored.bindings,
        graph_name="model",
        provenance=FIXTURE_PROVENANCE,
        block_bytes=64,
    )
    read = read_container(result.parts)
    supply = read.model.binding["model"]["weight"]
    assert len(supply.pieces or ()) > 1, (
        "exercise the pieces (a reader that does not fold them would cut the data)"
    )

    assert_fixed_bytes(path, fixed)
    if fault is None:
        return

    # Rewrite one byte of the fixed payload. The block's sha256 then disagrees with the
    # declaration, so it fails when read back (`assert_fixed_bytes` reads through
    # `ReadContainer.block`).
    # The scale's first block: at this block limit the 128-byte scale is split on the pieces (ADR
    # 0108 addendum 15), so it is a piece of the split scale.
    target = supply.encoding.scale_blocks[0] if fault == "scale" else (supply.pieces or ())[0][0]
    block = next(entry for entry in read.model.blocks if entry.id == target)
    part = container_parts(path)[block.part]
    raw = bytearray(part.read_bytes())
    # The location is not taken from the reader's declarations; the extracted data is searched
    # for inside the part (no copy of a private mapping).
    begin = raw.find(read.block(target))
    assert begin >= 0, target
    raw[begin] ^= 1
    part.write_bytes(raw)

    # The sha256 of the read-back fails first (the comparison in `assert_fixed_bytes` is never
    # reached).
    with pytest.raises((ValueError, ContainerFormatError)):
        assert_fixed_bytes(path, fixed)
