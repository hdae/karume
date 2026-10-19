"""Values read back from a container (`krm`) return to the values written when it was built.

It replaces the reader of the old shard sequence (`karume_ir` of safetensors + piece keys), so what
it checks is that "the expansion gives the same values even though the source of truth of the
storage moved to the **binding table**". The synthetic containers are made with the writer
(`karume.container.write_model_container`) — with hand-written bytes, the day the writer changes its
padding or how it cuts blocks, the expected values would go stale instead of the reader.
"""

from __future__ import annotations

import struct
from pathlib import Path

import torch
from weights import StoredWeights

from karume.container import Encoding, Provenance, write_model_container
from karume.ir import IrGraph, IrInitializer, IrInput, IrNode, IrStorage, IrValue

#: Graph name the container declares = the weights key of the manifest (container-v1 §2.1).
GRAPH = "model"


def _graph(key: str, shape: list[int], storage: IrStorage) -> IrGraph:
    """Minimal graph consuming one initializer with one node (IR v1 — the writer renames to v2)."""
    return IrGraph(
        symbols=["T"],
        inputs=[IrInput(name="x", dtype="f32", shape=["T", shape[-1]])],
        outputs=["y"],
        initializers={"p_w": IrInitializer(tensor=key, storage=storage)},
        values={
            "p_w": IrValue(dtype="f32", shape=shape),
            "y": IrValue(dtype="f32", shape=["T", shape[0]]),
        },
        nodes=[IrNode(op="matmul", ins=["x", "p_w"], outs=["y"], attrs={})],
    )


def _write(
    tmp_path: Path,
    key: str,
    shape: list[int],
    storage: IrStorage,
    encoding: Encoding,
    tensors: dict[str, bytes],
) -> StoredWeights:
    parts = write_model_container(
        tmp_path / "model.krm",
        _graph(key, shape, storage),
        tensors,
        {key: encoding},
        graph_name=GRAPH,
        provenance=Provenance(license="test"),
        part_bytes=4096,
    )
    return StoredWeights(parts, GRAPH)


def test_int4_restores_signed_values_and_group_scales(tmp_path):
    """packed i4 comes back as `u − 8`, and the scale comes from **the binding table's companion
    block**."""
    # Low nibble 1→−7, high nibble 15→7. On row 2, low 8→0, high 9→1.
    payload = bytes([0xF1] * 8) + bytes([0x98] * 8)
    stored = _write(
        tmp_path,
        "model.weight",
        [2, 16],
        IrStorage(dtype="i4", scale="model.weight_scale", group_size=16),
        Encoding("int4-sym-g", group_size=16, row_axis=0, scale_key="model.weight_scale"),
        {"model.weight": payload, "model.weight_scale": struct.pack("<2f", 0.5, 2.0)},
    )

    assert stored.storage("model.weight") == "i4"
    expected = torch.tensor([[-3.5, 3.5] * 8, [0.0, 2.0] * 8])
    assert torch.equal(stored.dequantized("model.weight"), expected)


def test_int8_scales_each_row_and_keeps_the_declared_shape(tmp_path):
    """i8 is per-channel — the scale count matches the row count, and the logical shape comes back
    from the declared shape."""
    payload = bytes([1, 2, 3, 4, 0xFF, 0xFE, 0xFD, 0xFC])  # 2 rows × 4 columns (lower row −1..−4)
    stored = _write(
        tmp_path,
        "model.weight",
        [2, 4],
        IrStorage(dtype="i8", scale="model.weight_scale"),
        Encoding("int8-sym", group_size=4, row_axis=0, scale_key="model.weight_scale"),
        {"model.weight": payload, "model.weight_scale": struct.pack("<2f", 0.5, 4.0)},
    )

    assert stored.storage("model.weight") == "i8"
    assert stored.shape("model.weight") == [2, 4]
    expected = torch.tensor([[0.5, 1.0, 1.5, 2.0], [-4.0, -8.0, -12.0, -16.0]])
    assert torch.equal(stored.dequantized("model.weight"), expected)


def test_a_name_that_has_no_body_in_the_container_fails_loudly(tmp_path):
    """An undeclared initializer is not silently skipped as "no value"."""
    stored = _write(
        tmp_path,
        "model.weight",
        [2, 4],
        IrStorage(dtype="f32"),
        Encoding("f32"),
        {"model.weight": struct.pack("<8f", *range(8))},
    )

    try:
        stored.tensor("model.missing")
    except ValueError as error:
        assert "model.missing" in str(error)
    else:
        raise AssertionError("an undeclared initializer was readable")
