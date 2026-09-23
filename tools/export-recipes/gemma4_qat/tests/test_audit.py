"""固定 payload の保存監査が、piece へ割れた値・scale の破損を検出する。"""

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
    # block 上限を 4 行（64 バイト）まで下げて、32 行の重みを piece 列へ割らせる。
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
    assert len(supply.pieces or ()) > 1, "piece 列を踏ませる（畳まない読み手なら実体が切れる）"

    assert_fixed_bytes(path, fixed)
    if fault is None:
        return

    # 固定 payload の 1 バイトを書き換える。block の sha256 は宣言と食い違うので、取り直しの
    # 段で落ちる（`assert_fixed_bytes` は `ReadContainer.block` 越しに読む）。
    target = supply.encoding.scale_block if fault == "scale" else (supply.pieces or ())[0][0]
    block = next(entry for entry in read.model.blocks if entry.id == target)
    part = container_parts(path)[block.part]
    raw = bytearray(part.read_bytes())
    # 在処は読み手の宣言から引かず、取り出した実体を part の中で探す（private な写像を写さない）。
    begin = raw.find(read.block(target))
    assert begin >= 0, target
    raw[begin] ^= 1
    part.write_bytes(raw)

    # 取り直しの sha256 が先に落ちる（`assert_fixed_bytes` の突合まで届かない）。
    with pytest.raises((ValueError, ContainerFormatError)):
        assert_fixed_bytes(path, fixed)
