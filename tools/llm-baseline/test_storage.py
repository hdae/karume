"""容器（`krm`）から読み返した実体が、焼いたときの値へ戻ること。

旧 shard 列の読み手（safetensors の `karume_ir` + piece キー）の置き換えなので、見るのは
「格納の正本が**束縛表**へ移っても展開式が同じ値を出す」ことである。合成の容器は書き手
（`karume.container.write_model_container`）で作る — 手書きのバイト列を置くと、書き手が
詰め物や block の切り方を変えた日に、読み手ではなく期待値が古びる。
"""

from __future__ import annotations

import struct
from pathlib import Path

import torch
from weights import StoredWeights

from karume.container import Encoding, Provenance, write_model_container
from karume.ir import IrGraph, IrInitializer, IrInput, IrNode, IrStorage, IrValue

#: 容器が名乗るグラフ名 = manifest の weights キー（container-v1 §12）。
GRAPH = "model"


def _graph(key: str, shape: list[int], storage: IrStorage) -> IrGraph:
    """initializer 1 本を 1 ノードで消費する最小グラフ（IR v1 — 書き手が v2 へ改名する）。"""
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
    """packed i4 は `u − 8` で戻り、scale は**束縛表の companion block**から来る。"""
    # 下位 nibble 1→−7、上位 nibble 15→7。2 行目は下位 8→0、上位 9→1。
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
    """i8 は per-channel — scale の本数が行数と一致し、論理形は宣言 shape から戻る。"""
    payload = bytes([1, 2, 3, 4, 0xFF, 0xFE, 0xFD, 0xFC])  # 2 行 × 4 列（下の行は −1..−4）
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
    """未宣言の initializer を「値が無い」で黙って飛ばさない。"""
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
        raise AssertionError("未宣言の initializer が読めてしまった")
