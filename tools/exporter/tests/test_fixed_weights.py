"""固定量子化の保存値と、曖昧な値の供給を拒否する契約（ADR 0097）。"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from karume.container import BLOCK_MAX_BYTES, Provenance, container_parts
from karume.emit import EmitError, FixedQuantizedWeight, stored_model
from karume.ir import IrGraph, IrInitializer, IrInput, IrNode, IrStorage, IrValue
from karume.pipeline import publish_model
from karume.publish import publish_container
from karume.verify import verify_container

PROVENANCE = Provenance(license="mit")

#: piece 分割を踏む block 上限（33 行の重みが 2 本以上の piece に割れる大きさ）。
SMALL_BLOCK = 256


def fixture(dtype: str = "i2"):
    rows, width = 33, 48
    factor = {"i2": 4, "i4": 2, "i8": 1}[dtype]
    packed = (
        (torch.arange(rows * width // factor, dtype=torch.int32) % 256)
        .to(torch.int8 if dtype == "i8" else torch.uint8)
        .reshape(rows, width // factor)
    )
    groups = 3 if dtype == "i4" else 1
    scale = torch.arange(1, rows * groups + 1, dtype=torch.float32).reshape(rows, groups) / 8
    graph = IrGraph(
        symbols=[],
        inputs=[IrInput(name="ids", dtype="i32", shape=[2])],
        outputs=["y"],
        initializers={"w": IrInitializer(tensor="layer.weight", storage=IrStorage(dtype="f32"))},
        values={
            "w": IrValue(dtype="f32", shape=[rows, width]),
            "y": IrValue(dtype="f32", shape=[2, width]),
        },
        nodes=[IrNode(op="embedding", ins=["w", "ids"], outs=["y"], attrs={"padding_idx": -1})],
    )
    tensors = {"layer.weight": torch.empty(rows, width, device="meta")}
    fixed = {"layer.weight": FixedQuantizedWeight(dtype, packed, scale)}
    return graph, tensors, fixed


def payloads(paths: tuple[Path, ...] | list[Path]) -> dict[str, bytes]:
    """コンテナの全 initializer と companion scale の **payload 部**（詰め物を含まない）。

    piece 列は親 1 本へ繋ぐ（合流が行範囲を持っているので、ここは block を順に繋ぐだけ）。
    """
    verified = verify_container(list(paths), blocks=True)
    result: dict[str, bytes] = {}
    for bound in verified.graphs.values():
        for name, supply in bound.supplies.items():
            result[name] = b"".join(
                verified.read.block(block.id)[: block.payload_bytes] for block in supply.blocks
            )
            if supply.scale is not None:
                result[f"karume.scale.{name}"] = verified.read.block(supply.scale.id)[
                    : supply.scale.payload_bytes
                ]
    return result


def write(tmp_path: Path, graph, tensors, fixed, *, block_bytes=BLOCK_MAX_BYTES, **kwargs):
    """格納変換 → 容器（テストが踏みたい寸法は block 上限の差し込みで作る）。"""
    stored = stored_model(graph, tensors, fixed_weights=fixed, **kwargs)
    result = publish_container(
        tmp_path / "model.krm",
        stored.graph,
        stored.tensors,
        stored.bindings,
        graph_name="fixed",
        provenance=PROVENANCE,
        block_bytes=block_bytes,
    )
    return stored.graph, list(result.parts)


class TestFixedQuantizedWeights:
    @pytest.mark.parametrize("dtype", ["i2", "i4", "i8"])
    @pytest.mark.parametrize("block_bytes", [SMALL_BLOCK, BLOCK_MAX_BYTES])
    def test_preserves_all_packed_bytes_and_scales_with_and_without_row_pieces(
        self, tmp_path, dtype, block_bytes
    ):
        graph, tensors, fixed = fixture(dtype)
        before = graph.to_json()

        committed, paths = write(tmp_path, graph, tensors, fixed, block_bytes=block_bytes)

        actual = payloads(paths)
        weight = fixed["layer.weight"]
        assert actual["layer.weight"] == weight.packed.numpy().tobytes()
        assert actual["karume.scale.layer.weight"] == weight.scale.numpy().tobytes()
        assert set(actual) == {"layer.weight", "karume.scale.layer.weight"}
        assert graph.to_json() == before
        assert tensors["layer.weight"].is_meta
        assert committed.initializers["w"].storage.dtype == dtype
        assert committed.initializers["w"].storage.group_size == (16 if dtype == "i4" else None)
        supply = verify_container(paths).graphs["fixed"].supplies["layer.weight"]
        if block_bytes == SMALL_BLOCK:
            assert len(supply.blocks) > 1
        else:
            assert len(supply.blocks) == 1

    @pytest.mark.parametrize(
        "defect",
        [
            "missing_fixed",
            "real_value",
            "extra_fixed",
            "automatic_dtype",
            "automatic_scales",
            "automatic_override",
            "packed_shape",
            "packed_dtype",
            "packed_stride",
            "scale_dtype",
            "scale_shape",
            "unknown_dtype",
            "non_weight",
            "graph_output",
            "duplicate_key",
            "semantic_dtype",
            "i2_alignment",
            "i4_group",
        ],
    )
    def test_rejects_ambiguous_or_unsupported_inputs_before_writing(self, tmp_path, defect):
        graph, tensors, fixed = fixture("i4" if defect == "i4_group" else "i2")
        key = "layer.weight"
        weight = fixed[key]
        kwargs = {}
        if defect == "missing_fixed":
            fixed = {}
        elif defect == "real_value":
            tensors[key] = torch.zeros_like(tensors[key], device="cpu")
        elif defect == "extra_fixed":
            fixed["unused"] = weight
        elif defect == "automatic_dtype":
            kwargs["weight_dtype"] = "i8"
        elif defect == "automatic_scales":
            kwargs["weight_scales"] = {key: weight.scale}
        elif defect == "automatic_override":
            kwargs["weight_dtype_overrides"] = {key: "f32"}
        elif defect == "packed_shape":
            fixed[key] = replace(weight, packed=weight.packed[:1])
        elif defect == "packed_dtype":
            fixed[key] = replace(weight, packed=weight.packed.to(torch.int8))
        elif defect == "packed_stride":
            fixed[key] = replace(weight, packed=weight.packed.T.contiguous().T)
        elif defect == "scale_dtype":
            fixed[key] = replace(weight, scale=weight.scale.to(torch.float64))
        elif defect == "scale_shape":
            fixed[key] = replace(weight, scale=weight.scale.expand(33, 2))
        elif defect == "unknown_dtype":
            fixed[key] = replace(weight, dtype="i1")
        elif defect == "non_weight":
            graph = replace(graph, nodes=[IrNode(op="mul", ins=["w", "w"], outs=["y"], attrs={})])
        elif defect == "graph_output":
            graph = replace(graph, outputs=["w"])
        elif defect == "duplicate_key":
            graph = replace(
                graph,
                initializers={**graph.initializers, "alias": graph.initializers["w"]},
                values={**graph.values, "alias": graph.values["w"]},
            )
        elif defect == "semantic_dtype":
            tensors[key] = tensors[key].to(torch.float16)
        elif defect == "i2_alignment":
            tensors[key] = torch.empty(33, 44, device="meta")
            graph = replace(
                graph, values={**graph.values, "w": IrValue(dtype="f32", shape=[33, 44])}
            )
            fixed[key] = replace(weight, packed=weight.packed[:, :11].contiguous())
        elif defect == "i4_group":
            fixed[key] = replace(weight, scale=weight.scale[:, :2].contiguous())
        with pytest.raises(EmitError):
            write(tmp_path, graph, tensors, fixed, **kwargs)
        assert list(tmp_path.iterdir()) == []

    def test_publish_validates_fixed_payload_and_preserves_previous_artifact_on_failure(
        self, tmp_path
    ):
        graph, tensors, fixed = fixture()
        path = tmp_path / "model.krm"
        committed = publish_model(
            path, graph, tensors, provenance=PROVENANCE, fixed_weights=fixed, graph_name="fixed"
        )
        assert committed.initializers["w"].storage.dtype == "i2"
        before = {p.name: p.read_bytes() for p in container_parts(path)}
        fixed["layer.weight"] = replace(
            fixed["layer.weight"], packed=torch.zeros(1, dtype=torch.uint8)
        )
        with pytest.raises(EmitError):
            publish_model(
                path,
                graph,
                tensors,
                provenance=PROVENANCE,
                fixed_weights=fixed,
                graph_name="fixed",
            )
        assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before

    def test_mixed_fixed_types_share_the_regular_writer_with_f32_initializers(self, tmp_path):
        base, _, _ = fixture()
        initializers = {}
        values = {}
        nodes = []
        tensors = {}
        fixed = {}
        outputs = []
        for dtype in ("i2", "i4", "i8"):
            graph, source, payload = fixture(dtype)
            name, key, result = f"w_{dtype}", f"layer.{dtype}", f"y_{dtype}"
            initializers[name] = IrInitializer(tensor=key, storage=IrStorage(dtype="f32"))
            values[name], values[result] = graph.values["w"], graph.values["y"]
            nodes.append(
                IrNode(op="embedding", ins=[name, "ids"], outs=[result], attrs={"padding_idx": -1})
            )
            tensors[key], fixed[key] = source["layer.weight"], payload["layer.weight"]
            outputs.append(result)
        # 同じ重みを embedding と linear の両方で読む共有は適格。
        initializers["head_bias"] = IrInitializer(
            tensor="head_bias", storage=IrStorage(dtype="f32")
        )
        tensors["head_bias"] = torch.zeros(33, dtype=torch.float32)
        values["head_bias"] = IrValue(dtype="f32", shape=[33])
        nodes.append(
            IrNode(op="linear", ins=[outputs[0], "w_i2", "head_bias"], outs=["scores"], attrs={})
        )
        values["scores"] = IrValue(dtype="f32", shape=[2, 33])
        outputs.append("scores")
        initializers["bias"] = IrInitializer(tensor="bias", storage=IrStorage(dtype="f32"))
        tensors["bias"] = torch.arange(48, dtype=torch.float32)
        values["bias"] = IrValue(dtype="f32", shape=[48])
        values["adjusted"] = values[outputs[0]]
        nodes.append(IrNode(op="add", ins=[outputs[0], "bias"], outs=["adjusted"], attrs={}))
        outputs.append("adjusted")
        graph = replace(
            base, initializers=initializers, values=values, nodes=nodes, outputs=outputs
        )
        path = tmp_path / "model.krm"
        committed = publish_model(
            path, graph, tensors, provenance=PROVENANCE, fixed_weights=fixed, graph_name="fixed"
        )
        actual = payloads(container_parts(path))
        assert actual["bias"] == tensors["bias"].numpy().tobytes()
        for dtype in ("i2", "i4", "i8"):
            key = f"layer.{dtype}"
            assert committed.initializers[f"w_{dtype}"].storage.dtype == dtype
            assert actual[key] == fixed[key].packed.numpy().tobytes()
            assert actual[f"karume.scale.{key}"] == fixed[key].scale.numpy().tobytes()
