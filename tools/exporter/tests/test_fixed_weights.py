"""固定量子化の保存値と、曖昧な値の供給を拒否する契約（ADR 0097）。"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from karume.emit import EmitError, FixedQuantizedWeight, write_model
from karume.ir import IrGraph, IrInitializer, IrInput, IrNode, IrStorage, IrValue
from karume.pipeline import publish_model
from karume.shards import parse_piece_key, resolve_shards
from karume.verify import verify_shards


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


def payloads(paths: list[Path]) -> dict[str, bytes]:
    result: dict[str, bytes] = {}
    for path in paths:
        data = path.read_bytes()
        length = int.from_bytes(data[:8], "little")
        base = 8 + length
        for name, entry in json.loads(data[8:base]).items():
            if name == "__metadata__":
                continue
            begin, end = entry["data_offsets"]
            piece = parse_piece_key(name)
            key = name if piece is None else piece[0]
            result[key] = result.get(key, b"") + data[base + begin : base + end]
    return result


class TestFixedQuantizedWeights:
    @pytest.mark.parametrize("dtype", ["i2", "i4", "i8"])
    @pytest.mark.parametrize("capacity", [512, 8192])
    def test_preserves_all_packed_bytes_and_scales_with_and_without_row_pieces(
        self, tmp_path, dtype, capacity
    ):
        graph, tensors, fixed = fixture(dtype)
        before = graph.to_json()
        paths = write_model(
            tmp_path / "model.safetensors",
            graph,
            tensors,
            fixed_weights=fixed,
            _shard_capacity=capacity,
        )
        verified = verify_shards(paths)
        actual = payloads(paths)
        weight = fixed["layer.weight"]
        assert actual["layer.weight"] == weight.packed.numpy().tobytes()
        assert actual["karume.scale.layer.weight"] == weight.scale.numpy().tobytes()
        assert set(actual) == {"layer.weight", "karume.scale.layer.weight"}
        assert graph.to_json() == before
        assert tensors["layer.weight"].is_meta
        assert verified.initializers["w"].storage.dtype == dtype
        assert verified.initializers["w"].storage.group_size == (16 if dtype == "i4" else None)
        if capacity == 512:
            assert len(paths) > 2
        else:
            assert len(paths) == 2

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
            write_model(
                tmp_path / "model.safetensors", graph, tensors, fixed_weights=fixed, **kwargs
            )
        assert list(tmp_path.iterdir()) == []

    def test_publish_validates_fixed_payload_and_preserves_previous_artifact_on_failure(
        self, tmp_path
    ):
        graph, tensors, fixed = fixture()
        path = tmp_path / "model.safetensors"
        verified = publish_model(path, graph, tensors, fixed_weights=fixed)
        assert verified.initializers["w"].storage.dtype == "i2"
        before = {p.name: p.read_bytes() for p in resolve_shards(path)}
        fixed["layer.weight"] = replace(
            fixed["layer.weight"], packed=torch.zeros(1, dtype=torch.uint8)
        )
        with pytest.raises(EmitError):
            publish_model(path, graph, tensors, fixed_weights=fixed)
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
        path = tmp_path / "model.safetensors"
        verified = publish_model(path, graph, tensors, fixed_weights=fixed)
        actual = payloads(resolve_shards(path))
        assert actual["bias"] == tensors["bias"].numpy().tobytes()
        for dtype in ("i2", "i4", "i8"):
            key = f"layer.{dtype}"
            assert verified.initializers[f"w_{dtype}"].storage.dtype == dtype
            assert actual[key] == fixed[key].packed.numpy().tobytes()
            assert actual[f"karume.scale.{key}"] == fixed[key].scale.numpy().tobytes()
