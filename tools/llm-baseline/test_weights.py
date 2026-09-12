import json

import pytest
import torch
from safetensors.torch import save_file
from weights import DiskPle, assert_srq_scales


def test_original_ple_row_lookup_preserves_values_and_repeated_indices(tmp_path):
    value = torch.arange(8 * 12, dtype=torch.float32).reshape(8, 12).div(7).to(torch.bfloat16)
    save_file(
        {"model.language_model.embed_tokens_per_layer.weight": value},
        tmp_path / "model.safetensors",
    )
    ids = torch.tensor([[7, 0, 7, 1]])
    module = DiskPle(tmp_path, 4, None)
    assert torch.equal(module(ids), value.float()[ids] * 4)


def test_quantized_ple_keeps_layer_scales_and_shard_boundaries(tmp_path):
    values = torch.arange(-24, 24, dtype=torch.int8).reshape(8, 2, 3)
    scales = torch.arange(1, 17, dtype=torch.float32).reshape(8, 2) / 16
    shards = []
    for index, (start, stop) in enumerate([(0, 3), (3, 8)]):
        file = f"ple-{index}.safetensors"
        save_file({"values": values[start:stop], "scales": scales[start:stop]}, tmp_path / file)
        shards.append({"file": file, "start": start, "stop": stop})
    index = tmp_path / "ple.json"
    index.write_text(json.dumps({"schema": 1, "embedScale": 4, "shards": shards}))
    module = DiskPle(tmp_path, 4, index)
    ids = torch.tensor([[7, 2, 3, 2, 0]])
    expected = (values.float() * scales.unsqueeze(-1)).flatten(1)[ids] * 4
    assert torch.equal(module(ids), expected)


def test_qat_reference_rejects_mismatched_activation_rounding():
    graph = {
        "nodes": [
            {"op": "static_quantize", "ins": ["x"], "outs": ["rounded"], "attrs": {"scale": 0.25}},
            {"op": "linear", "ins": ["rounded", "w"], "outs": ["y"]},
            {"op": "static_quantize", "ins": ["y"], "outs": ["z"], "attrs": {"scale": 0.5}},
        ]
    }
    assert assert_srq_scales(graph, "w", 0.25, 0.5) == 2
    with pytest.raises(ValueError, match="input SRQ"):
        assert_srq_scales(graph, "w", 0.125, 0.5)
    with pytest.raises(ValueError, match="output SRQ"):
        assert_srq_scales(graph, "w", 0.25, 1.0)
