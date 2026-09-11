"""固定 SRQ の数値保存と、単一ノード・記号次元の維持を検査する（ADR 0097）。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from torch import nn

from karume.custom_ops import static_quantize
from karume.ops import OpContractError


class TestStaticQuantize:
    def test_eager_matches_official_cpu_bits_including_boundaries_and_nonfinite_values(self):
        path = (
            Path(__file__).resolve().parents[3]
            / "packages/runtime/tests/fixtures/static-quantize-oracle.safetensors"
        )
        count = 0
        with safe_open(str(path), framework="pt") as file:
            scale_bits = json.loads(file.metadata()["scale_bits"])
            for index, value in enumerate(scale_bits):
                scale = torch.tensor([value], dtype=torch.int32).view(torch.float32).item()
                x, expected = file.get_tensor(f"x{index}"), file.get_tensor(f"y{index}")
                before = x.clone()
                actual = static_quantize(x, scale)
                assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))
                assert torch.equal(x.view(torch.int32), before.view(torch.int32))
                assert actual.data_ptr() != x.data_ptr()
                count += x.numel()
        assert count == 24416

    @pytest.mark.parametrize("scale", [-1.0, float("inf"), float("nan"), 0.1, 1e-50, 1e39])
    def test_eager_rejects_invalid_or_rounded_scales(self, scale):
        with pytest.raises(OpContractError):
            static_quantize(torch.ones(3), scale)

    def test_eager_rejects_integer_input(self):
        with pytest.raises(ValueError, match="f32"):
            static_quantize(torch.ones(3, dtype=torch.int32), 1.0)

    def test_export_keeps_one_node_and_a_dynamic_length(self, convert_module, dyn_t):
        class Quantize(nn.Module):
            def forward(self, x):
                return static_quantize(x, 0.09940945357084274)

        graph, tensors = convert_module(Quantize(), (torch.randn(7, 16),), ({0: dyn_t},))
        assert graph.required_ops == ["static_quantize"]
        assert len(graph.nodes) == 1
        assert graph.nodes[0].attrs == {"scale": 0.09940945357084274}
        assert graph.values[graph.outputs[0]].shape == graph.inputs[0].shape
        assert isinstance(graph.inputs[0].shape[0], str)
        assert not tensors

    @pytest.mark.parametrize("scale", [-1.0, 0.1, 1e-50, 1e39])
    def test_export_rejects_invalid_scales_even_when_fake_execution_skips_eager(
        self, convert_module, scale
    ):
        class Quantize(nn.Module):
            def forward(self, x):
                return static_quantize(x, scale)

        with pytest.raises(OpContractError):
            convert_module(Quantize(), (torch.ones(3),))
