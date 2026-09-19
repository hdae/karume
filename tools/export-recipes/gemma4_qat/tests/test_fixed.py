"""元整数の保持と SRQ / 共有 / PLE の故障を、実モデルなしで検出する。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from torch import nn

pytest.importorskip("transformers")
from transformers.integrations.gemma_quant import QuantizedEmbedding, QuantizedLinear

from gemma4.distribution import assert_gemma4_ple_shards, gemma4_ple_index, gemma4_ple_role
from gemma4_qat.checkpoint import TraceLinear, fixed_trace_weights, load_qat
from gemma4_qat.ple import write_ple
from gemma4_qat.tests.series_fixture import packed_embedding
from karume.dist import DistError


class TestPackedPle:
    @pytest.mark.parametrize("bits", [2, 4])
    def test_multiple_shards_preserve_bytes_and_upstream_probe(self, tmp_path: Path, bits: int):
        module = packed_embedding(bits)
        index = write_ple(module, 3, 32, tmp_path, shard_capacity=3 * 3 * (32 * bits // 8 + 4))
        assert len(index["shards"]) == 3
        parsed = gemma4_ple_index(tmp_path, storage=f"i{bits}")
        placements = {
            gemma4_ple_role(i): tmp_path / s["file"] for i, s in enumerate(index["shards"])
        }
        assert_gemma4_ple_shards(placements, parsed)
        with safe_open(str(tmp_path / "ple.probe.safetensors"), framework="pt") as handle:
            ids = handle.get_tensor("tokens")
            actual = handle.get_tensor("per_layer_inputs")
        with torch.inference_mode():
            expected = module(ids.to(torch.int64).unsqueeze(0)).reshape(1, len(ids), 3, 32)
        assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))
        with pytest.raises(DistError, match="schema"):
            gemma4_ple_index(tmp_path)
        with pytest.raises(DistError, match="storage"):
            gemma4_ple_index(tmp_path, storage="i4" if bits == 2 else "i2")

    @pytest.mark.parametrize("fault", ["schema", "storage", "gap", "dim", "scale"])
    def test_distribution_rejects_inconsistent_index(self, tmp_path: Path, fault: str):
        index = write_ple(packed_embedding(4), 3, 32, tmp_path)
        if fault == "gap":
            index["shards"][0]["start"] = 1
        else:
            key, value = {
                "schema": ("schema", 1),
                "storage": ("storage", "i2"),
                "dim": ("dim", 31),
                "scale": ("embedScale", 0),
            }[fault]
            index[key] = value
        (tmp_path / "ple.json").write_text(json.dumps(index))
        with pytest.raises(DistError):
            gemma4_ple_index(tmp_path, storage="i4")

    def test_shard_metadata_detects_a_different_generation(self, tmp_path: Path):
        index = write_ple(packed_embedding(2), 3, 32, tmp_path)
        parsed = gemma4_ple_index(tmp_path, storage="i2")
        parsed["embedScale"] += 1
        with pytest.raises(DistError, match="食い違う"):
            assert_gemma4_ple_shards({"ple_1": tmp_path / index["shards"][0]["file"]}, parsed)

    @pytest.mark.parametrize("fault", ["dtype", "shape", "negative", "nan", "embedScale"])
    def test_writer_rejects_invalid_fixed_values(self, tmp_path: Path, fault: str):
        module = packed_embedding(4)
        if fault == "dtype":
            module.embedding_quantized = nn.Parameter(
                module.embedding_quantized.to(torch.int8), requires_grad=False
            )
        elif fault == "shape":
            module.embedding_scale = nn.Parameter(torch.ones(9, 1), requires_grad=False)
        elif fault == "embedScale":
            module.scalar_embed_scale = 0
        else:
            module.embedding_scale[0, 0] = -1 if fault == "negative" else float("nan")
        with pytest.raises(ValueError):
            write_ple(module, 3, 32, tmp_path)


class TestTraceLinear:
    @pytest.mark.parametrize("scale", [0.0, 0.03125, 3.118110179901123])
    def test_explicit_srq_matches_upstream_when_given_the_same_weight(self, scale: float):
        source = QuantizedLinear(64, 8, num_bits=4).requires_grad_(False)
        source.weight.copy_(
            (torch.arange(source.weight.numel()) % 256).to(torch.uint8).reshape(source.weight.shape)
        )
        source.weight_scale.fill_(0.125)
        source.input_activation_scale.fill_(scale)
        source.output_activation_scale.fill_(scale)
        traced = TraceLinear(source)
        traced.weight = nn.Parameter(source._dequantize_weights().clone(), requires_grad=False)
        value = torch.arange(3 * 64, dtype=torch.float32).reshape(1, 3, 64) / 17 - 4
        assert torch.equal(traced(value).view(torch.int32), source(value).view(torch.int32))

    @pytest.mark.parametrize(
        ("input_scale", "output_scale", "rounds"),
        [(0.0, 0.0, 0), (0.125, 0.0, 1), (0.0, 0.125, 1), (0.125, 0.25, 2)],
    )
    def test_it_traces_only_the_srq_that_actually_rounds(
        self, input_scale: float, output_scale: float, rounds: int
    ):
        """scale=0 は恒等（ADR 0097 追記 2）なので IR に残さない。

        残すと runtime が decode のたびに恒等コピーを dispatch する（上流で 0 なのは lm_head の
        入出力だけなので、その 1 本は語彙全体の読み書きになる）。入力側・出力側は独立に判定する。
        """
        source = QuantizedLinear(64, 8, num_bits=4).requires_grad_(False)
        source.input_activation_scale.fill_(input_scale)
        source.output_activation_scale.fill_(output_scale)
        exported = torch.export.export(TraceLinear(source), (torch.zeros(1, 3, 64),))
        traced = [
            node
            for node in exported.graph.nodes
            if node.target is torch.ops.karume.static_quantize.default
        ]
        assert len(traced) == rounds

    @pytest.mark.parametrize("scale", [-1.0, float("nan"), float("inf")])
    def test_invalid_srq_scale_is_rejected(self, scale: float):
        source = QuantizedLinear(64, 8).requires_grad_(False)
        source.input_activation_scale.fill_(scale)
        with pytest.raises(ValueError, match="scale"):
            TraceLinear(source)


class TestCheckpointAdmission:
    @pytest.mark.parametrize(
        "model,config",
        [
            ("unknown", {}),
            ("e2b", {"quantization_config": {"quant_method": "other"}}),
            (
                "e4b",
                {
                    "quantization_config": {"quant_method": "gemma", "quantize_embeddings": True},
                    "text_config": {"hidden_size": 1536, "num_hidden_layers": 35},
                },
            ),
        ],
    )
    def test_rejects_wrong_family_before_loading_weights(self, tmp_path: Path, model: str, config):
        (tmp_path / "config.json").write_text(json.dumps(config))
        with pytest.raises(ValueError):
            load_qat(tmp_path, model)

    @pytest.mark.parametrize("fault", [None, "packed", "scale"])
    def test_ties_only_identical_fixed_head_and_embedding(self, fault):
        wrapper = nn.Module()
        wrapper.model = nn.Module()
        wrapper.model.model = nn.Module()
        embedding = QuantizedEmbedding(8, 64, torch.float32, num_bits=2).requires_grad_(False)
        head = QuantizedLinear(64, 8, num_bits=2).requires_grad_(False)
        embedding.embedding_quantized.zero_()
        embedding.embedding_scale.fill_(0.5)
        head.weight.zero_()
        head.weight_scale.fill_(0.5)
        head.input_activation_scale.zero_()
        head.output_activation_scale.zero_()
        wrapper.model.model.embed_tokens = embedding
        wrapper.model.lm_head = head
        if fault:
            (head.weight if fault == "packed" else head.weight_scale)[0, 0] = 1
            with pytest.raises(ValueError, match="一致しない"):
                fixed_trace_weights(wrapper)
        else:
            saved = fixed_trace_weights(wrapper)
            assert saved["model.lm_head.weight"].packed.data_ptr() == head.weight.data_ptr()
            assert wrapper.model.lm_head.weight is wrapper.model.model.embed_tokens.weight
