"""Checking the QAT text model and the shape-only trace. The original fixed integers are the only
source of values."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import torch
from torch import nn
from torch.nn import functional

from gemma4 import export_decode as decode
from gemma4.export_product import ProductChunkWrapper
from karume.custom_ops import static_quantize
from karume.emit import FixedQuantizedWeight

from .config import CHECKPOINTS, PLE_BITS

if TYPE_CHECKING:
    from transformers.integrations.gemma_quant import QuantizedEmbedding, QuantizedLinear


BITS_DTYPE: Mapping[int, Literal["i2", "i4", "i8"]] = {2: "i2", 4: "i4", 8: "i8"}


@dataclass(frozen=True)
class LoadedQat:
    wrapper: ProductChunkWrapper
    ple: QuantizedEmbedding
    config: Any


def load_qat(model_dir: Path, model_name: str) -> LoadedQat:
    """Check the official mobile E2B/E4B and keep only the text model and PLE."""
    from transformers import Gemma4ForConditionalGeneration
    from transformers.integrations.gemma_quant import QuantizedEmbedding, QuantizedLinear

    if model_name not in CHECKPOINTS:
        raise ValueError(f"unsupported QAT model: {model_name}")
    raw = json.loads((model_dir / "config.json").read_text())
    quant = raw.get("quantization_config", {})
    if quant.get("quant_method") != "gemma" or quant.get("quantize_embeddings") is not True:
        raise ValueError(
            "the mobile format with gemma fixed quantization and quantized embeddings is required"
        )
    config = raw["text_config"]
    expected = (1536, 35) if model_name == "e2b" else (2560, 42)
    if (config["hidden_size"], config["num_hidden_layers"]) != expected:
        raise ValueError(f"QAT {model_name}: hidden/layers differ from the official configuration")
    original = Gemma4ForConditionalGeneration.from_pretrained(
        model_dir,
        local_files_only=True,
        dtype=torch.float32,
        device_map="cpu",
        attn_implementation="eager",
    ).eval()
    text = original.model.language_model
    ple = text.embed_tokens_per_layer
    if not isinstance(ple, QuantizedEmbedding) or ple.num_bits != PLE_BITS[model_name]:
        raise ValueError("the QAT PLE storage does not match the model name")
    shim = nn.Module()
    shim.model, shim.lm_head, shim.config = text, original.lm_head, original.config.text_config
    if not isinstance(shim.lm_head, QuantizedLinear) or not isinstance(
        text.embed_tokens, QuantizedEmbedding
    ):
        raise ValueError("the QAT head and token embedding are not fixed-quantized")
    del text.embed_tokens_per_layer
    wrapper = ProductChunkWrapper(shim, nn.ModuleList()).eval()
    del wrapper.per_layer
    decode.swap_rope_inputs(shim)
    return LoadedQat(wrapper, ple, shim.config)


def _activation_scale(value: torch.Tensor) -> float:
    if value.device.type != "cpu" or value.dtype != torch.float32 or value.numel() != 1:
        raise ValueError("an SRQ scale must be a single CPU f32 element")
    scale = float(value.detach())
    if not torch.isfinite(value).all() or scale < 0:
        raise ValueError("an SRQ scale must be a non-negative finite number")
    return scale


class TraceLinear(nn.Module):
    """Not used to infer values. An expand of a single scalar avoids allocating huge f32
    weights."""

    def __init__(self, source: QuantizedLinear):
        super().__init__()
        self.weight = nn.Parameter(
            torch.zeros(1).expand(source.out_features, source.in_features), requires_grad=False
        )
        self.bias = source.bias
        self.input_scale = _activation_scale(source.input_activation_scale)
        self.output_scale = _activation_scale(source.output_activation_scale)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        # scale=0 is the identity by contract (ADR 0097 addendum 2), so no SRQ is inserted.
        # Inserting one would leave an identity `static_quantize` in the IR, and the runtime
        # queues a dispatch even with scale=0 — upstream has 0 only for lm_head's input and
        # output, and that output spans the whole vocabulary, so every decode step would run an
        # identity copy of the whole vocabulary. Input and output are decided independently (a
        # shape with 0 on one side only passes as it is).
        if self.input_scale != 0.0:
            value = static_quantize(value, self.input_scale)
        result = functional.linear(value, self.weight, self.bias)
        if self.output_scale != 0.0:
            result = static_quantize(result, self.output_scale)
        return result


class TraceEmbedding(nn.Module):
    """The embedding is shape-only too. The packed payload supplies the real values."""

    def __init__(self, source: QuantizedEmbedding):
        super().__init__()
        self.weight = nn.Parameter(
            torch.zeros(1).expand(source.num_embeddings, source.embedding_dim), requires_grad=False
        )
        self.embed_scale = source.scalar_embed_scale

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return functional.embedding(ids, self.weight) * self.embed_scale


def fixed_trace_weights(wrapper: ProductChunkWrapper) -> dict[str, FixedQuantizedWeight]:
    """Set aside every upstream packed value and swap in the trace structures. No ordinary
    inference runs after the swap."""
    from transformers.integrations.gemma_quant import QuantizedEmbedding, QuantizedLinear

    fixed = {}
    for name, module in list(wrapper.named_modules()):
        if isinstance(module, QuantizedEmbedding):
            replacement = TraceEmbedding(module)
            packed, scale, width = (
                module.embedding_quantized,
                module.embedding_scale,
                module.embedding_dim,
            )
        elif isinstance(module, QuantizedLinear):
            replacement = TraceLinear(module)
            packed, scale, width = module.weight, module.weight_scale, module.in_features
        else:
            if isinstance(module, nn.Linear) and name != "model.model.per_layer_model_projection":
                raise ValueError(f"{name}: an unexpected linear without fixed quantization")
            continue
        if module.num_bits not in BITS_DTYPE:
            raise ValueError(f"{name}: unsupported fixed bit width {module.num_bits}")
        if scale.dtype != torch.float32 or scale.shape != (packed.shape[0], 1):
            raise ValueError(f"{name}: a fixed per-row scale F32 [N,1] is required")
        if not torch.isfinite(scale).all() or not (scale > 0).all():
            raise ValueError(f"{name}: the fixed per-row scale is not positive and finite")
        if module.num_bits == 4:
            group = width & -width
            if group < 16:
                raise ValueError(f"{name}: the I4 group is below the minimum of 16")
            scale = scale.expand(scale.shape[0], width // group).contiguous()
        fixed[name + ".weight"] = FixedQuantizedWeight(
            BITS_DTYPE[module.num_bits], packed.detach(), scale.detach()
        )
        parent, _, leaf = name.rpartition(".")
        setattr(wrapper.get_submodule(parent), leaf, replacement)
    embedding, head = "model.model.embed_tokens.weight", "model.lm_head.weight"
    a, b = fixed[embedding], fixed[head]
    if (
        a.dtype != "i2"
        or a.dtype != b.dtype
        or not torch.equal(a.packed, b.packed)
        or not torch.equal(a.scale, b.scale)
    ):
        raise ValueError(
            "the fixed integers / scales of the QAT head and token embedding do not match"
        )
    wrapper.model.lm_head.weight = wrapper.model.model.embed_tokens.weight
    return fixed
