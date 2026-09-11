"""QAT text の検査と shape-only trace。元の固定整数を値の唯一の供給元にする。"""

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

from .config import CHECKPOINTS

if TYPE_CHECKING:
    from transformers.integrations.gemma_quant import QuantizedEmbedding, QuantizedLinear


BITS_DTYPE: Mapping[int, Literal["i2", "i4", "i8"]] = {2: "i2", 4: "i4", 8: "i8"}


@dataclass(frozen=True)
class LoadedQat:
    wrapper: ProductChunkWrapper
    ple: QuantizedEmbedding
    config: Any


def load_qat(model_dir: Path, model_name: str) -> LoadedQat:
    """公式 mobile の E2B/E4B を検査して text と PLE だけを保持する。"""
    from transformers import Gemma4ForConditionalGeneration
    from transformers.integrations.gemma_quant import QuantizedEmbedding, QuantizedLinear

    if model_name not in CHECKPOINTS:
        raise ValueError(f"未対応 QAT model: {model_name}")
    raw = json.loads((model_dir / "config.json").read_text())
    quant = raw.get("quantization_config", {})
    if quant.get("quant_method") != "gemma" or quant.get("quantize_embeddings") is not True:
        raise ValueError("gemma 固定量子化と量子化 embedding を持つ mobile 形式が必要")
    config = raw["text_config"]
    expected = (1536, 35) if model_name == "e2b" else (2560, 42)
    if (config["hidden_size"], config["num_hidden_layers"]) != expected:
        raise ValueError(f"QAT {model_name} の hidden/layers が公式構成と違う")
    original = Gemma4ForConditionalGeneration.from_pretrained(
        model_dir,
        local_files_only=True,
        dtype=torch.float32,
        device_map="cpu",
        attn_implementation="eager",
    ).eval()
    text = original.model.language_model
    ple = text.embed_tokens_per_layer
    if not isinstance(ple, QuantizedEmbedding) or ple.num_bits != (4 if model_name == "e2b" else 2):
        raise ValueError("QAT の PLE 格納がモデル名と合わない")
    shim = nn.Module()
    shim.model, shim.lm_head, shim.config = text, original.lm_head, original.config.text_config
    if not isinstance(shim.lm_head, QuantizedLinear) or not isinstance(
        text.embed_tokens, QuantizedEmbedding
    ):
        raise ValueError("QAT の head と token embedding が固定量子化でない")
    del text.embed_tokens_per_layer
    wrapper = ProductChunkWrapper(shim, nn.ModuleList()).eval()
    del wrapper.per_layer
    decode.swap_rope_inputs(shim)
    return LoadedQat(wrapper, ple, shim.config)


def _activation_scale(value: torch.Tensor) -> float:
    if value.device.type != "cpu" or value.dtype != torch.float32 or value.numel() != 1:
        raise ValueError("SRQ scale は CPU f32 の1要素が必要")
    scale = float(value.detach())
    if not torch.isfinite(value).all() or scale < 0:
        raise ValueError("SRQ scale は非負の有限数が必要")
    return scale


class TraceLinear(nn.Module):
    """値の推論には使わない。単一 scalar の expand で巨大な f32 重み確保を避ける。"""

    def __init__(self, source: QuantizedLinear):
        super().__init__()
        self.weight = nn.Parameter(
            torch.zeros(1).expand(source.out_features, source.in_features), requires_grad=False
        )
        self.bias = source.bias
        self.input_scale = _activation_scale(source.input_activation_scale)
        self.output_scale = _activation_scale(source.output_activation_scale)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return static_quantize(
            functional.linear(static_quantize(value, self.input_scale), self.weight, self.bias),
            self.output_scale,
        )


class TraceEmbedding(nn.Module):
    """embedding も shape-only。正式な値は packed payload が供給する。"""

    def __init__(self, source: QuantizedEmbedding):
        super().__init__()
        self.weight = nn.Parameter(
            torch.zeros(1).expand(source.num_embeddings, source.embedding_dim), requires_grad=False
        )
        self.embed_scale = source.scalar_embed_scale

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return functional.embedding(ids, self.weight) * self.embed_scale


def fixed_trace_weights(wrapper: ProductChunkWrapper) -> dict[str, FixedQuantizedWeight]:
    """上流の全 packed 値を退避し、trace 用の構造へ交換する。交換後は通常推論をしない。"""
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
                raise ValueError(f"{name}: 固定量子化されていない予期しない linear")
            continue
        if module.num_bits not in BITS_DTYPE:
            raise ValueError(f"{name}: 固定ビット数 {module.num_bits} に未対応")
        if scale.dtype != torch.float32 or scale.shape != (packed.shape[0], 1):
            raise ValueError(f"{name}: 固定の行 scale F32 [N,1] が必要")
        if not torch.isfinite(scale).all() or not (scale > 0).all():
            raise ValueError(f"{name}: 固定の行 scale が正の有限数でない")
        if module.num_bits == 4:
            group = width & -width
            if group < 16:
                raise ValueError(f"{name}: I4 の group 最小16を満たさない")
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
        raise ValueError("QAT head と token embedding の固定整数・scale が一致しない")
    wrapper.model.lm_head.weight = wrapper.model.model.embed_tokens.weight
    return fixed
