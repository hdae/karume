"""A tiny Qwen3.5 text tower with random weights, and its checkpoint on disk, for the text-tower
tests (no real weights needed).

The structure is the real model's in miniature: three GDN layers and one full-attention layer
(one period of `full_attention_interval` 4), GQA in both mixers (GDN value heads 2× the key heads,
attention heads 2× the k/v heads), partial rotary (8 of 32 dims) with an mRoPE section that covers
it. Every norm weight is drawn away from upstream's initial value (zeros for the zero-centred
norms, ones for the GDN norm), so a wrong `1 + w` offset changes the numbers.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file

#: The text configuration (fields of `Qwen3_5TextConfig`).
TINY_TEXT: dict[str, Any] = {
    "vocab_size": 320,
    "hidden_size": 64,
    "intermediate_size": 96,
    "num_hidden_layers": 4,
    "layer_types": ["linear_attention"] * 3 + ["full_attention"],
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 32,
    "linear_num_key_heads": 2,
    "linear_num_value_heads": 4,
    "linear_key_head_dim": 16,
    "linear_value_head_dim": 16,
    "linear_conv_kernel_dim": 4,
    "rms_norm_eps": 1e-6,
    "hidden_act": "silu",
    "rope_parameters": {
        "rope_type": "default",
        "rope_theta": 1e7,
        "partial_rotary_factor": 0.25,
        "mrope_section": [2, 1, 1],
        "mrope_interleaved": True,
    },
}

#: The checkpoint's key prefix (a `Qwen3_5Model` checkpoint, as MOSS-Image bundles it) and the
#: one vision-tower key the fixture adds (the reader must ignore it).
PREFIX = "language_model."
VISION_KEY = "visual.patch_embed.proj.weight"


def text_config(**overrides: Any) -> Any:
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

    return Qwen3_5TextConfig(**{**TINY_TEXT, **overrides})


def stock_model(seed: int = 20261031, **overrides: Any) -> Any:
    """The upstream `Qwen3_5TextModel` (f32, eval, sdpa) with seeded random weights, rounded to
    bf16 values (the checkpoint stores bf16) and the norms drawn away from their initial values."""
    from transformers.models.qwen3_5 import modeling_qwen3_5 as m

    config = text_config(**overrides)
    config._attn_implementation = "sdpa"
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = m.Qwen3_5TextModel(config).float().eval()
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if name.endswith("norm.weight"):
                    parameter.normal_(0.0, 0.2)
                    if name.endswith("linear_attn.norm.weight"):
                        parameter.add_(1.0)
                parameter.copy_(parameter.to(torch.bfloat16).float())
    return model


def write_checkpoint(
    model: Any, directory: Path, *, extra: dict[str, torch.Tensor] | None = None
) -> Path:
    """Save `model`'s weights as a two-shard BF16 `Qwen3_5Model`-style checkpoint (`language_model.`
    keys, one vision key, a `model.safetensors.index.json`) plus its `config.json`."""
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config

    directory.mkdir(parents=True, exist_ok=True)
    state = {
        f"{PREFIX}{name}": tensor.detach().to(torch.bfloat16).contiguous()
        for name, tensor in model.state_dict().items()
    }
    state[VISION_KEY] = torch.zeros(4, 3, dtype=torch.bfloat16)
    state.update(extra or {})
    names = sorted(state)
    shards = {
        "model-00001-of-00002.safetensors": names[: len(names) // 2],
        "model-00002-of-00002.safetensors": names[len(names) // 2 :],
    }
    weight_map: dict[str, str] = {}
    for shard, keys in shards.items():
        save_file({key: state[key] for key in keys}, str(directory / shard))
        weight_map.update(dict.fromkeys(keys, shard))
    index = {"metadata": {}, "weight_map": weight_map}
    (directory / "model.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")
    Qwen3_5Config(text_config=model.config.to_dict()).save_pretrained(directory)
    return directory
