"""A tiny Gemma 4 26B A4B-shaped checkpoint (`gemma4.export_moe`'s tests).

The `gemma4_text` class with the expert block on, in small numbers: 5 layers (sliding, sliding,
full, sliding, full — a cut may end on layer 2), 8 experts of which the router picks 2, K = V on the
full layers with 1 KV head against 2 on the sliding layers, head widths 16 / 32, no per-layer
embeddings, no KV sharing and a tied head. The checkpoint is BF16 in **two shards plus
`model.safetensors.index.json`**, keyed like 26B's (`model.language_model.*`, the stacked expert
weights without a `.weight` suffix, plus a vision tower and its embedder the export never reads).
Every linear, the embedding and every expert weight lies on a Q4_0 grid as the QAT checkpoint's do
({@link grid_weight}).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file

from karume.quantize import DEFAULT_GROUP_SIZE

VOCAB = 64
HIDDEN = 64
INTERMEDIATE = 96
MOE_INTERMEDIATE = 32
EXPERTS = 8
TOP_K = 2
SLIDING_DEPTH = 16
FULL_DEPTH = 32
HEADS = 4
SLIDING_KV = 2
FULL_KV = 1
WINDOW = 4
S, F = "sliding_attention", "full_attention"
LAYER_TYPES = (S, S, F, S, F)
LAYERS = len(LAYER_TYPES)
FULL_LAYERS = tuple(index for index, kind in enumerate(LAYER_TYPES) if kind == F)
#: The first full layer (a cut may end on it).
CUT = FULL_LAYERS[0] + 1
#: The linears stored as i4: q / k / v / o + 3 MLP + the router's projection on a sliding layer, no
#: `v_proj` on a full one.
INT4_LINEARS = 8 * (LAYERS - len(FULL_LAYERS)) + 7 * len(FULL_LAYERS)
#: The stacked expert weights stored as i4f16 (gate_up and down per layer).
EXPERT_WEIGHTS = 2 * LAYERS
TINY_SYM_MAX = 24
#: The tiny sliding ring (`window + SLIDING_SLACK_ROWS`).
RING = WINDOW + 8

#: The keys outside the text decoder (a vision tower and its embedder — 26B's prefixes).
VISION_KEYS = (
    "model.vision_tower.encoder.layers.0.mlp.fc.weight",
    "model.embed_vision.embedding_projection.weight",
)

#: The shard file names (HF's spelling).
SHARDS = ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors")


def text_config(**overrides: Any) -> Any:
    import transformers

    fields: dict[str, Any] = {
        "vocab_size": VOCAB,
        "hidden_size": HIDDEN,
        "intermediate_size": INTERMEDIATE,
        "moe_intermediate_size": MOE_INTERMEDIATE,
        "num_experts": EXPERTS,
        "top_k_experts": TOP_K,
        "enable_moe_block": True,
        "num_hidden_layers": LAYERS,
        "layer_types": list(LAYER_TYPES),
        "num_attention_heads": HEADS,
        "num_key_value_heads": SLIDING_KV,
        "num_global_key_value_heads": FULL_KV,
        "head_dim": SLIDING_DEPTH,
        "global_head_dim": FULL_DEPTH,
        "attention_k_eq_v": True,
        "num_kv_shared_layers": 0,
        "sliding_window": WINDOW,
        "final_logit_softcapping": 30.0,
        "max_position_embeddings": 256,
        "hidden_size_per_layer_input": 0,
        "tie_word_embeddings": True,
        **overrides,
    }
    return transformers.Gemma4TextConfig(**fields)


def grid_weight(rows: int, cols: int, seed: int) -> torch.Tensor:
    """A BF16 weight on a Q4_0 grid: `bf16(q·d)` per group of 32 along the row, codes in [−8, 7],
    a learned-looking step per group; group 0 of row 0 is all zero and group 1 of row 0 holds a −8
    (`test_export_unified.grid_weight`'s form)."""
    generator = torch.Generator().manual_seed(seed)
    groups = cols // DEFAULT_GROUP_SIZE
    shape = (rows, groups, DEFAULT_GROUP_SIZE)
    codes = torch.randint(-8, 8, shape, generator=generator)
    codes[0, 0] = 0
    codes[0, 1 % groups, 0] = -8
    # Steps in f16's normal range, as every expert step of 26B is (ADR 0145 Context).
    steps = 0.004 * torch.exp(0.3 * torch.randn((rows, groups), generator=generator))
    return (codes * steps.unsqueeze(-1)).reshape(rows, cols).to(torch.bfloat16)


def checkpoint_state(config: Any | None = None) -> dict[str, torch.Tensor]:
    """The text decoder's BF16 tensors keyed as 26B's checkpoint keys them, plus the vision keys."""
    import transformers

    torch.manual_seed(0)
    model = transformers.Gemma4ForCausalLM(config or text_config()).eval()
    state: dict[str, torch.Tensor] = {}
    for index, (key, value) in enumerate(model.state_dict().items()):
        if key == "lm_head.weight":
            continue
        tensor = value
        # `_proj.weight` (attention, MLP) and the router's `proj.weight`.
        if key.endswith("proj.weight") or key == "model.embed_tokens.weight":
            tensor = grid_weight(*value.shape, seed=index)
        elif key.endswith(("experts.gate_up_proj", "experts.down_proj")):
            experts, rows, cols = value.shape
            tensor = grid_weight(experts * rows, cols, seed=index).reshape(experts, rows, cols)
        elif key.endswith("layer_scalar"):
            tensor = torch.full_like(value, 0.75)
        elif key.endswith("per_expert_scale"):
            tensor = 0.5 + torch.rand_like(value)
        elif key.endswith(("norm.weight", "router.scale")):
            tensor = 1 + 0.1 * torch.randn_like(value)
        state[f"model.language_model.{key.removeprefix('model.')}"] = tensor.to(torch.bfloat16)
    for key in VISION_KEYS:
        state[key] = torch.randn(HIDDEN, 8).to(torch.bfloat16)
    return state


def write_checkpoint(
    directory: Path,
    *,
    extra: dict[str, torch.Tensor] | None = None,
    sharded: bool = True,
) -> Path:
    """The tiny checkpoint (BF16 shards + index, or one `model.safetensors`), its `config.json`,
    a stand-in `tokenizer.json` and a `generation_config.json` without `suppress_tokens` (26B's
    declares none)."""
    import transformers

    text = text_config()
    state = {**checkpoint_state(text), **(extra or {})}
    directory.mkdir(parents=True, exist_ok=True)
    tensors = {key: value.contiguous() for key, value in state.items()}
    if sharded:
        keys = sorted(tensors)
        # Layers 0..1 and the embedding in the first shard, the rest in the second.
        first = {
            key
            for key in keys
            if key.startswith(("model.language_model.layers.0.", "model.language_model.layers.1."))
            or key.endswith("embed_tokens.weight")
        }
        weight_map: dict[str, str] = {}
        for name, part in zip(SHARDS, (first, set(keys) - first), strict=True):
            save_file({key: tensors[key] for key in sorted(part)}, str(directory / name))
            weight_map.update(dict.fromkeys(part, name))
        (directory / "model.safetensors.index.json").write_text(
            json.dumps(
                {"metadata": {"total_size": 0}, "weight_map": dict(sorted(weight_map.items()))}
            ),
            encoding="utf-8",
        )
    else:
        save_file(tensors, str(directory / "model.safetensors"))
    transformers.Gemma4Config(text_config=text.to_dict()).save_pretrained(directory)
    (directory / "tokenizer.json").write_text('{"stand-in": true}', encoding="utf-8")
    (directory / "generation_config.json").write_text(
        json.dumps({"do_sample": True, "temperature": 1.0, "top_k": 8, "top_p": 0.9}),
        encoding="utf-8",
    )
    return directory
