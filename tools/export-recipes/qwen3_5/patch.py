"""The Qwen3.5 text tower as one IR graph with a symbolic length `T` (prefill only, B = 1, no
cache), and the recipe-side rewrites that make it export.

The wrapper {@link Qwen35TextTower} takes `input_ids [1, T]`, `positions [T]` and the RoPE rows
`rope_cos` / `rope_sin [1, T, rotary_dim]` ({@link INPUT_NAMES}) and returns `last_hidden_state
[1, T, hidden]` after the final norm — what MOSS-Image takes (`pipeline_moss_image.py`
`encode_prompt`). The decoder layers, the embedding, the norms' formula and the attention / GDN
mixers are upstream's modules (transformers 5.14.1, `models/qwen3_5/modeling_qwen3_5.py`); the
wrapper replaces only `Qwen3_5TextModel.forward`'s preamble (position ids, the mask mapping, the
rotary module), which the host now supplies.

## The rewrites (each one leaves `forward` of the upstream modules alone)

1. **The recurrence** (ADR 0135 decision 5): the instance attributes `chunk_gated_delta_rule` /
   `recurrent_gated_delta_rule` of every `Qwen3_5GatedDeltaNet` become {@link gated_delta_rule},
   which l2-normalises q / k ({@link l2_normalize}) and calls the custom op
   `karume::gated_delta_scan`. The wrapper receives q / k after `forward`'s `repeat_interleave`, so
   the head layout is r = 1 (ADR 0135 open point 4). The recurrent attribute is set to a function
   that fails: it is reached only with a cache, which this graph never has.
2. **The l2-normalisation** (ADR 0135 open point 2): `x·rsqrt(Σx² + eps)` written as
   `rms_norm(x, (d,), eps/d)·d^-½`, which exports as [rms_norm, mul] (`rsqrt` is not IR
   vocabulary). Not bit-identical to upstream (≤ 4 ULP — research 2026-10-31 §6.1).
3. **The causal conv** (ADR 0135 open point 3): the instance attribute `causal_conv1d_fn` becomes
   {@link causal_conv1d} (left pad K−1, `conv1d(padding=0)`, SiLU), so `forward` takes the branch
   without the trailing `[..., :T]` slice on the symbolic axis. The conv is bit-identical to
   upstream's at every T measured.
4. **The zero-centred norms** `Qwen3_5RMSNorm` (`x̂·(1 + w)` in f32 — the 2 norms of every layer,
   q_norm / k_norm of the full-attention layers, the final norm: 81 in Qwen3.5-9B) become
   {@link UnitOffsetRmsNorm}, whose stored weight is `1 + w` computed once in f32. Upstream
   computes the same `1.0 + w.float()` on every call, so the stored value is the value upstream
   multiplies by, bit for bit; the graph loses one `add` per norm. MUST: the container's
   `….weight` of these norms therefore holds `1 + w`, not the checkpoint value — the writer applies
   the offset to exactly the keys {@link prepare_text_tower} reports ({@link
   Qwen35TextTower.unit_offset_keys}).
5. **The attention implementation**: transformers' `sdpa` materialises `repeat_kv` whenever a mask
   is passed; {@link gqa_attention} (registered under {@link ATTENTION_NAME} — the `minicpm5` /
   `gemma4` precedent) calls SDPA with `enable_gqa=True`, so the IR attention reads k / v with
   their 4 heads. It never lets `is_causal` reach SDPA: a missing mask fails loudly.

## The causal mask: built in the graph from `positions`

The runtime's `attention` has no causal field (`aten_handlers._h_attention` rejects
`is_causal=True`), and the folded form of the precedents (`minicpm5` / `gemma4`: arange-only leaves
folded to a Tmax×Tmax f32 constant + `sym_prefix_slice`) would be 64 MiB at Tmax = 4096. The
wrapper instead builds the bool mask `positions[:, None] >= positions[None, :]` ({@link
causal_mask}) from the host-supplied `positions` (f32 `[T]`, 0 … T−1): a run-time `ge` that the
exporter's bool → additive rewrite (`normalize._additive_attn_mask`) turns into one `where(mask, 0,
−inf)` shared by every attention layer. Host input O(T), no T² constant, no T² upload.

## RoPE: host-supplied rows

Qwen3.5 rotates the first `rotary_dim = head_dim·partial_rotary_factor` (64 of 256) dimensions with
theta 1e7. Its rotary module is interleaved mRoPE over three position rows; for text-only input
the three rows are the same `arange(T)`, so the interleave copies equal values and the rows are
exactly 1-D RoPE. The rotary module writes `freqs_t[..., idx] = …` in place and computes `cos`,
which is not IR vocabulary, so it is not traced: the host passes `rope_cos` / `rope_sin` (the
`gemma4` ADR 0091 / MOSS `dit_text` precedent), and {@link rope_rows} is the formula's Python
mirror (f64, stored as f32 — not bit-identical to upstream's f32 rotary module; the bound is
ADR 0091 decision 2's position-proportional one).

MUST: transformers is imported inside functions (it is not in the default sync —
`tests/test_optional_group_imports.py`).
"""

from __future__ import annotations

import importlib.util
from collections.abc import Sequence
from typing import Any

import torch
from torch import nn
from torch.export import Dim
from torch.nn import functional

from karume import custom_ops  # noqa: F401 — registers torch.ops.karume

#: Graph input names (the order of {@link Qwen35TextTower.forward}'s arguments).
INPUT_NAMES = ("input_ids", "positions", "rope_cos", "rope_sin")

#: The symbolic length.
SYMBOL = "T"

#: The range of `T`. MOSS-Image's prompts are 10 … 4096 tokens (the empty prompt renders to 10 —
#: the chat template alone; `max_sequence_length` 4096), and the graph has nothing that needs more
#: than one token.
MIN_TOKENS = 1
MAX_TOKENS = 4096

#: The layer types this wrapper writes out (`config.layer_types`).
LINEAR_ATTENTION = "linear_attention"
FULL_ATTENTION = "full_attention"

#: The name {@link gqa_attention} is registered under. MUST NOT contain `sdpa` / `flash_attention`
#: / `flex_attention`: transformers branches into backend-specific dispatch checks on a substring
#: match of the name (the `minicpm5` / `gemma4` precedent).
ATTENTION_NAME = "karume_qwen3_5"

#: The l2-normalisation's epsilon (upstream `l2norm(x, dim=-1, eps=1e-6)`, called by both
#: recurrence functions with `use_qk_l2norm_in_kernel=True`).
L2NORM_EPS = 1e-6


class Qwen35ConfigError(ValueError):
    """The upstream configuration is outside what the wrapper writes out (it would compute
    something else silently)."""


def rotary_dim(config: Any) -> int:
    """The rotated width of a head (`head_dim · partial_rotary_factor` — upstream's
    `compute_default_rope_parameters`)."""
    head_dim = int(config.head_dim)
    factor = float(config.rope_parameters.get("partial_rotary_factor", 1.0))
    return int(head_dim * factor)


def check_supported(config: Any) -> None:
    """Fail loudly when the text configuration is outside what this wrapper writes out."""
    problems: list[str] = []
    kinds = set(config.layer_types[: config.num_hidden_layers])
    if not kinds <= {LINEAR_ATTENTION, FULL_ATTENTION}:
        problems.append(f"layer types {sorted(kinds)}")
    if len(config.layer_types) < config.num_hidden_layers:
        problems.append("fewer layer types than layers")
    if config.hidden_act != "silu":
        problems.append(f"hidden_act {config.hidden_act!r} (the causal-conv replacement is SiLU)")
    rope = config.rope_parameters
    if rope.get("rope_type", "default") != "default":
        problems.append(f"rope_type {rope.get('rope_type')!r} ({rope_rows.__name__} is 'default')")
    width = rotary_dim(config)
    if width < 2 or width % 2 or width > int(config.head_dim):
        problems.append(f"rotary_dim {width}")
    section = rope.get("mrope_section")
    if section is not None and sum(section) != width // 2:
        problems.append(f"mrope_section {section} does not cover rotary_dim/2 = {width // 2}")
    if config.linear_num_value_heads % config.linear_num_key_heads:
        problems.append("GDN value heads are not a multiple of the key heads")
    if config.num_attention_heads % config.num_key_value_heads:
        problems.append("attention heads are not a multiple of the k/v heads")
    if problems:
        raise Qwen35ConfigError("; ".join(problems))


def assert_torch_paths() -> None:
    """MUST (ADR 0135 decision 5): the stock torch `forward` and functions run — no hub kernel, no
    `flash-linear-attention`, no `causal-conv1d`.

    Where `kernels` is active, `@use_kernel_forward_from_hub` may replace `forward` itself (the
    attribute swaps would then miss); where `fla` is installed the GDN norm is `FusedRMSNormGated`
    and the recurrence a Triton kernel (the reference run would no longer be upstream's torch
    functions). The exporter's venv has none of them; this fails loudly if that changes.
    """
    from transformers.models.qwen3_5 import modeling_qwen3_5 as upstream

    present = [
        name for name in ("kernels", "fla", "causal_conv1d") if importlib.util.find_spec(name)
    ]
    bound = [
        name
        for name in ("causal_conv1d_fn", "chunk_gated_delta_rule", "FusedRMSNormGated")
        if getattr(upstream, name) is not None
    ]
    if present or bound:
        raise Qwen35ConfigError(
            f"upstream fast paths are installed ({present}) or bound ({bound}) — the recipe relies"
            " on the stock torch forward and functions"
        )


def l2_normalize(x: torch.Tensor, eps: float = L2NORM_EPS) -> torch.Tensor:
    """Upstream's `l2norm` (`x·rsqrt(Σx² + eps)`) as `rms_norm(x, (d,), eps/d)·d^-½`.

    `rms_norm(x, eps/d) = x / sqrt((Σx² + eps)/d)`, so the product is the same function; it
    exports as [rms_norm, mul] (ADR 0135 open point 2).
    """
    width = int(x.shape[-1])
    return functional.rms_norm(x, (width,), eps=eps / width) * (width**-0.5)


def gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    **kwargs: Any,
) -> tuple[torch.Tensor, None]:
    """The prefill recurrence through `karume::gated_delta_scan` (called as `forward` calls
    `chunk_gated_delta_rule`: q / k `[B,T,Hv,dk]` already repeated, v `[B,T,Hv,dv]`, g / β
    `[B,T,Hv]`).

    MUST: only the prefill from a zero state with the in-kernel l2-normalisation — anything else is
    another computation (a cache, a final state, packed sequences).
    """
    unexpected = sorted(name for name, given in kwargs.items() if given is not None)
    if initial_state is not None or output_final_state or not use_qk_l2norm_in_kernel or unexpected:
        raise NotImplementedError(
            "gated_delta_rule: only the stateless prefill with use_qk_l2norm_in_kernel=True"
            f" (initial_state={initial_state is not None}, output_final_state={output_final_state},"
            f" use_qk_l2norm_in_kernel={use_qk_l2norm_in_kernel}, kwargs={unexpected})"
        )
    output = torch.ops.karume.gated_delta_scan(
        l2_normalize(query), l2_normalize(key), value, g, beta
    )
    return output, None


def recurrent_gated_delta_rule(*args: Any, **kwargs: Any) -> None:
    """Upstream's single-token cached decode path. MUST NOT be reached (the graph has no cache)."""
    del args, kwargs
    raise NotImplementedError("the decode recurrence is not part of the prefill graph")


def causal_conv1d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: str | None = None,
    seq_idx: torch.Tensor | None = None,
    **kwargs: Any,
) -> torch.Tensor:
    """Drop-in for the `causal-conv1d` package's function as `forward` calls it (`x [B,C,T]`,
    `weight [C,K]` → `[B,C,T]`): left pad K−1, depthwise `conv1d(padding=0)`, SiLU."""
    if activation != "silu" or seq_idx is not None or kwargs:
        raise NotImplementedError(
            f"causal_conv1d: activation {activation!r}, seq_idx {seq_idx is not None},"
            f" kwargs {sorted(kwargs)} (SiLU without packed sequences only)"
        )
    channels, kernel = weight.shape
    padded = functional.pad(x, (kernel - 1, 0))
    out = functional.conv1d(padded, weight.unsqueeze(1), bias, padding=0, groups=channels)
    return functional.silu(out)


class UnitOffsetRmsNorm(nn.Module):
    """`Qwen3_5RMSNorm` with the `(1 + w)` folded into the stored weight.

    `forward` is upstream's line by line (`_norm(x.float()) · weight` then `type_as`), with
    `weight` = upstream's `1.0 + w.float()` computed once. The parameter keeps the name `weight` so
    the tensor key stays the checkpoint's; its value is the offset one.
    """

    def __init__(self, weight: torch.Tensor, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = x.float()
        output = hidden * torch.rsqrt(hidden.pow(2).mean(-1, keepdim=True) + self.eps)
        return (output * self.weight).type_as(x)


def unit_offset(weight: torch.Tensor) -> torch.Tensor:
    """The value a `Qwen3_5RMSNorm` multiplies by: `1.0 + w.float()` (upstream's expression)."""
    return 1.0 + weight.float()


def _replace_unit_offset_norms(root: nn.Module) -> tuple[str, ...]:
    """Replace every `Qwen3_5RMSNorm` under `root` by {@link UnitOffsetRmsNorm}; return their
    tensor keys (`<module>.weight`, in `root`'s name space).

    MUST: matched by exact type — `Qwen3_5RMSNormGated` (the GDN norm, `w · x̂` with weight
    initialised to ones) is a different class and is not offset.
    """
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5RMSNorm

    keys: list[str] = []
    for name, module in list(root.named_modules()):
        if type(module) is not Qwen3_5RMSNorm:
            continue
        parent_name, _, attribute = name.rpartition(".")
        parent = root.get_submodule(parent_name) if parent_name else root
        with torch.no_grad():
            replacement = UnitOffsetRmsNorm(unit_offset(module.weight.detach()), module.eps)
        setattr(parent, attribute, replacement)
        keys.append(f"{name}.weight")
    return tuple(keys)


def gqa_attention(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float = 0.0,
    scaling: float | None = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, None]:
    """SDPA with `enable_gqa=True` and the caller's mask (transformers' `AttentionInterface`
    convention: `(module, q, k, v, mask, dropout=…, scaling=…)` → `(out [B,M,H,D], weights)`).

    What it drops from upstream's `sdpa_attention_forward`: the `repeat_kv` materialisation and the
    automatic `is_causal`. MUST: a missing mask fails loudly — upstream reads it as "let SDPA be
    causal", which the IR cannot express, and passing it through would export a non-causal
    attention that no later check sees.
    """
    if attention_mask is None:
        raise ValueError(
            f"{type(module).__name__}: attention called without a mask (causality is the mask)"
        )
    if float(dropout) != 0.0:
        raise ValueError(f"dropout={dropout} (inference only)")
    unsupported = sorted(
        name for name in ("is_causal", "position_bias", "sliding_window") if kwargs.get(name)
    )
    if unsupported:
        raise ValueError(f"attention kwargs {unsupported} change the computation")
    output = functional.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attention_mask,
        dropout_p=0.0,
        scale=scaling,
        is_causal=False,
        enable_gqa=True,
    )
    return output.transpose(1, 2).contiguous(), None


def register_attention() -> None:
    """Register {@link gqa_attention} with transformers' attention registry (idempotent)."""
    from transformers import AttentionInterface

    AttentionInterface.register(ATTENTION_NAME, gqa_attention)


def causal_mask(positions: torch.Tensor) -> torch.Tensor:
    """The bool causal mask `[1, 1, T, T]` (`True` = attend): key position ≤ query position.

    MUST: `positions` is a graph input (f32 `[T]`), so the comparison stays a run-time `ge` — a
    mask whose leaves were `arange` would fold into a Tmax×Tmax constant. Written as `rows >= cols`
    (`aten.ge.Tensor` = IR `ge`, f32 × f32 → bool).
    """
    rows = positions.unsqueeze(-1)
    cols = positions.unsqueeze(0)
    return (rows >= cols).unsqueeze(0).unsqueeze(0)


class Qwen35TextTower(nn.Module):
    """`Qwen3_5TextModel` as `(input_ids, positions, rope_cos, rope_sin) → last_hidden_state`.

    Holds upstream's `embed_tokens`, `layers` and `norm` directly (the tensor keys are
    `embed_tokens.weight`, `layers.<n>.…`, `norm.weight`); the rotary module stays out (the host
    supplies its rows). Build it with {@link prepare_text_tower}, which applies the rewrites.

    MUST: an eval-mode f32 upstream only — the casts in upstream's norms and attention are no-ops
    in f32, and the graph is the f32 computation.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        config = model.config
        check_supported(config)
        if model.training:
            raise Qwen35ConfigError("wrap an eval() upstream (no dropout in the graph)")
        dtypes = {parameter.dtype for parameter in model.parameters()}
        if dtypes != {torch.float32}:
            raise Qwen35ConfigError(f"parameter dtypes {sorted(map(str, dtypes))} — f32 only")
        if config._attn_implementation != ATTENTION_NAME:
            raise Qwen35ConfigError(
                f"attention implementation {config._attn_implementation!r} — set it to"
                f" {ATTENTION_NAME!r} after register_attention()"
            )
        self.config = config
        self.embed_tokens = model.embed_tokens
        self.layers = model.layers[: config.num_hidden_layers]
        self.norm = model.norm
        self.layer_types = tuple(config.layer_types[: config.num_hidden_layers])
        #: Tensor keys whose stored value is `1 + w` (rewrite 4) — filled by
        #: {@link prepare_text_tower}.
        self.unit_offset_keys: tuple[str, ...] = ()

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.embed_tokens(input_ids)
        mask = causal_mask(positions)
        rope = (rope_cos, rope_sin)
        for layer, kind in zip(self.layers, self.layer_types, strict=True):
            hidden = layer(
                hidden,
                position_embeddings=rope,
                attention_mask=mask if kind == FULL_ATTENTION else None,
            )
        return self.norm(hidden)


def prepare_text_tower(model: nn.Module) -> Qwen35TextTower:
    """Wrap an eval f32 `Qwen3_5TextModel` (real or meta weights) and apply the rewrites.

    The upstream model is changed in place (its GDN attributes and norm submodules); the returned
    wrapper shares its modules.
    """
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet

    assert_torch_paths()
    tower = Qwen35TextTower(model)
    mixers = [
        module for module in tower.layers.modules() if isinstance(module, Qwen3_5GatedDeltaNet)
    ]
    expected = tower.layer_types.count(LINEAR_ATTENTION)
    if len(mixers) != expected:
        raise Qwen35ConfigError(f"{len(mixers)} GDN mixers for {expected} linear-attention layers")
    for mixer in mixers:
        mixer.chunk_gated_delta_rule = gated_delta_rule
        mixer.recurrent_gated_delta_rule = recurrent_gated_delta_rule
        mixer.causal_conv1d_fn = causal_conv1d
    tower.unit_offset_keys = _replace_unit_offset_norms(tower)
    full = tower.layer_types.count(FULL_ATTENTION)
    if len(tower.unit_offset_keys) != 2 * len(tower.layer_types) + 2 * full + 1:
        raise Qwen35ConfigError(
            f"{len(tower.unit_offset_keys)} zero-centred norms (expected 2 per layer, q/k per"
            " full-attention layer and the final norm)"
        )
    return tower


def dynamic_shapes() -> tuple[dict[int, Any], ...]:
    """One symbol for every input: the token axis of `input_ids`, `positions` and both RoPE rows
    (separate symbols would accept tables whose length differs from the tokens' — a silent wrong
    value)."""
    length = Dim(SYMBOL, min=MIN_TOKENS, max=MAX_TOKENS)
    return ({1: length}, {0: length}, {1: length}, {1: length})


def rope_rows(positions: Sequence[int], config: Any) -> tuple[torch.Tensor, torch.Tensor]:
    """`(rope_cos, rope_sin)` `[1, T, rotary_dim]` f32 for the given positions — the host formula.

    `inv_freq[i] = theta^(−2i/rotary_dim)` (i < rotary_dim/2), `angle = position·inv_freq`, row =
    `cat(angle, angle)`, computed in f64 and stored as f32. Text-only mRoPE is this 1-D form
    exactly (module docstring). Not bit-identical to upstream's rotary module, which computes
    every step in f32.
    """
    check_supported(config)
    theta = float(config.rope_parameters["rope_theta"])
    width = rotary_dim(config)
    exponents = torch.arange(0, width, 2, dtype=torch.float64) / width
    inverse = 1.0 / torch.pow(torch.tensor(theta, dtype=torch.float64), exponents)
    angles = torch.tensor(list(positions), dtype=torch.float64)[:, None] * inverse[None, :]
    rows = torch.cat((angles, angles), dim=-1)
    return (
        torch.cos(rows).to(torch.float32).unsqueeze(0),
        torch.sin(rows).to(torch.float32).unsqueeze(0),
    )


def rope_bound(positions: Sequence[int]) -> torch.Tensor:
    """ADR 0091 decision 2's bound on |host row − upstream row| per position: `2.5e-7 +
    position·1.2e-7` (the upper bound of an f32 angle's error)."""
    return 2.5e-7 + torch.tensor(list(positions), dtype=torch.float64) * 1.2e-7


def example_inputs(length: int, config: Any) -> tuple[torch.Tensor, ...]:
    """The graph inputs for a length-`length` prompt of token ids `ids` (positions 0 … T−1)."""
    if not MIN_TOKENS <= length <= MAX_TOKENS:
        raise ValueError(f"length {length} outside {MIN_TOKENS}..{MAX_TOKENS}")
    ids = (torch.arange(length, dtype=torch.long) * 7919 % config.vocab_size).unsqueeze(0)
    return inputs_for(ids, config)


def inputs_for(input_ids: torch.Tensor, config: Any) -> tuple[torch.Tensor, ...]:
    """The four graph inputs for token ids `[1, T]` (positions 0 … T−1, host RoPE rows)."""
    if input_ids.dim() != 2 or input_ids.shape[0] != 1:
        raise ValueError(f"input_ids {tuple(input_ids.shape)} — [1, T] only")
    length = int(input_ids.shape[1])
    positions = range(length)
    cos, sin = rope_rows(positions, config)
    return (
        input_ids.to(torch.long),
        torch.arange(length, dtype=torch.float32),
        cos,
        sin,
    )
