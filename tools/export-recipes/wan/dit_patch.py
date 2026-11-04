"""The patch layer that turns the Wan2.1 DiT (diffusers `WanTransformer3DModel`) into S-shaped IR
(ADR 0118).

The graph is **S-shaped** like Anima's (ADR 0034): the entry is after the patchify, the exit before
the unpatchify, and the only dimension symbol is the token length S. The following steps of the
upstream forward move to the host.

- **patchify / patch embedding**: the host turns the latent `[1,C,F,H,W]` into
  `tokens [1,S,C·pt·ph·pw]` (the last dimension ordered `(c, pt, ph, pw)`) ({@link dit_patchify}).
  The windows of `Conv3d(k = s = patch)` do not overlap, so it computes the same as a `Linear` with
  flattened weights ({@link patch_embedding_linear}).
- **unpatchify**: the output is `[1,S,pt·ph·pw·C]` right after `proj_out`. Its order is
  `(pt, ph, pw, c)`, **unlike** patchify ({@link dit_unpatchify} — the upstream rank-8 permute
  stays out of the graph).
- **The sinusoidal of the time embedding**: the host turns it into the graph input
  `timesteps_proj [1,256]` ({@link dit_timesteps_proj}). The MLP (256→1536→1536→9216) is inside the
  graph.
- **The 3D RoPE tables**: the host reorders the per-axis base tables (the asset `rope_base` —
  {@link dit_rope_base_tables}) into the graph inputs `rope_cos` / `rope_sin [1,S,1,head_dim]`
  ({@link dit_rope_tables}).

Only one rewrite stays inside the graph: the interleaved RoPE is applied as "swap adjacent pairs +
elementwise product with cos / sin" at rank 4 ({@link real_pair_rotary} — the same form as irodori's
real-valued version). Upstream builds rank 5 with `unflatten(-1, (-1, 2)).unbind(-1)` and writes it
back with the strided assignment `out[..., 0::2] = …`.

adaLN, qk-norm (`nn.RMSNorm` over all 1536 dimensions) and the text context (`[1,512,4096]`
projected on every row without a mask, cross-attn also without a mask) pass through **as upstream
has them** (the table of ADR 0118 decision 3).

MUST: the patched path is **eager-equivalent** to upstream. The RoPE rewrite is bit-identical (`a −
b = a + (−b)` and commutativity of addition — {@link real_pair_rotary}). The Linear form of the
patch embedding can differ from conv3d in reduction order, so the difference is recorded (measured
2026-10-02: 0 in all 8 cases on torch 2.13 CPU — `wan/tests/test_dit_patch.py` compares it with the
bound of the reduction-order difference on the real weights).

MUST: do not change the plain upstream path. attn1's processor is replaced, but the replacement
takes the real-valued form only when called with {@link RealPairRope}; when the upstream forward
(a plain tuple of tables) arrives it delegates to the upstream implementation as is. So the
reference is the upstream value whether it is taken before or after the replacement (no ordering
check like "take the reference before the replacement" of Anima's VAE patch is needed).

MUST: import diffusers inside functions (the `wan` group is not in the default sync —
`tests/test_optional_group_imports.py`).

## Wan2.2 TI2V (ADR 0121 decisions 3 / 4)

The 5B of the same class is written out with {@link WanDitTokensTi2v} (a single I2V-capable graph —
2 time inputs, a condition mask, a `where` on the modulation). The forward of 2.1's
{@link WanDitTokens} is not changed (the 1.3B containers and goldens stay byte-identical — decision
10). There are 2 upstream counterparts to compare with, one per time representation: T2V is the
upstream forward with a 1-D timestep ({@link reference_dit}); I2V is a reference wrapper that runs
the time MLP per value with M = 1 and hands per-token tables to the upstream blocks (the
`condition` of {@link reference_dit} — {@link _TokenwiseTimeEmbedder}).
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from functools import cache
from typing import Any, NamedTuple

import torch
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_leaves

#: The tensor keys of the host's per-axis base tables (block order `t → h → w`). The same spelling
#: and meaning as Anima's asset `rope_base` (cos / sin per axis × position × frequency; the values
#: are the upstream tables themselves).
ROPE_BASE_KEYS = ("cos_t", "sin_t", "cos_h", "sin_h", "cos_w", "sin_w")


class RealPairRope(NamedTuple):
    """The real-valued RoPE tables (graph inputs built by the host).

    `cos` / `sin` are `[1,S,1,head_dim]`, with the same value in the 2 components of an adjacent
    pair (exactly the `repeat_interleave` form of upstream `WanRotaryPosEmbed`). They are wrapped in
    a type so that the replaced processor can tell them from "the plain upstream tuple"
    ({@link WanRealPairAttnProcessor}).
    """

    cos: torch.Tensor
    sin: torch.Tensor


def real_pair_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply the interleaved RoPE at rank 4 (`x [B,S,H,D]`, tables `[1,S,1,D]`).

    Upstream (`apply_rotary_emb` in `WanAttnProcessor.__call__`) writes, for `x1 = x[..., 0::2]` and
    `x2 = x[..., 1::2]`, `out[0::2] = x1·cos − x2·sin` and `out[1::2] = x1·sin + x2·cos` (the tables
    are the cos of the even columns and the sin of the odd ones — the same value within an adjacent
    pair). Here, for the same `x`,

    - `x · cos` is `(x1·c, x2·c)`
    - `(−x2, x1)`, the adjacent pairs swapped with the first half negated, times `sin` is
      `(−x2·s, x1·s)`

    and the sum is `(x1·c − x2·s, x2·c + x1·s)`. The f32 rounding is the same too: multiplication is
    correctly rounded, so `(−x2)·s == −(x2·s)`; `a + (−b)` rounds the same as `a − b`; and
    `x2·c + x1·s` has the same value as `x1·s + x2·c` by commutativity of addition (the claim is
    **bit identity** — `wan/tests/test_dit_patch.py`).

    MUST: the intermediate is not rank 5 — that hits the rank limit of the strided kernels
    (`ops.STRIDED_RANK` = 4). The swap is done on `[B,S,H·D/2,2]` with the head axis collapsed, then
    the original shape is restored (the element order does not change).
    MUST: an axis of length 2 is taken with a length-1 `slice` (`select` is not in the IR vocabulary
    — as in irodori).
    MUST: pass `x` and the tables in the same dtype. Upstream casts the result back to `x`'s dtype
    with `.type_as(x)`, but that cast is not copied here (it is the identity in an f32 export, and
    it would add a node to the graph). The claim above that "the only difference is how RoPE is
    applied" holds only when the dtypes are the same, so a mismatch fails loudly (the paths through
    the S-shaped wrapper are only the f32 export and eager, with x and the tables in f32 — the f64
    reference goes through the plain upstream path).
    """
    if cos.dtype != x.dtype or sin.dtype != x.dtype:
        raise NotImplementedError(
            f"the dtype of the RoPE tables (cos {cos.dtype}, sin {sin.dtype}) differs from x's"
            f" {x.dtype} — upstream's `.type_as(x)` is not copied, so the values would not match"
            " upstream"
        )
    pairs = x.reshape(x.shape[0], x.shape[1], -1, 2)
    swapped = torch.cat([-pairs[..., 1:2], pairs[..., 0:1]], dim=-1).reshape(x.shape)
    return x * cos + swapped * sin


def _processor_base() -> type:
    """The upstream `WanAttnProcessor` (diffusers is imported inside functions — the module
    docstring)."""
    from diffusers.models.transformers.transformer_wan import WanAttnProcessor

    return WanAttnProcessor


# Third-party code notice. `WanRealPairAttnProcessor.__call__` below is adapted from
# `WanAttnProcessor.__call__` in huggingface/diffusers
# (`src/diffusers/models/transformers/transformer_wan.py`, `diffusers==0.39.0`). It is verbatim
# except for how the rotary embedding is applied (real-pair form) and the image-conditioning branch,
# which is rejected instead of carried over. License: Apache License, Version 2.0
# (http://www.apache.org/licenses/LICENSE-2.0). Upstream copyright notice, copied verbatim from the
# header of that file:
#
#   Copyright 2025 The Wan Team and The HuggingFace Team. All rights reserved.
def _real_pair_call(
    self: Any,
    attn: Any,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
    rotary_emb: Any = None,
) -> torch.Tensor:
    """An equivalent implementation of `WanAttnProcessor.__call__` (applies RoPE with
    {@link real_pair_rotary}).

    Called with tables other than {@link RealPairRope} (the plain upstream tuple, None), it
    **delegates to the upstream implementation as is**. The real-valued path is only for
    self-attention (`encoder_hidden_states is None`), and its only difference from upstream is how
    RoPE is applied (the q / k / v projections, qk-norm, head split, the attention call and the
    output projection are verbatim).
    """
    base = _processor_base()
    if not isinstance(rotary_emb, RealPairRope):
        return base.__call__(
            self, attn, hidden_states, encoder_hidden_states, attention_mask, rotary_emb
        )
    if encoder_hidden_states is not None or attn.add_k_proj is not None:
        raise NotImplementedError(
            "real-valued RoPE is for self-attention only (without the image condition's add_k_proj)"
            " — attn1 of Wan2.1 T2V"
        )
    from diffusers.models.attention_dispatch import dispatch_attention_fn
    from diffusers.models.transformers.transformer_wan import _get_qkv_projections

    query, key, value = _get_qkv_projections(attn, hidden_states, encoder_hidden_states)

    query = attn.norm_q(query)
    key = attn.norm_k(key)

    query = query.unflatten(2, (attn.heads, -1))
    key = key.unflatten(2, (attn.heads, -1))
    value = value.unflatten(2, (attn.heads, -1))

    query = real_pair_rotary(query, rotary_emb.cos, rotary_emb.sin)
    key = real_pair_rotary(key, rotary_emb.cos, rotary_emb.sin)

    hidden_states = dispatch_attention_fn(
        query,
        key,
        value,
        attn_mask=attention_mask,
        dropout_p=0.0,
        is_causal=False,
        backend=self._attention_backend,
        parallel_config=self._parallel_config,
    )
    hidden_states = hidden_states.flatten(2, 3)
    hidden_states = hidden_states.type_as(query)

    hidden_states = attn.to_out[0](hidden_states)
    hidden_states = attn.to_out[1](hidden_states)
    return hidden_states


@cache
def _real_pair_processor_class() -> type:
    """A subclass of `WanAttnProcessor` (with only `__call__` replaced by {@link _real_pair_call}).

    It is a subclass so that it inherits the upstream class attributes (`_attention_backend` /
    `_parallel_config`) as they are. The class is created after diffusers is imported (the MUST of
    the module docstring). Only one is created per process (so that the idempotence check of the
    replacement reduces to class identity).
    """
    base = _processor_base()
    return type("WanRealPairAttnProcessor", (base,), {"__call__": _real_pair_call})


def install_real_pair_processors(blocks: Iterable[nn.Module]) -> None:
    """Replace the processor of each block's attn1 (self-attention) with the real-valued RoPE
    version (idempotent).

    The replacement is per instance (class attributes are not touched). After the replacement the
    plain upstream forward still returns the upstream values (the delegation of
    {@link _real_pair_call}).
    """
    processor_class = _real_pair_processor_class()
    for block in blocks:
        if type(block.attn1.processor) is not processor_class:
            block.attn1.set_processor(processor_class())


def patch_embedding_linear(conv: nn.Conv3d) -> nn.Linear:
    """A `Linear(C·pt·ph·pw → Cout)` that computes the same as a non-overlapping `Conv3d(k = s =
    patch)`.

    The weight is a **view** (sharing storage) of the conv weight `[Cout, C, pt, ph, pw]` flattened
    to `[Cout, C·pt·ph·pw]`, and the bias is the same Parameter. They are shared so that when the
    f16 rounding (fake-quant — ADR 0006) is applied to the S-shaped wrapper, the same rounding
    reaches the upstream conv the reference is taken from (with a separate copy, only the reference
    would compute with the original weights, and quantization error would mix into the difference).

    The input order `(c, pt, ph, pw)` is exactly the C order of the flattening
    ({@link dit_patchify}).
    MUST: anything other than k = s, padding 0, dilation 1, groups 1 fails loudly (a conv whose
    windows overlap or leave gaps cannot be written as reshape + Linear).
    """
    kernel = tuple(int(size) for size in conv.kernel_size)
    if (
        tuple(int(size) for size in conv.stride) != kernel
        or any(int(size) != 0 for size in conv.padding)
        or any(int(size) != 1 for size in conv.dilation)
        or conv.groups != 1
        or conv.padding_mode != "zeros"
    ):
        raise NotImplementedError(
            f"the patch embedding's Conv3d is outside k = s, padding 0, dilation 1, groups 1"
            f" (k={kernel} s={tuple(conv.stride)} p={conv.padding} d={conv.dilation}"
            f" g={conv.groups}) — it would not compute the same as a Linear"
        )
    weight = conv.weight
    if not weight.is_contiguous():
        raise NotImplementedError(
            "the patch embedding's weight is not contiguous (no flattened view can be made)"
        )
    out_channels = int(weight.shape[0])
    linear = nn.Linear(int(weight[0].numel()), out_channels, bias=conv.bias is not None)
    linear.weight = nn.Parameter(weight.view(out_channels, -1), requires_grad=False)
    if conv.bias is not None:
        linear.bias = conv.bias
    return linear


def _assert_supported_config(model: nn.Module) -> None:
    """Check the configuration the S-shaped wrapper assumes (Wan2.1 T2V). Fail loudly otherwise."""
    config = model.config
    if config.image_dim is not None or config.added_kv_proj_dim is not None:
        raise NotImplementedError(
            "image-conditioned (I2V) configurations are not supported — Wan2.1 T2V only"
        )
    if config.pos_embed_seq_len is not None:
        raise NotImplementedError("configurations with pos_embed_seq_len are not supported")
    if config.qk_norm != "rms_norm_across_heads":
        raise NotImplementedError(
            f"qk_norm={config.qk_norm} is not supported (only rms_norm_across_heads)"
        )


# Third-party code notice. `WanDitTokens.forward` below is adapted from
# `WanTransformer3DModel.forward` and `WanTimeTextImageEmbedding.forward` in huggingface/diffusers
# (`src/diffusers/models/transformers/transformer_wan.py`, `diffusers==0.39.0`). The block loop, the
# condition embedding (minus the sinusoidal step) and the output norm / projection are verbatim; the
# patchify, the sinusoidal timestep projection, the RoPE table construction and the unpatchify were
# moved to the host. License: Apache License, Version 2.0
# (http://www.apache.org/licenses/LICENSE-2.0). Upstream copyright notice, copied verbatim from the
# header of that file:
#
#   Copyright 2025 The Wan Team and The HuggingFace Team. All rights reserved.
class WanDitTokens(nn.Module):
    """The **token form** of one forward of `WanTransformer3DModel` (S-shaped graph — ADR 0118
    decision 3).

    It differs from the upstream forward only at the entry and the exit; the body of 30 blocks calls
    the upstream modules as they are:

    - The input is **`tokens [1,S,C·pt·ph·pw]`** rather than `hidden_states [1,C,F,H,W]` (patchified
      by the host — {@link dit_patchify}). The patch embedding is the flattened Linear
      ({@link patch_embedding_linear})
    - The time is **`timesteps_proj [1,freq_dim]`** rather than the integer `timestep` (the host's
      sinusoidal — {@link dit_timesteps_proj})
    - The RoPE cos / sin tables become **graph inputs** (`[1,S,1,head_dim]` — built by the host from
      the base tables). attn1's processor is the real-valued version
      ({@link install_real_pair_processors})
    - The output is the tokens `[1,S,pt·ph·pw·C]` **before** the unpatchify (the inverse reordering
      is on the host — {@link dit_unpatchify})

    As a result, not one of F / H / W appears in the graph, and it can be written with the single
    token-length symbol `S`.

    `encoder_hidden_states` is `[1,512,4096]`: umT5's output cut at the valid length, then padded
    with zeros to 512 rows (on the host — `wan.pipeline_ref.pad_text_embeds`). `text_embedder` is
    applied to all 512 rows, and cross-attn has no mask (as upstream — the padded positions become a
    non-zero constant vector after the projection).

    It holds only references to the upstream components (the `patch_embedding` Linear is a view of
    the conv weights). It does not hold `rope` — the tables are graph inputs, so the buffers of
    `rope` (f32 tables lowered from float64) are not carried as initial values.

    Passing `blocks` runs only that subsequence (for truncation to measure how the error grows with
    the number of layers; the default is all layers).

    `install_processors=False` leaves the replacement of attn1's processors to the caller (the
    layer-by-layer CPU reference `wan.dit_reference.LayerwiseDit` — iterating the list reads the
    weights, so it is not iterated at construction; {@link install_real_pair_processors} is applied
    to each block as it is opened).
    """

    def __init__(
        self,
        model: nn.Module,
        blocks: nn.ModuleList | None = None,
        *,
        install_processors: bool = True,
    ) -> None:
        super().__init__()
        _assert_supported_config(model)
        self.patch_size = tuple(int(size) for size in model.config.patch_size)
        self.patch_embedding = patch_embedding_linear(model.patch_embedding)
        self.condition_embedder = model.condition_embedder
        self.blocks = model.blocks if blocks is None else blocks
        self.norm_out = model.norm_out
        self.proj_out = model.proj_out
        self.scale_shift_table = model.scale_shift_table
        if install_processors:
            install_real_pair_processors(self.blocks)

    def forward(
        self,
        tokens: torch.Tensor,
        timesteps_proj: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
    ) -> torch.Tensor:
        return self.forward_hidden(
            self.patch_embedding(tokens),
            timesteps_proj,
            encoder_hidden_states,
            rope_cos,
            rope_sin,
        )

    def forward_hidden(
        self,
        hidden_states: torch.Tensor,
        timesteps_proj: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
    ) -> torch.Tensor:
        """From **after** the patch embedding (`hidden_states [1,S,dim]`).

        It differs from {@link forward} only by the patch-embedding step. Fed the output of the
        upstream conv3d, it is **bit-identical** to the upstream forward (the measurement point
        showing that every rewrite other than the patch embedding is bit-identical —
        `wan/tests/test_dit_patch.py`).
        """
        output, _ = self._trunk(
            hidden_states, timesteps_proj, encoder_hidden_states, rope_cos, rope_sin, collect=False
        )
        return output

    def _trunk(
        self,
        hidden_states: torch.Tensor,
        timesteps_proj: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        *,
        collect: bool,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """The body (condition embedding → block list → output norm / projection).

        With `collect`, it also returns the output of each block.
        """
        embedder = self.condition_embedder
        # Upstream `WanTimeTextImageEmbedding.forward` (only the sinusoidal step moved to the host).
        time_embedder_dtype = next(iter(embedder.time_embedder.parameters())).dtype
        if timesteps_proj.dtype != time_embedder_dtype and time_embedder_dtype != torch.int8:
            timesteps_proj = timesteps_proj.to(time_embedder_dtype)
        temb = embedder.time_embedder(timesteps_proj).type_as(encoder_hidden_states)
        timestep_proj = embedder.time_proj(embedder.act_fn(temb))
        encoder_hidden_states = embedder.text_embedder(encoder_hidden_states)

        # batch_size, 6, inner_dim
        timestep_proj = timestep_proj.unflatten(1, (6, -1))

        rotary_emb = RealPairRope(rope_cos, rope_sin)
        collected: list[torch.Tensor] = []
        for block in self.blocks:
            hidden_states = block(hidden_states, encoder_hidden_states, timestep_proj, rotary_emb)
            if collect:
                collected.append(hidden_states)

        # batch_size, inner_dim
        shift, scale = (self.scale_shift_table + temb.unsqueeze(1)).chunk(2, dim=1)
        hidden_states = (self.norm_out(hidden_states.float()) * (1 + scale) + shift).type_as(
            hidden_states
        )
        return self.proj_out(hidden_states), collected


class WanDitTokensLayers(WanDitTokens):
    """A measurement form of {@link WanDitTokens} **with a per-layer exit added** (the growth of the
    error with the number of layers — decision 8).

    The outputs are the output of each block `[1,S,dim]` (in block order) and, last, the same final
    output `[1,S,pt·ph·pw·C]` as {@link WanDitTokens}. It is not a product graph (not shipped) — the
    inputs and weights are the same, so it runs on the same golden inputs and is compared layer by
    layer with the output of each upstream block (values taken with a forward hook).
    """

    def forward(  # type: ignore[override]
        self,
        tokens: torch.Tensor,
        timesteps_proj: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        output, collected = self._trunk(
            self.patch_embedding(tokens),
            timesteps_proj,
            encoder_hidden_states,
            rope_cos,
            rope_sin,
            collect=True,
        )
        return (*collected, output)


def _time_embedding(
    embedder: nn.Module, timesteps_proj: torch.Tensor, like: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the time MLP (`time_embedder` → SiLU → `time_proj`) on one row (`temb [1,dim]` and
    `timestep_proj [1,6·dim]`).

    The time step of upstream `WanTimeTextImageEmbedding.forward`, verbatim (the sinusoidal is on
    the host). It is the same formula as the same 5 lines of {@link WanDitTokens._trunk}; that one
    does not go through this function so that 2.1's wrapper graph does not move by a single byte
    (ADR 0121 decision 10 — a change to the recipe shared with 2.1 is confirmed by the 1.3B
    containers staying byte-identical).
    """
    time_embedder_dtype = next(iter(embedder.time_embedder.parameters())).dtype
    if timesteps_proj.dtype != time_embedder_dtype and time_embedder_dtype != torch.int8:
        timesteps_proj = timesteps_proj.to(time_embedder_dtype)
    temb = embedder.time_embedder(timesteps_proj).type_as(like)
    return temb, embedder.time_proj(embedder.act_fn(temb))


def _two_time_embeddings(
    embedder: nn.Module, generation: torch.Tensor, condition: torch.Tensor, like: torch.Tensor
) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
    """Run the time MLPs of the generated side and the conditioned side **twice with M = 1** (ADR
    0121 decision 3 — returns the `(temb, timestep_proj)` pairs, generated side first, then
    conditioned side).

    MUST: do not concatenate the 2 rows into one run with M = 2. torch CPU's linear differs in the
    last bit between M = 1 and M ≥ 2 (difference investigation §6.1) — combined, the generated-side
    row of T2V would no longer be computed in the same form as 2.1. In the IR this adds 3 linear
    nodes with the same weights (`wan.ti2v_export_dit.inspect_structure` pins the count).
    """
    return (
        _time_embedding(embedder, generation, like),
        _time_embedding(embedder, condition, like),
    )


# Third-party code notice. `_ti2v_block` below is adapted from `WanTransformerBlock.forward` in
# huggingface/diffusers (`src/diffusers/models/transformers/transformer_wan.py`,
# `diffusers==0.39.0`). The self-attention, cross-attention and feed-forward lines are verbatim;
# the modulation is computed per time value (generation / condition) and selected per token with
# `torch.where` right before each use instead of being read from a per-token table. License: Apache
# License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0). Upstream copyright notice,
# copied verbatim from the header of that file:
#
#   Copyright 2025 The Wan Team and The HuggingFace Team. All rights reserved.
def _ti2v_block(
    block: nn.Module,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    generation: torch.Tensor,
    condition: torch.Tensor,
    condition_mask: torch.Tensor,
    rotary_emb: RealPairRope,
) -> torch.Tensor:
    """Run one upstream block with the modulation **selected per token** from 2 time values
    (ADR 0121 decision 3).

    `generation` / `condition` are the outputs `[1,6,dim]` of the time MLP (the generated side's t
    and the conditioned side's t = 0). The 6 components are built per time value as `[1,1,dim]` (the
    same sum `scale_shift_table + timestep_proj` as the upstream branch for a 1-D timestep) and
    selected into `[1,S,dim]` with `where(condition_mask, conditioned side, generated side)`.

    MUST: write each component's `where` **right before** its use (the modulation before
    self-attention, the gate of its residual, the modulation before the FFN, the FFN's gate).
    Following upstream's order (building the 6 components together at the top of the block),
    torch.export keeps that order, so up to 6 `[1,S,dim]` tensors stay alive across the attention
    (one is 0.31 GiB in f32 at S = 27,280 — "where to place the where" in decision 3). Python's
    evaluation order (`a * (1 + where(…)) + where(…)` is left to right) becomes the IR node order as
    is (`wan.ti2v_export_dit.where_placement` pins it on the IR).

    Value equivalence: `where` is a selection, so it does not change values. In T2V (the mask all
    false) it is elementwise the same computation as the upstream branch for a 1-D timestep (a
    broadcast of `[1,1,dim]`), and in I2V as upstream's `temb.ndim == 4` branch (per-token tables —
    the reference wrapper of decision 4) (the claim is **bit identity** — `TestTheTi2vGraphForm` in
    `wan/tests/test_dit_patch_ti2v.py`).
    """
    table = block.scale_shift_table
    shift_g, scale_g, gate_g, c_shift_g, c_scale_g, c_gate_g = (table + generation.float()).chunk(
        6, dim=1
    )
    shift_c, scale_c, gate_c, c_shift_c, c_scale_c, c_gate_c = (table + condition.float()).chunk(
        6, dim=1
    )

    # 1. Self-attention
    norm_hidden_states = (
        block.norm1(hidden_states.float()) * (1 + torch.where(condition_mask, scale_c, scale_g))
        + torch.where(condition_mask, shift_c, shift_g)
    ).type_as(hidden_states)
    attn_output = block.attn1(norm_hidden_states, None, None, rotary_emb)
    hidden_states = (
        hidden_states.float() + attn_output * torch.where(condition_mask, gate_c, gate_g)
    ).type_as(hidden_states)

    # 2. Cross-attention
    norm_hidden_states = block.norm2(hidden_states.float()).type_as(hidden_states)
    attn_output = block.attn2(norm_hidden_states, encoder_hidden_states, None, None)
    hidden_states = hidden_states + attn_output

    # 3. Feed-forward
    norm_hidden_states = (
        block.norm3(hidden_states.float()) * (1 + torch.where(condition_mask, c_scale_c, c_scale_g))
        + torch.where(condition_mask, c_shift_c, c_shift_g)
    ).type_as(hidden_states)
    ff_output = block.ffn(norm_hidden_states)
    hidden_states = (
        hidden_states.float() + ff_output.float() * torch.where(condition_mask, c_gate_c, c_gate_g)
    ).type_as(hidden_states)

    return hidden_states


# Third-party code notice. `WanDitTokensTi2v.forward_hidden` below is adapted from
# `WanTransformer3DModel.forward` and `WanTimeTextImageEmbedding.forward` in huggingface/diffusers
# (`src/diffusers/models/transformers/transformer_wan.py`, `diffusers==0.39.0`). The text embedding
# and the output norm / projection are verbatim; the time MLP runs once per time value and the
# per-token modulation is selected with `torch.where` (see `_ti2v_block`). License: Apache License,
# Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0). Upstream copyright notice, copied
# verbatim from the header of that file:
#
#   Copyright 2025 The Wan Team and The HuggingFace Team. All rights reserved.
class WanDitTokensTi2v(WanDitTokens):
    """The I2V-capable S-shaped graph of Wan2.2 TI2V (ADR 0121 decision 3 — T2V and I2V in one
    graph).

    It adds 2 inputs after the 5 inputs of {@link WanDitTokens} (the positions of the first 5 are
    the same as 2.1's):

    - `timesteps_proj_condition [1,freq_dim]`: the time sinusoidal of the conditioned side (t = 0
      for I2V, the same value as the generated side for T2V — host {@link dit_timesteps_proj})
    - `condition_mask [1,S,1]` (bool): whether a token belongs to the conditioning frame (the first
      latent frame) (for I2V the first P = H'·W' tokens are true, for T2V all false — host
      {@link dit_condition_mask})

    The time MLP is run **twice with M = 1** (3 more linear nodes with the same weights — the 310
    dispatches of decision 2). It is not one run with M = 2 because torch CPU's linear differs in
    the last bit between M = 1 and M ≥ 2 (keeping the generated-side row of T2V computed in the same
    form as 2.1 — the "why" of decision 3). The per-block modulation is selected by
    {@link _ti2v_block}, the output modulation here, each with a `where` right before its use.

    MUST: do not use `WanDitTokens.forward` (2.1's 5 inputs) — this wrapper has 7 inputs.
    """

    def forward(  # type: ignore[override]
        self,
        tokens: torch.Tensor,
        timesteps_proj: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        timesteps_proj_condition: torch.Tensor,
        condition_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.forward_hidden(
            self.patch_embedding(tokens),
            timesteps_proj,
            encoder_hidden_states,
            rope_cos,
            rope_sin,
            timesteps_proj_condition,
            condition_mask,
        )

    def forward_hidden(  # type: ignore[override]
        self,
        hidden_states: torch.Tensor,
        timesteps_proj: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        timesteps_proj_condition: torch.Tensor,
        condition_mask: torch.Tensor,
    ) -> torch.Tensor:
        """From **after** the patch embedding (the TI2V version of
        {@link WanDitTokens.forward_hidden})."""
        embedder = self.condition_embedder
        (temb_g, proj_g), (temb_c, proj_c) = _two_time_embeddings(
            embedder, timesteps_proj, timesteps_proj_condition, encoder_hidden_states
        )
        encoder_hidden_states = embedder.text_embedder(encoder_hidden_states)

        # batch_size, 6, inner_dim
        proj_g = proj_g.unflatten(1, (6, -1))
        proj_c = proj_c.unflatten(1, (6, -1))

        rotary_emb = RealPairRope(rope_cos, rope_sin)
        for block in self.blocks:
            hidden_states = _ti2v_block(
                block,
                hidden_states,
                encoder_hidden_states,
                proj_g,
                proj_c,
                condition_mask,
                rotary_emb,
            )

        # batch_size, inner_dim
        shift_g, scale_g = (self.scale_shift_table + temb_g.unsqueeze(1)).chunk(2, dim=1)
        shift_c, scale_c = (self.scale_shift_table + temb_c.unsqueeze(1)).chunk(2, dim=1)
        hidden_states = (
            self.norm_out(hidden_states.float())
            * (1 + torch.where(condition_mask, scale_c, scale_g))
            + torch.where(condition_mask, shift_c, shift_g)
        ).type_as(hidden_states)
        return self.proj_out(hidden_states)


# ---- The Python-side source of truth for the steps moved to the host (what the TS mirrors are
# checked against) ----


def dit_patchify(latents: torch.Tensor, patch_size: tuple[int, int, int]) -> torch.Tensor:
    """`latents [1,C,F,H,W]` → `tokens [1,S,C·pt·ph·pw]` (last dimension `(c, pt, ph, pw)`).

    The token index is `(f·H' + h)·W' + w` (the order of upstream `conv3d(…).flatten(2).transpose(1,
    2)`), and the last dimension is in the same C order as the flattening of the conv3d weight
    `[Cout, C, pt, ph, pw]` (the counterpart of {@link patch_embedding_linear}).
    """
    patch_t, patch_h, patch_w = patch_size
    batch, channels, frames, height, width = latents.shape
    patches = latents.reshape(
        batch,
        channels,
        frames // patch_t,
        patch_t,
        height // patch_h,
        patch_h,
        width // patch_w,
        patch_w,
    )
    return patches.permute(0, 2, 4, 6, 1, 3, 5, 7).flatten(4, 7).flatten(1, 3)


# Third-party code notice. The body of `dit_unpatchify` below reproduces the unpatchify lines of
# `WanTransformer3DModel.forward` in huggingface/diffusers (`transformer_wan.py`,
# `diffusers==0.39.0`) verbatim. License: Apache License, Version 2.0. Upstream copyright notice,
# copied verbatim from the header of that file:
#
#   Copyright 2025 The Wan Team and The HuggingFace Team. All rights reserved.
def dit_unpatchify(
    tokens: torch.Tensor,
    latent_shape: tuple[int, int, int],
    patch_size: tuple[int, int, int],
) -> torch.Tensor:
    """`tokens [1,S,pt·ph·pw·C]` → `latents [1,C,F,H,W]` (the upstream exit, verbatim).

    MUST: **not the reverse of patchify** — the last dimension is ordered `(pt, ph, pw, c)`,
    distinct from patchify's `(c, pt, ph, pw)` (upstream `reshape(…, p_t, p_h, p_w, -1)` →
    `permute(0, 7, 1, 4, 2, 5, 3, 6)`).
    """
    frames, height, width = latent_shape
    p_t, p_h, p_w = patch_size
    batch_size = tokens.shape[0]
    post_patch_num_frames = frames // p_t
    post_patch_height = height // p_h
    post_patch_width = width // p_w
    hidden_states = tokens.reshape(
        batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
    )
    hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
    return hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)


def dit_timesteps_proj(model: nn.Module, timestep: torch.Tensor) -> torch.Tensor:
    """The time sinusoidal computed on the host (exactly upstream
    `condition_embedder.timesteps_proj`).

    `timestep` is the integer timestep the upstream pipeline passes (`[1]`, int64). The formula is
    not copied; the upstream module is called (`Timesteps(freq_dim, flip_sin_to_cos=True,
    downscale_freq_shift=0)` — cos first, sin second).
    """
    return model.condition_embedder.timesteps_proj(timestep)


def dit_condition_mask(
    latent_shape: tuple[int, int, int], patch_size: tuple[int, int, int], *, conditioned: bool
) -> torch.Tensor:
    """The graph input `condition_mask [1,S,1]` (bool — ADR 0121 decision 3).

    For I2V (`conditioned`) the P = H'·W' tokens of the first latent frame are true; for T2V all are
    false. The token index is `(f·H' + h)·W' + w` ({@link dit_patchify}), so the tokens of the first
    latent frame come first, contiguously. Upstream diffusers' I2V (the `expand_timesteps` branch of
    `WanImageToVideoPipeline`) thins out a `[1,1,F,H,W]` mask with the first latent frame set to 0
    by `[:, ::2, ::2]`, flattens it and multiplies it into t — the same tokens get t = 0.

    MUST: only for a temporal patch of 1 (first latent frame = first token frame).
    """
    patch_t, patch_h, patch_w = patch_size
    if patch_t != 1:
        raise NotImplementedError(
            f"a condition mask with temporal patch {patch_t} is not supported (only 1)"
        )
    frames, height, width = latent_shape
    per_frame = (height // patch_h) * (width // patch_w)
    mask = torch.zeros(1, frames * per_frame, 1, dtype=torch.bool)
    if conditioned:
        mask[:, :per_frame] = True
    return mask


def dit_rope_tables(
    rope: nn.Module, latent_shape: tuple[int, int, int]
) -> tuple[torch.Tensor, torch.Tensor]:
    """The RoPE cos / sin tables `[1,S,1,head_dim]` passed to the S-shaped graph (the output of
    upstream `WanRotaryPosEmbed`).

    `rope` is upstream's `model.rope` and `latent_shape` the latent's `(F, H, W)`. Upstream's `rope`
    looks only at the input's shape, so a zero probe is passed.
    MUST: do not call it on a machine that sees MPS (the tables would be built in float32 and vary
    by machine — `pipeline_ref.assert_no_mps` stops it before building).
    """
    probe = torch.zeros(1, 1, *latent_shape)
    return rope(probe)


def dit_rope_base_tables(rope: nn.Module) -> dict[str, torch.Tensor]:
    """Cut **the per-axis base tables** out of the upstream tables (`model.rope`) for the host (TS)
    to build the RoPE tables from.

    It returns the 6 tables `cos_t` / `sin_t` / `cos_h` / …, each `[rows, block width of the axis /
    2]` (row = position, column = that axis's frequency). The values are the output of upstream
    `model.rope` itself; the upstream tables are `repeat_interleave`d with the same value in each
    adjacent pair, so taking only the even columns gives the base table ({@link _assert_rope_base}
    confirms that the odd columns equal the even ones).

    ## Why the tables are baked

    Upstream computes the angles and the trigonometric functions in float64 and then lowers them to
    f32 (on a machine without MPS — `transformer_wan.py:375`). Copying the formula into TS, the last
    bit can diverge through implementation differences of f64 `pow` / division / `cos`. With baked
    base tables the host only reorders, and the tables are bit-identical to upstream (the same
    judgement as Anima's `rope-base.ts` — the MUST that TS does not compute trigonometric
    functions).

    ## Row count

    Exactly upstream's ceiling on the position table, `rope_max_seq_len` (1024). A latent where any
    of F' / H' / W' exceeds it cannot be built upstream either.

    MUST: the cut is made **from the output** of `model.rope` (the formula is not copied). One probe
    per axis is run, moving only that axis over `rows` positions and fixing the other 2 axes at
    position 0.
    """
    rows = int(rope.max_seq_len)
    patch_t, patch_h, patch_w = (int(size) for size in rope.patch_size)
    dims = (int(rope.t_dim), int(rope.h_dim), int(rope.w_dim))
    starts = (0, dims[0], dims[0] + dims[1])

    def probe(frames: int, height: int, width: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Call upstream's rope as is with a shape that gives `(ppf, pph, ppw) = (frames, height,
        width)`."""
        sample = torch.zeros(1, 1, frames * patch_t, height * patch_h, width * patch_w)
        cos, sin = rope(sample)
        return cos[0, :, 0], sin[0, :, 0]

    tables: dict[str, torch.Tensor] = {}
    for axis, (frames, height, width), start, dim in zip(
        ("t", "h", "w"),
        ((rows, 1, 1), (1, rows, 1), (1, 1, rows)),
        starts,
        dims,
        strict=True,
    ):
        cos, sin = probe(frames, height, width)
        block_cos = cos[:, start : start + dim]
        block_sin = sin[:, start : start + dim]
        if not (
            torch.equal(block_cos[:, 0::2], block_cos[:, 1::2])
            and torch.equal(block_sin[:, 0::2], block_sin[:, 1::2])
        ):
            raise AssertionError(
                f"block {axis} of the upstream RoPE table does not have the same value within"
                " adjacent pairs (the repeat_interleave premise broke — the even columns cannot"
                " serve as the base table)"
            )
        tables[f"cos_{axis}"] = block_cos[:, 0::2].contiguous()
        tables[f"sin_{axis}"] = block_sin[:, 0::2].contiguous()
    _assert_rope_base(tables, rows, tuple(dim // 2 for dim in dims))
    return tables


def _assert_rope_base(tables: dict[str, torch.Tensor], rows: int, widths: tuple[int, ...]) -> None:
    """Confirm that the cut of the base tables matches the upstream block boundaries (the check
    against a vacuous pass).

    A cut off by one block still has matching shapes (the block widths of h and w are equal). So it
    checks that **only that axis moves with the position**: the row at position 0 has angle 0, so
    cos = 1 / sin = 0, and at position 1 sin is non-zero. A cut of another block stays constant over
    the position and breaks the latter.
    """
    if set(tables) != set(ROPE_BASE_KEYS):
        raise AssertionError(
            f"the rope base table keys are {sorted(tables)} (expected {sorted(ROPE_BASE_KEYS)})"
        )
    for axis, width in zip(("t", "h", "w"), widths, strict=True):
        for kind in ("cos", "sin"):
            table = tables[f"{kind}_{axis}"]
            if table.dtype != torch.float32 or tuple(table.shape) != (rows, width):
                raise AssertionError(
                    f"rope base table {kind}_{axis} is {table.dtype} {tuple(table.shape)}"
                    f" (expected float32 {(rows, width)})"
                )
        if not torch.equal(tables[f"cos_{axis}"][0], torch.ones(width)):
            raise AssertionError(
                f"position 0 of rope base table cos_{axis} is not all 1 (misaligned cut)"
            )
        if not torch.equal(tables[f"sin_{axis}"][0], torch.zeros(width)):
            raise AssertionError(
                f"position 0 of rope base table sin_{axis} is not all 0 (misaligned cut)"
            )
        if torch.equal(tables[f"sin_{axis}"][1], torch.zeros(width)):
            raise AssertionError(
                f"rope base table sin_{axis} is all 0 at position 1 too (it does not move on this"
                " axis = misaligned cut)"
            )


@contextmanager
def flash_attention_only() -> Iterator[None]:
    """Pin SDPA within this block to CPU's flash path
    (`aten._scaled_dot_product_flash_attention_for_cpu`) (the CPU reference and the patched eager
    — not applied to export tracing).

    WHY: if the full-size 81-frame (S = 32,760) reference falls to the MATH backend, one score
    matrix `[1,12,S,S]` allocates 51.5 GB (f32) / 103 GB (f64) and dies with OOM (taking the tens of
    minutes of computation so far with it). torch 2.13.0+cpu chooses flash by default
    (`torch._fused_sdp_choice` — `wan/tests/test_dit_patch.py` checks it), so pinning does not
    change the numbers. With the pin, inputs flash does not take (a non-contiguous last dimension,
    for example) stop with a RuntimeError instead of silently falling to MATH.
    """
    with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
        yield


class TimestepCondition(NamedTuple):
    """The conditioned side's time for I2V (the input of the reference wrapper of decision 4).

    `timestep` is the integer timestep of the tokens of the conditioning frame (`[1]`, int64 — 0 for
    I2V), and `mask` the per-token condition mask `[1,S]` (bool — the graph input
    `condition_mask [1,S,1]` with its last axis dropped, {@link dit_condition_mask}).
    """

    timestep: torch.Tensor
    mask: torch.Tensor


class _TokenwiseTimeEmbedder(nn.Module):
    """A replacement that calls upstream's `condition_embedder` **twice with M = 1**, once per time
    value, and builds the per-token `temb [1,S,dim]` / `timestep_proj [1,S,6·dim]` (the reference
    wrapper of ADR 0121 decision 4).

    Given a 2-D timestep `[1,S]`, the upstream forward passes the flattened `[S]` and
    `timestep_seq_len` here, splits the returned `timestep_proj` into `[1,S,6,dim]` and feeds it to
    the blocks' `temb.ndim == 4` branch and the head's `ndim == 3` branch. The blocks and the head
    run upstream's code as is — this changes only how the time MLP is run (M = S → M = 1 per value),
    and the semantics are the same as the graph's ({@link WanDitTokensTi2v}).

    MUST: fail loudly if the flat timestep passed in does not match `where(mask, conditioned side,
    generated side)` (if upstream's flattening order changed, tokens and modulation would silently
    go out of step).
    """

    def __init__(
        self, embedder: nn.Module, generation: torch.Tensor, condition: TimestepCondition
    ) -> None:
        super().__init__()
        self.embedder = embedder
        self.generation = generation
        self.condition = condition

    def forward(
        self,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: torch.Tensor | None = None,
        timestep_seq_len: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        mask = self.condition.mask
        expected = torch.where(mask, self.condition.timestep, self.generation).flatten()
        if timestep_seq_len != mask.shape[1] or not torch.equal(timestep, expected):
            raise AssertionError(
                "the per-token timestep upstream passed does not match the condition mask"
                f" (seq_len {timestep_seq_len}, mask {list(mask.shape)})"
            )
        # Both runs call upstream's condition_embedder as is (the text projection runs twice with
        # the same value — the first run's return is used).
        temb_g, proj_g, projected, image = self.embedder(
            self.generation, encoder_hidden_states, encoder_hidden_states_image
        )
        temb_c, proj_c, _, _ = self.embedder(
            self.condition.timestep, encoder_hidden_states, encoder_hidden_states_image
        )
        select = mask.unsqueeze(-1)
        temb = torch.where(select, temb_c.unsqueeze(1), temb_g.unsqueeze(1))
        proj = torch.where(select, proj_c.unsqueeze(1), proj_g.unsqueeze(1))
        return temb, proj, projected, image


def reference_dit(
    model: nn.Module,
    latents: torch.Tensor,
    timestep: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    condition: TimestepCondition | None = None,
) -> torch.Tensor:
    """The output `[1,C,F,H,W]` of **the plain upstream** `WanTransformer3DModel.forward` (the
    reference value).

    The attention runs under {@link flash_attention_only} (for both the f32 and the f64 reference).

    With `condition`, a 2-D timestep `[1,S]` is passed to the upstream forward, and
    `condition_embedder` is replaced by {@link _TokenwiseTimeEmbedder} for the duration of the call
    (the I2V reference wrapper of ADR 0121 decision 4 — the time MLP with M = 1 per value). Without
    it, the upstream forward itself (with a 1-D timestep the same path as 2.1; with a 2-D one
    diffusers' path with M = S).
    """
    with flash_attention_only():
        if condition is None:
            return model(
                hidden_states=latents,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
                return_dict=False,
            )[0]
        if timestep.shape != (1,) or condition.timestep.shape != (1,):
            raise ValueError(
                "the reference wrapper's timestep is [1] on both the generated and the conditioned"
                f" side ({list(timestep.shape)} / {list(condition.timestep.shape)})"
            )
        original = model.condition_embedder
        model.condition_embedder = _TokenwiseTimeEmbedder(original, timestep, condition)
        try:
            return model(
                hidden_states=latents,
                timestep=torch.where(condition.mask, condition.timestep, timestep),
                encoder_hidden_states=encoder_hidden_states,
                return_dict=False,
            )[0]
        finally:
            model.condition_embedder = original


def reference_dit_layers(
    model: nn.Module,
    latents: torch.Tensor,
    timestep: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    condition: TimestepCondition | None = None,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """The output of {@link reference_dit} and the output `[1,S,dim]` of each upstream block (taken
    with a forward hook).

    The hooks are attached only for the duration of the call (the S-shaped wrapper runs the same
    blocks, so hooks left attached would also catch its calls).
    """
    collected: list[torch.Tensor] = []

    def capture(_module: nn.Module, _inputs: Any, output: torch.Tensor) -> None:
        collected.append(output.detach().clone())

    handles = [block.register_forward_hook(capture) for block in model.blocks]
    try:
        output = reference_dit(model, latents, timestep, encoder_hidden_states, condition)
    finally:
        for handle in handles:
            handle.remove()
    if len(collected) != len(model.blocks):
        raise AssertionError(
            f"collected {len(collected)} block outputs (there should be {len(model.blocks)})"
        )
    return output, collected


def _timestep_sinusoid_shapes(freq_dim: int) -> frozenset[tuple[int, ...]]:
    """The shapes of the values that upstream's time sinusoid (`Timesteps` →
    `get_timestep_embedding`, batch 1) builds in f32.

    `[1, 1]` of `timesteps[:, None].float()`; `[half]` of the frequency `arange` / division / `exp`;
    `[1, half]` of the angles, sin / cos and the flip slices; `[1, freq_dim]` of the concatenation
    (`half = freq_dim // 2`).
    """
    half = freq_dim // 2
    return frozenset({(1, 1), (half,), (1, half), (1, freq_dim)})


class _NarrowFloatWatch(TorchDispatchMode):
    """Record non-f64 floating-point values (with shapes not in `allowed`) inside an f64 forward."""

    def __init__(self, allowed: frozenset[tuple[int, ...]]) -> None:
        super().__init__()
        self.allowed = allowed
        self.found: list[str] = []

    def __torch_dispatch__(
        self,
        func: Any,
        types: Any,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        result = func(*args, **(kwargs or {}))
        for leaf in tree_leaves(result):
            if (
                isinstance(leaf, torch.Tensor)
                and leaf.is_floating_point()
                and leaf.dtype != torch.float64
                and tuple(leaf.shape) not in self.allowed
            ):
                self.found.append(f"{func}: {leaf.dtype} {tuple(leaf.shape)}")
        return result


@contextmanager
def _float_keeps_float64() -> Iterator[None]:
    """Within this block, make `Tensor.float()` a pass-through for f64 tensors (others unchanged).

    WHY: upstream moves FP32LayerNorm and the blocks' modulation and residuals to f32 with
    `.float()` (so that norms are computed in f32 when running in bf16 / f16). In an f32 model all
    of these are identities and not part of the computation's meaning, but in an f64 model, passed
    through as is, only the norms and residuals would drop to f32 and it would not be "a reference
    with activations in f64".
    """
    original = vars(torch.Tensor).get("float")
    narrow = torch.Tensor.float

    def keep_float64(tensor: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        if tensor.dtype == torch.float64:
            return tensor
        return narrow(tensor, *args, **kwargs)

    torch.Tensor.float = keep_float64  # type: ignore[method-assign]
    try:
        yield
    finally:
        if original is None:
            del torch.Tensor.float
        else:
            torch.Tensor.float = original  # type: ignore[method-assign]


@contextmanager
def float64_forward(model: nn.Module) -> Iterator[None]:
    """Run the upstream forward within this block **with activations in f64 too** (the attribution
    of error of ADR 0118 step 3, the full-size reference).

    The caller has already called `double()` on `model` (the weights are the f16-rounded values
    widened to f64 as they are — the same weights as the GPU). The caller also widens the inputs
    (the f32 latent and text context) to f64 without changing their values. It runs the plain
    upstream forward like the f32 reference, with only these 2 differences:

    - upstream's `.float()` moves are a pass-through for f64 ({@link _float_keeps_float64}).
    - if a non-f64 floating-point value is created, it stops on exit. The only exception is the
      intermediate values of the time sinusoid (the `[1,freq_dim]` that upstream `Timesteps`
      builds in f32 — the GPU gets the same f32 values as `timesteps_proj`, and they widen to f64
      at the entry of `time_embedder`), allowed by their shapes
      ({@link _timestep_sinusoid_shapes}). Allowing by element count would let other f32 values of
      the same size or smaller (per-head scales, for example) slip through. The reference is batch
      1, so the shapes are pinned to batch 1 too.
    """
    narrow = sorted(
        {str(tensor.dtype) for tensor in (*model.parameters(), *model.buffers())}
        - {str(torch.float64)}
    )
    if narrow:
        raise AssertionError(
            f"an f64 reference, but the model still has {narrow} weights / buffers"
        )
    watch = _NarrowFloatWatch(_timestep_sinusoid_shapes(int(model.config.freq_dim)))
    with _float_keeps_float64(), watch:
        yield
    if watch.found:
        raise AssertionError(
            f"non-f64 values were created inside the f64 reference: {watch.found[:5]}"
        )


def reference_dit_f64(
    model: nn.Module,
    latents: torch.Tensor,
    timestep: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
) -> torch.Tensor:
    """The output `[1,C,F,H,W]` of {@link reference_dit} run under {@link float64_forward} (kept in
    f64 — whether to round is up to the side that stores it)."""
    with float64_forward(model):
        return reference_dit(model, latents.double(), timestep, encoder_hidden_states.double())
