"""Patch layer that makes Irodori-TTS v4 exportable with torch.export.

The targets are the **text condition encoder** (the bundled ModernBERT-ja-310m backbone) and
Irodori's own Transformer modules (speaker encoder = `ReferenceLatentEncoder` / duration
predictor / DiT). There are four patches:

1. **Bring ModernBertAttention's qkv extraction down to rank ≤ 4** — the original builds the
   **rank 5** `Wqkv(x).view(B, T, 3, H, D)` and then splits it into three with
   `unbind(dim=-3)`. After decomposition that becomes a rank-5 `slice`, which hits the strided
   kernels' rank limit (`ops.STRIDED_RANK` = 4) and cannot be exported. Replacing it with
   slices that cut the last dimension into three equal parts puts the same elements, in the same
   order, into the same three tensors (the equivalence argument of `_flat_qkv_attention_forward`
   below).
2. **Rewrite the complex-valued RoPE in real arithmetic** — the original
   (`irodori_tts.model.apply_rotary_emb`) builds a complex64 `[B,S,H,D/2]` with
   `view_as_complex`, multiplies in complex and converts back with `view_as_real`. complex is
   not in the IR's dtype vocabulary, and expanding into real pairs at rank 5 (`[B,S,H,D/2,2]`)
   hits the rank limit instead. It is rewritten **at rank 4** as "swap adjacent pairs +
   elementwise multiply by cos/sin" (the equivalence argument of `_real_pair_apply_rotary_emb`
   below).
3. **Split an RMSNorm with a rank-2 weight into rms_norm + mul** — the q/k norm weight is
   `[heads, head_dim]` (a separate scale per head), which does not fit the IR's `rms_norm`
   contract "weight = rank 1 of the normalized axis length" (ADR 0017). The normalization itself
   is over the last dimension (head_dim) only, so it is split into `rms_norm` (no weight = ones)
   and a broadcast multiply by the weight. **The rank-1 weight path (every existing RMSNorm)
   passes through verbatim.**
4. **Fold LowRankAdaLN's weightless RMS into `rms_norm`** — `LowRankAdaLN.forward` writes the
   normalization out as an expression (`x * rsqrt(mean(x²)+eps)`) instead of using the
   `RMSNorm` module, and `normalize._fold_rms_norm` only matches the rank-1 form with a weight,
   so `aten.mean.dim` / `aten.rsqrt.default` stay in the IR. Replacing it with
   `F.rms_norm(x, (D,), None, eps)` (no weight = synthesized ones) puts it on the existing fold
   path as is (`_folded_rms_low_rank_adaln_forward` below).

MUST: the patched modules are **eager-equivalent** to the unpatched ones (`irodori/export.py`
measures this with the real weights on every golden case and fails the whole export on any
difference). For 3 and 4 the op sequences map one to one, so **bit identity** holds
structurally. 2 is exactly equivalent as a formula, but **bit identity depends on the shape**
(the note on `_real_pair_apply_rotary_emb` — measured 0 at this checkpoint's head_dim 64).

MUST: the patches replace class attributes **process-wide**, so the "unpatched reference" can
be taken only once per process. {@link patches_applied} says whether they are applied, and the
caller (`irodori/export.py`) rejects an ordering violation loudly (it blocks the check from
becoming vacuous, i.e. a false PASS).

NOTE: the patches replace transformers' modeling code, so the version is pinned to
`transformers==5.14.1` (the docstring of `irodori/export.py`).
"""

from __future__ import annotations

from typing import Any

import torch

#: Whether the patches are applied. Kept only to make the side effect of the process-wide
#: replacement visible (taking the "unpatched reference" after patching makes the equivalence
#: check vacuous).
_APPLIED = False

#: `irodori_tts.model.apply_rotary_emb` before replacement. It is the delegate for calls made
#: with a complex table; it is captured because looking the original up by name after the
#: module global is replaced would return this function itself (infinite recursion).
_ORIGINAL_APPLY_ROTARY_EMB: Any = None


def patches_applied() -> bool:
    """Whether the patches have already been applied in this process.

    The check that lets the side taking reference values (the equivalence check) assert that
    they are "not applied yet".
    """
    return _APPLIED


# Third-party code notice (decided 2026-09-25 — `irodori/THIRD_PARTY_NOTICES.md`).
# `_flat_qkv_attention_forward` below is adapted from `ModernBertAttention.forward` in
# huggingface/transformers (`src/transformers/models/modernbert/modeling_modernbert.py`,
# `transformers==5.14.1`). It is verbatim except for how qkv is split; that part was changed.
# License: Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0).
# Upstream copyright notice, copied verbatim from the header of that file:
#
#   Copyright 2024 Answer.AI, LightOn, and contributors, and the HuggingFace Inc. team. All rights reserved.  # noqa: E501
def _flat_qkv_attention_forward(
    self: Any,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
    attention_mask: torch.Tensor | None = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Equivalent implementation of `ModernBertAttention.forward` (builds no rank 5).

    The only difference from the original is **how qkv is split**; the rest is verbatim (RoPE,
    the arguments to the attention interface, the output reshape and `Wo` / `out_drop`).

    Why it is equivalent: the original's `view(B, T, 3, H, D)` only regroups the last dimension
    `3·H·D` as `(3, H, D)` in C order, so the i-th tensor of `unbind(dim=-3)` is exactly the
    interval `[i·H·D, (i+1)·H·D)` of the original last dimension. The `slice` +
    `view(B, T, H, D)` here takes the same interval in the same order (the only difference is
    whether a copy is made).
    """
    batch, seq = hidden_states.shape[0], hidden_states.shape[1]
    heads = self.config.num_attention_heads
    width = heads * self.head_dim
    qkv = self.Wqkv(hidden_states)
    query_states, key_states, value_states = (
        qkv[..., start : start + width].view(batch, seq, heads, self.head_dim).transpose(1, 2)
        for start in (0, width, 2 * width)
    )

    from transformers.models.modernbert import modeling_modernbert as modernbert

    cos, sin = position_embeddings
    query_states, key_states = modernbert.apply_rotary_pos_emb(
        query_states, key_states, cos, sin, unsqueeze_dim=1
    )

    attention_interface = modernbert.ALL_ATTENTION_FUNCTIONS.get_interface(
        self.config._attn_implementation, modernbert.eager_attention_forward
    )
    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=self.attention_dropout if self.training else 0.0,
        scaling=self.head_dim**-0.5,
        sliding_window=self.sliding_window,
        deterministic=self.deterministic_flash_attn,
        **kwargs,
    )

    attn_output = attn_output.reshape(batch, seq, width).contiguous()
    return self.out_drop(self.Wo(attn_output)), attn_weights


#: Order along axis 1 of the real-valued RoPE table (`[S, 2, D]`: `0` = cos, `1` = sin). The
#: table is built by {@link real_pair_rope_table}; the export script bakes Tmax rows and passes
#: it in.
ROPE_TABLE_COS, ROPE_TABLE_SIN = 0, 1


def real_pair_rope_table(head_dim: int, length: int) -> torch.Tensor:
    """Real-valued RoPE table `[length, 2, head_dim]` (the same angles as `irodori_tts`'s
    complex table).

    The real / imaginary parts of the complex64 `[S, D/2]` that upstream's
    `precompute_freqs_cis(D, S)` returns, **duplicated into adjacent pairs** to widen them to
    `[S, D]`. When the complex multiply `(a+bi)(c+di) = (ac−bd) + (ad+bc)i` is written as real
    rank-4 ops, both components of a pair need the same cos / sin value
    ({@link _real_pair_apply_rotary_emb}).

    MUST: the angle formula comes from the upstream function (not a copy) — if upstream changes θ
    or the dimension split, a copied formula would silently stay at the old one.
    """
    from irodori_tts.model import precompute_freqs_cis

    freqs = precompute_freqs_cis(head_dim, length)
    cos = torch.repeat_interleave(freqs.real, 2, dim=-1)
    sin = torch.repeat_interleave(freqs.imag, 2, dim=-1)
    return torch.stack([cos, sin], dim=1)


def _real_pair_apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Equivalent implementation of `irodori_tts.model.apply_rotary_emb` (builds no complex).

    When `freqs_cis` is complex it **delegates to the original as is** (this patch replaces the
    function module-wide, so it must not change the semantics for callers that did not convert
    the table to real). Only calls made with the real table `[S, 2, D]`
    ({@link real_pair_rope_table}) take the real-valued formula below.

    Why it is equivalent: the original treats each two adjacent components of `x`'s last
    dimension as a complex number `a+bi` and computes `(a+bi)(c+di) = (ac−bd) + (ad+bc)i`. Here,
    for the same `x`,

    - `x * cos` is `(a·c, b·c)`
    - `(−b, a)` (adjacent pairs swapped, the first negated) times `sin` is `(−b·d, a·d)`

    and the sum `(a·c − b·d, b·c + a·d)` matches component by component. Replacing `x − y` with
    `x + (−y)` does not change the f32 rounding (multiplication is correctly rounded, so
    `(−b)·d == −(b·d)`).

    NOTE: **bit identity depends on the shape**. At head_dim 64 (all 8 layers of this
    checkpoint) the measured difference is 0, but at head_dim 8 it is off by 1 ulp
    (`irodori/tests/test_patch.py` pins both). _Speculation_: torch's
    complex multiply picks a different kernel per last-dimension length, and some of them
    contract the multiply-add into an FMA — the error against a double-precision reference is of
    the same order for both, so neither is more correct. The precise claim is therefore
    "exactly equivalent as a formula, rounding confirmed by measurement", and the real-weight
    check (atol 0 in `irodori/export.py`) stands as a measurement at this geometry.

    MUST: no intermediate is rank 5 (`[B,S,H,D/2,2]`) — that hits the strided kernels' rank
    limit (`ops.STRIDED_RANK` = 4). The swap is done on `[B,S,H·D/2,2]` with the head axis
    collapsed, then reshaped back (the element order does not change).
    """
    if freqs_cis.is_complex():
        if _ORIGINAL_APPLY_ROTARY_EMB is None:
            raise AssertionError(
                "no delegate for the complex table (the apply_rotary_emb from before replacement)"
            )
        return _ORIGINAL_APPLY_ROTARY_EMB(x, freqs_cis)
    # MUST: axis 1 is taken with a length-1 `slice`, not `select`. `aten.select.int` on an axis
    # of length 2 is not in the IR vocabulary (normalize folds only length-1 axes) and fails as
    # an unsupported op. `unsqueeze(0)` on `[S,1,D]` gives the intended `[1,S,1,D]`.
    cos = freqs_cis[:, ROPE_TABLE_COS : ROPE_TABLE_COS + 1].unsqueeze(0)
    sin = freqs_cis[:, ROPE_TABLE_SIN : ROPE_TABLE_SIN + 1].unsqueeze(0)
    pairs = x.reshape(x.shape[0], x.shape[1], -1, 2)
    swapped = torch.cat([-pairs[..., 1:2], pairs[..., 0:1]], dim=-1).reshape(x.shape)
    return x * cos + swapped * sin


# Third-party code notice (decided 2026-09-25 — `irodori/THIRD_PARTY_NOTICES.md`).
# The rank-1 branch of `_split_weight_rms_norm_forward` below reproduces `RMSNorm.forward` in
# Aratako/Irodori-TTS (https://github.com/Aratako/Irodori-TTS, `irodori_tts/model.py`) verbatim;
# the rank ≥ 2 branch is new. License: MIT License (the upstream repository's `LICENSE`).
# Upstream copyright notice, copied verbatim from that `LICENSE`:
#
#   Copyright (c) 2026 Aratako
def _split_weight_rms_norm_forward(self: Any, x: torch.Tensor) -> torch.Tensor:
    """Equivalent implementation of `irodori_tts.model.RMSNorm.forward` (splits a rank ≥ 2
    weight).

    The rank-1 weight path runs **the original formula as is** (`normalize._fold_rms_norm`
    folds it into one node as before). Only a rank ≥ 2 weight (the q/k norms'
    `[heads, head_dim]`) is split: the normalization goes to `F.rms_norm` (no weight) and the
    per-head scale to a broadcast multiply.

    Why it is equivalent: the original computes `x·rsqrt(mean(x²)+eps)` over the last dimension
    (head_dim) and then multiplies by the weight. `F.rms_norm(x, (head_dim,), None, eps)` is the
    same reduction and the same formula, and the multiplication order does not change (the
    claim is **bit identity** — measured 0).
    """
    x_dtype = x.dtype
    x = x.float()
    if self.weight.ndim == 1:
        x = x * torch.rsqrt((x * x).mean(dim=-1, keepdim=True) + self.eps)
        return (x * self.weight).to(x_dtype)
    normalized = torch.nn.functional.rms_norm(x, (x.shape[-1],), None, self.eps)
    return (normalized * self.weight).to(x_dtype)


# Third-party code notice (decided 2026-09-25 — `irodori/THIRD_PARTY_NOTICES.md`).
# `_folded_rms_low_rank_adaln_forward` below is adapted from `LowRankAdaLN.forward` in
# Aratako/Irodori-TTS (https://github.com/Aratako/Irodori-TTS, `irodori_tts/model.py`). It is
# verbatim except for the normalization line; that line was changed.
# License: MIT License (the upstream repository's `LICENSE`).
# Upstream copyright notice, copied verbatim from that `LICENSE`:
#
#   Copyright (c) 2026 Aratako
def _folded_rms_low_rank_adaln_forward(
    self: Any, x: torch.Tensor, cond_embed: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Equivalent implementation of `irodori_tts.model.LowRankAdaLN.forward` (moves the
    normalization onto `rms_norm`).

    The only difference from the original is **the one normalization line**; the rest (the
    low-rank corrections of shift / scale / gate, the modulation and the `tanh` gate) is
    verbatim.

    Why it is equivalent: the original's `x * rsqrt((x*x).mean(-1, keepdim=True) + eps)` is the
    very definition of `F.rms_norm(x, (D,), None, eps)` (no weight = ones), with the same
    reduction axis and the same formula. The claim is **bit identity** (atol 0 in
    `irodori/export.py` measures it with the real weights).

    The reason for the replacement is on the vocabulary side: this normalization does not go
    through the `RMSNorm` module, so `normalize._fold_rms_norm` (which matches the rank-1 form
    with a weight) does not catch it and `aten.mean.dim` / `aten.rsqrt.default` would stay in the
    IR.
    """
    shift, scale, gate = cond_embed.chunk(3, dim=-1)
    shift = self.shift_up(self.shift_down(torch.nn.functional.silu(shift))) + shift
    scale = self.scale_up(self.scale_down(torch.nn.functional.silu(scale))) + scale
    gate = self.gate_up(self.gate_down(torch.nn.functional.silu(gate))) + gate

    x_dtype = x.dtype
    x = x.float()
    x = torch.nn.functional.rms_norm(x, (x.shape[-1],), None, self.eps)
    x = x * (1.0 + scale) + shift
    gate = torch.tanh(gate)
    return x.to(x_dtype), gate


def apply_patches() -> None:
    """Apply the ModernBERT / Irodori model patches process-wide (idempotent).

    MUST: callers call this only after they have taken the "unpatched reference"
    ({@link patches_applied}).
    """
    global _APPLIED, _ORIGINAL_APPLY_ROTARY_EMB
    if _APPLIED:
        return
    from irodori_tts import model as irodori_model
    from transformers.models.modernbert.modeling_modernbert import ModernBertAttention

    ModernBertAttention.forward = _flat_qkv_attention_forward
    _ORIGINAL_APPLY_ROTARY_EMB = irodori_model.apply_rotary_emb
    # SelfAttention.forward looks the function up by its module-global name, so replacing the
    # global reaches it.
    irodori_model.apply_rotary_emb = _real_pair_apply_rotary_emb
    irodori_model.RMSNorm.forward = _split_weight_rms_norm_forward
    irodori_model.LowRankAdaLN.forward = _folded_rms_low_rank_adaln_forward
    _APPLIED = True
