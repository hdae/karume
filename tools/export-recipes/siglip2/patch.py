"""Patch layer that makes the SigLIP2 vision tower exportable with torch.export.

The target is only the image side of `google/siglip2-base-patch16-224` (the text tower is not
included). The config has `model_type="siglip"`, so transformers builds the **`Siglip*`** classes
(`SiglipVisionModel` / `SiglipVisionEmbeddings` / `SiglipMultiheadAttentionPoolingHead`), not
`Siglip2*` (the naflex variants). In transformers 5.x there is no intermediate `vision_model`
wrapper; `SiglipVisionModel` holds `embeddings` / `encoder` / `post_layernorm` / `head` directly.

The patches come in **two steps by strength of equivalence**:

- **Shape folds ({@link apply_shape_patches}) — bit-identical**
  ① `padding="valid"` (a string) on `patch_embedding` selects the `aten.conv2d.padding` overload.
     The conversion stage's HANDLERS have only `aten.conv2d.default`, so it becomes the equivalent
     `(0, 0)`.
  ② `position_embedding(position_ids)` is a row gather against the i64 buffer
     `position_ids = arange(N)`, so its output is the weight itself. Rewriting it as a direct
     addition removes **the i64 initializer** (the IR's semantic dtypes are f32 / i32 / bool,
     with no i64 — docs/ir-v2.md) and the `aten.embedding` node at the same time.
- **Making the MAP head's q/k/v explicit ({@link apply_map_head_patch}) — not bit-identical**
  ③ `SiglipMultiheadAttentionPoolingHead` holds a `torch.nn.MultiheadAttention` as is and calls
     it with `need_weights` defaulting to True, so it drops to the manual path (multiply q by 1/√d,
     then bmm). Extracting the activation-side k/v from the packed `in_proj_weight` becomes an
     `aten.select.int` on an axis of length 2, and `normalize._select_to_squeeze` "leaves static
     axes of length 2 or more unrewritten", so it remains as an unsupported op.
     `probe.repeat(B,1,1)` likewise remains as `aten.repeat.default` (`probe` is a Parameter, so it
     does not become a lifted constant). Rewriting it as explicit linears from `in_proj_weight`
     split into thirds + `scaled_dot_product_attention` removes both at once and aligns it with the
     decomposition path of the 12 encoder layers (mul×2 + bmm + softmax + bmm).

MUST: neither patch has a fallback (the same discipline as `deberta.patch`). A model that breaks
the assumptions fails on the spot rather than silently dropping to another numeric path.

MUST: ③ is equivalent as a formula but **not bit-identical** — `nn.MultiheadAttention` multiplies
only q by 1/√d, while SDPA scales q and k symmetrically by √(1/√d). The measured range is spelled
only in {@link siglip2.measurements.MAP_HEAD_MAXDIFF} (kept on **the side that does not load
torch** — the distribution side pulls it from there. The scale the difference is divided by (the
vector's L2 norm) is on the same side, in {@link siglip2.measurements.MAP_HEAD_VECTOR_NORM}.
Relative, it is ~1e-7).
The golden expected values are taken from the module **after the replacement**
(`siglip2/export.py`), so the e2e checks do not span this difference. The measurement before and
after the replacement is taken on every run by `siglip2.export.py --verify`, which evaluates at
two points.

MUST: the patches replace class attributes **process-wide**, so "the reference before patching"
can be taken only once per process. {@link patches_applied} answers whether they have been
applied, and the caller (`siglip2/export.py`) refuses an order violation by failing loudly
(blocking the vacuous check = a false PASS).

NOTE: this replaces transformers' modeling code, so the version is pinned to
`transformers==5.14.1` (the `siglip2` group of pyproject.toml).
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

#: Expected activation (`hidden_act`). Both the MAP head's MLP and the encoder's MLP follow from
#: it, so if it differs, the op table the script promises (`gelu_tanh`) silently changes to
#: another op.
EXPECTED_HIDDEN_ACT = "gelu_pytorch_tanh"

#: Whether the shape folds (① + ②) have been applied.
_SHAPE_APPLIED = False
#: Whether making the MAP head's q/k/v explicit (③) has been applied.
_MAP_HEAD_APPLIED = False


def patches_applied() -> bool:
    """Whether this process has applied a class attribute replacement (in either step).

    The check that lets the side taking the reference values (`siglip2.export.py --verify`) assert
    "not yet applied" — taking "the reference before patching" after patching would make the
    equivalence check vacuous with a difference of 0.
    """
    return _SHAPE_APPLIED or _MAP_HEAD_APPLIED


def assert_supported(config: Any) -> None:
    """Check the vision config the replacements assume.

    This checks only what **would drop to another numeric path without a shape error** if it
    differed.
    """
    if config.hidden_act != EXPECTED_HIDDEN_ACT:
        raise ValueError(
            f"hidden_act is not {EXPECTED_HIDDEN_ACT} ({config.hidden_act}) —"
            " the op table the script promises would change"
        )
    if config.num_channels != 3:
        raise ValueError(
            f"num_channels is not 3 ({config.num_channels}) — outside the preprocessing contract"
        )
    if config.hidden_size % config.num_attention_heads:
        raise ValueError(
            f"hidden_size {config.hidden_size} is not divisible by num_attention_heads"
            f" {config.num_attention_heads}"
        )


def _folded_embeddings_forward(
    self: nn.Module, pixel_values: torch.Tensor, interpolate_pos_encoding: bool = False
) -> torch.Tensor:
    """Equivalent implementation of `SiglipVisionEmbeddings.forward` (the position embedding
    becomes a direct addition of the weight).

    The only difference from the original is replacing `self.position_embedding(self.position_ids)`
    with `self.position_embedding.weight` (the leading axis of 1 is equivalent through broadcast).
    {@link apply_shape_patches} checks on the actual buffer that `position_ids` is `arange(N)`.
    """
    if interpolate_pos_encoding:
        raise NotImplementedError(
            "interpolate_pos_encoding is a path the replacement does not have —"
            " the direct addition of the weight equals the gather only at a fixed resolution"
        )
    target_dtype = self.patch_embedding.weight.dtype
    patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))
    return patch_embeds.flatten(2).transpose(1, 2) + self.position_embedding.weight


def _map_head_forward(self: nn.Module, hidden_state: torch.Tensor) -> torch.Tensor:
    """Equivalent implementation of `SiglipMultiheadAttentionPoolingHead.forward` (q/k/v as
    explicit linears).

    The only difference from the original is how the attention is called; the rest (residual +
    layernorm + mlp and taking the first row) is verbatim. As inside `nn.MultiheadAttention`, q
    comes from the probe (a sequence of length 1) and k / v from the input sequence.
    """
    batch = hidden_state.shape[0]
    # MUST: `expand`, not `repeat` — probe is a Parameter, so it does not become a lifted
    # constant, and `aten.repeat.default` is not folded and remains an unsupported op with
    # batch>1 (measured).
    probe = self.probe.expand(batch, -1, -1)
    shape = (batch, -1, self.num_heads, self.head_dim)
    query = self.q_proj(probe).reshape(shape).transpose(1, 2)
    key = self.k_proj(hidden_state).reshape(shape).transpose(1, 2)
    value = self.v_proj(hidden_state).reshape(shape).transpose(1, 2)
    attended = nn.functional.scaled_dot_product_attention(query, key, value)
    attended = attended.transpose(1, 2).reshape(batch, -1, self.num_heads * self.head_dim)
    hidden_state = self.o_proj(attended)

    residual = hidden_state
    hidden_state = self.layernorm(hidden_state)
    hidden_state = residual + self.mlp(hidden_state)
    return hidden_state[:, 0]


def _assert_position_ids_are_arange(embeddings: nn.Module) -> None:
    """Check on the actual buffer that `position_ids` is `arange(N)` broadcast to `[1, N]`.

    The direct addition of the weight equals the row gather only in this form — with a different
    order, the rewrite still passes but becomes **a different position embedding** (the shape does
    not change, so nothing fails).

    NOTE: `image_size` need not be divisible by `patch_size`. Upstream sets the row count to
    `(image_size // patch_size) ** 2`, and the conv with padding='valid' gives
    `floor((image_size - patch_size) / patch_size) + 1 = image_size // patch_size`, so even when the
    edge is dropped both sides take the same floor (example = so400m's 384 / 14 = 27 grid; the
    bottom-right 6 pixels are dropped upstream as well). If the row counts disagreed, the direct
    addition would fail with a shape error.
    """
    positions = int(embeddings.position_embedding.weight.shape[0])
    expected = torch.arange(positions).reshape(1, positions)
    actual = embeddings.position_ids
    if actual.shape != expected.shape or not torch.equal(actual.cpu(), expected):
        raise ValueError(
            f"position_ids is not arange({positions}) in [1, N] form (shape"
            f" {tuple(actual.shape)}) — the direct addition of the weight would not equal the"
            " gather"
        )


def apply_shape_patches(embeddings: nn.Module) -> None:
    """Apply ① the conv's string padding fix and ② the position embedding fold (the
    **bit-identical** step).

    ① replaces an instance attribute, ② a class attribute. In both the operation sequences
    correspond one to one, so pooler_output is bit-identical (point 1 of
    `siglip2.export.py --verify` measures it on every run).
    """
    global _SHAPE_APPLIED
    padding = embeddings.patch_embedding.padding
    if padding != "valid":
        raise ValueError(f"patch_embedding.padding is not 'valid' ({padding!r})")
    embeddings.patch_embedding.padding = (0, 0)
    _assert_position_ids_are_arange(embeddings)
    type(embeddings).forward = _folded_embeddings_forward
    _SHAPE_APPLIED = True


def apply_map_head_patch(head: nn.Module) -> None:
    """③ Split the MAP head's `nn.MultiheadAttention` into explicit q/k/v linears (**not**
    bit-identical).

    The packed `in_proj_weight` `[3E, E]` / `in_proj_bias` `[3E]` are split into three parts of E
    each and moved onto three `nn.Linear(E, E)`, and `out_proj` is carried over as `o_proj`. The
    order of the split (q → k → v) follows `_in_projection_packed` of `nn.MultiheadAttention`
    verbatim.
    """
    global _MAP_HEAD_APPLIED
    attention = head.attention
    if not isinstance(attention, nn.MultiheadAttention):
        raise TypeError(f"head.attention is not nn.MultiheadAttention ({type(attention).__name__})")
    if attention.in_proj_weight is None or attention.in_proj_bias is None:
        raise ValueError("q/k/v with separate weights is a path the replacement does not have")
    if not attention.batch_first:
        raise ValueError(
            "batch_first=False is a path the replacement does not have (the shapes become"
            " seq-first)"
        )
    if attention.bias_k is not None or attention.bias_v is not None or attention.add_zero_attn:
        raise ValueError(
            "add_bias_kv / add_zero_attn is a path the replacement does not have (the formula"
            " changes)"
        )
    embed_dim = int(attention.embed_dim)
    num_heads = int(attention.num_heads)
    if embed_dim % num_heads:
        raise ValueError(f"embed_dim {embed_dim} is not divisible by num_heads {num_heads}")

    weight = attention.in_proj_weight.detach()
    bias = attention.in_proj_bias.detach()
    for index, name in enumerate(("q_proj", "k_proj", "v_proj")):
        rows = slice(index * embed_dim, (index + 1) * embed_dim)
        projection = nn.Linear(embed_dim, embed_dim, bias=True)
        projection.weight = nn.Parameter(weight[rows].clone())
        projection.bias = nn.Parameter(bias[rows].clone())
        setattr(head, name, projection)
    head.o_proj = attention.out_proj
    head.num_heads = num_heads
    head.head_dim = embed_dim // num_heads
    del head.attention

    type(head).forward = _map_head_forward
    _MAP_HEAD_APPLIED = True
