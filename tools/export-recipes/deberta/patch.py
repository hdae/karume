"""Patch layer that turns DeBERTa-v2's relative-position index tables into **graph inputs**
(ADR 0045 wave 3).

`DisentangledSelfAttention.disentangled_attention_bias` builds `c2p_pos` / `p2c_pos` from
`relative_pos` (`[1, T, T]`) with a clamp. These two tables are **constants determined by T
alone**, so torch.export folds them at Tmax=512 and bakes in two `[1,512,512]` i32 tensors
(measured 2,097,152 B = 0.68% of the distribution). Their contents are Toeplitz and transposes of
each other, so the actual information equals an i32 vector of 1,023 elements.

They move to external supply for the same reason ADR 0013 did so for the `(T,T)` tables of flow /
voice. As a side effect **the Tmax=512 limit disappears** (with nothing baked in, the table size
is no longer an export-time constant).

MUST: after the patch the model is **eager-equivalent** — it introduces no difference other than
receiving the tables from outside. Equivalence with the original upstream implementation is pinned
by `deberta/tests/test_patch.py` (tiny config, random weights, `torch.equal`). The goldens of
`deberta.export` are taken after the patch is applied, so they only compare patched versions with
each other, and the TS-side parity test only checks table generation (agreement with
{@link build_rel_pos_tables}).

MUST: the replacement accepts only the form that receives the tables as a tuple (**no fallback**).
Silently dropping to the original implementation would mean "green with 2MiB still baked in", and
nobody would see that the patch came off.

## What these tables are (the canonical formula)

`build_relative_position(q, k, bucket_size=256, max_position=512)` is `rel[i,j] = i − j`
log-bucketed by `make_log_bucket_position`. Since `att_span = pos_ebd_size = position_buckets`
= 256:

    c2p_pos[i,j] = clamp(bucket(i − j) + 256, 0, 511)
    p2c_pos[i,j] = clamp(bucket(j − i) + 256, 0, 511) = c2p_pos[j,i]

The `p2c` side is the transpose because `build_rpos` returns `relative_pos` as is when
query_size == key_size and then applies `clamp(-rel + att_span, …)` (the bucketing is an odd
function).
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

#: Whether the replacement has been applied (it replaces a class attribute process-wide, so once).
_applied = False


def patch_applied() -> bool:
    """Whether the replacement is applied (a query so callers' ordering mistakes fail loudly)."""
    return _applied


def build_rel_pos_tables(
    length: int, *, position_buckets: int, max_position: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build `(c2p_pos, p2c_pos)` for the actual length `length` (**the Python-side source of
    truth that the TS mirror follows**).

    Both are `[length, length]` int64. `disentangled_attention_bias` broadcasts them to
    `[B*H, T, T]` with `expand`, so no leading axis of size 1 is added (the equivalent of the
    original implementation's `squeeze(0)`).
    """
    from transformers.models.deberta_v2.modeling_deberta_v2 import make_log_bucket_position

    ids = torch.arange(length, dtype=torch.long)
    rel = ids[:, None] - ids[None, :]
    bucket = make_log_bucket_position(rel, position_buckets, max_position).to(torch.long)
    span = position_buckets
    c2p_pos = torch.clamp(bucket + span, 0, span * 2 - 1)
    p2c_pos = torch.clamp(-bucket + span, 0, span * 2 - 1)
    return c2p_pos, p2c_pos


def _external_disentangled_attention_bias(
    self: nn.Module,
    query_layer: torch.Tensor,
    key_layer: torch.Tensor,
    relative_pos: Any,
    rel_embeddings: torch.Tensor,
    scale_factor: int,
) -> torch.Tensor:
    """Equivalent implementation that receives `(c2p_pos, p2c_pos)` in place of `relative_pos`.

    The only difference from the original is "receive the tables instead of building them". The
    clamp / build_rpos / squeeze go away, and the remaining bmm → gather → scale map one to one.
    """
    if not isinstance(relative_pos, tuple) or len(relative_pos) != 2:
        raise TypeError(
            "only an externally supplied (c2p_pos, p2c_pos) pair is accepted — the patch is"
            f" half-applied (got {type(relative_pos).__name__})"
        )
    from transformers.models.deberta_v2.modeling_deberta_v2 import scaled_size_sqrt

    c2p_pos, p2c_pos = relative_pos
    att_span = self.pos_ebd_size
    rel_embeddings = rel_embeddings[0 : att_span * 2, :].unsqueeze(0)
    heads = self.num_attention_heads
    repeats = query_layer.size(0) // heads
    pos_query_layer = self.transpose_for_scores(self.query_proj(rel_embeddings), heads).repeat(
        repeats, 1, 1
    )
    pos_key_layer = self.transpose_for_scores(self.key_proj(rel_embeddings), heads).repeat(
        repeats, 1, 1
    )

    # content -> position
    scale = scaled_size_sqrt(pos_key_layer, scale_factor)
    c2p_att = torch.bmm(query_layer, pos_key_layer.transpose(-1, -2))
    c2p_att = torch.gather(
        c2p_att,
        dim=-1,
        index=c2p_pos.expand([query_layer.size(0), query_layer.size(1), c2p_pos.size(-1)]),
    )
    score = c2p_att / scale.to(dtype=c2p_att.dtype)

    # position -> content
    scale = scaled_size_sqrt(pos_query_layer, scale_factor)
    p2c_att = torch.bmm(key_layer, pos_query_layer.transpose(-1, -2))
    p2c_att = torch.gather(
        p2c_att,
        dim=-1,
        index=p2c_pos.expand([query_layer.size(0), key_layer.size(-2), key_layer.size(-2)]),
    ).transpose(-1, -2)
    return score + p2c_att / scale.to(dtype=p2c_att.dtype)


def apply_external_rel_pos_patch() -> None:
    """Replace `disentangled_attention_bias` with the externally supplied version (process-wide,
    once).

    MUST: it need not be called **before** the model is built (it replaces a class attribute, so it
    takes effect afterwards too), but it must be called before the goldens are taken — replacing
    after them computes only the expected values on the original path.
    """
    global _applied
    if _applied:
        return
    from transformers.models.deberta_v2 import modeling_deberta_v2

    modeling_deberta_v2.DisentangledSelfAttention.disentangled_attention_bias = (
        _external_disentangled_attention_bias
    )
    _applied = True


def assert_supported(config: Any) -> None:
    """Check the config the replacement assumes (outside it, a different formula runs silently).

    The original implementation branches four ways on `pos_att_type` / `share_att_key`, but the
    replacement implements only the one ku-nlp's char-wwm uses (c2p + p2c, shared projection).
    """
    pos_att_type = set(getattr(config, "pos_att_type", ()) or ())
    if pos_att_type != {"c2p", "p2c"}:
        raise ValueError(f"pos_att_type is not c2p+p2c ({sorted(pos_att_type)})")
    if not getattr(config, "share_att_key", False):
        raise ValueError("share_att_key=False is a path the replacement does not implement")
    if int(getattr(config, "position_buckets", 0)) <= 0:
        raise ValueError(f"position_buckets is not positive ({config.position_buckets})")
