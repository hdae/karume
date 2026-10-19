"""Lifted check of the RoPE buffers — model-independent export validation.

While the RoPE frequency seed (`inv_freq`) stays a buffer it is not a leaf of constant folding,
and sin / cos become necessary in the IR vocabulary. {@link lift_rope_buffers} demotes it to a
plain attribute (a lifted tensor constant) so that it becomes a folding leaf, and
{@link assert_rope_lifted} is the check that keeps a scan that found nothing from going on to
export.

The target is the transformers-style RoPE buffer naming (suffix match) and **does not depend on a
particular model / family** — Anima's Qwen3, EmbeddingGemma's Gemma3 and Irodori's text backbone
use the same form (recovered from Anima's patch layer in ADR 0065 stage 2).
"""

from __future__ import annotations

from torch import nn

#: **Buffer-name suffixes** of the RoPE frequency seed. While it stays a buffer it is not a leaf
#: of constant folding, and sin / cos remain in the IR.
#:
#: The match is by suffix rather than exact name because implementations that split RoPE per
#: layer_type use prefixed names (Gemma3 in transformers 5.14.1 has `sliding_attention_inv_freq` /
#: `full_attention_inv_freq` and their `original_*`). The existing bare names are their own
#: suffixes, so the accepted set only grows, and the hit count on existing models does not change.
ROPE_BUFFER_NAMES = ("inv_freq", "original_inv_freq")


def lift_rope_buffers(root: nn.Module) -> int:
    """Demote RoPE `inv_freq` buffers to plain attributes (lifted tensor constants).

    Folding leaves are not extended to parameters / buffers to avoid baking huge constants
    (convert._classify_foldable), but `inv_freq` is the seed of a position table with head_dim/2
    elements, and the folded result (the cos/sin table) also fits in Tmax × head_dim. Without the
    demotion, sin / cos become necessary in the IR vocabulary.

    MUST: when nothing can be demoted, the caller fails (if the attribute name changes upstream,
    the scan silently finds nothing and from then on sin/cos remain in the IR for every model — the
    check that keeps this from passing tautologically lives in the caller).

    The scan is a **suffix match** on {@link ROPE_BUFFER_NAMES} (some implementations have names
    with a layer_type prefix). `_buffers` is rewritten during the scan, so the names are collected
    first.
    """
    lifted = 0
    for module in root.modules():
        for name in [key for key in module._buffers if key.endswith(ROPE_BUFFER_NAMES)]:
            tensor = module._buffers.pop(name)
            setattr(module, name, tensor.detach().clone())
            lifted += 1
    return lifted


def _rope_attribute_count(root: nn.Module) -> int:
    """Count of places holding the RoPE frequency seed (buffer entries + demoted plain attributes).

    The demotion is a destructive `_buffers.pop`, so counting only buffer entries makes **a second
    pass over the same module** always count 0. `lift_rope_buffers` calls `setattr` **after** the
    pop, so `nn.Module.__setattr__` finds no buffer entry and the demoted name goes into the plain
    `__dict__` — counting both keeps the power to detect "the scan finds nothing (the attribute
    name changed upstream)" while making the check idempotent. The rule is the same suffix match as
    the demotion ({@link ROPE_BUFFER_NAMES}).
    """
    return sum(
        1
        for module in root.modules()
        for names in (module._buffers, module.__dict__)
        for key in names
        if key.endswith(ROPE_BUFFER_NAMES)
    )


def assert_rope_lifted(root: nn.Module, where: str) -> None:
    """Apply {@link lift_rope_buffers} and fail when no RoPE frequency seed exists (the check
    against a tautological pass).

    `where` is the caller's context that says which model / component a failure is about.

    A second pass over an already demoted module passes (what is counted is
    {@link _rope_attribute_count} = the total of buffer entries and plain attributes). Looking at
    the count the demotion returns would always give 0 on the second pass, and export would stop
    with a false "the attribute name changed upstream".
    """
    lift_rope_buffers(root)
    if _rope_attribute_count(root) == 0:
        raise ValueError(
            f"{where}: no RoPE buffer {ROPE_BUFFER_NAMES} was found"
            " — the demotion scan finds nothing (the attribute name may have changed upstream). "
            "Exporting as is leaves sin / cos in the IR"
        )
