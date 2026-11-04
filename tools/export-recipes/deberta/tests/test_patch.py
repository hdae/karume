"""Unit tests that the replacement in `deberta.patch` is eager-equivalent to the original
upstream implementation (no real weights needed).

The goldens of `deberta.export` are taken after the patch is applied (`export_variant`), so they
only compare patched versions with each other. This is the only place that directly pins
equivalence with upstream `DisentangledSelfAttention.disentangled_attention_bias` — when the
transformers pin is raised and the original implementation changes, this fails.

With a tiny config (random weights from a fixed seed), the original function and the replacement
get the same inputs and are compared with `torch.equal`. Introducing no difference other than
receiving the tables from outside is the patch's MUST, so there is no tolerance.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("transformers")
from transformers import DebertaV2Config
from transformers.models.deberta_v2 import modeling_deberta_v2 as upstream

from deberta import patch

#: The original implementation before the replacement. Replacing a class attribute is
#: process-wide, so it is **saved at import time** (reading it after the patch would make a
#: tautological test that compares the replacement with itself).
UPSTREAM_BIAS = upstream.DisentangledSelfAttention.disentangled_attention_bias

HIDDEN = 16
HEADS = 2
BATCH = 2
#: Bucket size. The real model (ku-nlp char-wwm) has 256 / maximum position 512 — scaled down
#: keeping the ratio.
POSITION_BUCKETS = 8
MAX_POSITION = 16
#: Sequence length. The longest reaches the log-bucket range (|i − j| > half the bucket size) and
#: both ends of the clamp.
SEQUENCE = 24
#: Sequence lengths checked for equivalence (linear bucket range only, into the log range, up to
#: the clamp).
LENGTHS = (5, 13, SEQUENCE)
#: `1 + len(pos_att_type)` (the same value as upstream `DisentangledSelfAttention.forward`).
SCALE_FACTOR = 3


@pytest.fixture(autouse=True)
def restore_upstream_bias():
    """Undo the replacement (the class-attribute replacement and `_applied` are process-wide — the
    next case would be polluted)."""
    applied = patch._applied
    try:
        yield
    finally:
        upstream.DisentangledSelfAttention.disentangled_attention_bias = UPSTREAM_BIAS
        patch._applied = applied


def _config(**overrides: object) -> DebertaV2Config:
    """The smallest config that takes the same path as ku-nlp char-wwm (c2p + p2c, shared
    projection, bucketing)."""
    settings: dict[str, object] = {
        "hidden_size": HIDDEN,
        "num_attention_heads": HEADS,
        "relative_attention": True,
        "pos_att_type": ["c2p", "p2c"],
        "share_att_key": True,
        "position_buckets": POSITION_BUCKETS,
        "max_relative_positions": -1,
        "max_position_embeddings": MAX_POSITION,
    }
    settings.update(overrides)
    return DebertaV2Config(**settings)


def _tiny(length: int = SEQUENCE) -> tuple[upstream.DisentangledSelfAttention, SimpleNamespace]:
    torch.manual_seed(0)
    attention = upstream.DisentangledSelfAttention(_config()).eval()
    head_dim = HIDDEN // HEADS
    inputs = SimpleNamespace(
        query=torch.randn(BATCH * HEADS, length, head_dim),
        key=torch.randn(BATCH * HEADS, length, head_dim),
        rel_embeddings=torch.randn(attention.pos_ebd_size * 2, HIDDEN),
    )
    return attention, inputs


def _tables(
    attention: upstream.DisentangledSelfAttention, length: int = SEQUENCE
) -> tuple[torch.Tensor, torch.Tensor]:
    """The same spelling as `deberta.export.build_graph_inputs` (bucket size and maximum position
    from the module)."""
    return patch.build_rel_pos_tables(
        length,
        position_buckets=attention.position_buckets,
        max_position=attention.max_relative_positions,
    )


class TestExternalRelPosPatch:
    @pytest.mark.parametrize("length", LENGTHS)
    def test_the_patched_bias_equals_the_upstream_one_bit_for_bit(self, length: int) -> None:
        assert UPSTREAM_BIAS is not patch._external_disentangled_attention_bias
        attention, inputs = _tiny(length)
        with torch.no_grad():
            expected = attention.disentangled_attention_bias(
                inputs.query, inputs.key, None, inputs.rel_embeddings, SCALE_FACTOR
            )

            patch.apply_external_rel_pos_patch()
            actual = attention.disentangled_attention_bias(
                inputs.query,
                inputs.key,
                _tables(attention, length),
                inputs.rel_embeddings,
                SCALE_FACTOR,
            )

        assert patch.patch_applied()
        assert torch.equal(actual, expected)

    def test_the_p2c_table_is_the_transpose_of_the_c2p_table(self) -> None:
        """The claim of the canonical formula (module docstring) — bucketing is an odd function, so
        p2c = c2pᵀ."""
        c2p_pos, p2c_pos = _tables(_tiny()[0])

        assert torch.equal(p2c_pos, c2p_pos.T)

    def test_swapping_the_two_tables_breaks_the_equivalence(self) -> None:
        """Counterpart (not a tautology): the equivalence depends on the table contents — swapping
        c2p and p2c breaks it."""
        attention, inputs = _tiny()
        with torch.no_grad():
            expected = UPSTREAM_BIAS(
                attention, inputs.query, inputs.key, None, inputs.rel_embeddings, SCALE_FACTOR
            )
            c2p_pos, p2c_pos = _tables(attention)
            swapped = patch._external_disentangled_attention_bias(
                attention,
                inputs.query,
                inputs.key,
                (p2c_pos, c2p_pos),
                inputs.rel_embeddings,
                SCALE_FACTOR,
            )

        assert not torch.equal(swapped, expected)

    @pytest.mark.parametrize(
        "relative_pos",
        [None, torch.zeros(1, SEQUENCE, SEQUENCE, dtype=torch.long), (None, None, None)],
        ids=["none", "tensor", "triple"],
    )
    def test_the_patched_bias_refuses_anything_but_the_table_pair(self, relative_pos) -> None:
        """The no-fallback MUST — silently dropping to the original path would mean "green with
        2MiB still baked in"."""
        attention, inputs = _tiny()

        with pytest.raises(TypeError, match="externally supplied"):
            patch._external_disentangled_attention_bias(
                attention,
                inputs.query,
                inputs.key,
                relative_pos,
                inputs.rel_embeddings,
                SCALE_FACTOR,
            )


class TestAssertSupported:
    def test_it_accepts_the_path_the_patch_implements(self) -> None:
        patch.assert_supported(_config())

    @pytest.mark.parametrize(
        ("overrides", "message"),
        [
            ({"pos_att_type": ["c2p"]}, "pos_att_type"),
            ({"share_att_key": False}, "share_att_key"),
            ({"position_buckets": 0}, "position_buckets"),
        ],
        ids=["c2p-only", "separate-projection", "no-buckets"],
    )
    def test_it_rejects_a_config_the_patch_does_not_implement(
        self, overrides: dict[str, object], message: str
    ) -> None:
        with pytest.raises(ValueError, match=message):
            patch.assert_supported(_config(**overrides))
