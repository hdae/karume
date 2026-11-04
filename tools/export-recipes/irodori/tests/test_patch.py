"""Regression tests that the rewrites of `irodori.patch` are equivalent to the original
implementation (the part that needs no real weights).

The end-to-end equivalence check with the real weights is `irodori/export.py` (it checks every
golden case on each emit and fails unless the result is bit-identical). Pinned here is only the
unit-level equivalence that runs with transformers / a clone of the Irodori implementation.

MUST: the claim is **bit-identical**. The only replacements are how qkv is taken apart, writing
RoPE without complex numbers, splitting the RMSNorm weight and folding AdaLN's weightless RMS,
none of which changes the order of operations or the rounding — passing on "the difference is
small" would let a mix-up (q/k/v in the wrong order, cos and sin swapped, the per-head weight
transposed, shift and scale swapped) slip through with "values that happen to be close".
"""

from __future__ import annotations

import pytest
import torch

from irodori import patch as patch_irodori

modernbert = pytest.importorskip("transformers.models.modernbert.modeling_modernbert")


class TestTheSplitArgumentItself:
    """Pins the basis of the patch's equivalence (unbind of rank 5 = splitting the last dim into
    three) with plain torch."""

    def test_unbind_of_the_rank5_view_equals_three_last_dim_slices(self):
        batch, seq, heads, head_dim = 2, 5, 3, 4
        width = heads * head_dim
        qkv = torch.randn(batch, seq, 3 * width)

        original = qkv.view(batch, seq, 3, heads, head_dim).unbind(dim=-3)
        flat = tuple(
            qkv[..., start : start + width].view(batch, seq, heads, head_dim)
            for start in (0, width, 2 * width)
        )

        for index, (lhs, rhs) in enumerate(zip(original, flat, strict=True)):
            assert torch.equal(lhs, rhs), f"slice {index} does not match"


def _tiny_attention(
    layer_idx: int,
) -> tuple[torch.nn.Module, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """Build a small `ModernBertAttention` with its input / RoPE table.

    NOTE: `num_hidden_layers` must be 3 or more — with 1, `layer_types` is only
    `["full_attention"]`, and the 5.14.1 config flattens the RoPE parameters into a non-nested
    form that fails strict dataclass validation (the 25 layers of the real weights mix the two
    kinds, so they do not take this path).
    """
    from transformers.models.modernbert.configuration_modernbert import ModernBertConfig

    config = ModernBertConfig(
        hidden_size=16,
        num_attention_heads=4,
        intermediate_size=32,
        num_hidden_layers=3,
        attention_dropout=0.0,
    )
    layer_type = config.layer_types[layer_idx]
    torch.manual_seed(0)
    attention = modernbert.ModernBertAttention(config=config, layer_idx=layer_idx).eval()
    hidden = torch.randn(1, 6, config.hidden_size)
    rotary = modernbert.ModernBertRotaryEmbedding(config=config)
    position_ids = torch.arange(hidden.shape[1]).unsqueeze(0)
    return attention, hidden, rotary(hidden, position_ids, layer_type)


class TestPatchedAttentionIsBitIdentical:
    @pytest.mark.parametrize("layer_idx", [0, 1])
    def test_the_patched_forward_reproduces_the_original(self, restore_forward, layer_idx):
        """Checks both full_attention (layer 0) and sliding_attention (layer 1).

        The sliding side is the path where `self.sliding_window` reaches the attention
        interface, so a copy mistake that drops one argument shows here.
        """
        attention, hidden, position_embeddings = _tiny_attention(layer_idx)
        with torch.no_grad():
            expected, _ = attention(hidden, position_embeddings=position_embeddings)

        patch_irodori.apply_patches()
        # MUST: check that the replacement really took effect (if `_APPLIED` stayed True from an
        # earlier test, apply would be a no-op and the test would pass tautologically by
        # comparing the original with itself).
        assert modernbert.ModernBertAttention.forward is patch_irodori._flat_qkv_attention_forward
        with torch.no_grad():
            actual, _ = attention(hidden, position_embeddings=position_embeddings)

        assert torch.equal(actual, expected)

    def test_applying_twice_is_a_no_op(self, restore_forward):
        patch_irodori.apply_patches()
        first = modernbert.ModernBertAttention.forward
        patch_irodori.apply_patches()

        assert modernbert.ModernBertAttention.forward is first
        assert patch_irodori.patches_applied()


class TestPatchedRotaryEmbedding:
    """RoPE with the complex table opened into real pairs reproduces the original implementation.

    Bit identity **depends on the shape** (the NOTE of `_real_pair_apply_rotary_emb`). The real
    weights' geometry (head_dim 64) must be bit-identical, and the side that diverges (head_dim
    8) is pinned with a **1 ulp bound** — loosening to "a small difference is fine" would let a
    mix-up such as swapping cos/sin slip through, so the diverging side gets a bound too.
    """

    #: The same head_dim as the real weights' speaker encoder / DiT (768/12 = 64, 1280/20 = 64).
    MODEL_HEAD_DIM = 64
    #: The geometry where the rounding diverges (the observation point of the regression).
    SMALL_HEAD_DIM = 8
    HEADS, LENGTH = 3, 11

    #: A bound of about 1 ulp for inputs in the range O(1) (f32 eps = 1.19e-7). The error of a
    #: mix-up is O(1), so this bound lets "rounding alone" through.
    ONE_ULP = 1.2e-7

    def _inputs(self, irodori_model, head_dim: int):
        torch.manual_seed(0)
        x = torch.randn(1, self.LENGTH, self.HEADS, head_dim)
        complex_table = irodori_model.precompute_freqs_cis(head_dim, self.LENGTH)
        real_table = patch_irodori.real_pair_rope_table(head_dim, self.LENGTH)
        return x, complex_table, real_table

    def test_the_model_geometry_is_bit_identical(self, restore_forward):
        irodori_model = restore_forward
        x, complex_table, real_table = self._inputs(irodori_model, self.MODEL_HEAD_DIM)
        with torch.no_grad():
            expected = irodori_model.apply_rotary_emb(x, complex_table)

        patch_irodori.apply_patches()
        with torch.no_grad():
            actual = irodori_model.apply_rotary_emb(x, real_table)

        assert torch.equal(actual, expected)

    def test_a_smaller_head_dim_stays_within_one_ulp(self, restore_forward):
        irodori_model = restore_forward
        x, complex_table, real_table = self._inputs(irodori_model, self.SMALL_HEAD_DIM)
        with torch.no_grad():
            expected = irodori_model.apply_rotary_emb(x, complex_table)

        patch_irodori.apply_patches()
        with torch.no_grad():
            actual = irodori_model.apply_rotary_emb(x, real_table)

        assert float((actual - expected).abs().max()) <= self.ONE_ULP

    def test_a_complex_table_still_takes_the_original_path(self, restore_forward):
        """MUST: the replacement is global, so it does not change the semantics for callers that
        did not move to real pairs."""
        irodori_model = restore_forward
        x, complex_table, _real = self._inputs(irodori_model, self.MODEL_HEAD_DIM)
        with torch.no_grad():
            expected = irodori_model.apply_rotary_emb(x, complex_table)

        patch_irodori.apply_patches()
        with torch.no_grad():
            actual = irodori_model.apply_rotary_emb(x, complex_table)

        assert torch.equal(actual, expected)

    def test_swapping_cos_and_sin_breaks_the_result(self, restore_forward):
        """Check against tautology — swapping the table's two planes changes the value (= the
        test sees the table's orientation)."""
        irodori_model = restore_forward
        x, complex_table, real_table = self._inputs(irodori_model, self.MODEL_HEAD_DIM)
        with torch.no_grad():
            expected = irodori_model.apply_rotary_emb(x, complex_table)

        patch_irodori.apply_patches()
        with torch.no_grad():
            actual = irodori_model.apply_rotary_emb(x, real_table.flip(1))

        assert not torch.allclose(actual, expected)


class TestPatchedRmsNormIsBitIdentical:
    """Splitting a rank-2 weight does not differ from the original by a single bit, and the
    rank-1 path stays verbatim."""

    HEADS, HEAD_DIM, EPS = 4, 6, 1e-5

    def test_a_rank2_weight_is_split_without_changing_the_value(self, restore_forward):
        irodori_model = restore_forward
        torch.manual_seed(1)
        norm = irodori_model.RMSNorm((self.HEADS, self.HEAD_DIM), eps=self.EPS)
        torch.nn.init.normal_(norm.weight)
        x = torch.randn(1, 5, self.HEADS, self.HEAD_DIM)
        with torch.no_grad():
            expected = norm(x)

        patch_irodori.apply_patches()
        with torch.no_grad():
            actual = norm(x)

        assert torch.equal(actual, expected)

    def test_a_rank1_weight_takes_the_verbatim_path(self, restore_forward):
        irodori_model = restore_forward
        torch.manual_seed(2)
        norm = irodori_model.RMSNorm(self.HEAD_DIM, eps=self.EPS)
        torch.nn.init.normal_(norm.weight)
        x = torch.randn(1, 5, self.HEAD_DIM)
        with torch.no_grad():
            expected = norm(x)

        patch_irodori.apply_patches()
        with torch.no_grad():
            actual = norm(x)

        assert torch.equal(actual, expected)


class TestPatchedLowRankAdaLnIsBitIdentical:
    """Moving the weightless RMS of the DiT's `LowRankAdaLN` onto `rms_norm` does not change the
    value."""

    MODEL_DIM, RANK, EPS = 12, 5, 1e-5

    def _module(self, irodori_model):
        """Build an AdaLN whose low-rank correction **is in effect**.

        MUST: refill the up-side weight / bias, initialized by `nn.init.zeros_`, with random
        values — with the plain initialization the correction term is the identity, and a copy
        mistake (swapping shift and scale, dropping `+ shift`) slips through as "both give the
        same value".
        """
        torch.manual_seed(3)
        module = irodori_model.LowRankAdaLN(
            model_dim=self.MODEL_DIM, rank=self.RANK, eps=self.EPS
        ).eval()
        for parameter in module.parameters():
            torch.nn.init.normal_(parameter)
        x = torch.randn(1, 4, self.MODEL_DIM)
        cond_embed = torch.randn(1, 1, 3 * self.MODEL_DIM)
        return module, x, cond_embed

    def test_the_patched_forward_reproduces_the_original(self, restore_forward):
        irodori_model = restore_forward
        module, x, cond_embed = self._module(irodori_model)
        with torch.no_grad():
            expected_x, expected_gate = module(x, cond_embed)

        patch_irodori.apply_patches()
        assert (
            irodori_model.LowRankAdaLN.forward is patch_irodori._folded_rms_low_rank_adaln_forward
        )
        with torch.no_grad():
            actual_x, actual_gate = module(x, cond_embed)

        assert torch.equal(actual_x, expected_x)
        assert torch.equal(actual_gate, expected_gate)

    def test_the_low_rank_correction_is_not_inert(self, restore_forward):
        """Check against tautology — if the correction had no effect, this equivalence check
        would see nothing."""
        irodori_model = restore_forward
        module, x, cond_embed = self._module(irodori_model)
        with torch.no_grad():
            with_correction, _gate = module(x, cond_embed)
            for name in ("shift_up", "scale_up", "gate_up"):
                torch.nn.init.zeros_(getattr(module, name).weight)
                torch.nn.init.zeros_(getattr(module, name).bias)
            without_correction, _gate = module(x, cond_embed)

        assert not torch.allclose(with_correction, without_correction)
