"""Regression tests that the rewrites of `siglip2.patch` are equivalent to the original
implementation (the part that needs no real weights).

The end-to-end equivalence check with real weights is `siglip2.export.py --verify` (evaluated at
two points). Here, unit-level equivalence that also runs in environments without transformers is
pinned against **skeletons that copy the upstream forward verbatim** ({@link StubEmbeddings} /
{@link StubMapHead}).

The strength of the claim differs per step (the `siglip2.patch` docstring):

- The shape folds (conv padding / position embedding) are **bit-identical**. The operation
  sequences correspond one to one, so accepting "the difference is small" could let a mix-up in
  the order of the position embedding through.
- Making the MAP head's q/k/v explicit is only **close in value**, because
  `nn.MultiheadAttention` multiplies only q by 1/√d while SDPA scales q and k symmetrically. The
  tolerance stays at f32 rounding width (1e-5), though, so that a q/k/v order swap or a wrong head
  split cannot pass.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from siglip2 import patch as patch_siglip2

#: Equivalence tolerance of the MAP head. Wide enough only for the difference in how
#: `nn.MultiheadAttention` and SDPA apply the scale; a q/k/v swap or a wrong head split does not fit
#: within this magnitude.
MAP_HEAD_ATOL = 1e-5

EMBED_DIM = 24
NUM_HEADS = 4
SEQUENCE = 7


class StubMapHead(nn.Module):
    """Skeleton of `SiglipMultiheadAttentionPoolingHead` (forward is a verbatim copy of
    upstream)."""

    def __init__(self, embed_dim: int = EMBED_DIM, num_heads: int = NUM_HEADS) -> None:
        super().__init__()
        self.probe = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.layernorm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 2 * embed_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(2 * embed_dim, embed_dim),
        )

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        batch_size = hidden_state.shape[0]
        probe = self.probe.repeat(batch_size, 1, 1)
        hidden_state = self.attention(probe, hidden_state, hidden_state)[0]
        residual = hidden_state
        hidden_state = self.layernorm(hidden_state)
        hidden_state = residual + self.mlp(hidden_state)
        return hidden_state[:, 0]


class StubEmbeddings(nn.Module):
    """Skeleton of `SiglipVisionEmbeddings` (forward is a verbatim copy of upstream)."""

    def __init__(self, image_size: int = 8, patch_size: int = 4, embed_dim: int = 6) -> None:
        super().__init__()
        self.patch_embedding = nn.Conv2d(
            in_channels=3,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            padding="valid",
        )
        positions = (image_size // patch_size) ** 2
        self.position_embedding = nn.Embedding(positions, embed_dim)
        self.register_buffer(
            "position_ids", torch.arange(positions).expand((1, -1)), persistent=False
        )

    def forward(
        self, pixel_values: torch.Tensor, interpolate_pos_encoding: bool = False
    ) -> torch.Tensor:
        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))
        embeddings = patch_embeds.flatten(2).transpose(1, 2)
        return embeddings + self.position_embedding(self.position_ids)


@pytest.fixture(autouse=True)
def restore_patched_attributes():
    """Undo every replacement (class attribute replacements are process-wide — they would
    contaminate the next case's reference)."""
    forwards = (StubMapHead.forward, StubEmbeddings.forward)
    flags = (patch_siglip2._SHAPE_APPLIED, patch_siglip2._MAP_HEAD_APPLIED)
    try:
        yield
    finally:
        StubMapHead.forward, StubEmbeddings.forward = forwards
        patch_siglip2._SHAPE_APPLIED, patch_siglip2._MAP_HEAD_APPLIED = flags


class Config:
    """Only the fields of `SiglipVisionConfig` that `assert_supported` reads."""

    def __init__(self, **overrides: object) -> None:
        self.hidden_act = patch_siglip2.EXPECTED_HIDDEN_ACT
        self.num_channels = 3
        self.image_size = 224
        self.patch_size = 16
        self.hidden_size = 768
        self.num_attention_heads = 12
        self.__dict__.update(overrides)


class TestAssertSupported:
    def test_accepts_the_shipped_vision_config(self):
        patch_siglip2.assert_supported(Config())

    def test_accepts_a_resolution_that_is_not_a_multiple_of_the_patch_size(self):
        """so400m is a 384 / 14 = 27 grid that drops the bottom-right 6 pixels (upstream as well)
        — a supported path.

        Both the patch count and the row count of the position embedding follow from the floor of
        `image_size // patch_size`, so the two sides agree even when it does not divide evenly
        (the NOTE of `_assert_position_ids_are_arange`).
        """
        patch_siglip2.assert_supported(
            Config(image_size=384, patch_size=14, hidden_size=1152, num_attention_heads=16)
        )

    @pytest.mark.parametrize(
        ("overrides", "message"),
        [
            ({"hidden_act": "gelu"}, "hidden_act"),
            ({"num_channels": 1}, "num_channels"),
            ({"num_attention_heads": 7}, "is not divisible by num_attention_heads"),
        ],
    )
    def test_fails_loudly_outside_the_written_path(self, overrides, message):
        with pytest.raises(ValueError, match=message):
            patch_siglip2.assert_supported(Config(**overrides))


class TestShapePatchesAreBitExact:
    """MUST: the claim is bit-identity (point 1 of `--verify` measures the same claim with real
    weights)."""

    @pytest.fixture
    def pixels(self):
        torch.manual_seed(0)
        return torch.randn(2, 3, 8, 8)

    def test_folded_position_embedding_is_bit_exact(self, pixels):
        embeddings = StubEmbeddings().eval()
        with torch.no_grad():
            reference = embeddings(pixels)

        patch_siglip2.apply_shape_patches(embeddings)
        with torch.no_grad():
            patched = embeddings(pixels)

        assert torch.equal(patched, reference)

    def test_conv_padding_becomes_the_default_overload_form(self, pixels):
        embeddings = StubEmbeddings().eval()

        patch_siglip2.apply_shape_patches(embeddings)

        assert embeddings.patch_embedding.padding == (0, 0)

    def test_interpolate_pos_encoding_is_refused(self, pixels):
        """A path that is equivalent only at a fixed resolution is not let through silently."""
        embeddings = StubEmbeddings().eval()
        patch_siglip2.apply_shape_patches(embeddings)

        with pytest.raises(NotImplementedError, match="interpolate_pos_encoding"):
            embeddings(pixels, interpolate_pos_encoding=True)

    def test_shuffled_position_ids_fail_loudly(self):
        """If the order is not `arange`, the direct addition becomes a different position
        embedding (with the shape still matching)."""
        embeddings = StubEmbeddings().eval()
        embeddings.position_ids = embeddings.position_ids.flip(-1)

        with pytest.raises(ValueError, match="arange"):
            patch_siglip2.apply_shape_patches(embeddings)


class TestMapHeadSplit:
    @pytest.fixture
    def hidden(self):
        torch.manual_seed(0)
        return torch.randn(3, SEQUENCE, EMBED_DIM)

    def test_split_qkv_matches_the_packed_multihead_attention(self, hidden):
        head = StubMapHead().eval()
        with torch.no_grad():
            reference = head(hidden)

        patch_siglip2.apply_map_head_patch(head)
        with torch.no_grad():
            patched = head(hidden)

        assert torch.allclose(patched, reference, atol=MAP_HEAD_ATOL, rtol=0.0)

    def test_the_packed_projection_is_split_in_qkv_order(self):
        """MUST: the order of the three-way split is q → k → v (a swap would still match in
        shape)."""
        head = StubMapHead().eval()
        packed = head.attention.in_proj_weight.detach().clone()

        patch_siglip2.apply_map_head_patch(head)

        for index, projection in enumerate((head.q_proj, head.k_proj, head.v_proj)):
            rows = slice(index * EMBED_DIM, (index + 1) * EMBED_DIM)
            assert torch.equal(projection.weight, packed[rows])

    def test_the_packed_attention_module_is_gone(self):
        """If the original path remained after the replacement, nobody could see that it was
        bypassed."""
        head = StubMapHead().eval()

        patch_siglip2.apply_map_head_patch(head)

        assert not hasattr(head, "attention")
        assert patch_siglip2.patches_applied()

    def test_probe_is_broadcast_not_repeated(self, hidden):
        """MUST: broadcast with `expand` (`repeat` remains as an unsupported op with batch>1).

        `expand` only reads its input, so rewriting `probe` changes the output — this checks by
        value that it was not folded into a constant (the shape alone shows no difference from
        `repeat`).
        """
        head = StubMapHead().eval()
        patch_siglip2.apply_map_head_patch(head)
        with torch.no_grad():
            before = head(hidden)
            head.probe += 1.0
            after = head(hidden)

        assert not torch.allclose(before, after)

    @pytest.mark.parametrize(
        ("mutate", "message"),
        [
            (lambda head: setattr(head, "attention", nn.Identity()), "MultiheadAttention"),
            (lambda head: setattr(head.attention, "batch_first", False), "batch_first"),
            (lambda head: setattr(head.attention, "add_zero_attn", True), "add_zero_attn"),
        ],
    )
    def test_unwritten_attention_shapes_fail_loudly(self, mutate, message):
        head = StubMapHead().eval()
        mutate(head)

        with pytest.raises((TypeError, ValueError), match=message):
            patch_siglip2.apply_map_head_patch(head)
