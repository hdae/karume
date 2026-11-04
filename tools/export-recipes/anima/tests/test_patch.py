"""Regression tests that each rewrite of anima.patch is equivalent to the original implementation
(the part that needs no real weights).

End-to-end equivalence on the real model is checked by `anima/export.py --verify` (manual, real
weights). Here only the equivalences at the level of pure functions / small modules that run
whenever diffusers is present are pinned. A pytest run without `--group anima` skips them.

MUST: rather than "the difference is small", **grade the strength of the claim per rewrite**.
Rewrites that only move data (upsample) are bit-identical; rewrites that change the order of
operations (conv3d → conv2d / channel L2) are checked down to the error dropping by orders of
magnitude in f64 (= it is rounding, not a difference in meaning).
"""

from __future__ import annotations

import pytest
import torch
from torch import nn
from torch.nn import functional

qwenimage = pytest.importorskip("diffusers.models.autoencoders.autoencoder_kl_qwenimage")

from _shared.vae_rank4 import l2_normalize  # noqa: E402
from anima import patch as patch_anima  # noqa: E402


class TestCausalConv3dToConv2d:
    @pytest.mark.parametrize(
        ("kernel", "padding"),
        [((3, 3, 3), (1, 1, 1)), ((1, 1, 1), (0, 0, 0)), ((3, 1, 1), (1, 0, 0))],
    )
    def test_t1_causal_conv3d_matches_the_flattened_conv2d(self, kernel, padding) -> None:
        """A causal conv3d with T=1 equals "a conv2d with the last temporal slice of the weight".

        It is measured in f64 to tell whether the residual is "a rounding difference of the
        reduction order" or "a real mismatch" (the former drops by orders of magnitude in f64).
        """
        torch.manual_seed(0)
        conv = qwenimage.QwenImageCausalConv3d(3, 4, kernel, padding=padding).to(torch.float64)
        x = torch.randn(1, 3, 5, 5, dtype=torch.float64)

        with torch.no_grad():
            reference = conv(x.unsqueeze(2))
            got = patch_anima._causal_conv3d_to_conv2d(conv)(x)

        assert reference.shape[2] == 1
        assert float((got - reference.squeeze(2)).abs().max()) < 1e-14

    def test_a_kernel_taller_than_the_causal_padding_is_rejected(self) -> None:
        """A shape whose temporal causal padding is too small (downsample3d's time_conv) fails."""
        conv = qwenimage.QwenImageCausalConv3d(3, 3, (3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0))

        with pytest.raises(
            NotImplementedError, match="causal padding is outside the equivalence condition"
        ):
            patch_anima._causal_conv3d_to_conv2d(conv)

    def test_padding_beyond_the_causal_amount_is_rejected(self) -> None:
        """Excess padding fails too — the window also covers the leading zeros and the output
        has two frames.

        A one-sided check that looks only for "too little" lets this shape through and silently
        takes only the first frame of a length-2 output as the conv2d output (`2·pad_t=2, kt=2`
        gives an output length of 2).
        """
        conv = qwenimage.QwenImageCausalConv3d(3, 3, (2, 1, 1), padding=(1, 0, 0))
        with torch.no_grad():
            reference = conv(torch.randn(1, 3, 1, 4, 4))
        assert reference.shape[2] == 2, (
            "premise: a shape outside the equivalence condition does not have length 1"
        )

        with pytest.raises(
            NotImplementedError, match="causal padding is outside the equivalence condition"
        ):
            patch_anima._causal_conv3d_to_conv2d(conv)

    def test_a_time_stride_other_than_one_is_rejected(self) -> None:
        """For a shape that subsamples in time, the "last slice of the weight" equivalence is a
        different claim."""
        conv = qwenimage.QwenImageCausalConv3d(3, 3, (3, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0))

        with pytest.raises(NotImplementedError, match="with time stride"):
            patch_anima._causal_conv3d_to_conv2d(conv)


class TestUpsample:
    def test_the_expand_form_matches_nearest_exact_bit_for_bit(self) -> None:
        """×2 upsampling is only data movement, so it is **bit-identical** (no room for
        rounding)."""
        torch.manual_seed(0)
        module = qwenimage.QwenImageUpsample(scale_factor=(2.0, 2.0), mode="nearest-exact")
        x = torch.randn(1, 3, 4, 5)
        reference = functional.interpolate(x, scale_factor=2.0, mode="nearest-exact")

        got = patch_anima._upsample_forward(module, x)

        assert torch.equal(got, reference)

    def test_a_non_integer_scale_is_rejected(self) -> None:
        """For a non-integer nearest-exact factor the index mapping does not match duplication (no
        silent approximation)."""
        module = qwenimage.QwenImageUpsample(scale_factor=(1.5, 1.5), mode="nearest-exact")

        with pytest.raises(NotImplementedError, match="other than ×2"):
            patch_anima._upsample_forward(module, torch.randn(1, 3, 4, 4))

    def test_a_scalar_scale_factor_is_accepted(self) -> None:
        """A scalar `scale_factor` gives the same duplication (the notation is absorbed here)."""
        torch.manual_seed(0)
        module = qwenimage.QwenImageUpsample(scale_factor=2, mode="nearest-exact")
        x = torch.randn(1, 3, 4, 5)

        got = patch_anima._upsample_forward(module, x)

        assert torch.equal(got, functional.interpolate(x, scale_factor=2.0, mode="nearest-exact"))

    def test_another_mode_is_rejected(self) -> None:
        """`nearest`, a different mapping, is not let through (the same ×2, but the output index
        maps differently)."""
        module = qwenimage.QwenImageUpsample(scale_factor=(2.0, 2.0), mode="nearest")

        with pytest.raises(NotImplementedError, match="other than nearest-exact"):
            patch_anima._upsample_forward(module, torch.randn(1, 3, 4, 4))


class TestChannelL2:
    def test_it_matches_f_normalize(self) -> None:
        """Channel-wise L2 expressed as permute + a sum over the last dimension still matches
        `F.normalize`.

        The `clamp_min(eps) → divide` order is the original implementation's too (not replaced by
        `+eps`).
        """
        torch.manual_seed(0)
        x = torch.randn(1, 6, 4, 5, dtype=torch.float64)

        got = l2_normalize(x, dim=1)

        assert float((got - functional.normalize(x, dim=1)).abs().max()) < 1e-14

    def test_a_zero_column_keeps_the_epsilon_floor(self) -> None:
        """Fault injection confirms that a zero input does not become 0/0 (`clamp_min` is in
        effect)."""
        zeros = torch.zeros(1, 3, 2, 2)

        got = l2_normalize(zeros, dim=1)

        assert torch.isfinite(got).all()
        assert torch.equal(got, zeros)


class TestRmsNormForward:
    def test_the_rank4_form_matches_the_rank5_original(self) -> None:
        """The rank-4 version with gamma reshaped to (1,C,1,1) matches the original (rank-5
        input)."""
        torch.manual_seed(0)
        module = qwenimage.QwenImageRMS_norm(6, images=False).to(torch.float64)
        with torch.no_grad():
            module.gamma.copy_(torch.rand_like(module.gamma) + 0.5)
        x = torch.randn(1, 6, 4, 5, dtype=torch.float64)
        with torch.no_grad():
            reference = module(x.unsqueeze(2)).squeeze(2)

        module.gamma = nn.Parameter(module.gamma.detach().reshape(1, -1, 1, 1))
        with torch.no_grad():
            got = patch_anima._rms_norm_forward(module, x)

        assert float((got - reference).abs().max()) < 1e-14

    def test_a_learned_bias_is_rejected(self) -> None:
        """The original's `+ self.bias` can be dropped only when bias is Python's 0.0."""
        module = qwenimage.QwenImageRMS_norm(4, bias=True)
        module.gamma = nn.Parameter(module.gamma.detach().reshape(1, -1, 1, 1))

        with pytest.raises(NotImplementedError, match="with a learned bias"):
            patch_anima._rms_norm_forward(module, torch.randn(1, 4, 2, 2))


class TestResample:
    def test_a_feat_cache_is_rejected(self) -> None:
        """`feat_cache` is never read with T=1 — when one is passed it fails rather than being
        silently dropped."""
        module = qwenimage.QwenImageResample(4, mode="none")

        with pytest.raises(NotImplementedError, match="feat_cache"):
            patch_anima._resample_forward(module, torch.randn(1, 4, 2, 2), feat_cache=[None])


# ---- #21 iteration T2: the DiT over S (token form) ---------------------------

#: The synthetic DiT's shape. **The patch height / width and the channel count all differ** — an
#: axis mix-up is the class of error that "becomes an involution on the measured shapes and
#: matches by accident" (ACTIVE_DESIGN's Pitfalls), so it is only caught by shapes that make it a
#: permutation with a cycle of length 3 or more.
TINY_DIT = {
    "in_channels": 3,
    "out_channels": 3,
    "num_attention_heads": 2,
    "attention_head_dim": 12,
    "num_layers": 1,
    "mlp_ratio": 1.0,
    "text_embed_dim": 8,
    "adaln_lora_dim": 4,
    "max_size": (4, 16, 12),
    "patch_size": (1, 2, 3),
    "rope_scale": (1.0, 4.0, 4.0),
    "concat_padding_mask": True,
    "extra_pos_embed_type": None,
}
#: The latent shape (H / W differ too — with a square an h/w mix-up hides as a transpose).
TINY_LATENT = (8, 12)


def _tiny_dit():
    from diffusers import CosmosTransformer3DModel

    torch.manual_seed(20260804)
    model = CosmosTransformer3DModel(**TINY_DIT)
    return model.to(torch.float32).eval()


def _tiny_inputs(model):
    torch.manual_seed(7)
    height, width = TINY_LATENT
    latents = torch.randn(1, model.config.in_channels, height, width)
    timestep = torch.full((1,), 0.37)
    embeds = torch.randn(1, 5, model.config.text_embed_dim)
    return latents, timestep, embeds


class TestDitTokens:
    """The S-form wrapper = the static wrapper with only its entry / exit moved to the host (#21
    iteration T2)."""

    def test_the_token_form_composes_back_to_the_static_wrapper_bit_exactly(self) -> None:
        """`unpatchify ∘ S form ∘ patchify` is **bit-identical** to the static wrapper.

        Both go through the same operations in the same order (the only difference is "which side
        builds patchify / rope"), so not even a rounding difference is expected. If this loosens,
        the main check on the real GPU (the PNG sha256 in `packages/e2e/tests/e2e_anima_test.ts`)
        loosens too.
        """
        model = _tiny_dit()
        latents, timestep, embeds = _tiny_inputs(model)
        height, width = TINY_LATENT
        patch = tuple(model.config.patch_size)
        proj = patch_anima.dit_timesteps_proj(model, timestep)

        with torch.no_grad():
            expected = patch_anima.AnimaDit(model, height, width)(latents, proj, embeds)
            tokens = patch_anima.AnimaDitTokens(model)(
                patch_anima.dit_patchify(latents, patch),
                proj,
                embeds,
                *patch_anima.dit_rope_tables(model, height, width),
            )
            got = patch_anima.dit_unpatchify(tokens, height, width, patch)

        assert got.shape == expected.shape
        assert torch.equal(got, expected)

    def test_patchify_matches_the_upstream_patch_embed_chain(self) -> None:
        """patchify's reordering matches the stage before proj in upstream's `CosmosPatchEmbed`
        (an independent oracle)."""
        from diffusers.models.transformers.transformer_cosmos import CosmosPatchEmbed

        model = _tiny_dit()
        latents, _, _ = _tiny_inputs(model)
        height, width = TINY_LATENT
        patch = tuple(model.config.patch_size)
        embed = CosmosPatchEmbed(1, 1, patch)
        embed.proj = nn.Identity()
        padded = torch.cat([latents, latents.new_zeros(1, 1, height, width)], dim=1).unsqueeze(2)

        with torch.no_grad():
            reference = embed(padded).flatten(1, 3)
            got = patch_anima.dit_patchify(latents, patch)

        assert torch.equal(got, reference)

    def test_unpatchify_matches_an_index_level_oracle(self) -> None:
        """State unpatchify's reordering independently **as an index formula** (not a copy of the
        view chain).

        The last dimension is ordered `(ph, pw, pt, C)` (**different** from patchify's
        `(C, pt, ph, pw)`), and the token index is `(f·H' + h)·W' + w`. An implementation "fixed"
        to the reverse order goes red here.
        """
        height, width = TINY_LATENT
        patch_t, patch_h, patch_w = (1, 2, 3)
        channels = 3
        rows, cols = height // patch_h, width // patch_w
        tokens = torch.arange(float(rows * cols * patch_t * patch_h * patch_w * channels)).reshape(
            1, rows * cols, patch_h * patch_w * patch_t * channels
        )

        got = patch_anima.dit_unpatchify(tokens, height, width, (patch_t, patch_h, patch_w))

        assert tuple(got.shape) == (1, channels, height, width)
        for channel in range(channels):
            for row in range(rows):
                for col in range(cols):
                    for inner_h in range(patch_h):
                        for inner_w in range(patch_w):
                            at = ((inner_h * patch_w + inner_w) * patch_t) * channels + channel
                            assert (
                                got[0, channel, row * patch_h + inner_h, col * patch_w + inner_w]
                                == (tokens[0, row * cols + col, at])
                            )

    def test_the_host_base_tables_rebuild_the_full_rope_table_bit_exactly(self) -> None:
        """Assembling from the per-axis base tables is **bit-identical** to `model.rope`'s table
        (the source of truth of the host implementation).

        torch's f32 `cos` / `sin` can be 1 ulp off correct rounding, which JS's `Math.cos` cannot
        reproduce. That is why the base tables are baked (the doc of `dit_rope_base_tables`).
        """
        model = _tiny_dit()
        height, width = TINY_LATENT
        patch = tuple(model.config.patch_size)
        base = patch_anima.dit_rope_base_tables(model)
        rows, cols = height // patch[1], width // patch[2]

        expected_cos, expected_sin = patch_anima.dit_rope_tables(model, height, width)
        built_cos, built_sin = [], []
        for row in range(rows):
            for col in range(cols):
                built_cos.append(
                    torch.cat([base["cos_t"][0], base["cos_h"][row], base["cos_w"][col]] * 2)
                )
                built_sin.append(
                    torch.cat([base["sin_t"][0], base["sin_h"][row], base["sin_w"][col]] * 2)
                )

        assert torch.equal(torch.stack(built_cos), expected_cos[0, 0])
        assert torch.equal(torch.stack(built_sin), expected_sin[0, 0])

    def test_the_base_tables_cover_the_models_own_position_ceiling(self) -> None:
        """The base tables have as many rows as upstream's `seq = arange(max(max_size))` (= the
        ceiling of the position table)."""
        model = _tiny_dit()
        base = patch_anima.dit_rope_base_tables(model)
        rows = max(model.rope.max_size)

        assert set(base) == set(patch_anima.ROPE_BASE_KEYS)
        for name, table in base.items():
            assert table.shape[0] == rows, name
        assert base["cos_t"].shape[1] == model.rope.dim_t // 2
        assert base["cos_h"].shape[1] == model.rope.dim_h // 2

    def test_a_misaligned_base_slice_is_rejected(self) -> None:
        """Vacuity check: a slice with shifted block boundaries "does not vary along that axis",
        so it fails."""
        model = _tiny_dit()
        base = patch_anima.dit_rope_base_tables(model)
        broken = dict(base)
        broken["sin_h"] = torch.zeros_like(base["sin_h"])

        with pytest.raises(AssertionError, match="does not vary along this axis"):
            patch_anima._assert_rope_base_shapes(
                broken,
                max(model.rope.max_size),
                (model.rope.dim_t // 2, model.rope.dim_h // 2, model.rope.dim_w // 2),
            )

    def test_the_token_graph_carries_no_resolution_dependent_initializer(self) -> None:
        """The S-form IR has **not one resolution-dependent baked constant** (the very design
        premise of iteration T2).

        The static form baked only three (the padding channel's `[1,1,H,W]` and the two rope
        tables' `[1,1,S,D]`), all rank 4. The weights are only linear / norm, so they stay within
        rank ≤ 2. Hence "zero initializers of rank ≥ 3" is itself the invariant, and it is
        guaranteed to break if an S-dependent constant gets folded in (it would be baked at the
        `Dim` upper bound = 16,384).
        """
        from torch.export import Dim

        from karume.convert import PRESERVED_OP_PREFIXES_WITH_ATTENTION
        from karume.pipeline import export_module

        model = _tiny_dit()
        latents, timestep, embeds = _tiny_inputs(model)
        height, width = TINY_LATENT
        patch = tuple(model.config.patch_size)
        tokens = Dim("S", min=2, max=64)
        graph, _ = export_module(
            patch_anima.AnimaDitTokens(model),
            (
                patch_anima.dit_patchify(latents, patch),
                patch_anima.dit_timesteps_proj(model, timestep),
                embeds,
                *patch_anima.dit_rope_tables(model, height, width),
            ),
            dynamic_shapes=({1: tokens}, None, None, {2: tokens}, {2: tokens}),
            symbol_names=("S",),
            preserved=PRESERVED_OP_PREFIXES_WITH_ATTENTION,
        )

        assert graph.symbols == ["S"]
        assert [entry.name for entry in graph.inputs] == [
            "tokens",
            "timesteps_proj",
            "encoder_hidden_states",
            "rope_cos",
            "rope_sin",
        ]
        assert graph.inputs[0].shape == [1, "S", 24]
        assert graph.inputs[3].shape == [1, 1, "S", 12]
        baked = {
            name: graph.values[name].shape
            for name in graph.initializers
            if len(graph.values[name].shape) >= 3
        }
        assert baked == {}, f"resolution-dependent baked constants remain: {baked}"


class TestTextEncoderExport:
    """`AnimaTextEncoder` (the form that passes no mask) exports in S form on a tiny Qwen3."""

    def test_the_text_encoder_exports_without_the_packed_sequence_mask(self) -> None:
        """Without pinning `find_packed_sequence_indices`, the detection during tracing leaves an
        `index` that looks up `cumsum` by position and an `eq` mask, and the exporter fails on an
        op outside the vocabulary (measured 2026-10-17 — since the ADR 0010 addendum of
        2026-10-10, after which the exporter no longer folds an index that looks up a symbolic
        axis by position).
        """
        transformers = pytest.importorskip("transformers")
        from torch.export import Dim

        from karume.pipeline import export_module

        config = transformers.Qwen3Config(
            vocab_size=64,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            max_position_embeddings=64,
            use_cache=False,
        )
        model = transformers.Qwen3Model(config).eval()
        patch_anima.disable_packed_sequence_detection()
        ids = torch.randint(0, 64, (1, 8), generator=torch.Generator().manual_seed(0))
        graph, _ = export_module(
            patch_anima.AnimaTextEncoder(model),
            (ids,),
            dynamic_shapes=({1: Dim("T", min=2, max=64)},),
        )

        assert graph.symbols == ["T"]
        assert [entry.name for entry in graph.inputs] == ["input_ids"]
