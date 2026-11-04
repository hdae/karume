"""The contract of `wan/vae_patch.py` (real weights — ADR 0118 decision 2, the acceptance of
step 4).

Three levels are pinned:

- **The cache table** equals decision 2's table (32 entries, shapes, the position of `time_conv`),
  and first does not hold the two `time_conv` entries.
- **Cache normalization** (always 2 frames, initialized to zero) does not change the values — a
  chunk loop that keeps the upstream decoder code and normalizes only the cache is
  **bit-identical** to the upstream untiled `_decode`.
- **The rewrite components** are eager-equivalent — rewrites that only move data (the temporal
  interleave, nearest ×2, folding frames into (b t)) are bit-identical, and rewrites that change the
  reduction order (the `sum` form of RMS_norm, the spatial padding moved into attrs) are inside
  bands derived from measurements. The whole chunk loop (the final form) is inside its band too.

The latent is spatially 8×8 (64 px) with 5 chunks (the smallest count that goes through the first
chunk, the second chunk on the `'Rep'` path of `time_conv`, and the third and later chunks where
the cache holds 2 frames). The real-size values at tile 32 with 9 chunks are measured by
`python -m wan.export_vae --verify` (the VAE section of the README).
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from wan import vae_patch

#: A small latent tile (spatial) and the chunk count.
TILE = 8
CHUNKS = 5

#: The band of the difference between the final form of the rewrite (the whole chunk loop) and the
#: upstream (ratio to the reference's maximum absolute value). Measured (real weights f32,
#: 2026-10-02): 1.95e-6 on this test's latent (tile 8, 5 chunks), 2.79e-6 at tile 32 with 9 chunks
#: (`--verify`). About 5 times the latter. The difference comes from the reduced form of RMS_norm
#: and the conv padding moved into attrs (the component tests below), and neither grows as chunks
#: stack up (the last chunk is 1.36e-6).
FINAL_RATIO_BAND = 1.5e-5

#: The difference (absolute) between the `sum` form of RMS_norm and the upstream `F.normalize`
#: (`linalg_vector_norm`). About 5 times the measured 1.19e-6 (384 channels, 4 frames, 8×8,
#: input N(0, 3²)).
RMS_NORM_BAND = 6e-6

#: The difference between the spatial padding folded into conv3d's attrs and the upstream (an
#: explicit `F.pad`) (ratio to the output's maximum absolute value). torch CPU picks a different
#: convolution path depending on how the padding is held and on the shape, so bit identity is not
#: expected — on this test's input (8×8) T = 1 is bit-identical and T = 4 is 6.67e-6, while in a
#: 16×16 trial it was the other way around, only T = 1 differed (2026-10-02). The band is about 5
#: times 6.67e-6.
CONV_PADDING_RATIO_BAND = 3.5e-5


@pytest.fixture(scope="module")
def vae(wan_snapshot):
    """The VAE of the pinned revision (CPU f32, no rounding — the equivalence claim does not depend
    on the weight values)."""
    pytest.importorskip("diffusers")
    from diffusers import AutoencoderKLWan

    return AutoencoderKLWan.from_pretrained(
        wan_snapshot, subfolder="vae", torch_dtype=torch.float32
    ).eval()


@pytest.fixture(scope="module")
def latents(vae) -> torch.Tensor:
    """A fixed-seed random latent `[1, 16, 5, 8, 8]`, denormalized (the same value range as in real
    use)."""
    generator = torch.Generator().manual_seed(7)
    mean = torch.tensor(vae.config.latents_mean).view(1, 16, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std).view(1, 16, 1, 1, 1)
    return torch.randn(1, 16, CHUNKS, TILE, TILE, generator=generator) * std + mean


@pytest.fixture(scope="module")
def reference(vae, latents) -> torch.Tensor:
    with torch.no_grad():
        return vae_patch.reference_decode_unclamped(vae, latents)


def _ratio(got: torch.Tensor, want: torch.Tensor) -> float:
    return float((got - want).abs().max() / want.abs().max())


class TestCacheTable:
    #: Decision 2's table (at latent tile 32). (key, shape, time_conv).
    ADR_TABLE = (
        [("00", (16, 2, 32, 32), False)]
        + [(f"{index:02d}", (384, 2, 32, 32), False) for index in range(1, 11)]
        + [("11", (384, 2, 32, 32), True), ("12", (192, 2, 64, 64), False)]
        + [(f"{index:02d}", (384, 2, 64, 64), False) for index in range(13, 18)]
        + [("18", (384, 2, 64, 64), True)]
        + [(f"{index:02d}", (192, 2, 128, 128), False) for index in range(19, 25)]
        + [(f"{index:02d}", (96, 2, 256, 256), False) for index in range(25, 32)]
    )

    def test_slots_follow_the_forward_order_of_the_decoder(self, vae):
        slots = vae_patch.cache_slots(vae)

        assert [(slot.key, slot.shape(32), slot.time_conv) for slot in slots] == self.ADR_TABLE

    def test_the_whole_cache_is_the_element_count_of_the_adr(self, vae):
        slots = vae_patch.cache_slots(vae)

        assert sum(torch.Size(slot.shape(32)).numel() for slot in slots) == 154_959_872

    def test_the_first_graph_drops_only_the_two_time_convs(self, vae):
        slots = vae_patch.cache_slots(vae)

        first = vae_patch.graph_slots(slots, first=True)

        assert len(first) == 30
        assert sorted({slot.key for slot in slots} - {slot.key for slot in first}) == ["11", "18"]


class TestCacheNormalization:
    def test_the_reference_is_the_upstream_decode_before_its_clamp(self, vae, latents, reference):
        with torch.no_grad():
            decoded = vae._decode(latents, return_dict=False)[0]

        assert torch.equal(reference.clamp(-1.0, 1.0), decoded)

    def test_two_zero_frames_reproduce_the_upstream_cache_bit_for_bit(
        self, vae, latents, reference
    ):
        """Keep the upstream code and only set the cache's initial value to 2 zero frames (the
        measurement behind decision 2's derivation)."""
        with torch.no_grad():
            normalized = vae_patch.normalized_cache_decode(vae, latents, TILE)

        assert torch.equal(normalized, reference)


@pytest.fixture(scope="module")
def decoded(vae, latents) -> torch.Tensor:
    """The chunk loop run on the final form of the rewrite (eager of the two chunk graphs)."""
    first = vae_patch.WanVaeChunkDecoder(vae, first=True).eval()
    following = vae_patch.WanVaeChunkDecoder(vae, first=False).eval()
    with torch.no_grad():
        return vae_patch.chunk_decode(first, following, latents[0])


class TestChunkDecode:
    def test_the_rewritten_chunk_loop_stays_within_the_band(self, decoded, reference):
        assert decoded.shape == reference.shape[1:] == (3, 1 + 4 * (CHUNKS - 1), 64, 64)
        assert _ratio(decoded, reference[0]) <= FINAL_RATIO_BAND

    def test_the_error_does_not_grow_with_the_chunks(self, decoded, reference):
        """A cache mix-up accumulates difference in later chunks — the last chunk also stays within
        the first band."""
        scale = reference.abs().max()
        last = (decoded[:, -4:] - reference[0, :, -4:]).abs().max() / scale

        assert float(last) <= FINAL_RATIO_BAND

    def test_a_cache_with_its_two_frames_swapped_is_far_outside_the_band(
        self, vae, latents, reference, monkeypatch: pytest.MonkeyPatch
    ):
        """Fault injection: reversing the order of the cache's 2 frames lands far outside the band
        (the band can tell the order apart)."""
        first = vae_patch.WanVaeChunkDecoder(vae, first=True).eval()
        following = vae_patch.WanVaeChunkDecoder(vae, first=False).eval()
        original = vae_patch.causal_conv3d

        def swapped(conv: nn.Module, x: torch.Tensor, cache: torch.Tensor):
            return original(conv, x, cache.flip(1))

        monkeypatch.setattr(vae_patch, "causal_conv3d", swapped)
        with torch.no_grad():
            broken = vae_patch.chunk_decode(first, following, latents[0])

        assert _ratio(broken, reference[0]) > 1e3 * FINAL_RATIO_BAND


class TestDataMovementIsBitExact:
    def test_the_time_interleave_matches_the_rank6_upstream(self):
        x = torch.randn(2 * 6, 3, 4, 5)
        upstream = x.unsqueeze(0).reshape(1, 2, 6, 3, 4, 5)
        upstream = torch.stack((upstream[:, 0], upstream[:, 1]), 3).reshape(1, 6, 6, 4, 5)

        assert torch.equal(vae_patch.interleave_frames(x), upstream[0])

    def test_nearest_exact_2x_is_the_upstream_upsample(self, vae):
        upsample = vae.decoder.up_blocks[0].upsamplers[0].resample[0]
        x = torch.randn(4, 24, 5, 7)

        assert torch.equal(vae_patch.nearest_exact_2x(upsample, x), upsample(x))

    def test_the_post_quant_pointwise_conv_is_the_upstream_one(self, vae):
        x = torch.randn(16, 1, TILE, TILE)

        with torch.no_grad():
            want = vae.post_quant_conv(x.unsqueeze(0))[0]

        assert torch.equal(vae_patch.pointwise_conv3d(vae.post_quant_conv, x), want)

    def test_the_2d_upsample_folds_frames_into_the_batch(self, vae):
        """upsample2d (up2) has no time_conv — only the form that folds frames into (b t) is
        checked."""
        module = vae.decoder.up_blocks[2].upsamplers[0]
        x = torch.randn(192, 4, TILE, TILE)

        with torch.no_grad():
            got = vae_patch.resample(module, x, cursor=None)  # type: ignore[arg-type]
            want = module(x.unsqueeze(0))[0]

        assert torch.equal(got, want)

    def test_the_attention_fold_is_exact_once_the_norm_is_the_upstream_one(
        self, vae, monkeypatch: pytest.MonkeyPatch
    ):
        """With only RMS_norm put back to the upstream one, the form that folds into (b t) is
        bit-identical (isolating where the difference comes from)."""
        block = vae.decoder.mid_block.attentions[0]
        x = torch.randn(384, 1, TILE, TILE)

        def upstream_norm(norm: nn.Module, frames: torch.Tensor, dim: int) -> torch.Tensor:
            return norm(frames)

        monkeypatch.setattr(vae_patch, "rms_norm", upstream_norm)

        with torch.no_grad():
            got = vae_patch.attention_block(block, x)
            want = block(x.unsqueeze(0))[0]

        assert torch.equal(got, want)


class TestReductionOrderStaysInTheBand:
    def test_rms_norm_over_the_channel_axis(self, vae):
        norm = vae.decoder.mid_block.resnets[0].norm1
        x = torch.randn(384, 4, TILE, TILE, generator=torch.Generator().manual_seed(2)) * 3

        with torch.no_grad():
            difference = (vae_patch.rms_norm(norm, x, dim=0) - norm(x.unsqueeze(0))[0]).abs()

        assert float(difference.max()) <= RMS_NORM_BAND

    @pytest.mark.parametrize("frames", [1, 4])
    def test_the_cached_conv_with_its_spatial_padding_in_the_attrs(self, vae, frames: int):
        conv = vae.decoder.mid_block.resnets[0].conv1
        generator = torch.Generator().manual_seed(frames)
        x = torch.randn(384, frames, TILE, TILE, generator=generator)
        cache = torch.randn(384, 2, TILE, TILE, generator=generator)

        with torch.no_grad():
            got, updated = vae_patch.causal_conv3d(conv, x, cache)
            want = conv(x.unsqueeze(0), cache.unsqueeze(0))[0]

        assert _ratio(got, want) <= CONV_PADDING_RATIO_BAND
        assert torch.equal(updated, torch.cat([cache, x], dim=1)[:, -2:])


class TestFailsLoudly:
    def test_a_half_precision_activation_is_refused_by_the_norm(self, vae):
        norm = vae.decoder.norm_out

        with pytest.raises(vae_patch.UnsupportedVaeError):
            vae_patch.rms_norm(norm, torch.zeros(96, 1, 2, 2, dtype=torch.float16), dim=0)

    def test_caches_in_another_order_are_refused(self, vae):
        module = vae_patch.WanVaeChunkDecoder(vae, first=False)
        cache = vae_patch.zero_caches(module.slots, TILE, first=False)
        reordered = dict(reversed(list(cache.items())))

        with pytest.raises(ValueError, match="the order is checked"):
            module(torch.zeros(16, 1, TILE, TILE), reordered)

    def test_a_cache_of_the_wrong_shape_is_refused(self, vae):
        conv = vae.decoder.conv_in

        with pytest.raises(ValueError, match="cache shape"):
            vae_patch.causal_conv3d(conv, torch.zeros(16, 1, 4, 4), torch.zeros(16, 1, 4, 4))

    def test_a_batched_latent_is_refused(self, vae):
        module = vae_patch.WanVaeChunkDecoder(vae, first=True)

        with pytest.raises(ValueError, match="unbatched"):
            module(torch.zeros(1, 16, 1, TILE, TILE), {})
