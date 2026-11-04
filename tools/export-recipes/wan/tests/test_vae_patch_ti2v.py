"""The contract of the Wan2.2 branch of `wan/vae_patch.py` (ADR 0121 decision 6, the acceptance of
step 4).

Pinned on two levels:

- **The synthetic 2.2 VAE** (always runs — conftest's `ti2v_synthetic_vae`; the upstream
  `AutoencoderKLWan` randomly initialized with the 5B's structure — residual up blocks, patch 2,
  three shortcuts — at small channel counts): agreement of the shortcuts (the first chunk's slice
  included) with the upstream, the cache table, bit identity of cache normalization, the final
  form's band, fault injection, the generation checks.
- **Real weights** (SKIP on a machine without the pinned 5B snapshot — the same discipline as the
  2.1 real-weights tests): the cache table at tile 16 (the architecture table) and the element
  counts, bit identity of the normalization and the final form's band at tile 8 with 5 chunks.

5 chunks is the count that goes through the first chunk, the second chunk on the `'Rep'` path of
`time_conv`, the third chunk where the regular cache holds 2 frames, and the fourth chunk where the
cache used by the first `time_conv` (which takes 1 latent frame) has both frames nonzero, and that
includes one chunk after every cache is filled (the smallest that goes through all of them is 4).
The real-size values (tile 16, 9 chunks) are measured by
`python -m wan.export_vae --model ti2v-5b --verify --chunks 9`.

The real-weights tests read the whole VAE (f32 including the encoder, about 2.82 GB). Measured on
2026-10-05 (sharing the machine with other heavy CPU work), the RSS peak is about 3.5 GiB and the
real-weights part of this file takes about 2 minutes. When RAM runs short alongside long CPU work,
leave those tests out with `-k 'not RealWeights'` and run them later.
"""

from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from wan import vae_patch

#: The synthetic VAE's latent tile (spatial), the real weights' latent tile, the chunk count.
SYNTHETIC_TILE = 4
REAL_TILE = 8
CHUNKS = 5

#: Wan2.2's patchify factor (`patch_size` of the upstream config).
PATCH_SIZE = 2

#: The band of the difference between the final form of the rewrite (the whole chunk loop) and the
#: upstream (ratio to the reference's maximum absolute value). Both are 5 times the measurement of
#: the decision case (this test's latent — seed 7, 5 chunks), rounded up to two significant digits.
#: The difference comes from the same reduced form of RMS_norm and the conv padding moved into attrs
#: as in 2.1 (`test_vae_patch.py`); the shortcuts only move data.
#: - Synthetic (tile 4): measured 1.05e-6 (the last chunk is 9.09e-7).
#: - Real weights f32 (tile 8, 2026-10-05): measured 5.31e-6 (the last chunk is 4.20e-6). It is
#:   larger than 2.1 at the same size (1.95e-6) because there are 2.7 to 4 times as many channels to
#:   reduce (a guess).
SYNTHETIC_FINAL_RATIO_BAND = 5.3e-6
REAL_FINAL_RATIO_BAND = 2.7e-5


def _ratio(got: torch.Tensor, want: torch.Tensor) -> float:
    return float((got - want).abs().max() / want.abs().max())


def _latents(vae, tile: int) -> torch.Tensor:
    """A fixed-seed random latent `[1, 48, 5, t, t]`, denormalized (the same value range as in real
    use)."""
    generator = torch.Generator().manual_seed(7)
    mean = torch.tensor(vae.config.latents_mean).view(1, 48, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std).view(1, 48, 1, 1, 1)
    return torch.randn(1, 48, CHUNKS, tile, tile, generator=generator) * std + mean


def _chunk_decode(vae, latents: torch.Tensor) -> torch.Tensor:
    first = vae_patch.WanVaeChunkDecoder(vae, first=True).eval()
    following = vae_patch.WanVaeChunkDecoder(vae, first=False).eval()
    with torch.no_grad():
        return vae_patch.chunk_decode(first, following, latents[0])


def _upstream_decode(vae, latents: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return vae._decode(latents, return_dict=False)[0]


def _unpatchify(frames: torch.Tensor) -> torch.Tensor:
    """The upstream `unpatchify` itself (pinning against a copy of our own would make the claim
    about the reference's origin tautological)."""
    from diffusers.models.autoencoders.autoencoder_kl_wan import unpatchify

    return unpatchify(frames, patch_size=PATCH_SIZE)


class TestAvgShortcut:
    """The wan side of the shortcut (`avg_shortcut` — it holds the first chunk's slice) against the
    upstream `DupUp3D`.

    `importorskip` is taken inside each test (it does not go through a fixture, so under the default
    sync without diffusers it becomes SKIP rather than FAIL — the same style as
    `test_vae_tiling.py`).
    """

    @pytest.mark.parametrize("first_chunk", [False, True])
    @pytest.mark.parametrize("frames", [1, 2])
    @pytest.mark.parametrize(
        ("channels", "out_channels", "factor_t"),
        [(8, 8, 2), (8, 4, 1), (1024, 1024, 2), (1024, 512, 1)],
    )
    def test_bit_exact_with_the_upstream_module(
        self, channels: int, out_channels: int, factor_t: int, frames: int, first_chunk: bool
    ):
        wan_vae = pytest.importorskip("diffusers.models.autoencoders.autoencoder_kl_wan")
        module = wan_vae.DupUp3D(channels, out_channels, factor_t=factor_t, factor_s=2)
        x = torch.randn(channels, frames, 3, 5, generator=torch.Generator().manual_seed(frames))

        got = vae_patch.avg_shortcut(module, x, first_chunk=first_chunk)

        assert torch.equal(got, module(x.unsqueeze(0), first_chunk=first_chunk)[0])

    def test_an_input_of_other_channels_is_refused(self):
        wan_vae = pytest.importorskip("diffusers.models.autoencoders.autoencoder_kl_wan")
        with pytest.raises(vae_patch.UnsupportedVaeError, match="input channels"):
            vae_patch.avg_shortcut(
                wan_vae.DupUp3D(8, 8, factor_t=2, factor_s=2),
                torch.zeros(4, 1, 2, 2),
                first_chunk=False,
            )


class TestCacheTableTi2v:
    #: The synthetic VAE's table (at latent tile 4). (key, shape, time_conv). The order is the same
    #: as the 5B's architecture table, with only the channels at 1/16 (the 5B has 48 / 1024 / 512 /
    #: 256).
    SYNTHETIC_TABLE = (
        [("00", (48, 2, 4, 4), False)]
        + [(f"{index:02d}", (64, 2, 4, 4), False) for index in range(1, 11)]
        + [("11", (64, 2, 4, 4), True)]
        + [(f"{index:02d}", (64, 2, 8, 8), False) for index in range(12, 18)]
        + [("18", (64, 2, 8, 8), True), ("19", (64, 2, 16, 16), False)]
        + [(f"{index:02d}", (32, 2, 16, 16), False) for index in range(20, 25)]
        + [("25", (32, 2, 32, 32), False)]
        + [(f"{index:02d}", (16, 2, 32, 32), False) for index in range(26, 32)]
    )

    def test_slots_follow_the_forward_order_of_the_decoder(self, ti2v_synthetic_vae):
        slots = vae_patch.cache_slots(ti2v_synthetic_vae)

        assert [(slot.key, slot.shape(SYNTHETIC_TILE), slot.time_conv) for slot in slots] == (
            self.SYNTHETIC_TABLE
        )

    def test_the_first_graph_drops_only_the_two_time_convs(self, ti2v_synthetic_vae):
        slots = vae_patch.cache_slots(ti2v_synthetic_vae)

        first = vae_patch.graph_slots(slots, first=True)

        assert len(first) == 30
        assert sorted({slot.key for slot in slots} - {slot.key for slot in first}) == ["11", "18"]


@pytest.fixture(scope="module")
def synthetic_latents(ti2v_synthetic_vae) -> torch.Tensor:
    return _latents(ti2v_synthetic_vae, SYNTHETIC_TILE)


@pytest.fixture(scope="module")
def synthetic_reference(ti2v_synthetic_vae, synthetic_latents) -> torch.Tensor:
    with torch.no_grad():
        return vae_patch.reference_decode_unclamped(ti2v_synthetic_vae, synthetic_latents)


class TestCacheNormalizationTi2v:
    def test_the_reference_is_the_upstream_decode_before_unpatchify_and_clamp(
        self, ti2v_synthetic_vae, synthetic_latents, synthetic_reference
    ):
        decoded = _upstream_decode(ti2v_synthetic_vae, synthetic_latents)

        assert synthetic_reference.shape == (1, 12, 1 + 4 * (CHUNKS - 1), 32, 32)
        assert torch.equal(_unpatchify(synthetic_reference).clamp(-1.0, 1.0), decoded)

    def test_two_zero_frames_reproduce_the_upstream_cache_bit_for_bit(
        self, ti2v_synthetic_vae, synthetic_latents, synthetic_reference
    ):
        with torch.no_grad():
            normalized = vae_patch.normalized_cache_decode(
                ti2v_synthetic_vae, synthetic_latents, SYNTHETIC_TILE
            )

        assert torch.equal(normalized, synthetic_reference)


class TestChunkDecodeTi2v:
    def test_the_rewritten_chunk_loop_stays_within_the_band(
        self, ti2v_synthetic_vae, synthetic_latents, synthetic_reference
    ):
        decoded = _chunk_decode(ti2v_synthetic_vae, synthetic_latents)

        assert decoded.shape == synthetic_reference.shape[1:]
        assert _ratio(decoded, synthetic_reference[0]) <= SYNTHETIC_FINAL_RATIO_BAND
        last = (decoded[:, -4:] - synthetic_reference[0, :, -4:]).abs().max()
        assert float(last / synthetic_reference.abs().max()) <= SYNTHETIC_FINAL_RATIO_BAND

    def test_a_cache_with_its_two_frames_swapped_is_outside_the_band(
        self, ti2v_synthetic_vae, synthetic_latents, synthetic_reference, monkeypatch
    ):
        original = vae_patch.causal_conv3d

        def swapped(conv: nn.Module, x: torch.Tensor, cache: torch.Tensor):
            return original(conv, x, cache.flip(1))

        monkeypatch.setattr(vae_patch, "causal_conv3d", swapped)
        broken = _chunk_decode(ti2v_synthetic_vae, synthetic_latents)

        assert _ratio(broken, synthetic_reference[0]) > SYNTHETIC_FINAL_RATIO_BAND

    def test_a_shortcut_with_its_h_and_w_digits_swapped_is_outside_the_band(
        self, ti2v_synthetic_vae, synthetic_latents, synthetic_reference, monkeypatch
    ):
        """Fault injection: in the shortcut of up2 (rep 2), swap the H and W digits of the channel
        index."""
        original = vae_patch.dup_up_3d

        def swapped(x: torch.Tensor, **factors: int) -> torch.Tensor:
            return original(x.transpose(2, 3), **factors).transpose(2, 3)

        monkeypatch.setattr(vae_patch, "dup_up_3d", swapped)
        broken = _chunk_decode(ti2v_synthetic_vae, synthetic_latents)

        assert _ratio(broken, synthetic_reference[0]) > SYNTHETIC_FINAL_RATIO_BAND


class TestTheFirstChunkSlice:
    """Fault injection: dropping the first chunk's slice from the shortcut makes the first graph
    fail on shapes.

    up0 gets through silently by the broadcast of `[C,1,…] + [C,2,…]`, but at up1 it becomes
    2 frames + 4 frames, which cannot be added — there is no shape where the values get through.
    It is checked through shape failures in eager and in export rather than through a GPU
    comparison (a graph with a piece missing cannot be exported).
    """

    @pytest.fixture
    def first_without_the_slice(self, ti2v_synthetic_vae, monkeypatch):
        original = vae_patch.avg_shortcut

        def ignores_first(module: nn.Module, x: torch.Tensor, *, first_chunk: bool):
            return original(module, x, first_chunk=False)

        monkeypatch.setattr(vae_patch, "avg_shortcut", ignores_first)
        module = vae_patch.WanVaeChunkDecoder(ti2v_synthetic_vae, first=True).eval()
        latent = torch.zeros(48, 1, SYNTHETIC_TILE, SYNTHETIC_TILE)
        caches = vae_patch.zero_caches(module.slots, SYNTHETIC_TILE, first=True)
        return module, (latent, caches)

    def test_eager_fails_on_the_shape(self, first_without_the_slice):
        module, inputs = first_without_the_slice

        with torch.no_grad(), pytest.raises(RuntimeError, match="must match"):
            module(*inputs)

    def test_export_fails_on_the_shape(self, first_without_the_slice):
        module, inputs = first_without_the_slice

        with torch.no_grad(), pytest.raises(RuntimeError, match="broadcast"):
            torch.export.export(module, inputs, strict=False)


class TestGenerationGate:
    def test_the_synthetic_ti2v_vae_is_accepted(self, ti2v_synthetic_vae):
        vae_patch.assert_supported(ti2v_synthetic_vae)

    @pytest.mark.parametrize(
        "overrides",
        [
            {"patch_size": None},
            {"is_residual": False},
            {"patch_size": 4, "out_channels": 48},
        ],
        ids=["residual-without-patchify", "patchify-without-residual", "patch-4"],
    )
    def test_a_generation_outside_the_table_is_refused(self, ti2v_synthetic_vae, overrides):
        from diffusers import AutoencoderKLWan

        vae = AutoencoderKLWan.from_config(ti2v_synthetic_vae.config, **overrides)

        with pytest.raises(vae_patch.UnsupportedVaeError, match="unsupported"):
            vae_patch.assert_supported(vae)

    def test_an_exit_that_is_not_the_patchify_space_is_refused(self, ti2v_synthetic_vae):
        from diffusers import AutoencoderKLWan

        vae = AutoencoderKLWan.from_config(ti2v_synthetic_vae.config, out_channels=3)

        with pytest.raises(vae_patch.UnsupportedVaeError, match="output channels"):
            vae_patch.assert_supported(vae)

    def test_a_block_whose_shortcut_and_upsampler_disagree_is_refused(self, ti2v_synthetic_vae):
        vae = copy.deepcopy(ti2v_synthetic_vae)
        vae.decoder.up_blocks[0].avg_shortcut = None

        with pytest.raises(vae_patch.UnsupportedVaeError, match="disagree on presence"):
            vae_patch.assert_supported(vae)

    def test_a_shortcut_with_another_time_factor_is_refused(self, ti2v_synthetic_vae):
        """up2's upsampler is upsample2d (time ×1) — with ×2 only in the shortcut, the 1 frame /
        4 frames break."""
        vae = copy.deepcopy(ti2v_synthetic_vae)
        vae.decoder.up_blocks[2].avg_shortcut.factor_t = 2

        with pytest.raises(vae_patch.UnsupportedVaeError, match="scale factors"):
            vae_patch.assert_supported(vae)

    def test_the_reference_also_refuses_an_unsupported_generation(self, ti2v_synthetic_vae):
        from diffusers import AutoencoderKLWan

        vae = AutoencoderKLWan.from_config(ti2v_synthetic_vae.config, patch_size=None)

        with pytest.raises(vae_patch.UnsupportedVaeError, match="unsupported"):
            vae_patch.reference_decode_unclamped(vae, torch.zeros(1, 48, 1, 2, 2))


# ---- Real weights (SKIP without the pinned 5B snapshot) ------------------------------------------


@pytest.fixture(scope="module")
def vae(wan22_snapshot):
    """The 5B VAE of the pinned revision (CPU f32, no rounding — the equivalence claim does not
    depend on the weight values)."""
    pytest.importorskip("diffusers")
    from diffusers import AutoencoderKLWan

    return AutoencoderKLWan.from_pretrained(
        wan22_snapshot, subfolder="vae", torch_dtype=torch.float32
    ).eval()


@pytest.fixture(scope="module")
def latents(vae) -> torch.Tensor:
    return _latents(vae, REAL_TILE)


@pytest.fixture(scope="module")
def reference(vae, latents) -> torch.Tensor:
    with torch.no_grad():
        return vae_patch.reference_decode_unclamped(vae, latents)


class TestCacheTableRealWeights:
    #: The architecture table (at latent tile 16 — the design table of ADR 0121 step 4).
    #: (key, shape, time_conv).
    TABLE = (
        [("00", (48, 2, 16, 16), False)]
        + [(f"{index:02d}", (1024, 2, 16, 16), False) for index in range(1, 11)]
        + [("11", (1024, 2, 16, 16), True)]
        + [(f"{index:02d}", (1024, 2, 32, 32), False) for index in range(12, 18)]
        + [("18", (1024, 2, 32, 32), True), ("19", (1024, 2, 64, 64), False)]
        + [(f"{index:02d}", (512, 2, 64, 64), False) for index in range(20, 25)]
        + [("25", (512, 2, 128, 128), False)]
        + [(f"{index:02d}", (256, 2, 128, 128), False) for index in range(26, 32)]
    )

    def test_slots_follow_the_forward_order_of_the_decoder(self, vae):
        slots = vae_patch.cache_slots(vae)

        assert [(slot.key, slot.shape(16), slot.time_conv) for slot in slots] == self.TABLE

    def test_the_whole_cache_is_the_element_count_of_the_design(self, vae):
        slots = vae_patch.cache_slots(vae)
        first = vae_patch.graph_slots(slots, first=True)

        assert sum(torch.Size(slot.shape(16)).numel() for slot in slots) == 116_940_800
        assert sum(torch.Size(slot.shape(16)).numel() for slot in first) == 114_319_360


class TestCacheNormalizationRealWeights:
    def test_the_reference_is_the_upstream_decode_before_unpatchify_and_clamp(
        self, vae, latents, reference
    ):
        decoded = _upstream_decode(vae, latents)

        assert reference.shape == (1, 12, 1 + 4 * (CHUNKS - 1), 64, 64)
        assert torch.equal(_unpatchify(reference).clamp(-1.0, 1.0), decoded)

    def test_two_zero_frames_reproduce_the_upstream_cache_bit_for_bit(
        self, vae, latents, reference
    ):
        with torch.no_grad():
            normalized = vae_patch.normalized_cache_decode(vae, latents, REAL_TILE)

        assert torch.equal(normalized, reference)


class TestChunkDecodeRealWeights:
    def test_the_rewritten_chunk_loop_stays_within_the_band(self, vae, latents, reference):
        decoded = _chunk_decode(vae, latents)

        assert decoded.shape == reference.shape[1:]
        assert _ratio(decoded, reference[0]) <= REAL_FINAL_RATIO_BAND
        last = (decoded[:, -4:] - reference[0, :, -4:]).abs().max()
        assert float(last / reference.abs().max()) <= REAL_FINAL_RATIO_BAND
