"""The contract of `wan/vae_encoder_patch.py` (ADR 0121 decision 11, the acceptance of step 9a).

Pinned on two levels:

- **The synthetic 2.2 VAE** (always runs — conftest's `ti2v_synthetic_vae`; the encoder has the
  same structure as the 5B's — four down blocks, shortcuts (1,2,4) / (2,2,4) / (2,2,4) / (1,1,1),
  a mid of Res → Attention → Res — with only the channels shrunk): per-component agreement with the
  upstream (the closed form of AvgDown3D is bit-identical at the 5B's real channel counts,
  ZeroPad2d and the down resample are bit-identical too, the fold onto the time slice is identical
  in f64), the mu band of the three chained graphs, fault injection, the configuration checks.
- **Real weights** (SKIP on a machine without the pinned 5B snapshot): the band between the mu of
  the three chained graphs and the mu of the upstream untiled encode (a 256×160 random image), and
  that the 1/std of the condition normalization is an f32 division.

Where it is not bit-identical and why (measured 2026-10-09 — each band is 5 times the measurement
of its decision case, rounded up to two significant digits):

- The fold onto the time slice (`LastSliceConv3d`): in f32 the reduction order differs from the
  upstream "conv3d with two zero frames packed in" (the CPU conv kernel takes different paths for
  time 3 and 1 — relative 5.6e-7). In f64 it is relative 8.7e-17, and the only terms that vanish
  are zero × weight (mathematically identical).
- The reduced form of the RMS norm (`l2_normalize` — the same as the decoder rewrite) and the
  sequence form that turns attention's 1×1 conv into linear also differ only in reduction order.
- The closed form of AvgDown3D, ZeroPad2d and the down resample are bit-identical to the upstream
  (the summation order is the same too).
"""

from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from wan import vae_encoder_patch

#: The size of the synthetic image (random in [-1, 1], before patchify — latent 4×6) and the seed.
SYNTHETIC_SIZE = (64, 96)
SYNTHETIC_SEED = 11

#: The size of the real-weights image (256×160, the same as the downscaling path — latent 10×16)
#: and the seed.
REAL_SIZE = (160, 256)
REAL_SEED = 20261009

#: The band of the difference between the mu of the three chained graphs and the upstream mu (ratio
#: to the reference's maximum absolute value).
#: - Synthetic (64×96, seed 11): measured 7.55e-7 × 5 → 3.8e-6.
#: - Real weights f32 (256×160, seed 20261009): measured 5.90e-7 × 5 → 3.0e-6 (the measurements on
#:   3 real images × 3 sizes are in the docstring of {@link TestEncodeChunk0RealWeights}).
#:
#: Measurement conditions (the values the bands derive from): 2026-10-09, Ryzen 5 7600,
#: torch 2.13.0+cpu (CPU capability AVX512), torch's default 6 threads. The difference comes from
#: the reduction order (the folded conv3d against the upstream time-3 conv3d, the reduced form of
#: the RMS norm), and how the CPU conv path is split can change with the thread count, so on the
#: same day it was measured again with `torch.set_num_threads` at 1 / 3 / 6 / 12: synthetic
#: 7.34e-7 to 7.55e-7, real weights 4.99e-7 to 6.35e-7 (worst with 1 thread), attention 6.48e-8 in
#: all four. All are at most 1/4.7 of the band. The test does not fix the thread count (it is a
#: check that runs under the default conditions).
SYNTHETIC_RATIO_BAND = 3.8e-6
REAL_RATIO_BAND = 3.0e-6

#: The band of the difference between the sequence-form attention and the upstream block
#: (synthetic, seed 4: measured 6.48e-8 × 5 → 3.3e-7 — the measurement conditions are the same as
#: for the bands above).
ATTENTION_RATIO_BAND = 3.3e-7

#: The pinned configuration of AvgDown3D (down blocks 0–3 — `(in, out, factor_t, factor_s)`).
PIN_SHORTCUTS = ((160, 160, 1, 2), (160, 320, 2, 2), (320, 640, 2, 2), (640, 640, 1, 1))


def _ratio(got: torch.Tensor, want: torch.Tensor) -> float:
    return float((got - want).abs().max() / want.abs().max())


def _bits(tensor: torch.Tensor) -> torch.Tensor:
    """The bit pattern of f32 (`torch.equal` treats −0 and +0 as equal — this sees down to the sign
    of zero)."""
    return tensor.contiguous().view(torch.int32)


def _image(size: tuple[int, int], seed: int) -> torch.Tensor:
    """A random image `[1, 3, 1, H, W]` in [-1, 1] (before patchify)."""
    generator = torch.Generator().manual_seed(seed)
    return torch.rand(1, 3, 1, *size, generator=generator) * 2 - 1


def _patchify(image: torch.Tensor) -> torch.Tensor:
    """The upstream `patchify` itself (pinning against a copy of our own would make the claim about
    the reference's origin tautological)."""
    from diffusers.models.autoencoders.autoencoder_kl_wan import patchify

    return patchify(image, patch_size=2)[0]


def _encode(vae, image: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return vae_encoder_patch.encode_chunk0(
            vae_encoder_patch.encoder_graphs(vae), _patchify(image)
        )


def _reference(vae, image: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return vae_encoder_patch.reference_mu(vae, image)[0]


class TestAvgDownShortcut:
    """The closed form of the shortcut against the upstream `AvgDown3D` (T = 1) — at the 5B's real
    channel counts."""

    @pytest.mark.parametrize(("channels", "out_channels", "factor_t", "factor_s"), PIN_SHORTCUTS)
    def test_bit_exact_with_the_upstream_module(
        self, channels: int, out_channels: int, factor_t: int, factor_s: int
    ):
        wan_vae = pytest.importorskip("diffusers.models.autoencoders.autoencoder_kl_wan")
        module = wan_vae.AvgDown3D(channels, out_channels, factor_t=factor_t, factor_s=factor_s)
        generator = torch.Generator().manual_seed(channels + out_channels)
        x = torch.randn(channels, 1, 22, 40, generator=generator)
        residual = torch.randn(out_channels, 1, 22 // factor_s, 40 // factor_s, generator=generator)

        got = vae_encoder_patch.avg_down_shortcut(module, x, residual)

        assert _bits(got).equal(_bits(residual + module(x.unsqueeze(0))[0]))

    def test_a_time_factor_2_shortcut_is_zero_on_even_channels_and_a_mean_on_odd_ones(self):
        """Pin the premise of the closed form on the upstream side: at T = 1 one zero frame is
        packed in front."""
        wan_vae = pytest.importorskip("diffusers.models.autoencoders.autoencoder_kl_wan")
        module = wan_vae.AvgDown3D(160, 320, factor_t=2, factor_s=2)
        x = torch.randn(1, 160, 1, 6, 10, generator=torch.Generator().manual_seed(3))

        out = module(x)[0]

        assert not out[0::2].any()
        corners = [x[0, :, 0, row::2, column::2] for row in (0, 1) for column in (0, 1)]
        mean = (((corners[0] + corners[1]) + corners[2]) + corners[3]) / 4
        assert torch.equal(out[1::2, 0], mean)

    def test_a_shortcut_outside_the_table_is_refused(self):
        wan_vae = pytest.importorskip("diffusers.models.autoencoders.autoencoder_kl_wan")
        module = wan_vae.AvgDown3D(8, 4, factor_t=1, factor_s=2)

        with pytest.raises(vae_encoder_patch.UnsupportedVaeError, match="unsupported"):
            vae_encoder_patch.avg_down_shortcut(module, torch.zeros(8, 1, 4, 4), torch.zeros(4))


class TestDownsample:
    def test_the_last_axis_pads_equal_the_upstream_zero_pad(self):
        x = torch.randn(1, 8, 9, 13, generator=torch.Generator().manual_seed(5))

        assert _bits(vae_encoder_patch.pad_right_bottom(x)).equal(
            _bits(nn.ZeroPad2d((0, 1, 0, 1))(x))
        )

    @pytest.mark.parametrize("mode", ["downsample2d", "downsample3d"])
    def test_chunk0_is_bit_exact_with_the_upstream_resample(self, mode: str):
        """downsample3d's `time_conv` does not run in chunk 0 (the upstream only pushes the input
        onto the cache)."""
        wan_vae = pytest.importorskip("diffusers.models.autoencoders.autoencoder_kl_wan")
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            module = wan_vae.WanResample(16, mode).eval()
        x = torch.randn(1, 16, 1, 10, 14, generator=torch.Generator().manual_seed(6))

        with torch.no_grad():
            want = module(x, feat_cache=[None], feat_idx=[0])[0]
            got = vae_encoder_patch.downsample(module, x[0])

        assert _bits(got).equal(_bits(want))


class TestLastSliceConv3d:
    @staticmethod
    def _causal(dtype: torch.dtype):
        wan_vae = pytest.importorskip("diffusers.models.autoencoders.autoencoder_kl_wan")
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(1)
            return wan_vae.WanCausalConv3d(16, 24, 3, padding=1).to(dtype).eval()

    def test_the_weight_is_the_last_time_slice(self):
        conv = self._causal(torch.float32)

        folded = vae_encoder_patch.LastSliceConv3d(conv)

        assert folded.weight.shape == (24, 16, 1, 3, 3)
        assert torch.equal(folded.weight, conv.weight[:, :, 2:])
        assert folded.padding == (0, 1, 1)

    @staticmethod
    def _input() -> torch.Tensor:
        generator = torch.Generator().manual_seed(2)
        return torch.randn(1, 16, 1, 9, 11, dtype=torch.float64, generator=generator)

    def test_the_fold_drops_only_zero_terms_in_f64(self):
        """Checking the mathematical identity (f64 — the remaining difference is only the reduction
        order, measured 8.7e-17)."""
        conv = self._causal(torch.float64)
        x = self._input()

        with torch.no_grad():
            want = conv(x)[0]
            got = vae_encoder_patch.LastSliceConv3d(conv)(x[0])

        assert _ratio(got, want) < 1e-14

    def test_folding_the_first_slice_breaks_the_identity_in_f64(self):
        """Fault injection: fold the wrong time slice (the first — the side that hits the zero frame
        in the upstream)."""
        conv = self._causal(torch.float64)
        x = self._input()
        folded = vae_encoder_patch.LastSliceConv3d(conv)
        folded.weight = nn.Parameter(conv.weight.detach()[:, :, :1].clone(), requires_grad=False)

        with torch.no_grad():
            broken = folded(x[0])
            want = conv(x)[0]

        assert _ratio(broken, want) > 0.1

    @pytest.mark.parametrize(
        ("kernel", "padding", "stride"),
        [((1, 1, 1), 0, 1), ((3, 1, 1), (0, 0, 0), (2, 1, 1))],
        ids=["pointwise", "time-conv-of-downsample3d"],
    )
    def test_a_conv_without_two_causal_zero_frames_is_refused(self, kernel, padding, stride):
        wan_vae = pytest.importorskip("diffusers.models.autoencoders.autoencoder_kl_wan")
        conv = wan_vae.WanCausalConv3d(4, 4, kernel, stride=stride, padding=padding)

        with pytest.raises(vae_encoder_patch.UnsupportedVaeError, match="cannot be folded"):
            vae_encoder_patch.LastSliceConv3d(conv)


class TestSequenceAttention:
    def test_the_sequence_form_follows_the_upstream_block(self, ti2v_synthetic_vae):
        """The `[C, S]` sequence form against the upstream `[1, C, 1, h, w]` attention (only the
        reduction order differs)."""
        block = ti2v_synthetic_vae.encoder.mid_block.attentions[0]
        x = torch.randn(1, 32, 1, 4, 6, generator=torch.Generator().manual_seed(4))

        with torch.no_grad():
            want = block(x)[0]
            got = vae_encoder_patch.VaeEncoderAttention(ti2v_synthetic_vae)(x[0].reshape(32, 24))

        assert _ratio(got.reshape(32, 1, 4, 6), want) <= ATTENTION_RATIO_BAND


@pytest.fixture(scope="module")
def synthetic_image() -> torch.Tensor:
    return _image(SYNTHETIC_SIZE, SYNTHETIC_SEED)


@pytest.fixture(scope="module")
def synthetic_reference(ti2v_synthetic_vae, synthetic_image) -> torch.Tensor:
    return _reference(ti2v_synthetic_vae, synthetic_image)


class TestEncodeChunk0:
    def test_the_three_graphs_stay_within_the_band(
        self, ti2v_synthetic_vae, synthetic_image, synthetic_reference
    ):
        got = _encode(ti2v_synthetic_vae, synthetic_image)

        assert got.shape == (48, 1, 4, 6) == synthetic_reference.shape
        assert _ratio(got, synthetic_reference) <= SYNTHETIC_RATIO_BAND

    def test_the_reference_is_the_first_half_of_the_upstream_moments(
        self, ti2v_synthetic_vae, synthetic_image, synthetic_reference
    ):
        """The reference's origin: `mode()` is the first 48 ch of the untiled `_encode` (after
        quant_conv)."""
        with torch.no_grad():
            moments = ti2v_synthetic_vae._encode(synthetic_image)

        assert torch.equal(synthetic_reference, moments[0, :48])

    def test_the_logvar_half_is_outside_the_band(
        self, ti2v_synthetic_vae, synthetic_image, synthetic_reference, monkeypatch
    ):
        """Fault injection: take the second half of quant_conv's output (logvar)."""
        original = vae_encoder_patch.pointwise_conv3d

        def logvar_first(conv: nn.Module, x: torch.Tensor) -> torch.Tensor:
            moments = original(conv, x)
            if conv is ti2v_synthetic_vae.quant_conv:
                return moments.roll(-48, dims=0)
            return moments

        monkeypatch.setattr(vae_encoder_patch, "pointwise_conv3d", logvar_first)
        broken = _encode(ti2v_synthetic_vae, synthetic_image)

        assert _ratio(broken, synthetic_reference) > 1e3 * SYNTHETIC_RATIO_BAND

    def test_a_patchify_with_the_height_and_width_digits_swapped_is_outside_the_band(
        self, ti2v_synthetic_vae, synthetic_image, synthetic_reference
    ):
        """Fault injection: swap r and q in the patchify channel `c·4 + r·2 + q` (r = the width
        sub-index, q = the height sub-index) (the spot where the host's patchify is easiest to get
        wrong)."""
        patches = _patchify(synthetic_image)
        _, _, height, width = patches.shape
        swapped = patches.reshape(3, 2, 2, 1, height, width).transpose(1, 2).reshape(patches.shape)

        with torch.no_grad():
            broken = vae_encoder_patch.encode_chunk0(
                vae_encoder_patch.encoder_graphs(ti2v_synthetic_vae), swapped
            )

        assert _ratio(broken, synthetic_reference) > 1e3 * SYNTHETIC_RATIO_BAND

    def test_a_shortcut_with_its_zero_and_data_halves_swapped_is_outside_the_band(
        self, ti2v_synthetic_vae, synthetic_image, synthetic_reference, monkeypatch
    ):
        """Fault injection: in a shortcut with time factor 2, add the average to the even frames
        (the zero side)."""
        original = vae_encoder_patch.avg_down_shortcut

        def swapped(module: nn.Module, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
            if int(module.factor_t) == 1:
                return original(module, x, residual)
            channels = residual.shape[0]
            flipped = residual.reshape(channels // 2, 2, *residual.shape[2:]).flip(1)
            out = original(module, x, flipped.reshape(residual.shape))
            return out.reshape(channels // 2, 2, *out.shape[2:]).flip(1).reshape(out.shape)

        monkeypatch.setattr(vae_encoder_patch, "avg_down_shortcut", swapped)
        broken = _encode(ti2v_synthetic_vae, synthetic_image)

        assert _ratio(broken, synthetic_reference) > 1e3 * SYNTHETIC_RATIO_BAND

    def test_the_pre_output_is_read_as_a_sequence_without_moving_bytes(self):
        hidden = torch.arange(32 * 4 * 6, dtype=torch.float32).reshape(32, 1, 4, 6)

        sequence = vae_encoder_patch.as_sequence(hidden)

        assert sequence.shape == (32, 24)
        assert sequence.data_ptr() == hidden.data_ptr()


class TestGenerationGate:
    def test_the_synthetic_ti2v_vae_is_accepted(self, ti2v_synthetic_vae):
        vae_encoder_patch.assert_supported(ti2v_synthetic_vae)

    def test_the_wan21_generation_is_refused(self, ti2v_synthetic_vae):
        from diffusers import AutoencoderKLWan

        vae = AutoencoderKLWan.from_config(
            ti2v_synthetic_vae.config,
            is_residual=False,
            patch_size=None,
            in_channels=3,
            out_channels=3,
        )

        with pytest.raises(vae_encoder_patch.UnsupportedVaeError, match="unsupported"):
            vae_encoder_patch.assert_supported(vae)

    def test_a_shortcut_group_outside_the_table_is_refused(self, ti2v_synthetic_vae):
        vae = copy.deepcopy(ti2v_synthetic_vae)
        vae.encoder.down_blocks[0].avg_shortcut.group_size = 8

        with pytest.raises(vae_encoder_patch.UnsupportedVaeError, match="unsupported"):
            vae_encoder_patch.assert_supported(vae)

    def test_a_shortcut_whose_time_factor_disagrees_with_the_downsampler_is_refused(
        self, ti2v_synthetic_vae
    ):
        """Down block 1 is downsample3d (time ×2) — with ×1 only in the shortcut, the channel order
        breaks (the table has (1, 2, 4), so the table check alone does not catch it)."""
        vae = copy.deepcopy(ti2v_synthetic_vae)
        vae.encoder.down_blocks[1].avg_shortcut.factor_t = 1

        with pytest.raises(vae_encoder_patch.UnsupportedVaeError, match="scale factors"):
            vae_encoder_patch.assert_supported(vae)

    def test_a_mid_block_with_two_attentions_is_refused(self, ti2v_synthetic_vae):
        vae = copy.deepcopy(ti2v_synthetic_vae)
        mid = vae.encoder.mid_block
        mid.attentions.append(copy.deepcopy(mid.attentions[0]))

        with pytest.raises(vae_encoder_patch.UnsupportedVaeError, match="mid block"):
            vae_encoder_patch.assert_supported(vae)

    def test_the_reference_refuses_the_tiled_encode(self, ti2v_synthetic_vae):
        vae = copy.deepcopy(ti2v_synthetic_vae)
        vae.enable_tiling()

        with pytest.raises(vae_encoder_patch.UnsupportedVaeError, match="non-tiled"):
            vae_encoder_patch.reference_mu(vae, torch.zeros(1, 3, 1, 32, 32))


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


class TestEncodeChunk0RealWeights:
    """Measured (2026-10-09, CPU 6 threads): at 256×160, seed 20261009 the ratio is 5.90e-7 (maximum
    absolute difference / the reference's maximum absolute value) → × 5 → 3.0e-6. The ratios on
    3 test images × 3 sizes (`python -m wan.export_vae_encoder --verify`) are 5.3e-7 to 2.09e-6
    (worst at cat-dog-baking's 1280×704, maximum absolute difference 5.07e-6) — the band derives
    from the decision case on a random image and is not widened by the real-image values."""

    def test_the_three_graphs_stay_within_the_band(self, vae):
        vae_encoder_patch.assert_supported(vae)
        image = _image(REAL_SIZE, REAL_SEED)

        got = _encode(vae, image)
        want = _reference(vae, image)

        assert got.shape == (48, 1, 10, 16) == want.shape
        assert _ratio(got, want) <= REAL_RATIO_BAND


@pytest.fixture(scope="module")
def config(wan22_snapshot) -> SimpleNamespace:
    """The latent statistics in `vae/config.json` of the pinned 5B (no weights are read)."""
    document = json.loads((wan22_snapshot / "vae" / "config.json").read_text(encoding="utf-8"))
    return SimpleNamespace(
        z_dim=document["z_dim"],
        latents_mean=document["latents_mean"],
        latents_std=document["latents_std"],
    )


class TestConditionNormalization:
    """The 1/std of the condition normalization is an f32 division (the upstream's
    `1.0 / torch.tensor(std)`) — it splits on some channels from f64 1/std rounded to f32 (the TS
    side shares `f32(1 / f32(std))` of `latents.ts`)."""

    def test_the_inverse_std_is_an_f32_division(self, config):
        mu = torch.ones(1, 48, 1, 1, 1)

        got = vae_encoder_patch.normalize_condition(SimpleNamespace(config=config), mu)

        mean = np.asarray(config.latents_mean, dtype=np.float32)
        inverse = np.float32(1) / np.asarray(config.latents_std, dtype=np.float32)
        want = (np.float32(1) - mean) * inverse
        assert np.array_equal(got.flatten().numpy().view(np.int32), want.view(np.int32))

    def test_the_two_spellings_of_the_inverse_std_differ_for_this_config(self, config):
        """The check above can tell the f64 way of writing 1/std apart (if the values were the same,
        it would not be a check)."""
        as_f32 = np.float32(1) / np.asarray(config.latents_std, dtype=np.float32)
        from_f64 = (1.0 / np.asarray(config.latents_std, dtype=np.float64)).astype(np.float32)

        assert np.count_nonzero(as_f32 != from_f64) > 0
