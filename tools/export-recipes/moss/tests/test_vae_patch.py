"""`moss/vae_patch.py` — the three decoder graphs on a tiny `AutoencoderKLFlux2` of the pinned
diffusers: the split decode is bit-identical to upstream's, the ×2 upsample rewrite is the nearest
interpolation, and the configuration check refuses what the rewrite does not assume (the real-size
equivalence is `moss.export_vae`'s gate 2)."""

from __future__ import annotations

import pytest
import torch
from torch.nn import functional

from _shared.vae_rank4 import repeat_2x
from moss import vae_patch
from moss.records import bits_equal

#: A decoder with the FLUX.2 small decoder's structure (four up blocks, three ×2 upsamples, a
#: channel change with a shortcut in the last two blocks) at widths of 4 … 16.
TINY_VAE = {
    "block_out_channels": (8, 8, 8, 8),
    "decoder_block_out_channels": (4, 8, 16, 16),
    "latent_channels": 4,
    "norm_num_groups": 4,
    "layers_per_block": 2,
}


def tiny_vae():
    """The tiny VAE in eval mode, with every parameter and the BatchNorm statistics moved off their
    initial values (a weight left at 1 / 0 hides a mix-up)."""
    diffusers = pytest.importorskip("diffusers")
    torch.manual_seed(0)
    vae = diffusers.AutoencoderKLFlux2(**TINY_VAE).eval()
    with torch.no_grad():
        for parameter in vae.parameters():
            parameter.add_(0.05 * torch.randn_like(parameter))
        vae.bn.running_mean.copy_(0.3 * torch.randn_like(vae.bn.running_mean))
        vae.bn.running_var.copy_(0.5 + torch.rand_like(vae.bn.running_var))
    return vae


@pytest.fixture(scope="module")
def vae():
    return tiny_vae()


class TestDecodeSplit:
    def test_the_three_graphs_chained_are_bit_identical_to_upstreams_decode(self, vae):
        z = torch.randn(1, 4, 5, 7, generator=torch.Generator().manual_seed(1))
        graphs = vae_patch.decoder_graphs(vae)
        with torch.no_grad():
            split = vae_patch.decode_split(graphs, z)
            upstream = vae.decode(z, return_dict=False)[0]
        assert split.shape == (1, 3, 40, 56)
        assert bits_equal(split, upstream)

    def test_the_graph_boundaries_are_upstreams_tensors(self, vae):
        from moss import pipeline_ref

        z = torch.randn(1, 4, 6, 3, generator=torch.Generator().manual_seed(2))
        pre, attention, _post = vae_patch.decoder_graphs(vae)
        # The graph modules hold upstream's modules, so they run after the recorder is closed (its
        # hooks would see them as a second decode).
        with pipeline_ref.VaeRecorder(vae) as recorder, torch.no_grad():
            vae.decode(z, return_dict=False)
        with torch.no_grad():
            hidden = pre(z)
            tokens = attention(hidden.reshape(1, hidden.shape[1], -1))
        assert bits_equal(hidden, recorder.tensors["vae.pre_attention"])
        assert bits_equal(tokens.reshape(hidden.shape), recorder.tensors["vae.post_attention"])

    def test_the_graph_modules_hold_upstreams_state_dict_keys(self, vae):
        keys = set()
        for graph in vae_patch.decoder_graphs(vae):
            own = set(graph.state_dict())
            assert not own & keys
            keys |= own
        upstream = {k for k in vae.state_dict() if k.startswith(("decoder.", "post_quant_conv."))}
        assert keys == upstream

    def test_the_spatial_scale_is_two_per_upsample(self, vae):
        assert vae_patch.spatial_scale(vae) == 8
        assert vae_patch.hidden_channels(vae) == 16


class TestUpsample:
    def test_the_repetition_is_nearest_and_nearest_exact_interpolation(self):
        x = torch.randn(2, 3, 5, 4, generator=torch.Generator().manual_seed(3))
        repeated = repeat_2x(x)
        nearest = functional.interpolate(x, scale_factor=2.0, mode="nearest")
        exact = functional.interpolate(x, scale_factor=2.0, mode="nearest-exact")
        assert bits_equal(repeated, nearest)
        assert bits_equal(repeated, exact)

    def test_upsample_2d_is_upstreams_forward(self, vae):
        upsampler = vae.decoder.up_blocks[0].upsamplers[0]
        x = torch.randn(1, upsampler.channels, 3, 5, generator=torch.Generator().manual_seed(4))
        with torch.no_grad():
            assert bits_equal(vae_patch.upsample_2d(upsampler, x), upsampler(x))
            with pytest.raises(vae_patch.UnsupportedVaeError, match="channels"):
                vae_patch.upsample_2d(upsampler, x[:, :-1])


class TestAssertSupported:
    def test_tiling_is_refused(self):
        vae = tiny_vae()
        vae.enable_tiling()
        with pytest.raises(vae_patch.UnsupportedVaeError, match="tiling"):
            vae_patch.assert_supported(vae)

    def test_a_transposed_upsample_is_refused(self):
        vae = tiny_vae()
        vae.decoder.up_blocks[1].upsamplers[0].use_conv_transpose = True
        with pytest.raises(vae_patch.UnsupportedVaeError, match="Upsample2D"):
            vae_patch.assert_supported(vae)

    def test_another_attention_processor_is_refused(self):
        from diffusers.models.attention_processor import AttnProcessor

        vae = tiny_vae()
        vae.decoder.mid_block.attentions[0].set_processor(AttnProcessor())
        with pytest.raises(vae_patch.UnsupportedVaeError, match="processor"):
            vae_patch.assert_supported(vae)

    def test_a_rescaled_attention_is_refused(self):
        vae = tiny_vae()
        vae.decoder.mid_block.attentions[0].rescale_output_factor = 2.0
        with pytest.raises(vae_patch.UnsupportedVaeError, match="rescale"):
            vae_patch.assert_supported(vae)
