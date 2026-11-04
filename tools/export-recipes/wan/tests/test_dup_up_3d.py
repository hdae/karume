"""The contract of `_shared.vae_rank4.dup_up_3d` — bit-identical to upstream `DupUp3D.forward`
(ADR 0121 step 4).

The counterpart is the upstream module itself (`DupUp3D` of diffusers 0.39.0 — it has no weights,
so it is built directly). The configurations are the shape of the three shortcuts of the Wan2.2
decoder (up0 / up1 = 1024 → 1024, ft 2, rep 8; up2 = 1024 → 512, ft 1, rep 2), versions of the
same shape with only the channels shrunk, and a rep 2, ft 2 version that makes the order of the
time digits show in the values ({@link CONFIGS}). The spatial size is 3×5 (different height and
width, so that mixing up H and W shows in the values), and the time size is T = 1 (the mid of a
chunk is 1 latent frame) and T = 2 (the input of up1).

The slice that drops the first time step in the first chunk belongs to the wan side
(`wan.vae_patch.avg_shortcut`), so this file compares against upstream with `first_chunk=False`
(the comparison including the slice is `test_vae_patch_ti2v.py`).
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from _shared.vae_rank4 import dup_up_3d

autoencoder_kl_wan = pytest.importorskip("diffusers.models.autoencoders.autoencoder_kl_wan")

#: (Cin, Cout, ft). The spatial factor fs is 2 for all three (`factor_s=2` of
#: `WanResidualUpBlock`). (16, 4, 2) is a shape the decoder does not have (rep 2, ft 2) and makes
#: the order in which the time digit a selects the channel show in the values — the real shape with
#: ft 2 has rep 8, and its 8 copies are all of the same input channel, so the order of a does not
#: show in the values.
CONFIGS = [(8, 8, 2), (8, 4, 1), (1024, 1024, 2), (1024, 512, 1), (16, 4, 2)]
SPATIAL_SCALE = 2


def _upstream(x: torch.Tensor, out_channels: int, factor_t: int) -> torch.Tensor:
    module = autoencoder_kl_wan.DupUp3D(
        x.shape[0], out_channels, factor_t=factor_t, factor_s=SPATIAL_SCALE
    )
    return module(x.unsqueeze(0), first_chunk=False)[0]


class TestMatchesTheUpstreamModule:
    @pytest.mark.parametrize("frames", [1, 2])
    @pytest.mark.parametrize(("channels", "out_channels", "factor_t"), CONFIGS)
    def test_bit_exact(self, channels: int, out_channels: int, factor_t: int, frames: int):
        x = torch.randn(channels, frames, 3, 5, generator=torch.Generator().manual_seed(frames))

        got = dup_up_3d(x, out_channels=out_channels, factor_t=factor_t, factor_s=SPATIAL_SCALE)

        assert got.shape == (out_channels, frames * factor_t, 6, 10)
        assert torch.equal(got, _upstream(x, out_channels, factor_t))

    def test_swapping_the_h_and_w_digits_is_caught(self):
        """Fault injection: a variant that swaps the H digit and the W digit of the channel index
        does not match.

        With rep 2 (the shape of up2) only the H digit selects the channel, so the swap shows in
        the values. With rep 8 (up0 / up1) the 8 copies are all of the same input channel and a
        digit mix-up does not show in the values (hence the injection uses (8, 4, 1)).
        """
        x = torch.randn(8, 2, 3, 5, generator=torch.Generator().manual_seed(3))

        swapped = dup_up_3d(
            x.transpose(2, 3), out_channels=4, factor_t=1, factor_s=SPATIAL_SCALE
        ).transpose(2, 3)

        assert not torch.equal(swapped, _upstream(x, 4, 1))


class TestStaysWithinRank4:
    @staticmethod
    def _max_rank(module: nn.Module, x: torch.Tensor) -> int:
        program = torch.export.export(module, (x,))
        values = (node.meta.get("val") for node in program.graph.nodes)
        return max(value.dim() for value in values if isinstance(value, torch.Tensor))

    def test_every_value_of_the_traced_function_is_rank4_or_lower(self):
        class Shortcut(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return dup_up_3d(x, out_channels=4, factor_t=2, factor_s=SPATIAL_SCALE)

        assert self._max_rank(Shortcut(), torch.randn(8, 2, 3, 5)) <= 4

    def test_the_upstream_forward_is_rank8(self):
        """The check can fail: the upstream forward goes through a rank-8 view."""

        class Upstream(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.shortcut = autoencoder_kl_wan.DupUp3D(8, 4, factor_t=2, factor_s=2)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.shortcut(x.unsqueeze(0))[0]

        assert self._max_rank(Upstream(), torch.randn(8, 2, 3, 5)) == 8


class TestFailsLoudly:
    def test_channels_that_do_not_divide_are_refused(self):
        with pytest.raises(ValueError, match="is not divisible"):
            dup_up_3d(torch.zeros(3, 1, 2, 2), out_channels=2, factor_t=1, factor_s=2)
