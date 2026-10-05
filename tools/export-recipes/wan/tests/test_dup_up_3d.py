"""`_shared.vae_rank4.dup_up_3d` の約束事 — 上流 `DupUp3D.forward` とビット一致すること
（ADR 0121 段 4）。

相手は上流のモジュールそのもの（diffusers 0.39.0 の `DupUp3D` — 重みを持たないので直接組む）。
構成は Wan2.2 の decoder のショートカット 3 本と同じ形（up0 / up1 = 1024 → 1024・ft 2・rep 8、
up2 = 1024 → 512・ft 1・rep 2）と、チャネルだけを縮めた同じ形の版、時間の桁の並びを値に出す
rep 2・ft 2 の版（{@link CONFIGS}）。空間は 3×5（縦横を変えて
H と W の取り違えを値に出す）、時間は T = 1（chunk の mid は潜在 1 フレーム）と T = 2
（up1 の入力）。

最初の chunk で時間の先頭を捨てる slice は wan 側（`wan.vae_patch.avg_shortcut`）が持つので、
ここは上流の `first_chunk=False` と比べる（slice を含めた比較は `test_vae_patch_ti2v.py`）。
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from _shared.vae_rank4 import dup_up_3d

autoencoder_kl_wan = pytest.importorskip("diffusers.models.autoencoders.autoencoder_kl_wan")

#: (Cin, Cout, ft)。空間の倍率 fs は 3 本とも 2（`WanResidualUpBlock` の `factor_s=2`）。
#: (16, 4, 2) は decoder に無い形（rep 2・ft 2）で、時間の桁 a がチャネルを選ぶ並びを値に出す —
#: ft 2 の実在の形は rep 8 で、8 個が全て同じ入力チャネルの写しなので a の並びが値に出ない。
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
        """故障注入: チャネルの添字の H の桁と W の桁を入れ替えた変種は一致しない。

        rep 2（up2 の形）では H の桁だけがチャネルを選ぶので値に出る。rep 8（up0 / up1）は 8 個が
        全て同じ入力チャネルの写しで、桁の取り違えは値に出ない（だから注入は (8, 4, 1) で行う）。
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
        """検査が落ちうること: 上流の forward は rank 8 の view を通る。"""

        class Upstream(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.shortcut = autoencoder_kl_wan.DupUp3D(8, 4, factor_t=2, factor_s=2)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.shortcut(x.unsqueeze(0))[0]

        assert self._max_rank(Upstream(), torch.randn(8, 2, 3, 5)) == 8


class TestFailsLoudly:
    def test_channels_that_do_not_divide_are_refused(self):
        with pytest.raises(ValueError, match="割り切れない"):
            dup_up_3d(torch.zeros(3, 1, 2, 2), out_channels=2, factor_t=1, factor_s=2)
