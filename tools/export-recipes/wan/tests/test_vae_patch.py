"""`wan/vae_patch.py` の約束事（実重み — ADR 0118 決定 2 / 段 4 の検収）。

固定するのは 3 層:

- **cache の表**が決定 2 の表（32 本・形・`time_conv` の位置）と一致し、first が `time_conv` の
  2 本を持たないこと。
- **cache の正規化**（常に 2 フレーム・初期値ゼロ）が値を変えないこと — 上流の decoder のコード
  のまま cache だけを正規化した chunk ループが、上流の非タイル `_decode` と**ビット一致**する。
- **書き直しの部品**の eager 同値 — データ移動だけの書き直し（時間インターリーブ・nearest ×2・
  フレームを (b t) に畳む形）はビット一致、縮約順が変わる書き直し（RMS_norm の `sum` 形・空間
  padding の attrs 化）は実測から導いた帯の内側。chunk ループ全体（最終形）も帯の内側。

潜在は空間 8×8（64 px）・5 chunk（最初の chunk・`time_conv` の `'Rep'` 経路を通る 2 chunk 目・
cache が 2 フレーム揃う 3 chunk 目以降を全て通る最小の本数）。タイル 32・9 chunk の実寸の値は
`python -m wan.export_vae --verify` が測る（README の VAE 節）。
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from wan import vae_patch

#: 小さな潜在タイル（空間）と chunk 数。
TILE = 8
CHUNKS = 5

#: 書き直しの最終形（chunk ループ全体）と上流の差の帯（参照の最大絶対値に対する比）。
#: 実測（実重み f32・2026-10-02）: このテストの潜在（タイル 8・5 chunk）で 1.95e-6、タイル 32・
#: 9 chunk で 2.79e-6（`--verify`）。後者の約 5 倍。差の出所は RMS_norm の縮約形と conv の padding
#: の attrs 化（下の部品テスト）で、どちらも chunk を重ねても伸びない（最後の chunk は 1.36e-6）。
FINAL_RATIO_BAND = 1.5e-5

#: RMS_norm の `sum` 形と上流 `F.normalize`（`linalg_vector_norm`）の差（絶対値）。
#: 実測 1.19e-6（チャネル 384・4 フレーム・8×8・入力 N(0, 3²)）の約 5 倍。
RMS_NORM_BAND = 6e-6

#: 空間 padding を conv3d の attrs へ畳んだ形と上流（明示の `F.pad`）の差（出力の最大絶対値に
#: 対する比）。torch CPU は padding の持ち方と形で別の畳み込み経路を選ぶので、ビット一致は期待
#: しない — このテストの入力（8×8）では T = 1 がビット一致・T = 4 が 6.67e-6 で、16×16 の試作では
#: 逆に T = 1 だけが差を持った（2026-10-02）。帯は 6.67e-6 の約 5 倍。
CONV_PADDING_RATIO_BAND = 3.5e-5


@pytest.fixture(scope="module")
def vae(wan_snapshot):
    """pin した revision の VAE（CPU f32・丸め無し — 同値の主張は重みの値に依らない）。"""
    pytest.importorskip("diffusers")
    from diffusers import AutoencoderKLWan

    return AutoencoderKLWan.from_pretrained(
        wan_snapshot, subfolder="vae", torch_dtype=torch.float32
    ).eval()


@pytest.fixture(scope="module")
def latents(vae) -> torch.Tensor:
    """逆正規化した固定 seed の乱数潜在 `[1, 16, 5, 8, 8]`（実運用と同じ値域）。"""
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
    #: 決定 2 の表（潜在タイル 32 のとき）。(キー, 形, time_conv)。
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
        """上流のコードのまま cache の初期値だけを 2 フレームのゼロにする（決定 2 の導出の
        実測）。"""
        with torch.no_grad():
            normalized = vae_patch.normalized_cache_decode(vae, latents, TILE)

        assert torch.equal(normalized, reference)


@pytest.fixture(scope="module")
def decoded(vae, latents) -> torch.Tensor:
    """書き直しの最終形（chunk グラフ 2 本の eager）で回した chunk ループ。"""
    first = vae_patch.WanVaeChunkDecoder(vae, first=True).eval()
    following = vae_patch.WanVaeChunkDecoder(vae, first=False).eval()
    with torch.no_grad():
        return vae_patch.chunk_decode(first, following, latents[0])


class TestChunkDecode:
    def test_the_rewritten_chunk_loop_stays_within_the_band(self, decoded, reference):
        assert decoded.shape == reference.shape[1:] == (3, 1 + 4 * (CHUNKS - 1), 64, 64)
        assert _ratio(decoded, reference[0]) <= FINAL_RATIO_BAND

    def test_the_error_does_not_grow_with_the_chunks(self, decoded, reference):
        """cache の取り違えは後の chunk ほど差を積む — 最後の chunk も最初の帯に収まる。"""
        scale = reference.abs().max()
        last = (decoded[:, -4:] - reference[0, :, -4:]).abs().max() / scale

        assert float(last) <= FINAL_RATIO_BAND

    def test_a_cache_with_its_two_frames_swapped_is_far_outside_the_band(
        self, vae, latents, reference, monkeypatch: pytest.MonkeyPatch
    ):
        """故障注入: cache の 2 フレームの順を逆にすると帯を大きく外れる（帯が順を見分け
        られる）。"""
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
        """upsample2d（up2）は time_conv を持たない — フレームを (b t) に畳む形だけを見る。"""
        module = vae.decoder.up_blocks[2].upsamplers[0]
        x = torch.randn(192, 4, TILE, TILE)

        with torch.no_grad():
            got = vae_patch.resample(module, x, cursor=None)  # type: ignore[arg-type]
            want = module(x.unsqueeze(0))[0]

        assert torch.equal(got, want)

    def test_the_attention_fold_is_exact_once_the_norm_is_the_upstream_one(
        self, vae, monkeypatch: pytest.MonkeyPatch
    ):
        """RMS_norm だけを上流のものに戻すと、(b t) に畳む形はビット一致する（差の出所を
        切り分ける）。"""
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

        with pytest.raises(ValueError, match="順"):
            module(torch.zeros(16, 1, TILE, TILE), reordered)

    def test_a_cache_of_the_wrong_shape_is_refused(self, vae):
        conv = vae.decoder.conv_in

        with pytest.raises(ValueError, match="cache の形"):
            vae_patch.causal_conv3d(conv, torch.zeros(16, 1, 4, 4), torch.zeros(16, 1, 4, 4))

    def test_a_batched_latent_is_refused(self, vae):
        module = vae_patch.WanVaeChunkDecoder(vae, first=True)

        with pytest.raises(ValueError, match="unbatched"):
            module(torch.zeros(1, 16, 1, TILE, TILE), {})
