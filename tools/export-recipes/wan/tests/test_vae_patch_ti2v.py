"""`wan/vae_patch.py` の Wan2.2 の分岐の約束事（ADR 0121 決定 6 / 段 4 の検収）。

2 層で固定する:

- **合成の 2.2 VAE**（常に走る — conftest の `ti2v_synthetic_vae`。上流 `AutoencoderKLWan` を 5B と
  同じ構造〈residual の up block・patch 2・ショートカット 3 本〉の小さなチャネルで乱数初期化した
  もの）: ショートカット（最初の chunk の slice を含む）の上流との一致・cache の表・cache の正規化の
  ビット一致・最終形の帯・故障注入・世代の門。
- **実重み**（pin した 5B の snapshot が無い機では SKIP — 2.1 の実重みのテストと同じ規律）:
  タイル 16 の cache の表（架構の表）と要素数・タイル 8 で 5 chunk の正規化のビット一致と
  最終形の帯。

5 chunk は、最初の chunk・`time_conv` の `'Rep'` 経路を通る 2 chunk 目・通常の cache が 2 フレーム
揃う 3 chunk 目・1 本目の `time_conv`（潜在 1 フレームを受ける）が使う cache が 2 フレームとも
非ゼロになる 4 chunk 目を通り、全ての cache が揃った後の chunk を 1 本含む本数（全てを通る最小は
4）。実寸（タイル 16・9 chunk）の値は
`python -m wan.export_vae --model ti2v-5b --verify --chunks 9` が測る。

実重みのテストは VAE 全体（encoder を含む f32 約 2.82 GB）を読む。2026-10-05 の実測（別の重い CPU
作業と同居）で RSS の山は約 3.5 GiB、このファイルの実重みの部分は約 2 分。長い CPU 作業と同居させて
RAM が足りないときは、そのテストを `-k 'not RealWeights'` で外して後で回す。
"""

from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from wan import vae_patch

#: 合成の VAE の潜在タイル（空間）・実重みの潜在タイル・chunk 数。
SYNTHETIC_TILE = 4
REAL_TILE = 8
CHUNKS = 5

#: Wan2.2 の patchify の倍率（上流 config の `patch_size`）。
PATCH_SIZE = 2

#: 書き直しの最終形（chunk ループ全体）と上流の差の帯（参照の最大絶対値に対する比）。どちらも
#: 決定用の実測（このテストの潜在 — seed 7・5 chunk）の 5 倍を有効数字 2 桁で切り上げた値。差の
#: 出所は 2.1 と同じ RMS_norm の縮約形と conv の padding の attrs 化（`test_vae_patch.py`）で、
#: ショートカットはデータ移動だけ。
#: - 合成（タイル 4）: 実測 1.05e-6（最後の chunk は 9.09e-7）。
#: - 実重み f32（タイル 8・2026-10-05）: 実測 5.31e-6（最後の chunk は 4.20e-6）。2.1 の同じ寸法
#:   （1.95e-6）より大きいのは、縮約するチャネルが 2.7〜4 倍あるため（推測）。
SYNTHETIC_FINAL_RATIO_BAND = 5.3e-6
REAL_FINAL_RATIO_BAND = 2.7e-5


def _ratio(got: torch.Tensor, want: torch.Tensor) -> float:
    return float((got - want).abs().max() / want.abs().max())


def _latents(vae, tile: int) -> torch.Tensor:
    """逆正規化した固定 seed の乱数潜在 `[1, 48, 5, t, t]`（実運用と同じ値域）。"""
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
    """上流の `unpatchify` そのもの（自前の写しで縛ると参照の素性の主張が恒真になる）。"""
    from diffusers.models.autoencoders.autoencoder_kl_wan import unpatchify

    return unpatchify(frames, patch_size=PATCH_SIZE)


class TestAvgShortcut:
    """ショートカットの wan 側（`avg_shortcut` — 最初の chunk の slice を持つ）と上流 `DupUp3D`。

    `importorskip` は各テストの中で受ける（fixture を経ないので、diffusers の無い既定の sync で
    FAIL にせず SKIP にする — `test_vae_tiling.py` と同じ流儀）。
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
        with pytest.raises(vae_patch.UnsupportedVaeError, match="入力チャネル"):
            vae_patch.avg_shortcut(
                wan_vae.DupUp3D(8, 8, factor_t=2, factor_s=2),
                torch.zeros(4, 1, 2, 2),
                first_chunk=False,
            )


class TestCacheTableTi2v:
    #: 合成の VAE の表（潜在タイル 4 のとき）。(キー, 形, time_conv)。並びは 5B の架構の表と同じで、
    #: チャネルだけが 1/16（5B は 48 / 1024 / 512 / 256）。
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
        """故障注入: up2（rep 2）のショートカットで、チャネルの添字の H と W の桁を取り違える。"""
        original = vae_patch.dup_up_3d

        def swapped(x: torch.Tensor, **factors: int) -> torch.Tensor:
            return original(x.transpose(2, 3), **factors).transpose(2, 3)

        monkeypatch.setattr(vae_patch, "dup_up_3d", swapped)
        broken = _chunk_decode(ti2v_synthetic_vae, synthetic_latents)

        assert _ratio(broken, synthetic_reference[0]) > SYNTHETIC_FINAL_RATIO_BAND


class TestTheFirstChunkSlice:
    """故障注入: ショートカットの最初の chunk の slice を落とすと、first のグラフは形で落ちる。

    up0 は `[C,1,…] + [C,2,…]` の broadcast で黙って通るが、up1 で 2 枚 + 4 枚になり加算できない —
    値が通ってしまう形は無い。GPU の照合ではなく eager と export の形の失敗で確かめる（欠けた
    グラフは export できない）。
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

        with pytest.raises(vae_patch.UnsupportedVaeError, match="未対応"):
            vae_patch.assert_supported(vae)

    def test_an_exit_that_is_not_the_patchify_space_is_refused(self, ti2v_synthetic_vae):
        from diffusers import AutoencoderKLWan

        vae = AutoencoderKLWan.from_config(ti2v_synthetic_vae.config, out_channels=3)

        with pytest.raises(vae_patch.UnsupportedVaeError, match="出口のチャネル"):
            vae_patch.assert_supported(vae)

    def test_a_block_whose_shortcut_and_upsampler_disagree_is_refused(self, ti2v_synthetic_vae):
        vae = copy.deepcopy(ti2v_synthetic_vae)
        vae.decoder.up_blocks[0].avg_shortcut = None

        with pytest.raises(vae_patch.UnsupportedVaeError, match="有無"):
            vae_patch.assert_supported(vae)

    def test_a_shortcut_with_another_time_factor_is_refused(self, ti2v_synthetic_vae):
        """up2 の upsampler は upsample2d（時間 ×1）— ショートカットだけ ×2 だと 1 枚 / 4 枚が
        崩れる。"""
        vae = copy.deepcopy(ti2v_synthetic_vae)
        vae.decoder.up_blocks[2].avg_shortcut.factor_t = 2

        with pytest.raises(vae_patch.UnsupportedVaeError, match="倍率"):
            vae_patch.assert_supported(vae)

    def test_the_reference_also_refuses_an_unsupported_generation(self, ti2v_synthetic_vae):
        from diffusers import AutoencoderKLWan

        vae = AutoencoderKLWan.from_config(ti2v_synthetic_vae.config, patch_size=None)

        with pytest.raises(vae_patch.UnsupportedVaeError, match="未対応"):
            vae_patch.reference_decode_unclamped(vae, torch.zeros(1, 48, 1, 2, 2))


# ---- 実重み（pin した 5B の snapshot が無ければ SKIP）---------------------------------


@pytest.fixture(scope="module")
def vae(wan22_snapshot):
    """pin した revision の 5B の VAE（CPU f32・丸め無し — 同値の主張は重みの値に依らない）。"""
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
    #: 架構の表（潜在タイル 16 のとき — ADR 0121 段 4 の設計の表）。(キー, 形, time_conv)。
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
