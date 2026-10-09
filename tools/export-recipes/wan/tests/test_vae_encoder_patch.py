"""`wan/vae_encoder_patch.py` の約束事（ADR 0121 決定 11・段 9a の検収）。

2 層で固定する:

- **合成の 2.2 VAE**（常に走る — conftest の `ti2v_synthetic_vae`。encoder は 5B と同じ構造〈down
  block 4 本・ショートカット (1,2,4) / (2,2,4) / (2,2,4) / (1,1,1)・mid の Res → Attention →
  Res〉でチャネルだけを縮めたもの）: 部品ごとの上流との一致（AvgDown3D の閉じた形は 5B の
  実寸のチャネルでビット一致・ZeroPad2d と down の resample もビット一致・時間スライスへの
  畳み込みは f64 で恒等）・
  3 グラフを繋いだ mu の帯・故障注入・構成の門。
- **実重み**（pin した 5B の snapshot が無い機では SKIP）: 3 グラフを繋いだ mu と上流の非タイル
  encode の mu の帯（256×160 の乱数画像）・条件の正規化の 1/std が f32 の割り算であること。

ビット一致しない箇所と理由（2026-10-09 の実測 — 帯はそれぞれの決定用の実測の 5 倍を有効数字
2 桁で切り上げた値）:

- 時間スライスへの畳み込み（`LastSliceConv3d`）: f32 では上流の「ゼロ 2 枚を詰めた conv3d」と
  縮約の順が違う（CPU の conv のカーネルが時間 3 と 1 で別の経路を通る — 相対 5.6e-7）。f64 では
  相対 8.7e-17 で、消えるのはゼロ × 重みの項だけ（数学的に恒等）。
- RMS norm の縮約形（`l2_normalize` — decoder の書き直しと同じ）と、attention の 1×1 conv を
  linear にした系列形も、縮約の順だけが違う。
- AvgDown3D の閉じた形・ZeroPad2d・down の resample は上流とビット一致（和の順も同じ）。
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

#: 合成の画像の寸法（[-1, 1] の乱数・patchify の前 — 潜在 4×6）と seed。
SYNTHETIC_SIZE = (64, 96)
SYNTHETIC_SEED = 11

#: 実重みの画像の寸法（縮小の経路と同じ 256×160 — 潜在 10×16）と seed。
REAL_SIZE = (160, 256)
REAL_SEED = 20261009

#: 3 グラフを繋いだ mu と上流の mu の差の帯（参照の最大絶対値に対する比）。
#: - 合成（64×96・seed 11）: 実測 7.55e-7 × 5 → 3.8e-6。
#: - 実重み f32（256×160・seed 20261009）: 実測 5.90e-7 × 5 → 3.0e-6（実画像 3 枚 × 3 寸法の
#:   実測は {@link TestEncodeChunk0RealWeights} の docstring）。
#:
#: 測定の条件（帯を導いた値）: 2026-10-09・Ryzen 5 7600・torch 2.13.0+cpu（CPU capability
#: AVX512）・torch の既定の 6 スレッド。差の出所は縮約の順（畳んだ conv3d と上流の時間 3 の
#: conv3d・RMS norm の縮約形）で、CPU の conv の経路の分割はスレッド数で変わりうるので、同じ日に
#: `torch.set_num_threads` を 1 / 3 / 6 / 12 にして測り直した: 合成 7.34e-7〜7.55e-7・実重み
#: 4.99e-7〜6.35e-7（最悪は 1 スレッド）・attention は 4 通りとも 6.48e-8。どれも帯の 1/4.7 以下。
#: スレッド数はテストで固定しない（既定の条件で回る門にする）。
SYNTHETIC_RATIO_BAND = 3.8e-6
REAL_RATIO_BAND = 3.0e-6

#: 系列形の attention と上流の block の差の帯（合成・seed 4: 実測 6.48e-8 × 5 → 3.3e-7 — 測定の
#: 条件は上の帯と同じ）。
ATTENTION_RATIO_BAND = 3.3e-7

#: AvgDown3D の pin の構成（down block 0〜3 — `(in, out, factor_t, factor_s)`）。
PIN_SHORTCUTS = ((160, 160, 1, 2), (160, 320, 2, 2), (320, 640, 2, 2), (640, 640, 1, 1))


def _ratio(got: torch.Tensor, want: torch.Tensor) -> float:
    return float((got - want).abs().max() / want.abs().max())


def _bits(tensor: torch.Tensor) -> torch.Tensor:
    """f32 のビット列（`torch.equal` は −0 と +0 を等しいとみなす — 符号付きゼロまで見る）。"""
    return tensor.contiguous().view(torch.int32)


def _image(size: tuple[int, int], seed: int) -> torch.Tensor:
    """[-1, 1] の乱数画像 `[1, 3, 1, H, W]`（patchify の前）。"""
    generator = torch.Generator().manual_seed(seed)
    return torch.rand(1, 3, 1, *size, generator=generator) * 2 - 1


def _patchify(image: torch.Tensor) -> torch.Tensor:
    """上流の `patchify` そのもの（自前の写しで縛ると参照の素性の主張が恒真になる）。"""
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
    """ショートカットの閉じた形と上流 `AvgDown3D`（T = 1）— 5B の実寸のチャネルで。"""

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
        """閉じた形の前提を上流の側で縛る: T = 1 の先頭にゼロが 1 枚詰められる。"""
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

        with pytest.raises(vae_encoder_patch.UnsupportedVaeError, match="未対応"):
            vae_encoder_patch.avg_down_shortcut(module, torch.zeros(8, 1, 4, 4), torch.zeros(4))


class TestDownsample:
    def test_the_last_axis_pads_equal_the_upstream_zero_pad(self):
        x = torch.randn(1, 8, 9, 13, generator=torch.Generator().manual_seed(5))

        assert _bits(vae_encoder_patch.pad_right_bottom(x)).equal(
            _bits(nn.ZeroPad2d((0, 1, 0, 1))(x))
        )

    @pytest.mark.parametrize("mode", ["downsample2d", "downsample3d"])
    def test_chunk0_is_bit_exact_with_the_upstream_resample(self, mode: str):
        """downsample3d の `time_conv` は chunk 0 で走らない（上流は入力を cache に積むだけ）。"""
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
        """数学的な恒等の確かめ（f64 — 残る差は縮約の順だけ・実測 8.7e-17）。"""
        conv = self._causal(torch.float64)
        x = self._input()

        with torch.no_grad():
            want = conv(x)[0]
            got = vae_encoder_patch.LastSliceConv3d(conv)(x[0])

        assert _ratio(got, want) < 1e-14

    def test_folding_the_first_slice_breaks_the_identity_in_f64(self):
        """故障注入: 畳む時間スライスを取り違える（先頭 = 上流ではゼロのフレームに掛かる側）。"""
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

        with pytest.raises(vae_encoder_patch.UnsupportedVaeError, match="畳めない"):
            vae_encoder_patch.LastSliceConv3d(conv)


class TestSequenceAttention:
    def test_the_sequence_form_follows_the_upstream_block(self, ti2v_synthetic_vae):
        """`[C, S]` の系列形と上流の `[1, C, 1, h, w]` の attention（縮約の順だけが違う）。"""
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
        """参照の素性: `mode()` は非タイルの `_encode`（quant_conv の後）の前半 48 ch。"""
        with torch.no_grad():
            moments = ti2v_synthetic_vae._encode(synthetic_image)

        assert torch.equal(synthetic_reference, moments[0, :48])

    def test_the_logvar_half_is_outside_the_band(
        self, ti2v_synthetic_vae, synthetic_image, synthetic_reference, monkeypatch
    ):
        """故障注入: quant_conv の出力の後半（logvar）を取る。"""
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
        """故障注入: patchify のチャネル `c·4 + r·2 + q`（r = 幅・q = 高さの副添字）の r と q を
        取り違える（ホストの patchify が最も取り違えやすい箇所）。"""
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
        """故障注入: 時間倍率 2 のショートカットで、平均を偶数番（ゼロの側）へ足す。"""
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

        with pytest.raises(vae_encoder_patch.UnsupportedVaeError, match="未対応"):
            vae_encoder_patch.assert_supported(vae)

    def test_a_shortcut_group_outside_the_table_is_refused(self, ti2v_synthetic_vae):
        vae = copy.deepcopy(ti2v_synthetic_vae)
        vae.encoder.down_blocks[0].avg_shortcut.group_size = 8

        with pytest.raises(vae_encoder_patch.UnsupportedVaeError, match="未対応"):
            vae_encoder_patch.assert_supported(vae)

    def test_a_shortcut_whose_time_factor_disagrees_with_the_downsampler_is_refused(
        self, ti2v_synthetic_vae
    ):
        """down block 1 は downsample3d（時間 ×2）— ショートカットだけ ×1 だとチャネルの並びが
        崩れる（表には (1, 2, 4) があるので、表の門だけでは掴めない）。"""
        vae = copy.deepcopy(ti2v_synthetic_vae)
        vae.encoder.down_blocks[1].avg_shortcut.factor_t = 1

        with pytest.raises(vae_encoder_patch.UnsupportedVaeError, match="倍率"):
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

        with pytest.raises(vae_encoder_patch.UnsupportedVaeError, match="非タイル"):
            vae_encoder_patch.reference_mu(vae, torch.zeros(1, 3, 1, 32, 32))


# ---- 実重み（pin した 5B の snapshot が無ければ SKIP）---------------------------------


@pytest.fixture(scope="module")
def vae(wan22_snapshot):
    """pin した revision の 5B の VAE（CPU f32・丸め無し — 同値の主張は重みの値に依らない）。"""
    pytest.importorskip("diffusers")
    from diffusers import AutoencoderKLWan

    return AutoencoderKLWan.from_pretrained(
        wan22_snapshot, subfolder="vae", torch_dtype=torch.float32
    ).eval()


class TestEncodeChunk0RealWeights:
    """実測（2026-10-09・CPU 6 スレッド）: 256×160・seed 20261009 で比 5.90e-7（最大絶対値の差 /
    参照の最大絶対値）→ × 5 → 3.0e-6。テスト画像 3 枚 × 3 寸法（`python -m wan.export_vae_encoder
    --verify`）の比は 5.3e-7〜2.09e-6（最悪は cat-dog-baking の 1280×704・最大絶対値の差 5.07e-6）—
    帯は乱数画像の決定用から導き、実画像の値で広げない。"""

    def test_the_three_graphs_stay_within_the_band(self, vae):
        vae_encoder_patch.assert_supported(vae)
        image = _image(REAL_SIZE, REAL_SEED)

        got = _encode(vae, image)
        want = _reference(vae, image)

        assert got.shape == (48, 1, 10, 16) == want.shape
        assert _ratio(got, want) <= REAL_RATIO_BAND


@pytest.fixture(scope="module")
def config(wan22_snapshot) -> SimpleNamespace:
    """pin した 5B の `vae/config.json` の潜在の統計（重みは読まない）。"""
    document = json.loads((wan22_snapshot / "vae" / "config.json").read_text(encoding="utf-8"))
    return SimpleNamespace(
        z_dim=document["z_dim"],
        latents_mean=document["latents_mean"],
        latents_std=document["latents_std"],
    )


class TestConditionNormalization:
    """条件の正規化の 1/std は f32 の割り算（上流の `1.0 / torch.tensor(std)`）— f64 の 1/std を
    f32 へ丸めた値とは一部のチャネルで割れる（TS は `latents.ts` の `f32(1 / f32(std))` を
    共有）。"""

    def test_the_inverse_std_is_an_f32_division(self, config):
        mu = torch.ones(1, 48, 1, 1, 1)

        got = vae_encoder_patch.normalize_condition(SimpleNamespace(config=config), mu)

        mean = np.asarray(config.latents_mean, dtype=np.float32)
        inverse = np.float32(1) / np.asarray(config.latents_std, dtype=np.float32)
        want = (np.float32(1) - mean) * inverse
        assert np.array_equal(got.flatten().numpy().view(np.int32), want.view(np.int32))

    def test_the_two_spellings_of_the_inverse_std_differ_for_this_config(self, config):
        """上の門が f64 の 1/std の書き方を見分けられること（同じ値なら門にならない）。"""
        as_f32 = np.float32(1) / np.asarray(config.latents_std, dtype=np.float32)
        from_f64 = (1.0 / np.asarray(config.latents_std, dtype=np.float64)).astype(np.float32)

        assert np.count_nonzero(as_f32 != from_f64) > 0
