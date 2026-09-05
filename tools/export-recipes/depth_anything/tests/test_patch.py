"""`depth_anything.patch` の書き換えが原実装と同値であることの回帰テスト（実重み不要分）。

実重み 518² での端から端までの同値は export 台本（`depth_anything.export.py --verify`）の
責務で、ここは「どの書き換えがどの強さで同値か」を単体で固定する。

主張の強さは 2 段（`depth_anything.patch` の docstring と対応）:

- **添字の同値**（`ConvTranspose2d` → 1×1 conv + pixel shuffle）は **整数値データでの
  `torch.equal`**。整数値なら f32 の積和が丸めを持たないので、縮約順序が違っても差は出ない
  — つまり「差 0」がそのまま**添字の写像が厳密に一致している**という主張になる
  （Cin / Cout / H / W を全て別の長さにして、軸の取り違えが素通りしないようにする）。
  実データ相当（乱数 f32）の側は縮約順序のぶんだけ動くので、丸め幅の tolerance で見る。
- **演算列の同値**（最終融合段の倍率 → 寸法指定 / 位置埋め込み補間の門）は
  **`torch.equal` = ビット一致**。演算を 1 つも増減させないので、「差が小さい」で通す形に
  すると寸法の取り違えが素通りする。
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from depth_anything import patch as patch_depth_anything
from depth_anything.patch import SubPixelUpsample

#: 乱数 f32 での `ConvTranspose2d` 分解の許容差（Cin 方向の縮約順序の差の丸め幅）。
#:
#: MUST: 実重み 518² 端から端の実測幅をここで名乗らない — その綴りは
#: {@link depth_anything.measurements.CONVT_MAXDIFF} の 1 つだけ（同ファイルの MUST。カードと
#: `NOTICE.md` が名乗る数と分かれると、片方だけ動いても散文としては妥当なままになる）。ここは
#: **合成モジュール**での上限で、実重みの実測幅を上回るよう余裕を取ってある。
SUB_PIXEL_ATOL = 1e-5


def _integer_transposed_conv(in_channels: int, out_channels: int, scale: int) -> nn.ConvTranspose2d:
    """重み・bias が**整数値**の `ConvTranspose2d(kernel == stride)`。

    整数値にするのは、f32 の積和が丸めを持たない領域に留めるため（縮約順序が違っても
    ビット一致するので、差 0 が添字の同値そのものを意味する）。
    """
    module = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=scale, stride=scale)
    generator = torch.Generator().manual_seed(20260814 + scale)
    with torch.no_grad():
        module.weight.copy_(torch.randint(-4, 5, module.weight.shape, generator=generator).float())
        module.bias.copy_(torch.randint(-4, 5, module.bias.shape, generator=generator).float())
    return module.eval()


class TestSubPixelUpsample:
    """`ConvTranspose2d(kernel == stride)` → 1×1 conv + pixel shuffle の同値。"""

    @pytest.mark.parametrize("scale", [2, 3, 4])
    def test_integer_data_matches_the_transposed_convolution_bit_for_bit(self, scale: int) -> None:
        """整数値データでは**ビット一致**（= 添字の写像が厳密に一致している）。

        Cin / Cout / H / W / batch を全て別の長さにする — 正方や Cin == Cout だけで見ると、
        重みの軸の取り違え（`[Cin, Cout, r, r]` の先頭 2 軸の入れ替え）や H/W の入れ替えが
        差 0 のまま通る。
        """
        source = _integer_transposed_conv(3, 5, scale)
        generator = torch.Generator().manual_seed(7)
        x = torch.randint(-3, 4, (2, 3, 4, 6), generator=generator).float()

        with torch.no_grad():
            expected = source(x)
            got = SubPixelUpsample(source)(x)

        assert tuple(got.shape) == tuple(expected.shape)
        assert torch.equal(got, expected)

    def test_random_float_data_stays_within_the_rounding_width(self) -> None:
        """乱数 f32 では縮約順序のぶんだけ動く（丸め幅に留まる）。"""
        source = nn.ConvTranspose2d(48, 48, kernel_size=4, stride=4).eval()
        generator = torch.Generator().manual_seed(11)
        x = torch.randn((1, 48, 37, 37), generator=generator)

        with torch.no_grad():
            expected = source(x)
            got = SubPixelUpsample(source)(x)

        assert not torch.equal(got, expected), "丸めが動かないなら tolerance の根拠が消える"
        assert float((got - expected).abs().max()) < SUB_PIXEL_ATOL

    @pytest.mark.parametrize("scale", [2, 3])
    def test_the_shuffle_matches_pixel_shuffle(self, scale: int) -> None:
        """並べ替えの部分は `F.pixel_shuffle` そのもの（rank-4 に割っても写像は同じ）。

        {@link SubPixelUpsample} の 1×1 conv を恒等（単位行列・bias 0）にすると、残るのは
        並べ替えだけになる。独立な参照（ATen の `pixel_shuffle`）と突き合わせる。
        """
        channels = 3
        source = nn.ConvTranspose2d(channels * scale * scale, channels, scale, stride=scale)
        with torch.no_grad():
            source.weight.zero_()
            source.bias.zero_()
            # weight は [Cin, Cout, r, r] — Cin = c·r² を (c, ki, kj) と読んで単位写像を作る。
            for out_channel in range(channels):
                for row in range(scale):
                    for column in range(scale):
                        index = (out_channel * scale + row) * scale + column
                        source.weight[index, out_channel, row, column] = 1.0

        generator = torch.Generator().manual_seed(13)
        x = torch.randn((2, channels * scale * scale, 4, 6), generator=generator)
        with torch.no_grad():
            got = SubPixelUpsample(source)(x)

        assert torch.equal(got, nn.functional.pixel_shuffle(x, scale))

    @pytest.mark.parametrize(
        ("kwargs", "why", "match"),
        [
            (
                {"kernel_size": 3, "stride": 2},
                "kernel != stride は窓が重なる",
                r"kernel_size=\(3, 3\) と stride=\(2, 2\)",
            ),
            (
                {"kernel_size": 2, "stride": 2, "padding": 1},
                "padding は端を落とす",
                r"padding=\(1, 1\)",
            ),
            (
                {"kernel_size": 2, "stride": 2, "output_padding": 1},
                "output_padding は端を伸ばす",
                r"output_padding=\(1, 1\)",
            ),
            (
                {"kernel_size": 2, "stride": 2, "dilation": 2},
                "dilation は窓を飛ばす",
                r"dilation=\(2, 2\)",
            ),
            ({"kernel_size": 2, "stride": 2, "groups": 2}, "groups は Cin を分ける", r"groups=2"),
            (
                {"kernel_size": 2, "stride": 2, "bias": False},
                "bias 無しは実測に無い形",
                r"bias 無し",
            ),
        ],
    )
    def test_forms_outside_the_decomposition_are_rejected(
        self, kwargs: dict[str, object], why: str, match: str
    ) -> None:
        """分解が成り立たない形は fail loudly（黙って別の数値へ落とさない）。

        文言まで見るのは、拒否理由が**上流の実値**を名乗ることが読み手の次の一手を決めるから。
        型だけを見ていると、検査の側で値を変換してから報告する退行（`dilation=(2, 2)` を
        `dilation=(1, 1)` と報告する形）が素通りする。
        """
        source = nn.ConvTranspose2d(4, 4, **kwargs)
        with pytest.raises(NotImplementedError, match=match):
            SubPixelUpsample(source)


class TestFeatureFusionStage:
    """最終融合段の `scale_factor=2` → `size=(2H, 2W)` の同値。"""

    @staticmethod
    def _stage(fusion_hidden_size: int) -> nn.Module:
        modeling = pytest.importorskip("transformers.models.depth_anything.modeling_depth_anything")
        configuration = pytest.importorskip(
            "transformers.models.depth_anything.configuration_depth_anything"
        )
        config = configuration.DepthAnythingConfig(
            fusion_hidden_size=fusion_hidden_size,
            neck_hidden_sizes=[fusion_hidden_size] * 4,
        )
        stage = modeling.DepthAnythingFeatureFusionStage(config).eval()
        generator = torch.Generator().manual_seed(17)
        with torch.no_grad():
            for parameter in stage.parameters():
                parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.1)
        return stage

    @staticmethod
    def _features(channels: int) -> list[torch.Tensor]:
        """reassemble 相当の 4 本（解像度は降順・**非正方**で H/W の取り違えを落とす）。"""
        generator = torch.Generator().manual_seed(19)
        return [
            torch.randn((1, channels, height, width), generator=generator)
            for height, width in ((40, 56), (20, 28), (10, 14), (5, 7))
        ]

    def test_the_rewritten_stage_is_bit_exact(self) -> None:
        """差し替え版と原実装の出力が**ビット一致**（4 段とも）。"""
        stage = self._stage(6)
        features = self._features(6)

        with torch.no_grad():
            expected = stage(features)
            got = patch_depth_anything._feature_fusion_stage_forward(stage, features)

        assert len(got) == len(expected)
        for index, (left, right) in enumerate(zip(got, expected, strict=True)):
            assert tuple(left.shape) == tuple(right.shape), index
            assert torch.equal(left, right), index

    def test_the_last_stage_doubles_the_spatial_size(self) -> None:
        """最終段だけは入力の 2 倍の寸法になる（原実装の `scale_factor=2` と同じ）。"""
        stage = self._stage(6)
        features = self._features(6)

        with torch.no_grad():
            got = patch_depth_anything._feature_fusion_stage_forward(stage, features)

        assert [tuple(item.shape[2:]) for item in got] == [(10, 14), (20, 28), (40, 56), (80, 112)]

    def test_the_residual_resize_branch_stays_dead(self) -> None:
        """残差 resize 枝（原実装の `align_corners=False`）が**構造的に**死んでいること。

        段 `i` の出力は `hidden_states[i+1]` の寸法へ揃うので、段 `i+1` の残差とは必ず同形に
        なる。入力側の解像度を崩しても同じ（寸法は残差そのものから採るので一緒に動く）—
        `patch_depth_anything._feature_fusion_stage_forward` がガードを持たない根拠。
        """
        stage = self._stage(6)
        features = self._features(6)
        features[1] = features[1][:, :, :-3, :-5]

        with torch.no_grad():
            got = patch_depth_anything._feature_fusion_stage_forward(stage, features)

        assert [tuple(item.shape[2:]) for item in got] == [(10, 14), (17, 23), (40, 56), (80, 112)]

    @pytest.mark.parametrize(("height", "width"), [(5, 7), (1, 3), (13, 13)])
    def test_size_and_scale_factor_agree_bit_for_bit(self, height: int, width: int) -> None:
        """書き換えの土台（`align_corners=True` では倍率が使われない）を直に実測する。

        aten の `area_pixel_compute_scale` は `align_corners=True` のとき入出力の寸法から
        `(in−1)/(out−1)` を作るので、倍率指定と寸法指定は**同じ座標式**になる。ここが崩れると
        差し替え版のビット一致の主張ごと崩れる。
        """
        generator = torch.Generator().manual_seed(23)
        x = torch.randn((1, 3, height, width), generator=generator)
        by_scale = nn.functional.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        by_size = nn.functional.interpolate(
            x, size=(height * 2, width * 2), mode="bilinear", align_corners=True
        )
        assert torch.equal(by_scale, by_size)


class TestInterpolatePosEncoding:
    """位置埋め込み補間の門（恒等 or 落とす）。"""

    @staticmethod
    def _embeddings(num_positions: int, dim: int = 4) -> nn.Module:
        module = nn.Module()
        module.position_embeddings = nn.Parameter(torch.randn((1, num_positions + 1, dim)))
        return module

    def test_the_matching_shape_returns_the_table_itself(self) -> None:
        """パッチ数が一致する形では位置埋め込みを**そのまま**返す（原実装の枝と同一）。"""
        module = self._embeddings(9)
        embeddings = torch.zeros((1, 10, 4))

        got = patch_depth_anything._interpolate_pos_encoding(module, embeddings, 42, 42)

        assert got is module.position_embeddings

    @pytest.mark.parametrize(
        ("patches", "height", "width"),
        [(16, 42, 42), (9, 42, 56)],
    )
    def test_shapes_that_would_need_interpolation_are_rejected(
        self, patches: int, height: int, width: int
    ) -> None:
        """補間が要る形（原実装が bicubic を出す形）は落とす — 語彙に無い op を作らせない。"""
        module = self._embeddings(9)
        embeddings = torch.zeros((1, patches + 1, 4))

        with pytest.raises(NotImplementedError, match="bicubic"):
            patch_depth_anything._interpolate_pos_encoding(module, embeddings, height, width)


class TestAssertSupported:
    """差し替え版が前提にする構成の検査（外れても shape エラーにならないものだけ）。"""

    @staticmethod
    def _config(**overrides: object) -> object:
        configuration = pytest.importorskip(
            "transformers.models.depth_anything.configuration_depth_anything"
        )
        config = configuration.DepthAnythingConfig()
        for key, value in overrides.items():
            setattr(config, key, value)
        return config

    def test_the_reference_configuration_passes(self) -> None:
        model = nn.Module()
        model.config = self._config()
        model.config.backbone_config.apply_layernorm = True
        model.config.backbone_config.reshape_hidden_states = False

        patch_depth_anything.assert_supported(model)

    @pytest.mark.parametrize(
        ("attribute", "value", "match"),
        [
            ("depth_estimation_type", "metric", "depth_estimation_type"),
            ("backbone_apply_layernorm", False, "apply_layernorm"),
            ("backbone_reshape_hidden_states", True, "reshape_hidden_states"),
        ],
    )
    def test_configurations_outside_the_patch_are_rejected(
        self, attribute: str, value: object, match: str
    ) -> None:
        model = nn.Module()
        model.config = self._config()
        model.config.backbone_config.apply_layernorm = True
        model.config.backbone_config.reshape_hidden_states = False
        if attribute.startswith("backbone_"):
            setattr(model.config.backbone_config, attribute.removeprefix("backbone_"), value)
        else:
            setattr(model.config, attribute, value)

        with pytest.raises(ValueError, match=match):
            patch_depth_anything.assert_supported(model)


class TestPretrainedResolution:
    """位置埋め込みの補間が起きない唯一の入力解像度（正方 1 点）の読み取り。"""

    @staticmethod
    def _model(image_size: object) -> nn.Module:
        """`config.backbone_config.image_size` だけを持つ最小の骨格（合成で足りる）。"""
        model = nn.Module()
        model.config = SimpleNamespace(backbone_config=SimpleNamespace(image_size=image_size))
        return model

    def test_a_square_image_size_is_returned_as_an_int(self) -> None:
        resolution = patch_depth_anything.pretrained_resolution(self._model(518))

        assert resolution == 518
        assert isinstance(resolution, int)

    @pytest.mark.parametrize("image_size", [[518, 392], (518, 392), [518, 518]])
    def test_a_pair_shaped_image_size_is_rejected(self, image_size: object) -> None:
        """MUST: 非正方は差し替え版が持たない経路 — 対の綴りは値が揃っていても落とす。

        list / tuple で来た時点で「補間が起きない 1 点」の前提が別の経路（上流の
        `interpolate_pos_encoding`）へ移るので、値が正方でも受けない。
        """
        with pytest.raises(ValueError, match="正方のみ"):
            patch_depth_anything.pretrained_resolution(self._model(image_size))
