"""patch_anima の各書き換えが原実装と同値であることの回帰テスト（実重み不要分）。

実モデルでの端から端までの同値検証は `export_anima.py --verify`（手動・実重み）。ここでは
diffusers があれば回る純関数 / 小モジュールレベルの同値だけを固定する。`--group anima` 無しの
pytest 実行では skip される。

MUST: 「差が小さい」ではなく**書き換えごとに主張の強さを分ける**。データ移動だけの書き換え
（upsample）はビット一致、演算順序が変わる書き換え（conv3d → conv2d / チャネル L2）は f64 で
桁が落ちること（= 丸めであって意味の違いでないこと）まで見る。
"""

from __future__ import annotations

import pytest
import torch
from torch import nn
from torch.nn import functional

qwenimage = pytest.importorskip("diffusers.models.autoencoders.autoencoder_kl_qwenimage")

from karume import patch_anima  # noqa: E402


class TestCausalConv3dToConv2d:
    @pytest.mark.parametrize(
        ("kernel", "padding"),
        [((3, 3, 3), (1, 1, 1)), ((1, 1, 1), (0, 0, 0)), ((3, 1, 1), (1, 0, 0))],
    )
    def test_t1_causal_conv3d_matches_the_flattened_conv2d(self, kernel, padding) -> None:
        """T=1 の因果 conv3d は「重みの時間方向 最終スライスによる conv2d」と一致する。

        f64 で測るのは、残差が「縮約順序の丸め差」か「本当の不一致」かを分けるため
        （前者なら f64 で桁が落ちる）。
        """
        torch.manual_seed(0)
        conv = qwenimage.QwenImageCausalConv3d(3, 4, kernel, padding=padding).to(torch.float64)
        x = torch.randn(1, 3, 5, 5, dtype=torch.float64)

        with torch.no_grad():
            reference = conv(x.unsqueeze(2))
            got = patch_anima._causal_conv3d_to_conv2d(conv)(x)

        assert reference.shape[2] == 1
        assert float((got - reference.squeeze(2)).abs().max()) < 1e-14

    def test_a_kernel_taller_than_the_causal_padding_is_rejected(self) -> None:
        """時間方向の因果パディングが足りない形（downsample3d の time_conv）は落とす。"""
        conv = qwenimage.QwenImageCausalConv3d(3, 3, (3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0))

        with pytest.raises(NotImplementedError, match="因果パディングが等価条件を外れる"):
            patch_anima._causal_conv3d_to_conv2d(conv)

    def test_padding_beyond_the_causal_amount_is_rejected(self) -> None:
        """パディング過多も落とす — 先頭のゼロにも窓が当たり、出力が 2 フレームになる。

        「不足だけ」を見る片側検査はこの形を素通りさせ、時間長 2 の出力の 1 枚目だけを
        conv2d の出力として黙って採ることになる（`2·pad_t=2, kt=2` で出力時間長 2）。
        """
        conv = qwenimage.QwenImageCausalConv3d(3, 3, (2, 1, 1), padding=(1, 0, 0))
        with torch.no_grad():
            reference = conv(torch.randn(1, 3, 1, 4, 4))
        assert reference.shape[2] == 2, "前提: 等価条件を外れた形は時間長が 1 にならない"

        with pytest.raises(NotImplementedError, match="因果パディングが等価条件を外れる"):
            patch_anima._causal_conv3d_to_conv2d(conv)

    def test_a_time_stride_other_than_one_is_rejected(self) -> None:
        """時間方向を間引く形は「重みの最終スライス」等価の主張が別物になる。"""
        conv = qwenimage.QwenImageCausalConv3d(3, 3, (3, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0))

        with pytest.raises(NotImplementedError, match="時間 stride"):
            patch_anima._causal_conv3d_to_conv2d(conv)


class TestUpsample:
    def test_the_expand_form_matches_nearest_exact_bit_for_bit(self) -> None:
        """×2 アップサンプルはデータ移動だけなので **ビット一致**（丸めの余地が無い）。"""
        torch.manual_seed(0)
        module = qwenimage.QwenImageUpsample(scale_factor=(2.0, 2.0), mode="nearest-exact")
        x = torch.randn(1, 3, 4, 5)
        reference = functional.interpolate(x, scale_factor=2.0, mode="nearest-exact")

        got = patch_anima._upsample_forward(module, x)

        assert torch.equal(got, reference)

    def test_a_non_integer_scale_is_rejected(self) -> None:
        """整数倍でない nearest-exact は添字の写像が複製と一致しない（黙って近似しない）。"""
        module = qwenimage.QwenImageUpsample(scale_factor=(1.5, 1.5), mode="nearest-exact")

        with pytest.raises(NotImplementedError, match="×2 以外"):
            patch_anima._upsample_forward(module, torch.randn(1, 3, 4, 4))

    def test_a_scalar_scale_factor_is_accepted(self) -> None:
        """スカラ表記の `scale_factor` も同じ複製になる（表記の吸収はここで済ませる）。"""
        torch.manual_seed(0)
        module = qwenimage.QwenImageUpsample(scale_factor=2, mode="nearest-exact")
        x = torch.randn(1, 3, 4, 5)

        got = patch_anima._upsample_forward(module, x)

        assert torch.equal(got, functional.interpolate(x, scale_factor=2.0, mode="nearest-exact"))

    def test_another_mode_is_rejected(self) -> None:
        """写像が違う `nearest` を素通りさせない（×2 は同じでも出力添字の写り方が別）。"""
        module = qwenimage.QwenImageUpsample(scale_factor=(2.0, 2.0), mode="nearest")

        with pytest.raises(NotImplementedError, match="nearest-exact 以外"):
            patch_anima._upsample_forward(module, torch.randn(1, 3, 4, 4))


class TestChannelL2:
    def test_it_matches_f_normalize(self) -> None:
        """チャネル方向 L2 は permute + 最終次元 sum で表しても `F.normalize` と一致する。

        `clamp_min(eps) → 除算` の順序も原実装のまま（`+eps` に置き換えない）。
        """
        torch.manual_seed(0)
        x = torch.randn(1, 6, 4, 5, dtype=torch.float64)

        got = patch_anima._l2_normalize_channels(x)

        assert float((got - functional.normalize(x, dim=1)).abs().max()) < 1e-14

    def test_a_zero_column_keeps_the_epsilon_floor(self) -> None:
        """ゼロ入力でも 0/0 にならない（`clamp_min` が効いている）ことを故障注入で確認する。"""
        zeros = torch.zeros(1, 3, 2, 2)

        got = patch_anima._l2_normalize_channels(zeros)

        assert torch.isfinite(got).all()
        assert torch.equal(got, zeros)


class TestRmsNormForward:
    def test_the_rank4_form_matches_the_rank5_original(self) -> None:
        """gamma を (1,C,1,1) へ整形した rank4 版が原実装（rank5 入力）と一致する。"""
        torch.manual_seed(0)
        module = qwenimage.QwenImageRMS_norm(6, images=False).to(torch.float64)
        with torch.no_grad():
            module.gamma.copy_(torch.rand_like(module.gamma) + 0.5)
        x = torch.randn(1, 6, 4, 5, dtype=torch.float64)
        with torch.no_grad():
            reference = module(x.unsqueeze(2)).squeeze(2)

        module.gamma = nn.Parameter(module.gamma.detach().reshape(1, -1, 1, 1))
        with torch.no_grad():
            got = patch_anima._rms_norm_forward(module, x)

        assert float((got - reference).abs().max()) < 1e-14

    def test_a_learned_bias_is_rejected(self) -> None:
        """原実装の `+ self.bias` を落とせるのは bias が Python の 0.0 のときだけ。"""
        module = qwenimage.QwenImageRMS_norm(4, bias=True)
        module.gamma = nn.Parameter(module.gamma.detach().reshape(1, -1, 1, 1))

        with pytest.raises(NotImplementedError, match="学習された bias"):
            patch_anima._rms_norm_forward(module, torch.randn(1, 4, 2, 2))


class TestResample:
    def test_a_feat_cache_is_rejected(self) -> None:
        """`feat_cache` は T=1 では参照されない — 渡されたら黙って捨てず落とす。"""
        module = qwenimage.QwenImageResample(4, mode="none")

        with pytest.raises(NotImplementedError, match="feat_cache"):
            patch_anima._resample_forward(module, torch.randn(1, 4, 2, 2), feat_cache=[None])


class TestLiftRopeBuffers:
    def test_it_moves_inv_freq_out_of_the_buffers(self) -> None:
        """`inv_freq` をバッファから素の属性へ降格する（定数畳み込みの葉にするため）。"""

        class Rotary(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.register_buffer("inv_freq", torch.arange(4.0), persistent=False)

        root = nn.Sequential(Rotary())

        lifted = patch_anima.lift_rope_buffers(root)

        assert lifted == 1
        assert "inv_freq" not in dict(root.named_buffers())
        assert torch.equal(root[0].inv_freq, torch.arange(4.0))

    def test_a_model_without_rope_buffers_is_rejected(self) -> None:
        """走査が空振りする形は落とす（恒真化の門 — 上流の属性名が変わると静かに壊れる）。"""
        with pytest.raises(ValueError, match="1 本も見つからない"):
            patch_anima._assert_rope_lifted(nn.Linear(2, 2), "テスト")


# ---- #21 波 T2: DiT の S 化（トークン形） -----------------------------------

#: 合成 DiT の形。**patch の高さ / 幅 / チャネル数を全部違う値にする** — 軸の取り違えは
#: 「実測形では対合になって偶然一致する」クラスの誤りなので（ACTIVE_DESIGN の Pitfalls）、
#: 巡回長 3 以上の並べ替えになる形でしか捕まらない。
TINY_DIT = {
    "in_channels": 3,
    "out_channels": 3,
    "num_attention_heads": 2,
    "attention_head_dim": 12,
    "num_layers": 1,
    "mlp_ratio": 1.0,
    "text_embed_dim": 8,
    "adaln_lora_dim": 4,
    "max_size": (4, 16, 12),
    "patch_size": (1, 2, 3),
    "rope_scale": (1.0, 4.0, 4.0),
    "concat_padding_mask": True,
    "extra_pos_embed_type": None,
}
#: latent の形（H / W も違う値にする — 正方だと h/w の取り違えが転置で隠れる）。
TINY_LATENT = (8, 12)


def _tiny_dit():
    from diffusers import CosmosTransformer3DModel

    torch.manual_seed(20260804)
    model = CosmosTransformer3DModel(**TINY_DIT)
    return model.to(torch.float32).eval()


def _tiny_inputs(model):
    torch.manual_seed(7)
    height, width = TINY_LATENT
    latents = torch.randn(1, model.config.in_channels, height, width)
    timestep = torch.full((1,), 0.37)
    embeds = torch.randn(1, 5, model.config.text_embed_dim)
    return latents, timestep, embeds


class TestDitTokens:
    """S 形ラッパ = 静的ラッパの入口 / 出口だけをホストへ出した形（#21 波 T2）。"""

    def test_the_token_form_composes_back_to_the_static_wrapper_bit_exactly(self) -> None:
        """`unpatchify ∘ S 形 ∘ patchify` は静的ラッパと**ビット同一**。

        両者は同じ演算列を同じ順で通る（差は「patchify / rope をどちら側で作るか」だけ）
        ので、丸め差すら出ないのが期待値。ここが緩むと実 GPU の主門
        （`packages/runtime/tests/e2e_anima_dyn_test.ts`）も緩む。
        """
        model = _tiny_dit()
        latents, timestep, embeds = _tiny_inputs(model)
        height, width = TINY_LATENT
        patch = tuple(model.config.patch_size)
        proj = patch_anima.dit_timesteps_proj(model, timestep)

        with torch.no_grad():
            expected = patch_anima.AnimaDit(model, height, width)(latents, proj, embeds)
            tokens = patch_anima.AnimaDitTokens(model)(
                patch_anima.dit_patchify(latents, patch),
                proj,
                embeds,
                *patch_anima.dit_rope_tables(model, height, width),
            )
            got = patch_anima.dit_unpatchify(tokens, height, width, patch)

        assert got.shape == expected.shape
        assert torch.equal(got, expected)

    def test_patchify_matches_the_upstream_patch_embed_chain(self) -> None:
        """patchify の並べ替えは上流 `CosmosPatchEmbed` の proj 前段と一致する（独立オラクル）。"""
        from diffusers.models.transformers.transformer_cosmos import CosmosPatchEmbed

        model = _tiny_dit()
        latents, _, _ = _tiny_inputs(model)
        height, width = TINY_LATENT
        patch = tuple(model.config.patch_size)
        embed = CosmosPatchEmbed(1, 1, patch)
        embed.proj = nn.Identity()
        padded = torch.cat([latents, latents.new_zeros(1, 1, height, width)], dim=1).unsqueeze(2)

        with torch.no_grad():
            reference = embed(padded).flatten(1, 3)
            got = patch_anima.dit_patchify(latents, patch)

        assert torch.equal(got, reference)

    def test_unpatchify_matches_an_index_level_oracle(self) -> None:
        """unpatchify の並べ替えを**添字の式**で独立に置く（view 連鎖の写しではない）。

        最終次元の並びは `(ph, pw, pt, C)`（patchify 側の `(C, pt, ph, pw)` と**別**）で、
        トークン添字は `(f·H' + h)·W' + w`。逆順に「直した」実装はここで赤くなる。
        """
        height, width = TINY_LATENT
        patch_t, patch_h, patch_w = (1, 2, 3)
        channels = 3
        rows, cols = height // patch_h, width // patch_w
        tokens = torch.arange(float(rows * cols * patch_t * patch_h * patch_w * channels)).reshape(
            1, rows * cols, patch_h * patch_w * patch_t * channels
        )

        got = patch_anima.dit_unpatchify(tokens, height, width, (patch_t, patch_h, patch_w))

        assert tuple(got.shape) == (1, channels, height, width)
        for channel in range(channels):
            for row in range(rows):
                for col in range(cols):
                    for inner_h in range(patch_h):
                        for inner_w in range(patch_w):
                            at = ((inner_h * patch_w + inner_w) * patch_t) * channels + channel
                            assert (
                                got[0, channel, row * patch_h + inner_h, col * patch_w + inner_w]
                                == (tokens[0, row * cols + col, at])
                            )

    def test_the_host_base_tables_rebuild_the_full_rope_table_bit_exactly(self) -> None:
        """軸別素表からの組み立てが `model.rope` の表と**ビット同一**（ホスト実装の正本）。

        torch の f32 `cos` / `sin` は正しい丸めと 1 ulp ずれることがあり、JS の `Math.cos`
        では再現できない。素表を焼くのはそのため（`dit_rope_base_tables` の doc）。
        """
        model = _tiny_dit()
        height, width = TINY_LATENT
        patch = tuple(model.config.patch_size)
        base = patch_anima.dit_rope_base_tables(model)
        rows, cols = height // patch[1], width // patch[2]

        expected_cos, expected_sin = patch_anima.dit_rope_tables(model, height, width)
        built_cos, built_sin = [], []
        for row in range(rows):
            for col in range(cols):
                built_cos.append(
                    torch.cat([base["cos_t"][0], base["cos_h"][row], base["cos_w"][col]] * 2)
                )
                built_sin.append(
                    torch.cat([base["sin_t"][0], base["sin_h"][row], base["sin_w"][col]] * 2)
                )

        assert torch.equal(torch.stack(built_cos), expected_cos[0, 0])
        assert torch.equal(torch.stack(built_sin), expected_sin[0, 0])

    def test_the_base_tables_cover_the_models_own_position_ceiling(self) -> None:
        """素表の行数は上流 `seq = arange(max(max_size))` の長さ（= 位置表の天井）。"""
        model = _tiny_dit()
        base = patch_anima.dit_rope_base_tables(model)
        rows = max(model.rope.max_size)

        assert set(base) == set(patch_anima.ROPE_BASE_KEYS)
        for name, table in base.items():
            assert table.shape[0] == rows, name
        assert base["cos_t"].shape[1] == model.rope.dim_t // 2
        assert base["cos_h"].shape[1] == model.rope.dim_h // 2

    def test_a_misaligned_base_slice_is_rejected(self) -> None:
        """恒真化の門: ブロック境界をずらした切り出しは「その軸で動かない」ので落ちる。"""
        model = _tiny_dit()
        base = patch_anima.dit_rope_base_tables(model)
        broken = dict(base)
        broken["sin_h"] = torch.zeros_like(base["sin_h"])

        with pytest.raises(AssertionError, match="動いていない"):
            patch_anima._assert_rope_base_shapes(
                broken,
                max(model.rope.max_size),
                (model.rope.dim_t // 2, model.rope.dim_h // 2, model.rope.dim_w // 2),
            )

    def test_the_token_graph_carries_no_resolution_dependent_initializer(self) -> None:
        """S 形の IR には**解像度依存の焼き込みが 1 本も無い**（波 T2 の設計前提そのもの）。

        静的形が焼いていたのは 3 本だけ（padding channel の `[1,1,H,W]` と rope 表 2 本の
        `[1,1,S,D]`）で、いずれも rank 4。重み側は linear / norm しか無いので rank ≤ 2 に
        収まる。したがって「rank ≥ 3 の initializer が 0 本」がそのまま invariant になり、
        S 依存の定数が畳み込まれたら（`Dim` の上限 = 16,384 で焼かれるので）必ず破れる。
        """
        from torch.export import Dim

        from karume.convert import PRESERVED_OP_PREFIXES_WITH_ATTENTION
        from karume.pipeline import export_module

        model = _tiny_dit()
        latents, timestep, embeds = _tiny_inputs(model)
        height, width = TINY_LATENT
        patch = tuple(model.config.patch_size)
        tokens = Dim("S", min=2, max=64)
        graph, _ = export_module(
            patch_anima.AnimaDitTokens(model),
            (
                patch_anima.dit_patchify(latents, patch),
                patch_anima.dit_timesteps_proj(model, timestep),
                embeds,
                *patch_anima.dit_rope_tables(model, height, width),
            ),
            dynamic_shapes=({1: tokens}, None, None, {2: tokens}, {2: tokens}),
            symbol_names=("S",),
            preserved=PRESERVED_OP_PREFIXES_WITH_ATTENTION,
        )

        assert graph.symbols == ["S"]
        assert [entry.name for entry in graph.inputs] == [
            "tokens",
            "timesteps_proj",
            "encoder_hidden_states",
            "rope_cos",
            "rope_sin",
        ]
        assert graph.inputs[0].shape == [1, "S", 24]
        assert graph.inputs[3].shape == [1, 1, "S", 12]
        baked = {
            name: graph.values[name].shape
            for name in graph.initializers
            if len(graph.values[name].shape) >= 3
        }
        assert baked == {}, f"解像度依存の焼き込みが残っている: {baked}"
