"""`wan/dit_patch.py`（DiT の S 形化）と `wan/export_dit.py` の約束事。

固定するのは、壊れても例外が出ず**数だけが静かにずれる**側（ADR 0118 決定 3・検収 段 2）:

- RoPE の実数化（interleave 形を rank 4 のまま）が上流の processor と**ビット一致**すること
- attn1 の processor の差し替えが上流の素の forward を変えないこと（参照の恒真化の門）
- patch 埋め込みの Linear 化と conv3d の差が縮約順の差の上界に収まること
- ホストの patchify / unpatchify / RoPE の素表が上流の並びとビット一致すること
- トークン形のラッパがホストの段と合成して上流の forward に戻ること
- S 形の export の形（記号 `S` 1 つ・入力名・op 集合・S 依存の焼き込みが無いこと）
- CPU の参照とパッチ後の eager の attention が flash 経路に固定されていること（S = 32,760 で MATH
  へ落ちると OOM — 黙って落ちずに止まる形）

合成モデル（乱数初期化の小さな `WanTransformer3DModel`）で回すものと、実重み（フィクスチャ
`wan_snapshot` — 無い機では SKIP）で回すものがある。合成モデルの形は **3 軸と C が全部違う値**にする
（軸の取り違えは正方では対合になって隠れる）。

`importorskip("diffusers")` は diffusers に依るテストの中（合成モデル・上流の部品を作る helper と
fixture）で受ける — モジュール直下に置くと、ケースの表・ホストの並び・flash の固定のような
diffusers に依らない不変条件まで既定の sync（`wan` グループ無し）で 1 本も回らない
（`test_vae_tiling.py` と同じ判断）。
"""

from __future__ import annotations

import pytest
import torch
from torch import nn
from torch.nn import functional
from torch.nn.attention import SDPBackend

from wan import dit_patch

#: 合成 DiT の形（head_dim 24 → RoPE の t / h / w は 8 / 8 / 8 次元）。
TINY_DIT = {
    "patch_size": (1, 2, 2),
    "num_attention_heads": 2,
    "attention_head_dim": 24,
    "in_channels": 3,
    "out_channels": 3,
    "text_dim": 8,
    "freq_dim": 16,
    "ffn_dim": 32,
    "num_layers": 2,
    "cross_attn_norm": True,
    "qk_norm": "rms_norm_across_heads",
    "eps": 1e-6,
    "rope_max_seq_len": 32,
}
#: 合成の潜在 `(F, H, W)` → 格子 F'·H'·W' = 2·3·5（3 軸とも違う値）。
TINY_LATENT = (2, 6, 10)

#: 製品の export（実重み）で観測した op 集合（`wan.export_dit` の要約の `ops`）。
#: 合成モデルも同じ構造なので同じ集合になる — op が増えたら語彙の判断が要る変更なので、
#: ここで止める。
EXPECTED_OPS = frozenset(
    {
        "add",
        "attention",
        "cat",
        "gelu_tanh",
        "layer_norm",
        "linear",
        "mul",
        "neg",
        "permute",
        "reshape",
        "rms_norm",
        "sigmoid",
        "slice",
    }
)


def _transformer_wan():
    """上流の `transformer_wan` モジュール（diffusers が無ければ呼んだテストを SKIP）。"""
    return pytest.importorskip("diffusers.models.transformers.transformer_wan")


def _tiny_dit() -> nn.Module:
    WanTransformer3DModel = _transformer_wan().WanTransformer3DModel  # noqa: N806

    torch.manual_seed(20261002)
    return WanTransformer3DModel(**TINY_DIT).to(torch.float32).eval()


def _tiny_inputs(model: nn.Module) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(7)
    latents = torch.randn(1, model.config.in_channels, *TINY_LATENT, generator=generator)
    timestep = torch.tensor([500], dtype=torch.int64)
    embeds = torch.randn(1, 7, model.config.text_dim, generator=generator)
    return latents, timestep, embeds


def _token_inputs(model: nn.Module, latents: torch.Tensor, timestep: torch.Tensor, embeds):
    patch = tuple(int(size) for size in model.config.patch_size)
    return (
        dit_patch.dit_patchify(latents, patch),
        dit_patch.dit_timesteps_proj(model, timestep),
        embeds,
        *dit_patch.dit_rope_tables(model.rope, tuple(latents.shape[2:])),
    )


class TestRealPairRotary:
    """interleave 形の RoPE の実数化（`real_pair_rotary` / 差し替えた processor）。"""

    def _attention_and_tables(self, dim: int, heads: int, latent: tuple[int, int, int]):
        upstream = _transformer_wan()
        WanAttention = upstream.WanAttention  # noqa: N806
        WanAttnProcessor = upstream.WanAttnProcessor  # noqa: N806
        WanRotaryPosEmbed = upstream.WanRotaryPosEmbed  # noqa: N806

        torch.manual_seed(11)
        attention = WanAttention(dim, heads, dim // heads, 1e-6, processor=WanAttnProcessor())
        rope = WanRotaryPosEmbed(dim // heads, (1, 2, 2), 1024)
        cos, sin = rope(torch.zeros(1, 1, *latent))
        hidden = torch.randn(1, cos.shape[1], dim)
        return attention.eval(), cos, sin, hidden

    @pytest.mark.parametrize(
        ("dim", "heads", "latent"),
        [(48, 2, (2, 6, 10)), (1536, 12, (2, 16, 24))],
        ids=["tiny", "wan2.1-1.3b-dims"],
    )
    def test_the_real_pair_form_matches_the_upstream_processor_bit_for_bit(
        self, dim: int, heads: int, latent: tuple[int, int, int]
    ) -> None:
        """`a − b = a + (−b)` と加算の可換で**ビット一致**。

        Wan2.1 の head_dim 128（RoPE の t / h / w = 44 / 42 / 42 次元）でも見る。
        """
        attention, cos, sin, hidden = self._attention_and_tables(dim, heads, latent)
        with torch.no_grad():
            expected = attention(hidden, None, None, (cos, sin))
            dit_patch.install_real_pair_processors(nn.ModuleList([_Holder(attention)]))
            got = attention(hidden, None, None, dit_patch.RealPairRope(cos, sin))

        # 恒真化の門: 差し替えが効いていないと `RealPairRope`（tuple）が上流の経路をそのまま通り、
        # 比較が自分自身との比較になる。
        assert type(attention.processor).__name__ == "WanRealPairAttnProcessor"
        assert torch.equal(got, expected)

    def test_real_pair_rotary_matches_the_upstream_formula_and_a_sign_fault_breaks_it(self) -> None:
        """独立オラクル: 上流の式（偶数 / 奇数列への strided 代入）と実数形をビットで比べる。"""
        _, cos, sin, _ = self._attention_and_tables(48, 2, (2, 6, 10))
        x = torch.randn(1, cos.shape[1], 2, 24)
        x1, x2 = x[..., 0::2], x[..., 1::2]
        expected = torch.empty_like(x)
        expected[..., 0::2] = x1 * cos[..., 0::2] - x2 * sin[..., 1::2]
        expected[..., 1::2] = x1 * sin[..., 1::2] + x2 * cos[..., 0::2]

        assert torch.equal(dit_patch.real_pair_rotary(x, cos, sin), expected)
        assert not torch.allclose(dit_patch.real_pair_rotary(x, cos, -sin), expected)

    def test_tables_in_another_dtype_than_x_are_rejected(self) -> None:
        """上流は `.type_as(x)` で x の dtype へ戻すが、実数形はその変換を写していない — dtype が
        割れた呼び出しは上流と違う dtype の値を黙って返す代わりに止まる。"""
        x = torch.randn(1, 3, 2, 8, dtype=torch.bfloat16)
        table = torch.ones(1, 3, 1, 8)

        with pytest.raises(NotImplementedError, match="type_as"):
            dit_patch.real_pair_rotary(x, table, table)
        assert dit_patch.real_pair_rotary(x, table.bfloat16(), table.bfloat16()).dtype == x.dtype

    def test_the_upstream_path_is_untouched_after_the_swap(self) -> None:
        """差し替え後も素の tuple の表は上流の実装へ委ねる（参照を採る順序を問わない根拠）。"""
        attention, cos, sin, hidden = self._attention_and_tables(48, 2, (2, 6, 10))
        with torch.no_grad():
            before = attention(hidden, None, None, (cos, sin))
            dit_patch.install_real_pair_processors(nn.ModuleList([_Holder(attention)]))
            after = attention(hidden, None, None, (cos, sin))

        assert torch.equal(after, before)

    def test_the_swap_is_idempotent(self) -> None:
        attention, *_ = self._attention_and_tables(48, 2, (1, 2, 2))
        holder = nn.ModuleList([_Holder(attention)])
        dit_patch.install_real_pair_processors(holder)
        first = attention.processor
        dit_patch.install_real_pair_processors(holder)

        assert attention.processor is first

    def test_real_pair_tables_on_cross_attention_are_rejected(self) -> None:
        attention, cos, sin, hidden = self._attention_and_tables(48, 2, (1, 2, 2))
        dit_patch.install_real_pair_processors(nn.ModuleList([_Holder(attention)]))
        with pytest.raises(NotImplementedError, match="self-attention"):
            attention(hidden, hidden, None, dit_patch.RealPairRope(cos, sin))


class _Holder(nn.Module):
    """`install_real_pair_processors` が受ける「`attn1` を持つブロック」の最小形。"""

    def __init__(self, attention: nn.Module) -> None:
        super().__init__()
        self.attn1 = attention


class TestPatchEmbedding:
    def test_the_linear_shares_storage_with_the_conv(self) -> None:
        """丸め（fake-quant）をラッパへ掛けると上流の conv にも届く（参照だけ元の重みにしない）。

        Linear の重みは conv の重みの view・bias は同じ Parameter。
        """
        model = _tiny_dit()
        linear = dit_patch.patch_embedding_linear(model.patch_embedding)
        with torch.no_grad():
            linear.weight.mul_(0.5)

        assert torch.equal(model.patch_embedding.weight.reshape(linear.weight.shape), linear.weight)
        assert linear.bias is model.patch_embedding.bias

    def test_an_overlapping_conv_is_rejected(self) -> None:
        conv = nn.Conv3d(3, 8, kernel_size=(1, 2, 2), stride=(1, 1, 1))
        with pytest.raises(NotImplementedError, match="k = s"):
            dit_patch.patch_embedding_linear(conv)


class TestHostFunctions:
    def test_patchify_matches_the_upstream_patch_embedding_order(self) -> None:
        """独立オラクル: 上流と同じ `Conv3d(k = s)` に 1-hot の重み → 窓の並び `(c,pt,ph,pw)`。"""
        from wan.dit_host_fixture import onehot_patchify

        latents = torch.randn(1, 3, *TINY_LATENT)
        got = dit_patch.dit_patchify(latents, (1, 2, 2))

        assert torch.equal(got, onehot_patchify(latents, (1, 2, 2)))

    def test_unpatchify_matches_an_index_level_oracle(self) -> None:
        """出口の最終次元は `(pt, ph, pw, c)`・トークン添字は `(f·H' + h)·W' + w`。

        添字の式で独立に置く（view 連鎖の写しではない）。
        """
        frames, height, width = TINY_LATENT
        patch_t, patch_h, patch_w = (1, 2, 2)
        channels = 3
        grid = (frames // patch_t, height // patch_h, width // patch_w)
        count = grid[0] * grid[1] * grid[2]
        tokens = torch.arange(float(count * patch_t * patch_h * patch_w * channels)).reshape(
            1, count, -1
        )

        got = dit_patch.dit_unpatchify(tokens, TINY_LATENT, (patch_t, patch_h, patch_w))

        assert tuple(got.shape) == (1, channels, frames, height, width)
        for c in range(channels):
            for f in range(frames):
                for h in range(height):
                    for w in range(width):
                        token = ((f // patch_t) * grid[1] + h // patch_h) * grid[2] + w // patch_w
                        inner = ((f % patch_t) * patch_h + h % patch_h) * patch_w + w % patch_w
                        assert got[0, c, f, h, w] == tokens[0, token, inner * channels + c]

    def test_the_base_tables_rebuild_the_upstream_rope_table_bit_exactly(self) -> None:
        """素表（偶数列）を並べ替えて 2 回ずつ並べると、上流 `WanRotaryPosEmbed` の表に戻る。

        設定は実寸（head_dim 128・位置 1024）。
        """
        WanRotaryPosEmbed = _transformer_wan().WanRotaryPosEmbed  # noqa: N806

        rope = WanRotaryPosEmbed(128, (1, 2, 2), 1024)
        latent = (3, 10, 14)
        base = dit_patch.dit_rope_base_tables(rope)
        cos, sin = dit_patch.dit_rope_tables(rope, latent)
        rows: list[tuple[torch.Tensor, torch.Tensor]] = []
        for f in range(latent[0]):
            for h in range(latent[1] // 2):
                for w in range(latent[2] // 2):
                    rows.append(
                        tuple(  # type: ignore[arg-type]
                            torch.cat(
                                [
                                    base[f"{kind}_t"][f],
                                    base[f"{kind}_h"][h],
                                    base[f"{kind}_w"][w],
                                ]
                            ).repeat_interleave(2)
                            for kind in ("cos", "sin")
                        )
                    )

        assert torch.equal(torch.stack([row[0] for row in rows]), cos[0, :, 0])
        assert torch.equal(torch.stack([row[1] for row in rows]), sin[0, :, 0])

    def test_the_base_tables_have_the_upstream_ceiling_and_widths(self) -> None:
        WanRotaryPosEmbed = _transformer_wan().WanRotaryPosEmbed  # noqa: N806

        base = dit_patch.dit_rope_base_tables(WanRotaryPosEmbed(128, (1, 2, 2), 1024))

        assert set(base) == set(dit_patch.ROPE_BASE_KEYS)
        assert {name: tuple(table.shape) for name, table in base.items()} == {
            "cos_t": (1024, 22),
            "sin_t": (1024, 22),
            "cos_h": (1024, 21),
            "sin_h": (1024, 21),
            "cos_w": (1024, 21),
            "sin_w": (1024, 21),
        }

    def test_a_misaligned_base_slice_is_rejected(self) -> None:
        """恒真化の門: ブロック境界をずらした切り出しは「その軸で動かない」ので落ちる。"""
        WanRotaryPosEmbed = _transformer_wan().WanRotaryPosEmbed  # noqa: N806

        base = dit_patch.dit_rope_base_tables(WanRotaryPosEmbed(24, (1, 2, 2), 32))
        broken = dict(base)
        broken["sin_h"] = torch.zeros_like(base["sin_h"])

        with pytest.raises(AssertionError, match="動いていない"):
            dit_patch._assert_rope_base(broken, 32, (4, 4, 4))


class TestDitTokens:
    def test_the_token_form_composes_back_to_the_upstream_forward_bit_exactly(self) -> None:
        """`unpatchify ∘ S 形 ∘ patchify` が上流の forward とビット同一（合成モデル）。"""
        model = _tiny_dit()
        latents, timestep, embeds = _tiny_inputs(model)
        with torch.no_grad():
            expected = dit_patch.reference_dit(model, latents, timestep, embeds)
            wrapper = dit_patch.WanDitTokens(model)
            tokens = wrapper(*_token_inputs(model, latents, timestep, embeds))
            hidden = model.patch_embedding(latents).flatten(2).transpose(1, 2)
            trunk = wrapper.forward_hidden(
                hidden, *_token_inputs(model, latents, timestep, embeds)[1:]
            )

        assert torch.equal(dit_patch.dit_unpatchify(trunk, TINY_LATENT, (1, 2, 2)), expected)
        got = dit_patch.dit_unpatchify(tokens, TINY_LATENT, (1, 2, 2))
        assert torch.allclose(got, expected, rtol=0, atol=1e-5)

    def test_the_layers_form_ends_with_the_product_output(self) -> None:
        """計測用（層別の出口）の最後の出力は製品のラッパと同じで、前の出力は上流の各ブロックの出力。"""
        model = _tiny_dit()
        latents, timestep, embeds = _tiny_inputs(model)
        inputs = _token_inputs(model, latents, timestep, embeds)
        with torch.no_grad():
            product = dit_patch.WanDitTokens(model)(*inputs)
            layers = dit_patch.WanDitTokensLayers(model)(*inputs)
            hidden = model.patch_embedding(latents).flatten(2).transpose(1, 2)
            _, blocks = dit_patch.reference_dit_layers(model, latents, timestep, embeds)

        assert len(layers) == len(model.blocks) + 1
        assert torch.equal(layers[-1], product)
        assert hidden.shape == layers[0].shape
        for got, want in zip(layers[:-1], blocks, strict=True):
            assert torch.allclose(got, want, rtol=0, atol=1e-5)

    def test_rounding_reaches_every_parameter_but_not_the_rope_tables(self) -> None:
        """f16 の丸めはラッパ経由で上流の全パラメータに届き、RoPE の表には届かない。

        全パラメータには conv の patch 埋め込み（ラッパの Linear はその view）も入る。
        """
        from wan.export_dit import round_to_f16

        model = _tiny_dit()
        rope_before = {name: buffer.clone() for name, buffer in model.rope.named_buffers()}
        round_to_f16(model, dit_patch.WanDitTokens(model))

        weight = model.patch_embedding.weight
        assert torch.equal(weight, weight.half().float())
        for name, buffer in model.rope.named_buffers():
            assert torch.equal(buffer, rope_before[name]), name


class TestExport:
    def test_the_token_graph_has_one_symbol_and_no_resolution_dependent_constant(self) -> None:
        """S 形の IR: 記号は `S` 1 つ・入力名と形・op 集合・S 依存の焼き込みが 1 本も無いこと。

        rank ≥ 3 の initializer は変調の表（`scale_shift_table` `[1,6,dim]` / `[1,2,dim]`）だけ。
        RoPE の表や潜在の形に依る定数が畳み込まれたら（`Dim` の上限で焼かれるので）ここで破れる。
        """
        from karume.convert import PRESERVED_OP_PREFIXES_WITH_ATTENTION
        from karume.pipeline import export_module
        from karume.verify import assert_op_contracts, assert_runtime_support
        from wan.export_dit import INPUT_NAMES, dynamic_shapes

        model = _tiny_dit()
        latents, timestep, embeds = _tiny_inputs(model)
        graph, tensors = export_module(
            dit_patch.WanDitTokens(model),
            _token_inputs(model, latents, timestep, embeds),
            dynamic_shapes=dynamic_shapes(),
            symbol_names=("S",),
            preserved=PRESERVED_OP_PREFIXES_WITH_ATTENTION,
        )

        assert graph.symbols == ["S"]
        assert [entry.name for entry in graph.inputs] == list(INPUT_NAMES)
        assert [list(entry.shape) for entry in graph.inputs] == [
            [1, "S", 12],
            [1, 16],
            [1, 7, 8],
            [1, "S", 1, 24],
            [1, "S", 1, 24],
        ]
        assert [graph.values[name].shape for name in graph.outputs] == [[1, "S", 12]]
        assert set(graph.required_ops) == EXPECTED_OPS
        assert {name for name, tensor in tensors.items() if tensor.dim() >= 3} == {
            "blocks.0.scale_shift_table",
            "blocks.1.scale_shift_table",
            "scale_shift_table",
        }
        assert_runtime_support(graph)
        assert_op_contracts(graph)

    def test_case_names_are_unique_and_every_role_has_a_case(self) -> None:
        from wan.export_dit import CASES

        names = [spec.name((1, 2, 2)) for spec in CASES]
        assert len(set(names)) == len(names)
        assert {spec.role for spec in CASES} == {
            "band",
            "accept",
            "growth",
            "full-band",
            "full-accept",
        }
        # 例示入力（先頭）は小さい方（`torch.export` はトレースで 1 回 forward を回す）。
        assert names[0].startswith("band-s00192")

    def test_full_size_cases_have_their_own_seeds(self) -> None:
        """実寸のケースは 832×480 の 33 フレーム（S = 14,040 — ADR 0118 段 3）と 81 フレーム
        （S = 32,760 — 段 8）の 8 本ずつで、帯を S ごとに独立に導くため seed が段 2 のどのケースとも
        違う（決定用と受入れの間でも、S の間でも違う）。"""
        from wan.export_dit import CASES

        full = [spec for spec in CASES if spec.full_size]
        assert [spec.name((1, 2, 2)) for spec in full] == [
            "full-band-s14040-t0999",
            "full-band-s14040-t0999-2",
            "full-band-s14040-t0750",
            "full-band-s14040-t0500",
            "full-band-s14040-t0250",
            "full-band-s14040-t0113",
            "full-accept-s14040-t0999",
            "full-accept-s14040-t0600",
            "full-band-s32760-t0999",
            "full-band-s32760-t0999-2",
            "full-band-s32760-t0750",
            "full-band-s32760-t0500",
            "full-band-s32760-t0250",
            "full-band-s32760-t0113",
            "full-accept-s32760-t0999",
            "full-accept-s32760-t0600",
        ]
        others = {spec.seed for spec in CASES if not spec.full_size}
        assert len({spec.seed for spec in full}) == len(full)
        assert not others & {spec.seed for spec in full}

    def test_only_two_full_size_cases_keep_block_outputs(self) -> None:
        """実寸の各ブロックの出力（S = 14,040 の 1 ケース 2.6 GB）は S = 14,040 の決定用 1 本と
        受入れ 1 本（どちらも t = 999）だけが持つ。S = 32,760 は 1 本も持たない（層別の probe が
        readback staging 約 6 GB で B570 に載らない）。S = 192 / 768 は全ケースが持つ（層ごとの
        記録の相手）。"""
        from wan.export_dit import CASES

        assert [spec.name((1, 2, 2)) for spec in CASES if spec.full_size and spec.blocks] == [
            "full-band-s14040-t0999",
            "full-accept-s14040-t0999",
        ]
        assert all(spec.blocks for spec in CASES if not spec.full_size)

    def test_the_largest_case_sits_exactly_on_the_declared_ceiling_of_s(self) -> None:
        """`Dim("S")` の上限 32,760 は 81 フレームの格子 21·30·52 ちょうど。全ケースの S が
        export の値域に入り、最大のケースが上限そのもの（受理集合の端を golden が覆う）。

        値域は宣言の定数ではなく ExportedProgram の `range_constraints`（export が実際に受けた
        制約）から読む。
        """
        from wan.export_dit import CASES, DIT_SYM_MAX, dynamic_shapes

        model = _tiny_dit()
        latents, timestep, embeds = _tiny_inputs(model)
        program = torch.export.export(
            dit_patch.WanDitTokens(model),
            _token_inputs(model, latents, timestep, embeds),
            dynamic_shapes=dynamic_shapes(),
        )
        ranges = {
            (int(bound.lower), int(bound.upper)) for bound in program.range_constraints.values()
        }
        tokens = [
            frames * (height // 2) * (width // 2)
            for frames, height, width in (spec.latent_shape for spec in CASES)
        ]

        assert ranges == {(2, 32_760)}
        assert DIT_SYM_MAX == 21 * 30 * 52
        assert all(2 <= count <= 32_760 for count in tokens)
        assert max(tokens) == 32_760


class TestFlashAttentionOnly:
    """CPU の参照とパッチ後の eager の attention を flash 経路に固定すること
    （`dit_patch.flash_attention_only` — S = 32,760 で MATH へ落ちるとスコア行列 1 枚が
    51.5 GB）。"""

    @staticmethod
    def _diffusers_layout(sequence: int, dtype: torch.dtype) -> torch.Tensor:
        """Wan2.1 の attention の形（heads 12・head_dim 128）を diffusers の native backend と同じ
        並び（`[B,S,H,D]` を permute した `[B,H,S,D]` — 最終次元だけが連続）で。値は選択に効かない
        ので未初期化（実寸でもページを触らない）。"""
        return torch.empty(1, sequence, 12, 128, dtype=dtype).permute(0, 2, 1, 3)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=["f32", "f64"])
    @pytest.mark.parametrize("sequence", [192, 14_040, 32_760])
    def test_the_default_choice_is_flash_for_the_wan_attention_shapes(
        self, dtype: torch.dtype, sequence: int
    ) -> None:
        """固定が数値を変えない前提: 固定しなくても既定の選択は flash（self-attention `S×S` と
        cross-attention `S×512`）。torch の更新でここが崩れたら、既存の golden は MATH の値で、固定
        すると数値が変わる — golden の作り直しが要る。"""
        query = self._diffusers_layout(sequence, dtype)
        for key in (query, self._diffusers_layout(512, dtype)):
            assert torch._fused_sdp_choice(query, key, key) == int(SDPBackend.FLASH_ATTENTION)

    def test_an_input_flash_cannot_take_raises_instead_of_falling_back_to_math(self) -> None:
        """故障注入: 最終次元が非連続の q（flash が受けない）は、既定では MATH へ黙って落ちて通り、
        固定の下では RuntimeError で止まる。"""
        query = torch.randn(1, 2, 24, 30).transpose(-1, -2)

        assert torch._fused_sdp_choice(query, query, query) == int(SDPBackend.MATH)
        functional.scaled_dot_product_attention(query, query, query)
        with dit_patch.flash_attention_only(), pytest.raises(RuntimeError):
            functional.scaled_dot_product_attention(query, query, query)

    def test_the_fixation_leaves_the_reference_bit_identical(self) -> None:
        """既定が flash の間は、固定した参照と素の forward がビット一致（合成モデル）。"""
        model = _tiny_dit()
        latents, timestep, embeds = _tiny_inputs(model)
        with torch.no_grad():
            plain = model(
                hidden_states=latents,
                timestep=timestep,
                encoder_hidden_states=embeds,
                return_dict=False,
            )[0]
            fixed = dit_patch.reference_dit(model, latents, timestep, embeds)

        assert torch.equal(fixed, plain)

    def test_every_reference_and_eager_attention_runs_with_math_disabled(self, monkeypatch) -> None:
        """f32 の参照・f64 の参照・パッチ後の eager（`export_dit.eager_report`）の SDPA の呼び出しが
        全部、MATH を切った状態で走る（固定を外すとここで破れる）。"""
        from wan.export_dit import Case, CaseSpec, eager_report, round_to_f16

        states: list[bool] = []
        original = functional.scaled_dot_product_attention

        def recording(*args, **kwargs):
            states.append(torch.backends.cuda.math_sdp_enabled())
            return original(*args, **kwargs)

        model = _tiny_dit()
        latents, timestep, embeds = _tiny_inputs(model)
        wrapper = dit_patch.WanDitTokens(model)
        round_to_f16(model, wrapper)
        monkeypatch.setattr(functional, "scaled_dot_product_attention", recording)
        with torch.no_grad():
            reference = dit_patch.reference_dit(model, latents, timestep, embeds)
            case = Case(
                name="tiny",
                spec=CaseSpec("band", TINY_LATENT, 500, 7, 0),
                inputs=_token_inputs(model, latents, timestep, embeds),
                latents=latents,
                timestep=timestep,
                reference=reference,
                reference_blocks=[],
                reference_seconds=0.0,
            )
            eager_report(wrapper, model, case)
            dit_patch.reference_dit_f64(model.double(), latents, timestep, embeds)

        # 1 forward で self / cross の 2 本 × 2 層。3 経路で 12 本以上（eager の trunk の回し直し
        # 分だけ増える）。
        assert len(states) >= 12
        assert not any(states)
        assert torch.backends.cuda.math_sdp_enabled()


class TestFloat64Reference:
    """実寸の f64 の参照（`dit_patch.float64_forward`）— 活性が全部 f64 で回ること。"""

    def _rounded_f64(self) -> nn.Module:
        from wan.export_dit import round_to_f16

        model = _tiny_dit()
        round_to_f16(model, dit_patch.WanDitTokens(model))
        return model.double()

    def test_the_f64_forward_stays_in_f64_and_restores_tensor_float(self) -> None:
        """出力は f64・f32 の参照とは丸めの差だけ違い、抜けた後の `Tensor.float()` は元の
        f32 化。"""
        model = self._rounded_f64()
        latents, timestep, embeds = _tiny_inputs(model)
        with torch.no_grad():
            got = dit_patch.reference_dit_f64(model, latents, timestep, embeds)
            expected = dit_patch.reference_dit(model.float(), latents, timestep, embeds)

        assert got.dtype == torch.float64
        assert 0 < float((got - expected.double()).abs().max()) < 1e-5 * float(got.abs().max())
        assert torch.ones(2, dtype=torch.float64).float().dtype == torch.float32
        assert "float" not in vars(torch.Tensor)

    def test_upstream_f32_casts_left_in_place_are_caught(self, monkeypatch) -> None:
        """`.float()` の素通しを外すと（上流の FP32LayerNorm と残差の f32 化がそのまま走ると）
        止まる。"""
        from contextlib import nullcontext

        model = self._rounded_f64()
        latents, timestep, embeds = _tiny_inputs(model)
        monkeypatch.setattr(dit_patch, "_float_keeps_float64", nullcontext)
        with torch.no_grad(), pytest.raises(AssertionError, match="f64 でない値"):
            dit_patch.reference_dit_f64(model, latents, timestep, embeds)

    @pytest.mark.parametrize("shape", [(2, 3), (16,), (1, 4)], ids=["2x3", "freq_dim", "1x4"])
    def test_a_small_f32_value_outside_the_timestep_sinusoid_is_caught(
        self, shape: tuple[int, ...]
    ) -> None:
        """許すのは時刻の sinusoid の形（`[1,1]` / `[half]` / `[1,half]` / `[1,freq_dim]`）だけ。

        合成モデルの freq_dim は 16 — 要素数 16 以下でも形が違えば止まる（大きさで許すと素通り
        する）。
        """
        model = self._rounded_f64()
        with pytest.raises(AssertionError, match="f64 でない値"), dit_patch.float64_forward(model):
            torch.ones(shape)

    def test_a_model_left_in_f32_is_rejected(self) -> None:
        model = _tiny_dit()
        latents, timestep, embeds = _tiny_inputs(model)
        with torch.no_grad(), pytest.raises(AssertionError, match="float32"):
            dit_patch.reference_dit_f64(model, latents, timestep, embeds)


# ---- 実重み（pin した revision — 無い機では SKIP） ----------------------------------


@pytest.fixture(scope="module")
def wan_transformer(wan_snapshot) -> nn.Module:
    """実重みの DiT（CPU f32・素のまま）。テストの間で丸め等の書き換えをしない MUST。"""
    pytest.importorskip("diffusers")
    from wan.export_dit import load_transformer

    return load_transformer()


class TestRealWeights:
    def test_the_patched_path_matches_the_upstream_forward(self, wan_transformer) -> None:
        """実重み・S = 192（非正方 2·8·12）で、patch 埋め込み以外の書き換えは上流と**ビット一致**。

        patch 埋め込みの Linear 化だけは縮約順が conv3d と違いうる（実測はこの機の torch CPU で差
        0）ので、差は縮約順の違いの上界 `2·γ_{K+1}·Σ|w·x|`（`γ_n = n·u / (1 − n·u)`・`u = 2⁻²⁴`・K =
        64 と bias）で押さえる — 実装の誤り（並びの取り違え）は値そのものの大きさで出る。
        """
        from wan.pipeline_ref import pad_text_embeds

        model = wan_transformer
        generator = torch.Generator().manual_seed(1)
        latents = torch.randn(1, 16, 2, 16, 24, generator=generator)
        timestep = torch.tensor([500], dtype=torch.int64)
        embeds = pad_text_embeds(torch.randn(37, 4096, generator=generator))
        inputs = _token_inputs(model, latents, timestep, embeds)
        with torch.no_grad():
            expected = dit_patch.reference_dit(model, latents, timestep, embeds)
            wrapper = dit_patch.WanDitTokens(model)
            hidden = model.patch_embedding(latents).flatten(2).transpose(1, 2)
            trunk = wrapper.forward_hidden(hidden, *inputs[1:])
            embedded = wrapper.patch_embedding(inputs[0])
            bound_terms = inputs[0].abs() @ wrapper.patch_embedding.weight.abs().T
            bound_terms = bound_terms + wrapper.patch_embedding.bias.abs()

        assert torch.equal(dit_patch.dit_unpatchify(trunk, (2, 16, 24), (1, 2, 2)), expected)
        unit = 2.0**-24
        terms = inputs[0].shape[-1] + 1
        gamma = terms * unit / (1 - terms * unit)
        assert bool(((embedded - hidden).abs() <= 2 * gamma * bound_terms).all())
