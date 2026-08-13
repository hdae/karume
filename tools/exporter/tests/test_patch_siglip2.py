"""`patch_siglip2` の書き換えが原実装と同値であることの回帰テスト（実重み不要分）。

実重みでの端から端までの同値検証は `export_siglip2.py --verify`（2 点評価）。ここでは
transformers を持たない環境でも回る単体レベルの同値を、**上流の forward を逐語で写した
骨格**（{@link StubEmbeddings} / {@link StubMapHead}）に対して固定する。

主張の強さは段ごとに違う（`patch_siglip2` の docstring）:

- 形の畳み込み（conv padding / 位置埋め込み）は **ビット一致**。演算列が 1 対 1 なので
  「差が小さい」で通す形にすると、位置埋め込みの並びの取り違えが素通りしうる。
- MAP head の q/k/v 明示化は **近い値**まで。`nn.MultiheadAttention` は q にだけ 1/√d を
  掛け、SDPA は q と k に対称に割るため。ただし tolerance は f32 の丸め幅（1e-5）に留めて、
  q/k/v の順序違いや head 分割の取り違えが通らないようにする。
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from karume import patch_siglip2

#: MAP head の同値許容差。`nn.MultiheadAttention` と SDPA のスケールの割り方の違いだけを
#: 許す幅で、q/k/v の入れ替えや head 分割の取り違えはこの桁に収まらない。
MAP_HEAD_ATOL = 1e-5

EMBED_DIM = 24
NUM_HEADS = 4
SEQUENCE = 7


class StubMapHead(nn.Module):
    """`SiglipMultiheadAttentionPoolingHead` の骨格（forward は上流の逐語写し）。"""

    def __init__(self, embed_dim: int = EMBED_DIM, num_heads: int = NUM_HEADS) -> None:
        super().__init__()
        self.probe = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.layernorm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 2 * embed_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(2 * embed_dim, embed_dim),
        )

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        batch_size = hidden_state.shape[0]
        probe = self.probe.repeat(batch_size, 1, 1)
        hidden_state = self.attention(probe, hidden_state, hidden_state)[0]
        residual = hidden_state
        hidden_state = self.layernorm(hidden_state)
        hidden_state = residual + self.mlp(hidden_state)
        return hidden_state[:, 0]


class StubEmbeddings(nn.Module):
    """`SiglipVisionEmbeddings` の骨格（forward は上流の逐語写し）。"""

    def __init__(self, image_size: int = 8, patch_size: int = 4, embed_dim: int = 6) -> None:
        super().__init__()
        self.patch_embedding = nn.Conv2d(
            in_channels=3,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            padding="valid",
        )
        positions = (image_size // patch_size) ** 2
        self.position_embedding = nn.Embedding(positions, embed_dim)
        self.register_buffer(
            "position_ids", torch.arange(positions).expand((1, -1)), persistent=False
        )

    def forward(
        self, pixel_values: torch.Tensor, interpolate_pos_encoding: bool = False
    ) -> torch.Tensor:
        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))
        embeddings = patch_embeds.flatten(2).transpose(1, 2)
        return embeddings + self.position_embedding(self.position_ids)


@pytest.fixture(autouse=True)
def restore_patched_attributes():
    """差し替えを全て戻す（クラス属性の差し替えはプロセス全域 — 次のケースの参照が汚れる）。"""
    forwards = (StubMapHead.forward, StubEmbeddings.forward)
    flags = (patch_siglip2._SHAPE_APPLIED, patch_siglip2._MAP_HEAD_APPLIED)
    try:
        yield
    finally:
        StubMapHead.forward, StubEmbeddings.forward = forwards
        patch_siglip2._SHAPE_APPLIED, patch_siglip2._MAP_HEAD_APPLIED = flags


class Config:
    """`SiglipVisionConfig` のうち `assert_supported` が見る欄だけ。"""

    def __init__(self, **overrides: object) -> None:
        self.hidden_act = patch_siglip2.EXPECTED_HIDDEN_ACT
        self.num_channels = 3
        self.image_size = 224
        self.patch_size = 16
        self.hidden_size = 768
        self.num_attention_heads = 12
        self.__dict__.update(overrides)


class TestAssertSupported:
    def test_accepts_the_shipped_vision_config(self):
        patch_siglip2.assert_supported(Config())

    def test_accepts_a_resolution_that_is_not_a_multiple_of_the_patch_size(self):
        """so400m は 384 / 14 = 27 グリッドで右下 6 画素が落ちる（上流ごと）— 書かれた経路。

        パッチ数も位置埋め込みの行数も `image_size // patch_size` の floor から決まるので、
        割り切れなくても両側が一致する（`_assert_position_ids_are_arange` の NOTE）。
        """
        patch_siglip2.assert_supported(
            Config(image_size=384, patch_size=14, hidden_size=1152, num_attention_heads=16)
        )

    @pytest.mark.parametrize(
        ("overrides", "message"),
        [
            ({"hidden_act": "gelu"}, "hidden_act"),
            ({"num_channels": 1}, "num_channels"),
            ({"num_attention_heads": 7}, "割り切れない"),
        ],
    )
    def test_fails_loudly_outside_the_written_path(self, overrides, message):
        with pytest.raises(ValueError, match=message):
            patch_siglip2.assert_supported(Config(**overrides))


class TestShapePatchesAreBitExact:
    """MUST: 主張はビット一致（`--verify` の 1 点目が実重みで同じ主張を採る）。"""

    @pytest.fixture
    def pixels(self):
        torch.manual_seed(0)
        return torch.randn(2, 3, 8, 8)

    def test_folded_position_embedding_is_bit_exact(self, pixels):
        embeddings = StubEmbeddings().eval()
        with torch.no_grad():
            reference = embeddings(pixels)

        patch_siglip2.apply_shape_patches(embeddings)
        with torch.no_grad():
            patched = embeddings(pixels)

        assert torch.equal(patched, reference)

    def test_conv_padding_becomes_the_default_overload_form(self, pixels):
        embeddings = StubEmbeddings().eval()

        patch_siglip2.apply_shape_patches(embeddings)

        assert embeddings.patch_embedding.padding == (0, 0)

    def test_interpolate_pos_encoding_is_refused(self, pixels):
        """固定解像度でしか同値でない経路は黙って通さない。"""
        embeddings = StubEmbeddings().eval()
        patch_siglip2.apply_shape_patches(embeddings)

        with pytest.raises(NotImplementedError, match="interpolate_pos_encoding"):
            embeddings(pixels, interpolate_pos_encoding=True)

    def test_shuffled_position_ids_fail_loudly(self):
        """並びが `arange` でなければ直加算は別の位置埋め込みになる（shape は合ったまま）。"""
        embeddings = StubEmbeddings().eval()
        embeddings.position_ids = embeddings.position_ids.flip(-1)

        with pytest.raises(ValueError, match="arange"):
            patch_siglip2.apply_shape_patches(embeddings)


class TestMapHeadSplit:
    @pytest.fixture
    def hidden(self):
        torch.manual_seed(0)
        return torch.randn(3, SEQUENCE, EMBED_DIM)

    def test_split_qkv_matches_the_packed_multihead_attention(self, hidden):
        head = StubMapHead().eval()
        with torch.no_grad():
            reference = head(hidden)

        patch_siglip2.apply_map_head_patch(head)
        with torch.no_grad():
            patched = head(hidden)

        assert torch.allclose(patched, reference, atol=MAP_HEAD_ATOL, rtol=0.0)

    def test_the_packed_projection_is_split_in_qkv_order(self):
        """MUST: 3 等分の順序は q → k → v（入れ替えても shape は合う）。"""
        head = StubMapHead().eval()
        packed = head.attention.in_proj_weight.detach().clone()

        patch_siglip2.apply_map_head_patch(head)

        for index, projection in enumerate((head.q_proj, head.k_proj, head.v_proj)):
            rows = slice(index * EMBED_DIM, (index + 1) * EMBED_DIM)
            assert torch.equal(projection.weight, packed[rows])

    def test_the_packed_attention_module_is_gone(self):
        """差し替え後に元の経路が残っていると、外れたことが誰にも見えなくなる。"""
        head = StubMapHead().eval()

        patch_siglip2.apply_map_head_patch(head)

        assert not hasattr(head, "attention")
        assert patch_siglip2.patches_applied()

    def test_probe_is_broadcast_not_repeated(self, hidden):
        """MUST: `expand` で広げる（`repeat` は batch>1 で未対応 op として残る）。

        `expand` は入力を読むだけなので、`probe` を書き換えると出力が変わる — 定数へ
        畳まれていないことを値で見る（形だけでは `repeat` との差が出ない）。
        """
        head = StubMapHead().eval()
        patch_siglip2.apply_map_head_patch(head)
        with torch.no_grad():
            before = head(hidden)
            head.probe += 1.0
            after = head(hidden)

        assert not torch.allclose(before, after)

    @pytest.mark.parametrize(
        ("mutate", "message"),
        [
            (lambda head: setattr(head, "attention", nn.Identity()), "MultiheadAttention"),
            (lambda head: setattr(head.attention, "batch_first", False), "batch_first"),
            (lambda head: setattr(head.attention, "add_zero_attn", True), "add_zero_attn"),
        ],
    )
    def test_unwritten_attention_shapes_fail_loudly(self, mutate, message):
        head = StubMapHead().eval()
        mutate(head)

        with pytest.raises((TypeError, ValueError), match=message):
            patch_siglip2.apply_map_head_patch(head)
