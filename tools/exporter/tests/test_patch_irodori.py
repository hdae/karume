"""`patch_irodori` の書き換えが原実装と同値であることの回帰テスト（実重み不要分）。

実重みでの端から端までの同値検証は `export_irodori.py`（emit のたびに全 golden ケースで
実測し、ビット一致でなければ落とす）。ここでは transformers があれば回る単体レベルの同値
だけを固定する。

MUST: 主張は**ビット一致**。差し替えたのは qkv の取り出し方だけで、演算順序も丸め方も
変わらない — 「差が小さい」で通す形にすると、割り方の取り違え（q/k/v の順序違い等）が
「たまたま近い値」で素通りしうる。
"""

from __future__ import annotations

import pytest
import torch

from karume import patch_irodori

modernbert = pytest.importorskip("transformers.models.modernbert.modeling_modernbert")


class TestTheSplitArgumentItself:
    """パッチの同値の根拠（rank 5 の unbind = 最終次元の 3 等分）を素の torch で固定する。"""

    def test_unbind_of_the_rank5_view_equals_three_last_dim_slices(self):
        batch, seq, heads, head_dim = 2, 5, 3, 4
        width = heads * head_dim
        qkv = torch.randn(batch, seq, 3 * width)

        original = qkv.view(batch, seq, 3, heads, head_dim).unbind(dim=-3)
        flat = tuple(
            qkv[..., start : start + width].view(batch, seq, heads, head_dim)
            for start in (0, width, 2 * width)
        )

        for index, (lhs, rhs) in enumerate(zip(original, flat, strict=True)):
            assert torch.equal(lhs, rhs), f"{index} 本目が一致しない"


@pytest.fixture
def restore_forward():
    """クラス属性の差し替えをテスト後に戻す（差し替えはプロセス全域）。"""
    original = modernbert.ModernBertAttention.forward
    applied = patch_irodori._APPLIED
    try:
        yield
    finally:
        modernbert.ModernBertAttention.forward = original
        patch_irodori._APPLIED = applied


def _tiny_attention(
    layer_idx: int,
) -> tuple[torch.nn.Module, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """小さな `ModernBertAttention` と入力 / RoPE 表を作る。

    NOTE: `num_hidden_layers` は 3 以上にする — 1 にすると `layer_types` が
    `["full_attention"]` だけになり、5.14.1 の config が RoPE パラメタを非階層形へ平坦化して
    strict dataclass の検証に落ちる（実重みの 25 層は 2 種混在なのでこの経路を踏まない）。
    """
    from transformers.models.modernbert.configuration_modernbert import ModernBertConfig

    config = ModernBertConfig(
        hidden_size=16,
        num_attention_heads=4,
        intermediate_size=32,
        num_hidden_layers=3,
        attention_dropout=0.0,
    )
    layer_type = config.layer_types[layer_idx]
    torch.manual_seed(0)
    attention = modernbert.ModernBertAttention(config=config, layer_idx=layer_idx).eval()
    hidden = torch.randn(1, 6, config.hidden_size)
    rotary = modernbert.ModernBertRotaryEmbedding(config=config)
    position_ids = torch.arange(hidden.shape[1]).unsqueeze(0)
    return attention, hidden, rotary(hidden, position_ids, layer_type)


class TestPatchedAttentionIsBitIdentical:
    @pytest.mark.parametrize("layer_idx", [0, 1])
    def test_the_patched_forward_reproduces_the_original(self, restore_forward, layer_idx):
        """full_attention（0 層目）と sliding_attention（1 層目）の両方で見る。

        sliding 側は `self.sliding_window` が attention interface へ渡る経路なので、引数を
        1 本落とした写し間違いはここで出る。
        """
        attention, hidden, position_embeddings = _tiny_attention(layer_idx)
        with torch.no_grad():
            expected, _ = attention(hidden, position_embeddings=position_embeddings)

        patch_irodori.apply_patches()
        # MUST: 差し替えが実際に効いたことを見る（`_APPLIED` が前段のテストから True のまま
        # 残っていると apply が no-op になり、原実装どうしの比較で恒真 PASS する）。
        assert modernbert.ModernBertAttention.forward is patch_irodori._flat_qkv_attention_forward
        with torch.no_grad():
            actual, _ = attention(hidden, position_embeddings=position_embeddings)

        assert torch.equal(actual, expected)

    def test_applying_twice_is_a_no_op(self, restore_forward):
        patch_irodori.apply_patches()
        first = modernbert.ModernBertAttention.forward
        patch_irodori.apply_patches()

        assert modernbert.ModernBertAttention.forward is first
        assert patch_irodori.patches_applied()
