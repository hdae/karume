"""`patch_irodori` の書き換えが原実装と同値であることの回帰テスト（実重み不要分）。

実重みでの端から端までの同値検証は `export_irodori.py`（emit のたびに全 golden ケースで
実測し、ビット一致でなければ落とす）。ここでは transformers / Irodori 実装 clone があれば
回る単体レベルの同値だけを固定する。

MUST: 主張は**ビット一致**。差し替えたのは qkv の取り出し方・RoPE の複素数を使わない書き方・
RMSNorm の weight 分割・AdaLN の weightless RMS の畳み込みだけで、演算順序も丸め方も
変わらない — 「差が小さい」で通す形にすると、取り違え（q/k/v の順序違い・cos と sin の
入れ替え・head ごと weight の転置・shift と scale の入れ替え）が「たまたま近い値」で
素通りしうる。
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
    """クラス属性の差し替えをテスト後に戻す（差し替えはプロセス全域）。

    `apply_patches` は `irodori_tts`（git 追跡外の clone — conftest が `sys.path` へ足す）も
    差し替えるので、無い環境ではこのフィクスチャを使うテストだけを skip する。
    """
    irodori_model = pytest.importorskip("irodori_tts.model")
    original = modernbert.ModernBertAttention.forward
    original_rope = irodori_model.apply_rotary_emb
    original_norm = irodori_model.RMSNorm.forward
    original_adaln = irodori_model.LowRankAdaLN.forward
    applied = patch_irodori._APPLIED
    try:
        yield irodori_model
    finally:
        modernbert.ModernBertAttention.forward = original
        irodori_model.apply_rotary_emb = original_rope
        irodori_model.RMSNorm.forward = original_norm
        irodori_model.LowRankAdaLN.forward = original_adaln
        patch_irodori._APPLIED = applied
        patch_irodori._ORIGINAL_APPLY_ROTARY_EMB = None


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


class TestPatchedRotaryEmbedding:
    """complex 表を実数対へ開いた RoPE が原実装を再現すること。

    ビット一致は**形依存**（`_real_pair_apply_rotary_emb` の NOTE）。実重みの幾何
    （head_dim 64）はビット一致を要求し、ずれる側（head_dim 8）は **1 ulp の上限**で固定する
    — 「差が小さければ良い」に緩めると cos/sin の入れ替えのような取り違えが素通りするので、
    ずれる側にも境界を置く。
    """

    #: 実重みの speaker encoder / DiT と同じ head_dim（768/12 = 64・1280/20 = 64）。
    MODEL_HEAD_DIM = 64
    #: 丸めが割れる側の幾何（回帰の観測点）。
    SMALL_HEAD_DIM = 8
    HEADS, LENGTH = 3, 11

    #: 値域 O(1) の入力に対する 1 ulp 相当の上限（f32 の eps = 1.19e-7）。取り違えの誤差は
    #: O(1) なので、この上限は「丸めだけ」を通す。
    ONE_ULP = 1.2e-7

    def _inputs(self, irodori_model, head_dim: int):
        torch.manual_seed(0)
        x = torch.randn(1, self.LENGTH, self.HEADS, head_dim)
        complex_table = irodori_model.precompute_freqs_cis(head_dim, self.LENGTH)
        real_table = patch_irodori.real_pair_rope_table(head_dim, self.LENGTH)
        return x, complex_table, real_table

    def test_the_model_geometry_is_bit_identical(self, restore_forward):
        irodori_model = restore_forward
        x, complex_table, real_table = self._inputs(irodori_model, self.MODEL_HEAD_DIM)
        with torch.no_grad():
            expected = irodori_model.apply_rotary_emb(x, complex_table)

        patch_irodori.apply_patches()
        with torch.no_grad():
            actual = irodori_model.apply_rotary_emb(x, real_table)

        assert torch.equal(actual, expected)

    def test_a_smaller_head_dim_stays_within_one_ulp(self, restore_forward):
        irodori_model = restore_forward
        x, complex_table, real_table = self._inputs(irodori_model, self.SMALL_HEAD_DIM)
        with torch.no_grad():
            expected = irodori_model.apply_rotary_emb(x, complex_table)

        patch_irodori.apply_patches()
        with torch.no_grad():
            actual = irodori_model.apply_rotary_emb(x, real_table)

        assert float((actual - expected).abs().max()) <= self.ONE_ULP

    def test_a_complex_table_still_takes_the_original_path(self, restore_forward):
        """MUST: 差し替えは大域なので、実数化していない呼び出し側の意味論を変えない。"""
        irodori_model = restore_forward
        x, complex_table, _real = self._inputs(irodori_model, self.MODEL_HEAD_DIM)
        with torch.no_grad():
            expected = irodori_model.apply_rotary_emb(x, complex_table)

        patch_irodori.apply_patches()
        with torch.no_grad():
            actual = irodori_model.apply_rotary_emb(x, complex_table)

        assert torch.equal(actual, expected)

    def test_swapping_cos_and_sin_breaks_the_result(self, restore_forward):
        """恒真化の門 — 表の 2 面を入れ替えたら値が変わる（= 表の向きを見ている）。"""
        irodori_model = restore_forward
        x, complex_table, real_table = self._inputs(irodori_model, self.MODEL_HEAD_DIM)
        with torch.no_grad():
            expected = irodori_model.apply_rotary_emb(x, complex_table)

        patch_irodori.apply_patches()
        with torch.no_grad():
            actual = irodori_model.apply_rotary_emb(x, real_table.flip(1))

        assert not torch.allclose(actual, expected)


class TestPatchedRmsNormIsBitIdentical:
    """rank-2 weight の分割が原実装と 1 ビットも違わず、rank-1 の経路は逐語のままであること。"""

    HEADS, HEAD_DIM, EPS = 4, 6, 1e-5

    def test_a_rank2_weight_is_split_without_changing_the_value(self, restore_forward):
        irodori_model = restore_forward
        torch.manual_seed(1)
        norm = irodori_model.RMSNorm((self.HEADS, self.HEAD_DIM), eps=self.EPS)
        torch.nn.init.normal_(norm.weight)
        x = torch.randn(1, 5, self.HEADS, self.HEAD_DIM)
        with torch.no_grad():
            expected = norm(x)

        patch_irodori.apply_patches()
        with torch.no_grad():
            actual = norm(x)

        assert torch.equal(actual, expected)

    def test_a_rank1_weight_takes_the_verbatim_path(self, restore_forward):
        irodori_model = restore_forward
        torch.manual_seed(2)
        norm = irodori_model.RMSNorm(self.HEAD_DIM, eps=self.EPS)
        torch.nn.init.normal_(norm.weight)
        x = torch.randn(1, 5, self.HEAD_DIM)
        with torch.no_grad():
            expected = norm(x)

        patch_irodori.apply_patches()
        with torch.no_grad():
            actual = norm(x)

        assert torch.equal(actual, expected)


class TestPatchedLowRankAdaLnIsBitIdentical:
    """DiT の `LowRankAdaLN` の weightless RMS を `rms_norm` へ寄せても値が変わらないこと。"""

    MODEL_DIM, RANK, EPS = 12, 5, 1e-5

    def _module(self, irodori_model):
        """低ランク補正が**効いている**状態の AdaLN を作る。

        MUST: `nn.init.zeros_` されている up 側の weight / bias を乱数で埋め直す — 素の
        初期化のままだと補正項が恒等になり、写し間違い（shift と scale の入れ替え・
        `+ shift` の落とし）が「どちらも同じ値」で素通りする。
        """
        torch.manual_seed(3)
        module = irodori_model.LowRankAdaLN(
            model_dim=self.MODEL_DIM, rank=self.RANK, eps=self.EPS
        ).eval()
        for parameter in module.parameters():
            torch.nn.init.normal_(parameter)
        x = torch.randn(1, 4, self.MODEL_DIM)
        cond_embed = torch.randn(1, 1, 3 * self.MODEL_DIM)
        return module, x, cond_embed

    def test_the_patched_forward_reproduces_the_original(self, restore_forward):
        irodori_model = restore_forward
        module, x, cond_embed = self._module(irodori_model)
        with torch.no_grad():
            expected_x, expected_gate = module(x, cond_embed)

        patch_irodori.apply_patches()
        assert (
            irodori_model.LowRankAdaLN.forward is patch_irodori._folded_rms_low_rank_adaln_forward
        )
        with torch.no_grad():
            actual_x, actual_gate = module(x, cond_embed)

        assert torch.equal(actual_x, expected_x)
        assert torch.equal(actual_gate, expected_gate)

    def test_the_low_rank_correction_is_not_inert(self, restore_forward):
        """恒真化の門 — 補正が効いていなければ、この同値検証は何も見ていない。"""
        irodori_model = restore_forward
        module, x, cond_embed = self._module(irodori_model)
        with torch.no_grad():
            with_correction, _gate = module(x, cond_embed)
            for name in ("shift_up", "scale_up", "gate_up"):
                torch.nn.init.zeros_(getattr(module, name).weight)
                torch.nn.init.zeros_(getattr(module, name).bias)
            without_correction, _gate = module(x, cond_embed)

        assert not torch.allclose(with_correction, without_correction)
