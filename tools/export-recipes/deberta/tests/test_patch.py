"""`deberta.patch` の差し替え版が上流の元実装と eager 同値であることの単体テスト（実重み不要）。

`deberta.export` の golden はパッチを当ててから採る（`export_variant`）ので、差し替え版どうしの
一致しか見ない。上流の `DisentangledSelfAttention.disentangled_attention_bias` との同値を直接
縛るのはここだけ — transformers のピンを上げて元実装が変わったとき、ここが落ちる。

tiny な config（重みは seed 固定の乱数）で、元の関数と差し替え版に同じ入力を与えて
`torch.equal` を見る。表を外から渡す以外の違いを持ち込まないのが patch の MUST なので、
許容差は置かない。
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("transformers")
from transformers import DebertaV2Config
from transformers.models.deberta_v2 import modeling_deberta_v2 as upstream

from deberta import patch

#: 差し替え前の元実装。クラス属性の差し替えはプロセス全域なので、**import 時に退避**する
#: （パッチを当てた後に読むと差し替え版どうしを比べる恒真のテストになる）。
UPSTREAM_BIAS = upstream.DisentangledSelfAttention.disentangled_attention_bias

HIDDEN = 16
HEADS = 2
BATCH = 2
#: バケット幅。実物（ku-nlp char-wwm）は 256 / 最大位置 512 — 比を保って縮める。
POSITION_BUCKETS = 8
MAX_POSITION = 16
#: 系列長。最長は対数バケットの域（|i − j| > バケット幅の半分）と clamp の両端を踏む長さにする。
SEQUENCE = 24
#: 同値を見る系列長（バケットの線形域だけ・対数域に入る・clamp まで届く）。
LENGTHS = (5, 13, SEQUENCE)
#: `1 + len(pos_att_type)`（上流 `DisentangledSelfAttention.forward` と同じ値）。
SCALE_FACTOR = 3


@pytest.fixture(autouse=True)
def restore_upstream_bias():
    """差し替えを戻す（クラス属性の差し替えと `_applied` はプロセス全域 — 次のケースが汚れる）。"""
    applied = patch._applied
    try:
        yield
    finally:
        upstream.DisentangledSelfAttention.disentangled_attention_bias = UPSTREAM_BIAS
        patch._applied = applied


def _config(**overrides: object) -> DebertaV2Config:
    """ku-nlp char-wwm と同じ経路（c2p + p2c・共有射影・バケット化）を通る最小の config。"""
    settings: dict[str, object] = {
        "hidden_size": HIDDEN,
        "num_attention_heads": HEADS,
        "relative_attention": True,
        "pos_att_type": ["c2p", "p2c"],
        "share_att_key": True,
        "position_buckets": POSITION_BUCKETS,
        "max_relative_positions": -1,
        "max_position_embeddings": MAX_POSITION,
    }
    settings.update(overrides)
    return DebertaV2Config(**settings)


def _tiny(length: int = SEQUENCE) -> tuple[upstream.DisentangledSelfAttention, SimpleNamespace]:
    torch.manual_seed(0)
    attention = upstream.DisentangledSelfAttention(_config()).eval()
    head_dim = HIDDEN // HEADS
    inputs = SimpleNamespace(
        query=torch.randn(BATCH * HEADS, length, head_dim),
        key=torch.randn(BATCH * HEADS, length, head_dim),
        rel_embeddings=torch.randn(attention.pos_ebd_size * 2, HIDDEN),
    )
    return attention, inputs


def _tables(
    attention: upstream.DisentangledSelfAttention, length: int = SEQUENCE
) -> tuple[torch.Tensor, torch.Tensor]:
    """`deberta.export.build_graph_inputs` と同じ綴り（バケット幅と最大位置はモジュールから）。"""
    return patch.build_rel_pos_tables(
        length,
        position_buckets=attention.position_buckets,
        max_position=attention.max_relative_positions,
    )


class TestExternalRelPosPatch:
    @pytest.mark.parametrize("length", LENGTHS)
    def test_the_patched_bias_equals_the_upstream_one_bit_for_bit(self, length: int) -> None:
        assert UPSTREAM_BIAS is not patch._external_disentangled_attention_bias
        attention, inputs = _tiny(length)
        with torch.no_grad():
            expected = attention.disentangled_attention_bias(
                inputs.query, inputs.key, None, inputs.rel_embeddings, SCALE_FACTOR
            )

            patch.apply_external_rel_pos_patch()
            actual = attention.disentangled_attention_bias(
                inputs.query,
                inputs.key,
                _tables(attention, length),
                inputs.rel_embeddings,
                SCALE_FACTOR,
            )

        assert patch.patch_applied()
        assert torch.equal(actual, expected)

    def test_the_p2c_table_is_the_transpose_of_the_c2p_table(self) -> None:
        """式の正本（モジュール docstring）の主張 — バケット化が奇関数なので p2c = c2pᵀ。"""
        c2p_pos, p2c_pos = _tables(_tiny()[0])

        assert torch.equal(p2c_pos, c2p_pos.T)

    def test_swapping_the_two_tables_breaks_the_equivalence(self) -> None:
        """対（恒真でない）: 同値の主張は表の中身で決まる — c2p と p2c を入れ替えると割れる。"""
        attention, inputs = _tiny()
        with torch.no_grad():
            expected = UPSTREAM_BIAS(
                attention, inputs.query, inputs.key, None, inputs.rel_embeddings, SCALE_FACTOR
            )
            c2p_pos, p2c_pos = _tables(attention)
            swapped = patch._external_disentangled_attention_bias(
                attention,
                inputs.query,
                inputs.key,
                (p2c_pos, c2p_pos),
                inputs.rel_embeddings,
                SCALE_FACTOR,
            )

        assert not torch.equal(swapped, expected)

    @pytest.mark.parametrize(
        "relative_pos",
        [None, torch.zeros(1, SEQUENCE, SEQUENCE, dtype=torch.long), (None, None, None)],
        ids=["none", "tensor", "triple"],
    )
    def test_the_patched_bias_refuses_anything_but_the_table_pair(self, relative_pos) -> None:
        """fallback を持たない MUST — 元の経路へ黙って落ちると「2MiB を焼いたまま緑」になる。"""
        attention, inputs = _tiny()

        with pytest.raises(TypeError, match="外部供給"):
            patch._external_disentangled_attention_bias(
                attention,
                inputs.query,
                inputs.key,
                relative_pos,
                inputs.rel_embeddings,
                SCALE_FACTOR,
            )


class TestAssertSupported:
    def test_it_accepts_the_path_the_patch_implements(self) -> None:
        patch.assert_supported(_config())

    @pytest.mark.parametrize(
        ("overrides", "message"),
        [
            ({"pos_att_type": ["c2p"]}, "pos_att_type"),
            ({"share_att_key": False}, "share_att_key"),
            ({"position_buckets": 0}, "position_buckets"),
        ],
        ids=["c2p-only", "separate-projection", "no-buckets"],
    )
    def test_it_rejects_a_config_the_patch_does_not_implement(
        self, overrides: dict[str, object], message: str
    ) -> None:
        with pytest.raises(ValueError, match=message):
            patch.assert_supported(_config(**overrides))
