"""`minicpm5/export.py` の台本レベルの約束事（実重み不要分）。

実重みの emit は手動（`deberta` / `embeddinggemma` のテストと同じ規律）。ここで固定するのは、
壊れると**偽 PASS** になる側の規律だけ:

- `gqa_sdpa_attention` が `repeat_kv` 実体化版と**同じ数値**を返すこと（迂回の正当性そのもの）
- mask 無し / dropout / `is_causal` が fail loudly（因果性が黙って消える経路を残さない）
- 加算 causal マスクが **prefix と可換**（ADR 0010 の畳み込みが成立する条件）
- `assert_ir_form` が repeat_kv 実体化形と mask のグラフ入力残りを**実際に検出**すること
  （tiny な実モデルを素の `sdpa` で export した形を被験体にする — 恒真化の門）
- sanity が期待継続の不一致と定数出力の両方を落とすこと（片方だけでは恒真になる）

transformers を要するケースだけ `importorskip` で SKIP する（既定 sync の CI ではモデル系
依存が入らない — ADR 0065 の 2 job 構成）。
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch.export import Dim

from karume.convert import PRESERVED_OP_PREFIXES_WITH_ATTENTION
from karume.ir import IrGraph, IrInitializer, IrInput, IrNode, IrStorage, IrValue
from karume.pipeline import export_to_file
from karume.rope import assert_rope_lifted
from minicpm5 import export as mc

#: tiny な被験体の形（head_dim は hidden/heads = 8 と**別の値**にする — 実物 MiniCPM5-1B の
#: `head_dim: 128` ≠ 1536/16 = 96 と同じ罠を踏ませるため）。
HEADS = 4
KV_HEADS = 2
HEAD_DIM = 16
LAYERS = 2
HIDDEN = 32
VOCAB = 64
TINY_SYM_MAX = 32


def _tiny_config(attn_implementation: str):
    transformers = pytest.importorskip("transformers")
    return transformers.LlamaConfig(
        hidden_size=HIDDEN,
        intermediate_size=48,
        num_hidden_layers=LAYERS,
        num_attention_heads=HEADS,
        num_key_value_heads=KV_HEADS,
        head_dim=HEAD_DIM,
        vocab_size=VOCAB,
        max_position_embeddings=256,
        rope_theta=5_000_000.0,
        tie_word_embeddings=False,
        attn_implementation=attn_implementation,
    )


def _tiny_checkpoint(tmp_path: Path) -> Path:
    """tiny な `LlamaForCausalLM` を実チェックポイントとして書く（`load_wrapper` の実経路用）。"""
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(0)
    model = transformers.LlamaForCausalLM(_tiny_config("sdpa")).to(torch.float32).eval()
    model.save_pretrained(tmp_path)
    return tmp_path


def _export_tiny(tmp_path: Path, wrapper) -> IrGraph:
    ids = torch.randint(0, VOCAB, (1, 7), dtype=torch.int64)
    return export_to_file(
        wrapper,
        (ids,),
        tmp_path / mc.MODEL_FILE,
        dynamic_shapes=({1: Dim("T", min=2, max=TINY_SYM_MAX)},),
        preserved=PRESERVED_OP_PREFIXES_WITH_ATTENTION,
    )


@pytest.fixture
def tiny_gqa(tmp_path):
    """台本の `load_wrapper` を通した tiny ラッパを export して `(graph, config)`。"""
    wrapper = mc.load_wrapper(_tiny_checkpoint(tmp_path / "ckpt"))
    return _export_tiny(tmp_path, wrapper), wrapper.model.config


@pytest.fixture
def tiny_plain_sdpa(tmp_path):
    """**素の `sdpa`** で読んだ同じモデルを export して `(graph, config)`（repeat_kv 実体化形）。"""
    transformers = pytest.importorskip("transformers")
    model = transformers.LlamaForCausalLM.from_pretrained(
        _tiny_checkpoint(tmp_path / "ckpt"), dtype=torch.float32, attn_implementation="sdpa"
    )
    model.eval()
    assert_rope_lifted(model, "tiny")
    wrapper = mc.CausalLmWrapper(model).eval()
    return _export_tiny(tmp_path, wrapper), model.config


class TestGqaSdpaAttentionMatchesTheMaterializedForm:
    """迂回の正当性: `enable_gqa=True` は `repeat_kv` してから呼ぶのと同じ数値でなければ
    ならない（これが崩れていれば「GQA 形は保てたが別の数値」になる）。"""

    @staticmethod
    def _reference(query, key, value, mask, scaling):
        repeats = query.shape[1] // key.shape[1]
        expanded = [
            tensor.repeat_interleave(repeats, dim=1).contiguous() for tensor in (key, value)
        ]
        output = torch.nn.functional.scaled_dot_product_attention(
            query, *expanded, attn_mask=mask, dropout_p=0.0, scale=scaling, is_causal=False
        )
        return output.transpose(1, 2).contiguous()

    def test_the_outputs_agree(self):
        torch.manual_seed(0)
        query = torch.randn(1, 8, 5, HEAD_DIM)
        key = torch.randn(1, 2, 5, HEAD_DIM)
        value = torch.randn(1, 2, 5, HEAD_DIM)
        mask = mc.additive_causal_mask(5)
        scaling = HEAD_DIM**-0.5

        actual, weights = mc.gqa_sdpa_attention(
            torch.nn.Identity(), query, key, value, mask, scaling=scaling
        )

        assert weights is None
        expected = self._reference(query, key, value, mask, scaling)
        assert torch.allclose(actual, expected, atol=1e-6)

    def test_a_wrong_group_mapping_would_be_visible(self):
        """故障注入: kv head を入れ替えると上の突合が壊れる（恒真でないことの裏取り）。"""
        torch.manual_seed(0)
        query = torch.randn(1, 8, 5, HEAD_DIM)
        key = torch.randn(1, 2, 5, HEAD_DIM)
        value = torch.randn(1, 2, 5, HEAD_DIM)
        mask = mc.additive_causal_mask(5)
        scaling = HEAD_DIM**-0.5

        actual, _ = mc.gqa_sdpa_attention(
            torch.nn.Identity(), query, key.flip(1), value, mask, scaling=scaling
        )

        expected = self._reference(query, key, value, mask, scaling)
        assert not torch.allclose(actual, expected, atol=1e-6)


class TestGqaSdpaAttentionGates:
    def test_a_missing_mask_fails_loudly(self):
        """MUST: mask 無しを素通しすると非因果の attention が黙って出る。"""
        query = torch.zeros(1, 4, 3, HEAD_DIM)
        key = torch.zeros(1, 2, 3, HEAD_DIM)

        with pytest.raises(ValueError, match="attn_mask 無し"):
            mc.gqa_sdpa_attention(torch.nn.Identity(), query, key, key, None)

    def test_dropout_fails_loudly(self):
        query = torch.zeros(1, 4, 3, HEAD_DIM)
        key = torch.zeros(1, 2, 3, HEAD_DIM)
        mask = mc.additive_causal_mask(3)

        with pytest.raises(ValueError, match="dropout"):
            mc.gqa_sdpa_attention(torch.nn.Identity(), query, key, key, mask, dropout=0.1)

    @pytest.mark.parametrize("name", ["is_causal", "position_bias"])
    def test_semantics_changing_kwargs_fail_loudly(self, name):
        """MUST: 無視すると eager 側と別の数値経路になったことがどこにも出ない。"""
        query = torch.zeros(1, 4, 3, HEAD_DIM)
        key = torch.zeros(1, 2, 3, HEAD_DIM)
        mask = mc.additive_causal_mask(3)

        with pytest.raises(ValueError, match=name):
            mc.gqa_sdpa_attention(
                torch.nn.Identity(), query, key, key, mask, **{name: torch.ones(1)}
            )


class TestAdditiveCausalMask:
    def test_the_band_is_zero_below_the_diagonal_and_neg_inf_above(self):
        mask = mc.additive_causal_mask(4)

        assert tuple(mask.shape) == (1, 1, 4, 4)
        assert mask.dtype is torch.float32
        assert torch.equal(mask[0, 0].tril(), torch.zeros(4, 4))
        assert bool(torch.isneginf(mask[0, 0]).sum() == 6)

    def test_the_prefix_of_a_longer_mask_is_the_shorter_mask(self):
        """MUST: prefix と可換（ADR 0010）— これが崩れると Tmax 畳み込みが黙って別の値になる。"""
        long_mask = mc.additive_causal_mask(TINY_SYM_MAX)

        prefix = long_mask[:, :, :5, :5]

        assert torch.equal(prefix, mc.additive_causal_mask(5))


def _ir_graph(heads: int, kv_heads: int, layers: int = LAYERS) -> IrGraph:
    """attention `layers` 本の最小 IR（mask は Tmax 定数 + sym_prefix_slice）。"""
    graph = IrGraph(symbols=["T"])
    graph.inputs.append(IrInput(name=mc.INPUT_IDS, dtype="i32", shape=[1, "T"]))
    graph.initializers["mask_const"] = IrInitializer(
        tensor="const.mask", storage=IrStorage(dtype="f32")
    )
    graph.values["mask_const"] = IrValue(dtype="f32", shape=[1, 1, TINY_SYM_MAX, TINY_SYM_MAX])
    graph.values["mask"] = IrValue(dtype="f32", shape=[1, 1, "T", "T"])
    graph.nodes.append(
        IrNode(
            op=mc.SYM_PREFIX_SLICE_OP,
            ins=["mask_const"],
            outs=["mask"],
            attrs={"sym": "T", "slices": [{"dim": 2, "coeff": 1, "offset": 0}]},
        )
    )
    for layer in range(layers):
        for slot, count in (("q", heads), ("k", kv_heads), ("v", kv_heads)):
            graph.values[f"{slot}{layer}"] = IrValue(dtype="f32", shape=[1, count, "T", HEAD_DIM])
        graph.values[f"attn{layer}"] = IrValue(dtype="f32", shape=[1, heads, "T", HEAD_DIM])
        graph.nodes.append(
            IrNode(
                op=mc.ATTENTION_OP,
                ins=[f"q{layer}", f"k{layer}", f"v{layer}", "mask"],
                outs=[f"attn{layer}"],
                attrs={"scale": 0.5},
            )
        )
    graph.outputs.append(f"attn{layers - 1}")
    return graph


TINY_IR_CONFIG = SimpleNamespace(
    num_attention_heads=HEADS,
    num_key_value_heads=KV_HEADS,
    head_dim=HEAD_DIM,
    num_hidden_layers=LAYERS,
)


class TestAssertIrForm:
    def test_the_grouped_shape_passes(self):
        form = mc.assert_ir_form(_ir_graph(HEADS, KV_HEADS), TINY_IR_CONFIG, TINY_SYM_MAX)

        assert form["heads"] == [HEADS, KV_HEADS, KV_HEADS]
        assert form["attention_nodes"] == LAYERS
        assert form["mask_constant"] == "mask_const"

    def test_a_materialized_kv_shape_is_rejected(self):
        """MUST: Hkv が H に化けた形は「数値は合うが検収の意味が消えた」資産。"""
        with pytest.raises(AssertionError, match="repeat_kv"):
            mc.assert_ir_form(_ir_graph(HEADS, HEADS), TINY_IR_CONFIG, TINY_SYM_MAX)

    def test_a_mask_left_as_a_graph_input_is_rejected(self):
        graph = _ir_graph(HEADS, KV_HEADS)
        graph.inputs.append(IrInput(name="attention_mask", dtype="f32", shape=[1, 1, "T", "T"]))

        with pytest.raises(AssertionError, match="グラフ入力"):
            mc.assert_ir_form(graph, TINY_IR_CONFIG, TINY_SYM_MAX)

    def test_a_mask_constant_of_the_wrong_extent_is_rejected(self):
        """Tmax より短い定数は「T が上限まで伸びると黙って足りない」形。"""
        graph = _ir_graph(HEADS, KV_HEADS)
        graph.values["mask_const"] = IrValue(dtype="f32", shape=[1, 1, 8, 8])

        with pytest.raises(AssertionError, match="Tmax 形"):
            mc.assert_ir_form(graph, TINY_IR_CONFIG, TINY_SYM_MAX)

    def test_a_missing_layer_is_rejected(self):
        with pytest.raises(AssertionError, match="層と一致しない"):
            mc.assert_ir_form(_ir_graph(HEADS, KV_HEADS, layers=1), TINY_IR_CONFIG, TINY_SYM_MAX)

    def test_a_derived_head_dim_is_rejected(self):
        """MUST: D は `config.head_dim` と突合する（hidden/heads から導出した値では通らない）。"""
        derived = SimpleNamespace(
            num_attention_heads=HEADS,
            num_key_value_heads=KV_HEADS,
            head_dim=HIDDEN // HEADS,
            num_hidden_layers=LAYERS,
        )

        with pytest.raises(AssertionError, match=r"config\.head_dim"):
            mc.assert_ir_form(_ir_graph(HEADS, KV_HEADS), derived, TINY_SYM_MAX)


class TestExportedForm:
    """tiny な実モデルを実際に export して形を見る（transformers が要る）。"""

    def test_the_graph_keeps_the_grouped_shape_and_folds_the_mask(self, tiny_gqa):
        graph, config = tiny_gqa

        form = mc.assert_ir_form(graph, config, TINY_SYM_MAX)

        assert [spec.name for spec in graph.inputs] == [mc.INPUT_IDS]
        assert form["heads"] == [HEADS, KV_HEADS, KV_HEADS]
        assert form["head_dim"] == HEAD_DIM
        assert mc.SYM_PREFIX_SLICE_OP in graph.required_ops

    def test_plain_sdpa_materializes_repeat_kv_and_is_rejected(self, tiny_plain_sdpa):
        """恒真化の門: 上流既定の経路は本当に Hkv を潰す（= 上の緑は検査が効いた結果）。"""
        graph, config = tiny_plain_sdpa

        with pytest.raises(AssertionError, match="repeat_kv"):
            mc.assert_ir_form(graph, config, TINY_SYM_MAX)


#: `_sanity` の被験体（ケース名 → 期待トークン id）と表示名。
SANITY_EXPECTED = {"a": 10, "b": 20}
SANITY_LABELS = {10: "Ġx", 20: "Ġy", 30: "Ġz"}


class TestSanity:
    def test_matching_greedy_tokens_pass(self):
        result = mc._sanity({"a": 10, "b": 20}, SANITY_EXPECTED, SANITY_LABELS)

        assert result == {"a": "10(Ġx)", "b": "20(Ġy)"}

    def test_a_mismatch_fails_loudly(self):
        with pytest.raises(AssertionError, match="期待継続と違う"):
            mc._sanity({"a": 30, "b": 20}, SANITY_EXPECTED, SANITY_LABELS)

    def test_a_constant_output_fails_loudly(self):
        """MUST: 期待表を緩めても残る検出線（両ケースが同じ 1 位 = 定数出力）。"""
        with pytest.raises(AssertionError, match="定数出力"):
            mc._sanity({"a": 10, "b": 10}, {"a": 10, "b": 10}, SANITY_LABELS)

    def test_a_case_set_mismatch_fails_loudly(self):
        with pytest.raises(AssertionError, match="ケース集合"):
            mc._sanity({"a": 10}, SANITY_EXPECTED, SANITY_LABELS)


class TestGreedyTokens:
    def test_the_last_position_is_the_one_that_is_read(self):
        """MUST: 最終位置。他の位置を見ると「継続の予測」ではない量を突合することになる。"""
        logits = torch.zeros(1, 3, 5)
        logits[0, 0, 4] = 1.0
        logits[0, -1, 2] = 1.0

        assert mc.greedy_tokens({"case": logits}) == {"case": 2}


class _StubTokenizer:
    """`encode(...).ids` だけを持つ最小のトークナイザ（期待継続の単一トークン検査用）。"""

    def __init__(self, ids: list[int]) -> None:
        self._ids = ids

    def encode(self, text: str, add_special_tokens: bool = True):
        """`tokenizers.Tokenizer.encode` の呼び出し規約だけ合わせる（引数は使わない）。"""
        return SimpleNamespace(ids=list(self._ids))


class TestExpectedTokenIds:
    def test_a_single_token_continuation_is_accepted(self):
        assert set(mc.expected_token_ids(_StubTokenizer([7])).values()) == {7}

    def test_a_multi_token_continuation_fails_loudly(self):
        """MUST: 複数トークンだと「1 位の一致」の意味が定まらない（黙って先頭だけ見ない）。"""
        with pytest.raises(AssertionError, match="単一トークン"):
            mc.expected_token_ids(_StubTokenizer([7, 8]))


class TestGoldenCases:
    def test_every_case_has_a_greedy_expectation(self):
        """MUST: 期待の無いケースを混ぜない（sanity が一部のケースだけ見る形にしない）。"""
        assert {name for name, _ in mc.GOLDEN_CASES} == set(mc.GREEDY_EXPECTATIONS)

    def test_the_expectations_are_not_all_the_same_token(self):
        """定数出力の検出線が恒真にならない条件（期待が 1 種類だと `_sanity` の後段が死ぬ）。"""
        assert len(set(mc.GREEDY_EXPECTATIONS.values())) > 1
