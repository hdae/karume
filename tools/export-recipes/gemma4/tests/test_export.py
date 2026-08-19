"""`gemma4/export.py` の台本レベルの約束事（実重み不要分）。

実重みの emit は手動（`minicpm5` / `embeddinggemma` のテストと同じ規律）。ここで固定するのは、
壊れると**偽 PASS** になる側の規律だけ:

- 加算マスク 2 種（causal / 帯）の境界と **prefix 可換**（ADR 0010 の畳み込みが成立する条件）
- PLE 35 分割が上流 `get_per_layer_inputs` と**ビット一致**すること（列の割り付けを間違えても
  形も型も合うので、golden の突合まで誰にも見えない）
- 量子化の対象割り付けが **排他かつ網羅**（tied `lm_head` の二重丸めの禁止 — ADR 0019 / 0069）
- `assert_ir_form` が repeat_kv 実体化・mask のグラフ入力残り・層種別の head_dim 取り違え・
  causal と帯の定数共有・格納 dtype の本数違いを**実際に検出**すること
- golden の長さ規律（窓より長いケースが 1 本以上）と sanity が実際に落とすこと

transformers を要するケースだけ `importorskip` で SKIP する（既定 sync の CI ではモデル系
依存が入らない — ADR 0065 の 2 job 構成）。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from gemma4 import export as gx
from karume.ir import IrGraph, IrInitializer, IrInput, IrNode, IrStorage, IrValue

#: tiny な被験体の形。head_dim は層種別で**違う値**にする（実物 E2B の 256 / 512 と同じ罠を
#: 踏ませるため — `hidden_size / num_attention_heads` はどちらとも違う）。
HEADS = 4
KV_HEADS = 1
SLIDING_DEPTH = 16
FULL_DEPTH = 32
HIDDEN = 64
PLE_DIM = 32
VOCAB = 64
WINDOW = 4
TINY_SYM_MAX = 32
LAYER_TYPES = (
    gx.SLIDING_ATTENTION,
    gx.SLIDING_ATTENTION,
    gx.FULL_ATTENTION,
)
DEPTHS = {gx.SLIDING_ATTENTION: SLIDING_DEPTH, gx.FULL_ATTENTION: FULL_DEPTH}


# ---- 加算マスク ------------------------------------------------------------


class TestAdditiveCausalMask:
    def test_the_band_is_zero_below_the_diagonal_and_neg_inf_above(self):
        mask = gx.additive_causal_mask(4)

        assert tuple(mask.shape) == (1, 1, 4, 4)
        assert mask.dtype is torch.float32
        assert torch.equal(mask[0, 0].tril(), torch.zeros(4, 4))
        assert bool(torch.isneginf(mask[0, 0]).sum() == 6)

    def test_the_prefix_of_a_longer_mask_is_the_shorter_mask(self):
        """MUST: prefix と可換（ADR 0010）— 崩れると Tmax 畳み込みが黙って別の値になる。"""
        long_mask = gx.additive_causal_mask(TINY_SYM_MAX)

        prefix = long_mask[:, :, :5, :5]

        assert torch.equal(prefix, gx.additive_causal_mask(5))


class TestAdditiveSlidingMask:
    def test_the_window_includes_self_and_excludes_the_row_one_step_beyond(self):
        """上流 `sliding_window_mask_function` と同値: `0 <= q - kv < window`。"""
        mask = gx.additive_sliding_mask(8, WINDOW)[0, 0]

        assert float(mask[5, 5]) == 0.0
        assert float(mask[5, 5 - (WINDOW - 1)]) == 0.0
        assert bool(torch.isneginf(mask[5, 5 - WINDOW]))

    def test_the_future_is_blocked(self):
        mask = gx.additive_sliding_mask(8, WINDOW)[0, 0]

        assert bool(torch.isneginf(mask[3, 4]))

    def test_a_short_sequence_makes_the_band_identical_to_causal(self):
        """T <= window では帯 = causal。golden に窓より長いケースが要る理由そのもの。"""
        assert torch.equal(
            gx.additive_sliding_mask(WINDOW, WINDOW), gx.additive_causal_mask(WINDOW)
        )

    def test_a_long_sequence_makes_the_band_differ_from_causal(self):
        assert not torch.equal(
            gx.additive_sliding_mask(WINDOW + 1, WINDOW), gx.additive_causal_mask(WINDOW + 1)
        )

    def test_the_prefix_of_a_longer_mask_is_the_shorter_mask(self):
        """MUST: 帯も prefix と可換（causal と同じ ADR 0010 の条件）。"""
        long_mask = gx.additive_sliding_mask(TINY_SYM_MAX, WINDOW)

        prefix = long_mask[:, :, :7, :7]

        assert torch.equal(prefix, gx.additive_sliding_mask(7, WINDOW))


# ---- PLE の 35 分割 --------------------------------------------------------


def _packed_table() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(VOCAB, len(LAYER_TYPES) * PLE_DIM)


def _split_tables(packed: torch.Tensor) -> nn.ModuleList:
    return nn.ModuleList(
        [
            nn.Embedding.from_pretrained(
                packed[:, index * PLE_DIM : (index + 1) * PLE_DIM].contiguous(), freeze=True
            )
            for index in range(len(LAYER_TYPES))
        ]
    )


class TestPerLayerInputs:
    """列の割り付け（層 i ↔ 列 `[i*D, (i+1)*D)`）が上流の reshape と一致すること。"""

    @staticmethod
    def _reference(packed: torch.Tensor, ids: torch.Tensor, scale: float) -> torch.Tensor:
        """上流 `get_per_layer_inputs` の式そのまま（1 枚 lookup → スケール → reshape）。"""
        looked_up = nn.functional.embedding(ids, packed) * scale
        return looked_up.reshape(*ids.shape, len(LAYER_TYPES), PLE_DIM)

    def test_the_split_reproduces_the_packed_lookup(self):
        packed = _packed_table()
        ids = torch.randint(0, VOCAB, (1, 5), dtype=torch.int64)
        scale = float(PLE_DIM) ** 0.5

        rebuilt = gx.per_layer_inputs(_split_tables(packed), ids, scale)

        assert torch.equal(rebuilt, self._reference(packed, ids, scale))

    def test_a_wrong_layer_order_would_be_visible(self):
        """故障注入: 層の並びを入れ替えると上の突合が壊れる（恒真でないことの裏取り）。"""
        packed = _packed_table()
        ids = torch.randint(0, VOCAB, (1, 5), dtype=torch.int64)
        scale = float(PLE_DIM) ** 0.5
        tables = _split_tables(packed)
        swapped = nn.ModuleList([tables[1], tables[0], tables[2]])

        rebuilt = gx.per_layer_inputs(swapped, ids, scale)

        assert not torch.equal(rebuilt, self._reference(packed, ids, scale))


def _tiny_text_config():
    """検査席の PLE 表（{@link gx.PLE_PROBE_ROWS} 行）を持つ tiny な text config。"""
    transformers = pytest.importorskip("transformers")
    return transformers.Gemma4TextConfig(
        vocab_size=VOCAB,
        vocab_size_per_layer_input=gx.PLE_PROBE_ROWS,
        hidden_size=HIDDEN,
        intermediate_size=32,
        num_hidden_layers=len(LAYER_TYPES),
        num_attention_heads=HEADS,
        num_key_value_heads=KV_HEADS,
        head_dim=SLIDING_DEPTH,
        global_head_dim=FULL_DEPTH,
        hidden_size_per_layer_input=PLE_DIM,
        layer_types=list(LAYER_TYPES),
        num_kv_shared_layers=1,
        sliding_window=WINDOW,
        use_double_wide_mlp=True,
        final_logit_softcapping=30.0,
        tie_word_embeddings=True,
        max_position_embeddings=256,
    )


@pytest.fixture
def tiny_model_and_tables():
    """`(model, tables, probe)` — 検査席に分割元の**散点** probe 行を載せた組。"""
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(0)
    model = transformers.Gemma4ForCausalLM(_tiny_text_config()).to(torch.float32).eval()
    packed = _packed_table()
    tables = _split_tables(packed)
    probe = gx.probe_rows(VOCAB)
    with torch.no_grad():
        model.model.embed_tokens_per_layer.weight.copy_(
            packed[torch.tensor(probe, dtype=torch.int64)]
        )
    return model, tables, probe


class TestProbeRows:
    def test_the_probe_scatters_across_both_ends_and_the_middle(self):
        """散点であること（Codex 波 H 指摘 H-05 — 連続 8 行だと 1 ブロックしか踏まない）。"""
        probe = gx.probe_rows(262144)

        assert len(probe) == gx.PLE_PROBE_ROWS
        assert len(set(probe)) == gx.PLE_PROBE_ROWS
        assert 0 in probe and 262143 in probe
        # 最初のブロック境界の両側と中央対。
        assert gx.PLE_ROW_BLOCK - 1 in probe and gx.PLE_ROW_BLOCK in probe
        assert 262144 // 2 in probe

    def test_a_tiny_table_still_yields_the_full_count(self):
        """ブロック境界の無い tiny 表でも本数は固定（検査席の行数が config に焼かれる）。"""
        probe = gx.probe_rows(VOCAB)

        assert len(probe) == gx.PLE_PROBE_ROWS
        assert len(set(probe)) == gx.PLE_PROBE_ROWS
        assert all(0 <= row < VOCAB for row in probe)
        assert 0 in probe and VOCAB - 1 in probe

    def test_a_table_shorter_than_the_probe_fails_loudly(self):
        with pytest.raises(ValueError, match="足りない"):
            gx.probe_rows(gx.PLE_PROBE_ROWS - 1)


class TestAssertPerLayerSplit:
    def test_a_faithful_split_passes(self, tiny_model_and_tables):
        model, tables, probe = tiny_model_and_tables

        gx.assert_per_layer_split(model, tables, probe)

    def test_a_shuffled_split_fails_loudly(self, tiny_model_and_tables):
        """MUST: 列の割り付け違いはここで落とす（形も型も合うので後段では見えない）。"""
        model, tables, probe = tiny_model_and_tables
        shuffled = nn.ModuleList([tables[2], tables[0], tables[1]])

        with pytest.raises(AssertionError, match="ビット一致しない"):
            gx.assert_per_layer_split(model, shuffled, probe)

    def test_a_split_of_the_wrong_width_fails_loudly(self, tiny_model_and_tables):
        model, tables, probe = tiny_model_and_tables
        short = nn.ModuleList([tables[0], tables[1]])

        with pytest.raises(AssertionError, match="形"):
            gx.assert_per_layer_split(model, short, probe)


# ---- 量子化の対象割り付け --------------------------------------------------


class TestQuantizationTargets:
    """i8（embedding 系）と i4（linear）の割り付けが排他かつ網羅であること。

    重みスロットになりうるモジュールは `nn.Embedding` と `nn.Linear` だけ（norm 系の weight は
    `rms_norm` の入力で、`emit.eligible_compressed_initializers` の適格集合の外）。
    """

    @staticmethod
    def _targets(wrapper: nn.Module) -> tuple[set[str], set[str], set[str]]:
        modules = dict(wrapper.named_modules())
        int8 = {
            name
            for name, module in modules.items()
            if gx.is_int8_module(name) and isinstance(module, nn.Embedding)
        }
        int4 = {
            name
            for name, module in modules.items()
            if gx.is_int4_module(name) and isinstance(module, nn.Linear)
        }
        weighted = {
            name for name, module in modules.items() if isinstance(module, nn.Embedding | nn.Linear)
        }
        return int8, int4, weighted

    def test_the_two_target_sets_are_disjoint(self, tiny_model_and_tables):
        """MUST: 排他（tied `lm_head` を両方に通すと二重丸めで scale 台帳が実値とずれる）。"""
        model, tables, _ = tiny_model_and_tables
        wrapper = gx.build_wrapper(model, tables)

        int8, int4, _ = self._targets(wrapper)

        assert int8 & int4 == set()

    def test_every_weight_slot_module_but_the_tied_head_is_covered(self, tiny_model_and_tables):
        """網羅: 唯一の抜けは tied `lm_head`（主 embedding 側の丸めが実体に効く）。"""
        model, tables, _ = tiny_model_and_tables
        wrapper = gx.build_wrapper(model, tables)

        int8, int4, weighted = self._targets(wrapper)

        assert gx.LM_HEAD_MODULE in weighted
        assert int8 | int4 == weighted - {gx.LM_HEAD_MODULE}

    def test_the_int8_side_is_the_embeddings(self, tiny_model_and_tables):
        model, tables, _ = tiny_model_and_tables
        wrapper = gx.build_wrapper(model, tables)

        int8, _, _ = self._targets(wrapper)

        assert int8 == {gx.EMBED_TOKENS_MODULE} | {
            f"{gx.PER_LAYER_PREFIX}{index}" for index in range(len(LAYER_TYPES))
        }

    def test_the_probe_table_is_gone_from_the_wrapper(self, tiny_model_and_tables):
        """MUST: 検査席の PLE 表を残すと「どちらの対象にも当たらない embedding」になる。"""
        model, tables, _ = tiny_model_and_tables

        wrapper = gx.build_wrapper(model, tables)

        assert not hasattr(wrapper.model.model, "embed_tokens_per_layer")


# ---- assert_ir_form --------------------------------------------------------


TINY_IR_CONFIG = SimpleNamespace(
    num_attention_heads=HEADS,
    num_key_value_heads=KV_HEADS,
    head_dim=SLIDING_DEPTH,
    global_head_dim=FULL_DEPTH,
    num_hidden_layers=len(LAYER_TYPES),
    layer_types=list(LAYER_TYPES),
)

#: 正例の格納内訳（i8 = embedding 系 / i4 = linear のダミー本数）。
STORAGE_COUNTS: Mapping[str, int] = {"i8": 4, "i4": 9}


def _ir_graph(
    *,
    kv_heads: int = KV_HEADS,
    layer_types: Sequence[str] = LAYER_TYPES,
    depth_override: Mapping[int, int] | None = None,
    single_mask: bool = False,
    storage_counts: Mapping[str, int] = STORAGE_COUNTS,
) -> IrGraph:
    """attention `len(layer_types)` 本の最小 IR（mask は層種別ごとの Tmax 定数 + スライス）。"""
    graph = IrGraph(symbols=["T"])
    graph.inputs.append(IrInput(name=gx.INPUT_IDS, dtype="i32", shape=[1, "T"]))
    kinds = {kind: "shared" if single_mask else kind for kind in set(layer_types)}
    for source in sorted(set(kinds.values())):
        constant = f"mask_const_{source}"
        graph.initializers[constant] = IrInitializer(
            tensor=f"const.{source}", storage=IrStorage(dtype="f32")
        )
        graph.values[constant] = IrValue(dtype="f32", shape=[1, 1, TINY_SYM_MAX, TINY_SYM_MAX])
        graph.values[f"mask_{source}"] = IrValue(dtype="f32", shape=[1, 1, "T", "T"])
        graph.nodes.append(
            IrNode(
                op=gx.SYM_PREFIX_SLICE_OP,
                ins=[constant],
                outs=[f"mask_{source}"],
                attrs={"sym": "T", "slices": [{"dim": 2, "coeff": 1, "offset": 0}]},
            )
        )
    for index, layer_type in enumerate(layer_types):
        depth = (depth_override or {}).get(index, DEPTHS[layer_type])
        for slot, count in (("q", HEADS), ("k", kv_heads), ("v", kv_heads)):
            graph.values[f"{slot}{index}"] = IrValue(dtype="f32", shape=[1, count, "T", depth])
        graph.values[f"attn{index}"] = IrValue(dtype="f32", shape=[1, HEADS, "T", depth])
        graph.nodes.append(
            IrNode(
                op=gx.ATTENTION_OP,
                ins=[f"q{index}", f"k{index}", f"v{index}", f"mask_{kinds[layer_type]}"],
                outs=[f"attn{index}"],
                attrs={"scale": 1.0},
            )
        )
    for dtype, count in storage_counts.items():
        for slot in range(count):
            name = f"weight_{dtype}_{slot}"
            graph.initializers[name] = IrInitializer(
                tensor=f"w.{dtype}.{slot}", storage=IrStorage(dtype=dtype)
            )
    graph.outputs.append(f"attn{len(layer_types) - 1}")
    return graph


class TestAssertIrForm:
    def test_the_grouped_shape_and_two_masks_pass(self):
        form = gx.assert_ir_form(_ir_graph(), TINY_IR_CONFIG, TINY_SYM_MAX, STORAGE_COUNTS)

        assert form["heads"] == [HEADS, KV_HEADS, KV_HEADS]
        assert form["attention_nodes"] == len(LAYER_TYPES)
        assert form["head_dim"] == {
            gx.SLIDING_ATTENTION: SLIDING_DEPTH,
            gx.FULL_ATTENTION: FULL_DEPTH,
        }
        assert form["storage"]["i4"] == STORAGE_COUNTS["i4"]

    def test_a_materialized_kv_shape_is_rejected(self):
        """MUST: Hkv が H に化けた形は「数値は合うが検収の意味が消えた」資産。"""
        with pytest.raises(AssertionError, match="repeat_kv"):
            gx.assert_ir_form(
                _ir_graph(kv_heads=HEADS), TINY_IR_CONFIG, TINY_SYM_MAX, STORAGE_COUNTS
            )

    def test_a_mask_left_as_a_graph_input_is_rejected(self):
        graph = _ir_graph()
        graph.inputs.append(IrInput(name="attention_mask", dtype="f32", shape=[1, 1, "T", "T"]))

        with pytest.raises(AssertionError, match="グラフ入力"):
            gx.assert_ir_form(graph, TINY_IR_CONFIG, TINY_SYM_MAX, STORAGE_COUNTS)

    def test_a_mask_constant_of_the_wrong_extent_is_rejected(self):
        """Tmax より短い定数は「T が上限まで伸びると黙って足りない」形。"""
        graph = _ir_graph()
        graph.values[f"mask_const_{gx.FULL_ATTENTION}"] = IrValue(dtype="f32", shape=[1, 1, 8, 8])

        with pytest.raises(AssertionError, match="Tmax 形"):
            gx.assert_ir_form(graph, TINY_IR_CONFIG, TINY_SYM_MAX, STORAGE_COUNTS)

    def test_a_missing_layer_is_rejected(self):
        with pytest.raises(AssertionError, match="層と一致しない"):
            gx.assert_ir_form(
                _ir_graph(layer_types=LAYER_TYPES[:2]),
                TINY_IR_CONFIG,
                TINY_SYM_MAX,
                STORAGE_COUNTS,
            )

    def test_a_head_dim_from_the_wrong_layer_type_is_rejected(self):
        """MUST: D は層種別で引く（sliding 層に global_head_dim が来たら落ちる）。"""
        graph = _ir_graph(depth_override={0: FULL_DEPTH})

        with pytest.raises(AssertionError, match="head_dim"):
            gx.assert_ir_form(graph, TINY_IR_CONFIG, TINY_SYM_MAX, STORAGE_COUNTS)

    def test_one_mask_constant_shared_by_both_layer_types_is_rejected(self):
        """causal と帯が同じ定数に畳まれた形 = 窓が効いていない（T <= 512 なら数値も一致する）。"""
        with pytest.raises(AssertionError, match="共有している"):
            gx.assert_ir_form(
                _ir_graph(single_mask=True), TINY_IR_CONFIG, TINY_SYM_MAX, STORAGE_COUNTS
            )

    def test_a_storage_census_that_misses_a_compressed_weight_is_rejected(self):
        """MUST: 適格判定を外れた重みは**黙って f32 で残る**ので、本数で捕まえる。"""
        graph = _ir_graph(storage_counts={"i8": STORAGE_COUNTS["i8"], "i4": 1})

        with pytest.raises(AssertionError, match="格納 dtype の本数"):
            gx.assert_ir_form(graph, TINY_IR_CONFIG, TINY_SYM_MAX, STORAGE_COUNTS)


# ---- golden の長さ規律 / sanity --------------------------------------------


def _case(name: str, length: int) -> tuple[str, torch.Tensor]:
    return name, torch.zeros(1, length, dtype=torch.int64)


class TestAssertCaseLengths:
    def test_a_case_set_with_a_long_case_passes(self):
        gx.assert_case_lengths((_case("short", 6), _case("long", WINDOW + 1)), TINY_SYM_MAX, WINDOW)

    def test_a_case_beyond_the_symbolic_range_fails_loudly(self):
        with pytest.raises(ValueError, match="記号次元の範囲"):
            gx.assert_case_lengths((_case("long", TINY_SYM_MAX + 1),), TINY_SYM_MAX, WINDOW)

    def test_a_case_set_that_never_exceeds_the_window_fails_loudly(self):
        """MUST: 窓を超えないと帯マスクが causal と一致し、sliding 層が無検証になる。"""
        with pytest.raises(ValueError, match="sliding_window"):
            gx.assert_case_lengths((_case("short", WINDOW),), TINY_SYM_MAX, WINDOW)


#: `_sanity` の被験体（ケース名 → 期待トークン id）と表示名。
SANITY_EXPECTED = {"a": 10, "b": 20}
SANITY_LABELS = {10: "▁x", 20: "▁y", 30: "▁z"}


class TestSanity:
    def test_matching_greedy_tokens_pass(self):
        result = gx._sanity({"a": 10, "b": 20}, SANITY_EXPECTED, SANITY_LABELS)

        assert result == {"a": "10(▁x)", "b": "20(▁y)"}

    def test_a_mismatch_fails_loudly(self):
        with pytest.raises(AssertionError, match="期待継続と違う"):
            gx._sanity({"a": 30, "b": 20}, SANITY_EXPECTED, SANITY_LABELS)

    def test_a_constant_output_fails_loudly(self):
        """MUST: 期待表を緩めても残る検出線（両ケースが同じ 1 位 = 定数出力）。"""
        with pytest.raises(AssertionError, match="定数出力"):
            gx._sanity({"a": 10, "b": 10}, {"a": 10, "b": 10}, SANITY_LABELS)

    def test_a_case_set_mismatch_fails_loudly(self):
        with pytest.raises(AssertionError, match="ケース集合"):
            gx._sanity({"a": 10}, SANITY_EXPECTED, SANITY_LABELS)


class TestGreedyTokens:
    def test_the_last_position_is_the_one_that_is_read(self):
        """MUST: 最終位置。他の位置を見ると「継続の予測」ではない量を突合することになる。"""
        logits = torch.zeros(1, 3, 5)
        logits[0, 0, 4] = 1.0
        logits[0, -1, 2] = 1.0

        assert gx.greedy_tokens({"case": logits}) == {"case": 2}


class _StubTokenizer:
    """`encode(...).ids` だけを持つ最小のトークナイザ（期待継続の単一トークン検査用）。"""

    def __init__(self, ids: list[int]) -> None:
        self._ids = ids

    def encode(self, text: str, add_special_tokens: bool = True):
        """`tokenizers.Tokenizer.encode` の呼び出し規約だけ合わせる（引数は使わない）。"""
        return SimpleNamespace(ids=list(self._ids))


class TestExpectedTokenIds:
    def test_a_single_token_continuation_is_accepted(self):
        assert set(gx.expected_token_ids(_StubTokenizer([7])).values()) == {7}

    def test_a_multi_token_continuation_fails_loudly(self):
        """MUST: 複数トークンだと「1 位の一致」の意味が定まらない（黙って先頭だけ見ない）。"""
        with pytest.raises(AssertionError, match="単一トークン"):
            gx.expected_token_ids(_StubTokenizer([7, 8]))


class TestGoldenCases:
    def test_every_case_has_a_greedy_expectation(self):
        """MUST: 期待の無いケースを混ぜない（sanity が一部のケースだけ見る形にしない）。"""
        assert {name for name, _ in gx.GOLDEN_CASES} == set(gx.GREEDY_EXPECTATIONS)

    def test_the_expectations_are_not_all_the_same_token(self):
        """定数出力の検出線が恒真にならない条件（期待が 1 種類だと `_sanity` の後段が死ぬ）。"""
        assert len(set(gx.GREEDY_EXPECTATIONS.values())) > 1
