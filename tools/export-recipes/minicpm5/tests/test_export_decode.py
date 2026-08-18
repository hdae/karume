"""`minicpm5/export_decode.py` の台本レベルの約束事（実重み不要分）。

実重みの emit は手動（1-shot 形のテストと同じ規律）。ここで固定するのは、壊れると**偽 PASS**
になる側の規律だけ:

- RoPE 表引きが元実装と**値同一**であること（差し替えの正当性そのもの）と、表で表せない
  実装（位置以外に依存する forward）が突合で落ちること
- `states_plan` が nodes 順に層番号つきスロットを割り当て、本数の食い違いで落ちること
- `assert_ir_form_decode` が「数値は合うが静かに壊れた」形を**実際に検出**すること
  （従来形の残り / repeat_kv 実体化 / window 付き / 容量の焼き込み / 残骸 op / 出口の取り違え）
- greedy の margin 門が下限以下の step を落とすこと（恒真でない余裕保証）
- `greedy_continuation` が **full re-forward**（毎 step 先頭から・位置は arange）で採ること

transformers を要するケースだけ `importorskip` で SKIP する（既定 sync の CI ではモデル系
依存が入らない — ADR 0065 の 2 job 構成）。
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.export import Dim

from karume.convert import PRESERVED_OP_PREFIXES_WITH_ATTENTION
from karume.ir import IrGraph, IrInitializer, IrInput, IrNode, IrStorage, IrValue
from karume.pipeline import export_module
from karume.states import StateAttentionSpec, StatesFormError, StatesPlan, to_states_form
from minicpm5 import export as one_shot
from minicpm5 import export_decode as decode
from minicpm5.tests.test_export import (
    HEAD_DIM,
    HEADS,
    HIDDEN,
    KV_HEADS,
    LAYERS,
    TINY_SYM_MAX,
    VOCAB,
    _tiny_checkpoint,
)

#: tiny な RoPE 表の位置数（{@link decode.assert_rope_table_matches} の probe が 17 と
#: `positions - 1` を踏むので 18 以上が要る）。
TINY_POSITIONS = 32

#: 加算 causal mask を焼いた定数の Tmax（手術で刈られる残骸 — ADR 0010）。
TINY_TMAX = TINY_SYM_MAX

TINY_IR_CONFIG = SimpleNamespace(
    num_attention_heads=HEADS,
    num_key_value_heads=KV_HEADS,
    head_dim=HEAD_DIM,
    num_hidden_layers=LAYERS,
)


class _StubRotary(nn.Module):
    """`LlamaRotaryEmbedding` の呼び出し規約だけを持つ被験体。

    値は**位置だけの決定的な関数**（表引きで厳密に再現できる形）。`x` は dtype / device の
    ためだけに受ける引数で、上流実装と同じく値は読まない。
    """

    def __init__(self, depth: int) -> None:
        super().__init__()
        self.scale = torch.arange(depth, dtype=torch.float32) * 0.017 + 0.31

    def angles(self, position_ids: torch.Tensor) -> torch.Tensor:
        return position_ids.unsqueeze(-1).to(torch.float32) * self.scale

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor):
        del x
        angles = self.angles(position_ids)
        return angles.cos(), angles.sin()


class _LengthDependentRotary(_StubRotary):
    """位置**以外**（系列長）にも依存する rotary — 表引きでは原理的に再現できない被験体。"""

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor):
        del x
        angles = self.angles(position_ids) + float(position_ids.shape[1])
        return angles.cos(), angles.sin()


@pytest.fixture
def probe() -> torch.Tensor:
    """rotary が dtype / device を読むためのダミー入力。"""
    return torch.zeros(1, 1, HIDDEN, dtype=torch.float32)


class TestRopeTable:
    def test_the_lookup_reproduces_the_reference(self, probe):
        rotary = _StubRotary(HEAD_DIM)

        table = decode.build_rope_table(rotary, probe, TINY_POSITIONS)

        # 例外が出なければ全 probe で torch.equal（値・shape・dtype・タプル構成の全一致）。
        decode.assert_rope_table_matches(rotary, table, probe, TINY_POSITIONS)
        assert table.cos_table.shape == (TINY_POSITIONS, HEAD_DIM)
        assert table.sin_table.shape == (TINY_POSITIONS, HEAD_DIM)

    def test_a_reference_that_is_not_a_function_of_position_is_rejected(self, probe):
        """MUST: 表で表せない実装を黙って表に落とすと、値が「それらしく」ずれる。"""
        rotary = _LengthDependentRotary(HEAD_DIM)
        table = decode.build_rope_table(rotary, probe, TINY_POSITIONS)

        with pytest.raises(AssertionError, match="表の引き方がずれている"):
            decode.assert_rope_table_matches(rotary, table, probe, TINY_POSITIONS)

    def test_a_shifted_table_is_rejected(self, probe):
        """故障注入: 表を 1 行ずらすと突合が落ちる（= 上の緑が恒真でないことの裏取り）。"""
        rotary = _StubRotary(HEAD_DIM)
        table = decode.build_rope_table(rotary, probe, TINY_POSITIONS)
        shifted = decode.RopeTable(table.cos_table.roll(1, 0), table.sin_table)

        with pytest.raises(AssertionError, match="表の引き方がずれている"):
            decode.assert_rope_table_matches(rotary, shifted, probe, TINY_POSITIONS)

    def test_the_swap_leaves_the_model_untouched_when_the_check_fails(self, probe):
        """MUST: 突合の失敗で模型が半端な状態（表引きだが値が違う）に残らない。"""
        model = SimpleNamespace(
            model=SimpleNamespace(rotary_emb=_LengthDependentRotary(HEAD_DIM)),
            config=SimpleNamespace(hidden_size=HIDDEN),
        )
        original = model.model.rotary_emb

        with pytest.raises(AssertionError):
            decode.swap_rope_table(model, TINY_POSITIONS)

        assert model.model.rotary_emb is original


def _pre_surgery_graph(kv_heads: int = KV_HEADS, layers: int = LAYERS) -> IrGraph:
    """従来形 attention `layers` 本を持つ decode chunk グラフの**骨組み**。

    検査対象は attention / `state_append` / states 節と、入力・出力・残骸 op の有無だけなので、
    q / k / v を作る側は「宣言 shape を持つ生きたノード」であればよい（op の意味は問わない）。

    MUST: 層は**直列に繋ぐ**（層 i の q/k/v は層 i-1 の出力から作る）。繋がない層は手術の
    刈り込み（`to_states_form` は出力から到達しないノードを落とす）で消え、被験体が黙って
    1 層のグラフになる。
    """
    graph = IrGraph(symbols=[decode.SEQ_SYMBOL])
    graph.inputs.append(IrInput(name=decode.INPUT_IDS, dtype="i32", shape=[1, "M"]))
    graph.inputs.append(IrInput(name=decode.POSITION_IDS, dtype="i32", shape=[1, "M"]))
    for name, shape in (
        ("tok.table", [VOCAB, HIDDEN]),
        ("rope.cos", [TINY_POSITIONS, HEAD_DIM]),
        ("mask.table", [1, 1, TINY_TMAX, TINY_TMAX]),
    ):
        graph.initializers[name] = IrInitializer(
            tensor=f"const.{name}", storage=IrStorage(dtype="f32")
        )
        graph.values[name] = IrValue(dtype="f32", shape=shape)

    graph.values["h"] = IrValue(dtype="f32", shape=[1, "M", HIDDEN])
    graph.values["cos"] = IrValue(dtype="f32", shape=[1, "M", HEAD_DIM])
    graph.values["mask"] = IrValue(dtype="f32", shape=[1, 1, "M", "M"])
    graph.nodes.append(
        IrNode(op="embedding", ins=["tok.table", decode.INPUT_IDS], outs=["h"], attrs={})
    )
    graph.nodes.append(
        IrNode(op="embedding", ins=["rope.cos", decode.POSITION_IDS], outs=["cos"], attrs={})
    )
    graph.nodes.append(
        IrNode(
            op=one_shot.SYM_PREFIX_SLICE_OP,
            ins=["mask.table"],
            outs=["mask"],
            attrs={"sym": "M", "slices": [{"dim": 2, "coeff": 1, "offset": 0}]},
        )
    )
    for layer in range(layers):
        source = "h" if layer == 0 else f"attn{layer - 1}"
        for slot, count in (("q", HEADS), ("k", kv_heads), ("v", kv_heads)):
            name = f"{slot}{layer}"
            graph.values[name] = IrValue(dtype="f32", shape=[1, count, "M", HEAD_DIM])
            graph.nodes.append(IrNode(op="mul", ins=[source, "cos"], outs=[name], attrs={}))
        graph.values[f"attn{layer}"] = IrValue(dtype="f32", shape=[1, HEADS, "M", HEAD_DIM])
        graph.nodes.append(
            IrNode(
                op=one_shot.ATTENTION_OP,
                ins=[f"q{layer}", f"k{layer}", f"v{layer}", "mask"],
                outs=[f"attn{layer}"],
                attrs={"scale": 0.5},
            )
        )
    graph.values["logits"] = IrValue(dtype="f32", shape=[1, "M", VOCAB])
    graph.values["token"] = IrValue(dtype="i32", shape=[1, "M", 1])
    graph.nodes.append(
        IrNode(op="mul", ins=[f"attn{layers - 1}", f"attn{layers - 1}"], outs=["logits"], attrs={})
    )
    graph.nodes.append(IrNode(op="argmax", ins=["logits"], outs=["token"], attrs={}))
    graph.outputs.extend(["logits", "token"])
    return graph


def _surgical_graph(**kwargs) -> IrGraph:
    """{@link _pre_surgery_graph} を全 attention について states 形へ手術したグラフ。"""
    graph = _pre_surgery_graph(**kwargs)
    return to_states_form(graph, decode.states_plan(graph, kwargs.get("layers", LAYERS)))


class TestStatesPlan:
    def test_the_slots_are_named_per_layer_in_node_order(self):
        graph = _pre_surgery_graph()

        plan = decode.states_plan(graph, LAYERS)

        assert plan.capacity_symbol == decode.CAPACITY_SYMBOL
        assert [(spec.k_slot, spec.v_slot) for spec in plan.attentions] == [
            ("l0.k", "l0.v"),
            ("l1.k", "l1.v"),
        ]
        # nodes 順の attention に 1 対 1（取り違えると別層の KV へ書く形になる）
        assert [spec.output for spec in plan.attentions] == ["attn0", "attn1"]
        assert all(spec.window is None and spec.capacity is None for spec in plan.attentions)

    def test_a_layer_count_mismatch_fails_loudly(self):
        """MUST: 取りこぼした層は chunk 局所 causal のまま残る（沈黙誤値）。"""
        graph = _pre_surgery_graph(layers=1)

        with pytest.raises(AssertionError, match="層と一致しない"):
            decode.states_plan(graph, LAYERS)


class TestAssertIrFormDecode:
    def test_the_states_form_passes(self):
        form = decode.assert_ir_form_decode(_surgical_graph(), TINY_IR_CONFIG)

        assert form["attention_nodes"] == LAYERS
        assert form["state_append_nodes"] == 2 * LAYERS
        assert form["slots"] == 2 * LAYERS
        assert form["slot_shape"] == [1, KV_HEADS, decode.CAPACITY_SYMBOL, HEAD_DIM]
        assert form["heads"] == [HEADS, KV_HEADS, KV_HEADS]

    def test_a_conventional_attention_left_behind_is_rejected(self):
        """MUST: 手術し損ねた層は過去を見ない（数値は動くので golden では捕まらない）。"""
        graph = _pre_surgery_graph()
        partial = StatesPlan(
            capacity_symbol=decode.CAPACITY_SYMBOL,
            attentions=(StateAttentionSpec(output="attn0", k_slot="l0.k", v_slot="l0.v"),),
        )

        with pytest.raises(AssertionError, match="q / k / v の 3 本ちょうど"):
            decode.assert_ir_form_decode(to_states_form(graph, partial), TINY_IR_CONFIG)

    def test_a_materialized_kv_shape_is_rejected(self):
        with pytest.raises(AssertionError, match="repeat_kv"):
            decode.assert_ir_form_decode(_surgical_graph(kv_heads=HEADS), TINY_IR_CONFIG)

    def test_a_derived_head_dim_is_rejected(self):
        derived = SimpleNamespace(
            num_attention_heads=HEADS,
            num_key_value_heads=KV_HEADS,
            head_dim=HIDDEN // HEADS,
            num_hidden_layers=LAYERS,
        )

        with pytest.raises(AssertionError, match=r"config\.head_dim"):
            decode.assert_ir_form_decode(_surgical_graph(), derived)

    def test_a_window_attribute_is_rejected(self):
        """全 context の full 層に window が付くと、窓外の過去を黙って切り捨てる。"""
        graph = _surgical_graph()
        index, node = next(
            (index, node)
            for index, node in enumerate(graph.nodes)
            if node.op == one_shot.ATTENTION_OP
        )
        graph.nodes[index] = replace(node, attrs={**node.attrs, "window": 4})

        with pytest.raises(AssertionError, match="window"):
            decode.assert_ir_form_decode(graph, TINY_IR_CONFIG)

    def test_a_baked_capacity_is_rejected(self):
        """容量を数値で焼くと `createGenerationContext` で容量を選べない（ADR 0066 決定 3）。"""
        graph = _pre_surgery_graph()
        baked = StatesPlan(
            capacity_symbol=decode.CAPACITY_SYMBOL,
            attentions=tuple(
                StateAttentionSpec(
                    output=f"attn{layer}",
                    k_slot=decode.slot_name(layer, "k"),
                    v_slot=decode.slot_name(layer, "v"),
                    capacity=8,
                )
                for layer in range(LAYERS)
            ),
        )

        with pytest.raises(AssertionError, match="容量は記号のまま残す"):
            decode.assert_ir_form_decode(to_states_form(graph, baked), TINY_IR_CONFIG)

    def test_a_missing_state_append_is_rejected(self):
        graph = _surgical_graph()
        graph.nodes = [node for node in graph.nodes if node.states.get("slot") != "l1.v"]

        with pytest.raises(AssertionError, match="全スロット 1 本ずつでない"):
            decode.assert_ir_form_decode(graph, TINY_IR_CONFIG)

    def test_a_mask_left_as_a_graph_input_is_rejected(self):
        graph = _surgical_graph()
        graph.inputs.append(IrInput(name="attention_mask", dtype="f32", shape=[1, 1, "M", "M"]))

        with pytest.raises(AssertionError, match="グラフ入力"):
            decode.assert_ir_form_decode(graph, TINY_IR_CONFIG)

    @pytest.mark.parametrize("op", decode.RESIDUE_OPS)
    def test_a_leftover_residue_op_is_rejected(self, op):
        """MUST: 誰も読まない Tmax 定数 / 畳み残した RoPE が配布物に居座る形を落とす。"""
        graph = _surgical_graph()
        graph.nodes.append(IrNode(op=op, ins=["logits"], outs=["residue"], attrs={}))

        with pytest.raises(AssertionError, match="手術で死ぬはずの op"):
            decode.assert_ir_form_decode(graph, TINY_IR_CONFIG)

    def test_an_exit_that_is_not_argmax_is_rejected(self):
        """出力 1 は decode 出口（ADR 0068 決定 4）— logits を 2 本出す形と取り違えない。"""
        graph = _surgical_graph()
        graph.outputs[1] = "logits"

        with pytest.raises(AssertionError, match="出力 1 の供給元"):
            decode.assert_ir_form_decode(graph, TINY_IR_CONFIG)

    def test_a_single_output_graph_is_rejected(self):
        graph = _surgical_graph()
        graph.outputs.pop()

        with pytest.raises(AssertionError, match="decode 出口は logits / token の 2 本"):
            decode.assert_ir_form_decode(graph, TINY_IR_CONFIG)

    def test_a_capacity_symbol_that_collides_with_an_input_symbol_fails_loudly(self):
        """記号の取り違え（容量を入力由来の記号に重ねる）は手術側が落とす。"""
        graph = _pre_surgery_graph()
        colliding = StatesPlan(
            capacity_symbol=decode.SEQ_SYMBOL,
            attentions=decode.states_plan(graph, LAYERS).attentions,
        )

        with pytest.raises(StatesFormError, match=r"graph\.symbols に既にある"):
            to_states_form(graph, colliding)


class TestGreedyMargins:
    def test_a_comfortable_margin_passes(self):
        decode.assert_greedy_margins({"a": [1.0, 0.5, 0.011], "b": [2.0]}, decode.MARGIN_FLOOR)

    def test_a_thin_margin_fails_loudly(self):
        """MUST: 余裕の無い step を golden に混ぜると GPU 偏差で列が割れる。"""
        with pytest.raises(AssertionError, match=r"margin が下限"):
            decode.assert_greedy_margins({"a": [1.0, 1e-3, 0.5]}, decode.MARGIN_FLOOR)

    def test_the_floor_itself_is_not_accepted(self):
        """境界は「超える」— ちょうど下限は通さない。"""
        with pytest.raises(AssertionError, match=r"margin が下限"):
            decode.assert_greedy_margins({"a": [decode.MARGIN_FLOOR]}, decode.MARGIN_FLOOR)

    def test_every_offending_case_is_reported_at_once(self):
        """MUST: 最初の 1 件で止めない — 外すケースの判断材料が 1 回の実走で揃わなくなる。"""
        with pytest.raises(AssertionError) as failure:
            decode.assert_greedy_margins(
                {"a": [1.0], "b": [1e-3], "c": [0.5, 0.0]}, decode.MARGIN_FLOOR
            )

        message = str(failure.value)
        assert "'b'" in message and "'c'" in message
        # 通ったケースは診断に出さない（外す対象の名前だけが並ぶ）
        assert "'a'" not in message


class _StubWrapper(nn.Module):
    """呼び出し形を記録する chunk ラッパの被験体（logits は最終位置だけを設計する）。

    最終位置の logits は「今の系列長」で決まる決め打ち行を返すので、期待 token 列と margin 列が
    手計算できる。
    """

    def __init__(self, rows: dict[int, list[float]]) -> None:
        super().__init__()
        self.rows = rows
        self.calls: list[tuple[list[int], list[int]]] = []

    def forward(self, input_ids: torch.Tensor, position_ids: torch.Tensor):
        self.calls.append((input_ids[0].tolist(), position_ids[0].tolist()))
        length = int(input_ids.shape[1])
        logits = torch.zeros(1, length, len(self.rows[length]))
        logits[0, -1] = torch.tensor(self.rows[length])
        return logits, logits.argmax(-1, keepdim=True)


class TestGreedyContinuation:
    def test_it_re_forwards_the_whole_prefix_each_step(self):
        """MUST: full re-forward（KV 経路を使わない）— 検収対象の機構で期待値を作らない。"""
        wrapper = _StubWrapper({2: [0.0, 3.0, 1.0], 3: [5.0, 0.0, 1.0], 4: [0.0, 0.0, 2.5]})

        tokens, margins = decode.greedy_continuation(wrapper, torch.tensor([[7, 8]]), 3)

        assert tokens == [1, 0, 2]
        assert margins == pytest.approx([2.0, 4.0, 2.5])
        assert wrapper.calls == [
            ([7, 8], [0, 1]),
            ([7, 8, 1], [0, 1, 2]),
            ([7, 8, 1, 0], [0, 1, 2, 3]),
        ]

    def test_the_positions_are_the_absolute_prefix(self):
        """位置は毎 step `arange(len)`（表引き RoPE が実位置で引けることの前提）。"""
        wrapper = _StubWrapper({3: [1.0, 0.0], 4: [0.0, 2.0]})

        decode.greedy_continuation(wrapper, torch.tensor([[4, 5, 6]]), 2)

        assert [positions for _, positions in wrapper.calls] == [[0, 1, 2], [0, 1, 2, 3]]


class TestCaseRoom:
    def test_a_prompt_that_fits_with_its_continuation_passes(self):
        decode.assert_case_room([("case", torch.zeros(1, 10, dtype=torch.int64))], 6, 16)

    def test_a_continuation_past_the_table_fails_loudly(self):
        """MUST: 表の外の位置は引けない（`F.embedding` は OOB を黙って返さない）。"""
        with pytest.raises(AssertionError, match="RoPE 表の位置数"):
            decode.assert_case_room([("case", torch.zeros(1, 11, dtype=torch.int64))], 6, 16)


class TestGreedyCases:
    def test_every_greedy_case_is_a_golden_case(self):
        """MUST: golden に無いケース名を greedy 側だけで持たない（prompt の出どころが消える）。"""
        assert set(decode.GREEDY_CASES) <= {name for name, _ in one_shot.GOLDEN_CASES}

    def test_every_greedy_case_has_a_first_token_expectation(self):
        """第 1 継続 token の突合（波 A との機構横断検証）が全採用ケースで効く条件。"""
        assert set(decode.GREEDY_CASES) <= set(one_shot.GREEDY_EXPECTATIONS)

    def test_the_adopted_expectations_are_not_all_the_same_token(self):
        """ケースを外した結果、`_sanity` の定数出力検出線が恒真にならないことの条件。"""
        adopted = {one_shot.GREEDY_EXPECTATIONS[name] for name in decode.GREEDY_CASES}

        assert len(adopted) > 1


def _export_tiny_decode(wrapper) -> tuple[IrGraph, dict]:
    ids = torch.randint(0, VOCAB, (1, 7), dtype=torch.int64)
    seq = Dim(decode.SEQ_SYMBOL, min=2, max=TINY_SYM_MAX)
    return export_module(
        wrapper,
        (ids, decode.positions_for(ids)),
        dynamic_shapes=({1: seq}, {1: seq}),
        symbol_names=(decode.SEQ_SYMBOL,),
        preserved=PRESERVED_OP_PREFIXES_WITH_ATTENTION,
    )


class TestExportedDecodeForm:
    """tiny な実モデルを実際に export → 手術 → コンテナ検証まで通す（transformers が要る）。"""

    @pytest.fixture
    def tiny_container(self, tmp_path):
        wrapper = decode.load_wrapper(_tiny_checkpoint(tmp_path / "ckpt"), positions=TINY_POSITIONS)
        graph, tensors = _export_tiny_decode(wrapper)
        surgical = to_states_form(graph, decode.states_plan(graph, LAYERS))
        verified = decode._write_container(surgical, tensors, tmp_path / one_shot.MODEL_FILE)
        return graph, verified, wrapper

    def test_the_container_is_a_verified_states_form_graph(self, tiny_container):
        pre, verified, wrapper = tiny_container

        form = decode.assert_ir_form_decode(verified, wrapper.model.config)

        assert form["attention_nodes"] == LAYERS
        assert form["state_append_nodes"] == 2 * LAYERS
        assert [spec.name for spec in verified.inputs] == [decode.INPUT_IDS, decode.POSITION_IDS]
        assert sorted(verified.symbols) == [decode.CAPACITY_SYMBOL, decode.SEQ_SYMBOL]
        # 手術は必ず何かを刈る（mask の Tmax² 定数）— 0 本なら畳み込みが効いていない
        assert len(pre.initializers) > len(verified.initializers)

    def test_the_mask_residue_is_gone_but_the_pre_surgery_graph_had_it(self, tiny_container):
        """恒真化の門: 刈る対象が export 側に**実在した**ことを同じ資産で見る。"""
        pre, verified, _ = tiny_container

        assert one_shot.SYM_PREFIX_SLICE_OP in pre.required_ops
        assert one_shot.SYM_PREFIX_SLICE_OP not in verified.required_ops

    def test_the_rope_is_a_table_lookup_not_folded_trig(self, tiny_container):
        """位置が入力になった以上、RoPE は畳み込みでなく `embedding` で引かれていること。"""
        _, verified, _ = tiny_container

        readers = [
            node
            for node in verified.nodes
            if node.op == "embedding" and decode.POSITION_IDS in node.ins
        ]

        assert len(readers) == 2
        assert "sin" not in verified.required_ops

    def test_the_wrapper_returns_logits_and_the_greedy_token(self, tmp_path):
        """MUST: 出力順は `[logits, token]`（ランタイムは slot 番号で読む）。"""
        wrapper = decode.load_wrapper(_tiny_checkpoint(tmp_path / "ckpt"), positions=TINY_POSITIONS)
        ids = torch.randint(0, VOCAB, (1, 5), dtype=torch.int64)

        with torch.no_grad():
            logits, token = wrapper(ids, decode.positions_for(ids))

        assert logits.shape == (1, 5, VOCAB)
        assert token.shape == (1, 5, 1)
        assert torch.equal(token, logits.argmax(-1, keepdim=True))
