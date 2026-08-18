"""states 形への IR 手術（`karume.states`）の振る舞いを固定する。

手術の受入条件は「書ける IR」ではなく「**ランタイムが読める IR**」なので、主要な形は
往復（`to_json` → `parse_ir_graph` → `assert_op_contracts` / `assert_runtime_support`）まで
通す。手術側で規則を再実装せず既存の門に預けている以上、門を通ることそのものが仕様。

被験グラフは合成の小グラフ（torch 不要）— 実モデルは 894 ノードあり、手術の対象は
「attention と mask の畳み込み残骸」だけなので、その形だけを最小で写す。
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence

import pytest

from karume.ir import IrGraph
from karume.ops import OpContractError
from karume.states import StateAttentionSpec, StatesFormError, StatesPlan, to_states_form
from karume.verify import IrError, assert_op_contracts, assert_runtime_support, parse_ir_graph

#: q の宣言 shape（`[B,H,M,D]` — H=8）と k / v の宣言 shape（`[B,Hkv,M,D]` — Hkv=2 の GQA 4:1）。
Q = [1, 8, "T", 16]
KV = [1, 2, "T", 16]

#: 加算 causal mask を焼いた定数の Tmax（`sym_prefix_slice` が先頭 T×T を切り出す — ADR 0010）。
TMAX = 64


def _input(name: str, shape: Sequence) -> dict:
    return {"name": name, "dtype": "f32", "shape": list(shape)}


def _value(shape: Sequence) -> dict:
    return {"dtype": "f32", "shape": list(shape)}


def _attention(q: str, k: str, v: str, out: str) -> dict:
    """従来形（mask 第 4 入力）の attention ノード。"""
    return {"op": "attention", "ins": [q, k, v, "mask"], "outs": [out], "attrs": {"scale": 0.25}}


#: Tmax² 定数 → 束縛後の T×T を切り出す畳み込み残骸。states 形では causal が述語計算になるので
#: 死ぬ（刈り込みの主目標）。
MASK_NODE = {
    "op": "sym_prefix_slice",
    "ins": ["mask.table"],
    "outs": ["mask"],
    "attrs": {
        "sym": "T",
        "slices": [
            {"dim": 2, "coeff": 1, "offset": 0},
            {"dim": 3, "coeff": 1, "offset": 0},
        ],
    },
}


def source_dict(**overrides) -> dict:
    """chunk 局所 causal self-attention 1 層。

    mask は Tmax² 定数 + `sym_prefix_slice`（実モデルの畳み込み結果と同じ形）で、attention の
    後段には**生きた** initializer を食うノードを 1 本置く（刈り込みが生きた宣言まで巻き込んで
    いないことの対照）。
    """
    graph = {
        "format": "karume-ir",
        "version": 1,
        "requires": {"ops": ["add", "attention", "sym_prefix_slice"]},
        "symbols": ["T"],
        "inputs": [_input("q", Q), _input("k", KV), _input("v", KV)],
        "outputs": ["o"],
        "initializers": {
            "mask.table": {"tensor": "mask.table", "storage": {"dtype": "f32"}},
            "bias": {"tensor": "attn.bias", "storage": {"dtype": "f32"}},
        },
        "values": {
            "mask.table": _value([1, 1, TMAX, TMAX]),
            "bias": _value([16]),
            "mask": _value([1, 1, "T", "T"]),
            "a": _value(Q),
            "o": _value(Q),
        },
        "nodes": [
            MASK_NODE,
            _attention("q", "k", "v", "a"),
            {"op": "add", "ins": ["a", "bias"], "outs": ["o"], "attrs": {}},
        ],
    }
    graph.update(overrides)
    if "requires" not in overrides:
        graph["requires"] = {"ops": sorted({node["op"] for node in graph["nodes"]})}
    return graph


def two_layer_dict(*, shared: bool = True) -> dict:
    """attention 2 本のグラフ。

    `shared=True` は KV 共有層の配線（層 1 が所有層の k/v 値テンソルをそのまま `ins` に取る —
    ADR 0067 決定 4）、`shared=False` は層ごとに独立した k/v を持つ形。
    """
    extra = [_input("q1", Q)] if shared else [_input("q1", Q), _input("k1", KV), _input("v1", KV)]
    k1, v1 = ("k", "v") if shared else ("k1", "v1")
    return {
        "format": "karume-ir",
        "version": 1,
        "requires": {"ops": ["add", "attention", "sym_prefix_slice"]},
        "symbols": ["T"],
        "inputs": [_input("q", Q), _input("k", KV), _input("v", KV), *extra],
        "outputs": ["o"],
        "initializers": {"mask.table": {"tensor": "mask.table", "storage": {"dtype": "f32"}}},
        "values": {
            "mask.table": _value([1, 1, TMAX, TMAX]),
            "mask": _value([1, 1, "T", "T"]),
            "a": _value(Q),
            "a1": _value(Q),
            "o": _value(Q),
        },
        "nodes": [
            MASK_NODE,
            _attention("q", "k", "v", "a"),
            _attention("q1", k1, v1, "a1"),
            {"op": "add", "ins": ["a", "a1"], "outs": ["o"], "attrs": {}},
        ],
    }


def source(**overrides) -> IrGraph:
    """被験グラフ。**パーサを通して作る** — 手術の入力が valid な IR であることが前提。"""
    return parse_ir_graph(json.dumps(source_dict(**overrides)))


def plan(*attentions: StateAttentionSpec, capacity_symbol: str = "C") -> StatesPlan:
    return StatesPlan(capacity_symbol=capacity_symbol, attentions=attentions)


#: 1 層グラフの標準の指定（full context・容量記号）。
SPEC = StateAttentionSpec(output="a", k_slot="l0.k", v_slot="l0.v")


def checked(graph: IrGraph) -> IrGraph:
    """手術結果を JSON へ落として**パーサと契約検査へ掛け直す**（往復が受入条件）。"""
    parsed = parse_ir_graph(graph.to_json())
    assert_op_contracts(parsed)
    assert_runtime_support(parsed)
    return parsed


def ops_of(graph: IrGraph) -> list[str]:
    return [node.op for node in graph.nodes]


class TestASingleLayerSurgery:
    def test_the_mask_input_is_replaced_by_the_states_field(self):
        """mask を落として `states` 欄を付ける（causal は述語計算 — ADR 0067 決定 4）。"""
        node = to_states_form(source(), plan(SPEC)).nodes[0]

        assert node.op == "attention"
        assert node.ins == ["q", "k", "v"]
        assert node.states == {"k": "l0.k", "v": "l0.v"}
        # 既存 attrs は素通り（半スケール契約は手術の関心事ではない）。
        assert node.attrs == {"scale": 0.25}

    def test_an_append_is_inserted_after_the_reader(self):
        """スロットごとに `state_append` を 1 本、読者の直後へ（ADR 0067 決定 5b の①）。

        後段の `add` は append の**後ろ**に残る = 残存ノードの相対順が保たれている。
        """
        graph = to_states_form(source(), plan(SPEC))

        assert ops_of(graph) == ["attention", "state_append", "state_append", "add"]
        assert [node.ins for node in graph.nodes[1:3]] == [["k"], ["v"]]
        assert [node.states for node in graph.nodes[1:3]] == [
            {"slot": "l0.k"},
            {"slot": "l0.v"},
        ]
        assert [node.outs for node in graph.nodes[1:3]] == [[], []]

    def test_the_slots_are_declared_with_the_capacity_symbol(self):
        """スロットは ins の `[B,Hkv,M,D]` から `[B,Hkv,C,D]` を導く（容量だけが差し替わる）。"""
        graph = to_states_form(source(), plan(SPEC))

        assert list(graph.states) == ["l0.k", "l0.v"]
        assert [slot.shape for slot in graph.states.values()] == [[1, 2, "C", 16]] * 2
        assert {slot.dtype for slot in graph.states.values()} == {"f32"}
        assert graph.symbols == ["T", "C"]

    def test_the_folded_mask_constant_and_its_producer_are_pruned(self):
        """Tmax² 定数と `sym_prefix_slice` は誰も読まなくなる = 配布物に残さない。"""
        graph = to_states_form(source(), plan(SPEC))

        assert "sym_prefix_slice" not in ops_of(graph)
        assert list(graph.initializers) == ["bias"]
        # 死んだ値の宣言も落ちる（孤立宣言は verify が拒否する）。
        assert list(graph.values) == ["bias", "a", "o"]

    def test_the_source_graph_is_left_untouched(self):
        """純関数 MUST — 手術前後の 2 本を同じ検証に掛けられないと出どころを切り分けられない。"""
        graph = source()
        before = graph.to_json()

        to_states_form(graph, plan(SPEC))

        assert graph.to_json() == before

    def test_the_result_passes_the_ir_gates(self):
        """states 検査・順序検査・shape 検査・列挙門の全部を往復で通る。"""
        graph = checked(to_states_form(source(), plan(SPEC)))

        assert graph.required_ops == ["add", "attention", "state_append"]


class TestSlidingWindow:
    def test_every_node_touching_the_slot_declares_the_window(self):
        """論理 col → 物理 row の写像は読み書き同式 MUST（ADR 0067 決定 4）。"""
        spec = StateAttentionSpec(output="a", k_slot="l0.k", v_slot="l0.v", window=32)

        graph = checked(to_states_form(source(), plan(spec)))

        assert [node.attrs.get("window") for node in graph.nodes[:3]] == [32, 32, 32]

    def test_a_window_larger_than_the_capacity_is_rejected_by_the_slot_contract(self):
        """スロット契約（`window ≤ C`）との整合 — 数値容量なら宣言だけで決まる。"""
        spec = StateAttentionSpec(output="a", k_slot="l0.k", v_slot="l0.v", window=128, capacity=64)

        with pytest.raises(OpContractError, match="スロット容量 64 を超える"):
            checked(to_states_form(source(), plan(spec)))


class TestSharedSlots:
    """KV 共有層（同じスロットを複数の attention が読む — Gemma 4 E2B の末尾 20 層）。"""

    def test_the_append_follows_the_last_reader(self):
        """append は**全読者の後**（ring wrap が今 step の読者の過去行を潰さない — 決定 5b）。"""
        graph = checked(
            to_states_form(
                parse_ir_graph(json.dumps(two_layer_dict())),
                plan(SPEC, StateAttentionSpec(output="a1", k_slot="l0.k", v_slot="l0.v")),
            )
        )

        assert ops_of(graph) == [
            "attention",
            "attention",
            "state_append",
            "state_append",
            "add",
        ]
        assert list(graph.states) == ["l0.k", "l0.v"]

    def test_readers_wiring_a_different_value_are_rejected(self):
        """共有層は所有層の k/v 値テンソルをそのまま配線する — 別の値なら配線ミス。"""
        graph = parse_ir_graph(json.dumps(two_layer_dict(shared=False)))

        with pytest.raises(StatesFormError, match=re.escape("へ書く値が読者ごとに違う")):
            to_states_form(
                graph, plan(SPEC, StateAttentionSpec(output="a1", k_slot="l0.k", v_slot="l0.v"))
            )

    def test_readers_disagreeing_on_the_capacity_are_rejected(self):
        """容量が読者ごとに違うと、同じスロットに 2 つの物理形を要求することになる。"""
        graph = parse_ir_graph(json.dumps(two_layer_dict()))
        shared = StateAttentionSpec(output="a1", k_slot="l0.k", v_slot="l0.v", capacity=128)

        with pytest.raises(StatesFormError, match=re.escape("導出 shape が読者ごとに違う")):
            to_states_form(graph, plan(SPEC, shared))

    def test_readers_disagreeing_on_the_window_are_rejected(self):
        """窓幅の食い違いは沈黙誤読（片方だけ別の物理 row を読む）。"""
        graph = parse_ir_graph(json.dumps(two_layer_dict()))
        shared = StateAttentionSpec(output="a1", k_slot="l0.k", v_slot="l0.v", window=32)

        with pytest.raises(StatesFormError, match=re.escape("window が読者ごとに違う")):
            to_states_form(graph, plan(SPEC, shared))


class TestCapacity:
    def test_a_numeric_capacity_replaces_the_symbol(self):
        spec = StateAttentionSpec(output="a", k_slot="l0.k", v_slot="l0.v", capacity=256)

        graph = checked(to_states_form(source(), plan(spec)))

        assert graph.states["l0.k"].shape == [1, 2, 256, 16]

    def test_the_symbol_is_absent_when_no_slot_uses_it(self):
        """使わない記号を宣言すると束縛点の無い記号としてグラフごと拒否される。"""
        spec = StateAttentionSpec(output="a", k_slot="l0.k", v_slot="l0.v", capacity=256)

        graph = checked(to_states_form(source(), plan(spec)))

        assert graph.symbols == ["T"]

    def test_a_mixed_plan_declares_the_symbol_once(self):
        """層ごとに容量が違う形（Gemma 4 の sliding / full 混在）。

        記号を使う層が 1 つでもあれば足す（数値容量の層は記号を使わない）。
        """
        layer1 = StateAttentionSpec(output="a1", k_slot="l1.k", v_slot="l1.v", capacity=TMAX)

        graph = checked(
            to_states_form(
                parse_ir_graph(json.dumps(two_layer_dict(shared=False))), plan(SPEC, layer1)
            )
        )

        assert graph.symbols == ["T", "C"]
        assert graph.states["l0.k"].shape == [1, 2, "C", 16]
        assert graph.states["l1.k"].shape == [1, 2, TMAX, 16]
        assert ops_of(graph) == [
            "attention",
            "state_append",
            "state_append",
            "attention",
            "state_append",
            "state_append",
            "add",
        ]


class TestRejectedSurgery:
    def test_an_unknown_target_is_rejected(self):
        with pytest.raises(StatesFormError, match="この値を定義するノードがグラフに無い"):
            to_states_form(source(), plan(StateAttentionSpec("ghost", "l0.k", "l0.v")))

    def test_a_node_that_is_not_attention_is_rejected(self):
        with pytest.raises(StatesFormError, match=re.escape("op が 'sym_prefix_slice'")):
            to_states_form(source(), plan(StateAttentionSpec("mask", "l0.k", "l0.v")))

    def test_an_attention_without_a_mask_is_rejected(self):
        """mask を持たない export は causal が別の形で表されている = 手術の前提が崩れている。"""
        nodes = [dict(node) for node in source_dict()["nodes"]]
        nodes[1] = {**nodes[1], "ins": ["q", "k", "v"]}

        with pytest.raises(StatesFormError, match="入力が 3 本"):
            to_states_form(source(nodes=nodes), plan(SPEC))

    def test_a_second_surgery_on_the_same_node_is_rejected(self):
        """二重手術は「今 step の k/v を 2 度書く」形になる（append も 2 本になる）。"""
        once = to_states_form(source(), plan(SPEC))

        with pytest.raises(StatesFormError, match="すでに states 形"):
            to_states_form(once, plan(SPEC))

    def test_a_capacity_symbol_colliding_with_an_existing_symbol_is_rejected(self):
        """入力由来の束縛を持つ記号を容量にも使うと、値 shape が容量で解けてしまう。"""
        with pytest.raises(StatesFormError, match=re.escape("graph.symbols に既にある")):
            to_states_form(source(), plan(SPEC, capacity_symbol="T"))

    @pytest.mark.parametrize("symbol", ["q", "o"])
    def test_a_capacity_symbol_colliding_with_a_value_name_is_rejected(self, symbol):
        """記号の欄へ値名（入力 / ノード出力）を渡した取り違えを検出する。"""
        with pytest.raises(StatesFormError, match="値名と同名"):
            to_states_form(source(), plan(SPEC, capacity_symbol=symbol))

    @pytest.mark.parametrize("window", [0, -1, True, 1.5])
    def test_a_window_outside_the_value_range_is_rejected(self, window):
        """bool は int の派生だが窓幅ではない（`True` が窓幅 1 として黙って通る）。"""
        spec = StateAttentionSpec(output="a", k_slot="l0.k", v_slot="l0.v", window=window)

        with pytest.raises(StatesFormError, match="が正整数でない"):
            to_states_form(source(), plan(spec))

    def test_an_input_orphaned_by_the_surgery_is_rejected(self):
        """mask をグラフ入力から作っている export は states 形へ落とせない（配線ミス）。"""
        graph = source_dict()
        nodes = [dict(node) for node in graph["nodes"]]
        nodes[1] = {**nodes[1], "ins": ["q", "k", "v", "m"]}

        with pytest.raises(StatesFormError, match=re.escape("入力 ['m'] が到達不能")):
            to_states_form(
                source(inputs=[*graph["inputs"], _input("m", [1, 1, "T", "T"])], nodes=nodes),
                plan(SPEC),
            )


class TestTheOrderCheckIsEffective:
    """手術の出す並びが「偶然通っている」のではないことの故障注入（ADR 0067 決定 5b の②）。"""

    def test_moving_an_append_before_its_reader_is_rejected(self):
        """append を読者の前へ動かすと順序検査が落ちる = 検査が実効。"""
        graph = to_states_form(source(), plan(SPEC)).to_dict()
        nodes = graph["nodes"]
        # attention と k の append を入れ替える（データ辺は入力しか使わないので SSA は valid の
        # まま — 落ちるのは順序検査だけ）。
        graph["nodes"] = [nodes[1], nodes[0], *nodes[2:]]
        parsed = parse_ir_graph(json.dumps(graph))

        with pytest.raises(IrError, match="より後に読者"):
            assert_op_contracts(parsed)
