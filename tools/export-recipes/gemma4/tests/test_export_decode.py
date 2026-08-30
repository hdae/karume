"""`gemma4/export_decode.py` の台本レベルの約束事（実重み不要分）。

実重みの emit は手動（1-shot 形のテストと同じ規律）。ここで固定するのは、壊れると**偽 PASS**
になる側の規律だけ:

- RoPE 表引きが元実装と**値同一**であること（差し替えの正当性そのもの）と、層種別の
  dispatch が効いていること（sliding の表で full 層を引く形が通らない）
- KV 共有の割り付け（所有層 = 共有開始より前で最後の同種層）が上流の規則と一致し、
  `states_plan` が所有層のスロットだけを 1 本ずつ作ること
- `assert_ir_form_decode` が「数値は合うが静かに壊れた」形を**実際に検出**すること
  （従来形の残り / 共有先の取り違え / window の付き違い / 層種別の head_dim / 容量の
  焼き込み / 残骸 op / 出口の取り違え / 格納 dtype の本数）
- greedy の margin 門が下限以下の step を落とすこと（恒真でない余裕保証）
- `greedy_continuation` が **full re-forward**（毎 step 先頭から・位置は arange）で採ること
- 帯 / 因果 mask は 1-shot 台本のものを**そのまま**使うこと（窓の意味論を 2 箇所に持たない）
- chunk 系列 2 本を駆動する variant の分岐（{@link decode.ChunkVariant}）— 組むラッパ・
  `last_row` の有無・golden の有無・出所記録の有無・要約の欄が系列ごとに正しく変わること

transformers を要するケースだけ `importorskip` で SKIP する（既定 sync の CI ではモデル系
依存が入らない — ADR 0065 の 2 job 構成）。
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
from torch import nn
from torch.export import Dim

from _shared import decode_series as shared
from gemma4 import export as gx
from gemma4 import export_decode as decode
from gemma4 import export_token as token_only
from gemma4 import provenance
from gemma4.tests.test_export import (
    FULL_DEPTH,
    HEADS,
    HIDDEN,
    KV_HEADS,
    PLE_DIM,
    SLIDING_DEPTH,
    TINY_SYM_MAX,
    VOCAB,
    WINDOW,
)
from karume.artifacts import SUPERSEDED_SUFFIX
from karume.convert import PRESERVED_OP_PREFIXES_WITH_ATTENTION
from karume.ir import IrGraph, IrInitializer, IrInput, IrNode, IrStorage, IrValue
from karume.pipeline import export_module
from karume.shards import shard_name
from karume.states import StatesFormError, StatesPlan, to_states_form

#: decode 側の tiny な層構成。**KV 共有が成立する最小形**にしてある — 共有開始（層 3）より
#: 前に sliding と full が 1 本ずつ「最後の同種層」として居り、共有側の 2 層がそれぞれを読む。
#: 1-shot 側の 3 層（`test_export.LAYER_TYPES`）は共有層の layer_type に所有層が無いので、
#: decode の手術には使えない。
DECODE_LAYER_TYPES = (
    gx.SLIDING_ATTENTION,
    gx.SLIDING_ATTENTION,
    gx.FULL_ATTENTION,
    gx.SLIDING_ATTENTION,
    gx.FULL_ATTENTION,
)

#: 共有する層数（後ろ 2 層 = 層 3 / 層 4）。所有層は sliding → 層 1・full → 層 2。
KV_SHARED_LAYERS = 2

#: 所有層の数（= スロットは この 2 倍）。
OWNER_LAYERS = len(DECODE_LAYER_TYPES) - KV_SHARED_LAYERS

DEPTHS = {gx.SLIDING_ATTENTION: SLIDING_DEPTH, gx.FULL_ATTENTION: FULL_DEPTH}

#: tiny な RoPE 表の位置数（{@link decode.assert_rope_table_matches} の probe が 17 と
#: `positions - 1` を踏むので 18 以上が要る）。
TINY_POSITIONS = 32

TINY_IR_CONFIG = SimpleNamespace(
    num_attention_heads=HEADS,
    num_key_value_heads=KV_HEADS,
    head_dim=SLIDING_DEPTH,
    global_head_dim=FULL_DEPTH,
    num_hidden_layers=len(DECODE_LAYER_TYPES),
    num_kv_shared_layers=KV_SHARED_LAYERS,
    layer_types=list(DECODE_LAYER_TYPES),
    sliding_window=WINDOW,
)

#: 正例の格納内訳（i8 = embedding 系 / i4 = linear のダミー本数）。
STORAGE_COUNTS = {"i8": 4, "i4": 9}


# ---- RoPE 表引き -----------------------------------------------------------


class _StubRotary(nn.Module):
    """`Gemma4TextRotaryEmbedding` の呼び出し規約だけを持つ被験体。

    値は**位置と layer_type だけの決定的な関数**（表引きで厳密に再現できる形）。`x` は
    dtype / device のためだけに受ける引数で、上流実装と同じく値は読まない。
    """

    def __init__(self, depths: dict[str, int]) -> None:
        super().__init__()
        self.depths = depths
        self.offsets = {name: index + 1 for index, name in enumerate(sorted(depths))}

    def angles(self, position_ids: torch.Tensor, layer_type: str) -> torch.Tensor:
        depth = self.depths[layer_type]
        scale = torch.arange(depth, dtype=torch.float32) * 0.017 + 0.31 * self.offsets[layer_type]
        return position_ids.unsqueeze(-1).to(torch.float32) * scale

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor, layer_type: str | None = None):
        del x
        angles = self.angles(position_ids, layer_type)
        return angles.cos(), angles.sin()


class _LengthDependentRotary(_StubRotary):
    """位置**以外**（系列長）にも依存する rotary — 表引きでは原理的に再現できない被験体。"""

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor, layer_type: str | None = None):
        del x
        angles = self.angles(position_ids, layer_type) + float(position_ids.shape[1])
        return angles.cos(), angles.sin()


@pytest.fixture
def probe() -> torch.Tensor:
    """rotary が dtype / device を読むためのダミー入力。"""
    return torch.zeros(1, 1, HIDDEN, dtype=torch.float32)


class TestRopeTable:
    def test_the_lookup_reproduces_the_reference_for_every_layer_type(self, probe):
        rotary = _StubRotary(DEPTHS)

        table = decode.build_rope_table(rotary, probe, TINY_POSITIONS, tuple(DEPTHS))

        # 例外が出なければ全 probe × 全 layer_type で torch.equal（値・shape・dtype の一致）。
        decode.assert_rope_table_matches(rotary, table, probe, TINY_POSITIONS, tuple(DEPTHS))
        for layer_type, depth in DEPTHS.items():
            cos, sin = table.tables_for(layer_type)
            assert cos.shape == (TINY_POSITIONS, depth)
            assert sin.shape == (TINY_POSITIONS, depth)

    def test_an_unknown_layer_type_fails_loudly(self, probe):
        """MUST: 既定の表へ落とすと「full 層が sliding の周波数で回る」形が黙って通る。"""
        rotary = _StubRotary(DEPTHS)
        table = decode.build_rope_table(rotary, probe, TINY_POSITIONS, (gx.SLIDING_ATTENTION,))

        with pytest.raises(KeyError, match=gx.FULL_ATTENTION):
            table.tables_for(gx.FULL_ATTENTION)

    def test_a_reference_that_is_not_a_function_of_position_is_rejected(self, probe):
        """MUST: 表で表せない実装を黙って表に落とすと、値が「それらしく」ずれる。"""
        rotary = _LengthDependentRotary(DEPTHS)
        table = decode.build_rope_table(rotary, probe, TINY_POSITIONS, tuple(DEPTHS))

        with pytest.raises(AssertionError, match="表の引き方がずれている"):
            decode.assert_rope_table_matches(rotary, table, probe, TINY_POSITIONS, tuple(DEPTHS))

    def test_a_shifted_table_is_rejected(self, probe):
        """故障注入: 表を 1 行ずらすと突合が落ちる（= 上の緑が恒真でないことの裏取り）。"""
        rotary = _StubRotary(DEPTHS)
        table = decode.build_rope_table(rotary, probe, TINY_POSITIONS, tuple(DEPTHS))
        shifted = decode.RopeTable(
            {
                layer_type: (cos.roll(1, 0), sin)
                for layer_type, (cos, sin) in ((name, table.tables_for(name)) for name in DEPTHS)
            }
        )

        with pytest.raises(AssertionError, match="表の引き方がずれている"):
            decode.assert_rope_table_matches(rotary, shifted, probe, TINY_POSITIONS, tuple(DEPTHS))

    def test_a_table_built_for_the_wrong_layer_type_is_rejected(self, probe):
        """故障注入: 層種別の組を入れ替えると突合が落ちる（dispatch が効いていることの裏取り）。

        NOTE: 幅が違う（16 / 32）ので shape で落ちる — 実物の 256 / 512 も同じ関係にある。
        """
        rotary = _StubRotary(DEPTHS)
        table = decode.build_rope_table(rotary, probe, TINY_POSITIONS, tuple(DEPTHS))
        swapped = decode.RopeTable(
            {
                gx.SLIDING_ATTENTION: table.tables_for(gx.FULL_ATTENTION),
                gx.FULL_ATTENTION: table.tables_for(gx.SLIDING_ATTENTION),
            }
        )

        with pytest.raises(AssertionError, match="元実装は"):
            decode.assert_rope_table_matches(rotary, swapped, probe, TINY_POSITIONS, tuple(DEPTHS))

    def test_the_swap_leaves_the_model_untouched_when_the_check_fails(self, probe):
        """MUST: 突合の失敗で模型が半端な状態（表引きだが値が違う）に残らない。"""
        model = SimpleNamespace(
            model=SimpleNamespace(rotary_emb=_LengthDependentRotary(DEPTHS)),
            config=SimpleNamespace(hidden_size=HIDDEN, layer_types=list(DECODE_LAYER_TYPES)),
        )
        original = model.model.rotary_emb

        with pytest.raises(AssertionError):
            decode.swap_rope_table(model, TINY_POSITIONS)

        assert model.model.rotary_emb is original


# ---- KV 共有の割り付け -----------------------------------------------------


class TestKvOwnerLayers:
    def test_the_owner_is_the_last_layer_of_its_type_before_the_sharing_point(self):
        assert decode.first_shared_layer(TINY_IR_CONFIG) == OWNER_LAYERS
        assert decode.kv_owner_layers(TINY_IR_CONFIG) == {
            gx.SLIDING_ATTENTION: 1,
            gx.FULL_ATTENTION: 2,
        }

    def test_the_real_e2b_shape_maps_to_layers_13_and_14(self):
        """実物 E2B（35 層 / 共有 20 層 / 4 sliding + 1 full の 7 周期）の割り付け。"""
        layer_types = [
            gx.FULL_ATTENTION if (index + 1) % 5 == 0 else gx.SLIDING_ATTENTION
            for index in range(35)
        ]
        config = SimpleNamespace(
            num_hidden_layers=35, num_kv_shared_layers=20, layer_types=layer_types
        )

        assert decode.kv_owner_layers(config) == {
            gx.SLIDING_ATTENTION: 13,
            gx.FULL_ATTENTION: 14,
        }
        assert decode.slot_layers(config)[15:] == tuple(
            14 if (index + 1) % 5 == 0 else 13 for index in range(15, 35)
        )

    def test_a_shared_layer_type_with_no_owner_fails_loudly(self):
        """MUST: 上流なら `shared_kv_states` の KeyError になる形（層構成の前提が崩れている）。"""
        config = SimpleNamespace(
            num_hidden_layers=3,
            num_kv_shared_layers=1,
            layer_types=[gx.SLIDING_ATTENTION, gx.SLIDING_ATTENTION, gx.FULL_ATTENTION],
        )

        with pytest.raises(AssertionError, match="書き出す層が共有開始"):
            decode.kv_owner_layers(config)

    def test_the_layers_before_the_sharing_point_own_their_own_slots(self):
        assert decode.slot_layers(TINY_IR_CONFIG) == (0, 1, 2, 1, 2)


# ---- 合成 IR（手術の前後） -------------------------------------------------


def _pre_surgery_graph(
    *,
    kv_heads: int = KV_HEADS,
    layer_types: tuple[str, ...] = DECODE_LAYER_TYPES,
    shared: int = KV_SHARED_LAYERS,
    storage_counts=STORAGE_COUNTS,
) -> IrGraph:
    """従来形 attention を持つ decode chunk グラフの**骨組み**。

    検査対象は attention / `state_append` / states 節と、入力・出力・残骸 op・格納内訳だけ
    なので、q / k / v を作る側は「宣言 shape を持つ生きたノード」であればよい（op の意味は
    問わない）。

    MUST: 層は**直列に繋ぐ**（層 i の q は層 i-1 の出力から作る）。繋がない層は手術の刈り込み
    （`to_states_form` は出力から到達しないノードを落とす）で消え、被験体が黙って 1 層の
    グラフになる。
    MUST: 共有層の k / v は**所有層と同じ値名**を指す（上流の `shared_kv_states` の形）。
    別名にすると手術の source 一致検査が落ちる — それが実物での配線の実証になっている。
    """
    boundary = len(layer_types) - shared
    owners: dict[str, int] = {
        layer_type: layer for layer, layer_type in enumerate(layer_types[:boundary])
    }
    graph = IrGraph(symbols=[decode.SEQ_SYMBOL])
    graph.inputs.append(IrInput(name=decode.INPUT_IDS, dtype="i32", shape=[1, "M"]))
    graph.inputs.append(IrInput(name=decode.POSITION_IDS, dtype="i32", shape=[1, "M"]))
    for name, shape in (
        ("tok.table", [VOCAB, HIDDEN]),
        ("rope.cos", [TINY_POSITIONS, SLIDING_DEPTH]),
        ("mask.table", [1, 1, TINY_SYM_MAX, TINY_SYM_MAX]),
    ):
        graph.initializers[name] = IrInitializer(
            tensor=f"const.{name}", storage=IrStorage(dtype="f32")
        )
        graph.values[name] = IrValue(dtype="f32", shape=shape)

    graph.values["h"] = IrValue(dtype="f32", shape=[1, "M", HIDDEN])
    graph.values["cos"] = IrValue(dtype="f32", shape=[1, "M", SLIDING_DEPTH])
    graph.values["mask"] = IrValue(dtype="f32", shape=[1, 1, "M", "M"])
    graph.nodes.append(
        IrNode(op="embedding", ins=["tok.table", decode.INPUT_IDS], outs=["h"], attrs={})
    )
    graph.nodes.append(
        IrNode(op="embedding", ins=["rope.cos", decode.POSITION_IDS], outs=["cos"], attrs={})
    )
    graph.nodes.append(
        IrNode(
            op=gx.SYM_PREFIX_SLICE_OP,
            ins=["mask.table"],
            outs=["mask"],
            attrs={"sym": "M", "slices": [{"dim": 2, "coeff": 1, "offset": 0}]},
        )
    )
    for layer, layer_type in enumerate(layer_types):
        depth = DEPTHS[layer_type]
        source = "h" if layer == 0 else f"attn{layer - 1}"
        graph.values[f"q{layer}"] = IrValue(dtype="f32", shape=[1, HEADS, "M", depth])
        graph.nodes.append(IrNode(op="mul", ins=[source, "cos"], outs=[f"q{layer}"], attrs={}))
        owner = layer if layer < boundary else owners[layer_type]
        if owner == layer:
            for slot in ("k", "v"):
                name = f"{slot}{layer}"
                graph.values[name] = IrValue(dtype="f32", shape=[1, kv_heads, "M", depth])
                graph.nodes.append(IrNode(op="mul", ins=[source, "cos"], outs=[name], attrs={}))
        graph.values[f"attn{layer}"] = IrValue(dtype="f32", shape=[1, HEADS, "M", depth])
        graph.nodes.append(
            IrNode(
                op=gx.ATTENTION_OP,
                ins=[f"q{layer}", f"k{owner}", f"v{owner}", "mask"],
                outs=[f"attn{layer}"],
                attrs={"scale": 1.0},
            )
        )
    graph.values["logits"] = IrValue(dtype="f32", shape=[1, "M", VOCAB])
    graph.values["token"] = IrValue(dtype="i32", shape=[1, "M", 1])
    last = f"attn{len(layer_types) - 1}"
    graph.nodes.append(IrNode(op="mul", ins=[last, last], outs=["logits"], attrs={}))
    graph.nodes.append(IrNode(op="argmax", ins=["logits"], outs=["token"], attrs={}))
    graph.outputs.extend(["logits", "token"])
    for dtype, count in storage_counts.items():
        for slot in range(count):
            name = f"weight_{dtype}_{slot}"
            graph.initializers[name] = IrInitializer(
                tensor=f"w.{dtype}.{slot}", storage=IrStorage(dtype=dtype)
            )
            graph.values[name] = IrValue(dtype="f32", shape=[2, 2])
            graph.nodes.append(IrNode(op="mul", ins=[name, "logits"], outs=[], attrs={}))
    return graph


def _surgical_graph(**kwargs) -> IrGraph:
    """{@link _pre_surgery_graph} を全 attention について states 形へ手術したグラフ。"""
    graph = _pre_surgery_graph(**kwargs)
    return to_states_form(graph, decode.states_plan(graph, TINY_IR_CONFIG))


class TestStatesPlan:
    def test_the_slots_are_the_owner_layers_only(self):
        graph = _pre_surgery_graph()

        plan = decode.states_plan(graph, TINY_IR_CONFIG)

        assert plan.capacity_symbol == decode.CAPACITY_SYMBOL
        assert [(spec.k_slot, spec.v_slot) for spec in plan.attentions] == [
            ("l0.k", "l0.v"),
            ("l1.k", "l1.v"),
            ("l2.k", "l2.v"),
            # 共有層は所有層のスロットを読む（自分のスロットを作らない）
            ("l1.k", "l1.v"),
            ("l2.k", "l2.v"),
        ]
        # nodes 順の attention に 1 対 1（取り違えると別層の KV を読む形になる）
        assert [spec.output for spec in plan.attentions] == [f"attn{layer}" for layer in range(5)]

    def test_the_window_follows_the_layer_type(self):
        plan = decode.states_plan(_pre_surgery_graph(), TINY_IR_CONFIG)

        assert [spec.window for spec in plan.attentions] == [WINDOW, WINDOW, None, WINDOW, None]
        # 容量は層種別で分かれる: sliding = window 実数（ring は window ちょうどで閉じる）/
        # full = 記号（実行時に選ぶ — ADR 0066 決定 3）。states_plan docstring が正本。
        assert [spec.capacity for spec in plan.attentions] == [
            WINDOW,
            WINDOW,
            None,
            WINDOW,
            None,
        ]

    def test_a_layer_count_mismatch_fails_loudly(self):
        """MUST: 取りこぼした層は chunk 局所 causal のまま残る（沈黙誤値）。"""
        graph = _pre_surgery_graph(layer_types=DECODE_LAYER_TYPES[:3], shared=0)

        with pytest.raises(AssertionError, match="層と一致しない"):
            decode.states_plan(graph, TINY_IR_CONFIG)

    def test_a_node_order_that_is_not_layer_order_fails_loudly(self):
        """MUST: 出現順 = 層順の前提そのものを見る（崩れるとスロット割りが黙ってずれる）。"""
        shuffled = (
            gx.FULL_ATTENTION,
            gx.SLIDING_ATTENTION,
            gx.SLIDING_ATTENTION,
            gx.SLIDING_ATTENTION,
            gx.FULL_ATTENTION,
        )
        graph = _pre_surgery_graph(layer_types=shuffled)

        with pytest.raises(AssertionError, match="nodes 順が層順でない"):
            decode.states_plan(graph, TINY_IR_CONFIG)


class TestAssertIrFormDecode:
    def test_the_states_form_passes(self):
        form = decode.assert_ir_form_decode(_surgical_graph(), TINY_IR_CONFIG, STORAGE_COUNTS)

        assert form["attention_nodes"] == len(DECODE_LAYER_TYPES)
        assert form["state_append_nodes"] == 2 * OWNER_LAYERS
        assert form["slots"] == 2 * OWNER_LAYERS
        assert form["kv_owners"] == {gx.SLIDING_ATTENTION: 1, gx.FULL_ATTENTION: 2}
        assert form["heads"] == [HEADS, KV_HEADS, KV_HEADS]
        assert form["head_dim"] == {
            gx.SLIDING_ATTENTION: SLIDING_DEPTH,
            gx.FULL_ATTENTION: FULL_DEPTH,
        }
        assert form["storage"]["i4"] == STORAGE_COUNTS["i4"]

    def test_the_shared_layers_read_the_owner_slots(self):
        """共有層の states 欄が所有層のスロット名であること（形検査の主眼）。"""
        graph = _surgical_graph()

        attentions = [node for node in graph.nodes if node.op == gx.ATTENTION_OP]

        assert [node.states for node in attentions] == [
            {"k": "l0.k", "v": "l0.v"},
            {"k": "l1.k", "v": "l1.v"},
            {"k": "l2.k", "v": "l2.v"},
            {"k": "l1.k", "v": "l1.v"},
            {"k": "l2.k", "v": "l2.v"},
        ]

    def test_a_conventional_attention_left_behind_is_rejected(self):
        """MUST: 手術し損ねた層は過去を見ない（数値は動くので golden では捕まらない）。"""
        graph = _pre_surgery_graph()
        partial = StatesPlan(
            capacity_symbol=decode.CAPACITY_SYMBOL,
            attentions=decode.states_plan(graph, TINY_IR_CONFIG).attentions[:-1],
        )

        with pytest.raises(AssertionError, match="q / k / v の 3 本ちょうど"):
            decode.assert_ir_form_decode(
                to_states_form(graph, partial), TINY_IR_CONFIG, STORAGE_COUNTS
            )

    def test_a_shared_layer_pointed_at_the_wrong_owner_is_rejected(self):
        """MUST: 共有先の取り違えは形も型も合う（別層の過去を読む沈黙誤値）。"""
        graph = _pre_surgery_graph()
        specs = list(decode.states_plan(graph, TINY_IR_CONFIG).attentions)
        # 最後の full 共有層（層 4）を sliding の所有層 1 のスロットへ向け直す。
        specs[4] = replace(specs[4], k_slot="l1.k", v_slot="l1.v", window=WINDOW)
        misrouted = StatesPlan(capacity_symbol=decode.CAPACITY_SYMBOL, attentions=tuple(specs))

        # 手術自身が「スロットへ書く値が読者ごとに違う」で落とす（実物の配線の実証）。
        with pytest.raises(StatesFormError, match="読者ごとに違う"):
            to_states_form(graph, misrouted)

    def test_a_materialized_kv_shape_is_rejected(self):
        with pytest.raises(AssertionError, match="repeat_kv"):
            decode.assert_ir_form_decode(
                _surgical_graph(kv_heads=HEADS), TINY_IR_CONFIG, STORAGE_COUNTS
            )

    def test_a_head_dim_from_the_wrong_layer_type_is_rejected(self):
        """MUST: D は層種別で引く（sliding 層に global_head_dim が来たら落ちる）。"""
        derived = SimpleNamespace(**{**vars(TINY_IR_CONFIG), "head_dim": FULL_DEPTH})

        with pytest.raises(AssertionError, match="head_dim"):
            decode.assert_ir_form_decode(_surgical_graph(), derived, STORAGE_COUNTS)

    def test_a_window_on_a_full_layer_is_rejected(self):
        """全 context の full 層に window が付くと、窓外の過去を黙って切り捨てる。"""
        graph = _surgical_graph()
        index, node = next(
            (index, node)
            for index, node in enumerate(graph.nodes)
            if node.op == gx.ATTENTION_OP and not node.attrs.get("window")
        )
        graph.nodes[index] = replace(node, attrs={**node.attrs, "window": WINDOW})

        with pytest.raises(AssertionError, match="window"):
            decode.assert_ir_form_decode(graph, TINY_IR_CONFIG, STORAGE_COUNTS)

    def test_a_missing_window_on_a_sliding_layer_is_rejected(self):
        """逆向き: sliding 層の window が落ちると窓が効かない（全 context を読む）。"""
        graph = _surgical_graph()
        index, node = next(
            (index, node)
            for index, node in enumerate(graph.nodes)
            if node.op == gx.ATTENTION_OP and node.attrs.get("window")
        )
        graph.nodes[index] = replace(
            node, attrs={key: value for key, value in node.attrs.items() if key != "window"}
        )

        with pytest.raises(AssertionError, match="window"):
            decode.assert_ir_form_decode(graph, TINY_IR_CONFIG, STORAGE_COUNTS)

    def test_a_baked_full_capacity_is_rejected(self):
        """full の容量を数値で焼くと `createGenerationContext` で容量を選べない（ADR 0066 決定 3）。"""
        graph = _pre_surgery_graph()
        baked = StatesPlan(
            capacity_symbol=decode.CAPACITY_SYMBOL,
            attentions=tuple(
                replace(spec, capacity=8) if spec.window is None else spec
                for spec in decode.states_plan(graph, TINY_IR_CONFIG).attentions
            ),
        )

        with pytest.raises(AssertionError, match="full の容量は記号のまま残す"):
            decode.assert_ir_form_decode(
                to_states_form(graph, baked), TINY_IR_CONFIG, STORAGE_COUNTS
            )

    def test_a_symbolic_sliding_capacity_is_rejected(self):
        """sliding の容量が記号のままだと window 超の死蔵行が戻る（緩めると素通りする側）。"""
        graph = _pre_surgery_graph()
        symbolic = StatesPlan(
            capacity_symbol=decode.CAPACITY_SYMBOL,
            attentions=tuple(
                replace(spec, capacity=None)
                for spec in decode.states_plan(graph, TINY_IR_CONFIG).attentions
            ),
        )

        with pytest.raises(AssertionError, match="sliding は window 実数ちょうど"):
            decode.assert_ir_form_decode(
                to_states_form(graph, symbolic), TINY_IR_CONFIG, STORAGE_COUNTS
            )

    def test_a_missing_state_append_is_rejected(self):
        graph = _surgical_graph()
        graph.nodes = [node for node in graph.nodes if node.states.get("slot") != "l2.v"]

        with pytest.raises(AssertionError, match="1 本ずつでない"):
            decode.assert_ir_form_decode(graph, TINY_IR_CONFIG, STORAGE_COUNTS)

    def test_a_mask_left_as_a_graph_input_is_rejected(self):
        graph = _surgical_graph()
        graph.inputs.append(IrInput(name="attention_mask", dtype="f32", shape=[1, 1, "M", "M"]))

        with pytest.raises(AssertionError, match="グラフ入力"):
            decode.assert_ir_form_decode(graph, TINY_IR_CONFIG, STORAGE_COUNTS)

    @pytest.mark.parametrize("op", decode.RESIDUE_OPS)
    def test_a_leftover_residue_op_is_rejected(self, op):
        """MUST: 誰も読まない Tmax 定数 / 畳み残した RoPE が配布物に居座る形を落とす。"""
        graph = _surgical_graph()
        graph.nodes.append(IrNode(op=op, ins=["logits"], outs=["residue"], attrs={}))

        with pytest.raises(AssertionError, match="手術で死ぬはずの op"):
            decode.assert_ir_form_decode(graph, TINY_IR_CONFIG, STORAGE_COUNTS)

    def test_an_exit_that_is_not_argmax_is_rejected(self):
        """出力 1 は decode 出口（ADR 0068 決定 4）— logits を 2 本出す形と取り違えない。"""
        graph = _surgical_graph()
        graph.outputs[1] = "logits"

        with pytest.raises(AssertionError, match="出力 1 の供給元"):
            decode.assert_ir_form_decode(graph, TINY_IR_CONFIG, STORAGE_COUNTS)

    def test_a_single_output_graph_is_rejected(self):
        graph = _surgical_graph()
        graph.outputs.pop()

        with pytest.raises(AssertionError, match="decode 出口は logits / token の 2 本"):
            decode.assert_ir_form_decode(graph, TINY_IR_CONFIG, STORAGE_COUNTS)

    def test_a_storage_census_that_misses_a_compressed_weight_is_rejected(self):
        """MUST: 適格判定を外れた重みは**黙って f32 で残る**ので、本数で捕まえる。"""
        graph = _surgical_graph(storage_counts={"i8": STORAGE_COUNTS["i8"], "i4": 1})

        with pytest.raises(AssertionError, match="格納 dtype の本数"):
            decode.assert_ir_form_decode(graph, TINY_IR_CONFIG, STORAGE_COUNTS)

    def test_a_capacity_symbol_that_collides_with_an_input_symbol_fails_loudly(self):
        """記号の取り違え（容量を入力由来の記号に重ねる）は手術側が落とす。"""
        graph = _pre_surgery_graph()
        colliding = StatesPlan(
            capacity_symbol=decode.SEQ_SYMBOL,
            attentions=decode.states_plan(graph, TINY_IR_CONFIG).attentions,
        )

        with pytest.raises(StatesFormError, match=r"graph\.symbols に既にある"):
            to_states_form(graph, colliding)


# ---- greedy の余裕門 / 継続の採り方 ----------------------------------------


class TestGreedyMargins:
    def test_a_comfortable_margin_passes(self):
        shared.assert_greedy_margins({"a": [1.0, 0.5, 0.026], "b": [2.0]}, decode.MARGIN_FLOOR)

    def test_the_floor_stays_above_the_gate_precondition(self):
        """生産側の床 > 消費側の前提（2 × atol 1e-2 — Codex 波 H 指摘 H-04）。

        下だと「台本は採るが検収門の margin 前提で落ちる」ケースが作れてしまい、門の
        『ここが落ちるのは台本と資産が食い違ったときだけ』が成立しない。atol の正本は
        `e2e_gemma4_greedy_test.ts` の PREFILL_ATOL（= 1e-2）— 変えたら両方を動かす。
        """
        assert decode.MARGIN_FLOOR > 2 * 1e-2

    def test_a_thin_margin_fails_loudly(self):
        """MUST: 余裕の無い step を golden に混ぜると GPU 偏差で列が割れる。"""
        with pytest.raises(AssertionError, match=r"margin が下限"):
            shared.assert_greedy_margins({"a": [1.0, 1e-3, 0.5]}, decode.MARGIN_FLOOR)

    def test_the_floor_itself_is_not_accepted(self):
        """境界は「超える」— ちょうど下限は通さない。"""
        with pytest.raises(AssertionError, match=r"margin が下限"):
            shared.assert_greedy_margins({"a": [decode.MARGIN_FLOOR]}, decode.MARGIN_FLOOR)

    def test_every_offending_case_is_reported_at_once(self):
        """MUST: 最初の 1 件で止めない — 外すケースの判断材料が 1 回の実走で揃わなくなる。"""
        with pytest.raises(AssertionError) as failure:
            shared.assert_greedy_margins(
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

        tokens, margins = shared.greedy_continuation(wrapper, torch.tensor([[7, 8]]), 3)

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

        shared.greedy_continuation(wrapper, torch.tensor([[4, 5, 6]]), 2)

        assert [positions for _, positions in wrapper.calls] == [[0, 1, 2], [0, 1, 2, 3]]


class TestCaseRoom:
    def test_a_prompt_that_fits_with_its_continuation_passes(self):
        shared.assert_case_room([("case", torch.zeros(1, 10, dtype=torch.int64))], 6, 16)

    def test_a_continuation_past_the_table_fails_loudly(self):
        """MUST: 表の外の位置は引けない（`F.embedding` は OOB を黙って返さない）。"""
        with pytest.raises(AssertionError, match="RoPE 表の位置数"):
            shared.assert_case_room([("case", torch.zeros(1, 11, dtype=torch.int64))], 6, 16)

    def test_the_real_long_case_fits_the_table(self):
        """長ケース（T=598）+ K=16 が表に収まること — 定数を動かしたときの気づき線。"""
        assert 598 + decode.GREEDY_STEPS <= decode.ROPE_TABLE_POSITIONS


class TestGreedyCases:
    def test_every_greedy_case_is_a_golden_case(self):
        """MUST: golden に無いケース名を greedy 側だけで持たない（prompt の出どころが消える）。"""
        assert set(decode.GREEDY_CASES) <= {name for name, _ in gx.GOLDEN_CASES}

    def test_every_greedy_case_has_a_first_token_expectation(self):
        """第 1 継続 token の突合（1-shot 台本との機構横断検証）が全採用ケースで効く条件。"""
        assert set(decode.GREEDY_CASES) <= set(gx.GREEDY_EXPECTATIONS)

    def test_the_adopted_expectations_are_not_all_the_same_token(self):
        """ケースを外した結果、`_sanity` の定数出力検出線が恒真にならないことの条件。"""
        adopted = {gx.GREEDY_EXPECTATIONS[name] for name in decode.GREEDY_CASES}

        assert len(adopted) > 1


# ---- tiny な実モデルでの一周 -----------------------------------------------


def _tiny_decode_config():
    """KV 共有が成立する tiny な text config（{@link DECODE_LAYER_TYPES}）。"""
    transformers = pytest.importorskip("transformers")
    return transformers.Gemma4TextConfig(
        vocab_size=VOCAB,
        vocab_size_per_layer_input=VOCAB,
        hidden_size=HIDDEN,
        intermediate_size=32,
        num_hidden_layers=len(DECODE_LAYER_TYPES),
        num_attention_heads=HEADS,
        num_key_value_heads=KV_HEADS,
        head_dim=SLIDING_DEPTH,
        global_head_dim=FULL_DEPTH,
        hidden_size_per_layer_input=PLE_DIM,
        layer_types=list(DECODE_LAYER_TYPES),
        num_kv_shared_layers=KV_SHARED_LAYERS,
        sliding_window=WINDOW,
        use_double_wide_mlp=True,
        final_logit_softcapping=30.0,
        tie_word_embeddings=True,
        max_position_embeddings=256,
    )


@pytest.fixture
def tiny_wrapper():
    """tiny な実モデルで組んだ {@link decode.DecodeChunkWrapper}（実重みは読まない）。"""
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(0)
    gx.register_attention()
    config = _tiny_decode_config()
    config._attn_implementation = gx.ATTENTION_NAME
    model = transformers.Gemma4ForCausalLM(config).to(torch.float32).eval()
    tables = nn.ModuleList(
        [
            nn.Embedding.from_pretrained(torch.randn(VOCAB, PLE_DIM), freeze=True)
            for _ in DECODE_LAYER_TYPES
        ]
    )
    del model.model.embed_tokens_per_layer
    decode.swap_rope_table(model, TINY_POSITIONS)
    return decode.DecodeChunkWrapper(model, tables).eval()


class TestExportedDecodeForm:
    """tiny な実モデルを実際に export → 手術 → コンテナ検証まで通す（transformers が要る）。"""

    @pytest.fixture
    def tiny_container(self, tiny_wrapper, tmp_path):
        """`export_series` と**同じ順序・同じ格納指定**で 1 周させた `(手術前, 検証済み, …)`。

        MUST: 混成量子化（既定 i8 + linear i4 + RoPE 表の f32 明示）まで含めて踏む —
        格納の計画は手術**後**のグラフに掛かるので、f32 で書く一周では「刈られた重みの
        指定が残っている」「表引きの cos/sin が i8 の適格に入る」といった decode 固有の
        破れが 1 つも見えない（後者は実際にここで見つかった — 2026-08-19）。
        """
        int8, int4, scales = gx.quantize_wrapper(tiny_wrapper)
        ids = torch.randint(0, VOCAB, (1, WINDOW + 3), dtype=torch.int64)
        seq = Dim(decode.SEQ_SYMBOL, min=2, max=TINY_SYM_MAX)
        graph, tensors = export_module(
            tiny_wrapper,
            (ids, decode.positions_for(ids)),
            dynamic_shapes=({1: seq}, {1: seq}),
            symbol_names=(decode.SEQ_SYMBOL,),
            preserved=PRESERVED_OP_PREFIXES_WITH_ATTENTION,
        )
        config = tiny_wrapper.model.config
        surgical = to_states_form(graph, decode.states_plan(graph, config))
        verified = decode._write_container(
            surgical,
            tensors,
            tmp_path / gx.MODEL_FILE,
            weight_dtype="i8",
            weight_scales=scales,
            weight_dtype_overrides={
                **dict.fromkeys(int4.scales, "i4"),
                **dict.fromkeys(decode.rope_table_keys(tiny_wrapper), "f32"),
            },
        )
        storage = {"i8": len(int8.scales), "i4": len(int4.scales)}
        return SimpleNamespace(pre=graph, verified=verified, config=config, storage=storage)

    def test_the_container_is_a_verified_states_form_graph(self, tiny_container):
        form = decode.assert_ir_form_decode(
            tiny_container.verified, tiny_container.config, tiny_container.storage
        )

        assert form["attention_nodes"] == len(DECODE_LAYER_TYPES)
        assert form["state_append_nodes"] == 2 * OWNER_LAYERS
        assert form["kv_owners"] == {gx.SLIDING_ATTENTION: 1, gx.FULL_ATTENTION: 2}
        assert [spec.name for spec in tiny_container.verified.inputs] == [
            decode.INPUT_IDS,
            decode.POSITION_IDS,
        ]
        assert sorted(tiny_container.verified.symbols) == [
            decode.CAPACITY_SYMBOL,
            decode.SEQ_SYMBOL,
        ]
        # 手術は必ず何かを刈る（層種別 2 本の mask 定数）— 0 本なら畳み込みが効いていない
        assert len(tiny_container.pre.initializers) > len(tiny_container.verified.initializers)

    def test_the_mask_residue_is_gone_but_the_pre_surgery_graph_had_it(self, tiny_container):
        """恒真化の門: 刈る対象が export 側に**実在した**ことを同じ資産で見る。"""
        assert gx.SYM_PREFIX_SLICE_OP in tiny_container.pre.required_ops
        assert gx.SYM_PREFIX_SLICE_OP not in tiny_container.verified.required_ops

    def test_the_rope_is_a_table_lookup_not_folded_trig(self, tiny_container):
        """位置が入力になった以上、RoPE は畳み込みでなく `embedding` で引かれていること。"""
        readers = [
            node
            for node in tiny_container.verified.nodes
            if node.op == "embedding" and decode.POSITION_IDS in node.ins
        ]

        # 層種別 2 組 × cos / sin
        assert len(readers) == 4
        assert "sin" not in tiny_container.verified.required_ops

    def test_the_rope_tables_stay_f32(self, tiny_container, tiny_wrapper):
        """MUST: 表引きの cos / sin は既定 i8 の**適格に入る**ので f32 を明示して外す。

        外さないと「重みスロット適格なのに scale が無い」で書き出しごと落ちる（実測）。
        丸めて通す形も採らない — 位置表の誤差は位置に沿って効く。
        """
        stored = {
            init.tensor: init.storage.dtype
            for init in tiny_container.verified.initializers.values()
        }
        keys = decode.rope_table_keys(tiny_wrapper)

        assert len(keys) == 2 * len(DEPTHS)
        assert {stored[key] for key in keys} == {"f32"}

    def test_every_linear_weight_is_stored_as_int4(self, tiny_container):
        """MUST: 手術で刈られた重みがあれば i4 の本数が合わない（黙って f32 で残る形の検出線）。"""
        storage = {
            name: init.storage.dtype for name, init in tiny_container.verified.initializers.items()
        }

        assert sum(dtype == "i4" for dtype in storage.values()) == tiny_container.storage["i4"]
        assert sum(dtype == "i8" for dtype in storage.values()) == tiny_container.storage["i8"]

    def test_the_masks_come_from_the_one_shot_recipe(self, tiny_wrapper, monkeypatch):
        """窓の意味論を decode 側で書き直していないこと（正本は 1-shot 台本の 2 本）。

        MUST: 帯を組み直すと「1-shot の golden と decode の golden が別の窓で採られる」形が
        作れてしまう（どちらも自分の中では整合するので、突合しないと気づけない）。
        """
        seen: list[tuple[str, int, int | None]] = []

        def spy(name, original):
            def wrapped(length, *rest):
                seen.append((name, int(length), *(int(value) for value in rest)))
                return original(length, *rest)

            return wrapped

        monkeypatch.setattr(gx, "additive_causal_mask", spy("causal", gx.additive_causal_mask))
        monkeypatch.setattr(gx, "additive_sliding_mask", spy("band", gx.additive_sliding_mask))
        ids = torch.randint(0, VOCAB, (1, WINDOW + 3), dtype=torch.int64)

        with torch.no_grad():
            tiny_wrapper(ids, decode.positions_for(ids))

        assert seen == [("causal", WINDOW + 3), ("band", WINDOW + 3, WINDOW)]

    def test_the_wrapper_returns_logits_and_the_greedy_token(self, tiny_wrapper):
        """MUST: 出力順は `[logits, token]`（ランタイムは slot 番号で読む）。"""
        ids = torch.randint(0, VOCAB, (1, 5), dtype=torch.int64)

        with torch.no_grad():
            logits, token = tiny_wrapper(ids, decode.positions_for(ids))

        assert logits.shape == (1, 5, VOCAB)
        assert token.shape == (1, 5, 1)
        assert torch.equal(token, logits.argmax(-1, keepdim=True))

    def test_the_shared_layers_reuse_the_owner_kv_tensors(self, tiny_wrapper):
        """手術の source 一致検査が効く前提 — 上流が同じ k/v 値を共有層へ配線していること。

        MUST: 検査は export 済みグラフの**値名**で見る（eager の tensor 同一性ではない）—
        手術が突き合わせるのはこの値名だから。
        """
        ids = torch.randint(0, VOCAB, (1, WINDOW + 3), dtype=torch.int64)
        seq = Dim(decode.SEQ_SYMBOL, min=2, max=TINY_SYM_MAX)
        graph, _ = export_module(
            tiny_wrapper,
            (ids, decode.positions_for(ids)),
            dynamic_shapes=({1: seq}, {1: seq}),
            symbol_names=(decode.SEQ_SYMBOL,),
            preserved=PRESERVED_OP_PREFIXES_WITH_ATTENTION,
        )

        attentions = [node for node in graph.nodes if node.op == gx.ATTENTION_OP]
        sources = [tuple(node.ins[1:3]) for node in attentions]

        assert sources[3] == sources[1]
        assert sources[4] == sources[2]


class TestPublish:
    """staging → final の入れ替え（`_publish`）。

    守るのは「**新旧の混ざった正規資産を作れない**」こと。ファイル単位で final へ書く形だと、
    門の途中で落ちた実走が「新 model + 旧 greedy」を残し、検収門が拒否済み資産で緑になれる。
    """

    @staticmethod
    def _series(root, name: str, files: dict[str, str]):
        directory = root / name
        directory.mkdir()
        for filename, body in files.items():
            (directory / filename).write_text(body)
        return directory

    def test_a_fresh_target_is_created(self, tmp_path):
        staging = self._series(tmp_path, "staging", {"model": "new"})

        shared._publish(staging, tmp_path / "final")

        assert (tmp_path / "final" / "model").read_text() == "new"
        assert not staging.exists()

    def test_an_existing_target_is_replaced_wholesale(self, tmp_path):
        """旧にだけあるファイル（例: 除外されたケースの greedy）が final に残らない。"""
        final = self._series(tmp_path, "final", {"model": "old", "greedy.stale": "old"})
        staging = self._series(tmp_path, "staging", {"model": "new"})

        shared._publish(staging, final)

        assert (final / "model").read_text() == "new"
        assert not (final / "greedy.stale").exists()
        assert not list(tmp_path.glob("final" + SUPERSEDED_SUFFIX))

    def test_a_failed_promotion_restores_the_old_series(self, tmp_path, monkeypatch):
        """昇格の rename が失敗しても、final は**完全な旧資産のまま**（不在にも混成にもならない）。

        NOTE: `os.replace` を staging 昇格の 1 回だけ失敗させる（退避と復元は本物を通す）。
        """
        final = self._series(tmp_path, "final", {"model": "old"})
        staging = self._series(tmp_path, "staging", {"model": "new"})
        real_replace = os.replace

        def failing_replace(src, dst):
            if Path(src) == staging:
                raise OSError("昇格に失敗")
            return real_replace(src, dst)

        monkeypatch.setattr(os, "replace", failing_replace)

        with pytest.raises(OSError, match="昇格に失敗"):
            shared._publish(staging, final)

        assert (final / "model").read_text() == "old"
        assert staging.exists()


# ---- variant 駆動の中核 ----------------------------------------------------


class TestChunkVariants:
    """chunk 系列 2 本の記述子（{@link decode.ChunkVariant}）が宣言する 4 欄。"""

    def test_the_logits_series_declares_the_two_output_exit_with_goldens(self):
        assert decode.DECODE.wrapper is decode.DecodeChunkWrapper
        assert decode.DECODE.token_only is False
        assert decode.DECODE.goldens is True

    def test_the_token_only_series_declares_the_other_exit_without_goldens(self):
        """MUST: golden を採らないのは、検収門が logits 系列の greedy 記録を流用するから。"""
        assert token_only.VARIANT.wrapper is token_only.TokenOnlyChunkWrapper
        assert token_only.VARIANT.token_only is True
        assert token_only.VARIANT.goldens is False

    def test_the_two_series_land_in_different_directories(self):
        """置き場を共有すると、出口の違う 2 つの資産が互いを上書きする。"""
        assert decode.DECODE.out_dir != token_only.VARIANT.out_dir

    def test_both_wrappers_share_the_chunk_module_space(self):
        """MUST: 量子化の対象述語と scale 台帳を再利用できる条件（FQN 空間の同一性）。"""
        assert issubclass(token_only.VARIANT.wrapper, decode.DECODE.wrapper)


class TestExportArgs:
    @pytest.fixture
    def seq(self):
        return Dim(decode.SEQ_SYMBOL, min=2, max=TINY_SYM_MAX)

    def test_the_logits_form_traces_ids_and_positions(self, seq):
        ids = torch.zeros(1, 5, dtype=torch.int64)

        args, shapes = decode._export_args(decode.DECODE, ids, seq)

        assert len(args) == 2
        assert torch.equal(args[1], torch.arange(5).unsqueeze(0))
        assert shapes == ({1: seq}, {1: seq})

    def test_the_token_only_form_adds_the_last_row_as_a_static_input(self, seq):
        """MUST: `last_row` は最終有効行 T−1 を指し、記号次元を持たない（M と紐づけない）。"""
        ids = torch.zeros(1, 5, dtype=torch.int64)

        args, shapes = decode._export_args(token_only.VARIANT, ids, seq)

        assert len(args) == 3
        assert args[2].tolist() == [4]
        assert args[2].dtype is torch.int64
        assert shapes[2] is None


@pytest.fixture
def tiny_materials(monkeypatch):
    """`one_shot.load_model_and_tables` を tiny な模型へ差し替える（実重みを読まない席）。"""
    transformers = pytest.importorskip("transformers")

    def materials(model_dir):
        torch.manual_seed(0)
        gx.register_attention()
        config = _tiny_decode_config()
        config._attn_implementation = gx.ATTENTION_NAME
        model = transformers.Gemma4ForCausalLM(config).to(torch.float32).eval()
        tables = nn.ModuleList(
            [
                nn.Embedding.from_pretrained(torch.randn(VOCAB, PLE_DIM), freeze=True)
                for _ in DECODE_LAYER_TYPES
            ]
        )
        return model, tables

    monkeypatch.setattr(gx, "load_model_and_tables", materials)


class TestLoadWrapper:
    @pytest.mark.parametrize(
        "variant",
        [decode.DECODE, token_only.VARIANT],
        ids=["logits", "token-only"],
    )
    def test_it_builds_the_variant_wrapper_on_a_table_rope(self, variant, tiny_materials):
        """variant で変わるのはラッパ型だけ（素材の読み方と RoPE の差し替えは共通）。"""
        wrapper = decode.load_wrapper(variant, Path("unused"), positions=TINY_POSITIONS)

        assert type(wrapper) is variant.wrapper
        assert isinstance(wrapper.model.model.rotary_emb, decode.RopeTable)
        # 検査席の PLE 表は落ちている（量子化の対象網羅の条件 — 1-shot 台本と同文）。
        assert not hasattr(wrapper.model.model, "embed_tokens_per_layer")
        assert not wrapper.training


#: driver の一周に使うケース名。greedy 採用集合と 1-shot の期待表の**両方**にある名前でないと、
#: 系列の門（`GREEDY_CASES` の絞り込みと第 1 継続の突合）が実物と違う枝を通る。
TINY_CASE_NAMES = ("capital-en", "capital-ja")

#: tiny 系列が据えるコンテナのファイル名。配布形は**常時分割**（ADR 0081）なので、単一
#: ファイルは出ない — tiny 模型は数 KB なので「グラフ shard + weight shard 1 本」の 2 本になる。
#: 連番は焼かずに {@link karume.shards.shard_name} から引く（規則が動けばここも一緒に動く）。
TINY_CONTAINER_FILES = [shard_name(gx.MODEL_FILE, index, 2) for index in (1, 2)]

#: 系列ごとの要約の欄（順序込み）。ここが変わると実走の記録の形が変わる。
DECODE_SUMMARY_KEYS = [
    "dir",
    "nodes",
    "outputs",
    "initializers",
    "pruned_initializers",
    "model_bytes",
    "ops",
    "symbols",
    "io",
    "greedy",
    "greedy_steps",
    "case_lengths",
    "quantized",
    "form",
    "margin_min",
    "continuation",
    "sanity",
]
TOKEN_ONLY_SUMMARY_KEYS = [
    "dir",
    "nodes",
    "outputs",
    "initializers",
    "model_bytes",
    "ops",
    "symbols",
    "case_lengths",
    "quantized",
    "form",
    "reference",
    "sanity",
]


class _StubSeriesTokenizer:
    """期待表の引き・ラベル・継続の復号に要る 3 面だけを持つトークナイザの被験体。"""

    def encode(self, text: str, add_special_tokens: bool = True):
        """`tokenizers.Tokenizer.encode` の呼び出し規約だけ合わせる（単一トークンを返す）。"""
        return SimpleNamespace(ids=[len(text) % VOCAB])

    def id_to_token(self, token: int) -> str:
        return f"<{token}>"

    def decode(self, tokens) -> str:
        return " ".join(f"<{token}>" for token in tokens)


@pytest.fixture
def tiny_series(monkeypatch, tiny_materials, tmp_path_factory):
    """実資産を読まずに {@link decode.export_series} を 1 周させるための差し替え一式。

    差し替えるのは**素材の出どころ**（模型・ケース・トークナイザ）と、tiny な乱数重みでは
    立てられない 2 つの期待だけ — 継続の余裕（床は `TestGreedyMargins` が固定）と最終位置の
    1 位（`test_export.TestSanity` が固定）。量子化 → export → 手術 → 書き出し → 形検査 →
    公開の経路は本物を通す。

    NOTE: 素材の席（指紋を採るチェックポイントと流用する golden 系列）は `tmp_path` の**外**
    に作る — 中に作ると「作業席も退避席も残らない」の検査が素材まで数えてしまう。
    """
    torch.manual_seed(1)
    cases = tuple(
        (name, torch.randint(0, VOCAB, (1, WINDOW + 3 - index), dtype=torch.int64))
        for index, name in enumerate(TINY_CASE_NAMES)
    )
    checkpoint = tmp_path_factory.mktemp("checkpoint")
    for name in provenance.FINGERPRINT_FILES:
        (checkpoint / name).write_bytes(name.encode())
    reference = tmp_path_factory.mktemp("reference-series")
    for name, ids in cases:
        save_file(
            {shared.PROMPT_KEY: ids[0].to(torch.int32).contiguous()},
            str(reference / f"{shared.GREEDY_PREFIX}{name}{shared.GREEDY_SUFFIX}"),
        )
    monkeypatch.setattr(gx, "build_cases", lambda model_dir, sym_max, window: cases)
    monkeypatch.setattr(gx, "load_tokenizer", lambda model_dir: _StubSeriesTokenizer())
    monkeypatch.setattr(decode, "MARGIN_FLOOR", 0.0)
    seen: dict[str, dict[str, int]] = {}

    def record(greedy, expected, labels):
        seen["greedy"] = dict(greedy)
        return {"stub": "ok"}

    monkeypatch.setattr(gx, "_sanity", record)
    return SimpleNamespace(cases=cases, sanity=seen, checkpoint=checkpoint, reference=reference)


class TestExportSeries:
    """variant 駆動の一周（tiny 模型）。系列で変わるのは出口・golden・据える単位・要約の欄。"""

    def test_the_logits_series_publishes_the_container_with_its_goldens(
        self, tiny_series, tmp_path
    ):
        out_dir = tmp_path / "series"

        summary = decode.export_series(
            decode.DECODE,
            tmp_path / "unused",
            out_dir,
            sym_max=TINY_SYM_MAX,
            positions=TINY_POSITIONS,
            steps=1,
        )

        assert sorted(path.name for path in out_dir.iterdir()) == sorted(
            [
                "greedy.capital-en.safetensors",
                "greedy.capital-ja.safetensors",
                "io.capital-en.safetensors",
                "io.capital-ja.safetensors",
                *TINY_CONTAINER_FILES,
            ]
        )
        assert summary["outputs"] == 2
        assert list(summary) == DECODE_SUMMARY_KEYS
        assert set(tiny_series.sanity["greedy"]) == set(TINY_CASE_NAMES)
        # 作業席も退避席も残らない（据え替えの後片付けは core の原語の担当）。
        assert list(tmp_path.iterdir()) == [out_dir]

    def test_the_token_only_series_publishes_the_container_with_its_provenance(
        self, tiny_series, tmp_path
    ):
        """golden を採らない系列は「コンテナ + 出所記録」を据える（io / greedy を書かない）。"""
        out_dir = tmp_path / "series"

        summary = decode.export_series(
            token_only.VARIANT,
            tiny_series.checkpoint,
            out_dir,
            sym_max=TINY_SYM_MAX,
            positions=TINY_POSITIONS,
            reference=tiny_series.reference,
        )

        assert sorted(path.name for path in out_dir.iterdir()) == sorted(
            [*TINY_CONTAINER_FILES, "reference.json"]
        )
        assert summary["outputs"] == 1
        assert list(summary) == TOKEN_ONLY_SUMMARY_KEYS
        # sanity は全ケースぶん（greedy 記録が無いので全長 forward で採る）。
        assert set(tiny_series.sanity["greedy"]) == set(TINY_CASE_NAMES)
        assert list(tmp_path.iterdir()) == [out_dir]

    def test_the_provenance_record_binds_the_reference_goldens(self, tiny_series, tmp_path):
        """記録が「元 checkpoint の指紋」と「流用する golden の digest」を束ねている。"""
        out_dir = tmp_path / "series"

        decode.export_series(
            token_only.VARIANT,
            tiny_series.checkpoint,
            out_dir,
            sym_max=TINY_SYM_MAX,
            positions=TINY_POSITIONS,
            reference=tiny_series.reference,
        )

        record = json.loads((out_dir / provenance.REFERENCE_FILE).read_text(encoding="utf-8"))
        assert record["schema"] == provenance.SCHEMA
        assert record["series"] == out_dir.name
        assert record["checkpoint"]["dir"] == tiny_series.checkpoint.name
        assert sorted(record["checkpoint"]["files"]) == sorted(provenance.FINGERPRINT_FILES)
        assert record["reference"]["series"] == tiny_series.reference.name
        goldens = record["reference"]["goldens"]
        assert sorted(goldens) == sorted(
            f"{shared.GREEDY_PREFIX}{name}{shared.GREEDY_SUFFIX}" for name in TINY_CASE_NAMES
        )
        for file, digest in goldens.items():
            raw = (tiny_series.reference / file).read_bytes()
            assert digest == {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}

    def test_a_missing_reference_golden_stops_the_token_only_series(self, tiny_series, tmp_path):
        """MUST: 流用先が欠けた組み合わせは席を作る前に落ちる（資産を 1 本も残さない）。"""
        out_dir = tmp_path / "series"
        for path in tiny_series.reference.iterdir():
            path.unlink()

        with pytest.raises(AssertionError, match="参照 golden"):
            decode.export_series(
                token_only.VARIANT,
                tiny_series.checkpoint,
                out_dir,
                sym_max=TINY_SYM_MAX,
                positions=TINY_POSITIONS,
                reference=tiny_series.reference,
            )

        assert list(tmp_path.iterdir()) == []

    def test_a_failing_gate_leaves_nothing_behind(self, tiny_series, tmp_path, monkeypatch):
        """MUST: 門より前に final へ置かない（落ちた実走が検収を通れる資産を残さない）。"""

        def refuse(greedy, expected, labels):
            raise AssertionError("最終位置の 1 位が期待継続と違う")

        monkeypatch.setattr(gx, "_sanity", refuse)
        out_dir = tmp_path / "series"

        with pytest.raises(AssertionError, match="期待継続と違う"):
            decode.export_series(
                decode.DECODE,
                tmp_path / "unused",
                out_dir,
                sym_max=TINY_SYM_MAX,
                positions=TINY_POSITIONS,
                steps=1,
            )

        assert list(tmp_path.iterdir()) == []
