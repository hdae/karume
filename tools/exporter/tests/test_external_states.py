"""借り手グラフの 3 宣言（external スロット / readonly attention / 共有 initializer）を固定する。

正本は ADR [0096](../../../docs/decisions/0096-speculative-decoding.md) 段 2 の契約で、TS 側の
鏡像は `packages/runtime/src/format/ir.ts` + `packages/runtime/src/runtime/plan.ts`。ここが見るのは
Python 側の 3 層すべて — 宣言の往復（`to_json` → `parse_ir_graph`）・契約と shape の門・
手術（`karume.states.to_external_states_form`）・書き出し（共有宣言を格納から外す）。

被験グラフは合成の小グラフ（torch 不要）。実 drafter は 4 層 × k 段の展開形だが、3 宣言の規律は
attention 1〜2 本で全部踏める。

故障注入（この 4 本が拒否されることが「検査が実効である」ことの担保 — 通ると**値だけ**が
静かに変わる）:

1. external スロットへの `state_append`（借り手が貸し手の ring を汚す）
2. readonly でない読者が external を読む（今 step の k/v を足して読む形）
3. 共有 initializer に `tensor` / `scale`（バイトの出どころが 2 箇所になる）
4. external と自前スロットの混在（確保規則がグラフから決まらない）
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
import torch

from karume.emit import EmitError, write_model
from karume.ir import (
    IrGraph,
    IrInitializer,
    IrInput,
    IrNode,
    IrShared,
    IrState,
    IrStorage,
    IrValue,
)
from karume.ops import OpContractError
from karume.quantize import channel_scale, quantize_to_int8
from karume.states import (
    ExternalAttentionSpec,
    ExternalStatesPlan,
    StatesFormError,
    to_external_states_form,
)
from karume.verify import (
    IrError,
    assert_op_contracts,
    assert_runtime_support,
    parse_ir_graph,
    verify_shards,
)

#: 借り手の q（`[B,H,1,D]` — H=4・M は 1 ちょうど MUST）と、trace 用に置く K / V 入力の形。
#: T は「固定の小さい値でよい」placeholder（手術で落ちるので配布形には残らない）。
Q_SLIDING = [1, 4, 1, 256]
KV_SLIDING = [1, 1, 8, 256]
Q_FULL = [1, 4, 1, 512]
KV_FULL = [1, 1, 8, 512]

#: 貸し手スロットの容量（sliding = window 512 + 余裕 8 の実数 / full = 記号）。
SLIDING_CAPACITY = 520
CAPACITY_SYMBOL = "C"
WINDOW = 512


def _input(name: str, shape: Sequence) -> dict:
    return {"name": name, "dtype": "f32", "shape": list(shape)}


def _value(shape: Sequence) -> dict:
    return {"dtype": "f32", "shape": list(shape)}


def source_dict() -> dict:
    """借り手の trace 直後（手術前）の形 — 従来形 attention 2 本（sliding / full）。

    mask は**焼いた定数**（q_len 1 の bidirectional = 全列許可の 0 行）。手術がノードの入力から
    外すと誰も読まなくなるので、刈り込みでそのまま死ぬ。
    """
    return {
        "format": "karume-ir",
        "version": 1,
        "requires": {"ops": ["add", "attention"]},
        "symbols": [],
        "inputs": [
            _input("qs", Q_SLIDING),
            _input("qf", Q_FULL),
            _input("sliding_k", KV_SLIDING),
            _input("sliding_v", KV_SLIDING),
            _input("full_k", KV_FULL),
            _input("full_v", KV_FULL),
        ],
        "outputs": ["o", "af"],
        "initializers": {
            "mask_s": {"tensor": "drafter.mask_s", "storage": {"dtype": "f32"}},
            "mask_f": {"tensor": "drafter.mask_f", "storage": {"dtype": "f32"}},
            "bias": {"tensor": "drafter.bias", "storage": {"dtype": "f32"}},
        },
        "values": {
            "mask_s": _value([1, 1, 1, 8]),
            "mask_f": _value([1, 1, 1, 8]),
            "bias": _value([256]),
            "as": _value(Q_SLIDING),
            "af": _value(Q_FULL),
            "o": _value(Q_SLIDING),
        },
        "nodes": [
            {
                "op": "attention",
                "ins": ["qs", "sliding_k", "sliding_v", "mask_s"],
                "outs": ["as"],
                "attrs": {"scale": 1.0},
            },
            {
                "op": "attention",
                "ins": ["qf", "full_k", "full_v", "mask_f"],
                "outs": ["af"],
                "attrs": {"scale": 1.0},
            },
            {"op": "add", "ins": ["as", "bias"], "outs": ["o"], "attrs": {}},
        ],
    }


def source() -> IrGraph:
    """被験グラフ。**パーサを通して作る** — 手術の入力が valid な IR であることが前提。"""
    return parse_ir_graph(json.dumps(source_dict()))


SLIDING_SPEC = ExternalAttentionSpec(
    output="as",
    k_slot="l13.k",
    v_slot="l13.v",
    k_input="sliding_k",
    v_input="sliding_v",
    kv_heads=1,
    head_dim=256,
    capacity=SLIDING_CAPACITY,
    window=WINDOW,
)
FULL_SPEC = ExternalAttentionSpec(
    output="af",
    k_slot="l14.k",
    v_slot="l14.v",
    k_input="full_k",
    v_input="full_v",
    kv_heads=1,
    head_dim=512,
    capacity=CAPACITY_SYMBOL,
)


def plan(*attentions: ExternalAttentionSpec) -> ExternalStatesPlan:
    return ExternalStatesPlan(capacity_symbol=CAPACITY_SYMBOL, attentions=attentions)


def checked(graph: IrGraph) -> IrGraph:
    """手術結果を JSON へ落として**パーサと契約検査へ掛け直す**（往復が受入条件）。"""
    parsed = parse_ir_graph(graph.to_json())
    assert_op_contracts(parsed)
    assert_runtime_support(parsed)
    return parsed


class TestTheExternalSurgery:
    def test_the_reader_becomes_a_readonly_node_with_only_q(self):
        """readonly 形は q の 1 本ちょうど（今 step の k/v も mask も持たない）。"""
        graph = checked(to_external_states_form(source(), plan(SLIDING_SPEC, FULL_SPEC)))

        sliding, full = (node for node in graph.nodes if node.op == "attention")
        assert sliding.ins == ["qs"]
        assert sliding.attrs == {"scale": 1.0, "readonly": True, "window": WINDOW}
        assert sliding.states == {"k": "l13.k", "v": "l13.v"}
        # full 層は `window` を宣言しない（欄の不存在がそのまま「全 context」— ADR 0067 決定 4）。
        assert full.attrs == {"scale": 1.0, "readonly": True}

    def test_no_append_is_inserted(self):
        """external の実体は貸し手のもの — 借り手は 1 本も書かない。"""
        graph = to_external_states_form(source(), plan(SLIDING_SPEC, FULL_SPEC))

        assert [node.op for node in graph.nodes] == ["attention", "attention", "add"]

    def test_the_slots_are_declared_external_from_the_plan(self):
        """スロットの形は plan の `[1, Hkv, capacity, D]`（placeholder の T は残らない）。"""
        graph = to_external_states_form(source(), plan(SLIDING_SPEC, FULL_SPEC))

        assert graph.states["l13.k"] == IrState(
            dtype="f32", shape=[1, 1, SLIDING_CAPACITY, 256], external=True
        )
        assert graph.states["l14.v"] == IrState(
            dtype="f32", shape=[1, 1, CAPACITY_SYMBOL, 512], external=True
        )

    def test_the_traced_kv_inputs_are_dropped(self):
        """K / V の placeholder は宣言ごと消える（呼び手はもう渡さない）。"""
        graph = to_external_states_form(source(), plan(SLIDING_SPEC, FULL_SPEC))

        assert [spec.name for spec in graph.inputs] == ["qs", "qf"]

    def test_the_baked_masks_are_pruned(self):
        """誰も読まなくなった mask 定数は刈られ、生きた initializer は残る。"""
        graph = to_external_states_form(source(), plan(SLIDING_SPEC, FULL_SPEC))

        assert sorted(graph.initializers) == ["bias"]
        assert "mask_s" not in graph.values

    def test_the_capacity_symbol_is_declared_only_when_used(self):
        """記号容量を使うスロットがあるときだけ `symbols` へ足す。"""
        with_symbol = to_external_states_form(source(), plan(SLIDING_SPEC, FULL_SPEC))
        assert with_symbol.symbols == [CAPACITY_SYMBOL]

    def test_the_input_declaration_order_is_preserved(self):
        """残る入力の並びは元のまま（ホストは順序で束ねる）。"""
        graph = to_external_states_form(source(), plan(FULL_SPEC, SLIDING_SPEC))

        assert [spec.name for spec in graph.inputs] == ["qs", "qf"]


class TestTheSurgeryRejectsMiswiring:
    def test_dropping_a_value_that_is_not_the_node_kv_is_rejected(self):
        """落とす入力がそのノードの k / v でなければ配線ミス（別の値が消える）。"""
        spec = ExternalAttentionSpec(
            output="as",
            k_slot="l13.k",
            v_slot="l13.v",
            k_input="qf",
            v_input="sliding_v",
            kv_heads=1,
            head_dim=256,
            capacity=SLIDING_CAPACITY,
            window=WINDOW,
        )

        with pytest.raises(StatesFormError, match=r"k に指定した 'qf'"):
            to_external_states_form(source(), plan(spec))

    def test_swapping_k_and_v_is_rejected(self):
        """K に元 V・V に元 K を指定した取り違え — 同形なので通ると値だけが静かに違う。"""
        spec = ExternalAttentionSpec(
            output="as",
            k_slot="l13.k",
            v_slot="l13.v",
            k_input="sliding_v",
            v_input="sliding_k",
            kv_heads=1,
            head_dim=256,
            capacity=SLIDING_CAPACITY,
            window=WINDOW,
        )

        with pytest.raises(StatesFormError, match=r"k に指定した 'sliding_v'"):
            to_external_states_form(source(), plan(spec))

    def test_naming_the_same_input_as_both_k_and_v_is_rejected(self):
        """K と V に同じ入力を指定した形（V スロットへ K が入る）も添字ごとの照合で落ちる。"""
        spec = ExternalAttentionSpec(
            output="as",
            k_slot="l13.k",
            v_slot="l13.v",
            k_input="sliding_k",
            v_input="sliding_k",
            kv_heads=1,
            head_dim=256,
            capacity=SLIDING_CAPACITY,
            window=WINDOW,
        )

        with pytest.raises(StatesFormError, match=r"v に指定した 'sliding_k'"):
            to_external_states_form(source(), plan(spec))

    def test_an_already_operated_node_is_rejected(self):
        """二重手術（すでに states 形）は従来の検出線がそのまま受ける。"""
        once = to_external_states_form(source(), plan(SLIDING_SPEC, FULL_SPEC))

        with pytest.raises(StatesFormError, match="すでに states 形"):
            to_external_states_form(once, plan(SLIDING_SPEC))


class TestTheDeclarationsRoundTrip:
    def test_external_false_is_not_written(self):
        """自前スロットのグラフ JSON は 1 バイトも動かない（既存資産の sha 門が動かない）。"""
        slot = IrState(dtype="f32", shape=[1, 1, 8, 4])

        assert slot.to_dict() == {"dtype": "f32", "shape": [1, 1, 8, 4]}

    def test_a_shared_initializer_writes_no_tensor_key(self):
        """共有宣言は `shared.tensor` だけ（自前のテンソルキーを持たない）。"""
        initializer = IrInitializer(
            shared=IrShared(tensor="model.lm_head.weight"), storage=IrStorage(dtype="i8")
        )

        assert initializer.to_dict() == {
            "shared": {"tensor": "model.lm_head.weight"},
            "storage": {"dtype": "i8"},
        }
        assert initializer.is_shared

    def test_the_operated_graph_survives_a_json_round_trip(self):
        """往復でバイトが動かない（宣言 3 種とも読み書きの受理集合が同じ）。"""
        graph = to_external_states_form(source(), plan(SLIDING_SPEC, FULL_SPEC))
        text = graph.to_json()

        assert parse_ir_graph(text).to_json() == text


def _operated_json(**mutate) -> dict:
    """手術済みグラフの JSON に共有 initializer を 1 本足したもの（故障注入の土台）。"""
    graph = to_external_states_form(source(), plan(SLIDING_SPEC, FULL_SPEC)).to_dict()
    graph["initializers"]["target_embed"] = {
        "shared": {"tensor": "model.lm_head.weight"},
        "storage": {"dtype": "i8"},
    }
    graph["values"]["target_embed"] = _value([16, 8])
    graph["nodes"] = [
        {
            "op": "embedding",
            "ins": ["target_embed", "tok"],
            "outs": ["e"],
            "attrs": {"padding_idx": -1},
        },
        *graph["nodes"],
    ]
    graph["inputs"] = [{"name": "tok", "dtype": "i32", "shape": [1, 1]}, *graph["inputs"]]
    graph["values"]["e"] = _value([1, 1, 8])
    graph["outputs"] = [*graph["outputs"], "e"]
    graph["requires"] = {"ops": sorted({node["op"] for node in graph["nodes"]})}
    graph.update(mutate)
    return graph


def _parse_and_check(graph: dict) -> IrGraph:
    parsed = parse_ir_graph(json.dumps(graph))
    assert_op_contracts(parsed)
    assert_runtime_support(parsed)
    return parsed


class TestTheGraphChecksAreEffective:
    """故障注入 — 4 本とも通ると「形も型も合ったまま値だけが違う」IR になる。"""

    def test_the_baseline_is_accepted(self):
        """土台そのものは通る（下の 4 本が「元から落ちていた」ではないことの対照）。"""
        assert _parse_and_check(_operated_json()).states["l13.k"].external is True

    def test_an_append_into_an_external_slot_is_rejected(self):
        graph = _operated_json()
        graph["nodes"] = [
            *graph["nodes"],
            {
                "op": "state_append",
                "ins": ["qs"],
                "outs": [],
                "attrs": {"window": WINDOW},
                "states": {"slot": "l13.k"},
            },
        ]
        graph["requires"] = {"ops": sorted({node["op"] for node in graph["nodes"]})}

        with pytest.raises(IrError, match="readonly でない形で触れている"):
            _parse_and_check(graph)

    def test_a_non_readonly_reader_of_an_external_slot_is_rejected(self):
        graph = _operated_json()
        for node in graph["nodes"]:
            if node.get("states", {}).get("k") == "l13.k":
                node["ins"] = ["qs", "sliding_k", "sliding_v"]
                node["attrs"] = {"scale": 1.0, "window": WINDOW}
        graph["inputs"] = [
            *graph["inputs"],
            _input("sliding_k", KV_SLIDING),
            _input("sliding_v", KV_SLIDING),
        ]

        with pytest.raises(IrError, match="readonly でない形で触れている"):
            _parse_and_check(graph)

    def test_a_readonly_reader_of_an_owned_slot_is_rejected(self):
        """逆向き — readonly が自前スロットを読む形（貸し手のいない借り手）。"""
        graph = _operated_json()
        for slot in graph["states"].values():
            slot.pop("external")

        with pytest.raises(IrError, match="自前スロット"):
            _parse_and_check(graph)

    def test_mixing_external_and_owned_slots_is_rejected(self):
        graph = _operated_json()
        graph["states"]["l13.k"].pop("external")

        with pytest.raises(IrError, match="混在している"):
            _parse_and_check(graph)

    def test_a_shared_initializer_with_a_tensor_key_is_rejected(self):
        graph = _operated_json()
        graph["initializers"]["target_embed"]["tensor"] = "drafter.embed"

        with pytest.raises(IrError, match="未知のキー 'tensor'"):
            _parse_and_check(graph)

    @pytest.mark.parametrize("key", ["scale", "group_size"])
    def test_a_shared_initializer_with_a_companion_descriptor_is_rejected(self, key):
        """付随実体（scale）と group の刻みは貸し手だけが持つ（写すと出どころが 2 箇所）。"""
        graph = _operated_json()
        graph["initializers"]["target_embed"]["storage"][key] = (
            32 if key == "group_size" else "drafter.embed.scale"
        )

        with pytest.raises(IrError, match="共有 initializer は"):
            _parse_and_check(graph)

    def test_a_readonly_node_with_three_inputs_is_rejected(self):
        graph = _operated_json()
        for node in graph["nodes"]:
            if node.get("attrs", {}).get("readonly"):
                node["ins"] = ["qs", "qs", "qs"]

        with pytest.raises(OpContractError, match="readonly 形は入力 1 本ちょうど"):
            _parse_and_check(graph)

    def test_a_readonly_node_with_two_query_rows_is_rejected(self):
        """M = 1 MUST（段 2 は「論理位置 P−1 の 1 行」だけを意味づける）。"""
        graph = _operated_json()
        graph["inputs"] = [
            _input("qs", [1, 4, 2, 256]) if spec["name"] == "qs" else spec
            for spec in graph["inputs"]
        ]
        graph["values"]["as"] = _value([1, 4, 2, 256])
        graph["values"]["o"] = _value([1, 4, 2, 256])

        with pytest.raises(OpContractError, match="M（軸 2）が 1 ちょうど"):
            _parse_and_check(graph)

    def test_a_slot_whose_head_dim_disagrees_with_q_is_rejected(self):
        """スロット取り違え（別種別のスロットを読む形）は D 軸で落ちる。"""
        graph = _operated_json()
        graph["states"]["l13.k"]["shape"] = [1, 1, SLIDING_CAPACITY, 512]
        graph["states"]["l13.v"]["shape"] = [1, 1, SLIDING_CAPACITY, 512]

        with pytest.raises(OpContractError, match="スロットと q の B / D が不一致"):
            _parse_and_check(graph)

    def test_a_window_wider_than_the_slot_capacity_is_rejected(self):
        graph = _operated_json()
        graph["states"]["l13.k"]["shape"] = [1, 1, 8, 256]
        graph["states"]["l13.v"]["shape"] = [1, 1, 8, 256]

        with pytest.raises(OpContractError, match="スロット容量 8 を超える"):
            _parse_and_check(graph)


def _shared_container_graph() -> tuple[IrGraph, dict[str, torch.Tensor], dict]:
    """共有 initializer 1 本 + 自前の重み 1 本を持つ最小コンテナ。"""
    weight = (torch.arange(16, dtype=torch.float32) % 7 - 3).reshape(4, 4)
    scale = channel_scale(weight, 0)
    rounded = quantize_to_int8(weight, scale).to(torch.float32) * scale
    graph = IrGraph(
        inputs=[
            IrInput(name="x", dtype="f32", shape=[1, 4]),
            IrInput(name="tok", dtype="i32", shape=[1, 1]),
        ],
        outputs=["h", "e"],
        initializers={
            "target_embed": IrInitializer(
                shared=IrShared(tensor="model.lm_head.weight"), storage=IrStorage(dtype="i8")
            ),
            "w": IrInitializer(tensor="own.w", storage=IrStorage(dtype="f32")),
            "b": IrInitializer(tensor="own.b", storage=IrStorage(dtype="f32")),
        },
        values={
            "target_embed": IrValue(dtype="f32", shape=[16, 8]),
            "w": IrValue(dtype="f32", shape=[4, 4]),
            "b": IrValue(dtype="f32", shape=[4]),
            "h": IrValue(dtype="f32", shape=[1, 4]),
            "e": IrValue(dtype="f32", shape=[1, 1, 8]),
        },
        nodes=[
            IrNode(op="linear", ins=["x", "w", "b"], outs=["h"], attrs={}),
            IrNode(
                op="embedding",
                ins=["target_embed", "tok"],
                outs=["e"],
                attrs={"padding_idx": -1},
            ),
        ],
    )
    tensors = {"own.w": rounded, "own.b": torch.zeros(4)}
    return graph, tensors, {"own.w": scale}


class TestTheContainerExcludesSharedDeclarations:
    def test_a_shared_declaration_needs_no_stored_tensor(self):
        """宣言 / 格納の完全一致から外れる（借り手の shard にバイトは 1 つも無い）。"""
        graph, tensors, scales = _shared_container_graph()

        with TemporaryDirectory() as staging:
            written = write_model(
                Path(staging) / "model.safetensors",
                graph,
                tensors,
                weight_dtype="i8",
                weight_scales=scales,
            )
            verified = verify_shards(written)

        # 宣言された storage はそのまま焼かれる（計画も変換も掛からない）。
        assert verified.initializers["target_embed"].shared == IrShared(
            tensor="model.lm_head.weight"
        )
        assert verified.initializers["target_embed"].storage == IrStorage(dtype="i8")
        # 自前の重みは従来どおり i8 へ計画される（除外が「全部素通し」ではない対照）。
        assert verified.initializers["w"].storage.dtype == "i8"

    def test_a_shared_tensor_key_present_in_the_container_is_surplus(self):
        """借り手が同名のバイトを持っていたら余剰（貸し手と二重に持つ形を拒否する）。"""
        graph, tensors, scales = _shared_container_graph()
        tensors = {**tensors, "model.lm_head.weight": torch.zeros(16, 8)}

        with TemporaryDirectory() as staging, pytest.raises(EmitError, match="余剰"):
            write_model(
                Path(staging) / "model.safetensors",
                graph,
                tensors,
                weight_dtype="i8",
                weight_scales=scales,
            )
