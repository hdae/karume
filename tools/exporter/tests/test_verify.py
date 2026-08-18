"""IR v1 の受理集合を固定する。

エクスポータ側の検証がランタイム（packages/runtime/src/format/ir.ts・container.ts）より広い受理集合を
持つと、「export は緑・ブラウザだけ落ちる」乖離になる。ここは規則 1 本ずつに
「受理する形」と「fail loudly する形」を並べて置く。
"""

from __future__ import annotations

import json
import re

import pytest
import torch
from safetensors.torch import save_file

from karume.ir import IR_METADATA_KEY
from karume.ops import OpContractError, state_window
from karume.verify import (
    ContainerError,
    IrError,
    _assert_scale_tensor,
    assert_op_contracts,
    assert_reader_layout,
    assert_runtime_support,
    parse_ir_graph,
    verify_model,
)


def base_graph() -> dict:
    """記号次元 1 本・initializer 1 本・ノード 1 本の最小の valid グラフ。"""
    return {
        "format": "karume-ir",
        "version": 1,
        "requires": {"ops": ["add"]},
        "symbols": ["T"],
        "inputs": [{"name": "x", "dtype": "f32", "shape": ["T", 4]}],
        "outputs": ["y"],
        "initializers": {"w": {"tensor": "enc.w", "storage": {"dtype": "f32"}}},
        "values": {
            "w": {"dtype": "f32", "shape": [4]},
            "y": {"dtype": "f32", "shape": ["T", 4]},
        },
        "nodes": [{"op": "add", "ins": ["x", "w"], "outs": ["y"], "attrs": {}}],
    }


def parse(**overrides):
    return parse_ir_graph(json.dumps({**base_graph(), **overrides}))


def write_container(path, graph: dict, tensors: dict[str, torch.Tensor], key=IR_METADATA_KEY):
    save_file(tensors, str(path), metadata={key: json.dumps(graph)})
    return path


class TestBaseline:
    def test_the_minimal_graph_is_accepted(self):
        graph = parse()

        assert graph.symbols == ["T"]
        assert graph.required_ops == ["add"]
        assert graph.values["y"].shape == ["T", 4]


class TestFormatAndVersion:
    def test_foreign_format_is_rejected(self):
        with pytest.raises(IrError, match=re.escape("graph.format")):
            parse(format="other-ir")

    def test_other_versions_are_rejected(self):
        with pytest.raises(IrError, match=re.escape("graph.version")):
            parse(version=2)

    def test_unknown_top_level_key_is_rejected(self):
        """未リリースにつき前方互換チャネルは持たない — 未知キーは黙って無視しない。"""
        with pytest.raises(IrError, match="未知のキー"):
            parse(metadata={})

    def test_missing_top_level_key_is_rejected(self):
        graph = base_graph()
        del graph["symbols"]

        with pytest.raises(IrError, match="必須キー"):
            parse_ir_graph(json.dumps(graph))


class TestRequiredOps:
    def test_undeclared_op_is_rejected(self):
        with pytest.raises(IrError, match="宣言漏れ"):
            parse(requires={"ops": []})

    def test_surplus_declaration_is_rejected(self):
        with pytest.raises(IrError, match="余剰"):
            parse(requires={"ops": ["add", "mul"]})

    def test_duplicate_declaration_is_rejected(self):
        with pytest.raises(IrError, match="重複"):
            parse(requires={"ops": ["add", "add"]})


class TestSymbols:
    def test_symbol_absent_from_the_declaration_is_rejected(self):
        with pytest.raises(IrError, match=re.escape("graph.symbols で宣言されていない")):
            parse(
                inputs=[{"name": "x", "dtype": "f32", "shape": ["U", 4]}],
                values={
                    "w": {"dtype": "f32", "shape": [4]},
                    "y": {"dtype": "f32", "shape": ["U", 4]},
                },
            )

    def test_symbol_never_appearing_in_an_input_is_rejected(self):
        """束縛は入力 shape の次元位置から取る — 値にしか現れない記号は束縛できない。"""
        with pytest.raises(IrError, match="次元位置に現れない"):
            parse(
                inputs=[{"name": "x", "dtype": "f32", "shape": [8, 4]}],
                values={
                    "w": {"dtype": "f32", "shape": [4]},
                    "y": {"dtype": "f32", "shape": ["T", 4]},
                },
            )

    def test_a_derived_dim_is_a_binding_source(self):
        """派生形（`2T`）でも解は一意なので束縛源になる（ADR 0057）。"""
        graph = parse(
            inputs=[{"name": "x", "dtype": "f32", "shape": ["2T", 4]}],
            values={
                "w": {"dtype": "f32", "shape": [4]},
                "y": {"dtype": "f32", "shape": ["2T", 4]},
            },
        )

        assert graph.symbols == ["T"]

    def test_symbol_name_outside_the_grammar_is_rejected(self):
        with pytest.raises(IrError, match="シンボル名"):
            parse(symbols=["2T"])

    def test_non_canonical_dim_spelling_is_rejected(self):
        with pytest.raises(IrError, match="正準文法"):
            parse(
                inputs=[{"name": "x", "dtype": "f32", "shape": ["T", 4]}],
                values={
                    "w": {"dtype": "f32", "shape": [4]},
                    "y": {"dtype": "f32", "shape": ["1T", 4]},
                },
            )

    def test_negative_dimension_is_rejected(self):
        with pytest.raises(IrError, match="非負整数でない"):
            parse(
                values={
                    "w": {"dtype": "f32", "shape": [-4]},
                    "y": {"dtype": "f32", "shape": ["T", 4]},
                }
            )


class TestSsaAndTopologicalOrder:
    def test_double_definition_is_rejected(self):
        with pytest.raises(IrError, match="SSA 単一代入違反"):
            parse(
                nodes=[
                    {"op": "add", "ins": ["x", "w"], "outs": ["y"], "attrs": {}},
                    {"op": "add", "ins": ["x", "w"], "outs": ["y"], "attrs": {}},
                ]
            )

    def test_forward_reference_is_rejected(self):
        with pytest.raises(IrError, match="前方参照"):
            parse(
                outputs=["y", "z"],
                values={
                    "w": {"dtype": "f32", "shape": [4]},
                    "y": {"dtype": "f32", "shape": ["T", 4]},
                    "z": {"dtype": "f32", "shape": ["T", 4]},
                },
                nodes=[
                    {"op": "add", "ins": ["x", "z"], "outs": ["y"], "attrs": {}},
                    {"op": "add", "ins": ["x", "w"], "outs": ["z"], "attrs": {}},
                ],
            )

    def test_undefined_output_is_rejected(self):
        with pytest.raises(IrError, match=re.escape("graph.outputs")):
            parse(outputs=["ghost"])

    def test_duplicate_output_is_rejected(self):
        with pytest.raises(IrError, match="重複"):
            parse(outputs=["y", "y"])

    def test_empty_outs_is_rejected_by_the_contract_layer(self):
        """0 本席（ADR 0067 決定 5）— パーサは `outs` の本数に意味を与えない。

        「0 本を許すのは契約が effect を宣言する op だけ」の執行点は契約層の出力数突合。
        パーサ側で本数を見ていた頃の拒否は、この 1 本に置き換わっている。
        """
        graph = parse(
            outputs=[],
            values={"w": {"dtype": "f32", "shape": [4]}},
            nodes=[{"op": "add", "ins": ["x", "w"], "outs": [], "attrs": {}}],
        )

        with pytest.raises(OpContractError, match="出力数が 0"):
            assert_op_contracts(graph)


class TestDeclarationCompleteness:
    def test_input_declared_again_in_values_is_rejected(self):
        with pytest.raises(IrError, match="二重宣言"):
            parse(
                values={
                    "x": {"dtype": "f32", "shape": ["T", 4]},
                    "w": {"dtype": "f32", "shape": [4]},
                    "y": {"dtype": "f32", "shape": ["T", 4]},
                }
            )

    def test_orphan_declaration_is_rejected(self):
        with pytest.raises(IrError, match="どのノードでも定義されない"):
            parse(
                values={
                    "w": {"dtype": "f32", "shape": [4]},
                    "y": {"dtype": "f32", "shape": ["T", 4]},
                    "ghost": {"dtype": "f32", "shape": [1]},
                }
            )

    def test_node_output_without_a_declaration_is_rejected(self):
        with pytest.raises(IrError, match="ノード出力"):
            parse(values={"w": {"dtype": "f32", "shape": [4]}})

    def test_initializer_without_a_declaration_is_rejected(self):
        with pytest.raises(IrError, match="values に dtype/shape 宣言が無い"):
            parse(
                initializers={
                    "w": {"tensor": "enc.w", "storage": {"dtype": "f32"}},
                    "extra": {"tensor": "enc.extra", "storage": {"dtype": "f32"}},
                }
            )

    def test_i32_initializer_pairs_with_i32_storage(self):
        """記号依存定数の焼き込み先（ADR 0010）— 意味論 i32 は生の int32 格納と組む。"""
        graph = parse(
            requires={"ops": ["mul"]},
            inputs=[{"name": "x", "dtype": "i32", "shape": ["T", 4]}],
            initializers={"w": {"tensor": "enc.w", "storage": {"dtype": "i32"}}},
            values={
                "w": {"dtype": "i32", "shape": [4]},
                "y": {"dtype": "i32", "shape": ["T", 4]},
            },
            nodes=[{"op": "mul", "ins": ["x", "w"], "outs": ["y"], "attrs": {}}],
        )

        assert graph.initializers["w"].storage.dtype == "i32"

    @pytest.mark.parametrize(
        ("semantic", "storage"),
        [("i32", "f32"), ("f32", "i32"), ("f32", "f16")],
        ids=["i32-as-f32", "f32-as-i32", "f32-as-f16-is-fine"],
    )
    def test_crossed_semantic_and_storage_dtypes_are_rejected(self, semantic, storage):
        """f32 の格納語彙（f32/f16/bf16/i8）と生の int32 は交差させない。

        交差を許すと i32 宣言の initializer が f16 のビット列として読まれる沈黙誤値になる。
        f32 × f16 だけは valid な組（宣言としては受理し、実行可否は対応表突合の層）。
        """

        def build():
            return parse(
                requires={"ops": ["mul"]},
                inputs=[{"name": "x", "dtype": semantic, "shape": ["T", 4]}],
                initializers={"w": {"tensor": "enc.w", "storage": {"dtype": storage}}},
                values={
                    "w": {"dtype": semantic, "shape": [4]},
                    "y": {"dtype": semantic, "shape": ["T", 4]},
                },
                nodes=[{"op": "mul", "ins": ["x", "w"], "outs": ["y"], "attrs": {}}],
            )

        if (semantic, storage) == ("f32", "f16"):
            assert build().initializers["w"].storage.dtype == "f16"
            return
        with pytest.raises(IrError, match="は組めない"):
            build()

    def test_bool_initializer_is_rejected(self):
        with pytest.raises(IrError, match="語彙外"):
            parse(
                requires={"ops": ["bitwise_not"]},
                inputs=[{"name": "x", "dtype": "bool", "shape": ["T", 4]}],
                outputs=["y"],
                initializers={"w": {"tensor": "enc.w", "storage": {"dtype": "f32"}}},
                values={
                    "w": {"dtype": "bool", "shape": [4]},
                    "y": {"dtype": "bool", "shape": ["T", 4]},
                },
                nodes=[{"op": "bitwise_not", "ins": ["x"], "outs": ["y"], "attrs": {}}],
            )

    def test_symbolic_initializer_shape_is_rejected(self):
        """束縛前に確定していない initializer は safetensors 側 shape と突合できない。"""
        with pytest.raises(IrError, match="記号次元は使えない"):
            parse(
                values={
                    "w": {"dtype": "f32", "shape": ["T"]},
                    "y": {"dtype": "f32", "shape": ["T", 4]},
                }
            )


class TestStorageDescriptor:
    def test_scale_on_a_non_quantized_storage_is_rejected(self):
        with pytest.raises(IrError, match="scale / group_size は付けられない"):
            parse(
                initializers={
                    "w": {"tensor": "enc.w", "storage": {"dtype": "f32", "scale": "enc.s"}}
                }
            )

    def test_group_size_on_a_non_quantized_storage_is_rejected(self):
        with pytest.raises(IrError, match="scale / group_size は付けられない"):
            parse(
                initializers={
                    "w": {"tensor": "enc.w", "storage": {"dtype": "bf16", "group_size": 32}}
                }
            )

    def test_non_positive_group_size_is_rejected(self):
        with pytest.raises(IrError, match="正整数でない"):
            parse(
                initializers={
                    "w": {
                        "tensor": "enc.w",
                        "storage": {"dtype": "i8", "scale": "enc.s", "group_size": 0},
                    }
                }
            )

    def test_group_size_beyond_the_safe_integer_range_is_rejected(self):
        """TS 側は JSON の数値を JS の number として読むので 2^53 以上は整数で持てない。"""
        with pytest.raises(IrError, match="安全整数"):
            parse(
                initializers={
                    "w": {
                        "tensor": "enc.w",
                        "storage": {"dtype": "i8", "scale": "enc.s", "group_size": 2**53},
                    }
                }
            )

    def test_i8_without_a_scale_is_rejected(self):
        """MUST: i8 は scale を明示宣言する
        （ADR 0019・TS 側 packages/runtime/src/format/ir.ts の鏡像）。

        既定 1.0 で補完すると、書き忘れが「全チャネル 1.0 で dequant した重み」に化けて
        ロードも実行も通ってしまう（差は O(scale) なのにどこにも例外が出ない）。
        """
        with pytest.raises(IrError, match="scale（scale テンソルのキー）が要る"):
            parse(initializers={"w": {"tensor": "enc.w", "storage": {"dtype": "i8"}}})

    def test_unknown_storage_key_is_rejected(self):
        with pytest.raises(IrError, match="未知のキー"):
            parse(
                initializers={
                    "w": {"tensor": "enc.w", "storage": {"dtype": "f32", "zero_point": 1}}
                }
            )

    def test_unknown_storage_dtype_is_rejected(self):
        with pytest.raises(IrError, match="語彙外"):
            parse(initializers={"w": {"tensor": "enc.w", "storage": {"dtype": "i2"}}})


class TestGroupQuantizedStorage:
    """group 量子化格納（i4 — ADR 0069 決定 2）の宣言規則。

    scale と group_size を必須にし、group 長は 2 冪かつ 16 以上・量子化軸（最終次元）が
    group 長で割り切れることを MUST とする（端数 group を作らない制約が、行境界・group 境界の
    バイト整列を保証している）。TS 側 `packages/runtime/src/format/ir.ts` の鏡像。
    """

    def _i4(self, storage: dict, last_dim: int = 32) -> dict:
        return {
            "initializers": {"w": {"tensor": "enc.w", "storage": storage}},
            "values": {
                "w": {"dtype": "f32", "shape": [4, last_dim]},
                "y": {"dtype": "f32", "shape": ["T", 4]},
            },
        }

    def test_a_well_formed_declaration_is_accepted(self):
        graph = parse(**self._i4({"dtype": "i4", "scale": "enc.s", "group_size": 32}))

        assert graph.initializers["w"].storage.group_size == 32
        assert graph.initializers["w"].storage.scale == "enc.s"

    def test_a_missing_group_size_is_rejected(self):
        with pytest.raises(IrError, match="group_size が要る"):
            parse(**self._i4({"dtype": "i4", "scale": "enc.s"}))

    def test_a_missing_scale_is_rejected(self):
        with pytest.raises(IrError, match="scale（scale テンソルのキー）が要る"):
            parse(**self._i4({"dtype": "i4", "group_size": 32}))

    @pytest.mark.parametrize("group_size", [24, 8], ids=["not-a-power-of-two", "below-16"])
    def test_a_group_size_outside_the_accepted_set_is_rejected(self, group_size):
        with pytest.raises(IrError, match="2 冪かつ 16 以上でない"):
            parse(**self._i4({"dtype": "i4", "scale": "enc.s", "group_size": group_size}))

    def test_a_quantization_axis_that_the_group_size_does_not_divide_is_rejected(self):
        with pytest.raises(IrError, match="割り切れない"):
            parse(**self._i4({"dtype": "i4", "scale": "enc.s", "group_size": 32}, last_dim=48))


def declared_states(states: dict, *, referenced: list[str] | None = None) -> dict:
    """スロットを参照する `state_append` を足したグラフの上書き集合。

    参照完全性（ADR 0067 決定 4 / 5 — 宣言されたスロットは 1 つ以上のノードから参照される
    MUST）が入ったので、「宣言だけして誰も参照しない」グラフはもう受理されない。スロット宣言
    そのものの受理集合を見るテストは、この足場の上で 1 点だけを動かす。

    `referenced` を渡すと参照するスロットを絞れる（孤立スロットの検出線を踏むため）。
    """
    names = list(states) if referenced is None else referenced
    nodes = [
        *base_graph()["nodes"],
        *(
            {"op": "state_append", "ins": ["x"], "outs": [], "attrs": {}, "states": {"slot": name}}
            for name in names
        ),
    ]
    return {
        "requires": {"ops": ["add", "state_append"] if names else ["add"]},
        "states": states,
        "nodes": nodes,
    }


class TestStateSlots:
    """名前付き state スロット（ADR 0066 決定 2）— 宣言側の受理集合。

    ノードからの参照（ADR 0067 の states 欄 / `state_append`）は TestNodeStates が持つ。ここは
    「参照は最小の足場で満たしたうえで、宣言 1 点だけを動かす」形に揃える。
    """

    def test_a_graph_without_the_section_has_no_slots(self):
        """節を持たないグラフ（= 既存の全モデル）は無風。"""
        graph = parse()

        assert graph.states == {}
        # MUST: 空の states は書き戻さない — 出すと既存モデルのグラフ JSON がバイト単位で変わる。
        assert "states" not in graph.to_dict()

    def test_an_empty_section_is_accepted(self):
        graph = parse(states={})

        assert graph.states == {}
        assert "states" not in graph.to_dict()

    def test_slots_with_different_capacities_are_accepted(self):
        """sliding 層と full 層は容量の違う別スロット（層 × 均一 KV の前提を作らない）。"""
        graph = parse(
            **declared_states(
                {
                    "layer0.k": {"dtype": "f32", "shape": [1, 2, 512, 128]},
                    "layer0.v": {"dtype": "f32", "shape": [1, 2, 512, 128]},
                    "layer1.k": {"dtype": "f32", "shape": [1, 2, 131072, 128]},
                }
            )
        )

        assert list(graph.states) == ["layer0.k", "layer0.v", "layer1.k"]
        assert graph.states["layer0.k"].shape == [1, 2, 512, 128]
        assert graph.states["layer1.k"].shape == [1, 2, 131072, 128]

    @pytest.mark.parametrize("shape", [[8], [1, 2, 3, 4], ["T", 4]])
    def test_accepted_shapes(self, shape):
        """rank 1..4 の両端と記号次元（束縛源は入力 shape または states shape の次元位置）。"""
        graph = parse(**declared_states({"cache": {"dtype": "f32", "shape": shape}}))

        assert graph.states["cache"].shape == shape

    def test_the_section_round_trips_in_place(self):
        """宣言節なので `values` の直後・`nodes` の前に戻す（TS 側の並びと同じ）。"""
        slot = {"dtype": "f32", "shape": [1, 2, 512, 128]}
        graph = parse(**declared_states({"layer0.k": slot}))
        keys = list(graph.to_dict())

        assert graph.to_dict()["states"] == {"layer0.k": slot}
        assert keys.index("states") == keys.index("values") + 1
        assert keys.index("nodes") == keys.index("states") + 1

    def test_a_slot_that_no_node_references_is_rejected(self):
        """参照完全性（ADR 0067 決定 4 / 5）— values の孤立宣言と同じ穴を塞ぐ。

        誰も参照しないスロットは GenerationContext が確保だけして誰も読まない容量
        （KV なら数十 MiB 単位）が黙って残る。
        """
        with pytest.raises(IrError, match=re.escape("graph.states['orphan']: どのノードからも")):
            parse(
                **declared_states(
                    {
                        "used": {"dtype": "f32", "shape": [4]},
                        "orphan": {"dtype": "f32", "shape": [4]},
                    },
                    referenced=["used"],
                )
            )

    def test_the_reserved_f16_storage_is_rejected_as_unsupported(self):
        """f16 は席の予約だけ（ADR 0066 追記 5）— 「語彙外」ではなく「未対応」で落とす。"""
        with pytest.raises(IrError, match="は未対応"):
            parse(states={"cache": {"dtype": "f16", "shape": [4]}})

    @pytest.mark.parametrize("dtype", ["i32", "bool", "bf16", "i8", "f64"])
    def test_other_dtypes_are_outside_the_vocabulary(self, dtype):
        with pytest.raises(IrError, match=re.escape("は語彙外（f32）")):
            parse(states={"cache": {"dtype": dtype, "shape": [4]}})

    @pytest.mark.parametrize("shape", [[1, 1, 1, 1, 1], []])
    def test_rank_outside_one_to_four_is_rejected(self, shape):
        with pytest.raises(IrError, match=re.escape("は 1..4 の外")):
            parse(states={"cache": {"dtype": "f32", "shape": shape}})

    def test_zero_capacity_is_rejected(self):
        """容量込みの具体形なので数値次元は正整数（`values` の非負とは違う）。"""
        with pytest.raises(IrError, match="正整数でない"):
            parse(states={"cache": {"dtype": "f32", "shape": [1, 0, 4]}})

    def test_negative_dimension_is_rejected(self):
        with pytest.raises(IrError, match="非負整数でない"):
            parse(states={"cache": {"dtype": "f32", "shape": [-4]}})

    def test_undeclared_symbol_is_rejected(self):
        with pytest.raises(IrError, match=re.escape("graph.symbols で宣言されていない")):
            parse(states={"cache": {"dtype": "f32", "shape": ["S", 4]}})

    def test_non_canonical_dim_spelling_is_rejected(self):
        with pytest.raises(IrError, match="正準文法"):
            parse(states={"cache": {"dtype": "f32", "shape": ["1T", 4]}})

    @pytest.mark.parametrize(
        ("states", "message"),
        [
            ([], "graph.states: オブジェクトでない"),
            ({"cache": 4}, "graph.states['cache']: オブジェクトでない"),
            ({"cache": {"dtype": "f32", "shape": 4}}, "shape: 配列でない"),
            ({"cache": {"shape": [4]}}, "必須キー 'dtype' が無い"),
            ({"cache": {"dtype": "f32", "shape": [4], "window": 512}}, "未知のキー 'window'"),
            # 空のスロット名は参照側の欄（ADR 0067）が受理しない = 参照できない宣言になる。
            ({"": {"dtype": "f32", "shape": [4]}}, "スロット名: 空でない文字列でない"),
        ],
    )
    def test_malformed_sections_are_rejected(self, states, message):
        with pytest.raises(IrError, match=re.escape(message)):
            parse(states=states)

    @pytest.mark.parametrize("name", ["x", "w", "y"])
    def test_a_slot_named_after_a_value_is_rejected(self, name):
        """スロット名は値名前空間と別だが、同名は取り違えの検出線として拒否する
        （入力 / initializer / ノード出力の 3 役）。
        """
        with pytest.raises(IrError, match=re.escape(f"graph.states['{name}']: 値名と同名")):
            parse(states={name: {"dtype": "f32", "shape": [4]}})


def kv_graph(**overrides) -> dict:
    """states 形 attention + `state_append` の最小グラフ（decode 1 step・1 層ぶん）。

    `q[1,8,1,16]` / `k`・`v[1,2,1,16]`（GQA 4:1）と容量 64 のスロット 2 本。ins は**今 step の
    chunk だけ**で、過去はスロットが持つ（ADR 0067 決定 4）。発行順は「全読者 → append」
    （決定 5b）。
    """
    graph = {
        "format": "karume-ir",
        "version": 1,
        "requires": {"ops": ["attention", "state_append"]},
        "symbols": [],
        "inputs": [
            {"name": "q", "dtype": "f32", "shape": [1, 8, 1, 16]},
            {"name": "k", "dtype": "f32", "shape": [1, 2, 1, 16]},
            {"name": "v", "dtype": "f32", "shape": [1, 2, 1, 16]},
        ],
        "outputs": ["o"],
        "initializers": {},
        "values": {"o": {"dtype": "f32", "shape": [1, 8, 1, 16]}},
        "states": {
            "l0.k": {"dtype": "f32", "shape": [1, 2, 64, 16]},
            "l0.v": {"dtype": "f32", "shape": [1, 2, 64, 16]},
        },
        "nodes": [
            {
                "op": "attention",
                "ins": ["q", "k", "v"],
                "outs": ["o"],
                "attrs": {"scale": 0.25},
                "states": {"k": "l0.k", "v": "l0.v"},
            },
            {
                "op": "state_append",
                "ins": ["k"],
                "outs": [],
                "attrs": {},
                "states": {"slot": "l0.k"},
            },
            {
                "op": "state_append",
                "ins": ["v"],
                "outs": [],
                "attrs": {},
                "states": {"slot": "l0.v"},
            },
        ],
    }
    graph.update(overrides)
    # requires.ops ≡ 使用 op 集合（明示指定が無ければノード列から導く — ノードを差し替える
    # テストごとに宣言を書き直さないため）。
    if "requires" not in overrides:
        graph["requires"] = {"ops": sorted({node["op"] for node in graph["nodes"]})}
    return graph


def parse_kv(**overrides):
    return parse_ir_graph(json.dumps(kv_graph(**overrides)))


def check_kv(**overrides):
    """パース → 契約検査（アリティ / states 欄 / 順序 / shape）まで通す。"""
    graph = parse_kv(**overrides)
    assert_op_contracts(graph)
    return graph


def kv_windows(*windows: int | None) -> list[dict]:
    """3 ノード（attention / append k / append v）の `window` だけを差し替えたノード列。"""
    nodes = [dict(node) for node in kv_graph()["nodes"]]
    for node, window in zip(nodes, windows, strict=True):
        node["attrs"] = dict(node["attrs"])
        if window is not None:
            node["attrs"]["window"] = window
    return nodes


def plain_attention_graph(attrs: dict) -> dict:
    """states 欄を持たない**従来形** attention の最小グラフ（欄の有無で分かれる契約の対照）。"""
    graph = kv_graph()
    del graph["states"]
    graph["requires"] = {"ops": ["attention"]}
    graph["nodes"] = [{"op": "attention", "ins": ["q", "k", "v"], "outs": ["o"], "attrs": attrs}]
    return graph


class TestNodeStates:
    """ノードの `states` 欄と `state_append`（ADR 0067 決定 4 / 5 / 5b）。

    TS 側（format/ir.ts の parseNodeStates・ops/contracts.ts の assertStateField・
    runtime/plan.ts の assertStateOrder・ops/shapes.ts の assertStateSlotForm）と**同じ
    受理集合・同じ拒否集合**であることがここの仕様。層が違うだけで、片側だけ緩むと
    「export は緑・ブラウザだけ落ちる」が state 軸で復活する。
    """

    def test_a_decode_step_graph_is_accepted(self):
        graph = check_kv()

        assert graph.nodes[0].states == {"k": "l0.k", "v": "l0.v"}
        assert graph.nodes[1].states == {"slot": "l0.k"}
        assert assert_runtime_support(graph) is None

    def test_the_field_round_trips_only_when_present(self):
        """欄の有無がそのまま形の判別（ADR 0067 決定 4）— 空欄は書き戻さない。"""
        node = parse_kv().nodes[0].to_dict()

        assert list(node) == ["op", "ins", "outs", "attrs", "states"]
        assert node["states"] == {"k": "l0.k", "v": "l0.v"}
        assert "states" not in parse().nodes[0].to_dict()

    def test_a_layer_without_an_append_is_accepted(self):
        """KV 共有層は「`state_append` ノードが無い」だけで表せる（ORT の kv_empty 相当）。"""
        graph = check_kv(nodes=kv_graph()["nodes"][:1])

        assert [node.op for node in graph.nodes] == ["attention"]

    def test_a_sliding_layer_is_accepted(self):
        """省略可能 attrs `window`（欄の不存在 = 全 context）。読み書きで同じ値を宣言する。"""
        graph = check_kv(nodes=kv_windows(64, 64, 64))

        assert graph.nodes[0].attrs["window"] == 64
        # 列挙門（capability 突合）も必須と省略可能の**和**で見る — 落とすと states 形の
        # 正しいグラフが「未実装 attrs」として拒否される。
        assert assert_runtime_support(graph) is None

    def test_an_undeclared_slot_reference_is_rejected(self):
        """未宣言スロットは実体を持たない名前 — 確保も束縛もされないまま実行段へ抜ける。"""
        nodes = [dict(node) for node in kv_graph()["nodes"]]
        nodes[0] = {**nodes[0], "states": {"k": "ghost", "v": "l0.v"}}

        with pytest.raises(IrError, match=re.escape("state スロット 'ghost' が graph.states")):
            parse_kv(nodes=nodes)

    def test_an_effect_op_with_outputs_is_rejected(self):
        """`state_append` は**値を定義しない**（出力 0 本の宣言は契約表の空列そのもの）。"""
        nodes = [dict(node) for node in kv_graph()["nodes"]]
        nodes[1] = {**nodes[1], "outs": ["z"]}
        values = {**kv_graph()["values"], "z": {"dtype": "f32", "shape": [1, 2, 1, 16]}}

        with pytest.raises(OpContractError, match="出力数が 1"):
            check_kv(nodes=nodes, values=values)

    def test_the_same_slot_in_two_fields_is_rejected(self):
        """`k` と `v` に同じスロットを書いた取り違えは shape では捕まらない（同形だから）。"""
        nodes = [dict(node) for node in kv_graph()["nodes"][:2]]
        nodes[0] = {**nodes[0], "states": {"k": "l0.k", "v": "l0.k"}}

        with pytest.raises(OpContractError, match="複数の欄から参照している"):
            check_kv(nodes=nodes, states={"l0.k": kv_graph()["states"]["l0.k"]})

    def test_a_states_form_attention_with_a_mask_is_rejected(self):
        """states 形は causal 固定で mask tensor を実体化しない（省略可能な末尾入力を取らない）。"""
        nodes = [dict(node) for node in kv_graph()["nodes"]]
        nodes[0] = {**nodes[0], "ins": ["q", "k", "v", "m"]}
        inputs = [*kv_graph()["inputs"], {"name": "m", "dtype": "f32", "shape": [1, 1, 1, 64]}]

        with pytest.raises(OpContractError, match="states 形は入力 3 本ちょうど"):
            check_kv(nodes=nodes, inputs=inputs)

    @pytest.mark.parametrize("window", [0, -1, 1.5, "64", True])
    def test_window_outside_the_value_range_is_rejected(self, window):
        with pytest.raises(OpContractError, match="1 以上の整数でない"):
            check_kv(nodes=kv_windows(window, window, window))

    @pytest.mark.parametrize("window", [8.0, 8e0, 64.0])
    def test_an_integral_float_window_is_accepted(self, window):
        """JSON の `8.0` / `8e0` は int 扱い（TS 側の受理集合と揃える）。

        TS は JSON.parse が単一の number を返すので「整数値の float」という区別が無く
        （`Number.isSafeInteger(8.0)` は true）、Python が float を丸ごと拒むと**ランタイムが
        読める graph をエクスポータの検証だけが読めない**乖離になる（受理集合はどちらの向きにも
        ずれてはいけない）。
        """
        graph = check_kv(nodes=kv_windows(window, window, window))

        for node in graph.nodes:
            assert isinstance(node.attrs["window"], float), "float を渡していない（空振り）"
            assert state_window(node.attrs, "t") == int(window)
            assert isinstance(state_window(node.attrs, "t"), int), "int へ正規化されていない"

    @pytest.mark.parametrize(
        "window",
        [1.5, float(2**53), float("inf"), float("-inf"), float("nan")],
    )
    def test_a_non_integral_or_out_of_range_float_window_is_rejected(self, window):
        """整数値でない / safe range 外の float は従来どおり拒否（正規化は受理集合を広げない）。

        非有限値は JSON 読み（`parse_graph_json` の非標準リテラル拒否）でも止まるので、ここは
        直接呼びで attrs 層そのものを踏む。
        """
        with pytest.raises(OpContractError, match="1 以上の整数でない"):
            state_window({"window": window}, "t")

    def test_window_on_a_plain_attention_is_rejected(self):
        """省略可能 attrs は states 形専用 — 従来形に書けると誰も読まない attr ができる。"""
        with pytest.raises(OpContractError, match="states 欄を持つノードでのみ"):
            assert_op_contracts(
                parse_ir_graph(json.dumps(plain_attention_graph({"scale": 0.25, "window": 8})))
            )

    def test_a_plain_attention_is_untouched(self):
        """欄が無いノードは従来契約そのまま（既存資産・既存門は無風）。"""
        graph = parse_ir_graph(json.dumps(plain_attention_graph({"scale": 0.25})))

        assert assert_op_contracts(graph) is None
        assert graph.nodes[0].states == {}

    def test_a_reader_after_the_append_is_rejected(self):
        """append より後の読者は「今 step の k/v を過去として二重に読む」（決定 5b の②）。"""
        nodes = kv_graph()["nodes"]

        with pytest.raises(IrError, match="より後に読者"):
            check_kv(nodes=[nodes[1], nodes[2], nodes[0]])

    def test_two_appends_to_one_slot_are_rejected(self):
        """1 step に 2 回書くと ring の位置式が二重に進む（決定 5b の①）。"""
        nodes = kv_graph()["nodes"]

        with pytest.raises(IrError, match=re.escape("state_append が 2 本")):
            check_kv(nodes=[nodes[0], nodes[1], nodes[2], nodes[1]])

    @pytest.mark.parametrize(
        ("windows", "why"),
        [
            ((64, 32, 64), "読み側と書き側で値が違う"),
            ((None, 64, 64), "読み側だけ宣言が無い（全 context と ring の取り違え）"),
            ((64, None, 64), "書き側だけ宣言が無い"),
        ],
    )
    def test_a_window_mismatch_on_one_slot_is_rejected(self, windows, why):
        """論理 col → 物理 row の写像は読み書き同式 MUST（決定 4）— 存在有無も値も一致する。"""
        with pytest.raises(IrError, match=re.escape("attrs.window が食い違う")):
            check_kv(nodes=kv_windows(*windows))

    def test_a_state_append_input_that_is_not_rank_four_is_rejected(self):
        """入力の宣言 shape は `[B,Hkv,M,D]` 固定（attention の ins と同じ物理 chunk 次元）。

        rank 5 は B / Hkv / D が**偶然そろう**（軸 3 が D のまま）ので、rank 検査だけが
        検出線になる形。
        """
        inputs = [*kv_graph()["inputs"], {"name": "k5", "dtype": "f32", "shape": [1, 2, 1, 16, 1]}]
        nodes = [dict(node) for node in kv_graph()["nodes"]]
        nodes[1] = {**nodes[1], "ins": ["k5"]}

        with pytest.raises(OpContractError, match=re.escape("[B,Hkv,M,D] の rank-4 のみ")):
            check_kv(inputs=inputs, nodes=nodes)

    def test_a_zero_length_chunk_is_rejected(self):
        """空軸の softmax は amax の identity が定義できない（従来形の N=0 拒否と同じ理由）。"""
        inputs = [
            {"name": "q", "dtype": "f32", "shape": [1, 8, 0, 16]},
            {"name": "k", "dtype": "f32", "shape": [1, 2, 0, 16]},
            {"name": "v", "dtype": "f32", "shape": [1, 2, 0, 16]},
        ]

        with pytest.raises(OpContractError, match="M が 0 の chunk"):
            check_kv(inputs=inputs, values={"o": {"dtype": "f32", "shape": [1, 8, 0, 16]}})

    def test_a_slot_that_is_not_rank_four_is_rejected(self):
        """スロットの物理形は `[B,Hkv,C,D]` 固定（ADR 0067 決定 4 の②）。"""
        slot = {"dtype": "f32", "shape": [1, 2, 64]}

        with pytest.raises(OpContractError, match=re.escape("[B,Hkv,C,D] の rank-4 のみ")):
            check_kv(states={"l0.k": slot, "l0.v": slot})

    def test_a_slot_whose_head_count_differs_from_the_input_is_rejected(self):
        """別の層のスロットを参照した形 — 容量が同じだと沈黙誤読になる。"""
        slot = {"dtype": "f32", "shape": [1, 4, 64, 16]}

        with pytest.raises(OpContractError, match="B / Hkv / D が不一致"):
            check_kv(states={"l0.k": slot, "l0.v": slot})

    def test_slots_of_different_capacity_for_k_and_v_are_rejected(self):
        """片方だけ先に wrap する ring は値が静かにずれる。"""
        with pytest.raises(OpContractError, match="同形でない"):
            check_kv(
                states={
                    "l0.k": {"dtype": "f32", "shape": [1, 2, 64, 16]},
                    "l0.v": {"dtype": "f32", "shape": [1, 2, 32, 16]},
                }
            )

    def test_a_window_larger_than_the_capacity_is_rejected(self):
        """ring 幅が容量を超えると、書いた行が同 step 中に自分で潰れる。"""
        with pytest.raises(OpContractError, match="スロット容量 64 を超える"):
            check_kv(nodes=kv_windows(65, 65, 65))

    def test_a_symbolic_capacity_defers_the_window_bound(self):
        """記号容量では `window <= C` を判定しない（束縛は `createGenerationContext` の層）。

        TS 側は束縛解決後の数値しか扱わないので常に判定できる — この 1 点だけが片側で保留に
        なる非対称で、「宣言だけでは決められないものを決めたことにしない」という既存の規律
        （broadcast 可否・整除 broadcast と同じ）に揃えた結果。
        """
        slot = {"dtype": "f32", "shape": [1, 2, "C", 16]}
        graph = check_kv(
            symbols=["C"],
            nodes=kv_windows(9999, 9999, 9999),
            states={
                "l0.k": slot,
                "l0.v": slot,
            },
        )

        assert graph.symbols == ["C"]

    def test_a_states_only_symbol_is_bindable(self):
        """states にしか現れない記号（KV 容量）も束縛点を持つ（ADR 0066 追記 7）。"""
        slot = {"dtype": "f32", "shape": [1, 2, "C", 16]}

        assert check_kv(symbols=["C"], states={"l0.k": slot, "l0.v": slot}).symbols == ["C"]

    def test_a_states_only_symbol_in_a_value_shape_is_rejected(self):
        """値 shape の解決に効くのは入力由来の束縛だけ — 現れると実行時に必ず束縛不能。"""
        slot = {"dtype": "f32", "shape": [1, 2, "C", 16]}

        with pytest.raises(IrError, match=re.escape("states 専用記号 'C' が値 shape に現れる")):
            parse_kv(
                symbols=["C"],
                states={"l0.k": slot, "l0.v": slot},
                values={"o": {"dtype": "f32", "shape": [1, 8, "C", 16]}},
            )

    def test_a_symbol_bound_nowhere_is_still_rejected(self):
        """束縛点は「入力 shape ∪ states shape」— どちらにも無い記号は従来どおり落とす。"""
        with pytest.raises(IrError, match="次元位置に現れない"):
            parse_kv(symbols=["C"])

    @pytest.mark.parametrize(
        ("states", "message"),
        [
            ([], "states: オブジェクトでない"),
            ({"k": 4}, "states['k']: 空でない文字列でない"),
            ({"k": ""}, "states['k']: 空でない文字列でない"),
            ({"": "l0.k"}, "states のキー: 空でない文字列でない"),
        ],
    )
    def test_malformed_fields_are_rejected(self, states, message):
        nodes = [dict(node) for node in kv_graph()["nodes"]]
        nodes[0] = {**nodes[0], "states": states}

        with pytest.raises(IrError, match=re.escape(message)):
            parse_kv(nodes=nodes)

    @pytest.mark.parametrize(
        ("states", "message"),
        [
            ({"k": "l0.k"}, "states 欄のキーが ['k']"),
            ({"k": "l0.k", "v": "l0.v", "extra": "l0.k"}, "states 欄のキーが ['extra', 'k', 'v']"),
        ],
    )
    def test_a_key_set_other_than_the_contract_is_rejected(self, states, message):
        """キー集合は契約の宣言と完全一致 — 部分集合を許すと `v` 書き忘れが通る。"""
        nodes = [dict(node) for node in kv_graph()["nodes"]]
        nodes[0] = {**nodes[0], "states": states}

        with pytest.raises(OpContractError, match=re.escape(message)):
            check_kv(nodes=nodes)

    def test_a_state_append_without_the_field_is_rejected(self):
        """`state_append` の states 欄は必須（書き先の無い書き込み op は存在しない）。"""
        nodes = [dict(node) for node in kv_graph()["nodes"]]
        del nodes[1]["states"]

        with pytest.raises(OpContractError, match="states 欄が無い"):
            check_kv(nodes=nodes)

    def test_an_op_without_a_states_contract_is_rejected(self):
        """欄を持てない op に欄を書いた形（契約が `None` = 持てない、の執行）。"""
        nodes = [
            *kv_graph()["nodes"],
            {"op": "relu", "ins": ["o"], "outs": ["r"], "attrs": {}, "states": {"k": "l0.k"}},
        ]
        values = {**kv_graph()["values"], "r": {"dtype": "f32", "shape": [1, 8, 1, 16]}}

        with pytest.raises(OpContractError, match="states 欄を持たない"):
            check_kv(nodes=nodes, values=values)


class TestIntegralFloatDimensions:
    """JSON の整数値 float（`1.0` / `1e0`）を TS と同じく受理する。

    TS は JSON.parse が単一の number を返すので `4` と `4.0` を区別できない（どちらも
    `Number.isSafeInteger` が true）。ここが float を丸ごと拒むと「ランタイムは読めるのに
    エクスポータの検証だけが落ちる」乖離になるので、両側で同じ受理集合を固定する。
    """

    @staticmethod
    def parse_raw_dim(raw: str, **overrides):
        """次元 1 個だけを生の JSON テキストで差し込む（json.dumps は `1e0` を `1.0` に畳む）。"""
        text = json.dumps({**base_graph(), **overrides})
        assert '"__DIM__"' in text
        return parse_ir_graph(text.replace('"__DIM__"', raw))

    @pytest.mark.parametrize("raw", ["4.0", "4e0", "0.4e1"])
    def test_integral_float_value_dims_are_accepted_as_int(self, raw):
        graph = self.parse_raw_dim(
            raw,
            values={
                "w": {"dtype": "f32", "shape": ["__DIM__"]},
                "y": {"dtype": "f32", "shape": ["T", 4]},
            },
        )

        assert graph.values["w"].shape == [4]
        # 正規化の結果は int（float のまま持つと to_dict が `4.0` を書き戻して往復が壊れる）。
        assert all(isinstance(dim, int) for dim in graph.values["w"].shape)

    @pytest.mark.parametrize("raw", ["4.0", "4e0"])
    def test_integral_float_state_dims_are_accepted_as_int(self, raw):
        graph = self.parse_raw_dim(
            raw, **declared_states({"cache": {"dtype": "f32", "shape": [1, "__DIM__"]}})
        )

        assert graph.states["cache"].shape == [1, 4]
        assert graph.to_dict()["states"]["cache"]["shape"] == [1, 4]

    @pytest.mark.parametrize("raw", ["1.5", "15e-1"])
    def test_non_integral_float_dims_are_rejected(self, raw):
        with pytest.raises(IrError, match="数値でも文字列でもない"):
            self.parse_raw_dim(
                raw,
                values={
                    "w": {"dtype": "f32", "shape": ["__DIM__"]},
                    "y": {"dtype": "f32", "shape": ["T", 4]},
                },
            )
        with pytest.raises(IrError, match="数値でも文字列でもない"):
            self.parse_raw_dim(raw, states={"cache": {"dtype": "f32", "shape": [1, "__DIM__"]}})

    def test_integral_float_beyond_the_safe_range_is_rejected(self):
        """`Number.isSafeInteger` と同じ上限で落ちる（TS だけが読めない値を通さない）。"""
        with pytest.raises(IrError, match="非負整数でない"):
            self.parse_raw_dim(
                "1e300",
                values={
                    "w": {"dtype": "f32", "shape": ["__DIM__"]},
                    "y": {"dtype": "f32", "shape": ["T", 4]},
                },
            )


class TestJsonLiterals:
    @pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity"])
    def test_non_standard_literals_are_rejected(self, literal):
        with pytest.raises(IrError, match="非標準リテラル"):
            parse_ir_graph('{"format":"karume-ir","version":' + literal + "}")

    def test_overflowing_exponent_is_rejected_at_the_value_level(self):
        """`1e999` は構文としては有効だが Infinity へ丸まる — 値で弾く。"""
        with pytest.raises(IrError, match="有限でない"):
            parse_ir_graph('{"format":"karume-ir","version":1e999}')

    def test_broken_json_is_reported_as_such(self):
        with pytest.raises(IrError, match="解析できない"):
            parse_ir_graph("{")


class TestRuntimeSupport:
    """ランタイム対応表との突合は op 名だけでなく dtype・attrs・格納 dtype まで見る。"""

    def test_the_baseline_graph_is_executable(self):
        assert assert_runtime_support(parse()) is None

    def test_unsupported_ops_are_enumerated_at_once(self):
        """語彙にすら無い op は**まとめて**列挙されて落ちる（1 件ずつ落とさない）。

        NOTE: 代表として使う op は語彙の拡張のたびに入れ替わる — conv_transpose1d は
        ADR 0015 で、conv2d は ADR 0017 で契約表に入った。ここは 3d 系（conv3d /
        conv_transpose2d）で**複数件の列挙**を固定する。
        """
        graph = parse(
            requires={"ops": ["conv3d", "conv_transpose2d"]},
            outputs=["z"],
            values={
                "w": {"dtype": "f32", "shape": [4]},
                "y": {"dtype": "f32", "shape": ["T", 4]},
                "z": {"dtype": "f32", "shape": ["T", 4]},
            },
            nodes=[
                {"op": "conv3d", "ins": ["x"], "outs": ["y"], "attrs": {}},
                {"op": "conv_transpose2d", "ins": ["y", "w"], "outs": ["z"], "attrs": {}},
            ],
        )

        with pytest.raises(ContainerError) as err:
            assert_runtime_support(graph)

        message = str(err.value)
        assert "非対応 op" in message
        assert "conv3d" in message and "conv_transpose2d" in message

    def test_every_semantic_dtype_can_be_transferred(self):
        """転送層の軸は意味論 dtype 全語彙を受理する（ADR 0009 で i32 / bool を解禁）。

        どのノードも消費しない入力にも転送層の制約は実在する（実行器は全 graph.inputs を
        転送する）ので、軸自体は残す。
        """
        graph = parse(
            inputs=[
                {"name": "x", "dtype": "f32", "shape": ["T", 4]},
                {"name": "ids", "dtype": "i32", "shape": ["T"]},
                {"name": "mask", "dtype": "bool", "shape": ["T"]},
            ]
        )

        assert_runtime_support(graph)

    def test_a_storage_only_dtype_cannot_be_a_graph_input(self):
        """反対側 — 転送できるのは意味論 dtype だけで、格納 dtype は入力に置けない。

        `f16` は initializer の格納としては valid（TestDeclarationCompleteness の
        f32-as-f16）なので、「格納できる dtype」と「転送できる dtype」が別語彙であることは
        こちらからしか固定できない。
        """
        with pytest.raises(IrError, match="意味論 dtype 'f16'"):
            parse(inputs=[{"name": "x", "dtype": "f16", "shape": ["T", 4]}])

    def test_an_op_that_does_not_accept_the_input_dtype_is_reported(self):
        """転送できても op が受理しない dtype はここで落ちる（軸が 2 本ある理由）。"""
        graph = parse(inputs=[{"name": "x", "dtype": "i32", "shape": ["T", 4]}])

        with pytest.raises(ContainerError, match="非対応 意味論 dtype"):
            assert_runtime_support(graph)

    def test_swapped_slots_of_a_per_slot_op_are_enumerated(self):
        """受理集合の**和**だけで突き合わせると素通りする形
        （packages/runtime/src/format/container.ts と同義）。

        gather の和は {f32, i32} なので、値と添字を逆に渡してもどちらの dtype も和には入る。
        列挙門がスロットを見ないと、契約検査まで落ちて 1 件ずつ止まることになり
        「非対応は全件列挙」の意図が壊れる。
        """
        graph = parse(
            requires={"ops": ["gather"]},
            inputs=[
                {"name": "src", "dtype": "i32", "shape": ["T", 4]},
                {"name": "idx", "dtype": "f32", "shape": ["T", 3]},
            ],
            initializers={},
            values={"y": {"dtype": "f32", "shape": ["T", 3]}},
            nodes=[{"op": "gather", "ins": ["src", "idx"], "outs": ["y"], "attrs": {}}],
        )

        with pytest.raises(ContainerError) as err:
            assert_runtime_support(graph)

        message = str(err.value)
        assert "非対応 意味論 dtype (2)" in message
        assert "値 'src': i32" in message and "値 'idx': f32" in message

    def test_the_output_of_a_per_slot_op_is_checked_against_the_value_slot(self):
        """出力はスロット 0（値の側）と同型 — 和で見ると gather の i32 出力が通ってしまう。"""
        graph = parse(
            requires={"ops": ["gather"]},
            inputs=[
                {"name": "src", "dtype": "f32", "shape": ["T", 4]},
                {"name": "idx", "dtype": "i32", "shape": ["T", 3]},
            ],
            initializers={},
            values={"y": {"dtype": "i32", "shape": ["T", 3]}},
            nodes=[{"op": "gather", "ins": ["src", "idx"], "outs": ["y"], "attrs": {}}],
        )

        with pytest.raises(ContainerError, match="値 'y': i32"):
            assert_runtime_support(graph)

    def test_swapped_output_slots_of_a_multi_output_op_are_enumerated(self):
        """多出力 op（topk）の出力宣言を slot 間で**入れ替えた**形
        （packages/runtime/tests/format_container_test.ts の同形テストと対）。

        全出力を slot 0 の受理集合で見る退行だと、値の側（f32）と添字の側（i32）がどちらも
        同じ集合に照らして判定され、片方しか列挙されない（あるいは両方素通りする）。
        """
        graph = parse(
            requires={"ops": ["topk"]},
            outputs=["v", "i"],
            initializers={},
            # slot 0 は値（f32）・slot 1 は添字（i32）なので、この 2 本は**どちらも**非対応
            values={
                "v": {"dtype": "i32", "shape": ["T", 2]},
                "i": {"dtype": "f32", "shape": ["T", 2]},
            },
            nodes=[{"op": "topk", "ins": ["x"], "outs": ["v", "i"], "attrs": {"k": 2}}],
        )

        with pytest.raises(ContainerError) as err:
            assert_runtime_support(graph)

        message = str(err.value)
        assert "非対応 意味論 dtype (2)" in message
        assert "値 'v': i32" in message and "値 'i': f32" in message

    def test_unsupported_storage_dtype_is_reported(self):
        graph = parse(initializers={"w": {"tensor": "enc.w", "storage": {"dtype": "bf16"}}})

        with pytest.raises(ContainerError, match="非対応 格納 dtype"):
            assert_runtime_support(graph)

    def test_group_size_on_a_storage_other_than_i4_is_reported(self):
        """group 量子化を受理する格納は i4 だけ（ADR 0069 決定 2）。

        i8 に付いた group_size は実行経路が無く、黙って無視すると group ごとの scale を
        per-channel として読む沈黙誤値になる。TS 側 assertRuntimeSupport の鏡像で、
        この検査が Python 側に無いと「verify は緑・ブラウザだけ落ちる」非対称になる。
        """
        graph = parse(
            initializers={
                "w": {
                    "tensor": "enc.w",
                    "storage": {"dtype": "i8", "scale": "enc.s", "group_size": 32},
                }
            }
        )

        with pytest.raises(ContainerError) as err:
            assert_runtime_support(graph)

        message = str(err.value)
        assert "非対応 group 量子化 (1): w" in message
        assert "i4 のみ" in message

    def test_unknown_attrs_on_a_known_op_are_reported(self):
        graph = parse(
            nodes=[{"op": "add", "ins": ["x", "w"], "outs": ["y"], "attrs": {"alpha": 2}}]
        )

        with pytest.raises(ContainerError, match="未実装 attrs"):
            assert_runtime_support(graph)


class TestOpContracts:
    def test_arity_violation_is_rejected(self):
        graph = parse(nodes=[{"op": "add", "ins": ["x"], "outs": ["y"], "attrs": {}}])

        with pytest.raises(OpContractError, match="入力数が 1"):
            assert_op_contracts(graph)

    def test_multiple_outputs_are_rejected(self):
        graph = parse(
            outputs=["y", "z"],
            values={
                "w": {"dtype": "f32", "shape": [4]},
                "y": {"dtype": "f32", "shape": ["T", 4]},
                "z": {"dtype": "f32", "shape": ["T", 4]},
            },
            nodes=[{"op": "add", "ins": ["x", "w"], "outs": ["y", "z"], "attrs": {}}],
        )

        with pytest.raises(OpContractError, match="出力数が 2"):
            assert_op_contracts(graph)

    def _cast_graph(self, attrs: dict, out_dtype: str = "i32") -> dict:
        return {
            "requires": {"ops": ["cast"]},
            "initializers": {},
            "values": {"y": {"dtype": out_dtype, "shape": ["T", 4]}},
            "nodes": [{"op": "cast", "ins": ["x"], "outs": ["y"], "attrs": attrs}],
        }

    def test_cast_declares_its_target_dtype_in_attrs(self):
        graph = parse(**self._cast_graph({"to": "i32"}))

        assert_op_contracts(graph)

    def test_missing_required_attr_is_rejected(self):
        """宣言キーは全て必須（未知キーだけ見る検査ではここが素通りする — ADR 0012）。"""
        graph = parse(**self._cast_graph({}))

        with pytest.raises(OpContractError, match="必須 attr 'to' が無い"):
            assert_op_contracts(graph)

    @pytest.mark.parametrize("value", ["i64", "float32", 3, None])
    def test_attr_value_outside_the_contract_is_rejected(self, value):
        graph = parse(**self._cast_graph({"to": value}))

        with pytest.raises(OpContractError, match="意味論 dtype でない"):
            assert_op_contracts(graph)

    def test_output_dtype_must_agree_with_the_contract(self):
        """attrs.to と values{} の宣言が食い違うグラフは受理しない。"""
        graph = parse(**self._cast_graph({"to": "bool"}, out_dtype="i32"))

        with pytest.raises(OpContractError, match="契約の 'bool' と違う"):
            assert_op_contracts(graph)

    def test_op_that_does_not_accept_the_input_dtype_is_rejected(self):
        graph = parse(
            requires={"ops": ["bitwise_not"]},
            initializers={},
            values={"y": {"dtype": "bool", "shape": ["T", 4]}},
            nodes=[{"op": "bitwise_not", "ins": ["x"], "outs": ["y"], "attrs": {}}],
        )

        with pytest.raises(OpContractError, match="'f32' を実行できない"):
            assert_op_contracts(graph)


class TestSymPrefixSlice:
    """記号 prefix スライス（ADR 0010）— attrs の形と、グラフ文脈が要る 3 つの契約。"""

    def _graph(self, attrs: dict, *, const_shape=(16, 16), out_shape=("T", "T")) -> dict:
        return {
            "requires": {"ops": ["sym_prefix_slice"]},
            "inputs": [{"name": "x", "dtype": "i32", "shape": ["T"]}],
            "outputs": ["y"],
            "initializers": {"table": {"tensor": "enc.table", "storage": {"dtype": "i32"}}},
            "values": {
                "table": {"dtype": "i32", "shape": list(const_shape)},
                "y": {"dtype": "i32", "shape": list(out_shape)},
            },
            "nodes": [{"op": "sym_prefix_slice", "ins": ["table"], "outs": ["y"], "attrs": attrs}],
        }

    def _slices(self, *dims: int) -> list[dict]:
        return [{"dim": dim, "coeff": 1, "offset": 0} for dim in dims]

    def test_a_well_formed_prefix_slice_is_accepted(self):
        graph = parse(**self._graph({"sym": "T", "slices": self._slices(0, 1)}))

        assert_op_contracts(graph)

    def test_an_unbound_symbol_is_rejected(self):
        """graph.symbols に無い sym はランタイムが束縛できず prefix 長が決まらない。"""
        graph = parse(**self._graph({"sym": "U", "slices": self._slices(0, 1)}))

        with pytest.raises(IrError, match=re.escape("graph.symbols")):
            assert_op_contracts(graph)

    def test_a_dim_outside_the_input_rank_is_rejected(self):
        graph = parse(**self._graph({"sym": "T", "slices": self._slices(0, 2)}))

        with pytest.raises(IrError, match="入力 rank 2 の外"):
            assert_op_contracts(graph)

    def test_a_symbolic_input_shape_is_rejected(self):
        """入力は Tmax で焼いた静的形 — 記号だと読み出し stride が実行ごとに縮む。

        initializer 経由なら parse_ir_graph の「記号次元は使えない」で先に落ちるので、
        ここは**記号 shape を持つグラフ入力**を食わせてこの検査だけを踏ませる。
        """
        graph = parse(
            requires={"ops": ["sym_prefix_slice"]},
            inputs=[{"name": "x", "dtype": "i32", "shape": ["T", 4]}],
            outputs=["y"],
            initializers={},
            values={"y": {"dtype": "i32", "shape": ["T", 2]}},
            nodes=[
                {
                    "op": "sym_prefix_slice",
                    "ins": ["x"],
                    "outs": ["y"],
                    "attrs": {"sym": "T", "slices": self._slices(0)},
                }
            ],
        )

        with pytest.raises(IrError, match="記号次元がある"):
            assert_op_contracts(graph)

    def test_duplicate_dims_are_rejected(self):
        graph = parse(**self._graph({"sym": "T", "slices": self._slices(0, 0)}))

        with pytest.raises(OpContractError, match="同じ dim が 2 度ある"):
            assert_op_contracts(graph)

    @pytest.mark.parametrize(
        "slices",
        [
            [{"dim": 0, "coeff": 0, "offset": 0}],
            [{"dim": 0, "coeff": 1, "offset": -1}],
            [{"dim": -1, "coeff": 1, "offset": 0}],
            [{"dim": 0, "coeff": 1}],
            [{"dim": 0, "coeff": 1, "offset": 0, "extra": 1}],
            [],
        ],
        ids=["coeff-0", "negative-offset", "negative-dim", "missing-key", "unknown-key", "empty"],
    )
    def test_malformed_slices_are_rejected(self, slices):
        graph = parse(**self._graph({"sym": "T", "slices": slices}))

        with pytest.raises(OpContractError):
            assert_op_contracts(graph)


class TestSlotDtypes:
    """スロット別 dtype 契約（ADR 0012 の拡張）— gather の src=f32 / index=i32。"""

    def _gather_graph(self, index_dtype: str = "i32", out_dtype: str = "f32") -> dict:
        return {
            "requires": {"ops": ["gather"]},
            "inputs": [
                {"name": "x", "dtype": "f32", "shape": ["T", 4]},
                {"name": "i", "dtype": index_dtype, "shape": ["T", 3]},
            ],
            "initializers": {},
            "values": {"y": {"dtype": out_dtype, "shape": ["T", 3]}},
            "nodes": [{"op": "gather", "ins": ["x", "i"], "outs": ["y"], "attrs": {}}],
        }

    def test_mixed_slot_dtypes_are_accepted(self):
        """スロット別契約は「入力 dtype の混在は拒否」の対象外（それが per-slot の意味）。"""
        graph = parse(**self._gather_graph())

        assert_op_contracts(graph)
        assert_runtime_support(graph)

    def test_index_slot_rejects_f32(self):
        graph = parse(**self._gather_graph(index_dtype="f32"))

        with pytest.raises(OpContractError, match="スロット 1 は"):
            assert_op_contracts(graph)

    def test_swapped_slots_are_rejected(self):
        """src と index を逆に渡した形（和との突合だけでは素通りする）。"""
        graph = parse(
            requires={"ops": ["gather"]},
            inputs=[
                {"name": "x", "dtype": "i32", "shape": ["T", 4]},
                {"name": "i", "dtype": "f32", "shape": ["T", 3]},
            ],
            initializers={},
            values={"y": {"dtype": "i32", "shape": ["T", 3]}},
            nodes=[{"op": "gather", "ins": ["x", "i"], "outs": ["y"], "attrs": {}}],
        )

        with pytest.raises(OpContractError, match="スロット 0 は"):
            assert_op_contracts(graph)

    def test_output_follows_the_value_slot(self):
        graph = parse(**self._gather_graph(out_dtype="i32"))

        with pytest.raises(OpContractError, match="契約の 'f32' と違う"):
            assert_op_contracts(graph)

    def test_uniform_ops_still_reject_mixed_dtypes(self):
        """uniform 契約の混在拒否は温存する（per-slot に潰していない）。"""
        graph = parse(
            requires={"ops": ["mul"]},
            inputs=[
                {"name": "x", "dtype": "f32", "shape": ["T", 4]},
                {"name": "i", "dtype": "i32", "shape": ["T", 4]},
            ],
            initializers={},
            values={"y": {"dtype": "f32", "shape": ["T", 4]}},
            nodes=[{"op": "mul", "ins": ["x", "i"], "outs": ["y"], "attrs": {}}],
        )

        with pytest.raises(OpContractError, match="入力 dtype が混在"):
            assert_op_contracts(graph)


class TestContainer:
    """配布形（safetensors）と宣言の突合。"""

    def test_missing_metadata_key_is_rejected(self, tmp_path):

        path = write_container(
            tmp_path / "m.safetensors", base_graph(), {"enc.w": torch.ones(4)}, key="other_ir"
        )

        with pytest.raises(ContainerError, match="Karume モデルではない"):
            verify_model(path)

    def test_missing_tensor_is_rejected(self, tmp_path):

        path = write_container(tmp_path / "m.safetensors", base_graph(), {"other": torch.ones(4)})

        with pytest.raises(ContainerError, match="がファイルに無い"):
            verify_model(path)

    def test_shape_mismatch_is_rejected(self, tmp_path):

        path = write_container(tmp_path / "m.safetensors", base_graph(), {"enc.w": torch.ones(5)})

        with pytest.raises(ContainerError, match="宣言 shape"):
            verify_model(path)

    def test_storage_dtype_mismatch_is_rejected(self, tmp_path):

        path = write_container(
            tmp_path / "m.safetensors", base_graph(), {"enc.w": torch.ones(4, dtype=torch.float16)}
        )

        with pytest.raises(ContainerError, match="格納 dtype"):
            verify_model(path)


def write_raw_container(path, header: dict, data: bytes = b""):
    """ヘッダ JSON を直に組んだ safetensors（`save_file` では作れない不正形の故障注入用）。

    ヘッダ長は 8 の倍数へ空白で詰める（`emit._HEADER_ALIGN` と同じ規約）。詰めないと
    データ節先頭が 4 バイト境界から外れ、整列検査が先に落ちて**添字の域の検査を踏めない**。
    """
    blob = json.dumps(header).encode("utf-8")
    blob += b" " * (-len(blob) % 8)
    path.write_bytes(len(blob).to_bytes(8, "little") + blob + data)
    return path


class TestReaderIndexRange:
    """ヘッダの添字は TS リーダの `asIndex`（非負・2^53−1 以下）と同じ受理集合。

    `packages/runtime/src/format/safetensors.ts` が拒否する値を verify が受理すると、
    「エクスポータは緑・ブラウザのリーダだけ落ちる」ファイルが配布形として残る。要素数の積
    だけを見る検査は、負の次元も安全整数超も**積が合ってしまう**形で素通りする。
    """

    def test_a_negative_dimension_is_rejected(self, tmp_path):
        """`[0,-1]` は積 0 でバイト長 0 と一致するので、積だけの検査では素通りする。"""
        header = {"t": {"dtype": "F32", "shape": [0, -1], "data_offsets": [0, 0]}}

        with pytest.raises(ContainerError, match="非負整数"):
            assert_reader_layout(write_raw_container(tmp_path / "m.safetensors", header))

    def test_a_dimension_beyond_the_safe_integer_range_is_rejected(self, tmp_path):
        header = {"t": {"dtype": "F32", "shape": [0, 2**53], "data_offsets": [0, 0]}}

        with pytest.raises(ContainerError, match="非負整数"):
            assert_reader_layout(write_raw_container(tmp_path / "m.safetensors", header))

    def test_an_element_count_beyond_the_safe_integer_range_is_rejected(self, tmp_path):
        """各次元は安全整数でも**積**が越える形（TS の elementCount と同じ段階で見る）。"""
        header = {"t": {"dtype": "F32", "shape": [2**27, 2**27], "data_offsets": [0, 0]}}

        with pytest.raises(ContainerError, match="要素数が安全整数"):
            assert_reader_layout(write_raw_container(tmp_path / "m.safetensors", header))

    def test_a_data_offset_beyond_the_safe_integer_range_is_rejected(self, tmp_path):
        """サイズ不一致より前に添字の域で落ちる（TS は asIndex を先に通す）。"""
        header = {"t": {"dtype": "F32", "shape": [2], "data_offsets": [0, 2**53]}}

        with pytest.raises(ContainerError, match="非負整数"):
            assert_reader_layout(write_raw_container(tmp_path / "m.safetensors", header))


class TestReaderEntryStructure:
    """ヘッダ 1 項目の構造も門の診断で落とす（素の添字は KeyError / TypeError で漏れる）。

    エクスポータが書いたファイルでは到達しない（`_save_ordered` が必ず 3 キーを書く）が、
    `karume verify` は外部で作られた safetensors も食う公開 CLI なので経路は実在する。
    診断の主語が「不正なファイル」から「エクスポータが壊れた」に化けるのを防ぐ。
    """

    @pytest.mark.parametrize(
        ("entry", "why"),
        [
            ({"shape": [2], "data_offsets": [0, 8]}, r"\['dtype'\] が無い"),
            ({"dtype": "F32", "data_offsets": [0, 8]}, r"\['shape'\] が無い"),
            ({"dtype": "F32", "shape": [2]}, r"\['data_offsets'\] が無い"),
            ("F32", "オブジェクトでない"),
            ({"dtype": "F32", "shape": 2, "data_offsets": [0, 8]}, "shape が配列でない"),
        ],
        ids=["no-dtype", "no-shape", "no-offsets", "not-an-object", "shape-not-a-list"],
    )
    def test_a_malformed_entry_is_reported_as_a_container_error(self, tmp_path, entry, why):
        path = write_raw_container(tmp_path / "m.safetensors", {"t": entry}, b"\0" * 8)

        with pytest.raises(ContainerError, match=why):
            assert_reader_layout(path)


class TestReaderPackedFourBit:
    """packed 4bit（ADR 0069 決定 2）— サイズは bit 単位・整列はテンソル**先頭** 4 バイト。

    `safetensors` ライブラリは `I4` を知らない（0.8.0 の dtype 語彙に無い）ので、ヘッダ JSON を
    直に読む `assert_reader_layout` が I4 を踏める唯一の門。TS 側リーダ
    （`packages/runtime/src/format/safetensors.ts`）の受理集合と 1 対 1 に保つ。
    """

    def test_a_packed_tensor_is_half_its_element_count(self, tmp_path):
        header = {"w": {"dtype": "I4", "shape": [3, 32], "data_offsets": [0, 48]}}

        assert_reader_layout(write_raw_container(tmp_path / "m.safetensors", header, b"\0" * 48))

    def test_an_odd_element_count_is_rejected(self, tmp_path):
        """bit 総量が byte 境界に乗らない形（末尾要素が半バイトだけ突き出す）。"""
        header = {"w": {"dtype": "I4", "shape": [3], "data_offsets": [0, 2]}}

        with pytest.raises(ContainerError, match="byte 境界に乗らない"):
            assert_reader_layout(write_raw_container(tmp_path / "m.safetensors", header, b"\0" * 2))

    def test_a_size_that_is_not_half_the_element_count_is_rejected(self, tmp_path):
        header = {"w": {"dtype": "I4", "shape": [3, 32], "data_offsets": [0, 96]}}

        with pytest.raises(ContainerError, match="サイズ不一致"):
            assert_reader_layout(
                write_raw_container(tmp_path / "m.safetensors", header, b"\0" * 96)
            )

    def test_a_tensor_start_off_the_four_byte_boundary_is_rejected(self, tmp_path):
        """I4 は要素整列（0.5 バイト）ではなく u32 束縛のための先頭 4 バイト整列を要求する。"""
        header = {
            "a": {"dtype": "I8", "shape": [2], "data_offsets": [0, 2]},
            "w": {"dtype": "I4", "shape": [4], "data_offsets": [2, 4]},
        }

        with pytest.raises(ContainerError, match="整列していない"):
            assert_reader_layout(write_raw_container(tmp_path / "m.safetensors", header, b"\0" * 4))


class TestHeaderLengthBound:
    """ヘッダ長 u64 はファイル実長で拘束してから read へ渡す（`dist.safetensors_header` と同型）。

    拘束しないと巨大値が `read` へ抜けて OverflowError / MemoryError になり、規則違反が
    門の例外（ContainerError）として出てこない。
    """

    @pytest.mark.parametrize("header_length", [2**64 - 1, 2**40], ids=["u64-max", "1TiB"])
    def test_an_oversized_header_length_is_reported_as_a_container_error(
        self, tmp_path, header_length
    ):
        path = tmp_path / "m.safetensors"
        path.write_bytes(header_length.to_bytes(8, "little") + b'{"__metadata__":{}}')

        with pytest.raises(ContainerError, match="ファイル長"):
            assert_reader_layout(path)

    def test_a_zero_header_length_is_rejected(self, tmp_path):
        path = tmp_path / "m.safetensors"
        path.write_bytes((0).to_bytes(8, "little"))

        with pytest.raises(ContainerError, match="ファイル長"):
            assert_reader_layout(path)


def i8_graph(scale_key: str = "enc.s") -> dict:
    """linear の重みを i8 + companion scale で格納する最小グラフ（ADR 0019）。"""
    graph = base_graph()
    graph["requires"] = {"ops": ["linear"]}
    graph["inputs"] = [{"name": "x", "dtype": "f32", "shape": ["T", 4]}]
    graph["initializers"] = {
        "w": {"tensor": "enc.w", "storage": {"dtype": "i8", "scale": scale_key}},
        "b": {"tensor": "enc.b", "storage": {"dtype": "f32"}},
    }
    graph["values"] = {
        "w": {"dtype": "f32", "shape": [3, 4]},
        "b": {"dtype": "f32", "shape": [3]},
        "y": {"dtype": "f32", "shape": ["T", 3]},
    }
    graph["nodes"] = [{"op": "linear", "ins": ["x", "w", "b"], "outs": ["y"], "attrs": {}}]
    return graph


def i8_tensors(**overrides: torch.Tensor) -> dict[str, torch.Tensor]:
    tensors = {
        "enc.w": torch.ones(3, 4, dtype=torch.int8),
        "enc.b": torch.ones(3),
        "enc.s": torch.ones(3, 1),
    }
    tensors.update(overrides)
    return tensors


class TestQuantizedScaleTensor:
    """companion scale の突合（`packages/runtime/src/format/container.ts` の鏡像）。

    scale は IR の値ではなく safetensors の**素のテンソル**なので、ここを緩めると誤りが
    ランタイムのロード時まで出ない — 「書けたが読めない」を配布物に残さないのが verify_model。
    """

    def test_a_well_formed_scale_is_accepted(self, tmp_path):
        path = write_container(tmp_path / "m.safetensors", i8_graph(), i8_tensors())

        assert verify_model(path).initializers["w"].storage.scale == "enc.s"

    def test_a_missing_scale_tensor_is_rejected(self, tmp_path):
        path = write_container(tmp_path / "m.safetensors", i8_graph("enc.missing"), i8_tensors())

        with pytest.raises(ContainerError, match="がファイルに無い"):
            verify_model(path)

    def test_a_non_f32_scale_is_rejected(self, tmp_path):
        """scale を別 dtype のビット列として読むと全チャネルが桁違いの値になる。"""
        path = write_container(
            tmp_path / "m.safetensors",
            i8_graph(),
            i8_tensors(**{"enc.s": torch.ones(3, 1, dtype=torch.float16)}),
        )

        with pytest.raises(ContainerError, match="F32 が必要"):
            verify_model(path)

    def test_a_scale_with_a_different_rank_is_rejected(self, tmp_path):
        path = write_container(
            tmp_path / "m.safetensors", i8_graph(), i8_tensors(**{"enc.s": torch.ones(3)})
        )

        with pytest.raises(ContainerError, match="rank"):
            verify_model(path)

    def test_a_non_broadcastable_scale_is_rejected(self, tmp_path):
        path = write_container(
            tmp_path / "m.safetensors", i8_graph(), i8_tensors(**{"enc.s": torch.ones(2, 1)})
        )

        with pytest.raises(ContainerError, match="broadcast できない"):
            verify_model(path)

    def test_a_broadcastable_but_wrong_axis_scale_is_rejected_for_an_eligible_weight(
        self, tmp_path
    ):
        """`[1,4]` は broadcast できるが、linear のチャネル軸は 0 なので受理しない。

        適格重み（圧縮のまま GPU 常駐）のカーネルは scale を平坦に `wscale[出力チャネル]`
        としか読まないので、この形は**沈黙誤値**になる — 実行側の
        `packages/runtime/src/runtime/executor.ts` assertChannelScale と同じ受理集合を
        ここでも張る。
        """
        path = write_container(
            tmp_path / "m.safetensors", i8_graph(), i8_tensors(**{"enc.s": torch.ones(1, 4)})
        )

        with pytest.raises(ContainerError, match="チャネル軸 0"):
            verify_model(path)

    def test_a_scale_key_colliding_with_a_real_tensor_is_rejected(self, tmp_path):
        """別の initializer の実体を scale として読むと、ロードは通って値だけが壊れる。"""
        path = write_container(tmp_path / "m.safetensors", i8_graph("enc.b"), i8_tensors())

        with pytest.raises(ContainerError, match="の実体と同じキー"):
            verify_model(path)


class _StubSafetensors:
    """`_assert_scale_tensor` が触る面（get_slice → get_dtype / get_shape）だけのスタブ。

    MUST: ここだけ実ファイルを使わない。`safetensors` ライブラリは `I4` を知らない
    （0.8.0 の dtype 語彙に無い）ので、group 形 scale を持つ配布形は `safe_open` の時点で
    開けず、この規則を実ファイル経由では踏めない。
    """

    class _Slice:
        def __init__(self, dtype: str, shape: tuple[int, ...]) -> None:
            self._dtype = dtype
            self._shape = shape

        def get_dtype(self) -> str:
            return self._dtype

        def get_shape(self) -> tuple[int, ...]:
            return self._shape

    def __init__(self, tensors: dict[str, tuple[str, tuple[int, ...]]]) -> None:
        self._tensors = tensors

    def get_slice(self, key: str) -> _Slice:
        dtype, shape = self._tensors[key]
        return self._Slice(dtype, shape)


class TestGroupScaleTensor:
    """group 形 scale（ADR 0069 決定 3）— 重みと同 rank・**最終次元だけ group 数**。

    per-channel の keepdim broadcast 形とは受理集合が交わらない別分岐なので、重みは
    `[4,64]` / group_size 32（= group 形 `[4,2]`）で見る。形の分岐が効いていないと、
    keepdim 形 `[4,1]` が「broadcast できる」として素通りし、group ごとの scale が
    1 チャネル 1 値として読まれる沈黙誤値になる。
    """

    def _check(
        self,
        scale_dtype: str = "F32",
        scale_shape: tuple[int, ...] = (4, 2),
        group_size: int = 32,
    ) -> None:
        graph = parse(
            initializers={
                "w": {
                    "tensor": "enc.w",
                    "storage": {"dtype": "i4", "scale": "enc.s", "group_size": group_size},
                }
            },
            values={
                "w": {"dtype": "f32", "shape": [4, 64]},
                "y": {"dtype": "f32", "shape": ["T", 4]},
            },
        )
        _assert_scale_tensor(
            _StubSafetensors({"enc.s": (scale_dtype, scale_shape)}),
            graph,
            {"enc.w", "enc.s"},
            "w",
            "enc.s",
            [4, 64],
            None,
            group_size,
        )

    def test_a_well_formed_group_scale_is_accepted(self):
        self._check()

    def test_a_keepdim_broadcast_scale_is_rejected(self):
        with pytest.raises(ContainerError, match="group 形"):
            self._check(scale_shape=(4, 1))

    def test_a_scale_with_a_different_rank_is_rejected(self):
        with pytest.raises(ContainerError, match="rank"):
            self._check(scale_shape=(8,))

    def test_a_non_f32_scale_is_rejected(self):
        with pytest.raises(ContainerError, match="F32 が必要"):
            self._check(scale_dtype="F16")


def conv_i8_graph(op: str) -> dict:
    """conv 系の重みを i8 で格納する最小グラフ（重みは `enc.w` / scale は `enc.s`）。

    重みは conv1d が `[Cout, Cin/groups, K] = [3,2,2]`、conv_transpose1d が
    `[Cin, Cout, K] = [2,3,2]`（転置レイアウト）。どちらも Cin ≠ Cout の非対称形なので、
    軸を取り違えた scale が「たまたま broadcast できる」形として残る。
    """
    weight_shape = [3, 2, 2] if op == "conv1d" else [2, 3, 2]
    out_length = 3 if op == "conv1d" else 8
    attrs = (
        {"stride": 1, "padding": 0, "dilation": 1, "groups": 1}
        if op == "conv1d"
        else {"stride": 2, "padding": 0}
    )
    return {
        "format": "karume-ir",
        "version": 1,
        "requires": {"ops": [op]},
        "symbols": [],
        "inputs": [{"name": "x", "dtype": "f32", "shape": [1, 2, 4]}],
        "outputs": ["y"],
        "initializers": {
            "w": {"tensor": "enc.w", "storage": {"dtype": "i8", "scale": "enc.s"}},
            "b": {"tensor": "enc.b", "storage": {"dtype": "f32"}},
        },
        "values": {
            "w": {"dtype": "f32", "shape": weight_shape},
            "b": {"dtype": "f32", "shape": [3]},
            "y": {"dtype": "f32", "shape": [1, 3, out_length]},
        },
        "nodes": [{"op": op, "ins": ["x", "w", "b"], "outs": ["y"], "attrs": attrs}],
    }


def conv_i8_tensors(op: str, scale: torch.Tensor) -> dict[str, torch.Tensor]:
    weight_shape = (3, 2, 2) if op == "conv1d" else (2, 3, 2)
    return {
        "enc.w": torch.ones(weight_shape, dtype=torch.int8),
        "enc.b": torch.ones(3),
        "enc.s": scale,
    }


class TestQuantizedScaleChannelAxis:
    """適格重みの scale はチャネル軸ちょうどの keepdim 形だけ
    （`packages/runtime/src/runtime/executor.ts` の assertChannelScale の鏡像）。

    軸は消費側 op から引く（`packages/runtime/src/ops.ts` の `WEIGHT_CHANNEL_AXES` —
    conv_transpose1d だけ転置レイアウトで 1）。重みの shape だけでは決まらないので、
    形の検査だけでは linear の `[1,4]` と conv_transpose1d の `[1,Cout,1]` を区別できない。
    """

    def test_conv1d_takes_the_first_axis(self, tmp_path):
        path = write_container(
            tmp_path / "m.safetensors",
            conv_i8_graph("conv1d"),
            conv_i8_tensors("conv1d", torch.ones(3, 1, 1)),
        )

        assert verify_model(path).initializers["w"].storage.scale == "enc.s"

    def test_conv_transpose1d_takes_the_second_axis(self, tmp_path):
        path = write_container(
            tmp_path / "m.safetensors",
            conv_i8_graph("conv_transpose1d"),
            conv_i8_tensors("conv_transpose1d", torch.ones(1, 3, 1)),
        )

        assert verify_model(path).initializers["w"].storage.scale == "enc.s"

    def test_conv_transpose1d_rejects_the_first_axis(self, tmp_path):
        """`[2,1,1]` は重み `[2,3,2]` へ broadcast できる — 軸を見なければ素通りする形。"""
        path = write_container(
            tmp_path / "m.safetensors",
            conv_i8_graph("conv_transpose1d"),
            conv_i8_tensors("conv_transpose1d", torch.ones(2, 1, 1)),
        )

        with pytest.raises(ContainerError, match="チャネル軸 1"):
            verify_model(path)

    def test_an_ineligible_weight_keeps_the_general_broadcast_rule(self, tmp_path):
        """適格外（重みスロット以外の消費）はホストで f32 展開されるので軸の概念が無い。

        展開は `packages/runtime/src/format/i8.ts` の decodeI8 で、keepdim broadcast を
        stride で引く — 軸を要求すると**ランタイムが読めるファイルを verify が落とす**。
        """
        graph = {
            "format": "karume-ir",
            "version": 1,
            "requires": {"ops": ["mul"]},
            "symbols": [],
            "inputs": [{"name": "x", "dtype": "f32", "shape": [3, 4]}],
            "outputs": ["y"],
            "initializers": {
                "w": {"tensor": "enc.w", "storage": {"dtype": "i8", "scale": "enc.s"}}
            },
            "values": {
                "w": {"dtype": "f32", "shape": [3, 4]},
                "y": {"dtype": "f32", "shape": [3, 4]},
            },
            "nodes": [{"op": "mul", "ins": ["x", "w"], "outs": ["y"], "attrs": {}}],
        }
        path = write_container(
            tmp_path / "m.safetensors",
            graph,
            {"enc.w": torch.ones(3, 4, dtype=torch.int8), "enc.s": torch.ones(1, 4)},
        )

        assert verify_model(path).initializers["w"].storage.scale == "enc.s"


class TestStridedRank:
    """strided コピー族の rank 上限（ADR 0011）—
    packages/runtime/src/ops.ts の assertStridedRank と同義。

    束縛に依らず宣言 shape の長さだけで決まるので、記号次元があってもここで見られる。
    """

    def _permute_graph(self, rank: int) -> dict:
        source = ["T", *([2] * (rank - 1))]
        target = [*([2] * (rank - 1)), "T"]
        return {
            "requires": {"ops": ["permute"]},
            "inputs": [{"name": "x", "dtype": "f32", "shape": source}],
            "initializers": {},
            "values": {"y": {"dtype": "f32", "shape": target}},
            "nodes": [
                {
                    "op": "permute",
                    "ins": ["x"],
                    "outs": ["y"],
                    "attrs": {"dims": [*range(1, rank), 0]},
                }
            ],
        }

    def test_rank_at_the_limit_is_accepted(self):
        assert_op_contracts(parse(**self._permute_graph(4)))

    def test_rank_beyond_the_limit_is_rejected(self):
        with pytest.raises(OpContractError, match="strided カーネルの上限"):
            assert_op_contracts(parse(**self._permute_graph(5)))

    def test_a_scalar_mask_is_rejected(self):
        """masked_fill の mask は右詰め broadcast の stride を組むので rank 1 以上が要る。"""
        graph = parse(
            requires={"ops": ["masked_fill"]},
            inputs=[
                {"name": "x", "dtype": "f32", "shape": ["T", 4]},
                {"name": "m", "dtype": "bool", "shape": []},
            ],
            initializers={},
            values={"y": {"dtype": "f32", "shape": ["T", 4]}},
            nodes=[
                {"op": "masked_fill", "ins": ["x", "m"], "outs": ["y"], "attrs": {"value": 0.0}}
            ],
        )

        with pytest.raises(OpContractError, match="strided カーネルの上限"):
            assert_op_contracts(graph)
