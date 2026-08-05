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
from karume.ops import IO_DTYPES, SEMANTIC_DTYPES, OpContractError
from karume.verify import (
    ContainerError,
    IrError,
    assert_op_contracts,
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

    def test_symbol_never_appearing_bare_in_an_input_is_rejected(self):
        """束縛は入力 shape の次元位置から取る — 素の形が無いと束縛できない。"""
        with pytest.raises(IrError, match="で現れない"):
            parse(
                inputs=[{"name": "x", "dtype": "f32", "shape": ["2T", 4]}],
                values={
                    "w": {"dtype": "f32", "shape": [4]},
                    "y": {"dtype": "f32", "shape": ["2T", 4]},
                },
            )

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

    def test_empty_outs_is_rejected(self):
        with pytest.raises(IrError, match="outs が空"):
            parse(nodes=[{"op": "add", "ins": ["x", "w"], "outs": [], "attrs": {}}])


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
            parse(initializers={"w": {"tensor": "enc.w", "storage": {"dtype": "i4"}}})


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
        転送する）ので、軸自体は残す — 受理集合が IO_DTYPES から導かれることを固定する。
        """
        assert IO_DTYPES == SEMANTIC_DTYPES

        graph = parse(
            inputs=[
                {"name": "x", "dtype": "f32", "shape": ["T", 4]},
                {"name": "ids", "dtype": "i32", "shape": ["T"]},
                {"name": "mask", "dtype": "bool", "shape": ["T"]},
            ]
        )

        assert_runtime_support(graph)

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

    def test_unsupported_storage_dtype_is_reported(self):
        graph = parse(initializers={"w": {"tensor": "enc.w", "storage": {"dtype": "bf16"}}})

        with pytest.raises(ContainerError, match="非対応 格納 dtype"):
            assert_runtime_support(graph)

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

    def test_a_broadcastable_but_wrong_axis_scale_is_accepted_here(self, tmp_path):
        """`[1,4]` は broadcast できるので**この層は通す**（TS 側 container.ts と同じ切り方）。

        カーネルは scale を平坦に `wscale[出力チャネル]` としか読まないので、この形は
        沈黙誤値になりうる。門は 2 段に分かれていて、書き出し側は `emit._store_i8`
        （軸を知っている）、実行側は `packages/runtime/src/runtime/executor.ts` の
        assertChannelScale が持つ。
        ここが「読める形か」しか見ないことを、逆に固定しておく。
        """
        path = write_container(
            tmp_path / "m.safetensors", i8_graph(), i8_tensors(**{"enc.s": torch.ones(1, 4)})
        )

        assert verify_model(path).initializers["w"].storage.scale == "enc.s"

    def test_a_scale_key_colliding_with_a_real_tensor_is_rejected(self, tmp_path):
        """別の initializer の実体を scale として読むと、ロードは通って値だけが壊れる。"""
        path = write_container(tmp_path / "m.safetensors", i8_graph("enc.b"), i8_tensors())

        with pytest.raises(ContainerError, match="の実体と同じキー"):
            verify_model(path)


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
