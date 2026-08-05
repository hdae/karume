"""IR v1 の受理規則を Python 側でも全部見る（docs/ir-v1.md）。

TS 側の正本は packages/runtime/src/format/ir.ts（グラフ単体の規則）・
packages/runtime/src/format/container.ts（配布形との
突合とランタイム対応表）。エクスポータが「書けるがランタイムが読めない」ファイルを
出さないよう、書き出し経路の最後で同じ規則を通す。

MUST: ここは fail loudly の門であって近似の場ではない — 未知キーも非正準表記も
黙って無視せず、必ず例外にする（未リリースにつき前方互換チャネルは持たない）。

    uv run karume verify ../../models/anima-turbo/transformer/model.f16.safetensors
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from safetensors import safe_open

from karume.dims import MAX_SAFE_INT, is_symbol_name, parse_dim, try_parse_dim
from karume.ir import (
    IR_FORMAT,
    IR_METADATA_KEY,
    IR_VERSION,
    IrGraph,
    IrInitializer,
    IrInput,
    IrNode,
    IrStorage,
    IrValue,
)
from karume.ops import (
    IO_DTYPES,
    M0_STORAGE_DTYPES,
    OP_CONTRACTS,
    STRIDED_RANK_OPS,
    assert_node_contract,
    assert_strided_rank,
    resolve_node_dtypes,
    sym_prefix_slice_attrs,
)
from karume.shapes import assert_graph_shapes, declared_shape


class IrError(ValueError):
    """グラフ JSON 単体で決まる規則の違反。"""


class ContainerError(ValueError):
    """配布形（safetensors + 埋め込みグラフ）の結合規則、または capability 不足。"""


TOP_LEVEL_KEYS = (
    "format",
    "version",
    "requires",
    "symbols",
    "inputs",
    "outputs",
    "initializers",
    "values",
    "nodes",
)

SEMANTIC_DTYPES = ("f32", "i32", "bool")
STORAGE_DTYPES = ("f32", "f16", "bf16", "i8", "i32")

#: 格納 dtype → safetensors dtype。f32/f16/bf16/i8 は意味論 f32 の符号化で、`i32` だけが
#: 生の int32（ADR 0010 の明示的な例外）。
STORAGE_ENCODING = {"f32": "F32", "f16": "F16", "bf16": "BF16", "i8": "I8", "i32": "I32"}

#: initializer の意味論 dtype → 許される格納 dtype（docs/ir-v1.md「値と型」）。
#: MUST: 交差を許さない — `i32` 宣言の initializer が f16 のビット列として読まれる
#: 沈黙誤値になる。bool の initializer は語彙に無い。
INITIALIZER_STORAGE = {"f32": ("f32", "f16", "bf16", "i8"), "i32": ("i32",)}


# ---- JSON 層 --------------------------------------------------------------


def parse_graph_json(text: str) -> Any:
    """グラフ JSON を読む。非有限値を含むものは受理しない。

    Python の json は `Infinity` / `NaN` リテラルを既定で受理するが、ブラウザの
    JSON.parse は落ちる。加えて `1e999` は構文として有効なまま Infinity へ丸まるので、
    リテラル名だけでなく**値レベル**で弾く（docs/ir-v1.md）。
    """

    def reject_constant(literal: str) -> float:
        raise IrError(f"グラフ JSON に非標準リテラル: {literal}")

    def parse_float(raw: str) -> float:
        value = float(raw)
        if not math.isfinite(value):
            raise IrError(f"グラフ JSON の数値 '{raw}' が有限でない")
        return value

    try:
        return json.loads(text, parse_constant=reject_constant, parse_float=parse_float)
    except IrError:
        raise
    except ValueError as cause:
        raise IrError(f"グラフ JSON を解析できない: {cause}") from cause


# ---- 構造ヘルパ -----------------------------------------------------------


def _as_object(value: Any, where: str) -> dict:
    if not isinstance(value, dict):
        raise IrError(f"{where}: オブジェクトでない")
    return value


def _as_array(value: Any, where: str) -> list:
    if not isinstance(value, list):
        raise IrError(f"{where}: 配列でない")
    return value


def _as_nonempty_str(value: Any, where: str) -> str:
    if not isinstance(value, str) or value == "":
        raise IrError(f"{where}: 空でない文字列でない")
    return value


def _as_unique_strings(value: Any, where: str) -> list[str]:
    """重複を許さない文字列配列（集合として扱う欄）。"""
    items = [
        _as_nonempty_str(item, f"{where}[{index}]")
        for index, item in enumerate(_as_array(value, where))
    ]
    seen: set[str] = set()
    for item in items:
        if item in seen:
            raise IrError(f"{where}: '{item}' が重複している")
        seen.add(item)
    return items


def _check_keys(
    obj: Mapping[str, Any], required: Sequence[str], optional: Sequence[str], where: str
) -> None:
    for key in required:
        if key not in obj:
            raise IrError(f"{where}: 必須キー '{key}' が無い")
    for key in obj:
        if key not in required and key not in optional:
            raise IrError(f"{where}: 未知のキー '{key}'")


def _as_semantic_dtype(value: Any, where: str) -> str:
    dtype = _as_nonempty_str(value, where)
    if dtype not in SEMANTIC_DTYPES:
        raise IrError(f"{where}: 意味論 dtype '{dtype}' は語彙外（{' / '.join(SEMANTIC_DTYPES)}）")
    return dtype


def _as_storage_dtype(value: Any, where: str) -> str:
    dtype = _as_nonempty_str(value, where)
    if dtype not in STORAGE_DTYPES:
        raise IrError(f"{where}: 格納 dtype '{dtype}' は語彙外（{' / '.join(STORAGE_DTYPES)}）")
    return dtype


def _parse_shape(value: Any, symbols: set[str], where: str) -> list[int | str]:
    shape: list[int | str] = []
    for index, dim in enumerate(_as_array(value, where)):
        at = f"{where}[{index}]"
        # bool は int の派生だが次元ではない。
        if isinstance(dim, bool):
            raise IrError(f"{at}: 次元が数値でも文字列でもない")
        if isinstance(dim, int):
            if dim < 0 or dim > MAX_SAFE_INT:
                raise IrError(f"{at}: 次元 {dim} が非負整数でない")
            shape.append(dim)
            continue
        if not isinstance(dim, str):
            raise IrError(f"{at}: 次元が数値でも文字列でもない")
        expr = try_parse_dim(dim)
        if expr is None:
            raise IrError(f"{at}: 次元式 '{dim}' が正準文法 coeff·sym+offset に適合しない")
        if expr.sym not in symbols:
            raise IrError(f"{at}: シンボル '{expr.sym}' が graph.symbols で宣言されていない")
        shape.append(dim)
    return shape


def _parse_storage(value: Any, where: str) -> IrStorage:
    obj = _as_object(value, where)
    _check_keys(obj, ["dtype"], ["scale", "group_size"], where)
    dtype = _as_storage_dtype(obj["dtype"], f"{where}.dtype")
    has_scale = "scale" in obj
    has_group_size = "group_size" in obj
    # scale / group_size は量子化格納の記述子。非量子化 dtype に付いているのは
    # エクスポータの取り違えなので受理しない（黙って無視すると格納の意味が二重化する）。
    if dtype != "i8" and (has_scale or has_group_size):
        raise IrError(f"{where}: 格納 dtype '{dtype}' に scale / group_size は付けられない")
    # MUST: i8 は scale を**明示宣言**する
    # （ADR 0019・TS 側 packages/runtime/src/format/ir.ts の鏡像）。
    # 既定 1.0 で補完すると、scale の書き忘れが「全チャネル 1.0 で dequant した重み」に化けて
    # ロードも実行も通ってしまう（差が O(scale) で出るのに、どこにも例外が出ない）。
    if dtype == "i8" and not has_scale:
        raise IrError(f"{where}: 格納 dtype 'i8' には scale（scale テンソルのキー）が要る")
    scale = _as_nonempty_str(obj["scale"], f"{where}.scale") if has_scale else None
    group_size = None
    if has_group_size:
        raw = obj["group_size"]
        if isinstance(raw, bool) or not isinstance(raw, int) or raw < 1:
            raise IrError(f"{where}.group_size: 正整数でない")
        group_size = raw
    return IrStorage(dtype=dtype, scale=scale, group_size=group_size)


# ---- グラフ単体の規則 -----------------------------------------------------


def parse_ir_graph(text: str) -> IrGraph:
    """グラフ JSON を検証しつつ読む（packages/runtime/src/format/ir.ts parseIrGraph と同義）。"""
    root = _as_object(parse_graph_json(text), "graph")
    _check_keys(root, TOP_LEVEL_KEYS, [], "graph")

    if root["format"] != IR_FORMAT:
        raise IrError(f"graph.format が '{IR_FORMAT}' でない: {root['format']!r}")
    if root["version"] != IR_VERSION or isinstance(root["version"], bool):
        raise IrError(f"graph.version が {IR_VERSION} でない: {root['version']!r}")

    requires = _as_object(root["requires"], "graph.requires")
    _check_keys(requires, ["ops"], [], "graph.requires")
    required_ops = _as_unique_strings(requires["ops"], "graph.requires.ops")

    symbols = _as_unique_strings(root["symbols"], "graph.symbols")
    for symbol in symbols:
        if not is_symbol_name(symbol):
            raise IrError(f"graph.symbols: シンボル名 '{symbol}' が不正")
    symbol_set = set(symbols)

    inputs: list[IrInput] = []
    for index, raw in enumerate(_as_array(root["inputs"], "graph.inputs")):
        where = f"graph.inputs[{index}]"
        obj = _as_object(raw, where)
        _check_keys(obj, ["name", "dtype", "shape"], [], where)
        inputs.append(
            IrInput(
                name=_as_nonempty_str(obj["name"], f"{where}.name"),
                dtype=_as_semantic_dtype(obj["dtype"], f"{where}.dtype"),
                shape=_parse_shape(obj["shape"], symbol_set, f"{where}.shape"),
            )
        )

    outputs = _as_unique_strings(root["outputs"], "graph.outputs")

    values: dict[str, IrValue] = {}
    for name, raw in _as_object(root["values"], "graph.values").items():
        where = f"graph.values['{name}']"
        obj = _as_object(raw, where)
        _check_keys(obj, ["dtype", "shape"], [], where)
        values[name] = IrValue(
            dtype=_as_semantic_dtype(obj["dtype"], f"{where}.dtype"),
            shape=_parse_shape(obj["shape"], symbol_set, f"{where}.shape"),
        )

    initializers: dict[str, IrInitializer] = {}
    for name, raw in _as_object(root["initializers"], "graph.initializers").items():
        where = f"graph.initializers['{name}']"
        obj = _as_object(raw, where)
        _check_keys(obj, ["tensor", "storage"], [], where)
        initializers[name] = IrInitializer(
            tensor=_as_nonempty_str(obj["tensor"], f"{where}.tensor"),
            storage=_parse_storage(obj["storage"], f"{where}.storage"),
        )

    nodes: list[IrNode] = []
    for index, raw in enumerate(_as_array(root["nodes"], "graph.nodes")):
        where = f"graph.nodes[{index}]"
        obj = _as_object(raw, where)
        _check_keys(obj, ["op", "ins", "outs", "attrs"], [], where)
        outs = [
            _as_nonempty_str(out, f"{where}.outs[{i}]")
            for i, out in enumerate(_as_array(obj["outs"], f"{where}.outs"))
        ]
        if not outs:
            raise IrError(f"{where}: outs が空（値を定義しないノードは静的 DAG に置けない）")
        nodes.append(
            IrNode(
                op=_as_nonempty_str(obj["op"], f"{where}.op"),
                ins=[
                    _as_nonempty_str(item, f"{where}.ins[{i}]")
                    for i, item in enumerate(_as_array(obj["ins"], f"{where}.ins"))
                ],
                outs=outs,
                attrs=_as_object(obj["attrs"], f"{where}.attrs"),
            )
        )

    _check_symbol_bindability(symbols, inputs)
    defined = _check_definitions(inputs, initializers, nodes, outputs)
    _check_declarations(inputs, initializers, values, nodes, defined)
    _check_required_ops(required_ops, nodes)

    return IrGraph(
        symbols=symbols,
        inputs=inputs,
        outputs=outputs,
        initializers=initializers,
        values=values,
        nodes=nodes,
    )


def _check_symbol_bindability(symbols: Sequence[str], inputs: Sequence[IrInput]) -> None:
    """束縛は入力 shape の次元位置から直接取る（要素数からの逆算はしない）ため、
    宣言されたシンボルは少なくとも 1 つの入力 shape に素の形で現れなければならない。
    """
    bindable = {
        expr.sym
        for spec in inputs
        for dim in spec.shape
        if isinstance(dim, str)
        for expr in [parse_dim(dim)]
        if expr.coeff == 1 and expr.offset == 0
    }
    for symbol in symbols:
        if symbol not in bindable:
            raise IrError(
                f"graph.symbols: '{symbol}' が入力 shape に素の形（'{symbol}'）で現れない"
                " — 束縛が取れない"
            )


def _check_definitions(
    inputs: Sequence[IrInput],
    initializers: Mapping[str, IrInitializer],
    nodes: Sequence[IrNode],
    outputs: Sequence[str],
) -> set[str]:
    """SSA 単一代入 + トポロジカル順（前方参照拒否）+ outputs の定義済み検査。"""
    defined: set[str] = set()

    def define(name: str, where: str) -> None:
        if name in defined:
            raise IrError(f"{where}: 値 '{name}' が二重に定義されている（SSA 単一代入違反）")
        defined.add(name)

    for spec in inputs:
        define(spec.name, "graph.inputs")
    for name in initializers:
        define(name, "graph.initializers")
    for index, node in enumerate(nodes):
        where = f"graph.nodes[{index}] ({node.op})"
        for ref in node.ins:
            # nodes はトポロジカル順で格納される MUST — 前方参照を許すと実行順が
            # 暗黙の依存解析任せになる。
            if ref not in defined:
                raise IrError(f"{where}: 入力 '{ref}' が未定義（前方参照または未宣言）")
        for out in node.outs:
            define(out, where)
    for output in outputs:
        if output not in defined:
            raise IrError(f"graph.outputs: '{output}' が未定義")
    return defined


def _check_declarations(
    inputs: Sequence[IrInput],
    initializers: Mapping[str, IrInitializer],
    values: Mapping[str, IrValue],
    nodes: Sequence[IrNode],
    defined: set[str],
) -> None:
    """宣言の完全性: 入力は inputs[] が、initializer とノード出力は values{} が、
    それぞれちょうど 1 回宣言する。孤立宣言（誰も定義しない values）も fail loudly。
    """
    input_names = {spec.name for spec in inputs}
    for name in values:
        if name in input_names:
            raise IrError(f"graph.values['{name}']: 入力は inputs[] で宣言済み（二重宣言）")
        if name not in defined:
            raise IrError(f"graph.values['{name}']: どのノードでも定義されない宣言")
    for name in initializers:
        if name not in values:
            raise IrError(f"graph.initializers['{name}']: values に dtype/shape 宣言が無い")
        # 意味論と格納の組は INITIALIZER_STORAGE だけが決める（格納 dtype の実行可否は
        # 別層 — 対応表突合）。
        allowed = INITIALIZER_STORAGE.get(values[name].dtype)
        if allowed is None:
            raise IrError(
                f"graph.values['{name}']: initializer の意味論 dtype '{values[name].dtype}' は"
                " 語彙外（f32 / i32 のみ）"
            )
        storage_dtype = initializers[name].storage.dtype
        if storage_dtype not in allowed:
            raise IrError(
                f"graph.initializers['{name}']: 意味論 dtype '{values[name].dtype}' に"
                f" 格納 dtype '{storage_dtype}' は組めない（{' / '.join(allowed)} のみ）"
            )
        # initializer は束縛前に確定していなければ safetensors 側 shape と突合できない。
        if any(not isinstance(dim, int) for dim in values[name].shape):
            raise IrError(f"graph.values['{name}']: initializer の shape に記号次元は使えない")
    for node in nodes:
        for out in node.outs:
            if out not in values:
                raise IrError(f"graph.values: ノード出力 '{out}' の dtype/shape 宣言が無い")


def _check_required_ops(required_ops: Sequence[str], nodes: Sequence[IrNode]) -> None:
    """requires.ops ≡ nodes で実際に使われる op 集合（ランタイム突合の前提）。"""
    used = {node.op for node in nodes}
    declared = set(required_ops)
    missing = sorted(used - declared)
    extra = sorted(declared - used)
    if missing or extra:
        raise IrError(
            f"graph.requires.ops が使用 op 集合と一致しない: 宣言漏れ {missing} / 余剰 {extra}"
        )


# ---- ランタイム対応表との突合 ---------------------------------------------


def _declared_dtype(graph: IrGraph, name: str) -> str:
    for spec in graph.inputs:
        if spec.name == name:
            return spec.dtype
    return graph.values[name].dtype


def _assert_sym_prefix_slice(graph: IrGraph, node: IrNode, where: str) -> None:
    """sym_prefix_slice の**グラフ文脈が要る**契約
    （ADR 0010 / packages/runtime/src/runtime/plan.ts と同義）。

    1. `sym` が graph.symbols にある（無ければランタイムが束縛を取れず prefix 長が決まらない）
    2. `dim` が入力 rank の内側
    3. 入力の宣言 shape が**記号を含まない静的形**（= Tmax 形）

    MUST: 3 は束縛後の数値 shape からは見分けが付かない（T = Tmax の run では一致する）。
    宣言の形を見られるここでしか検出できない。
    """
    sym, slices = sym_prefix_slice_attrs(node.attrs, where)
    if sym not in graph.symbols:
        raise IrError(
            f"{where}: sym_prefix_slice の sym '{sym}' が graph.symbols {graph.symbols} に無い"
        )
    source = declared_shape(graph, node.ins[0])
    if any(not isinstance(dim, int) for dim in source):
        raise IrError(
            f"{where}: sym_prefix_slice の入力 '{node.ins[0]}' の宣言 shape {source} に"
            " 記号次元がある（入力は Tmax で焼いた静的形でなければならない）"
        )
    for entry in slices:
        if entry["dim"] >= len(source):
            raise IrError(
                f"{where}: sym_prefix_slice の dim {entry['dim']} が入力 rank {len(source)} の外"
            )


def assert_runtime_support(graph: IrGraph) -> None:
    """M0 ランタイムが実行できる形かを突合する（packages/runtime/src/format/container.ts と同義）。

    MUST: op 名だけでなく**意味論 dtype と attrs まで**見る。名前だけの突合は
    「対応表にはあるのに実行時に落ちる」を作る（ADR 0005）。非対応は**全件列挙**する。
    """
    missing_ops: set[str] = set()
    # dtype 違反は宣言（値名）単位に重複除去する — 素朴に積むと件数が
    # 「直すべき宣言の本数」より多く出て、列挙の指標としての意味が薄れる。
    bad_dtypes: dict[str, str] = {}
    bad_attrs: list[str] = []
    # 転送層の軸。どのノードも消費しない入力にも制約が実在するので、ノード起点の突合とは
    # 別に見る（宣言順 = inputs が先）。
    for spec in graph.inputs:
        if spec.dtype not in IO_DTYPES:
            bad_dtypes[spec.name] = spec.dtype
    for index, node in enumerate(graph.nodes):
        contract = OP_CONTRACTS.get(node.op)
        if contract is None:
            missing_ops.add(node.op)
            continue
        where = f"nodes[{index}] ({node.op})"
        # MUST: 入力は**スロット別**の受理集合で見る。和だけで突き合わせると gather /
        # embedding / masked_fill のスロット取り違え（値と添字を逆に渡した形）が
        # 「どちらも和には入っている」として列挙門を素通りし、契約検査まで落ちて初めて
        # 1 件ずつ止まる（「非対応は全件列挙」の意図が壊れる）。
        for slot, name in enumerate(node.ins):
            # 契約よりも入力が多い形（アリティ違反）は契約検査の担当。ここは列挙門なので、
            # 対応するスロットが無いぶんは和で見て 1 件でも多く拾う。
            accept = contract.slot_accept(slot) if slot < contract.arity else contract.dtypes
            dtype = _declared_dtype(graph, name)
            if dtype not in accept:
                bad_dtypes[name] = dtype
        # MUST: 出力は契約表の**写像の値域**で見る（cast は attrs.to で決まるので語彙全体）。
        # 入力側の受理集合で代用すると、比較（f32 → bool）や bool の sum（→ i32）のように
        # dtype が変わる op で**正しいグラフが列挙門で落ちる**。
        out_accept = (
            SEMANTIC_DTYPES
            if contract.kind == "cast"
            else frozenset(contract.output_dtypes.values())
        )
        for name in node.outs:
            dtype = _declared_dtype(graph, name)
            if dtype not in out_accept:
                bad_dtypes[name] = dtype
        unknown = sorted(key for key in node.attrs if key not in contract.attrs)
        if unknown:
            bad_attrs.append(f"{where}: {', '.join(unknown)}")

    missing_storage: dict[str, list[str]] = {}
    for name, initializer in graph.initializers.items():
        dtype = initializer.storage.dtype
        if dtype not in M0_STORAGE_DTYPES:
            missing_storage.setdefault(dtype, []).append(name)

    if not (missing_ops or bad_dtypes or bad_attrs or missing_storage):
        return
    diagnostics: list[str] = []
    if missing_ops:
        diagnostics.append(f"非対応 op ({len(missing_ops)}): {', '.join(sorted(missing_ops))}")
    if bad_dtypes:
        listed = ", ".join(f"値 '{name}': {dtype}" for name, dtype in bad_dtypes.items())
        diagnostics.append(f"非対応 意味論 dtype ({len(bad_dtypes)}): {listed}")
    if bad_attrs:
        diagnostics.append(f"未実装 attrs ({len(bad_attrs)}): {'; '.join(bad_attrs)}")
    for dtype, users in sorted(missing_storage.items()):
        diagnostics.append(
            f"非対応 格納 dtype '{dtype}' ({len(users)}): {', '.join(sorted(users))}"
        )
    raise ContainerError(f"ランタイムの capability 不足 — {' / '.join(diagnostics)}")


def assert_op_contracts(graph: IrGraph) -> None:
    """毎ノードの契約検査（アリティ / 単一出力 / attrs スキーマ / 入出力 dtype 規則 / shape）。

    NOTE: attrs と dtype は assert_runtime_support も見るが層が違う — あちらは「モデル作者へ
    capability 不足を一度に列挙する門」、こちらは「対応表に載っている op が契約どおりに
    組まれているか」の検査。両者とも ops.py の契約表由来なので規則が割れることはない。
    """
    for index, node in enumerate(graph.nodes):
        where = f"graph.nodes[{index}]"
        contract = assert_node_contract(node, where)
        resolve_node_dtypes(
            contract,
            node,
            [_declared_dtype(graph, name) for name in node.ins],
            _declared_dtype(graph, node.outs[0]),
            where,
        )
        # strided コピー族の rank 上限（束縛に依らず宣言 shape の長さだけで決まる）。
        if contract.name in STRIDED_RANK_OPS:
            for slot, name in enumerate(node.ins):
                assert_strided_rank(
                    len(declared_shape(graph, name)), f"入力 {slot} '{name}'", where
                )
            out = node.outs[0]
            assert_strided_rank(len(declared_shape(graph, out)), f"出力 '{out}'", where)
        if contract.kind == "sym_prefix_slice":
            _assert_sym_prefix_slice(graph, node, where)
    # 出力 shape の突合は全ノードの宣言が揃ってから（shapes.py が契約の規則から独立に
    # 計算し、torch の meta 由来の宣言と食い違えば落とす）。
    assert_graph_shapes(graph)


# ---- 配布形（safetensors）との突合 -----------------------------------------

#: safetensors dtype → 要素バイト数（= Karume リーダが要求する整列）。
#: 正本は TS 側 `packages/runtime/src/format/safetensors.ts` の DTYPE_BYTES。
READER_DTYPE_BYTES = {
    "F32": 4,
    "F16": 2,
    "BF16": 2,
    "I8": 1,
    "U8": 1,
    "I32": 4,
    "U32": 4,
    "I64": 8,
    "BOOL": 1,
}

_HEADER_LENGTH_BYTES = 8


def assert_reader_layout(path: str | Path) -> None:
    """Karume のリーダ（`packages/runtime/src/format/safetensors.ts`）が読めるレイアウトかを見る。

    HF の `safe_open` は読めるのに Karume が読めないファイルが作れる — リーダは
    「データ節を隙間なく覆う」「各テンソルの**絶対** offset が要素サイズに整列している」を
    要求し、後者は要素数が奇数の F16（バイト長 ≡ 2 mod 4）の直後に F32 / I32 を置くと
    破れる（docs/limitations.md）。並び順はエクスポータの責務なので、**書いた側で**
    その責務を果たせているかをここで検査する。

    MUST: この検査は `safetensors` のリーダを通さない（通すと同じ規則の再実装ではなく
    「別のリーダが読めた」だけの主張になる）。ヘッダ JSON を直に読んで規則を写す。
    """
    with Path(path).open("rb") as handle:
        raw_length = handle.read(_HEADER_LENGTH_BYTES)
        if len(raw_length) != _HEADER_LENGTH_BYTES:
            raise ContainerError(
                f"ファイルが短すぎる: {len(raw_length)} バイト（ヘッダ長すら無い）"
            )
        header_length = int.from_bytes(raw_length, "little")
        header_bytes = handle.read(header_length)
        if len(header_bytes) != header_length:
            raise ContainerError(f"ヘッダ長 {header_length} がファイル長を超える")
        data_start = _HEADER_LENGTH_BYTES + header_length
        handle.seek(0, 2)
        data_length = handle.tell() - data_start

    try:
        header = json.loads(header_bytes)
    except ValueError as cause:
        raise ContainerError(f"safetensors ヘッダ JSON を解析できない: {cause}") from cause

    declared = []
    for name, entry in header.items():
        if name == "__metadata__":
            continue
        dtype = entry["dtype"]
        if dtype not in READER_DTYPE_BYTES:
            raise ContainerError(f"テンソル '{name}': リーダが知らない dtype '{dtype}'")
        begin, end = entry["data_offsets"]
        count = 1
        for dim in entry["shape"]:
            count *= dim
        if end - begin != count * READER_DTYPE_BYTES[dtype]:
            raise ContainerError(
                f"テンソル '{name}': サイズ不一致 offsets={end - begin} "
                f"期待={count * READER_DTYPE_BYTES[dtype]}（{dtype} {entry['shape']}）"
            )
        declared.append((begin, end, name, dtype))

    cursor = 0
    for begin, end, name, dtype in sorted(declared):
        if begin != cursor:
            raise ContainerError(
                f"テンソル '{name}': データ節が隙間なく覆われていない"
                f"（使用済み末尾={cursor} / このテンソルの開始={begin}）"
            )
        align = READER_DTYPE_BYTES[dtype]
        if (data_start + begin) % align != 0:
            raise ContainerError(
                f"テンソル '{name}': 絶対 offset {data_start + begin} が {dtype} の"
                f" 要素サイズ {align} に整列していない"
                "（奇数要素の F16 より後ろに 4 バイト型を置いていないか — "
                "並び順の規約は karume/emit.py）"
            )
        cursor = end
    if cursor != data_length:
        raise ContainerError(f"データ節末尾に未使用領域が {data_length - cursor} バイトある")


def _assert_scale_tensor(
    handle: Any,
    graph: IrGraph,
    keys: set[str],
    name: str,
    scale_key: str,
    weight_shape: list[Any],
) -> None:
    """量子化格納の scale テンソルを実ファイルと突き合わせる
    （`packages/runtime/src/format/container.ts` の鏡像）。

    MUST: 4 点すべてを見る。scale は IR の値ではなく safetensors の**素のテンソル**なので、
    ここを緩めるとロード時（ランタイム）まで誤りが出ない — 「書けたが読めない」ファイルを
    配布物として残さないのが `verify_model` の役目（ADR 0005）。

    1. **実在**する
    2. **F32**（scale を別 dtype のビット列として読むと全チャネルが桁違いの値になる）
    3. **重みと同 rank の keepdim broadcast 形**（各軸は 1 か重みと同値）
    4. **他 initializer の実体との名前衝突が無い**（別の重みを scale として読む形）

    NOTE: 「非 1 の軸が消費側 op の**チャネル軸**と一致するか」はここでは見ない（TS 側
    container.ts と同じ切り方 — あちらは executor が見る）。書き出し側の相当物は
    `emit._store_i8` の keepdim 形検査で、そちらは軸を知っているので厳密に見る。
    """
    where = f"initializer '{name}'"
    if scale_key not in keys:
        raise ContainerError(f"{where}: scale テンソル '{scale_key}' がファイルに無い")
    for other, initializer in graph.initializers.items():
        if initializer.tensor == scale_key:
            raise ContainerError(
                f"{where}: scale テンソル '{scale_key}' が initializer '{other}' の実体と同じキー"
            )
    view = handle.get_slice(scale_key)
    if view.get_dtype() != "F32":
        raise ContainerError(
            f"{where}: scale テンソル '{scale_key}' が {view.get_dtype()}（F32 が必要）"
        )
    shape = list(view.get_shape())
    if len(shape) != len(weight_shape):
        raise ContainerError(
            f"{where}: scale {shape} の rank が重み {weight_shape} と違う"
            "（keepdim broadcast 形が必要）"
        )
    if any(dim != 1 and dim != weight_shape[axis] for axis, dim in enumerate(shape)):
        raise ContainerError(f"{where}: scale {shape} が重み {weight_shape} へ broadcast できない")


def verify_model(path: str | Path) -> IrGraph:
    """配布形 1 ファイルを IR v1 の全規則で検証し、読めたグラフを返す。"""
    assert_reader_layout(path)
    with safe_open(str(path), framework="pt") as handle:
        metadata = handle.metadata() or {}
        text = metadata.get(IR_METADATA_KEY)
        if text is None:
            raise ContainerError(f"__metadata__.{IR_METADATA_KEY} が無い（Karume モデルではない）")
        graph = parse_ir_graph(text)
        keys = set(handle.keys())
        for name, initializer in graph.initializers.items():
            where = f"initializer '{name}'"
            if initializer.tensor not in keys:
                raise ContainerError(f"{where}: テンソル '{initializer.tensor}' がファイルに無い")
            view = handle.get_slice(initializer.tensor)
            expected = STORAGE_ENCODING[initializer.storage.dtype]
            if view.get_dtype() != expected:
                raise ContainerError(
                    f"{where}: 格納 dtype '{initializer.storage.dtype}' に対し safetensors 側が"
                    f" {view.get_dtype()}（{expected} が必要）"
                )
            declared = graph.values[name].shape
            actual = list(view.get_shape())
            if declared != actual:
                raise ContainerError(f"{where}: 宣言 shape {declared} ≠ 実テンソル {actual}")
            scale = initializer.storage.scale
            if scale is not None:
                _assert_scale_tensor(handle, graph, keys, name, scale, declared)
    assert_runtime_support(graph)
    assert_op_contracts(graph)
    return graph


# ---- CLI ------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="配布形 safetensors を IR v1 の全規則で検証する")
    parser.add_argument(
        "models", type=Path, nargs="+", help="検証する model.safetensors（複数指定可）"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """指定された配布形を 1 本ずつ検証する。

    MUST: 落ちたファイルで止める（残りを検証して最後にまとめない）— 例外は規則違反の
    位置まで綴ってあるので、そのまま送出するのが最も情報量が多い。
    """
    args = build_parser().parse_args(argv)
    for path in args.models:
        graph = verify_model(path)
        print(
            f"{path}: nodes={len(graph.nodes)} initializers={len(graph.initializers)}"
            f" inputs={len(graph.inputs)} outputs={len(graph.outputs)}"
            f" symbols={','.join(graph.symbols) if graph.symbols else '（静的）'}"
        )


if __name__ == "__main__":
    main()
