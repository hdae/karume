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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from karume.dims import MAX_SAFE_INT, is_symbol_name, parse_dim, try_parse_dim
from karume.emit import EmitError, eligible_compressed_initializers, weight_channel_axes
from karume.ir import (
    IR_FORMAT,
    IR_METADATA_KEY,
    IR_VERSION,
    MIN_GROUP_SIZE,
    IrDim,
    IrGraph,
    IrInitializer,
    IrInput,
    IrNode,
    IrState,
    IrStorage,
    IrValue,
)
from karume.ops import (
    IO_DTYPES,
    M0_STORAGE_DTYPES,
    OP_CONTRACTS,
    STATE_APPEND_OP,
    STRIDED_RANK_OPS,
    assert_node_contract,
    assert_strided_rank,
    resolve_node_dtypes,
    state_window,
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

#: 省略可能なトップレベル節。`states` を持たないグラフ（= 既存の全モデル）は無風
#: （ADR 0066 決定 2 の「states を出す最初のモデルまで無風」）。
OPTIONAL_TOP_LEVEL_KEYS = ("states",)

SEMANTIC_DTYPES = ("f32", "i32", "bool")
STORAGE_DTYPES = ("f32", "f16", "bf16", "i8", "i4", "i32")

#: scale / group_size の記述子を持てる格納 dtype（量子化格納）。TS 側
#: `packages/runtime/src/format/ir.ts` の QUANTIZED_STORAGE_DTYPES の鏡像。
QUANTIZED_STORAGE_DTYPES = ("i8", "i4")

#: state スロットの dtype 語彙。現状 f32 のみ（ADR 0066 決定 2）。
STATE_DTYPES = ("f32",)
#: 席だけが予約されている state スロットの dtype（ADR 0066 追記 5）。f16 格納は数値契約が
#: 変わるので ADR 0058 流儀の opt-in が要る — それが無いうちは「語彙外」ではなく**未対応**。
RESERVED_STATE_DTYPES = ("f16",)
#: state スロットの rank 上限（ADR 0066 決定 2 の「固定 rank（rank ≤ 4）」）。strided カーネルの
#: 上限（ops.STRIDED_RANK）と同じ数値だが理由が別なので定数を共有しない。
MAX_STATE_RANK = 4

#: 格納 dtype → safetensors dtype。f32/f16/bf16/i8/i4 は意味論 f32 の符号化で、`i32` だけが
#: 生の int32（ADR 0010 の明示的な例外）。`i4` は packed 4bit（ADR 0069 決定 2 — shape は
#: 論理形のままで、バイト数だけが bit 幅から決まる）。
STORAGE_ENCODING = {
    "f32": "F32",
    "f16": "F16",
    "bf16": "BF16",
    "i8": "I8",
    "i4": "I4",
    "i32": "I32",
}

#: initializer の意味論 dtype → 許される格納 dtype（docs/ir-v1.md「値と型」）。
#: MUST: 交差を許さない — `i32` 宣言の initializer が f16 のビット列として読まれる
#: 沈黙誤値になる。bool の initializer は語彙に無い。
INITIALIZER_STORAGE = {"f32": ("f32", "f16", "bf16", "i8", "i4"), "i32": ("i32",)}


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
        # MUST: JSON の `1.0` / `1e0` は int へ正規化する。TS 側は JSON.parse が単一の number を
        # 返すので整数値の float という区別が無く（`Number.isSafeInteger(1.0)` は true）、Python が
        # float を丸ごと拒むと**ランタイムが読める graph をエクスポータの検証だけが読めない**
        # 乖離になる（受理集合はどちらの向きにもずれてはいけない）。非整数値の float は
        # `is_integer()` が False なので下の「数値でも文字列でもない」で従来どおり落ち、非有限値は
        # そもそも parse_graph_json が JSON 読みの時点で弾く。safe range 超過も int 側の検査
        # （`> MAX_SAFE_INT`）に載る = TS の `Number.isSafeInteger` と同じ受理集合になる。
        if isinstance(dim, float) and dim.is_integer():
            dim = int(dim)
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


def _as_state_dtype(value: Any, where: str) -> str:
    dtype = _as_nonempty_str(value, where)
    if dtype in RESERVED_STATE_DTYPES:
        raise IrError(
            f"{where}: state スロットの dtype '{dtype}' は未対応"
            "（ADR 0066 追記 5 の席予約 — 数値契約の opt-in が要る）"
        )
    if dtype not in STATE_DTYPES:
        raise IrError(
            f"{where}: state スロットの dtype '{dtype}' は語彙外（{' / '.join(STATE_DTYPES)}）"
        )
    return dtype


def _parse_state(value: Any, symbols: set[str], where: str) -> IrState:
    """state スロット 1 本の宣言（ADR 0066 決定 2・TS 側 parseStateSlot の鏡像）。

    MUST: shape は**容量込みの具体形**なので数値次元は正整数（`values` の非負とは違う — 容量 0 の
    スロットは束縛できる実体を持たない）。rank は 1..MAX_STATE_RANK（容量軸を持たない rank 0 は
    「容量込み」を満たせない）。記号次元は `symbols` 宣言済みならよく、**states の shape も
    束縛点になる**（`createGenerationContext` が決める容量 — ADR 0066 追記 7。
    _check_symbol_bindability）。
    """
    obj = _as_object(value, where)
    _check_keys(obj, ["dtype", "shape"], [], where)
    dtype = _as_state_dtype(obj["dtype"], f"{where}.dtype")
    shape = _parse_shape(obj["shape"], symbols, f"{where}.shape")
    if not 1 <= len(shape) <= MAX_STATE_RANK:
        raise IrError(
            f"{where}.shape: rank {len(shape)} は 1..{MAX_STATE_RANK} の外"
            "（固定 rank の容量込み具体形 MUST）"
        )
    for index, dim in enumerate(shape):
        if isinstance(dim, int) and dim < 1:
            raise IrError(f"{where}.shape[{index}]: 次元 {dim} が正整数でない（容量が取れない）")
    return IrState(dtype=dtype, shape=shape)


def _parse_storage(value: Any, where: str) -> IrStorage:
    obj = _as_object(value, where)
    _check_keys(obj, ["dtype"], ["scale", "group_size"], where)
    dtype = _as_storage_dtype(obj["dtype"], f"{where}.dtype")
    has_scale = "scale" in obj
    has_group_size = "group_size" in obj
    # scale / group_size は量子化格納の記述子。非量子化 dtype に付いているのは
    # エクスポータの取り違えなので受理しない（黙って無視すると格納の意味が二重化する）。
    if dtype not in QUANTIZED_STORAGE_DTYPES and (has_scale or has_group_size):
        raise IrError(f"{where}: 格納 dtype '{dtype}' に scale / group_size は付けられない")
    # MUST: i8 / i4 は scale を**明示宣言**する
    # （ADR 0019 / 0069・TS 側 packages/runtime/src/format/ir.ts の鏡像）。
    # 既定 1.0 で補完すると、scale の書き忘れが「全チャネル 1.0 で dequant した重み」に化けて
    # ロードも実行も通ってしまう（差が O(scale) で出るのに、どこにも例外が出ない）。
    if dtype in QUANTIZED_STORAGE_DTYPES and not has_scale:
        raise IrError(f"{where}: 格納 dtype '{dtype}' には scale（scale テンソルのキー）が要る")
    # MUST: i4 は group_size を**明示宣言**する（ADR 0069 決定 2）。group 長が決まらない
    # 4bit 格納は scale の引き直し位置が決まらず、展開が黙って別の値を出す。
    if dtype == "i4" and not has_group_size:
        raise IrError(f"{where}: 格納 dtype 'i4' には group_size が要る（ADR 0069 決定 2）")
    scale = _as_nonempty_str(obj["scale"], f"{where}.scale") if has_scale else None
    group_size = None
    if has_group_size:
        raw = obj["group_size"]
        if isinstance(raw, bool) or not isinstance(raw, int) or raw < 1:
            raise IrError(f"{where}.group_size: 正整数でない")
        # TS 側は JSON の数値として読むので 2^53−1 を超える値は整数として持てない
        # （packages/runtime/src/format/ir.ts）。ここで受理するとランタイムだけが落ちる。
        if raw > MAX_SAFE_INT:
            raise IrError(f"{where}.group_size: {raw} が安全整数 2^53−1 を超える")
        # MUST: i4 の group 長は 2 冪かつ 16 以上（ADR 0069 決定 2 — ORT と同制約）。この制約が
        # 行境界・group 境界を常にバイト整列させ、末尾ゼロ詰め無しで u32 束縛が成立する。
        if dtype == "i4" and (raw & (raw - 1) != 0 or raw < MIN_GROUP_SIZE):
            raise IrError(
                f"{where}.group_size: {raw} が 2 冪かつ {MIN_GROUP_SIZE} 以上でない"
                "（ADR 0069 決定 2）"
            )
        group_size = raw
    return IrStorage(dtype=dtype, scale=scale, group_size=group_size)


def _parse_node_states(
    obj: Mapping[str, Any], slots: Mapping[str, IrState], where: str
) -> dict[str, str]:
    """ノードの `states` 欄（ADR 0067 決定 4・TS 側 parseNodeStates の鏡像）。

    ここで見るのは**グラフ単体で決まる 3 点**だけ: plain object であること・キーと値が空でない
    文字列であること・値が `graph.states` で宣言済みのスロット名であること。

    MUST: 未宣言スロットの参照は fail loudly。通すと「実体を持たない名前を読む」ノードが
    Session 構築を抜け、確保も束縛もされないまま実行段で初めて落ちる（値側の前方参照拒否と
    同じ層の規則）。

    NOTE: キー集合そのもの（`{k,v}` ちょうど / `{slot}` ちょうど）は op 契約の担当
    （ops.assert_node_contract）— パーサは op 語彙を知らない。
    """
    states: dict[str, str] = {}
    if "states" not in obj:
        return states
    for key, value in _as_object(obj["states"], f"{where}.states").items():
        _as_nonempty_str(key, f"{where}.states のキー")
        slot = _as_nonempty_str(value, f"{where}.states['{key}']")
        if slot not in slots:
            raise IrError(
                f"{where}.states['{key}']: state スロット '{slot}' が graph.states で"
                "宣言されていない"
            )
        states[key] = slot
    return states


# ---- グラフ単体の規則 -----------------------------------------------------


def parse_ir_graph(text: str) -> IrGraph:
    """グラフ JSON を検証しつつ読む（packages/runtime/src/format/ir.ts parseIrGraph と同義）。"""
    root = _as_object(parse_graph_json(text), "graph")
    _check_keys(root, TOP_LEVEL_KEYS, OPTIONAL_TOP_LEVEL_KEYS, "graph")

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

    # 省略は空スロット集合として扱う（節を持たないグラフが無風 — ADR 0066 決定 2）。
    states: dict[str, IrState] = {}
    for name, raw in _as_object(root.get("states", {}), "graph.states").items():
        # 空のスロット名は拒否する — 参照側の欄（ADR 0067）は空でない文字列だけを受理するので、
        # 通すと「宣言はできるが原理的に参照できないスロット」になる（values は孤立宣言検査が
        # 同じ穴を塞いでいる）。
        _as_nonempty_str(name, "graph.states のスロット名")
        states[name] = _parse_state(raw, symbol_set, f"graph.states['{name}']")

    nodes: list[IrNode] = []
    for index, raw in enumerate(_as_array(root["nodes"], "graph.nodes")):
        where = f"graph.nodes[{index}]"
        obj = _as_object(raw, where)
        _check_keys(obj, ["op", "ins", "outs", "attrs"], ["states"], where)
        # NOTE: `outs` の本数はここでは見ない（0 本 = 値を定義しない effect op が語彙に入った —
        # ADR 0067 決定 5）。「0 本を許すのは契約が effect を宣言する op だけ」の執行点は契約層
        # （assert_node_contract の出力数突合）で、パーサは本数に意味を与えない。
        nodes.append(
            IrNode(
                op=_as_nonempty_str(obj["op"], f"{where}.op"),
                ins=[
                    _as_nonempty_str(item, f"{where}.ins[{i}]")
                    for i, item in enumerate(_as_array(obj["ins"], f"{where}.ins"))
                ],
                outs=[
                    _as_nonempty_str(out, f"{where}.outs[{i}]")
                    for i, out in enumerate(_as_array(obj["outs"], f"{where}.outs"))
                ],
                attrs=_as_object(obj["attrs"], f"{where}.attrs"),
                states=_parse_node_states(obj, states, where),
            )
        )

    _check_symbol_bindability(symbols, inputs, states, values)
    defined = _check_definitions(inputs, initializers, nodes, outputs)
    _check_declarations(inputs, initializers, values, nodes, defined)
    _check_state_slots(states, values, defined, nodes)
    _check_required_ops(required_ops, nodes)

    return IrGraph(
        symbols=symbols,
        inputs=inputs,
        outputs=outputs,
        initializers=initializers,
        values=values,
        states=states,
        nodes=nodes,
    )


def _symbols_in(shape: Sequence[IrDim]) -> set[str]:
    """shape に現れるシンボル名（次元位置の出現のみ — 要素数からの逆算はしない）。"""
    return {parse_dim(dim).sym for dim in shape if isinstance(dim, str)}


def _check_symbol_bindability(
    symbols: Sequence[str],
    inputs: Sequence[IrInput],
    states: Mapping[str, IrState],
    values: Mapping[str, IrValue],
) -> None:
    """宣言されたシンボルは**束縛点を持つ** MUST。束縛点は 2 つ（ADR 0066 追記 7）:

    1. **入力 shape の次元位置**（run ごとの実寸から解く — TS 側 `runtime/plan.ts` の
       `bindSymbols`）。派生形（`2T` / `T+8`）でもよい — 1 次元 1 シンボルの一次式は実寸から
       解が一意に決まる（ADR 0057）。
    2. **states の shape の次元位置**（`createGenerationContext(spec.bindings)` が決める KV 容量 —
       context 生成時にユーザーが決める値なので、export 時定数に焼く形は ADR 0066 決定 3
       〈静的物理格納〉と矛盾する）。

    MUST: **states 専用記号（states にしか現れない記号）は値 shape に現れてはならない**
    （追記 7）。通常値 shape の解決に効くのは入力由来の束縛だけなので、現れると実行時に必ず
    束縛不能になる — 宣言の時点で落とす。
    """
    from_inputs: set[str] = set()
    for spec in inputs:
        from_inputs |= _symbols_in(spec.shape)
    from_states: set[str] = set()
    for slot in states.values():
        from_states |= _symbols_in(slot.shape)
    for symbol in symbols:
        if symbol not in from_inputs and symbol not in from_states:
            raise IrError(
                f"graph.symbols: '{symbol}' が入力 shape / states shape の次元位置に現れない"
                " — 束縛が取れない"
            )
    for name, value in values.items():
        for symbol in sorted(_symbols_in(value.shape)):
            if symbol not in from_states or symbol in from_inputs:
                continue
            raise IrError(
                f"graph.values['{name}']: states 専用記号 '{symbol}' が値 shape に現れる"
                "（値 shape の解決に効くのは入力由来の束縛だけ — ADR 0066 追記 7）"
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


def _check_group_quantized_shape(name: str, initializer: IrInitializer, value: IrValue) -> None:
    """group 量子化格納（i4）の宣言 shape と group 長の整合（ADR 0069 決定 2・
    TS 側 checkGroupQuantizedShape の鏡像）。

    量子化軸は**最終次元**（linear の in 軸）で、そこが `group_size` で割り切れることが MUST。
    端数 group を許すと最後の group だけ scale の担当範囲が短くなり、行境界が語境界からずれて
    平坦添字の展開が黙って別の値を出す（端数を作らない制約で整列問題そのものを消す設計）。
    """
    # 値域（2 冪かつ 16 以上）は _parse_storage が保証済み。存在は型の上でだけ optional なので、
    # 黙って読み飛ばさず言い直す（TS 側 checkGroupQuantizedShape と同じ流儀）。
    group_size = initializer.storage.group_size
    if group_size is None:
        raise IrError(f"graph.initializers['{name}']: 格納 i4 なのに group_size が無い")
    if not value.shape:
        raise IrError(f"graph.values['{name}']: 格納 i4 の initializer に量子化軸が無い（rank 0）")
    last_dim = value.shape[-1]
    if not isinstance(last_dim, int) or last_dim % group_size != 0:
        raise IrError(
            f"graph.values['{name}']: 格納 i4 の最終次元 {last_dim} が"
            f" group_size {group_size} で割り切れない（ADR 0069 決定 2）"
        )


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
        if storage_dtype == "i4":
            _check_group_quantized_shape(name, initializers[name], values[name])
    for node in nodes:
        for out in node.outs:
            if out not in values:
                raise IrError(f"graph.values: ノード出力 '{out}' の dtype/shape 宣言が無い")


def _check_state_slots(
    states: Mapping[str, IrState],
    values: Mapping[str, IrValue],
    defined: set[str],
    nodes: Sequence[IrNode],
) -> None:
    """state スロット名の検査（TS 側 checkStateSlots の鏡像）。

    スロットは値ではない（`ins` / `outs` で参照されず、ノードからは別の欄で名前参照する —
    ADR 0066 決定 1・0067 決定 4）ので**値名前空間とは別**だが、**同名は拒否する**: 別名前空間の
    同名は「スロット名を書くべき欄に値名を書いた / その逆」を検出できなくするだけで、表現力を
    何も足さない。scale テンソルのキーを他 initializer の実体と衝突させない規則（ADR 0019）と
    同じ流儀。

    **参照完全性**（ADR 0067 決定 4 / 5）: 宣言されたスロットは少なくとも 1 つのノードの
    `states` 欄から参照される MUST。誰も参照しないスロットは values の孤立宣言と同じ穴で、
    GenerationContext が確保だけして誰も読まない容量（KV なら数十 MiB 単位）が黙って残る。

    MUST: 衝突検査を**先**に置く（値名と同名のスロットは参照の有無に関わらず取り違えなので、
    「参照されていない」という別の診断に化けさせない）。
    """
    for name in states:
        if name in defined or name in values:
            raise IrError(
                f"graph.states['{name}']: 値名と同名"
                "（state スロットは値名前空間と別 — 取り違えを拒否する）"
            )
    referenced = {slot for node in nodes for slot in node.states.values()}
    for name in states:
        if name not in referenced:
            raise IrError(f"graph.states['{name}']: どのノードからも参照されない宣言")


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


def _show_window(touch: tuple[int, str, int | None]) -> str:
    """`_assert_state_order` の診断片（`(nodes 添字, op 名, window)` → 表示形）。"""
    index, op, window = touch
    return f"nodes[{index}] ({op}) は {'宣言なし' if window is None else window}"


def _assert_state_order(graph: IrGraph) -> None:
    """state effect の順序（ADR 0067 決定 5b の②・TS 側 `runtime/plan.ts` の assertStateOrder の
    鏡像）。

    state 参照は**テンソルのデータ辺を張らない**ため DAG のトポロジ順では順序が決まらず、
    契約は `nodes` **配列順**そのもの。束縛に依存しないので、スロットごとに 3 点を見る:

    1. `state_append` は 1 スロットにつき**1 本まで**（1 step に 2 回書く形は ring の位置式が
       二重に進み、読者が見る過去が step の途中で変わる）
    2. append が在るなら**そのスロットに触れる最後のノード**（append より後に読者が居ると、
       その読者は「今 step の k/v を過去として二重に読む」）
    3. 同一スロットに触れる全ノードの `window` は**存在有無も値も一致**（論理 col → 物理 row の
       写像は読み書き同式 MUST — ADR 0067 決定 4。読み側だけ別式にすると沈黙誤読）

    MUST: fail loudly。3 点とも「順序 / 宣言の誤り」が例外ではなく**別の値**として出る種類の
    破れなので、書き出しの時点でしか止められない。
    """
    touches: dict[str, list[tuple[int, str, int | None]]] = {}
    for index, node in enumerate(graph.nodes):
        if not node.states:
            continue
        # attrs の値域検査は assert_node_contract が済ませている（ここは引き直すだけ）。
        window = state_window(node.attrs, f"nodes[{index}] ({node.op})")
        for slot in node.states.values():
            touches.setdefault(slot, []).append((index, node.op, window))
    for slot, touched in touches.items():
        appends = [entry for entry in touched if entry[1] == STATE_APPEND_OP]
        if len(appends) > 1:
            listed = ", ".join(f"nodes[{index}]" for index, _, _ in appends)
            raise IrError(
                f"state スロット '{slot}': {STATE_APPEND_OP} が {len(appends)} 本（{listed}）"
                " — 1 step に 1 回まで（ADR 0067 決定 5b）"
            )
        last = touched[-1]
        if len(appends) == 1 and last[1] != STATE_APPEND_OP:
            raise IrError(
                f"state スロット '{slot}': {STATE_APPEND_OP}（nodes[{appends[0][0]}]）より後に"
                f"読者 nodes[{last[0]}] ({last[1]}) が居る"
                "（append は当該スロットに触れる最後のノード MUST — ADR 0067 決定 5b）"
            )
        first = touched[0]
        mismatch = next((entry for entry in touched if entry[2] != first[2]), None)
        if mismatch is not None:
            raise IrError(
                f"state スロット '{slot}': attrs.window が食い違う"
                f"（{_show_window(first)} / {_show_window(mismatch)}）"
                " — 論理 col → 物理 row の写像は読み書き同式 MUST（ADR 0067 決定 4）"
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
        # MUST: 出力は契約表の**写像の値域**を**出力 slot 別に**見る（cast は attrs.to で
        # 決まるので語彙全体）。入力側の受理集合で代用すると、比較（f32 → bool）や bool の
        # sum（→ i32）のように dtype が変わる op で**正しいグラフが列挙門で落ちる**。
        out_accept: tuple[frozenset[str], ...] = (
            (frozenset(SEMANTIC_DTYPES),)
            if contract.kind == "cast"
            else tuple(frozenset(slot.values()) for slot in contract.output_dtypes)
        )
        for slot, name in enumerate(node.outs):
            # 契約より出力が多い形（出力数違反）は入力側と同様に契約検査の担当。ここは列挙門
            # なので、余ったぶんは全 slot の和で見て 1 件でも多く拾う。
            accept = out_accept[slot] if slot < len(out_accept) else frozenset().union(*out_accept)
            dtype = _declared_dtype(graph, name)
            if dtype not in accept:
                bad_dtypes[name] = dtype
        # MUST: 必須と省略可能の**和**で見る（ADR 0067 の `window`）。省略可能なぶんを落とすと、
        # states 形の正しいグラフが「未実装 attrs」として capability 不足で拒否される
        # （TS 側 RUNTIME_SUPPORT.attrKeys の和と同じ射影）。
        unknown = sorted(
            key
            for key in node.attrs
            if key not in contract.attrs and key not in contract.optional_attrs
        )
        if unknown:
            bad_attrs.append(f"{where}: {', '.join(unknown)}")

    missing_storage: dict[str, list[str]] = {}
    # group 量子化を受理する格納は **i4 だけ**（ADR 0069 決定 2）。他の格納 dtype に付いた
    # group_size は実行経路が無く、黙って無視すると group ごとの scale を per-channel として
    # 読む沈黙誤値になるので、capability 不足で落とす（TS 側 assertRuntimeSupport の鏡像 —
    # ここが無いと「verify は緑・ブラウザだけ落ちる」非対称になる）。
    group_quantized: list[str] = []
    for name, initializer in graph.initializers.items():
        dtype = initializer.storage.dtype
        if dtype not in M0_STORAGE_DTYPES:
            missing_storage.setdefault(dtype, []).append(name)
            continue
        if dtype != "i4" and initializer.storage.group_size is not None:
            group_quantized.append(name)

    if not (missing_ops or bad_dtypes or bad_attrs or missing_storage or group_quantized):
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
    if group_quantized:
        diagnostics.append(
            f"非対応 group 量子化 ({len(group_quantized)}): {', '.join(sorted(group_quantized))}"
            "（group 量子化の格納は i4 のみ — ADR 0069）"
        )
    raise ContainerError(f"ランタイムの capability 不足 — {' / '.join(diagnostics)}")


def assert_op_contracts(graph: IrGraph) -> None:
    """毎ノードの契約検査（アリティ / 宣言出力数 / attrs スキーマ / 入出力 dtype 規則 / shape）と、
    ノード単体では決まらない規則（state effect の順序 — ADR 0067 決定 5b）。

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
            [_declared_dtype(graph, name) for name in node.outs],
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
    # state effect の順序は「ノード単体では決まらない」規則なので、全ノードの契約検査の後に
    # 1 回だけ見る（TS 側 validateGraphContracts の並びと同じ）。
    _assert_state_order(graph)
    # 出力 shape の突合は全ノードの宣言が揃ってから（shapes.py が契約の規則から独立に
    # 計算し、torch の meta 由来の宣言と食い違えば落とす）。
    assert_graph_shapes(graph)


# ---- 配布形（safetensors）との突合 -----------------------------------------

#: safetensors dtype → 1 要素の **bit** 数（サイズ表）。バイト長は `numel × bits / 8` の
#: 厳密一致で見る。正本は TS 側 `packages/runtime/src/format/safetensors.ts` の DTYPE_BITS。
#: MUST: 整列表（READER_DTYPE_ALIGN）と分けて持つ（ADR 0069 決定 2 の 3 面分離）— `I4` は
#: 1 バイトに 2 要素を詰めるので「要素サイズ = 整列」が成り立たない。
READER_DTYPE_BITS = {
    "F32": 32,
    "F16": 16,
    "BF16": 16,
    "I8": 8,
    "I4": 4,
    "U8": 8,
    "I32": 32,
    "U32": 32,
    "I64": 64,
    "BOOL": 8,
}

#: safetensors dtype → テンソル**先頭**に要求する byte 整列（整列表）。`I4` は要素整列の概念を
#: 持たず、展開カーネルが `array<u32>` として束縛する都合で 4（ADR 0069 決定 2）。
READER_DTYPE_ALIGN = {
    "F32": 4,
    "F16": 2,
    "BF16": 2,
    "I8": 1,
    "I4": 4,
    "U8": 1,
    "I32": 4,
    "U32": 4,
    "I64": 8,
    "BOOL": 1,
}

_HEADER_LENGTH_BYTES = 8


def _as_reader_index(value: Any, where: str, what: str) -> int:
    """TS リーダの `asIndex`（`packages/runtime/src/format/safetensors.ts`）と同じ受理集合。

    MUST: 非負であることだけでなく **2^53−1 以下**も見る。ヘッダ JSON の数値は TS 側では
    JS の number として読まれるので、これを超える値は整数として持てず必ず拒否される —
    Python 側が要素数の積だけを見て受理すると、「verify は緑・ブラウザのリーダだけ落ちる」
    ファイルが配布形として残る。bool は int の派生だが添字ではない。
    """
    if isinstance(value, bool) or not isinstance(value, int) or value < 0 or value > MAX_SAFE_INT:
        raise ContainerError(f"{where}: {what} {value!r} が非負整数（2^53−1 以下）でない")
    return value


#: ヘッダ 1 項目が必ず持つキー（TS リーダが読む欄）。
_READER_ENTRY_KEYS = ("dtype", "shape", "data_offsets")


def _as_reader_entry(value: Any, where: str) -> dict[str, Any]:
    """ヘッダ 1 項目の構造を検査する（`_as_reader_index` と同じ流儀の受理集合）。

    MUST: 素で添字しない。3 キーの欠落・項目がオブジェクトでない形・shape が配列でない形は
    `KeyError` / `TypeError` として漏れ、門の診断が「不正なファイル」ではなく「エクスポータが
    壊れた」に見える（ヘッダ長をファイル実長で拘束するのと同じ理由）。`karume verify` は
    外部で作られた safetensors も食う公開 CLI なので、到達経路が実在する。
    """
    if not isinstance(value, dict):
        raise ContainerError(f"{where}: ヘッダ項目がオブジェクトでない: {value!r}")
    missing = [key for key in _READER_ENTRY_KEYS if key not in value]
    if missing:
        raise ContainerError(f"{where}: ヘッダ項目に {missing} が無い")
    if not isinstance(value["shape"], list):
        raise ContainerError(f"{where}: shape が配列でない: {value['shape']!r}")
    return value


@dataclass(frozen=True)
class _StoredTensor:
    """自前リーダが読んだテンソル 1 本の宣言（safetensors dtype と**論理** shape）。"""

    dtype: str
    shape: list[int]


def _read_header(path: str | Path) -> tuple[dict[str, Any], int, int]:
    """ヘッダ JSON と `(データ節の絶対開始位置, データ節のバイト長)` を返す。

    MUST: 宣言長を read へ渡す前にファイル実長で拘束する（`dist.safetensors_header` と
    同型の防御）。u64 をそのまま渡すと規則違反が ContainerError ではなく
    OverflowError / MemoryError として漏れ、門の診断が「不正なファイル」ではなく
    「エクスポータが壊れた」に見える。
    """
    file = Path(path)
    file_size = file.stat().st_size
    with file.open("rb") as handle:
        raw_length = handle.read(_HEADER_LENGTH_BYTES)
        if len(raw_length) != _HEADER_LENGTH_BYTES:
            raise ContainerError(
                f"ファイルが短すぎる: {len(raw_length)} バイト（ヘッダ長すら無い）"
            )
        header_length = int.from_bytes(raw_length, "little")
        if header_length <= 0 or header_length > file_size - _HEADER_LENGTH_BYTES:
            raise ContainerError(
                f"ヘッダ長 {header_length} がファイル長 {file_size} と矛盾している"
            )
        header_bytes = handle.read(header_length)
        data_start = _HEADER_LENGTH_BYTES + header_length

    try:
        header = json.loads(header_bytes)
    except ValueError as cause:
        raise ContainerError(f"safetensors ヘッダ JSON を解析できない: {cause}") from cause
    if not isinstance(header, dict):
        raise ContainerError("safetensors ヘッダが最上位オブジェクトでない")
    return header, data_start, file_size - data_start


def _read_container(path: str | Path) -> tuple[Mapping[str, str], Mapping[str, _StoredTensor]]:
    """配布形を**自前で**読み、`(__metadata__, テンソルキー → 宣言)` を返す。

    MUST: `safetensors` のリーダを通さない。ライブラリ（0.8.0）の dtype 語彙に `I4` が無く、
    packed 4bit を含む配布形は `safe_open` の時点で開けない（ADR 0069 決定 2）— verify は
    Karume のリーダ（`packages/runtime/src/format/safetensors.ts`）の鏡像であるべきなので、
    読み口も自前で持つ。「別のリーダが読めた」は規則の再実装ではない、という
    `assert_reader_layout` と同じ理由でもある。

    NOTE: レイアウト規則（隙間なし・整列・宣言バイト長の一致）は `assert_reader_layout` の
    担当で、呼び出し側（`verify_model`）が**先に**通す。ここは宣言の読み取りだけ。
    """
    header, _, _ = _read_header(path)
    raw = header.get("__metadata__", {})
    if not isinstance(raw, dict) or any(
        not isinstance(key, str) or not isinstance(value, str) for key, value in raw.items()
    ):
        raise ContainerError("__metadata__ が文字列 → 文字列のマップでない")
    tensors: dict[str, _StoredTensor] = {}
    for name, value in header.items():
        if name == "__metadata__":
            continue
        where = f"テンソル '{name}'"
        entry = _as_reader_entry(value, where)
        shape = [
            _as_reader_index(dim, where, f"shape[{axis}] の次元")
            for axis, dim in enumerate(entry["shape"])
        ]
        tensors[name] = _StoredTensor(dtype=entry["dtype"], shape=shape)
    return raw, tensors


def assert_reader_layout(path: str | Path) -> None:
    """Karume のリーダ（`packages/runtime/src/format/safetensors.ts`）が読めるレイアウトかを見る。

    HF の `safe_open` は読めるのに Karume が読めないファイルが作れる — リーダは
    「データ節を隙間なく覆う」「各テンソルの**絶対** offset が dtype の整列単位に整列している」
    を要求し、後者は要素数が奇数の F16（バイト長 ≡ 2 mod 4）の直後に F32 / I32 / I4 を置くと
    破れる（docs/limitations.md）。並び順はエクスポータの責務なので、**書いた側で**
    その責務を果たせているかをここで検査する。

    MUST: この検査は `safetensors` のリーダを通さない（通すと同じ規則の再実装ではなく
    「別のリーダが読めた」だけの主張になる）。ヘッダ JSON を直に読んで規則を写す。
    """
    header, data_start, data_length = _read_header(path)

    declared = []
    for name, entry in header.items():
        if name == "__metadata__":
            continue
        where = f"テンソル '{name}'"
        entry = _as_reader_entry(entry, where)
        dtype = entry["dtype"]
        if dtype not in READER_DTYPE_BITS:
            raise ContainerError(f"{where}: リーダが知らない dtype '{dtype}'")
        offsets = entry["data_offsets"]
        if not isinstance(offsets, list) or len(offsets) != 2:
            raise ContainerError(f"{where}: data_offsets が 2 要素の配列でない: {offsets!r}")
        begin = _as_reader_index(offsets[0], where, "data_offsets")
        end = _as_reader_index(offsets[1], where, "data_offsets")
        count = 1
        for axis, dim in enumerate(entry["shape"]):
            count *= _as_reader_index(dim, where, f"shape[{axis}] の次元")
            # TS 側は積を 1 段ごとに安全整数で見る（elementCount）。積だけを最後に見ると
            # 途中で精度を失った要素数がバイト長と偶然一致する形を通してしまう。
            if count > MAX_SAFE_INT:
                raise ContainerError(f"{where}: 要素数が安全整数 2^53−1 を超える")
        bits = count * READER_DTYPE_BITS[dtype]
        # MUST: bit 総量が byte 境界に乗らない形（I4 の要素数が奇数）は fail loudly。末尾要素が
        # 半バイトだけ突き出すので、テンソルの長さが宣言から一意に決まらない。
        if bits % 8 != 0:
            raise ContainerError(
                f"{where}: {dtype}（1 要素 {READER_DTYPE_BITS[dtype]}bit）の要素数 {count} が"
                " 奇数で byte 境界に乗らない"
            )
        if end - begin != bits // 8:
            raise ContainerError(
                f"{where}: サイズ不一致 offsets={end - begin} "
                f"期待={bits // 8}（{dtype} {entry['shape']}）"
            )
        declared.append((begin, end, name, dtype))

    cursor = 0
    for begin, end, name, dtype in sorted(declared):
        if begin != cursor:
            raise ContainerError(
                f"テンソル '{name}': データ節が隙間なく覆われていない"
                f"（使用済み末尾={cursor} / このテンソルの開始={begin}）"
            )
        align = READER_DTYPE_ALIGN[dtype]
        if (data_start + begin) % align != 0:
            raise ContainerError(
                f"テンソル '{name}': 絶対 offset {data_start + begin} が {dtype} の"
                f" 整列単位 {align} バイトに整列していない"
                "（奇数要素の F16 より後ろに 4 バイト型を置いていないか — "
                "並び順の規約は karume/emit.py）"
            )
        cursor = end
    if cursor != data_length:
        raise ContainerError(f"データ節末尾に未使用領域が {data_length - cursor} バイトある")


def _assert_scale_tensor(
    stored: Mapping[str, _StoredTensor],
    graph: IrGraph,
    name: str,
    scale_key: str,
    weight_shape: list[Any],
    channel_axis: int | None,
    group_size: int | None = None,
) -> None:
    """量子化格納の scale テンソルを実ファイルと突き合わせる
    （`packages/runtime/src/format/container.ts` の鏡像）。

    MUST: 5 点すべてを見る。scale は IR の値ではなく safetensors の**素のテンソル**なので、
    ここを緩めるとロード時（ランタイム）まで誤りが出ない — 「書けたが読めない」ファイルを
    配布物として残さないのが `verify_model` の役目（ADR 0005）。

    1. **実在**する
    2. **F32**（scale を別 dtype のビット列として読むと全チャネルが桁違いの値になる）
    3. 形（`group_size` の有無で 2 通り — ADR 0069 決定 3）
       - per-channel（i8）: **重みと同 rank の keepdim broadcast 形**（各軸は 1 か重みと同値）
       - group（i4）: **重みと同 rank・最終次元だけ group 数**（`last_dim // group_size`）で
         他軸は重みと同値。keepdim broadcast 形とは受理集合が交わらないので**別分岐**
    4. `channel_axis` が決まる（= 適格重み）なら、**その軸だけが伸びた keepdim 形ちょうど**
       （group 形は量子化軸が最終次元に固定されているので 4 の対象外）
    5. **他 initializer の実体との名前衝突が無い**（別の重みを scale として読む形）

    NOTE: 3 と 4 の切り分けはランタイムの 2 経路そのもの。適格外（ホストで f32 展開 —
    `packages/runtime/src/format/i8.ts` の decodeI8）は汎用 keepdim broadcast を stride で
    引くので 3 まで、適格（圧縮のまま GPU 常駐）はカーネルが `wscale[出力チャネル]` と
    平坦に読むので `packages/runtime/src/runtime/executor.ts` の assertChannelScale が 4 を
    要求する。TS 側は container.ts が 3 まで・executor が 4 と層が分かれるが、verify は
    配布形 1 ファイルだけで両層を通せるのでここで両方見る（軸の導出はランタイム
    `plan.ts` の鏡像 = `emit.weight_channel_axes`）。
    """
    where = f"initializer '{name}'"
    view = stored.get(scale_key)
    if view is None:
        raise ContainerError(f"{where}: scale テンソル '{scale_key}' がファイルに無い")
    for other, initializer in graph.initializers.items():
        if initializer.tensor == scale_key:
            raise ContainerError(
                f"{where}: scale テンソル '{scale_key}' が initializer '{other}' の実体と同じキー"
            )
    if view.dtype != "F32":
        raise ContainerError(f"{where}: scale テンソル '{scale_key}' が {view.dtype}（F32 が必要）")
    shape = list(view.shape)
    form = "keepdim broadcast" if group_size is None else "group"
    if len(shape) != len(weight_shape):
        raise ContainerError(
            f"{where}: scale {shape} の rank が重み {weight_shape} と違う（{form} 形が必要）"
        )
    if group_size is not None:
        # 量子化軸（最終次元）が group 数に置き換わった形ちょうど。割り切れることは
        # parse_ir_graph が保証済み（ADR 0069 決定 2）。
        expected = [*weight_shape[:-1], weight_shape[-1] // group_size]
        if shape != expected:
            raise ContainerError(
                f"{where}: scale {shape} が重み {weight_shape} の group 形 {expected}"
                f"（group_size={group_size}）でない"
            )
        return
    if any(dim != 1 and dim != weight_shape[axis] for axis, dim in enumerate(shape)):
        raise ContainerError(f"{where}: scale {shape} が重み {weight_shape} へ broadcast できない")
    if channel_axis is None:
        return
    expected = [dim if axis == channel_axis else 1 for axis, dim in enumerate(weight_shape)]
    if shape != expected:
        raise ContainerError(
            f"{where}: scale {shape} が重み {weight_shape} のチャネル軸 {channel_axis} の"
            f" keepdim 形 {expected} でない（適格重みの scale はカーネルが平坦に引く）"
        )


def verify_model(path: str | Path) -> IrGraph:
    """配布形 1 ファイルを IR v1 の全規則で検証し、読めたグラフを返す。"""
    assert_reader_layout(path)
    metadata, stored = _read_container(path)
    text = metadata.get(IR_METADATA_KEY)
    if text is None:
        raise ContainerError(f"__metadata__.{IR_METADATA_KEY} が無い（Karume モデルではない）")
    graph = parse_ir_graph(text)
    # per-channel scale の受理形は「圧縮のまま GPU 常駐するか」で 2 通りに分かれる
    # （_assert_scale_tensor の 3 / 4）。判定も軸の導出もランタイム
    # （packages/runtime/src/runtime/plan.ts）と同じものを使う — 別実装にすると
    # 「verify は緑・ロードだけ落ちる」がこの 2 経路の境目で復活する。
    eligible = eligible_compressed_initializers(graph)
    try:
        channel_axes = weight_channel_axes(graph)
    except EmitError as cause:
        raise ContainerError(str(cause)) from cause
    for name, initializer in graph.initializers.items():
        where = f"initializer '{name}'"
        view = stored.get(initializer.tensor)
        if view is None:
            raise ContainerError(f"{where}: テンソル '{initializer.tensor}' がファイルに無い")
        expected = STORAGE_ENCODING[initializer.storage.dtype]
        if view.dtype != expected:
            raise ContainerError(
                f"{where}: 格納 dtype '{initializer.storage.dtype}' に対し safetensors 側が"
                f" {view.dtype}（{expected} が必要）"
            )
        declared = graph.values[name].shape
        if declared != view.shape:
            raise ContainerError(f"{where}: 宣言 shape {declared} ≠ 実テンソル {view.shape}")
        scale = initializer.storage.scale
        if scale is not None:
            # group 形の scale を要求するのは格納 i4 だけ（ADR 0069 決定 3）。i8 に付いた
            # group_size は語彙としては通る（実行できないことは assert_runtime_support が
            # 列挙する）ので、形の分岐は group_size の有無ではなく**格納 dtype**で決める。
            _assert_scale_tensor(
                stored,
                graph,
                name,
                scale,
                declared,
                channel_axes.get(name) if name in eligible else None,
                initializer.storage.group_size if initializer.storage.dtype == "i4" else None,
            )
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
