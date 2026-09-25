"""IR の受理規則を Python 側でも全部見る（docs/ir-v2.md）と、コンテナとの合流。

TS 側の正本は packages/runtime/src/format/ir.ts（グラフ単体の規則 — `parseIrDeclarationValue`）・
packages/runtime/src/format/container/*.ts（2 文書の構造・codec 台帳・合流）・
packages/runtime/src/ops/support.ts（ランタイム対応表との突合 — `assertRuntimeSupport`）。
エクスポータが「書けるがランタイムが読めない」ファイルを出さないよう、書き出し経路の最後で
同じ規則を通す。

MUST: ここは fail loudly の門であって近似の場ではない — 未知キーも非正準表記も
黙って無視せず、必ず例外にする（未リリースにつき前方互換チャネルは持たない）。

    uv run karume verify ../../models/karume-irodori-v4.1-small/v4.1-small/dit/model.i8.krm

`parse_ir_graph` は exporter 内部の器（IR v1 — `karume.ir`）の JSON を検証しつつ読む。使う経路は
3 つ: 容器の 2 文書から起こし直した文書の検査（{@link ir_graph_from_container}）・移行 CLI が
旧 shard の `__metadata__` を読む経路（container-v1 §12）・公開面（`karume.__all__`）。
配布形の検証（CLI）はコンテナだけを受ける。
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

from karume.container import (
    BLOCK_TAIL_ALIGN,
    BlockEncoding,
    ContainerFormatError,
    GraphDescriptor,
    ModelDescriptor,
    ReadContainer,
    codec_entry,
    container_parts,
    group_count,
    payload_bytes,
    per_channel_group_size,
    read_container,
)
from karume.dims import MAX_SAFE_INT, DimError, is_symbol_name, parse_dim, try_parse_dim
from karume.ir import (
    IR_FORMAT,
    IR_VERSION,
    MIN_GROUP_SIZE,
    IrDim,
    IrGraph,
    IrInitializer,
    IrInput,
    IrNode,
    IrShared,
    IrState,
    IrStorage,
    IrValue,
)
from karume.ops import (
    ATTENTION_OP,
    IO_DTYPES,
    M0_STORAGE_DTYPES,
    OP_CONTRACTS,
    STATE_APPEND_OP,
    STRIDED_RANK_OPS,
    OpContractError,
    assert_node_contract,
    assert_strided_rank,
    attention_readonly,
    resolve_node_dtypes,
    state_window,
    sym_prefix_slice_attrs,
)
from karume.shapes import assert_graph_shapes, declared_shape


class IrError(ValueError):
    """グラフ JSON 単体で決まる規則の違反。"""


class ContainerError(ValueError):
    """safetensors のレイアウト規則の違反・capability 不足・容器 → IR 文書の変換の不整合、
    および IR 受理規則違反の包み（{@link assert_ir_accepted}）。

    safetensors のレイアウト規則は移行 CLI の入力（旧 shard）と資産に掛かる。合流規則
    （2 文書 + 束縛表）の違反はこの型ではなく `karume.container.ContainerFormatError`
    （TS 側と同名の型）で出る — 型で分岐する呼び手は両方を捕まえる。
    """


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
STORAGE_DTYPES = ("f32", "f16", "bf16", "i8", "i4", "i2", "i32")

#: scale / group_size の記述子を持てる格納 dtype（量子化格納）。TS 側は codec 台帳
#: （`packages/runtime/src/format/container/codecs.ts` の CODEC_LEDGER）で `scale: "required"` の
#: codec の layout がこの 3 語に当たる。
QUANTIZED_STORAGE_DTYPES = ("i8", "i4", "i2")

#: state スロットの dtype 語彙。現状 f32 のみ（ADR 0066 決定 2）。
STATE_DTYPES = ("f32",)
#: 席だけが予約されている state スロットの dtype（ADR 0066 追記 5）。f16 格納は数値契約が
#: 変わるので ADR 0058 流儀の opt-in が要る — それが無いうちは「語彙外」ではなく**未対応**。
RESERVED_STATE_DTYPES = ("f16",)
#: state スロットの rank 上限（ADR 0066 決定 2 の「固定 rank（rank ≤ 4）」）。strided カーネルの
#: 上限（ops.STRIDED_RANK）と同じ数値だが理由が別なので定数を共有しない。
MAX_STATE_RANK = 4

#: initializer の意味論 dtype → 許される格納 dtype（docs/ir-v2.md「値と型」）。
#: MUST: 交差を許さない — `i32` 宣言の initializer が f16 のビット列として読まれる
#: 沈黙誤値になる。bool の initializer は語彙に無い。
INITIALIZER_STORAGE = {"f32": ("f32", "f16", "bf16", "i8", "i4", "i2"), "i32": ("i32",)}


# ---- JSON 層 --------------------------------------------------------------


def parse_graph_json(text: str) -> Any:
    """グラフ JSON を読む。非有限値を含むものは受理しない。

    Python の json は `Infinity` / `NaN` リテラルを既定で受理するが、ブラウザの
    JSON.parse は落ちる。加えて `1e999` は構文として有効なまま Infinity へ丸まるので、
    リテラル名だけでなく**値レベル**で弾く（docs/ir-v2.md）。
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


def _integral_float_to_int(value: Any) -> Any:
    """JSON の `1.0` / `1e0` のような整数値の float を int へ正規化する（他の値はそのまま）。

    MUST: 整数を読む欄（次元・group 長）は全てこれを通す。TS 側は JSON.parse が単一の number を
    返すので整数値の float という区別が無く（`Number.isSafeInteger(1.0)` は true）、Python が
    float を丸ごと拒むと**ランタイムが読める graph をエクスポータの検証だけが読めない**
    乖離になる（受理集合はどちらの向きにもずれてはいけない）。非整数値の float は
    `is_integer()` が False なので呼び手の型検査で従来どおり落ち、非有限値はそもそも JSON 読みの
    時点で弾かれる。safe range 超過は呼び手の int 側の検査に載る。
    """
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def _parse_shape(value: Any, symbols: set[str], where: str) -> list[int | str]:
    shape: list[int | str] = []
    for index, dim in enumerate(_as_array(value, where)):
        at = f"{where}[{index}]"
        # bool は int の派生だが次元ではない。
        if isinstance(dim, bool):
            raise IrError(f"{at}: 次元が数値でも文字列でもない")
        # safe range 超過は下の int 側の検査（`> MAX_SAFE_INT`）に載る = TS の
        # `Number.isSafeInteger` と同じ受理集合になる。
        dim = _integral_float_to_int(dim)
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

    省略可能キー `external`（ADR 0096 段 2）は「実体を自分で確保しない = 借り先 context の
    スロットを名前で引く」宣言。MUST: 受理するのは `true` だけ — `false` は欄の不存在と同義で、
    同じ意味に 2 通りの綴りを作ると「どちらの規則で読むべきか」が宣言から決まらなくなる。
    """
    obj = _as_object(value, where)
    _check_keys(obj, ["dtype", "shape"], ["external"], where)
    dtype = _as_state_dtype(obj["dtype"], f"{where}.dtype")
    shape = _parse_shape(obj["shape"], symbols, f"{where}.shape")
    external = False
    if "external" in obj:
        if obj["external"] is not True:
            raise IrError(
                f"{where}.external: true のみ（{obj['external']!r} — false は欄の不存在と同義）"
            )
        external = True
    if not 1 <= len(shape) <= MAX_STATE_RANK:
        raise IrError(
            f"{where}.shape: rank {len(shape)} は 1..{MAX_STATE_RANK} の外"
            "（固定 rank の容量込み具体形 MUST）"
        )
    for index, dim in enumerate(shape):
        if isinstance(dim, int) and dim < 1:
            raise IrError(f"{where}.shape[{index}]: 次元 {dim} が正整数でない（容量が取れない）")
    return IrState(dtype=dtype, shape=shape, external=external)


def _parse_storage(value: Any, where: str, *, shared: bool = False) -> IrStorage:
    """格納の記述子。

    `shared`（ADR 0096 段 2 の共有 initializer）は **`dtype` の 1 キーだけ**を持つ: 借り手の
    容器にはバイトが 1 つも無いので、付随実体を記述する欄（`scale`）も group の刻み
    （`group_size`）も**貸し手側だけが持つ**。写すと同じ事実が 2 箇所に生え、どちらで dequant
    するかが宣言から決まらない。i8 / i4 の「scale 必須」「i4 は group_size 必須」も
    この側には掛からない（掛けると共有宣言が原理的に書けない）。
    """
    obj = _as_object(value, where)
    _check_keys(obj, ["dtype"], ["scale", "group_size"], where)
    dtype = _as_storage_dtype(obj["dtype"], f"{where}.dtype")
    has_scale = "scale" in obj
    has_group_size = "group_size" in obj
    if shared and (has_scale or has_group_size):
        extra = sorted(key for key in ("scale", "group_size") if key in obj)
        raise IrError(
            f"{where}: 共有 initializer は {extra} を宣言できない"
            "（バイトを持たない側なので、付随実体と group の刻みは貸し手の常駐重みが正本）"
        )
    # scale / group_size は量子化格納の記述子。非量子化 dtype に付いているのは
    # エクスポータの取り違えなので受理しない（黙って無視すると格納の意味が二重化する）。
    if dtype not in QUANTIZED_STORAGE_DTYPES and (has_scale or has_group_size):
        raise IrError(f"{where}: 格納 dtype '{dtype}' に scale / group_size は付けられない")
    # MUST: i8 / i4 は scale を**明示宣言**する
    # （ADR 0019 / 0069・TS 側は codec 台帳の `scale: "required"` と
    # packages/runtime/src/format/container/descriptor.ts の parseEncoding の鏡像）。
    # 既定 1.0 で補完すると、scale の書き忘れが「全チャネル 1.0 で dequant した重み」に化けて
    # ロードも実行も通ってしまう（差が O(scale) で出るのに、どこにも例外が出ない）。
    if dtype in QUANTIZED_STORAGE_DTYPES and not has_scale and not shared:
        raise IrError(f"{where}: 格納 dtype '{dtype}' には scale（scale テンソルのキー）が要る")
    # MUST: i4 は group_size を**明示宣言**する（ADR 0069 決定 2）。group 長が決まらない
    # 4bit 格納は scale の引き直し位置が決まらず、展開が黙って別の値を出す。
    if dtype == "i2" and has_group_size:
        raise IrError(f"{where}: i2 は行ごとの scale のみ（group_size は付けられない）")
    if dtype == "i4" and not has_group_size and not shared:
        raise IrError(f"{where}: 格納 dtype 'i4' には group_size が要る（ADR 0069 決定 2）")
    scale = _as_nonempty_str(obj["scale"], f"{where}.scale") if has_scale else None
    group_size = None
    if has_group_size:
        raw = _integral_float_to_int(obj["group_size"])
        if isinstance(raw, bool) or not isinstance(raw, int) or raw < 1:
            raise IrError(f"{where}.group_size: 正整数でない")
        # TS 側は JSON の数値として読むので 2^53−1 を超える値は整数として持てない
        # （packages/runtime/src/format/container/descriptor.ts の parseEncoding — `groupSize` は
        # 安全整数のみ受理）。ここで受理するとランタイムだけが落ちる。
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
    """グラフ JSON（IR v1 の器）を検証しつつ読む。

    TS 側はグラフ単体の規則を packages/runtime/src/format/ir.ts の parseIrDeclarationValue、格納の
    規則（scale 必須・groupSize など）を packages/runtime/src/format/container/descriptor.ts の
    parseEncoding と packages/runtime/src/format/container/bind.ts の束縛（bindGraphs）が持つ
    （それらを合わせたものと同義）。bind.ts の mergedGraph は合流の結果を IrGraph に写すだけで、
    検査はしない。
    """
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
        # 空の値名は拒否する（TS 側 packages/runtime/src/format/ir.ts の鏡像）— 参照側
        # （入力名・ノードの入出力・グラフ出力）はどれも空でない文字列しか受理しないので、
        # 通すと「宣言はできるが原理的に参照できない値」になる。空名の values と空名の
        # initializers を**対で**書かれると孤立宣言の門が互いに満たされて発火しないため、
        # ここを緩めると参照不能な実テンソルが配布形に居座る。
        _as_nonempty_str(name, "graph.values の値名")
        where = f"graph.values['{name}']"
        obj = _as_object(raw, where)
        _check_keys(obj, ["dtype", "shape"], [], where)
        values[name] = IrValue(
            dtype=_as_semantic_dtype(obj["dtype"], f"{where}.dtype"),
            shape=_parse_shape(obj["shape"], symbol_set, f"{where}.shape"),
        )

    initializers: dict[str, IrInitializer] = {}
    for name, raw in _as_object(root["initializers"], "graph.initializers").items():
        # 空の initializer 名も同じ理由で拒否する（上の values と対 — 片方だけ塞ぐと
        # 「空名の対」がそのまま通る）。
        _as_nonempty_str(name, "graph.initializers の initializer 名")
        where = f"graph.initializers['{name}']"
        obj = _as_object(raw, where)
        # 共有 initializer（ADR 0096 段 2）は `tensor` の代わりに `shared` を持つ。**排他**で、
        # 両方書かれた形は「バイトを持つのか借りるのか」が宣言から決まらない
        # （`_check_keys` の必須キー集合を切り替えることが、そのまま排他の執行になる）。
        if "shared" in obj:
            _check_keys(obj, ["shared", "storage"], [], where)
            shared_obj = _as_object(obj["shared"], f"{where}.shared")
            _check_keys(shared_obj, ["tensor"], [], f"{where}.shared")
            initializers[name] = IrInitializer(
                shared=IrShared(
                    tensor=_as_nonempty_str(shared_obj["tensor"], f"{where}.shared.tensor")
                ),
                storage=_parse_storage(obj["storage"], f"{where}.storage", shared=True),
            )
            continue
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
    TS 側は合流層 packages/runtime/src/format/container/bind.ts の group の刻みの検査）。

    量子化軸は**格納行**（先頭次元を除く残りの平坦化 — linear `[O,I]` の in 軸・embedding
    `[V,D]` の D 軸・conv1d `[Cout,Cin,K]` の受容野 `Cin·K`）で、その行長が `group_size` で
    割り切れることが MUST（rank2 の重みでは「最終次元」と同値 — ADR 0069 決定 3 の rank 非依存
    規則）。端数 group を許すと最後の group だけ scale の担当範囲が短くなり、行境界が語境界から
    ずれて平坦添字の展開が黙って別の値を出す（端数を作らない制約で整列問題そのものを消す設計）。
    """
    # 値域（2 冪かつ 16 以上）は _parse_storage が保証済み。存在は型の上でだけ optional なので、
    # 黙って読み飛ばさず言い直す（TS 側 checkGroupQuantizedShape と同じ流儀）。
    group_size = initializer.storage.group_size
    if group_size is None:
        raise IrError(f"graph.initializers['{name}']: 格納 i4 なのに group_size が無い")
    if len(value.shape) < 2:
        raise IrError(
            f"graph.values['{name}']: 格納 i4 の initializer に量子化軸が無い"
            f"（rank {len(value.shape)} — 行軸と量子化軸で rank 2 以上が要る）"
        )
    row_length = 1
    for dim in value.shape[1:]:
        if not isinstance(dim, int):
            raise IrError(f"graph.values['{name}']: 格納 i4 の shape に記号次元は使えない")
        row_length *= dim
    if row_length % group_size != 0:
        raise IrError(
            f"graph.values['{name}']: 格納 i4 の行長 {row_length} が"
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
        # initializer は束縛前に確定していなければ束縛表の block 長と突合できない。
        if any(not isinstance(dim, int) for dim in values[name].shape):
            raise IrError(f"graph.values['{name}']: initializer の shape に記号次元は使えない")
        if storage_dtype == "i2":
            shape = values[name].shape
            if len(shape) != 2 or any(dim <= 0 for dim in shape) or shape[1] % 16:
                raise IrError(f"graph.values['{name}']: i2 は正の rank 2・行長は16の倍数が必要")
        # 共有 initializer は group_size を宣言できない（_parse_storage）— group の刻みは
        # 貸し手の常駐重みが正本で、貸し手の容器が合流層で見る（ADR 0096 段 2）。
        if storage_dtype == "i4" and not initializers[name].is_shared:
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

    external スロット（ADR 0096 段 2）には**さらに 3 点**が乗る（TS 側 `plan.ts` の
    `validateGraphContracts` / `assertStateOrder` と同じ規律）:

    4. external と自前スロットの**混在は拒否**（external が 1 本でもあるグラフは全スロットが
       external）。借り手 context は自前スロットを持たないので、混ざった宣言は「どちらの
       確保規則で作るか」がグラフから決まらない
    5. external への `state_append` は **0 本**（実体は貸し手のもの — 借り手が書くと、貸し手の
       論理長を進めないまま物理 ring を汚す）
    6. external の読者は **readonly attention だけ** / 逆に readonly が読むスロットは
       **external だけ**。普通の states 形 attention が external を読む形は「今 step の k/v を
       貸し手のスロットへ足したうえで読む」意味になり、5 と正面から食い違う

    MUST: fail loudly。3 点とも「順序 / 宣言の誤り」が例外ではなく**別の値**として出る種類の
    破れなので、書き出しの時点でしか止められない。
    """
    external = {name for name, slot in graph.states.items() if slot.external}
    if external and external != set(graph.states):
        owned = sorted(set(graph.states) - external)
        raise IrError(
            f"state スロット: external {sorted(external)} と自前 {owned} が混在している"
            "（external が 1 本でもあるグラフは全スロットが external MUST — ADR 0096 段 2）"
        )
    touches: dict[str, list[tuple[int, str, int | None]]] = {}
    for index, node in enumerate(graph.nodes):
        if not node.states:
            continue
        where = f"nodes[{index}] ({node.op})"
        # attrs の値域検査は assert_node_contract が済ませている（ここは引き直すだけ）。
        window = state_window(node.attrs, where)
        readonly = node.op == ATTENTION_OP and attention_readonly(node.attrs)
        for slot in node.states.values():
            touches.setdefault(slot, []).append((index, node.op, window))
            if slot in external and not readonly:
                raise IrError(
                    f"state スロット '{slot}' は external なのに {where} が readonly でない形で"
                    "触れている（読者は readonly attention だけ・書き込みは 0 本 MUST —"
                    " ADR 0096 段 2）"
                )
            if readonly and slot not in external:
                raise IrError(
                    f"{where}: readonly attention が自前スロット '{slot}' を読んでいる"
                    "（readonly が読めるのは external スロットだけ — ADR 0096 段 2）"
                )
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
    """M0 ランタイムが実行できる形かを突合する（packages/runtime/src/ops/support.ts の
    assertRuntimeSupport と同義）。

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
    # 読む沈黙誤値になるので、capability 不足で落とす。TS 側は合流層（bind.ts）がこの形を拒む:
    # codec 台帳の `grouping` が "channel" の codec では groupSize = 行長 MUST（group を持つ
    # codec は int4-sym-g だけ）。容器から起こした
    # 文書（{@link ir_graph_from_container}）は group codec にしか group_size を出さないので
    # ここに届かず、届くのは v1 文書を直に渡す経路だけ。
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


# ---- safetensors のレイアウト（旧 shard・資産）------------------------------

#: safetensors dtype → 1 要素の **bit** 数（サイズ表）。バイト長は `numel × bits / 8` の
#: 厳密一致で見る。TS 側 `packages/runtime/src/format/safetensors.ts` の DTYPE_BYTES の
#: **上位集合**で、方言 dtype の `I4` / `I2` はこの表だけが持つ（TS の読み手は受理しない —
#: 旧 shard を読む移行 CLI が要る）。受理は {@link _LEGACY_ONLY_DTYPES} が移行入力に限る。
#: MUST: 整列表（READER_DTYPE_ALIGN）と分けて持つ（ADR 0069 決定 2 の 3 面分離）— `I4` は
#: 1 バイトに 2 要素を詰めるので「要素サイズ = 整列」が成り立たない。
READER_DTYPE_BITS = {
    "F32": 32,
    "F16": 16,
    "BF16": 16,
    "I8": 8,
    "I4": 4,
    "I2": 2,
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
    "I2": 4,
    "U8": 1,
    "I32": 4,
    "U32": 4,
    "I64": 8,
    "BOOL": 1,
}

#: TS の読み手が拒否する方言 dtype（旧 shard の packed 4bit / 2bit）。`assert_reader_layout` は
#: 移行 CLI の入力検査（`allow_legacy_dtypes=True`）でだけ受理する — 資産の門で受理すると
#: 「Python の門は緑・ブラウザのリーダだけ落ちる」資産が書ける。
_LEGACY_ONLY_DTYPES = frozenset({"I4", "I2"})

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
    壊れた」に見える（ヘッダ長をファイル実長で拘束するのと同じ理由）。`karume migrate` は
    外部で作られた旧 shard を食う公開 CLI なので、到達経路が実在する。
    """
    if not isinstance(value, dict):
        raise ContainerError(f"{where}: ヘッダ項目がオブジェクトでない: {value!r}")
    missing = [key for key in _READER_ENTRY_KEYS if key not in value]
    if missing:
        raise ContainerError(f"{where}: ヘッダ項目に {missing} が無い")
    if not isinstance(value["shape"], list):
        raise ContainerError(f"{where}: shape が配列でない: {value['shape']!r}")
    return value


def _read_header(path: str | Path) -> tuple[dict[str, Any], int, int]:
    """ヘッダ JSON と `(データ節の絶対開始位置, データ節のバイト長)` を返す。

    MUST: 宣言長を read へ渡す前にファイル実長で拘束する（`legacy.safetensors_header` と
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


def assert_reader_layout(path: str | Path, *, allow_legacy_dtypes: bool = False) -> None:
    """safetensors のレイアウト規則を見る（TS 側 `format/safetensors.ts` と同じ規則）。

    HF の `safe_open` は読めるのに Karume が読めないファイルが作れる — リーダは
    「データ節を隙間なく覆う」「各テンソルの**絶対** offset が dtype の整列単位に整列している」
    を要求し、後者は要素数が奇数の F16（バイト長 ≡ 2 mod 4）の直後に F32 / I32 / I4 を置くと
    破れる（docs/limitations.md）。使う経路は 2 つ: 移行 CLI が旧 shard を読む前の入力検査
    （`karume.legacy`）と、資産の safetensors を書いた直後の門（recipe の資産の書き手）。
    方言 dtype の `I4` / `I2` は `allow_legacy_dtypes=True`（移行 CLI の入力検査）のときだけ
    受理する。既定は TS の読み手と同じ dtype 集合（資産の門）。

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
        if dtype in _LEGACY_ONLY_DTYPES and not allow_legacy_dtypes:
            raise ContainerError(
                f"{where}: リーダが知らない dtype '{dtype}'"
                "（方言 dtype — 旧 shard の移行入力でだけ受理）"
            )
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


# ---- コンテナ形式（krm / krg）との突合 -------------------------------------
#
# TS 側 `packages/runtime/src/format/container/bind.ts`（合流層）の鏡像。descriptor 単体で
# 決まる規則（2 文書の構造・block の配置・束縛表の過不足）は読み手
# （{@link karume.container.read_container}）が既に見ているので、ここが掛けるのは
# **宣言 shape を要する規則だけ**である:
#
# - 意味論 dtype と codec の組（`f32` の符号化 / `i32` は生の int32 — 交差は fail loudly）
# - payload 長 = 宣言 shape と packing から決まる値。block 長との差は詰め物（0 以上 4 未満）だけ
# - piece 列の末尾 = `shape[0]`。中間 piece に詰め物は無い
# - `rowAxis != 0` の initializer は piece 分割不可
# - group の刻み（per-channel は行長 / group codec は 2 冪 ≥ 16 で行長を割る）と scale 長
# - i2 経路（`int2-off` / `ternary`）の宣言 shape は正の rank 2 で行長が 16 の倍数


@dataclass(frozen=True)
class SupplyBlock:
    """実体 1 本ぶんの block（piece 列なら 1 piece）。"""

    id: str
    part: int
    offset: int
    length: int
    #: この block が運ぶ先頭次元の行範囲（丸ごと 1 本なら `[0, shape[0]]`・rank 0 は `[0, 1]`）。
    rows: tuple[int, int]
    #: payload のバイト長（block 長から末尾の詰め物を除いたもの）。
    payload_bytes: int


@dataclass(frozen=True)
class InitializerSupply:
    """initializer 1 本の供給計画（実体をどの block から取るか）。"""

    encoding: BlockEncoding
    #: 実体の block 列（丸ごとなら 1 本）。
    blocks: tuple[SupplyBlock, ...]
    #: 供給元。const 領域（part 1・グラフの所有）か重み側（モデル記述）か。
    origin: Literal["const", "model"]
    #: companion scale の block（量子化 codec のみ）。
    scale: SupplyBlock | None = None


@dataclass(frozen=True)
class BoundGraph:
    """グラフ 1 本の合流結果（宣言 + shared でない initializer 全部の供給計画）。"""

    declaration: Mapping[str, Any]
    supplies: Mapping[str, InitializerSupply]


def _located_block(
    graph: GraphDescriptor, model: ModelDescriptor | None
) -> Callable[[str, str], SupplyBlock]:
    """block id → 在処（const 目次は part 1・モデル目次は宣言の part）。"""
    const = {block.id: block for block in graph.const_blocks}
    data = {block.id: block for block in (model.blocks if model is not None else ())}

    def locate(block_id: str, where: str) -> SupplyBlock:
        found = const.get(block_id)
        part = 1
        if found is None:
            record = data.get(block_id)
            if record is None:
                raise ContainerFormatError(f"{where}: 未宣言の block '{block_id}'")
            found, part = record, record.part
        # rows / payload_bytes は呼び手（宣言 shape を知る側）が埋める。
        return SupplyBlock(found.id, part, found.offset, found.length, (0, 0), 0)

    return locate


def _declared_tensor(
    declaration: Mapping[str, Any], name: str, where: str
) -> tuple[str, list[int]]:
    """initializer の意味論 dtype と**具体 shape**（記号次元を持つ実体は在りえない）。"""
    values = declaration.get("values")
    if not isinstance(values, dict) or not isinstance(values.get(name), dict):
        raise ContainerFormatError(f"{where}: `values` に dtype / shape 宣言が無い")
    value = values[name]
    dtype = value.get("dtype")
    if dtype not in SEMANTIC_DTYPES:
        raise ContainerFormatError(f"{where}: 意味論 dtype が語彙外: {dtype!r}")
    raw = value.get("shape")
    if not isinstance(raw, list):
        raise ContainerFormatError(f"{where}: `values` の shape が配列でない")
    shape: list[int] = []
    for raw_dim in raw:
        # 容器の JSON 読みは float を int に変えない — graph の `_parse_shape` と同じ受理集合に
        # 揃える（ランタイムは `4.0` を次元 4 として読む）。
        dim = _integral_float_to_int(raw_dim)
        if not isinstance(dim, int) or isinstance(dim, bool) or dim < 0:
            raise ContainerFormatError(
                f"{where}: initializer の shape に記号次元は使えない（{dim!r}）"
            )
        shape.append(dim)
    return dtype, shape


def _assert_padded(block: SupplyBlock, payload: int, where: str) -> None:
    """block 長と payload 長の関係（container-v1 §4.1 — 詰め物は 0 以上 4 未満）。"""
    if not 0 <= block.length - payload < BLOCK_TAIL_ALIGN:
        raise ContainerFormatError(
            f"{where}: block '{block.id}' の長さ {block.length} が payload {payload} バイト +"
            f" 詰め物（{BLOCK_TAIL_ALIGN} 未満）でない"
        )


def _is_i2_shape(shape: Sequence[int]) -> bool:
    """i2 経路の論理形 `[rows, width]`（`packages/runtime/src/format/i2.ts` の鏡像）。"""
    return len(shape) == 2 and all(dim > 0 for dim in shape) and shape[1] % 16 == 0


def _plan_pieces(
    where: str,
    shape: Sequence[int],
    payload: int,
    encoding: BlockEncoding,
    pieces: Sequence[tuple[str, tuple[int, int]]],
    locate: Callable[[str, str], SupplyBlock],
) -> list[SupplyBlock]:
    """piece 列の供給計画（行範囲 → バイト範囲）。並びと被覆は読み手が見た後の段。"""
    if encoding.row_axis not in (None, 0):
        raise ContainerFormatError(
            f"{where}: rowAxis {encoding.row_axis} の initializer は piece 分割できない（規則④）"
        )
    rows = shape[0] if shape else 1
    if not shape or rows == 0 or payload % rows != 0:
        raise ContainerFormatError(
            f"{where}: payload {payload} バイトが先頭次元 {rows} 行で割り切れないので"
            " piece 分割できない"
        )
    if pieces[-1][1][1] != rows:
        raise ContainerFormatError(
            f"{where}: piece 列の末尾 {pieces[-1][1][1]} 行が宣言 shape の先頭次元 {rows} 行と"
            "違う（規則④）"
        )
    row_bytes = payload // rows
    planned: list[SupplyBlock] = []
    for index, (block_id, (begin, end)) in enumerate(pieces):
        by = f"{where} piece[{index}]"
        found = locate(block_id, by)
        piece_bytes = (end - begin) * row_bytes
        if index < len(pieces) - 1:
            # 中間 piece に詰め物は掛けられない（次の piece の先頭を潰す）。
            if found.length != piece_bytes:
                raise ContainerFormatError(
                    f"{by}: 中間 piece の block 長 {found.length} が行範囲のバイト数"
                    f" {piece_bytes} と違う（詰め物不可）"
                )
        else:
            _assert_padded(found, piece_bytes, by)
        planned.append(replace(found, rows=(begin, end), payload_bytes=piece_bytes))
    return planned


def _plan_supply(
    where: str,
    dtype: str,
    shape: Sequence[int],
    encoding: BlockEncoding,
    block: str | None,
    pieces: Sequence[tuple[str, tuple[int, int]]] | None,
    locate: Callable[[str, str], SupplyBlock],
    origin: Literal["const", "model"],
) -> InitializerSupply:
    """1 initializer ぶんの供給計画（宣言 shape × encoding × block 目次）。"""
    entry = codec_entry(encoding.codec)
    if entry.layout not in INITIALIZER_STORAGE.get(dtype, ()):
        raise ContainerFormatError(
            f"{where}: 意味論 dtype '{dtype}' に codec '{encoding.codec}' は組めない"
        )
    if entry.layout == "i2" and not _is_i2_shape(shape):
        raise ContainerFormatError(
            f"{where}: codec '{encoding.codec}' は正の rank 2・行長 16 の倍数の宣言 shape が要る"
            f"（[{','.join(str(dim) for dim in shape)}]）"
        )
    numel = math.prod(shape)
    payload = payload_bytes(encoding.codec, numel, where)
    rows = shape[0] if shape else 1
    if pieces is None:
        if block is None:
            raise ContainerFormatError(f"{where}: 供給形が `block` でも `pieces` でもない")
        found = locate(block, where)
        _assert_padded(found, payload, where)
        blocks = [replace(found, rows=(0, rows), payload_bytes=payload)]
    else:
        blocks = _plan_pieces(where, shape, payload, encoding, pieces, locate)
    if entry.scale == "forbidden":
        return InitializerSupply(encoding, tuple(blocks), origin)

    if encoding.row_axis is None or encoding.group_size is None or encoding.scale_block is None:
        raise ContainerFormatError(
            f"{where}: codec '{encoding.codec}' は量子化なので rowAxis / groupSize / scale が要る"
        )
    row_axis, group_size = encoding.row_axis, encoding.group_size
    if not shape or len(shape) <= row_axis:
        raise ContainerFormatError(
            f"{where}: rowAxis {row_axis} に対して宣言 shape"
            f" [{','.join(str(dim) for dim in shape)}] の rank が足りない"
        )
    row_count = shape[row_axis]
    row_length = numel // row_count if row_count else 0
    if entry.grouping == "channel" and group_size != per_channel_group_size(row_length):
        raise ContainerFormatError(
            f"{where}: codec '{encoding.codec}' は per-channel なので groupSize は行長"
            f" {per_channel_group_size(row_length)} に等しい MUST（宣言は {group_size}）"
        )
    if entry.grouping == "group":
        if group_size < MIN_GROUP_SIZE or group_size & (group_size - 1) != 0:
            raise ContainerFormatError(
                f"{where}: groupSize {group_size} が 2 冪かつ {MIN_GROUP_SIZE} 以上でない"
                "（ADR 0069 決定 2）"
            )
        if row_length % group_size != 0:
            raise ContainerFormatError(
                f"{where}: 行長 {row_length}（= numel / shape[{row_axis}]）が groupSize"
                f" {group_size} で割り切れない（ADR 0069 決定 2）"
            )
    scale_bytes = row_count * group_count(row_length, group_size) * 4
    scale = locate(encoding.scale_block, f"{where} scale")
    _assert_padded(scale, scale_bytes, f"{where} scale")
    return InitializerSupply(
        encoding,
        tuple(blocks),
        origin,
        replace(scale, rows=(0, row_count), payload_bytes=scale_bytes),
    )


def bind_graphs(graph: GraphDescriptor, model: ModelDescriptor | None) -> dict[str, BoundGraph]:
    """グラフ記述と束縛表を合流し、initializer ごとの供給計画を決める（TS `bindGraphs` の鏡像）。

    `model` が `None`（`krg`）のときは const 供給だけが埋まる — 重みが要る initializer は
    供給を持たない宣言として残る（`krg` だけでは Session を組めない）。
    """
    locate = _located_block(graph, model)
    constants = {(entry.graph, entry.initializer): entry for entry in graph.constants}
    bound: dict[str, BoundGraph] = {}
    for graph_name, declaration in graph.graphs.items():
        initializers = declaration.get("initializers")
        if not isinstance(initializers, dict):
            raise ContainerFormatError(f"graph '{graph_name}': `initializers` 節が無い")
        supplies: dict[str, InitializerSupply] = {}
        for name, init in initializers.items():
            if isinstance(init, dict) and init.get("shared") is True:
                continue
            where = f"graph '{graph_name}' initializer '{name}'"
            dtype, shape = _declared_tensor(declaration, name, where)
            constant = constants.get((graph_name, name))
            if constant is not None:
                supplies[name] = _plan_supply(
                    where, dtype, shape, constant.encoding, constant.block, None, locate, "const"
                )
                continue
            supply = (model.binding.get(graph_name, {}) if model is not None else {}).get(name)
            if supply is None:
                if model is None:
                    continue
                raise ContainerFormatError(f"{where}: 束縛表に供給が無い")
            supplies[name] = _plan_supply(
                where, dtype, shape, supply.encoding, supply.block, supply.pieces, locate, "model"
            )
        bound[graph_name] = BoundGraph(declaration, supplies)
    return bound


#: 供給を持たない initializer の格納（意味論 dtype → 生の格納）。共有宣言（ADR 0096 段 2）と
#: `krg`（束縛表そのものが無い）だけが該当する席で、どちらも**この容器に実体が無い**。
_PLAIN_STORAGE_FOR: Mapping[str, str] = {"f32": "f32", "i32": "i32"}


def _storage_document(encoding: BlockEncoding) -> dict[str, Any]:
    """束縛表の `encoding` → IR v1 の `storage` 記述子（{@link karume.container.Encoding} の逆）。

    `scale` は v1 では「scale テンソルのキー」だが、容器では scale は block なので **block id**
    をそのまま綴る（診断が現物の名前を指す）。`group_size` を出すのは group codec だけ — 旧
    per-channel の宣言は `group_size` を持たず、容器側の `groupSize`（= 行長）は台帳から導ける
    写しなので、戻すと「i8 に group_size が付いた」形（ランタイム支援の門が落とす形）になる。
    """
    entry = codec_entry(encoding.codec)
    document: dict[str, Any] = {"dtype": entry.layout}
    if entry.scale == "required":
        document["scale"] = encoding.scale_block
        if entry.grouping == "group":
            document["group_size"] = encoding.group_size
    return document


def ir_graph_from_container(read: ReadContainer, graph_name: str) -> IrGraph:
    """コンテナの 2 文書 → IR v1 の {@link IrGraph}（`parse_ir_graph` の検査群を全部通す）。

    グラフ記述が持つのは **IR v2 の文書**（docs/ir-v2.md — initializer 名がテンソルキーで、
    格納は束縛表へ出ている）なので、v1 との差は `version` と `initializers` の 2 点だけである。
    ここが戻すのは「その 2 点を埋めた v1 文書」を {@link parse_ir_graph} へ通した結果で、
    宣言の完全性・SSA・前方参照・記号の束縛可能性・state スロットの規則は**同じ 1 本**が見る
    （v2 用に検査を書き写すと、規則が動いた日に片方だけが古びる）。

    格納の正本は**束縛表**（`encoding.codec`）で、供給を持たない席（共有宣言と `krg`）だけは
    意味論 dtype の生の格納を置く — その席はこの容器に 1 バイトも持たないので、格納の主張は
    貸し手の容器（と、その容器に掛かる同じ門）が持つ。
    """
    declaration = read.graph.graphs.get(graph_name)
    if declaration is None:
        raise ContainerError(
            f"コンテナにグラフ '{graph_name}' が無い（宣言: {sorted(read.graph.graphs)}）"
        )
    where = f"graph '{graph_name}'"
    supplies = bind_graphs(read.graph, read.model)[graph_name].supplies
    root = _as_object(declaration, where)
    values = _as_object(root.get("values"), f"{where}.values")
    declared = _as_object(root.get("initializers"), f"{where}.initializers")
    initializers: dict[str, Any] = {}
    for name, raw in declared.items():
        supply = supplies.get(name)
        if supply is not None:
            initializers[name] = {"tensor": name, "storage": _storage_document(supply.encoding)}
            continue
        semantic = _as_object(values.get(name), f"{where}.values['{name}']").get("dtype")
        plain = _PLAIN_STORAGE_FOR.get(semantic) if isinstance(semantic, str) else None
        if plain is None:
            raise ContainerError(
                f"{where} initializer '{name}': 供給が無いのに意味論 dtype が"
                f" {semantic!r}（生の格納を持てない）"
            )
        shared = _as_object(raw, f"{where}.initializers['{name}']").get("shared") is True
        borrowed = {"shared": {"tensor": name}} if shared else {"tensor": name}
        initializers[name] = {**borrowed, "storage": {"dtype": plain}}
    document = {**root, "version": IR_VERSION, "initializers": initializers}
    return parse_ir_graph(json.dumps(document, separators=(",", ":"), allow_nan=False))


def assert_ir_accepted(read: ReadContainer) -> None:
    """容器が宣言する全グラフに **IR の受理規則**を掛ける（op 語彙 / ランタイム支援 / op 契約）。

    MUST: 配布形を作る経路と受け入れる経路の両方がここを通る（`karume dist` の入力検査と
    `karume verify`）。構造検査（{@link verify_container}）は「容器として開けるか」しか見ない
    ので、これが抜けると**語彙外の op を宣言した容器**が配布形に据わり、利用者の
    `createSession` で初めて落ちる（モジュール doc が掲げる目的そのもの）。
    """
    for name in sorted(read.graph.graphs):
        try:
            graph = ir_graph_from_container(read, name)
            assert_runtime_support(graph)
            assert_op_contracts(graph)
        except (ContainerError, ContainerFormatError, IrError, OpContractError, DimError) as cause:
            raise ContainerError(f"graph '{name}': {cause}") from cause


@dataclass(frozen=True)
class VerifiedContainer:
    """{@link verify_container} の結果（開いた読み手と、合流で決まった供給計画）。"""

    read: ReadContainer
    graphs: dict[str, BoundGraph]


def verify_container(paths: Sequence[str | Path], *, blocks: bool = False) -> VerifiedContainer:
    """コンテナ（`krm` の part 列 / `krg` 1 本）を開き、合流まで通す。

    `paths` は単一形なら 1 本、分割形なら **part 0 から順に**並べた part 列。既定で掛かるのは
    「宣言で決まる規則」全部（読み手の構造検査 + 上の合流層）で、**実バイトは読まない**
    （§7 のハッシュ 3 分離）。`blocks=True` は全 block を取り直して sha256 まで突き合わせ、
    `ternary` の payload のコード検査（§6.3）も掛ける — 「コンテナを検証する」経路はこの 1 本
    だけ MUST（CLI もここを通る）。
    """
    read = read_container([Path(path) for path in paths])
    if blocks:
        read.verify_blocks()
    bound = bind_graphs(read.graph, read.model)
    if blocks:
        _assert_ternary_payloads(read, bound)
    return VerifiedContainer(read, bound)


#: 2 bit コードのどれかが 0 のバイト（`ternary` の値域外 — container-v1 §6.3）。
_TERNARY_CODE_ZERO_BYTES = frozenset(
    byte for byte in range(256) if any((byte >> shift) & 3 == 0 for shift in (0, 2, 4, 6))
)


def _assert_ternary_payloads(read: ReadContainer, bound: Mapping[str, BoundGraph]) -> None:
    """`ternary` 宣言の追加条件: payload の全 2 bit コードが `{1, 2, 3}`（container-v1 §6.3）。

    TS `assertTernaryCodes`（ロード時に block を読むたびに掛かる）の鏡像。ここに無いと
    「verify は緑・ロードで初めて落ちる」形になる。見るのは payload だけ（末尾の詰め物は除く）。
    """
    for graph_name, graph in bound.items():
        for name, supply in graph.supplies.items():
            if supply.encoding.codec != "ternary":
                continue
            for block in supply.blocks:
                payload = read.block(block.id)[: block.payload_bytes]
                offending = next(
                    (
                        index
                        for index, byte in enumerate(payload)
                        if byte in _TERNARY_CODE_ZERO_BYTES
                    ),
                    None,
                )
                if offending is not None:
                    raise ContainerFormatError(
                        f"graph '{graph_name}' initializer '{name}': ternary の payload に"
                        f"コード 0（三値の値域外）がある（バイト {offending}）"
                    )


# ---- CLI ------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="コンテナ（krm / krg）を全規則で検証する（2 文書 + 合流 + 全 block の sha256）"
    )
    parser.add_argument(
        "models",
        type=Path,
        nargs="+",
        help="検証するコンテナ（複数指定可。part 0 のファイル・単一形・代表 path のどれでもよい"
        " — 分割形は連番の part 列としてまとめて検証する）",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """指定されたコンテナを 1 本ずつ検証する。

    引数は**コンテナの代表 path**（part 0 の現物を渡してもよい）で、分割されていれば連番の
    part 列へ解決してからまとめて検証する（{@link karume.container.container_parts}）—
    part 1 本だけを単体で検証しても「2 文書が無い」としか言えず、合流も block の sha256 も
    見られない。

    掛かるのは 2 文書の構造検査・合流（`bind_graphs`）・**全 block を取り直した sha256 の
    突合**と、IR の受理規則（{@link assert_ir_accepted} — 組み立て側の門と**同じ 1 本**）である。

    MUST: 落ちたファイルで止める（残りを検証して最後にまとめない）— 例外は規則違反の
    位置まで綴ってあるので、そのまま送出するのが最も情報量が多い。
    """
    args = build_parser().parse_args(argv)
    for path in args.models:
        print(_verify_container_line(path))


def _verify_container_line(path: Path) -> str:
    parts = container_parts(path)
    verified = verify_container(parts, blocks=True)
    read, bound = verified.read, verified.graphs
    assert_ir_accepted(read)
    assets = sorted(read.model.assets) if read.model is not None else []
    return (
        f"{path}: parts={len(parts)} blocks={len(read.block_ids)}"
        f" graphs={','.join(sorted(bound))}"
        f" initializers={sum(len(graph.supplies) for graph in bound.values())}"
        f" assets={','.join(assets) if assets else '（無し）'}"
    )


if __name__ == "__main__":
    main()
