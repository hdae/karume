"""コンテナ形式（`krm` / `krg`）の書き手と、自己検査用の読み手 — docs/container-v1.md。

正本は 2 つある: 物理形式・descriptor・束縛表・codec 台帳は `docs/container-v1.md`、グラフ JSON と
**正準直列化**は `docs/ir-v2.md`。実装としての正解判定は **TS 側の読み手**
（`packages/runtime/src/format/container/*.ts`）が開けることで、ここはその鏡像である
（`packages/runtime/tests/helpers/container-write.ts` が同じ規則のテスト用参照実装）。

この層が守る規則（どれも「黙って近似しない」側に倒す）:

- **正準直列化**: 空白なし・名前 map は code point 順・数値の綴りは ECMAScript `Number::toString`。
  同じ入力からは**バイト同一**の descriptor が出る（`krg` の同一性を内容ハッシュで判定する条件 —
  container-v1 §9 / §13.5）。
- **block**: 先頭 64 B 整列・長さ 4 の倍数（末尾の詰め物は**書き手が焼く**）・上限以下。
  中間 piece に詰め物は掛けられない（次の piece の先頭を潰す）ので、行の刻みを
  `4/gcd(rowBytes, 4)` に丸めて切る。
- **part**: part 0 = ヘッダ + 2 文書・part 1 = const 領域（空でも宣言）・part 2 以降 = 重み。
  長さ 0 の part の前に詰め物を挿まない（挿むと `krg` 抽出がバイトでずれる）。
- **メモリ**: テンソルは 1 本ずつ引いて 1 本ずつ手放す。全量をメモリに載せない。

MUST: 書き出しは「①配置を決める ②実体を 1 本ずつ流して block / part の sha256 を採る
③2 文書を組んで書く」の 3 段。part 0 は自分より後ろの part の sha256 を持つ。分割形は part 0 が
別ファイルなので②で data part を書き出しまで済ませ、実体を引くのは 1 度きり。単一形では
「先頭に置く文書が末尾の内容に依存する」— 一時ファイルを挟むか 2 度引くかのどちらかになり、
後者を採る（実体は**②と③で 2 度引く** — 一時ファイルはモデル全量ぶんの書き込みを 1 回増やす）。
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Buffer, Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, NamedTuple

from karume.ir import IrGraph
from karume.ops import WEIGHT_CHANNEL_AXES, WEIGHT_SLOTS

_MIB = 1024 * 1024

#: ヘッダの固定長（§1）: `[magic 4][u32 版][u64 グラフ記述長][u64 モデル記述長]`。
HEADER_BYTES = 24

#: このリポジトリが書く / 読むコンテナ版（§1）。
CONTAINER_VERSION = 1

#: モデル容器 / グラフ容器の magic（ASCII 4 文字 — §1。種別を持つのは magic だけ）。
MAGIC_MODEL = b"KRMC"
MAGIC_GRAPH = b"KRGC"

#: block の先頭と part の連結境界が満たす整列（§4.1 / §8）。
BLOCK_START_ALIGN = 64

#: block 長が満たす整列（§4.1 — 束縛表の規則②）。
BLOCK_TAIL_ALIGN = 4

#: 詰め物のバイト値（§4.1 — 0x00 固定）。
PAD_BYTE = b"\x00"

#: block 長の上限（§4.1 — 32 MiB **以下**。part 長からの派生ではない独立定数）。
BLOCK_MAX_BYTES = 32 * _MIB

#: 書き手が選べる part 長（part 2 以降 — §4.2）。
PART_LENGTH_CHOICES = (256 * _MIB, 512 * _MIB, 768 * _MIB, 1024 * _MIB)

#: part 長の既定（§4.2 — 段 3 の検収で 256 MiB 維持と裁定済み・ADR 0108 追記 5）。
DEFAULT_PART_BYTES = PART_LENGTH_CHOICES[0]

#: part 長の天井（part 0 / 1 を含む全 part — §10）。
PART_MAX_BYTES = 1024 * _MIB

#: part 件数の上限（§10 — 旧 `MAX_SHARDS` を継承）。
MAX_PARTS = 1024

#: 1 コンテナの block 件数の上限（const 目次とモデル目次の合計 — §10）。
MAX_BLOCKS = 65_536

#: 1 コンテナのグラフ件数の上限（§10）。
MAX_GRAPHS = 64

#: グラフ記述 / モデル記述それぞれのバイト長の上限（§10）。
MAX_DESCRIPTOR_BYTES = 32 * _MIB

#: descriptor / グラフ JSON の入れ子深さの上限（§10）。
MAX_JSON_DEPTH = 64

#: グラフ名の語彙（§2.1）。
GRAPH_NAME_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,64}$")

#: block id の語彙（§0）。
BLOCK_ID_PATTERN = re.compile(r"^[A-Za-z0-9._:-]{1,64}$")

#: sha256 欄の綴り（§0 — 小文字 16 進 64 文字）。
SHA256_HEX_PATTERN = re.compile(r"^[0-9a-f]{64}$")

#: const 領域行きの initializer を見分けるテンソルキーの接頭辞（`convert._add_const` の命名）。
CONST_KEY_PREFIX = "const."

#: block の役割（§2.2）。`"const"` は const 目次側にしか現れない。
BlockRole = Literal["weight", "scale", "zero-point", "asset"]


class AssetInput(NamedTuple):
    """資産 1 本の受け口（§2.2）— `(役割, 論理長, payload[, 専用 part か])`。

    `role` は **models 側の解釈者名**（`ple-index` / `rope-base` …）で、runtime は解釈しない。
    `length` は payload のバイト数（論理長）で、**宣言が先**に来る — block 長はこれを
    {@link BLOCK_TAIL_ALIGN} の倍数へ切り上げた値になり、消費側は末尾の 0x00 を推測で剥がない
    （ADR 0109 決定 4）。

    `payload` は生バイトか、引かれたときだけ実体を作る呼び出し。後者は**引かれるたびに同じ
    バイト列を返す** MUST（書き手は sha256 を採るときと書くときの 2 度引き、宣言と違う長さを
    落とす）。

    `dedicated_part` は**区間読みを要する資産**（PLE の `values` / `scales`）の印で、その
    block は専用 part に単独で置かれる MUST（container-v1 §4.2）— 全量読みの資産
    （索引・`rope_base`）どうしは 1 part を共有してよい。
    """

    role: str
    length: int
    payload: Buffer | Callable[[], Buffer]
    dedicated_part: bool = False


#: IR v2 のグラフ JSON の版（docs/ir-v2.md）。
IR_V2_FORMAT = "karume-ir"
IR_V2_VERSION = 2

#: 読み直しで part を digest するときの刻み（part は 1 GiB まで在りうる — 全量を器に載せない）。
_READ_CHUNK_BYTES = 8 * _MIB


class ContainerFormatError(Exception):
    """コンテナ形式の違反（未対応・不整合・破損）を 1 本に集める例外。"""


# ---------------------------------------------------------------------------
# 正準 JSON（docs/ir-v2.md「正準直列化」）
# ---------------------------------------------------------------------------

_REPR_PATTERN = re.compile(r"^(\d+)(?:\.(\d+))?(?:[eE]([+-]?\d+))?$")

#: ECMAScript `Number::toString` が指数表記に切り替える境界（§6.1.6.1.20 の手順 5 / 6 / 7）。
_ECMA_EXPONENT_CEILING = 21
_ECMA_EXPONENT_FLOOR = -6

#: JSON 数値として往復できる整数の絶対値上限（ECMAScript の Number は倍精度）。
_MAX_SAFE_INTEGER = 2**53 - 1


def _format_double(value: float) -> str:
    """ECMAScript `Number::toString` の綴り（docs/ir-v2.md「正準直列化」）。

    Python の `repr` は最短往復桁を出すが、指数の付け方が違う（`repr(1e-6)` = `1e-06`・
    `repr(10000.0)` = `10000.0`）。桁はそのまま使い、**指数の付け方だけ** ECMAScript の規則
    （整数値は小数点も指数も付けない / 指数になるのは `|x| < 1e-6` か `|x| >= 1e21` のときだけ /
    指数のゼロ詰めをしない / 正の指数に `+`）へ直す。
    """
    if value != value or value in (float("inf"), float("-inf")):
        raise ContainerFormatError(f"非有限数は JSON に書けない: {value!r}")
    if value == 0.0:
        # -0.0 も "0"（2 通りの綴りを持たない）。
        return "0"
    sign = "-" if value < 0 else ""
    matched = _REPR_PATTERN.match(repr(abs(value)))
    if matched is None:  # pragma: no cover - repr は必ずこの形
        raise ContainerFormatError(f"float の repr を読めない: {value!r}")
    integer_part = matched.group(1)
    fraction_part = matched.group(2) or ""
    exponent = matched.group(3)
    digits = integer_part + fraction_part
    # value = 0.<digits> × 10^point
    point = len(integer_part) + (int(exponent) if exponent else 0)
    without_leading = digits.lstrip("0")
    point -= len(digits) - len(without_leading)
    digits = without_leading.rstrip("0")
    count = len(digits)
    if count <= point <= _ECMA_EXPONENT_CEILING:
        return sign + digits + "0" * (point - count)
    if 0 < point <= _ECMA_EXPONENT_CEILING:
        return sign + digits[:point] + "." + digits[point:]
    if _ECMA_EXPONENT_FLOOR < point <= 0:
        return sign + "0." + "0" * (-point) + digits
    power = point - 1
    mantissa = digits if count == 1 else digits[0] + "." + digits[1:]
    return f"{sign}{mantissa}e{'+' if power >= 0 else '-'}{abs(power)}"


def _format_number(value: int | float) -> str:
    if isinstance(value, int):
        if -_MAX_SAFE_INTEGER <= value <= _MAX_SAFE_INTEGER:
            return str(value)
        # 安全整数を超える値は ECMAScript では倍精度へ丸まる。丸めた綴りを書く
        # （Python 側だけが余分な桁を持つ = 読み手と書き手で値が違う、を作らない）。
        return _format_double(float(value))
    return _format_double(value)


def _write_json(value: Any, out: list[str], depth: int) -> None:
    if depth > MAX_JSON_DEPTH:
        raise ContainerFormatError(f"JSON の入れ子が深さ上限 {MAX_JSON_DEPTH} を超えた")
    if value is None:
        out.append("null")
        return
    if value is True:
        out.append("true")
        return
    if value is False:
        out.append("false")
        return
    if isinstance(value, int | float):
        out.append(_format_number(value))
        return
    if isinstance(value, str):
        out.append(json.dumps(value, ensure_ascii=False))
        return
    if isinstance(value, list | tuple):
        out.append("[")
        for index, item in enumerate(value):
            if index:
                out.append(",")
            _write_json(item, out, depth + 1)
        out.append("]")
        return
    if isinstance(value, Mapping):
        out.append("{")
        for index, (key, item) in enumerate(value.items()):
            if not isinstance(key, str):
                raise ContainerFormatError(f"JSON のキーが文字列でない: {key!r}")
            if key == "__proto__":
                # 読み手（TS）はこのキーを拒否する（素の `{}` へ代入すると [[Prototype]] 設定に
                # 化けて own property が作られない）。書けたのに読めないものを作らない。
                raise ContainerFormatError("JSON のキー '__proto__' は書けない")
            if index:
                out.append(",")
            out.append(json.dumps(key, ensure_ascii=False))
            out.append(":")
            _write_json(item, out, depth + 1)
        out.append("}")
        return
    raise ContainerFormatError(f"JSON として表せない値: {type(value).__name__}")


def canonical_json(value: Any) -> str:
    """正準 JSON 文字列（空白なし・`ensure_ascii=False`・NaN / Inf 禁止）。

    **並べ替えはしない** — 固定スキーマのキー順も名前 map の code point 順も、組み立てた側
    （{@link ir_v2_document} / descriptor の組み立て）が決める。ここは「決めた順を綴る」だけの
    直列化器で、両方の責任を混ぜると「どちらの順が効いたのか」がテストで切り分けられなくなる。
    """
    out: list[str] = []
    _write_json(value, out, 0)
    return "".join(out)


def _canonical_free_form(value: Any) -> Any:
    """自由形の JSON 値（ノードの `attrs`）を再帰的に正準化する — map のキーを code point 順に。"""
    if isinstance(value, Mapping):
        return {key: _canonical_free_form(value[key]) for key in sorted(value)}
    if isinstance(value, list | tuple):
        return [_canonical_free_form(item) for item in value]
    return value


# ---------------------------------------------------------------------------
# codec 台帳（§6.2 / §6.3 — TS `format/container/codecs.ts` と同値）
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CodecPacking:
    """宣言バイト長と整列の正本（bit 数は `block_bytes * 8 / block_elements` の派生値）。"""

    block_elements: int
    block_bytes: int
    align_bytes: int


@dataclass(frozen=True)
class CodecEntry:
    #: 展開経路の種別（codec からの派生値 — 消費側はこれで分岐する）。綴りは旧 `storage.dtype`
    #: と同じ語彙で、`ternary` は `int2-off` と同じ i2 経路を共有する（§6.2 / ADR 0108 決定 13）。
    layout: Literal["f32", "f16", "bf16", "i32", "i8", "i4", "i2"]
    packing: CodecPacking
    #: `"required"` = 量子化（`rowAxis` / `groupSize` / `scale` が必須）・`"forbidden"` = 非量子化。
    scale: Literal["required", "forbidden"]
    #: `"channel"` = per-channel（`groupSize` は行長）・`"group"` = 2 冪 ≥ 16 の group 長。
    grouping: Literal["channel", "group"] | None


def _raw(layout: Literal["f32", "f16", "bf16", "i32"], block_bytes: int) -> CodecEntry:
    return CodecEntry(layout, CodecPacking(1, block_bytes, 4), "forbidden", None)


#: 初版の台帳（§6.3）。登録名は**資産に焼かれる**ので、既存エントリの値は動かさない。
#:
#: MUST: TS 側（`packages/runtime/src/format/container/codecs.ts`）と同じ表を**両側で持つ**。
#: 導出できる事実ではなく cross-package の不変条件なので、突合点はテスト
#: （`tests/test_container.py` の台帳突合）に置く。
CODEC_LEDGER: Mapping[str, CodecEntry] = {
    "f32": _raw("f32", 4),
    "f16": _raw("f16", 2),
    "bf16": _raw("bf16", 2),
    "i32": _raw("i32", 4),
    "int8-sym": CodecEntry("i8", CodecPacking(1, 1, 4), "required", "channel"),
    "int4-sym-g": CodecEntry("i4", CodecPacking(8, 4, 4), "required", "group"),
    "int2-off": CodecEntry("i2", CodecPacking(16, 4, 4), "required", "channel"),
    "ternary": CodecEntry("i2", CodecPacking(16, 4, 4), "required", "channel"),
}

#: group 量子化の group 長の下限（ADR 0069 決定 2）。
MIN_GROUP_SIZE = 16

#: scale の dtype の受理集合（初版は f32 のみ — §6.1）。
SCALE_DTYPES = ("f32",)

#: 旧 `storage.dtype` → 台帳の登録名（§6.3 の「旧 `storage.dtype`」行）。
#:
#: MUST: `ternary` へは写さない — 値域が `int2-off` の部分集合でも「三値である」という主張は
#: 量子化器の側がするもので、旧資産の i2 には `q = -2` が実測で 6 % 前後出る（§6.3）。
CODEC_FOR_STORAGE: Mapping[str, str] = {
    "f32": "f32",
    "f16": "f16",
    "bf16": "bf16",
    "i32": "i32",
    "i8": "int8-sym",
    "i4": "int4-sym-g",
    "i2": "int2-off",
}


def codec_entry(codec: str) -> CodecEntry:
    entry = CODEC_LEDGER.get(codec)
    if entry is None:
        raise ContainerFormatError(
            f"codec 台帳に '{codec}' が無い（台帳: {' / '.join(sorted(CODEC_LEDGER))}）"
        )
    return entry


def per_channel_group_size(row_length: int) -> int:
    """per-channel codec の `groupSize`（= 行長）。

    要素数 0 の退化形（`in_features = 0` など）は行長 0 になるが、`groupSize` は 1 以上 MUST なので
    1 に丸める（group 数は {@link group_count} が 1 に戻す）。TS 側 `codecs.ts` の鏡像。
    """
    return max(row_length, 1)


def group_count(row_length: int, group_size: int) -> int:
    """scale の group 数 `行長 / groupSize`（§6.1）。行長 0 の退化形は group 数 1。"""
    return 1 if row_length == 0 else row_length // group_size


def payload_bytes(codec: str, numel: int, where: str) -> int:
    """payload のバイト長（§6.1 の式）。端数の packing block は作れない。"""
    packing = codec_entry(codec).packing
    if numel % packing.block_elements != 0:
        raise ContainerFormatError(
            f"{where}: 要素数 {numel} が codec '{codec}' の packing"
            f"（{packing.block_elements} 要素 / block）で割り切れない"
        )
    return numel // packing.block_elements * packing.block_bytes


# ---------------------------------------------------------------------------
# IR v2 文書（docs/ir-v2.md §13 — initializer 名を実体の鍵にする）
# ---------------------------------------------------------------------------


def _rename_map(graph: IrGraph) -> dict[str, str]:
    """IR v1 の initializer 名 → v2 の名前（= テンソルキー）。

    v1 の名前は torch.export の placeholder 名で、上流の鍵は `tensor` 欄が別に持っていた。
    v2 は名前そのものが実体の鍵なので、`tensor`（共有宣言なら貸し手のキー）へ付け替える。
    """
    rename: dict[str, str] = {}
    for name, initializer in graph.initializers.items():
        if initializer.shared is not None:
            target = initializer.shared.tensor
        elif initializer.tensor is not None:
            target = initializer.tensor
        else:
            raise ContainerFormatError(
                f"initializer '{name}': `tensor` も `shared` も無い（IR v1 として不正）"
            )
        if not target:
            raise ContainerFormatError(f"initializer '{name}': テンソルキーが空文字列")
        rename[name] = target
    return rename


def _assert_no_rename_collision(graph: IrGraph, rename: Mapping[str, str]) -> None:
    """改名が値名を潰さないこと（潰すと宣言が黙って 1 本消える）。"""
    occupied = {spec.name: "inputs" for spec in graph.inputs}
    for name in graph.values:
        if name not in rename:
            occupied[name] = "values"
    by_target: dict[str, str] = {}
    for name, target in rename.items():
        owner = occupied.get(target)
        if owner is not None:
            raise ContainerFormatError(
                f"initializer '{name}' の改名先 '{target}' が既存の値名（{owner}）と衝突する"
            )
        previous = by_target.get(target)
        if previous is not None:
            raise ContainerFormatError(
                f"initializer '{name}' と '{previous}' の改名先 '{target}' が衝突する"
            )
        by_target[target] = name


def ir_v2_document(graph: IrGraph) -> dict[str, Any]:
    """IR v1 の {@link IrGraph} → **正準順**の IR v2 グラフ JSON（docs/ir-v2.md）。

    差分は 3 点: ①initializer をテンソルキーへ改名（`values` のキー・`nodes[].ins`・`outputs` も）
    ②`storage` を落として `{}` / `{"shared": true}` にする ③`version` は 2。

    キー順は仕様の例の順、名前を鍵に持つ map（`initializers` / `values` / `states` / `attrs` /
    ノードの `states`）と順序に意味の無い集合（`requires.ops` / `symbols`）は code point 順、
    空の `states` とノードの空 `states`・`external: false` は書かない。
    """
    rename = _rename_map(graph)
    _assert_no_rename_collision(graph, rename)

    def renamed(name: str) -> str:
        return rename.get(name, name)

    document: dict[str, Any] = {
        "format": IR_V2_FORMAT,
        "version": IR_V2_VERSION,
        "requires": {"ops": sorted(graph.required_ops)},
        "symbols": sorted(graph.symbols),
        "inputs": [spec.to_dict() for spec in graph.inputs],
        "outputs": [renamed(name) for name in graph.outputs],
        "initializers": {
            rename[name]: ({"shared": True} if initializer.shared is not None else {})
            for name, initializer in sorted(
                graph.initializers.items(), key=lambda item: rename[item[0]]
            )
        },
        "values": {
            new: {"dtype": value.dtype, "shape": list(value.shape)}
            for new, value in sorted(
                ((renamed(name), value) for name, value in graph.values.items()),
                key=lambda item: item[0],
            )
        },
    }
    if graph.states:
        document["states"] = {name: graph.states[name].to_dict() for name in sorted(graph.states)}
    document["nodes"] = [
        {
            "op": node.op,
            "ins": [renamed(name) for name in node.ins],
            "outs": list(node.outs),
            "attrs": _canonical_free_form(node.attrs),
            **(
                {"states": {key: node.states[key] for key in sorted(node.states)}}
                if node.states
                else {}
            ),
        }
        for node in graph.nodes
    ]
    return document


# ---------------------------------------------------------------------------
# descriptor の型（§2）
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Encoding:
    """書き手への格納の指定（テンソルキー単位）。descriptor の `encoding` の材料。

    `scale_key` は **companion scale のテンソルキー**（`tensors` から同じ口で引く）で、
    descriptor に載るのはそれを収めた block の id である。
    """

    codec: str
    group_size: int | None = None
    row_axis: int | None = None
    scale_key: str | None = None


@dataclass(frozen=True)
class BlockEncoding:
    """descriptor に載る格納の宣言（§6.1）。`packing` は台帳の写しなので欄に持たない。"""

    codec: str
    row_axis: int | None = None
    group_size: int | None = None
    scale_block: str | None = None

    def to_document(self) -> dict[str, Any]:
        entry = codec_entry(self.codec)
        packing = entry.packing
        document: dict[str, Any] = {
            "codec": self.codec,
            "packing": {
                "blockElements": packing.block_elements,
                "blockBytes": packing.block_bytes,
                "alignBytes": packing.align_bytes,
            },
        }
        quantized = entry.scale == "required"
        present = (self.row_axis, self.group_size, self.scale_block)
        if not quantized:
            if any(field is not None for field in present):
                raise ContainerFormatError(
                    f"codec '{self.codec}' は量子化でないので"
                    " rowAxis / groupSize / scale を書けない"
                )
            return document
        if any(field is None for field in present):
            raise ContainerFormatError(
                f"codec '{self.codec}' は量子化なので rowAxis / groupSize / scale が要る"
            )
        document["rowAxis"] = self.row_axis
        document["groupSize"] = self.group_size
        document["scale"] = {"block": self.scale_block, "dtype": SCALE_DTYPES[0]}
        return document


@dataclass(frozen=True)
class ConstBlockRecord:
    """const 領域の block（§2.1 — offset は const 領域の先頭からの相対）。"""

    id: str
    offset: int
    length: int
    sha256: str


@dataclass(frozen=True)
class ConstantBinding:
    graph: str
    initializer: str
    block: str
    encoding: BlockEncoding


@dataclass(frozen=True)
class DataBlockRecord:
    """part 2 以降の block（§2.2 — offset は所属 part の先頭からの相対）。"""

    id: str
    part: int
    offset: int
    length: int
    sha256: str
    role: BlockRole


@dataclass(frozen=True)
class PartRecord:
    index: int
    length: int
    sha256: str


@dataclass(frozen=True)
class AssetRecord:
    """モデル記述の `assets` 1 件（§2.2 — 名前 → block + 役割 + 論理長）。

    `length` は payload のバイト数で、`block` の長さは**これを
    {@link BLOCK_TAIL_ALIGN} の倍数へ切り上げた値** MUST（詰め物の量まで宣言で閉じる）。
    """

    block: str
    role: str
    length: int


@dataclass(frozen=True)
class WeightSupply:
    """束縛表の供給形（§5）。`block`（丸ごと 1 本）と `pieces`（2 本以上）は排他。"""

    encoding: BlockEncoding
    block: str | None = None
    pieces: tuple[tuple[str, tuple[int, int]], ...] | None = None

    def to_document(self) -> dict[str, Any]:
        if (self.block is None) == (self.pieces is None):
            raise ContainerFormatError("供給形は `block` と `pieces` のどちらか一方だけを持つ")
        if self.pieces is not None:
            if len(self.pieces) < 2:
                raise ContainerFormatError("`pieces` は 2 本以上（1 本なら `block` で書く）")
            supply: dict[str, Any] = {
                "pieces": [
                    {"block": block, "rows": [rows[0], rows[1]]} for block, rows in self.pieces
                ]
            }
        else:
            supply = {"block": self.block}
        supply["encoding"] = self.encoding.to_document()
        return supply


@dataclass(frozen=True)
class Provenance:
    """出所（§2.3 — 本文は載せない）。"""

    license: str
    notice: str | None = None
    upstream_revision: str | None = None
    writer: str | None = None

    def to_document(self) -> dict[str, Any]:
        document: dict[str, Any] = {"license": self.license}
        for key, value in (
            ("notice", self.notice),
            ("upstreamRevision", self.upstream_revision),
            ("writer", self.writer),
        ):
            if value is not None:
                document[key] = value
        return document


@dataclass(frozen=True)
class GraphDescriptor:
    """グラフ記述（§2.1）。`krm` と `krg` で**バイト単位に同一**。"""

    graphs: Mapping[str, Mapping[str, Any]]
    const_length: int
    const_blocks: tuple[ConstBlockRecord, ...]
    constants: tuple[ConstantBinding, ...]

    def to_document(self) -> dict[str, Any]:
        ops = sorted({op for graph in self.graphs.values() for op in graph["requires"]["ops"]})
        return {
            "format": "karume-container",
            "version": CONTAINER_VERSION,
            "capabilities": {"ops": ops, "features": []},
            "graphs": {name: self.graphs[name] for name in sorted(self.graphs)},
            "const": {
                "length": self.const_length,
                "blocks": [
                    {
                        "id": block.id,
                        "offset": block.offset,
                        "length": block.length,
                        "sha256": block.sha256,
                    }
                    for block in sorted(self.const_blocks, key=lambda block: block.offset)
                ],
                "constants": [
                    {
                        "graph": entry.graph,
                        "initializer": entry.initializer,
                        "block": entry.block,
                        "encoding": entry.encoding.to_document(),
                    }
                    for entry in sorted(
                        self.constants, key=lambda entry: (entry.graph, entry.initializer)
                    )
                ],
            },
        }


@dataclass(frozen=True)
class ModelDescriptor:
    """モデル記述（§2.2）。`krg` では存在しない。"""

    parts: tuple[PartRecord, ...]
    blocks: tuple[DataBlockRecord, ...]
    binding: Mapping[str, Mapping[str, WeightSupply]]
    assets: Mapping[str, AssetRecord]
    provenance: Provenance

    def to_document(self) -> dict[str, Any]:
        codecs = sorted(
            {
                supply.encoding.codec
                for supplies in self.binding.values()
                for supply in supplies.values()
            }
        )
        return {
            "format": "karume-model",
            "version": CONTAINER_VERSION,
            "codecs": codecs,
            "parts": [
                {"index": part.index, "length": part.length, "sha256": part.sha256}
                for part in sorted(self.parts, key=lambda part: part.index)
            ],
            "blocks": [
                {
                    "id": block.id,
                    "part": block.part,
                    "offset": block.offset,
                    "length": block.length,
                    "sha256": block.sha256,
                    "role": block.role,
                }
                for block in sorted(self.blocks, key=lambda block: (block.part, block.offset))
            ],
            "binding": {
                graph: {
                    name: self.binding[graph][name].to_document()
                    for name in sorted(self.binding[graph])
                }
                for graph in sorted(self.binding)
            },
            "assets": {
                name: {
                    "block": self.assets[name].block,
                    "role": self.assets[name].role,
                    "length": self.assets[name].length,
                }
                for name in sorted(self.assets)
            },
            "provenance": self.provenance.to_document(),
        }


def serialize_graph_descriptor(descriptor: GraphDescriptor) -> bytes:
    return _encode_descriptor(descriptor.to_document(), "グラフ記述")


def serialize_model_descriptor(descriptor: ModelDescriptor) -> bytes:
    return _encode_descriptor(descriptor.to_document(), "モデル記述")


def _encode_descriptor(document: Mapping[str, Any], label: str) -> bytes:
    encoded = canonical_json(document).encode("utf-8")
    if len(encoded) > MAX_DESCRIPTOR_BYTES:
        raise ContainerFormatError(
            f"{label}が上限 {MAX_DESCRIPTOR_BYTES} バイトを超える: {len(encoded)}"
        )
    return encoded


# ---------------------------------------------------------------------------
# IR の `storage` 宣言 → 束縛表（§6.3 の写像）
# ---------------------------------------------------------------------------
#
# MUST: この写像は**torch を要らない側**に置く。書き手（`karume.emit.stored_model`）と移行 CLI
# （`karume.migrate`）の両方が通る 1 本なので、格納変換の層（torch 依存）へ置くと移行 CLI が
# torch を import グラフへ引き込む（旧 shard を読むだけの経路に 1 GB 級の依存が乗る）。


def bakeable_initializers(graph: IrGraph) -> set[str]:
    """このコンテナに**実体を書く** initializer の名前（共有宣言を除いた集合）。

    共有 initializer（ADR 0096 段 2）は貸し手のバイトを借りるだけなので、格納の計画・適格判定・
    宣言と実体の突合はどれもこの集合を走査する（1 箇所に閉じる — 除外を各所に書き写すと、
    書き足した走査だけが `tensor = None` を引く）。
    """
    return {name for name, init in graph.initializers.items() if not init.is_shared}


def weight_channel_axes(graph: IrGraph) -> dict[str, int]:
    """重みスロットで消費される initializer → per-channel 軸
    （`packages/runtime/src/runtime/plan.ts` の鏡像）。

    軸は**消費側の op** から引く（重みの shape だけでは linear `[out,in]` と
    conv_transpose1d `[Cin,Cout,K]` を区別できない）。同じ initializer を軸の違う op が
    消費している場合は 1 つに決まらないので fail loudly。
    """
    bakeable = bakeable_initializers(graph)
    axes: dict[str, int] = {}
    for node in graph.nodes:
        slot = WEIGHT_SLOTS.get(node.op)
        if slot is None or slot >= len(node.ins):
            continue
        name = node.ins[slot]
        if name not in bakeable:
            continue
        axis = WEIGHT_CHANNEL_AXES[node.op]
        if axes.setdefault(name, axis) != axis:
            raise ContainerFormatError(
                f"initializer '{name}': 消費 op ごとに per-channel 軸が違う"
                f"（{axes[name]} と {axis}）— 1 本の scale では表せない"
            )
    return axes


def concrete_shape(graph: IrGraph, name: str, where: str) -> list[int]:
    """initializer の宣言 shape（記号次元を持つ実体は在りえない）。

    移行 CLI（`karume.migrate._assert_scale_layouts`）も同じ判定を要るので、束縛の導出
    （{@link container_bindings}）と**同じ 1 本**を公開面に置く。写しを持つと、記号次元の
    扱いが片方だけ動いた日に「書き手は落ちるのに移行は通る」形が作れる。
    """
    value = graph.values.get(name)
    if value is None:
        raise ContainerFormatError(f"{where}: `values` に宣言が無い")
    shape: list[int] = []
    for dim in value.shape:
        if not isinstance(dim, int):
            raise ContainerFormatError(f"{where}: initializer の shape に記号次元がある（{dim!r}）")
        shape.append(dim)
    return shape


def container_bindings(graph: IrGraph) -> dict[str, Encoding]:
    """IR の `storage` 宣言 → コンテナの束縛（**テンソルキー** → {@link Encoding}）。

    MUST: 書き手（{@link karume.emit.stored_model}）と移行 CLI（`karume.migrate`）は
    **この 1 本**から束縛を出す。2 経路で導くと、同じ資産から「宣言 i4 / 実体 i8」のように
    形も型も合う沈黙誤値が作れる（container-v1 §12 の写像規則）。

    共有 initializer（バイトを持たない宣言）は束縛を持たない。`rowAxis` は**消費側 op** から
    引き（{@link weight_channel_axes}）、per-channel の `groupSize` は行長（= numel / 行数）に
    なる。group codec（`int4-sym-g`）の scale は先頭次元を行として焼かれているので、消費 op が
    軸 1 を要求する形は写せない（fail loudly — 黙って軸 0 として宣言すると値が入れ替わる）。
    """
    axes = weight_channel_axes(graph)
    bindings: dict[str, Encoding] = {}
    for name, initializer in graph.initializers.items():
        if initializer.is_shared:
            continue
        where = f"initializer '{name}'"
        key = initializer.tensor
        if key is None:
            raise ContainerFormatError(f"{where}: `tensor` も `shared` も無い（IR として不正）")
        storage = initializer.storage
        codec = CODEC_FOR_STORAGE.get(storage.dtype)
        if codec is None:
            raise ContainerFormatError(
                f"{where}: 格納 dtype '{storage.dtype}' の写し先が codec 台帳に無い"
            )
        entry = codec_entry(codec)
        if entry.scale == "forbidden":
            bindings[key] = Encoding(codec)
            continue
        if storage.scale is None:
            raise ContainerFormatError(
                f"{where}: 量子化格納 '{storage.dtype}' なのに scale の宣言が無い"
            )
        shape = concrete_shape(graph, name, where)
        row_axis = axes.get(name, 0)
        if len(shape) <= row_axis:
            raise ContainerFormatError(
                f"{where}: rowAxis {row_axis} に対して宣言 shape {shape} が浅い"
            )
        if entry.grouping == "group":
            if storage.group_size is None:
                raise ContainerFormatError(f"{where}: group 量子化なのに group_size の宣言が無い")
            if row_axis != 0:
                raise ContainerFormatError(
                    f"{where}: 消費 op の per-channel 軸が {row_axis} だが、group scale は"
                    "先頭次元を行として焼かれている（写せる形が無い）"
                )
            group_size = storage.group_size
        else:
            row_count = shape[row_axis]
            group_size = per_channel_group_size(math.prod(shape) // row_count if row_count else 0)
        bindings[key] = Encoding(
            codec, group_size=group_size, row_axis=row_axis, scale_key=storage.scale
        )
    return bindings


# ---------------------------------------------------------------------------
# ヘッダと part の配置（§1 / §8 / §9）
# ---------------------------------------------------------------------------


def align_up(value: int, align: int) -> int:
    if value < 0:
        raise ContainerFormatError(f"整列対象が非負でない: {value}")
    return -(-value // align) * align


def write_header(*, kind: Literal["model", "graph"], graph_length: int, model_length: int) -> bytes:
    if kind == "graph" and model_length != 0:
        raise ContainerFormatError("krg はモデル記述を持てない（長さは 0 MUST）")
    if kind == "model" and model_length == 0:
        raise ContainerFormatError("krm はモデル記述を必ず持つ（長さ 0 は krg の形）")
    if graph_length == 0:
        raise ContainerFormatError("グラフ記述の長さが 0")
    magic = MAGIC_MODEL if kind == "model" else MAGIC_GRAPH
    return (
        magic
        + CONTAINER_VERSION.to_bytes(4, "little")
        + graph_length.to_bytes(8, "little")
        + model_length.to_bytes(8, "little")
    )


@dataclass(frozen=True)
class ContainerHeader:
    kind: Literal["model", "graph"]
    version: int
    graph_length: int
    model_length: int

    @property
    def part0_length(self) -> int:
        return HEADER_BYTES + self.graph_length + self.model_length


def read_header(raw: bytes) -> ContainerHeader:
    if len(raw) < HEADER_BYTES:
        raise ContainerFormatError(
            f"ヘッダに {HEADER_BYTES} バイト必要だが {len(raw)} バイトしかない"
        )
    magic = bytes(raw[0:4])
    if magic == MAGIC_MODEL:
        kind: Literal["model", "graph"] = "model"
    elif magic == MAGIC_GRAPH:
        kind = "graph"
    else:
        raise ContainerFormatError(
            f"未知の magic: {magic.hex(' ')}"
            f"（Karume のコンテナは {MAGIC_MODEL!r} / {MAGIC_GRAPH!r}）"
        )
    version = int.from_bytes(raw[4:8], "little")
    if version != CONTAINER_VERSION:
        raise ContainerFormatError(f"未対応のコンテナ版 {version}（対応は {CONTAINER_VERSION}）")
    graph_length = int.from_bytes(raw[8:16], "little")
    model_length = int.from_bytes(raw[16:24], "little")
    if graph_length == 0:
        raise ContainerFormatError("グラフ記述の長さが 0")
    for label, length in (("グラフ記述", graph_length), ("モデル記述", model_length)):
        if length > MAX_DESCRIPTOR_BYTES:
            raise ContainerFormatError(
                f"{label}の長さ {length} が上限 {MAX_DESCRIPTOR_BYTES} を超える"
            )
    if kind == "graph" and model_length != 0:
        raise ContainerFormatError(f"krg なのにモデル記述の長さが {model_length}")
    if kind == "model" and model_length == 0:
        raise ContainerFormatError("krm なのにモデル記述の長さが 0")
    return ContainerHeader(kind, version, graph_length, model_length)


#: 連番の桁数（`<stem>-NNNNN-of-NNNNN<suffix>` — §8。旧 shard と同じ綴り規約）。
SEQUENCE_DIGITS = 5

#: 連番が表せる最大の添字（桁数からの派生値 — part 件数の上限とは別の制約）。
_MAX_SEQUENCE_INDEX = 10**SEQUENCE_DIGITS - 1

#: 連番ファイル名の逆向き（`<stem>-NNNNN-of-NNNNN` → `<stem>`）。
_SEQUENCE_STEM = re.compile(rf"^(.+)-\d{{{SEQUENCE_DIGITS}}}-of-\d{{{SEQUENCE_DIGITS}}}$")


def numbered_name(name: str, index: int, total: int) -> str:
    """`<拡張子の前>-NNNNN-of-NNNNN<拡張子>`（`index` は 1 始まり — §8）。

    `name` は path 片でもよい（最終要素だけを書き換える）— 配布形の相対 path と手元の実 path が
    同じ綴りから出る。
    """
    if not 1 <= index <= total <= min(MAX_PARTS, _MAX_SEQUENCE_INDEX):
        raise ContainerFormatError(
            f"連番 {index}/{total} が 1..{min(MAX_PARTS, _MAX_SEQUENCE_INDEX)} の範囲に無い"
        )
    parsed = Path(name)
    numbered = f"{index:0{SEQUENCE_DIGITS}d}-of-{total:0{SEQUENCE_DIGITS}d}"
    return str(parsed.with_name(f"{parsed.stem}-{numbered}{parsed.suffix}"))


def numbered_path(path: Path, index: int, total: int) -> Path:
    """{@link numbered_name} の `Path` 版（親ディレクトリはそのまま）。"""
    return path.with_name(numbered_name(path.name, index, total))


def base_path(path: Path) -> Path:
    """連番のファイル名 → **代表 path**（連番でなければそのまま）。

    {@link numbered_name} の逆向き。手元の現物（`model.i8-00001-of-00003.krm`）を指した
    呼び出しを、黙って「1 本だけの容器」として扱わないための畳み込み。
    """
    name = Path(path.name)
    matched = _SEQUENCE_STEM.fullmatch(name.stem)
    return path if matched is None else path.with_name(f"{matched.group(1)}{name.suffix}")


def _sequence_pattern(path: Path) -> re.Pattern[str]:
    """`path` と同じ代表 path に属する連番ファイル名に一致する正規表現。

    stem / suffix は `re.escape` する — 実 path にはドットもハイフンも入るので、素で埋めると
    無関係なファイルを拾う（glob も同じ理由で使わない: `[` を含む名前が黙って別解釈になる）。
    """
    name = Path(path.name)
    stem, suffix = re.escape(name.stem), re.escape(name.suffix)
    return re.compile(rf"^{stem}-(\d{{{SEQUENCE_DIGITS}}})-of-(\d{{{SEQUENCE_DIGITS}}}){suffix}$")


def resolve_sequence(path: Path) -> tuple[Path, ...]:
    """代表 path → 実在する連番の列（分割されていなければ 1 要素）。

    返すのは常に添字順。分割されていない現物と存在しない現物はどちらも `(path,)` を返す
    （不在の診断は呼び手の門が持つ — ここで先回りすると綴りが 2 つに割れる）。

    MUST: 曖昧な現場は fail loudly。単一形と連番の**同居**、`of` の食い違い、番号の欠け /
    はみ出しは、どれも「どのバイト列を配るか」が一意に決まらない。
    """
    parent = path.parent
    if not parent.is_dir():
        return (path,)
    pattern = _sequence_pattern(path)
    found: dict[int, Path] = {}
    totals: set[int] = set()
    for entry in parent.iterdir():
        match = pattern.fullmatch(entry.name)
        if match is None or not entry.is_file():
            continue
        found[int(match.group(1))] = entry
        totals.add(int(match.group(2)))
    if not found:
        return (path,)
    if path.is_file():
        raise ContainerFormatError(
            f"{path}: 単一形と連番（{len(found)} 本）が同居している"
            " — 前回の書き出しの残骸を消してからやり直す"
        )
    if len(totals) != 1:
        raise ContainerFormatError(f"{path}: 連番の総数が {sorted(totals)} と食い違っている")
    total = totals.pop()
    missing = sorted(set(range(1, total + 1)) - set(found))
    surplus = sorted(set(found) - set(range(1, total + 1)))
    if missing or surplus:
        raise ContainerFormatError(
            f"{path}: 連番 1..{total} が揃っていない（欠け {missing} / はみ出し {surplus}）"
        )
    return tuple(found[index] for index in range(1, total + 1))


def sequence_siblings(path: Path) -> tuple[Path, ...]:
    """この代表 path の出力になりうる実在ファイル（代表 path + 連番の全件）。

    後片付け（前回の書き出しが別の分割数で残した現物）と一時ファイルの掃除が使う。番号の
    整合は見ない — **壊れた残骸ほど拾えなければ困る**ので、名前の形だけで拾う。
    """
    siblings = [path] if path.is_file() else []
    parent = path.parent
    if parent.is_dir():
        pattern = _sequence_pattern(path)
        siblings.extend(
            entry
            for entry in sorted(parent.iterdir())
            if pattern.fullmatch(entry.name) and entry.is_file()
        )
    return tuple(siblings)


def container_paths(path: Path, part_count: int) -> list[Path]:
    """分割形のファイル名（part 0 から・§8 の連番規約）。"""
    return [
        path.parent / numbered_name(path.name, index + 1, part_count) for index in range(part_count)
    ]


def container_parts(path: str | Path) -> tuple[Path, ...]:
    """コンテナ 1 本の part 列（part 0 のファイル / 単一形 / 代表 path のどれを渡してもよい）。

    分割形の part は §8 の連番規約なので、畳み込みと解決は {@link base_path} /
    {@link resolve_sequence} の 1 本道を借りる（単一形と連番の同居はそこが fail loudly で受ける）。
    """
    return resolve_sequence(base_path(Path(path)))


@dataclass(frozen=True)
class DocumentRef:
    """descriptor 1 文書ぶんの期待値（manifest `karume/5` の `container.descriptor`）。

    2 文書それぞれの**バイト長 + sha256**で、part 0 ファイルの sha256 とは別の事実である
    （ADR 0109 決定 3）。`graph` 文書の sha256 はそのまま `krg` の同一性（ADR 0108 決定 4）に使う。
    """

    length: int
    sha256: str

    def to_document(self) -> dict[str, Any]:
        return {"length": self.length, "sha256": self.sha256}


def read_descriptor_refs(path: Path) -> tuple[DocumentRef, DocumentRef]:
    """part 0 のファイルから 2 文書の `(バイト長, sha256)` を採る（グラフ記述 → モデル記述）。

    読むのはヘッダ + 2 文書だけ（part 0 は上限 32 MiB × 2 + 24 B なので全量に載る）。構造の
    妥当性は見ない — 検証は {@link karume.verify.verify_container} の担当で、ここは manifest へ
    焼く期待値を**置いた現物から**採るためだけの読み口である。

    MUST: 開くのは 1 度きりで、ヘッダの 24 B のために現物を丸ごと読まない — 呼び手
    （`karume.dist._materialize_family`）は「単一形かどうか」を判定する**前に**ここを通るので、
    全量読みにすると単一形の容器を weights 席へ挿した組み立てが、規則違反として落ちる前に
    数 GB を RAM へ載せる。
    """
    with path.open("rb") as handle:
        header = read_header(handle.read(HEADER_BYTES))
        if header.kind != "model":
            raise ContainerFormatError(f"{path}: krm でない（magic が {MAGIC_GRAPH!r}）")
        graph_bytes = handle.read(header.graph_length)
        model_bytes = handle.read(header.model_length)
    for label, raw, expected in (
        ("グラフ記述", graph_bytes, header.graph_length),
        ("モデル記述", model_bytes, header.model_length),
    ):
        if len(raw) != expected:
            raise ContainerFormatError(
                f"{path}: {label}が宣言の {expected} バイトに足りない（{len(raw)} バイト）"
            )
    return (
        DocumentRef(header.graph_length, hashlib.sha256(graph_bytes).hexdigest()),
        DocumentRef(header.model_length, hashlib.sha256(model_bytes).hexdigest()),
    )


def derive_part_offsets(part0_length: int, part_lengths: Sequence[int]) -> list[int]:
    """単一形で各 part が置かれる絶対 offset（§8）。

    MUST: **長さ 0 の part の前に詰め物を挿まない** — 挿むと、const が空の `krm` から抜いた `krg`
    と直接書いた `krg` がバイトでずれる（§3 / §9）。
    """
    offsets = [0]
    cursor = part0_length
    for length in part_lengths:
        start = cursor if length == 0 else align_up(cursor, BLOCK_START_ALIGN)
        offsets.append(start)
        cursor = start + length
    return offsets


# ---------------------------------------------------------------------------
# 配置（block / part の詰め方 — §3 / §4）
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Chunk:
    """1 block ぶんの実体の在処（テンソルキーとその中のバイト範囲・焼く詰め物の長さ）。"""

    key: str
    #: そのテンソルの payload 総バイト数（宣言から決まる値 — 実物と食い違えば fail loudly）。
    total: int
    begin: int
    end: int
    pad: int

    @property
    def length(self) -> int:
        return self.end - self.begin + self.pad


@dataclass(frozen=True)
class _Placed:
    id: str
    offset: int
    chunk: _Chunk


class _PartBuilder:
    """1 part ぶんの詰め込み器（offset は part 先頭からの相対）。"""

    def __init__(self) -> None:
        self._cursor = 0
        self._blocks: list[_Placed] = []

    def probe(self, lengths: Sequence[int]) -> int:
        cursor = self._cursor
        for length in lengths:
            cursor = align_up(cursor, BLOCK_START_ALIGN) + length
        return cursor

    def push(self, block_id: str, chunk: _Chunk) -> None:
        offset = align_up(self._cursor, BLOCK_START_ALIGN)
        self._blocks.append(_Placed(block_id, offset, chunk))
        self._cursor = offset + chunk.length

    @property
    def is_empty(self) -> bool:
        return not self._blocks

    @property
    def length(self) -> int:
        return self._cursor

    @property
    def blocks(self) -> list[_Placed]:
        return self._blocks


def _stored_chunk(key: str, total: int, begin: int, end: int, tail_pad: bool, where: str) -> _Chunk:
    """block へ収めるバイト範囲（末尾の詰め物は書き手が焼く — §4.1）。"""
    length = end - begin
    if length == 0:
        raise ContainerFormatError(f"{where}: 長さ 0 の block は作らない")
    if not tail_pad:
        # MUST: 中間 piece に詰め物は掛けられない（次の piece の先頭を潰す — 規則②）。
        if length % BLOCK_TAIL_ALIGN != 0:
            raise ContainerFormatError(
                f"{where}: 中間 piece の長さ {length} が {BLOCK_TAIL_ALIGN} の倍数でない"
            )
        return _Chunk(key, total, begin, end, 0)
    return _Chunk(key, total, begin, end, align_up(length, BLOCK_TAIL_ALIGN) - length)


def _cut_rows(
    payload: int, rows: int, block_bytes: int, where: str
) -> list[tuple[tuple[int, int], int, int, bool]]:
    """テンソルを block 上限以下の piece へ行境界で切る（切る必要が無ければ 1 本）。"""
    if align_up(payload, BLOCK_TAIL_ALIGN) <= block_bytes:
        return [((0, rows), 0, payload, True)]
    if rows <= 1:
        raise ContainerFormatError(f"{where}: 1 行で block 上限を超えるので切れない")
    if payload % rows != 0:
        raise ContainerFormatError(f"{where}: {payload} バイトが {rows} 行で割り切れない")
    row_bytes = payload // rows
    step = BLOCK_TAIL_ALIGN // math.gcd(row_bytes, BLOCK_TAIL_ALIGN)
    capacity = block_bytes // row_bytes
    rows_per_piece = capacity - (capacity % step)
    if rows_per_piece <= 0:
        raise ContainerFormatError(
            f"{where}: 1 行 {row_bytes} バイトでは block 上限 {block_bytes} に"
            f" {BLOCK_TAIL_ALIGN} バイト整列した行束を収められない"
        )
    cuts: list[tuple[tuple[int, int], int, int, bool]] = []
    begin = 0
    while begin < rows:
        end = min(begin + rows_per_piece, rows)
        cuts.append(((begin, end), begin * row_bytes, end * row_bytes, end == rows))
        begin += rows_per_piece
    return cuts


@dataclass(frozen=True)
class _TensorPlan:
    """1 initializer ぶんの「宣言から決まる」事実（実体を 1 バイトも読まずに決まる）。"""

    name: str
    encoding: Encoding
    payload: int
    rows: int
    scale_payload: int | None


def _numel(shape: Sequence[Any], where: str) -> int:
    count = 1
    for dim in shape:
        if not isinstance(dim, int) or dim < 0:
            raise ContainerFormatError(
                f"{where}: initializer の shape に記号次元は使えない（{dim!r}）"
            )
        count *= dim
    return count


def _plan_tensor(
    name: str, document: Mapping[str, Any], encoding: Encoding, where: str
) -> _TensorPlan:
    value = document["values"].get(name)
    if value is None:
        raise ContainerFormatError(f"{where}: `values` に dtype / shape 宣言が無い")
    shape = value["shape"]
    numel = _numel(shape, where)
    entry = codec_entry(encoding.codec)
    payload = payload_bytes(encoding.codec, numel, where)
    rows = shape[0] if shape else 1
    if entry.scale == "forbidden":
        if encoding.scale_key is not None or encoding.group_size is not None:
            raise ContainerFormatError(
                f"{where}: codec '{encoding.codec}' は scale / groupSize を持てない"
            )
        return _TensorPlan(name, encoding, payload, rows, None)
    if encoding.scale_key is None or encoding.group_size is None:
        raise ContainerFormatError(
            f"{where}: codec '{encoding.codec}' は scale と groupSize が要る"
        )
    row_axis = encoding.row_axis or 0
    if row_axis not in (0, 1):
        raise ContainerFormatError(f"{where}: rowAxis は 0 か 1（宣言は {row_axis}）")
    if len(shape) <= row_axis:
        raise ContainerFormatError(
            f"{where}: rowAxis {row_axis} に対して宣言 shape {list(shape)} の rank が足りない"
        )
    row_count = shape[row_axis]
    row_length = numel // row_count if row_count else 0
    if entry.grouping == "channel" and encoding.group_size != per_channel_group_size(row_length):
        raise ContainerFormatError(
            f"{where}: codec '{encoding.codec}' は per-channel なので groupSize は行長"
            f" {per_channel_group_size(row_length)} に等しい MUST（宣言は {encoding.group_size}）"
        )
    if entry.grouping == "group":
        size = encoding.group_size
        if size < MIN_GROUP_SIZE or size & (size - 1) != 0:
            raise ContainerFormatError(
                f"{where}: groupSize {size} が 2 冪かつ {MIN_GROUP_SIZE} 以上でない"
            )
        if row_length % size != 0:
            raise ContainerFormatError(
                f"{where}: 行長 {row_length} が groupSize {size} で割り切れない"
            )
    scale_payload = row_count * group_count(row_length, encoding.group_size) * 4
    return _TensorPlan(name, encoding, payload, rows, scale_payload)


class _Serial:
    """block id の連番（`w0001` / `s0002` … — 参照実装と同じ綴り）。"""

    def __init__(self) -> None:
        self._value = 0

    def next(self, prefix: str) -> str:
        self._value += 1
        return f"{prefix}{self._value:04d}"


@dataclass(frozen=True)
class _ConstRegion:
    length: int
    blocks: list[_Placed]
    bindings: list[tuple[str, str, BlockEncoding]]


def _build_const_region(
    graph_name: str, plans: Sequence[_TensorPlan], block_bytes: int
) -> _ConstRegion:
    builder = _PartBuilder()
    serial = _Serial()
    bindings: list[tuple[str, str, BlockEncoding]] = []
    for plan in plans:
        where = f"const {graph_name}/{plan.name}"
        chunk = _stored_chunk(plan.name, plan.payload, 0, plan.payload, True, where)
        if chunk.length > block_bytes:
            # const は piece 分割の機構を持たない（§3 — 束縛表が krg に無いので
            # piece 列を表せない）。
            raise ContainerFormatError(
                f"{where}: const は 1 block に収める必要があるが {chunk.length} バイト"
                f"（上限 {block_bytes}）"
            )
        block_id = serial.next("c")
        builder.push(block_id, chunk)
        scale_id: str | None = None
        if plan.scale_payload is not None:
            assert plan.encoding.scale_key is not None
            scale_id = serial.next("cs")
            builder.push(
                scale_id,
                _stored_chunk(
                    plan.encoding.scale_key,
                    plan.scale_payload,
                    0,
                    plan.scale_payload,
                    True,
                    f"{where} scale",
                ),
            )
        bindings.append(
            (
                plan.name,
                block_id,
                BlockEncoding(
                    plan.encoding.codec,
                    None if scale_id is None else (plan.encoding.row_axis or 0),
                    None if scale_id is None else plan.encoding.group_size,
                    scale_id,
                ),
            )
        )
    return _ConstRegion(builder.length, builder.blocks, bindings)


@dataclass(frozen=True)
class _DataPlan:
    parts: list[_PartBuilder]
    roles: dict[str, BlockRole]
    supplies: dict[str, WeightSupply]


def _plan_data_parts(
    graph_name: str, plans: Sequence[_TensorPlan], part_bytes: int, block_bytes: int
) -> _DataPlan:
    parts = [_PartBuilder()]
    roles: dict[str, BlockRole] = {}
    supplies: dict[str, WeightSupply] = {}
    serial = _Serial()

    def next_id(prefix: str, role: BlockRole) -> str:
        block_id = serial.next(prefix)
        roles[block_id] = role
        return block_id

    def place_group(items: Sequence[tuple[str, _Chunk]], where: str) -> None:
        """「同じ part に置く」必要がある一群をまとめて置く。入らなければ新しい part を開く。"""
        lengths = [chunk.length for _, chunk in items]
        if _PartBuilder().probe(lengths) > part_bytes:
            raise ContainerFormatError(
                f"{where}: 同一 part に置く必要がある一群が part 長 {part_bytes} を超える"
            )
        target = parts[-1]
        if target.probe(lengths) > part_bytes:
            target = _PartBuilder()
            parts.append(target)
        for block_id, chunk in items:
            target.push(block_id, chunk)

    for plan in plans:
        where = f"weight {graph_name}/{plan.name}"
        cuts = _cut_rows(plan.payload, plan.rows, block_bytes, where)
        if len(cuts) > 1 and (plan.encoding.row_axis or 0) != 0:
            # 規則④: scale の行範囲が piece の行範囲に対応しないので分割できない。
            raise ContainerFormatError(
                f"{where}: rowAxis {plan.encoding.row_axis} の initializer は piece 分割できない"
            )
        piece_ids = [next_id("w", "weight") for _ in cuts]
        head: list[tuple[str, _Chunk]] = [
            (
                piece_ids[0],
                _stored_chunk(
                    plan.name, plan.payload, cuts[0][1], cuts[0][2], cuts[0][3], f"{where} piece 0"
                ),
            )
        ]
        scale_id: str | None = None
        if plan.scale_payload is not None:
            assert plan.encoding.scale_key is not None
            scale_id = next_id("s", "scale")
            head.append(
                (
                    scale_id,
                    _stored_chunk(
                        plan.encoding.scale_key,
                        plan.scale_payload,
                        0,
                        plan.scale_payload,
                        True,
                        f"{where} scale",
                    ),
                )
            )
        # 規則③: piece 1 と scale は同一 part。残りの piece は独立に置ける。
        place_group(head, where)
        for index, (_, begin, end, last) in enumerate(cuts):
            if index == 0:
                continue
            place_group(
                [
                    (
                        piece_ids[index],
                        _stored_chunk(
                            plan.name, plan.payload, begin, end, last, f"{where} piece {index}"
                        ),
                    )
                ],
                f"{where} piece {index}",
            )
        encoding = BlockEncoding(
            plan.encoding.codec,
            None if scale_id is None else (plan.encoding.row_axis or 0),
            None if scale_id is None else plan.encoding.group_size,
            scale_id,
        )
        supplies[plan.name] = (
            WeightSupply(encoding, block=piece_ids[0])
            if len(cuts) == 1
            else WeightSupply(
                encoding,
                pieces=tuple((piece_ids[index], cut[0]) for index, cut in enumerate(cuts)),
            )
        )
    return _DataPlan([part for part in parts if not part.is_empty], roles, supplies)


@dataclass(frozen=True)
class _AssetPlan:
    parts: list[_PartBuilder]
    #: 資産名 → モデル記述の 1 件。
    bindings: dict[str, AssetRecord]


def _asset_view(name: str, asset: AssetInput) -> memoryview:
    """資産 1 本の実体（遅延の呼び出しはここで解決する）。呼び手は使い終わったら手放す。"""
    raw = asset.payload() if callable(asset.payload) else asset.payload
    try:
        return memoryview(raw).cast("B")
    except TypeError as cause:
        raise ContainerFormatError(f"資産 '{name}': payload がバイト列でない") from cause


def _plan_asset_parts(
    assets: Mapping[str, AssetInput], part_bytes: int, block_bytes: int
) -> _AssetPlan:
    """資産を**重みと同居しない part** へ並べる（§4.2 — 区間読みの資産を走査の後ろに置かない）。

    配置の順は**呼び手が渡した順**（PLE は token 順で渡る — 走査型の取得元で隣り合う token が
    別 part へ散らない）。descriptor の `assets` の綴りは正準直列化が code point 順にする。

    資産は piece 分割しない（行の刻みを知るのは役割ごとの索引で、descriptor は「名前 →
    block」までしか持たない）— 1 block に収まらなければ fail loudly。`dedicated_part` の資産は
    専用 part へ単独で置き、そうでない資産どうしは 1 part に同居してよい。
    """
    parts: list[_PartBuilder] = []
    bindings: dict[str, AssetRecord] = {}
    current: _PartBuilder | None = None
    for serial, (name, asset) in enumerate(assets.items()):
        where = f"資産 '{name}'"
        if not name:
            raise ContainerFormatError("資産名が空文字列")
        if not isinstance(asset.role, str) or not asset.role:
            raise ContainerFormatError(f"{where}: 役割が非空文字列でない（{asset.role!r}）")
        if not isinstance(asset.length, int) or isinstance(asset.length, bool):
            raise ContainerFormatError(f"{where}: 論理長が整数でない（{asset.length!r}）")
        chunk = _stored_chunk(name, asset.length, 0, asset.length, True, where)
        if chunk.length > block_bytes:
            raise ContainerFormatError(
                f"{where}: 資産は 1 block に収める必要があるが {chunk.length} バイト"
                f"（上限 {block_bytes}）"
            )
        if chunk.length > part_bytes:
            raise ContainerFormatError(
                f"{where}: block {chunk.length} バイトが part 長 {part_bytes} を超える"
            )
        if asset.dedicated_part or current is None or current.probe([chunk.length]) > part_bytes:
            current = _PartBuilder()
            parts.append(current)
        # block id は配置順の 0 始まり十進（`a.0` / `a.1` …）。
        block_id = f"a.{serial}"
        current.push(block_id, chunk)
        bindings[name] = AssetRecord(block_id, asset.role, asset.length)
        if asset.dedicated_part:
            # 単独 MUST: 次の資産はこの part に同居させない（§4.2）。
            current = None
    return _AssetPlan(parts, bindings)


# ---------------------------------------------------------------------------
# 実体の取り出し（1 本ずつ・持ち越しは連続するあいだだけ）
# ---------------------------------------------------------------------------


class _Payloads:
    """テンソルを**1 本ずつ**引く口。同じキーが連続するあいだだけ持ち越す。

    MUST: 持ち越しは 1 本。piece 列は同じ part の中で連続して置かれるので、この 1 本で
    「同じテンソルを piece の数だけ引き直す」を避けつつ、同時に生きる実体は常に 1 本になる。
    """

    def __init__(self, tensors: Mapping[str, Buffer]) -> None:
        self._tensors = tensors
        self._key: str | None = None
        self._view: memoryview | None = None

    def view(self, key: str, expected: int) -> memoryview:
        if key != self._key:
            self.release()
            try:
                raw = self._tensors[key]
            except KeyError:
                raise ContainerFormatError(f"テンソル '{key}' の実体が渡されていない") from None
            view = memoryview(raw).cast("B")
            if view.nbytes != expected:
                raise ContainerFormatError(
                    f"テンソル '{key}': 実体 {view.nbytes} バイトが宣言から決まる"
                    f" {expected} バイトと違う"
                )
            self._view = view
            self._key = key
        assert self._view is not None
        return self._view

    def chunk(self, chunk: _Chunk) -> memoryview:
        return self.view(chunk.key, chunk.total)[chunk.begin : chunk.end]

    def release(self) -> None:
        self._view = None
        self._key = None


class _Sources(Mapping[str, Buffer]):
    """テンソルと資産を 1 つの口に束ねる（block の `key` からどちらかを引く）。

    資産名とテンソルキーが衝突すると block の実体が黙って入れ替わるので、束ねる時点で
    全件列挙して拒否する。
    """

    def __init__(self, tensors: Mapping[str, Buffer], assets: Mapping[str, AssetInput]) -> None:
        clash = sorted(set(assets) & set(tensors))
        if clash:
            raise ContainerFormatError(f"資産名がテンソルキーと衝突している: {', '.join(clash)}")
        self._tensors = tensors
        self._assets = dict(assets)

    def __getitem__(self, key: str) -> Buffer:
        asset = self._assets.get(key)
        if asset is None:
            return self._tensors[key]
        return _asset_view(key, asset)

    def __iter__(self) -> Iterator[str]:
        return iter([*self._tensors, *self._assets])

    def __len__(self) -> int:
        return len(self._tensors) + len(self._assets)


def _region_stream(
    blocks: Sequence[_Placed], payloads: _Payloads
) -> Iterator[tuple[str | None, bytes | memoryview]]:
    """1 region（= part）ぶんを `(block id, バイト列)` の列として順に流す。

    MUST: **書き出しと sha256 はこの 1 本の生成器を共有する**（§4.1 の「詰め物込みで digest を
    取る」が両側で別々に綴られると、part の sha256 と実バイトが静かにずれる）。block に属さない
    隙間（整列のための詰め物）は id が `None` で、part の sha256 にだけ入る。
    """
    cursor = 0
    for placed in blocks:
        if placed.offset > cursor:
            yield None, PAD_BYTE * (placed.offset - cursor)
        yield placed.id, payloads.chunk(placed.chunk)
        if placed.chunk.pad:
            yield placed.id, PAD_BYTE * placed.chunk.pad
        cursor = placed.offset + placed.chunk.length


def _emit_region(
    blocks: Sequence[_Placed], payloads: _Payloads, target: Path | None
) -> tuple[str, dict[str, str]]:
    """region を 1 度の走査で流し、part の sha256 と block ごとの sha256（詰め物込み — §4.1）を
    採る。`target` が在れば**同じ走査で**そこへ書く。

    MUST: 書き出しと digest は {@link _region_stream} の 1 本を共有する（§4.1 の「詰め物込みで
    digest を取る」が両側で別々に綴られると、part の sha256 と実バイトが静かにずれる）。
    書きながら採れるので、分割形では実体を**1 度しか引かない**（格納変換は引くたびに走る）。
    """
    region = hashlib.sha256()
    digests: dict[str, Any] = {}
    handle = None if target is None else target.open("wb")
    try:
        for block_id, chunk in _region_stream(blocks, payloads):
            if handle is not None:
                handle.write(chunk)
            region.update(chunk)
            if block_id is not None:
                digests.setdefault(block_id, hashlib.sha256()).update(chunk)
    finally:
        if handle is not None:
            handle.close()
    return region.hexdigest(), {key: digest.hexdigest() for key, digest in digests.items()}


# ---------------------------------------------------------------------------
# 書き出し
# ---------------------------------------------------------------------------


def _owned_initializers(document: Mapping[str, Any]) -> list[str]:
    """バイトを自分で持つ initializer（shared は貸し手の常駐重みが実体なので外れる）。"""
    return [
        name
        for name, declaration in document["initializers"].items()
        if not declaration.get("shared", False)
    ]


def _assert_binding_cover(
    expected: Sequence[str], bindings: Mapping[str, Encoding], graph_name: str, what: str
) -> None:
    """束縛の過不足を**全件列挙で**拒否する（規則⑤ — 黙って太った / 痩せた配布形を作らない）。"""
    missing = sorted(name for name in expected if name not in bindings)
    surplus = sorted(set(bindings) - set(expected))
    if missing or surplus:
        raise ContainerFormatError(
            f"グラフ '{graph_name}' の束縛が{what}と一致しない:"
            f" 不足 [{', '.join(missing)}] / 余剰 [{', '.join(surplus)}]"
        )


def _split_initializers(
    document: Mapping[str, Any], bindings: Mapping[str, Encoding], graph_name: str
) -> tuple[list[_TensorPlan], list[_TensorPlan]]:
    """`(const 行き, 重み行き)` の配置計画（`krm` — 束縛は自前バイトの全 initializer を覆う）。"""
    owned = _owned_initializers(document)
    _assert_binding_cover(owned, bindings, graph_name, "initializer 宣言")
    consts: list[_TensorPlan] = []
    weights: list[_TensorPlan] = []
    for name in owned:
        plan = _plan_tensor(name, document, bindings[name], f"{graph_name}/{name}")
        (consts if name.startswith(CONST_KEY_PREFIX) else weights).append(plan)
    return consts, weights


def _const_initializers(
    document: Mapping[str, Any], bindings: Mapping[str, Encoding], graph_name: str
) -> list[_TensorPlan]:
    """const 領域の配置計画（`krg` — 束縛表に載るのは const だけ）。

    `krg` は**重みの束縛表を持たない**（§9）ので、重みの束縛を渡されたら余剰として拒否する。
    重みが要る initializer の宣言自体はグラフに残る（要求は `values` の宣言から導く）。
    """
    expected = [name for name in _owned_initializers(document) if name.startswith(CONST_KEY_PREFIX)]
    _assert_binding_cover(expected, bindings, graph_name, "const 供給の集合")
    return [
        _plan_tensor(name, document, bindings[name], f"{graph_name}/{name}") for name in expected
    ]


def _assert_graph_name(graph_name: str) -> None:
    if not GRAPH_NAME_PATTERN.match(graph_name):
        raise ContainerFormatError(
            f"グラフ名 '{graph_name}' が語彙外（{GRAPH_NAME_PATTERN.pattern}）"
        )


def _assert_block_budget(count: int) -> None:
    if count > MAX_BLOCKS:
        raise ContainerFormatError(f"block 件数 {count} が上限 {MAX_BLOCKS} を超える")


def write_graph_container(
    path: Path,
    graph: IrGraph,
    const_tensors: Mapping[str, Buffer],
    bindings: Mapping[str, Encoding],
    *,
    graph_name: str,
    block_bytes: int = BLOCK_MAX_BYTES,
) -> Path:
    """`krg`（グラフ容器）を書く — `[ヘッダ][グラフ記述][詰め物][const 領域]`（§9）。

    `bindings` は const の束縛だけを持つ（重みの束縛表は `krg` に無い）。同じグラフと同じ const から
    書いた `krg` は、`krm` からバイトコピーで抜いた `krg` と**バイト同一**になる。
    """
    _assert_graph_name(graph_name)
    if block_bytes > BLOCK_MAX_BYTES:
        raise ContainerFormatError(f"block 上限 {block_bytes} が仕様の {BLOCK_MAX_BYTES} を超える")
    document = ir_v2_document(graph)
    consts = _const_initializers(document, bindings, graph_name)
    region = _build_const_region(graph_name, consts, block_bytes)
    _assert_block_budget(len(region.blocks))
    payloads = _Payloads(const_tensors)
    _, block_hashes = _emit_region(region.blocks, payloads, None)
    payloads.release()
    descriptor = _graph_descriptor(graph_name, document, region, block_hashes)
    graph_bytes = serialize_graph_descriptor(descriptor)
    header = write_header(kind="graph", graph_length=len(graph_bytes), model_length=0)
    offsets = derive_part_offsets(HEADER_BYTES + len(graph_bytes), [region.length])
    path.parent.mkdir(parents=True, exist_ok=True)
    payloads = _Payloads(const_tensors)
    with path.open("wb") as handle:
        handle.write(header)
        handle.write(graph_bytes)
        handle.write(PAD_BYTE * (offsets[1] - HEADER_BYTES - len(graph_bytes)))
        for _, piece in _region_stream(region.blocks, payloads):
            handle.write(piece)
    payloads.release()
    return path


def _graph_descriptor(
    graph_name: str,
    document: Mapping[str, Any],
    region: _ConstRegion,
    block_hashes: Mapping[str, str],
) -> GraphDescriptor:
    return GraphDescriptor(
        graphs={graph_name: document},
        const_length=region.length,
        const_blocks=tuple(
            ConstBlockRecord(placed.id, placed.offset, placed.chunk.length, block_hashes[placed.id])
            for placed in region.blocks
        ),
        constants=tuple(
            ConstantBinding(graph_name, name, block_id, encoding)
            for name, block_id, encoding in region.bindings
        ),
    )


def write_model_container(
    path: Path,
    graph: IrGraph,
    tensors: Mapping[str, Buffer],
    bindings: Mapping[str, Encoding],
    *,
    graph_name: str,
    provenance: Provenance,
    assets: Mapping[str, AssetInput] = {},
    part_bytes: int = DEFAULT_PART_BYTES,
    block_bytes: int = BLOCK_MAX_BYTES,
    single: bool = False,
) -> list[Path]:
    """`krm`（モデル容器）を書く。返すのは書いたファイルの並び（part 0 から）。

    `tensors` は**テンソルキー → 生バイト**の口で、実体は 1 本ずつ引いて 1 本ずつ手放す
    （`Mapping` を遅延にすれば全量はメモリに載らない）。`bindings` は shared でない initializer
    ぜんぶを覆う MUST で、`const.` で始まるキーは const 領域（part 1）へ、それ以外は重み
    （part 2 以降）へ行く。

    `assets`（資産名 → {@link AssetInput}）は重みの part の**後ろ**に専用の part を並べて置く
    （§4.2 — 区間読みの資産を重みの走査の後ろに置かない）。並ぶ順は**渡した順**で、
    `dedicated_part` の資産は 1 block = 1 part になる。

    ファイル名は単一形が `path` そのもの、分割形が `<stem>-NNNNN-of-NNNNN<suffix>`（part 0 から）。
    """
    _assert_graph_name(graph_name)
    if block_bytes > BLOCK_MAX_BYTES:
        raise ContainerFormatError(f"block 上限 {block_bytes} が仕様の {BLOCK_MAX_BYTES} を超える")
    if not 0 < part_bytes <= PART_MAX_BYTES:
        raise ContainerFormatError(
            f"part 長 {part_bytes} が 1..{PART_MAX_BYTES}（part 長の天井）の外"
        )
    document = ir_v2_document(graph)
    consts, weights = _split_initializers(document, bindings, graph_name)
    region = _build_const_region(graph_name, consts, block_bytes)
    data = _plan_data_parts(graph_name, weights, part_bytes, block_bytes)
    asset_plan = _plan_asset_parts(assets, part_bytes, block_bytes)
    builders = [*data.parts, *asset_plan.parts]
    roles: dict[str, BlockRole] = {
        **data.roles,
        **{record.block: "asset" for record in asset_plan.bindings.values()},
    }
    _assert_block_budget(len(region.blocks) + sum(len(part.blocks) for part in builders))
    if len(builders) + 2 > MAX_PARTS:
        raise ContainerFormatError(f"part 件数 {len(builders) + 2} が上限 {MAX_PARTS} を超える")

    # ② 実体を 1 本ずつ流して block / part の sha256 を採る。
    #
    # **分割形はここで data part を書いてしまう**（part 0 は別ファイルなので最後に書ける）—
    # 実体を引くのは 1 度きりで、格納変換も 1 度しか走らない。単一形は part 0 が先頭に来るうえ、
    # その中身が後ろの part の sha256 に依存するので**2 度引く**（一時ファイルを挟むと
    # モデル全量ぶんの書き込みが 1 回増える — モジュール doc の選択）。
    path.parent.mkdir(parents=True, exist_ok=True)
    written = [path] if single else container_paths(path, len(builders) + 2)
    payloads = _Payloads(_Sources(tensors, assets))
    region_hash, block_hashes = _emit_region(
        region.blocks, payloads, None if single else written[1]
    )
    parts = [PartRecord(1, region.length, region_hash)]
    data_blocks: list[DataBlockRecord] = []
    for index, builder in enumerate(builders):
        part_hash, hashes = _emit_region(
            builder.blocks, payloads, None if single else written[index + 2]
        )
        block_hashes.update(hashes)
        parts.append(PartRecord(index + 2, builder.length, part_hash))
        for placed in builder.blocks:
            data_blocks.append(
                DataBlockRecord(
                    placed.id,
                    index + 2,
                    placed.offset,
                    placed.chunk.length,
                    hashes[placed.id],
                    roles[placed.id],
                )
            )
    payloads.release()

    # ③ 2 文書を組んで書く。
    graph_descriptor = _graph_descriptor(graph_name, document, region, block_hashes)
    model_descriptor = ModelDescriptor(
        parts=tuple(parts),
        blocks=tuple(data_blocks),
        binding={graph_name: data.supplies},
        assets=dict(asset_plan.bindings),
        provenance=provenance,
    )
    graph_bytes = serialize_graph_descriptor(graph_descriptor)
    model_bytes = serialize_model_descriptor(model_descriptor)
    header = write_header(
        kind="model", graph_length=len(graph_bytes), model_length=len(model_bytes)
    )
    part0_length = HEADER_BYTES + len(graph_bytes) + len(model_bytes)
    if part0_length > PART_MAX_BYTES:
        raise ContainerFormatError(
            f"part 0 の長さ {part0_length} が part 長の天井 {PART_MAX_BYTES} を超える"
        )
    if single:
        regions = [region.blocks, *(builder.blocks for builder in builders)]
        offsets = derive_part_offsets(part0_length, [part.length for part in parts])
        payloads = _Payloads(_Sources(tensors, assets))
        with path.open("wb") as handle:
            handle.write(header)
            handle.write(graph_bytes)
            handle.write(model_bytes)
            for index, blocks in enumerate(regions):
                handle.write(PAD_BYTE * (offsets[index + 1] - handle.tell()))
                for _, piece in _region_stream(blocks, payloads):
                    handle.write(piece)
        payloads.release()
        return [path]

    # data part は②で据わっている。残るのは part 0 だけ。
    with written[0].open("wb") as handle:
        handle.write(header)
        handle.write(graph_bytes)
        handle.write(model_bytes)
    return written


# ---------------------------------------------------------------------------
# 読み手（自己検査用 — §2 / §5 の宣言検査 + block の sha256 検証）
# ---------------------------------------------------------------------------


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContainerFormatError(message)


def _require_object(value: Any, where: str) -> dict[str, Any]:
    _require(isinstance(value, dict), f"{where} がオブジェクトでない")
    return value


def _require_keys(
    obj: Mapping[str, Any], required: Sequence[str], optional: Sequence[str], where: str
) -> None:
    for key in required:
        _require(key in obj, f"{where}.{key} が無い")
    known = set(required) | set(optional)
    for key in obj:
        _require(key in known, f"{where}: 未知のキー '{key}'")


def _require_index(value: Any, where: str) -> int:
    _require(
        isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= _MAX_SAFE_INTEGER,
        f"{where} が非負の安全整数でない: {value!r}",
    )
    return value


def _require_string(value: Any, where: str) -> str:
    _require(isinstance(value, str) and value != "", f"{where} が非空文字列でない")
    return value


def _require_sha256(value: Any, where: str) -> str:
    _require(
        isinstance(value, str) and SHA256_HEX_PATTERN.match(value) is not None,
        f"{where} が sha256（小文字 16 進 64 文字）でない: {value!r}",
    )
    return value


def _require_block_id(value: Any, where: str) -> str:
    block_id = _require_string(value, where)
    _require(
        BLOCK_ID_PATTERN.match(block_id) is not None,
        f"{where}: block id '{block_id}' が語彙外（{BLOCK_ID_PATTERN.pattern}）",
    )
    return block_id


def _parse_json_document(raw: bytes, where: str) -> Any:
    _require(
        len(raw) <= MAX_DESCRIPTOR_BYTES,
        f"{where} が上限 {MAX_DESCRIPTOR_BYTES} バイトを超える: {len(raw)}",
    )
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as cause:
        raise ContainerFormatError(f"{where} が UTF-8 として不正: {cause}") from cause
    try:
        # MUST: `parse_constant` を塞ぐ（NaN / Infinity リテラルは §0 で禁止）。
        parsed = json.loads(
            text,
            parse_constant=lambda literal: _require(False, f"{where}: 非有限数リテラル {literal}"),
        )
    except json.JSONDecodeError as cause:
        raise ContainerFormatError(f"{where} が JSON として不正: {cause}") from cause
    _assert_finite(parsed, where, 0)
    return parsed


def _assert_finite(value: Any, where: str, depth: int) -> None:
    """`1e999` のように**構文は正当だが Infinity に丸まる**数を弾く（§0）。"""
    _require(depth <= MAX_JSON_DEPTH, f"{where}: 入れ子が深さ上限 {MAX_JSON_DEPTH} を超えた")
    if isinstance(value, float):
        _require(math.isfinite(value), f"{where}: 非有限数（1e999 等の溢れを含む）")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _assert_finite(item, f"{where}[{index}]", depth + 1)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _assert_finite(item, f"{where}.{key}", depth + 1)


def _parse_encoding(value: Any, where: str) -> BlockEncoding:
    obj = _require_object(value, where)
    _require_keys(obj, ["codec", "packing"], ["rowAxis", "groupSize", "scale", "zeroPoint"], where)
    codec = _require_string(obj["codec"], f"{where}.codec")
    entry = codec_entry(codec)
    packing = _require_object(obj["packing"], f"{where}.packing")
    _require_keys(packing, ["blockElements", "blockBytes", "alignBytes"], [], f"{where}.packing")
    declared = (packing["blockElements"], packing["blockBytes"], packing["alignBytes"])
    expected = (entry.packing.block_elements, entry.packing.block_bytes, entry.packing.align_bytes)
    _require(
        declared == expected,
        f"{where}.packing: codec '{codec}' の台帳は {expected} だが宣言は {declared}"
        f"（別の版の台帳で書かれた資産）",
    )
    quantized = entry.scale == "required"
    for key in ("rowAxis", "groupSize", "scale"):
        _require(
            (key in obj) == quantized,
            f"{where}.{key}: codec '{codec}' は"
            + ("量子化なので必須" if quantized else "量子化でないので書けない"),
        )
    _require("zeroPoint" not in obj, f"{where}.zeroPoint: 初版の台帳は zeroPoint を持てない")
    if not quantized:
        return BlockEncoding(codec)
    row_axis = _require_index(obj["rowAxis"], f"{where}.rowAxis")
    _require(row_axis in (0, 1), f"{where}.rowAxis は 0 か 1（宣言は {row_axis}）")
    scale = _require_object(obj["scale"], f"{where}.scale")
    _require_keys(scale, ["block", "dtype"], [], f"{where}.scale")
    _require(
        scale["dtype"] in SCALE_DTYPES,
        f"{where}.scale.dtype が受理集合（{' / '.join(SCALE_DTYPES)}）に無い: {scale['dtype']!r}",
    )
    return BlockEncoding(
        codec,
        row_axis,
        _require_index(obj["groupSize"], f"{where}.groupSize"),
        _require_block_id(scale["block"], f"{where}.scale.block"),
    )


def _assert_block_shape(offset: int, length: int, where: str) -> None:
    _require(offset % BLOCK_START_ALIGN == 0, f"{where}: offset {offset} が 64 の倍数でない")
    _require(length != 0, f"{where}: length が 0")
    _require(length % BLOCK_TAIL_ALIGN == 0, f"{where}: length {length} が 4 の倍数でない")
    _require(
        length <= BLOCK_MAX_BYTES,
        f"{where}: length {length} が block 上限 {BLOCK_MAX_BYTES} を超える",
    )


def _assert_no_overlap(
    blocks: Sequence[tuple[str, int, int]], region_length: int, where: str
) -> None:
    cursor = 0
    for block_id, offset, length in sorted(blocks, key=lambda block: block[1]):
        _require(offset >= cursor, f"{where}: block '{block_id}' が直前の block と重なる")
        cursor = offset + length
        _require(
            cursor <= region_length,
            f"{where}: block '{block_id}' が領域長 {region_length} をはみ出す（末尾 {cursor}）",
        )


def _parse_graph_descriptor(raw: bytes) -> GraphDescriptor:
    where = "graphDescriptor"
    root = _require_object(_parse_json_document(raw, where), where)
    _require_keys(root, ["format", "version", "capabilities", "graphs", "const"], [], where)
    _require(root["format"] == "karume-container", f"{where}.format が 'karume-container' でない")
    _require(root["version"] == CONTAINER_VERSION, f"{where}.version が {CONTAINER_VERSION} でない")

    capabilities = _require_object(root["capabilities"], f"{where}.capabilities")
    _require_keys(capabilities, ["ops", "features"], [], f"{where}.capabilities")
    _require(capabilities["features"] == [], f"{where}.capabilities.features: 初版は空配列のみ")

    graphs = _require_object(root["graphs"], f"{where}.graphs")
    _require(len(graphs) > 0, f"{where}.graphs が空")
    _require(
        len(graphs) <= MAX_GRAPHS,
        f"{where}.graphs の件数 {len(graphs)} が上限 {MAX_GRAPHS} を超える",
    )
    union: set[str] = set()
    for name, declaration in graphs.items():
        _require(
            GRAPH_NAME_PATTERN.match(name) is not None,
            f"{where}.graphs: グラフ名 '{name}' が語彙外",
        )
        union |= set(_graph_required_ops(declaration, f"{where}.graphs['{name}']"))
    _require(
        capabilities["ops"] == sorted(union),
        f"{where}.capabilities.ops が graphs の requires.ops の和集合と一致しない",
    )

    region = _require_object(root["const"], f"{where}.const")
    _require_keys(region, ["length", "blocks", "constants"], [], f"{where}.const")
    length = _require_index(region["length"], f"{where}.const.length")
    blocks: list[ConstBlockRecord] = []
    previous = -1
    for index, raw_block in enumerate(region["blocks"]):
        block_where = f"{where}.const.blocks[{index}]"
        obj = _require_object(raw_block, block_where)
        _require_keys(obj, ["id", "offset", "length", "sha256"], [], block_where)
        record = ConstBlockRecord(
            _require_block_id(obj["id"], f"{block_where}.id"),
            _require_index(obj["offset"], f"{block_where}.offset"),
            _require_index(obj["length"], f"{block_where}.length"),
            _require_sha256(obj["sha256"], f"{block_where}.sha256"),
        )
        _assert_block_shape(record.offset, record.length, block_where)
        _require(record.offset > previous, f"{block_where}: offset {record.offset} が昇順でない")
        previous = record.offset
        blocks.append(record)
    _assert_no_overlap(
        [(block.id, block.offset, block.length) for block in blocks], length, f"{where}.const"
    )

    constants: list[ConstantBinding] = []
    for index, raw_entry in enumerate(region["constants"]):
        entry_where = f"{where}.const.constants[{index}]"
        obj = _require_object(raw_entry, entry_where)
        _require_keys(obj, ["graph", "initializer", "block", "encoding"], [], entry_where)
        constants.append(
            ConstantBinding(
                _require_string(obj["graph"], f"{entry_where}.graph"),
                _require_string(obj["initializer"], f"{entry_where}.initializer"),
                _require_block_id(obj["block"], f"{entry_where}.block"),
                _parse_encoding(obj["encoding"], f"{entry_where}.encoding"),
            )
        )
    descriptor = GraphDescriptor(graphs, length, tuple(blocks), tuple(constants))
    _validate_graph_descriptor(descriptor)
    return descriptor


def _graph_required_ops(declaration: Any, where: str) -> list[str]:
    obj = _require_object(declaration, where)
    _require(obj.get("format") == IR_V2_FORMAT, f"{where}.format が '{IR_V2_FORMAT}' でない")
    _require(obj.get("version") == IR_V2_VERSION, f"{where}.version が {IR_V2_VERSION} でない")
    requires = _require_object(obj.get("requires"), f"{where}.requires")
    # TS の読み手（`format/ir.ts` の `checkKeys(requires, ["ops"], [])`）と同じ受理集合。
    _require_keys(requires, ["ops"], [], f"{where}.requires")
    ops = requires["ops"]
    _require(isinstance(ops, list), f"{where}.requires.ops が配列でない: {ops!r}")
    return [_require_string(op, f"{where}.requires.ops[]") for op in ops]


def _graph_initializers(declaration: Mapping[str, Any], where: str) -> dict[str, bool]:
    """initializer 名 → shared か（束縛表の突合集合を決めるのに要る唯一の情報）。"""
    out: dict[str, bool] = {}
    declared = _require_object(declaration.get("initializers"), f"{where}.initializers")
    for name, value in declared.items():
        obj = _require_object(value, f"{where}.initializers['{name}']")
        _require_keys(obj, [], ["shared"], f"{where}.initializers['{name}']")
        if "shared" in obj:
            _require(
                obj["shared"] is True,
                f"{where}.initializers['{name}'].shared: 書けるのは true だけ",
            )
        out[name] = obj.get("shared", False) is True
    return out


def _validate_graph_descriptor(descriptor: GraphDescriptor) -> None:
    where = "graphDescriptor.const"
    _require(
        len(descriptor.const_blocks) <= MAX_BLOCKS,
        f"{where}.blocks の件数が上限 {MAX_BLOCKS} を超える",
    )
    ids = {block.id for block in descriptor.const_blocks}
    _require(len(ids) == len(descriptor.const_blocks), f"{where}.blocks: block id が重複")
    referenced: set[str] = set()

    def claim(block_id: str, by: str) -> None:
        _require(block_id in ids, f"{by}: const 目次に無い block '{block_id}'")
        _require(
            block_id not in referenced,
            f"{by}: block '{block_id}' が二重に束縛されている（1 block ≤ 1 binding）",
        )
        referenced.add(block_id)

    seen: set[tuple[str, str]] = set()
    for entry in descriptor.constants:
        entry_where = f"{where}.constants['{entry.graph}/{entry.initializer}']"
        declaration = descriptor.graphs.get(entry.graph)
        _require(declaration is not None, f"{entry_where}: 未宣言のグラフ '{entry.graph}'")
        assert declaration is not None
        initializers = _graph_initializers(declaration, f"graphDescriptor.graphs['{entry.graph}']")
        _require(
            entry.initializer in initializers,
            f"{entry_where}: グラフに initializer '{entry.initializer}' が無い",
        )
        _require(
            not initializers[entry.initializer],
            f"{entry_where}: shared 宣言は const では供給できない",
        )
        key = (entry.graph, entry.initializer)
        _require(key not in seen, f"{entry_where}: (graph, initializer) が重複")
        seen.add(key)
        claim(entry.block, entry_where)
        if entry.encoding.scale_block is not None:
            claim(entry.encoding.scale_block, f"{entry_where}.encoding.scale")
    for block_id in sorted(ids - referenced):
        raise ContainerFormatError(
            f"{where}: block '{block_id}' がどこからも参照されていない（余剰）"
        )


def _parse_supply(value: Any, where: str) -> WeightSupply:
    obj = _require_object(value, where)
    _require_keys(obj, ["encoding"], ["block", "pieces"], where)
    encoding = _parse_encoding(obj["encoding"], f"{where}.encoding")
    has_block = "block" in obj
    has_pieces = "pieces" in obj
    _require(has_block != has_pieces, f"{where}: 'block' と 'pieces' のどちらか一方だけを書く")
    if has_block:
        return WeightSupply(encoding, block=_require_block_id(obj["block"], f"{where}.block"))
    pieces: list[tuple[str, tuple[int, int]]] = []
    cursor = 0
    for index, raw_piece in enumerate(obj["pieces"]):
        piece_where = f"{where}.pieces[{index}]"
        piece = _require_object(raw_piece, piece_where)
        _require_keys(piece, ["block", "rows"], [], piece_where)
        rows = piece["rows"]
        _require(isinstance(rows, list) and len(rows) == 2, f"{piece_where}.rows の長さが 2 でない")
        begin = _require_index(rows[0], f"{piece_where}.rows[0]")
        end = _require_index(rows[1], f"{piece_where}.rows[1]")
        _require(end > begin, f"{piece_where}.rows [{begin}, {end}) が空区間")
        # 規則④: 行範囲を隙間なく被覆する（末尾 = shape[0] は宣言 shape を知る合流層が見る）。
        _require(begin == cursor, f"{piece_where}: 行 {cursor} から続かない（宣言は {begin}）")
        cursor = end
        pieces.append((_require_block_id(piece["block"], f"{piece_where}.block"), (begin, end)))
    _require(len(pieces) >= 2, f"{where}.pieces は 2 本以上")
    return WeightSupply(encoding, pieces=tuple(pieces))


def _parse_model_descriptor(raw: bytes) -> ModelDescriptor:
    where = "modelDescriptor"
    root = _require_object(_parse_json_document(raw, where), where)
    _require_keys(
        root,
        ["format", "version", "codecs", "parts", "blocks", "binding", "assets", "provenance"],
        [],
        where,
    )
    _require(root["format"] == "karume-model", f"{where}.format が 'karume-model' でない")
    _require(root["version"] == CONTAINER_VERSION, f"{where}.version が {CONTAINER_VERSION} でない")

    parts: list[PartRecord] = []
    for index, raw_part in enumerate(root["parts"]):
        part_where = f"{where}.parts[{index}]"
        obj = _require_object(raw_part, part_where)
        _require_keys(obj, ["index", "length", "sha256"], [], part_where)
        record = PartRecord(
            _require_index(obj["index"], f"{part_where}.index"),
            _require_index(obj["length"], f"{part_where}.length"),
            _require_sha256(obj["sha256"], f"{part_where}.sha256"),
        )
        _require(record.index == index + 1, f"{part_where}.index が {index + 1} でない")
        _require(
            record.length <= PART_MAX_BYTES,
            f"{part_where}.length {record.length} が part 長の天井を超える",
        )
        parts.append(record)
    _require(len(parts) > 0, f"{where}.parts が空（part 1 は長さ 0 でも宣言する）")
    _require(len(parts) <= MAX_PARTS, f"{where}.parts の件数が上限 {MAX_PARTS} を超える")

    blocks: list[DataBlockRecord] = []
    for index, raw_block in enumerate(root["blocks"]):
        block_where = f"{where}.blocks[{index}]"
        obj = _require_object(raw_block, block_where)
        _require_keys(obj, ["id", "part", "offset", "length", "sha256", "role"], [], block_where)
        role = _require_string(obj["role"], f"{block_where}.role")
        _require(
            role in ("weight", "scale", "zero-point", "asset"),
            f"{block_where}.role が語彙外: {role}",
        )
        record = DataBlockRecord(
            _require_block_id(obj["id"], f"{block_where}.id"),
            _require_index(obj["part"], f"{block_where}.part"),
            _require_index(obj["offset"], f"{block_where}.offset"),
            _require_index(obj["length"], f"{block_where}.length"),
            _require_sha256(obj["sha256"], f"{block_where}.sha256"),
            role,  # type: ignore[arg-type]
        )
        _require(record.part >= 2, f"{block_where}.part が 2 未満")
        _require(record.part <= len(parts), f"{block_where}.part: 未宣言の part {record.part}")
        _assert_block_shape(record.offset, record.length, block_where)
        blocks.append(record)
    for part in parts:
        _assert_no_overlap(
            [
                (block.id, block.offset, block.length)
                for block in blocks
                if block.part == part.index
            ],
            part.length,
            f"{where} part {part.index}",
        )

    binding: dict[str, dict[str, WeightSupply]] = {}
    for graph_name, raw_supplies in _require_object(root["binding"], f"{where}.binding").items():
        supplies = _require_object(raw_supplies, f"{where}.binding['{graph_name}']")
        binding[graph_name] = {
            name: _parse_supply(value, f"{where}.binding['{graph_name}']['{name}']")
            for name, value in supplies.items()
        }

    assets: dict[str, AssetRecord] = {}
    for name, raw_asset in _require_object(root["assets"], f"{where}.assets").items():
        asset_where = f"{where}.assets['{name}']"
        obj = _require_object(raw_asset, asset_where)
        _require_keys(obj, ["block", "role", "length"], [], asset_where)
        assets[name] = AssetRecord(
            _require_block_id(obj["block"], f"{asset_where}.block"),
            _require_string(obj["role"], f"{asset_where}.role"),
            _require_index(obj["length"], f"{asset_where}.length"),
        )

    prov = _require_object(root["provenance"], f"{where}.provenance")
    _require_keys(
        prov, ["license"], ["notice", "upstreamRevision", "writer"], f"{where}.provenance"
    )
    # 省略可の 3 欄も、在れば TS の読み手と同じく非空文字列を要る（受理集合の一致）。
    optional = {
        key: _require_string(prov[key], f"{where}.provenance.{key}") if key in prov else None
        for key in ("notice", "upstreamRevision", "writer")
    }
    provenance = Provenance(
        _require_string(prov["license"], f"{where}.provenance.license"),
        optional["notice"],
        optional["upstreamRevision"],
        optional["writer"],
    )

    descriptor = ModelDescriptor(tuple(parts), tuple(blocks), binding, assets, provenance)
    _validate_model_descriptor(descriptor, root["codecs"])
    return descriptor


def _validate_model_descriptor(descriptor: ModelDescriptor, codecs: Any) -> None:
    where = "modelDescriptor"
    by_id = {block.id: block for block in descriptor.blocks}
    _require(len(by_id) == len(descriptor.blocks), f"{where}.blocks: block id が重複")
    referenced: set[str] = set()

    def claim(block_id: str, role: BlockRole, by: str) -> DataBlockRecord:
        block = by_id.get(block_id)
        _require(block is not None, f"{by}: 目次に無い block '{block_id}'")
        assert block is not None
        _require(block.role == role, f"{by}: block '{block_id}' の role は '{block.role}'")
        _require(
            block_id not in referenced,
            f"{by}: block '{block_id}' が二重に束縛されている（1 block ≤ 1 binding）",
        )
        referenced.add(block_id)
        return block

    used: set[str] = set()
    for graph_name, supplies in descriptor.binding.items():
        for name, supply in supplies.items():
            entry_where = f"{where}.binding['{graph_name}']['{name}']"
            used.add(supply.encoding.codec)
            ids = (
                [supply.block]
                if supply.block is not None
                else [block for block, _ in supply.pieces or ()]
            )
            first = claim(ids[0], "weight", entry_where).part
            for block_id in ids[1:]:
                claim(block_id, "weight", entry_where)
            if supply.encoding.scale_block is not None:
                # 規則③: companion scale は実体（piece 列なら piece 1）と同一 part。
                scale_part = claim(
                    supply.encoding.scale_block, "scale", f"{entry_where}.encoding.scale"
                ).part
                _require(
                    scale_part == first,
                    f"{entry_where}.encoding.scale: scale が part {scale_part}・"
                    f"実体の先頭が part {first}（同一 part MUST）",
                )
    for name, asset in descriptor.assets.items():
        asset_where = f"{where}.assets['{name}']"
        block = claim(asset.block, "asset", asset_where)
        # 詰め物の量まで宣言で閉じる（消費側が末尾の 0x00 を推測で剥がない — §2.2）。
        _require(
            align_up(asset.length, BLOCK_TAIL_ALIGN) == block.length,
            f"{asset_where}.length: 論理長 {asset.length} を {BLOCK_TAIL_ALIGN} の倍数へ"
            f"切り上げた値が block '{asset.block}' の長さ {block.length} と違う",
        )
    for block_id in sorted(set(by_id) - referenced):
        raise ContainerFormatError(
            f"{where}: block '{block_id}' がどこからも参照されていない（余剰）"
        )
    declared = [_require_string(codec, f"{where}.codecs[]") for codec in codecs]
    for codec in declared:
        codec_entry(codec)
    _require(
        declared == sorted(used),
        f"{where}.codecs {declared} が束縛表の codec 集合 {sorted(used)} と一致しない",
    )


def _validate_against_graph(model: ModelDescriptor, graph: GraphDescriptor) -> None:
    """2 文書のあいだの突合（§2 / §5 の規則⑤）。"""
    _require(
        model.parts[0].length == graph.const_length,
        f"part 1 の長さ {model.parts[0].length} が const.length {graph.const_length} と違う",
    )
    const_ids = {block.id for block in graph.const_blocks}
    for block in model.blocks:
        _require(
            block.id not in const_ids,
            f"block id '{block.id}' が const 目次とモデル目次で衝突（素集合 MUST）",
        )
    supplied = {(entry.graph, entry.initializer) for entry in graph.constants}
    for graph_name in model.binding:
        _require(
            graph_name in graph.graphs,
            f"modelDescriptor.binding: 未宣言のグラフ '{graph_name}'",
        )
    for graph_name, declaration in graph.graphs.items():
        initializers = _graph_initializers(declaration, f"graphDescriptor.graphs['{graph_name}']")
        expected = sorted(
            name
            for name, shared in initializers.items()
            if not shared and (graph_name, name) not in supplied
        )
        bound = model.binding.get(graph_name, {})
        missing = [name for name in expected if name not in bound]
        surplus = sorted(set(bound) - set(expected))
        _require(
            not missing and not surplus,
            f"グラフ '{graph_name}' の束縛表が initializer 宣言と一致しない:"
            f" 不足 [{', '.join(missing)}] / 余剰 [{', '.join(surplus)}]",
        )


@dataclass(frozen=True)
class _PartSlice:
    path: Path
    offset: int
    length: int


class ReadContainer:
    """開いたコンテナ（自己検査用の読み手）。block は要るときに取り、取るたび sha256 を検証する。"""

    def __init__(
        self,
        header: ContainerHeader,
        graph_bytes: bytes,
        model_bytes: bytes,
        graph: GraphDescriptor,
        model: ModelDescriptor | None,
        slices: Sequence[_PartSlice],
    ) -> None:
        self.header = header
        self.graph_descriptor_bytes = graph_bytes
        self.model_descriptor_bytes = model_bytes
        self.graph = graph
        self.model = model
        self._slices = list(slices)
        self._located: dict[str, tuple[int, int, int, str]] = {}
        for block in graph.const_blocks:
            self._located[block.id] = (1, block.offset, block.length, block.sha256)
        for block in model.blocks if model is not None else ():
            self._located[block.id] = (block.part, block.offset, block.length, block.sha256)

    @property
    def block_ids(self) -> list[str]:
        return list(self._located)

    def _read(self, part: int, offset: int, length: int) -> bytes:
        _require(0 <= part < len(self._slices), f"part {part} は無い")
        target = self._slices[part]
        _require(
            offset + length <= target.length,
            f"part {part}: {offset}..{offset + length} が part 長 {target.length} をはみ出す",
        )
        with target.path.open("rb") as handle:
            handle.seek(target.offset + offset)
            raw = handle.read(length)
        _require(len(raw) == length, f"part {part}: {length} バイト読めなかった")
        return raw

    def block(self, block_id: str) -> bytes:
        """block を 1 本取り、宣言の sha256 と突き合わせて返す（§7 の cold 経路）。"""
        found = self._located.get(block_id)
        _require(found is not None, f"未宣言の block '{block_id}'")
        assert found is not None
        part, offset, length, expected = found
        raw = self._read(part, offset, length)
        actual = hashlib.sha256(raw).hexdigest()
        _require(
            actual == expected,
            f"block '{block_id}' の sha256 が宣言と違う（宣言 {expected} / 実物 {actual}）",
        )
        return raw

    def verify_blocks(self) -> dict[str, str]:
        """全 block を取り直して sha256 を検証する（block id → 宣言の sha256 を返す）。"""
        for block_id in self._located:
            self.block(block_id)
        return {block_id: found[3] for block_id, found in self._located.items()}

    def verify_parts(self) -> None:
        """part 全体の sha256 を宣言と突き合わせる（公開・再梱包の突合用 — §7）。

        part は block と違って 1 GiB まで在りうるので、**刻んで digest する**
        （全量を器に載せない）。
        """
        _require(self.model is not None, "krg は parts を宣言しない")
        assert self.model is not None
        for part in self.model.parts:
            digest = hashlib.sha256()
            for offset in range(0, part.length, _READ_CHUNK_BYTES):
                digest.update(
                    self._read(part.index, offset, min(_READ_CHUNK_BYTES, part.length - offset))
                )
            actual = digest.hexdigest()
            _require(
                actual == part.sha256,
                f"part {part.index} の sha256 が宣言と違う（宣言 {part.sha256} / 実物 {actual}）",
            )

    def extract_graph(self) -> bytes:
        """`[ヘッダ'][グラフ記述][詰め物][const 領域]` を組む（§9 のバイトコピー抽出）。"""
        length = self.graph.const_length
        region = b"" if length == 0 else self._read(1, 0, length)
        if self.model is not None:
            actual = hashlib.sha256(region).hexdigest()
            expected = self.model.parts[0].sha256
            _require(
                actual == expected,
                f"const 領域の sha256 が宣言と違う（宣言 {expected} / 実物 {actual}）",
            )
        header = write_header(
            kind="graph", graph_length=len(self.graph_descriptor_bytes), model_length=0
        )
        offsets = derive_part_offsets(HEADER_BYTES + len(self.graph_descriptor_bytes), [length])
        body = header + self.graph_descriptor_bytes
        return body + PAD_BYTE * (offsets[1] - len(body)) + region


def read_container(paths: Sequence[Path]) -> ReadContainer:
    """書いたものを読み直す（ヘッダ・2 文書の parse・§2 / §5 の宣言検査）。

    `paths` は単一形なら 1 本、分割形なら **part 0 から順に**並べた part 列。
    """
    _require(len(paths) > 0, "コンテナのファイルが 1 本も渡されていない")
    first = Path(paths[0])
    with first.open("rb") as handle:
        head = handle.read(HEADER_BYTES)
    header = read_header(head)
    part0 = header.part0_length
    with first.open("rb") as handle:
        handle.seek(HEADER_BYTES)
        graph_bytes = handle.read(header.graph_length)
        model_bytes = handle.read(header.model_length)
    _require(len(graph_bytes) == header.graph_length, "グラフ記述が宣言の長さに足りない")
    _require(len(model_bytes) == header.model_length, "モデル記述が宣言の長さに足りない")

    graph = _parse_graph_descriptor(graph_bytes)
    model: ModelDescriptor | None = None
    if header.kind == "model":
        model = _parse_model_descriptor(model_bytes)
        _validate_against_graph(model, graph)
    part_lengths = [graph.const_length] if model is None else [part.length for part in model.parts]

    if len(paths) == 1:
        offsets = derive_part_offsets(part0, part_lengths)
        total = offsets[-1] + part_lengths[-1] if part_lengths else part0
        size = first.stat().st_size
        _require(size == total, f"単一形の長さ {size} が宣言から導いた {total} と違う")
        slices = [_PartSlice(first, 0, part0)] + [
            _PartSlice(first, offsets[index + 1], length)
            for index, length in enumerate(part_lengths)
        ]
    else:
        _require(
            len(paths) == len(part_lengths) + 1,
            f"part が {len(paths)} 本だが宣言は {len(part_lengths) + 1} 本",
        )
        slices = [_PartSlice(first, 0, part0)]
        for index, length in enumerate(part_lengths):
            path = Path(paths[index + 1])
            _require(
                path.stat().st_size == length,
                f"part {index + 1} の長さ {path.stat().st_size} が宣言 {length} と違う",
            )
            slices.append(_PartSlice(path, 0, length))
        _require(
            first.stat().st_size == part0,
            f"part 0 の長さ {first.stat().st_size} が宣言 {part0} と違う",
        )
    return ReadContainer(header, graph_bytes, model_bytes, graph, model, slices)
