"""据えたコンテナ（`krm`）を**格納のまま**読み戻す（recipe 側の読み手・ADR 0065 決定 2）。

台本や配布 recipe が自分の書き出しを検算する（代表例: 上流 checkpoint とのバイト一致
〈vowel_detector〉・i4 系列からの重み読み戻し〈irodori〉・QAT の格納監査〈gemma4_qat〉 —
呼び手の全量は rg で引く）。どれも要るのは「テンソルキー → 宣言した格納形と生 payload」で、
意味論 f32 へ戻す前の姿である。

core（`karume.verify.verify_container`）が合流まで済ませた供給計画をそのまま畳むだけなので、
規則の写しは 1 行も持たない。block の sha256 は取り出しのたびに `ReadContainer.block` が
突き合わせる（container-v1 §7）ので、ここを通った payload は宣言と一致している。

MUST: 読むのは**配布形そのもの**（part 列）。代表 path 1 本を開く形にすると、分割形では
part 0（2 文書だけ）を読んで「テンソルが 1 本も無い」になる。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from karume.container import CONST_KEY_PREFIX, Provenance, codec_entry
from karume.dist import component_parts
from karume.verify import VerifiedContainer, verify_container


class ContainerReadError(ValueError):
    """容器から読みたいものが無い（資産の不在・`krg` を渡した）。

    リポの流儀は `Error` サブクラス（`DistError` / `PleError` / `ContainerFormatError`）。
    `KeyError` は `repr` が二重引用符で包まれて診断が読みにくいうえ、呼び手の `except` が
    「辞書の引き損ない」と区別できない。
    """


#: 読み手の入口（代表 path か、{@link open_container} で開いた容器）。
Source = Path | VerifiedContainer


def open_container(path: Path) -> VerifiedContainer:
    """部品 1 つを開いて合流まで通す（**開いた結果を使い回す**ための入口）。

    MUST: 同じ容器から何度も読む呼び手はここで 1 度だけ開く — 下の読み口へ代表 path を渡すと
    1 回ごとに `verify_container` が descriptor の parse と合流をやり直す（PLE の検収門は
    block ごとに読むので、block 数ぶん繰り返す形になっていた）。
    """
    return verify_container(component_parts(path))


def _opened(source: Source) -> VerifiedContainer:
    """代表 path なら開き、開いた容器ならそのまま返す。"""
    return source if isinstance(source, VerifiedContainer) else open_container(source)


@dataclass(frozen=True)
class StoredTensor:
    """initializer 1 本の格納実体（宣言 + 連結済みの payload）。"""

    #: codec の layout（`f32` / `f16` / `bf16` / `i32` / `i8` / `i4` / `i2`）。
    layout: str
    #: 宣言の具体 shape（論理形 — packed 格納でもここは要素数のまま）。
    shape: tuple[int, ...]
    #: 格納の生バイト（piece 列は行の順に連結済み）。
    payload: bytes
    #: companion scale の生バイト（f32・量子化 codec のみ）。
    scale: bytes | None = None
    #: 量子化 codec の group 長（per-channel は行長そのもの）。
    group_size: int | None = None
    #: 持ち上げ定数（`const.` 接頭辞のテンソルキー）か。
    constant: bool = False


def read_stored(source: Source) -> dict[str, StoredTensor]:
    """コンテナ 1 本の initializer を**テンソルキー**で引ける表にする。

    IR v2 では initializer の名前がテンソルキーそのものなので（docs/ir-v2.md）、返る鍵は
    書き手が `tensors` へ渡した鍵と同じである。共有宣言（貸し手を指す席）は供給を持たないので
    現れない — 実体は貸し手のキーで 1 本だけ並ぶ。
    """
    verified = _opened(source)
    read = verified.read
    stored: dict[str, StoredTensor] = {}
    for bound in verified.graphs.values():
        values = bound.declaration["values"]
        for name, supply in bound.supplies.items():
            payload = b"".join(
                read.block(block.id)[: block.payload_bytes] for block in supply.blocks
            )
            scale = (
                None
                if supply.scale is None
                else read.block(supply.scale.id)[: supply.scale.payload_bytes]
            )
            stored[name] = StoredTensor(
                layout=codec_entry(supply.encoding.codec).layout,
                shape=tuple(values[name]["shape"]),
                payload=payload,
                scale=scale,
                group_size=supply.encoding.group_size,
                constant=name.startswith(CONST_KEY_PREFIX),
            )
    return stored


def weight_tensors(stored: Mapping[str, StoredTensor]) -> dict[str, StoredTensor]:
    """持ち上げ定数を除いた席（= 上流の重みに対応するテンソルキーだけ）。"""
    return {key: value for key, value in stored.items() if not value.constant}


def read_layouts(source: Source) -> dict[str, str]:
    """テンソルキー → 格納の layout（**宣言だけ** — 実体は 1 バイトも読まない）。

    「貸し手の主表が i8 で常駐しているか」のような、実体を要らない突合の席。IR v2 のグラフ
    記述は `storage` を持たない（正本は束縛表の codec — docs/ir-v2.md）ので、格納を知る道は
    合流結果 1 本きりである。
    """
    verified = _opened(source)
    return {
        name: codec_entry(supply.encoding.codec).layout
        for bound in verified.graphs.values()
        for name, supply in bound.supplies.items()
    }


def read_asset_declarations(source: Source) -> dict[str, tuple[str, int]]:
    """資産名 → `(役割, 論理長)`（**宣言だけ** — payload は 1 バイトも読まない）。

    索引が名指しする block が実際に容器へ入っているかを、実体を読まずに突き合わせる席
    （PLE の `values` / `scales` は合計数 GB になるので、組み立ての門で読む形にはできない）。
    """
    model = _opened(source).read.model
    if model is None:
        raise ContainerReadError(f"{source}: krg には資産が無い")
    return {name: (record.role, record.length) for name, record in model.assets.items()}


def read_asset(source: Source, name: str) -> bytes:
    """容器の資産 1 本の payload（宣言の論理長で切る — 末尾の詰め物は剥がして返す）。

    資産は「重みではないが同じ容器で配るバイト列」の席（PLE 索引・`rope_base` — ADR 0109
    決定 4）で、宣言の `length` が先に来る。消費側が末尾の 0x00 を推測で剥がさないための
    規律なので、読み手もここ 1 本を通す。
    """
    read = _opened(source).read
    model = read.model
    if model is None:
        raise ContainerReadError(f"{source}: krg には資産が無い")
    record = model.assets.get(name)
    if record is None:
        raise ContainerReadError(f"{source}: 資産 '{name}' が無い（宣言: {sorted(model.assets)}）")
    return read.block(record.block)[: record.length]


def read_provenance(source: Source) -> Provenance:
    """容器が名乗る出所（`provenance` — container-v1 §2.3。**宣言だけ**を読む）。

    `krg` は出所を引き継がない（container-v1 §9）ので、渡されたら落とす。
    """
    model = _opened(source).read.model
    if model is None:
        raise ContainerReadError(f"{source}: krg は provenance を持たない")
    return model.provenance
