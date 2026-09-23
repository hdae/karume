"""旧配布形（safetensors 方言の shard 列）の**読み取り専用**の層（移行 CLI 専用）。

配布形は `krm`（container-v1）へ移ったので、safetensors 方言を**書く**コードはもう無い。
それでも HF 上の旧リポはリリースまで生きるので、`karume migrate` が旧形を読む経路だけが
ここに残る（container-v1 §12「旧形式を読む処理はこの CLI にだけ置く」）。

ここが読むのは旧規則の配布形も含む列（単一ファイル / fat グラフ shard / 尾部スラック /
piece キー `<親名>#NNNNN-of-NNNNN`）で、現行の門は掛けない（掛けたら移行できない）。分割
テンソルは読むときに**親 1 本へ畳む**ので、呼び手は常に「親の全バイト」だけを見る。

MUST: torch を持ち込まない。扱うのは**バイト列と宣言だけ**で、テンソルを torch へ起こして
持ち回ると dtype の往復（f16 / i4 の器）で沈黙誤値を作る余地が生まれる。

MUST: 新しいものを**書く**ために使わない。連番のファイル名規約（`<stem>-NNNNN-of-NNNNN`）は
コンテナ側（{@link karume.container.numbered_name}）が正本で、ここはそれを借りて旧形の列を
解決するだけである。
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from karume.container import resolve_sequence
from karume.ir import IR_METADATA_KEY
from karume.verify import assert_reader_layout

#: 生バイトの読み出し単位。数 GB を丸読みしないための唯一の要件で、値自体は素の I/O 単位。
_CHUNK_BYTES = 1 << 20

#: safetensors のヘッダ長欄のバイト数。
_HEADER_LENGTH_BYTES = 8

#: piece キーの連番の桁数（旧 shard 仕様 v3 — `packages/runtime` 側の正規表現と同じ綴り）。
_PIECE_DIGITS = 5

#: piece キーの綴り（`<親名>#NNNNN-of-NNNNN`）。親名は貪欲に取る — `#` を含む親名は旧書き手の
#: 出力に現れないので、末尾の連番だけが分割の印になる。
_PIECE_PATTERN = re.compile(rf"^(.+)#(\d{{{_PIECE_DIGITS}}})-of-(\d{{{_PIECE_DIGITS}}})$")


class LegacyFormatError(ValueError):
    """旧配布形が読めない（ヘッダの矛盾・piece 列の不整合・重複テンソル）。"""


def parse_piece_key(key: str) -> tuple[str, int, int] | None:
    """piece キー → `(親名, index, count)`（piece でなければ None）。

    MUST: `count >= 2` と `1 <= index <= count` を**ここで**見る。範囲外の綴り
    （`#00003-of-00002`・1 本しかない piece 列）を piece として通すと、それが列の進行状態の
    初期値になって違反の帰属が「piece 列の並び」へ移る。実際にはそのキー自体が旧配布形の
    誤りなので、piece と解釈せず 1 本のテンソルとして運ぶほうが直す側にとって決定的になる。
    """
    match = _PIECE_PATTERN.fullmatch(key)
    if match is None:
        return None
    index, count = int(match.group(2)), int(match.group(3))
    if count < 2 or not 1 <= index <= count:
        return None
    return match.group(1), index, count


def resolve_shards(path: Path) -> tuple[Path, ...]:
    """旧コンポーネントの代表 path → 実在する shard 列（読む順 = 連番順）。

    連番規約は新しい part と同じ（container-v1 §8）ので、解決は
    {@link karume.container.resolve_sequence} の 1 本道を借りる。
    """
    return resolve_sequence(path)


@dataclass(frozen=True)
class StoredEntry:
    """旧コンテナのテンソル 1 本の宣言（safetensors dtype・**論理** shape・バイト長）。"""

    name: str
    dtype: str
    shape: tuple[int, ...]
    nbytes: int


@dataclass(frozen=True)
class SourceTensor:
    """入力側のテンソル 1 本 — **親としての宣言**と、現物の在処。

    分割テンソルは piece が別々のファイルに散っているので、在処は 1 点ではなく**区間の列**に
    なる（`(ファイル, ファイル先頭からの絶対 offset, バイト長)` を piece の index 順に）。
    丸ごと 1 本のときは要素 1 つの列で、以降の経路（sha256・写し・行範囲の切り出し）は
    区間の列だけを見るので分岐しない。
    """

    entry: StoredEntry
    segments: tuple[tuple[Path, int, int], ...]


def _read_header(path: Path) -> tuple[Mapping[str, Any], int]:
    """ヘッダ JSON と**データ節の絶対開始位置**を返す。

    NOTE: レイアウト規則（既知 dtype・宣言バイト長の一致・隙間なし・整列）は
    {@link karume.verify.assert_reader_layout} の担当で、呼び出し側が**先に**通す。ここは
    通った後の読み出しだけなので、素で添字してよい。
    """
    size = path.stat().st_size
    with path.open("rb") as handle:
        header_length = int.from_bytes(handle.read(_HEADER_LENGTH_BYTES), "little")
        # 宣言長はファイル実長で拘束する（不正な 8 バイトをそのまま read すると巨大確保になる）。
        if header_length <= 0 or header_length > size - _HEADER_LENGTH_BYTES:
            raise LegacyFormatError(
                f"{path}: ヘッダ長 {header_length} がファイル長 {size} と矛盾する"
            )
        header = json.loads(handle.read(header_length))
    if not isinstance(header, dict):
        raise LegacyFormatError(f"{path}: safetensors ヘッダが最上位オブジェクトでない")
    return header, _HEADER_LENGTH_BYTES + header_length


def safetensors_header(path: Path) -> Mapping[str, Any]:
    """ヘッダ JSON だけを読む（数 GB のペイロードを舐めない）。"""
    return _read_header(path)[0]


def _read_metadata(path: Path, header: Mapping[str, Any]) -> dict[str, str]:
    """`__metadata__` を文字列 → 文字列のまま取り出す（**逐語**で持ち回す唯一の入口）。"""
    raw = header.get("__metadata__", {})
    if not isinstance(raw, dict) or any(
        not isinstance(key, str) or not isinstance(value, str) for key, value in raw.items()
    ):
        raise LegacyFormatError(f"{path}: __metadata__ が文字列 → 文字列のマップでない")
    return dict(raw)


@dataclass(frozen=True)
class _Fragment:
    """読み込み中の piece 1 本（畳む前 — 連番と収容 shard の並びを見るための材料）。"""

    shard: int
    index: int
    count: int
    entry: StoredEntry
    segment: tuple[Path, int, int]


def _join_pieces(name: str, fragments: Sequence[_Fragment]) -> SourceTensor:
    """piece 列を親 1 本へ畳む（旧読み手契約 5 を全部ここで見る）。

    `fragments` は**読む順**（shard 番号順）に並んでいる。畳んだ宣言は親の dtype・全体
    shape（先頭次元 = 各 piece の行数の和）・合計バイト長で、在処は piece の区間列になる。
    """
    shards = [fragment.shard for fragment in fragments]
    repeated = sorted({shard for shard in shards if shards.count(shard) > 1})
    if repeated:
        raise LegacyFormatError(
            f"テンソル '{name}': shard{repeated} に同じ親の piece が 2 本ある"
            "（piece は連続する shard に 1 本ずつ）"
        )
    # count >= 2 と index の域は `parse_piece_key` が既に見ている（域外の綴りは piece と解釈
    # されず、名前そのままの 1 本として運ばれる）。
    count = fragments[0].count
    if len(fragments) != count:
        raise LegacyFormatError(
            f"テンソル '{name}': piece が {len(fragments)} 本で宣言の総数 {count} と合わない"
        )
    head = fragments[0].entry
    rows = 0
    previous: int | None = None
    for position, fragment in enumerate(fragments, start=1):
        entry = fragment.entry
        if fragment.count != count:
            raise LegacyFormatError(
                f"テンソル '{name}': piece の総数が {count} と {fragment.count} で食い違っている"
            )
        if fragment.index != position:
            raise LegacyFormatError(
                f"テンソル '{name}': shard 順で {position} 本目の piece が index {fragment.index}"
                "（index は shard 順に 1 から増える）"
            )
        if previous is not None and fragment.shard != previous + 1:
            raise LegacyFormatError(
                f"テンソル '{name}': piece {fragment.index} が shard[{fragment.shard}]・前の"
                f" piece が shard[{previous}]（piece は連続する shard に 1 本ずつ）"
            )
        if entry.dtype != head.dtype:
            raise LegacyFormatError(
                f"テンソル '{name}': piece {fragment.index} の dtype が {entry.dtype}"
                f"（piece 1 は {head.dtype}）"
            )
        if entry.shape[1:] != head.shape[1:]:
            raise LegacyFormatError(
                f"テンソル '{name}': piece {fragment.index} の残り次元 {list(entry.shape[1:])} が"
                f" piece 1 の {list(head.shape[1:])} と違う"
            )
        if not entry.shape or entry.shape[0] < 1:
            raise LegacyFormatError(f"テンソル '{name}': piece {fragment.index} が 1 行未満")
        if position < count and entry.nbytes % 4:
            raise LegacyFormatError(
                f"テンソル '{name}': 末尾でない piece {fragment.index} が {entry.nbytes} バイトで"
                " 4 の倍数でない（読み手が親バッファへオフセット書きできない）"
            )
        rows += entry.shape[0]
        previous = fragment.shard
    return SourceTensor(
        entry=StoredEntry(
            name=name,
            dtype=head.dtype,
            shape=(rows, *head.shape[1:]),
            nbytes=sum(fragment.entry.nbytes for fragment in fragments),
        ),
        segments=tuple(fragment.segment for fragment in fragments),
    )


def read_component(paths: Sequence[Path]) -> tuple[dict[str, str], dict[str, SourceTensor]]:
    """旧コンポーネントの現物を読み、`(先頭 shard の __metadata__, 名前 → 現物)` を返す。

    受けるのは**旧規則の配布形も含む**列（単一ファイル / fat グラフ shard / 尾部スラック）。
    それでも「先頭だけが `karume_ir` を持つ」「同名テンソルが 2 本無い」は旧規則でも成り立って
    いた不変条件なので、ここで見る（破れているものを移行すると、どちらのバイト列を配るかが
    一意に決まらない）。

    分割テンソル（piece キー）は**親 1 本へ畳む**（{@link _join_pieces} が旧読み手契約 5 を
    検査する）。畳むのは形だけで判定する — 入力の `karume_ir` はまだ parse していないので
    「親名が宣言に在るか」はここでは見ず、移行側の突合（`migrate._assert_tensor_cover`）が
    余剰として落とす。
    """
    metadata: dict[str, str] = {}
    stored: dict[str, SourceTensor] = {}
    fragments: dict[str, list[_Fragment]] = {}
    owner: dict[str, Path] = {}
    for index, path in enumerate(paths):
        if not path.is_file():
            raise LegacyFormatError(f"移行の入力が無い: {path}")
        assert_reader_layout(path)
        header, data_start = _read_header(path)
        raw = _read_metadata(path, header)
        if index == 0:
            metadata = raw
        elif raw:
            raise LegacyFormatError(
                f"{path}: shard[{index}] が __metadata__ を持っている"
                f"（{sorted(raw)} — メタデータを載せるのは先頭 shard だけ）"
            )
        for name, spec in header.items():
            if name == "__metadata__":
                continue
            if name in owner:
                raise LegacyFormatError(
                    f"テンソル '{name}' が {owner[name].name} と {path.name} に重複している"
                )
            owner[name] = path
            begin, end = spec["data_offsets"]
            entry = StoredEntry(
                name=name, dtype=spec["dtype"], shape=tuple(spec["shape"]), nbytes=end - begin
            )
            segment = (path, data_start + begin, end - begin)
            parsed = parse_piece_key(name)
            if parsed is None:
                stored[name] = SourceTensor(entry=entry, segments=(segment,))
                continue
            parent, piece_index, piece_count = parsed
            fragments.setdefault(parent, []).append(
                _Fragment(
                    shard=index,
                    index=piece_index,
                    count=piece_count,
                    entry=replace(entry, name=parent),
                    segment=segment,
                )
            )
    for parent in sorted(fragments):
        if parent in stored:
            raise LegacyFormatError(
                f"テンソル '{parent}' が丸ごとと piece の両方でコンテナに居る"
                "（1 テンソルはどちらか一方 MUST）"
            )
        stored[parent] = _join_pieces(parent, fragments[parent])
    if IR_METADATA_KEY not in metadata:
        raise LegacyFormatError(
            f"{paths[0]}: __metadata__.{IR_METADATA_KEY} が無い（Karume の旧配布形ではない）"
        )
    return metadata, stored


def _range_chunks(source: SourceTensor, begin: int, end: int) -> Iterator[bytes]:
    """親の**バイト範囲** `[begin, end)` を読み出し単位ずつ流す（丸読みしない）。

    範囲は区間の列（= piece の並び）を跨いでよい — 旧形の切り目は移行後の block の切り目と
    無関係なので、読み手は「親の連続バイト列」だけを見る。
    """
    cursor = 0
    for path, offset, length in source.segments:
        stop = cursor + length
        if stop <= begin or cursor >= end:
            cursor = stop
            continue
        lower, upper = max(begin, cursor), min(end, stop)
        with path.open("rb") as handle:
            handle.seek(offset + lower - cursor)
            remaining = upper - lower
            while remaining:
                chunk = handle.read(min(_CHUNK_BYTES, remaining))
                if not chunk:
                    raise LegacyFormatError(
                        f"{path}: テンソル '{source.entry.name}' の宣言"
                        f"（{source.entry.nbytes} バイト）に対しデータ節が途中で尽きた"
                    )
                remaining -= len(chunk)
                yield chunk
        cursor = stop


def payload_chunks(source: SourceTensor) -> Iterator[bytes]:
    """1 本ぶん（親の全バイト）を読み出し単位ずつ流す。"""
    return _range_chunks(source, 0, source.entry.nbytes)
