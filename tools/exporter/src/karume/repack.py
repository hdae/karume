"""既存の配布形を現行の shard 仕様へ**詰め替える**（ADR 0081 の移行経路）。

    karume repack ../../outputs/series/vowel-detector-crnn-epoch3/model.safetensors

旧規則で焼いたコンポーネント（fat グラフ shard・単一ファイル・尾部スラック・旧い容量で
割った連番）を、現行の規則（グラフ shard = `karume_ir` だけ / 全 shard ≤
{@link karume.shards.SHARD_BYTE_LIMIT} / 最小本数→均し / 常時分割 / 容量に収まらないテンソルは
piece 列）で並べ直す。**再 export ではない** — GPTQ 校正はリグと実行のたびに
丸め解が動くので（docs/research/ 2026-08-24）、出荷済みの系列を焼き直すと配るバイトが変わる。
ここが動かすのは**容器の詰め方だけ**で、テンソルの生バイトと `karume_ir` は 1 バイトも変えない。

分割テンソル（`<親名>#NNNNN-of-NNNNN`）は読むときに**親 1 本へ畳む**ので、入力が piece でも
丸ごとでも、また出力の piece 本数が入力と違っても、突き合わせるのは常に「親の全バイト」に
なる（不変条件 1 が容量を変えた詰め替えでもそのまま効く）。

## 不変条件（MUST — 全部が書き出し後の**読み戻し**で機械検証される）

1. **テンソルの生バイトが旧と同一**: 名前 → (dtype, shape, バイト長, sha256) の写像が旧新で
   一致する。詰め替えは所属 shard と並び順（と piece の切り目）しか動かさないので、1 本でも
   中身が動いたらそれは詰め替えではない。
2. **`karume_ir` が逐語同一**: 旧グラフ shard のメタデータ文字列を**そのまま**新しい先頭
   shard へ載せる（parse → 再 serialize しない — 正準化・キー順・空白の差が「意味は同じだが
   バイトは別」を作り、pin した sha256 と配布形の対応が黙って切れる）。
3. **他の `__metadata__` キーも保存**: 先頭 shard の `__metadata__` を丸ごと持ち回る。
4. **{@link karume.verify.verify_shards} 全門緑**: 書いたものが IR v1 の全規則を満たす。

## 据え替え

既定は**インプレース**（同ディレクトリ）。旧ファイル群を読みながら同名を上書きすることは
できないので、`pipeline.publish_model` と同じ規律で進む: 一時 path の連番として書く →
検証門を全部通す → `os.replace` で本番の連番へ据える → **このコンポーネントの出力名**に
一致する前回の残骸（単一ファイル / 別の分割数）を消す。同居する `io.*.safetensors`
（E2E フィクスチャ）や provenance の類には 1 バイトも触らない — 拾うのは
{@link karume.shards.shard_siblings} が名前の形で拾うファイルだけ。

`--out` を渡すと**別**ディレクトリへ書く（入力は読むだけ）— 実適用の前に「同じバイトが出るか」を
確かめる経路。宛先が入力の代表 path へ畳まれる指定は fail loudly（後片付けが入力側の旧 shard を
消し、確かめるつもりの呼び出しが実適用になるため）。

MUST: torch を持ち込まない。ここが扱うのは**バイト列と宣言だけ**で、テンソルを torch へ
起こして書き戻すと dtype の往復（f16 / i4 の器）で沈黙誤値を作る余地が生まれる。読みは
自前の薄いリーダ（規則そのものは {@link karume.verify.assert_reader_layout} が先に張る）、
書きは {@link karume.emit.write_container}（並び順の規約と器の綴りの正本）を通す。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from uuid import uuid4

from karume.emit import ContainerEntry, container_order, write_container
from karume.ir import IR_METADATA_KEY
from karume.shards import (
    SHARD_DATA_CAPACITY,
    Piece,
    assert_co_shard,
    assert_shard_partition,
    pack_shards,
    parse_piece_key,
    resolve_shards,
    shard_path,
    shard_siblings,
)
from karume.verify import assert_reader_layout, parse_ir_graph, verify_shards

#: 生バイトの読み出し単位（sha256 と写しで共有）。数 GB を丸読みしないための唯一の要件で、
#: 値自体は素の I/O 単位（`dist.sha256_file` と同じ流儀）。
_CHUNK_BYTES = 1 << 20

_HEADER_LENGTH_BYTES = 8


class RepackError(ValueError):
    """詰め替えの前提が破れた（入力が読めない / 書いたものが旧と食い違う）。"""


@dataclass(frozen=True)
class _Stored:
    """入力側のテンソル 1 本 — **親としての宣言**と、現物の在処。

    分割テンソルは piece が別々のファイルに散っているので、在処は 1 点ではなく**区間の列**に
    なる（`(ファイル, ファイル先頭からの絶対 offset, バイト長)` を piece の index 順に）。
    丸ごと 1 本のときは要素 1 つの列で、以降の経路（sha256・写し・行範囲の切り出し）は
    区間の列だけを見るので分岐しない。
    """

    entry: ContainerEntry
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
            raise RepackError(f"{path}: ヘッダ長 {header_length} がファイル長 {size} と矛盾する")
        header = json.loads(handle.read(header_length))
    if not isinstance(header, dict):
        raise RepackError(f"{path}: safetensors ヘッダが最上位オブジェクトでない")
    return header, _HEADER_LENGTH_BYTES + header_length


def _read_metadata(path: Path, header: Mapping[str, Any]) -> dict[str, str]:
    """`__metadata__` を文字列 → 文字列のまま取り出す（**逐語**で持ち回す唯一の入口）。"""
    raw = header.get("__metadata__", {})
    if not isinstance(raw, dict) or any(
        not isinstance(key, str) or not isinstance(value, str) for key, value in raw.items()
    ):
        raise RepackError(f"{path}: __metadata__ が文字列 → 文字列のマップでない")
    return dict(raw)


@dataclass(frozen=True)
class _Fragment:
    """読み込み中の piece 1 本（畳む前 — 連番と収容 shard の並びを見るための材料）。"""

    shard: int
    index: int
    count: int
    entry: ContainerEntry
    segment: tuple[Path, int, int]


def _join_pieces(name: str, fragments: Sequence[_Fragment]) -> _Stored:
    """piece 列を親 1 本へ畳む（読み手契約 5 を全部ここで見る）。

    `fragments` は**読む順**（shard 番号順）に並んでいる。畳んだ宣言は親の dtype・全体
    shape（先頭次元 = 各 piece の行数の和）・合計バイト長で、在処は piece の区間列になる。
    """
    shards = [fragment.shard for fragment in fragments]
    repeated = sorted({shard for shard in shards if shards.count(shard) > 1})
    if repeated:
        raise RepackError(
            f"テンソル '{name}': shard{repeated} に同じ親の piece が 2 本ある"
            "（piece は連続する shard に 1 本ずつ）"
        )
    # count >= 2 と index の域は `parse_piece_key` が既に見ている（域外の綴りは piece と解釈
    # されず、名前そのままの 1 本として運ばれ、書き出し後の verify が余剰として落とす）。
    count = fragments[0].count
    if len(fragments) != count:
        raise RepackError(
            f"テンソル '{name}': piece が {len(fragments)} 本で宣言の総数 {count} と合わない"
        )
    head = fragments[0].entry
    rows = 0
    previous: int | None = None
    for position, fragment in enumerate(fragments, start=1):
        entry = fragment.entry
        if fragment.count != count:
            raise RepackError(
                f"テンソル '{name}': piece の総数が {count} と {fragment.count} で食い違っている"
            )
        if fragment.index != position:
            raise RepackError(
                f"テンソル '{name}': shard 順で {position} 本目の piece が index {fragment.index}"
                "（index は shard 順に 1 から増える）"
            )
        if previous is not None and fragment.shard != previous + 1:
            raise RepackError(
                f"テンソル '{name}': piece {fragment.index} が shard[{fragment.shard}]・前の"
                f" piece が shard[{previous}]（piece は連続する shard に 1 本ずつ）"
            )
        if entry.dtype != head.dtype:
            raise RepackError(
                f"テンソル '{name}': piece {fragment.index} の dtype が {entry.dtype}"
                f"（piece 1 は {head.dtype}）"
            )
        if entry.shape[1:] != head.shape[1:]:
            raise RepackError(
                f"テンソル '{name}': piece {fragment.index} の残り次元 {list(entry.shape[1:])} が"
                f" piece 1 の {list(head.shape[1:])} と違う"
            )
        if not entry.shape or entry.shape[0] < 1:
            raise RepackError(f"テンソル '{name}': piece {fragment.index} が 1 行未満")
        if position < count and entry.nbytes % 4:
            raise RepackError(
                f"テンソル '{name}': 末尾でない piece {fragment.index} が {entry.nbytes} バイトで"
                " 4 の倍数でない（読み手が親バッファへオフセット書きできない）"
            )
        rows += entry.shape[0]
        previous = fragment.shard
    return _Stored(
        entry=ContainerEntry(
            name=name,
            dtype=head.dtype,
            shape=(rows, *head.shape[1:]),
            nbytes=sum(fragment.entry.nbytes for fragment in fragments),
        ),
        segments=tuple(fragment.segment for fragment in fragments),
    )


def read_component(paths: Sequence[Path]) -> tuple[dict[str, str], dict[str, _Stored]]:
    """コンポーネントの現物を読み、`(先頭 shard の __metadata__, 名前 → 現物)` を返す。

    受けるのは**旧規則の配布形も含む**列（単一ファイル / fat グラフ shard / 尾部スラック）—
    現行の門（グラフ shard が空・バイト上限）は入力に掛けない（掛けたら移行できない）。
    それでも「先頭だけが `karume_ir` を持つ」「同名テンソルが 2 本無い」は旧規則でも成り立って
    いた不変条件なので、ここで見る（破れているものを詰め替えると、どちらのバイト列を配るかが
    一意に決まらない）。

    分割テンソル（piece キー）は**親 1 本へ畳む**（{@link _join_pieces} が読み手契約 5 を
    検査する）。畳むのは形だけで判定する — 入力の `karume_ir` はまだ parse していないので
    「親名が宣言に在るか」はここでは見ず、畳んだ親が宣言に無ければ書き出し後の
    {@link karume.verify.verify_shards} が余剰として落とす。
    """
    metadata: dict[str, str] = {}
    stored: dict[str, _Stored] = {}
    fragments: dict[str, list[_Fragment]] = {}
    owner: dict[str, Path] = {}
    for index, path in enumerate(paths):
        if not path.is_file():
            raise RepackError(f"詰め替えの入力が無い: {path}")
        assert_reader_layout(path)
        header, data_start = _read_header(path)
        raw = _read_metadata(path, header)
        if index == 0:
            metadata = raw
        elif raw:
            raise RepackError(
                f"{path}: shard[{index}] が __metadata__ を持っている"
                f"（{sorted(raw)} — メタデータを載せるのは先頭 shard だけ）"
            )
        for name, spec in header.items():
            if name == "__metadata__":
                continue
            if name in owner:
                raise RepackError(
                    f"テンソル '{name}' が {owner[name].name} と {path.name} に重複している"
                )
            owner[name] = path
            begin, end = spec["data_offsets"]
            entry = ContainerEntry(
                name=name, dtype=spec["dtype"], shape=tuple(spec["shape"]), nbytes=end - begin
            )
            segment = (path, data_start + begin, end - begin)
            parsed = parse_piece_key(name)
            if parsed is None:
                stored[name] = _Stored(entry=entry, segments=(segment,))
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
            raise RepackError(
                f"テンソル '{parent}' が丸ごとと piece の両方でコンテナに居る"
                "（1 テンソルはどちらか一方 MUST）"
            )
        stored[parent] = _join_pieces(parent, fragments[parent])
    if IR_METADATA_KEY not in metadata:
        raise RepackError(
            f"{paths[0]}: __metadata__.{IR_METADATA_KEY} が無い（Karume の配布形ではない）"
        )
    return metadata, stored


def _range_chunks(source: _Stored, begin: int, end: int) -> Iterator[bytes]:
    """親の**バイト範囲** `[begin, end)` を読み出し単位ずつ流す（丸読みしない）。

    範囲は区間の列（= piece の並び）を跨いでよい — 分割された入力を別の切り目で書き直すのが
    詰め替えの仕事なので、読み手は「親の連続バイト列」だけを見る。
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
                    raise RepackError(
                        f"{path}: テンソル '{source.entry.name}' の宣言"
                        f"（{source.entry.nbytes} バイト）に対しデータ節が途中で尽きた"
                    )
                remaining -= len(chunk)
                yield chunk
        cursor = stop


def _payload_chunks(source: _Stored) -> Iterator[bytes]:
    """1 本ぶん（親の全バイト）を読み出し単位ずつ流す。"""
    return _range_chunks(source, 0, source.entry.nbytes)


def _fingerprints(stored: Mapping[str, _Stored]) -> dict[str, tuple[str, tuple[int, ...], str]]:
    """名前 → (dtype, shape, sha256) — 詰め替えの前後で**一致する MUST** の写像。

    sha256 で見るのは、旧新を同時にメモリへ載せずに全バイトを突き合わせられる唯一の形だから
    （バイト長は digest の元に入っているうえ、宣言側でも別に一致を見る）。
    """
    prints: dict[str, tuple[str, tuple[int, ...], str]] = {}
    for name, source in sorted(stored.items()):
        digest = hashlib.sha256()
        for chunk in _payload_chunks(source):
            digest.update(chunk)
        prints[name] = (source.entry.dtype, source.entry.shape, digest.hexdigest())
    return prints


def _assert_same_bytes(
    before: Mapping[str, tuple[str, tuple[int, ...], str]],
    after: Mapping[str, tuple[str, tuple[int, ...], str]],
) -> None:
    """テンソルの写像が 1 本残らず同一であることを落とす（不変条件 1）。"""
    missing = sorted(set(before) - set(after))
    surplus = sorted(set(after) - set(before))
    if missing or surplus:
        raise RepackError(f"詰め替えでテンソルの集合が変わった（欠け {missing} / 余剰 {surplus}）")
    changed = sorted(name for name in before if before[name] != after[name])
    if changed:
        name = changed[0]
        raise RepackError(
            f"詰め替えでテンソルの中身が変わった（{len(changed)} 本 — 例: '{name}'"
            f" 旧 {before[name]} / 新 {after[name]}）"
        )


def plan_shards(
    stored: Mapping[str, _Stored], graph_text: str, capacity: int
) -> list[tuple[str | Piece, ...]]:
    """現行の規則で shard 群を決める（規則の正本は {@link karume.shards.pack_shards}）。

    並べる順は書き手の規約（{@link karume.emit.container_order}）そのままで、原子対
    （weight ↔ companion scale）は**宣言から**引く — 旧配布形では対が別 shard に居ることも
    ありうるので、現物の並びからは復元できない。`capacity` はデータ節に詰める上限。
    """
    graph = parse_ir_graph(graph_text)
    companions: dict[str, str] = {}
    for initializer in graph.initializers.values():
        scale = initializer.storage.scale
        if scale is not None:
            companions[initializer.tensor] = scale
            companions[scale] = initializer.tensor
    order = [entry.name for entry in container_order(source.entry for source in stored.values())]
    payload_bytes = {name: stored[name].entry.nbytes for name in order}
    shapes = {name: stored[name].entry.shape for name in order}
    groups = pack_shards(order, payload_bytes, shapes, companions, capacity=capacity)
    # 規則（pack_shards）と検査を分けて持つのは `emit._shard_groups` と同じ理由。
    assert_shard_partition(groups, order, shapes)
    assert_co_shard(groups, companions)
    return groups


def _piece_slice(source: _Stored, piece: Piece) -> tuple[_Stored, int, int]:
    """piece の行範囲 → 親の**バイト範囲**（`_range_chunks` へそのまま渡せる形）。

    1 行のバイト長は「親の合計バイト長 ÷ 行数」— 分割の可否を決めた
    {@link karume.shards} と同じ導出で、行が整数バイトであることは向こうが見ている。
    """
    rows = int(source.entry.shape[0])
    row_bytes = source.entry.nbytes // rows
    return source, piece.begin * row_bytes, piece.end * row_bytes


def _shard_reader(
    sources: Mapping[str, tuple[_Stored, int, int]],
) -> Callable[[ContainerEntry], Iterator[bytes]]:
    """entry → 生バイトの列（{@link karume.emit.write_container} へ渡す読み口）。"""

    def payload(entry: ContainerEntry) -> Iterator[bytes]:
        return _range_chunks(*sources[entry.name])

    return payload


def _write_shards(
    staged: Path,
    groups: Sequence[Sequence[str | Piece]],
    stored: Mapping[str, _Stored],
    metadata: Mapping[str, str],
) -> list[Path]:
    """shard 群を一時 path の連番として書く（先頭がグラフ shard）。

    分割テンソルの断片は「親のバイト範囲を写すだけ」— 入力の切り目と出力の切り目が違っても、
    流れるバイト列は親の連続部分列そのものになる。
    """
    written: list[Path] = []
    total = len(groups)
    for index, group in enumerate(groups, start=1):
        target = shard_path(staged, index, total)
        entries: list[ContainerEntry] = []
        sources: dict[str, tuple[_Stored, int, int]] = {}
        for member in group:
            if isinstance(member, Piece):
                source = stored[member.name]
                _, begin, end = _piece_slice(source, member)
                entries.append(
                    ContainerEntry(
                        name=member.key,
                        dtype=source.entry.dtype,
                        shape=(member.end - member.begin, *source.entry.shape[1:]),
                        nbytes=end - begin,
                    )
                )
                sources[member.key] = (source, begin, end)
            else:
                source = stored[member]
                entries.append(source.entry)
                sources[member] = (source, 0, source.entry.nbytes)
        write_container(
            target,
            container_order(entries),
            # MUST: `karume_ir` を載せるのは先頭 shard だけ（ADR 0070 決定 3）。
            metadata if index == 1 else {},
            _shard_reader(sources),
        )
        written.append(target)
    return written


def repack_component(
    path: str | Path,
    out_dir: str | Path | None = None,
    *,
    _shard_capacity: int | None = None,
) -> list[Path]:
    """コンポーネントの**代表 path** を現行の shard 列へ詰め替え、据えた path を順に返す。

    `out_dir` を渡すとそこへ同名で書く（入力は読むだけ）。渡さなければ**インプレース**で、
    旧ファイル群は据え替えの後に消える。

    MUST: `out_dir` の宛先が入力の代表 path と同じになる指定は `RepackError`。後片付けは
    宛先基準で走るので、そのまま進むと「入力は読むだけ」という `--out` の契約が破れ、実適用の
    前に確かめるつもりの呼び出しが黙ってインプレースの詰め替えになる。

    MUST: 据える名前は**常に** `-NNNNN-of-NNNNN` の連番（`karume.emit.write_model` と同じ規律
    — 単一ファイル配布形は廃止・ADR 0081）。テンソル 0 本のコンポーネントは 1 本の shard に
    なるが、そこで連番を落とすと同じコンポーネントを書き手が焼き直したときとファイル名が
    食い違い、両方が残った瞬間に `resolve_shards` が「単一ファイルと連番の同居」で止まる。

    `_shard_capacity` は**テストからのみ触る**データ節容量の差し込み（合成の小コンポーネントで
    分割を起こすため）— 公開ノブではない（配布形の不変条件・
    {@link karume.shards.SHARD_DATA_CAPACITY}）。

    MUST: 検証門は書いた**一時ファイル**に対して通し、通ってはじめて据える。落ちた回は一時
    ファイルだけを捨て、手元の配布形は 1 バイトも変えない（`pipeline.publish_model` と同じ
    規律 — 中断した回に「旧と新が混ざったコンポーネント」を残さない）。
    """
    source_paths = resolve_shards(Path(path))
    metadata, stored = read_component(source_paths)
    capacity = SHARD_DATA_CAPACITY if _shard_capacity is None else _shard_capacity
    groups = plan_shards(stored, metadata[IR_METADATA_KEY], capacity)

    final = Path(path) if out_dir is None else Path(out_dir) / Path(path).name
    if out_dir is not None and final.resolve() == Path(path).resolve():
        raise RepackError(
            f"--out の宛先 {final} が入力の代表 path と同じ"
            " — `--out` は別ディレクトリへ書く経路（入力は読むだけ）で、"
            "インプレースで詰め替えるなら `--out` を外す"
        )
    # 一意 suffix — 同じ final を狙う別プロセスの一時ファイルと衝突させない。
    staged = final.with_name(f"{final.name}.{uuid4().hex}.partial")
    before = _fingerprints(stored)
    try:
        written = _write_shards(staged, groups, stored, metadata)
        verify_shards(written)
        after_metadata, after_stored = read_component(written)
        if after_metadata != metadata:
            # 逐語同一 MUST（不変条件 2 / 3）。ここが動くと、意味が同じでもバイトが変わる。
            raise RepackError(
                f"{final}: __metadata__ が詰め替えで変わった"
                f"（旧 {sorted(metadata)} / 新 {sorted(after_metadata)}）"
            )
        _assert_same_bytes(before, _fingerprints(after_stored))
        total = len(written)
        published = [shard_path(final, index, total) for index in range(1, total + 1)]
        for staged_shard, target in zip(written, published, strict=True):
            os.replace(staged_shard, target)
    except BaseException:
        # 書き出しが途中で落ちた回は返り値が無いので、一時 path の**名前の形**から拾う。
        for leftover in shard_siblings(staged):
            leftover.unlink(missing_ok=True)
        raise
    for stale in shard_siblings(final):
        if stale not in published:
            stale.unlink()
    return published


# ---- CLI ------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="既存の配布形を現行の shard 仕様へ詰め替える（テンソルのバイトは変えない）"
    )
    parser.add_argument(
        "models",
        type=Path,
        nargs="+",
        help="詰め替えるコンポーネントの**代表 path**（分割済みなら連番へ解決する）",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="別ディレクトリへ書く（既定はインプレース — 旧ファイル群は据え替え後に消える）",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """指定されたコンポーネントを 1 本ずつ詰め替える。

    MUST: 落ちたところで止める（残りを詰め替えて最後にまとめない）— 例外は破れた不変条件まで
    綴ってあるので、そのまま送出するのが最も情報量が多い。
    """
    args = build_parser().parse_args(argv)
    claimed: dict[Path, Path] = {}
    for path in args.models:
        # `--out` に同名のコンポーネントを 2 本流すと、後の 1 本が先の 1 本を上書きする
        # （インプレースでは起こらない衝突なので、宛先が 1 箇所に畳まれる形だけを見る）。
        final = path if args.out is None else args.out / path.name
        if final in claimed:
            raise RepackError(f"{claimed[final]} と {path} の宛先が同じ: {final}")
        claimed[final] = path
    for path in args.models:
        sources = resolve_shards(path)
        published = repack_component(path, args.out)
        print(f"{path}: shards {len(sources)} → {len(published)} ({published[0].name} …)")


if __name__ == "__main__":
    main()
