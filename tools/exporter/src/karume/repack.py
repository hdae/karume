"""既存の配布形を shard 仕様 v2 へ**詰め替える**（ADR 0081 の移行経路）。

    karume repack ../../outputs/series/vowel-detector-crnn-epoch3/model.safetensors

旧規則で焼いたコンポーネント（fat グラフ shard・単一ファイル・尾部スラック）を、v2 の規則
（グラフ shard = `karume_ir` だけ / 全 shard ≤ {@link karume.shards.SHARD_BYTE_LIMIT} /
最小本数→均し / 常時分割）で並べ直す。**再 export ではない** — GPTQ 校正はリグと実行のたびに
丸め解が動くので（docs/research/ 2026-08-24）、出荷済みの系列を焼き直すと配るバイトが変わる。
ここが動かすのは**容器の詰め方だけ**で、テンソルの生バイトと `karume_ir` は 1 バイトも変えない。

## 不変条件（MUST — 全部が書き出し後の**読み戻し**で機械検証される）

1. **テンソルの生バイトが旧と同一**: 名前 → (dtype, shape, バイト長, sha256) の写像が旧新で
   一致する。詰め替えは所属 shard と並び順しか動かさないので、1 本でも中身が動いたら
   それは詰め替えではない。
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

`--out` を渡すと別ディレクトリへ書く（入力は読むだけ）— 実適用の前に「同じバイトが出るか」を
確かめる経路。

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
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from karume.emit import ContainerEntry, container_order, write_container
from karume.ir import IR_METADATA_KEY
from karume.shards import (
    SHARD_BYTE_LIMIT,
    assert_co_shard,
    assert_shard_partition,
    pack_shards,
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
    """旧配布形のテンソル 1 本 — **宣言**と、現物の在処（ファイルと絶対 offset）。"""

    entry: ContainerEntry
    path: Path
    #: ファイル先頭からの絶対 offset（データ節の開始 + ヘッダの相対 offset）。
    begin: int


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


def read_component(paths: Sequence[Path]) -> tuple[dict[str, str], dict[str, _Stored]]:
    """コンポーネントの現物を読み、`(先頭 shard の __metadata__, 名前 → 現物)` を返す。

    受けるのは**旧規則の配布形も含む**列（単一ファイル / fat グラフ shard / 尾部スラック）—
    v2 の門（グラフ shard が空・バイト上限）は入力に掛けない（掛けたら移行できない）。
    それでも「先頭だけが `karume_ir` を持つ」「同名テンソルが 2 本無い」は旧規則でも成り立って
    いた不変条件なので、ここで見る（破れているものを詰め替えると、どちらのバイト列を配るかが
    一意に決まらない）。
    """
    metadata: dict[str, str] = {}
    stored: dict[str, _Stored] = {}
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
            if name in stored:
                raise RepackError(
                    f"テンソル '{name}' が {stored[name].path.name} と {path.name} に重複している"
                )
            begin, end = spec["data_offsets"]
            stored[name] = _Stored(
                entry=ContainerEntry(
                    name=name,
                    dtype=spec["dtype"],
                    shape=tuple(spec["shape"]),
                    nbytes=end - begin,
                ),
                path=path,
                begin=data_start + begin,
            )
    if IR_METADATA_KEY not in metadata:
        raise RepackError(
            f"{paths[0]}: __metadata__.{IR_METADATA_KEY} が無い（Karume の配布形ではない）"
        )
    return metadata, stored


def _payload_chunks(source: _Stored) -> Iterator[bytes]:
    """1 本ぶんの生バイトを読み出し単位ずつ流す（丸読みしない）。"""
    with source.path.open("rb") as handle:
        handle.seek(source.begin)
        remaining = source.entry.nbytes
        while remaining:
            chunk = handle.read(min(_CHUNK_BYTES, remaining))
            if not chunk:
                raise RepackError(
                    f"{source.path}: テンソル '{source.entry.name}' の宣言"
                    f"（{source.entry.nbytes} バイト）に対しデータ節が途中で尽きた"
                )
            remaining -= len(chunk)
            yield chunk


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
    stored: Mapping[str, _Stored], graph_text: str, limit: int
) -> list[tuple[str, ...]]:
    """v2 の規則で shard 群を決める（規則の正本は {@link karume.shards.pack_shards}）。

    並べる順は書き手の規約（{@link karume.emit.container_order}）そのままで、原子対
    （weight ↔ companion scale）は**宣言から**引く — 旧配布形では対が別 shard に居ることも
    ありうるので、現物の並びからは復元できない。
    """
    graph = parse_ir_graph(graph_text)
    companions: dict[str, str] = {}
    for initializer in graph.initializers.values():
        scale = initializer.storage.scale
        if scale is not None:
            companions[initializer.tensor] = scale
            companions[scale] = initializer.tensor
    order = [entry.name for entry in container_order(source.entry for source in stored.values())]
    groups = pack_shards(
        order, {name: stored[name].entry.nbytes for name in order}, companions, limit
    )
    # 規則（pack_shards）と検査を分けて持つのは `emit._shard_groups` と同じ理由。
    assert_shard_partition(groups, order)
    assert_co_shard(groups, companions)
    return groups


def _write_shards(
    staged: Path,
    groups: Sequence[Sequence[str]],
    stored: Mapping[str, _Stored],
    metadata: Mapping[str, str],
) -> list[Path]:
    """shard 群を一時 path の連番として書く（先頭がグラフ shard）。"""
    written: list[Path] = []
    total = len(groups)
    for index, group in enumerate(groups, start=1):
        target = shard_path(staged, index, total)
        write_container(
            target,
            container_order(stored[name].entry for name in group),
            # MUST: `karume_ir` を載せるのは先頭 shard だけ（ADR 0070 決定 3）。
            metadata if index == 1 else {},
            lambda entry: _payload_chunks(stored[entry.name]),
        )
        written.append(target)
    return written


def repack_component(
    path: str | Path,
    out_dir: str | Path | None = None,
    *,
    _shard_byte_limit: int | None = None,
) -> list[Path]:
    """コンポーネントの**代表 path** を v2 の shard 列へ詰め替え、据えた path を順に返す。

    `out_dir` を渡すとそこへ同名で書く（入力は読むだけ）。渡さなければ**インプレース**で、
    旧ファイル群は据え替えの後に消える。

    `_shard_byte_limit` は**テストからのみ触る**上限の差し込み（合成の小コンポーネントで分割を
    起こすため）— 公開ノブではない（配布形の不変条件・{@link karume.shards.SHARD_BYTE_LIMIT}）。

    MUST: 検証門は書いた**一時ファイル**に対して通し、通ってはじめて据える。落ちた回は一時
    ファイルだけを捨て、手元の配布形は 1 バイトも変えない（`pipeline.publish_model` と同じ
    規律 — 中断した回に「旧と新が混ざったコンポーネント」を残さない）。
    """
    source_paths = resolve_shards(Path(path))
    metadata, stored = read_component(source_paths)
    limit = SHARD_BYTE_LIMIT if _shard_byte_limit is None else _shard_byte_limit
    groups = plan_shards(stored, metadata[IR_METADATA_KEY], limit)

    final = Path(path) if out_dir is None else Path(out_dir) / Path(path).name
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
        published = [
            final if total == 1 else shard_path(final, index, total)
            for index in range(1, total + 1)
        ]
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
        description="既存の配布形を shard 仕様 v2 へ詰め替える（テンソルのバイトは変えない）"
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
