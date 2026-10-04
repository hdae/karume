"""書き直した Wan の成果物を既存の系列と比べる CLI（ADR 0121 段 4 — 2.1 の系列の不変の確かめ）。

recipe を変えた後、2.1 の成果物（VAE の chunk グラフ・フィクスチャなど）を scratch へ書き直し、
既存の系列と `--require` に挙げた項目だけを比べる。門なので、設計で避けるのは「違うのに一致と
言う」こと（偽の一致）:

- 比べ方は 2 通り。
  - `*.safetensors`: テンソル名の集合・テンソルごとの dtype / 形 / 生バイト（ビット一致 — NaN の
    payload も -0.0 も畳まない）と、`__metadata__` の dict の等値（キー順は見ない）。safetensors
    0.8.0 の `__metadata__` はキー順がプロセスごとに変わるので、ファイルのバイトでは比べられない
    （ADR 0118 追記「再生成のバイト一致」）。ヘッダは自前で読み（{@link read_header}）、データ域が
    テンソルで隙間なく埋まっていることも確かめる — テンソルの外のバイトを黙って見落とさない。
    比較は名前で引いた中身の比較で、データ域でのテンソルの並び（data_offsets）とヘッダの空白の
    詰め物はわざと見ない（読み出す中身が同じなら同じ成果物）。`"__metadata__": null` は無いのと
    同じに扱う（safetensors 本体も null を「無い」として読む）。
  - それ以外（krm の part など）: サイズと sha256。
- ディレクトリの項目は、両側の相対パスの集合（再帰・ディレクトリも含む）が同じことを見てから
  各ファイルを比べる。どちらかの側に無い項目は食い違い（skip しない）。比べるファイルが 1 つも
  無いディレクトリの項目も食い違い — 何も比べていない一致は、両側の用意が壊れている印。
- シンボリックリンクは辿らない: 項目の中（項目そのもの・途中の成分・ディレクトリの中身）に
  あれば食い違いとして名指す。系列の書き手はリンクを作らないので、あるなら想定外 — 辿って
  比べると、両側が同じ先を指すだけで中身が何であれ一致になる。根（`--written` / `--existing`）は
  利用者が明示したパスなので辿ってよい。通常ファイルでもディレクトリでもない項目（FIFO 等）も
  同じく食い違い。種類は lstat で見てからパスで開くので、その間にリンクへ差し替える並行の
  書き手は想定しない（比べている最中の根に書き手は走らせない）。
- 両側が同じ実体（同じ inode — 同じ根を 2 回渡した・ハードリンク）なら食い違い。自分自身との
  比較は必ず一致するので門にならない。ただしコピーは別の実体なので一致する — 門が意味を持つのは、
  書いた側が空の scratch へ書き手を走らせた結果のときだけ（`wan/README.md` の手順）。

使い方（`tools/export-recipes` で）:

    uv run --group wan --inexact python -m wan.series_check --written <dir> --existing <dir> \\
      --require vae_decoder_first --require vae_chunks.band.safetensors

標準出力に JSON を 1 つ書く（比べた項目・ファイルごとの結果・食い違いと理由）。終了コードは
食い違いが無いときだけ 0、食い違いがあれば 1、引数の誤り（`--require` 無し・根の外か根そのものを
指す名前・NUL を含む名前・重複・別の `--require` の中に入れ子の名前・根がディレクトリでない）は 2、
読めないファイル（権限など — OSError）は 3（標準出力は空・理由は標準エラー）。

export_dit の `--check`（`series_mismatches` — 部品ディレクトリ直下のバイトだけ）とは別の口
（ADR 0121 段 4 の設計 — export_dit 側は変えない）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
import struct
import sys
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO

import numpy as np

#: 項目の種類（lstat の結果 — リンクは辿らない）。
FILE = "file"
DIR = "dir"
SYMLINK = "symlink"
OTHER = "other"
MISSING = "missing"

SIDE_LABELS = {"written": "書いた側", "existing": "既存"}

SAFETENSORS_SUFFIX = ".safetensors"

#: safetensors のヘッダ長の上限（safetensors 本体と同じ 100 MB — 壊れた長さで巨大な読み込みを
#: しない）。
MAX_HEADER_BYTES = 100_000_000

#: dtype → 1 要素のバイト数（safetensors の dtype 名）。ここに無い dtype（サブバイトの F4 等）は
#: 読めないものとして fail loudly — 要素数の数え方を推測しない。
ITEM_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E5M2": 1,
    "F8_E4M3": 1,
    "F8_E8M0": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}

#: テンソルのバイトを読む単位（16 MiB — 全 dtype の要素長の倍数。数百 MB のファイルでも
#: 同時に持つのは 2 × この大きさだけ）。
CHUNK_BYTES = 1 << 24


class SafetensorsFormatError(ValueError):
    """safetensors として読めない（ヘッダ・dtype・データ域の配置が取り決めから外れた）。"""


@dataclass(frozen=True)
class TensorEntry:
    """ヘッダのテンソル 1 本（`begin` / `end` はデータ域の先頭からのバイト位置）。"""

    dtype: str
    shape: tuple[int, ...]
    begin: int
    end: int


@dataclass(frozen=True)
class SafetensorsHeader:
    """safetensors のヘッダ（`data_start` はデータ域のファイル内の位置）。"""

    data_start: int
    tensors: dict[str, TensorEntry]
    #: `__metadata__` が無い（または null の）ファイルは None（空の dict とは区別する）。
    metadata: dict[str, str] | None


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    counts = Counter(key for key, _ in pairs)
    duplicated = sorted(key for key, count in counts.items() if count > 1)
    if duplicated:
        raise SafetensorsFormatError(f"ヘッダに重複したキー {duplicated}")
    return dict(pairs)


def _tensor_entry(name: str, raw: Any) -> TensorEntry:
    if not isinstance(raw, dict) or set(raw) != {"dtype", "shape", "data_offsets"}:
        raise SafetensorsFormatError(
            f"テンソル '{name}' の項目が dtype / shape / data_offsets でない"
        )
    dtype, shape, offsets = raw["dtype"], raw["shape"], raw["data_offsets"]
    if dtype not in ITEM_BYTES:
        raise SafetensorsFormatError(f"テンソル '{name}' の dtype {dtype!r} を知らない")
    if not isinstance(shape, list) or not all(type(dim) is int and dim >= 0 for dim in shape):
        raise SafetensorsFormatError(f"テンソル '{name}' の形 {shape!r} が非負整数の列でない")
    if (
        not isinstance(offsets, list)
        or len(offsets) != 2
        or not all(type(offset) is int for offset in offsets)
        or not 0 <= offsets[0] <= offsets[1]
    ):
        raise SafetensorsFormatError(f"テンソル '{name}' の data_offsets {offsets!r} が壊れている")
    begin, end = offsets
    expected = math.prod(shape) * ITEM_BYTES[dtype]
    if end - begin != expected:
        raise SafetensorsFormatError(
            f"テンソル '{name}' のバイト数 {end - begin} が"
            f"形 {shape} × {dtype} の {expected} と違う"
        )
    return TensorEntry(dtype, tuple(shape), begin, end)


def read_header(path: Path) -> SafetensorsHeader:
    """safetensors のヘッダを読んで取り決めを確かめる（テンソルのバイトは読まない）。

    データ域がテンソルで先頭から隙間なく・重なりなく埋まり、ファイルの末尾で終わることを要求する
    （safetensors 本体の検査と同じ向き）。テンソルの外にバイトがあると、テンソルとメタだけの
    比較はそれを見落とすので fail loudly。
    """
    size = path.stat().st_size
    with path.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise SafetensorsFormatError(
                f"ファイルが {len(prefix)} バイトしかない（ヘッダ長が無い）"
            )
        (length,) = struct.unpack("<Q", prefix)
        if length > MAX_HEADER_BYTES or 8 + length > size:
            raise SafetensorsFormatError(f"ヘッダ長 {length} がファイル {size} バイトに収まらない")
        raw = stream.read(length)
    try:
        header = json.loads(raw.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SafetensorsFormatError(f"ヘッダが UTF-8 の JSON でない: {error}") from error
    if not isinstance(header, dict):
        raise SafetensorsFormatError("ヘッダが JSON の object でない")
    metadata = header.pop("__metadata__", None)
    if metadata is not None and not (
        isinstance(metadata, dict) and all(isinstance(value, str) for value in metadata.values())
    ):
        raise SafetensorsFormatError("__metadata__ が文字列 → 文字列の object でない")
    tensors = {name: _tensor_entry(name, raw_entry) for name, raw_entry in header.items()}
    position = 0
    for name, entry in sorted(tensors.items(), key=lambda item: (item[1].begin, item[1].end)):
        if entry.begin != position:
            raise SafetensorsFormatError(
                f"テンソル '{name}' の位置 {entry.begin} の前が {position} で終わっている"
                "（データ域に隙間か重なり）"
            )
        position = entry.end
    data_bytes = size - 8 - length
    if position != data_bytes:
        raise SafetensorsFormatError(
            f"テンソルがデータ域 {data_bytes} バイトのうち {position} バイトまでしか覆わない"
        )
    return SafetensorsHeader(8 + length, tensors, metadata)


def _read_exact(stream: BinaryIO, length: int) -> bytes:
    data = stream.read(length)
    if len(data) != length:
        raise SafetensorsFormatError(f"{length} バイトを読むはずが {len(data)} バイトで終わった")
    return data


def _diff_tensor_bytes(
    written: BinaryIO, existing: BinaryIO, length: int, item_bytes: int
) -> tuple[int | None, int]:
    """位置を合わせた 2 本のストリームの `length` バイトを比べる（最初に違うバイトの位置・
    違う要素の数）。全部同じなら `(None, 0)`。"""
    first: int | None = None
    differing = 0
    for position in range(0, length, CHUNK_BYTES):
        size = min(CHUNK_BYTES, length - position)
        left = _read_exact(written, size)
        right = _read_exact(existing, size)
        if left == right:
            continue
        unequal = np.frombuffer(left, dtype=np.uint8) != np.frombuffer(right, dtype=np.uint8)
        if first is None:
            first = position + int(np.flatnonzero(unequal)[0])
        differing += int(unequal.reshape(-1, item_bytes).any(axis=1).sum())
    return first, differing


def _metadata_mismatches(
    path: str, written: dict[str, str] | None, existing: dict[str, str] | None
) -> list[dict[str, Any]]:
    if written is None or existing is None:
        if written is existing:
            return []
        side = "written" if written is None else "existing"
        return [{"path": path, "reason": f"__metadata__ が{SIDE_LABELS[side]}に無い"}]
    mismatches: list[dict[str, Any]] = []
    for key in sorted(written.keys() | existing.keys()):
        if key not in existing:
            mismatches.append(
                {
                    "path": path,
                    "reason": "メタのキーが既存に無い",
                    "key": key,
                    "written": written[key],
                }
            )
        elif key not in written:
            mismatches.append(
                {
                    "path": path,
                    "reason": "メタのキーが書いた側に無い",
                    "key": key,
                    "existing": existing[key],
                }
            )
        elif written[key] != existing[key]:
            mismatches.append(
                {
                    "path": path,
                    "reason": "メタの値が違う",
                    "key": key,
                    "written": written[key],
                    "existing": existing[key],
                }
            )
    return mismatches


def compare_safetensors(
    written: Path, existing: Path, path: str
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """safetensors 2 本をテンソル（名前・dtype・形・生バイト）とメタ（dict の等値）で比べる。"""
    headers: dict[str, SafetensorsHeader] = {}
    mismatches: list[dict[str, Any]] = []
    for side, file in (("written", written), ("existing", existing)):
        try:
            headers[side] = read_header(file)
        except SafetensorsFormatError as error:
            mismatches.append(
                {
                    "path": path,
                    "reason": f"{SIDE_LABELS[side]}が safetensors として読めない: {error}",
                }
            )
    result: dict[str, Any] = {"path": path, "compare": "safetensors"}
    if mismatches:
        return result | {"match": False}, mismatches
    left, right = headers["written"], headers["existing"]
    for name in sorted(left.tensors.keys() - right.tensors.keys()):
        mismatches.append({"path": path, "reason": "テンソルが既存に無い", "tensor": name})
    for name in sorted(right.tensors.keys() - left.tensors.keys()):
        mismatches.append({"path": path, "reason": "テンソルが書いた側に無い", "tensor": name})
    common = sorted(left.tensors.keys() & right.tensors.keys())
    with written.open("rb") as written_stream, existing.open("rb") as existing_stream:
        for name in common:
            mine, theirs = left.tensors[name], right.tensors[name]
            if mine.dtype != theirs.dtype:
                mismatches.append(
                    {
                        "path": path,
                        "reason": "dtype が違う",
                        "tensor": name,
                        "written": mine.dtype,
                        "existing": theirs.dtype,
                    }
                )
                continue
            if mine.shape != theirs.shape:
                mismatches.append(
                    {
                        "path": path,
                        "reason": "形が違う",
                        "tensor": name,
                        "written": list(mine.shape),
                        "existing": list(theirs.shape),
                    }
                )
                continue
            written_stream.seek(left.data_start + mine.begin)
            existing_stream.seek(right.data_start + theirs.begin)
            item_bytes = ITEM_BYTES[mine.dtype]
            first, differing = _diff_tensor_bytes(
                written_stream, existing_stream, mine.end - mine.begin, item_bytes
            )
            if first is not None:
                mismatches.append(
                    {
                        "path": path,
                        "reason": "値のビットが違う",
                        "tensor": name,
                        "dtype": mine.dtype,
                        "shape": list(mine.shape),
                        "first_differing_byte_in_tensor": first,
                        "first_differing_element": first // item_bytes,
                        "differing_elements": differing,
                        "elements": math.prod(mine.shape),
                    }
                )
    mismatches += _metadata_mismatches(path, left.metadata, right.metadata)
    result |= {
        "match": not mismatches,
        "tensors": len(common),
        "tensor_bytes": sum(entry.end - entry.begin for entry in left.tensors.values()),
    }
    return result, mismatches


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def compare_bytes(
    written: Path, existing: Path, path: str
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """safetensors 以外のファイル 2 本をサイズと sha256 で比べる。"""
    sides = {}
    for side, file in (("written", written), ("existing", existing)):
        sides[side] = {"size": file.stat().st_size, "sha256": _sha256(file)}
    result: dict[str, Any] = {"path": path, "compare": "sha256", **sides}
    if sides["written"] == sides["existing"]:
        return result | {"match": True}, []
    reason = (
        "サイズが違う" if sides["written"]["size"] != sides["existing"]["size"] else "sha256 が違う"
    )
    return result | {"match": False}, [{"path": path, "reason": reason, **sides}]


def _kind(mode: int) -> str:
    if stat.S_ISLNK(mode):
        return SYMLINK
    if stat.S_ISDIR(mode):
        return DIR
    if stat.S_ISREG(mode):
        return FILE
    return OTHER


def _lookup(root: Path, name: PurePosixPath) -> tuple[str, PurePosixPath]:
    """`root / name` の種類を、途中の成分も含めてリンクを辿らずに引く（種類・止まった位置）。

    途中の成分がリンク・FIFO 等ならその成分で止めて種類を返す。途中が通常ファイルなら項目は
    無い。
    """
    at = PurePosixPath()
    for index, part in enumerate(name.parts):
        at /= part
        try:
            kind = _kind((root / at).lstat().st_mode)
        except (FileNotFoundError, NotADirectoryError):
            return MISSING, at
        last = index == len(name.parts) - 1
        if not last and kind != DIR:
            return (MISSING if kind == FILE else kind), at
    return kind, at


def _listing(directory: Path) -> dict[str, str]:
    """ディレクトリの中身（相対パス → 種類・再帰・リンクは辿らずリンクとして載せる）。"""
    entries: dict[str, str] = {}
    pending = [PurePosixPath()]
    while pending:
        relative = pending.pop()
        with os.scandir(directory / relative) as scan:
            for entry in scan:
                child = relative / entry.name
                kind = _kind(entry.stat(follow_symlinks=False).st_mode)
                entries[str(child)] = kind
                if kind == DIR:
                    pending.append(child)
    return entries


def _unexpected_kind(path: str, side: str, kind: str) -> dict[str, Any]:
    label = SIDE_LABELS[side]
    if kind == MISSING:
        return {"path": path, "reason": f"{label}に無い"}
    if kind == SYMLINK:
        return {"path": path, "reason": f"{label}がシンボリックリンク（辿らない）"}
    return {"path": path, "reason": f"{label}が通常ファイルでもディレクトリでもない"}


def _same_inode(written: Path, existing: Path) -> bool:
    left, right = written.lstat(), existing.lstat()
    return (left.st_dev, left.st_ino) == (right.st_dev, right.st_ino)


def compare_file(
    written: Path, existing: Path, path: str
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """通常ファイル 2 本を比べる（`*.safetensors` はテンソルとメタ・それ以外は sha256）。"""
    if _same_inode(written, existing):
        return {"path": path, "compare": "none", "match": False}, [
            {"path": path, "reason": "両側が同じ実体（同じ inode — 自分自身との比較になる）"}
        ]
    if path.endswith(SAFETENSORS_SUFFIX):
        return compare_safetensors(written, existing, path)
    return compare_bytes(written, existing, path)


def _compare_tree(
    written: Path, existing: Path, name: PurePosixPath
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if _same_inode(written, existing):
        return [], [
            {"path": str(name), "reason": "両側が同じ実体（同じ inode — 自分自身との比較になる）"}
        ]
    left, right = _listing(written), _listing(existing)
    files: list[dict[str, Any]] = []
    mismatches: list[dict[str, Any]] = []
    for relative in sorted(left.keys() | right.keys()):
        path = str(name / relative)
        kinds = {"written": left.get(relative, MISSING), "existing": right.get(relative, MISSING)}
        unexpected = [
            _unexpected_kind(path, side, kind)
            for side, kind in kinds.items()
            if kind not in (FILE, DIR)
        ]
        if unexpected:
            mismatches += unexpected
        elif kinds["written"] != kinds["existing"]:
            mismatches.append(
                {
                    "path": path,
                    "reason": "種類が違う",
                    "written": kinds["written"],
                    "existing": kinds["existing"],
                }
            )
        elif kinds["written"] == FILE:
            result, found = compare_file(written / relative, existing / relative, path)
            files.append(result)
            mismatches += found
    return files, mismatches


def compare(written: Path, existing: Path, required: Sequence[PurePosixPath]) -> dict[str, Any]:
    """`required` の項目だけを比べた要約（`match` が True なら食い違い 0）。"""
    items: list[dict[str, Any]] = []
    files: list[dict[str, Any]] = []
    mismatches: list[dict[str, Any]] = []
    for name in required:
        found: list[dict[str, Any]] = []
        kinds = {}
        for side, root in (("written", written), ("existing", existing)):
            kind, at = _lookup(root, name)
            kinds[side] = kind
            if kind not in (FILE, DIR):
                found.append(_unexpected_kind(str(name if kind == MISSING else at), side, kind))
        if not found and kinds["written"] != kinds["existing"]:
            found.append(
                {
                    "path": str(name),
                    "reason": "種類が違う",
                    "written": kinds["written"],
                    "existing": kinds["existing"],
                }
            )
        compared: list[dict[str, Any]] = []
        if not found and kinds["written"] == FILE:
            result, found = compare_file(written / name, existing / name, str(name))
            compared = [result]
        elif not found:
            compared, found = _compare_tree(written / name, existing / name, name)
            if not compared and not found:
                found = [{"path": str(name), "reason": "必須ディレクトリに比べるファイルが無い"}]
        files += compared
        items.append({"name": str(name), **kinds, "files": len(compared), "match": not found})
        mismatches += found
    return {
        "written": str(written),
        "existing": str(existing),
        "match": not mismatches,
        "items": items,
        "files": files,
        "mismatches": mismatches,
    }


def require_name(text: str) -> PurePosixPath:
    """`--require` の値（根の中の相対パス — 絶対パス・`..`・根そのもの・NUL は拒む）。"""
    name = PurePosixPath(text)
    if name.is_absolute() or ".." in name.parts or not name.parts or "\0" in text:
        raise argparse.ArgumentTypeError(
            f"{text!r} は根の中の相対パスでない（絶対パス・'..'・根そのもの・NUL は比べない）"
        )
    return name


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--written", type=Path, required=True, help="書き直した側の根")
    parser.add_argument("--existing", type=Path, required=True, help="既存の系列の根")
    parser.add_argument(
        "--require",
        type=require_name,
        action="append",
        required=True,
        help="比べる項目（根からの相対パス・ファイルかディレクトリ — 繰り返し可）",
    )
    args = parser.parse_args(argv)
    for flag, root in (("--written", args.written), ("--existing", args.existing)):
        if not root.is_dir():
            parser.error(f"{flag} {root} がディレクトリでない")
    counts = Counter(args.require)
    duplicated = sorted(str(name) for name, count in counts.items() if count > 1)
    if duplicated:
        parser.error(f"--require が重複している: {duplicated}")
    # 入れ子の項目は同じファイルを 2 回比べて数を水増しするだけなので、引数の誤りとして拒む。
    nested = sorted(
        f"{inner} は {outer} の中" for inner in counts for outer in counts if outer in inner.parents
    )
    if nested:
        parser.error(f"--require が入れ子になっている: {nested}")
    try:
        summary = compare(args.written, args.existing, args.require)
    except OSError as error:
        print(f"series_check: ファイルが読めない（比較は未完了）: {error}", file=sys.stderr)
        return 3
    print(json.dumps(summary, indent=1, ensure_ascii=False))
    return 0 if summary["match"] else 1


if __name__ == "__main__":
    sys.exit(main())
