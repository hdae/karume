"""PLE（per-layer embedding）を**容器の資産**へ組む純関数（ADR 0085 / 0109 決定 4）。

PLE は語彙ぶんの行（`values` と `scales`）を持つ表で、重みではない — ホスト側が token ごとに
区間読みして GPU へ送る（ADR 0085）。sidecar（`ple.json` + `ple-NNNNN.safetensors`）の席は
コンテナの `assets` へ移った:

- `ple.values.<k>` / `ple.scales.<k>`（役割 `ple-values` / `ple-scales`）— **行の倍数**で
  block 上限以下に切った block 列。区間読みを要するので **1 block = 1 part**
  （container-v1 §4.2 の「専用 part に単独」）。
- `ple_index`（役割 `ple-index`・schema 3）— 行バイト数と block ごとの token 区間だけを持つ
  索引。読み手は token → (block, 行 offset) の翻訳をここからする。

MUST: 資産の組み立ては**この 1 本**を通る（移行 CLI も recipe も）。2 経路で綴ると、
「旧 sidecar から移した容器」と「recipe が直接書いた容器」で block の切り目が黙ってずれ、
同じ重みなのにバイトが違う配布形が出る。

MUST: 返した並びが**そのまま物理配置の順**になる（token 順 — 走査型の取得元で隣り合う token が
別 part へ散らない）。descriptor の `assets` の綴りは正準直列化が code point 順にするので、
並びの意味はファイル上の配置にだけ効く。
"""

from __future__ import annotations

from collections.abc import Buffer, Callable, Mapping
from typing import Any

from karume.container import BLOCK_MAX_BYTES, BLOCK_TAIL_ALIGN, AssetInput, canonical_json

#: 旧 manifest の assets に居た PLE 索引の名前。新しい容器でも同じ名前の資産になる。
PLE_INDEX_ASSET = "ple_index"

#: PLE 索引の版（block 列を指す形 — 旧 sidecar の schema 1 / 2 を置き換える）。
PLE_INDEX_SCHEMA = 3

#: PLE の資産の役割（models 側の解釈者名 — runtime は解釈しない）。
PLE_INDEX_ROLE = "ple-index"

#: 行の種別 → 役割。並びがそのまま配置の順になるので、`values` が先（token 走査の主）。
PLE_ROLES: Mapping[str, str] = {"values": "ple-values", "scales": "ple-scales"}

#: PLE の格納 → 1 バイトに詰まる要素数（`packages/models/src/gemma/ple-index.ts` の鏡像）。
PLE_PACK_FACTOR: Mapping[str, int] = {"i8": 1, "i2": 4, "i4": 2}

#: PLE の scale 1 個ぶんのバイト数（f32 — 同上）。
PLE_SCALE_BYTES = 4


class PleError(ValueError):
    """PLE の寸法が資産にできない（格納の詰め数で割り切れない・block に行が 1 本も入らない）。"""


def ple_row_bytes(storage: str, layers: int, dim: int) -> dict[str, int]:
    """行の種別 → 1 行のバイト数（`values` は packed・`scales` は層ごとの f32）。

    MUST: どちらも 4 の倍数であること（block は**行の倍数**で切るので、4 の倍数でなければ
    詰め物なしに block 末尾を 4 バイト整列できない — container-v1 §4.1）。
    """
    factor = PLE_PACK_FACTOR.get(storage)
    if factor is None:
        raise PleError(
            f"PLE の格納 '{storage}' を知らない（{' / '.join(sorted(PLE_PACK_FACTOR))}）"
        )
    if dim % factor != 0:
        raise PleError(f"dim {dim} が格納 '{storage}' の詰め数 {factor} で割り切れない")
    row_bytes = {"values": layers * dim // factor, "scales": layers * PLE_SCALE_BYTES}
    for key, stride in row_bytes.items():
        if stride % BLOCK_TAIL_ALIGN != 0:
            raise PleError(
                f"{key} の 1 行 {stride} バイトが {BLOCK_TAIL_ALIGN} の倍数でない"
                "（行の倍数で block に切れない）"
            )
    return row_bytes


def ple_block_ranges(
    *, storage: str, tokens: int, layers: int, dim: int, block_bytes: int = BLOCK_MAX_BYTES
) -> dict[str, tuple[tuple[int, int], ...]]:
    """行の種別 → block ごとの token 区間 `[start, stop)`（{@link ple_assets} と**同じ切り方**）。

    切り目は寸法だけで決まる（payload は 1 バイトも要らない）ので、実体を作る前に区間を知りたい
    呼び手がここを引く — recipe は「block 境界をまたぐ probe token」を実体より先に選ぶ。
    """
    ranges: dict[str, tuple[tuple[int, int], ...]] = {}
    for key, stride in ple_row_bytes(storage, layers, dim).items():
        per_block = block_bytes // stride
        if per_block < 1:
            raise PleError(f"{key} の 1 行 {stride} バイトが block 上限 {block_bytes} を超える")
        ranges[key] = tuple(
            (start, min(start + per_block, tokens)) for start in range(0, tokens, per_block)
        )
    return ranges


def ple_assets(
    *,
    storage: str,
    tokens: int,
    layers: int,
    dim: int,
    embed_scale: float,
    read_values: Callable[[int, int], Buffer],
    read_scales: Callable[[int, int], Buffer],
    block_bytes: int = BLOCK_MAX_BYTES,
) -> dict[str, AssetInput]:
    """PLE を容器の資産へ（索引 schema 3 + `values` / `scales` の block 列）。

    `read_values` / `read_scales` は**連結した行優先 payload のバイト範囲** `[begin, end)` を
    返す呼び出し（引かれるたびに同じバイト列を返す MUST — 書き手は 2 度引く）。移行 CLI は旧
    shard 列の区間読み、recipe は手元のテンソルの slice をここへ渡す。旧 shard の境界は意味を
    持たない（全部を token 順に連結して切り直す）。

    返る並びがそのまま物理配置の順（`values` の block 列 → `scales` の block 列 → 索引）。
    """
    row_bytes = ple_row_bytes(storage, layers, dim)
    cut = ple_block_ranges(
        storage=storage, tokens=tokens, layers=layers, dim=dim, block_bytes=block_bytes
    )
    readers = {"values": read_values, "scales": read_scales}
    assets: dict[str, AssetInput] = {}
    document: dict[str, Any] = {
        "schema": PLE_INDEX_SCHEMA,
        "storage": storage,
        "tokens": tokens,
        "layers": layers,
        "dim": dim,
        "embedScale": embed_scale,
    }
    for key, role in PLE_ROLES.items():
        stride = row_bytes[key]
        read = readers[key]
        blocks: list[dict[str, Any]] = []
        for start, stop in cut[key]:
            name = f"ple.{key}.{len(blocks)}"
            blocks.append({"asset": name, "start": start, "stop": stop})
            # 区間読みを要する block なので専用 part に単独で置く（container-v1 §4.2）。
            assets[name] = AssetInput(
                role,
                (stop - start) * stride,
                _slice(read, start * stride, stop * stride),
                dedicated_part=True,
            )
        document[key] = {"rowBytes": stride, "blocks": blocks}
    encoded = canonical_json(document).encode("utf-8")
    assets[PLE_INDEX_ASSET] = AssetInput(PLE_INDEX_ROLE, len(encoded), encoded)
    return assets


def _slice(read: Callable[[int, int], Buffer], begin: int, end: int) -> Callable[[], Buffer]:
    """1 block ぶんの遅延読み（引かれるまで実体を作らない）。"""

    def pull() -> Buffer:
        return read(begin, end)

    return pull
