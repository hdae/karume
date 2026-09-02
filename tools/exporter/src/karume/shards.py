"""配布コンテナの shard 分割規則（ADR 0081 + 0090 — shard 仕様 v3・テンソル分割）。

1 コンポーネントの safetensors を**グラフ shard（`karume_ir` だけ・データ節は空）+ weight
shard 列**へ決定的に割り付ける層。持つのは規則だけ（テンソルの実体も torch も知らない）—
書き手は {@link karume.emit}、配布形の宣言は {@link karume.dist}、読み返しの門は
{@link karume.verify} が、この 1 箇所の綴りを共有する。

規則は 2 層で、**読み手契約**（配布形の不変条件 — verify が門にする）と**書き手ポリシー**
（本数と cut 位置の決め方 — ここの裁量）は別物。後者を変えても既存の配布形は読めるままで、
前者を変えると読み手ごと動く。

## 読み手契約（フォーマット）

1. **shard 0 = グラフ shard**: `karume_ir` メタデータだけを持ち、データ節は**0 テンソル**
   （器は safetensors のまま）。グラフだけを先に取れる形にするのが目的で、admission
   （ADR 0070 決定 5）が「グラフ 1 本ぶんの DL / RAM」で判断できる。
2. **全 shard の**ファイル長**が {@link SHARD_BYTE_LIMIT} 以下**（読み手が見る上限はこの 1 本
   だけ — ADR 0090 決定 2）。データ節ではなく**ファイル長**で測るのが v2 からの変更点 —
   読み手が確保するのはファイル 1 本ぶんのバイト列で、ヘッダもそこに載る（hub は manifest の
   `size`、verify は実ファイル長で見る）。
3. **順序は変えない**: 詰める順は書き手が決めた書き出し順（ADR 0063 の整列規約）そのまま。
   並べ替えは shard の**中**でだけ起きる（各 shard は自分のテンソルだけで同じ規約を満たす
   独立に整合な safetensors になる）。
4. **weight と companion scale は原子対**（co-shard MUST — ADR 0070 決定 1）: 逐次消費は
   両方を同時に要求するので、shard を跨ぐと「参照を手放す」契約と両立しない。対の片方に
   到達した時点で相方ごと同じ shard へ入れる（= 相方が順序上あとに居ても引き寄せる）。
   重みが分割されているときは **piece 1 と同じ shard**（scale 自体は割らない）。
5. **テンソル分割（piece — ADR 0090 決定 1）**: 上限に収まらないテンソルは**先頭次元（行）の
   範囲**で複数 shard へ割って配る。キーは `<親名>#NNNNN-of-NNNNN`（{@link piece_key}）で、
   そのキーがこの形に一致し**親名が宣言（`initializer.tensor`）に在るときだけ** piece と解釈する
   （karume が書く通常のテンソル名に `#` は現れない — torch の state_dict キー由来）。
   - dtype は親と同一・shape は `[行数, *親.shape[1:]]`・行数 ≥ 1
   - piece 1..n は親の行 `[0, rows)` を順に**隙間なく**覆う
   - piece は**連続する shard に 1 本ずつ**（index は shard 順に増える。同じ shard に同じ親の
     piece が 2 本は違反）。piece 1 の shard には前のテンソルが、piece n の shard には後の
     テンソルが同居してよい
   - 1 テンソルは「丸ごと」か「piece 列」のどちらか一方（混在は違反）
   - **末尾以外の piece はバイト長が 4 の倍数 MUST** — 読み手が `queue.writeBuffer` で親
     バッファへオフセット書きするため。末尾 piece は任意長（読み手の末尾詰め物が整列する）
6. 重複禁止・`karume_ir` は shard 0 だけ・本数は {@link MAX_SHARDS} 以下。
7. **常時分割**（ADR 0081）: fat グラフ shard（shard 0 に実重みを載せる形）と単一ファイル
   配布形は**廃止**。テンソルを 1 本でも持つコンポーネントは「グラフ shard + weight shard
   1 本以上」になる。

## 書き手ポリシー（本数と cut 位置）

- 詰める大きさ（capacity）= {@link SHARD_DATA_CAPACITY}（= 受理上限 − ヘッダ余裕）。v2 の
  「実効目標 = max(目標, 最大単位)」はテンソル分割が入ったぶん**不要**になった（割れない
  単位がもう無いので、目標は常に目標のまま）。
- 容量に収まらない単位の重みは**行ブロック**（{@link SPLIT_BLOCK_BYTES} 刻み・末尾以外が
  4 バイト整列になる行数）へ砕き、**通常の単位として**詰める。詰め終わってから、同じ shard に
  落ちた同一親の連続ブロックを 1 本の {@link Piece} へ畳む。
- 本数 k = capacity の下での**最小連続分割数**（貪欲に詰めて数える — 連続分割では貪欲が最小）。
- k を固定して**均す**: 各 shard の目標 = 残量 ÷ 残 shard 数。単位（weight + scale の対 /
  行ブロック）を跨がず、**suffix 実行可能性ガード**（残りの単位が残りの shard へ capacity 内で
  収まる位置でだけ cut を打つ）を掛けるので、全 shard が capacity 以下に収まることは構造的に
  保たれる。
- cut 位置の選好（層割り・MoE のエキスパート割り等）は**将来の書き手ポリシー拡張**で、
  読み手契約を 1 文字も変えずに足せる（ADR 0081 の扉 — 今は実装しない）。

## ヘッダぶんの余裕

受理上限は**ファイル長**なので、書き手はデータ節を {@link SHARD_DATA_CAPACITY} までに留めて
ヘッダ（8 バイト長 + JSON + `karume_ir` 埋め込み）ぶんの {@link SHARD_HEADER_ALLOWANCE} を
空けておく。weight shard のヘッダはテンソル 1 本あたり 100〜150 バイトなので、1MiB は
7,000 本ぶんにあたる。固定の余裕にするのは「所属を決めるにはヘッダ長が要り、ヘッダ長を
決めるには所属が要る」という循環を避けるためで、それでも足りなかった回は verify の
ファイル長門が fail loudly で受ける（書き手側で黙って縮めない）。
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

#: 1 shard の**ファイル長**の受理上限（256MiB — 読み手契約。hub の `MAX_SHARD_BYTES` と同値）。
#: 固定定数・公開ノブにしない — 不変条件であって呼び手の好みではない（緩めた資産は特定の環境で
#: だけ読めなくなる）。書き手がデータ節に詰める大きさは {@link SHARD_DATA_CAPACITY}。
#:
#: 根拠 3 点:
#:
#: 1. ロード時のホスト RAM ピークは「定数 + 最大 shard 1 本」（器の使い回し — ADR 0070 追記
#:    2026-09-02）。shard 1 本の大きさがそのままピークの上乗せ分になるので、上限はモバイルの
#:    安全域に収まる値で切る。
#: 2. ファイル数・リクエスト数の増加が hub の 4 並列取得と読み手上限 1024 本の内側に収まる
#:    下限側の値（docs/research/2026-09-02-shard-size-ram-peak.md）。
#: 3. Chromium の単一 `ArrayBuffer` 上限 2,145,386,496 バイトの**十分下**。shard 1 本は
#:    ホスト側で 1 つの `ArrayBuffer` に載るので、この天井を割るのが必要条件
#:    （docs/limitations）。
SHARD_BYTE_LIMIT = 268_435_456

#: ヘッダのために空けておく余裕（1MiB — モジュール doc の「ヘッダぶんの余裕」）。
SHARD_HEADER_ALLOWANCE = 1_048_576

#: 書き手がデータ節に詰める上限（= 受理上限 − ヘッダ余裕）。
SHARD_DATA_CAPACITY = SHARD_BYTE_LIMIT - SHARD_HEADER_ALLOWANCE

#: 上限を超えるテンソルを砕く**行ブロック**の刻み（1MiB）。刻みが細かいほど shard の詰まり方は
#: 均一になり、粗いほど piece の本数（= ヘッダ項目と読み手の writeBuffer 呼び出し）が減る。
#: 実際の 1 ブロックは「この値に届く最小の行数」を 4 バイト整列（読み手契約 5）へ丸めた行数で、
#: 容量に収まらなければそこまで縮む（{@link _block_rows}）。
SPLIT_BLOCK_BYTES = 1_048_576

#: 1 dtype エントリが並べられる shard 数（ADR 0071 決定 2 — hub の `MAX_SHARDS` と同値）。
#: 焼く側で先に落とすため、書き手（分割）と宣言側（manifest 検査）が同じ綴りを引く。
MAX_SHARDS = 1024

#: 連番の桁数（`-NNNNN-of-NNNNN` / piece キーの `#NNNNN-of-NNNNN`）。HF の慣行に似せた形だが
#: `index.json` は作らない — 振り分け表の正本は manifest（ADR 0070 決定 1）。
_INDEX_DIGITS = 5
_MAX_INDEX = 10**_INDEX_DIGITS - 1

#: piece キーの綴り（`<親名>#NNNNN-of-NNNNN`）。親名は貪欲に取る — `#` を含む親名は karume の
#: 書き出しには現れないので、末尾の連番だけが分割の印になる。
_PIECE_PATTERN = re.compile(rf"^(.+)#(\d{{{_INDEX_DIGITS}}})-of-(\d{{{_INDEX_DIGITS}}})$")


class ShardError(ValueError):
    """分割規則の前提が破れた（分割できない単位・不完全な連番・両形の同居）。"""


def piece_key(name: str, index: int, count: int) -> str:
    """分割テンソルの 1 断片の safetensors キー（`<親名>#NNNNN-of-NNNNN`・`index` は 1 始まり）。

    綴りは読み手契約（モジュール doc の 5）— TS ランタイム側の正規表現と 1 文字も違わない。
    `count` が 2 未満なら分割ではないので fail loudly（丸ごと 1 本を piece キーで名乗ると、
    読み手が「丸ごとと piece の混在」を判定できる根拠が消える）。
    """
    if count < 2:
        raise ShardError(
            f"piece の総数 {count} が 2 未満（分割は 2 本以上 — 丸ごとは piece でない）"
        )
    if not 1 <= index <= count <= _MAX_INDEX:
        raise ShardError(f"piece 連番 {index}/{count} が 1..{_MAX_INDEX} の範囲に無い")
    return f"{name}#{index:0{_INDEX_DIGITS}d}-of-{count:0{_INDEX_DIGITS}d}"


def parse_piece_key(key: str) -> tuple[str, int, int] | None:
    """piece キー → `(親名, index, count)`（piece でなければ None）。

    ランタイム側 `packages/runtime/src/format/container.ts` の `parsePieceKey` の鏡像。

    MUST: `count >= 2` と `1 <= index <= count` を**ここで**見る。範囲外の綴り
    （`#00003-of-00002`・1 本しかない piece 列）を piece として通すと、それが列の進行状態の
    初期値になって違反の帰属が「piece 列の並び」へ移る。実際にはそのキー自体が配布形の誤り
    なので、piece と解釈せず**余剰テンソル**（どの initializer からも参照されない）として
    落とすほうが直す側にとって決定的になる。列そのものの整合（1..n が揃う・連続 shard・行の
    被覆）は畳む側の門が持つ（{@link assert_shard_partition} / `karume.verify` /
    `karume.repack`）。
    """
    match = _PIECE_PATTERN.fullmatch(key)
    if match is None:
        return None
    index, count = int(match.group(2)), int(match.group(3))
    if count < 2 or not 1 <= index <= count:
        return None
    return match.group(1), index, count


@dataclass(frozen=True)
class Piece:
    """分割テンソルの 1 断片（親の**行範囲** `[begin, end)`）。

    割り付けの結果に現れる値で、実バイトは持たない（このモジュールはテンソルの実体を知らない
    — 行範囲からバイト範囲を出すのは書き手側）。
    """

    #: 親テンソルのキー（safetensors のキー空間）。
    name: str
    #: 1 始まりの連番（shard 順に増える）。
    index: int
    #: この親の piece 総数（2 以上）。
    count: int
    #: 親の先頭次元での行範囲 `[begin, end)`。
    begin: int
    end: int

    @property
    def key(self) -> str:
        """この断片が safetensors に載るときのキー。"""
        return piece_key(self.name, self.index, self.count)


def shard_name(name: str, index: int, total: int) -> str:
    """`<拡張子の前>-NNNNN-of-NNNNN<拡張子>`（`index` は 1 始まり）。

    `name` は path 片でもよい（最終要素だけを書き換える）— 配布形の相対 path と系列の実 path が
    同じ綴りで導出できる。
    """
    if not 1 <= index <= total <= min(MAX_SHARDS, _MAX_INDEX):
        raise ShardError(
            f"shard 連番 {index}/{total} が 1..{min(MAX_SHARDS, _MAX_INDEX)} の範囲に無い"
        )
    path = PurePosixPath(name)
    numbered = f"{index:0{_INDEX_DIGITS}d}-of-{total:0{_INDEX_DIGITS}d}"
    return str(path.with_name(f"{path.stem}-{numbered}{path.suffix}"))


def shard_path(path: Path, index: int, total: int) -> Path:
    """{@link shard_name} の `Path` 版（親ディレクトリはそのまま）。"""
    return path.with_name(shard_name(path.name, index, total))


def _sequence_pattern(path: Path) -> re.Pattern[str]:
    """`path` と同じコンポーネントの shard ファイル名に一致する正規表現。

    stem / suffix は `re.escape` する — 実 path にはドットもハイフンも入るので、素で埋めると
    無関係なファイルを拾う（glob も同じ理由で使わない: `[` を含む名前が黙って別解釈になる）。
    """
    stem = re.escape(PurePosixPath(path.name).stem)
    suffix = re.escape(PurePosixPath(path.name).suffix)
    return re.compile(rf"^{stem}-(\d{{{_INDEX_DIGITS}}})-of-(\d{{{_INDEX_DIGITS}}}){suffix}$")


def resolve_shards(path: Path) -> tuple[Path, ...]:
    """コンポーネントの**代表 path** → 実在する shard 列（分割されていなければ 1 要素）。

    返すのは常に「読む順 = shard 番号順」で、先頭がグラフ shard。分割されていない資産と
    存在しない資産はどちらも `(path,)` を返す（不在の診断は呼び手の既存の門が持つ —
    ここで先回りすると「組み立ての入力が無い」の綴りが 2 つに割れる）。

    MUST: 曖昧な現場は fail loudly。単一ファイルと連番が**同居**している場合（分割の前後で
    形が変わったのに前回の出力が残っている）、連番の `of` が食い違う場合、番号に欠けや
    はみ出しがある場合は、どれも「どのバイト列を配るか」が一意に決まらない。黙って一方を
    選ぶと、前回の重みを今回の manifest で配る形になる。
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
        raise ShardError(
            f"{path}: 単一ファイルと shard 連番（{len(found)} 本）が同居している"
            " — 前回の書き出しの残骸を消してから組み立てる"
        )
    if len(totals) != 1:
        raise ShardError(f"{path}: shard 連番の総数が {sorted(totals)} と食い違っている")
    total = totals.pop()
    missing = sorted(set(range(1, total + 1)) - set(found))
    surplus = sorted(set(found) - set(range(1, total + 1)))
    if missing or surplus:
        raise ShardError(
            f"{path}: shard 連番 1..{total} が揃っていない（欠け {missing} / はみ出し {surplus}）"
        )
    return tuple(found[index] for index in range(1, total + 1))


def shard_siblings(path: Path) -> tuple[Path, ...]:
    """このコンポーネントの出力になりうる実在ファイル（代表 path + 連番の全件）。

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


@dataclass(frozen=True)
class _Block:
    """分割テンソルの**行ブロック 1 つ**（詰める前の単位 — 畳むと {@link Piece} になる）。

    ブロックのまま詰めるのは、割り付け（最小本数 → 均し）に「分割テンソル」という特例を
    足さないため。同じ shard に落ちた連続ブロックは {@link _fold_blocks} が 1 本の piece へ
    畳むので、読み手が受け取る本数は「shard を跨いだ回数」ちょうどになる。
    """

    name: str
    begin: int
    end: int
    nbytes: int


def _block_rows(row_bytes: int, capacity: int) -> int:
    """1 ブロックの行数（{@link SPLIT_BLOCK_BYTES} に届く最小行数を 4 バイト整列へ丸めた値）。

    末尾以外の piece はバイト長が 4 の倍数 MUST（読み手契約 5）なので、行数は
    `step = 4 / gcd(行バイト長, 4)` の倍数でなければならない — 行バイト長が 4 の倍数なら
    1 行刻み、偶数なら 2 行刻み、奇数なら 4 行刻み。刻みへ切り上げた行数が容量を超える場合は
    容量に収まる最大の刻み倍数まで**縮める**（テストの小容量と、1 行が MiB 級の巨大テンソルの
    両方がここを通る）。

    MUST: 1 ブロック（= `step` 行）すら容量に入らない形は fail loudly。これ以上細かい粒度が
    無いので、黙って上限を破るか読めない配布形を書くかしか残らない。
    """
    step = 1 if row_bytes % 4 == 0 else (2 if row_bytes % 2 == 0 else 4)
    fit = capacity // row_bytes
    if fit < step:
        raise ShardError(
            f"1 行 {row_bytes:,} バイト（4 バイト整列には {step} 行）が shard の容量"
            f" {capacity:,} を超える — これ以上細かく割れない"
        )
    wanted = -(-SPLIT_BLOCK_BYTES // row_bytes)
    rows = -(-wanted // step) * step
    return rows if rows <= fit else fit // step * step


def _row_blocks(name: str, nbytes: int, shape: Sequence[int] | None, capacity: int) -> list[_Block]:
    """1 テンソルを**先頭次元（行）の連続範囲**へ砕く（容量に収まるブロック列）。

    実データは読まない — 宣言（バイト長と shape）だけで決まるので、割り付けはピーク RAM に
    一切載らない。行の並びは safetensors の連続メモリ順そのものなので、行範囲はそのまま
    バイト範囲になる（`begin * 行バイト長` から）。
    """
    if shape is None:
        raise ShardError(f"テンソル '{name}': shape が分からないので行で割れない")
    if not shape:
        raise ShardError(f"テンソル '{name}': rank 0（行が無いので割れない）")
    rows = int(shape[0])
    if rows < 1:
        raise ShardError(f"テンソル '{name}': 先頭次元が {rows} 行（1 行以上でないと割れない）")
    if nbytes % rows:
        raise ShardError(
            f"テンソル '{name}': {nbytes:,} バイトが行数 {rows} で割り切れない"
            "（1 行が整数バイトでない形は行で割れない）"
        )
    row_bytes = nbytes // rows
    if row_bytes < 1:
        raise ShardError(f"テンソル '{name}': 1 行 0 バイト（割る意味が無い）")
    per = _block_rows(row_bytes, capacity)
    blocks: list[_Block] = []
    for begin in range(0, rows, per):
        end = min(begin + per, rows)
        blocks.append(_Block(name=name, begin=begin, end=end, nbytes=(end - begin) * row_bytes))
    return blocks


def _atomic_units(
    order: Sequence[str],
    payload_bytes: Mapping[str, int],
    shapes: Mapping[str, Sequence[int]],
    companions: Mapping[str, str],
    capacity: int,
) -> list[tuple[tuple[str | _Block, ...], int]]:
    """書き出し順を**跨げない単位**（単独テンソル / 対 / 行ブロック）へ畳む。

    対の片方に到達した時点で相方ごと 1 単位にする（相方が順序上あとに居ても引き寄せる —
    読み手契約の 4）。単位が容量に収まらないときは**重みを行ブロックへ砕き**、
    `[scale + ブロック 1], [ブロック 2], …` という**連続した単位列**をその位置に置く
    （ブロック 1 だけを引き寄せて残りが末尾に取り残される形を作らない）。

    MUST: 砕くのは対のうち**書き出し順で後ろ**の側 = 重み。companion scale は F32 群に居るので
    必ず前に来る（ADR 0063 の並び規約）ので、この選び方は「scale は割らない」（読み手契約 4）
    と同値になる。ブロックの大きさは `容量 − 相方` で頭打ちにするので、piece 1 と scale が
    同居できることは構造的に保たれ、piece が 2 本以上になることも保たれる（1 本で収まるなら
    そもそも分割条件を満たさない）。
    """
    units: list[tuple[tuple[str | _Block, ...], int]] = []
    assigned: set[str] = set()
    for name in order:
        if name in assigned:
            continue
        partner = companions.get(name)
        members = [name] if partner is None or partner in assigned else [name, partner]
        size = sum(payload_bytes[member] for member in members)
        assigned.update(members)
        if size <= capacity:
            units.append((tuple(members), size))
            continue
        split = members[-1]
        rest = members[:-1]
        rest_bytes = sum(payload_bytes[member] for member in rest)
        if rest_bytes >= capacity:
            raise ShardError(
                f"'{split}' の相方 {' + '.join(rest)} だけで {rest_bytes:,} バイト = shard の容量"
                f" {capacity:,} 以上（companion scale は割らない — co-shard MUST）"
            )
        blocks = _row_blocks(split, payload_bytes[split], shapes.get(split), capacity - rest_bytes)
        units.append(((*rest, blocks[0]), rest_bytes + blocks[0].nbytes))
        units.extend(((block,), block.nbytes) for block in blocks[1:])
    return units


def _fold_blocks(groups: Sequence[Sequence[str | _Block]]) -> list[tuple[str | Piece, ...]]:
    """同じ shard に落ちた同一親の**連続ブロック**を 1 本の {@link Piece} へ畳む。

    ブロックは 1 テンソルぶんが連続した単位列なので、同じ group に隣り合って落ちたブロックは
    必ず行範囲も連続している。畳んでから連番を振ると、piece の本数と index が「shard を跨いだ
    回数」そのものになる（刻みの細かさが読み手に漏れない）。
    """
    runs: list[list[str | _Block]] = []
    counts: dict[str, int] = {}
    for group in groups:
        row: list[str | _Block] = []
        for member in group:
            if not isinstance(member, _Block):
                row.append(member)
                continue
            previous = row[-1] if row else None
            if isinstance(previous, _Block) and previous.name == member.name:
                row[-1] = _Block(
                    name=member.name,
                    begin=previous.begin,
                    end=member.end,
                    nbytes=previous.nbytes + member.nbytes,
                )
                continue
            row.append(member)
            counts[member.name] = counts.get(member.name, 0) + 1
        runs.append(row)
    numbered: dict[str, int] = {}
    folded: list[tuple[str | Piece, ...]] = []
    for row in runs:
        members: list[str | Piece] = []
        for item in row:
            if not isinstance(item, _Block):
                members.append(item)
                continue
            numbered[item.name] = numbered.get(item.name, 0) + 1
            members.append(
                Piece(
                    name=item.name,
                    index=numbered[item.name],
                    count=counts[item.name],
                    begin=item.begin,
                    end=item.end,
                )
            )
        folded.append(tuple(members))
    return folded


def _suffix_shard_counts(sizes: Sequence[int], limit: int) -> list[int]:
    """`counts[i]` = `sizes[i:]` を上限内の**連続**分割へ収める最小本数（`counts[n] = 0`）。

    貪欲（入るだけ詰めて切る）が連続分割の最小本数そのものなので、各 i について貪欲の
    右端 `end` を取り `1 + counts[end]` で数える。`end` は i が減れば単調に減るので、
    右から 1 度なめるだけで全 i ぶんが出る（二重ループにしない）。

    使い道は 2 つで、`counts[0]` が本数 k、`counts[i]` が均しの**suffix 実行可能性ガード**
    （「ここで cut を打っても残りが残り shard に収まるか」）。
    """
    counts = [0] * (len(sizes) + 1)
    end = len(sizes)
    window = 0
    for start in range(len(sizes) - 1, -1, -1):
        window += sizes[start]
        while window > limit:
            end -= 1
            window -= sizes[end]
        counts[start] = 1 + counts[end]
    return counts


def pack_shards(
    order: Sequence[str],
    payload_bytes: Mapping[str, int],
    shapes: Mapping[str, Sequence[int]],
    companions: Mapping[str, str],
    capacity: int = SHARD_DATA_CAPACITY,
) -> list[tuple[str | Piece, ...]]:
    """書き出し順を shard 列へ割り付ける（規則はモジュール doc）。

    `order` は書き手が決めた**全体の**書き出し順、`payload_bytes` は名前 → データ節の
    バイト数、`shapes` は名前 → 論理 shape（**分割の要否と行の長さにだけ**使う）、
    `companions` は原子対の**対称**写像（weight → scale と scale → weight の両方）。
    返すのは shard ごとの member 列（丸ごとなら名前・分割なら {@link Piece}）で、
    **先頭は必ず空**（= グラフ shard・読み手契約の 1）。以降が weight shard で、並びは
    `order` の部分列 + 引き寄せた相方（shard の中の最終的な書き出し順は書き手が自分の規約で
    決め直す）。

    `capacity` は**データ節**に詰める上限（既定 {@link SHARD_DATA_CAPACITY}）。受理上限は
    ファイル長で測る（{@link SHARD_BYTE_LIMIT}）ので、ここが見るのはヘッダ余裕を引いた側。

    割り付けは 3 段。①容量に収まらない単位を**行ブロック**へ砕く（`_atomic_units`）
    ②**最小本数** k を貪欲で数える（`_suffix_shard_counts`）③k を固定して**均す** — 各 shard の
    目標を「残量 ÷ 残 shard 数」に取り、目標に届いたら cut を打つ。cut を打てるのは suffix
    実行可能性ガード（`counts[i] <= 残 shard 数 - 1`）が立つ位置だけなので、均しても最後の
    1 本が容量を破ることはない。容量に届いてしまった場合は目標に関係なく cut（そこがちょうど
    貪欲の右端で、ガードは必ず立っている）。最後に、同じ shard に落ちた連続ブロックを
    {@link Piece} へ畳む。

    検算（規則の逐語 — テストと対で固定する。capacity を明示した場合）:

    - 容量の 3.2 倍 → 0.8 倍 × 4 本。k = 4 を先に決めてから均すので端数 shard が出ない。
    - 容量の 0.6 倍の**対**が 3 つ → 3 本。隣接 2 つで 1.2 倍になるのでどの 2 つも同居できない
      （連続順序 + 対の原子性の下では k > ceil(総量 / 容量) がありうる — モジュール doc）。
    - 容量を超える単独テンソルは piece 列になり、連続する shard に 1 本ずつ並ぶ。

    MUST: weight shard を空で作らない。k は「全単位を容量内で覆う最小本数」なので、均しの
    ガードが立つ位置は常に残り単位が残っている位置になる。**テンソルが 1 本も無い**
    コンポーネントだけが例外で、そのときはグラフ shard 1 本（`[()]`）を返す。
    """
    if capacity <= 0:
        raise ShardError(f"shard の容量 {capacity} が正でない")
    units = _atomic_units(order, payload_bytes, shapes, companions, capacity)
    groups: list[list[str | _Block]] = [[]]
    if units:
        sizes = [size for _, size in units]
        counts = _suffix_shard_counts(sizes, capacity)
        total = counts[0]
        remaining = sum(sizes)
        index = 0
        for opened in range(total):
            left = total - opened
            # 残量を残 shard 数で割った目標（切り上げ — 端数は手前の shard が引き受ける）。
            target = -(-remaining // left)
            members: list[str | _Block] = []
            used = 0
            while index < len(units):
                unit, size = units[index]
                closable = bool(members) and counts[index] <= left - 1
                if closable and (used + size > capacity or used >= target):
                    break
                members.extend(unit)
                used += size
                index += 1
            groups.append(members)
            remaining -= used
    if len(groups) > MAX_SHARDS:
        raise ShardError(f"shard が {len(groups)} 本で上限 {MAX_SHARDS} を超えた")
    return _fold_blocks(groups)


def _owner_of(groups: Sequence[Sequence[str | Piece]]) -> dict[str, int]:
    """テンソル → 所属 shard 添字（分割テンソルは **piece 1** の shard）。"""
    owner: dict[str, int] = {}
    for index, group in enumerate(groups):
        for member in group:
            if isinstance(member, Piece):
                if member.index == 1:
                    owner[member.name] = index
            else:
                owner[member] = index
    return owner


def _assert_piece_run(
    name: str, placed: Sequence[tuple[int, Piece]], shapes: Mapping[str, Sequence[int]]
) -> None:
    """1 テンソルの piece 列が読み手契約 5 を満たすことを落とす（shard 順に受け取る）。"""
    shape = shapes.get(name)
    if shape is None or not shape:
        raise ShardError(f"テンソル '{name}': 分割されているのに親の shape が分からない")
    count = placed[0][1].count
    if len(placed) != count:
        raise ShardError(
            f"テンソル '{name}': piece が {len(placed)} 本で宣言の総数 {count} と合わない"
        )
    cursor = 0
    previous: int | None = None
    for position, (shard, piece) in enumerate(placed, start=1):
        if piece.count != count:
            raise ShardError(
                f"テンソル '{name}': piece の総数が {count} と {piece.count} で食い違っている"
            )
        if piece.index != position:
            raise ShardError(
                f"テンソル '{name}': shard 順で {position} 本目の piece が index {piece.index}"
                "（index は shard 順に 1 から増える）"
            )
        if previous is not None and shard != previous + 1:
            raise ShardError(
                f"テンソル '{name}': piece {piece.index} が shard[{shard}]・前の piece が"
                f" shard[{previous}]（piece は連続する shard に 1 本ずつ）"
            )
        if piece.end <= piece.begin:
            raise ShardError(
                f"テンソル '{name}': piece {piece.index} の行範囲"
                f" [{piece.begin}, {piece.end}) が空（1 行以上 MUST）"
            )
        if piece.begin != cursor:
            raise ShardError(
                f"テンソル '{name}': piece {piece.index} が行 {piece.begin} から始まる"
                f"（前の piece の末尾は {cursor} — 行は隙間なく覆う MUST）"
            )
        cursor = piece.end
        previous = shard
    if cursor != int(shape[0]):
        raise ShardError(
            f"テンソル '{name}': piece 列が行 {cursor} までしか覆っていない"
            f"（親は {int(shape[0])} 行）"
        )


def assert_shard_partition(
    groups: Sequence[Sequence[str | Piece]],
    names: Sequence[str],
    shapes: Mapping[str, Sequence[int]],
) -> None:
    """shard 群が対象テンソルの**分割**（欠け・重複・余剰なし）であることを落とす。

    ランタイム側の宣言完全性検査（ADR 0070 決定 1 — 全 shard 読了後の突合）を、書く側でも
    同じ集合に対して張る。ここが緩むと「配布形は書けたのにロードで落ちる」非対称になる。
    1 テンソルは**丸ごと 1 回**か、**index 順に連続 shard へ 1 本ずつ並ぶ piece 列**の
    どちらか一方（読み手契約 5 — 混在・欠け・重複・非連続・行の取りこぼしは全部ここで落ちる）。
    """
    whole: dict[str, int] = {}
    pieces: dict[str, list[tuple[int, Piece]]] = {}
    for index, group in enumerate(groups):
        for member in group:
            if isinstance(member, Piece):
                pieces.setdefault(member.name, []).append((index, member))
                continue
            if member in whole:
                raise ShardError(
                    f"テンソル '{member}' が shard[{whole[member]}] と"
                    f" shard[{index}] に重複している"
                )
            whole[member] = index
    mixed = sorted(set(whole) & set(pieces))
    if mixed:
        raise ShardError(
            f"テンソル {mixed} が丸ごとと piece の両方で割り付けられている"
            "（1 テンソルはどちらか一方 MUST）"
        )
    for name in sorted(pieces):
        _assert_piece_run(name, pieces[name], shapes)
    expected = set(names)
    seen = set(whole) | set(pieces)
    missing = sorted(expected - seen)
    surplus = sorted(seen - expected)
    if missing or surplus:
        raise ShardError(f"shard の割り付けが分割になっていない（欠け {missing} / 余剰 {surplus}）")


def assert_co_shard(groups: Sequence[Sequence[str | Piece]], companions: Mapping[str, str]) -> None:
    """weight と companion scale が同じ shard に居ることを落とす（co-shard MUST）。

    分割された重みは **piece 1** の shard が所属（読み手契約 4 の piece 版）— 逐次消費は
    「重みの先頭を読み始める時点で scale が要る」ので、scale は最初の断片と同居する。

    {@link pack_shards} は対を原子単位で詰めるので構造的に破れないが、**規則と検査は別物**に
    しておく（分割の入口が増えたとき、規則の写経ではなく検査が受け止める）。ランタイム側の
    `createShardValidator` の鏡像で、破れると「shard を跨いだ scale」を読む側が拒否する。
    """
    owner = _owner_of(groups)
    for name, partner in companions.items():
        if name not in owner or partner not in owner:
            continue
        if owner[name] != owner[partner]:
            raise ShardError(
                f"'{name}' が shard[{owner[name]}]・相方の '{partner}' が"
                f" shard[{owner[partner]}] に落ちている（weight と companion scale は同一 shard"
                " MUST — ADR 0070 決定 1）"
            )
