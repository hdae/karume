"""配布コンテナの shard 分割規則（ADR 0070 決定 1 / ADR 0071 決定 4 の解除）。

1 コンポーネントの safetensors を**グラフ shard（`karume_ir` + 先頭から詰めたテンソル）+
後続 shard 群**へ決定的に割り付ける層。持つのは規則だけ（テンソルの実体も torch も知らない）
— 書き手は {@link karume.emit}、配布形の宣言は {@link karume.dist}、読み返しの門は
{@link karume.verify} が、この 1 箇所の綴りを共有する。

## 割り付けの規則

1. **順序は変えない**: 詰める順は書き手が決めた書き出し順（ADR 0063 の整列規約）そのまま。
   並べ替えは shard の**中**でだけ起きる（各 shard は自分のテンソルだけで同じ規約を満たす
   独立に整合な safetensors になる）。
2. **weight と companion scale は原子対**（co-shard MUST — ADR 0070 決定 1）: 逐次消費は
   両方を同時に要求するので、shard を跨ぐと「参照を手放す」契約と両立しない。対の片方に
   到達した時点で相方ごと同じ shard へ入れる（= 相方が順序上あとに居ても引き寄せる）。
3. **payload 合計が {@link SHARD_BYTE_LIMIT} を超える手前で次の shard を開く**。総量が上限
   以下なら shard は 1 本のまま = **従来どおり単一ファイル**（既存資産の再 dist はバイト単位で
   不変）。
4. **尾部スラック**: カットを打つ直前に「まだ閉じていないバイト」（今の shard に積んだぶん +
   未割り付けの残り全部）を見て、それが {@link SHARD_TAIL_LIMIT} 以下ならカットを打たず、
   残りを今の shard へ詰め切って終わる。中途半端な端数 shard（数百 MB）を作らないための規則で、
   1.5GiB 以下の資産は分割ゼロになる。
5. 単一の対だけで上限を超える場合は fail loudly（黙って上限を破らない）。**尾部スラックは
   この門を緩めない**（判定は {@link SHARD_BYTE_LIMIT} 基準のまま）。

## ヘッダを勘定に入れない理由

上限に数えるのは**データ節のバイト数**だけで、safetensors ヘッダ（JSON + 8 バイト長 +
`karume_ir` 埋め込み）は数えない。上限 1GiB に対して余裕は 1GiB 近くあり（下の上限の根拠）、
グラフ JSON は実測で数 MB 級・ヘッダ本体はテンソル 1 本あたり 100 バイト前後なので、
1 shard あたりのヘッダが余裕を食い潰す形にならない。勘定に入れると「ヘッダ長を決めるには
所属を決める必要があり、所属を決めるにはヘッダ長が要る」という循環になるので、余裕で
吸収する側を選ぶ。
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath

#: 1 shard のデータ節の上限（1GiB）。**固定定数で、公開ノブにしない** — 配布形の不変条件で
#: あって呼び手の好みではない（緩めた資産は特定の環境でだけ読めなくなる）。
#:
#: 根拠 3 点:
#:
#: 1. Chromium の単一 `ArrayBuffer` 上限 2,145,386,496 バイトの**約 1/2**。shard 1 本は
#:    ホスト側で 1 つの `ArrayBuffer` に載るので、この天井を割るのが必要条件（docs/limitations）。
#:    半分に取るのは、ヘッダぶんの余裕（上のモジュール doc）と、取得層が同時に触る
#:    バイト列（検証中の 1 本 + 引き渡し中の 1 本）を見込むため。
#: 2. `streamAssets` の RAM ピークは O(最大 shard)（ADR 0070 決定 2）。1GiB ならモバイルの
#:    安全域に収まり、2GiB 級だと Pixel の実測失敗域（known-issues の BodyStreamBuffer）に入る。
#: 3. hub 側の同時 RAM 予算 1.5GiB（2026-08-25 の取得層 fix）と整合する — 1 shard を持ちながら
#:    次を取りに行っても予算内に収まる。
SHARD_BYTE_LIMIT = 1_073_741_824

#: **最後の** shard だけに許すデータ節の上限（1.5GiB）。ここまでなら端数を切らずに詰め切る
#: （モジュール doc の規則 4）。`SHARD_BYTE_LIMIT` と同じく固定定数で、公開ノブにしない。
#:
#: 根拠 3 点:
#:
#: 1. hub の同時 RAM 予算 1.5GiB（2026-08-25 の取得層 fix）と**同値**。1 shard を持ちながら
#:    次を取りに行ける上限をそのまま尾部の上限に据える（予算を 1 バイトも広げない）。
#: 2. Chromium の単一 `ArrayBuffer` 上限 2,145,386,496 バイトの約 75%。壁までの余裕は 500MB 級
#:    あり、ヘッダ（上限に数えない — 上のモジュール doc）を吸収してなお届かない。
#: 3. 端数 shard（数百 MB）を作らない。1.5GiB 以下の資産は分割ゼロで据わる
#:    （例: 1.11GiB の text_encoder・1.14GiB の turbo i4 は単一ファイルのまま）。
SHARD_TAIL_LIMIT = 1_610_612_736

#: 1 dtype エントリが並べられる shard 数（ADR 0071 決定 2 — hub の `MAX_SHARDS` と同値）。
#: 焼く側で先に落とすため、書き手（分割）と宣言側（manifest 検査）が同じ綴りを引く。
MAX_SHARDS = 1024

#: 連番の桁数（`-NNNNN-of-NNNNN`）。HF の慣行に似せた形だが `index.json` は作らない —
#: 振り分け表の正本は manifest（ADR 0070 決定 1）。
_INDEX_DIGITS = 5
_MAX_INDEX = 10**_INDEX_DIGITS - 1


class ShardError(ValueError):
    """分割規則の前提が破れた（上限を単独で超える対・不完全な連番・両形の同居）。"""


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


def pack_shards(
    order: Sequence[str],
    payload_bytes: Mapping[str, int],
    companions: Mapping[str, str],
    limit: int = SHARD_BYTE_LIMIT,
    tail_limit: int = SHARD_TAIL_LIMIT,
) -> list[tuple[str, ...]]:
    """書き出し順を shard へ逐次詰めする（モジュール doc の規則 1〜5）。

    `order` は書き手が決めた**全体の**書き出し順、`payload_bytes` は名前 → データ節の
    バイト数、`companions` は原子対の**対称**写像（weight → scale と scale → weight の両方）。
    返すのは shard ごとの名前列で、**並びは `order` の部分列 + 引き寄せた相方**（shard の中の
    最終的な書き出し順は書き手が自分の規約で決め直す）。

    カットの判定は 2 段。①今の shard が非空で `used + size` が `limit` を超える（従来）
    ②かつ**まだ閉じていないバイト**（`used` + 未割り付けの残り全部）が `tail_limit` を
    超える。②が偽なら残りは全部この shard に入り切って `tail_limit` 以下で収まるので、
    カットを打たずに詰め切る（= 端数 shard を作らない）。`used` を判定に含めるのが要で、
    未割り付けぶんだけを見ると「今の shard に積んだぶん + 残り」が `tail_limit` を破る
    畳み方を許してしまう。

    検算（規則の逐語 — テストと対で固定する）:

    - 3.2GiB → `[1GiB, 1GiB, 1.2GiB]`。1 本目・2 本目のカット判定では未閉が 3.2 / 2.2GiB で
      1.5GiB を超えるので普通に切り、3 本目の判定で未閉 1.2GiB ≤ 1.5GiB になって畳む。
    - 2.6GiB → `[1GiB, 1GiB, 0.6GiB]`。判定時の未閉は 2.6 / 1.6GiB でどちらも 1.5GiB 超なので
      従来どおり切る。端数 0.6GiB は残るが、1.6GiB を最後の shard にすると尾部上限を破る
      ので、これが正しい帰結。
    - 総量 1.11GiB → 単一（最初のカット判定で未閉 1.11GiB ≤ 1.5GiB）。

    従来と差が出るのは「未閉が (limit, tail_limit] に入る判定点」だけで、総量が `limit` 以下の
    資産（= 既存の単一ファイル配布）はカット判定自体が起きないのでバイト単位で不変。

    MUST: 空の shard を作らない。上限を超えるのは「今の shard が非空のとき」だけ新しい shard を
    開く条件で、単独で上限を超える対は次の shard へ送っても同じなので fail loudly（尾部
    スラックはこの門を緩めない — 判定は `limit` 基準のまま）。**唯一の例外が `order` 自体が
    空の場合**で、そのときは空の shard 1 本を返す — グラフ shard はテンソルが 1 本も無くても
    必ず在る（`karume_ir` を載せる器そのもの）。
    """
    if limit <= 0:
        raise ShardError(f"shard の上限 {limit} が正でない")
    if tail_limit < limit:
        raise ShardError(f"尾部の上限 {tail_limit:,} が shard 上限 {limit:,} を下回っている")
    units: list[tuple[tuple[str, ...], int]] = []
    assigned: set[str] = set()
    for name in order:
        if name in assigned:
            continue
        partner = companions.get(name)
        unit = (name,) if partner is None or partner in assigned else (name, partner)
        size = sum(payload_bytes[member] for member in unit)
        if size > limit:
            raise ShardError(
                f"{' + '.join(unit)} だけで {size:,} バイト = shard 上限 {limit:,} を超える"
                "（1 対は分割できない — co-shard MUST）"
            )
        assigned.update(unit)
        units.append((unit, size))
    groups: list[tuple[str, ...]] = []
    current: list[str] = []
    used = 0
    # まだ閉じていないバイト（= 総量 − 確定した shard のバイト）。カットのたびに減る。
    unclosed = sum(size for _, size in units)
    for unit, size in units:
        if current and used + size > limit and unclosed > tail_limit:
            groups.append(tuple(current))
            unclosed -= used
            current = []
            used = 0
        current.extend(unit)
        used += size
    if current or not groups:
        groups.append(tuple(current))
    if len(groups) > MAX_SHARDS:
        raise ShardError(f"shard が {len(groups)} 本で上限 {MAX_SHARDS} を超えた")
    return groups


def assert_shard_partition(groups: Sequence[Sequence[str]], names: Sequence[str]) -> None:
    """shard 群が対象テンソルの**分割**（欠け・重複・余剰なし）であることを落とす。

    ランタイム側の宣言完全性検査（ADR 0070 決定 1 — 全 shard 読了後の突合）を、書く側でも
    同じ集合に対して張る。ここが緩むと「配布形は書けたのにロードで落ちる」非対称になる。
    """
    seen: dict[str, int] = {}
    for index, group in enumerate(groups):
        for name in group:
            if name in seen:
                raise ShardError(
                    f"テンソル '{name}' が shard[{seen[name]}] と shard[{index}] に重複している"
                )
            seen[name] = index
    expected = set(names)
    missing = sorted(expected - set(seen))
    surplus = sorted(set(seen) - expected)
    if missing or surplus:
        raise ShardError(f"shard の割り付けが分割になっていない（欠け {missing} / 余剰 {surplus}）")


def assert_co_shard(groups: Sequence[Sequence[str]], companions: Mapping[str, str]) -> None:
    """weight と companion scale が同じ shard に居ることを落とす（co-shard MUST）。

    {@link pack_shards} は対を原子単位で詰めるので構造的に破れないが、**規則と検査は別物**に
    しておく（分割の入口が増えたとき、規則の写経ではなく検査が受け止める）。ランタイム側の
    `createShardValidator` の鏡像で、破れると「shard を跨いだ scale」を読む側が拒否する。
    """
    owner = {name: index for index, group in enumerate(groups) for name in group}
    for name, partner in companions.items():
        if name not in owner or partner not in owner:
            continue
        if owner[name] != owner[partner]:
            raise ShardError(
                f"'{name}' が shard[{owner[name]}]・相方の '{partner}' が"
                f" shard[{owner[partner]}] に落ちている（weight と companion scale は同一 shard"
                " MUST — ADR 0070 決定 1）"
            )
