"""配布コンテナの shard 分割規則（ADR 0081 — ADR 0070 決定 1 / ADR 0071 決定 4 の改訂）。

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
2. **全 shard のデータ節が {@link SHARD_BYTE_LIMIT} 以下**（上限はこの 1 本だけ）。単独で
   上限を超える対の reject は「1 shard に入らない」から自動で従うので、独立の定数にしない。
3. **順序は変えない**: 詰める順は書き手が決めた書き出し順（ADR 0063 の整列規約）そのまま。
   並べ替えは shard の**中**でだけ起きる（各 shard は自分のテンソルだけで同じ規約を満たす
   独立に整合な safetensors になる）。
4. **weight と companion scale は原子対**（co-shard MUST — ADR 0070 決定 1）: 逐次消費は
   両方を同時に要求するので、shard を跨ぐと「参照を手放す」契約と両立しない。対の片方に
   到達した時点で相方ごと同じ shard へ入れる（= 相方が順序上あとに居ても引き寄せる）。
5. 重複禁止・`karume_ir` は shard 0 だけ・本数は {@link MAX_SHARDS} 以下。
6. **常時分割**（ADR 0081）: fat グラフ shard（shard 0 に実重みを載せる形）と単一ファイル
   配布形は**廃止**。テンソルを 1 本でも持つコンポーネントは「グラフ shard + weight shard
   1 本以上」になる。

## 書き手ポリシー（本数と cut 位置）

- 本数 k = 上限の下での**最小連続分割数**（貪欲に詰めて数える — 連続分割では貪欲が最小）。
  NOTE: 連続順序 + 対の原子性の下では k が `ceil(総量 / 上限)` を上回る並びが理論上ある
  （例: 0.6GiB の対が 3 つ → 総量 1.8GiB でも 3 本）。ここは正直に受け入れる。
- k を固定して**均す**: 各 shard の目標 = 残量 ÷ 残 shard 数。単位（weight + scale の対）を
  跨がず、**suffix 実行可能性ガード**（残りの単位が残りの shard へ上限内で収まる位置でだけ
  cut を打つ）を掛けるので、全 shard が上限以下に収まることは構造的に保たれる。
- cut 位置の選好（層割り・MoE のエキスパート割り等）は**将来の書き手ポリシー拡張**で、
  読み手契約を 1 文字も変えずに足せる（ADR 0081 の扉 — 今は実装しない）。

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

#: 1 shard のデータ節の上限（1GiB）。**配布形の唯一の上限**で、固定定数・公開ノブにしない —
#: 不変条件であって呼び手の好みではない（緩めた資産は特定の環境でだけ読めなくなる）。
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


def _atomic_units(
    order: Sequence[str],
    payload_bytes: Mapping[str, int],
    companions: Mapping[str, str],
    limit: int,
) -> list[tuple[tuple[str, ...], int]]:
    """書き出し順を**跨げない単位**（単独テンソル / weight + companion scale の対）へ畳む。

    対の片方に到達した時点で相方ごと 1 単位にする（相方が順序上あとに居ても引き寄せる —
    読み手契約の 4）。単独で上限を超える単位は次の shard へ送っても同じなので fail loudly。
    """
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
    return units


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
    companions: Mapping[str, str],
    limit: int = SHARD_BYTE_LIMIT,
) -> list[tuple[str, ...]]:
    """書き出し順を shard 列へ割り付ける（規則はモジュール doc）。

    `order` は書き手が決めた**全体の**書き出し順、`payload_bytes` は名前 → データ節の
    バイト数、`companions` は原子対の**対称**写像（weight → scale と scale → weight の両方）。
    返すのは shard ごとの名前列で、**先頭は必ず空**（= グラフ shard・読み手契約の 1）。
    以降が weight shard で、並びは `order` の部分列 + 引き寄せた相方（shard の中の最終的な
    書き出し順は書き手が自分の規約で決め直す）。

    割り付けは 2 段。①**最小本数** k を貪欲で数える（`_suffix_shard_counts`）②k を固定して
    **均す** — 各 shard の目標を「残量 ÷ 残 shard 数」に取り、目標に届いたら cut を打つ。
    cut を打てるのは suffix 実行可能性ガード（`counts[i] <= 残 shard 数 - 1`）が立つ位置だけ
    なので、均しても最後の 1 本が上限を破ることはない。上限に届いてしまった場合は目標に
    関係なく cut（そこがちょうど貪欲の右端で、ガードは必ず立っている）。

    検算（規則の逐語 — テストと対で固定する）:

    - 3.2GiB → `[0.8GiB × 4]`。k = 4 を先に決めてから均すので端数 shard が出ない
      （旧・尾部スラック則の `[1GiB, 1GiB, 1.2GiB]` を置き換える形）。
    - 0.6GiB の単位 3 つ → `[0.6, 0.6, 0.6]`。総量 1.8GiB でも 2 本には詰め替えられない
      （連続順序 + 原子性の下では k > ceil(総量 / 上限) がありうる — モジュール doc）。

    MUST: weight shard を空で作らない。k は「全単位を上限内で覆う最小本数」なので、均しの
    ガードが立つ位置は常に残り単位が残っている位置になる。**テンソルが 1 本も無い**
    コンポーネントだけが例外で、そのときはグラフ shard 1 本（`[()]`）を返す。
    """
    if limit <= 0:
        raise ShardError(f"shard の上限 {limit} が正でない")
    units = _atomic_units(order, payload_bytes, companions, limit)
    groups: list[tuple[str, ...]] = [()]
    if units:
        sizes = [size for _, size in units]
        counts = _suffix_shard_counts(sizes, limit)
        total = counts[0]
        remaining = sum(sizes)
        index = 0
        for opened in range(total):
            left = total - opened
            # 残量を残 shard 数で割った目標（切り上げ — 端数は手前の shard が引き受ける）。
            target = -(-remaining // left)
            members: list[str] = []
            used = 0
            while index < len(units):
                unit, size = units[index]
                closable = bool(members) and counts[index] <= left - 1
                if closable and (used + size > limit or used >= target):
                    break
                members.extend(unit)
                used += size
                index += 1
            groups.append(tuple(members))
            remaining -= used
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
