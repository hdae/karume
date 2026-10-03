r"""Wan2.1 の前処理 `prompt_clean` を「表 + NFC」へ畳み、TS の鏡像を検証する
（ADR 0119 決定 2・案 b'）。

上流（diffusers 0.39.0 `pipelines/wan/pipeline_wan.py:78-93`）の `prompt_clean` は

    ftfy.fix_text → html.unescape × 2 → strip → `regex` の `\s+` → U+0020 → strip

で、ftfy 6.3.1 の `fix_text`（`ftfy/__init__.py:290-361`）は入力を「改行の直後」と
「1,000,000 コードポイントごと」で区切り、区切りごとに次の 1 周を**不動点まで**回す
（`fix_and_explain` — `:364-421`）:

    unescape_html → fix_encoding（ヒューリスティック）→ fix_c1_controls →
    fix_latin_ligatures → fix_character_width → uncurl_quotes → fix_line_breaks →
    fix_surrogates → remove_terminal_escapes → remove_control_chars → NFC

## 何を TS へ移し、何を拒むか

決定的な処理（表引きの 4 本・改行・端末エスケープ・NFC）だけを移す。表を持たない処理と
ヒューリスティックは、**それが効く入力を拒む**ことで no-op に固定する — 拒んだ入力では
上流の処理が何もしないので、残りの処理だけで上流と同じ文字列になる（黙って近似しない）:

  * **C1 制御文字（U+0080〜009F）** — `fix_c1_controls` は C1 にだけ効く。表の出力と
    NFC は C1 を作らない（{@link check_table_ranges}）ので、入口で 1 度見れば各周でも no-op。
  * **HTML の文字参照になりうる並び** — 各周の先頭で ftfy の `HTML_ENTITY_RE`
    （`&#?[0-9A-Za-z]{1,24};` — `ftfy/chardata.py:82`）が当たれば拒む（当たらなければ
    `unescape_html` は no-op）。`fix_text` の後で `html.unescape` の `_charref`
    （`html/__init__.py:118-120` — `&` の直後が `#数字`・`#x16 進`・
    `\t \n \f 空白 < & # ;` 以外の文字）が当たれば拒む（当たらなければ 2 回の unescape は
    no-op）。どちらも**正規表現の一致**で判定し、entity の名前表は持たない。帰結として
    `R&D` や `B&W` のように entity にならない並びも拒む（`a & b` と空白を挟めば通る）。
  * **mojibake の判定** — 各周の先頭で「ASCII でなく、`BADNESS_RE` が当たる」なら拒む
    （当たらなければ `fix_encoding` は no-op — `_fix_encoding_one_step_and_explain` の
    `:482`）。`BADNESS_RE` は構文木から JS の正規表現へ機械翻訳する（{@link badness_terms}）。
  * **孤立サロゲート** — TS の入口（`assertEncodableText`）が拒む。`fix_surrogates` は
    Python の str のサロゲート（対も孤立も）にだけ効き、JS の正当な文字列を UTF-8 で渡した
    str には現れない。
  * **UCD 16.0.0 で未割り当てのコードポイント** — ftfy の処理ではなく NFC の版差の防壁。
    TS の NFC は JS エンジンの `normalize` に任せる。割り当て済みの文字の NFC は Unicode の
    安定性方針で版をまたいで変わらないが、Python（UCD 16.0.0）で未割り当ての文字は、
    新しいエンジンが分解・結合クラスを持ちうる（語彙にも 60 個ある — U+0378 など）。
    そこで割れないよう拒む。

## 表の焼き方

判定の正本は Python 側（ftfy の表・`re` / `regex` の文字クラス・`unicodedata` 16.0.0）で、
TS で Unicode 判定を再実装しない（anima の `anima/text.py` と同じ規律）。全コードポイントを
実評価して閉区間表・写像表に畳み、{@link PromptCleanTables.to_json} が TS の読む形にする。

TS の実装（`packages/models/src/wan/text/prompt-clean.ts`）は {@link clean}（焼いた表だけを
引く Python の鏡像）の写経で、鏡像と上流 `prompt_clean` の一致は {@link verify_fuzz} と
{@link sweep_records} が網羅 + 乱択で検査する。外れたら何も書かない
（`wan.umt5_tokenizer` の main）。

MUST: ftfy / regex / diffusers は関数の中で import する（`wan` グループは既定の sync に
入らない — `tests/test_optional_group_imports.py`）。
"""

from __future__ import annotations

import bisect
import hashlib
import html
import re
import unicodedata
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any

Ranges = list[list[int]]

#: サロゲートを除く全コードポイント（Python の str に載る評価範囲）。
ALL_CODEPOINTS = [cp for cp in range(0x110000) if not 0xD800 <= cp <= 0xDFFF]

#: 正本の版（表はこの版の実挙動から焼く — 版が違えば焼かない）。
FTFY_VERSION = "6.3.1"
UNIDATA_VERSION = "16.0.0"

#: ftfy の `TextFixerConfig.max_decode_length`（区切りの最大コードポイント数）。
MAX_DECODE_LENGTH = 1_000_000

#: C1 制御文字（ADR 0119 決定 2 の拒否の範囲）。
C1_FIRST, C1_LAST = 0x80, 0x9F

#: 不動点の周回の上限（TS と同じ値）。上流の表では 2 周で止まる（{@link verify_fuzz} が
#: 観測の最大を記録する）。届かないのは表が壊れているときだけなので、TS は素の Error で落とす。
MAX_FIX_ROUNDS = 8

#: ftfy の `HTML_ENTITY_RE`（`ftfy/chardata.py:82`）。各周の先頭で当たれば拒む。
FTFY_ENTITY_PATTERN = r"&#?[0-9A-Za-z]{1,24};"

#: `html.unescape` の `_charref`（Python 3.14 `html/__init__.py:118-120`）。
HTML_CHARREF_PATTERN = r"&(#[0-9]+;?|#[xX][0-9a-fA-F]+;?|[^\t\n\f <&#;]{1,32};?)"

#: {@link HTML_CHARREF_PATTERN} が**当たるかどうか**だけを見る同値形（TS はこの形を持つ）。
#: `+` / `{1,32}` / `;?` は 1 文字目が当たれば必ず満たせるので、存在判定には 1 文字目だけでよい。
HTML_CHARREF_PROBE = r"&(?:#[0-9]|#[xX][0-9a-fA-F]|[^\t\n\f <&#;])"

#: ftfy の `ANSI_RE`（`ftfy/fixes.py:142`）。`\d` は Unicode の Nd（`re` の str パターン）。
ANSI_PATTERN = "\033\\[((?:\\d|;)*)([a-zA-Z])"


class PromptRejectedError(ValueError):
    """鏡像が受けない入力（TS の `PromptCleanError` と同じ理由の語彙）。"""

    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(f"{reason}: {detail}")
        self.reason = reason


# ---------------------------------------------------------------------------
# 閉区間表
# ---------------------------------------------------------------------------


def to_ranges(codepoints: Iterable[int]) -> Ranges:
    """昇順の閉区間リストへ畳む。"""
    out: Ranges = []
    for cp in sorted(set(codepoints)):
        if out and out[-1][1] == cp - 1:
            out[-1][1] = cp
        else:
            out.append([cp, cp])
    return out


def ranges_where(predicate: Callable[[str], bool], *, surrogates: bool = False) -> Ranges:
    """述語が真になるコードポイントの閉区間表（正本に 1 文字ずつ聞いて畳む）。"""
    domain = range(0x110000) if surrogates else ALL_CODEPOINTS
    return to_ranges(cp for cp in domain if predicate(chr(cp)))


class RangeSet:
    """閉区間表の所属判定（TS の `inCodeRanges` の鏡像 — 二分探索）。"""

    def __init__(self, ranges: Ranges) -> None:
        self.ranges = ranges
        self._starts = [start for start, _ in ranges]

    def __contains__(self, cp: int) -> bool:
        index = bisect.bisect_right(self._starts, cp) - 1
        return index >= 0 and cp <= self.ranges[index][1]


def _complement(ranges: Ranges) -> Ranges:
    out: Ranges = []
    next_cp = 0
    for start, end in ranges:
        if start > next_cp:
            out.append([next_cp, start - 1])
        next_cp = end + 1
    if next_cp <= 0x10FFFF:
        out.append([next_cp, 0x10FFFF])
    return out


def _union(*parts: Ranges) -> Ranges:
    merged: Ranges = []
    for start, end in sorted([list(r) for part in parts for r in part]):
        if merged and start <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


# ---------------------------------------------------------------------------
# BADNESS_RE の翻訳（構文木 → 中間表現 → JS / Python の正規表現）
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CharClass:
    r"""1 文字の集合（字面の文字・範囲・`\s` / `\w`・否定・`.` を全てここへ畳む）。"""

    ranges: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class OptionalChar:
    """直前の 1 文字の省略可（`?`）。"""

    item: CharClass


@dataclass(frozen=True)
class LineStart:
    """文字列の先頭（MULTILINE なしの `^`）。"""


Term = CharClass | OptionalChar | LineStart


def _category_ranges(category: Any) -> Ranges:
    r"""`re` の str パターンの文字カテゴリを全コードポイントで実評価する（`\s` / `\w`）。"""
    import re._constants as sre

    pattern = {sre.CATEGORY_SPACE: r"\s", sre.CATEGORY_WORD: r"\w"}.get(category)
    if pattern is None:
        raise ValueError(f"BADNESS_RE に想定外の文字カテゴリ {category} がある")
    compiled = re.compile(pattern)
    return ranges_where(lambda ch: compiled.fullmatch(ch) is not None, surrogates=True)


def _class_ranges(items: Sequence[tuple[Any, Any]]) -> Ranges:
    import re._constants as sre

    negate = False
    parts: list[Ranges] = []
    for op, av in items:
        if op is sre.NEGATE:
            negate = True
        elif op is sre.LITERAL:
            parts.append([[av, av]])
        elif op is sre.RANGE:
            parts.append([[av[0], av[1]]])
        elif op is sre.CATEGORY:
            parts.append(_category_ranges(av))
        else:
            raise ValueError(f"BADNESS_RE の文字クラスに想定外の要素 {op} がある")
    ranges = _union(*parts)
    return _complement(ranges) if negate else ranges


def _term(op: Any, av: Any) -> Term:
    import re._constants as sre

    if op is sre.LITERAL:
        return CharClass(((av, av),))
    if op is sre.IN:
        return CharClass(tuple((a, b) for a, b in _class_ranges(av)))
    if op is sre.ANY:
        # MUST: Python の `.`（DOTALL なし）は `\n` だけを除く。JS の `.` は `\r` と
        # U+2028 / 2029 も除くので、素の `.` へ訳すと改行の周りだけ判定が割れる。集合として
        # 書き下す。
        return CharClass(tuple((a, b) for a, b in _complement([[0x0A, 0x0A]])))
    if op is sre.MAX_REPEAT:
        low, high, inner = av
        if (low, high) != (0, 1) or len(inner) != 1:
            raise ValueError(f"BADNESS_RE に `?` 以外の繰り返し {low},{high} がある")
        item = _term(*inner[0])
        if not isinstance(item, CharClass):
            raise ValueError("BADNESS_RE の `?` が 1 文字の集合に掛かっていない")
        return OptionalChar(item)
    if op is sre.AT and av is sre.AT_BEGINNING:
        return LineStart()
    raise ValueError(f"BADNESS_RE に想定外の要素 {op} {av!r} がある")


def badness_terms() -> list[list[Term]]:
    r"""`ftfy.badness.BADNESS_RE` の構文木を「選択肢 × 項の列」の中間表現へ畳む。

    TS へ正規表現の**ソース文字列**を写すと、`re.VERBOSE` の空白・`\w` / `\s` の Unicode の範囲・
    `.` の改行の扱いが JS と黙って割れる。構文木から集合を実評価して書き下せば、JS 側は
    `\u{…}` の字面と範囲だけを見る（エンジンの Unicode の版に依らない）。
    """
    import re._constants as sre
    import re._parser as sre_parse

    from ftfy.badness import BADNESS_RE

    tree = sre_parse.parse(BADNESS_RE.pattern, BADNESS_RE.flags)
    if len(tree) != 1 or tree[0][0] is not sre.BRANCH:
        raise ValueError("BADNESS_RE の最上位が 1 つの選択（BRANCH）でない")
    _, branches = tree[0][1]
    return [[_term(op, av) for op, av in branch] for branch in branches]


def badness_literals() -> list[int]:
    r"""`BADNESS_RE` の字面に現れる非 ASCII の文字（`\s` / `\w`・否定・`.` 由来は含めない）。

    fuzz の池の境界（調査 §3.3 の「449 文字」）。字面の集合だけを数えるので、カテゴリ由来の
    広い集合（`\w` など）はここに入らない。
    """
    import re._constants as sre
    import re._parser as sre_parse

    from ftfy.badness import BADNESS_RE

    literal: set[int] = set()

    def walk(node: Any) -> None:
        for op, av in node:
            if op is sre.BRANCH:
                for branch in av[1]:
                    walk(branch)
            elif op is sre.MAX_REPEAT:
                walk(av[2])
            elif op is sre.LITERAL:
                literal.add(av)
            elif op is sre.IN:
                for item_op, item_av in av:
                    if item_op is sre.LITERAL:
                        literal.add(item_av)
                    elif item_op is sre.RANGE:
                        literal.update(range(item_av[0], item_av[1] + 1))

    walk(sre_parse.parse(BADNESS_RE.pattern, BADNESS_RE.flags))
    return sorted(cp for cp in literal if cp > 0x7F)


def _emit(terms: Sequence[Sequence[Term]], escape: Callable[[int], str]) -> str:
    def atom(item: CharClass) -> str:
        if len(item.ranges) == 1 and item.ranges[0][0] == item.ranges[0][1]:
            return escape(item.ranges[0][0])
        body = "".join(
            escape(start) if start == end else f"{escape(start)}-{escape(end)}"
            for start, end in item.ranges
        )
        return f"[{body}]"

    def term(item: Term) -> str:
        if isinstance(item, CharClass):
            return atom(item)
        if isinstance(item, OptionalChar):
            return atom(item.item) + "?"
        return "^"

    return "|".join("".join(term(item) for item in branch) for branch in terms)


def emit_js(terms: Sequence[Sequence[Term]]) -> str:
    """JS の正規表現ソース（`u` フラグで使う — 全ての文字を `\\u{…}` で書く）。"""
    return _emit(terms, lambda cp: f"\\u{{{cp:X}}}")


def emit_python(terms: Sequence[Sequence[Term]]) -> str:
    """同じ中間表現の Python 版（JS 版と同じ形で、エスケープの綴りだけが違う）。

    pytest がこれを `BADNESS_RE` と突き合わせる — 中間表現への畳み込みの同値を Python の中で
    確かめ、JS 側は綴りの写像（`\\u{…}`）だけを TS の parity テストが縛る。
    """
    return _emit(terms, lambda cp: f"\\U{cp:08x}")


# ---------------------------------------------------------------------------
# 表
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PromptCleanTables:
    """TS が引く表一式（{@link to_json} が資産・フィクスチャの `promptClean` になる）。"""

    #: `fix_latin_ligatures`（`ftfy/chardata.py:207-230` の LIGATURES）。
    ligatures: dict[int, str]
    #: `fix_character_width`（WIDTH_MAP — 全角・半角形の NFKC と U+3000 → U+0020）。
    width: dict[int, str]
    #: `uncurl_quotes` の SINGLE_QUOTE_RE / DOUBLE_QUOTE_RE が当たる文字。
    single_quotes: Ranges
    double_quotes: Ranges
    #: `remove_control_chars` が消す文字（CONTROL_CHARS）。
    control_chars: Ranges
    #: `remove_terminal_escapes` の `\d`（`re` の Unicode の Nd）。
    ansi_digits: Ranges
    #: UCD 16.0.0 で割り当て済み（`unicodedata.category != "Cn"`）。
    assigned: Ranges
    #: `str.strip()` が落とす文字（`str.isspace`）。
    strip_space: Ranges
    #: `whitespace_clean` の `regex` の `\s`。
    collapse_space: Ranges
    #: `BADNESS_RE` の中間表現。
    badness: list[list[Term]]

    def to_json(self) -> dict[str, Any]:
        return {
            "ligatures": [[cp, text] for cp, text in sorted(self.ligatures.items())],
            "width": [[cp, text] for cp, text in sorted(self.width.items())],
            "singleQuotes": self.single_quotes,
            "doubleQuotes": self.double_quotes,
            "controlChars": self.control_chars,
            "ansiDigits": self.ansi_digits,
            "assigned": self.assigned,
            "stripSpace": self.strip_space,
            "collapseSpace": self.collapse_space,
            # JS の正規表現ソース（`new RegExp(source, "u")`）。
            "badness": emit_js(self.badness),
        }


def check_versions() -> dict[str, str]:
    """正本の版を確かめる（違えば焼かない — 表は版の実挙動そのもの）。"""
    import sys

    import ftfy
    import regex

    if ftfy.__version__ != FTFY_VERSION:
        raise ValueError(f"ftfy {ftfy.__version__} — 表は {FTFY_VERSION} の実挙動から焼く")
    if unicodedata.unidata_version != UNIDATA_VERSION:
        raise ValueError(
            f"unicodedata {unicodedata.unidata_version} — NFC の正本は UCD {UNIDATA_VERSION}"
        )
    return {
        "ftfy": ftfy.__version__,
        "python": sys.version.split()[0],
        "regex": regex.__version__,
        "unicodedata": unicodedata.unidata_version,
    }


def check_upstream_shape() -> None:
    """鏡像が前提にしている上流の綴りを固定する（版を上げて変われば焼かない）。

    表に落とせない部分（正規表現の形・既定の設定）を字面で縛る。振る舞いの同値は
    {@link verify_fuzz} と {@link sweep_records} が別に見る。
    """
    from ftfy import TextFixerConfig, chardata, fixes

    if chardata.HTML_ENTITY_RE.pattern != FTFY_ENTITY_PATTERN:
        raise ValueError(f"ftfy の HTML_ENTITY_RE が変わった: {chardata.HTML_ENTITY_RE.pattern!r}")
    if html._charref.pattern != HTML_CHARREF_PATTERN:  # type: ignore[attr-defined]
        raise ValueError(f"html._charref が変わった: {html._charref.pattern!r}")  # type: ignore[attr-defined]
    if fixes.ANSI_RE.pattern != ANSI_PATTERN:
        raise ValueError(f"ftfy の ANSI_RE が変わった: {fixes.ANSI_RE.pattern!r}")
    config = TextFixerConfig()
    expected = {
        "unescape_html": "auto",
        "remove_terminal_escapes": True,
        "fix_encoding": True,
        "fix_c1_controls": True,
        "fix_latin_ligatures": True,
        "fix_character_width": True,
        "uncurl_quotes": True,
        "fix_line_breaks": True,
        "fix_surrogates": True,
        "remove_control_chars": True,
        "normalization": "NFC",
        "max_decode_length": MAX_DECODE_LENGTH,
    }
    actual = {key: getattr(config, key) for key in expected}
    if actual != expected:
        raise ValueError(f"ftfy の既定の設定が変わった: {actual}")


def build_tables() -> PromptCleanTables:
    """ftfy 6.3.1・`re` / `regex`・`unicodedata` 16.0.0 から表を焼く。"""
    import regex
    from ftfy import chardata

    check_versions()
    check_upstream_shape()
    re_digit = re.compile(r"\d")
    regex_space = regex.compile(r"\s")
    tables = PromptCleanTables(
        ligatures=dict(chardata.LIGATURES),
        width={cp: text for cp, text in chardata.WIDTH_MAP.items()},
        single_quotes=ranges_where(lambda ch: chardata.SINGLE_QUOTE_RE.fullmatch(ch) is not None),
        double_quotes=ranges_where(lambda ch: chardata.DOUBLE_QUOTE_RE.fullmatch(ch) is not None),
        control_chars=to_ranges(chardata.CONTROL_CHARS),
        ansi_digits=ranges_where(lambda ch: re_digit.fullmatch(ch) is not None),
        assigned=ranges_where(lambda ch: unicodedata.category(ch) != "Cn"),
        strip_space=ranges_where(str.isspace),
        collapse_space=ranges_where(lambda ch: regex_space.fullmatch(ch) is not None),
        badness=badness_terms(),
    )
    check_table_ranges(tables)
    return tables


def check_table_ranges(tables: PromptCleanTables) -> None:
    """表の値域の前提（入口で 1 度見れば各周でも成り立つ、の根拠）を検査する。

    表引きの出力が C1・未割り当て・サロゲートを作れば、入口の検査だけでは各周の no-op が
    言えなくなる。引用符の 2 集合が交わったり出力が自分に当たれば、2 本の置換を 1 回の写像で
    書いた TS と順序が割れる。
    """
    assigned = RangeSet(tables.assigned)
    outputs = "".join(tables.ligatures.values()) + "".join(tables.width.values()) + "'\"\n"
    for ch in outputs:
        cp = ord(ch)
        if C1_FIRST <= cp <= C1_LAST or 0xD800 <= cp <= 0xDFFF or cp not in assigned:
            raise ValueError(f"表の出力 U+{cp:04X} が C1 / サロゲート / 未割り当て")
    single, double = RangeSet(tables.single_quotes), RangeSet(tables.double_quotes)
    if any(cp in double for start, end in tables.single_quotes for cp in range(start, end + 1)):
        raise ValueError("SINGLE_QUOTE_RE と DOUBLE_QUOTE_RE の集合が交わる")
    if ord("'") in single or ord("'") in double or ord('"') in single or ord('"') in double:
        raise ValueError("引用符の置換先が引用符の集合に入っている")


# ---------------------------------------------------------------------------
# 鏡像（TS 実装の写経元 — 焼いた表だけを引く）
# ---------------------------------------------------------------------------


class CompiledTables:
    """{@link PromptCleanTables} を引きやすい形にしたもの。

    TS の `parsePromptCleanTables` の鏡像。
    """

    def __init__(self, tables: PromptCleanTables) -> None:
        self.tables = tables
        self.ligatures = {cp: text for cp, text in tables.ligatures.items()}
        self.width = {cp: text for cp, text in tables.width.items()}
        self.single = RangeSet(tables.single_quotes)
        self.double = RangeSet(tables.double_quotes)
        self.control = RangeSet(tables.control_chars)
        self.digits = RangeSet(tables.ansi_digits)
        self.assigned = RangeSet(tables.assigned)
        self.strip_space = RangeSet(tables.strip_space)
        self.collapse_space = RangeSet(tables.collapse_space)
        self.badness = re.compile(emit_python(tables.badness))
        self.ftfy_entity = re.compile(FTFY_ENTITY_PATTERN)
        self.html_charref = re.compile(HTML_CHARREF_PROBE)


def split_segments(text: str) -> list[str]:
    """`fix_text` の区切り（改行の直後で切る・1 区切りは最大 {@link MAX_DECODE_LENGTH}）。"""
    out: list[str] = []
    pos = 0
    while pos < len(text):
        textbreak = text.find("\n", pos) + 1
        if textbreak == 0:
            textbreak = len(text)
        textbreak = min(textbreak, pos + MAX_DECODE_LENGTH)
        out.append(text[pos:textbreak])
        pos = textbreak
    return out


def _translate(text: str, mapping: dict[int, str]) -> str:
    return "".join(mapping.get(ord(ch), ch) for ch in text)


def _uncurl_quotes(compiled: CompiledTables, text: str) -> str:
    out: list[str] = []
    for ch in text:
        cp = ord(ch)
        out.append('"' if cp in compiled.double else "'" if cp in compiled.single else ch)
    return "".join(out)


def _fix_line_breaks(text: str) -> str:
    return (
        text.replace("\r\n", "\n")
        .replace("\r", "\n")
        .replace(" ", "\n")
        .replace(" ", "\n")
        .replace("\u0085", "\n")
    )


def _remove_terminal_escapes(compiled: CompiledTables, text: str) -> str:
    """`ESC [ (\\d|;)* 英字` を消す（`re.sub` と同じく左から・重ならずに）。"""
    out: list[str] = []
    i, n = 0, len(text)
    while i < n:
        if text[i] == "\x1b" and i + 1 < n and text[i + 1] == "[":
            j = i + 2
            while j < n and (text[j] == ";" or ord(text[j]) in compiled.digits):
                j += 1
            if j < n and ("a" <= text[j] <= "z" or "A" <= text[j] <= "Z"):
                i = j + 1
                continue
        out.append(text[i])
        i += 1
    return "".join(out)


def _remove_control_chars(compiled: CompiledTables, text: str) -> str:
    return "".join(ch for ch in text if ord(ch) not in compiled.control)


def _strip(compiled: CompiledTables, text: str) -> str:
    start, end = 0, len(text)
    while start < end and ord(text[start]) in compiled.strip_space:
        start += 1
    while end > start and ord(text[end - 1]) in compiled.strip_space:
        end -= 1
    return text[start:end]


def _collapse(compiled: CompiledTables, text: str) -> str:
    out: list[str] = []
    in_space = False
    for ch in text:
        if ord(ch) in compiled.collapse_space:
            if not in_space:
                out.append(" ")
            in_space = True
        else:
            out.append(ch)
            in_space = False
    return "".join(out)


def fix_segment(compiled: CompiledTables, segment: str, rounds: Counter[int] | None = None) -> str:
    """1 区切りを不動点まで回す。

    `fix_and_explain` の 1 周から、拒んで no-op にした処理を除いたもの。
    """
    text = segment
    for round_index in range(1, MAX_FIX_ROUNDS + 1):
        before = text
        entity = compiled.ftfy_entity.search(text)
        if entity is not None:
            raise PromptRejectedError("entity", entity.group(0))
        if not text.isascii():
            bad = compiled.badness.search(text)
            if bad is not None:
                raise PromptRejectedError("mojibake", bad.group(0))
        text = _translate(text, compiled.ligatures)
        text = _translate(text, compiled.width)
        text = _uncurl_quotes(compiled, text)
        text = _fix_line_breaks(text)
        text = _remove_terminal_escapes(compiled, text)
        text = _remove_control_chars(compiled, text)
        text = unicodedata.normalize("NFC", text)
        if text == before:
            if rounds is not None:
                rounds[round_index] += 1
            return text
    raise RuntimeError(f"{MAX_FIX_ROUNDS} 周で不動点に届かない（表が壊れている）: {segment!r}")


def clean(compiled: CompiledTables, text: str, rounds: Counter[int] | None = None) -> str:
    """`prompt_clean` の鏡像。受けない入力は {@link PromptRejectedError}。"""
    for ch in text:
        cp = ord(ch)
        if 0xD800 <= cp <= 0xDFFF:
            raise PromptRejectedError("surrogate", f"U+{cp:04X}")
        if cp not in compiled.assigned:
            raise PromptRejectedError("unassigned", f"U+{cp:04X}")
        if C1_FIRST <= cp <= C1_LAST:
            raise PromptRejectedError("c1", f"U+{cp:04X}")
    fixed = "".join(fix_segment(compiled, segment, rounds) for segment in split_segments(text))
    charref = compiled.html_charref.search(fixed)
    if charref is not None:
        raise PromptRejectedError("entity", charref.group(0))
    return _strip(compiled, _collapse(compiled, _strip(compiled, fixed)))


def upstream_prompt_clean() -> Callable[[str], str]:
    """上流の `prompt_clean`（ftfy が入っていることを確かめてから返す）。"""
    from diffusers.pipelines.wan.pipeline_wan import prompt_clean
    from diffusers.utils import is_ftfy_available

    check_versions()
    if not is_ftfy_available():
        raise ValueError("diffusers が ftfy を見つけない — prompt_clean が fix_text を飛ばす")
    return prompt_clean


def record(
    compiled: CompiledTables,
    truth: Callable[[str], str],
    text: str,
    rounds: Counter[int] | None = None,
) -> str:
    """1 入力の決着（受理なら `A` + 上流の出力・拒否なら `R` + 理由）。

    MUST: 受理した入力では鏡像と上流の一致を必ず確かめてから**上流の出力**を記録する — 期待値は
    常に上流から採り、鏡像は「どれを拒むか」だけを決める。
    """
    try:
        mirrored = clean(compiled, text, rounds)
    except PromptRejectedError as rejected:
        return "R" + rejected.reason
    expected = truth(text)
    if mirrored != expected:
        raise ValueError(
            f"鏡像が上流の prompt_clean と違う\n  入力={[hex(ord(c)) for c in text]}\n"
            f"  上流={expected!r}\n  鏡像={mirrored!r}"
        )
    return "A" + expected


# ---------------------------------------------------------------------------
# 乱択（TS と同じ列を再生できる PRNG）と池
# ---------------------------------------------------------------------------


class XorShift32:
    """32 ビットの xorshift（Marsaglia 13/17/5）。TS の `xorshift32` と同じ列を出す。"""

    def __init__(self, seed: int) -> None:
        if not 0 < seed < 2**32:
            raise ValueError(f"xorshift32 の seed は 1..2^32-1（{seed}）")
        self.state = seed

    def next(self) -> int:
        x = self.state
        x ^= (x << 13) & 0xFFFFFFFF
        x ^= x >> 17
        x ^= (x << 5) & 0xFFFFFFFF
        self.state = x
        return x


def expand(ranges: Ranges) -> list[int]:
    return [cp for start, end in ranges for cp in range(start, end + 1)]


def fuzz_strings(pool: Sequence[int], seed: int, count: int, max_length: int) -> Iterator[str]:
    """長さ 1..max_length の乱択列（長さ → 各文字の順に PRNG を引く — TS と同じ手順）。"""
    rng = XorShift32(seed)
    for _ in range(count):
        length = 1 + rng.next() % max_length
        yield "".join(chr(pool[rng.next() % len(pool)]) for _ in range(length))


#: 安全な池の素材（調査 §3.3 の池と同じ構成 — 普通のプロンプトに出る文字と、決定的な処理が
#: 効く文字）。ここから `BADNESS_RE` の字面の文字・`&`・未割り当てを除く。
SAFE_POOL_BLOCKS: tuple[tuple[int, int, str], ...] = (
    (0x0000, 0x007F, "ASCII（制御文字・ESC を含む）"),
    (0x00A0, 0x017F, "Latin-1 補助・Latin 拡張 A（ĳ・ŉ を含む）"),
    (0x01C4, 0x01CC, "Latin の二重字（fix_latin_ligatures）"),
    (0x02BC, 0x02BC, "修飾字のアポストロフィ（uncurl_quotes）"),
    (0x0300, 0x036F, "結合文字"),
    (0x0370, 0x04FF, "ギリシャ・キリル"),
    (0x0600, 0x06FF, "アラビア"),
    (0x0900, 0x097F, "デーヴァナーガリー"),
    (0x0E00, 0x0E7F, "タイ"),
    (0x1100, 0x11FF, "ハングル字母"),
    (0x2000, 0x206F, "一般句読点（曲がった引用符・U+2028 / 2029・206A-206F）"),
    (0x3000, 0x30FF, "CJK 記号・仮名"),
    (0x4E00, 0x51FF, "CJK 統合漢字（先頭 1,024）"),
    (0xAC00, 0xADFF, "ハングル音節（先頭 512）"),
    (0xFB00, 0xFB06, "Latin の合字"),
    (0xFE00, 0xFE0F, "異体字セレクタ"),
    (0xFEFF, 0xFEFF, "BOM"),
    (0xFF00, 0xFFEF, "全角・半角形"),
    (0xFFF9, 0xFFFC, "注釈・置換オブジェクト"),
    (0x1F1E6, 0x1F1FF, "地域表示子"),
    (0x1F300, 0x1F64F, "絵文字"),
)

#: entity の門を叩く池の追加分（`&` の周りに来うる空白・全角の記号）。
ENTITY_POOL_EXTRA = (0x09, 0x0A, 0x0C, 0x0D, 0xA0, 0x3000, 0xFF03, 0xFF06, 0xFF1B)


def fuzz_pools(tables: PromptCleanTables) -> dict[str, Ranges]:
    """fuzz の池（TS はフィクスチャの区間表を展開して同じ列を引く）。

    安全な池は「`&` と `BADNESS_RE` の字面の文字を含まない」だけでなく、表引きの出力にも
    それらを出さない文字に絞る（全角の `＆` `￠` は `fix_character_width` で `&` `¢` になり、
    次の周で entity / mojibake の門に当たる — 安全な池の意味が崩れる）。
    """
    assigned = RangeSet(tables.assigned)
    literals = set(badness_literals())
    unsafe = {*literals, ord("&")}

    def maps_to_unsafe(cp: int) -> bool:
        output = tables.width.get(cp, "") + tables.ligatures.get(cp, "")
        return any(ord(ch) in unsafe for ch in output)

    blocks = [cp for start, end, _ in SAFE_POOL_BLOCKS for cp in range(start, end + 1)]
    safe = [cp for cp in blocks if cp in assigned and cp not in unsafe and not maps_to_unsafe(cp)]
    return {
        "safe": to_ranges(safe),
        "full": to_ranges([*safe, *literals, ord("&")]),
        "entity": to_ranges([*range(0x20, 0x7F), *ENTITY_POOL_EXTRA]),
    }


# ---------------------------------------------------------------------------
# フィクスチャ（TS が同じ列を再生して突き合わせる — 期待値は digest で持つ）
# ---------------------------------------------------------------------------

#: 区切り（記録の中には現れない — 受理の出力は制御文字を持たず、理由は ASCII の語）。
RECORD_SEPARATOR = "\x00"


def digest(records: Iterable[str]) -> str:
    """記録の列の SHA-256（各記録の後に {@link RECORD_SEPARATOR} を置いた UTF-8）。"""
    hasher = hashlib.sha256()
    for item in records:
        hasher.update((item + RECORD_SEPARATOR).encode("utf-8"))
    return hasher.hexdigest()


#: 塊ごとの digest の桁数（16 進 16 桁 = 64 ビット）。回帰の検出には十分で、フィクスチャを
#: 塊の数だけ太らせない。
BLOCK_DIGEST_HEX = 16


def chunked_digests(records: Sequence[str], chunk: int) -> list[str]:
    """`chunk` 件ごとの digest（外れたときに TS がどの塊かを言えるように分ける）。"""
    return [
        digest(records[i : i + chunk])[:BLOCK_DIGEST_HEX] for i in range(0, len(records), chunk)
    ]


@dataclass(frozen=True)
class FuzzRun:
    """乱択 1 本の定義（TS はこの定義とフィクスチャの池から同じ列を作る）。"""

    name: str
    pool: str
    seed: int
    count: int
    max_length: int = 40
    chunk: int = 1000


#: 前処理の鏡像の fuzz（調査 §3.3 の池と本数以上 — 安全な池 200,000 本・全池 50,000 本）。
CLEAN_RUNS = (
    FuzzRun("safe", "safe", seed=1, count=200_000),
    FuzzRun("full", "full", seed=2, count=50_000),
    FuzzRun("entity", "entity", seed=3, count=20_000),
)

#: `BADNESS_RE` の翻訳の fuzz（449 文字を混ぜた池 — 判定そのものを `is_bad` と比べる）。
BADNESS_RUNS = (FuzzRun("full", "full", seed=4, count=50_000),)

#: `BADNESS_RE` の Unicode 依存の構文（`\w`・`\s`・`.`・`[^…]`・`^`）を全コードポイントで叩く文脈。
BADNESS_PROBES: tuple[tuple[str, str], ...] = (
    ("word", "à«{}"),
    ("space", "{}À€"),
    ("space-optional", "a{}Ã "),
    ("any", "ГўВЂВ{}a"),
    ("negated", "œ{}"),
    ("line-start", "{}Ã "),
)

#: 前処理の鏡像を全コードポイントで叩く文脈（端の strip と中の畳み込みの両方）。
CLEAN_SWEEPS: tuple[tuple[str, str], ...] = (("alone", "{}"), ("inner", "a{}b"))

#: NFC の版差の掃引（UCD 16.0.0 で割り当て済みのコードポイントだけ — 未割り当ては入口で拒む）。
#: 並べ替え（ccc 1 / 230）・合成・ハングル・二重の文脈を混ぜる。
NFC_SWEEPS: tuple[tuple[str, str], ...] = (
    ("alone", "{}"),
    ("before-acute", "a{}́"),
    ("after-acute", "á{}"),
    ("before-overlay", "a{}̴"),
    ("double", "{}{}"),
    ("hangul", "ᄀ{}ᅡ"),
    ("between-marks", "ẽ{}̼"),
    ("double-acute", "{}{}́"),
)

#: 全コードポイントの掃引を区切る塊（外れたブロックを TS が名指しできるように）。
SWEEP_BLOCK = 0x1000


def sweep_blocks(
    domain: Sequence[int], record_of: Callable[[int], str], block: int = SWEEP_BLOCK
) -> list[str]:
    """`domain` を `block` 単位の塊に分け、塊ごとの digest を返す（空の塊も 1 本）。"""
    buckets: list[list[str]] = [[] for _ in range(0x110000 // block)]
    for cp in domain:
        buckets[cp // block].append(record_of(cp))
    return [digest(bucket)[:BLOCK_DIGEST_HEX] for bucket in buckets]


def sweep_records(
    compiled: CompiledTables, truth: Callable[[str], str], template: str
) -> list[str]:
    """前処理の全コードポイント掃引（`template` の `{}` に 1 文字を入れる）。"""
    return sweep_blocks(
        ALL_CODEPOINTS, lambda cp: record(compiled, truth, template.replace("{}", chr(cp)))
    )


def verify_fuzz(
    compiled: CompiledTables, truth: Callable[[str], str], run: FuzzRun, pool: Sequence[int]
) -> dict[str, Any]:
    """乱択 1 本を回し、塊ごとの digest と決着の内訳を返す。

    鏡像と上流が食い違えば {@link record} が落ちる。
    """
    rounds: Counter[int] = Counter()
    records: list[str] = []
    for text in fuzz_strings(pool, run.seed, run.count, run.max_length):
        records.append(record(compiled, truth, text, rounds))
    outcomes = Counter("accepted" if item.startswith("A") else item[1:] for item in records)
    return {
        "name": run.name,
        "pool": run.pool,
        "seed": run.seed,
        "count": run.count,
        "maxLength": run.max_length,
        "chunk": run.chunk,
        "chunks": chunked_digests(records, run.chunk),
        "outcomes": dict(sorted(outcomes.items())),
        "maxRounds": max(rounds) if rounds else 0,
    }


def badness_bits(pattern: re.Pattern[str], texts: Iterable[str]) -> str:
    """各入力に `search` が当たるかを `0` / `1` の列にしたもの。"""
    return "".join("1" if pattern.search(text) else "0" for text in texts)


def bits_digest(bits: str) -> str:
    return hashlib.sha256(bits.encode("ascii")).hexdigest()
