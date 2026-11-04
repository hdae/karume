r"""Folds the Wan2.1 preprocessing `prompt_clean` into "tables + NFC" and verifies the TS mirror
(ADR 0119 decision 2, option b').

Upstream's `prompt_clean` (diffusers 0.39.0 `pipelines/wan/pipeline_wan.py:78-93`) is

    ftfy.fix_text → html.unescape × 2 → strip → `regex` `\s+` → U+0020 → strip

and `fix_text` of ftfy 6.3.1 (`ftfy/__init__.py:290-361`) splits the input "right after each line
break" and "every 1,000,000 code points", and runs the following round on each segment **until a
fixed point** (`fix_and_explain` — `:364-421`):

    unescape_html → fix_encoding (heuristic) → fix_c1_controls →
    fix_latin_ligatures → fix_character_width → uncurl_quotes → fix_line_breaks →
    fix_surrogates → remove_terminal_escapes → remove_control_chars → NFC

## What moves to TS and what is rejected

Only the deterministic steps (the 4 table lookups, line breaks, terminal escapes, NFC) move. The
steps without a table and the heuristics are pinned to a no-op by **rejecting the inputs they act
on** — on a rejected input the upstream step does nothing, so the remaining steps alone produce the
same string as upstream (no silent approximation):

  * **C1 control characters (U+0080–009F)** — `fix_c1_controls` acts only on C1. The table outputs
    and NFC never produce C1 ({@link check_table_ranges}), so checking once at the entrance makes
    it a no-op in every round.
  * **Sequences that could be HTML character references** — at the start of each round, reject if
    ftfy's `HTML_ENTITY_RE` (`&#?[0-9A-Za-z]{1,24};` — `ftfy/chardata.py:82`) matches (if it does
    not, `unescape_html` is a no-op). After `fix_text`, reject if `_charref` of `html.unescape`
    (`html/__init__.py:118-120` — `&` followed by `#digits`, `#x hex`, or a character other than
    `\t \n \f space < & # ;`) matches (if it does not, the two unescapes are no-ops). Both are
    decided by **a regular-expression match**, without a table of entity names. As a consequence,
    sequences that are not entities, such as `R&D` or `B&W`, are rejected too (`a & b` with spaces
    passes).
  * **Mojibake detection** — at the start of each round, reject if "not ASCII and `BADNESS_RE`
    matches" (if not, `fix_encoding` is a no-op — `:482` of `_fix_encoding_one_step_and_explain`).
    `BADNESS_RE` is machine-translated from its syntax tree into a JS regular expression
    ({@link badness_terms}).
  * **Lone surrogates** — the TS entrance (`assertEncodableText`) rejects them. `fix_surrogates`
    acts only on surrogates in a Python str (paired or lone), which never appear in a str made from
    a valid JS string passed as UTF-8.
  * **Code points unassigned in UCD 16.0.0** — not an ftfy step but a guard against NFC version
    differences. The TS NFC is left to the JS engine's `normalize`. The NFC of assigned characters
    does not change across versions under Unicode's stability policy, but a character unassigned in
    Python (UCD 16.0.0) may get a decomposition or a combining class in a newer engine (the
    vocabulary has 60 of them too — U+0378 and others). They are rejected so that the two cannot
    disagree there.

## How the tables are baked

The source of truth of every decision is the Python side (ftfy's tables, the character classes of
`re` / `regex`, `unicodedata` 16.0.0); Unicode decisions are not reimplemented in TS (the same
discipline as anima's `anima/text.py`). Every code point is evaluated for real and folded into
closed-range tables and mapping tables, and {@link PromptCleanTables.to_json} turns them into the
form TS reads.

The TS implementation (`packages/models/src/wan/text/prompt-clean.ts`) is a transcription of
{@link clean} (the Python mirror that looks up only the baked tables), and the agreement between the
mirror and upstream `prompt_clean` is checked exhaustively and randomly by {@link verify_fuzz} and
{@link sweep_records}. On a mismatch nothing is written (main of `wan.umt5_tokenizer`).

MUST: import ftfy / regex / diffusers inside functions (the `wan` group is not in the default sync —
`tests/test_optional_group_imports.py`).
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

#: Every code point except surrogates (the range a Python str can hold).
ALL_CODEPOINTS = [cp for cp in range(0x110000) if not 0xD800 <= cp <= 0xDFFF]

#: The versions of the source of truth (the tables are baked from their actual behavior — with
#: another version nothing is baked).
FTFY_VERSION = "6.3.1"
UNIDATA_VERSION = "16.0.0"

#: The sha256 of the UTF-8 of the JS translation of `BADNESS_RE` (`emit_js` — the asset's
#: `badness`). MUST: the same value as `FTFY_BADNESS_SHA256` in
#: `packages/models/src/wan/text/prompt-clean.ts` (`tests/test_prompt_clean.py` compares the two),
#: which refuses to run any other expression from the asset. Checking it here makes a changed
#: translation (an ftfy bump, a change to `_emit`) fail at the writer instead of first showing up
#: as a red TS test. When the translation changes on purpose, update both values together.
FTFY_BADNESS_SHA256 = "7da149647fdaad8c293f44706c67b5649a6f76593cbcf9f23d9613680466312e"

#: ftfy's `TextFixerConfig.max_decode_length` (the maximum number of code points per segment).
MAX_DECODE_LENGTH = 1_000_000

#: C1 control characters (the rejected range of ADR 0119 decision 2).
C1_FIRST, C1_LAST = 0x80, 0x9F

#: The limit on fixed-point rounds (the same value as TS). With the upstream tables it stops after 2
#: rounds ({@link verify_fuzz} records the observed maximum). It is reached only when the tables are
#: broken, so TS fails with a plain Error.
MAX_FIX_ROUNDS = 8

#: ftfy's `HTML_ENTITY_RE` (`ftfy/chardata.py:82`). Rejected when it matches at the start of a
#: round.
FTFY_ENTITY_PATTERN = r"&#?[0-9A-Za-z]{1,24};"

#: `_charref` of `html.unescape` (Python 3.14 `html/__init__.py:118-120`).
HTML_CHARREF_PATTERN = r"&(#[0-9]+;?|#[xX][0-9a-fA-F]+;?|[^\t\n\f <&#;]{1,32};?)"

#: An equivalent form of {@link HTML_CHARREF_PATTERN} that only tells **whether it matches** (TS
#: holds this form). `+` / `{1,32}` / `;?` can always be satisfied once the first character matches,
#: so the first character alone decides existence.
HTML_CHARREF_PROBE = r"&(?:#[0-9]|#[xX][0-9a-fA-F]|[^\t\n\f <&#;])"

#: ftfy's `ANSI_RE` (`ftfy/fixes.py:142`). `\d` is Unicode Nd (a `re` str pattern).
ANSI_PATTERN = "\033\\[((?:\\d|;)*)([a-zA-Z])"


class PromptRejectedError(ValueError):
    """An input the mirror does not accept (the same reason vocabulary as the TS
    `PromptCleanError`)."""

    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(f"{reason}: {detail}")
        self.reason = reason


# ---------------------------------------------------------------------------
# Closed-range tables
# ---------------------------------------------------------------------------


def to_ranges(codepoints: Iterable[int]) -> Ranges:
    """Folds into an ascending list of closed ranges."""
    out: Ranges = []
    for cp in sorted(set(codepoints)):
        if out and out[-1][1] == cp - 1:
            out[-1][1] = cp
        else:
            out.append([cp, cp])
    return out


def ranges_where(predicate: Callable[[str], bool], *, surrogates: bool = False) -> Ranges:
    """The closed-range table of the code points where the predicate holds (asks the source of
    truth one character at a time and folds)."""
    domain = range(0x110000) if surrogates else ALL_CODEPOINTS
    return to_ranges(cp for cp in domain if predicate(chr(cp)))


class RangeSet:
    """Membership in a closed-range table (the mirror of TS `inCodeRanges` — binary search)."""

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
# Translation of BADNESS_RE (syntax tree → intermediate representation → JS / Python regex)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CharClass:
    r"""A set of single characters (literal characters, ranges, `\s` / `\w`, negation and `.` all
    fold into this)."""

    ranges: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class OptionalChar:
    """The preceding single character is optional (`?`)."""

    item: CharClass


@dataclass(frozen=True)
class LineStart:
    """The start of the string (`^` without MULTILINE)."""


Term = CharClass | OptionalChar | LineStart


def _category_ranges(category: Any) -> Ranges:
    r"""Evaluates a character category of a `re` str pattern over every code point (`\s` / `\w`)."""
    import re._constants as sre

    pattern = {sre.CATEGORY_SPACE: r"\s", sre.CATEGORY_WORD: r"\w"}.get(category)
    if pattern is None:
        raise ValueError(f"BADNESS_RE has an unexpected character category {category}")
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
            raise ValueError(f"BADNESS_RE has an unexpected element {op} in a character class")
    ranges = _union(*parts)
    return _complement(ranges) if negate else ranges


def _term(op: Any, av: Any) -> Term:
    import re._constants as sre

    if op is sre.LITERAL:
        return CharClass(((av, av),))
    if op is sre.IN:
        return CharClass(tuple((a, b) for a, b in _class_ranges(av)))
    if op is sre.ANY:
        # MUST: Python's `.` (without DOTALL) excludes only `\n`. JS `.` also excludes `\r` and
        # U+2028 / 2029, so translating to a bare `.` would split the decision only around line
        # breaks. Spell it out as a set.
        return CharClass(tuple((a, b) for a, b in _complement([[0x0A, 0x0A]])))
    if op is sre.MAX_REPEAT:
        low, high, inner = av
        if (low, high) != (0, 1) or len(inner) != 1:
            raise ValueError(f"BADNESS_RE has a repetition {low},{high} other than `?`")
        item = _term(*inner[0])
        if not isinstance(item, CharClass):
            raise ValueError("a `?` of BADNESS_RE does not apply to a single-character set")
        return OptionalChar(item)
    if op is sre.AT and av is sre.AT_BEGINNING:
        return LineStart()
    raise ValueError(f"BADNESS_RE has an unexpected element {op} {av!r}")


def badness_terms() -> list[list[Term]]:
    r"""Folds the syntax tree of `ftfy.badness.BADNESS_RE` into the intermediate representation
    "alternatives × sequences of terms".

    Copying the regular expression's **source string** to TS would make `re.VERBOSE` whitespace, the
    Unicode ranges of `\w` / `\s` and the line-break handling of `.` silently disagree with JS.
    Evaluating the sets from the syntax tree and spelling them out leaves JS looking only at
    `\u{…}` literals and ranges (independent of the engine's Unicode version).
    """
    import re._constants as sre
    import re._parser as sre_parse

    from ftfy.badness import BADNESS_RE

    tree = sre_parse.parse(BADNESS_RE.pattern, BADNESS_RE.flags)
    if len(tree) != 1 or tree[0][0] is not sre.BRANCH:
        raise ValueError("the top level of BADNESS_RE is not a single alternation (BRANCH)")
    _, branches = tree[0][1]
    return [[_term(op, av) for op, av in branch] for branch in branches]


def badness_literals() -> list[int]:
    r"""The non-ASCII characters that appear literally in `BADNESS_RE` (not those from `\s` / `\w`,
    negation or `.`).

    The boundary of the fuzz pool (the "449 characters" of research §3.3). Only the literal sets are
    counted, so the wide sets from categories (`\w` and so on) are not included.
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
    """The JS regular-expression source (used with the `u` flag — every character is written as
    `\\u{…}`)."""
    return _emit(terms, lambda cp: f"\\u{{{cp:X}}}")


def emit_python(terms: Sequence[Sequence[Term]]) -> str:
    """The Python version of the same intermediate representation (the same shape as the JS one;
    only the spelling of the escapes differs).

    pytest checks it against `BADNESS_RE` — the equivalence of folding into the intermediate
    representation is confirmed within Python, and the TS parity test pins only the spelling map
    (`\\u{…}`) on the JS side.
    """
    return _emit(terms, lambda cp: f"\\U{cp:08x}")


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PromptCleanTables:
    """The full set of tables TS looks up ({@link to_json} becomes `promptClean` of the asset and
    the fixture)."""

    #: `fix_latin_ligatures` (LIGATURES of `ftfy/chardata.py:207-230`).
    ligatures: dict[int, str]
    #: `fix_character_width` (WIDTH_MAP — NFKC of the full-width and half-width forms, and U+3000 →
    #: U+0020).
    width: dict[int, str]
    #: The characters that SINGLE_QUOTE_RE / DOUBLE_QUOTE_RE of `uncurl_quotes` match.
    single_quotes: Ranges
    double_quotes: Ranges
    #: The characters `remove_control_chars` removes (CONTROL_CHARS).
    control_chars: Ranges
    #: `\d` of `remove_terminal_escapes` (Unicode Nd of `re`).
    ansi_digits: Ranges
    #: Assigned in UCD 16.0.0 (`unicodedata.category != "Cn"`).
    assigned: Ranges
    #: The characters `str.strip()` drops (`str.isspace`).
    strip_space: Ranges
    #: `\s` of `regex` in `whitespace_clean`.
    collapse_space: Ranges
    #: The intermediate representation of `BADNESS_RE`.
    badness: list[list[Term]]

    def to_json(self) -> dict[str, Any]:
        badness = emit_js(self.badness)
        digest = hashlib.sha256(badness.encode("utf-8")).hexdigest()
        if digest != FTFY_BADNESS_SHA256:
            raise ValueError(
                f"badness: the JS translation of BADNESS_RE has sha256 {digest}, not the pinned"
                f" {FTFY_BADNESS_SHA256} — the TS reader would refuse the asset"
                " (update FTFY_BADNESS_SHA256 here and in prompt-clean.ts together)"
            )
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
            # The JS regular-expression source (`new RegExp(source, "u")`).
            "badness": badness,
        }


def check_versions() -> dict[str, str]:
    """Checks the versions of the source of truth (with other versions nothing is baked — the
    tables are those versions' actual behavior)."""
    import sys

    import ftfy
    import regex

    if ftfy.__version__ != FTFY_VERSION:
        raise ValueError(
            f"ftfy {ftfy.__version__} — the tables are baked from the behavior of {FTFY_VERSION}"
        )
    if unicodedata.unidata_version != UNIDATA_VERSION:
        raise ValueError(
            f"unicodedata {unicodedata.unidata_version} — the source of truth of NFC is UCD"
            f" {UNIDATA_VERSION}"
        )
    return {
        "ftfy": ftfy.__version__,
        "python": sys.version.split()[0],
        "regex": regex.__version__,
        "unicodedata": unicodedata.unidata_version,
    }


def check_upstream_shape() -> None:
    """Pins the upstream spellings the mirror relies on (if a version bump changes them, nothing is
    baked).

    Pins the parts that cannot go into tables (the shape of the regular expressions, the default
    settings) by their literal text. The behavioral equivalence is checked separately by
    {@link verify_fuzz} and {@link sweep_records}.
    """
    from ftfy import TextFixerConfig, chardata, fixes

    if chardata.HTML_ENTITY_RE.pattern != FTFY_ENTITY_PATTERN:
        raise ValueError(f"ftfy's HTML_ENTITY_RE changed: {chardata.HTML_ENTITY_RE.pattern!r}")
    if html._charref.pattern != HTML_CHARREF_PATTERN:  # type: ignore[attr-defined]
        raise ValueError(f"html._charref changed: {html._charref.pattern!r}")  # type: ignore[attr-defined]
    if fixes.ANSI_RE.pattern != ANSI_PATTERN:
        raise ValueError(f"ftfy's ANSI_RE changed: {fixes.ANSI_RE.pattern!r}")
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
        raise ValueError(f"ftfy's default settings changed: {actual}")


def build_tables() -> PromptCleanTables:
    """Bakes the tables from ftfy 6.3.1, `re` / `regex` and `unicodedata` 16.0.0."""
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
    """Checks the premises on the tables' output ranges (the grounds for "checking once at the
    entrance holds in every round").

    If a table lookup produced C1, an unassigned code point or a surrogate, the entrance check alone
    would no longer prove the no-op in every round. If the two quote sets intersected or an output
    matched itself, the order would disagree with TS, which writes the two replacements as one
    mapping.
    """
    assigned = RangeSet(tables.assigned)
    outputs = "".join(tables.ligatures.values()) + "".join(tables.width.values()) + "'\"\n"
    for ch in outputs:
        cp = ord(ch)
        if C1_FIRST <= cp <= C1_LAST or 0xD800 <= cp <= 0xDFFF or cp not in assigned:
            raise ValueError(f"table output U+{cp:04X} is C1 / a surrogate / unassigned")
    single, double = RangeSet(tables.single_quotes), RangeSet(tables.double_quotes)
    if any(cp in double for start, end in tables.single_quotes for cp in range(start, end + 1)):
        raise ValueError("the sets of SINGLE_QUOTE_RE and DOUBLE_QUOTE_RE intersect")
    if ord("'") in single or ord("'") in double or ord('"') in single or ord('"') in double:
        raise ValueError("a quote replacement is in the quote sets")


# ---------------------------------------------------------------------------
# The mirror (the source the TS implementation is transcribed from — looks up only the baked tables)
# ---------------------------------------------------------------------------


class CompiledTables:
    """{@link PromptCleanTables} in a form that is easy to look up.

    The mirror of TS `parsePromptCleanTables`.
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
    """The segments of `fix_text` (split right after each line break; at most
    {@link MAX_DECODE_LENGTH} per segment)."""
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
    """Removes `ESC [ (\\d|;)* letter` (left to right and non-overlapping, like `re.sub`)."""
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
    """Runs one segment to a fixed point.

    One round of `fix_and_explain` minus the steps pinned to a no-op by rejection.
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
    raise RuntimeError(
        f"no fixed point within {MAX_FIX_ROUNDS} rounds (the tables are broken): {segment!r}"
    )


def clean(compiled: CompiledTables, text: str, rounds: Counter[int] | None = None) -> str:
    """The mirror of `prompt_clean`. An input it does not accept raises
    {@link PromptRejectedError}."""
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
    """Upstream's `prompt_clean` (returned after checking that ftfy is installed)."""
    from diffusers.pipelines.wan.pipeline_wan import prompt_clean
    from diffusers.utils import is_ftfy_available

    check_versions()
    if not is_ftfy_available():
        raise ValueError("diffusers does not find ftfy — prompt_clean would skip fix_text")
    return prompt_clean


def record(
    compiled: CompiledTables,
    truth: Callable[[str], str],
    text: str,
    rounds: Counter[int] | None = None,
) -> str:
    """The outcome of one input (`A` + upstream's output when accepted, `R` + the reason when
    rejected).

    MUST: for an accepted input, always check that the mirror equals upstream before recording
    **upstream's output** — the expected value always comes from upstream, and the mirror decides
    only "which inputs to reject".
    """
    try:
        mirrored = clean(compiled, text, rounds)
    except PromptRejectedError as rejected:
        return "R" + rejected.reason
    expected = truth(text)
    if mirrored != expected:
        raise ValueError(
            "the mirror differs from upstream's prompt_clean\n"
            f"  input={[hex(ord(c)) for c in text]}\n"
            f"  upstream={expected!r}\n  mirror={mirrored!r}"
        )
    return "A" + expected


# ---------------------------------------------------------------------------
# Randomized testing (a PRNG that reproduces the same sequence as TS) and pools
# ---------------------------------------------------------------------------


class XorShift32:
    """32-bit xorshift (Marsaglia 13/17/5). Produces the same sequence as TS `xorshift32`."""

    def __init__(self, seed: int) -> None:
        if not 0 < seed < 2**32:
            raise ValueError(f"the xorshift32 seed must be in 1..2^32-1 (got {seed})")
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
    """Random strings of length 1..max_length (the PRNG is drawn for the length, then for each
    character — the same procedure as TS)."""
    rng = XorShift32(seed)
    for _ in range(count):
        length = 1 + rng.next() % max_length
        yield "".join(chr(pool[rng.next() % len(pool)]) for _ in range(length))


#: The material of the safe pool (the same composition as the pool of research §3.3 — characters
#: that appear in ordinary prompts and characters the deterministic steps act on). The literal
#: characters of `BADNESS_RE`, `&` and unassigned code points are removed from it.
SAFE_POOL_BLOCKS: tuple[tuple[int, int, str], ...] = (
    (0x0000, 0x007F, "ASCII (including control characters and ESC)"),
    (0x00A0, 0x017F, "Latin-1 Supplement, Latin Extended-A (including ĳ and ŉ)"),
    (0x01C4, 0x01CC, "Latin digraphs (fix_latin_ligatures)"),
    (0x02BC, 0x02BC, "modifier letter apostrophe (uncurl_quotes)"),
    (0x0300, 0x036F, "combining characters"),
    (0x0370, 0x04FF, "Greek, Cyrillic"),
    (0x0600, 0x06FF, "Arabic"),
    (0x0900, 0x097F, "Devanagari"),
    (0x0E00, 0x0E7F, "Thai"),
    (0x1100, 0x11FF, "Hangul Jamo"),
    (0x2000, 0x206F, "General Punctuation (curly quotes, U+2028 / 2029, 206A-206F)"),
    (0x3000, 0x30FF, "CJK symbols, kana"),
    (0x4E00, 0x51FF, "CJK Unified Ideographs (first 1,024)"),
    (0xAC00, 0xADFF, "Hangul Syllables (first 512)"),
    (0xFB00, 0xFB06, "Latin ligatures"),
    (0xFE00, 0xFE0F, "variation selectors"),
    (0xFEFF, 0xFEFF, "BOM"),
    (0xFF00, 0xFFEF, "Halfwidth and Fullwidth Forms"),
    (0xFFF9, 0xFFFC, "interlinear annotation, object replacement"),
    (0x1F1E6, 0x1F1FF, "regional indicators"),
    (0x1F300, 0x1F64F, "emoji"),
)

#: The additions to the pool that exercises the entity check (whitespace and full-width symbols
#: that can surround `&`).
ENTITY_POOL_EXTRA = (0x09, 0x0A, 0x0C, 0x0D, 0xA0, 0x3000, 0xFF03, 0xFF06, 0xFF1B)


def fuzz_pools(tables: PromptCleanTables) -> dict[str, Ranges]:
    """The fuzz pools (TS expands the range tables of the fixture and draws the same sequences).

    The safe pool is restricted not only to "contains neither `&` nor the literal characters of
    `BADNESS_RE`" but also to characters whose table-lookup output does not produce them either
    (the full-width `＆` `￠` become `&` `¢` through `fix_character_width` and hit the entity /
    mojibake checks in the next round — that would break what the safe pool means).
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
# Fixture (TS replays the same sequences and compares — the expected values are kept as digests)
# ---------------------------------------------------------------------------

#: The separator (it never appears inside a record — an accepted output has no control characters,
#: and a reason is an ASCII word).
RECORD_SEPARATOR = "\x00"


def digest(records: Iterable[str]) -> str:
    """SHA-256 of a sequence of records (UTF-8 with {@link RECORD_SEPARATOR} after each record)."""
    hasher = hashlib.sha256()
    for item in records:
        hasher.update((item + RECORD_SEPARATOR).encode("utf-8"))
    return hasher.hexdigest()


#: The number of digits of each chunk's digest (16 hex digits = 64 bits). Enough to detect
#: regressions, without bloating the fixture by the number of chunks.
BLOCK_DIGEST_HEX = 16


def chunked_digests(records: Sequence[str], chunk: int) -> list[str]:
    """A digest per `chunk` records (split so that on a mismatch TS can say which chunk)."""
    return [
        digest(records[i : i + chunk])[:BLOCK_DIGEST_HEX] for i in range(0, len(records), chunk)
    ]


@dataclass(frozen=True)
class FuzzRun:
    """The definition of one random run (TS builds the same sequence from this definition and the
    fixture's pools)."""

    name: str
    pool: str
    seed: int
    count: int
    max_length: int = 40
    chunk: int = 1000


#: Fuzzing of the preprocessing mirror (at least the pools and counts of research §3.3 — 200,000
#: strings from the safe pool, 50,000 from the full pool).
CLEAN_RUNS = (
    FuzzRun("safe", "safe", seed=1, count=200_000),
    FuzzRun("full", "full", seed=2, count=50_000),
    FuzzRun("entity", "entity", seed=3, count=20_000),
)

#: Fuzzing of the `BADNESS_RE` translation (a pool mixing in the 449 characters — the decision
#: itself is compared with `is_bad`).
BADNESS_RUNS = (FuzzRun("full", "full", seed=4, count=50_000),)

#: Contexts that exercise the Unicode-dependent syntax of `BADNESS_RE` (`\w`, `\s`, `.`, `[^…]`,
#: `^`) over every code point.
BADNESS_PROBES: tuple[tuple[str, str], ...] = (
    ("word", "à«{}"),
    ("space", "{}À€"),
    ("space-optional", "a{}Ã "),
    ("any", "ГўВЂВ{}a"),
    ("negated", "œ{}"),
    ("line-start", "{}Ã "),
)

#: Contexts that exercise the preprocessing mirror over every code point (both the strip at the
#: ends and the collapse in the middle).
CLEAN_SWEEPS: tuple[tuple[str, str], ...] = (("alone", "{}"), ("inner", "a{}b"))

#: The sweep for NFC version differences (only code points assigned in UCD 16.0.0 — unassigned ones
#: are rejected at the entrance). Mixes reordering (ccc 1 / 230), composition, Hangul and doubled
#: contexts.
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

#: The chunk that divides the sweep over every code point (so that TS can name the block that
#: mismatched).
SWEEP_BLOCK = 0x1000


def sweep_blocks(
    domain: Sequence[int], record_of: Callable[[int], str], block: int = SWEEP_BLOCK
) -> list[str]:
    """Splits `domain` into chunks of `block` and returns a digest per chunk (an empty chunk also
    gets one)."""
    buckets: list[list[str]] = [[] for _ in range(0x110000 // block)]
    for cp in domain:
        buckets[cp // block].append(record_of(cp))
    return [digest(bucket)[:BLOCK_DIGEST_HEX] for bucket in buckets]


def sweep_records(
    compiled: CompiledTables, truth: Callable[[str], str], template: str
) -> list[str]:
    """The sweep of the preprocessing over every code point (one character goes into `{}` of
    `template`)."""
    return sweep_blocks(
        ALL_CODEPOINTS, lambda cp: record(compiled, truth, template.replace("{}", chr(cp)))
    )


def verify_fuzz(
    compiled: CompiledTables, truth: Callable[[str], str], run: FuzzRun, pool: Sequence[int]
) -> dict[str, Any]:
    """Runs one random run and returns the digest per chunk and the breakdown of outcomes.

    If the mirror and upstream disagree, {@link record} fails.
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
    """Whether `search` matches each input, as a string of `0` / `1`."""
    return "".join("1" if pattern.search(text) else "0" for text in texts)


def bits_digest(bits: str) -> str:
    return hashlib.sha256(bits.encode("ascii")).hexdigest()
