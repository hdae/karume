r"""The Qwen2 byte-level BPE tokenizer folded into tables that are only looked up — the half shared
by the families whose text encoder is a Qwen model: Anima (`anima.demo`, Qwen3) and MOSS-Image
(`qwen3_5.tokenizer`, Qwen3.5). Shared, but not core knowledge, so it lives in `_shared` (ADR 0065
decision 2). The TS counterpart is `packages/models/src/text/qwen2-tokenizer.ts`.

This module holds three things:

  ① **Table extraction** — the `\p{L}` / `\p{N}` / `\s` that the pre_tokenizer regex uses, the
     `(?i:)` case fold of its contractions, and Qwen2's `NFC`, in the form the TS side only has to
     look up. **The TS side does not reimplement any of these Unicode decisions** — their source of
     truth is the Unicode tables on the Rust side (`tokenizers` / its regex engine /
     `unicode-normalization`), and the moment the JS engine's ICU version drifts from them the
     pre-token boundaries or the normalization result change and **only the id sequence becomes
     something else, with no exception and no warning**. Every code point is evaluated here and the
     answers are folded into closed-interval and mapping tables.

  ② **Reference implementation** — the mirror of the TS implementation (`Qwen2Reference`). From the
     tables baked in ① alone it yields the same ids as `AutoTokenizer`. Its being green is the
     mechanical proof that the tables carry every piece of information needed, and the TS side is a
     transcription of it.

  ③ **Verification** — exhaustive + random checks that the folding of ① is equivalent to the source
     of truth. It is the very ground for the folding, so the callers run it on every emit, and
     nothing is emitted when it fails (ADR 0005's fail loudly).

The family recipes own their own spellings (the upstream files, the cases, the asset layout); this
module knows only the Qwen2 tokenizer.

## How Qwen2's `NFC` is folded (from measurements)

`String.prototype.normalize("NFC")` / `unicodedata.normalize("NFC")` are **not the source of
truth**. The source of truth (`tokenizers.normalizers.NFC` = Rust's `unicode-normalization`) has
old Unicode tables, and measured (2026-08-03) the two disagree on **123 code points**:

  * 120 cp are treated by the source of truth as combining class 0 (starters) — ICU / `unicodedata`
    have ccc>0 for them and reorder them by canonical ordering, the source of truth does not (e.g.
    U+0818 U+089A).
  * 3 cp (U+113C2 / U+1611E / U+16D67) are newer canonical compositions, which ICU /
    `unicodedata` compose and the source of truth does not.

Both fold into the form "normalization breaks at that code point": **cut the string at the
segmenting code points, run plain NFC on each segment and concatenate; the segmenting code points
themselves take no part in normalization** (treated as a starter = involved in neither the
reordering nor the composition around them, which is exactly the source of truth's behaviour). The
set is baked by probing the source of truth, and the equivalence is checked on every code point × 11
contexts + random strings (`verify_nfc_segments`).
"""

from __future__ import annotations

import unicodedata
from collections.abc import Iterable, Sequence
from typing import Any

Ranges = list[list[int]]

#: Every code point except the surrogates (D800-DFFF cannot be held in a Python str, so they cannot
#: be evaluated).
ALL_CODEPOINTS = [cp for cp in range(0x110000) if not (0xD800 <= cp <= 0xDFFF)]


def to_ranges(codepoints: Iterable[int]) -> Ranges:
    """Folds into an ascending list of closed intervals."""
    out: Ranges = []
    for cp in sorted(codepoints):
        if out and out[-1][1] == cp - 1:
            out[-1][1] = cp
        else:
            out.append([cp, cp])
    return out


def in_ranges(ranges: Ranges, cp: int) -> bool:
    """Binary search over a closed-interval table (the mirror of the TS `inCodeRanges`)."""
    lo, hi = 0, len(ranges) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        start, end = ranges[mid]
        if cp < start:
            hi = mid - 1
        elif cp > end:
            lo = mid + 1
        else:
            return True
    return False


# ---------------------------------------------------------------------------
# ① Character classes (the \p{L} / \p{N} / \s that Qwen2's pre_tokenizer regex uses)
# ---------------------------------------------------------------------------


def build_char_classes() -> dict[str, Ranges]:
    r"""Asks the regex engine itself (the source of truth) one character at a time and folds the
    answers into closed-interval tables.

    Using `\p{L}` on the TS side would move the source of truth for the decision to the JS engine's
    ICU, and the moment it drifts from the Rust-side Unicode tables the result is **a different
    pre-token split** (a silent id-sequence mismatch).
    """
    from tokenizers import Regex, pre_tokenizers

    out: dict[str, Ranges] = {}
    for name, pattern in (("letter", r"\p{L}"), ("number", r"\p{N}"), ("space", r"\s")):
        splitter = pre_tokenizers.Split(Regex(pattern), behavior="removed", invert=False)
        out[name] = to_ranges(
            cp for cp in ALL_CODEPOINTS if splitter.pre_tokenize_str(chr(cp)) == []
        )
    return out


def build_case_fold() -> dict[int, int]:
    r"""Asks the source of truth which code points `(?i:)` equates with the characters of the
    contraction suffixes, and folds them.

    **Measured (2026-08-03): Rust's `(?i:)` is Unicode simple case folding, not an ASCII
    equivalent.** Trying every code point against the 8 characters that appear in the suffixes
    (s t r e v m l d) equates **U+017F (ſ) with s**, in addition to the ASCII capitals.

    Settling for an ASCII case flip makes `it'ſs` the single fragment `'ſs` (the source of truth
    gives `'ſ` + `s`), and the pre-token boundaries change. The table exists for this one entry,
    but the discipline is the same here: **the source of truth for the decision is not carried
    into TS** (neither `.lower()` nor `toLowerCase()` is the source of truth).

    Only the non-trivial equivalences are returned (the trivial pair `cp == the target character`
    is left out).
    """
    from tokenizers import Regex, pre_tokenizers

    out: dict[int, int] = {}
    for letter in sorted({ch for suffix in _APOSTROPHE_SUFFIXES for ch in suffix}):
        target = ord(letter)
        splitter = pre_tokenizers.Split(Regex(f"(?i:{letter})"), behavior="removed", invert=False)
        for cp in ALL_CODEPOINTS:
            if cp == target or splitter.pre_tokenize_str(chr(cp)) != []:
                continue
            if out.setdefault(cp, target) != target:
                raise ValueError(
                    f"U+{cp:04X} folds to both {chr(out[cp])!r} and {letter!r}"
                    " — each suffix would need its own table"
                )
    return out


# ---------------------------------------------------------------------------
# ① NFC (the segment table of the code points where the source of truth and plain NFC disagree)
# ---------------------------------------------------------------------------

#: The contexts that search for the segmenting code points; `{}` takes one character. A reordering
#: (a ccc difference) always shows in one of two directions — U+0334 (ccc 1) placed after, or
#: U+0301 (ccc 230) placed before (a ccc 1 code point does not move in the former). A missing
#: composition shows as the same character twice.
_NFC_DETECT_PROBES = (
    "{}",
    "a{}",
    "{}a",
    "a{}\u0301",
    "a\u0301{}",
    "a{}\u0334",
    "a\u0334{}",
    "{}{}",
)

#: The contexts of the exhaustive check of the segment model (deliberately **different** from the
#: detection contexts — the set the probes found is not checked only by the same probes). Hangul
#: L/V, Tibetan, the kana voiced sound mark and existing compositions are mixed in.
_NFC_VERIFY_PROBES = (
    "{}{}",
    "x{}y",
    "A{}\u0301",
    "\u0328{}",
    "{}\u0328",
    "\u1100{}\u1161",
    "\u1e0a{}\u0323",
    "e\u0303{}\u033c",
    "{}\u3099",
    "\u0f71{}\u0f72",
    "{}{}\u0301",
)

#: Characters always mixed into the NFC random test (combining marks, Hangul, emoji, variation
#: selectors, the ingredients of the newer compositions).
_NFC_FUZZ_SEEDS = (
    0x0041,
    0x00A0,
    0x00C0,
    0x000A,
    0x0300,
    0x0301,
    0x0302,
    0x0303,
    0x0304,
    0x0316,
    0x0328,
    0x0334,
    0x033C,
    0x05B0,
    0x064B,
    0x0653,
    0x093C,
    0x094D,
    0x0C3C,
    0x0EB3,
    0x0EBA,
    0x0F71,
    0x0F72,
    0x1AB0,
    0x1DC0,
    0x1E0A,
    0x20D0,
    0x3099,
    0x309A,
    0xAC00,
    0xFE0F,
    0xFE20,
    0x1100,
    0x1161,
    0x11A8,
    0x200D,
    0x113C1,
    0x113C2,
    0x113C5,
    0x1611E,
    0x16121,
    0x16D67,
    0x16D68,
    0x1F1EF,
    0x1F600,
    0x2F929,
    0xE0100,
)


def build_nfc_segments() -> Ranges:
    """Folds the code points where the source of truth's `NFC` and plain `unicodedata` disagree
    into a closed-interval table.

    The source of truth (Rust `unicode-normalization`) has old Unicode tables, and both its
    combining classes and its composition table disagree with ICU / `unicodedata` (the module
    docstring's section on Qwen2's NFC). The disagreeing code points are collected **by asking the
    source of truth one character at a time**, and TS / Python segment the string at them.
    """
    from tokenizers import normalizers

    norm = normalizers.NFC()
    out: list[int] = []
    for cp in ALL_CODEPOINTS:
        ch = chr(cp)
        for probe in _NFC_DETECT_PROBES:
            text = probe.replace("{}", ch)
            if norm.normalize_str(text) != unicodedata.normalize("NFC", text):
                out.append(cp)
                break
    return to_ranges(out)


def nfc_with_segments(text: str, segments: Ranges) -> str:
    """NFC with the segment table (the mirror of the TS implementation's `normalizeNfc`).

    MUST: a segmenting code point itself **takes no part** in normalization (it joins neither the
    segment before it nor the one after it). This copies the source of truth's behaviour exactly:
    it treats the code point as a starter, involved in neither reordering nor composition.
    """
    out: list[str] = []
    buf: list[str] = []
    for ch in text:
        if in_ranges(segments, ord(ch)):
            out.append(unicodedata.normalize("NFC", "".join(buf)))
            out.append(ch)
            buf = []
        else:
            buf.append(ch)
    out.append(unicodedata.normalize("NFC", "".join(buf)))
    return "".join(out)


def verify_nfc_segments(segments: Ranges, *, fuzz: int = 50_000) -> dict[str, Any]:
    """Checks, exhaustively and at random, that the segment model is equivalent to the source of
    truth's `NFC`.

    This is the emit gate. If even one code point is missing from the segment table the exhaustive
    part fails, and it also fails when **segmenting itself does not fit the source of truth** (for
    example, the source of truth had gained a newer composition rule). The seed is fixed, so the
    same table is checked against the same strings.
    """
    import random

    from tokenizers import normalizers

    norm = normalizers.NFC()

    for cp in ALL_CODEPOINTS:
        ch = chr(cp)
        for probe in _NFC_VERIFY_PROBES:
            text = probe.replace("{}", ch)
            got, want = nfc_with_segments(text, segments), norm.normalize_str(text)
            if got != want:
                raise ValueError(
                    f"U+{cp:04X} in context {probe!r}: NFC differs from the source of truth\n"
                    f"  truth={[hex(ord(c)) for c in want]}\n  table={[hex(ord(c)) for c in got]}"
                )

    alphabet = [chr(c) for c in range(0x20, 0x7F)]
    alphabet += [chr(c) for start, end in segments for c in range(start, end + 1)]
    alphabet += [chr(c) for c in _NFC_FUZZ_SEEDS]
    rng = random.Random(0)
    for _ in range(fuzz):
        s = "".join(rng.choice(alphabet) for _ in range(rng.randint(1, 12)))
        got, want = nfc_with_segments(s, segments), norm.normalize_str(s)
        if got != want:
            raise ValueError(
                "NFC differs from the source of truth on the random string "
                f"{[hex(ord(c)) for c in s]}: "
                f"truth={[hex(ord(c)) for c in want]} table={[hex(ord(c)) for c in got]}"
            )
    return {
        "segments": sum(end - start + 1 for start, end in segments),
        "ranges": len(segments),
        "probes": len(_NFC_VERIFY_PROBES),
        "checked": len(ALL_CODEPOINTS) * len(_NFC_VERIFY_PROBES),
        "fuzz": fuzz,
        "alphabet": len(alphabet),
    }


def nfc_fixture_cases(segments: Ranges, *, extra: int = 128, seed: int = 0) -> list[list[str]]:
    """`[input, the source of truth's output]` pairs that pin the TS side's reproduction of the
    source of truth's NFC.

    Two kinds are mixed:
      ① one per segmenting code point — a context where **plain NFC is always wrong** is chosen
         (it detects an implementation that does not read the table, or one whose table lacks a
         code point)
      ② seeded random strings — they catch JS's ICU drifting from `unicodedata` outside the
         segments (the emit gate can only see the Python implementation, so the fixture closes
         that hole)
    """
    import random

    from tokenizers import normalizers

    norm = normalizers.NFC()
    out: list[list[str]] = []
    for start, end in segments:
        for cp in range(start, end + 1):
            ch = chr(cp)
            for probe in _NFC_DETECT_PROBES:
                text = probe.replace("{}", ch)
                want = norm.normalize_str(text)
                if want != unicodedata.normalize("NFC", text):
                    out.append([text, want])
                    break
            else:  # pragma: no cover - unreachable: this runs after build_nfc_segments
                raise ValueError(f"U+{cp:04X} is in the segment table but no context disagrees")

    alphabet = [chr(c) for c in _NFC_FUZZ_SEEDS]
    alphabet += [chr(c) for start, end in segments for c in range(start, end + 1)]
    alphabet += [chr(c) for c in range(0x61, 0x7B)]
    rng = random.Random(seed)
    for _ in range(extra):
        text = "".join(rng.choice(alphabet) for _ in range(rng.randint(1, 10)))
        out.append([text, norm.normalize_str(text)])
    return out


#: The contexts of the exhaustive check of the pre-token scan (each makes one alternative fire);
#: `{}` takes one character. The last 5 are for the contractions (alternative ①) only. `'{}` alone
#: is **not enough** — with a one-character probe the result is the same single fragment whether ①
#: fires or not, so a case-insensitivity mix-up slips through (this hole was hit in practice). A
#: character after the probe makes "what ① left over" visible.
_PRETOKEN_PROBES = (
    "a{}b",
    "'{}",
    " {}\n",
    "{}{}",
    "1{}",
    "!{}z",
    "'{}z",  # one-character suffixes (s t m d)
    "'{}ez",  # the first character of a two-character suffix (re / ve)
    "'{}lz",  # ditto (ll)
    "'r{}z",  # the second character of a two-character suffix (the e of re / ve)
    "'l{}z",  # ditto (the l of ll)
)


def verify_qwen_pre_tokenize(
    pattern: str, classes: dict[str, Ranges], case_fold: dict[int, int]
) -> dict[str, Any]:
    """Checks that the hand-written scan agrees with the source of truth's Split on every code
    point × every context.

    Since the regex itself is not ported (the regex engine's Unicode tables are not carried into
    TS), the equivalence can only be established by measurement. This is the emit gate.
    """
    from tokenizers import Regex, pre_tokenizers

    splitter = pre_tokenizers.Split(Regex(pattern), behavior="isolated", invert=False)
    checked = 0
    for cp in ALL_CODEPOINTS:
        ch = chr(cp)
        for probe in _PRETOKEN_PROBES:
            text = probe.replace("{}", ch)
            want = [piece for piece, _ in splitter.pre_tokenize_str(text)]
            got = qwen_pre_tokenize(text, classes, case_fold)
            if got != want:
                raise ValueError(
                    f"U+{cp:04X} in context {probe!r}: the scan differs from the source of truth\n"
                    f"  truth={want}\n  scan={got}"
                )
            checked += 1
    return {"probes": len(_PRETOKEN_PROBES), "checked": checked}


# ---------------------------------------------------------------------------
# ② Reference implementation — Qwen2 (byte-level BPE)
# ---------------------------------------------------------------------------


def bytes_to_unicode() -> dict[int, str]:
    """GPT-2's byte → visible character table (an algorithm, so it does not depend on the Unicode
    version).

    **The only part that needs no baked table.** The TS side builds it the same way.
    """
    printable = list(range(ord("!"), ord("~") + 1))
    printable += list(range(ord("\xa1"), ord("\xac") + 1))
    printable += list(range(ord("\xae"), ord("\xff") + 1))
    mapped = list(printable)
    extra = 0
    for byte in range(256):
        if byte not in printable:
            printable.append(byte)
            mapped.append(256 + extra)
            extra += 1
    return {b: chr(c) for b, c in zip(printable, mapped, strict=True)}


def split_added_tokens(text: str, added: Sequence[str]) -> list[tuple[str, bool]]:
    """Cuts out the added tokens (special tokens) leftmost-longest.

    `tokenizers`' `AddedVocabulary` runs AhoCorasick with LeftmostLongest, so at the same position
    the longer token wins and at different positions the left one wins. A cut-out piece goes
    **through neither** normalization nor pre-tokenization and yields its id directly (the stage
    acts before normalization, so swapping the order gives something else).
    """
    if any(token == "" for token in added):
        # An empty string matches at every position and `i` does not advance (an endless loop).
        # Upstream's added_tokens are unlikely to hold an empty content, but failing beats spinning
        # silently.
        raise ValueError("an added token is the empty string — the cut cannot advance")
    out: list[tuple[str, bool]] = []
    buf: list[str] = []
    i = 0
    while i < len(text):
        hit = None
        for token in added:
            if text.startswith(token, i) and (hit is None or len(token) > len(hit)):
                hit = token
        if hit is None:
            buf.append(text[i])
            i += 1
            continue
        if buf:
            out.append(("".join(buf), False))
            buf = []
        out.append((hit, True))
        i += len(hit)
    if buf:
        out.append(("".join(buf), False))
    return out


_APOSTROPHE_SUFFIXES = ("s", "t", "re", "ve", "m", "ll", "d")


def qwen_pre_tokenize(
    text: str, classes: dict[str, Ranges], case_fold: dict[int, int]
) -> list[str]:
    r"""Reproduces Qwen2's Split regex with a hand-written scan.

    The source of truth's pattern (leftmost; among the alternatives, the first that matches from
    the top wins):
      ① `(?i:'s|'t|'re|'ve|'m|'ll|'d)`
      ② `[^\r\n\p{L}\p{N}]?\p{L}+`
      ③ `\p{N}`
      ④ ` ?[^\s\p{L}\p{N}]+[\r\n]*`
      ⑤ `\s*[\r\n]+`
      ⑥ `\s+(?!\S)`
      ⑦ `\s+`
    No alternative matches the empty string, so picking one per position and advancing is
    equivalent to find_iter.
    """
    letter, number, space = classes["letter"], classes["number"], classes["space"]

    def is_letter(k: int) -> bool:
        return in_ranges(letter, ord(text[k]))

    def is_number(k: int) -> bool:
        return in_ranges(number, ord(text[k]))

    def is_space(k: int) -> bool:
        return in_ranges(space, ord(text[k]))

    def is_other(k: int) -> bool:
        return not is_space(k) and not is_letter(k) and not is_number(k)

    def is_newline(k: int) -> bool:
        return text[k] in ("\r", "\n")

    out: list[str] = []
    i, n = 0, len(text)
    while i < n:
        end = -1
        # ① Contractions (an ASCII apostrophe + a case-insensitive suffix). Case-insensitivity is
        # looked up in the baked table (`.lower()` / `toLowerCase()` are not the source of truth —
        # the docstring of `build_case_fold`).
        if text[i] == "'":
            for suffix in _APOSTROPHE_SUFFIXES:
                if _matches_folded(text, i + 1, suffix, case_fold):
                    end = i + 1 + len(suffix)
                    break
        # ② One optional character (other than CR/LF, a letter or a number) + a run of letters. The
        # optional part is taken first, and given back when that fails.
        if end < 0:
            for skip in (1, 0):
                start = i + skip
                if skip == 1 and (is_newline(i) or is_letter(i) or is_number(i)):
                    continue
                k = start
                while k < n and is_letter(k):
                    k += 1
                if k > start:
                    end = k
                    break
        # ③ One number character (there is no `+`, so numbers always split one by one)
        if end < 0 and is_number(i):
            end = i + 1
        # ④ One optional ASCII space + a run of others + a run of CR/LF
        if end < 0:
            for skip in (1, 0):
                start = i + skip
                if skip == 1 and text[i] != " ":
                    continue
                k = start
                while k < n and is_other(k):
                    k += 1
                if k > start:
                    while k < n and is_newline(k):
                        k += 1
                    end = k
                    break
        # ⑤ The longest run of whitespace that ends in CR/LF
        if end < 0 and is_space(i):
            k = i
            while k < n and is_space(k):
                k += 1
            last = -1
            for m in range(i, k):
                if is_newline(m):
                    last = m
            if last >= 0:
                end = last + 1
        # ⑥ A run of whitespace (leaving the last character when a non-space follows) / ⑦ a run of
        # whitespace
        if end < 0 and is_space(i):
            k = i
            while k < n and is_space(k):
                k += 1
            end = k if k == n else max(i + 1, k - 1)
        if end <= i:
            raise ValueError(
                f"the pre-token scan does not advance (position {i}, U+{ord(text[i]):04X})"
            )
        out.append(text[i:end])
        i = end
    return out


def _matches_folded(text: str, at: int, suffix: str, case_fold: dict[int, int]) -> bool:
    """Whether `text[at:]` matches `suffix` case-insensitively (the equivalence is decided by the
    baked table alone)."""
    if at + len(suffix) > len(text):
        return False
    for k, want in enumerate(suffix):
        cp = ord(text[at + k])
        if cp != ord(want) and case_fold.get(cp) != ord(want):
            return False
    return True


class Qwen2Reference:
    """Reference implementation of the Qwen2 tokenizer (the mirror of the TS implementation).

    `used_vocab` / `used_merges` record what was used, to build the fixture's vocabulary subset
    (so the 151k / 151k entries are not committed whole).
    """

    def __init__(
        self,
        vocab: dict[str, int],
        merges: dict[tuple[str, str], int],
        added: dict[str, int],
        classes: dict[str, Ranges],
        case_fold: dict[int, int],
        nfc_segments: Ranges,
    ) -> None:
        self.vocab = vocab
        self.merges = merges
        self.added = added
        self.classes = classes
        self.case_fold = case_fold
        self.nfc_segments = nfc_segments
        self.byte_encoder = bytes_to_unicode()
        self.used_vocab: set[str] = set()
        self.used_merges: set[tuple[str, str]] = set()

    def bpe(self, piece: str) -> list[str]:
        symbols = list(piece)
        while len(symbols) > 1:
            best: tuple[int, int] | None = None
            for idx in range(len(symbols) - 1):
                rank = self.merges.get((symbols[idx], symbols[idx + 1]))
                if rank is not None and (best is None or rank < best[0]):
                    best = (rank, idx)
            if best is None:
                break
            _, idx = best
            self.used_merges.add((symbols[idx], symbols[idx + 1]))
            symbols[idx : idx + 2] = [symbols[idx] + symbols[idx + 1]]
        return symbols

    def encode(self, text: str, max_length: int | None = None) -> list[int]:
        """The post_processor adds no special tokens, so truncation simply keeps the head."""
        ids: list[int] = []
        for chunk, is_added in split_added_tokens(text, sorted(self.added, key=len, reverse=True)):
            if is_added:
                ids.append(self.added[chunk])
                continue
            # MUST: plain `unicodedata.normalize` is not used directly. It disagrees with the source
            # of truth's NFC on 123 code points (the module docstring's section on Qwen2's NFC) —
            # cut with the segment table first, then normalize.
            normalized = nfc_with_segments(chunk, self.nfc_segments)
            for piece in qwen_pre_tokenize(normalized, self.classes, self.case_fold):
                encoded = "".join(self.byte_encoder[b] for b in piece.encode("utf-8"))
                for token in self.bpe(encoded):
                    if token not in self.vocab:
                        raise ValueError(f"token {token!r} is not in the Qwen2 vocabulary")
                    self.used_vocab.add(token)
                    ids.append(self.vocab[token])
        return ids if max_length is None else ids[:max_length]
