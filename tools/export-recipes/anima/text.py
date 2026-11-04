r"""Folds Anima's prompt tokenizers (Qwen2 BPE / T5 Unigram) into tables that are only looked up.

The Qwen2 (byte-level BPE) half — the `\p{L}` / `\p{N}` / `\s` character classes, the `(?i:)` case
fold, the `NFC` segment table and the mirror `Qwen2Reference` — is shared with `qwen3_5.tokenizer`
and lives in `_shared/qwen2_tokenizer.py` (its docstring holds how those tables are folded). This
module keeps the T5 (Unigram) half and the reader of Anima's two `tokenizer.json`. It holds three
things:

  ① **Table extraction** — from HF's `tokenizer.json`, the form the TS side only has to look up.
     The main subject is T5's `Precompiled` normalizer (SentencePiece's DARTS trie). **The TS side
     does not reimplement the Unicode decisions** — their source of truth is the Unicode tables on
     the Rust side (`tokenizers`), and the moment the JS engine's ICU version drifts from them the
     normalization result changes and **only the id sequence becomes something else, with no
     exception and no warning**. This module evaluates every code point and folds the answers into
     closed-interval and mapping tables (the same discipline as `sbv2_demo.clean_text_ranges`).

  ② **Reference implementation** — the mirror of the TS implementation (`T5Reference`). From the
     tables baked in ① alone it yields the same ids as `AutoTokenizer`. Its being green is the
     mechanical proof that the tables carry every piece of information needed, and the TS side is
     a transcription of it (compared in `anima/tests/test_demo.py`).

  ③ **Verification** — exhaustive + random checks that the folding of ① is equivalent to the source
     of truth. It is the very ground for the folding, so it runs on every emit, and nothing is
     emitted when it fails (ADR 0005's fail loudly).

## How T5's `Precompiled` is folded (from measurements)

`tokenizers`' Precompiled is not SentencePiece's normalization; it actually behaves like this:

    for each grapheme cluster g:
      if g's UTF-8 length < 6 and "some prefix of g" is in the trie
        → emit the value of the **shortest** matching prefix and drop the rest of g
      otherwise look up one code point at a time (left as is when absent)

Measured evidence (all with `normalizers.Precompiled.normalize_str`):
  `A`+U+0301+U+0301 → `Á` (the third character disappears) / `A`+U+0302+U+0301 → `Â`
  (the 2cp prefix wins over the 3cp rule `Ấ`) / U+1D2C+U+0300 → `A` (the 1cp prefix wins).

That fixes the form the folding can take:

  * Only the keys that have no rule on a proper prefix and are under 6 UTF-8 bytes take effect. The
    rest are unreachable.
  * The cluster boundaries matter not as UAX#29 itself but only as far as a rule can fire. What is
    needed is (a) `extend`, the characters that join the preceding one, (b) `break_after`, the
    characters after which the cluster always breaks (= GB4's Control), (c) `prepend`, the ones
    that pull in what follows (GB9b), and (d) CRLF. Each is folded into closed intervals by
    **evaluating every code point with probes against the source of truth** (UAX#29's tables are
    not carried into TS).
  * Hangul, regional indicators and emoji ZWJ sequences are 6 bytes or more per unit and never
    enter the whole-replacement path, so the output does not change even without implementing
    their boundary rules (confirmed by the random test).
"""

from __future__ import annotations

import base64
import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from _shared.qwen2_tokenizer import ALL_CODEPOINTS, Ranges, in_ranges, split_added_tokens, to_ranges

#: The limit under which `Precompiled` tries a whole-cluster replacement (spm_precompiled's
#: `grapheme.len() < 6`).
CLUSTER_BYTE_LIMIT = 6

#: Unigram's unknown-character penalty (`tokenizers`' `K_UNK_PENALTY`).
UNK_PENALTY = 10.0

# ---------------------------------------------------------------------------
# ① Decoding the DARTS trie and minimising the rules
# ---------------------------------------------------------------------------


def decode_charsmap(blob: bytes) -> dict[str, str]:
    """Extract every (key, value) from the Darts-clone double array of `precompiled_charsmap`.

    The format is the same as spm_precompiled's: the trie byte length as u32 LE + the trie body +
    a pool of NUL-separated normalized strings. The meaning of a unit (has_leaf / value / label /
    offset) follows Darts-clone 0.32.
    """
    trie_size = struct.unpack_from("<I", blob, 0)[0]
    if trie_size % 4 != 0 or 4 + trie_size > len(blob):
        raise ValueError(f"invalid charsmap trie length {trie_size}")
    units = struct.unpack(f"<{trie_size // 4}I", blob[4 : 4 + trie_size])
    pool = blob[4 + trie_size :]

    def read_pool(index: int) -> str:
        end = index
        while end < len(pool) and pool[end] != 0:
            end += 1
        return pool[index:end].decode("utf-8")

    out: dict[str, str] = {}
    # Recursion would only go as deep as the longest key's byte length, but an explicit stack is
    # used.
    root = 0 ^ ((units[0] >> 10) << ((units[0] & (1 << 9)) >> 6))
    stack: list[tuple[int, bytes]] = [(root, b"")]
    while stack:
        node_pos, prefix = stack.pop()
        for byte in range(256):
            pos = node_pos ^ byte
            if pos >= len(units):
                continue
            unit = units[pos]
            if unit & ((1 << 31) | 0xFF) != byte:
                continue
            child = pos ^ ((unit >> 10) << ((unit & (1 << 9)) >> 6))
            if child >= len(units):
                continue
            key = prefix + bytes([byte])
            if (unit >> 8) & 1:
                out[key.decode("utf-8")] = read_pool(units[child] & ((1 << 31) - 1))
            stack.append((child, key))
    return out


@dataclass(frozen=True)
class SpmTables:
    """`Precompiled` folded into a form the TS side only looks up."""

    #: The mapping of one-code-point keys (key = one character).
    single: dict[str, str]
    #: The mapping of two-code-point keys (only those with no rule on a proper prefix = the ones
    #: that can fire).
    multi: dict[str, str]
    #: Characters that join the preceding character's cluster (UAX#29's Extend / ZWJ /
    #: SpacingMark).
    extend: Ranges
    #: The cluster always breaks right after these characters (UAX#29 GB4's Control).
    break_after: Ranges
    #: Characters that pull what follows into their cluster (UAX#29 GB9b's Prepend).
    prepend: Ranges

    def to_json(self) -> dict[str, Any]:
        return {
            "single": [[ord(k), v] for k, v in sorted(self.single.items())],
            "multi": [[ord(k[0]), ord(k[1]), v] for k, v in sorted(self.multi.items())],
            "extend": self.extend,
            "breakAfter": self.break_after,
            "prepend": self.prepend,
        }

    @staticmethod
    def from_json(raw: dict[str, Any]) -> SpmTables:
        return SpmTables(
            single={chr(cp): v for cp, v in raw["single"]},
            multi={chr(a) + chr(b): v for a, b, v in raw["multi"]},
            extend=raw["extend"],
            break_after=raw["breakAfter"],
            prepend=raw["prepend"],
        )


def build_spm_tables(blob: bytes) -> SpmTables:
    """Build the tables from the charsmap and the source of truth (`normalizers.Precompiled`).

    Rule minimisation: the whole replacement takes "the shortest matching prefix", so a key that
    has a rule on a proper prefix never fires. In addition, a cluster of 6 bytes or more never
    takes the whole-replacement path at all, so keys of 6 bytes or more are dropped too.
    """
    from tokenizers import normalizers

    norm = normalizers.Precompiled(blob)
    rules = decode_charsmap(blob)
    firable = {k: v for k, v in rules.items() if len(k.encode()) < CLUSTER_BYTE_LIMIT}
    minimal = {
        k: v for k, v in firable.items() if not any(k[:i] in rules for i in range(1, len(k)))
    }
    single = {k: v for k, v in minimal.items() if len(k) == 1}
    multi = {k: v for k, v in minimal.items() if len(k) > 1}
    if any(len(k) != 2 for k in multi):
        raise ValueError(
            "a rule that can fire spans more than two code points — the folding's premise broke"
        )

    # --- Probe: append x to "a 2cp cluster whose rule fires" to push it to 6 bytes or more.
    # If x joins the same cluster, the whole-replacement path becomes unusable, which shows as
    # the rule not firing.
    probe_short = next(k for k in multi if len(k.encode()) == 3)
    probe_long = next(k for k in multi if len(k.encode()) == 5)

    def joins(probe: str, x: str) -> bool | None:
        if len(probe.encode()) + len(x.encode()) < CLUSTER_BYTE_LIMIT:
            return None
        got = norm.normalize_str(probe + x)
        tail = single.get(x, x)
        if got == "".join(single.get(c, c) for c in probe) + tail:
            return True
        if got == minimal[probe] + tail:
            return False
        raise ValueError(
            f"unexpected output of the extend probe: {got!r} (probe={probe!r} x={x!r})"
        )

    extend: list[int] = []
    for cp in ALL_CODEPOINTS:
        x = chr(cp)
        verdict = joins(probe_short, x)
        if verdict is None:
            verdict = joins(probe_long, x)
        if verdict:
            extend.append(cp)

    # --- prepend: does c + "o" + U+0303 compose õ? If not, c pulls what follows into its
    # cluster (= the whole-replacement path is unusable and the rule does not fire).
    prepend_tail = "o" + chr(0x0303)
    prepend: list[int] = []
    for cp in ALL_CODEPOINTS:
        c = chr(cp)
        head = single.get(c, c)
        got = norm.normalize_str(c + prepend_tail)
        if got == head + prepend_tail:
            prepend.append(cp)
        elif got != head + chr(0x00F5):
            raise ValueError(f"unexpected output of the prepend probe: {got!r} (U+{cp:04X})")

    # --- break_after: append U+0EB3 ("extend, with a rule") and U+0337 ("extend, without a
    # rule"). With a break, [U+0EB3, U+0337] becomes a 5-byte cluster and U+0337 is dropped;
    # without one, the total is 6 bytes or more, the per-character path is taken and U+0337 stays.
    break_tail = chr(0x0EB3) + chr(0x0337)
    break_cluster = single[chr(0x0EB3)]
    break_perchar = single[chr(0x0EB3)] + chr(0x0337)
    break_after: list[int] = []
    for cp in ALL_CODEPOINTS:
        c = chr(cp)
        head = single.get(c, c)
        got = norm.normalize_str(c + break_tail)
        if got == head + break_cluster:
            break_after.append(cp)
        elif got != head + break_perchar:
            raise ValueError(f"unexpected output of the break-after probe: {got!r} (U+{cp:04X})")

    return SpmTables(
        single=single,
        multi=multi,
        extend=to_ranges(extend),
        break_after=to_ranges(break_after),
        prepend=to_ranges(prepend),
    )


def normalize_with_tables(tables: SpmTables, text: str) -> str:
    """Reproduce `Precompiled` from the baked tables alone (the mirror of the TS implementation
    `normalizeSpm`)."""
    out: list[str] = []
    i, n = 0, len(text)
    while i < n:
        j = i
        while j < n and in_ranges(tables.prepend, ord(text[j])):
            j += 1
        # A control character right after a prepend breaks there (it is not pulled in).
        if j < n and not (j > i and in_ranges(tables.break_after, ord(text[j]))):
            if text[j] == "\r" and j + 1 < n and text[j + 1] == "\n":
                j += 2  # GB3: CR × LF
            elif in_ranges(tables.break_after, ord(text[j])):
                j += 1
            else:
                j += 1
                while j < n and in_ranges(tables.extend, ord(text[j])):
                    j += 1
        cluster = text[i:j]
        replacement: str | None = None
        if len(cluster.encode()) < CLUSTER_BYTE_LIMIT:
            # The shortest matching prefix wins. The rest is dropped.
            for length in (1, 2):
                if length <= len(cluster):
                    hit = (tables.single if length == 1 else tables.multi).get(cluster[:length])
                    if hit is not None:
                        replacement = hit
                        break
        if replacement is not None:
            out.append(replacement)
        else:
            out.extend(tables.single.get(c, c) for c in cluster)
        i = j
    return "".join(out)


def _sample_ranges(ranges: Ranges, limit: int) -> list[int]:
    out: list[int] = []
    for start, end in ranges:
        for cp in range(start, min(end, start + 8) + 1):
            out.append(cp)
            if len(out) >= limit:
                return out
    return out


#: Characters always mixed into the random test (Hangul jamo / regional indicators / skin tones /
#: variation selectors / Thai / Indic). Seeds that make the test actually exercise the side where
#: "the 6-byte boundary spares us implementing the boundary rules".
_FUZZ_SEEDS = (
    0x3053,
    0x4E16,
    0x30AB,
    0x1F600,
    0x1F1EF,
    0x1F1F5,
    0x1F3FB,
    0xE0100,
    0xFE0F,
    0x1100,
    0x1161,
    0x11A8,
    0x0E01,
    0x0E33,
    0x0903,
    0x0915,
    0x093C,
    0x094D,
    0x0937,
    0x0D4E,
    0x2F929,
)


def verify_spm_tables(tables: SpmTables, blob: bytes, *, fuzz: int = 200_000) -> dict[str, Any]:
    """Check exhaustively + randomly that the folding is equivalent to the source of truth.

    (a) every code point alone (b) the keys of the rules that fire (c) random strings mixing
    characters taken from the tables. The seed is fixed, so the same tables check the same
    strings (the check is never unreproducible).
    """
    import random

    from tokenizers import normalizers

    norm = normalizers.Precompiled(blob)

    for cp in ALL_CODEPOINTS:
        c = chr(cp)
        got, want = norm.normalize_str(c), normalize_with_tables(tables, c)
        if got != want:
            raise ValueError(f"single cp mismatch U+{cp:04X}: truth={got!r} tables={want!r}")

    for key in list(tables.single) + list(tables.multi):
        got, want = norm.normalize_str(key), normalize_with_tables(tables, key)
        if got != want:
            raise ValueError(f"mismatch on rule key {key!r}: truth={got!r} tables={want!r}")

    alphabet = [chr(c) for c in range(0x00, 0x7F)]
    alphabet += [chr(c) for c in _sample_ranges(tables.extend, 400)]
    alphabet += [chr(c) for c in _sample_ranges(tables.break_after, 200)]
    alphabet += [chr(c) for c in _sample_ranges(tables.prepend, 64)]
    alphabet += sorted(tables.multi)
    keys = sorted(tables.single)
    alphabet += keys[:400] + keys[len(keys) // 2 : len(keys) // 2 + 400] + keys[-400:]
    alphabet += [chr(c) for c in _FUZZ_SEEDS]
    rng = random.Random(0)
    for _ in range(fuzz):
        s = "".join(rng.choice(alphabet) for _ in range(rng.randint(1, 8)))
        got, want = norm.normalize_str(s), normalize_with_tables(tables, s)
        if got != want:
            raise ValueError(
                f"mismatch on random string {[hex(ord(c)) for c in s]}: "
                f"truth={[hex(ord(c)) for c in got]} tables={[hex(ord(c)) for c in want]}"
            )
    return {
        "codepoints": len(ALL_CODEPOINTS),
        "rules": len(tables.single) + len(tables.multi),
        "fuzz": fuzz,
        "alphabet": len(alphabet),
    }


# ---------------------------------------------------------------------------
# ① The T5 side's whitespace (it reuses Qwen2's `\s` table)
# ---------------------------------------------------------------------------


def verify_char_classes(classes: dict[str, Ranges]) -> dict[str, Any]:
    r"""Check that `\s` equals the whitespace set of `WhitespaceSplit` (the T5 side's split).

    If they match, the T5 side can reuse the same table. If they drift, a separate table is
    needed, so it fails.
    """
    from tokenizers import pre_tokenizers

    ws = pre_tokenizers.WhitespaceSplit()
    for cp in ALL_CODEPOINTS:
        splits = len(ws.pre_tokenize_str("a" + chr(cp) + "b")) > 1
        if splits != in_ranges(classes["space"], cp):
            raise ValueError(f"U+{cp:04X}: WhitespaceSplit and \\s disagree")
    return {name: len(r) for name, r in classes.items()}


# ---------------------------------------------------------------------------
# ② Reference implementation — T5 (Unigram + Metaspace)
# ---------------------------------------------------------------------------

METASPACE = "▁"


def t5_pre_tokenize(text: str, space: Ranges) -> list[str]:
    """WhitespaceSplit → Metaspace (replacement=▁ / prepend_scheme=always / split).

    `split=true` is MergedWithNext, so a separating `▁` goes at the start of the **next** piece.
    """
    out: list[str] = []
    for word in _split_on_space(text, space):
        piece = word.replace(" ", METASPACE)
        if not piece.startswith(METASPACE):
            piece = METASPACE + piece
        start = 0
        for idx in range(1, len(piece)):
            if piece[idx] == METASPACE:
                out.append(piece[start:idx])
                start = idx
        out.append(piece[start:])
    return [p for p in out if p]


def _split_on_space(text: str, space: Ranges) -> list[str]:
    out: list[str] = []
    buf: list[str] = []
    for ch in text:
        if in_ranges(space, ord(ch)):
            if buf:
                out.append("".join(buf))
                buf = []
        else:
            buf.append(ch)
    if buf:
        out.append("".join(buf))
    return out


class T5Reference:
    """Reference implementation of the T5 (Unigram) tokenizer (the mirror of the TS
    implementation)."""

    def __init__(
        self,
        vocab: dict[str, tuple[int, float]],
        min_score: float,
        unk_id: int,
        eos_id: int,
        added: dict[str, int],
        tables: SpmTables,
        space: Ranges,
    ) -> None:
        self.vocab = vocab
        self.min_score = min_score
        self.unk_id = unk_id
        self.eos_id = eos_id
        self.added = added
        self.tables = tables
        self.space = space
        self.max_token = max(len(t) for t in vocab)
        self.used_vocab: set[str] = set()

    def viterbi_pieces(self, piece: str) -> list[str]:
        """Port `tokenizers`' `Lattice::viterbi` onto code-point positions.

        On a tie, "the one pushed onto end_nodes first" wins. They are pushed in ascending begin
        order (= of those ending at the same position, the longer comes first), so a tie takes
        the longer piece — the comparison MUST be a **strict `>`** (with `>=` the shorter wins,
        occasionally giving another split and silently changing the id sequence).
        """
        n = len(piece)
        unk_score = self.min_score - UNK_PENALTY
        # begin[pos] = the lengths of the nodes starting there (insertion order = ascending length)
        begin: list[list[int]] = []
        scores: list[list[float]] = []
        for pos in range(n):
            lengths: list[int] = []
            node_scores: list[float] = []
            for length in range(1, min(self.max_token, n - pos) + 1):
                entry = self.vocab.get(piece[pos : pos + length])
                if entry is not None:
                    lengths.append(length)
                    node_scores.append(entry[1])
            if 1 not in lengths:
                lengths.insert(0, 1)
                node_scores.insert(0, unk_score)
            begin.append(lengths)
            scores.append(node_scores)
        # end[e] = the nodes ending there (start position, slot), pushed in ascending begin order.
        end: list[list[tuple[int, int]]] = [[] for _ in range(n + 1)]
        for pos in range(n):
            for slot, length in enumerate(begin[pos]):
                end[pos + length].append((pos, slot))

        best: list[list[float]] = [[0.0] * len(begin[p]) for p in range(n)]
        prev: list[list[tuple[int, int] | None]] = [[None] * len(begin[p]) for p in range(n)]

        def best_left(pos: int) -> tuple[float, tuple[int, int] | None] | None:
            """The best node ending at pos. pos=0 is bos (score 0). None if unreachable."""
            if pos == 0:
                return 0.0, None
            found: tuple[float, tuple[int, int] | None] | None = None
            for lpos, lslot in end[pos]:
                candidate = best[lpos][lslot]
                if found is None or candidate > found[0]:
                    found = (candidate, (lpos, lslot))
            return found

        for pos in range(n):
            left = best_left(pos)
            if left is None:
                return []
            for slot in range(len(begin[pos])):
                best[pos][slot] = left[0] + scores[pos][slot]
                prev[pos][slot] = left[1]
        tail = best_left(n)
        if tail is None:
            return []
        path: list[tuple[int, int]] = []
        node = tail[1]
        while node is not None:
            path.append(node)
            node = prev[node[0]][node[1]]
        path.reverse()
        return [piece[p : p + begin[p][s]] for p, s in path]

    def encode_piece(self, piece: str) -> list[int]:
        """Consecutive unknown nodes are fused into one token (`fuse_unk`).

        NOTE: `fuse_unk` in `tokenizer.json` is `null`, but `tokenizers`' Unigram **fuses even
        when it is unspecified** (the Rust-side default). An implementation that does not fuse
        lists one unk per character on Japanese prompts, and the comparison with `AutoTokenizer`
        (`anima/tests/test_demo.py`) fails on the `japanese` case.
        """
        out: list[int] = []
        pending = ""
        for text in self.viterbi_pieces(piece):
            entry = self.vocab.get(text)
            if entry is None:
                pending += text
                continue
            if pending:
                out.append(self.unk_id)
                pending = ""
            self.used_vocab.add(text)
            out.append(entry[0])
        if pending:
            out.append(self.unk_id)
        return out

    def encode(self, text: str, max_length: int | None = None) -> list[int]:
        """MUST: truncate **before** appending `</s>` (the source of truth truncates before the
        post_processor).

        Appending after `max_length` would give 513 ids. As a result, T5's id sequence always
        ends with `</s>` and is always at least 1 long.
        """
        ids: list[int] = []
        for chunk, is_added in split_added_tokens(text, sorted(self.added, key=len, reverse=True)):
            if is_added:
                ids.append(self.added[chunk])
                continue
            normalized = normalize_with_tables(self.tables, chunk)
            for piece in t5_pre_tokenize(normalized, self.space):
                ids.extend(self.encode_piece(piece))
        if max_length is not None:
            ids = ids[: max_length - 1]
        return [*ids, self.eos_id]


# ---------------------------------------------------------------------------
# Reading tokenizer.json
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AnimaTokenizerSource:
    qwen_path: Path
    t5_path: Path

    def qwen_json(self) -> dict[str, Any]:
        return json.loads(self.qwen_path.read_text(encoding="utf-8"))

    def t5_json(self) -> dict[str, Any]:
        return json.loads(self.t5_path.read_text(encoding="utf-8"))


def t5_charsmap(t5: dict[str, Any]) -> bytes:
    normalizer = t5["normalizer"]
    if normalizer.get("type") != "Precompiled":
        raise ValueError(f"the T5 normalizer is not Precompiled: {normalizer.get('type')}")
    return base64.b64decode(normalizer["precompiled_charsmap"])


def added_token_map(raw: dict[str, Any]) -> dict[str, int]:
    return {entry["content"]: entry["id"] for entry in raw["added_tokens"]}
