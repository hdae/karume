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

#: `Precompiled` が丸ごと置換を試みる上限（spm_precompiled の `grapheme.len() < 6`）。
CLUSTER_BYTE_LIMIT = 6

#: Unigram の未知文字ペナルティ（`tokenizers` の `K_UNK_PENALTY`）。
UNK_PENALTY = 10.0

# ---------------------------------------------------------------------------
# ① DARTS トライの復号と規則の最小化
# ---------------------------------------------------------------------------


def decode_charsmap(blob: bytes) -> dict[str, str]:
    """`precompiled_charsmap` の Darts-clone 二重配列から全ての（鍵, 値）を取り出す。

    形式は spm_precompiled と同じ: u32 LE のトライバイト長 + トライ本体 + NUL 区切りの
    正規化文字列プール。単位の意味（has_leaf / value / label / offset）は Darts-clone 0.32
    に従う。
    """
    trie_size = struct.unpack_from("<I", blob, 0)[0]
    if trie_size % 4 != 0 or 4 + trie_size > len(blob):
        raise ValueError(f"charsmap のトライ長 {trie_size} が不正")
    units = struct.unpack(f"<{trie_size // 4}I", blob[4 : 4 + trie_size])
    pool = blob[4 + trie_size :]

    def read_pool(index: int) -> str:
        end = index
        while end < len(pool) and pool[end] != 0:
            end += 1
        return pool[index:end].decode("utf-8")

    out: dict[str, str] = {}
    # 再帰は最長鍵のバイト長しか深くならないが、明示スタックで回す。
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
    """`Precompiled` を TS が引くだけの形に畳んだもの。"""

    #: 1 コードポイント鍵の写像（鍵 = 1 文字）。
    single: dict[str, str]
    #: 2 コードポイント鍵の写像（真の接頭辞に規則が無いものだけ = 発火しうる分）。
    multi: dict[str, str]
    #: 直前の文字と同じクラスタに合流する文字（UAX#29 の Extend / ZWJ / SpacingMark 相当）。
    extend: Ranges
    #: この文字の直後で必ずクラスタが切れる（UAX#29 GB4 の Control 相当）。
    break_after: Ranges
    #: 後続をクラスタに引き込む文字（UAX#29 GB9b の Prepend 相当）。
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
    """charsmap と正本（`normalizers.Precompiled`）から表を作る。

    規則の最小化: 丸ごと置換は「最短の一致接頭辞」を採るので、真の接頭辞に規則を持つ鍵は
    永久に発火しない。加えてクラスタが 6 バイト以上だと丸ごと置換の経路自体が使われないので、
    6 バイト以上の鍵も落とす。
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
        raise ValueError("2 コードポイントを超える発火可能規則がある — 畳み方の前提が崩れた")

    # --- 探り針: 「規則が発火する 2cp クラスタ」に x を足して 6 バイト以上へ押し上げる。
    # x が同じクラスタに入れば丸ごと置換の経路が使えなくなり、規則が発火しないことで判る。
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
        raise ValueError(f"延長探り針が想定外の出力: {got!r}（probe={probe!r} x={x!r}）")

    extend: list[int] = []
    for cp in ALL_CODEPOINTS:
        x = chr(cp)
        verdict = joins(probe_short, x)
        if verdict is None:
            verdict = joins(probe_long, x)
        if verdict:
            extend.append(cp)

    # --- prepend: c + "o" + U+0303 で õ が合成されるか。合成されなければ c が後続を
    # 引き込んでクラスタ化している（= 丸ごと置換の経路が使えず規則が不発火）。
    prepend_tail = "o" + chr(0x0303)
    prepend: list[int] = []
    for cp in ALL_CODEPOINTS:
        c = chr(cp)
        head = single.get(c, c)
        got = norm.normalize_str(c + prepend_tail)
        if got == head + prepend_tail:
            prepend.append(cp)
        elif got != head + chr(0x00F5):
            raise ValueError(f"prepend 探り針が想定外の出力: {got!r}（U+{cp:04X}）")

    # --- break_after: 「延長かつ規則あり」の U+0EB3 と「延長かつ規則なし」の U+0337 を
    # 後ろに置く。切れれば [U+0EB3, U+0337] が 5 バイトのクラスタになり U+0337 が捨てられ、
    # 切れなければ合計 6 バイト以上で 1 文字ずつの経路になり U+0337 が残る。
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
            raise ValueError(f"break-after 探り針が想定外の出力: {got!r}（U+{cp:04X}）")

    return SpmTables(
        single=single,
        multi=multi,
        extend=to_ranges(extend),
        break_after=to_ranges(break_after),
        prepend=to_ranges(prepend),
    )


def normalize_with_tables(tables: SpmTables, text: str) -> str:
    """焼いた表だけで `Precompiled` を再現する（TS 実装 `normalizeSpm` の鏡像）。"""
    out: list[str] = []
    i, n = 0, len(text)
    while i < n:
        j = i
        while j < n and in_ranges(tables.prepend, ord(text[j])):
            j += 1
        # prepend の直後が制御文字ならそこで切れる（引き込まない）。
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
            # 最短の一致接頭辞が勝つ。残りは捨てられる。
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


#: 乱択試験に必ず混ぜる文字（ハングル字母 / 地域表示子 / 肌色 / 異体字 / タイ / インド系）。
#: 「6 バイト境界のおかげで境界規則を実装せずに済んでいる」側を実際に踏ませるための種。
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
    """畳み込みが正本と同値であることを網羅 + 乱択で確かめる。

    (a) 全コードポイント単体 (b) 発火規則の鍵そのもの (c) 表由来の文字を混ぜた乱択列。
    seed は固定なので、同じ表なら同じ列を検査する（再現しない検証にしない）。
    """
    import random

    from tokenizers import normalizers

    norm = normalizers.Precompiled(blob)

    for cp in ALL_CODEPOINTS:
        c = chr(cp)
        got, want = norm.normalize_str(c), normalize_with_tables(tables, c)
        if got != want:
            raise ValueError(f"単一 cp 不一致 U+{cp:04X}: 正本={got!r} 表={want!r}")

    for key in list(tables.single) + list(tables.multi):
        got, want = norm.normalize_str(key), normalize_with_tables(tables, key)
        if got != want:
            raise ValueError(f"規則鍵で不一致 {key!r}: 正本={got!r} 表={want!r}")

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
                f"乱択列で不一致 {[hex(ord(c)) for c in s]}: "
                f"正本={[hex(ord(c)) for c in got]} 表={[hex(ord(c)) for c in want]}"
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
    r"""`\s` が `WhitespaceSplit`（T5 側の分割）の空白集合と一致することを確かめる。

    一致するなら T5 側は同じ表を使い回せる。ずれたら別の表が要るので落とす。
    """
    from tokenizers import pre_tokenizers

    ws = pre_tokenizers.WhitespaceSplit()
    for cp in ALL_CODEPOINTS:
        splits = len(ws.pre_tokenize_str("a" + chr(cp) + "b")) > 1
        if splits != in_ranges(classes["space"], cp):
            raise ValueError(f"U+{cp:04X}: WhitespaceSplit と \\s の判定が食い違う")
    return {name: len(r) for name, r in classes.items()}


# ---------------------------------------------------------------------------
# ② 参照実装 — T5（Unigram + Metaspace）
# ---------------------------------------------------------------------------

METASPACE = "▁"


def t5_pre_tokenize(text: str, space: Ranges) -> list[str]:
    """WhitespaceSplit → Metaspace（replacement=▁ / prepend_scheme=always / split）。

    `split=true` は MergedWithNext なので、区切りの `▁` は**次の**断片の先頭に付く。
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
    """T5（Unigram）トークナイザの参照実装（TS 実装の鏡像）。"""

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
        """`tokenizers` の `Lattice::viterbi` をコードポイント位置で写す。

        同点は「end_nodes に先に積まれた方」が勝つ。積む順は begin 昇順（= 同じ位置で終わる
        なら長い方が先）なので、同点では長い断片が採られる — 比較は**厳密な `>`** でなければ
        ならない（`>=` にすると短い方が勝ち、稀に別の分割になって id 列が静かに変わる）。
        """
        n = len(piece)
        unk_score = self.min_score - UNK_PENALTY
        # begin[pos] = そこから始まるノードの長さ列（挿入順 = 長さ昇順）
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
        # end[e] = そこで終わるノード（開始位置, スロット）。begin 昇順に積む。
        end: list[list[tuple[int, int]]] = [[] for _ in range(n + 1)]
        for pos in range(n):
            for slot, length in enumerate(begin[pos]):
                end[pos + length].append((pos, slot))

        best: list[list[float]] = [[0.0] * len(begin[p]) for p in range(n)]
        prev: list[list[tuple[int, int] | None]] = [[None] * len(begin[p]) for p in range(n)]

        def best_left(pos: int) -> tuple[float, tuple[int, int] | None] | None:
            """pos で終わる最良ノード。pos=0 は bos（スコア 0）。到達不能なら None。"""
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
        """連続する未知ノードは 1 トークンに融合される（`fuse_unk`）。

        NOTE: `tokenizer.json` の `fuse_unk` は `null` だが、`tokenizers` の Unigram は
        **未指定でも融合する**（Rust 側の既定）。融合しない実装にすると日本語プロンプトで
        unk が 1 文字ずつ並び、`AutoTokenizer` との突合（`anima/tests/test_demo.py`）が
        `japanese` ケースで落ちる。
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
        """MUST: 切り詰めは `</s>` を足す**前**（正本の truncation は post_processor の前）。

        `max_length` の後に足すと 513 個になる。結果として T5 の id 列は常に `</s>` で
        終わり、長さは必ず 1 以上。
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
# tokenizer.json の読み出し
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
        raise ValueError(f"T5 の normalizer が Precompiled でない: {normalizer.get('type')}")
    return base64.b64decode(normalizer["precompiled_charsmap"])


def added_token_map(raw: dict[str, Any]) -> dict[str, int]:
    return {entry["content"]: entry["id"] for entry in raw["added_tokens"]}
