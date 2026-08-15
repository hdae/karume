r"""Anima のプロンプト用トークナイザ（Qwen2 BPE / T5 Unigram）を「引くだけ」の表へ畳む。

このモジュールが持つのは 3 つ:

  ① **表の抽出** — HF の `tokenizer.json` から、Deno 側が引くだけで済む形を作る。主眼は
     T5 の `Precompiled` 正規化器（SentencePiece の DARTS トライ）と、Qwen2 の
     pre_tokenizer 正規表現が使う `\p{L}` / `\p{N}` / `\s`、そして Qwen2 の `NFC`。
     **いずれも TS 側で Unicode 判定を再実装しない** — 判定の正本は Rust（`tokenizers` /
     その正規表現エンジン / `unicode-normalization`）側の Unicode 表で、JS エンジンの
     ICU 版とずれた瞬間に pre-token の切れ目や正規化結果が変わり、**例外も警告も出さずに
     id 列だけが別物**になる。ここが全コードポイントを実評価して閉区間表・写像表に畳む
     （`sbv2_demo.clean_text_ranges` と同じ規律）。

  ② **参照実装** — TS 実装の鏡像（`Qwen2Reference` / `T5Reference`）。①で焼いた表だけを
     入力に `AutoTokenizer` と同じ id 列を出す。これが緑であることが「表に必要な情報が
     揃っている」ことの機械証明で、TS 側はこの写経になる（突合は
     `anima/tests/test_demo.py`）。

  ③ **検証** — ①の畳み込みが正本と同値であることの網羅 + 乱択試験。畳めた根拠そのもの
     なので emit のたびに必ず通し、外れたら emit しない（ADR 0005 の fail loudly）。

## T5 `Precompiled` をどう畳んだか（実測に基づく）

`tokenizers` の Precompiled は SentencePiece の正規化とは別物で、実挙動はこう:

    書記素クラスタ g ごとに:
      g の UTF-8 長 < 6 かつ「g のいずれかの接頭辞」がトライにある
        → **最短**の一致接頭辞の値を出し、g の残りは捨てる
      さもなくば 1 コードポイントずつ引く（無ければその文字のまま）

実測の裏付け（いずれも `normalizers.Precompiled.normalize_str`）:
  `A`+U+0301+U+0301 → `Á`（3 文字目が消える）/ `A`+U+0302+U+0301 → `Â`
  （3cp 規則 `Ấ` ではなく 2cp 接頭辞が勝つ）/ U+1D2C+U+0300 → `A`（1cp 接頭辞が勝つ）。

ここから畳める形が決まる:

  * 効く規則は「真の接頭辞に規則を持たない、UTF-8 6 バイト未満」の鍵だけ。残りは到達不能。
  * クラスタ境界は UAX#29 そのものではなく「規則が発火しうる範囲」だけ効く。必要なのは
    (a) 直前に合流する文字 `extend` (b) 直後で必ず切れる文字 `break_after`（= GB4 の
    Control）(c) 後続を引き込む `prepend`（GB9b）(d) CRLF。いずれも**正本への探り針で
    全コードポイントを実評価**して閉区間に畳む（UAX#29 の表を TS に持ち込まない）。
  * ハングル / 地域表示子 / 絵文字 ZWJ 列は 1 組で 6 バイト以上になり丸ごと置換の経路に
    入らないため、境界規則を実装しなくても出力が変わらない（乱択試験で確認）。

## Qwen2 の `NFC` をどう畳んだか（実測に基づく）

`String.prototype.normalize("NFC")` / `unicodedata.normalize("NFC")` は**正本ではない**。
正本（`tokenizers.normalizers.NFC` = Rust の `unicode-normalization`）は Unicode 表が古く、
実測（2026-08-03）で **123 コードポイント**について両者の出力が割れる:

  * 120 cp は正本が結合クラス 0（starter）と見なす — ICU / `unicodedata` は ccc>0 なので
    canonical ordering で並べ替えるが、正本は並べ替えない（例 U+0818 U+089A）。
  * 3 cp（U+113C2 / U+1611E / U+16D67）は新しい canonical composition で、ICU /
    `unicodedata` は合成するが正本は合成しない。

どちらも「その cp のところで正規化が切れる」形に畳める: **分節 cp で文字列を切り、各節を
素の NFC に掛けて連結する。分節 cp 自身は正規化に参加させない**（starter として扱われる =
前後の並べ替えにも合成にも関わらない、が正本の挙動そのもの）。集合は正本への探り針で焼き、
同値は全コードポイント × 11 文脈 + 乱択で検査する（`verify_nfc_segments`）。
"""

from __future__ import annotations

import base64
import json
import struct
import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

Ranges = list[list[int]]

#: サロゲートを除く全コードポイント（Python str に載らない D800-DFFF は評価できない）。
ALL_CODEPOINTS = [cp for cp in range(0x110000) if not (0xD800 <= cp <= 0xDFFF)]

#: `Precompiled` が丸ごと置換を試みる上限（spm_precompiled の `grapheme.len() < 6`）。
CLUSTER_BYTE_LIMIT = 6

#: Unigram の未知文字ペナルティ（`tokenizers` の `K_UNK_PENALTY`）。
UNK_PENALTY = 10.0


def to_ranges(codepoints: Iterable[int]) -> Ranges:
    """昇順の閉区間リストへ畳む。"""
    out: Ranges = []
    for cp in sorted(codepoints):
        if out and out[-1][1] == cp - 1:
            out[-1][1] = cp
        else:
            out.append([cp, cp])
    return out


def in_ranges(ranges: Ranges, cp: int) -> bool:
    """閉区間表の二分探索（TS 側 `inCodeRanges` の鏡像）。"""
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
# ① 文字クラス（Qwen2 の pre_tokenizer 正規表現が使う \p{L} / \p{N} / \s）
# ---------------------------------------------------------------------------


def build_char_classes() -> dict[str, Ranges]:
    r"""正規表現エンジン（正本）そのものに 1 文字ずつ聞いて閉区間表に畳む。

    TS 側で `\p{L}` を使うと判定の正本が JS エンジンの ICU に移り、Rust 側の Unicode 表と
    ずれた瞬間に**別の pre-token 分割**になる（静かな id 列の不一致）。
    """
    from tokenizers import Regex, pre_tokenizers

    out: dict[str, Ranges] = {}
    for name, pattern in (("letter", r"\p{L}"), ("number", r"\p{N}"), ("space", r"\s")):
        splitter = pre_tokenizers.Split(Regex(pattern), behavior="removed", invert=False)
        out[name] = to_ranges(
            cp for cp in ALL_CODEPOINTS if splitter.pre_tokenize_str(chr(cp)) == []
        )
    return out


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


def build_case_fold() -> dict[int, int]:
    r"""`(?i:)` が短縮形の接尾辞文字と同一視するコードポイントを正本に聞いて畳む。

    **実測（2026-08-03）: Rust の `(?i:)` は ASCII 相当ではなく Unicode の simple case
    folding**。接尾辞に現れる 8 文字（s t r e v m l d）について全コードポイントを当てると、
    ASCII 大文字に加えて **U+017F（ſ）が s と同一視される**。

    ASCII の大小反転で済ませると `it'ſs` が `'ſs` の 1 断片になり（正本は `'ſ` + `s`）、
    pre-token の切れ目が変わる。この 1 件のためだけの表だが、**判定の正本を TS に持ち込ま
    ない**という規律はここでも同じ（`.lower()` / `toLowerCase()` はどちらも正本ではない）。

    返すのは「非自明な同一視」だけ（`cp == 対象文字` の自明な組は入れない）。
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
                    f"U+{cp:04X} が {chr(out[cp])!r} と {letter!r} の両方に畳まれる"
                    " — 接尾辞ごとに別の表が要る"
                )
    return out


# ---------------------------------------------------------------------------
# ① NFC（正本と素の NFC が割れるコードポイントの分節表）
# ---------------------------------------------------------------------------

#: 分節 cp を探す文脈。`{}` に 1 文字を入れる。並べ替え（ccc 差）は「後ろに ccc 1 の
#: U+0334 を置く」「前に ccc 230 の U+0301 を置く」の 2 方向で必ず表に出る（ccc 1 の cp は
#: 前者では動かない）。合成表の欠落は同じ文字を 2 つ並べる形で出る。
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

#: 分節モデルの網羅検査で使う文脈（検出用とは**別の**文脈を並べる — 探り針で拾った集合を
#: 探り針自身で確かめるだけにしない）。ハングル L/V・チベット・濁点・既存の合成を混ぜる。
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

#: NFC の乱択試験に必ず混ぜる文字（結合文字・ハングル・絵文字・異体字・新しい合成の材料）。
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
    """正本の `NFC` と素の `unicodedata` が割れるコードポイントを閉区間表に畳む。

    正本（Rust `unicode-normalization`）の Unicode 表は古く、結合クラスと合成表の両方が
    ICU / `unicodedata` と食い違う（モジュール docstring の「Qwen2 の NFC」節）。割れる cp を
    **正本に 1 文字ずつ聞いて**集め、TS / Python はその cp で文字列を分節する。
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
    """分節表を使った NFC（TS 実装 `normalizeNfc` の鏡像）。

    MUST: 分節 cp 自身は正規化に**参加させない**（前後どちらの節にも入れない）。正本が
    starter として扱う = 並べ替えにも合成にも関わらない、という挙動そのものを写す。
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
    """分節モデルが正本の `NFC` と同値であることを網羅 + 乱択で確かめる。

    ここが emit の門。分節表から 1 cp でも落ちれば網羅側が落ちるし、**分節という畳み方
    そのものが正本と合わない**場合（例: 正本が新しい合成規則を持っていた）もここで落ちる。
    seed は固定なので同じ表なら同じ列を検査する。
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
                    f"U+{cp:04X} の文脈 {probe!r}: NFC が正本と違う\n"
                    f"  正本={[hex(ord(c)) for c in want]}\n  表={[hex(ord(c)) for c in got]}"
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
                f"乱択列で NFC が正本と違う {[hex(ord(c)) for c in s]}: "
                f"正本={[hex(ord(c)) for c in want]} 表={[hex(ord(c)) for c in got]}"
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
    """TS 側が正本の NFC を再現できることを固定するための `[入力, 正本の出力]` 対。

    2 種類を混ぜる:
      ① 分節 cp ごとに 1 本 — **素の NFC では必ず外れる**文脈を選ぶ（表を引かない実装や
         表から 1 cp 落ちた実装の検出器になる）
      ② seed 固定の乱択 — 分節の外側で JS の ICU が `unicodedata` とずれた場合を捕まえる
         （emit の門は Python 実装しか見られないので、その穴をフィクスチャで塞ぐ）
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
            else:  # pragma: no cover - build_nfc_segments の後段なので到達しない
                raise ValueError(f"U+{cp:04X} が分節表にあるのに割れる文脈が無い")

    alphabet = [chr(c) for c in _NFC_FUZZ_SEEDS]
    alphabet += [chr(c) for start, end in segments for c in range(start, end + 1)]
    alphabet += [chr(c) for c in range(0x61, 0x7B)]
    rng = random.Random(seed)
    for _ in range(extra):
        text = "".join(rng.choice(alphabet) for _ in range(rng.randint(1, 10)))
        out.append([text, norm.normalize_str(text)])
    return out


#: pre-token 走査の網羅検査で使う文脈（選択肢を 1 つずつ踏ませる）。`{}` に 1 文字を入れる。
#: 後半 5 本は短縮形（選択肢①）専用。`'{}` だけでは**足りない** — 1 文字の探り針では①が
#: 発火してもしなくても同じ 1 断片になり、大小無視の取り違えが素通りする（実測でこの穴を
#: 踏んだ）。後ろに文字を足して「①が食った残り」が見える形にする。
_PRETOKEN_PROBES = (
    "a{}b",
    "'{}",
    " {}\n",
    "{}{}",
    "1{}",
    "!{}z",
    "'{}z",  # 1 文字接尾辞（s t m d）
    "'{}ez",  # 2 文字接尾辞の 1 文字目（re / ve）
    "'{}lz",  # 〃（ll）
    "'r{}z",  # 2 文字接尾辞の 2 文字目（re / ve の e）
    "'l{}z",  # 〃（ll の l）
)


def verify_qwen_pre_tokenize(
    pattern: str, classes: dict[str, Ranges], case_fold: dict[int, int]
) -> dict[str, Any]:
    """手書き走査が正本の Split と全コードポイント × 全文脈で一致することを確かめる。

    正規表現そのものを移植しない（TS に正規表現エンジンの Unicode 表を持ち込まない）以上、
    同値は実測でしか担保できない。ここが emit の門になる。
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
                    f"U+{cp:04X} の文脈 {probe!r}: 走査が正本と違う\n  正本={want}\n  走査={got}"
                )
            checked += 1
    return {"probes": len(_PRETOKEN_PROBES), "checked": checked}


# ---------------------------------------------------------------------------
# ② 参照実装 — Qwen2（byte-level BPE）
# ---------------------------------------------------------------------------


def bytes_to_unicode() -> dict[int, str]:
    """GPT-2 由来の byte → 可視文字表（アルゴリズムなので Unicode 版に依存しない）。

    **表を焼く必要が無い唯一の箇所**。TS 側も同じ手順で組み立てる。
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
    """追加語彙（特殊トークン）を leftmost-longest で切り出す。

    `tokenizers` の `AddedVocabulary` は AhoCorasick を LeftmostLongest で回すので、位置が
    同じなら長い方、位置が違えば左が勝つ。切り出した断片は正規化も pre-token 化も**通さず**
    id を直接出す（正規化より前に効く段なので、順序を入れ替えると別物になる）。
    """
    if any(token == "" for token in added):
        # 空文字は全位置で一致して `i` が進まない（無限ループ）。上流の added_tokens に
        # 空 content が入ることは考えにくいが、黙って回り続けるより落とす。
        raise ValueError("追加語彙に空文字がある — 切り出しが進まない")
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
    r"""Qwen2 の Split 正規表現を手書き走査で再現する。

    正本のパターン（leftmost・選択肢は上から順に最初に一致したものが勝つ）:
      ① `(?i:'s|'t|'re|'ve|'m|'ll|'d)`
      ② `[^\r\n\p{L}\p{N}]?\p{L}+`
      ③ `\p{N}`
      ④ ` ?[^\s\p{L}\p{N}]+[\r\n]*`
      ⑤ `\s*[\r\n]+`
      ⑥ `\s+(?!\S)`
      ⑦ `\s+`
    どの選択肢も空一致しないので、位置ごとに 1 つ選んで進むだけで find_iter と同値。
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
        # ① 短縮形（ASCII アポストロフィ + 大小無視の接尾辞）。大小無視は焼いた表で引く
        # （`.lower()` / `toLowerCase()` は正本ではない — `build_case_fold` の docstring）。
        if text[i] == "'":
            for suffix in _APOSTROPHE_SUFFIXES:
                if _matches_folded(text, i + 1, suffix, case_fold):
                    end = i + 1 + len(suffix)
                    break
        # ② 任意の 1 文字（CR/LF・字・数以外）+ 字の連続。任意部を先に食い、駄目なら戻る。
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
        # ③ 数字 1 文字（`+` が無いので必ず 1 文字ずつ切れる）
        if end < 0 and is_number(i):
            end = i + 1
        # ④ 任意の半角空白 1 文字 + その他の連続 + CR/LF の連続
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
        # ⑤ 空白の連続のうち、末尾が CR/LF になる最長のもの
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
        # ⑥ 空白の連続（非空白が続くなら最後の 1 文字を残す）/ ⑦ 空白の連続
        if end < 0 and is_space(i):
            k = i
            while k < n and is_space(k):
                k += 1
            end = k if k == n else max(i + 1, k - 1)
        if end <= i:
            raise ValueError(f"pre-token 走査が進まない（位置 {i}, U+{ord(text[i]):04X}）")
        out.append(text[i:end])
        i = end
    return out


def _matches_folded(text: str, at: int, suffix: str, case_fold: dict[int, int]) -> bool:
    """`text[at:]` が `suffix` に大小無視で一致するか（同一視は焼いた表だけで判定する）。"""
    if at + len(suffix) > len(text):
        return False
    for k, want in enumerate(suffix):
        cp = ord(text[at + k])
        if cp != ord(want) and case_fold.get(cp) != ord(want):
            return False
    return True


class Qwen2Reference:
    """Qwen2 トークナイザの参照実装（TS 実装の鏡像）。

    `used_vocab` / `used_merges` は fixture の語彙部分集合を作るための記録
    （151k / 151k を丸ごと commit しないため）。
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
        """post_processor が特殊トークンを足さないので、切り詰めは単純な先頭切り。"""
        ids: list[int] = []
        for chunk, is_added in split_added_tokens(text, sorted(self.added, key=len, reverse=True)):
            if is_added:
                ids.append(self.added[chunk])
                continue
            # MUST: 素の `unicodedata.normalize` を直に使わない。正本の NFC とは 123 cp で
            # 割れる（モジュール docstring の「Qwen2 の NFC」節）— 分節表で切ってから掛ける。
            normalized = nfc_with_segments(chunk, self.nfc_segments)
            for piece in qwen_pre_tokenize(normalized, self.classes, self.case_fold):
                encoded = "".join(self.byte_encoder[b] for b in piece.encode("utf-8"))
                for token in self.bpe(encoded):
                    if token not in self.vocab:
                        raise ValueError(f"Qwen2 語彙に無いトークン {token!r}")
                    self.used_vocab.add(token)
                    ids.append(self.vocab[token])
        return ids if max_length is None else ids[:max_length]


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
