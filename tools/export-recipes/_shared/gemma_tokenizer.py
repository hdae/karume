"""Compile the SPM-BPE tokenizer of the Gemma line (gemma4 / EmbeddingGemma) into an asset
(ADR 0084 decision 1; a helper shared by several families, hence `_shared` — ADR 0065 decision 2).

It does three things:

  1. **compile** — read the upstream `tokenizer.json` (32.2MB) and fold it into a small schema
     that the Deno side only has to look up. **An unknown configuration fails at compile time**
     ({@link assert_supported}) — not scattering the checks over run time is the very reason for
     this shape (ADR 0084, considered alternative b). The folded form holds only "vocab rows in
     id order / merges as id pairs / 256 byteIds / added tokens / the set of special ids"; none
     of the baked tables Qwen2 needed (`code-ranges` / `caseFold` / `nfcSegments`) is needed —
     Gemma's normalizer is a single `Replace(" " → "▁")` and its pre_tokenizer is only
     `Split(" ", MergedWithPrevious)`, so neither touches Unicode classes or NFC (ADR 0084
     Context 2).

  2. **parity fixture** — the expected values of the TS implementation. **They are collected
     only through a path that calls `tokenizers.Tokenizer` independently**, never through the
     folding of (1) or any Python-side preprocessing helper (ADR 0084 decision 7; the ADR 0048
     addendum = sharing the preprocessing between TS and Python makes parity vacuous).

  3. **asset subset** — the fixture is tracked in git, so the 262,144-row vocabulary and the
     514,906 merges cannot be stored whole. It is narrowed to the entries that are **substrings**
     of the case texts ({@link subset_keys}). A wrong narrowing never errs on the unsafe side:
     the expected values always come from upstream (the full asset), so a subset that is too
     small makes the TS-side comparison **fail**.

MUST: family-specific spellings (where the inputs live, series names, cases) are **held by the
caller (the family recipe)**. This module knows only Gemma's tokenizer configuration, and the
same functions serve both gemma4 and EmbeddingGemma unchanged (the split that actually makes
ADR 0084 decision 6's "one implementation, two assets" hold).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from _shared.paths import REPO_ROOT

#: The asset schema version (a field so the Deno side never silently reads a version it does
#: not know).
ASSET_FORMAT = "karume-gemma-tokenizer/1"

#: SentencePiece's whitespace replacement character (U+2581). Used by both the normalizer and
#: the decoder.
METASPACE = "▁"

#: The accepted upstream configuration (**exact-match** — one differing field means a different
#: tokenizer).
EXPECTED_NORMALIZER = {"type": "Replace", "pattern": {"String": " "}, "content": METASPACE}
EXPECTED_PRE_TOKENIZER = {
    "type": "Split",
    "pattern": {"String": " "},
    "behavior": "MergedWithPrevious",
    "invert": False,
}
EXPECTED_DECODER = {
    "type": "Sequence",
    "decoders": [
        {"type": "Replace", "pattern": {"String": METASPACE}, "content": " "},
        {"type": "ByteFallback"},
        {"type": "Fuse"},
    ],
}

#: The spellings the `spec` field takes (the Deno side exact-matches the same strings).
SPEC_NORMALIZER = "replace-space-with-metaspace"
SPEC_PRE_TOKENIZER = "split-space-merged-with-previous"
SPEC_DECODER = "metaspace-byte-fallback-fuse"

#: The two post_processor shapes (measured on the real assets — gemma4 passes through /
#: EmbeddingGemma wraps in bos…eos).
POST_NONE = "none"
POST_BOS_EOS = "bos-eos"

#: Spellings of the special tokens (common to both assets).
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"


def byte_token(byte: int) -> str:
    """The spelling of a byte_fallback token (same as `tokenizers`' `format!("<{:#04X}>")`)."""
    return f"<0x{byte:02X}>"


class UnsupportedTokenizerError(ValueError):
    """The upstream `tokenizer.json` is outside the accepted set (fails at compile time)."""


@dataclass(frozen=True)
class CompiledTokenizer:
    """The folded result (the shape just before it is written as JSON)."""

    #: The vocabulary in id order (row number = id).
    vocab: list[str]
    #: Merges as `(left id, right id)`. The order is the rank itself.
    merges: list[tuple[int, int]]
    #: The ids for bytes 0..255 (**all 256 explicit** — the MUST of ADR 0084 decision 1).
    byte_ids: list[int]
    #: Added tokens `(spelling, id)` (split off before normalization).
    added_tokens: list[tuple[str, int]]
    #: Ids of the added tokens with `special: true` (skipped by decode).
    special_ids: list[int]
    unk_id: int
    bos_id: int
    eos_id: int
    #: The post_processor shape ({@link POST_NONE} / {@link POST_BOS_EOS}).
    post_processor: str


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise UnsupportedTokenizerError(message)


def assert_supported(raw: Mapping[str, Any]) -> str:
    """Check that the upstream `tokenizer.json` is in the accepted set; return the post_processor
    shape.

    MUST: **compare by exact match** (not only whether a field is present). A normalizer that
    differs by one character changes how metaspace is inserted, and a different pre_tokenizer
    changes the units handed to BPE — neither raises; only the id sequence silently changes.
    """
    model = raw.get("model")
    _require(isinstance(model, Mapping), "no model field")
    _require(model.get("type") == "BPE", f"model.type is not BPE ({model.get('type')!r})")
    _require(model.get("dropout") is None, "model.dropout is not null (training-time dropout)")
    _require(
        model.get("continuing_subword_prefix") is None,
        "model.continuing_subword_prefix is not null",
    )
    _require(model.get("end_of_word_suffix") is None, "model.end_of_word_suffix is not null")
    _require(model.get("unk_token") == UNK_TOKEN, f"model.unk_token is not {UNK_TOKEN!r}")
    _require(model.get("byte_fallback") is True, "model.byte_fallback is not true")
    _require(model.get("fuse_unk") is True, "model.fuse_unk is not true")
    _require(model.get("ignore_merges") is False, "model.ignore_merges is not false")
    _require(raw.get("truncation") is None, "truncation is not null")
    _require(raw.get("padding") is None, "padding is not null")
    _require(
        raw.get("normalizer") == EXPECTED_NORMALIZER,
        f"normalizer is not {EXPECTED_NORMALIZER} ({raw.get('normalizer')!r})",
    )
    _require(
        raw.get("pre_tokenizer") == EXPECTED_PRE_TOKENIZER,
        f"pre_tokenizer is not {EXPECTED_PRE_TOKENIZER} ({raw.get('pre_tokenizer')!r})",
    )
    _require(
        raw.get("decoder") == EXPECTED_DECODER,
        f"decoder is not {EXPECTED_DECODER} ({raw.get('decoder')!r})",
    )
    return _post_processor_kind(raw.get("post_processor"))


def _post_processor_kind(post: Any) -> str:
    """Classify the post_processor as one of the two shapes (anything else fails).

    Only `single` is examined — the path that builds pairs (`pair`) is outside this port's scope,
    and a different spelling there does not change the behaviour of single (the accepted set is
    not narrowed needlessly).
    """
    _require(isinstance(post, Mapping), "no post_processor")
    _require(
        post.get("type") == "TemplateProcessing",
        f"post_processor.type is not TemplateProcessing ({post.get('type')!r})",
    )
    sequence_a = {"Sequence": {"id": "A", "type_id": 0}}
    single = post.get("single")
    if single == [sequence_a]:
        return POST_NONE
    wrapped = [
        {"SpecialToken": {"id": BOS_TOKEN, "type_id": 0}},
        sequence_a,
        {"SpecialToken": {"id": EOS_TOKEN, "type_id": 0}},
    ]
    if single == wrapped:
        return POST_BOS_EOS
    raise UnsupportedTokenizerError(f"post_processor.single has an unknown shape ({single!r})")


def compile_tokenizer(raw: Mapping[str, Any]) -> CompiledTokenizer:
    """Fold the upstream `tokenizer.json` (with the accepted-set check)."""
    post_processor = assert_supported(raw)
    model = raw["model"]
    vocab_map: Mapping[str, int] = model["vocab"]
    _require(len(set(vocab_map.values())) == len(vocab_map), "duplicate ids in the vocabulary")
    vocab = [""] * len(vocab_map)
    for token, token_id in vocab_map.items():
        _require(
            isinstance(token_id, int) and 0 <= token_id < len(vocab_map),
            f"token {token!r} has id {token_id!r} outside 0..{len(vocab_map) - 1}",
        )
        vocab[token_id] = token
    # MUST: look up the 256 bytes **explicitly**. In the real assets `<0x00>`..`<0xFF>` are
    # consecutive, but that is a measured fact about these assets, not a schema guarantee
    # (ADR 0084 decision 1).
    byte_ids: list[int] = []
    for byte in range(256):
        spelling = byte_token(byte)
        if spelling not in vocab_map:
            raise UnsupportedTokenizerError(
                f"byte_fallback token {spelling!r} is not in the vocabulary"
            )
        byte_ids.append(vocab_map[spelling])
    _require(len(set(byte_ids)) == 256, "duplicate ids among the byte_fallback tokens")
    # The decode side decides a byte run by "is the id in byteIds". The upstream `ByteFallback`
    # decoder decides by **the shape of the spelling**, so unless compile checks that exactly 256
    # tokens have that shape, the two decisions can disagree.
    shaped = [t for t in vocab_map if len(t) == 6 and t.startswith("<0x") and t.endswith(">")]
    _require(
        len(shaped) == 256,
        f"the vocabulary does not have 256 tokens of the form `<0xHH>` ({len(shaped)} — the"
        " byte decision of decode would split)",
    )

    merges: list[tuple[int, int]] = []
    seen_pairs: set[tuple[int, int]] = set()
    for entry in model["merges"]:
        # The old format's `"left right"` strings are not accepted (Gemma's tokens contain no
        # space, so they look splittable, but since some spellings contain a newline the
        # "line and pair" correspondence would be a different thing).
        _require(
            isinstance(entry, list) and len(entry) == 2,
            f"a merge entry is not [left, right] ({entry!r})",
        )
        left, right = entry
        for side in (left, right, left + right):
            if side not in vocab_map:
                raise UnsupportedTokenizerError(f"merge part {side!r} is not in the vocabulary")
        pair = (vocab_map[left], vocab_map[right])
        # MUST: do not let the later entry win. Upstream is a HashMap, so a duplicate pair
        # silently keeps only one and only the split rule changes (same reason as `setUnique` in
        # `asset-gates.ts`).
        _require(pair not in seen_pairs, f"merge pair {(left, right)!r} is duplicated")
        seen_pairs.add(pair)
        merges.append(pair)

    added_tokens: list[tuple[str, int]] = []
    special_ids: list[int] = []
    for entry in raw.get("added_tokens", []):
        for flag in ("single_word", "lstrip", "rstrip", "normalized"):
            _require(
                entry.get(flag) is False,
                f"added token {entry.get('content')!r} has {flag} not false"
                " (the split-off rule would be different)",
            )
        content = entry["content"]
        added_tokens.append((content, entry["id"]))
        if entry.get("special") is True:
            special_ids.append(entry["id"])
    _require(
        len({token for token, _ in added_tokens}) == len(added_tokens),
        "duplicate spellings among the added tokens",
    )

    ids_by_token = dict(added_tokens)
    for name in (UNK_TOKEN, BOS_TOKEN, EOS_TOKEN):
        _require(name in vocab_map, f"{name!r} is not in the vocabulary")
        if name in ids_by_token:
            _require(
                ids_by_token[name] == vocab_map[name],
                f"the added-token id and the vocabulary id of {name!r} disagree",
            )
    return CompiledTokenizer(
        vocab=vocab,
        merges=merges,
        byte_ids=byte_ids,
        added_tokens=added_tokens,
        special_ids=special_ids,
        unk_id=vocab_map[UNK_TOKEN],
        bos_id=vocab_map[BOS_TOKEN],
        eos_id=vocab_map[EOS_TOKEN],
        post_processor=post_processor,
    )


def spec_of(compiled: CompiledTokenizer) -> dict[str, str]:
    """The configuration declaration the Deno side exact-matches (the folded asset does not
    carry the upstream JSON)."""
    return {
        "normalizer": SPEC_NORMALIZER,
        "preTokenizer": SPEC_PRE_TOKENIZER,
        "decoder": SPEC_DECODER,
        "postProcessor": compiled.post_processor,
    }


def asset_payload(compiled: CompiledTokenizer, *, source: Mapping[str, Any]) -> dict[str, Any]:
    """The distribution JSON (`mergesText`: row number = rank; `vocab`: row number = id).

    Merges are held as **id pairs** rather than spellings because Gemma's vocabulary contains
    newlines (31 tokens, measured) — unlike Qwen2's byte-level vocabulary it cannot be folded
    into "`left right`" lines.
    """
    return {
        "format": ASSET_FORMAT,
        "source": dict(source),
        "spec": spec_of(compiled),
        "vocab": compiled.vocab,
        "mergesText": "\n".join(f"{left} {right}" for left, right in compiled.merges),
        "mergesCount": len(compiled.merges),
        "byteIds": compiled.byte_ids,
        "addedTokens": [[token, token_id] for token, token_id in compiled.added_tokens],
        "specialIds": compiled.special_ids,
        "unkId": compiled.unk_id,
        "bosId": compiled.bos_id,
        "eosId": compiled.eos_id,
    }


def file_digest(path: Path) -> dict[str, Any]:
    """`{bytes, sha256}` of `path` (provenance record)."""
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
            size += len(chunk)
    return {"bytes": size, "sha256": digest.hexdigest()}


def write_json(path: Path, payload: Mapping[str, Any]) -> int:
    """Write UTF-8 JSON and return its byte count (creates the parent directory)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    path.write_text(text, encoding="utf-8")
    return len(text.encode("utf-8"))


# ---------------------------------------------------------------------------
# parity fixture (expected values are taken independently from upstream `tokenizers`)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EncodeCase:
    """An encode case (`why` states in one line "the boundary where a naive port fails")."""

    name: str
    why: str
    text: str


@dataclass(frozen=True)
class DecodeCase:
    """A decode case (drives the id sequence directly — to hit byte runs encoding cannot
    produce)."""

    name: str
    why: str
    ids: list[int]


#: Encode cases run on both assets. Designed so that **each case holds one boundary where a
#: naive port fails**; invisible characters are always written as \uXXXX (they vanish or get
#: replaced while editing). Running the same battery on two assets is itself the proof of "one
#: implementation, two assets" (ADR 0084 decision 6).
SHARED_ENCODE_CASES: tuple[EncodeCase, ...] = (
    EncodeCase("english", "普通の英文 — 空白が metaspace へ落ちる基本形", "The quick brown fox."),
    EncodeCase(
        "english-long",
        "空白があっても pre_tokenizer は正規化後に切れ目を見つけられない（1 pre-token が全長）"
        " — merge の rank / 位置の同点が多発する長さ",
        "Machine learning models trained on large corpora can generate text that is "
        "surprisingly coherent, but they still fail at arithmetic and long-horizon planning.",
    ),
    EncodeCase(
        "japanese-long",
        "空白の無い長文 — 1 pre-token が入力全長になり merge queue が要る（ADR 0084 決定 3）",
        "むかしむかし、あるところにおじいさんとおばあさんが住んでいました。"
        "おじいさんは山へしばかりに、おばあさんは川へせんたくに行きました。"
        "おばあさんが川でせんたくをしていると、大きな桃がどんぶらこどんぶらこと"
        "流れてきました。おばあさんはその桃を家に持ち帰り、おじいさんと二人で"
        "切ってみると、中から元気な男の子が飛び出しました。二人はたいそう喜んで、"
        "その子を桃太郎と名づけ、たいせつに育てることにしたのでした。",
    ),
    EncodeCase(
        "cyrillic",
        "非ラテン・非 CJK（語彙内トークンと byte fallback の境目が別の位置に来る）",
        "Большая часть текста написана на русском языке.",
    ),
    EncodeCase("leading-space", "先頭の空白（metaspace が先頭に来る）", " leading"),
    EncodeCase("trailing-space", "末尾の空白（merge の右端が metaspace）", "trailing "),
    EncodeCase("double-space", "空白の連続（▁▁ 系の語彙へ畳まれる）", "a  b   c"),
    EncodeCase(
        "whitespace-run",
        "空白 30 連 — 長い metaspace 語彙（merges 表の先頭付近の rank）を踏む",
        "x" + " " * 30 + "y",
    ),
    EncodeCase(
        "tab-newline",
        "normalizer が触らない空白類（\\t / \\n は metaspace にならない）",
        "a\tb\nc\r\nd",
    ),
    EncodeCase(
        "newline-run",
        "改行 30 連 — 語彙に改行を含む綴りがある（行区切り資産が使えない証）",
        "\n" * 30,
    ),
    EncodeCase("digits", "数字列 — SPM 系は 1 桁ずつ切れる語彙を持つ", "1girl 2024 3.14159 100%"),
    EncodeCase(
        "url",
        "空白の無い長い ASCII（URL / base64 も 1 pre-token 全長になる）",
        "https://example.com/a/b/c?q=1&r=2#frag",
    ),
    EncodeCase("mixed", "日英混在 — 語彙の切り替わり", "1girl, 猫耳, ロングヘア, masterpiece"),
    EncodeCase(
        "emoji",
        "非 BMP + 地域表示子 + 異体字セレクタ（byte_fallback を踏む）",
        "a cat \U0001f408 and a flag \U0001f1ef\U0001f1f5 and \u2764\ufe0f love",
    ),
    EncodeCase(
        "zwj",
        "ZWJ 連結（1 書記素が複数コードポイント — 記号は**コードポイント**単位で始まる）",
        "family \U0001f468\u200d\U0001f469\u200d\U0001f466 photo",
    ),
    EncodeCase(
        "combining",
        "結合文字 — NFC を掛けない（掛けると id 列が変わる）",
        "A\u0301 vs \u00c1 and A\u0302\u0301",
    ),
    EncodeCase("thai", "結合文字を持つタイ語（語彙内と byte fallback が混ざる）", "กำลังทดสอบ"),
    EncodeCase(
        "control-bytes",
        "制御文字 — 語彙に無く byte_fallback へ落ちる（複数バイト run も含む）",
        "a\x00b\x1fcd",
    ),
    EncodeCase(
        "metaspace-literal",
        "入力に U+2581 が直接ある（正規化の出力と同じ文字が最初から入っている形）",
        "raw ▁ metaspace ▁▁ run",
    ),
    EncodeCase("empty", "空文字（記号が 0 個 — 走査の初期状態）", ""),
    EncodeCase("single-space", "空白 1 文字だけ（正規化後は metaspace 1 文字）", " "),
)


def subset_keys(texts: Iterable[str], *, max_token_chars: int) -> set[str]:
    """Build the key set used to cut a **subset** of the vocabulary and merges out of the case
    texts.

    At any point a BPE symbol is "a contiguous substring of the normalized string", so collecting
    every substring of up to `max_token_chars` characters is guaranteed to cover every symbol and
    pair that can appear during the scan (only byte_fallback symbols are not substrings, but
    {@link assert_no_byte_merges} checks that the real assets have no merge containing a byte
    token).

    NOTE: the only normalization applied here is "space → metaspace". Using the full text before
    the added tokens are split off is **deliberate**: every fragment after the split is a
    substring of this string (Replace substitutes per character) — leaning the cover to the wide
    side means a wrong narrowing can only show up as "the fixture is too small and the TS
    comparison fails".
    """
    keys: set[str] = set()
    for text in texts:
        chars = list(text.replace(" ", METASPACE))
        for start in range(len(chars)):
            piece = ""
            for length in range(1, min(max_token_chars, len(chars) - start) + 1):
                piece += chars[start + length - 1]
                keys.add(piece)
    return keys


def assert_no_byte_merges(compiled: CompiledTokenizer) -> None:
    """Check that no merge contains a byte_fallback token (the premise of the subset cover)."""
    byte_id_set = set(compiled.byte_ids)
    for left, right in compiled.merges:
        if left in byte_id_set or right in byte_id_set:
            raise UnsupportedTokenizerError(
                "a merge contains a byte_fallback token — the fixture subset cannot cover it",
            )


def asset_subset(
    compiled: CompiledTokenizer,
    *,
    texts: Sequence[str],
    required_ids: Iterable[int],
) -> dict[str, Any]:
    """Build the subset of the vocabulary and merges stored in the fixture.

    NOTE: it is a public name because the chat fixture (`gemma4/chat.py`) cuts the same way —
    with two implementations of the narrowing, one of them producing "a subset that is too small"
    would go unnoticed.
    """
    assert_no_byte_merges(compiled)
    max_token_chars = max(len(list(token)) for token in compiled.vocab)
    keys = subset_keys(texts, max_token_chars=max_token_chars)
    ids_by_token = {token: index for index, token in enumerate(compiled.vocab)}

    required_kept = set(required_ids)
    kept: set[int] = set(compiled.byte_ids)
    kept.add(compiled.unk_id)
    kept.add(compiled.bos_id)
    kept.add(compiled.eos_id)
    for token in keys:
        if token in ids_by_token:
            kept.add(ids_by_token[token])
    for token_id in required_kept:
        if token_id < len(compiled.vocab):
            kept.add(token_id)

    merges: list[list[int]] = []
    for rank, (left, right) in enumerate(compiled.merges):
        joined = compiled.vocab[left] + compiled.vocab[right]
        if joined in keys:
            merges.append([left, right, rank])
            kept.update((left, right, ids_by_token[joined]))

    # Narrow the added tokens to "those that can appear in the case texts". Since a dropped
    # spelling is not in the texts, the leftmost-longest split result does not change.
    added = [
        [token, token_id]
        for token, token_id in compiled.added_tokens
        if token in keys or token_id in required_kept
    ]
    special = set(compiled.special_ids)
    return {
        "vocab": sorted([token_id, compiled.vocab[token_id]] for token_id in kept),
        "merges": merges,
        "byteIds": compiled.byte_ids,
        "addedTokens": added,
        "specialIds": sorted(token_id for _, token_id in added if token_id in special),
        "unkId": compiled.unk_id,
        "bosId": compiled.bos_id,
        "eosId": compiled.eos_id,
    }


def build_fixture(
    *,
    tokenizer_json: Path,
    compiled: CompiledTokenizer,
    encode_cases: Sequence[EncodeCase],
    decode_cases: Sequence[DecodeCase],
    source: Mapping[str, Any],
) -> dict[str, Any]:
    """Take the TS implementation's expected values independently from upstream (ADR 0084
    decision 7).

    MUST: **not one line of this module's folding goes into the expected-value path** — if it
    did, "a wrong folding makes both sides wrong by the same amount" and the comparison becomes
    vacuous (the real case in the ADR 0048 addendum).
    """
    from tokenizers import Tokenizer  # lazy import (a dependency compile alone does not need)

    reference = Tokenizer.from_file(str(tokenizer_json))

    encode: list[dict[str, Any]] = []
    for case in encode_cases:
        encode.append(
            {
                "name": case.name,
                "why": case.why,
                "text": case.text,
                "ids": reference.encode(case.text, add_special_tokens=False).ids,
                "idsWithSpecials": reference.encode(case.text, add_special_tokens=True).ids,
            }
        )
    # Round trips (the encode output fed straight to decode) are added automatically — copying
    # them by hand lets one side go stale.
    all_decode = [
        DecodeCase(
            name=f"roundtrip-{row['name']}",
            why=f"符号化 {row['name']} の出力（特殊トークン込み）をそのまま復号する",
            ids=list(row["idsWithSpecials"]),
        )
        for row in encode
    ] + list(decode_cases)
    decode: list[dict[str, Any]] = []
    for case in all_decode:
        decode.append(
            {
                "name": case.name,
                "why": case.why,
                "ids": case.ids,
                "text": reference.decode(case.ids, skip_special_tokens=False),
                "textSkipSpecials": reference.decode(case.ids, skip_special_tokens=True),
            }
        )

    required = {token_id for row in encode for token_id in row["idsWithSpecials"]}
    required.update(token_id for row in decode for token_id in row["ids"])
    return {
        "source": dict(source),
        "spec": spec_of(compiled),
        "asset": asset_subset(
            compiled,
            texts=[case.text for case in encode_cases],
            required_ids=required,
        ),
        "encode": encode,
        "decode": decode,
    }


def emit(
    *,
    tokenizer_json: Path,
    asset_path: Path,
    fixture_path: Path,
    extra_cases: Sequence[EncodeCase] = (),
    source_path: str | None = None,
) -> None:
    """Compile one asset and collect its fixture — **always both**.

    There is no entry point that emits only one, because if the runtime asset and the fixture go
    stale separately, "the tests are green but the run alone produces different ids" (same reason
    as `anima/demo.py`).

    `source_path` names a `tokenizer.json` outside the repository (a hub snapshot) the way the
    git-tracked fixture may record it (e.g. `<repo>@<revision>/tokenizer.json`); by default the
    path is the repository-relative one.
    """
    if not tokenizer_json.is_file():
        raise SystemExit(f"[karume] no upstream tokenizer.json: {tokenizer_json}")
    raw = json.loads(tokenizer_json.read_text(encoding="utf-8"))
    compiled = compile_tokenizer(raw)
    source = {
        # The fixture is tracked in git, so the source is written repository-relative (never
        # commit an absolute path).
        "path": source_path
        if source_path is not None
        else str(tokenizer_json.relative_to(REPO_ROOT)),
        **file_digest(tokenizer_json),
        "vocabCount": len(compiled.vocab),
        "mergesCount": len(compiled.merges),
        "addedCount": len(compiled.added_tokens),
    }
    # Write only after both are built — so that a failure of the upstream collection
    # (`build_fixture`) does not leave a new asset next to an old fixture.
    payload = asset_payload(compiled, source=source)
    fixture = build_fixture(
        tokenizer_json=tokenizer_json,
        compiled=compiled,
        encode_cases=[*SHARED_ENCODE_CASES, *extra_cases],
        decode_cases=byte_run_decode_cases(compiled),
        source=source,
    )
    asset_bytes = write_json(asset_path, payload)
    fixture_bytes = write_json(fixture_path, fixture)
    print(f"[karume] asset   {asset_path} ({asset_bytes / 1e6:.1f} MB)")
    print(
        f"[karume] fixture {fixture_path} ({fixture_bytes / 1e3:.0f} KB;"
        f" vocab {len(fixture['asset']['vocab'])} / merges {len(fixture['asset']['merges'])} / "
        f"encode {len(fixture['encode'])} / decode {len(fixture['decode'])} cases)"
    )
    # The formatter decides the committed form (`vp check` formats the fixtures too).
    print(
        f"[karume] then format it from the repository root: `vp fmt"
        f" {fixture_path.relative_to(REPO_ROOT)}`"
    )


def byte_run_decode_cases(compiled: CompiledTokenizer) -> list[DecodeCase]:
    """Decode cases that drive byte runs directly (including shapes encoding cannot produce)."""
    byte_ids = compiled.byte_ids
    return [
        DecodeCase(
            name="byte-all-256",
            why="byte 0..255 を 1 続きの run で流す（全体が不正 UTF-8 = 1 バイト 1 置換文字）",
            ids=list(byte_ids),
        ),
        DecodeCase(
            name="byte-run-truncated",
            why="3 バイト文字の頭 2 バイトだけ（不完全 run — run 全体が置換文字になる）",
            ids=[byte_ids[0xE3], byte_ids[0x81]],
        ),
        DecodeCase(
            name="byte-run-valid",
            why="3 バイト文字がちょうど揃う run（1 文字に畳まれる）",
            ids=[byte_ids[0xE3], byte_ids[0x81], byte_ids[0x82]],
        ),
        DecodeCase(
            name="byte-run-then-token",
            why="不完全 run の直後に非 byte トークン（run の切れ目が確定する位置）",
            ids=[byte_ids[0xE3], byte_ids[0x81], compiled.eos_id],
        ),
        DecodeCase(
            name="byte-run-split-by-token",
            why="run が非 byte トークンで 2 つに割れる（片方だけ有効）",
            ids=[
                byte_ids[0xE3],
                byte_ids[0x81],
                byte_ids[0x82],
                compiled.bos_id,
                byte_ids[0xF0],
                byte_ids[0x9F],
            ],
        ),
        DecodeCase(
            name="specials-only",
            why="特殊トークンだけ（skip_special_tokens の分岐が出る唯一の形）",
            ids=[compiled.bos_id, compiled.eos_id],
        ),
    ]
