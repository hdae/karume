r"""Asset prep for the prompt layer (tokenizers) of the examples/anima demo (tables + fixture).

Where `anima/export.py` produces **the graph** and `anima/pipeline_ref.py` **the host-side numeric
truth**, this module handles only "prompt string → token id sequence". It does not touch the model
graph.

    uv run python -m anima.demo

Two outputs (**always both in one run** — unless they are built from the same tables, the runtime
assets and the fixture go stale separately and you get "the tests are green but only the demo has
a different id sequence"):

  ① runtime assets `outputs/series/anima-demo/text/` (under `.gitignore`, about 4.6MB in total)
       qwen2-tokenizer.json   vocab / merges / char-class table / NFC segment table / added tokens
       t5-tokenizer.json      vocab / scores / normalizer table (folded Precompiled) / added tokens
  ② parity fixture `packages/models/tests/fixtures/anima-text/parity.json`
     (**tracked in git**, about 470KB)
       the reference id sequences of all cases (PROMPT_CASES) and the **subset** of the vocab
       needed to reproduce them (split off so the 151k / 32k vocabs are not committed), plus the
       measured NFC pairs

MUST: do not put the assets directly under `models/<repo>/` (the distribution). That is a tree that
lists only the files the manifest declares and is **uploaded to HF as is**; an undeclared file
stops `verify_dist` (the same reason `outputs/misc/sbv2-demo/` / the series `anima-pipeline/` were
kept separate).

The source of truth for how the tables are folded and verified is the docstrings of
`anima/text.py` / `_shared/qwen2_tokenizer.py`. In short: **Unicode decisions are neither
reimplemented in TS nor delegated to a standard API** — the source of truth (Rust `tokenizers` /
its regex engine / `unicode-normalization`) is applied from Python to every code point and folded
into closed-interval tables and mapping tables. The equivalence of the folding is checked
exhaustively + by random sampling on every emit, and in addition the id sequences of the reference
implementation and `AutoTokenizer` are compared on 2,000 random samples. On a mismatch **nothing is
written** (ADR 0005).

After generation, run `vp fmt packages/models/tests/fixtures/anima-text/parity.json` from the
repository root (the formatter decides the committed form — the format check of `vp check` also
covers fixtures).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from _shared import qwen2_tokenizer as q2
from _shared.paths import REPO_ROOT, SERIES_ROOT

from . import text as at

DEFAULT_REPO = "circlestone-labs/Anima-Base-v1.0-Diffusers"

#: Location of the runtime assets (`outputs/` is in `.gitignore`). These 2 files go into the
#: distribution's `tokenizer/` / `tokenizer_2/` (`ANIMA_TOKENIZER_SERIES` in
#: `anima/distribution.py`), and on the TS side `packages/models/src/text/qwen2-tokenizer.ts` /
#: `packages/models/src/anima/text/t5-tokenizer.ts` read them through the manifest's assets
#: (no TS reads this location directly).
DEFAULT_ASSETS_DIR = SERIES_ROOT / "anima-demo" / "text"
QWEN_ASSET_FILE = "qwen2-tokenizer.json"
T5_ASSET_FILE = "t5-tokenizer.json"

#: Parity fixture (**tracked in git**). Read on the TS side by
#: `packages/models/tests/anima_tokenizer_test.ts`.
DEFAULT_FIXTURE_PATH = (
    REPO_ROOT / "packages" / "models" / "tests" / "fixtures" / "anima-text" / "parity.json"
)

#: Truncation length matching the source of truth's call. The default of `--max-sequence-length`
#: in `anima/pipeline_ref.py` (passed to both tokenizers as `max_length=512, truncation=True`).
MAX_LENGTH = 512

#: Prompt set for the parity check. Designed so that **each case holds one boundary where a naive
#: port fails**; invisible characters are always written as \uXXXX (they can vanish or be replaced
#: during editing). **Spellings that normalization collapses into other characters are treated the
#: same way** (the jamo sequence of `hangul` is visible, but once NFC is applied it turns into a
#: precomposed syllable, giving "precomposed × 2", and only the point of the case silently
#: disappears).
PROMPT_CASES: list[tuple[str, str, str]] = [
    (
        "tags",
        "1girl, solo, long hair, blue eyes, school uniform, cherry blossoms, "
        "outdoors, smile, upper body, masterpiece, best quality",
        "default prompt — a danbooru tag list",
    ),
    (
        "negative",
        "low quality, worst quality, blurry, bad anatomy, jpeg artifacts",
        "default negative prompt",
    ),
    (
        "digits",
        "1girl, 2boys, 3 cats, 2024, 1.5x, 100%",
        "Qwen2's `\\p{N}` splits one character at a time",
    ),
    (
        "ascii_punct",
        "a<b>c&d\"e'f`g~h|i\\j/k",
        "a run of symbols — the greediness of `[^\\s\\p{L}\\p{N}]+`",
    ),
    (
        "apostrophe",
        "it's a cat's toy, I'M HERE, they'll, we've, don't, I'd",
        "the case-insensitive branch `(?i:'s|'t|'re|'ve|'m|'ll|'d)`",
    ),
    (
        "apostrophe_fold",
        "it'\u017fs and it'S",
        "`(?i:)` is simple case folding — both U+017F and `'S` fold to s",
    ),
    (
        "long_s",
        "itſs and it's",
        "U+017F alone (no apostrophe — it does not enter the contraction branch)",
    ),
    (
        "japanese",
        "夕焼けの空、桜が舞う。少女がひとり立っている。",
        "Japanese (T5 fuses consecutive unk / Qwen2 is byte-level)",
    ),
    (
        "mixed",
        "1girl, 猫耳, ロングヘア, masterpiece",
        "mixed Japanese and English — switching between word and non-word",
    ),
    (
        "emoji",
        "a cat \U0001f408 and a flag \U0001f1ef\U0001f1f5 and ❤\ufe0f love",
        "emoji, regional indicators, variation selectors",
    ),
    (
        "zwj",
        "family \U0001f468\u200d\U0001f469\u200d\U0001f466 photo",
        "ZWJ sequence — T5 normalization maps ZWJ to a space",
    ),
    (
        "combining",
        "café vs cafe\u0301, A\u0301\u0301, ᴬ\u0300, e\u0303\u033c",
        "combining characters — hits Precompiled's 'shortest prefix wins, rest dropped' directly",
    ),
    (
        "odd_spaces",
        "wide\u00a0space and\u3000ideographic\u2028line\u000bvtab",
        "assorted spaces (NBSP / ideographic / line separator / vertical tab)",
    ),
    (
        "spaces",
        "  leading and   inner   and trailing  ",
        "runs of spaces, leading and trailing spaces — the backtracking of `\\s+(?!\\S)`",
    ),
    (
        "newlines",
        "line1\nline2\r\nline3\n\n\nline4",
        "newlines and CRLF (`\\s*[\\r\\n]+` and GB3)",
    ),
    (
        "only_space",
        "   ",
        "spaces only — T5 yields only `</s>`, which falls outside Dim(min=2)",
    ),
    (
        "single_char",
        "a",
        "1 character — Qwen2 yields 1 token, which falls outside Dim(min=2)",
    ),
    (
        "qwen_special",
        "<|endoftext|>a<|im_start|>b<|vision_pad|>",
        "Qwen2's added tokens are split off leftmost-longest before normalization",
    ),
    (
        "t5_special",
        "<extra_id_0> foo <extra_id_99> </s> <pad>",
        "T5's added tokens (the <extra_id_*> group)",
    ),
    (
        "special_lookalike",
        "<|endoftext|> <|endoftext|x <|not_a_token|>",
        "strings that look like special tokens — a partial match is not split off",
    ),
    (
        "control",
        "abc\ufeffd\u200be\u00adf\u0001g",
        "BOM, zero-width space, soft hyphen, control characters",
    ),
    (
        "metaspace",
        "under\u2581score",
        "the input contains U+2581 (Metaspace's replacement character)",
    ),
    (
        "long",
        ", ".join(["masterpiece", "best quality", "highly detailed", "1girl", "solo"] * 40),
        "a very long sequence — truncation at 512",
    ),
    (
        "very_long_word",
        "a" * 600,
        "one long word — BPE iteration and truncation",
    ),
    (
        "cjk_punct",
        "「猫」と『犬』、あるいは…！？",
        "Japanese punctuation (normalization of full-width symbols)",
    ),
    (
        "accents",
        "naïve café résumé Ångström ﬁ ﬂ ½ Ⅻ",
        "ligatures, fractions, Roman numerals (NFKC-style mappings)",
    ),
    (
        "thai_indic",
        "กำ क\u093c\u094dष ൎक",
        "Thai / Devanagari (SpacingMark, Prepend)",
    ),
    (
        "hangul",
        "각 \u1100\u1161\u11a8",
        "Hangul precomposed syllables and jamo sequences",
    ),
]


def _resolve(repo: str, filename: str, snapshot: Path | None) -> Path:
    if snapshot is not None:
        return snapshot / filename
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo, filename))


def check_upstream_shape(qwen_raw: dict[str, Any], t5_raw: dict[str, Any]) -> None:
    """Checks the structure of upstream `tokenizer.json` that the baked tables assume (fail loudly).

    Every item here is of the kind "when it is off, only the id sequence silently changes", and the
    AutoTokenizer comparison of `build_cases` catches it **only within what PROMPT_CASES exercises**
    (the same hole through which the NFC engine difference slipped past 28 cases). The premises
    themselves are pinned structurally.
    """
    if qwen_raw["normalizer"]["type"] != "NFC":
        raise ValueError(f"Qwen2's normalizer is not NFC: {qwen_raw['normalizer']['type']}")
    pre = qwen_raw["pre_tokenizer"]["pretokenizers"]
    if pre[0]["type"] != "Split":
        raise ValueError(f"Qwen2's first pre_tokenizer is not Split: {pre[0]['type']}")
    byte_level = [entry for entry in pre if entry["type"] == "ByteLevel"]
    if len(byte_level) != 1 or byte_level[0]["add_prefix_space"]:
        raise ValueError("Qwen2's ByteLevel is not a single stage with add_prefix_space=false")
    if t5_raw["model"].get("byte_fallback"):
        raise ValueError("T5's Unigram uses byte_fallback (breaks the premise of unk fusion)")

    # post_processor: Qwen2 **adds nothing** (which is why truncation can simply keep the head),
    # T5 adds exactly one `</s>` at the end (what the reference implementation adds itself).
    qwen_post = qwen_raw["post_processor"]
    if qwen_post["type"] != "TemplateProcessing" or any(
        "SpecialToken" in entry for entry in qwen_post["single"]
    ):
        raise ValueError(f"Qwen2's post_processor adds special tokens: {qwen_post}")
    t5_post = t5_raw["post_processor"]
    t5_specials = [
        entry["SpecialToken"]["id"] for entry in t5_post["single"] if "SpecialToken" in entry
    ]
    if t5_post["type"] != "TemplateProcessing" or t5_specials != ["</s>"]:
        raise ValueError(f"T5's post_processor does not add exactly one `</s>`: {t5_post}")

    # The flags of the added tokens should all be at their defaults (`normalized: true` matches
    # **after** normalization / `lstrip` `rstrip` eat the surrounding whitespace / `single_word`
    # requires word boundaries — each changes the meaning of the transcription, "split out
    # leftmost-longest before normalization").
    for name, raw in (("Qwen2", qwen_raw), ("T5", t5_raw)):
        for entry in raw["added_tokens"]:
            flags = {key: entry[key] for key in ("lstrip", "rstrip", "single_word", "normalized")}
            if any(flags.values()):
                raise ValueError(
                    f"{name}: the flags of added token {entry['content']!r} are not the"
                    f" defaults: {flags}"
                )


def build_assets(repo: str, snapshot: Path | None, fuzz: int) -> dict[str, Any]:
    """Builds and verifies, from the 2 `tokenizer.json` files, the tables the TS side only looks up.

    If the verification (equivalence of the folding) fails, it raises = **not a single byte is
    written**. It is the very justification of the folding, so `--fuzz 0` can thin it out, but the
    3 exhaustive checks cannot be removed.
    """
    qwen_path = _resolve(repo, "tokenizer/tokenizer.json", snapshot)
    t5_path = _resolve(repo, "t5_tokenizer/tokenizer.json", snapshot)
    source = at.AnimaTokenizerSource(qwen_path, t5_path)
    qwen_raw, t5_raw = source.qwen_json(), source.t5_json()

    if qwen_raw["model"]["type"] != "BPE" or t5_raw["model"]["type"] != "Unigram":
        raise ValueError("model.type differs from the expected (BPE / Unigram)")
    check_upstream_shape(qwen_raw, t5_raw)

    started = time.perf_counter()
    classes = q2.build_char_classes()
    class_stats = at.verify_char_classes(classes)
    case_fold = q2.build_case_fold()
    class_stats["caseFold"] = len(case_fold)
    print(f"[classes] {class_stats}（{time.perf_counter() - started:.1f}s）", flush=True)

    started = time.perf_counter()
    nfc_segments = q2.build_nfc_segments()
    nfc_stats = q2.verify_nfc_segments(nfc_segments, fuzz=fuzz)
    print(f"[nfc] {nfc_stats}（{time.perf_counter() - started:.1f}s）", flush=True)

    started = time.perf_counter()
    pattern = qwen_raw["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"]
    scan_stats = q2.verify_qwen_pre_tokenize(pattern, classes, case_fold)
    print(f"[pre-token] {scan_stats}（{time.perf_counter() - started:.1f}s）", flush=True)

    started = time.perf_counter()
    charsmap = at.t5_charsmap(t5_raw)
    tables = at.build_spm_tables(charsmap)
    table_stats = at.verify_spm_tables(tables, charsmap, fuzz=fuzz)
    print(f"[normalizer] {table_stats}（{time.perf_counter() - started:.1f}s）", flush=True)

    qwen_vocab: dict[str, int] = qwen_raw["model"]["vocab"]
    qwen_merges = [(left, right) for left, right in qwen_raw["model"]["merges"]]
    # The emit side checks that the line-separated format (line number = id / rank) cannot break.
    # This closes off silent breakage when the upstream vocab changes (the TS side only splits the
    # lines and cannot detect an anomaly).
    for token in qwen_vocab:
        if "\n" in token or " " in token:
            raise ValueError(
                f"Qwen2 vocab token {token!r} contains a newline / space — breaks the line format"
            )
    t5_vocab = [(token, float(score)) for token, score in t5_raw["model"]["vocab"]]
    for token, _ in t5_vocab:
        if "\n" in token:
            raise ValueError(
                f"T5 vocab token {token!r} contains a newline — breaks the line format"
            )

    ordered_qwen = [""] * len(qwen_vocab)
    for token, tid in qwen_vocab.items():
        ordered_qwen[tid] = token
    if any(token == "" for token in ordered_qwen):
        raise ValueError("Qwen2 vocab ids have a gap (breaks the premise line number = id)")

    qwen_asset = {
        "source": f"{repo}#tokenizer",
        "maxLength": MAX_LENGTH,
        "addedTokens": sorted(at.added_token_map(qwen_raw).items(), key=lambda kv: kv[1]),
        "classes": classes,
        # Case-insensitivity of the contractions `(?i:'s|…)`. Not ASCII-equivalent (U+017F → s).
        "caseFold": sorted([cp, target] for cp, target in case_fold.items()),
        # Code points where the source of truth's NFC and plain NFC split (segment here, then
        # normalize).
        "nfcSegments": nfc_segments,
        # 0-based line number = id. The byte-level vocab contains no newline or space (checked
        # above).
        "vocabText": "\n".join(ordered_qwen),
        # 0-based line number = rank.
        "mergesText": "\n".join(f"{left} {right}" for left, right in qwen_merges),
        # Expected line counts. A truncated `mergesText` gives a silent wrong value where "only the
        # BPE split changes", so the TS side can compare them with the result of the split.
        "vocabCount": len(ordered_qwen),
        "mergesCount": len(qwen_merges),
    }
    t5_asset = {
        "source": f"{repo}#t5_tokenizer",
        "maxLength": MAX_LENGTH,
        "unkId": t5_raw["model"]["unk_id"],
        "eosId": at.added_token_map(t5_raw)["</s>"],
        "addedTokens": sorted(at.added_token_map(t5_raw).items(), key=lambda kv: kv[1]),
        # T5's WhitespaceSplit is the same set as Qwen2's `\s` (measured by `verify_char_classes`).
        "space": classes["space"],
        "normalizer": tables.to_json(),
        "vocabText": "\n".join(token for token, _ in t5_vocab),
        "scores": [score for _, score in t5_vocab],
    }
    return {
        "qwen": qwen_asset,
        "t5": t5_asset,
        "classes": classes,
        "caseFold": case_fold,
        "nfcSegments": nfc_segments,
        "tables": tables,
        "qwenVocab": qwen_vocab,
        "qwenMerges": qwen_merges,
        "t5Vocab": t5_vocab,
        "stats": {
            "classes": class_stats,
            "nfc": nfc_stats,
            "preTokenize": scan_stats,
            "normalizer": table_stats,
        },
        "paths": {"qwen": str(qwen_path), "t5": str(t5_path)},
    }


def references(built: dict[str, Any]) -> tuple[q2.Qwen2Reference, at.T5Reference]:
    """The pair of reference implementations that read **only** the baked tables (mirrors of TS)."""
    qwen = q2.Qwen2Reference(
        vocab=built["qwenVocab"],
        merges={pair: rank for rank, pair in enumerate(built["qwenMerges"])},
        added=dict(built["qwen"]["addedTokens"]),
        classes=built["classes"],
        case_fold=built["caseFold"],
        nfc_segments=built["nfcSegments"],
    )
    t5_vocab = {token: (i, score) for i, (token, score) in enumerate(built["t5Vocab"])}
    t5 = at.T5Reference(
        vocab=t5_vocab,
        min_score=min(score for _, score in built["t5Vocab"]),
        unk_id=built["t5"]["unkId"],
        eos_id=built["t5"]["eosId"],
        added=dict(built["t5"]["addedTokens"]),
        tables=built["tables"],
        space=built["classes"]["space"],
    )
    return qwen, t5


def reference_ids(tokenizer: Any, text: str) -> list[int]:
    """Gets the true id sequence with **the same call** as `anima.pipeline_ref.encode_text`.

    MUST: no chat template (the plain `__call__`). `tokenizer/chat_template.jinja` is bundled but
    the Anima path does not use it; going through `apply_chat_template` prefixes the `<|im_start|>`
    family and makes the id sequence entirely different.
    """
    encoded = tokenizer([text], padding="longest", max_length=MAX_LENGTH, truncation=True)
    return list(encoded["input_ids"][0])


def build_cases(repo: str, built: dict[str, Any]) -> list[dict[str, Any]]:
    """Takes the true id sequences of all cases and checks the reference implementation matches.

    The source of truth is `transformers`' `AutoTokenizer`. This being green is the mechanical proof
    that "the baked tables hold all the information needed", and the TS implementation is a
    transcription of it. If even one case differs, **nothing is emitted** (no "id sequence the
    reference implementation cannot reproduce" is baked into the fixture).
    `anima/tests/test_demo.py` runs the same comparison against **the committed fixture** too (so
    that a change in upstream's tokenizer.json is noticed even without regenerating).
    """
    from transformers import AutoTokenizer

    truth = {
        "qwen": AutoTokenizer.from_pretrained(repo, subfolder="tokenizer"),
        "t5": AutoTokenizer.from_pretrained(repo, subfolder="t5_tokenizer"),
    }
    qwen_ref, t5_ref = references(built)

    cases: list[dict[str, Any]] = []
    for name, text, why in PROMPT_CASES:
        want_q = reference_ids(truth["qwen"], text)
        want_t = reference_ids(truth["t5"], text)
        got_q = qwen_ref.encode(text, MAX_LENGTH)
        got_t = t5_ref.encode(text, MAX_LENGTH)
        if got_q != want_q:
            raise ValueError(
                f"[{name}] Qwen2 id sequence differs from the source of truth\n"
                f"  truth={want_q}\n  reference={got_q}"
            )
        if got_t != want_t:
            raise ValueError(
                f"[{name}] T5 id sequence differs from the source of truth\n"
                f"  truth={want_t}\n  reference={got_t}"
            )
        cases.append(
            {
                "id": name,
                "why": why,
                "text": text,
                "qwenIds": want_q,
                "t5Ids": want_t,
                "t5Normalized": at.normalize_with_tables(built["tables"], text),
            }
        )
    return cases


#: Characters mixed into the random comparison of id sequences (emoji / regional indicators /
#: variation selectors / combining marks / hangul jamo / assorted spaces and controls / characters
#: from rare planes). Mechanically exercises combinations `PROMPT_CASES` does not hit.
_FUZZ_EXOTIC = (
    0x000A, 0x000D, 0x0009, 0x0020, 0x00A0, 0x2028, 0x3000, 0x200B, 0x200D, 0xFEFF,
    0x00AD, 0x0001, 0x0027, 0x0060, 0x005C, 0x002F, 0x0025, 0x2581, 0x017F, 0x00DF,
    0x0300, 0x0301, 0x0302, 0x0303, 0x0316, 0x0328, 0x0334, 0x033C, 0x093C, 0x094D,
    0x0E33, 0x0EB3, 0x0F71, 0x0F72, 0x3099, 0xFE0F, 0xE0100, 0x1100, 0x1161, 0x11A8,
    0xAC00, 0x304B, 0x3053, 0x4E16, 0x732B, 0xFF21, 0xFB01, 0x00BD, 0x216B, 0x2026,
    0x1F408, 0x1F600, 0x1F1EF, 0x1F1F5, 0x1F3FB, 0x1F468, 0x2F929, 0x0818, 0x089A,
)  # fmt: skip

#: Ordinary words used as random material (sequences close to real prompts are mixed in — if
#: everything were exotic, the ordinary BPE path would be thinly covered).
_FUZZ_WORDS = ("1girl", "solo", "masterpiece", "cat's", "café", "夕焼け", "2024", "<|endoftext|>")


def verify_reference_fuzz(
    repo: str, built: dict[str, Any], *, count: int = 2_000, seed: int = 0
) -> dict[str, Any]:
    """Compares id sequences of the reference implementation and AutoTokenizer on random prompts.

    `PROMPT_CASES` places "the boundaries where a naive port fails" by hand, one per case, and
    **no character outside them is checked at all** (the NFC engine difference split the id
    sequences through this hole). This is not exhaustive, but with fixed-seed random sampling it is
    a permanent check that mechanically exercises "combinations the hand-written script did not
    think of". If even one differs, nothing is emitted.
    """
    import random

    from transformers import AutoTokenizer

    truth = {
        "qwen": AutoTokenizer.from_pretrained(repo, subfolder="tokenizer"),
        "t5": AutoTokenizer.from_pretrained(repo, subfolder="t5_tokenizer"),
    }
    # The reference implementations accumulate `used_vocab`, so to keep the fixture's subset from
    # growing, **instances dedicated to this check** are built (build_fixture builds its own).
    qwen_ref, t5_ref = references(built)
    alphabet = [chr(cp) for cp in _FUZZ_EXOTIC]
    alphabet += [chr(cp) for start, end in built["nfcSegments"] for cp in range(start, end + 1)]
    alphabet += [chr(cp) for cp in range(0x20, 0x7F)]
    alphabet += list(_FUZZ_WORDS)
    rng = random.Random(seed)
    for index in range(count):
        text = "".join(rng.choice(alphabet) for _ in range(rng.randint(1, 16)))
        for which, ref, tokenizer in (
            ("Qwen2", qwen_ref, truth["qwen"]),
            ("T5", t5_ref, truth["t5"]),
        ):
            want = reference_ids(tokenizer, text)
            got = ref.encode(text, MAX_LENGTH)
            if got != want:
                raise ValueError(
                    f"Random sample {index}: the {which} id sequence differs from the source of"
                    " truth\n"
                    f"  input={[hex(ord(c)) for c in text]}\n  truth={want}\n  reference={got}"
                )
    return {"count": count, "seed": seed, "alphabet": len(alphabet)}


def build_fixture(built: dict[str, Any], cases: list[dict[str, Any]]) -> dict[str, Any]:
    """Fixture = reference id sequences + the **minimal** vocab / merges that reproduce them.

    The vocabs (151k / 32k) are not committed whole. The normalizer table and the char-class table
    are **the output of the folding itself** (= what is being verified), so they are not trimmed.
    """
    qwen_ref, t5_ref = references(built)
    for case in cases:
        qwen_ref.encode(case["text"], MAX_LENGTH)
        t5_ref.encode(case["text"], MAX_LENGTH)

    merge_rank = {pair: rank for rank, pair in enumerate(built["qwenMerges"])}
    t5_vocab = {token: (i, score) for i, (token, score) in enumerate(built["t5Vocab"])}
    return {
        "_doc": [
            "Parity fixture for the Anima prompt layer"
            " (generated by tools/export-recipes/anima/demo.py).",
            "The source of truth is transformers' AutoTokenizer; the id sequences are measured"
            " from it. Of the vocab and",
            "merges, only the **subset** needed to reproduce every case is included (so that the"
            " 151k / 32k are not",
            "committed). The normalizer table and the char-class tables are the output of the"
            " folding itself, so they are included whole.",
        ],
        "source": {"repo": built["qwen"]["source"].split("#")[0], "reference": "tokenizers"},
        "maxLength": MAX_LENGTH,
        "qwen": {
            "addedTokens": built["qwen"]["addedTokens"],
            "classes": built["classes"],
            "caseFold": built["qwen"]["caseFold"],
            "nfcSegments": built["nfcSegments"],
            # `[input, the source of truth's NFC output]` — the TS side cannot reproduce it with a
            # plain `normalize("NFC")` (an implementation that skips the segment table fails here).
            "nfcCases": q2.nfc_fixture_cases(built["nfcSegments"]),
            "vocab": {t: built["qwenVocab"][t] for t in sorted(qwen_ref.used_vocab)},
            "merges": sorted(
                ([left, right, merge_rank[(left, right)]] for left, right in qwen_ref.used_merges),
                key=lambda entry: entry[2],
            ),
        },
        "t5": {
            "addedTokens": built["t5"]["addedTokens"],
            "unkId": built["t5"]["unkId"],
            "eosId": built["t5"]["eosId"],
            # MUST: the score of unknown nodes and the Viterbi search width are determined by
            # **the whole vocab**. Computing them from the subset gives another split, so they are
            # kept apart from the subset.
            "minScore": min(score for _, score in built["t5Vocab"]),
            "maxTokenLength": max(len(token) for token, _ in built["t5Vocab"]),
            "space": built["classes"]["space"],
            "normalizer": built["tables"].to_json(),
            "vocab": [
                [token, t5_vocab[token][0], t5_vocab[token][1]]
                for token in sorted(t5_ref.used_vocab)
            ],
        },
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--snapshot", type=Path, default=None, help="use this HF snapshot directly")
    parser.add_argument("--assets-out", type=Path, default=DEFAULT_ASSETS_DIR)
    parser.add_argument("--fixtures-out", type=Path, default=DEFAULT_FIXTURE_PATH)
    parser.add_argument(
        "--fuzz", type=int, default=200_000, help="random samples for the folding check"
    )
    parser.add_argument(
        "--id-fuzz", type=int, default=2_000, help="random samples for the id comparison"
    )
    args = parser.parse_args()

    built = build_assets(args.repo, args.snapshot, args.fuzz)
    cases = build_cases(args.repo, built)
    print(f"[cases] {len(cases)} cases match the source of truth", flush=True)

    started = time.perf_counter()
    fuzz_stats = verify_reference_fuzz(args.repo, built, count=args.id_fuzz)
    print(f"[id-fuzz] {fuzz_stats}（{time.perf_counter() - started:.1f}s）", flush=True)

    args.assets_out.mkdir(parents=True, exist_ok=True)
    written: dict[str, Any] = {}
    for filename, payload in ((QWEN_ASSET_FILE, built["qwen"]), (T5_ASSET_FILE, built["t5"])):
        path = args.assets_out / filename
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        written[filename] = path.stat().st_size

    fixture = build_fixture(built, cases)
    args.fixtures_out.parent.mkdir(parents=True, exist_ok=True)
    args.fixtures_out.write_text(
        json.dumps(fixture, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )

    print(
        json.dumps(
            {
                "stats": {**built["stats"], "idFuzz": fuzz_stats},
                "sources": built["paths"],
                "assets": {"dir": str(args.assets_out), "bytes": written},
                "fixture": {
                    "path": str(args.fixtures_out),
                    "bytes": args.fixtures_out.stat().st_size,
                    "cases": len(cases),
                    "qwenVocabSubset": len(fixture["qwen"]["vocab"]),
                    "qwenMergeSubset": len(fixture["qwen"]["merges"]),
                    "t5VocabSubset": len(fixture["t5"]["vocab"]),
                },
            },
            indent=1,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
