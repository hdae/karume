r"""Bakes the Qwen3.5 tokenizer asset (format `karume-qwen3_5-tokenizer/1`) from the tokenizer
MOSS-Image pins, and the TS fixture that pins `packages/models/src/qwen3_5/`.

    uv run --with 'transformers==5.14.1' --inexact python -m qwen3_5.tokenizer [--fetch]

Two outputs, always from one run (made from the same tables, so the run-time asset and the fixture
cannot drift apart — the `anima.demo` / `wan.umt5_xxl_tokenizer` discipline):

  1. the run-time asset `outputs/series/qwen3_5-tokenizer/tokenizer.json` (untracked, about 7 MB) —
     the asset `tokenizer` of the MOSS-Image text-encoder distribution (wired on the MOSS branch);
  2. the fixture `packages/models/tests/fixtures/qwen3_5/tokenizer-cases.json` (tracked).

Format the fixture afterwards (`vp fmt packages/models/tests/fixtures/qwen3_5/` — the committed form
is the formatter's). `--fetch` downloads the pinned files when they are not in the HF cache (without
it a missing file fails). transformers is pinned for the run (no dependency group of its own — the
package doc of `qwen3_5`); `apply_chat_template` also needs jinja2, which the venv carries.

## Source of truth (measured 2026-10-31)

The id sequence is transformers 5.14.1 `AutoTokenizer` over `OpenMOSS-Team/MOSS-Image-SFT@f972bd4b…`
`tokenizer/` (three files, sha256-pinned in {@link SOURCE_FILES} and copied to a scratch directory
before reading), called as upstream's `encode_prompt` calls it:
`tokenizer(apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
add_generation_prompt=True, enable_thinking=True), truncation=True, max_length=4096)`.

- transformers' `Qwen2Tokenizer` rebuilds the pre-tokenizer from its class constant (no `\p{M}`) —
  the expression in MOSS's `tokenizer.json`; `Qwen/Qwen3.5-9B@c2022362`'s `tokenizer.json` carries
  `[\p{L}\p{M}]+` instead, but transformers builds the identical backend from either repo (compared
  2026-10-31). The emit gate reads the expression from the **built backend**, not from the file.
- The backend's shape is checked ({@link check_backend_shape}); the raw `tokenizer.json` path
  (`tokenizers.Tokenizer.from_file`) must give the same ids as transformers on every case.
- Unicode tables (`\p{L}` / `\p{N}` / `\s`, the `(?i:)` fold, the NFC segment table) are folded by
  asking the canonical engine one code point at a time and verified exhaustively — the builders are
  the Qwen2 ones shared with Anima (`_shared.qwen2_tokenizer`; same engine, same expression, same
  normaliser). Then the Python mirror (`_shared.qwen2_tokenizer.Qwen2Reference`, the TS
  implementation's mirror) must reproduce transformers on every case and on a seeded random sweep.
  Any mismatch writes nothing (ADR 0005).

## Asset (`karume-qwen3_5-tokenizer/1`)

    format       "karume-qwen3_5-tokenizer/1"
    source       {"repo", "revision", "subfolder", "files": {name: sha256}}
    versions     {"tokenizers", "transformers"} (the versions that decide the ids)
    maxLength    4096 (upstream's `max_sequence_length` default and cap)
    addedTokens  [[content, id], …] (33)
    classes      {"letter", "number", "space"} closed-interval tables
    caseFold     [[cp, suffix cp], …]
    nfcSegments  closed-interval table
    vocabText / vocabCount   newline-separated tokens, line number = id
    mergesText / mergesCount newline-separated `left right`, line number = rank

## Fixture (`tokenizer-cases.json`)

    _doc / assetFormat / source / versions / maxLength   as the asset
    qwen         addedTokens / classes / caseFold / nfcSegments (whole), nfcCases ([input, canonical
                 NFC]), vocab ({token: id}) and merges ([left, right, rank]) — the subset every case
                 needs (the whole vocabulary is not committed)
    pythonSpace  the code points of `str.isspace()` (what the template's `trim` strips)
    promptCases  {id, why, prompt} + either {rendered, ids (upstream, truncated at maxLength),
                 tokens (the untruncated count)} or {reject: "template"}
    textCases    {id, why, text, ids} — the tokenizer alone (hand-written + seeded random)

MUST: transformers / tokenizers / huggingface_hub are imported inside functions (recipe groups are
not in the default sync — `tests/test_optional_group_imports.py`).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import tempfile
import time
from pathlib import Path
from typing import Any

from _shared import qwen2_tokenizer as q2
from _shared.paths import REPO_ROOT, SERIES_ROOT

from . import tokenizer_cases as tc

#: The tokenizer MOSS-Image loads (`model_index.json`: `tokenizer = transformers.Qwen2Tokenizer`).
SOURCE_REPO = "OpenMOSS-Team/MOSS-Image-SFT"
SOURCE_REVISION = "f972bd4b4b9e3852133cc5da954d87bc32cd8fb9"
SOURCE_SUBFOLDER = "tokenizer"

#: The pinned files (sha256 of the bytes). `chat_template.jinja` is byte-identical to
#: `Qwen/Qwen3.5-9B@c2022362`'s (same git blob `a585dec8…`).
SOURCE_FILES: dict[str, str] = {
    "chat_template.jinja": "a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715",
    "tokenizer.json": "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523",
    "tokenizer_config.json": "bee8eba30f0eb4af73c0fe2cd06d0f89b657d7819941c438157ec42f7c80ea87",
}

#: The asset's format version (the TS reader refuses other versions).
ASSET_FORMAT = "karume-qwen3_5-tokenizer/1"

#: The versions recorded in the asset (they decide the ids).
ASSET_VERSIONS: tuple[str, ...] = ("tokenizers", "transformers")

#: Upstream's `max_sequence_length` (default and cap of `MOSSImagePipeline.__call__`).
MAX_LENGTH = 4096

#: The series directory of the run-time asset (`docs/assets-layout.md`).
SERIES_NAME = "qwen3_5-tokenizer"
ASSET_FILE = "tokenizer.json"

#: The fixture (consumer: `packages/models/tests/qwen3_5_tokenizer_test.ts`).
FIXTURE_PATH = (
    REPO_ROOT / "packages" / "models" / "tests" / "fixtures" / "qwen3_5" / "tokenizer-cases.json"
)

#: The chat template's spellings for one user message with the thinking generation prompt (the
#: mirror of `renderQwen3_5ChatPrompt` in `packages/models/src/qwen3_5/chat.ts`).
CHAT_PREFIX = "<|im_start|>user\n"
CHAT_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n"

#: The word whose ids the truncation of `boundary_in_piece` cuts through (14 ids).
LONG_WORD = " Pneumonoultramicroscopicsilicovolcanoconiosis"

#: `(id, repeats of "cat", tail, why)`. "cat" + " cat" × (n − 1) is n tokens, the template 10:
#: 4086 → exactly 4096; 4087 → one over (the cut drops the template's last `\n`); 4086 + the
#: 14-id word → the cut falls inside the word's ids. The emit gate checks each count.
BOUNDARY_CASES: list[tuple[str, int, str, str]] = [
    ("boundary_4096", 4086, "", "exactly maxLength tokens — accepted as is"),
    ("boundary_4097", 4087, "", "one token over — rejected by default; truncation drops the tail"),
    (
        "boundary_in_piece",
        4086,
        LONG_WORD,
        "the bound falls inside one pre-token's ids — truncation keeps a partial piece",
    ),
]

#: The untruncated token counts the boundary cases must have (checked at emit).
BOUNDARY_TOKENS = {"boundary_4096": 4096, "boundary_4097": 4097, "boundary_in_piece": 4110}

#: Characters mixed into the seeded random text cases and the id sweep (emoji, combining marks,
#: Unicode spaces, controls, CJK, Devanagari, the NFC segment examples).
_FUZZ_EXOTIC = (
    0x000A, 0x000D, 0x0009, 0x0020, 0x00A0, 0x2028, 0x3000, 0x200B, 0x200D, 0xFEFF,
    0x00AD, 0x0001, 0x0027, 0x0060, 0x005C, 0x002F, 0x0025, 0x017F, 0x00DF, 0x0085,
    0x0300, 0x0301, 0x0302, 0x0316, 0x0334, 0x093C, 0x094D, 0x093F, 0x0947, 0x0928,
    0x0E33, 0x3099, 0xFE0F, 0xE0100, 0x1100, 0x1161, 0x11A8, 0xAC00, 0x304B, 0x3053,
    0x4E16, 0x732B, 0xFF21, 0xFB01, 0x00BD, 0x2026, 0x1F408, 0x1F600, 0x1F1EF, 0x1F1F5,
    0x1F3FB, 0x2F929, 0x0818, 0x089A, 0x001C, 0x001F,
)  # fmt: skip

#: Words mixed into the random sweep (so the ordinary BPE path is not thinned out), including
#: added-token spellings.
_FUZZ_WORDS = (
    "cat",
    " dog",
    "it's",
    "café",
    "2026",
    "<|im_end|>",
    "<think>",
    "</think>",
    "<|endoftext|>",
    "<tool_response>",
)

#: Seeded random text cases committed to the fixture (the TS side is not covered by the emit gate).
RANDOM_TEXT_CASES = 64


class Qwen35TokenizerError(ValueError):
    """The pinned tokenizer, the folded tables or the expectations left the agreed form."""


# ---------------------------------------------------------------------------
# Source
# ---------------------------------------------------------------------------


def pinned_tokenizer_files(destination: Path, *, fetch: bool = False) -> None:
    """Copies the pinned files, sha256-checked, into `destination` (the reader reads only these).

    MUST: the check runs on the bytes that are written (read, check, then write), and nothing is
    written unless every file matches.
    """
    from huggingface_hub import hf_hub_download

    verified: dict[str, bytes] = {}
    for name, want in SOURCE_FILES.items():
        try:
            path = Path(
                hf_hub_download(
                    SOURCE_REPO,
                    f"{SOURCE_SUBFOLDER}/{name}",
                    revision=SOURCE_REVISION,
                    local_files_only=not fetch,
                )
            )
        except Exception as cause:  # the hub raises several types for "not cached"
            raise Qwen35TokenizerError(
                f"{SOURCE_REPO}@{SOURCE_REVISION} {SOURCE_SUBFOLDER}/{name} is not available"
                " — run with --fetch"
            ) from cause
        data = path.read_bytes()
        actual = hashlib.sha256(data).hexdigest()
        if actual != want:
            raise Qwen35TokenizerError(
                f"{path}: sha256 {actual} is not the pinned {want}"
                " — not reading a different tokenizer"
            )
        verified[name] = data
    destination.mkdir(parents=True, exist_ok=True)
    for name, data in verified.items():
        (destination / name).write_bytes(data)


def load_truth(directory: Path) -> Any:
    """transformers' `AutoTokenizer` over the pinned files (the source of truth)."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(directory)


def backend_json(tokenizer: Any) -> dict[str, Any]:
    """The backend transformers actually built (not the file it read)."""
    return json.loads(tokenizer.backend_tokenizer.to_str())


def check_backend_shape(backend: dict[str, Any]) -> None:
    """Fails loudly unless the backend is the shape the shared BPE engine reproduces.

    Every item here would change ids silently if it moved, and the case comparison only catches
    what the cases exercise.
    """
    if backend["normalizer"] != {"type": "NFC"}:
        raise Qwen35TokenizerError(f"normalizer is not NFC: {backend['normalizer']}")
    pre = backend["pre_tokenizer"]
    steps = pre.get("pretokenizers", []) if pre.get("type") == "Sequence" else []
    if (
        len(steps) != 2
        or steps[0].get("type") != "Split"
        or steps[0].get("behavior") != "Isolated"
        or steps[0].get("invert")
        or "Regex" not in steps[0].get("pattern", {})
        or steps[1].get("type") != "ByteLevel"
        or steps[1].get("add_prefix_space")
        or steps[1].get("use_regex")
    ):
        raise Qwen35TokenizerError(f"pre_tokenizer is not Split(Isolated) + ByteLevel: {pre}")
    post = backend["post_processor"]
    if post is not None and post.get("type") != "ByteLevel":
        # ByteLevel only trims offsets; anything else may add tokens.
        raise Qwen35TokenizerError(f"post_processor may add tokens: {post}")
    model = backend["model"]
    expected = {
        "type": "BPE",
        "dropout": None,
        "unk_token": None,
        "fuse_unk": False,
        "byte_fallback": False,
        "ignore_merges": False,
    }
    for key, want in expected.items():
        if model.get(key) != want:
            raise Qwen35TokenizerError(f"model.{key} is {model.get(key)!r}, not {want!r}")
    for key in ("continuing_subword_prefix", "end_of_word_suffix"):
        if model.get(key) not in (None, ""):
            raise Qwen35TokenizerError(f"model.{key} is {model.get(key)!r}")
    for entry in backend["added_tokens"]:
        flags = {key: entry[key] for key in ("lstrip", "rstrip", "single_word", "normalized")}
        if any(flags.values()):
            # Each flag changes "cut out leftmost-longest before normalisation".
            raise Qwen35TokenizerError(f"added token {entry['content']!r} has flags {flags}")


def pre_tokenizer_pattern(backend: dict[str, Any]) -> str:
    return backend["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"]


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------


def build_tables(backend: dict[str, Any], *, fuzz: int) -> dict[str, Any]:
    """Folds and verifies the Unicode tables, and lays out vocabulary / merges / added tokens."""
    started = time.perf_counter()
    classes = q2.build_char_classes()
    case_fold = q2.build_case_fold()
    print(f"[classes] {len(case_fold)} folds ({time.perf_counter() - started:.1f}s)", flush=True)

    started = time.perf_counter()
    nfc_segments = q2.build_nfc_segments()
    nfc_stats = q2.verify_nfc_segments(nfc_segments, fuzz=fuzz)
    print(f"[nfc] {nfc_stats} ({time.perf_counter() - started:.1f}s)", flush=True)

    started = time.perf_counter()
    scan_stats = q2.verify_qwen_pre_tokenize(pre_tokenizer_pattern(backend), classes, case_fold)
    print(f"[pre-token] {scan_stats} ({time.perf_counter() - started:.1f}s)", flush=True)

    vocab: dict[str, int] = backend["model"]["vocab"]
    merges = [tuple(pair) for pair in backend["model"]["merges"]]
    ordered = [""] * len(vocab)
    for token, token_id in vocab.items():
        if "\n" in token or " " in token:
            raise Qwen35TokenizerError(f"vocabulary token {token!r} breaks the line format")
        ordered[token_id] = token
    if any(token == "" for token in ordered):
        raise Qwen35TokenizerError("the vocabulary ids have a hole (line number = id breaks)")
    for pair in merges:
        if len(pair) != 2 or any(" " in side or "\n" in side for side in pair):
            raise Qwen35TokenizerError(f"merge {pair!r} breaks the line format")
    added = sorted(
        ((entry["content"], entry["id"]) for entry in backend["added_tokens"]),
        key=lambda pair: pair[1],
    )
    return {
        "classes": classes,
        "caseFold": case_fold,
        "nfcSegments": nfc_segments,
        "vocab": vocab,
        "ordered": ordered,
        "merges": merges,
        "added": added,
        "stats": {"nfc": nfc_stats, "preTokenize": scan_stats},
    }


def reference(tables: dict[str, Any]) -> q2.Qwen2Reference:
    """The Python mirror of the TS implementation, reading the folded tables only."""
    return q2.Qwen2Reference(
        vocab=tables["vocab"],
        merges={pair: rank for rank, pair in enumerate(tables["merges"])},
        added=dict(tables["added"]),
        classes=tables["classes"],
        case_fold=tables["caseFold"],
        nfc_segments=tables["nfcSegments"],
    )


def source_record() -> dict[str, Any]:
    return {
        "repo": SOURCE_REPO,
        "revision": SOURCE_REVISION,
        "subfolder": SOURCE_SUBFOLDER,
        "files": dict(sorted(SOURCE_FILES.items())),
    }


def versions_record() -> dict[str, str]:
    from importlib.metadata import version

    return {name: version(name) for name in ASSET_VERSIONS}


def build_asset(tables: dict[str, Any]) -> dict[str, Any]:
    return {
        "format": ASSET_FORMAT,
        "source": source_record(),
        "versions": versions_record(),
        "maxLength": MAX_LENGTH,
        "addedTokens": [list(pair) for pair in tables["added"]],
        "classes": tables["classes"],
        # `(?i:'s|…)` is Unicode simple case folding (U+017F → s), not ASCII.
        "caseFold": sorted([cp, target] for cp, target in tables["caseFold"].items()),
        # Code points where the canonical NFC and the plain NFC differ (segment there first).
        "nfcSegments": tables["nfcSegments"],
        "vocabText": "\n".join(tables["ordered"]),
        "vocabCount": len(tables["ordered"]),
        "mergesText": "\n".join(f"{left} {right}" for left, right in tables["merges"]),
        "mergesCount": len(tables["merges"]),
    }


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------


def render_chat_prompt(prompt: str) -> str:
    """The mirror of `renderQwen3_5ChatPrompt` (raises where the template raises)."""
    content = prompt.strip()
    if content.startswith("<tool_response>") and content.endswith("</tool_response>"):
        raise Qwen35TokenizerError("wrapped in <tool_response>…</tool_response>")
    return f"{CHAT_PREFIX}{content}{CHAT_SUFFIX}"


def boundary_prompt(repeats: int, tail: str) -> str:
    return "cat" + " cat" * (repeats - 1) + tail


def prompt_inputs() -> list[tuple[str, str, str]]:
    """Every `(id, prompt, why)` of `promptCases` (hand-written first, then the boundaries)."""
    boundaries = [
        (name, boundary_prompt(repeats, tail), why) for name, repeats, tail, why in BOUNDARY_CASES
    ]
    return [*tc.PROMPT_CASES, *boundaries]


def fuzz_alphabet(tables: dict[str, Any]) -> list[str]:
    alphabet = [chr(cp) for cp in _FUZZ_EXOTIC]
    alphabet += [
        chr(cp) for start, end in tables["nfcSegments"][:8] for cp in range(start, end + 1)
    ]
    alphabet += [chr(cp) for cp in range(0x20, 0x7F)]
    alphabet += list(_FUZZ_WORDS)
    return alphabet


def random_texts(tables: dict[str, Any], count: int, seed: int) -> list[str]:
    alphabet = fuzz_alphabet(tables)
    rng = random.Random(seed)
    return ["".join(rng.choice(alphabet) for _ in range(rng.randint(1, 16))) for _ in range(count)]


def text_inputs(tables: dict[str, Any]) -> list[tuple[str, str, str]]:
    randoms = [
        (f"random_{index:02d}", text, "seeded random text (seed 1)")
        for index, text in enumerate(random_texts(tables, RANDOM_TEXT_CASES, seed=1))
    ]
    return [*tc.TEXT_CASES, *randoms]


def upstream_render(tokenizer: Any, prompt: str) -> str | None:
    """Upstream's `apply_chat_template` call (None where the template raises)."""
    from jinja2.exceptions import TemplateError

    try:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
    except TemplateError:
        return None


def upstream_ids(tokenizer: Any, text: str, *, truncate: bool) -> list[int]:
    if truncate:
        return list(tokenizer(text, truncation=True, max_length=MAX_LENGTH)["input_ids"])
    return list(tokenizer(text)["input_ids"])


def build_cases(
    tokenizer: Any, raw: Any, tables: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], q2.Qwen2Reference]:
    """Takes upstream's expectations and checks the mirror and the raw-file path against them.

    Returns the reference instance that encoded exactly the cases (its used vocabulary / merges
    are the fixture's subset).
    """
    ref = reference(tables)

    def check_text(name: str, text: str, want: list[int], limit: int | None) -> None:
        got = ref.encode(text, limit)
        if got != want:
            raise Qwen35TokenizerError(f"[{name}] the mirror differs\n  truth={want}\n  ref={got}")
        file_ids = raw.encode(text, add_special_tokens=False).ids
        file_ids = file_ids if limit is None else file_ids[:limit]
        if file_ids != want:
            raise Qwen35TokenizerError(f"[{name}] tokenizer.json differs from transformers")

    prompt_cases: list[dict[str, Any]] = []
    for name, prompt, why in prompt_inputs():
        rendered = upstream_render(tokenizer, prompt)
        if rendered is None:
            try:
                render_chat_prompt(prompt)
            except Qwen35TokenizerError:
                prompt_cases.append(
                    {"id": name, "why": why, "prompt": prompt, "reject": "template"}
                )
                continue
            raise Qwen35TokenizerError(f"[{name}] upstream refuses, the mirror renders")
        if render_chat_prompt(prompt) != rendered:
            raise Qwen35TokenizerError(
                f"[{name}] the mirror renders {render_chat_prompt(prompt)!r}"
            )
        ids = upstream_ids(tokenizer, rendered, truncate=True)
        tokens = len(upstream_ids(tokenizer, rendered, truncate=False))
        if name in BOUNDARY_TOKENS and tokens != BOUNDARY_TOKENS[name]:
            raise Qwen35TokenizerError(f"[{name}] has {tokens} tokens, not {BOUNDARY_TOKENS[name]}")
        check_text(name, rendered, ids, MAX_LENGTH)
        prompt_cases.append(
            {
                "id": name,
                "why": why,
                "prompt": prompt,
                "rendered": rendered,
                "ids": ids,
                "tokens": tokens,
            }
        )

    text_cases: list[dict[str, Any]] = []
    for name, text, why in text_inputs(tables):
        ids = upstream_ids(tokenizer, text, truncate=False)
        check_text(name, text, ids, None)
        text_cases.append({"id": name, "why": why, "text": text, "ids": ids})
    return prompt_cases, text_cases, ref


def verify_random_sweep(
    tokenizer: Any, tables: dict[str, Any], *, count: int, seed: int = 0
) -> dict[str, Any]:
    """Seeded random texts: the mirror must equal transformers (a gate; nothing is committed)."""
    ref = reference(tables)
    for index, text in enumerate(random_texts(tables, count, seed)):
        want = upstream_ids(tokenizer, text, truncate=False)
        got = ref.encode(text)
        if got != want:
            raise Qwen35TokenizerError(
                f"random text {index} differs\n  input={[hex(ord(c)) for c in text]}\n"
                f"  truth={want}\n  ref={got}"
            )
    return {"count": count, "seed": seed}


def python_space() -> list[int]:
    """The code points `str.strip()` removes (surrogates cannot be in a str)."""
    return [cp for cp in q2.ALL_CODEPOINTS if chr(cp).isspace()]


def build_fixture(
    tables: dict[str, Any],
    prompt_cases: list[dict[str, Any]],
    text_cases: list[dict[str, Any]],
    ref: q2.Qwen2Reference,
) -> dict[str, Any]:
    merge_rank = {pair: rank for rank, pair in enumerate(tables["merges"])}
    asset = build_asset(tables)
    return {
        "_doc": [
            "Qwen3.5 tokenizer and MOSS-Image chat prompt expectations (generated by",
            "tools/export-recipes/qwen3_5/tokenizer.py — the schema is in its docstring). The ids",
            "are transformers 5.14.1 AutoTokenizer's over the pinned MOSS-Image tokenizer; the",
            "rendered prompts are its apply_chat_template's. vocab / merges are the subset the",
            "cases need; the Unicode tables are whole.",
        ],
        "assetFormat": ASSET_FORMAT,
        "source": asset["source"],
        "versions": asset["versions"],
        "maxLength": MAX_LENGTH,
        "qwen": {
            "addedTokens": asset["addedTokens"],
            "classes": asset["classes"],
            "caseFold": asset["caseFold"],
            "nfcSegments": asset["nfcSegments"],
            "nfcCases": q2.nfc_fixture_cases(tables["nfcSegments"]),
            "vocab": {token: tables["vocab"][token] for token in sorted(ref.used_vocab)},
            "merges": sorted(
                ([left, right, merge_rank[(left, right)]] for left, right in ref.used_merges),
                key=lambda entry: entry[2],
            ),
        },
        "pythonSpace": python_space(),
        "promptCases": prompt_cases,
        "textCases": text_cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--fetch", action="store_true", help="download the pinned files if missing")
    parser.add_argument("--asset-out", type=Path, default=SERIES_ROOT / SERIES_NAME / ASSET_FILE)
    parser.add_argument("--fixture-out", type=Path, default=FIXTURE_PATH)
    parser.add_argument("--fuzz", type=int, default=50_000, help="random NFC checks")
    parser.add_argument("--id-fuzz", type=int, default=2_000, help="random id-sequence checks")
    args = parser.parse_args()

    from tokenizers import Tokenizer

    with tempfile.TemporaryDirectory() as scratch:
        directory = Path(scratch)
        pinned_tokenizer_files(directory, fetch=args.fetch)
        tokenizer = load_truth(directory)
        raw = Tokenizer.from_file(str(directory / "tokenizer.json"))
        if tokenizer.split_special_tokens:
            raise Qwen35TokenizerError("split_special_tokens is on — added tokens would not split")
        backend = backend_json(tokenizer)
        check_backend_shape(backend)
        tables = build_tables(backend, fuzz=args.fuzz)
        prompt_cases, text_cases, ref = build_cases(tokenizer, raw, tables)
        print(f"[cases] {len(prompt_cases)} prompts / {len(text_cases)} texts agree", flush=True)
        started = time.perf_counter()
        sweep = verify_random_sweep(tokenizer, tables, count=args.id_fuzz)
        print(f"[id-sweep] {sweep} ({time.perf_counter() - started:.1f}s)", flush=True)

    asset = build_asset(tables)
    args.asset_out.parent.mkdir(parents=True, exist_ok=True)
    args.asset_out.write_text(json.dumps(asset, ensure_ascii=False), encoding="utf-8")
    fixture = build_fixture(tables, prompt_cases, text_cases, ref)
    args.fixture_out.parent.mkdir(parents=True, exist_ok=True)
    args.fixture_out.write_text(
        json.dumps(fixture, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "stats": {**tables["stats"], "idSweep": sweep},
                "asset": {"path": str(args.asset_out), "bytes": args.asset_out.stat().st_size},
                "fixture": {
                    "path": str(args.fixture_out),
                    "bytes": args.fixture_out.stat().st_size,
                    "promptCases": len(prompt_cases),
                    "textCases": len(text_cases),
                    "vocabSubset": len(fixture["qwen"]["vocab"]),
                    "mergeSubset": len(fixture["qwen"]["merges"]),
                },
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
