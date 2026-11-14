"""Bakes Wan2.1's umT5 tokenizer and prompt preprocessing into an asset that TS only looks up, and
writes the parity fixtures (ADR 0119 stage 10a — decisions 1, 2 and 4).

    uv run --group wan --inexact python -m wan.umt5_tokenizer

Three outputs (**always all of them in one run** — unless they are built from the same tables, the
runtime asset and the fixtures go stale separately and "the tests are green but only the runtime
gets a different id sequence". The same practice as anima's `anima/demo.py`):

  ① Runtime asset `outputs/series/wan2.1-umt5-tokenizer/tokenizer.json` (untracked, about 8 MB)
       Vocabulary and scores (line number = id), added tokens, whitespace set, preprocessing
       tables (`promptClean`). Packaging into the distribution is stage 10d.
  ② Parity fixture `packages/models/tests/fixtures/wan-text/parity.json` (tracked)
       Per-case expectations (an id sequence or a rejection reason), the **subset** of the
       vocabulary needed to reproduce them and the full preprocessing tables.
  ③ Preprocessing sweep fixture `packages/models/tests/fixtures/wan-text/prompt-clean-sweep.json`
       (tracked) — holds the expectations of the random and all-code-point sweeps as digests (TS
       replays the same sequences).

After generating, run `vp fmt packages/models/tests/fixtures/wan-text/` from the repository root
(the committed form is what the formatter produces).

## Source of truth for the tokenizer (ADR 0119 decision 1)

The source of truth for the id sequence is the actual behaviour of `AutoTokenizer` in the pinned
transformers 5.14.1 (on `tokenizer/` of Wan's pinned revision). transformers 5 reads
`tokenizer.json` and then rebuilds the T5 pipeline
(`models/t5/tokenization_t5.py:112-127` — no normalizer, `WhitespaceSplit` + `Metaspace`, Unigram
with unk_id 2). The asset is baked from the rebuilt backend (`backend_tokenizer.to_str()`), and its
vocabulary, scores and added tokens are checked to be identical to `tokenizer.json`.

Out-of-vocabulary characters are rejected (decision 1). The out-of-vocabulary id differs between
transformers 5 (id 2 = `<s>`) and `tokenizer.json` (id 3 = `<unk>`), so fixture generation always
checks that **the two paths give the same id sequence for every accepted input** (a check that the
result does not depend on the version). Conversely, a case where the two paths diverge is recorded
as out-of-vocabulary (a divergence other than "2 vs 3 at the same position" fails).

Besides out-of-vocabulary characters, the two paths diverge in two more shapes (measured on
2026-10-03 — of 60,000 random strings, every divergent one contained an out-of-vocabulary
character, an added token or a U+2581 right after whitespace, and every string with none of them
matched):

  * **Added tokens inside the text** (304 of them, such as `</s>` and `<extra_id_*>`) —
    transformers 5 splits on whitespace first, so `a </s> b` gives `[289, 1, 748, 1]`;
    `tokenizer.json` keeps the whitespace before the added token as `▁` (273), giving
    `[289, 273, 1, 748, 1]`.
  * **`▁` (U+2581) right after whitespace** — transformers 5 drops the whitespace and then splits
    on `▁` (`a ▁b` → `▁a` `▁b`); `tokenizer.json` replaces the whitespace itself with `▁`
    (→ `▁a` `▁` `▁b`).

In line with what decision 1 protects (the same id sequence as the reference of any version), these
two are rejected like out-of-vocabulary characters (ADR 0119 decision 1 says added tokens inside the
text are "split out as upstream does", but the investigation only swept single characters and
missed that the versions diverge — we chose to reject).

The upper limit 512 is upstream's `max_sequence_length` (`wan.pipeline_ref.MAX_SEQUENCE_LENGTH`);
upstream silently truncates, karume rejects (decision 4). The lower limit 2 (empty or whitespace
only = `</s>` alone) is rejected too.

MUST: import transformers / tokenizers / huggingface_hub inside functions (the `wan` group is not in
the default sync — `tests/test_optional_group_imports.py`). Do not import torch either (a
tokenizer-only script must not pull in heavy dependencies — this is also why `wan.pipeline_ref` is
not imported).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from _shared.paths import REPO_ROOT, SERIES_ROOT
from wan import prompt_clean as pc
from wan.prompts import FIXED_PROMPTS
from wan.sources import DEFAULT_MODEL, SOURCES, WAN21_MODELS, WanSourceError

#: Location in the series (`<name>-tokenizer/tokenizer.json` — docs/assets-layout.md).
SERIES_NAME = "wan2.1-umt5-tokenizer"
ASSET_FILE = "tokenizer.json"

#: Version of the asset format (so TS does not silently read a version it does not know).
ASSET_FORMAT = "karume-wan-umt5-tokenizer/1"

#: Where the fixtures go (consumed by `packages/models/tests/wan_text_*_test.ts`).
FIXTURE_DIR = REPO_ROOT / "packages" / "models" / "tests" / "fixtures" / "wan-text"
PARITY_FIXTURE = "parity.json"
SWEEP_FIXTURE = "prompt-clean-sweep.json"

#: Upstream's `max_sequence_length` (the same value as `wan.pipeline_ref.MAX_SEQUENCE_LENGTH` — held
#: here so that torch is not loaded; pytest checks that they agree).
MAX_LENGTH = 512

#: Lower bound of the accepted set (avoids the 0 / 1 specialization of `torch.export` — the same as
#: anima's `PROMPT_MIN_TOKENS`).
MIN_TOKENS = 2

#: The pinned transformers version (the source of truth for id sequences).
TRANSFORMERS_VERSION = "5.14.1"

#: Metaspace replacement character (U+2581).
METASPACE = "▁"

#: Lengths of the four fixed accepted prompts (investigation §1.4, the values of ADR 0118 stage 6).
FIXED_PROMPT_LENGTHS = {"boxing-cats": 28, "ferret": 118, "cat-dog-baking": 50, "negative": 126}

#: Boundary cases (id, input, purpose). Always write invisible characters as \\uXXXX.
BOUNDARY_CASES: tuple[tuple[str, str, str], ...] = (
    ("min-two", "cat", "exactly the accepted lower bound (`▁cat` + `</s>`, 2 tokens)"),
    ("max-512", "cat " * 511, "exactly the accepted upper bound (511 + `</s>` = 512)"),
    ("over-513", "cat " * 512, "the upper bound + 1 is rejected (upstream silently truncates)"),
    ("empty", "", "the empty string is rejected at length 1 (`</s>` alone)"),
    (
        "only-space",
        " \n\t　 ",
        "whitespace only (full-width spaces collapse too) is rejected at length 1",
    ),
    (
        "oov",
        "𠀋𡈽",
        "out-of-vocabulary characters are rejected (transformers 5: id 2, tokenizer.json: id 3)",
    ),
    (
        "oov-inner",
        "a cat 𠀋 here",
        "rejected with an out-of-vocabulary character in the middle too",
    ),
    (
        "added-eos",
        "a </s> b",
        "added tokens inside the text are rejected (the two paths give different id sequences)",
    ),
    ("added-extra", "<extra_id_0> cat", "the added tokens `<extra_id_*>` are rejected too"),
    (
        "metaspace",
        "a ▁b",
        "`▁` right after whitespace is rejected (the two paths segment it differently)",
    ),
    (
        "metaspace-inner",
        "a▁b cat",
        "`▁` not right after whitespace gives the same result on the two paths",
    ),
    ("user-defined", "[web] page", "the user-defined piece `[web]` is ordinary vocabulary (id 5)"),
    (
        "fullwidth-comma",
        "色调艳丽，过曝",
        "the full-width commas of the official negative prompt become `,`",
    ),
    ("curly", "“quoted” it’s", "curly quotes become ASCII"),
    ("ligature", "ﬁne ŉ", "ligatures are decomposed (`ŉ` goes through `ʼn` to `'n`)"),
    ("ansi", "\x1b[31mred\x1b[0m", "terminal escapes are removed"),
    (
        "ansi-unicode-digit",
        "\x1b[\u0661mred",
        "`\\d` in a terminal escape is Unicode Nd (not only ASCII)",
    ),
    ("line-breaks", "a\r\nb\rc d", "line breaks are normalized and whitespace collapsed"),
    ("nfc", "é café", "NFC composition"),
    ("accents", "café naïve résumé Zürich São Paulo", "accented Latin text is accepted"),
    (
        "width",
        "ＡＢＣ　ｶﾀｶﾅ",
        "full-width Latin letters and half-width katakana go to their standard forms",
    ),
    ("controls", "a\x07b﻿c", "control characters and the BOM are removed"),
    ("zero-width", "x​y", "a zero-width space is not whitespace (it stays)"),
    ("japanese", "夕焼けの海と猫", "Japanese"),
    ("ampersand-spaced", "salt & pepper", "`&` with whitespace around it is accepted"),
    (
        "mojibake",
        "cafÃ©",
        "rejected when the mojibake check is true (upstream repairs it by guessing)",
    ),
    ("entity-named", "salt &amp; pepper", "HTML character references are rejected"),
    (
        "entity-bare",
        "R&D",
        "a sequence that is not an entity is rejected too when a letter follows `&`",
    ),
    ("c1", "a\x85b", "C1 control characters are rejected"),
    (
        "unassigned",
        "a͸b",
        "characters unassigned in UCD 16.0.0 are rejected (U+0378 is in the vocabulary)",
    ),
)

#: Random cases (only characters of the accepted set — fixed seed).
RANDOM_CASES = 200
RANDOM_SEED = 5
RANDOM_MAX_LENGTH = 40
#: Weight of the whitespace mixed into the random pool (creates word breaks).
RANDOM_SPACE_WEIGHT = 2000


def tokenizer_snapshot(model: str = DEFAULT_MODEL) -> Path:
    """The `tokenizer/` directory of the snapshot at the pinned revision (does not go to the
    network — the location {@link load_truth} reads)."""
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    source = SOURCES[model]
    try:
        snapshot = Path(
            snapshot_download(
                source.repo,
                revision=source.revision,
                allow_patterns=["tokenizer/*"],
                local_files_only=True,
            )
        )
    except LocalEntryNotFoundError as error:
        raise WanSourceError(
            f"tokenizer/ of {source.repo}@{source.revision} is not in the HF cache — fetch it first"
            " with `uv run --group wan --inexact python -m wan.text_embeds --fetch`"
        ) from error
    if not (snapshot / "tokenizer" / "tokenizer.json").is_file():
        raise WanSourceError(f"{snapshot}/tokenizer/tokenizer.json is missing")
    return snapshot / "tokenizer"


@dataclass(frozen=True)
class Truth:
    """The two sources of truth (`AutoTokenizer` of transformers 5.14.1 and `tokenizer.json`)."""

    #: The id sequence of `tokenizer([text], add_special_tokens=True)` (not truncated).
    transformers: Callable[[str], list[int]]
    #: The mask length under upstream's call (`padding="max_length"`, `truncation=True`).
    masked_length: Callable[[str], int]
    #: The id sequence of `tokenizers.Tokenizer.from_file(tokenizer.json)`.
    tokenizer_json: Callable[[str], list[int]]
    #: The backend after transformers rebuilt it (the JSON of `to_str()`).
    backend: dict[str, Any]
    #: The contents of `tokenizer.json`.
    raw: dict[str, Any]
    #: Runs only the pre-tokenization (`WhitespaceSplit` → `Metaspace`).
    pre_tokenize: Callable[[str], list[str]]


def load_truth(directory: Path) -> Truth:
    """The two sources of truth of `directory` (a location that holds `tokenizer.json` and
    `tokenizer_config.json` directly).

    Wan passes the snapshot's `tokenizer/` ({@link tokenizer_snapshot}); `karume-umt5-xxl` passes a
    location holding only the two verified files copied from the original pin
    (`wan.umt5_xxl_tokenizer`).
    """
    import transformers
    from tokenizers import Tokenizer
    from transformers import AutoTokenizer

    if transformers.__version__ != TRANSFORMERS_VERSION:
        raise ValueError(
            f"transformers {transformers.__version__} — the source of truth for id sequences is"
            f" {TRANSFORMERS_VERSION}"
        )
    path = directory / "tokenizer.json"
    auto = AutoTokenizer.from_pretrained(directory)
    plain = Tokenizer.from_file(str(path))
    backend = auto.backend_tokenizer

    def masked_length(text: str) -> int:
        encoded = auto(
            [text],
            padding="max_length",
            max_length=MAX_LENGTH,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
        )
        return sum(encoded["attention_mask"][0])

    return Truth(
        transformers=lambda text: list(auto([text], add_special_tokens=True)["input_ids"][0]),
        masked_length=masked_length,
        tokenizer_json=lambda text: list(plain.encode(text).ids),
        backend=json.loads(backend.to_str()),
        raw=json.loads(path.read_text(encoding="utf-8")),
        pre_tokenize=lambda text: [
            piece for piece, _ in backend.pre_tokenizer.pre_tokenize_str(text)
        ],
    )


def check_upstream_shape(truth: Truth) -> None:
    """Checks the backend structure that the baked tables assume (fail loudly).

    TS holds only "split out added tokens → WhitespaceSplit → Metaspace (▁, always, split) → Unigram
    → trailing `</s>`". If the backend departs from this shape in a single field, the TS id sequence
    silently becomes something else.
    """
    backend = truth.backend
    if backend["normalizer"] is not None:
        raise ValueError(f"the backend has a normalizer: {backend['normalizer']}")
    expected_pre = {
        "type": "Sequence",
        "pretokenizers": [
            {"type": "WhitespaceSplit"},
            {
                "type": "Metaspace",
                "replacement": METASPACE,
                "prepend_scheme": "always",
                "split": True,
            },
        ],
    }
    if backend["pre_tokenizer"] != expected_pre:
        raise ValueError(
            f"the backend's pre_tokenizer is not the expected one: {backend['pre_tokenizer']}"
        )
    model = backend["model"]
    if model["type"] != "Unigram" or model["byte_fallback"] or model["unk_id"] != 2:
        raise ValueError(
            f"the backend's model is not Unigram with unk_id 2 and no byte_fallback: {model}"
        )
    post = backend["post_processor"]
    specials = [entry["SpecialToken"]["id"] for entry in post["single"] if "SpecialToken" in entry]
    if post["type"] != "TemplateProcessing" or specials != ["</s>"]:
        raise ValueError(f"the post_processor does not add exactly one `</s>`: {post}")
    if post["single"][-1] != {"SpecialToken": {"id": "</s>", "type_id": 0}}:
        raise ValueError(f"`</s>` is not appended at the end: {post['single']}")
    # If an added token's flags are not the defaults, the copied rule "split out leftmost-longest
    # before normalization" means something else.
    for entry in backend["added_tokens"]:
        flags = {key: entry[key] for key in ("lstrip", "rstrip", "single_word", "normalized")}
        if any(flags.values()):
            raise ValueError(
                f"the flags of added token {entry['content']!r} are not the defaults: {flags}"
            )
    # Vocabulary, scores and added tokens are identical to tokenizer.json (transformers 5 rebuilds
    # only the configuration).
    if backend["model"]["vocab"] != truth.raw["model"]["vocab"]:
        raise ValueError("the backend's vocabulary / scores differ from tokenizer.json")
    if backend["added_tokens"] != truth.raw["added_tokens"]:
        raise ValueError("the backend's added tokens differ from tokenizer.json")


def whitespace_split_ranges() -> pc.Ranges:
    """Collects the characters `WhitespaceSplit` treats as whitespace (Rust's `char::is_whitespace`)
    into ranges.

    Puts every code point into `a` + character + `b` and asks the source of truth whether it splits
    into two pieces.
    """
    from tokenizers import pre_tokenizers

    splitter = pre_tokenizers.WhitespaceSplit()
    return pc.ranges_where(lambda ch: len(splitter.pre_tokenize_str("a" + ch + "b")) > 1)


def t5_table_fields(truth: Truth) -> dict[str, Any]:
    """The T5 table fields of the asset (the 7 fields TS's `parseT5Tables` reads — in the asset's
    field order).

    Wan's asset ({@link build_asset}) and the `karume-umt5-xxl` asset (`wan.umt5_xxl_tokenizer`)
    both use this one function (there are not two ways of baking the tables). The backend structure
    check ({@link check_upstream_shape}) also runs here.
    """
    check_upstream_shape(truth)
    vocab = [(token, float(score)) for token, score in truth.backend["model"]["vocab"]]
    for token, _ in vocab:
        if "\n" in token or token == "":
            raise ValueError(
                f"a vocabulary token {token!r} contains a newline / is empty — the line"
                " separation breaks"
            )
    added = sorted(
        ((entry["content"], entry["id"]) for entry in truth.backend["added_tokens"]),
        key=lambda pair: pair[1],
    )
    eos = dict(added)["</s>"]
    return {
        "maxLength": MAX_LENGTH,
        "unkId": truth.backend["model"]["unk_id"],
        "eosId": eos,
        "addedTokens": [[content, token_id] for content, token_id in added],
        "space": whitespace_split_ranges(),
        # Line number, 0-origin = id (checked above that no token contains a newline or is empty).
        "vocabText": "\n".join(token for token, _ in vocab),
        "scores": [score for _, score in vocab],
    }


def build_asset(
    truth: Truth, tables: pc.PromptCleanTables, versions: Mapping[str, str], model: str
) -> dict[str, Any]:
    """The asset TS only looks up (the shape `parseWanTokenizerAsset` reads)."""
    source = SOURCES[model]
    return {
        "format": ASSET_FORMAT,
        "source": {"repo": source.repo, "revision": source.revision, "subfolder": "tokenizer"},
        "versions": dict(sorted(versions.items())),
        **t5_table_fields(truth),
        "promptClean": tables.to_json(),
    }


def asset_versions() -> dict[str, str]:
    """The asset's `versions` (the versions of the preprocessing source of truth + the tokenizers /
    transformers versions — the values {@link main} bakes)."""
    import tokenizers
    import transformers

    return {
        **pc.check_versions(),
        "tokenizers": tokenizers.__version__,
        "transformers": transformers.__version__,
    }


def split_added(text: str, added: Sequence[str]) -> list[tuple[str, bool]]:
    """Splits out added tokens leftmost-longest (mirror of TS's `splitAddedTokens`)."""
    out: list[tuple[str, bool]] = []
    buffer = ""
    i = 0
    while i < len(text):
        hit = max((token for token in added if text.startswith(token, i)), key=len, default=None)
        if hit is None:
            buffer += text[i]
            i += 1
            continue
        if buffer:
            out.append((buffer, False))
            buffer = ""
        out.append((hit, True))
        i += len(hit)
    if buffer:
        out.append((buffer, False))
    return out


def pieces_of(truth: Truth, added: Sequence[str], text: str) -> list[str]:
    """The pieces outside the added tokens (the units handed to Unigram)."""
    return [
        piece
        for chunk, is_added in split_added(text, added)
        if not is_added
        for piece in truth.pre_tokenize(chunk)
    ]


def version_split(added: Sequence[str], text: str) -> str | None:
    """The reason for a shape rejected before the tokenizer (a shape whose pre-tokenization diverges
    between the two paths) — added tokens inside the text, then `▁` right after whitespace. None if
    neither is present."""
    if any(is_added for _, is_added in split_added(text, added)):
        return "added-token"
    if " " + METASPACE in text:
        return "metaspace"
    return None


def expectation(truth: Truth, added: Sequence[str], cleaned: str) -> dict[str, Any]:
    """The expectation for a string that passed preprocessing (an id sequence, or the tokenizer
    side's rejection reason).

    The rejection order is the same as in TS: added token → `▁` right after whitespace →
    out-of-vocabulary / over the upper limit → under the lower limit.
    """
    reason = version_split(added, cleaned)
    if reason is not None:
        return {"reject": reason}
    return tokenized_expectation(truth, cleaned)


def tokenized_expectation(truth: Truth, text: str) -> dict[str, Any]:
    """The expectation for a string that passed {@link version_split} (the id sequence of the two
    paths, or out-of-vocabulary / over the upper limit / under the lower limit).

    MUST: accept only when the two paths give the same id sequence. Once added tokens and `▁` right
    after whitespace are excluded, a divergence is out-of-vocabulary, and the only possible shape is
    "same length, transformers 5 has 2 and tokenizer.json has 3 at the same positions" (anything
    else means the paths are configured differently = fail).
    """
    ids = truth.transformers(text)
    plain = truth.tokenizer_json(text)
    if ids != plain:
        if len(ids) != len(plain) or any(
            (a, b) != (2, 3) for a, b in zip(ids, plain, strict=True) if a != b
        ):
            raise ValueError(
                "the two paths differ in a way other than out-of-vocabulary (2 vs 3):"
                f" {text!r}\n  tf5={ids}\n  json={plain}"
            )
        if len(ids) > MAX_LENGTH:
            raise ValueError(
                f"a case that is both out-of-vocabulary and over the limit is not kept: {text!r}"
            )
        return {"reject": "oov"}
    if len(ids) > MAX_LENGTH:
        return {"reject": "too-long"}
    if len(ids) < MIN_TOKENS:
        return {"reject": "too-short"}
    if truth.masked_length(text) != len(ids):
        raise ValueError(
            f"the mask length under upstream's call differs from the id sequence length: {text!r}"
        )
    return {"ids": ids, "tokenizerJsonIds": plain}


def build_case(
    truth: Truth,
    compiled: pc.CompiledTables,
    clean_truth: Callable[[str], str],
    case_id: str,
    text: str,
    why: str,
) -> dict[str, Any]:
    """The expectation of one case. Preprocessing rejection → tokenizer rejection → acceptance, in
    that order (the same as TS)."""
    outcome = pc.record(compiled, clean_truth, text)
    base = {"id": case_id, "why": why, "text": text}
    if outcome.startswith("R"):
        return {**base, "reject": outcome[1:]}
    cleaned = outcome[1:]
    added = [entry["content"] for entry in truth.backend["added_tokens"]]
    return {**base, "cleaned": cleaned, **expectation(truth, added, cleaned)}


def random_case_pool(truth: Truth, tables: pc.PromptCleanTables) -> list[int]:
    """The pool of the random cases — the vocabulary entries that are a single character and touch
    neither the preprocessing checks nor the table lookups, plus whitespace."""
    singles = {ord(token) for token, _ in truth.backend["model"]["vocab"] if len(token) == 1}
    assigned = pc.RangeSet(tables.assigned)
    literals = set(pc.badness_literals())
    control = pc.RangeSet(tables.control_chars)
    space = pc.RangeSet(tables.collapse_space)
    chars = sorted(
        cp
        for cp in singles
        if cp in assigned
        and cp not in literals
        and cp != ord("&")
        and cp not in tables.width
        and cp not in tables.ligatures
        and cp not in control
        and cp not in space
        and not pc.C1_FIRST <= cp <= pc.C1_LAST
    )
    return [*chars, *([0x20] * RANDOM_SPACE_WEIGHT)]


def build_cases(
    truth: Truth,
    tables: pc.PromptCleanTables,
    clean_truth: Callable[[str], str],
    *,
    random_count: int = RANDOM_CASES,
) -> list[dict[str, Any]]:
    """Builds the expectations of all cases (the four fixed ones + boundaries + random)."""
    compiled = pc.CompiledTables(tables)
    cases = [
        build_case(truth, compiled, clean_truth, f"fixed-{p.name}", p.text, f"fixed {p.role}")
        for p in FIXED_PROMPTS
    ]
    for case in cases:
        name = case["id"].removeprefix("fixed-")
        if len(case.get("ids", [])) != FIXED_PROMPT_LENGTHS[name]:
            raise ValueError(
                f"the length of fixed prompt {name} differs from the investigation's value: {case}"
            )
    cases += [
        build_case(truth, compiled, clean_truth, case_id, text, why)
        for case_id, text, why in BOUNDARY_CASES
    ]
    pool = random_case_pool(truth, tables)
    for index, text in enumerate(
        pc.fuzz_strings(pool, RANDOM_SEED, random_count, RANDOM_MAX_LENGTH)
    ):
        cases.append(
            build_case(
                truth,
                compiled,
                clean_truth,
                f"random-{index}",
                text,
                "random (characters of the accepted set)",
            )
        )
    check_boundaries(cases)
    return cases


def check_boundaries(cases: Sequence[Mapping[str, Any]]) -> None:
    """Checks that each boundary case ends with its intended outcome (a case that misses its
    purpose is not baked)."""
    by_id = {case["id"]: case for case in cases}
    wants = {
        "min-two": lambda c: len(c.get("ids", [])) == MIN_TOKENS,
        "max-512": lambda c: len(c.get("ids", [])) == MAX_LENGTH,
        "over-513": lambda c: c.get("reject") == "too-long",
        "empty": lambda c: c.get("reject") == "too-short",
        "only-space": lambda c: c.get("reject") == "too-short",
        "oov": lambda c: c.get("reject") == "oov",
        "oov-inner": lambda c: c.get("reject") == "oov",
        "ansi-unicode-digit": lambda c: c.get("cleaned") == "red",
        "added-eos": lambda c: c.get("reject") == "added-token",
        "added-extra": lambda c: c.get("reject") == "added-token",
        "metaspace": lambda c: c.get("reject") == "metaspace",
        "metaspace-inner": lambda c: "ids" in c,
        "fullwidth-comma": lambda c: c.get("cleaned") == "色调艳丽,过曝",
        "mojibake": lambda c: c.get("reject") == "mojibake",
        "entity-named": lambda c: c.get("reject") == "entity",
        "entity-bare": lambda c: c.get("reject") == "entity",
        "c1": lambda c: c.get("reject") == "c1",
        "unassigned": lambda c: c.get("reject") == "unassigned",
    }
    for case_id, want in wants.items():
        if not want(by_id[case_id]):
            raise ValueError(
                f"boundary case {case_id} does not end with its intended outcome: {by_id[case_id]}"
            )
    accepted = [case for case in cases if case["id"].startswith("random-") and "ids" in case]
    if len(accepted) < len([c for c in cases if c["id"].startswith("random-")]) // 2:
        raise ValueError(
            "most random cases are rejected — the pool is not drawn from the accepted set"
        )


def vocab_subset(truth: Truth, texts: Iterable[str]) -> list[list[Any]]:
    """The vocabulary entries that are **substrings** of the pieces of all texts (the lattice of
    each piece is then the same as with the full vocabulary).

    Keeping only the pieces used by the segmentation can change how unknown nodes enter and how ties
    are won in the lattice of the subset. Keeping every substring of the pieces makes the lattice
    itself identical (the same idea as `subset_keys` in gemma's `_shared/gemma_tokenizer.py`).
    """
    vocab = {
        token: (token_id, float(score))
        for token_id, (token, score) in enumerate(truth.backend["model"]["vocab"])
    }
    added = [entry["content"] for entry in truth.backend["added_tokens"]]
    longest = max(len(token) for token in vocab)
    keep: set[str] = set()
    for text in texts:
        for piece in pieces_of(truth, added, text):
            for start in range(len(piece)):
                for end in range(start + 1, min(len(piece), start + longest) + 1):
                    if piece[start:end] in vocab:
                        keep.add(piece[start:end])
    return [[token, *vocab[token]] for token in sorted(keep, key=lambda t: vocab[t][0])]


def build_parity_fixture(
    truth: Truth,
    asset: Mapping[str, Any],
    cases: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """The id sequence parity fixture (a subset of the vocabulary, the full preprocessing
    tables)."""
    scores = asset["scores"]
    tokens = asset["vocabText"].split("\n")
    return {
        "_doc": [
            "Parity fixture of the Wan2.1 umT5 tokenizer and preprocessing"
            " (generated by tools/export-recipes/wan/umt5_tokenizer.py).",
            "The source of truth for id sequences is transformers 5.14.1's AutoTokenizer. Accepted",
            "cases also carry the id sequence of the tokenizer.json path (checked at generation",
            "that the two paths agree). The vocabulary is only the substrings of the pieces of all",
            "cases; the preprocessing tables (promptClean) are the folded output itself, so they"
            " are kept whole.",
        ],
        "source": asset["source"],
        "versions": asset["versions"],
        "maxLength": asset["maxLength"],
        "minTokens": MIN_TOKENS,
        "t5": {
            "addedTokens": asset["addedTokens"],
            "unkId": asset["unkId"],
            "eosId": asset["eosId"],
            # MUST: the unknown-node score and the search width come from the **whole vocabulary**
            # (never derived from the subset).
            "minScore": min(scores),
            "maxTokenLength": max(len(token) for token in tokens),
            "space": asset["space"],
            "vocab": vocab_subset(truth, [case["cleaned"] for case in cases if "cleaned" in case]),
        },
        "promptClean": asset["promptClean"],
        "cases": list(cases),
    }


def build_sweep_fixture(
    tables: pc.PromptCleanTables,
    clean_truth: Callable[[str], str],
    versions: Mapping[str, str],
    *,
    clean_runs: Sequence[pc.FuzzRun] = pc.CLEAN_RUNS,
    badness_runs: Sequence[pc.FuzzRun] = pc.BADNESS_RUNS,
    badness_probes: Sequence[tuple[str, str]] = pc.BADNESS_PROBES,
    clean_sweeps: Sequence[tuple[str, str]] = pc.CLEAN_SWEEPS,
    nfc_sweeps: Sequence[tuple[str, str]] = pc.NFC_SWEEPS,
    log: Callable[[str], None] = lambda _: None,
) -> dict[str, Any]:
    """The expectations (digests) of the preprocessing random and all-code-point sweeps.

    The expectations always come from upstream (`prompt_clean`, `BADNESS_RE`, `unicodedata`). The
    mirror only decides which inputs to reject; for an accepted input, {@link pc.record} fails
    unless it matches upstream.
    """
    import unicodedata

    from ftfy.badness import BADNESS_RE

    compiled = pc.CompiledTables(tables)
    pools = pc.fuzz_pools(tables)
    expanded = {name: pc.expand(ranges) for name, ranges in pools.items()}
    rng = pc.XorShift32(1)
    first = [rng.next() for _ in range(8)]

    clean = []
    for run in clean_runs:
        started = time.perf_counter()
        clean.append(pc.verify_fuzz(compiled, clean_truth, run, expanded[run.pool]))
        seconds = time.perf_counter() - started
        log(f"[clean-fuzz] {run.name}: {clean[-1]['outcomes']}（{seconds:.1f}s）")

    badness = []
    for run in badness_runs:
        texts = list(pc.fuzz_strings(expanded[run.pool], run.seed, run.count, run.max_length))
        bits = pc.badness_bits(BADNESS_RE, texts)
        if bits != pc.badness_bits(compiled.badness, texts):
            raise ValueError(
                "the intermediate form of BADNESS_RE diverges from upstream"
                f" (random run {run.name})"
            )
        badness.append(
            {
                "name": run.name,
                "pool": run.pool,
                "seed": run.seed,
                "count": run.count,
                "maxLength": run.max_length,
                "digest": pc.bits_digest(bits),
                "hits": bits.count("1"),
            }
        )
    for name, template in badness_probes:
        texts = [template.replace("{}", chr(cp)) for cp in pc.ALL_CODEPOINTS]
        bits = pc.badness_bits(BADNESS_RE, texts)
        if bits != pc.badness_bits(compiled.badness, texts):
            raise ValueError(
                f"the intermediate form of BADNESS_RE diverges from upstream (sweep {name})"
            )
        badness.append(
            {
                "name": name,
                "template": template,
                "digest": pc.bits_digest(bits),
                "hits": bits.count("1"),
            }
        )
    log(f"[badness] {[(entry['name'], entry['hits']) for entry in badness]}")

    clean_blocks = []
    for name, template in clean_sweeps:
        started = time.perf_counter()
        blocks = pc.sweep_records(compiled, clean_truth, template)
        clean_blocks.append({"name": name, "template": template, "blocks": blocks})
        log(f"[clean-sweep] {name}（{time.perf_counter() - started:.1f}s）")

    assigned = pc.RangeSet(tables.assigned)
    domain = [cp for cp in pc.ALL_CODEPOINTS if cp in assigned]
    nfc = [
        {
            "name": name,
            "template": template,
            "blocks": pc.sweep_blocks(
                domain,
                lambda cp, template=template: unicodedata.normalize(
                    "NFC", template.replace("{}", chr(cp))
                ),
            ),
        }
        for name, template in nfc_sweeps
    ]
    log(f"[nfc-sweep] {len(nfc)} contexts × {len(domain)} characters")

    return {
        "_doc": [
            "Expected values of the random and all-code-point sweeps of the Wan2.1 preprocessing"
            " (the mirror of prompt_clean; generated by"
            " tools/export-recipes/wan/umt5_tokenizer.py).",
            "The expected values come from upstream (diffusers 0.39.0's prompt_clean, ftfy 6.3.1's",
            "BADNESS_RE, unicodedata 16.0.0's NFC); the SHA-256 of the record sequence is kept",
            "per block. TS replays the same sequence from the same PRNG (xorshift32) and pools",
            "and compares. A record is A + the output when accepted, R + the reason when",
            "rejected; each block hashes the UTF-8 of its records, each followed by U+0000.",
        ],
        "versions": dict(sorted(versions.items())),
        "prng": {"name": "xorshift32", "seed": 1, "first": first},
        "block": pc.SWEEP_BLOCK,
        "pools": pools,
        "clean": clean,
        "badness": badness,
        "cleanSweeps": clean_blocks,
        "nfcSweeps": nfc,
    }


#: The line width of the repository formatter (`vp fmt` — Oxfmt's default print width, which
#: `vite.config.ts` does not override).
FMT_LINE_WIDTH = 100


def _display_width(text: str) -> int:
    import unicodedata

    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in text)


def fixture_json(value: Any, indent: int = 0) -> str:
    """The fixture's JSON (close to the Oxfmt form — an array that fits the line width on one
    line, anything else one element per line).

    Plain `json.dumps` with `indent` expands even the ranges `[start, end]` and the id sequences to
    one element per line, and the fixture grows several times over. Oxfmt puts an array on one line
    whenever it fits the line width, so writing it on one line here keeps the written file close to
    the committed form (the formatter's output) and small.
    """
    pad = " " * (indent + 2)
    if isinstance(value, dict):
        if not value:
            return "{}"
        items = [
            f"{pad}{json.dumps(key, ensure_ascii=False)}: {fixture_json(item, indent + 2)}"
            for key, item in value.items()
        ]
        return "{\n" + ",\n".join(items) + "\n" + " " * indent + "}"
    if isinstance(value, list):
        single = json.dumps(value, ensure_ascii=False, separators=(", ", ": "))
        flat = not any(isinstance(item, dict) for item in value)
        if flat and indent + _display_width(single) + 1 <= FMT_LINE_WIDTH:
            return single
        items = [f"{pad}{fixture_json(item, indent + 2)}" for item in value]
        return "[\n" + ",\n".join(items) + "\n" + " " * indent + "]"
    return json.dumps(value, ensure_ascii=False)


def write_text(path: Path, text: str) -> int:
    """Writes the file and returns its size in bytes (staging → replace — a failure midway leaves
    the old file in place)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(path.name + ".staging")
    staging.write_text(text, encoding="utf-8")
    staging.replace(path)
    return path.stat().st_size


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--model", default=DEFAULT_MODEL, choices=WAN21_MODELS)
    parser.add_argument("--assets-out", type=Path, default=SERIES_ROOT / SERIES_NAME)
    parser.add_argument("--fixtures-out", type=Path, default=FIXTURE_DIR)
    args = parser.parse_args(argv)

    def log(message: str) -> None:
        print(message, flush=True)

    started = time.perf_counter()
    truth = load_truth(tokenizer_snapshot(args.model))
    tables = pc.build_tables()
    versions = asset_versions()
    asset = build_asset(truth, tables, versions, args.model)
    log(f"[asset] vocabulary {len(asset['scores'])} ({time.perf_counter() - started:.1f}s)")

    clean_truth = pc.upstream_prompt_clean()
    cases = build_cases(truth, tables, clean_truth)
    outcomes = sorted({case.get("reject", "accepted") for case in cases})
    log(f"[cases] {len(cases)} cases ({outcomes})")
    parity = build_parity_fixture(truth, asset, cases)
    sweep = build_sweep_fixture(tables, clean_truth, versions, log=log)

    # Write only after every check has passed (ADR 0005's fail loudly — a failure midway writes
    # not a single byte).
    written = {
        "asset": write_text(args.assets_out / ASSET_FILE, json.dumps(asset, ensure_ascii=False)),
        "parity": write_text(args.fixtures_out / PARITY_FIXTURE, fixture_json(parity) + "\n"),
        "sweep": write_text(args.fixtures_out / SWEEP_FIXTURE, fixture_json(sweep) + "\n"),
    }
    log(
        json.dumps(
            {
                "bytes": written,
                "vocabSubset": len(parity["t5"]["vocab"]),
                "clean": [{run["name"]: run["outcomes"]} for run in sweep["clean"]],
                "maxRounds": max(run["maxRounds"] for run in sweep["clean"]),
                "seconds": round(time.perf_counter() - started, 1),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
