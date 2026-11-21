"""Compile of the EmbeddingGemma 2 tokenizer asset and harvest of the parity fixture.

    uv run --with 'transformers==5.23.0' python -m embeddinggemma2.tokenizer

Runs on the same machinery as gemma4 and the first EmbeddingGemma (`_shared/gemma_tokenizer.py`,
ADR 0084 decision 6). Measured on the pinned snapshot (2026-11-21): `tokenizer.json` is Gemma 4's
file from before Google's glitch-token fixes (the one `gemma4.sources` pins for 26B A4B, SHA-256
`cc8d3a0c…`) in every field but `post_processor` — the same 262,144 vocabulary rows in the same
order, the same 514,906 merges, the same 24 added tokens, normalizer, pre-tokenizer and decoder —
while the post-processor wraps the sequence in `<bos>` … `<eos>` (the pair template wraps each
sequence the same way). So the asset is its own: a Gemma 4 asset gives other ids with special
tokens, and the first EmbeddingGemma's vocabulary is another one (6,415 added tokens).

Two outputs (**always both in one run** — {@link _shared.gemma_tokenizer.emit}):

  ① runtime asset `outputs/series/embeddinggemma-2-tokenizer/tokenizer.json`
  ② parity fixture
     `packages/models/tests/fixtures/gemma-text/embeddinggemma-2-parity.json` (**tracked in git**)

Before anything is written the script checks that

- the files it reads are the pinned snapshot's ({@link embeddinggemma2.sources.assert_pinned}),
- the prompts of `config_sentence_transformers.json` are exactly {@link PROMPT_PREFIXES} (each
  has a case in {@link CASES} — a prompt added upstream fails here instead of going untested),
- transformers is {@link REQUIRED_TRANSFORMERS}, and its tokenizer for the snapshot gives the ids
  `tokenizers` gives for every case, with and without special tokens ({@link cross_check}). The
  fixture's expected values come from `tokenizers` alone (ADR 0084 decision 7); the cross-check
  shows the transformers tokenizer, the one upstream runs the model with, produces them too,
- the long case ({@link LONG_CASE}) is at least {@link MIN_LONG_TOKENS} tokens with `<bos>` …
  `<eos>` (past the 512-position sliding window of the model).

The shared battery ({@link _shared.gemma_tokenizer.SHARED_ENCODE_CASES}) supplies the plain texts
in several scripts and the empty text; {@link CASES} adds what is specific to this checkpoint.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from _shared.gemma_tokenizer import SHARED_ENCODE_CASES, EncodeCase, emit
from _shared.paths import REPO_ROOT, SERIES_ROOT
from embeddinggemma2 import sources

#: The series of the compiled asset (assets-layout: `<series name>-tokenizer/tokenizer.json`).
TOKENIZER_SERIES = "embeddinggemma-2-tokenizer"

#: The runtime asset (series output).
ASSET_PATH = SERIES_ROOT / TOKENIZER_SERIES / "tokenizer.json"

#: The parity fixture (tracked in git — read by `packages/models/tests/gemma4_tokenizer_test.ts`).
FIXTURE_PATH = (
    REPO_ROOT
    / "packages"
    / "models"
    / "tests"
    / "fixtures"
    / "gemma-text"
    / "embeddinggemma-2-parity.json"
)

#: The snapshot files the script reads: the tokenizer and what the transformers tokenizer is
#: loaded from ({@link cross_check}), and the prompts.
READ_FILES: tuple[str, ...] = (
    "tokenizer.json",
    "tokenizer_config.json",
    "config.json",
    "config_sentence_transformers.json",
)

#: The transformers release this family is pinned to (added per run with `uv run --with`).
REQUIRED_TRANSFORMERS = "5.23.0"

#: The fewest tokens (`<bos>` and `<eos>` included) the long case must give.
MIN_LONG_TOKENS = 600


class TokenizerScriptError(ValueError):
    """An input of the script is not what the fixture is built for (fails before writing)."""


#: Prompt name (slug) → the prefix and a text after it. One row per distinct prefix of
#: `config_sentence_transformers.json` (its 20 prompt names share 8 prefixes); the texts reuse the
#: first EmbeddingGemma's two and add one script per remaining prefix.
_PROMPTED: tuple[tuple[str, str, str, str], ...] = (
    ("query", "task: search result | query: ", "English", "What is the capital of Japan?"),
    (
        "document",
        "title: none | text: ",
        "Japanese",
        "東京は日本の首都であり、世界有数の大都市圏を形成している。",
    ),
    (
        "classification",
        "task: classification | query: ",
        "Korean",
        "이 영화는 정말 재미있었고 배우들의 연기도 훌륭했다.",
    ),
    (
        "clustering",
        "task: clustering | query: ",
        "Arabic (right to left)",
        "تعلم الآلة هو فرع من فروع الذكاء الاصطناعي.",
    ),
    (
        "code-retrieval",
        "task: code retrieval | query: ",
        "source code (indentation and a comment)",
        "def add(a, b):\n    return a + b  # the sum",
    ),
    (
        "fact-checking",
        "task: fact checking | query: ",
        "Devanagari (combining vowel signs)",
        "पृथ्वी सूर्य के चारों ओर घूमती है।",
    ),
    (
        "sentence-similarity",
        "task: sentence similarity | query: ",
        "Chinese (full-width punctuation)",
        "今天的天气非常好，我们去公园散步吧。",
    ),
    (
        "question-answering",
        "task: question answering | query: ",
        "Spanish (inverted marks and accents)",
        "¿Cuál es la montaña más alta del mundo?",
    ),
)

#: The distinct prompt prefixes of `config_sentence_transformers.json` at the pin.
PROMPT_PREFIXES: tuple[str, ...] = tuple(prefix for _, prefix, _, _ in _PROMPTED)

#: The long case: a document longer than the model's sliding window. Built from numbered
#: sentences, so every chunk of it differs from the others (digits split one per token).
LONG_CASE = EncodeCase(
    "long-past-window",
    f"a document of at least {MIN_LONG_TOKENS} tokens with `<bos>` … `<eos>` — past the"
    " 512-position sliding window, where a length limit or a quadratic scan would show",
    "title: none | text: "
    + " ".join(
        f"Note {n}: the archive keeps {n * 37 % 1000} pages on topic {chr(65 + n % 26)}."
        for n in range(1, 61)
    ),
)

#: The encode cases specific to EmbeddingGemma 2 (added to the shared battery).
CASES: tuple[EncodeCase, ...] = (
    *(
        EncodeCase(
            f"prompt-{slug}",
            f"the prompt prefix {prefix!r} of `config_sentence_transformers.json` before {kind}"
            " text — the real-world shape `<bos>` prompt text `<eos>`",
            prefix + text,
        )
        for slug, prefix, kind, text in _PROMPTED
    ),
    EncodeCase(
        "prompt-only",
        "a prompt prefix with an empty text (the input ends on the prefix's trailing space)",
        PROMPT_PREFIXES[0],
    ),
    EncodeCase(
        "gemma4-turn-markup",
        "Gemma 4's turn and channel markup is added vocabulary here (cut out before BPE)",
        "<|turn>user\nhello<turn|>\n<|turn>model\n<|channel>thought\n<channel|><|think|>",
    ),
    EncodeCase(
        "gemma4-media-markup",
        "the image, audio and video markup — added tokens numbered past the vocabulary's end"
        " (ids 255999 to 258884)",
        "<|image><|image|><image|> and <|audio><|audio|><audio|> and <|video|>",
    ),
    EncodeCase(
        "gemma4-tool-markup",
        'the tool markup and the escape token `<|"|>` glued to text on both sides',
        '<|tool><tool|><|tool_call>call:search{q:<|"|>tokyo<|"|>}<tool_call|>'
        "<|tool_response><tool_response|>",
    ),
    EncodeCase(
        "specials-in-text",
        "`<bos>` / `<eos>` / `<pad>` / `<mask>` inside the text (the post-processor's `<bos>` …"
        " `<eos>` doubles the first two; decode skips all four)",
        "<bos>hello<eos><pad><mask>",
    ),
    EncodeCase(
        "gemma3-turn-markers",
        "`<start_of_turn>` is added vocabulary in the first EmbeddingGemma's tokenizer, not in this"
        " one — it falls to BPE here",
        "<start_of_turn>user\nhi<end_of_turn>",
    ),
    LONG_CASE,
)


def assert_prompts(config: Mapping[str, Any]) -> None:
    """The prompts of `config_sentence_transformers.json` use exactly {@link PROMPT_PREFIXES}."""
    prompts = config.get("prompts")
    if not isinstance(prompts, Mapping) or not prompts:
        raise TokenizerScriptError("config_sentence_transformers.json has no prompts")
    found = set(prompts.values())
    if found != set(PROMPT_PREFIXES):
        raise TokenizerScriptError(
            "the prompt prefixes of config_sentence_transformers.json are not the cases':"
            f" without a case {sorted(found - set(PROMPT_PREFIXES))},"
            f" no longer upstream {sorted(set(PROMPT_PREFIXES) - found)}"
        )


def assert_transformers_version(version: str) -> None:
    """The transformers that is imported is {@link REQUIRED_TRANSFORMERS} (the synced
    environment's own release would otherwise win silently)."""
    if version != REQUIRED_TRANSFORMERS:
        raise TokenizerScriptError(
            f"transformers {version} is imported, the script needs {REQUIRED_TRANSFORMERS}: run"
            f" `uv run --with 'transformers=={REQUIRED_TRANSFORMERS}' python -m"
            " embeddinggemma2.tokenizer`"
        )


def cross_check(directory: Path, cases: Sequence[EncodeCase]) -> None:
    """The transformers tokenizer of the snapshot and `tokenizers` give the same ids for every
    case (with and without special tokens), and the long case is long enough.

    The transformers tokenizer is loaded from a copy of the pinned files alone ({@link READ_FILES}
    but the prompts), so nothing unpinned in the snapshot directory takes part.
    """
    import transformers
    from tokenizers import Tokenizer

    assert_transformers_version(transformers.__version__)
    reference = Tokenizer.from_file(str(directory / "tokenizer.json"))
    with tempfile.TemporaryDirectory() as scratch:
        for name in ("tokenizer.json", "tokenizer_config.json", "config.json"):
            shutil.copyfile(directory / name, Path(scratch) / name)
        upstream = transformers.AutoTokenizer.from_pretrained(scratch)
    for case in cases:
        for add_special_tokens in (False, True):
            expected = reference.encode(case.text, add_special_tokens=add_special_tokens).ids
            actual = upstream(case.text, add_special_tokens=add_special_tokens)["input_ids"]
            if actual != expected:
                raise TokenizerScriptError(
                    f"case {case.name} (add_special_tokens={add_special_tokens}): transformers"
                    f" gives {actual}, tokenizers {expected}"
                )
    count = len(reference.encode(LONG_CASE.text, add_special_tokens=True).ids)
    if count < MIN_LONG_TOKENS:
        raise TokenizerScriptError(
            f"case {LONG_CASE.name} is {count} tokens, fewer than {MIN_LONG_TOKENS}"
        )


def main() -> None:
    directory = sources.SNAPSHOT_DIR
    sources.assert_pinned(directory, READ_FILES)
    assert_prompts(
        json.loads((directory / "config_sentence_transformers.json").read_text(encoding="utf-8"))
    )
    cross_check(directory, (*SHARED_ENCODE_CASES, *CASES))
    emit(
        tokenizer_json=directory / "tokenizer.json",
        asset_path=ASSET_PATH,
        fixture_path=FIXTURE_PATH,
        extra_cases=CASES,
        source_path=f"{sources.PIN.label}/tokenizer.json",
    )


if __name__ == "__main__":
    main()
