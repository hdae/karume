"""Compiles a Gemma 4 model's tokenizer asset and takes its parity fixture (ADR 0084 decision 1).

Every model has a series of its own: `--model` picks E2B (the default), E4B, 12B or 26B. Until
Google's glitch-token fixes the `tokenizer.json` files were the same bytes; each fix drops another
set of BPE merges (E2B 267 and the vocabulary entry 249788 renamed to `<glitch_pruned_249788>`,
E4B 253, 12B 239 — same vocabulary otherwise), so some inputs tokenize differently on each model.
26B's is still the file from before the fixes.

    uv run python -m gemma4.tokenizer
    uv run python -m gemma4.tokenizer --model e4b
    uv run python -m gemma4.tokenizer --model 12b
    uv run python -m gemma4.tokenizer --model 26b

Two outputs (**always both in one run** — {@link _shared.gemma_tokenizer.emit}):

  ① the runtime asset `outputs/series/gemma4-e2b-tokenizer/tokenizer.json` (under `.gitignore`)
  ② the parity fixture
     `packages/models/tests/fixtures/gemma-text/gemma4-parity.json` (**tracked in git**)

E4B's, 12B's and 26B's series and fixtures carry the model's name (`gemma4-<model>-tokenizer` +
`gemma4-<model>-parity.json`); E2B's fixture keeps its name `gemma4-parity.json`. 12B's and 26B's
`tokenizer.json` are read from the pinned hub snapshots (`gemma4.sources`), E2B's and E4B's from the
hand-placed checkpoints. {@link EXTRA_CASES} adds per model encode cases where its tokenizer and
another model's split a text differently: E4B's run into merges E4B has and E2B's fix dropped,
12B's into merges E2B and E4B have and 12B's fix dropped, and E2B's into the vocabulary entry its
fix renamed.

How the asset is folded, the accepted set and how the subset is cut are defined by the docstring
of `_shared/gemma_tokenizer.py`. This file holds only what is specific to gemma4: where the inputs
live, where the outputs go, and encode cases that exercise **this checkpoint's chat markup**. Unlike
the Gemma 3 family, the chat markup is `<|turn>` / `<turn|>` / `<|channel>` / `<|think|>` (ADR 0084
Context 3), and `<start_of_turn>` is **not in this vocabulary**.

MUST: the chat template itself (`gemma.chatPrompt`) is out of scope for this step (ADR 0084
decision 5 = step 4). What is exercised here is only the encode-side property that special tokens
are split out as AddedVocabulary.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path

from _shared.gemma_tokenizer import EncodeCase, emit
from _shared.paths import REPO_ROOT, SERIES_ROOT
from gemma4 import sources

#: The models with a tokenizer series of their own (`gemma4.distribution.GEMMA4_TOKENIZER_SERIES`).
MODELS: tuple[str, ...] = ("e2b", "e4b", "12b", "26b")

#: The hand-placed upstream asset (`inputs/<family>/<name>/` — assets-layout).
TOKENIZER_JSON = sources.tokenizer_dir("e2b") / "tokenizer.json"

#: The runtime asset (series output).
ASSET_PATH = SERIES_ROOT / "gemma4-e2b-tokenizer" / "tokenizer.json"

#: The parity fixture (tracked in git — read by `gemma4_tokenizer_test.ts` on the Deno side).
FIXTURE_PATH = (
    REPO_ROOT / "packages" / "models" / "tests" / "fixtures" / "gemma-text" / "gemma4-parity.json"
)

#: Encode cases specific to gemma4 (added to the shared battery).
CASES: tuple[EncodeCase, ...] = (
    EncodeCase(
        "chat-turn",
        "this checkpoint's chat spellings (`<|turn>` / `<turn|>`) are cut out as added vocabulary",
        "<|turn>user\nこんにちは<turn|>\n<|turn>model\n",
    ),
    EncodeCase(
        "special-adjacent",
        "adjacent special tokens — the shape where the leftmost-longest cut runs straight through",
        "<|turn><turn|><|channel><|think|>",
    ),
    EncodeCase(
        "special-glued",
        "special tokens glued straight to the text (no space on either side = the normalization"
        " boundary moves)",
        "abc<|turn>def",
    ),
    EncodeCase(
        "bos-in-text",
        "`<bos>` inside the text (encode does not add it — only the chat function does; ADR 0084"
        " decision 5)",
        "<bos>hello",
    ),
    EncodeCase(
        "special-lookalike",
        "spellings one character short of a special token (they miss the added vocabulary and fall"
        " to BPE)",
        "<|turn <|turnx> <turn>",
    ),
)

#: Model → encode cases of its own on top of {@link CASES}. 12B's run into merges E2B and E4B have
#: and 12B dropped: E2B joins these pieces, 12B keeps them apart. E4B's run into merges E4B has and
#: E2B dropped: E4B joins them, E2B keeps them apart (the pytest checks that both tokenizers really
#: split each text differently). E2B's case is the vocabulary entry its fix renamed. 26B's
#: `tokenizer.json` is the file from before the fixes (`gemma4.sources.HUB_PINS`): its cases are
#: 12B's texts, which it joins and 12B keeps apart.
EXTRA_CASES: Mapping[str, tuple[EncodeCase, ...]] = {
    "e2b": (
        EncodeCase(
            "glitch-pruned",
            "the character of id 249788, renamed `<glitch_pruned_249788>` by E2B's fix: byte"
            " fallback now (E4B and 12B still map it to 249788)",
            "\U0002127a",
        ),
    ),
    "e4b": (
        EncodeCase(
            "e2b-dropped-merge",
            "a merge E4B has and E2B dropped (`Module` + `InitFlag`)",
            "ModuleInitFlag",
        ),
        EncodeCase(
            "e2b-dropped-merge-metaspace",
            "a dropped merge after a metaspace (`▁browsing` + `Stamp`)",
            " browsingStamp",
        ),
        EncodeCase(
            "e2b-dropped-merge-twice",
            "both merges that build the piece are dropped (`opencamera` + `studio`,"
            " `opencamer` + `astudio`)",
            "opencamerastudio",
        ),
    ),
    "12b": (
        EncodeCase(
            "dropped-merge",
            "a merge E2B and E4B have and 12B dropped (`Malay` + `Marks`)",
            "MalayMarks",
        ),
        EncodeCase(
            "dropped-merge-metaspace",
            "a dropped merge after a metaspace (`▁jinfo` + `Home`)",
            " jinfoHome",
        ),
        EncodeCase(
            "dropped-merge-twice",
            "both merges that build the piece are dropped (`HDR` + `Offsets`, `HDROff` + `sets`)",
            "HDROffsets",
        ),
    ),
    "26b": (
        EncodeCase(
            "12b-dropped-merge",
            "a merge 12B's fix dropped and 26B's tokenizer keeps (`Malay` + `Marks`)",
            "MalayMarks",
        ),
        EncodeCase(
            "12b-dropped-merge-metaspace",
            "a merge 12B dropped, after a metaspace (`▁jinfo` + `Home`)",
            " jinfoHome",
        ),
        EncodeCase(
            "12b-dropped-merge-twice",
            "both merges 12B dropped (`HDR` + `Offsets`, `HDROff` + `sets`)",
            "HDROffsets",
        ),
    ),
}


def asset_path(model: str) -> Path:
    """The model's runtime asset (`outputs/series/gemma4-<model>-tokenizer/tokenizer.json`)."""
    return SERIES_ROOT / f"gemma4-{model}-tokenizer" / ASSET_PATH.name


def fixture_path(model: str) -> Path:
    """The model's parity fixture (E2B's keeps its name; the others' are
    `gemma4-<model>-parity.json`)."""
    return FIXTURE_PATH if model == "e2b" else FIXTURE_PATH.with_name(f"gemma4-{model}-parity.json")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Compile a Gemma 4 tokenizer asset")
    parser.add_argument("--model", choices=MODELS, default="e2b")
    model = parser.parse_args(argv).model
    # The tokenizer source, not the weights' (12B's QAT weights ship the `tokenizer.json` from
    # before Google's glitch-token fixes — `gemma4.sources`).
    directory = sources.tokenizer_dir(model)
    pins = sources.HUB_PINS.get(model)
    if pins is not None:
        sources.assert_snapshot_pin(directory, pins.tokenizer)
    emit(
        tokenizer_json=directory / TOKENIZER_JSON.name,
        asset_path=asset_path(model),
        fixture_path=fixture_path(model),
        extra_cases=(*CASES, *EXTRA_CASES[model]),
        source_path=None if pins is None else f"{sources.tokenizer_label(model)}/tokenizer.json",
    )


if __name__ == "__main__":
    main()
