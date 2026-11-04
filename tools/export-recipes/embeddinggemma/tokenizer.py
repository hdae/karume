"""Compile of the EmbeddingGemma-300m tokenizer asset and harvest of the parity fixture.

    uv run python -m embeddinggemma.tokenizer

Runs on **the same script family** as gemma4 (ADR 0084 decision 6, ruling 9 — it rode along with
step 1a). What can be shared is the compile in `_shared/gemma_tokenizer.py` and the Deno-side
implementation; **the asset cannot be shared**: `.model.merges` is sha256 bit-identical to gemma4's
(514,906 entries), but `.model.vocab` differs in spelling in 6,206 of its 262,144 slots,
`added_tokens` is 24 vs 6,415 entries, and the post_processor adds `<bos>` … `<eos>` here while
gemma4's passes through (ADR 0084 Context 1).

Two outputs:

  ① runtime asset `outputs/series/embeddinggemma-300m-tokenizer/tokenizer.json`
  ② parity fixture
     `packages/models/tests/fixtures/gemma-text/embeddinggemma-parity.json` (**tracked in git**)

NOTE: the models pipeline / batch>1 export / attention_mask wiring are out of scope for this step
(they remain in the backlog's later item "complete EmbeddingGemma") — only the tokenizer part is
done here.
"""

from __future__ import annotations

from _shared.gemma_tokenizer import EncodeCase, emit
from _shared.paths import INPUTS_ROOT, REPO_ROOT, SERIES_ROOT

#: The hand-placed upstream asset (a snapshot of `google/embeddinggemma-300m`).
TOKENIZER_JSON = INPUTS_ROOT / "embeddinggemma" / "google-300m" / "tokenizer.json"

#: The runtime asset (series output).
ASSET_PATH = SERIES_ROOT / "embeddinggemma-300m-tokenizer" / "tokenizer.json"

#: The parity fixture (tracked in git — read by the Deno-side `gemma4_tokenizer_test.ts`).
FIXTURE_PATH = (
    REPO_ROOT
    / "packages"
    / "models"
    / "tests"
    / "fixtures"
    / "gemma-text"
    / "embeddinggemma-parity.json"
)

#: The encode cases specific to EmbeddingGemma (added to the shared battery).
CASES: tuple[EncodeCase, ...] = (
    EncodeCase(
        "prompt-query",
        "the query prefix upstream recommends (prompts.query of"
        " `config_sentence_transformers.json`)",
        "task: search result | query: What is the capital of Japan?",
    ),
    EncodeCase(
        "prompt-document",
        "likewise the document prefix — the real-world shape where post_processor adds bos…eos",
        "title: none | text: 東京は日本の首都であり、世界有数の大都市圏を形成している。",
    ),
    EncodeCase(
        "unused-added",
        "**non-special** added vocabulary (the 6,406 `<unusedN>` entries) — not skipped by decode",
        "<unused0>x<unused6241>",
    ),
    EncodeCase(
        "gemma3-turn-markers",
        "`<start_of_turn>` exists only in this vocabulary (gemma4's has none — proof that the"
        " assets differ)",
        "<start_of_turn>user\nhi<end_of_turn>",
    ),
    EncodeCase(
        "image-soft-token",
        "added vocabulary numbered **outside** the vocabulary table (id 262144 = the row after the"
        " 262,144 vocabulary rows)",
        "<start_of_image><image_soft_token><end_of_image>",
    ),
    EncodeCase(
        "bos-in-text",
        "`<bos>` inside the text (it overlaps with the one post_processor adds and is doubled)",
        "<bos>hello",
    ),
)


def main() -> None:
    emit(
        tokenizer_json=TOKENIZER_JSON,
        asset_path=ASSET_PATH,
        fixture_path=FIXTURE_PATH,
        extra_cases=CASES,
    )


if __name__ == "__main__":
    main()
