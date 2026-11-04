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
        "上流が推奨する query 前置き（`config_sentence_transformers.json` の prompts.query）",
        "task: search result | query: What is the capital of Japan?",
    ),
    EncodeCase(
        "prompt-document",
        "同じく document 前置き — post_processor が bos…eos を付ける形の実需",
        "title: none | text: 東京は日本の首都であり、世界有数の大都市圏を形成している。",
    ),
    EncodeCase(
        "unused-added",
        "**非 special** の追加語彙（6,406 本ある `<unusedN>` 系）— decode の skip 対象外",
        "<unused0>x<unused6241>",
    ),
    EncodeCase(
        "gemma3-turn-markers",
        "`<start_of_turn>` はこちらの語彙にだけ在る（gemma4 側には無い — 資産が別物である証）",
        "<start_of_turn>user\nhi<end_of_turn>",
    ),
    EncodeCase(
        "image-soft-token",
        "語彙表の**外**へ採番された追加語彙（id 262144 = 語彙 262,144 行の次）",
        "<start_of_image><image_soft_token><end_of_image>",
    ),
    EncodeCase(
        "bos-in-text",
        "`<bos>` が本文中にある（post_processor が付ける分と重なって二重になる形）",
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
