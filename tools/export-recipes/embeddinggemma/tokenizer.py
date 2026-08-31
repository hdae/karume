"""EmbeddingGemma-300m のトークナイザ資産の compile と parity フィクスチャ採取。

    uv run python -m embeddinggemma.tokenizer

gemma4 と**同じ台本系**で回る（ADR 0084 決定 6 の裁定 9 — 段 1a に同乗）。共用できるのは
`_shared/gemma_tokenizer.py` の compile と Deno 側の実装であって、**資産は共用できない**:
`.model.merges` は gemma4 と sha256 ビット同一（514,906 本）だが、`.model.vocab` は 262,144
本のうち 6,206 スロットで綴りが違い、`added_tokens` は 24 本 vs 6,415 本、post_processor は
gemma4 が素通しなのに対しこちらは `<bos>` … `<eos>` を付ける（ADR 0084 Context 1）。

出力は 2 系統:

  ① 実行時資産 `outputs/series/embeddinggemma-300m-tokenizer/tokenizer.json`
  ② パリティ用フィクスチャ
     `packages/models/tests/fixtures/gemma-text/embeddinggemma-parity.json`（**git 管理**）

NOTE: models pipeline / batch>1 export / attention_mask 配線は本段の射程外（backlog later
「EmbeddingGemma の完成」に残る）— ここで消化するのは tokenizer 部分だけ。
"""

from __future__ import annotations

from _shared.gemma_tokenizer import EncodeCase, emit
from _shared.paths import INPUTS_ROOT, REPO_ROOT, SERIES_ROOT

#: 上流の手置き資産（`google/embeddinggemma-300m` のスナップショット）。
TOKENIZER_JSON = INPUTS_ROOT / "embeddinggemma" / "google-300m" / "tokenizer.json"

#: 実行時資産（系列出力）。
ASSET_PATH = SERIES_ROOT / "embeddinggemma-300m-tokenizer" / "tokenizer.json"

#: パリティ用フィクスチャ（git 管理 — Deno 側 `gemma_tokenizer_test.ts` が読む）。
FIXTURE_PATH = (
    REPO_ROOT
    / "packages"
    / "models"
    / "tests"
    / "fixtures"
    / "gemma-text"
    / "embeddinggemma-parity.json"
)

#: EmbeddingGemma 固有の符号化ケース（共通の電池へ足す分）。
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
