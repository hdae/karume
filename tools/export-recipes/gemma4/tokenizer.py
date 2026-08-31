"""gemma4 E2B のトークナイザ資産の compile と parity フィクスチャ採取（ADR 0084 決定 1）。

    uv run python -m gemma4.tokenizer

出力は 2 系統（**1 回の実行で必ず両方** — {@link _shared.gemma_tokenizer.emit}）:

  ① 実行時資産 `outputs/series/gemma4-e2b-tokenizer/tokenizer.json`（`.gitignore` 配下）
  ② パリティ用フィクスチャ
     `packages/models/tests/fixtures/gemma-text/gemma4-parity.json`（**git 管理**）

畳み方・受理集合・部分集合の切り方は `_shared/gemma_tokenizer.py` の docstring が正本。
ここが持つのは gemma4 固有の綴りだけ: 入力の置き場・出力先・**この checkpoint の chat 綴り**
を踏む符号化ケース。chat の綴りは Gemma 3 系と違い `<|turn>` / `<turn|>` / `<|channel>` /
`<|think|>` で（ADR 0084 Context 3）、`<start_of_turn>` は**この語彙に無い**。

MUST: chat template そのもの（`gemma4ChatPrompt`）は本段の射程外（ADR 0084 決定 5 = 段 4）。
ここで踏むのは「特殊トークンが AddedVocabulary として切り出される」という符号化側の性質だけ。
"""

from __future__ import annotations

from _shared.gemma_tokenizer import EncodeCase, emit
from _shared.paths import INPUTS_ROOT, REPO_ROOT, SERIES_ROOT

#: 上流の手置き資産（`inputs/<family>/<name>/` — assets-layout）。
TOKENIZER_JSON = INPUTS_ROOT / "gemma4" / "gemma-4-E2B-it" / "tokenizer.json"

#: 実行時資産（系列出力）。
ASSET_PATH = SERIES_ROOT / "gemma4-e2b-tokenizer" / "tokenizer.json"

#: パリティ用フィクスチャ（git 管理 — Deno 側 `gemma_tokenizer_test.ts` が読む）。
FIXTURE_PATH = (
    REPO_ROOT / "packages" / "models" / "tests" / "fixtures" / "gemma-text" / "gemma4-parity.json"
)

#: gemma4 固有の符号化ケース（共通の電池へ足す分）。
CASES: tuple[EncodeCase, ...] = (
    EncodeCase(
        "chat-turn",
        "この checkpoint の chat 綴り（`<|turn>` / `<turn|>`）が追加語彙として切り出される",
        "<|turn>user\nこんにちは<turn|>\n<|turn>model\n",
    ),
    EncodeCase(
        "special-adjacent",
        "特殊トークンの隣接 — leftmost-longest の切り出しが素通しで進む形",
        "<|turn><turn|><|channel><|think|>",
    ),
    EncodeCase(
        "special-glued",
        "特殊トークンが地の文へ直接くっつく（前後に空白が無い = 正規化の境目が動く）",
        "abc<|turn>def",
    ),
    EncodeCase(
        "bos-in-text",
        "`<bos>` が本文中にある（encode は付けない — 付けるのは chat 関数だけ・ADR 0084 決定 5）",
        "<bos>hello",
    ),
    EncodeCase(
        "special-lookalike",
        "特殊トークンに 1 文字だけ足りない綴り（追加語彙に当たらず BPE へ落ちる）",
        "<|turn <|turnx> <turn>",
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
