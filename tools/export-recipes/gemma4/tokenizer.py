"""Compiles a Gemma 4 model's tokenizer asset and takes its parity fixture (ADR 0084 decision 1).

Every model has a series of its own: `--model` picks E2B (the default), E4B or 12B. Until Google's
glitch-token fixes the three `tokenizer.json` files were the same bytes; each fix drops another set
of BPE merges (E2B 267 and the vocabulary entry 249788 renamed to `<glitch_pruned_249788>`, E4B
253, 12B 239 — same vocabulary otherwise), so some inputs tokenize differently on each model.

    uv run python -m gemma4.tokenizer
    uv run python -m gemma4.tokenizer --model e4b
    uv run python -m gemma4.tokenizer --model 12b

出力は 2 系統（**1 回の実行で必ず両方** — {@link _shared.gemma_tokenizer.emit}）:

  ① 実行時資産 `outputs/series/gemma4-e2b-tokenizer/tokenizer.json`（`.gitignore` 配下）
  ② パリティ用フィクスチャ
     `packages/models/tests/fixtures/gemma-text/gemma4-parity.json`（**git 管理**）

E4B's and 12B's series and fixtures carry the model's name (`gemma4-<model>-tokenizer` +
`gemma4-<model>-parity.json`); E2B's fixture keeps its name `gemma4-parity.json`. 12B's
`tokenizer.json` is read from the pinned hub snapshot (`gemma4.sources`), E2B's and E4B's from the
hand-placed checkpoints. {@link EXTRA_CASES} adds per model encode cases where its tokenizer and
another model's split a text differently: E4B's run into merges E4B has and E2B's fix dropped,
12B's into merges E2B and E4B have and 12B's fix dropped, and E2B's into the vocabulary entry its
fix renamed.

畳み方・受理集合・部分集合の切り方は `_shared/gemma_tokenizer.py` の docstring が正本。
ここが持つのは gemma4 固有の綴りだけ: 入力の置き場・出力先・**この checkpoint の chat 綴り**
を踏む符号化ケース。chat の綴りは Gemma 3 系と違い `<|turn>` / `<turn|>` / `<|channel>` /
`<|think|>` で（ADR 0084 Context 3）、`<start_of_turn>` は**この語彙に無い**。

MUST: chat template そのもの（`gemma.chatPrompt`）は本段の射程外（ADR 0084 決定 5 = 段 4）。
ここで踏むのは「特殊トークンが AddedVocabulary として切り出される」という符号化側の性質だけ。
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path

from _shared.gemma_tokenizer import EncodeCase, emit
from _shared.paths import REPO_ROOT, SERIES_ROOT
from gemma4 import sources

#: The models with a tokenizer series of their own (`gemma4.distribution.GEMMA4_TOKENIZER_SERIES`).
MODELS: tuple[str, ...] = ("e2b", "e4b", "12b")

#: 上流の手置き資産（`inputs/<family>/<name>/` — assets-layout）。
TOKENIZER_JSON = sources.tokenizer_dir("e2b") / "tokenizer.json"

#: 実行時資産（系列出力）。
ASSET_PATH = SERIES_ROOT / "gemma4-e2b-tokenizer" / "tokenizer.json"

#: パリティ用フィクスチャ（git 管理 — Deno 側 `gemma4_tokenizer_test.ts` が読む）。
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

#: Model → encode cases of its own on top of {@link CASES}. 12B's run into merges E2B and E4B have
#: and 12B dropped: E2B joins these pieces, 12B keeps them apart. E4B's run into merges E4B has and
#: E2B dropped: E4B joins them, E2B keeps them apart (the pytest checks that both tokenizers really
#: split each text differently). E2B's case is the vocabulary entry its fix renamed.
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
