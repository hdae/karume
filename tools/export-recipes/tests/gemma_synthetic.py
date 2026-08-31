"""Gemma 系トークナイザの**合成資産**（実資産 32MB はローカルにしか無いのでテストはこれで回す）。

置き場が `tests/` の共有ヘルパなのは、compile の受理集合を叩く側（`test_gemma_tokenizer`）と
chat フィクスチャを叩く側（`gemma4/tests/test_chat`）が**同じ 1 本**を使うため — 合成資産の
綴りを 2 つ持つと、片方だけが上流の構成から外れても気づけない。

合成資産は `tokenizers.Tokenizer.from_str` にそのまま食わせられる形なので、期待値の採取経路
（`build_fixture` / `harvest_cases`）まで同じ台本で通る。
"""

from __future__ import annotations

from typing import Any

from _shared.gemma_tokenizer import (
    EXPECTED_DECODER,
    EXPECTED_NORMALIZER,
    EXPECTED_PRE_TOKENIZER,
    byte_token,
)

#: 合成資産の「普通の」語彙（byte 語彙 256 本の手前に置く特殊トークン）。
SPECIALS = ["<pad>", "<eos>", "<bos>", "<unk>"]

#: BPE で畳まれる素の語彙（`▁` は metaspace）。
PIECES = ["a", "b", "c", "▁", "ab", "abc", "▁a", "▁ab", "\n"]


def vocab() -> dict[str, int]:
    """id 順の語彙（特殊トークン → byte 語彙 256 本 → 素の語彙）。"""
    tokens = [*SPECIALS, *(byte_token(byte) for byte in range(256)), *PIECES]
    return {token: index for index, token in enumerate(tokens)}


def tokenizer_json(**overrides: Any) -> dict[str, Any]:
    """実資産と同じ構成の最小 `tokenizer.json`（`model.<欄>` で内側も差し替えられる）。"""
    raw: dict[str, Any] = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [
            {
                "id": index,
                "content": token,
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True,
            }
            for index, token in enumerate(SPECIALS)
        ],
        "normalizer": EXPECTED_NORMALIZER,
        "pre_tokenizer": EXPECTED_PRE_TOKENIZER,
        "post_processor": {
            "type": "TemplateProcessing",
            "single": [{"Sequence": {"id": "A", "type_id": 0}}],
            "pair": [{"Sequence": {"id": "A", "type_id": 0}}],
            "special_tokens": {},
        },
        "decoder": EXPECTED_DECODER,
        "model": {
            "type": "BPE",
            "dropout": None,
            "unk_token": "<unk>",
            "continuing_subword_prefix": None,
            "end_of_word_suffix": None,
            "fuse_unk": True,
            "byte_fallback": True,
            "ignore_merges": False,
            "vocab": vocab(),
            "merges": [["a", "b"], ["ab", "c"], ["▁", "a"], ["▁a", "b"]],
        },
    }
    for key, value in overrides.items():
        if key.startswith("model."):
            raw["model"][key.removeprefix("model.")] = value
        else:
            raw[key] = value
    return raw
