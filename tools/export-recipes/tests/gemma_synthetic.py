"""The **synthetic asset** of the Gemma tokenizers (the real 32 MB asset exists only locally, so
the tests run on this one).

It is a shared helper under `tests/` because the side that exercises the accepted set of compile
(`test_gemma_tokenizer`) and the side that exercises the chat fixtures (`gemma4/tests/test_chat`)
use **the same single copy** — with two spellings of the synthetic asset, one of them drifting
away from the upstream configuration would go unnoticed.

The synthetic asset is in a form `tokenizers.Tokenizer.from_str` accepts as is, so the same script
runs all the way through the path that captures the expected values (`build_fixture` /
`harvest_cases`).
"""

from __future__ import annotations

from typing import Any

from _shared.gemma_tokenizer import (
    EXPECTED_DECODER,
    EXPECTED_NORMALIZER,
    EXPECTED_PRE_TOKENIZER,
    byte_token,
)

#: The "ordinary" vocabulary of the synthetic asset (the special tokens placed before the 256
#: byte tokens).
SPECIALS = ["<pad>", "<eos>", "<bos>", "<unk>"]

#: The plain vocabulary BPE merges into (`▁` is the metaspace).
PIECES = ["a", "b", "c", "▁", "ab", "abc", "▁a", "▁ab", "\n"]


def vocab() -> dict[str, int]:
    """The vocabulary in id order (special tokens → the 256 byte tokens → the plain vocabulary)."""
    tokens = [*SPECIALS, *(byte_token(byte) for byte in range(256)), *PIECES]
    return {token: index for index, token in enumerate(tokens)}


def tokenizer_json(**overrides: Any) -> dict[str, Any]:
    """The smallest `tokenizer.json` configured like the real asset (`model.<field>` overrides the
    inner fields too)."""
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
