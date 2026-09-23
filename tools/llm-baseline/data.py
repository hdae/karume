"""モデルを走らせる前に評価集合と入力トークンを固定する。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from transformers import AutoTokenizer


def profiles(root: Path) -> dict:
    """評価できるモデル → 公式 checkpoint と**配布形**（`karume/5` の `karume.json`）。

    読むのは容器（`krm`）だけである。旧 shard 形のまま残している `-probe` 系列
    （gemma4 E4B / MiniCPM5-2B / Qwen3-0.6B の 2026-09 の実験）は**この表に席を持たない** —
    両読みは実装しない（container-v1 §12）。当時の測定そのものは時点スナップショットとして
    `docs/research/` に残っており、再計測するなら系列を容器へ焼き直してから表に足す。
    """
    return {
        "gemma4-e2b": {
            "family": "gemma4",
            "checkpoint": root / "inputs/gemma4/gemma-4-E2B-it",
            "distribution": root / "models/karume-gemma4",
            "model": "e2b",
        },
        "gemma4-qat-e2b": {
            "family": "gemma4-qat",
            "checkpoint": root / "inputs/gemma4-qat/gemma-4-E2B-it-qat-mobile-transformers",
            "distribution": root / "models/karume-gemma4-qat",
            "model": "e2b",
        },
        "gemma4-qat-e4b": {
            "family": "gemma4-qat",
            "checkpoint": root / "inputs/gemma4-qat/gemma-4-E4B-it-qat-mobile-transformers",
            "distribution": root / "models/karume-gemma4-qat",
            "model": "e4b",
        },
    }


def encode(tokenizer, text: str) -> list[int]:
    ids = tokenizer.encode(text, add_special_tokens=False)
    return ([tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []) + ids


def arc_input(tokenizer, row: dict) -> dict:
    prefix = "Question: " + row["question"] + "\nAnswer:"
    context = encode(tokenizer, prefix)
    choices = [encode(tokenizer, prefix + " " + answer) for answer in row["choices"]["text"]]
    if any(ids[: len(context)] != context or len(ids) == len(context) for ids in choices):
        raise ValueError(f"{row['id']}: 継続のtoken境界を固定できません")
    return {
        "id": row["id"],
        "choices": choices,
        "scoreStart": len(context),
        "answer": row["choices"]["label"].index(row["answerKey"]),
    }


def prepare(root: Path, data: Path, destination: Path, count: int = 64, chars: int = 8192) -> None:
    if count <= 0 or chars <= 0:
        raise ValueError("count と chars は正である必要があります")
    destination.mkdir()
    source = json.loads((data / "sources.json").read_text())
    for row in source:
        raw = (data / f"{row['name']}.parquet").read_bytes()
        if hashlib.sha256(raw).hexdigest() != row["sha256"]:
            raise ValueError("データの指紋が一致しません")
        normalized = (data / f"{row['name']}.json").read_bytes()
        if hashlib.sha256(normalized).hexdigest() != row["jsonSha256"]:
            raise ValueError("JSON の指紋が一致しません")
    tokenizers = {
        name: AutoTokenizer.from_pretrained(profile["checkpoint"], local_files_only=True)
        for name, profile in profiles(root).items()
    }
    rows = json.loads((data / "arc.json").read_text())
    selected = {name: [] for name in tokenizers}
    excluded = []
    for row in rows:
        encoded = {name: arc_input(tokenizer, row) for name, tokenizer in tokenizers.items()}
        if any(len(ids) > 128 for value in encoded.values() for ids in value["choices"]):
            excluded.append({"id": row["id"], "reason": "over-128-tokens"})
            continue
        for name, value in encoded.items():
            selected[name].append(value)
        if len(next(iter(selected.values()))) == count:
            break
    if len(next(iter(selected.values()))) != count:
        raise ValueError("指定数の問題がありません")
    wiki_rows = json.loads((data / "wiki.json").read_text())
    text = "\n\n".join(row["text"] for row in wiki_rows)[:chars]
    result = {
        "format": "karume-llm-quality/1",
        "sources": source,
        "arc": {
            "split": "test",
            "subset": "ARC-Easy",
            "count": count,
            "selection": "first questions fitting every tokenizer within 128 tokens",
            "excluded": excluded,
            "template": "Question: {question}\nAnswer: {choice}",
            "chatTemplate": False,
        },
        "wiki": {
            "chars": len(text),
            "textSha256": hashlib.sha256(text.encode()).hexdigest(),
            "window": 128,
            "stride": 64,
            "text": text,
        },
        "models": {},
    }
    for name, tokenizer in tokenizers.items():
        checkpoint = profiles(root)[name]["checkpoint"]
        result["models"][name] = {
            "arc": selected[name],
            "wikiIds": encode(tokenizer, text),
            "bosTokenId": tokenizer.bos_token_id,
            "tokenizerSha256": hashlib.sha256(
                (checkpoint / "tokenizer.json").read_bytes()
            ).hexdigest(),
        }
    with (destination / "suite.json").open("x") as output:
        json.dump(result, output, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--count", type=int, default=64)
    parser.add_argument("--chars", type=int, default=8192)
    args = parser.parse_args()
    prepare(args.root, args.data, args.out, args.count, args.chars)
