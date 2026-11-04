"""Fix the evaluation set and the input tokens before running the model."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from transformers import AutoTokenizer


def profiles(root: Path) -> dict:
    """Evaluable models → the official checkpoint, the **distribution** (`karume.json` of
    `karume/5`) and the **quant** whose stored `model` weights `--weights stored` measures.

    MUST: the quant is named, never taken from the manifest's `defaultQuant` — a distribution's
    default can move to another storage (the regular Gemma 4 line's became the original-weights
    `bf16` preset on 2026-11-04, whose PLE index this tool does not read), and the measurement would
    then silently be of other weights. The int4 presets of a line (`i4` / `i4-gemvpar` /
    `i4-fast`) load the same container, so `i4` names it.

    NOTE: since 2026-11-04 the regular line (`models/karume-gemma4`) ships `bf16` alone, so the
    `gemma4-e2b` profile's `i4` is not in its manifest and `--weights stored` fails loudly in
    `container_paths`; `--weights source` still scores the original checkpoint. Scoring the `bf16`
    preset (PLE index schema 5) or the q4_0 QAT line (schema 4) needs a PLE reader for those
    schemas first (backlog).

    Only containers (`krm`) are read. The `-probe` series still kept in the old shard form
    (the 2026-09 experiments with gemma4 E4B / MiniCPM5-2B / Qwen3-0.6B) **have no entry in this
    table** — dual reading is not implemented (container-v1 §12). The measurements of that time
    remain as point-in-time snapshots in `docs/research/`; to measure again, rebuild the series as
    containers first and then add them to the table.
    """
    return {
        "gemma4-e2b": {
            "family": "gemma4",
            "checkpoint": root / "inputs/gemma4/gemma-4-E2B-it",
            "distribution": root / "models/karume-gemma4",
            "model": "e2b",
            "quant": "i4",
        },
        "gemma4-qat-e2b": {
            "family": "gemma4-qat",
            "checkpoint": root / "inputs/gemma4-qat/gemma-4-E2B-it-qat-mobile-transformers",
            "distribution": root / "models/karume-gemma4-qat-mobile",
            "model": "e2b",
            "quant": "i4",
        },
        "gemma4-qat-e4b": {
            "family": "gemma4-qat",
            "checkpoint": root / "inputs/gemma4-qat/gemma-4-E4B-it-qat-mobile-transformers",
            "distribution": root / "models/karume-gemma4-qat-mobile",
            "model": "e4b",
            "quant": "i4",
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
        raise ValueError(f"{row['id']}: cannot fix the token boundary of the continuation")
    return {
        "id": row["id"],
        "choices": choices,
        "scoreStart": len(context),
        "answer": row["choices"]["label"].index(row["answerKey"]),
    }


def prepare(root: Path, data: Path, destination: Path, count: int = 64, chars: int = 8192) -> None:
    if count <= 0 or chars <= 0:
        raise ValueError("count and chars must be positive")
    destination.mkdir()
    source = json.loads((data / "sources.json").read_text())
    for row in source:
        raw = (data / f"{row['name']}.parquet").read_bytes()
        if hashlib.sha256(raw).hexdigest() != row["sha256"]:
            raise ValueError("data fingerprint does not match")
        normalized = (data / f"{row['name']}.json").read_bytes()
        if hashlib.sha256(normalized).hexdigest() != row["jsonSha256"]:
            raise ValueError("JSON fingerprint does not match")
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
        raise ValueError("the requested number of questions is not available")
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
