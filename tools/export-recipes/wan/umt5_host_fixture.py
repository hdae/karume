"""Write the fixture that checks Wan2.1 umT5's relative-position bucket tables against the TS
generator (`packages/models/src/umt5/relative-position.ts`) (ADR 0119 decision 3, step 10c).

    uv run --group wan --inexact python -m wan.umt5_host_fixture

Output (default `packages/models/tests/fixtures/wan-umt5/`):

    relative-position.safetensors  `bucket_by_distance [1023]` (the bucket of each distance =
                                   key position − query position = −511 to 511) and the tables
                                   of the representative valid lengths `table.lNNN [L, L]` (I32)
    relative-position.json         provenance, bucket configuration, valid-length range, graph
                                   input names, representative valid lengths and the digest of
                                   every valid length

The expected values come from the upstream formula (`UMT5Attention._relative_position_bucket`
called through {@link wan.umt5_patch} — the formula is not copied). The configuration (bucket
count, max_distance) is read from `text_encoder/config.json` of the pinned revision (the config
only — no weights are read; fetch it with
`uv run --group wan --inexact python -m wan.text_embeds --fetch`).

The digest pins every table for valid lengths 2 to 512: take the SHA-256 (32 B) of each table's
i32 little-endian row-major bytes, chain them in ascending valid-length order and take the SHA-256
(hex) of that. Chaining the tables themselves would be about 179 MB, and TS WebCrypto can only hash
bytes passed in one call, so each table is folded first.

The representative valid lengths ({@link REPRESENTATIVE_LENGTHS}) keep their whole tables (so a
digest mismatch can be pinned down to elements). They are lengths that cross bucket boundaries
(distances 8, 12, 16, 23, 32, 46, 64, 91 — research §4.3) and some of the lengths of the four fixed
prompts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file
from torch import nn

from _shared.paths import REPO_ROOT
from karume.convert import normalize_boundary_tensor
from wan import umt5_patch
from wan.sources import DEFAULT_MODEL, SOURCES, WAN21_MODELS, text_snapshot
from wan.umt5_tokenizer import MAX_LENGTH, MIN_TOKENS

DEFAULT_OUT = REPO_ROOT / "packages" / "models" / "tests" / "fixtures" / "wan-umt5"
TENSORS_FILE = "relative-position.safetensors"
META_FILE = "relative-position.json"

#: Valid lengths whose whole tables are kept: the lower bound 2; 8, up to distance 7 (all in the
#: linear range); 9, the first to enter the logarithmic range (distance 8); 17, crossing the
#: boundary 16; 28 / 50 / 126 of the fixed prompts; 92, the first to reach the top bucket 15
#: (distance 91).
REPRESENTATIVE_LENGTHS: tuple[int, ...] = (2, 8, 9, 17, 28, 50, 92, 126)

#: Tensor name of the per-distance buckets, and its first distance (= −(upper bound − 1)).
DISTANCE_KEY = "bucket_by_distance"
DISTANCE_OFFSET = -(MAX_LENGTH - 1)


def table_key(length: int) -> str:
    """Tensor name of a representative table (`table.l028` — zero-padded to sort by length)."""
    return f"table.l{length:03d}"


def table_bytes(table: torch.Tensor) -> bytes:
    """The table's i32 little-endian row-major bytes (the same order as a TS `Int32Array`)."""
    values = normalize_boundary_tensor(table, "bucket table").contiguous()
    return values.numpy().astype("<i4", copy=False).tobytes()


def tables_digest(attention: nn.Module) -> str:
    """Digest of every table for valid lengths 2 to 512 (in the form the module docstring gives)."""
    chained = hashlib.sha256()
    for length in range(MIN_TOKENS, MAX_LENGTH + 1):
        table = umt5_patch.relative_position_buckets(length, attention)
        chained.update(hashlib.sha256(table_bytes(table)).digest())
    return chained.hexdigest()


def bucket_by_distance(attention: nn.Module) -> torch.Tensor:
    """Buckets for distances −511 to 511 (the distances passed to the upstream formula as is)."""
    distances = torch.arange(DISTANCE_OFFSET, MAX_LENGTH, dtype=torch.long)
    return attention._relative_position_bucket(distances)


def check_consistent(attention: nn.Module, by_distance: torch.Tensor) -> None:
    """Check that the table at the upper bound is the Toeplitz expansion of the distance sequence
    (`table[i][j] = by_distance[j − i]`).

    MUST: checked before writing — if the distance sequence and the tables were written with
    opposite orientations, the TS side would match one and fail on the other, and the fixture could
    not tell which one is right.
    """
    table = umt5_patch.relative_position_buckets(MAX_LENGTH, attention)
    positions = torch.arange(MAX_LENGTH)
    expanded = by_distance[positions[None, :] - positions[:, None] - DISTANCE_OFFSET]
    if not torch.equal(table, expanded):
        raise AssertionError(
            "the upper-bound table differs from the Toeplitz expansion of the distance sequence"
            " (orientation or indices mixed up)"
        )


def build_fixture(
    config: Any, source: Mapping[str, str]
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Build the fixture's contents (tensors and meta) without writing files (the entry point
    pytest uses to check the fixture is up to date)."""
    import transformers

    attention = umt5_patch.bucket_attention(config)
    by_distance = bucket_by_distance(attention)
    check_consistent(attention, by_distance)
    tensors = {DISTANCE_KEY: normalize_boundary_tensor(by_distance, DISTANCE_KEY)}
    for length in REPRESENTATIVE_LENGTHS:
        table = umt5_patch.relative_position_buckets(length, attention)
        tensors[table_key(length)] = normalize_boundary_tensor(table, table_key(length))
    meta = {
        "_doc": [
            "Wan2.1 の umT5 の相対位置のバケット表"
            "（packages/models/src/umt5/relative-position.ts）の突き合わせ用 fixture。",
            "生成: tools/export-recipes/wan/umt5_host_fixture.py（期待値は上流の"
            " UMT5Attention._relative_position_bucket の出力 — 同ファイルの doc）。",
            "表の向きは上流の compute_bias と同じ table[i][j] = bucket(j − i)"
            "（キーの位置 − クエリの位置）。",
            "digest は有効長ごとの表（i32 LE・行優先）の SHA-256 を昇順に連ねたものの SHA-256。",
        ],
        "source": {
            **source,
            "config": "text_encoder/config.json",
            "transformers": transformers.__version__,
        },
        "num_buckets": int(config.relative_attention_num_buckets),
        "max_distance": int(config.relative_attention_max_distance),
        "bidirectional": not config.is_decoder,
        "min_length": MIN_TOKENS,
        "max_length": MAX_LENGTH,
        "input_names": list(umt5_patch.INPUT_NAMES),
        "distance_offset": DISTANCE_OFFSET,
        "representative_lengths": list(REPRESENTATIVE_LENGTHS),
        "digest": tables_digest(attention),
    }
    return tensors, meta


def pinned_config(model: str = DEFAULT_MODEL) -> Any:
    """`text_encoder/config.json` of the pinned revision (reads the config only)."""
    from transformers import UMT5Config

    return UMT5Config.from_pretrained(text_snapshot(model) / "text_encoder")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--model", default=DEFAULT_MODEL, choices=WAN21_MODELS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    source = SOURCES[args.model]
    tensors, meta = build_fixture(
        pinned_config(args.model), {"repo": source.repo, "revision": source.revision}
    )
    args.out.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(args.out / TENSORS_FILE))
    # Two-space indent (the JSON shape `deno fmt` writes — the fixture is tracked, so rewriting it
    # must not produce a diff).
    (args.out / META_FILE).write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    representatives = len(REPRESENTATIVE_LENGTHS)
    print(
        f"fixture OK: {args.out} (digest {meta['digest'][:16]}…, {representatives} representatives)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
