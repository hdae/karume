"""The Wan2.1 text embedding asset (stage 1 = precomputed embeddings — ADR 0118 decision 4, step 6).

Writes the output `[L_valid, 4096]` of the fixed prompts (3 positive + 1 negative) passed through
umT5-XXL to safetensors in f32. The host builds the DiT input `[1, 512, 4096]` (zero-padded after
the valid length) — the asset holds only the valid length (`L_valid` = the length where the
tokenizer's mask is 1).

## How it is made (the upstream path is called as is)

- The embeddings are made by calling diffusers 0.39.0's `WanPipeline._get_t5_prompt_embeds` as is
  (`prompt_clean` → the tokenizer with `padding="max_length"`, `max_length=512` →
  `UMT5EncoderModel` → cut at the valid length and zero-padded to 512). This module only cuts the
  returned `[1, 512, 4096]` at the valid length and copies no formula.
- umT5 is loaded **in the upstream's dtype (bf16)** (the official `t5_dtype` = bfloat16; the
  Diffusers documentation's example also uses `torch_dtype=torch.bfloat16`). Converting the output
  from bf16 to f32 does not change the values (bf16 is the upper 16 bits of f32).
- **A separate process, on the CPU.** The DiT / VAE are not loaded (umT5 and the DiT do not share
  the 31 GiB of host RAM — decision 4). The weights, fp32 safetensors in 5 shards (about 22.7 GB),
  are loaded as bf16 (about 11.4 GB). transformers 5's `from_pretrained` always loads in the
  memory-saving way, and `low_cpu_mem_usage` is ignored (it is not passed).
- Normalization is `prompt_clean` (`ftfy.fix_text` only when ftfy is installed —
  `pipeline_wan.py:78-81`). ftfy is pinned to `==6.3.1` in the wan dependency group (the "pinned" of
  decision 4). Both the original text before normalization and the string after it go into the
  metadata.

## The asset format (`text_embeds.safetensors`)

- One tensor per prompt (name = {@link wan.prompts.FixedPrompt.name}, `F32`, `[L_valid, 4096]`).
- The metadata puts JSON (sorted keys, no whitespace in the separators) under **a single key**
  ({@link METADATA_KEY}). MUST: do not use several keys — safetensors (Rust) writes the metadata
  from a HashMap, and the key order changes per process (measured 2026-10-02: three writes of the
  same input gave two different sha256 values). With a single key, the same input gives the same
  bytes.
- The location is the series directory `outputs/series/wan2.1-t2v-1.3b-text-embeds/` (a compile
  output without a graph — docs/assets-layout.md). In the distribution it is a model-level `assets`
  entry (independent of the quant — decision 4).

## Fixed prompts (maintainer's ruling 2026-10-02: examples of the official README / Diffusers docs)

The table's source of truth is {@link wan.prompts.FIXED_PROMPTS} (the card and the assembly gate
read the same table).

- `boxing-cats` (positive): the t2v-1.3B example of the official README (the same text as
  `EXAMPLE_PROMPT` in `generate.py`).
- `ferret` (positive): the Wan T2V example of the Diffusers documentation (kept as a multi-line
  string).
- `cat-dog-baking` (positive): the example of Diffusers' `WanPipeline.__call__` (the docstring that
  appears in the API reference).
- `negative` (negative): `sample_neg_prompt` of the official `wan/configs/shared_config.py`
  (Chinese).

Amount of motion: fighting (large), the camera's charge and leap (large, camera movement), kitchen
work (small).

    uv run --group wan --inexact python -m wan.text_embeds --fetch   # fetch umT5 and the tokenizer
    uv run --group wan --inexact python -m wan.text_embeds           # write the asset

MUST: diffusers / transformers / ftfy are imported inside functions (the `wan` group is not part of
the default sync — `tests/test_optional_group_imports.py`).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from _shared.paths import SERIES_ROOT
from wan.pipeline_ref import MAX_SEQUENCE_LENGTH, TEXT_DIM
from wan.prompts import FIXED_PROMPTS, FixedPrompt
from wan.sources import DEFAULT_MODEL, SOURCES, WAN21_MODELS, text_snapshot

#: The series directory (decision 4, decision 7).
SERIES_NAME = "wan2.1-t2v-1.3b-text-embeds"

#: The asset's file name.
ASSET_NAME = "text_embeds.safetensors"

#: The only metadata key (its value is JSON — the MUST of the module doc).
METADATA_KEY = "karume.wan.text_embeds"

#: The umT5 dtype (the same as the upstream's — the official `t5_dtype`).
TEXT_ENCODER_DTYPE = torch.bfloat16


class TextEmbedsError(ValueError):
    """The embedding asset's shape or metadata departed from the agreement of decision 4."""


def prompt_by_name(name: str) -> FixedPrompt:
    """Look up a fixed prompt by name (fail loudly when absent)."""
    for prompt in FIXED_PROMPTS:
        if prompt.name == name:
            return prompt
    raise TextEmbedsError(f"no fixed prompt {name!r} ({[p.name for p in FIXED_PROMPTS]})")


def normalize(text: str) -> str:
    """The upstream `prompt_clean` (`fix_text` when ftfy is installed → html unescape → collapsing
    whitespace).

    MUST: stop in an environment where ftfy cannot be imported — without ftfy the upstream silently
    skips `fix_text`, and the same prompt becomes a different string (a different embedding)
    (decision 4).
    """
    import ftfy  # noqa: F401 — presence check (the upstream branches on `is_ftfy_available()`)
    from diffusers.pipelines.wan.pipeline_wan import prompt_clean
    from diffusers.utils import is_ftfy_available

    if not is_ftfy_available():
        raise TextEmbedsError(
            "diffusers does not find ftfy — normalization would differ from upstream"
        )
    return prompt_clean(text)


def load_text_encoder(model: str = DEFAULT_MODEL) -> tuple[Any, Any]:
    """Load the tokenizer and umT5 (bf16, CPU, eval) of the pinned revision."""
    from transformers import AutoTokenizer, UMT5EncoderModel

    snapshot = text_snapshot(model)
    tokenizer = AutoTokenizer.from_pretrained(snapshot, subfolder="tokenizer")
    encoder = UMT5EncoderModel.from_pretrained(
        snapshot, subfolder="text_encoder", dtype=TEXT_ENCODER_DTYPE
    ).eval()
    if encoder.dtype != TEXT_ENCODER_DTYPE:
        raise TextEmbedsError(
            f"umT5 was loaded as {encoder.dtype} ({TEXT_ENCODER_DTYPE} was requested)"
        )
    return tokenizer, encoder


def valid_length(tokenizer: Any, normalized: str) -> int:
    """The length of the mask from tokenizing with the upstream's arguments (= `L_valid`).

    MUST: a prompt truncated beyond 512 is refused — the upstream truncates silently, but a fixed
    prompt that was cut would no longer correspond to its embedding.
    """
    inputs = tokenizer(
        [normalized],
        padding="max_length",
        max_length=MAX_SEQUENCE_LENGTH,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    length = int(inputs.attention_mask.gt(0).sum())
    unpadded = tokenizer([normalized], add_special_tokens=True).input_ids[0]
    if len(unpadded) > MAX_SEQUENCE_LENGTH:
        raise TextEmbedsError(f"{len(unpadded)} tokens exceed {MAX_SEQUENCE_LENGTH}")
    if length != len(unpadded):
        raise TextEmbedsError(f"mask length {length} differs from the token count {len(unpadded)}")
    return length


def encode(tokenizer: Any, encoder: Any, text: str) -> torch.Tensor:
    """Build `[1, 512, 4096]` with the upstream `_get_t5_prompt_embeds` and return the valid length
    `[L_valid, 4096]` in f32.

    It checks that everything after the valid length is exactly zero (the rows the upstream filled
    with `new_zeros`) — a wrong valid length fails here.
    """
    from diffusers import WanPipeline

    pipeline = WanPipeline(
        tokenizer=tokenizer, text_encoder=encoder, vae=None, scheduler=None, transformer=None
    )
    with torch.no_grad():
        padded = pipeline._get_t5_prompt_embeds(
            prompt=text,
            num_videos_per_prompt=1,
            max_sequence_length=MAX_SEQUENCE_LENGTH,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
    if padded.shape != (1, MAX_SEQUENCE_LENGTH, TEXT_DIM) or padded.dtype != torch.float32:
        raise TextEmbedsError(
            f"unexpected upstream embedding shape {tuple(padded.shape)} / {padded.dtype}"
        )
    length = valid_length(tokenizer, normalize(text))
    if torch.count_nonzero(padded[0, length:]) != 0:
        raise TextEmbedsError(f"a nonzero row after the valid length {length}")
    if torch.count_nonzero(padded[0, length - 1]) == 0:
        raise TextEmbedsError(
            f"the last row of the valid length {length} is zero — wrong valid length"
        )
    embeds = padded[0, :length].contiguous()
    if not bool(embeds.isfinite().all()):
        raise TextEmbedsError("the embedding has non-finite values")
    return embeds


def _versions() -> dict[str, str]:
    import diffusers
    import ftfy
    import transformers

    return {
        "diffusers": diffusers.__version__,
        "ftfy": ftfy.__version__,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
    }


def asset_metadata(
    prompts: Sequence[FixedPrompt],
    lengths: Mapping[str, int],
    normalized: Mapping[str, str],
    *,
    model: str,
    versions: Mapping[str, str],
) -> dict[str, Any]:
    """The asset's metadata (the shape before it is dumped to JSON)."""
    source = SOURCES[model]
    return {
        "source": {"repo": source.repo, "revision": source.revision},
        "text_encoder": {
            "class": "UMT5EncoderModel",
            "dtype": str(TEXT_ENCODER_DTYPE).removeprefix("torch."),
            "device": "cpu",
            "boundary": "last_hidden_state before the DiT text projection (valid rows only)",
        },
        "tokenizer": {"max_length": MAX_SEQUENCE_LENGTH, "padding": "max_length"},
        "normalizer": "diffusers.pipelines.wan.pipeline_wan.prompt_clean (ftfy.fix_text applied)",
        "versions": dict(versions),
        "prompts": [
            {
                "name": prompt.name,
                "role": prompt.role,
                "prompt": prompt.text,
                "normalized": normalized[prompt.name],
                "tokens": lengths[prompt.name],
                "source": {"url": prompt.url, "locator": prompt.locator},
            }
            for prompt in prompts
        ],
    }


def write_asset(
    path: Path, embeds: Mapping[str, torch.Tensor], metadata: Mapping[str, Any]
) -> None:
    """Write the asset (staging → replace). The correspondence of shapes, dtypes and metadata is
    checked before writing."""
    entries = metadata["prompts"]
    names = [entry["name"] for entry in entries]
    if sorted(names) != sorted(embeds) or len(set(names)) != len(names):
        raise TextEmbedsError(
            f"the metadata's prompts {names} do not match the tensors {sorted(embeds)}"
        )
    for entry in entries:
        tensor = embeds[entry["name"]]
        if tensor.dtype != torch.float32:
            raise TextEmbedsError(f"{entry['name']}: must be written as f32 ({tensor.dtype})")
        if tensor.dim() != 2 or tensor.shape[1] != TEXT_DIM:
            raise TextEmbedsError(
                f"{entry['name']}: [L_valid, {TEXT_DIM}]（{tuple(tensor.shape)}）"
            )
        if tensor.shape[0] != entry["tokens"] or not 0 < tensor.shape[0] <= MAX_SEQUENCE_LENGTH:
            raise TextEmbedsError(
                f"{entry['name']}: {tensor.shape[0]} rows differ from"
                f" the metadata's token count {entry['tokens']}"
            )
    payload = json.dumps(metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(path.name + ".staging")
    save_file(
        {name: tensor.contiguous() for name, tensor in embeds.items()},
        str(staging),
        metadata={METADATA_KEY: payload},
    )
    staging.replace(path)


def read_asset(path: Path) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Read the asset (tensors and metadata).

    Fail loudly unless the metadata has exactly the one key {@link METADATA_KEY}.
    """
    with safe_open(str(path), framework="pt") as handle:
        raw = handle.metadata() or {}
        if set(raw) != {METADATA_KEY}:
            raise TextEmbedsError(
                f"{path}: metadata keys {sorted(raw)} are not just {METADATA_KEY}"
            )
        tensors = {name: handle.get_tensor(name) for name in handle.keys()}  # noqa: SIM118
    return tensors, json.loads(raw[METADATA_KEY])


def _peak_rss_gib() -> float:
    """This process's peak RSS (GiB — Linux's `ru_maxrss` is in KiB)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1 << 20)


def generate(model: str, out: Path) -> dict[str, Any]:
    """Embed all the fixed prompts, write the asset, and return a summary."""
    started = time.perf_counter()
    tokenizer, encoder = load_text_encoder(model)
    loaded = time.perf_counter()
    print(f"[load] umT5 {TEXT_ENCODER_DTYPE}: {loaded - started:.1f} s", flush=True)
    embeds: dict[str, torch.Tensor] = {}
    normalized: dict[str, str] = {}
    lengths: dict[str, int] = {}
    seconds: dict[str, float] = {}
    for prompt in FIXED_PROMPTS:
        began = time.perf_counter()
        embeds[prompt.name] = encode(tokenizer, encoder, prompt.text)
        normalized[prompt.name] = normalize(prompt.text)
        lengths[prompt.name] = int(embeds[prompt.name].shape[0])
        seconds[prompt.name] = round(time.perf_counter() - began, 1)
        print(f"[encode] {prompt.name}: L_valid {lengths[prompt.name]} {seconds[prompt.name]} s")
    metadata = asset_metadata(FIXED_PROMPTS, lengths, normalized, model=model, versions=_versions())
    path = out / ASSET_NAME
    write_asset(path, embeds, metadata)
    data = path.read_bytes()
    return {
        "path": str(path),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "load_seconds": round(loaded - started, 1),
        "encode_seconds": seconds,
        "total_seconds": round(time.perf_counter() - started, 1),
        "peak_rss_gib": round(_peak_rss_gib(), 2),
        "torch_threads": torch.get_num_threads(),
        "prompts": [
            {
                "name": entry["name"],
                "role": entry["role"],
                "tokens": entry["tokens"],
                "normalized": entry["normalized"],
                "abs_max": float(embeds[entry["name"]].abs().max()),
            }
            for entry in metadata["prompts"]
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--model", default=DEFAULT_MODEL, choices=WAN21_MODELS)
    parser.add_argument("--out", type=Path, default=SERIES_ROOT / SERIES_NAME)
    parser.add_argument(
        "--fetch",
        action="store_true",
        help="only fetch umT5 and the tokenizer at the pinned revision",
    )
    args = parser.parse_args(argv)
    if args.fetch:
        print(f"snapshot: {text_snapshot(args.model, fetch=True)}")
        return 0
    print(json.dumps(generate(args.model, args.out), indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
