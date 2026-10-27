"""The MOSS-Image text-embedding asset (precomputed embeddings of the fixed prompts — ADR NNNN
decision 12, plan step 0 item 5).

Each fixed prompt ({@link moss.prompts.FIXED_PROMPTS}) goes through upstream
`MOSSImagePipeline.encode_prompt` unchanged, and its output row `[T, 4096]` is written in f32 to a
safetensors file. `T` is the prompt's token count under the chat template (no padding — upstream
runs each prompt at its true length); the DiT consumes exactly these rows.

## How it is made (upstream's path, called as it is)

- The tokenizer and the text encoder are the classes `model_index.json` names (`Qwen2Tokenizer`,
  `Qwen3_5Model`), loaded from the pinned snapshot as `DiffusionPipeline.from_pretrained` loads
  them, with the text encoder in **bf16** as upstream runs it (the quick start's
  `torch_dtype={"default": torch.bfloat16, ...}`). The output bf16 → f32 is exact.
- `encode_prompt` renders the chat template (`add_generation_prompt=True, enable_thinking=True`),
  tokenizes without padding and returns `last_hidden_state[0]` (after Qwen3.5's final RMSNorm).
  The token ids it fed the encoder are taken with a forward pre-hook (not re-derived) and written to
  the metadata; `T` must equal their count, and the empty negative must give the 10 ids
  {@link EMPTY_PROMPT_IDS} (plan §1.4 — T₋ = 10).
- **CPU, a separate process.** The DiT is not read here (16.8 GB of bf16 text encoder and the
  20.3 GB f32 DiT do not share the host RAM).
- transformers runs Qwen3.5's Gated DeltaNet layers through its torch implementation on CPU
  (`flash-linear-attention` / `causal-conv1d` are CUDA kernels and are not installed); the metadata
  records which path ran (`fast_path`). An encoder that takes the CUDA kernels can give other bits.
- The CPU kernels: the metadata records torch's CPU capability (the ISA ATen dispatches its own
  kernels to) and what decides the bf16 GEMM path ({@link kernel_path}).
- **Two thread counts.** Every prompt is embedded with torch at one thread and at the default thread
  count; the two runs must agree bit for bit (rows and token ids) before anything is written.

## The asset (`text_embeds.safetensors`)

- One tensor per prompt (name = {@link moss.prompts.FixedPrompt.name}, `F32`, `[T, 4096]`).
- One metadata key ({@link METADATA_KEY}) holding JSON (`moss.records`): the source pin, the
  prompts with their token ids, and the versions and kernel path that decide the values — never
  the run's time or machine, so the same inputs give the same bytes. The run's time, peak RAM and
  machine go to `text_embeds.json` in the stage-0 reference directory.
- Location: the series directory `outputs/series/moss-image-sft-text-embeds/` (a compile product
  with no graph — docs/assets-layout.md; Wan's text embeddings live the same way).

Both fetches are needed first: `--fetch` brings the root files and the copies of the pipeline code
under `transformer/` and `scheduler/` that `moss.sources.load_pipeline_module` checks before the
import (it also brings the DiT and the VAE, which this script does not read), and `--fetch --text`
the text encoder and the tokenizer.

    uv run --group moss --inexact python -m moss.sources --fetch         # root files, pipeline code
    uv run --group moss --inexact python -m moss.sources --fetch --text  # text encoder + tokenizer
    uv run --group moss --inexact python -m moss.text_embeds             # write the asset

MUST: diffusers / transformers are imported inside the functions (the `moss` group is not in the
default sync — `tests/test_optional_group_imports.py`).
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

from _shared.paths import SERIES_ROOT
from moss import pipeline_ref, records, sources
from moss.prompts import FIXED_PROMPTS, NEGATIVE, FixedPrompt

#: The series directory (a graphless compile product).
SERIES_NAME = "moss-image-sft-text-embeds"

#: The asset's file name.
ASSET_NAME = "text_embeds.safetensors"

#: The single metadata key (its value is JSON — `moss.records`).
METADATA_KEY = "karume.moss.text_embeds"

#: The text encoder's dtype (upstream's — the quick start loads it in bf16).
TEXT_ENCODER_DTYPE = torch.bfloat16

#: The token ids of the empty prompt under the chat template:
#: `<|im_start|>`, `user`, `Ċ`, `<|im_end|>`, `Ċ`, `<|im_start|>`, `assistant`, `Ċ`, `<think>`, `Ċ`
#: (plan §1.4, checked there with a port of the BPE over the pinned `tokenizer.json`). T₋ = 10 makes
#: the negative branch's sequence N + 10 long, which is why its attention is never a multiple of
#: four keys at square sizes (plan question 16).
EMPTY_PROMPT_IDS: tuple[int, ...] = (248045, 846, 198, 248046, 198, 248045, 74455, 198, 248068, 198)


class TextEmbedsError(ValueError):
    """The embedding asset or its making left the agreed form."""


def text_classes(snapshot: Path) -> dict[str, str]:
    """The transformers class names `model_index.json` gives the tokenizer and the text encoder."""
    index = pipeline_ref.model_index(snapshot)
    names: dict[str, str] = {}
    for part in ("tokenizer", "text_encoder"):
        library, name = index[part]
        if library != "transformers":
            raise TextEmbedsError(f"model_index.json names {part} from {library!r}")
        names[part] = name
    return names


def load_text_encoder(snapshot: Path) -> tuple[Any, Any]:
    """The tokenizer and the text encoder (bf16, CPU, eval) with the classes `model_index.json`
    names."""
    import transformers

    classes = {part: getattr(transformers, name) for part, name in text_classes(snapshot).items()}
    tokenizer = classes["tokenizer"].from_pretrained(snapshot / "tokenizer")
    encoder = (
        classes["text_encoder"]
        .from_pretrained(snapshot / "text_encoder", dtype=TEXT_ENCODER_DTYPE)
        .eval()
    )
    if any(p.dtype != TEXT_ENCODER_DTYPE for p in encoder.parameters()):
        raise TextEmbedsError(f"the text encoder is not {TEXT_ENCODER_DTYPE}")
    return tokenizer, encoder


def fast_path_available() -> bool:
    """Whether transformers' Qwen3.5 takes the CUDA kernels for Gated DeltaNet (else torch's)."""
    from transformers.models.qwen3_5 import modeling_qwen3_5

    return bool(modeling_qwen3_5.is_fast_path_available)


def kernel_path() -> dict[str, Any]:
    """What chooses the CPU kernels that compute the embedding, read from torch without running a
    kernel and without oneDNN's verbose log.

    - `cpu_capability`: the ISA ATen dispatches its own kernels to (elementwise ops, reductions,
      the torch path of the Gated DeltaNet).
    - `bf16_gemm`: torch 2.13 sends a bf16 `linear` to oneDNN's matmul when oneDNN is enabled and
      its bf16 device check passes (ATen's `use_mkldnn_bf16_matmul` — an implementation fact, not a
      guarantee); oneDNN then picks its kernel by the CPU's AVX512-BF16 and AMX support, recorded
      next to it. The kernel oneDNN actually ran is visible only through `ONEDNN_VERBOSE`, which
      logs every primitive call, so it is not recorded; on 2026-10-27 it showed
      `brg_matmul:avx512_core_bf16` for bf16 `[T, 4096] × [4096, 4096]` on an AVX512-BF16 CPU
      without AMX (T = 10 and 63).
    """
    return {
        "cpu_capability": torch.backends.cpu.get_cpu_capability(),
        "bf16_gemm": {
            "mkldnn_enabled": bool(torch.backends.mkldnn.enabled),
            "mkldnn_bf16_supported": bool(torch.ops.mkldnn._is_mkldnn_bf16_supported()),
            "avx512_bf16": bool(torch.cpu._is_avx512_bf16_supported()),
            "amx_tile": bool(torch.cpu._is_amx_tile_supported()),
        },
    }


def encode(pipeline: Any, text: str) -> tuple[torch.Tensor, list[int]]:
    """Upstream `encode_prompt(text)` → (`[T, 4096]` f32, the token ids the encoder received)."""
    seen: list[list[int]] = []

    def keep_ids(_module: Any, _args: Any, kwargs: dict[str, Any]) -> None:
        seen.append([int(i) for i in kwargs["input_ids"][0]])

    handle = pipeline.text_encoder.register_forward_pre_hook(keep_ids, with_kwargs=True)
    try:
        embeds, mask = pipeline.encode_prompt(text)
    finally:
        handle.remove()
    if len(seen) != 1:
        raise TextEmbedsError(f"the text encoder ran {len(seen)} times for one prompt")
    ids = seen[0]
    if embeds.dtype != TEXT_ENCODER_DTYPE or embeds.shape != (1, len(ids), pipeline_ref.TEXT_DIM):
        raise TextEmbedsError(f"encode_prompt returned {tuple(embeds.shape)} / {embeds.dtype}")
    if not bool(mask.all()):
        raise TextEmbedsError("a single prompt came back with a padded mask")
    # `encode_prompt` tokenizes with `truncation=True, max_length=4096`: a prompt of exactly 4096
    # tokens and a longer one cut to 4096 give the same ids, so T = 4096 cannot be told apart from a
    # truncated prompt and is refused (1 ≤ T < 4096).
    if len(ids) >= pipeline_ref.MAX_SEQUENCE_LENGTH:
        raise TextEmbedsError(
            f"{len(ids)} tokens — the prompt may have been truncated"
            f" (1 ≤ T < {pipeline_ref.MAX_SEQUENCE_LENGTH})"
        )
    rows = embeds[0].float()
    if not bool(rows.isfinite().all()):
        raise TextEmbedsError("the embedding has non-finite values")
    return rows, ids


def check_tokens(prompt: FixedPrompt, rows: torch.Tensor, ids: Sequence[int]) -> None:
    """`T` equals the token count; the empty negative is exactly {@link EMPTY_PROMPT_IDS}."""
    if rows.shape[0] != len(ids):
        raise TextEmbedsError(f"{prompt.name}: {rows.shape[0]} rows for {len(ids)} tokens")
    if prompt.text == "" and tuple(ids) != EMPTY_PROMPT_IDS:
        raise TextEmbedsError(f"{prompt.name}: the empty prompt gave {list(ids)}")


def asset_metadata(
    prompts: Sequence[FixedPrompt],
    token_ids: Mapping[str, Sequence[int]],
    *,
    classes: Mapping[str, str],
    run: Mapping[str, Any],
) -> dict[str, Any]:
    """The asset's metadata (before it is serialised); `classes` is {@link text_classes}."""
    return {
        "source": {
            "repo": sources.SOURCE.repo,
            "revision": sources.SOURCE.revision,
            "pipeline_code_sha256": sources.PIPELINE_CODE_SHA256,
        },
        "text_encoder": {
            "class": classes["text_encoder"],
            "dtype": str(TEXT_ENCODER_DTYPE).removeprefix("torch."),
            "device": "cpu",
            "boundary": "encode_prompt output: last_hidden_state[0] (after the final RMSNorm)",
        },
        "tokenizer": {
            "class": classes["tokenizer"],
            "chat_template": "tokenizer/chat_template.jinja",
        },
        "run": dict(run),
        "prompts": [
            {
                "name": prompt.name,
                "role": prompt.role,
                "prompt": prompt.text,
                "tokens": len(token_ids[prompt.name]),
                "token_ids": list(token_ids[prompt.name]),
                "source": {"url": prompt.url, "locator": prompt.locator},
            }
            for prompt in prompts
        ],
    }


def write_asset(
    path: Path, embeds: Mapping[str, torch.Tensor], metadata: Mapping[str, Any]
) -> None:
    """Check the tensors against the metadata, then write the asset."""
    entries = metadata["prompts"]
    names = [entry["name"] for entry in entries]
    if sorted(names) != sorted(embeds) or len(set(names)) != len(names):
        raise TextEmbedsError(f"metadata prompts {names} and tensors {sorted(embeds)} disagree")
    negatives = [entry for entry in entries if entry["role"] == NEGATIVE]
    if len(negatives) != 1:
        raise TextEmbedsError(f"{len(negatives)} negative prompts (one is agreed)")
    for entry in entries:
        tensor = embeds[entry["name"]]
        if tensor.dtype != torch.float32 or tensor.dim() != 2:
            raise TextEmbedsError(f"{entry['name']}: {tensor.dtype} {tuple(tensor.shape)}")
        rows = tensor.shape[0]
        if (
            tensor.shape[1] != pipeline_ref.TEXT_DIM
            or not 1 <= rows < pipeline_ref.MAX_SEQUENCE_LENGTH
        ):
            raise TextEmbedsError(
                f"{entry['name']}: [T, {pipeline_ref.TEXT_DIM}] with"
                f" 1 ≤ T < {pipeline_ref.MAX_SEQUENCE_LENGTH} expected, got {tuple(tensor.shape)}"
            )
        if rows != entry["tokens"] or rows != len(entry["token_ids"]):
            raise TextEmbedsError(f"{entry['name']}: {rows} rows, metadata says {entry['tokens']}")
    records.write_record(path, embeds, metadata, key=METADATA_KEY)


def read_asset(path: Path) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """The asset's tensors and metadata (fails loudly on another metadata layout)."""
    return records.read_record(path, key=METADATA_KEY)


def generate(out: Path) -> dict[str, Any]:
    """Embed every fixed prompt, write the asset and return the run's summary."""
    clock = records.Clock()
    snapshot, module = pipeline_ref.upstream(text=True)
    tokenizer, encoder = load_text_encoder(snapshot)
    pipeline = module.MOSSImagePipeline(
        transformer=None, scheduler=None, text_encoder=encoder, tokenizer=tokenizer, vae=None
    )
    load_seconds = clock.lap("load")
    default_threads = torch.get_num_threads()
    thread_counts = (1, default_threads) if default_threads != 1 else (1, 2)
    runs: list[tuple[dict[str, torch.Tensor], dict[str, list[int]]]] = []
    seconds: dict[str, dict[str, float]] = {}
    try:
        for threads in thread_counts:
            torch.set_num_threads(threads)
            embeds: dict[str, torch.Tensor] = {}
            token_ids: dict[str, list[int]] = {}
            laps: dict[str, float] = {}
            for prompt in FIXED_PROMPTS:
                rows, ids = encode(pipeline, prompt.text)
                check_tokens(prompt, rows, ids)
                embeds[prompt.name], token_ids[prompt.name] = rows, ids
                laps[prompt.name] = clock.lap(f"encode {prompt.name} (T = {len(ids)}, {threads} t)")
            runs.append((embeds, token_ids))
            seconds[f"{threads} thread(s)"] = laps
    finally:
        torch.set_num_threads(default_threads)
    (first, first_ids), (embeds, token_ids) = runs
    differing = [
        name
        for name in token_ids
        if first_ids[name] != token_ids[name] or not records.bits_equal(first[name], embeds[name])
    ]
    if differing:
        raise TextEmbedsError(f"the thread counts {thread_counts} disagree on {differing}")
    # Only what decides the values goes into the asset (the same inputs give the same bytes);
    # the run's time, RAM and machine go into the summary.
    run = {
        "versions": records.versions(),
        "fast_path": fast_path_available(),
        **kernel_path(),
        "thread_counts": list(thread_counts),
        "threads_agree": True,
    }
    metadata = asset_metadata(FIXED_PROMPTS, token_ids, classes=text_classes(snapshot), run=run)
    path = out / ASSET_NAME
    write_asset(path, embeds, metadata)
    data = path.read_bytes()
    return {
        "path": str(path),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        **run,
        "environment": records.environment(),
        "seconds": {"load": load_seconds, "encode": seconds, "total": clock.total()},
        "peak_rss_gib": round(records.peak_rss_gib(), 2),
        "prompts": {
            name: {"tokens": len(ids), "abs_max": float(embeds[name].abs().max())}
            for name, ids in token_ids.items()
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("--out", type=Path, default=SERIES_ROOT / SERIES_NAME)
    parser.add_argument(
        "--summary-dir",
        type=Path,
        default=None,
        help="where the run summary (time, RAM, machine) goes (default: the stage-0 reference dir)",
    )
    args = parser.parse_args(argv)
    summary = generate(args.out)
    text = json.dumps(summary, indent=1, ensure_ascii=False)
    print(text)
    summary_dir = args.summary_dir or records.default_reference_dir()
    summary_dir.mkdir(parents=True, exist_ok=True)
    (summary_dir / "text_embeds.json").write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
