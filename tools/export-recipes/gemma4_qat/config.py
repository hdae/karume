"""The QAT sources and series names. A small source of truth so that dist does not import
torch."""

from collections.abc import Mapping
from dataclasses import dataclass

from gemma4.sources import REGULAR_TOKENIZER_PINS, HubPin, ModelPins

CHECKPOINTS: Mapping[str, str] = {
    "e2b": "gemma-4-E2B-it-qat-mobile-transformers",
    "e4b": "gemma-4-E4B-it-qat-mobile-transformers",
}


@dataclass(frozen=True)
class QatSource:
    """The pinned upstream commits one model was exported from: the mobile QAT repository for the
    weights and their configuration, the regular `-it` repository for the tokenizer."""

    #: The weights pin (`model.safetensors`, `config.json` — `generation_config.json` is read from
    #: the same snapshot) and the tokenizer pin (`tokenizer.json`). Each names the SHA-256 of the
    #: files `reference.json` fingerprints from that source (`gemma4.provenance.FILES_BY_ROLE`):
    #: the LFS oid the HF API reports for `model.safetensors` and `tokenizer.json`, and the digest
    #: of `config.json` as served raw at that commit. The export and the distribution check the
    #: series' record against them, so the card's "at commit" lines name the bytes actually
    #: converted.
    pins: ModelPins
    #: `license:` of the upstream `README.md` front matter at both pinned commits (one value: the
    #: card attributes both sources under one licence).
    license: str


#: Model → the pinned upstream sources.
#:
#: - weights: the revisions the local inputs were fetched at on 2026-09-11
#:   (`inputs/gemma4-qat/<checkpoint>/.cache/huggingface/download/*.metadata`), still `main` of
#:   both repositories on 2026-11-01 (HF API; last modified 2026-07-20; not gated).
#: - tokenizer: the regular `-it` repository at Google's glitch-token fixes of 2026-11-03
#:   (`gemma4.sources.REGULAR_TOKENIZER_PINS` — the q4_0 QAT line's pins, one spelling). The
#:   mobile repositories still ship the `tokenizer.json` from before those fixes (SHA-256
#:   `cc8d3a0c…`, the same in both), and the maintainer approved shipping the fixed tokenizer
#:   (2026-11-04).
#:
#: The licence review is `THIRD_PARTY_NOTICES.md`; moving a pin needs that review again.
SOURCES: Mapping[str, QatSource] = {
    "e2b": QatSource(
        pins=ModelPins(
            weights=HubPin(
                repo=f"google/{CHECKPOINTS['e2b']}",
                revision="dd693ff40353f057ca5f07e945ad867f4afbf2ec",
                sha256={
                    "model.safetensors": (
                        "efab429012b97ab986c4d4838a46ff3ad95d618b42ce514771ca40fadc76a9a4"
                    ),
                    "config.json": (
                        "cf6d7dc22738b5e6beb364bac833d78b869f5a6ffd57dfc96c6be3f2abc80424"
                    ),
                },
            ),
            tokenizer=REGULAR_TOKENIZER_PINS["e2b"],
        ),
        license="apache-2.0",
    ),
    "e4b": QatSource(
        pins=ModelPins(
            weights=HubPin(
                repo=f"google/{CHECKPOINTS['e4b']}",
                revision="8e3ac78f1a4a97c19f5f194ab362a2206f71d961",
                sha256={
                    "model.safetensors": (
                        "391946da2e0bec22288e9fe50a4d31d2401c9570ba3baaf0ba43d644dadeb1d4"
                    ),
                    "config.json": (
                        "c4847fd621d80f160945b62f4a2d9413fc82b81f17955b49ae32ea9feb00c5c6"
                    ),
                },
            ),
            tokenizer=REGULAR_TOKENIZER_PINS["e4b"],
        ),
        license="apache-2.0",
    ),
}

#: Model → PLE storage bit width (the primary source is upstream's `QuantizedEmbedding.num_bits`).
#: The upstream check (`checkpoint.py`) and the distribution comparison (`distribution.py`) used
#: to spell the same expectation separately, so the Python-side spelling lives in this one place
#: (the TS side has it in `packages/models/src/gemma/qat.ts` — the language boundary cannot be
#: crossed, so the goal is "one place per language").
PLE_BITS: Mapping[str, int] = {"e2b": 4, "e4b": 2}

#: The trace-time upper bound of the symbol `M` (rows per prefill). The source of `trace.py`'s
#: `Dim` and of the `pipelineConfig.maxChunkLength` the distribution declares.
MAX_CHUNK_LENGTH = 768
MAX_SELECTED_ROWS = 9

#: The default conversation capacity (`capacity`) and rows per prefill (`chunkLength`). **The
#: same values as the regular Gemma** (decision of 2026-09-19; ADR 0097 addendum 7) — the initial
#: 128 / 32 reused the trace-bound constants as the conversation capacity, a convenience of the
#: conversion rather than the model's position limit (131,072 in the official config). Both are
#: run-time knobs, so users can override them with `--capacity` / `--chunk-length`.
DEFAULT_CAPACITY = 4096
DEFAULT_CHUNK_LENGTH = 768

#: The schema version of `reference.json`. Schema 2 dropped "the bytes-match flag that is always
#: True" and made the distribution side compare the number of checked fixed weights, the storage
#: breakdown and the PLE count with the actual ones.
#:
#: Schema 3 is **a promotion of how PLE is counted**: from the sidecar's shard count
#: (`pleShards`) to the block count of the container's asset (`pleBlocks`). Blocks are cut
#: ignoring the old shard boundaries, so the two are different numbers; had the schema stayed,
#: an old-generation record would pass the version check and fail only as "a missing field"
#: (without spelling out what disagrees). The migration brings records to schema 3 too
#: (`migrate_series.rewrite_qat_reference`).
REFERENCE_SCHEMA = 3


def checkpoint_name(model: str) -> str:
    if model not in CHECKPOINTS:
        raise ValueError(f"unsupported QAT model: {model}")
    return CHECKPOINTS[model]


def checkpoint_repo(model: str) -> str:
    """The upstream HF repository ID of `model` (`google/<checkpoint>`)."""
    return f"google/{checkpoint_name(model)}"


def series_name(model: str) -> str:
    checkpoint_name(model)
    return f"gemma4-qat-{model}-product"
