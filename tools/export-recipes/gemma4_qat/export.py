"""Save the official mobile QAT, without requantization, as the text graph, PLE and tokenizer."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

from _shared.gemma_tokenizer import asset_payload, compile_tokenizer
from _shared.paths import INPUTS_ROOT, SERIES_ROOT
from _shared.upstream import hub_snapshot_dir
from gemma4.distribution import GEMMA4_ROLE
from gemma4.export import MODEL_FILE, PROVENANCE
from gemma4.export_product import PLE_PROBE_FILE, PROBE_INPUTS_KEY, assert_ple_assets
from gemma4.provenance import assert_pinned_fingerprint, checkpoint_fingerprint
from gemma4.sources import assert_model_pins
from karume import publish_model
from karume.artifacts import staged_publication
from karume.container import container_parts
from karume.ple import PLE_INDEX_ASSET

from .audit import assert_fixed_bytes
from .checkpoint import load_qat
from .config import (
    CHECKPOINTS,
    MAX_CHUNK_LENGTH,
    MAX_SELECTED_ROWS,
    REFERENCE_SCHEMA,
    SOURCES,
    series_name,
)
from .ple import build_ple
from .trace import trace_qat

#: The tensor groups of the upstream checkpoint the text conversion **does not read** (counted by
#: their header spellings). The KV cache's SRQ scales are not read because the KV is f32 (as in
#: upstream transformers), and vision / audio are outside the text-only conversion (ADR 0097
#: decision 1). So that nothing is unsupported implicitly, "how much went unused" is kept in the
#: series record.
UNUSED_UPSTREAM_MARKERS: Mapping[str, tuple[str, ...]] = {
    "kvCacheScales": ("k_cache_scale", "v_cache_scale"),
    "vision": ("vision_tower", "embed_vision"),
    "audio": ("audio_tower", "embed_audio"),
}


def upstream_unused(model_dir: Path) -> dict[str, int]:
    """Read **only the headers** of the upstream safetensors and count, per group, the tensors the
    conversion does not use."""
    containers = sorted(model_dir.glob("*.safetensors"))
    if not containers:
        raise ValueError(f"{model_dir}: no safetensors at all")
    names = [name for container in containers for name in _upstream_tensor_names(container)]
    return {
        group: sum(any(marker in name for marker in markers) for name in names)
        for group, markers in UNUSED_UPSTREAM_MARKERS.items()
    }


def _upstream_tensor_names(path: Path) -> list[str]:
    """Take the tensor names from **only the header** of an upstream safetensors (not one byte of
    data is read)."""
    with safe_open(str(path), framework="np") as handle:
        return list(handle.keys())


def export_qat(
    model_dir: Path, destination: Path, model: str, *, tokenizer_dir: Path
) -> dict[str, Any]:
    """Make it one swap into place, the reference and the all-bytes check included.

    `model_dir` holds the mobile QAT weights, `config.json` and `generation_config.json`;
    `tokenizer_dir` the `tokenizer.json` (the regular `-it` repository — `config.SOURCES`). Both
    are checked against their pins before the load, and the fingerprint the record names once more
    on every byte read (the q4_0 QAT line's rule — `gemma4.export_product_qat`).
    """
    pins = SOURCES[model].pins
    assert_model_pins(model_dir, tokenizer_dir, pins)
    checkpoint = checkpoint_fingerprint(
        model_dir, pins.label, tokenizer_dir=tokenizer_dir, pins=pins
    )
    assert_pinned_fingerprint(checkpoint, pins)
    loaded = load_qat(model_dir, model)
    raw = json.loads((tokenizer_dir / "tokenizer.json").read_text())
    compiled = compile_tokenizer(raw)
    with staged_publication(destination) as staged:
        staged.mkdir(parents=True)
        ple = build_ple(
            loaded.ple,
            loaded.config.num_hidden_layers,
            loaded.config.hidden_size_per_layer_input,
            staged,
        )
        target = staged / MODEL_FILE
        try:
            traced = trace_qat(loaded, torch.tensor([[2, 105, 2364, 107]], dtype=torch.int64))
            publish_model(
                target,
                traced.graph,
                traced.tensors,
                provenance=PROVENANCE,
                # The graph name is **the component name** (= the key of weights in karume.json).
                # A QAT series' component name is the same single word as the regular Gemma's
                # (the weights of `gemma4_qat.distribution` use the same constant), not the series
                # name (`gemma4-qat-<model>-product`) (container-v1 §2.1).
                graph_name=GEMMA4_ROLE,
                fixed_weights=traced.fixed,
                assets=ple.assets,
            )
        finally:
            # MUST: delete the temporary files before the swap (the whole working directory is
            # swapped in, so leftovers would end up in the distribution).
            ple.discard()
        assert_fixed_bytes(target, traced.fixed)
        # PLE's bit-exact dequantization is checked **by reading back from the written
        # container** (the same check as the product series).
        index = json.loads(bytes(ple.assets[PLE_INDEX_ASSET].payload))
        with safe_open(str(staged / PLE_PROBE_FILE), framework="pt") as handle:
            reference = handle.get_tensor(PROBE_INPUTS_KEY)
        assert_ple_assets(target, index, ple.probe, reference)
        # The tokenizer's own repository: the asset is compiled from its `tokenizer.json`.
        source = {"repo": pins.tokenizer.repo, "checkpoint": checkpoint}
        (staged / "tokenizer.json").write_text(
            json.dumps(asset_payload(compiled, source=source), ensure_ascii=False) + "\n"
        )
        # dist does not reread the original config; it derives the configuration from the
        # declarations saved in the same swap.
        (staged / "config.json").write_bytes((model_dir / "config.json").read_bytes())
        (staged / "generation_config.json").write_bytes(
            (model_dir / "generation_config.json").read_bytes()
        )
        record = {
            "schema": REFERENCE_SCHEMA,
            "family": "gemma4-qat",
            "model": model,
            "checkpoint": checkpoint,
            "maxChunkLength": MAX_CHUNK_LENGTH,
            "maxSelectedRows": MAX_SELECTED_ROWS,
            "fixedWeights": len(traced.fixed),
            "storageCounts": {
                dtype: sum(value.dtype == dtype for value in traced.fixed.values())
                for dtype in ("i2", "i4", "i8")
            },
            "weightFiles": len(container_parts(target)),
            "pleBlocks": ple.blocks,
            "upstreamUnused": upstream_unused(model_dir),
        }
        (staged / "reference.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=tuple(CHECKPOINTS), default="e2b")
    parser.add_argument("--input", type=Path)
    parser.add_argument(
        "--tokenizer-input",
        type=Path,
        help="the directory holding tokenizer.json (default: the hub snapshot of the model's"
        " pinned tokenizer source)",
    )
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    torch.set_num_threads(4)
    source = args.input or INPUTS_ROOT / "gemma4-qat" / CHECKPOINTS[args.model]
    tokenizer_pin = SOURCES[args.model].pins.tokenizer
    tokenizer = args.tokenizer_input or hub_snapshot_dir(tokenizer_pin.repo, tokenizer_pin.revision)
    target = args.out or SERIES_ROOT / series_name(args.model)
    record = export_qat(source, target, args.model, tokenizer_dir=tokenizer)
    print(json.dumps(record, indent=2), flush=True)


if __name__ == "__main__":
    main()
