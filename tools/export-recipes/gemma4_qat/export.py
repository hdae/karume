"""公式 mobile QAT を再量子化せず、text グラフ・PLE・tokenizer として保存する。"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from _shared.gemma_tokenizer import asset_payload, compile_tokenizer
from _shared.paths import INPUTS_ROOT, SERIES_ROOT
from gemma4.provenance import checkpoint_fingerprint
from karume import publish_model
from karume.artifacts import staged_publication
from karume.shards import resolve_shards

from .audit import assert_fixed_bytes
from .checkpoint import load_qat
from .config import CHECKPOINTS, MAX_CHUNK_LENGTH, MAX_SELECTED_ROWS, series_name
from .ple import write_ple
from .trace import trace_qat


def export_qat(model_dir: Path, destination: Path, model: str) -> dict[str, Any]:
    """参照・全 bytes の検査も含めて1回の据え替えにする。"""
    checkpoint = checkpoint_fingerprint(model_dir)
    loaded = load_qat(model_dir, model)
    raw = json.loads((model_dir / "tokenizer.json").read_text())
    compiled = compile_tokenizer(raw)
    with staged_publication(destination) as staged:
        staged.mkdir(parents=True)
        index = write_ple(
            loaded.ple,
            loaded.config.num_hidden_layers,
            loaded.config.hidden_size_per_layer_input,
            staged,
        )
        traced = trace_qat(loaded, torch.tensor([[2, 105, 2364, 107]], dtype=torch.int64))
        target = staged / "model.safetensors"
        publish_model(target, traced.graph, traced.tensors, fixed_weights=traced.fixed)
        assert_fixed_bytes(target, traced.fixed)
        source = {"repo": f"google/{CHECKPOINTS[model]}", "checkpoint": checkpoint}
        (staged / "tokenizer.json").write_text(
            json.dumps(asset_payload(compiled, source=source), ensure_ascii=False) + "\n"
        )
        # dist は元 config を再読せず、同じ据え替えで保存した宣言から構成を導く。
        (staged / "config.json").write_bytes((model_dir / "config.json").read_bytes())
        (staged / "generation_config.json").write_bytes(
            (model_dir / "generation_config.json").read_bytes()
        )
        record = {
            "schema": 1,
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
            "weightFiles": len(resolve_shards(target)),
            "pleShards": len(index["shards"]),
            "fixedBytesExact": True,
            "pleBytesExact": True,
        }
        (staged / "reference.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=tuple(CHECKPOINTS), default="e2b")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    torch.set_num_threads(4)
    source = args.input or INPUTS_ROOT / "gemma4-qat" / CHECKPOINTS[args.model]
    target = args.out or SERIES_ROOT / series_name(args.model)
    print(json.dumps(export_qat(source, target, args.model), indent=2), flush=True)


if __name__ == "__main__":
    main()
