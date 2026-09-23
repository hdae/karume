"""公式 mobile QAT を再量子化せず、text グラフ・PLE・tokenizer として保存する。"""

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
from gemma4.export import MODEL_FILE, PROVENANCE
from gemma4.export_product import PLE_PROBE_FILE, PROBE_INPUTS_KEY, assert_ple_assets
from gemma4.provenance import checkpoint_fingerprint
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
    series_name,
)
from .ple import build_ple
from .trace import trace_qat

#: 上流 checkpoint のうち text 変換が**読まない**テンソル群（ヘッダの綴りで数える）。
#: KV cache の SRQ scale は f32 の KV を使うので読まず（上流 transformers も同じ）、
#: vision / audio は text 専用の変換対象外（ADR 0097 決定 1）。未対応が暗黙にならないよう、
#: 「使わなかった量」を系列の記録に残す。
UNUSED_UPSTREAM_MARKERS: Mapping[str, tuple[str, ...]] = {
    "kvCacheScales": ("k_cache_scale", "v_cache_scale"),
    "vision": ("vision_tower", "embed_vision"),
    "audio": ("audio_tower", "embed_audio"),
}


def upstream_unused(model_dir: Path) -> dict[str, int]:
    """上流 safetensors の**ヘッダだけ**を読み、変換が使わないテンソルを群ごとに数える。"""
    containers = sorted(model_dir.glob("*.safetensors"))
    if not containers:
        raise ValueError(f"{model_dir}: safetensors が 1 本も無い")
    names = [name for container in containers for name in _upstream_tensor_names(container)]
    return {
        group: sum(any(marker in name for marker in markers) for name in names)
        for group, markers in UNUSED_UPSTREAM_MARKERS.items()
    }


def _upstream_tensor_names(path: Path) -> list[str]:
    """上流 safetensors の**ヘッダだけ**からテンソル名を引く（実体は 1 バイトも読まない）。"""
    with safe_open(str(path), framework="np") as handle:
        return list(handle.keys())


def export_qat(model_dir: Path, destination: Path, model: str) -> dict[str, Any]:
    """参照・全 bytes の検査も含めて1回の据え替えにする。"""
    checkpoint = checkpoint_fingerprint(model_dir)
    loaded = load_qat(model_dir, model)
    raw = json.loads((model_dir / "tokenizer.json").read_text())
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
                # グラフ名は**部品名**（= 据え替え先のディレクトリ名 — 作業席の名前ではない）。
                graph_name=destination.name,
                fixed_weights=traced.fixed,
                assets=ple.assets,
            )
        finally:
            # MUST: 一時ファイルは据え替えの前に消す（作業席ごと据わるので、残すと配布物に混ざる）。
            ple.discard()
        assert_fixed_bytes(target, traced.fixed)
        # PLE の逆量子化ビット一致は**据えた容器から読み直して**見る（製品系列と同じ門）。
        index = json.loads(bytes(ple.assets[PLE_INDEX_ASSET].payload))
        with safe_open(str(staged / PLE_PROBE_FILE), framework="pt") as handle:
            reference = handle.get_tensor(PROBE_INPUTS_KEY)
        assert_ple_assets(target, index, ple.probe, reference)
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
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    torch.set_num_threads(4)
    source = args.input or INPUTS_ROOT / "gemma4-qat" / CHECKPOINTS[args.model]
    target = args.out or SERIES_ROOT / series_name(args.model)
    print(json.dumps(export_qat(source, target, args.model), indent=2), flush=True)


if __name__ == "__main__":
    main()
