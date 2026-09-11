"""固定 QAT の別ファミリ配布。グラフと PLE の共通門は Gemma から再利用する。"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from _shared.licenses import apache_license_2_0
from gemma4.distribution import (
    assert_gemma4_graph,
    assert_gemma4_ple_shards,
    assert_gemma4_tokenizer,
    gemma4_assets,
    gemma4_hidden_size,
    gemma4_max_position,
    gemma4_ple_index,
    gemma4_ple_role,
    gemma4_rope,
    gemma4_sampler,
    gemma4_text_config,
    gemma4_vocab_size,
)
from karume.dist import (
    Artifact,
    DistError,
    ModelPlan,
    Pipeline,
    WeightFiles,
    assert_storage,
    assert_storage_absent,
    ir_graph,
)
from karume.modelcard import CardMetadata, frontmatter, models, quants, render, require_pipeline

from .config import CHECKPOINTS, MAX_CHUNK_LENGTH, MAX_SELECTED_ROWS, checkpoint_name, series_name


def assert_qat_graph(graph: Mapping[str, Any]) -> None:
    """通常 Gemma や SRQ の落ちた QAT を同じ配布として通さない。"""
    nodes, initializers = graph["nodes"], graph["initializers"]
    producers = {value: node for node in nodes for value in node["outs"]}
    consumers: dict[str, list[Mapping[str, Any]]] = {}
    for node in nodes:
        for value in node["ins"]:
            consumers.setdefault(value, []).append(node)
    embeddings = [
        node for node in nodes if node["op"] == "embedding" and node["ins"][1] == "input_ids"
    ]
    if len(embeddings) != 1:
        raise DistError("QAT token embedding が1本でない")
    token_weight = embeddings[0]["ins"][0]
    if initializers[token_weight].get("storage", {}).get("dtype") != "i2":
        raise DistError("QAT token embedding は固定 I2 が必要")
    head_count = 0
    ordinary_count = 0
    for node in nodes:
        if node["op"] != "linear":
            continue
        weight = initializers.get(node["ins"][1], {})
        dtype = weight.get("storage", {}).get("dtype", "f32")
        if dtype == "f32":
            if weight.get("tensor") != "model.model.per_layer_model_projection.weight":
                raise DistError("QAT の固定量子化でない linear が許可した projection 以外に在る")
            ordinary_count += 1
            continue
        if dtype not in ("i2", "i4", "i8"):
            raise DistError(f"QAT linear の固定格納に未対応: {dtype}")
        before = producers.get(node["ins"][0], {})
        after = consumers.get(node["outs"][0], [])
        if (
            before.get("op") != "static_quantize"
            or len(after) != 1
            or after[0]["op"] != "static_quantize"
        ):
            raise DistError("QAT linear の前後に固定 SRQ が必要")
        head_count += node["ins"][1] == token_weight
    if head_count != 1 or ordinary_count != 1:
        raise DistError("QAT の共有 head または非量子化 projection の本数が違う")


def qat_plan(series_dir: Path, model: str) -> ModelPlan:
    """検査した1系列から配布計画を作る。元チェックポイントの再ダウンロードは不要。"""
    source = series_dir / series_name(model)
    reference = json.loads((source / "reference.json").read_text())
    if (
        reference.get("schema") != 1
        or reference.get("family") != "gemma4-qat"
        or reference.get("model") != model
        or reference.get("maxChunkLength") != MAX_CHUNK_LENGTH
        or reference.get("maxSelectedRows") != MAX_SELECTED_ROWS
        or reference.get("fixedBytesExact") is not True
        or reference.get("pleBytesExact") is not True
    ):
        raise DistError("QAT reference の family/model/trace範囲/固定bytes 検証が合わない")
    index = gemma4_ple_index(source, storage="i4" if model == "e2b" else "i2")
    container = source / "model.safetensors"
    for dtype in ("I2", "I4", "I8"):
        assert_storage("model", container, {"model": dtype})
    assert_storage_absent("model", container, {"model": ("F16",)})
    graph = ir_graph(container)
    assert_qat_graph(graph)
    config = gemma4_text_config(source)
    where = str(source / "config.json")
    rope = gemma4_rope(config, where)
    vocab = gemma4_vocab_size(graph, container)
    assert_gemma4_graph(graph, container, index, rope, gemma4_hidden_size(config, where))
    if index["tokens"] != vocab:
        raise DistError("QAT PLE とグラフの語彙数が違う")
    placements = {
        gemma4_ple_role(i): source / shard["file"] for i, shard in enumerate(index["shards"])
    }
    assert_gemma4_ple_shards(placements, index)
    assert_gemma4_tokenizer(source / "tokenizer.json", vocab)
    artifacts = {
        "model": Artifact("model/model.i4.safetensors", source=container),
        "tokenizer": Artifact("tokenizer/tokenizer.json", source=source / "tokenizer.json"),
        "ple_index": Artifact("ple/ple.json", source=source / "ple.json"),
    }
    for role, path in placements.items():
        artifacts[role] = Artifact(f"ple/{path.name}", source=path)
    max_position = gemma4_max_position(config, where)
    if max_position < MAX_CHUNK_LENGTH:
        raise DistError("QAT の位置上限が初期容量より小さい")
    return ModelPlan(
        name=model,
        pipeline="gemma4-qat/1",
        artifacts=artifacts,
        weights={"model": {"i4": WeightFiles("model")}},
        assets=gemma4_assets(index),
        quants={
            "i4": {
                "weights": {"model": "i4"},
                "session": {},
                "label": "Fixed mixed int2/int4/int8 with SRQ",
                "description": "Official mobile QAT integers and scales, preserved without "
                "requantization; fixed activation rounding (SRQ).",
            }
        },
        default_quant="i4",
        pipeline_config={
            "chunkLength": 32,
            "maxChunkLength": reference["maxChunkLength"],
            "capacity": MAX_CHUNK_LENGTH,
            "maxPosition": max_position,
            "rope": rope,
            "sampler": gemma4_sampler(source),
        },
    )


def repo_name(model: str) -> str:
    checkpoint_name(model)
    return "karume-gemma4-qat"


def render_card(manifest: Mapping[str, Any], repo: str) -> str:
    """数値・モデル一覧・量子化表を manifest から描く。固定値保持を再量子化と呼ばない。"""
    require_pipeline(manifest, "gemma4-qat/1")
    upstream = tuple(f"google/{checkpoint_name(model)}" for model in manifest["models"])
    metadata = CardMetadata(
        pipeline_tag="text-generation",
        base_model=upstream,
        license="apache-2.0",
        tags=("webgpu", "gemma", "text-generation"),
    )
    sections = [
        frontmatter(metadata),
        [
            "# Gemma 4 QAT Text Decoder — Karume",
            "",
            "Experimental text-only conversion of the official mobile QAT checkpoints.",
            "Fixed integers and scales are preserved; activations use "
            "static range quantization (SRQ).",
            "CPU and GPU reductions can cross rounding boundaries and select different tokens.",
            "Broad quality and long-context validation remain incomplete. MTP is not supported.",
            "See the upstream model cards for capabilities and usage guidance:",
            "",
            *[f"- [{repo}](https://huggingface.co/{repo})" for repo in upstream],
            "",
            "Licensed under Apache 2.0; see LICENSE.md and NOTICE.md.",
        ],
        models(manifest),
    ]
    for name, model in manifest["models"].items():
        sections.extend([[f"## {name}", ""], quants(model)])
    return render(sections)


NOTICE = (
    """# NOTICE

This distribution contains modified text decoders from the official Google Gemma 4 mobile QAT
checkpoints listed below, under the Apache License, Version 2.0 (see LICENSE.md).
The actual included models are identified in README.md and karume.json.

"""
    + "\n".join(f"- google/{name}" for name in CHECKPOINTS.values())
    + """

The text decoder was extracted, its graph was converted to Karume states form, per-layer embeddings
were moved to a host-read sidecar, and rotary inputs are generated on the host. Fixed quantized
integers and scales were retained without requantization. Vision, audio, and MTP are not included.
No upstream implementation source code is redistributed by this recipe.
"""
)

PIPELINE = Pipeline(
    default_model="e2b",
    repo_name=repo_name,
    plan=qat_plan,
    card_profiles={"gemma4-qat": render_card},
    root_files={"LICENSE.md": apache_license_2_0(), "NOTICE.md": NOTICE},
)
