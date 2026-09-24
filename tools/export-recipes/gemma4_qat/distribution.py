"""固定 QAT の別ファミリ配布。グラフと PLE の共通門は Gemma から再利用する。"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from _shared.container_read import read_layouts
from _shared.licenses import apache_license_2_0
from gemma4.distribution import (
    GEMMA4_ASSETS,
    GEMMA4_ROLE,
    assert_gemma4_graph,
    assert_gemma4_ple_assets,
    assert_gemma4_tokenizer,
    gemma4_hidden_size,
    gemma4_max_position,
    gemma4_pipeline_config,
    gemma4_ple_index,
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

from .config import (
    CHECKPOINTS,
    DEFAULT_CAPACITY,
    DEFAULT_CHUNK_LENGTH,
    MAX_CHUNK_LENGTH,
    MAX_SELECTED_ROWS,
    PLE_BITS,
    REFERENCE_SCHEMA,
    checkpoint_name,
    series_name,
)


def assert_qat_graph(graph: Mapping[str, Any], layouts: Mapping[str, str]) -> None:
    """通常 Gemma や SRQ の落ちた QAT を同じ配布として通さない。

    `layouts` はテンソルキー → 格納の layout（{@link _shared.container_read.read_layouts}）—
    IR v2 のグラフ記述は格納を持たない（正本は束縛表）ので、固定格納の判定はそちらから引く。
    IR v2 では initializer の名前がテンソルキーそのものなので、両者は同じ鍵で引ける。
    """
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
    if token_weight not in initializers or layouts.get(token_weight) != "i2":
        raise DistError("QAT token embedding は固定 i2 が必要")
    head_count = 0
    ordinary_count = 0
    for node in nodes:
        if node["op"] != "linear":
            continue
        key = node["ins"][1]
        dtype = layouts.get(key, "f32") if key in initializers else "f32"
        if dtype == "f32":
            if key != "model.model.per_layer_model_projection.weight":
                raise DistError("QAT の固定量子化でない linear が許可した projection 以外に在る")
            ordinary_count += 1
            continue
        if dtype not in ("i2", "i4", "i8"):
            raise DistError(f"QAT linear の固定格納に未対応: {dtype}")
        # 共有 head だけ前後の SRQ を省略してよい — 上流 lm_head の SRQ scale は入出力とも 0
        # （未較正 = 恒等）で、recipe は scale=0 の SRQ を IR に挟まない（`checkpoint.py` の
        # `TraceLinear`）。他の量子化 linear の scale は全て正なので、前後とも必須のまま。
        shared_head = node["ins"][1] == token_weight
        head_count += shared_head
        if shared_head:
            continue
        before = producers.get(node["ins"][0], {})
        after = consumers.get(node["outs"][0], [])
        if (
            before.get("op") != "static_quantize"
            or len(after) != 1
            or after[0]["op"] != "static_quantize"
        ):
            raise DistError("QAT linear の前後に固定 SRQ が必要")
    if head_count != 1 or ordinary_count != 1:
        raise DistError("QAT の共有 head または非量子化 projection の本数が違う")


#: 固定格納の dtype（`reference.json` の `storageCounts` の鍵でもある）。
FIXED_STORAGE_DTYPES: tuple[str, ...] = ("i2", "i4", "i8")


def fixed_storage_counts(graph: Mapping[str, Any], layouts: Mapping[str, str]) -> dict[str, int]:
    """束縛表の initializer を固定格納の語彙別に数える（`reference.json` の突合相手）。"""
    counts = dict.fromkeys(FIXED_STORAGE_DTYPES, 0)
    for key in graph["initializers"]:
        dtype = layouts.get(key)
        if dtype in counts:
            counts[dtype] += 1
    return counts


#: 系列ごとの既定 quant。実測で検収した席だけを既定にするので導出できず、宣言が要る
#: （ADR 0104）。`qat_plan` の中の分岐に書くと `qat_quants` のキーと二重管理になり、
#: 片方だけ書き換えると既定が存在しない quant を指すため、純データとして 1 箇所に置く。
QAT_DEFAULT_QUANT: Mapping[str, str] = {"e2b": "i4-fast", "e4b": "i4"}


def qat_quants(model: str) -> Mapping[str, Any]:
    """参照quantを保持し、検収済みE2Bに並列・融合の定義を足す（ADR 0104）。"""
    checkpoint_name(model)
    quant = {
        "weights": {"model": "i4"},
        "session": {},
        "label": "Fixed mixed int2/int4/int8 with SRQ",
        "description": "Official mobile QAT integers and scales, preserved without "
        "requantization; fixed activation rounding (SRQ).",
    }
    quants = {"i4": quant}
    if model == "e2b":
        quants["i4-gemvpar"] = {
            **quant,
            "session": {"linearGemvReduce": "parallel"},
            "label": "Fixed mixed QAT with parallel GEMV",
            "description": "The same fixed QAT weights and SRQ as i4, with parallel GEMV "
            "summation. Rounding and generated tokens can differ. "
            "Select i4 for the reference summation order.",
        }
        quants["i4-fast"] = {
            **quant,
            "session": {
                "linearGemvReduce": "parallel",
                "fuseRmsNormAdd": True,
                "fuseLinearStaticQuantize": True,
                "packedStaticQuantize": True,
            },
            "label": "Fixed mixed QAT with parallel GEMV and fusion",
            "description": "Same fixed QAT weights; parallel GEMV, RMS-add and linear-SRQ "
            "fusion, and packed int8 activations for E2B. Use i4 for reference summation "
            "or i4-gemvpar without fusion. Requires fusion-option support.",
        }
    return quants


#: 旧 sidecar 世代の記録が名乗る PLE の本数 → 容器の資産の block 本数（schema 3 の繰り上げ）。
REFERENCE_SHARDS_FIELD = "pleShards"
REFERENCE_BLOCKS_FIELD = "pleBlocks"


def assert_reference_generation(reference: Mapping[str, Any], where: Path) -> None:
    """検収記録が**今の世代**であること（版の不一致は欄の不在で綴る）。

    MUST: 旧世代を名指しで落とす — schema を据え置いたまま欄名だけ変えると、古い記録は版の
    門を素通りして「`pleBlocks`（= None）が現物と違う」でだけ落ち、実際に足りないのが
    **欄そのもの**であることがどこにも出ない。
    """
    schema = reference.get("schema")
    if schema == REFERENCE_SCHEMA:
        return
    if REFERENCE_SHARDS_FIELD in reference:
        raise DistError(
            f"{where}: 旧 sidecar 世代の検収記録（schema {schema} —"
            f" `{REFERENCE_SHARDS_FIELD}` を持ち `{REFERENCE_BLOCKS_FIELD}` を持たない）。"
            " PLE は容器の資産へ畳まれ、数える単位が shard から block へ変わった"
            " — 系列を移行（`migrate_series`）するか再 export する"
        )
    raise DistError(f"{where}: QAT reference の schema が {schema}（期待 {REFERENCE_SCHEMA}）")


def qat_plan(series_dir: Path, model: str) -> ModelPlan:
    """検査した1系列から配布計画を作る。元チェックポイントの再ダウンロードは不要。"""
    source = series_dir / series_name(model)
    reference = json.loads((source / "reference.json").read_text())
    assert_reference_generation(reference, source / "reference.json")
    if (
        reference.get("family") != "gemma4-qat"
        or reference.get("model") != model
        or reference.get("maxChunkLength") != MAX_CHUNK_LENGTH
        or reference.get("maxSelectedRows") != MAX_SELECTED_ROWS
    ):
        raise DistError("QAT reference の family/model/trace範囲が合わない")
    container = source / "model.krm"
    index = gemma4_ple_index(container, storage=f"i{PLE_BITS[model]}")
    for dtype in FIXED_STORAGE_DTYPES:
        assert_storage("model", container, {"model": dtype})
    assert_storage_absent("model", container, {"model": ("f16",)})
    graph = ir_graph(container)
    layouts = read_layouts(container)
    assert_qat_graph(graph, layouts)
    # 変換時の検証結果は「何本を照合したか」で突き合わせる（常に True のフラグは門にならない）。
    counts = fixed_storage_counts(graph, layouts)
    actual = {
        "fixedWeights": sum(counts.values()),
        "storageCounts": counts,
        REFERENCE_BLOCKS_FIELD: len(index["values"]["blocks"]),
    }
    differing = {
        field: (reference.get(field), value)
        for field, value in actual.items()
        if reference.get(field) != value
    }
    if differing:
        raise DistError(
            "QAT reference の固定重み本数・格納内訳・PLE block 本数が現物と違う: "
            + " / ".join(
                f"{field} 記録 {recorded} ≠ 現物 {value}"
                for field, (recorded, value) in sorted(differing.items())
            )
        )
    config = gemma4_text_config(source)
    where = str(source / "config.json")
    rope = gemma4_rope(config, where)
    vocab = gemma4_vocab_size(graph, container)
    assert_gemma4_graph(graph, container, index, rope, gemma4_hidden_size(config, where))
    if index["tokens"] != vocab:
        raise DistError("QAT PLE とグラフの語彙数が違う")
    assert_gemma4_ple_assets(container, index)
    assert_gemma4_tokenizer(source / "tokenizer.json", vocab)
    artifacts = {
        GEMMA4_ROLE: Artifact(f"{GEMMA4_ROLE}/model.i4.krm", source=container),
        "tokenizer": Artifact("tokenizer/tokenizer.json", source=source / "tokenizer.json"),
    }
    quant_modes = qat_quants(model)
    return ModelPlan(
        name=model,
        pipeline="gemma4-qat/1",
        artifacts=artifacts,
        # 部品名は通常 Gemma と同じ 1 語（書き手 `gemma4_qat/export.py` も同じ定数で
        # 容器のグラフを名乗る — container-v1 §2.1）。
        weights={GEMMA4_ROLE: {"i4": WeightFiles(GEMMA4_ROLE)}},
        assets=GEMMA4_ASSETS,
        quants=quant_modes,
        default_quant=QAT_DEFAULT_QUANT[model],
        # 3 式（chunkLength ≤ maxChunkLength / chunkLength ≤ capacity / capacity ≤ maxPosition）
        # の検査は通常 Gemma と同じ 1 実装を通す。trace 上限は provenance に記録した値を使う。
        pipeline_config=gemma4_pipeline_config(
            gemma4_max_position(config, where),
            rope,
            gemma4_sampler(source),
            chunk_length=DEFAULT_CHUNK_LENGTH,
            max_chunk_length=reference["maxChunkLength"],
            capacity=DEFAULT_CAPACITY,
        ),
    )


def repo_name(model: str) -> str:
    checkpoint_name(model)
    return "karume-gemma4-qat"


def render_card(manifest: Mapping[str, Any], repo: str, host_assets: Mapping[str, int] = {}) -> str:
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
        sections.extend([[f"## {name}", ""], quants(model, host_assets=host_assets)])
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
were moved to container assets the host reads, and rotary inputs are generated on the host.
Fixed quantized integers and scales were retained without requantization.
Vision, audio, and MTP are not included.
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
