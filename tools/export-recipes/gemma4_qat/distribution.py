"""固定 QAT の別ファミリ配布。グラフと PLE の共通門は Gemma から再利用する。"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from _shared.container_read import read_layouts
from _shared.licenses import apache_license_2_0
from _shared.upstream import hf_upstreams
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
from karume.modelcard import (
    CardMetadata,
    FileRefKey,
    frontmatter,
    introduction,
    models,
    quants,
    render,
    require_pipeline,
)

from .config import (
    CHECKPOINTS,
    DEFAULT_CAPACITY,
    DEFAULT_CHUNK_LENGTH,
    MAX_CHUNK_LENGTH,
    MAX_SELECTED_ROWS,
    PLE_BITS,
    REFERENCE_SCHEMA,
    SOURCES,
    checkpoint_name,
    checkpoint_repo,
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
#: E2B / E4B とも `i4-fast`（2026-09-26・Arc B570 の flag-bench で単体フラグが全て速度低下
#: なし → 全部入り — ADR 0104 追記 2026-09-26）。
QAT_DEFAULT_QUANT: Mapping[str, str] = {"e2b": "i4-fast", "e4b": "i4-fast"}

#: `i4-fast` の説明に足す実測の 1 文（GPU decode ms/step の `i4` 比）。E2B は従来どおり
#: 実測を説明に書かない。E4B は初めて席を持つので、既定にした根拠の数字を 1 つだけ添える
#: （表示欄の上限 200 字の内側）。
QAT_FAST_MEASURED_NOTE: Mapping[str, str] = {"e2b": "", "e4b": " E4B decode time -42% (B570)."}


def qat_quants(model: str) -> Mapping[str, Any]:
    """参照quantを保持し、同じ重みに並列・融合の定義を足す（ADR 0104）。

    `i4` は `session` を空に保つ = GEMV も attention も runtime の参照経路。`i4-gemvpar` は
    GEMV と attention の並列（融合なし）、`i4-fast` は単体で速度低下の無かったフラグの全部入りで、
    attention は `parallel` より GPU decode が速かった `"parallel-fused"`（ADR 0104 追記
    2026-09-26）。E2B と E4B は同じ 3 席・同じ宣言を持つ。
    """
    checkpoint_name(model)
    quant = {
        "weights": {"model": "i4"},
        "session": {},
        "label": "Fixed mixed int2/int4/int8 with SRQ",
        "description": "Official mobile QAT integers and scales, preserved without "
        "requantization; fixed activation rounding (SRQ). GEMV and attention use the "
        "reference summation order.",
    }
    return {
        "i4": quant,
        "i4-gemvpar": {
            **quant,
            "session": {"linearGemvReduce": "parallel", "stateAttentionReduce": "parallel"},
            "label": "Fixed mixed QAT with parallel GEMV and attention",
            "description": "The same fixed QAT weights and SRQ as i4, with parallel GEMV and "
            "attention summation. Rounding and generated tokens can differ. "
            "Select i4 for the reference summation order.",
        },
        "i4-fast": {
            **quant,
            "session": {
                "linearGemvReduce": "parallel",
                "fuseRmsNormAdd": True,
                "fuseLinearStaticQuantize": True,
                "fuseGeluMulStaticQuantize": True,
                "packedStaticQuantize": True,
                "stateAttentionReduce": "parallel-fused",
            },
            "label": "Fixed mixed QAT with parallel GEMV and fusion",
            "description": "Same QAT weights; parallel GEMV, fused parallel attention, RMS-add "
            "and linear-SRQ fusion, packed int8 activations. Use i4 for reference sums. "
            "Needs fusion-option support." + QAT_FAST_MEASURED_NOTE[model],
        },
    }


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


def assert_pinned_source(reference: Mapping[str, Any], model: str, where: Path) -> None:
    """The series was exported from the pinned upstream commits ({@link SOURCES} — the weights'
    and the tokenizer's).

    MUST: compare the checkpoint fingerprint `export.py` recorded with both pins. The card names
    the pinned commits as the sources; without this check, a re-export from a newer upstream
    commit (or a tokenizer from another repository) would be published under the old commit's
    name and licence review.
    """
    pins = SOURCES[model].pins
    checkpoint = reference.get("checkpoint")
    files = checkpoint.get("files") if isinstance(checkpoint, Mapping) else None
    recorded = files if isinstance(files, Mapping) else {}
    differing = sorted(
        name
        for pin in pins.by_role().values()
        for name, sha256 in pin.sha256.items()
        if not isinstance(recorded.get(name), Mapping) or recorded[name].get("sha256") != sha256
    )
    if differing:
        raise DistError(
            f"{where}: the series was not exported from {pins.label} (checkpoint SHA-256 differs"
            f" for {', '.join(differing)}) — re-export from the pinned revisions, or move the pin"
            " in `gemma4_qat/config.py` after a new licence review"
        )


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
    assert_pinned_source(reference, model, source / "reference.json")
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
    """The mobile QAT line's one repository (renamed from `karume-gemma4-qat` on 2026-11-04 —
    that name holds the q4_0 QAT line, `gemma4.distribution.QAT_Q4_0_PIPELINE`)."""
    checkpoint_name(model)
    return "karume-gemma4-qat-mobile"


#: The SPDX identifier of every pinned upstream ({@link SOURCES} — the weights' and the tokenizer's
#: repositories) and how the card spells it. A pin with another licence fails the render: the
#: card's attribution sentence would no longer be true.
QAT_LICENSE = "apache-2.0"
QAT_LICENSE_NAME = "Apache 2.0"
#: The `license_link` of the upstream cards at the pinned commits (an Apache 2.0 page on
#: ai.google.dev — the mobile repositories checked 2026-11-01, the regular `-it` ones 2026-11-04),
#: and the canonical text that `LICENSE.md` copies verbatim.
QAT_LICENSE_LINK = "https://ai.google.dev/gemma/docs/gemma_4_license"
QAT_LICENSE_TEXT_LINK = "https://www.apache.org/licenses/LICENSE-2.0"
#: The technical report the upstream cards link at the pinned commits.
QAT_TECHNICAL_REPORT = "https://arxiv.org/abs/2607.02770"


def _qat_attribution(manifest: Mapping[str, Any]) -> list[str]:
    """The attribution section: each model's upstreams at their pinned commits (the weights' and
    the tokenizer's), the changes made, and a pointer to the upstream cards for what the model can
    and cannot do."""
    lines = [
        "## Base weights and attribution",
        "",
        "Converted from the upstream checkpoints — the original checkpoints are not distributed"
        " here.",
        "",
    ]
    for name in manifest["models"]:
        pins = SOURCES[name].pins
        weights, tokenizer = pins.weights, pins.tokenizer
        lines.append(
            f"- **`{name}`**: the weights are [{weights.repo}](https://huggingface.co/{weights.repo})"
            f" at commit `{weights.revision}`; the tokenizer is that of its base model"
            f" [{tokenizer.repo}](https://huggingface.co/{tokenizer.repo}) at commit"
            f" `{tokenizer.revision}` (Google's glitch-token fixes of 2026-11-03, which the mobile"
            " repository has not received); both are licensed"
            f" **{QAT_LICENSE_NAME}** at those commits ([license]({QAT_LICENSE_LINK}) /"
            f" [full text]({QAT_LICENSE_TEXT_LINK}); a verbatim copy is in `LICENSE.md`)."
        )
    lines += [
        "- **Authors**: Gemma 4 and its mobile QAT checkpoints are released by Google DeepMind"
        f" ([technical report]({QAT_TECHNICAL_REPORT})).",
        "- **Changes made here** (listed in full in `NOTICE.md`, per Apache 2.0 §4(b)): the text",
        "  decoder was extracted and re-expressed in the Karume container format; the fixed",
        "  quantized integers, their scales and the static activation ranges were kept without",
        "  requantization; the per-layer embeddings were moved out of the graph into container",
        "  assets that the host gathers; the exit was narrowed to the selected rows' logits and",
        "  hidden states; rotary cos/sin arrive as host-generated inputs. Vision, audio and MTP",
        "  are not included. No retraining and no fine-tuning.",
        "- **What the model can and cannot do** — capabilities, limitations, intended use,",
        "  evaluations and safety guidance — is documented on the upstream model cards linked",
        "  above, and this repository does not restate it: read them before using it.",
    ]
    return lines


def render_card(
    manifest: Mapping[str, Any], repo: str, host_assets: Mapping[FileRefKey, int] = {}
) -> str:
    """数値・モデル一覧・量子化表を manifest から描く。固定値保持を再量子化と呼ばない。"""
    require_pipeline(manifest, "gemma4-qat/1")
    # Every redistributed upstream: per model its base model (the regular `-it` repository its
    # tokenizer comes from) and then its mobile QAT checkpoint, the order the q4_0 QAT line's card
    # lists them in (`gemma4.card`). `checkpoint_repo` refuses a model the line does not know.
    upstream: list[str] = []
    for model in manifest["models"]:
        checkpoint = checkpoint_repo(model)
        upstream += [SOURCES[model].pins.tokenizer.repo, checkpoint]
    for model in manifest["models"]:
        if SOURCES[model].license != QAT_LICENSE:
            raise ValueError(
                f"the pin of '{model}' declares {SOURCES[model].license}, but the card attributes"
                f" {QAT_LICENSE}"
            )
    metadata = CardMetadata(
        pipeline_tag="text-generation",
        base_model=tuple(upstream),
        # Against the listed `-it` base models the stored integers and scales are a quantized form
        # (the checkpoint's own QAT values, kept as they are) — the relation the q4_0 QAT line's
        # card declares too, and the upstream mobile cards declare against the same base models.
        base_model_relation="quantized",
        license=QAT_LICENSE,
        # The same spelling as the upstream cards (and the regular Gemma 4 card), so a reader
        # reaches the one licence page the upstream names.
        license_link=QAT_LICENSE_LINK,
        tags=("webgpu", "gemma", "text-generation"),
    )
    sections = [
        frontmatter(metadata),
        introduction(
            "Gemma 4 Mobile QAT Text Decoder — Karume", hf_upstreams(metadata.base_model)
        ),
        [
            "Experimental text-only conversion of the official mobile QAT checkpoints.",
            "Fixed integers and scales are preserved; activations use "
            "static range quantization (SRQ).",
            "CPU and GPU reductions can cross rounding boundaries and select different tokens.",
            "Broad quality and long-context validation remain incomplete. MTP is not supported.",
            "",
        ],
        _qat_attribution(manifest),
        [""],
        models(manifest),
    ]
    for name, model in manifest["models"].items():
        sections.extend([[f"## {name}", ""], quants(model, host_assets=host_assets)])
    return render(sections)


#: The notice (Apache 2.0 §4(b)): the upstream checkpoints and **the changes this repository made**.
#: Upstream has no NOTICE file and no copyright line (no LICENSE file either — the cards' front
#: matter and `license_link` carry the licence), so nothing is to be retained under §4(c)/(d).
#:
#: MUST: the text matches the distribution's content — it is valid prose as a value, so neither
#: `verify_dist` nor the manifest check would notice a mismatch before it is published.
NOTICE = (
    """# NOTICE

This repository redistributes modified forms of the Google Gemma 4 mobile QAT checkpoints and of
the tokenizers of their base models listed below (Google DeepMind), which are licensed under the
Apache License, Version 2.0 (see `LICENSE.md`). The models actually included are identified in
`README.md` and `karume.json`; `README.md` names the upstream commits each one was converted from.

"""
    + "\n".join(
        f"- `{checkpoint_repo(model)}` and the tokenizer of its base model"
        f" `{SOURCES[model].pins.tokenizer.repo}`"
        for model in CHECKPOINTS
    )
    + """

The following changes were made:

- The **text decoder only** was converted; the vision and audio towers are not included, the
  checkpoint's key/value-cache scales are not used, and no multi-token-prediction (MTP) drafter is
  included.
- The graph was re-expressed in the Karume container format (a `.krm` part sequence whose first
  part carries the graph and model descriptors) in a states form suited to chunked prefill and
  decode.
- **The fixed quantized integers and their scales were kept as they are**: the int2 / int4 / int8
  weights of the quantized linear layers, of the token embedding (shared with the output head)
  and of the per-layer embeddings were not requantized, and the static activation ranges (SRQ)
  around the quantized linear layers keep the checkpoint's scales. The row scales of int4 weights
  are repeated once per group of the storage layout (the same values). The output head's activation
  rounding, whose scales are 0 in the checkpoint (uncalibrated, the identity), is left out.
- The parameters the checkpoint does not quantize (the per-layer model projection and the norms,
  among others) are stored in float32, widened exactly from the checkpoint's values.
- The **per-layer embedding tables were moved out of the graph** into container assets that the
  host gathers, and the exit was narrowed to the selected rows' logits and final hidden states.
- **Rotary position embeddings were moved out of the graph**: the cosine and sine rows are built
  by the host from the declared parameters and passed in as ordinary graph inputs.

No retraining and no fine-tuning were performed. The original checkpoints are not distributed here.
"""
)

PIPELINE = Pipeline(
    default_model="e2b",
    repo_name=repo_name,
    plan=qat_plan,
    card_profiles={"gemma4-qat-mobile": render_card},
    root_files={"LICENSE.md": apache_license_2_0(), "NOTICE.md": NOTICE},
)
