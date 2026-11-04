"""The separate-family distribution of the fixed QAT. The shared graph and PLE checks are reused
from Gemma."""

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
    """Do not let the regular Gemma, or a QAT that lost its SRQ, through as this distribution.

    `layouts` is tensor key → storage layout ({@link _shared.container_read.read_layouts}) — the
    IR v2 graph description has no storage (the source of truth is the binding table), so the
    fixed-storage decision is taken from there. In IR v2 an initializer's name is the tensor key
    itself, so both are looked up by the same key.
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
        raise DistError("the QAT graph does not have exactly one token embedding")
    token_weight = embeddings[0]["ins"][0]
    if token_weight not in initializers or layouts.get(token_weight) != "i2":
        raise DistError("the QAT token embedding must be fixed i2")
    head_count = 0
    ordinary_count = 0
    for node in nodes:
        if node["op"] != "linear":
            continue
        key = node["ins"][1]
        dtype = layouts.get(key, "f32") if key in initializers else "f32"
        if dtype == "f32":
            if key != "model.model.per_layer_model_projection.weight":
                raise DistError(
                    "the QAT graph has a linear without fixed quantization other than the allowed"
                    " projection"
                )
            ordinary_count += 1
            continue
        if dtype not in ("i2", "i4", "i8"):
            raise DistError(f"unsupported fixed storage of a QAT linear: {dtype}")
        # Only the shared head may omit the SRQ on either side — upstream lm_head's SRQ scales are
        # 0 for both input and output (uncalibrated = identity), and the recipe does not put an
        # SRQ with scale=0 into the IR (`TraceLinear` in `checkpoint.py`). Every other quantized
        # linear has positive scales, so both sides stay required.
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
            raise DistError("a QAT linear needs a fixed SRQ before and after it")
    if head_count != 1 or ordinary_count != 1:
        raise DistError("wrong number of QAT shared heads or unquantized projections")


#: The fixed-storage dtypes (also the keys of `storageCounts` in `reference.json`).
FIXED_STORAGE_DTYPES: tuple[str, ...] = ("i2", "i4", "i8")


def fixed_storage_counts(graph: Mapping[str, Any], layouts: Mapping[str, str]) -> dict[str, int]:
    """Count the binding table's initializers per fixed-storage dtype (what `reference.json` is
    compared against)."""
    counts = dict.fromkeys(FIXED_STORAGE_DTYPES, 0)
    for key in graph["initializers"]:
        dtype = layouts.get(key)
        if dtype in counts:
            counts[dtype] += 1
    return counts


#: The default quant per series. Only presets accepted by measurement become the default, so it
#: cannot be derived and must be declared (ADR 0104). Written as a branch inside `qat_plan` it
#: would be maintained twice alongside the keys of `qat_quants`, and changing only one would make
#: the default point at a quant that does not exist, so it lives in one place as plain data.
#: Both E2B / E4B use `i4-fast` (2026-09-26; on the Arc B570 flag-bench no single flag slowed
#: down → all of them — ADR 0104 addendum 2026-09-26).
QAT_DEFAULT_QUANT: Mapping[str, str] = {"e2b": "i4-fast", "e4b": "i4-fast"}

#: The one measured sentence added to the description of `i4-fast` (GPU decode ms/step relative
#: to `i4`). E2B, as before, puts no measurement in its description. E4B has the preset for the
#: first time, so a single number backing the default is added (within the 200-character limit
#: of the display field).
QAT_FAST_MEASURED_NOTE: Mapping[str, str] = {"e2b": "", "e4b": " E4B decode time -42% (B570)."}


def qat_quants(model: str) -> Mapping[str, Any]:
    """Keep the reference quant and add parallel / fused definitions over the same weights (ADR
    0104).

    `i4` keeps `session` empty = both GEMV and attention take the runtime's reference path.
    `i4-gemvpar` parallelizes GEMV and attention (no fusion); `i4-fast` has every flag that
    showed no slowdown on its own, with attention `"parallel-fused"`, whose GPU decode was faster
    than `parallel` (ADR 0104 addendum 2026-09-26). E2B and E4B have the same three presets with
    the same declarations.
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


#: The PLE count a record of the old sidecar generation claims → the block count of the
#: container's asset (the schema 3 promotion).
REFERENCE_SHARDS_FIELD = "pleShards"
REFERENCE_BLOCKS_FIELD = "pleBlocks"


def assert_reference_generation(reference: Mapping[str, Any], where: Path) -> None:
    """The acceptance record is of **the current generation** (a version mismatch is spelled
    out as the missing field).

    MUST: name the old generation when failing — if only the field name changed while the schema
    stayed, an old record would pass the version check and fail only as "`pleBlocks` (= None)
    differs from the actual one", and nowhere would it say that what is really missing is **the
    field itself**.
    """
    schema = reference.get("schema")
    if schema == REFERENCE_SCHEMA:
        return
    if REFERENCE_SHARDS_FIELD in reference:
        raise DistError(
            f"{where}: an acceptance record of the old sidecar generation (schema {schema} —"
            f" it has `{REFERENCE_SHARDS_FIELD}` and no `{REFERENCE_BLOCKS_FIELD}`)."
            " PLE is now folded into the container's asset and the counting unit changed from"
            " shard to block — migrate the series (`migrate_series`) or export it again"
        )
    raise DistError(
        f"{where}: the QAT reference's schema is {schema} (expected {REFERENCE_SCHEMA})"
    )


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
    """Build the distribution plan from one checked series. The original checkpoint need not be
    downloaded again."""
    source = series_dir / series_name(model)
    reference = json.loads((source / "reference.json").read_text())
    assert_reference_generation(reference, source / "reference.json")
    if (
        reference.get("family") != "gemma4-qat"
        or reference.get("model") != model
        or reference.get("maxChunkLength") != MAX_CHUNK_LENGTH
        or reference.get("maxSelectedRows") != MAX_SELECTED_ROWS
    ):
        raise DistError("the QAT reference's family/model/trace range does not match")
    assert_pinned_source(reference, model, source / "reference.json")
    container = source / "model.krm"
    index = gemma4_ple_index(container, storage=f"i{PLE_BITS[model]}")
    for dtype in FIXED_STORAGE_DTYPES:
        assert_storage("model", container, {"model": dtype})
    assert_storage_absent("model", container, {"model": ("f16",)})
    graph = ir_graph(container)
    layouts = read_layouts(container)
    assert_qat_graph(graph, layouts)
    # The conversion-time verification is compared by "how many were checked" (a flag that is
    # always True is no check).
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
            "the QAT reference's fixed-weight count / storage breakdown / PLE block count differ"
            " from the actual ones: "
            + " / ".join(
                f"{field} recorded {recorded} ≠ actual {value}"
                for field, (recorded, value) in sorted(differing.items())
            )
        )
    config = gemma4_text_config(source)
    where = str(source / "config.json")
    rope = gemma4_rope(config, where)
    vocab = gemma4_vocab_size(graph, container)
    assert_gemma4_graph(graph, container, index, rope, gemma4_hidden_size(config, where))
    if index["tokens"] != vocab:
        raise DistError("the vocabulary sizes of the QAT PLE and the graph differ")
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
        # The component name is the same single word as the regular Gemma's (the writer
        # `gemma4_qat/export.py` names the container's graph with the same constant —
        # container-v1 §2.1).
        weights={GEMMA4_ROLE: {"i4": WeightFiles(GEMMA4_ROLE)}},
        assets=GEMMA4_ASSETS,
        quants=quant_modes,
        default_quant=QAT_DEFAULT_QUANT[model],
        # The three inequalities (chunkLength ≤ maxChunkLength / chunkLength ≤ capacity /
        # capacity ≤ maxPosition) are checked by the same single implementation as the regular
        # Gemma's. The trace bound is the value recorded in the provenance.
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
    """Draw the numbers, the model list and the quantization table from the manifest. Keeping the
    fixed values is never called requantization."""
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
        introduction("Gemma 4 Mobile QAT Text Decoder — Karume", hf_upstreams(metadata.base_model)),
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
