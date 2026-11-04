"""The distribution recipe of the umT5-XXL encoder — the distribution repository `karume-umt5-xxl`
(ADR 0119 addendum "design of step 10d" D).

It distributes one graph (`text_encoder` — the i8 dynamic-shape container of the series
`umt5-xxl-i8-dyn`) and the asset `tokenizer`, the original tokenizer written into one table (series
`umt5-xxl-tokenizer` — format `karume-umt5-tokenizer/1`, written by `wan.umt5_xxl_tokenizer`).
Since it carries the tokenizer, it can be used on its own without Wan (`Encoder` of TS
`@karume/models/umt5`). The tokenizer asset bundled with Wan's preprocessing tables
(`karume-wan-umt5-tokenizer/1`) stays an asset of the Wan repository as before (ibid. C). The Wan
distribution `karume-wan2.2` references this repository's container across repositories (ibid. A —
ADR 0109 decision 3; Wan2.1 is not published).

The provenance is the encoder of the original `google/umt5-xxl` (`shared.weight` and `encoder.*` —
the provenance the container declares is the pin of `wan.sources.UMT5_SOURCES`, ADR 0122
decision 1). Every tensor is bit-identical in f32 to the `text_encoder` of Wan2.1 Diffusers, and the
bf16 of Wan2.2 TI2V-5B Diffusers is its RNE rounding (measured 2026-10-04), so the card and the
NOTICE say so.

The public surface is the single {@link PIPELINE} — the repository's dist driver
(`tools/export-recipes/dist.py`) merges it into core's PIPELINES (`--pipeline umt5`).

The intake of a third-party compatible encoder (`wan.umt5_intake` — ADR 0122 decisions 5 / 6) is a
**local experimental mirror** that {@link intake_pipeline} builds only when the driver receives
`--intake`, and it does not call itself `karume-umt5-xxl` (repository name
{@link intake_repo_name}; the default output is outside `models/`). The provenance check compares
the record `intake.json` with the container's provenance, and the root files (LICENSE / NOTICE)
and the card are derived from the record — the original's Apache `LICENSE.md` and its
"No retraining" NOTICE are not attached. An intake with an undeclared license fails at planning time
(before a single byte is written) unless it is explicitly allowed (`--allow-undeclared-license`).

MUST: this module does not import torch (`import dist` does not load torch —
`TestImportingTheDriver` in `tests/test_dist_driver.py`). The writer (`wan.umt5_export`) loads
torch, so the spellings (series name, component name, input names, the attribute name of the
relative-position tables) live here, and the check in `wan/tests/test_umt5_distribution.py` checks
they match the writer's spellings. The writer also takes the weight-kind classification
({@link storage_kind}) from here (there are not two classifications).
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable, Mapping
from functools import partial
from pathlib import Path
from typing import Any

from _shared.container_read import ContainerReadError, read_layouts
from _shared.licenses import apache_license_2_0, cc_by_sa_4_0
from _shared.upstream import assert_upstream_provenance, hf_upstreams
from karume.dist import (
    LICENSE_FILENAME,
    NOTICE_FILENAME,
    Artifact,
    DistError,
    ModelPlan,
    Pipeline,
    WeightFiles,
    assert_component_present,
    assert_model_name,
    complete_quant_weights,
    ir_graph,
)
from karume.modelcard import (
    HF_OWNER,
    REVISION_PIN_COMMENT,
    CardMetadata,
    FileRefKey,
    Upstream,
    frontmatter,
    introduction,
    model_sections,
    models,
    quants,
    render,
    require_pipeline,
)
from wan import umt5_tokenizer, umt5_xxl_tokenizer
from wan.sources import UMT5_SOURCES, Umt5Source
from wan.umt5_intake import (
    UNDECLARED_LICENSE,
    Umt5Intake,
    Umt5IntakeError,
    assert_license_intent,
)

#: The pipeline contract (ADR 0041 §2 — per model). The TS reader is `umt5.Encoder` of
#: `@karume/models/umt5` (as of ADR 0119 addendum D it only named the component's role). Wan's
#: reader does not read the `karume.json` at the target of the cross-repository reference.
UMT5_PIPELINE = "umt5-encoder/1"

#: The distribution repository name (`karume-<family>-<variant>` of ADR 0092 decision 2).
UMT5_REPO_NAME = "karume-umt5-xxl"

#: The model name (the variant axis inside the repository — like `v4-small` of
#: `karume-irodori-v4-small`, the repository name without `karume-<family>-`). The path of Wan's
#: cross-repository reference points into this subtree (`xxl/text_encoder/…`).
UMT5_DEFAULT_MODEL = "xxl"

#: The series (the writer spells it `wan.umt5_export.SERIES_NAME`) and the representative name of
#: the container in it (`wan.umt5_export.MODEL_FILE` — the split form is
#: `model-0000N-of-0000M.krm`).
UMT5_SERIES = "umt5-xxl-i8-dyn"
UMT5_MODEL_FILE = "model.krm"

#: Component name = the weights key of the manifest = **the container's graph name**
#: (container-v1 §2.1) = the series' component directory name. The writer spells it
#: `wan.umt5_export.GRAPH_NAME` / `COMPONENT_DIR`. The Wan distribution references it across
#: repositories with the same key (`wan.distribution.WAN_TEXT_ENCODER_ROLE`).
UMT5_ROLE = "text_encoder"

#: Role name of the tokenizer asset = the key of the manifest's `assets` (the name TS
#: `umt5.Encoder` reads).
UMT5_TOKENIZER_ROLE = "tokenizer"

#: Output relative paths (**inside the model subtree**). The storage label goes into the path for
#: the same reason as Wan's DiT (adding a storage preset later does not move existing paths).
#:
#: MUST: the same spelling as the path of the same component in the Wan distribution
#: (`wan.distribution.WAN_OUTPUT_PATHS` takes it from here) — the cross-repository reference
#: resolves `<model name>/<this path>` as declared by the referenced `karume.json`, so a mismatch
#: makes the Wan assembly fail with "not in the referenced repository".
UMT5_OUTPUT_PATHS: Mapping[str, str] = {
    UMT5_ROLE: f"{UMT5_ROLE}/model.i8.krm",
    # Assets are `<role>/<file name>` (the same spelling rule as Wan's
    # `umt5_tokenizer/tokenizer.json`).
    UMT5_TOKENIZER_ROLE: f"{UMT5_TOKENIZER_ROLE}/{umt5_xxl_tokenizer.ASSET_FILE}",
}

#: The weights declaration (component → dtype label → placement role). A single i8 preset (ADR 0119
#: decision 5 — i4 is added as a separate preset once its quality is measured).
UMT5_WEIGHTS: Mapping[str, Mapping[str, WeightFiles]] = {UMT5_ROLE: {"i8": WeightFiles(UMT5_ROLE)}}

#: The assets declaration (unconditional files independent of the quant choice — ADR 0041 §3). One
#: table of the original tokenizer (no preprocessing tables — Wan's preprocessing lives in the Wan
#: repository's asset).
UMT5_ASSETS: Mapping[str, str] = {UMT5_TOKENIZER_ROLE: UMT5_TOKENIZER_ROLE}

#: Top-level fields of the tokenizer asset (the fields of the writer
#: `wan.umt5_xxl_tokenizer.build_asset` — an asset with Wan-specific fields such as the
#: preprocessing table `promptClean` fails).
UMT5_TOKENIZER_FIELDS: frozenset[str] = frozenset(
    {
        "format",
        "source",
        "versions",
        "maxLength",
        "unkId",
        "eosId",
        "addedTokens",
        "space",
        "vocabText",
        "scores",
    }
)

#: The quant table (one preset). `label` / `description` are display fields for the selection UI
#: (ADR 0075 decision 1 — English).
UMT5_QUANTS: Mapping[str, Any] = {
    "i8": {
        "weights": {},
        "session": {},
        "label": "int8 weights",
        "description": "Linear and vocabulary-embedding weights stored as int8 (one scale per"
        " output channel or row) and computed in f32; norms and relative-position tables stay f32.",
    }
}
UMT5_DEFAULT_QUANT = "i8"

#: There is no pipeline-owned configuration (no reader). `karume/5` states `{}` explicitly even when
#: empty.
UMT5_PIPELINE_CONFIG: Mapping[str, Any] = {}

#: Graph inputs (the writer spells them `wan.umt5_patch.INPUT_NAMES` / `SYMBOL`) — token ids
#: `[1, L]` and the relative-position bucket index table `[L, L]`, both i32 (ADR 0119 decisions
#: 3 / 4).
UMT5_IDS_INPUT = "input_ids"
UMT5_BUCKETS_INPUT = "relative_position_buckets"
UMT5_SYMBOL = "L"
UMT5_INPUTS: tuple[tuple[str, str, list[Any]], ...] = (
    (UMT5_IDS_INPUT, "i32", [1, UMT5_SYMBOL]),
    (UMT5_BUCKETS_INPUT, "i32", [UMT5_SYMBOL, UMT5_SYMBOL]),
)

#: Attribute name of the relative-position tables (the writer spells it
#: `wan.umt5_patch.RELATIVE_BIAS_ATTRIBUTE`).
UMT5_RELATIVE_BIAS = "relative_attention_bias"

#: Tensor key of the token embedding (tied — `shared.weight` in the checkpoint).
UMT5_EMBED_KEY = "encoder.embed_tokens.weight"

#: Weight kind → the required storage layout (ADR 0119 decision 5, addendum on the step 10c
#: preparation). `constant` is a constant the exporter lifted (`const.` — a zero linear bias and
#: the like).
#:
#: MUST: the relative-position tables are pinned to f32 — the exporter's default i8 also rounds
#: `nn.Embedding` (`QUANT_CHANNEL_AXES`), so if the writer failed to exclude the tables, loading
#: and running would still pass and only the attention bias would be silently rounded to an 8-bit
#: grid (in the step 10c preparation measurements, giving only one of the two instructions
#: failed).
UMT5_KIND_LAYOUTS: Mapping[str, str] = {
    "embed_tokens": "i8",
    "linear": "i8",
    "norm": "f32",
    UMT5_RELATIVE_BIAS: "f32",
    "constant": "f32",
}

#: The weight kinds that always appear in the binding table (`constant` depends on whether anything
#: was lifted, so it is not required).
UMT5_WEIGHT_KINDS: tuple[str, ...] = ("embed_tokens", "linear", "norm", UMT5_RELATIVE_BIAS)


def storage_kind(key: str) -> str:
    """The kind of a tensor key (relative-position table, token embedding, norm, linear,
    constant).

    The writer's (`wan.umt5_export`) acceptance table and the assembly's binding-table check
    ({@link assert_umt5_bindings}) use the same classification.
    """
    if key.endswith(f".{UMT5_RELATIVE_BIAS}.weight"):
        return UMT5_RELATIVE_BIAS
    if key == UMT5_EMBED_KEY:
        return "embed_tokens"
    if key.endswith("layer_norm.weight"):
        return "norm"
    if key.endswith(".weight"):
        return "linear"
    return "constant"


def umt5_bindings(container: Path) -> dict[str, dict[str, int]]:
    """Count the binding table by kind × storage layout (**declarations only** — no weight payload
    is read)."""
    try:
        layouts = read_layouts(container)
    except ContainerReadError as cause:
        raise DistError(f"{container}: {cause}") from cause
    counts: dict[str, Counter[str]] = {}
    for key, layout in layouts.items():
        counts.setdefault(storage_kind(key), Counter())[layout] += 1
    return {kind: dict(sorted(counter.items())) for kind, counter in sorted(counts.items())}


def assert_umt5_bindings(container: Path) -> None:
    """Check that the binding table's storage meets the per-kind requirements
    ({@link UMT5_KIND_LAYOUTS}).

    MUST: fail at assembly. The container's binding table accepts anything that passes the
    structural checks of `verify_container`, and the i8 preset's requirement alone ("contains i8")
    lets through both a container whose tables or norms were rounded to i8 and one whose linears
    stayed f32 — both load and run, and only the output values quietly change.
    """
    bindings = umt5_bindings(container)
    missing = [kind for kind in UMT5_WEIGHT_KINDS if kind not in bindings]
    if missing:
        raise DistError(
            f"{container}: the binding table has no weight kinds {missing} (actual: {bindings}) —"
            " not a umT5 encoder container"
        )
    wrong = {
        kind: sorted(layouts)
        for kind, layouts in bindings.items()
        if set(layouts) != {UMT5_KIND_LAYOUTS[kind]}
    }
    if wrong:
        expected = {kind: UMT5_KIND_LAYOUTS[kind] for kind in wrong}
        raise DistError(
            f"{container}: the binding table's storage differs from the per-kind requirements"
            f" (actual {wrong} / required {expected}) — linear and token embedding are i8, RMSNorm"
            " and the relative-position tables f32 (ADR 0119 decision 5)"
        )


def umt5_context_width(container: Path) -> int:
    """Check the graph's input / output contract and return the output width `W` (`[1, L, W]`).

    The inputs are {@link UMT5_INPUTS} (names, dtypes and shapes matching in this order), the only
    symbol is `L`, and the output is a single f32 `[1, L, W]`.

    MUST: fail at assembly. The reader (Wan's text stage) binds the inputs by name and packs the
    output rows into the DiT's context, so a container with different input names, symbols or
    output shape would fail only on the user's machine.
    """
    graph = ir_graph(container)
    inputs = graph.get("inputs")
    declared = (
        [
            (item.get("name"), item.get("dtype"), item.get("shape"))
            for item in inputs
            if isinstance(item, dict)
        ]
        if isinstance(inputs, list)
        else inputs
    )
    if declared != list(UMT5_INPUTS):
        raise DistError(
            f"{container}: the graph inputs {declared!r} are not {list(UMT5_INPUTS)}"
            " — the reader binds inputs by name (token ids [1, L] and bucket table [L, L], i32)"
        )
    if graph.get("symbols") != [UMT5_SYMBOL]:
        raise DistError(
            f"{container}: the symbolic dimensions {graph.get('symbols')!r} are not"
            f" ['{UMT5_SYMBOL}'] — only the valid length L varies (ADR 0119 decision 4)"
        )
    outputs = graph.get("outputs")
    values = graph.get("values")
    value = (
        values.get(outputs[0])
        if isinstance(outputs, list) and len(outputs) == 1 and isinstance(values, dict)
        else None
    )
    shape = value.get("shape") if isinstance(value, dict) else None
    if (
        not isinstance(value, dict)
        or value.get("dtype") != "f32"
        or not isinstance(shape, list)
        or len(shape) != 3
        or shape[:2] != [1, UMT5_SYMBOL]
        or not isinstance(shape[2], int)
        or isinstance(shape[2], bool)
        or shape[2] <= 0
    ):
        raise DistError(
            f"{container}: the graph outputs {outputs!r} ({value!r}) are not a single f32"
            f" [1, {UMT5_SYMBOL}, W]"
        )
    return shape[2]


def assert_umt5_encoder(container: Path, model: str, intake: Umt5Intake | None = None) -> int:
    """Apply every check of the umT5 container (provenance, binding table, input / output contract)
    and return the output width.

    The Wan distribution has the same container as a component referenced across repositories, so
    both plans go through this one function (the Wan plan also checks the i8 preset separately
    through its storage requirement table). The provenance is checked against the original's row
    (`wan.sources.UMT5_SOURCES`) or the intake record `intake` (a model not in the original's rows —
    ADR 0122 decision 5).
    """
    if intake is None:
        upstream = UMT5_SOURCES.get(model)
        if upstream is None:
            raise DistError(
                f"umT5 model {model!r} is unknown (known: {' / '.join(sorted(UMT5_SOURCES))}) —"
                " only models listed in the upstream provenance table (wan.sources.UMT5_SOURCES)"
                " are distributed (a third-party intake can only be built into a local"
                " experimental mirror with --intake)"
            )
        license_id, revision = upstream.source.license, upstream.source.revision
    else:
        if model != intake.name:
            raise DistError(f"model {model!r} differs from the intake name {intake.name!r}")
        license_id, revision = intake.license, intake.revision
    assert_component_present(container)
    # Check the provenance the container declares against the upstream pin (the original's table or
    # the intake record is the source of truth) — the binding table and the input / output shapes
    # also pass for another checkpoint with the same structure, so only the provenance closes this.
    assert_upstream_provenance(container, license=license_id, revision=revision)
    assert_umt5_bindings(container)
    return umt5_context_width(container)


def umt5_container(series_dir: Path) -> Path:
    """The container's representative path, from the series' parent directory
    (`outputs/series/`)."""
    return series_dir / UMT5_SERIES / UMT5_ROLE / UMT5_MODEL_FILE


def umt5_tokenizer_asset(series_dir: Path) -> Path:
    """The tokenizer asset's path, from the series' parent directory (`outputs/series/`)."""
    return series_dir / umt5_xxl_tokenizer.SERIES_NAME / umt5_xxl_tokenizer.ASSET_FILE


def assert_umt5_tokenizer(path: Path, model: str) -> None:
    """Check that the tokenizer asset agrees with the format TS reads, the upstream pin, the version
    that is the source of truth for the id sequences, and the accepted upper bound.

    MUST: fail at assembly. The asset is a table, so the container checks of `verify_dist` do not
    apply, and the reader only looks at the format and shape — a tokenizer from another revision,
    or a table written with another transformers version, would stay silent on the user's machine as
    "different id sequences". The set of fields is checked too (an asset that mixes in fields such
    as Wan's preprocessing table `promptClean` is not distributed even if it declares this
    format).

    Five things are checked: the format version (`wan.umt5_xxl_tokenizer.ASSET_FORMAT`), the set
    of fields ({@link UMT5_TOKENIZER_FIELDS}), the provenance (repo / revision / sha256 of the
    files) matching the row of `wan.sources.UMT5_SOURCES`, the transformers version that is the
    source of truth for the id sequences matching the pin
    (`wan.umt5_tokenizer.TRANSFORMERS_VERSION`), and `maxLength` matching the accepted upper bound
    ({@link UMT5_TOKENS}).
    """
    assert_component_present(path)
    try:
        asset = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as cause:
        raise DistError(f"{path}: the tokenizer asset cannot be read as JSON — {cause}") from cause
    if not isinstance(asset, dict):
        raise DistError(f"{path}: the tokenizer asset is not an object")
    if asset.get("format") != umt5_xxl_tokenizer.ASSET_FORMAT:
        raise DistError(
            f"{path}: format {asset.get('format')!r} is not {umt5_xxl_tokenizer.ASSET_FORMAT!r}"
            " — only the original tokenizer's table is distributed"
            " (`python -m wan.umt5_xxl_tokenizer`)"
        )
    if set(asset) != UMT5_TOKENIZER_FIELDS:
        raise DistError(
            f"{path}: the fields {sorted(asset)} are not {sorted(UMT5_TOKENIZER_FIELDS)} — an"
            " asset with fields the standalone tokenizer lacks (such as preprocessing tables) is"
            " not distributed"
        )
    row = UMT5_SOURCES.get(model)
    if row is None:
        raise DistError(
            f"umT5 model {model!r} is not in the upstream table (known: {sorted(UMT5_SOURCES)})"
        )
    expected_source = {
        "repo": row.source.repo,
        "revision": row.source.revision,
        "files": dict(sorted(row.tokenizer.items())),
    }
    if asset["source"] != expected_source:
        raise DistError(
            f"{path}: the tokenizer provenance {asset['source']!r} differs from the upstream pin"
            f" {expected_source!r} — a tokenizer from another checkpoint or with other bytes is"
            " not distributed"
        )
    versions = asset["versions"]
    pinned = versions.get("transformers") if isinstance(versions, dict) else None
    if pinned != umt5_tokenizer.TRANSFORMERS_VERSION:
        raise DistError(
            f"{path}: the transformers {pinned!r} the tokenizer asset was written with differs"
            f" from the version {umt5_tokenizer.TRANSFORMERS_VERSION!r} that is the source of truth"
            " for the id sequences"
        )
    if asset["maxLength"] != UMT5_TOKENS[1]:
        raise DistError(
            f"{path}: maxLength {asset['maxLength']!r} differs from the accepted upper bound"
            f" {UMT5_TOKENS[1]} — the graph's upper bound on the valid length (ADR 0119"
            " decision 4)"
        )


def umt5_repo_name(_model: str) -> str:
    """The distribution repository name (one, whatever the model)."""
    return UMT5_REPO_NAME


def umt5_plan(container: Path, tokenizer: Path, model: str = UMT5_DEFAULT_MODEL) -> ModelPlan:
    """Build the plan of one umT5-XXL encoder model (all checks and reads happen here).

    `tokenizer` is the original tokenizer asset ({@link umt5_tokenizer_asset}) — a required asset
    of a repository usable on its own; it fails without it.
    """
    assert_model_name(model)
    assert_umt5_encoder(container, model)
    assert_umt5_tokenizer(tokenizer, model)
    return _encoder_plan(container, model, tokenizer=tokenizer)


def _encoder_plan(container: Path, model: str, *, tokenizer: Path | None) -> ModelPlan:
    """The plan of an already-checked container (and asset) (the same preset and the same quant
    table for the original's row and for an intake).

    With `tokenizer` None, the plan has no assets (the local experimental mirror from an intake —
    {@link intake_plan}).
    """
    artifacts = {UMT5_ROLE: Artifact(UMT5_OUTPUT_PATHS[UMT5_ROLE], source=container)}
    if tokenizer is not None:
        artifacts[UMT5_TOKENIZER_ROLE] = Artifact(
            UMT5_OUTPUT_PATHS[UMT5_TOKENIZER_ROLE], source=tokenizer
        )
    return ModelPlan(
        name=model,
        pipeline=UMT5_PIPELINE,
        artifacts=artifacts,
        weights=UMT5_WEIGHTS,
        assets=UMT5_ASSETS if tokenizer is not None else {},
        quants=complete_quant_weights(UMT5_WEIGHTS, UMT5_QUANTS),
        default_quant=UMT5_DEFAULT_QUANT,
        pipeline_config=UMT5_PIPELINE_CONFIG,
    )


def umt5_dist_plan(series_dir: Path, model: str) -> ModelPlan:
    """Build the plan from the parent of `--series` (the CLI's dispatch target)."""
    return umt5_plan(umt5_container(series_dir), umt5_tokenizer_asset(series_dir), model)


#: The modification notice (Apache 2.0 §4(b)). It lists **the changes this repository made to the
#: upstream weights**.
#:
#: MUST: the text matches the contents of the distribution — as a value it is valid prose, so it
#: passes `verify_dist` and the manifest checks, and a mismatch is noticed only after
#: distribution. The upstream storage is F32 (research 2026-10-03 umt5-export-ram — even the
#: weights the container keeps in F32 have non-zero low 16 bits), so it does not say "a copy of
#: bf16" (only the card's Relation line describes the relation to Wan2.2's bf16). The upstream is
#: the original encoder (ADR 0122 decision 1) and tokenizer, and the original has no NOTICE file,
#: so there is no NOTICE text to carry over. The identity with the Wan2.1 checkpoint was measured on
#: 2026-10-04 (research `2026-10-04-umt5-upstream-provenance` §2.1 — the checked commit is named;
#: moving Wan's pin does not change the checked fact). The "rejected inputs" of the tokenizer item
#: come from the docstring of `wan.umt5_xxl_tokenizer` (measured 2026-10-15).
UMT5_NOTICE_MARKDOWN = """# NOTICE

This repository redistributes a modified form of the encoder of the umT5-XXL checkpoint
`google/umt5-xxl` listed in `README.md` (Google, licensed under the Apache License, Version 2.0 —
see `LICENSE.md`; that repository has no NOTICE file), and a converted form of that checkpoint's
tokenizer. That checkpoint holds the encoder in float32. Its encoder weights are bit-identical to
the float32 `text_encoder` folder of `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` (commit
`0fad780a534b6463e45facd96134c9f345acfa5b`, checked on 2026-10-04). The following changes were
made:

- Only the encoder was converted (the vocabulary embedding `shared.weight` and `encoder.*`); the
  decoder and `lm_head` are not included.
- The weights were converted into the Karume container format (a `.krm` part sequence whose first
  part carries the graph and model descriptors).
- **int8 weights**: the weight matrices of all linear layers and the vocabulary embedding were
  quantized from the source float32 values to the nearest step of a symmetric int8 grid, with one
  float32 scale per output channel (per row for the embedding). The relative-position bias tables
  and the RMSNorm weights keep the source float32 values. Computation runs in float32.
- The feed-forward activation `gelu_new` (the tanh approximation written out with a cube) was
  replaced by the equivalent `GELU(approximate="tanh")`. They are the same function and differ
  only in floating-point rounding.
- The graph runs on the valid tokens only: it takes the token ids and the relative-position bucket
  indices (computed on the host with the upstream bucketing rule) as inputs and has no attention
  mask, instead of padding the prompt to 512 tokens and masking the padding.
- **Tokenizer**: the tokenizer files `tokenizer.json` and `tokenizer_config.json` of the same
  commit were converted into one JSON table (`xxl/tokenizer/tokenizer.json`: the vocabulary and its
  scores, the added tokens, the set of whitespace characters and the maximum length of 512 tokens),
  taken from the tokenizer as `transformers` 5.14.1 loads those files. The host reproduces that
  tokenizer from the table without any text normalization, and it rejects the inputs on which
  `transformers` 5.14.1 and the original `tokenizer.json` produce different token ids (special
  tokens written in the text, `▁` right after a space, whitespace other than the plain space, a
  trailing space, characters outside the vocabulary), as well as inputs of more than 512 tokens,
  which the upstream would truncate.

No retraining and no fine-tuning were performed. The original checkpoint and its original
tokenizer files are not distributed here.
"""


# ---------------------------------------------------------------------------
# Model card
# ---------------------------------------------------------------------------

#: The pipeline contract this template can describe.
UMT5_SUPPORTED_PIPELINE = UMT5_PIPELINE

#: The HF pipeline tag (an encoder that turns text into embeddings).
UMT5_PIPELINE_TAG = "feature-extraction"

#: Where the original text is (the same text as `LICENSE.md` at the distribution repository root —
#: Apache 2.0 §4(a)).
UMT5_LICENSE_TEXT_LINK = "https://www.apache.org/licenses/LICENSE-2.0"

#: The name of the Wan distribution repository that references this encoder across repositories
#: (named by the card's "Used by" — only Wan2.2 is published, Wan2.1 is not; the same spelling as
#: `wan.distribution.WAN22_REPO_NAME`; that module imports this one, so it cannot be imported from
#: here — `wan/tests/test_umt5_distribution.py` checks they match).
UMT5_CONSUMER_REPO_NAME = "karume-wan2.2"

#: The accepted valid lengths (token count, including the final `</s>` — ADR 0119 decision 4; the
#: lower bound 2 avoids `torch.export`'s 0 / 1 specialization, the upper bound is the tokenizer
#: asset's `maxLength`; a fact not in the manifest).
UMT5_TOKENS = (2, 512)

#: The encoder's output width (umT5-XXL's `d_model` — the `[L, 4096]` the card's Usage states).
UMT5_HIDDEN_WIDTH = 4096


def _umt5_upstream(name: str) -> Umt5Source:
    """Model name → upstream provenance (no card is rendered without one — a card that cannot
    state its provenance is not produced)."""
    upstream = UMT5_SOURCES.get(name)
    if upstream is None:
        raise ValueError(
            f"the upstream of model '{name}' is not in the provenance table"
            f" (wan.sources.UMT5_SOURCES) (known: {sorted(UMT5_SOURCES)}) — a card that cannot"
            " state its provenance is not rendered"
        )
    return upstream


def _umt5_metadata(manifest: Mapping[str, Any]) -> CardMetadata:
    licenses = {_umt5_upstream(name).source.license for name in manifest["models"]}
    if len(licenses) != 1:
        raise ValueError(f"the models have different licenses ({sorted(licenses)})")
    return CardMetadata(
        pipeline_tag=UMT5_PIPELINE_TAG,
        base_model=tuple(_umt5_upstream(name).source.repo for name in manifest["models"]),
        # A distribution that requantized the F32 upstream to i8 (the doc of `CardMetadata` — a
        # distribution with a changed storage form).
        base_model_relation="quantized",
        license=next(iter(licenses)),
        tags=(UMT5_PIPELINE_TAG, "text-encoder", "umt5", "wan", "webgpu"),
    )


def _umt5_overview(manifest: Mapping[str, Any]) -> list[str]:
    low, high = UMT5_TOKENS
    return [
        "## What is this",
        "",
        "The encoder of **umT5-XXL** (`google/umt5-xxl`) as a standalone text encoder for the",
        "WebGPU inference runtime **Karume**: the encoder converted into Karume's container format",
        "(a `.krm` part sequence whose first part carries the graph and model descriptors) with",
        "int8 weights, and the checkpoint's tokenizer converted into one JSON table.",
        "",
        f"- One graph, `{UMT5_ROLE}`: the token ids `{UMT5_IDS_INPUT}` `[1, L]` and the",
        f"  relative-position bucket indices `{UMT5_BUCKETS_INPUT}` `[L, L]` (both int32) in,",
        f"  the encoder's last hidden states `[1, L, {UMT5_HIDDEN_WIDTH}]` in float32 out.",
        f"  `L` is the number of tokens of the prompt itself ({low} to {high}, the end token",
        "  `</s>` included) — there is no padding and no attention mask.",
        f"- One asset, `{UMT5_TOKENIZER_ROLE}` (format `{umt5_xxl_tokenizer.ASSET_FORMAT}`):",
        "  the tokenizer of `google/umt5-xxl` as one JSON table — the vocabulary and its scores,",
        "  the added tokens, the set of whitespace characters and the maximum length of",
        f"  {high} tokens. It carries no prompt cleaning.",
        "- The relative-position bucket indices depend on `L` only and are computed on the host:",
        "  entry `[i, j]` is the bucket of the distance `j − i` (key position minus query",
        "  position) under the upstream T5 rule for a bidirectional encoder with 32 buckets and a",
        "  maximum distance of 128 (`relative_attention_num_buckets` /",
        "  `relative_attention_max_distance` in the upstream `config.json`). Each direction has",
        "  16 buckets (keys after the query use 16–31): distances 0 to 7 get one bucket each,",
        "  and longer distances fall into 8 logarithmically spaced buckets that start at 8, 12,",
        "  16, 23, 32, 46, 64 and 91 (the last one covers every distance from 91 on).",
        f"- Exporter used for the conversion: `{manifest['generator']}`. The distribution manifest"
        f" is `karume.json` (`{manifest['format']}`).",
    ]


def _umt5_base_weights(manifest: Mapping[str, Any]) -> list[str]:
    """The attribution section. The upstream revision is shown as the full pinned 40 digits (the
    same value as the container's provenance)."""
    lines = [
        "## Base weights and attribution",
        "",
        "Converted into the container format — the original checkpoint is not distributed here.",
        "",
    ]
    for name in manifest["models"]:
        upstream = _umt5_upstream(name)
        source = upstream.source
        # The original encoder is at the repository root (no subfolder) — named by the range of
        # keys read.
        lines.append(
            f"- **`{name}`**: the encoder weights (`shared.weight` and `encoder.*`) of"
            f" [{source.repo}](https://huggingface.co/{source.repo}) at commit `{source.revision}`,"
            f" licensed **{source.license}** (as of retrieval;"
            f" [full text]({UMT5_LICENSE_TEXT_LINK}) — a verbatim copy is in `LICENSE.md`)."
            " The decoder and `lm_head` are not included."
        )
        files = " and ".join(
            f"`{file}` (SHA-256 `{digest[:16]}…`)"
            for file, digest in sorted(upstream.tokenizer.items())
        )
        lines.append(
            f"- **`{name}` tokenizer**: the files {files} of the same repository and commit,"
            " converted into one JSON table."
        )
    lines += [
        "- **Relation to the Wan checkpoints**: every tensor is bit-identical to the float32",
        "  `text_encoder` folder of `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` (commit `0fad780a…`), and",
        "  the bfloat16 `text_encoder` of `Wan-AI/Wan2.2-TI2V-5B-Diffusers` (commit `b8fff731…`)",
        "  is its round-to-nearest-even rounding (checked on 2026-10-04).",
        "- **Changes made here** (listed in full in `NOTICE.md`, per Apache 2.0 §4(b)):",
        "  only the encoder converted (no decoder, no `lm_head`);",
        "  conversion into the Karume container format; the weight matrices of the linear layers",
        "  and the vocabulary embedding quantized to int8 with one scale per output channel (per",
        "  row for the embedding), the relative-position bias tables and the RMSNorm weights kept",
        "  in float32 at the source values; `gelu_new` replaced by the equivalent",
        '  `GELU(approximate="tanh")`; the graph runs on the valid tokens with the bucket indices',
        "  as an input; the tokenizer converted into one JSON table. No retraining and no",
        "  fine-tuning.",
    ]
    return lines


def _consumer_link() -> str:
    """A link (Markdown) to the Wan repository that references this encoder across
    repositories."""
    repo = f"{HF_OWNER}/{UMT5_CONSUMER_REPO_NAME}"
    return f"[`{repo}`](https://huggingface.co/{repo})"


def _umt5_usage(repo: str) -> list[str]:
    """Usage: how to use the standalone public class `Encoder` (`@karume/models/umt5`), the input
    rules, and the Wan distribution that references this encoder across repositories.

    The `fromPretrained` source is the same object-ref form as the other families' cards plus the
    pin slot (`karume.modelcard.REVISION_PIN_COMMENT`). The swap paragraph (ADR 0122 decision 7,
    item 2) describes plugging a compatible encoder into Wan instead of this one, and states the pin
    rule (a string source follows main) and that it has no effect on the precomputed path. The
    input rules are the six rejections of `wan.umt5_xxl_tokenizer` (measured 2026-10-15).
    """
    low, high = UMT5_TOKENS
    return [
        "## Usage",
        "",
        "```ts",
        'import { Encoder } from "jsr:@karume/models/umt5";',
        "",
        "const encoder = await Encoder.fromPretrained({",
        f'  repo: "{repo}",',
        *REVISION_PIN_COMMENT,
        "});",
        'const { inputIds, hiddenStates } = await encoder.encode("a cat");'
        f" // hiddenStates: [L, {UMT5_HIDDEN_WIDTH}] f32, row-major",
        "await encoder.dispose();",
        "```",
        "",
        "`encode` tokenizes the text with the bundled tokenizer, builds the relative-position",
        "bucket indices, runs the graph and returns the token ids (`L` of them, ending with",
        f"`</s>`) with the encoder's last hidden states, one row of {UMT5_HIDDEN_WIDTH:,} floats",
        "per token.",
        "",
        "### Input rules",
        "",
        "The text is tokenized as it is — no prompt cleaning and no Unicode normalization — and",
        "`</s>` is appended. The token ids then equal those of `AutoTokenizer` in `transformers`",
        "5.14.1 and of the original `tokenizer.json` of `google/umt5-xxl` alike. Inputs on which",
        "those two disagree, or that the graph cannot take, are rejected with an error instead of",
        "being encoded:",
        "",
        "- characters outside the vocabulary;",
        "- special tokens written in the text (`</s>`, `<extra_id_0>` and the other added tokens);",
        "- `▁` (U+2581) right after a space;",
        "- whitespace other than the plain space U+0020 (tabs, line breaks, no-break spaces, the",
        "  ideographic space and so on), and a trailing space — leading spaces and runs of spaces",
        "  between words are accepted;",
        f"- more than {high} tokens (the upstream would truncate silently);",
        f"- fewer than {low} tokens (an empty text).",
        "",
        "### Used by",
        "",
        f"{_consumer_link()} (Wan2.2 TI2V 5B) references this encoder from its own `karume.json`",
        "at a pinned commit of this repository (the size and the SHA-256 of every part are",
        "declared there too), so `Ti2vPipeline.fromPretrained` (`@karume/models/wan`) on that",
        "repository fetches the `text_encoder` from here — you do not pass this repository",
        "yourself. It does not use this repository's tokenizer: its own tokenizer asset adds",
        "Wan's prompt cleaning.",
        "",
        "To run Wan2.2 with a compatible umT5-XXL encoder instead of this one (for example a",
        "fine-tune converted with the Karume export recipe), pass it through the component swap",
        "seat: `components: { text_encoder: { source: { repo, revision } } }` in the options of",
        "`Ti2vPipeline.fromPretrained`. Pin the replacement with a commit — a bare repository",
        "name follows `main`, so the encoder would move with every push while the Wan manifest",
        'pins this one. The swap needs the GPU text encoder: `textEncoder: "precomputed"` never',
        "loads umT5, so a swap has no effect there. The replacement is admitted only if its graph",
        "description has the same SHA-256 as the `text_encoder` declared by the Wan manifest and",
        "its weights are int8 with float32 tables; the tokenizer still comes from the Wan",
        "distribution and is not checked against the replacement.",
    ]


def _umt5_limits(model: Mapping[str, Any]) -> list[str]:
    """The declared device limits (from the manifest's `requiredLimits` — the whole section is
    absent when none were written)."""
    declared = {
        name: quant["requiredLimits"]
        for name, quant in model["quants"].items()
        if "requiredLimits" in quant
    }
    if not declared:
        return []
    lines = ["### Device limits", ""]
    for name, limits in declared.items():
        spelled = " / ".join(f"`{key}` ≥ {value:,} bytes" for key, value in limits.items())
        lines.append(f"- `{name}`: {spelled}.")
    lines += [
        "",
        "The largest resident buffer is the int8 vocabulary embedding, held as one buffer, so the",
        "device has to grant these limits.",
    ]
    return lines


def render_umt5_model_card(
    manifest: Mapping[str, Any], repo: str, host_assets: Mapping[FileRefKey, int] = {}
) -> str:
    """Build the `README.md` body of the umT5-XXL encoder distribution (a pure function, with a
    trailing newline).

    `repo` is this repository's HF repository ID, spelled into `Encoder.fromPretrained` of Usage.
    """
    require_pipeline(manifest, UMT5_SUPPORTED_PIPELINE)
    metadata = _umt5_metadata(manifest)
    return render(
        (
            frontmatter(metadata),
            introduction(
                "umT5-XXL text encoder (int8) — Karume", hf_upstreams(metadata.base_model)
            ),
            _umt5_overview(manifest),
            [""],
            _umt5_base_weights(manifest),
            [""],
            models(manifest),
            [""],
            _umt5_usage(repo),
            *model_sections(
                manifest,
                (partial(quants, host_assets=host_assets), _umt5_limits),
            ),
        )
    )


#: The `--pipeline umt5` row (the driver merges it into core's PIPELINES).
PIPELINE = Pipeline(
    default_model=UMT5_DEFAULT_MODEL,
    repo_name=umt5_repo_name,
    plan=umt5_dist_plan,
    # The attribution (upstream repository, license) is determined by the model name
    # (`wan.sources.UMT5_SOURCES`), so it is not an axis to choose.
    card_profiles={"umt5": render_umt5_model_card},
    # The redistribution conditions §4 of the upstream license (Apache 2.0) apply to the one
    # distribution repository (ADR 0092 decision 7).
    root_files={
        "LICENSE.md": apache_license_2_0(),
        "NOTICE.md": UMT5_NOTICE_MARKDOWN,
    },
)


# ---------------------------------------------------------------------------
# The local experimental mirror from an intake (ADR 0122 decisions 5 / 6)
# ---------------------------------------------------------------------------


def intake_series_name(name: str) -> str:
    """The series of a model from an intake (`umt5-xxl-<name>-i8-dyn`). The writer
    (`wan.umt5_export.intake_input`) and the assembly ({@link intake_container}) use the same
    spelling."""
    return f"umt5-xxl-{name}-i8-dyn"


def intake_container(series_dir: Path, name: str) -> Path:
    """The representative path of an intake's container, from the series' parent
    (`outputs/series/`)."""
    return series_dir / intake_series_name(name) / UMT5_ROLE / UMT5_MODEL_FILE


def intake_repo_name(name: str) -> str:
    """The repository name of a model from an intake (the card's spelling — it never calls itself
    `karume-umt5-xxl`).

    There is no publication path (ADR 0122 decision 5 — publishing would need a naming ruling), so
    it has no `karume-` prefix and its name says it is for local experiments.
    """
    return f"umt5-xxl-{name}-local"


#: Verbatim texts that can go into `LICENSE.md` for a declared license (identifier → text). An
#: identifier without a text fails before assembly. MIT needs a copyright line (the rights holder),
#: which cannot be derived from the record, so it is not listed.
INTAKE_LICENSE_TEXTS: Mapping[str, Callable[[], str]] = {
    "apache-2.0": apache_license_2_0,
    "cc-by-sa-4.0": cc_by_sa_4_0,
}

#: The attribution profile name of the intake mirror's card (only one — not a choice).
INTAKE_CARD_PROFILE = "umt5-intake"


def _intake_terms(intake: Umt5Intake) -> str:
    """One sentence on the upstream declaration (license) (the NOTICE and the card state the same
    fact)."""
    if intake.undeclared:
        return (
            f"That repository declares no license (recorded as `{UNDECLARED_LICENSE}`), so the"
            " terms of redistribution are unknown: this conversion is for local experiments only"
            " and must not be redistributed."
        )
    return (
        f"That repository declares the license `{intake.license}` (as of retrieval — a verbatim"
        " copy is in `LICENSE.md`)."
    )


def _intake_base(intake: Umt5Intake) -> str:
    if intake.base_model is None:
        return "It declares no base model."
    return "It declares the base model " + ", ".join(f"`{m}`" for m in intake.base_model) + "."


def _intake_precision(intake: Umt5Intake) -> str:
    """One sentence on the source dtype (never misstating where the rounding starts — ADR 0122
    decision 5)."""
    if intake.dtype == "BF16":
        return (
            "The upstream file stores bfloat16 values. They were widened to float32 without loss"
            " before quantization, so the int8 rounding starts from the bfloat16 values."
        )
    return "The upstream file stores float32 values."


def intake_notice(intake: Umt5Intake) -> str:
    """The `NOTICE.md` of the intake mirror (derived from the record — ADR 0122 decision 5).

    MUST: the upstream named is only the record's repository and revision. The "No retraining" of
    the original's NOTICE ({@link UMT5_NOTICE_MARKDOWN}) and sentences naming Wan or Google as the
    upstream are not written — the further training was done by a third party, and its upstream is
    not declared.
    """
    return "\n".join(
        [
            "# NOTICE",
            "",
            "This directory is a local experimental conversion and is not published. It holds a",
            f"modified form of the file `{intake.file.name}` of the Hugging Face repository",
            f"`{intake.repo}` at commit `{intake.revision}` (SHA-256 `{intake.file.sha256}`).",
            f"{_intake_terms(intake)} {_intake_base(intake)}",
            "",
            "The following changes were made:",
            "",
            "- The weights were converted into the Karume container format (a `.krm` part",
            "  sequence whose first part carries the graph and model descriptors).",
            f"- {_intake_precision(intake)}",
            "- **int8 weights**: the weight matrices of all linear layers and the vocabulary",
            "  embedding were quantized to the nearest step of a symmetric int8 grid, with one",
            "  float32 scale per output channel (per row for the embedding). The relative-position",
            "  bias tables and the RMSNorm weights keep their float32 values. Computation runs in",
            "  float32.",
            "- The feed-forward activation `gelu_new` (the tanh approximation written out with a",
            '  cube) was replaced by the equivalent `GELU(approximate="tanh")`. They are the same',
            "  function and differ only in floating-point rounding.",
            "- The graph runs on the valid tokens only: it takes the token ids and the",
            "  relative-position bucket indices (computed on the host) as inputs and has no",
            "  attention mask.",
            "",
            "The original file is not included here.",
            "",
        ]
    )


def intake_root_files(intake: Umt5Intake) -> dict[str, str]:
    """The root files of the intake mirror (derived from the record — ADR 0122 decisions 5 / 6).

    When declared, `LICENSE.md` is the verbatim text of that identifier (fails loudly if it is not
    in {@link INTAKE_LICENSE_TEXTS}). When undeclared, no `LICENSE.md` is placed (there is no text
    to state). Either way, `NOTICE.md` is {@link intake_notice}.
    """
    files = {NOTICE_FILENAME: intake_notice(intake)}
    if intake.undeclared:
        return files
    text = INTAKE_LICENSE_TEXTS.get(intake.license)
    if text is None:
        raise DistError(
            f"intake '{intake.name}': there is no text for the license {intake.license!r}"
            f" (available texts: {sorted(INTAKE_LICENSE_TEXTS)}) — LICENSE.md cannot be built"
        )
    return {LICENSE_FILENAME: text(), **files}


def _intake_header(intake: Umt5Intake) -> list[str]:
    """The top of the card (that it is for local experiments, and the upstream declaration — ADR
    0122 decision 6)."""
    if intake.undeclared:
        return [
            "> **Local experimental conversion — do not redistribute.** The upstream repository",
            f"> declares no license (recorded as `{UNDECLARED_LICENSE}`), so the terms of",
            "> redistribution are unknown. This directory was built only to try the encoder",
            "> locally and carries no `LICENSE.md`.",
        ]
    return [
        "> **Local experimental conversion — not published.** Publishing a third-party encoder",
        "> is a separate decision; this directory was built only to try the encoder locally.",
    ]


def _intake_overview(manifest: Mapping[str, Any], intake: Umt5Intake) -> list[str]:
    return [
        "## What is this",
        "",
        f"A third-party umT5-XXL-compatible text encoder (`{intake.name}`), converted into the",
        "WebGPU inference runtime **Karume**'s container format with int8 weights. It keeps the",
        "layout of the umT5-XXL encoder (`shared.weight` and `encoder.*`).",
        "",
        f"- One graph, `{UMT5_ROLE}`: the token ids `{UMT5_IDS_INPUT}` `[1, L]` and the",
        f"  relative-position bucket indices `{UMT5_BUCKETS_INPUT}` `[L, L]` (both int32) in,",
        "  the encoder's last hidden states `[1, L, d_model]` in float32 out.",
        "- The tokenizer and the bucket indices are not part of this directory.",
        f"- Exporter used for the conversion: `{manifest['generator']}`. The distribution manifest"
        f" is `karume.json` (`{manifest['format']}`).",
    ]


def _intake_attribution(intake: Umt5Intake) -> list[str]:
    repo = intake.repo
    license_line = (
        f"not declared by the upstream repository (`{UNDECLARED_LICENSE}`)."
        if intake.undeclared
        else f"declared `{intake.license}` (as of retrieval; a verbatim copy is in `LICENSE.md`)."
    )
    base_line = (
        "not declared."
        if intake.base_model is None
        else "declared " + ", ".join(f"`{m}`" for m in intake.base_model) + "."
    )
    base_link = f"[{intake.config.base_repo}](https://huggingface.co/{intake.config.base_repo})"
    config_lines = (
        [
            "- **Configuration**: assumed — the upstream has no `config.json`, so the",
            f"  configuration of {base_link} at commit `{intake.config.base_revision}` was copied.",
            "  The relative-position bucket settings cannot be derived from the weights; if the",
            "  upstream was trained with other settings, nothing here detects it.",
        ]
        if intake.config_assumed
        else [
            "- **Configuration**: the upstream `config.json`, checked against",
            f"  {base_link} at commit `{intake.config.base_revision}` (the same graph fields and",
            "  relative-position buckets).",
        ]
    )
    return [
        "## Base weights and attribution",
        "",
        "Converted into the container format — the original file is not included here.",
        "",
        f"- **Weights**: the file `{intake.file.name}` of [{repo}](https://huggingface.co/{repo})",
        f"  at commit `{intake.revision}` (SHA-256 `{intake.file.sha256}`), retrieved"
        f" {intake.fetched_at}.",
        f"- **License**: {license_line}",
        f"- **Base model**: {base_line}",
        f"- **Source precision**: {_intake_precision(intake)}",
        *config_lines,
        "- **Changes made here** (listed in full in `NOTICE.md`): conversion into the Karume",
        "  container format; the weight matrices of the linear layers and the vocabulary",
        "  embedding quantized to int8 with one scale per output channel (per row for the",
        "  embedding), the relative-position bias tables and the RMSNorm weights kept in float32;",
        '  `gelu_new` replaced by the equivalent `GELU(approximate="tanh")`; the graph runs on the',
        "  valid tokens with the bucket indices as an input.",
    ]


def render_intake_model_card(
    manifest: Mapping[str, Any],
    repo: str,
    host_assets: Mapping[FileRefKey, int] = {},
    *,
    intake: Umt5Intake,
) -> str:
    """The `README.md` of the intake mirror (derived from the record — states the source dtype, the
    provenance and where the configuration came from).

    The frontmatter's `license` is the record's value (the mark `NOASSERTION` when undeclared — the
    same spelling as the container's provenance). `base_model` is written only when declared.
    `repo` is accepted but unused (it is not published).
    """
    require_pipeline(manifest, UMT5_SUPPORTED_PIPELINE)
    head = frontmatter(
        CardMetadata(
            pipeline_tag=UMT5_PIPELINE_TAG,
            base_model=intake.base_model or (),
            license=intake.license,
            tags=(UMT5_PIPELINE_TAG, "text-encoder", "umt5", "local-experiment"),
        )
    )
    if intake.base_model is None:
        # An undeclared base is not stated as an empty `base_model:` (YAML null) — the field is
        # dropped.
        head = [line for line in head if line != "base_model:"]
    return render(
        (
            head,
            # The intake records the upstream repository and nothing else about its publisher,
            # so the house sentence names the HF account that publishes it.
            introduction(
                f"umT5-XXL compatible text encoder `{intake.name}` (int8) — Karume",
                (Upstream(intake.repo, intake.repo.partition("/")[0]),),
            ),
            _intake_header(intake),
            [""],
            _intake_overview(manifest, intake),
            [""],
            _intake_attribution(intake),
            [""],
            models(manifest),
            *model_sections(
                manifest,
                (partial(quants, host_assets=host_assets), _umt5_limits),
            ),
        )
    )


def intake_plan(
    container: Path, intake: Umt5Intake, model: str, *, allow_undeclared_license: bool
) -> ModelPlan:
    """The plan of a model from an intake (all checks and reads happen here — before a single byte
    is written).

    MUST: an intake with an undeclared license fails unless explicitly allowed
    (`--allow-undeclared-license`) (ADR 0122 decision 6, item 1). The container's provenance is
    checked against the record's license (the mark when undeclared) and revision
    ({@link assert_umt5_encoder}), so a marked container can only be paired with a marked record.
    """
    try:
        assert_license_intent(intake, allowed=allow_undeclared_license)
    except Umt5IntakeError as cause:
        raise DistError(str(cause)) from cause
    assert_model_name(model)
    assert_umt5_encoder(container, model, intake)
    # The intake mirror does not carry the original tokenizer asset (a third-party encoder never
    # presents the original's table as its own — the asset slot is not needed for local
    # experiments).
    return _encoder_plan(container, model, tokenizer=None)


def _intake_dist_plan(
    series_dir: Path, model: str, *, intake: Umt5Intake, allow_undeclared_license: bool
) -> ModelPlan:
    return intake_plan(
        intake_container(series_dir, intake.name),
        intake,
        model,
        allow_undeclared_license=allow_undeclared_license,
    )


def _intake_repo(_model: str, *, intake: Umt5Intake) -> str:
    return intake_repo_name(intake.name)


def intake_pipeline(intake: Umt5Intake, *, allow_undeclared_license: bool) -> Pipeline:
    """The `--pipeline umt5` row of one intake (the driver builds it only when it receives
    `--intake`).

    The root files are derived from the record ({@link intake_root_files} — a license without a
    text fails here). The default output (outside `models/`) and the check that fails when the
    output is under `models/` belong to the driver.
    """
    return Pipeline(
        default_model=intake.name,
        repo_name=partial(_intake_repo, intake=intake),
        plan=partial(
            _intake_dist_plan, intake=intake, allow_undeclared_license=allow_undeclared_license
        ),
        card_profiles={INTAKE_CARD_PROFILE: partial(render_intake_model_card, intake=intake)},
        root_files=intake_root_files(intake),
    )
