"""The EmbeddingGemma 2 distribution recipe — series layout, output path table, quant table,
`pipelineConfig` and the card choice (ADR 0065 decision 2).

The general-purpose assembly engine (placement, folding shared files, sha256, manifest,
staging/swap, verification) lives in `karume.dist`. What lives here is **the facts specific to
EmbeddingGemma 2** only.

The repository `karume-embeddinggemma-2` holds one model today, `text` — the text tower of
`google/embeddinggemma-2` (the other towers of the same checkpoint would be later keys of the same
repository). Its contract is `embeddinggemma/1`, the TypeScript family `embeddinggemma`, which the
first EmbeddingGemma will join later from a repository of its own (another licence, so another
`Pipeline` with other root files). The model ships one graph and one asset:

- `model` — the bf16 series' container (`embeddinggemma2/export.py --dtype bf16`, series
  {@link EMBEDDINGGEMMA2_BF16_SERIES}): the text tower, mean pooling and L2 normalisation as one
  graph, `input_ids` / `pool_mask` `[1, T]` in, a unit vector `[1, 768]` out. Every weight in a
  weight slot is the checkpoint's own bf16 bits; the rest is f32 holding the same values widened.
- `tokenizer` — the compiled tokenizer asset (`embeddinggemma2/tokenizer.py`), held to the pinned
  `tokenizer.json` it was compiled from.

One quant preset, `bf16` (the original weights, the default — ADR 0128 decision 4).

`pipelineConfig` carries **the model's own values**, read from their sources at build time and
checked against each other ({@link embeddinggemma2_pipeline_config}):

- `prompts` — prompt name → prefix, verbatim from the pinned `config_sentence_transformers.json`;
- `dimensions` — the Matryoshka dimensions, the full one first (upstream's model card; the first
  is checked against the graph's output width);
- `maxTokens` — the most tokens one text may encode to (`<bos>` and `<eos>` included): the upper
  bound of the graph's symbol `T`, read from the graph's folded constants and held to the export's
  `SYM_MAX`;
- `placeholderTokens` — modality → the id of its placeholder token in the vocabulary, from the
  pinned `config.json` (`image_token_id` / `audio_token_id` / `video_token_id`). The upstream
  composite model replaces such an id with the pad id before the embedding; the text tower here
  does not, so a reader refuses a text that encodes to one.

The public surface is the one {@link PIPELINE} (`karume.dist.Pipeline`) — the repository's dist
driver (`tools/export-recipes/dist.py`) merges it into the core's PIPELINES.

MUST: torch-free (`import dist` must not load torch — `tests/test_dist_driver.py`). The writer's
constants this module needs are mirrored here and held equal by the tests.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

from _shared.licenses import apache_license_2_0
from _shared.upstream import assert_upstream_provenance
from embeddinggemma2 import sources as upstream_sources
from embeddinggemma2.tokenizer import TOKENIZER_SERIES, TokenizerScriptError, assert_prompts
from karume.dist import (
    Artifact,
    DistError,
    ModelPlan,
    Pipeline,
    WeightFiles,
    assert_model_name,
    assert_storage,
    assert_storage_absent,
    complete_quant_weights,
    ir_graph,
)

from .card import (
    EMBEDDINGGEMMA2_LICENSE,
    EMBEDDINGGEMMA2_MODEL_ROLE,
    render_embeddinggemma2_model_card,
)

#: The pipeline contract (ADR 0041 §2 — per model): the TypeScript family `embeddinggemma`, major 1.
EMBEDDINGGEMMA_PIPELINE = "embeddinggemma/1"

#: The one model of the repository (the text tower), also the default.
EMBEDDINGGEMMA2_MODEL = "text"

#: The distribution repository name (`karume-<family>` of ADR 0092 decision 2 — one repository for
#: this checkpoint, whichever of its towers it carries).
EMBEDDINGGEMMA2_REPO_NAME = "karume-embeddinggemma-2"

#: The roles: the graph (= the container's graph name and the manifest's weights key —
#: container-v1 §2.1) and the tokenizer asset.
EMBEDDINGGEMMA2_ROLE = EMBEDDINGGEMMA2_MODEL_ROLE
EMBEDDINGGEMMA2_TOKENIZER_ROLE = "tokenizer"

#: The storage of the shipped weights (the dtype label and the quant preset's name).
EMBEDDINGGEMMA2_DTYPE = "bf16"

#: The series the graph is read from — the mirror of
#: `embeddinggemma2.export.DEFAULT_OUT_DIRS["bf16"]` (that module imports torch; the tests hold the
#: two equal).
EMBEDDINGGEMMA2_BF16_SERIES = "embeddinggemma-2-text-bf16"

#: The representative container path inside a series and the asset file inside the tokenizer
#: series (mirrors of `embeddinggemma.export.MODEL_FILE` and the name of
#: `embeddinggemma2.tokenizer.ASSET_PATH` — held equal by the tests).
EMBEDDINGGEMMA2_MODEL_FILE = "model.krm"
EMBEDDINGGEMMA2_TOKENIZER_FILE = "tokenizer.json"

#: The relative output paths (**inside the model subtree**) — the storage dtype in the file name, as
#: in the other families.
EMBEDDINGGEMMA2_OUTPUT_PATHS: Mapping[str, str] = {
    EMBEDDINGGEMMA2_ROLE: f"{EMBEDDINGGEMMA2_ROLE}/model.{EMBEDDINGGEMMA2_DTYPE}.krm",
    EMBEDDINGGEMMA2_TOKENIZER_ROLE: (
        f"{EMBEDDINGGEMMA2_TOKENIZER_ROLE}/{EMBEDDINGGEMMA2_TOKENIZER_FILE}"
    ),
}

#: The storage the graph's binding table **must** hold ({@link assert_storage}) — the f32 series
#: holds none and fails here.
EMBEDDINGGEMMA2_STORAGE_REQUIREMENTS: Mapping[str, str] = {EMBEDDINGGEMMA2_ROLE: "bf16"}

#: The storage the graph's binding table **must not** hold ({@link assert_storage_absent}). MUST:
#: every layout of the codec ledger but f32 / i32 / bf16 — a series in another compressed storage
#: also holds bf16-free f32 parts, and only this table tells it from the checkpoint's values (a
#: mixed-up series root cannot be told apart by the numeric gates — ADR 0027 / 0029).
EMBEDDINGGEMMA2_STORAGE_FORBIDDEN: Mapping[str, tuple[str, ...]] = {
    EMBEDDINGGEMMA2_ROLE: ("f16", "i8", "i4", "i4f16", "i2")
}

#: The weights declaration (dtype label → role). One label, so {@link complete_quant_weights}
#: fills the preset's mapping.
EMBEDDINGGEMMA2_WEIGHTS: Mapping[str, Mapping[str, WeightFiles]] = {
    EMBEDDINGGEMMA2_ROLE: {EMBEDDINGGEMMA2_DTYPE: WeightFiles(EMBEDDINGGEMMA2_ROLE)}
}

#: The assets declaration (asset name → role).
EMBEDDINGGEMMA2_ASSETS: Mapping[str, str] = {
    EMBEDDINGGEMMA2_TOKENIZER_ROLE: EMBEDDINGGEMMA2_TOKENIZER_ROLE
}

#: The one preset, `bf16`: the original weights (ADR 0128 decision 4), the checkpoint's own bf16
#: values with f32 compute in the reference summation order (`session` empty — ADR 0104). The
#: label is the original-weights label of ADR 0131 / 0154.
EMBEDDINGGEMMA2_QUANTS: Mapping[str, Any] = {
    EMBEDDINGGEMMA2_DTYPE: {
        "weights": {},
        "session": {},
        "label": "Original weights (bf16)",
        "description": "The checkpoint's own bf16 weights with f32 compute.",
    },
}
EMBEDDINGGEMMA2_DEFAULT_QUANT = EMBEDDINGGEMMA2_DTYPE

#: The graph's inputs in order (the writer's `GRAPH_INPUTS`, mirrored — held equal by the tests)
#: with their dtypes, and the token-length symbol.
EMBEDDINGGEMMA2_SYMBOL = "T"
EMBEDDINGGEMMA2_GRAPH_INPUTS: tuple[tuple[str, str], ...] = (
    ("input_ids", "i32"),
    ("pool_mask", "f32"),
)

#: The most tokens one text may encode to — the mirror of `embeddinggemma2.export.SYM_MAX` (held
#: equal by the tests; the upstream model's context), and checked against the bound the graph's
#: folded constants carry ({@link embeddinggemma2_graph_max_tokens}).
EMBEDDINGGEMMA2_MAX_TOKENS = 8192

#: The Matryoshka dimensions, the full one first. The upstream model card at the pinned revision
#: states them ("The supported dimensions are 768, 512, 256, and 128"); no configuration file
#: declares them, so they are this recipe's constant, and the first is checked against the graph.
EMBEDDINGGEMMA2_DIMENSIONS: tuple[int, ...] = (768, 512, 256, 128)

#: `pipelineConfig.placeholderTokens`' keys → the `config.json` fields they are read from.
EMBEDDINGGEMMA2_PLACEHOLDER_FIELDS: tuple[tuple[str, str], ...] = (
    ("image", "image_token_id"),
    ("audio", "audio_token_id"),
    ("video", "video_token_id"),
)

#: The snapshot files the plan reads (each checked against the pin before it is read).
EMBEDDINGGEMMA2_READ_FILES: tuple[str, ...] = (
    "config.json",
    "config_sentence_transformers.json",
    "tokenizer.json",
)

#: The format of the compiled tokenizer asset (the writer is `_shared/gemma_tokenizer.py`) and the
#: post-processor the graph's pooling was traced with (`<bos>` … `<eos>` — the pooled tokens
#: include both).
EMBEDDINGGEMMA2_TOKENIZER_FORMAT = "karume-gemma-tokenizer/1"
EMBEDDINGGEMMA2_POST_PROCESSOR = "bos-eos"

#: The upstream's similarity function (`config_sentence_transformers.json`) — the card says the dot
#: product of the unit vectors is the cosine, which holds for this one only.
EMBEDDINGGEMMA2_SIMILARITY = "cosine"


def embeddinggemma2_repo_name(_model: str) -> str:
    """The distribution repository name (the same for every model of the checkpoint)."""
    return EMBEDDINGGEMMA2_REPO_NAME


@dataclass(frozen=True)
class EmbeddingGemma2Sources:
    """The assembly inputs: the bf16 series, the tokenizer series and the pinned upstream snapshot
    (`inputs/` — not a generated artifact; only its small configuration files and `tokenizer.json`
    are read)."""

    series: Path
    tokenizer: Path
    snapshot: Path


def embeddinggemma2_sources(series_dir: Path) -> EmbeddingGemma2Sources:
    """Resolve the inputs from the series parent directory (`outputs/series/`) and the snapshot's
    place (`embeddinggemma2.sources.SNAPSHOT_DIR`)."""
    return EmbeddingGemma2Sources(
        series=series_dir / EMBEDDINGGEMMA2_BF16_SERIES,
        tokenizer=series_dir / TOKENIZER_SERIES,
        snapshot=upstream_sources.SNAPSHOT_DIR,
    )


def embeddinggemma2_placements(sources: EmbeddingGemma2Sources) -> dict[str, Path]:
    """Role → source file. Anything not here stays out (the series' `io.*.safetensors`)."""
    return {
        EMBEDDINGGEMMA2_ROLE: sources.series / EMBEDDINGGEMMA2_MODEL_FILE,
        EMBEDDINGGEMMA2_TOKENIZER_ROLE: sources.tokenizer / EMBEDDINGGEMMA2_TOKENIZER_FILE,
    }


def _read_json(path: Path, what: str) -> Any:
    if not path.is_file():
        raise DistError(f"missing assembly input ({what}): {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise DistError(f"{path} cannot be read as JSON ({what})") from error


def _positive_int(value: Any, where: str) -> int:
    # bool is a subclass of int: `true` must not pass as 1.
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise DistError(f"{where} is not a positive integer ({value!r})")
    return value


def embeddinggemma2_graph_max_tokens(graph: Mapping[str, Any], path: Path) -> int:
    """The upper bound of the token symbol `T`, read from the graph's folded constants.

    The IR's `symbols` lists names only; the bound lives in the constants traced at the maximum and
    sliced by `sym_prefix_slice` (the RoPE tables — the sliding window is a declared band, not a
    constant). Every slice of `T` must read a prefix of one extent — that extent is the most rows
    the graph can bind.

    MUST: fail on a slice whose form leaves the bound unreadable (a coefficient or an offset), on
    disagreeing extents and on a graph without one — `maxTokens` would otherwise be a number
    nothing in the asset backs.
    """
    values = graph.get("values")
    nodes = graph.get("nodes")
    if not isinstance(values, dict) or not isinstance(nodes, list):
        raise DistError(f"{path}: the IR metadata has no values / nodes")
    extents: set[int] = set()
    for node in nodes:
        if not isinstance(node, dict) or node.get("op") != "sym_prefix_slice":
            continue
        attrs = node.get("attrs", {})
        if attrs.get("sym") != EMBEDDINGGEMMA2_SYMBOL:
            continue
        source = values.get(node["ins"][0], {})
        shape = source.get("shape") if isinstance(source, dict) else None
        for cut in attrs.get("slices", []):
            if cut.get("coeff") != 1 or cut.get("offset") != 0:
                raise DistError(
                    f"{path}: a sym_prefix_slice of {node['ins'][0]} cuts {cut} — the bound of"
                    f" {EMBEDDINGGEMMA2_SYMBOL} cannot be read from it"
                )
            if not isinstance(shape, list) or not 0 <= cut.get("dim", -1) < len(shape):
                raise DistError(f"{path}: the sliced constant {node['ins'][0]} has no shape")
            extents.add(_positive_int(shape[cut["dim"]], f"{path}: {node['ins'][0]}'s extent"))
    if len(extents) != 1:
        raise DistError(
            f"{path}: the folded constants sliced by {EMBEDDINGGEMMA2_SYMBOL} have the extents"
            f" {sorted(extents)} — exactly one bound is expected"
        )
    return extents.pop()


def assert_embeddinggemma2_graph(graph: Mapping[str, Any], path: Path) -> tuple[int, int]:
    """Check the graph's contract before placement and return `(vocabulary size, output width)`.

    The reader binds `input_ids` (i32) and `pool_mask` (f32) `[1, T]` by name and reads one f32
    `[1, D]` vector: a graph of another form (a mask left as an input, a fixed length, the
    per-token states) has a valid manifest and fails only on the user's machine.
    """
    inputs = graph.get("inputs")
    declared = (
        tuple((item.get("name"), item.get("dtype"), item.get("shape")) for item in inputs)
        if isinstance(inputs, list)
        else None
    )
    expected = tuple(
        (name, dtype, [1, EMBEDDINGGEMMA2_SYMBOL]) for name, dtype in EMBEDDINGGEMMA2_GRAPH_INPUTS
    )
    if declared != expected:
        raise DistError(
            f"{path}: the graph inputs are {declared}, not {expected} — the reader binds them by"
            " name, dtype and shape"
        )
    if graph.get("symbols") != [EMBEDDINGGEMMA2_SYMBOL]:
        raise DistError(
            f"{path}: the graph's symbols are {graph.get('symbols')!r}, not"
            f" [{EMBEDDINGGEMMA2_SYMBOL!r}]"
        )
    outputs = graph.get("outputs")
    values = graph.get("values")
    if not isinstance(outputs, list) or len(outputs) != 1 or not isinstance(values, dict):
        raise DistError(f"{path}: the graph outputs are {outputs!r} — an embedding has one")
    output = values.get(outputs[0], {})
    shape = output.get("shape") if isinstance(output, dict) else None
    if (
        not isinstance(output, dict)
        or output.get("dtype") != "f32"
        or not isinstance(shape, list)
        or len(shape) != 2
        or shape[0] != 1
    ):
        raise DistError(f"{path}: the graph output '{outputs[0]}' is {output!r}, not f32 [1, D]")
    width = _positive_int(shape[1], f"{path}: the output width")
    lookups = [
        node
        for node in graph.get("nodes", [])
        if isinstance(node, dict) and node.get("op") == "embedding"
    ]
    if len(lookups) != 1:
        raise DistError(
            f"{path}: {len(lookups)} embedding lookups — the text tower has one (the token table)"
        )
    table = values.get(lookups[0]["ins"][0], {})
    table_shape = table.get("shape") if isinstance(table, dict) else None
    if not isinstance(table_shape, list) or len(table_shape) != 2:
        raise DistError(f"{path}: the token table's shape is {table_shape!r}, not [vocab, hidden]")
    return _positive_int(table_shape[0], f"{path}: the vocabulary size"), width


def assert_embeddinggemma2_tokenizer(
    path: Path, *, vocab_size: int, tokenizer_sha256: str, text_config: Mapping[str, Any]
) -> Mapping[str, Any]:
    """Check the compiled tokenizer asset against the graph and the pinned upstream, and return it.

    MUST: the asset was compiled from the pinned `tokenizer.json` (its recorded SHA-256), has the
    graph's vocabulary, wraps a text in `<bos>` … `<eos>` (what the graph was traced and pooled
    with) with the ids upstream's `text_config` names — a tokenizer of the same vocabulary that
    differs in any of these gives other id sequences with every id in range.
    """
    raw = _read_json(path, "the compiled tokenizer asset")
    if not isinstance(raw, dict) or raw.get("format") != EMBEDDINGGEMMA2_TOKENIZER_FORMAT:
        raise DistError(
            f"{path}: not a compiled tokenizer asset ({EMBEDDINGGEMMA2_TOKENIZER_FORMAT}) — compile"
            " it with `python -m embeddinggemma2.tokenizer`"
        )
    source = raw.get("source")
    recorded = source.get("sha256") if isinstance(source, dict) else None
    if recorded != tokenizer_sha256:
        raise DistError(
            f"{path} was compiled from a tokenizer.json with SHA-256 {recorded}, not the pinned"
            f" {tokenizer_sha256} — compile it again from the pinned snapshot"
        )
    vocab = raw.get("vocab")
    if not isinstance(vocab, list) or len(vocab) != vocab_size:
        rows = len(vocab) if isinstance(vocab, list) else None
        raise DistError(f"{path}: {rows} vocabulary rows, not the graph's {vocab_size}")
    spec = raw.get("spec")
    post = spec.get("postProcessor") if isinstance(spec, dict) else None
    if post != EMBEDDINGGEMMA2_POST_PROCESSOR:
        raise DistError(
            f"{path}: the post-processor is {post!r}, not {EMBEDDINGGEMMA2_POST_PROCESSOR!r} —"
            " the graph pools `<bos>` … `<eos>`"
        )
    for field, upstream in (("bosId", "bos_token_id"), ("eosId", "eos_token_id")):
        if raw.get(field) != text_config.get(upstream):
            raise DistError(
                f"{path}: {field} is {raw.get(field)!r}, upstream's text_config.{upstream} is"
                f" {text_config.get(upstream)!r}"
            )
    return raw


def embeddinggemma2_placeholder_tokens(
    config: Mapping[str, Any], tokenizer: Mapping[str, Any], vocab_size: int, where: str
) -> dict[str, int]:
    """`pipelineConfig.placeholderTokens` from the pinned `config.json`.

    MUST: every id is a special token of the tokenizer asset (an added token, split out before
    BPE — so a text holding the placeholder's spelling encodes to exactly that id) and inside the
    graph's vocabulary, and the ids differ — an id outside that would never be seen, and the
    reader's refusal would test nothing.
    """
    special = tokenizer.get("specialIds")
    special_ids = set(special) if isinstance(special, list) else set()
    placeholders: dict[str, int] = {}
    for kind, field in EMBEDDINGGEMMA2_PLACEHOLDER_FIELDS:
        token = config.get(field)
        if isinstance(token, bool) or not isinstance(token, int) or not 0 <= token < vocab_size:
            raise DistError(f"{where}: {field} is {token!r}, not an id of the {vocab_size} rows")
        if token not in special_ids:
            raise DistError(
                f"{where}: {field} {token} is not a special token of the tokenizer asset"
            )
        placeholders[kind] = token
    if len(set(placeholders.values())) != len(placeholders):
        raise DistError(f"{where}: the placeholder ids repeat ({placeholders})")
    return placeholders


def embeddinggemma2_prompts(config: Mapping[str, Any], where: str) -> dict[str, str]:
    """`pipelineConfig.prompts` — upstream's prompt names and prefixes, verbatim and in order.

    MUST: upstream declares no default prompt (`default_prompt_name` null) — "no prompt, no prefix"
    is then upstream's own behaviour; with a default, a call without a prompt would differ from
    upstream silently. The similarity function must be the cosine (the card's statement). Every
    prefix has a tokenizer parity case ({@link embeddinggemma2.tokenizer.assert_prompts}).
    """
    prompts = config.get("prompts")
    if (
        not isinstance(prompts, dict)
        or not prompts
        or not all(
            isinstance(name, str) and name and isinstance(prefix, str)
            for name, prefix in prompts.items()
        )
    ):
        raise DistError(f"{where}: prompts is not a non-empty object of names to prefixes")
    if config.get("default_prompt_name") is not None:
        raise DistError(
            f"{where}: default_prompt_name is {config.get('default_prompt_name')!r} — the reader"
            " adds no prefix without a prompt, which is upstream's behaviour only without a default"
        )
    if config.get("similarity_fn_name") != EMBEDDINGGEMMA2_SIMILARITY:
        raise DistError(
            f"{where}: similarity_fn_name is {config.get('similarity_fn_name')!r}, not"
            f" {EMBEDDINGGEMMA2_SIMILARITY!r}"
        )
    try:
        assert_prompts(config)
    except TokenizerScriptError as cause:
        raise DistError(f"{where}: {cause}") from cause
    return dict(prompts)


def embeddinggemma2_pipeline_config(
    *,
    prompts: Mapping[str, str],
    width: int,
    graph_max_tokens: int,
    placeholders: Mapping[str, int],
    where: str,
) -> dict[str, Any]:
    """Build `pipelineConfig` (the four fields of the TypeScript schema, in this order).

    MUST: the full dimension equals the graph's output width, the dimensions descend, and the
    token bound equals both the graph's and the writer's ({@link EMBEDDINGGEMMA2_MAX_TOKENS}) — a
    declaration the asset does not back fails here, not on the user's machine.
    """
    dimensions = list(EMBEDDINGGEMMA2_DIMENSIONS)
    if dimensions[0] != width:
        raise DistError(
            f"{where}: the full dimension {dimensions[0]} is not the graph's output width {width}"
        )
    if any(later >= earlier for earlier, later in pairwise(dimensions)):
        raise DistError(f"the dimensions {dimensions} do not descend")
    if graph_max_tokens != EMBEDDINGGEMMA2_MAX_TOKENS:
        raise DistError(
            f"{where}: the graph binds at most {graph_max_tokens} tokens, the declaration says"
            f" {EMBEDDINGGEMMA2_MAX_TOKENS} — the series was exported with another bound"
        )
    return {
        "prompts": dict(prompts),
        "dimensions": dimensions,
        "maxTokens": EMBEDDINGGEMMA2_MAX_TOKENS,
        "placeholderTokens": dict(placeholders),
    }


def embeddinggemma2_plan(
    sources: EmbeddingGemma2Sources,
    model: str = EMBEDDINGGEMMA2_MODEL,
    pin: upstream_sources.SourcePin | None = None,
) -> ModelPlan:
    """Build the plan of the one model (every check and read is done here; `pin` is the upstream
    pin the snapshot and the container are held to — `embeddinggemma2.sources.PIN` when omitted,
    read at call time)."""
    pin = upstream_sources.PIN if pin is None else pin
    assert_model_name(model)
    if model != EMBEDDINGGEMMA2_MODEL:
        raise DistError(
            f"unknown EmbeddingGemma 2 model '{model}' (known: {EMBEDDINGGEMMA2_MODEL})"
        )
    placements = embeddinggemma2_placements(sources)
    for role, source in placements.items():
        assert_storage(role, source, EMBEDDINGGEMMA2_STORAGE_REQUIREMENTS)
        assert_storage_absent(role, source, EMBEDDINGGEMMA2_STORAGE_FORBIDDEN)
    try:
        upstream_sources.assert_pinned(sources.snapshot, EMBEDDINGGEMMA2_READ_FILES, pin)
    except upstream_sources.SourcePinError as cause:
        raise DistError(str(cause)) from cause
    container = placements[EMBEDDINGGEMMA2_ROLE]
    graph = ir_graph(container)
    vocab_size, width = assert_embeddinggemma2_graph(graph, container)
    config_path = sources.snapshot / "config.json"
    config = _read_json(config_path, "the upstream config")
    text_config = config.get("text_config") if isinstance(config, dict) else None
    if not isinstance(text_config, dict):
        raise DistError(f"{config_path}: no text_config object")
    tokenizer = assert_embeddinggemma2_tokenizer(
        placements[EMBEDDINGGEMMA2_TOKENIZER_ROLE],
        vocab_size=vocab_size,
        tokenizer_sha256=pin.sha256["tokenizer.json"],
        text_config=text_config,
    )
    prompts_path = sources.snapshot / "config_sentence_transformers.json"
    prompts_config = _read_json(prompts_path, "the upstream prompts")
    if not isinstance(prompts_config, dict):
        raise DistError(f"{prompts_path}: the top level is not an object")
    # The container names the license and the commit it was exported from (the export bakes them).
    assert_upstream_provenance(container, license=EMBEDDINGGEMMA2_LICENSE, revision=pin.revision)
    pipeline_config = embeddinggemma2_pipeline_config(
        prompts=embeddinggemma2_prompts(prompts_config, str(prompts_path)),
        width=width,
        graph_max_tokens=embeddinggemma2_graph_max_tokens(graph, container),
        placeholders=embeddinggemma2_placeholder_tokens(
            config, tokenizer, vocab_size, str(config_path)
        ),
        where=str(container),
    )
    return ModelPlan(
        name=model,
        pipeline=EMBEDDINGGEMMA_PIPELINE,
        artifacts={
            role: Artifact(EMBEDDINGGEMMA2_OUTPUT_PATHS[role], source=source)
            for role, source in placements.items()
        },
        weights=EMBEDDINGGEMMA2_WEIGHTS,
        assets=EMBEDDINGGEMMA2_ASSETS,
        # requiredLimits is not written — the core derives and bakes it during assembly.
        quants=complete_quant_weights(EMBEDDINGGEMMA2_WEIGHTS, EMBEDDINGGEMMA2_QUANTS),
        default_quant=EMBEDDINGGEMMA2_DEFAULT_QUANT,
        pipeline_config=pipeline_config,
    )


def embeddinggemma2_dist_plan(series_dir: Path, model: str) -> ModelPlan:
    """Build the plan from the parent of `--series` (the CLI's dispatch target)."""
    return embeddinggemma2_plan(embeddinggemma2_sources(series_dir), model)


#: The modification notice (Apache 2.0 §4(b)) — **the changes this repository made to the upstream
#: weights**.
#:
#: MUST: the text matches the distribution's content — it is valid prose as a value, so it passes
#: `verify_dist` and the manifest checks, and a mismatch would only be noticed after distribution.
#: On the day another tower of the checkpoint is added as a model, this text changes with it.
EMBEDDINGGEMMA2_NOTICE_MARKDOWN = """# NOTICE

This repository redistributes a modified form of `google/embeddinggemma-2`, which is licensed under
the Apache License, Version 2.0 (see `LICENSE.md`). The following changes were made:

- The **text tower only** was extracted; the vision and audio towers were never read, so images,
  video and audio cannot be embedded from this repository.
- The text tower, together with the mean pooling and the L2 normalisation of the
  SentenceTransformer pipeline, was re-expressed as one graph in the Karume container format (a
  `.krm` part sequence whose first part carries the graph and model descriptors). The
  sliding-window attention mask was replaced by a band declared on the attention of each sliding
  layer, which the runtime applies from the token positions; the rotary position tables are
  constants of the graph, which bound the input length it accepts (stated in `README.md`).
- **The weights keep the checkpoint's BF16 values unchanged** (the parameters stored in float32,
  the norms among others, hold the same values widened).
- The tokenizer ships as a compiled form of the checkpoint's `tokenizer.json`, and the prompt
  prefixes of its `config_sentence_transformers.json` are carried in `karume.json`.

No retraining and no fine-tuning were performed. The original checkpoint is not distributed here.
"""


def embeddinggemma2_root_files() -> dict[str, str]:
    """The legal texts at the repository root (`karume.dist.Pipeline.root_files`).

    Upstream is Apache License 2.0: §4(a) asks for a copy of the license (the shared original,
    {@link _shared.licenses.apache_license_2_0}, read verbatim), §4(b) for a notice of the changes.
    Upstream has no `NOTICE` file (§4(d) does not apply), so `NOTICE.md` holds the §4(b) notice
    only — the Gemma 4 distribution's form.
    """
    return {
        "LICENSE.md": apache_license_2_0(),
        "NOTICE.md": EMBEDDINGGEMMA2_NOTICE_MARKDOWN,
    }


#: The one `--pipeline embeddinggemma-2` entry (the driver merges it into the core's PIPELINES).
PIPELINE = Pipeline(
    default_model=EMBEDDINGGEMMA2_MODEL,
    repo_name=embeddinggemma2_repo_name,
    plan=embeddinggemma2_dist_plan,
    # Attribution is determined by the model name alone (`EMBEDDINGGEMMA2_UPSTREAM`).
    card_profiles={"embeddinggemma2": render_embeddinggemma2_model_card},
    root_files=embeddinggemma2_root_files(),
)
