"""SigLIP2 distribution recipe — series layout, output path table, quant table and card selection
(ADR 0065 decision 2).

The general-purpose assembly engine (placement, folding into shared slots, sha256, manifest,
staging/swap, verification) lives in `karume.dist`. This module holds only **facts specific to
SigLIP2**: what to pick up from which series directory, at which path of the distribution and under
which dtype label to place it, and which quant is the default.

What ships is **a single graph, the vision tower** (the text tower is not included — the docstring
of `siglip2/export.py`). It is also the only asset needed to run; there is no tokenizer and no table
(`assets` is empty). The storage dtype is a single f32 series, so there is one quant slot.

**One repository, two models** (base and so400m live together in `karume-siglip2`, with base as the
default — ADR 0092 decision 1, "one repository per family", and decision 8). The former "one
repository, one model" constraint was withdrawn: the dimensions and the hidden size do differ, but
even living together the user pulls one `defaultModel`, and `pipelineConfig` declares the
dimensions per model (the reader knows them as soon as a model is chosen). Assembly is one pass over
two models, `--model base --model so400m --out <repository>`, and **the first `--model` becomes the
`defaultModel`** (`karume.dist.main`).

The numbers in `pipelineConfig` come from **two independent sources**: the preprocessing constants
from the upstream `preprocessor_config.json`, and `hiddenDim` from the output declaration of the
baked graph. Neither is copied by hand (for the same reason as the module doc of the TS-side source
of truth `packages/models/src/siglip2/config.ts`). The preprocessing dimensions and the baked
resolution are decided separately, so {@link assert_siglip2_graph} checks that they agree — a
mismatched pair takes the form "only the resize target differs", and the values break silently.

The public surface is the single {@link PIPELINE} (`karume.dist.Pipeline`) — the repository's dist
driver (`tools/export-recipes/dist.py`) merges it into the core's PIPELINES.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from _shared.licenses import apache_license_2_0
from _shared.paths import INPUTS_ROOT
from _shared.upstream import assert_upstream_provenance, snapshot_revision
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
    graph_inputs,
    ir_graph,
    preprocessor_channels,
    preprocessor_int,
)

from .card import (
    SIGLIP2_LICENSE,
    SIGLIP2_MAP_HEAD_DIFF,
    SIGLIP2_MAP_HEAD_NORM,
    SIGLIP2_UPSTREAM,
    render_siglip2_model_card,
)

#: Pipeline contract (ADR 0041 §2 — per model). The set the TS side accepts is
#: `SIGLIP2_PIPELINE_NAME` / `SIGLIP2_PIPELINE_MAJOR`.
SIGLIP2_PIPELINE = "siglip2/1"

#: Default model name (the model an assembly without `--model` includes). The set of accepted
#: spellings lives in the attribution table (`siglip2.card.SIGLIP2_UPSTREAM`).
SIGLIP2_DEFAULT_MODEL = "base"

#: Distribution repository name (`karume-<family>` of ADR 0092 decision 2). **Not derived from the
#: model name** — one repository per family, so whichever model is assembled there is one
#: destination.
SIGLIP2_REPO_NAME = "karume-siglip2"

#: Parent of the real weights and `preprocessor_config.json` (where `hf download google/<name>
#: --local-dir inputs/siglip2/<name>` unpacks — the same place as `siglip2.export.MODELS_ROOT`).
SIGLIP2_INPUTS_DIRNAME = "siglip2"

#: The only role name (the weights key of the manifest; `VISION` in the TS-side `pipeline.ts`).
SIGLIP2_ROLE = "vision"

#: Name of the graph input (`siglip2.export.INPUT_NAME`).
SIGLIP2_INPUT = "pixel_values"

#: Channel count of the input (3, which `siglip2.patch.assert_supported` also requires of the
#: config).
SIGLIP2_CHANNELS = 3

#: Source of the preprocessing constants (the same directory as the weights).
SIGLIP2_PREPROCESSOR_FILE = "preprocessor_config.json"

#: The only interpolation the TS side implements, and the matching `resample` value (**BILINEAR as
#: a PIL constant** — BICUBIC is 3. The class attribute defaults to BICUBIC, which makes this easy
#: to misread, but the checkpoint's config overrides it). The same measured target as
#: `EXPECTED_RESAMPLE` in `siglip2/preprocess.py`.
SIGLIP2_INTERPOLATION = "bilinear"
SIGLIP2_RESAMPLE = 2

#: Reciprocal of the divisor 255 that the TS-side `normalizeToNchw` holds. It is **not put in
#: pipelineConfig** (declaring a number that cannot be chosen at run time would create a second
#: source of truth — the NOTE in `config.ts`), so a checkpoint with a different value fails here.
SIGLIP2_RESCALE_FACTOR = 1.0 / 255.0

#: Relative output path (**inside the model subtree**) — the one place the placement table and the
#: manifest share. Putting the storage dtype in the file name follows Irodori (a spelling that
#: cannot be confused even when two series sit side by side).
SIGLIP2_OUTPUT_PATHS: Mapping[str, str] = {SIGLIP2_ROLE: f"{SIGLIP2_ROLE}/model.f32.krm"}

#: Storage dtype requirement (the same grounds as Anima / SBV2 / Irodori — a real incident in which
#: a plain asset passed assembly, loading and execution and stayed silent until the reference-match
#: check).
SIGLIP2_STORAGE_REQUIREMENTS: Mapping[str, str] = {SIGLIP2_ROLE: "f32"}

#: Storage vocabulary that **must not appear** in each role's binding table
#: ({@link assert_storage_absent}). {@link SIGLIP2_STORAGE_REQUIREMENTS} is a one-way check of
#: "is the required dtype present", and **compressed series also keep their ineligible weights**
#: (bias / norm / graph constants, the per-channel scales of i8, the group scales of i4) in f32, so
#: "contains f32" is true of f16 / i8 / i4 assets too — the presence check alone lets through the
#: mix-up of putting a compressed series in the f32 slot.
#:
#: This script (`siglip2/export.py`) bakes only f32, but what the forbidden table closes is **a
#: mixed-up series root** (`--series` points at another tree / a compressed series of another family
#: placed by hand), which happens regardless of the dtypes the script supports. A mixed-up series
#: root cannot in principle be detected by the numeric checks (ADR 0027 / 0029), so this is the only
#: detector.
#:
#: MUST: keep the forbidden storage **as a set per role**, and name **every** entry of the
#: vocabulary the storage check sees (the layouts of the codec ledger
#: `karume.container.CODEC_LEDGER` — `karume.dist.storage_dtypes`) other than f32 / i32 — if even
#: one is missing, that storage form alone passes silently (the same discipline as anima / irodori /
#: sbv2). I32 is not listed because i32 is plain storage, not compression (the plain side of
#: `karume.emit`), and normally appears in plain series as index tables.
SIGLIP2_STORAGE_FORBIDDEN: Mapping[str, tuple[str, ...]] = {
    SIGLIP2_ROLE: ("f16", "bf16", "i8", "i4", "i4f16", "i2")
}

#: Declaration of the weights (dtype label → role name). There is only one dtype, so the quant table
#: may be empty ({@link complete_quant_weights} fills it into a complete mapping).
SIGLIP2_WEIGHTS: Mapping[str, Mapping[str, WeightFiles]] = {
    SIGLIP2_ROLE: {"f32": WeightFiles(SIGLIP2_ROLE)}
}

#: Declaration of the assets. **Empty** (neither a tokenizer nor a table is needed — running needs
#: only the one graph).
SIGLIP2_ASSETS: Mapping[str, str] = {}

SIGLIP2_QUANTS: Mapping[str, Any] = {"f32": {"weights": {}, "session": {}}}
SIGLIP2_DEFAULT_QUANT = "f32"


def siglip2_checkpoint(model: str) -> str:
    """Model name → directory name of the upstream checkpoint (= series name = upstream HF
    repository name).

    `siglip2/export.py` uses the directory name of `--model-dir` as the series name as is, and that
    directory name is where `hf download <repository>` unpacks (= the last part of the repository
    name), so the only fact about the spelling is "which upstream repository this model is". It is
    derived from the attribution table (`siglip2.card.SIGLIP2_UPSTREAM`) and no second table is
    kept here — with two tables, the day only one moves can silently produce "attributing one
    model's weights as another model".
    """
    repo = SIGLIP2_UPSTREAM.get(model)
    if repo is None:
        raise DistError(
            f"unknown SigLIP2 model '{model}' (known: {' / '.join(sorted(SIGLIP2_UPSTREAM))})"
        )
    return repo.split("/", 1)[1]


def siglip2_repo_name(_model: str) -> str:
    """Distribution repository name (one repository per family, so the same one for every model).

    `Pipeline.repo_name` is the source of truth for the repository name shown in the card's Usage
    example (`karume.dist.resolve_repo`), and also answers the default output directory when a
    single model is assembled. Now that the family is folded into one repository it ignores the
    model name. The production assembly, which passes `--model` twice,
    needs `--out` (the driver does not derive a destination for several models —
    `dist.default_out_dir`).
    """
    return SIGLIP2_REPO_NAME


@dataclass(frozen=True)
class Siglip2Sources:
    """Assembly inputs: the series (one graph) and the location of the real weights (`inputs/` —
    not a generated artifact).

    The latter is needed to **derive, not hard-code,** the preprocessing constants of
    `pipelineConfig` (only `preprocessor_config.json` is read, so the 1.5GB of weights is never
    touched), and to compare the revision the container claims with the download record of the
    local checkpoint (`_shared.upstream.snapshot_revision`).
    """

    series: Path
    model: Path


def siglip2_sources(series_dir: Path, model: str = SIGLIP2_DEFAULT_MODEL) -> Siglip2Sources:
    """Resolve the inputs from the series parent directory (`outputs/series/`) and the spellings of
    `_shared.paths`."""
    checkpoint = siglip2_checkpoint(model)
    return Siglip2Sources(
        series=series_dir / checkpoint,
        model=INPUTS_ROOT / SIGLIP2_INPUTS_DIRNAME / checkpoint,
    )


def siglip2_placements(sources: Siglip2Sources) -> dict[str, Path]:
    """Role name → source file. The output paths live in {@link SIGLIP2_OUTPUT_PATHS}.

    Anything not in this table does not reach the output (this is what drops the
    `io.*.safetensors` files sitting in the series).
    """
    return {SIGLIP2_ROLE: sources.series / "model.krm"}


def siglip2_preprocessor(model_dir: Path) -> Mapping[str, Any]:
    """Read the upstream `preprocessor_config.json` (the only source of the preprocessing
    constants)."""
    path = model_dir / SIGLIP2_PREPROCESSOR_FILE
    if not path.is_file():
        raise DistError(f"missing assembly input: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise DistError(f"{path} cannot be read as JSON") from error
    if not isinstance(raw, dict):
        raise DistError(f"{path}: not a top-level object")
    return raw


def siglip2_pipeline_config(
    preprocessor: Mapping[str, Any], hidden_dim: int, where: str
) -> dict[str, Any]:
    """Build `pipelineConfig` (the 6 fields of the TS-side schema) from the preprocessing config and
    the baked graph.

    MUST: **do not accept** unsupported preprocessing (no silent approximation). The TS side has
    exactly one path, "antialiased bilinear → `/255` → `(x − mean) / std`", so a checkpoint whose
    `resample` / `rescale_factor` / `do_*` differ passes loading and execution with values off by
    up to 47/255 (measured with bicubic). This is the only detector.
    """
    size = preprocessor.get("size")
    if not isinstance(size, dict):
        raise DistError(f"{where} has no size / it is not an object ({size!r})")
    resample = preprocessor.get("resample")
    if resample != SIGLIP2_RESAMPLE:
        raise DistError(
            f"the resample of {where} is {resample!r} (expected {SIGLIP2_RESAMPLE} = PIL's"
            f" BILINEAR) — the TS-side preprocessing has only {SIGLIP2_INTERPOLATION}"
        )
    rescale = preprocessor.get("rescale_factor")
    if rescale != SIGLIP2_RESCALE_FACTOR:
        raise DistError(
            f"the rescale_factor of {where} is {rescale!r} (expected {SIGLIP2_RESCALE_FACTOR})"
            " — the TS side is fixed to dividing 8-bit pixels by 255"
        )
    for flag in ("do_resize", "do_rescale", "do_normalize"):
        if preprocessor.get(flag) is not True:
            raise DistError(
                f"{where}: {flag} is not true — the 3 preprocessing stages are incomplete"
            )
    for flag in ("do_center_crop", "do_pad"):
        if preprocessor.get(flag):
            raise DistError(f"{where}: {flag} is true — the TS side has neither crop nor pad")
    return {
        "imageWidth": preprocessor_int(size, "width", f"the size of {where}"),
        "imageHeight": preprocessor_int(size, "height", f"the size of {where}"),
        "imageMean": preprocessor_channels(
            preprocessor, "image_mean", where, channels=SIGLIP2_CHANNELS, positive=False
        ),
        "imageStd": preprocessor_channels(
            preprocessor, "image_std", where, channels=SIGLIP2_CHANNELS, positive=True
        ),
        "hiddenDim": hidden_dim,
        "interpolation": SIGLIP2_INTERPOLATION,
    }


def siglip2_hidden_dim(graph: Mapping[str, Any], path: Path) -> int:
    """Read the width of `pooler_output` from the baked graph's output declaration (not copied by
    hand).

    It also checks that there is exactly one output because, if another export that includes
    `last_hidden_state` or a multi-output version for goldens slipped in, **only the width would
    silently become a number with another meaning**.
    """
    outputs = graph.get("outputs")
    if not isinstance(outputs, list) or len(outputs) != 1:
        raise DistError(
            f"{path}: the graph outputs are {outputs!r} — the vision tower emits exactly one"
            " pooler_output"
        )
    values = graph.get("values")
    value = values.get(outputs[0]) if isinstance(values, dict) else None
    shape = value.get("shape") if isinstance(value, dict) else None
    if not isinstance(shape, list) or len(shape) != 2 or shape[0] != 1:
        raise DistError(
            f"{path}: the shape of graph output '{outputs[0]}' is not [1, hidden] ({shape!r})"
        )
    hidden = shape[1]
    if not isinstance(hidden, int) or isinstance(hidden, bool) or hidden <= 0:
        raise DistError(
            f"{path}: the hidden size of the graph output is not a positive integer ({hidden!r})"
        )
    return hidden


def assert_siglip2_graph(
    graph: Mapping[str, Any], path: Path, pipeline_config: Mapping[str, Any]
) -> None:
    """Check, before placement, that the graph's input **agrees with the preprocessing
    declaration**.

    MUST: do not drop this. The preprocessing dimensions (`preprocessor_config.json`) and the baked
    resolution (the weights' `config.json`) are decided separately, so pairing base's preprocessing
    config with so400m's graph fails nowhere up to this point — shipped mismatched, the Session's
    shape check fails on the user's machine without saying "which of the numbers is right".
    """
    inputs = graph_inputs(graph, path)
    if tuple(inputs) != (SIGLIP2_INPUT,):
        raise DistError(
            f"the graph inputs of {path} are {list(inputs)}, not the expected {[SIGLIP2_INPUT]}"
            " — the runtime binds by name, so a changed spelling cannot be bound"
        )
    symbols = graph.get("symbols")
    if not isinstance(symbols, list) or symbols:
        raise DistError(
            f"{path}: has symbolic dimensions {symbols!r} — the resolution and the patch count"
            " are fixed, so no axis varies"
        )
    expected = [
        1,
        SIGLIP2_CHANNELS,
        pipeline_config["imageHeight"],
        pipeline_config["imageWidth"],
    ]
    if inputs[SIGLIP2_INPUT] != expected:
        raise DistError(
            f"{path}: the input '{SIGLIP2_INPUT}' is {inputs[SIGLIP2_INPUT]!r}, while the size in"
            f" {SIGLIP2_PREPROCESSOR_FILE} expects {expected}"
            " — the preprocessing dimensions and the baked resolution come from different versions"
        )


def siglip2_plan(sources: Siglip2Sources, model: str = SIGLIP2_DEFAULT_MODEL) -> ModelPlan:
    """Build the plan for one SigLIP2 model (all checks and config reads happen here)."""
    assert_model_name(model)
    placements = siglip2_placements(sources)
    for role, source in placements.items():
        assert_storage(role, source, SIGLIP2_STORAGE_REQUIREMENTS)
        assert_storage_absent(role, source, SIGLIP2_STORAGE_FORBIDDEN)
    container = placements[SIGLIP2_ROLE]
    graph = ir_graph(container)
    pipeline_config = siglip2_pipeline_config(
        siglip2_preprocessor(sources.model),
        siglip2_hidden_dim(graph, container),
        str(sources.model / SIGLIP2_PREPROCESSOR_FILE),
    )
    assert_siglip2_graph(graph, container, pipeline_config)
    # Compare the provenance the container claims with the attribution table and with the
    # revision of the checkpoint the preprocessing was read from (`_shared.upstream` — the name
    # table alone does not close the check; the same shape as depth_anything).
    assert_upstream_provenance(
        container, license=SIGLIP2_LICENSE, revision=snapshot_revision(sources.model)
    )
    return ModelPlan(
        name=model,
        pipeline=SIGLIP2_PIPELINE,
        artifacts={
            role: Artifact(SIGLIP2_OUTPUT_PATHS[role], source=source)
            for role, source in placements.items()
        },
        weights=SIGLIP2_WEIGHTS,
        assets=SIGLIP2_ASSETS,
        quants=complete_quant_weights(SIGLIP2_WEIGHTS, SIGLIP2_QUANTS),
        default_quant=SIGLIP2_DEFAULT_QUANT,
        pipeline_config=pipeline_config,
    )


def siglip2_dist_plan(series_dir: Path, model: str) -> ModelPlan:
    """Build the plan for one SigLIP2 model from the parent of `--series` (the CLI's dispatch
    target)."""
    return siglip2_plan(siglip2_sources(series_dir, model), model)


#: Modification notice (Apache 2.0 §4(b)). Lists **the changes this repository made to the upstream
#: weights**.
#:
#: MUST: the text matches the contents of the distribution — it is valid prose as a value, so
#: `verify_dist` and the manifest checks let it through, and a mismatch cannot be noticed until it
#: has shipped. The measured MAP head range is not hard-coded but built from **the same text** as
#: the model card ({@link siglip2.card.SIGLIP2_MAP_HEAD_DIFF} and
#: {@link siglip2.card.SIGLIP2_MAP_HEAD_NORM} — `siglip2.measurements` is the source of truth for
#: both the number and the format). The size of the difference can only be read as a ratio to the
#: scale (the L2 norm), so both enter through the same path.
SIGLIP2_NOTICE_MARKDOWN = f"""# NOTICE

This repository redistributes a modified form of the SigLIP2 checkpoints listed in `README.md`,
which are licensed under the Apache License, Version 2.0 (see `LICENSE.md`). The following changes
were made:

- The **vision tower only** was extracted; the text tower was never read, so zero-shot
  classification and image/text similarity are not possible from this repository.
- The graph was re-expressed in the Karume container format (a `.krm` part sequence whose
  first part carries the graph and model descriptors).
- Two rewrites were needed to export the graph and are **bit-exact**: the patch embedding's
  string padding became its numeric equivalent, and the position-embedding row gather became a
  direct addition.
- The pooling (MAP) head's packed attention was rewritten with explicit q/k/v projections. That
  is equivalent up to floating-point rounding, **not bit-exact**: measured on the pooled vector,
  whose L2 norm is {SIGLIP2_MAP_HEAD_NORM}, the difference is {SIGLIP2_MAP_HEAD_DIFF}.
- **No quantization**: the stored weights are the source checkpoints' own f32 values.

No retraining and no fine-tuning were performed. The original checkpoints are not distributed here.
"""


#: The `--pipeline siglip2` entry (the driver merges it into the core's PIPELINES).
PIPELINE = Pipeline(
    default_model=SIGLIP2_DEFAULT_MODEL,
    repo_name=siglip2_repo_name,
    plan=siglip2_dist_plan,
    # Attribution is **determined uniquely by the model name** (`SIGLIP2_UPSTREAM` — base and
    # so400m are weights from different repositories), so it is not an axis to choose. Separate
    # profiles would let the operator make the mix-up of "shipping so400m under base's
    # attribution".
    card_profiles={"siglip2": render_siglip2_model_card},
    # The redistribution conditions of the upstream license (Apache 2.0 §4) apply once per
    # distribution repository, so the full text is also read once here regardless of how many
    # assemblies run (ADR 0092 decision 7).
    root_files={"LICENSE.md": apache_license_2_0(), "NOTICE.md": SIGLIP2_NOTICE_MARKDOWN},
)
