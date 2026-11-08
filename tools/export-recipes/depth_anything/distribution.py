"""The Depth Anything V2 distribution recipe — series layout, output path table, quant table and
card selection.

ADR 0065 decision 2. The general-purpose assembly engine (placement, folding of shared slots,
sha256, manifest, staging/swap, verification) lives in `karume.dist`. This module holds only
**facts specific to Depth Anything V2**: what to pick up from which series directory, at which
path of the distribution and under which dtype label to place it, and which quant is the default.

What is distributed is **one depth-map graph only** (`depth_anything/export.py` emits just
`predicted_depth`). That graph is also the only asset needed to run; there is no tokenizer and no
table (`assets` is empty). The storage dtype is a single f32 series, so there is one quant slot
too — so far the same shape as SigLIP2 / BiRefNet.

**The repository is per generation** (`karume-depth-anything-v2` — `karume-<family>-<generation>`
of ADR 0092 decision 2). The generation is in the name because V3 is a different architecture and
not the successor for single-image depth; the size is a **model axis** inside the generation
(side by side in the same repository). Only one size can be shipped today (the MUST below), so
only one actually appears.

MUST: **only sizes listed in the attribution table
({@link depth_anything.card.DEPTH_ANYTHING_UPSTREAM}) can be shipped**. Upstream, only Small is
Apache-2.0; Base / Large are CC BY-NC 4.0 (the MUST of that table carries the date of the check).
The script (`depth_anything/export.py`) can bake any size via `--model-dir`, so nothing may read
as "whatever was baked can be shipped" — the check has two stages: the model name is listed in
this table ({@link depth_anything_checkpoint}), and the provenance the series' container states
(the licence and revision in `provenance`) matches this table and the local checkpoint
({@link depth_anything_plan}). With the name alone, a container baked from Base and placed at
Small's series path would pass.

The numbers of `pipelineConfig` come from **two independent sources**: the preprocessing
constants (dimensions, statistics, filter) from the upstream `preprocessor_config.json`, and
whether they fit from **the baked graph's input / output declarations**. Neither is copied by
hand (for the same reason as the module doc of the TS-side source of truth
`packages/models/src/depth-anything/config.ts`). A mismatched combination breaks the values
quietly, with "only the resize target differs", so {@link assert_depth_anything_graph} measures
it before placement.

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
    DEPTH_ANYTHING_CONVT_DIFF,
    DEPTH_ANYTHING_LICENSE,
    DEPTH_ANYTHING_UPSTREAM,
    render_depth_anything_model_card,
)

#: The pipeline contract (ADR 0041 §2 — per model). The set the TS side accepts is
#: `DEPTH_ANYTHING_PIPELINE_NAME` / `DEPTH_ANYTHING_PIPELINE_MAJOR`.
DEPTH_ANYTHING_PIPELINE = "depth-anything/1"

#: The default model name (the size an assembly without `--model` includes). The set of accepted
#: spellings lives in the attribution table (`depth_anything.card.DEPTH_ANYTHING_UPSTREAM`).
DEPTH_ANYTHING_DEFAULT_MODEL = "small"

#: The distribution repository name (ADR 0092 decision 2). The series name is separate (the
#: lower-cased upstream repository name — {@link depth_anything_series_name}). V2 is part of the
#: name because V3 is a different architecture (DualDPT with multiple views + camera pose) and not
#: the successor for single-image depth — without the generation, the names would read as "the
#: newer one is better". **The size is not part of the name** — the size is a model axis inside
#: the per-generation repository, and putting it in the repository name would split the
#: repository the day a second size appears.
DEPTH_ANYTHING_REPO_NAME = "karume-depth-anything-v2"

#: Parent of the real weights and `preprocessor_config.json` (where `hf download
#: depth-anything/<name> --local-dir inputs/depth-anything/<name>` unpacks — the same place as
#: `depth_anything.export.MODELS_ROOT`).
DEPTH_ANYTHING_INPUTS_DIRNAME = "depth-anything"

#: The only role name (the weights key of the manifest; `DEPTH` in the TS-side `pipeline.ts`).
DEPTH_ANYTHING_ROLE = "depth"

#: The name of the graph input (`depth_anything.export.INPUT_NAME`).
DEPTH_ANYTHING_INPUT = "pixel_values"

#: The number of input channels (RGB). The output is a `[1, H, W]` map **without a channel
#: axis**.
DEPTH_ANYTHING_CHANNELS = 3

#: The source of the preprocessing constants (the same directory as the weights).
DEPTH_ANYTHING_PREPROCESSOR_FILE = "preprocessor_config.json"

#: The interpolation the TS side implements and the matching `resample` value (**BICUBIC as a PIL
#: constant**). It **differs** from the bilinear (2) of SigLIP2 / BiRefNet, so the set accepted by
#: the TS-side `config.ts` is also narrowed to the single value bicubic. What
#: `depth_anything.export.check_processor` measures from the real processor on every emit is
#: written here as the distribution's declaration.
DEPTH_ANYTHING_INTERPOLATION = "bicubic"
DEPTH_ANYTHING_RESAMPLE = 3

#: The reciprocal of the divisor 255 in the TS-side `normalizeToNchw`. It is **not in
#: pipelineConfig** (declaring a number that cannot be chosen at run time would create two sources
#: of truth — the NOTE in `config.ts`), so a checkpoint with a different value fails here.
DEPTH_ANYTHING_RESCALE_FACTOR = 1.0 / 255.0

#: The relative output paths (**inside the model subtree**) — the one place the placement table and
#: the manifest share.
DEPTH_ANYTHING_OUTPUT_PATHS: Mapping[str, str] = {
    DEPTH_ANYTHING_ROLE: f"{DEPTH_ANYTHING_ROLE}/model.f32.krm"
}

#: The required storage dtype (the same grounds as Anima / SBV2 / Irodori / SigLIP2 / BiRefNet —
#: an actual incident in which a plain asset passed assembly, loading and execution and stayed
#: silent until the reference-match check).
DEPTH_ANYTHING_STORAGE_REQUIREMENTS: Mapping[str, str] = {DEPTH_ANYTHING_ROLE: "f32"}

#: The storage vocabulary that **must not appear** in each role's binding table
#: ({@link assert_storage_absent}). {@link DEPTH_ANYTHING_STORAGE_REQUIREMENTS} is a one-way check
#: of "is the required dtype present", and **compressed series also keep their ineligible
#: weights** (bias / norm / graph constants, the per-channel scales of i8, the group scales of i4)
#: in f32, so "contains f32" holds for f16 / i8 / i4 assets too — a mix-up that puts a compressed
#: series into the f32 slot passes a presence check alone.
#:
#: This script (`depth_anything/export.py`) bakes only f32, but what the forbidden table closes is
#: **a mix-up of the series root** (`--series` pointing at another tree / a compressed series of
#: another family placed by hand), which happens regardless of the dtypes the script supports. A
#: mix-up of the series root cannot in principle be detected by a numeric check
#: (ADR 0027 / 0029), so this is the only detector.
#:
#: MUST: the forbidden storage is held **as a set per role**, and names **every** layout of the
#: vocabulary the storage check reads (the layouts of the codec ledger
#: `karume.container.CODEC_LEDGER` — `karume.dist.storage_dtypes`) except f32 / i32 — if even one
#: is missing, that storage alone passes silently (the same discipline as anima / irodori /
#: sbv2). I32 is not listed because i32 is plain storage, not compression (the plain side of
#: `karume.emit`), and it routinely appears in plain series as index tables.
DEPTH_ANYTHING_STORAGE_FORBIDDEN: Mapping[str, tuple[str, ...]] = {
    DEPTH_ANYTHING_ROLE: ("f16", "bf16", "i8", "i4", "i4f16", "i2")
}

#: The weights declaration (dtype label → role name). There is only one dtype, so the quant table
#: can be empty ({@link complete_quant_weights} fills it into a complete mapping).
DEPTH_ANYTHING_WEIGHTS: Mapping[str, Mapping[str, WeightFiles]] = {
    DEPTH_ANYTHING_ROLE: {"f32": WeightFiles(DEPTH_ANYTHING_ROLE)}
}

#: The assets declaration. **Empty** (running needs only the one graph).
DEPTH_ANYTHING_ASSETS: Mapping[str, str] = {}

DEPTH_ANYTHING_QUANTS: Mapping[str, Any] = {"f32": {"weights": {}, "session": {}}}
DEPTH_ANYTHING_DEFAULT_QUANT = "f32"


def depth_anything_checkpoint(model: str) -> str:
    """Model name → directory name of the upstream checkpoint (= the last part of the upstream HF
    repository name).

    `depth_anything/export.py` lower-cases the `--model-dir` directory name into the series name,
    and that directory name is where `hf download <repo>` unpacks, so there is only one fact
    behind the spelling: "which upstream repository this model is". It is derived from the
    attribution table (`depth_anything.card.DEPTH_ANYTHING_UPSTREAM`) and no second table is kept
    here (the same discipline as {@link siglip2.distribution.siglip2_checkpoint} of SigLIP2).

    MUST: a size not in the table fails here — the first stage of the "ship only what is
    Apache-2.0" check (the second stage compares the container's provenance — the MUST at the top
    of the module).
    """
    repo = DEPTH_ANYTHING_UPSTREAM.get(model)
    if repo is None:
        raise DistError(
            f"unknown Depth Anything V2 model '{model}'"
            f" (known: {' / '.join(sorted(DEPTH_ANYTHING_UPSTREAM))})"
            " — only the sizes that are Apache-2.0 upstream are in the attribution table"
        )
    return repo.split("/", 1)[1]


def depth_anything_series_name(model: str) -> str:
    """Model name → series directory name (the same expression as
    `depth_anything.export.default_out_dir`)."""
    return depth_anything_checkpoint(model).lower()


def depth_anything_repo_name(_model: str) -> str:
    """The distribution repository name (one repository per generation, so the same for every
    size — ADR 0092 decision 2)."""
    return DEPTH_ANYTHING_REPO_NAME


@dataclass(frozen=True)
class DepthAnythingSources:
    """The inputs of the assembly: the series (one graph) and the location of the real weights
    (`inputs/` — not generated).

    The latter is needed to **derive the preprocessing constants of `pipelineConfig` instead of
    hard-coding them** (only `preprocessor_config.json` is read, so the 99MB of weights are not
    touched), and to compare the revision the container states with the download record of the
    local checkpoint (`_shared.upstream.snapshot_revision`).
    """

    series: Path
    model: Path


def depth_anything_sources(
    series_dir: Path, model: str = DEPTH_ANYTHING_DEFAULT_MODEL
) -> DepthAnythingSources:
    """Resolve the inputs from the series' parent directory (`outputs/series/`) and the spellings
    of `_shared.paths`."""
    checkpoint = depth_anything_checkpoint(model)
    return DepthAnythingSources(
        series=series_dir / depth_anything_series_name(model),
        model=INPUTS_ROOT / DEPTH_ANYTHING_INPUTS_DIRNAME / checkpoint,
    )


def depth_anything_placements(sources: DepthAnythingSources) -> dict[str, Path]:
    """Role name → source file. The output paths live in {@link DEPTH_ANYTHING_OUTPUT_PATHS}.

    Nothing outside this table enters the output (this is what drops the `io.*.safetensors` that
    sit in the series).
    """
    return {DEPTH_ANYTHING_ROLE: sources.series / "model.krm"}


def depth_anything_preprocessor(model_dir: Path) -> Mapping[str, Any]:
    """Read the upstream `preprocessor_config.json` (the only source of the preprocessing
    constants)."""
    path = model_dir / DEPTH_ANYTHING_PREPROCESSOR_FILE
    if not path.is_file():
        raise DistError(f"missing assembly input: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise DistError(f"{path} cannot be read as JSON") from error
    if not isinstance(raw, dict):
        raise DistError(f"{path}: not a top-level object")
    return raw


def depth_anything_pipeline_config(preprocessor: Mapping[str, Any], where: str) -> dict[str, Any]:
    """Build `pipelineConfig` (the 5 fields of the TS-side schema) from the upstream
    preprocessing config.

    MUST: unsupported preprocessing is **not accepted** (no silent approximation). The TS side
    has only the one path "antialiased bicubic → `/255` → `(x − mean) / std`", so a checkpoint
    whose `resample` / `rescale_factor` / `do_*` deviate would load and run with shifted values.
    This is the only detector.

    NOTE: `keep_aspect_ratio` / `ensure_multiple_of` are **read but not copied into the
    declaration**. The baked graph accepts only one square size (DINOv2's position embeddings are
    tied to the patch grid), so a path that keeps the aspect ratio has nowhere to go, and
    declaring it would yield a field that reads as "could keep it but does not". The fact lives
    in the model card's prose (`_depth_anything_shape`).
    """
    size = preprocessor.get("size")
    if not isinstance(size, dict):
        raise DistError(f"{where} has no size / it is not an object ({size!r})")
    resample = preprocessor.get("resample")
    if resample != DEPTH_ANYTHING_RESAMPLE:
        raise DistError(
            f"the resample of {where} is {resample!r}"
            f" (expected {DEPTH_ANYTHING_RESAMPLE} = PIL's BICUBIC)"
            f" — the TS-side preprocessing has only {DEPTH_ANYTHING_INTERPOLATION}"
        )
    rescale = preprocessor.get("rescale_factor")
    if rescale != DEPTH_ANYTHING_RESCALE_FACTOR:
        raise DistError(
            f"the rescale_factor of {where} is {rescale!r} (expected"
            f" {DEPTH_ANYTHING_RESCALE_FACTOR}) — the TS side is fixed to dividing 8-bit pixels"
            " by 255"
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
            preprocessor, "image_mean", where, channels=DEPTH_ANYTHING_CHANNELS, positive=False
        ),
        "imageStd": preprocessor_channels(
            preprocessor, "image_std", where, channels=DEPTH_ANYTHING_CHANNELS, positive=True
        ),
        "interpolation": DEPTH_ANYTHING_INTERPOLATION,
    }


def assert_depth_anything_graph(
    graph: Mapping[str, Any], path: Path, pipeline_config: Mapping[str, Any]
) -> None:
    """Measure, before placement, that the graph's inputs and outputs **fit the preprocessing
    declaration**.

    MUST: do not drop this check. The preprocessing dimensions (`preprocessor_config.json`) and the
    baked resolution (`image_size` in the weights' `config.json`) are decided separately, so
    nothing up to here fails even for a combination of different sizes — shipped mismatched, the
    Session's shape check fails on the user's machine without saying "which number is right".

    MUST: check the output's **rank** too. The depth map is `[1, H, W]` (no channel axis), which
    cannot be told apart from `[1, 1, H, W]` by element count — the later stages that look up
    outputs by position see only the element count, so a graph baked with a channel axis would
    pass silently.
    """
    inputs = graph_inputs(graph, path)
    if tuple(inputs) != (DEPTH_ANYTHING_INPUT,):
        raise DistError(
            f"the graph inputs of {path} are {list(inputs)}, not the expected"
            f" {[DEPTH_ANYTHING_INPUT]} — the runtime binds by name, so a changed spelling cannot"
            " be bound"
        )
    symbols = graph.get("symbols")
    if not isinstance(symbols, list) or symbols:
        raise DistError(
            f"{path}: has symbolic dimensions {symbols!r} — the resolution and the patch count"
            " are fixed, so no axis varies"
        )
    height, width = pipeline_config["imageHeight"], pipeline_config["imageWidth"]
    expected_input = [1, DEPTH_ANYTHING_CHANNELS, height, width]
    if inputs[DEPTH_ANYTHING_INPUT] != expected_input:
        raise DistError(
            f"{path}: the input '{DEPTH_ANYTHING_INPUT}' is {inputs[DEPTH_ANYTHING_INPUT]!r},"
            f" while the size in {DEPTH_ANYTHING_PREPROCESSOR_FILE} expects {expected_input}"
            " — the preprocessing dimensions and the baked resolution come from different"
            " versions"
        )
    outputs = graph.get("outputs")
    if not isinstance(outputs, list) or len(outputs) != 1:
        raise DistError(
            f"{path}: the graph outputs are {outputs!r} — exactly one depth map is required"
        )
    values = graph.get("values")
    value = values.get(outputs[0]) if isinstance(values, dict) else None
    shape = value.get("shape") if isinstance(value, dict) else None
    expected_output = [1, height, width]
    if shape != expected_output:
        raise DistError(
            f"{path}: the shape of graph output '{outputs[0]}' is {shape!r}, expected"
            f" {expected_output} — the depth map has the input's dimensions and no channel axis"
        )


def depth_anything_plan(
    sources: DepthAnythingSources, model: str = DEPTH_ANYTHING_DEFAULT_MODEL
) -> ModelPlan:
    """Build the plan for one Depth Anything V2 model (all checks and reads are done here)."""
    assert_model_name(model)
    # A size that cannot state its attribution fails before the series is read (the MUST at the
    # top of the module).
    depth_anything_checkpoint(model)
    placements = depth_anything_placements(sources)
    for role, source in placements.items():
        assert_storage(role, source, DEPTH_ANYTHING_STORAGE_REQUIREMENTS)
        assert_storage_absent(role, source, DEPTH_ANYTHING_STORAGE_FORBIDDEN)
    container = placements[DEPTH_ANYTHING_ROLE]
    graph = ir_graph(container)
    pipeline_config = depth_anything_pipeline_config(
        depth_anything_preprocessor(sources.model),
        str(sources.model / DEPTH_ANYTHING_PREPROCESSOR_FILE),
    )
    assert_depth_anything_graph(graph, container, pipeline_config)
    # MUST: do not close the check with the name table alone (`depth_anything_checkpoint` above)
    # — storage, input / output shapes and preprocessing have the same shape in Small and Base,
    # so a container baked from Base (CC BY-NC 4.0) placed at the series path passes everything
    # up to here. Compare the provenance the container states with the attribution table and
    # with the revision of the checkpoint the preprocessing was read from (`_shared.upstream`).
    assert_upstream_provenance(
        container, license=DEPTH_ANYTHING_LICENSE, revision=snapshot_revision(sources.model)
    )
    return ModelPlan(
        name=model,
        pipeline=DEPTH_ANYTHING_PIPELINE,
        artifacts={
            role: Artifact(DEPTH_ANYTHING_OUTPUT_PATHS[role], source=source)
            for role, source in placements.items()
        },
        weights=DEPTH_ANYTHING_WEIGHTS,
        assets=DEPTH_ANYTHING_ASSETS,
        quants=complete_quant_weights(DEPTH_ANYTHING_WEIGHTS, DEPTH_ANYTHING_QUANTS),
        default_quant=DEPTH_ANYTHING_DEFAULT_QUANT,
        pipeline_config=pipeline_config,
    )


def depth_anything_dist_plan(series_dir: Path, model: str) -> ModelPlan:
    """Build the plan for one Depth Anything V2 model from the parent of `--series` (the CLI's
    dispatch target)."""
    return depth_anything_plan(depth_anything_sources(series_dir, model), model)


#: The modification notice (Apache 2.0 §4(b)). It lists **the changes this repository made to the
#: upstream weights**.
#:
#: MUST: the text matches the distribution's contents — as a value it is plausible prose, so both
#: `verify_dist` and the manifest checks pass it, and a mismatch cannot be noticed until after it
#: has shipped. The measured width of the ConvTranspose replacement is not hard-coded but taken
#: from the card's spelling ({@link depth_anything.card.DEPTH_ANYTHING_CONVT_DIFF}) (the body and
#: the card do not state different numbers).
DEPTH_ANYTHING_NOTICE_MARKDOWN = f"""# NOTICE

This repository redistributes a modified form of the Depth Anything V2 checkpoint listed in
`README.md`, which is licensed under the Apache License, Version 2.0 (see `LICENSE.md`). The
following changes were made:

- The graph was re-expressed in the Karume container format (a `.krm` part sequence whose
  first part carries the graph and model descriptors).
- Two rewrites were needed to export the graph and are **bit-exact**: the last fusion stage's
  upsample takes an explicit output size instead of a scale factor, and the position-embedding
  interpolation is pinned to the pretraining resolution where it is the identity.
- The DPT reassemble stage's transposed convolutions were **rewritten as a 1×1 convolution
  followed by a pixel shuffle**, which reorders the weights and the summation. They have
  `kernel == stride`, so it is the same sum in a different order — equivalent up to
  floating-point rounding, **not bit-exact**: the measured maximum difference is
  {DEPTH_ANYTHING_CONVT_DIFF} on depth values whose RMS is around 1.
- **No quantization**: the stored weights are the source checkpoint's own f32 values.

No retraining and no fine-tuning were performed. The original checkpoint is not distributed here.
"""


#: The one entry for `--pipeline depth-anything` (the driver merges it into the core's
#: PIPELINES).
PIPELINE = Pipeline(
    default_model=DEPTH_ANYTHING_DEFAULT_MODEL,
    repo_name=depth_anything_repo_name,
    plan=depth_anything_dist_plan,
    # Not a selectable axis, for the same reason as SigLIP2 / BiRefNet — the attribution
    # (upstream repository, licence) is determined uniquely by the model name. Separate profiles
    # would let an operator make the mistake of "shipping Base under Small's attribution" (and
    # Base is CC BY-NC 4.0 at that).
    card_profiles={"depth-anything": render_depth_anything_model_card},
    # The redistribution conditions of the upstream licence (Apache 2.0 §4) apply to the one
    # distribution repository, so the original text is read once here regardless of how many
    # assemblies run (ADR 0092 decision 7).
    root_files={
        "LICENSE.md": apache_license_2_0(),
        "NOTICE.md": DEPTH_ANYTHING_NOTICE_MARKDOWN,
    },
)
