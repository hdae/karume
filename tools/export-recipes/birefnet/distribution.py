"""Distribution recipe of the BiRefNet family — series layout, output path table, quant table and
choice of card.

ADR 0065 decision 2. The general-purpose assembly engine (placement, folding of shared presets,
sha256, manifest, staging/swap, verification) lives in `karume.dist`. What lives here is **only the
facts specific to the BiRefNet family**: what to pick up from which series directory, at which path
of the distribution and under which dtype label to list it, and which quant to make the default.

What is shipped is **a single matte graph** (`birefnet/export.py` emits only the final stage's one
logit). It is also the only asset the run needs — no tokenizer, no tables (`assets` is empty). The
quant presets (= storage dtypes) are **derived per checkpoint from the type of the upstream
storage** (ADR 0113 addendum 2026-10-11 — {@link birefnet_seat_dtypes}): no preset wider than the
upstream storage, and the default is the preset of the same type as the storage. HR (upstream f16)
has only the `f16` preset; Lucida (upstream f32) has `f32` (default) and `f16`. Each dtype's series
is a separate directory (`lucida-1024` / `lucida-1024-f16`), listed in the distribution as
`matte/model.{dtype}.krm`.

**Two repositories, two models in each.** Having two axes is what this family looks like, and
mixing them breaks the distribution silently:

- **The repository axis = the checkpoint** (upstream `BiRefNet_HR` and the derivative `lucida`). The
  derivative is a separate repository (ADR 0092 decision 1) and is not determined by `--model` — the
  **entries {@link PIPELINE} / {@link LUCIDA_PIPELINE} hold it fixed**.
- **The model axis = the resolution** ({@link BIREFNET_MODELS} = `"1024"` / `"2048"`, default
  `"1024"`). Each resolution is a **separate graph** with even the window masks and padding
  constants baked in as constants (the "resolution axis" of `birefnet/export.py`), so they share the
  same "1 repository, 2 models" container as SigLIP2's base / so400m (ADR 0092 decision 8). The user
  pulls one `defaultModel`, and `pipelineConfig` declares the dimensions per model. The assembly is
  one pass over the 2 models, `--model 1024 --model 2048 --out <repository>`, and **the first
  `--model` becomes `defaultModel`** (`karume.dist.main`).

Repository names are not derived; {@link BIREFNET_REPO_NAMES} holds them — `karume-lucida` is not
"one of the BiRefNet family" but a model upstream markets under its own name, and its spelling is a
naming decision, not something that follows from the checkpoint name (the same nature as SBV2's
family names).

This is why the public surface is split into **2 Pipelines** (the same shape as anima's official /
fine-tuned): `root_files` (`LICENSE.md` / `NOTICE.md` at the root of the distribution repository)
is one set fixed per Pipeline, and the MIT copyright lines differ per repository (HR has the one
ZhengPeng line, Lucida has 2 lines, the fine-tune's and the upstream's). Folding them into one would
make one of the repositories **claim a copyright that is not its own** or drop the upstream's
copyright notice — both stay plausible prose, so they pass `verify_dist` and the manifest checks,
and nobody could notice until after shipping. The card's attribution comes in through the same entry
(`checkpoint` of {@link render_birefnet_model_card}) — the attribution, the copyright lines and the
series path are all determined **from one checkpoint name**, so a mix-up across entries cannot even
be spelled.

The values of `pipelineConfig` come from **3 independent sources**:

- The resize target (`imageWidth` / `imageHeight`) is derived from **the input declaration of the
  baked graph** (not copied by hand).
- The normalization constants have **no** machine-readable source upstream — the BiRefNet family has
  nothing like SigLIP2's `preprocessor_config.json`; the fact is written only inside the bundled
  `handler.py`'s `ImagePreprocessor` (and the model card's usage example). So this module holds them
  as a declaration ({@link BIREFNET_IMAGE_MEAN}), and pytest cross-checks it against the script's
  copy (`birefnet.export.IMAGENET_MEAN`) on every run (the two tables are never left to move
  independently).
- The post-processing procedure (`postprocess`) is determined by the **checkpoint**
  ({@link BIREFNET_POSTPROCESS}). The upstream procedure differs per series (ADR 0125), and the
  loader (TS) holds no table of series (the MUST of `config.ts` — with third-party fine-tunes it
  would become a silent mix-up), so the assembly stage, which knows the checkpoint, writes it.

The public surface is two entries (`karume.dist.Pipeline`), {@link PIPELINE}
(`--pipeline birefnet`) and {@link LUCIDA_PIPELINE} (`--pipeline lucida`) — the repository's dist
driver (`tools/export-recipes/dist.py`) merges them into the core's PIPELINES.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, NamedTuple

from _shared.licenses import mit_license
from _shared.upstream import assert_upstream_provenance
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
)

from .card import (
    BIREFNET_CHECKPOINTS,
    BIREFNET_LICENSE,
    BIREFNET_POSTPROCESS_DESCRIPTIONS,
    BIREFNET_UPSTREAM,
    birefnet_storage_bits,
    birefnet_storage_lines,
    render_birefnet_model_card,
)

#: Pipeline contract (ADR 0041 §2 — per model). The TS side's accepted set is
#: `BIREFNET_PIPELINE_NAME` / `BIREFNET_PIPELINE_MAJOR`.
BIREFNET_PIPELINE = "birefnet/2"

#: Accepted set of shipped model names = **input resolutions** (this is the one source of truth for
#: the spelling). Each resolution is a separate series, so the model name itself says "which series
#: is shipped".
BIREFNET_MODELS: tuple[str, ...] = ("1024", "2048")

#: Default model name (the model an assembly without `--model` includes).
BIREFNET_DEFAULT_MODEL = "1024"

#: Spelling of the checkpoints (**held by the pipeline entry** — not the `--model` axis). They are
#: also the keys of the attribution table (`birefnet.card.BIREFNET_CHECKPOINTS`); the series path,
#: repository name, copyright lines and the card's attribution are all determined by this one name.
BIREFNET_HR_CHECKPOINT = "hr"
BIREFNET_LUCIDA_CHECKPOINT = "lucida"

#: Parent of the real weights (where `hf download <repository> --local-dir inputs/birefnet/<name>`
#: unpacks — the same place as `birefnet.export.MODELS_ROOT`). The series name follows from this
#: directory name.
BIREFNET_INPUTS_DIRNAME = "birefnet"

#: checkpoint → distribution repository name (the `karume-` prefix is the repository-name ruling of
#: 2026-08-09). **Not derived** — as the module doc says, the spelling is a naming decision.
BIREFNET_REPO_NAMES: Mapping[str, str] = {
    "hr": "karume-birefnet-hr",
    "lucida": "karume-lucida",
}

#: The only role name (the manifest's weights key; `MATTE` in the TS side's `pipeline.ts`).
BIREFNET_ROLE = "matte"

#: Name of the graph input (`birefnet.export.INPUT_NAME`).
BIREFNET_INPUT = "pixel_values"

#: Number of input channels (RGB) and of output (matte) channels.
BIREFNET_CHANNELS = 3
BIREFNET_MATTE_CHANNELS = 1

#: The only interpolation the TS side implements. Upstream (`handler.py` / the model card's usage
#: example) both pass `torchvision.transforms.Resize((S, S))` with the default interpolation =
#: bilinear.
BIREFNET_INTERPOLATION = "bilinear"

#: checkpoint → post-processing procedure (`pipelineConfig.postprocess` — spelled as the TS side's
#: `MATTE_PROCEDURES`). The interpolation and step order of each checkpoint's upstream procedure
#: (ADR 0125 decision 3):
#:
#: - HR: the bundled `handler.py` and the model card's usage example = σ → 8 bit → Pillow's BICUBIC.
#: - Lucida: the model card's usage example = σ → torchvision's antialiased bilinear (floating
#:   point) → 8 bit.
#:
#: HR's procedure follows upstream, by the rule set in advance, after a blind A/B (no difference
#: seen in any of the 8 pairs) (the results section of ADR 0125). Every value must be a key of the
#: card's description table (`birefnet.card`) (writing a procedure without a description gives a
#: distribution whose card cannot be rendered — checked at assembly by {@link birefnet_postprocess},
#: and for the whole table by pytest).
BIREFNET_POSTPROCESS: Mapping[str, str] = {
    BIREFNET_HR_CHECKPOINT: "u8-pil-bicubic",
    BIREFNET_LUCIDA_CHECKPOINT: "float-bilinear",
}

#: Normalization constants of the preprocessing (as the module doc says, upstream has no
#: machine-readable source, so they are held here). The source of truth is the bundled
#: `handler.py`'s `ImagePreprocessor` = ImageNet statistics.
BIREFNET_IMAGE_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
BIREFNET_IMAGE_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)

#: Vocabulary of storage dtypes (widest first — phase 1 = f32 / f16, ADR 0113). **Not the set of
#: presets**: the presets each checkpoint ships are narrowed from this vocabulary by the type of the
#: upstream storage ({@link birefnet_seat_dtypes}). The series-name suffix, the output path and the
#: storage requirement table are per-role facts held over the whole vocabulary (held separately,
#: only one of them would be updated on the day a dtype is added — the same discipline as irodori's
#: `IRODORI_DTYPE_ROLES`), and the set of presets decides which roles are assembled. i8 is added in
#: phase 2 together with a BiRefNet-specific i8 plan (kept in step with
#: `birefnet.export.WEIGHT_DTYPES` — cross-checked by pytest).
BIREFNET_WEIGHT_DTYPES: tuple[str, ...] = ("f32", "f16")

#: dtype of the uncompressed series (the only dtype without a suffix in the series name — a spelling
#: rule, a separate fact from the default preset).
BIREFNET_PLAIN_DTYPE = "f32"


def _stored_dtype(checkpoint: str) -> str:
    """Storage dtype of the upstream checkpoint (a checkpoint missing from the attribution table is
    refused by {@link birefnet_checkpoint})."""
    birefnet_checkpoint(checkpoint)
    return BIREFNET_CHECKPOINTS[checkpoint].stored_dtype


def birefnet_seat_dtypes(checkpoint: str) -> tuple[str, ...]:
    """Storage dtypes of the quant presets a checkpoint ships (in the order of
    {@link BIREFNET_WEIGHT_DTYPES}).

    Rule (ADR 0113 addendum 2026-10-11): **no preset wider than the upstream checkpoint's storage
    (`stored_dtype`)**. A wider preset only widens the checkpoint's values exactly: no information
    is added, while residency and download grow (HR's f32 preset has 919 MiB of resident weights at
    1024² — the f16 preset 509 MiB). A preset narrower than the storage holds rounded values, and
    NOTICE and the card call it a quantization. **Another type of the same width** as the storage
    (bf16 for an f16 checkpoint, on the day it joins the vocabulary) is also a rounding, not a
    preset that keeps the values (the default preset is chosen by matching type — {@link
    birefnet_default_quant}). HR (f16) → `("f16",)`, Lucida (f32) → `("f32", "f16")`.

    MUST: the set of presets comes from here and nowhere else — fetching the series
    ({@link birefnet_sources}), placement, weights, the quant table, the default preset, NOTICE and
    the series `pipelineConfig` is based on. A preset table kept by hand per checkpoint would move
    independently of the storage facts (`birefnet.card.BIREFNET_CHECKPOINTS`). Export's `--dtype` is
    narrowed by the same rule (`birefnet.export`).
    """
    stored = _stored_dtype(checkpoint)
    return tuple(
        dtype
        for dtype in BIREFNET_WEIGHT_DTYPES
        if birefnet_storage_bits(dtype) <= birefnet_storage_bits(stored)
    )


def birefnet_default_quant(checkpoint: str) -> str:
    """The checkpoint's default preset = **the preset of exactly the upstream storage dtype** (the
    only preset that keeps the values).

    It is not chosen by matching bit width — another type of the same width (bf16 for an f16
    checkpoint) is a rounding, and choosing by width could, on the day it joins the vocabulary, ship
    a preset that changes the values as "the default that keeps the values".

    MUST: do not assemble a checkpoint without such a preset — extrapolating another type's preset
    as the default would ship weights that differ from upstream's values when the user specifies
    nothing (or ship a heavy preset that only widens the values).
    """
    stored = _stored_dtype(checkpoint)
    seats = birefnet_seat_dtypes(checkpoint)
    if stored not in seats:
        raise DistError(
            f"BiRefNet family checkpoint '{checkpoint}' has no preset for its storage {stored}"
            f" (presets: {' / '.join(seats)}) — cannot choose the default preset"
        )
    return stored


def birefnet_role(dtype: str) -> str:
    """Placement role name (`matte_f16` — one word shared by the placement table, the output path
    and the storage dtype requirement)."""
    return f"{BIREFNET_ROLE}_{dtype}"


#: Relative output paths (**inside the model subtree**) — the one place shared by the placement
#: table and the manifest. Putting the storage dtype in the file name is the same shape as Anima /
#: SBV2 / Irodori (`model.f16.krm`), so that two series side by side in one directory cannot be
#: mistaken for each other.
BIREFNET_OUTPUT_PATHS: Mapping[str, str] = {
    birefnet_role(dtype): f"{BIREFNET_ROLE}/model.{dtype}.krm" for dtype in BIREFNET_WEIGHT_DTYPES
}

#: Storage dtype requirements (the same grounds as Anima / SBV2 / Irodori / SigLIP2 — a measured
#: incident where a plain asset went through assembly, loading and execution silently until the
#: reference-match check). In an f16 series only the eligible weight slots are f16, and biases /
#: norms / graph constants / deform's weights stay f32 (`birefnet/export.py`), so the requirement is
#: "contains that vocabulary". The reverse (an f32 series on the f16 preset) is caught here by the
#: requirement being absent.
BIREFNET_STORAGE_REQUIREMENTS: Mapping[str, str] = {
    birefnet_role(dtype): dtype for dtype in BIREFNET_WEIGHT_DTYPES
}

#: Storage vocabulary that **must not appear** in each role's binding table
#: ({@link assert_storage_absent}). {@link BIREFNET_STORAGE_REQUIREMENTS} is a one-way check of "is
#: the required dtype present", and **compressed series also hold their ineligible weights** (biases
#: / norms / graph constants, i8's per-channel scales, i4's group scales) in f32, so "contains f32"
#: is true even for f16 / i8 / i4 assets — plugging a compressed series into the f32 preset passes a
#: presence check alone.
#:
#: What the forbidden table closes is **a mix-up of the series root** (`--series` pointing at
#: another tree / a compressed series of another family placed by hand), which happens regardless of
#: the dtypes the script supports. A mix-up of the series root cannot in principle be detected by
#: the numeric checks (ADR 0027 / 0029), so this is the only detector.
#:
#: MUST: the forbidden storage is held **as a set per role**, and it names **all** of the vocabulary
#: the storage check sees (the layouts of the codec ledger `karume.container.CODEC_LEDGER` —
#: `karume.dist.storage_dtypes`) other than f32 / i32 — leaving even one out lets exactly that
#: storage form through silently (the same discipline as anima / irodori / sbv2). I32 is not listed
#: because i32 is plain storage, not compression (the plain side of `karume.emit`) — in fact this
#: family's series holds one i32 index table, so the binding table is f32 + i32 (measured
#: 2026-08-30), and forbidding i32 would turn the existing distribution red.
#:
#: The f16 preset has no forbidden table: an f32 series fails the requirement check (no f16), and
#: there are no mixed compressed series yet (irodori's i8 preset forbids i4 because the i4 series is
#: a mix that contains i8 — phase 2's i8 series is planned to have a single compressed dtype, and
#: plugging it into the f16 preset fails the requirement check).
BIREFNET_STORAGE_FORBIDDEN: Mapping[str, tuple[str, ...]] = {
    birefnet_role(BIREFNET_PLAIN_DTYPE): ("f16", "bf16", "i8", "i4", "i2")
}

#: Declaration of assets. **Empty** (the run needs only the one graph).
BIREFNET_ASSETS: Mapping[str, str] = {}


def birefnet_weights(checkpoint: str) -> dict[str, dict[str, WeightFiles]]:
    """Declaration of weights (dtype label → role name — the presets are
    {@link birefnet_seat_dtypes}).

    The one matte has as many dtypes as there are presets, so the quant table names them explicitly
    ({@link complete_quant_weights} fills them in only when there is a single dtype).
    """
    return {
        BIREFNET_ROLE: {
            dtype: WeightFiles(birefnet_role(dtype)) for dtype in birefnet_seat_dtypes(checkpoint)
        }
    }


class BirefnetQuantSeat(NamedTuple):
    """Display fields of one quant preset shown in the selection UI (ADR 0075)."""

    #: Short display name shown in the selection UI (64 characters at most).
    label: str
    #: Likewise, a one-line description (200 characters at most). It does not say that it is the
    #: default (`defaultQuant` points at it).
    description: str


#: (storage dtype of the preset, storage dtype of the upstream checkpoint) →
#: {@link BirefnetQuantSeat}. There are no execution knobs (`session`) — every preset computes in
#: f32; only the storage of the weights differs.
#:
#: The wording is looked up by the pair with the checkpoint's storage because the presets listed
#: differ per checkpoint ({@link birefnet_seat_dtypes}): the f16 preset's "a smaller download" holds
#: only in a repository that also lists the f32 preset (Lucida), and in a repository with only the
#: f16 preset (HR) it would be a comparison with nothing. The detailed facts of whether the values
#: change (whether it is a quantization) are in NOTICE and the card's attribution section. MUST: do
#: not assemble a preset whose pair is not in the table ({@link birefnet_quants}) — no extrapolated
#: wording goes into the selection UI.
BIREFNET_QUANT_SEATS: Mapping[tuple[str, str], BirefnetQuantSeat] = {
    ("f32", "f32"): BirefnetQuantSeat(
        label="Full precision (f32)",
        description="Every weight in f32 storage — the largest download and the most resident"
        " GPU memory.",
    ),
    ("f16", "f32"): BirefnetQuantSeat(
        label="Half-size weights (f16)",
        description="The linear, convolution and embedding (relative-position) weights stored as"
        " f16 and computed in f32 — a smaller download and less resident GPU memory.",
    ),
    ("f16", "f16"): BirefnetQuantSeat(
        label="Upstream precision (f16)",
        description="The checkpoint's own f16 values: the linear, convolution and embedding"
        " (relative-position) weights stay f16 in storage and are computed in f32.",
    ),
}


def birefnet_quants(checkpoint: str) -> dict[str, Any]:
    """The checkpoint's quant table (preset name = storage dtype, in the order of
    {@link birefnet_seat_dtypes})."""
    stored = _stored_dtype(checkpoint)
    quants: dict[str, Any] = {}
    for dtype in birefnet_seat_dtypes(checkpoint):
        seat = BIREFNET_QUANT_SEATS.get((dtype, stored))
        if seat is None:
            raise DistError(
                f"no display wording for an {dtype} storage preset of the {stored} checkpoint"
                f" '{checkpoint}' — extrapolated wording is not shown in the selection UI"
            )
        quants[dtype] = {
            "weights": {BIREFNET_ROLE: dtype},
            "session": {},
            "label": seat.label,
            "description": seat.description,
        }
    return quants


def birefnet_checkpoint(checkpoint: str) -> str:
    """Checkpoint name → directory name of the upstream checkpoint (= the tail of the upstream HF
    repository name).

    `birefnet/export.py` makes the directory name of `--model-dir` the series name, and that
    directory is where `hf download <repository>` unpacks, so there is only one fact about the
    spelling: "which upstream repository this checkpoint is". It is derived from the attribution
    table (`birefnet.card.BIREFNET_UPSTREAM`) instead of keeping a second table here (the same
    discipline as SigLIP2's {@link siglip2.distribution.siglip2_checkpoint}).
    """
    repo = BIREFNET_UPSTREAM.get(checkpoint)
    if repo is None:
        raise DistError(
            f"BiRefNet family checkpoint '{checkpoint}' is unknown"
            f" (known: {' / '.join(sorted(BIREFNET_UPSTREAM))})"
        )
    return repo.split("/", 1)[1]


def birefnet_directory_checkpoint(directory: str) -> str | None:
    """Directory name of the upstream checkpoint → checkpoint name (the inverse of
    {@link birefnet_checkpoint}).

    This is the **second** way `birefnet/export.py` looks up the preset rule
    ({@link birefnet_seat_dtypes}) from `--model-dir` (when the download record is missing,
    unreadable or matches no fixed value — {@link birefnet_revision_checkpoint}); it looks up by the
    same link as the series name (directory name = tail of the upstream HF repository name). A name
    missing from the attribution table gives `None` (an experimental `--model-dir` — the
    distribution assembly does not assemble a checkpoint missing from the attribution table).
    """
    for checkpoint in BIREFNET_UPSTREAM:
        if birefnet_checkpoint(checkpoint) == directory:
            return checkpoint
    return None


def birefnet_revision_checkpoint(revision: str) -> str | None:
    """Upstream commit SHA → the checkpoint name whose fixed value is that revision (`None` when
    none matches).

    This is the **first** way `birefnet/export.py` looks up the checkpoint from the download record
    of `--model-dir` (`_shared.upstream.snapshot_revision`). Unlike the name, it looks up by the
    origin of the contents, so renaming the directory does not break the preset rule. The fixed
    value (`revision` of `birefnet.card.BIREFNET_CHECKPOINTS`) is the same value the distribution
    check ({@link birefnet_plan}) compares the container's origin against.
    """
    for checkpoint, entry in BIREFNET_CHECKPOINTS.items():
        if entry.revision == revision:
            return checkpoint
    return None


def birefnet_resolution(model: str) -> int:
    """Model name → the side of the input that should be baked in (**the model name is the
    resolution itself**).

    This is the one place that goes through the accepted set; both the derivation of the series name
    and the cross-check of `pipelineConfig` use this number. If the two paths interpreted the
    spelling separately, "the series is 2048² but the declaration says 1024²" could be built.
    """
    if model not in BIREFNET_MODELS:
        raise DistError(
            f"BiRefNet family model '{model}' is not shipped"
            f" (shipped: {' / '.join(BIREFNET_MODELS)})"
            " — the model name is the shipped input resolution itself, and its series is separate"
        )
    return int(model)


def birefnet_series_name(checkpoint: str, model: str, dtype: str) -> str:
    """Checkpoint, resolution and storage dtype → series directory name.

    The same formula as `birefnet.export.default_out_dir` — f32 has no suffix (the existing series'
    spelling as is), the others get `-<dtype>`.
    """
    if dtype not in BIREFNET_WEIGHT_DTYPES:
        raise DistError(
            f"BiRefNet family storage dtype '{dtype}' is not shipped"
            f" (shipped: {' / '.join(BIREFNET_WEIGHT_DTYPES)})"
        )
    name = birefnet_checkpoint(checkpoint).lower().replace("_", "-")
    suffix = "" if dtype == BIREFNET_PLAIN_DTYPE else f"-{dtype}"
    return f"{name}-{birefnet_resolution(model)}{suffix}"


def birefnet_repo_name(checkpoint: str) -> str:
    """Distribution repository name of the checkpoint ({@link BIREFNET_REPO_NAMES}).

    `Pipeline.repo_name` is a slot that receives the model name, but the checkpoint decides the
    repository, so {@link PIPELINE} / {@link LUCIDA_PIPELINE} bind their own checkpoint and pass it
    here (assembling both resolutions in one pass needs `--out` — `dist.default_out_dir`).
    """
    name = BIREFNET_REPO_NAMES.get(checkpoint)
    if name is None:
        raise DistError(
            f"BiRefNet family checkpoint '{checkpoint}' has no repository name"
            f" (known: {' / '.join(sorted(BIREFNET_REPO_NAMES))})"
        )
    return name


@dataclass(frozen=True)
class BirefnetSources:
    """Inputs of the assembly: only the series per storage dtype (each a single graph).

    Unlike SigLIP2 there is no location of the real weights, because upstream has no
    machine-readable file that could be the source of the preprocessing constants (module doc).
    **Only the series path can say which weights at which resolution**, so the derivation of the
    series name ({@link birefnet_series_name}) is the only link to the attribution.
    """

    #: Storage dtype → series root (all of the checkpoint's presets {@link birefnet_seat_dtypes} —
    #: nothing is assembled if even one is missing, nor with a series of a dtype that is not a
    #: preset).
    series_by_dtype: Mapping[str, Path]


def birefnet_sources(
    series_dir: Path, checkpoint: str, model: str = BIREFNET_DEFAULT_MODEL
) -> BirefnetSources:
    """Look up the inputs from the series parent directory (`outputs/series/`), the checkpoint and
    the resolution.

    Only the series of that checkpoint's presets are looked up ({@link birefnet_seat_dtypes} — only
    the f16 series for HR, both the f32 and f16 series for Lucida). A series that is not a preset
    (HR's f32 series) is ignored even if it is on disk.
    """
    return BirefnetSources(
        series_by_dtype={
            dtype: series_dir / birefnet_series_name(checkpoint, model, dtype)
            for dtype in birefnet_seat_dtypes(checkpoint)
        }
    )


def birefnet_placements(sources: BirefnetSources) -> dict[str, Path]:
    """Role name → source file. The output paths are held by {@link BIREFNET_OUTPUT_PATHS}.

    What is not in this table does not go into the output (this drops the series' `io.*.safetensors`
    / `quality.json`).
    """
    return {
        birefnet_role(dtype): series / "model.krm"
        for dtype, series in sources.series_by_dtype.items()
    }


def birefnet_postprocess(checkpoint: str) -> str:
    """Checkpoint → post-processing procedure ({@link BIREFNET_POSTPROCESS}).

    MUST: do not assemble a checkpoint missing from the table — filling in a default procedure would
    silently produce a distribution that declares a post-processing different from upstream's.
    """
    procedure = BIREFNET_POSTPROCESS.get(checkpoint)
    if procedure is None:
        raise DistError(
            f"BiRefNet family checkpoint '{checkpoint}' has no post-processing procedure"
            f" (known: {' / '.join(sorted(BIREFNET_POSTPROCESS))})"
        )
    if procedure not in BIREFNET_POSTPROCESS_DESCRIPTIONS:
        raise DistError(
            f"the card cannot describe post-processing procedure '{procedure}'"
            f" (known: {' / '.join(BIREFNET_POSTPROCESS_DESCRIPTIONS)})"
        )
    return procedure


def birefnet_pipeline_config(
    graph: Mapping[str, Any], path: Path, model: str, checkpoint: str
) -> dict[str, Any]:
    """Build `pipelineConfig` (the 6 fields of the TS-side schema) from the baked graph, the
    normalization constants and the checkpoint.

    The resize target is the graph's input declaration itself — if the preprocessing stretched to
    other dimensions, the values would break silently while the shapes still fit. The input's name,
    rank, batch and channel count are also checked here (the later {@link assert_birefnet_graph}
    checks the output side).

    MUST: check the resolution read here against the **model name**. The model name is the
    manifest's key and the very value a user spells as `model: "2048"`, so assembling with one
    series mixed up would ship "a 1024 graph calling itself 2048" — it runs and verifies, still
    agrees with the `pipelineConfig` declaration, and only the name disagrees.
    """
    inputs = graph_inputs(graph, path)
    if tuple(inputs) != (BIREFNET_INPUT,):
        raise DistError(
            f"the graph inputs of {path} are {list(inputs)}, not the expected {[BIREFNET_INPUT]}"
            " — the runtime binds by name, so a changed spelling cannot be bound"
        )
    shape = inputs[BIREFNET_INPUT]
    if len(shape) != 4 or shape[0] != 1 or shape[1] != BIREFNET_CHANNELS:
        raise DistError(
            f"{path}: input '{BIREFNET_INPUT}' is {shape!r}"
            f" — expected [1, {BIREFNET_CHANNELS}, H, W] (both batch and channel count static)"
        )
    height, width = shape[2], shape[3]
    for axis, value in (("H", height), ("W", width)):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise DistError(
                f"{path}: in input '{BIREFNET_INPUT}', {axis} is not a positive integer ({value!r})"
            )
    resolution = birefnet_resolution(model)
    if (height, width) != (resolution, resolution):
        raise DistError(
            f"{path} is baked at {height}×{width} — it disagrees with what model '{model}' claims,"
            f" {resolution}² (each resolution is a separate graph and a separate series)"
        )
    return {
        "imageWidth": width,
        "imageHeight": height,
        "imageMean": list(BIREFNET_IMAGE_MEAN),
        "imageStd": list(BIREFNET_IMAGE_STD),
        "interpolation": BIREFNET_INTERPOLATION,
        "postprocess": birefnet_postprocess(checkpoint),
    }


def assert_birefnet_graph(
    graph: Mapping[str, Any], path: Path, pipeline_config: Mapping[str, Any]
) -> None:
    """Measure before placement that the output is **a single matte** with the same dimensions as
    the input.

    MUST: do not drop this. In training mode the wrapper of `birefnet/export.py` also returns the
    intermediate predictions of multi-scale supervision, so for a graph baked that way **a later
    stage that picks by position reads another value as α** (an implementation that only checks the
    element count would let the shape through as well). The same place checks that there are no
    symbolic dimensions — the resolution and the window masks are baked in as constants, and not a
    single axis can move.
    """
    symbols = graph.get("symbols")
    if not isinstance(symbols, list) or symbols:
        raise DistError(
            f"{path}: has symbolic dimensions {symbols!r} — the resolution and the window masks are"
            " baked in as constants"
        )
    outputs = graph.get("outputs")
    if not isinstance(outputs, list) or len(outputs) != 1:
        raise DistError(
            f"{path}: graph outputs are {outputs!r} — exactly one matte from the final stage is"
            " needed"
        )
    values = graph.get("values")
    value = values.get(outputs[0]) if isinstance(values, dict) else None
    shape = value.get("shape") if isinstance(value, dict) else None
    expected = [
        1,
        BIREFNET_MATTE_CHANNELS,
        pipeline_config["imageHeight"],
        pipeline_config["imageWidth"],
    ]
    if shape != expected:
        raise DistError(
            f"{path}: graph output '{outputs[0]}' has shape {shape!r}, expected {expected}"
            " — the matte is 1 channel with the same dimensions as the input"
        )


def birefnet_plan(
    sources: BirefnetSources, checkpoint: str, model: str = BIREFNET_DEFAULT_MODEL
) -> ModelPlan:
    """Build the plan for one BiRefNet family model (= one resolution) (all checks and reads happen
    here)."""
    assert_model_name(model)
    birefnet_resolution(model)
    # A checkpoint that cannot state its attribution fails before any series is read (the repository
    # name table and the attribution table can each be missing an entry independently).
    birefnet_checkpoint(checkpoint)
    birefnet_repo_name(checkpoint)
    birefnet_postprocess(checkpoint)
    seats = birefnet_seat_dtypes(checkpoint)
    default_quant = birefnet_default_quant(checkpoint)
    # MUST: the series are all of that checkpoint's presets, and no dtype that is not a preset — a
    # missing preset would become a preset in declaration only, and a surplus series would ship a
    # preset wider than the upstream storage (one that only widens the values).
    if set(sources.series_by_dtype) != set(seats):
        raise DistError(
            f"storage dtypes of the series {list(sources.series_by_dtype)} differ from checkpoint"
            f" '{checkpoint}''s presets {list(seats)} — the presets are all dtypes not wider than"
            " the upstream storage (birefnet_seat_dtypes)"
        )
    placements = birefnet_placements(sources)
    for role, source in placements.items():
        assert_storage(role, source, BIREFNET_STORAGE_REQUIREMENTS)
        assert_storage_absent(role, source, BIREFNET_STORAGE_FORBIDDEN)
    # MUST: check **every single series of the presets** (the same discipline as irodori). The f16
    # series is emitted by a separate process from f32, so checking only one lets "f32 is as
    # declared but only f16 has another resolution / another origin" through. `pipelineConfig` is
    # derived per series and checked for **equality** with the default preset's series (the
    # dimensions are the baked graph's declaration, so deriving them from one and copying to the
    # others would let series of different dimensions live together).
    configs: dict[str, dict[str, Any]] = {}
    revision = BIREFNET_CHECKPOINTS[checkpoint].revision
    for dtype in seats:
        container = placements[birefnet_role(dtype)]
        graph = ir_graph(container)
        configs[dtype] = birefnet_pipeline_config(graph, container, model, checkpoint)
        assert_birefnet_graph(graph, container, configs[dtype])
        # MUST: check the origin the container states against the attribution table's **fixed
        # values** (license and upstream revision — BIREFNET_CHECKPOINTS of `birefnet.card`) (the
        # same shape as wan). The license (MIT for both HR and Lucida), the dimensions and the
        # preprocessing agree across checkpoints, so without the revision a series of another
        # checkpoint ("a Lucida series on HR's preset", 2 presets cobbled together — 2026-09-26
        # review A-1) would assemble. All containers are checked against the same fixed value, so
        # this also closes the revision agreement between presets.
        assert_upstream_provenance(container, license=BIREFNET_LICENSE, revision=revision)
    pipeline_config = configs[default_quant]
    for dtype, config in configs.items():
        if config != pipeline_config:
            raise DistError(
                f"the {dtype} storage series' pipelineConfig {config} disagrees with the default"
                f" preset {default_quant}'s series' {pipeline_config} — presets of"
                " the same model are baked with the same dimensions and preprocessing"
            )
    weights = birefnet_weights(checkpoint)
    return ModelPlan(
        name=model,
        pipeline=BIREFNET_PIPELINE,
        artifacts={
            role: Artifact(BIREFNET_OUTPUT_PATHS[role], source=source)
            for role, source in placements.items()
        },
        weights=weights,
        assets=BIREFNET_ASSETS,
        quants=complete_quant_weights(weights, birefnet_quants(checkpoint)),
        default_quant=default_quant,
        pipeline_config=pipeline_config,
    )


def birefnet_dist_plan(series_dir: Path, model: str, checkpoint: str) -> ModelPlan:
    """Build the plan for one BiRefNet family model (= one resolution) from the `--series` parent
    (where the CLI dispatches to).

    MUST: `checkpoint` is **the only weights that Pipeline's repository ships** and is not
    determined by `--model` (a substitution of the same discipline as anima). The series path, the
    attribution and the copyright notice at the repository root are all determined by this one name,
    so a mix-up across entries — a distribution coming together while the copyright notice and
    modification notice at the repository root disagree with its contents — cannot even be spelled.
    """
    return birefnet_plan(birefnet_sources(series_dir, checkpoint, model), checkpoint, model)


#: MIT copyright lines (`LICENSE.md` at the root of the distribution repository). Both upstream
#: repositories state `license: mit` but **do not bundle the license text** (only the HF card
#: frontmatter), so the text is built from the shared template (`_shared.licenses.mit_license`) and
#: the holder lines live here.
#:
#: MUST: Lucida has 2 lines (the fine-tune's and the upstream's). MIT's "include the copyright
#: notice and permission notice" means a derivative cannot drop the upstream's notice either, and
#: the upstream card itself declares "the original copyright notice is preserved".
#:
#: NOTE: upstream does not state the years (neither repository has a `LICENSE` file). For ZhengPeng
#: the year of the BiRefNet paper (2024) was taken, for egeorcun the year the upstream card itself
#: gives the current weights (2026) — the manual license review before release (ADR 0065 decision 7)
#: has the final say.
BIREFNET_COPYRIGHTS: Mapping[str, tuple[str, ...]] = {
    BIREFNET_HR_CHECKPOINT: ("Copyright (c) 2024 ZhengPeng",),
    BIREFNET_LUCIDA_CHECKPOINT: (
        "Copyright (c) 2026 egeorcun",
        "Copyright (c) 2024 ZhengPeng",
    ),
}

#: Modification notice. MIT does not require one, but that this distribution moved only the
#: container without changing the storage form is the fact a user wants to check first, so one page
#: is shipped in the same place as `LICENSE.md`.
#:
#: MUST: the wording matches the distribution's contents — it is plausible prose as a value, so it
#: passes `verify_dist` and the manifest checks, and a disagreement could not be noticed until after
#: shipping. The per-preset storage description (`{storage}`) is built from the same single place as
#: the card's attribution section (`birefnet.card.birefnet_storage_lines`), from that checkpoint's
#: presets ({@link birefnet_seat_dtypes}) and storage.
BIREFNET_NOTICE_TEMPLATE = """# NOTICE

This repository redistributes a modified form of `{repo}`, which is licensed under
the MIT License (see `LICENSE.md`). The following changes were made:

- The graph was re-expressed in the Karume container format (a `.krm` part sequence whose
  first part carries the graph and model descriptors, followed by the weight parts they name).
- The upstream `forward` was rewritten layout-only and **bit-exact**: windowing, the
  shifted-window roll, spatial padding and the patch merges became equivalent operations.
- Two modules were rewritten in a form that is equivalent up to floating-point rounding:
  inference-time `BatchNorm2d` became a per-channel affine, and the ASPP image-level pooling
  became a two-stage sum.
- The decoder tail's 1×1 convolution and the bilinear upsample it used to follow were swapped
  in order. Both are linear on disjoint axes, so they commute; the rewrite is equivalent up to
  floating-point rounding and removes two full-resolution intermediates.
{storage}

No retraining and no fine-tuning were performed. The original checkpoint is not distributed here.
"""


def birefnet_root_files(checkpoint: str) -> dict[str, str]:
    """Legal texts placed at the root of the distribution repository
    (`karume.dist.Pipeline.root_files`).

    MIT only asks that "the copyright notice and permission notice be included in all copies", so
    `LICENSE.md` is the shared template's text + that repository's copyright lines
    ({@link BIREFNET_COPYRIGHTS}). The text itself is not touched (the moment it is reformatted, it
    stops being a copy of the permission notice).
    """
    return {
        "LICENSE.md": mit_license(BIREFNET_COPYRIGHTS[checkpoint]),
        "NOTICE.md": BIREFNET_NOTICE_TEMPLATE.format(
            repo=BIREFNET_UPSTREAM[checkpoint],
            storage="\n".join(birefnet_storage_lines(checkpoint, birefnet_seat_dtypes(checkpoint))),
        ),
    }


def _birefnet_pipeline(checkpoint: str) -> Pipeline:
    """Build the Pipeline for 1 checkpoint = 1 distribution repository (both resolutions live inside
    it).

    Binding the checkpoint **here, exactly once** is all this function does — the series path
    (`plan`), the repository name, the copyright lines at the repository root (`root_files`) and the
    card's attribution (`card_profiles`) all come from the same one spelling, so a shape where only
    one of them points at another checkpoint cannot be built.
    """
    return Pipeline(
        default_model=BIREFNET_DEFAULT_MODEL,
        repo_name=lambda _model: birefnet_repo_name(checkpoint),
        plan=lambda series_dir, model: birefnet_dist_plan(series_dir, model, checkpoint),
        # Not an axis to choose, for the same reason as SigLIP2 — the attribution (upstream
        # repository, license, training data) is determined uniquely by the checkpoint. Splitting
        # the profiles would let an operator make the mix-up "ship Lucida with BiRefNet_HR's
        # attribution".
        card_profiles={"birefnet": partial(render_birefnet_model_card, checkpoint=checkpoint)},
        root_files=birefnet_root_files(checkpoint),
    )


#: The `--pipeline birefnet` row (distribution repository `karume-birefnet-hr`).
#:
#: MUST: a **separate Pipeline** from the derivative (`lucida`) — the reason is the same paragraph
#: of the module doc.
PIPELINE = _birefnet_pipeline(BIREFNET_HR_CHECKPOINT)

#: The `--pipeline lucida` row (distribution repository `karume-lucida` — a third-party fine-tune of
#: BiRefNet_HR). The template is the same (the structure is identical, so the distribution has the
#: same form); only the contents and the legal texts at the repository root differ.
LUCIDA_PIPELINE = _birefnet_pipeline(BIREFNET_LUCIDA_CHECKPOINT)
