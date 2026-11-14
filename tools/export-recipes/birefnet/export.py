"""Script that writes a real-weight BiRefNet family model (background removal / salient object
segmentation) out to IR (docs/ir-v2.md) + container (krm) + golden io. The model is chosen with
`--model-dir` (default BiRefNet_HR — the "model axis" below).

The outputs go under `outputs/series/` and are kept out of commits by the repository root's
`.gitignore` (weights + baked constants come to 964MB at 1024²).

    uv run --group birefnet python -m birefnet.export
    uv run --group birefnet python -m birefnet.export --verify  # equivalence with pre-patch eager

transformers is **pinned at 5.14.1** (because `birefnet.patch` replaces class attributes of the
`trust_remote_code` dynamic module — the `birefnet` group of pyproject.toml).

## The model axis and the resolution axis

The series name is the directory name of `--model-dir` (lower-cased, hyphenated) **with
`--resolution` appended** (and, for anything but f32, the storage dtype suffix — the default is
`birefnet-hr-1024-f16`). The resolution is part of the spelling because a BiRefNet series is a
**separate graph** per resolution — the shifted-window masks and the zero constants of the H/W
padding are baked as resolution-dependent constants, so writing 2 resolutions into the same slot
would silently overwrite the earlier assets.

    uv run --group birefnet python -m birefnet.export --resolution 2048

The model axis accepts two (the attribution table's source of truth is
`birefnet.card.BIREFNET_UPSTREAM`):

- `inputs/birefnet/BiRefNet_HR/` — upstream `ZhengPeng7/BiRefNet_HR` (MIT). The default.
- `inputs/birefnet/lucida/` — upstream `egeorcun/lucida` (MIT). A fine-tune of BiRefNet_HR whose
  bundled `birefnet.py` / `BiRefNet_config.py` are **byte-identical** (measured — only the weights
  differ). So `birefnet.patch`'s replacements and `--verify`'s two-point evaluation work as they
  are.

      uv run --group birefnet python -m birefnet.export --model-dir <repo inputs>/birefnet/lucida

MUST: lucida's `lucida-m35-comfy.safetensors` is **out of scope** (exclude it from `hf download`).
As the upstream model card states, those are weights with `Normalize` **folded into** the first conv
for ComfyUI, which would apply it twice together with this script's preprocessing (ImageNet
normalization before the graph). The measurement agrees: the per-input-channel RMS of
`bb.patch_embed.proj.weight` relative to v7 is 4.352 / 4.453 / 4.428 = the reciprocal of {@link
IMAGENET_STD} (4.367 / 4.464 / 4.444) within 0.3%, and subtracting `Σ_c mean_c · Σ_hw w[c]` from the
bias brings it back to v7's bias distribution (mean −0.0197 / standard deviation 0.2790 / absolute
max 1.4296) within 0.2%. It could be accepted by making the presence of preprocessing a
`pipelineConfig` axis, but ① they are weights from an experimental branch that upstream describes as
"for use inside the pipeline (with the poster policy + SAM3 referee); bare use not recommended",
which a single-graph distribution cannot reproduce, and ② branching the distribution's preprocessing
for it would make every model carry 2 kinds of preprocessing — so it is **not accepted**.

**The default is 1024².** The original's inference resolution (General-HR in the bundled
`handler.py`) is 2048². The 2048² decoder tail had 2 intermediates above the 2GiB binding limit
(`[1, 192, 2048, 2048]` = 3.22GB and `[1, 240, 2048, 2048]` = 4.03GB), but `birefnet.patch`'s ⑨
(swapping the order of the 1×1 conv and the bilinear upsample) removes both. 2048² is shipped
together with karume's allocation (ADR 0093) (in one repository with the default 1024²), but on
Intel Arc B570 (Linux xe) one dispatch exceeds the 5-second job limit and the device is lost. The
constraint's source of truth is the section "BiRefNet family distributions are 2 models, 1024² and
2048²" of docs/limitations.md.

`--resolution` accepts **multiples of 64** only: the body's `PatchMerging` requires an even H/W at
each stage (S/4, S/8, S/16 even = S%32), and the half-resolution branch of `mul_scl_ipt='cat'` puts
the same requirement on S/2 (= S%64). Dimensions outside that fail here rather than with a reshape
error midway.

## Storage dtype series (ADR 0113 — phase 1 goes up to f16)

`--dtype f16` writes to a **separate series** (a spelling with `-f16` appended, like
`lucida-1024-f16/`). Sharing one would silently apply the f32 series' net (tolerances measured per
series) to the compressed assets, so the series are always separate (the same reason as ADR 0018 /
0027).

The writable dtypes and the default follow **the same rule** as the distribution's presets
({@link export_dtypes} — ADR 0113 addendum 2026-10-11): no dtype wider than the upstream
checkpoint's storage is written, and the default is the same type as the storage. BiRefNet_HR
(upstream f16) has only f16 (default f16; `--dtype f32` is refused — it only widens the f16 values
to f32, adds no information and is not shipped either); Lucida (upstream f32) has f32 (default) and
f16.

    uv run --group birefnet python -m birefnet.export --real-images
    uv run --group birefnet python -m birefnet.export --resolution 2048
    uv run --group birefnet python -m birefnet.export --model-dir <repo inputs>/birefnet/lucida \
        --dtype f16 --real-images

MUST: rounding (fake-quant — ADR 0006) happens **right after** `load_model` and **before**
`patch.apply` ({@link load_wrapper}). ⑦'s `ChannelAffine` is a derived constant whose α / β are
computed in f32 from the BatchNorm statistics, and α / β are inputs of mul / add (not weight slots),
so they are stored as f32. Rounding **after** apply would move α / β to f16 values as well, while
their storage stays f32 — not one byte of VRAM is saved and only the numbers move. Rounding before
apply derives α / β in f32 from the rounded statistics.

The upstream BiRefNet_HR checkpoint is **f16** (687 F16 / 0 F32 in the safetensors header), and
`load_model` only widens it to f32, so for HR's f16 series the rounding is the identity and it is
**lossless relative to upstream**. Lucida is f32, so it is lossy as usual.

**The quality-measurement harness** (compressed series only): in the same process, the f32 outputs
(the patched graph, 8 cases: 4 synthetic + 4 real images) are taken **before** rounding, and the
difference from the rounded weights' outputs is written to the series directory's `quality.json`
({@link measure_quality}). The real images go through regardless of `--real-images` (the golden's
condition "the resize is the identity" is not needed for a quality measurement where the baseline
and the compared side see the same input). If every floating-point value of the checkpoint is f16
({@link lossless_under}), the difference must be **exactly 0**, and otherwise it fails without
publishing ({@link assert_quality_gate} — a check against the harness going trivially true that
catches regressions of the rounding position). For a lossy checkpoint it fails when the mean
absolute α difference is at or above the provisional line {@link ALPHA_MAE_LIMIT} (0.5 LSB of the
8-bit α) (the kill criterion of deep dive E1 §6, ADR 0113 decision 5).

## What goes into the graph

`BiRefNet.forward` returns a **list**. Its elements are the intermediate predictions of multi-scale
supervision (m4 / m3 / m2) and the final matte, and **the intermediate predictions are appended only
inside the `self.training` branch** — so in eval there is 1 element (measured `len(preds) == 1` /
shape `[1, 1, S, S]`). The bundled `handler.py` also uses only `preds[-1]`. So the wrapper returns
`preds[-1]` and fails loudly when the element count is not 1 ({@link MatteLogits}) — a training-mode
graph is not written out silently.

The output is **the logits before the sigmoid** (the handler's `.sigmoid()` is the host's job). Two
reasons: keeping the graph a verbatim copy of the eager forward makes `--verify`'s cross-check
directly a claim about the execution stage, and the sigmoid flattens differences in the saturated
range, so applying it to the golden would lower the sensitivity to numeric regressions.

batch is **statically 1** (the window-mask addition of `birefnet.patch` is equivalent to the
original implementation only for B = 1). There are no dynamic axes (`symbol_names=()`).

MUST: the model is exported after `birefnet.patch`'s patches are applied. Without them the
conversion fails on Swin's rank-6 reshape / `roll` / step-2 slice and the like (the list of
replacements is in `birefnet.patch`'s docstring). The golden's expected values are taken from the
module **after** the replacement — the difference before and after the replacement is the patch
layer's responsibility, measured by `--verify`'s two-point evaluation.

MUST: run `patch.prepare(...)` after `apply` ({@link load_wrapper}). The shifted-window mask is a
resolution-dependent constant, baked only in this one eager forward.

MUST: `--verify` does not emit (the two cannot be combined in one process). The patches replace
class attributes process-wide, so after applying them for emit, "the reference before applying" can
no longer be taken. For the same reason it is exclusive with `--dtype` too (the same structural
reason as ADR 0027 decision 3 — `--verify`'s reference is the eager pass with the pre-patch f32
weights, and a reference with rounded weights is not used for that purpose).

## The input contract (preprocessing stays out of the graph)

The graph input is the **normalized** `pixel_values f32 [1, 3, S, S]`. Image decode / resize
(bilinear to `(S, S)`) / rescale (1/255) / normalize are karume's job; the constants' source of
truth is the bundled `handler.py`'s `ImagePreprocessor` (ImageNet statistics = {@link IMAGENET_MEAN}
/ {@link IMAGENET_STD}). **These are different statistics from SigLIP2 (mean = std = 0.5)**, so they
are held here rather than shared.

## The 2 golden groups (synthetic + real images) — both are kept

The default 4 cases are **synthetic images** ({@link build_cases}). `disc` is a **salient object**,
a bright disc on a dark background, and {@link _sanity} checks the **order** "mean logit inside the
disc > mean logit outside" (no threshold) — an output that is dead as a segmentation (uniform,
input-independent) fails here.

With `--real-images`, a golden through the 4 **real images** of `outputs/misc/corpus/`
({@link REAL_CASES}) is written in addition:

    uv run --group birefnet python -m birefnet.export --real-images

Its claim is the cross-check "TS preprocessing + karume inference" against "Python preprocessing +
torch inference", and the Python side is a verbatim copy of the bundled `handler.py`'s
`ImagePreprocessor` ({@link build_real_cases}). The discrimination is also on the real-image side —
**the foreground ratio of the 2 images with a person is larger than that of either of the 2
without** ({@link _real_sanity}, order only). The synthetic `disc` discriminates geometry and the
real images meaning; neither replaces the other.

NOTE: the real images are 1024×1024, so for the 1024² series the resize is the identity (of the
preprocessing, what this check covers is the normalization and the inference chain after it).
Parity of the resize itself is covered by `packages/models/tests/image_preprocess_test.ts`.

MUST: the real-image golden carries **the sha256 of the source image** in `__metadata__`. The images
are outputs baked by `examples/anima/eval-images.ts` and **frozen-copied by hand** into
`outputs/misc/corpus/`, so in an environment where the corpus was replaced but the golden was not
re-taken, the cross-check is not a cross-check but **different inputs** (and the values come out
looking plausible).

NOTE: `--real-images` also runs in the same `birefnet` group (SigLIP2 splits off
`siglip2-preprocess` because its export does not need torchvision — here torchvision is a base
dependency for `deform_conv2d`, and preprocessing adds only pillow, to open the PNGs).

## Output layout

    outputs/series/<series name>/model.krm
        weights and constants + descriptions of the 2 documents
    outputs/series/<series name>/io.<case>.safetensors
        input and torch CPU expected output
    outputs/series/<series name>/quality.json
        difference from the f32 outputs (compressed series only)

The tensor key convention of io is the same as tiny golden / DeBERTa / SigLIP2
(`input.<graph input name>` / `output.<position>`).
"""

from __future__ import annotations

import argparse
import hashlib
import io as io_module
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch import nn

from _shared.paths import INPUTS_ROOT, MISC_ROOT, SERIES_ROOT
from _shared.upstream import UpstreamProvenanceError, snapshot_provenance, snapshot_revision
from karume.artifacts import staged_publication
from karume.container import Provenance, container_parts
from karume.convert import normalize_boundary_tensor
from karume.dist import NOTICE_FILENAME
from karume.emit import storage_breakdown
from karume.ir import IrGraph
from karume.pipeline import export_to_file
from karume.quantize import RoundReport, round_weights_to_f16

from . import patch
from .distribution import (
    BIREFNET_ROLE,
    birefnet_default_quant,
    birefnet_directory_checkpoint,
    birefnet_revision_checkpoint,
    birefnet_seat_dtypes,
)

#: Parent of the real weights (where HF's 7 files are unpacked into `inputs/birefnet/<name>/`).
MODELS_ROOT = INPUTS_ROOT / "birefnet"

#: Location of the real-image corpus (a frozen copy = host assets, so deleting it means re-baking —
#: docs/assets-layout.md).
CORPUS_ROOT = MISC_ROOT / "corpus"

#: Default model (when `--model-dir` is not given).
DEFAULT_MODEL_DIR = MODELS_ROOT / "BiRefNet_HR"

#: Default resolution ("The model axis and the resolution axis" in the module docstring).
DEFAULT_RESOLUTION = 1024

#: Resolution step (S%32 = the body's PatchMerging / S%64 = the same requirement of the
#: half-resolution branch).
RESOLUTION_MULTIPLE = 64

MODEL_FILE = "model.krm"

#: Storage dtypes that can be written (phase 1 = up to f16 — ADR 0113). For i8 a copy is not enough
#: (the window attention's relative-position table and TailConv are not `nn.Linear` / `nn.Conv2d`
#: modules and get no scale; deform's `regular_conv` is rounded but stored in f32), so it is added
#: in phase 2 together with a BiRefNet-specific i8 plan.
WEIGHT_DTYPES = ("f32", "f16")

#: dtype of the uncompressed series (the only one without a suffix in the series name).
PLAIN_DTYPE = "f32"

#: Weights file of the upstream checkpoint (directly under where `hf download` unpacks — the census
#: of storage dtypes is read from it).
CHECKPOINT_FILE = "model.safetensors"

#: Record of the quality measurement (directly under the series directory — "the quality-measurement
#: harness" in the module docstring).
QUALITY_FILE = "quality.json"

#: Number of α steps of the product (`ALPHA_MAX` of `packages/models/src/birefnet/pipeline.ts` — 8
#: bit). The quality measurement also counts differences on the sigmoid α rounded to these steps
#: (what the user receives is an 8-bit α).
ALPHA_MAX = 255

#: Quality check of lossy series: upper bound of the mean absolute difference of α (after the
#: sigmoid) = **0.5 LSB** of the 8-bit α (0.5 / 255 ≈ 1.96e-3). At or above it, "on average it moves
#: by more than half an 8-bit step" — a **provisional line** placed at the entrance to degradation
#: visible in the product's 8-bit α (the kill criterion of deep dive E1 §6, ADR 0113 decision 5).
#: Lucida f16's measurement (2026-09-26: 7.0e-5 at 1024², 4.8e-5 at 2048²) has nearly 30× headroom.
#: Revisited with the phase 2 (i8) measurements.
ALPHA_MAE_LIMIT = 0.5 / ALPHA_MAX

IO_PREFIX = "io."
IO_SUFFIX = ".safetensors"
INPUT_PREFIX = "input."
OUTPUT_PREFIX = "output."

#: Name of the graph input (= the parameter name of {@link MatteLogits.forward}). Checked after
#: export and stops on a mismatch — since inputs are passed by position, a silent mismatch would
#: swap only the golden's inputs.
INPUT_NAME = "pixel_values"

#: Normalization constants of the preprocessing (source of truth = the bundled `handler.py`'s
#: `ImagePreprocessor`).
IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)

#: Random seed of the synthetic images (the `noise` case). Independent of the global seed.
SEED = 20260813

#: Radius of the disc in the `disc` case (as a fraction of the image's shorter side).
DISC_RADIUS = 0.3

#: Foreground / background colors of the `disc` case (RGB, range `[0, 1]`). **The 3 channels have
#: different values**, so a mixed-up channel order (statistics in the wrong order) shows in the
#: values.
DISC_FOREGROUND: tuple[float, float, float] = (0.92, 0.78, 0.35)
DISC_BACKGROUND: tuple[float, float, float] = (0.12, 0.16, 0.24)

#: One square of the `checker` case's checkerboard (pixels). A width not divisible by Swin's window
#: (12), so that the grid and the windows are not in phase by chance (`40 % 12 == 4`). It is
#: divisible by the patch (4) = in phase with the patch grid, so that coincidence is not avoided —
#: the `disc` case does the discrimination.
CHECKER_CELL = 40

#: Name of the case {@link _sanity} uses to discriminate (the one image with a salient object).
DISC_CASE = "disc"

#: Real-image cases = (case name, file name in `outputs/misc/corpus/`, content). The images were
#: baked by `examples/anima/eval-images.ts` (`deno task demo:eval-images`, which held the
#: prompts / seeds / resolution — removed in 0.15 with no replacement), and the adopted ones were
#: frozen-copied by hand. The case names are spelled like SigLIP2's `REAL_CASES` (the same 4
#: images, read from a different series directory).
REAL_CASES: tuple[tuple[str, str, str], ...] = (
    (
        "photo-portrait",
        "anima-default-1024x1024-defaultstep-seed42.png",
        "close-up of a person + row of cherry trees (clear foreground and background)",
    ),
    (
        "photo-landscape",
        "anima-default-1024x1024-defaultstep-seed43.png",
        "landscape (mountains, lake, forest, no people)",
    ),
    (
        "photo-corridor",
        "anima-default-1024x1024-defaultstep-seed44.png",
        "school corridor (strong perspective, vanishing point)",
    ),
    (
        "photo-street",
        "anima-default-1024x1024-defaultstep-seed45.png",
        "full-body person + street (busy background)",
    ),
)

#: The 2 groups the real-image discrimination looks at (salient object = the 2 images with a person
#: / the 2 without). {@link _real_sanity} checks **the order of the foreground ratios**. The TS
#: side's `packages/models/tests/e2e_birefnet_real_test.ts` applies the same groups to real GPU
#: outputs.
REAL_PERSON_CASES = ("photo-portrait", "photo-street")
REAL_SCENE_CASES = ("photo-landscape", "photo-corridor")

#: Fields of the real-image golden's `__metadata__` (identifying the source image — the module
#: docstring's MUST).
SOURCE_IMAGE_KEY = "source_image"
SOURCE_SHA256_KEY = "source_sha256"


def default_out_dir(model_dir: Path, resolution: int, dtype: str) -> Path:
    """Default location of the outputs (`outputs/series/<model name>-<resolution>{,-f16}/`).

    **Both** the model name and the resolution are baked into the series' spelling ("The model axis
    and the resolution axis" in the module docstring). HF spells it `BiRefNet_HR`, but the series
    name follows the existing spelling (lower case, hyphens) — everything in `outputs/series/` is
    named by the same convention.

    MUST: a separate directory per storage dtype (f32 has no suffix — the existing series' spelling
    as is). Sharing one would silently apply the f32 series' net to the compressed assets ("Storage
    dtype series" in the module docstring). The spelling agrees with
    `birefnet.distribution.birefnet_series_name` (the writer and the reader build it with the same
    formula).
    """
    suffix = "" if dtype == PLAIN_DTYPE else f"-{dtype}"
    return SERIES_ROOT / f"{model_dir.name.lower().replace('_', '-')}-{resolution}{suffix}"


def model_dir_checkpoint(model_dir: Path) -> str | None:
    """Which checkpoint the weights of `--model-dir` are (an attribution-table key, `None` when not
    in the attribution table).

    Lookup order:

    1. If the revision of the download record ({@link _shared.upstream.snapshot_revision}) matches a
       fixed value of the attribution table (`revision` of `birefnet.card.BIREFNET_CHECKPOINTS`),
       that checkpoint.
    2. When there is no download record or it matches no fixed value, the directory name (the same
       link as the series name — `birefnet.distribution.birefnet_directory_checkpoint`).

    The revision comes first because looking up by name alone would let an HR with a renamed
    directory fall to the experimental checkpoint, breaking the preset rule and silently writing the
    retired f32 series.

    A revision outside the fixed values combined with a rename is treated as experimental (a
    checkpoint not in the attribution table) — the distribution check
    (`birefnet.distribution.birefnet_plan`) refuses it by the container's revision.

    MUST: fail when 1 and 2 point at **different** checkpoints — writing with the contents
    (revision) and the name (spelling of the series name) disagreeing would put one checkpoint's
    weights under another checkpoint's series name.

    An unreadable record (the first line is not a commit SHA) falls to 2, like "no record" — the
    origin baked into the container is read again from the same record by {@link export_series},
    which fails loudly.
    """
    try:
        by_revision = birefnet_revision_checkpoint(snapshot_revision(model_dir))
    except UpstreamProvenanceError:
        by_revision = None
    by_name = birefnet_directory_checkpoint(model_dir.name)
    if by_revision is not None and by_name is not None and by_revision != by_name:
        raise SystemExit(
            f"{model_dir}: the download record's revision is the fixed value of checkpoint"
            f" '{by_revision}', but the directory name spells checkpoint '{by_name}' — the contents"
            " and the series name disagree (make the directory name the tail of the upstream HF"
            " repository name)"
        )
    return by_name if by_revision is None else by_revision


def export_dtypes(model_dir: Path) -> tuple[tuple[str, ...], str]:
    """Storage dtypes that can be written for the checkpoint of `--model-dir`, and the default (the
    same rule as the distribution's presets).

    The checkpoint is looked up from the download record's revision, or else the directory name
    ({@link model_dir_checkpoint}). The rule's source of truth is
    `birefnet.distribution.birefnet_seat_dtypes` / `birefnet_default_quant` (no dtype wider than the
    upstream storage; the default is the same type as the storage), and this only calls the same
    functions — if the writer and the distributor held the rule separately, it would bake series
    that cannot be shipped, or fail to bake the series of a shipped preset.

    A checkpoint not in the attribution table (an experimental `--model-dir`) has no input to the
    rule (the upstream storage) in the table, so it writes the whole vocabulary with an f32 default
    (the distribution assembly refuses that series as a checkpoint not in the attribution table).
    """
    checkpoint = model_dir_checkpoint(model_dir)
    if checkpoint is None:
        return WEIGHT_DTYPES, PLAIN_DTYPE
    return birefnet_seat_dtypes(checkpoint), birefnet_default_quant(checkpoint)


def upstream_provenance(model_dir: Path) -> Provenance:
    """Derive the origin baked into the container (container-v1 §2.3) from what is actually in
    `--model-dir`.

    The license identifier comes from the README's front matter, the revision from the download
    record of `hf download --local-dir` ({@link _shared.upstream.snapshot_provenance}). Series
    migration (`migrate_series`) calls the same function.

    It is not a constant because then a container baked from another checkpoint would also claim the
    attribution table's license. The cross-check against the attribution table (the card side's
    source of truth) lives on the assembly side (`birefnet.distribution`).
    """
    return snapshot_provenance(model_dir, notice=NOTICE_FILENAME)


class MatteLogits(nn.Module):
    """Export wrapper that returns only the final matte (the logits before the sigmoid,
    `[B, 1, S, S]`).

    MUST: fail when the element count is not 1 ("What goes into the graph" in the module docstring)
    — a model with `self.training` set prepends the intermediate predictions of multi-scale
    supervision, so silently taking `[-1]` would hide that "a training-mode graph was written out as
    an inference graph".
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        predictions = self.model(pixel_values)
        if len(predictions) != 1:
            raise ValueError(
                f"forward returned {len(predictions)} outputs (BiRefNet in eval returns 1) —"
                " a training-mode graph is not written out"
            )
        return predictions[-1]


def normalize_image(image: torch.Tensor) -> torch.Tensor:
    """Normalize a `[1, 3, S, S]` in `[0, 1]` with the ImageNet statistics (verbatim from
    handler.py)."""
    mean = torch.tensor(IMAGENET_MEAN).reshape(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).reshape(1, 3, 1, 1)
    return (image - mean) / std


def disc_mask(size: int) -> torch.Tensor:
    """Inside of the `disc` case's disc (bool `[S, S]`). {@link _sanity} uses the same formula."""
    axis = (torch.arange(size, dtype=torch.float32) + 0.5) / size * 2.0 - 1.0
    squared = axis.reshape(size, 1) ** 2 + axis.reshape(1, size) ** 2
    return squared <= DISC_RADIUS**2


def _disc_image(size: int) -> torch.Tensor:
    """`[1, 3, S, S]` with a bright disc (salient object) on a dark background."""
    foreground = torch.tensor(DISC_FOREGROUND).reshape(3, 1, 1)
    background = torch.tensor(DISC_BACKGROUND).reshape(3, 1, 1)
    return torch.where(disc_mask(size), foreground, background).unsqueeze(0)


def _ramp_image(size: int) -> torch.Tensor:
    """Diagonal ramp from top left to bottom right (phase shifted per channel)."""
    axis = torch.linspace(0.0, 1.0, size)
    plane = (axis.reshape(1, size) + axis.reshape(size, 1)) / 2.0
    shifts = torch.tensor([0.0, 0.15, 0.3]).reshape(3, 1, 1)
    return ((plane.unsqueeze(0) + shifts) % 1.0).unsqueeze(0)


def _checker_image(size: int) -> torch.Tensor:
    """Checkerboard ({@link CHECKER_CELL}-pixel squares). Light and dark are inverted per
    channel."""
    index = torch.arange(size) // CHECKER_CELL
    parity = ((index.reshape(1, size) + index.reshape(size, 1)) % 2).to(torch.float32)
    channels = torch.stack([parity, 1.0 - parity, parity * 0.5 + 0.25])
    return channels.unsqueeze(0)


def build_cases(resolution: int) -> tuple[tuple[str, torch.Tensor], ...]:
    """`(name, pixel_values)` of the 4 golden cases (**normalized** synthetic images).

    The 4 synthetic cases are for sensitivity to numeric regressions. `--real-images` adds the real
    images. Only {@link DISC_CASE} is given meaning as a salient object, and it is the ground that
    keeps {@link _sanity}'s discrimination from being trivially true (the other 3 are on the "no
    salient object" side).
    """
    generator = torch.Generator().manual_seed(SEED)
    noise = torch.rand((1, 3, resolution, resolution), generator=generator)
    return (
        (DISC_CASE, normalize_image(_disc_image(resolution))),
        ("ramp", normalize_image(_ramp_image(resolution))),
        ("checker", normalize_image(_checker_image(resolution))),
        ("noise", normalize_image(noise)),
    )


def build_real_cases(
    resolution: int, corpus_root: Path
) -> tuple[tuple[str, torch.Tensor, dict[str, str]], ...]:
    """`(name, pixel_values, __metadata__)` of the 4 real images ({@link REAL_CASES}).

    The preprocessing is **verbatim** from the bundled `handler.py`'s `ImagePreprocessor` (`Resize`
    → `ToTensor` → `Normalize`). It is not imported because `handler.py` loads cv2 / requests, which
    have nothing to do with inference, at the top of the module — the source of truth of the 3
    copied steps is there, and that of the values is {@link IMAGENET_MEAN} / {@link IMAGENET_STD}.

    A PIL image is passed to `Resize` to take the same path as the handler (torchvision's PIL branch
    = Pillow's `ImagingResample`). The TS side's `resizeRgb8` stretches with the same support, but
    the real images are 1024×1024 and the resize is the **identity** for the 1024² series, so no
    resize difference reaches this check (the module docstring's NOTE).

    MUST: stop **before** the heavy imports if even one image is missing (`--real-images` is an
    explicit request, so silently writing 3 images would hide the missing golden).
    """
    paths: list[tuple[str, str, Path]] = []
    for name, file_name, _why in REAL_CASES:
        path = corpus_root / file_name
        if not path.is_file():
            raise SystemExit(
                f"real image {path} is missing (bake it: deno task demo:eval-images"
                " --source <path of the Anima distribution> → frozen-copy the adopted ones into"
                " outputs/misc/corpus/)"
            )
        paths.append((name, file_name, path))

    from PIL import Image
    from torchvision import transforms

    transform = transforms.Compose(
        [
            transforms.Resize((resolution, resolution)),
            transforms.ToTensor(),
            transforms.Normalize(list(IMAGENET_MEAN), list(IMAGENET_STD)),
        ]
    )
    expected_shape = (1, 3, resolution, resolution)
    cases: list[tuple[str, torch.Tensor, dict[str, str]]] = []
    for name, file_name, path in paths:
        raw = path.read_bytes()
        with Image.open(io_module.BytesIO(raw)) as image:
            pixel_values = transform(image.convert("RGB")).unsqueeze(0)
        if tuple(pixel_values.shape) != expected_shape:
            raise AssertionError(
                f"pixel_values of {name} is {tuple(pixel_values.shape)}"
                f" (not the expected {expected_shape})"
            )
        cases.append(
            (
                name,
                pixel_values,
                {
                    SOURCE_IMAGE_KEY: file_name,
                    SOURCE_SHA256_KEY: hashlib.sha256(raw).hexdigest(),
                },
            )
        )
    return tuple(cases)


def assert_resolution(resolution: int) -> None:
    """Check that the resolution is on the step ("resolution axis" in the module docstring)."""
    if resolution <= 0 or resolution % RESOLUTION_MULTIPLE:
        raise SystemExit(
            f"--resolution {resolution} is not a multiple of {RESOLUTION_MULTIPLE}"
            " (PatchMerging requires an even H/W at each stage — steps of 64 including the"
            " half-resolution branch)"
        )


def load_model(model_dir: Path) -> nn.Module:
    """Load the real weights and check the configuration the replacement version assumes (before
    the patches).

    `trust_remote_code=True` is there to load the bundled `birefnet.py` (as the HF model card says).
    """
    from transformers import AutoModelForImageSegmentation

    model = AutoModelForImageSegmentation.from_pretrained(
        str(model_dir), trust_remote_code=True, dtype=torch.float32
    )
    model.eval()
    # Check the configuration before the heavy forward (fail without baking anything if the
    # replacement version does not cover it).
    patch.assert_supported(model)
    return model


def round_to_storage(model: nn.Module, dtype: str) -> RoundReport | None:
    """Round the weights to values representable in the storage dtype (f32 does nothing —
    fake-quant, ADR 0006).

    f16 rounds every parameter and every f32 buffer (`karume.quantize.round_weights_to_f16`). The
    ineligible ones (biases / BatchNorm statistics) are rounded in value too, but their storage
    stays f32 — emit's eligibility check makes only the weight slots f16. The values are rounded so
    that the correspondence of the golden and the GPU computing **the same numbers** (ADR 0006) also
    holds on the ineligible side.

    MUST: called from exactly one place, inside {@link load_wrapper} (right after `load_model`,
    before `patch.apply` — "Storage dtype series" in the module docstring).
    """
    if dtype not in WEIGHT_DTYPES:
        raise SystemExit(f"cannot write storage dtype '{dtype}' ({' / '.join(WEIGHT_DTYPES)})")
    if dtype == PLAIN_DTYPE:
        return None
    report = round_weights_to_f16(model)
    if report.parameters + report.buffers == 0:
        raise SystemExit(f"storage {dtype} was specified, but not a single weight was rounded")
    return report


def load_wrapper(
    model_dir: Path, resolution: int, dtype: str
) -> tuple[MatteLogits, RoundReport | None]:
    """Return an exportable wrapper with the weights rounded, the patches applied and the
    resolution-dependent constants baked.

    What is applied is all of `patch.apply` (①–⑥ + ⑦⑧ + **⑨, the decoder tail**) — only
    {@link verify_patches} applies the stages separately; emit always writes with everything
    applied. The second return value is the rounding count (`None` for f32).

    MUST: rounding happens **right after** `load_model` and **before** `patch.apply`
    ({@link round_to_storage}). Rounding afterwards would move ⑦'s α / β (derived constants computed
    in f32 from the BatchNorm statistics, stored as f32) to f16 values too — VRAM does not shrink
    and only the numbers move ("Storage dtype series" in the module docstring).

    MUST: both the replacement and {@link patch.prepare} come **before** the golden is taken.
    Applied afterwards, only the expected values would be computed on the original path, and it
    would go green while disagreeing with the graph.
    """
    model = load_model(model_dir)
    rounding = round_to_storage(model, dtype)
    wrapper = MatteLogits(model).eval()
    patch.apply(wrapper.model)
    example = build_cases(resolution)[0][1]
    baked = patch.prepare(wrapper, example)
    expected = (1, 1, resolution, resolution)
    if tuple(baked.shape) != expected:
        raise AssertionError(f"the patched output is {tuple(baked.shape)} (expected {expected})")
    return wrapper, rounding


def checkpoint_dtypes(model_dir: Path) -> dict[str, int]:
    """Census of the storage dtypes of the upstream checkpoint (safetensors dtype spelling → count).

    Reads only the header (not the tensor contents). {@link lossless_under} decides from this census
    "whether rounding should be the identity" — so that the quality check's expected value comes
    from an **independent fact**, the upstream storage, rather than from the rounded result (= the
    thing being checked).
    """
    path = model_dir / CHECKPOINT_FILE
    if not path.is_file():
        raise SystemExit(
            f"upstream checkpoint {path} is missing (cannot read the census of storage dtypes)"
        )
    census: dict[str, int] = {}
    with safe_open(str(path), framework="pt") as handle:
        for key in handle.keys():  # noqa: SIM118 — safe_open has only keys() (no __iter__)
            dtype = str(handle.get_slice(key).get_dtype())
            census[dtype] = census.get(dtype, 0) + 1
    return dict(sorted(census.items()))


#: safetensors dtype spellings that are **not floating point** (used for the census decision). The
#: floating-point side is not enumerated so that an unknown spelling (a new f8 variant and the like)
#: does not fall to "not floating point" — an unknown spelling is treated as floating point, the
#: side that does not claim losslessness.
_NON_FLOAT_DTYPES = frozenset({"BOOL", "U8", "I8", "U16", "I16", "U32", "I32", "U64", "I64"})


def lossless_under(dtype: str, census: Mapping[str, int]) -> bool:
    """Whether rounding to `dtype` should be the identity (lossless) for the upstream checkpoint.

    For f16, true only when "every floating-point value of the checkpoint is F16" (BiRefNet_HR — the
    module docstring). False if even one BF16 / F32 tensor exists (rounding can move values). f32 is
    always true (no rounding).

    NOTE: only the weights stored in the checkpoint are seen. For a model with buffers that
    `from_pretrained` computes at initialization and does not store in the checkpoint, rounding is
    not the identity unless those are representable in f16 — then {@link assert_quality_gate} fails
    on the non-zero difference and makes this premise suspect.
    """
    if dtype == PLAIN_DTYPE:
        return True
    if dtype != "f16":
        # MUST: do not silently reuse "lossless if everything is F16" for compression other than f16
        # (phase 2's i8 and the like) — it would be true for HR and become a check demanding bit
        # equality of an i8 series (2026-09-26 review F9).
        raise ValueError(f"no rule for whether rounding to storage {dtype} is lossless (f16 only)")
    floats = {name for name in census if name not in _NON_FLOAT_DTYPES}
    return floats == {"F16"}


def _write_io(
    wrapper: nn.Module,
    graph: IrGraph,
    cases: Sequence[tuple[str, torch.Tensor]],
    out_dir: Path,
    metadata: Mapping[str, Mapping[str, str]] | None = None,
) -> tuple[list[str], dict[str, torch.Tensor]]:
    """Write each case's input and torch CPU expected output to `io.<case>.safetensors`.

    `metadata` is case name → `__metadata__` (only the real-image cases have it — identifying the
    source image).

    The second return value is the expected outputs for the sanity record (passed in their
    `[1, 1, S, S]` shape).
    """
    if len(graph.outputs) != 1:
        raise AssertionError(f"{len(graph.outputs)} IR outputs (the final matte is 1)")
    written: list[str] = []
    mattes: dict[str, torch.Tensor] = {}
    for name, pixel_values in cases:
        with torch.no_grad():
            output = wrapper(pixel_values)
        # MUST: io is also lowered to the concrete representation of the IR's semantic dtypes (ADR
        # 0009's boundary normalization). Unless it matches the form the runtime receives, the Deno
        # side's E2E cannot read the golden.
        tensors = {
            f"{INPUT_PREFIX}{INPUT_NAME}": normalize_boundary_tensor(
                pixel_values, f"{name}: input '{INPUT_NAME}'"
            ),
            f"{OUTPUT_PREFIX}0": normalize_boundary_tensor(
                output.detach().contiguous(), f"{name}: output 0"
            ),
        }
        path = out_dir / f"{IO_PREFIX}{name}{IO_SUFFIX}"
        entry = None if metadata is None else metadata.get(name)
        save_file(tensors, str(path), metadata=None if entry is None else dict(entry))
        written.append(path.name)
        mattes[name] = output.detach()
    return written, mattes


def _sanity(mattes: dict[str, torch.Tensor]) -> dict[str, Any]:
    """Check by **order** that the salient object (the disc of {@link DISC_CASE}) is separated.

    MUST: do not make the sanity check trivially true. A range check (logits have no range) or "same
    inputs agree" verifies nothing. Checking the order of the mean inside and outside the disc fails
    both when the output collapses uniformly and when it stops depending on the input.

    It also checks that the cases **differ from each other** (4 identical outputs are what it looks
    like when the input does not arrive).
    """
    names = sorted(mattes)
    for index, name in enumerate(names):
        for other in names[index + 1 :]:
            if torch.equal(mattes[name], mattes[other]):
                raise AssertionError(
                    f"the outputs of {name} and {other} are identical (the input has no effect)"
                )

    matte = mattes[DISC_CASE].reshape(mattes[DISC_CASE].shape[-2], -1)
    inside = disc_mask(matte.shape[0])
    if matte.shape != inside.shape:
        raise AssertionError(
            f"{DISC_CASE}: the output {tuple(matte.shape)} differs from the disc's shape"
        )
    foreground = float(matte[inside].mean())
    background = float(matte[~inside].mean())
    if foreground <= background:
        raise AssertionError(
            f"mean logit inside the disc of {DISC_CASE}, {foreground:.4f}, is not above the"
            f" outside's {background:.4f} — the salient object is not separated"
        )
    ratios = {name: _foreground_ratio(tensor) for name, tensor in mattes.items()}
    report: dict[str, Any] = {
        "disc_logit_mean": {"inside": round(foreground, 4), "outside": round(background, 4)},
        "logit_range": {
            name: [round(float(tensor.min()), 3), round(float(tensor.max()), 3)]
            for name, tensor in mattes.items()
        },
        "foreground_ratio": {name: round(ratio, 4) for name, ratio in ratios.items()},
    }
    if set(REAL_PERSON_CASES) | set(REAL_SCENE_CASES) <= set(ratios):
        report["real_foreground"] = _real_sanity(ratios)
    return report


def _foreground_ratio(matte: torch.Tensor) -> float:
    """Area ratio of the foreground (logit > 0 = sigmoid > 0.5)."""
    return float((matte > 0.0).to(torch.float32).mean())


def _real_sanity(ratios: Mapping[str, float]) -> dict[str, float]:
    """Check the real-image discrimination (the 2 with a salient object > the 2 without) by
    **order**.

    MUST: check by order, without a threshold (the same convention as the synthetic side). The 2
    person images differ in composition (close-up / full body) and background (cherry trees /
    street), so this order holds only when "the salient object is captured". An output that
    collapses uniformly (all foreground / all background) and an input-independent output both put
    the two groups' foreground ratios level, so they fail.
    """
    worst_person = min(ratios[name] for name in REAL_PERSON_CASES)
    best_scene = max(ratios[name] for name in REAL_SCENE_CASES)
    if worst_person <= best_scene:
        raise AssertionError(
            f"the order of the real images' foreground ratios is reversed: the person side's"
            f" minimum {worst_person:.4f} <= the scene side's maximum {best_scene:.4f}"
        )
    return {name: round(ratios[name], 4) for name in (*REAL_PERSON_CASES, *REAL_SCENE_CASES)}


def _alpha8(logits: torch.Tensor) -> torch.Tensor:
    """Logits → the product's 8-bit α (`round(ALPHA_MAX · sigmoid)`).

    No resize back to the original resolution in between (the baseline and the compared side are
    outputs of the same size, so it is not needed). NOTE: the rounding and precision differ from the
    product (`packages/models/src/birefnet/pipeline.ts`) — here `torch.round` (round half to even)
    and an f64 sigmoid, in the product `Math.round` (halves round up) and f32 after the resize. Only
    pixels exactly on a half-step tie can be counted one step off.
    """
    return torch.round(torch.sigmoid(logits.to(torch.float64)) * ALPHA_MAX)


def _bit_exact(left: torch.Tensor, right: torch.Tensor) -> bool:
    """Whether the bit patterns of two f32 tensors agree.

    `torch.equal` is value `==`, so it treats `0.0` and `-0.0` as equal (measured). The lossless
    check's claim is "if rounding is the identity, the computation is the same", and a flipped
    signed zero is also outside that, so bit patterns are compared.
    """
    if left.dtype is not torch.float32 or right.dtype is not torch.float32:
        raise AssertionError(f"bit comparison is for f32 only ({left.dtype} / {right.dtype})")
    if left.shape != right.shape:
        return False
    return torch.equal(left.contiguous().view(torch.int32), right.contiguous().view(torch.int32))


def _case_quality(reference: torch.Tensor, rounded: torch.Tensor) -> dict[str, Any]:
    """Differences for one case (baseline = the f32 output before rounding).

    - `bit_exact`: bit-pattern equality ({@link _bit_exact} — stronger than a difference of 0)
    - `max_abs` / `rel_rms`: the maximum absolute logit difference, and the RMS of the difference
      divided by the baseline's RMS
    - `mask_mismatch`: fraction of pixels where the binary mask (logit > 0) disagrees
    - `alpha_mae` / `alpha_lsb_changed` / `alpha_max_lsb`: the mean absolute difference of α after
      the sigmoid, the fraction of pixels where the 8-bit α moved by one step or more, and the
      maximum number of steps (the product returns an 8-bit α)

    Reductions are taken in f64 (accumulating 1024² values in f32 could let rounding contribute more
    than the difference being measured).
    """
    if reference.shape != rounded.shape:
        raise AssertionError(
            f"the baseline {tuple(reference.shape)} and the compared side"
            f" {tuple(rounded.shape)} differ in shape"
        )
    base = reference.detach().to(torch.float64)
    other = rounded.detach().to(torch.float64)
    diff = other - base
    base_rms = float(base.square().mean().sqrt())
    alpha_diff = (torch.sigmoid(other) - torch.sigmoid(base)).abs()
    lsb = (_alpha8(rounded) - _alpha8(reference)).abs()
    return {
        "bit_exact": _bit_exact(reference, rounded),
        "max_abs": float(diff.abs().max()),
        "rel_rms": float(diff.square().mean().sqrt()) / base_rms if base_rms > 0 else 0.0,
        "mask_mismatch": float(((other > 0) != (base > 0)).to(torch.float64).mean()),
        "alpha_mae": float(alpha_diff.mean()),
        "alpha_lsb_changed": float((lsb > 0).to(torch.float64).mean()),
        "alpha_max_lsb": int(lsb.max()),
    }


def measure_quality(
    dtype: str,
    census: Mapping[str, int],
    reference: Mapping[str, torch.Tensor],
    rounded: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    """Return the output difference before and after rounding (the quantization error) in the shape
    of `quality.json`.

    The check lives in {@link assert_quality_gate} (this only measures).

    `reference` is the outputs taken with the f32 weights **before rounding** (the same patched
    graph, the same inputs), `rounded` the outputs of the rounded weights. The two sets of cases
    must agree (silently dropping a case present on only one side would erase that case's
    degradation from the record).

    `lossless_expected` comes from the checkpoint's storage ({@link lossless_under}) — derived from
    the rounded result, the decision "whether to expect a difference of 0" would depend on the very
    thing being checked and become trivially true.
    """
    if set(reference) != set(rounded):
        raise AssertionError(
            f"the quality-measurement cases disagree (baseline {sorted(reference)} / compared side"
            f" {sorted(rounded)})"
        )
    if not reference:
        raise AssertionError("no quality-measurement cases at all")
    cases = {name: _case_quality(reference[name], rounded[name]) for name in sorted(reference)}
    worst: dict[str, Any] = {
        "bit_exact": all(entry["bit_exact"] for entry in cases.values()),
        **{
            key: max(entry[key] for entry in cases.values())
            for key in (
                "max_abs",
                "rel_rms",
                "mask_mismatch",
                "alpha_mae",
                "alpha_lsb_changed",
                "alpha_max_lsb",
            )
        },
    }
    lossless = lossless_under(dtype, census)
    return {
        "dtype": dtype,
        "baseline": "f32 weights, same patched graph and inputs, taken in the same process"
        " before rounding",
        "checkpoint_dtypes": dict(census),
        "lossless_expected": lossless,
        "gate": "bit-exact" if lossless else "recorded",
        "cases": cases,
        "worst": worst,
    }


def assert_quality_gate(quality: Mapping[str, Any]) -> None:
    """The quality-measurement check (applied before publishing — on failure the whole series is
    discarded).

    - Series that should be lossless (an f16 series of a checkpoint that is all f16): every case is
      **bit-exact**. If not, suspect the rounding position (rounding after `patch.apply` moves ⑦'s
      α / β) or initialization-time buffers not stored in the checkpoint (the NOTE of
      {@link lossless_under}).
    - Lossy series (`gate` = `recorded` — no bit equality expected): fails when the worst mean
      absolute α difference is at or above {@link ALPHA_MAE_LIMIT} (0.5 LSB of the 8-bit α) (the
      provisional line of ADR 0113 decision 5). It also fails when the difference is **0 in every
      case**: the weights moved with rounding but the outputs did not move by a single bit, which is
      what it looks like when the baseline and the compared side see the same weights (the
      measurement harness has collapsed into something trivially true).
    """
    cases: Mapping[str, Mapping[str, Any]] = quality["cases"]
    if quality["lossless_expected"]:
        moved = sorted(name for name, entry in cases.items() if not entry["bit_exact"])
        if moved:
            raise AssertionError(
                f"the checkpoint is all f16, but the output rounded to {quality['dtype']} does not"
                f" match f32 bit for bit: {moved}"
                f" (max_abs {quality['worst']['max_abs']:.3e}) — suspect the rounding position"
                " (rounding after patch.apply moves ⑦'s α / β) or initialization-time buffers not"
                " stored in the checkpoint"
            )
        return
    if all(entry["bit_exact"] for entry in cases.values()):
        raise AssertionError(
            f"rounding to {quality['dtype']}, which should be lossy, left the output of every case"
            " bit-exact with f32 — the baseline and the compared side see the same weights (the"
            " quality measurement has become trivially true)"
        )
    worst_alpha = quality["worst"]["alpha_mae"]
    if worst_alpha >= ALPHA_MAE_LIMIT:
        over = sorted(
            name for name, entry in cases.items() if entry["alpha_mae"] >= ALPHA_MAE_LIMIT
        )
        raise AssertionError(
            f"the mean absolute α difference of the output rounded to {quality['dtype']},"
            f" {worst_alpha:.3e}, is at or above the provisional line {ALPHA_MAE_LIMIT:.3e}"
            f" (0.5 LSB of the 8-bit α): {over} — this preset is not shipped (ADR 0113 decision 5)"
        )


def _storage_summary(graph: IrGraph) -> dict[str, int]:
    """Storage breakdown (eligible = weights that stay compressed and resident on the GPU /
    ineligible = f32 storage — ADR 0006's diagnostic)."""
    breakdown = storage_breakdown(graph)
    return {
        "compressed_tensors": breakdown.compressed_tensors,
        "compressed_bytes": breakdown.compressed_bytes,
        "plain_tensors": breakdown.plain_tensors,
        "plain_bytes": breakdown.plain_bytes,
    }


def _forward_all(
    wrapper: nn.Module, cases: Sequence[tuple[str, torch.Tensor]]
) -> dict[str, torch.Tensor]:
    """Output per case (logits `[1, 1, S, S]`)."""
    outputs: dict[str, torch.Tensor] = {}
    for name, pixel_values in cases:
        with torch.no_grad():
            outputs[name] = wrapper(pixel_values).detach()
    return outputs


def reference_outputs(
    model_dir: Path, resolution: int, cases: Sequence[tuple[str, torch.Tensor]]
) -> dict[str, torch.Tensor]:
    """Take the quality measurement's baseline (outputs of the patched graph with the f32 weights
    **before rounding**).

    A separate wrapper is loaded in f32 and released once its outputs are taken (it is not held in
    RAM together with the rounding side's wrapper). The baseline is also taken with the patched
    graph because what is compared is **only the rounding of the weights** — comparing with the
    pre-patch eager would mix the rounding differences of ⑦⑧⑨ (`--verify`'s territory) into the
    quantization error.
    """
    wrapper, _ = load_wrapper(model_dir, resolution, PLAIN_DTYPE)
    return _forward_all(wrapper, cases)


def export_series(
    model_dir: Path,
    out_dir: Path,
    resolution: int,
    real_images: bool = False,
    corpus_root: Path = CORPUS_ROOT,
    *,
    dtype: str,
) -> dict[str, Any]:
    """Write the IR container and the golden io (and `quality.json` for compressed series), and
    return a summary.

    With `real_images` set, 4 real-image cases are **added** to the 4 synthetic ones (not replaced —
    "The 2 golden groups" in the module docstring).

    When `dtype` is not f32 the weights are rounded before the golden is taken
    ({@link load_wrapper}), and the difference from the f32 outputs before rounding is written to
    `quality.json` ("the quality-measurement harness" in the module docstring). The
    quality-measurement cases are the 8 of 4 synthetic + 4 real images, and the real images go
    through regardless of `real_images` (they enter the golden only with `real_images`).

    MUST: the outputs are written to a staging slot and put in place only after passing **every
    check** (input order, sanity, quality check). Placing them in final before the checks would let
    a failed run leave "assets that can pass the acceptance check" behind — the io golden is taken
    from the same broken wrapper, so it is self-consistent, and the TS side's cross-check **goes
    green** (the spelling of "when publishing is allowed" is {@link _shared.decode_series._publish};
    the discipline of swapping in and cleaning up is the core primitive {@link
    karume.artifacts.staged_publication}).
    """
    assert_resolution(resolution)
    if dtype not in WEIGHT_DTYPES:
        raise SystemExit(f"cannot write storage dtype '{dtype}' ({' / '.join(WEIGHT_DTYPES)})")
    compressed = dtype != PLAIN_DTYPE
    # The origin is read before the weights (a checkpoint without a record fails before the weights
    # are read).
    provenance = upstream_provenance(model_dir)
    # The quality check's expected value (should it be lossless) comes from the upstream storage —
    # read before the weights.
    census = checkpoint_dtypes(model_dir) if compressed else {}
    synthetic = build_cases(resolution)
    # MUST: the real images are built before the weights (if an image is missing, stop here rather
    # than after writing out 964MB). Compressed series always build them for the quality
    # measurement.
    real = build_real_cases(resolution, corpus_root) if real_images or compressed else ()
    real_inputs = [(name, pixel_values) for name, pixel_values, _md in real]
    cases = [*synthetic, *(real_inputs if real_images else ())]
    metadata = {name: md for name, _pixel_values, md in real} if real_images else {}
    quality_cases = [*synthetic, *real_inputs]
    # MUST: the baseline is taken **before** rounding (a separately loaded f32 wrapper —
    # {@link reference_outputs}).
    reference = reference_outputs(model_dir, resolution, quality_cases) if compressed else {}
    wrapper, rounding = load_wrapper(model_dir, resolution, dtype)
    out_dir.parent.mkdir(parents=True, exist_ok=True)

    _, example = synthetic[0]
    quality: dict[str, Any] | None = None
    with staged_publication(out_dir) as staged:
        # The writer creates the directory slot (the primitive does not create slots — it only
        # passes a path).
        staged.mkdir()
        # No dynamic axes (the resolution is fixed per series — the module docstring).
        graph = export_to_file(
            wrapper,
            (example,),
            staged / MODEL_FILE,
            provenance=provenance,
            # The graph name is the **component name** (= the weights key of karume.json). It is not
            # derived from the directory name — this family's container sits directly in the series
            # directory, so the directory name is the series name (`birefnet-hr-<resolution>`), not
            # the component name (container-v1 §2.1).
            graph_name=BIREFNET_ROLE,
            symbol_names=(),
            weight_dtype=dtype,
        )
        declared = tuple(item.name for item in graph.inputs)
        if declared != (INPUT_NAME,):
            raise AssertionError(
                f"the graph inputs are {declared}, not the expected {(INPUT_NAME,)}"
            )
        written, mattes = _write_io(wrapper, graph, cases, staged, metadata)
        # MUST: evaluated before publishing (the only non-trivial check of this series — on failure
        # the whole slot is discarded).
        sanity = _sanity(mattes)
        if compressed:
            # The outputs taken for the golden are reused as they are, and only the cases not in the
            # golden (the real images without `real_images`) are run additionally.
            rounded = {
                **_forward_all(wrapper, [case for case in quality_cases if case[0] not in mattes]),
                **{name: mattes[name] for name, _pixel_values in quality_cases if name in mattes},
            }
            if not real_images:
                # MUST: the real images not in the golden also get the foreground-ratio order check
                # on the rounded outputs ({@link _sanity} applies it when `real_images` is set). The
                # quality numbers only look at differences relative to the baseline, so that the
                # compressed series still discriminates is checked separately (2026-09-26 review
                # A-6).
                sanity["real_foreground"] = _real_sanity(
                    {name: _foreground_ratio(rounded[name]) for name, _pixel_values in real_inputs}
                )
            quality = measure_quality(dtype, census, reference, rounded)
            # The numbers are printed before the check (on failure the whole staging slot is
            # discarded, so the per-case differences survive only here).
            print(f"[quality] {json.dumps(quality['cases'], ensure_ascii=False)}", flush=True)
            # MUST: the quality check also comes before publishing (on failure the whole slot is
            # discarded).
            assert_quality_gate(quality)
            (staged / QUALITY_FILE).write_text(
                json.dumps(quality, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
            )
    return {
        "dir": str(out_dir),
        "dtype": dtype,
        "resolution": resolution,
        "nodes": len(graph.nodes),
        "outputs": len(graph.outputs),
        "initializers": len(graph.initializers),
        "model_bytes": sum(p.stat().st_size for p in container_parts(out_dir / MODEL_FILE)),
        "storage": _storage_summary(graph),
        "rounding": None if rounding is None else rounding.describe(),
        "ops": sorted(graph.required_ops),
        "symbols": list(graph.symbols),
        "io": written,
        "input_shape": list(example.shape),
        "real_images": {name: md[SOURCE_IMAGE_KEY] for name, md in metadata.items()},
        "sanity": sanity,
        "quality": None if quality is None else {**quality["worst"], "gate": quality["gate"]},
    }


def _diff_entry(
    stage: str,
    claim: str,
    got: dict[str, torch.Tensor],
    expected: dict[str, torch.Tensor],
) -> dict[str, Any]:
    """Equivalence report for one stage (case name → maxdiff, and bit equality over all cases).

    `bit_exact` is a stronger claim than "difference 0" (`0.0 == -0.0` is a difference of 0 but the
    bits differ), and it matters for stages where **bit equality is the claim itself**, like
    rewrites that only reorder.
    """
    return {
        "stage": stage,
        "claim": claim,
        "maxdiff": {
            name: float((got[name] - expected[name]).abs().max()) for name in sorted(expected)
        },
        "bit_exact": all(torch.equal(got[name], expected[name]) for name in expected),
    }


def verify_patches(model_dir: Path, resolution: int) -> list[dict[str, Any]]:
    """Measure the equivalence with the pre-patch eager at **3 points** (the same shape as
    `siglip2.export.py --verify`).

    1. The reorder-only rewrites (window / qkv / roll / pad / PatchMerging / image2patches / window
       mask) → **bit_exact is the claim itself**. The operation sequences correspond one to one, so
       the difference must be 0, and it fails here if not.
    2. Up to `BatchNorm2d` → per-channel affine and `AdaptiveAvgPool2d` → 2-stage sum → **report the
       maxdiff**. The lowest bits move by as much as ATen folds multiply-adds with FMA / reduces 2
       axes at once (⑦⑧ of `birefnet.patch`'s docstring).
    3. Up to swapping the order of the decoder tail's 1×1 conv and bilinear upsample → **report the
       maxdiff**. The same group as 2, "equivalent formulas, only the rounding differs"; what comes
       on top of 2's difference is this stage's contribution (⑨ of `birefnet.patch`'s docstring).

    MUST: the order is "**fix the reference values of all cases** → 1 → 2 → 3". The patches replace
    class attributes process-wide, so re-taking the reference per stage would make the references
    from stage 2 on post-patch values (trivially true).
    """
    assert_resolution(resolution)
    if patch.patches_applied():
        raise SystemExit(
            "cannot take the reference in a process where the patches are already applied (the"
            " equivalence check would become trivially true with a difference of 0)"
        )
    wrapper = MatteLogits(load_model(model_dir)).eval()
    cases = build_cases(resolution)

    def mattes() -> dict[str, torch.Tensor]:
        with torch.no_grad():
            return {name: wrapper(pixel_values) for name, pixel_values in cases}

    reference = mattes()

    patch.apply_layout_patches(wrapper.model)
    # The window mask is a resolution-dependent constant, so the first pass after patching bakes it
    # through prepare.
    patch.prepare(wrapper, cases[0][1])
    layout = _diff_entry("layout", "bit-exact", mattes(), reference)
    if not layout["bit_exact"]:
        raise AssertionError(
            f"the reorder-only rewrites are not bit-identical: maxdiff={layout['maxdiff']}"
        )
    patch.apply_module_patches(wrapper.model)
    modules = _diff_entry("modules", "max-abs-diff", mattes(), reference)
    patch.apply_tail_patches(wrapper.model)
    return [layout, modules, _diff_entry("tail", "max-abs-diff", mattes(), reference)]


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument(
        "--resolution",
        type=int,
        default=DEFAULT_RESOLUTION,
        help=f"side of the input (a multiple of {RESOLUTION_MULTIPLE}, default"
        f" {DEFAULT_RESOLUTION}). Each resolution is a separate graph (the baked constants change),"
        " so the series is separate too",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output directory (default outputs/series/<model name>-<resolution>{,-f16}/)",
    )
    parser.add_argument(
        "--dtype",
        choices=WEIGHT_DTYPES,
        default=None,
        help="storage dtype of the weights (default: the same type as the upstream checkpoint's"
        " storage — f16 for BiRefNet_HR, f32 for Lucida. A dtype wider than the upstream storage is"
        " refused. f16 rounds right after load, before the patches, stores only the eligible slots"
        " as f16, and writes the difference from the f32 outputs before rounding to quality.json —"
        " ADR 0113. **emit only**, exclusive with --verify)",
    )
    parser.add_argument(
        "--real-images",
        action="store_true",
        help=f"also write a golden through the 4 real images in `{CORPUS_ROOT.name}/` (added to the"
        " 4 synthetic cases). The preprocessing is verbatim from the bundled handler.py, and the"
        " sha256 of the source image goes into __metadata__",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="measure the equivalence with the pre-patch eager at 3 points (does not emit — in the"
        " same process the class-attribute replacement contaminates the reference, so the two"
        " cannot be combined)",
    )
    args = parser.parse_args(argv)
    if args.verify and args.dtype is not None:
        parser.error(
            "--dtype cannot be combined with --verify (--verify's reference is the eager pass with"
            " the pre-patch f32 weights — the same structural reason as ADR 0027 decision 3)"
        )
    if args.verify:
        for entry in verify_patches(args.model_dir, args.resolution):
            print(
                f"{entry['stage']} ({entry['claim']}): bit_exact={entry['bit_exact']}"
                f" maxdiff={entry['maxdiff']}"
            )
        return
    dtypes, default_dtype = export_dtypes(args.model_dir)
    if args.dtype is not None and args.dtype not in dtypes:
        parser.error(
            f"--dtype {args.dtype} is not written for {args.model_dir.name} (writable:"
            f" {' / '.join(dtypes)}) — a dtype wider than the upstream checkpoint's storage only"
            " widens the values exactly, adds no information and is not shipped either (ADR 0113"
            " addendum 2026-10-11)"
        )
    dtype = default_dtype if args.dtype is None else args.dtype
    out_dir = args.out
    if out_dir is None:
        out_dir = default_out_dir(args.model_dir, args.resolution, dtype)
    summary = export_series(
        args.model_dir, out_dir, args.resolution, real_images=args.real_images, dtype=dtype
    )
    print(json.dumps({"model_dir": str(args.model_dir), **summary}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
