"""Script that writes the real-weight Depth Anything V2 (single-image relative depth estimation)
out as IR (docs/ir-v2.md) + container (krm) + golden io. The model is chosen with `--model-dir`
(default Small — "Model axis" below).

The outputs go under `outputs/series/` and are kept out of commits by the repository root's
`.gitignore`.

    uv run --group depth-anything python -m depth_anything.export
    uv run --group depth-anything python -m depth_anything.export --verify  # check vs unpatched
    uv run --group depth-anything-preprocess python -m depth_anything.export --real-images

transformers is **pinned at 5.14.1** (because `depth_anything.patch` replaces class attributes of
the modeling code — the `depth-anything` group in pyproject.toml).

## Model axis

The series name is the `--model-dir` directory name (lower-cased). What it accepts is the weight
set of the transformers port (`DepthAnythingForDepthEstimation` = DINOv2 backbone + DPT neck /
head); the default is `inputs/depth-anything/Depth-Anything-V2-Small-hf/` (upstream
`depth-anything/Depth-Anything-V2-Small-hf` — **Apache-2.0**).

MUST: **the upstream licence of Base / Large is CC BY-NC 4.0** (only Small is Apache-2.0 —
measured 2026-08-14 through the HF API). The script accepts any of them via `--model-dir`, but
only Small can become a distribution. The container's provenance (`provenance`) is therefore not a
constant but **derived from the actual `--model-dir`**
({@link _shared.upstream.snapshot_provenance} — the README's `license` and the downloaded commit
SHA). A container baked from Base states `cc-by-nc-4.0`, and the assembly
(`depth_anything.distribution`) compares it with the attribution table and fails.

## Resolution (not an axis — fixed to one size)

The input is **one square size, the pretraining resolution** (`image_size=518` for Small / Base /
Large alike; patch 14 = 37×37 patches). DINOv2's position embeddings are used as is when the
patch count matches, and otherwise `aten.upsample_bicubic2d` (not in the IR vocabulary) appears.
② of `depth_anything.patch` inverts the condition to fail loudly, so here the resolution is just
read from the weights' `image_size` ({@link load_wrapper}).

## What goes into the graph

The one `predicted_depth` (`[B, H, W]`) returned by `DepthAnythingForDepthEstimation.forward`.
It is relative depth, so it has neither unit nor direction (larger means nearer). The graph runs
up to the ReLU at the end of the head; normalization for visualization (`depth / depth.max()` and
the like) is the host's job.

batch is **static 1**. There are no dynamic axes (`symbol_names=()`).

MUST: the model is exported after applying the patches of `depth_anything.patch`. Without them
the converter fails on the DPT reassemble's `nn.ConvTranspose2d` (2 `aten.convolution.default`)
(the list of replacements is in the docstring of `depth_anything.patch`). The golden expected
values are taken from the module **after the replacements** — the difference before and after is
the patch layer's responsibility, measured by `--verify` at 2 points.

MUST: `--verify` does not emit (the two cannot be combined in one process). The patches replace
class attributes process-wide, so once they are applied for the emit, "the reference before
applying" can no longer be taken.

## The input contract (preprocessing is not in the graph)

The graph input is the **normalized** `pixel_values f32 [1, 3, S, S]`. Decoding / resizing /
rescaling (1/255) / normalizing the image is Karume's job, and the source of truth for the
constants is the bundled `preprocessor_config.json` (the ImageNet statistics of
`DPTImageProcessor` = {@link IMAGENET_MEAN} / {@link IMAGENET_STD}; `resample=3` = **bicubic**;
`keep_aspect_ratio=true` is the identity for square input → 518²). **Both the statistics and the
filter differ from SigLIP2 (mean = std = 0.5, resample 2 = bilinear)**, so they are kept here
rather than shared. {@link check_processor} measures the constants **from the real processor** on
every `--real-images` run.

## The 2 groups of goldens (synthetic + real images) — both are kept

The default 4 cases are **synthetic images** ({@link build_cases}), and the discrimination lives
in {@link _sanity}:

- the outputs of the 4 cases **differ from each other** (the input has an effect)
- each case's output is **non-negative** (the head ends in a ReLU) and **not uniform**
  (min < max)
- **the correlation with the diagonal ramp is largest for the `ramp` case** (only the ordering is
  checked, no threshold). Of the 4 images only `ramp` has a monotonic depth cue, so this breaks
  when the output does not follow the input's geometry (uniform, input-independent, spatial
  mix-up)

With `--real-images` it also writes goldens through 4 **real images** in `outputs/misc/corpus/`
({@link REAL_CASES}). Their claim is a comparison of "TS preprocessing + Karume inference" against
"`DPTImageProcessor` + torch inference", and the Python side is **the source of truth itself**
(it calls the actual `AutoImageProcessor` instead of a hand copy — unlike the hand copy of
BiRefNet's handler.py, because here the public transformers class is itself the source of truth).

The real-image side has its own discrimination as well: each image gets a pair of **a near region
and a far region that follow from the composition** ({@link REAL_REGIONS}), and the only check is
that the mean depth of the near side exceeds the far side (no threshold —
{@link _real_sanity}). The synthetic `ramp` asks "can it follow a monotonic cue", the real images
ask "can it get the near/far layout of a photographed composition right", and neither replaces the
other.

MUST: real-image goldens carry **the sha256 of the source image** in `__metadata__`. The images
are outputs baked by `examples/anima/eval-images.ts` and **frozen into `outputs/misc/corpus/` by
hand**, so in an environment where the corpus was replaced but the goldens were not retaken, the
comparison is not a comparison: **the inputs differ** (and the values look plausible at first
sight).

## Output layout

    outputs/series/<series name>/model.krm             weights, constants + the 2 descriptors
    outputs/series/<series name>/io.<case>.safetensors input and torch CPU expected output

The tensor key convention of io is the same as tiny golden / DeBERTa / SigLIP2 / BiRefNet
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
from safetensors.torch import save_file
from torch import nn

from _shared.paths import INPUTS_ROOT, MISC_ROOT, SERIES_ROOT
from _shared.upstream import snapshot_provenance
from karume.artifacts import staged_publication
from karume.container import Provenance, container_parts
from karume.convert import normalize_boundary_tensor
from karume.dist import NOTICE_FILENAME
from karume.ir import IrGraph
from karume.pipeline import export_to_file

from . import patch
from .distribution import DEPTH_ANYTHING_ROLE
from .measurements import CONVT_MAXDIFF

#: Parent of the real weights (where the 3 HF files are unpacked into
#: `inputs/depth-anything/<name>/`).
MODELS_ROOT = INPUTS_ROOT / "depth-anything"

#: Location of the real-image corpus (a frozen copy = a host asset, so deleting it means baking
#: it again — docs/assets-layout.md).
CORPUS_ROOT = MISC_ROOT / "corpus"

#: The default model (when `--model-dir` is not given — the only Apache-2.0 one upstream).
DEFAULT_MODEL_DIR = MODELS_ROOT / "Depth-Anything-V2-Small-hf"

MODEL_FILE = "model.krm"

IO_PREFIX = "io."
IO_SUFFIX = ".safetensors"
INPUT_PREFIX = "input."
OUTPUT_PREFIX = "output."

#: The name of the graph input (= the argument name of {@link DepthMap.forward}). It is compared
#: after export and stops on a mismatch — inputs are passed by position, so a silent shift would
#: swap only the golden's inputs.
INPUT_NAME = "pixel_values"

#: The normalization constants of the preprocessing (source of truth = `image_mean` /
#: `image_std` of the bundled `preprocessor_config.json`).
IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)

#: `"resample"` of `preprocessor_config.json`. **BICUBIC as a PIL constant** (NEAREST=0 /
#: LANCZOS=1 / BILINEAR=2 / **BICUBIC=3** / BOX=4 / HAMMING=5) — not the 2 of SigLIP2.
EXPECTED_RESAMPLE = 3

#: `rescale_factor` (1/255. {@link check_processor} also checks that its reciprocal is exactly
#: 255.0 in f64).
EXPECTED_RESCALE = 1.0 / 255.0

#: Real-image cases = (case name, file name in `outputs/misc/corpus/`, content). **The source of
#: truth for the images is `examples/anima/eval-images.ts`** (it holds the prompts / seeds /
#: resolutions, and `deno task demo:eval-images` bakes the same 4 images again — the chosen ones
#: are frozen by hand). The case names are spelled as in the `REAL_CASES` of SigLIP2 / BiRefNet
#: (the same 4 images, just read from a different series directory).
REAL_CASES: tuple[tuple[str, str, str], ...] = (
    (
        "photo-portrait",
        "anima-default-1024x1024-defaultstep-seed42.png",
        "portrait close-up + avenue of cherry trees (the subject fills most of the frame, the"
        " background is distant)",
    ),
    (
        "photo-landscape",
        "anima-default-1024x1024-defaultstep-seed43.png",
        "landscape (three depth layers: grass in front → lake → mountains and sky)",
    ),
    (
        "photo-corridor",
        "anima-default-1024x1024-defaultstep-seed44.png",
        "school corridor (strong perspective; the only image with the vanishing point in the"
        " frame)",
    ),
    (
        "photo-street",
        "anima-default-1024x1024-defaultstep-seed45.png",
        "full-body person + street (the person in front, and the sky opening above the buildings)",
    ),
)

#: The "near region / far region" pairs the real-image discrimination looks at = case name →
#: (near, far). Each region is **a relative rectangle on the depth map** `(name, y0, y1, x0, x1)`
#: (ratios from 0 to 1; the top / left edge is 0).
#:
#: MUST: only pairs that follow uniquely from the composition ("person > sky", "floor in front >
#: vanishing point"). Depth is relative, so absolute amounts mean nothing, and this **ordering** is
#: the only claim deducible from the composition. The TS-side
#: `packages/models/tests/e2e_depth_anything_real_test.ts` applies the same rectangles to the real
#: GPU output (change the rectangles on both sides — moving only one silently turns the claim
#: "both agree" into something else).
REAL_REGIONS: Mapping[str, tuple[tuple[str, float, float, float, float], ...]] = {
    "photo-portrait": (
        ("subject", 0.85, 1.00, 0.35, 0.65),
        ("background", 0.30, 0.45, 0.02, 0.12),
    ),
    "photo-landscape": (
        ("foreground-ground", 0.92, 1.00, 0.30, 0.70),
        ("mountain-sky", 0.05, 0.20, 0.35, 0.65),
    ),
    "photo-corridor": (
        ("floor", 0.92, 1.00, 0.40, 0.60),
        ("vanishing-point", 0.64, 0.76, 0.40, 0.56),
    ),
    "photo-street": (
        ("person", 0.55, 0.85, 0.44, 0.56),
        ("sky", 0.02, 0.12, 0.40, 0.60),
    ),
}

#: The `__metadata__` fields of real-image goldens (identifying the source image — the MUST in the
#: module docstring).
SOURCE_IMAGE_KEY = "source_image"
SOURCE_SHA256_KEY = "source_sha256"

#: The random seed of the synthetic images (the `noise` case). Independent of the global seed.
SEED = 20260814

#: The circle radius of the `disc` case (a ratio, with the image's shorter side as 1).
DISC_RADIUS = 0.3

#: Foreground / background colour of the `disc` case (RGB, range `[0, 1]`). **The 3 channels have
#: different values**, so a channel mix-up (the preprocessing statistics in the wrong order) shows
#: in the values.
DISC_FOREGROUND: tuple[float, float, float] = (0.92, 0.78, 0.35)
DISC_BACKGROUND: tuple[float, float, float] = (0.12, 0.16, 0.24)

#: One square of the `checker` case's checkerboard (pixels). The width is not divisible by the
#: patch (14), to avoid the board and the patch grid happening to be in phase.
CHECKER_CELL = 40

#: The only case with a monotonic depth cue (the correlation of {@link _sanity} peaks here).
RAMP_CASE = "ramp"


def default_out_dir(model_dir: Path) -> Path:
    """The default location of the outputs (`outputs/series/<model name>/`).

    The resolution is not in the name because in this family the resolution is **not an axis**
    (only the one pretraining resolution goes through — "Resolution" in the module docstring).
    """
    return SERIES_ROOT / model_dir.name.lower()


def upstream_provenance(model_dir: Path) -> Provenance:
    """Derive the provenance baked into the container (container-v1 §2.3) from the actual
    `--model-dir`.

    The licence identifier comes from the README's front matter, the revision from the download
    record of `hf download --local-dir` ({@link _shared.upstream.snapshot_provenance}). Series
    migration (`migrate_series`) uses the same function.
    """
    return snapshot_provenance(model_dir, notice=NOTICE_FILENAME)


class DepthMap(nn.Module):
    """An export wrapper that returns only the relative depth map `[B, H, W]`."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.model(pixel_values).predicted_depth


def normalize_image(image: torch.Tensor) -> torch.Tensor:
    """Normalize a `[1, 3, S, S]` in `[0, 1]` with the ImageNet statistics (verbatim from
    DPTImageProcessor)."""
    mean = torch.tensor(IMAGENET_MEAN).reshape(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).reshape(1, 3, 1, 1)
    return (image - mean) / std


def disc_mask(size: int) -> torch.Tensor:
    """The inside of the `disc` case's circle (a `[S, S]` bool)."""
    axis = (torch.arange(size, dtype=torch.float32) + 0.5) / size * 2.0 - 1.0
    squared = axis.reshape(size, 1) ** 2 + axis.reshape(1, size) ** 2
    return squared <= DISC_RADIUS**2


def ramp_plane(size: int) -> torch.Tensor:
    """The diagonal ramp coordinates `[S, S]` (the base of the `ramp` case = what the correlation
    of {@link _sanity} is taken against)."""
    axis = torch.linspace(0.0, 1.0, size)
    return (axis.reshape(1, size) + axis.reshape(size, 1)) / 2.0


def _disc_image(size: int) -> torch.Tensor:
    """A `[1, 3, S, S]` with a bright circle on a dark background."""
    foreground = torch.tensor(DISC_FOREGROUND).reshape(3, 1, 1)
    background = torch.tensor(DISC_BACKGROUND).reshape(3, 1, 1)
    return torch.where(disc_mask(size), foreground, background).unsqueeze(0)


def _ramp_image(size: int) -> torch.Tensor:
    """A diagonal ramp from top left to bottom right (phase-shifted per channel)."""
    shifts = torch.tensor([0.0, 0.15, 0.3]).reshape(3, 1, 1)
    return ((ramp_plane(size).unsqueeze(0) + shifts) % 1.0).unsqueeze(0)


def _checker_image(size: int) -> torch.Tensor:
    """A checkerboard ({@link CHECKER_CELL}-pixel squares). Light and dark are inverted per
    channel."""
    index = torch.arange(size) // CHECKER_CELL
    parity = ((index.reshape(1, size) + index.reshape(size, 1)) % 2).to(torch.float32)
    channels = torch.stack([parity, 1.0 - parity, parity * 0.5 + 0.25])
    return channels.unsqueeze(0)


def build_cases(resolution: int) -> tuple[tuple[str, torch.Tensor], ...]:
    """`(name, pixel_values)` of the 4 golden cases (**normalized** synthetic images)."""
    generator = torch.Generator().manual_seed(SEED)
    noise = torch.rand((1, 3, resolution, resolution), generator=generator)
    return (
        ("disc", normalize_image(_disc_image(resolution))),
        (RAMP_CASE, normalize_image(_ramp_image(resolution))),
        ("checker", normalize_image(_checker_image(resolution))),
        ("noise", normalize_image(noise)),
    )


def check_processor(processor: Any) -> dict[str, Any]:
    """Measure **on the real object** that the image processor is the expected class with the
    expected constants.

    What this identifies is "the counterpart Karume's preprocessing layer must match", and on a
    mismatch it fails before writing a single image (the same discipline as
    `siglip2.preprocess.check_processor_shape`). In particular `resample` is 3 = bicubic, which
    **differs** from SigLIP2 (2 = bilinear), and Karume's default is bilinear (the value of the
    2 existing families), so if this moved while the filter passed to the TS-side `resizeRgb8`
    stayed the same, nobody would notice until the `pixel_values` check of the real-GPU E2E
    failed.

    NOTE: in DA-V2 the 3 of the config equals the default 3 of the `DPTImageProcessor` class
    attribute, so this check cannot tell "read the config" from "saw the default" (on the SigLIP2
    side they differ). What it can tell is **that the value changed**, which is enough for this
    family.
    """
    from transformers.image_processing_backends import TorchvisionBackend

    if not isinstance(processor, TorchvisionBackend):
        raise SystemExit(
            f"the image processor is not a TorchvisionBackend: {type(processor).__mro__}"
            " (the default of transformers 5.x is the fast side — falling back to the PIL side"
            " changes the resize implementation)"
        )
    if int(processor.resample) != EXPECTED_RESAMPLE:
        raise SystemExit(
            f"resample is {int(processor.resample)} (expected {EXPECTED_RESAMPLE} = PIL BICUBIC)"
        )
    if tuple(processor.image_mean) != IMAGENET_MEAN or tuple(processor.image_std) != IMAGENET_STD:
        raise SystemExit(f"mean/std are {processor.image_mean} / {processor.image_std}")
    if processor.rescale_factor != EXPECTED_RESCALE:
        raise SystemExit(f"rescale_factor is {processor.rescale_factor}")
    if 1.0 / processor.rescale_factor != 255.0:
        raise SystemExit(f"1/rescale_factor is {1.0 / processor.rescale_factor} (not 255.0)")
    for flag in ("do_resize", "do_rescale", "do_normalize"):
        if not getattr(processor, flag):
            raise SystemExit(f"{flag} is False (the 3 preprocessing stages are incomplete)")
    for flag in ("do_center_crop", "do_pad"):
        if getattr(processor, flag, False):
            raise SystemExit(f"{flag} is True (this layer has neither crop nor pad)")
    return {
        "class": type(processor).__name__,
        "backend": "TorchvisionBackend",
        "resample": int(processor.resample),
        "keep_aspect_ratio": bool(processor.keep_aspect_ratio),
        "ensure_multiple_of": int(processor.ensure_multiple_of),
    }


def build_real_cases(
    model_dir: Path, resolution: int, corpus_root: Path
) -> tuple[tuple[str, torch.Tensor, dict[str, str]], ...]:
    """`(name, pixel_values, __metadata__)` of the 4 real images ({@link REAL_CASES}).

    The preprocessing is **the real object** `AutoImageProcessor` returns (`DPTImageProcessor` =
    the fast side), not a hand copy. The BiRefNet side copies handler.py because the upstream
    preprocessing pulls in dependencies unrelated to inference; here the public transformers class
    is itself the source of truth.

    MUST: if even one image is missing, stop **before** the heavy imports (`--real-images` is an
    explicit request, so silently writing 3 images would hide the missing golden).
    """
    paths: list[tuple[str, str, Path]] = []
    for name, file_name, _why in REAL_CASES:
        path = corpus_root / file_name
        if not path.is_file():
            raise SystemExit(
                f"missing real image {path} (bake it: deno task demo:eval-images"
                " --source <path of the Anima distribution> → freeze the chosen ones into"
                " outputs/misc/corpus/)"
            )
        paths.append((name, file_name, path))

    from PIL import Image
    from transformers import AutoImageProcessor

    processor = AutoImageProcessor.from_pretrained(str(model_dir))
    check_processor(processor)
    expected_shape = (1, 3, resolution, resolution)
    cases: list[tuple[str, torch.Tensor, dict[str, str]]] = []
    for name, file_name, path in paths:
        raw = path.read_bytes()
        with Image.open(io_module.BytesIO(raw)) as image:
            pixel_values = processor(images=image.convert("RGB"), return_tensors="pt")[
                "pixel_values"
            ]
        if tuple(pixel_values.shape) != expected_shape:
            raise AssertionError(
                f"the pixel_values of {name} are {tuple(pixel_values.shape)}"
                f" (not the expected {expected_shape}) — a non-square image takes the"
                " keep_aspect_ratio path and does not match the baked graph's input shape"
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


def load_model(model_dir: Path) -> nn.Module:
    """Load the real weights and check the configuration the replacements assume (before
    patching)."""
    from transformers import AutoModelForDepthEstimation

    model = AutoModelForDepthEstimation.from_pretrained(str(model_dir), dtype=torch.float32)
    model.eval()
    # Check the configuration before the heavy forward (fail without baking anything when the
    # replacements do not cover it).
    patch.assert_supported(model)
    return model


def load_wrapper(model_dir: Path) -> tuple[DepthMap, int]:
    """Return the patched, exportable wrapper and the pretraining resolution.

    MUST: the replacements come **before** the goldens are taken. Applied afterwards, only the
    expected values would be computed on the original path, and the result would be green while
    disagreeing with the graph.
    """
    wrapper = DepthMap(load_model(model_dir)).eval()
    resolution = patch.pretrained_resolution(wrapper.model)
    patch.apply(wrapper.model)
    return wrapper, resolution


def _write_io(
    wrapper: nn.Module,
    graph: IrGraph,
    cases: Sequence[tuple[str, torch.Tensor]],
    out_dir: Path,
    metadata: Mapping[str, Mapping[str, str]] | None = None,
) -> tuple[list[str], dict[str, torch.Tensor]]:
    """Write each case's input and torch CPU expected output to `io.<case>.safetensors`.

    `metadata` maps a case name → `__metadata__` (only real-image cases have it — identifying the
    source image).

    The second return value is the expected outputs for the sanity record (passed in their
    `[1, H, W]` shape).
    """
    if len(graph.outputs) != 1:
        raise AssertionError(f"{len(graph.outputs)} IR outputs (expected exactly one depth map)")
    written: list[str] = []
    depths: dict[str, torch.Tensor] = {}
    for name, pixel_values in cases:
        with torch.no_grad():
            output = wrapper(pixel_values)
        # MUST: io is also lowered to the actual representation of the IR's semantic dtype (the
        # boundary normalization of ADR 0009). Unless it matches what the runtime receives, the
        # Deno-side E2E cannot read the golden.
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
        depths[name] = output.detach()
    return written, depths


def _ramp_correlation(depth: torch.Tensor) -> float:
    """The Pearson correlation between a depth map and the diagonal ramp coordinates (the
    discriminating quantity of {@link _sanity})."""
    plane = ramp_plane(depth.shape[-1])
    if depth.shape[-2:] != plane.shape:
        raise AssertionError(
            f"the output {tuple(depth.shape)} is not square (cannot be paired with the"
            " correlation target)"
        )
    left = (depth.reshape(-1) - depth.mean()).to(torch.float64)
    right = (plane.reshape(-1) - plane.mean()).to(torch.float64)
    return float((left * right).sum() / (left.norm() * right.norm()))


def _assert_distinct(depths: Mapping[str, torch.Tensor]) -> None:
    """The cases' outputs **differ from each other** (identical maps mean the input is not
    getting through)."""
    names = sorted(depths)
    for index, name in enumerate(names):
        for other in names[index + 1 :]:
            if torch.equal(depths[name], depths[other]):
                raise AssertionError(
                    f"the outputs of {name} and {other} are identical (the input has no effect)"
                )


def _region_mean(depth: torch.Tensor, region: tuple[str, float, float, float, float]) -> float:
    """The mean over a relative rectangle of the depth map (as spelled in
    {@link REAL_REGIONS})."""
    _label, y0, y1, x0, x1 = region
    height, width = depth.shape[-2], depth.shape[-1]
    patch = depth[..., int(y0 * height) : int(y1 * height), int(x0 * width) : int(x1 * width)]
    if patch.numel() == 0:
        raise AssertionError(f"region {region} became empty on a {height}×{width} map")
    return float(patch.mean())


def _real_sanity(depths: Mapping[str, torch.Tensor]) -> dict[str, dict[str, float]]:
    """Check the real-image discrimination (near region > far region, as the composition says)
    by **ordering**.

    MUST: check the ordering, with no threshold (the same approach as the synthetic side). The
    rectangles are only pairs determined uniquely by the composition ({@link REAL_REGIONS}), so a
    uniformly collapsed output and an input-independent output both fail, since the two regions'
    means come out equal. A vertical flip, a horizontal flip and an axis mix-up fail here too (the
    near side is always in the lower part / toward the centre of the frame).
    """
    _assert_distinct(depths)
    report: dict[str, dict[str, float]] = {}
    for name, depth in sorted(depths.items()):
        near, far = REAL_REGIONS[name]
        near_mean, far_mean = _region_mean(depth, near), _region_mean(depth, far)
        if near_mean <= far_mean:
            raise AssertionError(
                f"{name}: the mean depth of the near side {near[0]}, {near_mean:.4f}, is not"
                f" above the far side {far[0]}'s {far_mean:.4f} — the near/far layout of the"
                " composition is not captured"
            )
        report[name] = {near[0]: round(near_mean, 4), far[0]: round(far_mean, 4)}
    return report


def _sanity(depths: dict[str, torch.Tensor]) -> dict[str, Any]:
    """Check by **ordering** that the output follows the input's geometry ("sanity" in the
    module docstring).

    MUST: the sanity check must not be vacuous. A value-range check alone (non-negativity is
    nearly trivial after a ReLU) lets both a uniformly collapsed output and an input-independent
    output through. Checking **which case has the largest correlation** fails unless the one image
    with a monotonic depth cue stands out.

    MUST: it applies to **the 4 synthetic cases only** (the real images have
    {@link _real_sanity}). The correlation with the diagonal ramp claims "`ramp` is the largest of
    the 4 synthetic images"; mixing in photographs would let "a photo that happens to be bright
    along the diagonal" take the maximum and turn the discrimination into something else.
    """
    _assert_distinct(depths)
    for name, depth in depths.items():
        minimum, maximum = float(depth.min()), float(depth.max())
        if minimum < 0.0:
            raise AssertionError(
                f"{name}: the minimum {minimum} is negative (contradicts the ReLU at the end of"
                " the head)"
            )
        if not minimum < maximum:
            raise AssertionError(f"the output of {name} is uniform (min == max == {minimum})")

    correlations = {name: _ramp_correlation(depth) for name, depth in depths.items()}
    best = max(correlations, key=lambda name: correlations[name])
    if best != RAMP_CASE:
        raise AssertionError(
            f"the largest correlation with the diagonal ramp is {best!r}'s (expected"
            f" {RAMP_CASE!r}) — the output does not follow the input's geometry: {correlations}"
        )
    return {
        "depth_range": {
            name: [round(float(depth.min()), 3), round(float(depth.max()), 3)]
            for name, depth in depths.items()
        },
        "ramp_correlation": {name: round(value, 4) for name, value in correlations.items()},
    }


def export_series(
    model_dir: Path,
    out_dir: Path,
    real_images: bool = False,
    corpus_root: Path = CORPUS_ROOT,
) -> dict[str, Any]:
    """Write the IR container and the golden io, and return a summary.

    With `real_images` set, the 4 real-image cases are **added** to the 4 synthetic cases (not
    substituted — "The 2 groups of goldens" in the module docstring).

    MUST: the outputs are written to a staging location and put in place only after **every
    check** (input order, sanity) has passed. Placed in the final location before the checks, a
    failed run would leave "assets that pass the acceptance check" behind — the io goldens are
    taken from the same broken wrapper, so they are consistent with each other and the TS-side
    comparison **turns green** (the wording of "when publishing is allowed" is
    {@link _shared.decode_series._publish}; the discipline of swapping in and cleaning up is the
    core primitive {@link karume.artifacts.staged_publication}).
    """
    # Read the provenance before the weights (a checkpoint without a record fails before 99MB are
    # read).
    provenance = upstream_provenance(model_dir)
    wrapper, resolution = load_wrapper(model_dir)
    synthetic = build_cases(resolution)
    # MUST: build the real images before the emit (if an image is missing, stop here instead of
    # failing after writing all 99MB).
    real = build_real_cases(model_dir, resolution, corpus_root) if real_images else ()
    cases = [*synthetic, *((name, pixel_values) for name, pixel_values, _md in real)]
    metadata = {name: md for name, _pixel_values, md in real}
    out_dir.parent.mkdir(parents=True, exist_ok=True)

    _, example = synthetic[0]
    with staged_publication(out_dir) as staged:
        # The writer creates the directory (the primitive does not create it — it only hands over
        # a path).
        staged.mkdir()
        # No dynamic axes (the resolution is fixed to the one pretraining size — module
        # docstring).
        graph = export_to_file(
            wrapper,
            (example,),
            staged / MODEL_FILE,
            provenance=provenance,
            # The graph name is **the component name** (= the weights key of karume.json). It is
            # not derived from the directory name — this family's container sits directly in the
            # series, so the directory name is the series name (`depth-anything-v2-small-hf`),
            # not the component name (container-v1 §2.1).
            graph_name=DEPTH_ANYTHING_ROLE,
            symbol_names=(),
        )
        declared = tuple(item.name for item in graph.inputs)
        if declared != (INPUT_NAME,):
            raise AssertionError(
                f"the graph input order is {declared}, not the expected {(INPUT_NAME,)}"
            )
        written, depths = _write_io(wrapper, graph, cases, staged, metadata)
        # MUST: evaluate before publishing (the only non-vacuous check of this series — on
        # failure the staging location disappears with it). The near/far ordering of the real
        # images is checked inside the same staging location: the synthetic correlation sees only
        # geometric cues, so a regression that stops capturing the near/far layout of
        # photographed compositions is detected by this check alone.
        sanity = _sanity({name: depths[name] for name, _pixel_values in synthetic})
        real_sanity = _real_sanity({name: depths[name] for name in metadata}) if metadata else None
    summary = {
        "dir": str(out_dir),
        "resolution": resolution,
        "nodes": len(graph.nodes),
        "outputs": len(graph.outputs),
        "initializers": len(graph.initializers),
        "model_bytes": sum(p.stat().st_size for p in container_parts(out_dir / MODEL_FILE)),
        "ops": sorted(graph.required_ops),
        "symbols": list(graph.symbols),
        "io": written,
        "input_shape": list(example.shape),
        "output_shape": list(depths[synthetic[0][0]].shape),
        "real_images": {name: md[SOURCE_IMAGE_KEY] for name, md in metadata.items()},
        "sanity": sanity,
    }
    if real_sanity is not None:
        summary["real_sanity"] = real_sanity
    return summary


def _diff_entry(
    stage: str,
    claim: str,
    got: dict[str, torch.Tensor],
    expected: dict[str, torch.Tensor],
) -> dict[str, Any]:
    """The equivalence report of one step (case name → maxdiff, and the bit match over all
    cases).

    `bit_exact` is a stronger claim than "difference 0" (`0.0 == -0.0` has difference 0 but
    different bits), and it matters for steps where **the bit match is the claim itself**, such as
    a rewrite that adds or removes not a single operation.
    """
    return {
        "stage": stage,
        "claim": claim,
        "maxdiff": {
            name: float((got[name] - expected[name]).abs().max()) for name in sorted(expected)
        },
        "bit_exact": all(torch.equal(got[name], expected[name]) for name in expected),
    }


def verify_patches(model_dir: Path) -> list[dict[str, Any]]:
    """Measure the equivalence with the unpatched eager model at **2 points** (the same form as
    `birefnet.export.py --verify`).

    1. Last fusion stage scale factor → explicit size, and the check of the position-embedding
       interpolation (①②) → **bit_exact is the claim itself**. The operation sequences correspond
       one to one, so the difference must be 0, and it fails here otherwise.
    2. `ConvTranspose2d` → 1×1 conv + pixel shuffle (③) → **maxdiff is reported**. The least
       significant bits move by as much as the reduction order along Cin differs from ATen's
       transposed convolution.

    MUST: the order is "**fix the reference values of every case** → 1 → 2". The patches replace
    class attributes process-wide, so retaking the reference per step would make the second
    step's reference a patched value (a vacuous check).
    """
    if patch.patches_applied():
        raise SystemExit(
            "cannot take the reference in a process with patches applied (the equivalence check"
            " would become vacuous with difference 0)"
        )
    wrapper = DepthMap(load_model(model_dir)).eval()
    resolution = patch.pretrained_resolution(wrapper.model)
    cases = build_cases(resolution)

    def depths() -> dict[str, torch.Tensor]:
        with torch.no_grad():
            return {name: wrapper(pixel_values) for name, pixel_values in cases}

    reference = depths()

    patch.apply_layout_patches(wrapper.model)
    layout = _diff_entry("layout", "bit-exact", depths(), reference)
    if not layout["bit_exact"]:
        raise AssertionError(
            f"a rewrite with a one-to-one operation sequence is not bit-identical:"
            f" maxdiff={layout['maxdiff']}"
        )
    patch.apply_module_patches(wrapper.model)
    modules = _diff_entry("modules", "max-abs-diff", depths(), reference)
    # MUST: compare the published measured bound (`measurements.CONVT_MAXDIFF` — the number the
    # card and NOTICE.md state) with the re-measurement. With a report alone, the distributed text
    # would keep stating the old number even when a patch or upstream update widened the
    # difference (W-P6-3).
    worst = max(modules["maxdiff"].values())
    if worst > CONVT_MAXDIFF:
        raise AssertionError(
            f"the maxdiff {worst:g} of the ConvTranspose2d decomposition exceeded the published"
            f" CONVT_MAXDIFF {CONVT_MAXDIFF:g} — update measurements.py to the measurement and"
            " reissue the card and NOTICE.md"
        )
    return [layout, modules]


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output location (default: outputs/series/<model name>/)",
    )
    parser.add_argument(
        "--real-images",
        action="store_true",
        help=f"also write goldens through the 4 real images in `{CORPUS_ROOT.name}/` (added to"
        " the 4 synthetic cases). The preprocessing is the real AutoImageProcessor, and the"
        " source image's sha256 goes into __metadata__",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="measure the equivalence with the unpatched eager model at 2 points (no emit —"
        " in one process the class-attribute replacements contaminate the reference, so the two"
        " cannot be combined)",
    )
    args = parser.parse_args(argv)
    if args.verify:
        for entry in verify_patches(args.model_dir):
            print(
                f"{entry['stage']} ({entry['claim']}): bit_exact={entry['bit_exact']}"
                f" maxdiff={entry['maxdiff']}"
            )
        return
    out_dir = args.out if args.out is not None else default_out_dir(args.model_dir)
    summary = export_series(args.model_dir, out_dir, real_images=args.real_images)
    print(json.dumps({"model_dir": str(args.model_dir), **summary}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
