r"""**Golden of the upstream procedures** for BiRefNet's α-matte post-processing (ruling 4, ADR
0125 — a fixture tracked in git).

Input / output pairs for checking the TS-side post-processing (the 2 procedures of
`packages/models/src/birefnet/matte.ts`) against the final α produced by each series' upstream
functions themselves. The upstream functions are called as they are, and **only the quantization is
replaced with round-half-up** (ADR 0125 — the upstream truncation is the `astype(uint8)` idiom, not
an intent):

- **HR** (`u8-pil-bicubic` — the step order of the bundled `handler.py:131-137`): in numpy f64,
  `σ = 1 / (1 + exp(−x))` → uint8 by `floor(255σ + 0.5)` → Pillow's `Image.fromarray(q)` (L)
  `.resize(size)` (resample omitted = BICUBIC — `PIL/Image.py:2368`; a copy when the size is the
  same). Upstream's half precision is not copied (it is a choice for speed, and karume's logits are
  an f32 graph output to begin with).
- **Lucida** (`float-bilinear` — the model card's `README.md:86-89`): torch f32 `sigmoid` →
  `transforms.functional.resize(preds[0], (H, W))` (the tensor branch, bilinear, antialias left at
  its default) → `alpha.numpy() * 255` (f32). The `255α` before rounding is kept, and the final α is
  its `floor(x + 0.5)`.

    cd tools/export-recipes
    uv run --group birefnet python -m birefnet.matte_fixture \
        --logits-dir <repository>/outputs/verify/<environment key>/<date>_birefnet

`--logits-dir` is passed only when re-taking the real-image logit crops (the inputs). Without it the
crops of the fixture in git are used as inputs as they are, and only the expected values are
regenerated (it runs on machines without `outputs/` — `birefnet/tests/test_matte_fixture.py` also
checks against the current libraries through this path).

Two outputs (**tracked in git**):

    packages/models/tests/fixtures/birefnet-matte/matte.json
        — description, provenance, ε, inputs, cases
    packages/models/tests/fixtures/birefnet-matte/matte.safetensors
        — input logits and expected values

After generating, run `vp fmt packages/models/tests/fixtures/birefnet-matte/matte.json` from the
repository root (the formatter decides the committed form). The safetensors tensors are
`<input>.logits` (F32 `[h, w]`) per input, and `<case>.hr` (U8 `[H, W]` — HR's final α) and
`<case>.lucida` (F32 `[H, W]` — Lucida's `255α` before rounding) per case.

## Inputs (2 kinds)

- **Crops of real-image logits**: from the default preset's matte logits written by
  `e2e_birefnet_real_test.ts` (the f16 preset for HR, the f32 preset for Lucida —
  {@link REAL_SERIES}; `<series>-photo-{portrait,street}-matte.safetensors` — karume's graph output
  at 1024²), cut a `CROP_WIDTH × CROP_HEIGHT` window holding the most edge pixels
  (`|logit| < SOFT_LOGIT`) from 2 images of each series. From portrait, one row and a 2-wide column
  (length 1024) through the window's center are also taken, to sweep the production coefficient
  tables (1024 → 832 / 480 / 2048). The provenance (file, sha256, window) is kept in the input
  table.
- **Synthetic boundary cases**: one case per boundary where a naive port fails (step, thin line,
  smooth slope, saturation band, checkerboard overshoot, edge truncation, downscale with a much
  wider support). The logits are explicit finite values.

The target sizes are other than 1024² (upscale, downscale, non-square, one axis only, identity).

## Handling of the rounding boundaries

- HR: JS's `Math.exp` and numpy's `exp` can differ in the last ulp. For samples with
  `|255σ − (k + 0.5)| < TIE_MARGIN` the rounding direction can split between implementations, so
  **the generator rejects them** (pick other inputs). Otherwise the contract is 0 mismatches with
  the TS port.
- Lucida: torch's f32 multiply-accumulate and TS's f64 resize round differently. The maximum
  difference between the TS path's `255α` (an f64 port — {@link _ts_float_bilinear_scaled}) and
  torch's `255α` is measured, and the generator fails when it exceeds `LUCIDA_EPSILON`. The TS check
  allows only pixels that "match, or where the golden's `255α` is within ε of `k + 0.5`".

Pillow is imported inside the functions. The reason is to follow the recipe convention (packages a
dependency group declares are imported inside the functions that need them —
`tests/test_optional_group_imports.py` makes it a mechanical check), not that Pillow is missing —
Pillow itself is in the default sync too, through torchvision (a core dependency of the exporter)
(ADR 0125 decision 6). NOTE: the check derives the import name from the distribution name
(`pillow`), so importing `PIL` at module level would not be caught by it. The in-function import is
kept as a convention, without relying on the check.

The values are determined by the versions of torch / torchvision / Pillow / numpy, so the versions
are recorded in the provenance (`birefnet/tests/test_matte_fixture.py` regenerates with the current
versions and compares, turning a fixture that drifted silently with a version update red). Lucida's
last bit also changes with the CPU's instruction set (the SIMD kernel torch picks), so regenerate on
the reference environment (the development machine) — generated elsewhere, only lucida drifts by an
ulp.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np

from _shared.paths import REPO_ROOT

from .card import BIREFNET_CHECKPOINTS
from .distribution import BIREFNET_MODELS, BIREFNET_WEIGHT_DTYPES, birefnet_series_name

#: Location of the fixture (tracked in git).
FIXTURE_DIR = REPO_ROOT / "packages" / "models" / "tests" / "fixtures" / "birefnet-matte"
FIXTURE_JSON = "matte.json"
FIXTURE_TENSORS = "matte.safetensors"

#: Upper limit of the fixture's total size (a small fixture tracked in git — bytes).
MAX_TOTAL_BYTES = 1 << 20

#: Margin at HR's rounding boundary (samples with `|255σ − (k + 0.5)|` below this are rejected). A
#: last-ulp difference of exp is about 1e-13 at 255σ, so this leaves ample orders of magnitude.
TIE_MARGIN = 1e-9

#: Lucida's allowed difference (upper bound of the difference between torch's f32 `255α` and the
#: TS f64 path's `255α`). The smallest power of 10 that is at least twice the measurement
#: ({@link measure_lucida_delta} — 4.2e-3 on 2026-10-11, the largest on the 1024 → 480 downscaled
#: column) (ADR 0125). The main cause of the difference is that torch computes the coordinates and
#: weights in f32 (it matters more at larger indices).
LUCIDA_EPSILON = 1e-2

#: Crop window of the real images (width × height — non-square to catch mixed-up axes).
CROP_WIDTH = 40
CROP_HEIGHT = 32

#: Upper bound of |logit| for a pixel to count as an edge pixel (the band where σ is 0.2%–99.8%).
SOFT_LOGIT = 6.0

#: Lower bound of the logit for saturated foreground (window selection — {@link select_crop}). The
#: aim is to pick a window that surely contains saturated foreground. It sits well above 6.23 (ln
#: 509 — ADR 0125), the boundary where round-half-up quantization makes α 255, so that the window's
#: foreground is not filled only with values near the boundary. 8.32 is about 2 above the boundary
#: (1 − σ is about 1/8 of its value at the boundary). Changing it can change the window choice, so
#: regenerate the fixture if you change it.
SATURATED_LOGIT = 8.32

#: Minimum number of foreground and of background pixels a window contains (one window row).
MIN_SIDE_PIXELS = CROP_WIDTH

#: Stride of the window search (pixels).
CROP_STRIDE = 4

#: Sources of the real-image crops = (spelling of the series the e2e writes — `artifactName` of
#: `e2e_birefnet_real_test.ts`, prefix of the case names). The series of each checkpoint's default
#: distribution preset (HR has only the f16 preset — ADR 0113 addendum 2026-10-11; the logits of
#: HR's f16 preset are identical, down to the sha256, to those of the retired f32 preset). The
#: preset is derived from the series name ({@link real_series_seat}).
#:
#: The case-name prefix is not derived from the series name because case names lie outside the
#: fixture's provenance — when the source series (preset) changes they stay `real-birefnet-hr-*`,
#: keeping the regeneration diff within the provenance (file names).
REAL_SERIES: tuple[tuple[str, str], ...] = (
    ("birefnet-hr-1024-f16", "birefnet-hr"),
    ("lucida-1024", "lucida"),
)
REAL_IMAGES = ("photo-portrait", "photo-street")

#: The image the row and column crops (one row and one column sweeping the production coefficient
#: tables) are taken from.
LINE_IMAGE = "photo-portrait"

#: Width of the column crop (width 1 hits a torch bug — `REAL_TARGETS` below).
COLUMN_WIDTH = 2

#: One target-size row (name, width, height, intent).
Target = tuple[str, int, int, str]

#: Kind of real-image crop → target sizes.
#:
#: - `crop` (window `CROP_WIDTH × CROP_HEIGHT`): upscale, downscale, non-square, one axis only,
#:   identity.
#: - `row` / `column` (one row and a 2-wide column through the window's center — length 1024): the
#:   production coefficient tables (the 1024 → 832 / 480 downscales and the 1024 → 2048 upscale).
#:   Rounding of torch's f32 coordinates matters more at larger indices, so ε is measured at
#:   production lengths (measured only at window size it comes out small). The column is 2 wide
#:   because torch 2.13.0's CPU antialiased bilinear breaks the vertical interpolation on a
#:   **width-1 input** (every row becomes the first value — measured; an upstream bug, not something
#:   to copy — ADR 0125; {@link lucida_scaled} rejects it).
REAL_TARGETS: dict[str, tuple[Target, ...]] = {
    "crop": (
        ("up", 97, 83, "non-integer upscale on both axes"),
        ("down", 17, 13, "non-integer downscale on both axes (the support widens)"),
        (
            "mixed",
            25,
            61,
            "horizontal downscale, vertical upscale (catches swapped axes between the 2 passes)",
        ),
        (
            "horizontal-only",
            87,
            CROP_HEIGHT,
            "horizontal upscale only, same height (skips the vertical pass)",
        ),
        (
            "vertical-only",
            CROP_WIDTH,
            15,
            "vertical downscale only, same width (skips the horizontal pass)",
        ),
        (
            "identity",
            CROP_WIDTH,
            CROP_HEIGHT,
            "same size on both axes (resize is a copy — only the quantization remains)",
        ),
    ),
    "row": (
        ("down-832", 832, 1, "production coefficient table: 1024 → 832 (the width of 832×480)"),
        ("up-2048", 2048, 1, "production coefficient table: 1024 → 2048 (the width of 2048²)"),
    ),
    "column": (
        (
            "down-480",
            COLUMN_WIDTH,
            480,
            "production coefficient table: 1024 → 480 (the height of 832×480)",
        ),
        (
            "up-2048",
            COLUMN_WIDTH,
            2048,
            "production coefficient table: 1024 → 2048 (the height of 2048²)",
        ),
    ),
}


@dataclass(frozen=True)
class MatteInput:
    """One fixture input (logit map `[h, w]`, the target sizes applied, provenance)."""

    name: str
    why: str
    logits: np.ndarray
    targets: tuple[Target, ...]
    provenance: dict[str, Any] | None = None


@dataclass(frozen=True)
class MatteCase:
    """One fixture case (input × target size)."""

    name: str
    why: str
    input: str
    out_width: int
    out_height: int


def _row(values: Sequence[float]) -> np.ndarray:
    return np.asarray([values], dtype=np.float32)


def _checker(height: int, width: int, value: float) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width]
    return np.where((yy + xx) % 2 == 0, value, -value).astype(np.float32)


def _disc(height: int, width: int, radius: float, slope: float) -> np.ndarray:
    """A centered disc (logit = `slope · (radius − distance from the center)` — foreground with a
    soft edge)."""
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float64)
    distance = np.hypot(xx - (width - 1) / 2, yy - (height - 1) / 2)
    return (slope * (radius - distance)).astype(np.float32)


def _saturation_band(height: int, width: int) -> np.ndarray:
    """A map tiled row-major with logits 5.5–18 (crossing 6.23, the boundary where round-half-up
    gives 255, and 8.32 / 16.64, the truncation boundaries of upstream's half precision / f32)."""
    band = np.linspace(5.5, 18.0, height * width, dtype=np.float64)
    return band.reshape(height, width).astype(np.float32)


def _wide_ramp(width: int) -> np.ndarray:
    """2 rows of width `width` (a slope with a step — checks the sums of a downscale with a much
    wider support)."""
    xs = np.arange(width, dtype=np.float64)
    row = np.where(xs < width / 2, -9.0 + xs * 0.05, 7.5 - (xs - width / 2) * 0.03)
    return np.stack([row, row[::-1]]).astype(np.float32)


def synthetic_inputs() -> tuple[MatteInput, ...]:
    """Synthetic boundary cases (the logits are explicit finite values)."""
    return (
        MatteInput(
            "step",
            "a 0 / 255 step, 4 → 8 (one of the measured examples of R6B-001)",
            _row([-12.0, -12.0, 12.0, 12.0]),
            (
                (
                    "up",
                    8,
                    1,
                    "4 → 8 upscale (bicubic's negative lobes go past 0 and 255 and are clipped"
                    " by saturation)",
                ),
            ),
        ),
        MatteInput(
            "ramp",
            "a smooth ramp, 8 → 21 (one of the measured examples of R6B-001)",
            _row(np.linspace(-5.75, 6.25, 8).tolist()),
            (("up", 21, 1, "8 → 21 upscale"),),
        ),
        MatteInput(
            "thin",
            "a thin foreground, 6 → 15 (one of the measured examples of R6B-001 — the largest"
            " difference)",
            _row([-12.0, -12.0, 12.0, -12.0, -12.0, -12.0]),
            (("up", 15, 1, "6 → 15 upscale"),),
        ),
        MatteInput(
            "saturation",
            "a saturation band (logits 5.5–18 — crosses the boundary where the value becomes 255)",
            _saturation_band(4, 12),
            (
                ("up", 29, 9, "upscale"),
                ("down", 5, 3, "downscale"),
                ("identity", 12, 4, "identity (only the quantization boundaries)"),
            ),
        ),
        MatteInput(
            "checker",
            "a ±12 checkerboard (the path where bicubic's overshoot is clipped at 0 / 255)",
            _checker(8, 8, 12.0),
            (("up", 19, 20, "upscale"),),
        ),
        MatteInput(
            "disc",
            "a disc with a soft edge",
            _disc(29, 33, 8.7, 1.5),
            (("down", 7, 6, "downscale (the support widens 4–5×)"), ("up", 50, 44, "upscale")),
        ),
        MatteInput(
            "wide",
            "a ramp with a step, width 120",
            _wide_ramp(120),
            (("down", 7, 2, "width 120 → 7 (the support widens 17×, same height)"),),
        ),
        MatteInput(
            "tiny",
            "3×2 (taps are truncated at both ends)",
            np.asarray([[-3.25, 1.5, 7.25], [9.5, -0.75, 2.25]], dtype=np.float32),
            (("up", 7, 5, "3×2 → 7×5"),),
        ),
        MatteInput(
            "single",
            "1×1 (every output is 1 tap)",
            np.asarray([[2.5]], dtype=np.float32),
            (("up", 4, 3, "1×1 → 4×3"),),
        ),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_crop(logits: np.ndarray) -> tuple[int, int]:
    """Top-left `(x, y)` of the window with the most edge pixels (ties go to the first in row-major
    order).

    MUST: only windows with at least {@link MIN_SIDE_PIXELS} each of saturated foreground (pixels
    with a logit above {@link SATURATED_LOGIT}) and of background are candidates — choosing by the
    number of edge pixels alone drifts to windows that are merely soft without saturation (haze at
    the tips of hair and the like), and the fixture would not sweep the rounding of the saturation
    band.
    """
    soft = (np.abs(logits) < SOFT_LOGIT).astype(np.int64)
    foreground = (logits > SATURATED_LOGIT).astype(np.int64)
    background = (logits < -SOFT_LOGIT).astype(np.int64)

    def window_sums(mask: np.ndarray) -> np.ndarray:
        table = np.pad(mask.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
        ys = np.arange(0, logits.shape[0] - CROP_HEIGHT + 1, CROP_STRIDE)
        xs = np.arange(0, logits.shape[1] - CROP_WIDTH + 1, CROP_STRIDE)
        y0, x0 = np.meshgrid(ys, xs, indexing="ij")
        y1, x1 = y0 + CROP_HEIGHT, x0 + CROP_WIDTH
        return table[y1, x1] - table[y0, x1] - table[y1, x0] + table[y0, x0]

    eligible = (window_sums(foreground) >= MIN_SIDE_PIXELS) & (
        window_sums(background) >= MIN_SIDE_PIXELS
    )
    if not eligible.any():
        raise ValueError(
            "no window contains both saturated foreground and background"
            " (a matte without edges was passed)"
        )
    score = np.where(eligible, window_sums(soft), -1)
    row, column = np.unravel_index(int(np.argmax(score)), score.shape)
    return int(column) * CROP_STRIDE, int(row) * CROP_STRIDE


def _real_input(
    name: str, why: str, logits: np.ndarray, kind: str, source: dict[str, Any], rect: dict[str, int]
) -> MatteInput:
    window = logits[rect["y"] : rect["y"] + rect["height"], rect["x"] : rect["x"] + rect["width"]]
    return MatteInput(
        name,
        why,
        np.ascontiguousarray(window, dtype=np.float32),
        REAL_TARGETS[kind],
        {"kind": kind, **source, **rect},
    )


def real_series_seat(series: str) -> tuple[str, str, str]:
    """Series name → (checkpoint, model name = resolution, storage dtype of the preset).

    It is traced back by evaluating the source of truth of series-name spelling
    (`birefnet.distribution.birefnet_series_name`) for every combination — keeping the preset
    separately in {@link REAL_SERIES} would leave only the provenance's preset stale on the day the
    series changes.
    """
    matches = [
        (checkpoint, model, dtype)
        for checkpoint in BIREFNET_CHECKPOINTS
        for model in BIREFNET_MODELS
        for dtype in BIREFNET_WEIGHT_DTYPES
        if birefnet_series_name(checkpoint, model, dtype) == series
    ]
    if len(matches) != 1:
        raise ValueError(
            f"series name '{series}' does not match exactly one BiRefNet series spelling: {matches}"
        )
    return matches[0]


def extract_real_inputs(logits_dir: Path) -> tuple[MatteInput, ...]:
    """Make the real-image crops from the matte logits the e2e wrote (with provenance)."""
    from safetensors.numpy import load_file

    inputs = []
    for series, case in REAL_SERIES:
        _checkpoint, _model, seat = real_series_seat(series)
        for image in REAL_IMAGES:
            path = logits_dir / f"{series}-{image}-matte.safetensors"
            tensors = load_file(path)
            if len(tensors) != 1:
                raise ValueError(f"{path}: {len(tensors)} tensors (expected the one graph output)")
            ((tensor_name, tensor),) = tensors.items()
            if tensor.dtype != np.float32 or tensor.ndim != 4 or tensor.shape[:2] != (1, 1):
                raise ValueError(f"{path}: {tensor_name} is not F32 [1, 1, H, W] ({tensor.shape})")
            logits = tensor[0, 0]
            height, width = logits.shape
            source = {
                "file": f"outputs/verify/{logits_dir.parent.name}/{logits_dir.name}/{path.name}",
                "sha256": _sha256(path),
                "tensor": tensor_name,
            }
            stem = f"real-{case}-{image.removeprefix('photo-')}"
            why = f"real-image logits ({series}, {seat} preset, {image})"
            x, y = select_crop(logits)
            inputs.append(
                _real_input(
                    stem,
                    f"{why}, window near the edge",
                    logits,
                    "crop",
                    source,
                    {"x": x, "y": y, "width": CROP_WIDTH, "height": CROP_HEIGHT},
                )
            )
            if image != LINE_IMAGE:
                continue
            center_x, center_y = x + CROP_WIDTH // 2, y + CROP_HEIGHT // 2
            inputs.append(
                _real_input(
                    f"{stem}-row",
                    f"{why}, one row through the window's center",
                    logits,
                    "row",
                    source,
                    {"x": 0, "y": center_y, "width": width, "height": 1},
                )
            )
            inputs.append(
                _real_input(
                    f"{stem}-column",
                    f"{why}, {COLUMN_WIDTH}-wide column through the window's center",
                    logits,
                    "column",
                    source,
                    {"x": center_x, "y": 0, "width": COLUMN_WIDTH, "height": height},
                )
            )
    return tuple(inputs)


def committed_real_inputs(fixture_dir: Path = FIXTURE_DIR) -> tuple[MatteInput, ...]:
    """The real-image crops of the fixture in git (the input table's provenance and
    `<input>.logits`)."""
    from safetensors.numpy import load_file

    document = json.loads((fixture_dir / FIXTURE_JSON).read_text(encoding="utf-8"))
    tensors = load_file(fixture_dir / document["tensors"])
    return tuple(
        MatteInput(
            row["name"],
            row["why"],
            tensors[f"{row['name']}.logits"],
            REAL_TARGETS[row["provenance"]["kind"]],
            row["provenance"],
        )
        for row in document["inputs"]
        if row["provenance"] is not None
    )


def build_cases(
    real_inputs: Sequence[MatteInput],
) -> tuple[tuple[MatteInput, ...], tuple[MatteCase, ...]]:
    """The input table and the case table (real images, then synthetic — the same order on every
    regeneration)."""
    inputs = (*real_inputs, *synthetic_inputs())
    cases = tuple(
        MatteCase(f"{entry.name}.{label}", why, entry.name, width, height)
        for entry in inputs
        for label, width, height, why in entry.targets
    )
    names = [entry.name for entry in inputs] + [case.name for case in cases]
    if len(set(names)) != len(names):
        raise AssertionError(f"duplicate names: {names}")
    return inputs, cases


# ---- Upstream procedures (only the quantization replaced with round-half-up) ---------------------


def round_half_up(scaled: np.ndarray) -> np.ndarray:
    """`floor(x + 0.5)` to uint8 (0.5 rounds up — ADR 0125's quantization)."""
    return np.floor(scaled.astype(np.float64) + 0.5).astype(np.uint8)


def truncate(scaled: np.ndarray) -> np.ndarray:
    """Upstream's quantization itself (the truncation of `(x * 255).astype(uint8)` — kept only for
    comparison)."""
    return scaled.astype(np.uint8)


def sigmoid_f64(logits: np.ndarray) -> np.ndarray:
    """f64 `1 / (1 + exp(−x))` (the same formula as TS's `1 / (1 + Math.exp(-raw))`)."""
    return 1.0 / (1.0 + np.exp(-logits.astype(np.float64)))


def assert_no_ties(scaled: np.ndarray, where: str) -> None:
    """Reject samples whose rounding direction can split between exp implementations (less than
    {@link TIE_MARGIN} from `k + 0.5`)."""
    distance = np.abs(scaled - (np.floor(scaled) + 0.5))
    near = np.argwhere(distance < TIE_MARGIN)
    if near.size:
        raise ValueError(
            f"{where}: samples where 255σ is less than {TIE_MARGIN} from k + 0.5"
            f" ({near[:4].tolist()})"
            " — the rounding direction can split between exp implementations, so pick other inputs"
        )


def hr_alpha(
    logits: np.ndarray,
    out_width: int,
    out_height: int,
    quantize: Callable[[np.ndarray], np.ndarray] = round_half_up,
    *,
    allow_ties: bool = False,
) -> np.ndarray:
    """HR's upstream procedure (f64 σ → 8 bit → Pillow's `resize(size)` default = BICUBIC).

    `allow_ties` rounds tie samples ({@link assert_no_ties}) half up instead of rejecting them —
    used only when the caller excludes the output pixels tie samples reach itself
    (`birefnet.matte_ab_check`).
    """
    from PIL import Image

    scaled = sigmoid_f64(logits) * 255
    if quantize is round_half_up and not allow_ties:
        assert_no_ties(scaled, "HR")
    image = Image.fromarray(quantize(scaled))
    if image.mode != "L":
        raise AssertionError(f"the 8-bit single band does not open as L ({image.mode})")
    return np.asarray(image.resize((out_width, out_height)), dtype=np.uint8)


def lucida_scaled(logits: np.ndarray, out_width: int, out_height: int) -> np.ndarray:
    """`255α` before rounding of Lucida's upstream procedure (torch f32 σ → TF.resize → `× 255`)."""
    import torch
    from torchvision.transforms import functional as transforms_functional

    height, width = logits.shape
    if width == 1 and height > 1:
        # torch 2.13.0's CPU antialiased bilinear breaks the vertical axis of inputs of width 1 and
        # height 2 or more (every row gets the first row's value — measured). The upstream bug is
        # not baked into the golden (ADR 0125 — bugs are not copied).
        raise ValueError(f"an input of width 1 and height {height} hits torch's antialias bug")
    preds = torch.from_numpy(np.ascontiguousarray(logits, dtype=np.float32))[None, None].sigmoid()
    alpha = transforms_functional.resize(preds[0], [out_height, out_width]).squeeze(0)
    scaled = alpha.numpy() * 255
    if scaled.dtype != np.float32:
        raise AssertionError(f"255α is not f32 ({scaled.dtype})")
    return scaled


# ---- Port of the TS float-bilinear path (used only to measure ε) -------------------------------


def _bilinear_taps(in_size: int, out_size: int) -> list[tuple[int, np.ndarray]]:
    """Port of `buildTaps` (triangle filter) in `preprocess.ts`."""
    scale = in_size / out_size
    filter_scale = max(scale, 1.0)
    support = 1.0 * filter_scale
    taps = []
    for index in range(out_size):
        center = (index + 0.5) * scale
        start = max(0, int(np.floor(center - support + 0.5)))
        stop = min(in_size, int(np.floor(center + support + 0.5)))
        positions = (np.arange(start, stop, dtype=np.float64) - center + 0.5) / filter_scale
        weights = np.maximum(0.0, 1.0 - np.abs(positions))
        taps.append((start, weights / weights.sum()))
    return taps


def _ts_float_bilinear_scaled(logits: np.ndarray, out_width: int, out_height: int) -> np.ndarray:
    """`255α` before rounding of `float-bilinear` in `matte.ts`.

    σ computed in f64 and stored as f32 → f64 resize → stored as f32 → × 255 in f64
    (`resizePlaneF32` returns an f32 array).
    """
    sigma = sigmoid_f64(logits).astype(np.float32).astype(np.float64)
    height, width = sigma.shape
    middle = np.empty((height, out_width), dtype=np.float64)
    for x, (start, weights) in enumerate(_bilinear_taps(width, out_width)):
        middle[:, x] = sigma[:, start : start + len(weights)] @ weights
    out = np.empty((out_height, out_width), dtype=np.float64)
    for y, (start, weights) in enumerate(_bilinear_taps(height, out_height)):
        out[y] = weights @ middle[start : start + len(weights)]
    return out.astype(np.float32).astype(np.float64) * 255


def measure_lucida_delta(
    inputs: Sequence[MatteInput], cases: Sequence[MatteCase], scaled: dict[str, np.ndarray]
) -> float:
    """Maximum difference between torch's `255α` and the TS path's `255α` (all cases)."""
    by_name = {entry.name: entry for entry in inputs}
    worst = 0.0
    for case in cases:
        ours = _ts_float_bilinear_scaled(
            by_name[case.input].logits, case.out_width, case.out_height
        )
        worst = max(worst, float(np.max(np.abs(ours - scaled[case.name].astype(np.float64)))))
    return worst


# ---- fixture ------------------------------------------------------------------------------


def library_versions() -> dict[str, str]:
    """Versions of the libraries that determine the expected values (the fixture's provenance — the
    test compares them with the current environment)."""
    return {name: version(name) for name in ("torch", "torchvision", "pillow", "numpy")}


def build_fixture(
    real_inputs: Sequence[MatteInput],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """The fixture's JSON (description, provenance, ε, input table, case table) and the safetensors
    tensors."""
    inputs, cases = build_cases(real_inputs)
    by_name = {entry.name: entry for entry in inputs}
    tensors: dict[str, np.ndarray] = {}
    for entry in inputs:
        tensors[f"{entry.name}.logits"] = np.ascontiguousarray(entry.logits, dtype=np.float32)
    scaled: dict[str, np.ndarray] = {}
    for case in cases:
        logits = by_name[case.input].logits
        tensors[f"{case.name}.hr"] = hr_alpha(logits, case.out_width, case.out_height)
        scaled[case.name] = lucida_scaled(logits, case.out_width, case.out_height)
        tensors[f"{case.name}.lucida"] = np.ascontiguousarray(scaled[case.name])
    measured = measure_lucida_delta(inputs, cases, scaled)
    if not measured <= LUCIDA_EPSILON / 2:
        raise AssertionError(
            f"the 255α difference {measured} between torch and the TS path exceeds half of ε"
            f" {LUCIDA_EPSILON} (choose ε again — ADR 0125)"
        )
    document = {
        "_doc": [
            "Golden of the upstream post-processing steps of the BiRefNet α matte"
            " (generated by tools/export-recipes/birefnet/matte_fixture.py — ADR 0125).",
            "It calls the upstream functions as they are, with only the quantization replaced by"
            " round-half-up (floor(x + 0.5)).",
            "hr: numpy f64 σ → floor(255σ + 0.5) as uint8 → PIL.Image.fromarray(q) (L)"
            ".resize((W, H)) (resample omitted = BICUBIC). The contract is 0 mismatches with TS's"
            " u8-pil-bicubic (the generator rejects samples whose 255σ is less than tieMargin from"
            " k + 0.5).",
            "lucida: torch f32 sigmoid → torchvision.transforms.functional.resize(preds[0], "
            "(H, W)) (bilinear, antialias default) → × 255 (f32). The final α is floor(x + 0.5)."
            " TS's float-bilinear matches, or may be off by one step only where the golden's 255α"
            " is within epsilon of k + 0.5.",
            f"Tensors in {FIXTURE_TENSORS}: `<input>.logits` (F32 [h, w]), `<case>.hr`"
            " (U8 [H, W]), `<case>.lucida` (F32 [H, W] — 255α before rounding).",
            "The values are decided by the torch / torchvision / Pillow / numpy versions —"
            " regenerate when a version changes (the recipe's birefnet/tests/test_matte_fixture.py"
            " compares them with the current versions).",
            "lucida's last bit also changes with the CPU instruction set — regenerate and compare"
            " on the reference environment (the development machine) (docs/limitations.md).",
        ],
        "source": {
            "hr": "PIL.Image.Image.resize((W, H)) on PIL.Image.fromarray(floor(255σ_f64 + 0.5))",
            "lucida": (
                "torchvision.transforms.functional.resize(sigmoid_f32(x)[1, h, w], (H, W)) * 255"
            ),
            "versions": library_versions(),
        },
        "tieMargin": TIE_MARGIN,
        "lucida": {"epsilon": LUCIDA_EPSILON, "measuredMaxDelta": measured},
        "tensors": FIXTURE_TENSORS,
        "inputs": [
            {
                "name": entry.name,
                "why": entry.why,
                "width": int(entry.logits.shape[1]),
                "height": int(entry.logits.shape[0]),
                "provenance": entry.provenance,
            }
            for entry in inputs
        ],
        "cases": [
            {
                "name": case.name,
                "why": case.why,
                "input": case.input,
                "outWidth": case.out_width,
                "outHeight": case.out_height,
            }
            for case in cases
        ],
    }
    return document, tensors


def emit(fixture_dir: Path, real_inputs: Sequence[MatteInput]) -> dict[str, Any]:
    """Write the fixture and return a summary (nothing is written when the total exceeds
    {@link MAX_TOTAL_BYTES})."""
    from safetensors.numpy import save

    document, tensors = build_fixture(real_inputs)
    # NOTE: the versions are written only to the JSON's `source.versions` — the key order of
    # safetensors metadata changes on every write (measured with 2 or more keys), which would break
    # the byte equality of regeneration.
    payload = save(tensors)
    text = json.dumps(document, ensure_ascii=False, indent=2) + "\n"
    total = len(payload) + len(text.encode("utf-8"))
    if total > MAX_TOTAL_BYTES:
        raise AssertionError(
            f"the fixture's total of {total} bytes exceeds the limit {MAX_TOTAL_BYTES}"
        )
    fixture_dir.mkdir(parents=True, exist_ok=True)
    (fixture_dir / FIXTURE_TENSORS).write_bytes(payload)
    (fixture_dir / FIXTURE_JSON).write_text(text, encoding="utf-8")
    return {
        "dir": str(fixture_dir),
        "versions": document["source"]["versions"],
        "inputs": len(document["inputs"]),
        "cases": len(document["cases"]),
        "lucida": document["lucida"],
        "tensor_bytes": len(payload),
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--fixture-dir", type=Path, default=FIXTURE_DIR)
    parser.add_argument(
        "--logits-dir",
        type=Path,
        default=None,
        help="the e2e output to re-take the real-image logits from"
        " (outputs/verify/<environment key>/<date>_birefnet)",
    )
    args = parser.parse_args(argv)
    real_inputs = (
        extract_real_inputs(args.logits_dir.resolve())
        if args.logits_dir is not None
        else committed_real_inputs(args.fixture_dir)
    )
    print(json.dumps(emit(args.fixture_dir, real_inputs), indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
