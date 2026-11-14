r"""**Host-test fixture** for Pillow's LANCZOS (ADR 0121 decision 11, step 9a — tracked in git).

Input / output pairs for checking the TS LANCZOS (an internal function of the Wan2.2 I2V
preprocessing — a verbatim copy of the 22-bit fixed point of Pillow's `Resample.c`) against the
output of Pillow's `Image.resize(size, Image.LANCZOS)` in uint8. Built only from synthetic images
that touch neither model weights nor test images (`inputs/` — not tracked in git).

    cd tools/export-recipes
    uv run --group wan --inexact python -m wan.lanczos_fixture

Two output files (**tracked in git**):

    packages/models/tests/fixtures/wan-i2v/lanczos.json         — description, provenance, cases
    packages/models/tests/fixtures/wan-i2v/lanczos.safetensors  — pixels (U8)

After generating, run `vp fmt packages/models/tests/fixtures/wan-i2v/lanczos.json` from the
repository root (the formatter decides the committed form — the format check of `vp check` also
covers fixtures). The
pixels are not a JSON array because the formatter would expand it to one value per line and exceed
1 MB (the same split as `unipc.json` + `unipc.safetensors` of wan-scheduler).

The safetensors holds 2 tensors per case: `<name>.input` (`[height, width, 3]`) and
`<name>.resized` (`[outHeight, outWidth, 3]`). Both are RGB8, row-major (3 bytes per pixel).

## Case design

Not exhaustive: one case per "boundary where a naive port fails" (the same discipline as
`build_cases` of `siglip2/preprocess.py`) + the 1-D coefficient tables of real use + a random sweep:

- 2-D geometry: upscaling (integer and non-integer factors), downscaling (integer and non-integer
  factors, and a large widening of the support), one axis only (the other axis keeps its size and
  skips its pass — Pillow's `need_horizontal` / `need_vertical`), horizontal downscale with
  vertical upscale, identity (both passes skipped), edge truncation (3×2, 1×1), saturation (a
  checkerboard upscale exceeds 0 and 255 through the negative lobes).
- Coefficient tables of real use (1-row / 1-column high-frequency noise images — the output sweeps
  every column of the coefficient table): 832 → 1280 (width of crop / stretch for 1280×704),
  832 → 2219 (width of crop for 704×1280), 832 → 277 (width of crop for 256×160), 480 → 738 / 1280
  / 160 (height of each crop), 480 → 704 (height of stretch). Only the uint8 output is compared, so
  a ±1 in an integer coefficient is very unlikely to be detected (a shifted coefficient changes the
  output only when the multiply-accumulate crosses a 2^22 boundary — roughly pixel value / 2^22
  per sample). Matching outputs are not evidence that the coefficient tables match.
- Random sweep (fixed seed): {@link SWEEP_CASES} pairs of small sizes.

MUST: import Pillow inside functions (the `wan` group is not in the default sync —
`tests/test_optional_group_imports.py`). The values depend on the Pillow version, so the version is
recorded in the provenance (`wan/tests/test_lanczos_fixture.py` rebuilds with the current Pillow and
compares, turning a fixture that silently drifted with a version update red).
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np

from _shared.paths import REPO_ROOT

#: Where the fixture lives (tracked in git).
FIXTURE_DIR = REPO_ROOT / "packages" / "models" / "tests" / "fixtures" / "wan-i2v"
FIXTURE_JSON = "lanczos.json"
FIXTURE_TENSORS = "lanczos.safetensors"

#: The seed of the synthetic images' random numbers (the noise and the sweep sizes).
SEED = 20261009

#: The number of random sweep cases and the size limits (input side ≤ 32, output side ≤ 48).
SWEEP_CASES = 24
SWEEP_MAX_INPUT = 32
SWEEP_MAX_OUTPUT = 48

#: The limit on the fixture's total size (a small fixture tracked in git — bytes).
MAX_TOTAL_BYTES = 1 << 20


@dataclass(frozen=True)
class LanczosCase:
    """One fixture case (input image and output size)."""

    name: str
    why: str
    image: np.ndarray
    out_height: int
    out_width: int


def _noise(rng: np.random.Generator, height: int, width: int) -> np.ndarray:
    return rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)


def _gradient(height: int, width: int) -> np.ndarray:
    """A smooth pattern (a product of sines with a phase shift per channel)."""
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float64)
    channels = [
        127.5 + 127.5 * np.sin(xx / 5.0 + phase) * np.cos(yy / 3.0 - phase)
        for phase in (0.0, 1.0, 2.0)
    ]
    return np.stack(channels, axis=-1).round().clip(0, 255).astype(np.uint8)


def _checker(height: int, width: int) -> np.ndarray:
    """A checkerboard of single pixels (0 / 255 — the inversion differs per channel)."""
    yy, xx = np.mgrid[0:height, 0:width]
    parity = (yy + xx) % 2
    channels = [parity, 1 - parity, parity]
    return (np.stack(channels, axis=-1) * 255).astype(np.uint8)


def build_cases() -> tuple[LanczosCase, ...]:
    """The fixture cases (order and contents are set by the seed — rebuilding gives the same)."""
    rng = np.random.default_rng(SEED)
    cases = [
        LanczosCase("up-2x", "exactly 2x upscale on both axes", _noise(rng, 7, 11), 14, 22),
        LanczosCase(
            "up-odd", "non-integer upscale on both axes (13×9 → 50×31)", _gradient(9, 13), 31, 50
        ),
        LanczosCase(
            "down-3x",
            "exactly 1/3 downscale on both axes (the support stretches 3x)",
            _noise(rng, 45, 60),
            15,
            20,
        ),
        LanczosCase(
            "down-odd",
            "non-integer downscale on both axes (73×41 → 29×17)",
            _noise(rng, 41, 73),
            17,
            29,
        ),
        LanczosCase(
            "horizontal-only",
            "horizontal downscale only, same height (skips the vertical pass)",
            _noise(rng, 12, 40),
            12,
            23,
        ),
        LanczosCase(
            "vertical-only",
            "vertical upscale only, same width (skips the horizontal pass)",
            _noise(rng, 40, 12),
            57,
            12,
        ),
        LanczosCase(
            "mixed-axes",
            "horizontal downscale, vertical upscale"
            " (catches the axes of the two passes being swapped)",
            _gradient(9, 50),
            27,
            18,
        ),
        LanczosCase(
            "identity",
            "same size on both axes (skips both passes — output = input)",
            _noise(rng, 13, 21),
            13,
            21,
        ),
        LanczosCase(
            "tiny", "3×2 → 7×5 (taps are truncated at both edges)", _noise(rng, 2, 3), 5, 7
        ),
        LanczosCase(
            "single-pixel",
            "1×1 → 4×3 (every output has a single tap)",
            np.array([[[7, 128, 249]]], dtype=np.uint8),
            3,
            4,
        ),
        LanczosCase(
            "checker-ringing",
            "upscale of a 0 / 255 checkerboard"
            " (the negative lobes overshoot 0 and 255; the saturating clip path)",
            _checker(16, 16),
            20,
            24,
        ),
        LanczosCase(
            "large-down",
            "width 120 → 7 (the support stretches 17x and exceeds 100 taps)",
            _gradient(8, 120),
            8,
            7,
        ),
        LanczosCase(
            "cover-up",
            "a small upscale with the ratio of 832×480 → 1280×738 (52×30 → 80×46)",
            _noise(rng, 30, 52),
            46,
            80,
        ),
        LanczosCase(
            "cover-down",
            "a small downscale with the ratio of 832×480 → 277×160 (99×57 → 33×19)",
            _noise(rng, 57, 99),
            19,
            33,
        ),
    ]
    rows = (
        (1280, "width of the 1280×704 crop and stretch"),
        (2219, "width of the 704×1280 crop (2.67x upscale)"),
        (277, "width of the 256×160 crop (1/3 downscale)"),
    )
    for out_width, why in rows:
        cases.append(
            LanczosCase(
                f"row-832-{out_width}",
                f"production coefficient table: {why}",
                _noise(rng, 1, 832),
                1,
                out_width,
            )
        )
    columns = (
        (738, "height of the 1280×704 crop"),
        (1280, "height of the 704×1280 crop"),
        (160, "height of the 256×160 crop (exactly 1/3)"),
        (704, "height of the 1280×704 stretch"),
    )
    for out_height, why in columns:
        cases.append(
            LanczosCase(
                f"column-480-{out_height}",
                f"production coefficient table: {why}",
                _noise(rng, 480, 1),
                out_height,
                1,
            )
        )
    for index in range(SWEEP_CASES):
        height, width = (int(value) for value in rng.integers(1, SWEEP_MAX_INPUT + 1, size=2))
        out_height, out_width = (
            int(value) for value in rng.integers(1, SWEEP_MAX_OUTPUT + 1, size=2)
        )
        cases.append(
            LanczosCase(
                f"sweep-{index:02d}",
                "random geometry (fixed seed)",
                _noise(rng, height, width),
                out_height,
                out_width,
            )
        )
    names = [case.name for case in cases]
    if len(set(names)) != len(names):
        raise AssertionError(f"duplicate case names: {names}")
    return tuple(cases)


def pillow_lanczos(image: np.ndarray, out_height: int, out_width: int) -> np.ndarray:
    """Pillow's `Image.resize((out_width, out_height), Image.LANCZOS)` (RGB8 `[H, W, 3]`)."""
    from PIL import Image

    resized = Image.fromarray(image).resize((out_width, out_height), Image.LANCZOS)
    return np.asarray(resized, dtype=np.uint8)


def build_fixture() -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """The fixture's JSON (description, provenance, cases) and its safetensors tensors."""
    pillow = version("pillow")
    tensors: dict[str, np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    for case in build_cases():
        resized = pillow_lanczos(case.image, case.out_height, case.out_width)
        tensors[f"{case.name}.input"] = np.ascontiguousarray(case.image)
        tensors[f"{case.name}.resized"] = np.ascontiguousarray(resized)
        rows.append(
            {
                "name": case.name,
                "why": case.why,
                "height": int(case.image.shape[0]),
                "width": int(case.image.shape[1]),
                "outHeight": case.out_height,
                "outWidth": case.out_width,
            }
        )
    document = {
        "_doc": [
            "Input / output pairs of Pillow's LANCZOS (Image.resize(size, Image.LANCZOS))"
            " (generated by tools/export-recipes/wan/lanczos_fixture.py).",
            "The source of truth for the resize of the Wan2.2 I2V preprocessing (the official crop"
            " and the diffusers stretch); the TS side copies the 22-bit fixed point of Pillow's"
            " Resample.c verbatim and matches it exactly in uint8.",
            f"The pixels are the U8 tensors of {FIXTURE_TENSORS}: `<name>.input` [height, width, 3]"
            " and `<name>.resized` [outHeight, outWidth, 3] (RGB8, row-major).",
            "row-* / column-* sweep the production 1-D coefficient tables (832 → 1280 / 2219 / 277;"
            " 480 → 738 / 1280 / 160 / 704) with one row / one column of high-frequency noise"
            " (what is compared is the uint8 output — a ±1 in an integer coefficient shows in the"
            " output only when the multiply-accumulate crosses a 2^22 boundary, so this is no"
            " evidence that the coefficient tables match).",
            "The values depend on the Pillow version — regenerate after changing it (the recipe's"
            " wan/tests/test_lanczos_fixture.py checks them against the installed Pillow).",
        ],
        "source": {
            "reference": "PIL.Image.Image.resize(size, PIL.Image.LANCZOS)",
            "pillow": pillow,
            "seed": SEED,
        },
        "tensors": FIXTURE_TENSORS,
        "cases": rows,
    }
    return document, tensors


def emit(fixture_dir: Path) -> dict[str, Any]:
    """Writes the fixture and returns a summary (writes nothing when the total exceeds
    {@link MAX_TOTAL_BYTES})."""
    from safetensors.numpy import save

    document, tensors = build_fixture()
    payload = save(tensors, metadata={"pillow": document["source"]["pillow"]})
    text = json.dumps(document, ensure_ascii=False, indent=2) + "\n"
    total = len(payload) + len(text.encode("utf-8"))
    if total > MAX_TOTAL_BYTES:
        raise AssertionError(
            f"the fixture total of {total} bytes exceeds the limit {MAX_TOTAL_BYTES}"
        )
    fixture_dir.mkdir(parents=True, exist_ok=True)
    (fixture_dir / FIXTURE_TENSORS).write_bytes(payload)
    (fixture_dir / FIXTURE_JSON).write_text(text, encoding="utf-8")
    return {
        "dir": str(fixture_dir),
        "pillow": document["source"]["pillow"],
        "cases": len(document["cases"]),
        "tensor_bytes": len(payload),
        "samples": int(sum(tensor.size for tensor in tensors.values())),
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--fixture-dir", type=Path, default=FIXTURE_DIR)
    args = parser.parse_args(argv)
    print(json.dumps(emit(args.fixture_dir), indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
