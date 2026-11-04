r"""**Parity fixture** for SigLIP2's image preprocessing (resize → rescale → normalize).

Where `siglip2/export.py` emits the **graph**, this script deals only with what comes before it
(raw RGB8 pixels → normalized `pixel_values`). It does not touch the model weights either (it
reads only `preprocessor_config.json`). It is the image counterpart of `irodori/tokenizer_ref.py`,
which plays the same role for text.

    cd tools/export-recipes
    uv run --group siglip2-preprocess python -m siglip2.preprocess

There is one output (**tracked by git**, outside `.gitignore`):

    packages/models/tests/fixtures/image-preprocess/parity.json

After generating it, run `deno fmt packages/models/tests/fixtures/image-preprocess/parity.json`
(the formatter decides the committed form — `fmt --check` in `deno task verify` covers the fixtures
too).

## What the source of truth is

The object that `AutoImageProcessor.from_pretrained(<siglip2 weights>)` returns, that is, the
**`SiglipImageProcessor` (`TorchvisionBackend`)** of transformers 5.14.1. The v5 default is the fast
side, so this is what straightforward user code goes through, not the PIL side
(`SiglipImageProcessorPil`).

MUST: `"resample": 2` in `preprocessor_config.json` is **BILINEAR as a PIL constant**
(`PIL.Image.Resampling`: NEAREST=0 / LANCZOS=1 / **BILINEAR=2** / BICUBIC=3 / BOX=4 /
HAMMING=5). The class attribute defaults to `PILImageResampling.BICUBIC`, which makes "SigLIP is
bicubic" an easy misreading, but **the checkpoint's config overrides the default**. Measured,
applying bicubic is off by up to 47/255. {@link check_processor_shape} checks this 2 on every run.

resize is `torchvision.transforms.v2.functional.resize(..., antialias=True)`. This antialias path
is the same "separable resampling that stretches the support by the scale" as Pillow's
`ImagingResample`:

    scale       = input length / output length
    filterScale = max(scale, 1)
    support     = 1 * filterScale                    # the triangle filter's support radius is 1
    center      = (i + 0.5) * scale
    start       = max(0, floor(center - support + 0.5))
    stop        = min(input length, floor(center + support + 0.5))
    w[j]        = triangle((start + j - center + 0.5) / filterScale)  # normalized to Σw = 1

applied horizontally, then vertically, **rounding each pass's result back to uint8** (because the
uint8 paths of PIL / torchvision keep the intermediate buffer in uint8). The TS-side
`packages/models/src/image/preprocess.ts` writes this form out in f64. The difference in
accumulation precision (the reference uses f32) shifts only the samples at rounding boundaries by
1 LSB, so the TS-side check has a tolerance rather than requiring an exact match — the derivation is
at the top of `packages/models/tests/image_preprocess_test.ts`.

rescale and normalize are in the form folded by
`TorchvisionBackend._fuse_mean_std_and_rescale_factor` (it builds `mean * (1/rescale_factor)` and
`std * (1/rescale_factor)` and applies `(x - mean) / std` once). `1 / 0.00392156862745098` is
exactly 255.0 in f64 (measured).

## What is checked

Everything is measured before the emit, and if even one check fails **nothing is written** (fail
loudly):

- the image processor is a `TorchvisionBackend` (it has not dropped to the PIL side)
- `resample` / `image_mean` / `image_std` / `rescale_factor` / `do_*` are as expected, and
  **neither** center crop nor pad is enabled
- the preprocessing constants other than `size` are **identical** across all checkpoints under
  `inputs/siglip2/` (the substance of the claim that this preprocessing layer serves both 224 and
  384)
- the stored intermediate `resized` is **the real path itself** — passing `resized` through the
  fused normalization gives a result **bit-identical** to the full pipeline's `pixel_values`
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from _shared.paths import REPO_ROOT

from .export import DEFAULT_MODEL_DIR, MODELS_ROOT

#: Parity fixture (**tracked by git**). Read on the Deno side by
#: `packages/models/tests/image_preprocess_test.ts`.
DEFAULT_FIXTURE_PATH = (
    REPO_ROOT / "packages" / "models" / "tests" / "fixtures" / "image-preprocess" / "parity.json"
)

#: `"resample"` of `preprocessor_config.json`. **BILINEAR as a PIL constant** (module docstring).
EXPECTED_RESAMPLE = 2

#: Expected preprocessing constants (the same for both checkpoints — measured by
#: {@link check_sibling_configs}).
EXPECTED_MEAN = (0.5, 0.5, 0.5)
EXPECTED_STD = (0.5, 0.5, 0.5)
EXPECTED_RESCALE = 1.0 / 255.0

#: Fields required to match across all checkpoints (everything except `size`).
SHARED_CONFIG_KEYS = (
    "do_normalize",
    "do_rescale",
    "do_resize",
    "image_mean",
    "image_std",
    "image_processor_type",
    "rescale_factor",
    "resample",
)

#: Random seed of the synthetic images (drawn from one stream in case order — reordering the cases
#: changes the pixels of every case).
SEED = 20260813


def _noise(rng: np.random.Generator, height: int, width: int) -> np.ndarray:
    """Uniformly random RGB8 (high frequency — the form where rounding collisions show most)."""
    return rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)


def _smooth(height: int, width: int) -> np.ndarray:
    """Photo-like continuous-tone RGB8 (low frequency — the form where averaging takes effect on
    downscaling)."""
    yy, xx = np.mgrid[0:height, 0:width]
    red = 127.0 + 120.0 * np.sin(xx / 3.7 + yy / 5.1)
    green = 127.0 + 120.0 * np.cos(yy / 2.9)
    blue = 255.0 * (xx + yy) / max(height + width - 2, 1)
    return np.stack([red, green, blue], axis=-1).clip(0, 255).astype(np.uint8)


def build_cases() -> tuple[dict[str, Any], ...]:
    """The fixture's cases `(name, reason, image, output size)`.

    Designed not for exhaustiveness but to place one case per "boundary where a naive port fails"
    (the same discipline as `ENCODE_CASES` in `irodori/tokenizer_ref.py`). The dimensions are not
    raised to the production 224 / 384 because a single case would then be 150,528 f32 values and
    the fixture would not fit in git — resize is code that is uniform in the dimensions, and its
    only boundary cases are the **scale** (whether `scale >= 1`) and **clipping at the edges**, so
    small dimensions go through the same paths.
    """
    rng = np.random.default_rng(SEED)
    return (
        {
            "name": "down-smooth",
            "why": "両軸 2.75 倍の縮小（台が伸びてタップが 2 本より増える経路）",
            "image": _smooth(22, 33),
            "out": (8, 12),
        },
        {
            "name": "down-noise",
            "why": "同じ縮小を高周波画像で（丸めの衝突が最も出る形 = tolerance の実測根拠）",
            "image": _noise(rng, 22, 33),
            "out": (8, 12),
        },
        {
            "name": "up-4x",
            "why": "拡大（台が 1 に固定され、端では 1 タップに切り詰められる経路）",
            "image": _smooth(5, 7),
            "out": (20, 28),
        },
        {
            "name": "identity",
            "why": "入出力同寸（重みが恒等になること — 縮尺 1 で滲むと高周波画像が崩れる）",
            "image": _noise(rng, 9, 11),
            "out": (9, 11),
        },
        {
            "name": "mixed-axis",
            "why": "縦は 4 倍縮小・横は 3 倍拡大（2 パスの軸取り違えを検出する）",
            "image": _smooth(24, 5),
            "out": (6, 15),
        },
        {
            "name": "half",
            "why": "整数比 2 の縮小（台の端がちょうどサンプル位置に乗る）",
            "image": _noise(rng, 16, 16),
            "out": (8, 8),
        },
        {
            "name": "single-pixel",
            "why": "1×1 入力（全出力が 1 タップに切り詰められる — 重み和 0 の縁）",
            "image": np.full((1, 1, 3), (7, 128, 249), dtype=np.uint8),
            "out": (4, 4),
        },
    )


def load_processor(model_dir: Path) -> Any:
    """Load the `AutoImageProcessor` object (the same path as straightforward user code)."""
    from transformers import AutoImageProcessor

    return AutoImageProcessor.from_pretrained(model_dir)


def check_processor_shape(processor: Any) -> dict[str, Any]:
    """Check that the image processor has the expected type and constants (stop if not)."""
    from transformers.image_processing_backends import TorchvisionBackend

    if not isinstance(processor, TorchvisionBackend):
        raise SystemExit(
            f"the image processor is not a TorchvisionBackend: {type(processor).__mro__}"
            " (the transformers 5.x default is the fast side — on the PIL side the resize"
            " implementation differs)"
        )
    if int(processor.resample) != EXPECTED_RESAMPLE:
        raise SystemExit(
            f"resample is {int(processor.resample)} (expected {EXPECTED_RESAMPLE} = PIL BILINEAR)"
        )
    if tuple(processor.image_mean) != EXPECTED_MEAN or tuple(processor.image_std) != EXPECTED_STD:
        raise SystemExit(f"mean/std is {processor.image_mean} / {processor.image_std}")
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
    }


def check_sibling_configs(models_root: Path) -> dict[str, Any]:
    """Check that everything except `size` is identical across all checkpoints under
    `inputs/siglip2/`.

    This is the substance of the claim that this preprocessing layer serves both base (224) and
    so400m (384). Unless it is measured on every run that only the dimensions differ, a change to
    one checkpoint's constants would silently leave only one side correct.
    """
    found: dict[str, dict[str, Any]] = {}
    for path in sorted(models_root.glob("*/preprocessor_config.json")):
        found[path.parent.name] = json.loads(path.read_text(encoding="utf-8"))
    if not found:
        raise SystemExit(f"no preprocessor_config.json found: {models_root}")
    names = sorted(found)
    base = found[names[0]]
    for name in names[1:]:
        for key in SHARED_CONFIG_KEYS:
            if found[name].get(key) != base.get(key):
                raise SystemExit(
                    f"{name}: '{key}' differs from {names[0]}: "
                    f"{found[name].get(key)!r} != {base.get(key)!r}"
                )
    return {name: found[name]["size"] for name in names}


def reference(
    processor: Any, image: np.ndarray, out_height: int, out_width: int
) -> tuple[np.ndarray, np.ndarray]:
    """Take one case's `(RGB8 after resize, pixel_values f32 [3,H,W])` from the source of truth.

    MUST: `resized` must be **the real path itself** — if a separately called resize differed from
    the full pipeline's intermediate, the TS side would match "an intermediate that exists nowhere".
    This checks that it is bit-identical after the fused normalization.
    """
    from transformers.image_utils import SizeDict

    size = SizeDict(height=out_height, width=out_width)
    tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1)[None]
    resized = processor.resize(image=tensor, size=size, resample=processor.resample)
    if resized.dtype != torch.uint8:
        raise SystemExit(f"the resize output is {resized.dtype} (not uint8)")

    pixel_values = processor(
        images=image,
        size={"height": out_height, "width": out_width},
        input_data_format="channels_last",
        return_tensors="pt",
    )["pixel_values"]

    scale = 1.0 / processor.rescale_factor
    mean = torch.tensor([m * scale for m in processor.image_mean]).view(1, 3, 1, 1)
    std = torch.tensor([s * scale for s in processor.image_std]).view(1, 3, 1, 1)
    fused = (resized.to(dtype=torch.float32) - mean) / std
    if not torch.equal(fused, pixel_values):
        raise SystemExit(
            "the result of resize alone passed through the fused normalization does not match"
            f" pixel_values (max difference {(fused - pixel_values).abs().max().item():.6g})"
        )
    return resized[0].permute(1, 2, 0).numpy(), pixel_values[0].numpy()


def build_fixture(processor: Any, models_root: Path) -> dict[str, Any]:
    """Build the fixture body (returns after passing every case through the source of truth)."""
    cases: list[dict[str, Any]] = []
    for case in build_cases():
        image: np.ndarray = case["image"]
        out_height, out_width = case["out"]
        resized, pixel_values = reference(processor, image, out_height, out_width)
        cases.append(
            {
                "name": case["name"],
                "why": case["why"],
                "height": int(image.shape[0]),
                "width": int(image.shape[1]),
                "outHeight": out_height,
                "outWidth": out_width,
                "input": [int(v) for v in image.reshape(-1)],
                "resized": [int(v) for v in resized.reshape(-1)],
                "pixelValues": [float(v) for v in pixel_values.reshape(-1)],
            }
        )
    return {
        "_doc": [
            "SigLIP2 の画像前処理（resize → rescale → normalize）のパリティ用フィクスチャ",
            "（生成: tools/export-recipes/siglip2/preprocess.py）。",
            "正本は transformers 5.14.1 の SiglipImageProcessor（TorchvisionBackend）で、",
            "resize は torchvision の antialias 経路（PIL と同じ分離型リサンプリング）。",
            "**resample 2 は PIL の定数で BILINEAR**（BICUBIC は 3）— クラス属性の既定が",
            "BICUBIC なので読み違えやすいが、チェックポイントの config が上書きしている。",
            "input は RGB8 の行優先（画素あたり 3 バイト）、resized は resize 直後の RGB8、",
            "pixelValues は [3, outHeight, outWidth] の f32（batch 1 は畳んである）。",
            "寸法を実運用の 224 / 384 まで上げていないのは build_cases の docstring の理由。",
        ],
        "source": {
            "repo": "google/siglip2-*",
            "reference": "transformers.SiglipImageProcessor (TorchvisionBackend)",
            "transformers": "5.14.1",
        },
        "constants": {
            "imageMean": list(EXPECTED_MEAN),
            "imageStd": list(EXPECTED_STD),
            "rescaleFactor": EXPECTED_RESCALE,
        },
        "checkpoints": check_sibling_configs(models_root),
        "cases": cases,
    }


def emit(model_dir: Path, models_root: Path, fixture_path: Path) -> dict[str, Any]:
    """Write the fixture and return a summary (if a check fails, not a single byte is written)."""
    processor = load_processor(model_dir)
    shape = check_processor_shape(processor)
    fixture = build_fixture(processor, models_root)
    fixture_path.parent.mkdir(parents=True, exist_ok=True)
    fixture_path.write_text(
        json.dumps(fixture, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    return {
        "fixture": {"path": str(fixture_path), "bytes": fixture_path.stat().st_size},
        "processor": shape,
        "checkpoints": fixture["checkpoints"],
        "cases": {
            case["name"]: f"{case['height']}x{case['width']}"
            f" -> {case['outHeight']}x{case['outWidth']}"
            for case in fixture["cases"]
        },
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--models-root", type=Path, default=MODELS_ROOT)
    parser.add_argument("--fixture-out", type=Path, default=DEFAULT_FIXTURE_PATH)
    args = parser.parse_args(argv)
    summary = emit(args.model_dir, args.models_root, args.fixture_out)
    print(json.dumps({"model_dir": str(args.model_dir), **summary}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
