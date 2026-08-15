r"""SigLIP2 の画像前処理（resize → rescale → normalize）の**パリティ用フィクスチャ**。

`siglip2/export.py` が**グラフ**を出すのに対し、こちらが扱うのはグラフの手前
（生の RGB8 画素 → 正規化済み `pixel_values`）だけ。モデルの重みにも触らない
（読むのは `preprocessor_config.json` だけ）。`irodori/tokenizer_ref.py` と同じ役割の台本を、
テキストではなく画像で持つ。

    cd tools/export-recipes
    uv run --group siglip2-preprocess python -m siglip2.preprocess

出力は 1 本（**git 管理**・`.gitignore` の外）:

    packages/models/tests/fixtures/image-preprocess/parity.json

生成後は `deno fmt packages/models/tests/fixtures/image-preprocess/parity.json` を掛ける
（commit 形はフォーマッタが正 — `deno task verify` の `fmt --check` が fixtures も見る）。

## 正本は何か

`AutoImageProcessor.from_pretrained(<siglip2 の重み>)` が返す実体、つまり transformers
5.14.1 の **`SiglipImageProcessor`（`TorchvisionBackend`）**。v5 の既定は fast 側なので、
利用者が素直に書いたコードが通るのはこちらで、PIL 側（`SiglipImageProcessorPil`）ではない。

MUST: `preprocessor_config.json` の `"resample": 2` は **PIL の定数で BILINEAR**
（`PIL.Image.Resampling`: NEAREST=0 / LANCZOS=1 / **BILINEAR=2** / BICUBIC=3 / BOX=4 /
HAMMING=5）。クラス属性の既定が `PILImageResampling.BICUBIC` なので「SigLIP は bicubic」と
読み違えやすいが、**チェックポイントの config が既定を上書きしている**。実測でも bicubic を
当てると最大 47/255 ずれる。{@link check_processor_shape} が毎回この 2 を実測する。

resize は `torchvision.transforms.v2.functional.resize(..., antialias=True)`。この
antialias 経路は Pillow の `ImagingResample` と同じ「support を縮尺で伸ばす分離型
リサンプリング」で、

    scale       = 入力長 / 出力長
    filterScale = max(scale, 1)
    support     = 1 * filterScale                    # 三角フィルタの台の半径は 1
    center      = (i + 0.5) * scale
    start       = max(0, floor(center - support + 0.5))
    stop        = min(入力長, floor(center + support + 0.5))
    w[j]        = triangle((start + j - center + 0.5) / filterScale)  # Σw = 1 に正規化

を横 → 縦の順に掛け、**各パスの結果を uint8 へ丸め直す**（PIL / torchvision の uint8 経路が
中間バッファを uint8 で持つため）。TS 側 `packages/models/src/image/preprocess.ts` はこの形を
f64 で書き下したもの。積算精度の違い（参照は f32）で丸め境界の標本だけが 1 LSB ずれるので、
TS 側の門は完全一致ではなく幅を持つ — 導出は
`packages/models/tests/image_preprocess_test.ts` 冒頭。

rescale と normalize は `TorchvisionBackend._fuse_mean_std_and_rescale_factor` が畳んだ形
（`mean * (1/rescale_factor)` と `std * (1/rescale_factor)` を作って `(x - mean) / std` を
1 回だけ掛ける）。`1 / 0.00392156862745098` は f64 でちょうど 255.0 になる（実測）。

## 何を門にするか

emit の前に全て実測し、1 つでも外れたら**何も書かない**（fail loudly）:

- image processor の実体が `TorchvisionBackend` であること（PIL 側に落ちていない）
- `resample` / `image_mean` / `image_std` / `rescale_factor` / `do_*` が期待どおりで、
  center crop も pad も**入っていない**こと
- `inputs/siglip2/` 配下の全チェックポイントで、`size` 以外の前処理定数が**同一**であること
  （この前処理層は 224 と 384 の両方に効く、という主張の中身）
- 保存する中間 `resized` が**実経路そのもの**であること — `resized` を融合正規化に通した
  結果が、フルパイプラインの `pixel_values` と**ビット同一**であること
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

#: パリティ用フィクスチャ（**git 管理**）。Deno 側
#: `packages/models/tests/image_preprocess_test.ts` が読む。
DEFAULT_FIXTURE_PATH = (
    REPO_ROOT / "packages" / "models" / "tests" / "fixtures" / "image-preprocess" / "parity.json"
)

#: `preprocessor_config.json` の `"resample"`。**PIL の定数で BILINEAR**（モジュール docstring）。
EXPECTED_RESAMPLE = 2

#: 前処理定数の期待値（2 チェックポイントとも同じ — {@link check_sibling_configs} が実測）。
EXPECTED_MEAN = (0.5, 0.5, 0.5)
EXPECTED_STD = (0.5, 0.5, 0.5)
EXPECTED_RESCALE = 1.0 / 255.0

#: `size` 以外は全チェックポイントで一致することを要求する欄。
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

#: 合成画像の乱数種（ケース順に 1 本の系列から引く — 順序を変えると全ケースの画素が変わる）。
SEED = 20260813


def _noise(rng: np.random.Generator, height: int, width: int) -> np.ndarray:
    """一様乱数の RGB8（高周波 — 丸めの衝突が最も出る形）。"""
    return rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)


def _smooth(height: int, width: int) -> np.ndarray:
    """写真に近い連続階調の RGB8（低周波 — 縮小で平均が効く形）。"""
    yy, xx = np.mgrid[0:height, 0:width]
    red = 127.0 + 120.0 * np.sin(xx / 3.7 + yy / 5.1)
    green = 127.0 + 120.0 * np.cos(yy / 2.9)
    blue = 255.0 * (xx + yy) / max(height + width - 2, 1)
    return np.stack([red, green, blue], axis=-1).clip(0, 255).astype(np.uint8)


def build_cases() -> tuple[dict[str, Any], ...]:
    """フィクスチャのケース `(名前, 理由, 画像, 出力寸法)`。

    網羅ではなく「素朴な移植が落ちる境界」を 1 件ずつ置く設計（`irodori/tokenizer_ref.py` の
    `ENCODE_CASES` と同じ規律）。寸法を実運用の 224 / 384 まで上げないのは、1 ケースだけで
    f32 が 150,528 本になりフィクスチャが git に載らないため — resize は寸法に対して一様な
    コードで、境界の場合分けは**縮尺**（`scale >= 1` かどうか）と**端の切り詰め**にしか無い
    ので、小さい寸法でも同じ経路を通る。
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
    """`AutoImageProcessor` の実体を読む（利用者が素直に書いたときと同じ経路）。"""
    from transformers import AutoImageProcessor

    return AutoImageProcessor.from_pretrained(model_dir)


def check_processor_shape(processor: Any) -> dict[str, Any]:
    """image processor が期待どおりの実体・定数であることを実測する（外れたら止める）。"""
    from transformers.image_processing_backends import TorchvisionBackend

    if not isinstance(processor, TorchvisionBackend):
        raise SystemExit(
            f"image processor が TorchvisionBackend でない: {type(processor).__mro__}"
            "（transformers 5.x の既定は fast 側 — PIL 側に落ちていると resize の実装が変わる）"
        )
    if int(processor.resample) != EXPECTED_RESAMPLE:
        raise SystemExit(
            f"resample が {int(processor.resample)}（期待 {EXPECTED_RESAMPLE} = PIL BILINEAR）"
        )
    if tuple(processor.image_mean) != EXPECTED_MEAN or tuple(processor.image_std) != EXPECTED_STD:
        raise SystemExit(f"mean/std が {processor.image_mean} / {processor.image_std}")
    if processor.rescale_factor != EXPECTED_RESCALE:
        raise SystemExit(f"rescale_factor が {processor.rescale_factor}")
    if 1.0 / processor.rescale_factor != 255.0:
        raise SystemExit(f"1/rescale_factor が {1.0 / processor.rescale_factor}（255.0 でない）")
    for flag in ("do_resize", "do_rescale", "do_normalize"):
        if not getattr(processor, flag):
            raise SystemExit(f"{flag} が False（前処理の 3 段が揃っていない）")
    for flag in ("do_center_crop", "do_pad"):
        if getattr(processor, flag, False):
            raise SystemExit(f"{flag} が True（この層は crop も pad も持たない）")
    return {
        "class": type(processor).__name__,
        "backend": "TorchvisionBackend",
        "resample": int(processor.resample),
    }


def check_sibling_configs(models_root: Path) -> dict[str, Any]:
    """`inputs/siglip2/` 配下の全チェックポイントで `size` 以外が同一であることを実測する。

    この前処理層は base（224）と so400m（384）の両方に効く、という主張の中身。寸法だけが
    違うことを毎回実測しておかないと、片方の定数が変わったときに黙って片側だけ正しくなる。
    """
    found: dict[str, dict[str, Any]] = {}
    for path in sorted(models_root.glob("*/preprocessor_config.json")):
        found[path.parent.name] = json.loads(path.read_text(encoding="utf-8"))
    if not found:
        raise SystemExit(f"preprocessor_config.json が 1 つも無い: {models_root}")
    names = sorted(found)
    base = found[names[0]]
    for name in names[1:]:
        for key in SHARED_CONFIG_KEYS:
            if found[name].get(key) != base.get(key):
                raise SystemExit(
                    f"{name} の '{key}' が {names[0]} と違う: "
                    f"{found[name].get(key)!r} != {base.get(key)!r}"
                )
    return {name: found[name]["size"] for name in names}


def reference(
    processor: Any, image: np.ndarray, out_height: int, out_width: int
) -> tuple[np.ndarray, np.ndarray]:
    """1 ケースの `(resize 後の RGB8, pixel_values f32 [3,H,W])` を正本から採る。

    MUST: `resized` は**実経路そのもの**でなければならない — 別に呼んだ resize が
    フルパイプラインの中間と違えば、TS 側は「どこにも無い中間」に合わせてしまう。
    融合正規化を通してビット同一であることをここで実測する。
    """
    from transformers.image_utils import SizeDict

    size = SizeDict(height=out_height, width=out_width)
    tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1)[None]
    resized = processor.resize(image=tensor, size=size, resample=processor.resample)
    if resized.dtype != torch.uint8:
        raise SystemExit(f"resize の出力が {resized.dtype}（uint8 でない）")

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
            "resize 単体の結果を融合正規化に通した値が pixel_values と一致しない"
            f"（最大差 {(fused - pixel_values).abs().max().item():.6g}）"
        )
    return resized[0].permute(1, 2, 0).numpy(), pixel_values[0].numpy()


def build_fixture(processor: Any, models_root: Path) -> dict[str, Any]:
    """フィクスチャ本体を組む（全ケースを正本に通してから返す）。"""
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
    """フィクスチャを書き、要約を返す（検証に落ちたら 1 バイトも書かない）。"""
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--models-root", type=Path, default=MODELS_ROOT)
    parser.add_argument("--fixture-out", type=Path, default=DEFAULT_FIXTURE_PATH)
    args = parser.parse_args(argv)
    summary = emit(args.model_dir, args.models_root, args.fixture_out)
    print(json.dumps({"model_dir": str(args.model_dir), **summary}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
