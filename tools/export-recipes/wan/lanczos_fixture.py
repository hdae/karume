r"""Pillow の LANCZOS の**ホストテスト用 fixture**（ADR 0121 決定 11・段 9a — git 管理）。

TS 側の LANCZOS（Wan2.2 の I2V の前処理の内部関数 — Pillow `Resample.c` の 22 bit 固定小数点の
逐語の写し）を、Pillow の `Image.resize(size, Image.LANCZOS)` の出力と uint8 で突き合わせるための
入出力の組。モデルの重みにもテスト画像（`inputs/` — git 追跡外）にも触らない合成画像だけで作る。

    cd tools/export-recipes
    uv run --group wan --inexact python -m wan.lanczos_fixture

出力は 2 本（**git 管理**）:

    packages/models/tests/fixtures/wan-i2v/lanczos.json         — 説明・来歴・ケース表
    packages/models/tests/fixtures/wan-i2v/lanczos.safetensors  — 画素（U8）

生成後は `deno fmt packages/models/tests/fixtures/wan-i2v/lanczos.json` を掛ける（commit 形は
フォーマッタが正 — `deno task verify` の `fmt --check` が fixtures も見る）。画素を JSON の配列に
しないのは、フォーマッタが 1 値 1 行に開いて 1 MB を超えるため（wan-scheduler の `unipc.json` +
`unipc.safetensors` と同じ分け方）。

safetensors のテンソルはケースごとに 2 本: `<name>.input`（`[height, width, 3]`）と
`<name>.resized`（`[outHeight, outWidth, 3]`）。どちらも RGB8 の行優先（画素あたり 3 バイト）。

## ケースの設計

網羅ではなく「素朴な移植が落ちる境界」を 1 件ずつ置く（`siglip2/preprocess.py` の `build_cases`
と同じ規律）+ 実運用の 1 次元の係数表 + 乱択の掃引:

- 2 次元の幾何: 拡大（整数倍・非整数倍）・縮小（整数倍・非整数倍・台が大きく伸びる）・片軸だけ
  （もう片軸は寸法が変わらずパスを飛ばす — Pillow の `need_horizontal` / `need_vertical`）・
  横は縮小で縦は拡大・恒等（両パスを飛ばす）・端の切り詰め（3×2・1×1）・飽和（市松の拡大は
  負のローブで 0 と 255 を越える）。
- 実運用の係数表（1 行・1 列の高周波ノイズの画像 — 係数表の全列を出力で掃く）:
  832 → 1280（1280×704 の crop / stretch の横）・832 → 2219（704×1280 の crop の横）・
  832 → 277（256×160 の crop の横）・480 → 738 / 1280 / 160（各 crop の縦）・480 → 704
  （stretch の縦）。見るのは出力の uint8 だけなので、整数係数の ±1 はまず検出できない（ずれた
  係数が出力を変えるのは積和が 2^22 の境界をまたぐときだけ — 1 標本あたりおよそ 画素値 / 2^22）。
  出力の一致は係数表の一致の証拠にならない。
- 乱択の掃引（seed 固定）: 小さい寸法の組を {@link SWEEP_CASES} 本。

MUST: Pillow は関数の中で import する（`wan` グループは既定の sync に入らない —
`tests/test_optional_group_imports.py`）。値は Pillow の版で決まるので、版を来歴に記録する
（`wan/tests/test_lanczos_fixture.py` が今の Pillow で作り直して突き合わせ、版の更新で黙って
ずれた fixture を赤にする）。
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

#: fixture の置き場（git 管理）。
FIXTURE_DIR = REPO_ROOT / "packages" / "models" / "tests" / "fixtures" / "wan-i2v"
FIXTURE_JSON = "lanczos.json"
FIXTURE_TENSORS = "lanczos.safetensors"

#: 合成画像の乱数の seed（ノイズと掃引の寸法）。
SEED = 20261009

#: 乱択の掃引の本数と寸法の上限（入力の辺 ≤ 32・出力の辺 ≤ 48）。
SWEEP_CASES = 24
SWEEP_MAX_INPUT = 32
SWEEP_MAX_OUTPUT = 48

#: fixture の合計の上限（git 管理の小さな fixture — バイト）。
MAX_TOTAL_BYTES = 1 << 20


@dataclass(frozen=True)
class LanczosCase:
    """fixture の 1 ケース（入力画像と出力寸法）。"""

    name: str
    why: str
    image: np.ndarray
    out_height: int
    out_width: int


def _noise(rng: np.random.Generator, height: int, width: int) -> np.ndarray:
    return rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)


def _gradient(height: int, width: int) -> np.ndarray:
    """滑らかな模様（チャネルごとに位相をずらした正弦の積）。"""
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float64)
    channels = [
        127.5 + 127.5 * np.sin(xx / 5.0 + phase) * np.cos(yy / 3.0 - phase)
        for phase in (0.0, 1.0, 2.0)
    ]
    return np.stack(channels, axis=-1).round().clip(0, 255).astype(np.uint8)


def _checker(height: int, width: int) -> np.ndarray:
    """1 画素ごとの市松（0 / 255 — チャネルごとに反転を変える）。"""
    yy, xx = np.mgrid[0:height, 0:width]
    parity = (yy + xx) % 2
    channels = [parity, 1 - parity, parity]
    return (np.stack(channels, axis=-1) * 255).astype(np.uint8)


def build_cases() -> tuple[LanczosCase, ...]:
    """fixture のケース（並びと中身は seed で決まる — 作り直しても同じ）。"""
    rng = np.random.default_rng(SEED)
    cases = [
        LanczosCase("up-2x", "両軸ちょうど 2 倍の拡大", _noise(rng, 7, 11), 14, 22),
        LanczosCase("up-odd", "両軸とも非整数倍の拡大（13×9 → 50×31）", _gradient(9, 13), 31, 50),
        LanczosCase(
            "down-3x", "両軸ちょうど 1/3 の縮小（台が 3 倍に伸びる）", _noise(rng, 45, 60), 15, 20
        ),
        LanczosCase(
            "down-odd", "両軸とも非整数倍の縮小（73×41 → 29×17）", _noise(rng, 41, 73), 17, 29
        ),
        LanczosCase(
            "horizontal-only",
            "横だけ縮小・縦は同寸（縦のパスを飛ばす）",
            _noise(rng, 12, 40),
            12,
            23,
        ),
        LanczosCase(
            "vertical-only", "縦だけ拡大・横は同寸（横のパスを飛ばす）", _noise(rng, 40, 12), 57, 12
        ),
        LanczosCase(
            "mixed-axes",
            "横は縮小・縦は拡大（2 パスの軸の取り違えを掴む）",
            _gradient(9, 50),
            27,
            18,
        ),
        LanczosCase(
            "identity", "両軸とも同寸（両パスを飛ばす — 出力 = 入力）", _noise(rng, 13, 21), 13, 21
        ),
        LanczosCase("tiny", "3×2 → 7×5（両端でタップが切り詰められる）", _noise(rng, 2, 3), 5, 7),
        LanczosCase(
            "single-pixel",
            "1×1 → 4×3（全出力が 1 タップ）",
            np.array([[[7, 128, 249]]], dtype=np.uint8),
            3,
            4,
        ),
        LanczosCase(
            "checker-ringing",
            "市松 0 / 255 の拡大（負のローブで 0 と 255 を越え、飽和で切る経路）",
            _checker(16, 16),
            20,
            24,
        ),
        LanczosCase(
            "large-down",
            "横 120 → 7（台が 17 倍に伸びて 100 タップを超える）",
            _gradient(8, 120),
            8,
            7,
        ),
        LanczosCase(
            "cover-up",
            "832×480 → 1280×738 と同じ比の拡大を小さく（52×30 → 80×46）",
            _noise(rng, 30, 52),
            46,
            80,
        ),
        LanczosCase(
            "cover-down",
            "832×480 → 277×160 と同じ比の縮小を小さく（99×57 → 33×19）",
            _noise(rng, 57, 99),
            19,
            33,
        ),
    ]
    rows = (
        (1280, "1280×704 の crop と stretch の横"),
        (2219, "704×1280 の crop の横（2.67 倍の拡大）"),
        (277, "256×160 の crop の横（1/3 の縮小）"),
    )
    for out_width, why in rows:
        cases.append(
            LanczosCase(
                f"row-832-{out_width}", f"実運用の係数表: {why}", _noise(rng, 1, 832), 1, out_width
            )
        )
    columns = (
        (738, "1280×704 の crop の縦"),
        (1280, "704×1280 の crop の縦"),
        (160, "256×160 の crop の縦（ちょうど 1/3）"),
        (704, "1280×704 の stretch の縦"),
    )
    for out_height, why in columns:
        cases.append(
            LanczosCase(
                f"column-480-{out_height}",
                f"実運用の係数表: {why}",
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
                "乱択の幾何（seed 固定）",
                _noise(rng, height, width),
                out_height,
                out_width,
            )
        )
    names = [case.name for case in cases]
    if len(set(names)) != len(names):
        raise AssertionError(f"ケース名が重複している: {names}")
    return tuple(cases)


def pillow_lanczos(image: np.ndarray, out_height: int, out_width: int) -> np.ndarray:
    """Pillow の `Image.resize((out_width, out_height), Image.LANCZOS)`（RGB8 `[H, W, 3]`）。"""
    from PIL import Image

    resized = Image.fromarray(image).resize((out_width, out_height), Image.LANCZOS)
    return np.asarray(resized, dtype=np.uint8)


def build_fixture() -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """fixture の JSON（説明・来歴・ケース表）と safetensors のテンソル。"""
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
            "Pillow の LANCZOS（Image.resize(size, Image.LANCZOS)）の入出力の組"
            "（生成: tools/export-recipes/wan/lanczos_fixture.py）。",
            "Wan2.2 の I2V の前処理（公式の crop と diffusers の stretch）の resize の正本で、"
            "TS 側は"
            " Pillow の Resample.c の 22 bit 固定小数点を逐語で写し、uint8 で完全一致させる。",
            f"画素は {FIXTURE_TENSORS} の U8 テンソル `<name>.input` [height, width, 3] と"
            " `<name>.resized` [outHeight, outWidth, 3]（RGB8 の行優先）。",
            "row-* / column-* は実運用の 1 次元の係数表（832 → 1280 / 2219 / 277・"
            "480 → 738 / 1280 / 160"
            " / 704）を 1 行・1 列の高周波ノイズで掃く（見るのは出力の uint8 — 整数係数の ±1 は"
            "積和が 2^22 の境界をまたぐときしか出力に出ないので、"
            "係数表の一致の証拠にはならない）。",
            "値は Pillow の版で決まる — 版を変えたら作り直す（recipe の"
            " wan/tests/test_lanczos_fixture.py が今の Pillow と突き合わせる）。",
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
    """fixture を書き、要約を返す（合計が {@link MAX_TOTAL_BYTES} を超えたら何も書かない）。"""
    from safetensors.numpy import save

    document, tensors = build_fixture()
    payload = save(tensors, metadata={"pillow": document["source"]["pillow"]})
    text = json.dumps(document, ensure_ascii=False, indent=2) + "\n"
    total = len(payload) + len(text.encode("utf-8"))
    if total > MAX_TOTAL_BYTES:
        raise AssertionError(f"fixture の合計 {total} バイトが上限 {MAX_TOTAL_BYTES} を超える")
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
