r"""BiRefNet の α マットの後処理の**上流の手順の golden**（裁定 4・ADR 0125 — git 管理の fixture）。

TS 側の後処理（`packages/models/src/birefnet/matte.ts` の 2 つの手順）を、系列ごとの上流の関数
そのものが出す最終 α と突き合わせるための入出力の組。上流の関数をそのまま呼び、**量子化だけを
四捨五入へ差し替える**（ADR 0125 — 上流の切り捨ては `astype(uint8)` の慣用であって意図ではない）:

- **HR**（`u8-pil-bicubic` — 同梱 `handler.py:131-137` の段の順序）: numpy の f64 で
  `σ = 1 / (1 + exp(−x))` → `floor(255σ + 0.5)` で uint8 → Pillow の `Image.fromarray(q)`（L）
  `.resize(size)`（resample 省略 = BICUBIC — `PIL/Image.py:2368`・同寸なら写し）。上流の半精度は
  写さない（速度のための選択で、karume の logit はもともと f32 のグラフの出力）。
- **Lucida**（`float-bilinear` — モデルカード `README.md:86-89`）: torch の f32 の `sigmoid` →
  `transforms.functional.resize(preds[0], (H, W))`（テンソル枝・bilinear・antialias の既定のまま）→
  `alpha.numpy() * 255`（f32）。丸める前の `255α` を残し、最終 α はその `floor(x + 0.5)`。

    cd tools/export-recipes
    uv run --group birefnet python -m birefnet.matte_fixture \
        --logits-dir <リポ>/outputs/verify/<環境キー>/<日付>_birefnet

`--logits-dir` は実画像の logit の切り出し（入力）を採り直すときだけ渡す。省くと git にある
fixture の切り出しをそのまま入力にして、期待値だけを作り直す（`outputs/` が無い機でも回る —
`birefnet/tests/test_matte_fixture.py` もこの経路で今のライブラリと突き合わせる）。

出力は 2 本（**git 管理**）:

    packages/models/tests/fixtures/birefnet-matte/matte.json         — 説明・来歴・ε・入力・ケース
    packages/models/tests/fixtures/birefnet-matte/matte.safetensors  — 入力の logit と期待値

生成後は `deno fmt packages/models/tests/fixtures/birefnet-matte/matte.json` を掛ける（commit 形は
フォーマッタが正）。safetensors のテンソルは入力ごとに `<input>.logits`（F32 `[h, w]`）、ケース
ごとに `<case>.hr`（U8 `[H, W]` — HR の最終 α）と `<case>.lucida`（F32 `[H, W]` — Lucida の丸める
前の `255α`）。

## 入力（2 種類）

- **実画像の logit の切り出し**: `e2e_birefnet_real_test.ts` が書いた既定席のマット logits（HR は
  f16 席・Lucida は f32 席 — {@link REAL_SERIES}。`<系列>-photo-{portrait,street}-matte.safetensors`
  — 1024² の karume のグラフ出力）から、輪郭の画素（`|logit| < SOFT_LOGIT`）を最も多く含む
  `CROP_WIDTH × CROP_HEIGHT` の窓を両系列の 2 枚ずつ切り出す。portrait からは窓の中心を通る 1 行と
  幅 2 の列（長さ 1024）も採り、実運用の係数表（1024 → 832 / 480 / 2048）を掃く。来歴（ファイル・
  sha256・窓）を入力表に残す。
- **合成の境界ケース**: 素朴な移植が落ちる境界を 1 件ずつ（段差・細線・滑らかな斜面・飽和の帯・
  市松の overshoot・端の切り詰め・台が大きく伸びる縮小）。logit は有限の値を明示する。

目標寸法は 1024² 以外（拡大・縮小・非正方・片軸だけ・恒等）。

## 丸めの境界の扱い

- HR: JS の `Math.exp` と numpy の `exp` は最終 ulp が割れうる。`|255σ − (k + 0.5)| < TIE_MARGIN`
  の標本は丸めの向きが実装で割れうるので、**生成器が拒む**（入力を選び直す）。それ以外は TS の写しと
  0 不一致が契約。
- Lucida: torch の f32 の積和と TS の f64 の resize は丸めが違う。TS の経路の `255α`（f64 の写し —
  {@link _ts_float_bilinear_scaled}）と torch の `255α` の差の最大値を実測し、`LUCIDA_EPSILON` を
  超えたら生成器が落ちる。TS の門は「一致する、または golden の `255α` が `k + 0.5` から ε 以内」の
  画素だけを許す。

Pillow は関数の中で import する。理由は recipe の作法（依存グループが宣言するパッケージは要る
関数の中で import する — `tests/test_optional_group_imports.py` が機械の門にしている）に揃える
ためで、Pillow が無いからではない — Pillow 自体は torchvision（exporter の core の依存）経由で
既定の sync にも入る（ADR 0125 決定 6）。NOTE: 門は配布名から import 名を導く（`pillow`）ので、
`PIL` を module 直下で import しても門では掴めない。関数内の import は門に頼らず作法として守る。

値は torch / torchvision / Pillow / numpy の版で決まるので、版を来歴に記録する
（`birefnet/tests/test_matte_fixture.py` が今の版で作り直して突き合わせ、版の更新で黙ってずれた
fixture を赤にする）。
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

#: fixture の置き場（git 管理）。
FIXTURE_DIR = REPO_ROOT / "packages" / "models" / "tests" / "fixtures" / "birefnet-matte"
FIXTURE_JSON = "matte.json"
FIXTURE_TENSORS = "matte.safetensors"

#: fixture の合計の上限（git 管理の小さな fixture — バイト）。
MAX_TOTAL_BYTES = 1 << 20

#: HR の丸めの境界の余白（`|255σ − (k + 0.5)|` がこれ未満の標本を拒む）。exp の最終 ulp の差は
#: 255σ で 1e-13 程度なので、桁に十分な余裕を持たせる。
TIE_MARGIN = 1e-9

#: Lucida の許す差（torch の f32 の `255α` と TS の f64 の経路の `255α` の差の上限）。実測
#: （{@link measure_lucida_delta} — 2026-10-11 は 4.2e-3・最大は 1024 → 480 の縮小の列）の
#: 2 倍以上になる最小の 10 の冪（ADR 0125）。差の主因は torch が座標と重みを f32 で計算すること
#: （添字が大きいほど効く）。
LUCIDA_EPSILON = 1e-2

#: 実画像の切り出しの窓（幅 × 高さ — 非正方にして軸の取り違えを掴む）。
CROP_WIDTH = 40
CROP_HEIGHT = 32

#: 輪郭の画素とみなす logit の絶対値の上限（σ が 0.2% 〜 99.8% の帯）。
SOFT_LOGIT = 6.0

#: 飽和した前景とみなす logit の下限（窓の選び方 — {@link select_crop}）。目的は、飽和した前景を
#: 確実に含む窓を選ぶこと。四捨五入の量子化で α が 255 になる境目 6.23（ln 509 — ADR 0125）より
#: 十分に上に置き、窓の前景が境目の近くの値だけで埋まらないようにする。8.32 は境目より約 2 上
#: （1 − σ が境目の約 1/8）。変えると窓の選び方が変わりうるので、変えるなら fixture を作り直す。
SATURATED_LOGIT = 8.32

#: 窓が前景・背景のそれぞれについて最低限含む画素の数（窓の 1 行ぶん）。
MIN_SIDE_PIXELS = CROP_WIDTH

#: 窓の探索の歩幅（画素）。
CROP_STRIDE = 4

#: 実画像の切り出しの元 = (e2e が書く系列の綴り — `e2e_birefnet_real_test.ts` の `artifactName`,
#: ケース名の頭)。各 checkpoint の配布の既定席の系列（HR は f16 席だけ — ADR 0113 追記
#: 2026-10-11。HR の f16 席の logit は退役した f32 席と sha256 まで同じ）。席は系列名から導く
#: （{@link real_series_seat}）。
#:
#: ケース名の頭を系列名から導かないのは、ケース名が fixture の来歴の外の名前だから — 元の系列
#: （席）を替えても `real-birefnet-hr-*` のまま動かさず、作り直しの差分を来歴（ファイル名）に
#: 留める。
REAL_SERIES: tuple[tuple[str, str], ...] = (
    ("birefnet-hr-1024-f16", "birefnet-hr"),
    ("lucida-1024", "lucida"),
)
REAL_IMAGES = ("photo-portrait", "photo-street")

#: 行と列の切り出し（実運用の係数表を掃く 1 行・1 列）を採る画像。
LINE_IMAGE = "photo-portrait"

#: 列の切り出しの幅（幅 1 は torch の不具合を踏む — 下の `REAL_TARGETS`）。
COLUMN_WIDTH = 2

#: 目標寸法の 1 行（名前・幅・高さ・意図）。
Target = tuple[str, int, int, str]

#: 実画像の切り出しの種類 → 目標寸法。
#:
#: - `crop`（窓 `CROP_WIDTH × CROP_HEIGHT`）: 拡大・縮小・非正方・片軸だけ・恒等。
#: - `row` / `column`（窓の中心を通る 1 行と幅 2 の列 — 長さ 1024）: 実運用の係数表（1024 → 832 /
#:   480 の縮小と 1024 → 2048 の拡大）。torch の f32 の座標の丸めは添字が大きいほど効くので、ε は
#:   実運用の長さで測る（窓の大きさだけで測ると小さく出る）。列を幅 2 にするのは、torch 2.13.0 の
#:   CPU の antialias bilinear が**幅 1 の入力**で縦の補間を壊すため（全行が先頭の値になる — 実測。
#:   上流の不具合で写す対象ではない — ADR 0125。{@link lucida_scaled} が拒む）。
REAL_TARGETS: dict[str, tuple[Target, ...]] = {
    "crop": (
        ("up", 97, 83, "両軸とも非整数倍の拡大"),
        ("down", 17, 13, "両軸とも非整数倍の縮小（台が伸びる）"),
        ("mixed", 25, 61, "横は縮小・縦は拡大（2 パスの軸の取り違えを掴む）"),
        ("horizontal-only", 87, CROP_HEIGHT, "横だけ拡大・縦は同寸（縦のパスを飛ばす）"),
        ("vertical-only", CROP_WIDTH, 15, "縦だけ縮小・横は同寸（横のパスを飛ばす）"),
        ("identity", CROP_WIDTH, CROP_HEIGHT, "両軸とも同寸（resize は写し — 量子化だけが残る）"),
    ),
    "row": (
        ("down-832", 832, 1, "実運用の係数表: 1024 → 832（832×480 の横）"),
        ("up-2048", 2048, 1, "実運用の係数表: 1024 → 2048（2048² の横）"),
    ),
    "column": (
        ("down-480", COLUMN_WIDTH, 480, "実運用の係数表: 1024 → 480（832×480 の縦）"),
        ("up-2048", COLUMN_WIDTH, 2048, "実運用の係数表: 1024 → 2048（2048² の縦）"),
    ),
}


@dataclass(frozen=True)
class MatteInput:
    """fixture の入力 1 本（logit の地図 `[h, w]`・当てる目標寸法・来歴）。"""

    name: str
    why: str
    logits: np.ndarray
    targets: tuple[Target, ...]
    provenance: dict[str, Any] | None = None


@dataclass(frozen=True)
class MatteCase:
    """fixture の 1 ケース（入力 × 目標寸法）。"""

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
    """中心の円盤（logit = `slope · (radius − 中心からの距離)` — 輪郭が柔らかい前景）。"""
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float64)
    distance = np.hypot(xx - (width - 1) / 2, yy - (height - 1) / 2)
    return (slope * (radius - distance)).astype(np.float32)


def _saturation_band(height: int, width: int) -> np.ndarray:
    """logit 5.5〜18 を行優先に敷き詰めた地図（四捨五入で 255 になる境目 6.23 と、上流の
    半精度・f32 の切り捨ての境目 8.32 / 16.64 を跨ぐ）。"""
    band = np.linspace(5.5, 18.0, height * width, dtype=np.float64)
    return band.reshape(height, width).astype(np.float32)


def _wide_ramp(width: int) -> np.ndarray:
    """横 `width` の 2 行（段差のある斜面 — 台が大きく伸びる縮小の総和を見る）。"""
    xs = np.arange(width, dtype=np.float64)
    row = np.where(xs < width / 2, -9.0 + xs * 0.05, 7.5 - (xs - width / 2) * 0.03)
    return np.stack([row, row[::-1]]).astype(np.float32)


def synthetic_inputs() -> tuple[MatteInput, ...]:
    """合成の境界ケース（logit は有限の値を明示する）。"""
    return (
        MatteInput(
            "step",
            "段差 0 / 255 の 4 → 8（R6B-001 の実測の 1 例）",
            _row([-12.0, -12.0, 12.0, 12.0]),
            (("up", 8, 1, "4 → 8 の拡大（bicubic の負のローブが 0 と 255 を越えて飽和で切る）"),),
        ),
        MatteInput(
            "ramp",
            "滑らかな斜面の 8 → 21（R6B-001 の実測の 1 例）",
            _row(np.linspace(-5.75, 6.25, 8).tolist()),
            (("up", 21, 1, "8 → 21 の拡大"),),
        ),
        MatteInput(
            "thin",
            "細い前景の 6 → 15（R6B-001 の実測の 1 例 — 差が最も大きい）",
            _row([-12.0, -12.0, 12.0, -12.0, -12.0, -12.0]),
            (("up", 15, 1, "6 → 15 の拡大"),),
        ),
        MatteInput(
            "saturation",
            "飽和の帯（logit 5.5〜18 — 255 になる境目を跨ぐ）",
            _saturation_band(4, 12),
            (
                ("up", 29, 9, "拡大"),
                ("down", 5, 3, "縮小"),
                ("identity", 12, 4, "恒等（量子化の境目だけ）"),
            ),
        ),
        MatteInput(
            "checker",
            "市松 ±12（bicubic の overshoot を 0 / 255 で切る経路）",
            _checker(8, 8, 12.0),
            (("up", 19, 20, "拡大"),),
        ),
        MatteInput(
            "disc",
            "柔らかい輪郭の円盤",
            _disc(29, 33, 8.7, 1.5),
            (("down", 7, 6, "縮小（台が 4〜5 倍に伸びる）"), ("up", 50, 44, "拡大")),
        ),
        MatteInput(
            "wide",
            "横 120 の段差のある斜面",
            _wide_ramp(120),
            (("down", 7, 2, "横 120 → 7（台が 17 倍に伸びる・縦は同寸）"),),
        ),
        MatteInput(
            "tiny",
            "3×2（両端でタップが切り詰められる）",
            np.asarray([[-3.25, 1.5, 7.25], [9.5, -0.75, 2.25]], dtype=np.float32),
            (("up", 7, 5, "3×2 → 7×5"),),
        ),
        MatteInput(
            "single",
            "1×1（全出力が 1 タップ）",
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
    """輪郭の画素を最も多く含む窓の左上 `(x, y)`（同点は行優先で先）。

    MUST: 飽和した前景（logit が {@link SATURATED_LOGIT} を超える画素）と背景がどちらも
    {@link MIN_SIDE_PIXELS} 以上ある窓だけを候補にする — 輪郭の画素の数だけで選ぶと、柔らかい
    だけで飽和を持たない窓（髪の先の霞など）に寄り、飽和の帯の丸めを fixture が掃かない。
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
        raise ValueError("飽和した前景と背景の両方を含む窓が無い（輪郭の無いマットを渡した）")
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
    """系列名 → (checkpoint, モデル名 = 解像度, 席の格納 dtype)。

    系列名の綴りの正本（`birefnet.distribution.birefnet_series_name`）を全ての組で引いて逆に
    辿る — 席を {@link REAL_SERIES} に別に持つと、系列を替えた日に来歴の席だけが古いまま残る。
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
            f"系列名 '{series}' が BiRefNet の系列の綴りに 1 つで当たらない: {matches}"
        )
    return matches[0]


def extract_real_inputs(logits_dir: Path) -> tuple[MatteInput, ...]:
    """e2e が書いたマット logits から実画像の切り出しを作る（来歴つき）。"""
    from safetensors.numpy import load_file

    inputs = []
    for series, case in REAL_SERIES:
        _checkpoint, _model, seat = real_series_seat(series)
        for image in REAL_IMAGES:
            path = logits_dir / f"{series}-{image}-matte.safetensors"
            tensors = load_file(path)
            if len(tensors) != 1:
                raise ValueError(f"{path}: テンソルが {len(tensors)} 本（グラフ出力 1 本のはず）")
            ((tensor_name, tensor),) = tensors.items()
            if tensor.dtype != np.float32 or tensor.ndim != 4 or tensor.shape[:2] != (1, 1):
                raise ValueError(
                    f"{path}: {tensor_name} が F32 [1, 1, H, W] でない（{tensor.shape}）"
                )
            logits = tensor[0, 0]
            height, width = logits.shape
            source = {
                "file": f"outputs/verify/{logits_dir.parent.name}/{logits_dir.name}/{path.name}",
                "sha256": _sha256(path),
                "tensor": tensor_name,
            }
            stem = f"real-{case}-{image.removeprefix('photo-')}"
            why = f"実画像の logit（{series} の {seat} 席・{image}）"
            x, y = select_crop(logits)
            inputs.append(
                _real_input(
                    stem,
                    f"{why}の輪郭付近の窓",
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
                    f"{why}の窓の中心を通る 1 行",
                    logits,
                    "row",
                    source,
                    {"x": 0, "y": center_y, "width": width, "height": 1},
                )
            )
            inputs.append(
                _real_input(
                    f"{stem}-column",
                    f"{why}の窓の中心を通る幅 {COLUMN_WIDTH} の列",
                    logits,
                    "column",
                    source,
                    {"x": center_x, "y": 0, "width": COLUMN_WIDTH, "height": height},
                )
            )
    return tuple(inputs)


def committed_real_inputs(fixture_dir: Path = FIXTURE_DIR) -> tuple[MatteInput, ...]:
    """git にある fixture の実画像の切り出し（入力表の来歴と `<input>.logits`）。"""
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
    """入力表とケース表（実画像 → 合成の順 — 作り直しても同じ並び）。"""
    inputs = (*real_inputs, *synthetic_inputs())
    cases = tuple(
        MatteCase(f"{entry.name}.{label}", why, entry.name, width, height)
        for entry in inputs
        for label, width, height, why in entry.targets
    )
    names = [entry.name for entry in inputs] + [case.name for case in cases]
    if len(set(names)) != len(names):
        raise AssertionError(f"名前が重複している: {names}")
    return inputs, cases


# ---- 上流の手順（量子化だけを四捨五入へ差し替える）-----------------------------------------


def round_half_up(scaled: np.ndarray) -> np.ndarray:
    """`floor(x + 0.5)` を uint8 へ（0.5 は切り上げ — ADR 0125 の量子化）。"""
    return np.floor(scaled.astype(np.float64) + 0.5).astype(np.uint8)


def truncate(scaled: np.ndarray) -> np.ndarray:
    """上流の量子化そのもの（`(x * 255).astype(uint8)` の切り捨て — 比較のためだけに持つ）。"""
    return scaled.astype(np.uint8)


def sigmoid_f64(logits: np.ndarray) -> np.ndarray:
    """f64 の `1 / (1 + exp(−x))`（TS の `1 / (1 + Math.exp(-raw))` と同じ式）。"""
    return 1.0 / (1.0 + np.exp(-logits.astype(np.float64)))


def assert_no_ties(scaled: np.ndarray, where: str) -> None:
    """丸めの向きが exp の実装で割れうる標本（`k + 0.5` から {@link TIE_MARGIN} 未満）を拒む。"""
    distance = np.abs(scaled - (np.floor(scaled) + 0.5))
    near = np.argwhere(distance < TIE_MARGIN)
    if near.size:
        raise ValueError(
            f"{where}: 255σ が k + 0.5 から {TIE_MARGIN} 未満の標本がある（{near[:4].tolist()}）"
            " — 丸めの向きが exp の実装で割れうるので入力を選び直す"
        )


def hr_alpha(
    logits: np.ndarray,
    out_width: int,
    out_height: int,
    quantize: Callable[[np.ndarray], np.ndarray] = round_half_up,
    *,
    allow_ties: bool = False,
) -> np.ndarray:
    """HR の上流の手順（f64 の σ → 8bit → Pillow の `resize(size)` の既定 = BICUBIC）。

    `allow_ties` は同点の標本（{@link assert_no_ties}）を拒まずに四捨五入する — 同点の標本が届く
    出力画素を呼び手が自分で外す場合だけに使う（`birefnet.matte_ab_check`）。
    """
    from PIL import Image

    scaled = sigmoid_f64(logits) * 255
    if quantize is round_half_up and not allow_ties:
        assert_no_ties(scaled, "HR")
    image = Image.fromarray(quantize(scaled))
    if image.mode != "L":
        raise AssertionError(f"8bit の 1 バンドが L で開かれない（{image.mode}）")
    return np.asarray(image.resize((out_width, out_height)), dtype=np.uint8)


def lucida_scaled(logits: np.ndarray, out_width: int, out_height: int) -> np.ndarray:
    """Lucida の上流の手順の丸める前の `255α`（torch の f32 の σ → TF.resize → `× 255`）。"""
    import torch
    from torchvision.transforms import functional as transforms_functional

    height, width = logits.shape
    if width == 1 and height > 1:
        # torch 2.13.0 の CPU の antialias bilinear は幅 1・高さ 2 以上の入力の縦を壊す（全行が
        # 先頭の行の値 — 実測）。上流の不具合を golden に焼かない（ADR 0125 — バグは写さない）。
        raise ValueError(f"幅 1・高さ {height} の入力は torch の antialias の不具合を踏む")
    preds = torch.from_numpy(np.ascontiguousarray(logits, dtype=np.float32))[None, None].sigmoid()
    alpha = transforms_functional.resize(preds[0], [out_height, out_width]).squeeze(0)
    scaled = alpha.numpy() * 255
    if scaled.dtype != np.float32:
        raise AssertionError(f"255α が f32 でない（{scaled.dtype}）")
    return scaled


# ---- TS の float-bilinear の経路の写し（ε の実測だけに使う）------------------------------


def _bilinear_taps(in_size: int, out_size: int) -> list[tuple[int, np.ndarray]]:
    """`preprocess.ts` の `buildTaps`（三角フィルタ）の写し。"""
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
    """`matte.ts` の `float-bilinear` の丸める前の `255α`。

    σ を f64 で計算して f32 に格納 → f64 の resize → f32 に格納 → f64 で × 255（`resizePlaneF32`
    は f32 の配列を返す）。
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
    """torch の `255α` と TS の経路の `255α` の差の最大値（全ケース）。"""
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
    """期待値を決めるライブラリの版（fixture の来歴 — テストが今の環境と突き合わせる）。"""
    return {name: version(name) for name in ("torch", "torchvision", "pillow", "numpy")}


def build_fixture(
    real_inputs: Sequence[MatteInput],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """fixture の JSON（説明・来歴・ε・入力表・ケース表）と safetensors のテンソル。"""
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
            f"torch と TS の経路の 255α の差 {measured} が ε {LUCIDA_EPSILON} の半分を超えた"
            "（ε を決め直す — ADR 0125）"
        )
    document = {
        "_doc": [
            "BiRefNet の α マットの後処理の上流の手順の golden"
            "（生成: tools/export-recipes/birefnet/matte_fixture.py — ADR 0125）。",
            "上流の関数をそのまま呼び、量子化だけを四捨五入（floor(x + 0.5)）へ差し替えたもの。",
            "hr: numpy の f64 の σ → floor(255σ + 0.5) の uint8 → PIL.Image.fromarray(q)（L）"
            ".resize((W, H))（resample 省略 = BICUBIC）。TS の u8-pil-bicubic と 0 不一致が契約"
            "（生成器は 255σ が k + 0.5 から tieMargin 未満の標本を拒む）。",
            "lucida: torch の f32 の sigmoid → torchvision.transforms.functional.resize(preds[0], "
            "(H, W))（bilinear・antialias 既定）→ × 255（f32）。最終 α は floor(x + 0.5)。"
            "TS の float-bilinear は一致するか、golden の 255α が k + 0.5 から epsilon 以内の"
            "ときだけ 1 段ずれてよい。",
            f"テンソルは {FIXTURE_TENSORS} の `<input>.logits`（F32 [h, w]）・`<case>.hr`"
            "（U8 [H, W]）・`<case>.lucida`（F32 [H, W] — 丸める前の 255α）。",
            "値は torch / torchvision / Pillow / numpy の版で決まる — 版を変えたら作り直す"
            "（recipe の birefnet/tests/test_matte_fixture.py が今の版と突き合わせる）。",
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
    """fixture を書き、要約を返す（合計が {@link MAX_TOTAL_BYTES} を超えたら何も書かない）。"""
    from safetensors.numpy import save

    document, tensors = build_fixture(real_inputs)
    # NOTE: 版は JSON の `source.versions` にだけ書く — safetensors の metadata は鍵の並びが
    # 書くたびに変わり（2 鍵以上で実測）、作り直しのバイト一致が崩れる。
    payload = save(tensors)
    text = json.dumps(document, ensure_ascii=False, indent=2) + "\n"
    total = len(payload) + len(text.encode("utf-8"))
    if total > MAX_TOTAL_BYTES:
        raise AssertionError(f"fixture の合計 {total} バイトが上限 {MAX_TOTAL_BYTES} を超える")
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
        help="実画像の logit を採り直すときの e2e の出力"
        "（outputs/verify/<環境キー>/<日付>_birefnet）",
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
