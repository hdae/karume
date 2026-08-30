"""実重みの Depth Anything V2（単一画像の相対深度推定）を IR v1 コンテナ + golden io へ
書き出す台本。モデルは `--model-dir` で選ぶ（既定 Small — 下の「モデル軸」）。

生成物は `outputs/series/` 配下で、リポジトリ直下の `.gitignore` によりコミット対象外。

    uv run --group depth-anything python -m depth_anything.export
    uv run --group depth-anything python -m depth_anything.export --verify  # パッチ前との同値検証
    uv run --group depth-anything-preprocess python -m depth_anything.export --real-images

transformers は **5.14.1 でピン**する（`depth_anything.patch` がモデリングコードの
クラス属性を差し替えるため — pyproject.toml の `depth-anything` グループ）。

## モデル軸

系列名は `--model-dir` のディレクトリ名（小文字化）。受けているのは transformers 移植版
（`DepthAnythingForDepthEstimation` = DINOv2 backbone + DPT neck / head）の重み一式で、
既定は `inputs/depth-anything/Depth-Anything-V2-Small-hf/`（上流
`depth-anything/Depth-Anything-V2-Small-hf` — **Apache-2.0**）。

MUST: **Base / Large は上流のライセンスが CC BY-NC 4.0**（Small だけが Apache-2.0 —
2026-08-14 に HF の API で実測）。台本は `--model-dir` でどれでも受けるが、配布形にできるのは
Small だけ。

## 解像度（軸ではない — 1 点に固定）

入力は**事前学習解像度の正方 1 点**（Small / Base / Large とも `image_size=518`・patch 14 =
37×37 パッチ）。DINOv2 の位置埋め込みはパッチ数が合えばそのまま使われ、外れると
`aten.upsample_bicubic2d`（IR 語彙に無い）が生える。`depth_anything.patch` の ② が条件を
反転して fail loudly にしてあり、ここでは重みの `image_size` から解像度を引くだけ
（{@link load_wrapper}）。

## 何をグラフに載せるか

`DepthAnythingForDepthEstimation.forward` が返す `predicted_depth`（`[B, H, W]`）1 本。
相対深度なので単位も向きも持たない（大きいほど手前）。head 末尾の ReLU までがグラフで、
可視化のための正規化（`depth / depth.max()` 等）はホスト側の責務。

batch は **静的 1**。動的軸は無い（`symbol_names=()`）。

MUST: モデルは `depth_anything.patch` のパッチを当ててから export する。当てないと
DPT reassemble の `nn.ConvTranspose2d`（`aten.convolution.default` 2 本）で変換段が落ちる
（差し替えの一覧は `depth_anything.patch` の docstring）。golden の期待値は**差し替え後**の
モジュールから採る — 差し替え前後の差はパッチ層の責務で、`--verify` が 2 点評価で実測する。

MUST: `--verify` は emit しない（同一プロセスでは併用できない）。パッチはクラス属性の
プロセス全域差し替えなので、emit のために当てた後では「当てる前の参照」が採れない。

## 入力の約束（前処理はグラフに載せない）

グラフ入力は **正規化済みの** `pixel_values f32 [1, 3, S, S]`。画像の decode / resize /
rescale（1/255）/ normalize は karume 側の責務で、定数の正本は同梱
`preprocessor_config.json`（`DPTImageProcessor` の ImageNet 統計 = {@link IMAGENET_MEAN} /
{@link IMAGENET_STD}・`resample=3` = **bicubic**・`keep_aspect_ratio=true` は正方入力 →
518² では恒等）。**SigLIP2（mean = std = 0.5・resample 2 = bilinear）とは統計もフィルタも
別**なので、共有せずここに持つ。定数は {@link check_processor} が `--real-images` の
たびに**現物の processor から**実測する。

## golden の 2 群（合成画像 + 実画像）— どちらも残す

既定の 4 ケースは**合成画像**（{@link build_cases}）で、判別は {@link _sanity} が持つ:

- 4 ケースの出力が**互いに違う**（入力が効いている）
- 各ケースの出力が**非負**（head 末尾は ReLU）で**一様でない**（min < max）
- **対角ランプとの相関が `ramp` ケースで最大**（順序だけを見る・閾値を置かない）。単調な
  奥行き手掛かりを持つのは 4 枚のうち `ramp` だけなので、出力が入力の幾何を追えていない
  （一様・入力非依存・空間の取り違え）と崩れる

`--real-images` を付けると、それに加えて `outputs/demo/` の**実画像** 4 枚
（{@link REAL_CASES}）を通した golden も書く。こちらの主張は「TS 前処理 + karume 推論」対
「`DPTImageProcessor` + torch 推論」の突合で、Python 側は**正本そのもの**（写経ではなく
`AutoImageProcessor` の実体を呼ぶ — BiRefNet の handler.py 写経と違い、こちらは
transformers の公開クラスがそのまま正本だから）。

判別も実画像側が持つ: 各枚に**構図から言える近い領域 / 遠い領域**の対を置き（{@link
REAL_REGIONS}）、近側の深度平均が遠側を上回ることだけを見る（閾値は置かない —
{@link _real_sanity}）。合成画像の `ramp` が「単調な手掛かりを追えるか」、実画像が「実写
構図の遠近を当てられるか」で、片方が他方を代替しない。

MUST: 実画像 golden には**元画像の sha256** を `__metadata__` に載せる。画像は
`examples/anima/eval-images.ts` でいつでも焼き直せる生成物なので、焼き直したのに golden を
採り直していない環境では、突合ではなく**入力が違う**（そして値は一見それらしく出る）。

## 出力レイアウト

    outputs/series/<系列名>/model.safetensors     重み・定数 + __metadata__
    outputs/series/<系列名>/io.<case>.safetensors 入力と torch CPU 期待出力

io のテンソルキー規約は tiny golden / DeBERTa / SigLIP2 / BiRefNet と同じ
（`input.<グラフ入力名>` / `output.<位置>`）。
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

from _shared.paths import INPUTS_ROOT, OUTPUTS_ROOT, SERIES_ROOT
from karume.artifacts import staged_publication
from karume.convert import normalize_boundary_tensor
from karume.ir import IrGraph
from karume.pipeline import export_to_file
from karume.shards import resolve_shards

from . import patch

#: 実重みの親（`inputs/depth-anything/<名前>/` に HF の 3 ファイルを展開した先）。
MODELS_ROOT = INPUTS_ROOT / "depth-anything"

#: 実画像の置き場（`rm -rf` で安全に消せる席 — docs/assets-layout.md）。
DEMO_ROOT = OUTPUTS_ROOT / "demo"

#: 既定のモデル（`--model-dir` 未指定のとき — 上流で唯一の Apache-2.0）。
DEFAULT_MODEL_DIR = MODELS_ROOT / "Depth-Anything-V2-Small-hf"

MODEL_FILE = "model.safetensors"
IO_PREFIX = "io."
IO_SUFFIX = ".safetensors"
INPUT_PREFIX = "input."
OUTPUT_PREFIX = "output."

#: グラフ入力の名前（= {@link DepthMap.forward} の引数名）。export 後に突合してずれていたら
#: 止める — 位置で渡す以上、黙ってずれると golden の入力だけが入れ替わる。
INPUT_NAME = "pixel_values"

#: 前処理の正規化定数（正本 = 同梱 `preprocessor_config.json` の `image_mean` / `image_std`）。
IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)

#: `preprocessor_config.json` の `"resample"`。**PIL の定数で BICUBIC**（NEAREST=0 /
#: LANCZOS=1 / BILINEAR=2 / **BICUBIC=3** / BOX=4 / HAMMING=5）— SigLIP2 の 2 とは違う。
EXPECTED_RESAMPLE = 3

#: `rescale_factor`（1/255。f64 で逆数がちょうど 255.0 になることも
#: {@link check_processor} が見る）。
EXPECTED_RESCALE = 1.0 / 255.0

#: 実画像ケース = (ケース名, `outputs/demo/` のファイル名, 内容)。**画像の正本は
#: `examples/anima/eval-images.ts`**（プロンプト / seed / 解像度を持つのはあちらで、
#: `deno task demo:eval-images` でいつでも同じ 4 枚が焼き直せる）。ケース名は SigLIP2 /
#: BiRefNet の `REAL_CASES` と同じ綴り（同じ 4 枚を別の系列ディレクトリで読むだけ）。
REAL_CASES: tuple[tuple[str, str, str], ...] = (
    (
        "photo-portrait",
        "anima-default-1024x1024-defaultstep-seed42.png",
        "人物アップ + 桜並木（被写体が画面の大半・背景は遠景）",
    ),
    (
        "photo-landscape",
        "anima-default-1024x1024-defaultstep-seed43.png",
        "風景（手前の草地 → 湖 → 山と空の 3 段の奥行き）",
    ),
    (
        "photo-corridor",
        "anima-default-1024x1024-defaultstep-seed44.png",
        "校舎の廊下（強い遠近・消失点が画面内にある唯一の枚）",
    ),
    (
        "photo-street",
        "anima-default-1024x1024-defaultstep-seed45.png",
        "全身人物 + 街並み（手前の人物と、建物の上に抜ける空）",
    ),
)

#: 実画像の判別で見る「近い領域 / 遠い領域」の対 = ケース名 → (近側, 遠側)。各領域は
#: **深度マップ上の相対矩形** `(名前, y0, y1, x0, x1)`（0〜1 の比・上端 / 左端が 0）。
#:
#: MUST: 構図から一意に言える対だけを置く（「人物 > 空」「手前の床 > 消失点」）。深度は
#: 相対値なので絶対量には意味が無く、この**順序**だけが構図から演繹できる主張。TS 側の
#: `packages/models/tests/e2e_depth_anything_real_test.ts` が同じ矩形を実 GPU 出力に掛ける
#: （矩形を変えるなら両方 — 片方だけ動かすと「両者一致」の主張が黙って別物になる）。
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

#: 実画像 golden の `__metadata__` の欄（元画像の同定 — モジュール docstring の MUST）。
SOURCE_IMAGE_KEY = "source_image"
SOURCE_SHA256_KEY = "source_sha256"

#: 合成画像の乱数（`noise` ケース）。グローバル seed に依存しない。
SEED = 20260814

#: `disc` ケースの円の半径（画像の短辺を 1 とした比）。
DISC_RADIUS = 0.3

#: `disc` ケースの前景 / 背景色（RGB・値域 `[0, 1]`）。**3 チャネルが別々の値**なので、
#: チャネルの取り違え（前処理の統計の順序違い）が値に出る。
DISC_FOREGROUND: tuple[float, float, float] = (0.92, 0.78, 0.35)
DISC_BACKGROUND: tuple[float, float, float] = (0.12, 0.16, 0.24)

#: `checker` ケースの市松の 1 マス（画素）。patch（14）と割り切れない幅にして、格子と
#: パッチ格子が同位相になる偶然を避ける。
CHECKER_CELL = 40

#: 単調な奥行き手掛かりを持つ唯一のケース（{@link _sanity} の相関がここで最大になる）。
RAMP_CASE = "ramp"


def default_out_dir(model_dir: Path) -> Path:
    """生成物の既定の置き場（`outputs/series/<モデル名>/`）。

    解像度を綴りへ入れないのは、この family では解像度が**軸でない**ため（事前学習解像度の
    1 点しか通らない — モジュール docstring の「解像度」）。
    """
    return SERIES_ROOT / model_dir.name.lower()


class DepthMap(nn.Module):
    """相対深度マップ `[B, H, W]` 1 本だけを返す export 用ラッパ。"""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.model(pixel_values).predicted_depth


def normalize_image(image: torch.Tensor) -> torch.Tensor:
    """`[0, 1]` の `[1, 3, S, S]` を ImageNet 統計で正規化する（DPTImageProcessor の逐語）。"""
    mean = torch.tensor(IMAGENET_MEAN).reshape(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).reshape(1, 3, 1, 1)
    return (image - mean) / std


def disc_mask(size: int) -> torch.Tensor:
    """`disc` ケースの円内（`[S, S]` の bool）。"""
    axis = (torch.arange(size, dtype=torch.float32) + 0.5) / size * 2.0 - 1.0
    squared = axis.reshape(size, 1) ** 2 + axis.reshape(1, size) ** 2
    return squared <= DISC_RADIUS**2


def ramp_plane(size: int) -> torch.Tensor:
    """対角ランプの座標 `[S, S]`（`ramp` ケースの土台 = {@link _sanity} の相関の相手）。"""
    axis = torch.linspace(0.0, 1.0, size)
    return (axis.reshape(1, size) + axis.reshape(size, 1)) / 2.0


def _disc_image(size: int) -> torch.Tensor:
    """暗い背景に明るい円を置いた `[1, 3, S, S]`。"""
    foreground = torch.tensor(DISC_FOREGROUND).reshape(3, 1, 1)
    background = torch.tensor(DISC_BACKGROUND).reshape(3, 1, 1)
    return torch.where(disc_mask(size), foreground, background).unsqueeze(0)


def _ramp_image(size: int) -> torch.Tensor:
    """左上から右下への対角ランプ（チャネルごとに位相をずらす）。"""
    shifts = torch.tensor([0.0, 0.15, 0.3]).reshape(3, 1, 1)
    return ((ramp_plane(size).unsqueeze(0) + shifts) % 1.0).unsqueeze(0)


def _checker_image(size: int) -> torch.Tensor:
    """市松（{@link CHECKER_CELL} 画素角）。チャネルごとに明暗を反転させる。"""
    index = torch.arange(size) // CHECKER_CELL
    parity = ((index.reshape(1, size) + index.reshape(size, 1)) % 2).to(torch.float32)
    channels = torch.stack([parity, 1.0 - parity, parity * 0.5 + 0.25])
    return channels.unsqueeze(0)


def build_cases(resolution: int) -> tuple[tuple[str, torch.Tensor], ...]:
    """golden 4 ケースの `(名前, pixel_values)`（**正規化済み**の合成画像）。"""
    generator = torch.Generator().manual_seed(SEED)
    noise = torch.rand((1, 3, resolution, resolution), generator=generator)
    return (
        ("disc", normalize_image(_disc_image(resolution))),
        (RAMP_CASE, normalize_image(_ramp_image(resolution))),
        ("checker", normalize_image(_checker_image(resolution))),
        ("noise", normalize_image(noise)),
    )


def check_processor(processor: Any) -> dict[str, Any]:
    """image processor が期待どおりの実体・定数であることを**現物で**実測する。

    ここが見ているのは「karume 側の前処理層が合わせるべき相手」の同定で、外れたら 1 枚も
    書かずに落とす（`siglip2.preprocess.check_processor_shape` と同じ規律）。とくに
    `resample` は SigLIP2（2 = bilinear）と**違う** 3 = bicubic で、karume 側は既定が
    bilinear（既存 2 ファミリの値）なので、ここが動いたのに TS 側の `resizeRgb8` へ渡す
    フィルタが据え置かれると、実 GPU E2E の `pixel_values` 門が落ちるまで気づけない。

    NOTE: DA-V2 では config の 3 と `DPTImageProcessor` のクラス属性の既定 3 が一致するので、
    この門は「config を読んだか既定を見たか」は分けられない（SigLIP2 側はそこが分かれる）。
    分けられるのは**値が変わったこと**で、この family ではそれで足りる。
    """
    from transformers.image_processing_backends import TorchvisionBackend

    if not isinstance(processor, TorchvisionBackend):
        raise SystemExit(
            f"image processor が TorchvisionBackend でない: {type(processor).__mro__}"
            "（transformers 5.x の既定は fast 側 — PIL 側に落ちていると resize の実装が変わる）"
        )
    if int(processor.resample) != EXPECTED_RESAMPLE:
        raise SystemExit(
            f"resample が {int(processor.resample)}（期待 {EXPECTED_RESAMPLE} = PIL BICUBIC）"
        )
    if tuple(processor.image_mean) != IMAGENET_MEAN or tuple(processor.image_std) != IMAGENET_STD:
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
        "keep_aspect_ratio": bool(processor.keep_aspect_ratio),
        "ensure_multiple_of": int(processor.ensure_multiple_of),
    }


def build_real_cases(
    model_dir: Path, resolution: int, demo_root: Path
) -> tuple[tuple[str, torch.Tensor, dict[str, str]], ...]:
    """実画像 4 枚の `(名前, pixel_values, __metadata__)`（{@link REAL_CASES}）。

    前処理は `AutoImageProcessor` が返す**現物**（`DPTImageProcessor` = fast 側）で、写経
    ではない。BiRefNet 側が handler.py を写しているのは上流の前処理が推論と無関係な依存を
    引き込むためで、こちらは transformers の公開クラスがそのまま正本。

    MUST: 画像が 1 枚でも欠けていたら、重い import より**先に**止める（`--real-images` は
    明示の意思表示なので、黙って 3 枚で書くと golden の欠けに気づけない）。
    """
    paths: list[tuple[str, str, Path]] = []
    for name, file_name, _why in REAL_CASES:
        path = demo_root / file_name
        if not path.is_file():
            raise SystemExit(
                f"実画像 {path} が無い（生成: deno task demo:eval-images"
                " --source <Anima 配布形のパス>）"
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
                f"{name} の pixel_values が {tuple(pixel_values.shape)}"
                f"（期待の {expected_shape} と違う）— 正方でない画像は keep_aspect_ratio の"
                " 経路に入り、焼いたグラフの入力形と合わない"
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
    """実重みを読み、差し替え版が前提にする構成を検査する（パッチ前）。"""
    from transformers import AutoModelForDepthEstimation

    model = AutoModelForDepthEstimation.from_pretrained(str(model_dir), dtype=torch.float32)
    model.eval()
    # 重い forward の前に構成を見る（差し替え版の対象外なら 1 枚も焼かずに落とす）。
    patch.assert_supported(model)
    return model


def load_wrapper(model_dir: Path) -> tuple[DepthMap, int]:
    """パッチを当てた export 可能なラッパと、事前学習解像度を返す。

    MUST: 差し替えは golden を採る**前**。後に当てると期待値だけが元の経路で計算され、
    グラフと食い違ったまま緑になる。
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
    """各ケースの入力と torch CPU 期待出力を `io.<case>.safetensors` へ書く。

    `metadata` はケース名 → `__metadata__`（実画像ケースだけが持つ — 元画像の同定）。

    戻り値の 2 本目は sanity 記録用の期待出力（`[1, H, W]` の形のまま渡す）。
    """
    if len(graph.outputs) != 1:
        raise AssertionError(f"IR 出力が {len(graph.outputs)} 本（深度マップは 1 本）")
    written: list[str] = []
    depths: dict[str, torch.Tensor] = {}
    for name, pixel_values in cases:
        with torch.no_grad():
            output = wrapper(pixel_values)
        # MUST: io も IR の意味論 dtype の実表現へ落とす（ADR 0009 の境界正規化）。ランタイムが
        # 受け取る形と揃っていないと Deno 側 E2E が golden を読めない。
        tensors = {
            f"{INPUT_PREFIX}{INPUT_NAME}": normalize_boundary_tensor(
                pixel_values, f"{name} の入力 '{INPUT_NAME}'"
            ),
            f"{OUTPUT_PREFIX}0": normalize_boundary_tensor(
                output.detach().contiguous(), f"{name} の出力 0"
            ),
        }
        path = out_dir / f"{IO_PREFIX}{name}{IO_SUFFIX}"
        entry = None if metadata is None else metadata.get(name)
        save_file(tensors, str(path), metadata=None if entry is None else dict(entry))
        written.append(path.name)
        depths[name] = output.detach()
    return written, depths


def _ramp_correlation(depth: torch.Tensor) -> float:
    """深度マップと対角ランプ座標のピアソン相関（{@link _sanity} の判別量）。"""
    plane = ramp_plane(depth.shape[-1])
    if depth.shape[-2:] != plane.shape:
        raise AssertionError(f"出力 {tuple(depth.shape)} が正方でない（相関の相手と組めない）")
    left = (depth.reshape(-1) - depth.mean()).to(torch.float64)
    right = (plane.reshape(-1) - plane.mean()).to(torch.float64)
    return float((left * right).sum() / (left.norm() * right.norm()))


def _assert_distinct(depths: Mapping[str, torch.Tensor]) -> None:
    """ケースどうしの出力が**互いに違う**こと（同じ地図が並ぶのは入力が届いていない形）。"""
    names = sorted(depths)
    for index, name in enumerate(names):
        for other in names[index + 1 :]:
            if torch.equal(depths[name], depths[other]):
                raise AssertionError(f"{name} と {other} の出力が同一（入力が効いていない）")


def _region_mean(depth: torch.Tensor, region: tuple[str, float, float, float, float]) -> float:
    """深度マップの相対矩形（{@link REAL_REGIONS} の綴り）の平均。"""
    _label, y0, y1, x0, x1 = region
    height, width = depth.shape[-2], depth.shape[-1]
    patch = depth[..., int(y0 * height) : int(y1 * height), int(x0 * width) : int(x1 * width)]
    if patch.numel() == 0:
        raise AssertionError(f"領域 {region} が {height}×{width} の地図で空になった")
    return float(patch.mean())


def _real_sanity(depths: Mapping[str, torch.Tensor]) -> dict[str, dict[str, float]]:
    """実画像の判別（構図から言える近い領域 > 遠い領域）を**順序**で見る。

    MUST: 閾値を置かず順序で見る（合成画像側と同じ流儀）。矩形は構図から一意に決まる対だけを
    置いてあり（{@link REAL_REGIONS}）、一様に潰れた出力も入力非依存の出力も、両領域の平均が
    並ぶので落ちる。上下反転・左右反転・軸の取り違えもここで落ちる（近側は必ず画面下方 /
    中央寄りにある）。
    """
    _assert_distinct(depths)
    report: dict[str, dict[str, float]] = {}
    for name, depth in sorted(depths.items()):
        near, far = REAL_REGIONS[name]
        near_mean, far_mean = _region_mean(depth, near), _region_mean(depth, far)
        if near_mean <= far_mean:
            raise AssertionError(
                f"{name} の近側 {near[0]} の深度平均 {near_mean:.4f} が"
                f" 遠側 {far[0]} の {far_mean:.4f} 以下 — 構図の遠近を当てられていない"
            )
        report[name] = {near[0]: round(near_mean, 4), far[0]: round(far_mean, 4)}
    return report


def _sanity(depths: dict[str, torch.Tensor]) -> dict[str, Any]:
    """出力が入力の幾何を追えていることを**順序**で見る（モジュール docstring の「sanity」）。

    MUST: 恒真な sanity にしない。値域の検査だけ（ReLU なので非負は自明に近い）では、
    一様に潰れた出力も入力非依存の出力も素通りする。相関の**最大がどのケースか**まで見れば、
    単調な奥行き手掛かりを持つ 1 枚だけが立つ状態でなければ落ちる。

    MUST: 掛けるのは**合成 4 ケースだけ**（実画像は {@link _real_sanity} が別に持つ）。
    対角ランプとの相関は「合成 4 枚の中で `ramp` が最大」という主張で、実写を混ぜると
    「たまたま対角に明るい写真」が最大を奪って判別が別物になる。
    """
    _assert_distinct(depths)
    for name, depth in depths.items():
        minimum, maximum = float(depth.min()), float(depth.max())
        if minimum < 0.0:
            raise AssertionError(f"{name} の最小 {minimum} が負（head 末尾の ReLU と矛盾）")
        if not minimum < maximum:
            raise AssertionError(f"{name} の出力が一様（min == max == {minimum}）")

    correlations = {name: _ramp_correlation(depth) for name, depth in depths.items()}
    best = max(correlations, key=lambda name: correlations[name])
    if best != RAMP_CASE:
        raise AssertionError(
            f"対角ランプとの相関が最大なのが {best!r}（期待は {RAMP_CASE!r}）—"
            f" 出力が入力の幾何を追えていない: {correlations}"
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
    demo_root: Path = DEMO_ROOT,
) -> dict[str, Any]:
    """IR コンテナと golden io を書き、要約を返す。

    `real_images` を立てると合成 4 ケースに実画像 4 ケースを**足す**（置き換えない —
    モジュール docstring の「golden の 2 群」）。

    MUST: 生成物は作業席へ書き、**全ての門**（入力の並び・sanity）を通してから据える。門より前に
    final へ置くと、落ちた実走が「検収門を通れる資産」を残す — io golden は同じ壊れたラッパから
    採るので互いに整合し、TS 側の突合は**緑になる**（「いつ公開してよいか」の綴りは
    {@link _shared.decode_series._publish}・据え替えと後片付けの規律は core の原語
    {@link karume.artifacts.staged_publication}）。
    """
    wrapper, resolution = load_wrapper(model_dir)
    synthetic = build_cases(resolution)
    # MUST: 実画像は emit より先に組む（画像が欠けているなら、99MB を書き切ってから落とすの
    # ではなくここで止める）。
    real = build_real_cases(model_dir, resolution, demo_root) if real_images else ()
    cases = [*synthetic, *((name, pixel_values) for name, pixel_values, _md in real)]
    metadata = {name: md for name, _pixel_values, md in real}
    out_dir.parent.mkdir(parents=True, exist_ok=True)

    _, example = synthetic[0]
    with staged_publication(out_dir) as staged:
        # ディレクトリの席は書き手が作る（原語は席を作らない — path しか渡さない）。
        staged.mkdir()
        # 動的軸は無い（解像度は事前学習の 1 点に固定 — モジュール docstring）。
        graph = export_to_file(wrapper, (example,), staged / MODEL_FILE, symbol_names=())
        declared = tuple(item.name for item in graph.inputs)
        if declared != (INPUT_NAME,):
            raise AssertionError(f"グラフ入力の並びが {declared} で、期待の {(INPUT_NAME,)} と違う")
        written, depths = _write_io(wrapper, graph, cases, staged, metadata)
        # MUST: 公開より前に評価する（この系列で唯一の非恒真な検査 — 落ちたら席ごと消える）。
        sanity = _sanity({name: depths[name] for name, _pixel_values in synthetic})
    summary = {
        "dir": str(out_dir),
        "resolution": resolution,
        "nodes": len(graph.nodes),
        "outputs": len(graph.outputs),
        "initializers": len(graph.initializers),
        "model_bytes": sum(p.stat().st_size for p in resolve_shards(out_dir / MODEL_FILE)),
        "ops": sorted(graph.required_ops),
        "symbols": list(graph.symbols),
        "io": written,
        "input_shape": list(example.shape),
        "output_shape": list(depths[synthetic[0][0]].shape),
        "real_images": {name: md[SOURCE_IMAGE_KEY] for name, md in metadata.items()},
        "sanity": sanity,
    }
    if metadata:
        summary["real_sanity"] = _real_sanity({name: depths[name] for name in metadata})
    return summary


def _diff_entry(
    stage: str,
    claim: str,
    got: dict[str, torch.Tensor],
    expected: dict[str, torch.Tensor],
) -> dict[str, Any]:
    """1 段分の同値レポート（ケース名 → maxdiff と、全ケースのビット一致）。

    `bit_exact` は「差 0」より強い主張（`0.0 == -0.0` は差 0 だがビットは違う）で、演算を
    1 つも増減させない書き換えのように**ビット一致が主張の中身**である段で意味を持つ。
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
    """パッチ前 eager との同値を **2 点**で実測する（`birefnet.export.py --verify` と同じ形）。

    1. 最終融合段の倍率 → 寸法指定と、位置埋め込み補間の門（①②）→ **bit_exact が主張の
       中身**。演算列が 1 対 1 で対応するので差は 0 でなければならず、外れたらここで落とす。
    2. `ConvTranspose2d` → 1×1 conv + pixel shuffle（③）→ **maxdiff を報告**。Cin 方向の
       縮約順序が ATen の転置畳み込みと違うぶんだけ最下位ビットが動く。

    MUST: 順序は「**全ケースの参照値を確定** → 1 → 2」。パッチはクラス属性のプロセス全域
    差し替えなので、段ごとに参照を採り直すと 2 段目の参照がパッチ後の値になる（恒真化）。
    """
    if patch.patches_applied():
        raise SystemExit(
            "パッチ適用済みのプロセスでは参照を採れない（同値検証が差 0 で恒真化する）"
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
            f"演算列が 1 対 1 の書き換えがビット同一でない: maxdiff={layout['maxdiff']}"
        )
    patch.apply_module_patches(wrapper.model)
    return [layout, _diff_entry("modules", "max-abs-diff", depths(), reference)]


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="出力先（既定は outputs/series/<モデル名>/）",
    )
    parser.add_argument(
        "--real-images",
        action="store_true",
        help=f"`{DEMO_ROOT.name}/` の実画像 4 枚を通した golden も書く（合成 4 ケースに"
        "足す）。前処理は AutoImageProcessor の現物で、元画像の sha256 を __metadata__ に載せる",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="パッチ前 eager との同値を 2 点で実測する（emit はしない — 同一プロセスでは"
        "クラス属性の差し替えが参照を汚染するので併用できない）",
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
