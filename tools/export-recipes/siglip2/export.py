"""実重み SigLIP2 の **vision tower** を IR v1 コンテナ + golden io へ書き出す台本。

`embeddinggemma/export.py` と同じ役割（実重み・単一ベクトル出力の数値一致）を、画像側で
受け持つ。生成物は `outputs/series/` 配下で、リポジトリ直下の `.gitignore` によりコミット
対象外（重み 350MB〜1.7GB）。

    uv run --group siglip2 python -m siglip2.export
    uv run --group siglip2 python -m siglip2.export --verify  # パッチ前 eager との同値検証

transformers は **5.14.1 でピン**する（`siglip2.patch` がモデリングコードのクラス
属性を差し替えるため — pyproject.toml の `siglip2` グループ）。

## モデル軸

対象は `--model-dir` の 1 軸で、系列名はそのディレクトリ名から導く（`export_sbv2.py` /
`irodori/export.py` と同じ持ち方）— 出力先を固定にすると、別のモデルを書き出した瞬間に先の
系列が黙って上書きされる。

    uv run --group siglip2 python -m siglip2.export \
        --model-dir ../../inputs/siglip2/siglip2-so400m-patch14-384

規模の差（hidden / 層数 / head 数 / intermediate / patch / 解像度）は**全て config.json 由来**
で、台本はどこにも焼かない。合成画像の寸法も `config.image_size` / `config.patch_size` から
組む（{@link build_cases}）。

## 何をグラフに載せるか

`SiglipVisionModel` の **pooler_output 1 本だけ**（MAP head を通した `[B, hidden]`）。
`last_hidden_state` は出さない — 読み戻しが `[B, パッチ数, hidden]` と 2 桁大きく、埋め込み
用途では使われない（ADR 0026 の「出力を絞る」と同じ判断）。text tower は載せない。

batch は **静的 1**。画像側は解像度も系列長（= パッチ数）も固定で、動かす軸が無い。

MUST: SDPA は**保存しない**（既定の `PRESERVED_OP_PREFIXES` 11 本のまま）。attention は
エンコーダの層数 + MAP head の 1 本とも mul×2 + bmm + softmax + bmm へ分解される
（base は 12 + 1 / so400m は 27 + 1）。ADR 0023 の融合 attention はこの系列では opt-in しない。

MUST: モデルは `siglip2.patch` の 2 段パッチを当ててから export する（当てないと
`aten.conv2d.padding` / `aten.select.int` / `aten.embedding` の i64 initializer が残る）。
golden の期待値は**差し替え後**のモジュールから採る — 差し替え前後の差はパッチ層の責務で、
`--verify` が 2 点評価で実測する（1 点目 = 形の畳み込みだけ・ビット同一が主張の中身、
2 点目 = MAP head まで・maxdiff を報告）。

MUST: `--verify` は emit しない（同一プロセスでは併用できない）。パッチはクラス属性の
プロセス全域差し替えなので、emit のために当てた後では「当てる前の参照」が採れない。

## 入力の約束（前処理はグラフに載せない）

グラフ入力は **正規化済みの** `pixel_values f32 [1, 3, 解像度, 解像度]`（base 224 /
so400m 384）。画像の decode はホストの責務、resize（`config.image_size` へ bilinear）/
rescale（1/255）/ normalize（mean = std = 0.5 → `[-1, 1]`）は karume 側
（`packages/models/src/image/preprocess.ts`・パリティ台本は `siglip2/preprocess.py`）。
定数の正本は重みと同じ場所の `preprocessor_config.json`（2 モデルとも mean = std = 0.5・
**resample 2 = PIL の BILINEAR**〈BICUBIC は 3〉で、違うのは寸法だけ）。

## golden の 2 群（合成画像 + 実画像）

既定の 4 ケースは**合成画像**（{@link build_cases}）で、値域の端や勾配を踏むので数値回帰の
検出が鋭い。`--real-images` を付けると、それに加えて `outputs/demo/` の**実画像** 4 枚
（{@link REAL_CASES}）を通した golden も書く。実画像の側は穏やかな値しか踏まないかわりに、
**前処理を含めた鎖**（TS 側は PNG → `preprocess.ts` → karume 推論、こちら側は PIL →
`TorchvisionBackend` → torch 推論）の突合になる。どちらも残す（片方が他方を代替しない）。

    uv run --group siglip2-preprocess python -m siglip2.export --real-images

MUST: `--real-images` は **`siglip2-preprocess` グループ**で回す。実画像の前処理は
`AutoImageProcessor` の fast 側（`TorchvisionBackend`）が正本で、torchvision と pillow が要る
（`siglip2` グループは export に要らないその 2 つを持たない — pyproject.toml）。

MUST: 実画像 golden には**元画像の sha256** を `__metadata__` に載せる。画像は
`examples/anima/eval-images.ts` でいつでも焼き直せる生成物なので、焼き直したのに golden を
採り直していない環境では、突合ではなく**入力が違う**（そして値は一見それらしく出る）。

## 出力レイアウト

系列名は `--model-dir` のディレクトリ名（既定の `inputs/siglip2/siglip2-base-patch16-224/`
なら `siglip2-base-patch16-224`）:

    outputs/series/<系列名>/model.safetensors     重み・定数 + __metadata__
    outputs/series/<系列名>/io.<case>.safetensors 入力と torch CPU 期待出力

io のテンソルキー規約は tiny golden / DeBERTa / EmbeddingGemma と同じ
（`input.<グラフ入力名>` / `output.<位置>`）。
"""

from __future__ import annotations

import argparse
import hashlib
import io as io_module
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import save_file
from torch import nn

from _shared.paths import INPUTS_ROOT, OUTPUTS_ROOT, SERIES_ROOT
from karume.convert import normalize_boundary_tensor
from karume.ir import IrGraph
from karume.pipeline import export_to_file
from karume.shards import resolve_shards

from . import patch

#: 実重みの親（`hf download google/<名前> --local-dir inputs/siglip2/<名前>` の展開先）。
MODELS_ROOT = INPUTS_ROOT / "siglip2"

#: 実画像の置き場（`rm -rf` で安全に消せる席 — docs/assets-layout.md）。
DEMO_ROOT = OUTPUTS_ROOT / "demo"

#: 既定のモデル（`--model-dir` 未指定のとき）。
DEFAULT_MODEL_DIR = MODELS_ROOT / "siglip2-base-patch16-224"


def default_out_dir(model_dir: Path) -> Path:
    """生成物の既定の置き場（`outputs/series/<実重みのディレクトリ名>/`）。

    モデル名を系列の綴りへ焼くのは `export_sbv2.py` の `default_out_root` と同じ理由 —
    固定の綴りを共有すると、別のモデルを書き出した瞬間に先の資産が黙って上書きされる。

    格納 dtype は f32 のみなので dtype 別の枝は無い（f16 / i8 は別系列で決める話）。
    """
    return SERIES_ROOT / model_dir.name


MODEL_FILE = "model.safetensors"
IO_PREFIX = "io."
IO_SUFFIX = ".safetensors"
INPUT_PREFIX = "input."
OUTPUT_PREFIX = "output."

#: グラフ入力の名前（= `VisionPooler.forward` の引数名）。export 後に突合してずれていたら
#: 止める — 位置で渡す以上、黙ってずれると golden の入力だけが入れ替わる。
INPUT_NAME = "pixel_values"

#: 合成画像の乱数（`noise` ケース）。グローバル seed に依存しない。
SEED = 20260813

#: チャネルごとの明度オフセット。3 チャネルが同一の面だと、チャネルの取り違え（patch
#: 埋め込みの重みレイアウト誤り）が値に出ない。
CHANNEL_OFFSETS: tuple[float, ...] = (-0.2, 0.0, 0.2)

#: `ramp-dim` のコントラスト比（`ramp` と**構造は同じで強度だけ違う**近い対を作る）。
DIM_SCALE = 0.5

#: sanity で見る cosine の対（近い対, 遠い対）。近い対が遠い対を上回らなければ、
#: 「数値は動いているが画像埋め込みとして壊れている」— 生成時に fail loudly にする。
NEAR_PAIR = ("ramp", "ramp-dim")
FAR_PAIR = ("ramp", "checker")

#: 実画像ケース = (ケース名, `outputs/demo/` のファイル名, 内容)。**画像の正本は
#: `examples/anima/eval-images.ts`**（プロンプト / seed / 解像度を持つのはあちらで、
#: `deno task demo:eval-images` でいつでも同じ 4 枚が焼き直せる）。
REAL_CASES: tuple[tuple[str, str, str], ...] = (
    (
        "photo-portrait",
        "anima-default-1024x1024-defaultstep-seed42.png",
        "人物アップ + 桜並木（前景と背景が明確）",
    ),
    (
        "photo-landscape",
        "anima-default-1024x1024-defaultstep-seed43.png",
        "風景（山・湖・森・人物なし）",
    ),
    (
        "photo-corridor",
        "anima-default-1024x1024-defaultstep-seed44.png",
        "校舎の廊下（強い遠近・消失点）",
    ),
    (
        "photo-street",
        "anima-default-1024x1024-defaultstep-seed45.png",
        "全身人物 + 街並み（背景が複雑）",
    ),
)

#: 実画像の判別で見る 2 群（人物が写っている 2 枚 / 写っていない 2 枚）。**人物どうしの
#: cosine が、人物と風景の 4 対のどれよりも高い**ことを見る（合成画像の NEAR/FAR より
#: 意味のある判別 — 閾値は置かず順序そのものを見る）。TS 側の
#: `packages/runtime/tests/e2e_siglip2_test.ts` が同じ群を実 GPU 出力に掛ける。
REAL_PERSON_CASES = ("photo-portrait", "photo-street")
REAL_SCENE_CASES = ("photo-landscape", "photo-corridor")

#: 実画像 golden の `__metadata__` の欄（元画像の同定 — モジュール docstring の MUST）。
SOURCE_IMAGE_KEY = "source_image"
SOURCE_SHA256_KEY = "source_sha256"


class VisionPooler(nn.Module):
    """pooler_output（MAP head 経由の `[B, hidden]`）1 本だけを返す export 用ラッパ。"""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.model(pixel_values=pixel_values).pooler_output


def _stack_channels(plane: torch.Tensor, channels: int) -> torch.Tensor:
    """`[S, S]` の 1 面をチャネルごとにずらして `[1, C, S, S]` にする。"""
    offsets = torch.tensor(CHANNEL_OFFSETS[:channels]).reshape(channels, 1, 1)
    return (plane.unsqueeze(0) + offsets).clamp(-1.0, 1.0).unsqueeze(0)


def _ramp_plane(size: int) -> torch.Tensor:
    """左上から右下への対角ランプ（`[S, S]`・値域 `[-1, 1]`）。"""
    axis = torch.linspace(-1.0, 1.0, size)
    return (axis.reshape(1, size) + axis.reshape(size, 1)) / 2.0


def _checker_plane(size: int, cell: int) -> torch.Tensor:
    """パッチ格子に揃えた市松（`[S, S]`・値は ±1）。"""
    index = torch.arange(size) // cell
    parity = (index.reshape(1, size) + index.reshape(size, 1)) % 2
    return parity.to(torch.float32) * 2.0 - 1.0


def build_cases(config: Any) -> tuple[tuple[str, torch.Tensor], ...]:
    """golden 4 ケースの `(名前, pixel_values)`（**正規化済み**の合成画像）。

    実画像を使わないのはモジュール docstring の理由（前処理がまだ karume 側に無い）。
    構造だけを変えた対を持たせて、{@link _sanity} の cosine 順序が恒真にならないようにする:
    `ramp` と `ramp-dim` は同じ構造で強度だけ違い、`checker` は構造ごと違う。
    """
    size = int(config.image_size)
    channels = int(config.num_channels)
    ramp = _ramp_plane(size)
    generator = torch.Generator().manual_seed(SEED)
    noise = torch.rand((size, size), generator=generator) * 2.0 - 1.0
    return (
        ("ramp", _stack_channels(ramp, channels)),
        ("ramp-dim", _stack_channels(ramp * DIM_SCALE, channels)),
        ("checker", _stack_channels(_checker_plane(size, int(config.patch_size)), channels)),
        ("noise", _stack_channels(noise, channels)),
    )


def load_image_processor(model_dir: Path) -> Any:
    """実画像の前処理に使う image processor を、実体検査ごと**正本から**借りる。

    MUST: 検査を書き直さない（`siglip2/preprocess.py` が前処理の正本で、TS 側のパリティ門も
    あちらのフィクスチャで閉じている）— 2 本目の検査を持つと、片方だけ直したときに黙って
    食い違う。import が関数内なのは、あちらがこの台本から既定パスを取っており、モジュール
    先頭で読むと循環になるため。
    """
    from .preprocess import check_processor_shape, load_processor

    processor = load_processor(model_dir)
    check_processor_shape(processor)
    return processor


def build_real_cases(
    model_dir: Path, config: Any, demo_root: Path
) -> tuple[tuple[str, torch.Tensor, dict[str, str]], ...]:
    """実画像 4 枚の `(名前, pixel_values, __metadata__)`（{@link REAL_CASES}）。

    前処理は **TS 側と同じ経路**（`AutoImageProcessor` の fast 側 = `TorchvisionBackend`）を
    通す。この golden の主張は「TS 前処理 + karume 推論」対「Python 前処理 + torch 推論」の
    突合なので、ここで別の resize を使うと主張の中身が消える。

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

    processor = load_image_processor(model_dir)
    expected_shape = (1, int(config.num_channels), int(config.image_size), int(config.image_size))
    cases: list[tuple[str, torch.Tensor, dict[str, str]]] = []
    for name, file_name, path in paths:
        raw = path.read_bytes()
        with Image.open(io_module.BytesIO(raw)) as image:
            array = np.array(image.convert("RGB"))
        pixel_values = processor(
            images=array, input_data_format="channels_last", return_tensors="pt"
        )["pixel_values"]
        # MUST: 前処理の寸法が vision config と一致することを見る（`preprocessor_config.json`
        # と `config.json` は別ファイルで、食い違えば golden だけが別解像度で焼かれる）。
        if tuple(pixel_values.shape) != expected_shape:
            raise AssertionError(
                f"{name} の pixel_values が {tuple(pixel_values.shape)}"
                f"（config 由来の {expected_shape} と違う）"
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
    """vision tower の実重みを読み、差し替え版が前提にする config を検査する（パッチ前）。

    MUST: `attn_implementation="sdpa"` — エンコーダ 12 層の attention を SDPA として出させる
    （既定の分解表がそれを mul×2 + bmm + softmax + bmm に開く）。
    """
    from transformers import SiglipVisionModel

    model = SiglipVisionModel.from_pretrained(
        model_dir, dtype=torch.float32, attn_implementation="sdpa"
    )
    model.eval()
    patch.assert_supported(model.config)
    return model


def load_wrapper(model_dir: Path) -> VisionPooler:
    """パッチを 2 段とも当てた export 可能なラッパを返す。

    MUST: 差し替えは golden を採る**前**（`siglip2.patch` の docstring）— 後に当てると
    期待値だけが元の経路で計算され、グラフと食い違ったまま緑になる。
    """
    model = load_model(model_dir)
    patch.apply_shape_patches(model.embeddings)
    patch.apply_map_head_patch(model.head)
    return VisionPooler(model).eval()


def _write_io(
    wrapper: nn.Module,
    graph: IrGraph,
    cases: Sequence[tuple[str, torch.Tensor]],
    out_dir: Path,
    metadata: Mapping[str, Mapping[str, str]] | None = None,
) -> tuple[list[str], dict[str, torch.Tensor]]:
    """各ケースの入力と torch CPU 期待出力を `io.<case>.safetensors` へ書く。

    `metadata` はケース名 → `__metadata__`（実画像ケースだけが持つ — 元画像の同定）。

    戻り値の 2 本目は sanity 記録用の期待出力（`[B, H]` の形のまま渡す）。
    """
    if len(graph.outputs) != 1:
        raise AssertionError(f"IR 出力が {len(graph.outputs)} 本（pooler_output は 1 本）")
    written: list[str] = []
    pooled: dict[str, torch.Tensor] = {}
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
        pooled[name] = output.detach()
    return written, pooled


def _sanity(pooled: dict[str, torch.Tensor]) -> dict[str, Any]:
    """構造の近い対の cosine が、構造ごと違う対を上回ることを見る。

    MUST: 恒真な sanity にしない。pooler_output は L2 正規化されていないのでノルム検査は
    使えず、「同一入力どうしの cosine = 1」も何も検証しない。**別々の入力**どうしの順序で
    見れば、埋め込みが 1 点へ潰れた場合（近い対も遠い対も cosine 1）も、パッチ埋め込みや
    MAP head が死んだ場合も落ちる。
    """
    vectors = {name: output.reshape(-1) for name, output in pooled.items()}

    def cosine(pair: tuple[str, str]) -> float:
        first, second = vectors[pair[0]], vectors[pair[1]]
        return float(torch.dot(first, second) / (first.norm() * second.norm()))

    near, far = cosine(NEAR_PAIR), cosine(FAR_PAIR)
    if near <= far:
        raise AssertionError(
            f"cosine の順序が構造と逆: {NEAR_PAIR}={near:.4f} <= {FAR_PAIR}={far:.4f}"
        )
    report: dict[str, Any] = {
        "l2_norms": {name: round(float(vector.norm()), 5) for name, vector in vectors.items()},
        "cosine": {
            f"{NEAR_PAIR[0]}×{NEAR_PAIR[1]}": round(near, 4),
            f"{FAR_PAIR[0]}×{FAR_PAIR[1]}": round(far, 4),
            "ramp×noise": round(cosine(("ramp", "noise")), 4),
        },
    }
    if set(REAL_PERSON_CASES) | set(REAL_SCENE_CASES) <= set(vectors):
        report["real_cosine"] = _real_sanity(cosine)
    return report


def _real_sanity(cosine: Callable[[tuple[str, str]], float]) -> dict[str, float]:
    """実画像の判別（人物どうし > 人物と風景）を見る（{@link REAL_PERSON_CASES}）。

    MUST: 閾値を置かず**順序**で見る（合成画像側と同じ流儀）。人物 2 枚は構図
    （アップ / 全身）も背景（桜並木 / 街並み）も違うので、この順序が成り立つのは
    「人が写っている」という意味を捉えている場合だけになる。
    """
    person = cosine(REAL_PERSON_CASES)
    cross = {
        f"{first}×{second}": cosine((first, second))
        for first in REAL_PERSON_CASES
        for second in REAL_SCENE_CASES
    }
    worst_name, worst = max(cross.items(), key=lambda item: item[1])
    if person <= worst:
        raise AssertionError(
            f"実画像の cosine の順序が逆: {REAL_PERSON_CASES}={person:.4f}"
            f" <= {worst_name}={worst:.4f}"
        )
    return {
        f"{REAL_PERSON_CASES[0]}×{REAL_PERSON_CASES[1]}": round(person, 4),
        **{name: round(value, 4) for name, value in cross.items()},
    }


def export_series(
    model_dir: Path, out_dir: Path, real_images: bool = False, demo_root: Path = DEMO_ROOT
) -> dict[str, Any]:
    """IR コンテナと golden io を書き、要約を返す。

    `real_images` を立てると合成 4 ケースに実画像 4 ケースを**足す**（置き換えない —
    モジュール docstring の「golden の 2 群」）。
    """
    wrapper = load_wrapper(model_dir)
    synthetic = build_cases(wrapper.model.config)
    # MUST: 実画像は emit より先に組む（画像が欠けているなら、1.7GB を書き切ってから落とすの
    # ではなくここで止める）。
    real = build_real_cases(model_dir, wrapper.model.config, demo_root) if real_images else ()
    cases = [*synthetic, *((name, pixel_values) for name, pixel_values, _md in real)]
    metadata = {name: md for name, _pixel_values, md in real}
    out_dir.mkdir(parents=True, exist_ok=True)

    _, example = synthetic[0]
    # 動的軸は無い（解像度もパッチ数も固定 — モジュール docstring）。
    graph = export_to_file(wrapper, (example,), out_dir / MODEL_FILE, symbol_names=())
    declared = tuple(item.name for item in graph.inputs)
    if declared != (INPUT_NAME,):
        raise AssertionError(f"グラフ入力の並びが {declared} で、期待の {(INPUT_NAME,)} と違う")
    written, pooled = _write_io(wrapper, graph, cases, out_dir, metadata)
    return {
        "dir": str(out_dir),
        "nodes": len(graph.nodes),
        "outputs": len(graph.outputs),
        "initializers": len(graph.initializers),
        "model_bytes": sum(p.stat().st_size for p in resolve_shards(out_dir / MODEL_FILE)),
        "ops": sorted(graph.required_ops),
        "symbols": list(graph.symbols),
        "io": written,
        "input_shape": list(example.shape),
        "real_images": {name: md[SOURCE_IMAGE_KEY] for name, md in metadata.items()},
        "sanity": _sanity(pooled),
    }


def _diff_entry(
    stage: str,
    claim: str,
    got: dict[str, torch.Tensor],
    expected: dict[str, torch.Tensor],
) -> dict[str, Any]:
    """1 段分の同値レポート（ケース名 → maxdiff と、全ケースのビット一致）。

    `bit_exact` は「差 0」より強い主張（`0.0 == -0.0` は差 0 だがビットは違う）で、形の
    畳み込みのように**ビット一致が主張の中身**である段で意味を持つ。
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
    """パッチ前 eager との同値を **2 点**で実測する（`export_sbv2.py --verify` と同じ形）。

    1. 形の畳み込みだけ（conv padding + 位置埋め込み）→ **bit_exact が主張の中身**。演算列が
       1 対 1 で対応するので差は 0 でなければならず、外れたらここで落とす。
    2. MAP head の q/k/v 明示化まで → **maxdiff を報告**。`nn.MultiheadAttention` は q にだけ
       1/√d を掛けるのに対し SDPA は q と k に対称に割るのでビット一致にはならない
       （実測 7.75e-07〜2.38e-06 / ベクトルの L2 ノルムは 12.7〜13.1）。

    MUST: 順序は「**全ケースの参照値を確定** → 1 → 2」。パッチはクラス属性のプロセス全域
    差し替えなので、段ごとに参照を採り直すと 2 段目の参照がパッチ後の値になる（恒真化）。
    """
    if patch.patches_applied():
        raise SystemExit(
            "パッチ適用済みのプロセスでは参照を採れない（同値検証が差 0 で恒真化する）"
        )
    model = load_model(model_dir)
    cases = build_cases(model.config)
    with torch.no_grad():
        reference = {
            name: model(pixel_values=pixel_values).pooler_output for name, pixel_values in cases
        }

    def pooled() -> dict[str, torch.Tensor]:
        with torch.no_grad():
            return {
                name: model(pixel_values=pixel_values).pooler_output for name, pixel_values in cases
            }

    patch.apply_shape_patches(model.embeddings)
    folds = _diff_entry("shape-folds", "bit-exact", pooled(), reference)
    if not folds["bit_exact"]:
        raise AssertionError(
            f"形の畳み込みがビット同一でない: maxdiff={folds['maxdiff']} —"
            " conv padding / 位置埋め込みの書き換えが同値でない"
        )
    patch.apply_map_head_patch(model.head)
    return [folds, _diff_entry("map-head", "max-abs-diff", pooled(), reference)]


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="出力先（既定は outputs/series/<--model-dir のディレクトリ名>/）",
    )
    parser.add_argument(
        "--real-images",
        action="store_true",
        help="outputs/demo/ の実画像 4 枚の golden も書く（合成 4 ケースに足す）。"
        "torchvision / pillow が要るので `uv run --group siglip2-preprocess` で回す",
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
