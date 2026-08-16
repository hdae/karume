"""実重みの BiRefNet 系（背景抜き / salient object segmentation）を IR v1 コンテナ +
golden io へ書き出す台本。モデルは `--model-dir` で選ぶ（既定 BiRefNet_HR — 下の「モデル軸」）。

生成物は `outputs/series/` 配下で、リポジトリ直下の `.gitignore` によりコミット対象外
（重み + 焼いた定数で 1024² のとき 964MB）。

    uv run --group birefnet python -m birefnet.export
    uv run --group birefnet python -m birefnet.export --verify  # パッチ前 eager との同値検証

transformers は **5.14.1 でピン**する（`birefnet.patch` が `trust_remote_code` の
動的モジュールのクラス属性を差し替えるため — pyproject.toml の `birefnet` グループ）。

## モデル軸と解像度軸

系列名は `--model-dir` のディレクトリ名（小文字ハイフン化）に **`--resolution` を足した**
綴り（既定は `birefnet-hr-1024`）。解像度を綴りに入れるのは、BiRefNet の系列が解像度ごとに
**別のグラフ**になるため — shifted-window マスクも H/W padding のゼロ定数も解像度依存の
定数として焼かれるので、同じ席へ 2 解像度を書くと先の資産が黙って上書きされる。

    uv run --group birefnet python -m birefnet.export --resolution 2048

モデル軸で受けているのは 2 つ（帰属表の正本は `birefnet.card.BIREFNET_UPSTREAM`）:

- `inputs/birefnet/BiRefNet_HR/` — 上流 `ZhengPeng7/BiRefNet_HR`（MIT）。既定。
- `inputs/birefnet/lucida/` — 上流 `egeorcun/lucida`（MIT）。BiRefNet_HR の fine-tune で、
  同梱 `birefnet.py` / `BiRefNet_config.py` は**バイト単位で同一**（実測 — 差は重みだけ）。
  したがって `birefnet.patch` の差し替えも `--verify` の 2 点評価もそのまま通る。

      uv run --group birefnet python -m birefnet.export --model-dir <リポの inputs>/birefnet/lucida

MUST: lucida の `lucida-m35-comfy.safetensors` は**対象外**（`hf download` から除く）。上流の
モデルカードが明示するとおり ComfyUI 向けに `Normalize` を初段 conv へ**焼き込んだ**重みで、
この台本の前処理（ImageNet 正規化を掛けてからグラフへ渡す）と二重適用になる。実測でも裏が
取れている: `bb.patch_embed.proj.weight` の入力チャネル別 RMS が v7 比で 4.352 / 4.453 / 4.428
= {@link IMAGENET_STD} の逆数（4.367 / 4.464 / 4.444）に 0.3% で一致し、bias から
`Σ_c mean_c · Σ_hw w[c]` を戻すと v7 の bias 分布（平均 −0.0197 / 標準偏差 0.2790 / 絶対最大
1.4296）へ 0.2% で戻る。前処理の有無を `pipelineConfig` の軸にして受けることもできるが、
①上流が「pipeline の中で使うもの（poster policy + SAM3 referee 込み）で bare 運用は非推奨」と
書いている実験枝の重みで、グラフ 1 本の配布形では再現できない ②そのために配布形の前処理を
分岐させると、全モデルが 2 通りの前処理を持つ設計になる — ので**受け入れない**。

**既定は 1024²**。本家の推論解像度（同梱 `handler.py` の General-HR）は 2048² だが、
2048² は ①conv2d の dispatch 上限（docs/limitations.md の n タイル 65,536）に decoder の
1×1 conv が当たる ②中間テンソルが `[1, 192, 2048, 2048]` = 3.22GB になる、の 2 点で
karume 側の別工事が要る（export 段は 2048² でも通る — 落ちるとすれば実行段）。

`--resolution` は **64 の倍数**だけを受ける: 本体側の `PatchMerging` が各段で偶数 H/W を
要求し（S/4・S/8・S/16 が偶数 = S%32）、`mul_scl_ipt='cat'` の半解像度枝が同じ要求を
S/2 に掛ける（= S%64）。外れた寸法は途中の reshape エラーではなくここで落とす。

## 何をグラフに載せるか

`BiRefNet.forward` は **list** を返す。中の要素は multi-scale supervision の中間予測
（m4 / m3 / m2）と最終段のマットで、**中間予測は `self.training` 分岐の中でだけ append
される** — つまり eval では要素 1 本（実測 `len(preds) == 1` / shape `[1, 1, S, S]`）。
同梱 `handler.py` も `preds[-1]` だけを使う。よってラッパは `preds[-1]` を返し、要素数が
1 でなければ fail loudly（{@link MatteLogits}）— 学習モードのグラフを黙って書き出さない。

出す値は **sigmoid 前の logit**（handler の `.sigmoid()` はホスト側の責務）。理由は 2 つ:
グラフを eager forward の逐語に保つと `--verify` の突合がそのまま実行段の主張になること、
そして sigmoid は飽和域で差を潰すので、golden に掛けると数値回帰の感度が落ちること。

batch は **静的 1**（`birefnet.patch` の窓マスク加算が B = 1 でのみ元実装と同値）。動的軸は
無い（`symbol_names=()`）。

MUST: モデルは `birefnet.patch` のパッチを当ててから export する。当てないと Swin の
rank-6 reshape / `roll` / step-2 slice などで変換段が落ちる（差し替えの一覧は
`birefnet.patch` の docstring）。golden の期待値は**差し替え後**のモジュールから採る —
差し替え前後の差はパッチ層の責務で、`--verify` が 2 点評価で実測する。

MUST: `apply` の後に `patch.prepare(...)` を通す（{@link load_wrapper}）。
shifted-window マスクは解像度依存の定数で、この eager 1 forward でだけ焼かれる。

MUST: `--verify` は emit しない（同一プロセスでは併用できない）。パッチはクラス属性の
プロセス全域差し替えなので、emit のために当てた後では「当てる前の参照」が採れない。

## 入力の約束（前処理はグラフに載せない）

グラフ入力は **正規化済みの** `pixel_values f32 [1, 3, S, S]`。画像の decode / resize
（`(S, S)` へ bilinear）/ rescale（1/255）/ normalize は karume 側の責務で、定数の正本は
同梱 `handler.py` の `ImagePreprocessor`（ImageNet 統計 = {@link IMAGENET_MEAN} /
{@link IMAGENET_STD}）。**SigLIP2（mean = std = 0.5）とは別の統計**なので、共有せずここに
持つ。

## golden の 2 群（合成画像 + 実画像）— どちらも残す

既定の 4 ケースは**合成画像**（{@link build_cases}）。`disc` は暗い背景に明るい円を置いた
**顕著物体**で、{@link _sanity} が「円内の logit 平均 > 円外の logit 平均」という**順序**を
見る（閾値は置かない）— セグメンテーションとして死んだ出力（一様・入力非依存）はここで落ちる。

`--real-images` を付けると、それに加えて `outputs/demo/` の**実画像** 4 枚
（{@link REAL_CASES}）を通した golden も書く:

    uv run --group birefnet python -m birefnet.export --real-images

こちらの主張は「TS 前処理 + karume 推論」対「Python 前処理 + torch 推論」の突合で、
Python 側は同梱 `handler.py` の `ImagePreprocessor` の逐語（{@link build_real_cases}）。
判別も実画像側が持つ — **人物が写っている 2 枚の前景比が、写っていない 2 枚のどちらよりも
大きい**（{@link _real_sanity}・順序だけを見る）。合成画像の `disc` が幾何、実画像が意味の
判別で、片方が他方を代替しない。

NOTE: 実画像は 1024×1024 なので、1024² 系列に対しては resize が恒等になる（前処理のうち
この門に掛かるのは正規化と、その後の推論の鎖）。resize そのもののパリティは
`packages/models/tests/image_preprocess_test.ts` が受け持つ。

MUST: 実画像 golden には**元画像の sha256** を `__metadata__` に載せる。画像は
`examples/anima/eval-images.ts` でいつでも焼き直せる生成物なので、焼き直したのに golden を
採り直していない環境では、突合ではなく**入力が違う**（そして値は一見それらしく出る）。

NOTE: `--real-images` も同じ `birefnet` グループで回る（SigLIP2 が `siglip2-preprocess` を
分けているのは、あちらの export が torchvision を要らないため — こちらは `deform_conv2d` の
ために torchvision が基本依存で、前処理に足すのは PNG を開く pillow だけ）。

## 出力レイアウト

    outputs/series/<系列名>/model.safetensors     重み・定数 + __metadata__
    outputs/series/<系列名>/io.<case>.safetensors 入力と torch CPU 期待出力

io のテンソルキー規約は tiny golden / DeBERTa / SigLIP2 と同じ
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
from karume.convert import normalize_boundary_tensor
from karume.ir import IrGraph
from karume.pipeline import export_to_file

from . import patch

#: 実重みの親（`inputs/birefnet/<名前>/` に HF の 7 ファイルを展開した先）。
MODELS_ROOT = INPUTS_ROOT / "birefnet"

#: 実画像の置き場（`rm -rf` で安全に消せる席 — docs/assets-layout.md）。
DEMO_ROOT = OUTPUTS_ROOT / "demo"

#: 既定のモデル（`--model-dir` 未指定のとき）。
DEFAULT_MODEL_DIR = MODELS_ROOT / "BiRefNet_HR"

#: 既定の解像度（モジュール docstring の「モデル軸と解像度軸」）。
DEFAULT_RESOLUTION = 1024

#: 解像度の刻み（S%32 = 本体側の PatchMerging / S%64 = 半解像度枝の同じ要求）。
RESOLUTION_MULTIPLE = 64

MODEL_FILE = "model.safetensors"
IO_PREFIX = "io."
IO_SUFFIX = ".safetensors"
INPUT_PREFIX = "input."
OUTPUT_PREFIX = "output."

#: グラフ入力の名前（= {@link MatteLogits.forward} の引数名）。export 後に突合してずれて
#: いたら止める — 位置で渡す以上、黙ってずれると golden の入力だけが入れ替わる。
INPUT_NAME = "pixel_values"

#: 前処理の正規化定数（正本 = 同梱 `handler.py` の `ImagePreprocessor`）。
IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)

#: 合成画像の乱数（`noise` ケース）。グローバル seed に依存しない。
SEED = 20260813

#: `disc` ケースの円の半径（画像の短辺を 1 とした比）。
DISC_RADIUS = 0.3

#: `disc` ケースの前景 / 背景色（RGB・値域 `[0, 1]`）。**3 チャネルが別々の値**なので、
#: チャネルの取り違え（前処理の統計の順序違い）が値に出る。
DISC_FOREGROUND: tuple[float, float, float] = (0.92, 0.78, 0.35)
DISC_BACKGROUND: tuple[float, float, float] = (0.12, 0.16, 0.24)

#: `checker` ケースの市松の 1 マス（画素）。Swin の窓（12）とも patch（4）とも割り切れない
#: 幅にして、格子と窓が同位相になる偶然を避ける。
CHECKER_CELL = 40

#: {@link _sanity} が判別に使うケース名（顕著物体を置いた 1 枚）。
DISC_CASE = "disc"

#: 実画像ケース = (ケース名, `outputs/demo/` のファイル名, 内容)。**画像の正本は
#: `examples/anima/eval-images.ts`**（プロンプト / seed / 解像度を持つのはあちらで、
#: `deno task demo:eval-images` でいつでも同じ 4 枚が焼き直せる）。ケース名は SigLIP2 の
#: `REAL_CASES` と同じ綴り（同じ 4 枚を別の系列ディレクトリで読むだけ）。
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

#: 実画像の判別で見る 2 群（顕著物体 = 人物が写っている 2 枚 / 写っていない 2 枚）。
#: {@link _real_sanity} が**前景比の順序**を見る。TS 側の
#: `packages/models/tests/e2e_birefnet_real_test.ts` が同じ群を実 GPU 出力に掛ける。
REAL_PERSON_CASES = ("photo-portrait", "photo-street")
REAL_SCENE_CASES = ("photo-landscape", "photo-corridor")

#: 実画像 golden の `__metadata__` の欄（元画像の同定 — モジュール docstring の MUST）。
SOURCE_IMAGE_KEY = "source_image"
SOURCE_SHA256_KEY = "source_sha256"


def default_out_dir(model_dir: Path, resolution: int) -> Path:
    """生成物の既定の置き場（`outputs/series/<モデル名>-<解像度>/`）。

    モデル名と解像度の**両方**を系列の綴りへ焼く（モジュール docstring の「モデル軸と
    解像度軸」）。HF の綴りは `BiRefNet_HR` だが、系列名は既存の綴り（小文字ハイフン）へ
    倒す — `outputs/series/` は全系列が同じ流儀で並んでいる。
    """
    return SERIES_ROOT / f"{model_dir.name.lower().replace('_', '-')}-{resolution}"


class MatteLogits(nn.Module):
    """最終段のマット（sigmoid 前の logit `[B, 1, S, S]`）1 本だけを返す export 用ラッパ。

    MUST: 要素数が 1 でなければ落とす（モジュール docstring の「何をグラフに載せるか」）—
    `self.training` が立ったモデルは multi-scale supervision の中間予測を先頭に足すので、
    黙って `[-1]` を採ると「学習モードのグラフを推論用として書き出した」ことに気づけない。
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        predictions = self.model(pixel_values)
        if len(predictions) != 1:
            raise ValueError(
                f"forward が {len(predictions)} 本返した（eval の BiRefNet は 1 本）—"
                " 学習モードのグラフは書き出さない"
            )
        return predictions[-1]


def normalize_image(image: torch.Tensor) -> torch.Tensor:
    """`[0, 1]` の `[1, 3, S, S]` を ImageNet 統計で正規化する（handler.py の逐語）。"""
    mean = torch.tensor(IMAGENET_MEAN).reshape(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).reshape(1, 3, 1, 1)
    return (image - mean) / std


def disc_mask(size: int) -> torch.Tensor:
    """`disc` ケースの円内（`[S, S]` の bool）。{@link _sanity} が同じ式で引く。"""
    axis = (torch.arange(size, dtype=torch.float32) + 0.5) / size * 2.0 - 1.0
    squared = axis.reshape(size, 1) ** 2 + axis.reshape(1, size) ** 2
    return squared <= DISC_RADIUS**2


def _disc_image(size: int) -> torch.Tensor:
    """暗い背景に明るい円（顕著物体）を置いた `[1, 3, S, S]`。"""
    foreground = torch.tensor(DISC_FOREGROUND).reshape(3, 1, 1)
    background = torch.tensor(DISC_BACKGROUND).reshape(3, 1, 1)
    return torch.where(disc_mask(size), foreground, background).unsqueeze(0)


def _ramp_image(size: int) -> torch.Tensor:
    """左上から右下への対角ランプ（チャネルごとに位相をずらす）。"""
    axis = torch.linspace(0.0, 1.0, size)
    plane = (axis.reshape(1, size) + axis.reshape(size, 1)) / 2.0
    shifts = torch.tensor([0.0, 0.15, 0.3]).reshape(3, 1, 1)
    return ((plane.unsqueeze(0) + shifts) % 1.0).unsqueeze(0)


def _checker_image(size: int) -> torch.Tensor:
    """市松（{@link CHECKER_CELL} 画素角）。チャネルごとに明暗を反転させる。"""
    index = torch.arange(size) // CHECKER_CELL
    parity = ((index.reshape(1, size) + index.reshape(size, 1)) % 2).to(torch.float32)
    channels = torch.stack([parity, 1.0 - parity, parity * 0.5 + 0.25])
    return channels.unsqueeze(0)


def build_cases(resolution: int) -> tuple[tuple[str, torch.Tensor], ...]:
    """golden 4 ケースの `(名前, pixel_values)`（**正規化済み**の合成画像）。

    実画像を使わないのは、前処理（resize + ImageNet 正規化）がまだ karume 側に無いため。
    {@link DISC_CASE} だけは顕著物体として意味を持たせてあり、{@link _sanity} の判別が
    恒真にならない土台になっている（他の 3 枚は「顕著物体が無い」側）。
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
    resolution: int, demo_root: Path
) -> tuple[tuple[str, torch.Tensor, dict[str, str]], ...]:
    """実画像 4 枚の `(名前, pixel_values, __metadata__)`（{@link REAL_CASES}）。

    前処理は同梱 `handler.py` の `ImagePreprocessor` の**逐語**（`Resize` → `ToTensor` →
    `Normalize`）。あちらを import しないのは、`handler.py` が推論と無関係な cv2 / requests を
    モジュール先頭で読むため — 写した 3 段の正本はあちらで、値の正本は
    {@link IMAGENET_MEAN} / {@link IMAGENET_STD}。

    `Resize` に PIL 画像を渡すのは handler と同じ経路（torchvision の PIL 枝 = Pillow の
    `ImagingResample`）にするため。TS 側の `resizeRgb8` も同じ台の伸ばし方だが、実画像が
    1024×1024 で 1024² 系列に対しては**恒等**なので、この門に resize の差は乗らない
    （モジュール docstring の NOTE）。

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
                f"{name} の pixel_values が {tuple(pixel_values.shape)}"
                f"（期待の {expected_shape} と違う）"
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
    """解像度が刻みに乗っていることを見る（モジュール docstring の「解像度軸」）。"""
    if resolution <= 0 or resolution % RESOLUTION_MULTIPLE:
        raise SystemExit(
            f"--resolution {resolution} は {RESOLUTION_MULTIPLE} の倍数でない"
            "（PatchMerging が各段で偶数 H/W を要求する — 半解像度枝の分まで含めて 64 刻み）"
        )


def load_model(model_dir: Path) -> nn.Module:
    """実重みを読み、差し替え版が前提にする構成を検査する（パッチ前）。

    `trust_remote_code=True` は同梱 `birefnet.py` を読むため（HF のモデルカードどおり）。
    """
    from transformers import AutoModelForImageSegmentation

    model = AutoModelForImageSegmentation.from_pretrained(
        str(model_dir), trust_remote_code=True, dtype=torch.float32
    )
    model.eval()
    # 重い forward の前に構成を見る（差し替え版の対象外なら 1 枚も焼かずに落とす）。
    patch.assert_supported(model)
    return model


def load_wrapper(model_dir: Path, resolution: int) -> MatteLogits:
    """パッチを当て、解像度依存の定数を焼いた export 可能なラッパを返す。

    MUST: 差し替えも {@link patch.prepare} も golden を採る**前**。後に当てると
    期待値だけが元の経路で計算され、グラフと食い違ったまま緑になる。
    """
    wrapper = MatteLogits(load_model(model_dir)).eval()
    patch.apply(wrapper.model)
    example = build_cases(resolution)[0][1]
    baked = patch.prepare(wrapper, example)
    expected = (1, 1, resolution, resolution)
    if tuple(baked.shape) != expected:
        raise AssertionError(f"パッチ後の出力が {tuple(baked.shape)}（期待は {expected}）")
    return wrapper


def _write_io(
    wrapper: nn.Module,
    graph: IrGraph,
    cases: Sequence[tuple[str, torch.Tensor]],
    out_dir: Path,
    metadata: Mapping[str, Mapping[str, str]] | None = None,
) -> tuple[list[str], dict[str, torch.Tensor]]:
    """各ケースの入力と torch CPU 期待出力を `io.<case>.safetensors` へ書く。

    `metadata` はケース名 → `__metadata__`（実画像ケースだけが持つ — 元画像の同定）。

    戻り値の 2 本目は sanity 記録用の期待出力（`[1, 1, S, S]` の形のまま渡す）。
    """
    if len(graph.outputs) != 1:
        raise AssertionError(f"IR 出力が {len(graph.outputs)} 本（最終段のマットは 1 本）")
    written: list[str] = []
    mattes: dict[str, torch.Tensor] = {}
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
        mattes[name] = output.detach()
    return written, mattes


def _sanity(mattes: dict[str, torch.Tensor]) -> dict[str, Any]:
    """顕著物体（{@link DISC_CASE} の円）を分離できていることを**順序**で見る。

    MUST: 恒真な sanity にしない。値域の検査（logit なので範囲が無い）も「同じ入力どうしが
    一致する」も何も検証しない。円内平均と円外平均の順序で見れば、出力が一様に潰れた場合も、
    入力に依存しなくなった場合も落ちる。

    ケースどうしが**互いに違う**ことも見る（同じ出力が 4 本並ぶのは、入力が届いていない形）。
    """
    names = sorted(mattes)
    for index, name in enumerate(names):
        for other in names[index + 1 :]:
            if torch.equal(mattes[name], mattes[other]):
                raise AssertionError(f"{name} と {other} の出力が同一（入力が効いていない）")

    matte = mattes[DISC_CASE].reshape(mattes[DISC_CASE].shape[-2], -1)
    inside = disc_mask(matte.shape[0])
    if matte.shape != inside.shape:
        raise AssertionError(f"{DISC_CASE} の出力 {tuple(matte.shape)} が円の形と違う")
    foreground = float(matte[inside].mean())
    background = float(matte[~inside].mean())
    if foreground <= background:
        raise AssertionError(
            f"{DISC_CASE} の円内 logit 平均 {foreground:.4f} が円外 {background:.4f} 以下 —"
            " 顕著物体を分離できていない"
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
    """前景（logit > 0 = sigmoid > 0.5）の面積比。"""
    return float((matte > 0.0).to(torch.float32).mean())


def _real_sanity(ratios: Mapping[str, float]) -> dict[str, float]:
    """実画像の判別（顕著物体のある 2 枚 > 無い 2 枚）を**順序**で見る。

    MUST: 閾値を置かず順序で見る（合成画像側と同じ流儀）。人物 2 枚は構図（アップ / 全身）も
    背景（桜並木 / 街並み）も違うので、この順序が成り立つのは「顕著物体を捉えている」場合
    だけになる。一様に潰れた出力（全前景 / 全背景）も入力非依存の出力も、両群の前景比が
    並ぶので落ちる。
    """
    worst_person = min(ratios[name] for name in REAL_PERSON_CASES)
    best_scene = max(ratios[name] for name in REAL_SCENE_CASES)
    if worst_person <= best_scene:
        raise AssertionError(
            f"実画像の前景比の順序が逆: 人物側の最小 {worst_person:.4f}"
            f" <= 風景側の最大 {best_scene:.4f}"
        )
    return {name: round(ratios[name], 4) for name in (*REAL_PERSON_CASES, *REAL_SCENE_CASES)}


def export_series(
    model_dir: Path,
    out_dir: Path,
    resolution: int,
    real_images: bool = False,
    demo_root: Path = DEMO_ROOT,
) -> dict[str, Any]:
    """IR コンテナと golden io を書き、要約を返す。

    `real_images` を立てると合成 4 ケースに実画像 4 ケースを**足す**（置き換えない —
    モジュール docstring の「golden の 2 群」）。
    """
    assert_resolution(resolution)
    wrapper = load_wrapper(model_dir, resolution)
    synthetic = build_cases(resolution)
    # MUST: 実画像は emit より先に組む（画像が欠けているなら、964MB を書き切ってから落とすの
    # ではなくここで止める）。
    real = build_real_cases(resolution, demo_root) if real_images else ()
    cases = [*synthetic, *((name, pixel_values) for name, pixel_values, _md in real)]
    metadata = {name: md for name, _pixel_values, md in real}
    out_dir.mkdir(parents=True, exist_ok=True)

    _, example = synthetic[0]
    # 動的軸は無い（解像度は系列ごとに固定 — モジュール docstring）。
    graph = export_to_file(wrapper, (example,), out_dir / MODEL_FILE, symbol_names=())
    declared = tuple(item.name for item in graph.inputs)
    if declared != (INPUT_NAME,):
        raise AssertionError(f"グラフ入力の並びが {declared} で、期待の {(INPUT_NAME,)} と違う")
    written, mattes = _write_io(wrapper, graph, cases, out_dir, metadata)
    return {
        "dir": str(out_dir),
        "resolution": resolution,
        "nodes": len(graph.nodes),
        "outputs": len(graph.outputs),
        "initializers": len(graph.initializers),
        "model_bytes": (out_dir / MODEL_FILE).stat().st_size,
        "ops": sorted(graph.required_ops),
        "symbols": list(graph.symbols),
        "io": written,
        "input_shape": list(example.shape),
        "real_images": {name: md[SOURCE_IMAGE_KEY] for name, md in metadata.items()},
        "sanity": _sanity(mattes),
    }


def _diff_entry(
    stage: str,
    claim: str,
    got: dict[str, torch.Tensor],
    expected: dict[str, torch.Tensor],
) -> dict[str, Any]:
    """1 段分の同値レポート（ケース名 → maxdiff と、全ケースのビット一致）。

    `bit_exact` は「差 0」より強い主張（`0.0 == -0.0` は差 0 だがビットは違う）で、並べ替え
    だけの書き換えのように**ビット一致が主張の中身**である段で意味を持つ。
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
    """パッチ前 eager との同値を **2 点**で実測する（`siglip2.export.py --verify` と同じ形）。

    1. 並べ替えだけの書き換え（窓 / qkv / roll / pad / PatchMerging / image2patches / 窓
       マスク）→ **bit_exact が主張の中身**。演算列が 1 対 1 で対応するので差は 0 でなければ
       ならず、外れたらここで落とす。
    2. `BatchNorm2d` → per-channel affine と `AdaptiveAvgPool2d` → 2 段 sum まで →
       **maxdiff を報告**。ATen 側が積和を FMA で畳む / 2 軸を同時に縮約するぶんだけ最下位
       ビットが動く（`birefnet.patch` の docstring ⑦⑧）。

    MUST: 順序は「**全ケースの参照値を確定** → 1 → 2」。パッチはクラス属性のプロセス全域
    差し替えなので、段ごとに参照を採り直すと 2 段目の参照がパッチ後の値になる（恒真化）。
    """
    assert_resolution(resolution)
    if patch.patches_applied():
        raise SystemExit(
            "パッチ適用済みのプロセスでは参照を採れない（同値検証が差 0 で恒真化する）"
        )
    wrapper = MatteLogits(load_model(model_dir)).eval()
    cases = build_cases(resolution)

    def mattes() -> dict[str, torch.Tensor]:
        with torch.no_grad():
            return {name: wrapper(pixel_values) for name, pixel_values in cases}

    reference = mattes()

    patch.apply_layout_patches(wrapper.model)
    # 窓マスクは解像度依存の定数なので、パッチ後の 1 回目は prepare 経由で焼く。
    patch.prepare(wrapper, cases[0][1])
    layout = _diff_entry("layout", "bit-exact", mattes(), reference)
    if not layout["bit_exact"]:
        raise AssertionError(
            f"並べ替えだけの書き換えがビット同一でない: maxdiff={layout['maxdiff']}"
        )
    patch.apply_module_patches(wrapper.model)
    return [layout, _diff_entry("modules", "max-abs-diff", mattes(), reference)]


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument(
        "--resolution",
        type=int,
        default=DEFAULT_RESOLUTION,
        help=f"入力の一辺（{RESOLUTION_MULTIPLE} の倍数・既定 {DEFAULT_RESOLUTION}）。"
        "解像度ごとに別のグラフ（焼いた定数が変わる）なので系列も別になる",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="出力先（既定は outputs/series/<モデル名>-<解像度>/）",
    )
    parser.add_argument(
        "--real-images",
        action="store_true",
        help=f"`{DEMO_ROOT.name}/` の実画像 4 枚を通した golden も書く（合成 4 ケースに"
        "足す）。前処理は同梱 handler.py の逐語で、元画像の sha256 を __metadata__ に載せる",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="パッチ前 eager との同値を 2 点で実測する（emit はしない — 同一プロセスでは"
        "クラス属性の差し替えが参照を汚染するので併用できない）",
    )
    args = parser.parse_args(argv)
    if args.verify:
        for entry in verify_patches(args.model_dir, args.resolution):
            print(
                f"{entry['stage']} ({entry['claim']}): bit_exact={entry['bit_exact']}"
                f" maxdiff={entry['maxdiff']}"
            )
        return
    out_dir = args.out if args.out is not None else default_out_dir(args.model_dir, args.resolution)
    summary = export_series(args.model_dir, out_dir, args.resolution, real_images=args.real_images)
    print(json.dumps({"model_dir": str(args.model_dir), **summary}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
