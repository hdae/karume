"""BiRefNet 系の配布 recipe — 系列レイアウト・出力 path 表・quant 表・カードの選択。

ADR 0065 決定 2。汎用の組み立てエンジン（配置・共有席の畳み込み・sha256・manifest・
staging/swap・検証）は `karume.dist` が持つ。ここが持つのは **BiRefNet 系固有の事実**だけ:
どの系列ディレクトリから何を拾い、配布形のどの path へ、どの dtype ラベルで並べ、どの quant を
既定にするか。

配布するのは **マット 1 グラフだけ**（`birefnet/export.py` は最終段の logit 1 本しか出さない）。
実行に要る資産もそれ 1 本で、tokenizer も表も無い（`assets` は空）。格納 dtype は f32 / f16 の
2 系列で、quant 席も同じ 2 つ（ADR 0113 — 既定は f32 のまま）。系列は dtype ごとに別ディレクトリ
（`birefnet-hr-1024` / `birefnet-hr-1024-f16`）で、配布形では `matte/model.{dtype}.krm` に並ぶ。

**リポは 2 つ、各リポに 2 モデル**。軸が 2 本あることがこの family の形で、混ぜると配布形が
静かに壊れる:

- **リポの軸 = checkpoint**（上流 `BiRefNet_HR` と派生 `lucida`）。派生は別リポ（ADR 0092
  決定 1）で、`--model` からは決まらない — {@link PIPELINE} / {@link LUCIDA_PIPELINE} の
  **席が固定で持つ**。
- **モデルの軸 = 解像度**（{@link BIREFNET_MODELS} = `"1024"` / `"2048"`・既定 `"1024"`）。
  解像度は窓マスクとパディング定数まで定数として焼かれた**別のグラフ**（`birefnet/export.py`
  の「解像度軸」）なので、SigLIP2 の base / so400m と同じ「1 リポ 2 モデル」の器（ADR 0092
  決定 8）へ同居させる。利用者が引くのは `defaultModel` 1 つで、寸法は `pipelineConfig` が
  モデルごとに宣言する。組み立ては `--model 1024 --model 2048 --out <リポ>` の 2 モデル
  1 周で、**最初の `--model` が `defaultModel`** になる（`karume.dist.main`）。

リポ名は導出せず {@link BIREFNET_REPO_NAMES} が持つ — `karume-lucida` は「BiRefNet 系の
1 つ」ではなく上流が名前で売っているモデルで、綴りは命名の決定であって checkpoint 名から
決まらない（SBV2 のファミリー名と同じ性質）。

公開面が **2 つの Pipeline** に割れているのはこのため（anima の公式 / 追加学習と同じ形）:
`root_files`（配布リポ直下の `LICENSE.md` / `NOTICE.md`）は Pipeline に固定で載る 1 組で、
MIT の著作権行はリポごとに違う（HR は ZhengPeng の 1 行、Lucida は fine-tune 側と上流の
2 行）。1 つに畳むと、どちらかのリポが**自分のものでない著作権を名乗る**か、上流の著作権
表示を落とす — どちらも散文としては妥当なままなので `verify_dist` も manifest 検査も
素通りし、配ってからでないと誰も気づけない。カードの帰属も同じ席から入る
（{@link render_birefnet_model_card} の `checkpoint`）— 帰属・著作権行・系列 path が
**1 つの checkpoint 名から**決まるので、席を跨いだ取り違えは綴りとして作れない。

`pipelineConfig` の数の出どころは **2 つとも独立**:

- resize 先（`imageWidth` / `imageHeight`）は**焼かれたグラフの入力宣言**から導く（写経しない）。
- 正規化定数は上流に機械可読な出どころが**無い** — SigLIP2 の `preprocessor_config.json` に
  当たるものが BiRefNet 系には無く、事実は同梱 `handler.py` の `ImagePreprocessor`（と
  モデルカードの利用例）の中にしか書かれていない。したがってここが宣言として持ち
  （{@link BIREFNET_IMAGE_MEAN}）、台本側の写し（`birefnet.export.IMAGENET_MEAN`）との一致は
  pytest が毎回突き合わせる（2 表が独立に動く形にはしない）。

公開面は {@link PIPELINE}（`--pipeline birefnet`）と {@link LUCIDA_PIPELINE}
（`--pipeline lucida`）の 2 つ（`karume.dist.Pipeline`）— リポの dist ドライバ
（`tools/export-recipes/dist.py`）がこれを core の PIPELINES へ合成する。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, NamedTuple

from _shared.container_read import read_provenance
from _shared.licenses import mit_license
from _shared.upstream import assert_upstream_provenance
from karume.dist import (
    Artifact,
    DistError,
    ModelPlan,
    Pipeline,
    WeightFiles,
    assert_model_name,
    assert_storage,
    assert_storage_absent,
    complete_quant_weights,
    graph_inputs,
    ir_graph,
)

from .card import (
    BIREFNET_LICENSE,
    BIREFNET_UPSTREAM,
    birefnet_storage_lines,
    render_birefnet_model_card,
)

#: パイプライン契約（ADR 0041 §2 — モデル単位）。TS 側の受理集合は
#: `BIREFNET_PIPELINE_NAME` / `BIREFNET_PIPELINE_MAJOR`。
BIREFNET_PIPELINE = "birefnet/1"

#: 配るモデル名の受理集合 = **入力解像度**（綴りの正本はこの 1 つ）。系列は解像度ごとに別
#: なので、モデル名がそのまま「どの系列を配るか」を指す。
BIREFNET_MODELS: tuple[str, ...] = ("1024", "2048")

#: 既定のモデル名（`--model` を省いた組み立てが入れるモデル）。
BIREFNET_DEFAULT_MODEL = "1024"

#: checkpoint の綴り（**pipeline 席が持つ** — `--model` の軸ではない）。帰属表
#: （`birefnet.card.BIREFNET_CHECKPOINTS`）のキーでもあり、系列 path・リポ名・著作権行・
#: カードの帰属がこの 1 つから決まる。
BIREFNET_HR_CHECKPOINT = "hr"
BIREFNET_LUCIDA_CHECKPOINT = "lucida"

#: 実重みの親（`hf download <リポ> --local-dir inputs/birefnet/<名前>` の展開先 —
#: `birefnet.export.MODELS_ROOT` と同じ場所）。系列名はこのディレクトリ名から決まる。
BIREFNET_INPUTS_DIRNAME = "birefnet"

#: checkpoint → 配布リポ名（`karume-` prefix はリポ名裁定 2026-08-09）。**導出しない** —
#: 節の冒頭に書いたとおり、綴りは命名の決定。
BIREFNET_REPO_NAMES: Mapping[str, str] = {
    "hr": "karume-birefnet-hr",
    "lucida": "karume-lucida",
}

#: 唯一の役割名（manifest の weights キー・TS 側 `pipeline.ts` の `MATTE`）。
BIREFNET_ROLE = "matte"

#: グラフ入力の名前（`birefnet.export.INPUT_NAME`）。
BIREFNET_INPUT = "pixel_values"

#: 入力のチャネル数（RGB）と、出力（マット）のチャネル数。
BIREFNET_CHANNELS = 3
BIREFNET_MATTE_CHANNELS = 1

#: TS 側が実装している唯一の補間。上流（`handler.py` / モデルカードの利用例）はどちらも
#: `torchvision.transforms.Resize((S, S))` を既定の補間で通す = bilinear。
BIREFNET_INTERPOLATION = "bilinear"

#: 前処理の正規化定数（節の冒頭のとおり、上流に機械可読な出どころが無いのでここが持つ）。
#: 正本は同梱 `handler.py` の `ImagePreprocessor` = ImageNet 統計。
BIREFNET_IMAGE_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
BIREFNET_IMAGE_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)

#: 配る格納 dtype（段 1 = f32 / f16 — ADR 0113）。**系列 root・出力 path・格納の要求・weights 宣言・
#: 配置表の 5 つが同じ 1 表から組まれる**（別々に持つと、席を足した日に片方だけ更新される —
#: irodori の `IRODORI_DTYPE_ROLES` と同じ規律）。quant 席（{@link BIREFNET_QUANT_SEATS}）は表示欄を
#: 手で持つ別表なので、鍵の一致は pytest が突き合わせる。i8 は BiRefNet 固有の i8 計画と
#: 一緒に段 2 で足す（`birefnet.export.WEIGHT_DTYPES` と揃える — 突合は pytest）。
BIREFNET_WEIGHT_DTYPES: tuple[str, ...] = ("f32", "f16")

#: 圧縮していない系列の dtype（系列名に接尾が付かない唯一の席）。
BIREFNET_PLAIN_DTYPE = "f32"


def birefnet_role(dtype: str) -> str:
    """配置の役割名（`matte_f16` — 配置表・出力 path・格納 dtype 要求が共有する 1 語）。"""
    return f"{BIREFNET_ROLE}_{dtype}"


#: 出力の相対 path（**モデルサブツリー内**）— 配置表と manifest が共有する 1 箇所。
#: 格納 dtype をファイル名に出すのは Anima / SBV2 / Irodori と同じ形（`model.f16.krm`）で、
#: 1 つのディレクトリに系列 2 本が並んでも取り違えようがない綴りにするため。
BIREFNET_OUTPUT_PATHS: Mapping[str, str] = {
    birefnet_role(dtype): f"{BIREFNET_ROLE}/model.{dtype}.krm" for dtype in BIREFNET_WEIGHT_DTYPES
}

#: 格納 dtype の要求（Anima / SBV2 / Irodori / SigLIP2 と同じ根拠 — 素の資産が組み立て・
#: ロード・実行を全て通って参照一致の門まで沈黙した実測事故）。f16 系列は適格な重みスロット
#: だけが f16 で、bias / norm / グラフ定数 / deform の重みは f32 のまま（`birefnet/export.py`）
#: なので「その語彙を含む」を要求する。逆向き（f16 席に f32 系列）はここが要求の不在で落とす。
BIREFNET_STORAGE_REQUIREMENTS: Mapping[str, str] = {
    birefnet_role(dtype): dtype for dtype in BIREFNET_WEIGHT_DTYPES
}

#: 各役割の束縛表に**あってはならない**格納の語彙（{@link assert_storage_absent}）。
#: {@link BIREFNET_STORAGE_REQUIREMENTS} は「要求 dtype が在るか」の片方向検査で、**圧縮系列も
#: 適格外の重み**（bias / norm / グラフ定数・i8 の per-channel scale・i4 の group scale）を f32 で
#: 持つため「f32 を含む」は f16 / i8 / i4 の資産でも真になる — f32 席へ圧縮系列を挿し込む
#: 取り違えが存在検査だけでは素通りする。
#:
#: 禁止表が閉じるのは**系列 root の取り違え**（`--series` が別の木を指す / 別 family の圧縮系列を
#: 手で置く）で、台本が対応する dtype とは無関係に起こる。系列 root の取り違えは数値の門では
#: 原理的に検出できない（ADR 0027 / 0029）ので、ここが唯一の検出器。
#:
#: MUST: 禁止は**役割ごとに集合**で持ち、格納検査が見る語彙（codec 台帳
#: `karume.container.CODEC_LEDGER` の layout — `karume.dist.storage_dtypes`）の f32 / i32 以外を
#: **全部**名指しする — 1 つでも抜けると、抜けた格納形だけが黙って素通りする
#: （anima / irodori / sbv2 と同じ規律）。I32 を載せないのは、i32 が圧縮ではなく素の格納
#: （`karume.emit` の plain 側）だから — 実際この family の系列は i32 の添字表を 1 本持つので
#: 束縛表は f32 + i32（2026-08-30 の実測）で、i32 を禁じると既存の配布物が赤になる。
#:
#: f16 席は禁止表を持たない: f32 系列は要求検査（f16 が無い）で落ち、混成の圧縮系列はまだ
#: 無い（irodori の i8 席が i4 を禁じるのは i4 系列が i8 を含む混成だから — 段 2 の i8 系列は
#: 単一の圧縮 dtype の予定で、f16 席に挿せば要求検査で落ちる）。
BIREFNET_STORAGE_FORBIDDEN: Mapping[str, tuple[str, ...]] = {
    birefnet_role(BIREFNET_PLAIN_DTYPE): ("f16", "bf16", "i8", "i4", "i2")
}

#: weights の宣言（dtype ラベル → 役割名）。マット 1 本が f32 / f16 の 2 席を持つので、quant 表が
#: 名指しする（{@link complete_quant_weights} は dtype が 1 つのときしか埋めない）。
BIREFNET_WEIGHTS: Mapping[str, Mapping[str, WeightFiles]] = {
    BIREFNET_ROLE: {dtype: WeightFiles(birefnet_role(dtype)) for dtype in BIREFNET_WEIGHT_DTYPES}
}

#: assets の宣言。**空**（実行に要るのはグラフ 1 本だけ）。
BIREFNET_ASSETS: Mapping[str, str] = {}


class BirefnetQuantSeat(NamedTuple):
    """quant 席 1 つ（格納 dtype と、選択 UI へ出す表示欄 — ADR 0075）。"""

    dtype: str
    #: 選択 UI に出す短い表示名（64 字上限）。
    label: str
    #: 同・1 行の説明（200 字上限）。既定であることは書かない（`defaultQuant` が指している）。
    #: checkpoint に依らない文面にする（表は HR / Lucida の 2 リポで共有 — 値を変えるかどうかは
    #: checkpoint 次第なので、その事実は NOTICE とカードの帰属節が持つ）。
    description: str


#: quant 席の綴り → {@link BirefnetQuantSeat}。実行形ノブ（`session`）は持たない — 計算は
#: どちらの席も f32 で、違うのは重みの格納だけ。
BIREFNET_QUANT_SEATS: Mapping[str, BirefnetQuantSeat] = {
    "f32": BirefnetQuantSeat(
        "f32",
        label="Full precision (f32)",
        description="Every weight in f32 storage — the largest download and the most resident"
        " GPU memory.",
    ),
    "f16": BirefnetQuantSeat(
        "f16",
        label="Half-size weights (f16)",
        description="The linear, convolution and embedding (relative-position) weights stored as"
        " f16 and computed in f32 — a smaller download and less resident GPU memory.",
    ),
}

BIREFNET_QUANTS: Mapping[str, Any] = {
    name: {
        "weights": {BIREFNET_ROLE: seat.dtype},
        "session": {},
        "label": seat.label,
        "description": seat.description,
    }
    for name, seat in BIREFNET_QUANT_SEATS.items()
}

#: 既定は f32 のまま（ADR 0113 — 既定席の変更は配布の意味の変更で、段 2 の品質実測と目視を
#: 経てから裁定する。HR の f16 は無損失だが、公開済みの既定を黙って動かさない）。
BIREFNET_DEFAULT_QUANT = "f32"


def birefnet_checkpoint(checkpoint: str) -> str:
    """checkpoint 名 → 上流チェックポイントのディレクトリ名（= 上流の HF リポ名の末尾）。

    `birefnet/export.py` は `--model-dir` のディレクトリ名を系列名にし、そのディレクトリ名は
    `hf download <リポ>` の展開先なので、綴りの事実は「この checkpoint がどの上流リポか」
    1 つしかない。帰属表（`birefnet.card.BIREFNET_UPSTREAM`）から導いて、ここに 2 つ目の表を
    持たない（SigLIP2 の {@link siglip2.distribution.siglip2_checkpoint} と同じ規律）。
    """
    repo = BIREFNET_UPSTREAM.get(checkpoint)
    if repo is None:
        raise DistError(
            f"BiRefNet 系の checkpoint '{checkpoint}' は知らない"
            f"（既知: {' / '.join(sorted(BIREFNET_UPSTREAM))}）"
        )
    return repo.split("/", 1)[1]


def birefnet_resolution(model: str) -> int:
    """モデル名 → 焼かれているべき入力の一辺（**モデル名は解像度そのもの**）。

    受理集合を通るのはここ 1 か所で、系列名の導出と `pipelineConfig` の突合の両方がこの数を
    使う。2 つの経路が別々に綴りを解釈すると、「系列は 2048² なのに宣言は 1024²」という
    形が作れてしまう。
    """
    if model not in BIREFNET_MODELS:
        raise DistError(
            f"BiRefNet 系のモデル '{model}' は配らない（配るのは: {' / '.join(BIREFNET_MODELS)}）"
            " — モデル名は配る入力解像度そのもので、系列も別"
        )
    return int(model)


def birefnet_series_name(checkpoint: str, model: str, dtype: str = BIREFNET_PLAIN_DTYPE) -> str:
    """checkpoint・解像度・格納 dtype → 系列ディレクトリ名。

    `birefnet.export.default_out_dir` と同じ式 — f32 は接尾なし（既存系列の綴りのまま）・他は
    `-<dtype>`。
    """
    if dtype not in BIREFNET_WEIGHT_DTYPES:
        raise DistError(
            f"BiRefNet 系の格納 dtype '{dtype}' は配らない"
            f"（配るのは: {' / '.join(BIREFNET_WEIGHT_DTYPES)}）"
        )
    name = birefnet_checkpoint(checkpoint).lower().replace("_", "-")
    suffix = "" if dtype == BIREFNET_PLAIN_DTYPE else f"-{dtype}"
    return f"{name}-{birefnet_resolution(model)}{suffix}"


def birefnet_repo_name(checkpoint: str) -> str:
    """checkpoint の配布リポ名（{@link BIREFNET_REPO_NAMES}）。

    `Pipeline.repo_name` はモデル名を受ける席だが、リポを決めるのは checkpoint なので
    {@link PIPELINE} / {@link LUCIDA_PIPELINE} が自分の checkpoint を束ねてここへ渡す
    （解像度 2 つを 1 周で組むときは `--out` が要る — `dist.default_out_dir`）。
    """
    name = BIREFNET_REPO_NAMES.get(checkpoint)
    if name is None:
        raise DistError(
            f"BiRefNet 系の checkpoint '{checkpoint}' のリポ名が無い"
            f"（既知: {' / '.join(sorted(BIREFNET_REPO_NAMES))}）"
        )
    return name


@dataclass(frozen=True)
class BirefnetSources:
    """組み立ての入力。格納 dtype ごとの系列（どれもグラフ 1 本）だけ。

    SigLIP2 と違って実重みの置き場を持たないのは、前処理定数の出どころになる機械可読な
    ファイルが上流に無いから（節の冒頭）。**どの重みのどの解像度かを言えるのは系列 path
    だけ**なので、系列名の導出（{@link birefnet_series_name}）が帰属の唯一の紐づけになる。
    """

    #: 格納 dtype → 系列 root（{@link BIREFNET_WEIGHT_DTYPES} の全部 — 1 つでも欠けたら組まない）。
    series_by_dtype: Mapping[str, Path]


def birefnet_sources(
    series_dir: Path, checkpoint: str, model: str = BIREFNET_DEFAULT_MODEL
) -> BirefnetSources:
    """系列の親ディレクトリ（`outputs/series/`）と checkpoint・解像度から入力を引く。"""
    return BirefnetSources(
        series_by_dtype={
            dtype: series_dir / birefnet_series_name(checkpoint, model, dtype)
            for dtype in BIREFNET_WEIGHT_DTYPES
        }
    )


def birefnet_placements(sources: BirefnetSources) -> dict[str, Path]:
    """役割名 → 出所のファイル。出力の path は {@link BIREFNET_OUTPUT_PATHS} が持つ。

    この表に無いものは出力へ入らない（系列に並ぶ `io.*.safetensors` / `quality.json` はこれで
    落ちる）。
    """
    return {
        birefnet_role(dtype): sources.series_by_dtype[dtype] / "model.krm"
        for dtype in BIREFNET_WEIGHT_DTYPES
    }


def birefnet_pipeline_config(graph: Mapping[str, Any], path: Path, model: str) -> dict[str, Any]:
    """`pipelineConfig`（TS 側スキーマの 5 欄）を焼かれたグラフと正規化定数から組む。

    resize 先はグラフの入力宣言そのもの — 前処理が別の寸法へ伸ばすと、値が静かに崩れたまま
    shape だけ合う。入力の名前・階数・batch・チャネル数もここで見る（後段の
    {@link assert_birefnet_graph} が出力側を見る）。

    MUST: 読んだ解像度を**モデル名**と突き合わせる。モデル名は manifest のキーであり利用者が
    `model: "2048"` と綴る値そのものなので、系列を 1 つ取り違えたまま組むと「2048 と名乗る
    1024 のグラフ」が出る — 実行も検証も通り、`pipelineConfig` の宣言とも噛み合ったままで、
    食い違うのは名前だけになる。
    """
    inputs = graph_inputs(graph, path)
    if tuple(inputs) != (BIREFNET_INPUT,):
        raise DistError(
            f"{path} のグラフ入力が {list(inputs)} で、期待の {[BIREFNET_INPUT]} と違う"
            " — 実行側は名前で束ねるので、綴りが変われば束ねられない"
        )
    shape = inputs[BIREFNET_INPUT]
    if len(shape) != 4 or shape[0] != 1 or shape[1] != BIREFNET_CHANNELS:
        raise DistError(
            f"{path} の入力 '{BIREFNET_INPUT}' が {shape!r}"
            f" — 期待は [1, {BIREFNET_CHANNELS}, H, W]（batch もチャネル数も静的）"
        )
    height, width = shape[2], shape[3]
    for axis, value in (("H", height), ("W", width)):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise DistError(
                f"{path} の入力 '{BIREFNET_INPUT}' の {axis} が正の整数でない（{value!r}）"
            )
    resolution = birefnet_resolution(model)
    if (height, width) != (resolution, resolution):
        raise DistError(
            f"{path} は {height}×{width} で焼かれている — モデル '{model}' が名乗る"
            f" {resolution}² と食い違う（解像度ごとに別のグラフ・別の系列）"
        )
    return {
        "imageWidth": width,
        "imageHeight": height,
        "imageMean": list(BIREFNET_IMAGE_MEAN),
        "imageStd": list(BIREFNET_IMAGE_STD),
        "interpolation": BIREFNET_INTERPOLATION,
    }


def assert_birefnet_graph(
    graph: Mapping[str, Any], path: Path, pipeline_config: Mapping[str, Any]
) -> None:
    """出力が**マット 1 本**で、入力と同じ寸法であることを配置の前に実測する。

    MUST: 落とさない。`birefnet/export.py` のラッパは学習モードだと multi-scale supervision の
    中間予測まで返す形なので、そちら向けに焼かれたグラフは**位置で引く後段が別の値を α として
    読む**（要素数だけ見る実装なら shape も通る）。記号次元が無いことも同じ席で見る — 解像度も
    窓マスクも定数として焼かれており、動かせる軸は 1 本も無い。
    """
    symbols = graph.get("symbols")
    if not isinstance(symbols, list) or symbols:
        raise DistError(
            f"{path}: 記号次元 {symbols!r} がある — 解像度も窓マスクも定数として焼かれている"
        )
    outputs = graph.get("outputs")
    if not isinstance(outputs, list) or len(outputs) != 1:
        raise DistError(f"{path}: グラフ出力が {outputs!r} — 最終段のマット 1 本だけが要る")
    values = graph.get("values")
    value = values.get(outputs[0]) if isinstance(values, dict) else None
    shape = value.get("shape") if isinstance(value, dict) else None
    expected = [
        1,
        BIREFNET_MATTE_CHANNELS,
        pipeline_config["imageHeight"],
        pipeline_config["imageWidth"],
    ]
    if shape != expected:
        raise DistError(
            f"{path}: グラフ出力 '{outputs[0]}' の形が {shape!r}、期待は {expected}"
            " — マットは入力と同じ寸法の 1 チャネル"
        )


def birefnet_plan(
    sources: BirefnetSources, checkpoint: str, model: str = BIREFNET_DEFAULT_MODEL
) -> ModelPlan:
    """BiRefNet 系 1 モデル（= 1 解像度）ぶんの計画を組む（検査と読み取りをここで全部済ませる）。"""
    assert_model_name(model)
    birefnet_resolution(model)
    # 帰属を名乗れない checkpoint は系列を読む前に落とす（リポ名の表と帰属表は独立に欠けうる）。
    birefnet_checkpoint(checkpoint)
    birefnet_repo_name(checkpoint)
    placements = birefnet_placements(sources)
    for role, source in placements.items():
        assert_storage(role, source, BIREFNET_STORAGE_REQUIREMENTS)
        assert_storage_absent(role, source, BIREFNET_STORAGE_FORBIDDEN)
    # MUST: **格納 dtype の系列を 1 本残らず**検査する（irodori と同じ規律）。f16 系列は f32 とは
    # 別プロセスの emit なので、片方だけ見ると「f32 は宣言どおりだが f16 だけ別解像度 / 別の
    # 出所」が素通りする。`pipelineConfig` は系列ごとに導いて**一致**を見る（寸法は焼かれた
    # グラフの宣言なので、1 本から導いて他へ写すと別寸法の系列が同居できてしまう）。
    configs: dict[str, dict[str, Any]] = {}
    revisions: dict[str, str | None] = {}
    for dtype in BIREFNET_WEIGHT_DTYPES:
        container = placements[birefnet_role(dtype)]
        graph = ir_graph(container)
        configs[dtype] = birefnet_pipeline_config(graph, container, model)
        assert_birefnet_graph(graph, container, configs[dtype])
        # 容器が名乗る出所を帰属表へ突き合わせる（`_shared.upstream` — depth_anything と同じ
        # 形）。revision は「名乗っていること」まで: この family の入力は系列だけで、手元の
        # checkpoint を持たない（{@link BirefnetSources}）。
        assert_upstream_provenance(container, license=BIREFNET_LICENSE, revision=None)
        revisions[dtype] = read_provenance(container).upstream_revision
    # MUST: 全 dtype の系列が**同じ上流 revision** を名乗ること。ライセンス（HR / Lucida とも
    # MIT）と寸法・前処理は checkpoint を跨いで一致するので、revision を突き合わせないと「HR の
    # f32 系列 + Lucida の f16 系列」が 1 モデルの 2 席として組み上がる（2026-09-26 レビュー A-1）。
    if len(set(revisions.values())) != 1:
        listed = ", ".join(f"{dtype} = {revision}" for dtype, revision in revisions.items())
        raise DistError(
            f"格納 dtype の系列が名乗る上流 revision が食い違う（{listed}）— 1 モデルの席は"
            " 同じ checkpoint の同じ revision から焼く（別 checkpoint の系列を掴んでいる）"
        )
    pipeline_config = configs[BIREFNET_PLAIN_DTYPE]
    for dtype, config in configs.items():
        if config != pipeline_config:
            raise DistError(
                f"格納 {dtype} の系列の pipelineConfig {config} が {BIREFNET_PLAIN_DTYPE} 系列の"
                f" {pipeline_config} と食い違う — 同じモデルの席は同じ寸法・前処理で焼く"
            )
    return ModelPlan(
        name=model,
        pipeline=BIREFNET_PIPELINE,
        artifacts={
            role: Artifact(BIREFNET_OUTPUT_PATHS[role], source=source)
            for role, source in placements.items()
        },
        weights=BIREFNET_WEIGHTS,
        assets=BIREFNET_ASSETS,
        quants=complete_quant_weights(BIREFNET_WEIGHTS, BIREFNET_QUANTS),
        default_quant=BIREFNET_DEFAULT_QUANT,
        pipeline_config=pipeline_config,
    )


def birefnet_dist_plan(series_dir: Path, model: str, checkpoint: str) -> ModelPlan:
    """`--series` の親から BiRefNet 系 1 モデル（= 1 解像度）の計画を組む（CLI のディスパッチ先）。

    MUST: `checkpoint` は**その Pipeline のリポが配る唯一の重み**で、`--model` からは決まらない
    （anima と同じ規律の置き換え）。系列 path も帰属もリポ直下の著作権表示もこの 1 つから
    決まるので、席を跨いだ取り違え — リポ直下の著作権表示と改変告知が中身と食い違ったまま
    配布形が成立する形 — は綴りとして作れない。
    """
    return birefnet_plan(birefnet_sources(series_dir, checkpoint, model), checkpoint, model)


#: MIT の著作権行（配布リポ直下の `LICENSE.md`）。上流 2 リポとも `license: mit` を名乗る
#: 一方でライセンス**原文を同梱していない**（HF のカード frontmatter だけ）ので、原文は
#: 共有テンプレート（`_shared.licenses.mit_license`）から組み、権利者の行をここが持つ。
#:
#: MUST: Lucida は 2 行（fine-tune 側と上流）。MIT の「著作権表示と許諾表示を含めること」は
#: 派生でも上流の表示を落とせないという要求で、上流カードも「the original copyright notice
#: is preserved」と自己申告している。
#:
#: NOTE: 年は上流が明示していない（どちらのリポにも `LICENSE` ファイルが無い）。ZhengPeng は
#: BiRefNet 論文の年（2024）、egeorcun は上流カード自身が現行重みへ付けている年（2026）を
#: 採った — リリース前の人手ライセンス確認（ADR 0065 決定 7）が最終判断を持つ。
BIREFNET_COPYRIGHTS: Mapping[str, tuple[str, ...]] = {
    BIREFNET_HR_CHECKPOINT: ("Copyright (c) 2024 ZhengPeng",),
    BIREFNET_LUCIDA_CHECKPOINT: (
        "Copyright (c) 2026 egeorcun",
        "Copyright (c) 2024 ZhengPeng",
    ),
}

#: 改変告知。MIT は改変告知を要求しないが、格納形を変えずコンテナだけを移した配布形である
#: ことは利用者が最初に確かめたい事実なので、`LICENSE.md` と同じ席で 1 枚出す。
#:
#: MUST: 文面は配布形の中身と対応していること — 値としては妥当な散文なので `verify_dist` も
#: manifest 検査も素通りし、配ってからでないと食い違いに気づけない。席ごとの格納の説明
#: （`{storage}`）はカードの帰属節と同じ 1 か所（`birefnet.card.birefnet_storage_lines`）から、
#: 配る席の表（{@link BIREFNET_QUANT_SEATS}）と checkpoint の格納から組む。
BIREFNET_NOTICE_TEMPLATE = """# NOTICE

This repository redistributes a modified form of `{repo}`, which is licensed under
the MIT License (see `LICENSE.md`). The following changes were made:

- The graph was re-expressed in the Karume container format (a `.krm` part sequence whose
  first part carries the graph and model descriptors, followed by the weight parts they name).
- The upstream `forward` was rewritten layout-only and **bit-exact**: windowing, the
  shifted-window roll, spatial padding and the patch merges became equivalent operations.
- Two modules were rewritten in a form that is equivalent up to floating-point rounding:
  inference-time `BatchNorm2d` became a per-channel affine, and the ASPP image-level pooling
  became a two-stage sum.
- The decoder tail's 1×1 convolution and the bilinear upsample it used to follow were swapped
  in order. Both are linear on disjoint axes, so they commute; the rewrite is equivalent up to
  floating-point rounding and removes two full-resolution intermediates.
{storage}

No retraining and no fine-tuning were performed. The original checkpoint is not distributed here.
"""


def birefnet_root_files(checkpoint: str) -> dict[str, str]:
    """配布リポ直下へ入れる法的テキスト（`karume.dist.Pipeline.root_files`）。

    MIT は「著作権表示と許諾表示を全ての複製に含める」ことだけを求めるので、`LICENSE.md` は
    共有テンプレートの本文 + そのリポの著作権行（{@link BIREFNET_COPYRIGHTS}）。本文へは
    触らない（整形した瞬間に許諾表示のコピーではなくなる）。
    """
    return {
        "LICENSE.md": mit_license(BIREFNET_COPYRIGHTS[checkpoint]),
        "NOTICE.md": BIREFNET_NOTICE_TEMPLATE.format(
            repo=BIREFNET_UPSTREAM[checkpoint],
            storage="\n".join(
                birefnet_storage_lines(
                    checkpoint, [seat.dtype for seat in BIREFNET_QUANT_SEATS.values()]
                )
            ),
        ),
    }


def _birefnet_pipeline(checkpoint: str) -> Pipeline:
    """1 checkpoint = 1 配布リポぶんの Pipeline を組む（解像度 2 つがその中に同居する）。

    checkpoint を**ここで 1 回だけ**束ねるのがこの関数の全て — 系列 path（`plan`）・リポ名・
    リポ直下の著作権行（`root_files`）・カードの帰属（`card_profiles`）の 4 つが同じ 1 つの
    綴りから入るので、どれか 1 つだけが別の checkpoint を指す形が作れない。
    """
    return Pipeline(
        default_model=BIREFNET_DEFAULT_MODEL,
        repo_name=lambda _model: birefnet_repo_name(checkpoint),
        plan=lambda series_dir, model: birefnet_dist_plan(series_dir, model, checkpoint),
        # SigLIP2 と同じ理由で選ばせる軸にしない — 帰属（上流リポ・ライセンス・学習データ）は
        # checkpoint から一意に決まる。プロファイルを分けると「Lucida を BiRefNet_HR の帰属で
        # 配る」取り違えを操作者が起こせるようになる。
        card_profiles={"birefnet": partial(render_birefnet_model_card, checkpoint=checkpoint)},
        root_files=birefnet_root_files(checkpoint),
    )


#: `--pipeline birefnet` の 1 行（配布リポ `karume-birefnet-hr`）。
#:
#: MUST: 派生（`lucida`）と**別の Pipeline**にする — 理由はモジュール doc の同段落。
PIPELINE = _birefnet_pipeline(BIREFNET_HR_CHECKPOINT)

#: `--pipeline lucida` の 1 行（配布リポ `karume-lucida` — BiRefNet_HR の第三者 fine-tune）。
#: テンプレートは同じ（構造が同一なので配布形も同型）で、違うのは中身とリポ直下の法的
#: テキストだけ。
LUCIDA_PIPELINE = _birefnet_pipeline(BIREFNET_LUCIDA_CHECKPOINT)
