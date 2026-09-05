"""BiRefNet 系の配布 recipe — 系列レイアウト・出力 path 表・quant 表・カードの選択。

ADR 0065 決定 2。汎用の組み立てエンジン（配置・共有席の畳み込み・sha256・manifest・
staging/swap・検証）は `karume.dist` が持つ。ここが持つのは **BiRefNet 系固有の事実**だけ:
どの系列ディレクトリから何を拾い、配布形のどの path へ、どの dtype ラベルで並べ、どの quant を
既定にするか。

配布するのは **マット 1 グラフだけ**（`birefnet/export.py` は最終段の logit 1 本しか出さない）。
実行に要る資産もそれ 1 本で、tokenizer も表も無い（`assets` は空）。格納 dtype は f32 の
1 系列だけなので quant 席も 1 つ — ここまでは SigLIP2 と同じ形。

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
from typing import Any

from _shared.licenses import mit_license
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

from .card import BIREFNET_UPSTREAM, render_birefnet_model_card

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

#: 出力の相対 path（**モデルサブツリー内**）— 配置表と manifest が共有する 1 箇所。
BIREFNET_OUTPUT_PATHS: Mapping[str, str] = {BIREFNET_ROLE: f"{BIREFNET_ROLE}/model.f32.safetensors"}

#: 格納 dtype の要求（Anima / SBV2 / Irodori / SigLIP2 と同じ根拠 — 素の資産が組み立て・
#: ロード・実行を全て通って参照一致の門まで沈黙した実測事故）。
BIREFNET_STORAGE_REQUIREMENTS: Mapping[str, str] = {BIREFNET_ROLE: "F32"}

#: 各役割の safetensors ヘッダに**あってはならない**格納 dtype（{@link assert_storage_absent}）。
#: {@link BIREFNET_STORAGE_REQUIREMENTS} は「要求 dtype が在るか」の片方向検査で、**圧縮系列も
#: 適格外の重み**（bias / norm / グラフ定数・i8 の per-channel scale・i4 の group scale）を F32 で
#: 持つため「F32 を含む」は f16 / i8 / i4 の資産でも真になる — f32 席へ圧縮系列を挿し込む
#: 取り違えが存在検査だけでは素通りする。
#:
#: この台本（`birefnet/export.py`）は f32 しか焼かないが、禁止表が閉じるのは**系列 root の
#: 取り違え**（`--series` が別の木を指す / 別 family の圧縮系列を手で置く）で、台本が対応する
#: dtype とは無関係に起こる。系列 root の取り違えは数値の門では原理的に検出できない
#: （ADR 0027 / 0029）ので、ここが唯一の検出器。
#:
#: MUST: 禁止は**役割ごとに集合**で持ち、書き出しうる圧縮格納（`karume.emit.WEIGHT_DTYPES` の
#: f32 以外）を**全部**名指しする — 1 つでも抜けると、抜けた格納形だけが黙って素通りする
#: （anima / irodori / sbv2 と同じ規律）。I32 を載せないのは、i32 が圧縮ではなく素の格納
#: （`karume.emit` の plain 側）だから — 実際この family の系列は i32 の添字表を 1 本持つので
#: ヘッダは F32 + I32（2026-08-30 の実測）で、I32 を禁じると既存の配布物が赤になる。
BIREFNET_STORAGE_FORBIDDEN: Mapping[str, tuple[str, ...]] = {BIREFNET_ROLE: ("F16", "I8", "I4")}

#: weights の宣言（dtype ラベル → 役割名）。dtype が 1 つしかないので quant 表は空でよい。
BIREFNET_WEIGHTS: Mapping[str, Mapping[str, WeightFiles]] = {
    BIREFNET_ROLE: {"f32": WeightFiles(BIREFNET_ROLE)}
}

#: assets の宣言。**空**（実行に要るのはグラフ 1 本だけ）。
BIREFNET_ASSETS: Mapping[str, str] = {}

BIREFNET_QUANTS: Mapping[str, Any] = {"f32": {"weights": {}, "session": {}}}
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


def birefnet_series_name(checkpoint: str, model: str) -> str:
    """checkpoint と解像度 → 系列ディレクトリ名（`birefnet.export.default_out_dir` と同じ式）。"""
    name = birefnet_checkpoint(checkpoint).lower().replace("_", "-")
    return f"{name}-{birefnet_resolution(model)}"


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
    """組み立ての入力。系列（グラフ 1 本）だけ。

    SigLIP2 と違って実重みの置き場を持たないのは、前処理定数の出どころになる機械可読な
    ファイルが上流に無いから（節の冒頭）。**どの重みのどの解像度かを言えるのは系列 path
    だけ**なので、系列名の導出（{@link birefnet_series_name}）が帰属の唯一の紐づけになる。
    """

    series: Path


def birefnet_sources(
    series_dir: Path, checkpoint: str, model: str = BIREFNET_DEFAULT_MODEL
) -> BirefnetSources:
    """系列の親ディレクトリ（`outputs/series/`）と checkpoint・解像度から入力を引く。"""
    return BirefnetSources(series=series_dir / birefnet_series_name(checkpoint, model))


def birefnet_placements(sources: BirefnetSources) -> dict[str, Path]:
    """役割名 → 出所のファイル。出力の path は {@link BIREFNET_OUTPUT_PATHS} が持つ。

    この表に無いものは出力へ入らない（系列に並ぶ `io.*.safetensors` はこれで落ちる）。
    """
    return {BIREFNET_ROLE: sources.series / "model.safetensors"}


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
    container = placements[BIREFNET_ROLE]
    graph = ir_graph(container)
    pipeline_config = birefnet_pipeline_config(graph, container, model)
    assert_birefnet_graph(graph, container, pipeline_config)
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
#: manifest 検査も素通りし、配ってからでないと食い違いに気づけない。
BIREFNET_NOTICE_TEMPLATE = """# NOTICE

This repository redistributes a modified form of `{repo}`, which is licensed under
the MIT License (see `LICENSE.md`). The following changes were made:

- The graph was re-expressed in the Karume container format (a graph shard whose
  `__metadata__` carries the graph, followed by the weight shards it names).
- The upstream `forward` was rewritten layout-only and **bit-exact**: windowing, the
  shifted-window roll, spatial padding and the patch merges became equivalent operations.
- Two modules were rewritten in a form that is equivalent up to floating-point rounding:
  inference-time `BatchNorm2d` became a per-channel affine, and the ASPP image-level pooling
  became a two-stage sum.
- The decoder tail's 1×1 convolution and the bilinear upsample it used to follow were swapped
  in order. Both are linear on disjoint axes, so they commute; the rewrite is equivalent up to
  floating-point rounding and removes two full-resolution intermediates.
- **No quantization**: the stored weights are the source checkpoint's own f32 values.

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
        "NOTICE.md": BIREFNET_NOTICE_TEMPLATE.format(repo=BIREFNET_UPSTREAM[checkpoint]),
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
