"""BiRefNet 系の配布 recipe — 系列レイアウト・出力 path 表・quant 表・カードの選択。

ADR 0065 決定 2。汎用の組み立てエンジン（配置・共有席の畳み込み・sha256・manifest・
staging/swap・検証）は `karume.dist` が持つ。ここが持つのは **BiRefNet 系固有の事実**だけ:
どの系列ディレクトリから何を拾い、配布形のどの path へ、どの dtype ラベルで並べ、どの quant を
既定にするか。

配布するのは **マット 1 グラフだけ**（`birefnet/export.py` は最終段の logit 1 本しか出さない）。
実行に要る資産もそれ 1 本で、tokenizer も表も無い（`assets` は空）。格納 dtype は f32 の
1 系列だけなので quant 席も 1 つ — ここまでは SigLIP2 と同じ形。

**1 リポ 1 モデル**（ユーザー裁定 — BiRefNet_HR と Lucida は構造が同一で重みだけが違う
fine-tune 同士なので、まとめると「何も指定しなかったときにどちらの学習が動くか」が
既定に隠れる）。リポ名は導出せず {@link BIREFNET_REPO_NAMES} が持つ — `karume-lucida` は
「BiRefNet 系の 1 つ」ではなく上流が名前で売っているモデルで、綴りは命名の決定であって
モデル名から決まらない（SBV2 のファミリー名と同じ性質）。

`pipelineConfig` の数の出どころは **2 つとも独立**:

- resize 先（`imageWidth` / `imageHeight`）は**焼かれたグラフの入力宣言**から導く（写経しない）。
- 正規化定数は上流に機械可読な出どころが**無い** — SigLIP2 の `preprocessor_config.json` に
  当たるものが BiRefNet 系には無く、事実は同梱 `handler.py` の `ImagePreprocessor`（と
  モデルカードの利用例）の中にしか書かれていない。したがってここが宣言として持ち
  （{@link BIREFNET_IMAGE_MEAN}）、台本側の写し（`birefnet.export.IMAGENET_MEAN`）との一致は
  pytest が毎回突き合わせる（2 表が独立に動く形にはしない）。

公開面は {@link PIPELINE} 1 つ（`karume.dist.Pipeline`）— リポの dist ドライバ
（`tools/export-recipes/dist.py`）がこれを core の PIPELINES へ合成する。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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

#: 既定のモデル名。綴りの受理集合は帰属表（`birefnet.card.BIREFNET_UPSTREAM`）が持つ。
BIREFNET_DEFAULT_MODEL = "hr"

#: 実重みの親（`hf download <リポ> --local-dir inputs/birefnet/<名前>` の展開先 —
#: `birefnet.export.MODELS_ROOT` と同じ場所）。系列名はこのディレクトリ名から決まる。
BIREFNET_INPUTS_DIRNAME = "birefnet"

#: 単一モデルの配布リポ名（`karume-` prefix はリポ名裁定 2026-08-09）。**導出しない** —
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

#: 配る解像度。系列は解像度ごとに別（`birefnet/export.py` の「解像度軸」）なので、系列名を
#: 引くのにこの数が要る。**2048²（本家 handler の General-HR）は配らない** — conv2d の
#: dispatch 上限と中間 3.22GB で実行段が未実測（docs/limitations.md）。焼かれたグラフが別の
#: 解像度なら {@link birefnet_pipeline_config} が落とす。
BIREFNET_RESOLUTION = 1024

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


def birefnet_checkpoint(model: str) -> str:
    """モデル名 → 上流チェックポイントのディレクトリ名（= 上流の HF リポ名の末尾）。

    `birefnet/export.py` は `--model-dir` のディレクトリ名を系列名にし、そのディレクトリ名は
    `hf download <リポ>` の展開先なので、綴りの事実は「このモデルがどの上流リポか」1 つしか
    ない。帰属表（`birefnet.card.BIREFNET_UPSTREAM`）から導いて、ここに 2 つ目の表を
    持たない（SigLIP2 の {@link siglip2.distribution.siglip2_checkpoint} と同じ規律）。
    """
    repo = BIREFNET_UPSTREAM.get(model)
    if repo is None:
        raise DistError(
            f"BiRefNet 系のモデル '{model}' は知らない"
            f"（既知: {' / '.join(sorted(BIREFNET_UPSTREAM))}）"
        )
    return repo.split("/", 1)[1]


def birefnet_series_name(model: str) -> str:
    """モデル名 → 系列ディレクトリ名（`birefnet.export.default_out_dir` と同じ式）。"""
    return f"{birefnet_checkpoint(model).lower().replace('_', '-')}-{BIREFNET_RESOLUTION}"


def birefnet_repo_name(model: str) -> str:
    """単一モデルの配布リポ名（{@link BIREFNET_REPO_NAMES}）。"""
    name = BIREFNET_REPO_NAMES.get(model)
    if name is None:
        raise DistError(
            f"BiRefNet 系のモデル '{model}' のリポ名が無い"
            f"（既知: {' / '.join(sorted(BIREFNET_REPO_NAMES))}）"
        )
    return name


@dataclass(frozen=True)
class BirefnetSources:
    """組み立ての入力。系列（グラフ 1 本）だけ。

    SigLIP2 と違って実重みの置き場を持たないのは、前処理定数の出どころになる機械可読な
    ファイルが上流に無いから（節の冒頭）。**どのモデルの重みかを言えるのは系列 path だけ**
    なので、系列名の導出（{@link birefnet_series_name}）が帰属の唯一の紐づけになる。
    """

    series: Path


def birefnet_sources(series_dir: Path, model: str = BIREFNET_DEFAULT_MODEL) -> BirefnetSources:
    """系列の親ディレクトリ（`outputs/series/`）とモデル名から入力を引く。"""
    return BirefnetSources(series=series_dir / birefnet_series_name(model))


def birefnet_placements(sources: BirefnetSources) -> dict[str, Path]:
    """役割名 → 出所のファイル。出力の path は {@link BIREFNET_OUTPUT_PATHS} が持つ。

    この表に無いものは出力へ入らない（系列に並ぶ `io.*.safetensors` はこれで落ちる）。
    """
    return {BIREFNET_ROLE: sources.series / "model.safetensors"}


def birefnet_pipeline_config(graph: Mapping[str, Any], path: Path) -> dict[str, Any]:
    """`pipelineConfig`（TS 側スキーマの 5 欄）を焼かれたグラフと正規化定数から組む。

    resize 先はグラフの入力宣言そのもの — 前処理が別の寸法へ伸ばすと、値が静かに崩れたまま
    shape だけ合う。入力の名前・階数・batch・チャネル数もここで見る（後段の
    {@link assert_birefnet_graph} が出力側を見る）。
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
    if (height, width) != (BIREFNET_RESOLUTION, BIREFNET_RESOLUTION):
        raise DistError(
            f"{path} は {height}×{width} で焼かれている — 配るのは"
            f" {BIREFNET_RESOLUTION}² だけ（それ以外は実行段が未実測 — docs/limitations.md）"
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


def birefnet_plan(sources: BirefnetSources, model: str = BIREFNET_DEFAULT_MODEL) -> ModelPlan:
    """BiRefNet 系 1 モデルぶんの計画を組む（検査と config の読み取りをここで全部済ませる）。"""
    assert_model_name(model)
    # 帰属を名乗れないモデル名は系列を読む前に落とす（リポ名の表と帰属表は独立に欠けうる）。
    birefnet_checkpoint(model)
    birefnet_repo_name(model)
    placements = birefnet_placements(sources)
    for role, source in placements.items():
        assert_storage(role, source, BIREFNET_STORAGE_REQUIREMENTS)
        assert_storage_absent(role, source, BIREFNET_STORAGE_FORBIDDEN)
    container = placements[BIREFNET_ROLE]
    graph = ir_graph(container)
    pipeline_config = birefnet_pipeline_config(graph, container)
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


def birefnet_dist_plan(series_dir: Path, model: str) -> ModelPlan:
    """`--series` の親から BiRefNet 系 1 モデルの計画を組む（CLI のディスパッチ先）。"""
    return birefnet_plan(birefnet_sources(series_dir, model), model)


#: `--pipeline birefnet` の 1 行（ドライバが core の PIPELINES へ合成する）。
PIPELINE = Pipeline(
    default_model=BIREFNET_DEFAULT_MODEL,
    repo_name=birefnet_repo_name,
    plan=birefnet_dist_plan,
    # SigLIP2 と同じ理由で選ばせる軸にしない — 帰属（上流リポ・ライセンス・学習データ）は
    # モデル名から一意に決まる。プロファイルを分けると「Lucida を BiRefNet_HR の帰属で
    # 配る」取り違えを操作者が起こせるようになる。
    card_profiles={"birefnet": render_birefnet_model_card},
)
