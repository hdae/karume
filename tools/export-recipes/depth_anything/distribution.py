"""Depth Anything V2 の配布 recipe — 系列レイアウト・出力 path 表・quant 表・カードの選択。

ADR 0065 決定 2。汎用の組み立てエンジン（配置・共有席の畳み込み・sha256・manifest・
staging/swap・検証）は `karume.dist` が持つ。ここが持つのは **Depth Anything V2 固有の事実**
だけ: どの系列ディレクトリから何を拾い、配布形のどの path へ、どの dtype ラベルで並べ、
どの quant を既定にするか。

配布するのは **深度マップ 1 グラフだけ**（`depth_anything/export.py` は `predicted_depth`
1 本しか出さない）。実行に要る資産もそれ 1 本で、tokenizer も表も無い（`assets` は空）。
格納 dtype は f32 の 1 系列だけなので quant 席も 1 つ — ここまでは SigLIP2 / BiRefNet と同じ形。

**1 リポ 1 モデル = 1 サイズ**（`karume-depth-anything-v2-<サイズ>`）。サイズは重みも
埋め込み幅も違う別物で、同居させると利用者が何も指定しなかったときに引く既定が丸ごと変わる。

MUST: 配れるのは**帰属表（{@link depth_anything.card.DEPTH_ANYTHING_UPSTREAM}）に載っている
サイズだけ**。上流で Apache-2.0 なのは Small のみで、Base / Large は CC BY-NC 4.0（同表の MUST
に実地確認の日付つき）。台本（`depth_anything/export.py`）は `--model-dir` でどのサイズも
焼けるので、「焼けたものは配れる」と読める形にしない — この表が唯一の門。

`pipelineConfig` の数は **2 つの独立した出どころ**から来る: 前処理の定数（寸法・統計・
フィルタ）は上流の `preprocessor_config.json`、噛み合っているかは**焼かれたグラフの入出力
宣言**。どちらも写経しない（TS 側の正本 `packages/models/src/depth-anything/config.ts` の
モジュール doc と同じ理由）。ずれた組み合わせは「resize 先だけが違う」形で値が静かに崩れる
ので、{@link assert_depth_anything_graph} が配置の前に実測する。

公開面は {@link PIPELINE} 1 つ（`karume.dist.Pipeline`）— リポの dist ドライバ
（`tools/export-recipes/dist.py`）がこれを core の PIPELINES へ合成する。
"""

from __future__ import annotations

import json
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
    complete_quant_weights,
    graph_inputs,
    ir_graph,
    preprocessor_channels,
    preprocessor_int,
)
from karume.paths import INPUTS_ROOT

from .card import DEPTH_ANYTHING_UPSTREAM, render_depth_anything_model_card

#: パイプライン契約（ADR 0041 §2 — モデル単位）。TS 側の受理集合は
#: `DEPTH_ANYTHING_PIPELINE_NAME` / `DEPTH_ANYTHING_PIPELINE_MAJOR`。
DEPTH_ANYTHING_PIPELINE = "depth-anything/1"

#: 既定のモデル名（= 既定のリポ名 `karume-depth-anything-v2-small` の末尾）。綴りの受理集合は
#: 帰属表（`depth_anything.card.DEPTH_ANYTHING_UPSTREAM`）が持つ。
DEPTH_ANYTHING_DEFAULT_MODEL = "small"

#: リポ名の接頭辞（`karume-depth-anything-v2-<サイズ>`）。系列名は別（上流リポ名の小文字化 —
#: {@link depth_anything_series_name}）。V2 まで綴りへ入れるのは、
#: V3 が別アーキ（多視点 + カメラ姿勢の DualDPT）で単一画像 depth の後継ではないため —
#: 世代を落とすと「新しい方が良い」と読める並びになる。
DEPTH_ANYTHING_PREFIX = "depth-anything-v2"

#: 実重みと `preprocessor_config.json` の親（`hf download depth-anything/<名前> --local-dir
#: inputs/depth-anything/<名前>` の展開先 — `depth_anything.export.MODELS_ROOT` と同じ場所）。
DEPTH_ANYTHING_INPUTS_DIRNAME = "depth-anything"

#: 唯一の役割名（manifest の weights キー・TS 側 `pipeline.ts` の `DEPTH`）。
DEPTH_ANYTHING_ROLE = "depth"

#: グラフ入力の名前（`depth_anything.export.INPUT_NAME`）。
DEPTH_ANYTHING_INPUT = "pixel_values"

#: 入力のチャネル数（RGB）。出力は**チャネル軸を持たない** `[1, H, W]` の地図。
DEPTH_ANYTHING_CHANNELS = 3

#: 前処理定数の出どころ（重みと同じディレクトリ）。
DEPTH_ANYTHING_PREPROCESSOR_FILE = "preprocessor_config.json"

#: TS 側が実装している補間と、それに対応する `resample` の値（**PIL の定数で BICUBIC**）。
#: SigLIP2 / BiRefNet の bilinear（2）と**違う**ので、TS 側 `config.ts` の受理集合も 1 値だけ
#: bicubic に絞ってある。`depth_anything.export.check_processor` が emit のたびに現物から実測する
#: のと同じ対象を、こちらは配布形の宣言として書く。
DEPTH_ANYTHING_INTERPOLATION = "bicubic"
DEPTH_ANYTHING_RESAMPLE = 3

#: TS 側 `normalizeToNchw` が持つ除数 255 の逆数。**pipelineConfig には載せない**（実行時に
#: 選べない数を宣言だけ持たせると正本が 2 つになる — `config.ts` の NOTE）ので、違う値の
#: チェックポイントはここで落とす。
DEPTH_ANYTHING_RESCALE_FACTOR = 1.0 / 255.0

#: 出力の相対 path（**モデルサブツリー内**）— 配置表と manifest が共有する 1 箇所。
DEPTH_ANYTHING_OUTPUT_PATHS: Mapping[str, str] = {
    DEPTH_ANYTHING_ROLE: f"{DEPTH_ANYTHING_ROLE}/model.f32.safetensors"
}

#: 格納 dtype の要求（Anima / SBV2 / Irodori / SigLIP2 / BiRefNet と同じ根拠 — 素の資産が
#: 組み立て・ロード・実行を全て通って参照一致の門まで沈黙した実測事故）。
#:
#: NOTE: 禁止表（{@link assert_storage_absent}）は持たない — 圧縮系列が 1 本も無いので、
#: 「F32 を含む」で系列 × 格納 dtype が一意に決まる。f16 / i8 の席を足すときは**同時に**禁止表も
#: 足す（圧縮系列も適格外の重みを F32 で持つので、存在検査だけでは f32 席へ混入する）。
DEPTH_ANYTHING_STORAGE_REQUIREMENTS: Mapping[str, str] = {DEPTH_ANYTHING_ROLE: "F32"}

#: weights の宣言（dtype ラベル → 役割名）。dtype が 1 つしかないので quant 表は空でよい
#: （{@link complete_quant_weights} が完全写像へ埋める）。
DEPTH_ANYTHING_WEIGHTS: Mapping[str, Mapping[str, WeightFiles]] = {
    DEPTH_ANYTHING_ROLE: {"f32": WeightFiles(DEPTH_ANYTHING_ROLE)}
}

#: assets の宣言。**空**（実行に要るのはグラフ 1 本だけ）。
DEPTH_ANYTHING_ASSETS: Mapping[str, str] = {}

DEPTH_ANYTHING_QUANTS: Mapping[str, Any] = {"f32": {"weights": {}, "session": {}}}
DEPTH_ANYTHING_DEFAULT_QUANT = "f32"


def depth_anything_checkpoint(model: str) -> str:
    """モデル名 → 上流チェックポイントのディレクトリ名（= 上流の HF リポ名の末尾）。

    `depth_anything/export.py` は `--model-dir` のディレクトリ名を小文字化して系列名にし、
    そのディレクトリ名は `hf download <リポ>` の展開先なので、綴りの事実は「このモデルが
    どの上流リポか」1 つしかない。帰属表（`depth_anything.card.DEPTH_ANYTHING_UPSTREAM`）から
    導いて、ここに 2 つ目の表を持たない（SigLIP2 の
    {@link siglip2.distribution.siglip2_checkpoint} と同じ規律）。

    MUST: 表に無いサイズはここで落ちる — それが「Apache-2.0 のものしか配らない」門の実体
    （節の冒頭の MUST）。
    """
    repo = DEPTH_ANYTHING_UPSTREAM.get(model)
    if repo is None:
        raise DistError(
            f"Depth Anything V2 のモデル '{model}' は知らない"
            f"（既知: {' / '.join(sorted(DEPTH_ANYTHING_UPSTREAM))}）"
            " — 上流で Apache-2.0 のサイズだけが帰属表に載っている"
        )
    return repo.split("/", 1)[1]


def depth_anything_series_name(model: str) -> str:
    """モデル名 → 系列ディレクトリ名（`depth_anything.export.default_out_dir` と同じ式）。"""
    return depth_anything_checkpoint(model).lower()


def depth_anything_repo_name(model: str) -> str:
    """単一モデルの配布リポ名（`karume-` prefix はリポ名裁定 2026-08-09）。"""
    return f"karume-{DEPTH_ANYTHING_PREFIX}-{model}"


@dataclass(frozen=True)
class DepthAnythingSources:
    """組み立ての入力。系列（グラフ 1 本）と実重みの置き場（`inputs/` — 生成物ではない）。

    後者が要るのは `pipelineConfig` の前処理定数を**焼き込まずに導出**するため
    （読むのは `preprocessor_config.json` 1 本だけなので、99MB の重みには触らない）。
    """

    series: Path
    model: Path


def depth_anything_sources(
    series_dir: Path, model: str = DEPTH_ANYTHING_DEFAULT_MODEL
) -> DepthAnythingSources:
    """系列の親ディレクトリ（`outputs/series/`）と `karume.paths` の綴りから入力を引く。"""
    checkpoint = depth_anything_checkpoint(model)
    return DepthAnythingSources(
        series=series_dir / depth_anything_series_name(model),
        model=INPUTS_ROOT / DEPTH_ANYTHING_INPUTS_DIRNAME / checkpoint,
    )


def depth_anything_placements(sources: DepthAnythingSources) -> dict[str, Path]:
    """役割名 → 出所のファイル。出力の path は {@link DEPTH_ANYTHING_OUTPUT_PATHS} が持つ。

    この表に無いものは出力へ入らない（系列に並ぶ `io.*.safetensors` はこれで落ちる）。
    """
    return {DEPTH_ANYTHING_ROLE: sources.series / "model.safetensors"}


def depth_anything_preprocessor(model_dir: Path) -> Mapping[str, Any]:
    """上流の `preprocessor_config.json` を読む（前処理定数の唯一の出どころ）。"""
    path = model_dir / DEPTH_ANYTHING_PREPROCESSOR_FILE
    if not path.is_file():
        raise DistError(f"組み立ての入力が無い: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise DistError(f"{path} が JSON として読めない") from error
    if not isinstance(raw, dict):
        raise DistError(f"{path}: 最上位オブジェクトでない")
    return raw


def depth_anything_pipeline_config(preprocessor: Mapping[str, Any], where: str) -> dict[str, Any]:
    """`pipelineConfig`（TS 側スキーマの 5 欄）を上流の前処理 config から組む。

    MUST: 対応外の前処理は**受理しない**（黙って近似しない）。TS 側が持つのは
    「antialias 付き bicubic → `/255` → `(x − mean) / std`」の 1 本きりなので、
    `resample` / `rescale_factor` / `do_*` が外れたチェックポイントは、値がずれたまま
    ロードも実行も通る。ここが唯一の検出器になる。

    NOTE: `keep_aspect_ratio` / `ensure_multiple_of` は**読むが宣言へ写さない**。焼かれた
    グラフが正方 1 点でしか受け取らないので（DINOv2 の位置埋め込みがパッチ格子に紐づいて
    いる）、アスペクト比を保つ経路には行き先が無く、宣言だけ置くと「保てるのに保っていない」
    と読める欄になる。事実はモデルカードの散文が持つ（`_depth_anything_shape`）。
    """
    size = preprocessor.get("size")
    if not isinstance(size, dict):
        raise DistError(f"{where} に size が無い / オブジェクトでない（{size!r}）")
    resample = preprocessor.get("resample")
    if resample != DEPTH_ANYTHING_RESAMPLE:
        raise DistError(
            f"{where} の resample が {resample!r}"
            f"（期待 {DEPTH_ANYTHING_RESAMPLE} = PIL の BICUBIC）"
            f" — TS 側の前処理は {DEPTH_ANYTHING_INTERPOLATION} の 1 本しか持たない"
        )
    rescale = preprocessor.get("rescale_factor")
    if rescale != DEPTH_ANYTHING_RESCALE_FACTOR:
        raise DistError(
            f"{where} の rescale_factor が {rescale!r}（期待 {DEPTH_ANYTHING_RESCALE_FACTOR}）"
            " — TS 側は 8bit 画素を 255 で割る形で閉じている"
        )
    for flag in ("do_resize", "do_rescale", "do_normalize"):
        if preprocessor.get(flag) is not True:
            raise DistError(f"{where} の {flag} が真でない — 前処理の 3 段が揃っていない")
    for flag in ("do_center_crop", "do_pad"):
        if preprocessor.get(flag):
            raise DistError(f"{where} の {flag} が真 — TS 側は crop も pad も持たない")
    return {
        "imageWidth": preprocessor_int(size, "width", f"{where} の size"),
        "imageHeight": preprocessor_int(size, "height", f"{where} の size"),
        "imageMean": preprocessor_channels(
            preprocessor, "image_mean", where, channels=DEPTH_ANYTHING_CHANNELS, positive=False
        ),
        "imageStd": preprocessor_channels(
            preprocessor, "image_std", where, channels=DEPTH_ANYTHING_CHANNELS, positive=True
        ),
        "interpolation": DEPTH_ANYTHING_INTERPOLATION,
    }


def assert_depth_anything_graph(
    graph: Mapping[str, Any], path: Path, pipeline_config: Mapping[str, Any]
) -> None:
    """グラフの入出力が**前処理の宣言と噛み合う**ことを、配置の前に実測する。

    MUST: 落とさない。前処理の寸法（`preprocessor_config.json`）と焼かれた解像度（重みの
    `config.json` の `image_size`）は別々に決まるので、別サイズの組み合わせでもここまでは
    何も落ちない — ずれたまま配ると、利用者の手元で Session の shape 検査が「どちらの数が
    正しいのか」を伝えないまま落ちる。

    MUST: 出力の**階数**まで見る。深度地図は `[1, H, W]`（チャネル軸を持たない）で、
    `[1, 1, H, W]` と要素数では区別できない — 位置で引く後段は要素数しか見ないので、
    チャネル軸つきで焼かれたグラフは黙って通ってしまう。
    """
    inputs = graph_inputs(graph, path)
    if tuple(inputs) != (DEPTH_ANYTHING_INPUT,):
        raise DistError(
            f"{path} のグラフ入力が {list(inputs)} で、期待の {[DEPTH_ANYTHING_INPUT]} と違う"
            " — 実行側は名前で束ねるので、綴りが変われば束ねられない"
        )
    symbols = graph.get("symbols")
    if not isinstance(symbols, list) or symbols:
        raise DistError(
            f"{path}: 記号次元 {symbols!r} がある — 解像度もパッチ数も固定で、動かす軸は無い"
        )
    height, width = pipeline_config["imageHeight"], pipeline_config["imageWidth"]
    expected_input = [1, DEPTH_ANYTHING_CHANNELS, height, width]
    if inputs[DEPTH_ANYTHING_INPUT] != expected_input:
        raise DistError(
            f"{path} の入力 '{DEPTH_ANYTHING_INPUT}' が {inputs[DEPTH_ANYTHING_INPUT]!r}、"
            f"{DEPTH_ANYTHING_PREPROCESSOR_FILE} の size から組んだ期待は {expected_input}"
            " — 前処理の寸法と焼かれた解像度が別の版"
        )
    outputs = graph.get("outputs")
    if not isinstance(outputs, list) or len(outputs) != 1:
        raise DistError(f"{path}: グラフ出力が {outputs!r} — 深度マップ 1 本だけが要る")
    values = graph.get("values")
    value = values.get(outputs[0]) if isinstance(values, dict) else None
    shape = value.get("shape") if isinstance(value, dict) else None
    expected_output = [1, height, width]
    if shape != expected_output:
        raise DistError(
            f"{path}: グラフ出力 '{outputs[0]}' の形が {shape!r}、期待は {expected_output}"
            " — 深度マップは入力と同じ寸法・チャネル軸なし"
        )


def depth_anything_plan(
    sources: DepthAnythingSources, model: str = DEPTH_ANYTHING_DEFAULT_MODEL
) -> ModelPlan:
    """Depth Anything V2 の 1 モデルぶんの計画を組む（検査と読み取りをここで全部済ませる）。"""
    assert_model_name(model)
    # 帰属を名乗れないサイズは系列を読む前に落とす（節の冒頭の MUST）。
    depth_anything_checkpoint(model)
    placements = depth_anything_placements(sources)
    for role, source in placements.items():
        assert_storage(role, source, DEPTH_ANYTHING_STORAGE_REQUIREMENTS)
    container = placements[DEPTH_ANYTHING_ROLE]
    graph = ir_graph(container)
    pipeline_config = depth_anything_pipeline_config(
        depth_anything_preprocessor(sources.model),
        str(sources.model / DEPTH_ANYTHING_PREPROCESSOR_FILE),
    )
    assert_depth_anything_graph(graph, container, pipeline_config)
    return ModelPlan(
        name=model,
        pipeline=DEPTH_ANYTHING_PIPELINE,
        artifacts={
            role: Artifact(DEPTH_ANYTHING_OUTPUT_PATHS[role], source=source)
            for role, source in placements.items()
        },
        weights=DEPTH_ANYTHING_WEIGHTS,
        assets=DEPTH_ANYTHING_ASSETS,
        quants=complete_quant_weights(DEPTH_ANYTHING_WEIGHTS, DEPTH_ANYTHING_QUANTS),
        default_quant=DEPTH_ANYTHING_DEFAULT_QUANT,
        pipeline_config=pipeline_config,
    )


def depth_anything_dist_plan(series_dir: Path, model: str) -> ModelPlan:
    """`--series` の親から Depth Anything V2 の 1 モデルの計画を組む（CLI のディスパッチ先）。"""
    return depth_anything_plan(depth_anything_sources(series_dir, model), model)


#: `--pipeline depth-anything` の 1 行（ドライバが core の PIPELINES へ合成する）。
PIPELINE = Pipeline(
    default_model=DEPTH_ANYTHING_DEFAULT_MODEL,
    repo_name=depth_anything_repo_name,
    plan=depth_anything_dist_plan,
    # SigLIP2 / BiRefNet と同じ理由で選ばせる軸にしない — 帰属（上流リポ・ライセンス）は
    # モデル名から一意に決まる。プロファイルを分けると「Base を Small の帰属で配る」
    # 取り違えを操作者が起こせるようになる（しかも Base は CC BY-NC 4.0）。
    card_profiles={"depth-anything": render_depth_anything_model_card},
)
