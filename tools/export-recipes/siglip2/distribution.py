"""SigLIP2 の配布 recipe — 系列レイアウト・出力 path 表・quant 表・カードの選択（ADR 0065 決定 2）。

汎用の組み立てエンジン（配置・共有席の畳み込み・sha256・manifest・staging/swap・検証）は
`karume.dist` が持つ。ここが持つのは **SigLIP2 固有の事実**だけ: どの系列ディレクトリから何を
拾い、配布形のどの path へ、どの dtype ラベルで並べ、どの quant を既定にするか。

配布するのは **vision tower の 1 グラフだけ**（text tower は載っていない — `siglip2/export.py`
の docstring）。実行に要る資産もそれ 1 本で、tokenizer も表も無い（`assets` は空）。格納
dtype は f32 の 1 系列だけなので quant 席も 1 つ。

**1 リポ 1 モデル**（base と so400m は解像度も hidden も違う別物で、同居させると利用者が
何も指定しなかったときに引く既定が寸法ごと変わる）。ファミリー組み立ての機構は core の共有部が
持ったままだが、既定の出力先は `karume-siglip2-<モデル名>` の単一モデル形。

`pipelineConfig` の数は**2 つの独立した出どころ**から来る: 前処理の定数は上流の
`preprocessor_config.json`、`hiddenDim` は焼かれたグラフの出力宣言。どちらも写経しない
（TS 側の正本 `packages/models/src/siglip2/config.ts` のモジュール doc と同じ理由）。前処理の
寸法と焼かれた解像度は別々に決まるので、噛み合っていることは {@link assert_siglip2_graph} が
実測する — ずれた組み合わせは「resize 先だけが違う」形で、値が静かに崩れる。

公開面は {@link PIPELINE} 1 つ（`karume.dist.Pipeline`）— リポの dist ドライバ
（`tools/export-recipes/dist.py`）がこれを core の PIPELINES へ合成する。
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from _shared.paths import INPUTS_ROOT
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

from .card import SIGLIP2_UPSTREAM, render_siglip2_model_card

#: パイプライン契約（ADR 0041 §2 — モデル単位）。TS 側の受理集合は
#: `SIGLIP2_PIPELINE_NAME` / `SIGLIP2_PIPELINE_MAJOR`。
SIGLIP2_PIPELINE = "siglip2/1"

#: 既定のモデル名（= 既定のリポ名 `karume-siglip2-base` の末尾）。綴りの受理集合は帰属表
#: （`siglip2.card.SIGLIP2_UPSTREAM`）が持つ。
SIGLIP2_DEFAULT_MODEL = "base"

#: 系列名とリポ名の接頭辞（`karume-siglip2-<モデル名>`）。
SIGLIP2_PREFIX = "siglip2"

#: 実重みと `preprocessor_config.json` の親（`hf download google/<名前> --local-dir
#: inputs/siglip2/<名前>` の展開先 — `siglip2.export.MODELS_ROOT` と同じ場所）。
SIGLIP2_INPUTS_DIRNAME = "siglip2"

#: 唯一の役割名（manifest の weights キー・TS 側 `pipeline.ts` の `VISION`）。
SIGLIP2_ROLE = "vision"

#: グラフ入力の名前（`siglip2.export.INPUT_NAME`）。
SIGLIP2_INPUT = "pixel_values"

#: 入力のチャネル数（`siglip2.patch.assert_supported` が config 側でも要求する 3）。
SIGLIP2_CHANNELS = 3

#: 前処理定数の出どころ（重みと同じディレクトリ）。
SIGLIP2_PREPROCESSOR_FILE = "preprocessor_config.json"

#: TS 側が実装している唯一の補間と、それに対応する `resample` の値（**PIL の定数で BILINEAR**
#: — BICUBIC は 3。クラス属性の既定が BICUBIC なので読み違えやすいが、チェックポイントの
#: config が上書きしている）。`siglip2/preprocess.py` の `EXPECTED_RESAMPLE` と同じ実測対象。
SIGLIP2_INTERPOLATION = "bilinear"
SIGLIP2_RESAMPLE = 2

#: TS 側 `normalizeToNchw` が持つ除数 255 の逆数。**pipelineConfig には載せない**（実行時に
#: 選べない数を宣言だけ持たせると正本が 2 つになる — `config.ts` の NOTE）ので、違う値の
#: チェックポイントはここで落とす。
SIGLIP2_RESCALE_FACTOR = 1.0 / 255.0

#: 出力の相対 path（**モデルサブツリー内**）— 配置表と manifest が共有する 1 箇所。格納 dtype を
#: ファイル名に出すのは Irodori と同じ形（系列が 2 本並んでも取り違えようがない綴り）。
SIGLIP2_OUTPUT_PATHS: Mapping[str, str] = {SIGLIP2_ROLE: f"{SIGLIP2_ROLE}/model.f32.safetensors"}

#: 格納 dtype の要求（Anima / SBV2 / Irodori と同じ根拠 — 素の資産が組み立て・ロード・実行を
#: 全て通って参照一致の門まで沈黙した実測事故）。
#:
#: NOTE: 禁止表（{@link assert_storage_absent}）は持たない — 圧縮系列が 1 本も無いので、
#: 「F32 を含む」で系列 × 格納 dtype が一意に決まる。f16 / i8 の席を足すときは**同時に**禁止表も
#: 足す（圧縮系列も適格外の重みを F32 で持つので、存在検査だけでは f32 席へ混入する）。
SIGLIP2_STORAGE_REQUIREMENTS: Mapping[str, str] = {SIGLIP2_ROLE: "F32"}

#: weights の宣言（dtype ラベル → 役割名）。dtype が 1 つしかないので quant 表は空でよい
#: （{@link complete_quant_weights} が完全写像へ埋める）。
SIGLIP2_WEIGHTS: Mapping[str, Mapping[str, WeightFiles]] = {
    SIGLIP2_ROLE: {"f32": WeightFiles(SIGLIP2_ROLE)}
}

#: assets の宣言。**空**（tokenizer も表も要らない — 実行に要るのはグラフ 1 本だけ）。
SIGLIP2_ASSETS: Mapping[str, str] = {}

SIGLIP2_QUANTS: Mapping[str, Any] = {"f32": {"weights": {}, "session": {}}}
SIGLIP2_DEFAULT_QUANT = "f32"


def siglip2_checkpoint(model: str) -> str:
    """モデル名 → 上流チェックポイントのディレクトリ名（= 系列名 = 上流の HF リポ名）。

    `siglip2/export.py` は `--model-dir` のディレクトリ名をそのまま系列名にし、そのディレクトリ
    名は `hf download <リポ>` の展開先（= リポ名の末尾）なので、綴りの事実は「このモデルが
    どの上流リポか」1 つしかない。帰属表（`siglip2.card.SIGLIP2_UPSTREAM`）から導いて、
    ここに 2 つ目の表を持たない — 2 表になると、片方だけ動いたときに「別のモデルの重みを
    別のモデルとして帰属する」形が黙って作れる。
    """
    repo = SIGLIP2_UPSTREAM.get(model)
    if repo is None:
        raise DistError(
            f"SigLIP2 のモデル '{model}' は知らない（既知: {' / '.join(sorted(SIGLIP2_UPSTREAM))}）"
        )
    return repo.split("/", 1)[1]


def siglip2_repo_name(model: str) -> str:
    """単一モデルの配布リポ名（`karume-` prefix はリポ名裁定 2026-08-09）。"""
    return f"karume-{SIGLIP2_PREFIX}-{model}"


@dataclass(frozen=True)
class Siglip2Sources:
    """組み立ての入力。系列（グラフ 1 本）と実重みの置き場（`inputs/` — 生成物ではない）。

    後者が要るのは `pipelineConfig` の前処理定数を**焼き込まずに導出**するため
    （読むのは `preprocessor_config.json` 1 本だけなので、1.5GB の重みには触らない）。
    """

    series: Path
    model: Path


def siglip2_sources(series_dir: Path, model: str = SIGLIP2_DEFAULT_MODEL) -> Siglip2Sources:
    """系列の親ディレクトリ（`outputs/series/`）と `_shared.paths` の綴りから入力を引く。"""
    checkpoint = siglip2_checkpoint(model)
    return Siglip2Sources(
        series=series_dir / checkpoint,
        model=INPUTS_ROOT / SIGLIP2_INPUTS_DIRNAME / checkpoint,
    )


def siglip2_placements(sources: Siglip2Sources) -> dict[str, Path]:
    """役割名 → 出所のファイル。出力の path は {@link SIGLIP2_OUTPUT_PATHS} が持つ。

    この表に無いものは出力へ入らない（系列に並ぶ `io.*.safetensors` はこれで落ちる）。
    """
    return {SIGLIP2_ROLE: sources.series / "model.safetensors"}


def siglip2_preprocessor(model_dir: Path) -> Mapping[str, Any]:
    """上流の `preprocessor_config.json` を読む（前処理定数の唯一の出どころ）。"""
    path = model_dir / SIGLIP2_PREPROCESSOR_FILE
    if not path.is_file():
        raise DistError(f"組み立ての入力が無い: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise DistError(f"{path} が JSON として読めない") from error
    if not isinstance(raw, dict):
        raise DistError(f"{path}: 最上位オブジェクトでない")
    return raw


def siglip2_pipeline_config(
    preprocessor: Mapping[str, Any], hidden_dim: int, where: str
) -> dict[str, Any]:
    """`pipelineConfig`（TS 側スキーマの 6 欄）を前処理 config と焼かれたグラフから組む。

    MUST: 対応外の前処理は**受理しない**（黙って近似しない）。TS 側が持つのは
    「antialias 付き bilinear → `/255` → `(x − mean) / std`」の 1 本きりなので、
    `resample` / `rescale_factor` / `do_*` が外れたチェックポイントは、値が最大 47/255 ずれた
    まま（bicubic の実測）ロードも実行も通る。ここが唯一の検出器になる。
    """
    size = preprocessor.get("size")
    if not isinstance(size, dict):
        raise DistError(f"{where} に size が無い / オブジェクトでない（{size!r}）")
    resample = preprocessor.get("resample")
    if resample != SIGLIP2_RESAMPLE:
        raise DistError(
            f"{where} の resample が {resample!r}（期待 {SIGLIP2_RESAMPLE} = PIL の BILINEAR）"
            f" — TS 側の前処理は {SIGLIP2_INTERPOLATION} の 1 本しか持たない"
        )
    rescale = preprocessor.get("rescale_factor")
    if rescale != SIGLIP2_RESCALE_FACTOR:
        raise DistError(
            f"{where} の rescale_factor が {rescale!r}（期待 {SIGLIP2_RESCALE_FACTOR}）"
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
            preprocessor, "image_mean", where, channels=SIGLIP2_CHANNELS, positive=False
        ),
        "imageStd": preprocessor_channels(
            preprocessor, "image_std", where, channels=SIGLIP2_CHANNELS, positive=True
        ),
        "hiddenDim": hidden_dim,
        "interpolation": SIGLIP2_INTERPOLATION,
    }


def siglip2_hidden_dim(graph: Mapping[str, Any], path: Path) -> int:
    """焼かれたグラフの出力宣言から `pooler_output` の幅を読む（写経しない）。

    出力が 1 本であることまで見るのは、`last_hidden_state` 込みの別の export や golden 用の
    多出力版が混ざると、**幅だけが静かに別の意味の数**になるため。
    """
    outputs = graph.get("outputs")
    if not isinstance(outputs, list) or len(outputs) != 1:
        raise DistError(
            f"{path}: グラフ出力が {outputs!r} — vision tower が出すのは pooler_output 1 本だけ"
        )
    values = graph.get("values")
    value = values.get(outputs[0]) if isinstance(values, dict) else None
    shape = value.get("shape") if isinstance(value, dict) else None
    if not isinstance(shape, list) or len(shape) != 2 or shape[0] != 1:
        raise DistError(f"{path}: グラフ出力 '{outputs[0]}' の形が [1, hidden] でない（{shape!r}）")
    hidden = shape[1]
    if not isinstance(hidden, int) or isinstance(hidden, bool) or hidden <= 0:
        raise DistError(f"{path}: グラフ出力の hidden が正の整数でない（{hidden!r}）")
    return hidden


def assert_siglip2_graph(
    graph: Mapping[str, Any], path: Path, pipeline_config: Mapping[str, Any]
) -> None:
    """グラフの入力が**前処理の宣言と噛み合う**ことを、配置の前に実測する。

    MUST: 落とさない。前処理の寸法（`preprocessor_config.json`）と焼かれた解像度（重みの
    `config.json`）は別々に決まるので、base の前処理 config と so400m のグラフを組み合わせても
    ここまでは何も落ちない — ずれたまま配ると、利用者の手元で Session の shape 検査が
    「どちらの数が正しいのか」を伝えないまま落ちる。
    """
    inputs = graph_inputs(graph, path)
    if tuple(inputs) != (SIGLIP2_INPUT,):
        raise DistError(
            f"{path} のグラフ入力が {list(inputs)} で、期待の {[SIGLIP2_INPUT]} と違う"
            " — 実行側は名前で束ねるので、綴りが変われば束ねられない"
        )
    symbols = graph.get("symbols")
    if not isinstance(symbols, list) or symbols:
        raise DistError(
            f"{path}: 記号次元 {symbols!r} がある — 解像度もパッチ数も固定で、動かす軸は無い"
        )
    expected = [
        1,
        SIGLIP2_CHANNELS,
        pipeline_config["imageHeight"],
        pipeline_config["imageWidth"],
    ]
    if inputs[SIGLIP2_INPUT] != expected:
        raise DistError(
            f"{path} の入力 '{SIGLIP2_INPUT}' が {inputs[SIGLIP2_INPUT]!r}、"
            f"{SIGLIP2_PREPROCESSOR_FILE} の size から組んだ期待は {expected}"
            " — 前処理の寸法と焼かれた解像度が別の版"
        )


def siglip2_plan(sources: Siglip2Sources, model: str = SIGLIP2_DEFAULT_MODEL) -> ModelPlan:
    """SigLIP2 1 モデルぶんの計画を組む（検査と config の読み取りをここで全部済ませる）。"""
    assert_model_name(model)
    placements = siglip2_placements(sources)
    for role, source in placements.items():
        assert_storage(role, source, SIGLIP2_STORAGE_REQUIREMENTS)
    container = placements[SIGLIP2_ROLE]
    graph = ir_graph(container)
    pipeline_config = siglip2_pipeline_config(
        siglip2_preprocessor(sources.model),
        siglip2_hidden_dim(graph, container),
        str(sources.model / SIGLIP2_PREPROCESSOR_FILE),
    )
    assert_siglip2_graph(graph, container, pipeline_config)
    return ModelPlan(
        name=model,
        pipeline=SIGLIP2_PIPELINE,
        artifacts={
            role: Artifact(SIGLIP2_OUTPUT_PATHS[role], source=source)
            for role, source in placements.items()
        },
        weights=SIGLIP2_WEIGHTS,
        assets=SIGLIP2_ASSETS,
        quants=complete_quant_weights(SIGLIP2_WEIGHTS, SIGLIP2_QUANTS),
        default_quant=SIGLIP2_DEFAULT_QUANT,
        pipeline_config=pipeline_config,
    )


def siglip2_dist_plan(series_dir: Path, model: str) -> ModelPlan:
    """`--series` の親から SigLIP2 1 モデルの計画を組む（CLI のディスパッチ先）。"""
    return siglip2_plan(siglip2_sources(series_dir, model), model)


#: `--pipeline siglip2` の 1 行（ドライバが core の PIPELINES へ合成する）。
PIPELINE = Pipeline(
    default_model=SIGLIP2_DEFAULT_MODEL,
    repo_name=siglip2_repo_name,
    plan=siglip2_dist_plan,
    # 帰属は**モデル名から一意に決まる**（`SIGLIP2_UPSTREAM` — base / so400m は別リポの
    # 重み）ので、選ばせる軸にしない。プロファイルを分けると「so400m を base の帰属で
    # 配る」取り違えを操作者が起こせるようになる。
    card_profiles={"siglip2": render_siglip2_model_card},
)
