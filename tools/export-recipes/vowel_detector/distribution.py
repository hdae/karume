"""母音検出の配布 recipe — 系列レイアウト・出力 path 表・quant 表・カードの選択（ADR 0065 決定 2）。

汎用の組み立てエンジン（配置・共有席の畳み込み・sha256・manifest・staging/swap・検証）は
`karume.dist` が持つ。ここが持つのは **母音検出固有の事実**だけ: どの系列ディレクトリから何を
拾い、配布形のどの path へ、どの dtype ラベルで並べ、どの quant を既定にするか。

配布するのは **CRNN グラフ 1 本**と、特徴抽出が要る **mel 基底 1 本**（`assets` 席）。
格納 dtype は f32 の 1 系列だけなので quant 席も 1 つ。

## 長さバケットは無くなった（ADR 0056 / 0057）

かつては `aten.gru.input` が時間方向へ完全展開されて T を動的軸にできず、長さバケット 4 本
（250 / 500 / 1000 / 2000 フレーム）を weights の役割軸に並べて右ゼロ pad で丸めていた。
`gru_scan` への差し替え（ADR 0056）と派生次元からのシンボル束縛（ADR 0057）で、**任意長が
1 本のグラフで通る**ようになったので、役割は `crnn` 1 つだけになった。

副作用として **pad 由来の数値差が消える**: 逆方向 GRU が pad 側から状態を持ち帰るせいで
バケット経路の `.lab` は「末尾 40ms の pau」「20ms の境界ずれ」「発話中間の pau」の 3 型で
実長経路と割れていた（実音声 4 本 × pad 10 段の実測）。今はパイプラインが実長のまま回すので、
実重み E2E が固定している `.lab` と**同じもの**が配布形からも出る。

## `pipelineConfig` の出どころ

特徴の契約（`sampleRate` / `featureDim` / `classes`）は上流 `feature_config.json` の逐語。
`maxFrames` は**焼いたグラフの記号次元の上限**で、IR は値域を持たない（`docs/ir-v1.md` の
`symbols` は名前だけ）ので配布形にしか無い数になる — SBV2 の `maxTokens` / `maxFrames` と
同じ持ち方で、台本の定数との一致は `vowel_detector/tests/test_distribution.py` が突き合わせる。
噛み合っていることは {@link assert_vowel_detector_graph} が実測する。

公開面は {@link PIPELINE} 1 つ（`karume.dist.Pipeline`）— リポの dist ドライバ
（`tools/export-recipes/dist.py`）がこれを core の PIPELINES へ合成する。
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

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
    table_payload,
)
from karume.paths import INPUTS_ROOT

from .card import render_vowel_detector_model_card

#: パイプライン契約（ADR 0041 §2 — モデル単位）。TS 側の受理集合は
#: `VOWEL_DETECTOR_PIPELINE_NAME` / `VOWEL_DETECTOR_PIPELINE_MAJOR`。
VOWEL_DETECTOR_PIPELINE = "vowel-detector/1"

#: 既定のモデル名 = **チェックポイントの世代**（`vowel_detector.export.default_out_dir` が
#: 系列名へ焼く綴りと同じ導出 — `crnn_epoch3.pt` → `crnn-epoch3`）。学習し直した重みは別の
#: モデル名で並ぶ（manifest のキーが世代の正本）。
VOWEL_DETECTOR_DEFAULT_MODEL = "crnn-epoch3"

#: 系列名の接頭辞（`vowel-detector-<モデル名>` — 台本の `MODELS_ROOT.name` と同じ 1 語）。
VOWEL_DETECTOR_PREFIX = "vowel-detector"

#: 上流素材の置き場（`inputs/vowel-detector/` — `vowel_detector.export.MODELS_ROOT` と同じ）。
VOWEL_DETECTOR_INPUTS_DIRNAME = "vowel-detector"

#: グラフの役割名（配布形に CRNN は 1 本きり）。配置表・出力 path・weights 宣言が共有する 1 語。
VOWEL_DETECTOR_GRAPH_ROLE = "crnn"

#: `pipelineConfig` に載る**運用上限**（10ms フレーム数 = 10 分）。焼いたグラフの記号次元
#: `T`（20ms 格子）の上限 `vowel_detector.export.SYM_MAX` を入力側の単位へ直したもので、
#: 一致は `vowel_detector/tests/test_distribution.py` が突き合わせる。
#:
#: MUST: 台本の値と一致させる。IR は記号の値域を持たない（名前だけ）ので、宣言より長い入力を
#: 止められるのは**この数を読むパイプラインだけ**。ずれると超過は配布形の門ではなく利用者の
#: 手元の確保失敗として出る。
VOWEL_DETECTOR_MAX_FRAMES = 60_000

#: グラフ入力の名前（`vowel_detector.export.INPUT_NAME`）と、出力の時間軸の刻み（conv の
#: stride 2 — 入力 2 フレームで出力 1 フレーム）。
VOWEL_DETECTOR_INPUT = "features"
VOWEL_DETECTOR_TIME_STRIDE = 2

#: 時間軸の記号名（`vowel_detector.export` が `convert(symbol_names=("T",))` で焼く綴り）。
#: 入力は `2T`（10ms 格子）・出力は `T`（20ms 格子）で現れる。
VOWEL_DETECTOR_SYMBOL = "T"

#: 特徴の契約の出どころ（上流 `assets/feature_config.json` を `inputs/vowel-detector/` へ手置き）。
#: mel 基底もこの中にあり、配布形へは 1 テンソルの safetensors として移す。
VOWEL_DETECTOR_FEATURE_CONFIG_FILE = "feature_config.json"

#: 配布する mel 基底のテンソルキー（TS 側 `pipeline.ts` の `MEL_BASIS` と対）。
VOWEL_DETECTOR_MEL_BASIS_KEY = "mel_basis"

#: DSP 補助特徴の本数（有声性 / log エネルギー / 零交差率）。`feature_dim = n_mels + 3` の
#: 内部整合を見るためだけに持つ（値の正本は上流の特徴抽出）。
VOWEL_DETECTOR_DSP_DIM = 3

#: 出力クラスの本数（`feature_config.json` の `classes` — 並びは配らない側の関心事で、
#: **並びが id** であることの検査は TS 側 `config.ts` が持つ）。
VOWEL_DETECTOR_CLASS_COUNT = 8


#: 出力の相対 path（**モデルサブツリー内**）— 配置表と manifest が共有する 1 箇所。格納 dtype を
#: ファイル名に出すのは他ファミリと同じ形。
VOWEL_DETECTOR_OUTPUT_PATHS: Mapping[str, str] = {
    VOWEL_DETECTOR_GRAPH_ROLE: "model.f32.safetensors",
    VOWEL_DETECTOR_MEL_BASIS_KEY: f"features/{VOWEL_DETECTOR_MEL_BASIS_KEY}.safetensors",
}

#: 格納 dtype の要求（他ファミリと同じ根拠 — 素の資産が組み立て・ロード・実行を全て通って
#: 参照一致の門まで沈黙した実測事故）。mel 基底はこちらが書く F32 なので載せない。
#:
#: NOTE: 禁止表（{@link assert_storage_absent}）は持たない — 圧縮系列が 1 本も無いので、
#: 「F32 を含む」で系列 × 格納 dtype が一意に決まる。f16 / i8 の席を足すときは**同時に**禁止表も
#: 足す（圧縮系列も適格外の重みを F32 で持つので、存在検査だけでは f32 席へ混入する）。
VOWEL_DETECTOR_STORAGE_REQUIREMENTS: Mapping[str, str] = {VOWEL_DETECTOR_GRAPH_ROLE: "F32"}

#: weights の宣言（dtype ラベル → 役割名）。グラフは 1 本で dtype も 1 つしかないので quant 表は
#: 空でよい（{@link complete_quant_weights} が完全写像へ埋める）。
VOWEL_DETECTOR_WEIGHTS: Mapping[str, Mapping[str, WeightFiles]] = {
    VOWEL_DETECTOR_GRAPH_ROLE: {"f32": WeightFiles(VOWEL_DETECTOR_GRAPH_ROLE)}
}

#: assets の宣言（quant 選択に依存しない無条件ファイル — 特徴抽出の mel 基底 1 本）。
VOWEL_DETECTOR_ASSETS: Mapping[str, str] = {
    VOWEL_DETECTOR_MEL_BASIS_KEY: VOWEL_DETECTOR_MEL_BASIS_KEY
}

VOWEL_DETECTOR_QUANTS: Mapping[str, Any] = {"f32": {"weights": {}, "session": {}}}
VOWEL_DETECTOR_DEFAULT_QUANT = "f32"


def vowel_detector_series_name(model: str) -> str:
    """モデル名 → 系列ディレクトリ名（`vowel_detector.export.default_out_dir` と同じ式）。"""
    return f"{VOWEL_DETECTOR_PREFIX}-{model}"


def vowel_detector_repo_name(model: str) -> str:
    """配布リポ名。**モデル名を含めない** — チェックポイントの世代（`crnn-epoch3`）はリポの
    名前ではなく manifest のキーが綴る事実で、世代が上がるたびにリポが増える形にしない
    （上流の配布も `vowel-detector` 1 リポ）。
    """
    return f"karume-{VOWEL_DETECTOR_PREFIX}"


@dataclass(frozen=True)
class VowelDetectorSources:
    """組み立ての入力。CRNN 1 本の系列と、上流素材の置き場（`inputs/` — 生成物ではない）。

    後者が要るのは特徴の契約と mel 基底を**焼き込まずに導出**するため（読むのは
    `feature_config.json` 1 本だけ）。
    """

    #: 系列ディレクトリ群の親（`outputs/series/`）— 系列名はここから組む。
    series_dir: Path
    #: 上流素材（`feature_config.json` を置いたディレクトリ）。
    model: Path
    #: 系列を引くモデル名（世代）。
    model_name: str

    @property
    def series(self) -> Path:
        return self.series_dir / vowel_detector_series_name(self.model_name)


def vowel_detector_sources(
    series_dir: Path, model: str = VOWEL_DETECTOR_DEFAULT_MODEL
) -> VowelDetectorSources:
    """系列の親ディレクトリ（`outputs/series/`）と `karume.paths` の綴りから入力を引く。"""
    return VowelDetectorSources(
        series_dir=series_dir,
        model=INPUTS_ROOT / VOWEL_DETECTOR_INPUTS_DIRNAME,
        model_name=model,
    )


def vowel_detector_placements(sources: VowelDetectorSources) -> dict[str, Path]:
    """役割名 → 出所のファイル。出力の path は {@link VOWEL_DETECTOR_OUTPUT_PATHS} が持つ。

    この表に無いものは出力へ入らない（系列に並ぶ `io.*.safetensors` はこれで落ちる）。
    mel 基底は配置ではなく変換の出力なので、ここには現れない（SBV2 の表 2 本と同じ形）。
    """
    return {VOWEL_DETECTOR_GRAPH_ROLE: sources.series / "model.safetensors"}


def vowel_detector_feature_config(model_dir: Path) -> Mapping[str, Any]:
    """上流の `feature_config.json` を読む（特徴の契約と mel 基底の唯一の出どころ）。"""
    path = model_dir / VOWEL_DETECTOR_FEATURE_CONFIG_FILE
    if not path.is_file():
        raise DistError(
            f"組み立ての入力が無い: {path}"
            "（上流 vowel-detector の assets/feature_config.json をここへ置く）"
        )
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise DistError(f"{path} が JSON として読めない") from error
    if not isinstance(raw, dict):
        raise DistError(f"{path}: 最上位オブジェクトでない")
    return raw


def _vowel_detector_int(raw: Mapping[str, Any], key: str, where: str) -> int:
    """特徴 config の正整数フィールドを検査して読む。"""
    value = raw.get(key)
    # bool は int の派生。`"n_mels": true` を 1 として通すと寸法の突合が緩む。
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise DistError(f"{where} の {key} が正の整数でない（{value!r}）")
    return value


def vowel_detector_classes(raw: Mapping[str, Any], where: str) -> list[str]:
    """クラス語彙を読む（**並びのまま**配る — 並びがそのままクラス id）。

    NOTE: 語彙そのもの（`a`/`i`/…/`cons` の綴りと並び）はここが持たない。上流 config が唯一の
    出どころで、それを写した 2 つ目の表をここに置くと、片方だけ動いたときに「宣言は通るのに
    ラベルが置換されている」形が黙って作れる。**受理集合の検査はロード側**
    （`packages/models/src/vowel-detector/config.ts`）が持ち、ここは構造だけを見る。
    """
    value = raw.get("classes")
    if not isinstance(value, list) or len(value) != VOWEL_DETECTOR_CLASS_COUNT:
        raise DistError(
            f"{where} の classes が長さ {VOWEL_DETECTOR_CLASS_COUNT} の配列でない（{value!r}）"
        )
    for entry in value:
        if not isinstance(entry, str) or not entry:
            raise DistError(f"{where} の classes に空でない文字列でない要素がある（{value!r}）")
    if len(set(value)) != len(value):
        raise DistError(f"{where} の classes に重複がある（{value!r}）")
    return list(value)


def vowel_detector_mel_basis(raw: Mapping[str, Any], where: str) -> np.ndarray:
    """mel 基底 `[n_mels, n_fft // 2 + 1]` を f32 の行列として読む。

    MUST: 形は `n_mels` と `n_fft` の**両方**から組んだ期待と突き合わせる — 基底がずれても
    特徴は「それらしい別の値」になるだけで、shape も値域も合ったまま最後まで通る。

    MUST: 全行に正の要素があることまで見る。空の三角窓（帯域外へ落ちた mel チャネル）は
    その列を常に `log(1e-5)` に張り付かせるが、80 本の中の 1 本なので目視でも数値でも
    「静かにおかしい」形にしかならない。
    """
    n_mels = _vowel_detector_int(raw, "n_mels", where)
    n_fft = _vowel_detector_int(raw, "n_fft", where)
    expected = (n_mels, n_fft // 2 + 1)
    value = raw.get(VOWEL_DETECTOR_MEL_BASIS_KEY)
    if not isinstance(value, list):
        raise DistError(f"{where} の {VOWEL_DETECTOR_MEL_BASIS_KEY} が配列でない（{type(value)}）")
    try:
        basis = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError) as error:
        raise DistError(
            f"{where} の {VOWEL_DETECTOR_MEL_BASIS_KEY} が f32 行列にならない"
        ) from error
    if basis.shape != expected:
        raise DistError(
            f"{where} の {VOWEL_DETECTOR_MEL_BASIS_KEY} の形が {basis.shape}、"
            f"n_mels / n_fft から組んだ期待は {expected}"
        )
    if not np.isfinite(basis).all():
        raise DistError(f"{where} の {VOWEL_DETECTOR_MEL_BASIS_KEY} に有限でない要素がある")
    empty = [int(row) for row in np.flatnonzero(basis.max(axis=1) <= 0)]
    if empty:
        raise DistError(
            f"{where} の {VOWEL_DETECTOR_MEL_BASIS_KEY} に空の mel チャネルがある（行 {empty}）"
        )
    return basis


def vowel_detector_pipeline_config(feature_config: Mapping[str, Any], where: str) -> dict[str, Any]:
    """`pipelineConfig`（TS 側スキーマの 4 欄）を上流 config と台本の宣言から組む。

    `maxFrames` だけが**焼いたグラフ側の数**（{@link VOWEL_DETECTOR_MAX_FRAMES}）で、
    残り 3 欄は上流 config の逐語。
    """
    sample_rate = _vowel_detector_int(feature_config, "sample_rate", where)
    feature_dim = _vowel_detector_int(feature_config, "feature_dim", where)
    n_mels = _vowel_detector_int(feature_config, "n_mels", where)
    if feature_dim != n_mels + VOWEL_DETECTOR_DSP_DIM:
        raise DistError(
            f"{where} の feature_dim {feature_dim} が n_mels {n_mels} +"
            f" DSP {VOWEL_DETECTOR_DSP_DIM} と違う — 特徴の内訳が上流と食い違っている"
        )
    return {
        "sampleRate": sample_rate,
        "featureDim": feature_dim,
        "classes": vowel_detector_classes(feature_config, where),
        "maxFrames": VOWEL_DETECTOR_MAX_FRAMES,
    }


def assert_vowel_detector_graph(
    placements: Mapping[str, Path], pipeline_config: Mapping[str, Any]
) -> None:
    """CRNN グラフの入出力が `pipelineConfig` と噛み合うことを、配置の前に実測する。

    MUST: 落とさない。特徴次元とクラス数は上流 config 由来、形はグラフ由来で**別々に決まる**
    ので、別の特徴で学習された派生の重みと今の config を組み合わせても、ここまでは何も落ちない。

    MUST: 時間軸が**記号**（入力 `2T` / 出力 `T`）であることまで見る。長さを固定して焼いた
    古い形のグラフは名前も階数も同じで、載せてしまうと「その 1 長でしか動かない配布形」が
    manifest としては成立する（利用者の手元では、その長さ以外の全ての音声で落ちる）。
    """
    path = placements[VOWEL_DETECTOR_GRAPH_ROLE]
    graph = ir_graph(path)
    inputs = graph_inputs(graph, path)
    if tuple(inputs) != (VOWEL_DETECTOR_INPUT,):
        raise DistError(
            f"{path} のグラフ入力が {list(inputs)} で、期待の {[VOWEL_DETECTOR_INPUT]} と違う"
            " — 実行側は名前で束ねるので、綴りが変われば束ねられない"
        )
    if graph.get("symbols") != [VOWEL_DETECTOR_SYMBOL]:
        raise DistError(
            f"{path}: 記号次元が {graph.get('symbols')!r} — 期待は"
            f" [{VOWEL_DETECTOR_SYMBOL!r}]（時間軸 1 本だけが記号）"
        )
    feature_dim = pipeline_config["featureDim"]
    stride_dim = f"{VOWEL_DETECTOR_TIME_STRIDE}{VOWEL_DETECTOR_SYMBOL}"
    expected_input = [1, stride_dim, feature_dim]
    if inputs[VOWEL_DETECTOR_INPUT] != expected_input:
        raise DistError(
            f"{path} の入力 '{VOWEL_DETECTOR_INPUT}' が"
            f" {inputs[VOWEL_DETECTOR_INPUT]!r}、期待は {expected_input!r}"
            " — 10ms 格子の長さは記号 T の 2 倍（batch は静的 1）"
        )
    outputs = graph.get("outputs")
    if not isinstance(outputs, list) or len(outputs) != 1:
        raise DistError(f"{path}: グラフ出力が {outputs!r} — ロジット 1 本だけが要る")
    values = graph.get("values")
    value = values.get(outputs[0]) if isinstance(values, dict) else None
    out_shape = value.get("shape") if isinstance(value, dict) else None
    expected_output = [1, VOWEL_DETECTOR_SYMBOL, len(pipeline_config["classes"])]
    if out_shape != expected_output:
        raise DistError(
            f"{path}: グラフ出力 '{outputs[0]}' の形が {out_shape!r}、期待は {expected_output!r}"
            " — 出力は 20ms 格子（入力 2 フレームで 1 本）× クラス数"
        )


def vowel_detector_plan(
    sources: VowelDetectorSources, model: str = VOWEL_DETECTOR_DEFAULT_MODEL
) -> ModelPlan:
    """母音検出 1 モデルぶんの計画を組む（検査と config の読み取りをここで全部済ませる）。"""
    assert_model_name(model)
    placements = vowel_detector_placements(sources)
    for role, source in placements.items():
        assert_storage(role, source, VOWEL_DETECTOR_STORAGE_REQUIREMENTS)
    feature_config = vowel_detector_feature_config(sources.model)
    where = str(sources.model / VOWEL_DETECTOR_FEATURE_CONFIG_FILE)
    pipeline_config = vowel_detector_pipeline_config(feature_config, where)
    assert_vowel_detector_graph(placements, pipeline_config)
    artifacts = {
        role: Artifact(VOWEL_DETECTOR_OUTPUT_PATHS[role], source=source)
        for role, source in placements.items()
    }
    artifacts[VOWEL_DETECTOR_MEL_BASIS_KEY] = Artifact(
        VOWEL_DETECTOR_OUTPUT_PATHS[VOWEL_DETECTOR_MEL_BASIS_KEY],
        payload=table_payload(
            VOWEL_DETECTOR_MEL_BASIS_KEY, vowel_detector_mel_basis(feature_config, where)
        ),
    )
    return ModelPlan(
        name=model,
        pipeline=VOWEL_DETECTOR_PIPELINE,
        artifacts=artifacts,
        weights=VOWEL_DETECTOR_WEIGHTS,
        assets=VOWEL_DETECTOR_ASSETS,
        quants=complete_quant_weights(VOWEL_DETECTOR_WEIGHTS, VOWEL_DETECTOR_QUANTS),
        default_quant=VOWEL_DETECTOR_DEFAULT_QUANT,
        pipeline_config=pipeline_config,
    )


def vowel_detector_dist_plan(series_dir: Path, model: str) -> ModelPlan:
    """`--series` の親から母音検出 1 モデルの計画を組む（CLI のディスパッチ先）。"""
    return vowel_detector_plan(vowel_detector_sources(series_dir, model), model)


#: `--pipeline vowel-detector` の 1 行（ドライバが core の PIPELINES へ合成する）。
PIPELINE = Pipeline(
    default_model=VOWEL_DETECTOR_DEFAULT_MODEL,
    repo_name=vowel_detector_repo_name,
    plan=vowel_detector_dist_plan,
    # 帰属は 1 通りだけ（上流 1 リポ・1 ライセンス — 学習素材の帰属も重みに紐づいた
    # 1 組）。選択肢が無いので省略で通る。
    card_profiles={"vowel-detector": render_vowel_detector_model_card},
)
