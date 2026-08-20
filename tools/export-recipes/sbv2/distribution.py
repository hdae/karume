"""SBV2 の配布 recipe — 系列レイアウト・出力 path 表・quant 表・カードの選択（ADR 0065 決定 2）。

汎用の組み立てエンジン（配置・共有席の畳み込み・sha256・manifest・staging/swap・検証）は
`karume.dist` が持つ。ここが持つのは **SBV2 固有の事実**だけ: どの系列ディレクトリから何を
拾い、配布形のどの path へ、どの dtype ラベルで並べ、どの quant を既定にするか。

配布するのは**実行に要る 3 グラフ + ホスト資産 4 本**だけ（ADR 0038 §2 の SBV2 例は
5 グラフを並べるが、あれは形の例示）。`front` = enc_p + dp + sdp、`voice` = flow + dec の
融合なので、`dp` / `flow` / `dec` は golden 検証専用の単体グラフで配布形には入らない。
`text_encoder`（DeBERTa）に f16 席が無いのは `deberta/export.py` が f16 を持たないから
（f32 の 1.32GB は配布に非現実的）。既定の i8 は ADR 0026 が聴感ゲート込みで受理済みで、
i4 混成（linear と embedding が i4 group32）は `w8-bert4` quant の席として後から足した
（perf-ledger Q-1 — 既定はまだ i8）。`front` / `voice` にも i4 混成席があり（そちらは適格
linear だけが i4）、3 席とも i4 を選ぶのが `w4` quant（perf-ledger Q-1 の full-w4 側）。

ホスト資産のうち `style_vectors` / `speaker_embeddings` は**表を配って実行時に行を引く**形。
`front` / `voice` のグラフ入力 `style_vec[1,256]` / `g[1,512,1]` はこの 2 表から作られ、
名前 → 行の対応は `pipelineConfig` の `styles` / `speakers` が持つ（3 つで 1 組）。

`text_encoder` の出所は `deberta` recipe が書く系列だが、参照するのは**その出力 path だけ**
（コードの import は持たない — shared 席は資産の共有であって結合ではない）。

公開面は {@link PIPELINE} 1 つ（`karume.dist.Pipeline`）— リポの dist ドライバ
（`tools/export-recipes/dist.py`）がこれを core の PIPELINES へ合成する。
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
from safetensors import safe_open

from _shared.paths import INPUTS_ROOT, OUTPUTS_ROOT
from karume.dist import (
    Artifact,
    DistError,
    ModelPlan,
    Pipeline,
    WeightFiles,
    assert_model_name,
    assert_storage,
    complete_quant_weights,
    safetensors_header,
    table_payload,
)
from karume.ir import IR_METADATA_KEY
from sbv2.card import SBV2_CARD_PROFILES, render_sbv2_model_card

#: 既定のモデル名 — 系列の綴り（`sbv2-FN4{,-f16,-i8}`）と実重みの置き場を束ねる 1 語。
#: `sbv2.export.default_out_root` が `--model-dir` のディレクトリ名から系列名を作るので、
#: 読み手のこちらも同じ 1 語から組む。
SBV2_DEFAULT_MODEL = "FN4"

#: 系列名とリポ名の接頭辞（`sbv2-<モデル名>`）。
SBV2_SERIES_PREFIX = "sbv2"

#: パイプライン契約（ADR 0041 §2 — モデル単位）。
SBV2_PIPELINE = "sbv2/1"

#: DeBERTa の系列とその variant ディレクトリ（`deberta/export.py` の綴り）。モデル名に依らない
#: （ファミリー組み立てでは全モデルが同じ text_encoder を指し、`shared/` へ 1 回だけ入る）。
#: 22 層 variant なのは末尾 2 層が SBV2 の経路で死んでいるから（{@link SBV2_TEXT_ENCODER_LAYERS}）。
#: i4 系列は**混成**（linear と語彙表 = i4 group32・conv と相対位置表 = i8 — `deberta/export.py`
#: の `BASE_WEIGHT_DTYPES` / `I4_MODULE_TYPES`）で、`w8-bert4` quant の text_encoder 席になる。
SBV2_TEXT_ENCODER_SERIES = "deberta-i8"
SBV2_TEXT_ENCODER_I4_SERIES = "deberta-i4"
SBV2_TEXT_ENCODER_VARIANT = "sbv2-22layer"

#: 配布 text_encoder に残っている encoder 層の数。SBV2 が使う `hidden_states[-3]`
#: （`[0]` = embedding 出力・`[i+1]` = layer i の出力なので先頭から 22 番目 = layer 21 の出力）を
#: **グラフの最終出力**にするための層数で、参照実装の添字をこの 1 つの数へ解いてある。
SBV2_TEXT_ENCODER_LAYERS = 22

#: 配布 text_encoder のグラフ出力の本数。SBV2 が読むのは 1 本だけで、ランタイムは
#: `graph.outputs` を**全部** readback するため、全層出しのまま配ると毎 run で使わない
#: 22 本ぶんの staging + mapAsync を払う（ADR 0045 波 2 の実測 — T=512 で −10.6%）。
SBV2_TEXT_ENCODER_OUTPUTS = 1

#: 配布 text_encoder のグラフ入力の並び（`deberta.export.INPUT_ORDER` と同じ）。相対位置の
#: 添字表 2 本が**入力に居ること**が波 3 の成果そのもので、焼き込みへ戻ると 2MiB の死荷重が
#: 復活する（値は正しいままなので E2E では捕まらない）。
SBV2_TEXT_ENCODER_INPUTS: tuple[str, ...] = (
    "input_ids",
    "attention_mask",
    "c2p_pos",
    "p2c_pos",
)

#: initializer 名から encoder の層番号を拾う（`p_model_encoder_layer_<i>_...` — torch.export が
#: FQN を正規化した綴り）。層数の門はこれで数える。
SBV2_LAYER_PATTERN = re.compile(r"layer[._](\d+)[._]")

#: `sbv2.demo assets` が書くホスト資産の置き場と綴り。系列（IR + io）ではないので
#: `outputs/series/` の下ではない。
SBV2_DEMO_DIRNAME = "sbv2-demo"
SBV2_SYMBOLS_FILE = "symbols.json"
SBV2_TOKENIZER_FILE = "deberta-tokenizer.json"

#: 実重みと config の置き場（`sbv2.export.DEFAULT_MODEL_DIR` と同じ場所）。
SBV2_CONFIG_FILE = "config.json"
SBV2_STYLE_FILE = "style_vectors.npy"

#: 話者埋め込みの出所（ckpt のテンソルキー）。`front` / `voice` はどちらも `g[1,512,1]` を
#: グラフ入力に取るので、**この表が無いと配布形だけではグラフを実行できない**。
SBV2_SPEAKER_TENSOR = "emb_g.weight"

#: 配布する表のテンソルキー（`.npy` / ckpt を 1 テンソルの safetensors へ移すときの唯一のキー）。
SBV2_STYLE_KEY = "style_vectors"
SBV2_SPEAKER_KEY = "speaker_embeddings"

#: 出力の相対 path（**モデルサブツリー内**）— 配置表・変換先・manifest が共有する 1 箇所。
#: `style_vectors` / `speaker_embeddings` だけは配置ではなく変換の出力なので
#: {@link sbv2_placements} には現れない。
#: 役割名が dtype 接尾辞を持たない `text_encoder` だけ i8 席の綴りのままなのは、2 つ目の
#: 格納形（i4 混成）が後から生えた席だから — 既存席の役割名を動かすと配布形の path も
#: 動くので、増える側にだけ接尾辞を付ける。
SBV2_OUTPUT_PATHS: Mapping[str, str] = {
    "text_encoder": "text_encoder/model.i8.safetensors",
    "text_encoder_i4": "text_encoder/model.i4.safetensors",
    "front_f16": "front/model.f16.safetensors",
    "front_i8": "front/model.i8.safetensors",
    "front_i4": "front/model.i4.safetensors",
    "voice_f16": "voice/model.f16.safetensors",
    "voice_i8": "voice/model.i8.safetensors",
    "voice_i4": "voice/model.i4.safetensors",
    "tokenizer": "tokenizer/deberta-tokenizer.json",
    "symbols": "text/symbols.json",
    "style_vectors": "styles/style_vectors.safetensors",
    "speaker_embeddings": "speakers/speaker_embeddings.safetensors",
}

#: text_encoder の席（格納形ごとに 1 本）。MUST: 門 {@link assert_bert_hidden} は**席ごとに**
#: 掛ける — 2 本は別々に export された独立のコンテナなので、片方だけ層数・出力形・入力の並びが
#: ずれた配布形が普通に組み上がる（ずれても shape は合ったまま実行は通り、別の層の BERT 特徴で
#: 音が出るだけで沈黙する）。
SBV2_TEXT_ENCODER_ROLES: tuple[str, ...] = ("text_encoder", "text_encoder_i4")

#: 格納 dtype の要求（Anima の {@link STORAGE_REQUIREMENTS} と同じ根拠 — 素の F32 資産が
#: 組み立て・ロード・実行を全て通って参照一致の門まで沈黙した実測事故）。tokenizer / symbols
#: （JSON）と style_vectors（こちらが書く F32）はここに載せない。
#:
#: `text_encoder` は i8 系列なので I8、`text_encoder_i4` / `front_i4` / `voice_i4` は混成
#: （F32 + I8 + I4 が同居する）なので **I4 を要求する** — {@link assert_storage} は「要求 dtype が
#: ヘッダに在るか」の片方向検査なので、集合の中で 2 つの系列を区別できる唯一の dtype を名指し
#: する。I8 を要求しても i8 系列が素通りしてしまい、席の取り違えが沈黙する（i4 席に i8 系列が
#: 入ると、サイズだけが元に戻った配布形が層数も形も合ったまま組み上がる）。
SBV2_STORAGE_REQUIREMENTS: Mapping[str, str] = {
    "text_encoder": "I8",
    "text_encoder_i4": "I4",
    "front_f16": "F16",
    "front_i8": "I8",
    "front_i4": "I4",
    "voice_f16": "F16",
    "voice_i8": "I8",
    "voice_i4": "I4",
}

#: weights の宣言（dtype ラベル → 役割名）。dtype キーは ADR 0041 §3 の統一形（v1 の `{file}` /
#: `{variants}` の 2 形は消えた）。どの役割でも `i4` は**混成の系列**を指すラベルで、実体は
#: 「i4 適格な重みが i4 group32・残りは i8」— 適格の範囲は台本ごとに違い、`text_encoder` は
#: linear + 語彙表（`deberta/export.py`）、`front` / `voice` は linear だけ（`sbv2/export.py`）。
SBV2_WEIGHTS: Mapping[str, Mapping[str, WeightFiles]] = {
    "text_encoder": {"i8": WeightFiles("text_encoder"), "i4": WeightFiles("text_encoder_i4")},
    "front": {
        "f16": WeightFiles("front_f16"),
        "i8": WeightFiles("front_i8"),
        "i4": WeightFiles("front_i4"),
    },
    "voice": {
        "f16": WeightFiles("voice_f16"),
        "i8": WeightFiles("voice_i8"),
        "i4": WeightFiles("voice_i4"),
    },
}

#: assets の宣言（quant 選択に依存しない無条件ファイル）。
SBV2_ASSETS: Mapping[str, str] = {
    "tokenizer": "tokenizer",
    "symbols": "symbols",
    "style_vectors": "style_vectors",
    "speaker_embeddings": "speaker_embeddings",
}

#: quant 表。MUST: **dtype が 2 つ以上ある役割は全 quant が明示指定する**
#: （{@link complete_quant_weights}）— `text_encoder` に i4 席が生えた時点で、既定を勝手に
#: 選ぶ経路（黙って別の格納形が配られる）は塞がれている。
#:
#: `w8-bert4` は `w8` と同構成で `text_encoder` だけ i4 混成（BERT の linear と語彙表を i4
#: group32 に落とす）。`w4` は**3 席とも i4 混成**（session は空 = f32 compute のまま — 活性は
#: 動かさない）。
#: 数値は f32 同一性の指標では大きく動くが、聴感は一次通過（perf-ledger Q-1 /
#: research 2026-08-19 §6 — net_g 全役割 rtn で明らかな劣化なし）。
#:
#: NOTE: net_g 側の i4 の**サイズ利得はほぼ無い**（適格 linear は front 2 本 + voice 4 本だけ
#: で、配布形全体の 0.1% 未満 — research 2026-08-19 §3）。`w4` の意味は「配布形を丸ごと 4bit
#: 格納で通す席」で、取得量の削減はほぼ `text_encoder` が担う。
SBV2_QUANTS: Mapping[str, Any] = {
    "f16": {"weights": {"text_encoder": "i8", "front": "f16", "voice": "f16"}, "session": {}},
    "w8": {"weights": {"text_encoder": "i8", "front": "i8", "voice": "i8"}, "session": {}},
    "w8a8": {
        "weights": {"text_encoder": "i8", "front": "i8", "voice": "i8"},
        "session": {"linearCompute": "i8a8"},
    },
    "w8-bert4": {"weights": {"text_encoder": "i4", "front": "i8", "voice": "i8"}, "session": {}},
    "w4": {"weights": {"text_encoder": "i4", "front": "i4", "voice": "i4"}, "session": {}},
}

#: 既定は `w4`（2026-08-20 ユーザー裁定 — GPTQ 校正付き丸めの結線後、聴感で「ほぼ違いが
#: 分からない」+ 速度勝利〈取得 −30%・ロード 1.7 倍速・温間合成 ~4% 速〉。経緯と数値は
#: perf-ledger Q-1 / Q-6 と research 2026-08-20 §7）。`w8` は opt-in の参照系として残す —
#: 既定化前の既定で、WAV 参照門（e2e_sbv2_wav_test）の不変アンカーでもある。
SBV2_DEFAULT_QUANT = "w4"

#: `pipelineConfig.defaults` に載る実行時ノブ（`style_bert_vits2.constants` 由来）。綴りは
#: `symbols.json` の `defaults` と共有する — 同じ源から引いた同じ値が配布形の 2 つの資産に
#: 並ぶので、食い違いは組み立てで落とす（{@link sbv2_knob_defaults}）。
SBV2_KNOB_KEYS: tuple[str, ...] = (
    "style",
    "styleWeight",
    "sdpRatio",
    "noiseScale",
    "noiseScaleW",
    "lengthScale",
)

#: `pipelineConfig` に載る**運用上限**。焼いたグラフの記号次元の上限そのもので、
#: `maxTokens` = DeBERTa のトークン列 T（`deberta.export.SYM_MAX`。front の音素次元 P の上限
#: `sbv2.export.SYM_MAX` も同値）、`maxFrames` = flow / voice のフレーム次元 T
#: （`sbv2.export.FLOW_SYM_MAX`）。
#:
#: MUST: 台本の値と一致させる（`sbv2/tests/test_distribution.py` が突き合わせる）。相対位置の表は
#: ADR 0045 でホストへ外出しされ **T×T の確保はホスト側**（8·T² bytes 級）になったので、
#: 割当上限を知る術が配布形にしか無い — ずれると「宣言は通るのにグラフの表が足りない」
#: 形で利用者の手元でしか出ない。
SBV2_MAX_TOKENS = 512
SBV2_MAX_FRAMES = 4096


def sbv2_series_name(model: str) -> str:
    """系列名の幹（`outputs/series/<この名前>-{f16,i8}/`）。

    綴りは `sbv2.export.default_out_root` と同一 — 書き手と読み手が同じ 1 語から組む。
    """
    return f"{SBV2_SERIES_PREFIX}-{model}"


def sbv2_repo_name(model: str) -> str:
    """単一モデルの配布リポ名（`<DIST_ROOT>/<この名前>/` が 1 つの HF リポになる）。

    `karume-` を前置する（リポ名裁定 2026-08-09 — HF org を作らない代わりに配布リポは
    `karume-` prefix で名前空間を切る）。系列名（{@link sbv2_series_name}）には掛からない。
    """
    return f"karume-{SBV2_SERIES_PREFIX}-{model}"


@dataclass(frozen=True)
class Sbv2Sources:
    """組み立ての入力。系列 2 本のほかに、系列でない 2 つの置き場を跨ぐ。

    `demo` は `sbv2.demo assets` が書くホスト資産（`outputs/` 直下 — 系列ではない）、
    `model` は ckpt と `config.json` / `style_vectors.npy`（`inputs/` — 生成物ではない）。
    どちらも `--series` の下に無いので、系列の親から機械的に導けるのは前 3 つだけ。
    """

    series_f16: Path
    series_i8: Path
    series_i4: Path
    text_encoder: Path
    text_encoder_i4: Path
    demo: Path
    model: Path


def sbv2_sources(series_dir: Path, model: str = SBV2_DEFAULT_MODEL) -> Sbv2Sources:
    """系列の親ディレクトリ（`outputs/series/`）と `_shared.paths` の綴りから入力を引く。"""
    return Sbv2Sources(
        series_f16=series_dir / f"{sbv2_series_name(model)}-f16",
        series_i8=series_dir / f"{sbv2_series_name(model)}-i8",
        series_i4=series_dir / f"{sbv2_series_name(model)}-i4",
        text_encoder=series_dir / SBV2_TEXT_ENCODER_SERIES / SBV2_TEXT_ENCODER_VARIANT,
        text_encoder_i4=series_dir / SBV2_TEXT_ENCODER_I4_SERIES / SBV2_TEXT_ENCODER_VARIANT,
        demo=OUTPUTS_ROOT / SBV2_DEMO_DIRNAME,
        model=INPUTS_ROOT / SBV2_SERIES_PREFIX / model,
    )


def sbv2_placements(sources: Sbv2Sources) -> dict[str, Path]:
    """役割名 → 出所のファイル。出力の path は {@link SBV2_OUTPUT_PATHS} が持つ。

    この表に無いものは出力へ入らない（`io.*.safetensors` を落とす仕掛けはこれで足りる）。
    `style_vectors`（`.npy` → safetensors）と `speaker_embeddings`（ckpt の 1 テンソル →
    safetensors）は配置ではなく**変換**なのでここには現れない。
    """
    return {
        "text_encoder": sources.text_encoder / "model.safetensors",
        "text_encoder_i4": sources.text_encoder_i4 / "model.safetensors",
        "front_f16": sources.series_f16 / "front" / "model.safetensors",
        "front_i8": sources.series_i8 / "front" / "model.safetensors",
        "front_i4": sources.series_i4 / "front" / "model.safetensors",
        "voice_f16": sources.series_f16 / "voice" / "model.safetensors",
        "voice_i8": sources.series_i8 / "voice" / "model.safetensors",
        "voice_i4": sources.series_i4 / "voice" / "model.safetensors",
        "tokenizer": sources.demo / SBV2_TOKENIZER_FILE,
        "symbols": sources.demo / SBV2_SYMBOLS_FILE,
    }


def sbv2_knob_defaults(symbols_path: Path) -> dict[str, Any]:
    """実行時ノブの既定を `style_bert_vits2.constants` から引く（定数を写経しない）。

    `sbv2.demo.jp_extra_rules` が `symbols.json` の `defaults` へ焼くのと**同じ源・同じ値**。
    配布形には両方が並ぶので、食い違いをここで落とす — TS 側は `symbols.json` からノブを
    読み（`parseJpExtraRules`）、hub 利用側は `karume.json` の `pipelineConfig.defaults` を
    読むため、ずれると「どちらの既定で鳴ったのか」が沈黙で分かれる。

    NOTE: `style_bert_vits2` は optional な `sbv2` dependency-group なので import は関数内。
    Anima の組み立てと `karume.dist` の import 自体はこの依存に触れない。
    """
    from style_bert_vits2.constants import (
        DEFAULT_LENGTH,
        DEFAULT_NOISE,
        DEFAULT_NOISEW,
        DEFAULT_SDP_RATIO,
        DEFAULT_STYLE,
        DEFAULT_STYLE_WEIGHT,
    )

    knobs: dict[str, Any] = {
        "style": DEFAULT_STYLE,
        "styleWeight": DEFAULT_STYLE_WEIGHT,
        "sdpRatio": DEFAULT_SDP_RATIO,
        "noiseScale": DEFAULT_NOISE,
        "noiseScaleW": DEFAULT_NOISEW,
        "lengthScale": DEFAULT_LENGTH,
    }
    if not symbols_path.is_file():
        raise DistError(f"組み立ての入力が無い: {symbols_path}")
    shipped = json.loads(symbols_path.read_text(encoding="utf-8")).get("defaults")
    if not isinstance(shipped, dict):
        raise DistError(f"{symbols_path}: 'defaults' 節が無い（実行時ノブの写しの正本）")
    disagreed = [
        f"{key}: constants={knobs[key]!r} / symbols.json={shipped.get(key)!r}"
        for key in SBV2_KNOB_KEYS
        if shipped.get(key) != knobs[key]
    ]
    if disagreed:
        raise DistError(
            f"{symbols_path} の defaults が style_bert_vits2 の定数と食い違う"
            f"（{', '.join(disagreed)}）— 資産を焼いたときと今の package が別版。"
            "`sbv2.demo assets` を採り直す"
        )
    return knobs


def assert_bert_hidden(text_encoder: Path, symbols_path: Path) -> None:
    """配布 text_encoder が「SBV2 が使う層の出力を 1 本だけ出す」形であることを検査する。

    正しい組み合わせは **22 層 × 出力 1 本 × `bertHiddenFromEnd` 1** の 1 通りしかないが、
    層数と出力形は `deberta/export.py` の variant が、取り出し位置は `sbv2/demo.py` の定数が
    持つ**別々の台本**なので、片方だけ動いた配布形が普通に組み上がってしまう。

    MUST: ずれても shape は合ったままロードも実行も通り、**別の層の BERT 特徴で音が出る**
    だけで沈黙する（スタイル表・話者表の行数門と同じ機序）。3 つを別々に見るのは、それぞれが
    別の取り違えを捕まえるため — 層数は「どの層の出力か」、出力本数は「検証用の全層出し資産が
    混ざっていないか」、位置は「symbols.json だけ古いか」。
    """
    header = safetensors_header(text_encoder)
    metadata = header.get("__metadata__")
    if not isinstance(metadata, dict) or IR_METADATA_KEY not in metadata:
        raise DistError(f"{text_encoder}: IR メタデータ（{IR_METADATA_KEY}）が無い")
    try:
        graph = json.loads(metadata[IR_METADATA_KEY])
    except json.JSONDecodeError as error:
        raise DistError(f"{text_encoder}: IR メタデータが JSON として読めない") from error
    if not isinstance(graph, dict):
        raise DistError(f"{text_encoder}: IR メタデータが最上位オブジェクトでない")

    initializers = graph.get("initializers")
    if not isinstance(initializers, dict) or not initializers:
        raise DistError(f"{text_encoder}: IR メタデータに非空の initializers が無い")
    layers = {match.group(1) for name in initializers if (match := SBV2_LAYER_PATTERN.search(name))}
    if len(layers) != SBV2_TEXT_ENCODER_LAYERS:
        raise DistError(
            f"{text_encoder} の encoder は {len(layers)} 層で、期待の"
            f" {SBV2_TEXT_ENCODER_LAYERS} 層でない — SBV2 が使う hidden_states[-3] を最終出力に"
            "するには 22 層で切り詰めた variant が要る（deberta/export.py の VARIANTS）"
        )

    outputs = graph.get("outputs")
    if not isinstance(outputs, list) or len(outputs) != SBV2_TEXT_ENCODER_OUTPUTS:
        count = len(outputs) if isinstance(outputs, list) else outputs
        raise DistError(
            f"{text_encoder} のグラフ出力が {count} 本で、配布形が要求する"
            f" {SBV2_TEXT_ENCODER_OUTPUTS} 本でない — 全層出し（検証用）の資産が混ざっている"
        )

    inputs = graph.get("inputs")
    names = (
        tuple(item.get("name") for item in inputs if isinstance(item, dict))
        if isinstance(inputs, list)
        else ()
    )
    if names != SBV2_TEXT_ENCODER_INPUTS:
        raise DistError(
            f"{text_encoder} のグラフ入力が {list(names)} で、期待の"
            f" {list(SBV2_TEXT_ENCODER_INPUTS)} と違う — 相対位置の添字表が入力から外れると"
            "Tmax ぶんの定数（2MiB）が焼き戻る（値は正しいままなので E2E では捕まらない）"
        )

    shipped = json.loads(symbols_path.read_text(encoding="utf-8"))
    from_end = shipped.get("bertHiddenFromEnd") if isinstance(shipped, dict) else None
    if from_end != SBV2_TEXT_ENCODER_OUTPUTS:
        raise DistError(
            f"{symbols_path} の bertHiddenFromEnd={from_end!r} が、出力 1 本のグラフで唯一"
            f" 意味を持つ値 {SBV2_TEXT_ENCODER_OUTPUTS} でない — `sbv2.demo assets` を採り直す"
        )


def sbv2_config(model_dir: Path) -> Mapping[str, Any]:
    """`config.json` を読む（styles / speakers / 表の行数と列数の正本）。"""
    path = model_dir / SBV2_CONFIG_FILE
    if not path.is_file():
        raise DistError(f"組み立ての入力が無い: {path}")
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise DistError(f"{path}: JSON として読めない") from error
    if not isinstance(config, dict):
        raise DistError(f"{path}: 最上位がオブジェクトでない")
    return config


def _sbv2_section(config: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    """`config.json` の 1 節（`data` / `model`）を検査して読む。"""
    section = config.get(name)
    if not isinstance(section, dict):
        raise DistError(f"config.json に '{name}' 節が無い（実際: {section!r}）")
    return section


def _sbv2_id_map(data: Mapping[str, Any], key: str) -> dict[str, int]:
    """`config.json` の `data.<key>`（名前 → ID の非空マップ）を検査して読む。"""
    table = data.get(key)
    if not isinstance(table, dict) or not table:
        raise DistError(f"config.json の data.{key} が非空のマップでない（実際: {table!r}）")
    for name, value in table.items():
        if not isinstance(value, int) or isinstance(value, bool):
            raise DistError(f"config.json の data.{key}[{name!r}] が整数でない（{value!r}）")
    return dict(table)


def _sbv2_int(section: Mapping[str, Any], key: str, where: str) -> int:
    """`config.json` の整数フィールド（表の行数 / 列数の宣言）を検査して読む。"""
    value = section.get(key)
    # bool は int の派生。`"n_speakers": true` を 1 として通すと表の行数門が緩む。
    if not isinstance(value, int) or isinstance(value, bool):
        raise DistError(f"config.json の {where}.{key} が整数でない（{value!r}）")
    return value


def sbv2_pipeline_config(config: Mapping[str, Any], knobs: Mapping[str, Any]) -> dict[str, Any]:
    """`pipelineConfig` を config.json と実行時ノブから組む（表を焼き込まない）。

    MUST: `styles` / `speakers` はハードコードしない。ckpt が変われば名前も並びも変わり
    （この FN4 は Neutral / high / low / NSFW の 4 つ、別の ckpt は Neutral / Angry / … の
    7 つ）、写した表を配ると **shape は合ったまま別のスタイルの声が出る**。`defaults` の
    数値も同じ理由で `style_bert_vits2` から引いた値（{@link sbv2_knob_defaults}）を受ける。

    `defaults.speaker` は `spk2id` の先頭キー（`sbv2.demo.resolve_style_and_speaker` と同式）。
    `speakers` の名前 → 行の解決先は配布形の `speaker_embeddings`
    （{@link sbv2_speaker_embeddings}）、`styles` の解決先は `style_vectors`。

    `maxTokens` / `maxFrames` は ckpt に無い**焼いたグラフ側の数**なので
    {@link SBV2_MAX_TOKENS} / {@link SBV2_MAX_FRAMES} から載せる。
    """
    data = _sbv2_section(config, "data")
    styles = _sbv2_id_map(data, "style2id")
    speakers = _sbv2_id_map(data, "spk2id")
    missing = [key for key in SBV2_KNOB_KEYS if key not in knobs]
    if missing:
        raise DistError(f"実行時ノブの既定が足りない: {missing}")
    if knobs["style"] not in styles:
        raise DistError(
            f"既定スタイル {knobs['style']!r} が config の style2id {sorted(styles)} に無い"
            " — 存在しないスタイル名を既定に据えた配布形は起動時にしか落ちない"
        )
    return {
        "styles": styles,
        "speakers": speakers,
        "maxTokens": SBV2_MAX_TOKENS,
        "maxFrames": SBV2_MAX_FRAMES,
        "defaults": {
            "speaker": next(iter(speakers)),
            **{key: knobs[key] for key in SBV2_KNOB_KEYS},
        },
    }


def sbv2_style_vectors(model_dir: Path, config: Mapping[str, Any]) -> np.ndarray:
    """`style_vectors.npy` を検査して f32 の `[スタイル数, 256]` として読む。

    MUST: 行数が `data.num_styles` と `len(data.style2id)` の**両方**に一致すること。
    スタイルの ID は行番号そのものなので、行と名前がずれてもロードも実行も通り、
    **別のスタイルの声が出る**だけで沈黙する（表の行数を合わせる以外に検出手段がない）。
    """
    data = _sbv2_section(config, "data")
    path = model_dir / SBV2_STYLE_FILE
    if not path.is_file():
        raise DistError(f"組み立ての入力が無い: {path}")
    table = np.load(path)
    if table.ndim != 2:
        raise DistError(f"{path}: 形 {table.shape} が [スタイル数, 256] でない")
    styles = _sbv2_id_map(data, "style2id")
    num_styles = _sbv2_int(data, "num_styles", "data")
    if table.shape[0] != num_styles or table.shape[0] != len(styles):
        raise DistError(
            f"{path}: 行数 {table.shape[0]} が config の num_styles {num_styles} /"
            f" style2id {len(styles)} 件と一致しない — スタイル ID は行番号なので、"
            "ずれたまま配ると別のスタイルの声が出る"
        )
    return np.ascontiguousarray(table, dtype=np.float32)


def sbv2_ckpt(model_dir: Path) -> Path:
    """実重みの ckpt。`*.safetensors` の一意存在を要求する（`sbv2.export.load_net_g` と同じ）。

    複数あると「どれから話者埋め込みを引いたか」が黙って変わる。
    """
    ckpts = sorted(model_dir.glob("*.safetensors"))
    if len(ckpts) != 1:
        raise DistError(
            f"{model_dir} の ckpt が一意でない（{len(ckpts)} 件: {[p.name for p in ckpts]}）"
        )
    return ckpts[0]


def sbv2_speaker_embeddings(model_dir: Path, config: Mapping[str, Any]) -> np.ndarray:
    """ckpt の `emb_g.weight` を f32 の `[話者数, gin_channels]` として引く。

    `front` / `voice` はどちらも話者埋め込み `g` を**グラフ入力**に取るので、この表が無いと
    配布形だけではグラフを実行できない（デモ経路は `assets.safetensors` に焼いていた）。
    `style_vectors` と同じく「表を配って実行時に行を引く」形にする。

    MUST: 行数が `data.n_speakers` と `len(data.spk2id)` の**両方**に一致すること —
    話者 ID は行番号そのものなので、ずれてもロードも実行も通り、**別の話者の声が出る**
    だけで沈黙する（スタイル表の行数門と同じ機序）。列数は `model.gin_channels` に一致
    すること（こちらは config から導出できる値なので shape ごと縛れる）。

    MUST: ckpt は 251MB 級。`safe_open` の遅延読みで**このテンソル 1 本だけ**を引く
    （`load_file` は全量を numpy へ展開する）。
    """
    data = _sbv2_section(config, "data")
    model = _sbv2_section(config, "model")
    speakers = _sbv2_id_map(data, "spk2id")
    num_speakers = _sbv2_int(data, "n_speakers", "data")
    gin_channels = _sbv2_int(model, "gin_channels", "model")
    ckpt = sbv2_ckpt(model_dir)
    with safe_open(str(ckpt), framework="np") as handle:
        # `keys()` はヘッダのテンソル名一覧（dict ではない）。`get_tensor` はそのテンソルの
        # バイト範囲だけを読むので、251MB の ckpt から 2KB を引くのにファイル全量は載らない。
        available = handle.keys()
        if SBV2_SPEAKER_TENSOR not in available:
            raise DistError(
                f"{ckpt} に {SBV2_SPEAKER_TENSOR} が無い — 話者埋め込みの出所が変わった"
            )
        table = handle.get_tensor(SBV2_SPEAKER_TENSOR)
    if table.ndim != 2:
        raise DistError(f"{ckpt}: {SBV2_SPEAKER_TENSOR} の形 {table.shape} が 2 次元でない")
    if table.shape[0] != num_speakers or table.shape[0] != len(speakers):
        raise DistError(
            f"{ckpt}: {SBV2_SPEAKER_TENSOR} の行数 {table.shape[0]} が config の"
            f" n_speakers {num_speakers} / spk2id {len(speakers)} 件と一致しない —"
            "話者 ID は行番号なので、ずれたまま配ると別の話者の声が出る"
        )
    if table.shape[1] != gin_channels:
        raise DistError(
            f"{ckpt}: {SBV2_SPEAKER_TENSOR} の列数 {table.shape[1]} が config の"
            f" gin_channels {gin_channels} と一致しない — グラフ入力 g の幅と食い違う"
        )
    return np.ascontiguousarray(table, dtype=np.float32)


def sbv2_plan(
    sources: Sbv2Sources, knobs: Mapping[str, Any], model: str = SBV2_DEFAULT_MODEL
) -> ModelPlan:
    """SBV2 1 モデルぶんの計画を組む（検査と表の読み取りをここで全部済ませる）。

    ノブの既定を引数で受けるのは、値の**出所**（`style_bert_vits2` の定数 — optional な
    dependency-group）と配布形の**組み立て**を分けるため。出所の解決は
    {@link sbv2_knob_defaults} が持つ。
    """
    assert_model_name(model)
    placements = sbv2_placements(sources)
    config = sbv2_config(sources.model)
    pipeline_config = sbv2_pipeline_config(config, knobs)
    style_vectors = sbv2_style_vectors(sources.model, config)
    speaker_embeddings = sbv2_speaker_embeddings(sources.model, config)
    for role, source in placements.items():
        assert_storage(role, source, SBV2_STORAGE_REQUIREMENTS)
    for role in SBV2_TEXT_ENCODER_ROLES:
        assert_bert_hidden(placements[role], placements["symbols"])
    artifacts = {
        role: Artifact(SBV2_OUTPUT_PATHS[role], source=source)
        for role, source in placements.items()
    }
    artifacts["style_vectors"] = Artifact(
        SBV2_OUTPUT_PATHS["style_vectors"],
        payload=table_payload(SBV2_STYLE_KEY, style_vectors),
    )
    artifacts["speaker_embeddings"] = Artifact(
        SBV2_OUTPUT_PATHS["speaker_embeddings"],
        payload=table_payload(SBV2_SPEAKER_KEY, speaker_embeddings),
    )
    return ModelPlan(
        name=model,
        pipeline=SBV2_PIPELINE,
        artifacts=artifacts,
        weights=SBV2_WEIGHTS,
        assets=SBV2_ASSETS,
        quants=complete_quant_weights(SBV2_WEIGHTS, SBV2_QUANTS),
        default_quant=SBV2_DEFAULT_QUANT,
        pipeline_config=pipeline_config,
    )


def sbv2_dist_plan(series_dir: Path, model: str) -> ModelPlan:
    """`--series` の親から SBV2 1 モデルの計画を組む（CLI のディスパッチ先）。"""
    sources = sbv2_sources(series_dir, model)
    return sbv2_plan(sources, sbv2_knob_defaults(sources.demo / SBV2_SYMBOLS_FILE), model)


#: `--pipeline sbv2` の 1 行（ドライバが core の PIPELINES へ合成する）。
PIPELINE = Pipeline(
    default_model=SBV2_DEFAULT_MODEL,
    repo_name=sbv2_repo_name,
    plan=sbv2_dist_plan,
    card_profiles={
        name: partial(render_sbv2_model_card, profile=profile)
        for name, profile in SBV2_CARD_PROFILES.items()
    },
)
