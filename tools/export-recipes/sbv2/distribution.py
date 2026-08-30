"""SBV2 の配布 recipe — 系列レイアウト・出力 path 表・quant 表・カードの選択（ADR 0065 決定 2）。

汎用の組み立てエンジン（配置・共有席の畳み込み・sha256・manifest・staging/swap・検証）は
`karume.dist` が持つ。ここが持つのは **SBV2 固有の事実**だけ: どの系列ディレクトリから何を
拾い、配布形のどの path へ、どの dtype ラベルで並べ、どの quant を既定にするか。

配布するのは**実行に要る 3 グラフ + ホスト資産 4 本**だけ（ADR 0038 §2 の SBV2 例は
5 グラフを並べるが、あれは形の例示）。`front` = enc_p + dp + sdp、`voice` = flow + dec の
融合なので、`dp` / `flow` / `dec` は golden 検証専用の単体グラフで配布形には入らない。
`text_encoder`（DeBERTa）に f16 席が無いのは `deberta/export.py` が f16 を持たないから
（f32 の 1.32GB は配布に非現実的）。既定の i8 は ADR 0026 が聴感ゲート込みで受理済みで、
i4 混成（linear と embedding が i4 group32）は `i8+bert4` quant の席として後から足した
（perf-ledger Q-1 — 2026-08-20 から既定 quant）。`front` / `voice` にも i4 混成席があり
（適格な linear と conv1d が i4 — conv1d は波 J-5b の追補）、3 席とも i4 を選ぶのが `i4`
quant（速度 / サイズ優先の opt-in）。

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
from typing import Any, NamedTuple

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
    assert_storage_absent,
    complete_quant_weights,
    ir_graph,
    table_payload,
)
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
#: の `BASE_WEIGHT_DTYPES` / `I4_MODULE_TYPES`）で、`i8+bert4` quant の text_encoder 席になる。
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

#: 各役割の safetensors ヘッダに**あってはならない**格納 dtype（{@link assert_storage_absent}）。
#: {@link SBV2_STORAGE_REQUIREMENTS} の存在検査は片方向なので、**圧縮席どうしの取り違え**が
#: 素通りする — i4 系列は混成（F32 + I8 + I4）で、i4 適格外の重みは i8 のまま残るので
#: **必ず I8 を含む**。したがって i4 系列を `text_encoder` / `front_i8` / `voice_i8` の i8 席へ
#: 挿し込むと「I8 を含む」を満たしてしまい、組み立ても verify_dist も manifest 検査も全部通る。
#: 出来上がるのは「席名も path も `model.i8.safetensors` なのに中身は i4 混成」という配布形で、
#: i8 席は f32 compute なので実行も例外を出さず、音が i4 の品質で出るだけで沈黙する。
#: MUST: 禁止は**役割ごとに集合**で持つ（1 dtype だけ書くと 4 本目の系列が生えた日に、名指し
#: しなかったほうが黙って素通りする — anima / irodori と同じ規律）。f16 席は I8 / I4 の
#: 両方の不在で二重に締める。
SBV2_STORAGE_FORBIDDEN: Mapping[str, tuple[str, ...]] = {
    "text_encoder": ("I4",),
    "front_f16": ("I8", "I4"),
    "front_i8": ("I4",),
    "voice_f16": ("I8", "I4"),
    "voice_i8": ("I4",),
}

#: weights の宣言（dtype ラベル → 役割名）。dtype キーは ADR 0041 §3 の統一形（v1 の `{file}` /
#: `{variants}` の 2 形は消えた）。どの役割でも `i4` は**混成の系列**を指すラベルで、実体は
#: 「i4 適格な重みが i4 group32・残りは i8」— 適格の範囲は台本ごとに違い、`text_encoder` は
#: linear + 語彙表（`deberta/export.py`）、`front` / `voice` は linear + conv1d
#: （`sbv2/export.py` — groups == 1 かつ行長が group32 で割り切れるもの）。
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

#: 席名の部品上書きトークン → その weights 名（ADR 0074 決定 4 — **略称の定義は recipe が
#: 持ち、生成モデルカードの quant 表に対応を必ず出す**）。`bert` は役割名 `text_encoder` より
#: 短く読みやすいが、それ自体が語彙になるので対応を配布物へ必ず出す。
SBV2_QUANT_ABBREVIATIONS: Mapping[str, str] = {"bert": "text_encoder"}

#: quant 表。MUST: **dtype が 2 つ以上ある役割は全 quant が明示指定する**
#: （{@link complete_quant_weights}）— `text_encoder` に i4 席が生えた時点で、既定を勝手に
#: 選ぶ経路（黙って別の格納形が配られる）は塞がれている。
#:
#: 席名は ADR 0074 の文法 `<格納>[+<部品><ビット>]…[-<ノブ>]…`。SBV2 は全役割が同じ格納を持つ
#: ので基底 1 語で足り、`f16` 席だけ text_encoder が i8 なので `+bert8` が付く（実態に合わせる
#: — f16 席に f16 の text_encoder は無い〈`deberta/export.py` が f16 を持たない〉）。
#: `i8+bert4` は `i8` と同構成で `text_encoder` だけ i4 混成（BERT の linear と語彙表を i4
#: group32 に落とす）。`i4` は**3 席とも i4 混成**（session は空 = f32 compute のまま — 活性は
#: 動かさない）。
#: 数値は f32 同一性の指標では大きく動く。聴感は一次通過だが、`i4` は f32 比でテンションが
#: 少し低め（perf-ledger Q-1 / research 2026-08-19 §6 — net_g の RTN i4 由来）で、これが
#: 既定を `i8+bert4` に戻した根拠（{@link SBV2_DEFAULT_QUANT}）。
#:
#: `label` / `description` は選択 UI 向けの表示欄（ADR 0075 決定 1 — 英語・64 / 200 字上限）。
#: 既定であることは書かない（`defaultQuant` が既に指している — ADR 0075 決定 3）。
#:
#: NOTE: net_g 側の i4 は conv1d 追補（波 J-5b・ADR 0069 追記 7）で**サイズ利得の本体**に
#: なった — 適格 conv1d ≈56MiB が半減する（linear だけの時代は 6 本 = 配布形全体の 0.1% 未満で
#: 利得ほぼゼロだった）。`text_encoder` の削減（Q-1 / J-5a）と合わせ、`i4` は名実ともに
#: 「配布形を丸ごと 4bit 格納で通す席」。
SBV2_QUANTS: Mapping[str, Any] = {
    "f16+bert8": {
        "weights": {"text_encoder": "i8", "front": "f16", "voice": "f16"},
        "session": {},
        "label": "Highest fidelity (f16 synthesis)",
        "description": "front and voice in f16 storage, with the int8 text encoder — the largest"
        " download, and the closest match to the source checkpoint's audio.",
    },
    "i8": {
        "weights": {"text_encoder": "i8", "front": "i8", "voice": "i8"},
        "session": {},
        "label": "Half size (int8)",
        "description": "Every component stored as int8 and computed in f32 — roughly half the f16"
        " download, with the execution path left unchanged.",
    },
    "i8-a8": {
        "weights": {"text_encoder": "i8", "front": "i8", "voice": "i8"},
        "session": {"linearCompute": "a8"},
        "label": "int8 with int8 linear activations",
        "description": "The int8 weights with per-token int8 activations in the linear layers —"
        " faster on GPUs with dp4a, same download as the plain int8 seat.",
    },
    "i8+bert4": {
        "weights": {"text_encoder": "i4", "front": "i8", "voice": "i8"},
        "session": {},
        "label": "Balanced (int8 + int4 text encoder)",
        "description": "int8 synthesis with the text encoder in GPTQ-calibrated int4 (group-32):"
        " a markedly smaller download that stays audibly level with f32.",
    },
    "i4": {
        "weights": {"text_encoder": "i4", "front": "i4", "voice": "i4"},
        "session": {},
        "label": "Smallest (int4)",
        "description": "Text encoder, front and voice all in int4 group-32 — the smallest download"
        " and the fastest warm start, with slightly lower tension than f32.",
    },
}

#: 既定は `i8+bert4`（2026-08-20 ユーザー再裁定 — 既定は**品質最優先**。`i4` は聴感で f32 比の
#: 差が残る〈テンションが少し低め — net_g の RTN i4 由来〉ので、速度 / サイズを取りに行く人が
#: 明示して選ぶ opt-in の席に置く。`i8+bert4` は BERT だけ GPTQ 校正付きの i4 で、聴感は f32
#: とほぼ同一。経緯と数値は perf-ledger Q-1 / Q-6 と research 2026-08-20 §7）。`i8` は
#: opt-in の参照系として残る — WAV 参照門（e2e_sbv2_wav_test）の不変アンカー。
SBV2_DEFAULT_QUANT = "i8+bert4"

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

#: 記号次元の上限を系列へ書き残す出所記録（`sbv2.export` と `deberta.export` が作業席の中で
#: 書く）。綴りは書き手 2 つと読み手 1 つで一致していなければならないが、`deberta` は SBV2 の
#: 消費者ではないので**こちらを import しない**（shared 席は資産の共有であって結合ではない —
#: モジュール doc）。一致は `deberta/tests/test_export.py` が毎回突き合わせる。
EXPORT_PROVENANCE_FILE = "export_provenance.json"


class Sbv2SymExpectation(NamedTuple):
    """席が焼かれているべき記号次元の宣言（台本のターゲット名・IR の記号名・上限）。

    `baked` は「その上限が **artifact のバイトから読めるか**」— front / voice は相対位置の
    添字表を Tmax で焼いて `sym_prefix_slice` で切り出すので読めるが、text_encoder（DeBERTa）は
    その表が ADR 0045 波 3 でグラフ**入力**へ昇格したため、上限を運ぶ焼き込み定数を 1 本も
    持たない（2026-08-30 の出荷 4 コンテナ実測 — `sym_prefix_slice` は 0 本）。読めない席では
    {@link assert_baked_sym_max} の代わりに {@link assert_baked_sym_max_absent} を掛け、
    上限そのものの突合は {@link assert_sym_provenance}（記録）だけが持つ。
    """

    target: str
    symbol: str
    sym_max: int
    baked: bool = True


#: 席 → 記号次元の期待。`sbv2.export` / `deberta.export` の `--sym-max` は任意の値を通す研究用
#: ノブで、渡した値はそのまま `Dim` の max になる。ところが上の {@link SBV2_MAX_TOKENS} /
#: {@link SBV2_MAX_FRAMES} は定数で manifest へ焼かれるので、既定から外して採り直した系列を挿すと
#: **export は緑・配布も緑**のまま `pipelineConfig` だけが嘘になり、消費側で初めて上限超過に
#: 当たる（host 側の `maxFrames` 門は通過する）。2 つの門で塞ぐ:
#: {@link assert_sym_provenance}（書き出した側の記録）と {@link assert_baked_sym_max}
#: （**artifact 自身**の焼き込み定数 — `baked` の席だけ）。
#:
#: MUST: 列挙元は {@link SBV2_WEIGHTS} — dtype 席が 1 本増えた日に、名指ししなかった席だけが
#: 黙って素通りする形にしない。
#:
#: `text_encoder` の上限が {@link SBV2_MAX_TOKENS} なのは、この定数が front の音素次元 P と
#: DeBERTa のトークン次元 T の**両方**の上限を兼務しているから（`deberta.export.SYM_MAX` も同値）。
#: ターゲット名に variant 名を使うのは、`deberta/export.py` に `--target` が無く、どの形として
#: 焼いたかを名乗るのが variant（層数 × 出力形）1 つきりだから — dev-2layer / full-24layer を
#: 22 層の席名のディレクトリへ写した取り違えは、この記録だけが名指しできる。
SBV2_SYM_EXPECTATIONS: Mapping[str, Sbv2SymExpectation] = {
    files.file: expectation
    for name, expectation in (
        (
            "text_encoder",
            Sbv2SymExpectation(SBV2_TEXT_ENCODER_VARIANT, "T", SBV2_MAX_TOKENS, baked=False),
        ),
        ("front", Sbv2SymExpectation("front", "P", SBV2_MAX_TOKENS)),
        ("voice", Sbv2SymExpectation("voice", "T", SBV2_MAX_FRAMES)),
    )
    for files in SBV2_WEIGHTS[name].values()
}


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


def sbv2_ir_graph(path: Path) -> Mapping[str, Any]:
    """配布候補の safetensors ヘッダから IR のグラフ JSON を読む（テンソルは 1 バイトも読まない）。

    グラフを見る門が 2 つある（{@link assert_bert_hidden} の層 / 出力 / 入力と、
    {@link assert_baked_sym_max} の焼き込み次元）ので、読み取りと不整合の名指しはここ 1 本。

    実体は core の {@link karume.dist.ir_graph} をそのまま呼ぶ — 以前はここに同じ読み取りを
    写していたが、配布形が常時分割になった（ADR 0081）ときに**写しの側だけが代表 path を
    直接開いたまま**残った（グラフ shard を名指しで読むのは core 側だけ）。名前を残すのは
    上の 2 門が同じ綴りで引くため。
    """
    return ir_graph(path)


def sbv2_baked_sym_bounds(role: str, path: Path, symbol: str) -> list[int]:
    """焼き込み定数の静的次元から、そのグラフが受理する記号次元の上限を**全部**拾う。

    `sym_prefix_slice` は「Tmax で焼いた静的形の先頭 `coeff·sym+offset` を切り出す」op
    （ADR 0010）で、切り出し元の定数次元がそのまま**受理できる sym の上限**になる
    （超過はランタイムの `shapes.ts` が「Tmax 超過」で落とす）。したがって台本の `--sym-max`
    は資産の中に残っていて、記録を信じずに artifact だけから読める。

    「1 本も無い」を空リストで返して呼び手に判断を預けるのは、席によって意味が逆だから —
    front / voice では恒真化の兆候（{@link assert_baked_sym_max}）、text_encoder では
    **在るべき状態**（{@link assert_baked_sym_max_absent}）。
    """
    graph = sbv2_ir_graph(path)
    values = graph.get("values")
    nodes = graph.get("nodes")
    if not isinstance(values, dict) or not isinstance(nodes, list):
        raise DistError(f"{role}: {path} の IR メタデータに values / nodes が無い")
    bounds: list[int] = []
    for node in nodes:
        if not isinstance(node, dict):
            continue
        attrs = node.get("attrs")
        if node.get("op") != "sym_prefix_slice" or not isinstance(attrs, dict):
            continue
        if attrs.get("sym") != symbol:
            continue
        try:
            shape = values[node["ins"][0]]["shape"]
            for entry in attrs["slices"]:
                bounds.append((shape[entry["dim"]] - entry["offset"]) // entry["coeff"])
        except (KeyError, IndexError, TypeError, ZeroDivisionError) as error:
            raise DistError(f"{role}: {path} の sym_prefix_slice が読めない — {error}") from error
    return bounds


def sbv2_baked_sym_max(role: str, path: Path, symbol: str) -> int:
    """そのグラフが実際に受理する記号次元の上限（複数あれば最も厳しいもの）。

    MUST: 対象ノードが 1 本も無ければ落とす（恒真化の門）— 相対位置の表が入力へ昇格するなど
    グラフの形が変わると、この門は「何も見ずに緑」へ静かに退化する。
    """
    bounds = sbv2_baked_sym_bounds(role, path, symbol)
    if not bounds:
        raise DistError(
            f"{role}: {path} に記号 {symbol!r} の sym_prefix_slice が 1 本も無い —"
            "焼き込み定数から上限を読む門が恒真化している（グラフの形が変わった）"
        )
    return min(bounds)


def assert_baked_sym_max(role: str, path: Path, expectation: Sbv2SymExpectation) -> None:
    """焼いたグラフの記号次元の上限が `pipelineConfig` の宣言と一致することを見る。

    manifest の `maxTokens` / `maxFrames` は「焼いたグラフの記号次元の上限そのもの」と自称して
    いる（{@link SBV2_MAX_TOKENS}）のに、値は定数で焼かれていて artifact を一切見ていなかった
    — その切断をここで閉じる。記録（{@link assert_sym_provenance}）と違い、こちらは**現物の
    バイト**から導くので、記録の無い古い系列にも効く。
    """
    baked = sbv2_baked_sym_max(role, path, expectation.symbol)
    if baked != expectation.sym_max:
        raise DistError(
            f"{role}: {path} は記号 {expectation.symbol!r} を上限 {baked} で焼いているが、"
            f"配布の宣言は {expectation.sym_max}"
            f"（`sbv2.export --target {expectation.target} --sym-max` の非既定値で採った系列）—"
            "export も配布も緑のまま、消費側だけが Tmax 超過で落ちる形になる"
        )


def assert_baked_sym_max_absent(role: str, path: Path, expectation: Sbv2SymExpectation) -> None:
    """`baked=False` の席が、本当に上限を運ぶ焼き込み定数を持たないことを見る。

    text_encoder の上限は artifact から読めない（相対位置の表がグラフ入力 — ADR 0045 波 3）ので、
    突合は記録（{@link assert_sym_provenance}）だけが持ち、記録の無い系列は受理される。その
    「読めない」は**グラフの形の事実**であって恒久の保証ではないので、形が戻った日に席の分類が
    黙って古びる — 表が焼き戻れば `sym_prefix_slice` が生え、そのときは
    {@link assert_baked_sym_max} 側（`baked=True`）へ移すのが正しい。ここが落ちるのはその合図。

    NOTE: 読めない席で唯一 512 に見える静的次元は `rel_embeddings` の `[512, 1024]` だが、これは
    `2 × position_buckets`（= 256）で、`max_position_embeddings` と値が一致しているだけの別物
    （`deberta/patch.py` の対数バケット化）。上限の出どころとして読むと**偶然の一致に門を建てる**
    ことになるので、代わりにここは「上限を運ぶノードが無い」ことだけを見る。
    """
    bounds = sbv2_baked_sym_bounds(role, path, expectation.symbol)
    if bounds:
        raise DistError(
            f"{role}: {path} に記号 {expectation.symbol!r} の sym_prefix_slice が"
            f" {len(bounds)} 本ある（上限 {sorted(set(bounds))}）— この席は"
            "「上限は artifact から読めない」前提で記録だけを突き合わせている。"
            "焼き込みが戻ったなら SBV2_SYM_EXPECTATIONS の baked を真へ移す"
        )


def assert_sym_provenance(role: str, path: Path, expectation: Sbv2SymExpectation) -> None:
    """書き出した側が残した記録（`export_provenance.json`）を宣言と突き合わせる。

    NOTE: 記録が無い系列は受理する。同じ事実を artifact から導く {@link assert_baked_sym_max}
    が**常に**掛かっていて、こちらが上乗せするのは「どのターゲットとして焼いたか」だけ —
    記録の不在を拒否に倒すと、記録が生える前に焼いた全系列へ再 export（= 出荷済み配布形の
    sha が動く作業）を課すことになり、閉じる穴に見合わない。
    """
    record_path = path.parent / EXPORT_PROVENANCE_FILE
    if not record_path.is_file():
        return
    try:
        record = json.loads(record_path.read_text(encoding="utf-8"))
    except ValueError as error:
        raise DistError(f"{role}: 出所記録を解析できない: {record_path} — {error}") from error
    if not isinstance(record, Mapping):
        raise DistError(f"{role}: 出所記録 {record_path} が最上位オブジェクトでない")
    if record.get("target") != expectation.target:
        raise DistError(
            f"{role}: 出所記録の target が {record.get('target')!r} で、この席が要求する"
            f" {expectation.target!r} と違う（別ターゲットの系列を挿している）: {record_path}"
        )
    if record.get("sym_max") != expectation.sym_max:
        raise DistError(
            f"{role}: 出所記録の sym_max が {record.get('sym_max')!r} で、配布の宣言"
            f" {expectation.sym_max} と違う: {record_path}"
        )


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
    graph = sbv2_ir_graph(text_encoder)

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
        assert_storage_absent(role, source, SBV2_STORAGE_FORBIDDEN)
        expectation = SBV2_SYM_EXPECTATIONS.get(role)
        if expectation is not None:
            if expectation.baked:
                assert_baked_sym_max(role, source, expectation)
            else:
                assert_baked_sym_max_absent(role, source, expectation)
            assert_sym_provenance(role, source, expectation)
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
        name: partial(
            render_sbv2_model_card, profile=profile, abbreviations=SBV2_QUANT_ABBREVIATIONS
        )
        for name, profile in SBV2_CARD_PROFILES.items()
    },
)
