"""Irodori の配布 recipe — 系列レイアウト・出力 path 表・quant 表・カードの選択（ADR 0065 決定 2）。

汎用の組み立てエンジン（配置・共有席の畳み込み・sha256・manifest・staging/swap・検証）は
`karume.dist` が持つ。ここが持つのは **Irodori 固有の事実**だけ: どの系列ディレクトリから何を
拾い、配布形のどの path へ、どの dtype ラベルで並べ、どの quant を既定にするか。

配布するのは**実行に要る 8 グラフ + tokenizer 資産 1 本**だけ。8 のうち 2 本は波形 ↔ latent の
コーデック（DACVAE — 上流では別リポ・別重みだが、テキストから音声まで 1 リポで完走させるため
ここへ同梱する）。格納形は f32 / f16 / i8 の 3 系列（`dit` だけ **i4 の 4 本目**を持つ）で、
quant 席は 5 つ（`f32` / `f16` / `w8` / `w8a8` / `w4`）。`w8` と `w8a8` は**同じ i8 バイトを
共有**し、違うのは実行形ノブだけ。`w4` は `dit` だけを i4 系列から採り、他 7 役は `w8` と
同じ i8 バイトを共有する（唯一の混成席 — {@link IRODORI_QUANT_SEATS} の裁定）。

`pipelineConfig` は 2 系統に割れる: **モデル固有の数**（条件 state の宣言長・話者行数・
latent 幅・t_embed 幅）はチェックポイントの config から導出し、**実行時ノブ**（step 数・
CFG の強さと区間・秒数の clamp）は上流 `SamplingRequest` の既定を定数として持つ。前者は
焼き込むと重みを差し替えたときにホストだけが古い数を持って沈黙誤値になるので、必ず導出する
（TS 側の正本 `packages/models/src/irodori/config.ts` のモジュール doc と同じ理由）。

コーデック 2 グラフの出所は {@link irodori.dacvae.export} が書く**別系列**（`--model` の軸には
乗らない）。台本としては別だが、配布形は 1 リポに同梱するのでこの表が両方を並べる。

公開面は {@link PIPELINE} 1 つ（`karume.dist.Pipeline`）— リポの dist ドライバ
（`tools/export-recipes/dist.py`）がこれを core の PIPELINES へ合成する。
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

from _shared.paths import INPUTS_ROOT
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
    safetensors_header,
)

from .card import render_irodori_model_card

#: 既定のモデル名 — 系列（`outputs/series/irodori-<この名前>/`）と実重みの置き場
#: （`inputs/irodori/<この名前>/`）を束ねる 1 語。`irodori.tokenizer_ref.default_out_dir` が
#: `--model-dir` のディレクトリ名から系列名を作るので、読み手のこちらも同じ 1 語から組む。
IRODORI_DEFAULT_MODEL = "v4-small"

#: 系列名とリポ名の接頭辞（`irodori-<モデル名>`）。
IRODORI_SERIES_PREFIX = "irodori"

#: パイプライン契約（ADR 0041 §2 — モデル単位）。TS 側の受理集合は
#: `IRODORI_PIPELINE_NAME` / `IRODORI_PIPELINE_MAJOR`。
IRODORI_PIPELINE = "irodori/1"

#: チェックポイント（`inputs/irodori/<モデル名>/`）のファイル名と、`__metadata__` が持つ
#: config の綴り。どちらも `irodori/export.py`（`MODEL_FILE` / `MODEL_CONFIG_META_KEY`）と同じ
#: — 重みが実際に構成されたときの形の正本はチェックポイントの中にある（HF から引き直さない）。
IRODORI_CKPT_FILE = "model.safetensors"
IRODORI_CONFIG_META_KEY = "config_json"

#: 役割名 → 系列のターゲットディレクトリ名（`irodori.export.TARGETS` の綴り）。役割名は
#: manifest の weights / assets キーでもあるので、ハイフン綴りの系列名とはここで縁を切る。
IRODORI_SERIES_DIRS: Mapping[str, str] = {
    "backbone": "backbone",
    "text_proj": "text-proj",
    "caption_proj": "caption-proj",
    "speaker": "speaker",
    "duration": "duration",
    "dit": "dit",
}

#: コーデック（DACVAE）の 1 語 — **別リポ・別重み**なので系列も入力素材も専用の名前を持つ
#: （`outputs/series/<この名前>/` に export 済みグラフ・`inputs/irodori/<この名前>/` に
#: `irodori/dacvae/convert.py` が書いた `metadata.json`）。Irodori のモデル名（`v4-small`）
#: とは独立に動くので、`--model` の軸には乗せない。
IRODORI_CODEC_NAME = "dacvae-32dim"

#: 役割名 → コーデック系列のターゲットディレクトリ名（`irodori.dacvae.export.TARGETS` の綴り）。
IRODORI_CODEC_DIRS: Mapping[str, str] = {
    "codec_decoder": "decoder",
    "codec_encoder": "encoder",
}

#: グラフを持つ役割の全体（Irodori 本体 6 + コーデック 2）。
IRODORI_GRAPH_ROLES: tuple[str, ...] = (*IRODORI_SERIES_DIRS, *IRODORI_CODEC_DIRS)

#: `irodori/dacvae/convert.py` が書く構成ファイルと、そこから読む 2 つのキー。**sampleRate /
#: hopLength を直書きしない**ための出どころ（`irodori.dacvae.export.hop_length` と同じ式
#: — `hop_length = prod(encoder_rates)`）。
IRODORI_CODEC_METADATA_FILE = "metadata.json"
IRODORI_CODEC_SAMPLE_RATE_KEY = "sample_rate"
IRODORI_CODEC_RATES_KEY = "encoder_rates"

#: tokenizer 資産の出所（`irodori/tokenizer_ref.py` が系列の下へ書く 4 ファイルのうち、配布へ入る
#: のは資産本体だけ — golden / nfkc 表は検証用で実行に要らない）。
IRODORI_TOKENIZER_DIR = "tokenizer"
IRODORI_TOKENIZER_FILE = "tokenizer.json"

#: 配る格納 dtype。役割名は `<グラフ役割>_<dtype>` で、系列 root と 1:1 に対応する。
#: **quant 席の綴りとは別軸**（w8 / w8a8 はどちらも i8 系列を指す — 対応表は
#: {@link IRODORI_QUANT_SEATS}）。
IRODORI_WEIGHT_DTYPES: tuple[str, ...] = ("f32", "f16", "i8", "i4")

#: 格納 dtype → その系列が持つ**役割**（`irodori.export.DTYPE_TARGETS` /
#: `irodori.dacvae.export.WEIGHT_DTYPES` と対）。
#:
#: MUST: i4 は `dit` だけ。i4 の実行経路は linear の重みスロット限定（ADR 0069 決定 5）で、
#: DiT 以外の 7 役は quant 席 `w4` でも **i8 系列のバイトをそのまま共有する**
#: （{@link IRODORI_QUANT_SEATS}）— 他役割の i4 系列は書き出す側も持たない。表をここ 1 箇所に
#: 置くのは、出力 path / 格納 dtype 要求 / weights 宣言 / 配置表の 4 つが「どの (役割, dtype) が
#: 実在するか」で同じ判断をするため（別々に持つと、席を 1 つ足した日に片方だけ更新される）。
IRODORI_DTYPE_ROLES: Mapping[str, tuple[str, ...]] = {
    "f32": IRODORI_GRAPH_ROLES,
    "f16": IRODORI_GRAPH_ROLES,
    "i8": IRODORI_GRAPH_ROLES,
    "i4": ("dit",),
}

#: 圧縮していない系列の dtype（系列 root に接尾が付かない唯一の席で、quant に依存しない
#: 資産〈tokenizer〉の置き場でもある）。
IRODORI_PLAIN_DTYPE = "f32"

#: i4 系列が記録する校正条件（{@link assert_irodori_calib_provenance}）。校正の有無は**格納形を
#: 1 バイトも変えない**（格子は RTN i4 g32 のまま — 変わるのは丸め値と scale 台帳だけ）ので、
#: 資産・manifest・ヘッダのどれからも判別できない。それでいて品質差は裁定を分ける大きさなので、
#: anima と同じ規律で「書き出した側が事実を書き残す」。系列レイアウトの綴りは読み手（ここ）が
#: 持ち、書き手（`irodori/export.py`）はここから引く — 2 箇所で独立に動かさない。
CALIB_PROVENANCE_FILE = "calib_provenance.json"

#: 配布して良い丸め方式（{@link CALIB_PROVENANCE_FILE} の `method`）。`--no-calib` の素の RTN は
#: smoke 用で、配布資産にしない（`irodori/export.py` の該当 MUST）。
CALIB_SHIPPABLE_METHOD = "gptq"


def irodori_role(role: str, dtype: str) -> str:
    """役割名（`backbone_f16` — 配置表・出力 path・格納 dtype 要求が共有する 1 語）。"""
    return f"{role}_{dtype}"


#: 出力の相対 path（**モデルサブツリー内**）— 配置表と manifest が共有する 1 箇所。
#: 格納 dtype をファイル名に出すのは Anima / SBV2 と同じ形（`model.f16.safetensors`）で、
#: 1 つのディレクトリに系列 2 本が並んでも取り違えようがない綴りにするため。
IRODORI_OUTPUT_PATHS: Mapping[str, str] = {
    **{
        irodori_role(role, dtype): f"{role}/model.{dtype}.safetensors"
        for dtype, roles in IRODORI_DTYPE_ROLES.items()
        for role in roles
    },
    "tokenizer": f"{IRODORI_TOKENIZER_DIR}/{IRODORI_TOKENIZER_FILE}",
}

#: 各役割の safetensors ヘッダに**要求する**格納 dtype（Anima / SBV2 と同じ根拠 — 素の F32
#: 資産が組み立て・ロード・実行を全て通って参照一致の門まで沈黙した実測事故）。圧縮系列は
#: fake-quant 対象だけが F16 / I8 / I4 になる（bias / norm / グラフ定数、i8 の per-channel scale と
#: i4 の group scale は F32 のまま）ので「その dtype を含む」を要求する。tokenizer は JSON なので
#: 載せない。
IRODORI_STORAGE_REQUIREMENTS: Mapping[str, str] = {
    irodori_role(role, dtype): dtype.upper()
    for dtype, roles in IRODORI_DTYPE_ROLES.items()
    for role in roles
}

#: 各役割の safetensors ヘッダに**あってはならない**格納 dtype（{@link assert_storage_absent}）。
#: f32 席は「F32 を含む」だけでは圧縮系列の資産と区別できない（圧縮系列も適格外の重み
#: — bias / norm / グラフ定数 / i8 の per-channel scale / i4 の group scale — を F32 で持つ）ので、
#: **圧縮側の格納 dtype 全部**の不在を併せて要求して初めて系列 × 格納 dtype が集合として一意に
#: なる。逆向き（圧縮席へ f32 資産）は {@link assert_storage} が要求 dtype の不在で落とす。
#:
#: MUST: **i8 席も I4 の不在を要求する**（`dit` だけが両方の系列を持つ）。i4 系列は
#: **I4 + I8 + F32 の混成**（block 内の adaLN 以外 168 本が I4・block 外 5 本 + adaLN 144 本が
#: I8・bias / norm / scale が F32 — 聴感裁定 2026-08-23 で block 外と adaLN を i4 から外した。
#: `irodori.export._fake_quant_i4`）なので、
#: 「I8 を含む」という要求検査は i4 系列でも満たされてしまう。i8 席と i4 系列を分けているのは
#: **この禁止表だけ**で、外すと既定席 `w8a8` の `linearCompute: "i8a8"` が i4 常駐で走る w4a8
#: 経路（ADR 0076）へ黙って化ける。混成になる前も「i4 系列が i8 席を名乗れるかどうかが上流の
#: 適格率次第」で同じ穴が空いていた — 混成でその穴が常時開いた形になっただけ。
IRODORI_STORAGE_FORBIDDEN: Mapping[str, tuple[str, ...]] = {
    **{
        irodori_role(role, IRODORI_PLAIN_DTYPE): tuple(
            dtype.upper() for dtype in IRODORI_WEIGHT_DTYPES if dtype != IRODORI_PLAIN_DTYPE
        )
        for role in IRODORI_GRAPH_ROLES
    },
    **{irodori_role(role, "i8"): ("I4",) for role in IRODORI_DTYPE_ROLES["i4"]},
}

#: weights の宣言（dtype ラベル → 役割名）。8 グラフとも f32 / f16 / i8 の 3 席を持ち（`dit` は
#: さらに i4）、{@link complete_quant_weights} の自動補完は掛からないので quant 表が全役割を
#: 名指しする。
IRODORI_WEIGHTS: Mapping[str, Mapping[str, WeightFiles]] = {
    role: {
        dtype: WeightFiles(irodori_role(role, dtype))
        for dtype, roles in IRODORI_DTYPE_ROLES.items()
        if role in roles
    }
    for role in IRODORI_GRAPH_ROLES
}

#: assets の宣言（quant 選択に依存しない無条件ファイル）。
IRODORI_ASSETS: Mapping[str, str] = {"tokenizer": "tokenizer"}


class QuantSeat(NamedTuple):
    """quant 席 1 つ（既定の格納 dtype・実行形ノブ・役割ごとの例外）。

    **席名と系列 root の対応をここ 1 箇所だけで綴る**（`w8` / `w8a8` はどちらも `-i8` 系列を
    指し、バイトは 1 組を共有する — 違うのは `session` だけ）。
    """

    #: 例外に載らない役割の格納 dtype。
    dtype: str
    #: 実行形ノブ（`session` の語彙は manifest 所有）。
    session: Mapping[str, str]
    #: 役割ごとの例外（混成席のためだけの口 — 既定が全役割へ効く形は崩さない）。
    roles: Mapping[str, str] = {}


#: quant 席の綴り → {@link QuantSeat}。
#:
#: **8 役一律をやめるのは `w4` 席だけ**（2026-08-23 ユーザー裁定）。2026-08-12 の裁定は「i8 も
#: 一律」で、当時の根拠は混成にすると S（発話長）が席ごとにドリフトする実測だったが、その
#: ドリフトは **GPTQ 校正で消えた**（`docs/research/2026-08-20-gptq-awq-calibrated-rounding.md`
#: §6 — 校正付き丸めで S は f32 と完全一致）。`w8` / `w8a8` の 8 役一律は従来どおり。
#:
#: MUST: `w8a8` の `linearCompute` は **`dit` の Session にだけ**降りる（models 側 `pipeline.ts`
#: のモジュール doc）— DiT の linear 317 本が唯一の適格集合で、条件エンコーダ 5 本は 1 生成に
#: 1 回しか走らない。
#:
#: MUST: `w4` は **`linearCompute` を宣言しない**。i4 常駐 × i8 活性は w4a8 経路（ADR 0076・
#: group 部分縮約）に乗るが、その構成は irodori では**一度も測っていない**（perf-ledger の
#: 席は重み側のみ）。anima では同じ構成が実 GPU の画で品質裁定に落ちている（ADR 0076 決定 6 /
#: `anima/distribution.py` の同 MUST）ので、測る前に宣言だけ足すと「速いが荒い」を既定の顔で
#: 配ることになる。この席の存在理由は**サイズと VRAM**であって速度ではない。
IRODORI_QUANT_SEATS: Mapping[str, QuantSeat] = {
    "f32": QuantSeat("f32", {}),
    "f16": QuantSeat("f16", {}),
    "w8": QuantSeat("i8", {}),
    "w8a8": QuantSeat("i8", {"linearCompute": "i8a8"}),
    "w4": QuantSeat("i8", {}, {"dit": "i4"}),
}

IRODORI_QUANTS: Mapping[str, Any] = {
    name: {
        "weights": {role: seat.roles.get(role, seat.dtype) for role in IRODORI_GRAPH_ROLES},
        "session": dict(seat.session),
    }
    for name, seat in IRODORI_QUANT_SEATS.items()
}

#: 既定は `w8a8`（ユーザー聴感裁定 2026-08-12 — DAC + ヘッドホンで f32/f16/w8/w8a8 を通しで
#: 確認し「音質的な劣化という感じはしない」。配布 25.2% / DiT 常駐 0.37GB / wall ×1.12 が
#: 既定で効き、最も忠実な `f32` 席は残したまま明示で選べる。数値上の帯（sim LSD 5.64・
#: w8 golden との z maxAbs 2.97）は `e2e_irodori_w8a8_test.ts` の判別帯が持つ）。
IRODORI_DEFAULT_QUANT = "w8a8"

#: DACVAE のフレームレート（Hz）— 48kHz / hop 1920 = 25。`irodori.export.CODEC_FRAME_RATE` と
#: 同値で、コーデックが別リポ・別重みなのでチェックポイントの config には入っていない。
#: `sampleRate` / `hopLength` はコーデックの `metadata.json` から導出し、3 者の整合
#: （`sampleRate == frameRate × hopLength`）は {@link irodori_pipeline_config} が見る。
IRODORI_FRAME_RATE = 25

#: codec decoder のタイル分割で採用区間の両側へ足す latent フレーム数。**受容野由来のモデル
#: 定数**（片側 13,793 サンプル = 7.19 フレーム → 8 フレーム = 15,360 サンプルで覆う）で、
#: `metadata.json` には入っていないのでここが唯一の出どころ。実測の根拠は decoder 主経路に
#: 因果層が無く全層が対称 pad か厳密 `L·stride` の convT である（= 平行移動同変）こと。
IRODORI_CODEC_HALO_FRAMES = 8

#: 発話長 clamp の秒数（上流 `SamplingRequest.min_seconds` / `max_seconds` の既定）。
#: `max_seconds` は **`dit` の記号次元 S の上限を決めた値でもある**
#: （`irodori.export.DIT_MAX_SECONDS` が同じ 30.0 を「実装側の正本は SamplingRequest の既定」
#: として持つ）。1 つの定数から両方を組むのは、配布形の clamp と焼かれたグラフの上限が
#: 独立に動くと「S は通るのに RoPE 表が足りない」形で実行時にしか出ないため。
IRODORI_MIN_SECONDS = 0.5
IRODORI_MAX_SECONDS = 30.0

#: 実行時ノブの既定（上流 `SamplingRequest` の同名フィールドと `rf.sample_euler_rf_cfg` の
#: `init_scale`）。**モデル固有ではない**のでチェックポイントの config には無く、ここが唯一の
#: 出どころになる（Anima の `anima.distribution.ANIMA_PIPELINE_CONFIG` と同じ性格）。
#:
#: MUST: `speakerUncondMode` / `cfgGuidanceMode` は分岐用ではなく**宣言**（ADR 0047 決定 1）。
#: TS 側はこの 2 値以外を parse 時に拒否するので、別のモードで焼いた配布形は読まれる前に落ちる。
#: full-loop golden（`irodori/pipeline_ref.py` の `meta.json`）も同じ値で焼かれており、golden 再生成
#: と dist 再生成がずれたら E2E 門（`e2e_irodori_latent_test.ts`）が実効値 drift として落とす。
IRODORI_SAMPLING_DEFAULTS: Mapping[str, Any] = {
    "steps": 40,
    "initScale": 0.999,
    "cfgMinT": 0.5,
    "cfgMaxT": 1.0,
    "cfgScales": {"text": 3.0, "speaker": 5.0, "caption": 3.0},
    "minSeconds": IRODORI_MIN_SECONDS,
    "maxSeconds": IRODORI_MAX_SECONDS,
    "speakerUncondMode": "mask",
    "cfgGuidanceMode": "independent",
}

#: 各グラフに要求する `(入力名の並び, 出力の本数)`。**入力の並びは実行時に位置で読まれる形の
#: 正本**で、出力本数は「検証用の別資産が混ざっていないか」を見る席（SBV2 の
#: {@link sbv2.distribution.assert_bert_hidden} と同じ機序 — `caption_proj` が 1 出力の資産に
#: 差し替わると `caption_vec` を第 1 出力から採る別のベクトルで duration が回り、shape は
#: 合ったまま沈黙する）。
IRODORI_GRAPH_SHAPES: Mapping[str, tuple[tuple[str, ...], int]] = {
    "backbone": (("input_ids",), 1),
    "text_proj": (("hidden",), 1),
    "caption_proj": (("hidden",), 2),
    "speaker": (("latent",), 1),
    "duration": (
        ("text_state", "speaker_vec", "has_speaker", "caption_vec", "has_caption"),
        1,
    ),
    "dit": (("x_t", "t_embed", "mask", "text_state", "speaker_state", "caption_state"), 1),
    # コーデック 2 本は位置表もマスクも持たない純畳み込み網（入力 1 本 / 出力 1 本）。
    "codec_decoder": (("latent",), 1),
    "codec_encoder": (("wav",), 1),
}

#: `(役割, グラフ入力, 軸, pipelineConfig の欄)` — グラフの**静的**次元と宣言の突合表。
#: TS 側 `IrodoriPipeline.fromAssets` の `assertStaticDim` と同じ組み合わせを**焼く側でも**
#: 見る（配ってから利用者の手元で初めて落ちる形にしない）。
IRODORI_STATIC_DIMS: tuple[tuple[str, str, int, str], ...] = (
    ("dit", "x_t", 2, "latentDim"),
    ("dit", "t_embed", 1, "timestepEmbedDim"),
    ("dit", "text_state", 1, "maxTextLen"),
    ("dit", "text_state", 2, "textDim"),
    ("dit", "speaker_state", 1, "speakerRows"),
    ("dit", "speaker_state", 2, "speakerDim"),
    ("dit", "caption_state", 1, "maxCaptionLen"),
    ("dit", "caption_state", 2, "captionDim"),
    ("duration", "text_state", 2, "textDim"),
    ("duration", "speaker_vec", 1, "speakerDim"),
    ("duration", "caption_vec", 1, "captionDim"),
    # コーデックは**別リポ・別重み**なので、latent の幅と 1 フレームのサンプル数が Irodori 側の
    # 宣言と噛み合っている保証がここにしか無い（別次元の DACVAE を混ぜると shape は合ったまま
    # 別の声になる / 波形長だけが静かにずれる）。
    ("codec_decoder", "latent", 2, "latentDim"),
    ("codec_encoder", "wav", 2, "hopLength"),
)


def irodori_series_name(model: str) -> str:
    """系列名（`outputs/series/<この名前>/`）— `irodori.tokenizer_ref.default_out_dir`
    と同じ綴り。
    """
    return f"{IRODORI_SERIES_PREFIX}-{model}"


def irodori_repo_name(model: str) -> str:
    """単一モデルの配布リポ名（`karume-` prefix はリポ名裁定 2026-08-09）。"""
    return f"karume-{IRODORI_SERIES_PREFIX}-{model}"


@dataclass(frozen=True)
class IrodoriSources:
    """組み立ての入力。系列とチェックポイントの置き場（`inputs/` — 生成物ではない）が 2 組。

    チェックポイント側が要るのは `pipelineConfig` の数を**焼き込まずに導出**するため
    （`__metadata__` / `metadata.json` だけを読むので 2.9GB のペイロードは舐めない）。
    コーデックが別の 2 本を持つのは**別リポ・別重み**だから — Irodori のモデル名を動かしても
    コーデックは動かない。

    系列は**格納 dtype ごとに別ディレクトリ**（`irodori.export.default_out_root` /
    `irodori.dacvae.export.default_out_root` の dtype 接尾）。tokenizer 資産は quant に
    依存しないので素の系列（f32）側の 1 本だけを見る。
    """

    model: Path
    codec_model: Path
    #: 格納 dtype → 系列 root（{@link IRODORI_DTYPE_ROLES} が Irodori 側の役割を持つ dtype）。
    series_by_dtype: Mapping[str, Path]
    #: 同・コーデック側（コーデックは i4 系列を持たないので f32 / f16 / i8 の 3 本）。
    codec_series_by_dtype: Mapping[str, Path]

    @property
    def series(self) -> Path:
        """素の系列 root（quant に依存しない tokenizer 資産の置き場）。

        写しの欄を持たず毎回引くのは、系列 root が 2 箇所で独立に動く形を作らないため。
        """
        return self.series_by_dtype[IRODORI_PLAIN_DTYPE]

    @property
    def codec_series(self) -> Path:
        """素のコーデック系列 root。"""
        return self.codec_series_by_dtype[IRODORI_PLAIN_DTYPE]


def irodori_sources(series_dir: Path, model: str = IRODORI_DEFAULT_MODEL) -> IrodoriSources:
    """系列の親ディレクトリ（`outputs/series/`）と `_shared.paths` の綴りから入力を引く。

    dtype 接尾の綴りは `irodori.export.default_out_root` /
    `irodori.dacvae.export.default_out_root` と同一 — 書き手と読み手が同じ 1 語から組む。
    """
    suffix = {
        dtype: "" if dtype == IRODORI_PLAIN_DTYPE else f"-{dtype}"
        for dtype in IRODORI_WEIGHT_DTYPES
    }
    by_dtype = {
        dtype: series_dir / f"{irodori_series_name(model)}{tail}" for dtype, tail in suffix.items()
    }
    codec_by_dtype = {
        dtype: series_dir / f"{IRODORI_CODEC_NAME}{tail}"
        for dtype, tail in suffix.items()
        # コーデックが席を持たない dtype（i4）の root は**作らない** — 使わない path を配って
        # おくと、i4 の役割が増えた日に「在ることになっている系列」が黙って参照される。
        if not set(IRODORI_DTYPE_ROLES[dtype]).isdisjoint(IRODORI_CODEC_DIRS)
    }
    return IrodoriSources(
        model=INPUTS_ROOT / IRODORI_SERIES_PREFIX / model,
        codec_model=INPUTS_ROOT / IRODORI_SERIES_PREFIX / IRODORI_CODEC_NAME,
        series_by_dtype=by_dtype,
        codec_series_by_dtype=codec_by_dtype,
    )


def irodori_placements(sources: IrodoriSources) -> dict[str, Path]:
    """役割名 → 出所のファイル。出力の path は {@link IRODORI_OUTPUT_PATHS} が持つ。

    この表に無いものは出力へ入らない（`io.*.safetensors` と tokenizer の golden 3 本は
    これで落ちる）。
    """
    placements = {"tokenizer": sources.series / IRODORI_TOKENIZER_DIR / IRODORI_TOKENIZER_FILE}
    for dtype, roles in IRODORI_DTYPE_ROLES.items():
        for role in roles:
            series = (
                sources.series_by_dtype[dtype]
                if role in IRODORI_SERIES_DIRS
                else sources.codec_series_by_dtype[dtype]
            )
            directory = (IRODORI_SERIES_DIRS | IRODORI_CODEC_DIRS)[role]
            placements[irodori_role(role, dtype)] = series / directory / "model.safetensors"
    return placements


def irodori_model_config(model_dir: Path) -> Mapping[str, Any]:
    """チェックポイントの `__metadata__` から `config_json` を読む（ヘッダだけ読む）。

    `irodori.export.read_configs` と同じ出どころ・同じ理由（HF から config を引き直さない）。
    こちらが torch を経由しないのは、組み立てが要るのが JSON 1 本だけだから。
    """
    path = model_dir / IRODORI_CKPT_FILE
    if not path.is_file():
        raise DistError(f"組み立ての入力が無い: {path}")
    metadata = safetensors_header(path).get("__metadata__")
    if not isinstance(metadata, dict) or IRODORI_CONFIG_META_KEY not in metadata:
        raise DistError(f"{path} の __metadata__ に '{IRODORI_CONFIG_META_KEY}' が無い")
    try:
        config = json.loads(metadata[IRODORI_CONFIG_META_KEY])
    except json.JSONDecodeError as error:
        raise DistError(f"{path}: {IRODORI_CONFIG_META_KEY} が JSON として読めない") from error
    if not isinstance(config, dict):
        raise DistError(f"{path}: {IRODORI_CONFIG_META_KEY} が最上位オブジェクトでない")
    return config


def _irodori_int(config: Mapping[str, Any], key: str) -> int:
    """チェックポイント config の整数フィールド（宣言長 / 幅）を検査して読む。"""
    value = config.get(key)
    # bool は int の派生。`"latent_dim": true` を 1 として通すと幅の突合が緩む。
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise DistError(f"{IRODORI_CONFIG_META_KEY} の {key} が正の整数でない（{value!r}）")
    return value


def _irodori_float(config: Mapping[str, Any], key: str) -> float:
    """チェックポイント config の実数フィールド（秒数）を検査して読む。"""
    value = config.get(key)
    # NaN は比較が全て False なので `<= 0` を素通りし、下流の秒 → フレーム換算（int()）で
    # 一般例外として漏れる。Inf も同じ席で落ちる — 有限性をここで要求する。
    if (
        not isinstance(value, int | float)
        or isinstance(value, bool)
        or not math.isfinite(value)
        or value <= 0
    ):
        raise DistError(f"{IRODORI_CONFIG_META_KEY} の {key} が有限の正の数でない（{value!r}）")
    return float(value)


class IrodoriCodecNumbers(NamedTuple):
    """コーデックの `metadata.json` から導く 2 つ（秒 ↔ サンプル ↔ フレームの換算）。"""

    sample_rate: int
    hop_length: int


def irodori_codec_numbers(codec_model_dir: Path) -> IrodoriCodecNumbers:
    """`irodori/dacvae/convert.py` が書いた `metadata.json` から `sampleRate` / `hopLength` を導く。

    MUST: 写経しない — 別次元・別 hop の DACVAE へ差し替えたときに、ホストだけが古い換算を
    持ったまま「それらしい長さの音声」を出す（例外は出ない）。`hop_length` は
    `irodori.dacvae.export.hop_length` と同じ式（`prod(encoder_rates)` —
    `DACVAE.__init__` の綴り）。
    """
    path = codec_model_dir / IRODORI_CODEC_METADATA_FILE
    if not path.is_file():
        raise DistError(
            f"組み立ての入力が無い: {path}（`uv run python -m irodori.dacvae.convert` で作る）"
        )
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise DistError(f"{path} が JSON として読めない") from error
    kwargs = raw.get("kwargs") if isinstance(raw, dict) else None
    if not isinstance(kwargs, dict):
        raise DistError(f"{path} に 'kwargs' が無い")
    sample_rate = kwargs.get(IRODORI_CODEC_SAMPLE_RATE_KEY)
    if not isinstance(sample_rate, int) or isinstance(sample_rate, bool) or sample_rate <= 0:
        raise DistError(f"{path} の {IRODORI_CODEC_SAMPLE_RATE_KEY} が正の整数でない")
    rates = kwargs.get(IRODORI_CODEC_RATES_KEY)
    if not isinstance(rates, list) or not rates:
        raise DistError(f"{path} の {IRODORI_CODEC_RATES_KEY} が非空のリストでない")
    hop_length = 1
    for rate in rates:
        if not isinstance(rate, int) or isinstance(rate, bool) or rate <= 0:
            raise DistError(f"{path} の {IRODORI_CODEC_RATES_KEY} に正の整数でない要素がある")
        hop_length *= rate
    return IrodoriCodecNumbers(sample_rate=sample_rate, hop_length=hop_length)


def irodori_pipeline_config(
    config: Mapping[str, Any], codec: IrodoriCodecNumbers
) -> dict[str, Any]:
    """`pipelineConfig`（TS 側スキーマの 23 欄）をチェックポイント config から組む。

    MUST: モデル固有の数を写経しない — 重みを差し替えたときにホストだけが古い数を持つと、
    右 pad も行数計算もそのまま通って（shape は合う）**別の位置の条件を読んだ**結果が沈黙で出る。
    欄名と値域の正本は `packages/models/src/irodori/config.ts`。

    `latentDim` が 2 つの役割（`dit` の `x_t` の幅 = `latent_dim × latent_patch_size` と、
    参照 latent の 1 フレームの幅 = `latent_dim`）を兼ねているので、両者が一致する
    `latent_patch_size == 1` でなければ組めない — 兼ねられなくなったら TS 側の欄を割る話に
    なるので、黙って片方を選ばずここで落とす。
    """
    latent_patch = _irodori_int(config, "latent_patch_size")
    if latent_patch != 1:
        raise DistError(
            f"latent_patch_size が {latent_patch} — pipelineConfig の latentDim は"
            " `dit` の x_t 幅と参照 latent の 1 フレーム幅を兼ねており、1 でなければ"
            "両者が一致しない（TS 側の欄を割る変更が要る）"
        )
    # 秒 → フレーム（`frameRate`）と 秒 → サンプル → フレーム（コーデック由来の 2 値）が
    # 独立に動く形を作らない。TS 側も parse 時に同じ式を見るが、**配ってから落ちる**のを
    # 避けるためにここでも見る（別 hop のコーデックを混ぜた瞬間に気づける席）。
    if codec.sample_rate != IRODORI_FRAME_RATE * codec.hop_length:
        raise DistError(
            f"コーデックの sample_rate {codec.sample_rate} が frameRate {IRODORI_FRAME_RATE}"
            f" × hop_length {codec.hop_length} と違う"
        )
    frames = int(_irodori_float(config, "ref_max_seconds") * IRODORI_FRAME_RATE)
    speaker_patch = _irodori_int(config, "speaker_patch_size")
    return {
        "maxTextLen": _irodori_int(config, "max_text_len"),
        "maxCaptionLen": _irodori_int(config, "max_caption_len"),
        # 参照 latent の patch 後の上限（`irodori.export.speaker_sym_max`）+ 平均トークン 1 本。
        "speakerRows": frames // speaker_patch + 1,
        # 生成できる latent の上限（`irodori.export.dit_sym_max` と同じ式）。
        "ditSymMax": int(IRODORI_MAX_SECONDS * IRODORI_FRAME_RATE) // latent_patch,
        "frameRate": IRODORI_FRAME_RATE,
        "sampleRate": codec.sample_rate,
        "hopLength": codec.hop_length,
        "codecHaloFrames": IRODORI_CODEC_HALO_FRAMES,
        "latentDim": _irodori_int(config, "latent_dim"),
        "speakerPatchSize": speaker_patch,
        "speakerDim": _irodori_int(config, "speaker_dim"),
        "textDim": _irodori_int(config, "text_dim"),
        "captionDim": _irodori_int(config, "caption_dim"),
        "timestepEmbedDim": _irodori_int(config, "timestep_embed_dim"),
        **IRODORI_SAMPLING_DEFAULTS,
    }


def assert_irodori_graphs(
    placements: Mapping[str, Path], pipeline_config: Mapping[str, Any]
) -> None:
    """8 グラフが**読み出せて**、入力の並び・出力本数・静的次元が宣言どおりであることを見る。

    MUST: ずれても shape は合ったままロードも実行も通る組み合わせがある（`caption_proj` の
    出力本数・条件 state の宣言長・`speaker` の patch 幅）ので、配布形を並べる前にここで落とす。
    SBV2 の {@link sbv2.distribution.assert_bert_hidden} と同じ規律 — 別々の台本
    （`irodori/export.py` のターゲットと、この manifest）が持つ数を突き合わせる席がここしか無い。

    MUST: **格納 dtype の系列を 1 本残らず**掛ける。f16 系列は f32 とは別プロセスの emit なので、
    片方だけ検査すると「f32 は宣言どおりだが f16 だけ別の版」が素通りする（格納 dtype の一致は
    {@link assert_storage} が見るが、あちらはグラフ宣言を一切見ない）。i4 系列は `dit` 1 本しか
    持たない（{@link IRODORI_DTYPE_ROLES}）ので、その系列に**在る役割だけ**を掛ける。
    """
    for dtype, roles in IRODORI_DTYPE_ROLES.items():
        _assert_irodori_graph_set(
            {role: placements[irodori_role(role, dtype)] for role in roles},
            pipeline_config,
        )


def _assert_irodori_graph_set(
    placements: Mapping[str, Path], pipeline_config: Mapping[str, Any]
) -> None:
    """1 系列ぶんのグラフを検査する（`placements` のキーは dtype 接尾の無いグラフ役割名）。

    その系列に**在る役割だけ**を受ける（i4 系列は `dit` 1 本）。役割を跨ぐ突合（`speaker` の
    patch 幅）は両方が在るときだけ掛ける — 片方しか無い系列で「掛からなかった」ことは
    {@link assert_irodori_graphs} の全 dtype ループが他系列で埋める。
    """
    graphs = {role: ir_graph(path) for role, path in placements.items()}
    for role, (expected_inputs, expected_outputs) in IRODORI_GRAPH_SHAPES.items():
        if role not in graphs:
            continue
        path = placements[role]
        graph = graphs[role]
        names = tuple(graph_inputs(graph, path))
        if names != expected_inputs:
            raise DistError(
                f"{path} のグラフ入力が {list(names)} で、期待の {list(expected_inputs)} と違う"
                " — 実行側は名前で束ねるので、1 つでも綴りが変われば束ねられない"
                "（並びまで見るのは export 側の宣言順が動いていないことの証跡）"
            )
        outputs = graph.get("outputs")
        count = len(outputs) if isinstance(outputs, list) else outputs
        if count != expected_outputs:
            raise DistError(
                f"{path} のグラフ出力が {count} 本で、配布形が要求する {expected_outputs} 本で"
                "ない — 別のターゲットの資産が混ざっている"
            )
    for role, name, axis, field_name in IRODORI_STATIC_DIMS:
        if role not in graphs:
            continue
        declared = graph_inputs(graphs[role], placements[role])[name][axis]
        expected = pipeline_config[field_name]
        if declared != expected:
            raise DistError(
                f"{placements[role]} の入力 '{name}' の軸 {axis} が {declared!r}、"
                f"pipelineConfig の {field_name} は {expected} — チェックポイントの config と"
                "焼かれたグラフが別の版"
            )
    # 参照 latent は patch してから `speaker` へ渡す（ADR 0047 決定 4）ので、入力幅は 2 欄の積。
    if "speaker" in graphs:
        patched = pipeline_config["latentDim"] * pipeline_config["speakerPatchSize"]
        width = graph_inputs(graphs["speaker"], placements["speaker"])["latent"][2]
        if width != patched:
            raise DistError(
                f"{placements['speaker']} の入力 'latent' の軸 2 が {width!r}、pipelineConfig の"
                f" latentDim × speakerPatchSize は {patched}"
            )
    # `dit` の `mask` は「latent S + 条件 3 区間」の長さで宣言される（ADR 0046 の派生次元）。
    # 区間の合計がずれると、マスクの区間割りだけが黙って別の位置を指す。
    symbols = graphs["dit"].get("symbols")
    if not isinstance(symbols, list) or len(symbols) != 1:
        raise DistError(f"{placements['dit']}: 記号次元が 1 本でない（{symbols!r}）")
    total = sum(
        pipeline_config[field_name] for field_name in ("maxTextLen", "speakerRows", "maxCaptionLen")
    )
    declared_mask = graph_inputs(graphs["dit"], placements["dit"])["mask"][3]
    if declared_mask != f"{symbols[0]}+{total}":
        raise DistError(
            f"{placements['dit']} の入力 'mask' の軸 3 が {declared_mask!r}、pipelineConfig の"
            f" 条件 3 区間の合計は {total}（期待 '{symbols[0]}+{total}'）"
        )


def assert_irodori_calib_provenance(sources: IrodoriSources) -> None:
    """i4 系列が**校正付き**（GPTQ）で丸められたことを、書き出し側の記録で確かめる。

    MUST: 校正の有無は格納形を 1 バイトも変えない（格子は RTN i4 g32 のまま — 変わるのは
    丸め値と scale 台帳だけ）。したがって `verify_dist` の構造検査もヘッダ dtype 検査も
    {@link assert_irodori_graphs} も**素通りする**。`--no-calib` は smoke 用の opt-out なのに、
    その生成物が配布へ紛れても資産からは判別できず、出るのは音の劣化だけ — anima の
    `assert_calib_provenance` と同じ「別々の台本が持つ同じ事実は組み立て時に必ず突き合わせる」
    規律をここにも敷く。
    """
    directory = IRODORI_SERIES_DIRS["dit"]
    path = sources.series_by_dtype["i4"] / directory / CALIB_PROVENANCE_FILE
    if not path.is_file():
        raise DistError(
            f"i4 系列の校正条件の記録が無い: {path}"
            "（`python -m irodori.export --dtype i4` で再エクスポートすると書かれる）"
        )
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as cause:
        raise DistError(f"校正条件の記録を解析できない: {path} — {cause}") from cause
    method = record.get("method") if isinstance(record, dict) else None
    if method != CALIB_SHIPPABLE_METHOD:
        raise DistError(
            f"i4 系列が配布して良い丸め方式で作られていない: {path} は {method!r}、"
            f"配布可は {CALIB_SHIPPABLE_METHOD!r} — `--no-calib` の生成物は配布に使わない"
        )


def irodori_plan(sources: IrodoriSources, model: str = IRODORI_DEFAULT_MODEL) -> ModelPlan:
    """Irodori 1 モデルぶんの計画を組む（検査と config の読み取りをここで全部済ませる）。"""
    assert_model_name(model)
    placements = irodori_placements(sources)
    pipeline_config = irodori_pipeline_config(
        irodori_model_config(sources.model), irodori_codec_numbers(sources.codec_model)
    )
    assert_irodori_calib_provenance(sources)
    for role, source in placements.items():
        assert_storage(role, source, IRODORI_STORAGE_REQUIREMENTS)
        assert_storage_absent(role, source, IRODORI_STORAGE_FORBIDDEN)
    assert_irodori_graphs(placements, pipeline_config)
    return ModelPlan(
        name=model,
        pipeline=IRODORI_PIPELINE,
        artifacts={
            role: Artifact(IRODORI_OUTPUT_PATHS[role], source=source)
            for role, source in placements.items()
        },
        weights=IRODORI_WEIGHTS,
        assets=IRODORI_ASSETS,
        quants=complete_quant_weights(IRODORI_WEIGHTS, IRODORI_QUANTS),
        default_quant=IRODORI_DEFAULT_QUANT,
        pipeline_config=pipeline_config,
    )


def irodori_dist_plan(series_dir: Path, model: str) -> ModelPlan:
    """`--series` の親から Irodori 1 モデルの計画を組む（CLI のディスパッチ先）。"""
    return irodori_plan(irodori_sources(series_dir, model), model)


#: `--pipeline irodori` の 1 行（ドライバが core の PIPELINES へ合成する）。
PIPELINE = Pipeline(
    default_model=IRODORI_DEFAULT_MODEL,
    repo_name=irodori_repo_name,
    plan=irodori_dist_plan,
    # 帰属は 1 通りだけ（上流 1 リポの重みを格納形へ落とし直したもの）— 選択肢が無いので
    # 省略で通る。2 つ目のファミリーが生えた瞬間に明示が要求されはじめる。
    card_profiles={"irodori": render_irodori_model_card},
)
