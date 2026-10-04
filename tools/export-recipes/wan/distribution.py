"""Wan2.1 の配布 recipe — 系列レイアウト・出力 path 表・quant 表・カードの選択（ADR 0118 決定 7）。

汎用の組み立てエンジン（配置・共有席の畳み込み・sha256・manifest・staging/swap・検証）は
`karume.dist` が持つ。ここが持つのは **Wan2.1 固有の事実**だけ: どの系列ディレクトリから何を
拾い、配布形のどの path へ、どの dtype ラベルで並べ、どの quant を既定にするか。

配布するのはグラフ 3 本（DiT の S 形 `transformer`・VAE の chunk グラフ `vae_decoder_first` /
`vae_decoder_next` — 系列 `wan2.1-t2v-1.3b-f16-dyn`）と、モデル単位の資産 2 本（第 1 段の
テキスト埋め込み `text_embeds` — 決定 4・umT5 のトークナイザと前処理の表 `umt5_tokenizer` — ADR 0119
追記「段 10d の設計」C。どちらも quant 非依存の席・ADR 0109 決定 4）。`transformer` だけは格納ラベル
ごとに容器が 2 本ある（f16 と、系列 `wan2.1-t2v-1.3b-i8-dyn` の i8 — ADR 0120 決定 1）。RoPE の
軸別素表 `rope_base` は `transformer` の**容器の資産**（役割 `rope-base`）なので manifest の
`assets` には載らない（ADR 0109 決定 4 — Anima と同じ席）。

4 本目の部品 `text_encoder`（umT5-XXL encoder の i8 — 系列 `umt5-xxl-i8-dyn`）は計画には
自分の artifact として載るが、公開する配布形では**別リポ `karume-umt5-xxl` への越境参照**として
組む（ADR 0119 追記 A — dist の `--ref-*` 5 指定で、参照元は `--pipeline umt5` が組んだ配布形）。
容器の門（出所・束縛表・入出力の契約）は umT5 の配布 recipe と同じ 1 本
（`wan.umt5_distribution.assert_umt5_encoder`）を通す。

**リポは家族 1 つ・世代は別リポ**（`karume-wan2.1` — ADR 0092 決定 1 / 2）。モデルは世代の中の
軸で、今は `t2v-1.3b` 1 本。quant 席は 3 つ（{@link WAN_QUANTS}）: `f16`（DiT と VAE の重みを
f16 格納・活性 f32 — ADR 0118 決定 7・明示の指定で使う元の重みにいちばん近い席）と、DiT の重みを
i8 にした参照席 `f16+dit8` と実用席 `f16+dit8-a8-attn8-s16`（ADR 0120 決定 1）。既定は実用席
（ADR 0120 裁定 2026-10-04 の 4 — {@link WAN_DEFAULT_QUANT}）。

公開面は {@link PIPELINE} 1 つ — リポの dist ドライバ（`tools/export-recipes/dist.py`）がこれを
core の PIPELINES へ合成する。

MUST: このモジュールは torch を import しない（`import dist` が torch を読まない —
`tests/test_dist_driver.py` の `TestImportingTheDriver`）。書き手（`wan.export_dit` /
`wan.export_vae` / `wan.text_embeds`）は torch を読むので、綴り（系列名・資産名・メタのキー・
グラフ入力名）はここに置き、書き手の綴りとの一致は `wan/tests/test_distribution.py` の門が見る。
"""

from __future__ import annotations

import hashlib
import json
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

from _shared.container_read import ContainerReadError, read_asset, read_asset_declarations
from _shared.licenses import apache_license_2_0
from _shared.upstream import assert_upstream_provenance
from karume.dist import (
    Artifact,
    DistError,
    ModelPlan,
    Pipeline,
    WeightFiles,
    assert_component_present,
    assert_model_name,
    assert_storage,
    assert_storage_absent,
    complete_quant_weights,
    graph_inputs,
    ir_graph,
)
from wan import umt5_tokenizer
from wan.card import render_wan_model_card
from wan.prompts import FIXED_PROMPTS
from wan.sources import DEFAULT_MODEL, SOURCES, WAN21_MODELS
from wan.umt5_distribution import (
    UMT5_DEFAULT_MODEL,
    UMT5_OUTPUT_PATHS,
    UMT5_ROLE,
    assert_umt5_encoder,
    umt5_container,
)

#: パイプライン契約（ADR 0041 §2 — モデル単位）。TS 側の受理集合は `WAN_PIPELINE_NAME` /
#: `WAN_PIPELINE_MAJOR`（`packages/models/src/wan/config.ts`）。
WAN_PIPELINE = "wan/1"

#: 配布リポ名（ADR 0092 決定 1 / 2 — 家族 1 リポ・世代は別リポ。Wan2.2 は `karume-wan2.2`）。
WAN_REPO_NAME = "karume-wan2.1"

#: DiT と VAE の系列（決定 7 — 接尾辞 `-dyn` は ADR 0077 の慣例）。書き手の綴りは
#: `wan.export_dit.SERIES` / `wan.export_vae.SERIES_NAME`。
WAN_SERIES = "wan2.1-t2v-1.3b-f16-dyn"

#: DiT の i8 系列（ADR 0120 決定 2 — transformer だけ）。書き手の綴りは `wan.export_dit.I8_SERIES`。
WAN_I8_SERIES = "wan2.1-t2v-1.3b-i8-dyn"

#: テキスト埋め込みの系列（グラフを持たない compile 生成物の席 — docs/assets-layout.md）と
#: その資産のファイル名・メタの唯一のキー。書き手の綴りは `wan.text_embeds` の
#: `SERIES_NAME` / `ASSET_NAME` / `METADATA_KEY`。
WAN_TEXT_EMBEDS_SERIES = "wan2.1-t2v-1.3b-text-embeds"
WAN_TEXT_EMBEDS_FILE = "text_embeds.safetensors"
WAN_TEXT_EMBEDS_METADATA_KEY = "karume.wan.text_embeds"

#: 埋め込みを作った umT5 の dtype（メタの `text_encoder.dtype` — 決定 4。NOTICE とカードが
#: 「bfloat16 の上流の encoder で作った」と名乗る根拠）。書き手の綴りは
#: `wan.text_embeds.asset_metadata`（`TEXT_ENCODER_DTYPE` から作る）。
WAN_TEXT_ENCODER_DTYPE = "bfloat16"

#: 埋め込みの値を決める依存の版（メタの `versions` — 決定 4 の「固定」）。diffusers は上流の
#: `_get_t5_prompt_embeds` / `prompt_clean`、ftfy は正規化の規則、transformers は UMT5 の実装と
#: トークナイザ（版で語彙外の id の扱いが変わる — research 2026-10-03 umT5）を持つ。値は
#: pyproject の `wan` グループの `==` ピンと同じ（`wan/tests/test_distribution.py` の門が両方を
#: 突き合わせる）。
WAN_TEXT_EMBEDS_VERSIONS: Mapping[str, str] = {
    "diffusers": "0.39.0",
    "ftfy": "6.3.1",
    "transformers": "5.14.1",
}

#: 系列の部品ディレクトリに置かれる容器の代表名（分割形は `model-0000N-of-0000M.krm`）。
WAN_MODEL_FILE = "model.krm"

#: グラフを持つ部品名 = manifest の weights のキー = **容器のグラフ名**（container-v1 §2.1 —
#: ランタイムは `prepareContainer(opened, <weights キー>)` で名前で引く）。系列の部品
#: ディレクトリ名も同じ綴りだが、それは規約であって導出ではない（書き手は `TARGET` /
#: `TARGETS` / `GRAPH_NAME` を名乗る — `tests/test_graph_names.py` の門）。`text_encoder` は
#: umT5 の配布形と同じキー（越境参照の先の容器のグラフ名 — 綴りの正本は
#: `wan.umt5_distribution.UMT5_ROLE`）。並びは生成の段の順（text → DiT → VAE）。
WAN_TEXT_ENCODER_ROLE = UMT5_ROLE
WAN_TRANSFORMER_ROLE = "transformer"
WAN_VAE_FIRST_ROLE = "vae_decoder_first"
WAN_VAE_NEXT_ROLE = "vae_decoder_next"
WAN_GRAPH_ROLES: tuple[str, ...] = (
    WAN_TEXT_ENCODER_ROLE,
    WAN_TRANSFORMER_ROLE,
    WAN_VAE_FIRST_ROLE,
    WAN_VAE_NEXT_ROLE,
)

#: `transformer` の容器の配置の役割（格納ラベルごとに 1 本 — ADR 0120 決定 1。anima の
#: `transformer_f16` / `transformer_i8` と同じ綴り）。配置表と格納の要求 / 禁止表の鍵で、manifest の
#: weights のキーは容器のグラフ名 {@link WAN_TRANSFORMER_ROLE} のまま。
WAN_TRANSFORMER_F16_ROLE = "transformer_f16"
WAN_TRANSFORMER_I8_ROLE = "transformer_i8"
WAN_TRANSFORMER_ROLES: tuple[str, ...] = (WAN_TRANSFORMER_F16_ROLE, WAN_TRANSFORMER_I8_ROLE)

#: 容器を持つ配置の役割（text_encoder は i8 の 1 本・transformer は格納ラベルごとに 2 本・VAE は
#: f16 の 1 本ずつ）。
WAN_CONTAINER_ROLES: tuple[str, ...] = (
    WAN_TEXT_ENCODER_ROLE,
    *WAN_TRANSFORMER_ROLES,
    WAN_VAE_FIRST_ROLE,
    WAN_VAE_NEXT_ROLE,
)

#: モデル単位の資産（manifest の `assets` のキー = 役割名 — TS 側 `pipeline.ts` の `TEXT_EMBEDS`）。
WAN_TEXT_EMBEDS_ROLE = "text_embeds"

#: umT5 のトークナイザと前処理の表の資産（manifest の `assets` のキー = 役割名 — TS 側の読み手は
#: `parseWanTokenizerAsset`）。資産は quant にも経路にも依らず全数を取るので、umT5 のリポではなく
#: このリポに置く（越境にすると、umT5 を取らない precomputed の経路まで umT5 のリポに触る —
#: ADR 0119 追記 C）。
WAN_TOKENIZER_ROLE = "umt5_tokenizer"

#: トークナイザ資産が名乗る上流の部品（書き手 `wan.umt5_tokenizer.build_asset` の
#: `source.subfolder`）。
WAN_TOKENIZER_SUBFOLDER = "tokenizer"

#: トークナイザ資産の版のうち、id 列と前処理の結果を決めるもの（`wan` グループの `==` ピン —
#: {@link WAN_TEXT_EMBEDS_VERSIONS} と同じ値を引く。transformers は id 列の正本・ftfy は前処理の表の
#: 出所 — ADR 0119 決定 1 / 2）。
WAN_TOKENIZER_VERSIONS: tuple[str, ...] = ("ftfy", "transformers")

#: `transformer` の容器が宣言する RoPE 素表の資産名と役割（書き手は `wan.export_dit` の
#: `ROPE_BASE_ASSET` / `ROPE_BASE_ROLE`・読み手は TS 側 `pipeline.ts` の `ROPE_BASE`）。
WAN_ROPE_BASE_ASSET = "rope_base"
WAN_ROPE_BASE_ROLE = "rope-base"

#: VAE の chunk グラフの潜在入力（先頭の入力 `[C, 1, t, t]` — 残りの入力は cache）。書き手の綴りは
#: `wan.export_vae.LATENT_INPUT`・読み手は TS 側 `vae-chunks.ts` の `WAN_VAE_LATENT_INPUT`。
WAN_VAE_LATENT_INPUT = "latent"

#: DiT の文脈入力（`[1, rows, width]` — 埋め込み資産の行の幅と有効長の上限をここから引く）。
#: 書き手の綴りは `wan.export_dit.INPUT_NAMES`。
WAN_DIT_CONTEXT_INPUT = "encoder_hidden_states"

#: 出力の相対 path（**モデルサブツリー内**）— 配置表と manifest が共有する 1 箇所。重みの path に
#: 格納ラベル（`f16` / `i8`）を入れるのは、格納の席を足した日に既存の path を動かさないため
#: （Depth Anything の `model.f32.krm` と同じ綴り — i8 の席を足した ADR 0120 で f16 の path は
#: 動いていない）。
#:
#: MUST: `text_encoder` は umT5 の配布形の同じ部品の path（`UMT5_OUTPUT_PATHS`）を引く — 越境参照は
#: 参照元の `karume.json` が宣言する `<モデル名>/<この path>` の part を引き当てる
#: （`karume.dist.external_refs`）ので、綴りが割れると組み立てが「参照元に無い」で落ちる。
WAN_OUTPUT_PATHS: Mapping[str, str] = {
    WAN_TEXT_ENCODER_ROLE: UMT5_OUTPUT_PATHS[UMT5_ROLE],
    WAN_TRANSFORMER_F16_ROLE: f"{WAN_TRANSFORMER_ROLE}/model.f16.krm",
    WAN_TRANSFORMER_I8_ROLE: f"{WAN_TRANSFORMER_ROLE}/model.i8.krm",
    WAN_VAE_FIRST_ROLE: f"{WAN_VAE_FIRST_ROLE}/model.f16.krm",
    WAN_VAE_NEXT_ROLE: f"{WAN_VAE_NEXT_ROLE}/model.f16.krm",
    WAN_TEXT_EMBEDS_ROLE: f"{WAN_TEXT_EMBEDS_ROLE}/{WAN_TEXT_EMBEDS_FILE}",
    WAN_TOKENIZER_ROLE: f"{WAN_TOKENIZER_ROLE}/{umt5_tokenizer.ASSET_FILE}",
}

#: 格納 dtype の要求（素の f32 資産が組み立て・ロード・実行を全て通って参照一致の門まで沈黙した
#: 実測事故 — Anima / SBV2 / Depth Anything と同じ根拠）。f16 系列は fake-quant 対象だけが f16 に
#: なる（bias / norm は f32 のまま）ので「f16 を含む」を、i8 系列は linear の重みだけが i8 になる
#: （scale・bias・norm は f32）ので「i8 を含む」を要求する。umT5 の i8 系列の種類ごとの格納は
#: 束縛表の門（`wan.umt5_distribution.assert_umt5_bindings`）が別に見る。
WAN_STORAGE_REQUIREMENTS: Mapping[str, str] = {
    WAN_TEXT_ENCODER_ROLE: "i8",
    WAN_TRANSFORMER_F16_ROLE: "f16",
    WAN_TRANSFORMER_I8_ROLE: "i8",
    WAN_VAE_FIRST_ROLE: "f16",
    WAN_VAE_NEXT_ROLE: "f16",
}

#: codec 台帳の layout のうち圧縮格納の全部（`karume.container.CODEC_LEDGER` から f32 / i32 を
#: 除いたもの — 台帳との一致は `wan/tests/test_distribution.py` が見る）。
_COMPRESSED_LAYOUTS = ("bf16", "f16", "i8", "i4", "i2")

#: 各役割の束縛表に**あってはならない**格納の語彙（{@link assert_storage_absent}）。存在検査だけ
#: では、要求の格納を含む混成の系列を挿す取り違えが素通りする — 例えば i4 の混成系列（既定の
#: 格納が i8 — anima の i4 系列の形）は「i8 を含む」を満たすので、i8 の席へ挿しても要求検査を
#: 通る。f16 系列を i8 の席へ・i8 系列を f16 の席へ挿す取り違えは、要求の不在と禁止の両側で落ちる。
#:
#: MUST: 役割ごとに、圧縮格納のうち要求する 1 つを除いた**全部**を名指しする — 1 つでも抜けると、
#: 抜けた格納形だけが黙って素通りする（Depth Anything と同じ規律）。
WAN_STORAGE_FORBIDDEN: Mapping[str, tuple[str, ...]] = {
    role: tuple(layout for layout in _COMPRESSED_LAYOUTS if layout != required)
    for role, required in WAN_STORAGE_REQUIREMENTS.items()
}

#: weights の宣言（容器のグラフ名 → dtype ラベル → 配置の役割）。ラベルは格納 dtype 語彙で、
#: {@link WAN_STORAGE_REQUIREMENTS} が要求する格納形と 1:1（ADR 0041 §3）。`text_encoder` は
#: ラベルが i8 の 1 つなので、{@link complete_quant_weights} が全席へ埋める（席の weights は完全
#: 写像 — umT5 を持たない席は表せない。経路の選択は構築時のオプション — ADR 0119 追記 B）。
WAN_WEIGHTS: Mapping[str, Mapping[str, WeightFiles]] = {
    WAN_TEXT_ENCODER_ROLE: {"i8": WeightFiles(WAN_TEXT_ENCODER_ROLE)},
    WAN_TRANSFORMER_ROLE: {
        "f16": WeightFiles(WAN_TRANSFORMER_F16_ROLE),
        "i8": WeightFiles(WAN_TRANSFORMER_I8_ROLE),
    },
    WAN_VAE_FIRST_ROLE: {"f16": WeightFiles(WAN_VAE_FIRST_ROLE)},
    WAN_VAE_NEXT_ROLE: {"f16": WeightFiles(WAN_VAE_NEXT_ROLE)},
}

#: assets の宣言（quant 選択に依存しない無条件ファイル — ADR 0041 §3・決定 4）。
WAN_ASSETS: Mapping[str, str] = {
    WAN_TEXT_EMBEDS_ROLE: WAN_TEXT_EMBEDS_ROLE,
    WAN_TOKENIZER_ROLE: WAN_TOKENIZER_ROLE,
}

#: 席名の部品上書きトークン → その weights 名（ADR 0074 決定 4 — **略称の定義は recipe が持ち、
#: 生成モデルカードの quant 表に対応を必ず出す**）。Wan の基底格納は `f16`（VAE は f16 固定）で、
#: 圧縮が掛かるのは transformer だけなので席名は `f16+dit8…` になる（anima と同じ略称 — ADR 0120
#: 決定 1）。
WAN_QUANT_ABBREVIATIONS: Mapping[str, str] = {"dit": WAN_TRANSFORMER_ROLE}

#: quant 表（ADR 0118 決定 7・ADR 0120 決定 1）。VAE は格納ラベルが 1 つなので weights に
#: 書かない（{@link complete_quant_weights} が完全写像へ埋める）。transformer は f16 / i8 の 2 つ
#: なので席ごとに書く。`session` の 3 キーは TS 側の受理表（`WAN_SESSION_POLICY`）が受ける欄で、
#: 実用席の宣言は anima の同名の席と同じ。
#:
#: - `f16+dit8` は参照席（`session` が空 — 実用席と同じ i8 の重みで、自機 A/B 門の比較相手。
#:   ADR 0110 決定 5 ②）。
#: - 中間の席（`f16+dit8-a8` / `f16+dit8-a8-attn8`）は作らない（決定 1 — ノブ単位の切り分けは
#:   テストの manifest 上書きで足りる）。
#:
#: `label` / `description` は選択 UI 向けの表示欄（ADR 0075 決定 1 — 英語・64 / 200 字上限）。速度と
#: 品質は書かない — a8 の席の速度は B570 の実測（1 forward で −50% — ADR 0120 段 4）しか無く、
#: Metal では a8 が速くならない（調査 §3.1）ので、GPU を問わない表示欄には書けない。品質（段 6 の
#: 視認で `f16` と比べて明確な劣化なし — ADR 0120 裁定 2026-10-04 の 1）と席ごとの所要は、カード
#: （`wan/card.py`）が席を名乗って書く。既定であることも書かない（`defaultQuant` が指している —
#: ADR 0075 決定 3）。
WAN_QUANTS: Mapping[str, Any] = {
    "f16": {
        "weights": {WAN_TRANSFORMER_ROLE: "f16"},
        "session": {},
        "label": "Full quality (f16)",
        "description": "Transformer and VAE weights in f16 storage with f32 compute — the"
        " largest download.",
    },
    "f16+dit8": {
        "weights": {WAN_TRANSFORMER_ROLE: "i8"},
        "session": {},
        "label": "Half-size transformer (int8)",
        "description": "Transformer weights stored as int8 (one scale per output channel) and"
        " computed in f32; the VAE stays f16. About half the transformer download.",
    },
    "f16+dit8-a8-attn8-s16": {
        "weights": {WAN_TRANSFORMER_ROLE: "i8"},
        "session": {
            "linearCompute": "a8",
            "attentionCompute": "a8",
            "attentionScoreStorage": "f16",
        },
        "label": "int8 transformer, int8 activations",
        "description": "The int8 transformer with per-token int8 activations in its linear layers"
        " and attention (integer dot products), and attention scores held in f16.",
    },
}
#: 既定席は実用席（ADR 0120 裁定 2026-10-04 の 4 — 既定は今いちばん実用的な席: 軽く、品質の劣化が
#: 小さい。段 6 の視認で `f16` と比べて明確な劣化は無かった）。`f16` 席は明示の指定で使う参照側の
#: 席。
WAN_DEFAULT_QUANT = "f16+dit8-a8-attn8-s16"

#: パイプライン所有の設定（hub は素通し — ADR 0041 §2・TS 側のスキーマは
#: `packages/models/src/wan/config.ts`）。値は参照出力を作る設定（決定 5）:
#:
#: - `scheduler.shift` 3.0 — 上流 Diffusers 版の `scheduler/scheduler_config.json` の `flow_shift`
#:   （docstring「3.0 for 480P」）。製品の既定はこの値で起こし、視認の A/B で変えるかを決める
#:   （決定 5・裁定 2）。
#: - `defaults.steps` 50 / `defaults.guidance` 5.0 — 上流 `WanPipeline.__call__` の既定（決定 5）。
#:
#: UniPC の構造（solver の次数・bh2・`num_train_timesteps`）は宣言しない — TS の移植が実装している
#: 分岐はそれ 1 つで（`scheduler.ts` の `WAN_UNIPC_CONFIG`）、宣言できるようにすると検証していない
#: 組み合わせを配布側が選べてしまう。
WAN_PIPELINE_CONFIG: Mapping[str, Any] = {
    "scheduler": {"shift": 3.0},
    "defaults": {"steps": 50, "guidance": 5.0},
}


@dataclass(frozen=True)
class WanSources:
    """組み立ての入力 = 系列ディレクトリ 3 本（グラフ 3 本の f16 系列・DiT の i8 系列・テキスト
    埋め込みの系列）と、umT5 の容器（i8 系列）・トークナイザ資産。"""

    series: Path
    i8_series: Path
    text_embeds: Path
    #: umT5 の容器の代表 path（系列 `umt5-xxl-i8-dyn` —
    #: `wan.umt5_distribution.umt5_container`）。
    text_encoder: Path
    #: トークナイザ資産（系列 `wan2.1-umt5-tokenizer` — 書き手は `wan.umt5_tokenizer`）。
    tokenizer: Path


def wan_sources(series_dir: Path) -> WanSources:
    """系列の親ディレクトリ（`outputs/series/`）から入力を引く。"""
    return WanSources(
        series=series_dir / WAN_SERIES,
        i8_series=series_dir / WAN_I8_SERIES,
        text_embeds=series_dir / WAN_TEXT_EMBEDS_SERIES / WAN_TEXT_EMBEDS_FILE,
        text_encoder=umt5_container(series_dir),
        tokenizer=series_dir / umt5_tokenizer.SERIES_NAME / umt5_tokenizer.ASSET_FILE,
    )


def wan_placements(sources: WanSources) -> dict[str, Path]:
    """配置の役割 → 出所のファイル。出力の path は {@link WAN_OUTPUT_PATHS} が持つ。

    この表に無いものは出力へ入らない（系列に並ぶ `io.*` / `reference.*` / `vae_*.safetensors` /
    `pipeline_steps.*` の golden はこれで落ちる）。
    """
    return {
        WAN_TEXT_ENCODER_ROLE: sources.text_encoder,
        WAN_TRANSFORMER_F16_ROLE: sources.series / WAN_TRANSFORMER_ROLE / WAN_MODEL_FILE,
        WAN_TRANSFORMER_I8_ROLE: sources.i8_series / WAN_TRANSFORMER_ROLE / WAN_MODEL_FILE,
        WAN_VAE_FIRST_ROLE: sources.series / WAN_VAE_FIRST_ROLE / WAN_MODEL_FILE,
        WAN_VAE_NEXT_ROLE: sources.series / WAN_VAE_NEXT_ROLE / WAN_MODEL_FILE,
        WAN_TEXT_EMBEDS_ROLE: sources.text_embeds,
        WAN_TOKENIZER_ROLE: sources.tokenizer,
    }


def wan_repo_name(_model: str) -> str:
    """配布リポ名（家族 1 リポなので、どのモデルでも同じ 1 つ — ADR 0092 決定 1）。"""
    return WAN_REPO_NAME


def assert_rope_base(container: Path) -> None:
    """`transformer` の容器が RoPE 素表の資産を宣言していることを見る（payload は読まない）。

    MUST: 組み立てで落とす。素表を持たない DiT（静的形の別 export・資産を足す前の書き手）を
    transformer 席へ挿すと、組み立ても `verify_dist` も通り、利用者の `fromPretrained` が重みを
    落とした後で「資産 'rope_base' が無い」になる。
    """
    try:
        declared = read_asset_declarations(container)
    except ContainerReadError as cause:
        raise DistError(f"{container}: {cause}") from cause
    asset = declared.get(WAN_ROPE_BASE_ASSET)
    if asset is None:
        raise DistError(
            f"{container}: 資産 '{WAN_ROPE_BASE_ASSET}' が無い（宣言: {sorted(declared)}）—"
            " RoPE の素表を持たない DiT は配らない（`python -m wan.export_dit` で焼き直す）"
        )
    if asset[0] != WAN_ROPE_BASE_ROLE:
        raise DistError(
            f"{container}: 資産 '{WAN_ROPE_BASE_ASSET}' の役割が {asset[0]!r}"
            f"（期待 {WAN_ROPE_BASE_ROLE!r}）"
        )


def assert_shared_rope_base(containers: Sequence[Path]) -> None:
    """`transformer` の容器（格納ラベルごと — f16 / i8）の RoPE 素表がバイト同一であることを見る。

    MUST: 組み立てで落とす。素表は容器ごとに 1 本入り、TS 側は選んだ席の容器から読む。食い違ったまま
    配ると、片方の席だけが別の幾何の RoPE で走り、ロードも実行も通って映像だけが静かに壊れる
    （anima の `assert_shared_rope_base` と同じ理由）。i8 の容器の素表の誤りは、同じ i8 の容器
    どうしを比べる自機 A/B 門（ADR 0120 決定 4）では掴めない。
    """
    digests: dict[Path, str] = {}
    for path in containers:
        try:
            payload = read_asset(path, WAN_ROPE_BASE_ASSET)
        except ContainerReadError as cause:
            raise DistError(f"{path}: {cause}") from cause
        digests[path] = hashlib.sha256(payload).hexdigest()
    if len(set(digests.values())) != 1:
        listing = "\n".join(f"  {digest}  {path}" for path, digest in digests.items())
        raise DistError(
            f"資産 '{WAN_ROPE_BASE_ASSET}' が transformer の容器の間でバイト同一でない — 同じ幾何で"
            f"配れない。どちらが正かはここでは決められないので組み立てを止める:\n{listing}"
        )


def dit_context(container: Path) -> tuple[int, int]:
    """DiT の文脈入力 `[1, rows, width]` の `(rows, width)`（静的次元 — 記号なら落とす）。"""
    shape = graph_inputs(ir_graph(container), container).get(WAN_DIT_CONTEXT_INPUT)
    if (
        shape is None
        or len(shape) != 3
        or not all(isinstance(dim, int) and not isinstance(dim, bool) for dim in shape)
    ):
        raise DistError(
            f"{container}: グラフ入力 '{WAN_DIT_CONTEXT_INPUT}' が静的な [1, rows, width] でない"
            f"（{shape!r}）"
        )
    return shape[1], shape[2]


def assert_vae_chunk_pair(first: Path, following: Path) -> None:
    """`vae_decoder_first` / `vae_decoder_next` の 2 本が同じ組の chunk グラフであることを見る。

    規則は TS の `wanVaeChunkLayout`（`packages/models/src/wan/vae-chunks.ts`）と同じ: 潜在入力の
    形が 2 本で同じ（= タイル辺が同じ）で、first の cache 入力が next の cache 入力の部分列（同じ
    名前・同じ形・同じ順）。

    MUST: 組み立てで落とす。書き手は 2 本を別の容器へ書くので、片方だけを焼き直した系列（`--target`
    の部分更新・途中で落ちた旧版の書き手の実走）は、各容器の検査を全部通ったまま配布形に据わり、
    利用者の `fromPretrained` が admission で初めて拒む。同じ形で中身の世代だけが違う組は形からは
    見分けられない（GPU の chunk 列の照合だけが捕まえる）。
    """
    first_inputs = graph_inputs(ir_graph(first), first)
    next_inputs = graph_inputs(ir_graph(following), following)
    for path, inputs in ((first, first_inputs), (following, next_inputs)):
        if next(iter(inputs), None) != WAN_VAE_LATENT_INPUT:
            raise DistError(
                f"{path}: 先頭のグラフ入力が '{WAN_VAE_LATENT_INPUT}' でない（{list(inputs)}）"
            )
    if first_inputs[WAN_VAE_LATENT_INPUT] != next_inputs[WAN_VAE_LATENT_INPUT]:
        raise DistError(
            f"{first} / {following}: 潜在入力の形が first {first_inputs[WAN_VAE_LATENT_INPUT]} と"
            f" next {next_inputs[WAN_VAE_LATENT_INPUT]} で違う — 別のタイル辺で焼いた 2 本を"
            " 組にしない（`python -m wan.export_vae` で両方を焼き直す）"
        )
    next_caches = [name for name in next_inputs if name != WAN_VAE_LATENT_INPUT]
    cursor = 0
    for name, shape in first_inputs.items():
        if name == WAN_VAE_LATENT_INPUT:
            continue
        if next_inputs.get(name) != shape:
            raise DistError(
                f"{first}: cache 入力 '{name}' {shape} が next に同じ形で無い"
                f"（{next_inputs.get(name)}）— 別の組の chunk グラフを組にしない"
            )
        position = next_caches.index(name)
        if position < cursor:
            raise DistError(f"{first}: cache 入力の順が next と違う（'{name}'）")
        cursor = position + 1


def _safetensors_header(path: Path) -> tuple[dict[str, Any], dict[str, str]]:
    """safetensors のヘッダ（テンソルの宣言・メタ）だけを読む（本体は読まない・torch 不使用）。"""
    try:
        with path.open("rb") as stream:
            (length,) = struct.unpack("<Q", stream.read(8))
            header = json.loads(stream.read(length))
    except (OSError, struct.error, ValueError) as cause:
        raise DistError(f"{path}: safetensors のヘッダを読めない — {cause}") from cause
    if not isinstance(header, dict):
        raise DistError(f"{path}: safetensors のヘッダがオブジェクトでない")
    metadata = header.pop("__metadata__", {})
    if not isinstance(metadata, dict):
        raise DistError(f"{path}: safetensors のメタがオブジェクトでない")
    return header, metadata


def assert_text_embeds(path: Path, model: str, context: tuple[int, int]) -> None:
    """テキスト埋め込み資産が、上流の pin・固定プロンプトの表・DiT の文脈入力と噛み合うことを見る。

    MUST: 組み立てで落とす。資産はグラフでも重みでもない表なので `verify_dist` の容器検査には
    掛からず、別の revision の umT5 で作った資産・プロンプトの表と食い違う資産・幅の違う資産が
    そのまま配布形に据わる。読み手（TS の `parseWanTextEmbeds`）が見るのは形だけなので、出所の
    食い違いは利用者の手元で「別の文脈で生成された動画」として沈黙する。

    見るのは 6 つ: メタのキーが 1 つ（書き手の MUST — バイト同一の前提）・出所（repo / revision）が
    {@link SOURCES} の pin と一致・encoder の dtype と依存の版が決定 4 の固定値
    （{@link WAN_TEXT_ENCODER_DTYPE} / {@link WAN_TEXT_EMBEDS_VERSIONS} — NOTICE とカードの記述の
    根拠）・プロンプトの並び（名前・役割・原文）が {@link FIXED_PROMPTS} と一致（カードが同じ表から
    本文を描く）・各行の正規化後の文字列 `normalized` が空でない文字列で、原文と正規化後の文字列が
    2 つの行に当たらない（TS の `parseWanTextEmbeds` が同じ規則で拒む — 配る前に落とす）・テンソルが
    `F32 [tokens, width]` で `tokens` はメタのトークン数と一致し DiT の文脈の行数以下・`width` は
    文脈の幅。`normalize(prompt) == normalized` の一致は見ない（ここは ftfy を読まない — 書き手の
    テストが見る）。
    """
    assert_component_present(path)
    tensors, metadata = _safetensors_header(path)
    if set(metadata) != {WAN_TEXT_EMBEDS_METADATA_KEY}:
        raise DistError(
            f"{path}: メタのキー {sorted(metadata)} が {WAN_TEXT_EMBEDS_METADATA_KEY} 1 つでない"
        )
    try:
        meta = json.loads(metadata[WAN_TEXT_EMBEDS_METADATA_KEY])
    except ValueError as cause:
        raise DistError(f"{path}: メタが JSON として読めない — {cause}") from cause
    if not isinstance(meta, dict):
        raise DistError(f"{path}: メタがオブジェクトでない")
    source = SOURCES[model]
    expected_source = {"repo": source.repo, "revision": source.revision}
    if meta.get("source") != expected_source:
        raise DistError(
            f"{path}: 埋め込みの出所 {meta.get('source')!r} が上流の pin {expected_source!r} と違う"
            " — 別の checkpoint の umT5 で作った資産は配らない（`python -m wan.text_embeds`）"
        )
    encoder = meta.get("text_encoder")
    encoder_dtype = encoder.get("dtype") if isinstance(encoder, dict) else None
    if encoder_dtype != WAN_TEXT_ENCODER_DTYPE:
        raise DistError(
            f"{path}: 埋め込みを作った umT5 の dtype {encoder_dtype!r} が決定 4 の"
            f" {WAN_TEXT_ENCODER_DTYPE!r} でない — NOTICE とカードの記述と食い違う資産は配らない"
        )
    versions = meta.get("versions")
    pinned = (
        {name: versions.get(name) for name in WAN_TEXT_EMBEDS_VERSIONS}
        if isinstance(versions, dict)
        else None
    )
    if pinned != WAN_TEXT_EMBEDS_VERSIONS:
        raise DistError(
            f"{path}: 埋め込みを作った版 {pinned!r} が決定 4 の固定"
            f" {dict(WAN_TEXT_EMBEDS_VERSIONS)!r} と違う — 正規化と埋め込みの経路が版で変わる"
        )
    prompts = meta.get("prompts")
    if not isinstance(prompts, list) or not all(isinstance(entry, dict) for entry in prompts):
        raise DistError(f"{path}: メタの prompts が並びでない")
    declared = [(entry.get("name"), entry.get("role"), entry.get("prompt")) for entry in prompts]
    expected = [(prompt.name, prompt.role, prompt.text) for prompt in FIXED_PROMPTS]
    if declared != expected:
        raise DistError(
            f"{path}: プロンプトの並び {[name for name, _, _ in declared]} が固定プロンプトの表"
            f"（wan.prompts.FIXED_PROMPTS — {[name for name, _, _ in expected]}）と名前・役割・"
            "原文で一致しない — カードは表から本文を描くので、食い違ったまま配らない"
        )
    owners: dict[str, str] = {}
    for entry in prompts:
        name, normalized = entry["name"], entry.get("normalized")
        if not isinstance(normalized, str) or not normalized:
            raise DistError(
                f"{path}: '{name}' の normalized が空でない文字列でない（{normalized!r}）— TS の"
                " parseWanTextEmbeds が利用者の手元で拒む"
            )
        # 同じ行の原文と正規化後の文字列が等しいのは許す（集合で 1 つに畳む — TS と同じ）。
        for text in {entry["prompt"], normalized}:
            owner = owners.setdefault(text, name)
            if owner != name:
                raise DistError(
                    f"{path}: '{name}' の文字列が '{owner}' と同じ — どちらの埋め込みを使うかが"
                    " 決まらない（TS の parseWanTextEmbeds が拒む）"
                )
    rows, width = context
    if sorted(tensors) != sorted(name for name, _, _ in expected):
        raise DistError(f"{path}: テンソル {sorted(tensors)} がメタのプロンプトと対応しない")
    for entry in prompts:
        name, tokens = entry["name"], entry.get("tokens")
        tensor = tensors[name]
        shape = tensor.get("shape") if isinstance(tensor, dict) else None
        if (
            not isinstance(tensor, dict)
            or tensor.get("dtype") != "F32"
            or shape != [tokens, width]
            or not isinstance(tokens, int)
            or not 0 < tokens <= rows
        ):
            raise DistError(
                f"{path}: '{name}' が F32 [{tokens}, {width}]（有効長 1〜{rows}）でない"
                f"（{tensor!r}）— DiT の文脈入力 [1, {rows}, {width}] と噛み合わない"
            )


def assert_umt5_tokenizer(path: Path, model: str, rows: int) -> None:
    """トークナイザ資産が、TS の読む形式・上流の pin・依存の版・DiT の文脈の行数と噛み合うことを
    見る。

    MUST: 組み立てで落とす。資産は表なので `verify_dist` の容器検査には掛からず、読み手
    （`parseWanTokenizerAsset`）が見るのは形式と形だけ — 別の revision のトークナイザ・別の版の
    ftfy で焼いた前処理の表は、利用者の手元で「別の id 列」として沈黙する。

    見るのは 4 つ: 形式の版（`wan.umt5_tokenizer.ASSET_FORMAT`）・出所（repo / revision /
    subfolder）が {@link SOURCES} の pin と一致・版（{@link WAN_TOKENIZER_VERSIONS}）が `wan`
    グループのピンと一致・`maxLength`（TS が受ける token 数の上限）が DiT の文脈の行数以下（GPU
    経路の出力をその行数まで詰める — ADR 0119 決定 4）。
    """
    assert_component_present(path)
    try:
        asset = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as cause:
        raise DistError(f"{path}: トークナイザ資産が JSON として読めない — {cause}") from cause
    if not isinstance(asset, dict):
        raise DistError(f"{path}: トークナイザ資産がオブジェクトでない")
    if asset.get("format") != umt5_tokenizer.ASSET_FORMAT:
        raise DistError(
            f"{path}: 形式 {asset.get('format')!r} が {umt5_tokenizer.ASSET_FORMAT!r} でない"
            " — TS の parseWanTokenizerAsset は知らない版を読まない"
            "（`python -m wan.umt5_tokenizer`）"
        )
    source = SOURCES[model]
    expected_source = {
        "repo": source.repo,
        "revision": source.revision,
        "subfolder": WAN_TOKENIZER_SUBFOLDER,
    }
    if asset.get("source") != expected_source:
        raise DistError(
            f"{path}: トークナイザの出所 {asset.get('source')!r} が上流の pin {expected_source!r}"
            " と違う — 別の checkpoint のトークナイザは配らない"
        )
    versions = asset.get("versions")
    pinned = (
        {name: versions.get(name) for name in WAN_TOKENIZER_VERSIONS}
        if isinstance(versions, dict)
        else None
    )
    expected_versions = {name: WAN_TEXT_EMBEDS_VERSIONS[name] for name in WAN_TOKENIZER_VERSIONS}
    if pinned != expected_versions:
        raise DistError(
            f"{path}: トークナイザ資産を焼いた版 {pinned!r} が `wan` グループのピン"
            f" {expected_versions!r} と違う — id 列と前処理の表が版で変わる"
        )
    max_length = asset.get("maxLength")
    if (
        not isinstance(max_length, int)
        or isinstance(max_length, bool)
        or not 0 < max_length <= rows
    ):
        raise DistError(
            f"{path}: maxLength {max_length!r} が DiT の文脈の行数 1〜{rows} に収まらない"
            " — GPU 経路の出力を文脈へ詰められない"
        )


def wan_plan(sources: WanSources, model: str = DEFAULT_MODEL) -> ModelPlan:
    """Wan2.1 の 1 モデルぶんの計画を組む（検査と読み取りをここで全部済ませる — 何も書かない）。"""
    assert_model_name(model)
    # 門は取得元の表（`SOURCES` — Wan2.2 の行も持つ）ではなく Wan2.1 のモデルの表で閉じる。
    # 配布リポ名はモデルによらず `karume-wan2.1`（{@link wan_repo_name}）なので、表の全モデルを
    # 通すと 5B の出所を名乗る配布形が Wan2.1 の配布を置き換えうる。
    if model not in WAN21_MODELS:
        raise DistError(
            f"Wan2.1 のモデル {model!r} は知らない（既知: {' / '.join(WAN21_MODELS)}）—"
            f" Wan2.1 の配布（{WAN_REPO_NAME}）は Wan2.1 のモデル（wan.sources.WAN21_MODELS）だけを"
            " 配る"
        )
    placements = wan_placements(sources)
    upstream = SOURCES[model]
    for role in WAN_CONTAINER_ROLES:
        container = placements[role]
        assert_component_present(container)
        assert_storage(role, container, WAN_STORAGE_REQUIREMENTS)
        assert_storage_absent(role, container, WAN_STORAGE_FORBIDDEN)
        if role == WAN_TEXT_ENCODER_ROLE:
            # umT5 の容器の上流は Wan ではなく本家 `google/umt5-xxl`（ADR 0122 決定 1）— 出所は
            # 下の umT5 の門（`wan.sources.UMT5_SOURCES` の行）が突き合わせる。
            continue
        # 容器が名乗る出所を上流の pin（`wan.sources` が正本）へ突き合わせる — モデル名の表だけで
        # 門を閉じると、別の revision から焼いた容器が系列 path へ置かれたときに素通りする。
        assert_upstream_provenance(container, license=upstream.license, revision=upstream.revision)
    assert_vae_chunk_pair(placements[WAN_VAE_FIRST_ROLE], placements[WAN_VAE_NEXT_ROLE])
    # umT5 の容器は umT5 の配布形と同じ門（出所・束縛表・入出力の契約）を通す（越境参照の先は
    # この容器とバイト同一 — `karume.dist.external_refs` が見る）。モデル名は umT5 のリポの側の
    # 名前。
    encoder_width = assert_umt5_encoder(placements[WAN_TEXT_ENCODER_ROLE], UMT5_DEFAULT_MODEL)
    transformers = [placements[role] for role in WAN_TRANSFORMER_ROLES]
    for transformer in transformers:
        assert_rope_base(transformer)
        context = dit_context(transformer)
        # 埋め込み資産・トークナイザ・text_encoder は quant 非依存の 1 本なので、どの席の DiT の
        # 文脈入力とも噛み合う必要がある。
        assert_text_embeds(placements[WAN_TEXT_EMBEDS_ROLE], model, context)
        assert_umt5_tokenizer(placements[WAN_TOKENIZER_ROLE], model, context[0])
        if encoder_width != context[1]:
            raise DistError(
                f"{placements[WAN_TEXT_ENCODER_ROLE]}: umT5 の出力の幅 {encoder_width} が"
                f" {transformer} の文脈入力の幅 {context[1]} と違う — text 段の出力を DiT へ"
                " 渡せない"
            )
    assert_shared_rope_base(transformers)
    return ModelPlan(
        name=model,
        pipeline=WAN_PIPELINE,
        artifacts={
            role: Artifact(WAN_OUTPUT_PATHS[role], source=source)
            for role, source in placements.items()
        },
        weights=WAN_WEIGHTS,
        assets=WAN_ASSETS,
        quants=complete_quant_weights(WAN_WEIGHTS, WAN_QUANTS),
        default_quant=WAN_DEFAULT_QUANT,
        pipeline_config=WAN_PIPELINE_CONFIG,
    )


def wan_dist_plan(series_dir: Path, model: str) -> ModelPlan:
    """`--series` の親から Wan2.1 の 1 モデルの計画を組む（CLI のディスパッチ先）。"""
    return wan_plan(wan_sources(series_dir), model)


#: 改変告知（Apache 2.0 §4(b)）。**このリポが上流の重みへ加えた変更**を列挙する。
#:
#: MUST: 文面は配布形の中身と対応していること — 値としては妥当な散文なので `verify_dist` も
#: manifest 検査も素通りし、配ってからでないと食い違いに気づけない。
WAN_NOTICE_MARKDOWN = """# NOTICE

This repository redistributes a modified form of the Wan2.1 T2V 1.3B checkpoint listed in
`README.md` (Wan-AI, licensed under the Apache License, Version 2.0 — see `LICENSE.md`). The
following changes were made:

- The weights were converted into the Karume container format (a `.krm` part sequence whose first
  part carries the graph and model descriptors).
- **f16 storage**: every parameter of the float16 transformer and of the VAE decoder was rounded
  from the source float32 value to the nearest float16 value. Weight matrices and convolution
  kernels are stored as float16; the other parameters (biases, normalization weights) keep the
  rounded values in float32 storage. Computation runs in float32.
- **int8 transformer**: a second copy of the transformer stores the weight matrices of all its
  linear layers (the patch-embedding projection included) as int8, quantized from the source
  float32 values to the nearest step with one float32 scale per output channel (symmetric). Its
  other parameters (biases, normalization weights, modulation tables) keep the source float32
  values. The int8 weights are computed in float32, or, in the quants that declare it, multiplied
  with activations that are quantized to int8 per token at run time.
- The transformer graph takes patchified latent tokens and returns tokens: the patchify, the
  unpatchify and the sinusoidal timestep projection run on the host, and the patch-embedding
  convolution is applied as the equivalent linear layer. The rotary embedding keeps the upstream
  real-valued cos / sin tables, but the host builds them from per-axis base tables stored in the
  container, and the graph applies them in a pair-swap form (swap each adjacent pair, then multiply
  elementwise by the cos / sin tables).
- The VAE decoder was re-expressed as two graphs that decode one latent frame each (the first
  frame, and every later frame), with the causal convolution cache passed in and out of the graph
  instead of kept in a Python list. The host always decodes in overlapping tiles, so the output
  differs slightly from the upstream untiled decode.
- **The text encoder is referenced, not stored here.** `karume.json` references the umT5-XXL
  encoder of `google/umt5-xxl` (bit-identical in float32 to the `text_encoder` folder of this
  checkpoint), converted to int8, from the separate repository `karume-umt5-xxl` at a pinned commit
  (with the size and the SHA-256 of every part); the changes made to it are listed in that
  repository's own `NOTICE.md`.
- The umT5-XXL outputs of a fixed set of prompts (computed with the upstream encoder in bfloat16
  and stored as float32) are included as a precomputed asset, for use without the text encoder.
- The tokenizer of the checkpoint was converted into one JSON table (vocabulary, scores, added
  tokens and the whitespace set), together with lookup tables for the upstream prompt cleaning
  (evaluated from ftfy 6.3.1, the `regex` package and the Unicode 16.0.0 character database,
  including a translation of ftfy's mojibake-detection pattern) that the host uses to reproduce
  the cleaning or to reject a prompt.

No retraining and no fine-tuning were performed. The original checkpoint is not distributed here.
"""


#: `--pipeline wan` の 1 行（ドライバが core の PIPELINES へ合成する）。
PIPELINE = Pipeline(
    default_model=DEFAULT_MODEL,
    repo_name=wan_repo_name,
    plan=wan_dist_plan,
    # 帰属（上流リポ・ライセンス）はモデル名から一意に決まる（`wan.sources.SOURCES`）ので、
    # 選ばせる軸にしない。略称の対応表は manifest に無い事実なので、ここから渡す（anima と同じ形）。
    card_profiles={"wan": partial(render_wan_model_card, abbreviations=WAN_QUANT_ABBREVIATIONS)},
    # 上流ライセンス（Apache 2.0）の再配布条件 §4 は配布リポ 1 つに掛かるので、原文の読みも
    # 組み立ての回数によらずここで 1 回（ADR 0092 決定 7）。
    root_files={
        "LICENSE.md": apache_license_2_0(),
        "NOTICE.md": WAN_NOTICE_MARKDOWN,
    },
)
