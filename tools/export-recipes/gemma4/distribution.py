"""Gemma 4 の配布 recipe — 系列レイアウト・出力 path 表・quant 表・カードの選択（ADR 0065 決定 2）。

汎用の組み立てエンジン（配置・共有席の畳み込み・sha256・manifest・staging/swap・検証）は
`karume.dist` が持つ。ここが持つのは **gemma4 固有の事実**だけ: どの系列ディレクトリから何を
拾い、配布形のどの path へ、どの dtype ラベルで並べ、どの quant を既定にするか。

Per model the regular line ships the product graph in two containers (the int4 one below and the
original-weights one — next paragraph), the borrowing drafter for a model with an upstream drafter,
and one asset kind (ADR 0084 decision 5 / 0096 step 2 / 0109 decision 4): E2B ships three graphs,
E4B two, as it has no drafter ({@link gemma4_has_drafter}):

- `model` — 製品グラフのコンテナ（`gemma4/export_product.py` が書く `.krm` の part 列。PLE を
  グラフから外し、出口を最終行 logits にした 1 系列）。格納は**混成**で、埋め込みが i8・
  linear が packed i4。**PLE はこの容器の資産**として同梱される（ADR 0109 決定 4）— 索引
  `ple_index`（schema 3）と `ple.values.<k>` / `ple.scales.<k>` の block 列で、weights では
  ない（ホストが `per_layer_inputs` を組むための表）。読み手は
  `packages/models/src/gemma/ple-index.ts`。
- `drafter` — MTP drafter のコンテナ（`gemma4/export_drafter.py`・ADR 0096 段 2）。格納は
  **i8 単一**で、linear まで i8（i4 g32 に落とすと受理率が 1 〜 3 割落ちる — 台本の実測）。
  役割ごとに dtype ラベルが違うので、quant 表の `weights` 写像は 2 席とも埋まる。
- `tokenizer` — the compiled tokenizer asset (`gemma4/tokenizer.py` — ADR 0084 decision 1). Each
  model reads its own series today; models whose upstream `tokenizer.json` files are the same bytes
  may share one ({@link GEMMA4_TOKENIZER_SERIES}), and every model's asset is held to the upstream
  file it was compiled from ({@link assert_gemma4_tokenizer_source})

The regular line's `model` role has two storages, one container each: the int4 container above
(the `i4` presets) and the **original-weights** container `model_bf16`
(`gemma4/export_product.py --storage bf16` — {@link GEMMA4_ORIGINAL_ROLE}): every weight is the
checkpoint's own bf16 bits, the PLE table included (index schema 5 — a bf16 `values` table and no
`scales`). Its preset `bf16` is the default (ADR 0128 addendum 2026-11-04 — the maintainer's
ruling of 2026-11-04). The drafter is the same int8 container for both: it borrows
`model.lm_head.weight` from whichever target is loaded, so the plan pairs it with each of them.

The regular line ships E2B and E4B ({@link GEMMA4_REGULAR_MODELS}). 12B ships only in the q4_0
QAT line below — its one line, whose weights are already Google's QAT checkpoint. The plan handles
what sets 12B apart for that line: its `model` container has no PLE assets and its graph no
`per_layer_inputs` (the model has no per-layer embeddings — {@link gemma4_ple_dim}); it declares
its chat dialect (`pipelineConfig.chatDialect` `gemma4-unified`) and the ids generation never
selects (`pipelineConfig.suppressTokens`, from upstream's `suppress_tokens` —
{@link gemma4_generation_defaults}). Its upstream is two pinned hub snapshots of `gemma4.sources`:
the weights and their `config.json` come from Google's QAT checkpoint, the tokenizer and the
generation defaults from `google/gemma-4-12B-it` ({@link Gemma4Sources}).

## The q4_0 QAT line (`karume-gemma4-qat` — {@link QAT_Q4_0_PIPELINE})

A second distribution of the same `gemma4/1` contract: E2B / E4B / 12B from Google's q4_0 QAT
checkpoints, every text weight on its own 4-bit grid (int4 g32), each model with its QAT MTP
drafter. It runs {@link gemma4_plan} with another {@link Gemma4Line} — the graph, drafter,
tokenizer and configuration checks are the regular line's; what differs is read from the line:

- the series: E2B / E4B's `gemma4-<model>-qat-product` (`gemma4/export_product_qat.py`), 12B's
  `gemma4-12b-product` (its one line, already the QAT checkpoint), every drafter
  `gemma4-<model>-qat-drafter` (`gemma4/export_drafter_qat.py`) — {@link gemma4_qat_q4_0_sources};
- the storage: both roles int4 alone, int8 forbidden in both (the drafters are on their grid too,
  so the regular drafter's int8 requirement does not apply —
  {@link gemma4_qat_q4_0_storage_tables});
- the PLE index: schema 4, int4 values with one f32 step per group of
  {@link GEMMA4_QAT_PLE_GROUP_SIZE} ({@link gemma4_ple_index});
- the quants: `i4` (the default — the reference summation order) for every model, and for E2B /
  E4B the regular line's parallel and fused seats `i4-gemvpar` / `i4-fast` over the same storage
  ({@link gemma4_qat_q4_0_quants});
- the pins: every product container names its QAT weights' revision and every drafter container
  its QAT drafter's (`gemma4.sources.QAT_TARGET_PINS` / `QAT_DRAFTER_PINS`).

`pipelineConfig` は 2 系統に割れる（Irodori と同じ分け方）: **モデルが決める数**
（`maxPosition` = 上流 `text_config.max_position_embeddings`・`rope` = 層種別ごとの式の
パラメータ）は上流 `config.json` から導出し、**実行時ノブ**（`chunkLength` / `capacity`）と
**配布者の推奨サンプラ**（上流 `generation_config.json` の temperature / top_k / top_p —
ADR 0083 決定 7）はそれぞれの正本から引く。前者を写経すると、チェックポイントを差し替えた
日に宣言だけが古びて「宣言どおりに組んだ表が上流と別の角度で回る」形になる。

実行時ノブのうち `chunkLength` だけは**上限も宣言する**（`maxChunkLength`）— IR の `symbols`
は名前の列だけで記号の上限を持たないので、読み手は「この資産が受けられる chunk 行数」を
資産から導けない。焼く側（`gemma4.export.SYM_MAX`）が知っている唯一の数を宣言へ載せて、
trace 範囲の外の `chunkLength` を TS 側の門が落とせるようにする（2026-09-03 裁定）。

RoPE の cos / sin 表は**もう配布物に入らない** — ホスト（TS 側）が `rope` の宣言から実行時に
組む。したがって位置の上限を決めるのは資産ではなくモデルの宣言だけで、`capacity` は
`maxPosition` までの実行時ノブになる。

公開面は {@link PIPELINE} 1 つ（`karume.dist.Pipeline`）— リポの dist ドライバ
（`tools/export-recipes/dist.py`）がこれを core の PIPELINES へ合成する。
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from _shared.container_read import read_asset, read_asset_declarations
from _shared.licenses import apache_license_2_0
from _shared.upstream import assert_upstream_provenance
from karume.container import ContainerFormatError
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
    sha256_file,
)
from karume.ple import (
    PLE_FLOAT_BYTES,
    PLE_FLOAT_INDEX_SCHEMA,
    PLE_GROUPED_INDEX_SCHEMA,
    PLE_INDEX_ASSET,
    PLE_INDEX_ROLE,
    PLE_INDEX_SCHEMA,
    PLE_PACK_FACTOR,
    PLE_ROLES,
    PLE_SCALE_BYTES,
)
from karume.verify import ContainerError

from .card import (
    GEMMA4_DRAFTER_ROLE,
    GEMMA4_DRAFTER_UPSTREAM,
    GEMMA4_EMBEDDING_STORAGE,
    GEMMA4_LICENSE,
    GEMMA4_MODEL_ROLE,
    GEMMA4_PLE_MODELS,
    GEMMA4_QAT_DRAFTER_UPSTREAM,
    GEMMA4_QAT_Q4_0_WEIGHTS_UPSTREAM,
    GEMMA4_UPSTREAM,
    render_gemma4_model_card,
    render_gemma4_qat_q4_0_model_card,
)
from .chat import CHAT_DIALECTS, GEMMA4_CHAT_DIALECT, GENERATION_PROMPT_SUFFIX
from .rope import (
    BAKED_TABLE_INFIX,
    FULL_ATTENTION,
    HEAD_DIM_FIELD,
    SLIDING_ATTENTION,
    RopeSpecError,
    rope_specs,
)
from .sources import (
    QAT_DRAFTER_PINS,
    QAT_PINS,
    QAT_TARGET_PINS,
    HubPin,
    ModelPins,
    assert_model_pins,
    checkpoint_dir,
    qat_target_checkpoint_dir,
    qat_target_tokenizer_dir,
    tokenizer_dir,
)

#: パイプライン契約（ADR 0041 §2 — モデル単位）。TS 側の受理集合は
#: `GEMMA4_PIPELINE_NAME` / `GEMMA4_PIPELINE_MAJOR`（`packages/models/src/gemma/config.ts`）。
GEMMA4_PIPELINE = "gemma4/1"

#: 既定のモデル名（`--model` を省いた組み立てが入れるモデル）。綴りの受理集合は帰属表
#: （`gemma4.card.GEMMA4_UPSTREAM`）が持つ。
GEMMA4_DEFAULT_MODEL = "e2b"

#: The models the regular line (`karume-gemma4`) ships. 12B is not one of them: its only line is
#: the q4_0 QAT one (`karume-gemma4-qat` — {@link GEMMA4_QAT_Q4_0_LINE}), and the regular line
#: refuses it ({@link gemma4_sources} / {@link gemma4_plan}).
GEMMA4_REGULAR_MODELS: tuple[str, ...] = ("e2b", "e4b")

#: 系列名の接頭辞（`gemma4-<モデル名>-<接尾>`）。
GEMMA4_PREFIX = "gemma4"

#: 配布リポ名（ADR 0092 決定 1 の `karume-<family>`）。**モデル名から導かない** — the regular line
#: is one repository, whichever of its models is assembled.
GEMMA4_REPO_NAME = "karume-gemma4"

#: 系列の接尾（`gemma4/export_product.py` の `DEFAULT_OUT_DIR` と `gemma4/tokenizer.py` の
#: `ASSET_PATH` — 書き手と読み手が同じ 1 語から組む）。
GEMMA4_PRODUCT_SUFFIX = "product"
#: The original-weights product series' suffix (`outputs/series/gemma4-<model>-product-bf16/` —
#: `gemma4/export_product.py --storage bf16` writes under it, this module reads it).
GEMMA4_PRODUCT_BF16_SUFFIX = f"{GEMMA4_PRODUCT_SUFFIX}-bf16"
GEMMA4_TOKENIZER_SUFFIX = "tokenizer"
#: MTP drafter の系列接尾（`gemma4/export_drafter.py` の `DEFAULT_OUT_DIR`）。
GEMMA4_DRAFTER_SUFFIX = "drafter"
#: The q4_0 QAT line's series suffixes (`outputs/series/gemma4-<model>-qat-product/` and
#: `…-qat-drafter/` — `gemma4/export_product_qat.py` / `gemma4/export_drafter_qat.py` write under
#: these words, this module reads them).
GEMMA4_QAT_PRODUCT_SUFFIX = "qat-product"
GEMMA4_QAT_DRAFTER_SUFFIX = "qat-drafter"

#: 系列側のファイル名（`gemma4.export.MODEL_FILE` / `tokenizer.ASSET_PATH` の綴り）。
#: **代表 path** なので、分割されていれば {@link karume.dist.component_parts} が連番へ解決する。
GEMMA4_MODEL_FILE = "model.krm"
GEMMA4_TOKENIZER_FILE = "tokenizer.json"

#: 上流チェックポイントが持つ推奨サンプラの出どころ（ADR 0083 決定 7）と、モデルが決める数
#: （位置の上限・RoPE の式）の出どころ。text 部は `config.json` の `text_config` 節。
GEMMA4_GENERATION_CONFIG_FILE = "generation_config.json"
GEMMA4_CONFIG_FILE = "config.json"
#: The upstream checkpoint's tokenizer (the HF form, before compiling): what every model's compiled
#: asset is held against ({@link assert_gemma4_tokenizer_source}).
GEMMA4_CHECKPOINT_TOKENIZER_FILE = "tokenizer.json"
GEMMA4_TEXT_CONFIG_KEY = "text_config"
GEMMA4_MAX_POSITION_KEY = "max_position_embeddings"
GEMMA4_HIDDEN_SIZE_KEY = "hidden_size"

#: 役割名（manifest の weights / assets が指す内部キー）。
GEMMA4_ROLE = GEMMA4_MODEL_ROLE
#: The original-weights container's placement role: the second dtype of the `model` weights (the
#: manifest's component name stays `model` — the container's graph name).
GEMMA4_ORIGINAL_ROLE = "model_bf16"
GEMMA4_TOKENIZER_ROLE = "tokenizer"
# MTP drafter の役割名 `GEMMA4_DRAFTER_ROLE`（weights の 2 本目 — 貸し手 `model` が居ないと
# 単独では実行できない）は `gemma4.card` から import する — カードが manifest の weights から
# drafter の有無を引くので、綴りの正本を card 側に置いた（card → distribution の import は
# 循環になる）。drafter を配るモデルは `GEMMA4_DRAFTER_UPSTREAM` の表に載っているものだけ。

#: Model → the **model name** of the series that holds its compiled tokenizer asset
#: (`gemma4-<this>-tokenizer`).
#:
#: A model may borrow another model's series only while their upstream `tokenizer.json` files are
#: the same bytes (a compiled asset records its source path, so compiling it again for the second
#: model splits the bytes and the assembly can no longer fold it into `shared/`). MUST: the
#: premise is checked by machine at assembly — {@link assert_gemma4_tokenizer_source} holds the
#: asset's recorded source SHA-256 against the model's upstream `tokenizer.json`. The same check
#: holds a model's own series to its current upstream file (a series compiled before that file was
#: replaced is stale).
#: Every model reads its own series: E4B shared E2B's until Google's glitch-token fixes of
#: 2026-11-03 gave E2B's and E4B's `tokenizer.json` different merges (E2B drops 267, E4B 253), and
#: 12B's (fixed on 2026-10-30) drops 239 — the same vocabulary, but another model's asset would
#: give other ids for some inputs.
GEMMA4_TOKENIZER_SERIES: Mapping[str, str] = {"e2b": "e2b", "e4b": "e4b", "12b": "12b"}

#: グラフ入力の名前と並び（正本は `gemma4/export_product.py` — ラッパの forward 引数名）。
#: 実行側は名前で束ねるので、1 つでも綴りが変われば束ねられない。RoPE の 4 本はホストが
#: 実行時に組む cos / sin（綴りの正本は `gemma4.export_decode.rope_input_name`）。
GEMMA4_INPUT_IDS = "input_ids"
GEMMA4_PER_LAYER_INPUTS = "per_layer_inputs"
GEMMA4_LAST_ROW = "last_row"
GEMMA4_ROPE_INPUT_PREFIX = "rope_"
GEMMA4_ROPE_PARTS: tuple[str, ...] = ("cos", "sin")
GEMMA4_ROPE_LAYER_TYPES: tuple[str, ...] = (SLIDING_ATTENTION, FULL_ATTENTION)


def gemma4_rope_input_name(layer_type: str, part: str) -> str:
    """RoPE 派生入力 1 本の綴り（焼く側 `export_decode.rope_input_name` の鏡像）。"""
    return f"{GEMMA4_ROPE_INPUT_PREFIX}{layer_type}_{part}"


GEMMA4_ROPE_INPUTS: tuple[str, ...] = tuple(
    gemma4_rope_input_name(layer_type, part)
    for layer_type in GEMMA4_ROPE_LAYER_TYPES
    for part in GEMMA4_ROPE_PARTS
)
GEMMA4_GRAPH_INPUTS: tuple[str, ...] = (
    GEMMA4_INPUT_IDS,
    *GEMMA4_ROPE_INPUTS,
    GEMMA4_PER_LAYER_INPUTS,
    GEMMA4_LAST_ROW,
)
#: The inputs of a model without per-layer embeddings (12B — `gemma4/export_unified.py`): the
#: same graph contract minus `per_layer_inputs`.
GEMMA4_PLE_FREE_GRAPH_INPUTS: tuple[str, ...] = tuple(
    name for name in GEMMA4_GRAPH_INPUTS if name != GEMMA4_PER_LAYER_INPUTS
)

#: drafter グラフの入力（正本は `gemma4/export_drafter.py` — ラッパの forward 引数名）。
#: `token` = 直前に確定した token・`hidden` = target の最終 norm 後 hidden（製品グラフの出力 1）。
#: K / V は trace 用の入力を手術で external スロットへ置き換えたので、ここには載らない。
GEMMA4_DRAFTER_TOKEN = "token"
GEMMA4_DRAFTER_HIDDEN = "hidden"
GEMMA4_DRAFTER_GRAPH_INPUTS: tuple[str, ...] = (
    GEMMA4_DRAFTER_TOKEN,
    GEMMA4_DRAFTER_HIDDEN,
    *GEMMA4_ROPE_INPUTS,
)

#: drafter が 1 回の run で出す draft 本数（`gemma4.export_drafter.DRAFT_STEPS` の鏡像 —
#: 配布形の drafter はこの 3 段で焼かれている。実行時の k は `k <= 3` の範囲で選べ、それより
#: 多い段数が要るなら再 export）。出力の本数はこれと一致する MUST。
GEMMA4_DRAFT_STEPS = 3

#: 出力の相対 path（**モデルサブツリー内**）— 配置表と manifest が共有する 1 箇所。格納 dtype を
#: ファイル名に出すのは他 family と同じ形（系列が 2 本並んでも取り違えようがない綴り）。
#: PLE は `model` 容器の資産なので、この表に席を持たない（ADR 0109 決定 4）。
GEMMA4_OUTPUT_PATHS: Mapping[str, str] = {
    GEMMA4_ROLE: f"{GEMMA4_ROLE}/model.i4.krm",
    GEMMA4_ORIGINAL_ROLE: f"{GEMMA4_ROLE}/model.bf16.krm",
    GEMMA4_DRAFTER_ROLE: f"{GEMMA4_DRAFTER_ROLE}/model.i8.krm",
    GEMMA4_TOKENIZER_ROLE: f"{GEMMA4_TOKENIZER_SUFFIX}/{GEMMA4_TOKENIZER_FILE}",
}

#: 容器の束縛表に**必ず在る**格納の語彙。製品グラフは混成なので 2 つとも要求する（他
#: family の {@link assert_storage} は 1 つずつしか見ないので、表を 2 枚持って 2 度掛ける）。
#: i8 は埋め込み（i4 は runtime 上は適格〈ADR 0069 追記 6〉だが、この recipe は i8 に留める
#: 選択をしている）・i4 は linear の重み。
#: 片方だけを要求すると「埋め込みまで i4 に落ちた系列」「linear が i8 のままの系列」が
#: それぞれ素通りする — どちらも shape も manifest も正しいまま、品質と速度だけが変わる。
#: drafter は**単一格納**（linear まで i8 — `gemma4/export_drafter.py` の実測）なので、要求は
#: i8 の 1 枚だけで、2 枚目の表に行を持たない。
#: The embedding's storage is per model (`gemma4.card.GEMMA4_EMBEDDING_STORAGE`): E2B / E4B
#: require the int8 one on top of the int4 linears ({@link gemma4_storage_tables}).
#: The original-weights container holds bf16 (and f32 for the parameters bf16 does not apply to).
GEMMA4_STORAGE_REQUIREMENTS: Mapping[str, str] = {
    GEMMA4_ROLE: "i4",
    GEMMA4_ORIGINAL_ROLE: "bf16",
    GEMMA4_DRAFTER_ROLE: "i8",
}

#: The storage vocabulary each role's binding table **must not** hold
#: ({@link assert_storage_absent}). The scripts write two storage forms, each placed under its own
#: role: the int4 series (i8 + i4 + f32 — `model` and the int8 drafter) and the original-weights
#: series (bf16 + f32 — `model_bf16`). So f16 in any role, or a storage of the other form in a
#: role, can only mean a series root of another family or of the other role (a mixed-up series
#: root cannot be told apart by the numeric gates — ADR 0027 / 0029). As in the other families,
#: every compressed storage the writer can write that must not be there is named.
#: drafter は i4 も禁止側 — 存在検査（i8 が在る）は「linear だけ i4 に落ちた drafter」を
#: 素通りさせる（出力ヘッドの i8 で要求が満たされる）。受理率が 1 〜 3 割落ちるだけの資産は
#: 形も manifest も正しいままなので、ここが唯一の検出器になる。
#: The two `model` containers are held apart both ways: the int4 role forbids bf16, and the
#: original-weights role forbids every rounded storage — an int4 series placed as the `bf16` preset
#: (or the reverse) has the right graph and PLE shape, so only the storage tells it apart.
GEMMA4_STORAGE_FORBIDDEN: Mapping[str, tuple[str, ...]] = {
    GEMMA4_ROLE: ("f16", "bf16"),
    GEMMA4_ORIGINAL_ROLE: ("f16", "i8", "i4", "i2"),
    GEMMA4_DRAFTER_ROLE: ("f16", "bf16", "i4"),
}


@dataclass(frozen=True)
class Gemma4StorageTables:
    """One model's storage gates: what each role must hold (two tables, applied in turn — the
    product of E2B / E4B is mixed) and what it must not."""

    required: Mapping[str, str]
    also_required: Mapping[str, str]
    forbidden: Mapping[str, tuple[str, ...]]


def gemma4_storage_tables(model: str) -> Gemma4StorageTables:
    """The storage gates of `model` in the regular line, from its embedding storage
    (`GEMMA4_EMBEDDING_STORAGE`): its int8 embedding is required next to the int4 linears."""
    embedding = GEMMA4_EMBEDDING_STORAGE[model]
    return Gemma4StorageTables(
        required=GEMMA4_STORAGE_REQUIREMENTS,
        also_required={GEMMA4_ROLE: embedding},
        forbidden=GEMMA4_STORAGE_FORBIDDEN,
    )


#: The storage dtype labels (also the spelling of the quant seats). The drafter role has one
#: storage, so {@link complete_quant_weights} fills its seat in every preset (`drafter: "i8"`); the
#: `model` role has two, so each preset names its own (below).
GEMMA4_DTYPE = "i4"
GEMMA4_DRAFTER_DTYPE = "i8"
#: The original-weights container's dtype label, and the name of its preset.
GEMMA4_ORIGINAL_DTYPE = "bf16"

#: weights の宣言（dtype ラベル → 役割名）。分割は現物が決めるので、ここが指すのは代表 1 役。
#: MUST: drafter を **weights の 2 本目**として宣言する（assets ではない）— IR コンテナで
#: グラフを持ち、`createSession` が食う側の資産だからで、hub の `ResolveOptions.weights` が
#: 「既定では取らない」を表せる軸もここ 1 本しかない。
#: `model` has two dtypes (the int4 and the original-weights containers), so every preset names
#: the one it loads ({@link gemma4_quants} — `complete_quant_weights` refuses to pick one).
GEMMA4_WEIGHTS: Mapping[str, Mapping[str, WeightFiles]] = {
    GEMMA4_ROLE: {
        GEMMA4_DTYPE: WeightFiles(GEMMA4_ROLE),
        GEMMA4_ORIGINAL_DTYPE: WeightFiles(GEMMA4_ORIGINAL_ROLE),
    },
    GEMMA4_DRAFTER_ROLE: {GEMMA4_DRAFTER_DTYPE: WeightFiles(GEMMA4_DRAFTER_ROLE)},
}


def gemma4_has_drafter(model: str) -> bool:
    """このモデルが MTP drafter を配るか（drafter の上流表 `GEMMA4_DRAFTER_UPSTREAM` に載るか）。"""
    return model in GEMMA4_DRAFTER_UPSTREAM


def gemma4_weights(model: str) -> Mapping[str, Mapping[str, WeightFiles]]:
    """モデルの weights 宣言 — drafter を配らないモデルは `model` 役 1 本だけ。

    MUST: drafter の無いモデルに drafter 役を宣言しない。manifest の weights にその役が
    無いことが「このモデルは投機を張れない」の唯一の表現で、読み手は `speculative` 指定の
    ロードを weights の解決（hub `ResolveOptions.weights`）で fail loudly に落とす。
    """
    if gemma4_has_drafter(model):
        return GEMMA4_WEIGHTS
    return {GEMMA4_ROLE: GEMMA4_WEIGHTS[GEMMA4_ROLE]}


#: `i4` の説明のうち drafter に触れる 1 文（drafter を配るモデルの席にだけ付ける — 配って
#: いない drafter の格納を名乗らない）。
GEMMA4_DRAFTER_QUANT_NOTE = " The drafter head is int8 throughout."

#: 同じ格納系列に参照加算・並列加算・融合を用意する（ADR 0098 / 0104）。
#: `i4` は `session` を空に保つ = GEMV も attention も runtime の参照経路（ADR 0104 追記
#: 2026-09-26 — attention の縮約形も manifest の宣言になり、家族のコード既定は無い）。
#: `i4-gemvpar` は従来の実効（GEMV 並列 + attention 並列）を宣言で保存する席、`i4-fast` は
#: E2B の実測（2026-09-26・B570）で全フラグが速度低下なしだった束。
#: `i4` の説明の drafter の 1 文と、速度の実測の有無の 1 文は {@link gemma4_quants} が
#: モデルごとに足す。
#: `bf16` is the original-weights preset (ADR 0128 decision 4; the default by ADR 0128 addendum
#: 2026-11-04): the checkpoint's own bf16 values with f32 compute, in the reference summation
#: order (`session` empty).
GEMMA4_QUANTS: Mapping[str, Any] = {
    GEMMA4_DTYPE: {
        "weights": {GEMMA4_ROLE: GEMMA4_DTYPE},
        "session": {},
        "label": "Packed int4 linear, int8 embeddings",
        "description": "Packed int4 (group 32) linear weights and int8 embedding tables (a recipe"
        " choice); GEMV and attention use the reference summation order.",
    },
    "i4-gemvpar": {
        "weights": {GEMMA4_ROLE: GEMMA4_DTYPE},
        "session": {"linearGemvReduce": "parallel", "stateAttentionReduce": "parallel"},
        "label": "Packed int4 with parallel GEMV and attention",
        "description": "The same packed weights as i4, with parallel GEMV and attention "
        "summation. Rounding and generated tokens can differ. "
        "Select i4 for the reference order.",
    },
    "i4-fast": {
        "weights": {GEMMA4_ROLE: GEMMA4_DTYPE},
        "session": {
            "linearGemvReduce": "parallel",
            "fuseRmsNormAdd": True,
            "stateAttentionReduce": "parallel-fused",
        },
        "label": "Packed int4 with parallel GEMV, RMS fusion and fused attention",
        "description": "Same weights as i4; parallel GEMV, RMS-add fusion, fused parallel "
        "attention. Use i4 for reference summation, i4-gemvpar without fusion. "
        "Requires fusion-option support.",
    },
    GEMMA4_ORIGINAL_DTYPE: {
        "weights": {GEMMA4_ROLE: GEMMA4_ORIGINAL_DTYPE},
        "session": {},
        "label": "Original weights (bf16)",
        "description": "The checkpoint's own bf16 weights, the per-layer embeddings included, with"
        " f32 compute; GEMV and attention use the reference summation order.",
    },
}

#: 並列・融合の席（`i4-gemvpar` / `i4-fast`）の説明に足す、速度の実測の有無の 1 文。
#: MUST: キー集合は the regular line's models ({@link GEMMA4_REGULAR_MODELS}) と一致する（tests が
#: 固定する）— 計っていないモデルの席に「速い」と書かない（E4B の束は E4B のベンチ後に決める）。
GEMMA4_SPEED_NOTE: Mapping[str, str] = {
    "e2b": " Faster on tested E2B devices.",
    "e4b": " Speed not yet measured on E4B.",
}


def gemma4_quants(model: str) -> dict[str, Any]:
    """モデルの quant 表（席は全モデル共通の 4 つ・drafter を配るモデルだけ `i4` と `bf16` に 1 文
    足す — the two reference-order seats, one per `model` storage).

    並列・融合の 2 席には速度の実測の有無を 1 文足す（{@link GEMMA4_SPEED_NOTE}）。

    NOTE: E4B も E2B と同じ int4 の 3 席・同じ宣言を当面そのまま持つ — 束の中身は E4B の実測で
    決める。
    """
    note = GEMMA4_SPEED_NOTE[model]
    quants = {name: dict(quant) for name, quant in GEMMA4_QUANTS.items()}
    if gemma4_has_drafter(model):
        for name in (GEMMA4_DTYPE, GEMMA4_ORIGINAL_DTYPE):
            quants[name]["description"] += GEMMA4_DRAFTER_QUANT_NOTE
    for name in ("i4-gemvpar", "i4-fast"):
        quants[name]["description"] += note
    return quants


#: モデルごとの既定 quant。キー集合は the regular line's models ({@link GEMMA4_REGULAR_MODELS})
#: と一致する MUST（tests が固定する）。Both default to the original weights, `bf16` (the
#: maintainer's ruling of 2026-11-04 — ADR 0128 addendum 2026-11-04); the int4 seats stay, chosen
#: by name.
GEMMA4_DEFAULT_QUANT: Mapping[str, str] = dict.fromkeys(
    GEMMA4_REGULAR_MODELS, GEMMA4_ORIGINAL_DTYPE
)

#: 固定長 prefill chunk の行数（ADR 0066 決定 4 — context の計画時定数）。**実行時ノブ**なので
#: 資産からは導出できない。上限は記号 `M` の trace 時の上限（{@link GEMMA4_MAX_CHUNK_LENGTH}）。
GEMMA4_CHUNK_LENGTH = 768

#: 記号 `M`（1 chunk の行数）の上限。焼く側の `gemma4.export.SYM_MAX` の鏡像で、こちらは
#: torch を読まない側に置いた写し（同値は `tests/test_distribution.py` が突き合わせる）。
#:
#: これは `pipelineConfig.maxChunkLength` として**配布形に載る**（{@link gemma4_pipeline_config}）。
#: 資産からは導けない数（IR の `symbols` は名前の列だけ）なので、宣言が無いと読み手は
#: `chunkLength` の上書きが trace 範囲の内側かどうかを判定できない。
GEMMA4_MAX_CHUNK_LENGTH = 768

#: 出口の行数記号（`last_row[R]` と出力 2 本の行軸）。焼く側の `gemma4.export_product.ROW_SYMBOL`
#: の鏡像で、こちらは torch を読まない側に置いた写し（同値は `tests/test_distribution.py` が
#: 突き合わせる）。R は入力 shape が束縛するので、容量記号と違って配布形の宣言には載らない。
GEMMA4_ROW_SYMBOL = "R"

#: full スロットの容量（会話が使える最大の論理長）の**既定値**。同じく実行時ノブで、上限は
#: {@link gemma4_pipeline_config} がモデルの宣言（`maxPosition`）で押さえる。
#:
#: NOTE: **VRAM と会話長のトレードオフの政策値**。RoPE をホスト生成へ移したので資産側の
#: 位置上限は消え、`maxPosition`（E2B は 131,072）まで宣言できる — ここに置くのは
#: 「既定でどこまで確保するか」だけで、full スロットの常駐バイト数が容量に比例する
#: （`karume.limits`）ぶんが代償になる。
GEMMA4_CAPACITY = 4096

#: 上流 `generation_config.json` → `pipelineConfig.sampler` の欄名（TS 側 `SamplerSpec` の綴り）。
GEMMA4_SAMPLER_FIELDS: tuple[tuple[str, str], ...] = (
    ("temperature", "temperature"),
    ("top_k", "topK"),
    ("top_p", "topP"),
)

#: Upstream `generation_config.json`'s list of ids generation never selects → the optional
#: top-level `pipelineConfig.suppressTokens` ({@link gemma4_generation_defaults}). 12B declares
#: `[258883, 258882]` (end-of-audio / end-of-image); upstream `generate()` sets their logits to
#: −inf before greedy and sampled selection alike (`SuppressTokensLogitsProcessor`).
GEMMA4_SUPPRESS_TOKENS_KEY = "suppress_tokens"

#: `generation_config.json` keys the distribution does not carry, each with its reason. MUST: a
#: key that is in neither this table, {@link GEMMA4_SAMPLER_FIELDS} nor
#: {@link GEMMA4_SUPPRESS_TOKENS_KEY} fails loudly — a generation default upstream adds (as 12B
#: added `suppress_tokens`) must surface as a decision, not be dropped silently.
GEMMA4_GENERATION_NOT_CARRIED: Mapping[str, str] = {
    "bos_token_id": "the compiled tokenizer asset declares `<bos>` (`bosId`); the host prepends it",
    "eos_token_id": (
        "the stop set is derived from the tokenizer asset's added vocabulary; the chat fixture"
        " (`gemma4.chat.stop_tokens`) holds the derivation to this declaration"
    ),
    "pad_token_id": "nothing is padded (one unpadded sequence per run)",
    "do_sample": (
        "greedy or sampled decoding is the caller's choice per request; sampling uses the"
        " declared sampler"
    ),
    "transformers_version": "the version of the library that wrote the file, not a default",
}

#: compile 済みトークナイザ資産の形式識別子（書き手は `_shared/gemma_tokenizer.py`）。
GEMMA4_TOKENIZER_FORMAT = "karume-gemma-tokenizer/1"

#: PLE 索引の版と欄（読み手 `packages/models/src/gemma/ple-index.ts` の
#: `SCHEMA` / `INDEX_KEYS` / `TABLE_KEYS` / `BLOCK_KEYS` の鏡像 — 焼く側が先に落とす）。
#: 版そのもの（`PLE_INDEX_SCHEMA` / `PLE_GROUPED_INDEX_SCHEMA`）は書き手（`karume.ple`）と共有する。
GEMMA4_PLE_VALUES_KEY = "values"
GEMMA4_PLE_SCALES_KEY = "scales"
GEMMA4_PLE_INDEX_KEYS: tuple[str, ...] = (
    "schema",
    "storage",
    "tokens",
    "layers",
    "dim",
    "embedScale",
    GEMMA4_PLE_VALUES_KEY,
    GEMMA4_PLE_SCALES_KEY,
)
GEMMA4_PLE_TABLE_KEYS: tuple[str, ...] = ("rowBytes", "blocks")
GEMMA4_PLE_BLOCK_KEYS: tuple[str, ...] = ("asset", "start", "stop")
#: Schema 4's one more key (`karume.ple.PLE_GROUPED_INDEX_SCHEMA` — the reader's `GROUPED_SCHEMA`):
#: the number of consecutive values of a (token, layer) row that share one f32 scale.
GEMMA4_PLE_GROUP_SIZE_KEY = "groupSize"


def gemma4_checkpoint(model: str) -> str:
    """モデル名 → 上流チェックポイントのディレクトリ名（= 上流の HF リポ名の末尾）。

    綴りの事実は「このモデルがどの上流リポか」1 つしかないので、帰属表
    （`gemma4.card.GEMMA4_UPSTREAM`）から導いてここに 2 つ目の表を持たない（SigLIP2 と同じ規律）。
    """
    repo = GEMMA4_UPSTREAM.get(model)
    if repo is None:
        raise DistError(
            f"gemma4 のモデル '{model}' は知らない（既知: {' / '.join(sorted(GEMMA4_UPSTREAM))}）"
        )
    return repo.split("/", 1)[1]


def gemma4_repo_name(_model: str) -> str:
    """配布リポ名（家族 1 リポなので、どのモデルでも同じ 1 つ — `karume-` prefix はリポ名裁定
    2026-08-09）。

    `Pipeline.repo_name` はカードの Usage 例に載る repo 名の正本（`karume.dist.resolve_repo`）で、
    単一モデルを組んだときの既定の出力先も答える席。家族が 1 リポに畳まれた今はモデル名を
    見ない。
    """
    return GEMMA4_REPO_NAME


def gemma4_series_name(model: str, suffix: str) -> str:
    """系列名（`outputs/series/gemma4-<モデル名>-<接尾>/`）— 書き手の既定と同じ綴り。"""
    return f"{GEMMA4_PREFIX}-{model}-{suffix}"


@dataclass(frozen=True)
class Gemma4Sources:
    """The inputs of one model's assembly: its series (the product graph with the PLE as container
    assets, the drafter, the tokenizer asset) and the upstream checkpoint, of which only
    `config.json`, `generation_config.json` (the recommended sampler) and `tokenizer.json` (what the
    tokenizer asset is held against — {@link assert_gemma4_tokenizer_source}) are read, never the
    weights.

    `drafter` が `None` のモデルは drafter を配らない（{@link gemma4_has_drafter}）。

    The upstream side is two directories: `model` holds the weights' `config.json` (the numbers the
    model decides — {@link gemma4_text_config}) and `upstream_tokenizer` the `tokenizer.json` and
    `generation_config.json`. They are one checkpoint in the regular line. In the q4_0 QAT line the
    weights come from Google's QAT checkpoints, whose `tokenizer.json` is the one every Gemma 4
    repository shipped before Google's glitch-token fixes (the QAT repositories have not received
    them), so the tokenizer and the generation defaults are read from the regular `-it` repository
    (`gemma4.sources.QAT_TARGET_PINS`).

    `original` is the original-weights product series (the regular line's `bf16` preset —
    {@link GEMMA4_ORIGINAL_ROLE}); `None` for a line without that preset (the q4_0 QAT line).
    """

    product: Path
    drafter: Path | None
    tokenizer: Path
    model: Path
    upstream_tokenizer: Path
    original: Path | None = None


def gemma4_sources(series_dir: Path, model: str = GEMMA4_DEFAULT_MODEL) -> Gemma4Sources:
    """系列の親ディレクトリ（`outputs/series/`）と `_shared.paths` の綴りから入力を引く。

    The upstream directories come from `gemma4.sources.checkpoint_dir` / `tokenizer_dir`
    (`inputs/gemma4/…`). A model outside the regular line ({@link GEMMA4_REGULAR_MODELS} — 12B
    ships in the q4_0 QAT line, {@link gemma4_qat_q4_0_sources}) fails loudly.
    """
    gemma4_checkpoint(model)
    if model not in GEMMA4_REGULAR_MODELS:
        raise DistError(
            f"the regular gemma4 line has no model '{model}'"
            f" (known: {' / '.join(GEMMA4_REGULAR_MODELS)}; 12B ships in --pipeline gemma4-qat)"
        )
    return Gemma4Sources(
        product=series_dir / gemma4_series_name(model, GEMMA4_PRODUCT_SUFFIX),
        drafter=(
            series_dir / gemma4_series_name(model, GEMMA4_DRAFTER_SUFFIX)
            if gemma4_has_drafter(model)
            else None
        ),
        tokenizer=series_dir
        / gemma4_series_name(GEMMA4_TOKENIZER_SERIES[model], GEMMA4_TOKENIZER_SUFFIX),
        model=checkpoint_dir(model),
        upstream_tokenizer=tokenizer_dir(model),
        original=series_dir / gemma4_series_name(model, GEMMA4_PRODUCT_BF16_SUFFIX),
    )


def _read_json(path: Path, what: str) -> Any:
    if not path.is_file():
        raise DistError(f"{what}が無い: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise DistError(f"{path} が JSON として読めない") from error


def _positive_int(raw: Mapping[str, Any], key: str, where: str) -> int:
    value = raw.get(key)
    # bool は int の派生。`"tokens": true` を 1 として通すと行数の突合が緩む。
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise DistError(f"{where} の {key} が正の整数でない（{value!r}）")
    return value


def _offset(raw: Mapping[str, Any], key: str, where: str) -> int:
    value = raw.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise DistError(f"{where} の {key} が 0 以上の整数でない（{value!r}）")
    return value


def gemma4_ple_table(
    root: Mapping[str, Any], name: str, row_bytes: int, tokens: int, where: str
) -> dict[str, Any]:
    """索引の表 1 本（`values` / `scales`）を検査して読む。

    MUST: block の範囲は `[0, tokens)` の**隙間も重なりも無い昇順分割**であること。緩めると
    「引けない id がある索引」や「2 本が同じ id を持つ索引」が通り、後者は**どちらの行を
    引いたか**で結果が変わる（例外の出ない沈黙誤値）。
    MUST: `rowBytes` は宣言（層数 / 次元 / 格納）から決まる値と一致すること。ここがずれると
    行 offset の掛け算だけが静かにずれ、形も dtype も合ったまま別 token の行を引く。
    """
    at = f"{where}.{name}"
    table = root.get(name)
    if not isinstance(table, dict):
        raise DistError(f"{at}: オブジェクトでない")
    extra = sorted(set(table) - set(GEMMA4_PLE_TABLE_KEYS))
    if extra:
        raise DistError(f"{at}: 未知キー {extra}（許可: {list(GEMMA4_PLE_TABLE_KEYS)}）")
    declared = _positive_int(table, "rowBytes", at)
    if declared != row_bytes:
        raise DistError(f"{at}.rowBytes {declared} が宣言から決まる {row_bytes} と違う")
    blocks = table.get("blocks")
    if not isinstance(blocks, list) or not blocks:
        raise DistError(f"{at}.blocks が非空の配列でない")
    seen: set[str] = set()
    expected = 0
    for position, entry in enumerate(blocks):
        block_at = f"{at}.blocks[{position}]"
        if not isinstance(entry, dict):
            raise DistError(f"{block_at}: オブジェクトでない")
        unknown = sorted(set(entry) - set(GEMMA4_PLE_BLOCK_KEYS))
        if unknown:
            raise DistError(
                f"{block_at}: 未知キー {unknown}（許可: {list(GEMMA4_PLE_BLOCK_KEYS)}）"
            )
        asset = entry.get("asset")
        if not isinstance(asset, str) or not asset:
            raise DistError(f"{block_at}: asset が非空の文字列でない（{asset!r}）")
        if asset in seen:
            raise DistError(f"{block_at}: asset '{asset}' が重複している")
        seen.add(asset)
        start = _offset(entry, "start", block_at)
        stop = _offset(entry, "stop", block_at)
        if start != expected:
            raise DistError(
                f"{block_at}: start {start} が直前の block の末尾 {expected} と連続しない"
            )
        if stop <= start:
            raise DistError(f"{block_at}: 範囲 [{start}, {stop}) が空")
        expected = stop
    if expected != tokens:
        raise DistError(f"{at}: block の合計 {expected} 行が tokens {tokens} と違う")
    return {"rowBytes": row_bytes, "blocks": blocks}


def gemma4_ple_index(
    container: Path, *, storage: str = "i8", group_size: int | None = None
) -> Mapping[str, Any]:
    """`model` 容器の資産 `ple_index` を読んで**形まで**落とす。

    読み手（`packages/models/src/gemma/ple-index.ts` の `parseGemma4PleIndex`）の受理集合の
    鏡像（旧 sidecar の索引は配布形ごと退役した — ADR 0109 決定 1 の major 繰り上げ規則）。
    `storage` は呼び手が要求する格納（QAT は `i2` / `i4`）。`group_size` is the form the caller
    requires: `None` = schema 3 (one f32 scale per (token, layer) row), a group size = schema 4
    with exactly that `groupSize` (one f32 scale per group — the q4_0 QAT line's 32). The reader
    takes both; a distribution takes the one its line writes, so a series of the other form fails
    here.

    A float `storage` (`bf16` — `karume.ple.PLE_FLOAT_BYTES`) requires schema 5: a `values` table
    of the floats themselves and no `scales` (neither the key nor `groupSize`); the result then
    has no `scales` entry either.

    MUST: 配ってから利用者の手元で初めて落ちる形にしない — 同じ検査を組み立て側でも掛ける。
    """
    where = f"{container} の資産 '{PLE_INDEX_ASSET}'"
    float_bytes = PLE_FLOAT_BYTES.get(storage)
    if storage not in PLE_PACK_FACTOR and float_bytes is None:
        raise DistError(f"未対応 PLE storage: {storage}")
    if float_bytes is not None and group_size is not None:
        raise DistError(f"PLE storage '{storage}' has no scales, so no group size ({group_size})")
    if float_bytes is not None:
        schema = PLE_FLOAT_INDEX_SCHEMA
        allowed: tuple[str, ...] = tuple(
            key for key in GEMMA4_PLE_INDEX_KEYS if key != GEMMA4_PLE_SCALES_KEY
        )
    elif group_size is None:
        schema, allowed = PLE_INDEX_SCHEMA, GEMMA4_PLE_INDEX_KEYS
    else:
        schema = PLE_GROUPED_INDEX_SCHEMA
        allowed = (*GEMMA4_PLE_INDEX_KEYS, GEMMA4_PLE_GROUP_SIZE_KEY)
    try:
        payload = read_asset(container, PLE_INDEX_ASSET)
    except (KeyError, ContainerError, ContainerFormatError) as cause:
        raise DistError(f"{where} が読めない: {cause}") from cause
    try:
        raw = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DistError(f"{where} が JSON として読めない") from error
    if not isinstance(raw, dict):
        raise DistError(f"{where}: 最上位オブジェクトでない")
    unknown = sorted(set(raw) - set(allowed))
    if unknown:
        raise DistError(f"{where}: 未知キー {unknown}（許可: {list(allowed)}）")
    if raw.get("schema") != schema:
        raise DistError(
            f"{where}: schema が {raw.get('schema')!r}（期待 {schema}）"
            " — 旧 sidecar の索引と、この配布が書かない版は読まない"
        )
    if raw.get("storage") != storage:
        raise DistError(f"{where}: storage が {raw.get('storage')!r}（期待 {storage}）")
    tokens = _positive_int(raw, "tokens", where)
    layers = _positive_int(raw, "layers", where)
    dim = _positive_int(raw, "dim", where)
    scale = raw.get("embedScale")
    if (
        not isinstance(scale, int | float)
        or isinstance(scale, bool)
        or not math.isfinite(scale)
        or scale <= 0
    ):
        raise DistError(f"{where}: embedScale が正の有限数でない（{scale!r}）")
    if float_bytes is not None:
        return {
            "storage": storage,
            "tokens": tokens,
            "layers": layers,
            "dim": dim,
            "embedScale": scale,
            GEMMA4_PLE_VALUES_KEY: gemma4_ple_table(
                raw, GEMMA4_PLE_VALUES_KEY, layers * dim * float_bytes, tokens, where
            ),
        }
    factor = PLE_PACK_FACTOR[storage]
    if dim % factor:
        raise DistError(f"{where}: dim {dim} が格納 '{storage}' の詰め数 {factor} で割り切れない")
    # The scales per (token, layer) row: one (schema 3), or one per group (schema 4).
    groups = 1
    if group_size is not None:
        declared = _positive_int(raw, GEMMA4_PLE_GROUP_SIZE_KEY, where)
        if declared != group_size:
            raise DistError(
                f"{where}: {GEMMA4_PLE_GROUP_SIZE_KEY} is {declared}, expected {group_size}"
                " (the grid the line stores)"
            )
        if dim % group_size:
            raise DistError(
                f"{where}: {GEMMA4_PLE_GROUP_SIZE_KEY} {group_size} does not divide dim {dim}"
            )
        groups = dim // group_size
    return {
        "storage": storage,
        "tokens": tokens,
        "layers": layers,
        "dim": dim,
        "embedScale": scale,
        **({} if group_size is None else {GEMMA4_PLE_GROUP_SIZE_KEY: group_size}),
        GEMMA4_PLE_VALUES_KEY: gemma4_ple_table(
            raw, GEMMA4_PLE_VALUES_KEY, layers * dim // factor, tokens, where
        ),
        GEMMA4_PLE_SCALES_KEY: gemma4_ple_table(
            raw, GEMMA4_PLE_SCALES_KEY, layers * groups * PLE_SCALE_BYTES, tokens, where
        ),
    }


def gemma4_placements(sources: Gemma4Sources) -> dict[str, Path]:
    """役割名 → 出所のファイル。出力の path は {@link GEMMA4_OUTPUT_PATHS} が持つ。

    この表に無いものは出力へ入らない（製品系列に同居する `ple.probe.safetensors` と
    `reference.json`・drafter 系列の `drafter-golden.*.safetensors` はこれで落ちる — どれも
    検収と出所記録のためのもので実行に要らない）。PLE は `model` 容器の中なので席を持たない。
    drafter を配らないモデル（`sources.drafter is None`）は drafter の席も持たない。
    """
    placements = {GEMMA4_ROLE: sources.product / GEMMA4_MODEL_FILE}
    if sources.original is not None:
        placements[GEMMA4_ORIGINAL_ROLE] = sources.original / GEMMA4_MODEL_FILE
    if sources.drafter is not None:
        placements[GEMMA4_DRAFTER_ROLE] = sources.drafter / GEMMA4_MODEL_FILE
    placements[GEMMA4_TOKENIZER_ROLE] = sources.tokenizer / GEMMA4_TOKENIZER_FILE
    return placements


#: assets の宣言（asset 名 → 役割名）。PLE は容器の中へ移ったので、残るのはトークナイザ 1 本。
GEMMA4_ASSETS: Mapping[str, str] = {GEMMA4_TOKENIZER_ROLE: GEMMA4_TOKENIZER_ROLE}


def gemma4_vocab_size(graph: Mapping[str, Any], path: Path) -> int:
    """選択行 logits 出口の語彙数をグラフの出力宣言から読む（`[1, R, V]` — ADR 0083 決定 6）。

    出力が **2 本**（`logits[1, R, V]` → `hidden[1, R, H]` の順）であることまで見るのは、
    検収用の 2 系列（logits opt-in / token-only）が同じ系列名の下に紛れ込むと**幅だけが別の
    意味の数**になるため。行軸が記号 `R` なのは投機 verify が 1 回で複数行を採点するからで、
    通常の decode はそこを 1 に束縛する。V は主 embedding の行数そのもので、PLE（容器の資産）と
    トークナイザの相互照合（ADR 0085 決定 5）の基準になる。

    MUST: 順序まで見る（logits と hidden は行軸まで同型で、幅だけが違う）— 入れ替わった資産を
    受けると、V のつもりで hidden_size を読んだまま manifest が組み上がる。
    """
    outputs = graph.get("outputs")
    if not isinstance(outputs, list) or len(outputs) != 2:
        raise DistError(
            f"{path}: グラフ出力が {outputs!r} — 製品グラフの出口は選択行の logits + hidden の 2 本"
        )
    values = graph.get("values")
    widths: list[int] = []
    for slot, what in enumerate(("logits", "hidden")):
        value = values.get(outputs[slot]) if isinstance(values, dict) else None
        shape = value.get("shape") if isinstance(value, dict) else None
        if (
            not isinstance(shape, list)
            or len(shape) != 3
            or shape[0] != 1
            or shape[1] != GEMMA4_ROW_SYMBOL
        ):
            raise DistError(
                f"{path}: グラフ出力 '{outputs[slot]}'（{what}）の形が"
                f" [1, {GEMMA4_ROW_SYMBOL}, *] でない（{shape!r}）"
            )
        width = shape[2]
        if not isinstance(width, int) or isinstance(width, bool) or width <= 0:
            raise DistError(
                f"{path}: グラフ出力 '{outputs[slot]}' の幅が正の整数でない（{width!r}）"
            )
        widths.append(width)
    return widths[0]


def gemma4_text_config(model_dir: Path) -> SimpleNamespace:
    """上流 `config.json` の `text_config` 節（位置の上限と RoPE の式の出どころ）。

    属性アクセスの形へ寄せるのは {@link gemma4.rope.rope_specs} が transformers の config
    オブジェクトと同じ読み方をするため — 焼く側と配る側で**同じ導出コード**を通す。
    """
    where = str(model_dir / GEMMA4_CONFIG_FILE)
    raw = _read_json(model_dir / GEMMA4_CONFIG_FILE, "上流のモデル設定")
    if not isinstance(raw, dict):
        raise DistError(f"{where}: 最上位オブジェクトでない")
    text = raw.get(GEMMA4_TEXT_CONFIG_KEY)
    if not isinstance(text, dict):
        raise DistError(f"{where}: {GEMMA4_TEXT_CONFIG_KEY} がオブジェクトでない（{text!r}）")
    return SimpleNamespace(**text)


def gemma4_max_position(text_config: SimpleNamespace, where: str) -> int:
    """モデルが宣言する位置の上限（= `pipelineConfig.maxPosition`）。

    MUST: 写経しない（上流 `text_config.max_position_embeddings` が唯一の出どころ）。RoPE を
    ホスト生成へ移した以上、位置の上限を持っているのは資産ではなくモデルの宣言だけになる。
    """
    declared = getattr(text_config, GEMMA4_MAX_POSITION_KEY, None)
    if isinstance(declared, bool) or not isinstance(declared, int) or declared < 1:
        raise DistError(f"{where}: {GEMMA4_MAX_POSITION_KEY} が正の整数でない（{declared!r}）")
    return declared


def gemma4_hidden_size(text_config: SimpleNamespace, where: str) -> int:
    """モデルが宣言する hidden 幅（製品グラフの出力 1 = 選択行 hidden の幅）。

    MUST: 写経しない（上流 `text_config.hidden_size` が唯一の出どころ）。宣言とグラフは別々の
    正本から来るので、噛み合わせは {@link assert_gemma4_graph} でしか見られない — 出力 2 本は
    行軸まで同型なので、入れ替えを落とせるのは**幅を宣言と突き合わせる**この 1 点だけ。
    """
    declared = getattr(text_config, GEMMA4_HIDDEN_SIZE_KEY, None)
    if isinstance(declared, bool) or not isinstance(declared, int) or declared < 1:
        raise DistError(f"{where}: {GEMMA4_HIDDEN_SIZE_KEY} が正の整数でない（{declared!r}）")
    return declared


#: The upstream `text_config` field that sizes one layer's per-layer embedding (0 = no PLE).
GEMMA4_PLE_DIM_KEY = "hidden_size_per_layer_input"


def gemma4_ple_dim(text_config: Any, where: str) -> int:
    """The width of one layer's per-layer embedding the model declares; 0 = the model has none
    (12B's config says 0 — its class, `gemma4_unified`, has no PLE branch).

    Shared by the distribution and the export scripts (`text_config` is a namespace here and a
    transformers config there — both read attributes).

    MUST: read, never defaulted — a config without the field cannot say whether the graph should
    take `per_layer_inputs`, and guessing either way lets a graph / asset mix through.
    """
    declared = getattr(text_config, GEMMA4_PLE_DIM_KEY, None)
    if isinstance(declared, bool) or not isinstance(declared, int) or declared < 0:
        raise DistError(
            f"{where}: {GEMMA4_PLE_DIM_KEY} is not a non-negative integer ({declared!r})"
        )
    return declared


def gemma4_rope(text_config: SimpleNamespace, where: str) -> dict[str, Any]:
    """`pipelineConfig.rope`（層種別ごとの theta / headDim / rotaryDim）を config から導く。

    MUST: 受理外の rope_type / 係数は fail loudly（{@link gemma4.rope.layer_spec}）— 式が
    別物なのに theta と幅だけを写すと、ホストが**形も型も合う別の角度**で表を組む。
    MUST: 層種別は sliding / full の 2 つちょうど。増減はグラフ入力の本数
    （{@link GEMMA4_ROPE_INPUTS}）と噛み合わなくなる。
    """
    try:
        specs = rope_specs(text_config)
    except RopeSpecError as error:
        raise DistError(f"{where}: RoPE の宣言を導けない — {error}") from error
    if sorted(specs) != sorted(GEMMA4_ROPE_LAYER_TYPES):
        raise DistError(
            f"{where}: 層種別が {sorted(specs)} — {sorted(GEMMA4_ROPE_LAYER_TYPES)} でない"
        )
    return {layer_type: specs[layer_type].declaration() for layer_type in GEMMA4_ROPE_LAYER_TYPES}


def assert_gemma4_graph(
    graph: Mapping[str, Any],
    path: Path,
    index: Mapping[str, Any] | None,
    rope: Mapping[str, Any],
    hidden_size: int,
) -> None:
    """グラフ入力の並び・形・記号の割れ方を、配置の前に実測する。

    `index` is the container's PLE index, or `None` for a model without per-layer embeddings
    (12B): its graph takes {@link GEMMA4_PLE_FREE_GRAPH_INPUTS} — `per_layer_inputs` absent — and
    every other rule below holds unchanged.

    MUST: PLE の層数と層当たり次元は**グラフ入力の宣言**と**索引**の両方が持つ（前者は
    `per_layer_inputs[1, M, 35, 256]`・後者は `layers` / `dim`）。食い違ったまま配ると、
    ホストが組む表と GPU が読む形が別物になる — shape が合う組み合わせでは沈黙する
    （索引は同じ容器の資産だが、書き手が別々に組むので噛み合わせは残る）。

    MUST: RoPE 派生入力の幅は `pipelineConfig.rope` の `headDim` と一致すること。宣言と
    グラフは別々の正本（config / コンテナ）から来るので、噛み合わせはここでしか見られない
    — ホストが宣言どおりに組んだ表が入力の幅と違えば、実行時まで誰も気づけない。

    MUST: 出力 1（hidden）の幅は上流の `hidden_size` と一致すること。出力 2 本は行軸まで
    同型（どちらも f32 `[1, R, *]`）なので、**入れ替わりを落とせるのはこの幅の突合だけ**
    — 取り違えたまま通すと、語彙数のつもりで hidden_size を読んだ manifest が組み上がる。

    MUST: 表の initializer が 1 本も残っていないこと。派生入力を足したのに表も残っている形は
    常駐が戻るうえ、位置の上限が資産側へ逆戻りする。

    MUST: 呼び手は {@link gemma4_vocab_size} を**先に**通していること（出力が 2 本で、どちらも
    `[1, R, 幅]` の 3 軸であることを前提に幅だけを読む）。

    MUST: 記号は 3 本（chunk 行数 `M` / 出口の行数 `R` / full スロットの容量）で、**入力 shape
    から決まらないもの**がちょうど 1 本（容量記号）。TS 側 `gemma.Pipeline` はこの 1 本を容量の
    束縛点にするので、割れ方が変わるとロード時に落ちる（`capacitySymbolOf` の同じ検査）。
    `R` は `last_row[R]` が束縛するので、この勘定では自由記号に数えない。
    """
    inputs = graph_inputs(graph, path)
    names = tuple(inputs)
    expected_names = GEMMA4_GRAPH_INPUTS if index is not None else GEMMA4_PLE_FREE_GRAPH_INPUTS
    if names != expected_names:
        raise DistError(
            f"{path} のグラフ入力が {list(names)} で、期待の {list(expected_names)} と違う"
            " — 実行側は名前で束ねるので、1 つでも綴りが変われば束ねられない"
        )
    sequence = inputs[GEMMA4_INPUT_IDS][1]
    for layer_type in GEMMA4_ROPE_LAYER_TYPES:
        head_dim = rope[layer_type][HEAD_DIM_FIELD]
        for part in GEMMA4_ROPE_PARTS:
            name = gemma4_rope_input_name(layer_type, part)
            declared = inputs[name]
            if list(declared) != [1, sequence, head_dim]:
                raise DistError(
                    f"{path} の入力 '{name}' が {list(declared)!r} — 宣言した headDim から組んだ"
                    f" 期待 {[1, sequence, head_dim]} と違う（表とグラフが別世代）"
                )
    # IR v2 では initializer 名がテンソルキーそのもの（docs/ir-v2.md）— 綴りで拾う。
    baked = sorted(key for key in (graph.get("initializers") or {}) if BAKED_TABLE_INFIX in key)
    if baked:
        raise DistError(
            f"{path}: 焼き込んだ RoPE 表の initializer が {len(baked)} 本残っている: {baked[:4]}"
            " — ホスト生成へ外に出し切れていない世代の資産"
        )
    if index is not None:
        per_layer = inputs[GEMMA4_PER_LAYER_INPUTS]
        if len(per_layer) != 4:
            raise DistError(
                f"{path} の入力 '{GEMMA4_PER_LAYER_INPUTS}' が {per_layer!r} — [1, M, 層数, 次元]"
                "の 4 軸でない"
            )
        for axis, field in ((2, "layers"), (3, "dim")):
            if per_layer[axis] != index[field]:
                raise DistError(
                    f"{path} の入力 '{GEMMA4_PER_LAYER_INPUTS}' の軸 {axis} が"
                    f" {per_layer[axis]!r}、"
                    f"資産 '{PLE_INDEX_ASSET}' の {field} は {index[field]}"
                    " — グラフと PLE が別世代"
                )
    outputs = graph.get("outputs")
    values = graph.get("values")
    hidden = values.get(outputs[1]) if isinstance(values, dict) else None
    found = hidden.get("shape")[2] if isinstance(hidden, dict) else None
    if found != hidden_size:
        raise DistError(
            f"{path} の出力 1（hidden）の幅が {found!r} — 上流の"
            f" {GEMMA4_HIDDEN_SIZE_KEY} {hidden_size} と違う（出力 2 本が入れ替わっている"
            "／グラフと宣言が別世代）"
        )
    symbols = graph.get("symbols")
    if not isinstance(symbols, list):
        raise DistError(f"{path}: IR メタデータに symbols が無い")
    bound = {dim for shape in inputs.values() for dim in shape if isinstance(dim, str)}
    free = [symbol for symbol in symbols if symbol not in bound]
    if len(free) != 1:
        raise DistError(
            f"{path}: 入力 shape から決まらない記号が {len(free)} 本（{free}）"
            " — full スロットの容量記号 1 本であること"
        )


def assert_gemma4_drafter_graph(
    graph: Mapping[str, Any],
    path: Path,
    lender: Mapping[str, Any],
    lender_path: Path,
    rope: Mapping[str, Any],
    hidden_size: int,
) -> None:
    """drafter グラフ（借り手）の形と、**貸し手との噛み合わせ**を配置の前に実測する。

    借り手は単独では実行できない資産なので、見るべき性質の半分は「貸し手と一致していること」に
    なる。食い違ったまま配ると、ロード時に落ちるか（名前が違う）、**形も型も合ったまま別の列を
    読む**（容量が違う）。

    - グラフ入力が `token` / `hidden` / RoPE 4 本ちょうど（K/V の placeholder が残っていれば、
      呼び手が渡す値をどのノードも読まない形になる）
    - `hidden` の幅が上流の `hidden_size`（= 製品グラフの出力 1 の幅）と一致
    - RoPE 派生入力の幅が `pipelineConfig.rope` の `headDim` と一致（貸し手と同じ突合）
    - 出力が k 本ちょうど（配布形は {@link GEMMA4_DRAFT_STEPS} 本）
    - `states` が**全部 external**で、名前も形も**貸し手の宣言と 1 対 1**
    - initializer に**共有宣言が 1 本**で、指し先が貸し手コンテナに実在するテンソルキー
    - 記号は容量記号 1 本だけで、**貸し手と同じ綴り**（借り手の bindings は貸し手を継承する）
    """
    inputs = graph_inputs(graph, path)
    names = tuple(inputs)
    if names != GEMMA4_DRAFTER_GRAPH_INPUTS:
        raise DistError(
            f"{path} のグラフ入力が {list(names)} で、期待の"
            f" {list(GEMMA4_DRAFTER_GRAPH_INPUTS)} と違う"
            " — K/V の placeholder が手術で落ちていない世代の資産"
        )
    hidden = inputs[GEMMA4_DRAFTER_HIDDEN]
    if list(hidden) != [1, hidden_size]:
        raise DistError(
            f"{path} の入力 '{GEMMA4_DRAFTER_HIDDEN}' が {list(hidden)!r} —"
            f" [1, {hidden_size}]（target の最終 norm 後 hidden 1 行）でない"
        )
    for layer_type in GEMMA4_ROPE_LAYER_TYPES:
        head_dim = rope[layer_type][HEAD_DIM_FIELD]
        for part in GEMMA4_ROPE_PARTS:
            name = gemma4_rope_input_name(layer_type, part)
            declared = inputs[name]
            if list(declared) != [1, 1, head_dim]:
                raise DistError(
                    f"{path} の入力 '{name}' が {list(declared)!r} — 宣言した headDim から組んだ"
                    f" 期待 {[1, 1, head_dim]} と違う（表とグラフが別世代）"
                )
    outputs = graph.get("outputs")
    if not isinstance(outputs, list) or len(outputs) != GEMMA4_DRAFT_STEPS:
        raise DistError(f"{path} の IR 出力が {outputs!r} — draft {GEMMA4_DRAFT_STEPS} 本でない")

    borrowed = graph.get("states")
    lent = lender.get("states")
    if not isinstance(borrowed, dict) or not isinstance(lent, dict):
        raise DistError(f"{path}: IR メタデータに states が無い（借り手 / 貸し手のどちらか）")
    for name, slot in sorted(borrowed.items()):
        if not slot.get("external"):
            raise DistError(
                f"{path}: state スロット '{name}' が external でない"
                "（借り手は実体を持たない — 全スロットが external MUST）"
            )
        owner = lent.get(name)
        if owner is None:
            raise DistError(
                f"{path}: state スロット '{name}' が貸し手 {lender_path} に無い"
                f"（貸し手の宣言: {sorted(lent)}）"
            )
        if owner.get("dtype") != slot.get("dtype") or owner.get("shape") != slot.get("shape"):
            raise DistError(
                f"{path}: state スロット '{name}' が {slot.get('dtype')} {slot.get('shape')} —"
                f" 貸し手の {owner.get('dtype')} {owner.get('shape')} と違う（別世代の組）"
            )

    initializers = graph.get("initializers") or {}
    # IR v2 では initializer 名がテンソルキーそのもの（docs/ir-v2.md）なので、共有宣言の
    # **名前が指し先**になる（v1 の `shared.tensor` の席はもう無い）。
    shared = sorted(
        name
        for name, entry in initializers.items()
        if isinstance(entry, dict) and entry.get("shared") is True
    )
    if len(shared) != 1:
        raise DistError(f"{path}: 共有 initializer が {len(shared)} 本 {shared}（1 本ちょうど）")
    tensor = shared[0]
    lent_tensors = set(lender.get("initializers") or {})
    if tensor not in lent_tensors:
        raise DistError(
            f"{path}: 共有 initializer の指し先 '{tensor}' が貸し手 {lender_path} に無い"
            "（貸し手を焼き直した世代と噛み合っていない）"
        )

    symbols = graph.get("symbols")
    lent_symbols = lender.get("symbols")
    if not isinstance(symbols, list) or len(symbols) != 1:
        raise DistError(
            f"{path}: symbols が {symbols!r} — 容量記号 1 本だけであること"
            "（借り手の入力はどの記号も束縛しない）"
        )
    if not isinstance(lent_symbols, list) or symbols[0] not in lent_symbols:
        raise DistError(
            f"{path}: 容量記号 '{symbols[0]}' が貸し手 {lender_path} の symbols"
            f" {lent_symbols!r} に無い — 借り手の束縛は貸し手から継承する"
        )


def assert_gemma4_ple_assets(container: Path, index: Mapping[str, Any]) -> None:
    """索引が名指しする block が、その容器の資産として**同じ役割・同じ長さ**で在ることを見る。

    MUST: 索引と容器の宣言は別々に動きうる（索引は書き手が組んだ JSON・資産宣言は容器の
    モデル記述）ので、噛み合わせはここでしか見られない — 索引だけ差し替えた組み合わせは
    **形も dtype も合う**まま別 token の行を引く（読み手 `ple-index.ts` の
    `assertGemma4PleAssets` と同じ規律）。宣言しか読まないので数 GB の再読みにはならない。
    """
    try:
        declared = read_asset_declarations(container)
    except (KeyError, ContainerError, ContainerFormatError) as cause:
        raise DistError(f"{container}: 資産の宣言が読めない: {cause}") from cause
    seen = {PLE_INDEX_ASSET}
    if declared.get(PLE_INDEX_ASSET, (None, 0))[0] != PLE_INDEX_ROLE:
        raise DistError(
            f"{container}: 資産 '{PLE_INDEX_ASSET}' の役割が"
            f" {declared.get(PLE_INDEX_ASSET, (None, 0))[0]!r}（期待 '{PLE_INDEX_ROLE}'）"
        )
    # A float index (schema 5) has no `scales` table; a `ple-scales` asset in its container is
    # then surplus below.
    for key, role in PLE_ROLES.items():
        if key not in index:
            continue
        table = index[key]
        for block in table["blocks"]:
            name = str(block["asset"])
            seen.add(name)
            found = declared.get(name)
            if found is None:
                raise DistError(
                    f"{container}: 索引が名指しする資産 '{name}' が容器に無い"
                    f"（宣言: {sorted(declared)[:4]}…）"
                )
            expected = (int(block["stop"]) - int(block["start"])) * table["rowBytes"]
            if found != (role, expected):
                raise DistError(
                    f"{container}: 資産 '{name}' が {found}"
                    f"（索引から組んだ期待は ('{role}', {expected})）"
                )
    surplus = sorted(
        name
        for name, (role, _length) in declared.items()
        if name not in seen and role in {PLE_INDEX_ROLE, *PLE_ROLES.values()}
    )
    if surplus:
        raise DistError(
            f"{container}: 索引が指していない PLE の資産がある: {surplus} — 索引だけ古い組み合わせ"
        )


def assert_gemma4_no_ple_assets(container: Path) -> None:
    """A model without per-layer embeddings ships a container without PLE assets (the inverted
    form of {@link assert_gemma4_ple_assets}).

    MUST: fail loudly on any — a PLE index or block in a container whose graph takes no
    `per_layer_inputs` is another model's material (or a half-converted series), and the reader
    would either ignore it silently or, finding the index, build a PLE nobody feeds.
    """
    try:
        declared = read_asset_declarations(container)
    except (KeyError, ContainerError, ContainerFormatError) as cause:
        raise DistError(f"{container}: 資産の宣言が読めない: {cause}") from cause
    ple_roles = {PLE_INDEX_ROLE, *PLE_ROLES.values()}
    carried = sorted(name for name, (role, _length) in declared.items() if role in ple_roles)
    if carried:
        raise DistError(
            f"{container}: the model has no per-layer embeddings, yet the container carries PLE"
            f" assets {carried[:4]} — a graph / asset mix"
        )


def assert_gemma4_tokenizer(path: Path, vocab_size: int) -> None:
    """compile 済みトークナイザ資産が**この語彙で焼かれたもの**であることを見る。

    MUST: 行数まで突き合わせる（ADR 0085 決定 5 の相互照合を焼く側でも掛ける）— 別語彙の資産は
    id が範囲内に収まる限り**別 token の有効な行**を引き、例外なしで沈黙して壊れる。TS 側の
    admission も同じ検査を持つが、あちらは 4GiB を落とした後にしか走らない。
    """
    raw = _read_json(path, "compile 済みトークナイザ資産")
    if not isinstance(raw, dict):
        raise DistError(f"{path}: 最上位オブジェクトでない")
    if raw.get("format") != GEMMA4_TOKENIZER_FORMAT:
        raise DistError(
            f"{path}: format が {raw.get('format')!r}（期待 '{GEMMA4_TOKENIZER_FORMAT}'）"
            " — compile 台本（`python -m gemma4.tokenizer`）が書いた資産でない"
        )
    vocab = raw.get("vocab")
    if not isinstance(vocab, list):
        raise DistError(f"{path}: vocab が配列でない")
    if len(vocab) != vocab_size:
        raise DistError(
            f"{path}: vocab が {len(vocab)} 行で、製品グラフの語彙数 {vocab_size} と違う"
            " — 別の語彙で焼かれた組み合わせ"
        )


def assert_gemma4_tokenizer_source(path: Path, checkpoint_tokenizer: Path) -> None:
    """The compiled tokenizer asset was compiled from **the same bytes as this checkpoint's
    `tokenizer.json`** — the gate every model's asset passes ({@link gemma4_plan}).

    The compile script (`_shared.gemma_tokenizer.emit`) writes the source's SHA-256 into the asset's
    `source` field; comparing that field with the model's upstream file is the only way to see
    both failures it guards against: a series borrowed from another model whose upstream is no
    longer the same bytes ({@link GEMMA4_TOKENIZER_SERIES}), and a model's own series compiled
    before its upstream file was replaced (Google's glitch-token fixes of 2026-11-03 changed E2B's
    and E4B's `tokenizer.json`).

    MUST: 語彙数の一致（{@link assert_gemma4_tokenizer}）では足りない — 同じ語彙数で merges や
    特殊トークンだけが違うトークナイザは、id が範囲内に収まったまま**別の id 列**を作る。
    """
    raw = _read_json(path, "compile 済みトークナイザ資産")
    source = raw.get("source") if isinstance(raw, dict) else None
    recorded = source.get("sha256") if isinstance(source, dict) else None
    if not isinstance(recorded, str):
        raise DistError(
            f"{path}: no source.sha256 — an asset whose source cannot be checked is not distributed"
        )
    if not checkpoint_tokenizer.is_file():
        raise DistError(f"上流のトークナイザが無い: {checkpoint_tokenizer}")
    actual = sha256_file(checkpoint_tokenizer)
    if recorded != actual:
        raise DistError(
            f"{path} was compiled from a tokenizer.json with SHA-256 {recorded}, but"
            f" {checkpoint_tokenizer} is {actual} — the asset is stale or another model's"
            " (compile the model's tokenizer series again from this file:"
            " `python -m gemma4.tokenizer --model <model>`)"
        )


@dataclass(frozen=True)
class Gemma4GenerationDefaults:
    """What a distribution carries from upstream `generation_config.json`."""

    #: `pipelineConfig.sampler` (TS `SamplerSpec` field names).
    sampler: Mapping[str, Any]
    #: `pipelineConfig.suppressTokens` — unique ids, ascending; `None` when upstream declares none.
    suppress_tokens: tuple[int, ...] | None


def gemma4_generation_defaults(model_dir: Path) -> Gemma4GenerationDefaults:
    """Read upstream `generation_config.json`: the recommended sampler and the suppressed ids.

    MUST: every key is mapped ({@link GEMMA4_SAMPLER_FIELDS}, {@link GEMMA4_SUPPRESS_TOKENS_KEY})
    or listed as not carried with its reason ({@link GEMMA4_GENERATION_NOT_CARRIED}); any other
    key fails loudly. The suppressed ids must be a non-empty list of unique non-negative integers
    (the vocabulary bound is checked against the graph — {@link assert_gemma4_suppress_tokens}).
    """
    where = str(model_dir / GEMMA4_GENERATION_CONFIG_FILE)
    raw = _read_json(model_dir / GEMMA4_GENERATION_CONFIG_FILE, "上流の生成既定")
    if not isinstance(raw, dict):
        raise DistError(f"{where}: 最上位オブジェクトでない")
    known = {
        *(source for source, _ in GEMMA4_SAMPLER_FIELDS),
        GEMMA4_SUPPRESS_TOKENS_KEY,
        *GEMMA4_GENERATION_NOT_CARRIED,
    }
    unknown = sorted(set(raw) - known)
    if unknown:
        raise DistError(
            f"{where}: generation defaults {unknown} are neither carried nor listed as"
            " intentionally not carried (gemma4.distribution.GEMMA4_GENERATION_NOT_CARRIED)"
        )
    suppress = raw.get(GEMMA4_SUPPRESS_TOKENS_KEY)
    return Gemma4GenerationDefaults(
        sampler=_sampler(raw, where),
        suppress_tokens=None if suppress is None else _suppress_tokens(suppress, where),
    )


def _suppress_tokens(value: Any, where: str) -> tuple[int, ...]:
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(one, int) and not isinstance(one, bool) for one in value)
    ):
        raise DistError(
            f"{where}: {GEMMA4_SUPPRESS_TOKENS_KEY} is not a non-empty list of integers ({value!r})"
        )
    if len(set(value)) != len(value):
        raise DistError(f"{where}: {GEMMA4_SUPPRESS_TOKENS_KEY} repeats an id ({value!r})")
    if min(value) < 0:
        raise DistError(f"{where}: {GEMMA4_SUPPRESS_TOKENS_KEY} holds a negative id ({value!r})")
    return tuple(sorted(value))


def assert_gemma4_suppress_tokens(ids: tuple[int, ...], vocab_size: int, where: str) -> None:
    """Every suppressed id names a token of the graph's vocabulary (`[0, vocab)`)."""
    outside = [token for token in ids if token >= vocab_size]
    if outside:
        raise DistError(
            f"{where}: {GEMMA4_SUPPRESS_TOKENS_KEY} {outside} outside the graph's vocabulary"
            f" [0, {vocab_size})"
        )


def gemma4_sampler(model_dir: Path) -> dict[str, Any]:
    """The recommended sampler alone, for a pipeline whose `pipelineConfig` has no
    `suppressTokens` (the QAT family's).

    MUST: a checkpoint that declares suppressed ids fails loudly here — such a caller would drop
    them, and its generations would select ids upstream never selects.
    """
    defaults = gemma4_generation_defaults(model_dir)
    if defaults.suppress_tokens is not None:
        raise DistError(
            f"{model_dir / GEMMA4_GENERATION_CONFIG_FILE}: declares {GEMMA4_SUPPRESS_TOKENS_KEY}"
            f" {list(defaults.suppress_tokens)}, which this pipeline's configuration cannot carry"
        )
    return dict(defaults.sampler)


def _sampler(raw: Mapping[str, Any], where: str) -> dict[str, Any]:
    """上流 `generation_config.json` の推奨サンプラを `pipelineConfig.sampler` へ写す。

    MUST: 値を写経しない（ADR 0083 決定 7 — 「既定値は配布形が宣言する」の出どころは上流の
    宣言そのもの）。欄名だけは TS 側 `SamplerSpec` の綴りへ翻訳する（`top_k` → `topK`）。
    値域は TS 側 `gemma.parsePipelineConfig` と同じ — 配ってから parse で落ちる形にしない。
    """
    sampler: dict[str, Any] = {}
    for source, field in GEMMA4_SAMPLER_FIELDS:
        if source not in raw:
            raise DistError(f"{where}: {source} が無い — 推奨サンプラを宣言できない")
        value = raw[source]
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise DistError(f"{where}: {source} が数でない（{value!r}）")
        sampler[field] = value
    temperature = sampler["temperature"]
    if not math.isfinite(temperature) or temperature < 0:
        raise DistError(f"{where}: temperature が 0 以上の有限数でない（{temperature!r}）")
    top_k = sampler["topK"]
    if not isinstance(top_k, int) or top_k < 1:
        raise DistError(f"{where}: top_k が 1 以上の整数でない（{top_k!r}）")
    top_p = sampler["topP"]
    if not math.isfinite(top_p) or not 0 < top_p <= 1:
        raise DistError(f"{where}: top_p が (0, 1] の有限数でない（{top_p!r}）")
    sampler["temperature"] = float(temperature)
    sampler["topP"] = float(top_p)
    return sampler


def gemma4_pipeline_config(
    max_position: int,
    rope: Mapping[str, Any],
    sampler: Mapping[str, Any],
    *,
    chunk_length: int,
    max_chunk_length: int,
    capacity: int,
    chat_dialect: str = GEMMA4_CHAT_DIALECT,
    suppress_tokens: tuple[int, ...] | None = None,
) -> dict[str, Any]:
    """`pipelineConfig`（TS 側スキーマの 6 欄）を組む。

    `chat_dialect` is the optional seventh field `chatDialect` (`gemma4.chat.CHAT_DIALECTS` — a
    closed set). MUST: the default dialect is **not written** — an absent field reads as `gemma4`,
    so E2B / E4B (and the QAT family, which shares this function) keep their manifests byte for
    byte; only another dialect (12B's `gemma4-unified`) adds the key.

    `suppress_tokens` is the optional field `suppressTokens` (ids whose logits are −inf before any
    host token selection — {@link gemma4_generation_defaults}), written only when upstream
    declares them (12B), for the same byte-identity reason.

    実行時ノブの 3 つ（`chunk_length` / `max_chunk_length` / `capacity`）は**呼び手が渡す** —
    通常 Gemma は {@link GEMMA4_CHUNK_LENGTH} / {@link GEMMA4_MAX_CHUNK_LENGTH} /
    {@link GEMMA4_CAPACITY} を、固定 QAT は `gemma4_qat.config` の同種の定数を渡す。関係式の
    実装が 1 本しかないので、family が増えても「3 式のうち 1 式しか見ていない配布形」を作れない。

    MUST: 実行時ノブが**両側の上限の内側**に収まることをここで落とす。chunk の行数は記号 `M`
    の trace 時の上限（`max_chunk_length`）を超えられず、`capacity` は会話が使える最大の
    論理長なので位置は最大 `capacity - 1` まで進む — モデルの宣言（`maxPosition`）を超える
    容量は「宣言の内側なのに上流が想定していない位置を回す」形で、長い会話でだけ表面化する。

    MUST: その chunk の上限を `maxChunkLength` として**宣言にも載せる**。ここの検査が見るのは
    配布形が焼く既定値だけで、`chunkLength` は読み手の実行時ノブでもある — 宣言が無いと
    「trace 範囲の外の chunk 長で走る」形を読み手側で落とせない（IR の `symbols` は名前の列
    だけで上限を持たない）。

    NOTE: 上限の側は**写しの同値しか見ていない**。比較相手 {@link GEMMA4_MAX_CHUNK_LENGTH} は
    `gemma4.export.SYM_MAX` を写した定数で、系列を組んだときに実際に使われた `--sym-max` は
    どこにも記録されていない（資産からも読めない）。小さい `--sym-max` で trace した容器に対して
    `chunkLength: 768` を名乗る配布形は、この検査も同値テストも素通りする。
    """
    if not 2 <= chunk_length <= max_chunk_length:
        raise DistError(
            f"chunkLength {chunk_length} が [2, {max_chunk_length}] の外"
            "（下限は記号 M の下限・上限は trace 時の `Dim` の上限）"
        )
    if capacity < chunk_length:
        raise DistError(
            f"capacity {capacity} が chunkLength {chunk_length} より小さい"
            " — 1 chunk すら入らない容量は宣言できない"
        )
    if max_position < capacity:
        raise DistError(
            f"capacity {capacity} がモデルの位置上限 {max_position} を超えた"
            " — 容量いっぱいの会話が宣言の外の位置を回す"
        )
    if chat_dialect not in GENERATION_PROMPT_SUFFIX:
        raise DistError(
            f"chatDialect {chat_dialect!r} is not one of {sorted(GENERATION_PROMPT_SUFFIX)}"
        )
    config: dict[str, Any] = {
        "chunkLength": chunk_length,
        "maxChunkLength": max_chunk_length,
        "maxPosition": max_position,
        "capacity": capacity,
        "rope": {layer_type: dict(spec) for layer_type, spec in rope.items()},
        "sampler": dict(sampler),
    }
    if chat_dialect != GEMMA4_CHAT_DIALECT:
        config["chatDialect"] = chat_dialect
    if suppress_tokens is not None:
        if not suppress_tokens or list(suppress_tokens) != sorted(set(suppress_tokens)):
            raise DistError(
                f"suppressTokens {list(suppress_tokens)} is not a non-empty ascending list of"
                " unique ids"
            )
        config["suppressTokens"] = list(suppress_tokens)
    return config


@dataclass(frozen=True)
class Gemma4PleForm:
    """The PLE index a line's product container must carry ({@link gemma4_ple_index})."""

    #: The values' storage (`i8` / `i4` / `i2`).
    storage: str
    #: `None` = schema 3 (one f32 scale per (token, layer) row); a group size = schema 4.
    group_size: int | None = None


@dataclass(frozen=True)
class Gemma4Line:
    """What one distribution of the `gemma4/1` contract decides per model; {@link gemma4_plan}
    runs every other check, and builds the pipeline configuration, the same way for each line.

    The regular line ({@link GEMMA4_REGULAR_LINE} — `karume-gemma4`) and the q4_0 QAT line
    ({@link GEMMA4_QAT_Q4_0_LINE} — `karume-gemma4-qat`) differ in which models ship a
    drafter, what each role stores, the PLE index's form, the quant table and the pins — and in
    their models: 12B ships in the q4_0 QAT line only.
    """

    #: The models the line ships (a model outside it fails the plan).
    models: tuple[str, ...]
    #: Whether the model ships an MTP drafter (a weights role of its own).
    has_drafter: Callable[[str], bool]
    #: The model's storage gates.
    storage: Callable[[str], Gemma4StorageTables]
    #: The PLE index form (for the models with per-layer embeddings).
    ple: Gemma4PleForm
    #: Role → output path within the model subtree.
    output_paths: Mapping[str, str]
    #: The model's weights declaration (dtype label → role) — the drafter role only when it ships.
    weights: Callable[[str], Mapping[str, Mapping[str, WeightFiles]]]
    #: The model's quant table (before {@link complete_quant_weights}) and default quant.
    quants: Callable[[str], dict[str, Any]]
    default_quant: Mapping[str, str]
    #: Model → its pinned upstream sources (absent = hand-placed): the product containers must then
    #: name the weights' revision. (The tokenizer asset is held to the upstream `tokenizer.json`
    #: for every model, pinned or not.)
    pins: Mapping[str, ModelPins]
    #: Model → the pin of its drafter, whose revision the drafter container must name (absent =
    #: the drafter container is not held to a revision — the regular E2B drafter records none).
    drafter_pins: Mapping[str, HubPin]
    #: The PLE index form of the original-weights container ({@link GEMMA4_ORIGINAL_ROLE}); `None`
    #: = the line ships no original-weights container.
    original_ple: Gemma4PleForm | None = None


#: The regular line (`karume-gemma4` — {@link PIPELINE}): the tables above. Its checkpoints are
#: hand-placed (`inputs/gemma4/` — `gemma4.sources.checkpoint_dir`), so it has no pins.
GEMMA4_REGULAR_LINE = Gemma4Line(
    models=GEMMA4_REGULAR_MODELS,
    has_drafter=gemma4_has_drafter,
    storage=gemma4_storage_tables,
    ple=Gemma4PleForm(storage="i8"),
    output_paths=GEMMA4_OUTPUT_PATHS,
    weights=gemma4_weights,
    quants=gemma4_quants,
    default_quant=GEMMA4_DEFAULT_QUANT,
    pins={},
    drafter_pins={},
    original_ple=Gemma4PleForm(storage=GEMMA4_ORIGINAL_DTYPE),
)


def _gemma4_target(
    container: Path,
    form: Gemma4PleForm,
    ple_dim: int,
    rope: Mapping[str, Any],
    hidden_size: int,
) -> tuple[Mapping[str, Any], Mapping[str, Any] | None, int]:
    """Check one product container (its graph, its PLE index of `form` and the PLE assets against
    it) and return its graph, its PLE index (`None` for a model without per-layer embeddings) and
    its vocabulary — the same checks for each `model` container of a line."""
    index = (
        gemma4_ple_index(container, storage=form.storage, group_size=form.group_size)
        if ple_dim
        else None
    )
    graph = ir_graph(container)
    vocab_size = gemma4_vocab_size(graph, container)
    assert_gemma4_graph(graph, container, index, rope, hidden_size)
    if index is None:
        assert_gemma4_no_ple_assets(container)
    else:
        if index["tokens"] != vocab_size:
            raise DistError(
                f"{container} の資産 '{PLE_INDEX_ASSET}': tokens {index['tokens']} が製品グラフの"
                f"語彙数 {vocab_size} と違う — 別の語彙で焼かれた組み合わせ"
            )
        assert_gemma4_ple_assets(container, index)
    return graph, index, vocab_size


def gemma4_plan(
    sources: Gemma4Sources,
    model: str = GEMMA4_DEFAULT_MODEL,
    line: Gemma4Line = GEMMA4_REGULAR_LINE,
) -> ModelPlan:
    """gemma4 1 モデルぶんの計画を組む（検査と読み取りをここで全部済ませる）。

    `line` holds what the distribution decides per model ({@link Gemma4Line}); the regular line
    is the default.

    drafter の有無はモデルで決まる（`line.has_drafter`）。MUST: `sources` の drafter 席と
    食い違えば落とす — drafter を配るモデルの席が空なら黙って投機なしの配布形が出来、配らない
    モデルに席があれば別モデル向けの drafter が紛れ込む。
    """
    assert_model_name(model)
    gemma4_checkpoint(model)
    if model not in line.models:
        raise DistError(
            f"model '{model}' is not one of this line's models ({' / '.join(line.models)})"
        )
    if (sources.drafter is not None) != line.has_drafter(model):
        raise DistError(
            f"モデル '{model}' の drafter 系列が"
            f" {'無い' if sources.drafter is None else f'指定されている（{sources.drafter}）'}"
            f" — this line ships {'a' if line.has_drafter(model) else 'no'} drafter for it"
        )
    if (sources.original is not None) != (line.original_ple is not None):
        raise DistError(
            f"model '{model}': the original-weights series is"
            f" {'missing' if sources.original is None else f'given ({sources.original})'}"
            f" — this line ships {'an' if line.original_ple is not None else 'no'}"
            " original-weights container"
        )
    placements = gemma4_placements(sources)
    storage = line.storage(model)
    for role, source in placements.items():
        assert_storage(role, source, storage.required)
        assert_storage(role, source, storage.also_required)
        assert_storage_absent(role, source, storage.forbidden)
    text_config = gemma4_text_config(sources.model)
    where = str(sources.model / GEMMA4_CONFIG_FILE)
    rope = gemma4_rope(text_config, where)
    # MUST: the model table (what the card states) and the upstream config agree on whether the
    # model has per-layer embeddings; the graph and the container assets are then held to the
    # same answer below — a mix of the two forms fails before a byte is placed.
    ple_dim = gemma4_ple_dim(text_config, where)
    if (ple_dim > 0) != (model in GEMMA4_PLE_MODELS):
        raise DistError(
            f"{where}: {GEMMA4_PLE_DIM_KEY} is {ple_dim}, but model '{model}' is"
            f" {'' if model in GEMMA4_PLE_MODELS else 'not '}listed as carrying per-layer"
            " embeddings (gemma4.card.GEMMA4_PLE_MODELS)"
        )
    container = placements[GEMMA4_ROLE]
    hidden_size = gemma4_hidden_size(text_config, where)
    graph, index, vocab_size = _gemma4_target(container, line.ple, ple_dim, rope, hidden_size)
    # The targets the drafter is paired with: the int4 container and, when the line ships one, the
    # original-weights container (the drafter borrows the loaded target's table and slots).
    targets = [(container, graph)]
    original = placements.get(GEMMA4_ORIGINAL_ROLE)
    if original is not None:
        assert line.original_ple is not None  # checked against `sources.original` above
        original_graph, original_index, original_vocab = _gemma4_target(
            original, line.original_ple, ple_dim, rope, hidden_size
        )
        # MUST: both containers are the same model — one vocabulary, one PLE shape — or a preset
        # switch would load another model under the same entry (its `pipelineConfig` is shared).
        if original_vocab != vocab_size:
            raise DistError(
                f"{original}: vocabulary {original_vocab}, but {container} has {vocab_size}"
                " — the two presets' containers are not one model"
            )
        if index is not None and original_index is not None:
            for field in ("layers", "dim"):
                if original_index[field] != index[field]:
                    raise DistError(
                        f"{original}: PLE {field} {original_index[field]}, but {container} has"
                        f" {index[field]} — the two presets' containers are not one model"
                    )
        targets.append((original, original_graph))
    drafter_container = placements.get(GEMMA4_DRAFTER_ROLE)
    if drafter_container is not None:
        drafter_graph = ir_graph(drafter_container)
        for target, target_graph in targets:
            assert_gemma4_drafter_graph(
                drafter_graph, drafter_container, target_graph, target, rope, hidden_size
            )
    assert_gemma4_tokenizer(placements[GEMMA4_TOKENIZER_ROLE], vocab_size)
    # MUST: every model's asset is held to the upstream `tokenizer.json` this plan reads for it,
    # in every line. A borrowed series whose upstream is other bytes, a model's own series compiled
    # before its upstream file was replaced (Google's glitch-token fixes), and another model's or
    # another repository's file (the QAT weights' repositories ship the pre-fix tokenizer) all have
    # the right vocabulary and tokenize some inputs differently.
    assert_gemma4_tokenizer_source(
        placements[GEMMA4_TOKENIZER_ROLE],
        sources.upstream_tokenizer / GEMMA4_CHECKPOINT_TOKENIZER_FILE,
    )
    pins = line.pins.get(model)
    if pins is not None:
        # The container names the weights' commit it was exported from
        # (`gemma4/export_unified.py` bakes it); one from another revision fails here.
        for target, _ in targets:
            assert_upstream_provenance(
                target, license=GEMMA4_LICENSE, revision=pins.weights.revision
            )
    drafter_pin = line.drafter_pins.get(model)
    if drafter_container is not None and drafter_pin is not None:
        # The drafter container names its drafter's commit (`gemma4/export_drafter_qat.py` bakes
        # it): a drafter exported from another revision of the assistant fails here, as the
        # product does above.
        assert_upstream_provenance(
            drafter_container, license=GEMMA4_LICENSE, revision=drafter_pin.revision
        )
    generation = gemma4_generation_defaults(sources.upstream_tokenizer)
    if generation.suppress_tokens is not None:
        assert_gemma4_suppress_tokens(
            generation.suppress_tokens,
            vocab_size,
            str(sources.upstream_tokenizer / GEMMA4_GENERATION_CONFIG_FILE),
        )
    pipeline_config = gemma4_pipeline_config(
        gemma4_max_position(text_config, where),
        rope,
        generation.sampler,
        chunk_length=GEMMA4_CHUNK_LENGTH,
        max_chunk_length=GEMMA4_MAX_CHUNK_LENGTH,
        capacity=GEMMA4_CAPACITY,
        chat_dialect=CHAT_DIALECTS[model],
        suppress_tokens=generation.suppress_tokens,
    )
    weights = line.weights(model)
    return ModelPlan(
        name=model,
        pipeline=GEMMA4_PIPELINE,
        artifacts={
            role: Artifact(line.output_paths[role], source=source)
            for role, source in placements.items()
        },
        weights=weights,
        assets=GEMMA4_ASSETS,
        # requiredLimits は書かない — core の dist が組み立て時に一括導出して焼く
        # （karume/limits.py。計画側の手書きは二重管理として拒否される）。
        quants=complete_quant_weights(weights, line.quants(model)),
        default_quant=line.default_quant[model],
        pipeline_config=pipeline_config,
    )


def gemma4_dist_plan(series_dir: Path, model: str) -> ModelPlan:
    """`--series` の親から gemma4 1 モデルの計画を組む（CLI のディスパッチ先）。

    The regular line reads hand-placed checkpoints only ({@link GEMMA4_REGULAR_LINE} has no pins),
    so there is no snapshot to check here; the q4_0 QAT line's entry
    ({@link gemma4_qat_q4_0_dist_plan}) checks its pinned snapshots first.
    """
    return gemma4_plan(gemma4_sources(series_dir, model), model)


#: 改変告知（Apache 2.0 §4(b)）。**このリポが上流の重みへ加えた変更**を列挙する。
#:
#: MUST: 文面は配布形の中身と対応していること — 値としては妥当な散文なので `verify_dist` も
#: manifest 検査も素通りし、配ってからでないと食い違いに気づけない。
#:
#: MUST: **`GEMMA4_UPSTREAM` へモデルを足す日は、名指しの 1 行もここで足す** — §4(b) の告知は
#: 「再配布した全上流の改変」を述べる義務なので、モデルが増えたのに文面が据え置かれると
#: 告知が 1 モデルぶん欠ける。この席は `Pipeline.root_files`（core の型は `Mapping[str, str]`）で、
#: 組み立てる manifest を見られない — 散文を機械導出できないぶんを門で受ける
#: （`tests/test_distribution.py::TestGemma4LegalText` が帰属表と manifest の両方から検査し、
#: 2 件目が入った瞬間に赤くなる）。
GEMMA4_NOTICE_MARKDOWN = """# NOTICE

This repository redistributes modified forms of `google/gemma-4-E2B-it` and
`google/gemma-4-E4B-it`, which are licensed under the Apache License, Version 2.0 (see
`LICENSE.md`). The models actually included are identified in `README.md` and `karume.json`.
The following changes were made:

- The **text decoder only** was extracted; the vision and audio towers were never read.
- The graph was re-expressed in the Karume container format (a `.krm` part sequence whose
  first part carries the graph and model descriptors) in a states form suited to chunked prefill
  and decode.
- **The weights ship in two storages.** One container per model keeps the checkpoints' BF16
  weight values unchanged, the per-layer embedding tables included (the parameters it keeps in
  float32, the norms among others, hold the same values widened). In the other, **linear
  weights were quantized** to packed int4 (group 32), and the embedding tables to int8; its
  values are therefore not bit-identical to the source checkpoints.
- The exit was narrowed to the selected rows' logits and final hidden states, and the
  **per-layer embedding tables were moved out of the graph** into container assets that the host
  gathers.
- **Rotary position embeddings were moved out of the graph**: the cosine and sine rows are built
  by the host from the declared parameters and passed in as ordinary graph inputs.

For the `e2b` model only, this repository also redistributes a modified form of
`google/gemma-4-E2B-it-assistant` (the multi-token-prediction drafter head), which is licensed
under the same Apache License, Version 2.0.
The same changes apply — text decoder re-expressed in the Karume container format, weights
quantized — with two more: the drafter's clustered sparse output head was replaced by a dense
projection over the full vocabulary, and the drafter reads the key/value states and the embedding
table of the main model instead of carrying its own. **The drafter's weights are quantized to
int8 throughout**, linear layers included, rather than to packed int4 as the main model's linear
weights are in its quantized storage; the drafter is the same int8 container whichever of the main
model's two storages it runs with.

No retraining and no fine-tuning were performed. The original checkpoints are not distributed here.
"""


def gemma4_root_files() -> dict[str, str]:
    """配布リポ直下へ入れる法的テキスト（`karume.dist.Pipeline.root_files`）。

    上流は Apache License 2.0（`gemma4/README.md` 冒頭のライセンス方針 — 2026-09-01 の裁定）で、
    §4(a) は「派生物の受領者にライセンスのコピーを渡す」ことを、§4(b) は「改変したファイルに
    改変した旨の目立つ告知を付ける」ことを求める。ライセンス原文は family で共有する現物
    （{@link _shared.licenses.apache_license_2_0}）を**逐語で**読む — 整形や差し替えをすると
    コピーではなくなる。上流に `NOTICE` ファイルは無い（§4(d) は掛からない）ので、`NOTICE.md`
    は §4(b) の改変告知だけを持つ。
    """
    return {
        "LICENSE.md": apache_license_2_0(),
        "NOTICE.md": GEMMA4_NOTICE_MARKDOWN,
    }


#: `--pipeline gemma4` の 1 行（ドライバが core の PIPELINES へ合成する）。
PIPELINE = Pipeline(
    default_model=GEMMA4_DEFAULT_MODEL,
    repo_name=gemma4_repo_name,
    plan=gemma4_dist_plan,
    # 帰属は**モデル名から一意に決まる**（`GEMMA4_UPSTREAM`）ので、選ばせる軸にしない。
    card_profiles={"gemma4": render_gemma4_model_card},
    # 上流ライセンスの再配布条件（§4）は配布リポ 1 つに掛かるので、読みも組み立ての回数に
    # よらずここで 1 回。
    root_files=gemma4_root_files(),
)


# ---------------------------------------------------------------------------
# The q4_0 QAT line (`karume-gemma4-qat` — module docstring)
# ---------------------------------------------------------------------------

#: The q4_0 QAT line's distribution repository (one repository for its three models, like the
#: regular line's {@link GEMMA4_REPO_NAME}).
GEMMA4_QAT_Q4_0_REPO_NAME = "karume-gemma4-qat"

#: The line's one dtype label (every role stores int4 — the quant seat's name too). Its own word:
#: the regular line's labels ({@link GEMMA4_DTYPE} / {@link GEMMA4_DRAFTER_DTYPE}) follow that
#: line's storage, not this one's.
GEMMA4_QAT_Q4_0_DTYPE = "i4"

#: The q4_0 grid's group (`gemma4.export_product_qat.GROUP_SIZE` — the export's
#: `karume.quantize.DEFAULT_GROUP_SIZE`; a torch-free copy, held equal by the tests): the PLE
#: index's `groupSize`.
GEMMA4_QAT_PLE_GROUP_SIZE = 32

#: The line's output paths (model subtree) — the storage in the file name, as for the regular line.
GEMMA4_QAT_Q4_0_OUTPUT_PATHS: Mapping[str, str] = {
    GEMMA4_ROLE: f"{GEMMA4_ROLE}/model.{GEMMA4_QAT_Q4_0_DTYPE}.krm",
    GEMMA4_DRAFTER_ROLE: f"{GEMMA4_DRAFTER_ROLE}/model.{GEMMA4_QAT_Q4_0_DTYPE}.krm",
    GEMMA4_TOKENIZER_ROLE: f"{GEMMA4_TOKENIZER_SUFFIX}/{GEMMA4_TOKENIZER_FILE}",
}

#: The storage gates of every model of the line: each role holds int4 and none of the other
#: compressed storages the writer can write. MUST: int8 is forbidden in the drafter too — the
#: regular drafter is int8 throughout, and a regular drafter series placed here has the right
#: graph and manifest (it borrows a table of the same name), so only the storage tells it apart;
#: an int8 embedding (the regular E2B / E4B series) is caught the same way on the `model` role.
GEMMA4_QAT_Q4_0_STORAGE = Gemma4StorageTables(
    required={GEMMA4_ROLE: GEMMA4_QAT_Q4_0_DTYPE, GEMMA4_DRAFTER_ROLE: GEMMA4_QAT_Q4_0_DTYPE},
    also_required={},
    forbidden={
        GEMMA4_ROLE: ("f16", "bf16", "i8", "i2"),
        GEMMA4_DRAFTER_ROLE: ("f16", "bf16", "i8", "i2"),
    },
)


def gemma4_qat_q4_0_storage_tables(_model: str) -> Gemma4StorageTables:
    """The line's storage gates (the same for every model — {@link GEMMA4_QAT_Q4_0_STORAGE})."""
    return GEMMA4_QAT_Q4_0_STORAGE


def gemma4_qat_q4_0_has_drafter(model: str) -> bool:
    """Every model of the line ships its QAT drafter (`gemma4.card.GEMMA4_QAT_DRAFTER_UPSTREAM`)."""
    return model in GEMMA4_QAT_DRAFTER_UPSTREAM


def gemma4_qat_q4_0_weights(model: str) -> Mapping[str, Mapping[str, WeightFiles]]:
    """The weights declaration: `model` and, when it ships, `drafter`, both int4."""
    roles = (
        (GEMMA4_ROLE, GEMMA4_DRAFTER_ROLE) if gemma4_qat_q4_0_has_drafter(model) else (GEMMA4_ROLE,)
    )
    return {role: {GEMMA4_QAT_Q4_0_DTYPE: WeightFiles(role)} for role in roles}


#: The line's reference seat, every model's default: the reference summation order (`session`
#: empty — ADR 0104) on the checkpoints' own grid.
GEMMA4_QAT_Q4_0_QUANTS: Mapping[str, Any] = {
    GEMMA4_QAT_Q4_0_DTYPE: {
        "weights": {},
        "session": {},
        "label": "QAT int4 grid, drafter included",
        "description": "The q4_0 QAT checkpoints' own 4-bit grid (packed int4, group 32) for every"
        " linear weight, embedding table and drafter weight; GEMV and attention use the reference"
        " summation order.",
    },
}


#: The line's parallel and fused seats: the regular line's `i4-gemvpar` / `i4-fast` session flags
#: ({@link GEMMA4_QUANTS} — one source for both lines) over the line's one int4 storage, the same
#: g32 grid the regular int4 container stores. What applies (the plan of the mirror's graphs —
#: `linearGemvParallelEligible` / `planFusions`, 2026-11-04): every linear of the E2B / E4B product
#: but the int4 output head (`[262144, H]` — the regular line's head is int8) is on the parallel
#: GEMV's table (the regular line's rows at the same shapes and group: E2B's measured ones and E4B's
#: unmeasured copies of them — ADR 0098 addendum 2026-09-26), the RMS→add fusion takes the regular
#: line's 106 / 127 pairs, and the drafter has the regular int8 drafter's two eligible linears (of
#: 68). 12B gets no such seat: none of its linears is on the table and its width (3840) is not one
#: the fusion takes, so the seats would change only the attention summation under names that claim
#: a parallel GEMV and an RMS fusion.
GEMMA4_QAT_Q4_0_PARALLEL_QUANTS: Mapping[str, Any] = {
    "i4-gemvpar": {
        "weights": {},
        "session": GEMMA4_QUANTS["i4-gemvpar"]["session"],
        "label": "QAT int4 grid with parallel GEMV and attention",
        "description": "Same weights as i4; parallel GEMV (not the int4 output head) and attention."
        " Rounding and generated tokens can differ. Select i4 for the reference order.",
    },
    "i4-fast": {
        "weights": {},
        "session": GEMMA4_QUANTS["i4-fast"]["session"],
        "label": "QAT int4 grid with parallel GEMV, RMS fusion and fused attention",
        "description": "Same weights as i4; parallel GEMV (not the int4 output head), RMS-add"
        " fusion, fused attention. Use i4 for reference summation. Requires fusion-option support.",
    },
}

#: Model → the one sentence on speed its parallel and fused seats carry; the models with those
#: seats. MUST: a model enters only with a measurement behind the sentence — decode ms/token of
#: `i4` / `i4-gemvpar` / `i4-fast`, with and without the drafter, interleaved in one process
#: (2026-11-04, RTX 3080 Ti, Node + Dawn — `outputs/diag/gemma4-lines-2026-11-04/legA/`): both seats
#: were faster than `i4` on every prompt and arm measured (two runs) — without the drafter E2B by
#: 2.0–2.5× and E4B by 2.1–2.9×, with it both by 1.2–1.5×.
GEMMA4_QAT_Q4_0_SPEED_NOTE: Mapping[str, str] = {
    "e2b": " Faster than i4 on the tested E2B device.",
    "e4b": " Faster than i4 on the tested E4B device.",
}


def gemma4_qat_q4_0_quants(model: str) -> dict[str, Any]:
    """The line's quant table for `model`: the reference seat `i4` ({@link GEMMA4_QAT_Q4_0_QUANTS})
    and, for a model with a measured speed note ({@link GEMMA4_QAT_Q4_0_SPEED_NOTE}), the parallel
    and fused seats ({@link GEMMA4_QAT_Q4_0_PARALLEL_QUANTS}) with that note."""
    quants = {name: dict(quant) for name, quant in GEMMA4_QAT_Q4_0_QUANTS.items()}
    note = GEMMA4_QAT_Q4_0_SPEED_NOTE.get(model)
    if note is not None:
        for name, quant in GEMMA4_QAT_Q4_0_PARALLEL_QUANTS.items():
            quants[name] = {
                **quant,
                "session": dict(quant["session"]),
                "description": quant["description"] + note,
            }
    return quants


#: The line's models and their default quant (the reference seat `i4`). Key set =
#: `gemma4.card.GEMMA4_QAT_Q4_0_WEIGHTS_UPSTREAM` (tests).
GEMMA4_QAT_Q4_0_DEFAULT_QUANT: Mapping[str, str] = dict.fromkeys(
    GEMMA4_QAT_Q4_0_WEIGHTS_UPSTREAM, GEMMA4_QAT_Q4_0_DTYPE
)

#: The q4_0 QAT line (`karume-gemma4-qat` — {@link QAT_Q4_0_PIPELINE}).
GEMMA4_QAT_Q4_0_LINE = Gemma4Line(
    models=tuple(GEMMA4_QAT_Q4_0_WEIGHTS_UPSTREAM),
    has_drafter=gemma4_qat_q4_0_has_drafter,
    storage=gemma4_qat_q4_0_storage_tables,
    ple=Gemma4PleForm(storage=GEMMA4_QAT_Q4_0_DTYPE, group_size=GEMMA4_QAT_PLE_GROUP_SIZE),
    output_paths=GEMMA4_QAT_Q4_0_OUTPUT_PATHS,
    weights=gemma4_qat_q4_0_weights,
    quants=gemma4_qat_q4_0_quants,
    default_quant=GEMMA4_QAT_Q4_0_DEFAULT_QUANT,
    pins=QAT_TARGET_PINS,
    drafter_pins=QAT_DRAFTER_PINS,
)


def gemma4_qat_q4_0_product_suffix(model: str) -> str:
    """The suffix of model `model`'s product series in the line: E2B / E4B's QAT series (a line
    separate from the regular one — `gemma4.sources.QAT_PINS`), 12B's product series (its one
    line, already the QAT checkpoint). `gemma4.export_drafter_qat.lender_suffix` reads the lender
    by this rule."""
    return GEMMA4_QAT_PRODUCT_SUFFIX if model in QAT_PINS else GEMMA4_PRODUCT_SUFFIX


def gemma4_qat_q4_0_repo_name(_model: str) -> str:
    """The line's one repository ({@link GEMMA4_QAT_Q4_0_REPO_NAME})."""
    return GEMMA4_QAT_Q4_0_REPO_NAME


def gemma4_qat_q4_0_sources(series_dir: Path, model: str = GEMMA4_DEFAULT_MODEL) -> Gemma4Sources:
    """The inputs of model `model` of the line: its product and QAT drafter series, the regular
    tokenizer series ({@link GEMMA4_TOKENIZER_SERIES} — the line reads the regular `-it`
    tokenizers), and the hub snapshots of its QAT target pins (`gemma4.sources.QAT_TARGET_PINS`).

    A model outside the line fails loudly.
    """
    if model not in GEMMA4_QAT_Q4_0_WEIGHTS_UPSTREAM:
        raise DistError(
            f"the q4_0 QAT line has no model '{model}'"
            f" (known: {' / '.join(sorted(GEMMA4_QAT_Q4_0_WEIGHTS_UPSTREAM))})"
        )
    return Gemma4Sources(
        product=series_dir / gemma4_series_name(model, gemma4_qat_q4_0_product_suffix(model)),
        drafter=series_dir / gemma4_series_name(model, GEMMA4_QAT_DRAFTER_SUFFIX),
        tokenizer=series_dir
        / gemma4_series_name(GEMMA4_TOKENIZER_SERIES[model], GEMMA4_TOKENIZER_SUFFIX),
        model=qat_target_checkpoint_dir(model),
        upstream_tokenizer=qat_target_tokenizer_dir(model),
    )


def gemma4_qat_q4_0_plan(sources: Gemma4Sources, model: str = GEMMA4_DEFAULT_MODEL) -> ModelPlan:
    """One model's plan of the line ({@link gemma4_plan} with {@link GEMMA4_QAT_Q4_0_LINE})."""
    return gemma4_plan(sources, model, GEMMA4_QAT_Q4_0_LINE)


def gemma4_qat_q4_0_dist_plan(series_dir: Path, model: str) -> ModelPlan:
    """The CLI's entry for the line: both target snapshots are checked against their pins first
    ({@link gemma4_dist_plan}'s rule), then the plan is built."""
    sources = gemma4_qat_q4_0_sources(series_dir, model)
    try:
        assert_model_pins(sources.model, sources.upstream_tokenizer, QAT_TARGET_PINS[model])
    except ValueError as error:
        raise DistError(str(error)) from error
    return gemma4_qat_q4_0_plan(sources, model)


def _qat_q4_0_notice_sources() -> str:
    """The NOTICE's list of redistributed upstreams: per model its QAT checkpoint, its QAT drafter
    and its base model (the tokenizer's source)."""
    return "\n".join(
        f"- `{GEMMA4_QAT_Q4_0_WEIGHTS_UPSTREAM[model]}`, its drafter"
        f" `{GEMMA4_QAT_DRAFTER_UPSTREAM[model]}` and the tokenizer of its base model"
        f" `{GEMMA4_UPSTREAM[model]}`"
        for model in GEMMA4_QAT_Q4_0_WEIGHTS_UPSTREAM
    )


#: The line's modification notice (Apache 2.0 §4(b)). MUST: the text matches the distribution's
#: content ({@link GEMMA4_NOTICE_MARKDOWN}'s MUST — valid prose passes every machine check), and
#: it names every upstream of the line's attribution tables (built from them — tests).
GEMMA4_QAT_Q4_0_NOTICE_MARKDOWN = (
    """# NOTICE

This repository redistributes modified forms of Google's Gemma 4 q4_0 quantization-aware-trained
(QAT) checkpoints, of their QAT multi-token-prediction drafters and of the tokenizers of their base
models, listed below, which are licensed under the Apache License, Version 2.0 (see `LICENSE.md`).
The models actually included are identified in `README.md` and `karume.json`.

"""
    + _qat_q4_0_notice_sources()
    + """

The following changes were made:

- The **text decoder only** was extracted; the vision and audio towers (for the 12B checkpoint,
  its encoder-free vision and audio embedders) were never read.
- The graphs were re-expressed in the Karume container format (a `.krm` part sequence whose first
  part carries the graph and model descriptors) in a states form suited to chunked prefill and
  decode.
- **The weights keep the checkpoints' own 4-bit grid**: the linear weights and the embedding table
  of each model, the per-layer embedding tables of the E2B and E4B checkpoints and the drafters'
  weights are stored as packed int4 (group 32: the codes of the grid and one step per group,
  refitted from the stored values) instead of being rounded again. The values are therefore not
  bit-identical to the source checkpoints. The parameters off the grid (the norms, among others)
  are stored in float32.
- The exit was narrowed to the selected rows' logits and final hidden states. For the E2B and E4B
  checkpoints, the **per-layer embedding tables were moved out of the graph** into container
  assets that the host gathers (the 12B checkpoint has none).
- **Rotary position embeddings were moved out of the graph**: the cosine and sine rows are built
  by the host from the declared parameters and passed in as ordinary graph inputs.
- Each drafter's output head is the dense projection over the full vocabulary (a clustered sparse
  head an assistant checkpoint carries is not used), and each drafter reads the key/value states
  and the embedding table of its main model instead of carrying its own.

No retraining and no fine-tuning were performed. The original checkpoints are not distributed here.
"""
)


def gemma4_qat_q4_0_root_files() -> dict[str, str]:
    """The line's legal texts at the repository root ({@link gemma4_root_files}'s rule: the
    licence verbatim and the line's modification notice)."""
    return {
        "LICENSE.md": apache_license_2_0(),
        "NOTICE.md": GEMMA4_QAT_Q4_0_NOTICE_MARKDOWN,
    }


#: `--pipeline gemma4-qat` (the driver adds it to the core's PIPELINES).
QAT_Q4_0_PIPELINE = Pipeline(
    default_model=GEMMA4_DEFAULT_MODEL,
    repo_name=gemma4_qat_q4_0_repo_name,
    plan=gemma4_qat_q4_0_dist_plan,
    card_profiles={"gemma4-qat": render_gemma4_qat_q4_0_model_card},
    root_files=gemma4_qat_q4_0_root_files(),
)
