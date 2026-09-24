"""Gemma 4 の配布 recipe — 系列レイアウト・出力 path 表・quant 表・カードの選択（ADR 0065 決定 2）。

汎用の組み立てエンジン（配置・共有席の畳み込み・sha256・manifest・staging/swap・検証）は
`karume.dist` が持つ。ここが持つのは **gemma4 固有の事実**だけ: どの系列ディレクトリから何を
拾い、配布形のどの path へ、どの dtype ラベルで並べ、どの quant を既定にするか。

配布するのは**グラフ 2 本（製品 + 借り手の drafter）+ 付帯資産 1 種**
（ADR 0084 決定 5 / 0096 段 2 / 0109 決定 4）:

- `model` — 製品グラフのコンテナ（`gemma4/export_product.py` が書く `.krm` の part 列。PLE を
  グラフから外し、出口を最終行 logits にした 1 系列）。格納は**混成**で、埋め込みが i8・
  linear が packed i4。**PLE はこの容器の資産**として同梱される（ADR 0109 決定 4）— 索引
  `ple_index`（schema 3）と `ple.values.<k>` / `ple.scales.<k>` の block 列で、weights では
  ない（ホストが `per_layer_inputs` を組むための表）。読み手は
  `packages/models/src/gemma/ple-index.ts`。
- `drafter` — MTP drafter のコンテナ（`gemma4/export_drafter.py`・ADR 0096 段 2）。格納は
  **i8 単一**で、linear まで i8（i4 g32 に落とすと受理率が 1 〜 3 割落ちる — 台本の実測）。
  役割ごとに dtype ラベルが違うので、quant 表の `weights` 写像は 2 席とも埋まる。
- `tokenizer` — compile 済みトークナイザ資産（`gemma4/tokenizer.py`・ADR 0084 決定 1）

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
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from _shared.container_read import read_asset, read_asset_declarations
from _shared.licenses import apache_license_2_0
from _shared.paths import INPUTS_ROOT
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
)
from karume.ple import (
    PLE_INDEX_ASSET,
    PLE_INDEX_ROLE,
    PLE_INDEX_SCHEMA,
    PLE_PACK_FACTOR,
    PLE_ROLES,
    PLE_SCALE_BYTES,
)
from karume.verify import ContainerError

from .card import GEMMA4_UPSTREAM, render_gemma4_model_card
from .rope import (
    BAKED_TABLE_INFIX,
    FULL_ATTENTION,
    HEAD_DIM_FIELD,
    SLIDING_ATTENTION,
    RopeSpecError,
    rope_specs,
)

#: パイプライン契約（ADR 0041 §2 — モデル単位）。TS 側の受理集合は
#: `GEMMA4_PIPELINE_NAME` / `GEMMA4_PIPELINE_MAJOR`（`packages/models/src/gemma/config.ts`）。
GEMMA4_PIPELINE = "gemma4/1"

#: 既定のモデル名（`--model` を省いた組み立てが入れるモデル）。綴りの受理集合は帰属表
#: （`gemma4.card.GEMMA4_UPSTREAM`）が持つ。
GEMMA4_DEFAULT_MODEL = "e2b"

#: 系列名の接頭辞（`gemma4-<モデル名>-<接尾>`）。
GEMMA4_PREFIX = "gemma4"

#: 配布リポ名（ADR 0092 決定 1 の `karume-<family>`）。**モデル名から導かない** — 家族 1 リポ
#: なので、E2B / E4B / 12B のどれを組んでも行き先は 1 つ。
GEMMA4_REPO_NAME = "karume-gemma4"

#: 上流の手置き資産の親（`inputs/gemma4/<チェックポイント名>/` — assets-layout）。
GEMMA4_INPUTS_DIRNAME = "gemma4"

#: 系列の接尾（`gemma4/export_product.py` の `DEFAULT_OUT_DIR` と `gemma4/tokenizer.py` の
#: `ASSET_PATH` — 書き手と読み手が同じ 1 語から組む）。
GEMMA4_PRODUCT_SUFFIX = "product"
GEMMA4_TOKENIZER_SUFFIX = "tokenizer"
#: MTP drafter の系列接尾（`gemma4/export_drafter.py` の `DEFAULT_OUT_DIR`）。
GEMMA4_DRAFTER_SUFFIX = "drafter"

#: 系列側のファイル名（`gemma4.export.MODEL_FILE` / `tokenizer.ASSET_PATH` の綴り）。
#: **代表 path** なので、分割されていれば {@link karume.dist.component_parts} が連番へ解決する。
GEMMA4_MODEL_FILE = "model.krm"
GEMMA4_TOKENIZER_FILE = "tokenizer.json"

#: 上流チェックポイントが持つ推奨サンプラの出どころ（ADR 0083 決定 7）と、モデルが決める数
#: （位置の上限・RoPE の式）の出どころ。text 部は `config.json` の `text_config` 節。
GEMMA4_GENERATION_CONFIG_FILE = "generation_config.json"
GEMMA4_CONFIG_FILE = "config.json"
GEMMA4_TEXT_CONFIG_KEY = "text_config"
GEMMA4_MAX_POSITION_KEY = "max_position_embeddings"
GEMMA4_HIDDEN_SIZE_KEY = "hidden_size"

#: 役割名（manifest の weights / assets が指す内部キー）。
GEMMA4_ROLE = "model"
GEMMA4_TOKENIZER_ROLE = "tokenizer"
#: MTP drafter の役割名（weights の 2 本目 — 貸し手 `model` が居ないと単独では実行できない）。
GEMMA4_DRAFTER_ROLE = "drafter"

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
    GEMMA4_DRAFTER_ROLE: f"{GEMMA4_DRAFTER_ROLE}/model.i8.krm",
    GEMMA4_TOKENIZER_ROLE: f"{GEMMA4_TOKENIZER_SUFFIX}/{GEMMA4_TOKENIZER_FILE}",
}

#: 容器の束縛表に**必ず在る**格納の語彙。製品グラフは混成なので 2 つとも要求する（他
#: family の {@link assert_storage} は 1 つずつしか見ないので、表を 2 枚持って 2 度掛ける）。
#: i8 は埋め込み（i4 適格外・recipe README の "not int4-eligible"）・i4 は linear の重み。
#: 片方だけを要求すると「埋め込みまで i4 に落ちた系列」「linear が i8 のままの系列」が
#: それぞれ素通りする — どちらも shape も manifest も正しいまま、品質と速度だけが変わる。
#: drafter は**単一格納**（linear まで i8 — `gemma4/export_drafter.py` の実測）なので、要求は
#: i8 の 1 枚だけで、2 枚目の表に行を持たない。
GEMMA4_STORAGE_REQUIREMENTS: Mapping[str, str] = {GEMMA4_ROLE: "i4", GEMMA4_DRAFTER_ROLE: "i8"}
GEMMA4_STORAGE_ALSO_REQUIRED: Mapping[str, str] = {GEMMA4_ROLE: "i8"}

#: 各役割の束縛表に**あってはならない**格納の語彙（{@link assert_storage_absent}）。
#: 台本が焼く格納形は i8 + i4 + f32 の 1 系列だけなので、f16 の混入は「別 family の系列 root を
#: 指した」印にしかならない（系列 root の取り違えは数値の門では原理的に検出できない —
#: ADR 0027 / 0029。他 family と同じ規律で、書き出しうる圧縮格納のうち**在ってはならない側を
#: 全部**名指しする）。
#: drafter は i4 も禁止側 — 存在検査（i8 が在る）は「linear だけ i4 に落ちた drafter」を
#: 素通りさせる（出力ヘッドの i8 で要求が満たされる）。受理率が 1 〜 3 割落ちるだけの資産は
#: 形も manifest も正しいままなので、ここが唯一の検出器になる。
GEMMA4_STORAGE_FORBIDDEN: Mapping[str, tuple[str, ...]] = {
    GEMMA4_ROLE: ("f16",),
    GEMMA4_DRAFTER_ROLE: ("f16", "i4"),
}

#: 格納 dtype のラベル（quant 席の綴りでもある）。役割ごとに**基底格納が 1 つずつ**なので
#: {@link complete_quant_weights} の自動補完が quant 表の weights を 2 席とも埋める
#: （`{model: "i4", drafter: "i8"}`）。
GEMMA4_DTYPE = "i4"
GEMMA4_DRAFTER_DTYPE = "i8"

#: weights の宣言（dtype ラベル → 役割名）。分割は現物が決めるので、ここが指すのは代表 1 役。
#: MUST: drafter を **weights の 2 本目**として宣言する（assets ではない）— IR コンテナで
#: グラフを持ち、`createSession` が食う側の資産だからで、hub の `ResolveOptions.weights` が
#: 「既定では取らない」を表せる軸もここ 1 本しかない。
GEMMA4_WEIGHTS: Mapping[str, Mapping[str, WeightFiles]] = {
    GEMMA4_ROLE: {GEMMA4_DTYPE: WeightFiles(GEMMA4_ROLE)},
    GEMMA4_DRAFTER_ROLE: {GEMMA4_DRAFTER_DTYPE: WeightFiles(GEMMA4_DRAFTER_ROLE)},
}

#: 同じ格納系列に参照加算・GEMV並列加算・RMS融合を用意する（ADR 0098 / 0104）。
#: 明示したi4の意味を保持し、既定quantだけを高速化付きへ向ける。
GEMMA4_QUANTS: Mapping[str, Any] = {
    GEMMA4_DTYPE: {
        "weights": {},
        "session": {},
        "label": "Packed int4 linear, int8 embeddings",
        "description": "The only storage series: the main model's linear weights in packed int4"
        " (group 32) and its embedding tables in int8, which are not int4-eligible. The drafter"
        " head is int8 throughout.",
    },
    "i4-gemvpar": {
        "weights": {},
        "session": {"linearGemvReduce": "parallel"},
        "label": "Packed int4 with parallel GEMV",
        "description": "The same packed weights as i4, with parallel GEMV summation. "
        "Faster on tested E2B devices; rounding and generated tokens can differ. "
        "Select i4 for the reference summation order.",
    },
    "i4-fast": {
        "weights": {},
        "session": {"linearGemvReduce": "parallel", "fuseRmsNormAdd": True},
        "label": "Packed int4 with parallel GEMV and RMS fusion",
        "description": "Same weights as i4; parallel GEMV and RMS-add fusion for E2B. "
        "Use i4 for reference summation or i4-gemvpar without fusion. "
        "Requires fusion-option support.",
    },
}

GEMMA4_DEFAULT_QUANT = "i4-fast"

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

#: compile 済みトークナイザ資産の形式識別子（書き手は `_shared/gemma_tokenizer.py`）。
GEMMA4_TOKENIZER_FORMAT = "karume-gemma-tokenizer/1"

#: PLE 索引の版と欄（読み手 `packages/models/src/gemma/ple-index.ts` の
#: `SCHEMA` / `INDEX_KEYS` / `TABLE_KEYS` / `BLOCK_KEYS` の鏡像 — 焼く側が先に落とす）。
#: 版そのものは書き手（`karume.ple`）と共有する。
GEMMA4_PLE_SCHEMA = PLE_INDEX_SCHEMA
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
    """組み立ての入力。系列 2 本（製品グラフ + PLE sidecar / トークナイザ資産）と、
    上流チェックポイント（推奨サンプラの出どころ — 読むのは
    `generation_config.json` 1 本だけで、重みには触らない）。
    """

    product: Path
    drafter: Path
    tokenizer: Path
    model: Path


def gemma4_sources(series_dir: Path, model: str = GEMMA4_DEFAULT_MODEL) -> Gemma4Sources:
    """系列の親ディレクトリ（`outputs/series/`）と `_shared.paths` の綴りから入力を引く。"""
    return Gemma4Sources(
        product=series_dir / gemma4_series_name(model, GEMMA4_PRODUCT_SUFFIX),
        drafter=series_dir / gemma4_series_name(model, GEMMA4_DRAFTER_SUFFIX),
        tokenizer=series_dir / gemma4_series_name(model, GEMMA4_TOKENIZER_SUFFIX),
        model=INPUTS_ROOT / GEMMA4_INPUTS_DIRNAME / gemma4_checkpoint(model),
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


def gemma4_ple_index(container: Path, *, storage: str = "i8") -> Mapping[str, Any]:
    """`model` 容器の資産 `ple_index` を読んで**形まで**落とす。

    読み手（`packages/models/src/gemma/ple-index.ts` の `parseGemma4PleIndex`）の受理集合の
    鏡像で、**schema 3 だけ**を受ける（旧 sidecar の索引は配布形ごと退役した — ADR 0109
    決定 1 の major 繰り上げ規則）。`storage` は呼び手が要求する格納（QAT は `i2` / `i4`）。

    MUST: 配ってから利用者の手元で初めて落ちる形にしない — 同じ検査を組み立て側でも掛ける。
    """
    where = f"{container} の資産 '{PLE_INDEX_ASSET}'"
    if storage not in PLE_PACK_FACTOR:
        raise DistError(f"未対応 PLE storage: {storage}")
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
    unknown = sorted(set(raw) - set(GEMMA4_PLE_INDEX_KEYS))
    if unknown:
        raise DistError(f"{where}: 未知キー {unknown}（許可: {list(GEMMA4_PLE_INDEX_KEYS)}）")
    if raw.get("schema") != GEMMA4_PLE_SCHEMA:
        raise DistError(
            f"{where}: schema が {raw.get('schema')!r}（期待 {GEMMA4_PLE_SCHEMA}）"
            " — 旧 sidecar の索引は読まない"
        )
    if raw.get("storage") != storage:
        raise DistError(f"{where}: storage が {raw.get('storage')!r}（期待 {storage}）")
    tokens = _positive_int(raw, "tokens", where)
    layers = _positive_int(raw, "layers", where)
    dim = _positive_int(raw, "dim", where)
    factor = PLE_PACK_FACTOR[storage]
    if dim % factor:
        raise DistError(f"{where}: dim {dim} が格納 '{storage}' の詰め数 {factor} で割り切れない")
    scale = raw.get("embedScale")
    if (
        not isinstance(scale, int | float)
        or isinstance(scale, bool)
        or not math.isfinite(scale)
        or scale <= 0
    ):
        raise DistError(f"{where}: embedScale が正の有限数でない（{scale!r}）")
    return {
        "storage": storage,
        "tokens": tokens,
        "layers": layers,
        "dim": dim,
        "embedScale": scale,
        GEMMA4_PLE_VALUES_KEY: gemma4_ple_table(
            raw, GEMMA4_PLE_VALUES_KEY, layers * dim // factor, tokens, where
        ),
        GEMMA4_PLE_SCALES_KEY: gemma4_ple_table(
            raw, GEMMA4_PLE_SCALES_KEY, layers * PLE_SCALE_BYTES, tokens, where
        ),
    }


def gemma4_placements(sources: Gemma4Sources) -> dict[str, Path]:
    """役割名 → 出所のファイル。出力の path は {@link GEMMA4_OUTPUT_PATHS} が持つ。

    この表に無いものは出力へ入らない（製品系列に同居する `ple.probe.safetensors` と
    `reference.json`・drafter 系列の `drafter-golden.*.safetensors` はこれで落ちる — どれも
    検収と出所記録のためのもので実行に要らない）。PLE は `model` 容器の中なので席を持たない。
    """
    return {
        GEMMA4_ROLE: sources.product / GEMMA4_MODEL_FILE,
        GEMMA4_DRAFTER_ROLE: sources.drafter / GEMMA4_MODEL_FILE,
        GEMMA4_TOKENIZER_ROLE: sources.tokenizer / GEMMA4_TOKENIZER_FILE,
    }


#: assets の宣言（asset 名 → 役割名）。PLE は容器の中へ移ったので、残るのはトークナイザ 1 本。
GEMMA4_ASSETS: Mapping[str, str] = {GEMMA4_TOKENIZER_ROLE: GEMMA4_TOKENIZER_ROLE}


def gemma4_vocab_size(graph: Mapping[str, Any], path: Path) -> int:
    """選択行 logits 出口の語彙数をグラフの出力宣言から読む（`[1, R, V]` — ADR 0083 決定 6）。

    出力が **2 本**（`logits[1, R, V]` → `hidden[1, R, H]` の順）であることまで見るのは、
    検収用の 2 系列（logits opt-in / token-only）が同じ系列名の下に紛れ込むと**幅だけが別の
    意味の数**になるため。行軸が記号 `R` なのは投機 verify が 1 回で複数行を採点するからで、
    通常の decode はそこを 1 に束縛する。V は主 embedding の行数そのもので、PLE sidecar と
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
    index: Mapping[str, Any],
    rope: Mapping[str, Any],
    hidden_size: int,
) -> None:
    """グラフ入力の並び・形・記号の割れ方を、配置の前に実測する。

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
    から決まらないもの**がちょうど 1 本（容量記号）。TS 側 `Gemma4Pipeline` はこの 1 本を容量の
    束縛点にするので、割れ方が変わるとロード時に落ちる（`capacitySymbolOf` の同じ検査）。
    `R` は `last_row[R]` が束縛するので、この勘定では自由記号に数えない。
    """
    inputs = graph_inputs(graph, path)
    names = tuple(inputs)
    if names != GEMMA4_GRAPH_INPUTS:
        raise DistError(
            f"{path} のグラフ入力が {list(names)} で、期待の {list(GEMMA4_GRAPH_INPUTS)} と違う"
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
    per_layer = inputs[GEMMA4_PER_LAYER_INPUTS]
    if len(per_layer) != 4:
        raise DistError(
            f"{path} の入力 '{GEMMA4_PER_LAYER_INPUTS}' が {per_layer!r} — [1, M, 層数, 次元]"
            "の 4 軸でない"
        )
    for axis, field in ((2, "layers"), (3, "dim")):
        if per_layer[axis] != index[field]:
            raise DistError(
                f"{path} の入力 '{GEMMA4_PER_LAYER_INPUTS}' の軸 {axis} が {per_layer[axis]!r}、"
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
    for key, role in PLE_ROLES.items():
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


def gemma4_sampler(model_dir: Path) -> dict[str, Any]:
    """上流 `generation_config.json` の推奨サンプラを `pipelineConfig.sampler` へ写す。

    MUST: 値を写経しない（ADR 0083 決定 7 — 「既定値は配布形が宣言する」の出どころは上流の
    宣言そのもの）。欄名だけは TS 側 `SamplerSpec` の綴りへ翻訳する（`top_k` → `topK`）。
    値域は TS 側 `parseGemma4PipelineConfig` と同じ — 配ってから parse で落ちる形にしない。
    """
    where = str(model_dir / GEMMA4_GENERATION_CONFIG_FILE)
    raw = _read_json(model_dir / GEMMA4_GENERATION_CONFIG_FILE, "上流の生成既定")
    if not isinstance(raw, dict):
        raise DistError(f"{where}: 最上位オブジェクトでない")
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
) -> dict[str, Any]:
    """`pipelineConfig`（TS 側スキーマの 6 欄）を組む。

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
    return {
        "chunkLength": chunk_length,
        "maxChunkLength": max_chunk_length,
        "maxPosition": max_position,
        "capacity": capacity,
        "rope": {layer_type: dict(spec) for layer_type, spec in rope.items()},
        "sampler": dict(sampler),
    }


def gemma4_plan(sources: Gemma4Sources, model: str = GEMMA4_DEFAULT_MODEL) -> ModelPlan:
    """gemma4 1 モデルぶんの計画を組む（検査と読み取りをここで全部済ませる）。"""
    assert_model_name(model)
    placements = gemma4_placements(sources)
    for role, source in placements.items():
        assert_storage(role, source, GEMMA4_STORAGE_REQUIREMENTS)
        assert_storage(role, source, GEMMA4_STORAGE_ALSO_REQUIRED)
        assert_storage_absent(role, source, GEMMA4_STORAGE_FORBIDDEN)
    text_config = gemma4_text_config(sources.model)
    where = str(sources.model / GEMMA4_CONFIG_FILE)
    rope = gemma4_rope(text_config, where)
    container = placements[GEMMA4_ROLE]
    index = gemma4_ple_index(container)
    graph = ir_graph(container)
    vocab_size = gemma4_vocab_size(graph, container)
    hidden_size = gemma4_hidden_size(text_config, where)
    assert_gemma4_graph(graph, container, index, rope, hidden_size)
    drafter_container = placements[GEMMA4_DRAFTER_ROLE]
    assert_gemma4_drafter_graph(
        ir_graph(drafter_container), drafter_container, graph, container, rope, hidden_size
    )
    if index["tokens"] != vocab_size:
        raise DistError(
            f"{container} の資産 '{PLE_INDEX_ASSET}': tokens {index['tokens']} が製品グラフの"
            f"語彙数 {vocab_size} と違う — 別の語彙で焼かれた組み合わせ"
        )
    assert_gemma4_ple_assets(container, index)
    assert_gemma4_tokenizer(placements[GEMMA4_TOKENIZER_ROLE], vocab_size)
    pipeline_config = gemma4_pipeline_config(
        gemma4_max_position(text_config, where),
        rope,
        gemma4_sampler(sources.model),
        chunk_length=GEMMA4_CHUNK_LENGTH,
        max_chunk_length=GEMMA4_MAX_CHUNK_LENGTH,
        capacity=GEMMA4_CAPACITY,
    )
    return ModelPlan(
        name=model,
        pipeline=GEMMA4_PIPELINE,
        artifacts={
            role: Artifact(GEMMA4_OUTPUT_PATHS[role], source=source)
            for role, source in placements.items()
        },
        weights=GEMMA4_WEIGHTS,
        assets=GEMMA4_ASSETS,
        # requiredLimits は書かない — core の dist が組み立て時に一括導出して焼く
        # （karume/limits.py。計画側の手書きは二重管理として拒否される）。
        quants=complete_quant_weights(
            GEMMA4_WEIGHTS,
            GEMMA4_QUANTS if model == "e2b" else {GEMMA4_DTYPE: GEMMA4_QUANTS[GEMMA4_DTYPE]},
        ),
        default_quant=GEMMA4_DEFAULT_QUANT if model == "e2b" else GEMMA4_DTYPE,
        pipeline_config=pipeline_config,
    )


def gemma4_dist_plan(series_dir: Path, model: str) -> ModelPlan:
    """`--series` の親から gemma4 1 モデルの計画を組む（CLI のディスパッチ先）。"""
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

This repository redistributes a modified form of `google/gemma-4-E2B-it`, which is licensed under
the Apache License, Version 2.0 (see `LICENSE.md`). The following changes were made:

- The **text decoder only** was extracted; the vision and audio towers were never read.
- The graph was re-expressed in the Karume container format (a `.krm` part sequence whose
  first part carries the graph and model descriptors) in a states form suited to chunked prefill
  and decode.
- **Linear weights were quantized** to packed int4 (group 32) and the embedding tables to int8.
  The values are therefore not bit-identical to the source checkpoint.
- The **per-layer embedding tables were moved out of the graph** into container assets that the
  host gathers, and the exit was narrowed to the selected rows' logits and final hidden states.
- **Rotary position embeddings were moved out of the graph**: the cosine and sine rows are built
  by the host from the declared parameters and passed in as ordinary graph inputs.

This repository also redistributes a modified form of `google/gemma-4-E2B-it-assistant` (the
multi-token-prediction drafter head), which is licensed under the same Apache License, Version 2.0.
The same changes apply — text decoder re-expressed in the Karume container format, weights
quantized — with two more: the drafter's clustered sparse output head was replaced by a dense
projection over the full vocabulary, and the drafter reads the key/value states and the embedding
table of the main model instead of carrying its own. **The drafter's weights are quantized to
int8 throughout**, linear layers included, rather than to packed int4 as the main model's are.

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
