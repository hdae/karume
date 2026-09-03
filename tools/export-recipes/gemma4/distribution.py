"""Gemma 4 の配布 recipe — 系列レイアウト・出力 path 表・quant 表・カードの選択（ADR 0065 決定 2）。

汎用の組み立てエンジン（配置・共有席の畳み込み・sha256・manifest・staging/swap・検証）は
`karume.dist` が持つ。ここが持つのは **gemma4 固有の事実**だけ: どの系列ディレクトリから何を
拾い、配布形のどの path へ、どの dtype ラベルで並べ、どの quant を既定にするか。

配布するのは**製品グラフ 1 本 + 同じ digest set の付帯資産 2 種**（ADR 0084 決定 5）:

- `model` — 製品グラフのコンテナ（`gemma4/export_product.py` が書く shard 列。PLE を外し、
  出口を最終行 logits にした 1 系列）。格納は**混成**で、埋め込みが i8・linear が packed i4。
- `tokenizer` — compile 済みトークナイザ資産（`gemma4/tokenizer.py`・ADR 0084 決定 1）
- PLE sidecar — 索引 `ple.json` と token 範囲 shard（ADR 0085）。**weights ではない**
  （IR コンテナでもグラフでもなく、ホストが `per_layer_inputs` を組むための表）ので
  `assets` の席に載る。shard は 1 本ずつ独立の資産で、**asset 名は索引が書いた
  ファイル名そのもの** — 読み手（`packages/models/src/gemma/ple.ts`）は `ple.json` の
  `shards[].file` を鍵に引くので、そこに別の綴りを挟むと索引と取得キーの対応が
  「位置で合わせる」形になり、片方だけ並べ替えた組が黙って通る。

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

#: 既定のモデル名（= 既定のリポ名 `karume-gemma4-e2b` の末尾）。綴りの受理集合は帰属表
#: （`gemma4.card.GEMMA4_UPSTREAM`）が持つ。
GEMMA4_DEFAULT_MODEL = "e2b"

#: 系列名とリポ名の接頭辞（`karume-gemma4-<モデル名>`）。
GEMMA4_PREFIX = "gemma4"

#: 上流の手置き資産の親（`inputs/gemma4/<チェックポイント名>/` — assets-layout）。
GEMMA4_INPUTS_DIRNAME = "gemma4"

#: 系列の接尾（`gemma4/export_product.py` の `DEFAULT_OUT_DIR` と `gemma4/tokenizer.py` の
#: `ASSET_PATH` — 書き手と読み手が同じ 1 語から組む）。
GEMMA4_PRODUCT_SUFFIX = "product"
GEMMA4_TOKENIZER_SUFFIX = "tokenizer"

#: 系列側のファイル名（`gemma4.export.MODEL_FILE` / `export_product.PLE_INDEX_FILE` /
#: `tokenizer.ASSET_PATH` の綴り）。**代表 path** なので、分割されていれば
#: {@link karume.dist.component_shards} が連番へ解決する。
GEMMA4_MODEL_FILE = "model.safetensors"
GEMMA4_PLE_INDEX_FILE = "ple.json"
GEMMA4_TOKENIZER_FILE = "tokenizer.json"

#: 上流チェックポイントが持つ推奨サンプラの出どころ（ADR 0083 決定 7）と、モデルが決める数
#: （位置の上限・RoPE の式）の出どころ。text 部は `config.json` の `text_config` 節。
GEMMA4_GENERATION_CONFIG_FILE = "generation_config.json"
GEMMA4_CONFIG_FILE = "config.json"
GEMMA4_TEXT_CONFIG_KEY = "text_config"
GEMMA4_MAX_POSITION_KEY = "max_position_embeddings"

#: 役割名（manifest の weights / assets が指す内部キー）。
GEMMA4_ROLE = "model"
GEMMA4_TOKENIZER_ROLE = "tokenizer"
GEMMA4_PLE_INDEX_ROLE = "ple_index"
GEMMA4_PLE_ROLE_PREFIX = "ple_"

#: 配布形の中の PLE sidecar の置き場（モデルサブツリー内）。
GEMMA4_PLE_DIR = "ple"

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

#: 出力の相対 path（**モデルサブツリー内**）— 配置表と manifest が共有する 1 箇所。格納 dtype を
#: ファイル名に出すのは他 family と同じ形（系列が 2 本並んでも取り違えようがない綴り）。
#: PLE shard は索引が書いたファイル名をそのまま使うので、この表には代表の 3 席だけが載る。
GEMMA4_OUTPUT_PATHS: Mapping[str, str] = {
    GEMMA4_ROLE: f"{GEMMA4_ROLE}/model.i4.safetensors",
    GEMMA4_TOKENIZER_ROLE: f"{GEMMA4_TOKENIZER_SUFFIX}/{GEMMA4_TOKENIZER_FILE}",
    GEMMA4_PLE_INDEX_ROLE: f"{GEMMA4_PLE_DIR}/{GEMMA4_PLE_INDEX_FILE}",
}

#: コンテナのヘッダに**必ず在る**格納 dtype。混成なので 2 つとも要求する（他 family の
#: {@link assert_storage} は 1 dtype ずつしか見ないので、表を 2 枚持って 2 度掛ける）。
#: I8 は埋め込み（i4 適格外・recipe README の "not int4-eligible"）・I4 は linear の重み。
#: 片方だけを要求すると「埋め込みまで i4 に落ちた系列」「linear が i8 のままの系列」が
#: それぞれ素通りする — どちらも shape も manifest も正しいまま、品質と速度だけが変わる。
GEMMA4_STORAGE_REQUIREMENTS: Mapping[str, str] = {GEMMA4_ROLE: "I4"}
GEMMA4_STORAGE_ALSO_REQUIRED: Mapping[str, str] = {GEMMA4_ROLE: "I8"}

#: 各役割の safetensors ヘッダに**あってはならない**格納 dtype（{@link assert_storage_absent}）。
#: 台本が焼く格納形は i8 + i4 + f32 の 1 系列だけなので、F16 の混入は「別 family の系列 root を
#: 指した」印にしかならない（系列 root の取り違えは数値の門では原理的に検出できない —
#: ADR 0027 / 0029。他 family と同じ規律で、書き出しうる圧縮格納のうち**在ってはならない側を
#: 全部**名指しする）。
GEMMA4_STORAGE_FORBIDDEN: Mapping[str, tuple[str, ...]] = {GEMMA4_ROLE: ("F16",)}

#: 格納 dtype のラベル（quant 席の綴りでもある）。**基底格納は 1 つ**なので
#: {@link complete_quant_weights} の自動補完が quant 表の weights を埋める。
GEMMA4_DTYPE = "i4"

#: weights の宣言（dtype ラベル → 役割名）。分割は現物が決めるので、ここが指すのは代表 1 役。
GEMMA4_WEIGHTS: Mapping[str, Mapping[str, WeightFiles]] = {
    GEMMA4_ROLE: {GEMMA4_DTYPE: WeightFiles(GEMMA4_ROLE)}
}

#: quant 席（ADR 0074 の文法 `<格納>[+<部品><ビット>]…[-<ノブ>]…`）。1 席だけなのは格納系列が
#: 1 本しか無いため。`session` は空 — `Gemma4Pipeline` は Session の実行形ノブを結線していない
#: （宣言だけ足すと「名前だけの席」になる）。
GEMMA4_QUANTS: Mapping[str, Any] = {
    GEMMA4_DTYPE: {
        "weights": {},
        "session": {},
        "label": "Packed int4 linear, int8 embeddings",
        "description": "The only storage series: linear weights in packed int4 (group 32) and the"
        " embedding tables in int8, which are not int4-eligible.",
    }
}

GEMMA4_DEFAULT_QUANT = GEMMA4_DTYPE

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

#: PLE sidecar の索引の版と欄（読み手 `packages/models/src/gemma/ple.ts` の
#: `SCHEMA` / `INDEX_KEYS` / `SHARD_KEYS` の鏡像 — 焼く側が先に落とす）。
GEMMA4_PLE_SCHEMA = 1
GEMMA4_PLE_INDEX_KEYS: tuple[str, ...] = (
    "schema",
    "tokens",
    "layers",
    "dim",
    "embedScale",
    "shards",
)
GEMMA4_PLE_SHARD_KEYS: tuple[str, ...] = ("file", "start", "stop")

#: sidecar shard のテンソルキーと `__metadata__` の席（`export_product.py` の同名定数）。
GEMMA4_PLE_VALUES_KEY = "values"
GEMMA4_PLE_SCALES_KEY = "scales"
GEMMA4_PLE_METADATA_KEY = "karume_ple"


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


def gemma4_repo_name(model: str) -> str:
    """単一モデルの配布リポ名（`karume-` prefix はリポ名裁定 2026-08-09）。"""
    return f"karume-{GEMMA4_PREFIX}-{model}"


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
    tokenizer: Path
    model: Path


def gemma4_sources(series_dir: Path, model: str = GEMMA4_DEFAULT_MODEL) -> Gemma4Sources:
    """系列の親ディレクトリ（`outputs/series/`）と `_shared.paths` の綴りから入力を引く。"""
    return Gemma4Sources(
        product=series_dir / gemma4_series_name(model, GEMMA4_PRODUCT_SUFFIX),
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


def gemma4_ple_index(product: Path) -> Mapping[str, Any]:
    """PLE sidecar の索引を読んで**形まで**落とす（読み手 `ple.ts` の受理集合の鏡像）。

    MUST: 範囲は `[0, tokens)` の**隙間も重なりも無い昇順分割**であること。緩めると「引けない
    id がある索引」や「2 本が同じ id を持つ索引」が通り、後者は**どちらの行を引いたか**で
    結果が変わる（例外の出ない沈黙誤値）。読み手も同じ検査を持つが、配ってから利用者の手元で
    初めて落ちる形にしない。
    """
    path = product / GEMMA4_PLE_INDEX_FILE
    where = str(path)
    raw = _read_json(path, "PLE sidecar の索引")
    if not isinstance(raw, dict):
        raise DistError(f"{where}: 最上位オブジェクトでない")
    unknown = sorted(set(raw) - set(GEMMA4_PLE_INDEX_KEYS))
    if unknown:
        raise DistError(f"{where}: 未知キー {unknown}（許可: {list(GEMMA4_PLE_INDEX_KEYS)}）")
    if raw.get("schema") != GEMMA4_PLE_SCHEMA:
        raise DistError(f"{where}: schema が {raw.get('schema')!r}（期待 {GEMMA4_PLE_SCHEMA}）")
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
    shards = raw.get("shards")
    if not isinstance(shards, list) or not shards:
        raise DistError(f"{where}: shards が非空の配列でない")
    seen: set[str] = set()
    expected = 0
    for position, entry in enumerate(shards):
        at = f"{where} の shards[{position}]"
        if not isinstance(entry, dict):
            raise DistError(f"{at}: オブジェクトでない")
        extra = sorted(set(entry) - set(GEMMA4_PLE_SHARD_KEYS))
        if extra:
            raise DistError(f"{at}: 未知キー {extra}（許可: {list(GEMMA4_PLE_SHARD_KEYS)}）")
        file = entry.get("file")
        if not isinstance(file, str) or not file:
            raise DistError(f"{at}: file が非空の文字列でない（{file!r}）")
        if file in seen:
            raise DistError(f"{at}: file '{file}' が重複している")
        seen.add(file)
        start = _offset(entry, "start", at)
        stop = _offset(entry, "stop", at)
        if start != expected:
            raise DistError(f"{at}: start {start} が直前の shard の末尾 {expected} と連続しない")
        if stop <= start:
            raise DistError(f"{at}: 範囲 [{start}, {stop}) が空")
        expected = stop
    if expected != tokens:
        raise DistError(f"{where}: shard の合計 {expected} 行が tokens {tokens} と違う")
    return {"tokens": tokens, "layers": layers, "dim": dim, "embedScale": scale, "shards": shards}


def gemma4_ple_role(position: int) -> str:
    """PLE shard 1 本ぶんの役割名（索引の並び順 = 1 始まりの番号）。"""
    return f"{GEMMA4_PLE_ROLE_PREFIX}{position + 1}"


def gemma4_output_paths(index: Mapping[str, Any]) -> dict[str, str]:
    """役割名 → 配布形の相対 path（**モデルサブツリー内**）。

    PLE shard の相対 path は索引が書いたファイル名をそのまま使う — asset 名（= 取得キー）も
    同じ綴りなので、読み手は `ple.json` の `shards[].file` 1 本で取得キーも path も引ける。
    """
    paths = dict(GEMMA4_OUTPUT_PATHS)
    for position, shard in enumerate(index["shards"]):
        paths[gemma4_ple_role(position)] = f"{GEMMA4_PLE_DIR}/{shard['file']}"
    return paths


def gemma4_placements(sources: Gemma4Sources, index: Mapping[str, Any]) -> dict[str, Path]:
    """役割名 → 出所のファイル。出力の path は {@link gemma4_output_paths} が持つ。

    この表に無いものは出力へ入らない（製品系列に同居する `ple.probe.safetensors` と
    `reference.json` はこれで落ちる — どちらも検収と出所記録のためのもので実行に要らない）。
    """
    placements = {
        GEMMA4_ROLE: sources.product / GEMMA4_MODEL_FILE,
        GEMMA4_TOKENIZER_ROLE: sources.tokenizer / GEMMA4_TOKENIZER_FILE,
        GEMMA4_PLE_INDEX_ROLE: sources.product / GEMMA4_PLE_INDEX_FILE,
    }
    for position, shard in enumerate(index["shards"]):
        placements[gemma4_ple_role(position)] = sources.product / str(shard["file"])
    return placements


def gemma4_assets(index: Mapping[str, Any]) -> dict[str, str]:
    """assets の宣言（asset 名 → 役割名）。

    MUST: PLE shard の asset 名は**索引が書いたファイル名そのもの**。読み手は
    `readPleShard(shard.file)` で引くので、ここに別の綴り（連番や意味名）を挟むと索引と
    取得キーの対応が「並び順で合わせる」形になり、片方だけ並べ替えた組が黙って通る。
    """
    assets = {
        GEMMA4_TOKENIZER_ROLE: GEMMA4_TOKENIZER_ROLE,
        GEMMA4_PLE_INDEX_ROLE: GEMMA4_PLE_INDEX_ROLE,
    }
    for position, shard in enumerate(index["shards"]):
        assets[str(shard["file"])] = gemma4_ple_role(position)
    return assets


def gemma4_vocab_size(graph: Mapping[str, Any], path: Path) -> int:
    """最終行 logits 出口の語彙数をグラフの出力宣言から読む（`[1, 1, V]` — ADR 0083 決定 6）。

    出力が 1 本であることまで見るのは、検収用の 2 系列（logits opt-in / token-only）が同じ
    系列名の下に紛れ込むと**幅だけが別の意味の数**になるため。V は主 embedding の行数そのもの
    で、PLE sidecar とトークナイザの相互照合（ADR 0085 決定 5）の基準になる。
    """
    outputs = graph.get("outputs")
    if not isinstance(outputs, list) or len(outputs) != 1:
        raise DistError(
            f"{path}: グラフ出力が {outputs!r} — 製品グラフの出口は最終行 logits の 1 本だけ"
        )
    values = graph.get("values")
    value = values.get(outputs[0]) if isinstance(values, dict) else None
    shape = value.get("shape") if isinstance(value, dict) else None
    if not isinstance(shape, list) or len(shape) != 3 or shape[0] != 1 or shape[1] != 1:
        raise DistError(f"{path}: グラフ出力 '{outputs[0]}' の形が [1, 1, V] でない（{shape!r}）")
    vocab = shape[2]
    if not isinstance(vocab, int) or isinstance(vocab, bool) or vocab <= 0:
        raise DistError(f"{path}: グラフ出力の語彙数が正の整数でない（{vocab!r}）")
    return vocab


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
) -> None:
    """グラフ入力の並び・形・記号の割れ方を、配置の前に実測する。

    MUST: PLE の層数と層当たり次元は**グラフ入力の宣言**と**索引**の両方が持つ（前者は
    `per_layer_inputs[1, M, 35, 256]`・後者は `layers` / `dim`）。食い違ったまま配ると、
    ホストが組む表と GPU が読む形が別物になる — shape が合う組み合わせでは沈黙する。

    MUST: RoPE 派生入力の幅は `pipelineConfig.rope` の `headDim` と一致すること。宣言と
    グラフは別々の正本（config / コンテナ）から来るので、噛み合わせはここでしか見られない
    — ホストが宣言どおりに組んだ表が入力の幅と違えば、実行時まで誰も気づけない。

    MUST: 表の initializer が 1 本も残っていないこと。派生入力を足したのに表も残っている形は
    常駐が戻るうえ、位置の上限が資産側へ逆戻りする。

    MUST: 記号は 2 本で、**入力 shape から決まらないもの**がちょうど 1 本（full スロットの
    容量記号）。TS 側 `Gemma4Pipeline` はこの 1 本を容量の束縛点にするので、割れ方が変わると
    ロード時に落ちる（`capacitySymbolOf` の同じ検査）。
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
    baked = sorted(
        key
        for key, entry in (graph.get("initializers") or {}).items()
        if isinstance(entry, dict) and BAKED_TABLE_INFIX in str(entry.get("tensor", ""))
    )
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
                f"{GEMMA4_PLE_INDEX_FILE} の {field} は {index[field]}"
                " — グラフと PLE sidecar が別世代"
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


def assert_gemma4_ple_shards(placements: Mapping[str, Path], index: Mapping[str, Any]) -> None:
    """sidecar shard の現物が索引と同じ資産世代を名乗ることを、配置の前に見る。

    MUST: 範囲まで突き合わせる（読み手 `ple.ts` の `assertShardMetadata` と同じ規律）— 索引
    だけ差し替えた組み合わせは**形も dtype も合う**まま別 token の行を引く。ヘッダしか読まない
    ので 2.4GiB の再読みにはならない。
    """
    for position, shard in enumerate(index["shards"]):
        path = placements[gemma4_ple_role(position)]
        # 実在検査を先に置く（`assert_plan_sources` は組み立て側の門で、こちらの計画段では
        # まだ走っていない）— 素の OSError で落ちると「何を焼き直せばよいか」が伝わらない。
        if not path.is_file():
            raise DistError(f"組み立ての入力が無い: {path}（{GEMMA4_PLE_INDEX_FILE} が名指し）")
        header = safetensors_header(path)
        rows = int(shard["stop"]) - int(shard["start"])
        for key, dtype, shape in (
            (GEMMA4_PLE_VALUES_KEY, "I8", [rows, index["layers"], index["dim"]]),
            (GEMMA4_PLE_SCALES_KEY, "F32", [rows, index["layers"]]),
        ):
            spec = header.get(key)
            if not isinstance(spec, dict):
                raise DistError(f"{path}: テンソル '{key}' が無い（別形式の資産）")
            if spec.get("dtype") != dtype:
                raise DistError(
                    f"{path}: '{key}' の格納 dtype が {spec.get('dtype')!r}（{dtype} でない）"
                )
            if spec.get("shape") != shape:
                raise DistError(
                    f"{path}: '{key}' の形が {spec.get('shape')!r}（索引から組んだ期待は {shape}）"
                )
        metadata = header.get("__metadata__")
        raw = metadata.get(GEMMA4_PLE_METADATA_KEY) if isinstance(metadata, dict) else None
        if not isinstance(raw, str):
            raise DistError(
                f"{path}: __metadata__.{GEMMA4_PLE_METADATA_KEY} が無い（別形式の資産）"
            )
        declared = json.loads(raw)
        if not isinstance(declared, dict):
            raise DistError(f"{path}: {GEMMA4_PLE_METADATA_KEY} が最上位オブジェクトでない")
        expected = {
            "schema": GEMMA4_PLE_SCHEMA,
            "tokens": index["tokens"],
            "layers": index["layers"],
            "dim": index["dim"],
            "embedScale": index["embedScale"],
            "start": shard["start"],
            "stop": shard["stop"],
        }
        wrong = [
            f"{key} {declared.get(key)!r} ≠ {want!r}"
            for key, want in expected.items()
            if declared.get(key) != want
        ]
        if wrong:
            raise DistError(
                f"{path}: {GEMMA4_PLE_METADATA_KEY} が索引と食い違う（{' / '.join(wrong)}）"
                " — 片方だけ作り直した組み合わせ"
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
    max_position: int, rope: Mapping[str, Any], sampler: Mapping[str, Any]
) -> dict[str, Any]:
    """`pipelineConfig`（TS 側スキーマの 6 欄）を組む。

    MUST: 実行時ノブが**両側の上限の内側**に収まることをここで落とす。chunk の行数は記号 `M`
    の trace 時の上限（{@link GEMMA4_MAX_CHUNK_LENGTH}）を超えられず、`capacity` は会話が
    使える最大の論理長なので位置は最大 `capacity - 1` まで進む — モデルの宣言
    （`maxPosition`）を超える容量は「宣言の内側なのに上流が想定していない位置を回す」形で、
    長い会話でだけ表面化する。

    MUST: その chunk の上限を `maxChunkLength` として**宣言にも載せる**。ここの検査が見るのは
    配布形が焼く既定値だけで、`chunkLength` は読み手の実行時ノブでもある — 宣言が無いと
    「trace 範囲の外の chunk 長で走る」形を読み手側で落とせない（IR の `symbols` は名前の列
    だけで上限を持たない）。

    NOTE: 上限の側は**写しの同値しか見ていない**。比較相手 {@link GEMMA4_MAX_CHUNK_LENGTH} は
    `gemma4.export.SYM_MAX` を写した定数で、系列を組んだときに実際に使われた `--sym-max` は
    どこにも記録されていない（資産からも読めない）。小さい `--sym-max` で trace した容器に対して
    `chunkLength: 768` を名乗る配布形は、この検査も同値テストも素通りする。
    """
    if not 2 <= GEMMA4_CHUNK_LENGTH <= GEMMA4_MAX_CHUNK_LENGTH:
        raise DistError(
            f"chunkLength {GEMMA4_CHUNK_LENGTH} が [2, {GEMMA4_MAX_CHUNK_LENGTH}] の外"
            "（下限は記号 M の下限・上限は trace 時の `Dim` の上限）"
        )
    if GEMMA4_CAPACITY < GEMMA4_CHUNK_LENGTH:
        raise DistError(
            f"capacity {GEMMA4_CAPACITY} が chunkLength {GEMMA4_CHUNK_LENGTH} より小さい"
            " — 1 chunk すら入らない容量は宣言できない"
        )
    if max_position < GEMMA4_CAPACITY:
        raise DistError(
            f"capacity {GEMMA4_CAPACITY} がモデルの位置上限 {max_position} を超えた"
            " — 容量いっぱいの会話が宣言の外の位置を回す"
        )
    return {
        "chunkLength": GEMMA4_CHUNK_LENGTH,
        "maxChunkLength": GEMMA4_MAX_CHUNK_LENGTH,
        "maxPosition": max_position,
        "capacity": GEMMA4_CAPACITY,
        "rope": {layer_type: dict(spec) for layer_type, spec in rope.items()},
        "sampler": dict(sampler),
    }


def gemma4_plan(sources: Gemma4Sources, model: str = GEMMA4_DEFAULT_MODEL) -> ModelPlan:
    """gemma4 1 モデルぶんの計画を組む（検査と読み取りをここで全部済ませる）。"""
    assert_model_name(model)
    index = gemma4_ple_index(sources.product)
    placements = gemma4_placements(sources, index)
    for role, source in placements.items():
        assert_storage(role, source, GEMMA4_STORAGE_REQUIREMENTS)
        assert_storage(role, source, GEMMA4_STORAGE_ALSO_REQUIRED)
        assert_storage_absent(role, source, GEMMA4_STORAGE_FORBIDDEN)
    text_config = gemma4_text_config(sources.model)
    where = str(sources.model / GEMMA4_CONFIG_FILE)
    rope = gemma4_rope(text_config, where)
    container = placements[GEMMA4_ROLE]
    graph = ir_graph(container)
    vocab_size = gemma4_vocab_size(graph, container)
    assert_gemma4_graph(graph, container, index, rope)
    if index["tokens"] != vocab_size:
        raise DistError(
            f"{sources.product / GEMMA4_PLE_INDEX_FILE}: tokens {index['tokens']} が製品グラフの"
            f"語彙数 {vocab_size} と違う — 別の語彙で焼かれた組み合わせ"
        )
    assert_gemma4_ple_shards(placements, index)
    assert_gemma4_tokenizer(placements[GEMMA4_TOKENIZER_ROLE], vocab_size)
    pipeline_config = gemma4_pipeline_config(
        gemma4_max_position(text_config, where), rope, gemma4_sampler(sources.model)
    )
    output_paths = gemma4_output_paths(index)
    return ModelPlan(
        name=model,
        pipeline=GEMMA4_PIPELINE,
        artifacts={
            role: Artifact(output_paths[role], source=source) for role, source in placements.items()
        },
        weights=GEMMA4_WEIGHTS,
        assets=gemma4_assets(index),
        # requiredLimits は書かない — core の dist が組み立て時に一括導出して焼く
        # （karume/limits.py。計画側の手書きは二重管理として拒否される）。
        quants=complete_quant_weights(GEMMA4_WEIGHTS, GEMMA4_QUANTS),
        default_quant=GEMMA4_DEFAULT_QUANT,
        pipeline_config=pipeline_config,
    )


def gemma4_dist_plan(series_dir: Path, model: str) -> ModelPlan:
    """`--series` の親から gemma4 1 モデルの計画を組む（CLI のディスパッチ先）。"""
    return gemma4_plan(gemma4_sources(series_dir, model), model)


#: 上流ライセンスの原文（Apache License 2.0 の逐語コピー）。`Path(__file__)` 基準で引くのは、
#: cwd にも系列の置き場にも依存しないため（anima の `LICENSE_SOURCE_PATH` と同じ規律）。
GEMMA4_LICENSE_PATH = Path(__file__).parent / "apache_license_2_0.txt"

#: 改変告知（Apache 2.0 §4(b)）。**このリポが上流の重みへ加えた変更**を列挙する。
#:
#: MUST: 文面は配布形の中身と対応していること — 値としては妥当な散文なので `verify_dist` も
#: manifest 検査も素通りし、配ってからでないと食い違いに気づけない。
GEMMA4_NOTICE_MARKDOWN = """# NOTICE

This repository redistributes a modified form of `google/gemma-4-E2B-it`, which is licensed under
the Apache License, Version 2.0 (see `LICENSE.md`). The following changes were made:

- The **text decoder only** was extracted; the vision and audio towers were never read.
- The graph was re-expressed in the Karume container format (a safetensors file whose
  `__metadata__` carries the graph) in a states form suited to chunked prefill and decode.
- **Linear weights were quantized** to packed int4 (group 32) and the embedding tables to int8.
  The values are therefore not bit-identical to the source checkpoint.
- The **per-layer embedding tables were moved out of the graph** into a sidecar that the host
  gathers, and the exit was narrowed to the last row's logits.
- **Rotary position embeddings were moved out of the graph**: the cosine and sine rows are built
  by the host from the declared parameters and passed in as ordinary graph inputs.

No retraining and no fine-tuning were performed. The original checkpoint is not distributed here.
"""


def gemma4_root_files() -> dict[str, str]:
    """配布リポ直下へ入れる法的テキスト（`karume.dist.Pipeline.root_files`）。

    上流は Apache License 2.0（`gemma4/README.md` 冒頭のライセンス方針 — 2026-09-01 の裁定）で、
    §4(a) は「派生物の受領者にライセンスのコピーを渡す」ことを、§4(b) は「改変したファイルに
    改変した旨の目立つ告知を付ける」ことを求める。ライセンス原文は recipe に置いた現物
    （{@link GEMMA4_LICENSE_PATH}）を**逐語で**読む — 整形や差し替えをするとコピーではなくなる。
    上流に `NOTICE` ファイルは無い（§4(d) は掛からない）ので、`NOTICE.md` は §4(b) の改変告知
    だけを持つ。
    """
    return {
        "LICENSE.md": GEMMA4_LICENSE_PATH.read_text(encoding="utf-8"),
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
