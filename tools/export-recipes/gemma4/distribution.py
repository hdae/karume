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

`pipelineConfig` は 2 系統に割れる（Irodori と同じ分け方）: **資産世代ごとに動く数**
（`maxPosition` = 焼き込んだ RoPE 表の行数）はコンテナから導出し、**実行時ノブ**
（`chunkLength` / `capacity`）と**配布者の推奨サンプラ**（上流 `generation_config.json` の
temperature / top_k / top_p — ADR 0083 決定 7）はそれぞれの正本から引く。前者を写経すると
「表は 1024 行なのにホストだけ 4096 と思っている」形で、RoPE の外を引いた瞬間に落ちる。

公開面は {@link PIPELINE} 1 つ（`karume.dist.Pipeline`）— リポの dist ドライバ
（`tools/export-recipes/dist.py`）がこれを core の PIPELINES へ合成する。
"""

from __future__ import annotations

import json
import math
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
    assert_storage_absent,
    complete_quant_weights,
    graph_inputs,
    ir_graph,
    safetensors_header,
)

from .card import GEMMA4_UPSTREAM, render_gemma4_model_card

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

#: 上流チェックポイントが持つ推奨サンプラの出どころ（ADR 0083 決定 7）。
GEMMA4_GENERATION_CONFIG_FILE = "generation_config.json"

#: 役割名（manifest の weights / assets が指す内部キー）。
GEMMA4_ROLE = "model"
GEMMA4_TOKENIZER_ROLE = "tokenizer"
GEMMA4_PLE_INDEX_ROLE = "ple_index"
GEMMA4_PLE_ROLE_PREFIX = "ple_"

#: 配布形の中の PLE sidecar の置き場（モデルサブツリー内）。
GEMMA4_PLE_DIR = "ple"

#: グラフ入力の名前と並び（正本は `gemma4/export_product.py` — ラッパの forward 引数名）。
#: 実行側は名前で束ねるので、1 つでも綴りが変われば束ねられない。
GEMMA4_INPUT_IDS = "input_ids"
GEMMA4_POSITION_IDS = "position_ids"
GEMMA4_PER_LAYER_INPUTS = "per_layer_inputs"
GEMMA4_LAST_ROW = "last_row"
GEMMA4_GRAPH_INPUTS: tuple[str, ...] = (
    GEMMA4_INPUT_IDS,
    GEMMA4_POSITION_IDS,
    GEMMA4_PER_LAYER_INPUTS,
    GEMMA4_LAST_ROW,
)

#: RoPE 表の initializer テンソル名の接頭と接尾（`gemma4.export_decode.rope_table_keys` の綴り
#: — `<ROPE_TABLE_MODULE>.<layer_type>_<part>_table`）。**行数がそのまま `maxPosition`** で、
#: 4 本（full / sliding × cos / sin）が同じ行数を名乗ることまで見る。
GEMMA4_ROPE_TABLE_INFIX = "rotary_emb."
GEMMA4_ROPE_TABLE_SUFFIX = "_table"

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
#: 資産からは導出できない。値は検収門が通した組み合わせそのもの
#: （`packages/models/tests/e2e_gemma4_chat_test.ts` の `CHUNK_LENGTH`）で、chunk 長を動かすと
#: prefill の刻みが変わって token 列も動きうるため、golden を採った値をそのまま宣言する。
GEMMA4_CHUNK_LENGTH = 32

#: full スロットの容量（会話が使える最大の論理長）。同じく実行時ノブ。上限は
#: {@link gemma4_pipeline_config} が RoPE 表の行数で押さえる。
#:
#: NOTE: **VRAM と会話長のトレードオフの政策値**で、資産は `maxPosition`（現行 1024）まで
#: 引ける。2026-09-02 に 640 → 1024（= 表の上限）へ引き上げた（裁定「コンテキスト窓は可能な
#: 限り大きく」）。fromAssets 側の検収門（`e2e_gemma4_chat_test.ts` ほか）は 640 のままで、
#: 宣言どおりの組み合わせは配布形を読む `e2e_gemma4_pretrained_test.ts` /
#: `e2e_gemma4_directory_test.ts` が通す。golden は full スロットが `pastLength` 行しか読まない
#: ため不変（同 2 門で実測）。さらに上げるには RoPE 表の再 export（`--positions`）が要る。
GEMMA4_CAPACITY = 1024

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


def gemma4_max_position(graph: Mapping[str, Any], path: Path) -> int:
    """焼き込んだ RoPE 表の行数（= `pipelineConfig.maxPosition`）をコンテナから導く。

    MUST: 写経しない。`--positions` は台本の引数なので、宣言と焼かれた表は独立に動く —
    ずれると「宣言の内側なのに表の外を引く」形になり、実行時まで誰も気づけない。

    4 本（full / sliding × cos / sin）が**同じ行数**であることまで見るのは、層種ごとに別の表を
    焼く形（`export_decode.rope_table_keys`）だから — 片方だけ古い表が残ると、その層種だけが
    静かに別の角度で回る。
    """
    initializers = graph.get("initializers")
    values = graph.get("values")
    if not isinstance(initializers, dict) or not isinstance(values, dict):
        raise DistError(f"{path}: IR メタデータに initializers / values が無い")
    rows: dict[str, int] = {}
    for key, entry in initializers.items():
        tensor = entry.get("tensor") if isinstance(entry, dict) else None
        if not isinstance(tensor, str):
            continue
        if GEMMA4_ROPE_TABLE_INFIX not in tensor or not tensor.endswith(GEMMA4_ROPE_TABLE_SUFFIX):
            continue
        value = values.get(key)
        shape = value.get("shape") if isinstance(value, dict) else None
        if not isinstance(shape, list) or len(shape) != 2 or not isinstance(shape[0], int):
            raise DistError(f"{path}: RoPE 表 '{tensor}' の形が [位置数, 幅] でない（{shape!r}）")
        rows[tensor] = shape[0]
    if not rows:
        raise DistError(
            f"{path}: RoPE 表の initializer が 1 本も無い"
            f"（'…{GEMMA4_ROPE_TABLE_INFIX}<層種>_<cos|sin>{GEMMA4_ROPE_TABLE_SUFFIX}'）"
            " — 位置表を持たないグラフでは maxPosition を導けない"
        )
    if len(set(rows.values())) != 1:
        raise DistError(
            f"{path}: RoPE 表の行数が揃っていない（{dict(sorted(rows.items()))}）"
            " — 層種ごとに別世代の表が焼かれている"
        )
    return next(iter(rows.values()))


def assert_gemma4_graph(graph: Mapping[str, Any], path: Path, index: Mapping[str, Any]) -> None:
    """グラフ入力の並び・形・記号の割れ方を、配置の前に実測する。

    MUST: PLE の層数と層当たり次元は**グラフ入力の宣言**と**索引**の両方が持つ（前者は
    `per_layer_inputs[1, M, 35, 256]`・後者は `layers` / `dim`）。食い違ったまま配ると、
    ホストが組む表と GPU が読む形が別物になる — shape が合う組み合わせでは沈黙する。

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


def gemma4_pipeline_config(max_position: int, sampler: Mapping[str, Any]) -> dict[str, Any]:
    """`pipelineConfig`（TS 側スキーマの 4 欄）を組む。

    MUST: 実行時ノブが**焼かれた表の内側**に収まることをここで落とす。`capacity` は会話が
    使える最大の論理長なので、位置は最大 `capacity - 1` まで進む — RoPE 表の行数を超える宣言は
    「宣言の内側なのに表の外を引く」形になり、長い会話でだけ実行時に落ちる。
    """
    if GEMMA4_CHUNK_LENGTH < 2:
        raise DistError(
            f"chunkLength {GEMMA4_CHUNK_LENGTH} が 2 未満（記号 M の下限は 2 — 台本の `Dim`）"
        )
    if GEMMA4_CAPACITY < GEMMA4_CHUNK_LENGTH:
        raise DistError(
            f"capacity {GEMMA4_CAPACITY} が chunkLength {GEMMA4_CHUNK_LENGTH} より小さい"
            " — 1 chunk すら入らない容量は宣言できない"
        )
    if max_position < GEMMA4_CAPACITY:
        raise DistError(
            f"capacity {GEMMA4_CAPACITY} が焼き込んだ RoPE 表の行数 {max_position} を超えた"
            " — 容量いっぱいの会話が位置表の外を引く"
        )
    return {
        "chunkLength": GEMMA4_CHUNK_LENGTH,
        "maxPosition": max_position,
        "capacity": GEMMA4_CAPACITY,
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
    container = placements[GEMMA4_ROLE]
    graph = ir_graph(container)
    vocab_size = gemma4_vocab_size(graph, container)
    assert_gemma4_graph(graph, container, index)
    if index["tokens"] != vocab_size:
        raise DistError(
            f"{sources.product / GEMMA4_PLE_INDEX_FILE}: tokens {index['tokens']} が製品グラフの"
            f"語彙数 {vocab_size} と違う — 別の語彙で焼かれた組み合わせ"
        )
    assert_gemma4_ple_shards(placements, index)
    assert_gemma4_tokenizer(placements[GEMMA4_TOKENIZER_ROLE], vocab_size)
    pipeline_config = gemma4_pipeline_config(
        gemma4_max_position(graph, container), gemma4_sampler(sources.model)
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
- Rotary position tables were baked as constants for a fixed number of positions.

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
