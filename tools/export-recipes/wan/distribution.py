"""Wan2.1 の配布 recipe — 系列レイアウト・出力 path 表・quant 表・カードの選択（ADR 0118 決定 7）。

汎用の組み立てエンジン（配置・共有席の畳み込み・sha256・manifest・staging/swap・検証）は
`karume.dist` が持つ。ここが持つのは **Wan2.1 固有の事実**だけ: どの系列ディレクトリから何を
拾い、配布形のどの path へ、どの dtype ラベルで並べ、どの quant を既定にするか。

配布するのはグラフ 3 本（DiT の S 形 `transformer`・VAE の chunk グラフ `vae_decoder_first` /
`vae_decoder_next` — 系列 `wan2.1-t2v-1.3b-f16-dyn`）と、モデル単位の資産 `text_embeds`（第 1 段の
テキスト埋め込み — 決定 4。quant 非依存の席・ADR 0109 決定 4）。RoPE の軸別素表 `rope_base` は
`transformer` の**容器の資産**（役割 `rope-base`）なので manifest の `assets` には載らない
（ADR 0109 決定 4 — Anima と同じ席）。

**リポは家族 1 つ・世代は別リポ**（`karume-wan2.1` — ADR 0092 決定 1 / 2）。モデルは世代の中の
軸で、今は `t2v-1.3b` 1 本。quant 席は `f16` だけ（DiT と VAE の重みを f16 格納・活性 f32 —
決定 7。i8 / i4 は品質の実測の後）。

公開面は {@link PIPELINE} 1 つ — リポの dist ドライバ（`tools/export-recipes/dist.py`）がこれを
core の PIPELINES へ合成する。

MUST: このモジュールは torch を import しない（`import dist` が torch を読まない —
`tests/test_dist_driver.py` の `TestImportingTheDriver`）。書き手（`wan.export_dit` /
`wan.export_vae` / `wan.text_embeds`）は torch を読むので、綴り（系列名・資産名・メタのキー・
グラフ入力名）はここに置き、書き手の綴りとの一致は `wan/tests/test_distribution.py` の門が見る。
"""

from __future__ import annotations

import json
import struct
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from _shared.container_read import ContainerReadError, read_asset_declarations
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
from wan.card import render_wan_model_card
from wan.prompts import FIXED_PROMPTS
from wan.sources import DEFAULT_MODEL, SOURCES

#: パイプライン契約（ADR 0041 §2 — モデル単位）。TS 側の受理集合は `WAN_PIPELINE_NAME` /
#: `WAN_PIPELINE_MAJOR`（`packages/models/src/wan/config.ts`）。
WAN_PIPELINE = "wan/1"

#: 配布リポ名（ADR 0092 決定 1 / 2 — 家族 1 リポ・世代は別リポ。Wan2.2 は `karume-wan2.2`）。
WAN_REPO_NAME = "karume-wan2.1"

#: DiT と VAE の系列（決定 7 — 接尾辞 `-dyn` は ADR 0077 の慣例）。書き手の綴りは
#: `wan.export_dit.SERIES` / `wan.export_vae.SERIES_NAME`。
WAN_SERIES = "wan2.1-t2v-1.3b-f16-dyn"

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
#: `_get_t5_prompt_embeds` / `prompt_clean`、ftfy は正規化の規則を持つ。値は pyproject の `wan`
#: グループの `==` ピンと同じ（`wan/tests/test_distribution.py` の門が両方を突き合わせる）。
WAN_TEXT_EMBEDS_VERSIONS: Mapping[str, str] = {"diffusers": "0.39.0", "ftfy": "6.3.1"}

#: 系列の部品ディレクトリに置かれる容器の代表名（分割形は `model-0000N-of-0000M.krm`）。
WAN_MODEL_FILE = "model.krm"

#: グラフを持つ部品名 = manifest の weights のキー = **容器のグラフ名**（container-v1 §2.1 —
#: ランタイムは `prepareContainer(opened, <weights キー>)` で名前で引く）。系列の部品
#: ディレクトリ名も同じ綴りだが、それは規約であって導出ではない（書き手は `TARGET` /
#: `TARGETS` を名乗る — `tests/test_graph_names.py` の門）。
WAN_TRANSFORMER_ROLE = "transformer"
WAN_VAE_FIRST_ROLE = "vae_decoder_first"
WAN_VAE_NEXT_ROLE = "vae_decoder_next"
WAN_GRAPH_ROLES: tuple[str, ...] = (WAN_TRANSFORMER_ROLE, WAN_VAE_FIRST_ROLE, WAN_VAE_NEXT_ROLE)

#: モデル単位の資産（manifest の `assets` のキー = 役割名 — TS 側 `pipeline.ts` の `TEXT_EMBEDS`）。
WAN_TEXT_EMBEDS_ROLE = "text_embeds"

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
#: 格納ラベル（`f16`）を入れるのは、i8 / i4 の席を足した日に既存の path を動かさないため
#: （Depth Anything の `model.f32.krm` と同じ綴り）。
WAN_OUTPUT_PATHS: Mapping[str, str] = {
    WAN_TRANSFORMER_ROLE: f"{WAN_TRANSFORMER_ROLE}/model.f16.krm",
    WAN_VAE_FIRST_ROLE: f"{WAN_VAE_FIRST_ROLE}/model.f16.krm",
    WAN_VAE_NEXT_ROLE: f"{WAN_VAE_NEXT_ROLE}/model.f16.krm",
    WAN_TEXT_EMBEDS_ROLE: f"{WAN_TEXT_EMBEDS_ROLE}/{WAN_TEXT_EMBEDS_FILE}",
}

#: 格納 dtype の要求（素の f32 資産が組み立て・ロード・実行を全て通って参照一致の門まで沈黙した
#: 実測事故 — Anima / SBV2 / Depth Anything と同じ根拠）。f16 系列は fake-quant 対象だけが f16 に
#: なる（bias / norm は f32 のまま）ので「f16 を含む」を要求する。
WAN_STORAGE_REQUIREMENTS: Mapping[str, str] = {role: "f16" for role in WAN_GRAPH_ROLES}

#: 各役割の束縛表に**あってはならない**格納の語彙（{@link assert_storage_absent}）。存在検査だけ
#: では、f16 を含む混成の圧縮系列（i8 / i4 の系列も適格外の重みを f16 では持たないが、別 family の
#: f16 + i8 混成系列はありうる）を f16 席へ挿す取り違えが素通りする。
#:
#: MUST: codec 台帳の layout（`karume.container.CODEC_LEDGER`）から f32 / f16 / i32 を除いた
#: **全部**を名指しする — 1 つでも抜けると、抜けた格納形だけが黙って素通りする
#: （Depth Anything と同じ規律）。
WAN_STORAGE_FORBIDDEN: Mapping[str, tuple[str, ...]] = {
    role: ("bf16", "i8", "i4", "i2") for role in WAN_GRAPH_ROLES
}

#: weights の宣言（dtype ラベル → 役割名）。ラベルは格納 dtype 語彙で、
#: {@link WAN_STORAGE_REQUIREMENTS} が要求する格納形と 1:1（ADR 0041 §3）。
WAN_WEIGHTS: Mapping[str, Mapping[str, WeightFiles]] = {
    role: {"f16": WeightFiles(role)} for role in WAN_GRAPH_ROLES
}

#: assets の宣言（quant 選択に依存しない無条件ファイル — ADR 0041 §3・決定 4）。
WAN_ASSETS: Mapping[str, str] = {WAN_TEXT_EMBEDS_ROLE: WAN_TEXT_EMBEDS_ROLE}

#: quant 表。格納ラベルが 1 つしかないので weights は書かない（{@link complete_quant_weights} が
#: 完全写像へ埋める）。`session` は空 — TS 側の受理表（`WAN_SESSION_POLICY`）はどのノブも受けない。
#: `label` / `description` は選択 UI 向けの表示欄（ADR 0075 決定 1 — 英語・64 / 200 字上限）。
WAN_QUANTS: Mapping[str, Any] = {
    "f16": {
        "weights": {},
        "session": {},
        "label": "Full quality (f16)",
        "description": "Transformer and VAE weights in f16 storage with f32 compute — the only"
        " quant in this repository.",
    },
}
WAN_DEFAULT_QUANT = "f16"

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
    """組み立ての入力 = 系列ディレクトリ 2 本（グラフ 3 本の系列とテキスト埋め込みの系列）。"""

    series: Path
    text_embeds: Path


def wan_sources(series_dir: Path) -> WanSources:
    """系列の親ディレクトリ（`outputs/series/`）から入力を引く。"""
    return WanSources(
        series=series_dir / WAN_SERIES,
        text_embeds=series_dir / WAN_TEXT_EMBEDS_SERIES / WAN_TEXT_EMBEDS_FILE,
    )


def wan_placements(sources: WanSources) -> dict[str, Path]:
    """役割名 → 出所のファイル。出力の path は {@link WAN_OUTPUT_PATHS} が持つ。

    この表に無いものは出力へ入らない（系列に並ぶ `io.*` / `reference.*` / `vae_*.safetensors` /
    `pipeline_steps.*` の golden はこれで落ちる）。
    """
    placements = {role: sources.series / role / WAN_MODEL_FILE for role in WAN_GRAPH_ROLES}
    placements[WAN_TEXT_EMBEDS_ROLE] = sources.text_embeds
    return placements


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


def wan_plan(sources: WanSources, model: str = DEFAULT_MODEL) -> ModelPlan:
    """Wan2.1 の 1 モデルぶんの計画を組む（検査と読み取りをここで全部済ませる — 何も書かない）。"""
    assert_model_name(model)
    if model not in SOURCES:
        raise DistError(
            f"Wan2.1 のモデル {model!r} は知らない（既知: {' / '.join(sorted(SOURCES))}）—"
            " 上流の取得元の表（wan.sources.SOURCES）に載ったモデルだけを配る"
        )
    placements = wan_placements(sources)
    upstream = SOURCES[model]
    for role in WAN_GRAPH_ROLES:
        container = placements[role]
        assert_component_present(container)
        assert_storage(role, container, WAN_STORAGE_REQUIREMENTS)
        assert_storage_absent(role, container, WAN_STORAGE_FORBIDDEN)
        # 容器が名乗る出所を上流の pin（`wan.sources` が正本）へ突き合わせる — モデル名の表だけで
        # 門を閉じると、別の revision から焼いた容器が系列 path へ置かれたときに素通りする。
        assert_upstream_provenance(container, license=upstream.license, revision=upstream.revision)
    assert_vae_chunk_pair(placements[WAN_VAE_FIRST_ROLE], placements[WAN_VAE_NEXT_ROLE])
    transformer = placements[WAN_TRANSFORMER_ROLE]
    assert_rope_base(transformer)
    assert_text_embeds(placements[WAN_TEXT_EMBEDS_ROLE], model, dit_context(transformer))
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
- **f16 storage**: every transformer and VAE decoder parameter was rounded from the source float32
  value to the nearest float16 value. Weight matrices and convolution kernels are stored as float16;
  the other parameters (biases, normalization weights) keep the rounded values in float32 storage.
  Computation runs in float32.
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
- **The text encoder is not distributed.** Instead, the umT5-XXL outputs of a fixed set of prompts
  (computed with the upstream encoder in bfloat16 and stored as float32) are included as a
  precomputed asset.

No retraining and no fine-tuning were performed. The original checkpoint is not distributed here.
"""


#: `--pipeline wan` の 1 行（ドライバが core の PIPELINES へ合成する）。
PIPELINE = Pipeline(
    default_model=DEFAULT_MODEL,
    repo_name=wan_repo_name,
    plan=wan_dist_plan,
    # 帰属（上流リポ・ライセンス）はモデル名から一意に決まる（`wan.sources.SOURCES`）ので、
    # 選ばせる軸にしない。
    card_profiles={"wan": render_wan_model_card},
    # 上流ライセンス（Apache 2.0）の再配布条件 §4 は配布リポ 1 つに掛かるので、原文の読みも
    # 組み立ての回数によらずここで 1 回（ADR 0092 決定 7）。
    root_files={
        "LICENSE.md": apache_license_2_0(),
        "NOTICE.md": WAN_NOTICE_MARKDOWN,
    },
)
