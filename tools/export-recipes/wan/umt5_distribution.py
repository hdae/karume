"""umT5-XXL encoder の配布 recipe — 配布リポ `karume-umt5-xxl`（ADR 0119 追記「段 10d の設計」D）。

配るのはグラフ 1 本（`text_encoder` — 系列 `umt5-xxl-i8-dyn` の i8 の S 形容器）と、本家の
トークナイザを 1 本の表に焼いた資産 `tokenizer`（系列 `umt5-xxl-tokenizer` — 形式
`karume-umt5-tokenizer/1`・書き手は `wan.umt5_xxl_tokenizer`）。トークナイザを持つので Wan なしで
単体で使える（TS の `@karume/models/umt5` の `Umt5Encoder`）。Wan の前処理の表と束ねた
トークナイザ資産（`karume-wan-umt5-tokenizer/1`）は従来どおり Wan のリポの資産（同 C）。Wan の配布形
`karume-wan2.2` はこのリポの容器を越境参照する（同 A — ADR 0109 決定 3。Wan2.1 は公開しない）。

出所は本家 `google/umt5-xxl` の encoder（`shared.weight` と `encoder.*` — 容器が名乗る出所は
`wan.sources.UMT5_SOURCES` の pin・ADR 0122 決定 1）。Wan2.1 Diffusers の `text_encoder` とは f32 で
全テンソルがビット一致し、Wan2.2 TI2V-5B Diffusers の bf16 はその RNE 丸め（2026-10-04 の
実測）なので、カードと NOTICE はそう書く。

公開面は {@link PIPELINE} 1 つ — リポの dist ドライバ（`tools/export-recipes/dist.py`）がこれを
core の PIPELINES へ合成する（`--pipeline umt5`）。

第三者の互換 encoder の取り込み（`wan.umt5_intake` — ADR 0122 決定 5 / 6）は、ドライバが
`--intake` を受けたときだけ {@link intake_pipeline} で組む**手元の実験用ミラー**で、
`karume-umt5-xxl` を名乗らない（リポ名 {@link intake_repo_name}・既定の出力先は `models/` の外）。
出所の門は記録 `intake.json` と容器の provenance を突き合わせ、ルートのファイル（LICENSE /
NOTICE）とカードは記録から導く — 本家向けの Apache の `LICENSE.md` と「No retraining」の NOTICE は
付けない。ライセンス未宣言の取り込みは明示（`--allow-undeclared-license`）が無ければ計画の時点
（1 バイトも書く前）で落ちる。

MUST: このモジュールは torch を import しない（`import dist` が torch を読まない —
`tests/test_dist_driver.py` の `TestImportingTheDriver`）。書き手（`wan.umt5_export`）は torch を
読むので、綴り（系列名・部品名・入力名・相対位置の表の属性名）はここに置き、書き手の綴りとの一致は
`wan/tests/test_umt5_distribution.py` の門が見る。重みの種類の分類（{@link storage_kind}）は書き手も
ここから引く（分類を 2 本持たない）。
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable, Mapping
from functools import partial
from pathlib import Path
from typing import Any

from _shared.container_read import ContainerReadError, read_layouts
from _shared.licenses import apache_license_2_0, cc_by_sa_4_0
from _shared.upstream import assert_upstream_provenance
from karume.dist import (
    LICENSE_FILENAME,
    NOTICE_FILENAME,
    Artifact,
    DistError,
    ModelPlan,
    Pipeline,
    WeightFiles,
    assert_component_present,
    assert_model_name,
    complete_quant_weights,
    ir_graph,
)
from karume.modelcard import (
    HF_OWNER,
    REVISION_PIN_COMMENT,
    CardMetadata,
    FileRefKey,
    frontmatter,
    model_sections,
    models,
    quants,
    render,
    require_pipeline,
)
from wan import umt5_tokenizer, umt5_xxl_tokenizer
from wan.sources import UMT5_SOURCES, Umt5Source
from wan.umt5_intake import (
    UNDECLARED_LICENSE,
    Umt5Intake,
    Umt5IntakeError,
    assert_license_intent,
)

#: パイプライン契約（ADR 0041 §2 — モデル単位）。読む TS の家族は `@karume/models/umt5` の
#: `Umt5Encoder`（ADR 0119 追記 D の時点では部品の役を名乗るだけだった）。Wan の読み手は越境参照の
#: 先の `karume.json` を読まない。
UMT5_PIPELINE = "umt5-encoder/1"

#: 配布リポ名（ADR 0092 決定 2 の `karume-<family>-<変種>`）。
UMT5_REPO_NAME = "karume-umt5-xxl"

#: モデル名（リポの中の変種の軸 — `karume-irodori-v4-small` の `v4-small` と同じく、リポ名から
#: `karume-<family>-` を落とした綴り）。Wan の越境参照の path はこのサブツリーを指す
#: （`xxl/text_encoder/…`）。
UMT5_DEFAULT_MODEL = "xxl"

#: 系列（書き手の綴りは `wan.umt5_export.SERIES_NAME`）と、その中の容器の代表名
#: （`wan.umt5_export.MODEL_FILE` — 分割形は `model-0000N-of-0000M.krm`）。
UMT5_SERIES = "umt5-xxl-i8-dyn"
UMT5_MODEL_FILE = "model.krm"

#: 部品名 = manifest の weights のキー = **容器のグラフ名**（container-v1 §2.1）= 系列の部品
#: ディレクトリ名。書き手の綴りは `wan.umt5_export.GRAPH_NAME` / `COMPONENT_DIR`。Wan の配布形も
#: 同じキーで越境参照する（`wan.distribution.WAN_TEXT_ENCODER_ROLE`）。
UMT5_ROLE = "text_encoder"

#: トークナイザ資産の役割名 = manifest の `assets` のキー（TS の `Umt5Encoder` が読む名前）。
UMT5_TOKENIZER_ROLE = "tokenizer"

#: 出力の相対 path（**モデルサブツリー内**）。格納ラベルを path に入れるのは Wan の DiT と同じ理由
#: （格納の席を足した日に既存の path を動かさない）。
#:
#: MUST: Wan の配布形の同じ部品の path と同じ綴り（`wan.distribution.WAN_OUTPUT_PATHS` はここを
#: 引く）— 越境参照は参照元の `karume.json` が宣言する `<モデル名>/<この path>` を引き当てるので、
#: 割れると Wan の組み立てが「参照元に無い」で落ちる。
UMT5_OUTPUT_PATHS: Mapping[str, str] = {
    UMT5_ROLE: f"{UMT5_ROLE}/model.i8.krm",
    # 資産は `<役割>/<ファイル名>`（Wan の `umt5_tokenizer/tokenizer.json` と同じ綴りの規則）。
    UMT5_TOKENIZER_ROLE: f"{UMT5_TOKENIZER_ROLE}/{umt5_xxl_tokenizer.ASSET_FILE}",
}

#: weights の宣言（部品 → dtype ラベル → 配置の役割）。i8 の 1 席だけ（ADR 0119 決定 5 — i4 は
#: 品質を測ってから別の席として足す）。
UMT5_WEIGHTS: Mapping[str, Mapping[str, WeightFiles]] = {UMT5_ROLE: {"i8": WeightFiles(UMT5_ROLE)}}

#: assets の宣言（quant 選択に依存しない無条件ファイル — ADR 0041 §3）。本家のトークナイザの表
#: 1 本（前処理の表は持たない — Wan の前処理は Wan のリポの資産が持つ）。
UMT5_ASSETS: Mapping[str, str] = {UMT5_TOKENIZER_ROLE: UMT5_TOKENIZER_ROLE}

#: トークナイザ資産の最上位の欄（書き手 `wan.umt5_xxl_tokenizer.build_asset` の欄 — 前処理の表
#: `promptClean` などの Wan 固有の欄があれば落とす）。
UMT5_TOKENIZER_FIELDS: frozenset[str] = frozenset(
    {
        "format",
        "source",
        "versions",
        "maxLength",
        "unkId",
        "eosId",
        "addedTokens",
        "space",
        "vocabText",
        "scores",
    }
)

#: quant 表（席は 1 つ）。`label` / `description` は選択 UI 向けの表示欄（ADR 0075 決定 1 — 英語）。
UMT5_QUANTS: Mapping[str, Any] = {
    "i8": {
        "weights": {},
        "session": {},
        "label": "int8 weights",
        "description": "Linear and vocabulary-embedding weights stored as int8 (one scale per"
        " output channel or row) and computed in f32; norms and relative-position tables stay f32.",
    }
}
UMT5_DEFAULT_QUANT = "i8"

#: パイプライン所有の設定は無い（読み手が無い）。`karume/5` は空でも `{}` を明示する。
UMT5_PIPELINE_CONFIG: Mapping[str, Any] = {}

#: グラフ入力（書き手の綴りは `wan.umt5_patch.INPUT_NAMES` / `SYMBOL`）— token id `[1, L]` と
#: 相対位置のバケット添字表 `[L, L]`、どちらも i32（ADR 0119 決定 3 / 4）。
UMT5_IDS_INPUT = "input_ids"
UMT5_BUCKETS_INPUT = "relative_position_buckets"
UMT5_SYMBOL = "L"
UMT5_INPUTS: tuple[tuple[str, str, list[Any]], ...] = (
    (UMT5_IDS_INPUT, "i32", [1, UMT5_SYMBOL]),
    (UMT5_BUCKETS_INPUT, "i32", [UMT5_SYMBOL, UMT5_SYMBOL]),
)

#: 相対位置の表の属性名（書き手の綴りは `wan.umt5_patch.RELATIVE_BIAS_ATTRIBUTE`）。
UMT5_RELATIVE_BIAS = "relative_attention_bias"

#: 語彙埋め込みのテンソルキー（tied — checkpoint では `shared.weight`）。
UMT5_EMBED_KEY = "encoder.embed_tokens.weight"

#: 重みの種類 → 要求する格納の layout（ADR 0119 決定 5・段 10c の準備の追記）。`constant` は
#: exporter が持ち上げた定数（`const.` — linear の 0 の bias など）。
#:
#: MUST: 相対位置の表を f32 に縛る — exporter の i8 の既定は `nn.Embedding` も丸める
#: （`QUANT_CHANNEL_AXES`）ので、書き手が表を外し損ねると、ロードも実行も通って attention の
#: バイアスだけが 8 ビットの格子へ黙って丸まる（段 10c の準備の実測では片方の指定だけで落ちた）。
UMT5_KIND_LAYOUTS: Mapping[str, str] = {
    "embed_tokens": "i8",
    "linear": "i8",
    "norm": "f32",
    UMT5_RELATIVE_BIAS: "f32",
    "constant": "f32",
}

#: 束縛表に必ず現れる重みの種類（`constant` は持ち上げの有無で変わるので要求しない）。
UMT5_WEIGHT_KINDS: tuple[str, ...] = ("embed_tokens", "linear", "norm", UMT5_RELATIVE_BIAS)


def storage_kind(key: str) -> str:
    """テンソルキーの種類（相対位置の表・語彙埋め込み・norm・linear・定数）。

    書き手（`wan.umt5_export`）の検収の表と、組み立ての束縛表の門（{@link assert_umt5_bindings}）が
    同じ分類を引く。
    """
    if key.endswith(f".{UMT5_RELATIVE_BIAS}.weight"):
        return UMT5_RELATIVE_BIAS
    if key == UMT5_EMBED_KEY:
        return "embed_tokens"
    if key.endswith("layer_norm.weight"):
        return "norm"
    if key.endswith(".weight"):
        return "linear"
    return "constant"


def umt5_bindings(container: Path) -> dict[str, dict[str, int]]:
    """束縛表を種類 × 格納 layout で数える（**宣言だけ** — 重みの payload は読まない）。"""
    try:
        layouts = read_layouts(container)
    except ContainerReadError as cause:
        raise DistError(f"{container}: {cause}") from cause
    counts: dict[str, Counter[str]] = {}
    for key, layout in layouts.items():
        counts.setdefault(storage_kind(key), Counter())[layout] += 1
    return {kind: dict(sorted(counter.items())) for kind, counter in sorted(counts.items())}


def assert_umt5_bindings(container: Path) -> None:
    """束縛表の格納が種類ごとの要求（{@link UMT5_KIND_LAYOUTS}）どおりであることを見る。

    MUST: 組み立てで落とす。容器の束縛表は `verify_container` の構造検査を通る限り何でも受理され、
    i8 の席の要求（「i8 を含む」）だけでは、表や norm が i8 に丸まった容器も、linear が f32 のまま
    残った容器も素通りする — どちらもロードと実行を通って出力の値だけが静かに変わる。
    """
    bindings = umt5_bindings(container)
    missing = [kind for kind in UMT5_WEIGHT_KINDS if kind not in bindings]
    if missing:
        raise DistError(
            f"{container}: 束縛表に重みの種類 {missing} が無い（実際: {bindings}）— umT5 encoder の"
            " 容器でない"
        )
    wrong = {
        kind: sorted(layouts)
        for kind, layouts in bindings.items()
        if set(layouts) != {UMT5_KIND_LAYOUTS[kind]}
    }
    if wrong:
        expected = {kind: UMT5_KIND_LAYOUTS[kind] for kind in wrong}
        raise DistError(
            f"{container}: 束縛表の格納が種類ごとの要求と違う（実際 {wrong} / 要求 {expected}）—"
            " linear と語彙埋め込みは i8、RMSNorm と相対位置の表は f32（ADR 0119 決定 5）"
        )


def umt5_context_width(container: Path) -> int:
    """グラフの入出力の契約を見て、出力の幅 `W`（`[1, L, W]`）を返す。

    入力は {@link UMT5_INPUTS}（名前・dtype・形がこの並びで一致）、記号は `L` の 1 つだけ、出力は
    f32 の `[1, L, W]` 1 本。

    MUST: 組み立てで落とす。読み手（Wan の text 段）は入力を名前で束ね、出力の行を DiT の文脈へ
    詰めるので、入力名・記号・出力の形が割れた容器は利用者の手元で初めて落ちる。
    """
    graph = ir_graph(container)
    inputs = graph.get("inputs")
    declared = (
        [
            (item.get("name"), item.get("dtype"), item.get("shape"))
            for item in inputs
            if isinstance(item, dict)
        ]
        if isinstance(inputs, list)
        else inputs
    )
    if declared != list(UMT5_INPUTS):
        raise DistError(
            f"{container}: グラフ入力 {declared!r} が {list(UMT5_INPUTS)} でない"
            " — 読み手は入力を名前で束ねる（token id [1, L] とバケット表 [L, L]・i32）"
        )
    if graph.get("symbols") != [UMT5_SYMBOL]:
        raise DistError(
            f"{container}: 記号次元 {graph.get('symbols')!r} が ['{UMT5_SYMBOL}'] でない"
            " — 有効長 L の 1 つだけを動かす（ADR 0119 決定 4）"
        )
    outputs = graph.get("outputs")
    values = graph.get("values")
    value = (
        values.get(outputs[0])
        if isinstance(outputs, list) and len(outputs) == 1 and isinstance(values, dict)
        else None
    )
    shape = value.get("shape") if isinstance(value, dict) else None
    if (
        not isinstance(value, dict)
        or value.get("dtype") != "f32"
        or not isinstance(shape, list)
        or len(shape) != 3
        or shape[:2] != [1, UMT5_SYMBOL]
        or not isinstance(shape[2], int)
        or isinstance(shape[2], bool)
        or shape[2] <= 0
    ):
        raise DistError(
            f"{container}: グラフ出力 {outputs!r}（{value!r}）が f32 の [1, {UMT5_SYMBOL}, W]"
            " 1 本でない"
        )
    return shape[2]


def assert_umt5_encoder(container: Path, model: str, intake: Umt5Intake | None = None) -> int:
    """umT5 の容器の門（出所・束縛表・入出力の契約）を全部掛け、出力の幅を返す。

    Wan の配布形も同じ容器を越境参照する部品として持つので、両方の計画がこの 1 本を通る（Wan の
    計画は格納の要求表で別に i8 の席も見る）。出所の照合先は、本家の行（`wan.sources.
    UMT5_SOURCES`）か、取り込みの記録 `intake`（本家の行に無いモデル — ADR 0122 決定 5）。
    """
    if intake is None:
        upstream = UMT5_SOURCES.get(model)
        if upstream is None:
            raise DistError(
                f"umT5 のモデル {model!r} は知らない（既知: {' / '.join(sorted(UMT5_SOURCES))}）—"
                " 上流の出所の表（wan.sources.UMT5_SOURCES）に載ったモデルだけを配る（第三者の"
                " 取り込みは --intake で手元の実験用ミラーにだけ組める）"
            )
        license_id, revision = upstream.source.license, upstream.source.revision
    else:
        if model != intake.name:
            raise DistError(f"モデル {model!r} が取り込みの名前 {intake.name!r} と違う")
        license_id, revision = intake.license, intake.revision
    assert_component_present(container)
    # 容器が名乗る出所を上流の pin（本家の表か取り込みの記録が正本）へ突き合わせる — 束縛表と
    # 入出力の形は同じ構造の別の checkpoint でも通るので、出所でしか閉じられない。
    assert_upstream_provenance(container, license=license_id, revision=revision)
    assert_umt5_bindings(container)
    return umt5_context_width(container)


def umt5_container(series_dir: Path) -> Path:
    """系列の親ディレクトリ（`outputs/series/`）から容器の代表 path を引く。"""
    return series_dir / UMT5_SERIES / UMT5_ROLE / UMT5_MODEL_FILE


def umt5_tokenizer_asset(series_dir: Path) -> Path:
    """系列の親ディレクトリ（`outputs/series/`）からトークナイザ資産の path を引く。"""
    return series_dir / umt5_xxl_tokenizer.SERIES_NAME / umt5_xxl_tokenizer.ASSET_FILE


def assert_umt5_tokenizer(path: Path, model: str) -> None:
    """トークナイザ資産が、TS の読む形式・上流の pin・id 列の正本の版・受理の上限と噛み合うことを
    見る。

    MUST: 組み立てで落とす。資産は表なので `verify_dist` の容器検査には掛からず、読み手が見るのは
    形式と形だけ — 別の revision のトークナイザ・別の版の transformers で焼いた表は、利用者の手元で
    「別の id 列」として沈黙する。欄の集合も見る（Wan の前処理の表 `promptClean` のような欄が
    混ざった資産は、この形式を名乗っていても配らない）。

    見るのは 5 つ: 形式の版（`wan.umt5_xxl_tokenizer.ASSET_FORMAT`）・欄の集合
    （{@link UMT5_TOKENIZER_FIELDS}）・出所（repo / revision / ファイルの sha256）が
    `wan.sources.UMT5_SOURCES` の行と一致・id 列の正本の transformers の版が pin
    （`wan.umt5_tokenizer.TRANSFORMERS_VERSION`）と一致・`maxLength` が受理の上限
    （{@link UMT5_TOKENS}）と一致。
    """
    assert_component_present(path)
    try:
        asset = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as cause:
        raise DistError(f"{path}: トークナイザ資産が JSON として読めない — {cause}") from cause
    if not isinstance(asset, dict):
        raise DistError(f"{path}: トークナイザ資産がオブジェクトでない")
    if asset.get("format") != umt5_xxl_tokenizer.ASSET_FORMAT:
        raise DistError(
            f"{path}: 形式 {asset.get('format')!r} が {umt5_xxl_tokenizer.ASSET_FORMAT!r} でない"
            " — 本家のトークナイザの表だけを配る（`python -m wan.umt5_xxl_tokenizer`）"
        )
    if set(asset) != UMT5_TOKENIZER_FIELDS:
        raise DistError(
            f"{path}: 欄 {sorted(asset)} が {sorted(UMT5_TOKENIZER_FIELDS)} でない — 前処理の表など"
            " 単体のトークナイザに無い欄を持つ資産は配らない"
        )
    row = UMT5_SOURCES.get(model)
    if row is None:
        raise DistError(f"umT5 のモデル {model!r} は上流の表に無い（既知: {sorted(UMT5_SOURCES)}）")
    expected_source = {
        "repo": row.source.repo,
        "revision": row.source.revision,
        "files": dict(sorted(row.tokenizer.items())),
    }
    if asset["source"] != expected_source:
        raise DistError(
            f"{path}: トークナイザの出所 {asset['source']!r} が上流の pin {expected_source!r}"
            " と違う — 別の checkpoint・別のバイト列のトークナイザは配らない"
        )
    versions = asset["versions"]
    pinned = versions.get("transformers") if isinstance(versions, dict) else None
    if pinned != umt5_tokenizer.TRANSFORMERS_VERSION:
        raise DistError(
            f"{path}: トークナイザ資産を焼いた transformers {pinned!r} が id 列の正本の版"
            f" {umt5_tokenizer.TRANSFORMERS_VERSION!r} と違う"
        )
    if asset["maxLength"] != UMT5_TOKENS[1]:
        raise DistError(
            f"{path}: maxLength {asset['maxLength']!r} が受理の上限 {UMT5_TOKENS[1]} と違う"
            " — グラフの有効長の上限（ADR 0119 決定 4）"
        )


def umt5_repo_name(_model: str) -> str:
    """配布リポ名（モデルによらず 1 つ）。"""
    return UMT5_REPO_NAME


def umt5_plan(container: Path, tokenizer: Path, model: str = UMT5_DEFAULT_MODEL) -> ModelPlan:
    """umT5-XXL encoder の 1 モデルぶんの計画を組む（検査と読み取りをここで全部済ませる）。

    `tokenizer` は本家のトークナイザ資産（{@link umt5_tokenizer_asset}）— 単体で使えるリポの必須の
    資産で、無ければ落ちる。
    """
    assert_model_name(model)
    assert_umt5_encoder(container, model)
    assert_umt5_tokenizer(tokenizer, model)
    return _encoder_plan(container, model, tokenizer=tokenizer)


def _encoder_plan(container: Path, model: str, *, tokenizer: Path | None) -> ModelPlan:
    """検査を済ませた容器（と資産）の計画（本家の行と取り込みで同じ席・同じ quant 表）。

    `tokenizer` が None なら資産を持たない計画（取り込み由来の手元の実験用ミラー —
    {@link intake_plan}）。
    """
    artifacts = {UMT5_ROLE: Artifact(UMT5_OUTPUT_PATHS[UMT5_ROLE], source=container)}
    if tokenizer is not None:
        artifacts[UMT5_TOKENIZER_ROLE] = Artifact(
            UMT5_OUTPUT_PATHS[UMT5_TOKENIZER_ROLE], source=tokenizer
        )
    return ModelPlan(
        name=model,
        pipeline=UMT5_PIPELINE,
        artifacts=artifacts,
        weights=UMT5_WEIGHTS,
        assets=UMT5_ASSETS if tokenizer is not None else {},
        quants=complete_quant_weights(UMT5_WEIGHTS, UMT5_QUANTS),
        default_quant=UMT5_DEFAULT_QUANT,
        pipeline_config=UMT5_PIPELINE_CONFIG,
    )


def umt5_dist_plan(series_dir: Path, model: str) -> ModelPlan:
    """`--series` の親から計画を組む（CLI のディスパッチ先）。"""
    return umt5_plan(umt5_container(series_dir), umt5_tokenizer_asset(series_dir), model)


#: 改変告知（Apache 2.0 §4(b)）。**このリポが上流の重みへ加えた変更**を列挙する。
#:
#: MUST: 文面は配布形の中身と対応していること — 値としては妥当な散文なので `verify_dist` も
#: manifest 検査も素通りし、配ってからでないと食い違いに気づけない。上流の格納は F32
#: （research 2026-10-03 umt5-export-ram — 容器の F32 のままの重みも下位 16 ビットが 0 でない）
#: なので「bf16 の写し」とは書かない（Wan2.2 の bf16 との関係はカードの Relation の行だけが書く）。
#: 上流は本家の encoder（ADR 0122 決定 1）とトークナイザで、本家は NOTICE ファイルを持たないので、
#: 引き継ぐ NOTICE の本文は無い。Wan2.1 の checkpoint との同一は 2026-10-04 の実測（research
#: `2026-10-04-umt5-upstream-provenance` §2.1 — 確かめた commit を書く。Wan の pin を動かしても、
#: 確かめた事実としては変わらない）。トークナイザの項の「拒む入力」は `wan.umt5_xxl_tokenizer` の
#: docstring（2026-10-15 の実測）。
UMT5_NOTICE_MARKDOWN = """# NOTICE

This repository redistributes a modified form of the encoder of the umT5-XXL checkpoint
`google/umt5-xxl` listed in `README.md` (Google, licensed under the Apache License, Version 2.0 —
see `LICENSE.md`; that repository has no NOTICE file), and a converted form of that checkpoint's
tokenizer. That checkpoint holds the encoder in float32. Its encoder weights are bit-identical to
the float32 `text_encoder` folder of `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` (commit
`0fad780a534b6463e45facd96134c9f345acfa5b`, checked on 2026-10-04). The following changes were
made:

- Only the encoder was converted (the vocabulary embedding `shared.weight` and `encoder.*`); the
  decoder and `lm_head` are not included.
- The weights were converted into the Karume container format (a `.krm` part sequence whose first
  part carries the graph and model descriptors).
- **int8 weights**: the weight matrices of all linear layers and the vocabulary embedding were
  quantized from the source float32 values to the nearest step of a symmetric int8 grid, with one
  float32 scale per output channel (per row for the embedding). The relative-position bias tables
  and the RMSNorm weights keep the source float32 values. Computation runs in float32.
- The feed-forward activation `gelu_new` (the tanh approximation written out with a cube) was
  replaced by the equivalent `GELU(approximate="tanh")`. They are the same function and differ
  only in floating-point rounding.
- The graph runs on the valid tokens only: it takes the token ids and the relative-position bucket
  indices (computed on the host with the upstream bucketing rule) as inputs and has no attention
  mask, instead of padding the prompt to 512 tokens and masking the padding.
- **Tokenizer**: the tokenizer files `tokenizer.json` and `tokenizer_config.json` of the same
  commit were converted into one JSON table (`xxl/tokenizer/tokenizer.json`: the vocabulary and its
  scores, the added tokens, the set of whitespace characters and the maximum length of 512 tokens),
  taken from the tokenizer as `transformers` 5.14.1 loads those files. The host reproduces that
  tokenizer from the table without any text normalization, and it rejects the inputs on which
  `transformers` 5.14.1 and the original `tokenizer.json` produce different token ids (special
  tokens written in the text, `▁` right after a space, whitespace other than the plain space, a
  trailing space, characters outside the vocabulary), as well as inputs of more than 512 tokens,
  which the upstream would truncate.

No retraining and no fine-tuning were performed. The original checkpoint and its original
tokenizer files are not distributed here.
"""


# ---------------------------------------------------------------------------
# モデルカード
# ---------------------------------------------------------------------------

#: このテンプレートが説明できるパイプライン契約。
UMT5_SUPPORTED_PIPELINE = UMT5_PIPELINE

#: HF の pipeline tag（テキストから埋め込みを作る encoder）。
UMT5_PIPELINE_TAG = "feature-extraction"

#: 原文の在処（配布リポ直下の `LICENSE.md` と同じテキスト — Apache 2.0 §4(a)）。
UMT5_LICENSE_TEXT_LINK = "https://www.apache.org/licenses/LICENSE-2.0"

#: この encoder を越境参照する Wan の配布リポ名（カードの「Used by」が名指しする — 公開するのは
#: Wan2.2 だけで、Wan2.1 は公開しない。`wan.distribution.WAN22_REPO_NAME` と同じ綴り。あちらが
#: このモジュールを import するので、こちらからは import できない — 一致は
#: `wan/tests/test_umt5_distribution.py` が見る）。
UMT5_CONSUMER_REPO_NAME = "karume-wan2.2"

#: 受理する有効長（token 数・末尾の `</s>` を含む — ADR 0119 決定 4。下限は `torch.export` の
#: 0 / 1 特殊化を避けた 2・上限はトークナイザ資産の `maxLength`。manifest に無い事実）。
UMT5_TOKENS = (2, 512)

#: encoder の出力の幅（umT5-XXL の `d_model` — カードの Usage が名乗る `[L, 4096]`）。
UMT5_HIDDEN_WIDTH = 4096


def _umt5_upstream(name: str) -> Umt5Source:
    """モデル名 → 上流の出所（表に無ければ描かない — 出所を名乗れないカードは出さない）。"""
    upstream = UMT5_SOURCES.get(name)
    if upstream is None:
        raise ValueError(
            f"モデル '{name}' の上流が出所の表（wan.sources.UMT5_SOURCES）に無い"
            f"（既知: {sorted(UMT5_SOURCES)}）— 出所を名乗れないカードは描かない"
        )
    return upstream


def _umt5_metadata(manifest: Mapping[str, Any]) -> CardMetadata:
    licenses = {_umt5_upstream(name).source.license for name in manifest["models"]}
    if len(licenses) != 1:
        raise ValueError(f"モデルごとにライセンスが割れている（{sorted(licenses)}）")
    return CardMetadata(
        pipeline_tag=UMT5_PIPELINE_TAG,
        base_model=tuple(_umt5_upstream(name).source.repo for name in manifest["models"]),
        # F32 の上流を i8 へ落とし直した配布形（`CardMetadata` の doc — 格納形を変えた配布形）。
        base_model_relation="quantized",
        license=next(iter(licenses)),
        tags=(UMT5_PIPELINE_TAG, "text-encoder", "umt5", "wan", "webgpu"),
    )


def _umt5_overview(manifest: Mapping[str, Any]) -> list[str]:
    low, high = UMT5_TOKENS
    return [
        "## What is this",
        "",
        "The encoder of **umT5-XXL** (`google/umt5-xxl`) as a standalone text encoder for the",
        "WebGPU inference runtime **Karume**: the encoder converted into Karume's container format",
        "(a `.krm` part sequence whose first part carries the graph and model descriptors) with",
        "int8 weights, and the checkpoint's tokenizer converted into one JSON table.",
        "",
        f"- One graph, `{UMT5_ROLE}`: the token ids `{UMT5_IDS_INPUT}` `[1, L]` and the",
        f"  relative-position bucket indices `{UMT5_BUCKETS_INPUT}` `[L, L]` (both int32) in,",
        f"  the encoder's last hidden states `[1, L, {UMT5_HIDDEN_WIDTH}]` in float32 out.",
        f"  `L` is the number of tokens of the prompt itself ({low} to {high}, the end token",
        "  `</s>` included) — there is no padding and no attention mask.",
        f"- One asset, `{UMT5_TOKENIZER_ROLE}` (format `{umt5_xxl_tokenizer.ASSET_FORMAT}`):",
        "  the tokenizer of `google/umt5-xxl` as one JSON table — the vocabulary and its scores,",
        "  the added tokens, the set of whitespace characters and the maximum length of",
        f"  {high} tokens. It carries no prompt cleaning.",
        "- The relative-position bucket indices depend on `L` only and are computed on the host:",
        "  entry `[i, j]` is the bucket of the distance `j − i` (key position minus query",
        "  position) under the upstream T5 rule for a bidirectional encoder with 32 buckets and a",
        "  maximum distance of 128 (`relative_attention_num_buckets` /",
        "  `relative_attention_max_distance` in the upstream `config.json`). Each direction has",
        "  16 buckets (keys after the query use 16–31): distances 0 to 7 get one bucket each,",
        "  and longer distances fall into 8 logarithmically spaced buckets that start at 8, 12,",
        "  16, 23, 32, 46, 64 and 91 (the last one covers every distance from 91 on).",
        f"- Exporter used for the conversion: `{manifest['generator']}`. The distribution manifest"
        f" is `karume.json` (`{manifest['format']}`).",
    ]


def _umt5_base_weights(manifest: Mapping[str, Any]) -> list[str]:
    """帰属節。上流の revision は pin した 40 桁を全部出す（容器の provenance と同じ値）。"""
    lines = [
        "## Base weights and attribution",
        "",
        "Converted into the container format — the original checkpoint is not distributed here.",
        "",
    ]
    for name in manifest["models"]:
        upstream = _umt5_upstream(name)
        source = upstream.source
        # 本家の encoder はリポ直下にある（subfolder は無い）— 読んだキーの範囲で名指しする。
        lines.append(
            f"- **`{name}`**: the encoder weights (`shared.weight` and `encoder.*`) of"
            f" [{source.repo}](https://huggingface.co/{source.repo}) at commit `{source.revision}`,"
            f" licensed **{source.license}** (as of retrieval;"
            f" [full text]({UMT5_LICENSE_TEXT_LINK}) — a verbatim copy is in `LICENSE.md`)."
            " The decoder and `lm_head` are not included."
        )
        files = " and ".join(
            f"`{file}` (SHA-256 `{digest[:16]}…`)"
            for file, digest in sorted(upstream.tokenizer.items())
        )
        lines.append(
            f"- **`{name}` tokenizer**: the files {files} of the same repository and commit,"
            " converted into one JSON table."
        )
    lines += [
        "- **Relation to the Wan checkpoints**: every tensor is bit-identical to the float32",
        "  `text_encoder` folder of `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` (commit `0fad780a…`), and",
        "  the bfloat16 `text_encoder` of `Wan-AI/Wan2.2-TI2V-5B-Diffusers` (commit `b8fff731…`)",
        "  is its round-to-nearest-even rounding (checked on 2026-10-04).",
        "- **Changes made here** (listed in full in `NOTICE.md`, per Apache 2.0 §4(b)):",
        "  only the encoder converted (no decoder, no `lm_head`);",
        "  conversion into the Karume container format; the weight matrices of the linear layers",
        "  and the vocabulary embedding quantized to int8 with one scale per output channel (per",
        "  row for the embedding), the relative-position bias tables and the RMSNorm weights kept",
        "  in float32 at the source values; `gelu_new` replaced by the equivalent",
        '  `GELU(approximate="tanh")`; the graph runs on the valid tokens with the bucket indices',
        "  as an input; the tokenizer converted into one JSON table. No retraining and no",
        "  fine-tuning.",
    ]
    return lines


def _consumer_link() -> str:
    """この encoder を越境参照する Wan のリポへのリンク（Markdown）。"""
    repo = f"{HF_OWNER}/{UMT5_CONSUMER_REPO_NAME}"
    return f"[`{repo}`](https://huggingface.co/{repo})"


def _umt5_usage(repo: str) -> list[str]:
    """Usage: 単体の公開クラス `Umt5Encoder`（`@karume/models/umt5`）の使い方・入力の規則・
    この encoder を越境参照する Wan の配布形。

    `fromPretrained` の source は他の家族のカードと同じ object ref 形 + pin の席
    （`karume.modelcard.REVISION_PIN_COMMENT`）。差し替えの段落（ADR 0122 決定 7 の 2）は、Wan で
    この encoder の代わりに互換の encoder を差す使い方で、pin の規則（文字列の取得元は main 追従）と
    precomputed の経路では効かないことを書く。入力の規則は `wan.umt5_xxl_tokenizer` の拒否の 6 つ
    （2026-10-15 の実測）。
    """
    low, high = UMT5_TOKENS
    return [
        "## Usage",
        "",
        "```ts",
        'import { Umt5Encoder } from "jsr:@karume/models/umt5";',
        "",
        "const encoder = await Umt5Encoder.fromPretrained({",
        f'  repo: "{repo}",',
        *REVISION_PIN_COMMENT,
        "});",
        'const { inputIds, hiddenStates } = await encoder.encode("a cat");'
        f" // hiddenStates: [L, {UMT5_HIDDEN_WIDTH}] f32, row-major",
        "await encoder.dispose();",
        "```",
        "",
        "`encode` tokenizes the text with the bundled tokenizer, builds the relative-position",
        "bucket indices, runs the graph and returns the token ids (`L` of them, ending with",
        f"`</s>`) with the encoder's last hidden states, one row of {UMT5_HIDDEN_WIDTH:,} floats",
        "per token.",
        "",
        "### Input rules",
        "",
        "The text is tokenized as it is — no prompt cleaning and no Unicode normalization — and",
        "`</s>` is appended. The token ids then equal those of `AutoTokenizer` in `transformers`",
        "5.14.1 and of the original `tokenizer.json` of `google/umt5-xxl` alike. Inputs on which",
        "those two disagree, or that the graph cannot take, are rejected with an error instead of",
        "being encoded:",
        "",
        "- characters outside the vocabulary;",
        "- special tokens written in the text (`</s>`, `<extra_id_0>` and the other added tokens);",
        "- `▁` (U+2581) right after a space;",
        "- whitespace other than the plain space U+0020 (tabs, line breaks, no-break spaces, the",
        "  ideographic space and so on), and a trailing space — leading spaces and runs of spaces",
        "  between words are accepted;",
        f"- more than {high} tokens (the upstream would truncate silently);",
        f"- fewer than {low} tokens (an empty text).",
        "",
        "### Used by",
        "",
        f"{_consumer_link()} (Wan2.2 TI2V 5B) references this encoder from its own `karume.json`",
        "at a pinned commit of this repository (the size and the SHA-256 of every part are",
        "declared there too), so `WanTi2vPipeline.fromPretrained` on that repository fetches the",
        "`text_encoder` from here — you do not pass this repository yourself. It does not use this",
        "repository's tokenizer: its own tokenizer asset adds Wan's prompt cleaning.",
        "",
        "To run Wan2.2 with a compatible umT5-XXL encoder instead of this one (for example a",
        "fine-tune converted with the Karume export recipe), pass it through the component swap",
        "seat: `components: { text_encoder: { source: { repo, revision } } }` in the options of",
        "`WanTi2vPipeline.fromPretrained`. Pin the replacement with a commit — a bare repository",
        "name follows `main`, so the encoder would move with every push while the Wan manifest",
        'pins this one. The swap needs the GPU text encoder: `textEncoder: "precomputed"` never',
        "loads umT5, so a swap has no effect there. The replacement is admitted only if its graph",
        "description has the same SHA-256 as the `text_encoder` declared by the Wan manifest and",
        "its weights are int8 with float32 tables; the tokenizer still comes from the Wan",
        "distribution and is not checked against the replacement.",
    ]


def _umt5_limits(model: Mapping[str, Any]) -> list[str]:
    """宣言された device limit（manifest の `requiredLimits` から — 焼かれていなければ節ごと
    無い）。"""
    declared = {
        name: quant["requiredLimits"]
        for name, quant in model["quants"].items()
        if "requiredLimits" in quant
    }
    if not declared:
        return []
    lines = ["### Device limits", ""]
    for name, limits in declared.items():
        spelled = " / ".join(f"`{key}` ≥ {value:,} bytes" for key, value in limits.items())
        lines.append(f"- `{name}`: {spelled}.")
    lines += [
        "",
        "The largest resident buffer is the int8 vocabulary embedding, held as one buffer, so the",
        "device has to grant these limits.",
    ]
    return lines


def render_umt5_model_card(
    manifest: Mapping[str, Any], repo: str, host_assets: Mapping[FileRefKey, int] = {}
) -> str:
    """umT5-XXL encoder の配布形の `README.md` 本文を組み立てる（純関数・末尾改行つき）。

    `repo` は Usage の `Umt5Encoder.fromPretrained` に綴るこのリポの HF リポ ID。
    """
    require_pipeline(manifest, UMT5_SUPPORTED_PIPELINE)
    return render(
        (
            frontmatter(_umt5_metadata(manifest)),
            ["", "# umT5-XXL text encoder (int8) — Karume", ""],
            _umt5_overview(manifest),
            [""],
            _umt5_base_weights(manifest),
            [""],
            models(manifest),
            [""],
            _umt5_usage(repo),
            *model_sections(
                manifest,
                (partial(quants, host_assets=host_assets), _umt5_limits),
            ),
        )
    )


#: `--pipeline umt5` の 1 行（ドライバが core の PIPELINES へ合成する）。
PIPELINE = Pipeline(
    default_model=UMT5_DEFAULT_MODEL,
    repo_name=umt5_repo_name,
    plan=umt5_dist_plan,
    # 帰属（上流リポ・ライセンス）はモデル名から一意に決まる（`wan.sources.UMT5_SOURCES`）ので、
    # 選ばせる軸にしない。
    card_profiles={"umt5": render_umt5_model_card},
    # 上流ライセンス（Apache 2.0）の再配布条件 §4 は配布リポ 1 つに掛かる（ADR 0092 決定 7）。
    root_files={
        "LICENSE.md": apache_license_2_0(),
        "NOTICE.md": UMT5_NOTICE_MARKDOWN,
    },
)


# ---------------------------------------------------------------------------
# 取り込み由来の手元の実験用ミラー（ADR 0122 決定 5 / 6）
# ---------------------------------------------------------------------------


def intake_series_name(name: str) -> str:
    """取り込み由来のモデルの系列（`umt5-xxl-<名前>-i8-dyn`）。書き手
    （`wan.umt5_export.intake_input`）と組み立て（{@link intake_container}）が同じ綴りを引く。"""
    return f"umt5-xxl-{name}-i8-dyn"


def intake_container(series_dir: Path, name: str) -> Path:
    """系列の親（`outputs/series/`）から取り込み由来の容器の代表 path を引く。"""
    return series_dir / intake_series_name(name) / UMT5_ROLE / UMT5_MODEL_FILE


def intake_repo_name(name: str) -> str:
    """取り込み由来のモデルのリポ名（カードの綴り — `karume-umt5-xxl` を名乗らせない）。

    公開の経路は無い（ADR 0122 決定 5 — 公開するなら命名の裁定が要る）ので、`karume-` を
    付けず、手元の実験用であることを名前に出す。
    """
    return f"umt5-xxl-{name}-local"


#: 宣言済みのライセンスで `LICENSE.md` に置ける逐語の本文（識別子 → 本文）。本文の無い識別子は
#: 組み立ての前に落とす。MIT は著作権行（権利者）が要り、記録からは導けないので載せない。
INTAKE_LICENSE_TEXTS: Mapping[str, Callable[[], str]] = {
    "apache-2.0": apache_license_2_0,
    "cc-by-sa-4.0": cc_by_sa_4_0,
}

#: 取り込み由来のミラーのカードの帰属プロファイル名（1 つだけ — 選ばせない）。
INTAKE_CARD_PROFILE = "umt5-intake"


def _intake_terms(intake: Umt5Intake) -> str:
    """上流の宣言（ライセンス）の 1 文（NOTICE とカードが同じ事実を書く）。"""
    if intake.undeclared:
        return (
            f"That repository declares no license (recorded as `{UNDECLARED_LICENSE}`), so the"
            " terms of redistribution are unknown: this conversion is for local experiments only"
            " and must not be redistributed."
        )
    return (
        f"That repository declares the license `{intake.license}` (as of retrieval — a verbatim"
        " copy is in `LICENSE.md`)."
    )


def _intake_base(intake: Umt5Intake) -> str:
    if intake.base_model is None:
        return "It declares no base model."
    return "It declares the base model " + ", ".join(f"`{m}`" for m in intake.base_model) + "."


def _intake_precision(intake: Umt5Intake) -> str:
    """元の dtype の 1 文（丸めの出発点を偽らない — ADR 0122 決定 5）。"""
    if intake.dtype == "BF16":
        return (
            "The upstream file stores bfloat16 values. They were widened to float32 without loss"
            " before quantization, so the int8 rounding starts from the bfloat16 values."
        )
    return "The upstream file stores float32 values."


def intake_notice(intake: Umt5Intake) -> str:
    """取り込み由来のミラーの `NOTICE.md`（記録から導く — ADR 0122 決定 5）。

    MUST: 上流は記録のリポ・revision だけを名乗る。本家向けの NOTICE（{@link
    UMT5_NOTICE_MARKDOWN}）の「No retraining」や、Wan・Google を上流と名乗る文は書かない — 追加
    学習は第三者が行ったもので、その上流は宣言されていない。
    """
    return "\n".join(
        [
            "# NOTICE",
            "",
            "This directory is a local experimental conversion and is not published. It holds a",
            f"modified form of the file `{intake.file.name}` of the Hugging Face repository",
            f"`{intake.repo}` at commit `{intake.revision}` (SHA-256 `{intake.file.sha256}`).",
            f"{_intake_terms(intake)} {_intake_base(intake)}",
            "",
            "The following changes were made:",
            "",
            "- The weights were converted into the Karume container format (a `.krm` part",
            "  sequence whose first part carries the graph and model descriptors).",
            f"- {_intake_precision(intake)}",
            "- **int8 weights**: the weight matrices of all linear layers and the vocabulary",
            "  embedding were quantized to the nearest step of a symmetric int8 grid, with one",
            "  float32 scale per output channel (per row for the embedding). The relative-position",
            "  bias tables and the RMSNorm weights keep their float32 values. Computation runs in",
            "  float32.",
            "- The feed-forward activation `gelu_new` (the tanh approximation written out with a",
            '  cube) was replaced by the equivalent `GELU(approximate="tanh")`. They are the same',
            "  function and differ only in floating-point rounding.",
            "- The graph runs on the valid tokens only: it takes the token ids and the",
            "  relative-position bucket indices (computed on the host) as inputs and has no",
            "  attention mask.",
            "",
            "The original file is not included here.",
            "",
        ]
    )


def intake_root_files(intake: Umt5Intake) -> dict[str, str]:
    """取り込み由来のミラーのルートのファイル（記録から導く — ADR 0122 決定 5 / 6）。

    宣言済みなら `LICENSE.md` はその識別子の逐語の本文（{@link INTAKE_LICENSE_TEXTS} に無ければ
    fail loudly）。未宣言なら `LICENSE.md` を置かない（名乗れる本文が無い）。どちらも
    `NOTICE.md` は {@link intake_notice}。
    """
    files = {NOTICE_FILENAME: intake_notice(intake)}
    if intake.undeclared:
        return files
    text = INTAKE_LICENSE_TEXTS.get(intake.license)
    if text is None:
        raise DistError(
            f"取り込み '{intake.name}' のライセンス {intake.license!r} の本文が無い"
            f"（置ける本文: {sorted(INTAKE_LICENSE_TEXTS)}）— LICENSE.md を組めない"
        )
    return {LICENSE_FILENAME: text(), **files}


def _intake_header(intake: Umt5Intake) -> list[str]:
    """カードの冒頭（手元の実験用であることと、上流の宣言 — ADR 0122 決定 6）。"""
    if intake.undeclared:
        return [
            "> **Local experimental conversion — do not redistribute.** The upstream repository",
            f"> declares no license (recorded as `{UNDECLARED_LICENSE}`), so the terms of",
            "> redistribution are unknown. This directory was built only to try the encoder",
            "> locally and carries no `LICENSE.md`.",
        ]
    return [
        "> **Local experimental conversion — not published.** Publishing a third-party encoder",
        "> is a separate decision; this directory was built only to try the encoder locally.",
    ]


def _intake_overview(manifest: Mapping[str, Any], intake: Umt5Intake) -> list[str]:
    return [
        "## What is this",
        "",
        f"A third-party umT5-XXL-compatible text encoder (`{intake.name}`), converted into the",
        "WebGPU inference runtime **Karume**'s container format with int8 weights. It keeps the",
        "layout of the umT5-XXL encoder (`shared.weight` and `encoder.*`).",
        "",
        f"- One graph, `{UMT5_ROLE}`: the token ids `{UMT5_IDS_INPUT}` `[1, L]` and the",
        f"  relative-position bucket indices `{UMT5_BUCKETS_INPUT}` `[L, L]` (both int32) in,",
        "  the encoder's last hidden states `[1, L, d_model]` in float32 out.",
        "- The tokenizer and the bucket indices are not part of this directory.",
        f"- Exporter used for the conversion: `{manifest['generator']}`. The distribution manifest"
        f" is `karume.json` (`{manifest['format']}`).",
    ]


def _intake_attribution(intake: Umt5Intake) -> list[str]:
    repo = intake.repo
    license_line = (
        f"not declared by the upstream repository (`{UNDECLARED_LICENSE}`)."
        if intake.undeclared
        else f"declared `{intake.license}` (as of retrieval; a verbatim copy is in `LICENSE.md`)."
    )
    base_line = (
        "not declared."
        if intake.base_model is None
        else "declared " + ", ".join(f"`{m}`" for m in intake.base_model) + "."
    )
    base_link = f"[{intake.config.base_repo}](https://huggingface.co/{intake.config.base_repo})"
    config_lines = (
        [
            "- **Configuration**: assumed — the upstream has no `config.json`, so the",
            f"  configuration of {base_link} at commit `{intake.config.base_revision}` was copied.",
            "  The relative-position bucket settings cannot be derived from the weights; if the",
            "  upstream was trained with other settings, nothing here detects it.",
        ]
        if intake.config_assumed
        else [
            "- **Configuration**: the upstream `config.json`, checked against",
            f"  {base_link} at commit `{intake.config.base_revision}` (the same graph fields and",
            "  relative-position buckets).",
        ]
    )
    return [
        "## Base weights and attribution",
        "",
        "Converted into the container format — the original file is not included here.",
        "",
        f"- **Weights**: the file `{intake.file.name}` of [{repo}](https://huggingface.co/{repo})",
        f"  at commit `{intake.revision}` (SHA-256 `{intake.file.sha256}`), retrieved"
        f" {intake.fetched_at}.",
        f"- **License**: {license_line}",
        f"- **Base model**: {base_line}",
        f"- **Source precision**: {_intake_precision(intake)}",
        *config_lines,
        "- **Changes made here** (listed in full in `NOTICE.md`): conversion into the Karume",
        "  container format; the weight matrices of the linear layers and the vocabulary",
        "  embedding quantized to int8 with one scale per output channel (per row for the",
        "  embedding), the relative-position bias tables and the RMSNorm weights kept in float32;",
        '  `gelu_new` replaced by the equivalent `GELU(approximate="tanh")`; the graph runs on the',
        "  valid tokens with the bucket indices as an input.",
    ]


def render_intake_model_card(
    manifest: Mapping[str, Any],
    repo: str,
    host_assets: Mapping[FileRefKey, int] = {},
    *,
    intake: Umt5Intake,
) -> str:
    """取り込み由来のミラーの `README.md`（記録から導く — 元の dtype・出所・構成の出所を書く）。

    frontmatter の `license` は記録の値（未宣言なら印 `NOASSERTION` — 容器の provenance と同じ
    綴り）。`base_model` は宣言されたときだけ書く。`repo` は受けるだけで使わない（公開しない）。
    """
    require_pipeline(manifest, UMT5_SUPPORTED_PIPELINE)
    head = frontmatter(
        CardMetadata(
            pipeline_tag=UMT5_PIPELINE_TAG,
            base_model=intake.base_model or (),
            license=intake.license,
            tags=(UMT5_PIPELINE_TAG, "text-encoder", "umt5", "local-experiment"),
        )
    )
    if intake.base_model is None:
        # 宣言の無いベースを空の `base_model:`（YAML の null）として名乗らない — 欄ごと外す。
        head = [line for line in head if line != "base_model:"]
    return render(
        (
            head,
            ["", f"# umT5-XXL compatible text encoder `{intake.name}` (int8) — Karume", ""],
            _intake_header(intake),
            [""],
            _intake_overview(manifest, intake),
            [""],
            _intake_attribution(intake),
            [""],
            models(manifest),
            *model_sections(
                manifest,
                (partial(quants, host_assets=host_assets), _umt5_limits),
            ),
        )
    )


def intake_plan(
    container: Path, intake: Umt5Intake, model: str, *, allow_undeclared_license: bool
) -> ModelPlan:
    """取り込み由来のモデルの計画（検査と読み取りをここで全部済ませる — 1 バイトも書く前）。

    MUST: ライセンス未宣言の取り込みは、明示（`--allow-undeclared-license`）が無ければ
    落とす（ADR 0122 決定 6 の 1）。容器の provenance は記録のライセンス（未宣言なら印）と
    revision へ突き合わせる（{@link assert_umt5_encoder}）ので、印の容器は印の記録とだけ組める。
    """
    try:
        assert_license_intent(intake, allowed=allow_undeclared_license)
    except Umt5IntakeError as cause:
        raise DistError(str(cause)) from cause
    assert_model_name(model)
    assert_umt5_encoder(container, model, intake)
    # 取り込み由来のミラーは本家のトークナイザ資産を持たない（第三者の encoder に本家の表を
    # 名乗らせない — 資産の席は手元の実験の範囲で要らない）。
    return _encoder_plan(container, model, tokenizer=None)


def _intake_dist_plan(
    series_dir: Path, model: str, *, intake: Umt5Intake, allow_undeclared_license: bool
) -> ModelPlan:
    return intake_plan(
        intake_container(series_dir, intake.name),
        intake,
        model,
        allow_undeclared_license=allow_undeclared_license,
    )


def _intake_repo(_model: str, *, intake: Umt5Intake) -> str:
    return intake_repo_name(intake.name)


def intake_pipeline(intake: Umt5Intake, *, allow_undeclared_license: bool) -> Pipeline:
    """取り込み 1 つぶんの `--pipeline umt5` の行（ドライバが `--intake` を受けたときだけ組む）。

    ルートのファイルは記録から導く（{@link intake_root_files} — 本文の無いライセンスはここで
    落ちる）。既定の出力先（`models/` の外）と、出力先が `models/` の下なら落とす門はドライバが
    持つ。
    """
    return Pipeline(
        default_model=intake.name,
        repo_name=partial(_intake_repo, intake=intake),
        plan=partial(
            _intake_dist_plan, intake=intake, allow_undeclared_license=allow_undeclared_license
        ),
        card_profiles={INTAKE_CARD_PROFILE: partial(render_intake_model_card, intake=intake)},
        root_files=intake_root_files(intake),
    )
