"""umT5-XXL encoder の配布 recipe — 配布リポ `karume-umt5-xxl`（ADR 0119 追記「段 10d の設計」D）。

配るのはグラフ 1 本（`text_encoder` — 系列 `umt5-xxl-i8-dyn` の i8 の S 形容器）だけで、資産は
持たない（トークナイザは Wan の前処理の表と束ねた形式なので Wan のリポの資産 — 同 C）。Wan の配布形
`karume-wan2.1` はこのリポの容器を越境参照する（同 A — ADR 0109 決定 3）。読む TS の家族は無く、
pipeline 名 {@link UMT5_PIPELINE} は部品の役を名乗るだけ（`karume/5` は model ごとに pipeline と
quant 席を必須にする）。

出所は本家 `google/umt5-xxl` の encoder（`shared.weight` と `encoder.*` — 容器が名乗る出所は
`wan.sources.UMT5_SOURCES` の pin・ADR 0122 決定 1）。Wan2.1 Diffusers の `text_encoder` とは f32 で
全テンソルがビット一致し、Wan2.2 TI2V-5B Diffusers の bf16 はその RNE 丸め（2026-10-04 の
実測）なので、カードと NOTICE はそう書く。

公開面は {@link PIPELINE} 1 つ — リポの dist ドライバ（`tools/export-recipes/dist.py`）がこれを
core の PIPELINES へ合成する（`--pipeline umt5`）。

MUST: このモジュールは torch を import しない（`import dist` が torch を読まない —
`tests/test_dist_driver.py` の `TestImportingTheDriver`）。書き手（`wan.umt5_export`）は torch を
読むので、綴り（系列名・部品名・入力名・相対位置の表の属性名）はここに置き、書き手の綴りとの一致は
`wan/tests/test_umt5_distribution.py` の門が見る。重みの種類の分類（{@link storage_kind}）は書き手も
ここから引く（分類を 2 本持たない）。
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from functools import partial
from pathlib import Path
from typing import Any

from _shared.container_read import ContainerReadError, read_layouts
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
    complete_quant_weights,
    ir_graph,
)
from karume.modelcard import (
    HF_OWNER,
    CardMetadata,
    frontmatter,
    model_sections,
    models,
    quants,
    render,
    require_pipeline,
)
from wan.sources import UMT5_SOURCES, Umt5Source

#: パイプライン契約（ADR 0041 §2 — モデル単位）。読む TS の家族は無い — 部品の役を名乗る
#: （ADR 0119 追記 D）。Wan の読み手は越境参照の先の `karume.json` を読まないので、この名前を
#: 解釈する実装は要らない。
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

#: 出力の相対 path（**モデルサブツリー内**）。格納ラベルを path に入れるのは Wan の DiT と同じ理由
#: （格納の席を足した日に既存の path を動かさない）。
#:
#: MUST: Wan の配布形の同じ部品の path と同じ綴り（`wan.distribution.WAN_OUTPUT_PATHS` はここを
#: 引く）— 越境参照は参照元の `karume.json` が宣言する `<モデル名>/<この path>` を引き当てるので、
#: 割れると Wan の組み立てが「参照元に無い」で落ちる。
UMT5_OUTPUT_PATHS: Mapping[str, str] = {UMT5_ROLE: f"{UMT5_ROLE}/model.i8.krm"}

#: weights の宣言（部品 → dtype ラベル → 配置の役割）。i8 の 1 席だけ（ADR 0119 決定 5 — i4 は
#: 品質を測ってから別の席として足す）。
UMT5_WEIGHTS: Mapping[str, Mapping[str, WeightFiles]] = {UMT5_ROLE: {"i8": WeightFiles(UMT5_ROLE)}}

#: assets は持たない（トークナイザは Wan のリポの資産 — ADR 0119 追記 C）。
UMT5_ASSETS: Mapping[str, str] = {}

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


def assert_umt5_encoder(container: Path, model: str) -> int:
    """umT5 の容器の門（出所・束縛表・入出力の契約）を全部掛け、出力の幅を返す。

    Wan の配布形も同じ容器を越境参照する部品として持つので、両方の計画がこの 1 本を通る（Wan の
    計画は格納の要求表で別に i8 の席も見る）。
    """
    upstream = UMT5_SOURCES.get(model)
    if upstream is None:
        raise DistError(
            f"umT5 のモデル {model!r} は知らない（既知: {' / '.join(sorted(UMT5_SOURCES))}）—"
            " 上流の出所の表（wan.sources.UMT5_SOURCES）に載ったモデルだけを配る"
        )
    assert_component_present(container)
    # 容器が名乗る出所を上流の pin（`wan.sources.UMT5_SOURCES` が正本）へ突き合わせる — 束縛表と
    # 入出力の形は同じ構造の別の checkpoint でも通るので、出所でしか閉じられない。
    assert_upstream_provenance(
        container, license=upstream.source.license, revision=upstream.source.revision
    )
    assert_umt5_bindings(container)
    return umt5_context_width(container)


def umt5_container(series_dir: Path) -> Path:
    """系列の親ディレクトリ（`outputs/series/`）から容器の代表 path を引く。"""
    return series_dir / UMT5_SERIES / UMT5_ROLE / UMT5_MODEL_FILE


def umt5_repo_name(_model: str) -> str:
    """配布リポ名（モデルによらず 1 つ）。"""
    return UMT5_REPO_NAME


def umt5_plan(container: Path, model: str = UMT5_DEFAULT_MODEL) -> ModelPlan:
    """umT5-XXL encoder の 1 モデルぶんの計画を組む（検査と読み取りをここで全部済ませる）。"""
    assert_model_name(model)
    assert_umt5_encoder(container, model)
    return ModelPlan(
        name=model,
        pipeline=UMT5_PIPELINE,
        artifacts={UMT5_ROLE: Artifact(UMT5_OUTPUT_PATHS[UMT5_ROLE], source=container)},
        weights=UMT5_WEIGHTS,
        assets=UMT5_ASSETS,
        quants=complete_quant_weights(UMT5_WEIGHTS, UMT5_QUANTS),
        default_quant=UMT5_DEFAULT_QUANT,
        pipeline_config=UMT5_PIPELINE_CONFIG,
    )


def umt5_dist_plan(series_dir: Path, model: str) -> ModelPlan:
    """`--series` の親から計画を組む（CLI のディスパッチ先）。"""
    return umt5_plan(umt5_container(series_dir), model)


#: 改変告知（Apache 2.0 §4(b)）。**このリポが上流の重みへ加えた変更**を列挙する。
#:
#: MUST: 文面は配布形の中身と対応していること — 値としては妥当な散文なので `verify_dist` も
#: manifest 検査も素通りし、配ってからでないと食い違いに気づけない。上流の格納は F32
#: （research 2026-10-03 umt5-export-ram — 容器の F32 のままの重みも下位 16 ビットが 0 でない）
#: なので「bf16 の写し」とは書かない（Wan2.2 の bf16 との関係はカードの Relation の行だけが書く）。
#: 上流は本家の encoder（ADR 0122 決定 1）で、本家は NOTICE ファイルを持たないので、引き継ぐ NOTICE
#: の本文は無い。Wan2.1 の checkpoint との同一は 2026-10-04 の実測（research
#: `2026-10-04-umt5-upstream-provenance` §2.1 — 確かめた commit を書く。Wan の pin を動かしても、
#: 確かめた事実としては変わらない）。
UMT5_NOTICE_MARKDOWN = """# NOTICE

This repository redistributes a modified form of the encoder of the umT5-XXL checkpoint
`google/umt5-xxl` listed in `README.md` (Google, licensed under the Apache License, Version 2.0 —
see `LICENSE.md`; that repository has no NOTICE file). That checkpoint holds the encoder in
float32. Its encoder weights are bit-identical to the float32 `text_encoder` folder of
`Wan-AI/Wan2.1-T2V-1.3B-Diffusers` (commit `0fad780a534b6463e45facd96134c9f345acfa5b`, checked on
2026-10-04). The following changes were made:

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

No retraining and no fine-tuning were performed. The original checkpoint is not distributed here,
and neither is the tokenizer.
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

#: Wan の配布リポ名（カードの Usage が名指しする参照元 — `wan.distribution.WAN_REPO_NAME` と同じ
#: 綴り。あちらがこのモジュールを import するので、こちらからは import できない — 一致は
#: `wan/tests/test_umt5_distribution.py` が見る）。
UMT5_CONSUMER_REPO_NAME = "karume-wan2.1"

#: Wan の読み手が受ける有効長（token 数・末尾の `</s>` を含む — ADR 0119 決定 4。TS 側
#: `packages/models/src/wan/text/tokenizer.ts` の受理集合で、上限はトークナイザ資産の
#: `maxLength`。manifest に無い事実）。
UMT5_TOKENS = (2, 512)


def _consumer_link() -> str:
    """参照元の Wan のリポへのリンク（Markdown）。"""
    repo = f"{HF_OWNER}/{UMT5_CONSUMER_REPO_NAME}"
    return f"[`{repo}`](https://huggingface.co/{repo})"


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
        "The **umT5-XXL text encoder** that Wan2.1 uses, converted into the WebGPU inference",
        "runtime **Karume**'s container format (a `.krm` part sequence whose first part carries",
        "the graph and model descriptors), with int8 weights.",
        "",
        f"- One graph, `{UMT5_ROLE}`: the token ids `{UMT5_IDS_INPUT}` `[1, L]` and the",
        f"  relative-position bucket indices `{UMT5_BUCKETS_INPUT}` `[L, L]` (both int32) in,",
        "  the encoder's last hidden states `[1, L, d_model]` in float32 out. `L` is the number",
        f"  of tokens of the prompt itself (the Wan pipeline accepts {low} to {high}, the end",
        "  token included) — there is no padding and no attention mask.",
        "- The tokenizer and the bucket indices are not part of this repository: the Wan2.1",
        f"  distribution {_consumer_link()} carries the tokenizer together with",
        "  Wan's prompt cleaning, and the host builds the bucket indices.",
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
        source = _umt5_upstream(name).source
        # 本家の encoder はリポ直下にある（subfolder は無い）— 読んだキーの範囲で名指しする。
        lines.append(
            f"- **`{name}`**: the encoder weights (`shared.weight` and `encoder.*`) of"
            f" [{source.repo}](https://huggingface.co/{source.repo}) at commit `{source.revision}`,"
            f" licensed **{source.license}** (as of retrieval;"
            f" [full text]({UMT5_LICENSE_TEXT_LINK}) — a verbatim copy is in `LICENSE.md`)."
            " The decoder and `lm_head` are not included."
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
        "  as an input. No retraining and no fine-tuning.",
    ]
    return lines


def _umt5_usage() -> list[str]:
    """Usage: 単体の公開クラスは無い — Wan の配布形が越境参照で取る（ADR 0119 追記 A）。"""
    return [
        "## Usage",
        "",
        "There is no standalone pipeline class for this repository. The Wan2.1 distribution",
        f"{_consumer_link()} references the encoder from its own",
        "`karume.json` at a pinned commit of this repository (the size and the SHA-256 of every",
        "part are declared there too), so `WanPipeline.fromPretrained` on that repository fetches",
        "it from here — you do not pass this repository yourself.",
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
    manifest: Mapping[str, Any], repo: str, host_assets: Mapping[str, int] = {}
) -> str:
    """umT5-XXL encoder の配布形の `README.md` 本文を組み立てる（純関数・末尾改行つき）。

    `repo` は受けるだけで使わない — Usage が名指しするのは参照元の Wan のリポで、このリポ自身を
    `fromPretrained` に渡す使い方は無い（core の描き手の型に合わせる）。
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
            _umt5_usage(),
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
