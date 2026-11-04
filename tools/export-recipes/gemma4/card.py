"""gemma4 配布形のモデルカード（`README.md`）— manifest から機械導出する純関数。

汎用の描画部品（frontmatter・モデル一覧・quant 表・節の組み立て）は
`karume.modelcard` が持つ。ここが持つのは **gemma4 固有の事実**だけ: 帰属（出所・ライセンス）と、
この pipeline のカードに何を書くか。

帰属は**モデル名から一意に決まる**（{@link GEMMA4_UPSTREAM}）ので、プロファイルの軸には
割らない — 分けると「別のチェックポイントを別の帰属で配る」取り違えを操作者が起こせる。

ライセンス方針（2026-09-01 のユーザー裁定 — `gemma4/README.md` 冒頭）: 上流は **Apache 2.0** で、
カードは Apache 2.0 の帰属とライセンス文へのリンクを持ち、**使われ方の詳細は上流モデルカードへ
誘導する**（このプロジェクトが keep する情報ではない）。再配布条件のファイル（`LICENSE.md` /
`NOTICE.md`）はカードではなく `gemma4.distribution.gemma4_root_files` が配布リポ直下へ置く。

MUST: **数値・ダウンロード量・quant 表・dtype ラベルは 1 つ残らず manifest から導出する**
（`karume.modelcard` の同 MUST がそのまま掛かる）。ここが持ってよい定数は、manifest に
**存在しない事実**だけ。
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from functools import partial
from typing import Any

from _shared.upstream import hf_upstreams
from karume.modelcard import (
    CardMetadata,
    FileRefKey,
    default_model,
    from_pretrained,
    frontmatter,
    introduction,
    knob,
    model_sections,
    models,
    quants,
    render,
    require_pipeline,
)

#: このテンプレートが説明できるパイプライン契約（ADR 0041 §2 — モデル単位）。
GEMMA4_SUPPORTED_PIPELINE = "gemma4/1"

#: HF の pipeline tag。配布形が持つのは**テキストデコーダだけ**なので、上流カードの
#: `any-to-any`（画像・音声タワー込み）は名乗れない。
GEMMA4_PIPELINE_TAG = "text-generation"

#: 題は**この配布形が何を持つか**だけを名乗る。MUST: モデルの綴り（`E2B` など）を焼かない —
#: gemma4 は家族 1 リポ（ADR 0092 決定 1）で manifest に並ぶモデルは増えうるので、題が 1 つの
#: モデルを名乗ると別のモデルを足した日に嘘になる。名乗るべき綴りは
#: {@link _gemma4_upstream_spellings} が manifest から引く。
GEMMA4_TITLE = "Gemma 4 Text Decoder — Karume"

#: モデル名 → 上流チェックポイントの HF リポ ID。**この 1 表が「どのモデルが何の重みか」の
#: 唯一の事実**で、`gemma4.distribution` は系列名 / 入力素材のディレクトリ名をここから導く。
#: 載っていないモデル名は帰属を書けないので、カードは描かずに落ちる。
GEMMA4_UPSTREAM: Mapping[str, str] = {
    "e2b": "google/gemma-4-E2B-it",
    "e4b": "google/gemma-4-E4B-it",
    "12b": "google/gemma-4-12B-it",
}

#: Model → the checkpoint its weights are read from, for a model whose weights do not come from its
#: {@link GEMMA4_UPSTREAM} repository. 12B's are Google's quantization-aware-trained (QAT) Q4_0
#: checkpoint — "half-precision weights extracted from the QAT pipeline, for custom downstream
#: compilation", whose README front matter names `google/gemma-4-12B-it` as `base_model` and says
#: `license: apache-2.0`. {@link GEMMA4_UPSTREAM} stays the base model, and the source of the
#: tokenizer, the chat template and the generation defaults (`gemma4.sources.HUB_PINS`).
GEMMA4_WEIGHTS_UPSTREAM: Mapping[str, str] = {"12b": "google/gemma-4-12B-it-qat-q4_0-unquantized"}

#: Model → Google's q4_0 quantization-aware-trained checkpoint of its `-it` model (README front
#: matter at the pinned revisions: `license: apache-2.0`, `base_model` the regular `-it` model).
#: Same architecture and config as the regular checkpoint; every 2-D text weight lies on a q4_0
#: grid (`gemma4.export_product_qat`). Unlike the regular checkpoints, the KV-shared layers carry
#: no k / v residues. Read by `gemma4.sources.QAT_PINS` and attributed by the q4_0 QAT line's card
#: ({@link render_gemma4_qat_q4_0_model_card}).
GEMMA4_QAT_WEIGHTS_UPSTREAM: Mapping[str, str] = {
    "e2b": "google/gemma-4-E2B-it-qat-q4_0-unquantized",
    "e4b": "google/gemma-4-E4B-it-qat-q4_0-unquantized",
}

#: Model → the weights of its q4_0 QAT line (the line `karume-gemma4-qat` ships): E2B / E4B's
#: QAT checkpoints ({@link GEMMA4_QAT_WEIGHTS_UPSTREAM}) and 12B's one line, whose weights are
#: already its QAT checkpoint ({@link GEMMA4_WEIGHTS_UPSTREAM} — `gemma4.sources.QAT_TARGET_PINS`
#: pairs them the same way).
GEMMA4_QAT_Q4_0_WEIGHTS_UPSTREAM: Mapping[str, str] = {
    **GEMMA4_QAT_WEIGHTS_UPSTREAM,
    "12b": GEMMA4_WEIGHTS_UPSTREAM["12b"],
}

#: Model → Google's q4_0 quantization-aware-trained MTP drafter (the "assistant") of its QAT
#: target ({@link GEMMA4_QAT_Q4_0_WEIGHTS_UPSTREAM}). Google's model card: with a QAT target, the
#: assistant "must also be a QAT checkpoint with the same precision" — the regular E2B assistant
#: ({@link GEMMA4_DRAFTER_UPSTREAM}) does not pair with the QAT line. Read by
#: `gemma4.sources.QAT_DRAFTER_PINS`; the q4_0 QAT line ships every one of them.
GEMMA4_QAT_DRAFTER_UPSTREAM: Mapping[str, str] = {
    "e2b": "google/gemma-4-E2B-it-qat-q4_0-unquantized-assistant",
    "e4b": "google/gemma-4-E4B-it-qat-q4_0-unquantized-assistant",
    "12b": "google/gemma-4-12B-it-qat-q4_0-unquantized-assistant",
}

#: Models whose text decoder carries per-layer embeddings (PLE — container assets the host
#: gathers). 12B has none (`text_config.hidden_size_per_layer_input` is 0 and its class,
#: `gemma4_unified`, has no PLE branch). A checkpoint fact the manifest does not carry (the PLE
#: sits inside the `model` container), so the card keeps it here; `gemma4.distribution.gemma4_plan`
#: checks it against the upstream config of every model it assembles, so the two cannot drift.
GEMMA4_PLE_MODELS: frozenset[str] = frozenset({"e2b", "e4b"})

#: The storage of the upstream checkpoints' weights (BF16). A manifest `weights` dtype label other
#: than this one is a weight re-encoded in another storage form ({@link _gemma4_relation}); the
#: regular line's main model stores exactly this one ({@link _gemma4_storage_lines}).
GEMMA4_CHECKPOINT_DTYPE = "bf16"

#: Model → what the upstream checkpoint carries besides the text decoder (never read here, never
#: shipped). E2B / E4B hold vision and audio encoder towers; 12B is encoder-free and projects image
#: patches and audio frames into the embedding space through lightweight linear layers (upstream
#: model card at the pinned revision). Key set = {@link GEMMA4_UPSTREAM} (tests).
GEMMA4_FRONT_ENDS: Mapping[str, str] = {
    "e2b": "vision and audio towers",
    "e4b": "vision and audio towers",
    "12b": "encoder-free vision and audio embedders",
}

#: MTP drafter の weights の役割名（manifest の `weights` のキー）。カードは「drafter を
#: 運んでいるモデル」を manifest のこのキーから引くので、綴りの正本をここに置き、
#: `gemma4.distribution` が import する（distribution → card の一方向を保つため）。
GEMMA4_DRAFTER_ROLE = "drafter"
#: The main model's weights role (the spelling `gemma4.distribution.GEMMA4_ROLE` imports — the card
#: reads the storage of the main model from it, {@link _gemma4_storage_lines}).
GEMMA4_MODEL_ROLE = "model"

#: モデル名 → MTP drafter の上流（ADR 0096 段 2）。帰属表（モデル名 → 上流）とは別の席で
#: 持つ — `e2b` 1 モデルが 2 つの上流チェックポイントから出来ているため。**この表に無い
#: モデルは drafter を配らない**（`gemma4.distribution` が drafter 役の有無をここから決める —
#: E4B の drafter は未対応）。再配布している以上 Apache 2.0 §4(b) の告知対象で、綴りの正本は
#: ここ 1 箇所（`gemma4.export_drafter.ASSISTANT_REPO` / `NOTICE.md` の散文と同じもの）。
GEMMA4_DRAFTER_UPSTREAM: Mapping[str, str] = {"e2b": "google/gemma-4-E2B-it-assistant"}

#: ライセンス（実地確認 2026-09-01 — チェックポイント snapshot の README frontmatter が
#: `license: apache-2.0` を名乗り、その `license_link` 先が Apache 2.0 の本文を載せている）。
GEMMA4_LICENSE = "apache-2.0"
GEMMA4_LICENSE_LINK = "https://ai.google.dev/gemma/docs/gemma_4_license"
GEMMA4_LICENSE_TEXT_LINK = "https://www.apache.org/licenses/LICENSE-2.0"


def _gemma4_metadata(manifest: Mapping[str, Any]) -> CardMetadata:
    """frontmatter を manifest に並んだモデルから組む（`base_model` は再配布する上流の全部）。

    MUST: 未知のモデル名では描かない — 帰属の表に無いモデルを黙って落とすと、`base_model` が
    1 つ足りないカード（= 出所を名乗っていない再配布）が静かに出る。
    """
    upstream: list[str] = []
    for name in manifest["models"]:
        if name not in GEMMA4_UPSTREAM:
            raise ValueError(
                f"モデル '{name}' の上流が帰属表に無い（既知: {sorted(GEMMA4_UPSTREAM)}）"
                " — 出所を名乗れないカードは描かない"
            )
        upstream.append(GEMMA4_UPSTREAM[name])
    # MUST: drafter の上流も `base_model` に載せる — 再配布しているのは 2 本で、片方を
    # 落とすと「出所を名乗っていない再配布」になる（`GEMMA4_DRAFTER_UPSTREAM` の MUST）。
    # 逆に drafter を運んでいないモデルの drafter は名乗らない（配っていない上流を出所に書かない）。
    upstream.extend(_gemma4_drafter_upstreams(manifest))
    return CardMetadata(
        pipeline_tag=GEMMA4_PIPELINE_TAG,
        base_model=tuple(upstream),
        base_model_relation=_gemma4_relation(manifest),
        license=GEMMA4_LICENSE,
        # NOTE: `license_link` は HF が `license: other` のときに読む席だが、**上流カード自身が
        # SPDX 識別子と併記している**綴りへ揃える（読み手が上流と同じ 1 本のライセンス URL へ
        # 辿り着けることを優先した — 2026-09-01 のライセンス方針）。
        license_link=GEMMA4_LICENSE_LINK,
        tags=(GEMMA4_PIPELINE_TAG, "webgpu", "gemma", "llm"),
    )


def _gemma4_relation(manifest: Mapping[str, Any]) -> str | None:
    """`base_model_relation`: `quantized` when the manifest stores some weight in another storage
    than the checkpoints' own bf16 (E2B's int8 drafter), else none — a distribution that keeps the
    storage form is none of the Hub's relations (`karume.modelcard.CardMetadata`)."""
    stored = {
        dtype
        for model in manifest["models"].values()
        for dtypes in model["weights"].values()
        for dtype in dtypes
    }
    return "quantized" if stored - {GEMMA4_CHECKPOINT_DTYPE} else None


def _gemma4_drafter_upstreams(manifest: Mapping[str, Any]) -> list[str]:
    """manifest が drafter を**実際に運んでいる**モデルの drafter 上流（モデルの並び順）。

    drafter の有無はモデルごとに違う（E4B は drafter 無し）ので、有無は manifest の
    `weights` から引く（カードの数値・事実は manifest から導く MUST — モジュール docstring）。

    MUST: drafter を運んでいるのに上流表に無いモデルでは描かない — 出所を名乗れない
    再配布になる（{@link _gemma4_metadata} の帰属表の門と同じ理由）。
    """
    upstream: list[str] = []
    for name, model in manifest["models"].items():
        if GEMMA4_DRAFTER_ROLE not in model["weights"]:
            continue
        if name not in GEMMA4_DRAFTER_UPSTREAM:
            raise ValueError(
                f"モデル '{name}' は drafter を運んでいるが、drafter の上流が表に無い"
                f"（既知: {sorted(GEMMA4_DRAFTER_UPSTREAM)}）— 出所を名乗れないカードは描かない"
            )
        upstream.append(GEMMA4_DRAFTER_UPSTREAM[name])
    return upstream


def _gemma4_upstream_spellings(
    manifest: Mapping[str, Any], upstream: Mapping[str, str] = GEMMA4_UPSTREAM
) -> str:
    """散文が名乗る上流の綴り — **この配布形が実際に運んでいるモデル**だけを manifest から引く。

    `upstream` is the line's attribution table (the base models for the regular line, the QAT
    checkpoints for the q4_0 QAT line — {@link GEMMA4_QAT_Q4_0_WEIGHTS_UPSTREAM}).

    MUST: 散文にモデルの綴りを焼かない（{@link GEMMA4_TITLE} の MUST と同じ理由）。
    """
    return " / ".join(f"`{upstream[name]}`" for name in manifest["models"])


def _spell_models(names: Iterable[str]) -> str:
    return " / ".join(f"`{name}`" for name in names)


def _gemma4_ple_bullet(manifest: Mapping[str, Any]) -> list[str]:
    """The overview's per-layer-embedding bullet, per model when the manifest mixes PLE and
    PLE-free models (the text of an all-PLE manifest — E2B / E4B — is the one it always was)."""
    names = list(manifest["models"])
    carriers = [name for name in names if name in GEMMA4_PLE_MODELS]
    if carriers == names:
        return [
            "- **Per-layer embeddings are container assets, not graph weights.** They are gathered",
            "  on the host and handed to the graph as an input, which keeps them out of the GPU's",
            "  resident set; the loader reads only the vocabulary ranges a conversation actually",
            "  touches.",
        ]
    others = _spell_models(name for name in names if name not in GEMMA4_PLE_MODELS)
    if not carriers:
        return [
            f"- **No per-layer embeddings**: {others} has none, so every weight is a graph weight."
        ]
    return [
        f"- **Per-layer embeddings** ({_spell_models(carriers)}) **are container assets, not graph",
        "  weights.** They are gathered on the host and handed to the graph as an input, which",
        "  keeps them out of the GPU's resident set; the loader reads only the vocabulary ranges a",
        f"  conversation actually touches. {others} has no per-layer embeddings.",
    ]


def _gemma4_front_end_bullet(manifest: Mapping[str, Any]) -> list[str]:
    """The overview's text-only bullet (one sentence when every model's checkpoint carries the
    same kind of multimodal front end — the E2B / E4B text — else one clause per kind)."""
    by_kind: dict[str, list[str]] = {}
    for name in manifest["models"]:
        by_kind.setdefault(GEMMA4_FRONT_ENDS[name], []).append(name)
    if len(by_kind) == 1:
        return [
            "- **Not multimodal here.** The published checkpoint also carries"
            f" {next(iter(by_kind))};",
            "  this distribution contains neither, and cannot take images or audio.",
        ]
    kinds = "; ".join(f"{kind} for {_spell_models(names)}" for kind, names in by_kind.items())
    return [
        "- **Not multimodal here.** The published checkpoints also carry multimodal front ends",
        f"  ({kinds}); this distribution contains none of them, and cannot take images or audio.",
    ]


def _gemma4_position_limits(manifest: Mapping[str, Any]) -> str:
    """The overview's position limit: one number when every model declares the same (the E2B /
    E4B text), else one per model (12B declares 262144, E2B / E4B 131072)."""
    by_limit: dict[int, list[str]] = {}
    for name, model in manifest["models"].items():
        by_limit.setdefault(model["pipelineConfig"]["maxPosition"], []).append(name)
    if len(by_limit) == 1:
        return f"{next(iter(by_limit))} (the model's declared position limit)"
    limits = ", ".join(f"{limit} for {_spell_models(names)}" for limit, names in by_limit.items())
    return f"the model's declared position limit ({limits})"


def _gemma4_overview(
    manifest: Mapping[str, Any],
    upstream: Mapping[str, str] = GEMMA4_UPSTREAM,
    extra: Sequence[str] = (),
) -> list[str]:
    """The overview; `upstream` names the line's checkpoints ({@link _gemma4_upstream_spellings})
    and `extra` is the line's own bullets, placed after the tokenizer bullet."""
    config = default_model(manifest)["pipelineConfig"]
    return [
        "## What is this",
        "",
        f"A chat distribution: the **text decoder** of"
        f" {_gemma4_upstream_spellings(manifest, upstream)},",
        "converted into the WebGPU inference runtime **Karume**'s container format (a `.krm` part",
        "sequence whose first part carries the graph and model descriptors). Runs as-is in the",
        "browser and in Deno — string in, string out.",
        "",
        "- **Tokenizer and chat format ship with the weights.** `chat()` renders the turns,",
        "  encodes them, samples, and decodes incrementally, so callers hand over messages and",
        "  read back text fragments as they are decided.",
        *extra,
        *_gemma4_ple_bullet(manifest),
        f"- **Context**: the default capacity is {config['capacity']} tokens per conversation, and",
        "  the loader accepts any capacity between the chunk length in use and",
        f"  {_gemma4_position_limits(manifest)}; rotary cos/sin are",
        "  generated on the host per chunk from the declared `rope` parameters, so no position",
        f"  table ships. Prefill runs in chunks of {config['chunkLength']} rows by default, and",
        f"  the chunk length itself can be raised up to {config['maxChunkLength']} — the traced",
        "  upper bound of the chunk symbol, which the graph does not carry.",
        *_gemma4_front_end_bullet(manifest),
        "- Not readable by transformers (it's a different container with an embedded graph); the",
        f"  reader is a pipeline that implements `{GEMMA4_SUPPORTED_PIPELINE}`.",
        f"- Exporter used for the conversion: `{manifest['generator']}`. The distribution manifest",
        f"  is `karume.json` (`{manifest['format']}`).",
    ]


def _gemma4_storage_lines(manifest: Mapping[str, Any]) -> list[str]:
    """The attribution's opening sentence: the main model's weights ship as the checkpoint's own
    bf16 values. MUST: a manifest whose `model` role stores anything else is not drawn — the
    sentence would state a storage the distribution does not carry."""
    dtypes = {
        dtype
        for model in manifest["models"].values()
        for dtype in model["weights"][GEMMA4_MODEL_ROLE]
    }
    if dtypes != {GEMMA4_CHECKPOINT_DTYPE}:
        raise ValueError(
            f"the main model's weights are stored as {sorted(dtypes)} — this card states the"
            f" checkpoint's {GEMMA4_CHECKPOINT_DTYPE} values unchanged"
        )
    return [
        "Converted from the upstream checkpoint, its weights shipped unchanged (bf16) — the"
        " original",
        "checkpoint files are not distributed here.",
    ]


def _gemma4_base_weights(manifest: Mapping[str, Any]) -> list[str]:
    """帰属節。使われ方（能力・制限・安全性・評価）は**上流カードへ誘導する**。"""
    lines = [
        "## Base weights and attribution",
        "",
        *_gemma4_storage_lines(manifest),
        "",
    ]
    for name in manifest["models"]:
        repo = GEMMA4_UPSTREAM[name]
        lines.append(
            f"- **`{name}`**: [{repo}](https://huggingface.co/{repo}), licensed"
            f" **Apache 2.0** ([license]({GEMMA4_LICENSE_LINK}) /"
            f" [full text]({GEMMA4_LICENSE_TEXT_LINK}); a verbatim copy is in `LICENSE.md`)."
        )
    for drafter in _gemma4_drafter_upstreams(manifest):
        lines.append(
            f"- **`{drafter.rsplit('-', 1)[-1]}`**:"
            f" [{drafter}](https://huggingface.co/{drafter}), the"
            " multi-token-prediction drafter head, licensed **Apache 2.0** under the same terms."
            " It is redistributed here with its clustered sparse output head replaced by a dense"
            " projection over the full vocabulary, and it reads the key/value states and embedding"
            " table of the main model rather than carrying its own. Its own weights are quantized"
            " to int8 throughout, linear layers included."
        )
    lines += [
        *_GEMMA4_CHANGES,
        "  No retraining and no fine-tuning.",
        "- **What the model can and cannot do** — capabilities, limitations, intended use,",
        "  evaluations and safety guidance — is documented on the upstream model card, and this",
        f"  repository does not restate it: read [{GEMMA4_UPSTREAM[manifest['defaultModel']]}]"
        f"(https://huggingface.co/{GEMMA4_UPSTREAM[manifest['defaultModel']]}) before using it.",
    ]
    return lines


#: The "Changes made here" item (the regular line's models all carry per-layer embeddings).
_GEMMA4_CHANGES: tuple[str, ...] = (
    "- **Changes made here** (also listed in `NOTICE.md`, per Apache 2.0 §4(b)): the text",
    "  decoder was extracted and re-expressed in the Karume container format; the weights keep",
    "  the checkpoint's bf16 values unchanged; the per-layer embeddings were moved out of the",
    "  graph into container assets that the host gathers; the exit was narrowed to the selected",
    "  rows' logits and hidden states; rotary cos/sin arrive as host-generated inputs.",
)


def _gemma4_usage(
    manifest: Mapping[str, Any], repo: str, extra_options: Sequence[str] = ()
) -> list[str]:
    """Usage 例。**動く最小形**（`fromPretrained` → `chat`）を 1 本だけ出し、普通に触りそうな
    optional はコメントアウトで併記する（選べる値は manifest から機械導出）。

    `extra_options` is the line's own commented-out option lines, after `model` / `quant`.

    NOTE: `fromAssets` は案内しない（2026-08-29 裁定 — 全 family 共通）。
    """
    model_name = manifest["defaultModel"]
    model = default_model(manifest)
    quant = model["defaultQuant"]
    sampler = model["pipelineConfig"]["sampler"]
    model_names = " / ".join(sorted(manifest["models"]))
    quant_names = " / ".join(sorted(model["quants"]))
    return [
        "## Usage",
        "",
        "```ts",
        'import { Pipeline } from "jsr:@karume/models/gemma";',
        "",
        *from_pretrained(
            "Pipeline",
            repo,
            [
                f'  // model: "{model_name}", // default — available: {model_names}',
                f'  // quant: "{quant}", // default — available: {quant_names}',
                *extra_options,
            ],
            disposable="await using",
        ),
        "",
        "const stream = pipeline.chat([",
        '  { role: "user", content: "What is the capital of France?" },',
        "], {",
        "  maxNewTokens: 128,",
        "  // sampler: { temperature: 0 }, // greedy — overrides the recommended default below",
        "});",
        "",
        "// Fragments arrive as the decoder settles them (multi-byte characters are held back",
        "// until they are complete).",
        "for await (const text of stream) Deno.stdout.write(new TextEncoder().encode(text));",
        'console.log(await stream.done); // { reason: "eos" | "max-tokens" | "aborted", … }',
        "```",
        "",
        "Messages are plain `system` / `developer` / `user` / `assistant` turns; tool calls,",
        "thinking channels and image or audio parts are rejected rather than silently dropped.",
        "With no `sampler` in the request, generation uses this repository's recommended default:"
        f" temperature {knob(sampler['temperature'])}, top-k {knob(sampler['topK'])},"
        f" top-p {knob(sampler['topP'])}.",
        "Weights are fetched once and cached (verified against `karume.json`'s `size` / `sha256`).",
    ]


def _gemma4_device_limits(quants: Mapping[str, Mapping[str, Any]]) -> str:
    """`requiredLimits` を上限名ごとに畳む（全 quant で値が同じなら名前 1 回・値 1 つ）。

    値が quant 間で割れたとき（その上限を宣言しない quant があるときも含む）だけ、値ごとに
    quant 名を添える — 畳んだ 1 つの値を「全席の上限」と読ませない。
    """
    names = dict.fromkeys(
        name for quant in quants.values() for name in quant.get("requiredLimits", {})
    )
    parts: list[str] = []
    for name in names:
        holders: dict[int | None, list[str]] = {}
        for quant_name, quant in quants.items():
            holders.setdefault(quant.get("requiredLimits", {}).get(name), []).append(quant_name)
        if len(holders) == 1:
            parts.append(f"`{name}` ≥ {next(iter(holders)):,} B")
            continue
        split = "; ".join(
            f"≥ {value:,} B for {', '.join(f'`{quant_name}`' for quant_name in quant_names)}"
            for value, quant_names in holders.items()
            if value is not None
        )
        parts.append(f"`{name}` {split}")
    return " / ".join(parts)


def _gemma4_generation(model: Mapping[str, Any]) -> list[str]:
    """この配布形の生成側の宣言（利用者が渡すもの・上限・推奨既定がここで読める）。"""
    config = model["pipelineConfig"]
    sampler = config["sampler"]
    limits = _gemma4_device_limits(model["quants"])
    lines = [
        "### Generation",
        "",
        "Derived from the exported graph and the checkpoint's own `generation_config.json`, and",
        "checked against each other when this repository was assembled.",
        "",
        f"- **context**: {config['capacity']} tokens per conversation (prompt + generated) by"
        f" default; any capacity from the chunk length in use up to {config['maxPosition']}"
        " can be chosen at load time",
        f"- **prefill chunk**: {config['chunkLength']} rows per step by default, up to"
        f" {config['maxChunkLength']} (the traced upper bound of the chunk symbol)",
        f"- **recommended sampler**: temperature {knob(sampler['temperature'])}, top-k"
        f" {knob(sampler['topK'])}, top-p {knob(sampler['topP'])} — used when a request omits",
        "  `sampler`; pass `{ temperature: 0 }` for greedy decoding",
        "- **stop tokens** come from the tokenizer asset in this repository, not from the caller",
    ]
    if limits:
        lines.append(f"- **device limits**: {limits} (the largest single tensor must bind)")
    return lines


def render_gemma4_model_card(
    manifest: Mapping[str, Any], repo: str, host_assets: Mapping[FileRefKey, int] = {}
) -> str:
    """gemma4 配布形の `README.md` 本文を組み立てる（純関数・末尾改行つき）。"""
    require_pipeline(manifest, GEMMA4_SUPPORTED_PIPELINE)
    metadata = _gemma4_metadata(manifest)
    return render(
        (
            frontmatter(metadata),
            introduction(GEMMA4_TITLE, hf_upstreams(metadata.base_model)),
            _gemma4_overview(manifest),
            [""],
            _gemma4_base_weights(manifest),
            [""],
            models(manifest),
            [""],
            _gemma4_usage(manifest, repo),
            *model_sections(
                manifest, (partial(quants, host_assets=host_assets), _gemma4_generation)
            ),
        )
    )


# ---------------------------------------------------------------------------
# The q4_0 QAT line (`karume-gemma4-qat` — `gemma4.distribution.QAT_Q4_0_PIPELINE`)
# ---------------------------------------------------------------------------

#: The q4_0 QAT line's title (the regular title's MUST: no model spelling).
GEMMA4_QAT_Q4_0_TITLE = "Gemma 4 q4_0 QAT Text Decoder — Karume"


def _qat_q4_0_weights(name: str) -> str:
    """The QAT checkpoint model `name`'s weights come from; a model the line does not know fails
    the render (a card that cannot name its source is not drawn)."""
    weights = GEMMA4_QAT_Q4_0_WEIGHTS_UPSTREAM.get(name)
    if weights is None:
        raise ValueError(
            f"model '{name}' has no q4_0 QAT checkpoint in the attribution table"
            f" (known: {sorted(GEMMA4_QAT_Q4_0_WEIGHTS_UPSTREAM)}) — a card that cannot name its"
            " source is not drawn"
        )
    return weights


def _qat_q4_0_drafters(manifest: Mapping[str, Any]) -> list[tuple[str, str]]:
    """(model, QAT drafter upstream) for every model whose manifest entry carries a drafter, in
    the manifest's order (the regular card's rule: the manifest says which drafters ship)."""
    drafters: list[tuple[str, str]] = []
    for name, model in manifest["models"].items():
        if GEMMA4_DRAFTER_ROLE not in model["weights"]:
            continue
        upstream = GEMMA4_QAT_DRAFTER_UPSTREAM.get(name)
        if upstream is None:
            raise ValueError(
                f"model '{name}' carries a drafter, but no QAT drafter upstream is listed"
                f" (known: {sorted(GEMMA4_QAT_DRAFTER_UPSTREAM)}) — a card that cannot name its"
                " source is not drawn"
            )
        drafters.append((name, upstream))
    return drafters


def _qat_q4_0_metadata(manifest: Mapping[str, Any]) -> CardMetadata:
    """Every redistributed upstream: per model its base model, then its QAT checkpoint (the QAT
    checkpoint names the base model its `base_model`, so the base model comes first), then the
    QAT drafters the manifest carries."""
    upstream: list[str] = []
    for name in manifest["models"]:
        weights = _qat_q4_0_weights(name)
        upstream += [GEMMA4_UPSTREAM[name], weights]
    upstream += [drafter for _, drafter in _qat_q4_0_drafters(manifest)]
    return CardMetadata(
        pipeline_tag=GEMMA4_PIPELINE_TAG,
        base_model=tuple(upstream),
        # The stored int4 codes and steps are refitted from the checkpoint's BF16 values of a
        # quantization-aware-trained model — a quantized form of the base model.
        base_model_relation="quantized",
        license=GEMMA4_LICENSE,
        license_link=GEMMA4_LICENSE_LINK,
        tags=(GEMMA4_PIPELINE_TAG, "webgpu", "gemma", "llm", "qat"),
    )


def _qat_q4_0_overview_bullets(manifest: Mapping[str, Any]) -> list[str]:
    """The line's own overview bullets: the stored grid and the drafters (each derived from the
    manifest's models)."""
    names = list(manifest["models"])
    carriers = [name for name in names if name in GEMMA4_PLE_MODELS]
    stored = [
        "the linear weights",
        "the embedding table (also the output head)",
        *([f"the per-layer embeddings of {_spell_models(carriers)}"] if carriers else []),
    ]
    listed = f"{', '.join(stored[:-1])} and {stored[-1]}"
    lines = [
        "- **Every weight keeps the checkpoint's own 4-bit grid.**"
        f" {listed[0].upper()}{listed[1:]} are stored as the q4_0 codes",
        "  with one step per group of 32 values (packed int4) — not rounded again.",
    ]
    drafters = [name for name, _ in _qat_q4_0_drafters(manifest)]
    if drafters:
        lines += [
            f"- **Speculative decoding**: {_spell_models(drafters)} also ship Google's QAT",
            "  multi-token-prediction drafter, which borrows the main model's key/value states and",
            "  embedding table; pass `speculative: {}` when loading to use it.",
        ]
    return lines


def _qat_q4_0_base_weights(manifest: Mapping[str, Any]) -> list[str]:
    """The attribution section: each model's QAT checkpoint and base model, each drafter, the
    changes made, and a pointer to the upstream card (the regular card's rule)."""
    lines = [
        "## Base weights and attribution",
        "",
        "Converted from Google's quantization-aware-trained (QAT, Q4_0) checkpoints — the original",
        "checkpoints are not distributed here.",
        "",
    ]
    license_links = (
        f"([license]({GEMMA4_LICENSE_LINK}) / [full text]({GEMMA4_LICENSE_TEXT_LINK}); a verbatim"
        " copy is in `LICENSE.md`)"
    )
    for name in manifest["models"]:
        weights, base = _qat_q4_0_weights(name), GEMMA4_UPSTREAM[name]
        lines.append(
            f"- **`{name}`**: the weights are [{weights}](https://huggingface.co/{weights}),"
            " Google's QAT (Q4_0) checkpoint of the base model"
            f" [{base}](https://huggingface.co/{base}), which is also the source of the tokenizer,"
            " the chat template and the generation defaults; both are licensed **Apache 2.0**"
            f" {license_links}."
        )
    for name, drafter in _qat_q4_0_drafters(manifest):
        lines.append(
            f"- **`{name}` drafter**: [{drafter}](https://huggingface.co/{drafter}), the QAT"
            " multi-token-prediction drafter head, licensed **Apache 2.0** under the same terms."
            " It reads the key/value states and the embedding table of the main model rather than"
            " carrying its own, and runs its output head over the full vocabulary (a clustered"
            " sparse head an assistant checkpoint carries is not used)."
        )
    carriers = [name for name in manifest["models"] if name in GEMMA4_PLE_MODELS]
    ple = (
        f" the per-layer embeddings of {_spell_models(carriers)} were moved out of the graph into"
        " container assets that the host gathers, on the same grid;"
        if carriers
        else ""
    )
    lines += [
        "- **Changes made here** (also listed in `NOTICE.md`, per Apache 2.0 §4(b)): the text",
        "  decoder was extracted and re-expressed in the Karume container format; the linear",
        "  weights and the embedding table keep the checkpoint's own 4-bit grid, stored as packed",
        "  int4 (group 32: its codes and one step per group, refitted from the stored values)",
        f"  rather than rounded again;{ple} the exit was narrowed to the selected rows' logits and",
        "  hidden states; rotary cos/sin arrive as host-generated inputs; the drafters' weights",
        "  keep their own grid the same way.",
        "  No retraining and no fine-tuning.",
        "- **What the model can and cannot do** — capabilities, limitations, intended use,",
        "  evaluations and safety guidance — is documented on the upstream model card, and this",
        f"  repository does not restate it: read [{GEMMA4_UPSTREAM[manifest['defaultModel']]}]"
        f"(https://huggingface.co/{GEMMA4_UPSTREAM[manifest['defaultModel']]}) before using it.",
    ]
    return lines


def render_gemma4_qat_q4_0_model_card(
    manifest: Mapping[str, Any], repo: str, host_assets: Mapping[FileRefKey, int] = {}
) -> str:
    """The q4_0 QAT line's `README.md` (a pure function, trailing newline): the regular card's
    overview, usage and per-model sections, with the line's own title, attribution and bullets."""
    require_pipeline(manifest, GEMMA4_SUPPORTED_PIPELINE)
    drafters = [name for name, _ in _qat_q4_0_drafters(manifest)]
    speculative = (
        [f"  // speculative: {{}}, // load the MTP drafter — available for: {' / '.join(drafters)}"]
        if drafters
        else []
    )
    metadata = _qat_q4_0_metadata(manifest)
    return render(
        (
            frontmatter(metadata),
            introduction(GEMMA4_QAT_Q4_0_TITLE, hf_upstreams(metadata.base_model)),
            _gemma4_overview(
                manifest, GEMMA4_QAT_Q4_0_WEIGHTS_UPSTREAM, _qat_q4_0_overview_bullets(manifest)
            ),
            [""],
            _qat_q4_0_base_weights(manifest),
            [""],
            models(manifest),
            [""],
            _gemma4_usage(manifest, repo, speculative),
            *model_sections(
                manifest, (partial(quants, host_assets=host_assets), _gemma4_generation)
            ),
        )
    )
