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

from collections.abc import Mapping
from typing import Any

from karume.modelcard import (
    CardMetadata,
    default_model,
    from_pretrained,
    frontmatter,
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
GEMMA4_UPSTREAM: Mapping[str, str] = {"e2b": "google/gemma-4-E2B-it"}

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
    return CardMetadata(
        pipeline_tag=GEMMA4_PIPELINE_TAG,
        base_model=tuple(upstream),
        # 格納形を int4 / int8 へ落とし直しているので `quantized` が事実として当たる。
        base_model_relation="quantized",
        license=GEMMA4_LICENSE,
        # NOTE: `license_link` は HF が `license: other` のときに読む席だが、**上流カード自身が
        # SPDX 識別子と併記している**綴りへ揃える（読み手が上流と同じ 1 本のライセンス URL へ
        # 辿り着けることを優先した — 2026-09-01 のライセンス方針）。
        license_link=GEMMA4_LICENSE_LINK,
        tags=(GEMMA4_PIPELINE_TAG, "webgpu", "gemma", "llm"),
    )


def _gemma4_upstream_spellings(manifest: Mapping[str, Any]) -> str:
    """散文が名乗る上流の綴り — **この配布形が実際に運んでいるモデル**だけを manifest から引く。

    MUST: 散文にモデルの綴りを焼かない（{@link GEMMA4_TITLE} の MUST と同じ理由）。
    """
    return " / ".join(f"`{GEMMA4_UPSTREAM[name]}`" for name in manifest["models"])


def _gemma4_overview(manifest: Mapping[str, Any]) -> list[str]:
    config = default_model(manifest)["pipelineConfig"]
    return [
        "## What is this",
        "",
        f"A chat distribution: the **text decoder** of {_gemma4_upstream_spellings(manifest)},",
        "converted into the WebGPU inference runtime **Karume**'s container format (safetensors",
        "files carrying the graph in `__metadata__`). Runs as-is in the browser and in Deno —",
        "string in, string out.",
        "",
        "- **Tokenizer and chat format ship with the weights.** `chat()` renders the turns,",
        "  encodes them, samples, and decodes incrementally, so callers hand over messages and",
        "  read back text fragments as they are decided.",
        "- **Per-layer embeddings are a sidecar, not graph weights.** They are gathered on the",
        "  host and handed to the graph as an input, which keeps them out of the GPU's resident",
        "  set; the loader reads only the vocabulary ranges a conversation actually touches.",
        f"- **Context**: the default capacity is {config['capacity']} tokens per conversation, and",
        "  the loader accepts any capacity between the chunk length in use and",
        f"  {config['maxPosition']} (the model's declared position limit); rotary cos/sin are",
        "  generated on the host per chunk from the declared `rope` parameters, so no position",
        f"  table ships. Prefill runs in chunks of {config['chunkLength']} rows by default, and",
        f"  the chunk length itself can be raised up to {config['maxChunkLength']} — the traced",
        "  upper bound of the chunk symbol, which the graph does not carry.",
        "- **Not multimodal here.** The published checkpoint also carries vision and audio towers;",
        "  this distribution contains neither, and cannot take images or audio.",
        "- Not readable by transformers (it's a different container with an embedded graph); the",
        f"  reader is a pipeline that implements `{GEMMA4_SUPPORTED_PIPELINE}`.",
        f"- Exporter used for the conversion: `{manifest['generator']}`. The distribution manifest",
        f"  is `karume.json` (`{manifest['format']}`).",
    ]


def _gemma4_base_weights(manifest: Mapping[str, Any]) -> list[str]:
    """帰属節。使われ方（能力・制限・安全性・評価）は**上流カードへ誘導する**。"""
    lines = [
        "## Base weights and attribution",
        "",
        "Converted and quantized from the upstream checkpoint — the original weights are not",
        "distributed here.",
        "",
    ]
    for name in manifest["models"]:
        repo = GEMMA4_UPSTREAM[name]
        lines.append(
            f"- **`{name}`**: [{repo}](https://huggingface.co/{repo}), licensed"
            f" **Apache 2.0** ([license]({GEMMA4_LICENSE_LINK}) /"
            f" [full text]({GEMMA4_LICENSE_TEXT_LINK}); a verbatim copy is in `LICENSE.md`)."
        )
    lines += [
        "- **Changes made here** (also listed in `NOTICE.md`, per Apache 2.0 §4(b)): the text",
        "  decoder was extracted and re-expressed in the Karume container format; linear weights",
        "  were quantized to packed int4 (group 32) and the embedding tables to int8; the",
        "  per-layer embeddings were moved out of the graph into a host-gathered sidecar; the exit",
        "  was narrowed to the last row's logits; rotary cos/sin arrive as host-generated inputs.",
        "  No retraining and no fine-tuning.",
        "- **What the model can and cannot do** — capabilities, limitations, intended use,",
        "  evaluations and safety guidance — is documented on the upstream model card, and this",
        f"  repository does not restate it: read [{GEMMA4_UPSTREAM[manifest['defaultModel']]}]"
        f"(https://huggingface.co/{GEMMA4_UPSTREAM[manifest['defaultModel']]}) before using it.",
    ]
    return lines


def _gemma4_usage(manifest: Mapping[str, Any], repo: str) -> list[str]:
    """Usage 例。**動く最小形**（`fromPretrained` → `chat`）を 1 本だけ出し、普通に触りそうな
    optional はコメントアウトで併記する（選べる値は manifest から機械導出）。

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
        'import { Gemma4Pipeline } from "jsr:@karume/models/gemma";',
        "",
        *from_pretrained(
            "Gemma4Pipeline",
            repo,
            [
                f'  // model: "{model_name}", // default — available: {model_names}',
                f'  // quant: "{quant}", // default — available: {quant_names}',
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
        "Messages are plain `system` / `user` / `assistant` turns; tool calls, thinking channels",
        "and image or audio parts are rejected rather than silently dropped.",
        "With no `sampler` in the request, generation uses this repository's recommended default:"
        f" temperature {knob(sampler['temperature'])}, top-k {knob(sampler['topK'])},"
        f" top-p {knob(sampler['topP'])}.",
        "Weights are fetched once and cached (verified against `karume.json`'s `size` / `sha256`).",
    ]


def _gemma4_generation(model: Mapping[str, Any]) -> list[str]:
    """この配布形の生成側の宣言（利用者が渡すもの・上限・推奨既定がここで読める）。"""
    config = model["pipelineConfig"]
    sampler = config["sampler"]
    limits = " / ".join(
        f"`{name}` ≥ {value:,} B"
        for quant in model["quants"].values()
        for name, value in quant.get("requiredLimits", {}).items()
    )
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


def render_gemma4_model_card(manifest: Mapping[str, Any], repo: str) -> str:
    """gemma4 配布形の `README.md` 本文を組み立てる（純関数・末尾改行つき）。"""
    require_pipeline(manifest, GEMMA4_SUPPORTED_PIPELINE)
    return render(
        (
            frontmatter(_gemma4_metadata(manifest)),
            ["", f"# {GEMMA4_TITLE}", ""],
            _gemma4_overview(manifest),
            [""],
            _gemma4_base_weights(manifest),
            [""],
            models(manifest),
            [""],
            _gemma4_usage(manifest, repo),
            *model_sections(manifest, (quants, _gemma4_generation)),
        )
    )
