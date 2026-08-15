"""Anima 配布形のモデルカード（`README.md`）— manifest から機械導出する純関数。

汎用の描画部品（frontmatter・モデル一覧・ファイル表・quant 表・節の組み立て）は
`karume.modelcard` が持つ。ここが持つのは **Anima 固有の事実**だけ: 帰属
（base model・ライセンス・焼き込んだ LoRA の出所）と、この pipeline のカードに何を書くか。

MUST: **数値・ファイル一覧・quant 表・dtype ラベルは 1 つ残らず manifest から導出する**
（`karume.modelcard` の同 MUST がそのまま掛かる）。ここが持ってよい定数は、manifest に
**存在しない事実**だけ。

NOTE: `_frontmatter` などの描画部品は core 側で private 名のまま — recipe から名指しで
呼ぶのは ADR 0065 段 6（packaging）で公開名を決めるまでの形。
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from karume.modelcard import (
    CardMetadata,
    _default_model,
    _files,
    _frontmatter,
    _model_sections,
    _models,
    _quants,
    _render,
    _require_pipeline,
)

#: このテンプレートが説明できるパイプライン契約（ADR 0041 §2 — モデル単位）。
SUPPORTED_PIPELINE = "anima/1"

PIPELINE_TAG = "text-to-image"

#: ライセンス（実地確認）: base model のリポジトリ自身は license を宣言せず、カードは
#: 「元のモデルカードを見よ」として `circlestone-labs/Anima` を指す。実値はそちらにある。
ANIMA_METADATA = CardMetadata(
    pipeline_tag=PIPELINE_TAG,
    base_model=("circlestone-labs/Anima-Base-v1.0-Diffusers",),
    base_model_relation="quantized",
    license="other",
    license_name="circlestone-labs-non-commercial-license",
    license_link="https://huggingface.co/circlestone-labs/Anima/blob/main/LICENSE.md",
    tags=("text-to-image", "webgpu"),
)

ANIMA_TITLE = "Anima Turbo — Karume"

#: 焼き込んだ LoRA（manifest には現れない — 重みの中に畳まれているため）。
LORA_NAME = "Anima Turbo LoRA v0.2"
LORA_AUTHOR = "circlestone_labs"
LORA_SOURCE = "https://civitai.com/models/2560840?modelVersionId=2979642"
LORA_FILE = "anima-turbo-lora-v0.2.safetensors"
LORA_SHA256 = "1b55e40bdb1d0e5a78cb498f245fccfdaae97823265db957d2aabdcf4cd3caf1"
#: 出所の権限欄（実地確認: https://civitai.com/api/v1/models/2560840）。
LORA_PERMISSIONS = (
    ("allowNoCredit", "true"),
    ("allowCommercialUse", "Image / RentCivit / Rent"),
    ("allowDerivatives", "true"),
    ("allowDifferentLicense", "true"),
)


def _overview(manifest: Mapping[str, Any]) -> list[str]:
    defaults = _default_model(manifest)["pipelineConfig"]["defaults"]
    base_model = ANIMA_METADATA.base_model[0]
    return [
        "## What is this",
        "",
        f"A distribution that bakes **{LORA_NAME}** into"
        f" [{base_model}](https://huggingface.co/{base_model}) and converts it into the WebGPU",
        "inference runtime **Karume**'s container format (a single safetensors file = weights +",
        "a graph JSON embedded in `__metadata__`). Runs as-is in the browser and in Deno.",
        "",
        f"- A few-step distillation (from the LoRA) tuned for **{defaults['steps']} steps /"
        f" guidance {defaults['guidanceScale']}**.",
        "- Not readable by diffusers (it's a different container with an embedded graph); the"
        f" reader is a pipeline that implements `{SUPPORTED_PIPELINE}`.",
        f"- Exporter used for the conversion: `{manifest['generator']}`. The distribution manifest"
        f" is `karume.json` (`{manifest['format']}`).",
    ]


def _merged_lora() -> list[str]:
    return [
        "## Baked-in LoRA",
        "",
        "Folded into the weights — not distributed as a separate file.",
        "",
        f"- **Name**: {LORA_NAME}",
        f"- **Author**: {LORA_AUTHOR} (same author as the base model)",
        f"- **Source**: {LORA_SOURCE}",
        f"- **File**: `{LORA_FILE}`",
        f"- **sha256**: `{LORA_SHA256}`",
        "",
        "Permissions listed on the source page (as of retrieval):",
        "",
        *(f"- `{name}`: {value}" for name, value in LORA_PERMISSIONS),
    ]


def _usage(manifest: Mapping[str, Any], repo: str) -> list[str]:
    model = manifest["defaultModel"]
    quant = _default_model(manifest)["defaultQuant"]
    return [
        "## Usage",
        "",
        "```ts",
        'import { AnimaPipeline, encodePng } from "jsr:@karume/models";',
        "",
        f"// Both options may be omitted: model defaults to {model}, quant to {quant}.",
        f'using pipeline = await AnimaPipeline.fromPretrained("{repo}", {{',
        f'  model: "{model}",',
        f'  quant: "{quant}",',
        "});",
        "const image = await pipeline.generate({",
        '  prompt: "1girl, solo, long hair, blue eyes, school uniform, masterpiece",',
        "  seed: 42,",
        "});",
        "const png = await encodePng(image.data, image.width, image.height);",
        'await Deno.writeFile("anima.png", png);',
        "```",
        "",
        "Weights are fetched once and cached (verified against `karume.json`'s `size` / `sha256`).",
        "You can also load from a local directory (`AnimaPipeline.fromAssets`).",
    ]


def _defaults(model: Mapping[str, Any]) -> list[str]:
    defaults = model["pipelineConfig"]["defaults"]
    resolution = defaults["resolution"]
    guidance = defaults["guidanceScale"]
    lines = [
        "### Defaults",
        "",
        "Any knob not passed to `generate()` is filled in from the manifest's defaults.",
        "",
        f"- **steps**: {defaults['steps']}",
        f"- **guidanceScale**: {guidance}",
        f"- **resolution**: {resolution['width']} × {resolution['height']}",
        f"- **negativePrompt**: `{defaults['negativePrompt']}`",
    ]
    if guidance == 1:
        lines += [
            "",
            f"At guidance {guidance}, the second CFG branch is skipped, so"
            " **the negative prompt is not used**",
            "(it only takes effect once guidance is raised).",
        ]
    return lines


def render_model_card(manifest: Mapping[str, Any], repo: str) -> str:
    """Anima 配布形の `README.md` 本文を組み立てる（純関数・末尾改行つき）。"""
    _require_pipeline(manifest, SUPPORTED_PIPELINE)
    return _render(
        (
            _frontmatter(ANIMA_METADATA),
            ["", f"# {ANIMA_TITLE}", ""],
            _overview(manifest),
            [""],
            _merged_lora(),
            [""],
            _models(manifest),
            [""],
            _usage(manifest, repo),
            *_model_sections(manifest, (_files, _quants, _defaults)),
        )
    )
