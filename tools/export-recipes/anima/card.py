"""Anima 配布形のモデルカード（`README.md`）— manifest から機械導出する純関数。

汎用の描画部品（frontmatter・モデル一覧・ファイル表・quant 表・節の組み立て）は
`karume.modelcard` が持つ。ここが持つのは **Anima 固有の事実**だけ: 帰属
（base model・ライセンス・焼き込んだ LoRA の出所）と、この pipeline のカードに何を書くか。

MUST: **数値・ファイル一覧・quant 表・dtype ラベルは 1 つ残らず manifest から導出する**
（`karume.modelcard` の同 MUST がそのまま掛かる）。ここが持ってよい定数は、manifest に
**存在しない事実**だけ。
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from karume.modelcard import (
    CardMetadata,
    default_model,
    files,
    frontmatter,
    model_sections,
    models,
    quants,
    render,
    require_pipeline,
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

#: 上流ライセンス §3(b) が**逐語での掲示**を求める Attribution Notice（1 字も変えない）。
#: 配布リポ直下の `NOTICE.md`（`anima/distribution.py`）とこのカードの両方がここを引く —
#: 同じ法的文言を 2 箇所で独立に持つと、片方だけが条件を満たさない形へ静かに割れる。
ATTRIBUTION_NOTICE = (
    "The CircleStone Model is licensed by CircleStone Labs LLC under the CircleStone"
    " Non-Commercial License. Copyright CircleStone Labs LLC.\n"
    "IN NO EVENT SHALL CIRCLESTONE LABS LLC BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,"
    " WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION"
    " WITH USE OF THIS MODEL."
)

#: `resolution` ノブの受理集合を 1 行で言ったもの（**manifest に無い事実** — 受理集合の正本は
#: `packages/models/src/anima/resolution.ts` で、ADR 0038 §2 により manifest には書かない）。
#: 刻み = 空間圧縮 8 × patch 2、下限は VAE タイル decoder の latent 64（= 512px）、上限は
#: rope 素表の 128 行（= 2048px）。向こうが動いたらここも追随する写しである。
RESOLUTION_NOTE = (
    "Resolution — non-square is fine; each side on a 16 px grid, between 512 and 2048 px:"
)

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
    defaults = default_model(manifest)["pipelineConfig"]["defaults"]
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


def _license() -> list[str]:
    """ライセンス節 — 上流の再配布条件（§3(a) / (b) / (d)）を配布形のどこで満たしているか。

    Notice の本文はカードにも**逐語で**出す: §3(b) は「Distribution と並べて目立つように
    掲示する」ことを求めており、HF のリポジトリで最初に読まれるのはこのカードなので、
    同梱の `NOTICE.md` を指すだけでは掲示したことにならない。
    """
    return [
        "## License",
        "",
        "The weights derive from the CircleStone Anima base model and stay under the CircleStone",
        "Non-Commercial License (non-commercial use only). This repository ships `LICENSE.md`"
        " (the full",
        "license text) and `NOTICE.md` (this attribution plus the list of modifications).",
        "",
        ATTRIBUTION_NOTICE,
        "",
        f"- Baked-in LoRA: the official {LORA_NAME} ([source]({LORA_SOURCE})), folded into the"
        " weights at export.",
        "- This is not an official product of CircleStone Labs LLC, and it is not endorsed,"
        " approved or",
        "  validated by CircleStone Labs LLC.",
    ]


def _usage(manifest: Mapping[str, Any], repo: str) -> list[str]:
    """Usage 例の方針: 動く最小形は生かし、**普通のユースケースで使いそうな optional は
    コメントアウトで併記**する（選べる値も同じ行のコメントに列挙 — manifest から機械導出する
    ので、モデル / quant が増えれば列挙も追従する）。読者がコメントを外すだけで次の一歩へ
    進める形。
    """
    model_name = manifest["defaultModel"]
    model = default_model(manifest)
    quant = model["defaultQuant"]
    model_names = " / ".join(sorted(manifest["models"]))
    quant_names = " / ".join(sorted(model["quants"]))
    defaults = model["pipelineConfig"]["defaults"]
    resolution = defaults["resolution"]
    return [
        "## Usage",
        "",
        "```ts",
        'import { AnimaPipeline, encodePng } from "jsr:@karume/models";',
        "",
        f'using pipeline = await AnimaPipeline.fromPretrained("{repo}", {{',
        f'  // model: "{model_name}", // default — available: {model_names}',
        f'  // quant: "{quant}", // default — available: {quant_names}',
        "});",
        "",
        "const image = await pipeline.generate({",
        '  prompt: "1girl, solo, long hair, blue eyes, school uniform, masterpiece",',
        "",
        f"  // steps: {defaults['steps']}, // default — the baked-in LoRA is distilled for"
        " few-step sampling",
        f"  // {RESOLUTION_NOTE}",
        f"  // resolution: {{ width: {resolution['width']},"
        f" height: {resolution['height']} }}, // default",
        "",
        "  // Classifier-free guidance runs a second (uncond) branch — twice the work per step.",
        "  // It is skipped at guidanceScale 1, where a negativePrompt is refused rather than",
        "  // silently ignored, so the two lines below only make sense together:",
        "  // guidanceScale: 5,",
        f'  // negativePrompt: "{defaults["negativePrompt"]}",',
        "",
        "  seed: 42, // same seed + same request → same image",
        "});",
        "",
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
    require_pipeline(manifest, SUPPORTED_PIPELINE)
    return render(
        (
            frontmatter(ANIMA_METADATA),
            ["", f"# {ANIMA_TITLE}", ""],
            _overview(manifest),
            [""],
            _merged_lora(),
            [""],
            _license(),
            [""],
            models(manifest),
            [""],
            _usage(manifest, repo),
            *model_sections(manifest, (files, quants, _defaults)),
        )
    )
