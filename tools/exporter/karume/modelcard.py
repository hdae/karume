"""配布形のモデルカード（`README.md`）— manifest から機械導出する純関数。

HF は `README.md` の YAML frontmatter をモデルの機械可読メタデータとして読む（ADR 0037 §3 の
「そのまま HF リポとしてアップロードできる形」の最後の 1 枚）。ライセンスと由来は manifest に
書かない決定なので（ADR 0038 §6）、その責務はここが持つ。

MUST: **数値・ファイル一覧・preset 表は 1 つ残らず manifest から導出する**。手書きのサイズや
variant 名は資産と独立に動けてしまい、「表と現物が食い違う」失敗様式が manifest を導出物に
した意味ごと消える（ADR 0038 Context）。ここが持つ定数は、manifest に**存在しない事実**
（base model・ライセンス・焼き込んだ LoRA の出所）だけ。

MUST: 描画は決定的 — 同一 manifest なら**バイト単位で同一**。時刻・環境・集合の反復順に
依存するものを混ぜない（差分が出れば「資産が変わった」と読めることが再組み立ての前提）。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

#: 対応するパイプライン契約（ADR 0038 §1）。本文は Anima 固有なので、他の値は受け付けない。
SUPPORTED_PIPELINE = "anima/1"

#: HF の frontmatter（実地確認: https://huggingface.co/docs/hub/model-cards）。
#: `base_model_relation` の語彙は adapter / merge / quantized / finetune の 4 値で、この配布形は
#: 「base の重みを f16 と i8 の格納形へ落とし直したもの」なので `quantized` を採る。merge は
#: **Hub 上の base_model を 2 つ以上並べる**形に紐づいた値で、焼き込んだ LoRA は Hub にない
#: （出所は Civitai）ため正しく表現できない。LoRA 焼き込みの事実は本文で明示する。
LIBRARY_NAME = "karume"
PIPELINE_TAG = "text-to-image"
BASE_MODEL = "circlestone-labs/Anima-Base-v1.0-Diffusers"
BASE_MODEL_RELATION = "quantized"
TAGS = ("text-to-image", "webgpu")

#: この配布形を上げる HF リポジトリ ID（manifest は自分の在り処を知らない — 使い方の
#: スニペットに実 ID を綴るために持つ）。
HF_REPO = "hdae/anima-turbo"

#: ライセンス（実地確認）: base model のリポジトリ自身は license を宣言せず、カードは
#: 「元のモデルカードを見よ」として `circlestone-labs/Anima` を指す。実値はそちらにある。
LICENSE = "other"
LICENSE_NAME = "circlestone-labs-non-commercial-license"
LICENSE_LINK = "https://huggingface.co/circlestone-labs/Anima/blob/main/LICENSE.md"

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

#: 表に出す sha256 の桁数（完全な値は karume.json が持つ — 表は同一性の目視照合用）。
_SHA_DIGITS = 16

_UNITS = (("GiB", 1 << 30), ("MiB", 1 << 20), ("KiB", 1 << 10))


def format_size(size: int) -> str:
    """バイト数を「単位付き + 生バイト」で綴る（両方出す — 前者は目安、後者が manifest の値）。"""
    for unit, scale in _UNITS:
        if size >= scale:
            return f"{size / scale:.2f} {unit} ({size:,} B)"
    return f"{size:,} B"


def file_rows(manifest: Mapping[str, Any]) -> list[tuple[str, list[str], Mapping[str, Any]]]:
    """`(hub のキー, 参照する variant 一覧, ファイル参照)` を **path で一意化**して並べる。

    キーは hub の `resolve()` が返す綴り（`<component>` / `<component>.<extra>` — ADR 0038 §5）。
    一意化は rope_base のため — f16 / i8 の両 variant が同一ファイルを指すので（1 本化済み）、
    素直に並べると現物にない 2 行目が表に生える。付帯資産は本体の variant を並べ切ってから
    出す（本体 f16 / 付帯 / 本体 i8 と挟まると、どの行が何の重みか読めなくなる）。
    """
    rows: dict[str, tuple[str, list[str], Mapping[str, Any]]] = {}

    def add(key: str, label: str | None, ref: Mapping[str, Any]) -> None:
        _, labels, _ = rows.setdefault(ref["path"], (key, [], ref))
        if label is not None and label not in labels:
            labels.append(label)

    for name, component in manifest["components"].items():
        variants = (
            component["variants"].items() if "variants" in component else ((None, component),)
        )
        for label, entry in variants:
            add(name, label, entry["file"])
        for label, entry in variants:
            for extra, ref in entry.get("extras", {}).items():
                add(f"{name}.{extra}", label, ref)
    return list(rows.values())


def _frontmatter() -> list[str]:
    return [
        "---",
        f"library_name: {LIBRARY_NAME}",
        f"pipeline_tag: {PIPELINE_TAG}",
        f"base_model: {BASE_MODEL}",
        f"base_model_relation: {BASE_MODEL_RELATION}",
        f"license: {LICENSE}",
        f"license_name: {LICENSE_NAME}",
        f"license_link: {LICENSE_LINK}",
        "tags:",
        *(f"  - {tag}" for tag in TAGS),
        "---",
    ]


def _overview(manifest: Mapping[str, Any]) -> list[str]:
    defaults = manifest["pipelineConfig"]["defaults"]
    return [
        "## What is this",
        "",
        f"A distribution that bakes **{LORA_NAME}** into"
        f" [{BASE_MODEL}](https://huggingface.co/{BASE_MODEL}) and converts it into the WebGPU",
        "inference runtime **Karume**'s container format (a single safetensors file = weights +",
        "a graph JSON embedded in `__metadata__`). Runs as-is in the browser and in Deno.",
        "",
        f"- A few-step distillation (from the LoRA) tuned for **{defaults['steps']} steps /"
        f" guidance {defaults['guidanceScale']}**.",
        "- Not readable by diffusers (it's a different container with an embedded graph); the"
        f" reader is a pipeline that implements `{manifest['pipeline']}`.",
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


def _files(manifest: Mapping[str, Any]) -> list[str]:
    lines = [
        "## Files",
        "",
        "| Key | Variant | Path | Size | sha256 |",
        "| ---- | ------- | ---- | ------ | ------ |",
    ]
    for key, labels, ref in file_rows(manifest):
        variant = " / ".join(labels) if labels else "—"
        lines.append(
            f"| `{key}` | {variant} | `{ref['path']}` | {format_size(ref['size'])} |"
            f" `{ref['sha256'][:_SHA_DIGITS]}…` |"
        )
    lines += [
        "",
        "Only the first 16 hex digits of the sha256 are shown (the full value and `size` live in"
        " `karume.json` — verify against that at the fetch layer).",
        "Variant labels use the runtime's **storage dtype vocabulary** (`f16` / `i8`), not the"
        " `fp16` spelling common elsewhere in the ecosystem.",
    ]
    return lines


def _session(preset: Mapping[str, Any]) -> str:
    session = preset["session"]
    features = preset.get("gpuFeatures", {})
    parts = [f"`{key}` = `{value}`" for key, value in session.items()]
    parts += [f"requires `{key}`" for key, value in features.items() if value]
    return " / ".join(parts) if parts else "—"


def _presets(manifest: Mapping[str, Any]) -> list[str]:
    default = manifest["defaultPreset"]
    lines = [
        "## Presets",
        "",
        "| Preset | Weights | Compute |",
        "| ------ | ---- | ---- |",
    ]
    for name, preset in manifest["presets"].items():
        weights = " / ".join(
            f"`{component}` = `{label}`" for component, label in preset["weights"].items()
        )
        mark = " (default)" if name == default else ""
        lines.append(f"| `{name}`{mark} | {weights} | {_session(preset)} |")
    lines += [
        "",
        f"If no preset is given, it runs as `{default}` (the distribution's recommended default).",
    ]
    return lines


def _usage(manifest: Mapping[str, Any]) -> list[str]:
    default = manifest["defaultPreset"]
    return [
        "## Usage",
        "",
        "```ts",
        'import { AnimaPipeline, encodePng } from "jsr:@karume/models";',
        "",
        f"// The preset defaults to {default}.",
        f'using pipeline = await AnimaPipeline.fromPretrained("{HF_REPO}");',
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


def _defaults(manifest: Mapping[str, Any]) -> list[str]:
    defaults = manifest["pipelineConfig"]["defaults"]
    resolution = defaults["resolution"]
    guidance = defaults["guidanceScale"]
    lines = [
        "## Defaults",
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


def render_model_card(manifest: Mapping[str, Any]) -> str:
    """配布形の `README.md` 本文を組み立てる（純関数・末尾改行つき）。"""
    if manifest["pipeline"] != SUPPORTED_PIPELINE:
        raise ValueError(
            f"モデルカードの本文は {SUPPORTED_PIPELINE} 固有 — "
            f"pipeline '{manifest['pipeline']}' のカードは書けない"
        )
    sections: Sequence[Sequence[str]] = (
        _frontmatter(),
        ["", "# Anima Turbo — Karume", ""],
        _overview(manifest),
        [""],
        _merged_lora(),
        [""],
        _files(manifest),
        [""],
        _presets(manifest),
        [""],
        _usage(manifest),
        [""],
        _defaults(manifest),
    )
    return "\n".join(line for section in sections for line in section) + "\n"
