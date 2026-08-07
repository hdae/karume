"""配布形のモデルカード（`README.md`）— manifest から機械導出する純関数。

HF は `README.md` の YAML frontmatter をモデルの機械可読メタデータとして読む（ADR 0037 §3 の
「そのまま HF リポとしてアップロードできる形」の最後の 1 枚）。ライセンスと由来は manifest に
書かない決定なので（ADR 0038 §6）、その責務はここが持つ。

**pipeline 別のテンプレート**（`karume.dist` の {@link karume.dist.PIPELINES} と 1 対 1）。
`karume.dist` と同じ層分けにしてある — 共有するのは「frontmatter を組む / ファイル表を並べる /
preset 表を並べる」層だけで、何を説明し何を使い方に綴るかは pipeline ごとの節が持つ。

MUST: **数値・ファイル一覧・preset 表・variant ラベル・スタイル表・話者表は 1 つ残らず
manifest から導出する**。手書きのサイズや variant 名は資産と独立に動けてしまい、「表と現物が
食い違う」失敗様式が manifest を導出物にした意味ごと消える（ADR 0038 Context）。ここが持つ
定数は、manifest に**存在しない事実**（base model・ライセンス・焼き込んだ LoRA の出所）だけ。

MUST: 描画は決定的 — 同一 manifest なら**バイト単位で同一**。時刻・環境・集合の反復順に
依存するものを混ぜない（差分が出れば「資産が変わった」と読めることが再組み立ての前提）。
JSON 由来の dict は挿入順が保たれるので、`presets` / `styles` / `speakers` は manifest に
並んだ順のまま出す（並べ替えを挟むと「manifest の並び」という事実がカードから消える）。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

# ---- ① 共有部: frontmatter・ファイル表・preset 表 -----------------------------

#: HF の `library_name`（全 pipeline 共通 — 読み手はこのリポジトリの runtime 1 つだけ）。
LIBRARY_NAME = "karume"

#: 表に出す sha256 の桁数（完全な値は karume.json が持つ — 表は同一性の目視照合用）。
_SHA_DIGITS = 16

_UNITS = (("GiB", 1 << 30), ("MiB", 1 << 20), ("KiB", 1 << 10))


@dataclass(frozen=True)
class CardMetadata:
    """frontmatter に載る **manifest に無い事実**（ADR 0038 §6 で manifest から外した領分）。

    実地確認: https://huggingface.co/docs/hub/model-cards 。`base_model_relation` の語彙は
    adapter / merge / quantized / finetune の 4 値で、どちらの配布形も「base の重みを f16 と
    i8 の格納形へ落とし直したもの」なので `quantized` を採る。merge は **Hub 上の base_model を
    2 つ以上並べる**形に紐づいた値で、Hub にない出所（Anima の LoRA）を正しく表現できない。
    """

    pipeline_tag: str
    base_model: str
    base_model_relation: str
    license: str
    license_name: str
    license_link: str
    tags: tuple[str, ...]


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


def _require_pipeline(manifest: Mapping[str, Any], supported: str) -> None:
    """本文が前提にしている pipeline 契約であることを確かめる（違えば描かない）。

    テンプレートは pipeline 固有なので、別契約の manifest を食わせると「表は合っているのに
    説明だけ別のモデルの話」というカードが黙って出る。
    """
    if manifest["pipeline"] != supported:
        raise ValueError(
            f"モデルカードの本文は {supported} 固有 — "
            f"pipeline '{manifest['pipeline']}' のカードは書けない"
        )


def _frontmatter(metadata: CardMetadata) -> list[str]:
    return [
        "---",
        f"library_name: {LIBRARY_NAME}",
        f"pipeline_tag: {metadata.pipeline_tag}",
        f"base_model: {metadata.base_model}",
        f"base_model_relation: {metadata.base_model_relation}",
        f"license: {metadata.license}",
        f"license_name: {metadata.license_name}",
        f"license_link: {metadata.license_link}",
        "tags:",
        *(f"  - {tag}" for tag in metadata.tags),
        "---",
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


def _knob(value: Any) -> str:
    """`defaults` の 1 値を綴る（文字列だけコード体 — 数値と名前を見分けられるように）。"""
    return f"`{value}`" if isinstance(value, str) else f"{value}"


def _render(sections: Sequence[Sequence[str]]) -> str:
    """節の並びを 1 本の本文にする（末尾改行つき）。"""
    return "\n".join(line for section in sections for line in section) + "\n"


# ---- ② Anima（text-to-image）-------------------------------------------------

#: このテンプレートが説明できるパイプライン契約（ADR 0038 §1）。
SUPPORTED_PIPELINE = "anima/1"

PIPELINE_TAG = "text-to-image"

#: ライセンス（実地確認）: base model のリポジトリ自身は license を宣言せず、カードは
#: 「元のモデルカードを見よ」として `circlestone-labs/Anima` を指す。実値はそちらにある。
ANIMA_METADATA = CardMetadata(
    pipeline_tag=PIPELINE_TAG,
    base_model="circlestone-labs/Anima-Base-v1.0-Diffusers",
    base_model_relation="quantized",
    license="other",
    license_name="circlestone-labs-non-commercial-license",
    license_link="https://huggingface.co/circlestone-labs/Anima/blob/main/LICENSE.md",
    tags=("text-to-image", "webgpu"),
)

#: この配布形を上げる HF リポジトリ ID（manifest は自分の在り処を知らない — 使い方の
#: スニペットに実 ID を綴るために持つ）。
HF_REPO = "hdae/anima-turbo"

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
    defaults = manifest["pipelineConfig"]["defaults"]
    base_model = ANIMA_METADATA.base_model
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
    """Anima 配布形の `README.md` 本文を組み立てる（純関数・末尾改行つき）。"""
    _require_pipeline(manifest, SUPPORTED_PIPELINE)
    return _render(
        (
            _frontmatter(ANIMA_METADATA),
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
    )


# ---- ③ SBV2（text-to-speech）------------------------------------------------

#: このテンプレートが説明できるパイプライン契約（ADR 0038 §1）。
SBV2_SUPPORTED_PIPELINE = "sbv2/1"

SBV2_PIPELINE_TAG = "text-to-speech"

#: ライセンス（実地確認 2026-08-07 — https://huggingface.co/rufflet17/voice_models）: 出所の
#: リポジトリは **license を宣言しておらず、モデルカードも無い**。SPDX 識別子を当てられない
#: ので `other` を採り、`license_link` は「実際に条件が書かれている場所」= 出所のリポジトリを
#: 指す。公開者が改変自由としている事実は本文の帰属節に書く（HF の語彙では表せない）。
SBV2_METADATA = CardMetadata(
    pipeline_tag=SBV2_PIPELINE_TAG,
    base_model="rufflet17/voice_models",
    base_model_relation="quantized",
    license="other",
    license_name="rufflet17-voice-models-terms",
    license_link="https://huggingface.co/rufflet17/voice_models",
    tags=("text-to-speech", "webgpu", "japanese"),
)

#: この配布形を上げる HF リポジトリ ID（`karume.dist.SBV2_DIST_NAME` と同じ 1 語で綴る）。
SBV2_HF_REPO = "hdae/sbv2-FN4"

SBV2_TITLE = "Style-Bert-VITS2 FN4 — Karume"

#: 素になった重み（manifest には現れない — 変換されて格納形だけが残るため）。
SBV2_SOURCE_DIR = "FN/FN4/"
SBV2_ARCHITECTURE = "Style-Bert-VITS2 JP-Extra"
SBV2_SOURCE_VERSION = "2.6.1-JP-Extra"

#: `text_encoder` の素になった日本語 BERT（`export_deberta.py` の `MODEL_ID`）。manifest は
#: 役割名しか持たないので、帰属はカード側が負う。ライセンスは実地確認（2026-08-07）。
SBV2_TEXT_ENCODER_MODEL = "ku-nlp/deberta-v2-large-japanese-char-wwm"
SBV2_TEXT_ENCODER_LICENSE = "cc-by-sa-4.0"

#: 使い方スニペットのデモ文（日本語 TTS の入力なので日本語のまま — CLAUDE.md の言語規約）。
SBV2_DEMO_TEXT = "こんにちは、これはテストです。"


def _sbv2_overview(manifest: Mapping[str, Any]) -> list[str]:
    return [
        "## What is this",
        "",
        f"A Japanese text-to-speech distribution: a **{SBV2_ARCHITECTURE}** voice converted into",
        "the WebGPU inference runtime **Karume**'s container format (a single safetensors file =",
        "weights + a graph JSON embedded in `__metadata__`). Runs as-is in the browser and in"
        " Deno.",
        "",
        "- The acoustic chain is shipped as fused graphs: `text_encoder` (a Japanese DeBERTa),",
        "  `front` (phoneme encoder + duration predictors) and `voice` (flow + HiFi-GAN decoder).",
        "- Style and speaker are **looked up at run time** from the shipped tables — the names in",
        "  the tables below index the rows of `style_vectors` / `speaker_embeddings`.",
        "- Not readable by Style-Bert-VITS2 (it's a different container with an embedded graph);"
        f" the reader is a pipeline that implements `{manifest['pipeline']}`.",
        f"- Exporter used for the conversion: `{manifest['generator']}`. The distribution manifest"
        f" is `karume.json` (`{manifest['format']}`).",
    ]


def _sbv2_base_weights() -> list[str]:
    base_model = SBV2_METADATA.base_model
    return [
        "## Base weights and attribution",
        "",
        "Converted into the container format — the original checkpoint is not distributed here.",
        "",
        f"- **Voice**: `{SBV2_SOURCE_DIR}` of [{base_model}](https://huggingface.co/{base_model})",
        f"- **Architecture**: {SBV2_ARCHITECTURE} (`version: {SBV2_SOURCE_VERSION}`)",
        "- **Terms**: the publisher declares the model free to modify. The source repository",
        "  declares no SPDX license and carries no model card, so its page is where the",
        f"  governing terms live — hence `license: {SBV2_METADATA.license}` above, pointed at it.",
        f"- **Text encoder**: [{SBV2_TEXT_ENCODER_MODEL}]"
        f"(https://huggingface.co/{SBV2_TEXT_ENCODER_MODEL}),",
        f"  licensed **{SBV2_TEXT_ENCODER_LICENSE}** (as of retrieval). It is redistributed here",
        "  in the container format as the `text_encoder` component, so that license travels with",
        "  this repository too.",
    ]


def _name_id_rows(table: Mapping[str, int]) -> list[str]:
    """`名前 → ID` の表を manifest の並びのまま行にする（ID は配る表の行番号そのもの）。"""
    return [f"| `{name}` | {identifier} |" for name, identifier in table.items()]


def _sbv2_styles(manifest: Mapping[str, Any]) -> list[str]:
    """スタイル一覧（利用者が `style` に何を渡せるかをカードだけで知れるようにする節）。"""
    path = manifest["components"]["style_vectors"]["file"]["path"]
    return [
        "## Styles",
        "",
        "| Style | ID |",
        "| ----- | -- |",
        *_name_id_rows(manifest["pipelineConfig"]["styles"]),
        "",
        f"`style` takes one of these names — the ID is the row it selects in `{path}`.",
        "`styleWeight` blends between the average style (`0`) and the named one (`1`).",
    ]


def _sbv2_speakers(manifest: Mapping[str, Any]) -> list[str]:
    """話者一覧（`speaker` の受理集合 — スタイルと同じく行番号の対応表）。"""
    path = manifest["components"]["speaker_embeddings"]["file"]["path"]
    return [
        "## Speakers",
        "",
        "| Speaker | ID |",
        "| ------- | -- |",
        *_name_id_rows(manifest["pipelineConfig"]["speakers"]),
        "",
        f"`speaker` takes one of these names — the ID is the row it selects in `{path}`.",
    ]


def _sbv2_usage(manifest: Mapping[str, Any]) -> list[str]:
    default = manifest["defaultPreset"]
    style = manifest["pipelineConfig"]["defaults"]["style"]
    return [
        "## Usage",
        "",
        "```ts",
        'import { encodeWav, Sbv2Pipeline } from "jsr:@karume/models";',
        "",
        f"// The preset defaults to {default}.",
        f'using pipeline = await Sbv2Pipeline.fromPretrained("{SBV2_HF_REPO}");',
        "const audio = await pipeline.generate({",
        f'  text: "{SBV2_DEMO_TEXT}",',
        f'  style: "{style}",',
        "  seed: 42,",
        "});",
        'await Deno.writeFile("sbv2.wav", encodeWav(audio.data, audio.sampleRate));',
        "```",
        "",
        "`generate()` returns `{ sampleRate, data }`, where `data` is an f32 mono waveform —",
        "exactly what `encodeWav` takes.",
        "Weights are fetched once and cached (verified against `karume.json`'s `size` / `sha256`).",
        "You can also build from bytes you fetched yourself (`Sbv2Pipeline.fromAssets`).",
        "",
        "The Japanese analyzer dictionary the text front-end needs is **not** part of this",
        "repository: the pipeline fetches it on the first `generate()` and keeps it for the rest",
        "of the instance's life (pass your own through the `dictionary` option to skip the fetch).",
    ]


def _sbv2_defaults(manifest: Mapping[str, Any]) -> list[str]:
    defaults = manifest["pipelineConfig"]["defaults"]
    return [
        "## Defaults",
        "",
        "Any knob not passed to `generate()` is filled in from the manifest's defaults.",
        "",
        *(f"- **{key}**: {_knob(value)}" for key, value in defaults.items()),
        "",
        "`seed` is the one knob the manifest does not carry — it defaults to `0`, and the same"
        " seed with the same knobs gives the same waveform.",
    ]


def render_sbv2_model_card(manifest: Mapping[str, Any]) -> str:
    """SBV2 配布形の `README.md` 本文を組み立てる（純関数・末尾改行つき）。"""
    _require_pipeline(manifest, SBV2_SUPPORTED_PIPELINE)
    return _render(
        (
            _frontmatter(SBV2_METADATA),
            ["", f"# {SBV2_TITLE}", ""],
            _sbv2_overview(manifest),
            [""],
            _sbv2_base_weights(),
            [""],
            _files(manifest),
            [""],
            _presets(manifest),
            [""],
            _sbv2_styles(manifest),
            [""],
            _sbv2_speakers(manifest),
            [""],
            _sbv2_usage(manifest),
            [""],
            _sbv2_defaults(manifest),
        )
    )
