"""配布形のモデルカード（`README.md`）— manifest から機械導出する純関数。

HF は `README.md` の YAML frontmatter をモデルの機械可読メタデータとして読む（ADR 0037 §3 の
「そのまま HF リポとしてアップロードできる形」の最後の 1 枚）。ライセンスと由来は manifest に
書かない決定なので（ADR 0038 §6）、その責務はここが持つ。

**pipeline 別のテンプレート**（`karume.dist` の {@link karume.dist.PIPELINES} の各行に対応）。
`karume.dist` と同じ層分けにしてある — 共有するのは「frontmatter を組む / モデル一覧を並べる /
ファイル表を並べる / quant 表を並べる」層だけで、何を説明し何を使い方に綴るかは pipeline
ごとの節が持つ。

**帰属はテンプレートと別の軸**（{@link Sbv2CardProfile}）。同じ SBV2 のテンプレートでも、
どのファミリーの重みを配るかで出所・ライセンス・引用が丸ごと変わるので、そこだけを
プロファイルに分けて**呼び出し側に明示させる**。

manifest v2（`karume/2` — ADR 0041）は **1 リポに複数モデル**を持てるので、カードも
「リポ全体の説明 → モデル一覧 → 使い方 → モデルごとの節」の形にする。モデルごとの節が
`## Model: <name>` で、その中にファイル表・quant 表・（SBV2 は）スタイル表と話者表が並ぶ。
単一モデルのリポでも同じ形で描く（配布形のレイアウトが一様なのと同じ理由 — 2 個目が増えた
瞬間に構成が変わるカードは、読み手の目印も壊す）。

MUST: **数値・ファイル一覧・quant 表・dtype ラベル・スタイル表・話者表は 1 つ残らず manifest
から導出する**。手書きのサイズや dtype 名は資産と独立に動けてしまい、「表と現物が食い違う」
失敗様式が manifest を導出物にした意味ごと消える（ADR 0038 Context）。ここが持つ定数は、
manifest に**存在しない事実**（base model・ライセンス・焼き込んだ LoRA の出所）だけ。リポ ID も
manifest には無いので、組み立て先から引いた値を引数で受ける。

MUST: 描画は決定的 — 同一 manifest なら**バイト単位で同一**。時刻・環境・集合の反復順に
依存するものを混ぜない（差分が出れば「資産が変わった」と読めることが再組み立ての前提）。
JSON 由来の dict は挿入順が保たれるので、`models` / `quants` / `styles` / `speakers` は
manifest に並んだ順のまま出す（並べ替えを挟むと「manifest の並び」という事実がカードから消える）。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

#: モデルごとの節を描く関数（モデルエントリ → 行の並び）。
ModelSection = Callable[[Mapping[str, Any]], list[str]]

# ---- ① 共有部: frontmatter・モデル一覧・ファイル表・quant 表 ------------------

#: HF の `library_name`（全 pipeline 共通 — 読み手はこのリポジトリの runtime 1 つだけ）。
LIBRARY_NAME = "karume"

#: 配布先の HF アカウント。リポ名は組み立て先のディレクトリ名から決まる（`karume.dist` が
#: `<HF_OWNER>/<ディレクトリ名>` を渡す）ので、ここが持つのは所有者だけ。
HF_OWNER = "hdae"

#: 表に出す sha256 の桁数（完全な値は karume.json が持つ — 表は同一性の目視照合用）。
_SHA_DIGITS = 16

_UNITS = (("GiB", 1 << 30), ("MiB", 1 << 20), ("KiB", 1 << 10))


@dataclass(frozen=True)
class CardMetadata:
    """frontmatter に載る **manifest に無い事実**（ADR 0038 §6 で manifest から外した領分）。

    実地確認: https://huggingface.co/docs/hub/model-cards 。`base_model_relation` の語彙は
    adapter / merge / quantized / finetune の 4 値で、f16 / i8 の配布形は「base の重みを別の
    格納形へ落とし直したもの」なので `quantized` を採る。merge は **Hub 上の base_model を
    2 つ以上並べる**形に紐づいた値で、Hub にない出所（Anima の LoRA）を正しく表現できない。
    **格納形を変えない配布形は 4 値のどれでもない**ので席ごと空ける（同じ実地確認: 未指定なら
    Hub が推論する）— 4 値の中から一番近いものを当てると、カードが事実でない主張を持つ。

    `base_model` が並びなのは、**この配布形が再配布している上流を全部並べる**ため（重みの出所と
    再配布する text encoder は別リポ）。並べても関係は `quantized` のまま — 重みを融合した
    わけではなく、それぞれを格納形へ落とし直して 1 リポに同居させているだけ。
    """

    pipeline_tag: str
    base_model: tuple[str, ...]
    license: str
    tags: tuple[str, ...]
    #: 4 値のどれかが**事実として当たるとき**だけ書く（当たらなければ Hub の推論に任せる）。
    base_model_relation: str | None = None
    #: `license: other` のときだけ HF が読む 2 席。SPDX 識別子を当てられた配布形は**持たない**
    #: （空欄で並べると「名前の無い独自ライセンス」に読める）。
    license_name: str | None = None
    license_link: str | None = None


def format_size(size: int) -> str:
    """バイト数を「単位付き + 生バイト」で綴る（両方出す — 前者は目安、後者が manifest の値）。"""
    for unit, scale in _UNITS:
        if size >= scale:
            return f"{size / scale:.2f} {unit} ({size:,} B)"
    return f"{size:,} B"


def file_rows(model: Mapping[str, Any]) -> list[tuple[str, list[str], Mapping[str, Any]]]:
    """`(hub のキー, 参照する dtype 一覧, ファイル参照)` を **path で一意化**して並べる。

    キーは hub の `resolve()` が返す綴り（`<weights>` / `<weights>.<extra>` / `<asset>`）。
    一意化は rope_base のため — f16 / i8 の両 dtype が同一ファイルを指すので（1 本化済み）、
    素直に並べると現物にない 2 行目が表に生える。付帯資産は本体の dtype を並べ切ってから
    出す（本体 f16 / 付帯 / 本体 i8 と挟まると、どの行が何の重みか読めなくなる）。

    assets は quant 選択に依存しないので dtype 列は空（`—`）で並ぶ。
    """
    rows: dict[str, tuple[str, list[str], Mapping[str, Any]]] = {}

    def add(key: str, label: str | None, ref: Mapping[str, Any]) -> None:
        _, labels, _ = rows.setdefault(ref["path"], (key, [], ref))
        if label is not None and label not in labels:
            labels.append(label)

    for name, entry in model["weights"].items():
        for label, files in entry.items():
            add(name, label, files["file"])
        for label, files in entry.items():
            for extra, ref in files.get("extras", {}).items():
                add(f"{name}.{extra}", label, ref)
    for name, ref in model["assets"].items():
        add(name, None, ref)
    return list(rows.values())


def _require_pipeline(manifest: Mapping[str, Any], supported: str) -> None:
    """全モデルが本文の前提にしている pipeline 契約であることを確かめる（違えば描かない）。

    テンプレートは pipeline 固有なので、別契約の manifest を食わせると「表は合っているのに
    説明だけ別のモデルの話」というカードが黙って出る。ファミリーリポは別アーキを混ぜられる
    （ADR 0041 §2）ので、1 つでも違えば描けない。
    """
    for name, model in manifest["models"].items():
        if model["pipeline"] != supported:
            raise ValueError(
                f"モデルカードの本文は {supported} 固有 — "
                f"モデル '{name}' の pipeline '{model['pipeline']}' のカードは書けない"
            )


def _base_model(models: Sequence[str]) -> list[str]:
    """`base_model`（1 本なら scalar・複数なら YAML の並び — HF はどちらも読む）。"""
    if len(models) == 1:
        return [f"base_model: {models[0]}"]
    return ["base_model:", *(f"  - {model}" for model in models)]


def _frontmatter(metadata: CardMetadata) -> list[str]:
    return [
        "---",
        f"library_name: {LIBRARY_NAME}",
        f"pipeline_tag: {metadata.pipeline_tag}",
        *_base_model(metadata.base_model),
        *(
            []
            if metadata.base_model_relation is None
            else [f"base_model_relation: {metadata.base_model_relation}"]
        ),
        f"license: {metadata.license}",
        *([f"license_name: {metadata.license_name}"] if metadata.license_name is not None else []),
        *([f"license_link: {metadata.license_link}"] if metadata.license_link is not None else []),
        "tags:",
        *(f"  - {tag}" for tag in metadata.tags),
        "---",
    ]


def _models(manifest: Mapping[str, Any]) -> list[str]:
    """リポが載せているモデルの一覧（v2 で初めて機械可読になった軸 — ADR 0041 §2）。"""
    default = manifest["defaultModel"]
    lines = [
        "## Models",
        "",
        "| Model | Pipeline | Quants | Default quant |",
        "| ----- | -------- | ------ | ------------- |",
    ]
    for name, model in manifest["models"].items():
        mark = " (default)" if name == default else ""
        quants = " / ".join(f"`{quant}`" for quant in model["quants"])
        lines.append(
            f"| `{name}`{mark} | `{model['pipeline']}` | {quants} | `{model['defaultQuant']}` |"
        )
    lines += [
        "",
        f"`model` selects one of these; omitted, it is `{default}`."
        " `quant` defaults to that model's own default quant.",
    ]
    return lines


def _files(model: Mapping[str, Any]) -> list[str]:
    lines = [
        "### Files",
        "",
        "| Key | Dtype | Path | Size | sha256 |",
        "| ---- | ----- | ---- | ------ | ------ |",
    ]
    for key, labels, ref in file_rows(model):
        dtype = " / ".join(labels) if labels else "—"
        lines.append(
            f"| `{key}` | {dtype} | `{ref['path']}` | {format_size(ref['size'])} |"
            f" `{ref['sha256'][:_SHA_DIGITS]}…` |"
        )
    lines += [
        "",
        "Only the first 16 hex digits of the sha256 are shown (the full value and `size` live in"
        " `karume.json` — verify against that at the fetch layer).",
        "Dtype labels use the runtime's **storage dtype vocabulary** (`f16` / `i8`), not the"
        " `fp16` spelling common elsewhere in the ecosystem.",
        "A path under `shared/` is one this model shares byte for byte with another model in this"
        " repository (it is fetched and cached once).",
    ]
    return lines


def _session(quant: Mapping[str, Any]) -> str:
    session = quant["session"]
    features = quant.get("gpuFeatures", {})
    parts = [f"`{key}` = `{value}`" for key, value in session.items()]
    parts += [f"requires `{key}`" for key, value in features.items() if value]
    return " / ".join(parts) if parts else "—"


def _quants(model: Mapping[str, Any]) -> list[str]:
    default = model["defaultQuant"]
    lines = [
        "### Quants",
        "",
        "| Quant | Weights | Compute |",
        "| ----- | ---- | ---- |",
    ]
    for name, quant in model["quants"].items():
        weights = " / ".join(
            f"`{weight}` = `{label}`" for weight, label in quant["weights"].items()
        )
        mark = " (default)" if name == default else ""
        lines.append(f"| `{name}`{mark} | {weights} | {_session(quant)} |")
    lines += [
        "",
        f"If no quant is given, it runs as `{default}` (this model's recommended default).",
    ]
    return lines


def _knob(value: Any) -> str:
    """`defaults` の 1 値を綴る（文字列だけコード体 — 数値と名前を見分けられるように）。"""
    return f"`{value}`" if isinstance(value, str) else f"{value}"


def _render(sections: Sequence[Sequence[str]]) -> str:
    """節の並びを 1 本の本文にする（末尾改行つき）。"""
    return "\n".join(line for section in sections for line in section) + "\n"


def _model_sections(
    manifest: Mapping[str, Any],
    per_model: Sequence[ModelSection],
) -> list[Sequence[str]]:
    """モデルごとの節（`## Model: <name>`）を manifest の並びのまま組む。

    `per_model` は「モデルエントリを受けて節（行の並び）を返す」描き手の列 — pipeline 固有の
    節（SBV2 のスタイル表など）をここへ差し込むための唯一の軸。
    """
    sections: list[Sequence[str]] = []
    for name, model in manifest["models"].items():
        sections.append([""])
        sections.append([f"## Model: {name}", ""])
        for index, render in enumerate(per_model):
            if index:
                sections.append([""])
            sections.append(render(model))
    return sections


# ---- ② Anima（text-to-image）-------------------------------------------------

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


def _default_model(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    """既定モデルのエントリ（リポ全体を 1 つ紹介するときの代表 — 使い方の既定でもある）。"""
    return manifest["models"][manifest["defaultModel"]]


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


# ---- ③ SBV2（text-to-speech）------------------------------------------------

#: このテンプレートが説明できるパイプライン契約（ADR 0041 §2 — モデル単位）。
SBV2_SUPPORTED_PIPELINE = "sbv2/1"

SBV2_PIPELINE_TAG = "text-to-speech"

#: アーキテクチャ（どのファミリーも JP-Extra 系 — 違うのは重みの出所であって形ではない）。
SBV2_ARCHITECTURE = "Style-Bert-VITS2 JP-Extra"

#: `text_encoder` の素になった日本語 BERT（`export_deberta.py` の `MODEL_ID`）。manifest は
#: 役割名しか持たないので、帰属はカード側が負う。ライセンスは実地確認（2026-08-07）。
#: **どのファミリーでも同じ 1 本**を再配布するので、プロファイルの席ではなくここが持つ。
SBV2_TEXT_ENCODER_MODEL = "ku-nlp/deberta-v2-large-japanese-char-wwm"
SBV2_TEXT_ENCODER_LICENSE = "cc-by-sa-4.0"

#: 使い方スニペットのデモ文（日本語 TTS の入力なので日本語のまま — CLAUDE.md の言語規約）。
SBV2_DEMO_TEXT = "こんにちは、これはテストです。"


@dataclass(frozen=True)
class Sbv2CardProfile:
    """SBV2 カードの**帰属プロファイル** — 声のファミリーごとに違う「manifest に無い事実」。

    SBV2 のテンプレートは 1 つでも、帰属（出所・ライセンス・再配布の条件・引用）はファミリー
    ごとに**別の法的事実**になる。決め打ちのまま別ファミリーのリポへ描くと、表も使い方も
    正しいのに帰属だけが前のファミリーのまま残る — 配ってからでないと誰も気づけない誤りなので、
    ここが席として分けて持つ（選択は {@link SBV2_CARD_PROFILES} 経由で**明示**）。

    `attribution` が行の並びそのものなのは、この席に入るのが導出できない散文（ライセンス条項の
    要約・引用・再配布条件）だから。機械的に組める 2 行（Voices / Architecture）だけは
    {@link _sbv2_base_weights} が `source_dirs` / `source_version` から組む。
    """

    metadata: CardMetadata
    title: str
    #: 出所リポジトリの中で、この配布形の声になったディレクトリ群。
    source_dirs: tuple[str, ...]
    #: 出所の `config.json` が名乗る version 文字列。
    source_version: str
    #: Voices / Architecture に続く帰属の箇条（Markdown の行そのもの）。
    attribution: tuple[str, ...]


# ---- ③-a FN 系（単一モデル） -------------------------------------------------

#: ライセンス（実地確認 2026-08-07 — https://huggingface.co/rufflet17/voice_models）: 出所の
#: リポジトリは **license を宣言しておらず、モデルカードも無い**。SPDX 識別子を当てられない
#: ので `other` を採り、`license_link` は「実際に条件が書かれている場所」= 出所のリポジトリを
#: 指す。公開者が改変自由としている事実は本文の帰属節に書く（HF の語彙では表せない）。
SBV2_FN_METADATA = CardMetadata(
    pipeline_tag=SBV2_PIPELINE_TAG,
    base_model=("rufflet17/voice_models",),
    base_model_relation="quantized",
    license="other",
    license_name="rufflet17-voice-models-terms",
    license_link="https://huggingface.co/rufflet17/voice_models",
    tags=("text-to-speech", "webgpu", "japanese"),
)

SBV2_FN_PROFILE = Sbv2CardProfile(
    metadata=SBV2_FN_METADATA,
    title="Style-Bert-VITS2 — Karume",
    source_dirs=("FN/",),
    source_version="2.6.1-JP-Extra",
    attribution=(
        "- **Terms**: the publisher declares the model free to modify. The source repository",
        "  declares no SPDX license and carries no model card, so its page is where the",
        f"  governing terms live — hence `license: {SBV2_FN_METADATA.license}` above,"
        " pointed at it.",
        f"- **Text encoder**: [{SBV2_TEXT_ENCODER_MODEL}]"
        f"(https://huggingface.co/{SBV2_TEXT_ENCODER_MODEL}),",
        f"  licensed **{SBV2_TEXT_ENCODER_LICENSE}** (as of retrieval). It is redistributed here",
        "  in the container format as the `text_encoder` component, so that license travels with",
        "  this repository too.",
    ),
)


# ---- ③-b JVNV 系（ファミリー） -----------------------------------------------

#: ライセンス（実地確認 2026-08-09 — https://huggingface.co/litagin/style_bert_vits2_jvnv）:
#: 出所のリポジトリは「ライセンスは JVNV コーパスの cc-by-sa-4.0 を引き継ぐ」と明言している。
#: SPDX 標準タグが当たるので `license_name` / `license_link` は持たない（HF が識別子から解決
#: する席で、独自名を名乗らせると「名前の無い独自ライセンス」に読める）。
#: `base_model` に text encoder を併記するのは、この配布形が**両方を再配布している**ため。
#: 両者とも cc-by-sa-4.0 なので、SA（同一ライセンス継承）はリポジトリ全体で矛盾しない。
SBV2_JVNV_METADATA = CardMetadata(
    pipeline_tag=SBV2_PIPELINE_TAG,
    base_model=("litagin/style_bert_vits2_jvnv", SBV2_TEXT_ENCODER_MODEL),
    base_model_relation="quantized",
    license="cc-by-sa-4.0",
    tags=("text-to-speech", "webgpu", "japanese"),
)

#: CC BY-SA 4.0 の条文（BY の「ライセンス URL を示す」を、カード自身が満たすための 1 本）。
SBV2_JVNV_LICENSE_URL = "https://creativecommons.org/licenses/by-sa/4.0/"

SBV2_JVNV_PROFILE = Sbv2CardProfile(
    metadata=SBV2_JVNV_METADATA,
    title="Style-Bert-VITS2 JVNV — Karume",
    source_dirs=("jvnv-F1-jp/", "jvnv-F2-jp/", "jvnv-M1-jp/", "jvnv-M2-jp/"),
    source_version="2.0-JP-Extra",
    attribution=(
        f"- **Terms**: **[CC BY-SA 4.0]({SBV2_JVNV_LICENSE_URL})**, inherited from the JVNV"
        " corpus the",
        "  voices were trained on — the source repository states the corpus license carries over",
        "  to the models. Redistribution here keeps it: credit the authors, name the source and",
        "  the license URL, state that the weights were **converted to another format and"
        " quantized**",
        "  to i8, license any derivative work under CC BY-SA 4.0 as well, and impose no further",
        "  restrictions. There is no NonCommercial and no NoDerivatives clause — commercial use",
        "  and modification are both allowed.",
        "- **Changes made here**: conversion into the Karume container format and **i8"
        " quantization**",
        "  of the weights. No retraining, no fine-tuning — the voices are the source checkpoints",
        "  in a different storage form.",
        f"- **Text encoder**: [{SBV2_TEXT_ENCODER_MODEL}]"
        f"(https://huggingface.co/{SBV2_TEXT_ENCODER_MODEL}),",
        f"  licensed **{SBV2_TEXT_ENCODER_LICENSE}** (as of retrieval). It is redistributed here",
        "  in the container format as the `text_encoder` component — the same license as the",
        "  voices, so the share-alike term is consistent across everything in this repository.",
        "- **Training data**: the JVNV corpus, licensed CC BY-SA 4.0. Detai Xin, Junfeng Jiang,",
        "  Shinnosuke Takamichi, Yuki Saito, Akiko Aizawa, Hiroshi Saruwatari,",
        # 論文題は 1 行に収める（折り返すと、題での検索が本文にあるのに当たらなくなる）。
        "  *JVNV: A Corpus of Japanese Emotional Speech with Verbal Content and"
        " Nonverbal Expressions*,",
        "  [arXiv:2310.06072](https://arxiv.org/abs/2310.06072). Corpus page:",
        "  <https://sites.google.com/site/shinnosuketakamichi/research-topics/jvnv_corpus>",
        "- **Training implementation**:"
        " [Style-Bert-VITS2](https://github.com/litagin02/Style-Bert-VITS2)",
        "  (AGPL-3.0). Karume's runtime contains **none of that code**: it is an independent",
        "  implementation that reads these weights from its own container format, so the AGPL",
        "  terms govern the training implementation, not the runtime that plays these files.",
    ),
)

#: 選べる帰属プロファイル。**既定は置かない** — 省略時に片方を黙って選ぶと、新しいファミリーを
#: 配るたびに前のファミリーの帰属が沈黙で再発する（`karume.dist` が明示を要求する）。
SBV2_CARD_PROFILES: Mapping[str, Sbv2CardProfile] = {
    "fn": SBV2_FN_PROFILE,
    "jvnv": SBV2_JVNV_PROFILE,
}


def _sbv2_overview(manifest: Mapping[str, Any]) -> list[str]:
    return [
        "## What is this",
        "",
        f"A Japanese text-to-speech distribution: **{SBV2_ARCHITECTURE}** voices converted into",
        "the WebGPU inference runtime **Karume**'s container format (a single safetensors file =",
        "weights + a graph JSON embedded in `__metadata__`). Runs as-is in the browser and in"
        " Deno.",
        "",
        "- The acoustic chain is shipped as fused graphs: `text_encoder` (a Japanese DeBERTa),",
        "  `front` (phoneme encoder + duration predictors) and `voice` (flow + HiFi-GAN decoder).",
        "- Style and speaker are **looked up at run time** from the shipped tables — the names in",
        "  the tables below index the rows of `style_vectors` / `speaker_embeddings`.",
        "- Not readable by Style-Bert-VITS2 (it's a different container with an embedded graph);"
        f" the reader is a pipeline that implements `{SBV2_SUPPORTED_PIPELINE}`.",
        f"- Exporter used for the conversion: `{manifest['generator']}`. The distribution manifest"
        f" is `karume.json` (`{manifest['format']}`).",
    ]


def _sbv2_base_weights(profile: Sbv2CardProfile) -> list[str]:
    """帰属節。声の出所リポジトリは `base_model` の先頭（= 重みの出所そのもの）から引く。"""
    base_model = profile.metadata.base_model[0]
    dirs = " / ".join(f"`{name}`" for name in profile.source_dirs)
    return [
        "## Base weights and attribution",
        "",
        "Converted into the container format — the original checkpoints are not distributed here.",
        "",
        f"- **Voices**: {dirs} of [{base_model}](https://huggingface.co/{base_model})",
        f"- **Architecture**: {SBV2_ARCHITECTURE} (`version: {profile.source_version}`)",
        *profile.attribution,
    ]


def _name_id_rows(table: Mapping[str, int]) -> list[str]:
    """`名前 → ID` の表を manifest の並びのまま行にする（ID は配る表の行番号そのもの）。"""
    return [f"| `{name}` | {identifier} |" for name, identifier in table.items()]


def _sbv2_styles(model: Mapping[str, Any]) -> list[str]:
    """スタイル一覧（利用者が `style` に何を渡せるかをカードだけで知れるようにする節）。"""
    path = model["assets"]["style_vectors"]["path"]
    return [
        "### Styles",
        "",
        "| Style | ID |",
        "| ----- | -- |",
        *_name_id_rows(model["pipelineConfig"]["styles"]),
        "",
        f"`style` takes one of these names — the ID is the row it selects in `{path}`.",
        "`styleWeight` blends between the average style (`0`) and the named one (`1`).",
    ]


def _sbv2_speakers(model: Mapping[str, Any]) -> list[str]:
    """話者一覧（`speaker` の受理集合 — スタイルと同じく行番号の対応表）。"""
    path = model["assets"]["speaker_embeddings"]["path"]
    return [
        "### Speakers",
        "",
        "| Speaker | ID |",
        "| ------- | -- |",
        *_name_id_rows(model["pipelineConfig"]["speakers"]),
        "",
        f"`speaker` takes one of these names — the ID is the row it selects in `{path}`.",
    ]


def _sbv2_usage(manifest: Mapping[str, Any], repo: str) -> list[str]:
    model_name = manifest["defaultModel"]
    model = _default_model(manifest)
    quant = model["defaultQuant"]
    style = model["pipelineConfig"]["defaults"]["style"]
    return [
        "## Usage",
        "",
        "```ts",
        'import { encodeWav, Sbv2Pipeline } from "jsr:@karume/models";',
        "",
        f"// Both options may be omitted: model defaults to {model_name}, quant to {quant}.",
        f'using pipeline = await Sbv2Pipeline.fromPretrained("{repo}", {{',
        f'  model: "{model_name}",',
        f'  quant: "{quant}",',
        "});",
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


def _sbv2_defaults(model: Mapping[str, Any]) -> list[str]:
    defaults = model["pipelineConfig"]["defaults"]
    return [
        "### Defaults",
        "",
        "Any knob not passed to `generate()` is filled in from the manifest's defaults.",
        "",
        *(f"- **{key}**: {_knob(value)}" for key, value in defaults.items()),
        "",
        "`seed` is the one knob the manifest does not carry — it defaults to `0`, and the same"
        " seed with the same knobs gives the same waveform.",
    ]


def render_sbv2_model_card(manifest: Mapping[str, Any], repo: str, profile: Sbv2CardProfile) -> str:
    """SBV2 配布形の `README.md` 本文を組み立てる（純関数・末尾改行つき）。

    `profile` に既定を置かないのは MUST — 帰属はファミリーごとに違う事実で、既定を持たせた
    瞬間に「別ファミリーのリポへ前のファミリーの帰属を描く」経路が黙って生える。
    """
    _require_pipeline(manifest, SBV2_SUPPORTED_PIPELINE)
    return _render(
        (
            _frontmatter(profile.metadata),
            ["", f"# {profile.title}", ""],
            _sbv2_overview(manifest),
            [""],
            _sbv2_base_weights(profile),
            [""],
            _models(manifest),
            [""],
            _sbv2_usage(manifest, repo),
            *_model_sections(
                manifest, (_files, _quants, _sbv2_styles, _sbv2_speakers, _sbv2_defaults)
            ),
        )
    )


# ---- ④ Irodori（text-to-speech）----------------------------------------------

#: このテンプレートが説明できるパイプライン契約（ADR 0041 §2 — モデル単位）。
IRODORI_SUPPORTED_PIPELINE = "irodori/1"

IRODORI_PIPELINE_TAG = "text-to-speech"

IRODORI_TITLE = "Irodori-TTS v4 Small — Karume"

#: 重みの出所（`inputs/irodori/<モデル名>/` に手で置く HF リポ）と、その中の text backbone の
#: 素になった日本語 ModernBERT（チェックポイントの `config_json` が `text_tokenizer_repo` /
#: `text_encoder_revision` で名指ししている 1 本）。**どちらも再配布している**ので両方並べる。
#: ライセンスは実地確認（2026-08-12・HF の models API の `tags`）: どちらも `license: mit`。
IRODORI_BASE_MODEL = "Aratako/Irodori-TTS-v4-Small"
IRODORI_TEXT_BACKBONE_MODEL = "sbintuitions/modernbert-ja-310m"

#: 波形へ落とすコーデック（上流では別リポ・別重み — この配布形には**同梱している**）。
#: ライセンスは MIT（`docs/research/2026-08-11-irodori-source-recon.md` の実地確認）。
#: 再配布しているので `base_model` にも帰属節にも並べる。
IRODORI_CODEC_MODEL = "Aratako/Semantic-DACVAE-Japanese-32dim"

IRODORI_METADATA = CardMetadata(
    pipeline_tag=IRODORI_PIPELINE_TAG,
    base_model=(IRODORI_BASE_MODEL, IRODORI_TEXT_BACKBONE_MODEL, IRODORI_CODEC_MODEL),
    # `base_model_relation` は置かない — この配布形は格納形を変えず（f32 のまま）コンテナだけを
    # 移したもので、adapter / merge / quantized / finetune のどれでもない（CardMetadata の doc）。
    license="mit",
    tags=("text-to-speech", "webgpu", "japanese"),
)

#: 使い方スニペットのデモ入力（日本語 TTS の入力なので日本語のまま — CLAUDE.md の言語規約）。
IRODORI_DEMO_TEXT = "こんにちは、これはテストです。"
IRODORI_DEMO_CAPTION = "落ち着いた女性の声で、ゆっくりと丁寧に話している。"


def _irodori_overview(manifest: Mapping[str, Any]) -> list[str]:
    return [
        "## What is this",
        "",
        "A Japanese text-to-speech distribution: the **Irodori-TTS v4 (Small)** rectified-flow DiT",
        "converted into the WebGPU inference runtime **Karume**'s container format (a single",
        "safetensors file = weights + a graph JSON embedded in `__metadata__`). Runs as-is in the",
        "browser and in Deno.",
        "",
        "- Eight graphs make up the chain: a shared Japanese ModernBERT `backbone`, the",
        "  `text_proj` and `caption_proj` condition projectors, the reference-latent `speaker`",
        "  encoder, the `duration` predictor, the `dit` itself (one forward per Euler step, plus",
        "  one per classifier-free-guidance branch), and the DACVAE codec"
        f" ([{IRODORI_CODEC_MODEL}](https://huggingface.co/{IRODORI_CODEC_MODEL}))",
        "  as `codec_decoder` / `codec_encoder`.",
        "- **Text in, waveform out.** `generate()` returns f32 mono samples at the codec's sample",
        "  rate, ready for `encodeWav`. The decoder is run in tiles so that it also fits GPUs with",
        "  the default 128MiB storage-buffer limit; the tiling is bit-exact against a single-shot",
        "  decode.",
        "- The `codec_encoder` (waveform → latent) is shipped but **not wired up yet**: reference",
        "  voices are passed in as DACVAE latents, not as audio files.",
        "- Not readable by the upstream implementation (it's a different container with an embedded"
        f" graph); the reader is a pipeline that implements `{IRODORI_SUPPORTED_PIPELINE}`.",
        f"- Exporter used for the conversion: `{manifest['generator']}`. The distribution manifest"
        f" is `karume.json` (`{manifest['format']}`).",
    ]


def _irodori_base_weights() -> list[str]:
    """帰属節。格納形を変えていないので「変換したもの」としてだけ主張する。"""
    return [
        "## Base weights and attribution",
        "",
        "Converted into the container format — the original checkpoint is not distributed here.",
        "",
        f"- **Weights**: [{IRODORI_BASE_MODEL}](https://huggingface.co/{IRODORI_BASE_MODEL}),"
        " licensed **MIT**",
        "  (as of retrieval). The training / inference implementation it comes with",
        "  ([Aratako/Irodori-TTS](https://github.com/Aratako/Irodori-TTS)) is MIT as well;",
        "  Karume's runtime contains none of that code — it is an independent implementation that",
        "  reads these weights from its own container format.",
        f"- **Text backbone**: fine-tuned from"
        f" [{IRODORI_TEXT_BACKBONE_MODEL}](https://huggingface.co/{IRODORI_TEXT_BACKBONE_MODEL}),",
        "  licensed **MIT** (as of retrieval). It is redistributed here in the container format as",
        "  the `backbone` component, so that license travels with this repository too.",
        f"- **Codec**: [{IRODORI_CODEC_MODEL}](https://huggingface.co/{IRODORI_CODEC_MODEL}),"
        " licensed **MIT**",
        "  (as of retrieval). Upstream ships it as a separate repository; it is redistributed here",
        "  in the container format as the `codec_decoder` / `codec_encoder` components so that",
        "  text-to-audio runs from this repository alone.",
        "- **Changes made here**: conversion into the Karume container format only. No retraining,",
        "  no fine-tuning and **no quantization** — the weights are the source checkpoint's own",
        "  f32 values, re-laid out per graph.",
    ]


def _irodori_usage(manifest: Mapping[str, Any], repo: str) -> list[str]:
    model_name = manifest["defaultModel"]
    quant = _default_model(manifest)["defaultQuant"]
    return [
        "## Usage",
        "",
        "```ts",
        'import { encodeWav, IrodoriPipeline } from "jsr:@karume/models";',
        "",
        f"// Both options may be omitted: model defaults to {model_name}, quant to {quant}.",
        f'using pipeline = await IrodoriPipeline.fromPretrained("{repo}", {{',
        f'  model: "{model_name}",',
        f'  quant: "{quant}",',
        "});",
        "const audio = await pipeline.generate({",
        f'  text: "{IRODORI_DEMO_TEXT}",',
        f'  caption: "{IRODORI_DEMO_CAPTION}",',
        "  seed: 42,",
        "});",
        "await Deno.writeFile(",
        '  "out.wav",',
        "  encodeWav(audio.data, audio.sampleRate),",
        ");",
        "```",
        "",
        "`generate()` returns `{ data, sampleRate, frames, seed, forwards }`, where `data` is f32",
        "mono already trimmed to the predicted length. `generateLatent()` is the same run stopped",
        "one stage earlier: it returns `{ data, frames, latentDim, seed, forwards }` with the",
        "patched DACVAE latent, for callers that want the embedding rather than audio.",
        "`caption` and `speaker` are both optional: without them the voice is picked by the model",
        "alone, and the guidance branches for the missing conditions are skipped.",
        "Weights are fetched once and cached (verified against `karume.json`'s `size` / `sha256`).",
        "You can also build from bytes you fetched yourself (`IrodoriPipeline.fromAssets`).",
    ]


def _irodori_shape(model: Mapping[str, Any]) -> list[str]:
    """モデル固有の数（グラフの宣言と噛み合う側）— 利用者の入力上限がここで読める。"""
    config = model["pipelineConfig"]
    frame_rate = config["frameRate"]
    return [
        "### Shape",
        "",
        "Derived from the checkpoint's own config, and checked against the exported graphs when",
        "this repository was assembled.",
        "",
        f"- **text**: up to {config['maxTextLen']} tokens (BOS included), width"
        f" {config['textDim']}",
        f"- **caption**: up to {config['maxCaptionLen']} tokens (BOS included), width"
        f" {config['captionDim']}",
        f"- **reference speaker**: up to {config['speakerRows'] - 1} patched rows"
        f" ({config['speakerPatchSize']} latent frames each ="
        f" {(config['speakerRows'] - 1) * config['speakerPatchSize'] // frame_rate}s of audio),"
        f" width {config['speakerDim']}",
        f"- **latent**: up to {config['ditSymMax']} frames at {frame_rate} Hz"
        f" ({config['ditSymMax'] // frame_rate}s), width {config['latentDim']}",
        f"- **audio**: {config['sampleRate']} Hz mono, {config['hopLength']} samples per latent"
        " frame",
    ]


def _irodori_defaults(model: Mapping[str, Any]) -> list[str]:
    """実行時ノブ（`generate()` に渡さなかったものを埋める側）。"""
    config = model["pipelineConfig"]
    scales = config["cfgScales"]
    return [
        "### Defaults",
        "",
        "The sampler knobs are fixed by the manifest — `generate()` takes none of them.",
        "",
        f"- **steps**: {config['steps']} Euler steps (`initScale` {config['initScale']})",
        "- **guidance**: "
        + " / ".join(f"{name} {_knob(scale)}" for name, scale in scales.items())
        + f", applied for t in [{config['cfgMinT']}, {config['cfgMaxT']}]",
        f"- **duration**: clamped to [{config['minSeconds']}, {config['maxSeconds']}] seconds"
        " (the duration predictor decides within that, unless `durationSeconds` is passed)",
        "",
        "`seed` is the one knob the manifest does not carry — it defaults to `0`, and the same",
        "seed with the same request gives the same audio.",
    ]


def render_irodori_model_card(manifest: Mapping[str, Any], repo: str) -> str:
    """Irodori 配布形の `README.md` 本文を組み立てる（純関数・末尾改行つき）。"""
    _require_pipeline(manifest, IRODORI_SUPPORTED_PIPELINE)
    return _render(
        (
            _frontmatter(IRODORI_METADATA),
            ["", f"# {IRODORI_TITLE}", ""],
            _irodori_overview(manifest),
            [""],
            _irodori_base_weights(),
            [""],
            _models(manifest),
            [""],
            _irodori_usage(manifest, repo),
            *_model_sections(manifest, (_files, _quants, _irodori_shape, _irodori_defaults)),
        )
    )
