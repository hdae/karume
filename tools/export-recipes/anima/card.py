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
from dataclasses import dataclass
from functools import partial
from typing import Any

from karume.modelcard import (
    CardMetadata,
    default_model,
    files,
    from_pretrained,
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

#: 公式リポ（karume-anima — CircleStone の 5 変種同居）のタイトル。
ANIMA_OFFICIAL_TITLE = "Anima — Karume"

#: 追加学習リポ（karume-anima-extra — 第三者 fine-tune）のタイトル。
ANIMA_EXTRA_TITLE = "Anima Extra — Karume"


@dataclass(frozen=True)
class UpstreamModel:
    """このリポの 1 モデルの出所（**manifest には現れない事実**なのでここが持つ）。

    第三者 fine-tune は同じ base の Derivative なので上流ライセンスはそのまま流れてくるが、
    配布ページ側の許諾欄は作者ごとに違う。特に `allowDifferentLicense` が false の出所は
    「同じ条件で配ること」を要求するので、**出所ごとに逐語で載せる**。
    """

    title: str
    author: str
    source: str
    #: 変換元のファイル名（`None` = 上流の diffusers リポそのもの）。
    file: str | None
    #: 出所ページの許諾欄（実地確認 — 取得日は下の {@link PERMISSIONS_RETRIEVED}）。
    permissions: tuple[tuple[str, str], ...]


#: 許諾欄を実地確認した日（カードに「as of」で出す — 後から変わりうる事実だから）。
PERMISSIONS_RETRIEVED = "2026-08-22"

#: モデル名 → 出所。CircleStone 公式（base / aesthetic / turbo）は civitai の許諾欄を
#: 持たない — 掛かるのは CircleStone のライセンス 1 本だけ。公式変種の単一ファイル
#: checkpoint は HF `circlestone-labs/Anima`（civitai 2458426 と同一バイト配布）から。
UPSTREAM_MODELS: Mapping[str, UpstreamModel] = {
    "anima-turbo-v1.1": UpstreamModel(
        title="Anima Turbo v1.1",
        author="circlestone_labs",
        source="https://huggingface.co/circlestone-labs/Anima",
        file="anima-turbo-v1.1.safetensors",
        permissions=(),
    ),
    "anima-v1.0": UpstreamModel(
        title="Anima Base v1.0",
        author="circlestone_labs",
        source="https://huggingface.co/circlestone-labs/Anima-Base-v1.0-Diffusers",
        file=None,
        permissions=(),
    ),
    "anima-aesthetic-v1.1": UpstreamModel(
        title="Anima Aesthetic v1.1",
        author="circlestone_labs",
        source="https://huggingface.co/circlestone-labs/Anima",
        file="anima-aesthetic-v1.1.safetensors",
        permissions=(),
    ),
    # v1.0 世代も並行配布する（2026-09-01 ユーザー裁定 — この系はバージョン間で好みが
    # 分かれるため。ADR 0077 の「最新が優れるとは限らない」の実例）。
    "anima-turbo-v1.0": UpstreamModel(
        title="Anima Turbo v1.0",
        author="circlestone_labs",
        source="https://huggingface.co/circlestone-labs/Anima",
        file="anima-turbo-v1.0.safetensors",
        permissions=(),
    ),
    "anima-aesthetic-v1.0": UpstreamModel(
        title="Anima Aesthetic v1.0",
        author="circlestone_labs",
        source="https://huggingface.co/circlestone-labs/Anima",
        file="anima-aesthetic-v1.0.safetensors",
        permissions=(),
    ),
    "anima-wai-v1.0": UpstreamModel(
        title="WAI-ANIMA v1.0 (base 1.0)",
        author="WAI0731",
        source="https://civitai.com/models/2544636/wai-anima",
        file="waiANIMA_v10Base10.safetensors",
        permissions=(
            ("allowNoCredit", "true"),
            ("allowCommercialUse", "Image / RentCivit"),
            ("allowDerivatives", "true"),
            ("allowDifferentLicense", "true"),
        ),
    ),
    "anima-copycat-20260610": UpstreamModel(
        title="copycat-anima 20260610",
        author="calculater",
        source="https://civitai.com/models/2377376/copycat-anima",
        file="copycatAnima_20260610.safetensors",
        permissions=(
            ("allowNoCredit", "true"),
            ("allowCommercialUse", "Image / RentCivit"),
            ("allowDerivatives", "true"),
            ("allowDifferentLicense", "false"),
        ),
    ),
}

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

#: CFG の代償を 1 行で言ったもの（**両方のカードで同じ文面** — 定数にしないと、片方の行を
#: 折り返した日にもう片方のカード本文まで動く）。
CFG_NOTE = "  // Classifier-free guidance runs a second (uncond) branch — twice the work per step."


def _official_overview(manifest: Mapping[str, Any]) -> list[str]:
    model_name = manifest["defaultModel"]
    defaults = default_model(manifest)["pipelineConfig"]["defaults"]
    base_model = ANIMA_METADATA.base_model[0]
    lines = [
        "## What is this",
        "",
        "The official CircleStone **Anima** models —"
        f" [{base_model}](https://huggingface.co/{base_model}) and its official variants —",
        "converted into the WebGPU inference runtime **Karume**'s container format (safetensors =",
        "weights + a graph JSON embedded in `__metadata__`, split across numbered shards when a",
        "component is too large for one file). Runs as-is in the browser and in Deno.",
        "",
        f"- The default model is `{model_name}` — **{defaults['steps']} steps / guidance"
        f" {defaults['guidanceScale']}** by default.",
    ]
    if defaults["guidanceScale"] == 1:
        lines += [
            "  At guidance 1 the second CFG branch is skipped and **the negative prompt is not"
            " used** —",
            "  pick a model that runs classifier-free guidance (see the model list below) when"
            " you want",
            "  the negative prompt to take effect.",
        ]
    lines += [
        "- Not readable by diffusers (it's a different container with an embedded graph); the"
        f" reader is a pipeline that implements `{SUPPORTED_PIPELINE}`.",
        f"- Exporter used for the conversion: `{manifest['generator']}`. The distribution manifest"
        f" is `karume.json` (`{manifest['format']}`).",
    ]
    return lines


def _extra_overview(manifest: Mapping[str, Any]) -> list[str]:
    defaults = default_model(manifest)["pipelineConfig"]["defaults"]
    base_model = ANIMA_METADATA.base_model[0]
    return [
        "## What is this",
        "",
        "Community fine-tunes of the CircleStone **Anima** base model"
        f" ([{base_model}](https://huggingface.co/{base_model})),",
        "converted into the WebGPU inference runtime **Karume**'s container format (safetensors =",
        "weights + a graph JSON embedded in `__metadata__`, split across numbered shards when a",
        "component is too large for one file). Runs as-is in the browser and in Deno.",
        "",
        f"- Ordinary many-step sampling — **{defaults['steps']} steps / guidance"
        f" {defaults['guidanceScale']}** by default. Classifier-free guidance is on, which is",
        "  what makes the **negative prompt take effect**. The official models (base / Aesthetic /",
        "  Turbo) live in [hdae/karume-anima](https://huggingface.co/hdae/karume-anima).",
        "- Not readable by diffusers (it's a different container with an embedded graph); the"
        f" reader is a pipeline that implements `{SUPPORTED_PIPELINE}`.",
        f"- Exporter used for the conversion: `{manifest['generator']}`. The distribution manifest"
        f" is `karume.json` (`{manifest['format']}`).",
    ]


def _usage(manifest: Mapping[str, Any], repo: str) -> list[str]:
    """Usage 例の方針: 動く最小形は生かし、**普通のユースケースで使いそうな optional は
    コメントアウトで併記**する（選べる値も同じ行のコメントに列挙 — manifest から機械導出する
    ので、モデル / quant が増えれば列挙も追従する）。読者がコメントを外すだけで次の一歩へ
    進める形。

    NOTE: `fromAssets` は**案内しない**（2026-08-29 裁定）。分割配布形も読めるようになった
    （X2-101）が、あちらはバイト列を自分で持っている前提のローカルデバッグ向けの面で、HF から
    使う読者の普通の入口は `fromPretrained`。両方を並べると「どちらを使うのか」を読者に
    判断させることになる。
    """
    model_name = manifest["defaultModel"]
    model = default_model(manifest)
    quant = model["defaultQuant"]
    model_names = " / ".join(sorted(manifest["models"]))
    quant_names = " / ".join(sorted(model["quants"]))
    defaults = model["pipelineConfig"]["defaults"]
    resolution = defaults["resolution"]
    guidance = defaults["guidanceScale"]
    steps_note = (
        "the model is distilled for few-step sampling"
        if guidance == 1
        else "more steps trade time for detail"
    )
    cfg_lines = (
        [
            CFG_NOTE,
            "  // It is skipped at guidanceScale 1, where a negativePrompt is refused rather than",
            "  // silently ignored, so the two lines below only make sense together:",
            "  // guidanceScale: 5,",
            f'  // negativePrompt: "{defaults["negativePrompt"]}",',
        ]
        if guidance == 1
        else [
            CFG_NOTE,
            "  // It is on by default here, which is what makes the negative prompt take effect:",
            f"  // guidanceScale: {guidance}, // default",
            f'  // negativePrompt: "{defaults["negativePrompt"]}", // default',
        ]
    )
    return [
        "## Usage",
        "",
        "```ts",
        'import { AnimaPipeline, encodePng } from "jsr:@karume/models";',
        "",
        *from_pretrained(
            "AnimaPipeline",
            repo,
            [
                f'  // model: "{model_name}", // default — available: {model_names}',
                f'  // quant: "{quant}", // default — available: {quant_names}',
            ],
        ),
        "",
        "const image = await pipeline.generate({",
        '  prompt: "1girl, solo, long hair, blue eyes, school uniform, masterpiece",',
        "",
        f"  // steps: {defaults['steps']}, // default — {steps_note}",
        f"  // {RESOLUTION_NOTE}",
        f"  // resolution: {{ width: {resolution['width']},"
        f" height: {resolution['height']} }}, // default",
        "",
        *cfg_lines,
        "",
        "  seed: 42, // same seed + same request → same image",
        "});",
        "",
        "const png = await encodePng(image.data, image.width, image.height);",
        'await Deno.writeFile("anima.png", png);',
        "```",
        "",
        "Weights are fetched once and cached (verified against `karume.json`'s `size` / `sha256`).",
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


def _origins(manifest: Mapping[str, Any], intro: tuple[str, ...]) -> list[str]:
    """モデルごとの出所（**このリポに入っているモデルだけ**を manifest から引いて並べる）。

    MUST: 並びは manifest 由来にする — 出所の表を組み立ての引数と独立に持つと、モデルを
    1 つ落として組んだ日に「入っていないモデルの帰属」が載ったカードが出る。

    MUST: **順序も manifest のまま**（名前順に並べ替えない）。`models` 表とモデル別節は
    `karume.modelcard` が manifest の並びのまま組むので、ここだけ並べ替えると同じカードの中で
    2 通りの順序が混在する（実際に `anima-copycat-…` が先頭へ来て指摘された — 2026-08-22）。
    manifest の並びは既定モデルが先頭に来る組み立て順そのもので、読者にとっての推奨順でもある。
    """
    lines = [
        "## Models and their origins",
        "",
        *intro,
    ]
    for name in manifest["models"]:
        upstream = UPSTREAM_MODELS.get(name)
        if upstream is None:
            raise ValueError(f"モデル {name!r} の出所が card.py に無い — 帰属を書けない")
        lines += [
            "",
            f"### `{name}` — {upstream.title}",
            "",
            f"- **Author**: {upstream.author}",
            f"- **Source**: {upstream.source}",
        ]
        if upstream.file is not None:
            lines.append(f"- **Converted from**: `{upstream.file}`")
        if upstream.permissions:
            lines += [
                "",
                f"Permissions listed on the source page (as of {PERMISSIONS_RETRIEVED}):",
                "",
                *(f"- `{key}`: {value}" for key, value in upstream.permissions),
            ]
    return lines


def _base_license(manifest: Mapping[str, Any]) -> list[str]:
    """ライセンス節 — 上流の再配布条件と、第三者 fine-tune 側の許諾の**重ね合わせ**。

    Notice の本文はカードにも逐語で出す（turbo 側と同じ理由 — §3(b) の掲示要件）。
    """
    # 並びは出所節と同じく manifest のまま（同 MUST — カード内で順序を 2 通りにしない）。
    fine_tunes = [name for name in manifest["models"] if UPSTREAM_MODELS[name].permissions]
    same_license = [
        name
        for name in fine_tunes
        if dict(UPSTREAM_MODELS[name].permissions).get("allowDifferentLicense") == "false"
    ]
    lines = [
        "## License",
        "",
        "Every model here derives from the CircleStone Anima base model and stays under the",
        "CircleStone Non-Commercial License (non-commercial use only). This repository ships",
        "`LICENSE.md` (the full license text) and `NOTICE.md` (this attribution plus the list of",
        "modifications).",
        "",
        ATTRIBUTION_NOTICE,
        "",
        "- Outputs you generate are yours to use for any purpose, including commercially",
        "  (license §2(e)); the non-commercial restriction applies to the model weights and",
        "  derivatives, not to outputs.",
    ]
    if fine_tunes:
        lines += [
            "- The community fine-tunes are redistributed under the permissions their source"
            " pages state",
            "  (listed per model above); those permissions do not widen the base model's license.",
        ]
    if same_license:
        listed = ", ".join(f"`{name}`" for name in same_license)
        lines += [
            f"- {listed}: the source page sets `allowDifferentLicense` to false, so this"
            " redistribution",
            "  keeps the same terms — do not relicense it.",
        ]
    lines += [
        "- This is not an official product of CircleStone Labs LLC, and it is not endorsed,"
        " approved or",
        "  validated by CircleStone Labs LLC.",
    ]
    return lines


#: 出所節の導入（リポごとに事実が違う — 公式 / 追加学習）。
OFFICIAL_ORIGINS_INTRO = (
    "Each model below is an official CircleStone release — the Anima base model itself or an",
    "official variant of it. The text encoder, VAE and tokenizers are shared across them.",
)
EXTRA_ORIGINS_INTRO = (
    "Each model below is a community fine-tune of the CircleStone Anima base model. The text",
    "encoder, VAE and tokenizers are shared with the official repository",
    "([hdae/karume-anima](https://huggingface.co/hdae/karume-anima)) through pinned",
    "cross-repository references in `karume.json`.",
)


def render_base_card(
    manifest: Mapping[str, Any], repo: str, abbreviations: Mapping[str, str]
) -> str:
    """公式リポ（CircleStone の 5 変種同居）配布形の `README.md` 本文。

    `abbreviations` は席名の部品上書きトークンの対応表（正本は `anima.distribution` —
    ADR 0074 決定 4）。manifest に無い事実なので、定数として写さず引数で受ける。
    """
    require_pipeline(manifest, SUPPORTED_PIPELINE)
    return render(
        (
            frontmatter(ANIMA_METADATA),
            ["", f"# {ANIMA_OFFICIAL_TITLE}", ""],
            _official_overview(manifest),
            [""],
            _origins(manifest, OFFICIAL_ORIGINS_INTRO),
            [""],
            _base_license(manifest),
            [""],
            models(manifest),
            [""],
            _usage(manifest, repo),
            *model_sections(
                manifest,
                (files, partial(quants, abbreviations=abbreviations), _defaults),
            ),
        )
    )


def render_extra_card(
    manifest: Mapping[str, Any], repo: str, abbreviations: Mapping[str, str]
) -> str:
    """追加学習リポ（第三者 fine-tune）配布形の `README.md` 本文。"""
    require_pipeline(manifest, SUPPORTED_PIPELINE)
    return render(
        (
            frontmatter(ANIMA_METADATA),
            ["", f"# {ANIMA_EXTRA_TITLE}", ""],
            _extra_overview(manifest),
            [""],
            _origins(manifest, EXTRA_ORIGINS_INTRO),
            [""],
            _base_license(manifest),
            [""],
            models(manifest),
            [""],
            _usage(manifest, repo),
            *model_sections(
                manifest,
                (files, partial(quants, abbreviations=abbreviations), _defaults),
            ),
        )
    )
