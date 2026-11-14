"""The model card (`README.md`) of the Anima distribution — pure functions over the manifest.

The generic rendering parts (frontmatter, model list, quant table, section assembly) live in
`karume.modelcard`. This module holds only **Anima-specific facts**: attribution (base model,
licence, the origin of baked-in LoRAs) and what this pipeline's card says.

MUST: **every number, download size, quant table and dtype label is derived from the manifest**
(the same MUST of `karume.modelcard` applies as is). The only constants allowed here are **facts
that do not exist** in the manifest.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import partial
from typing import Any

from _shared.upstream import hf_upstreams
from karume.modelcard import (
    CardMetadata,
    FileRefKey,
    Upstream,
    default_model,
    from_pretrained,
    frontmatter,
    introduction,
    model_sections,
    models,
    quants,
    render,
    require_pipeline,
)

#: The pipeline contract this template can describe (ADR 0041 §2 — per model).
SUPPORTED_PIPELINE = "anima/1"

PIPELINE_TAG = "text-to-image"

#: Licence (checked at the source): the base model's repository declares no licence in its card's
#: front matter and points to the original model card, `circlestone-labs/Anima`. It still carries a
#: v1.0 LICENSE.md (last changed 2026-05-22 — stale), but its weights are byte-identical, tensor by
#: tensor, to files that `circlestone-labs/Anima@f973fc41` publishes under v1.2 (measured
#: 2026-10-29), so v1.2 governs (ruling 37 a; the evidence is in THIRD_PARTY_NOTICES.md).
#: `license_link` is pinned to the commit of the bundled text: pointing at `main` would let the link
#: and the bundled LICENSE.md disagree silently on the day upstream publishes a new version. That
#: commit is `6e0b2da2` (2026-11-12, upstream's fix of two section references in v1.2), not the
#: `f973fc41` above — the weights were compared at `f973fc41`, and the text moved on after it.
ANIMA_METADATA = CardMetadata(
    pipeline_tag=PIPELINE_TAG,
    base_model=("circlestone-labs/Anima-Base-v1.0-Diffusers",),
    base_model_relation="quantized",
    license="other",
    license_name="circlestone-labs-non-commercial-license",
    license_link=(
        "https://huggingface.co/circlestone-labs/Anima/blob/"
        "6e0b2da2112573380abfb6a1435a60a994b254f2/LICENSE.md"
    ),
    tags=("text-to-image", "webgpu"),
)

#: Title of the official repository (karume-anima — CircleStone's 5 variants together).
ANIMA_OFFICIAL_TITLE = "Anima — Karume"

#: Title of the extra repository (karume-anima-extra — third-party fine-tunes).
ANIMA_EXTRA_TITLE = "Anima Extra — Karume"


@dataclass(frozen=True)
class UpstreamModel:
    """The origin of one model in this repository (held here: **a fact absent from the manifest**).

    Third-party fine-tunes are Derivatives of the same base, so the upstream licence flows through
    unchanged, but the permission fields on the distribution pages differ per author. In
    particular, an origin with `allowDifferentLicense` false requires "distribute under the same
    terms", so the fields are **listed verbatim per origin**.
    """

    title: str
    author: str
    source: str
    #: File name of the conversion source (`None` = the upstream diffusers repository itself).
    file: str | None
    #: Permission fields of the origin page (checked at the source — retrieved on
    #: {@link PERMISSIONS_RETRIEVED} below).
    permissions: tuple[tuple[str, str], ...]


#: Date the permission fields were checked at the source (shown "as of" in the card — the facts
#: may change later).
PERMISSIONS_RETRIEVED = "2026-08-22"

#: Model name → origin. The official CircleStone models (base / aesthetic / turbo) have no civitai
#: permission fields — only the CircleStone licence applies. The single-file checkpoints of the
#: official variants come from HF `circlestone-labs/Anima` (byte-identical to civitai 2458426).
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
    # The v1.0 generation is distributed alongside (maintainer ruling of 2026-09-01 — preferences
    # between versions differ for this family; an instance of ADR 0077's "the latest is not
    # necessarily better").
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

#: The Attribution Notice that §3(b) of the upstream licence requires **to be displayed verbatim**
#: (do not change a single character). Both `NOTICE.md` at the root of the distribution repository
#: (`anima/distribution.py`) and this card read it from here — holding the same legal wording in two
#: places independently lets one side silently drift into a form that fails the terms.
ATTRIBUTION_NOTICE = (
    "The CircleStone Model is licensed by CircleStone Labs LLC under the CircleStone"
    " Non-Commercial License. Copyright CircleStone Labs LLC.\n"
    "IN NO EVENT SHALL CIRCLESTONE LABS LLC BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,"
    " WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION"
    " WITH USE OF THIS MODEL."
)

#: The latter half of §3(a) of the upstream licence — the fixed sentence that **specifies** to
#: recipients that "the rights to use the CircleStone Models / Derivatives are granted directly by
#: the Company under this licence". The first half (providing a copy of the licence) is met by
#: bundling `LICENSE.md` verbatim, the latter half by this sentence (ruling of 2026-09-25). Party
#: names, defined terms and the licence name follow the spelling of the original
#: (`circlestone_license.txt`, its preamble and §1 / §3(a)). Like {@link ATTRIBUTION_NOTICE}, both
#: `NOTICE.md` (`anima/distribution.py`) and this card read it from here — holding the legal wording
#: in two places independently lets one side drift.
RIGHTS_GRANT_STATEMENT = (
    "Any rights to use the CircleStone Models and/or Derivatives in this repository are granted"
    " to you directly by CircleStone Labs LLC under the CircleStone Labs Non-Commercial License"
    " (LICENSE.md)."
)

#: The NVIDIA Open Model License part of the legal texts (rulings 39 a / 40 a, 2026-10-29). The
#: upstream card states that Anima is a "Derivative Model" of `nvidia/Cosmos-Predict2-2B-Text2Image`
#: and so is subject to that agreement insofar as it applies to Derivative Models. Agreement §3.1
#: requires its attribution notice, verbatim, in a "Notice" text file shipped with every copy (the
#: first paragraph — keep it on its own line, word for word); §3.2 requires "Built on NVIDIA Cosmos"
#: in the related product documentation (the last paragraph, shown in the card too). The agreement
#: text itself is the second part of `LICENSE.md` (`anima/distribution.py`, vendored from
#: <https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/>, version
#: of October 24, 2025). Like {@link ATTRIBUTION_NOTICE}, `NOTICE.md` and this card both read this
#: one constant so that the two copies of the legal wording cannot drift apart.
NVIDIA_NOTICE = (
    "Licensed by NVIDIA Corporation under the NVIDIA Open Model License\n"
    "\n"
    "The CircleStone Anima model is a Derivative Model of NVIDIA Cosmos-Predict2-2B-Text2Image"
    " (https://huggingface.co/nvidia/Cosmos-Predict2-2B-Text2Image), so the weights in this"
    " repository are also subject to the NVIDIA Open Model License Agreement"
    " (https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/)"
    " insofar as it applies to Derivative Models. A copy of that agreement is included in"
    " LICENSE.md. Both licenses apply; the NVIDIA agreement does not widen the CircleStone"
    " Non-Commercial License.\n"
    "\n"
    "Built on NVIDIA Cosmos."
)

#: The accepted set of the `resolution` knob in one line (**a fact absent from the manifest** — the
#: source of truth of the accepted set is `packages/models/src/anima/resolution.ts`, and per ADR
#: 0038 §2 it is not written into the manifest). Step = spatial compression 8 × patch 2; the lower
#: bound is latent 64 of the VAE tiled decoder (= 512px), the upper bound the 128 rows of the raw
#: rope table (= 2048px). This is a copy that follows when that file moves.
RESOLUTION_NOTE = (
    "Resolution — non-square is fine; each side on a 16 px grid, between 512 and 2048 px:"
)

#: Note for the top band (above 1920px). **The accepted set is not narrowed** (ruling of 2026-09-04
#: — the distribution's freedom is not cut without a measurement); instead the card says it runs at
#: positions outside the declaration. The row count of the raw table is set by the upper bound of
#: **the t axis** (128 rows), and the declaration of the h / w axes (`rope.max_size` = 120 tokens =
#: 1920px) is lower. Rows above 1920px are computed with the same formula as upstream, so the
#: numbers are defined and bit equality with the static form holds — any difference would only be
#: quality from positions outside the training distribution (unmeasured). A copy of the same fact as
#: the NOTE of `MAX_LATENT_SIDE` in `packages/models/src/anima/resolution.ts`.
RESOLUTION_EXTRAPOLATION_NOTE = (
    "  // Sides above 1920 px sit outside the range this model declares (its RoPE tables",
    "  // declare 120 tokens = 1920 px per side); those positions are extrapolated with the",
    "  // same formula, so what can differ there is the image quality.",
)

#: The cost of CFG in one line (**the same wording in both cards** — without a constant, the day
#: one side's line is rewrapped the other card's body moves too).
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
        "converted into the WebGPU inference runtime **Karume**'s container format (a `.krm` part",
        "sequence whose first part carries the graph and model descriptors). Runs as-is in the",
        "browser and in Deno.",
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
        "converted into the WebGPU inference runtime **Karume**'s container format (a `.krm` part",
        "sequence whose first part carries the graph and model descriptors). Runs as-is in the",
        "browser and in Deno.",
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
    """Policy of the Usage example: keep the minimal working form, and **list the optionals an
    ordinary use case is likely to need as commented-out lines** (the selectable values are listed
    in the same line's comment — derived from the manifest, so the list follows as models / quants
    are added). The reader takes the next step just by uncommenting.

    NOTE: `fromAssets` is **not shown** (ruling of 2026-08-29). The split distribution can be read
    now too (X2-101), but that surface is for local debugging and assumes you hold the bytes
    yourself; the ordinary entry point for a reader using HF is `fromPretrained`. Listing both
    would make the reader decide "which one to use".
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
        'import { encodePng } from "jsr:@karume/models";',
        'import { Pipeline } from "jsr:@karume/models/anima";',
        "",
        *from_pretrained(
            "Pipeline",
            repo,
            [
                f'  // model: "{model_name}", // default — available: {model_names}',
                f'  // quant: "{quant}", // default — available: {quant_names}',
            ],
            disposable="await using",
        ),
        "",
        "const image = await pipeline.generate({",
        '  prompt: "1girl, solo, long hair, blue eyes, school uniform, masterpiece",',
        "",
        f"  // steps: {defaults['steps']}, // default — {steps_note}",
        f"  // {RESOLUTION_NOTE}",
        *RESOLUTION_EXTRAPOLATION_NOTE,
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


def _upstream_model(name: str) -> UpstreamModel:
    """The origin of model `name` (the title's house sentence and the origins section read the
    same one); a model without one fails the render."""
    upstream = UPSTREAM_MODELS.get(name)
    if upstream is None:
        raise ValueError(f"Model {name!r} has no origin in card.py — cannot write its attribution")
    return upstream


def _origins(manifest: Mapping[str, Any], intro: tuple[str, ...]) -> list[str]:
    """Origins per model (**only the models in this repository**, listed from the manifest).

    MUST: the list comes from the manifest — holding the origin table independently of the
    assembly arguments means that the day the repository is assembled with one model left out, a
    card carrying "the attribution of a model that is not included" goes out.

    MUST: **keep the manifest's order too** (do not sort by name). `karume.modelcard` builds the
    `models` table and the per-model sections in manifest order, so sorting only here mixes two
    orders in the same card (it actually happened — `anima-copycat-…` came first and was pointed
    out, 2026-08-22). The manifest order is the assembly order itself, with the default model
    first, and it is also the order recommended to the reader.
    """
    lines = [
        "## Models and their origins",
        "",
        *intro,
    ]
    for name in manifest["models"]:
        upstream = _upstream_model(name)
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
    """Licence section — the **combination** of the upstream redistribution terms and the
    third-party fine-tunes' permissions.

    The Notice text is also shown verbatim in the card (the same reason as on the turbo side — the
    display requirement of §3(b)). The statement of the rights grant in the latter half of §3(a)
    ({@link RIGHTS_GRANT_STATEMENT}) is also emitted from the same single constant as `NOTICE.md`.
    """
    # Manifest order, as in the origins section (the same MUST — no two orders in one card).
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
        "CircleStone Labs Non-Commercial License v1.2 (non-commercial use only). This",
        "repository ships `LICENSE.md` (the full text of that license, followed by the NVIDIA",
        "Open Model License Agreement) and `NOTICE.md` (these attribution notices plus the list",
        "of modifications).",
        "",
        RIGHTS_GRANT_STATEMENT,
        "",
        ATTRIBUTION_NOTICE,
        "",
        NVIDIA_NOTICE,
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


#: Introduction of the origins section (the facts differ per repository — official / extra).
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


def _introduction(manifest: Mapping[str, Any], title: str) -> list[str]:
    """The title and the house sentence over the models this repository holds (manifest order).

    The models are named by their titles — the fine-tunes are not on the Hub. An official model is
    CircleStone's (the publisher of the base model's account); a community fine-tune (one with
    permissions listed on its source page, as {@link _base_license} reads them) is its author's.
    The §3(d) non-endorsement sentence of the License section stays as the licence words it; this
    one does not replace it.
    """
    licensor = hf_upstreams(ANIMA_METADATA.base_model)[0].author
    upstreams: list[Upstream] = []
    for name in manifest["models"]:
        upstream = _upstream_model(name)
        author = upstream.author if upstream.permissions else licensor
        upstreams.append(Upstream(upstream.title, author))
    return introduction(title, upstreams)


def render_base_card(
    manifest: Mapping[str, Any],
    repo: str,
    abbreviations: Mapping[str, str],
    host_assets: Mapping[FileRefKey, int] = {},
) -> str:
    """The `README.md` body of the official repository's distribution (CircleStone's 5 variants).

    `abbreviations` maps the component-override tokens of preset names (source of truth:
    `anima.distribution` — ADR 0074 decision 4). It is a fact absent from the manifest, so it is
    taken as an argument rather than copied as a constant.
    """
    require_pipeline(manifest, SUPPORTED_PIPELINE)
    return render(
        (
            frontmatter(ANIMA_METADATA),
            _introduction(manifest, ANIMA_OFFICIAL_TITLE),
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
                (partial(quants, abbreviations=abbreviations, host_assets=host_assets), _defaults),
            ),
        )
    )


def render_extra_card(
    manifest: Mapping[str, Any],
    repo: str,
    abbreviations: Mapping[str, str],
    host_assets: Mapping[FileRefKey, int] = {},
) -> str:
    """The `README.md` body of the extra repository's distribution (third-party fine-tunes)."""
    require_pipeline(manifest, SUPPORTED_PIPELINE)
    return render(
        (
            frontmatter(ANIMA_METADATA),
            _introduction(manifest, ANIMA_EXTRA_TITLE),
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
                (partial(quants, abbreviations=abbreviations, host_assets=host_assets), _defaults),
            ),
        )
    )
