"""The model card (`README.md`) of the MOSS-Image distribution — a pure function of the manifest.

SPEC: docs/spec/distribution-cards.md#the-model-card

The generic rendering parts (front matter, the model list, the quant table, the section assembly)
are `karume.modelcard`'s. This module holds what is specific to MOSS-Image: the attribution
(upstream, pinned revision, licence, the VAE's origin) and what the card of this pipeline says.

MUST: every number, download size, quant row, dtype label and default comes from the manifest
(`karume.modelcard`'s MUST). The constants here are facts the manifest does not hold — the upstream
pins (`moss.sources` is their source of truth) and the public entry's names.

MUST: no torch import (`import dist` must not load it — `tests/test_dist_driver.py`).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from functools import partial
from typing import Any

from _shared.upstream import hf_upstreams
from karume.modelcard import (
    APACHE_LICENSE_REFERENCE,
    APACHE_MODIFICATION_CLAUSE,
    RUNTIME_SENTENCE,
    CardMetadata,
    FileRefKey,
    UpstreamQuote,
    changes_made_here,
    default_model,
    from_pretrained,
    frontmatter,
    hf_link,
    introduction,
    knob,
    model_sections,
    models,
    pinned,
    quants,
    quotation,
    render,
    require_pipeline,
    upstream_cards,
)
from moss.sources import SOURCE, TEXT_ENCODER_ORIGIN, VAE_ORIGINS

#: The pipeline contract this template describes (ADR 0041 §2).
MOSS_SUPPORTED_PIPELINE = "moss-image/1"

#: The HF pipeline tag (upstream's).
MOSS_PIPELINE_TAG = "text-to-image"

MOSS_TITLE = "MOSS-Image-SFT — Karume"


#: The upstream technical report (the arXiv link of the MOSS-Image-SFT model card at the pinned
#: revision).
MOSS_TECHNICAL_REPORT = "https://arxiv.org/abs/2610.23374"

#: Which part of the decode path each VAE origin supplies (keyed by repository; a new origin in
#: `moss.sources.VAE_ORIGINS` without an entry fails the render). The encoder and `quant_conv` are
#: not exported (`moss.export_vae`), so the FLUX.2 [klein] 4B VAE supplies only the statistics.
MOSS_VAE_PARTS: Mapping[str, str] = {
    "black-forest-labs/FLUX.2-small-decoder": "the decoder and `post_quant_conv`",
    "black-forest-labs/FLUX.2-klein-4B": "the latent batch-normalization statistics (`vae/`)",
}


#: The non-endorsement statement Black Forest Labs' cards ask for ("Trademarks & IP": the BFL and
#: FLUX marks in modified versions "must not ... imply sponsorship or endorsement"; maintainer
#: decision 2026-11-01: kept only where an upstream asks for it, so it names Black Forest Labs
#: alone). The card's attribution section and `NOTICE.md` both write it from here.
MOSS_BFL_NON_ENDORSEMENT = (
    "This is not an official release of Black Forest Labs, and it is not endorsed by them; the"
    " Black Forest Labs and FLUX names and marks are used only to describe where the VAE weights"
    " come from."
)

#: What Black Forest Labs' cards of both VAE origins state of their use, quoted verbatim (the cards
#: at the pinned commits of `moss.sources.VAE_ORIGINS`, read 2026-11-27): both point to its usage
#: policies without a link, and the FLUX.2 [klein] 4B card says its statements do not modify the
#: license.
_VAE_REVISIONS = {origin.source.repo: origin.source.revision for origin in VAE_ORIGINS}
MOSS_BFL_QUOTES: Mapping[str, UpstreamQuote] = {
    "black-forest-labs/FLUX.2-small-decoder": UpstreamQuote(
        repo="black-forest-labs/FLUX.2-small-decoder",
        revision=_VAE_REVISIONS["black-forest-labs/FLUX.2-small-decoder"],
        sections=("Out-of-Scope Use",),
        paragraphs=(
            "This model and its derivatives may not be used outside the scope of the license,"
            " including for unlawful, fraudulent, defamatory, abusive, or otherwise violative"
            " purposes as further explained in our Usage Policies.",
        ),
    ),
    "black-forest-labs/FLUX.2-klein-4B": UpstreamQuote(
        repo="black-forest-labs/FLUX.2-klein-4B",
        revision=_VAE_REVISIONS["black-forest-labs/FLUX.2-klein-4B"],
        sections=("Out-of-Scope Use", "Responsible AI Development"),
        paragraphs=(
            "Nothing contained in this Model Card should be interpreted as or deemed a restriction"
            " or modification to the license the model is released under.",
            "7. Policies. Acceptable use of our models and access to our API are governed by"
            " policies set out in applicable documentation, including FLUX Non-Commercial License"
            " (for our non-commercial open-weight users); Developer Terms of Service, Self-Hosted"
            " Commercial License Terms, and Usage Policy (for our commercial open-weight model"
            " users); and Developer Terms of Service, FLUX API Service Terms, and Usage Policy (for"
            " our API users). These prohibit the generation of unlawful content or the use of"
            " generated content for unlawful, defamatory, or abusive purposes.",
        ),
    ),
}

#: Every quotation of an upstream card the card writes (checked against the cached cards by
#: `tests/test_dist_driver.py`).
UPSTREAM_QUOTES: tuple[UpstreamQuote, ...] = tuple(MOSS_BFL_QUOTES.values())


def _metadata() -> CardMetadata:
    """The front matter: `base_model` lists every upstream whose bytes the repository stores — the
    checkpoint, the text encoder's release and both VAE origins, FLUX.2 [klein] 4B included for the
    latent statistics it supplies (the maintainer's ruling of 2026-11-27: every stored upstream is
    listed)."""
    return CardMetadata(
        pipeline_tag=MOSS_PIPELINE_TAG,
        base_model=(
            SOURCE.repo,
            TEXT_ENCODER_ORIGIN.repo,
            *(origin.source.repo for origin in VAE_ORIGINS),
        ),
        license=SOURCE.license,
        tags=(MOSS_PIPELINE_TAG, "webgpu", "moss-image"),
        # The transformer and the text encoder are re-encoded in int8 (the VAE keeps its f32).
        base_model_relation="quantized",
    )


def _overview(manifest: Mapping[str, Any]) -> list[str]:
    return [
        "## What is this",
        "",
        "A text-to-image distribution: MOSS-Image-SFT (a 5B single-stream diffusion transformer",
        "with a Qwen3.5-9B text encoder and the FLUX.2 VAE), converted into the WebGPU inference",
        "runtime **Karume**'s container format (a `.krm` part sequence whose first part carries",
        "the graph and model descriptors).",
        RUNTIME_SENTENCE,
        "",
        "- Everything runs in Karume: the chat template and the tokenizer on the host, then the",
        "  text encoder, the transformer and the VAE on the GPU, one stage at a time (each stage",
        "  closes before the next one opens, so the text encoder and the transformer are never",
        "  resident together).",
        "- The negative branch of classifier-free guidance is upstream's: the transformer without",
        "  block 22 (zero-based). Guidance 1 runs no negative branch.",
        "- The VAE decodes untiled, as upstream does. The largest accepted area is set by the",
        "  device's storage-binding limit (about 2.8 million pixels at 2 GiB).",
        "- The seeded noise is Karume's own generator, not torch's: the same seed does not give",
        "  upstream's image.",
        "- Not readable by diffusers (it's a different container with an embedded graph); the"
        f" reader is a pipeline that implements `{MOSS_SUPPORTED_PIPELINE}`.",
        f"- Exporter used for the conversion: `{manifest['generator']}`. The distribution manifest"
        f" is `karume.json` (`{manifest['format']}`).",
    ]


def _base_weights() -> list[str]:
    lines = [
        "## Base weights and attribution",
        "",
        "Converted and partly quantized — the original checkpoint is not distributed here.",
        "",
        f"- **Weights**: {pinned(SOURCE.repo, SOURCE.revision)}, licensed **{SOURCE.license}**"
        f" {APACHE_LICENSE_REFERENCE}.",
        f"- **Authors**: MOSS-Image is released by OpenMOSS ([technical report]"
        f"({MOSS_TECHNICAL_REPORT})).",
        "- **Text encoder** (`text_encoder`, `tokenizer`): the checkpoint's bundled Qwen3.5 text"
        f" encoder, from {pinned(TEXT_ENCODER_ORIGIN.repo, TEXT_ENCODER_ORIGIN.revision)}"
        f" ({hf_upstreams([TEXT_ENCODER_ORIGIN.repo])[0].author}), licensed"
        f" **{TEXT_ENCODER_ORIGIN.license}** (as of retrieval), and its tokenizer. Only the text"
        " tower is redistributed here; its copyright notice is kept in `NOTICE.md`.",
        "- **VAE decoder** (`vae_decoder_pre` / `vae_decoder_attn` / `vae_decoder_post`): the"
        " checkpoint's bundled FLUX.2 VAE with the FLUX.2 small decoder (Black Forest Labs)."
        " Only the decode path is redistributed; the VAE encoder is not included:",
    ]
    lines += [
        f"  - {MOSS_VAE_PARTS[origin.source.repo]}:"
        f" {pinned(origin.source.repo, origin.source.revision)}, licensed"
        f" **{origin.source.license}** (as of retrieval)"
        for origin in VAE_ORIGINS
    ]
    lines += [
        # Maintainer decision 2026-11-01: passed through as information. Neither card links the
        # policy at the pinned revisions, so the cards' own words are quoted (the maintainer's
        # ruling of 2026-11-27 on upstream statements).
        "  - Black Forest Labs' model cards of both origins point to its usage policies, without a"
        " link of their own; this is informational, not a term of their **apache-2.0** license.",
        *quotation(
            MOSS_BFL_QUOTES["black-forest-labs/FLUX.2-small-decoder"],
            "    The FLUX.2 small decoder's card",
            indent="    ",
        ),
        "",
        *quotation(
            MOSS_BFL_QUOTES["black-forest-labs/FLUX.2-klein-4B"],
            "    The FLUX.2 [klein] 4B card",
            indent="    ",
        ),
    ]
    lines += [
        f"{changes_made_here(APACHE_MODIFICATION_CLAUSE)} the",
        "  transformer's and the text encoder's weight matrices are stored as int8 (one scale per",
        "  output channel); the VAE keeps the source f32 values. The transformer is split into a",
        "  text-refiner graph and the joint-sequence graph (plus the same graph without block 22,",
        "  which borrows its weights), the VAE decoder into three graphs around its middle",
        "  attention, and the text encoder keeps its text tower only. No retraining and no",
        "  fine-tuning.",
        f"- {MOSS_BFL_NON_ENDORSEMENT}",
        *upstream_cards([hf_link(repo) for repo in _metadata().base_model]),
    ]
    return lines


def _usage(manifest: Mapping[str, Any], repo: str) -> list[str]:
    """The minimal working call, with the knobs a normal use reaches for commented out (their
    values from the manifest)."""
    model_name = manifest["defaultModel"]
    model = default_model(manifest)
    defaults = model["pipelineConfig"]["defaults"]
    shift = model["pipelineConfig"]["scheduler"]["shift"]
    model_names = " / ".join(sorted(manifest["models"]))
    quant_names = " / ".join(model["quants"])
    return [
        "## Usage",
        "",
        "```ts",
        'import { imageToRgba, Pipeline } from "jsr:@karume/models/moss";',
        "",
        *from_pretrained(
            "Pipeline",
            repo,
            [
                f'  // model: "{model_name}", // default — available: {model_names}',
                f'  // quant: "{model["defaultQuant"]}", // default — available: {quant_names}',
            ],
            disposable="await using",
        ),
        "",
        "const image = await pipeline.generate({",
        '  prompt: "A red bicycle leaning against a white wall, morning light",',
        f"  // width: {defaults['width']}, height: {defaults['height']}, // pixels, multiples"
        " of 16",
        f"  // steps: {defaults['steps']}, guidance: {defaults['guidance']}, shift: {shift},",
        f"  // negativePrompt: {json.dumps(defaults['negativePrompt'], ensure_ascii=False)},",
        "  // seed: 0,",
        "});",
        "",
        "// [3, height, width] f32 in [-1, 1] → 8-bit RGBA",
        "const rgba = imageToRgba(image);",
        "```",
        "",
        "Weights are fetched once and cached (verified against `karume.json`'s `size` / `sha256`).",
        "Concurrent `generate` calls are queued.",
    ]


def _defaults(model: Mapping[str, Any]) -> list[str]:
    """The generation defaults the manifest declares (upstream's)."""
    config = model["pipelineConfig"]
    defaults = config["defaults"]
    return [
        "### Generation defaults",
        "",
        "Upstream's own defaults, used for every field a request leaves out:",
        "",
        f"- size {defaults['width']} × {defaults['height']}, {defaults['steps']} steps,"
        f" guidance {defaults['guidance']}, shift {config['scheduler']['shift']}",
        f"- negative prompt: {_prompt(defaults['negativePrompt'])} (used when guidance ≠ 1)",
    ]


def _prompt(text: str) -> str:
    """A prompt in prose (the empty one has no visible code span)."""
    return "the empty prompt" if text == "" else knob(text)


def render_moss_model_card(
    manifest: Mapping[str, Any],
    repo: str,
    host_assets: Mapping[FileRefKey, int] = {},
) -> str:
    """The `README.md` body of the MOSS-Image distribution (a pure function; trailing newline).

    The preset names' component token `dit` is the weights name itself, so the quant table needs
    no abbreviation line (`karume.modelcard.quants`).
    """
    require_pipeline(manifest, MOSS_SUPPORTED_PIPELINE)
    metadata = _metadata()
    return render(
        (
            frontmatter(metadata),
            # Beside, not instead of, the FLUX marks sentence of the attribution section — that one
            # is the wording Black Forest Labs' cards ask for.
            introduction(MOSS_TITLE, hf_upstreams(metadata.base_model)),
            _overview(manifest),
            [""],
            _base_weights(),
            [""],
            models(manifest),
            [""],
            _usage(manifest, repo),
            *model_sections(
                manifest,
                (
                    partial(quants, host_assets=host_assets),
                    _defaults,
                ),
            ),
        )
    )
