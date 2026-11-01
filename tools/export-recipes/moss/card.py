"""The model card (`README.md`) of the MOSS-Image distribution — a pure function of the manifest.

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

from karume.modelcard import (
    CardMetadata,
    FileRefKey,
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
from moss.sources import SOURCE, TEXT_ENCODER_ORIGIN, VAE_ORIGINS

#: The pipeline contract this template describes (ADR 0041 §2).
MOSS_SUPPORTED_PIPELINE = "moss-image/1"

#: The HF pipeline tag (upstream's).
MOSS_PIPELINE_TAG = "text-to-image"

MOSS_TITLE = "MOSS-Image-SFT — Karume"

#: Where the licence text lives (the same text as the repository's `LICENSE.md` — Apache 2.0 §4(a)).
MOSS_LICENSE_TEXT_LINK = "https://www.apache.org/licenses/LICENSE-2.0"

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


#: The VAE origin whose weights the decoder graphs carry (the one that accounts for `decoder.`).
#: The other origin, FLUX.2 [klein] 4B, supplies only the 256 latent BatchNorm statistics, so it is
#: not a `base_model`: listing it would show this repository as a quantization of a 4B transformer
#: in the Hub's model tree.
MOSS_VAE_DECODER_ORIGIN = next(o.source for o in VAE_ORIGINS if "decoder." in o.prefixes)


def _metadata() -> CardMetadata:
    return CardMetadata(
        pipeline_tag=MOSS_PIPELINE_TAG,
        base_model=(SOURCE.repo, TEXT_ENCODER_ORIGIN.repo, MOSS_VAE_DECODER_ORIGIN.repo),
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
        "the graph and model descriptors). Runs in the browser, in Node and in Deno.",
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
        f"- **Weights**: [{SOURCE.repo}](https://huggingface.co/{SOURCE.repo}) at commit"
        f" `{SOURCE.revision}`, licensed **{SOURCE.license}** (as of retrieval;"
        f" [full text]({MOSS_LICENSE_TEXT_LINK}) — a verbatim copy is in `LICENSE.md`).",
        f"- **Authors**: MOSS-Image is released by OpenMOSS ([technical report]"
        f"({MOSS_TECHNICAL_REPORT})).",
        "- **Text encoder** (`text_encoder`, `tokenizer`): the checkpoint's bundled Qwen3.5 text"
        " encoder, from"
        f" [{TEXT_ENCODER_ORIGIN.repo}](https://huggingface.co/{TEXT_ENCODER_ORIGIN.repo}) at"
        f" commit `{TEXT_ENCODER_ORIGIN.revision}` (the Qwen team, Alibaba Cloud), licensed"
        f" **{TEXT_ENCODER_ORIGIN.license}** (as of retrieval), and its tokenizer. Only the text"
        " tower is redistributed here; its copyright notice is kept in `NOTICE.md`.",
        "- **VAE decoder** (`vae_decoder_pre` / `vae_decoder_attn` / `vae_decoder_post`): the"
        " checkpoint's bundled FLUX.2 VAE with the FLUX.2 small decoder (Black Forest Labs)."
        " Only the decode path is redistributed; the VAE encoder is not included:",
    ]
    lines += [
        f"  - {MOSS_VAE_PARTS[origin.source.repo]}:"
        f" [{origin.source.repo}](https://huggingface.co/{origin.source.repo}) at commit"
        f" `{origin.source.revision}`, licensed **{origin.source.license}** (as of retrieval)"
        for origin in VAE_ORIGINS
    ]
    cards = ", ".join(
        f"[{origin.source.repo}](https://huggingface.co/{origin.source.repo}/blob/"
        f"{origin.source.revision}/README.md)"
        for origin in VAE_ORIGINS
    )
    lines += [
        # Maintainer decision 2026-11-01: passed through as information. Neither card links the
        # policy at the pinned revisions, so the line points to the cards themselves.
        f"  - Black Forest Labs' model cards of both origins ({cards}) also point to the BFL Usage"
        " Policy, without a link of their own. This is informational: it is not a term of the"
        " Apache License, Version 2.0.",
    ]
    lines += [
        "- **Changes made here** (listed in full in `NOTICE.md`, per Apache 2.0 §4(b)): the",
        "  transformer's and the text encoder's weight matrices are stored as int8 (one scale per",
        "  output channel); the VAE keeps the source f32 values. The transformer is split into a",
        "  text-refiner graph and the joint-sequence graph (plus the same graph without block 22,",
        "  which borrows its weights), the VAE decoder into three graphs around its middle",
        "  attention, and the text encoder keeps its text tower only. No retraining and no",
        "  fine-tuning.",
        # Black Forest Labs only: of the upstreams, only its cards ask for this ("Trademarks & IP":
        # the BFL and FLUX marks in modified versions "must not ... imply sponsorship or
        # endorsement"). Maintainer decision 2026-11-01.
        "- This is not an official release of Black Forest Labs, and it is not endorsed by them;",
        "  the Black Forest Labs and FLUX names are used only to describe where the VAE weights",
        "  come from.",
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
    return render(
        (
            frontmatter(_metadata()),
            ["", f"# {MOSS_TITLE}", ""],
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
