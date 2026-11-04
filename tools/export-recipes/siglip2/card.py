"""Model card (`README.md`) of the SigLIP2 distribution — a pure function derived mechanically from
the manifest.

The general-purpose rendering parts (frontmatter, model list, quant table, section assembly) live in
`karume.modelcard`. This module holds only **facts specific to SigLIP2**: attribution (source,
license, citation) and what this pipeline's card says.

Attribution is **determined uniquely by the model name** ({@link SIGLIP2_UPSTREAM}), so it is not
split into a profile axis — splitting it would let the operator make the mix-up of "shipping
so400m under base's attribution".

MUST: **every number, download size, quant table and dtype label is derived from the manifest**
(the same MUST of `karume.modelcard` applies as is). The only constants this module may hold are
**facts that do not exist** in the manifest.
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import partial
from typing import Any

from _shared.upstream import hf_upstreams
from karume.modelcard import (
    CardMetadata,
    FileRefKey,
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

from .measurements import map_head_diff_text, map_head_norm_text

#: The pipeline contract this template can describe (ADR 0041 §2 — per model).
SIGLIP2_SUPPORTED_PIPELINE = "siglip2/1"

#: HF pipeline tag. This distribution carries **only the vision tower**, so it cannot claim the
#: upstream card's `zero-shot-image-classification` (that needs the text tower).
SIGLIP2_PIPELINE_TAG = "image-feature-extraction"

SIGLIP2_TITLE = "SigLIP2 Vision Tower — Karume"

#: Model name → HF repository ID of the upstream checkpoint. **This one table is the only fact of
#: "which model is which weights"**; `siglip2.distribution` derives the series name and the input
#: directory name from it (both are the upstream repository name itself — the spelling of
#: `siglip2/export.py`). A model name not listed has no attribution to write, so the card is not
#: rendered and fails.
#:
#: License checked at the source (2026-08-13 — https://huggingface.co/google/siglip2-base-patch16-224):
#: `license: apache-2.0`. The same for so400m.
SIGLIP2_UPSTREAM: Mapping[str, str] = {
    "base": "google/siglip2-base-patch16-224",
    "so400m": "google/siglip2-so400m-patch14-384",
}

SIGLIP2_LICENSE = "apache-2.0"

#: Where the full text lives (the same text as `LICENSE.md` at the distribution repository's root —
#: Apache 2.0 §4(a)).
SIGLIP2_LICENSE_TEXT_LINK = "https://www.apache.org/licenses/LICENSE-2.0"

#: The measured range of the difference that replacing the MAP head (step ③ of `siglip2.patch`)
#: introduces, as body text. It is **not bit-identical**, so the attribution section states it (a
#: value `siglip2/export.py --verify` measures on both series every run). This module holds neither
#: the number nor the format — the source of truth is
#: {@link siglip2.measurements.map_head_diff_text}.
SIGLIP2_MAP_HEAD_DIFF = map_head_diff_text()

#: The scale that gives the difference above its meaning (the L2 norm of the pooled vector). The
#: size of the difference can only be read as a ratio to this number, so the notice and the card
#: must state **the same single word**. The source of truth is
#: {@link siglip2.measurements.map_head_norm_text}.
SIGLIP2_MAP_HEAD_NORM = map_head_norm_text()


def _siglip2_metadata(manifest: Mapping[str, Any]) -> CardMetadata:
    """Build the frontmatter from the models listed in the manifest (`base_model` is every upstream
    that is redistributed).

    MUST: do not render for an unknown model name — silently dropping a model missing from the
    attribution table quietly produces a card with one `base_model` short (= a redistribution that
    does not name its source).
    """
    upstream: list[str] = []
    for name in manifest["models"]:
        if name not in SIGLIP2_UPSTREAM:
            raise ValueError(
                f"the upstream of model '{name}' is not in the attribution table (known:"
                f" {sorted(SIGLIP2_UPSTREAM)}) — a card that cannot name its source is not rendered"
            )
        upstream.append(SIGLIP2_UPSTREAM[name])
    return CardMetadata(
        pipeline_tag=SIGLIP2_PIPELINE_TAG,
        base_model=tuple(upstream),
        # No `base_model_relation` — this only moves the weights into another container without
        # changing their storage (still f32), so it is none of adapter / merge / quantized /
        # finetune (the CardMetadata doc).
        license=SIGLIP2_LICENSE,
        tags=(SIGLIP2_PIPELINE_TAG, "webgpu", "siglip", "vision"),
    )


def _siglip2_overview(manifest: Mapping[str, Any]) -> list[str]:
    config = default_model(manifest)["pipelineConfig"]
    return [
        "## What is this",
        "",
        "An image embedding distribution: the **vision tower** of SigLIP2, converted into the",
        "WebGPU inference runtime **Karume**'s container format (a `.krm` part sequence whose",
        "first part carries the graph and model descriptors). Runs as-is in the browser and in",
        "Deno.",
        "",
        "- One graph, one call: pixels in, `pooler_output` out — the pooled `[1,"
        f" {config['hiddenDim']}]` vector from the attention-pooling (MAP) head.",
        "- **Preprocessing is included.** The pipeline resizes to"
        f" {config['imageWidth']} × {config['imageHeight']}, rescales and normalizes with the",
        "  constants below, so callers hand over raw RGB8 pixels. Decoding PNG / JPEG is *not*",
        "  part of this — use `createImageBitmap` in the browser, or any decoder in Deno.",
        "- **The text tower is not here.** Zero-shot classification and image/text similarity need",
        "  it (plus the logit scale and bias that go with it), so this repository cannot do them;",
        "  what it does is turn an image into a vector.",
        "- Not readable by transformers (it's a different container with an embedded graph); the"
        f" reader is a pipeline that implements `{SIGLIP2_SUPPORTED_PIPELINE}`.",
        f"- Exporter used for the conversion: `{manifest['generator']}`. The distribution manifest"
        f" is `karume.json` (`{manifest['format']}`).",
    ]


def _siglip2_base_weights(manifest: Mapping[str, Any]) -> list[str]:
    """The attribution section. The storage is unchanged, so it claims only "converted"."""
    lines = [
        "## Base weights and attribution",
        "",
        "Converted into the container format — the original checkpoints are not distributed here.",
        "",
    ]
    for name in manifest["models"]:
        repo = SIGLIP2_UPSTREAM[name]
        lines.append(
            f"- **`{name}`**: [{repo}](https://huggingface.co/{repo}), licensed"
            f" **{SIGLIP2_LICENSE}** (as of retrieval;"
            f" [full text]({SIGLIP2_LICENSE_TEXT_LINK}) — a verbatim copy is in `LICENSE.md`)."
        )
    lines += [
        "- **Changes made here** (also listed in `NOTICE.md`, per Apache 2.0 §4(b)): conversion"
        " into the Karume container format, vision tower only.",
        "  No retraining, no fine-tuning and **no quantization** — the weights are the source",
        "  checkpoint's own f32 values. Two rewrites were needed to export the graph: the patch",
        "  embedding's padding and the position embedding lookup were folded into equivalent",
        "  operations (bit-exact), and the pooling head's attention was rewritten with explicit",
        "  q/k/v projections, which is equivalent up to floating-point rounding"
        f" ({SIGLIP2_MAP_HEAD_DIFF}",
        f"  measured on the pooled vector, whose L2 norm is {SIGLIP2_MAP_HEAD_NORM}).",
    ]
    return lines


def _siglip2_usage(manifest: Mapping[str, Any], repo: str) -> list[str]:
    """Policy of the Usage example: keep the minimal working form, and **list the optionals an
    ordinary use case is likely to need as commented-out lines** (with the selectable values
    enumerated in a comment on the same line — derived mechanically from the manifest, so the list
    follows when series / quants are added). The reader takes the next step just by uncommenting.

    `embed()` takes a single image only (preprocessing is fixed by the distribution's constants —
    the public surface of `siglip2.Pipeline`), so this pipeline's optional knobs are only model and
    quant.

    NOTE: `fromAssets` is **not advertised** (ruling of 2026-08-29). The split distribution form
    became readable too (X2-101), but that surface is meant for local debugging and assumes the
    caller holds the bytes; the ordinary entry point for a reader using HF is `fromPretrained`.
    Listing both would leave the reader to decide "which one to use".
    """
    model_name = manifest["defaultModel"]
    model = default_model(manifest)
    quant = model["defaultQuant"]
    model_names = " / ".join(sorted(manifest["models"]))
    quant_names = " / ".join(sorted(model["quants"]))
    return [
        "## Usage",
        "",
        "```ts",
        'import { Pipeline } from "jsr:@karume/models/siglip2";',
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
        "// RGB8, row-major, 3 bytes per pixel. Decoding is the caller's job.",
        "const embedding = await pipeline.embed({ data: pixels, width, height });",
        "",
        "// pooler_output is not L2-normalized — normalize it yourself for cosine similarity:",
        "const norm = Math.sqrt(embedding.reduce((sum, value) => sum + value * value, 0));",
        "const unit = embedding.map((value) => value / norm);",
        "```",
        "",
        "`embed()` returns the pooled vector as an f32 array, exactly as the graph produces it.",
        "It keeps one GPU session alive for the lifetime of the pipeline, so embedding many images",
        "uploads the weights once; concurrent calls are queued rather than run side by side.",
        "Weights are fetched once and cached (verified against `karume.json`'s `size` / `sha256`).",
    ]


def _siglip2_shape(model: Mapping[str, Any]) -> list[str]:
    """Preprocessing constants and the input / output shapes (what the user passes in and gets
    back can be read here)."""
    config = model["pipelineConfig"]
    mean = " / ".join(str(value) for value in config["imageMean"])
    std = " / ".join(str(value) for value in config["imageStd"])
    return [
        "### Input and output",
        "",
        "Derived from the checkpoint's own `preprocessor_config.json` and the exported graph, and",
        "checked against each other when this repository was assembled.",
        "",
        f"- **input**: RGB8 pixels, resized to {config['imageWidth']} ×"
        f" {config['imageHeight']} ({config['interpolation']}, antialiased). The aspect ratio is",
        "  **not** preserved — there is no crop and no padding, matching the upstream processor.",
        f"- **normalization**: `(pixel / 255 - mean) / std`, mean {mean}, std {std}",
        f"- **output**: `pooler_output`, {config['hiddenDim']} f32 values, not L2-normalized",
    ]


def render_siglip2_model_card(
    manifest: Mapping[str, Any], repo: str, host_assets: Mapping[FileRefKey, int] = {}
) -> str:
    """Assemble the `README.md` body of the SigLIP2 distribution (pure function, with a trailing
    newline)."""
    require_pipeline(manifest, SIGLIP2_SUPPORTED_PIPELINE)
    metadata = _siglip2_metadata(manifest)
    return render(
        (
            frontmatter(metadata),
            introduction(SIGLIP2_TITLE, hf_upstreams(metadata.base_model)),
            _siglip2_overview(manifest),
            [""],
            _siglip2_base_weights(manifest),
            [""],
            models(manifest),
            [""],
            _siglip2_usage(manifest, repo),
            *model_sections(manifest, (partial(quants, host_assets=host_assets), _siglip2_shape)),
        )
    )
