"""The model card (`README.md`) of the Depth Anything V2 distribution — a pure function derived
mechanically from the manifest.

The general-purpose rendering parts (frontmatter, model list, quant table, section assembly) live
in `karume.modelcard`. This module holds only **facts specific to Depth Anything V2**: the
attribution (source, licence, citation) and what this pipeline's card says.

The attribution is **determined uniquely by the size** ({@link DEPTH_ANYTHING_UPSTREAM}), so it is
not split into a profile axis — splitting it would let an operator make the mistake of "shipping
Base under Small's attribution" (and Base is CC BY-NC 4.0 at that).

MUST: **every number, download size, quant table and dtype label is derived from the manifest**
(the same MUST of `karume.modelcard` applies unchanged). The only constants allowed here are facts
that **do not exist** in the manifest.
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

from .measurements import convt_diff_text

#: The pipeline contract this template can describe (ADR 0041 §2 — per model).
DEPTH_ANYTHING_SUPPORTED_PIPELINE = "depth-anything/1"

#: The HF pipeline tag (same as upstream). The output is one relative-depth map, not metric depth.
DEPTH_ANYTHING_PIPELINE_TAG = "depth-estimation"

#: The upstream paper (Depth Anything V2).
DEPTH_ANYTHING_PAPER = "arxiv.org/abs/2406.09414"

#: Model name → HF repository ID of the upstream checkpoint. **This one table is the only fact
#: about "which model is which weights"**; `depth_anything.distribution` derives the series name
#: and the input directory name from it (both are the upstream repository name itself — the
#: spelling of `depth_anything/export.py`). A model name not listed here has no attribution to
#: write, so the card fails instead of rendering.
#:
#: MUST: **only Small is Apache-2.0** (checked 2026-08-14 against the HF model API:
#: `depth-anything/Depth-Anything-V2-Base-hf` and `-Large-hf` are both `license:
#: cc-by-nc-4.0`). The size axis itself is the distribution's model axis (models side by side in
#: one per-generation repository — `depth_anything.distribution`), but **only what this table
#: lists can be shipped**. The check has two stages, the name and the provenance the series'
#: container states (`provenance.license` must equal this table's licence), and that is where an
#: accident of redistributing NC weights under an Apache-2.0 card stops. Adding Base / Large goes
#: together with splitting the licence field per model ({@link DEPTH_ANYTHING_LICENSE} holds a
#: single value today).
DEPTH_ANYTHING_UPSTREAM: Mapping[str, str] = {
    "small": "depth-anything/Depth-Anything-V2-Small-hf",
}

#: Checked 2026-08-14 against the HF model API: `license: apache-2.0`. The frontmatter is written
#: on the premise that **every model in the table has this one value** (the MUST above).
DEPTH_ANYTHING_LICENSE = "apache-2.0"

#: Where the original text lives (the same text as `LICENSE.md` at the distribution repository's
#: root — Apache 2.0 §4(a)).
DEPTH_ANYTHING_LICENSE_TEXT_LINK = "https://www.apache.org/licenses/LICENSE-2.0"

#: Measured width of the difference introduced by replacing `ConvTranspose2d` with a 1×1 conv +
#: pixel shuffle (③ of `depth_anything.patch`). It is **not bit-identical**, so the attribution
#: section states it (a value `depth_anything/export.py --verify` measures every time; the RMS of
#: the depth is about 1.0). This holds neither the number nor its format — the source of truth is
#: {@link depth_anything.measurements.convt_diff_text}.
DEPTH_ANYTHING_CONVT_DIFF = convt_diff_text()


def _depth_anything_repo(name: str) -> str:
    """Model name → upstream repository ID (no rendering when it is not in the attribution
    table)."""
    repo = DEPTH_ANYTHING_UPSTREAM.get(name)
    if repo is None:
        raise ValueError(
            f"the upstream of model '{name}' is not in the attribution table (known:"
            f" {sorted(DEPTH_ANYTHING_UPSTREAM)}) — a card that cannot state its source is not"
            " rendered"
        )
    return repo


def _depth_anything_title(manifest: Mapping[str, Any]) -> str:
    """The title (`Depth Anything V2 Small — Karume`).

    The size is spelled **from the upstream repository name** (the default model's upstream
    repository name becomes the title as is) — with a second table, a card that "sells Base
    weights as Small" could be produced silently when only one of them moves.

    NOTE: the repository is per generation and the size is a model axis inside it
    (`depth_anything.distribution`), so a title that names only the default size holds only while
    Small is the one size that can be shipped. When a second size is added (together with the
    per-model licence split of the {@link DEPTH_ANYTHING_UPSTREAM} MUST), the title changes to the
    generation name — changing it now would only reissue distribution cards with the same content.
    """
    checkpoint = _depth_anything_repo(manifest["defaultModel"]).split("/", 1)[1]
    return f"{checkpoint.removesuffix('-hf').replace('-', ' ')} — Karume"


def _depth_anything_metadata(manifest: Mapping[str, Any]) -> CardMetadata:
    """Build the frontmatter from the models listed in the manifest (`base_model` is every
    upstream that is redistributed).

    MUST: do not render for an unknown model name — silently dropping a model missing from the
    attribution table would quietly produce a card with one `base_model` short (= a redistribution
    that does not state its source).
    """
    return CardMetadata(
        pipeline_tag=DEPTH_ANYTHING_PIPELINE_TAG,
        base_model=tuple(_depth_anything_repo(name) for name in manifest["models"]),
        # No `base_model_relation` — only the container changed while the storage stayed the
        # same (still f32), so it is none of adapter / merge / quantized / finetune (the doc of
        # CardMetadata).
        license=DEPTH_ANYTHING_LICENSE,
        tags=(DEPTH_ANYTHING_PIPELINE_TAG, "depth", "monocular-depth-estimation", "webgpu"),
    )


def _depth_anything_overview(manifest: Mapping[str, Any]) -> list[str]:
    config = default_model(manifest)["pipelineConfig"]
    return [
        "## What is this",
        "",
        "A monocular **relative depth** distribution, converted into the WebGPU inference runtime",
        "**Karume**'s container format (a `.krm` part sequence whose first part carries the graph",
        "and model descriptors). Runs as-is in the browser and in Deno.",
        "",
        "- One graph, one call: pixels in, one **f32 depth value per pixel** out, at the same size",
        "  as the image you handed over.",
        "- **Relative depth has no unit and no origin** — larger means nearer, and only the",
        "  ordering carries meaning. This is not a metric-depth checkpoint: the numbers are not",
        "  metres and are not comparable across images.",
        "- **Pre- and post-processing are included.** The pipeline resizes to"
        f" {config['imageWidth']} × {config['imageHeight']}, normalizes with the constants below,",
        "  runs the graph, and scales the depth map back to the original resolution. Decoding",
        "  PNG / JPEG is *not* part of this — use `createImageBitmap` in the browser, or any",
        "  decoder in Deno.",
        "- **Normalization and colouring are yours.** The pipeline returns the raw f32 map, not a",
        "  `[0, 1]` image: folding it with min/max would throw the scale away, and which colour",
        "  map to use is a decision this repository should not make for you.",
        "- Not readable by transformers (it's a different container with an embedded graph); the"
        f" reader is a pipeline that implements `{DEPTH_ANYTHING_SUPPORTED_PIPELINE}`.",
        f"- Exporter used for the conversion: `{manifest['generator']}`. The distribution manifest"
        f" is `karume.json` (`{manifest['format']}`).",
    ]


def _depth_anything_base_weights(manifest: Mapping[str, Any]) -> list[str]:
    """The attribution section. The storage is unchanged, so it claims only "converted"."""
    lines = [
        "## Base weights and attribution",
        "",
        "Converted into the container format — the original checkpoints are not distributed here.",
        "",
    ]
    for name in manifest["models"]:
        repo = _depth_anything_repo(name)
        lines.append(
            f"- **`{name}`**: [{repo}](https://huggingface.co/{repo}), licensed"
            f" **{DEPTH_ANYTHING_LICENSE}** (as of retrieval;"
            f" [full text]({DEPTH_ANYTHING_LICENSE_TEXT_LINK}) — a verbatim copy is in"
            " `LICENSE.md`)."
        )
    lines += [
        "- **Only the Small checkpoint is Apache-2.0.** Upstream ships Base and Large under"
        " **CC BY-NC 4.0**, so they are not converted or redistributed here.",
        "- **Training data**: see the upstream model card. Depth Anything V2 is distilled from a",
        "  teacher trained on synthetic data and then trained on pseudo-labelled real images —",
        "  check the upstream sources against your own use case.",
        f"- **Architecture**: DINOv2 backbone + DPT head"
        f" ([{DEPTH_ANYTHING_PAPER}](https://{DEPTH_ANYTHING_PAPER})).",
        "- **Changes made here** (also listed in `NOTICE.md`, per Apache 2.0 §4(b)):",
        "  conversion into the Karume container format. No retraining, no fine-tuning and",
        "  **no quantization** — the weights are the source checkpoint's own f32 values. The",
        "  graph is the upstream `forward` with two layout-only rewrites (the last fusion",
        "  stage's upsample takes an explicit output size instead of a scale factor, and the",
        "  position-embedding interpolation is pinned to the pretraining resolution where it",
        "  is the identity — both bit-exact), plus one module rewrite that is equivalent up",
        "  to floating-point rounding: the DPT reassemble stage's transposed convolutions",
        "  became a 1×1 convolution followed by a pixel shuffle (they have",
        "  `kernel == stride`, so the two are exactly the same sum in a different order —",
        f"  measured max {DEPTH_ANYTHING_CONVT_DIFF} on depth values whose RMS is around 1).",
    ]
    return lines


def _depth_anything_usage(manifest: Mapping[str, Any], repo: str) -> list[str]:
    """The policy of the usage example: keep the minimal working form, and **list the options a
    typical use case is likely to need as commented-out lines** (with the selectable values
    enumerated in the comment on the same line — derived mechanically from the manifest, so the
    enumeration follows when sizes / quants are added). Readers take the next step just by
    uncommenting.

    `estimate()` accepts a single image only (the resolution is the square pretraining resolution
    baked into the graph and cannot be chosen at run time — the public surface of
    `depthAnything.Pipeline`), so this pipeline's optional knobs are just two, model and quant.

    NOTE: `fromAssets` is **not advertised** (ruling of 2026-08-29). The split distribution can
    now be read as well (X2-101), but that is a surface for local debugging that assumes you hold
    the bytes yourself; the ordinary entry point for readers using HF is `fromPretrained`. Showing
    both would leave readers to decide which one to use.
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
        'import { Pipeline } from "jsr:@karume/models/depth-anything";',
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
        "const depth = await pipeline.estimate({ data: pixels, width, height });",
        "",
        "// depth.data is one f32 per pixel, same width/height as the input, larger = nearer.",
        "// Fold it to [0, 1] yourself when you want to look at it:",
        "let min = Infinity;",
        "let max = -Infinity;",
        "for (const value of depth.data) {",
        "  if (value < min) min = value;",
        "  if (value > max) max = value;",
        "}",
        "const span = max - min;",
        "const gray = depth.data.map((value) => (span > 0 ? (value - min) / span : 0));",
        "```",
        "",
        "`estimate()` keeps one GPU session alive for the lifetime of the pipeline, so processing",
        "many images uploads the weights once; concurrent calls are queued rather than run side by",
        "side. Weights are fetched once and cached (verified against `karume.json`'s `size` /",
        "`sha256`).",
    ]


def _depth_anything_shape(model: Mapping[str, Any]) -> list[str]:
    """The preprocessing constants and the input / output shapes (what users pass in and get
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
        f" {config['imageHeight']} ({config['interpolation']}, antialiased). Note the filter:"
        " this checkpoint asks for **bicubic**, unlike most image towers.",
        "- The aspect ratio is **not** preserved. The graph is baked at the single square",
        "  pretraining resolution (the position embeddings are tied to the patch grid), so a",
        "  non-square photo is stretched rather than letterboxed. Crop it yourself first if that",
        "  matters for your images.",
        f"- **normalization**: `(pixel / 255 - mean) / std`, mean {mean}, std {std}",
        "- **output**: one f32 per pixel at the size of the image you passed in — relative depth,",
        "  non-negative (the head ends in a ReLU, so far regions sit at exactly 0), larger =",
        "  nearer. The graph itself emits the map at the resized resolution; scaling it back to",
        "  your image happens on the host, bilinearly.",
    ]


def render_depth_anything_model_card(
    manifest: Mapping[str, Any], repo: str, host_assets: Mapping[FileRefKey, int] = {}
) -> str:
    """Assemble the `README.md` body of the Depth Anything V2 distribution (pure function, with a
    trailing newline)."""
    require_pipeline(manifest, DEPTH_ANYTHING_SUPPORTED_PIPELINE)
    metadata = _depth_anything_metadata(manifest)
    return render(
        (
            frontmatter(metadata),
            introduction(_depth_anything_title(manifest), hf_upstreams(metadata.base_model)),
            _depth_anything_overview(manifest),
            [""],
            _depth_anything_base_weights(manifest),
            [""],
            models(manifest),
            [""],
            _depth_anything_usage(manifest, repo),
            *model_sections(
                manifest, (partial(quants, host_assets=host_assets), _depth_anything_shape)
            ),
        )
    )
