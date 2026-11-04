"""Model card (`README.md`) of the BiRefNet family distributions — a pure function derived
mechanically from the manifest.

The general-purpose rendering parts (frontmatter, model list, quant table, section assembly) live in
`karume.modelcard`. What lives here is **only the facts specific to the BiRefNet family**: the
attribution (origin, license, citation, training data) and what this pipeline's card says.

The attribution is **determined uniquely by the checkpoint** ({@link BIREFNET_CHECKPOINTS}), so it
is not split into a profile axis — splitting it would let an operator make the mix-up "ship Lucida
with BiRefNet_HR's attribution". The checkpoint is the very axis of the distribution repository
(1 repository = 1 checkpoint), so it is not read from the manifest: **the caller's pipeline entry
binds it and passes it in** (`birefnet.distribution._birefnet_pipeline`). The model names listed in
the manifest are **resolutions** (`"1024"` / `"2048"`), not the attribution axis.

MUST: **every number, download size, quant table and dtype label is derived from the manifest**
(the same MUST of `karume.modelcard` applies as is). The only constants allowed here are facts that
**do not exist in the manifest**.
"""

from __future__ import annotations

import textwrap
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
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

#: The pipeline contract this template can describe (ADR 0041 §2 — per model).
BIREFNET_SUPPORTED_PIPELINE = "birefnet/2"

#: HF pipeline tag (the same for both upstream repositories). The output is a single alpha matte,
#: with neither class labels nor instance segmentation.
BIREFNET_PIPELINE_TAG = "image-segmentation"


@dataclass(frozen=True)
class BirefnetResources:
    """Measured execution resources of one model (= one resolution).

    Not an exception to the MUST at the top: `requiredLimits` counts only the resident part
    (weights, state), so the binding size the intermediate tensors need and the total allocation
    are **facts that do not exist in the manifest** (the semantics of ADR 0089 decision 3). They are
    the constraint a reader wants to check first, so the card carries them.
    """

    #: Total GPU allocation = the runtime's estimate (`peakAccountedBytes` of
    #: `PreparedModel.estimate`): resident weights + intermediate region + input + output readback.
    total: str
    #: The region the intermediate tensors need. **It does not include the resident weights** — that
    #: number lives on the manifest's container side (the part list; the quant table derives it),
    #: and
    #: copying it here would give the same quantity two sources of truth.
    intermediates: str
    #: The phrase that names one binding above WebGPU's default `maxStorageBufferBindingSize`
    #: (128MiB), **together with what it was measured to be** (in a form the reader can read as
    #: "does
    #: not run on default limits").
    binding: str
    #: Run time per image.
    run: str


@dataclass(frozen=True)
class BirefnetResourceMeasurement:
    """Measured execution resources of one checkpoint — the measured quant preset, the conditions
    and the numbers per resolution.

    It is kept per checkpoint because the measured preset and conditions differ per checkpoint (HR
    on the f16 preset, Lucida on the f32 preset — copying one table into both cards would make one
    of them claim numbers for a preset it did not measure). The values are measured after ADR 0093's
    static liveness packing and the recipe's patch ⑨ (swapping the order of the decoder tail's 1×1
    conv and the bilinear upsample); the source of truth is the BiRefNet section of
    docs/limitations.md.
    """

    #: The quant preset that was measured. MUST: the card states the total allocation **as this
    #: preset's number** — it does not extrapolate the total of other presets (the MUST at the top).
    #: The card of a model whose manifest lacks this preset is not rendered.
    quant: str
    #: Measurement conditions (date, machine — this is the nature of the measurement itself, so it
    #: is
    #: not derived from `pipelineConfig`).
    conditions: str
    #: Model name (= resolution) → numbers. MUST: no entry for an unmeasured resolution — every
    #: number
    #: the card states is tied to a measurement. The lookup uses the manifest's `imageWidth`
    #: ({@link _birefnet_resources}), so rendering at an unmeasured resolution fails.
    by_resolution: Mapping[str, BirefnetResources]


#: Post-processing procedure (the manifest's `pipelineConfig.postprocess` — spelled as the TS side's
#: `MATTE_PROCEDURES`) → the card's description (interpolation, order of the steps, quantization).
#: The procedure's name is in the manifest, but the description of what it does is a fact that
#: **does not exist in the manifest**, so it lives here (not an exception to the MUST at the top).
#:
#: MUST: do not describe an unknown procedure — publishing an extrapolated description would ship a
#: card that disagrees with the distribution's actual post-processing. The procedures' source of
#: truth is ADR 0125 and `packages/models/src/birefnet/matte.ts`.
BIREFNET_POSTPROCESS_DESCRIPTIONS: Mapping[str, tuple[str, ...]] = {
    "u8-pil-bicubic": (
        "- **post-processing** (`u8-pil-bicubic`, the order upstream uses for this checkpoint):",
        "  the sigmoid of the logits is rounded to 8 bits at the graph's resolution, then resized",
        "  back to the image's size with 8-bit bicubic resampling that matches Pillow's `BICUBIC`",
        "  bit for bit. Rounding is half-up (the upstream code truncates).",
    ),
    "float-bilinear": (
        "- **post-processing** (`float-bilinear`, the order upstream uses for this checkpoint):",
        "  the sigmoid of the logits is resized back to the image's size in floating point with",
        "  antialiased bilinear interpolation, then rounded to 8 bits. Rounding is half-up (the",
        "  upstream code truncates).",
    ),
}

#: The upstream paper (BiRefNet — the reference shared by the whole family).
BIREFNET_PAPER = "arxiv.org/abs/2401.03407"


@dataclass(frozen=True)
class BirefnetCheckpoint:
    """The **facts not in the manifest** for one checkpoint (origin, display name, extra
    attribution lines).

    **This one table is the only fact about "which repository holds which weights"**:
    `birefnet.distribution` takes the series name (= the spelling derived from the upstream
    repository name — the convention of `birefnet/export.py`) and the attribution from here. A
    checkpoint that is not listed cannot state its origin, so the card fails instead of rendering.
    """

    #: HF repository ID of the upstream checkpoint.
    repo: str
    #: Upstream revision of the shipped weights (the HF commit SHA). The assembly
    #: (`birefnet.distribution`) checks the origin the series' container states
    #: (`provenance.upstreamRevision` — the script bakes it from the download record of
    #: `--model-dir`) against this value. It is compared with a fixed value, so a series baked from
    #: another checkpoint or another revision cannot be assembled even if its name matches. The
    #: value was confirmed from the `hf download` record of `inputs/birefnet/<name>`
    #: (`_shared.upstream.snapshot_revision`) (the cross-check lives in
    #: `birefnet/tests/test_export.py`, on machines that have the download record).
    revision: str
    #: The card's heading (a model upstream markets under a name is called by that name).
    title: str
    #: First line of the overview (what the weights can do).
    tagline: str
    #: Extra lines of the attribution section (training-data licenses and the like — facts that
    #: differ per upstream).
    attribution: tuple[str, ...]
    #: Floating-point storage dtype of the upstream checkpoint (measured from the safetensors header
    #: 2026-09-25 — HR has 687 F16 tensors, Lucida 687 F32). It determines the quant presets that
    #: are shipped (no preset wider than this storage; the default is the preset of the same type —
    #: `birefnet.distribution.birefnet_seat_dtypes`), which presets hold the values unchanged
    #: ({@link birefnet_storage_lines}), and `base_model_relation` ({@link _birefnet_relation}). The
    #: quality check at export time reads the same fact directly from the header
    #: (`birefnet.export.checkpoint_dtypes`), so the cross-check of the two lives in
    #: `birefnet/tests/test_export.py`, on machines that have the real weights.
    stored_dtype: str
    #: The measured execution resources the card states ({@link BirefnetResourceMeasurement}).
    resources: BirefnetResourceMeasurement


#: Checked against the source (2026-08-13 — HF model API): both are `license: mit`.
BIREFNET_LICENSE = "mit"

BIREFNET_CHECKPOINTS: Mapping[str, BirefnetCheckpoint] = {
    "hr": BirefnetCheckpoint(
        repo="ZhengPeng7/BiRefNet_HR",
        revision="a7a562f6fd16021180f2f4348f4de003a2d3d1e1",
        title="BiRefNet HR — Karume",
        tagline="the high-resolution general-purpose BiRefNet checkpoint",
        attribution=(
            "- **Upstream training data**: see the upstream model card and the BiRefNet"
            " repository. Several of the datasets it draws on (DIS5K, COD10K, HRSOD and"
            " others) are distributed for research purposes — check them against your own use"
            " case.",
        ),
        stored_dtype="f16",
        # The total = the runtime estimate's `peakAccountedBytes` (weights + intermediate region +
        # input + output readback — 1024²: 431.2 + 688.6 + 12 + 4 = 1,135.8 MiB; 2048²: 431.2 +
        # 2,725.3 + 48 + 16 = 3,220.5 MiB), converted at 1 GiB = 1,024 MiB and rounded. The memory
        # numbers were re-derived 2026-10-26 after the transient placement changed from first-fit
        # to least region growth (perf-ledger H-42 — the same estimate still gives the old
        # 750.7 / 2,948.0 MiB under first-fit); the run times are the 2026-10-16 measurement.
        # `run` is the range of the 5 runs after the warm-up run (the one that misses the plan); the
        # first of the 5 includes allocating the intermediate region.
        # The intermediate region and the largest binding are the same as for the f32 preset
        # (activations are f32 on the f16 preset too). The mask of Swin's shifted-window attention
        # is a run-time intermediate, not a resident weight (built inside the graph from a table of
        # the 4 window kinds and per-window indices — `birefnet.patch`), so the weights are almost
        # the same at every resolution, and the 1024² intermediate region grew by 1.5 MiB for it
        # (at 2048² it fits inside the liveness peak and does not change).
        resources=BirefnetResourceMeasurement(
            quant="f16",
            conditions=(
                "memory from the runtime's estimate, 2026-10-26; run time measured 2026-10-16"
                " on an RTX 3080 Ti"
            ),
            by_resolution={
                "1024": BirefnetResources(
                    total="about 1.1 GiB",
                    intermediates="689 MiB",
                    binding="the largest single storage buffer is 320 MiB",
                    run="about 1.4 s",
                ),
                "2048": BirefnetResources(
                    total="about 3.1 GiB",
                    intermediates="2,725 MiB",
                    binding="the largest single storage buffer is 1,280 MiB",
                    run="7.0 – 7.2 s",
                ),
            },
        ),
    ),
    "lucida": BirefnetCheckpoint(
        repo="egeorcun/lucida",
        revision="6cbedc9722652dc9a3df91dd871f0c4f3334e922",
        title="Lucida (BiRefNet) — Karume",
        tagline=(
            "a BiRefNet fine-tune aimed at camouflage, transparency, text/logos and illustration"
        ),
        attribution=(
            "- **Fine-tune of**:"
            " [ZhengPeng7/BiRefNet_HR](https://huggingface.co/ZhengPeng7/BiRefNet_HR)"
            " (**mit**) — architecture and initial weights; the upstream copyright notice is"
            " preserved.",
            "- **Illustration training data** includes"
            " [ToonOut](https://huggingface.co/datasets/joelseytre/toonout) (**CC-BY-4.0**);"
            " other sets it draws on (P3M-10k, COD10K, DIS5K) are distributed for research"
            " purposes — check them against your own use case.",
            "- **Not included**: `lucida-m35-comfy.safetensors`. That experimental variant has"
            " the `Normalize` step folded into its first convolution (it expects raw `[0, 1]`"
            " pixels), so the preprocessing this pipeline applies would be counted twice, and"
            " upstream ships it to run inside a larger ComfyUI pipeline rather than bare.",
        ),
        stored_dtype="f32",
        # The total = the runtime estimate's `peakAccountedBytes` (weights + intermediate region +
        # input + output readback — 1024²: 842.5 + 688.6 + 12 + 4 = 1,547.1 MiB; 2048²: 842.5 +
        # 2,725.3 + 48 + 16 = 3,631.8 MiB), converted at 1 GiB = 1,024 MiB and rounded. The memory
        # numbers were re-derived 2026-10-26 after the transient placement changed from first-fit
        # to least region growth (perf-ledger H-42 — the same estimate still gives the old
        # 750.7 / 2,948.0 MiB under first-fit); the run times are the 2026-10-16 measurement.
        # `run` is the range of the 5 runs after the warm-up run (the one that misses the plan); the
        # first of the 5 includes allocating the intermediate region.
        # The effect of the mask becoming a run-time intermediate is the same as for HR. `run` was
        # measured with only Lucida's 2 series run back to back after letting the GPU cool down
        # (measured straight after HR's 2 series, 1024² slowed down run by run to 1.69–2.01 s —
        # raw data in outputs/misc/birefnet-resources-2026-10-16/lucida-rerun/).
        resources=BirefnetResourceMeasurement(
            quant="f32",
            conditions=(
                "memory from the runtime's estimate, 2026-10-26; run time measured 2026-10-16"
                " on an RTX 3080 Ti"
            ),
            by_resolution={
                "1024": BirefnetResources(
                    total="about 1.5 GiB",
                    intermediates="689 MiB",
                    binding="the largest single storage buffer is 320 MiB",
                    run="about 1.4 s",
                ),
                "2048": BirefnetResources(
                    total="about 3.5 GiB",
                    intermediates="2,725 MiB",
                    binding="the largest single storage buffer is 1,280 MiB",
                    run="6.7 – 6.9 s",
                ),
            },
        ),
    ),
}

#: checkpoint → upstream repository ID. **Derived from {@link BIREFNET_CHECKPOINTS}**, not a second
#: table (if only one of two tables moved, "attribute one repository's weights to another upstream"
#: could be built silently).
BIREFNET_UPSTREAM: Mapping[str, str] = {
    name: entry.repo for name, entry in BIREFNET_CHECKPOINTS.items()
}

#: The manifest's weights key (= `birefnet.distribution.BIREFNET_ROLE`). distribution imports this
#: module, so it is spelled here as well to avoid a cycle — `birefnet/tests/test_card.py` checks
#: that the two agree.
BIREFNET_MATTE_WEIGHTS = "matte"

#: Storage dtype → bit width. It determines the rule for the shipped presets (no preset wider than
#: the checkpoint's storage — `birefnet.distribution.birefnet_seat_dtypes`) and whether a preset
#: holds the checkpoint's values unchanged (a narrower preset rounds).
#: MUST: do not describe an unknown dtype (phase 2's i8 is added here together with its preset —
#: no extrapolated wording goes into legal text).
_STORAGE_BITS: Mapping[str, int] = {"f32": 32, "f16": 16}


def birefnet_storage_bits(dtype: str) -> int:
    """Bit width of a storage dtype ({@link _STORAGE_BITS} — an unknown dtype fails)."""
    bits = _STORAGE_BITS.get(dtype)
    if bits is None:
        raise ValueError(
            f"no wording for storage dtype '{dtype}' (known: {' / '.join(_STORAGE_BITS)})"
            " — extrapolated wording is not put into NOTICE / the card"
        )
    return bits


#: Width the storage description wraps at (about 100 characters, like the other lines of the NOTICE
#: / card templates).
_STORAGE_LINE_WIDTH = 100


def _wrapped(text: str, indent: str) -> str:
    """Wrap one Markdown list item (continuation lines are indented to the item's text)."""
    return textwrap.fill(
        text,
        width=_STORAGE_LINE_WIDTH,
        initial_indent=indent,
        subsequent_indent=" " * len(indent),
        break_long_words=False,
        break_on_hyphens=False,
    )


#: The weights stored as f16 (the side emit's eligibility check compresses as weight slots — the
#: relative-position table is an embedding weight slot, and deform's offset / modulator convs are
#: ordinary convs too).
_F16_STORED = "the linear, convolution and embedding (relative-position) weights"

#: What stays in f32 storage: deform's sampling weights (`regular_conv` — weights passed to
#: `deform_conv2d`, with no weight slot) and the biases / norms. ⑦'s BatchNorm is a derived constant
#: whose α / β are computed in f32 from the statistics, so it is described as "derived values", not
#: "widened values" (2026-09-26 review A-3 / F4).
_F32_KEPT = (
    "the deformable convolutions' sampling weights (`regular_conv`), the biases and the"
    " remaining normalization parameters"
)
_BATCHNORM_AFFINE = "inference-time BatchNorm is a derived per-channel affine in f32"


def _seat_storage_line(dtype: str, stored: str) -> str:
    """Storage description of one quant preset (the wording depends on its relation to the
    checkpoint's storage `stored`).

    There is no wording for a preset wider than the checkpoint's storage (an f32 preset of an f16
    checkpoint) — a preset that only widens the values exactly is not shipped (the rule of
    `birefnet.distribution.birefnet_seat_dtypes`), so a manifest with such a preset fails instead of
    rendering.
    """
    if dtype == "f32" and stored == "f32":
        return _wrapped("`f32`: the checkpoint's own f32 values, unchanged.", "  - ")
    if dtype == "f16" and stored == "f16":
        return _wrapped(
            f"`f16`: the checkpoint's own f16 values, kept as f16 in {_F16_STORED}; {_F32_KEPT}"
            f" are widened exactly to f32, and {_BATCHNORM_AFFINE}.",
            "  - ",
        )
    if dtype == "f16" and birefnet_storage_bits(stored) > birefnet_storage_bits(dtype):
        return _wrapped(
            "`f16`: **quantized** — every weight was rounded to the nearest f16 value before"
            f" export; {_F16_STORED} are stored as f16, {_F32_KEPT} keep f32 storage,"
            f" {_BATCHNORM_AFFINE}, and all computation stays in f32.",
            "  - ",
        )
    raise ValueError(f"no wording for an {dtype} storage preset of an {stored} checkpoint")


def birefnet_storage_lines(checkpoint: str, dtypes: Sequence[str]) -> list[str]:
    """Storage description per quant preset (NOTICE and the card's attribution section take it from
    **the same single place**).

    `dtypes` are the storage dtypes of the distribution's presets (the card passes them from the
    manifest, NOTICE from the distribution recipe's table). The heading depends on "do all presets
    hold the checkpoint's values unchanged" — HR's checkpoint is itself f16, so its f16 preset is
    lossless and stays **No quantization**; Lucida's is f32, so its f16 preset is a quantization
    ({@link BirefnetCheckpoint.stored_dtype}).

    MUST: the wording matches the distribution's contents (the MUST of NOTICE — it stays plausible
    prose and passes both `verify_dist` and the manifest checks, so it is built from the facts).
    """
    stored = _birefnet_entry(checkpoint).stored_dtype
    if not dtypes:
        raise ValueError("no quant presets at all — cannot build the storage description")
    lossless = all(
        birefnet_storage_bits(dtype) >= birefnet_storage_bits(stored) for dtype in dtypes
    )
    # With one preset (HR), write it in the singular — "every quant seat" confuses the reader in a
    # repository with no preset to compare against.
    seats = "every quant seat stores" if len(dtypes) > 1 else "the quant seat stores"
    heading = (
        f"**No quantization**: {seats} the source checkpoint's own values"
        f" (the checkpoint itself is stored in {stored}):"
        if lossless
        else f"**Quantization**: the source checkpoint is stored in {stored}; per quant seat:"
    )
    return [_wrapped(heading, "- "), *(_seat_storage_line(dtype, stored) for dtype in dtypes)]


def _birefnet_entry(checkpoint: str) -> BirefnetCheckpoint:
    """Look up the attribution of the checkpoint this repository ships (1 repository, 1 checkpoint).

    MUST: do not render for an unknown checkpoint — silently dropping weights missing from the
    attribution table would quietly ship a redistribution that does not state its origin.
    """
    entry = BIREFNET_CHECKPOINTS.get(checkpoint)
    if entry is None:
        raise ValueError(
            f"the upstream of checkpoint '{checkpoint}' is not in the attribution table"
            f" (known: {sorted(BIREFNET_CHECKPOINTS)}) — a card that cannot state its origin is not"
            " rendered"
        )
    return entry


def _birefnet_resources(
    measurement: BirefnetResourceMeasurement, config: Mapping[str, Any]
) -> BirefnetResources:
    """Look up the measurement for this model's resolution
    ({@link BirefnetResourceMeasurement.by_resolution}).

    The key is the manifest's `imageWidth` — the lookup goes by the **declared resolution**, not by
    the spelling of the model name, so renaming does not break the link to the measurement.
    """
    resolution = str(config["imageWidth"])
    resources = measurement.by_resolution.get(resolution)
    if resources is None:
        raise ValueError(
            f"execution resources for a {resolution}² input were not measured"
            f" (measured: {' / '.join(measurement.by_resolution)}) — unmeasured numbers are not"
            " stated"
        )
    return resources


def _seat_dtypes(model: Mapping[str, Any]) -> list[str]:
    """Matte storage dtypes the model's quant presets point at (manifest order, no duplicates)."""
    seen: list[str] = []
    for quant in model["quants"].values():
        dtype = quant["weights"][BIREFNET_MATTE_WEIGHTS]
        if dtype not in seen:
            seen.append(dtype)
    return seen


def _birefnet_relation(manifest: Mapping[str, Any], entry: BirefnetCheckpoint) -> str | None:
    """Derive `base_model_relation` (`quantized` only when the default quant's downloaded file
    rounds the values).

    The same "default quant's downloaded file" rule as irodori / sbv2 / anima. When the default is a
    preset that does not change the values (f32, or HR's f16), it is none of the 4 values, so it is
    left out (the CardMetadata doc — picking the closest of the 4 values would give the card a claim
    that is not true).
    """
    model = default_model(manifest)
    dtype = model["quants"][model["defaultQuant"]]["weights"][BIREFNET_MATTE_WEIGHTS]
    if birefnet_storage_bits(dtype) < birefnet_storage_bits(entry.stored_dtype):
        return "quantized"
    return None


def _birefnet_metadata(manifest: Mapping[str, Any], entry: BirefnetCheckpoint) -> CardMetadata:
    """Build the frontmatter (`base_model` is the one upstream this repository redistributes).

    Even with two models listed there is one upstream — what shares the repository is the same
    checkpoint **at different resolutions**, not different weights.
    """
    return CardMetadata(
        pipeline_tag=BIREFNET_PIPELINE_TAG,
        base_model=(entry.repo,),
        base_model_relation=_birefnet_relation(manifest, entry),
        license=BIREFNET_LICENSE,
        tags=(BIREFNET_PIPELINE_TAG, "background-removal", "webgpu", "birefnet"),
    )


def _birefnet_overview(manifest: Mapping[str, Any], entry: BirefnetCheckpoint) -> list[str]:
    config = default_model(manifest)["pipelineConfig"]
    default_name = manifest["defaultModel"]
    return [
        "## What is this",
        "",
        "A background-removal distribution, converted into the WebGPU inference runtime",
        "**Karume**'s container format (a `.krm` part sequence whose first part carries the graph",
        "and model descriptors). Runs as-is in the browser and in Deno.",
        "",
        f"The weights are {entry.tagline}.",
        "",
        "- One graph, one call: pixels in, an **8-bit alpha matte** out, at the same size as the",
        "  image you handed over.",
        "- **One model per input resolution.** The resolution is baked into the graph (window",
        "  masks and padding constants are per-resolution), so it is picked with `model`, not at",
        "  call time. Each model declares its own resize target below; the bigger one holds more",
        "  detail at the edges and costs more GPU memory and time.",
        "- **Pre- and post-processing are included.** The pipeline resizes to"
        f" {config['imageWidth']} × {config['imageHeight']} for the default model"
        f" `{default_name}`,",
        "  normalizes with the constants below, then takes the sigmoid of the logits, scales the",
        "  matte back to the original resolution and rounds it to 8 bits — in the order and with",
        "  the interpolation upstream uses for this checkpoint (listed per model below). Decoding",
        "  PNG / JPEG is *not* part of this — use `createImageBitmap` in the browser, or any",
        "  decoder in Deno.",
        "- **Compositing is yours.** The pipeline returns the matte, not a cut-out: whether alpha",
        "  goes into an RGBA buffer, gets composited over a flat colour, or is fed to a colour",
        "  decontamination pass is a decision this repository should not make for you.",
        "- Not readable by transformers (it's a different container with an embedded graph); the"
        f" reader is a pipeline that implements `{BIREFNET_SUPPORTED_PIPELINE}`.",
        f"- Exporter used for the conversion: `{manifest['generator']}`. The distribution manifest"
        f" is `karume.json` (`{manifest['format']}`).",
    ]


def _birefnet_base_weights(manifest: Mapping[str, Any], checkpoint: str) -> list[str]:
    """Attribution section: the list of changes and the storage per quant preset
    ({@link birefnet_storage_lines}).

    The presets' storage dtypes are taken from the manifest (the module doc's MUST — dtype labels
    are not kept by hand).
    """
    entry = _birefnet_entry(checkpoint)
    return [
        "## Base weights and attribution",
        "",
        f"- **Weights**: [{entry.repo}](https://huggingface.co/{entry.repo}), licensed"
        f" **{BIREFNET_LICENSE}** (as of retrieval) — a verbatim copy of the license, carrying"
        " the copyright notices that apply to this repository, is in `LICENSE.md`.",
        *entry.attribution,
        f"- **Architecture**: BiRefNet ([{BIREFNET_PAPER}](https://{BIREFNET_PAPER})).",
        "- **Changes made here** (also listed in `NOTICE.md`): conversion into the Karume",
        "  container format. No retraining and no fine-tuning; the stored values per quant seat",
        "  are listed below. The graph is the upstream",
        "  `forward` with layout-only rewrites (windowing, the shifted-window roll, spatial",
        "  padding and the patch merges were folded into equivalent operations — bit-exact),",
        "  plus three rewrites that are equivalent up to floating-point rounding:",
        "  inference-time `BatchNorm2d` became a per-channel affine, the ASPP image-level",
        "  pooling became a two-stage sum, and the decoder tail's 1×1 convolution was swapped",
        "  with the bilinear upsample it used to follow (both are linear, so they commute — this",
        "  removes two full-resolution intermediates).",
        *birefnet_storage_lines(checkpoint, _seat_dtypes(default_model(manifest))),
    ]


def _birefnet_usage(manifest: Mapping[str, Any], repo: str) -> list[str]:
    """Policy of the usage example: keep the minimal working form live, and **list the optional
    arguments an ordinary use case is likely to need as commented-out lines** (the selectable values
    go in a comment on the same line — derived mechanically from the manifest, so the list follows
    when models / quants are added). The reader takes the next step just by uncommenting.

    `segment()` takes only one image (the resolution is baked into the graph and cannot be chosen at
    run time — the public surface of `birefnet.Pipeline`), so this pipeline's only optional knobs
    are model and quant.

    NOTE: `fromAssets` is **not advertised** (ruling of 2026-08-29). It can read split distributions
    now (X2-101), but it is a surface for local debugging that assumes you hold the bytes yourself;
    the ordinary entry point for a reader using HF is `fromPretrained`. Listing both would leave the
    reader to decide "which one do I use".
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
        'import { Pipeline } from "jsr:@karume/models/birefnet";',
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
        "const image = { data: pixels, width, height };",
        "const matte = await pipeline.segment(image);",
        "",
        "// matte.data is one alpha byte per pixel, same width/height as the input.",
        "// Straight alpha into an RGBA buffer (canvas, ImageData, an RGBA encoder):",
        "const rgba = new Uint8Array(matte.width * matte.height * 4);",
        "for (let i = 0; i < matte.data.length; i += 1) {",
        "  for (let c = 0; c < 3; c += 1) rgba[i * 4 + c] = image.data[i * 3 + c];",
        "  rgba[i * 4 + 3] = matte.data[i];",
        "}",
        "```",
        "",
        "`segment()` keeps one GPU session alive for the lifetime of the pipeline, so processing",
        "many images uploads the weights once; concurrent calls are queued rather than run side by",
        "side. Weights are fetched once and cached (verified against `karume.json`'s `size` /",
        "`sha256`).",
    ]


def _birefnet_postprocess(config: Mapping[str, Any]) -> tuple[str, ...]:
    """Look up the description of this model's post-processing procedure
    ({@link BIREFNET_POSTPROCESS_DESCRIPTIONS}).

    A manifest without the field fails with the same wording as a procedure without a description (a
    bare `KeyError` does not say what is missing).
    """
    procedure = config.get("postprocess")
    description = BIREFNET_POSTPROCESS_DESCRIPTIONS.get(procedure)
    if description is None:
        raise ValueError(
            f"no description for post-processing procedure {procedure!r}"
            f" (known: {' / '.join(BIREFNET_POSTPROCESS_DESCRIPTIONS)}) — an extrapolated"
            " description is not published"
        )
    return description


def _birefnet_shape(model: Mapping[str, Any], *, entry: BirefnetCheckpoint) -> list[str]:
    """Preprocessing constants and input / output shapes (what the user passes in and gets back can
    be read here).

    The execution resources are a separate measurement per model (= per resolution), so they go in
    this section — the total allocation differs by more than 2× between 1024² and 2048², and that is
    exactly what decides "which one to pick". The measurement is this repository's checkpoint's
    ({@link BirefnetCheckpoint.resources}).
    """
    config = model["pipelineConfig"]
    measurement = entry.resources
    resources = _birefnet_resources(measurement, config)
    if measurement.quant not in model["quants"]:
        raise ValueError(
            f"the quant '{measurement.quant}' the execution resources were measured with is not"
            f" among this model's presets (presets: {sorted(model['quants'])}) — numbers for an"
            " unmeasured preset are not stated"
        )
    mean = " / ".join(str(value) for value in config["imageMean"])
    std = " / ".join(str(value) for value in config["imageStd"])
    return [
        "### Input and output",
        "",
        "The resize target is the exported graph's own input shape; the normalization constants",
        "are the upstream preprocessor's (ImageNet statistics).",
        "",
        f"- **input**: RGB8 pixels, resized to {config['imageWidth']} ×"
        f" {config['imageHeight']} ({config['interpolation']}, antialiased). The aspect ratio is",
        "  **not** preserved — there is no crop and no padding, matching the upstream processor.",
        f"- **normalization**: `(pixel / 255 - mean) / std`, mean {mean}, std {std}",
        "- **output**: one alpha byte per pixel at the size of the image you passed in (the graph",
        "  itself emits pre-sigmoid logits; the sigmoid and the resize back happen on the host).",
        *_birefnet_postprocess(config),
        f"- **required GPU memory**: {resources.total} allocated in total with the"
        f" `{measurement.quant}` quant — its",
        f"  weights listed above, resident, plus {resources.intermediates} for the intermediate"
        " tensors — and",
        f"  {resources.binding} ({measurement.conditions}); one image takes {resources.run}.",
        "  WebGPU's default `maxStorageBufferBindingSize` is 128 MiB, so this is in practice a",
        "  desktop-class GPU requirement. `karume.json` does not declare it: the declared limits",
        "  cover the resident weights and state, not the intermediate tensors a run allocates.",
    ]


def render_birefnet_model_card(
    manifest: Mapping[str, Any],
    repo: str,
    checkpoint: str,
    host_assets: Mapping[FileRefKey, int] = {},
) -> str:
    """Assemble the `README.md` body of a BiRefNet family distribution (pure function, trailing
    newline).

    `checkpoint` is the weights this repository ships (`"hr"` / `"lucida"`) — the model names in the
    manifest are resolutions, so the attribution cannot be determined from them. The pipeline entry
    passes it in (module doc).
    """
    require_pipeline(manifest, BIREFNET_SUPPORTED_PIPELINE)
    entry = _birefnet_entry(checkpoint)
    metadata = _birefnet_metadata(manifest, entry)
    return render(
        (
            frontmatter(metadata),
            introduction(entry.title, hf_upstreams(metadata.base_model)),
            _birefnet_overview(manifest, entry),
            [""],
            _birefnet_base_weights(manifest, checkpoint),
            [""],
            models(manifest),
            [""],
            _birefnet_usage(manifest, repo),
            *model_sections(
                manifest,
                (partial(quants, host_assets=host_assets), partial(_birefnet_shape, entry=entry)),
            ),
        )
    )
