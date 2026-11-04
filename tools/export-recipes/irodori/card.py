"""Model card (`README.md`) of the Irodori distribution — pure functions derived from the
manifest.

The general-purpose rendering parts (frontmatter, model list, quant table, assembling the sections)
live in `karume.modelcard`. This module holds only **facts specific to Irodori**: attribution (where
the weights come from, the bundled codec and text backbone, the licenses) and what this pipeline's
card says.

There is a single attribution profile (the set of main model + bundled text backbone + codec
(+ its original Apache 2.0 weights) is the same for every model; only the main model's upstream
repository changes) — unlike SBV2 there is no axis along which the origin splits by voice family,
so {@link irodori.distribution.PIPELINE} puts a single slot in `card_profiles`.

MUST: **every number, download size, quant table and dtype label is derived from the manifest**
(the same MUST of `karume.modelcard` applies as is). The only constants allowed here are facts that
**do not exist in the manifest**.
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
    default_model,
    from_pretrained,
    frontmatter,
    introduction,
    knob,
    model_sections,
    models,
    quants,
    render,
    require_pipeline,
)

#: The pipeline contract this template can describe (ADR 0041 §2 — per model).
IRODORI_SUPPORTED_PIPELINE = "irodori/1"

IRODORI_PIPELINE_TAG = "text-to-speech"


@dataclass(frozen=True)
class IrodoriUpstream:
    """Upstream facts of one distributed model (not in the manifest, so held here — the MUST at
    the top)."""

    #: Where the weights come from (the HF repository placed by hand in
    #: `inputs/irodori/<model name>/`).
    repo: str
    #: Display name used in the title and the bold text of the overview (matches what upstream
    #: calls itself).
    display: str


#: Distributed model name → upstream facts. v4.1-small is the successor with only the duration
#: predictor retrained (the other 683 tensors are bit-identical to v4 and the tokenizer is
#: byte-identical — measured 2026-09-01).
#: A model name missing from the table fails loudly — silently writing another version's
#: attribution would make the stated origin disagree with the actual weights.
#: Licenses checked at the source (2026-08-12 / v4.1 on 2026-09-01, `tags` of the HF models API):
#: all are `license: mit`.
IRODORI_UPSTREAMS: Mapping[str, IrodoriUpstream] = {
    "v4-small": IrodoriUpstream(
        repo="Aratako/Irodori-TTS-v4-Small", display="Irodori-TTS v4 Small"
    ),
    "v4.1-small": IrodoriUpstream(
        repo="Aratako/Irodori-TTS-v4.1-Small", display="Irodori-TTS v4.1 Small"
    ),
}

#: The Japanese ModernBERT the text backbone was made from (the one the checkpoint's
#: `config_json` names in `text_tokenizer_repo` / `text_encoder_revision`). The same for every
#: upstream version (the text encoder is frozen from v4 to v4.1 — the same root as the measurement
#: above). It is redistributed, so it is listed in the attribution.
#: License checked at the source (2026-08-12): `license: mit`.
IRODORI_TEXT_BACKBONE_MODEL = "sbintuitions/modernbert-ja-310m"


def irodori_upstream(model_name: str) -> IrodoriUpstream:
    """Look up the upstream facts of a distributed model name (an unknown name fails loudly)."""
    if model_name not in IRODORI_UPSTREAMS:
        known = " / ".join(sorted(IRODORI_UPSTREAMS))
        raise ValueError(
            f"model name '{model_name}' has no upstream in {__name__}.IRODORI_UPSTREAMS"
            f" (known: {known}) — add the origin and display name of a new version to that table"
            " first"
        )
    return IRODORI_UPSTREAMS[model_name]


#: The codec that turns latents into a waveform (a separate repository with separate weights
#: upstream — **bundled** in this distribution).
#: Its license is MIT (checked at the source in `docs/research/2026-08-11-irodori-source-recon.md`).
#: It is redistributed, so it is listed both in `base_model` and in the attribution section.
IRODORI_CODEC_MODEL = "Aratako/Semantic-DACVAE-Japanese-32dim"

#: The upstream chain of the codec weights (checked 2026-09-24 against the HF models API and
#: each README).
#: The `base_model` of `IRODORI_CODEC_MODEL` (`47376ee2…`, `license: mit`) is
#: `IRODORI_CODEC_PARENT_MODEL` (`96adcf19…`, `license: mit`), whose README says "originally
#: facebook/dacvae-watermarked" / "Weights derived from facebook/dacvae-watermarked".
#: The parent's `base_model` is `IRODORI_CODEC_ORIGIN_MODEL` (`8680102d…`, `license: apache-2.0`).
#: The original weights are Apache 2.0, so the distribution repository carries the Apache license
#: text and attribution in addition to the MIT attribution
#: (`irodori.distribution.irodori_root_files`). The `LICENSE` of the original code repository
#: (GitHub `facebookresearch/dacvae`) is Apache 2.0 too. Neither the HF side nor the GitHub side has
#: a `NOTICE` file (file listing of 2026-09-24), so there is no upstream notice to preserve.
#:
#: NOTE: the end of the README of HF `facebook/dacvae-watermarked` says "licensed under the SAM
#: License", which disagrees with the same repository's license tag (`apache-2.0`). The GitHub
#: README says "licensed under Apache-2.0", and discussions/1 of the same repository confirms that
#: the README statement is the mistake (checked by the maintainer 2026-09-24). Here we follow the
#: machine-readable tag and the bundled `LICENSE`.
IRODORI_CODEC_PARENT_MODEL = "Aratako/Semantic-DACVAE-Japanese"
IRODORI_CODEC_ORIGIN_MODEL = "facebook/dacvae-watermarked"

#: License identifier of the distribution (the HF tag in the card's frontmatter — upstream
#: Irodori-TTS and the direct upstream of the bundled codec, {@link IRODORI_CODEC_MODEL}, are both
#: `license: mit`). The provenance baked into the main model's containers
#: (`irodori.export.PROVENANCE`) is taken from here too — the two tables must not move
#: independently.
#:
#: NOTE: MIT alone does not cover the codec weights — their original weights are the Apache 2.0
#: {@link IRODORI_CODEC_ORIGIN_MODEL} (the chain above), so the codec containers' identifier is
#: held separately by {@link IRODORI_CODEC_LICENSE}.
IRODORI_LICENSE = "mit"

#: License identifier baked into the codec containers (`codec_decoder` / `codec_encoder` —
#: `irodori.dacvae.export.PROVENANCE`). It states the reality of the chain above (an MIT derivative
#: + Apache 2.0 original weights) by joining the HF identifiers with SPDX `AND` — so that when a
#: component is taken out on its own (the `components` slot of `fromPretrained`), the container
#: alone shows that the conditions of Apache 2.0 §4(a)(b) apply (2026-09-24 full review W-RC4-1,
#: ruling of 2026-09-25). The container's `license` only carries the identifier string; its readers
#: (core `karume.container` / TS `descriptor.ts`) do not check the vocabulary.
IRODORI_CODEC_LICENSE = "mit AND apache-2.0"


def _irodori_metadata(upstream: IrodoriUpstream) -> CardMetadata:
    return CardMetadata(
        pipeline_tag=IRODORI_PIPELINE_TAG,
        base_model=(upstream.repo, IRODORI_TEXT_BACKBONE_MODEL, IRODORI_CODEC_MODEL),
        # What the default quant (`i8-a8`) downloads is the int8 series, so `quantized` is
        # declared (same shape as sbv2 / anima.
        # The old "kept as f32, not quantized" was a stale premise from before the i8 series was
        # bundled).
        base_model_relation="quantized",
        license=IRODORI_LICENSE,
        tags=("text-to-speech", "webgpu", "japanese"),
    )


#: Demo inputs of the usage snippet (Japanese TTS inputs, so they stay Japanese — the language
#: rules of CLAUDE.md).
IRODORI_DEMO_TEXT = "こんにちは、これはテストです。"
IRODORI_DEMO_CAPTION = "落ち着いた女性の声で、ゆっくりと丁寧に話している。"


def _irodori_overview(manifest: Mapping[str, Any], upstream: IrodoriUpstream) -> list[str]:
    sample_rate = default_model(manifest)["pipelineConfig"]["sampleRate"]
    return [
        "## What is this",
        "",
        f"A Japanese text-to-speech distribution: the **{upstream.display}** rectified-flow DiT",
        "converted into the WebGPU inference runtime **Karume**'s container format (a `.krm` part",
        "sequence whose first part carries the graph and model descriptors). Runs as-is in the",
        "browser and in Deno.",
        "",
        "- Nine graphs make up the chain: a shared Japanese ModernBERT `backbone`, the",
        "  `text_proj` and `caption_proj` condition projectors, the reference-latent `speaker`",
        "  encoder, the `duration` predictor, the DiT split in two — `dit_context` (the",
        "  conditioning keys / values, once per generation) and `dit` (one forward per Euler step,",
        "  plus one per classifier-free-guidance branch) — and the DACVAE codec"
        f" ([{IRODORI_CODEC_MODEL}](https://huggingface.co/{IRODORI_CODEC_MODEL}))",
        "  as `codec_decoder` / `codec_encoder`.",
        "- **Text in, waveform out.** `generate()` returns f32 mono samples at the codec's sample",
        "  rate, ready for `encodeWav`. The decoder is run in tiles so that it also fits GPUs with",
        "  the default 128MiB storage-buffer limit; the tiling is bit-exact against a single-shot",
        "  decode.",
        "- **Voice cloning is wired up both ways**: a reference speaker is passed either as audio",
        "  (mono f32 — what `decodeWav` returns), which `codec_encoder` turns into a DACVAE",
        "  latent, or as such a latent directly. Reference audio must already be at this",
        f"  distribution's own {sample_rate} Hz: there is no resampler, and a mismatch is refused",
        "  rather than silently converted. Unlike the decoder, `codec_encoder` is not tiled, so a",
        "  long reference can exceed the default 128MiB storage-buffer limit.",
        "- Not readable by the upstream implementation (it's a different container with an embedded"
        f" graph); the reader is a pipeline that implements `{IRODORI_SUPPORTED_PIPELINE}`.",
        f"- Exporter used for the conversion: `{manifest['generator']}`. The distribution manifest"
        f" is `karume.json` (`{manifest['format']}`).",
    ]


def _irodori_base_weights(upstream: IrodoriUpstream) -> list[str]:
    """The attribution section. It claims conversion + quantization (producing the int8 series)
    — no retraining or fine-tuning."""
    return [
        "## Base weights and attribution",
        "",
        "Converted into the container format — the original checkpoint is not distributed here.",
        "",
        f"- **Weights**: [{upstream.repo}](https://huggingface.co/{upstream.repo}),"
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
        "  text-to-audio runs from this repository alone. Its weights derive from"
        f" [{IRODORI_CODEC_ORIGIN_MODEL}](https://huggingface.co/{IRODORI_CODEC_ORIGIN_MODEL}),",
        f"  licensed **Apache-2.0** (as of retrieval), by way of {IRODORI_CODEC_PARENT_MODEL};",
        "  the Apache License 2.0 text is included in `LICENSE.md` for the codec components.",
        "- **Changes made here** (also listed in `NOTICE.md`): conversion into the Karume",
        "  container format and **quantization** of the weights — every component is stored",
        "  as `f32` / `f16` / `i8` series, and the DiT (`dit` / `dit_context`) adds an `i4` series",
        "  rounded with GPTQ calibration (the quant table below says which storage each quant",
        "  selects; `f32` is not quantized). The codec stores its weight-normalized convolutions",
        "  as effective weights, keeps only the mean half of `quantizer.in_proj` in",
        "  `codec_encoder`, and bypasses the watermarking branch in `codec_decoder`; constant",
        "  sub-expressions are precomputed. No retraining, no fine-tuning.",
    ]


def _irodori_usage(manifest: Mapping[str, Any], repo: str) -> list[str]:
    """Policy of the usage example: keep the minimal working form, and **list the optional
    arguments an ordinary use case is likely to need, commented out** (the selectable values are
    listed in a comment on the same line — derived from the manifest, so the list follows when
    series are added). The reader takes the next step just by uncommenting.

    NOTE: `fromAssets` is **not advertised** (ruling of 2026-08-29). It can read the split
    distribution too (X2-101), but it is a surface for local debugging that assumes you hold the
    bytes yourself; the ordinary entry point for a reader using HF is `fromPretrained`. Listing
    both would leave the reader to decide which one to use.
    """
    model_name = manifest["defaultModel"]
    model = default_model(manifest)
    quant = model["defaultQuant"]
    model_names = " / ".join(sorted(manifest["models"]))
    quant_names = " / ".join(sorted(model["quants"]))
    sample_rate = model["pipelineConfig"]["sampleRate"]
    return [
        "## Usage",
        "",
        "```ts",
        'import { decodeWav, encodeWav } from "jsr:@karume/models";',
        'import { Pipeline } from "jsr:@karume/models/irodori";',
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
        "const audio = await pipeline.generate({",
        f'  text: "{IRODORI_DEMO_TEXT}",',
        "",
        "  // Voice Design — describe the voice in Japanese prose:",
        f'  // caption: "{IRODORI_DEMO_CAPTION}",',
        "",
        "  // Voice cloning — condition on a reference speaker. The WAV must already be",
        f"  // {sample_rate} Hz — any channel count is averaged to mono; there is no resampler,",
        "  // so a sample-rate mismatch is refused:",
        '  // speaker: { audio: decodeWav(await Deno.readFile("reference.wav")) },',
        "  // ...or pass a DACVAE latent you saved earlier instead of the audio:",
        "  // speaker: { latent: savedLatent },",
        "",
        "  seed: 42, // same seed + same inputs → same audio",
        "  // durationSeconds: 5, // override the predicted utterance length (seconds)",
        "});",
        "",
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
    ]


def _irodori_shape(model: Mapping[str, Any]) -> list[str]:
    """Model-specific numbers (the side that meshes with the graph declarations) — the user's
    input limits can be read here."""
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
    """Runtime knobs (the side that fills in what was not passed to `generate()`)."""
    config = model["pipelineConfig"]
    scales = config["cfgScales"]
    return [
        "### Defaults",
        "",
        "The sampler knobs are fixed by the manifest — `generate()` takes none of them.",
        "",
        f"- **steps**: {config['steps']} Euler steps (`initScale` {config['initScale']})",
        "- **guidance**: "
        + " / ".join(f"{name} {knob(scale)}" for name, scale in scales.items())
        + f", applied for t in [{config['cfgMinT']}, {config['cfgMaxT']}]",
        f"- **duration**: clamped to [{config['minSeconds']}, {config['maxSeconds']}] seconds"
        " (the duration predictor decides within that, unless `durationSeconds` is passed)",
        "",
        "`seed` is the one knob the manifest does not carry — it defaults to `0`, and the same",
        "seed with the same request gives the same audio.",
    ]


def render_irodori_model_card(
    manifest: Mapping[str, Any], repo: str, host_assets: Mapping[FileRefKey, int] = {}
) -> str:
    """Assemble the `README.md` body of the Irodori distribution (pure function, trailing
    newline included)."""
    require_pipeline(manifest, IRODORI_SUPPORTED_PIPELINE)
    upstream = irodori_upstream(manifest["defaultModel"])
    metadata = _irodori_metadata(upstream)
    return render(
        (
            frontmatter(metadata),
            introduction(f"{upstream.display} — Karume", hf_upstreams(metadata.base_model)),
            _irodori_overview(manifest, upstream),
            [""],
            _irodori_base_weights(upstream),
            [""],
            models(manifest),
            [""],
            _irodori_usage(manifest, repo),
            *model_sections(
                manifest,
                (partial(quants, host_assets=host_assets), _irodori_shape, _irodori_defaults),
            ),
        )
    )
