"""Model card (`README.md`) of the Irodori distribution — pure functions derived from the
manifest.

SPEC: docs/spec/distribution-cards.md#the-model-card

The general-purpose rendering parts (frontmatter, model list, quant table, assembling the sections)
live in `karume.modelcard`. This module holds only **facts specific to Irodori**: attribution (where
the weights come from, the bundled codec and text backbone, the licenses) and what this pipeline's
card says.

There is a single attribution profile (the set of main model + bundled text backbone + codec
(+ its original Apache 2.0 weights) is the same for every model; only the main model's upstream
repository changes) — unlike SBV2 there is no axis along which the origin splits by voice family,
so {@link irodori.distribution.PIPELINE} puts a single slot in `card_profiles`. The v4-Large line
is the exception that has its own Pipeline ({@link irodori.distribution.LARGE_PIPELINE}): another
license (the Gemma Terms of Use), another text encoder (T5Gemma 2), its own attribution
({@link render_irodori_large_model_card}).

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
    RUNTIME_SENTENCE,
    CardMetadata,
    FileRefKey,
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
    render,
    require_pipeline,
    upstream_cards,
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
    #: The upstream commit the weights were taken from (THIRD_PARTY_NOTICES.md — the fetch record
    #: of the snapshot in `inputs/irodori/<model name>/`), pinned in the attribution.
    revision: str
    #: The model upstream distilled these weights from (a MeanFlow student names its RF teacher —
    #: ADR 0150), or `None`.
    teacher: str | None = None


#: Distributed model name → upstream facts. v4.1-small is the successor with only the duration
#: predictor retrained (the other 683 tensors are bit-identical to v4 and the tokenizer is
#: byte-identical — measured 2026-09-01).
#: v4.1-small-mf is upstream's MeanFlow student of v4.1-small (ADR 0150 — the whole DiT retrained
#: with an interval embedding added; everything else is v4.1-small's, measured 2026-11-14).
#: A model name missing from the table fails loudly — silently writing another version's
#: attribution would make the stated origin disagree with the actual weights.
#: Licenses checked at the source (2026-08-12 / v4.1 on 2026-09-01 / v4.1 MF on 2026-11-14 —
#: `license: mit` in the model card's frontmatter at the revision below): all are `license: mit`.
IRODORI_UPSTREAMS: Mapping[str, IrodoriUpstream] = {
    "v4-small": IrodoriUpstream(
        repo="Aratako/Irodori-TTS-v4-Small",
        display="Irodori-TTS v4 Small",
        revision="4c92c7ee2bb15c19a97cf4e86d24fd6bf33b0135",
    ),
    "v4.1-small": IrodoriUpstream(
        repo="Aratako/Irodori-TTS-v4.1-Small",
        display="Irodori-TTS v4.1 Small",
        revision="2b28324dc263ed5e6638b3cf3dd94c82ead07b4b",
    ),
    "v4.1-small-mf": IrodoriUpstream(
        repo="Aratako/Irodori-TTS-v4.1-Small-MF",
        display="Irodori-TTS v4.1 Small MF",
        revision="ccc78f5d480b6e51b69b2d5042a14c4da04fea6e",
        teacher="Aratako/Irodori-TTS-v4.1-Small",
    ),
}

#: The models of the **v4-Large line** — their own repositories and their own Pipeline
#: (`irodori.distribution.LARGE_PIPELINE`), apart from {@link IRODORI_UPSTREAMS}: upstream tags
#: them `license: gemma`, so the legal texts at the repository root differ from the MIT line's, and
#: `root_files` is one fixed set per Pipeline. A model name of one table is refused by the other
#: line's card ({@link irodori_large_upstream}).
#: License read at the source (2026-11-14): `license: gemma` in the frontmatter of upstream's
#: `README.md` at the revision below (the snapshot in `inputs/irodori/v4-large/` — its download
#: records name this commit for `README.md`, `GEMMA_TERMS_OF_USE.md` and `NOTICE`).
IRODORI_LARGE_UPSTREAMS: Mapping[str, IrodoriUpstream] = {
    "v4-large": IrodoriUpstream(
        repo="Aratako/Irodori-TTS-v4-Large",
        display="Irodori-TTS v4 Large",
        revision="2e0c55428ce97268a507f1feeb2478f8d9148e8b",
    ),
}

#: The origin of v4-Large's text encoder and tokenizer (upstream's model card: "the fine-tuned text
#: encoder from google/t5gemma-2-1b-1b"; the checkpoint's `config_json` names it in
#: `text_tokenizer_repo` / `caption_tokenizer_repo`). It is redistributed (the `backbone` component
#: and the tokenizer asset), so it is listed in `base_model` and in the attribution.
IRODORI_LARGE_TEXT_ENCODER_MODEL = "google/t5gemma-2-1b-1b"

#: The commit of {@link IRODORI_LARGE_TEXT_ENCODER_MODEL} the v4-Large checkpoint's `config_json`
#: names as its `text_encoder_revision` (read 2026-11-14 — THIRD_PARTY_NOTICES.md).
IRODORI_LARGE_TEXT_ENCODER_REVISION = "dd0a2683227859151b1730ca3a63087df5b5f39b"

#: License identifier of the v4-Large line (the HF tag in the card's frontmatter — upstream's own
#: tag). The same value is the license every container a v4-Large model exports must carry
#: (`irodori.distribution.assert_irodori_container_licenses`).
IRODORI_LARGE_LICENSE = "gemma"

#: The two Gemma documents upstream's model card links (its "License" section) — the card links the
#: same URLs; their verbatim texts are in the repository's `LICENSE.md`.
GEMMA_TERMS_URL = "https://ai.google.dev/gemma/terms"
GEMMA_PROHIBITED_USE_POLICY_URL = "https://ai.google.dev/gemma/prohibited_use_policy"

#: The `flowParameterization` value of a MeanFlow model in `pipelineConfig` (the spelling of
#: `irodori.distribution.IRODORI_FLOW_MEANFLOW`, which imports this module — kept as a literal
#: here to avoid the cycle; `tests/test_card.py` pins that the two agree).
IRODORI_MEANFLOW = "meanflow"


def _is_meanflow(model: Mapping[str, Any]) -> bool:
    """Whether a manifest model entry declares the MeanFlow sampler (absent = RF)."""
    return model["pipelineConfig"].get("flowParameterization") == IRODORI_MEANFLOW


#: The Japanese ModernBERT the text backbone was made from (the one the checkpoint's
#: `config_json` names in `text_tokenizer_repo` / `text_encoder_revision`). The same for every
#: upstream version (the text encoder is frozen from v4 to v4.1 — the same root as the measurement
#: above). It is redistributed, so it is listed in the attribution.
#: License checked at the source (2026-08-12): `license: mit`.
IRODORI_TEXT_BACKBONE_MODEL = "sbintuitions/modernbert-ja-310m"
IRODORI_TEXT_BACKBONE_LICENSE = "mit"

#: The commit of {@link IRODORI_TEXT_BACKBONE_MODEL} the checkpoints' `config_json` names as their
#: `text_encoder_revision` (the same in v4-small, v4.1-small and v4.1-small-MF, read 2026-11-26 —
#: THIRD_PARTY_NOTICES.md).
IRODORI_TEXT_BACKBONE_REVISION = "77675fc96a7e445e982e2ba90246b816efc74ec6"


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


def irodori_large_upstream(model_name: str) -> IrodoriUpstream:
    """Look up the upstream facts of a v4-Large-line model ({@link IRODORI_LARGE_UPSTREAMS}; any
    other name fails loudly — an MIT-line model never gets this line's Gemma attribution)."""
    if model_name not in IRODORI_LARGE_UPSTREAMS:
        known = " / ".join(sorted(IRODORI_LARGE_UPSTREAMS))
        raise ValueError(
            f"model name '{model_name}' has no upstream in {__name__}.IRODORI_LARGE_UPSTREAMS"
            f" (known: {known}) — the models of the MIT line are in IRODORI_UPSTREAMS and build"
            " with `--pipeline irodori`"
        )
    return IRODORI_LARGE_UPSTREAMS[model_name]


#: The codec that turns latents into a waveform (a separate repository with separate weights
#: upstream — **bundled** in this distribution).
#: Its license is MIT (checked at the source in `docs/research/2026-08-11-irodori-source-recon.md`).
#: It is redistributed, so it is listed both in `base_model` and in the attribution section.
IRODORI_CODEC_MODEL = "Aratako/Semantic-DACVAE-Japanese-32dim"

#: The commit of {@link IRODORI_CODEC_MODEL} the codec containers were exported from (the HF
#: cache's `refs/main` — the local fetch record, THIRD_PARTY_NOTICES.md).
IRODORI_CODEC_REVISION = "47376ee24834d7a05a48ebabfe3cde29b3c5e214"

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

#: The HF license identifiers of the codec's chain above (`IRODORI_CODEC_MODEL` and its parent are
#: `license: mit`, the original weights `license: apache-2.0`).
IRODORI_CODEC_MODEL_LICENSE = "mit"
IRODORI_CODEC_ORIGIN_LICENSE = "apache-2.0"

#: The license of upstream's training / inference implementation (GitHub `Aratako/Irodori-TTS` —
#: its `LICENSE` is the MIT License), named by its HF identifier as the card names every license.
IRODORI_IMPLEMENTATION_LICENSE = "mit"

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
        # What the default quant (`i8-a8`; `i8` for the MeanFlow student) downloads is the int8
        # series, so `quantized` is declared (same shape as sbv2 / anima.
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


def _irodori_overview(
    manifest: Mapping[str, Any],
    upstream: IrodoriUpstream,
    backbone: str = "a shared Japanese ModernBERT `backbone`",
) -> list[str]:
    """The overview section (`backbone` names the text encoder the line's checkpoints carry)."""
    model = default_model(manifest)
    sample_rate = model["pipelineConfig"]["sampleRate"]
    meanflow = _is_meanflow(model)
    flow = "MeanFlow-distilled" if meanflow else "rectified-flow"
    step = (
        (
            "one forward per sampling step — guidance was distilled in, so there are no",
            "  classifier-free-guidance branches",
        )
        if meanflow
        else (
            "one forward per Euler step,",
            "  plus one per classifier-free-guidance branch",
        )
    )
    return [
        "## What is this",
        "",
        f"A Japanese text-to-speech distribution: the **{upstream.display}** {flow} DiT",
        "converted into the WebGPU inference runtime **Karume**'s container format (a `.krm` part",
        "sequence whose first part carries the graph and model descriptors).",
        RUNTIME_SENTENCE,
        "",
        f"- Nine graphs make up the chain: {backbone}, the",
        "  `text_proj` and `caption_proj` condition projectors, the reference-latent `speaker`",
        "  encoder, the `duration` predictor, the DiT split in two — `dit_context` (the",
        f"  conditioning keys / values, once per generation) and `dit` ({step[0]}",
        f"{step[1]}) — and the DACVAE codec"
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


def _irodori_weights_line(upstream: IrodoriUpstream) -> list[str]:
    """The first bullet of the attribution: where the weights come from (with the upstream
    revision and the distillation teacher when the table names them)."""
    lines = [
        f"- **Weights**: {pinned(upstream.repo, upstream.revision)},"
        f" licensed **{IRODORI_LICENSE}**",
    ]
    if upstream.teacher is not None:
        lines += [
            "  (as of retrieval). Upstream distilled it from"
            f" [{upstream.teacher}](https://huggingface.co/{upstream.teacher}) with MeanFlow:",
            "  the whole DiT was retrained (with an added interval embedding) to take few, large",
            "  steps, and the guidance was fused into the teacher's targets, so it runs without",
            "  classifier-free guidance. Its condition encoders and duration predictor are the",
            "  teacher's. The training / inference implementation it comes with",
        ]
        return lines
    return [*lines, "  (as of retrieval). The training / inference implementation it comes with"]


def _irodori_backbone_lines(model: Mapping[str, Any]) -> list[str]:
    """Where the text backbone lives (the end of its attribution bullet): stored here, or
    referenced in the repository that stores it — read from the manifest."""
    holder = _component_holder(model, ("backbone",))
    if holder is None:
        return [
            f"  licensed **{IRODORI_TEXT_BACKBONE_LICENSE}** (as of retrieval). It is redistributed"
            " here in the container format as",
            "  the `backbone` component, so that license travels with this repository too.",
        ]
    return [
        f"  licensed **{IRODORI_TEXT_BACKBONE_LICENSE}** (as of retrieval). This repository does"
        " not store it: the manifest",
        f"  references the containers `{holder}` ships as the `backbone` component,",
        "  and they are fetched from there. That license applies to the backbone this distribution",
        "  runs.",
    ]


def _irodori_codec_lines(model: Mapping[str, Any]) -> list[str]:
    """Where the codec lives (the middle of its attribution bullet, up to the link to its original
    weights): stored here, or referenced in the repository that stores it — read from the
    manifest."""
    holder = _component_holder(model, IRODORI_CODEC_COMPONENTS)
    origin = (
        f" [{IRODORI_CODEC_ORIGIN_MODEL}](https://huggingface.co/{IRODORI_CODEC_ORIGIN_MODEL}),"
    )
    if holder is None:
        return [
            "  (as of retrieval). Upstream ships it as a separate repository; it is redistributed"
            " here",
            "  in the container format as the `codec_decoder` / `codec_encoder` components so that",
            f"  text-to-audio runs from this repository alone. Its weights derive from{origin}",
        ]
    return [
        "  (as of retrieval). Upstream ships it as a separate repository. This repository does not",
        f"  store it: the manifest references the containers `{holder}` ships as the",
        "  `codec_decoder` / `codec_encoder` components, and they are fetched from there so that",
        f"  text-to-audio runs from this distribution. Its weights derive from{origin}",
    ]


def _irodori_base_weights(manifest: Mapping[str, Any], upstream: IrodoriUpstream) -> list[str]:
    """The attribution section. It claims conversion + quantization (producing the int8 series)
    — no retraining or fine-tuning. Where the text backbone and the codec live is read from the
    manifest: a component this repository stores is redistributed here, one it references is
    fetched from the repository named."""
    model = default_model(manifest)
    return [
        "## Base weights and attribution",
        "",
        "Converted into the container format — the original checkpoint is not distributed here.",
        "",
        *_irodori_weights_line(upstream),
        "  ([Aratako/Irodori-TTS](https://github.com/Aratako/Irodori-TTS)) is licensed"
        f" **{IRODORI_IMPLEMENTATION_LICENSE}** as well;",
        "  Karume's runtime contains none of that code — it is an independent implementation that",
        "  reads these weights from its own container format.",
        "- **Text backbone**: fine-tuned from"
        f" {pinned(IRODORI_TEXT_BACKBONE_MODEL, IRODORI_TEXT_BACKBONE_REVISION)},",
        *_irodori_backbone_lines(model),
        f"- **Codec**: {pinned(IRODORI_CODEC_MODEL, IRODORI_CODEC_REVISION)},"
        f" licensed **{IRODORI_CODEC_MODEL_LICENSE}**",
        *_irodori_codec_lines(model),
        f"  licensed **{IRODORI_CODEC_ORIGIN_LICENSE}** (as of retrieval), by way of"
        f" {IRODORI_CODEC_PARENT_MODEL};",
        "  a verbatim copy of that license is in `LICENSE.md` for the codec components.",
        f"{changes_made_here()} conversion into the Karume",
        "  container format and **quantization** of the weights — every component is stored",
        "  as `f32` / `f16` / `i8` series, and the DiT (`dit` / `dit_context`) adds an `i4` series",
        "  rounded with GPTQ calibration (the quant table below says which storage each quant",
        "  selects; `f32` is not quantized). The codec stores its weight-normalized convolutions",
        "  as effective weights, keeps only the mean half of `quantizer.in_proj` in",
        "  `codec_encoder`, and bypasses the watermarking branch in `codec_decoder`; constant",
        "  sub-expressions are precomputed. No retraining, no fine-tuning.",
        *upstream_cards([hf_link(repo) for repo in _irodori_metadata(upstream).base_model]),
    ]


def _irodori_usage(
    manifest: Mapping[str, Any], repo: str, rare_token_split: bool = True
) -> list[str]:
    """Policy of the usage example: keep the minimal working form, and **list the optional
    arguments an ordinary use case is likely to need, commented out** (the selectable values are
    listed in a comment on the same line — derived from the manifest, so the list follows when
    series are added). The reader takes the next step just by uncommenting.

    `rare_token_split` is whether the line's tokenizer asset carries the rare-token split table
    (ADR 0149 — the MIT line's does; v4-Large's Gemma tokenizer does not, as upstream applies no
    split to it). Without the table `splitRareTokens` has nothing to turn off, so it is not offered.

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
        *(
            ["  // splitRareTokens: false, // keep rare long words as single tokens (see Defaults)"]
            if rare_token_split
            else []
        ),
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


def _irodori_sampler_lines(config: Mapping[str, Any]) -> list[str]:
    """The step and guidance bullets of Defaults (a MeanFlow model has no runtime guidance)."""
    if config.get("flowParameterization") == IRODORI_MEANFLOW:
        return [
            f"- **steps**: {config['steps']} MeanFlow steps (t from 1 to 0 in equal intervals, one",
            "  conditional DiT forward each)",
            "- **guidance**: none at run time — it was fused into the teacher's targets during",
            "  distillation",
        ]
    scales = config["cfgScales"]
    return [
        f"- **steps**: {config['steps']} Euler steps (`initScale` {config['initScale']})",
        "- **guidance**: "
        + " / ".join(f"{name} {knob(scale)}" for name, scale in scales.items())
        + f", applied for t in [{config['cfgMinT']}, {config['cfgMaxT']}]",
    ]


def _irodori_defaults(model: Mapping[str, Any], rare_token_split: bool = True) -> list[str]:
    """Runtime knobs (the side that fills in what was not passed to `generate()`).
    `rare_token_split`: as in {@link _irodori_usage}."""
    config = model["pipelineConfig"]
    split = (
        [
            "`splitRareTokens` follows the tokenizer asset: when it carries the rare-token split",
            "table (upstream's default for this tokenizer), the text — never the caption — is"
            " encoded",
            "without the rare long tokens the model tends to misread, falling back to shorter"
            " pieces;",
            "`false` turns that off. A tokenizer without the table encodes unsplit, and asking it"
            " for",
            "the split is refused.",
        ]
        if rare_token_split
        else [
            "The tokenizer asset carries no rare-token split table (upstream splits no tokens for",
            "this tokenizer), so the text is encoded unsplit, and asking for the split",
            "(`splitRareTokens: true`) is refused.",
        ]
    )
    return [
        "### Defaults",
        "",
        "The sampler knobs are fixed by the manifest — `generate()` takes none of them.",
        "",
        *_irodori_sampler_lines(config),
        f"- **duration**: clamped to [{config['minSeconds']}, {config['maxSeconds']}] seconds"
        " (the duration predictor decides within that, unless `durationSeconds` is passed)",
        "",
        "`seed` is the one knob the manifest does not carry — it defaults to `0`, and the same",
        "seed with the same request gives the same audio.",
        "",
        *split,
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
            _irodori_base_weights(manifest, upstream),
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


#: The codec's two components (the keys of `irodori.distribution.IRODORI_CODEC_DIRS`, which imports
#: this module — kept as literals here to avoid the cycle; `tests/test_card.py` pins that the two
#: agree).
IRODORI_CODEC_COMPONENTS: tuple[str, ...] = ("codec_decoder", "codec_encoder")

#: The DiT pair (`irodori.distribution.IRODORI_DIT_ROLES`, the same cycle and the same pin).
IRODORI_DIT_COMPONENTS: tuple[str, ...] = ("dit", "dit_context")


def _irodori_large_metadata(upstream: IrodoriUpstream) -> CardMetadata:
    return CardMetadata(
        pipeline_tag=IRODORI_PIPELINE_TAG,
        base_model=(upstream.repo, IRODORI_LARGE_TEXT_ENCODER_MODEL, IRODORI_CODEC_MODEL),
        # Every preset stores the weights in another form (f16 / int8 / int4 — no f32 preset), so
        # `quantized` is declared, as for the MIT line.
        base_model_relation="quantized",
        # Upstream's own tag (no `license_name` / `license_link`: HF reads those with `other`
        # only, and upstream's frontmatter has neither).
        license=IRODORI_LARGE_LICENSE,
        tags=("text-to-speech", "webgpu", "japanese"),
    )


def irodori_storage_series(model: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    """`(the storage labels every component has, the labels only the DiT pair adds)`, in
    declaration order — what the attribution's list of changes says, read from the manifest.

    A label that neither every component nor exactly the DiT pair has fails: the sentence could not
    describe it.
    """
    weights = model["weights"]
    labels = list(dict.fromkeys(label for entry in weights.values() for label in entry))
    common = [label for label in labels if all(label in entry for entry in weights.values())]
    dit = [
        label
        for label in labels
        if label not in common
        and {name for name, entry in weights.items() if label in entry}
        == set(IRODORI_DIT_COMPONENTS)
    ]
    stray = [label for label in labels if label not in common and label not in dit]
    if stray:
        raise ValueError(
            f"storage labels {stray} are neither on every component nor exactly on the DiT pair"
            " — the card's list of changes cannot describe them"
        )
    return common, dit


def _component_holder(model: Mapping[str, Any], components: tuple[str, ...]) -> str | None:
    """The repository the parts of `components` live in (`None` = this repository) — read from the
    manifest's part references (ADR 0038 §7). Components split across places fail: the card
    states one place for them."""
    holders = {
        part.get("repo")
        for role in components
        for entry in model["weights"][role].values()
        for part in entry["container"]["parts"]
    }
    if len(holders) != 1:
        raise ValueError(
            f"the components {list(components)} are split across {sorted(map(str, holders))}"
            " — a component lives in one place"
        )
    (holder,) = holders
    return holder


def _irodori_large_codec_line(model: Mapping[str, Any]) -> str:
    """Where the codec components live: referenced in another repository, or stored here."""
    holder = _component_holder(model, IRODORI_CODEC_COMPONENTS)
    if holder is None:
        return "  (as of retrieval). It is redistributed here in the container format as the"
    return (
        f"  (as of retrieval). This repository does not store it: the manifest references the"
        f" containers `{holder}` ships as"
    )


def _irodori_large_base_weights(
    manifest: Mapping[str, Any], upstream: IrodoriUpstream
) -> list[str]:
    """The attribution section of the v4-Large line. Every license statement repeats what upstream
    states (its model card and tags) — it claims conversion + quantization, no retraining."""
    model = default_model(manifest)
    common, dit = irodori_storage_series(model)
    if dit not in ([], ["i4"]):
        raise ValueError(
            f"the DiT pair adds {dit} — the card says only an `i4` series rounded with GPTQ"
            " calibration (the rounding `irodori.distribution` requires of it)"
        )
    stored = " / ".join(f"`{label}`" for label in common)
    series = f"every component is stored as {stored} series" + (
        ", and the DiT (`dit` / `dit_context`) adds an `i4` series rounded with GPTQ calibration"
        if dit
        else ""
    )
    selects = "selects" if "f32" in common else "selects; no `f32` series is distributed"
    return [
        "## Base weights and attribution",
        "",
        "Converted into the container format — the original checkpoint is not distributed here.",
        "",
        f"- **Weights**: {pinned(upstream.repo, upstream.revision)},",
        f"  licensed **{IRODORI_LARGE_LICENSE}** (as of retrieval). Upstream's model card"
        " states that the model is",
        f"  subject to the [Gemma Terms of Use]({GEMMA_TERMS_URL}) because its shared text /"
        " caption encoder is",
        "  derived from"
        f" [{IRODORI_LARGE_TEXT_ENCODER_MODEL}](https://huggingface.co/{IRODORI_LARGE_TEXT_ENCODER_MODEL}),"
        " and that use and",
        "  redistribution must comply with those terms, including the"
        f" [Gemma Prohibited Use Policy]({GEMMA_PROHIBITED_USE_POLICY_URL}),",
        "  as well as the additional ethical restrictions listed on that model card. Both Gemma",
        "  texts are included in `LICENSE.md`, and the notice upstream ships with the model is in",
        "  `NOTICE.md`. The training / inference implementation it comes with",
        "  ([Aratako/Irodori-TTS](https://github.com/Aratako/Irodori-TTS)) is licensed"
        f" **{IRODORI_IMPLEMENTATION_LICENSE}**;",
        "  Karume's runtime contains none of that code — it is an independent implementation that",
        "  reads these weights from its own container format.",
        "- **Text encoder**: the `backbone` component is the checkpoint's fine-tuned text encoder",
        "  of"
        f" {pinned(IRODORI_LARGE_TEXT_ENCODER_MODEL, IRODORI_LARGE_TEXT_ENCODER_REVISION)},"
        " redistributed here in the container",
        "  format; the tokenizer asset is upstream's tokenizer of the same model, converted into",
        "  Karume's tokenizer format.",
        f"- **Codec**: {pinned(IRODORI_CODEC_MODEL, IRODORI_CODEC_REVISION)},"
        f" licensed **{IRODORI_CODEC_MODEL_LICENSE}**",
        _irodori_large_codec_line(model),
        "  the `codec_decoder` / `codec_encoder` components so that text-to-audio runs from this",
        "  distribution. Its weights derive from"
        f" [{IRODORI_CODEC_ORIGIN_MODEL}](https://huggingface.co/{IRODORI_CODEC_ORIGIN_MODEL}),",
        f"  licensed **{IRODORI_CODEC_ORIGIN_LICENSE}** (as of retrieval), by way of"
        f" {IRODORI_CODEC_PARENT_MODEL};",
        "  verbatim copies of both licenses are in `LICENSE.md` for the codec components.",
        f"{changes_made_here()} conversion into the Karume",
        f"  container format and **quantization** of the weights — {series}",
        f"  (the quant table below says which storage each quant {selects}).",
        "  The text encoder's sliding-window attention mask is precomputed as one constant, and",
        "  its embedding's end-of-image row is selected elementwise. The codec stores its",
        "  weight-normalized convolutions as effective weights, keeps only the mean half of",
        "  `quantizer.in_proj` in `codec_encoder`, and bypasses the watermarking branch in",
        "  `codec_decoder`; constant sub-expressions are precomputed. No retraining, no",
        "  fine-tuning.",
        *upstream_cards([hf_link(repo) for repo in _irodori_large_metadata(upstream).base_model]),
    ]


def render_irodori_large_model_card(
    manifest: Mapping[str, Any], repo: str, host_assets: Mapping[FileRefKey, int] = {}
) -> str:
    """Assemble the `README.md` body of a v4-Large-line distribution (pure function, trailing
    newline included). The same template as {@link render_irodori_model_card} except the text
    encoder, the attribution section and the absent rare-token split."""
    require_pipeline(manifest, IRODORI_SUPPORTED_PIPELINE)
    upstream = irodori_large_upstream(manifest["defaultModel"])
    metadata = _irodori_large_metadata(upstream)
    return render(
        (
            frontmatter(metadata),
            introduction(f"{upstream.display} — Karume", hf_upstreams(metadata.base_model)),
            _irodori_overview(
                manifest, upstream, backbone="the shared T5Gemma 2 text encoder `backbone`"
            ),
            [""],
            _irodori_large_base_weights(manifest, upstream),
            [""],
            models(manifest),
            [""],
            _irodori_usage(manifest, repo, rare_token_split=False),
            *model_sections(
                manifest,
                (
                    partial(quants, host_assets=host_assets),
                    _irodori_shape,
                    partial(_irodori_defaults, rare_token_split=False),
                ),
            ),
        )
    )
