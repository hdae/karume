"""Model card (`README.md`) of the vowel-detection distribution — a pure function derived
mechanically from the manifest.

The general-purpose rendering parts (frontmatter, model list, quant table, section assembly) live in
`karume.modelcard`. What lives here is **only the facts specific to vowel detection**: attribution
(source, license, training material) and what this pipeline's card says.

There is only one attribution (one upstream repository, one license), so there is no profile axis.

MUST: **every number, download size, quant table and dtype label is derived from the manifest**
(the same MUST of `karume.modelcard` applies unchanged). The only constants allowed here are facts
that **do not exist in the manifest**.
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import partial
from typing import Any

from karume.modelcard import (
    CardMetadata,
    FileRefKey,
    default_model,
    from_pretrained,
    frontmatter,
    model_sections,
    models,
    quants,
    render,
    require_pipeline,
)

#: The pipeline contract this template can describe (ADR 0041 §2 — per model).
VOWEL_DETECTOR_SUPPORTED_PIPELINE = "vowel-detector/1"

#: The HF pipeline tag. Per-frame 8-class classification is closest to `audio-classification`
#: (HF's vocabulary has no entry for frame classification). The body explains what it outputs.
VOWEL_DETECTOR_PIPELINE_TAG = "audio-classification"

VOWEL_DETECTOR_TITLE = "Vowel Detector for Lip-sync — Karume"

#: The upstream distribution (an ONNX version of the same weights). **These are the weights being
#: redistributed**, so it goes in `base_model`. The license is upstream's `LICENSE` / `NOTICE.txt`
#: (MIT, (c) 2026 Spectopathy).
VOWEL_DETECTOR_UPSTREAM = "hdae/vowel-detector"
VOWEL_DETECTOR_LICENSE = "mit"

#: Attribution of the training pipeline (verbatim from upstream's `NOTICE.txt`). **MIT does not
#: require redistributing the NOTICE, but upstream states explicitly that it wants the attribution
#: kept when the model is distributed**, so the card carries it mechanically. It is kept as one
#: table so that the attribution cannot go missing silently when only one copy changes.
VOWEL_DETECTOR_ATTRIBUTION: tuple[str, ...] = (
    "- **Teacher backbone (not distributed)**:"
    " [reazon-research/japanese-hubert-base-k2]"
    "(https://huggingface.co/reazon-research/japanese-hubert-base-k2)"
    " (**Apache-2.0**), © Reazon Human Interaction Laboratory. Used only to train an internal"
    " aligner/classifier; the teacher is **not** part of these weights — the CRNN is distilled"
    " from it.",
    "- **Reading scripts**:"
    " [ROHAN4600](https://github.com/mmorise/rohan4600) (**CC0-1.0**, © Masanori Morise) and the"
    " [ITA corpus](https://github.com/mmorise/ita-corpus) (public domain).",
    "- **Real speech**:"
    " [Common Voice ja v25.0](https://commonvoice.mozilla.org/ja/datasets) (**CC0-1.0**),"
    " © Mozilla Foundation and Common Voice contributors — weak g2p labels,"
    " confidence-filtered.",
    "- **Synthesized speech**: generated with"
    " [Style-Bert-VITS2](https://github.com/litagin02/Style-Bert-VITS2) (**AGPL-3.0**, used as a"
    " tool only — not distributed with, and not part of, these weights) from"
    " [AivisHub](https://hub.aivis-project.com/) voice models under"
    " [ACML 1.0](https://github.com/Aivis-Project/ACML/blob/master/ACML-1.0.md) or CC0:"
    " まお / コハク / morioki / fumifumi / 阿井田茂 / にせ / ろてじん / らせつん / 観測症"
    " (ACML 1.0) and zonoko (CC0). ACML 1.0 permits every use outside its listed prohibitions,"
    " and machine-learning training is not among them.",
    "- **Evaluation only (does not influence the weights)**: JSUT basic5000, with reference"
    " boundaries from automatic forced alignment.",
)


def _vowel_detector_metadata() -> CardMetadata:
    return CardMetadata(
        pipeline_tag=VOWEL_DETECTOR_PIPELINE_TAG,
        base_model=(VOWEL_DETECTOR_UPSTREAM,),
        # No `base_model_relation` — this moves only the container without changing the storage
        # (still f32), so it is none of adapter / merge / quantized / finetune (CardMetadata's doc).
        license=VOWEL_DETECTOR_LICENSE,
        tags=(VOWEL_DETECTOR_PIPELINE_TAG, "lip-sync", "japanese", "webgpu"),
    )


#: Hop of the input frames (10 ms grid). It is part of the feature-extraction contract itself and is
#: not in `pipelineConfig` (a number that cannot be chosen at run time is not declared for show —
#: the decision of `src/vowel-detector/config.ts`). It is needed here **only to state durations in
#: seconds**.
#:
#: The assembly stage checks that it matches the upstream config's `hop`
#: ({@link vowel_detector.distribution.vowel_detector_pipeline_config}) — without that check, the
#: day upstream changes the hop, the card alone would silently state the wrong number of seconds.
VOWEL_DETECTOR_HOP = 160


def _vowel_detector_seconds(config: Mapping[str, Any], frames: int) -> str:
    """Frame count → seconds (derived from `sampleRate` — seconds are not held as a declaration)."""
    per_second = config["sampleRate"] / VOWEL_DETECTOR_HOP
    return f"{frames / per_second:.1f}"


def _vowel_detector_milliseconds(config: Mapping[str, Any], frames: int) -> str:
    """Frame count → milliseconds (the lower bound is under one second, so seconds with one decimal
    place would collapse it to 0.0)."""
    per_second = config["sampleRate"] / VOWEL_DETECTOR_HOP
    return f"{frames * 1000 / per_second:g}"


def _vowel_detector_overview(manifest: Mapping[str, Any]) -> list[str]:
    config = default_model(manifest)["pipelineConfig"]
    shortest = _vowel_detector_milliseconds(config, config["minFrames"])
    longest = _vowel_detector_seconds(config, config["maxFrames"])
    return [
        "## What is this",
        "",
        "A small Japanese **vowel-sequence detector for lip-sync**, converted into the WebGPU",
        "inference runtime **Karume**'s container format (a `.krm` part sequence whose first part",
        "carries the graph and model descriptors). Runs as-is in the browser and in Deno.",
        "",
        "- Audio in, a **`.lab` timeline** out: `start end label` lines over the 7 lip-sync"
        " classes (`a` / `i` / `u` / `e` / `o` / `N` / `pau`), on a 20 ms grid.",
        "- **Feature extraction and post-processing are included.** The pipeline computes the"
        f" {config['featureDim']}-dimensional features (log-mel + DSP) with the mel filterbank"
        " shipped here, then runs Viterbi smoothing, short-run merging and consonant absorption"
        " on the logits.",
        "- **Decoding and resampling are yours.** The entry point is a"
        f" `Float32Array` of {config['sampleRate']} Hz mono samples; this repository ships no"
        " WAV parser and no resampler.",
        f"- **One graph for every length from {shortest} ms to {longest} s.** The time axis is"
        " symbolic, so any clip in that range runs through the same graph with no padding and no"
        f" length buckets. Audio longer than {longest} s is rejected rather than silently"
        f" truncated — split it and call twice. Audio shorter than {shortest} ms is rejected"
        " rather than padded.",
        "- Not readable by transformers (it's a different container with an embedded graph); the"
        f" reader is a pipeline that implements `{VOWEL_DETECTOR_SUPPORTED_PIPELINE}`.",
        f"- Exporter used for the conversion: `{manifest['generator']}`. The distribution manifest"
        f" is `karume.json` (`{manifest['format']}`).",
    ]


def _vowel_detector_base_weights() -> list[str]:
    """The attribution section. The storage is unchanged, so it claims only to be a conversion."""
    return [
        "## Base weights and attribution",
        "",
        f"- **Weights**: [{VOWEL_DETECTOR_UPSTREAM}]"
        f"(https://huggingface.co/{VOWEL_DETECTOR_UPSTREAM}), licensed"
        f" **{VOWEL_DETECTOR_LICENSE}** — the same trained CRNN this repository re-exports.",
        "- **Architecture**: two 1-D convolutions (the second halves the time axis) followed by a",
        "  2-layer bidirectional GRU and a linear head. Bidirectional means the model is",
        "  **offline** — it needs the whole utterance, not a stream.",
        "- **Changes made here**: conversion into the Karume container format. No retraining, no",
        "  fine-tuning and **no quantization** — the weights are the source checkpoint's own f32",
        "  values, byte for byte. The graph is the upstream `forward`, with the GRU expressed as",
        "  one scan node per layer and direction so that the time axis stays symbolic (bit-exact",
        "  against eager `nn.GRU`, checked on every export).",
        "",
        "The upstream project asks that the attributions below travel with the model, so they are",
        "reproduced here in full.",
        "",
        *VOWEL_DETECTOR_ATTRIBUTION,
    ]


def _vowel_detector_usage(manifest: Mapping[str, Any], repo: str) -> list[str]:
    """Policy of the Usage example: keep the minimal working form live, and **list the options a
    typical use case is likely to need alongside it, commented out** (the selectable values are
    listed in a comment on the same line — derived mechanically from the manifest, so the list
    follows when models / quants are added). The reader takes the next step just by uncommenting.

    `detect()` takes only one waveform (there is no run-time field to choose, post-processing
    thresholds included — the public surface of `vowelDetector.Pipeline`), so this pipeline's only
    optional knobs are model and quant.

    NOTE: `fromAssets` is **not advertised** (ruling of 2026-08-29). The split distribution form can
    be read as well now (X2-101), but that surface is meant for local debugging and assumes the
    caller holds the bytes; the ordinary entry point for a reader using HF is `fromPretrained`.
    Listing both would leave the reader to decide which one to use.
    """
    model_name = manifest["defaultModel"]
    model = default_model(manifest)
    config = model["pipelineConfig"]
    model_names = " / ".join(sorted(manifest["models"]))
    quant_names = " / ".join(sorted(model["quants"]))
    return [
        "## Usage",
        "",
        "```ts",
        'import { Pipeline } from "jsr:@karume/models/vowel-detector";',
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
        f"// {config['sampleRate']} Hz mono samples in [-1, 1]. Decoding and resampling are the",
        "// caller's job (`decodeWav` from the same package reads a WAV, but never resamples).",
        "const { segments, lab } = await pipeline.detect(samples);",
        "",
        "// lab is the ready-to-write file body; segments is the same timeline as objects:",
        '// [{ start: 0.04, end: 0.4, label: "a" }, ...]',
        'await Deno.writeTextFile("voice.lab", lab);',
        "```",
        "",
        "`detect()` builds a GPU session per call and tears it down afterwards; concurrent calls",
        "are queued rather than run side by side.",
        "Weights are fetched once and cached (verified against `karume.json`'s `size` /",
        "`sha256`).",
    ]


def _vowel_detector_shape(model: Mapping[str, Any]) -> list[str]:
    """Input, output and operating limits (what the user passes in and gets back, in one place)."""
    config = model["pipelineConfig"]
    floor = config["minFrames"]
    limit = config["maxFrames"]
    return [
        "### Input, output and limits",
        "",
        "The feature contract comes from the upstream feature configuration; the shapes are the",
        "input and output declarations of the exported graph, checked against it when this",
        "repository was assembled.",
        "",
        f"- **input**: `Float32Array`, {config['sampleRate']} Hz, mono, samples in [-1, 1].",
        f"- **features** (computed for you): {config['featureDim']} dimensions per 10 ms frame —"
        " log-mel normalized per utterance, plus voicing, log-energy and zero-crossing rate.",
        f"- **classes**: {', '.join(f'`{name}`' for name in config['classes'])} — in this order"
        " (the order *is* the class id). `cons` is absorbed into the neighbouring vowel during",
        "  post-processing, so it never reaches the `.lab`.",
        f"- **length**: from **{floor} frames** ({_vowel_detector_milliseconds(config, floor)} ms)"
        f" up to **{limit} frames of 10 ms**"
        f" ({_vowel_detector_seconds(config, limit)} s). The clip runs at its own length — there",
        "  is no padding and no bucketing, so the numbers do not depend on how long the clip is.",
        "  An odd number of frames drops the last one (the output grid is 20 ms).",
        "- **output**: one `.lab` line per run of frames, at 20 ms resolution.",
        "",
        "The limits are the symbolic range the graph was exported with, not a property of the",
        "weights: outside it the pipeline refuses the clip instead of padding or truncating it.",
        "Split longer recordings and call `detect` per part.",
    ]


def render_vowel_detector_model_card(
    manifest: Mapping[str, Any], repo: str, host_assets: Mapping[FileRefKey, int] = {}
) -> str:
    """Assemble the `README.md` body of the vowel-detection distribution (a pure function, with a
    trailing newline)."""
    require_pipeline(manifest, VOWEL_DETECTOR_SUPPORTED_PIPELINE)
    return render(
        (
            frontmatter(_vowel_detector_metadata()),
            # The one card without the house non-endorsement sentence
            # (`karume.modelcard.introduction`): its upstream is our own release (`hdae/`), and
            # the sentence would disclaim ourselves (`_shared.upstream.hf_upstreams` refuses it).
            ["", f"# {VOWEL_DETECTOR_TITLE}", ""],
            _vowel_detector_overview(manifest),
            [""],
            _vowel_detector_base_weights(),
            [""],
            models(manifest),
            [""],
            _vowel_detector_usage(manifest, repo),
            *model_sections(
                manifest, (partial(quants, host_assets=host_assets), _vowel_detector_shape)
            ),
        )
    )
