"""The model card (`README.md`) of the EmbeddingGemma 2 distribution — a pure function derived by
machine from the manifest.

SPEC: docs/spec/distribution-cards.md#the-model-card

The general-purpose rendering parts (frontmatter, model list, quant table, assembling the
sections) live in `karume.modelcard`. What lives here is **the facts specific to EmbeddingGemma 2**
only: attribution (source, license) and what this pipeline's card says.

Attribution is **determined by the model name alone** ({@link EMBEDDINGGEMMA2_UPSTREAM}), so it is
not split into a profile axis (the gemma4 / siglip2 rule). The files with the redistribution terms
(`LICENSE.md` / `NOTICE.md`) are placed at the repository root by
`embeddinggemma2.distribution.embeddinggemma2_root_files`, not by the card. What the model can and
cannot do is left to the upstream model card, which the card links.

MUST: **every number, download size, quant table and dtype label is derived from the manifest**
(the same MUST of `karume.modelcard` applies unchanged). The only constants this module may hold
are facts that **do not exist in the manifest** (the upstream's context window and its statement on
precision, what each storage form holds, and the measured distance of the int8 form from the
original weights with its source).
"""

from __future__ import annotations

import math
import textwrap
from collections.abc import Mapping
from functools import partial
from typing import Any

from _shared.upstream import GEMMA_LICENSE_PAGE, hf_upstreams
from embeddinggemma2.sources import PIN
from karume.modelcard import (
    RUNTIME_SENTENCE,
    CardMetadata,
    FileRefKey,
    default_model,
    from_pretrained,
    frontmatter,
    introduction,
    model_sections,
    models,
    pinned,
    quants,
    render,
    require_pipeline,
)

#: The pipeline contract this template can describe (ADR 0041 §2 — per model). The TypeScript family
#: `embeddinggemma` reads it.
EMBEDDINGGEMMA2_SUPPORTED_PIPELINE = "embeddinggemma/1"

#: The HF pipeline tag: the upstream card's own (`feature-extraction`) — a text, an image or an
#: audio embedding is feature extraction too.
EMBEDDINGGEMMA2_PIPELINE_TAG = "feature-extraction"

#: The title names only what this distribution carries: the text tower, and each tower the
#: manifest ships ({@link _tower}) — never video.
EMBEDDINGGEMMA2_TITLE = "EmbeddingGemma 2 Text Embedding — Karume"
EMBEDDINGGEMMA2_VISION_TITLE = "EmbeddingGemma 2 Text and Image Embedding — Karume"
EMBEDDINGGEMMA2_AUDIO_TITLE = "EmbeddingGemma 2 Text and Audio Embedding — Karume"
EMBEDDINGGEMMA2_VISION_AUDIO_TITLE = "EmbeddingGemma 2 Text, Image and Audio Embedding — Karume"

#: The weights roles of the two towers (the spellings `embeddinggemma2.distribution` writes).
EMBEDDINGGEMMA2_VISION_ROLE = "vision"
EMBEDDINGGEMMA2_AUDIO_ROLE = "audio"

#: Weights role of a tower → dtype label → what that storage of the tower holds. MUST: every
#: preset maps a tower to the storage it gives the text tower, among these — the card states it
#: that way ({@link _tower}, {@link _tower_storage_lines}).
EMBEDDINGGEMMA2_TOWER_STORAGES: Mapping[str, Mapping[str, str]] = {
    EMBEDDINGGEMMA2_VISION_ROLE: {
        "bf16": "the checkpoint's weights unchanged",
        "i8": "every linear layer and the two learned position tables rounded to int8 with one"
        " f32 scale per row",
    },
    EMBEDDINGGEMMA2_AUDIO_ROLE: {
        "bf16": "the checkpoint's weights unchanged",
        "i8": "every linear layer and depthwise convolution rounded to int8 with one f32 scale"
        " per row",
    },
}

#: What stays f32 in every storage of a tower.
EMBEDDINGGEMMA2_TOWER_F32: Mapping[str, str] = {
    EMBEDDINGGEMMA2_VISION_ROLE: "its norms and its folded rotary tables",
    EMBEDDINGGEMMA2_AUDIO_ROLE: "its norms, biases and folded constants",
}

#: The audio tower's exception in `i8` (`export_audio.I8_KEPT_BF16_WEIGHTS` declares it, with the
#: torch-CPU screening of 2026-11-23 as its reason) — one sentence, only beside an `i8` storage.
EMBEDDINGGEMMA2_AUDIO_I8_EXCEPTION = (
    "In `i8` its two subsampling convolutions keep their bf16 values: rounded, they alone moved"
    " single soft tokens of speech far from the original."
)

#: The audio one soft token stands for, in milliseconds (upstream's `processor_config.json`,
#: `audio_ms_per_token`: four mel frames of a 10 ms hop) — a fact of the front end, not of the
#: manifest; the card turns `pipelineConfig.audio.maxSoftTokens` into seconds with it.
EMBEDDINGGEMMA2_AUDIO_MS_PER_TOKEN = 40

#: Model name → the HF repository ID of the upstream checkpoint. **This one table is the only record
#: of "which model is which weights"**; a model name it does not list cannot be attributed, so the
#: card fails instead of rendering. The one model is the text tower (`text`); later keys of the
#: same repository would add the other towers of the same checkpoint.
EMBEDDINGGEMMA2_UPSTREAM: Mapping[str, str] = {"text": "google/embeddinggemma-2"}

#: License (the README front matter of the pinned snapshot says `license: apache-2.0` —
#: `embeddinggemma2.sources.PIN`; not gated).
EMBEDDINGGEMMA2_LICENSE = "apache-2.0"
EMBEDDINGGEMMA2_LICENSE_TEXT_LINK = "https://www.apache.org/licenses/LICENSE-2.0"

#: The sentence of the upstream card (its "Ethics & Safety" section, at the pinned revision) on the
#: use policy, **verbatim, with upstream's own link**. The card and `NOTICE.md` quote this one
#: value: the sentence is upstream's statement, so it is quoted and not paraphrased, and the policy
#: is upstream's document, which can change, so it is linked and not copied.
EMBEDDINGGEMMA2_USE_POLICY_SENTENCE = (
    "Deployments must adhere to the"
    " [Gemma Prohibited Use Policy](https://ai.google.dev/gemma/prohibited_use_policy)."
)

#: The storage of the upstream checkpoint's weights (BF16). The card states that the default preset
#: ships exactly this one ({@link _storage_lines}).
EMBEDDINGGEMMA2_CHECKPOINT_DTYPE = "bf16"

#: The int8 storage's dtype label (the runtime's storage vocabulary).
EMBEDDINGGEMMA2_I8_DTYPE = "i8"

#: Dtype label of the `model` role → (the word the precision line uses, what that storage holds).
#: MUST: the card describes these storages only — a manifest that stores the text tower otherwise
#: is not drawn ({@link _storage_lines}).
EMBEDDINGGEMMA2_STORAGES: Mapping[str, tuple[str, str]] = {
    EMBEDDINGGEMMA2_CHECKPOINT_DTYPE: ("bf16", "the checkpoint's weights unchanged (bf16)."),
    EMBEDDINGGEMMA2_I8_DTYPE: (
        "int8",
        "the token table and every linear layer rounded to int8, one f32 scale per row; the"
        " norms and the per-layer scalars are the checkpoint's values, as in `bf16`.",
    ),
}

#: Where the int8 form's measured distance from the original weights comes from (a record of the
#: Karume repository).
EMBEDDINGGEMMA2_I8_SCREENING_SOURCE = "docs/research/2026-11-22-embeddinggemma2-quant-screen.md"

#: The measured distance of the int8 form from the original weights (the screening above,
#: configuration `i8-all` — the rounding the `i8` files store; numbers copied from its tables).
#: MUST: stated as one measurement on that small set, never as a benchmark, and only for a manifest
#: with an `i8` preset ({@link _i8_screening}).
EMBEDDINGGEMMA2_I8_SCREENING: tuple[str, ...] = (
    "One measurement, not a benchmark: a screening on torch (CPU, f32 arithmetic) with the text",
    "tower's weights rounded exactly as the `i8` files store them, against the checkpoint's own",
    "weights, over 58 texts — the ten test texts of this conversion (12 to 8,192 tokens) and 48",
    "short ones, 10 of them in Japanese (2026-11-22). At the full 768 values:",
    "",
    "- the cosine of each text's embedding to the original weights' is 0.999396 at the lowest and",
    "  0.999842 on average;",
    "- ranking 20 queries against 20 documents, every query still ranks its own document first,",
    "  and 0.74% of the 3,800 (query, document, document) orders are flipped;",
    "- the shorter widths are no further off (lowest cosine 0.999411 / 0.999474 / 0.999620 at",
    "  512 / 256 / 128 values).",
    "",
    f"Source: `{EMBEDDINGGEMMA2_I8_SCREENING_SOURCE}` in the Karume repository.",
)

#: Where the towers' int8 forms' measured distance comes from (a record of the Karume repository).
EMBEDDINGGEMMA2_I8_TOWER_SCREENING_SOURCE = (
    "docs/research/2026-11-23-embeddinggemma2-tower-quant-screen.md"
)

#: The towers' measured distance (the screening above, configuration `all-i8` on its quality set —
#: every tower an input passes through rounded as the `i8` files store it; numbers copied from its
#: re-run with the shipped form). One bullet per tower, drawn only for a tower the manifest ships
#: with an `i8` storage ({@link _i8_screening}).
EMBEDDINGGEMMA2_I8_TOWER_SCREENING_LEAD: tuple[str, ...] = (
    "The towers' int8 weights were screened the same way (2026-11-23): each input embedded alone,",
    "every tower it passes through rounded as the `i8` files store it, at the full 768 values:",
    "",
)
EMBEDDINGGEMMA2_I8_TOWER_SCREENING: Mapping[str, tuple[str, ...]] = {
    EMBEDDINGGEMMA2_VISION_ROLE: (
        "- 32 generated images: the cosine to the original weights' embedding is 0.999648 at the",
        "  lowest and 0.999812 on average.",
    ),
    EMBEDDINGGEMMA2_AUDIO_ROLE: (
        "- 32 clips of Japanese synthetic speech (8 sentences): 0.999493 at the lowest and",
        "  0.999759 on average.",
    ),
}

#: The weights role of the text tower (the spelling `embeddinggemma2.distribution` imports — the
#: card reads the storage of the main model from it).
EMBEDDINGGEMMA2_MODEL_ROLE = "model"

#: The upstream model's context window in tokens (upstream model card at the pinned revision:
#: "Context Window | 8,192 tokens"). A fact of the upstream, not of this distribution — the
#: distribution's own limit is `pipelineConfig.maxTokens`.
EMBEDDINGGEMMA2_UPSTREAM_CONTEXT_TOKENS = 8192

#: The upstream model card's statement on precision (pinned revision, "Run inference in `bfloat16`
#: or `float32`. Do not use `float16`." — its activations exceed float16's range). The card pairs
#: it with what this repository's presets compute in ({@link _precision_lines}).
EMBEDDINGGEMMA2_PRECISION_STATEMENT = (
    "upstream says to run the model in bfloat16 or float32, never float16: its activations exceed"
    " float16's range, and float16 returns NaN or silently degraded embeddings"
)


def _revision(repo: str) -> str:
    """The commit the stored weights of `repo` were exported from (`embeddinggemma2.sources.PIN` —
    every container's provenance is checked against it at assembly)."""
    if repo != PIN.repo:
        raise ValueError(f"{repo} is not the pinned source {PIN.repo} — no commit to name")
    return PIN.revision


def _metadata(manifest: Mapping[str, Any]) -> CardMetadata:
    """The frontmatter, from the models the manifest lists (`base_model` = every redistributed
    upstream).

    MUST: do not render for an unknown model name — dropping it silently would emit a card that
    does not name its source.
    """
    upstream: list[str] = []
    for name in manifest["models"]:
        if name not in EMBEDDINGGEMMA2_UPSTREAM:
            raise ValueError(
                f"model '{name}' has no upstream in the attribution table"
                f" (known: {sorted(EMBEDDINGGEMMA2_UPSTREAM)}) — a card that cannot name its"
                " source is not drawn"
            )
        repo = EMBEDDINGGEMMA2_UPSTREAM[name]
        if repo not in upstream:
            upstream.append(repo)
    return CardMetadata(
        pipeline_tag=EMBEDDINGGEMMA2_PIPELINE_TAG,
        base_model=tuple(upstream),
        # No `base_model_relation`: the default preset's download keeps the checkpoint's storage
        # (bf16), so the distribution is none of adapter / merge / quantized / finetune — the
        # "default quant's downloaded file" rule of the other cards (birefnet, irodori, sbv2,
        # anima); the `i8` preset is an option beside it ({@link _storage_lines} refuses a
        # manifest where that is not so). The gemma4 card follows another rule: `quantized` as
        # soon as any stored weight differs from the checkpoint's, default or not
        # (`_gemma4_relation`).
        license=EMBEDDINGGEMMA2_LICENSE,
        # The license page the upstream card links beside its Apache 2.0 (its header names it, as
        # Gemma 4's cards do — `_shared.upstream.GEMMA_LICENSE_PAGE`).
        license_link=GEMMA_LICENSE_PAGE,
        tags=(EMBEDDINGGEMMA2_PIPELINE_TAG, "sentence-similarity", "webgpu", "embedding", "gemma"),
    )


def _spell(values: Any) -> str:
    return " / ".join(str(value) for value in values)


def _input_length(max_tokens: int) -> list[str]:
    """The input-length line: the distribution's limit, against the upstream model's context.

    MUST: a limit above the upstream context is not drawn (the card would promise lengths the
    model was not trained for); "for now" is said only while the limit is below it.
    """
    if max_tokens > EMBEDDINGGEMMA2_UPSTREAM_CONTEXT_TOKENS:
        raise ValueError(
            f"maxTokens {max_tokens} is above the upstream model's context of"
            f" {EMBEDDINGGEMMA2_UPSTREAM_CONTEXT_TOKENS} tokens"
        )
    if max_tokens == EMBEDDINGGEMMA2_UPSTREAM_CONTEXT_TOKENS:
        return [
            f"- **Input length**: at most {max_tokens:,} tokens including `<bos>` and `<eos>` (the",
            "  upstream model's whole context). A longer text is refused, never truncated.",
        ]
    return [
        f"- **Input length**: at most {max_tokens:,} tokens including `<bos>` and `<eos>` for now",
        f"  (the upstream model's context is {EMBEDDINGGEMMA2_UPSTREAM_CONTEXT_TOKENS:,} tokens). A"
        " longer text is refused, never",
        "  truncated.",
    ]


def _tower(manifest: Mapping[str, Any], role: str) -> Mapping[str, Any] | None:
    """The default model's `pipelineConfig.<role>` when the manifest ships the tower of that
    weights role (`vision` / `audio` — the declaration has the role's name), else `None`.

    MUST: the tower's weights and the declaration come together, on every model; the tower is
    stored in exactly the text tower's storages, each one the card describes, and every preset
    maps it to the storage it gives the text tower — the card states it that way, so a manifest
    where that is not so is not drawn.
    """
    shipped = {name: role in model["weights"] for name, model in manifest["models"].items()}
    for name, model in manifest["models"].items():
        declared = role in model["pipelineConfig"]
        if declared != shipped[name]:
            raise ValueError(
                f"model '{name}': the {role} weights are {'' if shipped[name] else 'not '}shipped"
                f" but pipelineConfig.{role} is {'' if declared else 'not '}declared"
            )
        if not shipped[name]:
            continue
        storages = set(model["weights"][role])
        text = set(model["weights"][EMBEDDINGGEMMA2_MODEL_ROLE])
        mismatched = sorted(
            quant_name
            for quant_name, quant in model["quants"].items()
            if quant["weights"][role] != quant["weights"][EMBEDDINGGEMMA2_MODEL_ROLE]
        )
        if storages != text or not storages <= set(EMBEDDINGGEMMA2_TOWER_STORAGES[role]):
            raise ValueError(
                f"model '{name}': the {role} tower is stored as {sorted(storages)}, the text"
                f" tower as {sorted(text)} — this card states that the tower ships in the"
                " storage each preset gives the text tower"
            )
        if mismatched:
            raise ValueError(
                f"model '{name}': the quants {mismatched} map the {role} tower to another storage"
                " than the text tower — this card states that the tower ships in the storage"
                " each preset gives the text tower"
            )
    if len(set(shipped.values())) > 1:
        raise ValueError(f"the {role} tower ships with some models only ({shipped})")
    return default_model(manifest)["pipelineConfig"].get(role)


def _vision(manifest: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """The default model's `pipelineConfig.vision` when the manifest ships the vision tower."""
    return _tower(manifest, EMBEDDINGGEMMA2_VISION_ROLE)


def _audio(manifest: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """The default model's `pipelineConfig.audio` when the manifest ships the audio tower."""
    return _tower(manifest, EMBEDDINGGEMMA2_AUDIO_ROLE)


def _title(manifest: Mapping[str, Any]) -> str:
    """The card's title, by the towers the manifest ships."""
    vision, audio = _vision(manifest) is not None, _audio(manifest) is not None
    if vision and audio:
        return EMBEDDINGGEMMA2_VISION_AUDIO_TITLE
    if audio:
        return EMBEDDINGGEMMA2_AUDIO_TITLE
    return EMBEDDINGGEMMA2_VISION_TITLE if vision else EMBEDDINGGEMMA2_TITLE


def _audio_seconds(audio: Mapping[str, Any]) -> str:
    """The longest audio, in seconds, from `pipelineConfig.audio.maxSoftTokens`."""
    return f"{audio['maxSoftTokens'] * EMBEDDINGGEMMA2_AUDIO_MS_PER_TOKEN / 1000:g}"


#: What the overview says of the scope of a manifest without the vision tower.
_TEXT_ONLY_SCOPE: tuple[str, ...] = (
    "- **Text only.** The upstream checkpoint also carries vision and audio towers; this",
    "  distribution contains neither, and cannot embed images, video or audio. A text holding",
    "  the image, audio or video placeholder token is refused.",
)


def _image_scope(budget: int) -> list[str]:
    """What the overview says of images, for a manifest with the vision tower at `budget` soft
    tokens (a square image resizes to `isqrt(budget)²` of them — the resize rule's `floor` of
    `sqrt(budget)` blocks a side)."""
    return [
        "- **Text and images.** Built with `vision: true`, the pipeline also fetches the",
        "  vision tower and takes image parts (decoded RGB8 pixels of any size) beside the text",
        "  parts, in the order given. An image is resized to keep its aspect and becomes up to",
        f"  {budget} soft tokens (a square one {math.isqrt(budget) ** 2}). The audio tower is not",
        "  carried: neither video nor audio can be embedded. A text holding the image, audio or",
        "  video placeholder token is refused.",
    ]


def _audio_scope(audio: Mapping[str, Any]) -> list[str]:
    """What the overview says of audio, for a manifest with the audio tower and without the vision
    tower."""
    rate = f"{audio['sampleRate']:,}"
    per_token = EMBEDDINGGEMMA2_AUDIO_MS_PER_TOKEN
    longest = f"{audio['maxSoftTokens']} ({_audio_seconds(audio)} s)"
    return [
        "- **Text and audio.** Built with `audio: true`, the pipeline also fetches the audio",
        f"  tower and takes audio parts (mono f32 samples at {rate} Hz) beside the text parts,",
        f"  in the order given. An audio becomes one soft token per {per_token} ms, up to",
        f"  {longest}; a longer one is refused, never truncated, and so is another sample",
        "  rate (resample first). The vision tower is not carried: neither images nor video can",
        "  be embedded. A text holding the image, audio or video placeholder token is refused.",
    ]


def _image_and_audio_scope(vision: Mapping[str, Any], audio: Mapping[str, Any]) -> list[str]:
    """What the overview says of images and audio, for a manifest with both towers."""
    budget = vision["softTokenBudget"]
    square = math.isqrt(budget) ** 2
    rate = f"{audio['sampleRate']:,}"
    per_token = EMBEDDINGGEMMA2_AUDIO_MS_PER_TOKEN
    longest = f"{audio['maxSoftTokens']} ({_audio_seconds(audio)} s)"
    return [
        "- **Text, images and audio.** Built with `vision: true`, the pipeline also fetches the",
        "  vision tower and takes image parts (decoded RGB8 pixels of any size); built with",
        "  `audio: true`, it fetches the audio tower and takes audio parts (mono f32 samples at",
        f"  {rate} Hz). Both may be on, and the parts go beside the text parts in the order",
        "  given. An image is resized to keep its aspect and becomes up to",
        f"  {budget} soft tokens (a square one {square}); an audio becomes one soft token per",
        f"  {per_token} ms, up to {longest} — a longer audio or another sample rate is refused,",
        "  never truncated or resampled. Video cannot be embedded. A text holding the image,",
        "  audio or video placeholder token is refused.",
    ]


def _overview(manifest: Mapping[str, Any]) -> list[str]:
    config = default_model(manifest)["pipelineConfig"]
    dimensions = config["dimensions"]
    vision = _vision(manifest)
    audio = _audio(manifest)
    upstream = EMBEDDINGGEMMA2_UPSTREAM[manifest["defaultModel"]]
    if vision is None and audio is None:
        opening = [
            f"A text embedding distribution: the **text tower** of `{upstream}`, with its mean"
            " pooling and",
        ]
        runs = "A string goes in and a unit vector comes out."
        graphs = "  of the prompt you name, tokenizes on the host and runs one graph."
        scope = list(_TEXT_ONLY_SCOPE)
    elif audio is None:
        opening = [
            "A text and image embedding distribution: the **text and vision towers** of",
            f"`{upstream}`, with the mean pooling and",
        ]
        runs = "A text, an image or both go in and a unit vector comes out."
        graphs = (
            "  of the prompt you name, tokenizes on the host, runs the vision tower once per image"
            " and the text graph once."
        )
        scope = _image_scope(vision["softTokenBudget"])
    elif vision is None:
        opening = [
            "A text and audio embedding distribution: the **text and audio towers** of",
            f"`{upstream}`, with the mean pooling and",
        ]
        runs = "A text, an audio or both go in and a unit vector comes out."
        graphs = (
            "  of the prompt you name, tokenizes and computes the log-mel frames on the host, runs"
            " the audio tower once per audio and the text graph once."
        )
        scope = _audio_scope(audio)
    else:
        opening = [
            "A text, image and audio embedding distribution: the **text, vision and audio towers**",
            f"of `{upstream}`, with the mean pooling and",
        ]
        runs = "Text, images and audio go in, in any mix, and a unit vector comes out."
        graphs = (
            "  of the prompt you name, tokenizes (and computes an audio's log-mel frames) on the"
            " host, runs a tower once per image or audio and the text graph once."
        )
        scope = _image_and_audio_scope(vision, audio)
    return [
        "## What is this",
        "",
        *opening,
        "L2 normalisation, converted into the WebGPU inference runtime **Karume**'s container",
        "format (a `.krm` part sequence whose first part carries the graph and model descriptors).",
        RUNTIME_SENTENCE,
        runs,
        "",
        "- **The tokenizer and the prompts ship with the weights.** `embed()` prepends the prefix",
        graphs,
        f"- **Output**: a unit vector of {dimensions[0]} values. The model was trained with",
        "  Matryoshka Representation Learning, so a shorter vector of"
        f" {_spell(dimensions[1:])} values can be asked for:",
        "  the leading values, re-normalised to unit length.",
        *_input_length(config["maxTokens"]),
        *scope,
        "- Not readable by transformers or sentence-transformers (it's a different container with",
        "  an embedded graph); the reader is a pipeline that implements"
        f" `{EMBEDDINGGEMMA2_SUPPORTED_PIPELINE}`.",
        f"- Exporter used for the conversion: `{manifest['generator']}`. The distribution manifest",
        f"  is `karume.json` (`{manifest['format']}`).",
    ]


def _stored_dtypes(manifest: Mapping[str, Any]) -> list[str]:
    """The storages of the `model` role, in the order {@link EMBEDDINGGEMMA2_STORAGES} lists them.

    MUST: a manifest that stores the text tower in a storage the card does not describe, or
    without the checkpoint's own, is not drawn — the card would state a storage the distribution
    does not carry.
    """
    dtypes = {
        dtype
        for model in manifest["models"].values()
        for dtype in model["weights"][EMBEDDINGGEMMA2_MODEL_ROLE]
    }
    if EMBEDDINGGEMMA2_CHECKPOINT_DTYPE not in dtypes or not dtypes <= set(
        EMBEDDINGGEMMA2_STORAGES
    ):
        raise ValueError(
            f"the text tower's weights are stored as {sorted(dtypes)} — this card describes the"
            f" checkpoint's {EMBEDDINGGEMMA2_CHECKPOINT_DTYPE} values, optionally beside"
            " "
            + " / ".join(
                dtype
                for dtype in EMBEDDINGGEMMA2_STORAGES
                if dtype != EMBEDDINGGEMMA2_CHECKPOINT_DTYPE
            )
        )
    return [dtype for dtype in EMBEDDINGGEMMA2_STORAGES if dtype in dtypes]


def _storage_lines(manifest: Mapping[str, Any]) -> list[str]:
    """The attribution's opening: the original checkpoint is not shipped, and what each storage of
    the weights holds.

    MUST: a manifest whose default preset does not ship the checkpoint's own values is not drawn —
    the card carries no `base_model_relation` on that ground ({@link _metadata}).
    """
    dtypes = _stored_dtypes(manifest)
    for name, model in manifest["models"].items():
        default = model["quants"][model["defaultQuant"]]["weights"][EMBEDDINGGEMMA2_MODEL_ROLE]
        if default != EMBEDDINGGEMMA2_CHECKPOINT_DTYPE:
            raise ValueError(
                f"model '{name}': the default quant '{model['defaultQuant']}' stores the text"
                f" tower as {default} — this card states that the default download is the"
                f" checkpoint's {EMBEDDINGGEMMA2_CHECKPOINT_DTYPE} values"
            )
    return [
        "Converted from the upstream checkpoint — the original checkpoint files are not",
        "distributed here. The weights of the text tower ship in the storage each quant names:",
        "",
        *(f"- `{dtype}`: {EMBEDDINGGEMMA2_STORAGES[dtype][1]}" for dtype in dtypes),
        *_tower_storage_lines(manifest),
    ]


def _tower_storage_lines(manifest: Mapping[str, Any]) -> list[str]:
    """What each tower holds in each storage (a paragraph per tower after the storage list — none
    for a manifest without towers). The storages are checked by {@link _tower}; they are the text
    tower's ({@link _stored_dtypes})."""
    dtypes = _stored_dtypes(manifest)
    lines: list[str] = []
    for role, tower, option in (
        (EMBEDDINGGEMMA2_VISION_ROLE, _vision(manifest), "vision"),
        (EMBEDDINGGEMMA2_AUDIO_ROLE, _audio(manifest), "audio"),
    ):
        if tower is None:
            continue
        holds = EMBEDDINGGEMMA2_TOWER_STORAGES[role]
        if len(dtypes) == 1:
            stored = f"ships as {holds[dtypes[0]]} ({dtypes[0]})"
        else:
            stored = "ships in the storage each preset gives the text tower: " + ", ".join(
                f"in `{dtype}` {holds[dtype]}" for dtype in dtypes
            )
        exception = (
            [EMBEDDINGGEMMA2_AUDIO_I8_EXCEPTION]
            if role == EMBEDDINGGEMMA2_AUDIO_ROLE and EMBEDDINGGEMMA2_I8_DTYPE in dtypes
            else []
        )
        f32 = EMBEDDINGGEMMA2_TOWER_F32[role]
        paragraph = " ".join(
            [
                f"The {role} tower (`{role}`) {stored}; {f32} stay f32.",
                *exception,
                f"A pipeline built without `{option}: true` does not fetch it (the Download column"
                " below counts it).",
            ]
        )
        lines += ["", *textwrap.wrap(paragraph, 96, break_long_words=False, break_on_hyphens=False)]
    return lines


def _precision_lines(manifest: Mapping[str, Any]) -> list[str]:
    """Upstream's statement on precision, and what this repository's presets compute in.

    MUST: a preset with a `session` of its own is not drawn — every preset here is the reference
    one (f32 compute, `session` empty — ADR 0104), and a session option could change that, making
    the sentence false.
    """
    for name, model in manifest["models"].items():
        for quant_name, quant in model["quants"].items():
            if quant["session"]:
                raise ValueError(
                    f"model '{name}', quant '{quant_name}' declares a session"
                    f" {quant['session']} — this card states that every preset computes in f32"
                )
    stored = " or ".join(EMBEDDINGGEMMA2_STORAGES[dtype][0] for dtype in _stored_dtypes(manifest))
    return [
        f"- **Precision**: {EMBEDDINGGEMMA2_PRECISION_STATEMENT}. Every preset of this repository",
        f"  computes in f32 over the stored {stored} weights.",
    ]


def _base_weights(manifest: Mapping[str, Any]) -> list[str]:
    """The attribution section. Use (capabilities, limitations, safety, evaluation) is **left to
    the upstream card**."""
    lines = [
        "## Base weights and attribution",
        "",
        *_storage_lines(manifest),
        "",
        # A paragraph between the two lists: CommonMark joins two adjacent lists of one marker.
        "The upstream model, its licence and what was changed:",
        "",
    ]
    for name in manifest["models"]:
        repo = EMBEDDINGGEMMA2_UPSTREAM[name]
        lines.append(
            f"- **`{name}`**: {pinned(repo, _revision(repo))}, licensed"
            f" **{EMBEDDINGGEMMA2_LICENSE}**"
            f" ([full text]({EMBEDDINGGEMMA2_LICENSE_TEXT_LINK}); a verbatim copy is in"
            " `LICENSE.md`)."
        )
    upstream = EMBEDDINGGEMMA2_UPSTREAM[manifest["defaultModel"]]
    weights_change = (
        [
            "  checkpoint's bf16 values unchanged in `bf16`, and an int8-quantized copy (`i8`) was",
            "  added.",
        ]
        if EMBEDDINGGEMMA2_I8_DTYPE in _stored_dtypes(manifest)
        else ["  checkpoint's bf16 values unchanged."]
    )
    tower_weights = (
        "its weights ship in the storage each preset gives the text tower (the checkpoint's bf16"
        " values unchanged in `bf16`, rounded to int8 in `i8`)."
        if EMBEDDINGGEMMA2_I8_DTYPE in _stored_dtypes(manifest)
        else "its weights keep the checkpoint's bf16 values unchanged in every preset."
    )
    changes = {
        EMBEDDINGGEMMA2_VISION_ROLE: "The vision tower and its projection were re-expressed as a"
        " second graph that reads an image's real patches, the patches of one soft token side by"
        " side, with its two-dimensional rotary embedding folded into tables gathered by the patch"
        " positions; ",
        EMBEDDINGGEMMA2_AUDIO_ROLE: "The audio tower and its projection were re-expressed as a"
        " graph of their own that reads an audio's log-mel frames (computed on the host) with"
        " their frame mask, its local attention computed over each query's window of 12 keys; ",
    }
    vision_change, audio_change = (
        []
        if tower is None
        else textwrap.wrap(
            changes[role] + tower_weights,
            92,
            initial_indent="  ",
            subsequent_indent="  ",
            break_long_words=False,
            break_on_hyphens=False,
        )
        for role, tower in (
            (EMBEDDINGGEMMA2_VISION_ROLE, _vision(manifest)),
            (EMBEDDINGGEMMA2_AUDIO_ROLE, _audio(manifest)),
        )
    )
    lines += [
        "- **Changes made here** (also listed in `NOTICE.md`, per Apache 2.0 §4(b)): the text",
        "  tower was extracted and re-expressed, with the mean pooling and the L2 normalisation,",
        "  as one graph in the Karume container format; the sliding-window attention mask is a",
        "  band declared on the attention, applied from the token positions; the weights keep the",
        *weights_change,
        *vision_change,
        *audio_change,
        "  No retraining and no fine-tuning.",
        *_precision_lines(manifest),
        "- **What the model can and cannot do** — capabilities, limitations, intended use,",
        "  evaluations and safety guidance — is documented on the upstream model card, and this",
        f"  repository does not restate it: read [{upstream}](https://huggingface.co/{upstream})",
        "  before using it.",
        '- **The upstream use policy**: the upstream model card states, under "Ethics & Safety":',
        "",
        f"  > {EMBEDDINGGEMMA2_USE_POLICY_SENTENCE}",
        "",
        "  The policy is upstream's document: read it at that link. It is not reproduced here.",
    ]
    return lines


#: What the usage section says of one call, without the vision tower and with it.
_TEXT_CALLS: tuple[str, ...] = (
    "`embed()` takes one text per call and returns an f32 unit vector. It keeps one GPU",
    "session alive for the lifetime of the pipeline, so embedding many texts uploads the",
    "weights once;",
    "concurrent calls are queued rather than run side by side. A prompt name the repository",
    "does not declare, a dimension it does not list or a text over the token limit is refused.",
)
_IMAGE_CALLS: tuple[str, ...] = (
    "`embed()` takes one input per call (a text, or text and image parts) and returns an",
    "f32 unit vector. It keeps its GPU sessions alive for the lifetime of the pipeline, so",
    "embedding many inputs uploads the weights once;",
    "concurrent calls are queued rather than run side by side. A prompt name the repository",
    "does not declare, a dimension it does not list, an input over the token limit or an",
    "image part on a pipeline built without `vision: true` is refused.",
)
_AUDIO_CALLS: tuple[str, ...] = (
    "`embed()` takes one input per call (a text, or text and audio parts) and returns an",
    "f32 unit vector. It keeps its GPU sessions alive for the lifetime of the pipeline, so",
    "embedding many inputs uploads the weights once;",
    "concurrent calls are queued rather than run side by side. A prompt name the repository",
    "does not declare, a dimension it does not list, an input over the token limit or an",
    "audio part on a pipeline built without `audio: true` is refused.",
)
_IMAGE_AND_AUDIO_CALLS: tuple[str, ...] = (
    "`embed()` takes one input per call (a text, or text, image and audio parts) and returns",
    "an f32 unit vector. It keeps its GPU sessions alive for the lifetime of the pipeline, so",
    "embedding many inputs uploads the weights once;",
    "concurrent calls are queued rather than run side by side. A prompt name the repository",
    "does not declare, a dimension it does not list, an input over the token limit, an image",
    "part on a pipeline built without `vision: true` or an audio part on one built without",
    "`audio: true` is refused.",
)


def _usage(manifest: Mapping[str, Any], repo: str) -> list[str]:
    """The usage example: the minimal working form (`fromPretrained` → `embed`), with the options
    one would commonly touch as comments (the selectable values derived from the manifest).

    NOTE: `fromAssets` is not shown (ruling of 2026-08-29 — the same for every family).
    """
    model_name = manifest["defaultModel"]
    model = default_model(manifest)
    config = model["pipelineConfig"]
    quant = model["defaultQuant"]
    model_names = " / ".join(sorted(manifest["models"]))
    quant_names = " / ".join(sorted(model["quants"]))
    dimensions = config["dimensions"]
    vision = _vision(manifest) is not None
    audio_config = _audio(manifest)
    audio = audio_config is not None
    vision_option = (
        ["  // vision: true, // also fetches the vision tower, for image parts"] if vision else []
    )
    audio_option = (
        ["  // audio: true, // also fetches the audio tower, for audio parts"] if audio else []
    )
    image_example = (
        [
            "",
            "// Built with `vision: true`, an image (decoded RGB8 pixels) is a content part,",
            "// beside text parts if you like:",
            "// const image = { data, width, height }; // Uint8Array of width × height × 3",
            '// const picture = await pipeline.embed([{ type: "image", image }]);',
        ]
        if vision
        else []
    )
    audio_example = []
    if audio_config is not None:
        rate = audio_config["sampleRate"]
        audio_example = [
            "",
            f"// Built with `audio: true`, an audio (mono f32 samples at {rate} Hz) is a",
            "// content part too:",
            f"// const audio = {{ data, sampleRate: {rate} }}; // a Float32Array of at most"
            f" {_audio_seconds(audio_config)} s",
            '// const clip = await pipeline.embed([{ type: "audio", audio }]);',
        ]
    if vision and audio:
        calls = _IMAGE_AND_AUDIO_CALLS
    elif audio:
        calls = _AUDIO_CALLS
    else:
        calls = _IMAGE_CALLS if vision else _TEXT_CALLS
    return [
        "## Usage",
        "",
        "```ts",
        'import { Pipeline } from "jsr:@karume/models/embeddinggemma";',
        "",
        *from_pretrained(
            "Pipeline",
            repo,
            [
                f'  // model: "{model_name}", // default — available: {model_names}',
                f'  // quant: "{quant}", // default — available: {quant_names}',
                *vision_option,
                *audio_option,
            ],
            disposable="await using",
        ),
        "",
        'const query = await pipeline.embed("how do I reset my password", { prompt: "query" });',
        "const document = await pipeline.embed(",
        '  "Open Settings, choose Account and select Reset password.",',
        '  { prompt: "document" },',
        ");",
        "",
        "// Unit vectors: their dot product is the cosine similarity.",
        "const score = query.reduce((sum, value, index) => sum + value * document[index], 0);",
        "",
        "// A shorter Matryoshka vector (queries and documents must share a dimension):",
        f'// await pipeline.embed(text, {{ prompt: "query", dimension: {dimensions[-1]} }});'
        f" // available: {_spell(dimensions)}",
        *image_example,
        *audio_example,
        "```",
        "",
        *calls,
        "Weights are fetched once and cached (verified against `karume.json`'s `size` / `sha256`).",
    ]


def _prompts_and_limits(model: Mapping[str, Any]) -> list[str]:
    """The prompts (grouped by prefix, in the order upstream declares them), the dimensions and the
    input limits of one model."""
    config = model["pipelineConfig"]
    by_prefix: dict[str, list[str]] = {}
    for name, prefix in config["prompts"].items():
        by_prefix.setdefault(prefix, []).append(name)
    placeholders = ", ".join(
        f"{kind} {token}" for kind, token in config["placeholderTokens"].items()
    )
    lines = [
        "### Prompts, dimensions and limits",
        "",
        "Read from the checkpoint's own `config_sentence_transformers.json` and `config.json` and",
        "the exported graph, and checked against each other when this repository was assembled.",
        "",
        "`prompt` names one of these; its prefix is prepended to the text as is (no prompt, no",
        "prefix). Pick it by the role a text plays — `query` for a search query and `document`",
        "for what it searches, for instance.",
        "",
        "| Prefix | Prompt names |",
        "| ------ | ------------ |",
    ]
    for prefix, names in by_prefix.items():
        # A table cell escapes the pipe, inside a code span too (GitHub Flavored Markdown).
        cell = prefix.replace("|", "\\|")
        lines.append(f"| `{cell}` | {', '.join(f'`{name}`' for name in names)} |")
    vision = config.get("vision")
    audio = config.get("audio")
    limit = config["maxTokens"]
    if vision is None and audio is None:
        tokens = f"- **tokens**: at most {limit:,} per text, `<bos>` and `<eos>` included"
    elif audio is None:
        tokens = (
            f"- **tokens**: at most {limit:,} per input, `<bos>` and `<eos>` included; each image"
            f" counts its soft tokens (at most {vision['softTokenBudget']}) and its two markers"
        )
    elif vision is None:
        tokens = (
            f"- **tokens**: at most {limit:,} per input, `<bos>` and `<eos>` included; each audio"
            f" counts its soft tokens (at most {audio['maxSoftTokens']}) and its two markers"
        )
    else:
        tokens = (
            f"- **tokens**: at most {limit:,} per input, `<bos>` and `<eos>` included; each image"
            f" and each audio counts its soft tokens (at most {vision['softTokenBudget']} for an"
            f" image, {audio['maxSoftTokens']} for an audio) and its two markers"
        )
    lines += [
        "",
        f"- **dimensions**: {_spell(config['dimensions'])} (the first is the full vector)",
        tokens,
        f"- **refused tokens**: the multimodal placeholders ({placeholders}) in a text",
    ]
    return lines


def _i8_screening(model: Mapping[str, Any]) -> list[str]:
    """The measured distance of the `i8` preset's weights from the original ones — only for a
    model with a preset that stores the text tower as int8 (none, no section), with a bullet for
    each tower it stores as int8 too."""
    if not any(
        quant["weights"][EMBEDDINGGEMMA2_MODEL_ROLE] == EMBEDDINGGEMMA2_I8_DTYPE
        for quant in model["quants"].values()
    ):
        return []
    towers = [
        line
        for role, lines in EMBEDDINGGEMMA2_I8_TOWER_SCREENING.items()
        if EMBEDDINGGEMMA2_I8_DTYPE in model["weights"].get(role, {})
        for line in lines
    ]
    tower_lines = (
        [
            "",
            *EMBEDDINGGEMMA2_I8_TOWER_SCREENING_LEAD,
            *towers,
            "",
            f"Source: `{EMBEDDINGGEMMA2_I8_TOWER_SCREENING_SOURCE}` in the Karume repository.",
        ]
        if towers
        else []
    )
    return [
        "### How far the int8 weights are from the original ones",
        "",
        *EMBEDDINGGEMMA2_I8_SCREENING,
        *tower_lines,
    ]


def render_embeddinggemma2_model_card(
    manifest: Mapping[str, Any], repo: str, host_assets: Mapping[FileRefKey, int] = {}
) -> str:
    """Build the body of the EmbeddingGemma 2 distribution's `README.md` (a pure function,
    trailing newline)."""
    require_pipeline(manifest, EMBEDDINGGEMMA2_SUPPORTED_PIPELINE)
    metadata = _metadata(manifest)
    return render(
        (
            frontmatter(metadata),
            introduction(_title(manifest), hf_upstreams(metadata.base_model)),
            _overview(manifest),
            [""],
            _base_weights(manifest),
            [""],
            models(manifest),
            [""],
            _usage(manifest, repo),
            *model_sections(
                manifest,
                (partial(quants, host_assets=host_assets), _i8_screening, _prompts_and_limits),
            ),
        )
    )
