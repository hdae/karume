"""Model card (`README.md`) of the SBV2 distribution — a pure function derived mechanically from
the manifest.

SPEC: docs/spec/distribution-cards.md#the-model-card

The generic rendering pieces (frontmatter, model list, quant table, section assembly) live in
`karume.modelcard`. This module holds only **the facts specific to SBV2**: attribution (where the
voices come from, licenses, citations, the redistributed text encoder) and what the card of this
pipeline says.

**Attribution is an axis separate from the template** ({@link Sbv2CardProfile}). Even with the same
SBV2 template, the source, license and citations change completely depending on which family's
weights are shipped, so only that part is split out into a profile. Which profile renders the card
is **decided by the Pipeline** — each voice family has its own Pipeline (`sbv2` = JVNV —
`sbv2/distribution.py`) that holds exactly one profile, so the caller has nothing to choose.

MUST: **every number, download size, quant table, dtype label, style table and speaker table is
derived from the manifest** (the same MUST of `karume.modelcard` applies as is). The only constants
allowed here are facts that **do not exist** in the manifest.
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
    Measurement,
    UpstreamQuote,
    changes_made_here,
    default_model,
    from_pretrained,
    frontmatter,
    hf_link,
    introduction,
    knob,
    measurements,
    model_sections,
    models,
    pinned,
    quants,
    quotation,
    render,
    require_pipeline,
    upstream_cards,
)

#: The pipeline contract this template can describe (ADR 0041 §2 — per model).
SBV2_SUPPORTED_PIPELINE = "sbv2/1"

SBV2_PIPELINE_TAG = "text-to-speech"

#: Architecture (every family is JP-Extra — what differs is the source of the weights, not the
#: shape).
SBV2_ARCHITECTURE = "Style-Bert-VITS2 JP-Extra"

#: The Japanese BERT that `text_encoder` was made from (`MODEL_ID` of `deberta/export.py`). The
#: manifest holds only the role name, so the card carries the attribution. License checked on the
#: source (2026-08-07). **Every family redistributes the same single model**, so it lives here
#: rather than in a profile field.
SBV2_TEXT_ENCODER_MODEL = "ku-nlp/deberta-v2-large-japanese-char-wwm"
SBV2_TEXT_ENCODER_LICENSE = "cc-by-sa-4.0"

#: The commit of {@link SBV2_TEXT_ENCODER_MODEL} the text encoder is exported from (`MODEL_REVISION`
#: of `deberta/export.py`, which bakes it into the container's provenance — kept as a literal here,
#: as importing that module would load torch; `tests/test_card.py` checks the two agree).
SBV2_TEXT_ENCODER_REVISION = "547b0e8b044fba3f9b84d0ab9f990440bd130c8b"

#: The number of encoder layers left in the distributed text_encoder. It is the layer count that
#: makes `hidden_states[-3]`, which SBV2 uses (`[0]` = embedding output, `[i+1]` = output of layer
#: i, so the 22nd from the start = the output of layer 21), **the graph's final output**; the
#: reference implementation's index is resolved into this one number. It lives here because the
#: card states the truncation as a change (CC BY-SA 4.0 asks for the changes to be indicated); the
#: assembly (`sbv2.distribution`) checks the series against the same number and its `NOTICE.md`
#: states it too.
SBV2_TEXT_ENCODER_LAYERS = 22

#: Demo sentence of the usage snippet (an input of the Japanese TTS, so it stays Japanese — the
#: language rule of CLAUDE.md).
SBV2_DEMO_TEXT = "こんにちは、これはテストです。"

#: **How each quant was rounded** (a fact the manifest does not hold — the table only says "which
#: quant uses which storage dtype"; how that dtype was produced (per-channel RTN / GPTQ calibration
#: / group-32) is knowledge of the scripts).
#: A value is the lines of one list item. The first line is the text that follows the quant name;
#: the following lines are Markdown continuation lines and carry their own indentation (the same
#: "the lines themselves" form as `attribution`).
#:
#: No default mark is written here — the default is taken from the manifest's `defaultQuant`
#: ({@link _sbv2_quants}). Writing it here would make the table and the notes go wrong separately
#: when the default moves.
SBV2_QUANT_ROUNDING: Mapping[str, tuple[str, ...]] = {
    "f16+bert8": (
        "`front` / `voice` in `f16`, each weight rounded to the nearest `f16` value;",
        "  `text_encoder` in `i8`, rounded per output channel (plain RTN).",
    ),
    "i8": ("every weight in `i8`, rounded per output channel (plain RTN).",),
    "i8-a8": (
        "the same stored weights as `i8`. Nothing more is rounded at export time: the",
        "  linear layers quantize their input activations to int8 per token at run time.",
    ),
    "i8+bert4": (
        "`text_encoder` in `i4` group-32: its linear layers rounded with **GPTQ",
        "  calibration** (a 48-sentence Japanese corpus), its embedding table plain RTN.",
        "  `front` / `voice` stay `i8`.",
    ),
    "i4": (
        "the same text encoder as `i8+bert4`, plus `front` / `voice` linear and conv1d",
        "  weights in `i4` group-32 with plain RTN.",
    ),
}

#: What the measurements below were taken on, when it was not these voices: they predate this
#: repository's voices (2026-08-03 / 04) or used another voice of the same architecture (the
#: records of 2026-08-20 name it). The text encoder is the same in every voice, so its timing
#: needs no subject.
_SBV2_OTHER_VOICE = f"another {SBV2_ARCHITECTURE} voice"

#: The development machine's GPU on those dates (ADR 0026 names it for 2026-08-03; the records
#: of 2026-08-20 name no device, and the machine had this GPU then).
_SBV2_MEASURED_ON = "NVIDIA GeForce RTX 3080 Ti"

#: What the presets measured, written under every voice's quant table (the maintainer's ruling
#: of 2026-11-27: a measurement is a note naming its device and date, never a preset's
#: description). Sources: ADR 0029 (2026-08-04 — the torch simulation of the stored weights,
#: the text encoder in f32 for every configuration), ADR 0026 (2026-08-03 — the text encoder
#: stage of the demo sentence, 17 tokens), perf-ledger Q-5 and research 2026-08-20 §7 (warm
#: synthesis after the conv1d int4 addendum) and the maintainer's listening of 2026-08-20 after
#: that addendum.
SBV2_MEASUREMENTS: tuple[Measurement, ...] = (
    Measurement(
        "f16+bert8",
        "with `front` and `voice` in f16 and the text encoder in f32, the audio differed from"
        " f32's by an SNR of 40.5 dB and a log-spectral distance of 0.36 dB, against 8.6 dB and"
        " 3.20 dB in int8",
        "a torch simulation of the stored weights on the CPU",
        "2026-08-04",
        _SBV2_OTHER_VOICE,
    ),
    Measurement(
        "i8-a8",
        "the text encoder stage of a 17-token sentence took 228 ms against 279 ms without the"
        " int8 activations",
        _SBV2_MEASURED_ON,
        "2026-08-03",
    ),
    Measurement(
        "i8+bert4",
        "a listening check against f32 heard it nearly identical",
        _SBV2_MEASURED_ON,
        "2026-08-20",
        _SBV2_OTHER_VOICE,
    ),
    Measurement(
        "i4",
        "warm synthesis took about 500 ms, the fastest of `i8`, `i8+bert4` and `i4`",
        _SBV2_MEASURED_ON,
        "2026-08-20",
        _SBV2_OTHER_VOICE,
    ),
    Measurement(
        "i4",
        "a listening check against f32 heard slightly lower tension",
        _SBV2_MEASURED_ON,
        "2026-08-20",
        _SBV2_OTHER_VOICE,
    ),
)


@dataclass(frozen=True)
class Sbv2CardProfile:
    """The **attribution profile** of the SBV2 card — the "facts not in the manifest" that differ
    per voice family.

    There is one SBV2 template, but attribution (source, license, redistribution terms, citations)
    is **a different legal fact** for each family. Hard-coded, rendering into another family's
    repository would leave the table and the usage correct while the attribution stays that of the
    previous family — an error nobody can notice until after shipping, so it is held here as a
    separate field (which one is used is fixed by each family's Pipeline holding exactly one —
    `PIPELINE` in `sbv2/distribution.py`).

    `attribution` is the lines themselves because what goes into this field is prose that cannot be
    derived (summaries of license terms, citations, redistribution terms). Only the two lines that
    can be built mechanically (Voices / Architecture) are built by {@link _sbv2_base_weights} from
    `source_dirs` / `source_version`.
    """

    metadata: CardMetadata
    title: str
    #: The directories of the source repository that became the voices of this distribution.
    source_dirs: tuple[str, ...]
    #: The version string the source `config.json` declares (the SBV2 generation — it does not
    #: identify a checkpoint).
    source_version: str
    #: Commit SHA of the source repository (the container's `provenance.upstreamRevision` — "the
    #: revision of the upstream checkpoint" of container-v1 §2.3). A fact separate from
    #: {@link source_version} and not put into the same field (the version is shared by every model
    #: of a generation, so it cannot identify a checkpoint — 2026-09-24 repository-wide review
    #: W-RC5-3). `None` for a family whose retrieved revision has not been confirmed (the container
    #: declares no revision — a guessed value is not baked in).
    source_revision: str | None
    #: The attribution items that follow Voices / Architecture (the Markdown lines themselves).
    attribution: tuple[str, ...]


# ---- JVNV (a family) ----------------------------------------------------

#: License (checked on the source 2026-08-09 —
#: https://huggingface.co/litagin/style_bert_vits2_jvnv): the source repository states that "the
#: license inherits the cc-by-sa-4.0 of the JVNV corpus".
#: A standard SPDX tag applies, so there is no `license_name` / `license_link` (HF resolves that
#: field from the identifier; declaring a custom name would read as "an unnamed custom license").
#: `base_model` also lists the text encoder because this distribution **redistributes both**.
#: Both are cc-by-sa-4.0, so the SA (ShareAlike) term is consistent across the whole repository.
SBV2_JVNV_METADATA = CardMetadata(
    pipeline_tag=SBV2_PIPELINE_TAG,
    base_model=("litagin/style_bert_vits2_jvnv", SBV2_TEXT_ENCODER_MODEL),
    base_model_relation="quantized",
    license="cc-by-sa-4.0",
    tags=("text-to-speech", "webgpu", "japanese"),
)

#: The clause of CC BY-SA 4.0 that asks for the notice of the changes (§3(a)(1)(B) — "indicate if
#: You modified the Licensed Material"), as the changes bullet cites it.
SBV2_MODIFICATION_CLAUSE = "CC BY-SA 4.0 §3(a)(1)(B)"

#: The CC BY-SA 4.0 legal text (the one link with which the card itself meets BY's "indicate the
#: license URL").
SBV2_JVNV_LICENSE_URL = "https://creativecommons.org/licenses/by-sa/4.0/"

#: Paper and distribution page of the voices' training data (the JVNV corpus). The card's
#: attribution and the `NOTICE.md` at the root of the distribution repository
#: (`sbv2.distribution.SBV2_JVNV_NOTICE_MARKDOWN`) use the same spelling.
SBV2_JVNV_CORPUS_PAPER = "arXiv:2310.06072"
SBV2_JVNV_CORPUS_PAPER_URL = "https://arxiv.org/abs/2310.06072"
SBV2_JVNV_CORPUS_PAGE = (
    "https://sites.google.com/site/shinnosuketakamichi/research-topics/jvnv_corpus"
)

#: The source repository's statement that the voices inherit the corpus's license, **verbatim** (it
#: is written in Japanese — a statement of upstream's, quoted, not reworded: the maintainer's ruling
#: of 2026-11-27), with its translation, which is Karume's and is labeled so. The card at the pinned
#: commit, read 2026-11-27. `NOTICE.md` keeps one line stating the basis and refers to the card.
SBV2_JVNV_LICENSE_QUOTE = UpstreamQuote(
    repo=SBV2_JVNV_METADATA.base_model[0],
    revision="205830ca1d49e666ddfbf2a755f0108e9cade4dd",
    sections=("Style-Bert-VITS2 JVNVコーパスモデル",),
    paragraphs=("ライセンスはJVNVコーパスのcc-by-sa-4.0を引き継ぎます。",),
)
SBV2_JVNV_LICENSE_QUOTE_IN_ENGLISH = (
    '"The license carries over the cc-by-sa-4.0 of the JVNV corpus."'
)

#: Every quotation of an upstream card the card writes (checked against the cached cards by
#: `tests/test_dist_driver.py`).
UPSTREAM_QUOTES: tuple[UpstreamQuote, ...] = (SBV2_JVNV_LICENSE_QUOTE,)

SBV2_JVNV_PROFILE = Sbv2CardProfile(
    metadata=SBV2_JVNV_METADATA,
    title="Style-Bert-VITS2 JVNV — Karume",
    source_dirs=("jvnv-F1-jp/", "jvnv-F2-jp/", "jvnv-M1-jp/", "jvnv-M2-jp/"),
    source_version="2.0-JP-Extra",
    # main of `litagin/style_bert_vits2_jvnv` at retrieval time (`refs/main` of the HF cache — the
    # value written at 2026-08-09 08:07:52, in the middle of retrieving the 4 voices (08:07:32–
    # 08:08:02, `inputs/sbv2/{F1,F2,M1,M2}`). Checked against the files on 2026-09-25).
    source_revision=SBV2_JVNV_LICENSE_QUOTE.revision,
    attribution=(
        f"- **Terms**: licensed **{SBV2_JVNV_METADATA.license}** ([legal code]"
        f"({SBV2_JVNV_LICENSE_URL}) — a verbatim copy is in `LICENSE.md`), the license of the JVNV"
        " corpus the",
        "  voices were trained on. Redistribution here keeps it: credit the authors, name the"
        " source and",
        "  the license URL, state that the weights were **converted to another format and"
        " quantized**",
        "  (the next bullet lists every change),",
        f"  license any derivative work under **{SBV2_JVNV_METADATA.license}** as well, and impose"
        " no further",
        "  restrictions. There is no NonCommercial and no NoDerivatives clause — commercial use",
        "  and modification are both allowed.",
        *quotation(SBV2_JVNV_LICENSE_QUOTE, "  The source repository's model card", indent="  "),
        "",
        f"  In English: {SBV2_JVNV_LICENSE_QUOTE_IN_ENGLISH}",
        f"{changes_made_here(SBV2_MODIFICATION_CLAUSE)} conversion into the Karume container"
        " format, **truncation of the",
        f"  text encoder to its first {SBV2_TEXT_ENCODER_LAYERS} encoder layers** (it now ends at"
        " the hidden state",
        "  Style-Bert-VITS2 reads; the layers after it were dropped) and **quantization** of the",
        "  weights — the voices in `f16` / `i8` / mixed, the text encoder in `i8` / mixed,",
        "  where the mixed form stores the group-quantizable weights as `i4` and everything else",
        "  as `i8` (the quant table below says which storage each quant selects). No retraining,",
        "  no fine-tuning — the voices are the source checkpoints in a different storage form.",
        f"- **Text encoder**: {pinned(SBV2_TEXT_ENCODER_MODEL, SBV2_TEXT_ENCODER_REVISION)},",
        f"  licensed **{SBV2_TEXT_ENCODER_LICENSE}** (as of retrieval). It is redistributed here",
        "  in the container format as the `text_encoder` component — the same license as the",
        "  voices, so the share-alike term is consistent across everything in this repository.",
        f"- **Training data**: the JVNV corpus, licensed **{SBV2_JVNV_METADATA.license}**. Detai"
        " Xin, Junfeng Jiang,",
        "  Shinnosuke Takamichi, Yuki Saito, Akiko Aizawa, Hiroshi Saruwatari,",
        # The paper title stays on one line (if wrapped, a search for the title would miss it even
        # though it is in the text).
        "  *JVNV: A Corpus of Japanese Emotional Speech with Verbal Content and"
        " Nonverbal Expressions*,",
        f"  [{SBV2_JVNV_CORPUS_PAPER}]({SBV2_JVNV_CORPUS_PAPER_URL}); the"
        f" [corpus page]({SBV2_JVNV_CORPUS_PAGE}).",
        "- **Training implementation**:"
        " [Style-Bert-VITS2](https://github.com/litagin02/Style-Bert-VITS2)",
        "  (licensed **agpl-3.0**). Karume is a separate implementation and does not contain the"
        " original",
        "  implementation.",
    ),
)

#: The selectable attribution profiles. **There is no default** — silently picking one when it is
#: omitted would silently bring back the previous family's attribution every time a new family is
#: shipped (`karume.dist` requires it explicitly).
SBV2_CARD_PROFILES: Mapping[str, Sbv2CardProfile] = {
    "jvnv": SBV2_JVNV_PROFILE,
}


def _sbv2_overview(manifest: Mapping[str, Any]) -> list[str]:
    return [
        "## What is this",
        "",
        f"A Japanese text-to-speech distribution: **{SBV2_ARCHITECTURE}** voices converted into",
        "the WebGPU inference runtime **Karume**'s container format (a `.krm` part sequence whose",
        "first part carries the graph and model descriptors).",
        RUNTIME_SENTENCE,
        "",
        "- The acoustic chain is shipped as fused graphs: `text_encoder` (a Japanese DeBERTa),",
        "  `front` (phoneme encoder + duration predictors) and `voice` (flow + HiFi-GAN decoder).",
        "- Style and speaker are **looked up at run time** from the shipped tables — the names in",
        "  the tables below index the rows of `style_vectors` / `speaker_embeddings`.",
        "- Not readable by Style-Bert-VITS2 (it's a different container with an embedded graph);"
        f" the reader is a pipeline that implements `{SBV2_SUPPORTED_PIPELINE}`.",
        f"- Exporter used for the conversion: `{manifest['generator']}`. The distribution manifest"
        f" is `karume.json` (`{manifest['format']}`).",
    ]


def _sbv2_base_weights(profile: Sbv2CardProfile) -> list[str]:
    """Attribution section. The voices' source repository is taken from the first `base_model` entry
    (= the source of the weights itself)."""
    base_model = profile.metadata.base_model[0]
    dirs = " / ".join(f"`{name}`" for name in profile.source_dirs)
    # A family whose retrieved revision is not recorded names its source unpinned (no commit is
    # guessed — the profile's `source_revision`).
    voices = (
        hf_link(base_model)
        if profile.source_revision is None
        else pinned(base_model, profile.source_revision)
    )
    return [
        "## Base weights and attribution",
        "",
        "Converted into the container format — the original checkpoints are not distributed here.",
        "",
        f"- **Voices**: {dirs} of {voices}",
        f"- **Architecture**: {SBV2_ARCHITECTURE} (`version: {profile.source_version}`)",
        *profile.attribution,
        *upstream_cards([hf_link(repo) for repo in profile.metadata.base_model]),
    ]


def _sbv2_quants(
    model: Mapping[str, Any],
    *,
    abbreviations: Mapping[str, str],
    host_assets: Mapping[FileRefKey, int] = {},
) -> list[str]:
    """The generic quant table plus SBV2's notes on **how the weights were rounded**.

    The table can only say "which quant uses which storage dtype" (the manifest holds no more than
    that), so it cannot tell whether an `i4` is GPTQ-calibrated or plain RTN — the reason the
    default was chosen for quality disappears from the card. Notes are attached **only to the quants
    this model declares** (the same discipline as `_sbv2_knob` — describing a quant the distribution
    does not have makes the card say more than what it can ship).

    A different responsibility from the display fields (`label` / `description`) — those are the one
    line from the manifest saying "what one quant is", while this is the scripts' knowledge of "how
    that storage was produced" (ADR 0075 decision 5). What the presets measured follows
    ({@link SBV2_MEASUREMENTS}).
    """
    default = model["defaultQuant"]
    table = quants(model, abbreviations=abbreviations, host_assets=host_assets)
    notes: list[str] = []
    for name in model["quants"]:
        rounding = SBV2_QUANT_ROUNDING.get(name)
        if rounding is None:
            continue
        mark = " (default)" if name == default else ""
        head, *rest = rounding
        notes += [f"- `{name}`{mark} — {head}", *rest]
    rounded = [] if not notes else ["", "How the stored weights were rounded:", "", *notes]
    return [*table, *rounded, *measurements(model, SBV2_MEASUREMENTS)]


def _name_id_rows(table: Mapping[str, int]) -> list[str]:
    """Turns a `name → ID` table into rows in the manifest's order (the ID is the row number of the
    shipped table itself)."""
    return [f"| `{name}` | {identifier} |" for name, identifier in table.items()]


def _sbv2_styles(model: Mapping[str, Any]) -> list[str]:
    """Style list (the section that lets a user learn from the card alone what `style` accepts)."""
    path = model["assets"]["style_vectors"]["path"]
    return [
        "### Styles",
        "",
        "| Style | ID |",
        "| ----- | -- |",
        *_name_id_rows(model["pipelineConfig"]["styles"]),
        "",
        f"`style` takes one of these names — the ID is the row it selects in `{path}`.",
        "`styleWeight` blends between the average style (`0`) and the named one (`1`).",
    ]


def _sbv2_speakers(model: Mapping[str, Any]) -> list[str]:
    """Speaker list (the values `speaker` accepts — a row-number mapping, like the styles)."""
    path = model["assets"]["speaker_embeddings"]["path"]
    return [
        "### Speakers",
        "",
        "| Speaker | ID |",
        "| ------- | -- |",
        *_name_id_rows(model["pipelineConfig"]["speakers"]),
        "",
        f"`speaker` takes one of these names — the ID is the row it selects in `{path}`.",
    ]


def _sbv2_knob(defaults: Mapping[str, Any], key: str, note: str) -> list[str]:
    """Turns one optional knob of `generate()` into a comment line (only knobs whose default is in
    the manifest).

    Keys the manifest does not hold are **not written** — recommending a knob missing from
    `defaults` makes the card say more than "the values selectable in this distribution" (the same
    judgement as `### Defaults` not keeping a copy of `defaults`, for the same reason).
    """
    if key not in defaults:
        return []
    value = defaults[key]
    literal = f'"{value}"' if isinstance(value, str) else f"{value}"
    return [f"  // {key}: {literal}, // default — {note}"]


def _sbv2_usage(manifest: Mapping[str, Any], repo: str) -> list[str]:
    """Policy of the usage example: keep the minimal working form, and **list the optionals an
    ordinary use case is likely to need as commented-out lines** (the selectable values are listed
    in the comment on the same line — derived mechanically from the manifest, so the lists follow
    when voices / styles / quants are added). The reader moves to the next step by just
    uncommenting.


    NOTE: `fromAssets` is **not shown** (ruling of 2026-08-29). It can read split distributions too
    (X2-101), but it is a surface for local debugging that assumes the caller holds the bytes; the
    ordinary entry point for a reader using HF is `fromPretrained`. Showing both would leave the
    reader to decide "which one to use".
    """
    model_name = manifest["defaultModel"]
    model = default_model(manifest)
    quant = model["defaultQuant"]
    config = model["pipelineConfig"]
    defaults = config["defaults"]
    model_names = " / ".join(sorted(manifest["models"]))
    quant_names = " / ".join(sorted(model["quants"]))
    style_names = " / ".join(sorted(config["styles"]))
    speaker_names = " / ".join(sorted(config["speakers"]))
    return [
        "## Usage",
        "",
        "```ts",
        'import { analyzeWithWords } from "jsr:@hdae/yomi";',
        'import { getDictionary } from "jsr:@hdae/yomi/loader";',
        'import { encodeWav } from "jsr:@karume/models";',
        'import { Pipeline, toUtterance } from "jsr:@karume/models/sbv2";',
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
        "// Text analysis (readings and accents) is the caller's job:",
        "const utterance = toUtterance(analyzeWithWords(await getDictionary(),"
        f' "{SBV2_DEMO_TEXT}"));',
        "",
        "const audio = await pipeline.generate(utterance, {",
        "  // Voice — the names below come from the tables further down; every model in",
        "  // this repository brings its own set:",
        *_sbv2_knob(defaults, "style", f"available: {style_names}"),
        *_sbv2_knob(defaults, "styleWeight", "0 = the average style, 1 = the named one"),
        *_sbv2_knob(defaults, "speaker", f"available: {speaker_names}"),
        "",
        "  // Delivery:",
        *_sbv2_knob(defaults, "lengthScale", "larger is slower"),
        *_sbv2_knob(defaults, "sdpRatio", "1 = stochastic duration, 0 = deterministic"),
        *_sbv2_knob(defaults, "noiseScale", "sampling noise on z_p"),
        *_sbv2_knob(defaults, "noiseScaleW", "sampling noise inside the stochastic predictor"),
        "",
        "  seed: 42, // same seed + same knobs → same waveform",
        "});",
        'await Deno.writeFile("sbv2.wav", encodeWav(audio.data, audio.sampleRate));',
        "```",
        "",
        "`generate()` returns `{ sampleRate, data }`, where `data` is an f32 mono waveform —",
        "exactly what `encodeWav` takes.",
        "Weights are fetched once and cached (verified against `karume.json`'s `size` / `sha256`).",
        "",
        "`generate()` takes an already-analyzed utterance: text analysis (readings and accents)",
        "belongs to the caller, and the package fetches no dictionary of its own.",
        "The snippet uses [`@hdae/yomi`](https://jsr.io/@hdae/yomi), whose dictionary",
        "(about 19 MB) is fetched on the first `getDictionary()` and read from the Cache API",
        "afterwards; to fix a reading, put yomi's overlay dictionary in front of the analysis.",
        "The SBV2 page of the",
        "[Karume Playground](https://github.com/hdae/karume/tree/main/apps/playground/src/models/sbv2)",
        "is the worked version.",
    ]


def _sbv2_defaults(model: Mapping[str, Any]) -> list[str]:
    defaults = model["pipelineConfig"]["defaults"]
    return [
        "### Defaults",
        "",
        "Any knob not passed to `generate()` is filled in from the manifest's defaults.",
        "",
        *(f"- **{key}**: {knob(value)}" for key, value in defaults.items()),
        "",
        "`seed` is the one knob the manifest does not carry — it defaults to `0`, and the same"
        " seed with the same knobs gives the same waveform.",
    ]


def render_sbv2_model_card(
    manifest: Mapping[str, Any],
    repo: str,
    profile: Sbv2CardProfile,
    abbreviations: Mapping[str, str],
    host_assets: Mapping[FileRefKey, int] = {},
) -> str:
    """Builds the `README.md` body of the SBV2 distribution (a pure function, with a trailing
    newline).

    `profile` has no default, MUST — attribution is a fact that differs per family, and the moment
    it has a default, a path that "renders the previous family's attribution into another family's
    repository" silently appears.

    `abbreviations` maps the component-override tokens of quant names (the source of truth is
    `sbv2.distribution` — ADR 0074 decision 4). It is a fact not in the manifest, so it is taken as
    an argument rather than copied as a constant.
    """
    require_pipeline(manifest, SBV2_SUPPORTED_PIPELINE)
    return render(
        (
            frontmatter(profile.metadata),
            introduction(profile.title, hf_upstreams(profile.metadata.base_model)),
            _sbv2_overview(manifest),
            [""],
            _sbv2_base_weights(profile),
            [""],
            models(manifest),
            [""],
            _sbv2_usage(manifest, repo),
            *model_sections(
                manifest,
                (
                    partial(_sbv2_quants, abbreviations=abbreviations, host_assets=host_assets),
                    _sbv2_styles,
                    _sbv2_speakers,
                    _sbv2_defaults,
                ),
            ),
        )
    )
