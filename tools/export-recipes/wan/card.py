"""The model card (`README.md`) of the Wan distribution — pure functions derived mechanically from
the manifest (Wan2.1 / Wan2.2).

The general-purpose rendering parts (frontmatter, model list, quant table, assembling the sections)
live in `karume.modelcard`. This module owns only **the facts of each Wan generation**: attribution
(provenance, pinned revision, license) and what this pipeline's card says (the fixed prompts of
step 1, the accepted inputs, the numeric checks, the measured resources).

The facts that differ per generation are collected in the generation table {@link WanCard}
({@link WAN21_CARD} / {@link WAN22_CARD}); the card's skeleton (section order, attribution lines,
Usage, prompts, the frame of the accepted inputs, defaults, declared limits) is drawn by the single
{@link render_wan_model_card}. A generation table holds the facts that can be passed as values
(pipeline contract, model set, accepted sets, public class name) and the renderers of the sections
whose prose differs entirely (overview, summary of modifications, verification scope, measured
resources).

MUST: **every number, download size, quant table, dtype label and default is derived from the
manifest** (the same MUST of `karume.modelcard` applies as is — the text_encoder download size, the
target of a cross-repository reference and the declared device limits also come from the manifest).
The only constants allowed here are facts **that are not in the manifest** — the upstream sources
and pins (`wan.sources.SOURCES` is the source of truth), the table of fixed prompts
(`wan.prompts.FIXED_PROMPTS` is the source of truth — the assembly check
`wan.distribution.assert_text_embeds` matches it against the asset's metadata, so the text drawn
here cannot drift from the shipped assets), the accepted sets of the TS side (`acceptedSizes` /
`minFrames` / `maxFrames` of `WAN21_GENERATION` in `packages/models/src/wan/descriptor.ts`, the
choice of text path `textEncoder` — ADR 0119 addendum B, the prompt acceptance rules — ADR 0119
decisions 1 / 2 / 4 and addendum 10a; Wan2.2 uses `WAN22_TI2V_GENERATION`) and the measured
resources (`_wan21_resources` / `_wan22_resources` — ADR 0089 decision 3).

MUST: do not import torch (`import dist` does not load torch —
`TestImportingTheDriver` in `tests/test_dist_driver.py`).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
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
from wan.prompts import FIXED_PROMPTS
from wan.sources import SOURCES, UMT5_SOURCES, WAN21_MODELS, WAN22_MODELS, WAN22_TEXT_MODEL
from wan.umt5_distribution import UMT5_DEFAULT_MODEL, UMT5_ROLE

#: The pipeline contract this template can describe (ADR 0041 §2 — per model).
WAN_SUPPORTED_PIPELINE = "wan/1"

#: The HF pipeline tag (the same as upstream).
WAN_PIPELINE_TAG = "text-to-video"

#: The upstream technical report.
WAN_PAPER = "arxiv.org/abs/2503.20314"

#: Where the original text lives (the same text as `LICENSE.md` at the distribution repository's
#: root — Apache 2.0 §4(a)).
WAN_LICENSE_TEXT_LINK = "https://www.apache.org/licenses/LICENSE-2.0"

#: The sizes and frame counts the TS side accepts (a copy of the accepted set in
#: `packages/models/src/wan/descriptor.ts` — a fact not in the manifest).
#: MUST: the same values as the TS side (the tests on both sides compare through
#: `packages/models/tests/fixtures/wan-card-limits.json` — changing one side only turns them red).
WAN_ACCEPTED_SIZES: tuple[tuple[int, int], ...] = ((832, 480), (480, 832))
WAN_FRAMES = (5, 81)

#: The TS side's generation defaults (a copy of `WAN21_GENERATION.defaults` — drawn by the Usage
#: comment).
WAN_DEFAULT_SIZE = (832, 480)
WAN_DEFAULT_FRAMES = 33

#: The construction option that chooses the text path (the second argument of the TS side's
#: `wan.Pipeline.fromPretrained` — ADR 0119 addendum B; a fact not in the manifest). The default is
#: the GPU path (decision 7).
WAN_TEXT_ENCODER_OPTION = "textEncoder"
WAN_TEXT_ENCODER_PATHS = ("gpu", "precomputed")

#: The prompt token counts the GPU path accepts (including the trailing `</s>` — ADR 0119 decision
#: 4; the TS side's accepted set, whose upper bound is the tokenizer asset's `maxLength`).
WAN_PROMPT_TOKENS = (2, 512)

#: The umT5 component name (the key in the manifest's weights — the same source of truth as
#: `wan.distribution.WAN_TEXT_ENCODER_ROLE`).
WAN_TEXT_ENCODER_COMPONENT = UMT5_ROLE

#: The text encoder's upstream (the original encoder — ADR 0122 decisions 1 / 3. It is bit-identical
#: in f32 to this checkpoint's `text_encoder`, but the provenance names the original — so that it
#: does not disagree with the card of the umT5 distribution).
WAN_TEXT_ENCODER_UPSTREAM = UMT5_SOURCES[UMT5_DEFAULT_MODEL].source.repo

#: The quant preset whose end-to-end resources (the per-stage table of ``_wan_resources``) were
#: measured. MUST: the card states the numbers **as numbers of this preset** — a distribution
#: without this preset does not draw them (no numbers for a preset that was not measured — the same
#: as the BiRefNet card).
WAN_RESOURCE_QUANT = "f16"

#: The measured transformer per preset (ADR 0120 addenda "Results of steps 3 / 4 / 5" (DiT alone,
#: normal mode, B570), "Material for step 6" (the practical preset's 50-step 33-frame end-to-end
#: run, the precomputed path) and "The two sha rows of the 50-step practical preset" (the
#: practical preset's 81-frame end-to-end run: wall 3,703 s = DiT stage 3,383 s + VAE stage 319 s)).
#: A row is `(frames, 1 forward, DiT-alone allocation, 50-step end-to-end run)`. The `f16` rows are
#: the baseline, and their end-to-end column is the same run as the per-stage table above (ADR 0118
#: steps 6 / 8). The reference preset `f16+dit8` takes the same time per forward as `f16` in the
#: r check measurement (S = 14,040) — its allocation and 81 frames were not measured.
#: MUST: an unmeasured cell says "not measured" / "not run" and is never filled with an estimate. A
#: preset missing from the table is stated as "not measured" (the preset order stays the
#: manifest's).
WAN_QUANT_TRANSFORMER: Mapping[str, tuple[tuple[int, str, str, str], ...]] = {
    "f16": (
        (33, "17.1 s", "5.15 GiB", "~30 minutes"),
        (81, "68.5 s", "6.17 GiB", "~2 hours"),
    ),
    "f16+dit8": (
        (33, "same as `f16`", "not measured", "not run"),
        (81, "not measured", "not measured", "not run"),
    ),
    "f16+dit8-a8-attn8-s16": (
        (33, "8.5 s", "4.02 GiB", "952 s (~16 minutes)"),
        (81, "34.0 s", "5.46 GiB", "3,703 s (~62 minutes)"),
    ),
}

#: The step-1 latent relative RMS error of the practical preset (against the reference preset
#: `f16+dit8` — measured by the on-machine A/B check of ADR 0120 step 4). `(preset, reference
#: preset, 33 frames, 81 frames)`. The visual result the card states next to it is ADR 0120 ruling
#: 2026-10-04 item 1 (the 12 pairs of step 6 — `f16` and the practical preset, 50 steps, 33 frames).
WAN_PRACTICAL_QUANT_ERROR = ("f16+dit8-a8-attn8-s16", "f16+dit8", "0.107", "0.210")

#: The pipeline contract the Wan2.2 TI2V 5B card can describe (`WAN_TI2V_PIPELINE_NAME` /
#: `WAN_TI2V_PIPELINE_MAJOR` in the TS side's `packages/models/src/wan/config.ts`).
WAN22_SUPPORTED_PIPELINE = "wan-ti2v/1"

#: The sizes and frame counts the TS side accepts (a copy of `WAN22_TI2V_GENERATION` in
#: `packages/models/src/wan/descriptor.ts` — ADR 0121 addendum "Accepted sizes to the 2 official
#: sizes").
#: MUST: the same values as the TS side (the tests on both sides compare through
#: `packages/models/tests/fixtures/wan-ti2v-card-limits.json` — changing one side only turns them
#: red).
WAN22_ACCEPTED_SIZES: tuple[tuple[int, int], ...] = ((1280, 704), (704, 1280))
WAN22_FRAMES = (5, 121)

#: The TS side's generation defaults (a copy of `WAN22_TI2V_GENERATION.defaults` — the maintainer's
#: ruling 2026-10-09, ADR 0121 addendum "Results of step 7"). NOTE: the fixture
#: `wan-ti2v-card-limits.json` does not carry the defaults yet, so nothing compares them with the TS
#: side — when a default changes, fix both descriptor.ts and this place.
WAN22_DEFAULT_SIZE = (1280, 704)
WAN22_DEFAULT_FRAMES = 33

#: Measurements per Wan2.2 preset. A row is `(frames, 1 forward, peak VRAM of the DiT stage, DiT
#: stage, VAE stage, end to end)`. Every 50-step end-to-end run is 1280×704, shift 5, guidance 5,
#: seed 42 (times are DiT stage / VAE stage / wall). The 33 / 49-frame rows are B570, Deno 2.9.6,
#: 2026-10-05, the precomputed path (the peak VRAM is fdinfo's for the DiT stage). Only the
#: 121-frame row differs in both GPU and method (the item below).
#:
#: - The 33-frame rows are, for both presets, opt-in end-to-end runs of **the product class**
#:   (`wan.Ti2vPipeline`) (the "50-step end-to-end run" of ADR 0121 addendum "Results of step 6" —
#:   reference preset 2,142.9 / 439.2 / 2,582.5 s, 7.859 GiB; practical preset 959.9 / 438.5 /
#:   1,398.8 s, 7.843 GiB).
#: - The 49-frame rows are end-to-end runs of **a generation script** that calls the same product
#:   components rather than the product class (the trial table of the addendum "Accepted sizes to
#:   the 2 official sizes"). For the same request with the practical preset at 33 frames, the
#:   generation script's DiT-stage peak (7.72 GiB) was 0.12 GiB lower than the product class's
#:   (7.843 GiB) (the output was byte-identical), so the card notes that direction.
#: - The 121-frame row (practical preset only) is the development machine's run from ADR 0121
#:   addendum "The RTX 3080 Ti lanes and the full verify": RTX 3080 Ti (12,288 MiB), Deno 2.9.6,
#:   2026-10-06, **the GPU umT5 path**, cat-dog-baking, a generation script
#:   (`outputs/diag/wan22-121.ts` — calls internal APIs with a family whose descriptor only widens
#:   the upper bound to 121, and reads the distribution). Wall 3,787.9 s = umT5 9.1 / DiT 3,068.2 /
#:   VAE 709.8 s. **The peak VRAM is nvidia-smi's for the whole GPU** (DiT stage 11,361 MiB = 11.09
#:   GiB — not fdinfo), and the times include thermal limiting (93 ℃, throttle `sw_thermal`), so
#:   they cannot be compared with the B570 rows — the card's note states that row's GPU, runtime,
#:   method and thermal limiting. Record: `outputs/misc/wan22-121-2026-10-06/record.json`. The
#:   reference preset's 121-frame 50-step run was not done, so it has no row (no estimate fills it).
#: - 1 forward is the time from the addendum "Results of step 2" (2026-10-04) (DiT alone, normal
#:   mode, batch 1, S = 7,920 = 1280×704×33). 1 forward at 49 frames (S = 11,440) and at 121 frames
#:   (S = 27,280) was not measured.
#:
#: - The `f16+dit8-a8-attn8-online` row (ADR 0132 "Results"): RTX 3080 Ti, Node + Dawn,
#:   2026-10-23, the DiT measured-mode e2e at S = 27,280 (one forward, the second run of the same
#:   input) — 1 forward 21.7 s and the DiT allocation 9.01 GiB by the runtime's own count (neither
#:   nvidia-smi nor fdinfo); the card's note names that reading. It is the only measured 121-frame
#:   forward. The preset has not run 50 steps, so its stage / VAE / clip cells are "not run", and
#:   it has no 33 / 49-frame rows (not measured).
#:
#: MUST: an unmeasured cell says "not measured" and is never filled with an estimate. A preset
#: missing from the table is stated as "not measured" (the preset order stays the manifest's — the
#: same treatment as 2.1's {@link WAN_QUANT_TRANSFORMER}).
WAN22_RESOURCES: Mapping[str, tuple[tuple[int, str, str, str, str, str], ...]] = {
    "f16+dit8": (
        (33, "21.8 s", "7.86 GiB", "2,143 s", "439 s", "43 min 3 s"),
        (49, "not measured", "8.45 GiB", "3,360 s", "647 s", "66 min 53 s"),
    ),
    "f16+dit8-a8-attn8-s16": (
        (33, "9.9 s", "7.84 GiB", "960 s", "439 s", "23 min 19 s"),
        (49, "not measured", "8.49 GiB", "1,504 s", "655 s", "36 min 5 s"),
        (121, "not measured", "11.09 GiB", "3,068 s", "710 s", "63 min 8 s"),
    ),
    "f16+dit8-a8-attn8-online": ((121, "21.7 s", "9.01 GiB", "not run", "not run", "not run"),),
}

#: The public class the Wan2.2 Usage calls (`@karume/models/wan`). The resources note names it too
#: (the class that ran the 33-frame end-to-end run).
WAN22_PIPELINE_CLASS = "Ti2vPipeline"

#: The step-1 latent relative RMS error of Wan2.2's practical preset (against the reference preset
#: `f16+dit8` — measured by the on-machine A/B check of ADR 0121 step 7, RTX 3080 Ti, 2026-10-09).
#: `(preset, reference preset, 33 frames, 121 frames)`. The visual result the card states next to it
#: is the maintainer's judgement in the same addendum (12 pairs of the reference and practical
#: presets, 50 steps, 1280×704, 33 frames — no degradation).
WAN22_PRACTICAL_QUANT_ERROR = ("f16+dit8-a8-attn8-s16", "f16+dit8", "0.044", "0.049")
#: The on-machine A/B of the I2V practical preset (ADR 0121 step 9d, i2v-33f of
#: `e2e_wan_ti2v_ab_test.ts` — 1280×704, 33 frames, one conditioning image, fit crop, opt-in): the
#: step-1 latent relRMS. The band is × 2 rounded up (9.3e-2).
WAN22_PRACTICAL_QUANT_I2V_ERROR_33 = "0.046"
#: The online preset's (the default on 2026-10-23 until `conv8`) step-1 latent relative RMS error
#: against the reference preset (ADR 0132 "Results" — `e2e_wan_ti2v_ab_test.ts`, RTX 3080 Ti, Node +
#: Dawn, 2026-10-23: 4.23e-2 at 1280×704×33, inside the s16 preset's band 8.8e-2).
#: `(preset, reference, 33 frames)`. The 121-frame check has not run (the reference preset's
#: 121-frame step does not fit the 12 GiB device under Node + Dawn — docs/known-issues.md), and the
#: visual check was done for the s16 preset only; the card says both.
WAN22_ONLINE_QUANT_ERROR = ("f16+dit8-a8-attn8-online", "f16+dit8", "0.042")
#: The two presets of the int8 VAE decoder (ADR 0133 decision 6): `(reference, practical)`. Their
#: the practical one is the default since 2026-10-23 (decision 8 — the decoder A/B, the timing and
#: the maintainer's visual check are in the card's prose). Neither has a row in
#: {@link WAN22_RESOURCES}, so the table's "not measured yet" line names them.
WAN22_INT8_VAE_QUANTS = ("i8", "i8-a8-attn8-online-conv8")


@dataclass(frozen=True)
class WanCard:
    """The generation table of the card — the facts (values) that differ per generation and the
    renderers of the sections whose prose differs entirely.

    The skeleton (section order, attribution lines, Usage, prompts, the frame of the accepted
    inputs, defaults, declared limits) is drawn by the single {@link render_wan_model_card}.
    """

    #: The generation's name (attribution lines and check messages — `Wan2.1` / `Wan2.2`).
    generation: str
    #: The checkpoint name shown in the card's heading.
    title: str
    #: The pipeline contract this card can describe (ADR 0041 §2).
    supported_pipeline: str
    #: The models this generation's distribution ships (the attribution check — not every model of
    #: the sources table passes).
    models: tuple[str, ...]
    #: The public class the Usage calls (`@karume/models`).
    pipeline_class: str
    #: The upstream technical report (the one the upstream README cites — None omits the line when
    #: it cannot be confirmed).
    paper: str | None
    #: A copy of the TS side's accepted set (the list of sizes `(width, height)`, the frame count
    #: `(min, max)`).
    accepted_sizes: tuple[tuple[int, int], ...]
    frames: tuple[int, int]
    #: The TS side's generation defaults (size `(width, height)` and frame count — a copy of
    #: `defaults` in `descriptor.ts`). Drawn by the Usage comment. Not derived from the head of the
    #: accepted set (the defaults are a ruling separate from the accepted set — for Wan2.2, ADR 0121
    #: addendum "Results of step 7").
    default_size: tuple[int, int]
    default_frames: int
    #: The "range checked on the GPU" lines written after the accepted-frames line (the first
    #: element continues on the same line as the range).
    frames_checked: tuple[str, ...]
    #: The items of the "Determinism and verification" section that follow the bit-identity item.
    verification: tuple[str, ...]
    #: The "What is this" section (from the manifest — it draws where the text_encoder comes from).
    overview: Callable[[Mapping[str, Any]], list[str]]
    #: The items of the attribution section after the per-model provenance lines (authors, summary
    #: of modifications, text encoder, assets).
    attribution_notes: Callable[[Mapping[str, Any]], list[str]]
    #: The "Resources" section (the measured table and its notes — ADR 0089 decision 3).
    resources: Callable[[Mapping[str, Any]], list[str]]
    #: The lines below exist only for the generation that takes I2V (Wan2.2). They default to empty
    #: — the Wan2.1 card stays byte-identical (2.1's `wan.Pipeline` rejects `image` with
    #: `ModelInputError` — ADR 0121 step 9b). Tags added to the frontmatter after the pipeline tag
    #: (HF has one pipeline tag, so it stays text-to-video — I2V is stated by a tag).
    extra_tags: tuple[str, ...] = ()
    #: Lines added at the end of the Usage section (the I2V example and the note on image decoding).
    usage_extra: tuple[str, ...] = ()
    #: Items added after the seed item in the accepted-inputs section (`image` / `fit`).
    inputs_extra: tuple[str, ...] = ()


def _upstream(name: str, card: WanCard) -> Any:
    """Model name → upstream source (a model of another generation is not drawn — this card states
    only the facts of its generation's distribution. The sources table also holds rows of the other
    generation, so being in the table is not enough)."""
    if name not in card.models:
        raise ValueError(
            f"model '{name}' is not a {card.generation} model"
            f" (known: {list(card.models)}) — a {card.generation} card does not state the"
            " provenance of another generation"
        )
    return SOURCES[name]


def _wan_metadata(manifest: Mapping[str, Any], card: WanCard) -> CardMetadata:
    """Build the frontmatter from the models listed in the manifest (`base_model` is the upstream of
    the manifest's models — HF's meaning of "the model it derives from". The provenance of
    Wan2.2's two text assets (the Wan2.1 checkpoint) is not a model it derives from, so it is
    left out here and the attribution section states it)."""
    licenses = {_upstream(name, card).license for name in manifest["models"]}
    if len(licenses) != 1:
        raise ValueError(
            f"the license differs per model ({sorted(licenses)}) — it cannot be written as one"
            " value"
        )
    return CardMetadata(
        pipeline_tag=WAN_PIPELINE_TAG,
        base_model=tuple(_upstream(name, card).repo for name in manifest["models"]),
        # A distribution that lowers the f32 upstream to f16 / i8 (the doc of `CardMetadata` — f16 /
        # i8 distributions are `quantized`).
        base_model_relation="quantized",
        license=next(iter(licenses)),
        tags=(WAN_PIPELINE_TAG, *card.extra_tags, "video-generation", "wan", "webgpu"),
    )


def _text_encoder(manifest: Mapping[str, Any]) -> tuple[int, tuple[str, str] | None]:
    """The `(byte count, cross-repository reference target)` of the text_encoder container that the
    default model's default quant fetches.

    The reference target is `(repo, revision)` (the same for every part of a container — ADR 0109
    decision 3), or `None` for a distribution stored in its own repository. The card draws "where it
    is fetched from" from this fact of the manifest.
    """
    model = default_model(manifest)
    weights = model["weights"].get(WAN_TEXT_ENCODER_COMPONENT)
    if weights is None:
        raise ValueError(
            f"the manifest's weights have no '{WAN_TEXT_ENCODER_COMPONENT}' — the GPU text path"
            " cannot be drawn"
        )
    label = model["quants"][model["defaultQuant"]]["weights"][WAN_TEXT_ENCODER_COMPONENT]
    parts = weights[label]["container"]["parts"]
    borrowed = {(ref["repo"], ref["revision"]) for ref in parts if "repo" in ref}
    if len(borrowed) > 1:
        raise ValueError(
            "the text_encoder's cross-repository reference has more than one target:"
            f" {sorted(borrowed)}"
        )
    return sum(ref["size"] for ref in parts), next(iter(borrowed), None)


def _gib(size: int) -> str:
    """Spell a byte count in GiB to 3 digits (the same format as this card's resource numbers)."""
    return f"{size / (1 << 30):.2f} GiB"


def _encoder_whereabouts(manifest: Mapping[str, Any]) -> str:
    """The clause for where the text_encoder lives (this repository or the referenced repository —
    from the manifest)."""
    _, borrowed = _text_encoder(manifest)
    return (
        "stored in this repository"
        if borrowed is None
        else f"referenced from [`{borrowed[0]}`](https://huggingface.co/{borrowed[0]})"
    )


def _reader_lines(manifest: Mapping[str, Any], pipeline: str) -> list[str]:
    """The last 2 items of the overview (the reader contract, and the exporter and manifest format
    used for the conversion)."""
    return [
        "- Not readable by diffusers (it's a different container with an embedded graph); the"
        f" reader is a pipeline that implements `{pipeline}`.",
        f"- Exporter used for the conversion: `{manifest['generator']}`. The distribution manifest"
        f" is `karume.json` (`{manifest['format']}`).",
    ]


def _wan21_overview(manifest: Mapping[str, Any]) -> list[str]:
    where = _encoder_whereabouts(manifest)
    return [
        "## What is this",
        "",
        "A **text-to-video** distribution of Wan2.1 T2V 1.3B, converted into the WebGPU",
        "inference runtime **Karume**'s container format (a `.krm` part sequence whose first part",
        "carries the graph and model descriptors).",
        "",
        f"- Four graphs: `{WAN_TEXT_ENCODER_COMPONENT}` (the umT5-XXL text encoder in int8,"
        f" {where}),",
        "  `transformer` (the diffusion transformer, on patchified latent tokens) and",
        "  `vae_decoder_first` / `vae_decoder_next` (the video VAE decoder, one latent frame per",
        "  call, with the causal cache passed in and out).",
        "- The rest runs on the host in TypeScript: the prompt cleaning and the tokenizer (or,",
        f'  with `{WAN_TEXT_ENCODER_OPTION}: "{WAN_TEXT_ENCODER_PATHS[1]}"`, the lookup of the',
        "  precomputed embeddings), the relative-position buckets, the patchify and RoPE tables,",
        "  classifier-free guidance as two batch-1 passes, the flow-matching UniPC scheduler, and",
        "  the tiled VAE decode. The output is `[3, frames, height, width]` float32 in `[-1, 1]`.",
        "- Verified end to end in Deno (Intel Arc B570, Deno 2.9.6): with the text encoder on the",
        "  GPU in 2-step runs (a fixed prompt and a free prompt, 832 × 480, 33 frames, each pinned",
        "  by the SHA-256 of its frames), and with the precomputed embeddings in full 50-step",
        "  runs. A 50-step run through the text encoder has not been done yet.",
        "- In a browser, Chrome on an NVIDIA GeForce RTX 5070 Ti finished one 50-step run (the",
        "  `f16` quant with the precomputed embeddings, 832 × 480, 81 frames) in 46.1 minutes.",
        "  The text encoder on the GPU and the int8 quants have not been run in a browser yet.",
        *_reader_lines(manifest, WAN_SUPPORTED_PIPELINE),
    ]


def _wan22_overview(manifest: Mapping[str, Any]) -> list[str]:
    """The Wan2.2 overview. The verification scope is ADR 0121 addendum "Results of step 6" (the 6
    sha rows are all the reference preset at 17 frames — 1280×704 and 704×1280 with precomputed
    embeddings, the GPU text path at 1280×704 only; the 50-step end-to-end runs of the product
    class are 2 presets at 1280×704 with 33 frames). The 49 / 121-frame end-to-end runs of the
    generation script are stated by the accepted-inputs section ({@link _WAN22_FRAMES_CHECKED}).
    The browser has only one end-to-end run: the practical preset with the GPU umT5 at
    1280×704×121 in Chrome on an RTX 5070 Ti (ADR 0121 addendum "Chrome on the RTX 5070 Ti" —
    wall 2,417.5 s; gpu-lab widened the descriptor's upper bound and called the internal API).

    I2V (ADR 0121 decision 11, addendum "Results of step 9b"): the conditioning has the form of
    diffusers' `expand_timesteps` (`withWanFirstFrameCondition` / `wanConditionMask` in
    `packages/models/src/wan/dit-loop.ts` — the condition mask is true for the tokens of the first
    latent frame, and the conditioned side's time is t = 0). It was verified only by the sha rows
    of the RTX 3080 Ti on the precomputed path, 2 steps × 9 frames and the opt-in 50 steps × 33
    frames (details in {@link _WAN22_VERIFICATION}). In the browser, only the runs that finished in
    Chrome on the RTX 5070 Ti (gpu-lab, 2026-10-10) with the practical preset, the GPU umT5, a
    conditioning image 896×1152 → 704×1280 (crop), 20 steps and shift 7 at 33 / 121 frames (wall
    390.0 / 1,346.5 s — the primary record is the untracked
    `outputs/bench-browser/wan-browser-2026-10-10T14-11-07.191Z.json`). They are not compared with
    reference cases (gpu-lab's design), so no quality is claimed."""
    where = _encoder_whereabouts(manifest)
    return [
        "## What is this",
        "",
        "A **text-to-video** and **image-to-video** distribution of Wan2.2 TI2V 5B, converted into",
        "the WebGPU inference runtime **Karume**'s container format (a `.krm` part sequence whose",
        "first part carries the graph and model descriptors).",
        "",
        f"- Nine graphs: `{WAN_TEXT_ENCODER_COMPONENT}` (the umT5-XXL text encoder in int8,"
        f" {where}),",
        "  `transformer` (the diffusion transformer in int8, on patchified latent tokens),",
        "  `vae_decoder_first` / `vae_decoder_next` (the Wan2.2 video VAE decoder, one latent",
        "  frame per call, with the causal cache passed in and out), `vae_decoder_first_32` /",
        "  `vae_decoder_next_32` (a second decoder pair at a 32-latent tile that borrows the",
        "  16-tile pair's weights, so there is no extra weight download; the application chooses",
        "  the tile explicitly, 16 by default, and the 10 GB class keeps 16) and",
        "  `vae_encoder_pre` / `vae_encoder_attn` / `vae_encoder_post` (the Wan2.2 video VAE",
        "  encoder for one conditioning image, split around its middle attention).",
        "- Text to video by default; image to video when `generate()` is given an `image` (the",
        "  first frame — see Usage and Accepted inputs). The image is fitted to the output size,",
        "  encoded once by the VAE encoder, and at every step the first latent frame of the",
        "  transformer's input is replaced with it (the diffusers form: the tokens of that frame",
        "  are marked by the condition mask and take timestep 0). The first output frame is",
        "  therefore the fitted image after a round trip through the VAE.",
        "- The rest runs on the host in TypeScript: the prompt cleaning and the tokenizer (or,",
        f'  with `{WAN_TEXT_ENCODER_OPTION}: "{WAN_TEXT_ENCODER_PATHS[1]}"`, the lookup of the',
        "  precomputed embeddings), the relative-position buckets, the patchify and RoPE tables,",
        "  classifier-free guidance as two batch-1 passes, the flow-matching UniPC scheduler, the",
        "  tiled VAE decode and the unpatchify of its output, and for image to video the resize,",
        "  normalization and patchify of the image and the replacement of the first latent frame.",
        "  The output is `[3, frames, height, width]` float32 in `[-1, 1]`, at 24 fps.",
        "- Text to video was verified end to end in Deno (Intel Arc B570, Deno 2.9.6) with the",
        "  `f16+dit8` quant in 2-step runs of 17 frames — at 1280 × 704 and 704 × 1280 with the",
        "  precomputed embeddings, and at 1280 × 704 with the text encoder on the GPU — each",
        "  pinned by the SHA-256 of its frames; and in full 50-step runs of both quants at",
        "  1280 × 704 with 33 frames (see Resources).",
        "- Image to video was verified end to end in Deno on an NVIDIA GeForce RTX 3080 Ti with",
        "  the precomputed embeddings, in 2-step runs of 9 frames and (opt-in) 50-step runs of 33",
        "  frames pinned by the SHA-256 of their frames (see Determinism and verification).",
        "- In a browser, Chrome on an NVIDIA GeForce RTX 5070 Ti finished one 50-step run (the",
        "  `f16+dit8-a8-attn8-s16` quant with the text encoder on the GPU, 1280 × 704, 121 frames)",
        "  in 40.3 minutes. Shorter text-to-video clips, the `f16+dit8` quant and the precomputed",
        "  embeddings have not been run in a browser yet. Image to video finished there with the",
        "  same quant and the text encoder on the GPU, from an 896 × 1152 image cropped to",
        "  704 × 1280, in 20-step runs with shift 7: 390.0 s for 33 frames and 1,346.5 s for 121",
        "  frames. These browser runs were not compared with a reference.",
        *_reader_lines(manifest, WAN22_SUPPORTED_PIPELINE),
    ]


def _wan_base_weights(manifest: Mapping[str, Any], card: WanCard) -> list[str]:
    """The attribution section. The upstream revision is shown as the full pinned 40 digits (the
    same value as the container's provenance)."""
    lines = [
        "## Base weights and attribution",
        "",
        "Converted into the container format — the original checkpoint is not distributed here.",
        "",
    ]
    for name in manifest["models"]:
        source = _upstream(name, card)
        lines.append(
            f"- **`{name}`**: [{source.repo}](https://huggingface.co/{source.repo}) at commit"
            f" `{source.revision}`, licensed **{source.license}** (as of retrieval;"
            f" [full text]({WAN_LICENSE_TEXT_LINK}) — a verbatim copy is in `LICENSE.md`)."
        )
    if card.paper is not None:
        lines.append(
            f"- **Authors**: {card.generation} is released by Wan-AI"
            f" ([technical report](https://{card.paper}))."
        )
    return lines + card.attribution_notes(manifest)


def _wan21_attribution_notes(manifest: Mapping[str, Any]) -> list[str]:
    """Wan2.1's summary of modifications, where the text encoder lives, and the precomputed
    embeddings."""
    lines = [
        "- **Changes made here** (listed in full in `NOTICE.md`, per Apache 2.0 §4(b)):",
        "  conversion into the Karume container format; every parameter of the float16",
        "  transformer and the VAE rounded to the nearest float16 value (weight matrices and",
        "  kernels stored as float16, computation in float32); a second, int8 copy of the",
        "  transformer whose linear weights are quantized per output channel (biases and",
        "  normalization weights kept in float32 at the source values);",
        "  the transformer graph re-expressed on patchified tokens, with the RoPE tables built on",
        "  the host from per-axis base tables and applied in a pair-swap form;",
        "  the VAE decoder re-expressed as two one-frame graphs with an explicit causal cache,",
        "  always decoded in overlapping tiles; the tokenizer converted into one table together",
        "  with lookup tables for the upstream prompt cleaning. No retraining and no fine-tuning.",
    ]
    _, borrowed = _text_encoder(manifest)
    if borrowed is None:
        lines.append(
            f"- **The text encoder** (the encoder of `{WAN_TEXT_ENCODER_UPSTREAM}`, bit-identical"
            " in float32 to this checkpoint's `text_encoder`, in int8) is stored in this"
            " repository."
        )
    else:
        repo, revision = borrowed
        lines += [
            "- **The text encoder is not stored here.** `karume.json` references the encoder of",
            f"  `{WAN_TEXT_ENCODER_UPSTREAM}` (bit-identical in float32 to this checkpoint's",
            "  `text_encoder`), converted to int8, at commit"
            f" `{revision[:16]}…` of [`{repo}`](https://huggingface.co/{repo})",
            "  (with the size and the SHA-256 of every part); that repository's `NOTICE.md` lists",
            "  the changes made to it.",
        ]
    lines += [
        "- **Precomputed embeddings**: the repository also ships the encoder's outputs for a",
        "  fixed set of prompts (see below), computed with the upstream encoder in bfloat16 and",
        "  stored as float32, for use without the text encoder.",
    ]
    return lines


def _wan22_attribution_notes(manifest: Mapping[str, Any]) -> list[str]:
    """Wan2.2's summary of modifications, where the text encoder lives, and the 2 assets made from
    the Wan2.1 checkpoint.

    umT5 is described as the original encoder, and this checkpoint's bf16 `text_encoder` as its
    rounding (ADR 0122 decision 8). The provenance of the 2 assets is the Wan2.1 pin (ADR 0121
    decision 9 — the assembly checks `wan.distribution.assert_text_embeds` / `assert_umt5_tokenizer`
    compare against that pin). That the tokenizer files are the same in the 2 checkpoints was
    confirmed by matching sha256 of the 4 files in `tokenizer/` at the 2 pinned revisions
    (2026-10-06).
    """
    text = SOURCES[WAN22_TEXT_MODEL]
    lines = [
        "- **Changes made here** (listed in full in `NOTICE.md`, per Apache 2.0 §4(b)):",
        "  conversion into the Karume container format; the transformer shipped only in int8,",
        "  its linear weights quantized per output channel (biases and normalization weights kept",
        "  in float32 at the source values); every parameter of the VAE decoder rounded to the",
        "  nearest float16 value (weight matrices and kernels stored as float16, computation in",
        "  float32), and a second copy of the decoder with its convolution and linear weights",
        "  quantized to int8 per output channel from those rounded values (computation in float32,",
        "  or on per-pixel int8 activations in the quant that declares it); the transformer graph",
        "  re-expressed on patchified tokens, with the RoPE tables",
        "  built on the host from per-axis base tables and applied in a pair-swap form, and with",
        "  the image-conditioning inputs (a condition mask and a second timestep — in image to",
        "  video the tokens of the first latent frame are masked in and take timestep 0, in text",
        "  to video the mask is all false); the VAE decoder re-expressed as two one-frame graphs",
        "  with an explicit causal cache that return the patchified output for the host to",
        "  unpatchify, always decoded in overlapping tiles; the part of the VAE encoder that",
        "  encodes one conditioning image re-expressed as three graphs split around its middle",
        "  attention, with its parameters rounded to float16 like the decoder's; the tokenizer",
        "  converted into one table together with lookup tables for the upstream prompt cleaning.",
        "  No retraining and no fine-tuning.",
    ]
    _, borrowed = _text_encoder(manifest)
    if borrowed is None:
        lines.append(
            f"- **The text encoder** (the encoder of `{WAN_TEXT_ENCODER_UPSTREAM}`, which this"
            " checkpoint's `text_encoder` holds rounded to bfloat16, in int8) is stored in this"
            " repository."
        )
    else:
        repo, revision = borrowed
        lines += [
            "- **The text encoder is not stored here.** `karume.json` references the encoder of",
            f"  `{WAN_TEXT_ENCODER_UPSTREAM}` (this checkpoint's `text_encoder` holds the same",
            "  encoder rounded to bfloat16), converted to int8, at commit"
            f" `{revision[:16]}…` of [`{repo}`](https://huggingface.co/{repo})",
            "  (with the size and the SHA-256 of every part); that repository's `NOTICE.md` lists",
            "  the changes made to it.",
        ]
    lines += [
        "- **Precomputed embeddings and tokenizer**: the repository also ships the encoder's",
        "  outputs for a fixed set of prompts (see below), computed in bfloat16 with the umT5-XXL",
        f"  encoder of [{text.repo}](https://huggingface.co/{text.repo}) at commit"
        f" `{text.revision}`",
        "  and stored as float32, for use without the text encoder, and the tokenizer table",
        "  converted from the same commit (its tokenizer files are the same as this checkpoint's).",
    ]
    return lines


def _wan_usage(manifest: Mapping[str, Any], repo: str, card: WanCard) -> list[str]:
    """The Usage example: the minimal form that runs, with the optional settings one would commonly
    use shown commented out.

    NOTE: `fromAssets` is not shown (the same ruling as the Depth Anything card — the entry point
    for readers using HF is `fromPretrained`).
    """
    model_name = manifest["defaultModel"]
    model = default_model(manifest)
    quant = model["defaultQuant"]
    model_names = " / ".join(manifest["models"])
    quant_names = " / ".join(model["quants"])
    width, height = card.default_size
    alternatives = " or ".join(
        f"{other_width} × {other_height}"
        for other_width, other_height in card.accepted_sizes
        if (other_width, other_height) != card.default_size
    )
    return [
        "## Usage",
        "",
        "```ts",
        'import { encodePng } from "jsr:@karume/models";',
        f'import {{ frameToRgba, {card.pipeline_class} }} from "jsr:@karume/models/wan";',
        "",
        *from_pretrained(
            card.pipeline_class,
            repo,
            [
                f'  // model: "{model_name}", // default — available: {model_names}',
                f'  // quant: "{quant}", // default — available: {quant_names}',
                f'  // {WAN_TEXT_ENCODER_OPTION}: "{WAN_TEXT_ENCODER_PATHS[0]}", // default —'
                f' "{WAN_TEXT_ENCODER_PATHS[1]}" skips the text encoder (see Prompts below)',
            ],
            disposable="await using",
        ),
        "",
        "const video = await pipeline.generate({",
        '  prompt: "A cat walks on the grass, realistic style.",',
        "  seed: 42,",
        '  // negativePrompt: "low quality, blurry", // default: the official negative prompt',
        f"  // frames: {card.default_frames}, // 4n+1, {card.frames[0]} to {card.frames[1]}",
        f"  // width: {width}, height: {height}, // or {alternatives}",
        "});",
        "",
        "for (let frame = 0; frame < video.frames; frame += 1) {",
        "  const rgba = frameToRgba(video, frame);",
        "  const png = await encodePng(rgba, video.width, video.height);",
        '  await Deno.writeFile(`frame-${String(frame).padStart(2, "0")}.png`, png);',
        "}",
        "```",
        "",
        "`generate()` opens one stage at a time — the text encoder session is disposed before the",
        "transformer session is opened, and the transformer session before the VAE sessions — so",
        "no two stages are resident together. Concurrent calls are queued. Weights are fetched",
        "once and cached (verified against `karume.json`'s `size` / `sha256`). GPU memory and time",
        "per clip are listed under Resources below.",
        *card.usage_extra,
    ]


def _wan_prompts() -> list[str]:
    """The accepted set of prompts (the rules of the GPU path, and the table of fixed prompts of the
    precomputed path — the original texts are shown verbatim)."""
    gpu, precomputed = WAN_TEXT_ENCODER_PATHS
    low, high = WAN_PROMPT_TOKENS
    lines = [
        "## Prompts",
        "",
        f'With the text encoder (`{WAN_TEXT_ENCODER_OPTION}: "{gpu}"`, the default), `generate()`',
        "accepts free text. The prompt goes through a port of the upstream prompt cleaning and",
        "the umT5 tokenizer on the host, and these are rejected with `ModelInputError` before",
        "anything runs on the GPU, where the upstream pipeline would silently truncate them,",
        "repair them or map them to an unknown token instead:",
        "",
        f"- fewer than {low} or more than {high} tokens (the end token included) — an empty prompt,"
        " or",
        "  one too long;",
        "- characters outside the tokenizer's vocabulary, and special tokens such as `</s>`",
        "  written in the text;",
        "- what the cleaning would treat as an HTML character reference (write `R & D`, not",
        "  `R&D`) or as mojibake;",
        "- C1 control characters and code points unassigned in Unicode 16.0.0.",
        "",
        "When `negativePrompt` is omitted, the official negative prompt (the `negative` row",
        "below) is used, and it goes through the text encoder too.",
        "",
        f'With `{WAN_TEXT_ENCODER_OPTION}: "{precomputed}"`, the text encoder is not fetched: the'
        " repository's",
        "umT5-XXL outputs of the prompts below are used, and `generate()` accepts exactly these",
        "strings (the original text or its normalized form — `pipeline.prompts` lists both). Any",
        "other string is rejected with `ModelInputError`. The texts are shown here for reading;",
        "pass the strings from `pipeline.prompts`, since the match is exact (`ferret`, for one,",
        "starts and ends with a line break).",
        "",
        "| Name | Role | Source |",
        "| ---- | ---- | ------ |",
    ]
    lines += [
        f"| `{prompt.name}` | {prompt.role} | [source]({prompt.url}) |" for prompt in FIXED_PROMPTS
    ]
    for prompt in FIXED_PROMPTS:
        lines += ["", f"`{prompt.name}`:", "", "```text", prompt.text.strip("\n"), "```"]
    return lines


#: The range of Wan2.1's accepted frame counts that was checked end to end on the GPU (ADR 0118
#: steps 6 / 8, ADR 0119 step 10c).
_WAN21_FRAMES_CHECKED = (
    "Only 832 × 480 has been checked end to end on",
    "  the GPU: 33 and 81 frames with the precomputed embeddings, and 33 frames with the",
    "  text encoder.",
)

#: Wan2.1's comparison with upstream (the 2-step end-to-end run of the `f16` preset, and one
#: full-size transformer forward against an f64 reference — ADR 0118) and the note on tiled decode.
_WAN21_VERIFICATION = (
    "- **Against the upstream reference** (the `f16` quant, with the precomputed",
    "  embeddings): a 2-step run with injected noise is compared with diffusers on CPU in",
    "  float32 (the same f16-rounded weights, the same tiled decode), and one transformer",
    "  forward at each full size (33 and 81 frames) against a float64 reference. Differences",
    "  stay within tolerances measured on separate decision cases; the remaining gap is",
    "  float32 rounding in the GPU matrix products, not a porting difference.",
    "- **Tiled decode**: the VAE always decodes in overlapping tiles, so the frames differ",
    "  slightly from the upstream untiled decode.",
)

#: The range of Wan2.2's accepted frame counts that was checked end to end on the GPU (the 2-step
#: 17-frame sha rows and the 50-step end-to-end runs of the product class (33 frames) of ADR 0121
#: addendum "Results of step 6"; the generation script's 50-step trial runs of the addendum
#: "Accepted sizes to the 2 official sizes" (49 frames — not run with the product class; generation
#: script ≡ product was shown only for the practical preset at 33 frames); the development machine's
#: 121 frames of the addendum "The RTX 3080 Ti lanes and the full verify" (practical preset,
#: generation script, DiT-stage peak 11,361 MiB / 12,288 MiB — headroom 927 MiB); and the 121 frames
#: of the addendum "Chrome on the RTX 5070 Ti" (gpu-lab)). The 3080 Ti peak is nvidia-smi's value
#: for the whole GPU, and 0.9 GiB is the difference from the physical capacity — what bites first in
#: Deno is the ceiling on total allocation (97% of the driver's budget, moving over time), so the
#: real headroom can be thinner (the Wan2.2 section of docs/limitations.md — an estimate). The
#: ceiling moves over time, so the card does not state it. On the B570 (Deno's total-allocation
#: ceiling 9,600 MiB) the checked lengths go up to 57 frames (the trial table of the addendum
#: "Accepted sizes to the 2 official sizes" — 49 for both presets, 57 for the practical preset only,
#: both with the generation script). Longer clips are stated flatly as **unsupported**, as settled
#: by ADR 0121 addendum "Step 10 — accepting 121 frames". The reason they are unsupported is that
#: they were not run on the B570; whether they would fit is an estimate: the table "Estimate of
#: capacity and time" in Context estimated only 2 points — at 81 frames (S = 18,480) the practical
#: preset does not fit, and at 121 neither preset fits (for 121, the DiT stage's diagnostic value
#: (weights + backing) is 10.64 GiB even on the 3080 Ti, above the ceiling). 61 to 77 and 85 to 117
#: have no estimate either, so the card says "unknown" (it does not say they are expected not to
#: fit). NOTE: a check keeps the card from copying 2.1's default of 81
#: (`test_it_carries_none_of_the_wan21_sizes_or_frame_counts`), so 81 is written as "the length
#: after 77 frames" (with 4n+1 the one after 77 is 81 — not a boundary the estimate found). The
#: accepted set is not per machine, so admission does not reject it, and the card says that a clip
#: that does not fit fails partway through the run (an OOM errorScope, or device lost near the
#: ceiling). It also says that the B570 rows of the resources table stop at 49 (57 is not in the
#: table). 704×1280 has only the 2-step × 17-frame sha rows (every 50-step end-to-end run is
#: 1280×704). All of this is about T2V (the checked range of I2V is stated by
#: {@link _WAN22_I2V_INPUTS} and {@link _WAN22_VERIFICATION}).
_WAN22_FRAMES_CHECKED = (
    "Text to video has been checked end to end",
    "  on the GPU: 17 frames at both sizes in 2-step runs, and 33 frames at 1280 × 704 in",
    "  50-step runs. 49 frames at",
    "  1280 × 704 ran 50 steps through the same pipeline stages, driven by a development script",
    "  rather than the pipeline class.",
    "  121 frames at 1280 × 704 (the `f16+dit8-a8-attn8-s16` quant) ran 50 steps the same way on",
    "  an NVIDIA GeForce RTX 3080 Ti (12 GiB) in Deno, and in Chrome on an NVIDIA GeForce",
    "  RTX 5070 Ti. 704 × 1280 has been run only in the 2-step runs at 17 frames.",
    "  On the RTX 3080 Ti, the memory in use on the whole GPU (read with nvidia-smi) peaked at",
    "  11.09 GiB during the transformer stage, about 0.9 GiB below the card's 12 GiB; by the",
    "  runtime's own count, that stage allocated 10.64 GiB of weights and buffers. The headroom",
    "  Deno actually has can be thinner, since Deno stops allocating at a ceiling below the",
    "  card's size that varies over time, and on a 12 GiB GPU shared with other programs",
    "  121 frames may not fit (an estimate — not run).",
    "  On a GPU with about 10 GB, such as the Intel Arc B570 (where Deno can allocate about",
    "  9.4 GiB in total), clips up to 57 frames have been checked: the longest 50-step runs",
    "  there were 49 frames with both quants and 57 frames with the `f16+dit8-a8-attn8-s16`",
    "  quant, driven by the development script (its rows under Resources stop at 49 frames).",
    "  Longer clips are not supported there, because none of them has been run on the B570. A",
    "  memory estimate made before the runs covers only two of those lengths and puts both",
    "  beyond that ceiling: the next length after 77 frames with the `f16+dit8-a8-attn8-s16`",
    "  quant, and 121 frames with both quants (the 10.64 GiB the transformer stage allocated at",
    "  121 frames on the RTX 3080 Ti is above it too). The other lengths were not estimated, so",
    "  whether they fit is not known. The accepted set does not depend on the GPU, so such a",
    "  request is still accepted, and if it does not fit it fails during the run — with an",
    "  out-of-memory error (`GpuOutOfMemoryError`), or a lost device near the limit — rather",
    "  than with `ModelInputError`, and the time spent on the stages before it is lost.",
)

#: Wan2.2's comparison with upstream (the reference preset's 2-step end-to-end run — ADR 0121
#: addendum "Results of step 6"; one full-size transformer forward against an f64 reference — the r
#: check of the addendum "Results of step 2" (S = 8,190 / 7,920)) and the note on tiled decode
#: (blended in patchified space, then unpatchified — decision 6). The practical preset's on-machine
#: A/B check (step 7) is not measured, so the card says it was not compared numerically and states
#: only the grounds for making it the default (12 visual clips).
#:
#: The I2V items (the addenda "Results of step 9a" and "Results of step 9b", and the test constants
#: — the encoder band `WAN_I2V_ENCODER_BAND` 7.8e-5
#: (`packages/models/tests/helpers/wan-i2v-image.ts` — worst of the deciding cases 1.550e-5 × 5, the
#: reference's largest absolute value 2.89 to 4.04, the 5 fault injections at 15,000 times the band
#: or more), the end-to-end bands `LATENT_RATIO_BANDS` 2.9e-3 / 3.5e-3, `FINAL_RATIO_BAND` 3.9e-3,
#: `FRAME_RATIO_BAND` 8.9e-3 (`e2e_wan_ti2v_i2v_pipeline_test.ts` — the worst of the 2 deciding
#: cases × 5, the acceptance case ferret inside the band), the 5 fault injections at 138 to 358
#: times the band on latents.1, and T2V's latents.1 band 9.5e-4 (`e2e_wan_ti2v_pipeline_test.ts`)).
#: Why the bands are wider than T2V's is the isolation of the addendum "Results of step 9b" (the DiT
#: amplifies the maxAbs 1.26e-5 of cat-dog-baking's conditioning latent about 250 times on the
#: conditioning tokens; with the reference's conditioning swapped in, it returns to T2V's order).
#: The ratio to the T2V band is stated for latents.1 only (latents.0 is of a different order from
#: T2V's band 3.6e-5 and cannot be generalized as "about 4 times"; the frame band is narrower for
#: I2V). Every sha row is on the RTX 3080 Ti, on the precomputed path (default lane = boxing-cats of
#: the reference and practical presets; opt-in = cat-dog-baking / ferret / 704×1280). The 50-step
#: I2V (1280×704, 33 frames) has 2 opt-in sha rows, for the reference and the practical presets (the
#: practical one is pinned to differ from the reference by at least 1 bit — `136cee60` / `28ab44fe`,
#: `KARUME_WAN_TI2V_I2V_FULL=1`). The practical preset's quality numbers (the A/B and visual check
#: of step 7) were measured on T2V. The preset comparison for I2V was done in step 9d (A/B 0.046, 12
#: visual pairs without breakdown — 2026-10-10), and the I2V part is written in the same sentence.
#: The card claims no I2V quality relative to upstream.
#: NOTE: the size of the source image (832×480) is not written — it would trip the check that keeps
#: 2.1's sizes out (`test_it_carries_none_of_the_wan21_sizes_or_frame_counts`).
_WAN22_VERIFICATION = (
    "- **Text to video against the upstream reference** (the `f16+dit8` quant, with the",
    "  precomputed embeddings): a 2-step run at 1280 × 704 with 17 frames and injected noise is",
    "  compared with diffusers on CPU in float32 (the same int8 transformer weights, the same",
    "  f16-rounded VAE weights and tiled decode), and single transformer forwards of up to",
    "  8,190 tokens (1280 × 704 with 33 frames among them) against a float64 reference.",
    "  Differences stay within tolerances measured on separate decision cases. The",
    "  `f16+dit8-a8-attn8-s16` quant is compared with `f16+dit8` on the same GPU: the first",
    "  step's latent at 1280 × 704 with 33 and 121 frames stays within twice the measured",
    "  relative error, differs from `f16+dit8`, and repeats bit for bit. It was the default until",
    "  2026-10-23, chosen after a visual check of 12 clips (3 prompts × seeds 42–45, 1280 × 704",
    "  with 33 frames, 50 steps) on an NVIDIA GeForce RTX 3080 Ti in Deno on 2026-10-06, and the",
    "  same 12 clips side by side with `f16+dit8` showed no clear degradation on 2026-10-09. On",
    "  image to video the same first-step check holds at 1280 × 704 with 33 frames (one",
    f"  image, `crop`, an opt-in case: relative RMS error {WAN22_PRACTICAL_QUANT_I2V_ERROR_33},",
    "  within twice the measured error), and 12 clips (3 images × seeds 42–45, 1280 × 704 with",
    "  33 frames, 50 steps) side by side with `f16+dit8` showed no clear degradation on",
    f"  2026-10-10 either. `{WAN22_ONLINE_QUANT_ERROR[0]}`, the default from 2026-10-23 until",
    f"  `{WAN22_INT8_VAE_QUANTS[1]}` took over the same day, was compared the same way on text to",
    "  video at 1280 × 704 with 33 frames, under Node with Dawn on the same GPU: relative RMS",
    f"  error {WAN22_ONLINE_QUANT_ERROR[2]}, within the same tolerance. Its 121-frame comparison,",
    "  its visual check and its image-to-video comparison have not been done yet.",
    f"  `{WAN22_INT8_VAE_QUANTS[1]}`, the default since 2026-10-23, keeps that transformer and",
    "  changes only the VAE decoder (int8 weights, int8 activations in its 3-D convolutions): at",
    "  the first decoded frame its relative RMS error is 0.013 against the same int8 weights",
    f"  computed in f32 (`{WAN22_INT8_VAE_QUANTS[0]}`; the gate is 0.026) and 0.032 against the",
    f"  f16 decoder (`{WAN22_INT8_VAE_QUANTS[0]}` itself 0.030), and 8 clips (2 prompts × seeds",
    "  42–45, 1280 × 704 with 33 frames, 50 steps) side by side with",
    f"  `{WAN22_ONLINE_QUANT_ERROR[0]}` showed no visible difference on 2026-10-23.",
    "- **Image to video against the upstream reference** (the `f16+dit8` quant, with the",
    "  precomputed embeddings, on an NVIDIA GeForce RTX 3080 Ti): the resized image and the",
    "  encoder input match Pillow and the upstream preprocessing bit for bit. The encoded image",
    "  is compared with the upstream untiled VAE encode (diffusers on CPU in float32) for three",
    "  images, each at the two accepted sizes and at a small 256 × 160 size outside them: the",
    "  largest absolute difference of the normalized latent stays within 7.8e-5, where the",
    "  latent's values reach about 3 to 4. A 2-step run at 1280 × 704",
    "  with 9 frames, injected noise and the same three images is compared with diffusers'",
    "  `WanImageToVideoPipeline` on CPU in float32 (the same int8 transformer weights, the same",
    "  f16-rounded VAE weights and tiled decode). The largest difference, relative to the",
    "  reference's largest value, stays within 2.9e-3 for the latent after the first step,",
    "  3.5e-3 after the second, 3.9e-3 for the latent passed to the VAE and 8.9e-3 for the",
    "  frames, and the encoded image inside the run stays within the encoder's 7.8e-5. Each",
    "  tolerance is five times the worst of two deciding images, and the third image is",
    "  checked against it.",
    "- **Why the image-to-video latent tolerances are wider** than the text-to-video ones",
    "  (3.5e-3 against 9.5e-4 after the second step): the difference gathers in the tokens of",
    "  the conditioning frame. On the image that sets these tolerances, the encoder's output",
    "  differs from the reference by 1.26e-5 at most (well inside its own tolerance), and the",
    "  transformer amplifies that about 250 times on those tokens. With the reference's encoded",
    "  image put in its place, the difference falls back to the text-to-video size, so it does",
    "  not come from the transformer's own precision.",
    "- **Injected faults**: five deliberate faults in the conditioning (an all-false condition",
    "  mask, the mask shifted by one token, the two timestep inputs swapped, the first frame",
    "  not replaced, the encoded image not normalized) put the latent after the second step 138",
    "  to 358 times its tolerance away, and each also fails a direct check of the input it",
    "  breaks. Five faults in the encoder land 15,000 times its tolerance away or more.",
    "- **Pinned image-to-video runs**: the same request with the same `image` and `fit`",
    "  repeats bit for bit, and the SHA-256 of the frames is pinned for one image at",
    "  1280 × 704 with 9 frames (2 steps) with both quants — the `f16+dit8-a8-attn8-s16` result",
    "  is also required to differ from the `f16+dit8` one, so the quant is known to be in use —",
    "  and, in an opt-in set, for the other two images and for 704 × 1280 with the `f16+dit8`",
    "  quant. In the same opt-in set, 50-step image-to-video runs at 1280 × 704 with 33 frames",
    "  are pinned with both quants on the NVIDIA GeForce RTX 3080 Ti, the",
    "  `f16+dit8-a8-attn8-s16` result again required to differ from the `f16+dit8` one.",
    "- **Tiled decode**: the VAE always decodes in overlapping tiles, blended in the patchified",
    "  space before the unpatchify, so the frames differ slightly from the upstream untiled",
    "  decode.",
)

#: The I2V items of Wan2.2's accepted inputs (`image` / `fit` — `Ti2vGenerateRequest` in
#: `packages/models/src/wan/ti2v-pipeline.ts`, the check `planWanRequest` in `plan.ts`, the
#: preprocessing in `i2v-preprocess.ts`). The size is chosen from the accepted set with the official
#: comparison `max(r / rc, rc / r)`, ties going to landscape (`selectWanI2vSize`) — with an accepted
#: set of 2 sizes this equals "width ≥ height → 1280×704". An explicit field must name a size in the
#: accepted set, else `ModelInputError`. The official code also picks sizes outside the accepted set
#: (1248×704 for 16:9). LANCZOS is a verbatim port of Pillow 12.3.0
#: (`packages/models/src/image/lanczos.ts` — every uint8 matches on the fixture and the sweep). A
#: misspelled `fit` is a plain `Error`; `fit` without `image` is `ModelInputError` (ADR 0121
#: addendum "Results of step 9b"). The pixels are used as given (no color-space conversion).
_WAN22_I2V_INPUTS = (
    "- **image**: the first frame for image to video, as decoded RGB8 pixels",
    "  `{ data, width, height }` (`data` a `Uint8Array` with 3 bytes per pixel, row by row; any",
    "  size), used as given (no color-space conversion). Without `width` / `height`, the",
    "  accepted size closest to the image's aspect ratio is used — 1280 × 704 when the image is",
    "  at least as wide as it is tall, 704 × 1280 otherwise. A given `width` and/or `height` must",
    "  match one accepted size, and that size is used. The official Wan2.2 code also picks",
    "  sizes outside this set (1248 × 704 for a 16:9 image), so the same image can lose more to",
    "  the crop here.",
    '- **fit** (only with `image`): `"crop"` (the default, as in the official Wan2.2 code)',
    "  scales the image to cover the output size, keeping its aspect ratio, and cuts out the",
    '  center; `"stretch"` (as in diffusers\' `WanImageToVideoPipeline`) resizes the width and',
    "  the height separately. Both resize with a port of Pillow's LANCZOS filter that gives the",
    "  same bytes as Pillow 12.3.0 for the same RGB8 input. `fit` without `image` throws",
    "  `ModelInputError`; a value other than these two throws a plain `Error` (the one",
    "  exception to the sentence below).",
)

#: The I2V example added to Wan2.2's Usage (decoding the image is the caller's job — `generate()`
#: takes RGB8. The Deno example uses fast-png / jpeg-js (`formatAsRGBA: false` gives RGB directly —
#: `examples/wan/decode-image.ts`), the browser createImageBitmap and a canvas (which yield RGBA)).
#: The differences of the decoding step from the reference (EXIF orientation, alpha, differences
#: between JPEG decoders) copy the same facts as `examples/wan/README.md` and
#: `tools/gpu-lab/README.md`. Deno's I2V was run on the precomputed path only (the combination with
#: the GPU umT5 has run neither in e2e nor in the 50-step sha rows — ADR 0121 addendum "Results of
#: step 9b", `e2e_wan_ti2v_i2v_pipeline_test.ts`). The combination with the GPU umT5 finished only
#: in Chrome on the RTX 5070 Ti (gpu-lab, 2026-10-10) (the docstring of {@link _wan22_overview}).
#: The stage order is `pipeline.ts`'s "(encoder →) text → DiT → VAE" (the text stage only on the
#: `"gpu"` path).
_WAN22_I2V_USAGE = (
    "",
    "For image to video, pass the first frame as `image`. Decoding the image file is up to the",
    "caller: on Node or Deno an image decoder such as fast-png and jpeg-js works, and in a",
    "browser `createImageBitmap` and a canvas work. `generate()` takes 8-bit",
    "RGB, so convert what the decoder gives: a canvas gives RGBA, and so does jpeg-js by",
    "default (`formatAsRGBA: false` gives RGB) — drop the alpha channel.",
    "",
    "The bit-for-bit match with Pillow (Accepted inputs) starts from the RGB8 pixels, so the",
    "decoding decides whether a file gives the reference's input. The official Wan2.2 code",
    "reads the file with Pillow, drops the alpha channel without compositing it over a",
    "background, and does not apply the EXIF orientation. `createImageBitmap` applies the",
    "orientation by default, and a canvas can lose precision in partly transparent pixels (it",
    "stores premultiplied alpha). For PNG, fast-png gives the same pixels as Pillow; JPEG",
    "decoders differ slightly, so jpeg-js and a browser can give pixels a little different from",
    "Pillow's for the same JPEG file.",
    "",
    "```ts",
    "// `rgb`: the decoded image, 3 bytes (R, G, B) per pixel, row by row.",
    "const clip = await pipeline.generate({",
    '  prompt: "A cat walks on the grass, realistic style.",',
    "  image: { data: rgb, width: imageWidth, height: imageHeight },",
    "  seed: 42,",
    '  // fit: "crop", // default — "stretch" resizes without keeping the aspect ratio',
    "  // width / height: omitted — the accepted size closest to the image's aspect ratio",
    "});",
    "```",
    "",
    "With an `image`, the VAE encoder stage runs first and is disposed before the next stage",
    "(the text encoder, or the transformer with the precomputed embeddings) is opened. In Deno,",
    "image to video has so far been run only with the precomputed embeddings; with the text",
    "encoder on the GPU (the default in this example) it has been run only in Chrome (see What",
    "is this).",
)


def _wan_inputs(card: WanCard) -> list[str]:
    """The accepted inputs and the explanation of the numeric checks (bit identity, comparison with
    the reference)."""
    sizes = " or ".join(f"{width} × {height}" for width, height in card.accepted_sizes)
    low, high = card.frames
    first, *rest = card.frames_checked
    return [
        "## Accepted inputs",
        "",
        "- **prompt** / **negativePrompt**: see Prompts above.",
        f"- **size**: {sizes}.",
        f"- **frames**: 4n+1 from {low} to {high}. {first}",
        *rest,
        "- **steps** ≥ 1, **guidance** ≥ 1 (1 turns classifier-free guidance off and the negative",
        "  prompt is then rejected), **shift** > 0.",
        "- **seed** (the host noise generator — not torch's `randn`) or the initial noise as",
        "  `latents`.",
        *card.inputs_extra,
        "",
        "Anything outside these sets throws `ModelInputError` before any weight reaches the GPU.",
        "",
        "## Determinism and verification",
        "",
        "- **Bit-identical runs**: the same request (prompt, seed or latents, knobs) produces the",
        "  same bytes on the same GPU and driver. Release verification pins the SHA-256 of the",
        "  8-bit frames per test environment and fails on any change — the check is never relaxed",
        "  to a tolerance.",
        *card.verification,
    ]


def _wan21_resources(manifest: Mapping[str, Any]) -> list[str]:
    """Guidance on execution resources (ADR 0089 decision 3 — the allocations of intermediate
    tensors and the binding sizes are not counted by the manifest's `requiredLimits`: **facts
    that are not in the manifest**. The same treatment as the BiRefNet card).

    The numbers are measured 50-step end-to-end runs on the B570 in Deno — 33 frames on 2026-10-02
    (the acceptance of ADR 0118 step 6), 81 frames on 2026-10-03 (step 8): VRAM is the per-stage
    peak of fdinfo's `drm-total-vram0`; the times are one transformer pass (batch 1) and the whole
    tiled VAE decode.
    1 pass × 100 + VAE agrees with the end-to-end run (33 frames: 16.8 × 100 + 129 ≈ 1,809 s /
    1,811 s; 81 frames: 68.6 × 100 + 315 ≈ 7,175 s / 7,173 s). The intermediate tensor sizes are
    computed from the shapes (the FFN's `[S, 8960]` f32 is 503,193,600 B at S = 14,040 and
    1,174,118,400 B at S = 32,760; one row block of the self-attention scores is 2,146,435,200 B at
    81 frames — inside the B570's binding limit of 2,147,483,644 B).

    The text stage numbers (GPU path) are measured in ADR 0119 addendum "The GPU check of step 10c
    and the results of step 10d-4" (one 2-step 33-frame end-to-end run, and one forward of umT5
    alone). The per-preset transformer numbers are as the provenance of `WAN_QUANT_TRANSFORMER`
    says.

    MUST: no numbers for conditions that were not measured — no estimates for the other sizes and
    frame counts of the accepted set or for other GPUs. The browser was checked only with one
    81-frame end-to-end run of the `f16` preset on the asset path in Chrome on the RTX 5070 Ti (ADR
    0118 addendum "Results of step 9"), so everything else is stated as unchecked.
    """
    for name, model in manifest["models"].items():
        if WAN_RESOURCE_QUANT not in model["quants"]:
            raise ValueError(
                f"quant '{WAN_RESOURCE_QUANT}', whose execution resources were measured, is not a"
                f" preset of model '{name}' (presets: {sorted(model['quants'])}) — no numbers are"
                " stated for a preset that was not measured"
            )
    # The preset order stays the manifest's. A preset missing from the table gets no estimate and is
    # stated as not measured.
    seats = dict.fromkeys(
        quant for model in manifest["models"].values() for quant in model["quants"]
    )
    measured = [quant for quant in seats if quant in WAN_QUANT_TRANSFORMER]
    unmeasured = [f"`{quant}`" for quant in seats if quant not in WAN_QUANT_TRANSFORMER]
    practical, reference, error_33, error_81 = WAN_PRACTICAL_QUANT_ERROR
    return [
        "## Resources",
        "",
        "Measured on an Intel Arc B570 in Deno 2.9.6, with the"
        f" `{WAN_RESOURCE_QUANT}` quant at 832 × 480,",
        "50 steps and guidance 5 (classifier-free guidance runs two transformer passes per step,",
        "so 100 passes); 33 frames on 2026-10-02 and 81 frames on 2026-10-03:",
        "",
        "| Frames | Transformer peak | VAE peak | Pass   | VAE decode | Clip        |",
        "| ------ | ---------------- | -------- | ------ | ---------- | ----------- |",
        "| 33     | 6.19 GiB         | 3.32 GiB | 16.8 s | 129 s      | ~30 minutes |",
        "| 81     | 7.31 GiB         | 3.78 GiB | 68.6 s | 315 s      | ~2 hours    |",
        "",
        "Pass is one transformer pass (batch 1); VAE decode is the tiled decode of the whole clip.",
        "These runs used the precomputed embeddings.",
        "",
        f'With the text encoder (`{WAN_TEXT_ENCODER_OPTION}: "{WAN_TEXT_ENCODER_PATHS[0]}"`),'
        " measured on 2026-10-03 in a 2-step run at",
        "832 × 480 and 33 frames: the text encoder stage (the umT5-XXL encoder run for the",
        "positive and the negative prompt, including building its session) took 10.4 s and",
        "peaked at 6.30 GiB; after it was disposed, 0.08 GiB more than before the stage remained",
        "allocated. In the same run the transformer stage peaked at 5.94 GiB and the VAE stage at",
        "3.32 GiB. One forward of the text encoder alone takes 0.14 s (8 tokens) to 1.73 s",
        "(488 tokens). A 50-step run through the text encoder has not been done yet.",
        "",
        "Per quant, at 832 × 480 (the 50-step clips with the precomputed embeddings and",
        "guidance 5):",
        "",
        "| Quant | Frames | Pass | Transformer allocation | 50-step clip |",
        "| ----- | ------ | ---- | ---------------------- | ------------ |",
        *(
            f"| `{quant}` | {frames} | {forward} | {allocation} | {clip} |"
            for quant in measured
            for frames, forward, allocation, clip in WAN_QUANT_TRANSFORMER[quant]
        ),
        "",
        "Pass and allocation here are of the transformer run alone (one pass, batch 1); the",
        "allocation is the runtime's own count of what it allocated, not the driver's total.",
        "",
        *(
            [f"The other quants ({' / '.join(unmeasured)}) have not been measured yet.", ""]
            if unmeasured
            else []
        ),
        *(
            [
                f"- **Quality of `{practical}`**: after the first step its latent differs from",
                f"  `{reference}`'s (the same int8 weights, computed in float32) by a relative",
                f"  RMS error of {error_33} at 33 frames and {error_81} at 81 frames, mostly",
                "  from the int8 attention. Side by side with `f16` on twelve 50-step clips at",
                "  33 frames (seeds 42 to 45 with the three fixed prompts), no clear degradation",
                "  was seen.",
            ]
            if practical in seats and reference in seats
            else []
        ),
        *_download_and_memory(manifest),
        "- **Storage buffer size**: at these sizes some of the transformer's intermediate tensors",
        "  are larger than WebGPU's default `maxStorageBufferBindingSize` (128 MiB) — the",
        "  feed-forward activation `[S, 8960]` in float32 alone is about 480 MiB at 33 frames",
        "  (S = 14,040) and about 1.09 GiB at 81 frames (S = 32,760), and one row block of the",
        "  self-attention scores at 81 frames is 2.00 GiB, just under the 2 GiB binding limit the",
        "  adapter reports.",
        "  The runtime requests the adapter's own limits, which Deno grants on the B570; in a",
        "  browser the environment has to grant the adapter's limits as well. Chrome on the",
        "  RTX 5070 Ti granted them (2 GiB buffers) and ran the 81-frame clip above; other",
        "  browsers and GPUs have not been checked.",
        *_undeclared(manifest),
    ]


def _wan22_resources(manifest: Mapping[str, Any]) -> list[str]:
    """Guidance on Wan2.2's execution resources (ADR 0089 decision 3 — the same treatment as 2.1's
    {@link _wan21_resources}).

    The numbers are as the provenance of {@link WAN22_RESOURCES} says. The VAE-stage peak is written
    only from end-to-end runs at accepted frame counts: 33 frames is 4.360 / 4.361 GiB from the
    product class's runs (reference / practical preset — addendum "Results of step 6", record
    `outputs/bench/karume-wan2.2/2026-10-05_stage6-gpu/g1-full.log`), 49 frames is 4.513 / 4.512 GiB
    from the generation script's runs (`vramPeaksGiB` in the record
    `outputs/verify/deno-intel-graphics-bmg-g21/2026-10-05_wan22-visual/`). The range 4.36 to 4.52
    GiB in the note of the trial table includes the 57-frame trial outside the accepted set, so it
    is not used.

    The intermediate tensor sizes are computed from the shapes (the FFN's `[S, 14336]` f32 is
    454,164,480 B at S = 7,920, 656,015,360 B at S = 11,440 and 1,564,344,320 B at S = 27,280 —
    `ffn_dim` from the pinned `transformer/config.json`). The largest binding at 121 frames, about
    1.96 GiB (a row block of the self-attention scores), is the value from ADR 0121 addendum
    (2026-10-07) "Step 10 — accepting 121 frames".

    The 121-frame row and the text encoder stage's 9.1 s are the RTX 3080 Ti development machine's
    run (the provenance of {@link WAN22_RESOURCES}). The browser numbers are the wall time 2,417.5 s
    of the same request in Chrome on the RTX 5070 Ti and the adapter / device
    `maxStorageBufferBindingSize` 2,147,483,644 B (ADR 0121 addendum "Chrome on the RTX 5070 Ti") —
    both are stated as values of another GPU and runtime.

    MUST: no numbers for conditions that were not measured (the same discipline as 2.1). Unmeasured
    combinations of preset, frame count, GPU and browser are stated as not measured yet. The
    practical preset's quality is given as the relative RMS error of step 7's on-machine A/B check
    ({@link WAN22_PRACTICAL_QUANT_ERROR}) and the visual judgement, both stated as measured on T2V
    (the preset comparison for I2V is step 9d — no I2V quality is claimed).

    The I2V resources are not put in the table (the table is all T2V). Only what was measured is
    written: the peak VRAM of the encoder alone in its own process, 1,421 MiB (ADR 0121 addendum
    "Results of step 9a" — nvidia-smi, RTX 3080 Ti), and the wall time of 50-step I2V (1280×704, 33
    frames, precomputed — reference preset 1,395.3 s, practical preset 723.5 s, encoder stage 0.5 /
    0.8 s; the wall time in `2026-10-09_wan-ti2v-i2v-pipeline-full/results.json` under
    `outputs/verify/deno-nvidia-geforce-rtx-3080-ti/`. The ADR's 23 min 20 s / 12 min 8 s are the
    `elapsedMs` of the same runs). Those runs
    did not read VRAM (n/a), so the peak of one I2V clip is stated as not measured.
    """
    seats = dict.fromkeys(
        quant for model in manifest["models"].values() for quant in model["quants"]
    )
    measured = [quant for quant in seats if quant in WAN22_RESOURCES]
    if not measured:
        raise ValueError(
            "no quant whose execution resources were measured"
            f" ({sorted(WAN22_RESOURCES)}) is among the manifest's presets ({list(seats)}) — no"
            " numbers are stated for a preset that was not measured"
        )
    unmeasured = [f"`{quant}`" for quant in seats if quant not in WAN22_RESOURCES]
    practical, reference, error_33, error_121 = WAN22_PRACTICAL_QUANT_ERROR
    online, _, online_error_33 = WAN22_ONLINE_QUANT_ERROR
    int8_vae_reference, int8_vae_practical = WAN22_INT8_VAE_QUANTS
    online_row = online in measured
    return [
        "## Resources",
        "",
        "The 33- and 49-frame rows were measured on an Intel Arc B570 in Deno 2.9.6 on 2026-10-05,",
        "at 1280 × 704 with 50 steps, shift 5 and guidance 5 (classifier-free guidance runs two",
        f"transformer passes per step, so 100 passes); the `{practical}` 121-frame row at the",
        "same size, steps, shift and guidance on an NVIDIA GeForce RTX 3080 Ti (12 GiB) in Deno",
        "2.9.6 on 2026-10-06"
        + (f"; the `{online}` row is described below the table:" if online_row else ":"),
        "",
        "| Quant | Frames | Pass | Transformer peak | Transformer stage | VAE decode | Clip |",
        "| ----- | ------ | ---- | ---------------- | ----------------- | ---------- | ---- |",
        *(
            f"| `{quant}` | {frames} | {forward} | {peak} | {stage} | {decode} | {clip} |"
            for quant in measured
            for frames, forward, peak, stage, decode, clip in WAN22_RESOURCES[quant]
        ),
        "",
        "Pass is one transformer pass (batch 1, 7,920 tokens) timed on its own on 2026-10-04"
        + (f", except in the `{online}` row." if online_row else "."),
        "For the B570 rows, Transformer peak is the total allocation during the transformer stage",
        "(the driver's fdinfo). The stage and the tiled decode of the whole clip are timed within",
        "the 50-step run. The 33-frame runs went",
        f"through `{WAN22_PIPELINE_CLASS}`; the 49-frame runs drove the same pipeline stages",
        "from a development script instead, whose transformer peak read 0.12 GiB lower than the",
        "class's in the one run measured both ways (33 frames). The VAE stage peaked at 4.36 GiB",
        "at 33 frames and 4.51 GiB at 49 frames with either quant.",
        "",
        f"The `{practical}` 121-frame row comes from a different GPU and a different reading, so",
        "it is not",
        "comparable with the B570 rows: it was driven by the same development script with another",
        "prompt (`cat-dog-baking`, where the B570 rows used `boxing-cats`) and the text encoder on",
        "the GPU (that stage took 9.1 s and is included in Clip); its transformer peak is the",
        "memory in use on the whole GPU as read by nvidia-smi, not the driver's fdinfo; and its",
        "times include thermal throttling (the GPU reached 93 °C). The 121-frame clip has not",
        "been run for 50 steps with the `f16+dit8` quant.",
        "",
        *(
            [
                f"The `{online}` row is one transformer pass at 121 frames (batch 1, 27,280",
                "tokens) timed on its own on the same NVIDIA GeForce RTX 3080 Ti under Node with",
                "Dawn on 2026-10-23 (the second run of the same input); its Transformer peak is",
                "the runtime's own count of what the transformer allocated in that run (neither",
                "nvidia-smi nor the driver's fdinfo). That quant has not been run for 50 steps, so",
                "its stage, decode and clip times are not given, and it has not been measured at",
                "33 or 49 frames.",
                "",
            ]
            if online_row
            else []
        ),
        *(
            [f"The other quants ({' / '.join(unmeasured)}) have not been measured yet.", ""]
            if unmeasured
            else []
        ),
        f'The text encoder stage (`{WAN_TEXT_ENCODER_OPTION}: "{WAN_TEXT_ENCODER_PATHS[0]}"`) has'
        " been measured with this",
        "distribution only in the 121-frame run above. In a browser, Chrome on an NVIDIA GeForce",
        "RTX 5070 Ti finished the same 121-frame request in 2,417.5 s (40.3 minutes) — a different",
        "GPU and runtime from the rows above.",
        "",
        "All of the above is text to video. Image to video adds the VAE encoder stage, which runs",
        "once before the others and is released before the next stage opens. Run on its own on",
        "the RTX 3080 Ti in Deno 2.9.6, the encoder kept the whole process at a peak of 1,421 MiB",
        "(nvidia-smi) while encoding images at 1280 × 704, 704 × 1280 and 256 × 160.",
        *(
            [
                "On the same GPU on 2026-10-09, 50-step image-to-video runs at 1280 × 704 with 33",
                "frames and the precomputed embeddings took 1,395.3 s with"
                f" `{reference}` and 723.5 s",
                f"with `{practical}` in all, the encoder stage under 1 s of it; the GPU memory of",
                "those runs was not read, so an image-to-video clip's peak has not been measured.",
            ]
            if practical in seats and reference in seats
            else [
                "Image-to-video clips have not been measured for this table.",
            ]
        ),
        "",
        *(
            [
                f"- **Quality of `{online}`**: after the first step its latent differs from",
                f"  `{reference}`'s (the same int8 weights, computed in float32) by a relative RMS",
                f"  error of {online_error_33} at 33 frames (text to video, on the NVIDIA GeForce",
                "  RTX 3080 Ti under Node with Dawn on 2026-10-23); at 121 frames it has not been",
                f"  measured, because `{reference}`'s 121-frame step does not fit that 12 GiB GPU",
                "  under Node with Dawn. The visual check of twelve 50-step clips was made with",
                f"  `{practical}` only — the same check of `{online}` is still pending — and its",
                "  image-to-video error has not been measured.",
            ]
            if online in seats and reference in seats
            else []
        ),
        *(
            [
                f"- **Quality of `{practical}`**: after the first step its latent differs from",
                f"  `{reference}`'s (the same int8 weights, computed in float32) by a relative",
                f"  RMS error of {error_33} at 33 frames and {error_121} at 121 frames. Side",
                f"  by side with `{reference}` on twelve 50-step clips at 1280 × 704 with 33",
                "  frames (seeds 42 to 45 with the three fixed prompts), no clear degradation",
                "  was seen. Both were measured with text to video; on image to video the",
                f"  first-step error is {WAN22_PRACTICAL_QUANT_I2V_ERROR_33} at 33 frames (one",
                "  image, `crop`) and twelve 50-step clips (3 images × seeds 42–45) side by side",
                "  showed no clear degradation either.",
            ]
            if practical in seats and reference in seats
            else []
        ),
        *(
            [
                f"- **`{int8_vae_reference}` and `{int8_vae_practical}`**: both store the VAE",
                "  decoder's weights as int8 (one scale per output channel) as well as the",
                "  transformer's; `conv8` computes the decoder's 3-D convolutions on per-pixel",
                "  int8 activations (integer dot products). On the NVIDIA GeForce RTX 3080 Ti",
                "  under Node with Dawn (2026-10-23, 1280 × 704 with 33 frames, 50 steps, 8 clips)",
                f"  `{int8_vae_practical}` decoded the clip in 87 s against 168 s for `{online}`",
                "  (the whole clip 533 s against 611 s) with a GPU-wide peak of 6.2 GiB",
                "  (nvidia-smi).",
            ]
            if int8_vae_reference in seats and int8_vae_practical in seats
            else []
        ),
        *_download_and_memory(
            manifest,
            peaks="the B570 rows' peaks are of the total allocation (the driver's fdinfo), "
            + ("" if online_row else "and ")
            + f"the `{practical}` 121-frame row's is the memory in use on the whole GPU"
            " (nvidia-smi)"
            + (f", and the `{online}` row's is the runtime's own count" if online_row else ""),
        ),
        "- **Storage buffer size**: at these sizes some of the transformer's intermediate tensors",
        "  are larger than WebGPU's default `maxStorageBufferBindingSize` (128 MiB) — the",
        "  feed-forward activation `[S, 14336]` in float32 alone is about 433 MiB at 33 frames",
        "  (S = 7,920), about 626 MiB at 49 frames (S = 11,440) and about 1.46 GiB at 121 frames",
        "  (S = 27,280); the largest binding at 121 frames is about 1.96 GiB (a block of rows of",
        "  the self-attention scores). The runtime requests the adapter's own limits, which Deno",
        "  grants on the B570 and the RTX 3080 Ti; in a browser the environment has to grant the",
        "  adapter's limits as well. Chrome on the RTX 5070 Ti granted them (bindings of up to",
        "  2,147,483,644 bytes) and ran the 121-frame clip above; other browsers and GPUs have not",
        "  been checked.",
        *_undeclared(manifest),
    ]


def _download_and_memory(
    manifest: Mapping[str, Any],
    peaks: str = "the peaks are of the total allocation (the driver's fdinfo)",
) -> list[str]:
    """The 2 items of the resources note on the text_encoder download size (from the manifest) and
    on how to read the VRAM peaks.

    `peaks` is the sentence on how the peaks were measured. 2.1 keeps the default (all fdinfo).
    2.2's table mixes in an nvidia-smi row (121 frames), so it passes the method per row.
    """
    encoder_bytes, borrowed = _text_encoder(manifest)
    source = "" if borrowed is None else f" from `{borrowed[0]}`"
    return [
        f"- **Download**: the quant table's Download column includes the text encoder"
        f" ({_gib(encoder_bytes)}{source});",
        f'  with `{WAN_TEXT_ENCODER_OPTION}: "{WAN_TEXT_ENCODER_PATHS[1]}"` it is not fetched.',
        f"- **GPU memory**: {peaks}. The",
        "  stages are never resident together, so a clip's peak is the largest stage peak.",
    ]


def _undeclared(manifest: Mapping[str, Any]) -> list[str]:
    """The end of the resources note (that the manifest does not declare these numbers, and the
    declared limits)."""
    return [
        "- `karume.json` does not declare these figures: its declared limits cover the resident",
        "  weights and state, not the intermediate tensors a run allocates.",
        *_declared_limits(manifest),
    ]


def _declared_limits(manifest: Mapping[str, Any]) -> list[str]:
    """The device limits the manifest declares per quant (`requiredLimits` — no line when absent).

    Wan's declaration is set by the text_encoder's i8 vocabulary embedding (one buffer) — every
    preset picks the same text_encoder (weights is a complete mapping), so every preset has the same
    declaration. A manifest that differs per preset cannot be drawn in this one line, so it fails
    (no estimate folds it into one).
    """
    declared = [quant.get("requiredLimits") for quant in default_model(manifest)["quants"].values()]
    if not any(declared):
        return []
    if any(limits != declared[0] for limits in declared):
        raise ValueError(
            f"requiredLimits differ per preset ({declared}) — only a declaration that applies to"
            " every preset is drawn"
        )
    spelled = " and ".join(f"`{key}` ≥ {value:,} bytes" for key, value in declared[0].items())
    return [
        f"- **Declared limits**: every quant declares {spelled} in `karume.json` — the largest",
        "  resident weight is the text encoder's int8 vocabulary embedding, held as one buffer.",
        "  The declaration belongs to the quant, so it applies whether or not the text encoder is",
        "  used.",
    ]


def _wan_defaults(model: Mapping[str, Any]) -> list[str]:
    """The generation defaults held by `pipelineConfig` (from the manifest)."""
    config = model["pipelineConfig"]
    defaults = config["defaults"]
    return [
        "### Defaults",
        "",
        "Any of these knobs not passed to `generate()` is filled in from the manifest:",
        "",
        f"- **steps**: {knob(defaults['steps'])}",
        f"- **guidance**: {knob(defaults['guidance'])}",
        f"- **shift** (flow-matching shift): {knob(config['scheduler']['shift'])}",
    ]


#: The generation table of the card of Wan2.1 T2V 1.3B (`karume-wan2.1`).
WAN21_CARD = WanCard(
    generation="Wan2.1",
    title="Wan2.1 T2V 1.3B",
    supported_pipeline=WAN_SUPPORTED_PIPELINE,
    models=WAN21_MODELS,
    pipeline_class="Pipeline",
    paper=WAN_PAPER,
    accepted_sizes=WAN_ACCEPTED_SIZES,
    frames=WAN_FRAMES,
    default_size=WAN_DEFAULT_SIZE,
    default_frames=WAN_DEFAULT_FRAMES,
    frames_checked=_WAN21_FRAMES_CHECKED,
    verification=_WAN21_VERIFICATION,
    overview=_wan21_overview,
    attribution_notes=_wan21_attribution_notes,
    resources=_wan21_resources,
)

#: The generation table of the card of Wan2.2 TI2V 5B (`karume-wan2.2`). The technical report is the
#: one the upstream `README.md` at the pinned revision cites (the same arXiv 2503.20314 as Wan2.1 —
#: confirmed on 2026-10-06 in the README of the pin in the HF cache).
WAN22_CARD = WanCard(
    generation="Wan2.2",
    title="Wan2.2 TI2V 5B",
    supported_pipeline=WAN22_SUPPORTED_PIPELINE,
    models=WAN22_MODELS,
    pipeline_class=WAN22_PIPELINE_CLASS,
    paper=WAN_PAPER,
    accepted_sizes=WAN22_ACCEPTED_SIZES,
    frames=WAN22_FRAMES,
    default_size=WAN22_DEFAULT_SIZE,
    default_frames=WAN22_DEFAULT_FRAMES,
    frames_checked=_WAN22_FRAMES_CHECKED,
    verification=_WAN22_VERIFICATION,
    overview=_wan22_overview,
    attribution_notes=_wan22_attribution_notes,
    resources=_wan22_resources,
    extra_tags=("image-to-video",),
    usage_extra=_WAN22_I2V_USAGE,
    inputs_extra=_WAN22_I2V_INPUTS,
)


def render_wan_model_card(
    manifest: Mapping[str, Any],
    repo: str,
    abbreviations: Mapping[str, str],
    host_assets: Mapping[FileRefKey, int] = {},
    card: WanCard = WAN21_CARD,
) -> str:
    """Assemble the body of the Wan distribution's `README.md` (pure function, with a trailing
    newline).

    `abbreviations` maps the component-override tokens of preset names (the source of truth is
    `WAN_QUANT_ABBREVIATIONS` in `wan.distribution` — ADR 0074 decision 4). It is a fact not in the
    manifest, so it is taken as an argument rather than copied as a constant (the same form as the
    anima card). `card` is the generation table (the distribution recipe's Pipeline bundles it per
    generation — the default is Wan2.1).
    """
    require_pipeline(manifest, card.supported_pipeline)
    metadata = _wan_metadata(manifest, card)
    return render(
        (
            frontmatter(metadata),
            introduction(f"{card.title} — Karume", hf_upstreams(metadata.base_model)),
            card.overview(manifest),
            [""],
            _wan_base_weights(manifest, card),
            [""],
            models(manifest),
            [""],
            _wan_usage(manifest, repo, card),
            [""],
            _wan_prompts(),
            [""],
            _wan_inputs(card),
            [""],
            card.resources(manifest),
            *model_sections(
                manifest,
                (
                    partial(quants, abbreviations=abbreviations, host_assets=host_assets),
                    _wan_defaults,
                ),
            ),
        )
    )
