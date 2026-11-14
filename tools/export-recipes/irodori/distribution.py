"""Irodori's distribution recipe — series layout, output paths, quants, card (ADR 0065 decision 2).

The generic assembly engine (placement, folding of shared slots, sha256, manifest, staging/swap,
verification) lives in `karume.dist`. This module holds only **Irodori-specific facts**: what to
pick from which series directory, at which path of the distribution and under which dtype label to
lay it out, and which quant is the default.

Only **the 9 graphs needed at run time + 1 tokenizer asset** are distributed. Two of the 9 are the
waveform ↔ latent codec (DACVAE — a separate repository with separate weights upstream, bundled
here so that text-to-speech runs end to end from one repository), and two are the split DiT pair
(`dit_context` for the condition-side K/V projections and the per-step `dit` — ADR 0114). Storage
comes in 3 series, f32 / f16 / i8 (only the DiT's two graphs have **a 4th, i4**), and there are 5
quant presets (`f32` / `f16` / `i8` / `i8-a8` / `i8+dit4`). `i8` and `i8-a8` **share the same i8
bytes** and differ only in the execution knobs. `i8+dit4` takes only the DiT's two graphs (`dit` /
`dit_context`) from the i4 series and shares the same i8 bytes as `i8` for the other 7 roles (the
only mixed preset — the ruling in {@link IRODORI_QUANT_SEATS}).

`pipelineConfig` splits into two kinds: **model-specific numbers** (the declared length of the
condition states, the speaker row count, the latent width, the t_embed width) are derived from the
checkpoint's config, and **run-time knobs** (step count, CFG strength and interval, the clamp on
seconds) are kept as constants taken from the defaults of upstream's `SamplingRequest`. The former
are always derived, because baking them in means that once the weights are swapped only the host
holds stale numbers and the values come out silently wrong (the same reason as in the module doc of
the TS source of truth `packages/models/src/irodori/config.ts`).

The two codec graphs come from a **separate series** written by {@link irodori.dacvae.export} (not
on the `--model` axis). It is a separate script, but the distribution bundles both in one
repository, so this table lays out both.

The public surface is one {@link PIPELINE} (`karume.dist.Pipeline`) — the repository's dist driver
(`tools/export-recipes/dist.py`) merges it into core's PIPELINES.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, NamedTuple

from safetensors import safe_open

from _shared.calib_provenance import calib_complaint
from _shared.licenses import apache_license_2_0, mit_license
from _shared.paths import DIST_ROOT, INPUTS_ROOT
from karume.container import ContainerFormatError
from karume.dist import (
    Artifact,
    CrossRepoReference,
    DistError,
    ModelPlan,
    Pipeline,
    WeightFiles,
    assert_model_name,
    assert_storage,
    assert_storage_absent,
    complete_quant_weights,
    component_parts,
    graph_inputs,
    ir_graph,
    sha256_file,
)
from karume.verify import ContainerError, verify_container

from .card import (
    IRODORI_CODEC_MODEL,
    IRODORI_CODEC_ORIGIN_MODEL,
    IRODORI_CODEC_PARENT_MODEL,
    IRODORI_TEXT_BACKBONE_MODEL,
    render_irodori_model_card,
)

#: The default model name — the one word that ties together the series
#: (`outputs/series/irodori-<this name>/`) and where the real weights live
#: (`inputs/irodori/<this name>/`). `irodori.tokenizer_ref.default_out_dir` builds the series name
#: from the directory name of `--model-dir`, so the reader here builds from the same word too.
IRODORI_DEFAULT_MODEL = "v4-small"

#: Prefix of the series and repository names (`irodori-<model name>`).
IRODORI_SERIES_PREFIX = "irodori"

#: Pipeline contract (ADR 0041 §2 — per model). The set the TS side accepts is
#: `IRODORI_PIPELINE_NAME` / `IRODORI_PIPELINE_MAJOR`.
IRODORI_PIPELINE = "irodori/1"

#: The checkpoint's file name (`inputs/irodori/<model name>/`) and the spelling of the config key in
#: its `__metadata__`. Both are the same as in `irodori/export.py` (`CHECKPOINT_FILE` /
#: `MODEL_CONFIG_META_KEY`) — the source of truth for the shape the weights were actually built
#: with is inside the checkpoint (it is not re-fetched from HF).
IRODORI_CKPT_FILE = "model.safetensors"
IRODORI_CONFIG_META_KEY = "config_json"

#: The two flow parameterizations upstream knows, spelled as upstream spells them
#: (`TextToLatentRFDiT.__init__` accepts exactly these): the rectified-flow velocity of v4 /
#: v4.1, and MeanFlow, whose DiT also takes the length of the sampling interval (ADR 0150).
IRODORI_FLOW_RF = "rf_velocity"
IRODORI_FLOW_MEANFLOW = "meanflow"

#: The key of the checkpoint's `config_json` that names the flow parameterization (absent in the
#: v4 / v4.1 checkpoints — upstream's `ModelConfig` default is {@link IRODORI_FLOW_RF}).
IRODORI_FLOW_CONFIG_KEY = "flow_parameterization"

#: Role name → target directory name of the series (spelled as in `irodori.export.TARGETS`). Role
#: names are also the keys of the manifest's weights / assets, so they are decoupled here from the
#: hyphenated series names.
IRODORI_SERIES_DIRS: Mapping[str, str] = {
    "backbone": "backbone",
    "text_proj": "text-proj",
    "caption_proj": "caption-proj",
    "speaker": "speaker",
    "duration": "duration",
    "dit": "dit",
    "dit_context": "dit-context",
}

#: The one word for the codec (DACVAE) — **a separate repository with separate weights**, so its
#: series and its input material have their own name (the exported graphs in
#: `outputs/series/<this name>/`, the `metadata.json` written by `irodori/dacvae/convert.py` in
#: `inputs/irodori/<this name>/`). It moves independently of Irodori's model name (`v4-small`), so
#: it is not put on the `--model` axis.
IRODORI_CODEC_NAME = "dacvae-32dim"

#: Role name → target directory name of the codec series (spelled as in
#: `irodori.dacvae.export.TARGETS`).
IRODORI_CODEC_DIRS: Mapping[str, str] = {
    "codec_decoder": "decoder",
    "codec_encoder": "encoder",
}

#: Every role that has a graph (Irodori's own 7 + the codec's 2).
IRODORI_GRAPH_ROLES: tuple[str, ...] = (*IRODORI_SERIES_DIRS, *IRODORI_CODEC_DIRS)

#: Target directory name of the series → role name (the reverse of the 2 tables above). **The table
#: a writer uses to name a container's graph**; the graph name MUST match the key in the manifest's
#: weights (container-v1 §2.1 — the runtime looks it up with `prepareContainer(opened, <weights
#: key>)`). In Irodori the directory names (`caption-proj` / the codec's `decoder`) are spelled
#: differently from the keys (`caption_proj` / `codec_decoder`), so a writer that names its graphs
#: after the directory silently bakes **containers whose graphs the runtime cannot find**.
IRODORI_SERIES_ROLES: Mapping[str, str] = {
    directory: role for role, directory in IRODORI_SERIES_DIRS.items()
}
IRODORI_CODEC_ROLES: Mapping[str, str] = {
    directory: role for role, directory in IRODORI_CODEC_DIRS.items()
}

#: The configuration file `irodori/dacvae/convert.py` writes and the 2 keys read from it. The source
#: that keeps **sampleRate / hopLength from being hard-coded** (the same formula as
#: `irodori.dacvae.export.hop_length` — `hop_length = prod(encoder_rates)`).
IRODORI_CODEC_METADATA_FILE = "metadata.json"
IRODORI_CODEC_SAMPLE_RATE_KEY = "sample_rate"
IRODORI_CODEC_RATES_KEY = "encoder_rates"

#: Where the tokenizer asset comes from (of the 4 files `irodori/tokenizer_ref.py` writes under the
#: series, only the asset itself goes into the distribution — the golden / NFKC tables are for
#: verification and not needed at run time).
IRODORI_TOKENIZER_DIR = "tokenizer"
IRODORI_TOKENIZER_FILE = "tokenizer.json"

#: The storage dtypes shipped. Role names are `<graph role>_<dtype>` and map 1:1 to series roots.
#: **An axis separate from the quant preset names** (`i8` / `i8-a8` both point at the i8 series —
#: the mapping is {@link IRODORI_QUANT_SEATS}).
IRODORI_WEIGHT_DTYPES: tuple[str, ...] = ("f32", "f16", "i8", "i4")

#: Storage dtype → the **roles** its series has (paired with `irodori.export.DTYPE_TARGETS` /
#: `irodori.dacvae.export.WEIGHT_DTYPES`).
#:
#: MUST: i4 covers only the DiT's two graphs (`dit` / `dit_context`). The i4 execution path is
#: limited to linear weight slots (ADR 0069 decision 5), and the 7 non-DiT roles **share the bytes
#: of the i8 series as is** even in the quant preset `i8+dit4` ({@link IRODORI_QUANT_SEATS}) — the
#: writer has no i4 series for the other roles either. `dit_context` is **included** in i4 because,
#: before the split, its 72 linears (the condition-side K/V projections) were stored as i4 inside
#: `dit` — dropping them to i8 would move the preset's numbers (ADR 0114). The table lives in this
#: one place because the output paths / storage dtype requirements / weights declarations /
#: placement table all make the same judgement of "which (role, dtype) exists" (kept separately,
#: the day a preset is added only one of them gets updated).
IRODORI_DTYPE_ROLES: Mapping[str, tuple[str, ...]] = {
    "f32": IRODORI_GRAPH_ROLES,
    "f16": IRODORI_GRAPH_ROLES,
    "i8": IRODORI_GRAPH_ROLES,
    "i4": ("dit", "dit_context"),
}

#: The dtype of the uncompressed series (the only slot whose series root has no suffix, and also
#: where the quant-independent asset, the tokenizer, lives).
IRODORI_PLAIN_DTYPE = "f32"

#: The calibration conditions the i4 series records ({@link assert_irodori_calib_provenance}).
#: Whether calibration happened **does not change a single byte of the storage format** (the grid
#: stays RTN i4 g32 — only the rounded values and the scale ledger change), so it cannot be told
#: from the assets, the manifest or the headers. Yet the quality difference is large enough to
#: decide a ruling, so the same discipline as anima applies: "the side that wrote it records the
#: fact". The reader (here) owns the spelling of the series layout, and the writer
#: (`irodori/export.py`) takes it from here — the two places do not move independently.
CALIB_PROVENANCE_FILE = "calib_provenance.json"

#: The rounding method allowed for distribution (`method` in {@link CALIB_PROVENANCE_FILE}). The
#: plain RTN of `--no-calib` is for smoke runs and is not shipped as a distribution asset (the
#: corresponding MUST in `irodori/export.py`).
CALIB_SHIPPABLE_METHOD = "gptq"


def irodori_calib_floor(flow: str = IRODORI_FLOW_RF) -> Mapping[str, int]:
    """The minimum calibration budget allowed for distribution (required only **when the field is
    present** in {@link CALIB_PROVENANCE_FILE}).

    The values are not copied but taken from the source of truth — the corpus size from
    {@link irodori.calib_cases.CALIB_CASES}, the captured step count from the full length of the
    reference loop of the checkpoint's flow parameterization (`irodori.pipeline_ref.reference_steps`
    — 40 Euler steps for RF, 4 MeanFlow steps — = the default the export captures). For RF both are
    exactly the conditions under which the listening ruling of 2026-08-23 was made, and assets
    baked with the smoke budget `--calib-steps 1` would ship "a configuration below the ruling"
    (the storage format does not change by a single byte, so it cannot be read from the assets).
    A MeanFlow loop has only 4 steps, so its whole loop is its floor (ADR 0150).

    NOTE (deferred import): `irodori.pipeline_ref` imports this module back, so it cannot be read at
    module level. `irodori.calib_cases` forms the same cycle through pipeline_ref.
    """
    from .calib_cases import CALIB_CASES
    from .pipeline_ref import reference_steps

    return {"cases": len(CALIB_CASES), "steps": reference_steps(flow)}


def irodori_role(role: str, dtype: str) -> str:
    """Role name (`backbone_f16` — one word shared by placements, output paths, storage checks)."""
    return f"{role}_{dtype}"


#: Relative output paths (**inside the model subtree**) — the one place the placement table and the
#: manifest share. The storage dtype appears in the file name, the same form as Anima / SBV2
#: (`model.f16.krm`), so that two series side by side in one directory cannot be mistaken.
IRODORI_OUTPUT_PATHS: Mapping[str, str] = {
    **{
        irodori_role(role, dtype): f"{role}/model.{dtype}.krm"
        for dtype, roles in IRODORI_DTYPE_ROLES.items()
        for role in roles
    },
    "tokenizer": f"{IRODORI_TOKENIZER_DIR}/{IRODORI_TOKENIZER_FILE}",
}

#: The storage vocabulary **required** in each role's binding table (the same grounds as Anima /
#: SBV2 — a measured incident where a plain f32 asset passed assembly, loading and execution and
#: stayed silent until the reference-match check). In a compressed series only the fake-quant
#: targets become f16 / i8 / i4 (bias / norm / graph constants, the i8 per-channel scales and the
#: i4 group scales stay f32), so "contains that vocabulary" is required. The tokenizer is JSON, so
#: it is not listed.
IRODORI_STORAGE_REQUIREMENTS: Mapping[str, str] = {
    irodori_role(role, dtype): dtype
    for dtype, roles in IRODORI_DTYPE_ROLES.items()
    for role in roles
}

#: The storage vocabulary that **must not appear** in each role's binding table
#: ({@link assert_storage_absent}). For the f32 preset, "contains f32" alone cannot tell it apart
#: from an asset of a compressed series (compressed series also hold the ineligible weights — bias
#: / norm / graph constants / i8 per-channel scales / i4 group scales — in f32), so only by also
#: requiring the absence of **every compressed storage vocabulary** does series × storage become
#: unique as a set. The reverse direction (an f32 asset in a compressed preset) is caught by
#: {@link assert_storage} through the missing requirement.
#:
#: MUST: **the i8 preset also requires the absence of i4** (only the DiT's two graphs have both
#: series). The i4 series of `dit` is **a mix of i4 + i8 + f32** (the 96 linears inside the blocks,
#: excluding adaLN and the condition-side K/V projections, are i4; the 5 outside the blocks + the
#: 144 adaLN linears are i8; bias / norm / scale are f32 — the listening ruling of 2026-08-23 took
#: the out-of-block linears and adaLN out of i4. `irodori.export._fake_quant_i4`), and the i4
#: series of `dit_context` has its 72 condition-side K/V projections in i4 and its norms in f32, so
#: a requirement check of "contains i8" is satisfied by the i4 series too. **This forbidden table
#: alone** separates the i8 preset from the i4 series; without it, `linearCompute: "a8"` of the
#: default preset `i8-a8` silently turns into the w4a8 path running on resident i4 (ADR 0076).
#: Before the mix, the same hole existed as "whether the i4 series can pose as the i8 preset depends
#: on upstream's eligibility ratio" — the mix only left that hole open all the time.
IRODORI_STORAGE_FORBIDDEN: Mapping[str, tuple[str, ...]] = {
    **{
        irodori_role(role, IRODORI_PLAIN_DTYPE): tuple(
            dtype for dtype in IRODORI_WEIGHT_DTYPES if dtype != IRODORI_PLAIN_DTYPE
        )
        for role in IRODORI_GRAPH_ROLES
    },
    **{irodori_role(role, "i8"): ("i4",) for role in IRODORI_DTYPE_ROLES["i4"]},
}

#: The weights declarations (dtype label → role name). All 9 graphs have the 3 slots f32 / f16 / i8
#: (the DiT's two graphs also i4), and the automatic completion of {@link complete_quant_weights}
#: does not apply, so the quant table names every role.
IRODORI_WEIGHTS: Mapping[str, Mapping[str, WeightFiles]] = {
    role: {
        dtype: WeightFiles(irodori_role(role, dtype))
        for dtype, roles in IRODORI_DTYPE_ROLES.items()
        if role in roles
    }
    for role in IRODORI_GRAPH_ROLES
}

#: The assets declarations (unconditional files that do not depend on the quant choice).
IRODORI_ASSETS: Mapping[str, str] = {"tokenizer": "tokenizer"}


class QuantSeat(NamedTuple):
    """One quant preset (base storage dtype, execution knobs, per-role exceptions, display text).

    **The mapping between preset name and series root is spelled in this one place only** (`i8` /
    `i8-a8` both point at the `-i8` series and share one set of bytes — only `session` differs).
    """

    #: Storage dtype of the roles not listed in the exceptions.
    dtype: str
    #: Execution knobs (the manifest owns the vocabulary of `session`).
    session: Mapping[str, str]
    #: Short display name shown in the selection UI (ADR 0075 decision 1 — 64-character limit).
    label: str
    #: Likewise, a one-line description (200-character limit). It does not say that it is the
    #: default (`defaultQuant` points to it).
    description: str
    #: Per-role exceptions (an opening for mixed presets only — the default still applies to every
    #: role).
    roles: Mapping[str, str] = {}


#: Component-override token of a preset name → its weights name (ADR 0074 decision 4). **Empty for
#: Irodori** — the token `dit` of the mixed preset `i8+dit4` is the weights name itself (not an
#: abbreviation), so emitting the mapping would only add a "`dit` is `dit`" line to the card.
#:
#: NOTE: even after the DiT was split into 2 graphs (ADR 0114), the token `dit` refers to the whole
#: DiT (`dit` and the condition-side `dit_context`). The shape of this table (one token → one
#: weights name) cannot name both, and renaming the preset itself would be a breaking change to the
#: distribution, so the preset's description (`description`) covers both within what it calls the
#: DiT.
IRODORI_QUANT_ABBREVIATIONS: Mapping[str, str] = {}

#: Quant preset name → {@link QuantSeat}. Preset names follow ADR 0074's grammar
#: `<storage>[+<component><bits>]…[-<knob>]…`, where `<storage>` is **the base storage shared by
#: every role**.
#:
#: **Only the `i8+dit4` preset stops being uniform across the 8 roles** (maintainer ruling of
#: 2026-08-23). The ruling of 2026-08-12 was "i8 uniform too", grounded at the time in a measurement
#: showing that a mix made S (the utterance length) drift between presets, but that drift
#: **disappeared with GPTQ calibration** (`docs/research/2026-08-20-gptq-awq-calibrated-rounding.md`
#: §6 — with calibrated rounding S matches f32 exactly). `i8` / `i8-a8` stay uniform across the 8
#: roles as before.
#:
#: MUST: `linearCompute` of `i8-a8` reaches **only the Sessions of the DiT's two graphs (`dit` /
#: `dit_context`)** (the module doc of `pipeline.ts` on the models side) — the DiT's 317 linears
#: are the only eligible set, and the 5 condition encoders run only once per generation.
#: `dit_context` (the 72 condition-side K/V projections) also runs in this preset because, before
#: the split, those 72 ran inside `dit` with this preset's execution form — running them another
#: way would move the numbers (ADR 0114).
#:
#: MUST: `i8+dit4` **does not declare `linearCompute`**. Resident i4 × i8 activations go through
#: the w4a8 path (ADR 0076 — partial group reduction), but that configuration has **never been
#: measured** for irodori (the perf-ledger entry covers the weight side only). In anima the same
#: configuration failed a quality ruling on real-GPU images (ADR 0076 decision 6 / the same MUST in
#: `anima/distribution.py`), so adding only the declaration before measuring would ship "fast but
#: rough" with the face of a default. This preset exists for **size and VRAM**, not speed.
IRODORI_QUANT_SEATS: Mapping[str, QuantSeat] = {
    "f32": QuantSeat(
        "f32",
        {},
        label="Full precision (f32)",
        description="Every graph in f32 storage — the largest download, with no quantization"
        " applied.",
    ),
    "f16": QuantSeat(
        "f16",
        {},
        label="Half size (f16)",
        description="Every graph stored as f16 and computed in f32 — half the f32 download,"
        " with no audible difference in the listening check.",
    ),
    "i8": QuantSeat(
        "i8",
        {},
        label="Quarter size (int8)",
        description="Every graph stored as int8 and computed in f32 — about a quarter of the f32"
        " download, with the execution path left unchanged.",
    ),
    "i8-a8": QuantSeat(
        "i8",
        {"linearCompute": "a8"},
        label="Balanced (int8, int8 linear)",
        description="The int8 weights with per-token int8 activations in the DiT's linear layers"
        " — the same download as plain int8, and faster per Euler step.",
    ),
    "i8+dit4": QuantSeat(
        "i8",
        {},
        label="Lowest memory (int4 DiT)",
        description="The DiT in GPTQ-calibrated int4 (group-32) while the other seven graphs stay"
        " int8 — the smallest download and the least resident memory.",
        roles={"dit": "i4", "dit_context": "i4"},
    ),
}


def irodori_quants(model: str) -> dict[str, Any]:
    """The manifest's quant table of one model: {@link IRODORI_QUANT_SEATS}, with the descriptions
    of {@link IRODORI_MODEL_SEAT_DESCRIPTIONS} in place of the family's where the model has its
    own (the storage, the knobs and the labels are the same for every model)."""
    descriptions = IRODORI_MODEL_SEAT_DESCRIPTIONS.get(model, {})
    return {
        name: {
            "weights": {role: seat.roles.get(role, seat.dtype) for role in IRODORI_GRAPH_ROLES},
            "session": dict(seat.session),
            "label": seat.label,
            "description": descriptions.get(name, seat.description),
        }
        for name, seat in IRODORI_QUANT_SEATS.items()
    }


#: The family's default is `i8-a8` (the maintainer's listening ruling of 2026-08-12 — the 4
#: presets were checked end to end with a DAC + headphones: "it does not feel like a loss of audio
#: quality". The distribution at 25.2% / DiT resident 0.37GB / wall ×1.12 take effect by default,
#: and the most faithful `f32` preset stays available to pick explicitly. The numeric band (sim
#: LSD 5.64, z maxAbs 2.97 against the i8 golden) is held by the discriminating band of
#: `e2e_irodori_ab_test.ts`). A model with its own ruling is in
#: {@link IRODORI_MODEL_DEFAULT_QUANTS}.
IRODORI_DEFAULT_QUANT = "i8-a8"

#: DACVAE frame rate (Hz) — 48kHz / hop 1920 = 25. The same value as
#: `irodori.export.CODEC_FRAME_RATE`; the codec is a separate repository with separate weights, so
#: it is not in the checkpoint's config. `sampleRate` / `hopLength` are derived from the codec's
#: `metadata.json`, and the consistency of the three (`sampleRate == frameRate × hopLength`) is
#: checked by {@link irodori_pipeline_config}.
IRODORI_FRAME_RATE = 25

#: The latent frames added on both sides of the kept range when the codec decoder is tiled. **A
#: model constant from the receptive field** (13,793 samples per side = 7.19 frames → covered with
#: 8 frames = 15,360 samples); it is not in `metadata.json`, so this is its only source. The
#: measured grounds: the decoder's main path has no causal layer, and every layer is a symmetric pad
#: or a convT of exactly `L·stride` (= translation equivariant).
IRODORI_CODEC_HALO_FRAMES = 8

#: The utterance-length clamp in seconds (the defaults of upstream `SamplingRequest.min_seconds` /
#: `max_seconds`). `max_seconds` **is also the value that set the upper bound of `dit`'s symbolic
#: dimension S** (`irodori.export.DIT_MAX_SECONDS` holds the same 30.0 as "the implementation's
#: source of truth is SamplingRequest's default"). Both are built from one constant because if the
#: distribution's clamp and the baked graph's upper bound moved independently, the failure would
#: only show at run time, as "S passes but the RoPE table is too short".
IRODORI_MIN_SECONDS = 0.5
IRODORI_MAX_SECONDS = 30.0

#: Defaults of the run-time knobs (the same-named fields of upstream `SamplingRequest` and the
#: `init_scale` of `rf.sample_euler_rf_cfg`). **Not model-specific**, so they are not in the
#: checkpoint's config and this is their only source (the same nature as Anima's
#: `anima.distribution.ANIMA_PIPELINE_CONFIG`).
#:
#: MUST: `speakerUncondMode` / `cfgGuidanceMode` are **declarations**, not switches (ADR 0047
#: decision 1). The TS side rejects any other value at parse time, so a distribution baked with
#: another mode fails before it is read. The full-loop golden (`meta.json` of
#: `irodori/pipeline_ref.py`) is baked with the same values too, and if golden regeneration and dist
#: regeneration drift apart, the E2E check (`e2e_irodori_latent_test.ts`) fails it as drift of the
#: effective values.
IRODORI_SAMPLING_DEFAULTS: Mapping[str, Any] = {
    "steps": 40,
    "initScale": 0.999,
    "cfgMinT": 0.5,
    "cfgMaxT": 1.0,
    "cfgScales": {"text": 3.0, "speaker": 5.0, "caption": 3.0},
    "minSeconds": IRODORI_MIN_SECONDS,
    "maxSeconds": IRODORI_MAX_SECONDS,
    "speakerUncondMode": "mask",
    "cfgGuidanceMode": "independent",
}

#: The run-time knobs of a **MeanFlow** checkpoint (ADR 0150): its parameterization, upstream's
#: default step count for it (`inference_runtime` uses `4 if is_meanflow else 40` when no count is
#: requested) and the same clamp on seconds. Upstream ignores every RF-only control for such a
#: checkpoint (CFG scales, guidance mode and CFG time bounds, initial-noise truncation — and the
#: initial scale of the t schedule, since its loop runs t from 1 to 0 — `docs/meanflow.md`), so
#: none of them is declared: the TS side refuses them, and karume does not accept a value it would
#: ignore. `speakerUncondMode` is an RF declaration too (it only shapes CFG's uncond branch).
IRODORI_MEANFLOW_SAMPLING_DEFAULTS: Mapping[str, Any] = {
    "flowParameterization": IRODORI_FLOW_MEANFLOW,
    "steps": 4,
    "minSeconds": IRODORI_MIN_SECONDS,
    "maxSeconds": IRODORI_MAX_SECONDS,
}

#: The `(input names in order, number of outputs)` required of each graph. **The input order is the
#: source of truth for the form read by position at run time**, and the output count is where "a
#: separate verification asset slipped in" is caught (the same mechanism as SBV2's
#: {@link sbv2.distribution.assert_bert_hidden} — if `caption_proj` were swapped for a 1-output
#: asset, duration would run on a different vector taken from the first output as `caption_vec`,
#: and with the shapes still matching it would stay silent).
#:
#: The DiT's two graphs (`dit_context` / `dit`) are not listed here — the number of boundary
#: tensors depends on the block count (ADR 0114), so it cannot be written as a fixed order and is
#: derived by matching the declarations of the two ({@link _assert_irodori_dit_boundary}).
IRODORI_GRAPH_SHAPES: Mapping[str, tuple[tuple[str, ...], int]] = {
    "backbone": (("input_ids",), 1),
    "text_proj": (("hidden",), 1),
    "caption_proj": (("hidden",), 2),
    "speaker": (("latent",), 1),
    "duration": (
        ("text_state", "speaker_vec", "has_speaker", "caption_vec", "has_caption"),
        1,
    ),
    # The 2 codec graphs are pure convolutional networks with neither position tables nor masks
    # (1 input / 1 output).
    "codec_decoder": (("latent",), 1),
    "codec_encoder": (("wav",), 1),
}

#: The input order of `dit_context` (the 3 condition states — the raw projector outputs right-padded
#: to Tmax. `text_norm` / `caption_norm` are applied inside this graph).
IRODORI_CONTEXT_INPUTS: tuple[str, ...] = ("text_state", "speaker_state", "caption_state")

#: The first 3 inputs of `dit`. The condition-side K/V ({@link irodori_context_kv_names}) follow.
IRODORI_DIT_HEAD_INPUTS: tuple[str, ...] = ("x_t", "t_embed", "mask")

#: The first 4 inputs of a MeanFlow `dit`: `delta_embed` is the timestep embedding of the
#: interval length (`t − t_next`), the same width and form as `t_embed` (upstream's
#: `delta_cond_module` input — ADR 0150). The condition-side K/V follow, as in
#: {@link IRODORI_DIT_HEAD_INPUTS}.
IRODORI_MEANFLOW_DIT_HEAD_INPUTS: tuple[str, ...] = ("x_t", "t_embed", "delta_embed", "mask")


def irodori_flow_parameterization(config: Mapping[str, Any]) -> str:
    """The flow parameterization a checkpoint's `config_json` declares.

    Absent means {@link IRODORI_FLOW_RF} (upstream's `ModelConfig` default — the v4 / v4.1
    checkpoints carry no such key). The value is normalized the way upstream normalizes it
    (`str(...).strip().lower()` in `TextToLatentRFDiT.__init__`), and anything outside the two
    parameterizations fails — upstream refuses it too, and guessing would build the wrong sampler.
    """
    raw = config.get(IRODORI_FLOW_CONFIG_KEY, IRODORI_FLOW_RF)
    if not isinstance(raw, str):
        raise DistError(
            f"{IRODORI_CONFIG_META_KEY}: {IRODORI_FLOW_CONFIG_KEY} is not a string ({raw!r})"
        )
    flow = raw.strip().lower()
    if flow not in (IRODORI_FLOW_RF, IRODORI_FLOW_MEANFLOW):
        raise DistError(
            f"{IRODORI_CONFIG_META_KEY}: {IRODORI_FLOW_CONFIG_KEY} is {raw!r}, not"
            f" '{IRODORI_FLOW_RF}' or '{IRODORI_FLOW_MEANFLOW}'"
        )
    return flow


def irodori_dit_head_inputs(flow: str) -> tuple[str, ...]:
    """The inputs `dit` takes from the host per forward, for a flow parameterization (the
    condition-side K/V follow them)."""
    if flow == IRODORI_FLOW_MEANFLOW:
        return IRODORI_MEANFLOW_DIT_HEAD_INPUTS
    if flow == IRODORI_FLOW_RF:
        return IRODORI_DIT_HEAD_INPUTS
    raise DistError(f"Unknown flow parameterization {flow!r}")


def irodori_context_kv_names(blocks: int) -> tuple[str, ...]:
    """Boundary names of the condition-side K/V (outputs of `dit_context` = the inputs of `dit`
    from the 4th on).

    Ordered per block b as `context_k_<b>` → `context_v_<b>`, in ascending b. The runtime binds the
    resident tensors by these **names** (`copyOutputs` and the inputs of `dit`), so the writer
    (`irodori.export` renames the IR's boundary names) and the reader
    ({@link _assert_irodori_dit_boundary}) build them from the spelling in this one place.

    MUST: the block count is not baked in — the caller passes it from the upstream configuration
    (`len(model.blocks)` on the export side) or from the declarations of the baked graph (the
    output count of `dit_context` on the assembly side).
    """
    if blocks < 1:
        raise ValueError(f"The DiT block count is {blocks} (must be at least 1)")
    return tuple(
        name for index in range(blocks) for name in (f"context_k_{index}", f"context_v_{index}")
    )


#: `(role, graph input, axis, pipelineConfig field)` — the table matching the graphs' **static**
#: dimensions against the declarations. The same combinations as `assertStaticDim` in the TS side's
#: `irodori.Pipeline.fromAssets` are checked **on the baking side too** (so that the failure does
#: not first appear on a user's machine after shipping). The 3 condition states are inputs of
#: `dit_context` (they moved from `dit` in ADR 0114).
IRODORI_STATIC_DIMS: tuple[tuple[str, str, int, str], ...] = (
    ("dit", "x_t", 2, "latentDim"),
    ("dit", "t_embed", 1, "timestepEmbedDim"),
    ("dit_context", "text_state", 1, "maxTextLen"),
    ("dit_context", "text_state", 2, "textDim"),
    ("dit_context", "speaker_state", 1, "speakerRows"),
    ("dit_context", "speaker_state", 2, "speakerDim"),
    ("dit_context", "caption_state", 1, "maxCaptionLen"),
    ("dit_context", "caption_state", 2, "captionDim"),
    ("duration", "text_state", 2, "textDim"),
    ("duration", "speaker_vec", 1, "speakerDim"),
    ("duration", "caption_vec", 1, "captionDim"),
    # The codec is **a separate repository with separate weights**, so this is the only place that
    # guarantees its latent width and samples per frame mesh with Irodori's declarations (mixing in
    # a DACVAE of another dimension gives a different voice with the shapes still matching / only
    # the waveform length silently shifts).
    ("codec_decoder", "latent", 2, "latentDim"),
    ("codec_encoder", "wav", 2, "hopLength"),
)


def irodori_series_name(model: str) -> str:
    """Series name (`outputs/series/<this name>/`) — the same spelling as
    `irodori.tokenizer_ref.default_out_dir`.
    """
    return f"{IRODORI_SERIES_PREFIX}-{model}"


def irodori_repo_name(model: str) -> str:
    """Distribution repository of one model (the `karume-` prefix: repo-name ruling 2026-08-09)."""
    return f"karume-{IRODORI_SERIES_PREFIX}-{model}"


class PublishedContainer(NamedTuple):
    """One container a model takes as **published**, not from its series
    ({@link IRODORI_PUBLISHED}): where its bytes are and the digest that pins them."""

    #: The model whose repository holds the bytes (the model itself, or the one it borrows from).
    holder: str
    #: The representative path in the holder's local mirror
    #: (`<dist_root>/<holder repository>/<holder>/<role path>`).
    path: Path
    #: sha256 of part 0 as published ({@link IRODORI_PUBLISHED_PART0}).
    part0_sha256: str


@dataclass(frozen=True)
class IrodoriSources:
    """The assembly inputs. Two sets: the series, and where the checkpoints live (`inputs/` — not
    generated artifacts).

    The checkpoint side is needed to **derive the `pipelineConfig` numbers instead of baking them
    in** (only `__metadata__` / `metadata.json` are read, so the 2.9GB payload is not scanned). The
    codec has its own 2 because it is **a separate repository with separate weights** — moving
    Irodori's model name does not move the codec.

    The series are **a separate directory per storage dtype** (the dtype suffix of
    `irodori.export.default_out_root` / `irodori.dacvae.export.default_out_root`). The tokenizer
    asset does not depend on the quant, so only the one in the plain series (f32) is looked at.
    """

    model: Path
    codec_model: Path
    #: Storage dtype → series root (the dtypes for which {@link IRODORI_DTYPE_ROLES} has Irodori
    #: roles).
    series_by_dtype: Mapping[str, Path]
    #: Likewise for the codec (the codec has no i4 series, so 3: f32 / f16 / i8).
    codec_series_by_dtype: Mapping[str, Path]
    #: Role name (`<graph role>_<dtype>`) → the **published container** the role is taken as,
    #: instead of the series (empty for a model that exports everything it ships —
    #: {@link irodori_published_sources}).
    published: Mapping[str, PublishedContainer] = MappingProxyType({})

    @property
    def series(self) -> Path:
        """The plain series root (where the quant-independent tokenizer asset lives).

        It is looked up every time instead of kept in a copied field, so that the series root never
        moves independently in two places.
        """
        return self.series_by_dtype[IRODORI_PLAIN_DTYPE]

    @property
    def codec_series(self) -> Path:
        """The plain codec series root."""
        return self.codec_series_by_dtype[IRODORI_PLAIN_DTYPE]


def irodori_sources(
    series_dir: Path, model: str = IRODORI_DEFAULT_MODEL, dist_root: Path | None = None
) -> IrodoriSources:
    """Look up the inputs from the series' parent directory (`outputs/series/`) and the spellings
    of `_shared.paths`.

    The dtype suffixes are spelled the same as in `irodori.export.default_out_root` /
    `irodori.dacvae.export.default_out_root` — writer and reader build from the same word. The
    roles a model takes as published containers ({@link IRODORI_PUBLISHED}) are read from the
    mirrors under `dist_root` (default `models/` — the same place the dist driver reconciles
    references against).
    """
    suffix = {
        dtype: "" if dtype == IRODORI_PLAIN_DTYPE else f"-{dtype}"
        for dtype in IRODORI_WEIGHT_DTYPES
    }
    by_dtype = {
        dtype: series_dir / f"{irodori_series_name(model)}{tail}" for dtype, tail in suffix.items()
    }
    codec_by_dtype = {
        dtype: series_dir / f"{IRODORI_CODEC_NAME}{tail}"
        for dtype, tail in suffix.items()
        # No root is **created** for a dtype in which the codec has no slot (i4) — handing out an
        # unused path means that, the day an i4 role is added, a series "assumed to exist" is
        # silently referenced.
        if not set(IRODORI_DTYPE_ROLES[dtype]).isdisjoint(IRODORI_CODEC_DIRS)
    }
    return IrodoriSources(
        model=INPUTS_ROOT / IRODORI_SERIES_PREFIX / model,
        codec_model=INPUTS_ROOT / IRODORI_SERIES_PREFIX / IRODORI_CODEC_NAME,
        series_by_dtype=by_dtype,
        codec_series_by_dtype=codec_by_dtype,
        published=irodori_published_sources(model, DIST_ROOT if dist_root is None else dist_root),
    )


def irodori_placements(sources: IrodoriSources) -> dict[str, Path]:
    """Role name → source file. The output paths are held by {@link IRODORI_OUTPUT_PATHS}.

    Anything not in this table does not go into the output (`io.*.safetensors` and the 3 tokenizer
    golden files are dropped this way).
    """
    placements = {"tokenizer": sources.series / IRODORI_TOKENIZER_DIR / IRODORI_TOKENIZER_FILE}
    for dtype, roles in IRODORI_DTYPE_ROLES.items():
        for role in roles:
            name = irodori_role(role, dtype)
            if name in sources.published:
                # The published container itself (never re-exported).
                placements[name] = sources.published[name].path
                continue
            series = (
                sources.series_by_dtype[dtype]
                if role in IRODORI_SERIES_DIRS
                else sources.codec_series_by_dtype[dtype]
            )
            directory = (IRODORI_SERIES_DIRS | IRODORI_CODEC_DIRS)[role]
            placements[name] = series / directory / "model.krm"
    return placements


def irodori_model_config(model_dir: Path) -> Mapping[str, Any]:
    """Read `config_json` from the checkpoint's `__metadata__` (only the header is read).

    The same source and the same reason as `irodori.export.read_configs` (the config is not
    re-fetched from HF). This one does not go through torch because assembly needs only one JSON.
    """
    path = model_dir / IRODORI_CKPT_FILE
    if not path.is_file():
        raise DistError(f"Assembly input missing: {path}")
    with safe_open(str(path), framework="np") as handle:
        metadata = handle.metadata()
    if not isinstance(metadata, dict) or IRODORI_CONFIG_META_KEY not in metadata:
        raise DistError(f"The __metadata__ of {path} has no '{IRODORI_CONFIG_META_KEY}'")
    try:
        config = json.loads(metadata[IRODORI_CONFIG_META_KEY])
    except json.JSONDecodeError as error:
        raise DistError(f"{path}: {IRODORI_CONFIG_META_KEY} cannot be parsed as JSON") from error
    if not isinstance(config, dict):
        raise DistError(f"{path}: {IRODORI_CONFIG_META_KEY} is not a top-level object")
    return config


def _irodori_int(config: Mapping[str, Any], key: str) -> int:
    """Read and check an integer field (declared length / width) of the checkpoint config."""
    value = config.get(key)
    # bool is a subclass of int. Letting `"latent_dim": true` through as 1 would loosen the width
    # check.
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise DistError(f"{IRODORI_CONFIG_META_KEY}: {key} is not a positive integer ({value!r})")
    return value


def _irodori_float(config: Mapping[str, Any], key: str) -> float:
    """Read and check a real-valued field (seconds) of the checkpoint config."""
    value = config.get(key)
    # NaN compares False with everything, so it slips through `<= 0` and leaks out as a generic
    # exception in the downstream seconds → frames conversion (int()). Inf fails at the same spot —
    # finiteness is required here.
    if (
        not isinstance(value, int | float)
        or isinstance(value, bool)
        or not math.isfinite(value)
        or value <= 0
    ):
        raise DistError(
            f"{IRODORI_CONFIG_META_KEY}: {key} is not a finite positive number ({value!r})"
        )
    return float(value)


class IrodoriCodecNumbers(NamedTuple):
    """Two numbers derived from the codec's `metadata.json` (seconds ↔ samples ↔ frames)."""

    sample_rate: int
    hop_length: int


def irodori_codec_numbers(codec_model_dir: Path) -> IrodoriCodecNumbers:
    """Derive `sampleRate` / `hopLength` from the `metadata.json` that
    `irodori/dacvae/convert.py` wrote.

    MUST: do not copy the values — after swapping in a DACVAE of another dimension or hop, only the
    host would keep the old conversion and output "audio of a plausible length" (no exception is
    raised). `hop_length` uses the same formula as `irodori.dacvae.export.hop_length`
    (`prod(encoder_rates)` — as spelled in `DACVAE.__init__`).
    """
    path = codec_model_dir / IRODORI_CODEC_METADATA_FILE
    if not path.is_file():
        raise DistError(
            f"Assembly input missing: {path} (created by `uv run python -m irodori.dacvae.convert`)"
        )
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise DistError(f"{path} cannot be parsed as JSON") from error
    kwargs = raw.get("kwargs") if isinstance(raw, dict) else None
    if not isinstance(kwargs, dict):
        raise DistError(f"{path} has no 'kwargs'")
    sample_rate = kwargs.get(IRODORI_CODEC_SAMPLE_RATE_KEY)
    if not isinstance(sample_rate, int) or isinstance(sample_rate, bool) or sample_rate <= 0:
        raise DistError(f"{path}: {IRODORI_CODEC_SAMPLE_RATE_KEY} is not a positive integer")
    rates = kwargs.get(IRODORI_CODEC_RATES_KEY)
    if not isinstance(rates, list) or not rates:
        raise DistError(f"{path}: {IRODORI_CODEC_RATES_KEY} is not a non-empty list")
    hop_length = 1
    for rate in rates:
        if not isinstance(rate, int) or isinstance(rate, bool) or rate <= 0:
            raise DistError(
                f"{path}: {IRODORI_CODEC_RATES_KEY} has an element that is not a positive integer"
            )
        hop_length *= rate
    return IrodoriCodecNumbers(sample_rate=sample_rate, hop_length=hop_length)


def irodori_pipeline_config(
    config: Mapping[str, Any], codec: IrodoriCodecNumbers
) -> dict[str, Any]:
    """Build `pipelineConfig` (the 23 fields of the TS-side schema) from the checkpoint config.

    The sampler part follows the checkpoint's own flow parameterization
    ({@link irodori_flow_parameterization}): the RF knobs of {@link IRODORI_SAMPLING_DEFAULTS}, or
    for a MeanFlow checkpoint {@link IRODORI_MEANFLOW_SAMPLING_DEFAULTS} (its declaration, its
    step count and the clamp on seconds — ADR 0150). The RF form carries no
    `flowParameterization` (absent means RF on the TS side, so the published manifests stay as
    they are).

    MUST: do not copy model-specific numbers — if only the host holds stale numbers after the
    weights are swapped, the right padding and the row counting still pass (the shapes match) and a
    result that **read the conditions at different positions** comes out silently. The source of
    truth for the field names and value ranges is `packages/models/src/irodori/config.ts`.

    `latentDim` serves two roles (the width of `dit`'s `x_t` = `latent_dim × latent_patch_size`,
    and the width of one frame of the reference latent = `latent_dim`), so it can only be built at
    `latent_patch_size == 1`, where the two agree — once it can no longer serve both, the TS field
    has to be split, so this fails here instead of silently picking one.
    """
    latent_patch = _irodori_int(config, "latent_patch_size")
    if latent_patch != 1:
        raise DistError(
            f"latent_patch_size is {latent_patch} — pipelineConfig's latentDim serves as both"
            " the x_t width of `dit` and the frame width of the reference latent, and the two"
            " differ unless it is 1 (the TS field would have to be split)"
        )
    # Do not let seconds → frames (`frameRate`) and seconds → samples → frames (the 2 values that
    # come from the codec) move independently. The TS side checks the same formula at parse time,
    # but it is checked here too to avoid **failing after shipping** (the spot that notices the
    # moment a codec with another hop is mixed in).
    if codec.sample_rate != IRODORI_FRAME_RATE * codec.hop_length:
        raise DistError(
            f"The codec's sample_rate {codec.sample_rate} differs from frameRate"
            f" {IRODORI_FRAME_RATE} × hop_length {codec.hop_length}"
        )
    frames = int(_irodori_float(config, "ref_max_seconds") * IRODORI_FRAME_RATE)
    speaker_patch = _irodori_int(config, "speaker_patch_size")
    return {
        "maxTextLen": _irodori_int(config, "max_text_len"),
        "maxCaptionLen": _irodori_int(config, "max_caption_len"),
        # Upper bound of the reference latent after patching (`irodori.export.speaker_sym_max`)
        # + 1 mean token.
        "speakerRows": frames // speaker_patch + 1,
        # Upper bound of the latent that can be generated (same formula as
        # `irodori.export.dit_sym_max`).
        "ditSymMax": int(IRODORI_MAX_SECONDS * IRODORI_FRAME_RATE) // latent_patch,
        "frameRate": IRODORI_FRAME_RATE,
        "sampleRate": codec.sample_rate,
        "hopLength": codec.hop_length,
        "codecHaloFrames": IRODORI_CODEC_HALO_FRAMES,
        "latentDim": _irodori_int(config, "latent_dim"),
        "speakerPatchSize": speaker_patch,
        "speakerDim": _irodori_int(config, "speaker_dim"),
        "textDim": _irodori_int(config, "text_dim"),
        "captionDim": _irodori_int(config, "caption_dim"),
        "timestepEmbedDim": _irodori_int(config, "timestep_embed_dim"),
        **(
            IRODORI_MEANFLOW_SAMPLING_DEFAULTS
            if irodori_flow_parameterization(config) == IRODORI_FLOW_MEANFLOW
            else IRODORI_SAMPLING_DEFAULTS
        ),
    }


def assert_irodori_graphs(
    placements: Mapping[str, Path], pipeline_config: Mapping[str, Any]
) -> None:
    """Check that the 9 graphs **can be read** and that their input order, output count and static
    dimensions are as declared.

    MUST: some mismatches still load and run with the shapes matching (`caption_proj`'s output
    count, the declared lengths of the condition states, `speaker`'s patch width), so they fail
    here before the distribution is laid out. The same discipline as SBV2's
    {@link sbv2.distribution.assert_bert_hidden} — this is the only place that matches the numbers
    held by separate scripts (the targets of `irodori/export.py` and this manifest).

    MUST: run it over **every storage dtype series, without exception**. The f16 series is emitted
    by a process separate from f32, so checking only one lets "f32 is as declared but only f16 is
    another version" through (agreement of the storage dtype is checked by {@link assert_storage},
    but that one never looks at graph declarations). The i4 series has only the DiT's two graphs
    ({@link IRODORI_DTYPE_ROLES}), so only **the roles present** in that series are checked.
    """
    for dtype, roles in IRODORI_DTYPE_ROLES.items():
        _assert_irodori_graph_set(
            {role: placements[irodori_role(role, dtype)] for role in roles},
            pipeline_config,
        )


def _assert_irodori_graph_set(
    placements: Mapping[str, Path], pipeline_config: Mapping[str, Any]
) -> None:
    """Check the graphs of one series (the keys of `placements` are graph role names without the
    dtype suffix).

    Takes **only the roles present** in that series (the i4 series has the DiT's two graphs).
    Cross-role checks (`speaker`'s patch width) run only when both are present — what "did not run"
    in a series that has only one of them is covered in the other series by the all-dtype loop of
    {@link assert_irodori_graphs}. The DiT's two graphs are the exception: they **only make sense
    as a pair** (`dit` takes the outputs of `dit_context` as inputs), so every series is required to
    have both ({@link _assert_irodori_dit_boundary}).
    """
    graphs = {role: ir_graph(path) for role, path in placements.items()}
    for role, (expected_inputs, expected_outputs) in IRODORI_GRAPH_SHAPES.items():
        if role not in graphs:
            continue
        path = placements[role]
        graph = graphs[role]
        names = tuple(graph_inputs(graph, path))
        if names != expected_inputs:
            raise DistError(
                f"The graph inputs of {path} are {list(names)}, not the expected"
                f" {list(expected_inputs)} — the runtime binds them by name, so if even one"
                " spelling changes they cannot be bound (the order is checked too, as evidence"
                " that the export side's declaration order has not moved)"
            )
        outputs = graph.get("outputs")
        count = len(outputs) if isinstance(outputs, list) else outputs
        if count != expected_outputs:
            raise DistError(
                f"The graph of {path} has {count} outputs, not the {expected_outputs} the"
                " distribution requires — an asset of another target is mixed in"
            )
    for role, name, axis, field_name in IRODORI_STATIC_DIMS:
        if role not in graphs:
            continue
        declared = graph_inputs(graphs[role], placements[role])[name][axis]
        expected = pipeline_config[field_name]
        if declared != expected:
            raise DistError(
                f"{placements[role]}: input '{name}', axis {axis}, is {declared!r},"
                f" while pipelineConfig's {field_name} is {expected} — the checkpoint config and"
                " the baked graph are different versions"
            )
    # The reference latent is patched before it goes to `speaker` (ADR 0047 decision 4), so the
    # input width is the product of the 2 fields.
    if "speaker" in graphs:
        patched = pipeline_config["latentDim"] * pipeline_config["speakerPatchSize"]
        width = graph_inputs(graphs["speaker"], placements["speaker"])["latent"][2]
        if width != patched:
            raise DistError(
                f"Axis 2 of input 'latent' of {placements['speaker']} is {width!r}, while"
                f" pipelineConfig's latentDim × speakerPatchSize is {patched}"
            )
    _assert_irodori_dit_boundary(graphs, placements, pipeline_config)


def _assert_irodori_dit_boundary(
    graphs: Mapping[str, Mapping[str, Any]],
    placements: Mapping[str, Path],
    pipeline_config: Mapping[str, Any],
) -> None:
    """Match the boundary of the DiT's two graphs (`dit_context` → `dit`) against the declarations
    of both and pipelineConfig.

    MUST: the number of boundary tensors (= 2 × the block count) is **derived from the outputs of
    the baked `dit_context`** (no copied block count — ADR 0114). Then the following are checked:

    - the inputs of `dit_context` are {@link IRODORI_CONTEXT_INPUTS} and its outputs are in the
      order of {@link irodori_context_kv_names} (the runtime binds resident tensors by name)
    - the inputs of `dit` are the head inputs of the declared flow parameterization
      ({@link irodori_dit_head_inputs} — `flowParameterization` of pipelineConfig, absent = RF) +
      the same boundary names in order, and it has 1 output; a MeanFlow `delta_embed` has the
      same shape as `t_embed` (the same embedding of another scalar — ADR 0150)
    - each boundary tensor has the same shape on both sides, and the condition-side length (axis 1)
      equals the sum of the 3 condition segments of `mask` — otherwise the K/V concatenation of
      `dit` keeps a different length while only the mask points at the segments
    - axis 3 of `dit`'s `mask` is `S+<sum of the 3 condition segments>` (the derived dimension of
      ADR 0046)
    """
    present = {role for role in ("dit_context", "dit") if role in graphs}
    if not present:
        return
    if present != {"dit_context", "dit"}:
        missing = sorted({"dit_context", "dit"} - present)
        raise DistError(
            f"One side of the DiT pair, {missing}, is missing from the series — `dit` takes the"
            " outputs of `dit_context` as inputs, so one alone cannot run (ADR 0114)"
        )
    context, dit = graphs["dit_context"], graphs["dit"]
    context_path, dit_path = placements["dit_context"], placements["dit"]
    names = tuple(graph_inputs(context, context_path))
    if names != IRODORI_CONTEXT_INPUTS:
        raise DistError(
            f"The graph inputs of {context_path} are {list(names)}, not the expected"
            f" {list(IRODORI_CONTEXT_INPUTS)}"
        )
    outputs = context.get("outputs")
    if not isinstance(outputs, list) or not outputs or len(outputs) % 2:
        raise DistError(
            f"The graph outputs of {context_path}, {outputs!r}, are not K / V pairs"
            " (2 × the block count)"
        )
    boundary = irodori_context_kv_names(len(outputs) // 2)
    if tuple(outputs) != boundary:
        raise DistError(
            f"The graph outputs of {context_path} are {outputs[:4]}…, not in the order of the"
            f" boundary names {list(boundary[:4])}… — the runtime binds resident tensors by name,"
            " so a different spelling or order reads the K / V of another block"
        )
    dit_inputs = graph_inputs(dit, dit_path)
    flow = pipeline_config.get("flowParameterization", IRODORI_FLOW_RF)
    expected = (*irodori_dit_head_inputs(flow), *boundary)
    if tuple(dit_inputs) != expected:
        raise DistError(
            f"The graph inputs of {dit_path} are {list(dit_inputs)[:5]}… ({len(dit_inputs)}),"
            f" not the order derived from the outputs of {context_path}, {list(expected[:5])}…"
            f" ({len(expected)})"
        )
    dit_outputs = dit.get("outputs")
    if not isinstance(dit_outputs, list) or len(dit_outputs) != 1:
        raise DistError(f"The graph of {dit_path} does not have exactly 1 output ({dit_outputs!r})")
    if flow == IRODORI_FLOW_MEANFLOW and dit_inputs["delta_embed"] != dit_inputs["t_embed"]:
        raise DistError(
            f"{dit_path}: input 'delta_embed' is {dit_inputs['delta_embed']!r}, while 't_embed' is"
            f" {dit_inputs['t_embed']!r} — both are the host's timestep embedding of one scalar"
        )
    total = sum(
        pipeline_config[field_name] for field_name in ("maxTextLen", "speakerRows", "maxCaptionLen")
    )
    values = context.get("values")
    if not isinstance(values, Mapping):
        raise DistError(f"{context_path}: the IR metadata has no values")
    for name in boundary:
        entry = values.get(name)
        produced = entry.get("shape") if isinstance(entry, Mapping) else None
        consumed = dit_inputs[name]
        if produced != consumed:
            raise DistError(
                f"The shape of boundary '{name}' disagrees: {context_path} outputs {produced!r} /"
                f" {dit_path} takes {consumed!r}"
            )
        if len(consumed) < 2 or consumed[1] != total:
            raise DistError(
                f"{dit_path}: axis 1 of input '{name}' is {consumed[1:2]!r}, while the sum of"
                f" pipelineConfig's 3 condition segments is {total} — the K / V concatenation and"
                " the mask's segments would have different lengths"
            )
    # The `mask` of `dit` is declared with the length "latent S + the 3 condition segments" (the
    # derived dimension of ADR 0046). If the sum of the segments is off, only the mask's segments
    # silently point at other positions.
    symbols = dit.get("symbols")
    if not isinstance(symbols, list) or len(symbols) != 1:
        raise DistError(f"{dit_path}: there is not exactly 1 symbolic dimension ({symbols!r})")
    declared_mask = dit_inputs["mask"][3]
    if declared_mask != f"{symbols[0]}+{total}":
        raise DistError(
            f"Axis 3 of input 'mask' of {dit_path} is {declared_mask!r}, while the sum of"
            f" pipelineConfig's 3 condition segments is {total} (expected '{symbols[0]}+{total}')"
        )


def assert_irodori_calib_provenance(sources: IrodoriSources, flow: str = IRODORI_FLOW_RF) -> None:
    """Confirm from the writer's record that the i4 series was rounded with **calibration allowed
    for distribution** (GPTQ × a budget at or above the floor).

    MUST: neither the calibration method nor the budget changes a single byte of the storage format
    (the grid stays RTN i4 g32 — only the rounded values and the scale ledger change). So the
    structural checks and header dtype checks of `verify_dist` and {@link assert_irodori_graphs}
    all **let it through**. `--no-calib` and `--calib-steps 1` are opt-outs for smoke runs, yet if
    their output slipped into a distribution it could not be told from the assets and the only
    symptom would be degraded audio — so the same discipline as anima's `assert_calib_provenance`
    applies here too: "the same fact held by separate scripts is always matched at assembly time".
    The judgement itself has its source of truth in `_shared.calib_provenance` (decided by the same
    single implementation as the golden re-bake check, `irodori.pipeline_ref._shippable_calib`).

    **Every** role that has an i4 series (the DiT's two graphs — {@link IRODORI_DTYPE_ROLES}) is
    checked. The two are the result of rounding the same `TextToLatentRFDiT` in one GPTQ run and
    writing it out through 2 wrappers (ADR 0114), so the records must be identical — if they
    differ, series from separate export runs (another budget, another rounding) sit side by side in
    one preset.

    A role taken as a published container ({@link IrodoriSources.published}) is not checked: its
    series is not what ships, and the published bytes are pinned by digest instead.
    """
    records: dict[str, Any] = {}
    for role in IRODORI_DTYPE_ROLES["i4"]:
        if irodori_role(role, "i4") in sources.published:
            continue
        path = sources.series_by_dtype["i4"] / IRODORI_SERIES_DIRS[role] / CALIB_PROVENANCE_FILE
        if not path.is_file():
            raise DistError(
                f"The i4 series has no record of its calibration conditions: {path}"
                " (re-exporting with `python -m irodori.export --dtype i4` writes it)"
            )
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except ValueError as cause:
            raise DistError(
                f"Cannot parse the record of the calibration conditions: {path} — {cause}"
            ) from cause
        complaint = calib_complaint(
            record, method=CALIB_SHIPPABLE_METHOD, at_least=irodori_calib_floor(flow)
        )
        if complaint is not None:
            raise DistError(
                f"The calibration conditions of the i4 series do not meet the requirement: {path}:"
                f" {complaint} — output baked with `--no-calib` / a smoke budget is not used for"
                " distribution"
            )
        records[role] = record
    if len({json.dumps(record, sort_keys=True) for record in records.values()}) > 1:
        raise DistError(
            f"The records of the i4 series' calibration conditions differ per role ({records}) —"
            " the DiT's two graphs must have been written by one export run (rewrite both with"
            " `python -m irodori.export --dtype i4`)"
        )


def irodori_plan(sources: IrodoriSources, model: str = IRODORI_DEFAULT_MODEL) -> ModelPlan:
    """Build the plan of one Irodori model (every check and the config reading are done here)."""
    assert_model_name(model)
    placements = irodori_placements(sources)
    config = irodori_model_config(sources.model)
    pipeline_config = irodori_pipeline_config(config, irodori_codec_numbers(sources.codec_model))
    assert_irodori_calib_provenance(sources, irodori_flow_parameterization(config))
    assert_irodori_borrowed_weights(sources, model)
    assert_irodori_published_containers(sources)
    for role, source in placements.items():
        assert_storage(role, source, IRODORI_STORAGE_REQUIREMENTS)
        assert_storage_absent(role, source, IRODORI_STORAGE_FORBIDDEN)
    assert_irodori_graphs(placements, pipeline_config)
    return ModelPlan(
        name=model,
        pipeline=IRODORI_PIPELINE,
        artifacts={
            role: Artifact(IRODORI_OUTPUT_PATHS[role], source=source)
            for role, source in placements.items()
        },
        weights=IRODORI_WEIGHTS,
        assets=IRODORI_ASSETS,
        quants=complete_quant_weights(IRODORI_WEIGHTS, irodori_quants(model)),
        default_quant=IRODORI_MODEL_DEFAULT_QUANTS.get(model, IRODORI_DEFAULT_QUANT),
        pipeline_config=pipeline_config,
    )


def irodori_dist_plan(series_dir: Path, model: str) -> ModelPlan:
    """Build the plan of one Irodori model from the parent of `--series` (the CLI's dispatch
    target)."""
    return irodori_plan(irodori_sources(series_dir, model), model)


#: The two rectified-flow models, one repository each (ADR 0092 decision 1).
IRODORI_V4_MODEL = "v4-small"
IRODORI_V41_MODEL = "v4.1-small"

#: The MeanFlow student distilled from v4.1-small by upstream (ADR 0150) — its own repository.
IRODORI_V41_MF_MODEL = "v4.1-small-mf"

#: Model → its default quant preset, for a model whose default is not the family's
#: {@link IRODORI_DEFAULT_QUANT}. v4.1-small-mf: `i8` by the maintainer's listening of 2026-11-14
#: (`i8` was fine; `i8-a8` showed no audible degradation either — ADR 0150).
IRODORI_MODEL_DEFAULT_QUANTS: Mapping[str, str] = {IRODORI_V41_MF_MODEL: "i8"}

#: Model → (preset name → its own description), for a model whose presets were judged apart from
#: the family's rulings ({@link irodori_quants}). The family's descriptions carry the v4 listening
#: check and the v4 speed measurement; for v4.1-small-mf they say only what was established for
#: that model — the maintainer's listening of 2026-11-14 covered `i8` and `i8-a8`, and nothing
#: else was judged or measured for it (ADR 0150). `f32` and `i8+dit4` state facts of the storage
#: only, so they keep the family's text.
IRODORI_MODEL_SEAT_DESCRIPTIONS: Mapping[str, Mapping[str, str]] = {
    IRODORI_V41_MF_MODEL: {
        "f16": "Every graph stored as f16 and computed in f32 — half the f32 download.",
        "i8": "Every graph stored as int8 and computed in f32 — about a quarter of the f32"
        " download; it sounded fine in a listening check.",
        "i8-a8": "The int8 weights with per-token int8 activations in the DiT's linear layers"
        " — the same download as plain int8, with no audible degradation in a listening check.",
    },
}

#: Roles of v4.1-small that are byte-identical to v4-small, so v4.1-small's distribution
#: references them in v4-small's repository instead of storing them a second time (ADR 0038 §7
#: addendum 2026-10-28; 26 roles = 90 files, 5,637 MiB, measured 2026-10-28).
#:
#: Derived from the dtype tables, minus the containers whose bytes differ between the two models:
#: upstream v4.1 retrained only the duration predictor (`duration_*`), and the i4 DiT (`dit_i4`)
#: is GPTQ-calibrated through each model's own reference loop — its parts differ too (measured
#: 2026-10-28; `dit_context_i4` is identical). MUST: a container stays local **whole** when any of
#: its parts differs — a reference covers every part of a container or none (ADR 0109 decision 3),
#: and a role declared here whose bytes differ fails the build (`karume.dist.external_refs`).
IRODORI_V41_SHARED_ROLES: tuple[str, ...] = (
    *(
        irodori_role(role, dtype)
        for dtype, roles in IRODORI_DTYPE_ROLES.items()
        for role in roles
        if role != "duration" and (role, dtype) != ("dit", "i4")
    ),
    *IRODORI_ASSETS.values(),
)


#: The graph roles of the DiT pair — the only ones a MeanFlow distillation retrains (upstream's
#: recipe trains the whole student DiT and freezes the condition encoders and the duration
#: predictor — `docs/meanflow.md`; measured for v4.1-small-mf 2026-11-14: only DiT-side tensors
#: differ from v4.1-small, plus the 3 of `delta_cond_module`).
IRODORI_DIT_ROLES: tuple[str, ...] = ("dit", "dit_context")


def _v41_owner(role: str) -> str:
    """The model whose repository physically holds the bytes of a v4.1-small role: v4.1-small
    itself for what it stores, v4-small for what it borrows ({@link IRODORI_V41_SHARED_ROLES})."""
    return IRODORI_V4_MODEL if role in IRODORI_V41_SHARED_ROLES else IRODORI_V41_MODEL


#: Model → (role name → the model whose repository holds the **published container** the role is
#: taken as). Such a role is not re-exported: the assembly reads the container from the holder's
#: mirror ({@link irodori_published_sources}) and requires it to be the published bytes
#: ({@link IRODORI_PUBLISHED_PART0}, {@link assert_irodori_published_containers}). When the holder
#: is another model, the role is also a cross-repository reference there
#: ({@link irodori_references}); when it is the model itself, the container is stored again as is.
#:
#: Why the published bytes and not the series: a fresh export cannot reproduce them. The published
#: non-DiT containers predate the `provenance.notice` field (part 0 differs), and the GPTQ i4 series
#: rounds about 16% of the DiT weights differently on another machine (measured 2026-11-14 — an
#: environment difference, not an upstream one). A reassembly that only adds or changes a
#: non-container file (the rare-token split table of ADR 0149, the card) must therefore not move a
#: published container. Re-exporting a model's containers is a deliberate change: remove its
#: entries here and in {@link IRODORI_PUBLISHED_PART0}.
#:
#: - v4-small and v4.1-small take every graph role as published: v4-small from its own repository,
#:   v4.1-small its own roles (`duration_*`, `dit_i4`) from its own and the rest from v4-small
#:   ({@link IRODORI_V41_SHARED_ROLES}).
#: - v4.1-small-mf (ADR 0150) exports only the DiT pair, whose bytes are new, and takes every other
#:   graph role from the repository that physically holds it — references are not followed further
#:   (`karume.dist.external_refs`), so a role v4.1-small itself borrows points at v4-small directly:
#:   `duration_*` at v4.1-small, the rest at v4-small. That a borrowed graph is the student's own is
#:   checked on the inputs: every checkpoint tensor it is built from, and the configs, equal the
#:   holder's ({@link assert_irodori_borrowed_weights}).
#:
#: The tokenizer asset is not a container and always comes from the series
#: ({@link IRODORI_TOKENIZER_HOLDERS} says where it is referenced instead of stored).
IRODORI_PUBLISHED: Mapping[str, Mapping[str, str]] = {
    IRODORI_V4_MODEL: {
        irodori_role(role, dtype): IRODORI_V4_MODEL
        for dtype, roles in IRODORI_DTYPE_ROLES.items()
        for role in roles
    },
    IRODORI_V41_MODEL: {
        irodori_role(role, dtype): _v41_owner(irodori_role(role, dtype))
        for dtype, roles in IRODORI_DTYPE_ROLES.items()
        for role in roles
    },
    IRODORI_V41_MF_MODEL: {
        irodori_role(role, dtype): _v41_owner(irodori_role(role, dtype))
        for dtype, roles in IRODORI_DTYPE_ROLES.items()
        for role in roles
        if role not in IRODORI_DIT_ROLES
    },
}

#: Holder model → (role name → sha256 of **part 0** of its published container). Part 0's
#: descriptor declares the sha256 of every other part (container-v1 §7), so this one digest pins
#: the whole container. Recorded 2026-11-14 from the Hub's main of `karume-irodori-v4-small` and
#: `karume-irodori-v4.1-small` (LFS oids; the local mirrors were byte-compared with every file of
#: those commits the same day).
IRODORI_PUBLISHED_PART0: Mapping[str, Mapping[str, str]] = {
    IRODORI_V4_MODEL: {
        "backbone_f16": "f7da9ed1cc9424b89217515aaa15aa44eee43f3a10d8f183ce4da0d310dda5b5",
        "backbone_f32": "4bf46f4feed0c8c51491d6d3e086b5e68322a31cd29cc3f7d9a5003f2195008c",
        "backbone_i8": "4faccb6361996b0efc66b123fd76547326cf7c0d141de8d400a193e77c366a4b",
        "caption_proj_f16": "8499b683a87ece8dcec9c7febb8c3b2f8a2fb508a8f7e68d1e966b45c4d309b2",
        "caption_proj_f32": "66e346f9060a83a73cc60efe47a4797d7a57b5e26545bed59cd0ceb354e4bebc",
        "caption_proj_i8": "ceb3057974e623fe49e44ec0bb8c2cab8dfede84c6af2a5746ffb28e344afa22",
        "codec_decoder_f16": "45c1acd0a0a1d7b5a9bb26d926cc392724f93c328e3b0ef70f35ca33dbaab729",
        "codec_decoder_f32": "0240e41c08ba0bde8683bd807f20a046af0a6cc3db448d0a7330b3eec6f15db6",
        "codec_decoder_i8": "ea18b51377c3bc97593d1410c3567cd9ceac2d4b6f52421866404e6eec6dda38",
        "codec_encoder_f16": "ec0547b90a0a5a63cf329561af9e2c0c69c2daec1c7a0973d395d1c922b84d71",
        "codec_encoder_f32": "d8da3a7ae8f82f1ca004a7bf3ff4a0f2aec34ab6325721fa83d7e91d200cd08d",
        "codec_encoder_i8": "6c890e50ef6755f8e780e706ac13b67a5cdfdd8595f1f6cd3eb54f4c1002f748",
        "dit_f16": "efbf0477e04bf491c9f27ad1b1610ca636cd3a4ca5b2175fd4b1b51e893a4495",
        "dit_f32": "a3aa39059f6818548e38c6ceb23a55523b22fe3099d34dc81015fe68635044b3",
        "dit_i4": "e92aca74e1cb1619d33a927ded3dc16cbae7ed0063a8a109de6fb88b27f60ba3",
        "dit_i8": "7d5bd3804ca24274bd6d26aa18d42b2af42cf9350ce7708b98247ba96943a21b",
        "dit_context_f16": "22d3838f813e56c6a9cff674d3e86c3102d42b992522c3c1a020d964cecc17b1",
        "dit_context_f32": "3a80725741e9bfbd8aa88084e4ad7f62b68f48c848f270ef3d5a35813ce50d3e",
        "dit_context_i4": "09a060761f79c243e44ecfa8556babf3a232e8886291f06cec421bb8f206b2e0",
        "dit_context_i8": "af029d181069d971626bff723095decf467ae84134184be0b5200309a65a7246",
        "duration_f16": "45cc2c32b12f3e1b5d69326ffb49f9f22f1f18f84a85b7f00d0aea520d930c93",
        "duration_f32": "4ada94f3600d0bd7944d8e0e5f5637d5e19900e1c17e22bc27d073255ec18bae",
        "duration_i8": "a05dfc0d9f4338780ee8992d64d0c25adb0ccbea0524e7f450d5a2e2d74f8232",
        "speaker_f16": "e410fab58eed5a6a24451b6c80612457dade40fcf11cb7b3155e5526cc872df8",
        "speaker_f32": "2d317da551e0484e9bc8485a7cfcf1e6ce31e7660fb58f167f495f279b5b7f6a",
        "speaker_i8": "d1ad8e2978a95cd25e70d314ce5db2fcead74c27b8e52ebe74fe7a23c2d66338",
        "text_proj_f16": "1fafc49e129bb96d2380d2ee77b6893f67c3bfa47454753f36aa0c6d099a0245",
        "text_proj_f32": "4e64f1bba21f6f8f8d5321f748cbec93565c85c2f2c821f941ead26ead8074c5",
        "text_proj_i8": "fc56eb9388a46ee7252d8f2ad3cff9e532cebffa105913d0ce988fde146a7cff",
    },
    IRODORI_V41_MODEL: {
        "dit_i4": "09ac1ac698596ad5f9ff652b5b174a968ce2ada97b0e7de1f50d0614bb20f0a8",
        "duration_f16": "b424211ef9b9cc131a796dd9e6dfb6550bc0d8d54e7ad52920b65499213472e3",
        "duration_f32": "a7e80841ce62c9086d4a3ee12f20c7fdb6e69e2e1cd331e6db14d2e9c1a259b2",
        "duration_i8": "a59a496c4213d12792cee402012802032f92fa8a15ae37db16cd35c8878bdb98",
    },
}

#: Model → the model whose repository holds its tokenizer asset, for the models that reference it
#: there instead of storing it. The three checkpoints name the same text tokenizer, and the asset
#: they produce is byte-identical (measured 2026-11-14, rare-token split table included — ADR
#: 0149); the core's reference check compares the bytes on every build, so a model whose asset
#: drifted fails instead of pointing at another tokenizer. ADR 0150: v4.1-small-mf references it
#: too, now that v4-small's asset carries the table.
IRODORI_TOKENIZER_HOLDERS: Mapping[str, str] = {
    IRODORI_V41_MODEL: IRODORI_V4_MODEL,
    IRODORI_V41_MF_MODEL: IRODORI_V4_MODEL,
}


def irodori_published_sources(model: str, dist_root: Path) -> Mapping[str, PublishedContainer]:
    """Role name → the published container a model takes the role as ({@link IRODORI_PUBLISHED}),
    in the holder's local mirror (`<dist_root>/<holder repository>/<holder>/<role path>`), with
    its pinned digest.

    MUST: a declared role without a pin fails — taking a container "as published" with nothing to
    say which bytes were published would accept whatever the mirror holds.
    """
    published: dict[str, PublishedContainer] = {}
    for role, holder in IRODORI_PUBLISHED.get(model, {}).items():
        digest = IRODORI_PUBLISHED_PART0.get(holder, {}).get(role)
        if digest is None:
            raise DistError(
                f"{model} takes {role} as published by {holder}, but there is no published digest"
                " for it (IRODORI_PUBLISHED_PART0)"
            )
        published[role] = PublishedContainer(
            holder=holder,
            path=dist_root / irodori_repo_name(holder) / holder / IRODORI_OUTPUT_PATHS[role],
            part0_sha256=digest,
        )
    return MappingProxyType(published)


def assert_irodori_published_containers(sources: IrodoriSources) -> None:
    """Check that every container taken as published ({@link IrodoriSources.published}) is the
    published bytes: part 0 has the pinned sha256, and every other part has the sha256 part 0's
    descriptor declares for it.

    MUST: every part is hashed, not a sample — a mirror rebuilt locally from the series differs in
    some parts only (the i4 rounding), and a stored copy is placed as is while a borrowed one is
    only referenced, so nothing later compares it with the descriptor. Restoring the mirror from the
    Hub is the fix, never re-pinning.
    """
    for role, published in sources.published.items():
        parts = component_parts(published.path)
        if not all(part.is_file() for part in parts):
            raise DistError(
                f"{role}: the published container of {irodori_repo_name(published.holder)} is not"
                f" in its mirror: {published.path}"
            )
        actual = sha256_file(parts[0])
        if actual != published.part0_sha256:
            raise DistError(
                f"{role}: {parts[0]} is not the published container of"
                f" {irodori_repo_name(published.holder)} (part 0 sha256 {actual}, published"
                f" {published.part0_sha256}) — restore the mirror from the Hub instead of"
                " rebuilding it from the series"
            )
        if len(parts) == 1:
            continue
        try:
            descriptor = verify_container(parts).read.model
        except (ContainerError, ContainerFormatError) as cause:
            raise DistError(f"{role}: {parts[0]} cannot be read as a container: {cause}") from cause
        declared = (
            {} if descriptor is None else {part.index: part.sha256 for part in descriptor.parts}
        )
        for index, part in enumerate(parts[1:], start=1):
            if sha256_file(part) != declared.get(index):
                raise DistError(
                    f"{role}: {part} is not the part {index} the published part 0 declares"
                    f" ({declared.get(index)}) — restore the mirror from the Hub"
                )


#: Graph role → the checkpoint's state_dict prefixes it is built from (the attribute names of
#: `irodori_tts/model.py`, as `irodori.export` loads them — `text_norm` sits inside `duration`,
#: `caption_norm` inside `caption_proj`'s second output, `speaker_norm` inside `speaker`). The
#: codec is a separate repository with its own weights, so it has none.
#:
#: The DiT pair (borrowed by v4.1-small, whose DiT upstream did not retrain): `dit` is built from
#: the blocks and the modules around them; `dit_context` from the blocks' condition-side
#: projections and the two condition norms it applies (`irodori.export.DitContextGraph`). It is
#: compared over every block tensor — a superset of what it reads, so a borrower whose blocks
#: differ anywhere is refused rather than half-checked.
IRODORI_ROLE_TENSOR_PREFIXES: Mapping[str, tuple[str, ...]] = {
    "backbone": ("pretrained_text_backbone.",),
    "text_proj": ("text_encoder.",),
    "caption_proj": ("caption_encoder.", "caption_norm."),
    "speaker": ("speaker_encoder.", "speaker_norm."),
    "duration": ("duration_predictor.", "text_norm."),
    "dit": ("blocks.", "cond_module.", "in_proj.", "out_proj.", "out_norm."),
    "dit_context": ("blocks.", "text_norm.", "caption_norm."),
    "codec_decoder": (),
    "codec_encoder": (),
}

#: The `config_json` keys that may differ between a borrower and the owner (only the flow
#: parameterization — it changes the DiT, never the borrowed graphs).
IRODORI_BORROW_CONFIG_EXEMPT = frozenset({IRODORI_FLOW_CONFIG_KEY})


def _checkpoint_tensors(path: Path, prefixes: tuple[str, ...]) -> dict[str, bytes]:
    """The raw bytes (and dtype / shape) of every tensor of a checkpoint under `prefixes`."""
    picked: dict[str, bytes] = {}
    with safe_open(str(path), framework="np") as handle:
        for key in handle.keys():  # noqa: SIM118 — safe_open has no __iter__
            if key.startswith(prefixes):
                array = handle.get_tensor(key)
                picked[key] = f"{array.dtype}:{list(array.shape)}:".encode() + array.tobytes()
    return picked


def _checkpoint_metadata(path: Path) -> Mapping[str, str]:
    with safe_open(str(path), framework="np") as handle:
        metadata = handle.metadata()
    if not isinstance(metadata, dict):
        raise DistError(f"{path} has no __metadata__")
    return metadata


def assert_irodori_borrowed_weights(sources: IrodoriSources, model: str) -> None:
    """Check that every role `model` borrows — takes as a published container another model holds
    ({@link IrodoriSources.published}) — is built from **exactly the owner's inputs**: the
    checkpoint tensors under its prefixes ({@link IRODORI_ROLE_TENSOR_PREFIXES}) are
    byte-identical, the text-encoder config is equal, and the model config differs at most in
    {@link IRODORI_BORROW_CONFIG_EXEMPT}.

    MUST: a borrowed role is the owner's published container, never re-exported, so this is the
    check that the borrowing is the same graph and the same weights — without it a borrower whose
    duration predictor (say) had been retrained would ship the owner's silently. The owner's
    checkpoint is the sibling input directory (`inputs/irodori/<owner>/`).
    """
    borrowed = {
        role: published.holder
        for role, published in sources.published.items()
        if published.holder != model
    }
    if not borrowed:
        return
    own = sources.model / IRODORI_CKPT_FILE
    if not own.is_file():
        raise DistError(f"Assembly input missing: {own}")
    owners: dict[str, set[str]] = {}
    for role, owner in borrowed.items():
        # `<graph role>_<dtype>` — the dtype never contains `_` ({@link IRODORI_WEIGHT_DTYPES}).
        owners.setdefault(owner, set()).add(role.rsplit("_", 1)[0])
    for owner, graph_roles in sorted(owners.items()):
        theirs = sources.model.parent / owner / IRODORI_CKPT_FILE
        if not theirs.is_file():
            raise DistError(
                f"{model} borrows {sorted(graph_roles)} from {owner}, whose checkpoint is needed to"
                f" check that the weights are the same: {theirs} is missing"
            )
        mine_meta, their_meta = _checkpoint_metadata(own), _checkpoint_metadata(theirs)
        mine_config = json.loads(mine_meta[IRODORI_CONFIG_META_KEY])
        their_config = json.loads(their_meta[IRODORI_CONFIG_META_KEY])
        differing = sorted(
            key
            for key in set(mine_config) | set(their_config)
            if key not in IRODORI_BORROW_CONFIG_EXEMPT
            and mine_config.get(key) != their_config.get(key)
        )
        # Compared as JSON values: the two checkpoints serialize the same config with another key
        # order (measured 2026-11-14 — v4-small / v4.1-small / v4.1-small-mf).
        text_key = "text_encoder_config_json"
        if json.loads(mine_meta.get(text_key, "null")) != json.loads(
            their_meta.get(text_key, "null")
        ):
            differing.append(text_key)
        if differing:
            raise DistError(
                f"{model} borrows {sorted(graph_roles)} from {owner}, but the configs differ in"
                f" {differing} — the borrowed graphs would not be the ones this checkpoint builds"
            )
        prefixes = tuple(
            prefix
            for graph_role in sorted(graph_roles)
            for prefix in IRODORI_ROLE_TENSOR_PREFIXES[graph_role]
        )
        if not prefixes:
            continue
        mine, theirs_tensors = (
            _checkpoint_tensors(own, prefixes),
            _checkpoint_tensors(theirs, prefixes),
        )
        if not mine:
            raise DistError(f"{own} has no tensor under {list(prefixes)}")
        unequal = sorted(
            key
            for key in set(mine) | set(theirs_tensors)
            if mine.get(key) != theirs_tensors.get(key)
        )
        if unequal:
            raise DistError(
                f"{model} borrows {sorted(graph_roles)} from {owner}, but {len(unequal)} checkpoint"
                f" tensors differ ({unequal[:3]}…) — a borrowed role must be the owner's weights"
            )


def irodori_references(model: str) -> tuple[CrossRepoReference, ...]:
    """`Pipeline.references` of Irodori: every role a model takes from another model's repository
    — the published containers of {@link IRODORI_PUBLISHED} whose holder is another model, and the
    tokenizer asset of {@link IRODORI_TOKENIZER_HOLDERS} — one reference per holder, in name order
    (v4-small first). v4.1-small references {@link IRODORI_V41_SHARED_ROLES} at v4-small;
    v4.1-small-mf references v4-small and v4.1-small; v4-small stays self-contained (it is the one
    upstream retrained from, and the e2e checks read its mirror file by file)."""
    borrowed = {
        role: holder for role, holder in IRODORI_PUBLISHED.get(model, {}).items() if holder != model
    }
    tokenizer_holder = IRODORI_TOKENIZER_HOLDERS.get(model)
    if tokenizer_holder is not None:
        borrowed.update(dict.fromkeys(IRODORI_ASSETS.values(), tokenizer_holder))
    owners = sorted(set(borrowed.values()))
    return tuple(
        CrossRepoReference(
            repo_name=irodori_repo_name(owner),
            model=owner,
            roles=tuple(role for role, holder in borrowed.items() if holder == owner),
        )
        for owner in owners
    )


#: The MIT copyright lines (`LICENSE.md` at the root of the distribution repository). The 3 upstream
#: HF repositories (the 2 versions of the model and the codec) only declare `license: mit` and do
#: not bundle the license text, so the text is built from the shared template.
#:
#: - The Aratako line is the line of the `LICENSE` in the upstream implementation repository
#:   (GitHub `Aratako/Irodori-TTS`), applied to the same author's weights (model and codec) too
#:   (maintainer ruling of 2026-09-24).
#: - The SB Intuitions line is the `LICENSE` line bundled with the base of the text backbone
#:   (`sbintuitions/modernbert-ja-310m`) (checked on upstream main on 2026-09-24). The backbone is
#:   a fine-tune of it, so MIT's "include the copyright notice in all copies" cannot be dropped in a
#:   redistribution of the derivative either (the same direction as lucida's 2 lines).
#:
#: Ordered closest to the distribution first, per the convention of `mit_license`. Both the v4 and
#: the v4.1 repository get the same 2 lines (the backbone is identical in both versions — measured
#: for `irodori.card.IRODORI_TEXT_BACKBONE_MODEL`; both versions use the same single codec from a
#: separate repository).
IRODORI_COPYRIGHTS: tuple[str, ...] = (
    "Copyright (c) 2026 Aratako",
    "Copyright (c) 2025 SB Intuitions",
)

#: The components covered by Apache 2.0 (the keys of the manifest's weights = the keys of
#: {@link IRODORI_CODEC_DIRS}). NOTICE and the heading of `LICENSE.md` build from the same word —
#: so that neither ends up pointing at other components.
_IRODORI_CODEC_COMPONENTS = " / ".join(f"`{role}`" for role in IRODORI_CODEC_DIRS)

#: The modification notice. MIT does not require one, but that this is a quantized distribution
#: with a changed storage format is the first fact users want to confirm, so one is emitted in the
#: same slot as `LICENSE.md`. Upstream says the codec weights derive from
#: `facebook/dacvae-watermarked` (Apache 2.0), and Apache 2.0 §4(b) requires modified files to carry
#: a notice of the modification. So, together with MIT's attribution, **every modification that
#: reaches the stored values** is listed (the 3 codec items come from the preprocessing and boundary
#: sections of `irodori/dacvae/export.py` — folding in weight_norm, trimming the encoder's `in_proj`
#: to the mean side, bypassing the watermark branch). Do not write "the f32 series keeps upstream's
#: values" — the codec's f32 series differs from upstream's stored values by the 3 items above (a
#: bit-exact output has been measured, but the stored values are not identical).
#:
#: The codec's original weights ({@link irodori.card.IRODORI_CODEC_ORIGIN_MODEL} — Apache 2.0) are
#: attributed by name. Apache 2.0 §4(a) requires a copy of the license text
#: ({@link irodori_license_markdown}), and §4(c) requires the attributions to be retained. What
#: §4(d) requires retaining (upstream's `NOTICE`) exists neither on HF nor on GitHub (the file
#: listings of 2026-09-24 — the history is in the comment on the same constant in `irodori.card`).
#:
#: MUST: do not name the model's upstream repository (write "the checkpoint listed in
#: `README.md`"). `root_files` is one set fixed per Pipeline, and this Pipeline builds the 2
#: repositories v4 / v4.1, so naming a version would make one repository's notice disagree with its
#: contents. The card names the version from the manifest.
#:
#: MUST: the text must correspond to the distribution's contents — it is valid prose as a value, so
#: `verify_dist` and the manifest checks let it through, and a mismatch would only be noticed after
#: shipping. The component names are built from the keys of {@link IRODORI_WEIGHTS} (= the
#: manifest's weights), and the 2 bundled upstream models are taken from the same constants as the
#: card.

IRODORI_NOTICE_MARKDOWN = f"""# NOTICE

This repository redistributes a modified form of the Irodori-TTS checkpoint listed in `README.md`
and of the DACVAE codec `{IRODORI_CODEC_MODEL}` (which upstream ships as a
separate repository), both licensed under the MIT License (see `LICENSE.md`). The checkpoint's
text backbone was fine-tuned from `{IRODORI_TEXT_BACKBONE_MODEL}`, which is licensed under the
MIT License as well; its copyright notice is kept in `LICENSE.md`.

The codec's weights derive from
[`{IRODORI_CODEC_ORIGIN_MODEL}`](https://huggingface.co/{IRODORI_CODEC_ORIGIN_MODEL}), which is
licensed under the Apache License, Version 2.0: the model card of `{IRODORI_CODEC_MODEL}` states
that it was derived from
[`{IRODORI_CODEC_PARENT_MODEL}`](https://huggingface.co/{IRODORI_CODEC_PARENT_MODEL}) (MIT),
itself fine-tuned from `{IRODORI_CODEC_ORIGIN_MODEL}`. The Apache License 2.0 text is included in
`LICENSE.md` and covers the codec components ({_IRODORI_CODEC_COMPONENTS}); the changes made to
them are listed below.

The following changes were made:

- The model and the codec were split into these graphs and re-expressed in the Karume container
  format (a `.krm` part sequence whose first part carries the graph and model descriptors):
  {", ".join(f"`{name}`" for name in IRODORI_WEIGHTS)}.
- The codec was adapted for deterministic inference. Its weight-normalized convolutions are stored
  as their effective weights (weight normalization folded in, in place of the upstream
  `weight_g` / `weight_v` pairs). `codec_encoder` keeps only the mean half of
  `quantizer.in_proj` (the first half of its output channels; the scale half is unused when
  encoding deterministically). `codec_decoder` bypasses the watermarking branch, keeping only its
  waveform output layers on the decoding path.
- Constant sub-expressions of the graphs, such as rotary position tables and the reciprocals in
  the codec's Snake activations, were precomputed and stored as constants.
- **The weights were quantized**: every graph is also stored as `f16` and `i8`, and the DiT's two
  graphs (`dit` and `dit_context`) add an `i4` series rounded with GPTQ calibration. The `f32`
  series is not quantized.

No retraining and no fine-tuning were performed. The original checkpoints are not distributed here.
"""


def irodori_license_markdown() -> str:
    """`LICENSE.md` — the MIT (all components) and Apache 2.0 (the 2 codec components) texts side
    by side in one file.

    `root_files` has a single `LICENSE.md` slot (`karume.dist.LEGAL_PATHS`), so the second license
    text follows a separator (`---`) and a heading that names its scope. The file starts with the
    MIT text + the copyright lines ({@link IRODORI_COPYRIGHTS}) as is — MIT covers every component,
    the codec included (`IRODORI_CODEC_MODEL` and its parent both declare `license: mit`).

    MUST: neither license text is touched (reformatting it would stop it from being a copy of the
    permission notice). The Apache text breaks into Markdown paragraphs and lists because of its
    leading indentation, so it goes verbatim inside a fence (``` does not occur in the text — the
    digest freeze in `tests/test_licenses.py` catches any change to it).
    """
    return (
        mit_license(IRODORI_COPYRIGHTS)
        + "\n---\n\n"
        + f"## Apache License 2.0 — codec components ({_IRODORI_CODEC_COMPONENTS})\n\n"
        + f"The codec components {_IRODORI_CODEC_COMPONENTS} derive from"
        + f" `{IRODORI_CODEC_ORIGIN_MODEL}`, which is\n"
        + "licensed under the Apache License, Version 2.0 (see `NOTICE.md`). They are covered by"
        + " the license\nbelow in addition to the MIT License above; every other component is"
        + " covered by the MIT License\nalone.\n\n"
        + "```text\n"
        + apache_license_2_0()
        + "```\n"
    )


def irodori_root_files() -> dict[str, str]:
    """Legal texts at the root of the distribution repository (`karume.dist.Pipeline.root_files`).

    `LICENSE.md` holds MIT + Apache 2.0 side by side ({@link irodori_license_markdown}), and
    `NOTICE.md` holds the attribution and the modification notice ({@link IRODORI_NOTICE_MARKDOWN}).
    """
    return {
        "LICENSE.md": irodori_license_markdown(),
        "NOTICE.md": IRODORI_NOTICE_MARKDOWN,
    }


#: The one entry for `--pipeline irodori` (the driver merges it into core's PIPELINES).
PIPELINE = Pipeline(
    default_model=IRODORI_DEFAULT_MODEL,
    repo_name=irodori_repo_name,
    plan=irodori_dist_plan,
    # There is only one attribution profile (the set of model + text backbone + codec (+ its
    # original Apache 2.0 weights) is the same for every model; only the model's upstream repository
    # changes) — with no choice to make, it may be omitted. The moment a second family appears, it
    # must be given explicitly.
    card_profiles={"irodori": render_irodori_model_card},
    # The redistribution terms of the upstream licenses (MIT, and Apache 2.0 of the codec's original
    # weights) apply to the one distribution repository, so they are set here once regardless of how
    # many times it is assembled (ADR 0092 decision 7).
    root_files=irodori_root_files(),
    references=irodori_references,
)
