"""Anima's distribution recipe — series layout, output paths, quants, card (ADR 0065 decision 2).

The generic assembly engine (placement, folding of shared slots, sha256, manifest, staging/swap,
verification) lives in `karume.dist`. This module holds only **Anima-specific facts**: what to pick
from which series directory, at which path of the distribution and under which dtype label to lay
it out, and which quant is the default.

The public surface is two Pipelines ({@link OFFICIAL_PIPELINE} / {@link EXTRA_PIPELINE} — ADR
0087) — the repository's dist driver (`tools/export-recipes/dist.py`) merges them into core's
PIPELINES.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

from _shared.calib_provenance import calib_complaint
from _shared.container_read import ContainerReadError, read_asset
from anima.card import (
    ATTRIBUTION_NOTICE,
    NVIDIA_NOTICE,
    RIGHTS_GRANT_STATEMENT,
    render_base_card,
    render_extra_card,
)
from karume.dist import (
    Artifact,
    CrossRepoReference,
    DistError,
    ModelPlan,
    Pipeline,
    WeightFiles,
    assert_component_present,
    assert_model_name,
    assert_storage,
    assert_storage_absent,
    complete_quant_weights,
)

#: Name and role of the per-axis raw tables the host builds the rope tables from (an **asset** of
#: the container — ADR 0109 decision 4). The content is the bytes of one safetensors file as is (the
#: old `rope_base.safetensors` folded in), and the container does not interpret it.
#:
#: MUST: the spelling matches the copy table of the migration CLI (`karume.migrate.EXTRA_ASSETS`) —
#: if the asset name differed between "containers migrated from the old repository" and
#: "containers written directly", the models side could not look it up. The writer
#: (`anima/export.py`) reads it from here too (the side that needs torch imports from the side that
#: does not, never the reverse).
ROPE_BASE_ASSET = "rope_base"
ROPE_BASE_ROLE = "rope-base"

#: Model name of the official Turbo variant (= the default model of the official repository — the
#: upstream README recommends "Turbo first"; ruling of 2026-09-01). Upstream's naming is used as is
#: (ADR 0077). The old `anima-turbo` (a distribution with turbo LoRA v0.2 baked in,
#: hdae/karume-anima-turbo) was replaced by this official checkpoint version and retired from the
#: accepted set (2026-09-01 — the old baked series `outputs/series/anima-turbo-*-dyn` carry
#: `lora_provenance.json`, so plugging one into the new slot by mistake fails in
#: {@link assert_lora_provenance}).
ANIMA_TURBO_MODEL_NAME = "anima-turbo-v1.1"

#: Model name of the plain base distribution. It is used with many steps + CFG, and the negative
#: prompt takes effect only for this line (base / aesthetic / extra) (turbo runs with CFG=1 and does
#: not compute the uncond side).
ANIMA_BASE_MODEL_NAME = "anima-v1.0"

#: Model name of the official Aesthetic variant (the official fine-tune of base toward a
#: high-quality style — upstream's naming as is). It carries its own text_conditioner, because 64
#: tensors still differ from base after f16 rounding (measured 2026-09-01).
ANIMA_AESTHETIC_MODEL_NAME = "anima-aesthetic-v1.1"

#: Pipeline contract (ADR 0041 §2 — per model).
ANIMA_PIPELINE = "anima/1"

#: Series that do not depend on the model name (base's text path / VAE, and the output of the
#: script that writes the tokenizers).
#: MUST: `anima-f16` is a series baked **without baking in a LoRA** (the absence of
#: `lora_provenance.json` is the record of that). turbo's text_conditioner component measured
#: lora_B all zeros = a no-op (the same NOTE in `anima/pipeline_ref.py`), so the turbo slot may
#: share this too.
ANIMA_BASE_SERIES = "anima-f16"
ANIMA_TOKENIZER_SERIES = "anima-demo"

#: Storage label → shared series of the text stack (text_encoder / text_conditioner / vae_decoder —
#: the 3 roles that do not depend on the DiT's storage). Every model reads f16; bf16 is read only by
#: the models that have the original-weights preset `bf16` (all 4 roles bf16 — ADR 0131 decision 3)
#: in {@link AnimaModel.storages}. The bf16 series is also baked without baking in a LoRA (the same
#: MUST and the same safety net as {@link ANIMA_BASE_SERIES} — {@link assert_lora_provenance}).
ANIMA_TEXT_STACK_SERIES: Mapping[str, str] = {"f16": ANIMA_BASE_SERIES, "bf16": "anima-bf16"}


def text_stack_role(component: str, storage: str) -> str:
    """The role name of a text-stack placement (a key of {@link OUTPUT_PATHS}).

    f16 keeps the **bare component name** (`text_encoder` and the like), so adding a storage does
    not rename an existing role; the other storages are `<component>_<storage>` (the same form as
    the transformer's `transformer_<storage>`). The extra repository's cross-repository references
    are derived through this function ({@link ANIMA_EXTRA_SHARED_ROLES}).
    """
    return component if storage == "f16" else f"{component}_{storage}"


#: File that keeps the attribution of a baked-in LoRA (directly under the series target). The
#: reader (here) owns the spelling of the series layout and the writer (`anima/export.py`) reads it
#: from here — it does not move independently in two places.
LORA_PROVENANCE_FILE = "lora_provenance.json"

#: Calibration conditions an i4 series records ({@link assert_calib_provenance}). Calibration
#: **does not change a single byte of the storage form**, so it cannot be told from the assets, the
#: manifest or the header. Yet the quality difference is large enough to split the visual verdict
#: (plain RTN = "a blurry impression overall" / GPTQ = "keeps high quality" — research 2026-08-21
#: §6), so under the same discipline as the LoRA attribution "the side that wrote it records the
#: facts".
CALIB_PROVENANCE_FILE = "calib_provenance.json"

#: The rounding method allowed for distribution (`method` of {@link CALIB_PROVENANCE_FILE}). The
#: plain RTN of `--no-calib` is for smoke runs and is not made into distribution assets (the
#: corresponding MUST in `anima/export.py`).
CALIB_SHIPPABLE_METHOD = "gptq"

#: Spelling of the i4 quantization-sensitivity experiment variant (in-block adaLN + outside the
#: blocks stored as i8). The series directory suffix and the `method` suffix of
#: {@link CALIB_PROVENANCE_FILE} are built from **the same single word**. The writer is
#: `--i4-adaln-i8` of `anima/export.py` (the reader owns the spelling under the same discipline as
#: the LoRA / calibration records — it does not move independently in two places).
ADALN_I8_TAG = "adaln8"

#: The rounding method the same variant records. The key is that it is spelled **so as not to match
#: {@link CALIB_SHIPPABLE_METHOD}** (the variant passes the storage form, the counts, `verify_dist`
#: and the header check alike, so the only net that keeps it out of the distribution path is the
#: method match of {@link assert_calib_provenance}). Only the assembly for visual evaluation
#: (`anima.eval_dist`) requires it.
ADALN_I8_CALIB_METHOD = f"{CALIB_SHIPPABLE_METHOD}-{ADALN_I8_TAG}"

#: Devices that can run the GPTQ calibration (capture + rounding) (the writer is `--calib-device`
#: of `anima/export.py`).
CALIB_DEVICES = ("cpu", "cuda")

#: **The calibration device allowed for distribution**. An i4 baked on any other device gets the
#: device name in the `method` of {@link CALIB_PROVENANCE_FILE} (`gptq-cuda`) and fails by name in
#: the method match of {@link assert_calib_provenance}. The spelling changes for the same reason as
#: {@link ADALN_I8_TAG} — **GPTQ's rounding solution changes with the device** (both the order of
#: the f64 reductions and the linalg implementation differ), yet the storage form and the counts do
#: not differ by a byte, so `verify_dist` and the header check pass it. The only net that keeps it
#: out of the distribution path is the method match.
CALIB_SHIPPABLE_DEVICE = "cpu"

#: The original text of the upstream weights licence (placed verbatim next to this recipe). The
#: distribution is a Distribution of a Derivative, so §3(a) applies. §3(a) has two requirements —
#: the first half "provide a copy of the licence to third parties" and the latter half "specify
#: that the rights to use the CircleStone Models / Derivatives are granted to the recipient
#: directly by the Company under this licence". A summary or a rewrite does not satisfy the first
#: half, so the text is shipped **without changing a single byte** as `LICENSE.md` at the root of
#: the distribution repository. The latter half is satisfied by putting the fixed sentence
#: {@link RIGHTS_GRANT_STATEMENT} (source of truth: `anima/card.py`) in both `NOTICE.md` and the
#: card (ruling of 2026-09-25). It is resolved from `Path(__file__)` so that it depends neither on
#: the cwd nor on where the series live.
LICENSE_SOURCE_PATH = Path(__file__).parent / "circlestone_license.txt"

#: The NVIDIA Open Model License Agreement, transcribed verbatim into plain text from
#: <https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/>
#: ("Version Release Date: October 24, 2025"; NVIDIA publishes no plain-text release). Anima is a
#: Derivative Model of `nvidia/Cosmos-Predict2-2B-Text2Image`, and agreement §3.1 requires a copy of
#: the agreement with every distributed copy (rulings 39 a / 40 a, 2026-10-29). It becomes the
#: second part of `LICENSE.md` ({@link license_markdown}). MUST: never reformat it — the sha256 in
#: the tests freezes it, and a change of that digest records a deliberate re-transcription.
NVIDIA_LICENSE_SOURCE_PATH = Path(__file__).parent / "nvidia_open_model_license.txt"

#: Heading of the second licence in `LICENSE.md` — it names the agreement and the scope it applies
#: to (the same layout as the two licences of the Irodori distribution).
NVIDIA_LICENSE_HEADING = (
    "## NVIDIA Open Model License Agreement — applies in addition to the CircleStone license above"
)

#: The modification that appears in every repository's notice (conversion into the container
#: format).
#:
#: MUST: the list of modifications is held **per repository** (one list is not reused). The turbo
#: and base repositories differ in whether anything is baked in and in the number of storage series,
#: so the moment one list is shared, one of the notices contradicts the facts — and since the value
#: is plausible prose, it passes `verify_dist` and the manifest checks and only surfaces after
#: distribution.
#: MUST: hold only one wording, correct regardless of the part count (no variants chosen by looking
#: at the manifest) — `root_files` is fixed when the Pipeline is built, so a manifest-dependent
#: branch cannot live here. The container description uses the same phrase as the other series'
#: NOTICE and the card's overview — if the "notice of the modifications" that §3(d)(i) requires
#: contradicts the actual output form, it is as if the modifications were not stated.
CONTAINER_MODIFICATION = (
    "- The weights were converted into the container format of the WebGPU inference"
    " runtime Karume\n"
    "  (a `.krm` part sequence whose first part carries the graph and model descriptors)."
)


def notice_markdown(modifications: tuple[str, ...]) -> str:
    """`NOTICE.md` at the root of the distribution repository.

    One page satisfies the latter half of §3(a) (stating that the rights are granted directly by
    the Company — {@link RIGHTS_GRANT_STATEMENT}) + §3(b) (display of the Attribution Notice) +
    §3(d)(i) (stating the modification **inside the Attribution Notice**) + §3(d)(iii) (not to be
    mistaken for an official product). The verbatim block is {@link ATTRIBUTION_NOTICE} (source of
    truth: `anima/card.py`); the rest describes facts on the Karume side. The modifications are
    placed inside the Notice section rather than in a section of their own to match the wording of
    §3(d)(i) ("include in the Attribution Notice") strictly.
    """
    return (
        "\n".join(
            [
                "# Notice",
                "",
                "## Attribution Notice",
                "",
                ATTRIBUTION_NOTICE,
                "",
                "As required by the license, this Attribution Notice also states that the",
                "applicable CircleStone Model has been modified: this distribution is a Derivative",
                "of the CircleStone Anima model, modified as follows:",
                "",
                *modifications,
                "",
                "## Not an official product",
                "",
                "This is not an official product of CircleStone Labs LLC, and it is not endorsed,",
                "approved or validated by CircleStone Labs LLC.",
                "",
                # NVIDIA Open Model License §3.1 / §3.2 (rulings 39 a / 40 a, 2026-10-29).
                "## NVIDIA Open Model License",
                "",
                NVIDIA_NOTICE,
                "",
                "## License",
                "",
                RIGHTS_GRANT_STATEMENT,
                "",
                "The full license text is distributed alongside this repository as LICENSE.md.",
            ]
        )
        + "\n"
    )


#: Notice of the official repository (karume-anima — CircleStone's 5 variants together).
#: The old base notice's "added an int4 series" contradicted the actual distribution, which had no
#: i4 preset (corrected on 2026-09-01 — the same-day ruling that removed i4 from every model removed
#: the int4 line itself; the LoRA baking line of the old turbo notice also went away with the move
#: to the official checkpoint).
#: The bf16 line (ADR 0131 decision 3 — the original-weights preset of every model) was added on
#: 2026-10-29; until then the notice named only the f16 and int8 series. bf16 keeps the upstream
#: BF16 values unchanged (no rounding, no quantization), so the line does not call it a quantized
#: series and says that the f16 series is the rounded one.
OFFICIAL_NOTICE_MARKDOWN = notice_markdown(
    (
        CONTAINER_MODIFICATION,
        "- An int8-quantized series of the transformer was added alongside the f16 one.",
        "- A bf16 series was added alongside the f16 one. It keeps the values of the upstream BF16",
        "  weights unchanged (no rounding or quantization); the f16 series rounds them to f16.",
    )
)

#: Notice of the extra repository (karume-anima-extra — third-party fine-tunes). The origins are
#: left to the README (the same reason as the NOTE above).
EXTRA_NOTICE_MARKDOWN = notice_markdown(
    (
        CONTAINER_MODIFICATION,
        "- An int8-quantized series of the transformer was added alongside the f16 one.",
        "- A bf16 series was added alongside the f16 one. It keeps the values of the upstream BF16",
        "  weights unchanged (no rounding or quantization); the f16 series rounds them to f16.",
        "- Every model in this repository is a community fine-tune of the CircleStone",
        "  Anima base model; its origin, author and license terms are stated in README.md.",
    )
)

#: Relative output paths (**inside a model subtree**) — the one place the placement table and the
#: manifest share. They are looked up by role name only, so the spelling cannot move independently
#: in two places.
#:
#: MUST: do not move the paths of the f16 text stack (`text_encoder/model.krm` and the like) — the
#: extra repository (karume-anima-extra) references these paths across repositories. The bf16 text
#: stack sits next to it as separate files named `.bf16` ({@link text_stack_role}).
OUTPUT_PATHS: Mapping[str, str] = {
    "text_encoder": "text_encoder/model.krm",
    "text_conditioner": "text_conditioner/model.krm",
    "transformer_f16": "transformer/model.f16.krm",
    "transformer_i8": "transformer/model.i8.krm",
    "transformer_i4": "transformer/model.i4.krm",
    "vae_decoder": "vae_decoder/model.krm",
    "tokenizer": "tokenizer/qwen2-tokenizer.json",
    "tokenizer_2": "tokenizer_2/t5-tokenizer.json",
    "text_encoder_bf16": "text_encoder/model.bf16.krm",
    "text_conditioner_bf16": "text_conditioner/model.bf16.krm",
    "transformer_bf16": "transformer/model.bf16.krm",
    "vae_decoder_bf16": "vae_decoder/model.bf16.krm",
}

#: Component-override token of a preset name → its weights name (ADR 0074 decision 4 — **the
#: recipe owns the abbreviations, and the quant table of the generated model card always shows the
#: mapping**). Anima's base storage is `f16` (only the original-weights preset uses `bf16` for all 4
#: roles) and only the transformer is compressed, so the preset names are `f16+dit<bits>`.
ANIMA_QUANT_ABBREVIATIONS: Mapping[str, str] = {"dit": "transformer"}

#: The quant table (`presets` in v1 — renamed in ADR 0041 §3). The manifest owns the vocabulary of
#: `session`. Preset names follow ADR 0074's grammar `<storage>[+<component><bits>]…[-<knob>]…`,
#: where `<storage>` is **the base storage shared by every role** (f16 for Anima; bf16 only for the
#: original-weights preset). **Weights the table does not name are not written** —
#: {@link anima_quants} fills them with the `<storage>` of the preset name
#: ({@link seat_base_storage} — a value that can simply be copied is not duplicated once per quant).
#:
#: `label` / `description` are display fields for the selection UI (ADR 0075 decision 1 — English,
#: 64 / 200 character limits). They are written from what the preset actually is (storage, knobs,
#: benefit): the id serves the machine, and people read these 2 fields. Being the default is not
#: mentioned (`defaultQuant` already points at it — ADR 0075 decision 3).
ANIMA_QUANTS: Mapping[str, Any] = {
    "f16": {
        "weights": {"transformer": "f16"},
        "session": {},
        "label": "Full quality (f16)",
        "description": "Transformer in f16 storage with f32 compute — the largest download,"
        " and the reference the other quants here are judged against.",
    },
    "f16+dit8": {
        "weights": {"transformer": "i8"},
        "session": {},
        "label": "Half size (int8 transformer)",
        "description": "Transformer stored as int8 and computed in f32: roughly half its f16"
        " download, with the execution path left unchanged.",
    },
    "f16+dit8-a8": {
        "weights": {"transformer": "i8"},
        "session": {"linearCompute": "a8"},
        "label": "Half size, int8 linear",
        "description": "The int8 transformer with per-token int8 activations in its linear"
        " layers — faster on GPUs with dp4a, same download.",
    },
    "f16+dit8-a8-attn8": {
        "weights": {"transformer": "i8"},
        "session": {"linearCompute": "a8", "attentionCompute": "a8"},
        "label": "Half size, int8 linear and attention",
        "description": "Adds int8 activations inside attention on top of the int8 linear path;"
        " same weights, one more integer stage per step.",
    },
    "f16+dit8-a8-attn8-s16": {
        "weights": {"transformer": "i8"},
        "session": {
            "linearCompute": "a8",
            "attentionCompute": "a8",
            "attentionScoreStorage": "f16",
        },
        "label": "Balanced (int8)",
        "description": "The int8 linear and attention path with attention scores held in f16 —"
        " the fastest of the int8 seats here, at f16-level image quality.",
    },
    # The 2 presets with resident i4 (run J-4a — the low-VRAM presets).
    #
    # MUST: **do not declare `linearCompute`**. Note that **the reason changed on 2026-08-21** —
    # it used to be "the a8 predicate requires resident i8, so declaring it makes a fake preset
    # that changes not a single byte", but the w4a8 implementation (ADR 0076) put resident i4 on the
    # integer dot-product path as well. The reason now is **quality**: declaring it makes the linear
    # activations per-token i8, and on real-GPU images the maintainer's visual verdict was "broken
    # details, rough lines" (2026-08-21, research record
    # `docs/research/2026-08-21-anima-i4-seat-speed.md` §6). Speed recovers a lot, 1,640 → 955
    # ms/step, but the reason these presets exist is **size and VRAM**, not speed (if speed is
    # needed, the default `f16+dit8-a8-attn8-s16` is faster at 823 ms/step). The 2 attention knobs
    # do not look at the weight slots, so they work as is with resident i4, and no degradation was
    # seen visually, so they are declared.
    "f16+dit4": {
        "weights": {"transformer": "i4"},
        "session": {},
        "label": "Smallest (int4 transformer)",
        "description": "Transformer weights in GPTQ-calibrated int4 (group-32) with f32 compute —"
        " the smallest download and the least resident memory.",
    },
    "f16+dit4-attn8-s16": {
        "weights": {"transformer": "i4"},
        "session": {"attentionCompute": "a8", "attentionScoreStorage": "f16"},
        "label": "Smallest, int8 attention",
        "description": "The int4 transformer with int8 activations and f16 scores in attention:"
        " the low-memory seat, without slowing down as much as plain int4.",
    },
    "f16-c16": {
        "weights": {"transformer": "f16"},
        "session": {"linearCompute": "f16", "attentionCompute": "f16"},
        "gpuFeatures": {"shaderF16": True},
        "label": "Full quality, f16 compute",
        "description": "f16 storage computed in f16 throughout. Needs the shader-f16 GPU feature,"
        " and trades numerical headroom for speed.",
    },
    # The original-weights preset (ADR 0131 decision 3 — the upstream checkpoint has all 4
    # components in BF16, so bf16 storage is upstream's bits as is). The preset appears only for
    # models that have `bf16` in {@link AnimaModel.storages} (derived by {@link anima_quants}).
    "bf16": {
        "weights": {
            "text_encoder": "bf16",
            "text_conditioner": "bf16",
            "transformer": "bf16",
            "vae_decoder": "bf16",
        },
        "session": {},
        "label": "Original weights (bf16)",
        "description": "All four components stored as bf16 with f32 compute — the upstream"
        " checkpoint's own values, bit for bit. Same size as f16.",
    },
}

ANIMA_DEFAULT_QUANT = "f16+dit8-a8-attn8-s16"

#: Pipeline-owned settings (the hub passes them through — ADR 0041 §2). The values match the
#: implementation constants of the migration source: `shift` / `numTrainTimesteps` are the
#: sampler's `ANIMA_SHIFT` / `ANIMA_NUM_TRAIN_TIMESTEPS` (equal to the exporter's `SHIFT` /
#: `NUM_TRAIN_TIMESTEPS` = scheduler_config.json), and `steps` / `guidanceScale` are the turbo
#: defaults (8 steps / cfg 1 — ADR 0038 Examples is authoritative; the visual quality check, the
#: final benchmark and the PNG reference sha were all taken at 8 steps, and the distribution default
#: matches them; the 10 of the migration source's CLI has no verification history).
#: `negativePrompt` is the default negative prompt. Only `resolution` does not take the migration
#: source CLI's default (512) — that 512 is "the minimum of the static assets", not a recommended
#: value, and the distribution is a single S form (ADR 0038 §4) that does not depend on the
#: resolution. The recommended distribution default is 1024², as in ADR 0038 Examples.
#: `type` (the sampler kind) **keeps Euler as the distribution default** (re-ruling of 2026-08-25 —
#: follow upstream Anima's recommended sampler; the maintainer's policy is "DPM++ 2M is one of the
#: options"). 0.5.0 once declared dpmpp-2m, reverted the same day. The observations of the visual
#: A/B (at the same step count dpmpp-2m is as good or better — seed 42 about even, seed 7 better in
#: composition and mid-ground detail, no added breakage) remain valid as observations; only the
#: ruling on the default sides with upstream's recommendation. DPM++ 2M is chosen with the
#: request-side `sampler` option (0.5.1). The default when omitted is also "euler" (config.ts on
#: the models side), but it is declared explicitly so that the ruling can be read.
ANIMA_SCHEDULER: Mapping[str, Any] = {"type": "euler", "shift": 3, "numTrainTimesteps": 1000}

#: The default negative prompt. **Not used** by the turbo preset (CFG=1 never computes the uncond
#: side), but the field stays — it is the value that takes effect the moment a user raises the
#: guidance. On the plain models that run with CFG, it also goes through the uncond branch of the i4
#: **calibration inputs** (`anima.calib`).
ANIMA_NEGATIVE_PROMPT = "low quality, worst quality, blurry, bad anatomy, jpeg artifacts"

#: The recommended distribution resolution (ADR 0038 Examples). The migration source CLI's 512 is
#: "the minimum of the static assets", not a recommended value, and the distribution is a single S
#: form (ADR 0038 §4) that does not depend on the resolution.
ANIMA_RESOLUTION: Mapping[str, int] = {"width": 1024, "height": 1024}

ANIMA_TURBO_PIPELINE_CONFIG: Mapping[str, Any] = {
    "scheduler": ANIMA_SCHEDULER,
    "defaults": {
        "steps": 8,
        "guidanceScale": 1,
        "resolution": ANIMA_RESOLUTION,
        "negativePrompt": ANIMA_NEGATIVE_PROMPT,
    },
}

#: Defaults of the plain base models. Unlike turbo they **use CFG** (= the negative prompt takes
#: effect), so one step is 2 forwards, and the step count goes back to the usual range before
#: distillation.
#:
#: The step count is **20**, the visual verdict after rendering 20–32 at 3 prompts × 4 levels
#: (2026-08-22). The policy for the default is "the fastest within the range without quality
#: problems", and the difference between 20 and 32 was small regardless of subject (changing the
#: step count changes the whole sigma sequence, so **it gives a different image rather than a more
#: detailed version of the same one** — judge by "from where breakage decreases and it
#: stabilises"). A user who wants more passes `steps`.
ANIMA_BASE_PIPELINE_CONFIG: Mapping[str, Any] = {
    "scheduler": ANIMA_SCHEDULER,
    "defaults": {
        "steps": 20,
        "guidanceScale": 4,
        "resolution": ANIMA_RESOLUTION,
        "negativePrompt": ANIMA_NEGATIVE_PROMPT,
    },
}

#: Defaults of the official Aesthetic variant. Uses CFG, like base.
#:
#: The step count is **30**, the visual verdict after rendering 20 / 30 / 50 at seeds 42-45 × 4
#: subjects (person, landscape, perspective, full body + street) (2026-09-01, confirmed for both
#: v1.0 and v1.1). The middle ground between "50 seems better" and "but not a dramatic improvement
#: over 20" — also the low end of the upstream README's general recommendation (30-50 steps / CFG
#: 4-5). A user who wants more passes `steps`.
ANIMA_AESTHETIC_PIPELINE_CONFIG: Mapping[str, Any] = {
    "scheduler": ANIMA_SCHEDULER,
    "defaults": {
        "steps": 30,
        "guidanceScale": 4,
        "resolution": ANIMA_RESOLUTION,
        "negativePrompt": ANIMA_NEGATIVE_PROMPT,
    },
}

#: The storage vocabulary **required** in each role's binding table (a presence check). An actual
#: incident is the reason: plain f32 assets, meant as an f16 series but built without `--dtype`,
#: passed assembly, loading and execution and **were not caught until the PNG reference
#: comparison**. The binding table, not the series directory name, is authoritative for the
#: storage form. In an f16 series only the fake-quant targets become f16 (norm / bias and the like
#: stay f32), so "contains f16" is required. The tokenizers (JSON) are not listed here.
#: An i4 series is **mixed** (f32 + i8 + i4 together), so **i4 is required** — {@link
#: assert_storage} checks "the required vocabulary is in the binding table", so requiring i8 would
#: let an i8 series pass into the i4 slot and the mix-up would stay silent (the same discipline as
#: sbv2's i4 preset).
STORAGE_REQUIREMENTS: Mapping[str, str] = {
    "text_encoder": "f16",
    "text_conditioner": "f16",
    "transformer_f16": "f16",
    "transformer_i8": "i8",
    "transformer_i4": "i4",
    "vae_decoder": "f16",
    "text_encoder_bf16": "bf16",
    "text_conditioner_bf16": "bf16",
    "transformer_bf16": "bf16",
    "vae_decoder_bf16": "bf16",
}

#: The storage vocabulary that **must not appear** in each role's binding table ({@link
#: assert_storage_absent}). A presence check alone lets **a mix-up between compressed slots** pass —
#: an i4 series is mixed and its default storage is i8 (`BASE_WEIGHT_DTYPES` in `anima/export.py`),
#: so it **always contains i8**. Plugging an i4 series into `transformer_i8` by mistake therefore
#: satisfies "contains i8" and passes assembly, verify_dist and loading. The real damage shows up
#: in the default quant `f16+dit8-a8-attn8-s16`: since `c285f97` the predicate of the declared
#: `linearCompute: "a8"` accepts resident i4 too (ADR 0076), so with resident i4 it does not fail
#: loudly and runs on **the numeric contract of w4a8** (partial group reductions) — the
#: configuration ADR 0076 decision 6 decided "not to put in a preset because the images degrade"
#: ships silently in the default preset while the preset name still says int8.
#: MUST: hold the prohibitions **as a set per role** (with only one, the day a 4th series appears
#: the one not named passes silently — the same discipline as irodori). The f16 slot is closed
#: twice, by the absence of i8 and of i4.
#: MUST: mix-ups between the bf16 series (ADR 0131) and the other series fail in both directions
#: too — a bf16 series is bf16 + f32 and an f16 series f16 + f32, so neither satisfies the other's
#: requirement, but relying on the requirement alone goes silent the day "a mix containing both"
#: appears. The bf16 roles forbid f16 / i8 / i4, and the other roles forbid bf16.
ANIMA_STORAGE_FORBIDDEN: Mapping[str, tuple[str, ...]] = {
    "text_encoder": ("bf16",),
    "text_conditioner": ("bf16",),
    "vae_decoder": ("bf16",),
    "transformer_f16": ("bf16", "i8", "i4"),
    "transformer_i8": ("bf16", "i4"),
    "transformer_i4": ("bf16",),
    "text_encoder_bf16": ("f16", "i8", "i4"),
    "text_conditioner_bf16": ("f16", "i8", "i4"),
    "transformer_bf16": ("f16", "i8", "i4"),
    "vae_decoder_bf16": ("f16", "i8", "i4"),
}

#: Names of the components that carry a graph = the keys of the manifest's weights = **the
#: container's graph names** (container-v1 §2.1 — the runtime looks them up by name with
#: `prepareContainer(opened, <weights key>)`).
#:
#: MUST: the writer (`TARGET_*` in `anima/export.py`) reads them from here. For Anima the series
#: directory names have the same spelling, but that is **a convention, not a derivation** — the
#: writer writes into a staging directory (`<component>.staging/`), so deriving from the directory
#: would give every series the staging directory's name as its graph name (the MUST of
#: `karume.pipeline._assert_graph_name`).
ANIMA_TEXT_ENCODER_ROLE = "text_encoder"
ANIMA_TEXT_CONDITIONER_ROLE = "text_conditioner"
ANIMA_TRANSFORMER_ROLE = "transformer"
ANIMA_VAE_DECODER_ROLE = "vae_decoder"

#: Declaration of the weights (dtype label → role name). The labels are **storage dtype
#: vocabulary**, 1:1 with the storage forms {@link STORAGE_REQUIREMENTS} requires (ADR 0041 §3).
#: `i4` is a label for **a mixed series**: "i4-eligible weights in i4 group32, the rest i8"
#: (`anima/export.py`).
ANIMA_WEIGHTS: Mapping[str, Mapping[str, WeightFiles]] = {
    ANIMA_TEXT_ENCODER_ROLE: {
        "f16": WeightFiles("text_encoder"),
        "bf16": WeightFiles("text_encoder_bf16"),
    },
    ANIMA_TEXT_CONDITIONER_ROLE: {
        "f16": WeightFiles("text_conditioner"),
        "bf16": WeightFiles("text_conditioner_bf16"),
    },
    ANIMA_TRANSFORMER_ROLE: {
        "f16": WeightFiles("transformer_f16"),
        "i8": WeightFiles("transformer_i8"),
        "i4": WeightFiles("transformer_i4"),
        "bf16": WeightFiles("transformer_bf16"),
    },
    ANIMA_VAE_DECODER_ROLE: {
        "f16": WeightFiles("vae_decoder"),
        "bf16": WeightFiles("vae_decoder_bf16"),
    },
}

#: Declaration of the assets (unconditional files that do not depend on the quant choice — ADR
#: 0041 §3).
ANIMA_ASSETS: Mapping[str, str] = {"tokenizer": "tokenizer", "tokenizer_2": "tokenizer_2"}


@dataclass(frozen=True)
class AnimaModel:
    """The Anima-specific facts of one model (how its series are picked, its presets, its defaults).

    MUST: **hold the LoRA as a sha256, not a bool**. "Not baked" and "baked but not recorded" cannot
    be told apart from the assets, so the side without an expected value (`None`) actively checks
    **that there is no record** ({@link assert_lora_provenance}). With a check in one direction
    only, plugging a turbo series into the slot of a plain model would pass.
    """

    #: sha256 of the baked-in LoRA. `None` = plain (nothing baked in).
    lora_sha256: str | None
    #: The transformer's storage dtype labels. Series of a storage form not listed here are
    #: **neither required nor declared** (2026-08-23, run J-4 ②: the calibration conditions are now
    #: derived from {@link pipeline_config} (`anima.calib.calib_conditions`), so the i4 preset is no
    #: longer turbo-only). The text stack's storage labels (keys of
    #: {@link ANIMA_TEXT_STACK_SERIES}) are read from here too — a model with `bf16` also requires
    #: and declares the bf16 series of the 3 text-stack roles (the original-weights preset is bf16
    #: for all 4 roles).
    storages: tuple[str, ...]
    #: Whether the model has its own text_conditioner (for f16 — bf16 is decided by
    #: {@link text_conditioner_series}).
    #: `False` = the plain shared series ({@link ANIMA_BASE_SERIES}).
    #: Third-party fine-tunes retrain not only the DiT but also the llm_adapter (=
    #: text_conditioner), so sharing would run with **another model's text conditioning** (only the
    #: images silently drift).
    own_text_conditioner: bool
    #: The manifest's `pipelineConfig` (steps / guidance differ per model). **The i4 calibration
    #: conditions are derived from it too** (`anima.calib.calib_conditions`) — so that the sigma
    #: sequence and the CFG branches the calibration sees are exactly the distribution's runtime
    #: conditions, the export side keeps no copy. Moving this also moves the rounding targets of the
    #: next i4 export.
    pipeline_config: Mapping[str, Any]
    #: The rounding method required of an i4 series (`method` of `calib_provenance.json`). The
    #: distribution has only one ({@link CALIB_SHIPPABLE_METHOD}), and **only the visual-evaluation
    #: slot changes it** (`anima.eval_dist` — a spec for assembling and viewing the
    #: quantization-sensitivity experiment variants locally). The default stays the constant, so the
    #: 4 models on the distribution path never spell out this field.
    calib_method: str = CALIB_SHIPPABLE_METHOD


#: Model name → facts. Also where the repositories split ({@link OFFICIAL_MODELS} / {@link
#: EXTRA_MODELS}).
#:
#: NOTE: **no model has "i4"** (both the presets and the files disappear through the derivation of
#: {@link anima_quants} / {@link anima_weights}) — the visual verdict on i4 (2026-08-24) found
#: large composition divergence and skipped distribution, and the re-upload for 0.5.0 (2026-08-25)
#: ruled it out again. The i4 preset only the old fused turbo had was not carried over to the
#: official checkpoints either (ADR 0087; ruling of 2026-09-01). Condition for bringing it back =
#: identifying the quantization-sensitive parts around adaLN (the old series are kept in
#: outputs/series/ as *-i4-dyn — the calibrated backup series-archive was deleted by the clean-up
#: ruling of 2026-08-30).
ANIMA_MODELS: Mapping[str, AnimaModel] = {
    # Official Turbo (the distilled checkpoint itself — the old anima-turbo's LoRA baking is no
    # longer needed). lora_sha256=None is the active "not baked" check and rejects plugging in an
    # old fused series that carries lora_provenance.json. The text_conditioner measured
    # bit-identical to base after f16 rounding (2026-09-01), so f16 uses the shared series (in bf16
    # the upstream bits differ in 60 / 118 tensors — measured 2026-10-17, only near-zero elements up
    # to 2.98e-8 — so it has its own series: {@link text_conditioner_series}).
    # The i4 preset the old fused turbo had is **not carried over** (maintainer ruling of
    # 2026-09-01 — every model is without i4; the lever to bring it back is `anima.eval_dist`, as
    # for the plain model).
    ANIMA_TURBO_MODEL_NAME: AnimaModel(
        lora_sha256=None,
        storages=("f16", "i8", "bf16"),
        own_text_conditioner=False,
        pipeline_config=ANIMA_TURBO_PIPELINE_CONFIG,
    ),
    # Every model has the original-weights preset `bf16` (ADR 0131 decision 3, addendum
    # 2026-10-17).
    ANIMA_BASE_MODEL_NAME: AnimaModel(
        lora_sha256=None,
        storages=("f16", "i8", "bf16"),
        own_text_conditioner=False,
        pipeline_config=ANIMA_BASE_PIPELINE_CONFIG,
    ),
    # Official Aesthetic. No i4 preset, on the same ruling as the plain model (the NOTE above — no
    # model has i4).
    ANIMA_AESTHETIC_MODEL_NAME: AnimaModel(
        lora_sha256=None,
        storages=("f16", "i8", "bf16"),
        own_text_conditioner=True,
        pipeline_config=ANIMA_AESTHETIC_PIPELINE_CONFIG,
    ),
    # The 2 official variants of the v1.0 generation are distributed alongside (maintainer ruling
    # of 2026-09-01 — preferences between versions differ for this family, so a way back to the old
    # version is kept = exactly the motivation of ADR 0077). Both have their own conditioner,
    # measured to differ from base in 64/118 tensors after f16 rounding (2026-09-01).
    "anima-turbo-v1.0": AnimaModel(
        lora_sha256=None,
        storages=("f16", "i8", "bf16"),
        own_text_conditioner=True,
        pipeline_config=ANIMA_TURBO_PIPELINE_CONFIG,
    ),
    "anima-aesthetic-v1.0": AnimaModel(
        lora_sha256=None,
        storages=("f16", "i8", "bf16"),
        own_text_conditioner=True,
        pipeline_config=ANIMA_AESTHETIC_PIPELINE_CONFIG,
    ),
    "anima-wai-v1.0": AnimaModel(
        lora_sha256=None,
        storages=("f16", "i8", "bf16"),
        own_text_conditioner=True,
        pipeline_config=ANIMA_BASE_PIPELINE_CONFIG,
    ),
    "anima-copycat-20260610": AnimaModel(
        lora_sha256=None,
        storages=("f16", "i8", "bf16"),
        own_text_conditioner=True,
        pipeline_config=ANIMA_BASE_PIPELINE_CONFIG,
    ),
}

#: The accepted set per repository. **A different Pipeline means a different legal text (NOTICE) at
#: the root**, so assembling with the wrong one silently yields "a repository whose modification
#: notice contradicts its contents" — it fails at the planning stage. The split is **official /
#: extra** (ruling of 2026-09-01 — ADR 0087): the official repository (karume-anima) holds
#: CircleStone's 5 variants with Turbo as the default, and the extra repository
#: (karume-anima-extra) holds third-party fine-tunes and references the text stack in the official
#: repository across repositories.
OFFICIAL_MODELS: tuple[str, ...] = (
    ANIMA_TURBO_MODEL_NAME,
    ANIMA_BASE_MODEL_NAME,
    ANIMA_AESTHETIC_MODEL_NAME,
    "anima-turbo-v1.0",
    "anima-aesthetic-v1.0",
)
EXTRA_MODELS: tuple[str, ...] = ("anima-wai-v1.0", "anima-copycat-20260610")


@dataclass(frozen=True)
class AnimaSources:
    """The series directories that are the inputs of the assembly.

    The text path and the VAE do not depend on the DiT's storage dtype, so one series is shared per
    text-stack storage label (ADR 0019 — f16 for every model, bf16 only for the models with the
    original-weights preset). The transformer splits into f16 / i8 / i4 / bf16 series.
    """

    #: Storage dtype label → transformer series (in the order of {@link AnimaModel.storages}).
    transformer: Mapping[str, Path]
    #: Text-stack storage label → the series holding text_encoder / VAE
    #: ({@link ANIMA_TEXT_STACK_SERIES}).
    base: Mapping[str, Path]
    #: Likewise, the series holding text_conditioner (for a plain model the same as {@link base},
    #: with the same keys).
    text_conditioner: Mapping[str, Path]
    tokenizers: Path


def text_conditioner_series(model: str, spec: AnimaModel, storage: str) -> str:
    """The text_conditioner's series name (per storage label).

    For f16 it is decided by {@link AnimaModel.own_text_conditioner} (the shared series if
    bit-identical after f16 rounding). bf16 **keeps the upstream bits**, so every model but base has
    its own series — turbo-v1.1's llm_adapter is identical to base after f16 rounding, but in bf16
    60 / 118 tensors differ (measured 2026-10-17, max 2.98e-8, only near-zero elements). The
    distribution assembly folds only byte-identical components into `shared/`, so the sharing
    optimisation is not lost even though it is not decided here.
    """
    if storage == "bf16":
        if model == ANIMA_BASE_MODEL_NAME:
            return ANIMA_TEXT_STACK_SERIES[storage]
        return f"{model}-bf16"
    return f"{model}-{storage}" if spec.own_text_conditioner else ANIMA_TEXT_STACK_SERIES[storage]


def anima_sources(series_dir: Path, model: str = ANIMA_BASE_MODEL_NAME) -> AnimaSources:
    """Picks one model's series from the series parent directory (`outputs/series/`).

    The model name applies only to the transformer series and to the series of models with their own
    text_conditioner ({@link text_conditioner_series} — in bf16 every model but base has its own).
    text_encoder / VAE / tokenizers are parts upstream fine-tunes do not touch, so for every model
    they come from the one shared series per storage label.
    """
    spec = anima_model(model)
    text_storages = [storage for storage in spec.storages if storage in ANIMA_TEXT_STACK_SERIES]
    return AnimaSources(
        transformer={storage: series_dir / f"{model}-{storage}-dyn" for storage in spec.storages},
        base={storage: series_dir / ANIMA_TEXT_STACK_SERIES[storage] for storage in text_storages},
        text_conditioner={
            storage: series_dir / text_conditioner_series(model, spec, storage)
            for storage in text_storages
        },
        tokenizers=series_dir / ANIMA_TOKENIZER_SERIES / "text",
    )


def anima_model(model: str) -> AnimaModel:
    """Looks up the facts by model name (an unknown name fails with the choices listed)."""
    spec = ANIMA_MODELS.get(model)
    if spec is None:
        choices = ", ".join(ANIMA_MODELS)
        raise DistError(f"Unknown Anima model name: {model!r} (choices: {choices})")
    return spec


def transformer_series(sources: AnimaSources) -> tuple[Path, ...]:
    """The transformer series per storage dtype (**cross-series comparisons MUST read this list**).

    The byte-identity check of the raw rope tables and the LoRA attribution comparison are both
    checks that "walk every series", and keeping the enumeration in 2 places means that the day a
    storage slot is added only one gets updated — a series that slipped through the net passes the
    check, and either flaw is silent at runtime (a geometry mismatch breaks only the images, an
    attribution mismatch makes only the README false).
    """
    return tuple(sources.transformer.values())


def assert_shared_rope_base(sources: AnimaSources) -> None:
    """Checks that the raw rope tables of every transformer series are byte-identical.

    MUST: the raw table goes into each of the f16 / i8 / i4 series as the container's **asset**
    `rope_base` (ADR 0109 decision 4). Byte identity across all of them is checked with sha256 —
    shipping them mismatched would make one of the quants "run with rope tables of a different
    geometry", passing loading and execution while only the images silently break. In `karume/4`
    they were folded into one file, so this was also the choice of "which one to ship"; now that
    each container holds its own, **only the equality check** remains.
    """
    containers = [series / "transformer" / "model.krm" for series in transformer_series(sources)]
    for path in containers:
        assert_component_present(path)
    digests = {path: hashlib.sha256(_rope_base_of(path)).hexdigest() for path in containers}
    if len(set(digests.values())) != 1:
        listing = "\n".join(f"  {digest}  {path}" for path, digest in digests.items())
        raise DistError(
            f"Asset '{ROPE_BASE_ASSET}' is not byte-identical across the series — they cannot"
            " ship with the same geometry. Which one is right cannot be decided here, so the"
            f" assembly stops:\n{listing}"
        )


def _rope_base_of(path: Path) -> bytes:
    """Reads the `rope_base` asset (translating the reader's exception into **assembly terms**).

    MUST: the diagnostic is a `DistError`. Only `--dit-graph dyn` series carry `rope_base`, so the
    operational mistake of pointing the transformer slot at a static series ends up here — raising
    the reader's exception as is would make an assembly failure look like "a bug in container
    reading" (the same approach as `assert_storage` translating `ContainerFormatError`).
    """
    try:
        return read_asset(path, ROPE_BASE_ASSET)
    except ContainerReadError as cause:
        raise DistError(f"{path}: {cause}") from cause


def assert_lora_provenance(sources: AnimaSources, expected: str | None) -> None:
    """Checks that the LoRA sha256 recorded by the transformer and text_conditioner series
    matches the model's declaration.

    MUST: the model's declaration (`lora_sha256` of {@link AnimaModel}) is the only record of
    "which LoRA this distribution has baked in", yet the baked LoRA cannot be recovered from the
    fused weights. Without the comparison, re-exporting with a replaced LoRA would still assemble
    with the old / wrong sha256 — the value is well-formed 64-digit hex, so it passes the structural
    checks of `verify_dist` and **stays silent**. "The same fact held by separate scripts is always
    compared at assembly time" (the same discipline as the byte-identity check of rope_base).

    MUST: with `expected is None` (a plain model) **check that there is no record**. Fused and
    plain weights do not differ by a single byte in the form of the assets, so plugging a turbo
    series into a plain model's slot is caught by no other check. The writer (`anima/export.py`)
    deletes the record of every target it did not bake a LoRA into (a total function), so **the
    absence of the record is the evidence of "not baked"**.

    MUST: the writer bakes into and records for the 2 targets of `LORA_PREFIXES` (transformer /
    text_conditioner), so the records of the text_conditioner series (the shared {@link
    ANIMA_BASE_SERIES} and a model's own `<model>-f16`) are checked against the same expected value.
    If the net covered only the transformer, plugging a conditioner series with a baked LoRA into a
    distribution slot would pass, and nobody would check the "no LoRA" MUST of `anima-f16`.
    """
    records = [
        series / "transformer" / LORA_PROVENANCE_FILE for series in transformer_series(sources)
    ]
    records += [
        series / "text_conditioner" / LORA_PROVENANCE_FILE
        for series in sources.text_conditioner.values()
    ]
    for path in records:
        if expected is None:
            if path.is_file():
                raise DistError(
                    "The slot of a model without a baked LoRA received a series that carries a"
                    f" baking record: {path}"
                    " (re-exporting without `--lora` removes the record)"
                )
            continue
        if not path.is_file():
            raise DistError(
                f"No record of the baked-in LoRA: {path}"
                " (re-exporting with `python -m anima.export --lora …` writes it)"
            )
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except ValueError as cause:
            raise DistError(f"Cannot parse the LoRA record: {path} — {cause}") from cause
        digest = record.get("sha256") if isinstance(record, dict) else None
        if digest != expected:
            raise DistError(
                f"The baked-in LoRA differs from the card's declaration: {path} has {digest!r},"
                f" anima/card.py has {expected!r} — which one is right cannot be decided here"
            )


def assert_calib_provenance(sources: AnimaSources, spec: AnimaModel) -> None:
    """Checks from the writer's record that the i4 series was rounded **under this model's
    conditions and with a calibration allowed for distribution** (GPTQ × a budget at or above the
    floor × the model's steps / CFG).

    MUST: neither the calibration method nor its budget nor its conditions change a single byte of
    the storage form (`research/2026-08-21-anima-i4-seat-speed.md` §6 — the file size is identical
    to the RTN version byte for byte). The structural checks of `verify_dist` and the header dtype
    check therefore **pass it**. `--no-calib` is a smoke-run opt-out, yet if its output slipped
    into a distribution it could not be told from the assets, and all that shows is "blurry overall"
    images — the same discipline as the LoRA attribution ({@link assert_lora_provenance}), "the
    same fact held by separate scripts is always compared at assembly time", applies here too.

    MUST: **take `spec` and compare down to the conditions** (the required rounding method is also
    read from {@link AnimaModel.calib_method} — the distribution has one, and only the
    visual-evaluation slot changes it). `--model` of `anima.export` is a knob only for looking up
    the calibration conditions (it does not affect the stored bytes the way LoRA baking does), so
    assets built from the plain weights with `--model anima-turbo-v1.1` look like "a correct plain
    i4" — only the calibration ran with turbo's 8 steps, CFG 1. The conditions are derived from the
    model's `pipeline_config` (`anima.calib.calib_conditions`), so re-reading them from the same
    single place makes the comparison possible.

    The decision itself lives in `_shared.calib_provenance` (the same single implementation as
    irodori's 2 readers).
    """
    from .calib_prompts import DEFAULT_CALIB_PROMPTS

    path = sources.transformer["i4"] / "transformer" / CALIB_PROVENANCE_FILE
    if not path.is_file():
        raise DistError(
            f"The i4 series has no record of its calibration conditions: {path}"
            " (re-exporting with `python -m anima.export --dtype i4` writes it)"
        )
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as cause:
        raise DistError(
            f"Cannot parse the record of the calibration conditions: {path} — {cause}"
        ) from cause
    defaults = spec.pipeline_config["defaults"]
    complaint = calib_complaint(
        record,
        method=spec.calib_method,
        at_least={"prompts": DEFAULT_CALIB_PROMPTS},
        exactly={"steps": int(defaults["steps"]), "guidance": float(defaults["guidanceScale"])},
    )
    if complaint is not None:
        raise DistError(
            f"The calibration conditions of the i4 series do not meet the requirement: {path}:"
            f" {complaint}"
            " — output baked with `--no-calib` / a smoke budget / another model's conditions / a"
            " per-role experiment variant is not used for distribution"
        )


def anima_placements(sources: AnimaSources) -> dict[str, Path]:
    """Role name → source file. The output paths are held by {@link OUTPUT_PATHS}.

    Anything not in this table does not go into the output (this is all it takes to drop
    `io.*.safetensors`).
    """
    placements: dict[str, Path] = {}
    for storage, series in sources.base.items():
        placements[text_stack_role(ANIMA_TEXT_ENCODER_ROLE, storage)] = (
            series / "text_encoder" / "model.krm"
        )
        placements[text_stack_role(ANIMA_TEXT_CONDITIONER_ROLE, storage)] = (
            sources.text_conditioner[storage] / "text_conditioner" / "model.krm"
        )
        placements[text_stack_role(ANIMA_VAE_DECODER_ROLE, storage)] = (
            series / "vae_decoder" / "model.krm"
        )
    placements["tokenizer"] = sources.tokenizers / "qwen2-tokenizer.json"
    placements["tokenizer_2"] = sources.tokenizers / "t5-tokenizer.json"
    for storage, series in sources.transformer.items():
        placements[f"transformer_{storage}"] = series / "transformer" / "model.krm"
    return placements


def anima_weights(spec: AnimaModel) -> Mapping[str, Mapping[str, WeightFiles]]:
    """The weights declaration narrowed to the model's storage forms (no labels without a preset).

    One rule narrows every role — the text stack's `f16` is in every model's storages and stays,
    and `bf16` stays only for the models with the original-weights preset.
    """
    return {
        role: {label: files for label, files in labels.items() if label in spec.storages}
        for role, labels in ANIMA_WEIGHTS.items()
    }


def seat_base_storage(name: str) -> str:
    """The `<storage>` part of a preset name (before the first `+` / `-`).

    In ADR 0074's grammar this is the base storage shared by every role.
    """
    return re.split(r"[+-]", name, maxsplit=1)[0]


def anima_quants(spec: AnimaModel) -> Mapping[str, Any]:
    """Narrows to the quant presets the model's storage forms support; fills the unnamed roles.

    The roles filled are the ones the table does not name.

    MUST: which presets to keep is **derived from the declared storage forms** (no separate
    hard-coded list of preset names). Holding it in 2 places means that the day a storage form is
    added the presets do not grow / the day one is removed the presets remain.

    MUST: a role that is not named is filled with **the `<storage>` of the preset name** ({@link
    seat_base_storage}) — for a model whose text stack has the 2 labels f16 / bf16, core's {@link
    complete_quant_weights} fails with "several exist but none is specified" (it does not pick a
    default on its own), so the basis for the choice is the grammar of the preset name. If the
    filled label does not exist for the role, core fails with "no such dtype".
    """
    weights = anima_weights(spec)
    return {
        name: {
            **entry,
            "weights": {
                role: entry["weights"].get(role, seat_base_storage(name)) for role in weights
            },
        }
        for name, entry in ANIMA_QUANTS.items()
        if entry["weights"]["transformer"] in spec.storages
    }


def anima_plan(
    sources: AnimaSources,
    model: str = ANIMA_BASE_MODEL_NAME,
    *,
    spec: AnimaModel | None = None,
) -> ModelPlan:
    """Builds the plan for one Anima model (all checks are done here — not a byte is written).

    `spec` is **the slot for an explicit override** (when omitted it is looked up by model name =
    the distribution path). Only the visual-evaluation assembly (`anima.eval_dist`) overrides it,
    passing a spec that "puts the i4 preset back on the plain model and spells the required rounding
    method as the experiment variant" — an entry point that keeps the distribution's accepted set
    ({@link ANIMA_MODELS}) from moving for experiments, never used the other way round (the
    distribution path spelling out a spec).
    """
    assert_model_name(model)
    spec = anima_model(model) if spec is None else spec
    assert_lora_provenance(sources, spec.lora_sha256)
    if "i4" in spec.storages:
        assert_calib_provenance(sources, spec)
    assert_shared_rope_base(sources)
    placements = anima_placements(sources)
    for role, source in placements.items():
        assert_storage(role, source, STORAGE_REQUIREMENTS)
        assert_storage_absent(role, source, ANIMA_STORAGE_FORBIDDEN)
    weights = anima_weights(spec)
    return ModelPlan(
        name=model,
        pipeline=ANIMA_PIPELINE,
        artifacts={
            role: Artifact(OUTPUT_PATHS[role], source=source) for role, source in placements.items()
        },
        weights=weights,
        assets=ANIMA_ASSETS,
        quants=complete_quant_weights(weights, anima_quants(spec)),
        default_quant=ANIMA_DEFAULT_QUANT,
        pipeline_config=spec.pipeline_config,
    )


def anima_dist_plan(series_dir: Path, model: str, allowed: tuple[str, ...]) -> ModelPlan:
    """Builds the plan for one Anima model from the `--series` parent (the CLI dispatches here).

    MUST: `allowed` is **the models allowed in that Pipeline's repository**. A different Pipeline
    means a different NOTICE (modification notice) at the repository root, so assembling with the
    wrong one silently yields "a repository whose notice contradicts its contents" — the manifest
    and verify_dist stay structurally correct, so nobody would notice until after distribution.
    """
    if model not in allowed:
        choices = ", ".join(allowed)
        raise DistError(
            f"Model {model!r} does not belong in this pipeline's repository (allowed: {choices})"
            " — the modification notice at the repository root would contradict its contents, so"
            " the assembly stops"
        )
    return anima_plan(anima_sources(series_dir, model), model)


def license_markdown() -> str:
    """`LICENSE.md` — the CircleStone license, then the NVIDIA Open Model License Agreement.

    `root_files` has a single `LICENSE.md` slot (`karume.dist.LEGAL_PATHS`), so the second text
    follows a separator (`---`) and a heading that states its scope, as in the Irodori
    distribution. The CircleStone text comes first, byte for byte ({@link LICENSE_SOURCE_PATH});
    it has no trailing newline, so the separator starts with a blank line (without it, Markdown
    would read the last paragraph and `---` as a setext heading).

    MUST: neither text is touched — reformatting either makes it no longer a copy of that
    license. The NVIDIA text is numbered plain text, so it sits verbatim inside a fence (it
    contains no ``` — the digest freeze in the tests catches any change to the file).
    """
    return (
        LICENSE_SOURCE_PATH.read_text(encoding="utf-8")
        + "\n\n---\n\n"
        + NVIDIA_LICENSE_HEADING
        + "\n\n"
        + "The CircleStone Anima model is a Derivative Model of NVIDIA"
        + " Cosmos-Predict2-2B-Text2Image, so every\n"
        + "model in this repository is covered by the agreement below in addition to the"
        + " CircleStone license\nabove (see `NOTICE.md`). Both licenses apply; the agreement"
        + " below does not widen the CircleStone\nlicense.\n\n"
        + "```text\n"
        + NVIDIA_LICENSE_SOURCE_PATH.read_text(encoding="utf-8")
        + "```\n"
    )


def root_files(notice: str) -> dict[str, str]:
    """The legal texts at the distribution repository's root (`karume.dist.Pipeline.root_files`).

    The original licence text is read **verbatim** from the file placed in the recipe ({@link
    LICENSE_SOURCE_PATH}) — reformatting or replacing it here would make it no longer "a copy of
    this licence" in the sense of §3(a). Only the modification notice (`NOTICE.md`) differs per
    repository, so only it is taken as an argument.
    `LICENSE.md` carries the NVIDIA Open Model License Agreement after the CircleStone text
    ({@link license_markdown}).
    """
    return {
        "LICENSE.md": license_markdown(),
        "NOTICE.md": notice,
    }


#: Distribution name of the official repository (the `karume-` prefix comes from the repository
#: naming ruling of 2026-08-09 — a namespace in place of an HF org).
OFFICIAL_REPO_NAME = "karume-anima"

#: Distribution name of the extra repository.
EXTRA_REPO_NAME = "karume-anima-extra"


def _official_repo_name(_model: str) -> str:
    """The official repository name (the model name is ignored).

    ADR 0087 folded the 5 variants into one repository.

    `Pipeline.repo_name` is the source of truth for the repository name in the card's Usage example
    (`karume.dist.resolve_repo`), and it also answers the default output location when a single
    model is assembled. If this returned `karume-<model name>`, the output location without `--out`
    and the card's Usage example would name a repository that does not exist.
    """
    return OFFICIAL_REPO_NAME


def _extra_repo_name(_model: str) -> str:
    """The extra repository name (one repository with several models, like the official one)."""
    return EXTRA_REPO_NAME


#: Roles every fine-tune of the extra repository borrows from the official repository instead of
#: storing them (ADR 0087; ADR 0038 §7 addendum 2026-10-28): the text encoder and the VAE decoder
#: in every text-stack storage, and both tokenizers — the official text stack, byte-identical. Not
#: `text_conditioner`: every extra model has its own (`own_text_conditioner=True`).
ANIMA_EXTRA_SHARED_ROLES: tuple[str, ...] = (
    *(
        text_stack_role(component, storage)
        for storage in ANIMA_TEXT_STACK_SERIES
        for component in (ANIMA_TEXT_ENCODER_ROLE, ANIMA_VAE_DECODER_ROLE)
    ),
    *ANIMA_ASSETS.values(),
)


def _extra_references(_model: str) -> tuple[CrossRepoReference, ...]:
    """`Pipeline.references` of the extra repository: every model borrows the official text
    stack. The referenced model is the base (`anima-v1.0`) — the text stack is built from the base
    series, and the official repository folds it into `shared/` for all five variants."""
    return (
        CrossRepoReference(
            repo_name=OFFICIAL_REPO_NAME,
            model=ANIMA_BASE_MODEL_NAME,
            roles=ANIMA_EXTRA_SHARED_ROLES,
        ),
    )


#: The entry for `--pipeline anima` (the official repository karume-anima — CircleStone's 5
#: variants together, default = Turbo (upstream README's recommendation; ruling of 2026-09-01)).
#:
#: MUST: a **separate Pipeline** from extra. `root_files` is one fixed set per Pipeline, so folding
#: them into one would make the modification notice false in one of the repositories (failing the
#: requirement of §3(d)(i)).
OFFICIAL_PIPELINE = Pipeline(
    default_model=ANIMA_TURBO_MODEL_NAME,
    repo_name=_official_repo_name,
    plan=lambda series_dir, model: anima_dist_plan(series_dir, model, OFFICIAL_MODELS),
    card_profiles={"anima": partial(render_base_card, abbreviations=ANIMA_QUANT_ABBREVIATIONS)},
    # The upstream licence's redistribution terms (§3) apply per distribution repository, so the
    # texts are read **once, here**, however many times the assembly runs.
    root_files=root_files(OFFICIAL_NOTICE_MARKDOWN),
)

#: The entry for `--pipeline anima-extra` (the extra repository karume-anima-extra — third-party
#: fine-tunes).
#: The text stack is referenced in the official repository ({@link _extra_references}; the
#: publication order is docs/release-runbook.md §0).
EXTRA_PIPELINE = Pipeline(
    default_model="anima-wai-v1.0",
    repo_name=_extra_repo_name,
    plan=lambda series_dir, model: anima_dist_plan(series_dir, model, EXTRA_MODELS),
    card_profiles={
        "anima-extra": partial(render_extra_card, abbreviations=ANIMA_QUANT_ABBREVIATIONS)
    },
    root_files=root_files(EXTRA_NOTICE_MARKDOWN),
    references=_extra_references,
)
