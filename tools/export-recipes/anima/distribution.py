"""Anima's distribution recipe — series layout, output paths, quants, card (ADR 0065 decision 2).

The generic assembly engine (placement, folding of shared slots, sha256, manifest, staging/swap,
verification) lives in `karume.dist`. This module holds only **Anima-specific facts**: what to pick
from which series directory, at which path of the distribution and under which dtype label to lay
it out, and which quant is the default.

The public surface is three Pipelines ({@link OFFICIAL_PIPELINE} / {@link EXTRA_PIPELINE} — ADR
0087 — and {@link ANIMA_2_9B_PIPELINE}, its addendum of 2026-11-24) — the repository's dist driver
(`tools/export-recipes/dist.py`) merges them into core's PIPELINES.
"""

from __future__ import annotations

import atexit
import hashlib
import json
import re
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

from _shared.calib_provenance import calib_complaint
from _shared.container_read import ContainerReadError, read_asset
from _shared.paths import DIST_ROOT
from anima.card import (
    ATTRIBUTION_NOTICE,
    NVIDIA_NOTICE,
    RIGHTS_GRANT_STATEMENT,
    render_2_9b_card,
    render_base_card,
    render_extra_card,
)
from karume.container import ContainerFormatError
from karume.dist import (
    MANIFEST_FILENAME,
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
    component_parts,
    sha256_file,
)
from karume.verify import ContainerError, verify_container

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
#: high-quality style — upstream's naming as is). It carries its own text_conditioner (its
#: llm_adapter differs from base — {@link text_conditioner_series}).
ANIMA_AESTHETIC_MODEL_NAME = "anima-aesthetic-v1.1"

#: Model name of Anima-2.9B (HF `Gazingstars123/Anima-2.9B`; the upstream's version name
#: `preview-v1` is kept — ADR 0077): a third party's layer expansion of the Anima base model, its
#: transformer deepened from 28 to 40 blocks of the same width and trained further. It is a
#: modified model rather than a fine-tune, so it has a repository of its own
#: ({@link ANIMA_2_9B_REPO_NAME} — ADR 0087 addendum of 2026-11-24). Its text encoder, VAE decoder
#: and tokenizers are the base model's bytes; its text conditioner and transformer are its own.
ANIMA_2_9B_MODEL_NAME = "anima-2.9b-preview-v1"

#: Pipeline contract (ADR 0041 §2 — per model).
ANIMA_PIPELINE = "anima/1"

#: Series that do not depend on the model name (base's text path / VAE, and the output of the
#: script that writes the tokenizers).
#: MUST: `anima-bf16` is a series baked **without baking in a LoRA** (the absence of
#: `lora_provenance.json` is the record of that — {@link assert_lora_provenance}).
ANIMA_BASE_SERIES = "anima-bf16"
ANIMA_TOKENIZER_SERIES = "anima-demo"

#: Storage label → shared series of the text stack (text_encoder / text_conditioner / vae_decoder —
#: the 3 roles that do not depend on the DiT's storage). The text stack is stored in bf16 only —
#: every upstream component is BF16, so bf16 storage keeps the upstream values (ADR 0154; the f16
#: series left the distribution).
ANIMA_TEXT_STACK_SERIES: Mapping[str, str] = {"bf16": ANIMA_BASE_SERIES}


def text_stack_role(component: str, storage: str) -> str:
    """The role name of a text-stack placement (a key of {@link OUTPUT_PATHS}):
    `<component>_<storage>`, the same form as the transformer's `transformer_<storage>`.

    The extra repository's cross-repository references are derived through this function
    ({@link ANIMA_EXTRA_SHARED_ROLES}).
    """
    return f"{component}_{storage}"


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
#: 2026-10-29; until then the notice named only the f16 and int8 series. Since ADR 0154 bf16 is the
#: only float series (the f16 one left the distribution), so no line names an f16 series any more:
#: bf16 keeps the upstream BF16 values unchanged (no rounding, no quantization), and the int8
#: series is the only one whose values differ from upstream.
OFFICIAL_NOTICE_MARKDOWN = notice_markdown(
    (
        CONTAINER_MODIFICATION,
        "- An int8-quantized series of the transformer was added alongside the bf16 one.",
        "- The bf16 series of every component keeps the values of the upstream BF16 weights",
        "  unchanged (no rounding or quantization).",
    )
)

#: Notice of the extra repository (karume-anima-extra — third-party fine-tunes). The origins are
#: left to the README (the same reason as the NOTE above).
EXTRA_NOTICE_MARKDOWN = notice_markdown(
    (
        CONTAINER_MODIFICATION,
        "- An int8-quantized series of the transformer was added alongside the bf16 one.",
        "- The bf16 series of every component keeps the values of the upstream BF16 weights",
        "  unchanged (no rounding or quantization).",
        "- Every model in this repository is a community fine-tune of the CircleStone",
        "  Anima base model; its origin, author and license terms are stated in README.md.",
    )
)

#: Notice of the 2.9B repository (karume-anima-2.9b — Anima-2.9B). The upstream file is BF16
#: throughout (928 tensors, header read 2026-11-24), so the bf16 line holds as in the other two.
#: The model is a third party's modification of the CircleStone model, not Karume's, so the line
#: that says so names it as such and leaves the origin to the README (the same reason as extra's).
ANIMA_2_9B_NOTICE_MARKDOWN = notice_markdown(
    (
        CONTAINER_MODIFICATION,
        "- An int8-quantized series of the transformer was added alongside the bf16 one.",
        "- The bf16 series of every component keeps the values of the upstream BF16 weights",
        "  unchanged (no rounding or quantization).",
        "- The model in this repository is a third party's layer expansion of the CircleStone",
        "  Anima base model: its transformer was deepened from 28 to 40 blocks and trained",
        "  further. Its origin, author and license terms are stated in README.md.",
    )
)

#: Relative output paths (**inside a model subtree**) — the one place the placement table and the
#: manifest share. They are looked up by role name only, so the spelling cannot move independently
#: in two places.
#:
#: MUST: do not move the paths of the bf16 text stack (`text_encoder/model.bf16.krm` and the like)
#: — they are the published paths, and the extra repository (karume-anima-extra) references them
#: across repositories ({@link text_stack_role}).
OUTPUT_PATHS: Mapping[str, str] = {
    "transformer_i8": "transformer/model.i8.krm",
    "transformer_i4": "transformer/model.i4.krm",
    "tokenizer": "tokenizer/qwen2-tokenizer.json",
    "tokenizer_2": "tokenizer_2/t5-tokenizer.json",
    "text_encoder_bf16": "text_encoder/model.bf16.krm",
    "text_conditioner_bf16": "text_conditioner/model.bf16.krm",
    "transformer_bf16": "transformer/model.bf16.krm",
    "vae_decoder_bf16": "vae_decoder/model.bf16.krm",
}

#: Component-override token of a preset name → its weights name (ADR 0074 decision 4 — **the
#: recipe owns the abbreviations, and the quant table of the generated model card always shows the
#: mapping**). Anima's base storage is `bf16` (every upstream component is BF16 — ADR 0154) and
#: only the transformer is compressed, so the preset names are `bf16+dit<bits>`.
ANIMA_QUANT_ABBREVIATIONS: Mapping[str, str] = {"dit": "transformer"}

#: The quant table (`presets` in v1 — renamed in ADR 0041 §3). The manifest owns the vocabulary of
#: `session`. Preset names follow ADR 0074's grammar `<storage>[+<component><bits>]…[-<knob>]…`,
#: where `<storage>` is **the base storage shared by every role** (bf16 for Anima — ADR 0154).
#: **Weights the table does not name are not written** — {@link anima_quants} fills them with the
#: `<storage>` of the preset name ({@link seat_base_storage} — a value that can simply be copied is
#: not duplicated once per quant).
#:
#: `label` / `description` are display fields for the selection UI (ADR 0075 decision 1 — English,
#: 64 / 200 character limits). They state what the preset does — storage, quantization, compute —
#: and no speed or image-quality claim, none being measured on the bf16 text stack yet: the id
#: serves the machine, and people read these 2 fields. Being the default is not mentioned
#: (`defaultQuant` already points at it — ADR 0075 decision 3).
ANIMA_QUANTS: Mapping[str, Any] = {
    # The original-weights preset (ADR 0131 decision 3 — the upstream checkpoint has all 4
    # components in BF16, so bf16 storage is upstream's bits as is). The preset appears only for
    # models that have `bf16` in {@link AnimaModel.storages} (derived by {@link anima_quants}).
    "bf16": {
        "weights": {"transformer": "bf16"},
        "session": {},
        "label": "Original weights (bf16)",
        "description": "All four components stored as bf16 with f32 compute — the upstream"
        " checkpoint's own values, bit for bit, and the largest download.",
    },
    "bf16+dit8": {
        "weights": {"transformer": "i8"},
        "session": {},
        "label": "int8 transformer",
        "description": "Transformer stored as int8 (half its bf16 size) and computed in f32; the"
        " text encoder, text conditioner and VAE decoder stay bf16.",
    },
    "bf16+dit8-a8": {
        "weights": {"transformer": "i8"},
        "session": {"linearCompute": "a8"},
        "label": "int8 transformer, int8 linear",
        "description": "The int8 transformer with per-token int8 activations in its linear"
        " layers, multiplied as integers (dp4a where the GPU has it); the same files as bf16+dit8.",
    },
    "bf16+dit8-a8-attn8": {
        "weights": {"transformer": "i8"},
        "session": {"linearCompute": "a8", "attentionCompute": "a8"},
        "label": "int8 transformer, int8 linear and attention",
        "description": "As bf16+dit8-a8, with int8 activations inside attention too; the same files"
        " as bf16+dit8.",
    },
    "bf16+dit8-a8-attn8-s16": {
        "weights": {"transformer": "i8"},
        "session": {
            "linearCompute": "a8",
            "attentionCompute": "a8",
            "attentionScoreStorage": "f16",
        },
        "label": "int8 transformer, int8 linear and attention, f16 scores",
        "description": "As bf16+dit8-a8-attn8, with the attention scores held in f16; the same"
        " files as bf16+dit8.",
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
    # needed, the default `bf16+dit8-a8-attn8-s16` is faster at 823 ms/step). The 2 attention knobs
    # do not look at the weight slots, so they work as is with resident i4, and no degradation was
    # seen visually, so they are declared.
    "bf16+dit4": {
        "weights": {"transformer": "i4"},
        "session": {},
        "label": "int4 transformer",
        "description": "Transformer stored as GPTQ-calibrated int4 (group-32) and computed in f32;"
        " the other components stay bf16 — the smallest download and the least resident memory.",
    },
    "bf16+dit4-attn8-s16": {
        "weights": {"transformer": "i4"},
        "session": {"attentionCompute": "a8", "attentionScoreStorage": "f16"},
        "label": "int4 transformer, int8 attention, f16 scores",
        "description": "The int4 transformer with int8 activations inside attention and the"
        " attention scores held in f16; the same files as bf16+dit4.",
    },
    # The original weights computed in f16 (the `-c16` knob reaches the transformer only — the
    # session knobs apply to the DiT). The weights are the `bf16` preset's own; only the compute
    # path differs.
    "bf16-c16": {
        "weights": {"transformer": "bf16"},
        "session": {"linearCompute": "f16", "attentionCompute": "f16"},
        "gpuFeatures": {"shaderF16": True},
        "label": "Original weights, f16 transformer compute",
        "description": "The bf16 preset's files, with the transformer's linear layers and attention"
        " computed in f16; the text encoder, text conditioner and VAE decoder compute in f32. Needs"
        " the shader-f16 GPU feature.",
    },
}

ANIMA_DEFAULT_QUANT = "bf16+dit8-a8-attn8-s16"

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
#: storage form. In a bf16 series only the eligible weights become bf16 (norm / bias and the like
#: stay f32), so "contains bf16" is required. The tokenizers (JSON) are not listed here.
#: An i4 series is **mixed** (f32 + i8 + i4 together), so **i4 is required** — {@link
#: assert_storage} checks "the required vocabulary is in the binding table", so requiring i8 would
#: let an i8 series pass into the i4 slot and the mix-up would stay silent (the same discipline as
#: sbv2's i4 preset).
STORAGE_REQUIREMENTS: Mapping[str, str] = {
    "transformer_i8": "i8",
    "transformer_i4": "i4",
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
#: in the default quant `bf16+dit8-a8-attn8-s16`: since `c285f97` the predicate of the declared
#: `linearCompute: "a8"` accepts resident i4 too (ADR 0076), so with resident i4 it does not fail
#: loudly and runs on **the numeric contract of w4a8** (partial group reductions) — the
#: configuration ADR 0076 decision 6 decided "not to put in a preset because the images degrade"
#: ships silently in the default preset while the preset name still says int8.
#: MUST: hold the prohibitions **as a set per role** (with only one, the day a 4th series appears
#: the one not named passes silently — the same discipline as irodori).
#: MUST: every role forbids f16. No slot takes an f16 series any more (ADR 0154), but the exporter
#: still writes them (`--dtype f16`) and old ones stay on disk — an f16 series is f16 + f32, so it
#: already fails every requirement, yet relying on the requirement alone goes silent the day "a mix
#: containing both" appears. The bf16 roles forbid f16 / i8 / i4, and the compressed roles forbid
#: f16 / bf16 (and i8 forbids i4 — above).
ANIMA_STORAGE_FORBIDDEN: Mapping[str, tuple[str, ...]] = {
    "transformer_i8": ("f16", "bf16", "i4"),
    "transformer_i4": ("f16", "bf16"),
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
    ANIMA_TEXT_ENCODER_ROLE: {"bf16": WeightFiles("text_encoder_bf16")},
    ANIMA_TEXT_CONDITIONER_ROLE: {"bf16": WeightFiles("text_conditioner_bf16")},
    ANIMA_TRANSFORMER_ROLE: {
        "i8": WeightFiles("transformer_i8"),
        "i4": WeightFiles("transformer_i4"),
        "bf16": WeightFiles("transformer_bf16"),
    },
    ANIMA_VAE_DECODER_ROLE: {"bf16": WeightFiles("vae_decoder_bf16")},
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
    #: and declares the bf16 series of the 3 text-stack roles (every preset stores the text stack
    #: in bf16).
    storages: tuple[str, ...]
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
    # old fused series that carries lora_provenance.json. Its text_conditioner differs from base
    # in the upstream bits (60 / 118 tensors — measured 2026-10-17, only near-zero elements up to
    # 2.98e-8), so it has its own series ({@link text_conditioner_series}).
    # The i4 preset the old fused turbo had is **not carried over** (maintainer ruling of
    # 2026-09-01 — every model is without i4; the lever to bring it back is `anima.eval_dist`, as
    # for the plain model).
    ANIMA_TURBO_MODEL_NAME: AnimaModel(
        lora_sha256=None,
        storages=("i8", "bf16"),
        pipeline_config=ANIMA_TURBO_PIPELINE_CONFIG,
    ),
    # Every model stores all 4 components in bf16, the upstream checkpoints' own dtype (ADR 0131
    # decision 3, addendum 2026-10-17; ADR 0154), and the transformer in i8 besides.
    ANIMA_BASE_MODEL_NAME: AnimaModel(
        lora_sha256=None,
        storages=("i8", "bf16"),
        pipeline_config=ANIMA_BASE_PIPELINE_CONFIG,
    ),
    # Official Aesthetic. No i4 preset, on the same ruling as the plain model (the NOTE above — no
    # model has i4).
    ANIMA_AESTHETIC_MODEL_NAME: AnimaModel(
        lora_sha256=None,
        storages=("i8", "bf16"),
        pipeline_config=ANIMA_AESTHETIC_PIPELINE_CONFIG,
    ),
    # The 2 official variants of the v1.0 generation are distributed alongside (maintainer ruling
    # of 2026-09-01 — preferences between versions differ for this family, so a way back to the old
    # version is kept = exactly the motivation of ADR 0077).
    "anima-turbo-v1.0": AnimaModel(
        lora_sha256=None,
        storages=("i8", "bf16"),
        pipeline_config=ANIMA_TURBO_PIPELINE_CONFIG,
    ),
    "anima-aesthetic-v1.0": AnimaModel(
        lora_sha256=None,
        storages=("i8", "bf16"),
        pipeline_config=ANIMA_AESTHETIC_PIPELINE_CONFIG,
    ),
    "anima-wai-v1.0": AnimaModel(
        lora_sha256=None,
        storages=("i8", "bf16"),
        pipeline_config=ANIMA_BASE_PIPELINE_CONFIG,
    ),
    "anima-copycat-20260610": AnimaModel(
        lora_sha256=None,
        storages=("i8", "bf16"),
        pipeline_config=ANIMA_BASE_PIPELINE_CONFIG,
    ),
    # The upstream card recommends 28-50 steps at CFG 3.5-5; the defaults stay the base model's (20
    # steps, CFG 4, Euler shift 3), the only configuration its images were made and checked with
    # here — whether to adopt the upstream's range is open (docs/backlog.md).
    ANIMA_2_9B_MODEL_NAME: AnimaModel(
        lora_sha256=None,
        storages=("i8", "bf16"),
        pipeline_config=ANIMA_BASE_PIPELINE_CONFIG,
    ),
}

#: The accepted set per repository. **A different Pipeline means a different legal text (NOTICE) at
#: the root**, so assembling with the wrong one silently yields "a repository whose modification
#: notice contradicts its contents" — it fails at the planning stage. The split is **official /
#: extra** (ruling of 2026-09-01 — ADR 0087): the official repository (karume-anima) holds
#: CircleStone's 5 variants with Turbo as the default, and the extra repository
#: (karume-anima-extra) holds third-party fine-tunes and references the text stack in the official
#: repository across repositories. A third repository (karume-anima-2.9b) holds Anima-2.9B, a
#: layer-expanded model rather than a fine-tune (ruling of 2026-11-24 — ADR 0087 addendum); it
#: references the official text stack the same way.
OFFICIAL_MODELS: tuple[str, ...] = (
    ANIMA_TURBO_MODEL_NAME,
    ANIMA_BASE_MODEL_NAME,
    ANIMA_AESTHETIC_MODEL_NAME,
    "anima-turbo-v1.0",
    "anima-aesthetic-v1.0",
)
EXTRA_MODELS: tuple[str, ...] = ("anima-wai-v1.0", "anima-copycat-20260610")
ANIMA_2_9B_MODELS: tuple[str, ...] = (ANIMA_2_9B_MODEL_NAME,)


@dataclass(frozen=True)
class AnimaSources:
    """The series directories that are the inputs of the assembly.

    The text path and the VAE do not depend on the DiT's storage dtype, so one series is shared per
    text-stack storage label (ADR 0019 — bf16, {@link ANIMA_TEXT_STACK_SERIES}). The transformer
    splits into bf16 / i8 / i4 series.
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


def text_conditioner_series(model: str, storage: str) -> str:
    """The text_conditioner's series name (per storage label).

    bf16 **keeps the upstream bits**, so every model but base has its own series — turbo-v1.1's
    llm_adapter differs from base in 60 / 118 tensors (measured 2026-10-17, max 2.98e-8, only
    near-zero elements), and third-party fine-tunes retrain it (sharing would run with **another
    model's text conditioning**, and only the images would silently drift). The distribution
    assembly folds only byte-identical components into `shared/`, so the sharing optimisation is
    not lost even though it is not decided here.
    """
    if model == ANIMA_BASE_MODEL_NAME:
        return ANIMA_TEXT_STACK_SERIES[storage]
    return f"{model}-{storage}"


def anima_sources(series_dir: Path, model: str = ANIMA_BASE_MODEL_NAME) -> AnimaSources:
    """Picks one model's series from the series parent directory (`outputs/series/`).

    The model name applies only to the transformer series and to the text_conditioner series
    ({@link text_conditioner_series} — every model but base has its own). text_encoder / VAE /
    tokenizers are parts upstream fine-tunes do not touch, so for every model they come from the
    one shared series per storage label.
    """
    spec = anima_model(model)
    text_storages = [storage for storage in spec.storages if storage in ANIMA_TEXT_STACK_SERIES]
    return AnimaSources(
        transformer={storage: series_dir / f"{model}-{storage}-dyn" for storage in spec.storages},
        base={storage: series_dir / ANIMA_TEXT_STACK_SERIES[storage] for storage in text_storages},
        text_conditioner={
            storage: series_dir / text_conditioner_series(model, storage)
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

    The LoRA attribution comparison walks every series, and the byte-identity check of the raw
    rope tables walks the transformer containers placed from them ({@link anima_placements} reads
    the same mapping) — keeping a second enumeration means that the day a storage slot is added
    only one gets updated, a series that slipped through the net passes the check, and either flaw
    is silent at runtime (a geometry mismatch breaks only the images, an attribution mismatch makes
    only the README false).
    """
    return tuple(sources.transformer.values())


def transformer_containers(placements: Mapping[str, Path]) -> tuple[Path, ...]:
    """The transformer containers among a plan's placements (`transformer_<storage>` roles), in
    the order they are placed — whether they come from the series or from the published mirrors."""
    prefix = f"{ANIMA_TRANSFORMER_ROLE}_"
    return tuple(path for role, path in placements.items() if role.startswith(prefix))


def assert_shared_rope_base(containers: Sequence[Path]) -> None:
    """Checks that the raw rope tables of every transformer container are byte-identical.

    MUST: the raw table goes into each of the bf16 / i8 / i4 series as the container's **asset**
    `rope_base` (ADR 0109 decision 4). Byte identity across all of them is checked with sha256 —
    shipping them mismatched would make one of the quants "run with rope tables of a different
    geometry", passing loading and execution while only the images silently break. In `karume/4`
    they were folded into one file, so this was also the choice of "which one to ship"; now that
    each container holds its own, **only the equality check** remains.
    """
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
    ANIMA_BASE_SERIES} and a model's own `<model>-bf16`) are checked against the same expected
    value. If the net covered only the transformer, plugging a conditioner series with a baked LoRA
    into a distribution slot would pass, and nobody would check the "no LoRA" MUST of
    {@link ANIMA_BASE_SERIES}.
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

    One rule narrows every role — the text stack's `bf16` is in every model's storages and stays,
    and the transformer keeps only the storages the model declares.
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
    seat_base_storage}) — for a role with several labels, core's {@link complete_quant_weights}
    fails with "several exist but none is specified" (it does not pick a default on its own), so
    the basis for the choice is the grammar of the preset name, not the count of labels a role
    happens to have. If the filled label does not exist for the role, core fails with "no such
    dtype".
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


#: Holder model → (role name → sha256 of **part 0** of the published container the role is taken as;
#: for a tokenizer, of the file). A model listed here is built from the published containers in the
#: local mirrors, not from its series ({@link anima_published_sources} — ADR 0154 decision 6): its
#: series are no longer on disk, and an export would not reproduce the published bytes anyway (the
#: i8 rounding may differ on another machine). Part 0's descriptor declares the sha256 of every
#: other part (container-v1 §7), so this one digest pins the whole container. What is compared
#: when is {@link assert_anima_published}'s: part 0, the tokenizers and every part of a borrowed
#: container before a byte is written; the other parts of a stored container as the assembly
#: places them, in its staging directory.
#:
#: The text encoder, the VAE decoder and the tokenizers are held by the base model for every model
#: ({@link anima_published_holder}); the rest by the model itself.
#:
#: A model without an entry is not published yet: it is built from its series, except the roles
#: another model holds, which it takes as that model published them ({@link
#: anima_unpublished_sources} — today Anima-2.9B, whose text encoder, VAE decoder and tokenizers are
#: the base's). Its own entry is added once its containers are published, and its later builds take
#: them from the mirror like every other model's.
#:
#: Recorded 2026-11-18 from the published manifests of `karume-anima` at
#: `a5fac03dc6563657b4e5c797c78cc3c813912a3e` and `karume-anima-extra` at
#: `15731a612517df0064db69e8cb7c9825fce00c91`; the local mirrors' manifests were compared with them
#: the same day (every entry equal), and every part 0 and tokenizer file with its declared sha256.
#: Re-exporting a model's containers is a deliberate change: remove its entry here.
ANIMA_PUBLISHED_PART0: Mapping[str, Mapping[str, str]] = {
    ANIMA_BASE_MODEL_NAME: {
        "text_encoder_bf16": "44aca04aacb815e0a0f0ae2b8b51d0007ae3413df4945c044bdc9cbc8275c7c6",
        "text_conditioner_bf16": "2aaffce6c4ea2728dedffffd62a0ed190ccaefb68916b73e6432f6b5527e08d9",
        "transformer_bf16": "e92187b16939639ffd826a1dabe2edf6eb47ffaf49274acc2b1248bbbf93ff80",
        "transformer_i8": "7a46f4caa9e8e63579b54ba5ee2dcb03f29d545853943dd98da19f7e0dc8a724",
        "vae_decoder_bf16": "201baa4ad52a988942b2abc34a7f9fb55c4af3427b44a9b94e5f6c4ba5fc294e",
        "tokenizer": "0a7d6057ac8a2fe459551b89682b46f4e0b207a9081d12346683c9254138af2f",
        "tokenizer_2": "f86dfe21b12a175a03144162f6d9b6a64746d19e73cd8d543b7dfae1db3c1825",
    },
    ANIMA_TURBO_MODEL_NAME: {
        "text_conditioner_bf16": "4d1d281b97ac0d8ed0f0c18cf7639f4e812c357a6b421a3b7de189f61391c376",
        "transformer_bf16": "c778fbf7f7220b84635f1bf0dc32cacdf0f5f9d3a2073c303f9d8c0608ae5327",
        "transformer_i8": "2c7669008b538be906d5abfacda27201dabca9e175fc6e9d0bdbe0132937600e",
    },
    ANIMA_AESTHETIC_MODEL_NAME: {
        "text_conditioner_bf16": "36af2bd9f1111adcefcfd9bff001aed776a96960a7419d93f86f2a1fff9ee0ad",
        "transformer_bf16": "0d0fc2e226640815964a2742347787639a7a7ad3d3f62ba1e4e89cc1468e4da0",
        "transformer_i8": "052a07a5c31a0b43e870af9589289eff290e3c7533f0b9f3d55c6449906fdd9b",
    },
    "anima-turbo-v1.0": {
        "text_conditioner_bf16": "36af2bd9f1111adcefcfd9bff001aed776a96960a7419d93f86f2a1fff9ee0ad",
        "transformer_bf16": "27d169b8a5c94c083b400bf6b147ea25888a688a14284a592ab0f367b268720d",
        "transformer_i8": "56f7f29ef497e5a8d9776a0b7bf4bb92703684f4454cfc4abef5c8afdae5981f",
    },
    "anima-aesthetic-v1.0": {
        "text_conditioner_bf16": "36af2bd9f1111adcefcfd9bff001aed776a96960a7419d93f86f2a1fff9ee0ad",
        "transformer_bf16": "fa43dfa95ba893533c6483be6ab7ad766d9b7f4d8b5d92ebc19704fe1714d5f8",
        "transformer_i8": "c8aa7b87a70da51b08ab00c5d2af417bdcdcc7c159fcb72a2e170e8407112ed2",
    },
    "anima-wai-v1.0": {
        "text_conditioner_bf16": "fc772692d9caf35faf44e8317d3877c7a3e434a9f4f98850dc51dc964e18b87a",
        "transformer_bf16": "bda5fcfded41bc5b678ab1f1cfe7651d7e3173020303cd27bcac2d21a40a5f95",
        "transformer_i8": "ab22af4bb9f10548d7bc54a2473f3095f56527dedc71247595427c8ef39965f1",
    },
    "anima-copycat-20260610": {
        "text_conditioner_bf16": "f8e9ac410b92502d6d62e7623f3d20ebc0eabddd539a8ba268e5fc5b512f039c",
        "transformer_bf16": "eefac1ff92f0b1a6da8673be517a3dd7302b4b8d6fcae2adeb5e88e3f2e462a7",
        "transformer_i8": "4deb4f2b3abc412d5e8ed43f67a63be3c11d23fec1c4d311a2a9ebc91103a3e2",
    },
}


@dataclass(frozen=True)
class AnimaPublished:
    """One model's assembly inputs taken as **published** from the local mirrors
    ({@link ANIMA_PUBLISHED_PART0}), in place of its series ({@link AnimaSources})."""

    #: Role name → the representative path of the container in the part view (its parts side by
    #: side as links to the mirror's files), or the tokenizer file in the mirror.
    placements: Mapping[str, Path]
    #: Role name → the published sha256 of part 0 (of the file, for a tokenizer).
    part0_sha256: Mapping[str, str]
    #: Role name → the repository whose mirror holds it.
    holders: Mapping[str, str]
    #: The repository the model is built into ({@link anima_repository}).
    repository: str

    @property
    def borrowed(self) -> tuple[str, ...]:
        """The roles held by another repository: referenced across repositories, never placed in
        this one."""
        return tuple(role for role, repo in self.holders.items() if repo != self.repository)


def anima_repository(model: str) -> str:
    """The distribution repository a model is published in."""
    if model in OFFICIAL_MODELS:
        return OFFICIAL_REPO_NAME
    if model in ANIMA_2_9B_MODELS:
        return ANIMA_2_9B_REPO_NAME
    return EXTRA_REPO_NAME


def anima_published_holder(model: str, role: str) -> tuple[str, str]:
    """(repository, model) whose published entry a role of `model` is taken from.

    The text encoder, the VAE decoder and the tokenizers are the base model's for every model
    ({@link ANIMA_EXTRA_SHARED_ROLES} — the parts fine-tunes and the layer expansion do not touch;
    the official repository folds them into `shared/`, and the extra and 2.9B repositories
    reference them there instead of storing them). Every other role is the model's own, in its own
    repository.
    """
    if role in ANIMA_EXTRA_SHARED_ROLES:
        return OFFICIAL_REPO_NAME, ANIMA_BASE_MODEL_NAME
    return anima_repository(model), model


def _published_refs(manifest: Mapping[str, Any], repo: str, holder: str, role: str) -> list[Any]:
    """The file references the holder's manifest entry declares for a role (every part, in order;
    one file for a tokenizer)."""
    entry = manifest.get("models", {}).get(holder)
    if entry is None:
        raise DistError(f"The mirror of {repo} has no model {holder!r} — it cannot provide {role}")
    asset = next((name for name, files in ANIMA_ASSETS.items() if files == role), None)
    if asset is not None:
        refs = [entry["assets"][asset]] if asset in entry.get("assets", {}) else []
    else:
        component, label = next(
            (component, label)
            for component, labels in ANIMA_WEIGHTS.items()
            for label, files in labels.items()
            if files.file == role
        )
        declared = entry.get("weights", {}).get(component, {}).get(label, {})
        refs = list(declared.get("container", {}).get("parts", []))
    if not refs:
        raise DistError(f"The mirror of {repo} declares no {role} for {holder}")
    if any("repo" in ref for ref in refs):
        raise DistError(
            f"{role} of {holder} is a cross-repository reference in the mirror of {repo} — it is"
            " taken from the repository that holds the files"
        )
    return refs


def _published_roles(model: str) -> list[str]:
    """Every role a model places: its weights files in every label it declares, then the
    tokenizers."""
    return [
        *(
            files.file
            for labels in anima_weights(anima_model(model)).values()
            for files in labels.values()
        ),
        *ANIMA_ASSETS.values(),
    ]


def _mirror_roles(model: str) -> list[str]:
    """The roles a build of `model` takes as published from the local mirrors: every role of a
    published model ({@link ANIMA_PUBLISHED_PART0}); for a model not published yet, only the roles
    another model holds ({@link anima_published_holder}) — its own come from its series."""
    roles = _published_roles(model)
    if model in ANIMA_PUBLISHED_PART0:
        return roles
    return [role for role in roles if anima_published_holder(model, role)[1] != model]


def _part_view_root(model: str) -> Path:
    """A new directory for one build's part view (links only), removed when the process exits —
    the plan only points at it, and the assembly that reads it runs later in the same process."""
    root = Path(tempfile.mkdtemp(prefix=f"anima-parts-{model}-"))
    atexit.register(partial(shutil.rmtree, root, ignore_errors=True))
    return root


def anima_published_sources(
    model: str, dist_root: Path, view_root: Path | None = None
) -> AnimaPublished:
    """The published containers `model` is built from, read from the local mirrors under
    `dist_root` (`<dist_root>/<repository>/karume.json` — `models/` by default).

    A mirror folds the parts two or more models share into `shared/`, so one container's parts can
    sit in two directories, while the assembly reads a container as the parts beside its
    representative path (container-v1 §8). The parts are therefore linked side by side into a part
    view (`<view_root>/<repository>/<holder>/<output path>` — links only; the mirror is only read),
    and placement copies the bytes they point at. A tokenizer is one file and is read in place.
    Without `view_root` the view is a temporary directory removed when the process exits.

    It reads the manifests and links, nothing more: whether the bytes are the published ones is
    checked by {@link assert_anima_published}. The roles are {@link _mirror_roles}': every role of
    a published model, only the roles another model holds for a model not published yet (its own
    come from its series — {@link anima_unpublished_sources}). A storage the mirrors do not hold
    (an i4 transformer) still comes from series ({@link anima_sources}).
    """
    view = _part_view_root(model) if view_root is None else view_root
    roles = _mirror_roles(model)
    manifests: dict[str, Mapping[str, Any]] = {}
    placements: dict[str, Path] = {}
    pins: dict[str, str] = {}
    holders: dict[str, str] = {}
    for role in roles:
        repo, holder = anima_published_holder(model, role)
        pin = ANIMA_PUBLISHED_PART0.get(holder, {}).get(role)
        if pin is None:
            raise DistError(
                f"{model} takes {role} as published by {holder}, but there is no published sha256"
                " for it (ANIMA_PUBLISHED_PART0)"
            )
        if repo not in manifests:
            path = dist_root / repo / MANIFEST_FILENAME
            if not path.is_file():
                raise DistError(
                    f"No local mirror of {repo} to take the published containers from: {path}"
                )
            manifests[repo] = json.loads(path.read_text(encoding="utf-8"))
        refs = _published_refs(manifests[repo], repo, holder, role)
        files = [dist_root / repo / ref["path"] for ref in refs]
        missing = [str(file) for file in files if not file.is_file()]
        if missing:
            raise DistError(
                f"{role}: the mirror of {repo} lacks files its manifest declares: {missing}"
            )
        pins[role] = pin
        holders[role] = repo
        if role in ANIMA_ASSETS.values():
            placements[role] = files[0]
            continue
        representative = view / repo / holder / OUTPUT_PATHS[role]
        representative.parent.mkdir(parents=True, exist_ok=True)
        for file in files:
            link = representative.parent / file.name
            if not link.is_symlink():
                link.symlink_to(file.absolute())
        placements[role] = representative
    return AnimaPublished(
        placements=placements,
        part0_sha256=pins,
        holders=holders,
        repository=anima_repository(model),
    )


def assert_anima_published(published: AnimaPublished) -> None:
    """Check, before a byte is written, that every container taken as published starts with the
    pinned part 0 and every tokenizer is the pinned file ({@link ANIMA_PUBLISHED_PART0}), and that
    every other part of a **borrowed** container ({@link AnimaPublished.borrowed}) has the sha256
    part 0's descriptor declares for it.

    The other parts of a container this repository stores are not hashed here: the assembly
    hashes each copy it places in the staging directory and compares it with part 0's declaration
    (`karume.dist.assemble_family`), so a mismatch discards the staging before anything is put in
    place. MUST: a borrowed container is hashed here, every part of it, because nothing later
    compares it with the descriptor — it is only referenced, the assembly does not compare
    cross-repository parts, and the reference check compares the mirror's file with itself (the
    same reason as Irodori's `assert_irodori_published_containers`). The placed copies of part 0
    and of the tokenizers are not compared with the pins again (a shared one is compared with its
    source). Restoring the mirror from the Hub is the fix for a mismatch, never re-pinning.
    """
    for role, path in published.placements.items():
        parts = component_parts(path)
        actual = sha256_file(parts[0])
        if actual != published.part0_sha256[role]:
            raise DistError(
                f"{role}: {parts[0]} is not the published file of {published.holders[role]}"
                f" (sha256 {actual}, published {published.part0_sha256[role]}) — restore the"
                " mirror from the Hub instead of rebuilding it from the series"
            )
        if role not in published.borrowed or len(parts) == 1:
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
                    f"{role}: {part} is not the part {index} the published part 0 of"
                    f" {published.holders[role]} declares ({declared.get(index)}) — restore the"
                    " mirror from the Hub"
                )


@dataclass(frozen=True)
class AnimaUnpublished:
    """The assembly inputs of a model not published yet ({@link anima_unpublished_sources}): the
    roles it stores itself from its series, and the roles another model holds as that model
    published them."""

    #: The model's series ({@link anima_sources}). Only the roles {@link published} does not hold
    #: are placed from it; the series' own records (the LoRA attribution, the i4 calibration
    #: conditions) are checked as for a build from series.
    series: AnimaSources
    #: The roles another model holds ({@link _mirror_roles} — the base's text encoder, VAE decoder
    #: and tokenizers), taken from the local mirrors and pinned like a published model's
    #: ({@link assert_anima_published}).
    published: AnimaPublished


def anima_unpublished_sources(
    series_dir: Path, model: str, dist_root: Path, view_root: Path | None = None
) -> AnimaUnpublished:
    """The inputs of a model {@link ANIMA_PUBLISHED_PART0} does not list: its own roles from the
    series under `series_dir` (`outputs/series/`), the roles another model holds from the local
    mirrors under `dist_root` (`models/`).

    The base's text-stack series is no longer on disk (`outputs/series/anima-bf16` — ADR 0154,
    Context), and an extra or 2.9B repository only references those roles: taking
    them as published is what the reference is reconciled against anyway, and the pin proves they
    are the published bytes. A model that holds every role itself (the base, re-exported on purpose)
    reads no mirror.
    """
    return AnimaUnpublished(
        series=anima_sources(series_dir, model),
        published=anima_published_sources(model, dist_root, view_root),
    )


def anima_plan(
    sources: AnimaSources | AnimaPublished | AnimaUnpublished,
    model: str = ANIMA_BASE_MODEL_NAME,
    *,
    spec: AnimaModel | None = None,
) -> ModelPlan:
    """Builds the plan for one Anima model (all checks are done here — not a byte is written).

    `sources` is the model's series ({@link AnimaSources}), its published containers
    ({@link AnimaPublished}), or both for a model not published yet ({@link AnimaUnpublished} —
    its own roles from the series, the roles another model holds as published, each placed where
    the series would have put it). The series' own records (the LoRA attribution, the i4
    calibration conditions) are checked on series only: a published container is pinned by its
    digest instead and keeps the provenance it was published with. The rope-table equality and the
    storage checks apply to every input.

    `spec` is **the slot for an explicit override** (when omitted it is looked up by model name =
    the distribution path). Only the visual-evaluation assembly (`anima.eval_dist`) overrides it,
    passing a spec that "puts the i4 preset back on the plain model and spells the required rounding
    method as the experiment variant" — an entry point that keeps the distribution's accepted set
    ({@link ANIMA_MODELS}) from moving for experiments, never used the other way round (the
    distribution path spelling out a spec).
    """
    assert_model_name(model)
    spec = anima_model(model) if spec is None else spec
    if isinstance(sources, AnimaPublished):
        assert_anima_published(sources)
        placements = dict(sources.placements)
    else:
        series = sources.series if isinstance(sources, AnimaUnpublished) else sources
        assert_lora_provenance(series, spec.lora_sha256)
        if "i4" in spec.storages:
            assert_calib_provenance(series, spec)
        placements = anima_placements(series)
        if isinstance(sources, AnimaUnpublished):
            assert_anima_published(sources.published)
            placements = {
                role: sources.published.placements.get(role, path)
                for role, path in placements.items()
            }
    assert_shared_rope_base(transformer_containers(placements))
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
    """Builds the plan for one Anima model (the CLI dispatches here): from the published containers
    in the local mirrors under {@link DIST_ROOT} for a model {@link ANIMA_PUBLISHED_PART0} lists;
    otherwise its own roles from the `--series` parent and the roles another model holds from the
    mirrors ({@link anima_unpublished_sources}).

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
    if model in ANIMA_PUBLISHED_PART0:
        return anima_plan(anima_published_sources(model, DIST_ROOT), model)
    return anima_plan(anima_unpublished_sources(series_dir, model, DIST_ROOT), model)


def assert_output_keeps_mirrors(
    out_dir: Path,
    models: Sequence[str],
    allowed: tuple[str, ...],
    dist_root: Path | None = None,
) -> None:
    """Refuse a build whose output directory is a local mirror it takes published containers from
    ({@link anima_published_sources}), unless the build holds every model that mirror's manifest
    holds. `allowed` is the models of the pipeline that builds (as in {@link anima_dist_plan});
    `dist_root` is {@link DIST_ROOT} unless given.

    MUST: checked before a byte is written or staged. The assembly replaces its output wholesale
    (`karume.dist.assemble_family`), and without `--out` the output is the repository's mirror
    (`models/<repository>`) while the default selection is one model: such a build would replace
    the mirror with a distribution that lacks the other models, and their published containers,
    which no series can reproduce, would be nowhere on disk. A build that selects every model of
    the mirror into the mirror itself places the same containers again, so it is allowed.
    """
    root = DIST_ROOT if dist_root is None else dist_root
    target = out_dir.resolve()
    # A model outside `allowed` is refused by the plan with its own message; one not published yet
    # reads the mirrors for the roles another model holds only.
    read = sorted(
        {
            anima_published_holder(model, role)[0]
            for model in models
            if model in allowed
            for role in _mirror_roles(model)
        }
    )
    for repo in read:
        mirror = root / repo
        manifest = mirror / MANIFEST_FILENAME
        if mirror.resolve() != target or not manifest.is_file():
            continue
        held = list(json.loads(manifest.read_text(encoding="utf-8")).get("models", {}))
        lost = [name for name in held if name not in models]
        if not lost:
            continue
        keep = [*models, *lost]
        instead = (
            "pass another --out, or select every model of the mirror: "
            + " ".join(f"--model {name}" for name in keep)
            if all(name in allowed for name in keep)
            else "pass another --out"
        )
        raise DistError(
            f"The output {out_dir} is the local mirror of {repo}, which this build reads the"
            f" published containers from, and the build replaces it wholesale: {lost} would be"
            f" lost from it (their published containers would be nowhere on disk) — {instead}"
        )


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

#: Distribution name of the 2.9B repository (Anima-2.9B — {@link ANIMA_2_9B_MODEL_NAME}).
ANIMA_2_9B_REPO_NAME = "karume-anima-2.9b"


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


def _2_9b_repo_name(_model: str) -> str:
    """The 2.9B repository name (the model name is ignored, as for the other two)."""
    return ANIMA_2_9B_REPO_NAME


#: Roles every model outside the official repository — the fine-tunes of the extra repository and
#: the model of the 2.9B one — borrows from the official repository instead of storing them (ADR
#: 0087; ADR 0038 §7 addendum 2026-10-28): the text encoder and the VAE decoder in every text-stack
#: storage, and both tokenizers — the official text stack, byte-identical. Not `text_conditioner`:
#: every such model has its own ({@link text_conditioner_series}).
ANIMA_EXTRA_SHARED_ROLES: tuple[str, ...] = (
    *(
        text_stack_role(component, storage)
        for storage in ANIMA_TEXT_STACK_SERIES
        for component in (ANIMA_TEXT_ENCODER_ROLE, ANIMA_VAE_DECODER_ROLE)
    ),
    *ANIMA_ASSETS.values(),
)


def _official_text_stack_references(_model: str) -> tuple[CrossRepoReference, ...]:
    """`Pipeline.references` of the extra and 2.9B repositories: every model borrows the official
    text stack. The referenced model is the base (`anima-v1.0`) — it holds the text stack, and the
    official repository folds it into `shared/` for all five variants."""
    return (
        CrossRepoReference(
            repo_name=OFFICIAL_REPO_NAME,
            model=ANIMA_BASE_MODEL_NAME,
            roles=ANIMA_EXTRA_SHARED_ROLES,
        ),
    )


#: The family's name map (`karume-name-map/1` — `docs/spec/convert.md`, "The name map"), beside
#: this recipe, where both the upstream tensor names and the initializer names are known. Every
#: repository carries it (`karume.dist.Pipeline.name_map`): the fine-tunes of the extra one ship in
#: the same layouts as the official variants, and so does the 40-block model of the 2.9B one (the
#: rules rename by pattern, whatever the block count).
NAME_MAP_PATH = Path(__file__).with_name("name-map.json")

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
    name_map=NAME_MAP_PATH,
)

#: The entry for `--pipeline anima-extra` (the extra repository karume-anima-extra — third-party
#: fine-tunes).
#: The text stack is referenced in the official repository ({@link
#: _official_text_stack_references}; the publication order is docs/release-runbook.md §0).
EXTRA_PIPELINE = Pipeline(
    default_model="anima-wai-v1.0",
    repo_name=_extra_repo_name,
    plan=lambda series_dir, model: anima_dist_plan(series_dir, model, EXTRA_MODELS),
    card_profiles={
        "anima-extra": partial(render_extra_card, abbreviations=ANIMA_QUANT_ABBREVIATIONS)
    },
    root_files=root_files(EXTRA_NOTICE_MARKDOWN),
    references=_official_text_stack_references,
    name_map=NAME_MAP_PATH,
)

#: The entry for `--pipeline anima-2.9b` (the 2.9B repository karume-anima-2.9b — Anima-2.9B).
#: MUST: a separate Pipeline from the other two, for the same reason as theirs: its notice names a
#: third party's layer expansion, which neither of theirs does. It borrows the official text stack
#: like extra ({@link _official_text_stack_references}); until its first publication its own roles
#: come from its series ({@link anima_unpublished_sources}).
ANIMA_2_9B_PIPELINE = Pipeline(
    default_model=ANIMA_2_9B_MODEL_NAME,
    repo_name=_2_9b_repo_name,
    plan=lambda series_dir, model: anima_dist_plan(series_dir, model, ANIMA_2_9B_MODELS),
    card_profiles={
        "anima-2.9b": partial(render_2_9b_card, abbreviations=ANIMA_QUANT_ABBREVIATIONS)
    },
    root_files=root_files(ANIMA_2_9B_NOTICE_MARKDOWN),
    references=_official_text_stack_references,
    name_map=NAME_MAP_PATH,
)
