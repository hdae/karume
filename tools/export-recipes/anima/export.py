"""The script that writes the four Anima components out as IR v2 + golden io (the emit targets of
ADR 0016).

The starting point is the diffusers version `circlestone-labs/Anima-Base-v1.0-Diffusers` (recon in
docs/research/2026-08-02-anima-recon.md). The plain diffusers modules include rank5–8, feat_cache
and embeddings that belong on the host, so the curation is done by `anima.patch` (the patch layer +
wrappers).

There are four targets (all of ADR 0016; no fused target is built — host processing sits between
the components, so it is structurally impossible):

- **`text_encoder`** — Qwen3's `last_hidden_state`. Dynamic `T` (prompt length ≤ 512).
- **`text_conditioner`** — **two independent symbols**, `Tsrc` (Qwen3 output length) / `Ttgt` (T5
  id sequence length). The cases vary both lengths, so a mix-up that binds one in place of the other
  is exposed by the numbers.
- **`transformer`** — one denoise step of CosmosDiT. The default is **a static graph at a fixed
  resolution** (512px). `--dit-graph dyn` gives **an additional series with the single token-length
  symbol `S`** (#21 phase T2).
- **`vae_decoder`** — image decode at T=1. **Static** (512px by default).

    uv sync --all-groups                                        # once in tools/
    uv run python -m anima.export --out /path/to/out
    uv run python -m anima.export --target vae_decoder --out /path/to/out
    uv run python -m anima.export --verify vae_decoder
    uv run python -m anima.export --dtype f16   # → outputs/series/anima-f16/
    uv run python -m anima.export --dtype i8    # → …/anima-i8/ (DiT only)
    uv run python -m anima.export --dtype i4    # → …/anima-i4/ (DiT only, mixed)
    uv run python -m anima.export --dtype f16 --dit-graph dyn  # → …-f16-dyn/
    uv run python -m anima.export --dtype bf16  # → …/anima-bf16/ (4 roles, upstream's BF16 as is)

MUST: `--dit-graph dyn` is **an additional series for the transformer only** and does not replace
the static series (the premise of phase T2 is that not one existing asset, E2E test or tolerance
moves). Building patchify / unpatchify / the rope tables moves to the host, so besides the weights
the container holds **the asset `rope_base`** (the per-axis base tables from which the host builds
the rope tables — ADR 0109 decision 4).

MUST: `--dtype i8` is **for the transformer only** (the series design of ADR 0019). The DiT's
−1.87GiB is the dominant term, and text / cond / VAE share `outputs/series/anima-f16/`. Moreover the
VAE takes "the temporal slice of CausalConv3d" at patch time, so rounding first would give a
per-channel scale whose amax counts **the elements that are thrown away** (unlike f16's per-element
rounding, a scale moves the value of every element). The CLI mechanically rejects `--dtype i8` for
the other targets.

MUST: `--dtype i4` is **for the transformer only** too (the same reason and the same table as i8 —
{@link DTYPE_TARGETS}), and its content is **mixed** (the eligible weights of {@link
I4_MODULE_TYPES} = i4 group32, the rest = i8 per-channel). The i4 execution path is limited to the
weight slots of linear / embedding / conv1d (ADR 0069 decision 5 and its supplement), so a
single-dtype i4 series cannot be built in principle — the series' **default** storage is i8 ({@link
BASE_WEIGHT_DTYPES}), and each eligible weight is switched to i4 through `weight_dtype_overrides`
(the same form as the mixed deberta / gemma4 series).

The linears inside the DiT blocks are **rounded with GPTQ calibration** (the default — perf-ledger
Q-6 / `anima.calib`). The storage form does not change by a byte (the grid stays RTN i4 g32); only
the rounded values and the contents of the scale ledger change. The calibration inputs are the
first `--calib-prompts` prompts of {@link anima.calib_prompts.CALIB_PROMPTS} (default
{@link DEFAULT_CALIB_PROMPTS}) passed through **a reference denoise** (512², fixed seed, step count
and CFG derived from the distribution defaults of `--model` — {@link anima.calib.calib_conditions}),
capturing the input of the first block across steps (both the cond / uncond branches when CFG > 1).
The i4-eligible weights **outside** the blocks ({@link NON_STAGE_I4_WEIGHTS}) are out of reach of
the calibration driver, so they are rounded with plain RTN **first** (matching the calibration
inputs to the conditions at distribution run time). A calibration failure fails loudly, and there is
no branch that silently falls back to plain RTN — only an explicit `--no-calib` opts out (not used
for distribution assets).

`--i4-adaln-i8` (OFF by default) is **an experimental variant for quantization sensitivity**: it
moves the adaLN inside the blocks (`linear_1` / `linear_2` of `norm1` / `norm2` / `norm3` = 168
layers in anima-v1.0) and the 5 outside the blocks to i8 storage, and applies GPTQ i4 only to the
remaining linears inside the blocks ({@link _adaln_i8_names}). It was the improvement candidate when
the i4 of the three base models was skipped from distribution by the visual review ruling (research
`2026-08-24-gptq-expansion-quality.md` §5), and the same kind of configuration recovered the
listening quality in irodori's w4 preset (R3 of §1 there). Both the series name and the `method` of
`calib_provenance.json` get `-adaln8`, so it **fails the distribution consistency check** (for
visual evaluation only — assembled with `python -m anima.eval_dist`).

`--calib-device` (default `cpu`) is the knob that moves **only the capture + GPTQ** of the
calibration to another device. `cuda` is a path for turning sensitivity experiments around faster,
and **its rounding solution differs from the CPU's** (both the order of the f64 reductions and the
linalg implementation differ), so it is not used for distribution — the `method` of
`calib_provenance.json` carries the device name (`gptq-cuda`) and, like `-adaln8`, it fails the
distribution consistency check. The stage-decomposition match check, the RTN outside the blocks and
the i8 rounding stay on the CPU (confining the confound of the device difference to the rounding
side).

MUST: `--dtype f16` takes the reference and the golden **after rounding the weights to
f16-representable values** (fake-quant — ADR 0006), and stores only the eligible weight slots in
f16 (ADR 0018). The rounding is applied by each builder right after it builds the model — moving it
after the reference would give "only the reference has the original weights", the E2E difference
would become a combination of quantization error and implementation error, and the tolerance would
lose its meaning. The output goes somewhere **separate** from the f32 series (default
`outputs/series/anima-f16/`).

MUST: `--verify` and emit **cannot be combined in one process** (the CLI mechanically rejects it).
The VAE patch replaces class attributes process-wide, so if the emit side applied the patch first,
the "pre-patch reference" could no longer be taken, and the equivalence check would pass vacuously
as a false PASS (ADR 0013 / 0016).

MUST: `--verify` **takes exactly one target**. Allowing several would make the CLI hold a table of
pairwise exclusions, and every hole in the table would become a false PASS (the precedent of
export_sbv2).

MUST: `--lora` is merged **before the DiT's layer truncation (`--num-layers`)**. Merging after it
would silently discard the LoRA entries for the truncated layers as "no target", and the check
against dropped entries (`fuse_lora`'s fail loudly) would not work on the reduced model.

Output layout (the same convention as tiny golden / SBV2):

    <out>/<target>/model.krm              weights, constants + the description in two documents
                                          (+ the asset `rope_base`)
    <out>/<target>/io.<case>.safetensors  inputs and the expected outputs from torch CPU
    <out>/<target>/lora_provenance.json   attribution of the merged LoRA (only for targets with
                                          `--lora` merged)
    <out>/<target>/calib_provenance.json  the i4 rounding conditions (only for the transformer of
                                          `--dtype i4`)
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save, save_file
from torch import nn
from torch.export import Dim

from _shared.paths import SERIES_ROOT
from _shared.upstream import card_license_identifier
from anima.distribution import (
    ADALN_I8_TAG,
    ANIMA_MODELS,
    ANIMA_TEXT_CONDITIONER_ROLE,
    ANIMA_TEXT_ENCODER_ROLE,
    ANIMA_TRANSFORMER_ROLE,
    ANIMA_VAE_DECODER_ROLE,
    CALIB_DEVICES,
    CALIB_PROVENANCE_FILE,
    CALIB_SHIPPABLE_DEVICE,
    LORA_PROVENANCE_FILE,
    ROPE_BASE_ASSET,
    ROPE_BASE_ROLE,
)
from karume.artifacts import staged_publication
from karume.container import AssetInput, Provenance, container_parts
from karume.convert import (
    PRESERVED_OP_PREFIXES,
    PRESERVED_OP_PREFIXES_WITH_ATTENTION,
    normalize_boundary_tensor,
)
from karume.dist import NOTICE_FILENAME, sha256_file
from karume.emit import storage_breakdown
from karume.ir import IrGraph
from karume.pipeline import export_to_file
from karume.quantize import (
    DEFAULT_GROUP_SIZE,
    Int8Report,
    channel_rows,
    fake_quant_int4,
    fake_quant_int8,
    iter_quant_targets,
    round_weights_to_bf16,
    round_weights_to_f16,
)

from . import calib, patch
from .calib_prompts import CALIB_PROMPTS, DEFAULT_CALIB_PROMPTS, calibration_prompts
from .card import ANIMA_METADATA

#: The storage dtypes this recipe handles (like irodori / deberta, **a recipe-specific set**). Not
#: taken from core's `karume.emit.WEIGHT_DTYPES` — the set core can write (i4 added — ADR 0069) and
#: the set for which anima has series and acceptance checks are separate decisions, and coupling
#: them would turn an extension on the core side straight into something this CLI accepts.
WEIGHT_DTYPES: tuple[str, ...] = ("f32", "f16", "bf16", "i8", "i4")

#: The **module types** rounded to i4 group32 (= paired with the ops that have an i4 execution path
#: — `I4_WEIGHT_OPS` of `karume.emit`). The only quantizable type present in the DiT is `nn.Linear`
#: (patchify is `CosmosPatchEmbed.proj` = `nn.Linear` too, and there is no embedding table and no
#: conv — the same fact as `W4_OP_TYPES` of `anima/measure_quant.py`), but it is written by type so
#: as not to write it by name. `nn.Embedding` is listed too so that, if upstream brings in a lookup
#: table, it is picked up **by the general form of the eligibility rule** (if an `nn.Embedding` that
#: is not looked up appears, emit's explicit-assignment check fails naming that FQN as "not
#: eligible", so no named exclusion like deberta's `NON_LOOKUP_EMBEDDINGS` is placed in advance).
#: Conv types are not included — the DiT has no conv. On the core side conv1d is i4-eligible too
#: (`groups == 1` ∧ the row length divisible — ADR 0069 addendum 7), but there is no actual need to
#: add `nn.Conv1d` here.
I4_MODULE_TYPES: tuple[type[nn.Module], ...] = (nn.Linear, nn.Embedding)

#: The i4-eligible weights **outside** the DiT blocks (FQNs inside the wrapper). Calibrated rounding
#: drives stage by stage, so it can round only inside the blocks, and the eligible ones outside are
#: handled by plain RTN i4 ({@link _round_i4_calibrated}).
#:
#: They are declared by name so as not to create "a silent reclassification" — measured (real
#: weights, 28 blocks), the i4-eligible weights outside the blocks are these 5 (`patch_embed.proj`
#: is not eligible because its quantization axis 68 is not divisible by g32, and rope has no
#: weights). If an upstream configuration change adds or removes eligible weights, the check in
#: {@link _round_i4_calibrated} fails naming the FQN (neither silently sending it to the RTN side =
#: a thinner calibration, nor silently dropping it to i8 = only the size reverts, can be read from
#: the numbers).
NON_STAGE_I4_WEIGHTS: frozenset[str] = frozenset(
    {
        "model.time_embed.t_embedder.linear_1",
        "model.time_embed.t_embedder.linear_2",
        "model.norm_out.linear_1",
        "model.norm_out.linear_2",
        "model.proj_out",
    }
)

#: Series name → the **default** storage dtype passed to `export_to_file`. Only the i4 series
#: defaults to i8, and the eligible weights are switched to i4 one by one with
#: `weight_dtype_overrides` (the same form as deberta's `BASE_WEIGHT_DTYPES`).
BASE_WEIGHT_DTYPES: Mapping[str, str] = {
    "f32": "f32",
    "f16": "f16",
    "bf16": "bf16",
    "i8": "i8",
    "i4": "i8",
}

#: Where the real weights come from (HF Hub, cached locally — recon §7).
DEFAULT_REPO = "circlestone-labs/Anima-Base-v1.0-Diffusers"
#: The default location of the outputs (a **series** per storage dtype). One more level of
#: subdirectory per target name. The parent is `SERIES_ROOT` (= outputs/series/) — models/ holds
#: only distributions (_shared.paths).
#:
#: MUST: the f16 series is **a separate directory** (ADR 0018). The f32 series (`anima/`) stays
#: independent as the net that measures "implementation error without quantization", and the f16
#: series checks the implementation error of quantization on top of it. Overwriting the same place
#: would lose one of the nets.
DEFAULT_OUT_ROOTS = {
    "f32": SERIES_ROOT / "anima",
    "f16": SERIES_ROOT / "anima-f16",
    "bf16": SERIES_ROOT / "anima-bf16",
    "i8": SERIES_ROOT / "anima-i8",
    "i4": SERIES_ROOT / "anima-i4",
}

#: Target name = **component name** (= the key of weights in karume.json = the container's graph
#: name). The source of truth for the spelling is the side that does not need torch
#: (`anima/distribution.py`) — the series directory names use the same spelling, but the writer
#: writes into a staging location, so the name is not derived from the directory (container-v1
#: §2.1).
TARGET_TEXT_ENCODER = ANIMA_TEXT_ENCODER_ROLE
TARGET_TEXT_CONDITIONER = ANIMA_TEXT_CONDITIONER_ROLE
TARGET_TRANSFORMER = ANIMA_TRANSFORMER_ROLE
TARGET_VAE_DECODER = ANIMA_VAE_DECODER_ROLE
TARGETS = (
    TARGET_TEXT_ENCODER,
    TARGET_TEXT_CONDITIONER,
    TARGET_TRANSFORMER,
    TARGET_VAE_DECODER,
)

#: The targets emitted per storage dtype (ADR 0019 — the i8 / i4 series are DiT only).
#: MUST: besides narrowing the default, an explicit `--target` is rejected too (the reason is in the
#: module docstring).
DTYPE_TARGETS = {
    "f32": TARGETS,
    "f16": TARGETS,
    # The original-weights preset `bf16` is bf16 for all 4 roles (ADR 0131 decision 3 — upstream is
    # BF16 for all 4 components).
    "bf16": TARGETS,
    "i8": (TARGET_TRANSFORMER,),
    "i4": (TARGET_TRANSFORMER,),
}

#: The shape of the DiT graph. `dyn` = **the additional series** with the single token-length symbol
#: `S` (#21 phase T2).
DIT_GRAPHS = ("static", "dyn")

#: The series directory suffix of `--dit-graph dyn` (to keep the default out apart from the static
#: series).
DYN_SUFFIX = "-dyn"

MODEL_FILE = "model.krm"

#: The provenance baked into the containers (container-v1 §2.3). The licence identifier comes from
#: the card's source of truth ({@link anima.card.ANIMA_METADATA}) so the two never drift apart. HF's
#: `license` is `other`, so the identifier is `license_name` (what the upstream
#: `circlestone-labs/Anima` card declares — {@link _shared.upstream.card_license_identifier}, which
#: fails loudly when it cannot decide). The identifier carries no version: the version lives in the
#: bundled LICENSE.md ("CircleStone Labs Non-Commercial License v1.2") and in `license_link`, pinned
#: to the commit of that text (ruling 37 a, 2026-10-29: v1.2 governs).
PROVENANCE = Provenance(
    license=card_license_identifier(ANIMA_METADATA, "anima.card.ANIMA_METADATA"),
    notice=NOTICE_FILENAME,
)

IO_PREFIX = "io."
IO_SUFFIX = ".safetensors"
INPUT_PREFIX = "input."
OUTPUT_PREFIX = "output."

#: The upper bound of the symbolic dimensions. Both text ones are 512 (the pipeline's `max_length`).
SYM_MAX = 512

#: The number of rows of `encoder_hidden_states` the DiT receives. **A different quantity from
#: `SYM_MAX`**, coming from `AnimaTextConditioner.config.min_sequence_length` (the host zero-pads
#: the conditioner's output up to this — the same value as ADR 0016 / anima.pipeline_ref.
#: MIN_SEQUENCE_LENGTH). Piggybacking it on the upper bound of the symbolic dimensions would
#: silently change the DiT's **static** graph shape the moment `--sym-max` moved (that one is not a
#: symbol, so binding does not absorb it).
MIN_SEQUENCE_LENGTH = 512

#: The generation resolution (px). The DiT / VAE are static graphs at a fixed resolution, so the
#: value is the graph shape itself.
RESOLUTION = 512
#: The VAE's spatial compression ratio (latent resolution = resolution / this).
SPATIAL_COMPRESSION = 8

#: The resolutions the golden cases of the S form (`--dit-graph dyn`) take. **Two-point evaluation**
#: (the same approach as DeBERTa T / SBV2 P), covering the two resolutions used in practice. The
#: first is the example input of the export.
DIT_DYN_RESOLUTIONS = (512, 1024)
#: The upper bound of `Dim("S")`. **The graph side has no token-length constraint** (the rope tables
#: moved to inputs, so there are zero S-dependent baked constants), so this declares the model-side
#: ceiling "128 rows of the base table × 128 = 2048px" as is. Having an upper bound at all is the
#: constant-folding requirement of ADR 0010.
DIT_SYM_MAX = 16384
#: The lower bound of the same. The line that avoids 0 / 1 specialisation (the same as `Dim("T",
#: min=2)` of the text ones).
DIT_SYM_MIN = 2

#: Random numbers are derived from here (independent of the global seed — so that regeneration is
#: byte-identical).
SEED = 20260802


def _generator(salt: int) -> torch.Generator:
    return torch.Generator().manual_seed(SEED + salt)


@dataclass(frozen=True)
class Component:
    """The curated export material of one target.

    `cases` are the golden case names and input tuples (in the same positions as `input_names`).
    `reference` is the output of eager **before the patch**, taken only for `--verify` (the VAE
    patch replaces class attributes process-wide, so a reference not taken before the patch is
    applied makes the check vacuous — ADR 0013).
    """

    module: nn.Module
    dynamic_shapes: Any
    input_names: tuple[str, ...]
    cases: tuple[tuple[str, tuple[torch.Tensor, ...]], ...]
    reference: tuple[tuple[torch.Tensor, ...], ...] | None
    symbol_names: tuple[str, ...] = ("T",)
    #: The i8 / i4 scale ledger (FQN → scale). The keys are **FQNs as seen from `module`**, the same
    #: space as the container's initializer names (= the keys of the binding table) (empty except
    #: for `--dtype i8` / `i4`).
    weight_scales: Mapping[str, torch.Tensor] = field(default_factory=dict)
    #: Per-weight storage dtype assignments (tensor key → dtype). Only the mixed i4 series fills it
    #: (it takes precedence over the default {@link BASE_WEIGHT_DTYPES} —
    #: `karume.emit._plan_weight_dtype`).
    weight_dtype_overrides: Mapping[str, str] = field(default_factory=dict)
    #: A post-stage applied to the graph output before the `--verify` comparison (a verbatim
    #: reproduction of **the stage moved to the host**). The S-form DiT has unpatchify on the host,
    #: so the output has to pass through here to be compared with the reference (the latent of
    #: diffusers before the patch). The second argument is **the case index** — in a series whose
    #: resolution differs per case, the shape of the post-stage changes per case too, so one
    #: identity function is not enough. The default is the identity.
    verify_adapter: Callable[[torch.Tensor, int], torch.Tensor] | None = None
    #: The base tables the host uses outside the graph (carried in the container's asset
    #: `rope_base`; empty by default).
    host_tables: Mapping[str, torch.Tensor] = field(default_factory=dict)

    @property
    def example(self) -> tuple[torch.Tensor, ...]:
        """The example input of the export (**the first case**). dynamic_shapes correspond by
        position."""
        return self.cases[0][1]


def _eval_cases(
    run: Callable[..., torch.Tensor], cases: Sequence[tuple[str, tuple[torch.Tensor, ...]]]
) -> tuple[tuple[torch.Tensor, ...], ...]:
    with torch.no_grad():
        return tuple((run(*inputs),) for _, inputs in cases)


# ---- ① Qwen3 text encoder -----------------------------------------------------


def build_text_encoder(args: argparse.Namespace, verify: bool) -> Component:
    from transformers import Qwen3Model

    model = Qwen3Model.from_pretrained(
        args.repo, subfolder="text_encoder", dtype=torch.float32, attn_implementation="sdpa"
    )
    model.config.use_cache = False
    if args.num_layers is not None:
        model.layers = model.layers[: args.num_layers]
    model.eval()
    # The premise of the wrapper that passes no mask (pin packed sequence detection during tracing
    # to None, the same as eager).
    patch.disable_packed_sequence_detection()
    _fake_quant(args, model, TARGET_TEXT_ENCODER)
    generator = _generator(0)
    # Spread the lengths between the lower bound (2 — the line that avoids 0/1 specialisation) and
    # the upper bound of 512. One case at exactly Tmax exercises implementations that depend on the
    # declared upper bound (building a plan at Tmax and the like).
    cases = tuple(
        (
            f"t{length:03d}",
            (
                torch.randint(
                    5, model.config.vocab_size, (1, length), generator=generator, dtype=torch.long
                ),
            ),
        )
        for length in (5, 24, args.sym_max)
    )
    reference = None
    if verify:
        # The reference is the original path given an all-ones mask (the wrapper passes no mask).
        reference = _eval_cases(
            lambda ids: (
                model(
                    input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False
                ).last_hidden_state
            ),
            cases,
        )
    length = Dim("T", min=2, max=args.sym_max)
    return Component(
        module=patch.AnimaTextEncoder(model),
        dynamic_shapes=({1: length},),
        input_names=("input_ids",),
        cases=cases,
        reference=reference,
    )


# ---- ② Text conditioner -------------------------------------------------------


def build_text_conditioner(args: argparse.Namespace, verify: bool) -> Component:
    from diffusers import AnimaTextConditioner

    model = AnimaTextConditioner.from_pretrained(args.repo, subfolder="text_conditioner")
    model.to(torch.float32).eval()
    _apply_lora(args, model, TARGET_TEXT_CONDITIONER)
    _fake_quant(args, model, TARGET_TEXT_CONDITIONER)
    generator = _generator(1)
    # MUST: give Tsrc and Ttgt **different values in each case**. With equal lengths, a mix-up that
    # "binds one in place of the other" does not show in the numbers (the two-symbol mix-up
    # detection of ADR 0016).
    cases = tuple(
        (
            f"case{index}",
            (
                torch.randn(1, source, model.config.source_dim, generator=generator),
                torch.randint(0, model.config.target_vocab_size, (1, target), generator=generator),
            ),
        )
        for index, (source, target) in enumerate(((24, 16), (48, 23)))
    )
    reference = None
    if verify:
        reference = _eval_cases(
            lambda source, target: patch.reference_conditioner(model, source, target), cases
        )
    source_length = Dim("Tsrc", min=2, max=args.sym_max)
    target_length = Dim("Ttgt", min=2, max=args.sym_max)
    return Component(
        module=patch.AnimaConditioner(model),
        dynamic_shapes=({1: source_length}, {1: target_length}),
        input_names=("source_hidden_states", "target_input_ids"),
        cases=cases,
        reference=reference,
        symbol_names=("Tsrc", "Ttgt"),
    )


# ---- ③ CosmosDiT -----------------------------------------------------------

#: The random salt of the wrapper inputs fed to the stage-decomposition match check of the
#: calibration (`anima.calib.assert_stage_split`). Kept **separate** from the random sequence of the
#: golden cases (salt 2) — the probe is a diagnostic input that never reaches an asset, so mixing
#: them would make the golden values move with the presence of the probe (= with `--dtype`).
PROBE_SALT = 5

#: The timestep of the probe (on the 0–1 sigma scale — the value itself means nothing. The check
#: only compares two paths on the same input, so it can be independent of the two golden points).
PROBE_TIMESTEP = 0.5


def _dit_probe(model: nn.Module, latent: int) -> tuple[torch.Tensor, ...]:
    """The probe input of the static-shape wrapper ({@link patch.AnimaDit}) (used only with
    `--dtype i4`)."""
    generator = _generator(PROBE_SALT)
    return (
        torch.randn(1, model.config.in_channels, latent, latent, generator=generator),
        patch.dit_timesteps_proj(model, torch.full((1,), PROBE_TIMESTEP)),
        torch.randn(1, MIN_SEQUENCE_LENGTH, model.config.text_embed_dim, generator=generator),
    )


def _dit_tokens_probe(model: nn.Module, patch_size: tuple[int, ...]) -> tuple[torch.Tensor, ...]:
    """The probe input of the S-form wrapper ({@link patch.AnimaDitTokens}).

    The resolution is **the smaller one** of the S-form golden (= also the calibration resolution) —
    the check only looks at the path match of the block list, so one point is enough, and choosing
    the larger one would add minutes to the check alone.
    """
    generator = _generator(PROBE_SALT)
    side = DIT_DYN_RESOLUTIONS[0] // SPATIAL_COMPRESSION
    latents = torch.randn(1, model.config.in_channels, side, side, generator=generator)
    return (
        patch.dit_patchify(latents, patch_size),
        patch.dit_timesteps_proj(model, torch.full((1,), PROBE_TIMESTEP)),
        torch.randn(1, MIN_SEQUENCE_LENGTH, model.config.text_embed_dim, generator=generator),
        *patch.dit_rope_tables(model, side, side),
    )


def build_transformer(args: argparse.Namespace, verify: bool) -> Component:
    from diffusers import CosmosTransformer3DModel

    model = CosmosTransformer3DModel.from_pretrained(args.repo, subfolder="transformer")
    model.to(torch.float32).eval()
    # MUST: merge **before** truncating the layers (so that the check against dropped entries covers
    # all layers even on a reduced model).
    _apply_lora(args, model, TARGET_TRANSFORMER)
    if args.num_layers is not None:
        model.transformer_blocks = model.transformer_blocks[: args.num_layers]
    if args.dit_graph == "dyn":
        return _build_transformer_tokens(args, verify, model)
    latent = args.resolution // SPATIAL_COMPRESSION
    # MUST: apply the rounding **through the wrapper that is exported** (to align the keys of the i8
    # scale ledger with the FQN space `torch.export` sees — the docstring of `_fake_quant`). The
    # wrapper references the same Parameters, so the same rounding also applies to the reference
    # taken below (from the raw `model`). Rounding after the truncation is enough (truncated layers
    # appear in neither the export nor the reference). Unlike the LoRA, it does not "check for
    # dropped entries across all layers", so there is no ordering MUST.
    module = patch.AnimaDit(model, latent, latent)
    scales, dtype_overrides = _fake_quant(
        args, module, TARGET_TRANSFORMER, calib_probe=_dit_probe(model, latent)
    )
    generator = _generator(2)
    # The timestep is a continuous value in 0–1 (FlowMatch's sigma scale). Give the two points
    # different values so that the golden numbers exercise the timestep embedding really taking
    # effect as a graph input.
    raw = [
        (
            torch.randn(1, model.config.in_channels, latent, latent, generator=generator),
            torch.full((1,), step),
            torch.randn(1, MIN_SEQUENCE_LENGTH, model.config.text_embed_dim, generator=generator),
        )
        for step in (0.7, 0.15)
    ]
    reference = None
    if verify:
        with torch.no_grad():
            reference = tuple((patch.reference_dit(model, *inputs),) for inputs in raw)
    cases = tuple(
        (
            f"t{int(step.item() * 1000):04d}",
            (latents, patch.dit_timesteps_proj(model, step), encoder_hidden_states),
        )
        for latents, step, encoder_hidden_states in raw
    )
    return Component(
        module=module,
        dynamic_shapes=None,
        input_names=("latents", "timesteps_proj", "encoder_hidden_states"),
        cases=cases,
        reference=reference,
        weight_scales=scales,
        weight_dtype_overrides=dtype_overrides,
    )


def _build_transformer_tokens(
    args: argparse.Namespace, verify: bool, model: nn.Module
) -> Component:
    """The S-form DiT (a single token-length symbol) — **an additional series** (#21 phase T2).

    It differs from the static form only at the entry and the exit (the doc of
    `anima.patch.AnimaDitTokens`). What is decided here are three points about the cases:

    - **The two-point evaluation varies both resolution and timestep** (`DIT_DYN_RESOLUTIONS`).
      Two cases with the same S would not show in the numbers an implementation in which "the
      binding of S has no effect" (the same discipline as not giving the conditioner's two symbols
      the same length).
    - The example input (= `cases[0]`) is **the smaller one**. `torch.export` runs one forward while
      tracing, so putting the larger one first would add minutes to the export alone.
    - The reference of `--verify` is **the latent of diffusers before the patch** (`reference_dit`),
      and the comparison is made against the composition "host patchify → S form → host
      unpatchify" (`verify_adapter`). Comparing against the static wrapper would make the
      verification one level shallower, since the static wrapper's own equivalence can only be
      taken in a separate process.
    """
    patch_size = tuple(int(size) for size in model.config.patch_size)
    module = patch.AnimaDitTokens(model)
    scales, dtype_overrides = _fake_quant(
        args,
        module,
        TARGET_TRANSFORMER,
        calib_probe=_dit_tokens_probe(model, patch_size),
    )
    generator = _generator(2)
    latents = [
        (
            resolution // SPATIAL_COMPRESSION,
            torch.randn(
                1,
                model.config.in_channels,
                resolution // SPATIAL_COMPRESSION,
                resolution // SPATIAL_COMPRESSION,
                generator=generator,
            ),
            torch.full((1,), step),
            torch.randn(1, MIN_SEQUENCE_LENGTH, model.config.text_embed_dim, generator=generator),
        )
        for resolution, step in zip(DIT_DYN_RESOLUTIONS, (0.7, 0.15), strict=True)
    ]
    reference = None
    if verify:
        with torch.no_grad():
            reference = tuple(
                (patch.reference_dit(model, latent, step, embeds),)
                for _, latent, step, embeds in latents
            )
    cases = tuple(
        (
            f"s{(side // patch_size[1]) * (side // patch_size[2]):05d}"
            f"t{int(step.item() * 1000):04d}",
            (
                patch.dit_patchify(latent, patch_size),
                patch.dit_timesteps_proj(model, step),
                embeds,
                *patch.dit_rope_tables(model, side, side),
            ),
        )
        for side, latent, step, embeds in latents
    )
    tokens = Dim("S", min=DIT_SYM_MIN, max=DIT_SYM_MAX)
    sides = [side for side, *_ in latents]
    return Component(
        module=module,
        # The rope tables are declared with the same `S` too (separate symbols would accept a form
        # where "the tables and the body have different lengths", a silent wrong value).
        dynamic_shapes=({1: tokens}, None, None, {2: tokens}, {2: tokens}),
        input_names=("tokens", "timesteps_proj", "encoder_hidden_states", "rope_cos", "rope_sin"),
        cases=cases,
        reference=reference,
        symbol_names=("S",),
        weight_scales=scales,
        weight_dtype_overrides=dtype_overrides,
        verify_adapter=lambda output, index: patch.dit_unpatchify(
            output, sides[index], sides[index], patch_size
        ),
        host_tables=patch.dit_rope_base_tables(model),
    )


# ---- ④ VAE decoder ---------------------------------------------------------


def build_vae_decoder(args: argparse.Namespace, verify: bool) -> Component:
    from diffusers import AutoencoderKLQwenImage

    vae = AutoencoderKLQwenImage.from_pretrained(args.repo, subfolder="vae")
    vae.to(torch.float32).eval()
    _fake_quant(args, vae, TARGET_VAE_DECODER)
    latent = args.resolution // SPATIAL_COMPRESSION
    generator = _generator(3)
    cases = tuple(
        (f"case{index}", (torch.randn(1, vae.config.z_dim, latent, latent, generator=generator),))
        for index in range(2)
    )
    reference = None
    if verify:
        # MUST: the patch replaces class attributes process-wide — take the reference before the
        # patch.
        _assert_vae_unpatched("taking the vae_decoder reference")
        reference = _eval_cases(lambda latents: patch.reference_vae_decode(vae, latents), cases)
    patch.apply_vae_decoder_patch(vae)
    return Component(
        module=patch.AnimaVaeDecoder(vae),
        dynamic_shapes=None,
        input_names=("latents",),
        cases=cases,
        reference=reference,
    )


BUILDERS: dict[str, Callable[[argparse.Namespace, bool], Component]] = {
    TARGET_TEXT_ENCODER: build_text_encoder,
    TARGET_TEXT_CONDITIONER: build_text_conditioner,
    TARGET_TRANSFORMER: build_transformer,
    TARGET_VAE_DECODER: build_vae_decoder,
}

#: Per target, "the high-level ops at which decomposition stops" (ADR 0023).
#:
#: **Only the DiT (transformer) and the VAE decoder** keep SDPA as one fused `attention` node. Their
#: attention calls are all, as measured, **mask-free, non-causal, dropout 0, no GQA, rank-4** (§1 of
#: the fusion design recon).
#:
#: MUST: `text_encoder` is **decomposed** as before — it has a causal mask with −inf folded in,
#: which hits the fail loudly of `_h_attention`, so preserving it would make the export fail. The
#: safe-softmax guard removal pass of ADR 0016 also runs on this decomposed path.
#: NOTE: `text_conditioner` has no mask and could be fused, but its GPU time is 0.10ms/18.7ms (below
#: the measurement limit), so it is outside the scope of v1 (design recon §1.3). Widening it takes
#: just one line here.
TARGET_PRESERVED: dict[str, tuple[str, ...]] = {
    TARGET_TEXT_ENCODER: PRESERVED_OP_PREFIXES,
    TARGET_TEXT_CONDITIONER: PRESERVED_OP_PREFIXES,
    TARGET_TRANSFORMER: PRESERVED_OP_PREFIXES_WITH_ATTENTION,
    TARGET_VAE_DECODER: PRESERVED_OP_PREFIXES_WITH_ATTENTION,
}

#: The targets `--lora` applies to (recon §1 — the distillation LoRA targets only the DiT and the
#: conditioner).
LORA_PREFIXES = {
    TARGET_TRANSFORMER: "transformer",
    TARGET_TEXT_CONDITIONER: "text_conditioner",
}

#: The targets `--num-layers` (layer truncation) applies to (only Qwen3's layer list and the DiT's
#: block list).
NUM_LAYERS_TARGETS = (TARGET_TEXT_ENCODER, TARGET_TRANSFORMER)


def _assert_vae_unpatched(where: str) -> None:
    """Pin that the reference is taken **before** the VAE patch is applied (the guard against a
    vacuous pass — ADR 0013).

    A "reference" taken after the patch is the patched value itself, and the equivalence check
    passes vacuously as a false PASS. It breaks in the direction of **the difference always being
    0**, so a green check is no evidence at all on its own.
    """
    if patch.vae_patches_applied():
        raise RuntimeError(
            f"{where}: VAE パッチ適用後に参照値を採ろうとした — 同値検証が恒真化する"
            "（順序は「全ケースの参照を確定 → パッチ適用 → 比較」）"
        )


def _i4_module_names(model: nn.Module) -> frozenset[str]:
    """The FQN set of the modules rounded to i4 group32 (the single source of the **exclusive
    split** of the mixed i8+i4).

    Eligibility is the conjunction of two conditions: ① the module is one of
    {@link I4_MODULE_TYPES} (emit's i4 eligibility = the weight slots of linear / embedding only —
    ADR 0069 decision 5 and its supplement; explicitly assigning i4 to a tensor outside it makes
    emit fail loudly) ② the quantization axis is divisible by the group length (MUST: i4 makes no
    partial group — decision 2 of the same ADR; a weight that fails it is **excluded from the
    targets instead of dropping the whole configuration**, so it falls to the i8 side).

    In the DiT only the patchify entry fails ② (the in axis of `patch_embed.proj` = 17ch × 2×2 = 68
    is not divisible by g32), but MUST: **it is not excluded by name** — divisibility moves when the
    resolution, patch_size or input channels move, so a named exclusion becomes a lie the moment
    upstream changes.

    The target enumeration goes through core (`iter_quant_targets`) to keep the set the rounding
    sees and the set counted here a single implementation. The i8-side predicate is written as "not
    in this set" for the same reason: building the two predicates from separate spellings would
    leave the weights that fall into neither **silently in f32** (the values stay correct and only
    the size reverts, so it goes unnoticed unless one looks at the numbers).
    """
    return frozenset(
        fqn.removesuffix(".weight")
        for fqn, weight, axis in iter_quant_targets(model, I4_MODULE_TYPES)
        if channel_rows(weight, axis).shape[-1] % DEFAULT_GROUP_SIZE == 0
    )


def _adaln_i8_names(model: nn.Module) -> frozenset[str]:
    """The FQN set of the weights that the sensitivity-experiment variant (`--i4-adaln-i8`) takes
    out of i4 and **rounds to i8**.

    It is the union of two groups, both "on the side that produces the modulation" = the roles
    considered highly sensitive to quantization:

    1. **The adaLN inside the blocks** (`linear_1` / `linear_2` of `norm1` / `norm2` / `norm3` —
       28 blocks × 6 = 168 in anima-v1.0). Matched by segment ({@link anima.calib.is_adaln}); no
       named list is kept.
    2. **The 5 outside the blocks** ({@link NON_STAGE_I4_WEIGHTS}). Four of them are the modulation
       lineage itself (`time_embed.t_embedder` produces the temb passed to every adaLN, and
       `norm_out` is the final `CosmosAdaLayerNorm`), and the remaining `proj_out` is the one that,
       because the calibration driver works per stage, never went through GPTQ and was rounded with
       plain RTN i4 — irodori's w4 preset adopted **the configuration that moved these two groups
       back to i8 together** (R3 of §1 in research `2026-08-24-gptq-expansion-quality.md`), and a
       configuration that moves only one of them has not been verified anywhere.

    MUST: fail loudly if even one adaLN spelling is not found ({@link
    anima.calib.adaln_segments_seen}) — if upstream renamed only some of them, just the ones meant
    to be excluded would silently return to i4, and the storage form and the count check would stay
    self-consistent and green.
    """
    stage_names = calib.stage_linear_names(calib.dit_stages(model))
    missing = sorted(calib.ADALN_SEGMENTS - calib.adaln_segments_seen(stage_names))
    if missing:
        raise AssertionError(
            f"adaLN linear spellings with no match in the DiT blocks: {missing}"
            f" (spellings searched: {sorted(calib.ADALN_SEGMENTS)}) — upstream has changed the"
            " attribute names of the modulation"
        )
    return frozenset(name for name in stage_names if calib.is_adaln(name)) | NON_STAGE_I4_WEIGHTS


def _round_i4_plain(
    model: nn.Module, names: frozenset[str], target: str, label: str
) -> Mapping[str, torch.Tensor]:
    """Round the named set with plain RTN i4 g32 (the grid of the first stage itself, without
    calibration).

    `label` is the object of the diagnostic line (the form that fits "rounded <label> to i4
    groups").
    """
    report = fake_quant_int4(model, include=lambda name: name in names, op_types=I4_MODULE_TYPES)
    print(
        f"[fake-quant] {target}: rounded {label} to i4 groups (RTN) — {report.describe()}",
        flush=True,
    )
    return report.scales


def _round_i4_calibrated(
    args: argparse.Namespace,
    wrapper: nn.Module,
    target: str,
    i4_names: frozenset[str],
    excluded: frozenset[str],
    probe: Sequence[torch.Tensor],
    round_int8: Callable[[], Int8Report],
) -> tuple[Mapping[str, torch.Tensor], Int8Report]:
    """Round the weights stored as i4, split exclusively into "outside the blocks = plain RTN" and
    "inside the blocks = GPTQ".

    `i4_names` is **everything stored as i4**, and `excluded` is the ones that are i4-eligible but
    go to i8 (empty by default; the 173 of {@link _adaln_i8_names} in the sensitivity-experiment
    variant). Everything eligible is the union of the two, and the two checks below apply to that
    (= the facts the graph decides) — the variant only moves the split, and not one of the nets that
    watch "whether the upstream configuration changed" is loosened.

    Order (① → ② → ③ → ④; **②–④ are MUST, ① is SHOULD**):

    1. **Pass the stage-decomposition match check before rounding**
       (`anima.calib.assert_stage_split`) — rounding with a misaligned decomposition would ship
       "GPTQ of a different path", and the numbers would still come out normally. It is only a
       SHOULD because the check is read-only and **moving it later does not change the shipped
       bytes** (the gain of putting it first is failing before hours of calibration).
    2. **Round outside the blocks first** — by default that is RTN i4 (at distribution run time,
       what enters the blocks is the temb made by the i4 `time_embed`, and the latent carried across
       steps has passed through the i4 `norm_out` / `proj_out`). Doing it later would ship rounding
       targets chosen from "activations that passed through an f32 surrounding", combined with a
       rounded surrounding. **The i8 side is also rounded before capturing the calibration inputs,
       for the same reason** (`round_int8` of {@link _fake_quant_i4} is called here) — by default
       that is `patch_embed.proj` alone (the patchify entry = the input of block 0 itself); in the
       variant the 173 excluded ones join it, and the adaLN inside the blocks becomes i8 at this
       point too (at distribution run time the modulation is produced by the i8 adaLN, so the
       calibration inputs are taken in that state).
    3. Generating the calibration inputs (capturing the reference denoise) and the check that the
       accompanying arguments match.
    4. Round the linears inside the blocks with GPTQ × the RTN grid. **The exclusion is made
       exclusive through core's `include`** (`anima.calib.is_adaln` — one implementation decides
       both the rounded set and the set stored as i4).

    MUST: the eligible weights outside the blocks match {@link NON_STAGE_I4_WEIGHTS}, and every
    linear inside the blocks is i4-eligible — a violation of either means the upstream configuration
    changed.

    NOTE (device): `--calib-device` moves only ③④ (capture + GPTQ); ①② run on the CPU (confining the
    confound of the device difference to the rounding method). The return always goes back to the
    CPU through try/finally — emit and the golden are on the CPU path, and weights left on the GPU
    would break the later diagnostics in another way. With the default `cpu`, both `Module.to` and
    `Tensor.to` are no-ops, so not one byte of the distribution path moves.
    """
    stages = calib.dit_stages(wrapper)
    stage_names = calib.stage_linear_names(stages)
    eligible = i4_names | excluded
    unaligned = sorted(stage_names - eligible)
    if unaligned:
        raise AssertionError(
            f"DiT block linears {unaligned[:3]} are not i4-eligible (quantization axis not"
            f" divisible by g{DEFAULT_GROUP_SIZE}) — calibration drives whole stages, so there is"
            " no way out by leaving out one layer"
        )
    non_stage = eligible - stage_names
    if non_stage != NON_STAGE_I4_WEIGHTS:
        raise AssertionError(
            f"the i4-eligible weights outside the DiT blocks are {sorted(non_stage)}, which differs"
            f" from the declaration (NON_STAGE_I4_WEIGHTS = {sorted(NON_STAGE_I4_WEIGHTS)})"
            " — the upstream configuration changed and the i4 split has moved"
        )
    graph_batch = calib.assert_stage_split(wrapper, probe, stages)
    plain_names = non_stage - excluded
    plain = (
        _round_i4_plain(wrapper, plain_names, target, "the eligible weights outside the blocks")
        if plain_names
        else {}
    )
    int8 = round_int8()
    prompts = calibration_prompts(args.calib_prompts)
    conditions = calib.calib_conditions(args.model)
    # The device seam is **here** (right after the i8 rounding) — ①② above (the stage-decomposition
    # match check, the RTN outside the blocks, the i8 rounding) run on the CPU. GPTQ's rounding
    # solution changes with the device, so unless the range where the device difference mixes in is
    # isolated to "capture + GPTQ", one cannot say what difference a visual A/B is looking at.
    wrapper.to(args.calib_device)
    try:
        batches = calib.capture_stage_batches(
            wrapper.model, prompts, repo=args.repo, conditions=conditions
        )
        calib.assert_calib_batches_match_graph(graph_batch, batches)
        rig = calib.CalibRig(stages=stages, batches=batches)
        # MUST: the set the predicate selects = the set inside the blocks stored as i4. The default
        # (no exclusions) passes `None` and **has no predicate at all** — a predicate that re-spells
        # "everything" would drift silently the day the definition of the scan moves, following
        # only on one side.
        stage_i4 = stage_names - excluded
        report, ledger = calib.calibrate_i4(
            rig, None if not excluded else lambda local: not calib.is_adaln(local)
        )
    finally:
        # MUST: what goes to emit are CPU weights (both the graph capture and the golden are on the
        # CPU path). Restore on an exception as well — left on the GPU after a failure, the later
        # diagnostics would turn into a "the weights are missing" shape.
        wrapper.to("cpu")
        if args.calib_device == "cuda":
            torch.cuda.empty_cache()
    calib.assert_calib_covers_scan(report, stage_i4)
    # The ledger comes back on the device where the rounding ran — the storage side (emit) assumes
    # CPU tensors, so it is collected here (`plain` was produced on the CPU as is).
    calibrated = {fqn: scale.cpu() for fqn, scale in ledger.scales.items()}
    # MUST: the two paths are disjoint (an overlap would mean rounding the same weight twice, and
    # only the values would silently go wrong).
    overlap = sorted(set(plain) & set(calibrated))
    if overlap:
        raise AssertionError(
            f"the two i4 paths round the same weights (double rounding): {overlap[:3]}"
        )
    scope = (
        "the DiT block linears"
        if not excluded
        else f"the {len(stage_i4)} non-adaLN linears of the DiT blocks"
        f" (the {len(excluded - NON_STAGE_I4_WEIGHTS)} adaLN + the"
        f" {len(NON_STAGE_I4_WEIGHTS)} outside the blocks are stored as i8 — --i4-adaln-i8)"
    )
    print(
        f"[fake-quant] {target}: rounded {scope} with GPTQ calibration"
        f" — {report.describe()} / {len(prompts)} calibration prompts, {len(batches)} batches"
        f", {rig.tokens:,} tokens ({args.model}: {conditions.steps} steps"
        f", CFG {conditions.guidance:g}, {conditions.branches} branches)",
        flush=True,
    )
    return {**plain, **calibrated}, int8


def _fake_quant_i4(
    args: argparse.Namespace,
    model: nn.Module,
    target: str,
    probe: Sequence[torch.Tensor] | None,
) -> tuple[Mapping[str, torch.Tensor], Mapping[str, str]]:
    """The rounding of the mixed i4 series (stored as i4 = eligible − excluded; the rest = i8
    per-channel).

    The two predicates split the eligible set of {@link _i4_module_names} **exclusively** (the mixed
    MUST of `quantize.py`). The returned override is the map "assign every key of the i4 scale
    ledger to i4", and the emit side fails loudly if it cannot satisfy an explicit assignment (the
    series' default storage is i8, so what went to i8 is not in the override = stays at the
    default).

    `--i4-adaln-i8` (OFF by default) is the sensitivity-experiment variant that decides the i4 set
    by **subtracting** {@link _adaln_i8_names} from the eligible set, and the i8 predicate
    (`name not in i4_names`) picks up what was subtracted automatically. By default the exclusion is
    empty, so neither the path this function takes nor the rounded set changes by a single byte.

    The eligible weights are rounded with calibration by default ({@link _round_i4_calibrated}).
    Only with `--no-calib` is everything stored as i4 rounded with plain RTN — a test / smoke
    opt-out that is **not used for distribution assets**; "shipped uncalibrated while meaning to
    calibrate" cannot be read from the assets, so the diagnostic line states it.
    """
    eligible = _i4_module_names(model)
    # MUST: subtract the exclusions **after computing eligibility** (trimming by name before the
    # eligibility check would make the weights that dropped out of eligibility because the upstream
    # configuration changed indistinguishable from "the weights left out on purpose").
    excluded = _adaln_i8_names(model) if args.i4_adaln_i8 else frozenset()
    outside = sorted(excluded - eligible)
    if outside:
        raise AssertionError(
            f"weights assigned to i8, {outside[:3]}, are not i4-eligible ({len(eligible)}"
            f" eligible / {len(excluded)} excluded) — the upstream configuration changed and the"
            " exclusion spellings match nothing"
        )
    i4_names = eligible - excluded

    def round_int8() -> Int8Report:
        return fake_quant_int8(model, include=lambda name: name not in i4_names)

    if args.no_calib:
        print(
            f"[fake-quant] {target}: --no-calib — plain RTN without calibration"
            " (not for distribution assets; quality is below the baseline of perf-ledger Q-6)",
            flush=True,
        )
        int4_scales = _round_i4_plain(model, i4_names, target, "the weights stored as i4")
        int8 = round_int8()
    else:
        if probe is None:
            raise AssertionError(
                f"{target}: calibrated i4 needs the probe of the stage-decomposition match check"
                " (the builder did not pass one — there is no branch that silently falls back to no"
                " calibration)"
            )
        # On the calibration path the i8 rounding is inserted **before capturing the calibration
        # inputs** (ordering MUST ② — where it is called is owned by {@link _round_i4_calibrated}).
        # Calling it before the eligibility check would swap the diagnostics, failing on "no i8
        # target at all" before the eligibility breakage.
        int4_scales, int8 = _round_i4_calibrated(
            args, model, target, i4_names, excluded, probe, round_int8
        )
    print(f"[fake-quant] {target}: the rest are i8 per-channel — {int8.describe()}", flush=True)
    # MUST: the rounded set = the stored set (the keys of the override). A mismatch takes the form
    # "counted as i4 but not rounded", and an i4 preset turns green with i8 weights mixed in (only
    # the size silently reverts).
    expected = {f"{name}.weight" for name in i4_names}
    if set(int4_scales) != expected:
        raise AssertionError(
            f"格納 i4 の {len(i4_names)} 本に対し丸めたのは {len(int4_scales)} 本"
            f"（過不足: {sorted(set(int4_scales) ^ expected)[:3]}）"
        )
    # MUST: the excluded ones are rounded as i8. The i8 predicate is the negation of `i4_names`, so
    # structurally it cannot miss, but if this misfires they "leak from both i4 and i8 and stay f32"
    # — a breakage where the values are correct and only the size reverts, which shows in neither
    # the storage assignment nor the count check.
    unrounded = sorted(f"{name}.weight" for name in excluded if f"{name}.weight" not in int8.scales)
    if unrounded:
        raise AssertionError(
            f"of the {len(excluded)} weights taken out of i4, {unrounded[:3]} are not rounded as i8"
            " either (they are on neither the i4 nor the i8 side and remain f32)"
        )
    return {**int8.scales, **int4_scales}, dict.fromkeys(int4_scales, "i4")


def _fake_quant(
    args: argparse.Namespace,
    model: nn.Module,
    target: str,
    *,
    calib_probe: Sequence[torch.Tensor] | None = None,
) -> tuple[Mapping[str, torch.Tensor], Mapping[str, str]]:
    """Round the weights to values representable in the storage dtype, and return the scale ledger
    and the per-weight storage assignments (ADR 0006).

    MUST: each builder calls this **before taking `reference`** and **after** merging `--lora`.
    Reversing the order would ① compute only the reference with the original weights and mix
    quantization error into the E2E difference ② leave the merged ΔW outside the rounding, so that
    re-rounding at storage time breaks the correspondence with the golden.

    MUST (i8 / i4 only): apply it to **the very `nn.Module` that is exported**. The keys of the
    scale ledger are the FQNs seen here, and unless they equal the container's initializer names (=
    the keys of the binding table = the FQNs `torch.export` sees) the matching on the emit side
    misses (matching by `id()` is forbidden — ADR 0006).

    `calib_probe` is the wrapper input that i4's calibrated rounding feeds to the
    stage-decomposition match check (built by the builder with **a separate salt** so as not to move
    the random sequence of the golden cases). If it is missing when calibrating, fail loudly — no
    branch silently falls back to "no probe, so no calibration".
    """
    if args.dtype == "f32":
        return {}, {}
    if args.dtype == "i8":
        report = fake_quant_int8(model)
        print(f"[fake-quant] {target}: rounded to i8 per-channel — {report.describe()}", flush=True)
        return report.scales, {}
    if args.dtype == "i4":
        return _fake_quant_i4(args, model, target, calib_probe)
    if args.dtype == "bf16":
        rounded = round_weights_to_bf16(model)
        print(
            f"[fake-quant] {target}: rounded to bf16-representable values — {rounded.describe()}",
            flush=True,
        )
        return {}, {}
    rounded = round_weights_to_f16(model)
    print(
        f"[fake-quant] {target}: rounded to f16-representable values — {rounded.describe()}",
        flush=True,
    )
    return {}, {}


def _write_lora_provenance(args: argparse.Namespace, target: str, out_dir: Path) -> str | None:
    """Write the attribution of the merged LoRA (file name + sha256) into the series, and return the
    file name.

    "Which LoRA was merged" cannot be recovered from the fused weights, so **the side that merged it
    writing the fact down** is the only way. The model's declaration (`AnimaModel.lora_sha256` in
    `anima/distribution.py`) is checked against this record by `assert_lora_provenance` in the same
    file.

    It is written only for targets that were actually merged (placing it in the series of a target
    not in `LORA_PREFIXES` would make the untrue claim "this model was merged too"). It does not go
    into the distribution — only the roles the placement table of `anima/distribution.py` allows
    reach the output.

    MUST: **delete a stale record** on the side that was not merged (a total function). When a
    series taken with `--lora` is re-taken without `--lora`, if only the weights went back to plain
    and the record survived from the previous run, `assert_lora_provenance` — which only looks at
    "record present × sha matches" — would let the "withdrawal" through and have the distribution
    README print a false attribution. That the record exists exactly when "the current `model.krm`
    was merged" is the only thing this mechanism rests on. NOTE: now that `emit_target` swaps in the
    whole staging location, on the CLI path the location is fresh every time and this effectively
    does nothing — it still stays a total function because the invariant must not change for writers
    that call it directly (tests, a future entry point).
    """
    if args.lora is None or target not in LORA_PREFIXES:
        (out_dir / LORA_PROVENANCE_FILE).unlink(missing_ok=True)
        return None
    record = {"file": args.lora.name, "sha256": sha256_file(args.lora)}
    (out_dir / LORA_PROVENANCE_FILE).write_text(
        json.dumps(record, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return LORA_PROVENANCE_FILE


def _write_calib_provenance(args: argparse.Namespace, target: str, out_dir: Path) -> str | None:
    """Write the rounding conditions of the i4 series (method, grid, corpus, resolution, steps, CFG,
    device) into the series, and return the file name.

    Calibration or its absence **does not change the storage form by a byte**, so it cannot be
    recovered from the assets — as with `--lora`, the only way is "the side that wrote it out writes
    the fact down". `assert_calib_provenance` in `anima/distribution.py` checks this record at
    assembly time and stops outputs of `--no-calib` from slipping into a distribution.

    MUST: outside i4, **delete a stale record** (a total function — the same reason as
    `_write_lora_provenance`, and the same in that the swap makes it a no-op). When a series taken
    as i4 is re-taken with another dtype and only the record survives from the previous run, the
    untrue claim "calibrated" would remain.
    """
    if args.dtype != "i4" or target != TARGET_TRANSFORMER:
        (out_dir / CALIB_PROVENANCE_FILE).unlink(missing_ok=True)
        return None
    # Write it **even** with `--no-calib` (record `rtn` instead of deleting) — an absence could also
    # read as "an old export", and the assembly side could not reject "tried to ship uncalibrated"
    # by name.
    #
    # `guidance` is the field added on 2026-08-23 (phase J-4 ②), when the calibration conditions
    # became per model. The reader (`distribution.assert_calib_provenance` →
    # `_shared.calib_provenance.calib_complaint`) judges only the fields present in the record and
    # accepts an absent field (the MUST of that module), so adding the field changes not one
    # acceptance of the existing turbo series' records (the form before the addition) — MUST: keep
    # it to a form that does not require re-exporting existing assets (unlike assets, the cost of
    # rebuilding a record is the rounding time itself).
    conditions = None if args.no_calib else calib.calib_conditions(args.model)
    # The sensitivity-experiment variant changes **the method's spelling itself** (`gptq` →
    # `gptq-adaln8`). Even with the same rounding method, a different "which weights it was applied
    # to" is a different way of rounding, and not one byte of the storage form tells them apart —
    # making it a spelling that the distribution side's consistency check
    # (`distribution.CALIB_SHIPPABLE_METHOD`) rejects by name as is is the only way to keep variants
    # baked for visual review out of the distribution path ({@link ADALN_I8_TAG}).
    method = "rtn" if args.no_calib else calib.CALIB_METHOD
    if args.i4_adaln_i8:
        method = f"{method}-{ADALN_I8_TAG}"
    # The calibration device also changes **the method's spelling itself** (`gptq` → `gptq-cuda`).
    # GPTQ's rounding solution changes with the device (both the order of the f64 reductions and the
    # linalg implementation differ), so i4 baked on cuda is not "i4 baked under the distribution
    # conditions" — and not one byte of the storage form tells them apart. The `device` field is a
    # fact for people and for later cross-checks; what stops distribution is the spelling ({@link
    # anima.distribution.CALIB_SHIPPABLE_DEVICE}).
    if args.calib_device != CALIB_SHIPPABLE_DEVICE:
        method = f"{method}-{args.calib_device}"
    record: dict[str, object] = {
        "method": method,
        "group_size": calib.CALIB_GRID.group_size,
        "grid": calib.CALIB_GRID.kind,
        "prompts": 0 if args.no_calib else args.calib_prompts,
        "resolution": 0 if args.no_calib else calib.CALIB_RESOLUTION,
        "steps": 0 if conditions is None else conditions.steps,
        "guidance": 0 if conditions is None else conditions.guidance,
        "text_dtype": calib.CALIB_TEXT_DTYPE,
        "device": args.calib_device,
    }
    (out_dir / CALIB_PROVENANCE_FILE).write_text(
        json.dumps(record, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return CALIB_PROVENANCE_FILE


def _apply_lora(args: argparse.Namespace, model: nn.Module, target: str) -> None:
    if args.lora is None:
        return
    from .lora import fuse_lora, load_lora_state_dict

    state = load_lora_state_dict(args.lora)
    report = fuse_lora(model, state, LORA_PREFIXES[target], args.lora_scale)
    print(f"[lora] {target}: {report.describe()}", flush=True)


# ---- emit / golden ---------------------------------------------------------


def _write_io(component: Component, graph: IrGraph, out_dir: Path) -> list[str]:
    """Write each case's inputs and the expected torch CPU outputs to `io.<case>.safetensors`.

    MUST: first pin that the declared names of the graph inputs match the wrapper's argument names
    (the IR's input names are forward's argument names as is). Writing by position while they
    disagree would make only the golden keys silently point at a different input.
    """
    declared = [entry.name for entry in graph.inputs]
    if declared != list(component.input_names):
        raise AssertionError(
            f"グラフ入力名が宣言と不一致: {declared} vs {list(component.input_names)}"
        )
    written: list[str] = []
    for name, inputs in component.cases:
        with torch.no_grad():
            result = component.module(*inputs)
        outputs = (result,) if isinstance(result, torch.Tensor) else tuple(result)
        if len(outputs) != len(graph.outputs):
            raise AssertionError(
                f"{name}: {len(outputs)} eager outputs vs {len(graph.outputs)} IR outputs"
            )
        # MUST: io is also lowered to the actual representation of the IR's semantic dtype (the
        # boundary normalisation of ADR 0009).
        tensors = {
            f"{INPUT_PREFIX}{key}": normalize_boundary_tensor(value, f"{name}: input '{key}'")
            for key, value in zip(component.input_names, inputs, strict=True)
        }
        for index, value in enumerate(outputs):
            tensors[f"{OUTPUT_PREFIX}{index}"] = normalize_boundary_tensor(
                value.detach().contiguous(), f"{name}: output {index}"
            )
        path = out_dir / f"{IO_PREFIX}{name}{IO_SUFFIX}"
        save_file(tensors, str(path))
        written.append(path.name)
    return written


def _host_table_assets(component: Component) -> dict[str, AssetInput]:
    """The host base tables into the container's assets (empty for targets without any).

    The payload is **the safetensors bytes themselves** of one base table. The tensor container is
    kept because the reader (the rope assembly on the models side) needs the per-axis names and
    shapes — an asset is the slot for "bytes that are not weights but ship in the same container",
    and the container does not interpret their content (ADR 0109 decision 4).
    """
    if not component.host_tables:
        return {}
    payload = save(dict(component.host_tables))
    return {ROPE_BASE_ASSET: AssetInput(ROPE_BASE_ROLE, len(payload), payload)}


def emit_target(target: str, args: argparse.Namespace, out_dir: Path) -> dict[str, Any]:
    """Write the IR container and the golden io of one target, and return a summary.

    MUST: the outputs are written to a staging location and put in place only after passing **every
    check** (output count, boundary normalisation). Placing them at the final location before the
    checks would let a failed run leave "assets that can pass the acceptance check" — the io golden
    is taken from the same broken component, so they are consistent with each other and the TS-side
    comparison **turns green** (the spelling of "when publishing is allowed" is
    {@link _shared.decode_series._publish}; the discipline of the swap and the cleanup is core's
    primitive {@link karume.artifacts.staged_publication}). The unit put in place is **the whole
    target directory**, so a set whose container and provenance records (LoRA / calibration)
    disagree cannot be created either.

    MUST: `target` is an element of {@link TARGETS} (= component name = the key of weights in
    `karume.json`). The spelling named here becomes the container's graph name as is, so accepting a
    string outside the table would silently bake **a container the runtime cannot look up** (the day
    an off-table entry appeared in `BUILDERS`, the spelling check would stay green).
    """
    if target not in TARGETS:
        raise AssertionError(
            f"未知のターゲット '{target}'（既知: {', '.join(TARGETS)}）"
            " — ターゲット名はそのまま容器のグラフ名になる"
        )
    started = time.perf_counter()
    component = BUILDERS[target](args, False)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    with staged_publication(out_dir) as staged:
        # The writer creates the directory slot (the primitive does not create it — it only passes a
        # path).
        staged.mkdir()
        graph = export_to_file(
            component.module,
            component.example,
            staged / MODEL_FILE,
            provenance=PROVENANCE,
            # The graph name is **the component name** (= the key of weights in karume.json). The
            # target name is the component name as is, so it is not derived from the directory name
            # (container-v1 §2.1).
            graph_name=target,
            assets=_host_table_assets(component),
            dynamic_shapes=component.dynamic_shapes,
            symbol_names=component.symbol_names,
            weight_dtype=BASE_WEIGHT_DTYPES[args.dtype],
            weight_scales=component.weight_scales,
            weight_dtype_overrides=component.weight_dtype_overrides,
            preserved=TARGET_PRESERVED[target],
        )
        written = _write_io(component, graph, staged)
        provenance = _write_lora_provenance(args, target, staged)
        if provenance is not None:
            written.append(provenance)
        calib_record = _write_calib_provenance(args, target, staged)
        if calib_record is not None:
            written.append(calib_record)
    breakdown = storage_breakdown(graph)
    return {
        "target": target,
        "dir": str(out_dir),
        "dtype": args.dtype,
        "dit_graph": args.dit_graph if target == TARGET_TRANSFORMER else "static",
        # The conditions of the calibrated rounding (0 outside i4 and with `--no-calib`) — this
        # summary goes only to stdout, so it is **for people to read**. Machine cross-checks are
        # owned by `calib_provenance.json` ({@link _write_calib_provenance} →
        # `distribution.assert_calib_provenance`).
        "calib_prompts": 0 if args.dtype != "i4" or args.no_calib else args.calib_prompts,
        "calib_resolution": 0 if args.dtype != "i4" or args.no_calib else calib.CALIB_RESOLUTION,
        # The per-role storage split (the sensitivity-experiment variant). Emitted so that the
        # record of an export lasting hours can be read from one summary — machine cross-checks are
        # owned by the `method` of `calib_provenance.json`.
        "i4_adaln_i8": args.dtype == "i4" and args.i4_adaln_i8,
        "nodes": len(graph.nodes),
        "outputs": len(graph.outputs),
        "initializers": len(graph.initializers),
        # Eligible = weights resident on the GPU still compressed / not eligible = expanded to f32
        # at load time (the diagnostics of ADR 0006).
        "compressed_tensors": breakdown.compressed_tensors,
        "compressed_bytes": breakdown.compressed_bytes,
        "plain_tensors": breakdown.plain_tensors,
        "plain_bytes": breakdown.plain_bytes,
        # The i8 companion scales (the runtime's residentCompressedBytes adds the same thing).
        "scale_bytes": breakdown.scale_bytes,
        "model_bytes": sum(p.stat().st_size for p in container_parts(out_dir / MODEL_FILE)),
        "ops": sorted(graph.required_ops),
        "symbols": list(graph.symbols),
        "io": written,
        "case_shapes": {
            name: [list(tensor.shape) for tensor in inputs] for name, inputs in component.cases
        },
        "seconds": round(time.perf_counter() - started, 1),
    }


# ---- Eager equivalence before and after the patch ------------------------------


def verify_target(target: str, args: argparse.Namespace) -> list[dict[str, Any]]:
    """Measure the eager equivalence before and after the patch layer with the real weights
    (ADR 0016).

    `bit_exact` is a stronger claim than "difference 0" (`0.0 == -0.0` has difference 0 but
    different bits), and it matters for checks where **bit identity is the content of the claim**,
    such as dropping the masks of Qwen3 / the conditioner.
    """
    component = BUILDERS[target](args, True)
    if component.reference is None:
        raise AssertionError(f"{target}: --verify, but no reference was taken")
    report: list[dict[str, Any]] = []
    for index, ((name, inputs), expected) in enumerate(
        zip(component.cases, component.reference, strict=True)
    ):
        with torch.no_grad():
            result = component.module(*inputs)
            if component.verify_adapter is not None:
                result = component.verify_adapter(result, index)
        outputs = (result,) if isinstance(result, torch.Tensor) else tuple(result)
        report.append(
            {
                "case": name,
                "shapes": [list(tensor.shape) for tensor in inputs],
                "maxdiff": max(
                    float((got - want).abs().max())
                    for got, want in zip(outputs, expected, strict=True)
                ),
                "bit_exact": all(
                    torch.equal(got, want) for got, want in zip(outputs, expected, strict=True)
                ),
            }
        )
    return report


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output directory (default: the series for each --dtype —"
        " outputs/series/anima{,-f16,-bf16,-i8,-i4}/, ADR 0018)",
    )
    parser.add_argument(
        "--dtype",
        choices=WEIGHT_DTYPES,
        default="f32",
        help=(
            "storage dtype of the weights (f16 / i8 / i4 are fake-quantized and then only the"
            " eligible slots are stored compressed. i4 is mixed: eligible weights are group32 i4,"
            " the rest i8 — ADR 0069. The linears inside the DiT blocks are rounded to i4 with GPTQ"
            " calibration (perf-ledger Q-6). i8 / i4 are transformer only — ADR 0019)"
        ),
    )
    parser.add_argument(
        "--target",
        action="append",
        choices=TARGETS,
        default=None,
        help=f"target to emit (repeatable; default all: {', '.join(TARGETS)})",
    )
    parser.add_argument(
        "--verify",
        choices=TARGETS,
        default=None,
        help="verify eager equivalence before and after the patch (no emit; one target per"
        " process)",
    )
    parser.add_argument(
        "--dit-graph",
        choices=DIT_GRAPHS,
        default=DIT_GRAPHS[0],
        help=(
            "shape of the DiT graph (dyn = the additional series with the single token-length"
            " symbol S, transformer only. -dyn is appended to the default out — #21 phase T2)"
        ),
    )
    parser.add_argument(
        "--sym-max", type=int, default=SYM_MAX, help="upper bound of the text symbolic dimensions"
    )
    parser.add_argument(
        "--resolution", type=int, default=RESOLUTION, help="generation resolution (px)"
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=None,
        help="layer truncation of the DiT / Qwen3 (for development iterations — the LoRA is merged"
        " before truncation)",
    )
    parser.add_argument(
        "--lora",
        type=Path,
        default=None,
        help="LoRA merged into the weights before export (applies to transformer /"
        " text_conditioner)",
    )
    parser.add_argument("--lora-scale", type=float, default=1.0, help="scale of the LoRA")
    parser.add_argument(
        "--calib-prompts",
        type=int,
        default=DEFAULT_CALIB_PROMPTS,
        help=f"number of prompts for the i4 GPTQ calibration (the first N; default"
        f" {DEFAULT_CALIB_PROMPTS}, at most {len(CALIB_PROMPTS)} — the count is an axis for raising"
        " quality and scales the export's CPU time linearly)",
    )
    parser.add_argument(
        "--no-calib",
        action="store_true",
        help="round i4 with plain RTN without calibration (for tests / smoke — not for distribution"
        " assets)",
    )
    parser.add_argument(
        "--calib-device",
        choices=CALIB_DEVICES,
        default=CALIB_SHIPPABLE_DEVICE,
        help="device on which the GPTQ calibration (capture + rounding) runs"
        f" (default {CALIB_SHIPPABLE_DEVICE}). cuda is for sensitivity experiments — its rounding"
        " solution differs from the CPU's, so it is not used on the distribution path (the method"
        " in calib_provenance carries the device name and fails the distribution consistency"
        " check)",
    )
    parser.add_argument(
        "--i4-adaln-i8",
        action="store_true",
        help="quantization-sensitivity experiment variant of i4: moves the adaLN inside the blocks"
        " (linear_1/2 of norm1/2/3) and the"
        f" {len(NON_STAGE_I4_WEIGHTS)} outside the blocks to i8 storage, and makes only the rest i4"
        f" (default OFF; the series name and the method in calib_provenance get -{ADALN_I8_TAG},"
        " which fails the distribution consistency check = for visual evaluation only)",
    )
    parser.add_argument(
        "--model",
        choices=tuple(ANIMA_MODELS),
        default=None,
        help="distribution model name from which the calibration conditions (steps / CFG / negative"
        " prompt) are taken (required for the --dtype i4 calibration — no default)",
    )
    args = parser.parse_args(argv)
    if args.out is None:
        # MUST: a variant goes to **a separate directory** even with the default out (so that a
        # variant for visual review never overwrites, at the same path, the i4 series baked under
        # the distribution conditions). The calibration device is the same hole — i4 baked on cuda
        # has a different rounding solution, yet sharing the default out would silently clobber the
        # series of the distribution conditions. The suffix order is `-adaln8` → `-cuda` → `-dyn`,
        # so that the graph shape lines up at the end when grepping.
        root = DEFAULT_OUT_ROOTS[args.dtype]
        name = root.name + (f"-{ADALN_I8_TAG}" if args.i4_adaln_i8 else "")
        if args.calib_device != CALIB_SHIPPABLE_DEVICE:
            name += f"-{args.calib_device}"
        args.out = root.with_name(name + (DYN_SUFFIX if args.dit_graph == "dyn" else ""))

    # MUST: mechanically reject combining them in one process. The VAE patch is a process-wide
    # replacement, so if the emit side applied it first, "the pre-patch reference" would be
    # contaminated and the equivalence check would pass vacuously as a false PASS (ADR 0013 / 0016).
    # Keeping the order right inside main is not left to discipline.
    if args.verify is not None and args.target is not None:
        parser.error(
            "--verify and --target cannot be combined in one process (the process-wide replacement"
            " of the VAE patch contaminates the reference and makes the equivalence check vacuous)"
        )

    # MUST: reject a target the storage dtype cannot handle **even when given explicitly** (only
    # narrowing the default would let `--target vae_decoder --dtype i8` through, and assets of a
    # form ADR 0019 ruled out would silently appear). `--verify` is checked against the same table
    # (no dtype that passes only for verification).
    allowed = DTYPE_TARGETS[args.dtype]
    requested = [*(args.target or []), *([args.verify] if args.verify is not None else [])]
    outside = sorted({name for name in requested if name not in allowed})
    if outside:
        parser.error(
            f"--dtype {args.dtype} handles only the targets {', '.join(allowed)}"
            f" (given: {', '.join(outside)} — the series design of ADR 0019)"
        )

    # MUST: the calibration knobs apply only to i4. Accepting them for another series would let
    # through an unreadable request such as "an f16 asset meant to be without calibration" (f16 /
    # i8 / f32 have no calibration path at all).
    if args.dtype != "i4" and (args.no_calib or args.calib_prompts != DEFAULT_CALIB_PROMPTS):
        parser.error(
            f"--calib-prompts / --no-calib apply only to --dtype i4 (given {args.dtype})"
            " — calibrated rounding is the path of the i4 series (perf-ledger Q-6)"
        )

    # MUST: the per-role storage split also applies only to i4 (other series have no i4 / i8 split
    # at all). Accepting it would let through an unreadable request: "an f16 asset meant to have its
    # adaLN in i8".
    if args.dtype != "i4" and args.i4_adaln_i8:
        parser.error(f"--i4-adaln-i8 applies only to --dtype i4 (given {args.dtype})")

    # MUST: the calibration conditions have to be named **per model** (no default). The step count
    # and CFG are derived from the distribution defaults (`calib.calib_conditions`), so a base-model
    # i4 that forgot `--model` must not be silently calibrated under turbo's conditions (8 steps,
    # CFG 1) — the resulting assets would have the right storage form and count, pass `verify_dist`
    # and the header checks, and `calib_provenance.json` would record those conditions as fact (a
    # record that lies makes the whole cross-check mechanism meaningless).
    calibrating = args.dtype == "i4" and not args.no_calib
    if calibrating and args.model is None:
        parser.error(
            f"the --dtype i4 calibration needs --model (choices: {', '.join(ANIMA_MODELS)})"
            " — the calibration steps / CFG come from the distribution defaults"
            " (anima.calib.calib_conditions)"
        )
    if args.model is not None and not calibrating:
        knob = f"--dtype {args.dtype}" + (", --no-calib" if args.no_calib else "")
        parser.error(
            f"--model is a knob only for taking the --dtype i4 calibration conditions"
            f" (given {knob})"
        )

    # MUST: the calibration device is also a knob that applies only when calibrating (the same as
    # `--calib-prompts` / `--no-calib`). Accepting it would let through an unreadable request: "an
    # f16 asset meant to be baked on the GPU".
    if args.calib_device != CALIB_SHIPPABLE_DEVICE and not calibrating:
        knob = f"--dtype {args.dtype}" + (", --no-calib" if args.no_calib else "")
        parser.error(f"--calib-device applies only to the --dtype i4 calibration (given {knob})")

    # MUST: fail on an unusable device **before the calibration starts**. Calibration is a path of
    # tens of minutes to hours, and failing at a `Module.to` midway would throw away the state in
    # which the RTN outside the blocks and the i8 rounding are already done.
    if args.calib_device == "cuda" and not torch.cuda.is_available():
        parser.error(
            "--calib-device cuda was given but torch sees no CUDA (CPU build of torch / no driver"
            " found — there is no branch that silently falls back to the CPU)"
        )

    # MUST: do not silently accept a knob that has no effect. The S form is transformer only (the
    # other 3 targets do not depend on the resolution, so they are shared), and the resolutions are
    # decided by the case table DIT_DYN_RESOLUTIONS (`--resolution` has no effect at all).
    # Accepting it would produce "an asset meant to be 512 that actually appeared unrelated".
    if args.dit_graph == "dyn":
        others = sorted({name for name in requested if name != TARGET_TRANSFORMER})
        if others:
            parser.error(
                f"--dit-graph dyn is for {TARGET_TRANSFORMER} only (given: {', '.join(others)})"
            )
        if args.resolution != RESOLUTION:
            parser.error(
                "--resolution has no effect with --dit-graph dyn"
                f" (the golden resolutions are fixed at {DIT_DYN_RESOLUTIONS} — the graph has no"
                " resolution)"
            )
        if args.target is None and args.verify is None:
            args.target = [TARGET_TRANSFORMER]

    # The targets that actually run. Without `--target` it is the dtype's default set (`allowed`),
    # so the knob checks below judge by this set, not by `requested` (so as not to reject when the
    # default includes a LoRA target).
    targets = (
        [args.verify]
        if args.verify is not None
        else list(dict.fromkeys(args.target if args.target is not None else allowed))
    )

    # MUST: do not silently accept a knob that has no effect (the same strength as `--dit-graph dyn`
    # / the calibration knobs). A `--lora` with no LoRA-capable target among the targets merges
    # nothing and writes no record — "a series meant to have the LoRA merged" would appear without a
    # record. `--lora-scale` has no effect without `--lora`, and `--num-layers` truncates nothing
    # without a target that has a layer list.
    if args.lora is not None and not any(name in LORA_PREFIXES for name in targets):
        parser.error(
            f"--lora applies only to the targets {', '.join(LORA_PREFIXES)}"
            f" (targets: {', '.join(targets)})"
        )
    if args.lora is None and args.lora_scale != 1.0:
        parser.error(f"--lora-scale applies only together with --lora (given {args.lora_scale})")
    if args.num_layers is not None and not any(name in NUM_LAYERS_TARGETS for name in targets):
        parser.error(
            f"--num-layers applies only to the targets {', '.join(NUM_LAYERS_TARGETS)}"
            f" (targets: {', '.join(targets)})"
        )

    # MUST: the calibration resolution matches the graph resolution. The calibration inputs are
    # fixed at `calib.CALIB_RESOLUTION` (a design that keeps the conditions under which the quality
    # ruling was made from moving — the NOTE there), so varying `--resolution` would silently
    # produce a graph baked with "rounding targets chosen at another resolution" (only the
    # activation token count changes, and the numbers come out normally). dyn already rejected
    # `--resolution` above, so this applies only to the static form.
    if args.dtype == "i4" and not args.no_calib and args.resolution != calib.CALIB_RESOLUTION:
        parser.error(
            f"the --dtype i4 calibration is fixed at {calib.CALIB_RESOLUTION}px (given"
            f" {args.resolution}px) — moving the conditions under which the quality ruling was made"
            " means re-collecting the evidence (calib.CALIB_RESOLUTION)"
        )

    if args.verify is not None:
        report = verify_target(args.verify, args)
        for entry in report:
            print(
                f"{args.verify} {entry['case']}: shapes={entry['shapes']}"
                f" maxdiff={entry['maxdiff']:.3e} bit_exact={entry['bit_exact']}"
            )
        print(f"worst maxdiff = {max(entry['maxdiff'] for entry in report):.3e}")
        print(f"bit_exact all = {all(entry['bit_exact'] for entry in report)}")
        return

    summaries = [emit_target(target, args, args.out / target) for target in targets]
    print(json.dumps(summaries, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
