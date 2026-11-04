"""Rig that runs **calibrated rounding** (GPTQ × RTN grid) for anima's i4 series (perf-ledger Q-6).

`karume.quant_calib.calibrate_stages` is a driver that "advances one stage at a time and re-picks
the rounding targets from the activations that actually flow through that layer". The storage grid
is **not a single byte different** from {@link karume.quantize.quantize_to_int4}
(`scale = group amax / 7`, `q ∈ [−7,+7]`). Only the rounded values and the contents of the scale
ledger change; neither the storage path on the `emit` side nor the byte structure of the
distribution moves at all.

This module provides the three things the core driver requires, plus one anima-specific check:

1. **The stage list** — the DiT's `transformer_blocks` laid out as they are ({@link dit_stages}).
   `CosmosTransformerBlock.forward` "takes hidden positionally and the rest as keywords and
   returns hidden", so the blocks are **not wrapped** (re-wrapping could let the copy drift from
   upstream). The prefix lives in **the FQN space of the exported wrapper**
   ({@link STAGE_PREFIX} = `model.transformer_blocks`) — the measurement rig
   (`anima.measure_quant`) drives the bare `CosmosTransformer3DModel` and uses
   `transformer_blocks.<i>`, whereas export goes through `patch.AnimaDit`, which inserts one
   `model.` level. Dropping it makes the scale ledger's keys miss the container's initializer
   names (= the binding table's keys).
2. **The input to the first stage** — {@link capture_stage_batches} **actually runs the
   reference denoise** and catches the `(args, kwargs)` to the first block with a
   forward_pre_hook. Not rebuilding it by hand follows the same reasoning as deberta and the
   measurement rig; for a diffusion model, in addition, "the activations move with sigma", so the
   capture spans **all steps** (looking at a single step leaves the distribution of the later
   steps out of the calibration).
3. **The stage-split equality check** — {@link assert_stage_split}. deberta's check of the same
   name asks "does the `EncoderStage` copy drift from the upstream loop", but anima uses the
   blocks themselves as stages, so that form would be degenerate. Instead it checks that "running
   the wrapper's block loop stage by stage is bit-identical to the input to `norm_out` in one
   forward of the wrapper" (MUST be measured before a single rounding is applied).
4. **The check that the calibration inputs and the graph share the same auxiliary arguments** —
   {@link assert_calib_batches_match_graph}. The capture comes from the forward of the bare
   `CosmosTransformer3DModel`, but the weights that are rounded are the wrapper's
   (`patch.AnimaDit`). If the set of auxiliary arguments the two pass to a block differs, we ship
   "rounding targets chosen under conditions other than the distributed graph's" (the numbers come
   out normally, so the golden stays green too). Only **the set of keyword names** is checked;
   values, shapes and positional arguments are not compared (it does not claim "the path is the
   same" — it is a check that only catches a mismatch in the set).

NOTE (why this does not share the rig of `anima.measure_quant`): that one is scaffolding for 3
grids × quality measurement, and both its target and its driver stay within "the bare model". The
export side needs the wrapper's FQN space, the distribution conditions (round what is outside the
stages first) and extraction of a shippable ledger. Merging the copies into one would make one side
follow silently when the other side's premises move — instead each rig verifies itself
independently with its own equality checks.
"""

from __future__ import annotations

import gc
import inspect
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.utils.hooks import RemovableHandle

from karume.quant_calib import (
    CalibReport,
    GridSpec,
    StageBatch,
    StageSpec,
    calibrate_stages,
)
from karume.quantize import DEFAULT_GROUP_SIZE, Int4Report, iter_quant_targets

from . import pipeline_ref
from .distribution import anima_model

#: In-model FQN prefix of the stage list. `patch.AnimaDit.model` / `patch.AnimaDitTokens.model` is
#: a `CosmosTransformer3DModel`, so a block's FQN is `model.transformer_blocks.<index>.…`.
STAGE_PREFIX = "model.transformer_blocks"

#: Method of the calibrated rounding. Methods other than GPTQ return no scale ledger (a MUST of
#: `karume.quant_calib`).
CALIB_METHOD = "gptq"

#: In-block FQN segments that denote adaLN (the layers that produce the modulation's shift / scale /
#: gate).
#:
#: Each `CosmosTransformerBlock` has the `CosmosAdaLayerNormZero` modules `norm1` / `norm2` /
#: `norm3`, each holding `linear_1` / `linear_2` (anima-v1.0 = 28 blocks × 3 × 2 = **168
#: weights**; confirmed by scanning the header of the real checkpoint). The sensitivity-experiment
#: variant (`--i4-adaln-i8` of `anima.export`) takes these 168 out of i4 and rounds them to i8 —
#: the same configuration restored the listening quality of irodori's w4 preset (research
#: `2026-08-24-gptq-expansion-quality.md` §1), and it was named as the candidate for improving the
#: visual verdict on plain i4 (§5 of the same = distribution skipped).
ADALN_SEGMENTS = frozenset({"norm1", "norm2", "norm3"})

#: Storage grid of the rounding targets. **`rtn` × g32 = the same grid as the existing i4 series**
#: (the only shipping path).
CALIB_GRID = GridSpec(kind="rtn", group_size=DEFAULT_GROUP_SIZE)

#: Calibration resolution (px, square). The same as the measurement conditions of run J-2
#: (`docs/research/2026-08-20-…` §6), and it **does not follow `--resolution`** — unless the
#: conditions under which the quality verdict (nearly identical to f32) was reached are fixed, every
#: change of resolution would mean re-establishing the evidence. In exchange for not following,
#: **the CLI rejects a mismatch** (`anima.export.main` — so that rounding targets chosen at 512px
#: are not baked into a graph of another resolution). The number of tokens per prompt = the number
#: of samples of GPTQ's `H = Σ XᵀX`, so raising it makes the calibration more effective but grows
#: the export's CPU time linearly (the axis for better quality next to the prompt count — the NOTE
#: of `calib_prompts`).
CALIB_RESOLUTION = 512

#: Rounding time per calibration batch (seconds, CPU). A **linear-extrapolation coefficient**: the
#: **measured distribution export of the same recipe** (512px, 4 prompts × 8 steps = 32 batches in
#: 2,287 seconds — `docs/research/2026-08-21-anima-i4-seat-speed.md` §6) divided by the batch
#: count. Used only to print a one-line estimate at the start (not as the basis for any decision
#: — it naturally drifts when the machine or the number of stages changes). It is smaller than the
#: J-2 measurement under other conditions (933 seconds for 10 batches = 93 seconds/batch), but that
#: one measured 1 prompt × 10 steps, a different condition.
CALIB_SECONDS_PER_BATCH = 72

#: The tokenizer's `max_length` (pipeline default — `--max-sequence-length` of `pipeline_ref`).
CALIB_MAX_SEQUENCE_LENGTH = 512

#: Storage dtype of the text front end when building the calibration inputs. **Matches the
#: distribution conditions** — the distributed `text_encoder` / `text_conditioner` are F16
#: (`anima.distribution.STORAGE_REQUIREMENTS`), and calibration is a mechanism that picks rounding
#: targets from "the activations that actually flow in the distribution", so the front end's
#: rounding is matched too (the same reason the weights outside the blocks are rounded with RTN
#: first). The J-2 measurement used an f32 front end, but the f16 / f32 difference of the front end
#: is judged too small to show in the statistics of H = Σ XᵀX (**no measurement of this judgement
#: is recorded in research** — measure again if a decision to revert or widen the conditions comes
#: up).
CALIB_TEXT_DTYPE = "f16"


@dataclass(frozen=True)
class CalibConditions:
    """The conditions of the reference denoise that **differ per model**.

    Derived by {@link calib_conditions}.

    They are not constants because turbo (8 steps, CFG 1) and the plain base models (20 steps,
    CFG 4) differ both in the sigma sequence the calibration sees and in the set of branches —
    baking in one would make the other ship rounding targets chosen under "conditions that the
    distribution never runs" (the storage form and the counts stay correct, so it cannot be read
    from the assets). Resolution, the text front end's dtype and the token length do not depend on
    the model, so they are not here and remain module constants.
    """

    #: Step count of the reference denoise (`defaults.steps` of the distribution).
    steps: int
    #: CFG scale (`defaults.guidanceScale` of the distribution).
    guidance: float
    #: Prompt fed to the uncond branch (`defaults.negativePrompt` of the distribution). Never used
    #: when `guidance == 1.0` ({@link pipeline_ref.reference_steps} does not compute the uncond
    #: branch).
    negative_prompt: str

    @property
    def branches(self) -> int:
        """Number of forwards through the DiT in one step (= calibration batches one step adds).

        MUST: the predicate has **the same form** as `skip_uncond` in
        {@link pipeline_ref.reference_steps} (`== 1.0`). If they drift, the check on the captured
        batch count disagrees with the real branch count and fails either way — rejecting a correct
        capture or passing a wrong one.
        """
        return 1 if self.guidance == 1.0 else 2


def calib_conditions(model: str) -> CalibConditions:
    """**Derives** the calibration conditions from the distribution defaults.

    The defaults are `pipeline_config` of `anima.distribution`.

    MUST: keep no separate copy. If the calibration conditions and the distribution defaults could
    be updated independently, the day the default step count moves only the calibration would run
    on the old sigma sequence, and the mismatch would show up nowhere (the storage form does not
    change by a single byte; the images just get slightly blurrier). For turbo this yields 8 steps
    / CFG 1, not a single bit different from when they were module constants.
    """
    defaults = anima_model(model).pipeline_config["defaults"]
    return CalibConditions(
        steps=int(defaults["steps"]),
        guidance=float(defaults["guidanceScale"]),
        negative_prompt=str(defaults["negativePrompt"]),
    )


def dit_stages(wrapper: nn.Module) -> tuple[StageSpec, ...]:
    """Returns the DiT blocks in execution order as `(in-model FQN prefix, block)`.

    The prefix includes the block index, so a stage-local FQN (`attn1.to_q.weight`) maps straight
    back to the in-wrapper FQN (`model.transformer_blocks.0.attn1.to_q.weight`).
    """
    return tuple(
        (f"{STAGE_PREFIX}.{index}", block)
        for index, block in enumerate(wrapper.model.transformer_blocks)
    )


def stage_linear_names(stages: Sequence[StageSpec]) -> frozenset[str]:
    """In-wrapper FQNs of the stage list's `nn.Linear` modules (**module names**, no `.weight`).

    Target selection shares core's `iter_quant_targets` — if "what the calibration rounds" and
    "what the caller counts as inside a stage" were separate copied implementations, weights that
    belong to neither would silently fall back to i8.
    """
    return frozenset(
        f"{prefix}.{local}".removesuffix(".weight")
        for prefix, stage in stages
        for local, _weight, _axis in iter_quant_targets(stage, (nn.Linear,))
    )


def is_adaln(name: str) -> bool:
    """Whether a module FQN is under adaLN (contains {@link ADALN_SEGMENTS} **as a segment**).

    The test matches the elements split on `.` because a substring match invites mix-ups such as
    being pulled by the spelling of `norm_out`, while a suffix match would miss the children under
    adaLN (`norm1.linear_1`). It gives the same answer for a stage-local FQN (`norm1.linear_1`)
    and for an in-wrapper FQN (`model.transformer_blocks.0.norm1.linear_1`), so the calibration's
    `include` and the set split on the export side are decided by one implementation (a copy would
    split the set rounded to i4 from the set stored as i4).

    NOTE: use the test **only inside the blocks**. The DiT also has `norm_out` outside the blocks
    (`CosmosAdaLayerNorm` = the same modulation role), but that one is named explicitly by
    {@link anima.export.NON_STAGE_I4_WEIGHTS}, and its segment is spelled differently too.
    """
    return not ADALN_SEGMENTS.isdisjoint(name.split("."))


def adaln_segments_seen(names: Iterable[str]) -> frozenset[str]:
    """The spellings of {@link ADALN_SEGMENTS} that appeared **as segments** in `names`.

    Where {@link is_adaln} returns "did any one match", this returns **which ones matched**. Both
    are counted because the 168 weights moved back to i8 are the **union** of 3 segments — if
    upstream renames only some of them (e.g. `norm3`), `is_adaln` keeps returning true on the
    remaining spellings, so a check that only looks for "not a single adaLN weight" passes, and the
    count check builds both sides from the same classifier and stays green while self-consistent
    (= only the 56 weights that were meant to be excluded silently go back to i4).
    """
    seen: set[str] = set()
    for name in names:
        seen.update(ADALN_SEGMENTS.intersection(name.split(".")))
    return frozenset(seen)


def _on_cpu(value: Any) -> Any:
    """Moves a captured auxiliary argument to the CPU (Tensors and **plain** tuple / list items).

    It looks inside the tuples because a block receives rope as a `(cos, sin)` pair — passing them
    through untouched would leave part of the calibration batches in VRAM with
    `--calib-device cuda`. The reverse direction (the device move right before running a stage) is
    held in the same form by core's `karume.quant_calib`. On the CPU path `Tensor.cpu()` returns
    self, so the default capture does not change by a single bit.
    """
    if isinstance(value, torch.Tensor):
        return value.cpu()
    if type(value) is tuple:
        return tuple(_on_cpu(item) for item in value)
    if type(value) is list:
        return [_on_cpu(item) for item in value]
    return value


def _block_kwargs(
    names: Sequence[str], args: Sequence[Any], kwargs: dict[str, Any]
) -> dict[str, Any]:
    """Folds a block's positional arguments into keywords by **its signature's names** (no hidden).

    The names are not held as a copied list so that the spellings cannot silently swap when
    upstream reorders the arguments (the same approach as
    `anima.measure_quant.capture_stage_batches`).
    """
    return {**dict(zip(names[1:], args[1:], strict=False)), **kwargs}


def assert_stage_split(
    wrapper: nn.Module, probe: Sequence[torch.Tensor], stages: Sequence[StageSpec]
) -> StageBatch:
    """Checks that the stage-by-stage block loop is **bit-identical to one forward of the wrapper**.

    The wrapper's output has gone on through unpatchify (static form) or `proj_out` (S form) and is
    not the output of the block list itself, so the comparison target is **the input to
    `norm_out`**, captured with a forward_pre_hook. This pins down by measurement, before any
    rounding, that "how the wrapper calls the blocks" and "how the calibration calls the blocks"
    are the same.

    MUST: compare with `torch.equal`, not an approximate tolerance — the same modules are called in
    the same order, so a mismatch means the split goes through a path different from the real one.

    Returns the `(args, kwargs)` the wrapper passed to the first block (the reference for {@link
    assert_calib_batches_match_graph}).
    """
    model = wrapper.model
    blocks = model.transformer_blocks
    names = tuple(inspect.signature(blocks[0].forward).parameters)
    captured: list[StageBatch] = []
    tail: list[torch.Tensor] = []

    def catch_block(_module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        captured.append(((args[0].detach(),), _block_kwargs(names, args, kwargs)))

    def catch_tail(_module: nn.Module, args: tuple[Any, ...]) -> None:
        tail.append(args[0].detach())

    handles = [
        blocks[0].register_forward_pre_hook(catch_block, with_kwargs=True),
        model.norm_out.register_forward_pre_hook(catch_tail),
    ]
    try:
        with torch.no_grad():
            wrapper(*probe)
    finally:
        for handle in handles:
            handle.remove()
    if len(captured) != 1 or len(tail) != 1:
        raise AssertionError(
            f"One forward of the wrapper hit the first block {len(captured)} times / norm_out"
            f" {len(tail)} times (both are expected once — the wrapper's code disagrees with what"
            " the script assumes)"
        )
    args, kwargs = captured[0]
    with torch.no_grad():
        hidden = args[0]
        for _prefix, stage in stages:
            hidden = stage(hidden, **kwargs)
    if not torch.equal(hidden, tail[0]):
        raise AssertionError(
            "The final hidden of the stage split is not bit-identical to the output of the"
            " wrapper's block loop"
            f" (max abs difference {float((hidden - tail[0]).abs().max()):.4e})"
            " — the stage list has drifted from the wrapper's path"
        )
    return captured[0]


def capture_stage_batches(
    model: nn.Module,
    prompts: Sequence[str],
    *,
    repo: str,
    conditions: CalibConditions,
    resolution: int = CALIB_RESOLUTION,
) -> tuple[StageBatch, ...]:
    """Runs the reference denoise per prompt and catches the first block's input across all steps.

    The path reuses the reference fixture (`anima.pipeline_ref`) — the text front end
    ({@link pipeline_ref.load_text_stack} / {@link pipeline_ref.encode_prompt}), the sigma sequence
    and `reference_steps` are used as they are. No copy is made so that "the activations the
    calibration sees" and "the activations that flow in the distribution" stay one implementation.
    Only `timesteps_proj_table` is not used — it is an entry point that **loads a second copy of
    the DiT**, and the instance to be rounded is already here (`reference_steps` builds the
    timestep embedding inside the plain forward).

    MUST: **with CFG > 1, capture both the cond and the uncond branches**. The plain base models
    run with CFG, so the activations that pass through the DiT at distribution time include as many
    from the negative prompt side as from cond — restricting to cond would pick the rounding
    targets from "half of the activations that actually flow". `reference_steps` runs the forward
    twice per step when `guidance != 1.0`, so the first block's forward_pre_hook captures **both as
    they are** (the branches are not assembled here). The count check also multiplies by
    {@link CalibConditions.branches}.

    MUST be deterministic: fixed seed (`pipeline_ref.SEED`), and the only random draw is
    `latents_init` here (the CFG branches have no randomness). The initial noise is **shared by all
    prompts** (the conditions of run J-2 widened as they are — the axes that vary are limited to
    the prompt and the resolution).

    MUST: the caller calls this **before a single in-block rounding is applied** (afterwards, the
    rounding targets of a weight would be chosen from activations produced by that same rounded
    weight — a loop).

    NOTE (device): the reference denoise runs on the device `model` lives on (on the
    `--calib-device cuda` sensitivity-experiment path the DiT is on the GPU). **The text front end
    stays on the CPU** — Qwen3 + the conditioner cannot share VRAM with the f32 DiT, so only
    `embeds` / `negative` / `latents_init` move to the GPU once the conditions are built. The
    initial noise is **generated on the CPU and then moved** (`torch.randn`'s random sequence
    differs per device, so generating on the device would split the bits from the CPU path).
    Captured batches are always moved to the CPU and kept there (the CPU is the only place for
    calibration batches — transport when a stage runs is handled by the JIT move in
    `karume.quant_calib`).
    """
    if not prompts:
        raise AssertionError("No calibration prompts (calibrated i4 cannot work with zero inputs)")
    stack = pipeline_ref.load_text_stack(repo, CALIB_TEXT_DTYPE)
    embeds = [
        pipeline_ref.encode_prompt(stack, CALIB_MAX_SEQUENCE_LENGTH, prompt)[
            "encoder_hidden_states"
        ]
        for prompt in prompts
    ]
    # The uncond branch's prompt is never used with CFG=1, but it is always prepared without a
    # branch — the "use / do not use" decision stays in one place, the MUST of `reference_steps`
    # (holding it in two places means that the day one side changes how it treats CFG, uncond runs
    # with the same embedding as cond, and the same activations are added to H twice while only the
    # batch count stays correct).
    negative = pipeline_ref.encode_prompt(
        stack, CALIB_MAX_SEQUENCE_LENGTH, conditions.negative_prompt
    )["encoder_hidden_states"]
    # Release the text front end (Qwen3 + conditioner) once the conditions are built — do not
    # overlap it with the resident DiT.
    del stack
    gc.collect()

    sigmas = pipeline_ref.sigma_schedule(conditions.steps, pipeline_ref.SHIFT)
    latent = resolution // pipeline_ref.SPATIAL_COMPRESSION
    latents_init = torch.randn(
        (1, pipeline_ref.LATENT_CHANNELS, 1, latent, latent),
        generator=torch.Generator().manual_seed(pipeline_ref.SEED),
        dtype=torch.float32,
    )
    device = next(model.parameters()).device
    embeds = [embed.to(device) for embed in embeds]
    negative = negative.to(device)
    latents_init = latents_init.to(device)
    blocks = model.transformer_blocks
    names = tuple(inspect.signature(blocks[0].forward).parameters)
    batches: list[StageBatch] = []

    def catch(_module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        # Captures are moved to the CPU as they accumulate (batch count × hidden does not stay
        # resident in VRAM). On the CPU path `Tensor.cpu()` returns self, so the default capture
        # does not change by a single bit.
        captured = _block_kwargs(names, args, kwargs)
        batches.append(
            (
                (args[0].detach().cpu(),),
                {name: _on_cpu(value) for name, value in captured.items()},
            )
        )

    handle = blocks[0].register_forward_pre_hook(catch, with_kwargs=True)
    try:
        for index, embed in enumerate(embeds):
            print(f"[calib] reference denoise for prompt {index + 1}/{len(embeds)}", flush=True)
            pipeline_ref.reference_steps(
                model,
                latents_init,
                embed,
                negative,
                sigmas,
                (resolution, resolution),
                conditions.steps,
                conditions.guidance,
            )
    finally:
        handle.remove()
    expected = len(prompts) * conditions.steps * conditions.branches
    if len(batches) != expected:
        raise AssertionError(
            f"Got {len(batches)} calibration batches, not prompts {len(prompts)}"
            f" × steps {conditions.steps} × branches {conditions.branches}"
            f" (CFG {conditions.guidance}) = {expected}"
            " (the DiT's block loop disagrees with what the script assumes)"
        )
    return tuple(batches)


def assert_calib_batches_match_graph(
    graph_batch: StageBatch, batches: Sequence[StageBatch]
) -> None:
    """Checks calibration inputs reach the blocks with **the graph's set of auxiliary arguments**.

    The capture comes from the forward of the bare `CosmosTransformer3DModel`, but the weights
    that are rounded belong to the exported wrapper. If the sets of keywords the two pass to a
    block differ, we ship "rounding targets chosen under conditions other than the distributed
    graph's" — and the numbers come out normally (the golden stays green too).

    Only **the set of keyword names** is checked. Values, shapes and positional arguments are not
    compared, so an input with the same set but different contents (e.g. a temb taken at another
    sigma) passes — what this check claims is "the set of arguments is the same", not "the path is
    the same".

    MUST: fail loudly. If this stays silent, then on the day upstream adds an argument to the
    block, a state where only the calibration sees it (or does not) slips in, unreadable from the
    assets.
    """
    wanted = set(graph_batch[1])
    for index, (_args, kwargs) in enumerate(batches):
        if set(kwargs) != wanted:
            raise AssertionError(
                f"Calibration batch {index}: the block auxiliary arguments {sorted(kwargs)}"
                f" differ from the graph's {sorted(wanted)} — the reference denoise and the"
                " wrapper call the blocks differently"
            )


@dataclass(frozen=True)
class CalibRig:
    """The scaffolding for one calibration (the stage list and the inputs to the first stage)."""

    stages: tuple[StageSpec, ...]
    batches: tuple[StageBatch, ...]

    @property
    def tokens(self) -> int:
        """Total hidden tokens fed to the calibration.

        A diagnostic that records a reduced run next to the numbers.
        """
        return sum(int(args[0].shape[-2]) for args, _kwargs in self.batches)


def _announce_stages(stages: Sequence[StageSpec]) -> list[RemovableHandle]:
    """Hooks that print one progress line on each stage's **first** forward (returns the handles).

    The returned handles are for removing the hooks.

    `karume.quant_calib` prints nothing, so a calibration in the default setup is silent for hours
    from the start line to the final diagnostic line and cannot be told apart from a hang. core is
    a boundary (ADR 0065), so it is left untouched and observed from the caller. Only the first
    call is reported because a stage is called "observation × batch count + advance × batch count"
    times — the number of lines printed equals the number of stages, and the wait is divided at
    that granularity.
    """
    total = len(stages)

    def announce(index: int):
        seen = False

        def hook(_module: nn.Module, _args: tuple[Any, ...]) -> None:
            nonlocal seen
            if seen:
                return
            seen = True
            print(f"[calib] stage {index + 1}/{total}", flush=True)

        return hook

    return [
        stage.register_forward_pre_hook(announce(index))
        for index, (_prefix, stage) in enumerate(stages)
    ]


def calibrate_i4(
    rig: CalibRig, include: Callable[[str], bool] | None = None
) -> tuple[CalibReport, Int4Report]:
    """Applies stage-by-stage GPTQ and returns `(report, scale ledger)` (the rounding is in place).

    `include` is a predicate on **stage-local module FQNs** and is passed straight to core's
    `calibrate_stages` (`None` = every `nn.Linear` in the stages = the default distribution path).
    Only the sensitivity-experiment variant (`--i4-adaln-i8` of `anima.export`) passes a
    predicate, taking the 168 adaLN weights out of GPTQ and sending them to the i8 side.

    MUST: the set the predicate selects and the set stored as i4 are **identical** — if they
    drift, the check "scanned count = rounded count" loses its meaning and weights that were never
    rounded get an i4 storage spec. The measurement is {@link assert_calib_covers_scan} (the caller
    passes the set stored as i4 to compare against).

    MUST: a report without a ledger fails loudly — only `gptq` × `rtn` has a shipping path (a MUST
    of `karume.quant_calib`), and going on without a ledger puts weights without a scale into the
    i4 preset.
    """
    # The device is read from the stages (the same source core uses to derive the compute device —
    # building only the printout from another source could make the diagnostic line disagree with
    # where it actually ran). The estimate's coefficient is a CPU measurement, so on a
    # `device=cuda` line the order of magnitude is off (that is a path for sensitivity experiments
    # only).
    device = next(rig.stages[0][1].parameters()).device
    print(
        f"[calib] starting GPTQ calibration (device={device.type})"
        f" — {len(rig.stages)} stages × {len(rig.batches)} batches"
        f" (estimate about {len(rig.batches) * CALIB_SECONDS_PER_BATCH / 60:.0f} minutes"
        " — linear extrapolation from the J-2 measurement)",
        flush=True,
    )
    handles = _announce_stages(rig.stages)
    try:
        report = calibrate_stages(
            rig.stages, rig.batches, method=CALIB_METHOD, spec=CALIB_GRID, include=include
        )
    finally:
        for handle in handles:
            handle.remove()
    ledger = report.int4
    if ledger is None:
        raise AssertionError(
            f"The calibration returned no scale ledger ({report.describe()})"
            " — only gptq × rtn can be shipped (a MUST of karume.quant_calib)"
        )
    return report, ledger


def assert_calib_covers_scan(report: CalibReport, scan: frozenset[str]) -> None:
    """Checks the layers the calibration rounded match **the in-block set stored as i4** exactly.

    `scan` is a set of module names without `.weight` (the form of {@link stage_linear_names}; by
    default the stages' scan itself, and in the sensitivity-experiment variant the side without
    adaLN = identical to the set the `include` of {@link calibrate_i4} selects).

    MUST: fail loudly. A missed rounding moves quality to the **better side**, so passing silently
    cannot be read from the numbers. On top of that, the missed weights end up on neither the i8
    side nor the i4 side, while only the storage spec demands i4.

    What it catches is the case where "**the actual calibration run drifted from the scan**" (core
    dropped a target / other `stages` were passed / the ledger extraction dropped a layer). When the
    scan itself changes (the target types of `iter_quant_targets`, the stage split), **both sides
    move together and this check stays silent** — that is covered by the comparison "scan == i4
    eligible set" on the `export.py` side (an independent source derived from the graph).
    """
    rounded = {layer.fqn for layer in report.layers}
    wanted = {f"{name}.weight" for name in scan}
    missing = sorted(wanted - rounded)
    extra = sorted(rounded - wanted)
    if missing or extra or report.modules != len(wanted):
        raise AssertionError(
            f"The {report.modules} layers the calibration rounded do not match the {len(wanted)}"
            f" of the scan (not rounded {missing[:3]} / not in the scan {extra[:3]})"
        )
