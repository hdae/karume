"""Rig that runs the **calibrated rounding** (GPTQ × RTN grid) of Irodori's i4 series
(perf-ledger Q-2 / Q-3).

`karume.quant_calib.calibrate_stages` is a driver that "advances one stage at a time and re-picks
the rounding targets from the activations that actually flow into that layer", and its storage grid
is **byte-for-byte the same** as {@link karume.quantize.quantize_to_int4}
(`scale = group amax / 7`, `q ∈ [−7,+7]`). Only the rounded values and the contents of the scale
ledger change; neither the storage path on the emit side nor the byte structure of the
distribution moves at all.

This module provides the three things the core driver requires, plus two Irodori-specific checks:

1. **The stage sequence** — the DiT's `blocks` laid out as they are ({@link dit_stages}). Upstream
   `DiffusionBlock.forward` already "takes hidden as the first positional argument and returns
   hidden", so it is **not wrapped**. The prefix {@link STAGE_PREFIX} is `blocks`, which is
   **also the FQN space of the distribution wrappers (`irodori.export.DitGraph` /
   `DitContextGraph`)** — the wrappers hold the DiT's subtrees under **the same attribute names**
   `blocks.<b>.…` / `in_proj` / `out_proj` / `cond_module` (which is why the two DiT rows of
   `irodori.export.TARGET_SCALE_SOURCES` need no remapping). The weights of one block are split
   across the two wrappers (the condition-side K/V projections in `dit-context`, the rest in
   `dit` — ADR 0114), but a stage stays a whole block (the calibration driver is the same as
   before the split). If this drifts, the scale ledger's keys miss the containers' initializer
   names (= the keys of the binding table).
2. **The input to the first stage** — {@link capture_stage_batches} **actually runs the reference
   denoise** and captures the `(args, kwargs)` of the first block with a forward_pre_hook. It does
   not rebuild them itself, for the same reason as the measurement rig (`irodori.measure_quant`);
   in a diffusion model "the activations also move with t", so they are taken **across steps**.
3. **The stage-split identity check** — {@link assert_stage_split}. It checks that "the result of
   running the stages one by one is bit-identical to the hidden that entered `out_norm` in the same
   forward" (a MUST measured before a single rounding is applied).
4. **The coverage check** — {@link assert_calib_covers_scan}. The layers the calibration rounded =
   the set stored as i4 (the 168 found by scanning the stages, minus adaLN —
   {@link ADALN_SEGMENTS}).

MUST (why the driver runs **the bare DiT** — a constraint specific to this recipe): the rounding
of `irodori.export` happens "right after `load_*`, before the pre-patch reference" (the ordering
MUST of `irodori.export.fake_quant`), while the distribution wrappers (`DitContextGraph` /
`DitGraph`) **only work after patching** (they pass a real-valued RoPE table — the module docstring
of `irodori.patch`). Both cannot hold at once, so the calibration's reference loop is taken on
**upstream's canonical path** (`irodori_tts.rf.sample_euler_rf_cfg` running the bare
`TextToLatentRFDiT`). That the bare path and the composition of the two distribution wrappers give
the same values is measured **within the same export run** on every golden case by the eager
equivalence check of `irodori.export` (`EAGER_EQUIV_ATOL = 0`, including the equivalence check of
the split) — together with "stages one by one ≡ the bare block loop" (the check here), two
measurements close the claim that the stage split matches the path of the distributed graphs.

MUST: the reference loop runs with `use_context_kv_cache=False` ({@link capture_stage_batches}) —
upstream's default (True) passes the condition K/V in **a separate cache per block**, which does
not fit the core driver that reuses one kwargs for every stage, and it breaks silently before the
stage-split identity check. The distributed graphs build the condition-side K/V once in a separate
graph (`dit-context`) (ADR 0114 — the same semantics as upstream's cache), but the values are the
output of the same `project_context_kv`, so they are bit-identical to the activations taken with
False. Running the condition-side projections **inside** the stage is also what lets those 72
modules see their own inputs in GPTQ.

NOTE (why this is not shared with the rig of `irodori.measure_quant`): that one is scaffolding for
3 grids × quality measurement, and its driver is confined to **the patched distribution wrappers**
(it keeps a copy called `DitBlockStage`). The export side needs the bare pre-patch path, the
distribution conditions (round what is outside the stages first) and extraction of a shippable
ledger. Merging the copies into one would let one side silently follow when the other side's
premise moves — instead each rig verifies itself independently with its own identity check.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

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
from karume.quantize import DEFAULT_GROUP_SIZE, Int4Report, channel_rows, iter_quant_targets

from . import export as ex
from . import pipeline_ref as ip

#: FQN prefix of the stage sequence (`TextToLatentRFDiT.blocks` — the same as the attribute name
#: in the distribution wrappers).
STAGE_PREFIX = "blocks"

#: Method of the calibrated rounding. Methods other than GPTQ return no scale ledger (the MUST of
#: `karume.quant_calib`).
CALIB_METHOD = "gptq"

#: Storage grid of the rounding targets. **`rtn` × g32 = the existing i4 grid itself** (the only
#: shipping path), the same setting as `gptq-rtn` in `irodori.measure_quant.CALIB_CONFIGS`.
CALIB_GRID = GridSpec(kind="rtn", group_size=DEFAULT_GROUP_SIZE)

#: Default number of steps captured for calibration (per case). **The same as the measurement
#: default of iteration J-2** (the `--calib-steps` default of `irodori.measure_quant` = the full
#: reference loop) — so as not to fall below the conditions the quality ruling was made under.
#: Lowering it with `--calib-steps` is for smoke runs, and the value used is recorded in
#: `calib_provenance.json`.
CALIB_STEPS = ip.NUM_STEPS

#: The arguments of `DiffusionBlock.forward` that **have a batch axis** (the cond side = row 0 is
#: sliced out). Upstream's independent CFG runs cond and uncond **batched into one forward** (it
#: concatenates `cfg_batch_mult` copies of `x_t`), so only row 0 of the captured batch is the cond
#: side.
BATCHED_BLOCK_ARGS = frozenset(
    {
        "x",
        "cond_embed",
        "text_state",
        "text_mask",
        "speaker_state",
        "speaker_mask",
        "caption_state",
        "caption_mask",
    }
)

#: The arguments of the same forward without a batch axis (passed on to the next stage as they
#: are). `context_kv` is always `None` by the MUST of {@link capture_stage_batches}.
UNBATCHED_BLOCK_ARGS = frozenset({"freqs_cis", "self_mask", "context_kv"})

#: Name of the argument that carries hidden (the core driver passes hidden as a **positional
#: argument**, so it is removed from kwargs).
HIDDEN_ARG = "x"

#: FQN segments that point at adaLN (the layers that produce the modulation's scale / shift /
#: gate).
#:
#: The i8+dit4 slot takes the 144 in-block modules found here (12 blocks × 12) out of i4 and
#: **rounds them to i8** — listening ruling of 2026-08-23: the modulation's scale / shift / gate
#: are highly sensitive to quantization, and moving them back to i8 brings the delivery closer to
#: f32 (cost +13.1 MiB = dit i4 payload +6.0%).
ADALN_SEGMENTS = frozenset({"attention_adaln", "mlp_adaln"})


class _CalibStepsReached(Exception):  # noqa: N818 — a stop signal, not an error
    """Signal that the required number of steps has been captured (a sentinel that cuts the
    reference loop short)."""


@dataclass(frozen=True)
class CalibContext:
    """Everything "other than the weights" that calibration needs (built and passed in by
    `irodori.export.export_series`).

    The tokenizers and the conditions are built by calling the functions of `irodori.pipeline_ref`
    as they are — a copy would start producing "the activations seen in calibration" and "the
    activations that flow in evaluation" through different preprocessing.
    """

    model_dir: Path
    source: Any
    config: Any
    text_config: Mapping[str, Any]
    model_config: Mapping[str, Any]

    @property
    def text_cap(self) -> int:
        return int(self.model_config["max_text_len"])

    @property
    def caption_cap(self) -> int:
        return int(self.model_config["max_caption_len"])


class CapturedRun(NamedTuple):
    """The capture of one case.

    `batches` is the calibration input sliced to the cond side (row 0), `graph_batch` is **the raw
    first forward** (batch axis as is), and `block_loop_output` is the hidden that entered
    `out_norm` in **the same forward**. The last two are the two sides of the comparison in
    {@link assert_stage_split}; both come from the same forward, so there is no room for "comparing
    different conditions".
    """

    batches: tuple[StageBatch, ...]
    graph_batch: StageBatch
    block_loop_output: torch.Tensor


def dit_stages(dit: nn.Module) -> tuple[StageSpec, ...]:
    """Return the DiT blocks in execution order as `(in-model FQN prefix, block)`.

    The prefix includes the block index, so a local FQN within a stage (`attention.wq.weight`)
    maps straight back to the in-model FQN (`blocks.0.attention.wq.weight`).
    """
    return tuple((f"{STAGE_PREFIX}.{index}", block) for index, block in enumerate(dit.blocks))


def stage_linear_names(stages: Sequence[StageSpec]) -> frozenset[str]:
    """FQNs of the `nn.Linear` modules the stage sequence holds (**module names**, with
    `.weight` dropped).

    Target selection shares core's `iter_quant_targets` — copying "what calibration rounds" and
    "what the caller counts as inside the stages" into separate implementations would leave weights
    that fall into neither silently in f32.
    """
    return frozenset(
        f"{prefix}.{local}".removesuffix(".weight")
        for prefix, stage in stages
        for local, _weight, _axis in iter_quant_targets(stage, (nn.Linear,))
    )


def is_adaln(name: str) -> bool:
    """Whether a module FQN is under adaLN (whether it contains {@link ADALN_SEGMENTS} **as a
    segment**).

    The test matches the elements split at `.` because a substring match would let `attention`
    catch `attention_adaln`, and a suffix match would miss the children under adaLN
    (`attention_adaln.1`). It gives the same answer for a local FQN within a stage
    (`attention_adaln.1`) and for the in-model FQN (`blocks.0.attention_adaln.1`), so the
    calibration's `include` and the export side's set split are decided by one implementation (a
    copy would split the set rounded to i4 from the set stored as i4).
    """
    return not ADALN_SEGMENTS.isdisjoint(name.split("."))


def adaln_segments_seen(names: Iterable[str]) -> frozenset[str]:
    """The spellings of {@link ADALN_SEGMENTS} that appear **as segments** in `names`.

    Where {@link is_adaln} returns "did any of them match", this returns **which ones matched**.
    Both are counted because the 144 modules moved back to i8 are decided by the **union** of the
    two segments — if upstream renamed only one side (e.g. `mlp_adaln`), `is_adaln` would keep
    returning true on the remaining side, so the "no adaLN at all" check would pass, and the i4
    count check, which builds both sides from the same classifier, would stay self-consistent and
    green (= a configuration rejected by the listening ruling would ship silently).
    """
    seen: set[str] = set()
    for name in names:
        seen.update(ADALN_SEGMENTS.intersection(name.split(".")))
    return frozenset(seen)


def dit_i4_names(dit: nn.Module, sym_max: int) -> frozenset[str]:
    """Module FQNs of the DiT that are i4-eligible (= every `nn.Linear` in the distributed
    graphs).

    NOTE: what the i4 series actually **stores as i4 is only the 168 in-block modules minus adaLN**
    (the 5 outside the blocks and the 144 adaLN modules are stored as i8 — the attribution result of
    the listening ruling of 2026-08-23; {@link ADALN_SEGMENTS} / `irodori.export._fake_quant_i4`).
    This returns "all the linears in the distributed graphs", those included, and the g32
    divisibility condition below applies to the 149 that go to i8 as well — stricter than storage
    requires, but it is **not relaxed**: relaxing it would let a configuration change in which
    "one of the modules sent to i8 became non-divisible by g32" pass silently, and the contents of
    the i4 slot would be decided by upstream's eligibility rate on every run.

    Eligibility is the conjunction of two conditions:

    1. **The weight is in the distributed graphs** — decided by intersecting with the **union** of
       the `named_parameters` of the DiT's two distribution wrappers
       (`irodori.export.dit_wrappers` — `dit-context` / `dit`) (the same set as owned by the single
       `DitGraph` before the split — ADR 0114). The bare `TextToLatentRFDiT` **holds copies of**
       backbone / projector / speaker / duration **inside it** (the NOTE of
       `irodori.pipeline_ref`), so without filtering by name, linears of other roles would turn
       into i4 too. There is no explicit list (`blocks` / `in_proj` / `out_proj` / `cond_module`)
       so that, on the day upstream changes the DiT's structure, a stale list does not pass.
    2. **The quantization axis is divisible by the group length** — i4 MUST NOT create a partial
       group (ADR 0069 decision 2). In the real weights (12 blocks) all 317 linears are divisible
       by g32 (the k histogram of `docs/research/2026-08-12-irodori-quant-recon.md`), but this is
       **not assumed; it fails loudly** — calibration drives whole stages, so there is no way to
       leave out a single module, and silently dropping that one to i8 would make the "scanned
       count = rounded count" check impossible.

    MUST: zero eligible modules fail loudly (do not silence "`--dtype i4` was requested but there
    is no i4 at all" — the same discipline as ADR 0006).
    """
    owned = {
        name.removesuffix(".weight")
        for wrapper in ex.dit_wrappers(dit, sym_max).by_target().values()
        for name, _parameter in wrapper.named_parameters()
    }
    names: set[str] = set()
    for fqn, weight, axis in iter_quant_targets(dit, (nn.Linear,)):
        name = fqn.removesuffix(".weight")
        if name not in owned:
            continue
        length = int(channel_rows(weight, axis).shape[-1])
        if length % DEFAULT_GROUP_SIZE:
            raise AssertionError(
                f"{fqn}: quantization axis {length} is not divisible by g{DEFAULT_GROUP_SIZE}"
                " (calibration is driven per stage, so there is no way to leave out a single"
                " module)"
            )
        names.add(name)
    if not names:
        raise AssertionError(
            "the DiT has no i4-eligible linear at all"
            " (the structure of the DiT wrappers disagrees with what the script assumes)"
        )
    return frozenset(names)


def _split_block_kwargs(kwargs: Mapping[str, Any], batch_rows: int) -> StageBatch:
    """Split the block's call arguments into `(one hidden, the remaining kwargs)` and slice out
    the cond-side rows.

    MUST: pin the set of arguments as observed (fail loudly) — if upstream added or removed
    arguments of `DiffusionBlock.forward`, the batch-axis classification would silently go wrong
    and ship "rounding targets chosen under different conditions" (the numbers come out normally,
    so the golden stays green too).

    `batch_rows` is the number of rows kept as the cond side (**always 1** — upstream's independent
    CFG batches `[cond, uncond…]` into one forward, with cond in row 0).
    """
    wanted = BATCHED_BLOCK_ARGS | UNBATCHED_BLOCK_ARGS
    if set(kwargs) != wanted:
        raise AssertionError(
            f"the block call arguments are {sorted(kwargs)}, not the expected {sorted(wanted)}"
            " (the spelling of upstream DiffusionBlock.forward has moved)"
        )
    if kwargs["context_kv"] is not None:
        raise AssertionError(
            "a condition K/V cache is being passed to the block"
            " (the reference loop MUST run with use_context_kv_cache=False — otherwise one"
            " kwargs would reuse a cache that is separate per block)"
        )
    sliced = {
        name: (value[:batch_rows] if name in BATCHED_BLOCK_ARGS and value is not None else value)
        for name, value in kwargs.items()
    }
    hidden = sliced.pop(HIDDEN_ARG)
    return ((hidden.detach(),), {name: _detached(value) for name, value in sliced.items()})


def _detached(value: Any) -> Any:
    return value.detach() if isinstance(value, torch.Tensor) else value


def _reference_latent(case: ip.PipelineCase, config: Any) -> torch.Tensor:
    """The reference latent (built the same way as ③ of `irodori.pipeline_ref.run_case`)."""
    if case.reference is None:
        return torch.zeros((1, int(config.speaker_patch_size), int(config.latent_dim)))
    generator = torch.Generator().manual_seed(case.reference.seed)
    return torch.randn(1, case.reference.frames, int(config.latent_dim), generator=generator)


def _load_tokenizers(context: CalibContext) -> tuple[Any, Any]:
    """The text-side and caption-side tokenizers (the same source as `irodori.pipeline_ref`)."""
    from tokenizers import Tokenizer

    return (
        Tokenizer.from_file(str(context.model_dir / ex.TOKENIZER_FILE)),
        ip.upstream_caption_tokenizer(
            context.model_dir, bool(context.config.caption_add_bos_resolved)
        ),
    )


def _capture_case(
    dit: nn.Module,
    case: ip.PipelineCase,
    context: CalibContext,
    tokenizers: tuple[Any, Any],
    steps: int,
    label: str,
) -> CapturedRun:
    """Run the reference denoise of one case and capture **one batch per step** (cond side).

    Upstream's independent CFG is 1 step = **1 forward** (cond and uncond batched together), so
    the number of calls to the first block is the step count. Taking only row 0 is what "the cond
    side's forward only" means, the same capture semantics as the measurement of iteration J-2
    (`irodori.measure_quant.capture_case_batches`).

    Once `steps` have been captured, a sentinel cuts the reference loop short — the upstream loop
    would run the remaining steps too, so without cutting it a reduced run would not work.
    """
    from irodori_tts.rf import sample_euler_rf_cfg

    tokenizer, caption_tokenizer = tokenizers
    bos_id = int(context.text_config["bos_token_id"])
    pad_id = int(context.text_config["pad_token_id"])
    text_ids = ip._packed_ids(
        tokenizer, case.text, bos_id, context.text_cap, context.source.normalize_text
    )
    caption_padded, caption_mask = ip.upstream_caption_condition(
        caption_tokenizer, case.caption, context.caption_cap
    )
    reference_latent = _reference_latent(case, context.config)
    has_speaker = case.reference is not None
    reference_mask = torch.full(reference_latent.shape[:2], has_speaker, dtype=torch.bool)
    sequence = ip.upstream_sequence_length(
        dit,
        text_ids,
        caption_padded,
        caption_mask,
        reference_latent,
        has_speaker,
        context.text_config,
        context.model_config,
        context.config,
    )
    text_padded, text_mask = ip._right_pad_ids(text_ids, context.text_cap, pad_id)

    batches: list[StageBatch] = []
    graph_batch: list[StageBatch] = []
    tail: list[torch.Tensor] = []

    def on_block(_module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        if args:
            raise AssertionError(
                f"the first block was called with {len(args)} positional arguments"
                " (upstream forward_with_encoded_conditions passes everything as keywords)"
            )
        if len(batches) >= steps:
            raise _CalibStepsReached
        if not graph_batch:
            graph_batch.append(
                (
                    (kwargs[HIDDEN_ARG].detach(),),
                    {
                        name: _detached(value)
                        for name, value in kwargs.items()
                        if name != HIDDEN_ARG
                    },
                )
            )
        batches.append(_split_block_kwargs(kwargs, 1))

    def on_tail(_module: nn.Module, args: tuple[Any, ...]) -> None:
        if not tail:
            tail.append(args[0].detach())

    handles = [
        dit.blocks[0].register_forward_pre_hook(on_block, with_kwargs=True),
        dit.out_norm.register_forward_pre_hook(on_tail),
    ]
    try:
        with torch.no_grad():
            sample_euler_rf_cfg(
                model=dit,
                text_input_ids=text_padded,
                text_mask=text_mask,
                ref_latent=reference_latent,
                ref_mask=reference_mask,
                sequence_length=sequence,
                caption_input_ids=caption_padded,
                caption_mask=caption_mask,
                num_steps=ip.NUM_STEPS,
                cfg_scale_text=ip.CFG_SCALES["text"],
                cfg_scale_caption=ip.CFG_SCALES["caption"],
                cfg_scale_speaker=ip.CFG_SCALES["speaker"] if has_speaker else 0.0,
                cfg_guidance_mode="independent",
                cfg_min_t=ip.CFG_MIN_T,
                cfg_max_t=ip.CFG_MAX_T,
                seed=case.seed,
                use_context_kv_cache=False,
            )
    except _CalibStepsReached:
        pass
    finally:
        for handle in handles:
            handle.remove()
    if len(batches) != steps or not tail:
        raise AssertionError(
            f"{case.name}: captured calibration input for only {len(batches)} steps"
            f" (requested {steps} steps / block loop tails {len(tail)})"
            " — the spelling of upstream sample_euler_rf_cfg disagrees with what the script"
            " assumes"
        )
    print(f"[{label}] {case.name}: S={sequence} / captured {len(batches)} steps", flush=True)
    return CapturedRun(tuple(batches), graph_batch[0], tail[0])


def capture_stage_batches(
    dit: nn.Module,
    cases: Sequence[ip.PipelineCase],
    context: CalibContext,
    *,
    steps: int = CALIB_STEPS,
    label: str = "calib",
) -> tuple[CapturedRun, ...]:
    """Run the reference denoise for each calibration case and collect the inputs to the first
    block.

    MUST: the caller calls this **before a single rounding of the GPTQ targets (the 168 in-block
    modules) is applied** (afterwards, the rounding targets of a weight would be chosen from
    activations produced by that same rounded weight — a cycle). Conversely, **what ships as i8 is
    rounded first** (the 5 outside the blocks, the 144 adaLN modules, the inner copies of other
    roles) — at distribution run time, what enters the blocks is the hidden produced by the i8
    `in_proj` and the condition embedding produced by the i8 `cond_module`, and inside the blocks
    the modulation produced by adaLN is applied with i8 values too. Rounding them later would ship
    "rounding targets chosen around f32 neighbours" combined with i8 neighbours (the ordering MUST
    of `irodori.export._round_i4_calibrated`).
    """
    if not cases:
        raise AssertionError("no calibration cases (calibrated i4 cannot work with zero inputs)")
    if steps < 1:
        raise AssertionError(f"the captured step count must be at least 1 (got {steps})")
    tokenizers = _load_tokenizers(context)
    return tuple(_capture_case(dit, case, context, tokenizers, steps, label) for case in cases)


def assert_stage_split(stages: Sequence[StageSpec], run: CapturedRun) -> None:
    """Check that the stage-by-stage block loop is **bit-identical to one forward of the bare
    DiT**.

    What is compared is the input to `out_norm` (the output of the block loop itself), and both
    sides were taken by {@link CapturedRun} from **the same forward**.

    MUST: compare with `torch.equal`, not an approximate tolerance — the same modules are called in
    the same order, so a mismatch means the split goes down a different path from the real one.

    MUST: call it before a single rounding is applied. Rounding through a drifted split would ship
    "GPTQ for a different path", and the numbers would still come out normally (the golden stays
    green too).
    """
    args, kwargs = run.graph_batch
    with torch.no_grad():
        hidden = args[0]
        for _prefix, stage in stages:
            hidden = stage(hidden, **kwargs)
    if not torch.equal(hidden, run.block_loop_output):
        raise AssertionError(
            "the final hidden of the stage split is not bit-identical to the output of the bare"
            " DiT's block loop"
            f" (max abs diff {float((hidden - run.block_loop_output).abs().max()):.4e})"
            " — the stage sequence has drifted from upstream's path"
        )


def _announce_stages(stages: Sequence[StageSpec]) -> list[RemovableHandle]:
    """Install hooks that print one progress line on the **first** forward of each stage (the
    return value is the handles to remove them).

    `karume.quant_calib` prints nothing, so a calibration in the default setup is silent for hours
    between the start line and the final diagnostics line and cannot be told apart from a hang.
    Core is a boundary (ADR 0065), so it is not touched; the caller peeks in instead (the same
    approach as `anima.calib._announce_stages`).
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
    stages: Sequence[StageSpec],
    batches: Sequence[StageBatch],
    include: Callable[[str], bool] | None = None,
) -> tuple[CalibReport, Int4Report]:
    """Apply stage-by-stage GPTQ and return `(report, scale ledger)` (the rounding is in place).

    `include` is a predicate on **local module FQNs within a stage**, passed straight to core's
    `calibrate_stages` (None = every `nn.Linear` in the stage). The i8+dit4 slot moves the 144
    adaLN modules back to i8, so the caller (`irodori.export._round_i4_calibrated`) passes the
    negation of {@link is_adaln} and puts only the 168 in-block modules through GPTQ.

    MUST: the set the predicate selects and the set stored as i4 are **identical** — if they drift,
    the "scanned count = rounded count" check loses its meaning and an unrounded weight gets an i4
    storage spec. It is measured by {@link assert_calib_covers_scan} (the caller passes the set
    stored as i4 and compares).

    MUST: a report without a ledger fails loudly — only `gptq` × `rtn` has a shipping path (the
    MUST of `karume.quant_calib`), and going on without a ledger would put weights without scales
    into the i4 slot.
    """
    tokens = sum(int(args[0].shape[1]) for args, _kwargs in batches)
    print(
        f"[calib] starting GPTQ calibration — {len(stages)} stages × {len(batches)} batches"
        f", {tokens:,} hidden tokens in total",
        flush=True,
    )
    handles = _announce_stages(stages)
    try:
        report = calibrate_stages(
            stages, batches, method=CALIB_METHOD, spec=CALIB_GRID, include=include
        )
    finally:
        for handle in handles:
            handle.remove()
    ledger = report.int4
    if ledger is None:
        raise AssertionError(
            f"calibration returned no scale ledger ({report.describe()})"
            " — only gptq × rtn can ship (the MUST of karume.quant_calib)"
        )
    return report, ledger


def assert_calib_covers_scan(report: CalibReport, scan: frozenset[str]) -> None:
    """Check that the layers the calibration rounded match **the set stored as i4** exactly, with
    nothing missing or extra.

    `scan` is a set of module names with `.weight` dropped (in the form of
    {@link stage_linear_names}; for the i8+dit4 slot, the 168 without adaLN — identical to the set
    the `include` of {@link calibrate_i4} selects).

    MUST: fail loudly. If the spelling of the stages or the target types changed and part of a block
    dropped out of calibration, the missed rounding would show up on the **better** side of quality
    (the pass-through cannot be read from the numbers). Moreover, the missed weights would then go
    into neither the i8 side nor the i4 side, while only the storage spec demands i4.
    """
    rounded = {layer.fqn for layer in report.layers}
    wanted = {f"{name}.weight" for name in scan}
    missing = sorted(wanted - rounded)
    extra = sorted(rounded - wanted)
    if missing or extra or report.modules != len(wanted):
        raise AssertionError(
            f"the {report.modules} modules the calibration rounded do not match the"
            f" {len(wanted)} of the scan"
            f" (not rounded {missing[:3]} / not in the scan {extra[:3]})"
        )
