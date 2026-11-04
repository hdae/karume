"""Rig that runs the **calibrated rounding** (GPTQ × RTN grid) of the i4 series (perf-ledger Q-6).

`karume.quant_calib.calibrate_stages` is a driver that "advances one stage at a time and re-picks
the rounding targets from the activations that actually flow through that layer". The storage
grid is **not one byte different** from {@link karume.quantize.quantize_to_int4}
(`scale = group amax / 7`, `q ∈ [−7,+7]`). Only the rounded values and the contents of the scale
ledger change; neither the storage path on the `emit` side nor the byte structure of the
distribution moves at all.

This module provides the three things the core driver requires:

1. **The stage list** — a sequence of {@link EncoderStage} that copies the loop of
   `DebertaV2Encoder.forward`. Each entry is a `(in-model FQN prefix, module)` pair, and the prefix
   lives in **the FQN space of the wrapper being exported** ({@link STAGE_PREFIX}) — unless the
   scale ledger's keys live in the same space as the safetensors tensor keys, the matching on the
   emit side finds nothing (the FQN discipline MUST of `deberta.export._fake_quant`).
2. **The inputs to the first stage** — instead of rebuilding the embeddings, the 4-D mask, the
   relative-position index tables and `rel_embeddings` by hand, a forward_pre_hook captures
   **the call of the first layer itself** ({@link capture_stage_batches}).
3. **The stage-split equivalence check** — running the captured inputs through the stage list in
   order must give a final hidden that is **bit-identical** to the corresponding output of the
   wrapper forward without hooks ({@link assert_stage_split}).

MUST: measure 3 before rounding. {@link EncoderStage} is a copy of the upstream loop, so it drifts
silently when `DebertaV2Encoder.forward` changes on the transformers side — rounding on the
drifted copy ships "GPTQ of a different path", and the numbers still come out normally (the
goldens stay green as well).

NOTE (why this is not shared with the rig of `sbv2.measure_quant`): that rig drives the **plain**
`DebertaV2Model`, whereas export drives the wrapper after
`deberta.patch.apply_external_rel_pos_patch()` has been applied, so the actual signature of a
stage (`relative_pos` arrives as a tuple) is not guaranteed to be the same. Merging the copies into
one would make one side follow silently when the other's assumption moves. Each rig verifies itself
independently with its own equivalence check.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from karume.quant_calib import (
    CalibReport,
    GridSpec,
    StageBatch,
    StageSpec,
    calibrate_stages,
)
from karume.quantize import DEFAULT_GROUP_SIZE, Int4Report, iter_quant_targets

#: The in-model FQN prefix of the stage list. `HiddenStatesWrapper.model` is a `DebertaV2Model`,
#: so a layer's FQN is `model.encoder.layer.<layer index>.…` ({@link EncoderStage} names its child
#: by the layer index, so the prefix needs to go only this far).
STAGE_PREFIX = "model.encoder.layer"

#: The keyword name {@link EncoderStage.forward} receives for the ConvLayer. Apart from the 4-D
#: mask it passes to the layers, `DebertaV2Encoder` passes **the plain 2-D mask** to the ConvLayer.
INPUT_MASK_KWARG = "input_mask"

#: The calibrated-rounding method. Only GPTQ returns a scale ledger (a MUST of
#: `karume.quant_calib`).
CALIB_METHOD = "gptq"

#: The storage grid rounded onto. **`rtn` × g32 = the same grid as the existing i4 series** (the
#: only shipping path).
CALIB_GRID = GridSpec(kind="rtn", group_size=DEFAULT_GROUP_SIZE)


class _FirstStageReached(Exception):  # noqa: N818 — a stop signal, not a failure, so not an Error
    """Signal that the first stage's inputs are complete (a sentinel that cuts the calibration
    forward short)."""


class EncoderStage(nn.Module):
    """Stage wrapper that wraps one encoder layer into the form "receives hidden as a positional
    argument".

    There are two reasons to wrap, and both come from the call shape of
    `DebertaV2Encoder.forward` (transformers 5.14.1):

    1. The mask is passed to a layer as a **positional argument**
       (`layer_module(next_kv, attention_mask, …)`). `calibrate_stages` is a driver that passes
       "the selected output as **the only positional argument**" to the next stage, so unless the
       mask is moved to a keyword, the second stage onwards fails.
    2. **Only the first layer** has the ConvLayer on its output (`i == 0` inside the loop — it
       blends the stage's input hidden back in as a residual). Dropping it makes the second stage
       onwards see a hidden different from the real one, and the calibration picks rounding
       targets from different activations.

    The layer kwargs (`relative_pos` / `rel_embeddings` / `query_states` / `output_attentions`)
    are built once **before** the loop and are identical for every layer, so they fit the default
    of `calibrate_stages` (kwargs unchanged between stages) as they are.

    The child module is named by **the layer index** so that the local FQN within a stage
    (`0.attention.self.query_proj.weight`) turns back into the in-model FQN just by prepending
    {@link STAGE_PREFIX}. The ConvLayer is registered as a child too, but it holds no `nn.Linear`
    at all (`Conv1d` + `LayerNorm`), so the target set of the scan (`iter_quant_targets`, limited
    to `nn.Linear`) does not move.
    """

    def __init__(self, index: int, layer: nn.Module, conv: nn.Module | None) -> None:
        super().__init__()
        self.child = str(index)
        self.add_module(self.child, layer)
        self.conv = conv

    def forward(
        self, hidden: torch.Tensor, *, input_mask: torch.Tensor, **layer_kwargs: Any
    ) -> torch.Tensor:
        output = getattr(self, self.child)(hidden, **layer_kwargs)[0]
        if self.conv is None:
            return output
        return self.conv(hidden, output, input_mask)


def _encoder(wrapper: nn.Module) -> nn.Module:
    """Take the `DebertaV2Encoder` out of the wrapper (fail loudly if the structure is not the
    expected one)."""
    encoder = wrapper.model.encoder
    if encoder.conv is None:
        raise AssertionError(
            "the DeBERTa encoder has no ConvLayer (a structure without the first layer's residual"
            " blend — the model's structure differs from what the script assumes)"
        )
    return encoder


def encoder_stages(wrapper: nn.Module) -> tuple[StageSpec, ...]:
    """Return the encoder layers in execution order as `(in-model FQN prefix, stage)`.

    It lists **every layer in the exported graph** (unlike the rig of `sbv2.measure_quant`, which
    drops the dead layers at the end) — this side ships the already truncated model as is, so every
    layer present is a weight of the distribution.
    """
    encoder = _encoder(wrapper)
    return tuple(
        (STAGE_PREFIX, EncoderStage(index, layer, encoder.conv if index == 0 else None))
        for index, layer in enumerate(encoder.layer)
    )


def stage_linear_names(stages: Sequence[StageSpec]) -> frozenset[str]:
    """The in-model FQNs of the `nn.Linear` modules the stage list holds (**module names**, without
    `.weight`).

    Target selection shares the core's `iter_quant_targets` — if "what the calibration rounds" and
    "what the caller counts as inside a stage" were separate copied implementations, a weight in
    neither would silently drop to i8.
    """
    return frozenset(
        f"{prefix}.{local}".removesuffix(".weight")
        for prefix, stage in stages
        for local, _weight, _axis in iter_quant_targets(stage, (nn.Linear,))
    )


def capture_stage_batches(
    wrapper: nn.Module, inputs: Sequence[Sequence[torch.Tensor]]
) -> tuple[StageBatch, ...]:
    """Capture the hidden and the accompanying arguments of the first stage with a
    forward_pre_hook (Catcher).

    `inputs` are **the wrapper's positional arguments themselves** (in the order of
    `deberta.export.INPUT_ORDER`). What is captured is "the hidden after the embeddings, and the
    4-D mask / relative positions / `rel_embeddings` the encoder built"; rebuilding them by hand
    would silently diverge from the transformers side.

    The 2-D mask the ConvLayer needs is not passed to the layers, so one more tensor is captured
    from **the ConvLayer's call**. The sentinel stops at the ConvLayer — only the first layer runs
    up to that point.

    MUST: if the forward completes without everything captured, fail loudly — it signals that the
    stage spelling disagrees with the model's structure, and carrying on silently would only end in
    a "zero calibration inputs" diagnostic on the core side.
    """
    encoder = _encoder(wrapper)
    captured: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    masks: list[torch.Tensor] = []

    def catch_layer(_module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        captured.append((args, dict(kwargs)))

    def catch_conv(_module: nn.Module, args: tuple[Any, ...]) -> None:
        masks.append(args[2])
        raise _FirstStageReached

    handles = [
        encoder.layer[0].register_forward_pre_hook(catch_layer, with_kwargs=True),
        encoder.conv.register_forward_pre_hook(catch_conv),
    ]
    batches: list[StageBatch] = []
    try:
        for index, args in enumerate(inputs):
            captured.clear()
            masks.clear()
            try:
                with torch.no_grad():
                    wrapper(*args)
            except _FirstStageReached:
                pass
            if len(captured) != 1 or len(masks) != 1:
                raise AssertionError(
                    f"calibration input {index} did not capture the first stage's inputs"
                    f" (layer {len(captured)} / ConvLayer {len(masks)})"
                    " — the stage spelling disagrees with the model's structure"
                )
            layer_args, layer_kwargs = captured[0]
            if len(layer_args) != 2:
                raise AssertionError(
                    f"the first layer was called with {len(layer_args)} positional arguments"
                    " (2 expected, hidden and mask — the call shape on the transformers side has"
                    " changed)"
                )
            hidden, attention_mask = layer_args
            batches.append(
                (
                    (hidden.detach(),),
                    {"attention_mask": attention_mask, INPUT_MASK_KWARG: masks[0], **layer_kwargs},
                )
            )
    finally:
        for handle in handles:
            handle.remove()
    return tuple(batches)


def assert_stage_split(
    wrapper: nn.Module,
    probe: Sequence[torch.Tensor],
    batch: StageBatch,
    stages: Sequence[StageSpec],
) -> None:
    """Check that running the stage list in order is **bit-identical** to the wrapper's final
    hidden.

    `probe` is the wrapper's positional arguments, and `batch` is the first-stage input captured
    from **the same input**. The wrapper's output ends with the final layer's hidden (it is last
    with or without `single_output`), so the comparison target is `outputs[-1]`.

    MUST: measure this before rounding, with `torch.equal` rather than an approximate tolerance —
    since the same modules are called in the same order, a mismatch means the split takes a path
    different from the real one.
    """
    args, kwargs = batch
    with torch.no_grad():
        reference = wrapper(*probe)[-1]
        hidden = args[0]
        for _prefix, stage in stages:
            hidden = stage(hidden, **kwargs)
    if not torch.equal(hidden, reference):
        raise AssertionError(
            "the final hidden of the stage split is not bit-identical to the wrapper output"
            f" (max abs diff {float((hidden - reference).abs().max()):.4e})"
            " — the EncoderStage copy has drifted from DebertaV2Encoder.forward"
        )


@dataclass(frozen=True)
class CalibRig:
    """Scaffolding for one calibration run (the stage list and the inputs to the first stage)."""

    stages: tuple[StageSpec, ...]
    batches: tuple[StageBatch, ...]

    @property
    def tokens(self) -> int:
        """Total tokens of the hidden states fed to the calibration (a diagnostic that records a
        reduced run next to the numbers)."""
        return sum(int(args[0].shape[-2]) for args, _kwargs in self.batches)


def build_rig(
    wrapper: nn.Module, stages: Sequence[StageSpec], inputs: Sequence[Sequence[torch.Tensor]]
) -> CalibRig:
    """Build the calibration scaffolding (Catcher → stage-split equivalence check).

    MUST: fail loudly with zero calibration inputs — no branch silently drops to plain RTN
    ("shipped uncalibrated while meaning to calibrate" cannot be read from the assets).
    """
    if not inputs:
        raise AssertionError(
            "no calibration inputs at all (calibrated i4 cannot be built from zero inputs)"
        )
    batches = capture_stage_batches(wrapper, inputs)
    assert_stage_split(wrapper, inputs[0], batches[0], stages)
    return CalibRig(stages=tuple(stages), batches=batches)


def calibrate_i4(rig: CalibRig, include: Callable[[str], bool]) -> tuple[CalibReport, Int4Report]:
    """Apply stage-by-stage GPTQ and return `(report, scale ledger)` (the rounding is in place).

    `include` is a predicate on **the local module FQN within a stage**
    (`0.attention.self.query_proj`) (the contract of the core's `iter_quant_targets`).

    MUST: a report without a ledger fails loudly — only `gptq` × `rtn` has a shipping path (a MUST
    of `karume.quant_calib`), and going on without a ledger would put weights without scales into
    the i4 storage.
    """
    report = calibrate_stages(
        rig.stages, rig.batches, method=CALIB_METHOD, spec=CALIB_GRID, include=include
    )
    ledger = report.int4
    if ledger is None:
        raise AssertionError(
            f"the calibration returned no scale ledger ({report.describe()})"
            " — only gptq × rtn can ship (a MUST of karume.quant_calib)"
        )
    return report, ledger
