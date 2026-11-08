"""Gemma 4's routed experts through `karume::expert_linear` (ADR 0145 decision 5).

Upstream's experts module (`Gemma4TextExperts`, transformers 5.14.1) dispatches its forward through
transformers' `ExpertsInterface` by `config._experts_implementation`. This module registers one
more implementation, {@link EXPERTS_IMPLEMENTATION} (`karume`), instead of copying upstream's
`forward` or replacing the class (the way ADR 0135 decision 5 swaps a function, not a class):

    gate_up = expert_linear(x, index, gate_up_proj)      # [T, k, 2·I]
    mixed   = act(gate) · up                             # upstream's `_apply_gate`
    down    = expert_linear(mixed, index, down_proj)     # [T, k, H]
    out     = Σ_k down · routing_weight                  # in pick order

`expert_linear` is the custom op `karume::expert_linear` (`karume.custom_ops`): the trace keeps it
as one node with a symbolic token axis, and its eager body is upstream's per-expert `F.linear` for
each pick. The weighted sum runs over the k picks in pick order, where upstream's eager adds in
expert-id order (its `grouped_mm` and `batched_mm` sum in pick order): parity is at the level of
the numbers (ADR 0145 decision 2).

A reference that holds the experts as the container's packed bytes replaces each stacked weight with
a {@link PackedExperts}; the forward then calls the op's own eager body
(`karume.custom_ops.expert_linear_body`) with a getter that dequantises the one expert it is asked
for, so the trace and the reference share one per-pick summation and only the picked experts are
ever dequantised.

MUST: transformers is imported inside functions.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

import karume.custom_ops  # noqa: F401  -- registers karume::expert_linear
from karume.custom_ops import EXPERT_LINEAR_UPSTREAM, expert_linear_body
from karume.emit import f16_step_table, unpack_int4
from karume.quantize import dequantize_int4

#: The name of the implementation in transformers' `ExpertsInterface` — the value of
#: `config._experts_implementation` for the trace and the reference alike.
EXPERTS_IMPLEMENTATION = "karume"

#: What the goldens' record names as the experts implementation that ran (ADR 0145 decision 6).
EXPERTS_RECORD: dict[str, str] = {
    "implementation": EXPERTS_IMPLEMENTATION,
    "body": EXPERT_LINEAR_UPSTREAM,
    "sum": "over the k picks in pick order",
}


class ExpertsError(RuntimeError):
    """The experts are not the form this module routes."""


class PackedExperts:
    """One stacked expert weight `[E, out, in]` held as the container stores it: the packed i4
    codes `[E·out, in/2]` and one step per group, rounded to f16 the way the writer rounds them
    (`karume.emit.f16_step_table` — ADR 0145 decision 3) and kept as f16.

    {@link expert} dequantises one expert to F32 (`f32(code) · f32(step)`, the kernel's numeric
    contract), so a forward that reads only the picked experts holds one `[out, in]` F32 tensor at a
    time. `packed` is shared with the container's material (not copied).

    Not a module and not a buffer on purpose: nothing walks it as a model tensor (a NaN check over
    the model's buffers would allocate a mask the size of the codes).
    """

    def __init__(
        self, packed: torch.Tensor, steps: torch.Tensor, shape: tuple[int, int, int], where: str
    ) -> None:
        experts, out_features, in_features = shape
        rows = experts * out_features
        if packed.dtype != torch.uint8 or list(packed.shape) != [rows, in_features // 2]:
            raise ExpertsError(f"{where}: packed codes {list(packed.shape)} do not fit {shape}")
        if steps.dtype != torch.float32 or steps.dim() != 2 or int(steps.shape[0]) != rows:
            raise ExpertsError(f"{where}: steps {list(steps.shape)} do not fit {shape}")
        self.where = where
        self.experts = experts
        self.out_features = out_features
        self.in_features = in_features
        self.packed = packed
        self.steps = rounded_steps(packed, steps, out_features, where)

    def expert(self, expert: int) -> torch.Tensor:
        """Expert `expert`'s weight `[out, in]` as F32 — what the container's bytes decode to."""
        rows = slice(expert * self.out_features, (expert + 1) * self.out_features)
        codes = unpack_int4(self.packed[rows], (self.out_features, self.in_features))
        return dequantize_int4(codes, self.steps[rows].to(torch.float32))


def rounded_steps(
    packed: torch.Tensor, steps: torch.Tensor, rows_per_expert: int, where: str
) -> torch.Tensor:
    """The F32 steps rounded to f16 by the writer's own function (`karume.emit.f16_step_table`),
    one expert at a time, as an f16 tensor of the steps' shape.

    MUST: the rounding is the writer's, not a copy of it — the reference and the goldens use the
    rounded step (ADR 0006), so the two cannot drift. The table rounds each step on its own (and
    floors an all-zero group the same way in any block), so rounding expert by expert gives the
    whole tensor's table; one expert at a time keeps the writer's checks (the round trip from the
    table's bytes) from holding a whole tensor's dequantised copies.
    """
    rows, groups = (int(dim) for dim in steps.shape)
    rounded = torch.empty((rows, groups), dtype=torch.float16)
    for start in range(0, rows, rows_per_expert):
        stop = start + rows_per_expert
        table = f16_step_table(packed[start:stop], steps[start:stop], f"{where} rows {start}..")
        rounded[start:stop] = table[: (stop - start) * groups].reshape(stop - start, groups)
    return rounded


def register_experts() -> None:
    """Register {@link karume_experts_forward} as {@link EXPERTS_IMPLEMENTATION} (idempotent — the
    interface's mapping is process-wide)."""
    from transformers.integrations.moe import ExpertsInterface

    ExpertsInterface.register(EXPERTS_IMPLEMENTATION, karume_experts_forward)


def use_karume_experts(config: Any) -> None:
    """Register the implementation and name it in `config` — MUST: before the model is built
    (upstream resolves `_experts_implementation` in `PreTrainedModel.__init__`)."""
    register_experts()
    config._experts_implementation = EXPERTS_IMPLEMENTATION


def assert_karume_experts(target: Any) -> None:
    """`target` (a config, or a built model and every experts module in it) names
    {@link EXPERTS_IMPLEMENTATION}.

    MUST: checked after the model is built — an implementation that is unset or set after
    construction falls back to upstream's `grouped_mm` (`modeling_utils.get_correct_experts_
    implementation`), which the trace refuses loudly but a reference would run silently.
    """
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextExperts

    if isinstance(target, nn.Module):
        configs = [("the model", target.config)] + [
            (name, module.config)
            for name, module in target.named_modules()
            if isinstance(module, Gemma4TextExperts)
        ]
    else:
        configs = [("the config", target)]
    for where, config in configs:
        found = getattr(config, "_experts_implementation", None)
        if found != EXPERTS_IMPLEMENTATION:
            raise ExpertsError(
                f"{where} runs its experts through {found!r}, not {EXPERTS_IMPLEMENTATION!r}"
                " (gemma4.experts.use_karume_experts before the model is built)"
            )


def _project(weight: Any, x: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """One routed projection: the op over a stacked weight (the trace, an F32 load), or the op's
    own body over a {@link PackedExperts} (the reference)."""
    if isinstance(weight, PackedExperts):
        return expert_linear_body(x, index, weight.expert, weight.experts, weight.out_features)
    return torch.ops.karume.expert_linear(x, index, weight)


def karume_experts_forward(
    self: nn.Module,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> torch.Tensor:
    """`Gemma4TextExperts.forward` through `karume::expert_linear` (module docstring): hidden
    `[T, H]`, index and weights `[T, k]` → `[T, H]`.

    MUST: only upstream's default layout — gate rows then up rows in one `gate_up_proj`, not
    transposed, no bias; another layout fails loudly instead of being read as this one.
    """
    layout = {
        "is_concatenated": True,
        "is_transposed": False,
        "has_bias": False,
        "has_gate": True,
    }
    wrong = {name: getattr(self, name, None) for name, want in layout.items()}
    wrong = {name: got for name, got in wrong.items() if got != layout[name]}
    if wrong:
        raise ExpertsError(f"experts layout {wrong} is not upstream's default {layout}")
    gate_up = _project(self.gate_up_proj, hidden_states, top_k_index)
    mixed = self._apply_gate(gate_up)
    down = _project(self.down_proj, mixed, top_k_index)
    return (down * top_k_weights.unsqueeze(-1)).sum(dim=1)
