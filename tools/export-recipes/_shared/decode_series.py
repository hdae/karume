"""Checks and file operations shared by the scripts of decode series (ADR 0065 decision 2 — they
cannot be promoted to core).

Only **the layer of the states-form decode scripts that holds no model knowledge** lives here:
how a greedy continuation is taken ({@link greedy_continuation}), its margin check
({@link assert_greedy_margins}), writing the golden ({@link _write_greedy}), the capacity check
of the position table ({@link assert_case_room}) and the timing of publication ({@link _publish}
— the swap itself is core's {@link karume.artifacts.swap_into_place}). None of it reads the
config or layer_type, so the arguments need no per-family generalisation.

MUST: **this module is the single source of truth for these rules** (the family scripts import
it). Tightening a check in one family alone silently weakens the other — in fact, while they were
verbatim copies, the MUST of the margin floor existed only on the gemma4 side (review G3-03 of
2026-08-19).

What is **not lifted here**: `_write_container` (it branches on whether quantization arguments
are given) and `_write_io` (the number of outputs and the input names are family-shaped). This
does not cross the boundary where generalising would make the code harder to read.

The dependency direction is recipe → core only (`_shared` does not import families — spellings
it needs are held here).
"""

from __future__ import annotations

import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

import torch
from safetensors.torch import save_file
from torch import nn

from karume.artifacts import swap_into_place
from karume.convert import normalize_boundary_tensor

#: File name of the greedy golden (`greedy.<case>.safetensors`). Its readers are the TS-side
#: acceptance check and `minicpm5/sweep_w4.py`. The extension is the same as the one-shot
#: `io.*`, but `_shared` does not import families, so the spelling is held here.
GREEDY_PREFIX = "greedy."
GREEDY_SUFFIX = ".safetensors"

#: Tensor keys of `greedy.<case>.safetensors`.
PROMPT_KEY = "prompt"
EXPECTED_KEY = "expected"
MARGIN_KEY = "margin"


def positions_for(ids: torch.Tensor) -> torch.Tensor:
    """The full-length unpadded position row `[[0, 1, …, T-1]]` (one chunk consumes the whole
    prompt)."""
    return torch.arange(int(ids.shape[1])).unsqueeze(0)


def greedy_continuation(
    wrapper: nn.Module, ids: torch.Tensor, steps: int, *, label: str = ""
) -> tuple[list[int], list[float]]:
    """`(tokens, margins)` of a `steps`-step greedy continuation that **calls the chunk wrapper
    again on the full length**.

    MUST: take it with a full re-forward (every step computed from the start with standard
    causal attention). Taking it through the KV-cache path would build the expected values with
    the very mechanism the runtime wants to check, and the check would be green even when both
    are wrong in the same direction. The slowness is accepted (`steps` full-length forwards per
    case).

    `margin` is each step's top1 − top2 logit gap. It guarantees that GPU-execution deviation
    cannot flip this sequence; the threshold check lives in {@link assert_greedy_margins}.
    """
    tokens: list[int] = []
    margins: list[float] = []
    current = ids
    for step in range(steps):
        with torch.no_grad():
            logits, _ = wrapper(current, positions_for(current))
        best = torch.topk(logits[0, -1], 2)
        token = int(best.indices[0])
        tokens.append(token)
        margins.append(float(best.values[0] - best.values[1]))
        current = torch.cat([current, best.indices[:1].unsqueeze(0)], dim=1)
        print(
            f"[greedy] {label} step {step + 1}/{steps} token={token} margin={margins[-1]:.4g}",
            file=sys.stderr,
            flush=True,
        )
    return tokens, margins


def assert_greedy_margins(margins: Mapping[str, Sequence[float]], floor: float) -> None:
    """Check that the margin of every step of every case exceeds the floor (the multi-step
    version of iteration A's margin-guarantee check).

    MUST: fail loudly. Mixing a small-margin step into the golden makes a check where "the top-1
    swaps through deviation even when the GPU side is right", and no red can be trusted after
    that.
    MUST: apply it **once, after every case has been measured** (do not stop at the first case).
    The check's job is to inform "which cases can become golden"; stopping at the first one would
    mean redoing the real run after every exclusion (one real-weight run takes tens of minutes).
    """
    weak = {
        name: {step: margin for step, margin in enumerate(values) if margin <= floor}
        for name, values in margins.items()
    }
    offenders = {name: steps for name, steps in weak.items() if steps}
    if offenders:
        raise AssertionError(
            f"steps with a margin at or below the floor {floor}: {offenders}"
            " (do not lower K; remove those cases from GREEDY_CASES)"
        )


def _write_greedy(
    wrapper: nn.Module,
    cases: Sequence[tuple[str, torch.Tensor]],
    out_dir: Path,
    *,
    steps: int,
    floor: float,
) -> tuple[list[str], dict[str, list[int]], dict[str, list[float]]]:
    """Measure the greedy continuations of the adopted cases and write them after applying the
    check **once every case is measured**.

    Nothing is written before the check, so a failure never leaves a golden in which "only some
    cases are new".
    """
    tokens: dict[str, list[int]] = {}
    margins: dict[str, list[float]] = {}
    for name, ids in cases:
        tokens[name], margins[name] = greedy_continuation(wrapper, ids, steps, label=name)
    assert_greedy_margins(margins, floor)

    written: list[str] = []
    for name, ids in cases:
        continuation, margin = tokens[name], margins[name]
        tensors = {
            PROMPT_KEY: normalize_boundary_tensor(ids[0], f"prompt of {name}"),
            EXPECTED_KEY: normalize_boundary_tensor(
                torch.tensor(continuation, dtype=torch.int64), f"expected of {name}"
            ),
            MARGIN_KEY: torch.tensor(margin, dtype=torch.float32).contiguous(),
        }
        path = out_dir / f"{GREEDY_PREFIX}{name}{GREEDY_SUFFIX}"
        save_file(tensors, str(path))
        written.append(path.name)
    return written, tokens, margins


def assert_case_room(cases: Sequence[tuple[str, torch.Tensor]], steps: int, positions: int) -> None:
    """Check that prompt + continuation fits in the RoPE table's positions (nothing outside the
    table can be looked up)."""
    for name, ids in cases:
        total = int(ids.shape[1]) + steps
        if total > positions:
            raise AssertionError(
                f"{name}: prompt {int(ids.shape[1])} + {steps} steps = {total} exceeds"
                f" the RoPE table's {positions} positions"
            )


def _publish(staging: Path, final: Path) -> None:
    """Swap a staging directory that passed every check into final **as a whole**.

    MUST: publication comes **after all of** the shape checks, the margin check and the sanity
    checks (guaranteed by the structure of each family's `export_series`, the caller). Writing
    into final file by file would, on a failure of a check halfway, leave a **mixed canonical
    asset** of "new model + old greedy", and the acceptance check could go green on an asset that
    was already rejected.

    The rules of the swap itself (move aside → promote → restore on failure, cleaning up what an
    interruption left behind) belong to the core primitive
    ({@link karume.artifacts.swap_into_place}) — this function only spells out "when publishing
    is allowed".

    NOTE: every production caller has moved to calling core directly (CX-1.1/1.4 of 2026-08-19),
    and only the tests of both families reference it. It is still **kept on purpose** (the
    maintainer's decision of 2026-08-19) — as the documentation anchor of the "timing of
    publication" rule. Not a target of dead-code cleanup.
    """
    swap_into_place(staging, final)
