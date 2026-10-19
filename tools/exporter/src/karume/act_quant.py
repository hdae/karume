"""**Per-token symmetric int8** fake-quant of activations (the torch mirror of the w8a8 path).

The source of truth on the runtime side is `packages/runtime/src/kernels/quantize-rows.ts` (design
= `docs/research/2026-08-03-dp4a-w8a8-design.md` §4.2); this module is a **mirror of its numeric
spec**. Unlike the weight side (`quantize.fake_quant_int8`), activations are only known at run
time, so the exporter cannot round them ahead of time — the side that takes reference fixtures
attaches a `forward_pre_hook` and reproduces "the values the runtime actually consumes" (ADR
0006's fake-quant methodology extended to activations).

    s = clamp(rowmax(|x|) / 127, f32 tiny)      … row = last axis (per-token)
    q = clamp(round(x / s), -127, +127)         … **closed at ±127** (−128 is not used)
    x̂ = q · s

MUST: `torch.round` rounds half to even and matches WGSL's `round` (confirmed by observation).
Replacing it with a half-up implementation (JS `Math.round` and the like) shifts by ±1 step at the
lattice boundaries.

MUST: the scale is built with the same **multiplication `amax * float32(1/127)`** as the runtime
(a mirror of the MUST in quantize-rows.ts — torch's division `amax / 127` is correctly rounded, so
a few % of rows get an `s` 1 ULP away from the multiplication form; aligned to the multiplication
form on 2026-08-31). The only remaining non-mirrored part is the division `x / s`; the GPU side
gets WGSL's 2.5 ULP allowance there, so it can wobble by ±1 step near half-integer boundaries
(quantize-rows.ts § Division precision — bit identity fails only there).
"""

from __future__ import annotations

import torch
from torch import nn

from .quantize import INT8_MAX, QuantizeError

#: Alignment of `k` that comes from packing the i8 payload 4 per flat index.
PACK_ALIGN = 4


def quantize_rows_parts(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return {@link quantize_rows} split into two parts, `(integer values, row scale)`.

    A caller that simulates the integer contraction (the `(Σ q_a·q_w)·(s_a·s_w)` form) needs the
    integers before they fold into `q·s`. Taking them from **the same single expression** as the
    fake-quant removes any room for the two to drift from the spec separately.
    """
    amax = x.abs().amax(dim=-1, keepdim=True)
    # MUST: multiplication by f32(1/127), not division (runtime mirror — module docstring).
    # A Python float is a weak scalar, so it drops to amax's dtype (f32) before the multiply.
    scale = torch.clamp(amax * (1.0 / INT8_MAX), min=torch.finfo(torch.float32).tiny)
    return torch.round(x / scale).clamp_(-INT8_MAX, INT8_MAX), scale


def quantize_rows(x: torch.Tensor) -> torch.Tensor:
    """Per-token symmetric i8 fake-quant that treats the last axis as the row."""
    quantized, scale = quantize_rows_parts(x)
    return quantized * scale


def is_eligible(module: nn.Module) -> bool:
    """Mirror of the runtime's eligibility rule (`nn.Linear` with `k % 4 == 0`, bias or not).

    The runtime additionally requires "the weight is i8 and resident on the GPU"
    (`linearCompute: "a8"`). In an i8 series every eligible `nn.Linear` weight is resident as i8, so
    the count matches the one taken here.
    """
    return isinstance(module, nn.Linear) and module.in_features % PACK_ALIGN == 0


def attach_act_quant(model: nn.Module) -> tuple[list[object], int]:
    """Attach hooks that drop the inputs of eligible `nn.Linear` layers to per-token i8.

    Returns `(removal handles, number attached)`. **Returning the count is the point**: taking a
    reference with 0 attached would not reveal that "w8 numbers were taken for what was meant to be
    w8a8" (the same practice as ADR 0006's always-on diagnostics).
    """
    handles: list[object] = []
    attached = 0

    def make_pre(name: str):
        def pre(_module: nn.Module, args: tuple[object, ...]):
            # MUST: a call that cannot be rounded raises instead of passing through (the same
            # discipline as its sibling `quant_calib._make_pre_hook`). Passing through would let a
            # reference with activation quantization missing on that one layer slip past the
            # "number attached" diagnostic and be taken as w8, and the tolerance check cannot tell
            # that difference from quantization error.
            if not args or not isinstance(args[0], torch.Tensor):
                raise QuantizeError(
                    f"'{name}': the linear targeted by activation quantization is not called with"
                    " a positional Tensor argument (its input cannot be rounded, so this layer"
                    " alone would give a w8 reference)"
                )
            return (quantize_rows(args[0]), *args[1:])

        return pre

    for name, module in model.named_modules():
        if not is_eligible(module):
            continue
        handles.append(module.register_forward_pre_hook(make_pre(name), with_kwargs=False))
        attached += 1
    return handles, attached


def detach_act_quant(handles: list[object]) -> None:
    """Remove every handle returned by {@link attach_act_quant}."""
    for handle in handles:
        remove = getattr(handle, "remove", None)
        if remove is not None:
            remove()
    handles.clear()
