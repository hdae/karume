"""Bake a LoRA **into the weights before export** (ADR 0016).

A Karume artifact is the static graph torch.export emits, so a LoRA is handled as a **weight
transform at export time**, not as a runtime mechanism. Merging it changes no IR node and needs no
runtime-side implementation at all (if adapter switching is ever needed, the design is redone then).

The target is Anima's distillation LoRAs (turbo and the like). The distributed format uses
non-diffusers naming, so it is converted to diffusers naming with **the conversion function shipped
in diffusers** before use (no naming table of our own — upstream is authoritative). After the
conversion the keys split into two groups: `transformer.*` and `text_conditioner.*`.

The distributed format carries no alpha, so the default scale is 1.0 (`scale` overrides it).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

_A_SUFFIX = ".lora_A.weight"
_B_SUFFIX = ".lora_B.weight"


@dataclass(frozen=True)
class FuseReport:
    """The result of a merge. The caller logs it so that a person can notice an anomaly."""

    merged: int
    """The number of Linear layers merged."""

    max_relative_delta: float
    """The maximum of `‖ΔW‖ / ‖W‖`. An extreme value means the scale is wrong."""

    @property
    def is_noop(self) -> bool:
        """No weight of this component moved (every `lora_B` is 0)."""
        return self.max_relative_delta == 0.0

    def describe(self) -> str:
        if self.is_noop:
            return (
                f"{self.merged} 層は ΔW=0 — LoRA に含まれるが学習されていない"
                "（この成分は再エクスポート不要）"
            )
        return f"merged into {self.merged} layers (max ‖ΔW‖/‖W‖ {self.max_relative_delta:.3f})"


def load_lora_state_dict(path: Path) -> dict[str, torch.Tensor]:
    """Read a LoRA file and convert it to diffusers naming.

    MUST: raise if **the whole file is inert** (every `lora_B` is 0). LoRA initialises B to 0, so an
    untrained file or a mixed-up file takes the shape "readable, but does nothing" — it keeps
    producing the same image as the base without anyone noticing. A zero component can be normal,
    so that case is left to the caller through the return value of {@link fuse_lora}.
    """
    from diffusers.loaders.lora_conversion_utils import (
        _convert_non_diffusers_anima_lora_to_diffusers as convert,
    )
    from safetensors.torch import load_file

    raw = load_file(str(path))
    if not raw:
        raise ValueError(f"{path}: the file holds no tensor")
    converted = convert(raw)
    live = sum(
        1
        for key, value in converted.items()
        if key.endswith(_B_SUFFIX) and float(value.abs().max()) > 0
    )
    if live == 0:
        raise ValueError(
            f"{path}: every lora_B is 0 — applying it gives the same result as the base"
        )
    return converted


def fuse_lora(
    model: nn.Module,
    state_dict: dict[str, torch.Tensor],
    prefix: str,
    scale: float = 1.0,
) -> FuseReport:
    """Add the LoRA under `prefix` into the weights of `model` (in place).

    ΔW = (B @ A)·scale is **computed in f32** and then added in the original dtype (building B@A in
    low precision turns the cancellation of the small-rank product straight into weight error).

    MUST: an unresolvable target or a shape mismatch **always raises**. Skipping it silently outputs
    "a degraded model with the LoRA on only some layers" as if nothing happened, and the images stay
    just slightly worse — especially hard to notice with a distillation LoRA. This also checks that
    **every key under prefix is consumed as an A/B pair** (scanning from `lora_A` alone would
    silently drop `.alpha` or weights with another suffix = the declaration and the implementation
    disagree).
    """
    head = f"{prefix}."
    unconsumed = {key for key in state_dict if key.startswith(head)}
    targets = sorted(
        key.removeprefix(head).removesuffix(_A_SUFFIX)
        for key in unconsumed
        if key.endswith(_A_SUFFIX)
    )
    if not targets:
        raise ValueError(f"LoRA に {prefix} 向けの層が 1 件も無い")

    params = dict(model.named_parameters())
    max_relative = 0.0
    for target in targets:
        down = state_dict.get(f"{head}{target}{_A_SUFFIX}")
        up = state_dict.get(f"{head}{target}{_B_SUFFIX}")
        if down is None or up is None:
            raise ValueError(f"{prefix}: {target} の lora_A / lora_B が揃っていない")
        unconsumed -= {f"{head}{target}{_A_SUFFIX}", f"{head}{target}{_B_SUFFIX}"}
        weight = params.get(f"{target}.weight")
        if weight is None:
            raise ValueError(f"{prefix}: {target}.weight がモデルに無い（命名変換の取りこぼし）")
        delta = (up.to(torch.float32) @ down.to(torch.float32)) * scale
        if delta.shape != weight.shape:
            raise ValueError(
                f"{prefix}: {target} の形が違う"
                f" LoRA={tuple(delta.shape)} 重み={tuple(weight.shape)}"
            )
        norm = float(torch.linalg.vector_norm(weight.detach()))
        if norm > 0:
            max_relative = max(max_relative, float(torch.linalg.vector_norm(delta)) / norm)
        with torch.no_grad():
            weight.add_(delta.to(weight.dtype))

    if unconsumed:
        raise ValueError(
            f"{prefix}: A/B ペアとして消費できないキーが {len(unconsumed)} 件残った"
            f"（{', '.join(sorted(unconsumed)[:5])}）— 焼き込まれない成分がある"
        )
    return FuseReport(merged=len(targets), max_relative_delta=max_relative)
