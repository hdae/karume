"""Pins the behavior of baking a LoRA in (ADR 0016).

Two claims: (1) ΔW=(B@A)·scale is computed in f32 and added to the original weights; (2) an
unresolvable target or a shape mismatch **always raises** (skipping it silently would quietly
output "a degraded model with the LoRA on only some layers").

`load_lora_state_dict` calls the naming conversion bundled with diffusers, so it is not covered
here (upstream is the source of truth for the conversion table — we do not keep our own).
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from anima.lora import fuse_lora


class Target(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([nn.Linear(4, 3, bias=False) for _ in range(2)])


def _state(scale_b: float = 1.0, *, rank: int = 2) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(3)
    state: dict[str, torch.Tensor] = {}
    for index in range(2):
        head = f"transformer.blocks.{index}"
        state[f"{head}.lora_A.weight"] = torch.randn(rank, 4, generator=generator)
        state[f"{head}.lora_B.weight"] = torch.randn(3, rank, generator=generator) * scale_b
    return state


class TestFuseLora:
    def test_the_delta_is_added_to_every_target_weight(self):
        model = Target()
        before = [block.weight.detach().clone() for block in model.blocks]
        state = _state()

        report = fuse_lora(model, state, "transformer", scale=0.5)

        assert report.merged == 2
        for index, block in enumerate(model.blocks):
            delta = (
                state[f"transformer.blocks.{index}.lora_B.weight"]
                @ state[f"transformer.blocks.{index}.lora_A.weight"]
            ) * 0.5
            torch.testing.assert_close(block.weight, before[index] + delta)

    def test_the_scale_actually_scales(self):
        """An implementation that ignores the scale gives the same weights for any scale (detects a
        vacuous test)."""
        half, full = Target(), Target()
        full.load_state_dict(half.state_dict())
        state = _state()

        fuse_lora(half, state, "transformer", scale=0.5)
        fuse_lora(full, state, "transformer", scale=1.0)

        assert not torch.equal(half.blocks[0].weight, full.blocks[0].weight)

    def test_an_all_zero_lora_b_is_reported_as_a_noop(self):
        """ΔW=0 for a component can be legitimate — it is reported in the return value, not
        raised."""
        model = Target()

        report = fuse_lora(model, _state(scale_b=0.0), "transformer")

        assert report.is_noop
        assert "not trained" in report.describe()

    def test_a_prefix_with_no_targets_is_rejected(self):
        with pytest.raises(ValueError, match="has no layers for"):
            fuse_lora(Target(), _state(), "text_conditioner")

    def test_a_missing_lora_b_is_rejected(self):
        state = _state()
        del state["transformer.blocks.1.lora_B.weight"]

        with pytest.raises(ValueError, match="does not have both lora_A and lora_B"):
            fuse_lora(Target(), state, "transformer")

    def test_a_target_absent_from_the_model_is_rejected(self):
        """A key missed by the naming conversion yields "a degraded model with only part applied",
        so it always fails."""
        state = _state()
        for suffix in ("lora_A", "lora_B"):
            key = f"transformer.blocks.1.{suffix}.weight"
            state[key.replace("blocks.1", "blocks.9")] = state.pop(key)

        with pytest.raises(ValueError, match="is not in the model"):
            fuse_lora(Target(), state, "transformer")

    def test_a_leftover_key_under_the_prefix_is_rejected(self):
        """A key not consumed as an A/B pair is "a component that is not baked in" — it is not
        dropped silently.

        An implementation that only scans from `lora_A` lets this through (even if `.alpha` or
        weights with another suffix get into the distributed format, it does not "always raise on
        an unresolvable target" as declared).
        """
        state = _state()
        state["transformer.blocks.0.lora_C.weight"] = torch.zeros(3, 4)

        with pytest.raises(ValueError, match="cannot be consumed as A/B pairs"):
            fuse_lora(Target(), state, "transformer")

    def test_a_shape_mismatch_is_rejected(self):
        state = _state()
        state["transformer.blocks.0.lora_A.weight"] = torch.randn(2, 5)

        with pytest.raises(ValueError, match="has a different shape"):
            fuse_lora(Target(), state, "transformer")
