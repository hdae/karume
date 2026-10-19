"""Regression tests for RoPE buffer demotion and the check that fails on no match (`karume.rope`).

The subject is a model-independent scan, so synthetic nn.Modules alone pin it (no upstream package
needed — recovered from the Anima patch layer at ADR 0065 stage 2).
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from karume import rope


class TestLiftRopeBuffers:
    def test_it_moves_inv_freq_out_of_the_buffers(self) -> None:
        """Demotes `inv_freq` from a buffer to a plain attribute (a leaf for constant folding)."""

        class Rotary(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.register_buffer("inv_freq", torch.arange(4.0), persistent=False)

        root = nn.Sequential(Rotary())

        lifted = rope.lift_rope_buffers(root)

        assert lifted == 1
        assert "inv_freq" not in dict(root.named_buffers())
        assert torch.equal(root[0].inv_freq, torch.arange(4.0))

    def test_it_matches_layer_type_prefixed_names(self) -> None:
        """Also demotes names with a layer_type prefix (Gemma3 has `<layer_type>_inv_freq`).

        The detector behind the suffix match. Reverting to an exact match makes this find 0, and
        sin / cos stay in the IR when that model is exported.
        """

        class LayeredRotary(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                for name in (
                    "sliding_attention_inv_freq",
                    "full_attention_inv_freq",
                    "sliding_attention_original_inv_freq",
                    "full_attention_original_inv_freq",
                ):
                    self.register_buffer(name, torch.arange(4.0), persistent=False)
                # Buffers without the suffix are out of scope (the scan does not pick up
                # unrelated constants).
                self.register_buffer("attention_scale", torch.ones(1), persistent=False)

        root = LayeredRotary()

        lifted = rope.lift_rope_buffers(root)

        assert lifted == 4
        assert list(dict(root.named_buffers())) == ["attention_scale"]

    def test_a_model_without_rope_buffers_is_rejected(self) -> None:
        """A scan that finds nothing fails (the check against a tautological pass — it would break
        silently when an upstream attribute name changes)."""
        with pytest.raises(ValueError, match=r"no RoPE buffer .* was found"):
            rope.assert_rope_lifted(nn.Linear(2, 2), "test")


class Rotary(nn.Module):
    """Synthetic module with exactly one `inv_freq` buffer."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("inv_freq", torch.arange(4.0), persistent=False)


class TestAssertRopeLiftedIsIdempotent:
    """The check fails only when "no RoPE frequency seed exists, neither as a buffer nor as a plain
    attribute".

    Demotion is the destructive `_buffers.pop`, so judging by "how many were demoted" would always
    give 0 on a second pass over the same module, and export would stop with a false "the attribute
    name changed upstream".
    """

    def test_applying_it_twice_to_the_same_module_still_passes(self) -> None:
        """A second pass passes with 0 buffers left (the first only moved them to attributes)."""
        root = nn.Sequential(Rotary())

        rope.assert_rope_lifted(root, "test")
        rope.assert_rope_lifted(root, "test")

        assert "inv_freq" not in dict(root.named_buffers())

    def test_a_child_passes_after_the_parent_was_lifted(self) -> None:
        """Passes on a child after the parent (the parent's scan already demoted the child)."""
        child = Rotary()
        parent = nn.Sequential(child)

        rope.assert_rope_lifted(parent, "parent")
        rope.assert_rope_lifted(child, "child")

    def test_a_module_that_never_had_rope_buffers_is_still_rejected(self) -> None:
        """Idempotence keeps the negative side alive (detects a check that became tautological)."""
        with pytest.raises(ValueError, match=r"no RoPE buffer .* was found"):
            rope.assert_rope_lifted(nn.Sequential(nn.Linear(2, 2)), "test")
