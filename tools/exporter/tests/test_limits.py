"""Unit tests of the derivation of a quant's `requiredLimits` (`karume.limits`).

Only the **pure-function layer** is covered here — the wiring that walks the entry points from the
real thing (the series containers) is `TestRequiredLimits` in `test_dist.py`. Here "what counts as
demand" and "from where a value is written" are pinned with synthetic declarations alone.
"""

from __future__ import annotations

from typing import Any, ClassVar

import pytest

from karume.limits import (
    WEBGPU_DEFAULT_LIMITS,
    LimitsError,
    max_state_slot_bytes,
    required_limits,
    state_bindings,
)

#: The guaranteed defaults of the WebGPU spec (aliases for readability — the check of the values is
#: {@link TestWebgpuDefaults}).
_BINDING = WEBGPU_DEFAULT_LIMITS["maxStorageBufferBindingSize"]
_BUFFER = WEBGPU_DEFAULT_LIMITS["maxBufferSize"]


def _tensor(begin: int, end: int, dtype: str = "F32") -> dict[str, Any]:
    """One tensor of a safetensors header (the derivation reads only `data_offsets`)."""
    return {"dtype": dtype, "shape": [end - begin], "data_offsets": [begin, end]}


class TestWebgpuDefaults:
    """The defaults are **a copy of an external spec**, so the values themselves are the check."""

    def test_the_defaults_are_the_two_size_limits_of_the_spec(self) -> None:
        """Off by even one byte, the "write / do not write" boundary drifts from real devices."""
        assert WEBGPU_DEFAULT_LIMITS == {
            "maxBufferSize": 268_435_456,
            "maxStorageBufferBindingSize": 134_217_728,
        }

    def test_it_names_no_workgroup_limit(self) -> None:
        """The workgroup limits are decided by the kernel design and cannot be derived from the
        distribution (fields that are not written)."""
        assert not any(name.startswith("maxCompute") for name in WEBGPU_DEFAULT_LIMITS)


class TestRequiredLimitsSelection:
    """Only **the fields that exceed the guaranteed defaults** are written (the semantics "no field
    = runs on default-spec devices")."""

    def test_a_demand_within_both_defaults_declares_nothing(self) -> None:
        assert required_limits(1024) == {}

    def test_a_demand_exactly_at_the_binding_default_declares_nothing(self) -> None:
        """The boundary (exactly the default) is satisfied — writing it into the requirement adds a
        field that constrains nothing."""
        assert required_limits(_BINDING) == {}

    def test_a_demand_past_the_binding_default_declares_only_the_binding(self) -> None:
        """The two have different defaults, so in the 128MiB–256MiB band only the binding becomes a
        requirement."""
        demand = _BINDING + 1

        assert required_limits(demand) == {"maxStorageBufferBindingSize": demand}

    def test_a_demand_exactly_at_the_buffer_default_still_leaves_the_buffer_out(self) -> None:
        """The buffer-side boundary (= exactly the default) is not written either. The binding side
        is exceeded, so it stays."""
        assert required_limits(_BUFFER) == {"maxStorageBufferBindingSize": _BUFFER}

    def test_a_demand_past_both_defaults_declares_both_with_the_same_value(self) -> None:
        """Weights and state are both 1 buffer = 1 binding, so the two required values are the
        same."""
        demand = _BUFFER + 1

        assert required_limits(demand) == {
            "maxBufferSize": demand,
            "maxStorageBufferBindingSize": demand,
        }

    def test_the_declared_names_keep_a_stable_order(self) -> None:
        """A rebuild does not reorder the fields (the manifest bytes do not move without a
        reason)."""
        assert list(required_limits(_BUFFER + 1)) == [
            "maxBufferSize",
            "maxStorageBufferBindingSize",
        ]

    def test_it_refuses_a_negative_demand(self) -> None:
        with pytest.raises(LimitsError, match="negative"):
            required_limits(-1)


class TestWhatTheDeclarationDoesNotCover:
    """The declaration guarantees only the **resident part** (stored tensors and state slots) — not
    the intermediates.

    The source of truth for the rule and its reason is the MUST in the module docstring of
    `karume.limits`. The actual demand after fusion depends on the limits the device actually
    granted (the row-blocking step is set there), so it is in principle not determined by the
    distribution, and writing the largest node output **before** fusion would wrongly reject
    "devices that run with row blocks" before the download. Forms whose intermediates exceed the
    limits are caught by the run-time check at Session construction.

    This test names the ruling of 2026-09-04 ("the resident part only") **itself** — flipping it
    (counting the intermediates too) changes the expected value below from `{}` to one with both
    limits.
    """

    #: The declared shape of an intermediate node output (960 MiB in f32 = 7.5 times the binding
    #: default; the birefnet 1024² shape).
    _INTERMEDIATE_SHAPE: ClassVar[list[int]] = [1, 240, 1024, 1024]

    def test_a_graph_whose_intermediates_dwarf_its_weights_declares_nothing(self) -> None:
        """Even with node outputs at 7.5 times the default, a graph without state has a demand of 0
        (no field appears)."""
        graph = {
            "values": {"cat": {"dtype": "f32", "shape": self._INTERMEDIATE_SHAPE}},
            "nodes": [{"op": "cat", "ins": ["w0", "w1"], "outs": ["cat"], "attrs": {}}],
        }

        demand = max_state_slot_bytes(graph, {}, "where")

        assert demand == 0
        assert required_limits(demand) == {}

    def test_the_same_number_would_be_declared_if_it_came_from_a_stored_tensor(self) -> None:
        """The check against a tautological test — the value is dropped not "because it is small"
        but "because it is not at an entry point".

        The same byte count appearing as a **stored** tensor writes both fields. The case above
        gives `{}` because node outputs are at neither entry point of the demand (the header and
        the state declaration).
        """
        intermediate = 240 * 1024 * 1024 * 4

        assert intermediate == 960 * 1024 * 1024
        assert intermediate > _BUFFER
        assert required_limits(intermediate) == {
            "maxBufferSize": intermediate,
            "maxStorageBufferBindingSize": intermediate,
        }


class TestStateSlotDemand:
    """The second source of demand — the state slots that **stay resident for the whole capacity**
    (ADR 0066 decision 2)."""

    #: The binding is taken from **the largest capacity the distribution allows** (`maxPosition`)
    #: — the default capacity (`capacity`) is a run-time knob the host can override, so writing
    #: from it would give an insufficient declaration.
    _CONFIG: ClassVar[dict[str, Any]] = {"chunkLength": 32, "capacity": 640, "maxPosition": 4096}

    def test_a_graph_without_states_has_no_state_demand(self) -> None:
        """Every existing model without states is unaffected (the field itself is absent)."""
        assert max_state_slot_bytes({"symbols": [], "nodes": []}, self._CONFIG, "where") == 0

    def test_a_numeric_slot_is_counted_as_f32_elements(self) -> None:
        """A sliding slot holds the window width as a number (f32 = 4 bytes/element)."""
        graph = {"states": {"l0.k": {"dtype": "f32", "shape": [1, 2, 512, 256]}}}

        assert max_state_slot_bytes(graph, self._CONFIG, "where") == 2 * 512 * 256 * 4

    def test_a_symbolic_slot_is_bound_by_the_pipeline_config_max_position(self) -> None:
        """A full slot is written with the capacity symbol as is — the value it is evaluated with is
        the **largest** capacity the distribution allows."""
        graph = {"states": {"l4.k": {"dtype": "f32", "shape": [1, 1, "C", 512]}}}

        assert max_state_slot_bytes(graph, self._CONFIG, "where") == 4096 * 512 * 4

    def test_the_default_capacity_is_not_what_binds_the_symbol(self) -> None:
        """MUST: writing from the default capacity gives a declaration that "fails the moment a
        capacity larger than the default is chosen"."""
        graph = {"states": {"l4.k": {"dtype": "f32", "shape": [1, 1, "C", 512]}}}

        demand = max_state_slot_bytes(graph, self._CONFIG, "where")

        assert demand != self._CONFIG["capacity"] * 512 * 4

    def test_it_evaluates_a_derived_dimension(self) -> None:
        """The derived form `coeff·sym+offset` is also bound and counted (the dimension language of
        ADR 0057)."""
        graph = {"states": {"s": {"dtype": "f32", "shape": ["2C+8"]}}}

        assert max_state_slot_bytes(graph, self._CONFIG, "where") == (2 * 4096 + 8) * 4

    def test_it_takes_the_largest_slot_of_the_graph(self) -> None:
        """Each slot is a separate buffer — the binding / buffer demand is the largest, not the
        sum."""
        graph = {
            "states": {
                "sliding": {"dtype": "f32", "shape": [1, 1, 512, 256]},
                "full": {"dtype": "f32", "shape": [1, 1, "C", 256]},
            }
        }

        assert max_state_slot_bytes(graph, self._CONFIG, "where") == 4096 * 256 * 4

    def test_a_large_capacity_outgrows_any_single_tensor(self) -> None:
        """Weights split at the block limit, but a slot stays one buffer — state dominates in long
        conversations.

        gemma4's full slot (`[1, 1, C, 512]`) counted at capacity 131,072 is exactly 256MiB = the
        buffer default itself, which falls into the band that exceeds only the binding default
        (128MiB).
        """
        graph = {"states": {"full": {"dtype": "f32", "shape": [1, 1, "C", 512]}}}

        demand = max_state_slot_bytes(graph, {"maxPosition": 131_072}, "where")

        assert demand == 131_072 * 512 * 4 == _BUFFER
        assert required_limits(demand) == {"maxStorageBufferBindingSize": demand}

    def test_the_gemma4_state_stays_under_its_int8_embedding(self) -> None:
        """gemma4 E2B at actual size: the full slot at the limit of 131,072 is smaller than the main
        embedding.

        The reason this family's written values do not move when the binding moves from the default
        capacity (`capacity`) to the limit (`maxPosition`). The full slot is `[1, 1, C, 512]` in f32
        = 256MiB; the main embedding (the same object as the tied lm_head) is 262,144 × 1,536 =
        384MiB in i8 storage — the demand stays the latter, so both fields of `requiredLimits` stay
        at 384MiB.
        """
        graph = {"states": {"full": {"dtype": "f32", "shape": [1, 1, "C", 512]}}}

        state = max_state_slot_bytes(graph, {"maxPosition": 131_072}, "where")
        embedding = 262_144 * 1_536  # i8 = 1 byte/element

        assert state == 256 * 1024**2
        assert embedding == 384 * 1024**2
        assert embedding > state
        assert required_limits(embedding) == {
            "maxBufferSize": embedding,
            "maxStorageBufferBindingSize": embedding,
        }

    @pytest.mark.parametrize(
        "config",
        [
            {},
            {"maxPosition": 0},
            {"maxPosition": "640"},
            {"maxPosition": True},
            # A distribution with only the default capacity (a generation that declares no
            # limit) is refused too.
            {"capacity": 640},
        ],
    )
    def test_it_refuses_a_capacity_the_distribution_does_not_pin(self, config: Any) -> None:
        """Without a binding, fail loudly — silently leaving the state out gives "declared, yet
        insufficient"."""
        graph = {"states": {"full": {"dtype": "f32", "shape": [1, 1, "C", 512]}}}

        with pytest.raises(LimitsError, match="maxPosition"):
            max_state_slot_bytes(graph, config, "where")

    def test_it_refuses_two_symbols_it_cannot_tell_apart(self) -> None:
        """There is one capacity symbol — with two, the distribution cannot tell which one is the
        capacity."""
        states = {
            "a": {"dtype": "f32", "shape": [1, "C", 8]},
            "b": {"dtype": "f32", "shape": [1, "D", 8]},
        }

        with pytest.raises(LimitsError, match="several"):
            state_bindings(states, self._CONFIG, "where")

    def test_a_numeric_only_graph_needs_no_binding(self) -> None:
        """Without a symbol no capacity field is needed (a family without a pipelineConfig passes
        too)."""
        states = {"a": {"dtype": "f32", "shape": [1, 512, 8]}}

        assert state_bindings(states, {}, "where") == {}

    def test_it_refuses_a_slot_dtype_it_cannot_size(self) -> None:
        """f16 is reserved as an option only (ADR 0066, addendum 2026-08-17 item 5) — the size is
        not guessed."""
        graph = {"states": {"s": {"dtype": "f16", "shape": [1, 512]}}}

        with pytest.raises(LimitsError, match="dtype"):
            max_state_slot_bytes(graph, self._CONFIG, "where")

    @pytest.mark.parametrize("shape", [[], "512", [0, 8], [1, 1.5], [1, None]])
    def test_it_refuses_a_shape_that_is_not_a_concrete_capacity(self, shape: Any) -> None:
        graph = {"states": {"s": {"dtype": "f32", "shape": shape}}}

        with pytest.raises(LimitsError):
            max_state_slot_bytes(graph, self._CONFIG, "where")
