"""Builds Gemma 4's RoPE cos / sin rows **from the formula** (the Python mirror of the TS source of
truth), and derives their declaration.

The RoPE tables are no longer baked into the graph (they are **derived inputs**, not initializers —
the module docstring of `gemma4.export_decode`). The host builds the tables (on the TS side,
`packages/models/src/gemma/rope.ts`), and the distribution declares not the tables themselves but
only **the parameters of the formula** (theta / headDim / rotaryDim per layer type) and the
position limit. This module owns two things:

- {@link rope_specs} — **derives** the declaration (`pipelineConfig.rope`) from the upstream
  config. Copying the values by hand would leave only the declaration stale on the day the
  checkpoint is replaced.
- {@link rope_rows} — builds cos / sin rows from that declaration. **The same formula** as the TS
  side, computed in f64 and only stored as f32. The tables the scripts use for goldens / io /
  example inputs are built by this path (taken with the same formula as the tables the
  distribution reads — the acceptance gate never reads an expected sequence taken with another
  table).

MUST: this is not bit-identical to upstream. Upstream's rotary runs inv_freq, the angles and cos /
sin **all in f32** (modeling_gemma4.py:1170-1179), so it differs from values computed in f64 and
then rounded from the last ULP on (the larger the position, the more the f32 representation error
of the angle matters; measured over a full sweep of positions 0..131,071 the maximum is 9.44e-3
(at ≈129,293), exactly at position 131,071 it is 4.77e-3, and the worst ratio to the tolerance
band is 0.756). The comparison uses a tolerance proportional to the position
({@link gemma4.tests.test_rope}) — tightening it to "bit-exact" would make a gate that fails the
correct implementation.

MUST: do not import torch. `gemma4.distribution` (the distribution recipe) reads this module, so a
heavy dependency would make the distribution side's import, and the collection of the CI jobs that
need no real weights, heavier by that much. The formula can be written with numpy alone, so there
is no reason to pay for it. NOTE: no gate checks this (torch is a required dependency of the core
`karume` and is always installed, so it is outside the scope of
`tests/test_optional_group_imports.py`).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

#: Spellings of the layer types (the vocabulary of upstream `config.layer_types`). `gemma4.export`
#: re-exports them (the same string is not spelled in two places).
FULL_ATTENTION = "full_attention"
SLIDING_ATTENTION = "sliding_attention"

#: Accepted rope_types. `default` uses every frequency; with `proportional` only the first
#: `rotaryDim/2` are non-zero and the rest are zero frequencies (= columns of cos 1 / sin 0).
#: Anything else is a different formula, so it fails loudly.
DEFAULT_ROPE_TYPE = "default"
PROPORTIONAL_ROPE_TYPE = "proportional"
ROPE_TYPES = (DEFAULT_ROPE_TYPE, PROPORTIONAL_ROPE_TYPE)

#: Fragment of the initializer names left by the retired "bake the tables into the graph" form
#: (`…rotary_emb.<layer type>_cos_table`). The source of truth that lets the baking side
#: ({@link gemma4.export_decode.assert_rope_inputs}) and the distributing side
#: ({@link gemma4.distribution.assert_gemma4_graph}) reject the residue with **the same spelling**.
BAKED_TABLE_INFIX = "rotary_emb."

#: Field names of `pipelineConfig.rope` (spelled as `gemma.parsePipelineConfig` on the TS side).
THETA_FIELD = "theta"
HEAD_DIM_FIELD = "headDim"
ROTARY_DIM_FIELD = "rotaryDim"


class RopeSpecError(ValueError):
    """The RoPE declaration cannot be derived from the upstream config (an unknown rope_type, a
    coefficient that cannot be mirrored)."""


@dataclass(frozen=True)
class RopeLayerSpec:
    """The RoPE formula of one layer type (the contents of `pipelineConfig.rope.<layer type>`).

    `head_dim` is the width of one row (the last dimension of cos / sin), `rotary_dim` the width
    that **actually rotates**. The rest beyond `rotary_dim < head_dim` is zero frequency and enters
    the row as constant columns of cos = 1 / sin = 0 (the `proportional` of the full layers — the
    nope part of upstream `_compute_proportional_rope_parameters`).
    """

    theta: float
    head_dim: int
    rotary_dim: int

    def __post_init__(self) -> None:
        if self.theta <= 1.0:
            raise RopeSpecError(f"theta {self.theta} is not greater than 1")
        for name, value in (("headDim", self.head_dim), ("rotaryDim", self.rotary_dim)):
            if value < 2 or value % 2:
                raise RopeSpecError(f"{name} {value} is not an even number of at least 2")
        if self.rotary_dim > self.head_dim:
            raise RopeSpecError(f"rotaryDim {self.rotary_dim} exceeds headDim {self.head_dim}")

    def declaration(self) -> dict[str, Any]:
        """The form placed in `pipelineConfig.rope` (field names in the TS side's spelling)."""
        return {
            THETA_FIELD: float(self.theta),
            HEAD_DIM_FIELD: int(self.head_dim),
            ROTARY_DIM_FIELD: int(self.rotary_dim),
        }


def _rope_parameters(config: Any, layer_type: str) -> Mapping[str, Any]:
    parameters = getattr(config, "rope_parameters", None)
    if not isinstance(parameters, Mapping):
        raise RopeSpecError(f"config.rope_parameters is not an object ({parameters!r})")
    entry = parameters.get(layer_type)
    if not isinstance(entry, Mapping):
        raise RopeSpecError(f"config.rope_parameters['{layer_type}'] is missing ({entry!r})")
    return entry


def _head_dim(config: Any, layer_type: str, rope_type: str) -> int:
    """Decide head_dim from **the same source as upstream** (the `head_dim_key` branch — lines
    1106-1120).

    Only a full layer with `proportional` reads `global_head_dim`. Without matching this, the full
    layers' table would be built 256 wide and would not fit attention's 512 width.
    """
    key = (
        "global_head_dim"
        if layer_type == FULL_ATTENTION and rope_type == PROPORTIONAL_ROPE_TYPE
        else "head_dim"
    )
    declared = getattr(config, key, None)
    if not declared:
        declared = int(config.hidden_size) // int(config.num_attention_heads)
    return int(declared)


def layer_spec(config: Any, layer_type: str) -> RopeLayerSpec:
    """Derive the declaration of one layer type from the upstream config (not a single value is
    copied by hand).

    MUST: a rope_type / `factor` / `attention_factor` outside what is accepted fails loudly.
    Copying only theta and the widths when the formula is different would make the host build the
    table with **different angles that still match in shape and type** (a silent wrong value).
    """
    parameters = _rope_parameters(config, layer_type)
    rope_type = parameters.get("rope_type", DEFAULT_ROPE_TYPE)
    if rope_type not in ROPE_TYPES:
        raise RopeSpecError(
            f"layer type '{layer_type}': rope_type '{rope_type}' — mirrorable: {list(ROPE_TYPES)}"
        )
    for name in ("factor", "attention_factor"):
        value = parameters.get(name, 1.0)
        if float(value) != 1.0:
            raise RopeSpecError(
                f"layer type '{layer_type}' has {name} {value} — not 1 means another formula"
            )
    head_dim = _head_dim(config, layer_type, rope_type)
    if rope_type == PROPORTIONAL_ROPE_TYPE:
        proportion = float(parameters.get("partial_rotary_factor", 1.0))
        rotary_dim = 2 * int(proportion * head_dim // 2)
    else:
        rotary_dim = head_dim
    theta = parameters.get("rope_theta")
    if not isinstance(theta, int | float) or isinstance(theta, bool):
        raise RopeSpecError(
            f"layer type '{layer_type}' has a rope_theta that is not a number ({theta!r})"
        )
    return RopeLayerSpec(theta=float(theta), head_dim=head_dim, rotary_dim=rotary_dim)


def rope_specs(config: Any) -> dict[str, RopeLayerSpec]:
    """The declarations of the layer types that appear in `config.layer_types` (**in order of
    appearance**)."""
    layer_types = getattr(config, "layer_types", None)
    if not isinstance(layer_types, Sequence) or isinstance(layer_types, str) or not layer_types:
        raise RopeSpecError(f"config.layer_types is not a non-empty array ({layer_types!r})")
    return {
        layer_type: layer_spec(config, layer_type)
        for layer_type in dict.fromkeys(str(name) for name in layer_types)
    }


def inverse_frequencies(spec: RopeLayerSpec) -> np.ndarray:
    """Frequency seeds `[headDim/2]` (f64). Only the first `rotaryDim/2` are non-zero.

    `invFreq[i] = theta ** (-(2i) / headDim)`. Upstream writes `1.0 / theta ** ((2i)/headDim)`,
    but in f64 the direction of the division differs only in the last ULP (and bit-exactness does
    not hold anyway — module docstring), so this matches **the spelling** of the TS source of truth.
    """
    half = spec.head_dim // 2
    frequencies = np.zeros(half, dtype=np.float64)
    rotated = spec.rotary_dim // 2
    index = np.arange(rotated, dtype=np.float64)
    frequencies[:rotated] = float(spec.theta) ** (-(2.0 * index) / float(spec.head_dim))
    return frequencies


def rope_rows(spec: RopeLayerSpec, positions: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    """Positions → `(cos, sin)` rows (f32 `[len(positions), headDim]`).

    The angle is `pos × invFreq`, and a row is `cat(freqs, freqs)` (the first and second halves are
    identical — upstream `emb = torch.cat((freqs, freqs), dim=-1)`). `attention_scaling` is 1 for
    both accepted rope_types, so it is not applied (any value but 1 is rejected by
    {@link layer_spec}).

    MUST: compute in f64 and **round only once, at the end**. Holding the angles in f32 makes the
    error grow with the position (upstream is the side that does that; this one does not follow it
    — module docstring).
    """
    angles = np.asarray(positions, dtype=np.float64).reshape(-1, 1) * inverse_frequencies(spec)
    doubled = np.concatenate((angles, angles), axis=1)
    return np.cos(doubled).astype(np.float32), np.sin(doubled).astype(np.float32)
