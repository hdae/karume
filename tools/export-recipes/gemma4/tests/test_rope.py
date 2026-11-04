"""The RoPE formula (`gemma4/rope.py`) and its comparison with the upstream implementation within
a **position-proportional tolerance**.

The RoPE tables are no longer baked into the graph (the host builds them from the declaration), so
the basis of correctness moved from "the table is upstream's output itself" to "**the formula
produces the same angles as upstream**". This file pins those two:

- {@link gemma4.rope.rope_specs} derives theta / headDim / rotaryDim from the upstream config
  (including the branch where only full layers read `global_head_dim`), and fails loudly on a
  declaration it cannot mirror
- {@link gemma4.rope.rope_rows} matches the actual output of upstream `Gemma4TextRotaryEmbedding`
  within a tolerance proportional to the position

MUST: the comparison target is **the actual output of the upstream module** (not the same formula
written twice). That it is not vacuous is backed by fault injections (shifted position, swapped
layer type, replaced theta, replaced rotaryDim).

MUST: no bit-exact comparison. Upstream runs inv_freq, the angles and cos / sin all in f32, so it
always differs from values built in f64 and then rounded (the module docstring of `gemma4.rope`).
The tolerance is {@link atol_for} — a constant term of a few times the 1 ULP of cos / sin
(5.96e-8), and a position-proportional term set with about 2× headroom over the measured "f32
representation error of the angle × derivative" (4.8e-3 at P=131,071).

The cases that call upstream need transformers, so they SKIP through `importorskip` (the two-job
setup of ADR 0065).
"""

from __future__ import annotations

import base64
import json
from collections.abc import Sequence
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from gemma4 import rope
from gemma4.tests import rope_fixture

#: The two terms of the tolerance (constant = a few ULPs of implementation difference /
#: position-proportional = the f32 representation error of the angle).
ATOL_BASE = 2.5e-7
ATOL_PER_POSITION = 1.2e-7

#: The positions compared. Besides the scattered points ({@link rope_fixture.FIXTURE_POSITIONS}),
#: **the contiguous 0..64** are checked too — scattered points alone could let through a form that
#: "matches only at the start and the limit" (frequencies in reverse order, for example).
CONTIGUOUS_POSITIONS: tuple[int, ...] = tuple(range(65))

#: The fields that tie the hand-copied declaration ({@link rope_fixture.E2B_TEXT_CONFIG_FIELDS})
#: to the real checkpoint. Only these 4 affect the RoPE angles; the other fields are scaffolding
#: for building the rotary module (layer count, hidden_size and so on) and do not affect the
#: cos / sin values.
UPSTREAM_ROPE_FIELDS: tuple[str, ...] = (
    "max_position_embeddings",
    "head_dim",
    "global_head_dim",
    "rope_parameters",
)


def atol_for(position: int) -> float:
    """The tolerance at one position (it widens in proportion to the position)."""
    return ATOL_BASE + position * ATOL_PER_POSITION


def assert_rope_parity(
    got: np.ndarray, want: np.ndarray, positions: Sequence[int], where: str
) -> None:
    """Compares element by element within `atol(pos)` (the worst element goes into the
    diagnostic)."""
    if got.shape != want.shape:
        raise AssertionError(f"{where}: shape is {got.shape} — upstream's is {want.shape}")
    tolerance = np.array([atol_for(position) for position in positions], dtype=np.float64)
    difference = np.abs(got.astype(np.float64) - want.astype(np.float64))
    excess = difference / tolerance[:, None]
    row, column = np.unravel_index(int(excess.argmax()), excess.shape)
    if excess[row, column] > 1.0:
        raise AssertionError(
            f"{where}: position {positions[row]}, column {column} differs by"
            f" {difference[row, column]:.3e} (tolerance {tolerance[row]:.3e})"
        )


@pytest.fixture
def e2b_config():
    pytest.importorskip("transformers")
    return rope_fixture.e2b_text_config()


class TestRopeSpecs:
    """Deriving the declaration — no value is copied by hand, and a declaration that cannot be
    mirrored fails."""

    def test_it_derives_both_layer_types_from_the_upstream_config(self, e2b_config) -> None:
        specs = rope.rope_specs(e2b_config)

        assert specs[rope.SLIDING_ATTENTION] == rope.RopeLayerSpec(
            theta=10000.0, head_dim=256, rotary_dim=256
        )
        # Only full layers read `global_head_dim` (512), and `proportional` rotates only the first
        # 128.
        assert specs[rope.FULL_ATTENTION] == rope.RopeLayerSpec(
            theta=1000000.0, head_dim=512, rotary_dim=128
        )

    def test_the_order_follows_the_layer_type_order(self, e2b_config) -> None:
        """Order of appearance (sliding first) — the same basis as the order of the graph inputs."""
        assert list(rope.rope_specs(e2b_config)) == [
            rope.SLIDING_ATTENTION,
            rope.FULL_ATTENTION,
        ]

    def test_it_falls_back_to_the_derived_head_dim(self) -> None:
        """A config without `head_dim` uses `hidden_size / num_attention_heads` (the same as
        upstream)."""
        config = SimpleNamespace(
            hidden_size=64,
            num_attention_heads=4,
            layer_types=[rope.SLIDING_ATTENTION],
            rope_parameters={rope.SLIDING_ATTENTION: {"rope_type": "default", "rope_theta": 1e4}},
        )

        assert rope.rope_specs(config)[rope.SLIDING_ATTENTION].head_dim == 16

    @pytest.mark.parametrize(
        ("override", "message"),
        [
            ({"rope_type": "yarn"}, "rope_type"),
            ({"factor": 2.0}, "factor"),
            ({"attention_factor": 0.5}, "attention_factor"),
            ({"rope_theta": "10000"}, "rope_theta"),
        ],
    )
    def test_it_refuses_a_declaration_it_cannot_mirror(
        self, override: dict[str, object], message: str
    ) -> None:
        """MUST: mirroring only theta and the width when the formula is a different one would make
        the host build the table with other angles."""
        config = SimpleNamespace(
            head_dim=16,
            hidden_size=64,
            num_attention_heads=4,
            layer_types=[rope.SLIDING_ATTENTION],
            rope_parameters={
                rope.SLIDING_ATTENTION: {
                    "rope_type": "default",
                    "rope_theta": 1e4,
                    **override,
                }
            },
        )

        with pytest.raises(rope.RopeSpecError, match=message):
            rope.rope_specs(config)

    def test_it_refuses_a_config_without_layer_types(self) -> None:
        with pytest.raises(rope.RopeSpecError, match="layer_types"):
            rope.rope_specs(SimpleNamespace(layer_types=[]))

    @pytest.mark.parametrize(
        ("theta", "head_dim", "rotary_dim"),
        [(1.0, 8, 8), (1e4, 7, 6), (1e4, 8, 10)],
    )
    def test_the_spec_itself_refuses_impossible_numbers(
        self, theta: float, head_dim: int, rotary_dim: int
    ) -> None:
        with pytest.raises(rope.RopeSpecError):
            rope.RopeLayerSpec(theta=theta, head_dim=head_dim, rotary_dim=rotary_dim)


class TestInverseFrequencies:
    def test_the_nope_tail_is_exactly_zero(self) -> None:
        """The zero frequencies of `proportional` are the slots that become constant columns of
        cos = 1 / sin = 0 (192 of them on full layers)."""
        spec = rope.RopeLayerSpec(theta=1e6, head_dim=512, rotary_dim=128)

        frequencies = rope.inverse_frequencies(spec)

        assert frequencies.shape == (256,)
        assert np.count_nonzero(frequencies) == 64
        assert not frequencies[64:].any()

    def test_the_row_repeats_its_first_half(self) -> None:
        """A row is `cat(freqs, freqs)` (the same order as upstream) — the two halves are
        identical."""
        spec = rope.RopeLayerSpec(theta=1e4, head_dim=16, rotary_dim=16)

        cos, sin = rope.rope_rows(spec, [7])

        assert cos.shape == (1, 16)
        assert np.array_equal(cos[0, :8], cos[0, 8:])
        assert np.array_equal(sin[0, :8], sin[0, 8:])

    def test_position_zero_is_the_identity_rotation(self) -> None:
        cos, sin = rope.rope_rows(rope.RopeLayerSpec(theta=1e4, head_dim=8, rotary_dim=8), [0])

        assert np.array_equal(cos[0], np.ones(8, dtype=np.float32))
        assert np.array_equal(sin[0], np.zeros(8, dtype=np.float32))

    def test_the_rows_are_stored_as_float32(self) -> None:
        """Only the storage is f32 (the computation is f64) — the same contract as the TS side
        passing f32 tables."""
        cos, sin = rope.rope_rows(rope.RopeLayerSpec(theta=1e4, head_dim=8, rotary_dim=8), [1, 2])

        assert cos.dtype == np.float32
        assert sin.dtype == np.float32


class TestUpstreamParity:
    """Comparison with the upstream implementation (the fault injections below back that it is
    **not vacuous**)."""

    @pytest.fixture
    def upstream(self, e2b_config):
        def tables(positions: Sequence[int]):
            return rope_fixture.upstream_tables(e2b_config, positions)

        return tables

    @pytest.mark.parametrize(
        "positions",
        [rope_fixture.FIXTURE_POSITIONS, CONTIGUOUS_POSITIONS],
        ids=["scattered", "contiguous"],
    )
    @pytest.mark.parametrize(
        "layer_type", [rope.SLIDING_ATTENTION, rope.FULL_ATTENTION], ids=["sliding", "full"]
    )
    def test_the_mirror_matches_the_upstream_module(
        self, e2b_config, upstream, layer_type: str, positions: Sequence[int]
    ) -> None:
        spec = rope.rope_specs(e2b_config)[layer_type]
        want_cos, want_sin = upstream(positions)[layer_type]

        cos, sin = rope.rope_rows(spec, positions)

        assert_rope_parity(cos, want_cos, positions, f"{layer_type} cos")
        assert_rope_parity(sin, want_sin, positions, f"{layer_type} sin")

    def test_the_tolerance_is_not_slack(self, e2b_config, upstream) -> None:
        """Pins how many times the measurement the tolerance is — too loose and the fault injections
        pass as well.

        What is checked is the worst ratio over the fixture's 13 points (measured 0.548 = 1.8×
        headroom, at P=511 of full sin). The upper bound 0.9 is the line "below 1.1× headroom,
        upstream's f32 error is larger than assumed", and the worst ratio over a full sweep of
        positions 0..131,071, 0.756 (1.3× headroom), is inside it too. The lower bound 0.01
        rejects the opposite, too-loose side (100× headroom). {@link ATOL_PER_POSITION} is a
        coefficient taken from an **upper bound** of the angle's f32 representation error, so do
        not move it because of this ratio (to move it, retake the premise instead).
        """
        positions = rope_fixture.FIXTURE_POSITIONS
        worst = 0.0
        for layer_type, spec in rope.rope_specs(e2b_config).items():
            want_cos, want_sin = upstream(positions)[layer_type]
            cos, sin = rope.rope_rows(spec, positions)
            for got, want in ((cos, want_cos), (sin, want_sin)):
                difference = np.abs(got.astype(np.float64) - want.astype(np.float64))
                for row, position in enumerate(positions):
                    worst = max(worst, float(difference[row].max()) / atol_for(position))

        assert 0.01 < worst < 0.9, f"the ratio of the measurement to the tolerance is {worst}"

    def test_a_position_shifted_by_one_is_rejected(self, e2b_config, upstream) -> None:
        """Fault injection: shift the position by 1 (the same breakage as a "table with shifted
        rows" in the table-lookup era)."""
        positions = rope_fixture.FIXTURE_POSITIONS
        spec = rope.rope_specs(e2b_config)[rope.SLIDING_ATTENTION]
        want_cos, _ = upstream(positions)[rope.SLIDING_ATTENTION]

        cos, _ = rope.rope_rows(spec, [position + 1 for position in positions])

        with pytest.raises(AssertionError):
            assert_rope_parity(cos, want_cos, positions, "shifted")

    def test_a_table_built_for_the_other_layer_type_is_rejected(self, e2b_config, upstream) -> None:
        """Fault injection: swap the layer type (theta and the width both differ, so it fails at the
        shape stage)."""
        positions = rope_fixture.FIXTURE_POSITIONS
        specs = rope.rope_specs(e2b_config)
        want_cos, _ = upstream(positions)[rope.FULL_ATTENTION]

        cos, _ = rope.rope_rows(specs[rope.SLIDING_ATTENTION], positions)

        with pytest.raises(AssertionError, match="shape is"):
            assert_rope_parity(cos, want_cos, positions, "swapped")

    def test_a_wrong_theta_is_rejected(self, e2b_config, upstream) -> None:
        """Fault injection: full-layer theta 1e6 → 1e5 (the width and the order still match, only
        the angles differ)."""
        positions = rope_fixture.FIXTURE_POSITIONS
        spec = replace(rope.rope_specs(e2b_config)[rope.FULL_ATTENTION], theta=1e5)
        want_cos, _ = upstream(positions)[rope.FULL_ATTENTION]

        cos, _ = rope.rope_rows(spec, positions)

        with pytest.raises(AssertionError):
            assert_rope_parity(cos, want_cos, positions, "theta")

    def test_a_full_rotary_dim_on_the_partial_layer_is_rejected(self, e2b_config, upstream) -> None:
        """Fault injection: full-layer rotaryDim set to headDim (the 192 zero frequencies start
        rotating)."""
        positions = rope_fixture.FIXTURE_POSITIONS
        spec = rope.rope_specs(e2b_config)[rope.FULL_ATTENTION]
        broken = replace(spec, rotary_dim=spec.head_dim)
        want_cos, _ = upstream(positions)[rope.FULL_ATTENTION]

        cos, _ = rope.rope_rows(broken, positions)

        with pytest.raises(AssertionError):
            assert_rope_parity(cos, want_cos, positions, "rotaryDim")


class TestFixture:
    """The fixture handed to the TS side matches upstream **as it is now**."""

    def test_the_checked_in_fixture_is_what_the_generator_writes(self, e2b_config) -> None:
        """MUST: regenerating gives **the same contents** — a stale fixture quietly makes the TS
        side's check meaningless.

        The repository's formatter owns the layout (since 2026-10-16 Oxfmt formats JSON too and
        folds short arrays onto one line), so the JSON values are compared, not the bytes. After
        rewriting, run `vp fmt`.
        """
        stored = rope_fixture.FIXTURE_PATH.read_text(encoding="utf-8")

        assert json.loads(stored) == json.loads(rope_fixture.fixture_json())

    def test_the_declaration_is_the_e2b_one(self) -> None:
        stored = json.loads(rope_fixture.FIXTURE_PATH.read_text(encoding="utf-8"))

        assert stored["spec"] == {
            rope.FULL_ATTENTION: {"theta": 1000000.0, "headDim": 512, "rotaryDim": 128},
            rope.SLIDING_ATTENTION: {"theta": 10000.0, "headDim": 256, "rotaryDim": 256},
        }
        assert stored["positions"] == list(rope_fixture.FIXTURE_POSITIONS)

    @pytest.mark.skipif(
        not rope_fixture.E2B_CONFIG_PATH.exists(),
        reason=f"no upstream checkpoint ({rope_fixture.E2B_CONFIG_PATH})",
    )
    def test_the_declaration_matches_the_upstream_checkpoint(self) -> None:
        """MUST (the module docstring of `rope_fixture`): the hand-copied declaration holds the same
        numbers as the real `config.json`.

        The two tests above only check "the fixture follows the declaration" — if the declaration
        itself drifts from the real checkpoint, the fixture, the TS implementation and the Python
        implementation stay consistent with each other and disagree **only with the distributed
        form** (the breakage when upstream ships a version that changes E2B's theta).
        """
        text_config = json.loads(rope_fixture.E2B_CONFIG_PATH.read_text(encoding="utf-8"))[
            "text_config"
        ]

        for field in UPSTREAM_ROPE_FIELDS:
            assert rope_fixture.E2B_TEXT_CONFIG_FIELDS[field] == text_config[field], field

    def test_the_tables_decode_to_the_declared_shape(self) -> None:
        """The base64 is little-endian f32 `positions × headDim` (the premise of how the TS side
        reads it)."""
        stored = json.loads(rope_fixture.FIXTURE_PATH.read_text(encoding="utf-8"))
        rows = len(stored["positions"])

        for layer_type, tables in stored["tables"].items():
            head_dim = stored["spec"][layer_type]["headDim"]
            for part, encoded in tables.items():
                values = np.frombuffer(base64.b64decode(encoded), dtype="<f4")
                assert values.size == rows * head_dim, f"{layer_type} {part}"
