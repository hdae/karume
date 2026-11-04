"""Behaviour of the quantization formulas of `minicpm5/sweep_w4.py` (ADR 0069 decision 3).

**Pure-formula tests** that need neither real weights, nor the decode greedy assets, nor
transformers. The sweep itself (the side that loads the model and measures) is run by hand; what
is pinned here is only the properties whose breakage "silently turns the measurements into
something else":

- symmetric is idempotent, restores a group's amax element exactly, and `q` never drops to −8
  (the very premise of the storage form)
- an all-zero group comes back as 0 (the lower-clamp path — the bare formula would give NaN)
- the scales are separate **per group** (a large-amax group does not crush a small-amax one)
- a divisibility violation fails loudly (the quality of a form with partial groups is not measured)
- asymmetric restores min / max and stays within `u ∈ [1,15]`, and a degenerate group comes back
  constant
- on a distribution that favours asymmetric, its error really is smaller than symmetric's
  (evidence that it is **not vacuous**)
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch
from torch import nn

from karume.quantize import QuantizeError
from minicpm5 import export as one_shot
from minicpm5 import sweep_w4 as sweep


def relative_error(weight: torch.Tensor, fq: torch.Tensor) -> float:
    """`‖w − fq‖₂ / ‖w‖₂` (the same quantity the sweep takes per family)."""
    return float(torch.linalg.vector_norm(fq - weight) / torch.linalg.vector_norm(weight))


class TestSymmetricFakeQuant:
    def test_reapplying_it_is_bit_identical(self):
        """MUST: idempotent (ADR 0069 decision 3) — if it broke, rounded weights would move with
        every requantization."""
        torch.manual_seed(0)
        weight = torch.randn(5, 128)

        once = sweep.fake_quant_symmetric(weight, 32)
        twice = sweep.fake_quant_symmetric(once, 32)

        assert torch.equal(once, twice)

    def test_the_extreme_of_each_group_is_restored_exactly(self):
        """The amax element lands on `q = ±7` and comes back exactly as `q·s` (the very reason for
        idempotence)."""
        weight = torch.tensor([[0.5, -3.0, 1.25, 0.75, -0.5, 2.0, 8.0, -1.0]])

        fq = sweep.fake_quant_symmetric(weight, 4)

        # The amax of group 0 is -3.0 (index 1), the amax of group 1 is 8.0 (index 2).
        assert float(fq[0, 1]) == -3.0
        assert float(fq[0, 6]) == 8.0

    def test_an_all_zero_group_stays_zero(self):
        """Without the lower clamp it divides by `amax / 7 = 0` and gives NaN (all-zero groups do
        exist)."""
        weight = torch.zeros(2, 16)

        fq = sweep.fake_quant_symmetric(weight, 8)

        assert torch.equal(fq, torch.zeros(2, 16))

    def test_the_levels_never_reach_minus_eight(self):
        """MUST: `q ∈ [−7, +7]`. Allowing −8 breaks the exact restoration of the amax element (=
        idempotence)."""
        torch.manual_seed(1)
        weight = torch.randn(3, 64)

        q, _ = sweep.symmetric_components(weight, 16)

        assert int(q.min()) >= -sweep.INT4_MAX
        assert int(q.max()) <= sweep.INT4_MAX
        # Guarding against a vacuous pass: both ends are really reached (not just the range of
        # `clamp`).
        assert int(q.min()) == -sweep.INT4_MAX
        assert int(q.max()) == sweep.INT4_MAX


#: Two groups whose amplitudes differ by 4 orders of magnitude (with a per-tensor / per-channel
#: scale the smaller one would all be crushed to 0).
LOUD_GROUP = [100.0, -80.0, 60.0, -40.0]
QUIET_GROUP = [0.01, -0.008, 0.006, -0.004]


class TestGroupBoundaries:
    """The scales are separate per group — the very reason for "groups along K"."""

    def test_each_group_gets_its_own_scale(self):
        weight = torch.tensor([LOUD_GROUP + QUIET_GROUP])

        _, scale = sweep.symmetric_components(weight, 4)

        assert scale.shape == (1, 2, 1)
        assert float(scale[0, 0, 0]) == pytest.approx(100.0 / sweep.INT4_MAX)
        assert float(scale[0, 1, 0]) == pytest.approx(0.01 / sweep.INT4_MAX)

    def test_the_quiet_group_survives_the_loud_one(self):
        """The error of a large-amax group does not leak into a small-amax group (a leak would
        quietly crush it to 0)."""
        weight = torch.tensor([LOUD_GROUP + QUIET_GROUP])

        fq = sweep.fake_quant_symmetric(weight, 4)

        quiet = fq[0, 4:]
        assert bool((quiet != 0).all())
        # The quantization error is bounded by s/2 = amax/14 (set by the largest amplitude in the
        # group).
        assert torch.allclose(quiet, torch.tensor(QUIET_GROUP), atol=0.01 / 14)


class TestDivisibility:
    def test_a_group_size_that_does_not_divide_the_in_axis_fails_loudly(self):
        """MUST: partial groups are a form that cannot be stored (ADR 0069 decision 2) — their
        quality is not measured."""
        weight = torch.zeros(2, 100)

        with pytest.raises(QuantizeError, match="is not divisible by"):
            sweep.fake_quant_symmetric(weight, 32)


class TestAsymmetricFakeQuant:
    def test_the_group_extremes_are_restored(self):
        """`u = 1` lands on min and `u = 15` on max (asymmetric uses its whole range)."""
        weight = torch.tensor([[2.0, 3.5, 9.0, 4.25, -6.0, -1.0, 0.5, -2.5]])

        fq = sweep.fake_quant_asymmetric(weight, 4)

        assert float(fq[0, 0]) == pytest.approx(2.0)  # min of group 0
        assert float(fq[0, 2]) == pytest.approx(9.0)  # max of group 0
        assert float(fq[0, 4]) == pytest.approx(-6.0)  # min of group 1
        assert float(fq[0, 6]) == pytest.approx(0.5)  # max of group 1

    def test_the_levels_stay_in_one_to_fifteen(self):
        """15 levels with 0 unused (the same range as symmetric — the pack format does not change
        with or without a zero-point)."""
        torch.manual_seed(2)
        weight = torch.randn(3, 64) * 5.0 + 2.0

        u, _, _ = sweep.asymmetric_components(weight, 16)

        assert int(u.min()) == sweep.UINT4_MIN
        assert int(u.max()) == sweep.UINT4_MAX

    def test_a_degenerate_group_stays_constant(self):
        """MUST: `max == min` falls back to the symmetric formula. With the min-max formula a
        constant group is crushed to ~0."""
        weight = torch.tensor([[3.5, 3.5, 3.5, 3.5, -2.0, -2.0, -2.0, -2.0]])

        fq = sweep.fake_quant_asymmetric(weight, 4)

        assert torch.allclose(fq, weight)

    def test_a_degenerate_zero_group_stays_zero(self):
        """Degenerate and all zero (the path where `|c|/7` also falls to the lower clamp)."""
        fq = sweep.fake_quant_asymmetric(torch.zeros(2, 8), 4)

        assert torch.equal(fq, torch.zeros(2, 8))

    def test_it_beats_the_symmetric_form_on_a_one_sided_group(self):
        """The condition for the asymmetric measurement column to mean something — on a one-sided
        group its error really is smaller.

        Symmetric spreads its levels over `[−amax, amax]`, so on an all-positive group half of the
        levels go unused. An implementation that does not turn this green is not measuring "the
        asymmetric upper bound" (the sweep's column would be vacuous).
        """
        torch.manual_seed(3)
        weight = torch.rand(4, 64) + 10.0

        symmetric = relative_error(weight, sweep.fake_quant_symmetric(weight, 32))
        asymmetric = relative_error(weight, sweep.fake_quant_asymmetric(weight, 32))

        assert asymmetric < symmetric / 10


# ---- Calibrated rounding (iteration J-2) ------------------------------------


class TinyAttention(nn.Module):
    """The two of `q_proj` / `o_proj` (the family names follow the spelling of
    {@link sweep.LINEAR_FAMILIES})."""

    def __init__(self, features: int) -> None:
        super().__init__()
        self.q_proj = nn.Linear(features, features, bias=False)
        self.o_proj = nn.Linear(features, features, bias=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.o_proj(self.q_proj(hidden))


#: kwargs passed with the same value to every layer (a stand-in for Llama's mask / position_ids).
TINY_SHIFT = 0.0


class TinyLayer(nn.Module):
    """A stage equivalent to a decoder layer (the effect of the kwargs is observed through the
    **addition** of `shift`)."""

    def __init__(self, features: int) -> None:
        super().__init__()
        self.self_attn = TinyAttention(features)

    def forward(self, hidden: torch.Tensor, shift: float = 0.0) -> torch.Tensor:
        return self.self_attn(hidden + shift)


@dataclass(frozen=True)
class TinyInnerOutput:
    """A stand-in for `BaseModelOutputWithPast` (holds only `last_hidden_state`)."""

    last_hidden_state: torch.Tensor


class TinyInner(nn.Module):
    """Equivalent to `LlamaModel` (`.layers` / `.embed_tokens` / `.norm`).

    `forward` is the same "embedding → the layers in order → final norm" as upstream, and the
    return value sits in the same field (`last_hidden_state`) — this is what
    {@link sweep.assert_stage_split} compares against.
    """

    def __init__(self, vocab: int, features: int, layers: int) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, features)
        self.layers = nn.ModuleList(TinyLayer(features) for _ in range(layers))
        self.norm = nn.LayerNorm(features)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        use_cache: bool = False,
    ) -> TinyInnerOutput:
        """The kwargs are **identical for every layer** (Llama's form — the premise under which
        `calibrate_stages` carries them unchanged across stages).

        The layers differ through the weights of each `TinyLayer` (random initialization), so a
        dropped stage is detected regardless of the kwargs.
        """
        hidden = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden = layer(hidden, shift=TINY_SHIFT)
        return TinyInnerOutput(self.norm(hidden))


class TinyCausal(nn.Module):
    """Equivalent to `LlamaForCausalLM` (`.model` is a `TinyInner`)."""

    def __init__(self, vocab: int, features: int, layers: int) -> None:
        super().__init__()
        self.model = TinyInner(vocab, features, layers)


class TinyWrapper(nn.Module):
    """Equivalent to `CausalLmWrapper` — the `model.model.layers` shape `sweep.decoder_stages`
    walks."""

    def __init__(self, vocab: int = 8, features: int = 32, layers: int = 2) -> None:
        super().__init__()
        self.model = TinyCausal(vocab, features, layers)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        mask = one_shot.additive_causal_mask(int(input_ids.shape[1]))
        return self.model.model(
            input_ids=input_ids, attention_mask=mask, use_cache=False
        ).last_hidden_state


class SilentWrapper(TinyWrapper):
    """A forward that **never calls** the first stage (the form that triggers the Catcher's fail
    loudly)."""

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.model.embed_tokens(input_ids)


def tiny_inputs(count: int = 3) -> tuple[torch.Tensor, ...]:
    """Id sequences of different lengths (a stand-in for calibration inputs — no tokenizer)."""
    return tuple(
        torch.arange(2 + index, dtype=torch.int64).unsqueeze(0) % 8 for index in range(count)
    )


def tiny_rig(wrapper: TinyWrapper, inputs: tuple[torch.Tensor, ...]) -> sweep.CalibRig:
    stages = sweep.decoder_stages(wrapper)
    scan, counts = sweep.calib_targets(stages)
    return sweep.CalibRig(
        stages=stages,
        scan=scan,
        counts=counts,
        batches=sweep.capture_stage_batches(wrapper, stages[0][1], inputs),
    )


class TestCalibConfigs:
    def test_every_calib_config_is_selectable(self):
        """The calibration grid is among the choices of `--only` (= `ALL_CONFIGS`)."""
        names = [config.name for config in sweep.ALL_CONFIGS]

        assert all(config.name in names for config in sweep.CALIB_CONFIGS)

    def test_selecting_a_calib_config_keeps_the_baseline_first(self):
        """MUST: even a partial rerun goes through the baseline's sanity check first (the existing
        practice)."""
        chosen = sweep.select_configs(["gptq-rtn"])

        assert [config.name for config in chosen] == [sweep.BASELINE_NAME, "gptq-rtn"]

    def test_the_group_size_is_pinned_for_every_calib_config(self):
        """Calibration does not change the grid — the g axis is not swept, for the same reason as
        the method grid."""
        assert all(
            config.grid.group_size == sweep.CALIB_GROUP_SIZE for config in sweep.CALIB_CONFIGS
        )

    def test_no_calib_config_fits_its_table_on_a_subsample_by_default(self):
        """The default is a full fit (a subsample only when `--kmeans-shared-stride` is passed)."""
        assert all(config.grid.fit_stride == 1 for config in sweep.CALIB_CONFIGS)

    def test_the_label_names_both_the_method_and_the_grid(self):
        """The table's "method" column is the pair of method and storage grid (with only one of them
        the row cannot be read)."""
        labels = {config.name: config.label for config in sweep.CALIB_CONFIGS}

        assert labels["gptq-nf4"] == "gptq/nf4"
        assert labels["awq-gptq-rtn"] == "awq+gptq/rtn"

    def test_the_group_scale_grids_project_five_bits(self):
        """4bit + one f32 scale per group of 32 = 5.0 bpw (the same storage as the method grid)."""
        counts = sweep.TargetCounts(modules=2, channels=8, elements=3200)

        assert sweep.group_scale_bits(counts) / counts.elements == pytest.approx(5.0)

    def test_the_calibrated_codebook_costs_one_table_per_layer(self):
        """Core's `kmeans_shared` is a table **within a layer** (a different formula from the method
        grid's single overall table)."""
        counts = sweep.TargetCounts(modules=2, channels=8, elements=3200)

        difference = sweep.layer_table_bits(counts) - sweep.group_scale_bits(counts)

        assert difference == sweep.CODEBOOK_BITS * counts.modules


class TestCalibScan:
    def test_it_lists_every_decoder_linear_with_a_model_wide_fqn(self):
        """The prefix that puts the ledger keys in the same FQN space as `Int4Report` (core's
        contract)."""
        wrapper = TinyWrapper(layers=2)

        scan, counts = sweep.calib_targets(sweep.decoder_stages(wrapper))

        assert sorted(scan) == [
            "model.model.layers.0.self_attn.o_proj.weight",
            "model.model.layers.0.self_attn.q_proj.weight",
            "model.model.layers.1.self_attn.o_proj.weight",
            "model.model.layers.1.self_attn.q_proj.weight",
        ]
        assert counts.modules == 4

    def test_the_vocabulary_table_is_not_in_the_scan(self):
        """`embed_tokens` / `lm_head` are outside the stages — not in the calibration's target
        set."""
        scan, _ = sweep.calib_targets(sweep.decoder_stages(TinyWrapper()))

        assert not any(sweep.EMBED_TOKENS in fqn or sweep.LM_HEAD in fqn for fqn in scan)


class TestCatcher:
    def test_it_captures_one_batch_per_calibration_text(self):
        wrapper = TinyWrapper()
        inputs = tiny_inputs(3)

        batches = sweep.capture_stage_batches(wrapper, wrapper.model.model.layers[0], inputs)

        assert len(batches) == 3
        assert [int(args[0].shape[1]) for args, _kwargs in batches] == [2, 3, 4]

    def test_it_captures_the_keyword_arguments_of_the_call(self):
        """Dropping the kwargs (mask / position_ids …) would make rerunning the stages give
        different numbers."""
        wrapper = TinyWrapper()

        batches = sweep.capture_stage_batches(
            wrapper, wrapper.model.model.layers[0], tiny_inputs(1)
        )

        assert batches[0][1] == {"shift": 0.0}

    def test_the_hook_is_removed_even_though_the_forward_was_aborted(self):
        """Cutting short with the sentinel still cleans up (a leftover hook would make every later
        forward fail)."""
        wrapper = TinyWrapper()
        first = wrapper.model.model.layers[0]

        sweep.capture_stage_batches(wrapper, first, tiny_inputs(1))

        assert not first._forward_pre_hooks
        with torch.no_grad():
            wrapper(tiny_inputs(1)[0])

    def test_a_forward_that_never_reaches_the_first_stage_fails_loudly(self):
        """MUST: letting a pass-through go silently would hide it until the "zero calibration
        inputs" diagnostic."""
        wrapper = SilentWrapper()

        with pytest.raises(AssertionError, match="ran to completion"):
            sweep.capture_stage_batches(wrapper, wrapper.model.model.layers[0], tiny_inputs(1))


class TestCalibrationRun:
    def test_it_rounds_every_scanned_linear(self):
        """The calibrated rounding reaches every tensor of the scan (a layer it misses fails the
        check)."""
        torch.manual_seed(0)
        wrapper = TinyWrapper()
        rig = tiny_rig(wrapper, tiny_inputs(2))
        before = {fqn: weight.detach().clone() for fqn, weight in rig.scan.items()}

        report = sweep.apply_calib(sweep.CALIB_CONFIGS[0], rig, shared_stride=0)
        sweep.assert_calib_covers_scan(report, rig.scan, "gptq-rtn")

        assert report.modules == len(rig.scan)
        assert all(not torch.equal(rig.scan[fqn], value) for fqn, value in before.items())

    def test_every_calib_config_runs_end_to_end(self):
        """All 5 go through core's driver (every pair of method and storage grid is alive)."""
        for config in sweep.CALIB_CONFIGS:
            torch.manual_seed(1)
            wrapper = TinyWrapper()
            rig = tiny_rig(wrapper, tiny_inputs(2))

            report = sweep.apply_calib(config, rig, shared_stride=0)
            sweep.assert_calib_covers_scan(report, rig.scan, config.name)

            assert report.method == config.method
            assert report.grid == config.grid.kind

    def test_a_subsampled_table_is_reported(self):
        """MUST: when `fit_stride` is used it is stated in the output (`describe` picks it up)."""
        torch.manual_seed(2)
        wrapper = TinyWrapper()
        rig = tiny_rig(wrapper, tiny_inputs(2))
        kmeans = next(config for config in sweep.CALIB_CONFIGS if config.name == "gptq-kmeans")

        report = sweep.apply_calib(kmeans, rig, shared_stride=4)

        assert "fit_stride 4" in report.describe()

    def test_a_scan_the_calibration_missed_fails_loudly(self):
        """Blocking a vacuous pass — a layer not in the scan / not rounded fails the check."""
        torch.manual_seed(3)
        wrapper = TinyWrapper()
        rig = tiny_rig(wrapper, tiny_inputs(2))
        report = sweep.apply_calib(sweep.CALIB_CONFIGS[0], rig, shared_stride=0)
        widened = {**rig.scan, "model.model.layers.9.self_attn.q_proj.weight": torch.zeros(1)}

        with pytest.raises(AssertionError, match="do not match the"):
            sweep.assert_calib_covers_scan(report, widened, "gptq-rtn")


class TestStageSplit:
    """The split check ({@link sweep.assert_stage_split}) — the premise that calibration sees "the
    same activations as the real model"."""

    def test_the_sequential_stages_reproduce_the_model_bit_for_bit(self):
        torch.manual_seed(4)
        wrapper = TinyWrapper()
        inputs = tiny_inputs(1)
        stages = sweep.decoder_stages(wrapper)
        batches = sweep.capture_stage_batches(wrapper, stages[0][1], inputs)

        sweep.assert_stage_split(wrapper, inputs[0], batches[0], stages)

    def test_a_dropped_stage_is_caught(self):
        """Even with a stage dropped the numbers come out normally — only the split check rejects
        it."""
        torch.manual_seed(5)
        wrapper = TinyWrapper()
        inputs = tiny_inputs(1)
        stages = sweep.decoder_stages(wrapper)
        batches = sweep.capture_stage_batches(wrapper, stages[0][1], inputs)

        with pytest.raises(AssertionError, match="not bit-identical"):
            sweep.assert_stage_split(wrapper, inputs[0], batches[0], stages[:-1])
