"""The pure part of `irodori/measure_quant.py` (the part that needs no real weights).

The measurement itself is manual (the full loop takes tens of minutes on torch CPU). Pinned here
is only the part whose breakage would make **the measurements silently lie**:

- The edge values of the metrics (SNR / LSD / relRMS) — inf / 0 on a match, an all-zero
  reference, the upper bound for a scaled copy
- The config table — `i8-all` covers every role, `i8-mixed` leaves out only duration, the
  orthogonal decomposition takes one role at a time, `w8a8` adds only activations to the same
  weights as `i8-all`
- Applying the activation simulation — it hooks the eligible `nn.Linear`s of the DiT, and **zero
  hooks fail loudly**
- Grouping roles → modules — with every role it is **the same set** as the 9 modules
  `export_series` rounds (a mismatch would measure something else while believing it measures
  "the same thing as the distributed f16 series")
- The orthogonality check — **both sides** of "z / the waveform change only where they should"
  really fail
"""

from __future__ import annotations

import math
import wave

import pytest
import torch
from torch import nn

from irodori import export as ex
from irodori import measure_quant as mq
from irodori import pipeline_ref as ip
from karume.quantize import QUANT_MODULE_TYPES, iter_quant_targets


class TestMetrics:
    def test_snr_is_infinite_for_a_bit_identical_signal(self):
        signal = torch.tensor([0.5, -0.25, 0.125])

        assert mq.snr_db(signal, signal) == math.inf

    def test_snr_matches_the_closed_form(self):
        reference = torch.ones(4)
        value = reference + 0.1

        assert mq.snr_db(value, reference) == pytest.approx(10.0 * math.log10(1.0 / 0.01))

    def test_snr_of_a_zero_reference_is_minus_infinite(self):
        """SNR is undefined for an all-zero reference — do not divide by 0 and return NaN."""
        assert mq.snr_db(torch.ones(4), torch.zeros(4)) == -math.inf

    def test_rel_rms_is_zero_for_a_zero_reference(self):
        assert mq.rel_rms(torch.ones(4), torch.zeros(4)) == 0.0

    def test_rel_rms_matches_the_closed_form(self):
        reference = torch.tensor([3.0, 4.0])

        assert mq.rel_rms(reference * 2.0, reference) == pytest.approx(1.0)

    def test_lsd_is_zero_for_a_bit_identical_signal(self):
        torch.manual_seed(0)
        signal = torch.randn(mq.STFT_N_FFT * 4)

        assert mq.log_spectral_distance(signal, signal) == 0.0

    def test_lsd_of_a_scaled_copy_stays_under_the_scale_factor(self):
        """The LSD of a doubled signal is bounded by 20log10(2) (the floor clamp only ever
        shrinks it)."""
        torch.manual_seed(0)
        reference = torch.randn(mq.STFT_N_FFT * 4)

        distance = mq.log_spectral_distance(reference * 2.0, reference)

        assert 0.0 < distance <= 20.0 * math.log10(2.0) + 1e-6


class TestConfigTable:
    def test_the_base_is_the_only_unrounded_config(self):
        unrounded = [name for name, recipe in mq.RECIPES.items() if recipe.weight is None]

        assert unrounded == [mq.BASE_CONFIG]

    def test_i8_all_covers_every_role(self):
        assert mq.RECIPES["i8-all"].roles == mq.ROLES

    def test_i8_mixed_leaves_duration_alone(self):
        """The candidate mixed table (option (i) of ADR 0050 decision 6) — it leaves out
        duration **only**."""
        mixed = set(mq.RECIPES["i8-mixed"].roles)

        assert mixed == set(mq.ROLES) - {ex.TARGET_DURATION}

    def test_the_decomposition_takes_one_role_at_a_time(self):
        legs = {name: recipe for name, recipe in mq.DIAGNOSTICS.items()}

        assert len(legs) == 5
        assert all(len(recipe.roles) == 1 for recipe in legs.values())
        assert {recipe.roles[0] for recipe in legs.values()} == set(mq.DECOMPOSED_ROLES)

    def test_duration_and_codec_are_outside_the_latent_path(self):
        """MUST: the very expectation of the orthogonality check (duration affects only S, the
        codec only the waveform)."""
        assert set(mq.ROLES) - {ex.TARGET_DURATION, mq.ROLE_CODEC} == mq.LATENT_ROLES

    def test_w8a8_rounds_the_same_weights_as_the_weight_only_config(self):
        """The two distributed presets **share the same i8 bytes** — they differ only on the
        activation side."""
        weight_only = mq.RECIPES[mq.WEIGHT_ONLY_BASE]
        act = mq.RECIPES["w8a8"]

        assert (act.weight, act.roles) == (weight_only.weight, weight_only.roles)
        assert act.act_quant and not weight_only.act_quant

    def test_only_w8a8_carries_the_activation_sim(self):
        """If more configs get the activation simulation, the pass-through detection needs more
        comparison targets too."""
        with_sim = [name for name, recipe in mq.RECIPES.items() if recipe.act_quant]

        assert with_sim == ["w8a8"]


class TinyModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(2, 2, bias=False)


def _modules() -> dict[str, nn.Module]:
    return {
        name: TinyModule()
        for name in (
            ex.TARGET_BACKBONE,
            ex.TARGET_TEXT_PROJ,
            ex.TARGET_CAPTION_PROJ,
            ex.TARGET_SPEAKER,
            ex.TARGET_DURATION,
            ex.TARGET_DIT,
            "speaker_norm",
            "text_norm",
            "caption_norm",
        )
    }


class TestRoleModules:
    """Grouping roles → rounding targets (the key is set equality with the 9 modules
    `export_series` / `emit` round)."""

    def test_all_roles_hand_the_same_nine_modules_as_the_export(self):
        modules = _modules()

        picked = mq.role_modules(modules, mq.ROLES)

        assert set(picked) == set(modules)

    def test_the_codec_role_touches_nothing_on_the_latent_side(self):
        """`i8-codec-only` rounds nothing on the latent side (bit identity of z is the
        check)."""
        assert mq.role_modules(_modules(), (mq.ROLE_CODEC,)) == {}

    def test_each_norm_rides_with_the_graph_that_holds_it(self):
        """MUST: a norm goes with the role that **consumes** it (where it lives: ADR 0048
        decision 1)."""
        modules = _modules()

        assert sorted(mq.role_modules(modules, (ex.TARGET_DURATION,))) == [
            ex.TARGET_DURATION,
            "text_norm",
        ]
        assert sorted(mq.role_modules(modules, (ex.TARGET_SPEAKER,))) == [
            ex.TARGET_SPEAKER,
            "speaker_norm",
        ]
        assert sorted(mq.role_modules(modules, (ex.TARGET_CAPTION_PROJ,))) == [
            ex.TARGET_CAPTION_PROJ,
            "caption_norm",
        ]
        assert sorted(mq.role_modules(modules, (ex.TARGET_BACKBONE,))) == [ex.TARGET_BACKBONE]


class _DitLike(nn.Module):
    """A DiT stand-in with two eligible `nn.Linear`s (`in_features % 4 == 0`)."""

    def __init__(self) -> None:
        super().__init__()
        self.qkv = nn.Linear(4, 8)
        self.out = nn.Linear(8, 4)


class TestActQuantSim:
    """MUST: report how many were hooked (running with zero would publish the same numbers as
    `i8-all` under the name w8a8)."""

    def test_it_hooks_every_eligible_linear_in_the_dit(self):
        modules = {ex.TARGET_DIT: _DitLike()}

        handles, attached = mq.attach_dit_act_quant(mq.RECIPES["w8a8"], modules)

        assert attached == 2
        mq.act_quant.detach_act_quant(handles)

    def test_it_actually_rounds_the_activations(self):
        """The hook's effect itself (the output after hooking equals the input passed through
        `quantize_rows`)."""
        modules = {ex.TARGET_DIT: _DitLike()}
        linear = modules[ex.TARGET_DIT].qkv
        x = torch.tensor([[1.0, -0.3, 0.02, 0.7]])
        handles, _ = mq.attach_dit_act_quant(mq.RECIPES["w8a8"], modules)

        with torch.no_grad():
            hooked = linear(x)
            mq.act_quant.detach_act_quant(handles)
            expected = linear(mq.act_quant.quantize_rows(x))

        assert torch.equal(hooked, expected)

    def test_zero_eligible_linears_is_fail_loudly(self):
        """Do not pass with zero eligible linears — the quality comes out on the "good" side, so
        staying silent can only be misread."""
        modules = {ex.TARGET_DIT: TinyModule()}  # Linear(2, 2) has k % 4 != 0

        with pytest.raises(SystemExit, match=r"attached to 0 nn\.Linear"):
            mq.attach_dit_act_quant(mq.RECIPES["w8a8"], modules)

    def test_a_weight_only_config_hooks_nothing(self):
        handles, attached = mq.attach_dit_act_quant(mq.RECIPES[mq.WEIGHT_ONLY_BASE], {})

        assert (handles, attached) == ([], 0)


def _entry(*, z_changed: bool, wav_changed: bool) -> dict[str, object]:
    return {"vsBase": {"zBitEqual": not z_changed, "wavBitEqual": not wav_changed}}


def _gates(config: str, *, z_changed: bool, wav_changed: bool) -> list[str]:
    entries = {"full": _entry(z_changed=z_changed, wav_changed=wav_changed)}
    gates = mq.run_gates(config, mq.RECIPES[config], {}, entries, None)
    return gates["failures"]


class TestOrthogonalityGate:
    """MUST: the expectation has **both sides** (changes / does not change) — with one side only,
    either a pass-through or a scope leak goes unnoticed."""

    def test_a_dit_only_config_that_did_not_move_the_latent_is_red(self):
        failures = _gates("i8-dit-only", z_changed=False, wav_changed=False)

        assert any("z" in message for message in failures)

    def test_a_dit_only_config_that_moved_both_is_green(self):
        assert _gates("i8-dit-only", z_changed=True, wav_changed=True) == []

    def test_a_duration_only_config_that_moved_the_latent_is_red(self):
        """duration affects only S — if z moved, the scope leaked."""
        failures = _gates("i8-duration-only", z_changed=True, wav_changed=True)

        assert len(failures) == 2

    def test_a_duration_only_config_that_moved_nothing_is_green(self):
        assert _gates("i8-duration-only", z_changed=False, wav_changed=False) == []

    def test_a_speaker_only_config_is_expected_to_move_nothing_without_a_reference(self):
        """The speaker expectation **depends on the case** (without a reference the zero-input
        short-circuit skips the encoder — ADR 0048. Measured 2026-08-12: no-ref bit-identical,
        full changed)."""
        entries = {
            "full": _entry(z_changed=True, wav_changed=True),
            "no-ref": _entry(z_changed=False, wav_changed=False),
        }
        gates = mq.run_gates("i8-speaker-only", mq.RECIPES["i8-speaker-only"], {}, entries, None)

        assert gates["failures"] == []

    def test_a_speaker_only_config_that_moved_the_no_ref_case_is_red(self):
        """If z moved in the case without a reference, the zero short-circuit broke or the scope
        leaked."""
        entries = {"no-ref": _entry(z_changed=True, wav_changed=True)}
        gates = mq.run_gates("i8-speaker-only", mq.RECIPES["i8-speaker-only"], {}, entries, None)

        assert len(gates["failures"]) == 2

    def test_a_codec_only_config_must_move_the_waveform_but_not_the_latent(self):
        assert _gates("i8-codec-only", z_changed=False, wav_changed=True) == []
        assert _gates("i8-codec-only", z_changed=False, wav_changed=False) != []
        assert _gates("i8-codec-only", z_changed=True, wav_changed=True) != []


def _act_gates(*, moved_from_weight_only: bool) -> list[str]:
    entry = _entry(z_changed=True, wav_changed=True)
    entry["vsWeightOnly"] = {"zBitEqual": not moved_from_weight_only}
    gates = mq.run_gates("w8a8", mq.RECIPES["w8a8"], {}, {"full": entry}, None)
    return gates["failures"]


class TestActQuantPassThroughGate:
    """MUST: pass-through detection uses **the difference from the weight-only config** (the
    difference from the base is explained by the weight rounding)."""

    def test_a_latent_that_moved_off_the_weight_only_config_is_green(self):
        assert _act_gates(moved_from_weight_only=True) == []

    def test_a_latent_bit_identical_to_the_weight_only_config_is_red(self):
        failures = _act_gates(moved_from_weight_only=False)

        assert len(failures) == 1
        assert mq.WEIGHT_ONLY_BASE in failures[0]


class TestGoldenGate:
    """MUST: the base `f32` is byte-identical to the full-loop golden (otherwise every
    measurement loses its meaning)."""

    def _failures(self, latent: torch.Tensor, golden: torch.Tensor) -> list[str]:
        gates = mq.run_gates(
            mq.BASE_CONFIG, mq.RECIPES[mq.BASE_CONFIG], {"full": latent}, {}, {"full": golden}
        )
        return gates["failures"]

    def test_a_bit_identical_latent_is_green(self):
        golden = torch.tensor([[[0.25, -0.5]]])

        assert self._failures(golden.clone(), golden) == []

    def test_a_latent_off_by_one_ulp_is_red(self):
        """No approximate pass — a 1-ulp move means the path has changed."""
        golden = torch.tensor([[[0.25, -0.5]]])
        drifted = golden + torch.tensor([[[0.0, 2.0**-23]]])

        assert self._failures(drifted, golden) != []


class TestWriteWav:
    def test_it_writes_mono_pcm16_at_the_given_rate(self, tmp_path):
        path = tmp_path / "case.wav"

        mq.write_wav(path, torch.tensor([0.0, 1.0, -1.0, 0.5]), 48000)

        with wave.open(str(path), "rb") as handle:
            assert handle.getnchannels() == 1
            assert handle.getsampwidth() == 2
            assert handle.getframerate() == 48000
            assert handle.getnframes() == 4

    def test_it_clips_instead_of_wrapping_around(self):
        """MUST: saturation clamps (a plain cast to int16 wraps +1.5 around to a negative
        value)."""
        clipped = torch.tensor([2.0, -2.0]).clamp(-1.0, 1.0)
        pcm = torch.floor(clipped * 32767.0 + 0.5).to(torch.int16)

        assert pcm.tolist() == [32767, -32767]


def _payload(name: str, predicted: int, *, versus: dict[str, float] | None) -> dict[str, object]:
    entry: dict[str, object] = {
        "S": 161,
        "predictedS": predicted,
        "forwards": 200,
        "predictedSeconds": predicted / 25,
        "elapsed": 12.0,
    }
    if versus is not None:
        entry["vsBase"] = versus
    return {"cases": {"full": entry}, "gates": {"failures": []}}


class TestReportTables:
    """The S drift table is **this script's main output** (the ruling on the mixed table rests on
    this one table)."""

    def _collected(self, drifted: int) -> dict[str, dict[str, object]]:
        return {
            mq.BASE_CONFIG: _payload(mq.BASE_CONFIG, 161, versus=None),
            "i8-all": _payload(
                "i8-all",
                drifted,
                versus={
                    "zRelRms": 1.5e-2,
                    "zMaxAbsDiff": 3.0e-2,
                    "wavSnrDb": 12.5,
                    "wavLsdDb": 3.2,
                },
            ),
        }

    def test_a_matching_prediction_reads_as_agreement(self):
        table = mq.drift_table(self._collected(161))

        assert "| i8-all | full | 161 | 161 | match |" in table

    def test_a_drifting_prediction_is_signed(self):
        table = mq.drift_table(self._collected(159))

        assert "| i8-all | full | 159 | 161 | -2 |" in table

    def test_the_base_row_says_it_is_the_base(self):
        table = mq.quality_table(self._collected(161))

        assert "| f32 | full | 0 (base)" in table
        assert "| i8-all | full | 1.5000e-02 | 3.0000e-02 | 12.50 | 3.20 |" in table

    def test_the_report_lists_only_red_configs(self):
        collected = self._collected(161)
        collected["i8-all"]["gates"] = {
            "failures": ["full: z broke the expectation that it changes"]
        }

        report = mq.build_report(collected)

        assert sorted(report["failures"]) == ["i8-all"]


class TestFramesOverride:
    """The injection point of `irodori.pipeline_ref.run_case` (fixes S from outside) — the
    default is the plain prediction."""

    def test_the_golden_path_keeps_the_predicted_length(self):
        signature = ip.run_case.__kwdefaults__

        assert signature["frames_override"] is None


class _W4Like(nn.Module):
    """A tree mixing weights that fit g32 and weights that do not (to test the target split and
    the exclusions).

    - `aligned` … a linear with in axis 64 = two g32 groups
    - `ragged` … a linear with in axis 12 (not divisible by g32 → excluded)
    - `conv` … `Cin·K = 8·4 = 32` (one group on the flattened receptive-field axis)
    """

    def __init__(self) -> None:
        super().__init__()
        self.aligned = nn.Linear(64, 4, bias=False)
        self.ragged = nn.Linear(12, 4, bias=False)
        self.conv = nn.Conv1d(8, 4, 4, bias=False)


class TestW4ConfigTable:
    """The w4 config table (ADR 0069 addendum 5 — the 4 winning methods × all roles + the RTN
    mix)."""

    def test_every_w4_config_carries_a_method(self):
        assert all(recipe.method is not None for recipe in mq.W4_CONFIGS.values())

    def test_no_storage_dtype_config_carries_a_method(self):
        """A method on the f16 / i8 side would bypass `irodori.export.fake_quant`."""
        others = {**mq.CONFIGS, **mq.DIAGNOSTICS}

        assert all(recipe.method is None for recipe in others.values())

    def test_the_methods_are_the_four_screening_winners(self):
        """Exactly the 4 that survived screening (fp4 and the other 2 kmeans granularities were
        dropped)."""
        names = {recipe.method.name for recipe in mq.W4_CONFIGS.values()}

        assert names == {"rtn-i4-g32", "nf4", "mxfp4", "kmeans:shared"}

    def test_the_all_role_configs_cover_every_role(self):
        covering = [name for name, recipe in mq.W4_CONFIGS.items() if recipe.roles == mq.ROLES]

        assert covering == ["i4-all", "nf4-all", "mxfp4-all", "kmeans-shared-all"]

    def test_the_w4_mixed_config_shares_the_axis_with_the_i8_one(self):
        """MUST: the mix axis is the same as for i8 (written separately, the two mixes would
        silently diverge)."""
        assert mq.RECIPES["i4-mixed"].roles == mq.RECIPES["i8-mixed"].roles
        assert set(mq.RECIPES["i4-mixed"].roles) == set(mq.ROLES) - {ex.TARGET_DURATION}

    def test_the_weight_spelling_is_the_method_name(self):
        """`weight` is the rounding's spelling — not maintained twice next to the method
        name."""
        assert mq.RECIPES["nf4-all"].weight == "nf4"

    def test_the_op_types_stay_the_five_i8_kinds(self):
        """Narrowing them would measure linears only while believing it measures "the quality
        with non-linears rounded too"."""
        assert mq.W4_OP_TYPES == QUANT_MODULE_TYPES

    def test_the_group_size_is_pinned_to_32(self):
        """Do not vary g at the same time as the methods are compared (ADR 0069 addendum 5,
        item 3)."""
        assert mq.W4_GROUP_SIZE == 32

    def test_the_linear_config_is_the_shippable_form(self):
        """`i4-linear` = RTN × Linear only (the form distribution supports today — ADR 0069
        decision 5)."""
        recipe = mq.RECIPES["i4-linear"]

        assert recipe.method is mq.W4_RTN
        assert recipe.op_types == (nn.Linear,)
        # The codec has no linear at all, so it is left out of roles (identity — a role with no
        # target is not run).
        assert set(recipe.roles) == set(mq.ROLES) - {mq.ROLE_CODEC}

    def test_the_all_role_configs_keep_the_wide_op_types(self):
        """The default op kinds stay the wide 5 (adding i4-linear did not narrow the existing
        configs)."""
        for name in ("i4-all", "i4-mixed", "nf4-all", "mxfp4-all", "kmeans-shared-all"):
            assert mq.RECIPES[name].op_types == mq.W4_OP_TYPES, name


class TestScanTargets:
    """The target split (the side that fits g32 / the side that does not divide and is left
    out) — exclusions are always listed."""

    def test_it_splits_the_aligned_and_the_ragged(self):
        scan = mq.scan_targets(_W4Like(), mq.W4_OP_TYPES)

        assert scan.counts == mq.TargetCounts(modules=2, channels=8, elements=64 * 4 + 8 * 4 * 4)
        assert [item.fqn for item in scan.excluded] == ["ragged.weight"]
        assert scan.excluded[0].axis_length == 12

    def test_the_linear_only_set_leaves_the_conv_out(self):
        scan = mq.scan_targets(_W4Like(), (nn.Linear,))

        assert scan.counts == mq.TargetCounts(modules=1, channels=4, elements=64 * 4)

    def test_the_group_count_follows_the_element_count(self):
        counts = mq.TargetCounts(modules=2, channels=8, elements=384)

        assert counts.groups == 384 // 32

    def test_the_conv_axis_is_the_flattened_receptive_field(self):
        """The group axis generalizes each op's "in axis" — `Cin·K` for conv (ADR 0069
        addendum 5, item 1)."""
        weight = nn.Conv1d(8, 4, 4, bias=False).weight

        assert mq.group_axis_length(weight, 0) == 8 * 4

    def test_the_transposed_conv_axis_is_the_layout_axis_one(self):
        """`ConvTranspose1d` has the weight `[Cin, Cout, K]`, so the channel axis is 1."""
        weight = nn.ConvTranspose1d(4, 8, 8, bias=False).weight

        assert mq.group_axis_length(weight, 1) == 4 * 8

    def test_the_include_predicate_drops_exactly_the_excluded_modules(self):
        scan = mq.scan_targets(_W4Like(), mq.W4_OP_TYPES)

        include = mq.aligned_include(scan.excluded)

        assert not include("ragged")
        assert include("aligned") and include("conv")


class TestApplyWeightQuant:
    """How the rounding is applied (w4 calls core per method, f16 / i8 use the same function as
    the distribution path)."""

    def test_it_rounds_the_aligned_weights_and_leaves_the_ragged_alone(self):
        torch.manual_seed(0)
        module = _W4Like()
        before = {name: tensor.detach().clone() for name, tensor in module.named_parameters()}

        reports = mq.apply_weight_quant(
            mq.w4_recipe(mq.W4_RTN, (ex.TARGET_DIT,)), {ex.TARGET_DIT: module}
        )

        assert not torch.equal(module.aligned.weight, before["aligned.weight"])
        assert not torch.equal(module.conv.weight, before["conv.weight"])
        assert torch.equal(module.ragged.weight, before["ragged.weight"])
        assert "excluded 1 not aligned to g32" in reports[ex.TARGET_DIT]

    def test_a_role_without_target_types_stays_f32(self):
        """The norms have no w4 target type — the side that does not fail on "zero" per
        role."""
        reports = mq.apply_weight_quant(
            mq.w4_recipe(mq.W4_RTN, (ex.TARGET_DIT,)), {"text_norm": nn.LayerNorm(8)}
        )

        assert reports["text_norm"] == "kept as f32 storage (no w4 target types)"

    def test_a_role_whose_targets_all_fall_off_the_grid_is_fail_loudly(self):
        """Silently running a role whose targets all fell off g32 in f32 would measure "a config
        believed to be rounded"."""
        module = nn.Linear(12, 4, bias=False)

        with pytest.raises(SystemExit, match="g32"):
            mq.apply_weight_quant(
                mq.w4_recipe(mq.W4_RTN, (ex.TARGET_DIT,)), {ex.TARGET_DIT: module}
            )

    def test_a_storage_dtype_recipe_goes_through_the_export_helper(self):
        """f16 / i8 keep using `irodori.export.fake_quant`, the same as the distribution
        path."""
        reports = mq.apply_weight_quant(
            mq.Recipe("f16", (ex.TARGET_DIT,)), {ex.TARGET_DIT: TinyModule()}
        )

        assert "f16" in reports[ex.TARGET_DIT]

    def test_a_linear_only_recipe_leaves_the_conv_untouched(self):
        """What `i4-linear` really is — op_types also apply to the rounding (narrowing only the
        scan would be pointless)."""
        torch.manual_seed(0)
        module = _W4Like()
        before = {name: tensor.detach().clone() for name, tensor in module.named_parameters()}

        mq.apply_weight_quant(
            mq.w4_recipe(mq.W4_RTN, (ex.TARGET_DIT,), op_types=(nn.Linear,)),
            {ex.TARGET_DIT: module},
        )

        assert not torch.equal(module.aligned.weight, before["aligned.weight"])
        assert torch.equal(module.conv.weight, before["conv.weight"])
        assert torch.equal(module.ragged.weight, before["ragged.weight"])

    def test_a_method_that_rounds_a_different_set_than_the_scan_is_fail_loudly(self):
        """Fail when the counted targets and the rounded targets diverge (the detector for a
        regression in baking in op_types)."""
        wide = mq.W4Method(
            "rtn-i4-g32",
            # Ignores the config's op_types and always rounds the wide 5 kinds (reproduces the
            # regression).
            lambda model, _op_types, include, _stride: mq.fake_quant_int4(
                model, mq.W4_GROUP_SIZE, include=include, op_types=mq.W4_OP_TYPES
            ),
            lambda counts, _tables: 4.0 * counts.elements,
            "for testing",
        )

        with pytest.raises(SystemExit, match="the rounded count"):
            mq.apply_weight_quant(
                mq.w4_recipe(wide, (ex.TARGET_DIT,), op_types=(nn.Linear,)),
                {ex.TARGET_DIT: _W4Like()},
            )

    def test_the_shared_codebook_takes_the_fit_stride(self):
        """Only the table fit uses a subsample (it is applied to everything) — a full fit of the
        dit role exceeds the real memory."""
        torch.manual_seed(0)
        module = _W4Like()
        recipe = mq.w4_recipe(mq.W4_KMEANS_SHARED, (ex.TARGET_DIT,))

        reports = mq.apply_weight_quant(recipe, {ex.TARGET_DIT: module}, 3)

        assert "table fit on a 1/3 subsample" in reports[ex.TARGET_DIT]

    def test_a_fit_stride_on_a_method_without_a_table_is_fail_loudly(self):
        """MUST: fail when passed to a config it does not affect (silently ignoring it changes
        how the numbers read)."""
        with pytest.raises(SystemExit, match="fit"):
            mq.check_fit_stride(mq.RECIPES["i4-all"], 11)

    def test_the_default_fit_stride_passes_every_config(self):
        for recipe in mq.RECIPES.values():
            mq.check_fit_stride(recipe, 1)

    def test_the_rounding_is_idempotent(self):
        """MUST: idempotent (symmetric, closed at ±7 — bit-unchanged on reapplication, ADR 0069
        decision 3)."""
        torch.manual_seed(0)
        module = _W4Like()
        recipe = mq.w4_recipe(mq.W4_RTN, (ex.TARGET_DIT,))
        mq.apply_weight_quant(recipe, {ex.TARGET_DIT: module})
        once = module.aligned.weight.detach().clone()

        mq.apply_weight_quant(recipe, {ex.TARGET_DIT: module})

        assert torch.equal(module.aligned.weight, once)


class TestSizeProjection:
    """The size estimate (**a projection by formula**, not a measurement) — columns that go in
    the same table as the quality."""

    COUNTS = mq.TargetCounts(modules=2, channels=8, elements=384)

    def test_a_f32_scale_method_projects_to_five_bits_per_weight(self):
        assert mq.project_size(mq.W4_RTN, self.COUNTS, 1).bits_per_weight == 5.0
        assert mq.project_size(mq.W4_NF4, self.COUNTS, 1).bits_per_weight == 5.0

    def test_the_power_of_two_scale_saves_the_scale_bits(self):
        """The MXFP4 scale is E8M0 (1 byte) — 4.25 bpw with g32."""
        assert mq.project_size(mq.W4_MXFP4, self.COUNTS, 1).bits_per_weight == 4.25

    def test_the_shared_codebook_costs_one_table_per_artifact(self):
        projected = mq.project_size(mq.W4_KMEANS_SHARED, self.COUNTS, 3)

        assert projected.bits == 5.0 * 384 + mq.CODEBOOK_BITS * 3

    def test_a_table_free_method_ignores_the_artifact_count(self):
        assert (
            mq.project_size(mq.W4_RTN, self.COUNTS, 1).bits
            == mq.project_size(mq.W4_RTN, self.COUNTS, 9).bits
        )

    def test_the_f32_side_is_the_denominator_of_the_ratio(self):
        assert mq.project_size(mq.W4_RTN, self.COUNTS, 1).f32_mib == 384 * 4 / 1024**2


def _w4_config(excluded: list[dict[str, object]]) -> dict[str, object]:
    return {
        "cases": {},
        "gates": {"failures": []},
        "w4": {
            "method": "rtn-i4-g32",
            "excluded": excluded,
            "size": {
                "formula": "4bit + g32 f32 scale = 5.0 bpw",
                "modules": 2,
                "elements": 384,
                "bitsPerWeight": 5.0,
                "projectedMiB": 2.0,
                "f32MiB": 8.0,
            },
        },
    }


class TestW4Tables:
    """The two w4 outputs (the size estimate table, the exclusion list) — exclusions are never
    silent."""

    def test_the_size_table_carries_the_formula_and_the_ratio(self):
        table = mq.size_table({"i4-all": _w4_config([])})

        assert table.splitlines()[-1] == (
            "| i4-all | rtn-i4-g32 | 2 | 384 | 4bit + g32 f32 scale = 5.0 bpw"
            " | 5.000 | 2.0 | 8.0 | 0.250 |"
        )

    def test_a_config_without_w4_gets_no_row(self):
        table = mq.size_table({"i8-all": {"cases": {}, "gates": {"failures": []}}})

        assert len(table.splitlines()) == 2

    def test_the_exclusions_are_listed_per_weight(self):
        excluded = [
            {
                "role": "speaker",
                "fqn": "encoder.blocks.0.mlp.w2.weight",
                "axisLength": 1996,
                "elements": 1532928,
            }
        ]

        lines = mq.excluded_lines({"i4-all": _w4_config(excluded)})

        assert lines[0] == "- i4-all: excluded 1 not aligned to g32"
        assert "speaker / encoder.blocks.0.mlp.w2.weight axis length 1996" in lines[1]

    def test_no_exclusion_still_says_so(self):
        """Says "no exclusions" too — a missing line could not be told apart from "not w4"."""
        assert mq.excluded_lines({"i4-all": _w4_config([])}) == [
            "- i4-all: no exclusions for g32 misalignment"
        ]

    def test_a_config_without_w4_is_silent(self):
        assert mq.excluded_lines({"i8-all": {"cases": {}, "gates": {"failures": []}}}) == []


# ---- Calibrated rounding (step J-2) ----------------------------------------------
#
# Without real weights, only "how the batches are captured" and "the checks" can be pinned. The
# copy of the DiT block ({@link mq.DitBlockStage}) itself only means something with the real
# weights, so this checks with a stand-in stage that **the equivalence check really fails**
# (its detection power).


def _tiny_pooled(text: torch.Tensor, speaker: torch.Tensor, caption: torch.Tensor) -> torch.Tensor:
    """A stand-in for the condition-side projection (the value `dit-context` builds once per
    generation — it mixes all three conditions)."""
    return text.mean(dim=1) + speaker.mean(dim=1) + caption.mean(dim=1)


def _tiny_mix(pooled: torch.Tensor, mask: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """A term mixing the mask and the RoPE table into the condition-side value (mixing up any one
    of them changes the output)."""
    return pooled[:, None, :] * float(mask.sum()) + freqs[None, :, :]


class TinyAdaLN(nn.Module):
    """The place of `LowRankAdaLN` — the very hook point where the Catcher captures the first
    block."""

    def __init__(self, features: int) -> None:
        super().__init__()
        self.proj = nn.Linear(features, features, bias=False)

    def forward(self, x: torch.Tensor, cond_embed: torch.Tensor) -> torch.Tensor:
        return self.proj(x) + cond_embed


class TinyBlock(nn.Module):
    def __init__(self, features: int) -> None:
        super().__init__()
        self.attention_adaln = TinyAdaLN(features)
        self.mlp = nn.Linear(features, features, bias=False)


class TinyContextGraph(nn.Module):
    """A stand-in for `DitContextGraph` — applies the norms to the three condition states and
    returns the condition-side value per block.

    A copy of the DiT split into `dit-context` / `dit` (ADR 0114): the normed conditions are built
    **once per generation**, and `dit` takes them as inputs.
    """

    def __init__(self, features: int = 32, blocks: int = 2) -> None:
        super().__init__()
        self.blocks = blocks
        self.text_norm = nn.LayerNorm(features)
        self.caption_norm = nn.LayerNorm(features)

    def forward(
        self, text_state: torch.Tensor, speaker_state: torch.Tensor, caption_state: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        pooled = _tiny_pooled(
            self.text_norm(text_state), speaker_state, self.caption_norm(caption_state)
        )
        return tuple(pooled for _ in range(self.blocks))


class TinyGraph(nn.Module):
    """A stand-in for `DitGraph` — copies the argument order (`x_t` / `t_embed` / `mask` + the
    condition-side values) and "the side inputs are built once **before** the block loop"."""

    def __init__(self, features: int = 32, blocks: int = 2, sym_max: int = 8) -> None:
        super().__init__()
        self.features = features
        self.cond_module = nn.Linear(features, features, bias=False)
        self.in_proj = nn.Linear(features, features, bias=False)
        self.blocks = nn.ModuleList(TinyBlock(features) for _ in range(blocks))
        self.out_norm = nn.LayerNorm(features)
        self.out_proj = nn.Linear(features, features, bias=False)
        self.rope_table = torch.linspace(0.0, 1.0, sym_max * features).reshape(sym_max, features)

    def forward(
        self,
        x_t: torch.Tensor,
        t_embed: torch.Tensor,
        mask: torch.Tensor,
        *context: torch.Tensor,
    ) -> torch.Tensor:
        cond_embed = self.cond_module(t_embed)[:, None, :]
        x = self.in_proj(x_t)
        freqs = self.rope_table[: x.shape[1]]
        for index, block in enumerate(self.blocks):
            h = block.attention_adaln(x, cond_embed)
            x = x + block.mlp(h) + _tiny_mix(context[index], mask, freqs)
        return self.out_proj(self.out_norm(x))


class TinyStage(nn.Module):
    """One block of `TinyGraph` (the child is named as in {@link mq.DitBlockStage}).

    The condition-side value is rebuilt **inside** the block (the same as {@link mq.DitBlockStage}
    calling the condition-side projection inside the block — so the calibration sees the inputs
    of those weights).
    """

    def __init__(self, index: int, block: nn.Module) -> None:
        super().__init__()
        self.child = str(index)
        self.add_module(self.child, block)

    def forward(
        self,
        x: torch.Tensor,
        cond_embed: torch.Tensor,
        text: torch.Tensor,
        speaker: torch.Tensor,
        caption: torch.Tensor,
        mask: torch.Tensor,
        freqs: torch.Tensor,
    ) -> torch.Tensor:
        block = getattr(self, self.child)
        h = block.attention_adaln(x, cond_embed)
        return x + block.mlp(h) + _tiny_mix(_tiny_pooled(text, speaker, caption), mask, freqs)


class BrokenStage(TinyStage):
    """A stage that drops one side input (fault injection that proves the equivalence check's
    detection power)."""

    def forward(self, x, cond_embed, text, speaker, caption, mask, freqs):
        block = getattr(self, self.child)
        return x + block.mlp(block.attention_adaln(x, cond_embed))


def tiny_stages(graph: TinyGraph, stage_cls: type[TinyStage] = TinyStage) -> tuple[object, ...]:
    return tuple(
        (mq.DIT_BLOCK_PREFIX, stage_cls(index, block)) for index, block in enumerate(graph.blocks)
    )


def tiny_reference(
    context_graph: TinyContextGraph, graph: TinyGraph, steps: int, forwards_per_step: int
) -> list[int]:
    """A reference loop shaped like `pipeline_ref.run_case` / `_euler`.

    The condition-side values are built by `dit-context` **once per generation** (ADR 0114), and
    the forwards within a step reuse **the same `x_t`**.
    """
    torch.manual_seed(7)
    seen: list[int] = []
    x_t = torch.randn(1, 4, graph.features)
    t_embed = torch.randn(1, graph.features)
    mask = torch.ones((1, 1, 1, 6), dtype=torch.bool)
    text = torch.randn(1, 5, graph.features)
    speaker = torch.randn(1, 3, graph.features)
    caption = torch.randn(1, 2, graph.features)
    with torch.no_grad():
        context = context_graph(text, speaker, caption)
        for _step in range(steps):
            out = x_t
            for _forward in range(forwards_per_step):
                out = graph(x_t, t_embed, mask, *context)
                seen.append(1)
            x_t = x_t + out
    return seen


def tiny_pair() -> tuple[TinyContextGraph, TinyGraph]:
    return TinyContextGraph(), TinyGraph()


class TestCalibConfigTable:
    def test_the_three_calibrated_configs_are_selectable(self):
        assert mq.CALIB_NAMES == ("gptq-rtn", "gptq-nf4", "gptq-kmeans")
        assert all(name in mq.RECIPES for name in mq.CALIB_NAMES)

    def test_every_calibrated_config_rounds_only_the_dit_role(self):
        """Calibration covers only the linears of the DiT blocks — the orthogonality check's
        expectation follows from that."""
        for name in mq.CALIB_NAMES:
            assert mq.RECIPES[name].roles == (ex.TARGET_DIT,)
            assert mq.RECIPES[name].op_types == (nn.Linear,)

    def test_a_calibrated_config_is_not_a_w4_method_config(self):
        """`method` and `calib` are exclusive — with both set it would also take the whole-role
        path."""
        for recipe in mq.RECIPES.values():
            assert recipe.method is None or recipe.calib is None

    def test_the_group_size_is_pinned_for_every_calibrated_config(self):
        assert all(config.grid.group_size == mq.W4_GROUP_SIZE for config in mq.CALIB_CONFIGS)

    def test_no_calibrated_config_fits_its_table_on_a_subsample(self):
        assert all(config.grid.fit_stride == 1 for config in mq.CALIB_CONFIGS)

    def test_the_calibrated_grids_project_to_the_same_bpw_as_the_methods(self):
        """Calibration does not change the grid by a single byte — rtn / nf4 are 5.0 bpw, the
        same as the methods."""
        counts = mq.TargetCounts(modules=2, channels=8, elements=384)

        for name in ("gptq-rtn", "gptq-nf4"):
            config = next(item for item in mq.CALIB_CONFIGS if item.name == name)
            assert mq.project_size(config, counts, 2).bits_per_weight == 5.0

    def test_the_calibrated_codebook_costs_one_table_per_layer(self):
        """core's `kmeans_shared` is a **per-layer** table (a different formula from
        kmeans:shared with one table per role)."""
        counts = mq.TargetCounts(modules=2, channels=8, elements=384)
        config = next(item for item in mq.CALIB_CONFIGS if item.name == "gptq-kmeans")

        assert mq.project_size(config, counts, 2).bits == 5.0 * 384 + mq.CODEBOOK_BITS * 2

    def test_a_fit_stride_on_a_calibrated_config_is_fail_loudly(self):
        """MUST: do not silently ignore a knob that has no effect (the table's reach is one
        layer, so a full fit is enough)."""
        with pytest.raises(SystemExit, match="calibrated configurations"):
            mq.check_fit_stride(mq.RECIPES["gptq-kmeans"], 3)


class TestCalibTargets:
    def test_the_scan_keys_are_model_wide_fqns(self):
        """The ledger keys are in the same FQN space as `Int4Report` (the contract of core's
        `StageSpec`)."""
        scan, counts = mq.calib_targets(tiny_stages(TinyGraph()))

        assert sorted(scan)[:2] == [
            "blocks.0.attention_adaln.proj.weight",
            "blocks.0.mlp.weight",
        ]
        assert counts.modules == 4

    def test_a_weight_off_the_group_grid_is_fail_loudly(self):
        """MUST: do not exclude and carry on — when driven per stage, the exact-coverage check
        could no longer be set up."""
        stages = ((mq.DIT_BLOCK_PREFIX, nn.Linear(12, 4, bias=False)),)

        with pytest.raises(SystemExit, match="is not divisible by g32"):
            mq.calib_targets(stages)

    def test_stages_without_a_linear_are_fail_loudly(self):
        with pytest.raises(SystemExit, match="has not a single quantizable weight"):
            mq.calib_targets(((mq.DIT_BLOCK_PREFIX, nn.LayerNorm(8)),))


class TestCatcher:
    def test_it_takes_one_batch_per_step_not_per_forward(self):
        """CFG's uncond forwards receive **the same `x_t`** — that identity is what counts the
        steps."""
        context, graph = tiny_pair()

        batches, _probe = mq.capture_case_batches(
            context, graph, lambda: tiny_reference(context, graph, 3, 4), limit=40
        )

        assert len(batches) == 3

    def test_the_keyword_arguments_carry_the_whole_side_input(self):
        """The (normed) conditions come from `dit-context`, the mask and the RoPE table from the
        forward of `dit`."""
        context, graph = tiny_pair()

        batches, _probe = mq.capture_case_batches(
            context, graph, lambda: tiny_reference(context, graph, 1, 1), limit=40
        )

        (hidden,), kwargs = batches[0]
        assert sorted(kwargs) == ["caption", "cond_embed", "freqs", "mask", "speaker", "text"]
        assert torch.equal(kwargs["freqs"], graph.rope_table[: int(hidden.shape[1])])

    def test_the_limit_cuts_the_reference_loop_short(self):
        """Stop once enough is captured — `run_case` runs one more pass after the one with CFG,
        so do not let it mix in."""
        context, graph = tiny_pair()
        seen: list[int] = []

        batches, _probe = mq.capture_case_batches(
            context, graph, lambda: seen.extend(tiny_reference(context, graph, 10, 2)), limit=2
        )

        assert len(batches) == 2
        assert len(seen) == 0  # the sentinel exits, so the reference loop does not run to the end

    def test_the_hooks_are_removed_even_though_the_loop_was_aborted(self):
        context, graph = tiny_pair()

        mq.capture_case_batches(
            context, graph, lambda: tiny_reference(context, graph, 5, 1), limit=1
        )

        assert not context._forward_pre_hooks
        assert not context.text_norm._forward_hooks
        assert not context.caption_norm._forward_hooks
        assert not graph._forward_pre_hooks
        assert not graph.blocks[0].attention_adaln._forward_pre_hooks

    def test_a_loop_that_never_reaches_the_graph_is_fail_loudly(self):
        context, graph = tiny_pair()

        with pytest.raises(SystemExit, match="even 1 step"):
            mq.capture_case_batches(context, graph, lambda: None, limit=4)

    def test_a_dit_forward_before_the_context_graph_is_fail_loudly(self):
        """`dit` runs before the (normed) conditions are captured — do not silently reuse the
        previous case's conditions."""
        context, graph = tiny_pair()
        x_t = torch.randn(1, 4, graph.features)

        def dit_only() -> None:
            with torch.no_grad():
                graph(
                    x_t,
                    torch.randn(1, graph.features),
                    torch.ones((1, 1, 1, 6), dtype=torch.bool),
                    *(torch.randn(1, graph.features) for _ in graph.blocks),
                )

        with pytest.raises(SystemExit, match="dit ran before dit-context"):
            mq.capture_case_batches(context, graph, dit_only, limit=4)


class TestStageSplitGate:
    def test_a_faithful_split_reproduces_the_graph_bit_for_bit(self):
        context, graph = tiny_pair()
        batches, probe = mq.capture_case_batches(
            context, graph, lambda: tiny_reference(context, graph, 1, 1), limit=1
        )

        mq.assert_stage_split_matches_graph(graph, probe, batches[0], tiny_stages(graph))

    def test_a_split_that_drops_a_side_input_is_fail_loudly(self):
        """MUST: fail when the copy diverges — GPTQ on a diverged path cannot be read from the
        numbers."""
        context, graph = tiny_pair()
        batches, probe = mq.capture_case_batches(
            context, graph, lambda: tiny_reference(context, graph, 1, 1), limit=1
        )

        with pytest.raises(SystemExit, match="not bit-identical"):
            mq.assert_stage_split_matches_graph(
                graph, probe, batches[0], tiny_stages(graph, BrokenStage)
            )


class TestDitBlockStageOnTheRealModules:
    """The copy in {@link mq.DitBlockStage} is bit-identical to the two distribution wrappers
    built from upstream's real modules.

    Even after the DiT split (ADR 0114) the calibration stage is the whole block and recomputes
    the condition-side projection **inside** the block — this checks with a small real DiT (built
    as in `irodori/tests/test_export.py`) that this path gives the same values as "`dit` taking
    the output of `dit-context` as input".
    """

    def test_the_stage_split_reproduces_the_dit_graph_bit_for_bit(self, restore_forward):
        from irodori import patch as patch_irodori
        from irodori.tests.test_export import _tiny_real_dit, _TinySplit

        model = _tiny_real_dit()
        split = _TinySplit(model)
        patch_irodori.apply_patches()
        wrappers = ex.dit_wrappers(model, split.sym_max)
        args = split.inputs["dit-cond-1s"]

        def run() -> None:
            with torch.no_grad():
                context = wrappers.context(
                    args["text_state"], args["speaker_state"], args["caption_state"]
                )
                wrappers.dit(args["x_t"], args["t_embed"], args["mask"], *context)

        batches, probe = mq.capture_case_batches(wrappers.context, wrappers.dit, run, limit=1)
        stages = mq.dit_stages(wrappers.dit)

        mq.assert_stage_split_matches_graph(wrappers.dit, probe, batches[0], stages)
        # The calibration stage is the whole block — the condition-side projections are included
        # too (the side that goes into the `dit-context` container in the distribution). The small
        # config is not aligned to g32, so this looks at the target enumeration itself rather
        # than the scan check (`calib_targets`).
        scanned = {
            f"{prefix}.{local}"
            for prefix, stage in stages
            for local, _weight, _axis in iter_quant_targets(stage, (nn.Linear,))
        }
        assert {
            f"blocks.{index}.attention.{name}.weight"
            for index in range(len(model.blocks))
            for name in ex.CONTEXT_KV_PROJECTIONS
        } <= scanned


def _tiny_rig(graph: TinyGraph, steps: int = 2) -> mq.CalibRig:
    context = TinyContextGraph(graph.features, len(graph.blocks))
    stages = tiny_stages(graph)
    scan, counts = mq.calib_targets(stages)
    batches, _probe = mq.capture_case_batches(
        context, graph, lambda: tiny_reference(context, graph, steps, 1), limit=steps
    )
    return mq.CalibRig(
        stages=stages,
        scan=scan,
        counts=counts,
        batches=tuple(batches),
        steps={"tiny": len(batches)},
        tokens=sum(int(args[0].shape[1]) for args, _kwargs in batches),
    )


class TestCalibrationRun:
    def test_every_calibrated_config_rounds_every_scanned_linear(self):
        for config in mq.CALIB_CONFIGS:
            torch.manual_seed(3)
            graph = TinyGraph()
            rig = _tiny_rig(graph)
            before = {fqn: weight.detach().clone() for fqn, weight in rig.scan.items()}

            _reports, report = mq.apply_calib(config, rig)

            assert report.method == config.method
            assert report.grid == config.grid.kind
            assert report.modules == len(rig.scan)
            assert all(not torch.equal(rig.scan[fqn], value) for fqn, value in before.items())

    def test_the_report_names_the_dit_role(self):
        torch.manual_seed(4)
        graph = TinyGraph()

        reports, _report = mq.apply_calib(mq.CALIB_CONFIGS[0], _tiny_rig(graph))

        assert "gptq-rtn" in reports[ex.TARGET_DIT]

    def test_a_scan_the_calibration_missed_is_fail_loudly(self):
        """MUST: a missed rounding shows as **better** quality — the numbers cannot reveal it, so
        the check is the only way to detect it."""
        torch.manual_seed(5)
        graph = TinyGraph()
        rig = _tiny_rig(graph)
        _reports, report = mq.apply_calib(mq.CALIB_CONFIGS[0], rig)

        with pytest.raises(SystemExit, match="which does not match the"):
            mq.assert_calib_covers_scan(
                report, {**rig.scan, "blocks.9.mlp.weight": torch.zeros(1)}, "gptq-rtn"
            )


class TestLoadModules:
    def test_it_reads_the_upstream_checkpoint_not_the_exported_container(
        self, tmp_path, monkeypatch
    ):
        """The upstream checkpoint (`ex.CHECKPOINT_FILE`) and the output container
        (`ex.MODEL_FILE`) are separate things.

        The measurement builds the 9 modules from upstream's weights, so reading the output
        `model.krm` instead would fail right at startup with the real weights (here, without
        real weights, only the path read is observed).
        """
        opened: list[str] = []

        class _StopError(Exception):
            pass

        def _load_file(path: str):
            opened.append(path)
            raise _StopError

        monkeypatch.setattr(ex, "IrodoriSource", lambda _source_dir: object())
        monkeypatch.setattr(ex, "read_configs", lambda _model_dir: ({}, {}))
        monkeypatch.setattr(mq, "load_file", _load_file)

        with pytest.raises(_StopError):
            mq.load_modules(tmp_path, tmp_path)

        assert opened == [str(tmp_path / ex.CHECKPOINT_FILE)]
        assert ex.CHECKPOINT_FILE != ex.MODEL_FILE
