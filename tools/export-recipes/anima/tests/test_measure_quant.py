"""The rules of the parts of `anima/measure_quant.py` that **need no real weights**.

The measurement itself is manual (the 7.29GiB DiT is rounded and run once per configuration).
What is pinned here is only what, when broken, raises nothing at run time and **silently makes
the measurements lie**:

- the symbols, order and baseline of the 10 configurations **stay as they are** (ADR 0030 /
  ACTIVE_DESIGN refer to configurations by name — stability of external references)
- `--w4-screen` **does not stack methods** (applying one without restoring would measure "MXFP4
  on top of NF4", and the method comparison itself would not hold)
- the properties of the target set — groups are counted on the same axis as "the axis the
  rounding splits into groups"; weights not divisible by g32 are excluded and **listed**; the
  exclusion actually reaches the rounding
- fail loudly when a non-linear quantizable type appears (the premise of this script is linear
  only = matches i4's execution path)
- the size-projection formulas — with the g32 group scale, rtn / nf4 are 5.0 bpw, mxfp4 is
  4.25 bpw, kmeans:shared adds one table (where the numbers in the table come from)
- the derivation of `kmeans:shared`'s fit_stride (the whole set within the budget; at large scale
  an odd number coprime with the group length)
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from anima import measure_quant as mq
from karume.quantize import iter_quant_targets


class TinyDit(nn.Module):
    """2 linears divisible by g32 + 1 that is not (= a scaled-down `patch_embed.proj`)."""

    def __init__(self) -> None:
        super().__init__()
        self.patch_embed = nn.Linear(68, 8, bias=False)
        self.attn = nn.Linear(64, 32, bias=False)
        self.ff = nn.Linear(32, 64, bias=False)


def _method(name: str) -> mq.W4Method:
    return next(method for method in mq.W4_METHODS if method.name == name)


def _targets(**overrides: int) -> mq.W4Targets:
    """By default the minimal form "32 elements = 1 group, 1 module" (no exclusions)."""
    base = {"modules": 1, "elements": 32, "groups": 1}
    return mq.W4Targets(excluded={}, **{**base, **overrides})


class TestExistingConfigsAreFrozen:
    """MUST: the symbols, order and baseline of the 10 configurations stay as they are (ADR 0030
    refers to configurations by name)."""

    def test_the_ten_configs_keep_their_names_and_order(self):
        assert mq.CONFIG_NAMES == (
            "f32",
            "w8",
            "w8a8",
            "w8a8-qk",
            "w8a8-pv",
            "w8a8-qkpv",
            "w8-qkpv",
            "w8a8-qkpv-s16",
            "w8a8-s16",
            "w8a8-qkpv-statsq",
        )

    def test_the_baseline_is_still_w8a8(self):
        assert mq.BASELINE == "w8a8"

    def test_the_w4_screen_does_not_share_the_stacking_table(self):
        """The method screening is a separate path — it borrows nothing from the table of 10
        configurations."""
        assert set(mq.W4_METHOD_NAMES).isdisjoint(mq.CONFIG_NAMES)


class TestTargetScan:
    def test_it_counts_groups_on_the_axis_the_rounding_splits(self):
        """Groups are counted from the flattened form of `channel_rows` (= the axis the rounding
        splits into groups)."""
        targets = mq.scan_w4_targets(TinyDit())

        # attn `[32,64]` … 32 rows × (64/32) groups / ff `[64,32]` … 64 rows × 1 group.
        assert (targets.modules, targets.elements) == (2, 32 * 64 + 64 * 32)
        assert targets.groups == 32 * 2 + 64

    def test_an_indivisible_quantization_axis_is_excluded_and_listed(self):
        """MUST: a partial group is a shape that cannot be stored (ADR 0069 decision 2) — do not
        silently give up on the whole."""
        targets = mq.scan_w4_targets(TinyDit())

        assert targets.excluded == {"patch_embed.weight": 68}
        assert not targets.include("patch_embed")
        assert targets.include("attn")

    def test_a_non_linear_quantizable_weight_fails_loudly(self):
        """If conv / embedding appear, the premise "linear only = the form the distribution
        supports" breaks."""

        class WithConv(TinyDit):
            def __init__(self) -> None:
                super().__init__()
                self.conv = nn.Conv2d(4, 8, 2)

        with pytest.raises(AssertionError, match="非 linear"):
            mq.scan_w4_targets(WithConv())

    def test_a_fully_indivisible_model_fails_loudly(self):
        class Odd(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.fc = nn.Linear(30, 2, bias=False)

        with pytest.raises(AssertionError, match="1 本も無い"):
            mq.scan_w4_targets(Odd())

    def test_the_include_predicate_matches_the_root_module_spelling(self):
        """The predicate is called with **module FQNs** (the root is the empty string → the weight
        FQN is `weight`)."""
        targets = mq.scan_w4_targets(nn.Linear(64, 8, bias=False))

        assert targets.modules == 1
        assert targets.include("")


class TestExclusionReachesTheRounding:
    """The exclusion must not only "be listed" but **reach the rounding** (being listed alone leaves
    silently wrong values)."""

    def test_the_excluded_weight_is_untouched_by_every_method(self):
        torch.manual_seed(0)
        for method in mq.W4_METHODS:
            model = TinyDit()
            targets = mq.scan_w4_targets(model)
            skipped = model.patch_embed.weight.detach().clone()
            rounded = model.attn.weight.detach().clone()

            report = method.apply(model, targets.include, 1)

            assert report.modules == targets.modules == 2
            assert torch.equal(model.patch_embed.weight, skipped)
            # The targets actually moved (= the predicate is not broken towards "exclude
            # everything").
            assert not torch.equal(model.attn.weight, rounded)

    def test_without_the_predicate_the_indivisible_weight_would_fail_loudly(self):
        """Prevents a vacuous test — without the exclusion, core fails on the partial group (the
        condition for the test above to mean something)."""
        model = TinyDit()

        with pytest.raises(Exception, match="is not divisible"):
            _method("rtn-i4-g32").apply(model, lambda _name: True, 1)


class TestMethodsAreNotStacked:
    """MUST: restore to pristine per configuration (without it, the comparison between methods does
    not hold)."""

    def _pristine(self, model: nn.Module, targets: mq.W4Targets):
        weights = {
            fqn: weight
            for fqn, weight, _axis in iter_quant_targets(model, mq.W4_OP_TYPES)
            if fqn not in targets.excluded
        }
        return weights, {fqn: weight.detach().clone() for fqn, weight in weights.items()}

    def test_restore_brings_back_the_original_weights_bit_for_bit(self):
        torch.manual_seed(0)
        model = TinyDit()
        targets = mq.scan_w4_targets(model)
        weights, pristine = self._pristine(model, targets)

        _method("nf4").apply(model, targets.include, 1)
        mq.restore_w4(weights, pristine)

        assert all(torch.equal(weights[fqn], pristine[fqn]) for fqn in pristine)

    def test_a_restored_run_matches_a_fresh_one(self):
        """rtn applied after restoring from another method is bit-identical to rtn applied to the
        original weights."""
        torch.manual_seed(0)
        model = TinyDit()
        targets = mq.scan_w4_targets(model)
        weights, pristine = self._pristine(model, targets)

        _method("mxfp4").apply(model, targets.include, 1)
        mq.restore_w4(weights, pristine)
        _method("rtn-i4-g32").apply(model, targets.include, 1)
        after_restore = weights["attn.weight"].detach().clone()

        mq.restore_w4(weights, pristine)
        _method("rtn-i4-g32").apply(model, targets.include, 1)

        assert torch.equal(after_restore, weights["attn.weight"])

    def test_stacking_the_two_methods_would_have_changed_the_answer(self):
        """Prevents a vacuous test — stacking without restoring actually gives different values."""
        torch.manual_seed(0)
        model = TinyDit()
        targets = mq.scan_w4_targets(model)
        weights, pristine = self._pristine(model, targets)

        _method("mxfp4").apply(model, targets.include, 1)
        _method("rtn-i4-g32").apply(model, targets.include, 1)
        stacked = weights["attn.weight"].detach().clone()

        mq.restore_w4(weights, pristine)
        _method("rtn-i4-g32").apply(model, targets.include, 1)

        assert not torch.equal(stacked, weights["attn.weight"])


class TestSizeProjection:
    """Where the bpw in the table comes from (`int4-sym-g` of `docs/container-v1.md` §6.3 and OCP
    MX, to the letter)."""

    @pytest.mark.parametrize("name", ["rtn-i4-g32", "nf4"])
    def test_group_absmax_methods_land_on_five_bits(self, name):
        """4-bit payload + an F32 scale per group of 32 = 4 + 32/32 = 5.0 bpw."""
        projection = mq.w4_size_projection(_method(name), _targets(elements=3200, groups=100))

        assert projection["bitsPerWeight"] == pytest.approx(5.0)

    def test_mxfp4_lands_on_four_and_a_quarter_bits(self):
        """The shared scale is E8M0 (8 bits), so 4 + 8/32 = 4.25 bpw."""
        projection = mq.w4_size_projection(_method("mxfp4"), _targets(elements=3200, groups=100))

        assert projection["bitsPerWeight"] == pytest.approx(4.25)

    def test_the_shared_codebook_costs_one_table_for_the_whole_model(self):
        """One table, but group scales are needed (= the same 5.0 bpw as i4 + one table)."""
        targets = _targets(modules=10, elements=3200, groups=100)

        projection = mq.w4_size_projection(_method("kmeans:shared"), targets)

        assert projection["bitsPerWeight"] == pytest.approx(5.0 + 32 * 16 / 3200)

    def test_the_ratio_is_the_bpw_against_f32(self):
        """The ratio is "what % of f32 storage" — derived from the same numbers as the two MiB
        columns."""
        projection = mq.w4_size_projection(_method("nf4"), _targets(elements=3200, groups=100))

        assert projection["ratio"] == pytest.approx(5.0 / 32.0)
        assert projection["projectedMiB"] == pytest.approx(projection["f32MiB"] * 5.0 / 32.0)

    def test_one_mebibyte_is_the_binary_one(self):
        """2^20, as the `MiB` notation says (mixing in 10^6 shifts the table's numbers by 5%)."""
        assert mq.MIB == 2**20


class TestKmeansFitStride:
    def test_a_model_inside_the_budget_fits_on_the_full_set(self):
        assert mq.w4_fit_stride(mq.W4_KMEANS_FIT_BUDGET) == 1

    def test_the_sample_stays_inside_the_budget(self):
        elements = 1_956_249_600  # Anima DiT, measured (453 linears, after excluding 1)

        stride = mq.w4_fit_stride(elements)

        assert elements // stride <= mq.W4_KMEANS_FIT_BUDGET

    def test_the_stride_is_coprime_with_the_group_size(self):
        """MUST: an even stride touches only the even lanes of a group (the group length is a power
        of 2)."""
        for elements in (mq.W4_KMEANS_FIT_BUDGET * 2, mq.W4_KMEANS_FIT_BUDGET * 97 + 1):
            assert mq.w4_fit_stride(elements) % 2 == 1


class TestMethodTable:
    def test_the_screened_winners_are_the_four_methods(self):
        assert mq.W4_METHOD_NAMES == ("rtn-i4-g32", "nf4", "mxfp4", "kmeans:shared")

    def test_the_group_size_is_pinned_for_every_method(self):
        """The only axis of the method comparison is "how the grid is laid out" — the group length
        does not move (ADR 0069 addendum 5)."""
        assert mq.W4_GROUP_SIZE == 32

    def test_selecting_a_subset_keeps_the_declaration_order(self):
        chosen = mq.select_w4_methods(["mxfp4", "rtn-i4-g32"])

        assert [method.name for method in chosen] == ["rtn-i4-g32", "mxfp4"]

    def test_an_empty_selection_runs_everything(self):
        assert mq.select_w4_methods([]) == mq.W4_METHODS


def _payload(**overrides: object) -> dict[str, object]:
    entry = {
        "method": "nf4",
        "quantReport": "nf4: modules 453 / 1,956,249,600 elements",
        "elapsed": 612.0,
        "latentRelRms": [1.5e-3, 2.5e-3],
        "psnrF32": 31.25,
        "psnrUint8": 31.3,
        "imageRelRms": 4.2e-2,
        "uint8MaxDiff": 41.0,
        "moved": True,
        "size": {
            "modules": 453,
            "elements": 3200,
            "groups": 100,
            "bitsPerWeight": 5.0,
            "projectedMiB": 1.9,
            "f32MiB": 12.2,
            "ratio": 0.15625,
            "formula": "4·N + 32·G",
        },
    }
    base = {
        "generated": "2026-08-19 12:00:00",
        "script": "measure_quant.py --w4-screen",
        "torch": "2.9.0",
        "prompt": "1girl",
        "steps": 2,
        "resolution": 512,
        "guidanceScale": 1.0,
        "seed": 20260802,
        "groupSize": 32,
        "kmeansFitStride": 99,
        "kmeansFitSamplesApprox": 19_760_096,
        "targets": {"opTypes": ["Linear"], "modules": 453, "elements": 3200, "groups": 100},
        "excluded": [{"weight": "patch_embed.proj.weight", "quantAxis": 68}],
        "imageNote": "image_*.png",
        # The baseline row has no quality columns (comparing with itself carries no information,
        # and PSNR ∞ is invalid in JSON too).
        "configs": {
            "f32": {"method": "f32", "quantReport": "no rounding (baseline)", "elapsed": 9.0},
            "nf4": entry,
        },
    }
    return {**base, **overrides}


class TestReport:
    """The table is built from the payload alone (so that the JSON and the markdown cannot
    disagree)."""

    def test_it_carries_the_size_formula_and_the_quality_row(self):
        report = mq.w4_report_markdown(_payload())

        assert "| `nf4` | 5.000 | 1.9 | 12.2 | 15.6% | 4·N + 32·G |" in report
        assert "| `nf4` | 31.25 | 31.30 | 4.2000e-02 | 41 | 612s |" in report

    def test_the_excluded_weight_is_named_in_the_report(self):
        """MUST: state the exclusions in the output (do not silently give up on the whole)."""
        report = mq.w4_report_markdown(_payload())

        assert "`patch_embed.proj.weight`（軸 68）" in report

    def test_a_sampled_codebook_fit_is_stated(self):
        """MUST: state it when fit_stride was used (the MUST of the core docstring)."""
        report = mq.w4_report_markdown(_payload())

        assert "1/99 の等間隔部分標本" in report
        assert "適用は全量" in report

    def test_a_full_fit_says_nothing_about_sampling(self):
        report = mq.w4_report_markdown(_payload(kmeansFitStride=1))

        assert "部分標本" not in report

    def test_a_config_that_did_not_move_is_marked_up(self):
        payload = _payload()
        payload["configs"]["nf4"] = {**payload["configs"]["nf4"], "moved": False}

        assert "**NG（素通り）**" in mq.w4_report_markdown(payload)


# ---- calibrated rounding (iteration J-2) ------------------------------------
#
# Without real weights, only "how the targets are split", "how inputs are captured" and "the
# checks" can be pinned. The GPTQ numbers themselves belong to the tests of core
# (`karume.quant_calib`).


class TinyCalibBlock(nn.Module):
    """Stand-in for `CosmosTransformerBlock` — copies **the names and order of the arguments** (the
    basis of the Catcher's mapping)."""

    def __init__(self, features: int) -> None:
        super().__init__()
        self.attn1 = nn.Linear(features, features, bias=False)
        self.ff = nn.Linear(features, features, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        embedded_timestep: torch.Tensor,
        temb: torch.Tensor | None = None,
        image_rotary_emb: torch.Tensor | None = None,
        extra_pos_emb: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        controlnet_residual: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden = self.attn1(hidden_states) + encoder_hidden_states.mean(dim=1)[:, None, :]
        hidden = hidden + self.ff(hidden) + embedded_timestep[:, None, :]
        return hidden_states + hidden


class TinyCalibDit(nn.Module):
    """Stand-in for `CosmosTransformer3DModel` (also copies calling the blocks with **all positional
    arguments**)."""

    def __init__(self, features: int = 32, blocks: int = 2) -> None:
        super().__init__()
        self.features = features
        self.transformer_blocks = nn.ModuleList(TinyCalibBlock(features) for _ in range(blocks))

    def forward(
        self,
        hidden: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        embedded_timestep: torch.Tensor,
        temb: torch.Tensor,
    ) -> torch.Tensor:
        for block in self.transformer_blocks:
            hidden = block(
                hidden, encoder_hidden_states, embedded_timestep, temb, None, None, None, None
            )
        return hidden


def tiny_denoise(model: TinyCalibDit, steps: int) -> dict[str, torch.Tensor]:
    """A reference loop of the same shape as `reference_steps` (1 step = 1 DiT forward —
    CFG=1)."""
    torch.manual_seed(11)
    x = torch.randn(1, 4, model.features)
    encoder = torch.randn(1, 5, model.features)
    embedded = torch.randn(1, model.features)
    temb = torch.randn(1, model.features)
    out: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for index in range(steps):
            x = x + model(x, encoder, embedded, temb)
            out[f"latents_step{index + 1:04d}"] = x
    return out


def _calib_rig(model: TinyCalibDit, steps: int = 2) -> mq.CalibRig:
    stages = mq.calib_stages(model)
    scan, targets = mq.scan_calib_targets(stages)
    _result, batches = mq.capture_stage_batches(model, lambda: tiny_denoise(model, steps), steps)
    return mq.CalibRig(stages=stages, scan=scan, targets=targets, batches=batches)


class TestCalibConfigTable:
    def test_the_three_calibrated_configs_are_selectable(self):
        assert mq.CALIB_NAMES == ("gptq-rtn", "gptq-nf4", "gptq-kmeans")

    def test_selecting_a_subset_keeps_the_declaration_order(self):
        chosen = mq.select_calib(["gptq-kmeans", "gptq-rtn"])

        assert [config.name for config in chosen] == ["gptq-rtn", "gptq-kmeans"]

    def test_an_empty_selection_runs_every_calibrated_config(self):
        assert mq.select_calib([]) == mq.CALIB_CONFIGS

    def test_selecting_only_a_method_leaves_no_calibrated_config(self):
        """`--w4-only nf4` selects only the method — it does not pull in calibrated
        configurations."""
        assert mq.select_calib(["nf4"]) == ()

    def test_selecting_only_a_calibrated_config_leaves_no_method(self):
        assert mq.select_w4_methods(["gptq-rtn"]) == ()

    def test_the_group_size_is_pinned_for_every_calibrated_config(self):
        assert all(config.grid.group_size == mq.W4_GROUP_SIZE for config in mq.CALIB_CONFIGS)

    def test_no_calibrated_config_fits_its_table_on_a_subsample(self):
        """The calibration's tables are laid out **per layer**, so a full fit fits in the working
        memory (no stride needed)."""
        assert all(config.grid.fit_stride == 1 for config in mq.CALIB_CONFIGS)

    def test_the_calibrated_grids_project_to_the_same_bpw_as_the_methods(self):
        """Calibration changes not one byte of the grid — rtn / nf4 are 5.0 bpw, the same as the
        methods."""
        for name in ("gptq-rtn", "gptq-nf4"):
            config = next(item for item in mq.CALIB_CONFIGS if item.name == name)
            assert mq.w4_size_projection(config, _targets())["bitsPerWeight"] == 5.0

    def test_the_calibrated_codebook_costs_one_table_per_layer(self):
        config = next(item for item in mq.CALIB_CONFIGS if item.name == "gptq-kmeans")
        targets = _targets(modules=3, elements=3200, groups=100)

        projection = mq.w4_size_projection(config, targets)

        tables = mq.W4_CODEBOOK_ENTRY_BITS * mq.DEFAULT_CODEBOOK_LEVELS * 3
        assert projection["bitsPerWeight"] == pytest.approx(5.0 + tables / 3200)


class TestCalibTargets:
    def test_the_stage_prefix_carries_the_block_index(self):
        """Prepending it to a local FQN (`attn1.weight`) gives back the model-wide FQN (core's
        `StageSpec`)."""
        stages = mq.calib_stages(TinyCalibDit())

        assert [prefix for prefix, _stage in stages] == [
            "transformer_blocks.0",
            "transformer_blocks.1",
        ]

    def test_the_scan_keys_are_model_wide_fqns(self):
        scan, targets = mq.scan_calib_targets(mq.calib_stages(TinyCalibDit()))

        assert sorted(scan)[:2] == [
            "transformer_blocks.0.attn1.weight",
            "transformer_blocks.0.ff.weight",
        ]
        assert targets.modules == 4
        assert targets.groups == 4 * 32 * (32 // mq.W4_GROUP_SIZE)

    def test_a_non_linear_quantizable_weight_is_fail_loudly(self):
        """Linear only = matches i4's execution path is the premise of this column (the same text as
        `scan_w4_targets`)."""
        stages = (("transformer_blocks.0", nn.Conv1d(32, 8, 3, bias=False)),)

        with pytest.raises(AssertionError, match="非 linear"):
            mq.scan_calib_targets(stages)

    def test_a_weight_off_the_group_grid_is_fail_loudly(self):
        """MUST: do not exclude and continue — driving per stage would make the exact-coverage check
        impossible."""
        stages = (("transformer_blocks.0", nn.Linear(12, 4, bias=False)),)

        with pytest.raises(AssertionError, match="割り切れない"):
            mq.scan_calib_targets(stages)

    def test_stages_without_a_linear_are_fail_loudly(self):
        with pytest.raises(AssertionError, match="1 本も無い"):
            mq.scan_calib_targets((("transformer_blocks.0", nn.LayerNorm(8)),))


class TestCatcher:
    def test_it_takes_one_batch_per_step(self):
        model = TinyCalibDit()

        _result, batches = mq.capture_stage_batches(model, lambda: tiny_denoise(model, 3), 3)

        assert len(batches) == 3

    def test_the_positional_arguments_are_named_from_the_block_signature(self):
        """No copied list of names is kept — even if upstream reorders them, the spellings do not
        silently swap."""
        model = TinyCalibDit()

        _result, batches = mq.capture_stage_batches(model, lambda: tiny_denoise(model, 1), 1)

        _args, kwargs = batches[0]
        assert list(kwargs) == [
            "encoder_hidden_states",
            "embedded_timestep",
            "temb",
            "image_rotary_emb",
            "extra_pos_emb",
            "attention_mask",
            "controlnet_residual",
        ]

    def test_the_limit_caps_the_batches_without_cutting_the_reference(self):
        """The baseline f32 run itself goes to the end (latents are needed for every step)."""
        model = TinyCalibDit()

        result, batches = mq.capture_stage_batches(model, lambda: tiny_denoise(model, 4), 2)

        assert len(batches) == 2
        assert len(result) == 4

    def test_the_hook_is_removed_afterwards(self):
        model = TinyCalibDit()

        mq.capture_stage_batches(model, lambda: tiny_denoise(model, 1), 1)

        assert not model.transformer_blocks[0]._forward_pre_hooks

    def test_a_reference_that_never_reaches_the_blocks_is_fail_loudly(self):
        with pytest.raises(AssertionError, match="1 step も"):
            mq.capture_stage_batches(TinyCalibDit(), dict, 2)


class TestCalibrationRun:
    def test_every_calibrated_config_rounds_every_scanned_linear(self):
        for config in mq.CALIB_CONFIGS:
            torch.manual_seed(13)
            model = TinyCalibDit()
            rig = _calib_rig(model)
            before = {fqn: weight.detach().clone() for fqn, weight in rig.scan.items()}

            report = mq.apply_calib(config, rig)

            assert report.method == config.method
            assert report.grid == config.grid.kind
            assert report.modules == len(rig.scan)
            assert all(not torch.equal(rig.scan[fqn], value) for fqn, value in before.items())

    def test_a_scan_the_calibration_missed_is_fail_loudly(self):
        """MUST: a missed rounding shows up as a **better** PSNR — the check is the only way to
        detect it."""
        torch.manual_seed(17)
        model = TinyCalibDit()
        rig = _calib_rig(model)
        report = mq.apply_calib(mq.CALIB_CONFIGS[0], rig)

        with pytest.raises(AssertionError, match="一致しない"):
            mq.assert_calib_covers_scan(
                report, {**rig.scan, "transformer_blocks.9.ff.weight": torch.zeros(1)}, "gptq-rtn"
            )


#: The target properties of the calibration column (a stand-in for what `run_w4_screen` writes
#: into the JSON).
CALIB_META = {
    "name": mq.CALIB_TARGET,
    "stages": 28,
    "modules": 280,
    "elements": 1_900_000_000,
    "groups": 59_375_000,
    "batches": 10,
}


class TestCalibReport:
    def test_the_calibrated_target_set_is_spelled_out(self):
        """MUST: the table states that the target set differs from the 4 methods' (neither bpw nor
        quality can be compared directly)."""
        report = mq.w4_report_markdown(_payload(calibTargets=CALIB_META))

        assert mq.CALIB_TARGET in report
        assert "280 本" in report

    def test_a_run_without_calibrated_configs_says_nothing_about_them(self):
        assert mq.CALIB_TARGET not in mq.w4_report_markdown(_payload(calibTargets=None))


def _stub_run() -> mq.DitRun:
    """The minimal `DitRun` to get through `main()` without real weights (nobody reads the
    numbers)."""
    return mq.DitRun(latents={}, linear_stats={}, attn_stats={}, diagnostics={}, attention_nodes=[])


def _stub_the_measurement(monkeypatch: pytest.MonkeyPatch, *, failed: list[str]) -> None:
    """Replace every measurement step of `main()` and fix only the summary to the given
    verdicts."""
    monkeypatch.setattr(mq, "run_dit", lambda _args: _stub_run())
    monkeypatch.setattr(mq, "write_layer_csv", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(mq, "decode_all", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(mq, "save_images", lambda *_args, **_kwargs: "image_*.png")
    monkeypatch.setattr(
        mq,
        "build_summary",
        lambda *_args, **_kwargs: {
            "pt": {},
            "configs": {},
            "gates": {label: "… → NG" for label in failed} or {"any": "OK"},
            "failed": failed,
            "inject": None,
        },
    )
    monkeypatch.setattr(mq, "build_report", lambda *_args, **_kwargs: "")


class TestTheQualityGateReachesTheExitCode:
    """This script fills quality gate (1) of ADR 0019, so red does not end with just a printout.

    The `--w4-screen` path (`main_w4`) has raised `AssertionError` from the start — if the two
    paths of the same script differed in strength, an automated run would miss only the red of
    the 10-configuration side.
    """

    def test_a_red_gate_ends_the_run_with_an_error(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _stub_the_measurement(monkeypatch, failed=["`w8a8` の SDPA 発火計数"])
        monkeypatch.setattr(
            "sys.argv", ["measure_quant.py", "--out", str(tmp_path), "--no-latents"]
        )

        with pytest.raises(AssertionError, match="品質ゲートが赤"):
            mq.main()

    def test_an_injected_failure_is_allowed_to_be_red(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`--inject` demonstrates the detection power; being red is the very purpose of this path
        (not blocked)."""
        _stub_the_measurement(monkeypatch, failed=["`w8a8` の SDPA 発火計数"])
        monkeypatch.setattr(
            "sys.argv",
            [
                "measure_quant.py",
                "--out",
                str(tmp_path),
                "--no-latents",
                "--inject",
                "drop-attn-quant",
            ],
        )

        mq.main()

    def test_all_green_ends_normally(self, tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
        _stub_the_measurement(monkeypatch, failed=[])
        monkeypatch.setattr(
            "sys.argv", ["measure_quant.py", "--out", str(tmp_path), "--no-latents"]
        )

        mq.main()
