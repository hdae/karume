"""The pure part of `embeddinggemma/measure_quant.py` (what needs no real weights).

The measurement itself is manual (rounding the 1.2 GB of real weights for 15 configs). What is
pinned here is only the side whose breakage makes **the measurements silently lie**:

- **Do not stack methods** — each config goes back to pristine (without it the run measures "MXFP4
  on top of NF4", and the comparison of methods itself does not hold)
- The size projection formulas — with the g=32 group scale, i4 / FP4 / NF4 come to 5.0 bpw, MXFP4
  to 4.25 bpw, and the codebooks include the cost of their tables (where the numbers in the table
  come from)
- The metrics — the boundary values and scale invariance of the cosine, the pair matrix being the
  10 pairs of the upper triangle, the detection of a collapsed semantic order
- The properties of the target set — the group count is taken on the same axis "the rounding splits
  groups on", and a divisibility violation fails loudly
- Blocking vacuity — the gates on rounding passing through and on whether `op_types` takes effect
  **actually fail**
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pytest
import torch
from torch import nn

from embeddinggemma import export as eg
from embeddinggemma import measure_quant as mq
from karume.quantize import fake_quant_int4, iter_quant_targets


class TinyModel(nn.Module):
    """One linear + one embedding (both in axes are divisible by group 32)."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(32, 4, bias=False)
        self.embed = nn.Embedding(8, 32)


def _stats(**overrides: int) -> mq.TargetStats:
    """The default is the smallest form "32 elements = 1 group, 1 tensor, 1 channel"."""
    base = {"modules": 1, "elements": 32, "groups": 1, "channels": 1}
    return mq.TargetStats(**{**base, **overrides})


def _method(name: str) -> mq.Method:
    return next(method for method in mq.METHODS if method.name == name)


class TestSizeProjection:
    """Where the bpw in the table comes from (verbatim `int4-sym-g` of `docs/container-v1.md` §6.3
    and OCP MX)."""

    @pytest.mark.parametrize("name", ["rtn-i4-g32", "fp4", "nf4"])
    def test_group_absmax_methods_land_on_five_bits(self, name):
        """4-bit payload + an F32 scale per group of 32 = 4 + 32/32 = 5.0 bpw."""
        projection = mq.size_projection(_method(name), _stats(elements=3200, groups=100))

        assert projection["bitsPerWeight"] == pytest.approx(5.0)

    def test_mxfp4_lands_on_four_and_a_quarter_bits(self):
        """The shared scale is E8M0 (8 bits), so 4 + 8/32 = 4.25 bpw."""
        projection = mq.size_projection(_method("mxfp4"), _stats(elements=3200, groups=100))

        assert projection["bitsPerWeight"] == pytest.approx(4.25)

    def test_the_per_tensor_codebook_costs_one_table_per_tensor(self):
        """No scale, but a table per layer (16 centroids × F32) is added instead."""
        stats = _stats(modules=10, elements=3200, groups=100, channels=40)

        projection = mq.size_projection(_method("kmeans:per_tensor"), stats)

        assert projection["bitsPerWeight"] == pytest.approx(4.0 + 32 * 16 * 10 / 3200)

    def test_the_per_channel_codebook_costs_one_table_per_channel(self):
        """The table cost is proportional to **the number of channels** (the only difference from
        per_tensor)."""
        stats = _stats(modules=10, elements=3200, groups=100, channels=40)

        projection = mq.size_projection(_method("kmeans:per_channel"), stats)

        assert projection["bitsPerWeight"] == pytest.approx(4.0 + 32 * 16 * 40 / 3200)

    def test_the_shared_codebook_costs_one_table_for_the_whole_model(self):
        """One table, but a group scale is needed (= the same 5.0 bpw as i4 + one table)."""
        stats = _stats(modules=10, elements=3200, groups=100, channels=40)

        projection = mq.size_projection(_method("kmeans:shared"), stats)

        assert projection["bitsPerWeight"] == pytest.approx(5.0 + 32 * 16 / 3200)

    def test_the_ratio_is_the_bpw_against_f32(self):
        """The ratio is "what % of f32 storage" — derived from the same numbers as the two MiB
        columns."""
        projection = mq.size_projection(_method("nf4"), _stats(elements=3200, groups=100))

        assert projection["ratio"] == pytest.approx(5.0 / 32.0)
        assert projection["projectedMiB"] == pytest.approx(projection["f32MiB"] * 5.0 / 32.0)


class TestTargetStats:
    def test_it_counts_groups_on_the_axis_the_rounding_splits(self):
        """The group count is taken from the flattened form of `channel_rows` (= the axis the
        rounding splits groups on)."""
        model = TinyModel()

        linear = mq.target_stats(model, mq.TARGET_LINEAR)
        widened = mq.target_stats(model, mq.TARGET_WITH_EMBEDDING)

        # linear `[4,32]` … 4 rows × (32/32) group / embedding `[8,32]` … 8 rows × 1 group.
        assert (linear.modules, linear.elements, linear.groups, linear.channels) == (1, 128, 4, 4)
        assert (widened.modules, widened.elements) == (2, 128 + 256)
        assert (widened.groups, widened.channels) == (4 + 8, 4 + 8)

    def test_an_indivisible_quantization_axis_fails_loudly(self):
        """MUST: a partial group is a shape that cannot be stored (ADR 0069 decision 2) — no bpw for
        it."""

        class Odd(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.fc = nn.Linear(30, 2, bias=False)

        with pytest.raises(AssertionError, match="is not divisible"):
            mq.target_stats(Odd(), mq.TARGET_LINEAR)

    def test_an_empty_target_fails_loudly(self):
        with pytest.raises(AssertionError, match="no quantizable weight"):
            mq.target_stats(nn.Module(), mq.TARGET_LINEAR)


class TestTargets:
    def test_the_narrow_target_leaves_the_vocabulary_table_alone(self):
        """The 2 target forms differ by the one `embed_tokens` (exactly the expected value of gate
        ②)."""
        model = TinyModel()

        narrow = [fqn for fqn, _w, _a in iter_quant_targets(model, mq.TARGET_LINEAR.op_types)]
        wide = [fqn for fqn, _w, _a in iter_quant_targets(model, mq.WIDEST_OP_TYPES)]

        assert narrow == ["fc.weight"]
        assert wide == ["fc.weight", "embed.weight"]


class TestMethodsAreNotStacked:
    """MUST: each config goes back to pristine (without it the comparison between methods does not
    hold)."""

    def _pristine(self, model: nn.Module) -> tuple[dict, dict]:
        weights = {fqn: w for fqn, w, _a in iter_quant_targets(model, mq.WIDEST_OP_TYPES)}
        return weights, {fqn: w.detach().clone() for fqn, w in weights.items()}

    def test_restore_brings_back_the_original_weights_bit_for_bit(self):
        torch.manual_seed(0)
        model = TinyModel()
        weights, pristine = self._pristine(model)

        _method("nf4").apply(model, mq.TARGET_WITH_EMBEDDING.op_types)
        mq.restore(weights, pristine)

        assert all(torch.equal(weights[fqn], pristine[fqn]) for fqn in pristine)

    def test_a_restored_run_matches_a_fresh_one(self):
        """i4 applied after restoring from another method is bit-identical to i4 applied to the
        plain weights."""
        torch.manual_seed(0)
        model = TinyModel()
        weights, pristine = self._pristine(model)

        _method("fp4").apply(model, mq.TARGET_LINEAR.op_types)
        mq.restore(weights, pristine)
        fake_quant_int4(model, mq.GROUP_SIZE, op_types=mq.TARGET_LINEAR.op_types)
        after_restore = weights["fc.weight"].detach().clone()

        mq.restore(weights, pristine)
        fake_quant_int4(model, mq.GROUP_SIZE, op_types=mq.TARGET_LINEAR.op_types)

        assert torch.equal(after_restore, weights["fc.weight"])

    def test_stacking_the_two_methods_would_have_changed_the_answer(self):
        """Guard against vacuity — stacking without restoring really gives other values (the
        condition under which the test above means something)."""
        torch.manual_seed(0)
        model = TinyModel()
        weights, pristine = self._pristine(model)

        _method("fp4").apply(model, mq.TARGET_LINEAR.op_types)
        fake_quant_int4(model, mq.GROUP_SIZE, op_types=mq.TARGET_LINEAR.op_types)
        stacked = weights["fc.weight"].detach().clone()

        mq.restore(weights, pristine)
        fake_quant_int4(model, mq.GROUP_SIZE, op_types=mq.TARGET_LINEAR.op_types)

        assert not torch.equal(stacked, weights["fc.weight"])


class TestCosine:
    def test_an_identical_vector_scores_one(self):
        vector = torch.tensor([0.6, 0.8, 0.0])

        assert mq.cosine(vector, vector) == pytest.approx(1.0)

    def test_an_opposite_vector_scores_minus_one(self):
        vector = torch.tensor([0.6, 0.8])

        assert mq.cosine(-vector, vector) == pytest.approx(-1.0)

    def test_it_is_scale_invariant(self):
        """MUST: do not let norm breakage leak into the cosine (the norm is a separate column)."""
        vector = torch.tensor([1.0, 2.0, 3.0])

        assert mq.cosine(vector * 7.0, vector) == pytest.approx(1.0)

    def test_a_zero_vector_fails_loudly(self):
        """All zeros is not about rounding but a sign of a broken path — do not return NaN and keep
        measuring."""
        with pytest.raises(AssertionError, match="all zeros"):
            mq.cosine(torch.zeros(3), torch.ones(3))


def _vectors(**overrides: torch.Tensor) -> dict[str, torch.Tensor]:
    """Synthetic embeddings for the 5 cases (by default mutually orthogonal unit vectors)."""
    base = {name: torch.eye(len(mq.CASE_NAMES))[index] for index, name in enumerate(mq.CASE_NAMES)}
    return {**base, **overrides}


class TestPairCosines:
    def test_it_covers_the_upper_triangle_only(self):
        pairs = mq.pair_cosines(_vectors())

        assert len(pairs) == len(mq.CASE_NAMES) * (len(mq.CASE_NAMES) - 1) // 2
        assert mq.pair_label(eg.NEAR_PAIR) in pairs
        assert mq.pair_label(eg.FAR_PAIR) in pairs

    def test_orthogonal_cases_score_zero(self):
        pairs = mq.pair_cosines(_vectors())

        assert all(value == pytest.approx(0.0) for value in pairs.values())


class TestMeasure:
    def _near_beats_far(self) -> dict[str, torch.Tensor]:
        """Synthetic embeddings in which only the near pair (query-en × document-en) is aligned."""
        near_partner = (
            torch.eye(len(mq.CASE_NAMES))[0] * 0.8 + torch.eye(len(mq.CASE_NAMES))[1] * 0.6
        )
        return _vectors(**{eg.NEAR_PAIR[1]: near_partner})

    def test_the_base_against_itself_is_a_perfect_row(self):
        vectors = self._near_beats_far()

        result = mq.measure(vectors, vectors)

        assert result["caseCosineMin"] == pytest.approx(1.0)
        assert result["pairDriftMaxAbs"] == pytest.approx(0.0)

    def test_it_keeps_the_semantic_order_when_the_near_pair_leads(self):
        vectors = self._near_beats_far()

        result = mq.measure(vectors, vectors)["order"]

        assert result["holds"]
        assert result["nearCosine"] == pytest.approx(0.8)
        assert result["farCosine"] == pytest.approx(0.0)
        assert result["margin"] == pytest.approx(0.8)

    def test_it_reports_a_collapsed_order(self):
        """MUST: checked independently of the cosine against the baseline — when every vector is
        close the cos stays high while the order collapses."""
        base = self._near_beats_far()
        collapsed = _vectors(**{eg.FAR_PAIR[1]: base[eg.NEAR_PAIR[0]]})

        result = mq.measure(collapsed, base)["order"]

        assert not result["holds"]
        assert result["farCosine"] == pytest.approx(1.0)

    def test_the_pair_drift_names_the_worst_pair(self):
        base = self._near_beats_far()
        moved = _vectors(**{eg.FAR_PAIR[1]: base[eg.NEAR_PAIR[0]]})

        result = mq.measure(moved, base)

        assert result["pairDriftWorst"] == mq.pair_label(eg.FAR_PAIR)
        assert result["pairDriftMaxAbs"] == pytest.approx(1.0)

    def test_the_case_cosine_ignores_a_pure_norm_change(self):
        """When only the norm moves the cosine stays 1 (a quantity `l2Norms` covers)."""
        base = self._near_beats_far()
        scaled = {name: vector * 0.5 for name, vector in base.items()}

        result = mq.measure(scaled, base)

        assert result["caseCosineMin"] == pytest.approx(1.0)
        assert all(value == pytest.approx(0.5, abs=1e-6) for value in result["l2Norms"].values())


def _entry(*, modules: int, moved: bool = True, cosine: float = 0.99) -> dict[str, object]:
    return {
        "moved": moved,
        "size": {"modules": modules},
        "caseCosine": dict.fromkeys(mq.CASE_NAMES, cosine),
    }


def _configs(**overrides: dict[str, object]) -> dict[str, dict[str, object]]:
    method = _method("nf4")
    configs = {
        mq.BASE_CONFIG: {"caseCosine": dict.fromkeys(mq.CASE_NAMES, 1.0)},
        mq.config_name(method, mq.TARGET_LINEAR): _entry(modules=170),
        mq.config_name(method, mq.TARGET_WITH_EMBEDDING): _entry(modules=171, cosine=0.98),
    }
    return {**configs, **overrides}


def _failures(configs: dict[str, dict[str, object]]) -> list[str]:
    return mq.run_gates(configs, (_method("nf4"),))["failures"]


class TestGates:
    """MUST: a pass-through always lies on the "good quality" side — never let it go silent."""

    def test_a_well_formed_pair_of_configs_is_green(self):
        assert _failures(_configs()) == []

    def test_a_config_bit_identical_to_the_base_is_red(self):
        configs = _configs()
        configs[mq.config_name(_method("nf4"), mq.TARGET_LINEAR)] = _entry(modules=170, moved=False)

        assert any("passed through" in message for message in _failures(configs))

    def test_a_widened_target_that_did_not_add_the_embedding_is_red(self):
        """If the `op_types` switch has no effect, a table that claims to have measured the
        vocabulary table comes out."""
        configs = _configs()
        configs[mq.config_name(_method("nf4"), mq.TARGET_WITH_EMBEDDING)] = _entry(
            modules=170, cosine=0.98
        )

        assert any("op_types" in message for message in _failures(configs))

    def test_a_widened_target_with_identical_embeddings_is_red(self):
        """If only the count grows and the values stay the same, the rounding of the vocabulary
        table does not reach the output."""
        configs = _configs()
        configs[mq.config_name(_method("nf4"), mq.TARGET_WITH_EMBEDDING)] = _entry(modules=171)

        assert any("does not reach the output" in message for message in _failures(configs))


class TestMethodTable:
    def test_every_method_is_measured_on_both_targets(self):
        assert len(mq.METHODS) == 7
        assert [target.name for target in mq.TARGETS] == ["linear", "linear+embedding"]

    def test_the_group_size_is_pinned_for_every_method(self):
        """The axis of the method comparison is only "the grid layout" — the group length does not
        move (the maintainer's ruling)."""
        assert mq.GROUP_SIZE == 32

    def test_selecting_a_subset_keeps_the_declaration_order(self):
        chosen = mq.select_methods(["nf4", "rtn-i4-g32"])

        assert [method.name for method in chosen] == ["rtn-i4-g32", "nf4"]

    def test_an_empty_selection_runs_everything(self):
        assert mq.select_methods([]) == mq.METHODS


class TestTables:
    def _report(self) -> dict[str, object]:
        entry = {
            "method": "nf4",
            "target": "linear",
            "caseCosineMin": 0.9987,
            "caseCosineMean": 0.9991,
            "order": {"nearCosine": 0.71, "farCosine": 0.32, "holds": True},
            "pairDriftMaxAbs": 1.5e-3,
            "size": {
                "modules": 170,
                "elements": 3200,
                "bitsPerWeight": 5.0,
                "projectedMiB": 1.9,
                "f32MiB": 12.2,
                "ratio": 0.15625,
                "formula": "4·N + 32·G",
            },
        }
        return {"configs": {"nf4/linear": entry}}

    def test_the_quality_table_carries_the_order_verdict(self):
        table = mq.quality_table(self._report())

        assert "| nf4 | linear | 170 | 0.998700 | 0.999100 | 0.7100 | 0.3200 | held |" in table

    def test_a_collapsed_order_is_marked_up(self):
        report = self._report()
        report["configs"]["nf4/linear"]["order"]["holds"] = False

        assert "**collapsed**" in mq.quality_table(report)

    def test_the_size_table_shows_the_formula(self):
        table = mq.size_table(self._report())

        assert "| nf4 | linear | 3,200 | 5.000 | 1.9 | 12.2 | 15.6% | 4·N + 32·G |" in table


class TestMibConversion:
    def test_one_mebibyte_is_the_binary_one(self):
        """2^20, as the `MiB` notation says (mixing in 10^6 shifts the table's numbers by 5%)."""
        assert mq.MIB == 2**20
        assert math.log2(mq.MIB) == 20


# ---- Calibrated rounding (work item J-2) ---------------------------------------------


class TinyAttention(nn.Module):
    """2 linear quantization targets (the in axis is divisible by group 32)."""

    def __init__(self, features: int) -> None:
        super().__init__()
        self.q_proj = nn.Linear(features, features, bias=False)
        self.o_proj = nn.Linear(features, features, bias=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.o_proj(self.q_proj(hidden))


class TinyDecoderLayer(nn.Module):
    """A decoder layer stand-in. `bias` stands in for "kwargs that differ per layer_type" (observed
    through addition)."""

    def __init__(self, features: int) -> None:
        super().__init__()
        self.self_attn = TinyAttention(features)

    def forward(self, hidden: torch.Tensor, bias: float = 0.0) -> torch.Tensor:
        return self.self_attn(hidden + bias)


class TinyConfig:
    """A stand-in for `Gemma3TextConfig` (holds only `layer_types`)."""

    def __init__(self, layer_types: tuple[str, ...]) -> None:
        self.layer_types = layer_types


@dataclass(frozen=True)
class TinyTextOutput:
    """A stand-in for `BaseModelOutputWithPast` (holds only `last_hidden_state`)."""

    last_hidden_state: torch.Tensor


class TinyText(nn.Module):
    """A `Gemma3TextModel` stand-in (`.config.layer_types` / `.layers` / `.embed_tokens` / `.norm`).

    `forward` is the same "embedding → the layers in turn → final norm" as upstream, and the return
    value has the same field (`last_hidden_state`) — this is what {@link mq.assert_stage_split}
    compares against.
    """

    def __init__(self, vocab: int, features: int, layer_types: tuple[str, ...]) -> None:
        super().__init__()
        self.config = TinyConfig(layer_types)
        self.embed_tokens = nn.Embedding(vocab, features)
        self.layers = nn.ModuleList(TinyDecoderLayer(features) for _ in layer_types)
        self.norm = nn.LayerNorm(features)

    def forward(self, input_ids: torch.Tensor, use_cache: bool = False) -> TinyTextOutput:
        hidden = self.embed_tokens(input_ids)
        for kind, layer in zip(self.config.layer_types, self.layers, strict=True):
            hidden = layer(hidden, bias=TINY_BIAS[kind])
        return TinyTextOutput(self.norm(hidden))


#: kwargs per layer_type (a stand-in for Gemma3's two kinds, sliding / full).
TINY_BIAS = {"sliding_attention": 0.5, "full_attention": -0.25}

#: sliding first and one full in between (the form in which `capture_stage_batches` captures 2
#: kinds).
TINY_LAYER_TYPES = ("sliding_attention", "sliding_attention", "full_attention")


class TinyWrapper(nn.Module):
    """An `EmbeddingWrapper` stand-in — holds one Dense **outside** the decoder (outside the
    calibration targets)."""

    def __init__(self, vocab: int = 8, features: int = 32) -> None:
        super().__init__()
        self.model = TinyText(vocab, features, TINY_LAYER_TYPES)
        self.dense2 = nn.Linear(features, features, bias=False)

    def forward(self, input_ids: torch.Tensor, pool_mask: torch.Tensor) -> torch.Tensor:
        hidden = self.model(input_ids=input_ids, use_cache=False).last_hidden_state
        pooled = torch.sum(hidden * pool_mask.unsqueeze(-1), dim=1)
        return self.dense2(pooled)


def tiny_inputs(count: int = 2) -> tuple[torch.Tensor, ...]:
    """id sequences of different lengths (a stand-in for calibration inputs — no tokenizer)."""
    return tuple(
        torch.arange(2 + index, dtype=torch.int64).unsqueeze(0) % 8 for index in range(count)
    )


def tiny_rig(wrapper: TinyWrapper, inputs: tuple[torch.Tensor, ...]) -> mq.CalibRig:
    stages = mq.decoder_stages(wrapper)
    scan, stats = mq.calib_targets(stages)
    return mq.CalibRig(
        stages=stages,
        scan=scan,
        stats=stats,
        batches=mq.capture_stage_batches(wrapper, inputs),
    )


class TestCalibConfigs:
    def test_every_calib_config_is_selectable(self):
        assert tuple(config.name for config in mq.CALIB_CONFIGS) == mq.CALIB_NAMES

    def test_selecting_one_config_keeps_the_declaration_order(self):
        chosen = mq.select_calib(["awq-rtn", "gptq-rtn"])

        assert [config.name for config in chosen] == ["gptq-rtn", "awq-rtn"]

    def test_an_empty_selection_runs_every_calib_config(self):
        assert mq.select_calib([]) == mq.CALIB_CONFIGS

    def test_selecting_only_a_method_leaves_no_calib_config(self):
        """`--only nf4` is the method only — it does not drag in the calibrated configs."""
        assert mq.select_calib(["nf4"]) == ()

    def test_the_group_size_is_pinned_for_every_calib_config(self):
        assert all(config.grid.group_size == mq.GROUP_SIZE for config in mq.CALIB_CONFIGS)

    def test_no_calib_config_fits_its_table_on_a_subsample_by_default(self):
        assert all(config.grid.fit_stride == 1 for config in mq.CALIB_CONFIGS)

    def test_the_label_names_both_the_method_and_the_grid(self):
        labels = {config.name: config.label for config in mq.CALIB_CONFIGS}

        assert labels["gptq-nf4"] == "gptq/nf4"
        assert labels["awq-gptq-rtn"] == "awq+gptq/rtn"

    def test_the_size_projection_takes_a_calib_config(self):
        """Calibration does not change the grid — the bpw of the RTN grid is the same 5.0 as on the
        method side."""
        config = next(c for c in mq.CALIB_CONFIGS if c.name == "gptq-rtn")

        projection = mq.size_projection(config, _stats(elements=3200, groups=100))

        assert projection["bitsPerWeight"] == pytest.approx(5.0)

    def test_the_calibrated_codebook_costs_one_table_per_layer(self):
        """Core's `kmeans_shared` has a table **within a layer** (a different formula from
        kmeans:shared with one table overall)."""
        config = next(c for c in mq.CALIB_CONFIGS if c.name == "gptq-kmeans")
        stats = _stats(modules=4, elements=3200, groups=100)

        projection = mq.size_projection(config, stats)

        tables = mq.CODEBOOK_ENTRY_BITS * mq.DEFAULT_CODEBOOK_LEVELS * stats.modules
        assert projection["bitsPerWeight"] == pytest.approx(5.0 + tables / stats.elements)


class TestCalibStages:
    def test_the_stage_fqns_are_model_wide(self):
        """Align the ledger keys with the same FQN space as `Int4Report` (core's `StageSpec`
        contract)."""
        scan, stats = mq.calib_targets(mq.decoder_stages(TinyWrapper()))

        assert sorted(scan)[:2] == [
            "model.layers.0.self_attn.o_proj.weight",
            "model.layers.0.self_attn.q_proj.weight",
        ]
        assert stats.modules == 2 * len(TINY_LAYER_TYPES)

    def test_the_dense_outside_the_decoder_is_not_in_the_scan(self):
        """The 2 Dense stages are outside the decoder — not in the calibration target set (where the
        table's target name comes from)."""
        scan, _stats = mq.calib_targets(mq.decoder_stages(TinyWrapper()))

        assert not any("dense" in fqn for fqn in scan)

    def test_each_stage_keeps_its_layer_type(self):
        stages = mq.decoder_stages(TinyWrapper())

        assert [stage.layer_type for _prefix, stage in stages] == list(TINY_LAYER_TYPES)

    def test_a_layer_type_table_that_does_not_match_the_layers_fails_loudly(self):
        wrapper = TinyWrapper()
        wrapper.model.config.layer_types = TINY_LAYER_TYPES[:-1]

        with pytest.raises(AssertionError, match="disagree with"):
            mq.decoder_stages(wrapper)

    def test_a_stage_applies_the_keyword_arguments_of_its_own_layer_type(self):
        """MUST: do not mix up the mask / RoPE table that differ per layer (why `LayerStage`
        exists)."""
        torch.manual_seed(0)
        wrapper = TinyWrapper()
        stages = mq.decoder_stages(wrapper)
        hidden = torch.randn(1, 3, 32)
        layer_kwargs = {kind: {"bias": bias} for kind, bias in TINY_BIAS.items()}

        with torch.no_grad():
            full = stages[2][1](hidden, layer_kwargs=layer_kwargs)
            direct = wrapper.model.layers[2](hidden, bias=TINY_BIAS["full_attention"])

        assert torch.equal(full, direct)


class TestCatcher:
    def test_it_captures_one_batch_per_calibration_text(self):
        wrapper = TinyWrapper()

        batches = mq.capture_stage_batches(wrapper, tiny_inputs(3))

        assert len(batches) == 3
        assert [int(args[0].shape[1]) for args, _kwargs in batches] == [2, 3, 4]

    def test_it_captures_the_keyword_arguments_of_every_layer_type(self):
        """Capturing only one layer_type leaves the layers of the other kind unable to rerun."""
        wrapper = TinyWrapper()

        _args, kwargs = mq.capture_stage_batches(wrapper, tiny_inputs(1))[0]

        assert kwargs[mq.LAYER_KWARGS] == {
            "sliding_attention": {"bias": 0.5},
            "full_attention": {"bias": -0.25},
        }

    def test_the_hooks_are_removed_even_though_the_forward_was_aborted(self):
        wrapper = TinyWrapper()

        mq.capture_stage_batches(wrapper, tiny_inputs(1))

        assert not any(layer._forward_pre_hooks for layer in wrapper.model.layers)


class TestStageSplit:
    """The split agreement check ({@link mq.assert_stage_split}) — the premise that calibration sees
    "the same activations as the real model"."""

    def test_the_sequential_stages_reproduce_the_model_bit_for_bit(self):
        torch.manual_seed(6)
        wrapper = TinyWrapper()
        inputs = tiny_inputs(1)
        batches = mq.capture_stage_batches(wrapper, inputs)

        mq.assert_stage_split(wrapper, inputs[0], batches[0], mq.decoder_stages(wrapper))

    def test_a_dropped_stage_is_caught(self):
        """With one stage dropped the numbers still come out ordinary — only the agreement check
        fails it."""
        torch.manual_seed(7)
        wrapper = TinyWrapper()
        inputs = tiny_inputs(1)
        batches = mq.capture_stage_batches(wrapper, inputs)
        stages = mq.decoder_stages(wrapper)

        with pytest.raises(AssertionError, match="not bit-identical"):
            mq.assert_stage_split(wrapper, inputs[0], batches[0], stages[:-1])

    def test_a_stage_carrying_the_wrong_layer_type_is_caught(self):
        """Specific to embeddinggemma — reject a sequence that mixed up the kwargs differing per
        layer."""
        torch.manual_seed(8)
        wrapper = TinyWrapper()
        inputs = tiny_inputs(1)
        batches = mq.capture_stage_batches(wrapper, inputs)
        swapped = {
            "sliding_attention": "full_attention",
            "full_attention": "sliding_attention",
        }
        stages = tuple(
            (prefix, mq.LayerStage(index, wrapper.model.layers[index], swapped[stage.layer_type]))
            for index, (prefix, stage) in enumerate(mq.decoder_stages(wrapper))
        )

        with pytest.raises(AssertionError, match="not bit-identical"):
            mq.assert_stage_split(wrapper, inputs[0], batches[0], stages)


class TestCalibrationRun:
    def test_it_rounds_every_scanned_linear(self):
        torch.manual_seed(1)
        wrapper = TinyWrapper()
        rig = tiny_rig(wrapper, tiny_inputs(2))
        before = {fqn: weight.detach().clone() for fqn, weight in rig.scan.items()}

        report = mq.apply_calib(mq.CALIB_CONFIGS[0], rig)
        mq.assert_calib_covers_scan(report, rig.scan, "gptq-rtn")

        assert report.modules == len(rig.scan)
        assert all(not torch.equal(rig.scan[fqn], value) for fqn, value in before.items())

    def test_the_dense_outside_the_decoder_is_left_alone(self):
        """Calibration is inside the decoder only — the Dense outside does not move by a single
        bit."""
        torch.manual_seed(2)
        wrapper = TinyWrapper()
        rig = tiny_rig(wrapper, tiny_inputs(2))
        dense = wrapper.dense2.weight.detach().clone()

        mq.apply_calib(mq.CALIB_CONFIGS[0], rig)

        assert torch.equal(wrapper.dense2.weight, dense)

    def test_every_calib_config_runs_end_to_end(self):
        for config in mq.CALIB_CONFIGS:
            torch.manual_seed(3)
            wrapper = TinyWrapper()
            rig = tiny_rig(wrapper, tiny_inputs(2))

            report = mq.apply_calib(config, rig)
            mq.assert_calib_covers_scan(report, rig.scan, config.name)

            assert report.method == config.method
            assert report.grid == config.grid.kind

    def test_a_subsampled_table_is_reported(self):
        """MUST: when `fit_stride` is used, the output says so (`describe` picks it up)."""
        torch.manual_seed(4)
        wrapper = TinyWrapper()
        rig = tiny_rig(wrapper, tiny_inputs(2))
        kmeans = next(config for config in mq.CALIB_CONFIGS if config.name == "gptq-kmeans")

        report = mq.apply_calib(kmeans, rig, shared_stride=4)

        assert "fit_stride 4" in report.describe()

    def test_a_scan_the_calibration_missed_fails_loudly(self):
        torch.manual_seed(5)
        wrapper = TinyWrapper()
        rig = tiny_rig(wrapper, tiny_inputs(2))
        report = mq.apply_calib(mq.CALIB_CONFIGS[0], rig)
        widened = {**rig.scan, "model.layers.9.self_attn.q_proj.weight": torch.zeros(1)}

        with pytest.raises(AssertionError, match="does not match"):
            mq.assert_calib_covers_scan(report, widened, "gptq-rtn")
