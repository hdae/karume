"""Unit tests around w4 in `sbv2.measure_quant` (**only the logic that needs no assets**).

The measurements themselves, with real weights and dumps, belong to the script's check gates
(`measure_quant.run_gates`). What is isolated here are the 4 parts that hold without assets:

    ① the census of the target set (excluding weights not divisible by the group length, folded
       into `include`)
    ② the size projection formula (effective bpw per method)
    ③ the dispatch between methods (the 4 round to values different from each other, and not a
       single element of an excluded weight is touched)
    ④ matching against the distributed form (only the linear weight slots are picked by tensor name)
    ⑤ the CLI override of the group length g (`--w4-group-size` reaches **both** the eligibility
       check and the rounding)

The rounding implementations themselves (the numbers of i4 / NF4 / MXFP4 / k-means) belong to the
core-side tests — this file measures only "how the script dispatches to core", and not a single
formula is copied.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest
import torch
from torch import nn

from karume.container import Provenance
from karume.emit import stored_model
from karume.ir import IrGraph, IrInitializer, IrInput, IrNode, IrStorage, IrValue
from karume.publish import publish_container
from karume.quantize import channel_scale
from sbv2 import measure_quant as measure

#: The initialization seed of the synthetic model (the tests that look at differences between
#: methods assume non-degenerate values).
SEED = 20260819

#: The default group length (g when `--w4-group-size` is not given). Tests that vary g read as
#: differences from this.
DEFAULT_G = measure.DEFAULT_GROUP_SIZE


class TinyNet(nn.Module):
    """A synthetic model holding 4 of the 5 op kinds w4 targets, **both eligible and excluded**.

    The pairs are made by whether the quantization axis (= each op's in axis) is divisible by the
    group length 32: `linear` 64 / `odd_linear` 48 / `conv` receptive field 4·8 = 32 /
    `depthwise` receptive field 1·3 = 3.
    """

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(64, 8)
        self.odd_linear = nn.Linear(48, 4)
        self.conv = nn.Conv1d(4, 6, 8)
        self.depthwise = nn.Conv1d(6, 6, 3, groups=6)
        self.emb = nn.Embedding(10, 32)


@pytest.fixture
def tiny() -> TinyNet:
    torch.manual_seed(SEED)
    return TinyNet()


class TestW4Census:
    def test_it_excludes_weights_whose_quantization_axis_is_not_divisible(self, tiny) -> None:
        census = measure.census_w4_targets(tiny, measure.W4_ROLES["all"], DEFAULT_G)
        assert sorted(census.excluded) == ["depthwise", "odd_linear"]
        assert sorted(census.eligible) == ["conv", "emb", "linear"]

    def test_it_counts_the_excluded_elements(self, tiny) -> None:
        census = measure.census_w4_targets(tiny, measure.W4_ROLES["all"], DEFAULT_G)
        assert census.excluded_elements == 4 * 48 + 6 * 1 * 3

    def test_it_counts_the_eligible_targets_per_role(self, tiny) -> None:
        census = measure.census_w4_targets(tiny, measure.W4_ROLES["all"], DEFAULT_G)
        by_role = {role: (c.modules, c.channels, c.elements) for role, c in census.by_role.items()}
        assert by_role == {
            "Conv1d": (1, 6, 6 * 4 * 8),
            "Embedding": (1, 10, 10 * 32),
            "Linear": (1, 8, 8 * 64),
        }
        assert census.counts.elements == 192 + 320 + 512
        assert census.counts.groups == (192 + 320 + 512) // DEFAULT_G

    def test_it_narrows_the_targets_to_linear_for_the_shippable_role(self, tiny) -> None:
        census = measure.census_w4_targets(tiny, measure.W4_ROLES["linear"], DEFAULT_G)
        assert census.eligible == ("linear",)
        assert census.excluded == ("odd_linear",)

    def test_it_folds_the_exclusion_into_the_core_include_predicate(self, tiny) -> None:
        include = measure.census_w4_targets(tiny, measure.W4_ROLES["all"], DEFAULT_G).include()
        assert [
            name for name in ("linear", "odd_linear", "conv", "depthwise") if include(name)
        ] == [
            "linear",
            "conv",
        ]


class TestW4SizeProjection:
    #: 65536 elements = 2048 groups (g32) — a size chosen so the bpw divides evenly.
    COUNTS = measure.TargetCounts(modules=2, channels=8, elements=65536, group_size=DEFAULT_G)
    METHODS = measure.w4_methods(DEFAULT_G)

    @pytest.mark.parametrize(
        ("kind", "expected_bpw"),
        [(measure.RTN_KIND, 5.0), ("nf4", 5.0), ("mxfp4", 4.25)],
    )
    def test_it_projects_the_effective_bits_per_weight(self, kind, expected_bpw) -> None:
        projection = measure.SizeProjection(
            counts=self.COUNTS,
            bits=self.METHODS[kind].projected_bits(self.COUNTS),
            formula=self.METHODS[kind].formula,
        )
        assert projection.bits_per_weight == expected_bpw

    def test_it_charges_the_shared_codebook_to_kmeans(self) -> None:
        """The cost of the table goes in the same table as the quality — only k-means is heavier, by
        one table overall."""
        rtn = self.METHODS[measure.RTN_KIND].projected_bits(self.COUNTS)
        kmeans = self.METHODS["kmeans:shared"].projected_bits(self.COUNTS)
        assert kmeans - rtn == measure.CODEBOOK_BITS

    def test_it_reports_mib_against_the_f32_baseline(self) -> None:
        projection = measure.SizeProjection(
            counts=self.COUNTS,
            bits=self.METHODS[measure.RTN_KIND].projected_bits(self.COUNTS),
            formula="",
        )
        assert projection.f32_mib == 65536 * 4 / 1024**2
        assert projection.projected_mib == projection.f32_mib * 5.0 / 32


class TestW4ProjectionRows:
    def test_it_marks_which_rows_have_a_quality_measurement(self) -> None:
        """Which rows carry a measurement is taken from CONFIGS (writing it in a separate table
        would let one side go stale)."""
        methods = measure.w4_methods(DEFAULT_G)
        counts = dict.fromkeys(
            measure.PROJECTION_TARGETS,
            measure.TargetCounts(modules=1, channels=1, elements=32, group_size=DEFAULT_G),
        )
        measured = {
            (row["target"], row["method"]): row["measured_by"]
            for row in measure.build_projections(counts, methods)
        }
        assert measured[("net_g:all", "rtn-i4-g32")] == "w4-rtn"
        assert measured[("net_g:all", "kmeans:shared")] == "w4-kmeans"
        assert measured[("bert:linear", "nf4")] == "bert-w4-nf4"
        # net_g limited to linear is **a size-only row** (quality is not measured — too few
        # targets).
        assert all(measured[("net_g:linear", method.name)] is None for method in methods.values())
        # For BERT only the 2 methods with a distributable form are measured.
        assert measured[("bert:linear", "mxfp4")] is None


class TestW4Methods:
    METHODS = measure.w4_methods(DEFAULT_G)

    def test_it_rounds_only_the_eligible_weights(self, tiny) -> None:
        census = measure.census_w4_targets(tiny, measure.W4_ROLES["all"], DEFAULT_G)
        untouched = {name: getattr(tiny, name).weight.clone() for name in census.excluded}
        report = self.METHODS[measure.RTN_KIND].apply(
            tiny, measure.W4_ROLES["all"], census.include()
        )
        assert (report.modules, report.elements) == (
            census.counts.modules,
            census.counts.elements,
        )
        for name, before in untouched.items():
            assert torch.equal(getattr(tiny, name).weight, before)

    def test_it_moves_the_eligible_weights(self, tiny) -> None:
        census = measure.census_w4_targets(tiny, measure.W4_ROLES["all"], DEFAULT_G)
        before = {name: getattr(tiny, name).weight.clone() for name in census.eligible}
        self.METHODS[measure.RTN_KIND].apply(tiny, measure.W4_ROLES["all"], census.include())
        for name, weight in before.items():
            assert not torch.equal(getattr(tiny, name).weight, weight), name

    def test_it_keeps_the_four_methods_distinct(self) -> None:
        """Detects a dispatch slip between methods (applying the same rounding twice) — the same
        form as the script's gate."""
        rounded: dict[str, torch.Tensor] = {}
        for name in self.METHODS:
            torch.manual_seed(SEED)
            model = TinyNet()
            census = measure.census_w4_targets(model, measure.W4_ROLES["all"], DEFAULT_G)
            self.METHODS[name].apply(model, measure.W4_ROLES["all"], census.include())
            rounded[name] = model.linear.weight.clone()
        names = list(rounded)
        for index, left in enumerate(names):
            for right in names[index + 1 :]:
                assert not torch.equal(rounded[left], rounded[right]), f"{left} and {right}"


class TestW4GroupSizeOption:
    """Acceptance and default of `--w4-group-size` (the g axis of iteration J-3)."""

    def test_the_default_is_the_core_storage_default(self) -> None:
        """The default is 32 — a run without the option uses the same grid as the past research
        records."""
        assert DEFAULT_G == 32
        assert measure.build_parser().parse_args([]).w4_group_size == 32

    @pytest.mark.parametrize("value", [16, 32, 64, 128])
    def test_it_accepts_a_power_of_two_from_sixteen(self, value) -> None:
        parsed = measure.build_parser().parse_args(["--w4-group-size", str(value)])

        assert parsed.w4_group_size == value

    @pytest.mark.parametrize("raw", ["24", "8", "0", "-32", "g32"])
    def test_it_rejects_a_group_core_cannot_store(self, raw) -> None:
        """MUST: reject values outside the accepted set in argparse (if emit rejected it after the
        measurement, a whole run would be lost)."""
        with pytest.raises(SystemExit):
            measure.build_parser().parse_args(["--w4-group-size", raw])

    def test_the_rejection_names_the_storage_rule(self) -> None:
        with pytest.raises(argparse.ArgumentTypeError, match="2 冪かつ"):
            measure.parse_w4_group_size("24")


class TestW4MethodNames:
    def test_the_rtn_name_carries_the_group_size(self) -> None:
        """At the default 32 the spelling is the same as before (does not break matching against
        past research records)."""
        assert measure.rtn_method_name(DEFAULT_G) == "rtn-i4-g32"
        assert measure.rtn_method_name(16) == "rtn-i4-g16"

    def test_the_table_keys_do_not_move_with_g(self) -> None:
        """If the keys moved with g, the configuration table (what `w4-rtn` points to) could not be
        looked up per g."""
        assert list(measure.w4_methods(64)) == list(measure.w4_methods(DEFAULT_G))
        assert measure.CONFIGS["w4-rtn"].method in measure.w4_methods(64)

    def test_the_formula_states_the_group_and_its_bits_per_weight(self) -> None:
        """The formula goes into the projection table as is — if changing g left the bpw unchanged,
        the table could not be read."""
        assert (
            measure.w4_methods(DEFAULT_G)[measure.RTN_KIND].formula
            == "4bit + g32 f32 scale = 5.0 bpw"
        )
        assert measure.w4_methods(64)[measure.RTN_KIND].formula == "4bit + g64 f32 scale = 4.5 bpw"


class TestW4GroupSizeReach:
    """The given g reaches **both the eligibility check and the rounding** (if only one moved, they
    would silently split)."""

    def test_a_smaller_group_widens_the_eligible_set(self, tiny) -> None:
        """At g16 the linear with receptive field 48 is divisible too (evidence that the eligibility
        check looks at g)."""
        census = measure.census_w4_targets(tiny, measure.W4_ROLES["all"], 16)

        assert sorted(census.eligible) == ["conv", "emb", "linear", "odd_linear"]
        assert census.excluded == ("depthwise",)

    def test_a_larger_group_narrows_it(self, tiny) -> None:
        census = measure.census_w4_targets(tiny, measure.W4_ROLES["all"], 64)

        assert census.eligible == ("linear",)
        assert sorted(census.excluded) == ["conv", "depthwise", "emb", "odd_linear"]

    def test_the_counts_carry_the_group_they_were_counted_with(self, tiny) -> None:
        """The counts carry g, so the bpw projection can never use another g."""
        census = measure.census_w4_targets(tiny, measure.W4_ROLES["all"], 16)

        assert census.counts.group_size == 16
        assert census.counts.groups == census.counts.elements // 16

    def test_counts_from_different_groups_cannot_be_summed(self) -> None:
        """A sum such as front + voice that mixes g fails loudly (the hole of silently taking one
        side's g)."""
        left = measure.TargetCounts(modules=1, channels=1, elements=64, group_size=16)
        right = measure.TargetCounts(modules=1, channels=1, elements=64, group_size=32)

        with pytest.raises(AssertionError, match="group 長の違う計数"):
            _ = left + right

    @pytest.mark.parametrize("group_size", [16, 64])
    @pytest.mark.parametrize("kind", list(measure.w4_methods(measure.DEFAULT_GROUP_SIZE)))
    def test_the_rounding_covers_exactly_the_eligible_set(self, kind, group_size) -> None:
        """The eligibility check and the rounding look at **the same g** (for all 4 methods).

        If the method side kept holding 32, core would reject the receptive-field-48 linear that
        became eligible at g16 for its partial group — it shows up as an exception, not a value,
        so this one test is the check against forgetting to pass g.
        """
        torch.manual_seed(SEED)
        model = TinyNet()
        census = measure.census_w4_targets(model, measure.W4_ROLES["all"], group_size)

        report = measure.w4_methods(group_size)[kind].apply(
            model, measure.W4_ROLES["all"], census.include()
        )

        assert (report.modules, report.elements) == (
            census.counts.modules,
            census.counts.elements,
        )

    def test_the_rtn_report_states_the_group_it_used(self) -> None:
        """RTN's count report states g verbatim (the string that goes into the `[bert:…]` line and
        `weight_quant`)."""
        torch.manual_seed(SEED)
        model = TinyNet()
        census = measure.census_w4_targets(model, measure.W4_ROLES["all"], 16)

        report = measure.w4_methods(16)[measure.RTN_KIND].apply(
            model, measure.W4_ROLES["all"], census.include()
        )

        assert "group 16" in report.describe()

    @pytest.mark.parametrize("kind", list(measure.w4_methods(measure.DEFAULT_GROUP_SIZE)))
    def test_every_method_rounds_to_other_values_under_another_group(self, kind) -> None:
        """g reaches the `fake_quant_*` of **all** 4 methods (forgetting to pass it moves not a
        single value).

        Which weights move differs per method — MXFP4's scale is a power of 2, so splitting the
        group lands on the same value unless the exponent of amax changes (measured: only the
        normally distributed embedding moves). So what is pinned is only "at least one moves".
        """
        rounded: dict[int, dict[str, torch.Tensor]] = {}
        for group_size in (16, DEFAULT_G):
            torch.manual_seed(SEED)
            model = TinyNet()
            census = measure.census_w4_targets(model, measure.W4_ROLES["all"], group_size)
            measure.w4_methods(group_size)[kind].apply(
                model, measure.W4_ROLES["all"], census.include()
            )
            rounded[group_size] = {
                name: getattr(model, name).weight.clone() for name in ("linear", "conv", "emb")
            }

        moved = [
            name
            for name, weight in rounded[16].items()
            if not torch.equal(weight, rounded[DEFAULT_G][name])
        ]
        assert moved, f"{kind}: the rounding does not move with g"


#: The part count of the synthetic distributed form (2 documents + const region + weights).
DIST_PARTS = 3

_LINEAR_KEY = "enc_p.style_proj.weight"
_LINEAR_BIAS_KEY = "enc_p.style_proj.bias"
_EMB_KEY = "enc_p.emb.weight"


def _write_dist(root: Path) -> None:
    """Writes the minimal distributed form (one i8 linear + one i8 embedding) as a container.

    Both are rank-2 i8, so the failure mode "picking by shape alone mixes in the embedding" can be
    reproduced as it is (the docstring of {@link measure.project_distribution}).
    """
    root.mkdir(parents=True, exist_ok=True)
    initializers = {
        "w_linear": IrInitializer(tensor=_LINEAR_KEY, storage=IrStorage(dtype="f32")),
        "b_linear": IrInitializer(tensor=_LINEAR_BIAS_KEY, storage=IrStorage(dtype="f32")),
        "w_emb": IrInitializer(tensor=_EMB_KEY, storage=IrStorage(dtype="f32")),
    }
    weights = {_LINEAR_KEY: torch.zeros(8, 64), _EMB_KEY: torch.zeros(4, 32)}
    # linear's contract is 3 inputs (bias included) — the bias stays rank-1 f32 and is not counted
    # among the projection targets (rank-2 i8).
    tensors = {**weights, _LINEAR_BIAS_KEY: torch.zeros(8)}
    # The graph inputs (`x` / `ids`) are declared in inputs[] only — repeating them in values is a
    # double declaration that breaks the IR's acceptance rules (the publish check rejects it).
    values = {
        "w_linear": IrValue(dtype="f32", shape=[8, 64]),
        "b_linear": IrValue(dtype="f32", shape=[8]),
        "w_emb": IrValue(dtype="f32", shape=[4, 32]),
        "h": IrValue(dtype="f32", shape=[1, 8]),
        "e": IrValue(dtype="f32", shape=[1, 1, 32]),
    }
    graph = IrGraph(
        inputs=[
            IrInput(name="x", dtype="f32", shape=[1, 64]),
            IrInput(name="ids", dtype="i32", shape=[1, 1]),
        ],
        outputs=["h", "e"],
        initializers=initializers,
        values=values,
        nodes=[
            IrNode(op="linear", ins=["x", "w_linear", "b_linear"], outs=["h"], attrs={}),
            IrNode(op="embedding", ins=["w_emb", "ids"], outs=["e"], attrs={"padding_idx": -1}),
        ],
    )
    scales = {key: channel_scale(tensor, 0) for key, tensor in weights.items()}
    stored = stored_model(graph, tensors, weight_dtype="i8", weight_scales=scales)
    publish_container(
        root / "model.i8.krm",
        stored.graph,
        stored.tensors,
        stored.bindings,
        graph_name="front",
        provenance=Provenance(license="apache-2.0", writer="karume-fixture"),
    )


class TestDistributionProjection:
    LINEAR = frozenset({"enc_p.style_proj.weight"})

    def test_it_projects_only_the_linear_weight_slots(self, tmp_path: Path) -> None:
        _write_dist(tmp_path / "F1" / "front")
        projection = measure.project_distribution(tmp_path, self.LINEAR, DEFAULT_G)
        group = projection["groups"]["話者ごとの net_g"]
        assert group["tensors"] == 1
        # i8 = 8·64 + 4·8 bytes / i4 = 8·64/2 + 4·8·(64/32) bytes
        assert group["current_bytes"] == 8 * 64 + 4 * 8
        assert group["projected_bytes"] == 8 * 64 // 2 + 4 * 8 * (64 // 32)
        assert projection["delta_bytes"] == group["delta_bytes"]

    def test_a_larger_group_charges_fewer_scale_bytes(self, tmp_path: Path) -> None:
        """The shrink projection uses the same g too (raising g lowers the scale cost
        accordingly)."""
        _write_dist(tmp_path / "F1" / "front")

        projection = measure.project_distribution(tmp_path, self.LINEAR, 64)

        assert projection["group_size"] == 64
        assert projection["groups"]["話者ごとの net_g"]["projected_bytes"] == (
            8 * 64 // 2 + 4 * 8 * (64 // 64)
        )

    def test_it_reports_the_rank2_int8_tensors_it_did_not_match(self, tmp_path: Path) -> None:
        """Picking by the shape "rank-2 I8" alone mixes in the embedding — reports how many were
        passed over."""
        _write_dist(tmp_path / "F1" / "front")
        projection = measure.project_distribution(tmp_path, self.LINEAR, DEFAULT_G)
        assert projection["rank2_i8_not_linear"] == 1

    def test_it_separates_the_shared_subtree_from_the_per_speaker_one(self, tmp_path: Path) -> None:
        _write_dist(tmp_path / "F1" / "front")
        _write_dist(tmp_path / "shared" / "text_encoder")
        projection = measure.project_distribution(tmp_path, self.LINEAR, DEFAULT_G)
        assert sorted(projection["groups"]) == [
            "shared（DeBERTa text_encoder）",
            "話者ごとの net_g",
        ]
        assert projection["files"] == 2 * DIST_PARTS

    def test_it_measures_the_shrink_against_the_real_file_bytes(self, tmp_path: Path) -> None:
        _write_dist(tmp_path / "F1" / "front")
        projection = measure.project_distribution(tmp_path, self.LINEAR, DEFAULT_G)
        total = sum(path.stat().st_size for path in tmp_path.rglob("*") if path.is_file())
        assert projection["total_bytes"] == total
        assert projection["shrink_of_total"] == projection["delta_bytes"] / total


class TestConfigSelection:
    def test_it_runs_every_config_by_default(self) -> None:
        assert measure.selected_configs(None) == tuple(measure.CONFIGS)

    def test_it_always_keeps_f32_as_the_snr_baseline(self) -> None:
        assert measure.selected_configs("w4-rtn") == ("f32", "w4-rtn")

    def test_it_keeps_the_declaration_order(self) -> None:
        assert measure.selected_configs("w4-nf4,w8") == ("f32", "w8", "w4-nf4")

    def test_it_rejects_an_unknown_config(self) -> None:
        with pytest.raises(SystemExit, match="未知の構成"):
            measure.selected_configs("w4-int3")


# ---- calibrated rounding (iteration J-2) ------------------------------------


class TinyLayer(nn.Module):
    """A stage body standing in for `DebertaV2Layer` (the mask's effect is made observable as a
    **multiplication**).

    Returning a `(hidden, None)` tuple is the same as the real one — it checks that
    {@link measure.EncoderStage} picks the first element.
    """

    def __init__(self, features: int) -> None:
        super().__init__()
        self.query_proj = nn.Linear(features, features, bias=False)
        self.dense = nn.Linear(features, features, bias=False)

    def forward(
        self,
        hidden: torch.Tensor,
        attention_mask: torch.Tensor,
        relative_pos: torch.Tensor | None = None,
        output_attentions: bool = False,
    ) -> tuple[torch.Tensor, None]:
        return (self.dense(torch.tanh(self.query_proj(hidden * attention_mask))), None)


class TinyConv(nn.Module):
    """Stands in for `ConvLayer` — **holds no `nn.Linear` at all** (does not affect the scan's
    target set)."""

    def __init__(self, features: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(features, features, 3, padding=1)

    def forward(
        self, hidden: torch.Tensor, residual: torch.Tensor, input_mask: torch.Tensor
    ) -> torch.Tensor:
        out = self.conv(hidden.transpose(1, 2)).transpose(1, 2)
        return (residual + torch.tanh(out)) * input_mask.unsqueeze(-1)


class TinyEncoder(nn.Module):
    """Stands in for `DebertaV2Encoder` — passes the mask as a **positional argument** and applies
    the conv only to the first layer's output."""

    def __init__(self, features: int, layers: int) -> None:
        super().__init__()
        self.layer = nn.ModuleList(TinyLayer(features) for _ in range(layers))
        self.conv = TinyConv(features)

    def forward(
        self, hidden: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        input_mask = attention_mask.to(hidden.dtype)
        expanded = input_mask.unsqueeze(-1)
        relative_pos = torch.arange(int(hidden.shape[1]), dtype=torch.int64)
        states = [hidden]
        for index, layer in enumerate(self.layer):
            output = layer(hidden, expanded, relative_pos=relative_pos, output_attentions=False)[0]
            if index == 0:
                output = self.conv(hidden, output, input_mask)
            hidden = output
            states.append(hidden)
        return tuple(states)


class TinyBert(nn.Module):
    """Stands in for `DebertaV2Model` — the `encoder.layer` / `encoder.conv` that
    `measure.bert_stages` walks.

    forward returns **the list of all hidden states**, like the real one with
    `output_hidden_states=True` (so that the equivalence of the stage decomposition can be matched
    by layer number).
    """

    def __init__(self, vocab: int = 16, features: int = 32, layers: int = 4) -> None:
        super().__init__()
        self.embeddings = nn.Embedding(vocab, features)
        self.encoder = TinyEncoder(features, layers)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        return self.encoder(self.embeddings(input_ids), attention_mask)


class SilentBert(TinyBert):
    """A forward that **does not call** the first stage (exercises the Catcher's fail-loudly
    path)."""

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        return (self.embeddings(input_ids),)


def tiny_calib_inputs(count: int = 3) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    """`(input_ids, attention_mask)` of different lengths (stand-ins for calibration inputs — no
    tokenizer involved)."""
    return tuple(
        (
            torch.arange(3 + index, dtype=torch.int64).unsqueeze(0) % 16,
            torch.ones(1, 3 + index, dtype=torch.int64),
        )
        for index in range(count)
    )


def tiny_rig(bert: TinyBert, inputs) -> measure.CalibRig:
    """An asset-free version of `build_calib_rig` (only the disjointness check, which needs the
    dump's meta, is left out)."""
    stages = measure.bert_stages(bert)
    scan, counts = measure.calib_targets(stages)
    return measure.CalibRig(
        stages=stages,
        scan=scan,
        counts=counts,
        batches=measure.capture_stage_batches(bert, inputs),
    )


class TestCalibConfigs:
    def test_the_calibrated_configs_are_selectable(self) -> None:
        assert measure.selected_configs("bert-gptq-rtn,bert-gptq-kmeans") == (
            "f32",
            "bert-gptq-rtn",
            "bert-gptq-kmeans",
        )

    def test_they_keep_net_g_in_f32(self) -> None:
        """Configurations that vary only BERT — not a single weight of the generator network is
        rounded (the same contract as the existing bert-w4-*)."""
        for name in measure.CALIB_CONFIGS:
            recipe = measure.CONFIGS[f"bert-{name}"]
            assert (recipe.weight, recipe.act, recipe.scope) == (None, None, ())

    def test_the_group_size_is_pinned_for_every_calib_config(self) -> None:
        """Calibration does not change the grid — what varies is the method, not g.

        The pinned value is 32, the same as the default, but it does **not follow**
        `--w4-group-size` (an axis of the method grid only). Going back to the spelling from when
        the value was shared would silently move the calibration grid too whenever g varies.
        """
        assert measure.CALIB_GROUP_SIZE == DEFAULT_G
        assert all(
            config.grid.group_size == measure.CALIB_GROUP_SIZE
            for config in measure.CALIB_CONFIGS.values()
        )

    def test_no_calib_config_fits_its_table_on_a_subsample(self) -> None:
        """The table is always fitted on the full set (no subsample escape hatch on the net_g side
        either)."""
        assert all(config.grid.fit_stride == 1 for config in measure.CALIB_CONFIGS.values())

    def test_the_label_names_both_the_method_and_the_grid(self) -> None:
        """The method names of the `w4` section pair the method with the storage grid (one alone
        would make the rows unreadable)."""
        labels = {name: config.label for name, config in measure.CALIB_CONFIGS.items()}

        assert labels == {
            "gptq-rtn": "gptq/rtn",
            "gptq-nf4": "gptq/nf4",
            "gptq-kmeans": "gptq/kmeans_shared",
        }

    def test_the_group_scale_grids_project_five_bits(self) -> None:
        counts = measure.TargetCounts(
            modules=2, channels=8, elements=65536, group_size=measure.CALIB_GROUP_SIZE
        )

        assert measure.group_scale_bits(counts) / counts.elements == 5.0

    def test_the_calibrated_codebook_costs_one_table_per_layer(self) -> None:
        """core's `kmeans_shared` is a **per-layer** table — a different formula from
        `kmeans:shared`, one table overall."""
        counts = measure.TargetCounts(
            modules=2, channels=8, elements=65536, group_size=measure.CALIB_GROUP_SIZE
        )

        difference = measure.layer_table_bits(counts) - measure.group_scale_bits(counts)

        assert difference == measure.CODEBOOK_BITS * counts.modules


class TestBertVariant:
    METHODS = measure.w4_methods(DEFAULT_G)

    def test_it_keys_the_plain_and_calibrated_roundings_in_one_space(self) -> None:
        assert measure.bert_variant(measure.CONFIGS["bert-w4-nf4"], self.METHODS) == "nf4"
        assert measure.bert_variant(measure.CONFIGS["bert-gptq-nf4"], self.METHODS) == "gptq-nf4"
        assert measure.bert_variant(measure.CONFIGS["f32"], self.METHODS) is None

    def test_it_keys_the_plain_rounding_by_the_name_that_carries_g(self) -> None:
        """The field name is the method name with g baked in — two runs with different g get
        separate `w4.bert_quant` fields."""
        recipe = measure.CONFIGS["bert-w4-rtn"]

        assert measure.bert_variant(recipe, self.METHODS) == "rtn-i4-g32"
        assert measure.bert_variant(recipe, measure.w4_methods(16)) == "rtn-i4-g16"

    def test_the_two_name_spaces_do_not_collide(self) -> None:
        """If they overlapped, one key could not look up one entry (the feature cache would silently
        mix)."""
        names = {method.name for method in self.METHODS.values()}
        assert not names & set(measure.CALIB_CONFIGS)

    def test_a_recipe_that_asks_for_both_is_rejected(self) -> None:
        recipe = measure.Recipe(None, None, scope=(), bert_method="nf4", bert_calib="gptq-nf4")

        with pytest.raises(AssertionError, match="どちらか一方"):
            measure.bert_variant(recipe, self.METHODS)


class TestCalibCorpus:
    def test_it_takes_the_head_of_the_corpus(self) -> None:
        assert measure.calib_corpus(4) == measure.CALIB_TEXTS[:4]
        assert measure.calib_corpus(None) == measure.CALIB_TEXTS

    def test_it_rejects_a_non_positive_limit(self) -> None:
        with pytest.raises(ValueError, match="1 以上"):
            measure.calib_corpus(0)


class TestCalibDisjoint:
    """MUST: calibration and evaluation are disjoint (an overlap would make the number measure
    "could it reproduce a sentence seen in calibration")."""

    EVALUATED = ("こんにちは、これはテストです。", "こんにちは,これはテストです.")

    def test_the_shipped_corpus_is_separate_from_the_evaluation_text(self) -> None:
        measure.assert_calib_disjoint(measure.CALIB_TEXTS, self.EVALUATED)

    def test_it_catches_a_calib_text_that_quotes_the_evaluation_text(self) -> None:
        quoting = ("昨日、" + self.EVALUATED[0] + "と言われた。",)

        with pytest.raises(AssertionError, match="重なっている"):
            measure.assert_calib_disjoint(quoting, self.EVALUATED)

    def test_it_catches_a_calib_text_quoted_by_the_evaluation_text(self) -> None:
        """A partial match is not found in one direction only (`in` is checked both ways)."""
        quoted = ("これはテストです。",)

        with pytest.raises(AssertionError, match="重なっている"):
            measure.assert_calib_disjoint(quoted, self.EVALUATED)


class TestBertStages:
    def test_it_stops_at_the_layer_the_feature_is_taken_from(self) -> None:
        """The last layer does not affect the feature by a single bit, so it is not a stage (the
        position of `bert_feature_of`)."""
        bert = TinyBert(layers=4)

        stages = measure.bert_stages(bert)

        assert len(stages) == 4 - (measure.demo.BERT_HIDDEN_FROM_END - 1)

    def test_only_the_first_stage_carries_the_conv(self) -> None:
        """Only the first layer has ConvLayer applied to its output (`i == 0` in
        `DebertaV2Encoder`)."""
        stages = measure.bert_stages(TinyBert())

        assert [stage.conv is not None for _prefix, stage in stages] == [True, False]

    def test_the_scan_uses_model_wide_fqns(self) -> None:
        """The prefix that aligns the ledger keys with the FQN space of `Int4Report` (core's
        contract)."""
        scan, counts = measure.calib_targets(measure.bert_stages(TinyBert()))

        assert sorted(scan) == [
            "encoder.layer.0.dense.weight",
            "encoder.layer.0.query_proj.weight",
            "encoder.layer.1.dense.weight",
            "encoder.layer.1.query_proj.weight",
        ]
        assert counts.modules == 4

    def test_the_scan_counts_carry_the_calibration_grid(self) -> None:
        """The g the calibration counts carry is **the calibration grid** (not the method axis's g —
        the projection comes from here)."""
        _scan, counts = measure.calib_targets(measure.bert_stages(TinyBert()))

        assert counts.group_size == measure.CALIB_GROUP_SIZE

    def test_the_conv_is_not_in_the_scan(self) -> None:
        """ConvLayer is registered as a child but holds no `nn.Linear`, so it does not move the
        target set."""
        scan, _counts = measure.calib_targets(measure.bert_stages(TinyBert()))

        assert not any("conv" in fqn for fqn in scan)

    def test_a_weight_whose_axis_is_not_divisible_fails_loudly(self) -> None:
        """Unlike the net_g census, it is **not excluded** — no path to silently shrink is made on
        the calibration side."""
        bert = TinyBert(features=32)
        bert.encoder.layer[0].dense = nn.Linear(48, 48, bias=False)

        with pytest.raises(AssertionError, match="割り切れない"):
            measure.calib_targets(measure.bert_stages(bert))


class TestCatcher:
    def test_it_captures_one_batch_per_calibration_text(self) -> None:
        bert = TinyBert()

        batches = measure.capture_stage_batches(bert, tiny_calib_inputs(3))

        assert len(batches) == 3
        assert [int(args[0].shape[1]) for args, _kwargs in batches] == [3, 4, 5]

    def test_it_moves_the_mask_from_positional_to_keyword(self) -> None:
        """`calibrate_stages` passes only one positional argument to the next stage (core's
        driver)."""
        bert = TinyBert()

        (args, kwargs), *_ = measure.capture_stage_batches(bert, tiny_calib_inputs(1))

        assert len(args) == 1
        assert sorted(kwargs) == [
            "attention_mask",
            measure.INPUT_MASK_KWARG,
            "output_attentions",
            "relative_pos",
        ]

    def test_it_captures_the_two_dimensional_mask_the_conv_needs(self) -> None:
        """The mask passed to the layers and the one passed to ConvLayer are different things
        (copying one would make one side leave the spec)."""
        bert = TinyBert()

        (_args, kwargs), *_ = measure.capture_stage_batches(bert, tiny_calib_inputs(1))

        assert kwargs[measure.INPUT_MASK_KWARG].dim() == 2
        assert kwargs["attention_mask"].dim() == 3

    def test_the_hooks_are_removed_even_though_the_forward_was_aborted(self) -> None:
        """Cleanup completes even when a sentinel aborts the forward (leftover hooks would make
        every later forward fail)."""
        bert = TinyBert()

        measure.capture_stage_batches(bert, tiny_calib_inputs(1))

        assert not bert.encoder.layer[0]._forward_pre_hooks
        assert not bert.encoder.conv._forward_pre_hooks
        with torch.no_grad():
            bert(*tiny_calib_inputs(1)[0])

    def test_a_forward_that_never_reaches_the_first_stage_fails_loudly(self) -> None:
        """MUST: letting a pass-through go silently would hide it until the "zero calibration
        inputs" diagnostic."""
        with pytest.raises(AssertionError, match="揃わなかった"):
            measure.capture_stage_batches(SilentBert(), tiny_calib_inputs(1))


class TestStageChain:
    def test_the_stage_chain_reproduces_the_encoder(self) -> None:
        """The stage decomposition produces **the same hidden** as the real encoder (the detector
        for a dropped first conv)."""
        torch.manual_seed(SEED)
        bert = TinyBert()
        inputs = tiny_calib_inputs(1)
        with torch.no_grad():
            states = bert(*inputs[0])
        rig = tiny_rig(bert, inputs)
        args, kwargs = rig.batches[0]

        hidden = args[0]
        with torch.no_grad():
            for _prefix, stage in rig.stages:
                hidden = stage(hidden, **kwargs)

        assert torch.equal(hidden, states[len(rig.stages)])

    def test_dropping_the_conv_moves_the_first_stage(self) -> None:
        """Fault injection for the detector above — dropping the conv really gives another
        hidden."""
        torch.manual_seed(SEED)
        bert = TinyBert()
        inputs = tiny_calib_inputs(1)
        with torch.no_grad():
            states = bert(*inputs[0])
        args, kwargs = measure.capture_stage_batches(bert, inputs)[0]
        bare = measure.EncoderStage(0, bert.encoder.layer[0], None)

        with torch.no_grad():
            output = bare(args[0], **kwargs)

        assert not torch.equal(output, states[1])


class TestCalibrationRun:
    def test_it_rounds_every_scanned_linear(self) -> None:
        torch.manual_seed(SEED)
        bert = TinyBert()
        rig = tiny_rig(bert, tiny_calib_inputs(2))
        before = {fqn: weight.detach().clone() for fqn, weight in rig.scan.items()}

        report = measure.apply_calib(measure.CALIB_CONFIGS["gptq-rtn"], rig)
        measure.assert_calib_covers_scan(report, rig.scan, "gptq-rtn")

        assert report.modules == len(rig.scan)
        assert all(not torch.equal(rig.scan[fqn], value) for fqn, value in before.items())

    def test_it_leaves_the_layers_beyond_the_feature_untouched(self) -> None:
        """Layers outside the stages stay f32 (evidence that the target set is a subset of
        `bert:linear`)."""
        torch.manual_seed(SEED)
        bert = TinyBert(layers=4)
        beyond = bert.encoder.layer[3].query_proj.weight.detach().clone()
        rig = tiny_rig(bert, tiny_calib_inputs(2))

        measure.apply_calib(measure.CALIB_CONFIGS["gptq-rtn"], rig)

        assert torch.equal(bert.encoder.layer[3].query_proj.weight, beyond)

    def test_every_calib_config_runs_end_to_end(self) -> None:
        """All 3 go through core's driver (every pair of method and storage grid is alive)."""
        for name, config in measure.CALIB_CONFIGS.items():
            torch.manual_seed(SEED)
            bert = TinyBert()
            rig = tiny_rig(bert, tiny_calib_inputs(2))

            report = measure.apply_calib(config, rig)
            measure.assert_calib_covers_scan(report, rig.scan, name)

            assert report.method == config.method
            assert report.grid == config.grid.kind

    def test_the_three_grids_round_to_different_values(self) -> None:
        """Detects a dispatch slip between storage grids (applying the same grid twice is
        bit-identical)."""
        rounded: dict[str, torch.Tensor] = {}
        for name, config in measure.CALIB_CONFIGS.items():
            torch.manual_seed(SEED)
            bert = TinyBert()
            rig = tiny_rig(bert, tiny_calib_inputs(2))
            measure.apply_calib(config, rig)
            rounded[name] = bert.encoder.layer[0].query_proj.weight.clone()
        names = list(rounded)
        for index, left in enumerate(names):
            for right in names[index + 1 :]:
                assert not torch.equal(rounded[left], rounded[right]), f"{left} and {right}"

    def test_a_scan_the_calibration_missed_fails_loudly(self) -> None:
        """Blocks a vacuous check — a missed rounding shows up on the **good side** of SNR / LSD, so
        the numbers cannot reveal it."""
        torch.manual_seed(SEED)
        bert = TinyBert()
        rig = tiny_rig(bert, tiny_calib_inputs(2))
        report = measure.apply_calib(measure.CALIB_CONFIGS["gptq-rtn"], rig)
        widened = {**rig.scan, "encoder.layer.9.query_proj.weight": torch.zeros(1)}

        with pytest.raises(AssertionError, match="一致しない"):
            measure.assert_calib_covers_scan(report, widened, "gptq-rtn")
