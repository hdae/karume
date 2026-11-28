"""The QAT storage and rounding checks detect a mix-up with the regular Gemma."""

from copy import deepcopy

import pytest

from gemma4_qat.config import CHECKPOINTS, checkpoint_name, series_name
from gemma4_qat.distribution import (
    QAT_DEFAULT_QUANT,
    QAT_MEASUREMENTS,
    assert_qat_graph,
    qat_quants,
    repo_name,
)
from karume.dist import DistError, assert_quant_presentation

#: The tensor key of the only linear allowed to stay unquantized (the one the structural check
#: allows by name).
PROJECTION = "model.model.per_layer_model_projection.weight"


def graph_fixture():
    """The valid minimal QAT graph (**an IR v2 document** — initializer name = tensor key).

    There are two quantized linears — the shared head (the same initializer as the token
    embedding) and one quite ordinary quantized linear. The former has no SRQ on either side
    (upstream lm_head's SRQ scales are 0 = identity, so the recipe inserts none); the latter has
    both. This asymmetry is the contract itself, so faults are injected on **the ordinary
    linear's side**.
    """
    return {
        "initializers": {"head": {}, "mlp": {}, PROJECTION: {}},
        "nodes": [
            {"op": "embedding", "ins": ["head", "input_ids"], "outs": ["embedded"]},
            {"op": "linear", "ins": ["embedded", PROJECTION], "outs": ["projected"]},
            {"op": "static_quantize", "ins": ["projected"], "outs": ["rounded"]},
            {"op": "linear", "ins": ["rounded", "mlp"], "outs": ["mlp_out"]},
            {"op": "static_quantize", "ins": ["mlp_out"], "outs": ["hidden"]},
            {"op": "linear", "ins": ["hidden", "head"], "outs": ["logits"]},
        ],
    }


def layouts_fixture():
    """The binding table of {@link graph_fixture} (tensor key → storage layout).

    The IR v2 graph description has no storage (the source of truth is the binding table —
    docs/ir-v2.md), so storage faults are injected on this side.
    """
    return {"head": "i2", "mlp": "i4", PROJECTION: "f32"}


#: The position of the head linear (the last node) of {@link graph_fixture}.
HEAD_NODE = 5


def wrap_head_with_srq(graph):
    """Add SRQ before and after the head linear (the shape built with scale>0 — a series that
    does not omit it)."""
    head = graph["nodes"][HEAD_NODE]
    graph["nodes"].insert(
        HEAD_NODE, {"op": "static_quantize", "ins": [head["ins"][0]], "outs": ["head_in"]}
    )
    head["ins"][0] = "head_in"
    head["outs"] = ["head_out"]
    graph["nodes"].append({"op": "static_quantize", "ins": ["head_out"], "outs": ["logits"]})
    return graph


class TestQatGraph:
    def test_accepts_fixed_head_sharing_and_explicit_srq(self):
        graph = graph_fixture()
        before = deepcopy(graph)
        assert_qat_graph(graph, layouts_fixture())
        assert graph == before

    def test_accepts_a_shared_head_that_still_carries_its_srq(self):
        """The head's SRQ "may be omitted", not "is forbidden" (the shape built with scale>0
        passes too)."""
        assert_qat_graph(wrap_head_with_srq(graph_fixture()), layouts_fixture())

    @pytest.mark.parametrize(
        "fault", ["embedding", "before", "after", "extra_consumer", "unquantized", "untied"]
    )
    def test_detects_a_broken_qat_contract(self, fault):
        graph = graph_fixture()
        layouts = layouts_fixture()
        if fault == "embedding":
            layouts["head"] = "i8"
        elif fault in ("before", "after"):
            graph["nodes"][2 if fault == "before" else 4]["op"] = "reshape"
        elif fault == "extra_consumer":
            graph["nodes"].append({"op": "reshape", "ins": ["mlp_out"], "outs": ["unrounded"]})
        elif fault == "unquantized":
            graph["initializers"]["another.linear.weight"] = graph["initializers"].pop(PROJECTION)
            graph["nodes"][1]["ins"][1] = "another.linear.weight"
            layouts["another.linear.weight"] = layouts.pop(PROJECTION)
        else:
            # A head that is no longer shared becomes an ordinary quantized linear, so SRQ is
            # added — to narrow the remaining violation to the single "not exactly one shared
            # head" and not confuse it with the missing-SRQ path.
            graph["initializers"]["separate"] = {}
            layouts["separate"] = "i2"
            wrap_head_with_srq(graph)["nodes"][HEAD_NODE + 1]["ins"][1] = "separate"
        with pytest.raises(DistError):
            assert_qat_graph(graph, layouts)

    def test_only_the_shared_head_may_omit_its_srq(self):
        """Only the shared head may omit it — the same omission by an ordinary quantized linear
        fails."""
        assert_qat_graph(graph_fixture(), layouts_fixture())
        graph = graph_fixture()
        graph["nodes"][2]["op"] = "reshape"
        with pytest.raises(DistError, match="fixed SRQ"):
            assert_qat_graph(graph, layouts_fixture())

    @pytest.mark.parametrize("model", ["e2b", "e4b"])
    def test_models_share_only_the_qat_family(self, model):
        assert repo_name(model) == "karume-gemma4-qat-mobile"
        assert series_name(model) == f"gemma4-qat-{model}-product"
        assert checkpoint_name(model).endswith("-qat-mobile-transformers")

    def test_unknown_model_is_rejected(self):
        with pytest.raises(ValueError, match="unsupported"):
            series_name("12b")


class TestQatQuants:
    @pytest.mark.parametrize("model", ["e2b", "e4b"])
    def test_every_quant_fits_the_manifest_presentation_contract(self, model):
        for name, quant in qat_quants(model).items():
            assert_quant_presentation(f"{model}.{name}", quant)

    @pytest.mark.parametrize("model", ["e2b", "e4b"])
    def test_parallel_uses_the_same_weights_and_keeps_the_reference(self, model):
        """E2B / E4B have the same three presets with the same declarations (E4B got its presets
        from the measurement of 2026-09-26)."""
        modes = qat_quants(model)
        assert list(modes) == ["i4", "i4-gemvpar", "i4-fast"]
        assert modes["i4"]["session"] == {}
        assert modes["i4-gemvpar"]["weights"] == modes["i4"]["weights"]
        assert modes["i4-gemvpar"]["session"] == {"numerics": {"level": "reorder"}}
        assert modes["i4-fast"]["weights"] == modes["i4"]["weights"]
        assert modes["i4-fast"]["session"] == {
            "numerics": {"level": "reorder"},
            "packedStaticQuantize": True,
        }

    def test_the_descriptions_state_no_measurement_and_only_e4b_has_a_note(self):
        """E2B and E4B have the same descriptions: E4B's measurement of `i4-fast` is a dated note
        on the card (`QAT_MEASUREMENTS` — docs/spec/distribution-cards.md#presets-and-the-quant-
        tables), E2B has none."""
        e2b, e4b = qat_quants("e2b"), qat_quants("e4b")
        assert e4b == e2b
        assert "B570" not in e4b["i4-fast"]["description"]
        assert set(QAT_MEASUREMENTS) == {"e4b"}
        (note,) = QAT_MEASUREMENTS["e4b"]
        assert (note.preset, note.device, note.date) == (
            "i4-fast",
            "Intel Arc B570, Deno",
            "2026-09-26",
        )
        assert "41.7% shorter than with `i4`" in note.statement

    def test_unknown_model_is_rejected(self):
        with pytest.raises(ValueError, match="unsupported"):
            qat_quants("12b")


class TestQatDefaultQuant:
    @pytest.mark.parametrize("model", ["e2b", "e4b"])
    def test_measured_series_takes_the_fused_default(self, model):
        assert QAT_DEFAULT_QUANT[model] == "i4-fast"

    @pytest.mark.parametrize("model", ["e2b", "e4b"])
    def test_default_names_a_quant_the_model_offers(self, model):
        assert QAT_DEFAULT_QUANT[model] in qat_quants(model)

    def test_every_checkpoint_declares_a_default(self):
        assert set(QAT_DEFAULT_QUANT) == set(CHECKPOINTS)
