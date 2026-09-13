"""QAT の格納・丸めの検査が、通常 Gemma との取り違えを検出する。"""

from copy import deepcopy

import pytest

from gemma4_qat.config import checkpoint_name, series_name
from gemma4_qat.distribution import assert_qat_graph, qat_quants, repo_name
from karume.dist import DistError


def graph_fixture():
    return {
        "initializers": {
            "weight": {"tensor": "head", "storage": {"dtype": "i2"}},
            "projection": {
                "tensor": "model.model.per_layer_model_projection.weight",
                "storage": {"dtype": "f32"},
            },
        },
        "nodes": [
            {"op": "embedding", "ins": ["weight", "input_ids"], "outs": ["embedded"]},
            {"op": "linear", "ins": ["embedded", "projection"], "outs": ["projected"]},
            {"op": "static_quantize", "ins": ["projected"], "outs": ["rounded"]},
            {"op": "linear", "ins": ["rounded", "weight"], "outs": ["head"]},
            {"op": "static_quantize", "ins": ["head"], "outs": ["logits"]},
        ],
    }


class TestQatGraph:
    def test_accepts_fixed_head_sharing_and_explicit_srq(self):
        graph = graph_fixture()
        before = deepcopy(graph)
        assert_qat_graph(graph)
        assert graph == before

    @pytest.mark.parametrize(
        "fault", ["embedding", "before", "after", "extra_consumer", "unquantized", "untied"]
    )
    def test_detects_a_broken_qat_contract(self, fault):
        graph = graph_fixture()
        if fault == "embedding":
            graph["initializers"]["weight"]["storage"]["dtype"] = "i8"
        elif fault in ("before", "after"):
            graph["nodes"][2 if fault == "before" else 4]["op"] = "reshape"
        elif fault == "extra_consumer":
            graph["nodes"].append({"op": "reshape", "ins": ["head"], "outs": ["unrounded"]})
        elif fault == "unquantized":
            graph["initializers"]["projection"]["tensor"] = "another.linear.weight"
        else:
            graph["initializers"]["separate"] = deepcopy(graph["initializers"]["weight"])
            graph["nodes"][3]["ins"][1] = "separate"
        with pytest.raises(DistError):
            assert_qat_graph(graph)

    @pytest.mark.parametrize("model", ["e2b", "e4b"])
    def test_models_share_only_the_qat_family(self, model):
        assert repo_name(model) == "karume-gemma4-qat"
        assert series_name(model) == f"gemma4-qat-{model}-product"
        assert checkpoint_name(model).endswith("-qat-mobile-transformers")

    def test_unknown_model_is_rejected(self):
        with pytest.raises(ValueError, match="未対応"):
            series_name("12b")


class TestQatQuants:
    def test_parallel_uses_the_same_weights_and_keeps_the_reference(self):
        modes = qat_quants("e2b")
        assert list(modes) == ["i4", "i4-gemvpar"]
        assert modes["i4"]["session"] == {}
        assert modes["i4-gemvpar"]["weights"] == modes["i4"]["weights"]
        assert modes["i4-gemvpar"]["session"] == {"linearGemvReduce": "parallel"}

    def test_unmeasured_e4b_keeps_its_single_reference_mode(self):
        modes = qat_quants("e4b")
        assert list(modes) == ["i4"]
        assert modes["i4"]["session"] == {}

    def test_unknown_model_is_rejected(self):
        with pytest.raises(ValueError, match="未対応"):
            qat_quants("12b")
