"""QAT の格納・丸めの検査が、通常 Gemma との取り違えを検出する。"""

from copy import deepcopy

import pytest

from gemma4_qat.config import CHECKPOINTS, checkpoint_name, series_name
from gemma4_qat.distribution import QAT_DEFAULT_QUANT, assert_qat_graph, qat_quants, repo_name
from karume.dist import DistError, assert_quant_presentation


def graph_fixture():
    """正当な最小の QAT グラフ。

    量子化 linear は 2 本 — 共有 head（token embedding と同じ initializer）と、ごく普通の
    量子化 linear 1 本。前者は前後の SRQ を持たない（上流 lm_head の SRQ scale が 0 = 恒等
    なので recipe が挟まない）、後者は前後とも持つ。この非対称が契約そのものなので、
    故障注入は**普通の linear 側**へ掛ける。
    """
    return {
        "initializers": {
            "weight": {"tensor": "head", "storage": {"dtype": "i2"}},
            "mlp": {"tensor": "mlp", "storage": {"dtype": "i4"}},
            "projection": {
                "tensor": "model.model.per_layer_model_projection.weight",
                "storage": {"dtype": "f32"},
            },
        },
        "nodes": [
            {"op": "embedding", "ins": ["weight", "input_ids"], "outs": ["embedded"]},
            {"op": "linear", "ins": ["embedded", "projection"], "outs": ["projected"]},
            {"op": "static_quantize", "ins": ["projected"], "outs": ["rounded"]},
            {"op": "linear", "ins": ["rounded", "mlp"], "outs": ["mlp_out"]},
            {"op": "static_quantize", "ins": ["mlp_out"], "outs": ["hidden"]},
            {"op": "linear", "ins": ["hidden", "weight"], "outs": ["logits"]},
        ],
    }


#: {@link graph_fixture} の head linear（最後のノード）の位置。
HEAD_NODE = 5


def wrap_head_with_srq(graph):
    """head linear の前後へ SRQ を足す（scale>0 で焼かれた形 — 省略しない側の系列）。"""
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
        assert_qat_graph(graph)
        assert graph == before

    def test_accepts_a_shared_head_that_still_carries_its_srq(self):
        """head の SRQ は「省略してよい」であって禁止ではない（scale>0 で焼かれた形も通す）。"""
        assert_qat_graph(wrap_head_with_srq(graph_fixture()))

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
            graph["nodes"].append({"op": "reshape", "ins": ["mlp_out"], "outs": ["unrounded"]})
        elif fault == "unquantized":
            graph["initializers"]["projection"]["tensor"] = "another.linear.weight"
        else:
            # 共有でなくなった head は普通の量子化 linear になるので SRQ を補う — 残る違反を
            # 「共有 head が 1 本でない」1 つに絞り、SRQ 欠落の経路と取り違えないため。
            graph["initializers"]["separate"] = deepcopy(graph["initializers"]["weight"])
            wrap_head_with_srq(graph)["nodes"][HEAD_NODE + 1]["ins"][1] = "separate"
        with pytest.raises(DistError):
            assert_qat_graph(graph)

    def test_only_the_shared_head_may_omit_its_srq(self):
        """省略を許すのは共有 head だけ — 同じ省略を普通の量子化 linear がすると落ちる。"""
        assert_qat_graph(graph_fixture())
        graph = graph_fixture()
        graph["nodes"][2]["op"] = "reshape"
        with pytest.raises(DistError, match="固定 SRQ"):
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
    @pytest.mark.parametrize("model", ["e2b", "e4b"])
    def test_every_quant_fits_the_manifest_presentation_contract(self, model):
        for name, quant in qat_quants(model).items():
            assert_quant_presentation(f"{model}.{name}", quant)

    def test_parallel_uses_the_same_weights_and_keeps_the_reference(self):
        modes = qat_quants("e2b")
        assert list(modes) == ["i4", "i4-gemvpar", "i4-fast"]
        assert modes["i4"]["session"] == {}
        assert modes["i4-gemvpar"]["weights"] == modes["i4"]["weights"]
        assert modes["i4-gemvpar"]["session"] == {"linearGemvReduce": "parallel"}
        assert modes["i4-fast"]["weights"] == modes["i4"]["weights"]
        assert modes["i4-fast"]["session"] == {
            "linearGemvReduce": "parallel",
            "fuseRmsNormAdd": True,
            "fuseLinearStaticQuantize": True,
            "packedStaticQuantize": True,
        }

    def test_unmeasured_e4b_keeps_its_single_reference_mode(self):
        modes = qat_quants("e4b")
        assert list(modes) == ["i4"]
        assert modes["i4"]["session"] == {}

    def test_unknown_model_is_rejected(self):
        with pytest.raises(ValueError, match="未対応"):
            qat_quants("12b")


class TestQatDefaultQuant:
    @pytest.mark.parametrize(("model", "expected"), [("e2b", "i4-fast"), ("e4b", "i4")])
    def test_measured_series_takes_the_fused_default(self, model, expected):
        assert QAT_DEFAULT_QUANT[model] == expected

    @pytest.mark.parametrize("model", ["e2b", "e4b"])
    def test_default_names_a_quant_the_model_offers(self, model):
        assert QAT_DEFAULT_QUANT[model] in qat_quants(model)

    def test_every_checkpoint_declares_a_default(self):
        assert set(QAT_DEFAULT_QUANT) == set(CHECKPOINTS)
