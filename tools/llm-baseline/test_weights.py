import json

import pytest
import torch
from safetensors.torch import save_file
from weights import (
    ContainerPle,
    DiskPle,
    assert_qat_checked_range,
    assert_srq_scales,
    ple_placement,
)


def test_original_ple_row_lookup_preserves_values_and_repeated_indices(tmp_path):
    value = torch.arange(8 * 12, dtype=torch.float32).reshape(8, 12).div(7).to(torch.bfloat16)
    save_file(
        {"model.language_model.embed_tokens_per_layer.weight": value},
        tmp_path / "model.safetensors",
    )
    ids = torch.tensor([[7, 0, 7, 1]])
    module = DiskPle(tmp_path, 4)
    assert torch.equal(module(ids), value.float()[ids] * 4)


#: 合成の PLE（容器の資産の形 — schema 3）。`values` と `scales` は**別々に**切られる。
TOKENS, LAYERS, DIM = 8, 2, 4
VALUES = torch.arange(-32, 32, dtype=torch.int8).reshape(TOKENS, LAYERS, DIM)
SCALES = (torch.arange(1, TOKENS * LAYERS + 1, dtype=torch.float32) / 16).reshape(TOKENS, LAYERS)


def _ple_index() -> dict:
    return {
        "schema": 3,
        "storage": "i8",
        "tokens": TOKENS,
        "layers": LAYERS,
        "dim": DIM,
        "embedScale": 4,
        # values は 2 block（境界は token 3）・scales は 1 block — 行の種別で切り方が違う。
        "values": {
            "rowBytes": LAYERS * DIM,
            "blocks": [
                {"asset": "ple.values.0", "start": 0, "stop": 3},
                {"asset": "ple.values.1", "start": 3, "stop": TOKENS},
            ],
        },
        "scales": {
            "rowBytes": LAYERS * 4,
            "blocks": [{"asset": "ple.scales.0", "start": 0, "stop": TOKENS}],
        },
    }


def _reader():
    assets = {
        "ple.values.0": VALUES[0:3].numpy().tobytes(),
        "ple.values.1": VALUES[3:TOKENS].numpy().tobytes(),
        "ple.scales.0": SCALES.numpy().tobytes(),
    }
    return lambda name, offset, length: assets[name][offset : offset + length]


def test_quantized_ple_keeps_layer_scales_and_block_boundaries():
    """block 境界をまたぐ token 列でも、層ごとの scale が行に掛かったまま戻る。"""
    module = ContainerPle(_ple_index(), _reader(), 4)
    ids = torch.tensor([[7, 2, 3, 2, 0]])
    expected = (VALUES.float() * SCALES.unsqueeze(-1)).flatten(1)[ids] * 4

    assert torch.equal(module(ids), expected)


def test_the_ple_record_names_the_row_lookup_that_was_installed(tmp_path):
    """loader.json の `ple` は、容器の資産を引く経路と公式 checkpoint を引く経路を取り違えない。"""
    save_file(
        {"model.language_model.embed_tokens_per_layer.weight": torch.zeros(8, 12)},
        tmp_path / "model.safetensors",
    )

    assert ple_placement(ContainerPle(_ple_index(), _reader(), 4)) == "container-row-lookup"
    assert ple_placement(DiskPle(tmp_path, 4)) == "disk-row-lookup"
    with pytest.raises(TypeError, match="未知の module"):
        ple_placement(torch.nn.Embedding(8, 12))


def test_a_ple_index_of_another_schema_is_refused():
    index = {**_ple_index(), "schema": 1}
    with pytest.raises(ValueError, match="schema"):
        ContainerPle(index, _reader(), 4)


def test_a_ple_storage_this_tool_does_not_expand_is_refused():
    """QAT の PLE（i4 / i2）は生バイトの一致で見る — 展開式をここへ増やさない。"""
    index = {**_ple_index(), "storage": "i4"}
    with pytest.raises(ValueError, match="i8"):
        ContainerPle(index, _reader(), 4)


def test_an_embed_scale_that_disagrees_with_the_official_model_is_refused():
    with pytest.raises(ValueError, match="embedScale"):
        ContainerPle(_ple_index(), _reader(), 8)


def test_a_token_outside_every_block_is_refused():
    index = json.loads(json.dumps(_ple_index()))
    index["values"]["blocks"] = index["values"]["blocks"][:1]
    module = ContainerPle(index, _reader(), 4)
    with pytest.raises(ValueError, match="token 5"):
        module(torch.tensor([[5]]))


def test_qat_reference_rejects_mismatched_activation_rounding():
    graph = {
        "nodes": [
            {"op": "static_quantize", "ins": ["x"], "outs": ["rounded"], "attrs": {"scale": 0.25}},
            {"op": "linear", "ins": ["rounded", "w"], "outs": ["y"]},
            {"op": "static_quantize", "ins": ["y"], "outs": ["z"], "attrs": {"scale": 0.5}},
        ]
    }
    assert assert_srq_scales(graph, "w", 0.25, 0.5) == 2
    with pytest.raises(ValueError, match="input SRQ"):
        assert_srq_scales(graph, "w", 0.125, 0.5)
    with pytest.raises(ValueError, match="output SRQ"):
        assert_srq_scales(graph, "w", 0.25, 1.0)


def _qat_graph(initializers: dict) -> dict:
    return {"initializers": initializers}


def test_qat_reference_checks_every_declared_initializer():
    """検査が回る集合（供給計画）と、グラフ宣言の集合が一致する。

    `shared` 宣言（貸し手のバイトを借りる）は容器に実体を持たないので供給計画に入らない。
    宣言だけが `shared` へ書き換われば、その重みは黙って検査対象から外れる —— 本数が減る
    だけで赤にならない形なので、ここが両方向で落とす。
    """
    declared = {
        "model.a.weight": {},
        "model.b.weight": {},
        "const.table": {},
    }
    assert_qat_checked_range(_qat_graph(declared), ["model.a.weight", "model.b.weight"])

    # 宣言にあって供給に無い（焼き漏らし・`shared` への書き換え）。
    with pytest.raises(ValueError, match=r"model\.b\.weight"):
        assert_qat_checked_range(_qat_graph(declared), ["model.a.weight"])

    # 供給にあって宣言に無い（別のグラフの実体を読んでいる）。
    with pytest.raises(ValueError, match=r"model\.c\.weight"):
        assert_qat_checked_range(
            _qat_graph(declared), ["model.a.weight", "model.b.weight", "model.c.weight"]
        )

    # `shared` 宣言は**どちらの集合からも**外れる（借り物は容器に実体を持たない）。
    borrowed = {**declared, "model.b.weight": {"shared": True}}
    assert_qat_checked_range(_qat_graph(borrowed), ["model.a.weight"])
