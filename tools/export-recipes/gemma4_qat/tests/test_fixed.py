"""元整数の保持と SRQ / 共有 / PLE の故障を、実モデルなしで検出する。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from torch import nn

pytest.importorskip("transformers")
from container_series import write_component
from transformers.integrations.gemma_quant import QuantizedEmbedding, QuantizedLinear

from gemma4.distribution import assert_gemma4_ple_assets, gemma4_ple_index
from gemma4.export_product import assert_ple_assets
from gemma4_qat.checkpoint import TraceLinear, fixed_trace_weights, load_qat
from gemma4_qat.ple import build_ple
from gemma4_qat.tests.series_fixture import packed_embedding, qat_container
from karume.container import AssetInput
from karume.dist import DistError
from karume.ple import PLE_INDEX_ASSET, PLE_INDEX_ROLE


def _publish(tmp_path: Path, bits: int, *, bend=None, rows: int = 9):
    """packed PLE を資産として載せた容器を据え、`(代表 path, 索引, probe の参照)` を返す。

    `bend` は索引 JSON だけを 1 箇所曲げる口（資産の block 列はそのまま — 「索引だけ古い
    組み合わせ」を作る席）。
    """
    module = packed_embedding(bits, rows=rows)
    build = build_ple(module, 3, 32, tmp_path)
    assets = dict(build.assets)
    if bend is not None:
        index = json.loads(bytes(assets[PLE_INDEX_ASSET].payload))
        bend(index)
        payload = json.dumps(index, ensure_ascii=False).encode("utf-8")
        assets[PLE_INDEX_ASSET] = AssetInput(PLE_INDEX_ROLE, len(payload), payload)
    container = tmp_path / "model.krm"
    write_component(container, qat_container(vocab=rows, assets=assets))
    build.discard()
    return container, json.loads(bytes(assets[PLE_INDEX_ASSET].payload)), module


class TestPackedPle:
    @pytest.mark.parametrize("bits", [2, 4])
    def test_the_blocks_preserve_bytes_and_the_upstream_probe(self, tmp_path: Path, bits: int):
        container, index, module = _publish(tmp_path, bits)

        parsed = gemma4_ple_index(container, storage=f"i{bits}")
        assert_gemma4_ple_assets(container, parsed)
        assert parsed["values"]["blocks"][-1]["stop"] == index["tokens"]

        with safe_open(str(tmp_path / "ple.probe.safetensors"), framework="pt") as handle:
            ids = handle.get_tensor("tokens")
            actual = handle.get_tensor("per_layer_inputs")
        with torch.inference_mode():
            expected = module(ids.to(torch.int64).unsqueeze(0)).reshape(1, len(ids), 3, 32)
        assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))

        with pytest.raises(DistError, match="storage"):
            gemma4_ple_index(container, storage="i4" if bits == 2 else "i2")

    @pytest.mark.parametrize("fault", ["schema", "storage", "gap", "dim", "scale", "unknown"])
    def test_distribution_rejects_an_inconsistent_index(self, tmp_path: Path, fault: str):
        def bend(index: dict) -> None:
            if fault == "gap":
                index["values"]["blocks"][0]["start"] = 1
                return
            key, value = {
                "schema": ("schema", 1),
                "storage": ("storage", "i2"),
                "dim": ("dim", 31),
                "scale": ("embedScale", 0),
                "unknown": ("strategy", "token-major"),
            }[fault]
            index[key] = value

        container, _, _ = _publish(tmp_path, 4, bend=bend)
        with pytest.raises(DistError):
            gemma4_ple_index(container, storage="i4")

    def test_an_index_that_names_an_absent_asset_is_detected(self, tmp_path: Path):
        """索引だけ差し替えた組み合わせは**形も dtype も合う**まま別 token の行を引く。"""
        container, _, _ = _publish(
            tmp_path,
            2,
            bend=lambda index: index["values"]["blocks"][0].__setitem__(
                "asset", "ple.values.absent"
            ),
        )
        parsed = gemma4_ple_index(container, storage="i2")

        with pytest.raises(DistError, match="容器に無い"):
            assert_gemma4_ple_assets(container, parsed)

    @pytest.mark.parametrize("fault", ["dtype", "shape", "negative", "nan", "embedScale"])
    def test_writer_rejects_invalid_fixed_values(self, tmp_path: Path, fault: str):
        module = packed_embedding(4)
        if fault == "dtype":
            module.embedding_quantized = nn.Parameter(
                module.embedding_quantized.to(torch.int8), requires_grad=False
            )
        elif fault == "shape":
            module.embedding_scale = nn.Parameter(torch.ones(9, 1), requires_grad=False)
        elif fault == "embedScale":
            module.scalar_embed_scale = 0
        else:
            module.embedding_scale[0, 0] = -1 if fault == "negative" else float("nan")
        with pytest.raises(ValueError):
            build_ple(module, 3, 32, tmp_path)


def _probe(tmp_path: Path) -> list[int]:
    with safe_open(str(tmp_path / "ple.probe.safetensors"), framework="pt") as handle:
        return [int(token) for token in handle.get_tensor("tokens")]


class TestPackedPleGate:
    """export が呼ぶ PLE のビット一致門（`assert_ple_assets`）は packed の格納でも同じ 1 本。

    参照は上流 `QuantizedEmbedding` の出力のまま — 門の展開が詰め順か offset を取り違えると
    （この fixture の各バイトは下位と上位で違う値を持つので）ビット一致が崩れる。
    """

    @pytest.mark.parametrize("bits", [2, 4])
    def test_the_written_packed_rows_rebuild_the_upstream_output_bit_for_bit(
        self, tmp_path: Path, bits: int
    ):
        container, index, module = _publish(tmp_path, bits)
        probe = _probe(tmp_path)
        with torch.inference_mode():
            reference = module(torch.tensor([probe], dtype=torch.int64)).reshape(
                1, len(probe), 3, 32
            )

        assert index["storage"] == f"i{bits}"
        assert_ple_assets(container, index, probe, reference)

    @pytest.mark.parametrize(("bits", "flip"), [(2, 0x04), (4, 0x10)])
    def test_one_flipped_field_in_a_probe_row_is_detected(
        self, tmp_path: Path, bits: int, flip: int
    ):
        """容器が持つ行と 1 要素だけ違う上流（i4 は上位 nibble・i2 は 2 番目の 2bit）。"""
        container, index, _ = _publish(tmp_path, bits)
        probe = _probe(tmp_path)
        bent = packed_embedding(bits, rows=9)
        bent.embedding_quantized[probe[0], 0] ^= flip
        with torch.inference_mode():
            reference = bent(torch.tensor([probe], dtype=torch.int64)).reshape(1, len(probe), 3, 32)

        with pytest.raises(AssertionError, match="ビット一致しない"):
            assert_ple_assets(container, index, probe, reference)

    def test_a_layer_shifted_scale_is_detected(self, tmp_path: Path):
        """scale の層ずれは形も型も dtype も合う（`torch.equal` でしか捕まらない）。"""
        container, index, module = _publish(tmp_path, 2)
        probe = _probe(tmp_path)
        with torch.inference_mode():
            reference = module(torch.tensor([probe], dtype=torch.int64)).reshape(
                1, len(probe), 3, 32
            )
        shifted = reference.clone()
        shifted[0, :, 0] = reference[0, :, 1]

        with pytest.raises(AssertionError, match="ビット一致しない"):
            assert_ple_assets(container, index, probe, shifted)

    def test_a_row_length_that_disagrees_with_the_storage_is_rejected(self, tmp_path: Path):
        """行長は索引の `rowBytes` が正本 — 格納と寸法から導く値と割れた索引は読まない。"""
        container, index, module = _publish(tmp_path, 4)
        probe = _probe(tmp_path)
        with torch.inference_mode():
            reference = module(torch.tensor([probe], dtype=torch.int64)).reshape(
                1, len(probe), 3, 32
            )
        bent = {**index, "values": {**index["values"], "rowBytes": 3 * 32}}

        with pytest.raises(AssertionError, match="rowBytes"):
            assert_ple_assets(container, bent, probe, reference)


class TestTraceLinear:
    @pytest.mark.parametrize("scale", [0.0, 0.03125, 3.118110179901123])
    def test_explicit_srq_matches_upstream_when_given_the_same_weight(self, scale: float):
        source = QuantizedLinear(64, 8, num_bits=4).requires_grad_(False)
        source.weight.copy_(
            (torch.arange(source.weight.numel()) % 256).to(torch.uint8).reshape(source.weight.shape)
        )
        source.weight_scale.fill_(0.125)
        source.input_activation_scale.fill_(scale)
        source.output_activation_scale.fill_(scale)
        traced = TraceLinear(source)
        traced.weight = nn.Parameter(source._dequantize_weights().clone(), requires_grad=False)
        value = torch.arange(3 * 64, dtype=torch.float32).reshape(1, 3, 64) / 17 - 4
        assert torch.equal(traced(value).view(torch.int32), source(value).view(torch.int32))

    @pytest.mark.parametrize(
        ("input_scale", "output_scale", "rounds"),
        [(0.0, 0.0, 0), (0.125, 0.0, 1), (0.0, 0.125, 1), (0.125, 0.25, 2)],
    )
    def test_it_traces_only_the_srq_that_actually_rounds(
        self, input_scale: float, output_scale: float, rounds: int
    ):
        """scale=0 は恒等（ADR 0097 追記 2）なので IR に残さない。

        残すと runtime が decode のたびに恒等コピーを dispatch する（上流で 0 なのは lm_head の
        入出力だけなので、その 1 本は語彙全体の読み書きになる）。入力側・出力側は独立に判定する。
        """
        source = QuantizedLinear(64, 8, num_bits=4).requires_grad_(False)
        source.input_activation_scale.fill_(input_scale)
        source.output_activation_scale.fill_(output_scale)
        exported = torch.export.export(TraceLinear(source), (torch.zeros(1, 3, 64),))
        traced = [
            node
            for node in exported.graph.nodes
            if node.target is torch.ops.karume.static_quantize.default
        ]
        assert len(traced) == rounds

    @pytest.mark.parametrize("scale", [-1.0, float("nan"), float("inf")])
    def test_invalid_srq_scale_is_rejected(self, scale: float):
        source = QuantizedLinear(64, 8).requires_grad_(False)
        source.input_activation_scale.fill_(scale)
        with pytest.raises(ValueError, match="scale"):
            TraceLinear(source)


class TestCheckpointAdmission:
    @pytest.mark.parametrize(
        "model,config",
        [
            ("unknown", {}),
            ("e2b", {"quantization_config": {"quant_method": "other"}}),
            (
                "e4b",
                {
                    "quantization_config": {"quant_method": "gemma", "quantize_embeddings": True},
                    "text_config": {"hidden_size": 1536, "num_hidden_layers": 35},
                },
            ),
        ],
    )
    def test_rejects_wrong_family_before_loading_weights(self, tmp_path: Path, model: str, config):
        (tmp_path / "config.json").write_text(json.dumps(config))
        with pytest.raises(ValueError):
            load_qat(tmp_path, model)

    @pytest.mark.parametrize("fault", [None, "packed", "scale"])
    def test_ties_only_identical_fixed_head_and_embedding(self, fault):
        wrapper = nn.Module()
        wrapper.model = nn.Module()
        wrapper.model.model = nn.Module()
        embedding = QuantizedEmbedding(8, 64, torch.float32, num_bits=2).requires_grad_(False)
        head = QuantizedLinear(64, 8, num_bits=2).requires_grad_(False)
        embedding.embedding_quantized.zero_()
        embedding.embedding_scale.fill_(0.5)
        head.weight.zero_()
        head.weight_scale.fill_(0.5)
        head.input_activation_scale.zero_()
        head.output_activation_scale.zero_()
        wrapper.model.model.embed_tokens = embedding
        wrapper.model.lm_head = head
        if fault:
            (head.weight if fault == "packed" else head.weight_scale)[0, 0] = 1
            with pytest.raises(ValueError, match="一致しない"):
                fixed_trace_weights(wrapper)
        else:
            saved = fixed_trace_weights(wrapper)
            assert saved["model.lm_head.weight"].packed.data_ptr() == head.weight.data_ptr()
            assert wrapper.model.lm_head.weight is wrapper.model.model.embed_tokens.weight


class TestThePleSpill:
    """PLE の実体は**作業席の一時ファイル**へ落ち、据え替えの前に消える。

    上流の packed 実体を掴む読み口にすると、`trace_qat` と書き出しの間ずっと常駐する
    （製品系列の `gemma4.export_product._spill_tables` と同じ規律）。
    """

    def test_the_payload_lands_in_the_working_seat(self, tmp_path: Path) -> None:
        build = build_ple(packed_embedding(4, rows=9), 3, 32, tmp_path)

        assert len(build.spills) == 2
        assert all(path.stat().st_size > 0 for path in build.spills)

    def test_discard_removes_the_temporary_files(self, tmp_path: Path) -> None:
        build = build_ple(packed_embedding(4, rows=9), 3, 32, tmp_path)

        build.discard()

        assert [path for path in build.spills if path.exists()] == []

    def test_the_assets_still_read_after_the_module_is_released(self, tmp_path: Path) -> None:
        """実体はファイル側にあるので、上流モジュールを手放しても payload が引ける。"""
        module = packed_embedding(4, rows=9)
        build = build_ple(module, 3, 32, tmp_path)
        index = json.loads(bytes(build.assets[PLE_INDEX_ASSET].payload))
        name = str(index["values"]["blocks"][0]["asset"])
        del module

        payload = build.assets[name].payload

        assert len(bytes(payload())) == build.assets[name].length
