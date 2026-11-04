"""Storage conversion of the distribution. If the declaration (graph) and the stored tensors fall
out of step, not a single byte is written.

There are two observation points: **the committed declaration** (the graph `stored_model` returns)
and **the installed container** (the block payloads and the codecs of the binding table). The
physical format of the container itself belongs to `test_container.py`, and the merge rules that
need declared shapes to `test_verify_container.py`.
"""

from __future__ import annotations

import json
import re
import weakref
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from karume.container import (
    BLOCK_MAX_BYTES,
    DEFAULT_PART_BYTES,
    ContainerFormatError,
    Provenance,
    codec_entry,
    container_parts,
    ir_v2_document,
    weight_channel_axes,
)
from karume.emit import (
    INT4_OFFSET,
    EmitError,
    eligible_compressed_initializers,
    i4_eligible_initializers,
    pack_int4,
    storage_breakdown,
    stored_model,
    unpack_int4,
)
from karume.ir import (
    IrGraph,
    IrInitializer,
    IrInput,
    IrNode,
    IrShared,
    IrStorage,
    IrValue,
)
from karume.ops import OpContractError
from karume.publish import publish_container
from karume.quantize import (
    channel_scale,
    dequantize_int4,
    group_scale,
    quantize_to_int4,
    quantize_to_int8,
)
from karume.verify import ir_graph_from_container, verify_container

#: Provenance of the synthetic model (no distribution without `--license` — container-v1 §12).
PROVENANCE = Provenance(license="mit")

#: Graph name of the synthetic model (one graph per container — ADR 0109 decision 2).
GRAPH_NAME = "emit"


def sample_graph() -> tuple[IrGraph, dict[str, torch.Tensor]]:
    graph = IrGraph(
        symbols=["T"],
        inputs=[IrInput(name="x", dtype="f32", shape=["T", 4])],
        outputs=["y"],
        initializers={"w": IrInitializer(tensor="enc.w", storage=IrStorage(dtype="f32"))},
        values={
            "w": IrValue(dtype="f32", shape=[4]),
            "y": IrValue(dtype="f32", shape=["T", 4]),
        },
        nodes=[IrNode(op="add", ins=["x", "w"], outs=["y"], attrs={})],
    )
    return graph, {"enc.w": torch.ones(4)}


class TestRoundTrip:
    def test_the_container_supplies_the_tensor_key_and_carries_the_provenance(self, tmp_path):
        """Opening the written container and taking it through the merge supplies the one graph's
        initializer by tensor key, and the model descriptor carries the provenance."""
        graph, tensors = sample_graph()

        path = write_component(tmp_path / "model.krm", graph, tensors)

        verified = bound(path)
        assert set(verified.graphs) == {GRAPH_NAME}
        # In v2 the initializer name is **the tensor key** (container-v1 §13.1).
        assert set(verified.graphs[GRAPH_NAME].supplies) == {"enc.w"}
        assert verified.read.model is not None
        assert verified.read.model.provenance == PROVENANCE

    def test_the_graph_read_back_from_the_file_is_the_written_declaration(self, tmp_path):
        """The declaration rebuilt from the two documents of a fully verified container = the
        declaration the writer committed.

        The container's documents are IR v2 (initializer names are tensor keys — container-v1
        §13.1), so the graphs are compared as v2 documents and the storage per tensor key through
        the initializer declarations.
        """
        graph, tensors = sample_graph()

        path = write_component(tmp_path / "model.krm", graph, tensors)

        written = written_graph(path)
        read_back = ir_graph_from_container(bound(path).read, GRAPH_NAME)
        assert written.to_dict() == graph.to_dict()
        assert ir_v2_document(read_back) == ir_v2_document(written)
        assert {init.tensor: init.to_dict() for init in read_back.initializers.values()} == {
            init.tensor: init.to_dict() for init in written.initializers.values()
        }


class TestDeclarationAgreement:
    def test_missing_tensor_fails_loudly(self, tmp_path):
        graph, _ = sample_graph()

        with pytest.raises(EmitError, match="missing"):
            write_model(tmp_path / "model.krm", graph, {})

    def test_unreferenced_tensor_fails_loudly(self, tmp_path):
        graph, tensors = sample_graph()

        with pytest.raises(EmitError, match="extra"):
            write_model(tmp_path / "model.krm", graph, {**tensors, "stray": torch.zeros(2)})


def rounded(*shape: int) -> torch.Tensor:
    """Random values rounded to f16-representable values (= fake-quantized weights)."""
    return torch.randn(*shape).to(torch.float16).to(torch.float32)


def weight_graph(
    *, share_weight: bool = False, unused: bool = False, output_weight: bool = False
) -> tuple[IrGraph, dict[str, torch.Tensor]]:
    """A graph with linear (even weight element count) + embedding (**odd** weight element count).

    `share_weight` also consumes the linear weight in an elementwise op (should be ineligible),
    `unused` adds one initializer nobody consumes (ineligible as well), and `output_weight` makes
    the linear weight a graph output as is (ineligible as well — the runtime's readback assumes
    semantic f32).
    """
    nodes = [
        IrNode(op="linear", ins=["x", "w", "b"], outs=["h"], attrs={}),
        IrNode(op="embedding", ins=["emb", "idx"], outs=["e"], attrs={"padding_idx": -1}),
    ]
    initializers = {
        "w": IrInitializer(tensor="enc.w", storage=IrStorage(dtype="f32")),
        "b": IrInitializer(tensor="enc.b", storage=IrStorage(dtype="f32")),
        "emb": IrInitializer(tensor="enc.emb", storage=IrStorage(dtype="f32")),
    }
    values = {
        "w": IrValue(dtype="f32", shape=[3, 4]),
        "b": IrValue(dtype="f32", shape=[3]),
        "emb": IrValue(dtype="f32", shape=[3, 5]),
        "h": IrValue(dtype="f32", shape=["T", 3]),
        "e": IrValue(dtype="f32", shape=["T", 5]),
    }
    tensors = {"enc.w": rounded(3, 4), "enc.b": rounded(3), "enc.emb": rounded(3, 5)}
    outputs = ["h", "e"]
    if share_weight:
        # The same tensor serves as the linear weight and as a mul operand (the latter reads it
        # as f32).
        nodes.append(IrNode(op="mul", ins=["w", "w"], outs=["w2"], attrs={}))
        values["w2"] = IrValue(dtype="f32", shape=[3, 4])
        outputs.append("w2")
    if unused:
        initializers["dead"] = IrInitializer(tensor="enc.dead", storage=IrStorage(dtype="f32"))
        values["dead"] = IrValue(dtype="f32", shape=[6])
        tensors["enc.dead"] = rounded(6)
    if output_weight:
        # IR allows an initializer name as a graph output as is (the value is a constant that does
        # not depend on execution).
        outputs.append("w")
    graph = IrGraph(
        symbols=["T"],
        inputs=[
            IrInput(name="x", dtype="f32", shape=["T", 4]),
            IrInput(name="idx", dtype="i32", shape=["T"]),
        ],
        outputs=outputs,
        initializers=initializers,
        values=values,
        nodes=nodes,
    )
    return graph, tensors


def compressed_view(
    graph: IrGraph, dtypes: dict[str, str], *, group_size: int | None = None
) -> IrGraph:
    """The declaration view `write_model` should commit to the container (the caller's graph is
    not rewritten).

    The key rule of companion scales (`karume.scale.<tensor key>`) is spelled out on the test
    side without calling the implementation — taking it from the implementation would make the
    comparison tautological. `group_size` is a declaration field of i4 (ADR 0069 decision 2) and
    does not appear in i8 / f16 declarations.
    """
    initializers = dict(graph.initializers)
    for name, dtype in dtypes.items():
        key = initializers[name].tensor
        scale = f"karume.scale.{key}" if dtype in ("i8", "i4") else None
        initializers[name] = IrInitializer(
            tensor=key,
            storage=IrStorage(
                dtype=dtype, scale=scale, group_size=group_size if dtype == "i4" else None
            ),
        )
    return replace(graph, initializers=initializers)


class TestEligibility:
    """The eligibility check mirrors `packages/runtime/src/runtime/plan.ts` —
    if they drift, the VRAM saving silently disappears.
    """

    def test_only_weight_slots_are_eligible(self):
        graph, _ = weight_graph()

        assert eligible_compressed_initializers(graph) == {"w", "emb"}

    def test_a_weight_consumed_elsewhere_is_disqualified(self):
        """Any consumption outside a weight slot makes it ineligible (that consumer reads f32)."""
        graph, _ = weight_graph(share_weight=True)

        assert eligible_compressed_initializers(graph) == {"emb"}

    def test_an_unconsumed_initializer_is_disqualified(self):
        """Counting bytes no execution uses as "GPU-resident compression" skews the diagnostic."""
        graph, _ = weight_graph(unused=True)

        assert "dead" not in eligible_compressed_initializers(graph)

    def test_an_initializer_in_the_graph_outputs_is_disqualified(self):
        """readback copies from the weight buffer assuming semantic f32 (4 bytes / element).

        Kept resident compressed, the copy would overrun the real buffer and fail validation, or
        at tiny sizes silently return reinterpreted bits.
        """
        graph, _ = weight_graph(output_weight=True)

        # Only the name in the outputs is disqualified (the embedding table of the same graph stays
        # eligible)
        assert eligible_compressed_initializers(graph) == {"emb"}


class TestF16Storage:
    def test_eligible_weights_are_stored_as_f16_and_bias_stays_f32(self, tmp_path):
        graph, tensors = weight_graph()

        path = write_component(tmp_path / "model.krm", graph, tensors, weight_dtype="f16")

        # The declaration is observed on the file side (`write_model` does not rewrite the caller's
        # graph).
        declared = written_graph(path).initializers
        assert declared["w"].storage.dtype == "f16"
        assert declared["emb"].storage.dtype == "f16"
        # MUST: bias is always f32 (the root-cause fix of the prototype's f16 demotion bug —
        # ADR 0006).
        assert declared["b"].storage.dtype == "f32"
        assert stored_layout(path, "enc.w") == "f16"
        assert stored_layout(path, "enc.b") == "f32"

    def test_a_weight_in_the_graph_outputs_is_not_stored_compressed(self, tmp_path):
        """A weight among the graph outputs is written as f32 (the counterpart of the runtime's
        eligibility check).

        If the mirror drifted, "the exporter writes f16 and the runtime reads it back as f32"
        would disagree, and a model unreadable from the moment it loads would pass as a
        distribution.
        """
        graph, tensors = weight_graph(output_weight=True)

        path = write_component(tmp_path / "model.krm", graph, tensors, weight_dtype="f16")

        declared = written_graph(path).initializers
        assert declared["w"].storage.dtype == "f32"
        assert declared["emb"].storage.dtype == "f16"
        assert stored_layout(path, "enc.w") == "f32"
        assert stored_layout(path, "enc.emb") == "f16"
        assert written_graph(path).to_dict() == compressed_view(graph, {"emb": "f16"}).to_dict()

    def test_the_f16_file_passes_the_full_verification(self, tmp_path):
        graph, tensors = weight_graph()

        path = write_component(tmp_path / "model.krm", graph, tensors, weight_dtype="f16")

        expected = compressed_view(graph, {"w": "f16", "emb": "f16"})
        assert written_graph(path).to_dict() == expected.to_dict()

    def test_the_stored_values_match_the_rounded_weights_bit_for_bit(self, tmp_path):
        graph, tensors = weight_graph()
        expected = tensors["enc.w"].clone()

        path = write_component(tmp_path / "model.krm", graph, tensors, weight_dtype="f16")

        stored = stored_tensor(path, "enc.w", torch.float16, expected.shape).to(torch.float32)
        assert torch.equal(stored, expected)

    def test_an_unrounded_eligible_weight_fails_loudly(self, tmp_path):
        """A forgotten rounding / a wrong rounding order is not let through silently (ADR 0006).

        Letting it through gives "golden on the original values, execution on rounded values",
        and the E2E difference becomes a mix of quantization error and implementation error — it
        only pushes towards looser tolerances, so the tests stay green while losing their power.
        """
        graph, tensors = weight_graph()
        tensors["enc.w"] = torch.full((3, 4), 1.0 / 3.0)

        with pytest.raises(EmitError, match="fake-quant"):
            write_model(tmp_path / "model.krm", graph, tensors, weight_dtype="f16")

    def test_a_graph_without_eligible_weights_fails_loudly(self, tmp_path):
        """Does not silence "f16 requested but 0 MB eligible" (the writer side of ADR 0006's
        standing diagnostic)."""
        graph, tensors = sample_graph()

        with pytest.raises(EmitError, match="Not a single compressed storage was planned"):
            write_model(tmp_path / "model.krm", graph, tensors, weight_dtype="f16")

    def test_an_unknown_weight_dtype_fails_loudly(self, tmp_path):
        graph, tensors = weight_graph()

        # i2 is in the IR vocabulary but has no writer (only fixed packed QAT series carry it —
        # ADR 0097).
        with pytest.raises(EmitError, match="cannot be written"):
            write_model(tmp_path / "model.krm", graph, tensors, weight_dtype="i2")

    def test_the_breakdown_counts_eligible_and_plain_bytes(self, tmp_path):
        graph, tensors = weight_graph()

        path = write_component(tmp_path / "model.krm", graph, tensors, weight_dtype="f16")
        # The breakdown counts from **the declaration of the written file** (the caller's graph
        # holds no compressed declarations).
        breakdown = storage_breakdown(written_graph(path))

        assert breakdown.compressed_tensors == 2
        assert breakdown.compressed_bytes == (3 * 4 + 3 * 5) * 2
        assert breakdown.plain_tensors == 1
        assert breakdown.plain_bytes == 3 * 4

    def test_the_breakdown_counts_a_declared_bf16_initializer(self):
        """The breakdown counts from the declaration alone — it can count bf16 in a hand-built
        graph that never went through the writer.

        The breakdown is a layer that only counts bytes, so it must not reach a state where it
        **cannot count a readable graph** (a KeyError on a dtype in the vocabulary). The breakdown
        of bf16 written by the writer is checked by `TestBf16Storage`.
        """
        graph, _ = weight_graph()
        graph.initializers["w"] = IrInitializer(tensor="enc.w", storage=IrStorage(dtype="bf16"))

        breakdown = storage_breakdown(graph)

        assert breakdown.compressed_tensors == 1
        assert breakdown.compressed_bytes == 3 * 4 * 2
        assert breakdown.plain_tensors == 2
        assert breakdown.plain_bytes == (3 + 3 * 5) * 4
        assert breakdown.scale_bytes == 0

    def test_the_breakdown_does_not_count_a_shared_initializer(self):
        """A shared declaration (ADR 0096 stage 2) only borrows the lender's bytes and writes not a
        single byte into this container.

        The breakdown counts "the bytes written into this container" — counting the lender's
        weights would skew the standing diagnostic (ADR 0006).
        """
        graph = IrGraph(
            symbols=[],
            inputs=[IrInput(name="x", dtype="f32", shape=[2, 8])],
            outputs=["y", "z"],
            initializers={
                "w": IrInitializer(tensor="own.w", storage=IrStorage(dtype="f32")),
                "lent": IrInitializer(
                    shared=IrShared(tensor="lender.w"), storage=IrStorage(dtype="f16")
                ),
            },
            values={
                "w": IrValue(dtype="f32", shape=[4, 8]),
                "lent": IrValue(dtype="f32", shape=[16, 8]),
                "y": IrValue(dtype="f32", shape=[2, 4]),
                "z": IrValue(dtype="f32", shape=[2, 16]),
            },
            nodes=[
                IrNode(op="linear", ins=["x", "w"], outs=["y"], attrs={}),
                IrNode(op="linear", ins=["x", "lent"], outs=["z"], attrs={}),
            ],
        )

        breakdown = storage_breakdown(graph)

        assert breakdown.plain_tensors == 1
        assert breakdown.plain_bytes == 4 * 8 * 4
        assert breakdown.compressed_tensors == 0
        assert breakdown.compressed_bytes == 0


def bf16_weight_graph(**kwargs) -> tuple[IrGraph, dict[str, torch.Tensor]]:
    """`weight_graph` fake-quantized to bf16.

    The values of `weight_graph` are rounded to f16 and cannot be represented in bf16.
    """
    graph, tensors = weight_graph(**kwargs)
    return graph, {
        key: value.to(torch.bfloat16).to(torch.float32) for key, value in tensors.items()
    }


class TestBf16Storage:
    """bf16 storage (ADR 0131) — the eligibility rules and the writing skeleton are those of f16."""

    def test_eligible_weights_are_stored_as_bf16_and_bias_stays_f32(self, tmp_path):
        graph, tensors = bf16_weight_graph()

        path = write_component(tmp_path / "model.krm", graph, tensors, weight_dtype="bf16")

        declared = written_graph(path).initializers
        assert declared["w"].storage.dtype == "bf16"
        assert declared["emb"].storage.dtype == "bf16"
        assert declared["b"].storage.dtype == "f32"
        assert stored_layout(path, "enc.w") == "bf16"
        assert stored_layout(path, "enc.emb") == "bf16"
        assert stored_layout(path, "enc.b") == "f32"
        expected = compressed_view(graph, {"w": "bf16", "emb": "bf16"})
        assert written_graph(path).to_dict() == expected.to_dict()

    def test_the_eligibility_is_the_same_as_f16(self, tmp_path):
        """The eligible set is the same as for f16 (no narrowing like i4's).

        A weight among the graph outputs is written as f32, as with f16.
        """
        graph, tensors = bf16_weight_graph(output_weight=True)

        path = write_component(tmp_path / "model.krm", graph, tensors, weight_dtype="bf16")

        assert written_graph(path).to_dict() == compressed_view(graph, {"emb": "bf16"}).to_dict()

    @pytest.mark.parametrize("key", ["enc.w", "enc.emb"])
    def test_the_payload_is_the_upper_half_of_each_f32(self, tmp_path, key):
        """payload = little-endian bf16, 2 bytes / element (including the odd-sized `enc.emb`).

        The reference is the upper 16 bits cut directly from the f32 bits — using
        `to(bfloat16)` as the reference would only retrace the writer's conversion.
        """
        graph, tensors = bf16_weight_graph()
        bits = tensors[key].reshape(-1).view(torch.int32)
        upper = (bits >> 16).to(torch.int16)

        path = write_component(tmp_path / "model.krm", graph, tensors, weight_dtype="bf16")

        assert stored_bytes(path, key) == upper.numpy().tobytes()
        converted = tensors[key].to(torch.bfloat16).view(torch.int16)
        assert stored_bytes(path, key) == converted.numpy().tobytes()

    def test_the_stored_values_read_back_to_the_rounded_weights_bit_for_bit(self, tmp_path):
        graph, tensors = bf16_weight_graph()
        expected = tensors["enc.emb"].clone()

        path = write_component(tmp_path / "model.krm", graph, tensors, weight_dtype="bf16")

        stored = stored_tensor(path, "enc.emb", torch.bfloat16, expected.shape).to(torch.float32)
        assert torch.equal(stored.view(torch.int32), expected.view(torch.int32))

    def test_an_f16_exact_but_not_bf16_exact_weight_fails_loudly(self, tmp_path):
        """Rounded for f16 is not rounded for bf16 — rounding for the wrong storage is not let
        through silently.

        `1 + 2**-9` is exact in f16 (10-bit mantissa) and not representable in bf16 (7-bit
        mantissa).
        """
        graph, tensors = bf16_weight_graph()
        tensors["enc.w"] = torch.full((3, 4), 1 + 2**-9)

        with pytest.raises(EmitError, match=r"contains values bf16 cannot represent.*fake-quant"):
            write_model(tmp_path / "model.krm", graph, tensors, weight_dtype="bf16")

    def test_the_breakdown_counts_the_written_bf16_bytes(self, tmp_path):
        graph, tensors = bf16_weight_graph()

        path = write_component(tmp_path / "model.krm", graph, tensors, weight_dtype="bf16")
        breakdown = storage_breakdown(written_graph(path))

        assert breakdown.compressed_tensors == 2
        assert breakdown.compressed_bytes == (3 * 4 + 3 * 5) * 2
        assert breakdown.plain_tensors == 1
        assert breakdown.plain_bytes == 3 * 4
        assert breakdown.scale_bytes == 0


def int8_weight_graph() -> tuple[IrGraph, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """`weight_graph` fake-quantized to i8 (graph / tensors / scale ledger).

    `enc.w` is the linear weight (axis 0, `[3,4]`), `enc.emb` the embedding table (axis 0,
    `[3,5]`). Axis 1 of conv_transpose1d is taken separately by `TestI8ChannelAxes`.
    """
    graph, tensors = weight_graph()
    scales: dict[str, torch.Tensor] = {}
    for key in ("enc.w", "enc.emb"):
        scale = channel_scale(tensors[key], 0)
        tensors[key] = quantize_to_int8(tensors[key], scale).to(torch.float32) * scale
        scales[key] = scale
    return graph, tensors, scales


class TestI8ChannelAxes:
    """The per-channel axis comes from **the consuming op** (the weight shape alone does not
    decide it)."""

    def test_the_axis_comes_from_the_consuming_op(self):
        graph, _ = weight_graph()

        assert weight_channel_axes(graph) == {"w": 0, "emb": 0}

    def test_conv_transpose1d_uses_the_second_axis(self):
        graph, _ = weight_graph()
        graph.nodes[0] = IrNode(op="conv_transpose1d", ins=["x", "w", "b"], outs=["h"], attrs={})

        assert weight_channel_axes(graph)["w"] == 1

    def test_conflicting_axes_fail_loudly(self):
        """A weight consumed by ops with different axes cannot be expressed with one scale."""
        graph, _ = weight_graph()
        graph.nodes.append(
            IrNode(op="conv_transpose1d", ins=["x", "w", "b"], outs=["h2"], attrs={})
        )
        graph.values["h2"] = IrValue(dtype="f32", shape=["T", 3])
        graph.outputs.append("h2")

        # The binding-table mapping lives on the torch-free side (`karume.container`), so the
        # exception is the container's vocabulary too.
        with pytest.raises(ContainerFormatError, match="per-channel axis differs"):
            weight_channel_axes(graph)


class TestI8Storage:
    def test_eligible_weights_are_stored_as_i8_with_a_scale_and_bias_stays_f32(self, tmp_path):
        graph, tensors, scales = int8_weight_graph()

        path = write_component(
            tmp_path / "model.krm", graph, tensors, weight_dtype="i8", weight_scales=scales
        )

        declared = written_graph(path).initializers
        assert declared["w"].storage.dtype == "i8"
        assert declared["w"].storage.scale == "karume.scale.enc.w"
        assert declared["emb"].storage.dtype == "i8"
        # MUST: bias is always f32 (the root-cause fix of the prototype's demotion bug —
        # ADR 0006).
        assert declared["b"].storage.dtype == "f32"
        assert stored_layout(path, "enc.w") == "i8"
        assert stored_layout(path, "enc.b") == "f32"
        # The scale is fixed to f32 (container-v1 §6.1), and per-channel it has exactly one entry
        # per row.
        assert len(stored_bytes(path, "karume.scale.enc.w")) == 3 * 4

    def test_the_i8_file_passes_the_full_verification(self, tmp_path):
        graph, tensors, scales = int8_weight_graph()

        path = write_component(
            tmp_path / "model.krm", graph, tensors, weight_dtype="i8", weight_scales=scales
        )

        expected = compressed_view(graph, {"w": "i8", "emb": "i8"})
        assert written_graph(path).to_dict() == expected.to_dict()

    def test_the_stored_values_reconstruct_the_weights_bit_for_bit(self, tmp_path):
        """`q8·scale` **matches** the fake-quantized weight **bit for bit** (the very meaning of
        the storage)."""
        graph, tensors, scales = int8_weight_graph()
        expected = tensors["enc.w"].clone()

        path = write_component(
            tmp_path / "model.krm", graph, tensors, weight_dtype="i8", weight_scales=scales
        )

        stored = stored_tensor(path, "enc.w", torch.int8, expected.shape).to(torch.float32)
        scale = stored_tensor(path, "karume.scale.enc.w", torch.float32, (expected.shape[0], 1))
        assert torch.equal(stored * scale, expected)

    def test_an_unrounded_eligible_weight_fails_loudly(self, tmp_path):
        """A forgotten rounding / a wrong order is not let through silently (ADR 0006)."""
        graph, tensors, scales = int8_weight_graph()
        tensors["enc.w"] = torch.full((3, 4), 1.0 / 3.0)

        with pytest.raises(EmitError, match="not bit-identical"):
            write_model(
                tmp_path / "model.krm",
                graph,
                tensors,
                weight_dtype="i8",
                weight_scales=scales,
            )

    def test_a_scale_that_is_not_the_one_fake_quant_used_fails_loudly(self, tmp_path):
        """Writing with a scale **other than** the one fake-quant used fails the inverse-conversion
        gate.

        The general form of failing ADR 0019's "pass the scale on as is, without recomputing it".
        A difference of even 1 ulp makes no stored value correspond to golden, and the difference
        shows only in values, never as an exception, so writing is stopped here.
        """
        graph, tensors, scales = int8_weight_graph()
        drifted = {key: value * 1.0000002 for key, value in scales.items()}

        with pytest.raises(EmitError, match="not bit-identical"):
            write_model(
                tmp_path / "model.krm",
                graph,
                tensors,
                weight_dtype="i8",
                weight_scales=drifted,
            )

    def test_recomputing_the_scale_from_a_quantized_weight_is_a_fixed_point(self):
        """NOTE (observed, against expectation): **recomputing from a correctly fake-quantized
        weight gives the same value**.

        `q` is clamped to ±127, so the element of largest magnitude always lands on `q = ±127`,
        and `amax(|q·s|)/127 = fl(fl(127·s)/127) = s` (an f32 fixed point — zero counterexamples
        in 8.9e7 random samples). So ADR 0019's "no recomputation" is a rule **the
        inverse-conversion gate cannot detect**; the reason to keep it is that the scale silently
        changes when "the weight at emit time is not the effective weight / the axis differs /
        the formula differs". This fixed-point property is the same property as idempotency
        (test_quantize.py), so it is pinned here too (if it breaks, both break at once).
        """
        _, tensors, scales = int8_weight_graph()

        for key, scale in scales.items():
            assert torch.equal(channel_scale(tensors[key], 0), scale), key

    def test_a_scale_that_is_not_f32_fails_loudly(self, tmp_path):
        """The companion scale is fixed to F32 (ADR 0019). The writer could serialize F16 as is.

        The inverse-conversion equality check would pass for "fake-quantized with the same f16
        scale", so only the dtype check at the planning step detects it (the later
        `verify_container` runs after writing).
        """
        graph, tensors, scales = int8_weight_graph()
        half = {key: value.to(torch.float16) for key, value in scales.items()}
        for key, scale in half.items():
            tensors[key] = quantize_to_int8(tensors[key], scale).to(torch.float32) * scale

        with pytest.raises(EmitError, match="scale dtype"):
            write_model(
                tmp_path / "model.krm",
                graph,
                tensors,
                weight_dtype="i8",
                weight_scales=half,
            )

    def test_a_missing_scale_fails_loudly(self, tmp_path):
        """Eligible but without a scale = a weight fake-quant never reached (not dropped to f32)."""
        graph, tensors, scales = int8_weight_graph()
        del scales["enc.emb"]

        with pytest.raises(EmitError, match="has no per-channel scale"):
            write_model(
                tmp_path / "model.krm",
                graph,
                tensors,
                weight_dtype="i8",
                weight_scales=scales,
            )

    def test_a_scale_with_the_wrong_shape_fails_loudly(self, tmp_path):
        """A shape that merely broadcasts (`[1,4]`) disagrees with how the kernel reads it — a
        silent wrong value."""
        graph, tensors, scales = int8_weight_graph()
        scales["enc.w"] = torch.ones(1, 4)

        with pytest.raises(EmitError, match="keepdim shape"):
            write_model(
                tmp_path / "model.krm",
                graph,
                tensors,
                weight_dtype="i8",
                weight_scales=scales,
            )

    def test_a_scale_key_colliding_with_a_real_tensor_fails_loudly(self, tmp_path):
        """A scale key colliding with a real tensor would "read another weight as the scale"."""
        graph, tensors, scales = int8_weight_graph()
        graph.initializers["b"] = IrInitializer(
            tensor="karume.scale.enc.w", storage=IrStorage(dtype="f32")
        )
        tensors["karume.scale.enc.w"] = tensors.pop("enc.b")

        with pytest.raises(EmitError, match="collides"):
            write_model(
                tmp_path / "model.krm",
                graph,
                tensors,
                weight_dtype="i8",
                weight_scales=scales,
            )

    def test_a_graph_without_eligible_weights_fails_loudly(self, tmp_path):
        graph, tensors = sample_graph()

        with pytest.raises(EmitError, match="Not a single compressed storage was planned"):
            write_model(tmp_path / "model.krm", graph, tensors, weight_dtype="i8")

    def test_the_breakdown_counts_i8_bytes_and_the_scale_overhead(self, tmp_path):
        graph, tensors, scales = int8_weight_graph()

        path = write_component(
            tmp_path / "model.krm", graph, tensors, weight_dtype="i8", weight_scales=scales
        )
        # The breakdown counts from **the declaration of the written file** (the caller's graph
        # holds no compressed declarations).
        breakdown = storage_breakdown(written_graph(path))

        assert breakdown.compressed_tensors == 2
        assert breakdown.compressed_bytes == 3 * 4 + 3 * 5  # i8 = 1 byte/element
        # The scale is output channels × 4 bytes (axis 0 has length 3 for both w and emb).
        assert breakdown.scale_bytes == (3 + 3) * 4
        assert breakdown.plain_tensors == 1
        assert breakdown.plain_bytes == 3 * 4


#: Installed container → the declaration committed in that run (the observation point of **the
#: declaration after writing**).

#: The container holds only IR v2 documents, and `storage` has moved into the binding table
#: (container-v1 §13.1), so the writer's return value is the source of truth for "which
#: initializer was written with which storage".
_COMMITTED: dict[Path, IrGraph] = {}


def write_model(path, graph: IrGraph, tensors, **kwargs) -> list[Path]:
    """Storage conversion → container (the single path `stored_model` → `publish_container`).

    The tests keep this entry point because the subjects of the storage rules (eligibility,
    scale, fail loudly) span two steps, "planning" and "writing" — the conversion is deferred,
    so checks that read real data run only when the payload is pulled (= when the container is
    written).
    """
    block_bytes = kwargs.pop("_block_bytes", BLOCK_MAX_BYTES)
    part_bytes = kwargs.pop("_part_bytes", DEFAULT_PART_BYTES)
    stored = stored_model(graph, tensors, **kwargs)
    final = Path(path).with_suffix(".krm")
    result = publish_container(
        final,
        stored.graph,
        stored.tensors,
        stored.bindings,
        graph_name=GRAPH_NAME,
        provenance=PROVENANCE,
        part_bytes=part_bytes,
        block_bytes=block_bytes,
    )
    _COMMITTED[final] = stored.graph
    return list(result.parts)


def write_component(path, *args, **kwargs) -> Path:
    """Write and return the **representative path** (the real files are numbered parts —
    container-v1 §8)."""
    write_model(path, *args, **kwargs)
    return Path(path).with_suffix(".krm")


def written_graph(path) -> IrGraph:
    """The declaration committed in that run (a view holding the `storage` actually written)."""
    return _COMMITTED[Path(path).with_suffix(".krm")]


def bound(path):
    """The installed container opened and taken through the merge (the binding-table observation
    point)."""
    return verify_container(container_parts(Path(path).with_suffix(".krm")), blocks=True)


def stored_layout(path, key: str) -> str:
    """The **decode path** of one tensor (the codec ledger's `layout` — it replaces the old
    safetensors dtype)."""
    supply = bound(path).graphs[GRAPH_NAME].supplies[key]
    return codec_entry(supply.encoding.codec).layout


def stored_bytes(path, key: str) -> bytes:
    """The payload of one tensor (piece lists joined into the parent; no trailing padding).

    Passing `karume.scale.<key>` returns the payload of the companion scale.
    """
    verified = bound(path)
    supplies = verified.graphs[GRAPH_NAME].supplies
    weight = key.removeprefix("karume.scale.")
    supply = supplies[weight]
    if key != weight:
        assert supply.scale is not None, f"'{weight}' has no companion scale"
        return verified.read.block(supply.scale.id)[: supply.scale.payload_bytes]
    return b"".join(verified.read.block(block.id)[: block.payload_bytes] for block in supply.blocks)


def stored_payload(path, name: str) -> torch.Tensor:
    """Take one tensor's raw bytes as a uint8 tensor (takes the representative path)."""
    return torch.frombuffer(bytearray(stored_bytes(path, name)), dtype=torch.uint8)


def stored_tensor(path, key: str, dtype: torch.dtype, shape=None) -> torch.Tensor:
    """Read the payload back as `dtype` (reshaped to `shape` when given)."""
    flat = torch.frombuffer(bytearray(stored_bytes(path, key)), dtype=dtype)
    return flat if shape is None else flat.reshape(shape)


def asymmetric_nibbles(count: int) -> torch.Tensor:
    """An asymmetric pattern `[1,−2,3,−4,…]` whose neighbours all differ (range ±7).

    The detector of the pack order (ADR 0069 decision 4 ①) — with a symmetric pattern the round
    trip would pass even with the upper and lower nibbles swapped.
    """
    index = torch.arange(count)
    return (((index % 7) + 1) * torch.where(index % 2 == 0, 1, -1)).to(torch.int8)


class TestPackedFourBitOrder:
    """The source of truth of the pack order (ADR 0069 decision 4) — "element 2i is the low nibble
    / 2i+1 the high nibble".

    A free parameter on which upstream projects also differ; a mistake only ever gives **a silent
    wrong value with matching shape and type**. A round trip (pack → unpack) alone cannot detect
    pack and unpack being wrong in the same direction, so it is paired with a pin of hand-written
    byte values.
    """

    def test_the_even_element_goes_to_the_low_nibble(self):
        """`q = [1, 2]` → `u = [9, 10]` → one byte `0xA9` (packed the other way round: `0x9A`)."""
        packed = pack_int4(torch.tensor([1, 2], dtype=torch.int8))

        assert packed.dtype is torch.uint8
        assert packed.tolist() == [0xA9]

    def test_the_stored_nibbles_are_offset_by_eight_and_never_zero(self):
        """The stored value is `u = q + 8` (range [1,15], 15 levels with 0 unused — ADR 0069
        decision 3)."""
        packed = pack_int4(torch.tensor([-7, 7, 0, -7], dtype=torch.int8))

        assert INT4_OFFSET == 8
        assert packed.tolist() == [0xF1, 0x18]
        nibbles = [byte & 0x0F for byte in packed.tolist()] + [
            byte >> 4 for byte in packed.tolist()
        ]
        assert 0 not in nibbles

    @pytest.mark.parametrize(
        "shape",
        [(1, 16), (3, 16), (2, 48), (5, 20), (7, 4), (2, 3, 6)],
        ids=["one-row", "two-words", "six-words", "row-straddles", "short-rows", "rank3"],
    )
    def test_the_round_trip_preserves_every_position(self, shape):
        """Every position matches exactly, including shapes whose row length does not meet a word
        boundary (4 bytes = 8 elements).

        The flat-index trap (the 4-bit version of ADR 0019's isomorphism detector) — an
        implementation that repacks row by row shifts positions in shapes such as `(5,20)` /
        `(7,4)`, where a row ends in the middle of a word.
        """
        count = 1
        for dim in shape:
            count *= dim
        quantized = asymmetric_nibbles(count).reshape(shape)

        packed = pack_int4(quantized)

        assert packed.numel() == count // 2
        assert torch.equal(unpack_int4(packed, shape), quantized)

    def test_a_split_half_pack_does_not_survive_the_round_trip(self):
        """Fault injection: packing in the llama.cpp Q4_0-style split-half order (first half = low
        / second half = high) makes our unpack return other columns **with matching shape and
        type**.

        Confirms that the detector really detects — if this passed, the round-trip test would be
        tautological.
        """
        quantized = asymmetric_nibbles(16)
        nibbles = (quantized + INT4_OFFSET).to(torch.uint8)
        half = nibbles.numel() // 2

        split = nibbles[:half] | (nibbles[half:] << 4)

        restored = unpack_int4(split, (16,))
        assert restored.shape == quantized.shape
        assert not torch.equal(restored, quantized)

    def test_an_odd_element_count_fails_loudly(self):
        """A last element sticking out by half a byte cannot be packed (the byte length would not
        follow from the declaration)."""
        with pytest.raises(EmitError, match="is odd"):
            pack_int4(torch.zeros(3, dtype=torch.int8))


def conv1d_attrs(groups: int = 1) -> dict:
    """conv1d attrs (all four must be declared — `ops.CONV1D_ATTRS`)."""
    return {"stride": 1, "padding": 0, "dilation": 1, "groups": groups}


def int4_weight_graph(
    *,
    embedding: bool = False,
    conv: bool = False,
    wide: bool = False,
    group_size: int = 16,
    conv_shape: tuple[int, ...] = (3, 2, 16),
    conv_groups: int = 1,
    conv_op: str = "conv1d",
) -> tuple[IrGraph, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """A graph with the linear weight `[3,32]` fake-quantized to group-symmetric i4.

    The in axis is 32 and the group 16, so one row holds two groups (the group-form scale
    `[3,2]`). `embedding` adds **an embedding table** (the second i4-eligible weight slot — the
    extension of ADR 0069 decision 5), `conv` adds **a conv weight** (by default the conv1d
    `[3,2,16]` with `groups == 1` = the third i4-eligible one; `conv_shape` / `conv_groups` /
    `conv_op` also take the branches that fall outside eligibility), and `wide` adds **a linear
    with row length 48** (divisible by the shipped g 16 but not by
    `quantize.DEFAULT_GROUP_SIZE` = 32).

    The conv weight is rounded and put into the ledger **only in an eligible shape** (emit leaves
    an ineligible one as f32, so the ledger would go unused even if it were rounded).
    """
    nodes = [IrNode(op="linear", ins=["x", "w", "b"], outs=["h"], attrs={})]
    inputs = [IrInput(name="x", dtype="f32", shape=["T", 32])]
    initializers = {
        "w": IrInitializer(tensor="enc.w", storage=IrStorage(dtype="f32")),
        "b": IrInitializer(tensor="enc.b", storage=IrStorage(dtype="f32")),
    }
    values = {
        "w": IrValue(dtype="f32", shape=[3, 32]),
        "b": IrValue(dtype="f32", shape=[3]),
        "h": IrValue(dtype="f32", shape=["T", 3]),
    }
    tensors = {"enc.w": torch.randn(3, 32), "enc.b": torch.randn(3)}
    outputs = ["h"]
    if embedding:
        nodes.append(
            IrNode(op="embedding", ins=["emb", "idx"], outs=["e"], attrs={"padding_idx": -1})
        )
        inputs.append(IrInput(name="idx", dtype="i32", shape=["T"]))
        initializers["emb"] = IrInitializer(tensor="enc.emb", storage=IrStorage(dtype="f32"))
        values["emb"] = IrValue(dtype="f32", shape=[3, 32])
        values["e"] = IrValue(dtype="f32", shape=["T", 32])
        tensors["enc.emb"] = torch.randn(3, 32)
        outputs.append("e")
    if wide:
        nodes.append(IrNode(op="linear", ins=["xw", "ww", "bw"], outs=["hw"], attrs={}))
        inputs.append(IrInput(name="xw", dtype="f32", shape=["T", 48]))
        initializers["ww"] = IrInitializer(tensor="enc.ww", storage=IrStorage(dtype="f32"))
        initializers["bw"] = IrInitializer(tensor="enc.bw", storage=IrStorage(dtype="f32"))
        values["ww"] = IrValue(dtype="f32", shape=[3, 48])
        values["bw"] = IrValue(dtype="f32", shape=[3])
        values["hw"] = IrValue(dtype="f32", shape=["T", 3])
        tensors["enc.ww"] = torch.randn(3, 48)
        tensors["enc.bw"] = torch.randn(3)
        outputs.append("hw")
    if conv:
        # The conv input is rank 3 (`x[B,Cin,L]`) — L = K + 1, so the output length is 2
        # (stride / dilation 1, padding 0). Kept in a form that passes verify's shape inference.
        channels, kernel = conv_shape[1] * conv_groups, conv_shape[2]
        attrs = conv1d_attrs(conv_groups) if conv_op == "conv1d" else {"stride": 1, "padding": 0}
        nodes.append(IrNode(op=conv_op, ins=["cx", "cw", "cb"], outs=["c"], attrs=attrs))
        inputs.append(IrInput(name="cx", dtype="f32", shape=[1, channels, kernel + 1]))
        initializers["cw"] = IrInitializer(tensor="enc.cw", storage=IrStorage(dtype="f32"))
        initializers["cb"] = IrInitializer(tensor="enc.cb", storage=IrStorage(dtype="f32"))
        values["cw"] = IrValue(dtype="f32", shape=list(conv_shape))
        values["cb"] = IrValue(dtype="f32", shape=[conv_shape[0]])
        values["c"] = IrValue(dtype="f32", shape=[1, conv_shape[0], 2])
        tensors["enc.cw"] = torch.randn(*conv_shape)
        tensors["enc.cb"] = torch.randn(conv_shape[0])
        outputs.append("c")
    graph = IrGraph(
        symbols=["T"],
        inputs=inputs,
        outputs=outputs,
        initializers=initializers,
        values=values,
        nodes=nodes,
    )
    scales = {}
    for key in ["enc.w"] + (["enc.ww"] if wide else []) + (["enc.cw"] if conv else []):
        weight = tensors[key]
        if weight.numel() // weight.shape[0] % group_size:
            continue  # partial group (ADR 0069 decision 2) — cannot be rounded = not eligible
        scale = group_scale(weight, group_size)
        tensors[key] = dequantize_int4(quantize_to_int4(weight, scale), scale)
        scales[key] = scale
    return graph, tensors, scales


def write_int4(path, graph: IrGraph, tensors, scales):
    return write_component(path, graph, tensors, weight_dtype="i4", weight_scales=scales)


def int4_embedding_graph() -> tuple[IrGraph, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """A graph with **both** linear and embedding rounded to group-symmetric i4 (the positive form
    of the embedding extension).

    For the embedding table `[V,D]`, `channel_rows` is the identity, so the group scale is
    `[V, D/group]`, "the weight's rank with only the last dimension a group count" — it can be
    stored in exactly the same form as linear.
    """
    graph, tensors, scales = int4_weight_graph(embedding=True)
    emb_scale = group_scale(tensors["enc.emb"], 16)
    tensors["enc.emb"] = dequantize_int4(quantize_to_int4(tensors["enc.emb"], emb_scale), emb_scale)
    scales["enc.emb"] = emb_scale
    return graph, tensors, scales


class TestI4Eligibility:
    """i4 eligibility is **limited to the weight slots of linear / embedding / conv1d** (ADR 0069
    decision 5 + the embedding / conv1d extensions — only these three kernels have an i4 decode
    path).

    For conv1d only the igemm variant can decode, so it needs `groups == 1`, and the stored row
    (the flattened `Cin·K`) must also be divisible by the group length.
    """

    def test_linear_and_embedding_weight_slots_are_listed(self):
        graph, _, _ = int4_weight_graph(embedding=True)

        assert eligible_compressed_initializers(graph) == {"w", "emb"}
        assert i4_eligible_initializers(graph) == {"w", "emb"}

    def test_a_dense_conv1d_weight_is_i4_eligible(self):
        """A conv1d with `groups == 1` is i4-eligible (the extension of phase J-5b).

        The row length is **the flattened receptive field** (`Cin·K` = 32), not the last
        dimension (K = 16) — it splits into 2 groups of 16. Looking only at K, `[3,2,16]` would
        pass as "16 % 16 == 0", and groups spanning the receptive field would be read with
        another scale — a silent wrong value.
        """
        graph, _, _ = int4_weight_graph(conv=True)

        assert eligible_compressed_initializers(graph) == {"w", "cw"}
        assert i4_eligible_initializers(graph, group_size=16) == {"w", "cw"}

    def test_a_grouped_conv1d_weight_is_not_i4_eligible(self):
        """`groups > 1` runs on the direct kernel, which has no i4 decode path."""
        graph, _, _ = int4_weight_graph(conv=True, conv_groups=2, conv_shape=(4, 2, 16))

        assert eligible_compressed_initializers(graph) == {"w", "cw"}
        assert i4_eligible_initializers(graph, group_size=16) == {"w"}

    def test_a_conv1d_without_a_declared_groups_attr_fails_loudly(self):
        """A missing `groups` declaration is not filled with a default of 1 (ADR 0012; the same
        handling as the runtime).

        Silently assuming 1 would keep a depthwise weight resident as i4, and the direct kernel
        would read the packed bytes as f32 (a silent wrong value with no exception).
        """
        graph, _, _ = int4_weight_graph(conv=True)
        without_groups = {key: value for key, value in conv1d_attrs().items() if key != "groups"}
        graph.nodes[-1] = replace(graph.nodes[-1], attrs=without_groups)

        with pytest.raises(OpContractError, match=re.escape("attrs.groups")):
            i4_eligible_initializers(graph, group_size=16)

    def test_a_conv1d_whose_flattened_row_is_not_divisible_is_not_i4_eligible(self):
        """A row length `Cin·K` = 24 leaves a partial group with group 16 (ADR 0069 decision 2).

        Counting it as eligible would fail the whole export as "a weight fake-quant never
        reached" — an ordinary graph with linear and a non-divisible conv1d could no longer be
        written as i4.
        """
        graph, _, _ = int4_weight_graph(conv=True, conv_shape=(3, 3, 8))

        assert eligible_compressed_initializers(graph) == {"w", "cw"}
        assert i4_eligible_initializers(graph, group_size=16) == {"w"}

    def test_a_conv_transpose1d_weight_is_not_i4_eligible(self):
        """The transposed layout `[Cin,Cout,K]` does not lead with the row axis = the pack order
        and the row order disagree.

        The maintainer's ruling of 2026-08-20 does not buy a permuted pack, so there is no decode
        path for it at all.
        """
        graph, _, _ = int4_weight_graph(conv=True, conv_op="conv_transpose1d")

        assert eligible_compressed_initializers(graph) == {"w", "cw"}
        assert i4_eligible_initializers(graph, group_size=16) == {"w"}

    def test_a_weight_shared_with_another_weight_slot_is_excluded(self):
        """A weight also shared with a weight slot that has no i4 path is not eligible
        (the decode path would not be unique)."""
        graph, _, _ = int4_weight_graph()
        graph.nodes.append(
            IrNode(op="conv1d", ins=["x", "w", "b"], outs=["c"], attrs=conv1d_attrs(groups=2))
        )

        assert i4_eligible_initializers(graph, group_size=16) == set()


class TestI4Storage:
    def test_eligible_linear_weights_are_stored_as_i4_with_a_group_scale(self, tmp_path):
        graph, tensors, scales = int4_weight_graph()

        path = write_int4(tmp_path / "model.krm", graph, tensors, scales)

        declared = written_graph(path).initializers
        assert declared["w"].storage.dtype == "i4"
        assert declared["w"].storage.scale == "karume.scale.enc.w"
        assert declared["w"].storage.group_size == 16
        # MUST: bias is always f32 (the root-cause fix of the prototype's demotion bug — ADR 0006).
        assert declared["b"].storage.dtype == "f32"
        assert stored_layout(path, "enc.w") == "i4"
        # The declared shape stays logical; only the payload byte length follows from the bit
        # width.
        assert len(stored_bytes(path, "enc.w")) == 3 * 32 // 2
        # The scale is f32, 32 / 16 = 2 groups × 3 rows.
        assert len(stored_bytes(path, "karume.scale.enc.w")) == 3 * 2 * 4

    def test_the_stored_bytes_reconstruct_the_weight_bit_for_bit(self, tmp_path):
        """`dequant(unpack(stored bytes))` **matches** the fake-quantized weight **bit for bit**.

        The same formula as emit's check (`_convert_for_storage`), but this one starts from **the
        bytes that reached the file** — pinning the correspondence including the writing path
        (order, offset, the memoryview cast).
        """
        graph, tensors, scales = int4_weight_graph()
        expected = tensors["enc.w"].clone()

        path = write_int4(tmp_path / "model.krm", graph, tensors, scales)

        packed = stored_payload(path, "enc.w")
        restored = dequantize_int4(unpack_int4(packed, (3, 32)), scales["enc.w"])
        assert torch.equal(restored, expected)

    def test_the_emitted_i4_file_satisfies_the_reader_layout_rules(self, tmp_path):
        graph, tensors, scales = int4_weight_graph()

        path = write_int4(tmp_path / "model.krm", graph, tensors, scales)

        # Passes if nothing is raised (the reader checks 64 B block alignment and lengths that are
        # multiples of 4 B).
        bound(path)

    def test_the_i4_file_passes_the_full_verification(self, tmp_path):
        """The i4 write passes every rule of the re-read and the merge to the end (the execution
        capability is open).

        This being green is exactly the re-read of the writing path
        (`karume.container.read_container` → the merge `verify.bind_graphs`) accepting
        `int4-sym-g` blocks. It passes every rule: declaration, shape, group-form scale, runtime
        support, op contracts.
        """
        graph, tensors, scales = int4_weight_graph()

        path = write_int4(tmp_path / "model.krm", graph, tensors, scales)

        expected = compressed_view(graph, {"w": "i4"}, group_size=16)
        assert written_graph(path).to_dict() == expected.to_dict()

    def test_the_breakdown_counts_i4_bytes_and_the_group_scale_overhead(self, tmp_path):
        graph, tensors, scales = int4_weight_graph()

        path = write_int4(tmp_path / "model.krm", graph, tensors, scales)
        breakdown = storage_breakdown(written_graph(path))

        assert breakdown.compressed_tensors == 1
        assert breakdown.compressed_bytes == 3 * 32 // 2  # i4 = 0.5 byte/element
        # The scale is the group count (3 rows × 2 groups) × 4 bytes.
        assert breakdown.scale_bytes == 3 * 2 * 4
        assert breakdown.plain_tensors == 1
        assert breakdown.plain_bytes == 3 * 4

    def test_a_dense_conv1d_weight_is_stored_as_i4_with_a_rank2_group_scale(self, tmp_path):
        """A conv1d `[3,2,16]` with `groups == 1` is stored as i4 with a **rank-2** scale
        (phase J-5b).

        The scale shape is "rows = the leading dimension, last dimension = row length / group
        length" whatever the rank (the rank-independent rule of ADR 0069 decision 3) — not
        `[3,2,1]` with the weight's rank.
        """
        graph, tensors, scales = int4_weight_graph(conv=True)

        path = write_int4(tmp_path / "model.krm", graph, tensors, scales)

        declared = written_graph(path).initializers
        assert declared["cw"].storage.dtype == "i4"
        assert declared["cw"].storage.group_size == 16
        assert stored_layout(path, "enc.cw") == "i4"
        assert len(stored_bytes(path, "enc.cw")) == 3 * 2 * 16 // 2
        assert len(stored_bytes(path, "karume.scale.enc.cw")) == 3 * 2 * 4
        assert declared["cb"].storage.dtype == "f32", "bias is always f32 (ADR 0006)"

    def test_the_stored_conv1d_bytes_reconstruct_the_rank3_weight_bit_for_bit(self, tmp_path):
        """At rank 3 too, `dequant(unpack(stored bytes))` **matches** the rounded weight **bit for
        bit**.

        The nibble order of the payload is the weight's contiguous memory order (the row-major
        `[O,Cin,K]` = the same bytes as the flat `[O,Cin·K]`) — a wrong row folding still matches
        shape, type and byte length, so this check is the only detection point (ADR 0069
        decision 4 ③).
        """
        graph, tensors, scales = int4_weight_graph(conv=True)
        expected = tensors["enc.cw"].clone()

        path = write_int4(tmp_path / "model.krm", graph, tensors, scales)

        packed = stored_payload(path, "enc.cw")
        restored = dequantize_int4(unpack_int4(packed, (3, 2, 16)), scales["enc.cw"])
        assert torch.equal(restored, expected)

    def test_the_conv1d_i4_file_passes_the_full_verification(self, tmp_path):
        """The emit → verify_container round trip runs to the end for rank-3 i4 too.

        The group-form checks (`_check_group_quantized_shape` / the scale-length rule of
        `verify.bind_graphs`) accept a rank-2 scale for a rank-3 weight — if they still required
        the same rank, a file that could be written could not be read.
        """
        graph, tensors, scales = int4_weight_graph(conv=True)

        path = write_int4(tmp_path / "model.krm", graph, tensors, scales)

        declared = written_graph(path).initializers
        assert declared["cw"].storage.dtype == "i4"
        assert declared["w"].storage.dtype == "i4"

    def test_a_grouped_conv1d_weight_stays_f32(self, tmp_path):
        """An ineligible conv1d is not made i4 but stays f32 (Codex phase F finding I4-ELIG-01).

        Failing the whole export here would make an ordinary graph with linear + depthwise conv
        impossible to write as i4. The design paired with the runtime's fallback (outside
        eligible ∩ i4Eligible, decoded on the CPU) is "compress only the weights of ops with a
        decode path, f32 for the rest", the same as i8's ineligible weights.
        """
        # Cout is 4, divisible by groups (the default 3 breaks the conv1d contract [groups divides
        # Cin and Cout] and gives a graph the runtime parser rejects — an error in the subject).
        graph, tensors, scales = int4_weight_graph(conv=True, conv_groups=2, conv_shape=(4, 2, 16))

        path = write_int4(tmp_path / "model.krm", graph, tensors, scales)

        declared = written_graph(path).initializers
        assert declared["w"].storage.dtype == "i4"
        assert declared["cw"].storage.dtype == "f32"

    def test_a_row_only_the_shipped_group_length_divides_is_still_stored_as_i4(self, tmp_path):
        """A linear with row length 48 rounded with g16 is also stored as i4 (the divisor is the
        actual shipped g).

        Fixing the eligibility divisor to the default g (32) would drop only this weight from
        eligibility on `48 % 32 != 0`, and its values, still rounded to the i4 grid, would be
        **stored as f32** — with even one linear of row length 32 the "0 eligible" check does not
        fire either, so a distribution that carries the quality loss without shrinking would ship
        with no exception and no diagnostic (ADR 0006's "compression requested but effectively
        f32").
        """
        graph, tensors, scales = int4_weight_graph(wide=True)

        path = write_int4(tmp_path / "model.krm", graph, tensors, scales)

        declared = written_graph(path).initializers
        assert declared["w"].storage.dtype == "i4", (
            "the row-length-32 side (divisible by the default g too)"
        )
        assert declared["ww"].storage.dtype == "i4", (
            "the row-length-48 side (divisible only by the shipped g 16)"
        )
        assert declared["ww"].storage.group_size == 16
        # The scale has 48 / 16 = 3 groups (with the default g the declaration itself would not
        # exist).
        assert len(stored_bytes(path, "karume.scale.enc.ww")) == 3 * 3 * 4

    def test_an_embedding_table_is_stored_as_i4_with_a_group_scale(self, tmp_path):
        """The embedding table `[V,D]` is stored as i4 too (the embedding extension of ADR 0069
        decision 5).

        The scale has exactly linear's group form (same rank, only the last dimension a group
        count), and the declaration carries `storage.group_size` too — one more slot, still one
        storage rule.
        """
        graph, tensors, scales = int4_embedding_graph()

        path = write_int4(tmp_path / "model.krm", graph, tensors, scales)

        declared = written_graph(path).initializers
        assert declared["emb"].storage.dtype == "i4"
        assert declared["emb"].storage.scale == "karume.scale.enc.emb"
        assert declared["emb"].storage.group_size == 16
        assert stored_layout(path, "enc.emb") == "i4"
        assert len(stored_bytes(path, "enc.emb")) == 3 * 32 // 2
        assert len(stored_bytes(path, "karume.scale.enc.emb")) == 3 * 2 * 4

    def test_the_stored_embedding_bytes_reconstruct_the_table_bit_for_bit(self, tmp_path):
        """For the embedding table too, `dequant(unpack(stored bytes))` matches the rounded weight
        bit for bit."""
        graph, tensors, scales = int4_embedding_graph()
        expected = tensors["enc.emb"].clone()

        path = write_int4(tmp_path / "model.krm", graph, tensors, scales)

        packed = stored_payload(path, "enc.emb")
        restored = dequantize_int4(unpack_int4(packed, (3, 32)), scales["enc.emb"])
        assert torch.equal(restored, expected)

    def test_the_emitted_embedding_i4_file_satisfies_the_reader_layout_rules(self, tmp_path):
        """Two I4 tensors side by side stay "gapless and 4-byte aligned" (the order of ADR 0069
        addendum 2)."""
        graph, tensors, scales = int4_embedding_graph()

        path = write_int4(tmp_path / "model.krm", graph, tensors, scales)

        bound(path)

    def test_the_embedding_i4_file_passes_the_full_verification(self, tmp_path):
        """The emit → verify_container round trip runs to the end for embedding i4 too.

        The group-form check (`_check_group_quantized_shape` — last dimension % group_size) holds
        as is on the D axis of the embedding `[V,D]`.
        """
        graph, tensors, scales = int4_embedding_graph()

        path = write_int4(tmp_path / "model.krm", graph, tensors, scales)

        expected = compressed_view(graph, {"w": "i4", "emb": "i4"}, group_size=16)
        assert written_graph(path).to_dict() == expected.to_dict()

    def test_the_breakdown_counts_both_i4_tensors(self, tmp_path):
        """The i4 bytes of the breakdown cover the two tensors, linear and embedding (0.5 byte /
        element)."""
        graph, tensors, scales = int4_embedding_graph()

        path = write_int4(tmp_path / "model.krm", graph, tensors, scales)
        breakdown = storage_breakdown(written_graph(path))

        assert breakdown.compressed_tensors == 2
        assert breakdown.compressed_bytes == 2 * (3 * 32 // 2)
        assert breakdown.scale_bytes == 2 * (3 * 2 * 4)
        assert breakdown.plain_tensors == 1
        assert breakdown.plain_bytes == 3 * 4

    def test_a_graph_whose_only_weight_is_shared_with_conv_fails_loudly(self, tmp_path):
        """A graph whose only weight is shared by linear and a depthwise conv fails as "0
        eligible"."""
        graph, tensors, scales = int4_weight_graph()
        graph.nodes.append(
            IrNode(op="conv1d", ins=["x", "w", "b"], outs=["c"], attrs=conv1d_attrs(groups=2))
        )
        graph.values["c"] = IrValue(dtype="f32", shape=["T", 3])
        graph.outputs.append("c")

        with pytest.raises(EmitError, match="Not a single compressed storage was planned"):
            write_int4(tmp_path / "model.krm", graph, tensors, scales)

    def test_a_missing_scale_fails_loudly(self, tmp_path):
        """Eligible but without a scale = a weight fake-quant never reached (ADR 0006)."""
        graph, tensors, _ = int4_weight_graph()

        with pytest.raises(EmitError, match="has no group scale"):
            write_int4(tmp_path / "model.krm", graph, tensors, {})

    def test_a_scale_that_is_not_in_group_form_fails_loudly(self, tmp_path):
        graph, tensors, _ = int4_weight_graph()

        with pytest.raises(EmitError, match="group shape"):
            write_int4(tmp_path / "model.krm", graph, tensors, {"enc.w": torch.ones(1, 2)})

    def test_a_group_size_outside_the_accepted_set_fails_loudly(self, tmp_path):
        """A scale `[3,4]` means group 8 — a power of 2 but below 16 (ADR 0069 decision 2).

        Failing in `verify` after writing would "leave an unreadable file in the distribution",
        so it fails at the planning step before writing.
        """
        graph, tensors, _ = int4_weight_graph()

        with pytest.raises(EmitError, match="is not a power of two of at least 16"):
            write_int4(tmp_path / "model.krm", graph, tensors, {"enc.w": torch.ones(3, 4)})

    def test_a_scale_that_is_not_f32_fails_loudly(self, tmp_path):
        """The companion scale is fixed to F32 (for the same reason as i8 — the
        inverse-conversion equality check cannot detect it)."""
        graph, tensors, scales = int4_weight_graph()
        half = {key: value.to(torch.float16) for key, value in scales.items()}

        with pytest.raises(EmitError, match="scale dtype"):
            write_int4(tmp_path / "model.krm", graph, tensors, half)

    def test_an_unrounded_eligible_weight_fails_loudly(self, tmp_path):
        """A forgotten rounding / a wrong order is not let through silently (ADR 0006)."""
        graph, tensors, scales = int4_weight_graph()
        tensors["enc.w"] = torch.full((3, 32), 1.0 / 3.0)

        with pytest.raises(EmitError, match="not bit-identical"):
            write_int4(tmp_path / "model.krm", graph, tensors, scales)

    def test_a_scale_that_is_not_the_one_fake_quant_used_fails_loudly(self, tmp_path):
        """Writing with a scale **other than** the one fake-quant used fails the inverse-conversion
        gate."""
        graph, tensors, scales = int4_weight_graph()
        drifted = {key: value * 1.0000002 for key, value in scales.items()}

        with pytest.raises(EmitError, match="not bit-identical"):
            write_int4(tmp_path / "model.krm", graph, tensors, drifted)

    def test_a_tampered_packed_byte_is_caught_by_the_round_trip_gate(self, tmp_path, monkeypatch):
        """Fault injection: rewriting a single packed byte fires the inverse-conversion bit-identity
        gate (ADR 0069 decision 4 ③).

        Neither shape nor dtype nor byte length changes, so there is no other detection point.
        """
        from karume import emit

        pack = emit.pack_int4

        def tamper(quantized):
            packed = pack(quantized)
            packed[0] ^= 0x10  # shift the high nibble by one level
            return packed

        monkeypatch.setattr(emit, "pack_int4", tamper)
        graph, tensors, scales = int4_weight_graph()

        with pytest.raises(EmitError, match="not bit-identical"):
            write_int4(tmp_path / "model.krm", graph, tensors, scales)

    def test_a_graph_without_eligible_weights_fails_loudly(self, tmp_path):
        graph, tensors = sample_graph()

        with pytest.raises(EmitError, match="Not a single compressed storage was planned"):
            write_model(tmp_path / "model.krm", graph, tensors, weight_dtype="i4")


class TestFailureLeavesTheGraphUntouched:
    """If even one tensor fails, no declaration is rewritten (the commit comes after every byte is
    written).

    There are two eligible weights (`emb` / `w`), and each case below is "one passes, the other
    fails". Rewriting at the point of failure would leave a compressed declaration for one of them
    in the caller's `graph` even though writing stopped with an exception, and paths that reuse
    that graph (rewriting the same graph with another dtype / counting the breakdown) would
    silently answer with declarations that disagree with the data.
    """

    def test_a_failed_f16_pass_leaves_no_partial_declaration(self, tmp_path):
        graph, tensors = weight_graph()
        tensors["enc.w"] = torch.full((3, 4), 1.0 / 3.0)
        before = json.loads(graph.to_json())

        with pytest.raises(EmitError, match="fake-quant"):
            write_model(tmp_path / "model.krm", graph, tensors, weight_dtype="f16")

        assert graph.initializers["emb"].storage.dtype == "f32"
        assert json.loads(graph.to_json()) == before

    def test_a_failed_i8_pass_leaves_no_partial_declaration(self, tmp_path):
        graph, tensors, scales = int8_weight_graph()
        del scales["enc.w"]
        before = json.loads(graph.to_json())

        with pytest.raises(EmitError, match="has no per-channel scale"):
            write_model(
                tmp_path / "model.krm",
                graph,
                tensors,
                weight_dtype="i8",
                weight_scales=scales,
            )

        assert graph.initializers["emb"].storage.dtype == "f32"
        assert json.loads(graph.to_json()) == before

    def test_a_failure_while_the_payload_is_pulled_leaves_no_declaration(self, tmp_path):
        """The inverse-conversion gate runs **when the data is pulled** (the planning step reads
        not a single byte).

        The installing layer (`karume.publish`) discards everything including `.partial`, so no
        file remains, and the caller's `graph` must not have been rewritten at all.
        """
        graph, tensors, scales = int8_weight_graph()
        tensors["enc.w"] = torch.full((3, 4), 1.0 / 3.0)
        before = json.loads(graph.to_json())
        path = tmp_path / "model.krm"

        # Planning alone passes (checks that read real data wait until the payload is pulled).
        stored_model(graph, tensors, weight_dtype="i8", weight_scales=scales)

        with pytest.raises(EmitError, match="not bit-identical"):
            write_model(path, graph, tensors, weight_dtype="i8", weight_scales=scales)

        assert list(tmp_path.iterdir()) == []
        assert json.loads(graph.to_json()) == before


class TestSuccessLeavesTheGraphUntouched:
    """Even when writing succeeds, the caller's `graph` does not change by a single byte.

    Writing the commit back into the caller's graph would leave the previous compressed
    declarations on the path that rewrites the same graph with another dtype (the form the
    `stored_model` docstring names as a reuse path) — the f32 plan is empty and does **not
    restore** declarations, so the second one would be a broken container with "declaration f16
    / data F32".
    """

    def test_a_successful_f16_write_leaves_the_declarations_alone(self, tmp_path):
        graph, tensors = weight_graph()
        before = json.loads(graph.to_json())

        write_model(tmp_path / "model.krm", graph, tensors, weight_dtype="f16")

        assert json.loads(graph.to_json()) == before

    def test_writing_the_same_graph_as_f32_after_f16_stays_readable(self, tmp_path):
        graph, tensors = weight_graph()

        write_model(tmp_path / "half.safetensors", graph, tensors, weight_dtype="f16")
        plain = write_component(tmp_path / "plain.safetensors", graph, tensors)

        assert stored_layout(plain, "enc.w") == "f32"
        # verify_container fails here if declaration and data disagree.
        assert written_graph(plain).to_dict() == graph.to_dict()


class TestStreamingConversion:
    """The compression conversion runs **one tensor at a time, when pulled** (the storage of
    ADR 0018 / 0019 itself is unchanged).

    Converting everything up front would keep the compressed set alive together with the caller's
    f32 set, and peak RAM would be the sum of both (at Irodori scale, f16 1.72GB / i8 0.87GB on
    top of f32 3.44GB).

    Each tensor is pulled twice (the pass that **takes the sha256 while writing**, and the re-read
    verification before installing), and it MUST return **the same bytes every time** — a
    non-deterministic conversion would produce a container whose real bytes differ from the
    declared sha256.
    """

    def test_only_one_converted_tensor_is_alive_at_a_time(self, tmp_path, monkeypatch):
        """Fault injection: reverting to an implementation that collects the converted tensors
        before writing keeps the previous one alive and fails."""
        from karume import emit

        alive: list[weakref.ReferenceType] = []
        convert = emit._convert_for_storage

        def spy(key, tensor, conversion):
            # By the time the next tensor starts, the previous converted one should be freed.
            assert [ref for ref in alive if ref() is not None] == []
            converted = convert(key, tensor, conversion)
            alive.append(weakref.ref(converted))
            return converted

        monkeypatch.setattr(emit, "_convert_for_storage", spy)
        graph, tensors = weight_graph()

        write_model(tmp_path / "model.krm", graph, tensors, weight_dtype="f16")

        # 2 tensors (enc.w / enc.emb) × 2 passes — the one that takes the sha256 **while
        # writing**, and the one pulled again by the re-read verification before installing. The
        # split form can write part 0 last, so writing and digest are a single pass (only the
        # single form, which puts part 0 first, pulls twice — container-v1 §4.1).
        assert len(alive) == 4

    def test_the_streamed_f16_bytes_match_a_pre_converted_write(self, tmp_path):
        """The bytes written while streaming = the container given everything pre-converted as
        plain bytes.

        The block / part layout **follows from the declaration** (decided without reading a single
        byte of data), so the path with the conversion and the path without give the same layout
        and the same payloads. A wrong derivation shifts whole offsets here.
        """
        graph, tensors = weight_graph()
        compressed = {"enc.w", "enc.emb"}
        stored = stored_model(graph, tensors, weight_dtype="f16")
        pre = {
            key: (value.to(torch.float16) if key in compressed else value).numpy().tobytes()
            for key, value in tensors.items()
        }

        streamed = write_component(
            tmp_path / "streamed.safetensors", graph, tensors, weight_dtype="f16"
        )
        reference = publish_container(
            tmp_path / "reference.krm",
            stored.graph,
            pre,
            stored.bindings,
            graph_name=GRAPH_NAME,
            provenance=PROVENANCE,
        )

        assert [path.read_bytes() for path in container_parts(streamed)] == [
            path.read_bytes() for path in reference.parts
        ]

    def test_the_streamed_i8_bytes_match_a_pre_converted_write(self, tmp_path):
        """The same for i8 (takes the form with companion scales added)."""
        graph, tensors, scales = int8_weight_graph()
        stored = stored_model(graph, tensors, weight_dtype="i8", weight_scales=scales)
        pre = {key: value.numpy().tobytes() for key, value in tensors.items()}
        for key, scale in scales.items():
            pre[key] = quantize_to_int8(tensors[key], scale).numpy().tobytes()
            pre[f"karume.scale.{key}"] = scale.numpy().tobytes()

        streamed = write_component(
            tmp_path / "streamed.safetensors",
            graph,
            tensors,
            weight_dtype="i8",
            weight_scales=scales,
        )
        reference = publish_container(
            tmp_path / "reference.krm",
            stored.graph,
            pre,
            stored.bindings,
            graph_name=GRAPH_NAME,
            provenance=PROVENANCE,
        )

        assert [path.read_bytes() for path in container_parts(streamed)] == [
            path.read_bytes() for path in reference.parts
        ]


class TestJsonCompliance:
    def test_non_finite_values_are_refused_at_serialization(self):
        """NaN / Infinity are not standard JSON literals — fail at serialization time."""
        graph, _ = sample_graph()
        graph.nodes[0] = IrNode(op="add", ins=["x", "w"], outs=["y"], attrs={"value": float("inf")})

        with pytest.raises(ValueError, match="JSON compliant"):
            graph.to_json()

    def test_ordinary_graphs_serialize_to_standard_json(self):
        graph, _ = sample_graph()

        restored = json.loads(graph.to_json(), parse_constant=lambda lit: pytest.fail(lit))

        assert restored["requires"]["ops"] == ["add"]


def mixed_weight_graph() -> tuple[IrGraph, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """A graph with both fake-quantized for the mix linear = i4 / embedding = i8.

    The minimal form of the LLM layout "embedding / tied lm_head in i8, the linear body in i4"
    (first seen with Gemma 4 E2B).
    """
    graph, tensors, scales = int4_weight_graph(embedding=True)
    emb_scale = channel_scale(tensors["enc.emb"], 0)
    tensors["enc.emb"] = quantize_to_int8(tensors["enc.emb"], emb_scale).to(torch.float32) * (
        emb_scale
    )
    scales["enc.emb"] = emb_scale
    return graph, tensors, scales


class TestMixedStorage:
    """`weight_dtype_overrides` — explicit per-tensor choices (override the default; fail loudly
    when they cannot be met)."""

    def test_an_override_mixes_i8_into_an_i4_default(self, tmp_path):
        graph, tensors, scales = mixed_weight_graph()

        path = write_component(
            tmp_path / "model.krm",
            graph,
            tensors,
            weight_dtype="i4",
            weight_scales=scales,
            weight_dtype_overrides={"enc.emb": "i8"},
        )

        declared = written_graph(path).initializers
        assert declared["w"].storage.dtype == "i4"
        assert declared["w"].storage.group_size == 16
        assert declared["emb"].storage.dtype == "i8"
        assert declared["emb"].storage.scale == "karume.scale.enc.emb"
        assert declared["b"].storage.dtype == "f32"
        assert stored_layout(path, "enc.w") == "i4"
        assert stored_layout(path, "enc.emb") == "i8"

    def test_overrides_compress_even_with_a_f32_default(self, tmp_path):
        """A default of f32 used to "return an empty plan at once" — compression by explicit
        choices alone must pass too."""
        graph, tensors, scales = mixed_weight_graph()

        path = write_component(
            tmp_path / "model.krm",
            graph,
            tensors,
            weight_scales=scales,
            weight_dtype_overrides={"enc.emb": "i8"},
        )

        declared = written_graph(path).initializers
        assert declared["emb"].storage.dtype == "i8"
        assert declared["w"].storage.dtype == "f32"

    def test_an_override_to_f32_exempts_a_weight_from_the_default(self, tmp_path):
        graph, tensors = weight_graph()

        path = write_component(
            tmp_path / "model.krm",
            graph,
            tensors,
            weight_dtype="f16",
            weight_dtype_overrides={"enc.w": "f32"},
        )

        declared = written_graph(path).initializers
        assert declared["w"].storage.dtype == "f32"
        assert declared["emb"].storage.dtype == "f16"

    def test_an_unknown_override_key_fails_loudly(self, tmp_path):
        graph, tensors = weight_graph()

        with pytest.raises(EmitError, match="are not the tensor of any initializer"):
            write_model(
                tmp_path / "model.krm",
                graph,
                tensors,
                weight_dtype="f16",
                weight_dtype_overrides={"enc.typo": "f16"},
            )

    def test_an_override_on_an_ineligible_weight_fails_loudly(self, tmp_path):
        """The default dtype quietly leaves ineligible weights as f32, but an explicit choice is
        never silently turned into another storage."""
        graph, tensors = weight_graph(share_weight=True)

        with pytest.raises(EmitError, match="not eligible"):
            write_model(
                tmp_path / "model.krm",
                graph,
                tensors,
                weight_dtype_overrides={"enc.w": "f16"},
            )

    def test_an_explicit_i4_on_a_conv_weight_fails_loudly(self, tmp_path):
        """An i4-ineligible conv (depthwise here) has no i4 decode path even if generally eligible.

        The default quietly leaves it as f32, but an explicit choice fails loudly (the line drawn
        by `_plan_weight_dtype`).
        """
        graph, tensors, scales = int4_weight_graph(conv=True, conv_groups=2, conv_shape=(4, 2, 16))

        with pytest.raises(EmitError, match="is only for weight slots of"):
            write_model(
                tmp_path / "model.krm",
                graph,
                tensors,
                weight_scales=scales,
                weight_dtype_overrides={"enc.cw": "i4"},
            )

    def test_an_unknown_override_dtype_fails_loudly(self, tmp_path):
        graph, tensors = weight_graph()

        with pytest.raises(EmitError, match="cannot be written"):
            write_model(
                tmp_path / "model.krm",
                graph,
                tensors,
                weight_dtype_overrides={"enc.w": "f64"},
            )

    def test_an_override_on_a_non_f32_tensor_fails_loudly(self, tmp_path):
        """The default quietly skips non-f32 data, but an explicit choice fails with "compressed
        storage takes f32 data only".

        i32 storage (a symbol-dependent constant — ADR 0010) flowing into a weight slot. Its
        semantics differ entirely, so this is the branch that never silently compresses (the
        counterpart of the other 3 branches).
        """
        graph, tensors, scales = mixed_weight_graph()
        tensors["enc.emb"] = torch.zeros(3, 32, dtype=torch.int32)

        with pytest.raises(EmitError, match="compressed storage is for f32 data only"):
            write_model(
                tmp_path / "model.krm",
                graph,
                tensors,
                weight_scales=scales,
                weight_dtype_overrides={"enc.emb": "i8"},
            )


def conv2d_only_graph() -> tuple[IrGraph, dict[str, torch.Tensor]]:
    """A graph with a weight slot but **no i4 decode path** (one conv2d — the VAE's minimal
    form)."""
    graph = IrGraph(
        symbols=[],
        inputs=[IrInput(name="x", dtype="f32", shape=[1, 2, 4, 4])],
        outputs=["y"],
        initializers={
            "w": IrInitializer(tensor="enc.w", storage=IrStorage(dtype="f32")),
            "b": IrInitializer(tensor="enc.b", storage=IrStorage(dtype="f32")),
        },
        values={
            "w": IrValue(dtype="f32", shape=[3, 2, 3, 3]),
            "b": IrValue(dtype="f32", shape=[3]),
            "y": IrValue(dtype="f32", shape=[1, 3, 2, 2]),
        },
        nodes=[
            IrNode(
                op="conv2d",
                ins=["x", "w", "b"],
                outs=["y"],
                attrs={"stride": [1, 1], "padding": [0, 0], "dilation": [1, 1], "groups": 1},
            )
        ],
    )
    return graph, {"enc.w": rounded(3, 2, 3, 3), "enc.b": rounded(3)}


class TestThePlannedCompressionGate:
    """The "compression requested but 0 MB eligible" check asks **whether no compressed storage
    at all was planned** (ADR 0006's standing diagnostic).

    Even with zero tensors in the default dtype's storage, the intent is met if explicit choices
    cover every eligible tensor with another dtype — failing that form too would make a
    distribution written out with per-tensor mixed storage impossible. That a truly empty plan
    still fails is covered by
    {@link TestF16Storage.test_a_graph_without_eligible_weights_fails_loudly} and the 3 tests
    after it.
    """

    def test_overrides_covering_every_eligible_weight_pass_the_gate(self, tmp_path):
        """Passes when explicit i4 choices are planned, even with no default i8 declaration."""
        graph, tensors, scales = int4_weight_graph()

        path = write_component(
            tmp_path / "model.krm",
            graph,
            tensors,
            weight_dtype="i8",
            weight_scales=scales,
            weight_dtype_overrides={"enc.w": "i4"},
        )

        assert written_graph(path).initializers["w"].storage.dtype == "i4"

    def test_the_i4_message_names_the_expansion_path(self, tmp_path):
        """With 0 i4-eligible tensors, the cause is "no decode path", not "no weight slot".

        A conv2d weight is generally eligible, but no kernel can decode it as i4 (ADR 0069
        decision 5). "Is i4 requested for a graph without fused-op weights" would suspect the
        graph's structure and miss.
        """
        graph, tensors = conv2d_only_graph()

        with pytest.raises(EmitError) as err:
            write_model(tmp_path / "model.krm", graph, tensors, weight_dtype="i4")

        assert "i4 decode path (linear / embedding / conv1d" in str(err.value)

    def test_the_message_reports_the_default_and_the_explicit_dtypes(self, tmp_path):
        """The diagnostic shows the default and explicit breakdown (so the reader sees which choice
        matched nothing)."""
        graph, tensors = sample_graph()

        with pytest.raises(EmitError) as err:
            write_model(tmp_path / "model.krm", graph, tensors, weight_dtype="f16")

        assert "default f16" in str(err.value)
        assert "explicit none" in str(err.value)


class TestInitializerKeyInjectivity:
    """Declarations are per name, data per key — unless name ↔ tensor key is 1:1, it fails at the
    planning step."""

    def test_two_initializers_sharing_one_tensor_key_fail_loudly(self, tmp_path):
        """If only one were eligible, it would be a silent wrong value "data packed / declaration
        f32"."""
        graph, tensors = weight_graph()
        graph = replace(
            graph,
            initializers={
                **graph.initializers,
                "w_alias": IrInitializer(tensor="enc.w", storage=IrStorage(dtype="f32")),
            },
            values={
                **graph.values,
                "w_alias": IrValue(dtype="f32", shape=[3, 4]),
                "a": IrValue(dtype="f32", shape=[3, 4]),
            },
            nodes=[
                *graph.nodes,
                IrNode(op="mul", ins=["w_alias", "w_alias"], outs=["a"], attrs={}),
            ],
        )

        with pytest.raises(EmitError, match="are not 1:1"):
            write_model(tmp_path / "model.krm", graph, tensors, weight_dtype="f16")
