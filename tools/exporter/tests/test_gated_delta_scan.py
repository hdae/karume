"""`karume::gated_delta_scan` and its handler (extended composite layer — ADR 0135, the value form).

Pinned here: the eager body is upstream's prefill function (bit for bit where transformers is
installed, and inside the acceptance band of the runtime fixture everywhere), the op exports as one
IR node with a symbolic length axis and survives the write → read back → verify round trip, and the
handler narrows arity, rank and dtype.
"""

from __future__ import annotations

import json

import pytest
import torch
from conftest import RUNTIME_FIXTURES, node_ops, only_node
from safetensors import safe_open
from torch import nn

from karume.container import Provenance, container_parts
from karume.custom_ops import GATED_DELTA_RULE_UPSTREAM, gated_delta_scan
from karume.ops import OpContractError
from karume.pipeline import export_to_file
from karume.verify import verify_container

#: The runtime's fixture (written by tools/export-recipes/_shared/gated_delta_fixture.py from
#: upstream's own f32 functions and an f64 oracle).
FIXTURE = RUNTIME_FIXTURES / "gated-delta-scan.safetensors"


def _inputs(time=7, key_heads=2, value_heads=4, dim=128, seed=20261031):
    """q / k l2-normalised (the contract), g ≤ 0, β in (0, 1)."""
    generator = torch.Generator().manual_seed(seed)
    q = nn.functional.normalize(torch.randn(1, time, key_heads, dim, generator=generator), dim=-1)
    k = nn.functional.normalize(torch.randn(1, time, key_heads, dim, generator=generator), dim=-1)
    v = torch.randn(1, time, value_heads, dim, generator=generator)
    g = -torch.rand(1, time, value_heads, generator=generator) * 2
    beta = torch.rand(1, time, value_heads, generator=generator) * 0.8 + 0.1
    return q, k, v, g, beta


class Scan(nn.Module):
    def forward(self, q, k, v, g, beta):
        return torch.ops.karume.gated_delta_scan(q, k, v, g, beta)


class TestEagerBody:
    def test_it_is_upstreams_prefill_function_bit_for_bit(self):
        """MUST: the body is a copy of `torch_chunk_gated_delta_rule` (ADR 0135 decision 5),
        called after the head repeat with `use_qk_l2norm_in_kernel=False`. The goldens are taken
        from it, so a drift would make the exporter and the runtime agree on a wrong value."""
        upstream = pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5")
        # T = 70 spans two 64-token chunks with padding; r = 2.
        q, k, v, g, beta = _inputs(time=70)

        expected, _ = upstream.torch_chunk_gated_delta_rule(
            q.repeat_interleave(2, dim=2),
            k.repeat_interleave(2, dim=2),
            v,
            g=g,
            beta=beta,
            initial_state=None,
            output_final_state=False,
            use_qk_l2norm_in_kernel=False,
        )

        assert torch.equal(gated_delta_scan(q, k, v, g, beta), expected)

    def test_it_names_the_upstream_version_it_copies(self):
        assert GATED_DELTA_RULE_UPSTREAM.startswith("transformers 5.14.1")

    def test_it_stays_inside_the_band_of_the_runtime_fixture(self):
        """Against the f64 oracle of the fixture's stored case, within twice the larger upstream
        f32 error (the acceptance band — ADR 0135 decision 6), and next to the stored upstream
        chunked output (the same function; its last bits may depend on the CPU's kernels)."""
        with safe_open(str(FIXTURE), framework="pt") as file:
            records = {record["name"]: record for record in json.loads(file.metadata()["cases"])}
            record = records["mixed_B2_T16"]
            tensors = {
                name: file.get_tensor(f"mixed_B2_T16.{name}")
                for name in ("q", "k", "v", "g", "beta", "o_chunk", "o_oracle_head")
            }
            tail = file.get_tensor("mixed_B2_T16.o_oracle_tail")
        oracle = tensors["o_oracle_head"].double() + tail.double()

        output = gated_delta_scan(*(tensors[name] for name in ("q", "k", "v", "g", "beta")))

        error = (output.double() - oracle).abs()
        assert error.max().item() <= record["band"]["max_abs"]
        assert (error.norm() / oracle.norm()).item() <= record["band"]["rel_l2"]
        assert (output - tensors["o_chunk"]).abs().max().item() <= 1e-7

    def test_an_indivisible_head_broadcast_is_rejected(self):
        q, k, v, g, beta = _inputs(key_heads=3, value_heads=4)

        with pytest.raises(ValueError, match="multiple"):
            gated_delta_scan(q, k, v, g, beta)


class TestExport:
    def test_it_stays_one_node_with_a_symbolic_length_axis(self, convert_module, dyn_t):
        """MUST: T stays symbolic in one node — the reason the op exists (upstream's Python loops
        specialise it). The body sits behind `register_fake`, so neither the trace nor
        `run_decompositions(curated_decompositions())` enters it."""
        args = _inputs()
        graph, tensors = convert_module(Scan(), args, dynamic_shapes=({1: dyn_t},) * 5)

        assert node_ops(graph) == ["gated_delta_scan"]
        node = only_node(graph, "gated_delta_scan")
        # empty attrs: the q scale and the head broadcast are derived from the shapes
        assert node.attrs == {}
        assert node.ins == [spec.name for spec in graph.inputs]
        assert graph.values[node.outs[0]].shape == [1, "T", 4, 128]
        assert not tensors

    def test_it_round_trips_through_the_container(self, tmp_path):
        """Export → write → read back → full verification: the one node and the symbolic T are what
        the runtime reads."""
        final = tmp_path / "model.krm"

        export_to_file(
            Scan(),
            _inputs(),
            final,
            provenance=Provenance(license="mit"),
            graph_name="scan",
            dynamic_shapes=({1: torch.export.Dim("T", min=2, max=64)},) * 5,
        )

        declaration = (
            verify_container(container_parts(final), blocks=True).graphs["scan"].declaration
        )
        assert declaration["requires"]["ops"] == ["gated_delta_scan"]
        assert declaration["symbols"] == ["T"]
        assert [node["op"] for node in declaration["nodes"]] == ["gated_delta_scan"]
        assert declaration["values"][declaration["outputs"][0]]["shape"] == [1, "T", 4, 128]
        assert all(spec["shape"][1] == "T" for spec in declaration["inputs"])

    def test_a_non_f32_input_is_rejected(self, convert_module):
        """The contract is uniform f32 (a bf16 model's casts stay outside the op)."""
        args = tuple(tensor.double() for tensor in _inputs())

        with pytest.raises(NotImplementedError) as err:
            convert_module(Scan(), args)

        assert "f32" in str(err.value)

    def test_a_folded_head_axis_is_rejected(self, convert_module):
        """MUST: q / k / v are rank 4 `[B,T,H,d]`; a rank-3 g / beta. A folded head axis would
        silently scan the wrong rows, so it fails here."""
        q, k, v, g, beta = _inputs()

        class Folded(nn.Module):
            def forward(self, q, k, v, g, beta):
                return torch.ops.karume.gated_delta_scan(q.flatten(2), k, v, g, beta)

        with pytest.raises(NotImplementedError) as err:
            convert_module(Folded(), (q, k, v, g, beta))

        assert "rank" in str(err.value)

    def test_the_shape_rule_rejects_an_indivisible_head_broadcast(self, convert_module):
        """Hv % Hk ≠ 0 fails at export time (the fake does not check shapes; the shape rule does —
        `shapes._gated_delta_scan`)."""

        class Indivisible(nn.Module):
            def forward(self, q, k, v, g, beta):
                return torch.ops.karume.gated_delta_scan(
                    q, k, v[:, :, :3], g[:, :, :3], beta[:, :, :3]
                )

        with pytest.raises(OpContractError, match="multiple"):
            convert_module(Indivisible(), _inputs(key_heads=2, value_heads=4))
