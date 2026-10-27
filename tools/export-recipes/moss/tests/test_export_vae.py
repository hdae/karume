"""`moss/export_vae.py` on the tiny VAE of `test_vae_patch.py`: gate 1 (the census against the
module structure, the declared inputs and outputs, a baked symbol-dependent constant, the f32
storage and the disjoint cover of the decoder's parameters) and its fault injections, the write
path of the three containers with the asset `latent_batch_norm`, gate 3 (the host tables) and the
eager / weight checks of gate 2 with their fault injections (the real-size run is the script
itself)."""

from __future__ import annotations

import dataclasses

import pytest
import torch
from safetensors.torch import load
from torch import nn
from torch.export import Dim

from _shared.container_read import read_asset, read_asset_declarations
from karume.ir import IrStorage
from karume.pipeline import export_module
from moss import export_vae, vae_patch, vae_reference
from moss.export_vae import BATCH_NORM_ASSET, BATCH_NORM_ROLE, VaeCase
from moss.tests.test_vae_patch import tiny_vae
from moss.vae_patch import GRAPHS, VAE_DECODER_ATTN, VAE_DECODER_POST, VAE_DECODER_PRE

#: Small trace bounds for the tiny model (no tiny extent equals them).
BOUNDS = {"latent_max": 64, "sequence_max": 4096}


@pytest.fixture(scope="module")
def vae():
    return tiny_vae()


@pytest.fixture(scope="module")
def traced_graphs(vae):
    """The three tiny graphs as traced (their initializers carry the decoder's tensor keys)."""
    modules = export_vae.graph_modules(vae)
    return {name: export_vae.trace(name, modules[name], vae, **BOUNDS).graph for name in GRAPHS}


@pytest.fixture(scope="module")
def written(vae, tmp_path_factory):
    directory = tmp_path_factory.mktemp("series")
    report = export_vae.write_graphs(directory, vae, **BOUNDS)
    return directory, report


class TestStructure:
    def test_the_tiny_graphs_pass_gate_1_with_the_structure_census(self, vae):
        modules = export_vae.graph_modules(vae)
        expected = {
            VAE_DECODER_PRE: {
                "add": 1,
                "conv2d": 4,
                "div": 1,
                "group_norm": 2,
                "mul": 2,
                "sigmoid": 2,
            },
            VAE_DECODER_ATTN: {
                "add": 1,
                "attention": 1,
                "div": 1,
                "group_norm": 1,
                "linear": 4,
                "permute": 6,
                "reshape": 4,
            },
            # 13 resnets (mid resnet 1 + 4 × 3), 2 shortcuts, 3 upsamples (2 expand + 4 reshape
            # each), conv_out; 27 group norms and 27 SiLUs (2 per resnet + conv_act).
            VAE_DECODER_POST: {
                "add": 13,
                "conv2d": 32,
                "div": 13,
                "expand": 6,
                "group_norm": 27,
                "mul": 27,
                "reshape": 12,
                "sigmoid": 27,
            },
        }
        for name in GRAPHS:
            traced = export_vae.trace(name, modules[name], vae, **BOUNDS)
            report = export_vae.inspect_structure(
                name, traced, modules[name], vae, bounds=list(BOUNDS.values())
            )
            assert report["failures"] == []
            assert report["ops"] == expected[name]
        assert export_vae.declared_io(VAE_DECODER_POST, vae) == (
            ["hidden", [1, 16, "h", "w"]],
            [1, 3, "8h", "8w"],
        )

    def test_a_group_norm_run_as_another_op_fails_the_census(self, vae):
        """The GroupNorm module is still there but its forward runs another normalisation (a
        layer_norm over the channels) — the census against the structure turns red, as it does for
        any GroupNorm that does not reach the IR as one `group_norm` node."""

        class OtherNorm(nn.Module):
            def __init__(self, norm: nn.GroupNorm) -> None:
                super().__init__()
                self.norm = norm

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                channels = self.norm.num_channels
                normed = nn.functional.layer_norm(x.transpose(1, 2), [channels], eps=self.norm.eps)
                return normed.transpose(1, 2)

        module = export_vae.graph_modules(vae)[VAE_DECODER_ATTN]
        attention = module.decoder.mid_block.attentions[0]
        original = attention.group_norm
        attention.group_norm = OtherNorm(original)
        try:
            traced = export_vae.trace(VAE_DECODER_ATTN, module, vae, **BOUNDS)
            report = export_vae.inspect_structure(
                VAE_DECODER_ATTN, traced, module, vae, bounds=list(BOUNDS.values())
            )
        finally:
            attention.group_norm = original
        assert any("census" in failure for failure in report["failures"])

    def test_a_baked_symbol_dependent_constant_is_named(self):
        """Fault injection: a constant that depends on h, w alone is folded at the symbol maximum
        (ADR 0010) — gate 1 must name it (the initializer has a dimension equal to the bound). One
        symbol only: a folded constant over two symbols already fails the export."""

        class Baked(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x + torch.zeros(1, 1, x.shape[2], 1)

        bound = BOUNDS["latent_max"]
        graph, _ = export_module(
            Baked(),
            (torch.randn(1, 4, 8, 6),),
            dynamic_shapes=({2: Dim("h", min=2, max=bound), 3: Dim("w", min=2, max=bound)},),
            symbol_names=("h", "w"),
        )
        found = export_vae.symbol_constants(graph, list(BOUNDS.values()))
        assert any("symbol bound" in entry for entry in found)


class TestStorage:
    """`inspect_storage` (gate 1's storage items) on the tiny graphs and its fault injections."""

    def test_the_tiny_graphs_are_f32_disjoint_and_cover_the_decoder(self, vae, traced_graphs):
        report = export_vae.inspect_storage(traced_graphs, vae)
        assert report["failures"] == []
        assert sum(report["weights"].values()) == report["parameters"]

    @staticmethod
    def _with(graph, initializers):
        return dataclasses.replace(graph, initializers=initializers)

    def test_a_storage_other_than_f32_is_named(self, vae, traced_graphs):
        graph = traced_graphs[VAE_DECODER_POST]
        name, init = next(iter(graph.initializers.items()))
        changed = {**graph.initializers, name: dataclasses.replace(init, storage=IrStorage("f16"))}
        graphs = {**traced_graphs, VAE_DECODER_POST: self._with(graph, changed)}
        failures = export_vae.inspect_storage(graphs, vae)["failures"]
        assert failures == [f"{VAE_DECODER_POST}: storage other than f32 on {[init.tensor]}"]

    def test_a_key_in_two_graphs_is_named(self, vae, traced_graphs):
        lender = next(iter(traced_graphs[VAE_DECODER_PRE].initializers.values()))
        graph = traced_graphs[VAE_DECODER_ATTN]
        changed = {**graph.initializers, "duplicate": lender}
        graphs = {**traced_graphs, VAE_DECODER_ATTN: self._with(graph, changed)}
        failures = export_vae.inspect_storage(graphs, vae)["failures"]
        assert failures == [
            f"{VAE_DECODER_ATTN}: tensor keys shared with another graph {[lender.tensor]}"
        ]

    def test_a_parameter_in_no_graph_is_named(self, vae, traced_graphs):
        graph = traced_graphs[VAE_DECODER_POST]
        name, init = next(iter(graph.initializers.items()))
        changed = {key: value for key, value in graph.initializers.items() if key != name}
        graphs = {**traced_graphs, VAE_DECODER_POST: self._with(graph, changed)}
        failures = export_vae.inspect_storage(graphs, vae)["failures"]
        assert failures == [f"decoder parameters in no graph: {[init.tensor]}"]


class TestWrite:
    def test_the_containers_hold_f32_weights_bit_for_bit_and_the_asset(self, vae, written):
        directory, report = written
        assert report["failures"] == {}
        paths = export_vae.container_paths(directory)
        for name in GRAPHS:
            checked = export_vae.container_weights(paths[name], vae)
            assert checked["differing"] == []
            assert checked["counts"]["f32"] > 0
        assert (
            read_asset_declarations(paths[VAE_DECODER_PRE])[BATCH_NORM_ASSET][0] == BATCH_NORM_ROLE
        )
        tables = load(read_asset(paths[VAE_DECODER_PRE], BATCH_NORM_ASSET))
        expected = export_vae.batch_norm_tables(vae)
        assert set(tables) == {"mean", "std"}
        for key in ("mean", "std"):
            assert torch.equal(tables[key], expected[key])
        assert report["storage"]["weights"][VAE_DECODER_ATTN] > 0

    def test_a_changed_weight_is_reported(self, vae, written):
        directory, _ = written
        path = export_vae.container_paths(directory)[VAE_DECODER_POST]
        other = tiny_vae()
        with torch.no_grad():
            other.decoder.conv_out.bias.add_(1e-3)
        assert export_vae.container_weights(path, other)["differing"] == ["decoder.conv_out.bias"]


@pytest.fixture(scope="module")
def material(vae):
    """A golden and a stage-0 record of the tiny VAE (its 16 DiT-space channels — the real
    `case_tensors` draws the real model's 128)."""
    from moss import pipeline_ref

    latents = torch.randn(1, 16, 3, 2, generator=torch.Generator().manual_seed(6))
    denormalized = pipeline_ref.denormalize(vae, latents)
    z = pipeline_ref.unpatchify(denormalized)
    with torch.no_grad():
        decoded = vae_patch.decode_reference(vae, z)
    image = decoded.clamp(-1, 1)
    case = VaeCase(48, 32, vae_reference.SEED, "accept")
    goldens = {case: {"latents": latents, "z": z, "decoded": decoded, "image": image}}
    record = {
        "latents": latents,
        "denormalized": denormalized,
        "vae.z": z,
        "vae.decoded": decoded,
        "image": image,
    }
    return goldens, {(48, 32): record}


class TestHostTables:
    def test_the_asset_reproduces_upstreams_restoration_and_the_record(self, vae, material):
        goldens, stage0 = material
        tables = export_vae.batch_norm_tables(vae)
        assert export_vae.host_tables_gate(tables, vae, goldens, stage0)["failures"] == []

    def test_an_asset_off_by_one_ulp_fails(self, vae, material):
        goldens, stage0 = material
        tables = export_vae.batch_norm_tables(vae)
        bits = tables["std"].view(torch.int32).clone()
        bits[3] += 1
        broken = {"mean": tables["mean"], "std": bits.view(torch.float32)}
        failures = export_vae.host_tables_gate(broken, vae, goldens, stage0)["failures"]
        assert any("denormalize" in failure for failure in failures)

    def test_a_golden_that_is_not_the_record_fails(self, vae, material):
        goldens, stage0 = material
        (case,) = goldens
        moved = {case: {**goldens[case], "decoded": goldens[case]["decoded"] + 1e-3}}
        failures = export_vae.host_tables_gate(
            export_vae.batch_norm_tables(vae), vae, moved, stage0
        )["failures"]
        assert any("'decoded'" in failure for failure in failures)


class TestEager:
    def test_the_split_decode_passes_and_a_moved_graph_fails(self, vae):
        z = torch.randn(1, 4, 4, 6, generator=torch.Generator().manual_seed(5))
        graphs = export_vae.graph_modules(vae)
        assert export_vae.eager_case(vae, graphs, z)["failures"] == []

        class Nudged(nn.Module):
            def __init__(self, inner: nn.Module) -> None:
                super().__init__()
                self.inner = inner

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.nextafter(self.inner(x), torch.tensor(float("inf")))

        moved = {**graphs, VAE_DECODER_POST: Nudged(graphs[VAE_DECODER_POST])}
        failures = export_vae.eager_case(vae, moved, z)["failures"]
        assert len(failures) == 1 and failures[0].startswith("decode")


class TestCases:
    def test_the_cases_split_band_and_accept_per_size_with_the_stage0_seed(self):
        for height, width in export_vae.SIZES:
            roles = [c.role for c in export_vae.CASES if (c.height, c.width) == (height, width)]
            assert roles == ["band"] * 3 + ["accept"] * 2
        assert {c.seed for c in export_vae.CASES if c.role == "accept"} >= {vae_reference.SEED}
        assert export_vae.CASES_WITH_CAP[-1] == VaeCase(1696, 1648, export_vae.CAP_SEED, "cap")
        # The cap case's largest tensor [192, H, W] f32 fits a 2 GiB binding, the next row does not.
        assert 192 * 4 * 1696 * 1648 <= 2**31 < 192 * 4 * 1696 * (1648 + 16)
        assert vae_patch.GRAPHS == GRAPHS
