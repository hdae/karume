"""`moss/export_dit.py` — gate 1 (structure) on tiny traces and its fault injections, the shared
declarations of `dit_skip22`, the bf16 checkpoint reader and the per-row i8, the write path of
`write_series` on a tiny bf16 checkpoint (with both assets of `dit`), gate 3 (the host tables) and
the container weight check of gate 2 (the real-size run is the script itself).

The pipeline code comes from the pinned snapshot ({@link moss_snapshot}; SKIP when not cached).
"""

from __future__ import annotations

import json

import pytest
import torch
from safetensors.torch import load, save_file
from torch import nn
from torch.export import Dim

from _shared.container_read import read_asset, read_asset_declarations
from karume.emit import FixedQuantizedWeight
from karume.ir import IrGraph
from karume.quantize import channel_scale, fake_quant_int8, quantize_to_int8
from moss import dit_patch, export_dit, host_fixture, records
from moss.dit_patch import DIT, DIT_SKIP22, DIT_TEXT
from moss.records import bits_equal
from moss.tests.tiny import SYM_MAX, TINY, TINY_EXPECTED, tiny_model


@pytest.fixture(scope="module")
def module(moss_snapshot):
    pytest.importorskip("diffusers")
    from moss import sources

    return sources.load_pipeline_module(moss_snapshot)


@pytest.fixture(scope="module")
def tiny(module):
    return tiny_model(module)


@pytest.fixture(scope="module")
def traced(tiny):
    return {
        name: export_dit.trace(name, tiny, TINY, sym_max=SYM_MAX, device="cpu")
        for name in export_dit.GRAPHS
    }


class TestStructureGate:
    @pytest.mark.parametrize("name", export_dit.GRAPHS)
    def test_each_graph_passes(self, traced, name):
        report = export_dit.inspect_structure(
            name, traced[name], TINY, sym_max=SYM_MAX, expected=TINY_EXPECTED[name]
        )

        assert report["failures"] == []
        assert report["symbols"] == [export_dit.symbol_of(name)]

    def test_the_real_census_is_the_plans_count(self):
        """258 / 11 linear weights (plan step 1: 42 × 6 + 6; 1 + 2 × 5)."""
        assert export_dit.EXPECTED[DIT]["linear"] == 42 * 6 + 6
        assert export_dit.EXPECTED[DIT_TEXT]["linear"] == 1 + 2 * 5
        assert export_dit.EXPECTED[DIT_SKIP22]["linear"] == 41 * 6 + 6

    def test_a_wrong_census_is_named(self, traced):
        expected = {**TINY_EXPECTED[DIT], "where": 2}

        report = export_dit.inspect_structure(
            DIT, traced[DIT], TINY, sym_max=SYM_MAX, expected=expected
        )

        assert report["failures"] == ["1 where nodes (expected 2)"]

    def test_a_baked_symbol_dependent_constant_is_named(self):
        """Fault injection: a constant that depends on S alone is folded at the symbol maximum
        (ADR 0010) — the gate must see it."""

        class Baked(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x + torch.zeros(1, x.shape[1], 4, 4)

        from karume.pipeline import export_module

        graph, _ = export_module(
            Baked(),
            (torch.randn(1, 10, 4, 4),),
            dynamic_shapes=({1: Dim("S", min=2, max=SYM_MAX)},),
            symbol_names=("S",),
        )

        assert export_dit.symbol_constants(graph, SYM_MAX) != []

    def test_the_symbol_maxima_are_no_static_extent(self):
        """Otherwise a weight of that extent reads as a baked constant (or hides one)."""
        static = {3072, 4096, 8192, 16384, 9216, 18432, 6144, 128, 256, 24}

        assert not set(export_dit.SYM_MAX.values()) & static


@pytest.fixture(scope="module")
def written(traced, tmp_path_factory):
    """The tiny `dit` written with f32 weights (the lender) and read back."""
    root = tmp_path_factory.mktemp("series")
    path = root / DIT / export_dit.MODEL_FILE
    path.parent.mkdir()
    tensors = dict(traced[DIT].tensors)
    export_dit.write_container(path, traced[DIT].graph, export_dit.Material(tensors, {}), name=DIT)
    return root, export_dit.read_graph(path, DIT)


class TestSharedSkipGraph:
    def test_every_initializer_is_borrowed_from_dit(self, traced, written):
        root, lender = written
        path = root / DIT_SKIP22 / export_dit.MODEL_FILE
        path.parent.mkdir(exist_ok=True)

        export_dit.write_container(
            path,
            export_dit.share_lender_weights(traced[DIT_SKIP22].graph, lender),
            export_dit.Material({}, {}),
            name=DIT_SKIP22,
        )
        report = export_dit.inspect_shared(export_dit.read_graph(path, DIT_SKIP22), lender)

        assert report["failures"] == []
        assert report["owned"] == 0
        assert report["shared"] == len(lender.initializers) - 10  # block 22's 10 tensors

    def test_a_tensor_the_lender_lacks_fails_loudly(self, traced, written):
        _root, lender = written
        missing = next(key for key in lender.initializers if key.startswith("layers.0."))
        initializers = {k: v for k, v in lender.initializers.items() if k != missing}
        broken = IrGraph(
            symbols=lender.symbols,
            inputs=lender.inputs,
            outputs=lender.outputs,
            initializers=initializers,
            values=lender.values,
            states=lender.states,
            nodes=lender.nodes,
        )

        with pytest.raises(export_dit.DitExportError, match="no tensor in the lender"):
            export_dit.share_lender_weights(traced[DIT_SKIP22].graph, broken)

    def test_an_owned_initializer_is_named(self, traced, written):
        _root, lender = written

        report = export_dit.inspect_shared(traced[DIT_SKIP22].graph, lender)

        assert report["owned"] > 0
        assert report["failures"]


def _checkpoint(tmp_path, tensors):
    """A one-shard diffusers-style checkpoint with an index."""
    file = "diffusion_pytorch_model-00001-of-00001.safetensors"
    save_file(tensors, str(tmp_path / file))
    index = {"metadata": {}, "weight_map": dict.fromkeys(tensors, file)}
    (tmp_path / export_dit.CHECKPOINT_INDEX).write_text(json.dumps(index), encoding="utf-8")
    return export_dit.Checkpoint(tmp_path)


class TestCheckpointAndQuantize:
    def test_the_chunked_i8_dequantizes_to_the_fake_quant_weight(self, tmp_path):
        """The container's i8 is the value gate 2 compares with (fake-quant of the same f32)."""
        generator = torch.Generator().manual_seed(0)
        weight = torch.randn(7, 5, generator=generator).to(torch.bfloat16)
        checkpoint = _checkpoint(tmp_path, {"w": weight})
        linear = nn.Linear(5, 7, bias=False)
        with torch.no_grad():
            linear.weight.copy_(weight.to(torch.float32))
        fake_quant_int8(linear)

        fixed = export_dit.quantize_rows(checkpoint, "w", chunk_rows=3)

        assert torch.equal(fixed.packed.to(torch.float32) * fixed.scale, linear.weight)

    def test_a_tensor_that_is_not_bf16_fails_loudly(self, tmp_path):
        checkpoint = _checkpoint(tmp_path, {"w": torch.zeros(2, 2)})

        with pytest.raises(export_dit.DitExportError, match="BF16 expected"):
            checkpoint.read("w")


@pytest.fixture(scope="module")
def scheduler_config(moss_snapshot, module):
    """The pinned scheduler config (the training grid of the tiny series is the real one)."""
    return host_fixture.scheduler_config(moss_snapshot, module)


@pytest.fixture(scope="module")
def series(module, scheduler_config, tmp_path_factory):
    """The tiny series written by `write_graphs` from a bf16 checkpoint of the tiny DiT, its gate 1
    reports, and the reference gate 2 compares with (the checkpoint's values in f32, fake-quant)."""
    root = tmp_path_factory.mktemp("write-path")
    reference = tiny_model(module)
    with torch.no_grad():
        for parameter in reference.parameters():
            parameter.copy_(parameter.to(torch.bfloat16))
    checkpoint_dir = root / "transformer"
    checkpoint_dir.mkdir()
    checkpoint = _checkpoint(
        checkpoint_dir,
        {
            key: value.detach().to(torch.bfloat16).contiguous()
            for key, value in reference.named_parameters()
        },
    )
    directory = root / "series"
    directory.mkdir()
    graphs = export_dit.write_graphs(
        directory,
        module,
        TINY,
        scheduler_config,
        checkpoint,
        clock=records.Clock(),
        sym_max=SYM_MAX,
        expected=TINY_EXPECTED,
    )
    fake_quant_int8(reference)
    return directory, graphs, reference


class TestWritePath:
    """`write_graphs` — the production write path of `write_series` — on a tiny bf16 checkpoint:
    meta trace → the checkpoint's weights (linears per-row i8, the rest f32) → the three containers
    (`dit` with `rope_base`) → gate 1's storage and sharing checks → every written weight is the
    fake-quant weight gate 2 compares with."""

    def test_every_graph_passes_gate_1(self, series):
        _root, graphs, _reference = series

        assert {name: report["failures"] for name, report in graphs.items()} == {
            name: [] for name in export_dit.GRAPHS
        }
        assert graphs[DIT]["i8_weights"] == TINY_EXPECTED[DIT]["linear"]
        assert graphs[DIT_TEXT]["i8_weights"] == TINY_EXPECTED[DIT_TEXT]["linear"]
        assert graphs[DIT_SKIP22]["owned"] == 0

    def test_dit_carries_the_rope_base_tables(self, module, series):
        root, _graphs, _reference = series
        path = export_dit.container_paths(root)[DIT]
        head_dim = TINY["dim"] // TINY["num_heads"]
        expected = dit_patch.rope_base_tables(module._RoPE(head_dim, TINY["rope_base"]))

        role, _length = read_asset_declarations(path)[export_dit.ROPE_BASE_ASSET]
        tables = load(read_asset(path, export_dit.ROPE_BASE_ASSET))

        assert role == export_dit.ROPE_BASE_ROLE
        assert tables.keys() == expected.keys()
        assert all(bits_equal(tables[key], expected[key]) for key in expected)

    def test_dit_carries_the_training_grid_tables(self, module, scheduler_config, series):
        """Upstream's training grid and, per grid timestep, the sinusoid upstream's forward feeds
        its time MLP (`freq_dim` of the tiny config)."""
        root, _graphs, _reference = series
        path = export_dit.container_paths(root)[DIT]
        sigmas, timesteps = host_fixture.training_grid(module, scheduler_config)
        model = host_fixture.tiny_transformer(module, TINY["freq_dim"])

        declarations = read_asset_declarations(path)
        tables = load(read_asset(path, export_dit.TRAINING_GRID_ASSET))

        assert declarations[export_dit.TRAINING_GRID_ASSET][0] == export_dit.TRAINING_GRID_ROLE
        assert sorted(tables) == ["sigmas", "time_sinusoid", "timesteps"]
        assert bits_equal(tables["sigmas"], sigmas)
        assert bits_equal(tables["timesteps"], timesteps)
        assert tables["time_sinusoid"].shape == (1000, TINY["freq_dim"])
        for row in (0, 1, 500, 999):
            expected = host_fixture.time_sinusoid(model, timesteps[row])
            assert bits_equal(tables["time_sinusoid"][row : row + 1], expected), row

    def test_rope_base_keeps_the_first_asset_block(self, series):
        """The assets are laid out in the order passed: `rope_base` stays block `a.0`, so adding
        `training_grid` leaves its bytes where they were."""
        from _shared.container_read import open_container

        root, _graphs, _reference = series
        model = open_container(export_dit.container_paths(root)[DIT]).read.model

        assert model.assets[export_dit.ROPE_BASE_ASSET].block == "a.0"
        assert model.assets[export_dit.TRAINING_GRID_ASSET].block == "a.1"

    @pytest.mark.parametrize("name", [DIT, DIT_TEXT])
    def test_the_written_weights_are_the_fake_quant_weights(self, series, name):
        root, _graphs, reference = series

        report = export_dit.container_weights(export_dit.container_paths(root)[name], reference)

        assert report["differing"] == []
        assert report["counts"]["i8"] == TINY_EXPECTED[name]["linear"]


class TestHostTablesGate:
    """Gate 3 against host records written the way `moss.host_fixture` writes them."""

    @staticmethod
    def _record(path, tables):
        records.write_record(
            path,
            {key: tables[name] for name, key in export_dit.TRAINING_GRID_RECORD_KEYS.items()},
            {"note": "test"},
            key=host_fixture.METADATA_KEY,
        )

    @pytest.fixture
    def dit_and_tables(self, series):
        root, _graphs, _reference = series
        path = export_dit.container_paths(root)[DIT]
        return path, load(read_asset(path, export_dit.TRAINING_GRID_ASSET))

    def test_the_tables_of_the_record_pass(self, dit_and_tables, tmp_path):
        path, tables = dit_and_tables
        self._record(tmp_path / "host.safetensors", tables)

        report = export_dit.host_tables_gate(path, tmp_path / "host.safetensors")

        assert report["failures"] == []
        assert report["tables"]["time_sinusoid"] == [1000, TINY["freq_dim"]]

    def test_a_sinusoid_one_ulp_off_is_named(self, dit_and_tables, tmp_path):
        path, tables = dit_and_tables
        moved = dict(tables)
        moved["time_sinusoid"] = tables["time_sinusoid"].clone()
        value = moved["time_sinusoid"][7, 3]
        moved["time_sinusoid"][7, 3] = torch.nextafter(value, torch.tensor(2.0))
        self._record(tmp_path / "host.safetensors", moved)

        report = export_dit.host_tables_gate(path, tmp_path / "host.safetensors")

        assert report["failures"] == [
            "training_grid 'time_sinusoid' differs from the host record's 'time_sinusoid'"
        ]

    def test_a_record_without_a_table_is_named(self, dit_and_tables, tmp_path):
        path, tables = dit_and_tables
        keys = export_dit.TRAINING_GRID_RECORD_KEYS
        records.write_record(
            tmp_path / "host.safetensors",
            {key: tables[name] for name, key in keys.items() if name != "timesteps"},
            {"note": "test"},
            key=host_fixture.METADATA_KEY,
        )

        report = export_dit.host_tables_gate(path, tmp_path / "host.safetensors")

        assert report["failures"] == [f"the host record has no '{keys['timesteps']}'"]

    def test_a_container_without_a_table_is_named(self, dit_and_tables, tmp_path, monkeypatch):
        """Fault injection: the asset read back lacks `timesteps`."""
        path, tables = dit_and_tables
        self._record(tmp_path / "host.safetensors", tables)
        monkeypatch.setattr(
            export_dit,
            "load",
            lambda raw: {name: value for name, value in load(raw).items() if name != "timesteps"},
        )

        report = export_dit.host_tables_gate(path, tmp_path / "host.safetensors")

        assert report["failures"] == ["training_grid tables ['sigmas', 'time_sinusoid']"]

    def test_the_host_record_option_is_refused_outside_write(self, capsys, tmp_path):
        # `--out`: were the option accepted, the summary would land in tmp_path.
        with pytest.raises(SystemExit):
            export_dit.main(
                ["inspect", "--host-record", "host.safetensors", "--out", str(tmp_path)]
            )

        assert "--host-record applies to write only" in capsys.readouterr().err

    def test_a_missing_record_fails_loudly(self, dit_and_tables, tmp_path):
        path, _tables = dit_and_tables

        with pytest.raises(export_dit.DitExportError, match="no stage-0 host record"):
            export_dit.host_tables_gate(path, tmp_path / "host.safetensors")


class TestContainerWeights:
    @pytest.fixture
    def model_and_container(self, module, tmp_path):
        model = tiny_model(module)
        traced = export_dit.trace(DIT_TEXT, model, TINY, sym_max=SYM_MAX, device="cpu")
        fake_quant_int8(model)
        parameters = dict(model.named_parameters())
        fixed = {}
        for key in traced.targets:
            weight = parameters[key].detach()
            scale = channel_scale(weight, 0)
            fixed[key] = FixedQuantizedWeight("i8", quantize_to_int8(weight, scale), scale)
        # The targets' bytes are `fixed`; their logical tensors are meta (the emit contract).
        tensors = {
            key: torch.empty_like(value, device="meta") if key in fixed else value.detach()
            for key, value in traced.tensors.items()
        }
        path = tmp_path / DIT_TEXT / export_dit.MODEL_FILE
        path.parent.mkdir()
        export_dit.write_container(
            path, traced.graph, export_dit.Material(tensors, fixed), name=DIT_TEXT
        )
        return model, path

    def test_the_written_weights_are_the_compared_weights(self, model_and_container):
        model, path = model_and_container

        report = export_dit.container_weights(path, model)

        assert report["differing"] == []
        assert report["counts"]["i8"] == 11

    def test_a_weight_one_ulp_off_is_named(self, model_and_container):
        model, path = model_and_container
        weight = model.text_embedder.blocks[1].ffn_down.weight
        with torch.no_grad():
            weight.view(-1)[0] = torch.nextafter(weight.view(-1)[0], torch.tensor(float("inf")))

        report = export_dit.container_weights(path, model)

        assert report["differing"] == ["text_embedder.blocks.1.ffn_down.weight"]
