"""`moss/dit_reference.py` on the tiny DiT: the container reader returns the fake-quant weights,
the layer-by-layer runs equal the eager graphs (f32 and f64, skip 22 included) with one block alive
at a time, the f64 guard catches a narrowing, and the writer's goldens have the agreed form and
resume. The real-size cases are the script itself.

The pipeline code comes from the pinned snapshot ({@link moss_snapshot}; SKIP when not cached).
"""

from __future__ import annotations

import copy
import hashlib
import json
from contextlib import nullcontext

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from karume.container import container_parts
from moss import dit_patch, dit_reference, export_dit, host_fixture, records
from moss.dit_patch import DIT, DIT_SKIP22, DIT_TEXT, SKIP_BLOCK
from moss.dit_reference import ACCEPT, BAND, DitCase, TextCase
from moss.prompts import NEGATIVE, POSITIVE
from moss.records import bits_equal
from moss.tests.tiny import SYM_MAX, TINY, TINY_EXPECTED, tiny_model


@pytest.fixture(scope="module")
def module(moss_snapshot):
    pytest.importorskip("diffusers")
    from moss import sources

    return sources.load_pipeline_module(moss_snapshot)


@pytest.fixture(scope="module")
def series(module, moss_snapshot, tmp_path_factory):
    """The tiny series (`export_dit.write_graphs` on a bf16 checkpoint of the tiny DiT) and the
    tiny DiT with the weights the series stores (bf16 values, fake-quant)."""
    root = tmp_path_factory.mktemp("reference-series")
    reference = tiny_model(module)
    with torch.no_grad():
        for parameter in reference.parameters():
            parameter.copy_(parameter.to(torch.bfloat16))
    checkpoint_dir = root / "transformer"
    checkpoint_dir.mkdir()
    tensors = {
        key: value.detach().to(torch.bfloat16).contiguous()
        for key, value in reference.named_parameters()
    }
    file = "diffusion_pytorch_model-00001-of-00001.safetensors"
    save_file(tensors, str(checkpoint_dir / file))
    index = {"metadata": {}, "weight_map": dict.fromkeys(tensors, file)}
    (checkpoint_dir / export_dit.CHECKPOINT_INDEX).write_text(json.dumps(index), encoding="utf-8")
    directory = root / "series"
    directory.mkdir()
    export_dit.write_graphs(
        directory,
        module,
        TINY,
        host_fixture.scheduler_config(moss_snapshot, module),
        export_dit.Checkpoint(checkpoint_dir),
        clock=records.Clock(),
        sym_max=SYM_MAX,
        expected=TINY_EXPECTED,
    )
    from karume.quantize import fake_quant_int8

    fake_quant_int8(reference)
    return directory, reference


@pytest.fixture(scope="module")
def embeds():
    """Tiny stand-ins of the stage-0 embeddings (`[T, text_dim]`)."""
    generator = torch.Generator().manual_seed(1)
    return {
        "fridge-note": torch.randn(5, TINY["text_dim"], generator=generator),
        "negative": torch.randn(3, TINY["text_dim"], generator=generator),
    }


@pytest.fixture(scope="module")
def writer(module, series, embeds):
    directory, _reference = series
    return dit_reference.open_writer(directory, embeds, module, TINY)


def _latents(case):
    generator = torch.Generator().manual_seed(case.seed)
    height, width = 4, 6
    return torch.randn(1, TINY["latent_channels"], height, width, generator=generator)


def _dit_inputs(writer, embeds, grid_index=7):
    generator = torch.Generator().manual_seed(2)
    latents = torch.randn(1, TINY["latent_channels"], 4, 6, generator=generator)
    text = writer.text_context("fridge-note")
    return dit_reference.dit_inputs(writer.assets, latents, text, grid_index)


class TestContainerWeights:
    def test_every_weight_is_the_fake_quant_weight(self, series):
        directory, reference = series
        weights = dit_reference.ContainerWeights(export_dit.container_paths(directory)[DIT])
        parameters = dict(reference.named_parameters())

        keys = sorted(weights.names() & set(parameters))

        assert len(keys) > 100
        assert all(bits_equal(weights.tensor(key), parameters[key].detach()) for key in keys)

    def test_an_unknown_key_fails_loudly(self, series):
        directory, _reference = series
        weights = dit_reference.ContainerWeights(export_dit.container_paths(directory)[DIT_TEXT])

        with pytest.raises(dit_reference.DitReferenceError, match="no tensor"):
            weights.tensor("layers.0.proj.q.weight")


class TestLayerwise:
    """The layer-by-layer modules equal the eager graphs on the series' fake-quant weights."""

    @pytest.mark.parametrize("skip", [False, True])
    def test_the_f32_dit_is_the_eager_graph(self, writer, series, embeds, skip):
        _directory, reference = series
        inputs = _dit_inputs(writer, embeds)
        skip_layers = frozenset({SKIP_BLOCK}) if skip else frozenset()

        actual = writer.layerwise.dit(torch.float32, inputs, skip=skip)
        with torch.no_grad():
            expected = dit_patch.MossDit(reference, skip_layers=skip_layers)(*inputs)

        assert bits_equal(actual, expected)
        opened = writer.layerwise.last_layers.opened
        assert opened == [i for i in range(TINY["num_layers"]) if not skip or i != SKIP_BLOCK]

    def test_one_block_is_alive_at_a_time(self, writer, embeds):
        writer.layerwise.dit(torch.float32, _dit_inputs(writer, embeds), skip=False)

        assert writer.layerwise.last_layers.peak_live == 1

    @pytest.mark.parametrize("skip", [False, True])
    def test_the_f64_dit_is_the_eager_graph_in_f64(self, writer, series, embeds, skip):
        _directory, reference = series
        inputs = _dit_inputs(writer, embeds)
        wide = copy.deepcopy(reference).double()
        skip_layers = frozenset({SKIP_BLOCK}) if skip else frozenset()

        actual = writer.layerwise.dit(torch.float64, inputs, skip=skip)
        with torch.no_grad(), dit_reference.float64_forward():
            expected = dit_patch.MossDit(wide, skip_layers=skip_layers)(
                *(v.double() if v.is_floating_point() else v for v in inputs)
            )

        assert actual.dtype == torch.float64
        assert bits_equal(actual, expected)

    def test_the_skip_changes_the_output(self, writer, embeds):
        inputs = _dit_inputs(writer, embeds)

        full = writer.layerwise.dit(torch.float32, inputs, skip=False)
        skipped = writer.layerwise.dit(torch.float32, inputs, skip=True)

        assert not torch.equal(full, skipped)

    def test_the_f64_run_moves_off_the_f32_run_by_rounding_only(self, writer, embeds):
        inputs = _dit_inputs(writer, embeds)

        ratio = dit_reference.ratio(
            writer.layerwise.dit(torch.float32, inputs, skip=False),
            writer.layerwise.dit(torch.float64, inputs, skip=False),
        )

        assert 0 < ratio < 1e-4

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_the_text_graph_is_the_eager_graph(self, writer, series, embeds, dtype):
        _directory, reference = series
        inputs = dit_reference.text_inputs(writer.assets, embeds["fridge-note"])
        model = reference if dtype == torch.float32 else copy.deepcopy(reference).double()

        actual = writer.layerwise.text(dtype, inputs)
        guard = dit_reference.float64_forward() if dtype == torch.float64 else nullcontext()
        with torch.no_grad(), guard:
            expected = dit_patch.MossDitText(model)(*(v.to(dtype) for v in inputs))

        assert bits_equal(actual, expected)

    def test_the_text_rope_is_the_text_rows_of_the_joint_table(self, writer, embeds):
        """`dit_text`'s positions are the joint sequence's text positions, whatever the image."""
        text = embeds["fridge-note"]
        joint = dit_patch.rope_table(writer.assets.rope_base, 3, 5, int(text.shape[0]))

        _embeds, rope_cos, rope_sin = dit_reference.text_inputs(writer.assets, text)
        expected_cos, expected_sin = dit_patch.split_rope(joint[:, 3 * 5 :])

        assert bits_equal(rope_cos, expected_cos)
        assert bits_equal(rope_sin, expected_sin)

    def test_a_narrowing_in_the_f64_run_fails_loudly(self, writer, embeds, module, monkeypatch):
        """Fault injection: an f32 cast inside a block (one upstream `.float()` would do the same
        without the guard)."""
        situ = module._situ
        monkeypatch.setattr(
            module, "_situ", lambda hidden, gate: situ(hidden.to(torch.float32), gate)
        )

        with pytest.raises(dit_reference.DitReferenceError, match="non-f64 value"):
            writer.layerwise.dit(torch.float64, _dit_inputs(writer, embeds), skip=False)

    def test_the_float_cast_is_restored(self, writer, embeds):
        before = torch.Tensor.float

        writer.layerwise.dit(torch.float64, _dit_inputs(writer, embeds), skip=True)

        assert torch.Tensor.float is before
        assert torch.ones(1, dtype=torch.float64).float().dtype == torch.float32


CASES = (
    DitCase("small", BAND, POSITIVE, 3),
    DitCase("small", ACCEPT, NEGATIVE, 3),
    TextCase(ACCEPT, "fridge-note", 1),
)


@pytest.fixture(scope="module")
def written(module, series, embeds, tmp_path_factory):
    directory, _reference = series
    out = tmp_path_factory.mktemp("goldens")
    writer = dit_reference.open_writer(directory, embeds, module, TINY)
    writer.latents = _latents
    summary = dit_reference.write(CASES, out=out, writer=writer)
    return out, summary, writer


def _stored_inputs(writer, tensors):
    """A DiT golden's inputs rebuilt from its stored latents, text context and grid index."""
    index = int(tensors["grid_index"][0])
    return dit_reference.dit_inputs(
        writer.assets, tensors["latents"], tensors["text_context"], index
    )


def _assert_two_parts(tensors, output64):
    """`output.f64` / `output.f64.lo` are the split of the f64 run, bit for bit."""
    high, low = dit_reference.split_float64(output64)
    assert bits_equal(tensors["output.f64"], high)
    assert bits_equal(tensors["output.f64.lo"], low)


class TestGoldens:
    def _read(self, path):
        with safe_open(str(path), framework="pt") as handle:
            metadata = json.loads(handle.metadata()[dit_reference.METADATA_KEY])
        return load_file(str(path)), metadata

    def test_the_goldens_name_every_container_by_its_part0_digest(self, written, series):
        out, _summary, _writer = written
        directory, _reference = series
        paths = export_dit.container_paths(directory)

        _tensors, metadata = self._read(dit_reference.golden_path(out, CASES[0]))

        assert metadata["containers"] == {
            name: hashlib.sha256(container_parts(paths[name])[0].read_bytes()).hexdigest()
            for name in (DIT, DIT_SKIP22, DIT_TEXT)
        }

    @pytest.mark.parametrize("index", [0, 1])
    def test_a_dit_golden_holds_the_f64_run_of_its_graph(self, written, index):
        """Fault injection guard: an `output.f64` from the f32 run, or a negative golden computed
        without the skip, would not be the f64 run of the case's own graph."""
        out, _summary, writer = written
        case = CASES[index]
        tensors, metadata = self._read(dit_reference.golden_path(out, case))
        inputs = _stored_inputs(writer, tensors)
        cells = tuple(tensors["latents"].shape[2:])

        rows64 = writer.layerwise.dit(torch.float64, inputs, skip=case.graph == DIT_SKIP22)

        _assert_two_parts(tensors, dit_patch.head_rows(rows64, *cells))
        assert not torch.equal(tensors["output"], tensors["output.f64"])
        assert 0 < metadata["f32_ratio"] < 1e-4

    def test_a_dit_golden_carries_its_inputs_outputs_and_containers(self, written):
        out, _summary, writer = written
        case = CASES[0]

        tensors, metadata = self._read(dit_reference.golden_path(out, case))

        assert metadata["version"] == dit_reference.WRITER_VERSION
        assert metadata["containers"] == writer.containers
        assert metadata["graph"] == DIT and metadata["role"] == BAND
        assert (metadata["N"], metadata["T"], metadata["S"]) == (24, 5, 29)
        assert tensors["output"].shape == tensors["output.f64"].shape == (1, 8, 4, 6)
        assert tensors["output.rows"].shape == (1, 29, 8)
        assert bits_equal(dit_patch.head_rows(tensors["output.rows"], 4, 6), tensors["output"])
        assert tensors["grid_index"].tolist() == [3]
        assert bits_equal(
            tensors["timestep"], writer.assets.training_grid["timesteps"][3:4].contiguous()
        )
        assert sorted(k for k in tensors if k.startswith("input.")) == sorted(
            f"input.{name}" for name in dit_patch.DIT_INPUTS
        )

    def test_the_stored_inputs_are_the_host_functions_of_the_stored_latents(self, written):
        out, _summary, writer = written

        tensors, _metadata = self._read(dit_reference.golden_path(out, CASES[1]))
        inputs = dit_reference.dit_inputs(
            writer.assets, tensors["latents"], tensors["text_context"], 3
        )

        for name, value in zip(dit_patch.DIT_INPUTS, inputs, strict=True):
            assert bits_equal(tensors[f"input.{name}"], value.contiguous()), name
        # Not through `dit_reference.dit_inputs` (the code under test): the training grid's row of
        # the case's index and the RoPE table of the stored sizes.
        sinusoid = writer.assets.training_grid["time_sinusoid"][3:4]
        assert bits_equal(tensors["input.timesteps_proj"], sinusoid)
        height, width = tensors["latents"].shape[2:]
        tokens = int(tensors["text_context"].shape[1])
        table = dit_patch.rope_table(writer.assets.rope_base, height, width, tokens)
        rope_cos, rope_sin = dit_patch.split_rope(table)
        assert bits_equal(tensors["input.rope_cos"], rope_cos)
        assert bits_equal(tensors["input.rope_sin"], rope_sin)

    def test_a_negative_golden_is_the_skip_graph(self, written):
        out, _summary, writer = written

        tensors, metadata = self._read(dit_reference.golden_path(out, CASES[1]))
        inputs = _stored_inputs(writer, tensors)

        assert metadata["graph"] == DIT_SKIP22
        assert metadata["T"] == 3
        skipped = writer.layerwise.dit(torch.float32, inputs, skip=True)
        assert bits_equal(tensors["output.rows"], skipped)
        full = writer.layerwise.dit(torch.float32, inputs, skip=False)
        assert not torch.equal(tensors["output.rows"], full)

    def test_a_text_golden_at_one_token(self, written):
        out, _summary, writer = written

        tensors, metadata = self._read(dit_reference.golden_path(out, CASES[2]))

        assert metadata["graph"] == DIT_TEXT and metadata["T"] == 1
        assert tensors["text_embeds"].shape == (1, 1, TINY["text_dim"])
        assert tensors["output"].shape == tensors["output.f64"].shape == (1, 1, TINY["dim"])
        assert 0 < metadata["f32_ratio"] < 1e-4
        inputs = (tensors["text_embeds"], tensors["rope_cos"], tensors["rope_sin"])
        _assert_two_parts(tensors, writer.layerwise.text(torch.float64, inputs))

    def test_a_second_run_finds_every_golden_current(self, written, module, series, embeds):
        out, _summary, _writer = written
        directory, _reference = series
        writer = dit_reference.open_writer(directory, embeds, module, TINY)
        writer.latents = _latents

        summary = dit_reference.write(CASES, out=out, writer=writer)

        assert [row["status"] for row in summary["cases"]] == ["current"] * len(CASES)

    def test_a_golden_of_other_containers_is_redone(self, written, module, series, embeds):
        out, _summary, _writer = written
        directory, _reference = series
        writer = dit_reference.open_writer(directory, embeds, module, TINY)
        writer.latents = _latents
        writer.containers = {**writer.containers, DIT: "0" * 64}

        summary = dit_reference.write(CASES[2:], out=out, writer=writer)

        assert [row["status"] for row in summary["cases"]] == ["written"]


class TestSplitFloat64:
    def test_the_two_parts_sum_to_the_value_within_two_to_the_minus_48(self):
        value = torch.tensor([1800.123456789012345, -3.0e-5, 1.0 / 3.0, 0.0], dtype=torch.float64)

        high, low = dit_reference.split_float64(value)

        assert high.dtype == low.dtype == torch.float32
        assert bits_equal(high, value.to(torch.float32))
        error = (high.double() + low.double() - value).abs()
        assert bool((error <= value.abs() * 2.0**-48).all())
        assert bool((low != 0).any())

    def test_an_f32_value_is_refused(self):
        with pytest.raises(dit_reference.DitReferenceError, match="takes f64"):
            dit_reference.split_float64(torch.ones(2))


class TestCaseTable:
    def test_select_names_an_unknown_case(self):
        with pytest.raises(dit_reference.DitReferenceError, match=r"unknown cases \['nope'\]"):
            dit_reference.select(dit_reference.CASES, [], ["nope"])

    def test_select_by_group_and_name(self):
        text = dit_reference.select(dit_reference.CASES, ["text"], [])
        one = dit_reference.select(dit_reference.CASES, [], [dit_reference.CASES[-1].name])

        assert text == list(dit_reference.TEXT_CASES)
        assert one == [dit_reference.CASES[-1]]

    def test_list_prints_one_row_per_selected_case(self, capsys):
        assert dit_reference.main(["list", "--group", "text"]) == 0

        rows = capsys.readouterr().out.splitlines()
        assert [row.split("\t")[0] for row in rows] == [c.name for c in dit_reference.TEXT_CASES]

    def test_six_band_and_four_acceptance_cases_per_group(self):
        for group in dit_reference.GROUPS:
            cases = [case for case in dit_reference.CASES if case.group == group]
            roles = [case.role for case in cases]
            assert (roles.count(BAND), roles.count(ACCEPT)) == (6, 4), group

    def test_both_branches_in_each_dit_group_and_role(self):
        for case_group in dit_reference.DIT_GROUPS:
            for role in (BAND, ACCEPT):
                branches = [
                    case.branch
                    for case in dit_reference.DIT_CASES
                    if case.group == case_group and case.role == role
                ]
                assert branches.count(POSITIVE) == branches.count(NEGATIVE), (case_group, role)

    def test_the_text_cases_include_one_token_as_an_acceptance_case(self):
        """ADR 0138 decision 3 MUST: `dit_text` was traced with T ≥ 2."""
        assert any(c.length == 1 and c.role == ACCEPT for c in dit_reference.TEXT_CASES)

    def test_case_names_are_unique(self):
        names = [case.name for case in dit_reference.CASES]

        assert len(names) == len(set(names))
