"""The contract of `wan/dit_reference.py` (the layer-by-layer f64 / f32 reference of the DiT) and
`wan/dit_probe.py` (the RAM probe for the 5B) (ADR 0121 decision 7, step 0).

It pins the side where a breakage raises no exception and **only the numbers drift silently**:

- layer-by-layer f32 / f64 is bit-identical to the reference run on the whole upstream model (f32
  is `reference_dit_layers`, f64 is `model.double()` + `reference_dit_f64` — how the Wan2.1 goldens
  are made) (with the rounding of both the f16 and the i8 series, and with all 3 sources: the
  in-memory model, the container and the checkpoint)
- the i8 dequantization of the container is done in f32 (doing it in f64 splits from the existing
  goldens — the values are `test_it_reads_every_weight_at_the_fake_quant_value`, and the rejection
  of a source that returns f64 values is `test_a_source_returning_f64_is_rejected`)
- only one block is materialised at a time (the premise of the RAM estimate for the 5B in f64)
- the f64 path stops if it creates a floating-point value other than f64 (weight loading is
  outside the watch)
- the table of upstream mmap offsets matches the actual addresses of `from_pretrained`
- the comparison (`compare_tensors`) catches a one-bit difference
- the graph of the meta trace of the 5B probe has the same structure as the real-weight export,
  and the per-row-block i8 gives the same values as fake-quant

Runs only on a synthetic model (a small randomly initialised `WanTransformer3DModel` —
`test_dit_patch.TINY_DIT`). The comparison with the real 1.3B goldens takes long, so it is
`python -m wan.dit_reference compare` (runbook).
"""

from __future__ import annotations

import copy
import json
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

from wan import dit_patch, dit_reference, export_dit
from wan.tests.test_dit_patch import TINY_DIT, TINY_LATENT

#: The synthetic layer count (num_layers of TINY_DIT).
LAYERS = int(TINY_DIT["num_layers"])


def _tiny(seed: int = 20261004) -> nn.Module:
    transformer_wan = pytest.importorskip("diffusers.models.transformers.transformer_wan")
    torch.manual_seed(seed)
    return transformer_wan.WanTransformer3DModel(**TINY_DIT).to(torch.float32).eval()


def _inputs(seed: int = 11) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    latents = torch.randn(1, TINY_DIT["in_channels"], *TINY_LATENT, generator=generator)
    timestep = torch.tensor([750], dtype=torch.int64)
    embeds = torch.randn(1, 5, TINY_DIT["text_dim"], generator=generator)
    return latents, timestep, embeds


def _rounded(dtype: str, seed: int = 20261004) -> nn.Module:
    model = _tiny(seed)
    export_dit.fake_quant(dtype, model, dit_patch.WanDitTokens(model))
    return model


def _whole_f64(model: nn.Module, inputs: tuple[torch.Tensor, ...]) -> torch.Tensor:
    """How the Wan2.1 f64 reference is made (the whole model in f64, then the upstream forward)."""
    with torch.no_grad():
        return dit_patch.reference_dit_f64(copy.deepcopy(model).double(), *inputs)


@pytest.mark.parametrize("dtype", ["f16", "i8"])
class TestLayerwiseMatchesTheWholeModel:
    """Layer by layer with the in-memory model as the source = the upstream forward run on the
    whole model (bit-identical)."""

    def test_f32_output_and_every_block_are_bit_identical(self, dtype: str) -> None:
        model = _rounded(dtype)
        inputs = _inputs()
        with torch.no_grad():
            expected, blocks = dit_patch.reference_dit_layers(model, *inputs)

        got = dit_reference.LayerwiseDit(model.config).forward(
            dit_reference.ModuleWeights(model), torch.float32, *inputs, collect_blocks=True
        )

        assert torch.equal(got.output, expected)
        assert len(got.blocks) == LAYERS
        assert all(torch.equal(a, b) for a, b in zip(got.blocks, blocks, strict=True))

    def test_f64_output_is_bit_identical_to_the_double_model(self, dtype: str) -> None:
        model = _rounded(dtype)
        inputs = _inputs()

        got = dit_reference.LayerwiseDit(model.config).forward(
            dit_reference.ModuleWeights(model), torch.float64, *inputs
        )

        assert got.output.dtype == torch.float64
        assert got.blocks == []
        assert torch.equal(got.output, _whole_f64(model, inputs))

    def test_a_different_weight_breaks_the_identity(self, dtype: str) -> None:
        """Fault injection: moving one weight by 1 ULP breaks the match (not a vacuous check)."""
        model = _rounded(dtype)
        inputs = _inputs()
        expected = _whole_f64(model, inputs)

        class Nudged:
            def tensor(self, key: str) -> torch.Tensor:
                value = dit_reference.ModuleWeights(model).tensor(key)
                if key == "blocks.1.ffn.net.2.weight":
                    return torch.nextafter(value, torch.full_like(value, float("inf")))
                return value

        got = dit_reference.LayerwiseDit(model.config).forward(Nudged(), torch.float64, *inputs)

        assert not torch.equal(got.output, expected)


class TestTheFloat64Path:
    def test_an_upstream_f32_cast_left_in_place_is_caught(self, monkeypatch) -> None:
        """Removing the `.float()` pass-through (so that upstream's f32 conversion of the norms and
        residuals runs as is) stops it."""
        model = _rounded("i8")
        monkeypatch.setattr(dit_patch, "_float_keeps_float64", nullcontext)

        with pytest.raises(dit_reference.DitReferenceError, match="f64 でない値"):
            dit_reference.LayerwiseDit(model.config).forward(
                dit_reference.ModuleWeights(model), torch.float64, *_inputs()
            )

    def test_the_tensor_float_method_is_restored(self) -> None:
        model = _rounded("f16")
        dit_reference.LayerwiseDit(model.config).forward(
            dit_reference.ModuleWeights(model), torch.float64, *_inputs()
        )

        assert torch.ones(2, dtype=torch.float64).float().dtype == torch.float32
        assert "float" not in vars(torch.Tensor)

    def test_a_two_dimensional_timestep_stops_at_the_entrance_naming_the_reference_wrapper(
        self,
    ) -> None:
        """A two-dimensional timestep (`[1,S]` — diffusers' M = S path) is for observation and is
        not made f64 — instead of failing deep inside the watch, it stops at the entrance with a
        message saying that per-token times are passed through the reference wrapper of step 1
        (`condition=`)."""
        model = _rounded("i8")
        latents, _, embeds = _inputs()
        tokens = TINY_LATENT[0] * (TINY_LATENT[1] // 2) * (TINY_LATENT[2] // 2)
        per_token = torch.full((1, tokens), 750, dtype=torch.int64)

        with pytest.raises(dit_reference.DitReferenceError, match="段 1"):
            dit_reference.LayerwiseDit(model.config).forward(
                dit_reference.ModuleWeights(model), torch.float64, latents, per_token, embeds
            )

    def test_a_source_returning_f64_is_rejected(self) -> None:
        """The source returns f32 — accepting f64 values could let "weights dequantized in f64"
        slip through."""
        model = _rounded("i8")

        class Wide:
            def tensor(self, key: str) -> torch.Tensor:
                return dit_reference.ModuleWeights(model).tensor(key).double()

        with pytest.raises(dit_reference.DitReferenceError, match="float64"):
            dit_reference.LayerwiseDit(model.config).forward(Wide(), torch.float64, *_inputs())

    def test_a_wrongly_shaped_weight_fails_loudly(self) -> None:
        """Only the patch embedding (the Linear-form shape) may have its shape restored."""
        model = _rounded("i8")

        class Flat:
            def tensor(self, key: str) -> torch.Tensor:
                value = dit_reference.ModuleWeights(model).tensor(key)
                return value.reshape(-1) if key == "proj_out.weight" else value

        with pytest.raises(dit_reference.DitReferenceError, match=r"proj_out\.weight"):
            dit_reference.LayerwiseDit(model.config).forward(Flat(), torch.float32, *_inputs())


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory) -> Path:
    """A synthetic DiT in the shape of upstream `save_pretrained` (config + a single
    safetensors)."""
    directory = tmp_path_factory.mktemp("dit-tiny")
    _tiny().save_pretrained(directory)
    return directory


def _upstream(checkpoint: Path) -> nn.Module:
    from diffusers import WanTransformer3DModel

    return WanTransformer3DModel.from_pretrained(checkpoint, torch_dtype=torch.float32).eval()


def _container(model: nn.Module, dtype: str, directory: Path) -> Path:
    """Round the synthetic model and write a container of the same form as the series (an export
    with the same arguments as `export_dit.emit`)."""
    from karume.container import Provenance
    from karume.convert import PRESERVED_OP_PREFIXES_WITH_ATTENTION
    from karume.pipeline import export_to_file

    wrapper = dit_patch.WanDitTokens(model)
    _, scales = export_dit.fake_quant(dtype, model, wrapper)
    latents, timestep, embeds = _inputs()
    inputs = (
        dit_patch.dit_patchify(latents, (1, 2, 2)),
        dit_patch.dit_timesteps_proj(model, timestep),
        embeds,
        *dit_patch.dit_rope_tables(model.rope, TINY_LATENT),
    )
    path = directory / export_dit.MODEL_FILE
    export_to_file(
        wrapper,
        inputs,
        path,
        provenance=Provenance(license="apache-2.0"),
        graph_name=export_dit.TARGET,
        dynamic_shapes=export_dit.dynamic_shapes(),
        symbol_names=("S",),
        weight_dtype=dtype,
        weight_scales=scales,
        preserved=PRESERVED_OP_PREFIXES_WITH_ATTENTION,
    )
    return path


@pytest.mark.parametrize("dtype", ["f16", "i8"])
class TestTheContainerSource:
    """Layer by layer read from the series container (the 5B path) = the whole-model upstream
    reference."""

    def test_it_reads_every_weight_at_the_fake_quant_value(
        self, dtype: str, checkpoint: Path, tmp_path: Path
    ) -> None:
        """i8 is packed × scale multiplied in f32 (the same as fake-quant); f16 is f16 → f32."""
        model = _upstream(checkpoint)
        path = _container(model, dtype, tmp_path)
        source = dit_reference.ContainerDitWeights(path)
        writer = dit_reference.LayerwiseDit(model.config)
        state = model.state_dict()

        for key in writer.weight_keys():
            assert torch.equal(source.tensor(key).reshape(state[key].shape), state[key]), key

    def test_f64_and_aligned_f32_are_bit_identical_to_the_upstream_reference(
        self, dtype: str, checkpoint: Path, tmp_path: Path
    ) -> None:
        model = _upstream(checkpoint)
        path = _container(model, dtype, tmp_path)
        inputs = _inputs()
        with torch.no_grad():
            expected32, blocks = dit_patch.reference_dit_layers(model, *inputs)
        writer = dit_reference.LayerwiseDit(
            model.config, alignment=dit_reference.upstream_alignment(checkpoint)
        )
        source = dit_reference.ContainerDitWeights(path)

        got32 = writer.forward(source, torch.float32, *inputs, collect_blocks=True)
        got64 = writer.forward(source, torch.float64, *inputs)

        assert torch.equal(got32.output, expected32)
        assert all(torch.equal(a, b) for a, b in zip(got32.blocks, blocks, strict=True))
        assert torch.equal(got64.output, _whole_f64(model, inputs))


class TestOneBlockAtATime:
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=["f32", "f64"])
    def test_the_previous_block_is_released_before_the_next_is_read(
        self, dtype: torch.dtype
    ) -> None:
        """When the weights of the next block start to load, 0 layer-by-layer blocks are
        materialised (not meta).

        The loop variable of upstream's `for block in self.blocks` still holds the previous block
        while the next one is requested, so unless the previous block's weights are released, 2
        blocks are alive at once (1.22 → 2.44 GiB for the 5B in f64).
        """
        import gc

        transformer_wan = pytest.importorskip("diffusers.models.transformers.transformer_wan")
        model = _rounded("i8")
        upstream_blocks = {id(block) for block in model.blocks}
        writer = dit_reference.LayerwiseDit(model.config)
        first = next(iter(writer.block_shapes))
        weights = dit_reference.ModuleWeights(model)
        live: list[int] = []

        class Counting:
            def tensor(self, key: str) -> torch.Tensor:
                if key.startswith("blocks.") and key.endswith(f".{first}"):
                    gc.collect()  # unreachable leftover cycles (not yet collected) are not "alive"
                    live.append(
                        sum(
                            type(item) is transformer_wan.WanTransformerBlock
                            and id(item) not in upstream_blocks
                            and any(not p.is_meta for p in item.parameters())
                            for item in gc.get_objects()
                        )
                    )
                return weights.tensor(key)

        writer.forward(Counting(), dtype, *_inputs())

        assert live == [0] * LAYERS


class TestTheUpstreamAlignment:
    def test_it_predicts_where_from_pretrained_puts_every_weight(self, checkpoint: Path) -> None:
        """The offset table = the offsets from a 64-byte boundary of the actual addresses of the
        weights loaded by `from_pretrained`."""
        model = _upstream(checkpoint)
        table = dit_reference.upstream_alignment(checkpoint)

        actual = {
            key: value.data_ptr() % dit_reference.ALIGNMENT_BYTES
            for key, value in model.state_dict().items()
        }
        assert table == actual

    @pytest.mark.parametrize("offset", [0, 8, 36, 60])
    def test_placing_keeps_the_value_and_lands_on_the_offset(self, offset: int) -> None:
        value = torch.randn(5, 7)

        placed = dit_reference._placed(value, offset)

        assert placed.data_ptr() % dit_reference.ALIGNMENT_BYTES == offset
        assert placed.is_contiguous()
        assert torch.equal(placed, value)

    def test_a_key_outside_the_table_fails_loudly(self) -> None:
        model = _rounded("i8")
        writer = dit_reference.LayerwiseDit(model.config, alignment={})

        with pytest.raises(dit_reference.DitReferenceError, match="ずれが無い"):
            writer.forward(dit_reference.ModuleWeights(model), torch.float32, *_inputs())


class TestTheCheckpointSource:
    """The source that rounds from the checkpoint one weight at a time = rounding applied to the
    whole model (bit-identical)."""

    @pytest.mark.parametrize("dtype", ["f16", "i8"])
    def test_each_weight_equals_the_whole_model_rounding(
        self, dtype: str, checkpoint: Path
    ) -> None:
        model = _upstream(checkpoint)
        export_dit.fake_quant(dtype, model, dit_patch.WanDitTokens(model))
        source = dit_reference.CheckpointDitWeights(
            checkpoint, dtype, dit_reference.quant_keys(model.config)
        )

        for key, value in model.state_dict().items():
            assert torch.equal(source.tensor(key), value), key

    def test_the_unrounded_source_returns_the_checkpoint_values(self, checkpoint: Path) -> None:
        model = _upstream(checkpoint)
        source = dit_reference.CheckpointDitWeights(checkpoint, "none")

        for key, value in model.state_dict().items():
            assert torch.equal(source.tensor(key), value), key

    def test_the_quant_keys_are_the_weights_the_export_rounds(self) -> None:
        """The quantization targets = the keys of the scale ledger of the export's
        `fake_quant_i8` (the linears and the patch embedding)."""
        model = _tiny()
        report = export_dit.fake_quant_i8(model, dit_patch.WanDitTokens(model))

        assert dit_reference.quant_keys(model.config) == frozenset(report.scales)
        assert "patch_embedding.weight" in report.scales

    def test_i8_without_targets_is_refused(self, checkpoint: Path) -> None:
        with pytest.raises(dit_reference.DitReferenceError, match="quant_keys"):
            dit_reference.CheckpointDitWeights(checkpoint, "i8")


class TestCompareTensors:
    def _file(self, tmp_path: Path, tensors: dict[str, torch.Tensor]) -> Path:
        from safetensors.torch import save_file

        path = tmp_path / "reference.case.safetensors"
        save_file(tensors, str(path))
        return path

    def test_identical_tensors_have_no_mismatch(self, tmp_path: Path) -> None:
        tensors = {"output": torch.randn(3, 4), "timestep": torch.tensor([7], dtype=torch.int32)}

        result = dit_reference.compare_tensors(tensors, self._file(tmp_path, tensors))

        assert result == {"keys": ["output", "timestep"], "mismatched": {}}

    def test_one_flipped_bit_and_a_missing_key_are_named(self, tmp_path: Path) -> None:
        tensors = {"output": torch.randn(3, 4), "block.00": torch.randn(2)}
        path = self._file(tmp_path, tensors)
        flipped = tensors["output"].clone()
        flipped.view(torch.int32)[1, 2] ^= 1

        result = dit_reference.compare_tensors({"output": flipped, "extra": torch.ones(1)}, path)

        assert set(result["mismatched"]) == {"output", "block.00", "extra"}
        assert "ビットが違う" in result["mismatched"]["output"]
        assert result["mismatched"]["block.00"] == "書いた側に無い"
        assert result["mismatched"]["extra"] == "既存に無い"

    def test_negative_zero_is_not_equal_to_zero(self, tmp_path: Path) -> None:
        path = self._file(tmp_path, {"output": torch.zeros(2)})

        result = dit_reference.compare_tensors({"output": torch.tensor([0.0, -0.0])}, path)

        assert "output" in result["mismatched"]

    def test_only_the_named_keys_are_compared(self, tmp_path: Path) -> None:
        path = self._file(tmp_path, {"output": torch.zeros(2), "output.f64": torch.ones(2)})

        result = dit_reference.compare_tensors({"output.f64": torch.ones(2)}, path, ["output.f64"])

        assert result == {"keys": ["output.f64"], "mismatched": {}}


def _row(case: str, mode: str, version: int = dit_reference.COMPARE_VERSION) -> dict[str, Any]:
    return {"case": case, "status": "equal", "mode": mode, "compare_version": version}


class TestTheResumeMarker:
    """The rows of `results.jsonl` this run counts as "done" (the resume marker)."""

    def _write(self, tmp_path: Path, text: str) -> Path:
        path = tmp_path / dit_reference.RESULTS_FILE
        path.write_text(text, encoding="utf-8")
        return path

    def _lines(self, *rows: dict[str, Any]) -> str:
        return "".join(json.dumps(row) + "\n" for row in rows)

    def test_an_f64_only_row_does_not_count_for_a_full_run(self, tmp_path: Path) -> None:
        """Skipping a full run because of an f64-only row would turn green without comparing the
        f32 output and blocks."""
        path = self._write(tmp_path, self._lines(_row("a", "f64-only"), _row("b", "full")))

        assert sorted(dit_reference._done(path, "full")) == ["b"]
        assert sorted(dit_reference._done(path, "f64-only")) == ["a", "b"]

    def test_a_row_of_another_compare_version_does_not_count(self, tmp_path: Path) -> None:
        old = {"case": "a", "status": "equal"}  # a row without the version field (version 1)
        path = self._write(
            tmp_path,
            self._lines(
                old, _row("b", "full", dit_reference.COMPARE_VERSION - 1), _row("c", "full")
            ),
        )

        assert sorted(dit_reference._done(path, "full")) == ["c"]

    def test_a_half_written_last_line_is_dropped_and_cut_off(self, tmp_path: Path) -> None:
        whole = self._lines(_row("a", "full"))
        path = self._write(tmp_path, whole + json.dumps(_row("b", "full"))[:17])

        assert sorted(dit_reference._done(path, "full")) == ["a"]
        assert path.read_text(encoding="utf-8") == whole

    def test_an_unreadable_line_before_the_last_fails_loudly(self, tmp_path: Path) -> None:
        path = self._write(tmp_path, '{"case": \n' + self._lines(_row("a", "full")))

        with pytest.raises(dit_reference.DitReferenceError, match=":1 の行が読めない"):
            dit_reference._done(path, "full")

    def test_the_numeric_environment_records_the_mkl_switches(self, monkeypatch) -> None:
        monkeypatch.setenv("MKL_CBWR", "COMPATIBLE")
        monkeypatch.setenv("OMP_NUM_THREADS", "3")

        environment = dit_reference.numeric_environment()

        assert environment["env"]["MKL_CBWR"] == "COMPATIBLE"
        assert environment["env"]["OMP_NUM_THREADS"] == "3"
        assert environment["threads"] == torch.get_num_threads()
        assert environment["torch"] == torch.__version__


class TestTheExportDitRoute:
    """`export_dit.float64_references` goes through the layer-by-layer path and does not change
    the Wan2.1 whole-model f64 values."""

    def test_it_reproduces_the_whole_model_f64_reference(self, monkeypatch) -> None:
        model = _rounded("i8")
        spec = export_dit.CaseSpec("full-band", TINY_LATENT, 999, 5, 2, blocks=False)
        monkeypatch.setattr(export_dit, "TEXT_DIM", TINY_DIT["text_dim"])
        monkeypatch.setattr(export_dit, "pad_text_embeds", lambda embeds: embeds.unsqueeze(0))
        inputs = export_dit.case_inputs(model, spec)

        references = export_dit.float64_references(model, [spec], "i8")

        (reference,) = references.values()
        assert list(references) == [spec.name((1, 2, 2))]
        assert torch.equal(reference.output, _whole_f64(model, inputs))

    @pytest.mark.parametrize("dtype", ["f16", "i8"])
    def test_a_model_before_the_fake_quant_is_refused(self, dtype: str) -> None:
        """If the call order breaks and a model before rounding is passed, it stops (no unrounded
        f64 golden is written)."""
        spec = export_dit.CaseSpec("full-band", TINY_LATENT, 999, 5, 2, blocks=False)

        with pytest.raises(AssertionError, match="丸めの前の重み"):
            export_dit.float64_references(_tiny(), [spec], dtype)

    def test_cases_without_an_f64_reference_take_nothing(self) -> None:
        spec = export_dit.CaseSpec("band", TINY_LATENT, 999, 5, 2)

        assert export_dit.float64_references(_rounded("f16"), [spec], "f16") == {}


# ---- the 5B probe (`wan.dit_probe`) ------------------------------------------------------


def _stage(name: str) -> Any:
    from wan.umt5_export import StageRecord

    return nullcontext(StageRecord(name))


class TestTheProbe:
    def test_the_meta_trace_has_the_structure_of_the_real_weight_export(self) -> None:
        """The trace of meta weights + meta example inputs = the same ops, wiring and value shapes
        as the real-weight export.

        Only the value names (FX sequence numbers) differ — compared by structure (each node output
        is replaced by the position of its producer).
        """
        from karume.convert import PRESERVED_OP_PREFIXES_WITH_ATTENTION
        from karume.pipeline import export_module
        from wan import dit_probe

        model = _tiny()
        config = dict(model.config)
        real_inputs = tuple(
            torch.randn(tensor.shape) for tensor in dit_probe.meta_inputs(config, _PROBE_SPEC)
        )
        meta_graph, tensors, targets = dit_probe.trace(config, sym_max=256)
        real_graph, _ = export_module(
            dit_patch.WanDitTokens(model),
            real_inputs,
            dynamic_shapes=dit_probe.dynamic_shapes(256),
            symbol_names=("S",),
            preserved=PRESERVED_OP_PREFIXES_WITH_ATTENTION,
        )

        assert _structure(meta_graph) == _structure(real_graph)
        assert len(targets) == len(
            export_dit.fake_quant_i8(model, dit_patch.WanDitTokens(model)).scales
        )
        assert sorted(key for key, value in tensors.items() if value.is_meta) == sorted(
            dict(model.named_parameters())
        )

    def test_prepare_rounds_like_the_fake_quant_and_stores_i8_linears(self, checkpoint: Path):
        """Per-row-block i8 (the patch embedding with its Conv3d seen as two-dimensional) = the
        whole-model fake-quant."""
        from wan import dit_probe

        export, summary = dit_probe.prepare(checkpoint, stage=_stage, sym_max=256, chunk_rows=5)
        model = _upstream(checkpoint)
        report = export_dit.fake_quant_i8(model, dit_patch.WanDitTokens(model))
        state = model.state_dict()

        assert sorted(export.fixed) == sorted(report.scales)
        for key, weight in export.fixed.items():
            restored = (weight.packed.to(torch.float32) * weight.scale).reshape(state[key].shape)
            assert torch.equal(restored, state[key]), key
        assert all(torch.equal(export.tensors[key], state[key]) for key in export.plain)
        stored = dit_probe.store(export)
        assert stored["storage"] == {
            "f32": len(export.graph.initializers) - len(export.fixed),
            "i8": len(export.fixed),
        }
        assert summary["packed_bytes"] == sum(state[key].numel() for key in export.fixed)


class TestTheProbeMemory:
    def test_the_estimate_counts_the_real_parameters(self) -> None:
        """f32 = all upstream parameters; layer-by-layer f64 = outside the blocks + 1 block (the
        count from the meta skeleton equals the count of the materialised model)."""
        from wan import dit_probe

        model = _tiny()
        parameters = dict(model.named_parameters())
        outer = sum(p.numel() for name, p in parameters.items() if not name.startswith("blocks."))
        block = sum(p.numel() for name, p in parameters.items() if name.startswith("blocks.0."))

        estimate = dit_probe.eager_memory(dict(model.config))

        assert estimate["f32"] == sum(p.numel() for p in parameters.values()) * 4
        assert estimate["layerwise_f64"] == (outer + block) * 8
        assert estimate["peak"] == (
            estimate["f32"] + estimate["layerwise_f64"] + dit_probe.EAGER_BASE_BYTES
        )

    def test_eager_stops_before_loading_when_memory_is_short(
        self, checkpoint: Path, tmp_path: Path, monkeypatch
    ) -> None:
        from wan import dit_probe

        def never(*_args: Any) -> None:
            raise AssertionError("read the upstream although the RAM is insufficient")

        monkeypatch.setattr(dit_probe, "transformer_dir", lambda _model: checkpoint)
        monkeypatch.setattr(dit_probe, "mem_available", lambda: 2**30)
        monkeypatch.setattr(export_dit, "load_transformer", never)

        with pytest.raises(dit_probe.DitProbeError, match=r"MemAvailable before load is 1\.00 GiB"):
            dit_probe.run_eager("t2v-1.3b", tmp_path)

    def test_the_default_outputs_live_under_the_bench_root(self) -> None:
        from _shared.paths import BENCH_ROOT
        from wan import dit_probe
        from wan.sources import SOURCES

        assert set(dit_probe.BENCH_NAMES) == set(SOURCES)
        for model in SOURCES:
            assert dit_probe.default_out(model).parent.parent == BENCH_ROOT
        assert dit_reference.DEFAULT_OUT.parent.parent == BENCH_ROOT


_PROBE_SPEC = export_dit.CaseSpec("band", TINY_LATENT, 999, 5, 1)


def _structure(graph: Any) -> list[tuple[str, list[str], str, list[str]]]:
    """The structure of a graph (op, input producers, attrs, output shapes) — independent of value
    names."""
    document = json.loads(graph.to_json())
    produced: dict[str, str] = {}
    rows = []
    for position, node in enumerate(document["nodes"]):
        sources = [produced.get(name, name) for name in node["ins"]]
        shapes = [json.dumps(document["values"][name], sort_keys=True) for name in node["outs"]]
        rows.append(
            (node["op"], sources, json.dumps(node.get("attrs", {}), sort_keys=True), shapes)
        )
        for index, name in enumerate(node["outs"]):
            produced[name] = f"#{position}.{index}"
    return rows


# ---- layer-by-layer per-block entry points, isolating the f32 / f64 gap (ADR 0121 step 1) ----


class TestTheBlockHooks:
    def test_on_output_sees_every_block_output_without_keeping_it(self) -> None:
        """The block outputs `on_output` receives = the outputs `collect_blocks` collects (same
        order, same values)."""
        model = _rounded("i8")
        writer = dit_reference.LayerwiseDit(model.config)
        seen: list[tuple[int, torch.Tensor]] = []

        got = writer.forward(
            dit_reference.ModuleWeights(model),
            torch.float32,
            *_inputs(),
            collect_blocks=True,
            on_output=lambda index, output: seen.append((index, output.clone())),
        )

        assert [index for index, _ in seen] == list(range(LAYERS))
        assert all(torch.equal(a, b) for (_, a), b in zip(seen, got.blocks, strict=True))

    def test_on_output_runs_outside_the_float64_watch(self) -> None:
        """Even if the receiver of the f64 run creates f32 values (the comparison with the record),
        the f64 watch does not stop it."""
        model = _rounded("i8")
        writer = dit_reference.LayerwiseDit(model.config)

        got = writer.forward(
            dit_reference.ModuleWeights(model),
            torch.float64,
            *_inputs(),
            on_output=lambda _index, output: output.float().sum(),
        )

        assert got.output.dtype == torch.float64

    def test_blocks_runs_only_the_leading_blocks(self) -> None:
        model = _rounded("i8")
        writer = dit_reference.LayerwiseDit(model.config)
        source = dit_reference.ModuleWeights(model)
        full = writer.forward(source, torch.float32, *_inputs(), collect_blocks=True)

        cut = writer.forward(source, torch.float32, *_inputs(), collect_blocks=True, blocks=1)

        assert len(cut.blocks) == 1
        assert torch.equal(cut.blocks[0], full.blocks[0])
        assert not torch.equal(cut.output, full.output)

    @pytest.mark.parametrize("blocks", [0, LAYERS + 1])
    def test_blocks_outside_the_model_fail_loudly(self, blocks: int) -> None:
        model = _rounded("i8")

        with pytest.raises(dit_reference.DitReferenceError, match="回すブロックの数"):
            dit_reference.LayerwiseDit(model.config).forward(
                dit_reference.ModuleWeights(model), torch.float32, *_inputs(), blocks=blocks
            )


def _tiny_text(monkeypatch) -> None:
    from wan import dit_probe

    monkeypatch.setattr(export_dit, "TEXT_DIM", TINY_DIT["text_dim"])
    monkeypatch.setattr(export_dit, "pad_text_embeds", lambda embeds: embeds.unsqueeze(0))
    monkeypatch.setattr(dit_probe, "pad_text_embeds", lambda embeds: embeds.unsqueeze(0))


class TestTheProbeDiagnosis:
    """The isolation entry points of `wan.dit_probe reference` (text replacement, per-block
    differences, truncation)."""

    def test_a_named_text_keeps_the_latents_and_swaps_only_the_context(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        from safetensors.torch import save_file

        from wan import dit_probe
        from wan.text_embeds import METADATA_KEY

        _tiny_text(monkeypatch)
        asset = tmp_path / "text_embeds.safetensors"
        embeds = torch.randn(3, TINY_DIT["text_dim"])
        save_file({"boxing-cats": embeds}, str(asset), metadata={METADATA_KEY: "{}"})
        monkeypatch.setattr(dit_probe, "TEXT_EMBEDS_ASSET", asset)
        writer = dit_reference.LayerwiseDit(_tiny().config)
        spec = export_dit.CaseSpec("probe", TINY_LATENT, 999, 5, 1)

        synthetic = dit_probe.probe_inputs(writer, spec)
        named = dit_probe.probe_inputs(writer, spec, "boxing-cats")

        assert all(
            torch.equal(a, b)
            for a, b in zip(synthetic, export_dit.case_inputs(writer, spec), strict=True)
        )
        assert torch.equal(named[0], synthetic[0])
        assert torch.equal(named[1], synthetic[1])
        assert torch.equal(named[2], embeds.unsqueeze(0))
        with pytest.raises(dit_probe.DitProbeError, match="has no 'ferret'"):
            dit_probe.probe_inputs(writer, spec, "ferret")

    def test_growth_records_one_row_per_block_and_cleans_its_spill(
        self, checkpoint: Path, tmp_path: Path, monkeypatch
    ) -> None:
        from wan import dit_probe

        _tiny_text(monkeypatch)
        monkeypatch.setattr(dit_probe, "transformer_dir", lambda _model: checkpoint)

        summary = dit_probe.run_reference(
            "ti2v-5b", tmp_path, TINY_LATENT, timestep=500, growth=True
        )

        rows = summary["growth"]
        assert [row["block"] for row in rows] == list(range(LAYERS))
        assert all(0 < row["ratio"] < 1e-3 and 0 < row["rel_rms"] < 1e-3 for row in rows)
        assert summary["timestep"] == 500
        assert summary["output"]["ratio"] == summary["f32_vs_f64"]
        lines = (tmp_path / f"growth.{summary['case']}.jsonl").read_text().splitlines()
        assert [json.loads(line)["block"] for line in lines] == list(range(LAYERS))
        assert not (tmp_path / dit_probe.GROWTH_SPILL).exists()

    def test_blocks_names_the_case_and_stops_early(
        self, checkpoint: Path, tmp_path: Path, monkeypatch
    ) -> None:
        from wan import dit_probe

        _tiny_text(monkeypatch)
        monkeypatch.setattr(dit_probe, "transformer_dir", lambda _model: checkpoint)

        summary = dit_probe.run_reference("ti2v-5b", tmp_path, TINY_LATENT, growth=True, blocks=1)

        assert summary["case"].endswith("-b01")
        assert summary["blocks"] == 1
        assert [row["block"] for row in summary["growth"]] == [0]


class TestTheProbeCaseFlag:
    def test_a_case_spells_the_grid_time_text_and_optional_blocks(self) -> None:
        from wan import dit_probe

        assert dit_probe._probe_run("4,30,52:999:boxing-cats") == dit_probe.ProbeRun(
            (4, 30, 52), 999, "boxing-cats", None
        )
        assert dit_probe._probe_run("9,44,80:500:synthetic:6").blocks == 6

    @pytest.mark.parametrize("text", ["4,30,52:999", "4,30:999:synthetic", "a:b:c:d:e"])
    def test_a_malformed_case_is_rejected(self, text: str) -> None:
        import argparse

        from wan import dit_probe

        with pytest.raises((argparse.ArgumentTypeError, ValueError)):
            dit_probe._probe_run(text)

    def test_case_and_the_single_case_flags_do_not_mix(self) -> None:
        from wan import dit_probe

        with pytest.raises(SystemExit):
            dit_probe.main(["reference", "--case", "4,30,52:999:synthetic", "--timestep", "500"])

    def test_every_case_writes_its_own_summary(
        self, checkpoint: Path, tmp_path: Path, monkeypatch
    ) -> None:
        from wan import dit_probe

        _tiny_text(monkeypatch)
        monkeypatch.setattr(dit_probe, "transformer_dir", lambda _model: checkpoint)

        code = dit_probe.main(
            [
                "reference",
                "--out",
                str(tmp_path),
                "--growth",
                "--case",
                "2,6,10:999:synthetic",
                "--case",
                "2,6,10:500:synthetic:1",
            ]
        )

        assert code == 0
        assert sorted(path.name for path in tmp_path.glob("reference.*.json")) == [
            "reference.probe-s00030-t0500-b01.json",
            "reference.probe-s00030-t0999.json",
        ]
