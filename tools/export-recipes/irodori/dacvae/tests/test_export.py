"""Script-level contracts of `irodori/dacvae/export.py` (the part that needs no real weights).

Emitting the real weights is manual (the same discipline as the tests of the other export
scripts). What is pinned here is only the discipline whose breakage would produce a **false
PASS**:

- the reference (the golden's expected values) can only be taken **before** `remove_weight_norm`
  (if it could be taken afterwards, the equivalence check of the folding would become a
  tautology)
- taking the reference fails if the watermark bypass is not in form ② (the output is not a
  waveform)
- fails if the main-path extraction disagrees with the `Decoder` itself
- fails if the `in_proj` truncation takes the second half (scale)
- fails if `remove_weight_norm` changes values
- fails if the round trip (encoder → decoder) is not a plausible waveform
- fails if Snake's constant subtree is not folded and `reciprocal` remains
- `_write_io` does not write a graph that is not one input, one output
- fails if a case's length exceeds the symbolic dimension's upper bound
"""

from __future__ import annotations

import types

import pytest
import torch
from safetensors.torch import load_file, save_file
from torch import nn

from irodori.dacvae import export as ex
from karume.ir import IrGraph, IrInitializer, IrInput, IrNode, IrStorage, IrValue
from karume.pipeline import export_to_file
from karume.quantize import quantize_to_int8


class Scale(nn.Module):
    """A constant-gain layer (a stand-in where a difference in order shows in the values)."""

    def __init__(self, gain: float) -> None:
        super().__init__()
        self.gain = gain

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gain


class TinyBlock(nn.Module):
    """Stand-in for `DecoderBlock` (cuts every `_chunk_size` and runs only the multiple chunks)."""

    def __init__(self) -> None:
        super().__init__()
        self._chunk_size = 2
        self.block = nn.ModuleList([Scale(g) for g in (2.0, 3.0, 5.0, 7.0, 11.0, 13.0)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Only chunk 0 (gain 2,3) and chunk 2 (gain 11,13) are the main path.
        return x * (2.0 * 3.0 * 11.0 * 13.0)


class TinyWatermark(nn.Module):
    """Stand-in for `wm_model.encoder_block` (`pre[3]` is the position replaced by Identity in
    form ②)."""

    def __init__(self) -> None:
        super().__init__()
        self.pre = nn.Sequential(Scale(17.0), Scale(19.0), Scale(23.0), Scale(29.0))

    def forward_no_conv(self, x: torch.Tensor) -> torch.Tensor:
        return self.pre[2](self.pre[1](self.pre[0](x)))


class TinyDecoder(nn.Module):
    """Stand-in for `Decoder`. Passing `extra` mixes in a layer off the main path (fault
    injection)."""

    def __init__(self, extra: float = 1.0) -> None:
        super().__init__()
        self.model = nn.ModuleList([Scale(1.5), TinyBlock()])
        self.wm_model = nn.Module()
        self.wm_model.encoder_block = TinyWatermark()
        self.extra = extra

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.model:
            x = layer(x)
        return self.wm_model.encoder_block.forward_no_conv(x) * self.extra


class TinyQuantizer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.out_proj = nn.Conv1d(2, 2, 1)
        self.in_proj = nn.Conv1d(2, 4, 1)


class TinyCodec(nn.Module):
    """Stand-in with only the attributes `_pristine_outputs` / `_remove_weight_norm_evidence`
    read."""

    def __init__(self, extra: float = 1.0, weight_norm: bool = True) -> None:
        super().__init__()
        self.decoder = TinyDecoder(extra)
        self.quantizer = TinyQuantizer()
        self.encoder = nn.Conv1d(1, 2, 1)
        if weight_norm:
            nn.utils.weight_norm(self.quantizer.in_proj)
        self.eval()


DECODER_CASES = (("a", torch.ones(1, 3, 2)),)
ENCODER_CASES = (("b", torch.ones(1, 3, 4)),)


class SingleChannelDecoder(TinyDecoder):
    """Stand-in whose output has 1 channel (a waveform) — the same property as the real form-②
    bypass."""


class MultiChannelDecoder(nn.Module):
    """Stand-in where the form-② bypass did not take effect (fault injection that returns the
    96ch equivalent as is)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


def _codec_with_decoder(decoder: nn.Module) -> nn.Module:
    codec = TinyCodec()
    codec.decoder = decoder
    return codec


class TestPristineOrdering:
    """MUST: the reference can only be taken before `remove_weight_norm` (guarding against
    tautology)."""

    def test_taking_a_reference_after_folding_fails_loudly(self):
        codec = TinyCodec(weight_norm=False)
        codec.decoder = _one_channel_decoder()

        with pytest.raises(
            AssertionError, match="tried to take the reference after folding weight_norm"
        ):
            ex._pristine_outputs(codec, DECODER_CASES, ENCODER_CASES)

    def test_taking_a_reference_before_folding_is_allowed(self):
        codec = TinyCodec()
        codec.decoder = _one_channel_decoder()

        pristine = ex._pristine_outputs(codec, DECODER_CASES, ENCODER_CASES)

        assert set(pristine) == {ex.TARGET_DECODER, ex.TARGET_ENCODER}
        assert tuple(pristine[ex.TARGET_DECODER]["a"].shape) == (1, 1, 3)
        # in_proj has 4 outputs, so the mean side has 2 channels → transposed to [1,T,2].
        assert tuple(pristine[ex.TARGET_ENCODER]["b"].shape) == (1, 12, 2)

    def test_a_multi_channel_decode_fails_loudly(self):
        """MUST: a half-done bypass with only `alpha = 0.0` can only be noticed from the shape."""
        codec = _codec_with_decoder(MultiChannelDecoder())

        with pytest.raises(AssertionError, match="watermark bypass"):
            ex._pristine_outputs(codec, DECODER_CASES, ENCODER_CASES)


def _one_channel_decoder() -> nn.Module:
    """Stand-in that takes `[1,2,T]` and returns `[1,1,T]` (the same shape as the waveform
    head)."""

    class Head(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x[:, :1] * 2.0

    return Head()


class TestMainPathEvidence:
    """MUST: do not leave "the watermarking branch contributes nothing to decode" as a mere
    claim."""

    def test_a_faithful_decoder_measures_zero(self):
        decoder = TinyDecoder()
        out_proj = nn.Identity()

        evidence = ex._main_path_evidence(decoder, DECODER_CASES, out_proj)

        assert evidence == {"a": 0.0}

    def test_a_decoder_that_takes_another_route_fails_loudly(self):
        decoder = TinyDecoder(extra=1.0001)

        with pytest.raises(AssertionError, match="main-path extraction"):
            ex._main_path_evidence(decoder, DECODER_CASES, nn.Identity())

    def test_the_extraction_keeps_only_the_chunk_multiples(self):
        extracted = ex.main_path(TinyDecoder())

        # head(1.5) + chunk 0 (2,3) + chunk 2 (11,13) + pre[0..2] (17,19,23).
        assert [float(layer.gain) for layer in extracted] == [
            1.5,
            2.0,
            3.0,
            11.0,
            13.0,
            17.0,
            19.0,
            23.0,
        ]


class TestInProjTruncation:
    def test_the_first_half_measures_within_the_gate_tolerance(self):
        """NOTE: the expectation is not exactly 0.0 — the two sides compared may get different conv
        kernel selections, and bit identity is not a torch spec guarantee (the derivation table of
        `IN_PROJ_TRUNCATION_ATOL`). A 1 ulp difference was also measured when an added test
        shifted the global RNG sequence (2026-08-13)."""
        codec = TinyCodec(weight_norm=False)
        trimmed = ex.truncated_in_proj(codec.quantizer.in_proj)

        evidence = ex._in_proj_truncation_evidence(codec, trimmed, ENCODER_CASES)

        assert set(evidence) == {"b"}
        assert 0.0 <= evidence["b"] <= ex.IN_PROJ_TRUNCATION_ATOL

    def test_taking_the_second_half_fails_loudly(self):
        """MUST: a mean / scale mix-up matches in shape and dtype, so only this reveals it."""
        codec = TinyCodec(weight_norm=False)
        in_proj = codec.quantizer.in_proj
        half = in_proj.out_channels // 2
        wrong = nn.Conv1d(in_proj.in_channels, half, 1)
        with torch.no_grad():
            wrong.weight.copy_(in_proj.weight[half:])
            wrong.bias.copy_(in_proj.bias[half:])

        with pytest.raises(AssertionError, match="chunk\\(2\\)\\[0\\]"):
            ex._in_proj_truncation_evidence(codec, wrong.eval(), ENCODER_CASES)

    def test_truncating_before_folding_fails_loudly(self):
        codec = TinyCodec()

        with pytest.raises(SystemExit, match="weight_norm of in_proj is not folded"):
            ex.truncated_in_proj(codec.quantizer.in_proj)

    def test_an_odd_output_count_fails_loudly(self):
        with pytest.raises(SystemExit, match="is not even"):
            ex.truncated_in_proj(nn.Conv1d(2, 3, 1))


class TestFoldEvidence:
    def test_an_unchanged_output_measures_zero(self):
        codec = TinyCodec()
        codec.decoder = _one_channel_decoder()
        pristine = ex._pristine_outputs(codec, DECODER_CASES, ENCODER_CASES)
        ex.fold_weight_norm(codec)

        evidence = ex._remove_weight_norm_evidence(codec, DECODER_CASES, ENCODER_CASES, pristine)

        assert evidence[ex.TARGET_DECODER] == {"a": 0.0}
        assert evidence[ex.TARGET_ENCODER] == {"b": 0.0}

    def test_a_changed_output_fails_loudly(self):
        codec = TinyCodec()
        codec.decoder = _one_channel_decoder()
        pristine = ex._pristine_outputs(codec, DECODER_CASES, ENCODER_CASES)
        ex.fold_weight_norm(codec)
        with torch.no_grad():
            codec.quantizer.in_proj.bias.add_(1e-3)

        with pytest.raises(AssertionError, match="the output changed across remove_weight_norm"):
            ex._remove_weight_norm_evidence(codec, DECODER_CASES, ENCODER_CASES, pristine)

    def test_measuring_before_folding_fails_loudly(self):
        codec = TinyCodec()
        codec.decoder = _one_channel_decoder()
        pristine = ex._pristine_outputs(codec, DECODER_CASES, ENCODER_CASES)

        with pytest.raises(AssertionError, match="weight_norm is still present"):
            ex._remove_weight_norm_evidence(codec, DECODER_CASES, ENCODER_CASES, pristine)

    def test_a_model_without_weight_norm_fails_loudly(self):
        with pytest.raises(SystemExit, match="no module has weight_norm"):
            ex.fold_weight_norm(TinyCodec(weight_norm=False))


class FakeSnake(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, 2, 1))


class TestLiftSnakeAlphas:
    def test_alphas_become_plain_attributes(self):
        model = nn.Sequential(FakeSnake(), FakeSnake())
        source = types.SimpleNamespace(snake_cls=FakeSnake)

        assert ex.lift_snake_alphas(source, model) == 2
        assert [name for name, _ in model.named_parameters()] == []
        assert all(isinstance(module.alpha, torch.Tensor) for module in model)

    def test_an_empty_scan_fails_loudly(self):
        source = types.SimpleNamespace(snake_cls=FakeSnake)

        with pytest.raises(SystemExit, match="no Snake1d found"):
            ex.lift_snake_alphas(source, nn.Sequential(nn.Identity()))


class Passthrough(nn.Module):
    """Stand-in for the round trip (`gain` injects a correlation fault, `shift` a length
    mix-up)."""

    def __init__(self, gain: float = 1.0, noise: float = 0.0, trim: int = 0) -> None:
        super().__init__()
        self.gain, self.noise, self.trim = gain, noise, trim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flat = x.reshape(1, 1, -1) * self.gain
        if self.noise:
            generator = torch.Generator().manual_seed(0)
            flat = flat + torch.randn(flat.shape, generator=generator) * self.noise
        return flat[..., : flat.shape[-1] - self.trim]


class TestRoundtripEvidence:
    @pytest.fixture
    def wave(self):
        return torch.sin(torch.linspace(0, 40, 800)).reshape(1, 10, 80)

    def test_a_faithful_roundtrip_reports_the_correlation(self, wave):
        evidence = ex._roundtrip_evidence(nn.Identity(), Passthrough(), wave)

        assert evidence["correlation"] == 1.0

    def test_an_uncorrelated_roundtrip_fails_loudly(self, wave):
        with pytest.raises(AssertionError, match="round-trip correlation"):
            ex._roundtrip_evidence(nn.Identity(), Passthrough(gain=0.0, noise=1.0), wave)

    def test_a_length_mismatch_fails_loudly(self, wave):
        with pytest.raises(AssertionError, match="round-trip length"):
            ex._roundtrip_evidence(nn.Identity(), Passthrough(trim=8), wave)

    def test_a_non_finite_output_fails_loudly(self, wave):
        with pytest.raises(AssertionError, match="non-finite"):
            ex._roundtrip_evidence(nn.Identity(), Passthrough(gain=float("inf")), wave)


class TestWrapperEquivalence:
    def test_an_identical_output_measures_zero(self):
        value = torch.ones(1, 2, 3)

        assert ex._check_wrapper_equivalence(nn.Identity(), value, value, "test", 0.0) == 0.0

    def test_a_shape_mismatch_fails_loudly(self):
        with pytest.raises(AssertionError, match="wrapper output shape"):
            ex._check_wrapper_equivalence(
                nn.Flatten(), torch.ones(1, 2, 3), torch.ones(1, 2, 3), "test", 0.0
            )

    def test_a_difference_over_atol_fails_loudly(self):
        with pytest.raises(AssertionError, match="eager equivalence broke"):
            ex._check_wrapper_equivalence(
                Scale(1.5), torch.ones(1, 2, 3), torch.ones(1, 2, 3), "test", 0.0
            )


def _graph_with_ops(*ops: str) -> IrGraph:
    return IrGraph(
        symbols=["S"],
        inputs=[IrInput(name="latent", dtype="f32", shape=[1, "S", 2])],
        outputs=["out"],
        initializers={"w": IrInitializer(tensor="w", storage=IrStorage(dtype="f32"))},
        values={"out": IrValue(dtype="f32", shape=[1, "S", 2])},
        nodes=[IrNode(op=op, ins=["latent"], outs=["out"], attrs={}) for op in ops],
    )


class TestSnakeFolded:
    def test_a_folded_graph_passes(self):
        ex.assert_snake_folded(_graph_with_ops("sin", "mul", "add"), "test")

    def test_a_leaked_reciprocal_fails_loudly(self):
        with pytest.raises(AssertionError, match="constant subtree was not folded"):
            ex.assert_snake_folded(_graph_with_ops("sin", "reciprocal"), "test")

    def test_a_graph_without_sin_fails_loudly(self):
        with pytest.raises(AssertionError, match="no sin in the graph"):
            ex.assert_snake_folded(_graph_with_ops("mul", "add"), "test")


class TinyGraph(nn.Module):
    """Minimal graph to run `_write_io` (one input named `latent`, one output)."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 2, bias=False)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.fc(latent)


class TestWriteIo:
    @pytest.fixture
    def exported(self, tmp_path):
        torch.manual_seed(0)
        module = TinyGraph()
        graph = export_to_file(
            module,
            (torch.randn(1, 3, 4),),
            tmp_path / ex.MODEL_FILE,
            provenance=ex.PROVENANCE,
            graph_name="tiny",
        )
        return module, graph, tmp_path

    def test_writes_one_file_per_case(self, exported):
        module, graph, out_dir = exported
        latent = torch.randn(1, 3, 4)
        with torch.no_grad():
            expected = {"a": module(latent)}

        written = ex._write_io(graph, {"a": latent}, expected, out_dir)

        assert written == [f"{ex.IO_PREFIX}a{ex.IO_SUFFIX}"]
        tensors = load_file(str(out_dir / written[0]))
        assert sorted(tensors) == ["input.latent", "output.0"]
        assert torch.equal(tensors["output.0"], expected["a"])

    def test_a_multi_output_graph_fails_loudly(self, exported):
        """MUST: do not silently write io that breaks the one-input, one-output convention."""
        _module, graph, out_dir = exported
        two = IrGraph(
            inputs=graph.inputs,
            outputs=[*graph.outputs, "second"],
            nodes=graph.nodes,
            values={**graph.values, "second": IrValue(dtype="f32", shape=[1, 3, 2])},
            initializers=graph.initializers,
            symbols=graph.symbols,
        )

        with pytest.raises(AssertionError, match="are not exactly one each"):
            ex._write_io(two, {"a": torch.randn(1, 3, 4)}, {"a": torch.zeros(1, 3, 2)}, out_dir)


class TestDecoderCases:
    @pytest.fixture
    def latent_dir(self, tmp_path):
        for name, frames in (("full", 5), ("no-ref", 3)):
            latent = torch.arange(frames * 2, dtype=torch.float32).reshape(1, frames, 2)
            save_file(
                {ex.LATENT_KEY: latent},
                str(tmp_path / f"{ex.LATENT_CASE_PREFIX}{name}{ex.IO_SUFFIX}"),
            )
        return tmp_path

    def test_lengths_follow_the_table_and_are_deterministic(self, latent_dir):
        first = ex.build_decoder_cases(latent_dir, 750)
        second = ex.build_decoder_cases(latent_dir, 750)

        assert [name for name, _ in first] == [name for name, _s, _l in ex.DECODER_CASES]
        assert {name: int(z.shape[1]) for name, z in first} == {
            "z-min": ex.MIN_SYM_LENGTH,
            "z-no-ref": 3,
            "z-full": 5,
            "z-max": 750,
        }
        for (_name, lhs), (_same, rhs) in zip(first, second, strict=True):
            assert torch.equal(lhs, rhs)

    def test_the_long_case_repeats_the_real_latent(self, latent_dir):
        cases = dict(ex.build_decoder_cases(latent_dir, 750))

        source = cases["z-full"]
        assert torch.equal(cases["z-max"][:, : source.shape[1]], source)
        assert torch.equal(cases["z-max"][:, source.shape[1] : 2 * source.shape[1]], source)

    def test_a_case_over_the_symbolic_cap_fails_loudly(self, latent_dir):
        with pytest.raises(SystemExit, match="symbolic dimension range"):
            ex.build_decoder_cases(latent_dir, 8)

    def test_a_missing_latent_fails_loudly(self, tmp_path):
        with pytest.raises(SystemExit, match="real latent missing"):
            ex.build_decoder_cases(tmp_path, 750)


class TestBypassWatermark:
    def test_both_halves_of_the_bypass_are_applied(self):
        decoder = TinyDecoder()

        ex.bypass_watermark(decoder)

        assert decoder.alpha == 0.0
        assert torch.equal(
            decoder.watermark(torch.ones(1, 1, 4)), torch.full((1, 1, 4), 17.0 * 19.0 * 23.0)
        )


class TestTargetCli:
    def test_all_targets_by_default(self, monkeypatch):
        seen: dict[str, object] = {}
        monkeypatch.setattr(ex, "export_series", lambda *_a, **kw: seen.update(kw) or {"dir": "x"})
        ex.main([])

        assert seen["targets"] == ex.TARGETS

    def test_a_single_target_is_forwarded(self, monkeypatch):
        seen: dict[str, object] = {}
        monkeypatch.setattr(ex, "export_series", lambda *_a, **kw: seen.update(kw) or {"dir": "x"})
        ex.main(["--target", ex.TARGET_ENCODER])

        assert seen["targets"] == (ex.TARGET_ENCODER,)

    def test_the_default_out_root_is_the_weight_directory_name(self, tmp_path):
        assert ex.default_out_root(tmp_path / "dacvae-32dim").name == "dacvae-32dim"

    def test_the_dtype_is_forwarded(self, monkeypatch):
        seen: dict[str, object] = {}
        monkeypatch.setattr(ex, "export_series", lambda *_a, **kw: seen.update(kw) or {"dir": "x"})
        ex.main(["--dtype", "f16"])

        assert seen["dtype"] == "f16"


def _is_f16_exact(tensor: torch.Tensor) -> bool:
    """Whether it sits on the f16 grid (the same predicate as emit's eligibility test)."""
    return bool(torch.equal(tensor, tensor.to(torch.float16).to(torch.float32)))


class TestWeightDtypeSeries:
    """Storage dtype series (ADR 0018 / 0019 / 0027 / 0050) — only what would produce a **false
    PASS** if broken.

    No real weights are used (the wiring discipline does not depend on the model). Verifying the
    numbers themselves is owned by the Deno-side E2E and emit's eligibility test.
    """

    def test_each_dtype_gets_its_own_series_root(self):
        """MUST: compressed series get separate directories (sharing one would apply the f32
        checks to compressed assets)."""
        roots = {
            dtype: ex.default_out_root(ex.DEFAULT_MODEL_DIR, dtype) for dtype in ex.WEIGHT_DTYPES
        }

        assert set(roots) == {"f32", "f16", "i8"}
        assert len(set(roots.values())) == len(roots)
        assert roots["f32"].name == "dacvae-32dim"
        assert roots["f16"].name == "dacvae-32dim-f16"
        assert roots["i8"].name == "dacvae-32dim-i8"

    def test_f32_leaves_the_weights_untouched(self):
        # MUST: restore the RNG after seeding (`fork_rng`) — leaving the global RNG behind would
        # make later tests build their modules with **different random numbers**, and this addition
        # would move a measurement check somewhere far away.
        with torch.random.fork_rng():
            torch.manual_seed(0)
            model = TinyCodec(weight_norm=False)
            before = model.encoder.weight.clone()

            quantized = ex._fake_quant("f32", model)

            assert quantized.report is None
            assert quantized.scales == {}
            assert torch.equal(model.encoder.weight, before)

    def test_f16_rounds_every_parameter_of_the_exported_model(self):
        with torch.random.fork_rng():
            torch.manual_seed(0)
            model = TinyCodec(weight_norm=False)
            assert not _is_f16_exact(model.encoder.weight), (
                "already on the grid before rounding, so there is no detection power"
            )

            quantized = ex._fake_quant("f16", model)

            assert quantized.report is not None
            assert quantized.scales == {}
            assert all(_is_f16_exact(tensor) for tensor in model.parameters())

    def test_i8_quantizes_the_convolutions_and_hands_out_scales(self):
        with torch.random.fork_rng():
            torch.manual_seed(0)
            model = TinyCodec(weight_norm=False)

            quantized = ex._fake_quant("i8", model)

            assert sorted(quantized.scales) == [
                "encoder.weight",
                "quantizer.in_proj.weight",
                "quantizer.out_proj.weight",
            ]
            for key, scale in quantized.scales.items():
                weight = model.get_parameter(key)
                restored = quantize_to_int8(weight, scale).to(torch.float32) * scale
                assert torch.equal(restored, weight), "not on the i8 grid"

    def test_a_lifted_snake_alpha_stays_f32(self):
        """The demoted `alpha` is not a weight slot (a lifted constant), so it is **not rounded**.

        The golden is computed with the f32 alpha too, so if only the rounded side moved, the two
        would disagree.
        """
        with torch.random.fork_rng():
            torch.manual_seed(0)
            model = nn.Sequential(FakeSnake(), nn.Conv1d(2, 2, 1))
            with torch.no_grad():
                model[0].alpha.copy_(torch.full((1, 2, 1), 1.0 + 2.0**-13))
            source = types.SimpleNamespace(snake_cls=FakeSnake)
            ex.lift_snake_alphas(source, model)
            alpha = model[0].alpha.clone()
            assert not _is_f16_exact(alpha), (
                "already on the grid before rounding, so there is no detection power"
            )

            ex._fake_quant("f16", model)

            assert torch.equal(model[0].alpha, alpha)
            assert _is_f16_exact(model[1].weight)


class TestTargetScales:
    """The table that remaps the i8 scale ledger to **in-wrapper FQNs** (`TARGET_SCALE_SOURCES`).

    MUST: if this breaks, emit fails with "eligible but no scale" (it does not pass with broken
    values). The failure happens far away, so the correspondence between the table and the
    wrappers is pinned here.
    """

    def _quantized(self) -> tuple[nn.Module, dict[str, torch.Tensor]]:
        with torch.random.fork_rng():
            torch.manual_seed(0)
            model = TinyCodec(weight_norm=False)
            return model, dict(ex._fake_quant("i8", model).scales)

    def test_the_table_covers_every_target(self):
        assert sorted(ex.TARGET_SCALE_SOURCES) == sorted(ex.TARGETS)

    def test_the_decoder_takes_only_its_own_branch(self):
        """`quantizer.out_proj` is remapped to `out_proj`; encoder / in_proj are dropped."""
        model, scales = self._quantized()
        wrapper = ex.DecoderGraph(model.quantizer.out_proj, model.decoder)

        picked = ex._target_scales(ex.TARGET_DECODER, wrapper, scales)

        assert sorted(picked) == ["out_proj.weight"]
        assert torch.equal(picked["out_proj.weight"], scales["quantizer.out_proj.weight"])

    def test_the_encoder_gets_the_front_rows_of_the_in_proj_scale(self):
        """MUST: the scale of the truncated `in_proj` is exactly **the codec's first rows** (not
        recomputed).

        Recomputing could move it by 1 ulp through the f32 division, and emit's inverse-transform
        bit-identity check would fail. Here the check emit actually performs is run with the same
        formula, to confirm that it passes.
        """
        model, scales = self._quantized()
        trimmed = ex.truncated_in_proj(model.quantizer.in_proj)
        wrapper = ex.EncoderGraph(model.encoder, trimmed)

        picked = ex._target_scales(ex.TARGET_ENCODER, wrapper, scales)

        assert sorted(picked) == ["encoder.weight", "in_proj.weight"]
        scale = picked["in_proj.weight"]
        assert torch.equal(scale, scales["quantizer.in_proj.weight"][: trimmed.out_channels])
        restored = quantize_to_int8(trimmed.weight, scale).to(torch.float32) * scale
        assert torch.equal(restored, trimmed.weight), (
            "a shape that fails emit's inverse-transform bit-identity check"
        )

    def test_a_missing_in_proj_scale_fails_loudly(self):
        model, scales = self._quantized()
        wrapper = ex.EncoderGraph(model.encoder, ex.truncated_in_proj(model.quantizer.in_proj))
        del scales[ex.FULL_IN_PROJ_KEY]

        with pytest.raises(SystemExit, match="cannot map the truncation"):
            ex._target_scales(ex.TARGET_ENCODER, wrapper, scales)

    def test_f16_hands_no_scales_to_emit(self):
        model = TinyCodec(weight_norm=False)
        wrapper = ex.DecoderGraph(model.quantizer.out_proj, model.decoder)

        assert ex._target_scales(ex.TARGET_DECODER, wrapper, {}) == {}


class TestFakeQuant:
    """MUST: do not silence "a storage dtype was requested but nothing was rounded"."""

    def test_f16_rounds_the_model_it_is_handed(self):
        with torch.random.fork_rng():
            torch.manual_seed(0)
            model = TinyCodec(weight_norm=False)

            result = ex._fake_quant("f16", model)

            assert result.report is not None
            assert all(
                torch.equal(tensor, tensor.to(torch.float16).to(torch.float32))
                for tensor in model.parameters()
            )

    def test_f16_refuses_a_model_with_nothing_to_round(self):
        """There is no "zero modules" check at the entrance on this side, so the rounded count
        itself is checked."""
        with pytest.raises(SystemExit, match="no weights were rounded"):
            ex._fake_quant("f16", nn.Identity())


class TestProvenance:
    """The codec containers' provenance shows the Apache 2.0 chain and where the NOTICE is, even
    when taken out on their own."""

    def test_the_codec_names_both_licenses_of_its_chain(self) -> None:
        """MUST: do not claim `mit` alone — the original weights (facebook/dacvae-watermarked) are
        Apache 2.0."""
        from irodori.card import IRODORI_CODEC_LICENSE

        assert ex.PROVENANCE.license == IRODORI_CODEC_LICENSE == "mit AND apache-2.0"

    def test_the_codec_points_at_the_notice(self) -> None:
        from karume.dist import NOTICE_FILENAME

        assert ex.PROVENANCE.notice == NOTICE_FILENAME
