"""Script-level contract of `vowel_detector/export.py` (the part that needs no real weights).

Emitting with the real weights is manual (the same discipline as the existing tests of
`siglip2/export.py` / `birefnet/export.py`). Here **tiny synthetic weights** ({@link tiny_module}
— the same `Crnn` with only the dimensions shrunk) run the same path once, pinning the rules whose
breakage would produce a **false PASS**:

- the copied model definition has the same keys and shapes as the upstream state_dict (the only
  check that can catch "a drifted copy" in an environment without the real weights —
  {@link TestTranscription})
- the emitted initializers are **byte-identical** to the state_dict, and no initializer that does
  not come from the weights slips in ({@link TestCheckpointBytes})
- the length is on the 20 ms grid (an odd length leaves a partial frame at the end)
- **the time axis is baked as a symbol** (`2T` — the reason this script exists; ADR 0056 / 0057)
- the golden expected values are taken from the **reference path** (`nn.GRU`), and bit identity
  with the replacement path is checked on every emit ({@link TestWriteIo})
- the golden synthetic cases **differ from each other** (using the same input twice makes the
  sanity check vacuous)
- `_sanity` checks the **order** of probability mass (a range check or self-agreement is vacuous)
- `--verify` does not emit

The replacement layer itself (bit identity with `nn.GRU`, unsupported shapes failing loudly) is
covered by `vowel_detector/tests/test_patch.py`.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file
from torch.export import Dim

from _shared.paths import SERIES_ROOT
from karume.container import container_parts
from karume.pipeline import export_module, export_to_file, publish_model
from karume.verify import verify_container
from vowel_detector import export as vd
from vowel_detector.distribution import VOWEL_DETECTOR_GRAPH_ROLE, VOWEL_DETECTOR_WEIGHTS

#: Dimensions of the tiny synthetic weights (only the 83 feature dimensions match
#: {@link vd.build_cases}).
TINY_HIDDEN = 4
TINY_GRU_HIDDEN = 3

#: Length used by the tiny export (above `MIN_LENGTH`, and the GRU unrolling stays within a few
#: dozen nodes).
TINY_LENGTH = 8

#: Range of log-mel z values measured on a real utterance (1.1 seconds; the docstring of
#: {@link vd.build_cases}).
MEASURED_LOG_MEL_MIN = -1.04
MEASURED_LOG_MEL_MAX = 3.72

#: Regression case that goes beyond the range (the only case explicitly exempt from the range claim
#: above).
RAMP_CASE = "ramp"

#: The 22 weights (keys and shapes) of upstream's `training/src/vowel_detector/crnn.py` + the
#: trained `.pt`. **The contract table of the copy**: if the copy's structure drifts, it fails here.
UPSTREAM_PARAMETERS: dict[str, tuple[int, ...]] = {
    "conv.0.weight": (160, 83, 5),
    "conv.0.bias": (160,),
    "conv.2.weight": (160, 160, 3),
    "conv.2.bias": (160,),
    "gru.weight_ih_l0": (384, 160),
    "gru.weight_hh_l0": (384, 128),
    "gru.bias_ih_l0": (384,),
    "gru.bias_hh_l0": (384,),
    "gru.weight_ih_l0_reverse": (384, 160),
    "gru.weight_hh_l0_reverse": (384, 128),
    "gru.bias_ih_l0_reverse": (384,),
    "gru.bias_hh_l0_reverse": (384,),
    "gru.weight_ih_l1": (384, 256),
    "gru.weight_hh_l1": (384, 128),
    "gru.bias_ih_l1": (384,),
    "gru.bias_hh_l1": (384,),
    "gru.weight_ih_l1_reverse": (384, 256),
    "gru.weight_hh_l1_reverse": (384, 128),
    "gru.bias_ih_l1_reverse": (384,),
    "gru.bias_hh_l1_reverse": (384,),
    "head.weight": (8, 256),
    "head.bias": (8,),
}

#: Total number of upstream parameters (measured on `crnn_epoch3.pt`).
UPSTREAM_PARAMETER_COUNT = 664_744


def _probabilities(pau: float, vowel: float) -> torch.Tensor:
    """Logits `[1, 2, 8]` whose mean P(pau) and mean vowel mass are the given values.

    `_sanity` judges by the mass after softmax, so the log is taken to go back
    (the rest is split evenly between the 2 classes N / cons).
    """
    rest = (1.0 - pau - vowel) / 2.0
    row = [vowel / vd.VOWEL_COUNT] * vd.VOWEL_COUNT + [rest, pau, rest]
    return torch.log(torch.tensor([[row, row]]))


def _pooled(silence_pau: float, voiced_vowel: float) -> dict[str, torch.Tensor]:
    """Logits for the 4 cases (only silence's P(pau) and voiced's vowel mass vary)."""
    return {
        vd.SILENCE_CASE: _probabilities(silence_pau, 0.3),
        vd.VOICED_CASE: _probabilities(0.02, voiced_vowel),
        "noise": _probabilities(0.03, 0.1),
        "ramp": _probabilities(0.01, 0.2),
    }


@pytest.fixture
def tiny_module() -> vd.Crnn:
    """The same `Crnn` with only the dimensions shrunk (no real weights needed)."""
    torch.manual_seed(0)
    return vd.Crnn(hidden=TINY_HIDDEN, gru_hidden=TINY_GRU_HIDDEN).eval()


def _tiny_shapes() -> dict[str, dict[int, Dim]]:
    """**The same declaration** as the script (symbolic length `2T`)."""
    return {vd.INPUT_NAME: {1: 2 * Dim("T", min=vd.SYM_MIN, max=vd.SYM_MAX)}}


def _export_tiny(module: vd.Crnn, out_dir: Path):
    """Export one graph with **the same declaration** as the script."""
    example = vd.build_cases(TINY_LENGTH)[0][1]
    return export_to_file(
        module,
        (example,),
        out_dir / vd.MODEL_FILE,
        provenance=vd.PROVENANCE,
        graph_name="tiny",
        dynamic_shapes=_tiny_shapes(),
        symbol_names=("T",),
    )


@pytest.fixture
def exported(tmp_path: Path, tiny_module: vd.Crnn):
    """Export the tiny module once and return `(module, graph, out_dir)`."""
    return tiny_module, _export_tiny(tiny_module, tmp_path), tmp_path


class TestTranscription:
    def test_the_transcribed_definition_has_the_upstream_parameters(self) -> None:
        """MUST: the copy's keys and shapes match upstream (if they drift, the real weights cannot
        be loaded)."""
        state_dict = vd.Crnn().state_dict()

        assert {name: tuple(tensor.shape) for name, tensor in state_dict.items()} == (
            UPSTREAM_PARAMETERS
        )
        assert sum(tensor.numel() for tensor in state_dict.values()) == UPSTREAM_PARAMETER_COUNT

    def test_the_output_grid_is_half_the_input(self, tiny_module: vd.Crnn) -> None:
        """conv stride 2 folds 10 ms into 20 ms (the basis of the post-processing's frame_sec
        0.02)."""
        features = vd.build_cases(TINY_LENGTH)[0][1]

        with torch.no_grad():
            logits = tiny_module(features)

        assert tuple(logits.shape) == (1, TINY_LENGTH // 2, len(vd.CLASSES))

    def test_the_class_table_matches_the_upstream_order(self) -> None:
        """The class order is a convention shared with post-processing (if pau moves, sanity looks
        at something else)."""
        assert vd.CLASSES == ("a", "i", "u", "e", "o", "N", "pau", "cons")
        assert vd.CLASSES[vd.PAU_INDEX] == "pau"
        assert vd.CLASSES[: vd.VOWEL_COUNT] == ("a", "i", "u", "e", "o")


class TestSeriesLayout:
    def test_the_default_output_dir_is_a_series(self) -> None:
        """Series output goes under `outputs/series/` (not the distribution's `models/` —
        _shared.paths)."""
        assert vd.default_out_dir(vd.DEFAULT_CKPT).parent == SERIES_ROOT

    def test_the_series_name_does_not_carry_a_length(self) -> None:
        """There is one graph (the time axis is symbolic) — spelling a length would make it look
        like "an asset per length"."""
        assert vd.default_out_dir(vd.DEFAULT_CKPT).name == "vowel-detector-crnn-epoch3"

    def test_each_checkpoint_gets_its_own_series(self) -> None:
        """MUST: a different epoch is a different series (writing to the same location would erase
        the earlier weights)."""
        other = vd.MODELS_ROOT / "crnn_epoch2.pt"

        assert vd.default_out_dir(other) != vd.default_out_dir(vd.DEFAULT_CKPT)


class TestSymbolicLength:
    """The reason this script exists — length is **not part of what the graph accepts**."""

    def test_the_time_axis_is_symbolic(self, exported) -> None:
        """The input is `2T` (10 ms grid) and the output is `T` (20 ms grid)."""
        _module, graph, _out_dir = exported

        assert graph.symbols == ["T"]
        assert graph.inputs[0].shape == [1, "2T", vd.FEATURE_DIM]
        assert graph.values[graph.outputs[0]].shape == [1, "T", len(vd.CLASSES)]

    def test_the_graph_does_not_grow_with_the_traced_length(
        self, exported, tmp_path: Path, tiny_module: vd.Crnn
    ) -> None:
        """MUST: the node count does not depend on the length of the tracing example.

        If it did, the GRU would still be unrolled along time (= the symbol is not preserved).
        The earlier form had 38 nodes per input frame, 8,434 nodes at T10=200.
        """
        _module, short, _out_dir = exported
        long_dir = tmp_path / "long"
        long_dir.mkdir()
        example = vd.build_cases(TINY_LENGTH * 4)[0][1]
        long = export_to_file(
            tiny_module,
            (example,),
            long_dir / vd.MODEL_FILE,
            provenance=vd.PROVENANCE,
            graph_name="tiny",
            dynamic_shapes=_tiny_shapes(),
            symbol_names=("T",),
        )

        assert [node.op for node in long.nodes] == [node.op for node in short.nodes]

    def test_the_scan_ops_are_one_node_per_layer_and_direction(self, exported) -> None:
        """Multiple layers and both directions are expressed **as a sequence of nodes** (ADR 0056
        decision 7 — no variable arity)."""
        _module, graph, _out_dir = exported
        ops = [node.op for node in graph.nodes]

        assert ops.count("gru_scan") == vd.GRU_LAYERS
        assert ops.count("gru_scan_reverse") == vd.GRU_LAYERS
        # One `cat` per layer joining both directions.
        assert ops.count("cat") == vd.GRU_LAYERS


class TestLength:
    @pytest.mark.parametrize("length", [201, 3, 0, -2])
    def test_a_length_off_the_grid_fails_loudly(self, length: int) -> None:
        """An odd length leaves a partial frame holding only 10 ms at the end (breaking the 20 ms
        grid)."""
        with pytest.raises(SystemExit, match="20 ms grid"):
            vd.assert_length(length)

    @pytest.mark.parametrize("length", [vd.MIN_LENGTH, 200, 1000])
    def test_a_length_on_the_grid_passes(self, length: int) -> None:
        vd.assert_length(length)


class TestCheckpoint:
    def test_a_missing_checkpoint_fails_loudly(self, tmp_path: Path) -> None:
        with pytest.raises(SystemExit, match="checkpoint not found"):
            vd.load_checkpoint(tmp_path / "absent.pt")

    def test_an_unexpected_wrapper_fails_loudly(self, tmp_path: Path) -> None:
        """MUST: stop when the outer shape changes (do not silently treat another key as the
        state_dict)."""
        path = tmp_path / "ckpt.pt"
        torch.save({"state_dict": {}, "epoch": 0}, path)

        with pytest.raises(SystemExit, match="the outer wrapper keys"):
            vd.load_checkpoint(path)

    def test_it_loads_on_cpu(self, tmp_path: Path, tiny_module: vd.Crnn) -> None:
        """MUST: `map_location="cpu"` (the upstream training script saves on CUDA)."""
        path = tmp_path / "ckpt.pt"
        torch.save({vd.STATE_DICT_KEY: tiny_module.state_dict(), "epoch": 3}, path)

        state_dict = vd.load_checkpoint(path)

        assert set(state_dict) == set(tiny_module.state_dict())
        assert all(tensor.device.type == "cpu" for tensor in state_dict.values())

    def test_a_shape_mismatch_is_caught_by_strict_loading(self, tiny_module: vd.Crnn) -> None:
        """MUST: `strict=True` catches a drifted copy (it does not silently load only a part)."""
        with pytest.raises(RuntimeError, match=r"size mismatch|Missing key"):
            vd.load_module(tiny_module.state_dict())


class TestCheckpointBytes:
    def test_every_checkpoint_tensor_is_byte_identical(self, exported) -> None:
        module, _graph, out_dir = exported

        matched = vd.assert_checkpoint_bytes(out_dir / vd.MODEL_FILE, module.state_dict())

        assert matched == len(module.state_dict())

    def test_a_changed_byte_fails_loudly(self, exported) -> None:
        """MUST: fail when a value changes (conversion is "read and write only" — values do not
        change)."""
        module, _graph, out_dir = exported
        state_dict = dict(module.state_dict())
        tampered = state_dict["head.bias"].clone()
        tampered[0] += 1.0
        state_dict["head.bias"] = tampered

        with pytest.raises(AssertionError, match="the bytes do not match"):
            vd.assert_checkpoint_bytes(out_dir / vd.MODEL_FILE, state_dict)

    def test_a_missing_initializer_fails_loudly(self, exported) -> None:
        """MUST: fail if even one ckpt key is missing from the container (do not silently compare
        only a part)."""
        module, _graph, out_dir = exported
        state_dict = dict(module.state_dict())
        state_dict["conv.0.absent"] = state_dict["conv.0.bias"].clone()

        with pytest.raises(AssertionError, match="missing from the initializers"):
            vd.assert_checkpoint_bytes(out_dir / vd.MODEL_FILE, state_dict)

    def test_a_weight_that_is_not_in_the_checkpoint_fails_loudly(self, exported) -> None:
        """MUST: catch a weight stored under another name (only folded constants may carry
        `const.`)."""
        module, _graph, out_dir = exported
        state_dict = dict(module.state_dict())
        del state_dict["head.bias"]

        with pytest.raises(AssertionError, match="initializers that do not come from the weights"):
            vd.assert_checkpoint_bytes(out_dir / vd.MODEL_FILE, state_dict)


class TestGoldenCases:
    def test_cases_have_the_declared_shape_and_dtype(self) -> None:
        for name, features in vd.build_cases(TINY_LENGTH):
            assert tuple(features.shape) == (1, TINY_LENGTH, vd.FEATURE_DIM), name
            assert features.dtype is torch.float32, name

    def test_the_dsp_dimensions_stay_in_the_measured_range(self) -> None:
        """MUST: do not take the golden with features outside the range of real speech (that would
        compare on another distribution)."""
        for name, features in vd.build_cases(TINY_LENGTH):
            voicing, log_energy, zcr = (features[0, 0, vd.N_MELS + i].item() for i in range(3))
            assert 0.0 <= voicing <= 1.0, name
            assert -1.0 <= log_energy <= 0.0, name
            assert 0.0 <= zcr <= 1.0, name

    def test_the_log_mel_plane_stays_in_the_measured_range_except_for_the_ramp(self) -> None:
        """log-mel is subject to the same claim (checking only the 3 DSP dimensions would leave the
        80 dimensions unchecked).

        `ramp` alone is a regression case that goes beyond the range, so it is excluded by name
        (the same split as the docstring of {@link vowel_detector.export.build_cases}).
        """
        for name, features in vd.build_cases(TINY_LENGTH):
            if name == RAMP_CASE:
                continue
            log_mel = features[0, :, : vd.N_MELS]
            assert float(log_mel.min()) >= MEASURED_LOG_MEL_MIN, name
            assert float(log_mel.max()) <= MEASURED_LOG_MEL_MAX, name

    def test_the_ramp_case_deliberately_leaves_the_measured_range(self) -> None:
        """The exclusion has not degenerated into "it was all inside anyway" (keeps the exclusion
        above from being vacuous)."""
        ramp = dict(vd.build_cases(TINY_LENGTH))[RAMP_CASE][0, :, : vd.N_MELS]

        assert float(ramp.min()) < MEASURED_LOG_MEL_MIN
        assert float(ramp.max()) > MEASURED_LOG_MEL_MAX

    def test_every_case_is_a_different_input(self) -> None:
        """MUST: do not use the same features twice — the sanity order would become vacuous."""
        cases = vd.build_cases(TINY_LENGTH)

        for index, (name, features) in enumerate(cases):
            for other_name, other in cases[index + 1 :]:
                assert not torch.equal(features, other), f"{name} and {other_name} are identical"

    def test_the_sanity_cases_exist(self) -> None:
        names = {name for name, _ in vd.build_cases(TINY_LENGTH)}

        assert {vd.SILENCE_CASE, vd.VOICED_CASE} <= names


class TestWriteIo:
    def test_writes_one_file_per_case_with_the_declared_keys(self, exported) -> None:
        module, graph, out_dir = exported
        cases = vd.build_cases(TINY_LENGTH)

        written, logits = vd._write_io(module, graph, cases, out_dir)

        assert written == [f"{vd.IO_PREFIX}{name}{vd.IO_SUFFIX}" for name, _ in cases]
        tensors = load_file(str(out_dir / written[0]))
        assert set(tensors) == {f"{vd.INPUT_PREFIX}{vd.INPUT_NAME}", f"{vd.OUTPUT_PREFIX}0"}
        assert tuple(logits[vd.SILENCE_CASE].shape) == (1, TINY_LENGTH // 2, len(vd.CLASSES))

    def test_the_expected_values_come_from_the_reference_path(self, exported) -> None:
        """MUST: the expected values come from the `nn.GRU` path (not the replacement path).

        The two are bit-identical, so values cannot tell them apart — the method is swapped to see
        **which one was called**.
        """
        module, graph, out_dir = exported
        seen: list[str] = []
        original = module.reference_forward
        module.reference_forward = lambda features: (  # type: ignore[method-assign]
            seen.append("reference") or original(features)
        )

        vd._write_io(module, graph, vd.build_cases(TINY_LENGTH)[:1], out_dir)

        assert seen == ["reference"]

    def test_a_patched_path_that_diverges_fails_loudly(self, exported) -> None:
        """MUST: stop an emit whose replacement is not bit-identical to `nn.GRU` (the standing
        check)."""
        module, graph, out_dir = exported
        original = module.forward
        module.forward = lambda features: original(features) + 1e-6  # type: ignore[method-assign]

        with pytest.raises(AssertionError, match=r"not bit-identical to nn\.GRU"):
            vd._write_io(module, graph, vd.build_cases(TINY_LENGTH)[:1], out_dir)

    def test_more_than_one_graph_output_fails_loudly(self, tmp_path: Path) -> None:
        """MUST: the output is one logits tensor — a second one would silently shift io's positional
        convention."""

        class TwoOutputs(vd.Crnn):
            def forward(self, features: torch.Tensor):  # type: ignore[override]
                logits = super().forward(features)
                return logits, logits * 2.0

        torch.manual_seed(0)
        module = TwoOutputs(hidden=TINY_HIDDEN, gru_hidden=TINY_GRU_HIDDEN).eval()
        graph = _export_tiny(module, tmp_path)

        with pytest.raises(AssertionError, match="logits are one output"):
            vd._write_io(module, graph, vd.build_cases(TINY_LENGTH), tmp_path)


class TestSanity:
    def test_passes_when_both_orders_hold(self) -> None:
        result = vd._sanity(_pooled(silence_pau=0.2, voiced_vowel=0.7))

        assert result["pau_mean"][vd.SILENCE_CASE] > result["pau_mean"][vd.VOICED_CASE]
        assert result["vowel_mass_mean"][vd.VOICED_CASE] > result["vowel_mass_mean"]["noise"]

    def test_fails_loudly_when_silence_is_not_the_most_paused(self) -> None:
        """MUST: if pau does not rise on silence, the weight layout or the way input arrives is
        broken."""
        with pytest.raises(AssertionError, match=r"mean P\(pau\) is not the largest"):
            vd._sanity(_pooled(silence_pau=0.01, voiced_vowel=0.7))

    def test_fails_loudly_when_voiced_is_not_the_most_vowel_like(self) -> None:
        with pytest.raises(AssertionError, match="mean vowel mass is not the largest"):
            vd._sanity(_pooled(silence_pau=0.2, voiced_vowel=0.05))

    def test_fails_loudly_when_every_case_collapses_to_one_output(self) -> None:
        """MUST: catch an input that has no effect (4 identical values verify nothing)."""
        same = _probabilities(0.2, 0.3)
        pooled = dict.fromkeys((vd.SILENCE_CASE, vd.VOICED_CASE, "noise", "ramp"), same)

        with pytest.raises(AssertionError, match="the input has no effect"):
            vd._sanity(pooled)


class TestCli:
    def test_verify_does_not_emit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen: list[str] = []
        monkeypatch.setattr(
            vd,
            "verify_patch",
            lambda _ckpt, _length: (
                seen.append("verify")
                or [
                    {
                        "stage": "gru-scan-t8",
                        "claim": "bit-exact",
                        "bit_exact": True,
                        "maxdiff": {},
                    }
                ]
            ),
        )
        monkeypatch.setattr(vd, "export_series", lambda *_a, **_kw: pytest.fail("--verify emitted"))

        vd.main(["--verify"])

        assert seen == ["verify"]

    def test_without_verify_it_emits(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen: list[str] = []
        monkeypatch.setattr(
            vd, "export_series", lambda *_a, **_kw: seen.append("emit") or {"dir": "x"}
        )
        monkeypatch.setattr(vd, "verify_patch", lambda *_a: pytest.fail("an emit ran --verify"))

        vd.main([])

        assert seen == ["emit"]

    def test_the_output_dir_does_not_follow_the_golden_length(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """MUST: the series **does not follow** `--length` — there is one graph, and the length is
        about the golden only."""
        seen: list[Path] = []
        monkeypatch.setattr(
            vd, "export_series", lambda _ckpt, out, _length: seen.append(out) or {"dir": str(out)}
        )

        vd.main(["--length", "500"])

        assert seen == [vd.default_out_dir(vd.DEFAULT_CKPT)]


class TestCheckpointBytesOfASplitComponent:
    """The comparison holds for a distribution that contains split tensors (pieces —
    container-v1 §4.2) too.

    The block limit is lowered artificially so that the conv weight (1,660 bytes per row even when
    tiny) is split by rows. A reader that does not fold them back would turn every state_dict key
    into "missing", so this one test decides whether the reassembly happens.
    """

    def split_export(self, tmp_path: Path, module: vd.Crnn) -> Path:
        """Write with the block limit set to 2 KiB (the size parameter of `publish_model`)."""
        example = vd.build_cases(TINY_LENGTH)[0][1]
        graph, tensors = export_module(
            module, (example,), dynamic_shapes=_tiny_shapes(), symbol_names=("T",)
        )
        publish_model(
            tmp_path / vd.MODEL_FILE,
            graph,
            tensors,
            provenance=vd.PROVENANCE,
            graph_name="tiny",
            _block_bytes=2048,
        )
        return tmp_path / vd.MODEL_FILE

    def test_the_component_really_carries_pieces(self, tmp_path, tiny_module) -> None:
        """Observing the precondition — without a single piece this test would check nothing."""
        path = self.split_export(tmp_path, tiny_module)

        verified = verify_container(container_parts(path))
        split = [
            name
            for bound in verified.graphs.values()
            for name, supply in bound.supplies.items()
            if len(supply.blocks) > 1
        ]
        assert split

    def test_every_checkpoint_tensor_is_still_byte_identical(self, tmp_path, tiny_module) -> None:
        path = self.split_export(tmp_path, tiny_module)

        matched = vd.assert_checkpoint_bytes(path, tiny_module.state_dict())

        assert matched == len(tiny_module.state_dict())

    def test_a_changed_byte_still_fails_loudly(self, tmp_path, tiny_module) -> None:
        """The comparison is on the folded data (splitting does not loosen the check)."""
        path = self.split_export(tmp_path, tiny_module)
        state_dict = dict(tiny_module.state_dict())
        tampered = state_dict["conv.0.weight"].clone()
        tampered[0, 0, 0] += 1.0
        state_dict["conv.0.weight"] = tampered

        with pytest.raises(AssertionError, match="the bytes do not match"):
            vd.assert_checkpoint_bytes(path, state_dict)


class TestStagedPublication:
    """MUST: publish only after every check passes (a failed run is removed together with its
    staging directory).

    The core measures the discipline of `staged_publication` on its own, but the family-side wiring
    (where `staged.mkdir()` sits, the order in which the checks are evaluated) is exercised only
    here. If a failed run left assets in the final location, the io golden would be taken from the
    same broken module, and the TS-side comparison would go green.
    """

    @staticmethod
    def _stage_tiny(monkeypatch, module: vd.Crnn) -> None:
        """Set things up so that `export_series` can run once without real weights."""
        monkeypatch.setattr(vd, "load_checkpoint", lambda _ckpt: {})
        monkeypatch.setattr(vd, "load_module", lambda _state: module)
        monkeypatch.setattr(vd, "assert_checkpoint_bytes", lambda _path, _state: 0)

    def test_a_passing_run_leaves_the_series_in_place(
        self, monkeypatch, tmp_path: Path, tiny_module: vd.Crnn
    ) -> None:
        """The counterpart that shows it is not vacuous (passing the checks publishes) — without it
        the assertions below would be vacuous."""
        self._stage_tiny(monkeypatch, tiny_module)
        monkeypatch.setattr(vd, "_sanity", lambda _logits: {})
        out_dir = tmp_path / "series"

        summary = vd.export_series(tmp_path / "ckpt.pt", out_dir, TINY_LENGTH)

        assert out_dir.is_dir()
        assert summary["dir"] == str(out_dir)

    def test_a_failing_sanity_leaves_nothing_behind(
        self, monkeypatch, tmp_path: Path, tiny_module: vd.Crnn
    ) -> None:
        self._stage_tiny(monkeypatch, tiny_module)

        def _reject(_logits):
            raise AssertionError("the order is broken")

        monkeypatch.setattr(vd, "_sanity", _reject)
        out_dir = tmp_path / "series"

        with pytest.raises(AssertionError, match="the order is broken"):
            vd.export_series(tmp_path / "ckpt.pt", out_dir, TINY_LENGTH)

        assert not out_dir.exists()

    def test_a_failing_checkpoint_byte_gate_leaves_nothing_behind(
        self, monkeypatch, tmp_path: Path, tiny_module: vd.Crnn
    ) -> None:
        """The bit-identity check (the copy did not change the values) is also evaluated before
        publishing."""
        self._stage_tiny(monkeypatch, tiny_module)

        def _reject(_path, _state):
            raise AssertionError("the bytes do not match")

        monkeypatch.setattr(vd, "assert_checkpoint_bytes", _reject)
        monkeypatch.setattr(vd, "_sanity", lambda _logits: {})
        out_dir = tmp_path / "series"

        with pytest.raises(AssertionError, match="the bytes do not match"):
            vd.export_series(tmp_path / "ckpt.pt", out_dir, TINY_LENGTH)

        assert not out_dir.exists()


class TestGraphNameIsThePartName:
    """The graph name of the published container is the **component name** (= the weights key of
    `karume.json`).

    The runtime looks a graph up by name with `prepareContainer(opened, <weights key>)`
    (container-v1 §2.1). This family's container sits directly in the series, so **the directory
    name is the series name**, not the component name, and a writer that derived it from the
    directory would silently bake "a container the runtime cannot look up" (the writer does not
    check the graph name against the weights key — that check is the assembly check of
    `karume dist`, which is reached only after running the export with the real weights).

    The spelling check across all families is applied on the AST by `tests/test_graph_names.py`,
    but **writing the real thing and reading it back** happens only here — the location is spelled
    differently from the component name, so going back to deriving it fails.
    """

    def test_the_published_container_names_the_weights_key(
        self, monkeypatch, tmp_path: Path, tiny_module: vd.Crnn
    ) -> None:
        TestStagedPublication._stage_tiny(monkeypatch, tiny_module)
        monkeypatch.setattr(vd, "_sanity", lambda _logits: {})
        # A location spelled differently from the component name (`crnn`) — the same shape as the
        # real series name.
        out_dir = tmp_path / "vowel-detector-jsut-crnn"

        vd.export_series(tmp_path / "ckpt.pt", out_dir, TINY_LENGTH)

        read = verify_container(container_parts(out_dir / vd.MODEL_FILE)).read

        assert list(read.graph.graphs) == [VOWEL_DETECTOR_GRAPH_ROLE]
        assert VOWEL_DETECTOR_GRAPH_ROLE in VOWEL_DETECTOR_WEIGHTS
