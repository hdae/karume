"""Pins the contract of the SBV2 export script (dp / front / flow / dec / voice) (M1-P3 iterations
1, 6, 7).

The first half (how golden inputs are made and the CLI exclusivity) needs no real weights and
always runs. The second half runs only where the real weights under `inputs/sbv2/F1/` and the
`sbv2` dependency group are present, and SKIPs otherwise — the weights are about 251MB and outside
the repository, and the dependencies are not in the default `uv sync`.

NOTE: the front / flow / voice exports apply the patches process-wide, and dec / voice fold the
weights with `remove_weight_norm`. The dp tests are affected by neither (they go through no MHA /
FFN / spline and are not dec), so there is no order dependence, but **only the checks that take
"the reference from before" run in a separate process** (a subprocess run of
`--verify <target>`) — the exclusivity rule of ADR 0013 itself.
"""

from __future__ import annotations

import importlib.util
import inspect
import json
import math
import re
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from safetensors import safe_open
from torch import nn

from _shared.paths import REPO_ROOT, SERIES_ROOT
from karume.container import codec_entry, container_parts
from karume.dims import parse_dim
from karume.emit import EmitError
from karume.ops import EMITTABLE_OPS
from karume.quantize import (
    QUANT_CHANNEL_AXES,
    channel_scale,
    fake_quant_int8,
    quantize_to_int8,
)
from karume.verify import verify_container
from sbv2 import export as export_sbv2
from sbv2 import patch as patch_sbv2
from sbv2.card import SBV2_CARD_PROFILES

MODEL_DIR = export_sbv2.DEFAULT_MODEL_DIR


def _speaker_dir(root: Path) -> Path:
    """A `--model-dir` following the convention (a jvnv speaker id —
    `export_sbv2.SBV2_FAMILY_DIRS`).

    Even the synthetic setup follows the spelling convention because `sbv2_provenance` derives
    **legal facts** from it (a name of no family fails loudly — it never silently claims one side's
    license).
    """
    speaker = root / export_sbv2.SBV2_FAMILY_DIRS["jvnv"][0]
    speaker.mkdir(parents=True, exist_ok=True)
    return speaker


_WEIGHTS_PRESENT = (
    MODEL_DIR.is_dir()
    and any(MODEL_DIR.glob("*.safetensors"))
    and (MODEL_DIR / export_sbv2.CONFIG_FILE).is_file()
)
_PACKAGE_PRESENT = importlib.util.find_spec("style_bert_vits2") is not None


def _verified_graph(path: Path) -> Mapping[str, Any]:
    """Verifies the placed container **by every rule** and returns the IR v2 graph document.

    Replaces `verify_model` of `karume/4`. It recomputes even the blocks' sha256 (`blocks=True`),
    so a distributed form that "was written but cannot be read" fails here. `karume/5` is one graph
    per container (ADR 0109 decision 2), so the only value of `graphs` is returned.
    """
    verified = verify_container(container_parts(path), blocks=True)
    return next(iter(verified.read.graph.graphs.values()))


def _storage_layouts(path: Path) -> dict[str, str]:
    """Tensor key → storage layout (the IR v2 graph description holds no storage — the binding table
    is the source of truth)."""
    verified = verify_container(container_parts(path))
    return {
        name: codec_entry(supply.encoding.codec).layout
        for bound in verified.graphs.values()
        for name, supply in bound.supplies.items()
    }


def _scale_blocks(path: Path) -> dict[str, str | None]:
    """Tensor key → block id of the companion scale (`None` when not quantized)."""
    verified = verify_container(container_parts(path))
    return {
        name: (None if supply.scale is None else supply.scale.id)
        for bound in verified.graphs.values()
        for name, supply in bound.supplies.items()
    }


requires_weights = pytest.mark.skipif(
    not (_WEIGHTS_PRESENT and _PACKAGE_PRESENT),
    reason=(
        f"no real weights ({MODEL_DIR}) or no sbv2 dependency group (requires"
        " `uv sync --all-groups` and the weights in place — tools/export-recipes/sbv2/README.md)"
    ),
)


class TestGoldenInputs:
    """Generating golden inputs does not depend on real weights — this always runs."""

    def test_hidden_has_the_encoder_output_shape(self):
        assert export_sbv2.make_hidden(37).shape == (1, 192, 37)
        assert export_sbv2.make_hidden(37).dtype == torch.float32

    def test_hidden_is_reproducible_for_the_same_length(self):
        # Byte-identical reproduction of the output is the premise of a golden (no dependence on the
        # global seed).
        assert torch.equal(export_sbv2.make_hidden(37), export_sbv2.make_hidden(37))

    def test_hidden_differs_between_lengths(self):
        # A seed is derived per length (reusing prefixes of the same sequence would make the cases
        # with other P "slices of the same input", lowering the power to detect P-dependent bugs).
        short = export_sbv2.make_hidden(37)[..., :16]
        assert not torch.equal(short, export_sbv2.make_hidden(16))

    def test_mask_is_all_ones_without_padding(self):
        assert torch.equal(export_sbv2.make_mask(16, 0), torch.ones(1, 1, 16))

    def test_mask_zeroes_exactly_the_trailing_padding(self):
        mask = export_sbv2.make_mask(16, 5)
        assert mask.shape == (1, 1, 16)
        assert torch.equal(mask[..., :11], torch.ones(1, 1, 11))
        assert torch.equal(mask[..., 11:], torch.zeros(1, 1, 5))

    @pytest.mark.parametrize("pad", [-1, 16, 20])
    def test_out_of_range_padding_is_rejected(self, pad):
        with pytest.raises(ValueError):
            export_sbv2.make_mask(16, pad)

    def test_case_table_covers_the_declared_range_and_the_mask_path(self):
        lengths = [length for _, length, _ in export_sbv2.GOLDEN_CASES]
        padded = [name for name, _, pad in export_sbv2.GOLDEN_CASES if pad > 0]

        # Exercise both ends: the lower bound 2 (the line that avoids 0/1 specialization) and the
        # declared upper bound SYM_MAX.
        assert min(lengths) == 2
        assert max(lengths) == export_sbv2.SYM_MAX
        assert len(set(lengths)) == len(lengths)
        # Exactly one case has zeros in x_mask (the only detector of the mask path).
        assert padded

    def test_case_inputs_are_keyed_by_the_graph_input_names(self):
        g = torch.zeros(1, 512, 1)
        built = export_sbv2.build_cases(g)

        assert [name for name, _ in built] == [name for name, _, _ in export_sbv2.GOLDEN_CASES]
        for name, args in built:
            assert set(args) == {"h", "x_mask", "g"}, name
            assert args["g"] is g, name


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    """Exports dp into a temporary directory (does not touch the repository's outputs)."""
    out = tmp_path_factory.mktemp("sbv2") / export_sbv2.TARGET_DP
    return out, export_sbv2.export_dp(MODEL_DIR, out)


@pytest.fixture(scope="module")
def net_g():
    return export_sbv2.load_net_g(MODEL_DIR)[0]


@requires_weights
class TestDurationPredictorExport:
    def test_the_container_passes_the_full_verification(self, exported):
        out, _ = exported

        _verified_graph(out / export_sbv2.MODEL_FILE)

    def test_the_graph_declares_the_wrapper_argument_names(self, exported):
        out, summary = exported
        graph = _verified_graph(out / export_sbv2.MODEL_FILE)

        assert [spec["name"] for spec in graph["inputs"]] == ["h", "x_mask", "g"]
        assert [spec["shape"] for spec in graph["inputs"]] == [
            [1, 192, "P"],
            [1, 1, "P"],
            [1, 512, 1],
        ]
        assert summary["symbols"] == ["P"]

    def test_the_graph_stays_inside_the_current_contract(self, exported):
        _, summary = exported

        # The premise of iteration 1 itself: dp passes without widening the vocabulary (recon §2).
        # If it grows, fail loudly.
        assert set(summary["ops"]) <= set(EMITTABLE_OPS)
        assert set(summary["ops"]) == {"add", "conv1d", "layer_norm", "mul", "permute", "relu"}

    def test_the_io_files_follow_the_naming_convention(self, exported):
        out, summary = exported

        expected = [
            f"{export_sbv2.IO_PREFIX}{name}{export_sbv2.IO_SUFFIX}"
            for name, _, _ in export_sbv2.GOLDEN_CASES
        ]
        assert summary["io"] == expected
        for name in expected:
            with safe_open(str(out / name), framework="pt") as handle:
                keys = set(handle.keys())
            assert keys == {"input.h", "input.x_mask", "input.g", "output.0"}

    @pytest.mark.parametrize("case", export_sbv2.GOLDEN_CASES, ids=lambda case: case[0])
    def test_io_shapes_bind_the_symbol_to_the_case_length(self, exported, case):
        out, _ = exported
        name, length, _ = case
        graph = _verified_graph(out / export_sbv2.MODEL_FILE)
        tensors = _io_tensors(out, name)

        for spec in graph["inputs"]:
            actual = list(tensors[f"input.{spec['name']}"].shape)
            declared = [
                length if isinstance(dim, str) and parse_dim(dim).sym == "P" else dim
                for dim in spec["shape"]
            ]
            assert actual == declared, spec["name"]
        assert list(tensors["output.0"].shape) == [1, 1, length]
        assert tensors["output.0"].dtype == torch.float32

    @pytest.mark.parametrize("case", export_sbv2.GOLDEN_CASES, ids=lambda case: case[0])
    def test_the_golden_output_reproduces_a_fresh_eager_forward(self, exported, net_g, case):
        """The stored expected values are bit-identical to torch's output for the same inputs.

        `_write_io` mixing up cases / shifting the input-output pairing fails here (P differs
        across cases, so a shift already breaks the shapes).
        """
        out, _ = exported
        name, _, _ = case
        module = export_sbv2.DurationPredictorGraph(net_g.dp)
        tensors = _io_tensors(out, name)
        with torch.no_grad():
            fresh = module(tensors["input.h"], tensors["input.x_mask"], tensors["input.g"])

        assert torch.equal(fresh, tensors["output.0"])

    @pytest.mark.parametrize("case", export_sbv2.GOLDEN_CASES, ids=lambda case: case[0])
    def test_a_corrupted_golden_is_detected(self, exported, net_g, case):
        """Fault injection — pins that corrupting one element of the expected values turns the test
        above red.

        Forms that overlook a corrupted value (reading another tensor through a wrong key /
        comparing with itself) cannot be detected by the test above alone (it stays vacuously
        green). This is the guard against a vacuous check.
        """
        out, _ = exported
        name, _, _ = case
        module = export_sbv2.DurationPredictorGraph(net_g.dp)
        tensors = _io_tensors(out, name)
        with torch.no_grad():
            fresh = module(tensors["input.h"], tensors["input.x_mask"], tensors["input.g"])
        corrupted = tensors["output.0"].clone()
        corrupted[0, 0, 0] += 1e-3

        assert not torch.equal(fresh, corrupted)

    def test_the_padded_case_zeroes_the_masked_tail(self, exported):
        """If the 4 places of `x * x_mask` take effect, the output of the padding columns is exactly
        0."""
        out, _ = exported
        name, length, pad = next(case for case in export_sbv2.GOLDEN_CASES if case[2] > 0)
        output = _io_tensors(out, name)["output.0"]

        assert torch.equal(output[..., length - pad :], torch.zeros(1, 1, pad))

    def test_the_masked_tail_assertion_is_not_vacuous(self, exported, net_g):
        """Fault injection — replacing the mask with all ones makes the tail non-zero.

        h has values in the padding columns too, so without the mask values leak. Without this,
        the test above could be green even for "a form that is always 0 anyway".
        """
        out, _ = exported
        name, length, pad = next(case for case in export_sbv2.GOLDEN_CASES if case[2] > 0)
        module = export_sbv2.DurationPredictorGraph(net_g.dp)
        tensors = _io_tensors(out, name)
        with torch.no_grad():
            leaked = module(tensors["input.h"], torch.ones(1, 1, length), tensors["input.g"])

        assert not torch.equal(leaked[..., length - pad :], torch.zeros(1, 1, pad))

    def test_regeneration_is_byte_identical(self, exported, tmp_path):
        """With a fixed seed the output is byte-identical in the same environment (the premise of a
        golden)."""
        out, _ = exported
        again = tmp_path / export_sbv2.TARGET_DP
        export_sbv2.export_dp(MODEL_DIR, again)

        for name in sorted(path.name for path in out.iterdir()):
            assert (again / name).read_bytes() == (out / name).read_bytes(), name


class TestWindowSizeGate:
    """The window-size check scans **the whole** net_g (no real weights needed — synthetic modules
    exercise it per layer).

    In the real weights every layer has the same window size, so the real-weight tests would be
    green even with "a check that looks only at the first layer". Only the synthetic side, which
    can build a model with a different window size per layer, can detect a shrunken scan range.
    """

    @staticmethod
    def _layered(sizes) -> torch.nn.Module:
        """A nested module that only has `attn_layers[i].window_size = sizes[i]`."""
        root = torch.nn.Module()
        layers = torch.nn.ModuleList()
        for size in sizes:
            layer = torch.nn.Module()
            layer.window_size = size
            layers.append(layer)
        root.attn_layers = layers
        return root

    def test_a_uniform_expected_window_size_passes(self):
        export_sbv2._assert_window_size(self._layered([export_sbv2.EXPECTED_WINDOW_SIZE] * 3))

    def test_a_mismatch_in_a_later_layer_is_rejected(self):
        """Even when the first layer matches the premise, a mismatch in a deeper layer fails (the
        detector of the scan range itself).

        `FlowReverse` hands **the same table** to every coupling × every layer, so if even one
        layer has another window size, that layer silently reads embeddings of another width. A
        check that looks only at the first layer lets this form through.
        """
        sizes = [export_sbv2.EXPECTED_WINDOW_SIZE] * 3
        sizes[-1] += 1

        with pytest.raises(ValueError, match="窓幅"):
            export_sbv2._assert_window_size(self._layered(sizes))

    def test_layers_without_relative_attention_are_ignored(self):
        """Layers with `window_size = None` are skipped (layers without relative-position attention
        have this form)."""
        export_sbv2._assert_window_size(self._layered([export_sbv2.EXPECTED_WINDOW_SIZE, None]))

    def test_a_model_without_any_window_size_is_rejected(self):
        """Guard against a vacuous check — fails when an attribute rename makes the scan find
        nothing."""
        with pytest.raises(ValueError, match="恒真化"):
            export_sbv2._assert_window_size(torch.nn.Module())


@requires_weights
class TestLoadAsserts:
    def test_a_window_size_mismatch_is_rejected(self):
        """Fault injection — a net_g whose window size drifted from the premise fails at load (ADR
        0013).

        A window-size mismatch raises no shape error and silently reads wrong embeddings — the
        "silent wrong value" class — so this pins that the assert really fires.
        """
        net_g = export_sbv2.load_net_g(MODEL_DIR)[0]
        attn = net_g.flow.flows[0].enc.attn_layers[0]
        original = attn.window_size
        attn.window_size = original + 1
        try:
            with pytest.raises(ValueError, match="窓幅"):
                export_sbv2._assert_window_size(net_g)
        finally:
            attn.window_size = original

    def test_a_residual_weight_norm_parameter_is_rejected(self):
        """Fault injection — fails when a parameter derived from weight_norm remains."""
        module = torch.nn.Module()
        module.register_parameter("weight_v", torch.nn.Parameter(torch.zeros(1)))

        with pytest.raises(ValueError, match="weight_norm"):
            export_sbv2._assert_no_weight_norm(module)

    def test_the_speaker_id_is_range_checked(self):
        net_g = export_sbv2.load_net_g(MODEL_DIR)[0]

        with pytest.raises(ValueError, match="話者"):
            export_sbv2.speaker_embedding(net_g, net_g.emb_g.num_embeddings)

    def test_the_speaker_embedding_comes_from_the_real_weights(self):
        net_g = export_sbv2.load_net_g(MODEL_DIR)[0]
        g = export_sbv2.speaker_embedding(net_g)

        assert g.shape == (1, 512, 1)
        # If taken from the real weights it is not a zero vector (detects uninitialized values and
        # mix-ups).
        assert float(g.abs().max()) > 0


class TestVerifyExclusivity:
    """The checks against process-wide patch effects (no real weights needed — only argument parsing
    and the applied flag are looked at)."""

    def test_the_cli_refuses_verify_together_with_an_emit_target(self, monkeypatch):
        """Combining `--verify` with emit is rejected at argument parsing (ADR 0013).

        If combined, the emit side would apply the patches first and `--verify`'s "reference
        before patching" would already be the patched values — the difference would always be 0,
        a **false PASS**.
        """
        monkeypatch.setattr(
            sys, "argv", ["export_sbv2.py", "--verify", "front", "--target", "front"]
        )

        with pytest.raises(SystemExit) as raised:
            export_sbv2.main()

        assert raised.value.code == 2

    def test_the_cli_takes_exactly_one_verify_target(self, monkeypatch):
        """One target per verify process — argument parsing structurally refuses more than one.

        With a pairwise exclusion table ("MHA-patched targets exclude each other / dec's remove
        excludes only voice / rounding excludes everything"), any hole in the table would become a
        false PASS. A `--verify` that takes one value has **no combinations that can contaminate**
        (the implementation of ADR 0013's exclusivity rule).
        """
        monkeypatch.setattr(sys, "argv", ["export_sbv2.py", "--verify", "flow", "voice"])

        with pytest.raises(SystemExit) as raised:
            export_sbv2.main()

        assert raised.value.code == 2

    def test_the_cli_refuses_an_unknown_verify_target(self, monkeypatch):
        """dp is not a verify target (it goes through neither patches nor preprocessing, so the
        reference and the subject are identical = vacuous)."""
        monkeypatch.setattr(sys, "argv", ["export_sbv2.py", "--verify", "dp"])

        with pytest.raises(SystemExit) as raised:
            export_sbv2.main()

        assert raised.value.code == 2
        assert set(export_sbv2.VERIFIERS) == set(export_sbv2.TARGETS) - {export_sbv2.TARGET_DP}

    def test_taking_a_reference_after_patching_is_rejected(self, monkeypatch):
        """Detects the order violation itself (misuse from paths that bypass the CLI fails too)."""
        monkeypatch.setattr(patch_sbv2, "_APPLIED", True)

        with pytest.raises(RuntimeError, match="恒真化"):
            export_sbv2.reference_front_outputs(object(), {})

    @pytest.mark.parametrize("where", ["front", "flow", "voice"])
    def test_the_patch_gate_is_shared_by_every_mha_target(self, monkeypatch, where):
        """The patch-contamination check is **one and the same** for the 4 targets (not a privilege
        of front alone)."""
        monkeypatch.setattr(patch_sbv2, "_APPLIED", True)

        with pytest.raises(RuntimeError, match="恒真化"):
            export_sbv2._assert_patches_not_applied(f"taking the {where} reference")

    def test_the_dec_gate_points_the_other_way(self):
        """Only dec's check requires that "the derived parameters **remain**".

        The contamination source is `remove_weight_norm` (a destructive fold), so reusing
        `patches_applied()` as is would let a reference be taken on an already-removed net_g and
        compare two post-remove states (a difference of 0 is trivial).
        """
        removed = torch.nn.Module()
        removed.register_parameter("weight", torch.nn.Parameter(torch.zeros(1)))
        with pytest.raises(RuntimeError, match="恒真化"):
            export_sbv2._assert_weight_norm_present(removed, "taking the dec reference")

        intact = torch.nn.Module()
        intact.register_parameter("weight_v", torch.nn.Parameter(torch.zeros(1)))
        export_sbv2._assert_weight_norm_present(intact, "taking the dec reference")


@requires_weights
class TestVerifyGatePlacement:
    """Measures that the checks **sit on the reference-taking path** (functions do not bypass them).

    A test of the helper alone only says "the check fires". If the verify functions did not call
    it, an order violation would be a false PASS, so `verify_*` is actually called in a
    contaminated state to see that it fails.
    """

    @pytest.mark.parametrize(
        "verifier", [export_sbv2.verify_flow, export_sbv2.verify_voice], ids=["flow", "voice"]
    )
    def test_a_patched_process_cannot_take_references(self, monkeypatch, verifier):
        monkeypatch.setattr(patch_sbv2, "_APPLIED", True)

        # Match the specific check (matching the substring both checks share, which does not tell
        # which one failed, would be green even if the dec-side check failed).
        with pytest.raises(RuntimeError, match="パッチ適用後"):
            verifier(MODEL_DIR, cases=[(2, 0)])

    @pytest.mark.parametrize(
        "verifier", [export_sbv2.verify_dec, export_sbv2.verify_voice], ids=["dec", "voice"]
    )
    def test_an_already_removed_dec_cannot_be_the_reference(self, monkeypatch, verifier):
        """The form of a second verify run in one process (remove is idempotent, so it would pass
        silently)."""
        net_g, hps = export_sbv2.load_net_g(MODEL_DIR)
        export_sbv2.ensure_dec_plain(net_g)
        monkeypatch.setattr(export_sbv2, "load_net_g", lambda _model_dir: (net_g, hps))
        # Narrow the contamination source to remove only (voice also has the patch check, and
        # failing on that first would not show that "dec's check sits on the path").
        monkeypatch.setattr(patch_sbv2, "_APPLIED", False)

        with pytest.raises(RuntimeError, match="remove_weight_norm 済み"):
            verifier(MODEL_DIR, cases=[(2, 0)])


class TestSymMaxPerTarget:
    """Per-target defaults of sym_max (ADR 0013 — a mix-up is silent)."""

    def test_every_target_declares_its_upper_bound(self):
        assert set(export_sbv2.TARGET_SYM_MAX) == set(export_sbv2.TARGETS)

    def test_the_front_family_and_the_flow_family_differ_by_an_order(self):
        """front=512 (phonemes) and flow=4096 (frames) — they differ by an order of magnitude, so a
        mix-up is fatal."""
        assert export_sbv2.TARGET_SYM_MAX[export_sbv2.TARGET_FRONT] == export_sbv2.SYM_MAX
        assert export_sbv2.TARGET_SYM_MAX[export_sbv2.TARGET_VOICE] == export_sbv2.FLOW_SYM_MAX
        assert export_sbv2.SYM_MAX < export_sbv2.FLOW_SYM_MAX

    def test_a_shared_sym_max_across_targets_is_refused(self, monkeypatch):
        """Handing one `--sym-max` to several targets is rejected.

        The defaults differ by an order of magnitude, so handing one out always makes one side
        wrong (and silently).
        """
        monkeypatch.setattr(sys, "argv", ["export_sbv2.py", "--sym-max", "1024"])

        with pytest.raises(SystemExit) as raised:
            export_sbv2.main()

        assert raised.value.code == 2


class TestStagedTarget:
    """The staging-directory rule (X2-104) and the provenance record of symbolic dimensions (CG4-3)
    — neither needs real weights.

    What is protected is one thing: "nothing goes to final before the checks". If a failed real run
    left assets in final, the io goldens, taken from the same broken module, would agree with them,
    and the TS-side matching **would be green**.
    """

    def test_it_publishes_the_seat_and_leaves_the_provenance_record(self, tmp_path: Path):
        out_dir = tmp_path / "series" / export_sbv2.TARGET_VOICE

        with export_sbv2._staged_target(
            export_sbv2.TARGET_VOICE, out_dir, export_sbv2.FLOW_SYM_MAX
        ) as staged:
            assert staged != out_dir
            (staged / export_sbv2.MODEL_FILE).write_bytes(b"container")

        assert (out_dir / export_sbv2.MODEL_FILE).read_bytes() == b"container"
        record = json.loads(
            (out_dir / export_sbv2.EXPORT_PROVENANCE_FILE).read_text(encoding="utf-8")
        )
        assert record == {
            "target": export_sbv2.TARGET_VOICE,
            "sym_max": export_sbv2.FLOW_SYM_MAX,
        }
        # Neither the staging nor the backup directory remains (cleaning up after the swap is the
        # job of the core primitive).
        assert sorted(path.name for path in out_dir.parent.iterdir()) == [out_dir.name]

    def test_a_failing_gate_leaves_the_previous_series_untouched(self, tmp_path: Path):
        """A real run whose checks fail does not move a single byte of final (the previous assets
        stay as they are)."""
        out_dir = tmp_path / "series" / export_sbv2.TARGET_FRONT
        out_dir.mkdir(parents=True)
        (out_dir / export_sbv2.MODEL_FILE).write_bytes(b"previous")

        with (
            pytest.raises(AssertionError, match="sanity"),
            export_sbv2._staged_target(
                export_sbv2.TARGET_FRONT, out_dir, export_sbv2.SYM_MAX
            ) as staged,
        ):
            (staged / export_sbv2.MODEL_FILE).write_bytes(b"broken")
            raise AssertionError("sanity failed")

        assert (out_dir / export_sbv2.MODEL_FILE).read_bytes() == b"previous"
        assert not (out_dir / export_sbv2.EXPORT_PROVENANCE_FILE).exists()
        assert sorted(path.name for path in out_dir.parent.iterdir()) == [out_dir.name]

    def test_the_record_carries_a_non_default_sym_max(self, tmp_path: Path):
        """A deviating `--sym-max` is kept in the record (the distribution side's matching looks at
        this value — CG4-3)."""
        out_dir = tmp_path / "series" / export_sbv2.TARGET_VOICE

        with export_sbv2._staged_target(export_sbv2.TARGET_VOICE, out_dir, 1024):
            pass

        record = json.loads(
            (out_dir / export_sbv2.EXPORT_PROVENANCE_FILE).read_text(encoding="utf-8")
        )
        assert record["sym_max"] == 1024


class _TinyDp(nn.Module):
    """The minimal skeleton of `DurationPredictor` (`forward(h, x_mask, g=…)` and 192/512 channels).

    `DurationPredictorGraph` can wrap it as is, so "does the dtype knob reach the end of export"
    can be run without real weights.
    """

    def __init__(self) -> None:
        super().__init__()
        self.cond = nn.Conv1d(512, 192, 1)
        self.conv = nn.Conv1d(192, 1, 3, padding=1)

    def forward(
        self, x: torch.Tensor, x_mask: torch.Tensor, g: torch.Tensor | None = None
    ) -> torch.Tensor:
        assert g is not None
        return self.conv((x + self.cond(g)) * x_mask) * x_mask


class _TinyHybrid(nn.Module):
    """The minimal skeleton of the i4 mix (eligible linear / conv1d = i4 group32, the rest = i8).

    `fc`'s in axis is 64 because of the MUST that i4 makes no partial group (ADR 0069 decision 2).
    `odd` (in axis 48) is **a linear that does not divide** — the detector of the branch that falls
    out of eligibility to the i8 side; in the real net_g all 6 divide (FN4 measurement), so only
    here can it be exercised. `conv`'s row length (`Cin·K` = 12) does not divide either, so it
    stays on the i8 side, and "is the split exclusive" becomes observable (with an empty i8 side,
    `fake_quant_int8` would have 0 targets and the branch would not be visible).
    """

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv1d(4, 4, 3)
        self.fc = nn.Linear(64, 4)
        self.odd = nn.Linear(48, 4)


class _TinyI4Census(nn.Module):
    """A skeleton for enumerating the i4 eligible set **exactly** (each branch fails for one reason
    only).

    Against the 3 eligible ones (`fc` / `wide` / `deep`), the dimensions of the 4 ineligible ones
    are chosen so that **exactly one reason** holds for each (all other conditions are met) — if
    the reasons overlapped, removing one of the checks would leave the set unchanged and the test
    would stop protecting that check.
    """

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(64, 4)  # eligible (the existing linear branch, row length 64)
        self.wide = nn.Conv1d(32, 4, kernel_size=1)  # eligible (row length Cin·K = 32)
        self.deep = nn.Conv1d(16, 4, kernel_size=2)  # eligible (row length = 16·2 = 32)
        self.narrow = nn.Linear(48, 4)  # row length 48 is not divisible by group32
        self.short = nn.Conv1d(4, 4, kernel_size=3)  # row length 12 is not divisible
        self.depthwise = nn.Conv1d(64, 32, kernel_size=1, groups=2)  # row length 32, groups > 1
        self.up = nn.ConvTranspose1d(4, 32, kernel_size=1)  # row length 32, another type
        self.table = nn.Embedding(4, 32)  # row length 32, out of scope this time (stays i8)


class _TinyAllI4(nn.Module):
    """A skeleton whose quantization targets are **all** i4-eligible (in the real model `flow` has
    this form — all conv1d)."""

    def __init__(self) -> None:
        super().__init__()
        self.pre = nn.Conv1d(32, 4, kernel_size=1)
        self.post = nn.Conv1d(16, 4, kernel_size=2)


class _TinyNetG(nn.Module):
    """Only the surface of net_g that `export_dp` touches (`dp` and the speaker embedding)."""

    def __init__(self) -> None:
        super().__init__()
        self.dp = _TinyDp()
        self.emb_g = nn.Embedding(1, 512)


#: Golden cases for the dtype wiring tests (the 5 real-weight cases are not needed — 2 short ones
#: suffice).
_TINY_CASES = (("p2", 2, 0), ("padded", 16, 5))


def _weights_are_f16_exact(module: nn.Module) -> bool:
    """Whether every f32 parameter lies on the f16 grid (the same predicate as emit's eligibility
    check)."""
    return all(
        torch.equal(tensor, tensor.to(torch.float16).to(torch.float32))
        for tensor in module.parameters()
        if tensor.dtype is torch.float32
    )


def _weights_are_int8_exact(module: nn.Module) -> bool:
    """Whether every weight slot comes back bit-identically as `q8 · scale` (the predicate of emit's
    inverse-transform check).

    `scale` is **recomputed from the current weights** — a predicate that holds because
    recomputing the scale from quantized weights is a fixed point in f32 (ADR 0019's quantization
    closed under ±127), and weights that were not quantized fail it (a detector of a forgotten
    quantization). False when there is no target at all (prevents a vacuous check).
    """
    checked = 0
    for sub in module.modules():
        axes = {axis for cls, axis in QUANT_CHANNEL_AXES.items() if isinstance(sub, cls)}
        weight = getattr(sub, "weight", None) if axes else None
        if weight is None:
            continue
        scale = channel_scale(weight, next(iter(axes)))
        if not torch.equal(quantize_to_int8(weight, scale).to(torch.float32) * scale, weight):
            return False
        checked += 1
    return checked > 0


class TestWeightDtypeSeries:
    """Storage dtype series (ADR 0018 / 0019) — pins only the rules that turn into a **false PASS**
    when broken.

    No real weights are used (the wiring rules do not depend on the model). The numbers themselves
    are checked by the Deno-side E2E (the f16 / i8 series of
    `packages/runtime/tests/e2e_sbv2_test.ts`) and emit's round-trip bit-identity check.
    """

    def test_the_default_output_root_is_a_separate_series_per_dtype(self):
        """MUST: compressed series are separate directories (ADR 0018 / 0019) — sharing one would
        put them under the f32 checks."""
        roots = {
            dtype: export_sbv2.default_out_root(export_sbv2.DEFAULT_MODEL_DIR, dtype)
            for dtype in export_sbv2.WEIGHT_DTYPES
        }

        assert set(roots) == {"f32", "f16", "i8", "i4"}
        assert len(set(roots.values())) == len(roots)
        # Series live under `outputs/series/` (`models/` is only for distributed forms — ADR 0037).
        assert roots == {
            "f32": SERIES_ROOT / "sbv2-F1",
            "f16": SERIES_ROOT / "sbv2-F1-f16",
            "i8": SERIES_ROOT / "sbv2-F1-i8",
            "i4": SERIES_ROOT / "sbv2-F1-i4",
        }

    def test_the_i4_series_is_stored_as_i8_by_default(self):
        """i4 is **a mix** (only eligible linear / conv1d are i4, the rest i8) — it cannot be a
        single dtype.

        Since the i4 execution path is limited to `emit.I4_WEIGHT_OPS` (ADR 0069 decision 5 and
        its addendum), an i4 default would leave the ineligible ones (embedding /
        `ConvTranspose1d` / depthwise conv) silently in f32. The only form is an i8 default with
        per-weight overrides for the eligible ones.
        """
        assert set(export_sbv2.BASE_WEIGHT_DTYPES) == set(export_sbv2.WEIGHT_DTYPES)
        assert export_sbv2.BASE_WEIGHT_DTYPES["i4"] == "i8"

    def test_the_series_name_carries_the_weights_directory_name(self):
        """A separate series per speaker (a shared spelling would silently overwrite another
        speaker's assets)."""
        other = export_sbv2.default_out_root(Path("inputs/sbv2/OTHER"), "f32")

        assert other == SERIES_ROOT / "sbv2-OTHER"
        assert other != export_sbv2.default_out_root(export_sbv2.DEFAULT_MODEL_DIR, "f32")

    def test_every_target_takes_the_dtype_knob(self):
        """All 5 targets take the series knob (if even one lacked it, f32 assets would mix into an
        f16 series)."""
        for target, exporter in export_sbv2.EXPORTERS.items():
            parameter = inspect.signature(exporter).parameters.get("dtype")
            assert parameter is not None, target
            assert parameter.default == "f32", target

    def test_the_cli_refuses_a_compressed_dtype_together_with_verify(self, monkeypatch):
        """MUST: `--dtype` is emit-only (treated like `--sym-max`).

        Verification is an eager comparison that ignores the storage format, and for dec / voice
        the rounding can only be applied **after** `remove_weight_norm` while the reference is
        taken **before** remove. Allowing the combination would compare "rounded vs unrounded",
        `bit_exact` would drop to False regardless of the assets, and the check would lose its
        meaning.
        """
        monkeypatch.setattr(sys, "argv", ["export_sbv2.py", "--verify", "dec", "--dtype", "f16"])

        with pytest.raises(SystemExit) as raised:
            export_sbv2.main()

        assert raised.value.code == 2

    def test_the_cli_refuses_i8_together_with_verify(self, monkeypatch):
        """i8 is treated the same (no exception table growing with every compressed dtype — all are
        emit-only)."""
        monkeypatch.setattr(sys, "argv", ["export_sbv2.py", "--verify", "dec", "--dtype", "i8"])

        with pytest.raises(SystemExit) as raised:
            export_sbv2.main()

        assert raised.value.code == 2

    def test_f32_leaves_the_weights_untouched(self):
        torch.manual_seed(0)
        module = _TinyDp()
        before = module.conv.weight.clone()

        export_sbv2._fake_quant("f32", module, export_sbv2.TARGET_DP)

        assert torch.equal(module.conv.weight, before)

    def test_f16_rounds_the_module_that_gets_exported(self):
        torch.manual_seed(0)
        module = _TinyDp()
        assert not _weights_are_f16_exact(module), (
            "no detection power if it is on the grid before rounding"
        )

        export_sbv2._fake_quant("f16", module, export_sbv2.TARGET_DP)

        assert _weights_are_f16_exact(module)

    def test_i8_quantizes_the_module_that_gets_exported(self):
        """i8 is per-channel symmetric (ADR 0019). The ledger keys are FQNs (the space emit matches
        in)."""
        torch.manual_seed(0)
        module = _TinyDp()
        assert not _weights_are_int8_exact(module), (
            "no detection power if it is on the grid before quantization"
        )

        scales, overrides = export_sbv2._fake_quant("i8", module, export_sbv2.TARGET_DP)

        assert overrides == {}, "a single-dtype i8 series needs no per-weight storage overrides"
        assert _weights_are_int8_exact(module)
        assert set(scales) == {"cond.weight", "conv.weight"}

    def test_i4_splits_the_eligible_linears_and_the_rest_exclusively(self):
        """MUST: the i8 / i4 targets are exclusive (the mix MUST of `quantize.py` — double rounding
        is a silent wrong value).

        The hole of a weight left in neither is checked at the same time (the merged ledger covers
        every weight). Where a weight went can be read from the scale's **shape**: i4 has the group
        shape `[channels, groups]`, i8 the keepdim shape of the weight's rank. `odd` (in axis 48)
        falls out of i4 eligibility to the i8 side — if this were dropped silently, "the i4 series
        lost targets" would not be observable.
        """
        torch.manual_seed(0)
        module = _TinyHybrid()

        scales, overrides = export_sbv2._fake_quant("i4", module, export_sbv2.TARGET_FRONT)

        assert overrides == {"fc.weight": "i4"}
        assert set(scales) == {"fc.weight", "odd.weight", "conv.weight"}
        # in axis 64 / group 32 gives 2 groups — distinguishable by shape from the keepdim form ([4,
        # 1]).
        assert list(scales["fc.weight"].shape) == [4, 2]
        assert list(scales["odd.weight"].shape) == [4, 1]
        assert list(scales["conv.weight"].shape) == [4, 1, 1]

    def test_the_i4_target_set_is_linears_plus_dense_conv1ds(self):
        """Eligible = type × `groups == 1` × divisible row length (the conv1d addendum of iteration
        J-5b).

        The **whole** set is matched (not a subset) — picking too many makes emit fail loudly,
        but missing some passes silently as "the i4 series lost targets".
        """
        assert export_sbv2._i4_module_names(_TinyI4Census()) == {"fc", "wide", "deep"}

    def test_i4_rounds_a_conv1d_weight_with_a_rank2_group_scale(self):
        """conv1d's scale is rank 2 (`[Cout, (Cin·K)/g]`) — a different shape from i8's keepdim form
        `[4,1,1]`.

        Reading the destination from the shape follows the same convention as linear: a rank-3
        weight with a rank-3 scale has fallen to the i8 side.
        """
        torch.manual_seed(0)
        module = _TinyI4Census()

        scales, overrides = export_sbv2._fake_quant("i4", module, export_sbv2.TARGET_VOICE)

        assert set(overrides) == {"fc.weight", "wide.weight", "deep.weight"}
        assert list(scales["deep.weight"].shape) == [4, 1]
        assert list(scales["short.weight"].shape) == [4, 1, 1], (
            "an ineligible conv has i8's keepdim shape"
        )
        assert list(scales["up.weight"].shape) == [1, 32, 1], (
            "the i8 axis of the transposed layout is 1"
        )
        assert list(scales["table.weight"].shape) == [4, 1]  # embedding is i8 (out of scope)

    def test_i4_skips_the_i8_pass_when_every_target_is_eligible(self):
        """A series whose targets are all i4-eligible does not fail (it never hits
        `fake_quant_int8`'s "0 targets").

        The design checks before the call that 0 is **the result of the enumeration**; writing the
        i8 side only as the negation of "not in i4" would kill the export for a series made only of
        conv1d (the real `flow`).
        """
        torch.manual_seed(0)
        module = _TinyAllI4()

        scales, overrides = export_sbv2._fake_quant("i4", module, export_sbv2.TARGET_FLOW)

        assert set(scales) == {"pre.weight", "post.weight"}
        assert set(overrides) == set(scales), "every weight is in the explicit i4 overrides"

    def test_rounding_before_removing_the_weight_norm_leaves_the_grid(self):
        """Backing for the MUST: rounding comes **after** `remove_weight_norm` (as
        `ensure_dec_plain` announces).

        While weight_norm is active, `weight` is a derived value, and the effective weight to
        round is `g · v / ‖v‖`. Rounding `weight_g` / `weight_v` first does not put the effective
        weight on the f16 grid — only the delivery size shrinks and the numbers become different.
        The detector is emit's round-trip bit-identity check
        (`test_emit.py::test_an_unrounded_eligible_weight_fails_loudly`), and this predicate is
        exactly what that check looks at.
        """
        torch.manual_seed(0)
        conv = torch.nn.utils.weight_norm(nn.Conv1d(4, 4, 3))
        rounded = export_sbv2.round_weights_to_f16(conv)
        assert rounded.parameters >= 2, "confirms that weight_g / weight_v were rounded"

        torch.nn.utils.remove_weight_norm(conv)

        assert not _weights_are_f16_exact(conv)

    @pytest.mark.parametrize("dtype", ["f32", "f16", "i8", "i4"])
    def test_the_dtype_reaches_the_container_and_the_golden(self, monkeypatch, tmp_path, dtype):
        """The dtype reaches the end of export (the storage declaration), and the golden is taken
        with the **rounded** weights.

        Taking the golden before rounding would mean "golden on the original values, execution on
        the rounded values", and the E2E difference would turn into a mix of quantization error and
        implementation error (it only pushes toward a looser tolerance, so detection power drops
        while staying green). Here the module after export (= already rounded) is run again as is
        and must be bit-identical to the golden, which pins that order.

        `i4` is **the only path by which a non-empty `weight_dtype_overrides` reaches emit**, so
        without it here the wiring that builds the distribution's `front_i4` / `voice_i4` would
        never run in this module. `_TinyDp`'s two conv1d have row lengths 512 / 576, both divisible
        by group32, so all are i4-eligible and the branch where the i8 side has "0 targets" is
        exercised too.
        """
        torch.manual_seed(0)
        net_g = _TinyNetG()
        hps = SimpleNamespace(version="tiny")
        monkeypatch.setattr(export_sbv2, "load_net_g", lambda _model_dir: (net_g, hps))
        out = tmp_path / export_sbv2.TARGET_DP

        # The name of `--model-dir` decides the voice family (the source of truth of provenance and
        # license — `export_sbv2.SBV2_FAMILY_DIRS`), so the synthetic setup uses the conventional
        # spelling too.
        summary = export_sbv2.export_dp(_speaker_dir(tmp_path), out, cases=_TINY_CASES, dtype=dtype)

        stored = _storage_layouts(out / export_sbv2.MODEL_FILE)
        compressed = {name for name, storage in stored.items() if storage in ("f16", "i8", "i4")}
        if dtype == "f32":
            assert compressed == set(), stored
            assert summary["compressed_tensors"] == 0
        else:
            # Eligible = only the 2 conv weights (bias is always f32 — ADR 0006).
            assert {stored[name] for name in compressed} == {dtype}, stored
            assert len(compressed) == 2, stored
            assert summary["compressed_tensors"] == 2
        if dtype in ("i8", "i4"):
            # The companion scale declaration (ADR 0019 / 0069) — without it the values cannot be
            # restored.
            scales = _scale_blocks(out / export_sbv2.MODEL_FILE)
            scale_keys = {scales[name] for name in compressed}
            assert None not in scale_keys, stored
            assert len(scale_keys) == len(compressed), "the scale keys are not separate per weight"
            assert summary["scale_bytes"] > 0
        else:
            assert summary["scale_bytes"] == 0

        assert summary["dtype"] == dtype

        module = export_sbv2.DurationPredictorGraph(net_g.dp)
        assert _weights_are_f16_exact(module) == (dtype == "f16")
        assert _weights_are_int8_exact(module) == (dtype == "i8")
        for name, _length, _pad in _TINY_CASES:
            tensors = _io_tensors(out, name)
            with torch.no_grad():
                fresh = module(tensors["input.h"], tensors["input.x_mask"], tensors["input.g"])
            assert torch.equal(fresh, tensors["output.0"]), name

    def test_dropping_the_overrides_fails_loudly_instead_of_falling_back_to_i8(
        self, monkeypatch, tmp_path
    ):
        """Blocks a vacuous check: direct evidence that the storage of `--dtype i4` is carried by
        **the explicit overrides**.

        The base of an i4 series is i8 ({@link export_sbv2.BASE_WEIGHT_DTYPES}), so dropping the
        overrides gives the form "declares i8 storage while holding an i4 group scale ledger". emit
        rejects that as a mismatch with the keepdim shape of a per-channel scale — **there is no
        path that silently falls to i8** (backing that the case above looks at the overrides, not
        the dtype name).
        """
        torch.manual_seed(0)
        net_g = _TinyNetG()
        hps = SimpleNamespace(version="tiny")
        monkeypatch.setattr(export_sbv2, "load_net_g", lambda _model_dir: (net_g, hps))
        real = export_sbv2._fake_quant
        monkeypatch.setattr(
            export_sbv2,
            "_fake_quant",
            lambda dtype, module, target: (real(dtype, module, target)[0], {}),
        )
        out = tmp_path / export_sbv2.TARGET_DP

        with pytest.raises(EmitError, match="keepdim shape"):
            export_sbv2.export_dp(_speaker_dir(tmp_path), out, cases=_TINY_CASES, dtype="i4")

        # final is not created before the checks (the MUST of `_staged_target`).
        assert not out.exists()


@requires_weights
class TestInt8IdempotenceOnRealWeights:
    """i8 fake-quant is idempotent on **real weights** too (ADR 0019's "quantization closed under
    ±127 is a fixed point in f32").

    `_fake_quant` is designed to depend only on the order "after the effective weights are final,
    before the golden is taken", but dec / voice are taken from the same net_g lineage (the voice
    wrapper rounds flow and dec at once), a form where re-application across targets can happen.
    The idempotence test on a synthetic model (`test_quantize.py`) has another weight distribution,
    so **the fixed-point property on the real weights' amax distribution** is checked separately —
    if it broke, "the golden taken with the first rounding" and "the stored values written with the
    second rounding" would disagree, and emit's inverse-transform bit-identity check would fail
    (= not silent, but red with an unclear cause).
    """

    @pytest.mark.parametrize("target", ["dec", "voice"])
    def test_reapplying_the_quantization_is_bit_identical(self, target):
        net_g = export_sbv2.load_net_g(MODEL_DIR)[0]
        export_sbv2.ensure_dec_plain(net_g)
        # Only the weights are looked at, so no patches are applied (applying them would contaminate
        # the whole process and drag in other tests that take references in the same process —
        # ADR 0013's exclusivity rule).
        module = net_g.dec if target == "dec" else patch_sbv2.Sbv2Voice(net_g)

        first = fake_quant_int8(module)
        snapshot = {name: module.get_parameter(name).detach().clone() for name in first.scales}
        second = fake_quant_int8(module)

        assert first.modules == len(snapshot) > 0
        for name, before in snapshot.items():
            assert torch.equal(module.get_parameter(name), before), name
            assert torch.equal(second.scales[name], first.scales[name]), f"{name}: scale"


#: One case line of the verify report (`P=   2 pad=0: ... bit_exact=True`).
_VERIFY_CASE_LINE = re.compile(r"^[PT]=\s*\d+ pad=\d+:")


@pytest.fixture(scope="module")
def verify_report():
    """Runs `--verify <target>` once in **a separate process** and returns the summary (per target).

    MUST: run it in a separate process — the patches and `remove_weight_norm` contaminate the whole
    process, so mixing with other tests that take references in the same process (the export
    fixtures) would make the equivalence check vacuous and a false PASS. This is the exclusivity
    rule of ADR 0013 in action.
    """
    cache: dict[str, dict] = {}

    def run(target: str) -> dict:
        if target not in cache:
            completed = subprocess.run(
                [sys.executable, "-m", "sbv2.export", "--verify", target],
                cwd=str(REPO_ROOT / "tools" / "export-recipes"),
                capture_output=True,
                text=True,
                check=True,
            )
            lines = completed.stdout.splitlines()
            worst = [line for line in lines if line.startswith("worst maxdiff = ")]
            exact = [line for line in lines if line.startswith("bit_exact all = ")]
            assert len(worst) == 1 and len(exact) == 1, completed.stdout
            cache[target] = {
                "cases": [line for line in lines if _VERIFY_CASE_LINE.match(line)],
                "worst": float(worst[0].split("=", 1)[1]),
                "bit_exact": exact[0].split("=", 1)[1].strip() == "True",
                "stdout": completed.stdout,
            }
        return cache[target]

    return run


@requires_weights
class TestEquivalenceVerification:
    """Eager equivalence with the reference implementation (measured in **a separate process** — the
    exclusivity rule in action)."""

    @pytest.mark.parametrize(
        ("target", "expected_cases"),
        [
            (export_sbv2.TARGET_FRONT, export_sbv2.VERIFY_CASES),
            (export_sbv2.TARGET_FLOW, export_sbv2.FLOW_VERIFY_CASES),
            (export_sbv2.TARGET_DEC, export_sbv2.FLOW_VERIFY_CASES),
            (export_sbv2.TARGET_VOICE, export_sbv2.FLOW_VERIFY_CASES),
        ],
    )
    def test_every_verify_case_is_reported(self, verify_report, target, expected_cases):
        """The verify cases have not silently shrunk (shrinking the table shrinks the scope of the
        equivalence claim)."""
        report = verify_report(target)

        assert len(report["cases"]) == len(expected_cases), report["stdout"]

    def test_the_front_patch_is_equivalent_to_the_reference(self, verify_report):
        # Measured 2.02e-5 (logw_sdp at P=512). The threshold is 5 times that; if a patch did more
        # than "build the same formula in another order" (= the implementation changed), the
        # difference would be O(1) and fail here.
        report = verify_report(export_sbv2.TARGET_FRONT)

        assert report["worst"] < 1e-4, report["stdout"]

    def test_the_flow_patch_is_equivalent_to_the_reference(self, verify_report):
        """FlowReverse (table inputs, split→slice, exp(−logs) folding) is equivalent to the original
        reverse path.

        Measured worst 1.43e-6 (T=512). An order of magnitude below front because it does not go
        through the spline. The threshold is about 7 times that.
        """
        report = verify_report(export_sbv2.TARGET_FLOW)

        assert report["worst"] < 1e-5, report["stdout"]

    def test_removing_the_weight_norm_is_bit_exact_in_every_case(self, verify_report):
        """**Closes the open item of recon §6**: bit identity of remove_weight_norm.

        At the time of recon there was a measurement of one case only (z=(1,192,50)), and it was
        stated that the exact reproduction of the effective weight `g·v/‖v‖` in f32 is not a spec
        guarantee. Here all 10 cases, wider than the goldens, are matched to establish **maxdiff 0
        and bit identity**. Bit identity is required so that a mix-up of `0.0` and `-0.0` (a
        difference of 0, but different bits) does not pass.
        """
        report = verify_report(export_sbv2.TARGET_DEC)

        assert report["bit_exact"], report["stdout"]
        assert report["worst"] == 0.0, report["stdout"]

    def test_the_voice_fusion_is_equivalent_to_the_reference_chain(self, verify_report):
        """The fused voice is equivalent to the composition "unpatched flow + dec with weight_norm
        active".

        Measured worst 1.25e-6 (T=203). **Fusion does not exceed flow's own difference
        (1.43e-6)** — dec buries tiny input differences in the conv reductions and does not amplify
        the error.
        """
        report = verify_report(export_sbv2.TARGET_VOICE)

        assert report["worst"] < 1e-5, report["stdout"]


@pytest.fixture(scope="module")
def exported_front(tmp_path_factory):
    """Exports front into a temporary directory (does not touch the repository's outputs).

    MUST: this fixture applies the patches process-wide. Checks that need the reference from before
    patching run in a separate process (`TestFrontVerify`).
    """
    out = tmp_path_factory.mktemp("sbv2-front") / export_sbv2.TARGET_FRONT
    return out, export_sbv2.export_front(MODEL_DIR, out)


@requires_weights
class TestFrontExport:
    def test_the_container_passes_the_full_verification(self, exported_front):
        out, _ = exported_front

        _verified_graph(out / export_sbv2.MODEL_FILE)

    def test_the_graph_declares_the_wrapper_argument_names(self, exported_front):
        out, summary = exported_front
        graph = _verified_graph(out / export_sbv2.MODEL_FILE)

        assert [spec["name"] for spec in graph["inputs"]] == list(export_sbv2.FRONT_INPUT_ORDER)
        assert [spec["shape"] for spec in graph["inputs"]] == [
            [1, "P"],
            [1, 1, "P"],
            [1, "P"],
            [1, "P"],
            [1, 1024, "P"],
            [1, 256],
            [1, 512, 1],
            [1, 2, "P"],
        ]
        assert summary["symbols"] == ["P"]
        assert summary["outputs"] == len(export_sbv2.FRONT_OUTPUT_NAMES)

    def test_the_graph_stays_inside_the_current_contract(self, exported_front):
        _, summary = exported_front

        # The premise of iteration 6: front passes **without widening the vocabulary** (the ops
        # added in iterations 3–5 suffice).
        assert set(summary["ops"]) <= set(EMITTABLE_OPS)

    def test_the_relative_position_tables_are_baked_at_pmax(self, exported_front):
        """The relative-position attention tables are baked at Pmax + `sym_prefix_slice` (ADR 0013).

        Without baked tables, either quadratic shape expressions (2P−1 and the like) remain or
        P-dependent index computation survives at run time. **There are 3 tables** (idx_k / valid
        on the key side, idx_v on the value side), and every input must be a static shape without
        symbols.
        """
        out, _ = exported_front
        graph = _verified_graph(out / export_sbv2.MODEL_FILE)
        sources = []
        for node in graph["nodes"]:
            if node["op"] != "sym_prefix_slice":
                continue
            assert node["attrs"]["sym"] == "P"
            shape = graph["values"][node["ins"][0]]["shape"]
            assert all(isinstance(dim, int) for dim in shape), shape
            sources.append((tuple(shape), graph["values"][node["ins"][0]]["dtype"]))

        pmax = export_sbv2.SYM_MAX
        window = 2 * export_sbv2.EXPECTED_WINDOW_SIZE + 1
        assert sorted(sources) == sorted(
            [
                ((1, 1, pmax, pmax), "i32"),  # idx_k = clamp(w + j − i, 0, 2w)
                ((1, 1, pmax, pmax), "f32"),  # valid = [|j − i| ≤ w]
                ((1, 1, pmax, window), "i32"),  # idx_v = i + c (value side)
            ]
        )

    def test_the_baked_tables_stay_within_the_expected_budget(self, exported_front):
        """The baked constants are about 2MB (pins in numbers the basis of ADR 0013's "no real harm
        for front").

        The same approach for flow (Tmax=4096) would be O(Tmax²), 134MB — if the premise of the
        ruling, that baking is fine precisely because front's Pmax=512, broke, it would show here.
        """
        out, _ = exported_front
        graph = _verified_graph(out / export_sbv2.MODEL_FILE)
        baked = sum(
            math.prod(graph["values"][name]["shape"]) * 4
            for name in graph["initializers"]
            if name.startswith("const.")
        )

        assert baked < 3 * 1024 * 1024, baked

    def test_the_io_files_follow_the_naming_convention(self, exported_front):
        out, summary = exported_front

        expected = [
            f"{export_sbv2.IO_PREFIX}{name}{export_sbv2.IO_SUFFIX}"
            for name, _, _ in export_sbv2.GOLDEN_CASES
        ]
        assert summary["io"] == expected
        for name in expected:
            with safe_open(str(out / name), framework="pt") as handle:
                keys = set(handle.keys())
            assert keys == {
                *(f"input.{declared}" for declared in export_sbv2.FRONT_INPUT_ORDER),
                *(f"output.{index}" for index in range(len(export_sbv2.FRONT_OUTPUT_NAMES))),
            }

    @pytest.mark.parametrize("case", export_sbv2.GOLDEN_CASES, ids=lambda case: case[0])
    def test_io_shapes_bind_the_symbol_to_the_case_length(self, exported_front, case):
        out, _ = exported_front
        name, length, _ = case
        graph = _verified_graph(out / export_sbv2.MODEL_FILE)
        tensors = _io_tensors(out, name)

        for spec in graph["inputs"]:
            actual = list(tensors[f"input.{spec['name']}"].shape)
            declared = [
                length if isinstance(dim, str) and parse_dim(dim).sym == "P" else dim
                for dim in spec["shape"]
            ]
            assert actual == declared, spec["name"]
        # logw_sdp / logw_dp are [1,1,P], m_p / logs_p are [1,192,P].
        assert [list(tensors[f"output.{index}"].shape) for index in range(4)] == [
            [1, 1, length],
            [1, 1, length],
            [1, 192, length],
            [1, 192, length],
        ]

    @pytest.mark.parametrize("case", export_sbv2.GOLDEN_CASES, ids=lambda case: case[0])
    def test_the_golden_outputs_reproduce_a_fresh_eager_forward(
        self, exported_front, front_module, case
    ):
        """The stored expected values are bit-identical to the (patched) torch output for the same
        inputs."""
        out, _ = exported_front
        name, _, _ = case
        tensors = _io_tensors(out, name)
        with torch.no_grad():
            fresh = front_module(
                *(tensors[f"input.{declared}"] for declared in export_sbv2.FRONT_INPUT_ORDER)
            )

        for index, value in enumerate(fresh):
            assert torch.equal(value, tensors[f"output.{index}"]), index

    @pytest.mark.parametrize("case", export_sbv2.GOLDEN_CASES, ids=lambda case: case[0])
    def test_a_corrupted_golden_is_detected(self, exported_front, front_module, case):
        """Fault injection — corrupting one element of the expected values turns the test above red
        (the guard against a vacuous check)."""
        out, _ = exported_front
        name, _, _ = case
        tensors = _io_tensors(out, name)
        with torch.no_grad():
            fresh = front_module(
                *(tensors[f"input.{declared}"] for declared in export_sbv2.FRONT_INPUT_ORDER)
            )
        corrupted = tensors["output.0"].clone()
        corrupted[0, 0, 0] += 1e-3

        assert not torch.equal(fresh[0], corrupted)

    def test_the_padded_case_zeroes_the_masked_tail_of_every_output(self, exported_front):
        """The mask multiplication takes effect on all 4 outputs (the padding columns are exactly
        0)."""
        out, _ = exported_front
        name, length, pad = next(case for case in export_sbv2.GOLDEN_CASES if case[2] > 0)
        tensors = _io_tensors(out, name)

        for index in range(len(export_sbv2.FRONT_OUTPUT_NAMES)):
            tail = tensors[f"output.{index}"][..., length - pad :]
            assert torch.equal(tail, torch.zeros_like(tail)), index

    def test_the_masked_tail_assertion_is_not_vacuous(self, exported_front, front_module):
        """Fault injection — replacing the mask with all ones makes the tail non-zero.

        x / tone / bert have values in the padding columns too, so without the mask values leak.
        """
        out, _ = exported_front
        name, length, pad = next(case for case in export_sbv2.GOLDEN_CASES if case[2] > 0)
        tensors = _io_tensors(out, name)
        args = [tensors[f"input.{declared}"] for declared in export_sbv2.FRONT_INPUT_ORDER]
        args[export_sbv2.FRONT_INPUT_ORDER.index("x_mask")] = torch.ones(1, 1, length)
        with torch.no_grad():
            leaked = front_module(*args)

        for index, value in enumerate(leaked):
            tail = value[..., length - pad :]
            assert not torch.equal(tail, torch.zeros_like(tail)), index

    def test_regeneration_is_byte_identical(self, exported_front, tmp_path):
        """With a fixed seed the output is byte-identical in the same environment (the premise of a
        golden).

        If the random numbers, z_noise included, depended on the global seed, this would fail.
        """
        out, _ = exported_front
        again = tmp_path / export_sbv2.TARGET_FRONT
        export_sbv2.export_front(MODEL_DIR, again)

        for name in sorted(path.name for path in out.iterdir()):
            assert (again / name).read_bytes() == (out / name).read_bytes(), name


@pytest.fixture(scope="module")
def front_module(exported_front):
    """An eager module that computes the same as the golden (the patches were applied on the
    export side)."""
    net_g = export_sbv2.load_net_g(MODEL_DIR)[0]
    patch_sbv2.apply_all_patches()
    return patch_sbv2.Sbv2Front(net_g)


@requires_weights
class TestFrontInputs:
    """front's golden inputs (they use assets derived from the real weights, hence
    requires_weights)."""

    def test_the_noise_matches_what_the_reference_draws_internally(self):
        """`make_noise` gives **the same sequence** as the reference implementation's internal randn
        (the very premise of the equivalence check).

        The reference side goes through the global generator via `torch.manual_seed(...)`, the
        golden side through a `Generator`. That both return the same sequence is a torch
        implementation detail, so it is pinned explicitly here (if they split, the equivalence
        check would compare "two different noises" — the difference would be large and noticed,
        but red with an unclear cause).
        """
        for length in (2, 37):
            torch.manual_seed(export_sbv2._noise_seed(length))
            reference = torch.randn(1, 2, length) * export_sbv2.NOISE_SCALE

            assert torch.equal(export_sbv2.make_noise(length), reference), length

    def test_the_style_vector_comes_from_the_real_asset(self):
        style = export_sbv2.style_vector(MODEL_DIR)

        assert style.shape == (1, 256)
        assert style.dtype == torch.float32
        assert float(style.abs().max()) > 0

    def test_an_out_of_range_style_id_is_rejected(self):
        with pytest.raises(ValueError, match="スタイル"):
            export_sbv2.style_vector(MODEL_DIR, 10_000)

    def test_the_case_inputs_are_keyed_by_the_graph_input_names(self):
        g = torch.zeros(1, 512, 1)
        style = torch.zeros(1, 256)
        built = export_sbv2.build_front_cases(g, style)

        assert [name for name, _ in built] == [name for name, _, _ in export_sbv2.GOLDEN_CASES]
        for name, args in built:
            # x_lengths is only for the reference implementation and is not in the golden (not an IR
            # input).
            assert set(args) == set(export_sbv2.FRONT_INPUT_ORDER), name

    def test_the_padded_case_carries_values_in_the_masked_tail(self):
        """The form that serves as the mask path's detector (all-zero padding columns could not
        detect a leak)."""
        _, length, pad = next(case for case in export_sbv2.GOLDEN_CASES if case[2] > 0)
        inputs = export_sbv2.front_inputs(length, pad, torch.zeros(1, 512, 1), torch.zeros(1, 256))

        assert int(inputs["x"][..., length - pad :].abs().sum()) > 0
        assert float(inputs["bert"][..., length - pad :].abs().sum()) > 0
        assert float(inputs["z_noise"][..., length - pad :].abs().sum()) > 0


@pytest.fixture(scope="module")
def exported_flow(tmp_path_factory):
    """Exports flow into a temporary directory (applies the patches process-wide)."""
    out = tmp_path_factory.mktemp("sbv2-flow") / export_sbv2.TARGET_FLOW
    return out, export_sbv2.export_flow(MODEL_DIR, out)


@pytest.fixture(scope="module")
def exported_dec(tmp_path_factory):
    out = tmp_path_factory.mktemp("sbv2-dec") / export_sbv2.TARGET_DEC
    return out, export_sbv2.export_dec(MODEL_DIR, out)


@pytest.fixture(scope="module")
def exported_voice(tmp_path_factory):
    out = tmp_path_factory.mktemp("sbv2-voice") / export_sbv2.TARGET_VOICE
    return out, export_sbv2.export_voice(MODEL_DIR, out)


@pytest.fixture(scope="module")
def flow_module(exported_flow):
    """An eager module that computes the same as the golden (the patches were applied on the
    export side)."""
    net_g = export_sbv2.load_net_g(MODEL_DIR)[0]
    patch_sbv2.apply_all_patches()
    return patch_sbv2.FlowReverse(net_g)


@pytest.fixture(scope="module")
def dec_module(exported_dec):
    net_g = export_sbv2.load_net_g(MODEL_DIR)[0]
    export_sbv2.ensure_dec_plain(net_g)
    return net_g.dec


@pytest.fixture(scope="module")
def voice_module(exported_voice):
    net_g = export_sbv2.load_net_g(MODEL_DIR)[0]
    patch_sbv2.apply_all_patches()
    export_sbv2.ensure_dec_plain(net_g)
    return patch_sbv2.Sbv2Voice(net_g)


@requires_weights
class TestFlowExport:
    """flow — the only lineage that promoted the relative-position tables to **graph inputs**
    (shared with voice)."""

    def test_the_container_passes_the_full_verification(self, exported_flow):
        out, _ = exported_flow

        _verified_graph(out / export_sbv2.MODEL_FILE)

    def test_the_graph_declares_the_wrapper_argument_names(self, exported_flow):
        out, summary = exported_flow
        graph = _verified_graph(out / export_sbv2.MODEL_FILE)

        assert [spec["name"] for spec in graph["inputs"]] == list(export_sbv2.FLOW_INPUT_ORDER)
        assert [spec["shape"] for spec in graph["inputs"]] == [
            [1, 192, "T"],
            [1, 1, "T"],
            [1, 512, 1],
            ["T", "T"],
            ["T", "T"],
        ]
        # The tables are i64 → i32 (boundary normalization — ADR 0009) and f32.
        assert [spec["dtype"] for spec in graph["inputs"]][3:] == ["i32", "f32"]
        assert summary["symbols"] == ["T"]

    def test_the_graph_stays_inside_the_current_contract(self, exported_flow):
        _, summary = exported_flow

        # The premise of iteration 7: flow too passes **without widening the vocabulary**. Whether
        # the rank-4 matmul lowers to view→bmm→view through the core decomposition (an open item of
        # recon §6) is answered here — bmm appearing without matmul means it did.
        assert set(summary["ops"]) <= set(EMITTABLE_OPS)
        assert "bmm" in summary["ops"]
        assert "matmul" not in summary["ops"]

    def test_the_relative_position_tables_are_inputs_not_constants(self, exported_flow):
        """The `(T,T)` tables are not baked (ADR 0013's ruling itself).

        Baked at sym_max=4096, `idx_k` + `valid` alone would be 134MB. The only bake allowed to
        remain is the value side's `idx_v` `(Tmax, 2w+1)`, which is 4096×9, about 150KB.
        """
        out, _ = exported_flow
        graph = _verified_graph(out / export_sbv2.MODEL_FILE)
        sources = [
            tuple(graph["values"][node["ins"][0]]["shape"])
            for node in graph["nodes"]
            if node["op"] == "sym_prefix_slice"
        ]

        tmax = export_sbv2.FLOW_SYM_MAX
        window = 2 * export_sbv2.EXPECTED_WINDOW_SIZE + 1
        assert sources == [(1, 1, tmax, window)]

    def test_the_baked_constants_stay_far_below_the_promoted_tables(self, exported_flow):
        """The baked constants total under 1MB (the control for "134MB without promotion")."""
        out, _ = exported_flow
        graph = _verified_graph(out / export_sbv2.MODEL_FILE)
        baked = sum(
            math.prod(graph["values"][name]["shape"]) * 4
            for name in graph["initializers"]
            if name.startswith("const.")
        )

        assert baked < 1024 * 1024, baked

    def test_the_io_files_follow_the_naming_convention(self, exported_flow):
        out, summary = exported_flow

        expected = [
            f"{export_sbv2.IO_PREFIX}{name}{export_sbv2.IO_SUFFIX}"
            for name, _, _ in export_sbv2.GOLDEN_CASES
        ]
        assert summary["io"] == expected
        for name in expected:
            with safe_open(str(out / name), framework="pt") as handle:
                keys = set(handle.keys())
            assert keys == {
                *(f"input.{declared}" for declared in export_sbv2.FLOW_INPUT_ORDER),
                "output.0",
            }

    @pytest.mark.parametrize("case", export_sbv2.GOLDEN_CASES, ids=lambda case: case[0])
    def test_io_shapes_bind_the_symbol_to_the_case_length(self, exported_flow, case):
        out, _ = exported_flow
        name, length, _ = case
        graph = _verified_graph(out / export_sbv2.MODEL_FILE)
        tensors = _io_tensors(out, name)

        for spec in graph["inputs"]:
            actual = list(tensors[f"input.{spec['name']}"].shape)
            declared = [
                length if isinstance(dim, str) and parse_dim(dim).sym == "T" else dim
                for dim in spec["shape"]
            ]
            assert actual == declared, spec["name"]
        assert list(tensors["output.0"].shape) == [1, 192, length]

    @pytest.mark.parametrize("case", export_sbv2.GOLDEN_CASES, ids=lambda case: case[0])
    def test_the_golden_output_reproduces_a_fresh_eager_forward(
        self, exported_flow, flow_module, case
    ):
        out, _ = exported_flow
        name, _, _ = case
        tensors = _io_tensors(out, name)
        with torch.no_grad():
            fresh = flow_module(*(tensors[f"input.{d}"] for d in export_sbv2.FLOW_INPUT_ORDER))

        assert torch.equal(fresh, tensors["output.0"])

    @pytest.mark.parametrize("case", export_sbv2.GOLDEN_CASES, ids=lambda case: case[0])
    def test_a_corrupted_golden_is_detected(self, exported_flow, flow_module, case):
        """Fault injection — corrupting one element of the expected values turns the test above red
        (the guard against a vacuous check)."""
        out, _ = exported_flow
        name, _, _ = case
        tensors = _io_tensors(out, name)
        with torch.no_grad():
            fresh = flow_module(*(tensors[f"input.{d}"] for d in export_sbv2.FLOW_INPUT_ORDER))
        corrupted = tensors["output.0"].clone()
        corrupted[0, 0, 0] += 1e-3

        assert not torch.equal(fresh, corrupted)

    def test_a_shifted_table_changes_the_golden(self, exported_flow, flow_module):
        """Fault injection — shifting the table by 1 disagrees with the golden (checked on real
        weights, all 24 layers).

        Detection on one synthetic attention layer belongs to `test_patch.py`. This measures that
        "the tables carried as graph inputs really are used by the attention of all 24 layers" — if
        the tables were dropped midway and fell back to in-graph construction, shifting them would
        not move the values.
        """
        out, _ = exported_flow
        name, _, _ = next(case for case in export_sbv2.GOLDEN_CASES if case[1] > 2 * 4)
        tensors = _io_tensors(out, name)
        args = [tensors[f"input.{d}"] for d in export_sbv2.FLOW_INPUT_ORDER]
        index = export_sbv2.FLOW_INPUT_ORDER.index("idx_k")
        args[index] = torch.clamp(args[index].to(torch.int64) + 1, 0, 2 * 4)
        with torch.no_grad():
            shifted = flow_module(*args)

        assert not torch.equal(shifted, tensors["output.0"])

    def test_the_padded_case_zeroes_the_masked_tail(self, exported_flow):
        """If the `y_mask` multiplication takes effect, the padding columns' output is exactly 0.

        From the second stage on, both halves of a coupling go through the masked side, so the
        final output has a zero tail on all 192 channels (zeros on one side only would mean the
        coupling order is mixed up).
        """
        out, _ = exported_flow
        name, length, pad = next(case for case in export_sbv2.GOLDEN_CASES if case[2] > 0)
        output = _io_tensors(out, name)["output.0"]
        tail = output[..., length - pad :]

        assert torch.equal(tail, torch.zeros_like(tail))

    def test_the_masked_tail_assertion_is_not_vacuous(self, exported_flow, flow_module):
        """Fault injection — replacing the mask with all ones makes the tail non-zero."""
        out, _ = exported_flow
        name, length, pad = next(case for case in export_sbv2.GOLDEN_CASES if case[2] > 0)
        tensors = _io_tensors(out, name)
        args = [tensors[f"input.{d}"] for d in export_sbv2.FLOW_INPUT_ORDER]
        args[export_sbv2.FLOW_INPUT_ORDER.index("y_mask")] = torch.ones(1, 1, length)
        with torch.no_grad():
            leaked = flow_module(*args)
        tail = leaked[..., length - pad :]

        assert not torch.equal(tail, torch.zeros_like(tail))

    def test_regeneration_is_byte_identical(self, exported_flow, tmp_path):
        out, _ = exported_flow
        again = tmp_path / export_sbv2.TARGET_FLOW
        export_sbv2.export_flow(MODEL_DIR, again)

        for name in sorted(path.name for path in out.iterdir()):
            assert (again / name).read_bytes() == (out / name).read_bytes(), name


@requires_weights
class TestDecExport:
    """dec — needs no patches; the only preprocessing is `remove_weight_norm`."""

    def test_the_container_passes_the_full_verification(self, exported_dec):
        out, _ = exported_dec

        _verified_graph(out / export_sbv2.MODEL_FILE)

    def test_the_graph_declares_the_generator_argument_names(self, exported_dec):
        out, summary = exported_dec
        graph = _verified_graph(out / export_sbv2.MODEL_FILE)

        # There is no wrapper, so the argument names of `Generator.forward(x, g)` appear as they
        # are.
        assert [spec["name"] for spec in graph["inputs"]] == list(export_sbv2.DEC_INPUT_ORDER)
        assert [spec["shape"] for spec in graph["inputs"]] == [[1, 192, "T"], [1, 512, 1]]
        assert summary["symbols"] == ["T"]

    def test_the_graph_stays_inside_the_current_contract(self, exported_dec):
        _, summary = exported_dec

        # The ops iteration 5 (conv family extension) added for dec appear as they are. If the ops
        # grow, "another path appeared without anyone noticing", so fail loudly.
        assert set(summary["ops"]) <= set(EMITTABLE_OPS)
        assert set(summary["ops"]) == {
            "add",
            "conv1d",
            "conv_transpose1d",
            "div",
            "leaky_relu",
            "tanh",
        }

    def test_the_output_length_is_the_upsampling_product(self, exported_dec):
        """The output length is exactly 512·T (ConvTranspose's `pad=(k−u)//2` takes effect)."""
        out, _ = exported_dec
        graph = _verified_graph(out / export_sbv2.MODEL_FILE)
        shape = graph["values"][graph["outputs"][0]]["shape"]

        assert shape[:2] == [1, 1]
        assert parse_dim(shape[2]).coeff == 512
        assert parse_dim(shape[2]).offset == 0

    def test_the_io_files_follow_the_naming_convention(self, exported_dec):
        out, summary = exported_dec

        for name in summary["io"]:
            with safe_open(str(out / name), framework="pt") as handle:
                keys = set(handle.keys())
            assert keys == {"input.x", "input.g", "output.0"}

    @pytest.mark.parametrize("case", export_sbv2.GOLDEN_CASES, ids=lambda case: case[0])
    def test_io_shapes_bind_the_symbol_to_the_case_length(self, exported_dec, case):
        out, _ = exported_dec
        name, length, _ = case
        tensors = _io_tensors(out, name)

        assert list(tensors["input.x"].shape) == [1, 192, length]
        assert list(tensors["output.0"].shape) == [1, 1, 512 * length]

    @pytest.mark.parametrize("case", export_sbv2.GOLDEN_CASES, ids=lambda case: case[0])
    def test_the_golden_output_reproduces_a_fresh_eager_forward(
        self, exported_dec, dec_module, case
    ):
        out, _ = exported_dec
        name, _, _ = case
        tensors = _io_tensors(out, name)
        with torch.no_grad():
            fresh = dec_module(tensors["input.x"], g=tensors["input.g"])

        assert torch.equal(fresh, tensors["output.0"])

    @pytest.mark.parametrize("case", export_sbv2.GOLDEN_CASES, ids=lambda case: case[0])
    def test_a_corrupted_golden_is_detected(self, exported_dec, dec_module, case):
        out, _ = exported_dec
        name, _, _ = case
        tensors = _io_tensors(out, name)
        with torch.no_grad():
            fresh = dec_module(tensors["input.x"], g=tensors["input.g"])
        corrupted = tensors["output.0"].clone()
        corrupted[0, 0, 0] += 1e-3

        assert not torch.equal(fresh, corrupted)

    def test_removing_the_weight_norm_is_idempotent(self):
        """Idempotence of `ensure_dec_plain` (the fused voice goes through the same net_g twice)."""
        net_g = export_sbv2.load_net_g(MODEL_DIR)[0]
        export_sbv2.ensure_dec_plain(net_g)
        weight = net_g.dec.conv_pre.weight.clone()

        export_sbv2.ensure_dec_plain(net_g)

        assert torch.equal(net_g.dec.conv_pre.weight, weight)

    def test_a_dec_with_weight_norm_is_refused_by_the_assertion(self):
        """Fault injection — asserting without going through remove fails (the guard of the order
        constraint)."""
        net_g = export_sbv2.load_net_g(MODEL_DIR)[0]

        with pytest.raises(ValueError, match="weight_norm"):
            export_sbv2._assert_no_weight_norm(net_g.dec)

    def test_regeneration_is_byte_identical(self, exported_dec, tmp_path):
        out, _ = exported_dec
        again = tmp_path / export_sbv2.TARGET_DEC
        export_sbv2.export_dec(MODEL_DIR, again)

        for name in sorted(path.name for path in out.iterdir()):
            assert (again / name).read_bytes() == (out / name).read_bytes(), name


@requires_weights
class TestVoiceExport:
    """voice — the fusion of flow + dec. **When this target passes, the whole SBV2 chain is
    complete**."""

    def test_the_container_passes_the_full_verification(self, exported_voice):
        out, _ = exported_voice

        _verified_graph(out / export_sbv2.MODEL_FILE)

    def test_the_graph_takes_the_flow_inputs_and_returns_audio(self, exported_voice):
        out, summary = exported_voice
        graph = _verified_graph(out / export_sbv2.MODEL_FILE)

        assert [spec["name"] for spec in graph["inputs"]] == list(export_sbv2.FLOW_INPUT_ORDER)
        assert summary["outputs"] == 1
        assert parse_dim(graph["values"][graph["outputs"][0]]["shape"][2]).coeff == 512

    def test_the_fusion_covers_both_halves(self, exported_voice, exported_flow, exported_dec):
        """The fused graph holds the ops of **both** flow and dec (neither has been dropped).

        The node count should also be close to the sum of the two — if it falls far short, one of
        the stages has disappeared.
        """
        _, voice = exported_voice
        _, flow = exported_flow
        _, dec = exported_dec

        assert set(voice["ops"]) == set(flow["ops"]) | set(dec["ops"])
        assert voice["nodes"] >= flow["nodes"] + dec["nodes"] - 10

    @pytest.mark.parametrize("case", export_sbv2.GOLDEN_CASES, ids=lambda case: case[0])
    def test_io_shapes_bind_the_symbol_to_the_case_length(self, exported_voice, case):
        out, _ = exported_voice
        name, length, _ = case
        tensors = _io_tensors(out, name)

        assert list(tensors["input.z_p"].shape) == [1, 192, length]
        assert list(tensors["input.idx_k"].shape) == [length, length]
        assert list(tensors["output.0"].shape) == [1, 1, 512 * length]

    @pytest.mark.parametrize("case", export_sbv2.GOLDEN_CASES, ids=lambda case: case[0])
    def test_the_golden_output_reproduces_a_fresh_eager_forward(
        self, exported_voice, voice_module, case
    ):
        out, _ = exported_voice
        name, _, _ = case
        tensors = _io_tensors(out, name)
        with torch.no_grad():
            fresh = voice_module(*(tensors[f"input.{d}"] for d in export_sbv2.FLOW_INPUT_ORDER))

        assert torch.equal(fresh, tensors["output.0"])

    @pytest.mark.parametrize("case", export_sbv2.GOLDEN_CASES, ids=lambda case: case[0])
    def test_a_corrupted_golden_is_detected(self, exported_voice, voice_module, case):
        out, _ = exported_voice
        name, _, _ = case
        tensors = _io_tensors(out, name)
        with torch.no_grad():
            fresh = voice_module(*(tensors[f"input.{d}"] for d in export_sbv2.FLOW_INPUT_ORDER))
        corrupted = tensors["output.0"].clone()
        corrupted[0, 0, 0] += 1e-3

        assert not torch.equal(fresh, corrupted)

    def test_the_fused_output_matches_the_two_stage_goldens(
        self, exported_voice, exported_flow, dec_module
    ):
        """The fused golden matches "flow's golden → dec" (measures what the fusion means).

        flow and voice share their inputs (`build_flow_cases`), so multiplying flow's expected
        output z by `y_mask` and passing it through dec should give voice's expected output. A
        mixed-up stage order or a missing `z * y_mask` fails here.
        """
        voice_out, _ = exported_voice
        flow_out, _ = exported_flow
        name, _, _ = next(case for case in export_sbv2.GOLDEN_CASES if case[2] > 0)
        flow_tensors = _io_tensors(flow_out, name)
        voice_tensors = _io_tensors(voice_out, name)
        with torch.no_grad():
            staged = dec_module(
                flow_tensors["output.0"] * flow_tensors["input.y_mask"],
                g=flow_tensors["input.g"],
            )

        assert torch.equal(staged, voice_tensors["output.0"])

    def test_regeneration_is_byte_identical(self, exported_voice, tmp_path):
        out, _ = exported_voice
        again = tmp_path / export_sbv2.TARGET_VOICE
        export_sbv2.export_voice(MODEL_DIR, again)

        for name in sorted(path.name for path in out.iterdir()):
            assert (again / name).read_bytes() == (out / name).read_bytes(), name


def _io_tensors(out_dir, case_name) -> dict[str, torch.Tensor]:
    path = out_dir / f"{export_sbv2.IO_PREFIX}{case_name}{export_sbv2.IO_SUFFIX}"
    with safe_open(str(path), framework="pt") as handle:
        # safe_open is not a Mapping, so keys() is the only way to enumerate.
        return {key: handle.get_tensor(key) for key in handle.keys()}  # noqa: SIM118


class TestTheVoiceFamilyIsDecidedByAnAllowList:
    """The decision that derives **legal facts** (the license) from the spelling of `--model-dir`.

    Back when it was a binary "jvnv unless it starts with `FN`", an unexpected directory name
    silently became a distributed form claiming jvnv's `cc-by-sa-4.0` — a silent wrong value
    nobody could notice until after distribution, so a name outside the allow list MUST fail
    loudly.
    """

    @pytest.mark.parametrize(
        ("name", "family"), [("FN4", "fn"), ("FN", "fn"), ("F1", "jvnv"), ("M2", "jvnv")]
    )
    def test_a_known_prefix_resolves_to_its_family(self, name: str, family: str) -> None:
        assert export_sbv2.sbv2_family(Path("inputs/sbv2") / name) == family

    @pytest.mark.parametrize("name", ["my-fn-voice", "F3", "sbv2", "copy-of-FN4"])
    def test_an_unknown_directory_name_fails_loudly(self, name: str) -> None:
        # Fails with a dedicated type (with a plain `ValueError`, the caller's `except` could not
        # tell this legal-fact check from a generic "bad argument" — the repository's convention is
        # an `Error` subclass).
        with pytest.raises(
            export_sbv2.Sbv2FamilyError, match="どの声のファミリーの綴りにも当たらない"
        ):
            export_sbv2.sbv2_family(Path("inputs/sbv2") / name)

    def test_every_family_in_the_allow_list_has_a_card_profile(self) -> None:
        """The allow list and the card's attribution table are the same set (closes the form where
        only one grows)."""
        assert set(export_sbv2.SBV2_FAMILY_DIRS) == set(SBV2_CARD_PROFILES)

    @pytest.mark.parametrize(("name", "family"), [("FN4", "fn"), ("F1", "jvnv")])
    def test_the_license_comes_from_the_card_profile(self, name: str, family: str) -> None:
        provenance = export_sbv2.sbv2_provenance(Path("inputs/sbv2") / name)

        assert provenance.license == SBV2_CARD_PROFILES[family].metadata.license
        assert provenance.upstream_revision == SBV2_CARD_PROFILES[family].source_revision

    def test_the_revision_seat_carries_a_commit_sha_not_the_config_version(self) -> None:
        """`upstreamRevision` is the upstream checkpoint's revision (container-v1 §2.3).

        The config's `version` (`2.0-JP-Extra`) is shared by every model of the same generation,
        so it cannot identify a checkpoint — since minicpm5 writes a SHA into the same field, the
        field's meaning is not split (W-RC5-3).
        """
        provenance = export_sbv2.sbv2_provenance(Path("inputs/sbv2") / "F1")

        assert provenance.upstream_revision is not None
        assert re.fullmatch(r"[0-9a-f]{40}", provenance.upstream_revision)
        assert provenance.upstream_revision != SBV2_CARD_PROFILES["jvnv"].source_version
