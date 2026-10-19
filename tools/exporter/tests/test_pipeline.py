"""Atomicity of the single path (export → storage conversion → write → read back and verify →
swap).

The final path is not touched until verification passes — a run that fails midway does not change
the previous distribution by a single byte.
"""

from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch import nn

from karume import publish, verify
from karume.container import ContainerFormatError, Provenance, container_parts
from karume.ir import IrNode, IrValue
from karume.ops import OpContractError
from karume.pipeline import export_module, export_to_file, publish_model
from karume.verify import ContainerError, verify_container

PROVENANCE = Provenance(license="apache-2.0", writer="karume-test")


class Biased(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.arange(4, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.bias


class TwoWeights(nn.Module):
    """Holds two stored tensors (f32, 4 elements = 16 bytes each) — the minimal shape that exercises
    the part split."""

    def __init__(self) -> None:
        super().__init__()
        self.left = nn.Parameter(torch.arange(4, dtype=torch.float32))
        self.right = nn.Parameter(torch.arange(4, dtype=torch.float32) * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x + self.left) * self.right


EXAMPLE = (torch.randn(2, 4),)
TWO_EXAMPLE = (torch.randn(2, 4),)

#: An artifact already shipped (bytes that a failed re-export must not lose).
SENTINEL = b"already-shipped"

#: One weight of `TwoWeights` (f32, 4 elements). Lowering the part length to this gives one part
#: per weight.
ONE_WEIGHT_BYTES = 16


def break_write(monkeypatch: pytest.MonkeyPatch) -> None:
    """The write fails partway through (the shape of a full disk or an I/O fault)."""

    def half_written(path, *args, **kwargs):
        Path(path).write_bytes(b"half")
        raise OSError("failed while writing")

    monkeypatch.setattr(publish, "write_model_container", half_written)


def break_verify(monkeypatch: pytest.MonkeyPatch) -> None:
    """The written file cannot be read back (= it could not run once distributed)."""

    def reject(paths):
        raise ContainerFormatError("unreadable file")

    monkeypatch.setattr(publish, "read_container", reject)


def published(module: nn.Module, example, path: Path, **kwargs):
    """Convert with `export_module`, then place it (injecting the part length is the publish side's
    option)."""
    kwargs.setdefault("graph_name", "series")
    return publish_model(path, *export_module(module, example), provenance=PROVENANCE, **kwargs)


class TestAtomicReplacement:
    @pytest.mark.parametrize(
        ("inject", "error"), [(break_write, OSError), (break_verify, ContainerFormatError)]
    )
    def test_a_failed_export_leaves_the_existing_artifact_untouched(
        self, tmp_path, monkeypatch, inject, error
    ):
        final = tmp_path / "model.krm"
        final.write_bytes(SENTINEL)
        inject(monkeypatch)

        with pytest.raises(error):
            export_to_file(Biased(), EXAMPLE, final, provenance=PROVENANCE, graph_name="biased")

        assert final.read_bytes() == SENTINEL
        assert [entry.name for entry in tmp_path.iterdir()] == [final.name]

    @pytest.mark.parametrize(
        ("inject", "error"), [(break_write, OSError), (break_verify, ContainerFormatError)]
    )
    def test_a_failed_export_creates_nothing_when_there_is_no_previous_artifact(
        self, tmp_path, monkeypatch, inject, error
    ):
        final = tmp_path / "model.krm"
        inject(monkeypatch)

        with pytest.raises(error):
            export_to_file(Biased(), EXAMPLE, final, provenance=PROVENANCE, graph_name="biased")

        assert list(tmp_path.iterdir()) == []

    def test_the_delivered_bytes_do_not_depend_on_the_staging_path(self, tmp_path):
        """Going through a temporary file does not change the distribution's bytes (the name does
        not enter the contents)."""
        left = tmp_path / "series" / "model.krm"
        right = tmp_path / "series2" / "model.krm"

        export_to_file(Biased(), EXAMPLE, left, provenance=PROVENANCE, graph_name="series")
        export_to_file(Biased(), EXAMPLE, right, provenance=PROVENANCE, graph_name="series")

        assert [hashlib.sha256(p.read_bytes()).hexdigest() for p in container_parts(left)] == [
            hashlib.sha256(p.read_bytes()).hexdigest() for p in container_parts(right)
        ]

    @pytest.mark.parametrize("name", ["model.f32.safetensors", "model", "model.krm.tmp"])
    def test_a_path_without_the_container_suffix_is_rejected(self, tmp_path, name):
        """A path other than `.krm` fails instead of being silently reinterpreted, and nothing is
        placed.

        The distribution has exactly one form, `krm` (container-v1 §8).
        """
        with pytest.raises(ValueError, match=r"\.krm"):
            export_to_file(
                Biased(), EXAMPLE, tmp_path / name, provenance=PROVENANCE, graph_name="biased"
            )

        assert list(tmp_path.iterdir()) == []

    def test_publish_model_rejects_a_path_without_the_container_suffix(self, tmp_path):
        """The entry that takes an already exported graph fails the same way (no reinterpretation is
        left anywhere)."""
        with pytest.raises(ValueError, match=r"\.krm"):
            published(Biased(), EXAMPLE, tmp_path / "model.safetensors")

        assert list(tmp_path.iterdir()) == []


class TestPartitionedPublication:
    """The container is placed as a **numbered sequence of parts** (container-v1 §8 — the official
    HF distribution is the split form).

    The number of weight parts is raised by lowering the part length for the small synthetic
    tensors.
    """

    def test_it_publishes_the_numbered_sequence(self, tmp_path):
        # The weights are 2 × 16 bytes. With a part length of 16 each weight gets its own part
        # (+ part 0 and part 1).
        final = tmp_path / "model.krm"

        published(TwoWeights(), TWO_EXAMPLE, final, _part_bytes=ONE_WEIGHT_BYTES)

        assert sorted(entry.name for entry in tmp_path.iterdir()) == [
            "model-00001-of-00004.krm",
            "model-00002-of-00004.krm",
            "model-00003-of-00004.krm",
            "model-00004-of-00004.krm",
        ]

    def test_the_published_parts_pass_the_full_verification(self, tmp_path):
        final = tmp_path / "model.krm"

        graph = published(TwoWeights(), TWO_EXAMPLE, final, _part_bytes=ONE_WEIGHT_BYTES)

        written = container_parts(final)
        assert len(written) == 4
        # The caller names the graph (the component name `published` passes) — the default from
        # the parent directory name picked up the staging path's name and was retired
        # (`pipeline._assert_graph_name`).
        bound = verify_container(written, blocks=True).graphs["series"]
        assert sorted(bound.supplies) == sorted(
            graph.initializers[name].tensor for name in graph.initializers
        )

    def test_a_failed_export_leaves_no_part_behind(self, tmp_path, monkeypatch):
        """Temporary files written partway through the split are not left either (the whole
        `.partial` sequence is discarded)."""
        break_verify(monkeypatch)

        with pytest.raises(ContainerFormatError):
            published(TwoWeights(), TWO_EXAMPLE, tmp_path / "model.krm")

        assert list(tmp_path.iterdir()) == []

    def test_it_clears_the_parts_of_a_previous_export(self, tmp_path):
        """A re-export with a different part count leaves no debris (a mix is a state the assembly
        rejects)."""
        final = tmp_path / "model.krm"
        published(TwoWeights(), TWO_EXAMPLE, final)
        assert len(container_parts(final)) == 3

        published(TwoWeights(), TWO_EXAMPLE, final, _part_bytes=ONE_WEIGHT_BYTES)

        assert not final.exists()
        assert len(container_parts(final)) == 4
        assert len(list(tmp_path.iterdir())) == 4

    def test_it_clears_the_parts_of_a_previous_split(self, tmp_path):
        """Even a re-export with fewer parts does not leave the previous sequence behind."""
        final = tmp_path / "model.krm"
        published(TwoWeights(), TWO_EXAMPLE, final, _part_bytes=ONE_WEIGHT_BYTES)

        published(TwoWeights(), TWO_EXAMPLE, final)

        assert sorted(entry.name for entry in tmp_path.iterdir()) == [
            "model-00001-of-00003.krm",
            "model-00002-of-00003.krm",
            "model-00003-of-00003.krm",
        ]

    def test_the_cli_verifies_the_container_from_its_representative_path(self, tmp_path, capsys):
        """`karume verify <representative path>` resolves the sequence and verifies it as a whole.

        Verifying a single part on its own can only say "the 2 documents are missing", so the CLI
        argument is **the container's representative path** whatever the number of parts.
        """
        final = tmp_path / "model.krm"
        published(TwoWeights(), TWO_EXAMPLE, final, _part_bytes=ONE_WEIGHT_BYTES)

        verify.main([str(final)])

        assert "parts=4" in capsys.readouterr().out


class TestTheIrAcceptanceGate:
    """The IR acceptance rules (runtime support + op contracts) are applied **before** the write.

    The scripts insert graph surgery such as `to_states_form`, so the conversion step's checks
    alone cannot stop a container that "can be written but the runtime cannot read" — a graph with
    one op added from outside the vocabulary MUST fail without placing a single byte of krm (the
    write side of the purpose stated in the module doc of `verify.py`).
    """

    def _graph_with_an_unknown_op(self):
        graph, tensors = export_module(Biased(), EXAMPLE)
        out = "karume_test_unknown"
        graph.values[out] = IrValue(dtype="f32", shape=list(graph.values[graph.outputs[0]].shape))
        graph.nodes.append(
            IrNode(op="totally_unknown_op", ins=[graph.outputs[0]], outs=[out], attrs={})
        )
        graph.outputs.append(out)
        return graph, tensors

    def test_an_op_outside_the_vocabulary_places_nothing(self, tmp_path: Path) -> None:
        graph, tensors = self._graph_with_an_unknown_op()

        with pytest.raises(ContainerError, match="unsupported ops"):
            publish_model(
                tmp_path / "model.krm",
                graph,
                tensors,
                provenance=PROVENANCE,
                graph_name="series",
            )

        assert list(tmp_path.iterdir()) == []

    def test_a_contract_violation_places_nothing(self, tmp_path: Path) -> None:
        """Even inside the vocabulary, nothing is placed when the **contract** (the input count) is
        broken."""
        graph, tensors = export_module(Biased(), EXAMPLE)
        graph.nodes[-1] = replace(graph.nodes[-1], ins=graph.nodes[-1].ins[:1])

        with pytest.raises(OpContractError):
            publish_model(
                tmp_path / "model.krm",
                graph,
                tensors,
                provenance=PROVENANCE,
                graph_name="series",
            )

        assert list(tmp_path.iterdir()) == []

    def test_the_same_graph_without_the_extra_op_is_published(self, tmp_path: Path) -> None:
        """Control — the flip side showing that the check does not "always fail"."""
        graph, tensors = export_module(Biased(), EXAMPLE)

        publish_model(
            tmp_path / "model.krm", graph, tensors, provenance=PROVENANCE, graph_name="series"
        )

        assert container_parts(tmp_path / "model.krm")


class TestTheGraphNameIsNamedByTheCaller:
    """The graph name has no default (handled like `provenance`).

    When the default was the parent directory name, every recipe writing to a staging path
    (`<component>.staging/`) wrote a container whose graph was named `<component>.staging` — the
    vocabulary allows `.`, so it did not fail loudly either, and only when the runtime looked it
    up by component name did it become "the container has no such graph".
    """

    def test_the_container_carries_the_name_the_caller_gave(self, tmp_path: Path) -> None:
        staging = tmp_path / "dit.staging"
        staging.mkdir()

        published(Biased(), EXAMPLE, staging / "model.krm", graph_name="dit")

        bound = verify_container(container_parts(staging / "model.krm")).graphs

        assert sorted(bound) == ["dit"]

    def test_a_name_outside_the_vocabulary_fails_loudly(self, tmp_path: Path) -> None:
        with pytest.raises(ContainerFormatError, match="vocabulary"):
            published(Biased(), EXAMPLE, tmp_path / "model.krm", graph_name="dit/1")

        assert list(tmp_path.iterdir()) == []

    def test_a_name_with_a_trailing_newline_fails_loudly(self, tmp_path: Path) -> None:
        """Python's `$` also matches just before a trailing newline — a spelling that `.match` lets
        through and TS rejects."""
        with pytest.raises(ContainerFormatError, match="vocabulary"):
            published(Biased(), EXAMPLE, tmp_path / "model.krm", graph_name="dit\n")

        assert list(tmp_path.iterdir()) == []


class TestTheGraphWideGate:
    """Before publication the **graph-wide** structural checks (duplicate outputs, SSA, forward
    references, symbol binding) apply too.

    Per-op contract checks alone cannot stop an IR that went through graph surgery and ended up
    "each node correct but broken as a whole". A `.krm` that the runtime's parser rejects is not
    published (the failure is not carried over to `karume dist` or `karume verify`).
    """

    @staticmethod
    def _duplicate_output(graph):
        graph.outputs = [graph.outputs[0], graph.outputs[0]]

    @staticmethod
    def _unbound_symbol(graph):
        graph.symbols = [*graph.symbols, "U"]

    @staticmethod
    def _forward_reference(graph):
        graph.nodes = list(reversed(graph.nodes))

    @pytest.mark.parametrize(
        ("broken", "message"),
        [
            ("_duplicate_output", "is duplicated"),
            ("_unbound_symbol", "cannot be bound"),
            ("_forward_reference", r"is undefined \(forward reference or undeclared\)"),
        ],
        ids=["duplicate-output", "unbound-symbol", "forward-reference"],
    )
    def test_a_graph_broken_as_a_whole_places_nothing(
        self, tmp_path: Path, broken: str, message: str
    ) -> None:
        graph, tensors = export_module(TwoWeights(), TWO_EXAMPLE)
        assert len(graph.nodes) == 2, "a forward reference needs 2 nodes (a premise of the subject)"
        getattr(self, broken)(graph)

        with pytest.raises(publish.PublishError, match=message):
            publish_model(
                tmp_path / "model.krm",
                graph,
                tensors,
                provenance=PROVENANCE,
                graph_name="series",
            )

        assert list(tmp_path.iterdir()) == []
