"""Generation of the tiny golden fixtures, and the conventions the downstream E2E
(`packages/runtime/tests/e2e_golden_test.ts`) relies on, pinned.

There are three conventions: ① every op of the contract table is exercised by some model ② the
layout and the tensor keys follow the README ③ regenerating with the fixed seed gives the same
bytes.
"""

from __future__ import annotations

import itertools
import os
import re
from pathlib import Path

import pytest
import torch
from safetensors import safe_open

from karume import goldens, publish
from karume.container import container_parts
from karume.dims import eval_dim, parse_dim
from karume.goldens import (
    GOLDEN_SPECS,
    GOLDEN_T,
    INPUT_PREFIX,
    IO_FILE,
    MODEL_FILE,
    OUTPUT_PREFIX,
    generate_all,
    generate_golden,
)
from karume.ir import IrGraph
from karume.ops import EMITTABLE_OPS
from karume.verify import verify_container

#: Where the committed goldens live. **The test side owns the repository spelling** — the generator
#: (`karume.goldens`) only takes the location as an argument and does not know the repo topology
#: (ADR 0065 Consequences).
GOLDEN_ROOT = (
    Path(__file__).resolve().parents[3] / "packages" / "runtime" / "tests" / "fixtures" / "golden"
)

#: Downstream tests read these files, so they stay small enough to keep in the repository.
MAX_FILE_BYTES = 16 * 1024

#: Semantic dtype → the torch dtype stored in io.safetensors (the boundary normalization of ADR
#: 0009).
IO_ENCODING = {"f32": torch.float32, "i32": torch.int32, "bool": torch.uint32}


@pytest.fixture(scope="module")
def generated(tmp_path_factory):
    """Regenerates every golden into a temporary directory (the repository's files are not
    touched)."""
    root = tmp_path_factory.mktemp("golden")
    return root, generate_all(root)


def _io_tensors(root, name) -> dict[str, torch.Tensor]:
    with safe_open(str(root / name / IO_FILE), framework="pt") as handle:
        # safe_open is not a Mapping, so keys() is the only way to enumerate.
        return {key: handle.get_tensor(key) for key in handle.keys()}  # noqa: SIM118


def _bindings(graph: IrGraph, io: dict[str, torch.Tensor]) -> dict[str, int]:
    """Binds the symbols from the dimension positions of the input shapes (no back-calculation from
    element counts)."""
    bindings: dict[str, int] = {}
    for spec in graph.inputs:
        actual = io[f"{INPUT_PREFIX}{spec.name}"].shape
        for index, dim in enumerate(spec.shape):
            if not isinstance(dim, str):
                continue
            expr = parse_dim(dim)
            if expr.coeff == 1 and expr.offset == 0:
                bindings[expr.sym] = int(actual[index])
    return bindings


#: The marker of the golden ledger in the README (the first table after this line is "model name |
#: symbolic dimensions | IR ops").
COVERAGE_MARKER = "Current models and coverage:"
#: One row of the coverage table (the header and separator rows carry no name, so they drop out).
COVERAGE_ROW = re.compile(r"^\| `([a-z0-9_]+)`\s+\|")


def _readme_coverage_names() -> list[str]:
    """The model names in the README's coverage table (cut by position so that other tables such as
    the Module table do not mix in)."""
    text = (Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")
    _, marker, after = text.partition(COVERAGE_MARKER)
    assert marker, f"the README has no '{COVERAGE_MARKER}' (cannot locate the table)"
    rows = itertools.takewhile(
        lambda line: line.startswith("|"),
        itertools.dropwhile(lambda line: not line.startswith("|"), after.splitlines()),
    )
    return [match.group(1) for match in map(COVERAGE_ROW.match, rows) if match]


class TestCoverage:
    def test_the_readme_ledger_lists_exactly_the_generated_models(self):
        """Makes drift between the ledger and the generated models a check (the visible side of the
        contract "add an op, add a golden" — ADR 0005).

        The table is the primary source for recipe authors, so leaving it to manual updates makes
        it lag the same way on every op addition (it actually lagged by the two of ADR 0017).
        """
        assert set(_readme_coverage_names()) == {spec.name for spec in GOLDEN_SPECS}

    def test_every_contract_op_is_exercised_by_some_model(self, generated):
        _, graphs = generated
        covered = {op for graph in graphs.values() for op in graph.required_ops}

        assert covered == set(EMITTABLE_OPS)

    def test_a_model_carries_a_symbolic_input_dimension(self, generated):
        _, graphs = generated

        assert any(graph.symbols for graph in graphs.values())

    def test_a_model_exercises_right_aligned_broadcast(self, generated):
        """At least one binary op has operands of different ranks."""
        _, graphs = generated
        ranks = [
            {len(_declared_shape(graph, name)) for name in node.ins}
            for graph in graphs.values()
            for node in graph.nodes
            if node.op in {"add", "sub", "mul", "div"}
        ]

        assert any(len(distinct) > 1 for distinct in ranks)

    def test_a_model_carries_a_non_f32_output(self, generated):
        """Puts the non-f32 readback path into the E2E (ADR 0009) — with f32 only it is never
        exercised."""
        _, graphs = generated
        out_dtypes = {
            graph.values[name].dtype for graph in graphs.values() for name in graph.outputs
        }

        assert out_dtypes - {"f32"}

    def test_a_model_exercises_the_i64_input_boundary(self, generated):
        """torch's default integer i64 is normalized to i32 in the input declarations (ADR 0009)."""
        _, graphs = generated
        in_dtypes = {spec.dtype for graph in graphs.values() for spec in graph.inputs}

        assert "i32" in in_dtypes

    def test_the_bmm_model_uses_four_distinct_axis_lengths(self, generated):
        """B / M / K / N all have different lengths (ACTIVE_DESIGN Pitfalls).

        The mistake of building the batch stride from the product of the neighboring dimensions
        does not show in the numbers on a shape where two axes have the same length. This fails the
        moment the golden is shrunk to a symmetric shape.
        """
        _, graphs = generated
        bmms = [
            (_declared_shape(graph, node.ins[0]), _declared_shape(graph, node.ins[1]))
            for graph in graphs.values()
            for node in graph.nodes
            if node.op == "bmm"
        ]

        assert bmms
        assert any(
            len({a[0], a[1], a[2], b[2]}) == 4 and len({a[1] * a[2], a[2] * b[2], a[1] * b[2]}) == 3
            for a, b in bmms
        )

    def test_the_attention_model_uses_five_distinct_axis_lengths(self, generated):
        """The fused-attention golden has B / H / M / N / D all of different lengths (ADR 0023).

        The kernel folds B and H into one batch axis, so with B=1 (the observed shape) or B==H a
        mix-up of the axes does not show in the values (detection limit ① of the design recon,
        docs/research/2026-08-03-attention-design.md §4.6). This fails the moment the golden is
        shrunk to a symmetric shape.
        """
        _, graphs = generated
        attentions = [
            [_declared_shape(graph, name) for name in node.ins]
            for graph in graphs.values()
            for node in graph.nodes
            if node.op == "attention"
        ]

        assert attentions, "no golden exercises attention"
        assert any(len({q[0], q[1], q[2], k[2], q[3]}) == 5 for q, k, _ in attentions), (
            f"the axes do not have 5 distinct lengths: {attentions}"
        )

    def test_the_decomposed_attention_models_are_untouched(self, generated):
        """Preserving SDPA is per target (ADR 0023) — the existing goldens stay in decomposed form.

        `attention_block` is a hand-written decomposition with `torch.bmm` + `torch.softmax`, and
        `masked_scores` does not go through SDPA at all. If this moves, the scope of preservation
        has leaked.
        """
        _, graphs = generated

        for name in ("attention_block", "masked_scores"):
            ops = {node.op for node in graphs[name].nodes}
            assert "attention" not in ops, f"attention leaked into {name}"
        assert "bmm" in {node.op for node in graphs["attention_block"].nodes}
        assert "softmax" in {node.op for node in graphs["masked_scores"].nodes}

    def test_every_gather_index_is_i32_from_both_sources(self, generated):
        """The index slot of gather is always semantic i32 (the per-slot dtype contract — ADR 0012).

        It comes from two places: a graph input (gather_last_dim) and **the prefix of an i32
        initializer** baked from a symbol-dependent subtree at Tmax (symbolic_table — ADR 0010).
        Both are required in the goldens — with only one, one of the two i32 index paths stays
        untested.
        """
        _, graphs = generated
        index_slots = [
            (model, graph, node.ins[1])
            for model, graph in graphs.items()
            for node in graph.nodes
            if node.op == "gather"
        ]

        assert index_slots
        sources = set()
        for model, graph, name in index_slots:
            declared = [spec for spec in graph.inputs if spec.name == name]
            dtype = declared[0].dtype if declared else graph.values[name].dtype
            assert dtype == "i32", f"{model}: gather index '{name}' is not i32"
            sources.add("input" if declared else "folded")
        assert sources == {"input", "folded"}

    def test_a_model_runs_matmul_over_initializer_weights(self, generated):
        _, graphs = generated
        matmuls = [
            node
            for graph in graphs.values()
            for node in graph.nodes
            if node.op == "matmul" and node.ins[1] in graph.initializers
        ]

        assert matmuls


def _declared_shape(graph: IrGraph, name: str) -> list:
    for spec in graph.inputs:
        if spec.name == name:
            return spec.shape
    return graph.values[name].shape


def _model_parts(root: Path, name: str) -> tuple[Path, ...]:
    """The model's actual files (the container's parts — from part 0).

    A distribution is always split into numbered parts (container-v1 §8), so the representative
    path `model.krm` itself is never written — counting the actual files is `container_parts`'s
    job.
    """
    return container_parts(root / name / MODEL_FILE)


class TestLayout:
    @pytest.mark.parametrize("spec", GOLDEN_SPECS, ids=lambda s: s.name)
    def test_each_model_directory_has_the_parts_and_the_io(self, generated, spec):
        root, _ = generated

        parts = _model_parts(root, spec.name)
        assert all(part.is_file() for part in parts)
        assert (root / spec.name / IO_FILE).is_file()
        # Part 0 (the two documents) + part 1 (the const region) are **always** there
        # (container-v1 §8), so even a golden without a single weight has 2 or more, and the
        # representative path itself is not written.
        assert len(parts) >= 2
        assert not (root / spec.name / MODEL_FILE).exists()

    @pytest.mark.parametrize("spec", GOLDEN_SPECS, ids=lambda s: s.name)
    def test_each_model_passes_the_full_verification(self, generated, spec):
        root, _ = generated

        verify_container(_model_parts(root, spec.name), blocks=True)

    @pytest.mark.parametrize("spec", GOLDEN_SPECS, ids=lambda s: s.name)
    def test_io_keys_follow_the_naming_convention(self, generated, spec):
        root, graphs = generated
        graph = graphs[spec.name]

        io = _io_tensors(root, spec.name)

        assert set(io) == {f"{INPUT_PREFIX}{s.name}" for s in graph.inputs} | {
            f"{OUTPUT_PREFIX}{index}" for index in range(len(graph.outputs))
        }

    @pytest.mark.parametrize("spec", GOLDEN_SPECS, ids=lambda s: s.name)
    def test_io_dtypes_use_the_runtime_representation(self, generated, spec):
        """io is written in the actual representation of the semantic dtype (i64 → int32 / bool →
        uint32 0/1).

        The downstream E2E builds its views on this mapping, so a mismatch makes the golden
        unreadable.
        """
        root, graphs = generated
        graph = graphs[spec.name]
        io = _io_tensors(root, spec.name)

        for declared in graph.inputs:
            assert io[f"{INPUT_PREFIX}{declared.name}"].dtype is IO_ENCODING[declared.dtype]
        for index, name in enumerate(graph.outputs):
            assert io[f"{OUTPUT_PREFIX}{index}"].dtype is IO_ENCODING[graph.values[name].dtype]

    @pytest.mark.parametrize("spec", GOLDEN_SPECS, ids=lambda s: s.name)
    def test_io_shapes_agree_with_the_declarations(self, generated, spec):
        """The declared shape resolved with the bindings matches the actual tensor (a semantic check
        of the layout)."""
        root, graphs = generated
        graph = graphs[spec.name]
        io = _io_tensors(root, spec.name)
        bindings = _bindings(graph, io)

        for index, name in enumerate(graph.outputs):
            declared = [
                eval_dim(parse_dim(dim), bindings) if isinstance(dim, str) else dim
                for dim in _declared_shape(graph, name)
            ]
            assert declared == list(io[f"{OUTPUT_PREFIX}{index}"].shape)

    @pytest.mark.parametrize("spec", GOLDEN_SPECS, ids=lambda s: s.name)
    def test_symbolic_models_are_bound_to_the_golden_length(self, generated, spec):
        root, graphs = generated
        graph = graphs[spec.name]

        bindings = _bindings(graph, _io_tensors(root, spec.name))

        assert set(bindings) == set(graph.symbols)
        assert all(value == GOLDEN_T for value in bindings.values())

    @pytest.mark.parametrize("spec", GOLDEN_SPECS, ids=lambda s: s.name)
    def test_fixtures_stay_small(self, generated, spec):
        root, _ = generated

        for path in (*_model_parts(root, spec.name), root / spec.name / IO_FILE):
            assert path.stat().st_size < MAX_FILE_BYTES


class TestDeterminism:
    """Regeneration with the fixed seed is byte-identical (the same discipline as codegen
    determinism).

    The byte match of io (torch CPU's expected outputs), however, is **for the reference environment
    only** — oneDNN dispatches gemm / conv kernels by the CPU's ISA, so the last bit of the results
    depends on the machine (observed on the first CI run, 2026-08-16: on the GitHub runner only the
    2 specs activations / conv2d_block differ by ±1–2 ulp). The model (graph + fixed-seed weights —
    torch's CPU RNG is deterministic across machines) is required to match bytes in every
    environment. The same nature as the sha256 reference check being for the reference environment
    only (docs/limitations.md): the check is split by environment instead of being loosened.
    """

    @pytest.mark.parametrize("spec", GOLDEN_SPECS, ids=lambda s: s.name)
    def test_two_runs_produce_the_same_model_bytes(self, generated, tmp_path, spec):
        """Generating twice from the same seed matches bytes (determinism itself is the subject).

        A different location does not move the bytes (file names do not go into the content).
        """
        root, _ = generated
        generate_golden(spec, tmp_path)

        assert [path.read_bytes() for path in _model_parts(root, spec.name)] == [
            path.read_bytes() for path in _model_parts(tmp_path, spec.name)
        ]

    @pytest.mark.parametrize("spec", GOLDEN_SPECS, ids=lambda s: s.name)
    def test_regeneration_matches_the_committed_model(self, generated, spec):
        root, _ = generated

        generated_parts = _model_parts(root, spec.name)
        committed = _model_parts(GOLDEN_ROOT, spec.name)
        assert [path.name for path in committed] == [path.name for path in generated_parts], (
            f"generated files are not committed: {GOLDEN_ROOT / spec.name}"
        )
        assert [path.read_bytes() for path in committed] == [
            path.read_bytes() for path in generated_parts
        ]

    @pytest.mark.skipif(
        os.environ.get("CI") == "true",
        reason="the last bit of io's torch CPU output depends on the machine (oneDNN's per-ISA "
        "kernels) — the byte match is for the reference environment only",
    )
    @pytest.mark.parametrize("spec", GOLDEN_SPECS, ids=lambda s: s.name)
    def test_regeneration_matches_the_committed_reference_io(self, generated, spec):
        root, _ = generated

        committed = GOLDEN_ROOT / spec.name / IO_FILE
        assert committed.is_file(), f"generated files are not committed: {committed}"
        assert committed.read_bytes() == (root / spec.name / IO_FILE).read_bytes()


def _tree_bytes(directory: Path) -> dict[str, bytes]:
    return {
        path.relative_to(directory).as_posix(): path.read_bytes()
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


class TestAFailedRegenerationKeepsTheOldFixture:
    """Even when the generation of one model fails partway, the previous fixture does not change by
    a single byte.

    Writing the model straight to its final location and checking afterwards leaves "new model +
    old io" when the check fails (the MUST of `publish_container` — called outside the staging
    directory, the previous artifacts can be lost). The previous content is replaced by sentinels
    so that it differs from the regenerated bytes (if deterministic generation wrote the same bytes
    back, an overwrite could not be detected).
    """

    SPEC = GOLDEN_SPECS[0]

    @pytest.fixture
    def previous(self, tmp_path):
        generate_golden(self.SPEC, tmp_path)
        directory = tmp_path / self.SPEC.name
        for index, path in enumerate(sorted(directory.iterdir())):
            path.write_bytes(f"previous-{index}".encode())
        return tmp_path, _tree_bytes(directory)

    def test_a_failed_check_after_publication_leaves_it_untouched(self, previous, monkeypatch):
        root, before = previous

        def refuse(where, tensor):
            raise AssertionError(f"{where}: a run that failed the check")

        monkeypatch.setattr(goldens, "_assert_not_trivial", refuse)
        with pytest.raises(AssertionError, match="a run that failed the check"):
            generate_golden(self.SPEC, root)

        assert _tree_bytes(root / self.SPEC.name) == before
        assert sorted(path.name for path in root.iterdir()) == [self.SPEC.name]

    def test_a_failed_part_replacement_leaves_it_untouched(self, previous, monkeypatch):
        root, before = previous
        real = os.replace

        def flaky(source, target):
            # Which swap it is is decided by the target name — `publish.os` is the os module
            # itself, so counting calls in order would also count the swap of the staging
            # directory (`staged_publication`), and a change in publish's internal order would turn
            # this into a test that fails a different step. The second part comes after the first
            # is in place, so it is "partway through the swap of the parts".
            if re.search(r"-00002-of-\d{5}\.krm$", os.fspath(target)):
                raise OSError("a run that failed partway through the swap")
            real(source, target)

        monkeypatch.setattr(publish.os, "replace", flaky)
        with pytest.raises(OSError, match="partway through the swap"):
            generate_golden(self.SPEC, root)

        assert _tree_bytes(root / self.SPEC.name) == before
        assert sorted(path.name for path in root.iterdir()) == [self.SPEC.name]
