"""The publication rule of the emit scripts — the mechanical check of "nothing is placed in final
before the checks", across every family.

The invariant the 2 decode-series scripts keep with `karume.artifacts.staged_publication` is
applied in the same form to the scripts of the one-shot series. Only one thing is guarded:

    Every call that writes an output (`export_to_file` / `_write_io` / `save_file` …) is
    **inside** the `with` of the staging directory.

If even one leaks outside, a failed run leaves "an asset that can pass the acceptance check" in
final — the io goldens are captured from the same broken wrapper, so they agree with each other,
and the TS-side match **goes green** (the rule is spelled out in `_shared/decode_series._publish`).
This check is needed because that state does not look like "the cleanup after a failed run was
forgotten" and **stays silent until the next acceptance check**.

The source is read as an AST because checking it by running would need the real weights (several
GB) or a tiny model per family. Only "the relative position of the calls" is checked here; the
rule of the swap itself (move aside → promote → restore on failure) is measured by the core's
`tools/exporter/tests/`.

"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

RECIPES_ROOT = Path(__file__).resolve().parent.parent

#: The names of the calls that open a staging directory (wrappers inside a family count too —
#: `sbv2._staged_target` wraps `staged_publication` and writes the provenance record into the same
#: directory; `wan.export_vae._staged_set` opens several directories together and places them as
#: one set after all the checks pass).
STAGING_CALLS = frozenset({"staged_publication", "_staged_target", "_staged_set"})

#: The calls that write outputs. The moment even one is outside the staging directory, that script
#: is back to "placing in final before the checks".
WRITER_CALLS = frozenset(
    {
        "export_to_file",
        "_write_container",
        "_write_io",
        "_write_mirror_io",
        "_write_greedy",
        "_write_export_provenance",
        "_write_lora_provenance",
        "_write_calib_provenance",
        "save_file",
        # The 2 used by gemma4's product script (writing the PLE shards and the provenance
        # record). Without their names in the table, listing that script would still go green
        # without looking at a single "place that writes".
        "write_ple_shards",
        "write_record",
        # The fixed-QAT container and PLE are also placed inside the same staging directory.
        "publish_model",
        "write_ple",
        # The writers of the container and goldens of the Wan2.2 TI2V DiT script
        # (`wan/ti2v_export_dit.py`).
        "write_container",
        "golden_case",
        # MOSS-Image's DiT script (`moss/export_dit.py`) writes its three containers through this
        # function, which `write_series` calls inside the staged seat (the tests call it on a tiny
        # model). The VAE script (`moss/export_vae.py`) has a `write_graphs` of its own, called the
        # same way.
        "write_graphs",
        # Gemma 4 26B's cut points and cut goldens (`gemma4/export_moe.py`), written inside the
        # staged seat of its two entries.
        "write_cut_points",
        "write_cut_goldens",
    }
)

#: (script, the function that emits). The one-shot scripts + the decode scripts that set the
#: example.
#:
#: MUST: every existing script is listed here (both directions: table → existing,
#: {@link TestTheGateItselfCanFail.test_every_listed_script_exists}, and existing → table,
#: {@link TestEveryStagingScriptIsListed}). With one direction only, **a missing entry stays silent
#: forever**.
EMIT_ENTRIES: tuple[tuple[str, str], ...] = (
    ("irodori/export.py", "export_series"),
    ("gemma4/export.py", "export_series"),
    ("gemma4/export_decode.py", "export_series"),
    ("gemma4/export_product.py", "export_series"),
    ("gemma4/export_product_qat.py", "export_series"),
    ("gemma4/export_unified.py", "export_series"),
    ("gemma4/export_moe.py", "export_series"),
    ("gemma4/export_moe.py", "export_cut"),
    ("gemma4/export_vision.py", "export_series"),
    ("gemma4_qat/export.py", "export_qat"),
    ("gemma4/export_drafter.py", "export_series"),
    ("gemma4/export_drafter_qat.py", "export_series"),
    ("minicpm5/export.py", "export_series"),
    ("minicpm5/export_decode.py", "export_series"),
    ("qwen3_5/export.py", "write"),
    ("embeddinggemma/export.py", "export_series"),
    ("deberta/export.py", "export_variant"),
    ("anima/export.py", "emit_target"),
    ("sbv2/export.py", "export_dp"),
    ("sbv2/export.py", "export_front"),
    ("sbv2/export.py", "export_flow"),
    ("sbv2/export.py", "export_dec"),
    ("sbv2/export.py", "export_voice"),
    ("siglip2/export.py", "export_series"),
    ("birefnet/export.py", "export_series"),
    ("depth_anything/export.py", "export_series"),
    ("vowel_detector/export.py", "export_series"),
    ("irodori/dacvae/export.py", "export_series"),
    ("wan/export_dit.py", "emit"),
    ("wan/export_vae.py", "emit_targets"),
    ("wan/export_vae.py", "emit_shared_targets"),
    ("wan/export_vae_encoder.py", "emit_series"),
    ("wan/ti2v_export_dit.py", "write_series"),
    ("moss/export_dit.py", "write_series"),
    ("moss/export_vae.py", "write_series"),
)


def _called_name(node: ast.AST) -> str | None:
    """The name of a call (both `f(...)` and `mod.f(...)` are taken by their last component)."""
    if not isinstance(node, ast.Call):
        return None
    target = node.func
    if isinstance(target, ast.Name):
        return target.id
    if isinstance(target, ast.Attribute):
        return target.attr
    return None


def _emit_function(source: str, name: str) -> ast.FunctionDef:
    for node in ast.parse(source).body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} is not at module level (the script's spelling moved)")


def _staged_bodies(function: ast.FunctionDef) -> list[list[ast.stmt]]:
    """The bodies of the `with` statements that open a staging directory (found inside a for loop
    too)."""
    return [
        node.body
        for node in ast.walk(function)
        if isinstance(node, ast.With)
        and any(_called_name(item.context_expr) in STAGING_CALLS for item in node.items)
    ]


def _writer_calls(nodes: list[ast.stmt]) -> set[tuple[str, int]]:
    return {
        (name, call.lineno)
        for statement in nodes
        for call in ast.walk(statement)
        if (name := _called_name(call)) in WRITER_CALLS
    }


@pytest.mark.parametrize(("script", "function"), EMIT_ENTRIES, ids=lambda value: str(value))
class TestEmitPublishesThroughAStagedSeat:
    @staticmethod
    def _function(script: str, function: str) -> ast.FunctionDef:
        return _emit_function((RECIPES_ROOT / script).read_text(encoding="utf-8"), function)

    def test_it_opens_a_staged_seat(self, script: str, function: str) -> None:
        assert _staged_bodies(self._function(script, function)), (
            f"{script}:{function} opens no staging directory — the outputs are written straight"
            " into final"
        )

    def test_every_write_happens_inside_that_seat(self, script: str, function: str) -> None:
        """Names any write left outside the staging directory by name and line number."""
        node = self._function(script, function)
        inside = set().union(*(_writer_calls(body) for body in _staged_bodies(node)))
        outside = _writer_calls(node.body) - inside

        assert not outside, (
            f"{script}:{function} writes outside the staging directory {sorted(outside)} —"
            " placing in final before the checks lets a failed run leave an asset that can pass"
            " the acceptance check"
        )


class TestTheGateItselfCanFail:
    """The check has not degenerated into "green without looking at anything" (rules out both an
    empty target table and a vacuous predicate)."""

    def test_it_names_a_writer_left_outside_the_seat(self) -> None:
        source = (
            "def export_series(out_dir):\n"
            "    export_to_file(module, args, out_dir / MODEL_FILE)\n"
            "    with staged_publication(out_dir) as staged:\n"
            "        _write_io(module, graph, cases, staged)\n"
        )
        node = _emit_function(source, "export_series")
        inside = set().union(*(_writer_calls(body) for body in _staged_bodies(node)))

        assert _writer_calls(node.body) - inside == {("export_to_file", 2)}

    def test_it_sees_no_seat_when_there_is_none(self) -> None:
        source = "def export_series(out_dir):\n    export_to_file(module, args, out_dir)\n"

        assert _staged_bodies(_emit_function(source, "export_series")) == []

    def test_every_listed_script_exists(self) -> None:
        """Fails here when a spelling in the target table moves (a file renamed or moved)."""
        for script, _function in EMIT_ENTRIES:
            assert (RECIPES_ROOT / script).is_file(), script


#: **The definition itself** of `_staged_target` (the wrapper opens `staged_publication` inside,
#: so a scan would pick it up as "a module-level function that opens a staging directory").
STAGING_WRAPPER_DEFINITIONS = frozenset({"_staged_target"})

#: The script name patterns scanned. Besides `export*.py`, the scripts of derived models of the
#: same family, `<derived>_export_<component>.py` (`wan/ti2v_export_dit.py` — the Wan2.2 TI2V
#: DiT).
SCANNED_SCRIPTS = ("*/**/export*.py", "*/**/*_export_*.py")


def _scanned_scripts() -> list[Path]:
    return sorted({path for pattern in SCANNED_SCRIPTS for path in RECIPES_ROOT.glob(pattern)})


def _staging_functions() -> set[tuple[str, str]]:
    """List the module-level functions that open a staging directory by scanning the sources
    (the existing → table side).

    The scan is limited to the script name patterns ({@link SCANNED_SCRIPTS}) so that the mock
    `with` statements of tests do not become false positives (`sbv2/tests/test_export.py` starts
    with `test_`, so it is not picked up).
    """
    found: set[tuple[str, str]] = set()
    for path in _scanned_scripts():
        if "tests" in path.parts or "__pycache__" in path.parts:
            continue
        for node in ast.parse(path.read_text(encoding="utf-8")).body:
            if not isinstance(node, ast.FunctionDef):
                continue
            if node.name in STAGING_WRAPPER_DEFINITIONS or not _staged_bodies(node):
                continue
            found.add((str(path.relative_to(RECIPES_ROOT)), node.name))
    return found


class TestEveryStagingScriptIsListed:
    """The reverse direction, existing → table (with only table → existing, a missing entry stays
    silent forever)."""

    def test_the_scan_finds_the_scripts(self) -> None:
        """If the scan finds nothing, this check is vacuous."""
        assert len(_staging_functions()) >= 19

    def test_the_scan_reaches_the_derived_model_scripts(self) -> None:
        """The `<derived>_export_<component>.py` pattern is scanned too (`export*.py` alone would
        miss it)."""
        assert ("wan/ti2v_export_dit.py", "write_series") in _staging_functions()

    def test_the_ti2v_writer_has_writes_for_the_gate_to_see(self) -> None:
        """The writes of TI2V's `write_series` (the container and the goldens) match the table of
        writer names — otherwise it would go green without looking at a single "place that
        writes"."""
        node = _emit_function(
            (RECIPES_ROOT / "wan/ti2v_export_dit.py").read_text(encoding="utf-8"), "write_series"
        )

        assert {name for name, _ in _writer_calls(node.body)} == {"write_container", "golden_case"}

    def test_no_staging_script_is_missing_from_the_table(self) -> None:
        missing = sorted(_staging_functions() - set(EMIT_ENTRIES))

        assert missing == [], (
            f"scripts that open a staging directory but have no publication-rule check: {missing}"
            " — add them to EMIT_ENTRIES (a script outside the table could go back to placing in"
            " final before the checks)"
        )
