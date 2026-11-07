"""The mechanical check that a container's graph name = the component name in the distribution
(= the key of weights in `karume.json`), across every family.

The runtime looks a graph up **by name** (`prepareContainer(opened, <weights key>)` —
container-v1 §2.1). The migration CLI writes under the same key (`karume.migrate._convert_unit`
passes `graph_name=unit.component`), so a writer that uses another spelling makes "the migrated
mirror" and "the re-exported series" **different things**.

Two things are guarded:

1. What is passed to `graph_name=` is **the component-name constant**, not a name derived from the
   path it is placed at. Directory names do not match component names: in the families that put
   the container directly under the series (siglip2 / birefnet / depth_anything / vowel_detector /
   gemma4 / minicpm5 / embeddinggemma / gemma4_qat) the directory name is the **series name**
   (`siglip2-so400m-patch14-384`); irodori has the keys `caption_proj` / `codec_decoder` for
   `caption-proj` / `decoder`, and deberta has the key `text_encoder` (SBV2's consuming slot) for
   `full-24layer`.
2. The set of spellings the scripts use matches that family's distribution plan (the weights of
   `<family>/distribution.py`).

This check is needed because the writer (the single export path) does not match the graph name
against the weights keys. `karume dist` ({@link karume.dist.assert_weight_components_verified})
matches the real containers before assembly, but that is reached only after an export with real
weights — this check catches the mismatched spelling at the source level.

The source is read as an AST because checking it by running would need the real weights (several
GB) or a tiny model per family (the same reason as `tests/test_staged_publication.py`). The
counterpart that writes the real thing and reads it back is the one
{@link vowel_detector.tests.test_export.TestGraphNameIsThePartName} in
`vowel_detector/tests/test_export.py`.
"""

from __future__ import annotations

import ast
import importlib
from collections.abc import Mapping
from pathlib import Path

import pytest

RECIPES_ROOT = Path(__file__).resolve().parent.parent


#: (script, emit function, the `graph_name=` expression, the module constant that holds the
#: values that expression can take).
#:
#: The expression and value columns are separate because some scripts pass it through a loop
#: variable (anima's `target`, irodori's `IRODORI_SERIES_ROLES[target]`) — the value column holds
#: only **a module-level constant name**, and {@link _declared_values} folds a `str` / sequence /
#: table alike into a set of spellings.
#:
#: MUST: paired with the existing → table direction ({@link TestEveryCallSiteIsListed}).
ENTRIES: tuple[tuple[str, str, str, str], ...] = (
    ("anima/export.py", "emit_target", "target", "TARGETS"),
    ("deberta/export.py", "export_variant", "GRAPH_NAME", "GRAPH_NAME"),
    ("embeddinggemma/export.py", "export_series", "GRAPH_NAME", "GRAPH_NAME"),
    ("gemma4/export.py", "export_series", "GEMMA4_ROLE", "GEMMA4_ROLE"),
    ("gemma4/export_decode.py", "export_series", "GEMMA4_ROLE", "GEMMA4_ROLE"),
    ("gemma4/export_drafter.py", "export_series", "GEMMA4_DRAFTER_ROLE", "GEMMA4_DRAFTER_ROLE"),
    (
        "gemma4/export_drafter_qat.py",
        "export_series",
        "GEMMA4_DRAFTER_ROLE",
        "GEMMA4_DRAFTER_ROLE",
    ),
    ("gemma4/export_product.py", "export_series", "GEMMA4_ROLE", "GEMMA4_ROLE"),
    ("gemma4/export_product_qat.py", "export_series", "GEMMA4_ROLE", "GEMMA4_ROLE"),
    ("gemma4/export_unified.py", "write_container", "GEMMA4_ROLE", "GEMMA4_ROLE"),
    ("gemma4/export_vision.py", "write_container", "GEMMA4_VISION_ROLE", "GEMMA4_VISION_ROLE"),
    ("gemma4_qat/export.py", "export_qat", "GEMMA4_ROLE", "GEMMA4_ROLE"),
    (
        "irodori/dacvae/export.py",
        "export_series",
        "IRODORI_CODEC_ROLES[target]",
        "IRODORI_CODEC_ROLES",
    ),
    ("irodori/export.py", "export_series", "IRODORI_SERIES_ROLES[target]", "IRODORI_SERIES_ROLES"),
    ("minicpm5/export.py", "export_series", "GRAPH_NAME", "GRAPH_NAME"),
    ("minicpm5/export_decode.py", "export_series", "one_shot.GRAPH_NAME", "GRAPH_NAME"),
    ("qwen3_5/export.py", "write_container", "GRAPH_NAME", "GRAPH_NAME"),
    ("sbv2/export.py", "export_dec", "TARGET_DEC", "TARGET_DEC"),
    ("sbv2/export.py", "export_dp", "TARGET_DP", "TARGET_DP"),
    ("sbv2/export.py", "export_flow", "TARGET_FLOW", "TARGET_FLOW"),
    ("sbv2/export.py", "export_front", "TARGET_FRONT", "TARGET_FRONT"),
    ("sbv2/export.py", "export_voice", "TARGET_VOICE", "TARGET_VOICE"),
    ("siglip2/export.py", "export_series", "SIGLIP2_ROLE", "SIGLIP2_ROLE"),
    ("birefnet/export.py", "export_series", "BIREFNET_ROLE", "BIREFNET_ROLE"),
    ("depth_anything/export.py", "export_series", "DEPTH_ANYTHING_ROLE", "DEPTH_ANYTHING_ROLE"),
    (
        "vowel_detector/export.py",
        "export_series",
        "VOWEL_DETECTOR_GRAPH_ROLE",
        "VOWEL_DETECTOR_GRAPH_ROLE",
    ),
    ("wan/export_dit.py", "emit", "TARGET", "TARGET"),
    ("wan/export_vae.py", "emit_targets", "target", "TARGETS"),
    ("wan/export_vae.py", "emit_shared_targets", "name", "SHARED_TARGETS"),
    ("wan/export_vae_encoder.py", "emit_series", "target", "TARGETS"),
    ("wan/umt5_export.py", "write_container", "GRAPH_NAME", "GRAPH_NAME"),
    ("wan/ti2v_export_dit.py", "write_container", "TARGET", "TARGET"),
    ("moss/export_dit.py", "write_container", "name", "GRAPHS"),
    ("moss/export_vae.py", "write_graphs", "name", "GRAPHS"),
)

#: Where to look up the value constant of a script whose constant lives in **another module**,
#: like `minicpm5/export_decode.py`.
VALUE_MODULES: Mapping[str, str] = {"minicpm5/export_decode.py": "minicpm5.export"}


def _module_name(script: str) -> str:
    """A script's relative path → its import name (`irodori/dacvae/export.py` →
    `irodori.dacvae.export`)."""
    return script.removesuffix(".py").replace("/", ".")


def _function(script: str, name: str) -> ast.FunctionDef:
    source = (RECIPES_ROOT / script).read_text(encoding="utf-8")
    for node in ast.parse(source).body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{script}: {name} is not at module level (the script's spelling moved)")


def _graph_name_arguments(node: ast.AST) -> list[ast.expr]:
    """Every expression passed to `graph_name=` within that tree."""
    return [
        keyword.value
        for call in ast.walk(node)
        if isinstance(call, ast.Call)
        for keyword in call.keywords
        if keyword.arg == "graph_name"
    ]


def _derived_from_a_path(expression: ast.expr) -> bool:
    """Whether the expression contains `<something>.name` (= a name derived from the path it is
    placed at).

    Only the attribute spelling is looked at (`Path.name` and `Path.parent.name` have the same
    shape) — a constant referenced through a module, like `one_shot.GRAPH_NAME`, has another
    attribute name and does not match.
    """
    return any(
        isinstance(node, ast.Attribute) and node.attr == "name" for node in ast.walk(expression)
    )


def _declared_values(script: str, constant: str) -> set[str]:
    """The set of spellings a script can pass to `graph_name=` (the constant is read from the
    module itself)."""
    module = importlib.import_module(VALUE_MODULES.get(script, _module_name(script)))
    value = getattr(module, constant)
    if isinstance(value, str):
        return {value}
    if isinstance(value, Mapping):
        return set(value.values())
    return set(value)


def _scanned_call_sites() -> set[tuple[str, str]]:
    """List the functions that pass `graph_name=` from **every Python source** of the recipes
    (existing → table).

    MUST: the scan is `rglob("*.py")` + {@link ast.walk} — not narrowed by name (`export*.py`),
    by location (`<family>/…`) or to module level. Narrowing it makes "the scope the check claims"
    wider than what it implements, and a writer at the repository root (`migrate_series.py`) or a
    call site in a class method passes **green without being in the table** (it measurably missed
    one).
    """
    found: set[tuple[str, str]] = set()
    for path in sorted(RECIPES_ROOT.rglob("*.py")):
        if "tests" in path.parts or "__pycache__" in path.parts or ".venv" in path.parts:
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and _graph_name_arguments(
                node
            ):
                found.add((str(path.relative_to(RECIPES_ROOT)), node.name))
    return found


#: Callers kept out of the table (internal helpers that **take `graph_name` and pass it
#: through**). The name comes from the calling emit function, so the spelling is not decided here.
FORWARDERS: frozenset[tuple[str, str]] = frozenset(
    {
        ("gemma4/export_decode.py", "_write_container"),
        ("minicpm5/export_decode.py", "_write_container"),
    }
)

#: Callers kept out of the table (the migration driver, which **looks the spelling up in the
#: family table**). {@link ENTRIES} can only express "the script names a constant", but this driver
#: runs the series directory → component name mapping, so the correctness of its spellings is
#: checked by the family-table check of `tests/test_migrate_series.py`.
TABLE_DRIVEN: frozenset[tuple[str, str]] = frozenset({("migrate_series.py", "migrate_series")})


@pytest.mark.parametrize(
    ("script", "function", "expression", "constant"), ENTRIES, ids=lambda value: str(value)
)
class TestTheEmitScriptNamesThePartName:
    def test_it_passes_the_declared_constant(
        self, script: str, function: str, expression: str, constant: str
    ) -> None:
        """The `graph_name=` expression matches the table's spelling verbatim (this fails when the
        expression moves)."""
        passed = {
            ast.unparse(argument) for argument in _graph_name_arguments(_function(script, function))
        }

        assert passed == {expression}, (
            f"{script}:{function} passes {sorted(passed)} to graph_name —"
            f" the table spells {expression}"
        )

    def test_it_does_not_derive_the_name_from_a_directory(
        self, script: str, function: str, expression: str, constant: str
    ) -> None:
        """A name derived from the path it is placed at is not the component name (the reason
        this check exists)."""
        derived = [
            ast.unparse(argument)
            for argument in _graph_name_arguments(_function(script, function))
            if _derived_from_a_path(argument)
        ]

        assert derived == [], (
            f"{script}:{function} derives graph_name from the location name {derived} —"
            " a directory name does not match the component name (the key of weights in"
            " karume.json)"
        )


class TestTheNamesAreTheWeightsKeys:
    """The set of spellings the scripts use ↔ the weights keys of the distribution plan.

    Both directions are checked: writing a spelling that is not a key makes **a container the
    runtime cannot look up**, and a surplus key means **that component has no writer** (= the
    distribution cannot be built).
    """

    @staticmethod
    def _named(family: str) -> set[str]:
        names: set[str] = set()
        for script, function, _expression, constant in ENTRIES:
            if script.split("/")[0] == family:
                assert _graph_name_arguments(_function(script, function))
                names |= _declared_values(script, constant)
        assert names, f"{family}: the table has no script at all"
        return names

    @staticmethod
    def _weights(family: str) -> set[str]:
        distribution = importlib.import_module(f"{family}.distribution")
        table = getattr(distribution, f"{family.upper()}_WEIGHTS")
        return set(table)

    @pytest.mark.parametrize("family", ["anima", "irodori", "siglip2", "depth_anything", "wan"])
    def test_the_scripts_cover_exactly_the_weights_keys(self, family: str) -> None:
        assert self._named(family) == self._weights(family)

    def test_gemma4_covers_exactly_the_weights_keys_of_each_line(self) -> None:
        """`vision` is 12B's in the q4_0 QAT line alone: the regular line's table is every script's
        name but that one, and the QAT line's 12B entry all of them."""
        from gemma4.distribution import GEMMA4_VISION_ROLE, gemma4_qat_q4_0_weights

        named = self._named("gemma4")

        assert named - {GEMMA4_VISION_ROLE} == self._weights("gemma4")
        assert named == set(gemma4_qat_q4_0_weights("12b"))

    def test_birefnet_covers_exactly_the_weights_keys_of_every_checkpoint(self) -> None:
        """BiRefNet's weights are per checkpoint (the slots follow the upstream storage —
        `birefnet_weights`), so they are not spelled as a table named after the family. They are
        checked per checkpoint, by name."""
        from birefnet.card import BIREFNET_CHECKPOINTS
        from birefnet.distribution import birefnet_weights

        for checkpoint in BIREFNET_CHECKPOINTS:
            assert self._named("birefnet") == set(birefnet_weights(checkpoint)), checkpoint

    def test_vowel_detector_covers_exactly_its_weights_keys(self) -> None:
        """`VOWEL_DETECTOR_WEIGHTS` is not spelled after the family name, so it is checked by
        name."""
        from vowel_detector.distribution import VOWEL_DETECTOR_WEIGHTS

        assert self._named("vowel_detector") == set(VOWEL_DETECTOR_WEIGHTS)

    def test_sbv2_covers_the_distributed_seats_and_nothing_else(self) -> None:
        """In SBV2, `dp` / `flow` / `dec` are only for golden verification (not in the
        distribution).

        `text_encoder` is written by another script (`deberta/export.py`), so the SBV2 scripts name
        only the remaining 2 slots.
        """
        from sbv2.distribution import SBV2_FRONT_ROLE, SBV2_VOICE_ROLE, SBV2_WEIGHTS

        named = self._named("sbv2")

        assert named & set(SBV2_WEIGHTS) == {SBV2_FRONT_ROLE, SBV2_VOICE_ROLE}
        assert named - set(SBV2_WEIGHTS) == {"dp", "flow", "dec"}

    def test_deberta_names_the_seat_its_consumer_declares(self) -> None:
        """The producer (deberta) does not import the consumer (SBV2), so the copies are matched
        against each other."""
        from sbv2.distribution import SBV2_TEXT_ENCODER_COMPONENT, SBV2_WEIGHTS

        assert self._named("deberta") == {SBV2_TEXT_ENCODER_COMPONENT}
        assert SBV2_TEXT_ENCODER_COMPONENT in SBV2_WEIGHTS

    def test_umt5_names_the_seat_both_repositories_declare(self) -> None:
        """The umT5 container sits in the same slot of 2 distributions — the umT5 repository (its
        own) and the Wan repository (a cross-repository reference — ADR 0119 addenda A / D). The
        match of the `wan` family (the parametrize above) looks only at Wan's weights, so it is
        also matched by name against the weights of the umT5 repository."""
        from wan.distribution import WAN_WEIGHTS
        from wan.umt5_distribution import UMT5_WEIGHTS

        named = _declared_values("wan/umt5_export.py", "GRAPH_NAME")

        assert named == set(UMT5_WEIGHTS)
        assert named <= set(WAN_WEIGHTS)

    def test_gemma4_qat_names_the_seat_its_own_plan_declares(self) -> None:
        """QAT has the distribution plan of a separate family, but shares the single component name
        of the plain Gemma."""
        from gemma4.distribution import GEMMA4_ROLE

        assert self._named("gemma4_qat") == {GEMMA4_ROLE}

    def test_moss_names_its_dit_and_vae_graphs(self) -> None:
        """The DiT script names the three graphs of the MOSS-Image ADR (`dit` / `dit_skip22`,
        which borrows every weight of `dit` / `dit_text` — `moss.dit_patch`), and the VAE script
        its three decoder graphs (`moss.vae_patch`). The distribution plan's seventh component,
        `text_encoder`, is written by the Qwen3.5 text recipe (`qwen3_5.export`), which does not
        import its consumer — so the union is matched against `MOSS_WEIGHTS` here."""
        from moss.distribution import MOSS_WEIGHTS

        assert self._named("moss") == {
            "dit",
            "dit_skip22",
            "dit_text",
            "vae_decoder_pre",
            "vae_decoder_attn",
            "vae_decoder_post",
        }
        assert self._named("moss") | self._named("qwen3_5") == set(MOSS_WEIGHTS)

    @pytest.mark.parametrize("family", ["embeddinggemma", "minicpm5"])
    def test_the_solo_families_name_the_single_part_convention(self, family: str) -> None:
        """Pin the spelling of the 2 families that build no distribution (they have no
        `distribution.py`).

        So that the containers already in place can be used as is on the day one is built, they
        use the convention of a single-component series (`model` — the same as gemma4 /
        gemma4_qat).
        """
        assert self._named(family) == {"model"}


class TestEveryCallSiteIsListed:
    """The reverse direction, existing → table (with only table → existing, a missing entry stays
    silent forever)."""

    def test_the_scan_finds_the_call_sites(self) -> None:
        """If the scan finds nothing, this check is vacuous."""
        assert len(_scanned_call_sites()) >= len(ENTRIES)

    def test_no_call_site_is_missing_from_the_table(self) -> None:
        listed = {(script, function) for script, function, _expression, _constant in ENTRIES}
        missing = sorted(_scanned_call_sites() - listed - FORWARDERS - TABLE_DRIVEN)

        assert missing == [], (
            f"scripts that pass graph_name but have no spelling check: {missing} —"
            " add them to ENTRIES (a script outside the table could go back to using the location"
            " name)"
        )

    def test_the_scan_reaches_a_writer_that_is_not_an_export_script(self) -> None:
        """The scan is not narrowed to `export*.py` / `<family>/…` (the miss of F-4).

        The migration driver is `migrate_series.py` at the repository root, outside the scripts'
        pattern in both name and location. Going back to a form where this fails leaves a writer
        outside the table silent forever.
        """
        assert _scanned_call_sites() >= TABLE_DRIVEN

    def test_every_table_driven_writer_really_reads_a_table(self) -> None:
        """The table-driven callers are checked for **existence** and shape too (a vanished
        spelling left in an exclusion table hides a missing entry)."""
        for script, function in sorted(TABLE_DRIVEN):
            arguments = _graph_name_arguments(_function(script, function))
            assert arguments, f"{script}:{function} does not pass graph_name"
            assert not any(_derived_from_a_path(argument) for argument in arguments), (
                f"{script}:{function} derives graph_name from the location name"
            )

    def test_every_listed_script_exists(self) -> None:
        for script, _function, _expression, _constant in ENTRIES:
            assert (RECIPES_ROOT / script).is_file(), script

    def test_every_forwarder_really_forwards(self) -> None:
        """The pass-through callers are checked for **existence** too (a vanished spelling left in
        an exclusion table hides a missing entry)."""
        for script, function in sorted(FORWARDERS):
            node = _function(script, function)
            assert [ast.unparse(a) for a in _graph_name_arguments(node)] == ["graph_name"]
            assert any(argument.arg == "graph_name" for argument in node.args.kwonlyargs)


#: The repository root (`RECIPES_ROOT` is `tools/export-recipes/`).
REPO_ROOT = RECIPES_ROOT.parent.parent

#: The source of truth of the spec (`docs/container-v1.md`).
DOCS_ROOT = REPO_ROOT / "docs"

#: The spec section that holds the text of the rule "graph name = key of weights" (the heading
#: line itself).
CONTAINER_SPEC_SECTION = "### 2.1 Graph descriptor"

#: The migration CLI contract. It holds the graph name's default (the parent directory name) and
#: refers to §2.1 for the rule.
MIGRATE_SPEC_SECTION = "## 12. Migration CLI contract"

#: Spellings that **must not be used** as the pointer for the graph-name topic.
#:
#: Decision 3 is about the `container` field (the descriptor's expected values + the parts' FileRef
#: list) and decision 8 about the whole-repository mode of the migration CLI; **neither states the
#: graph name**. Constraining only one lets the same kind of wrong pointer come back under the
#: other spelling (after decision 3 was cleared, it actually reappeared in 6 places as decision 8).
WRONG_POINTERS = (
    "ADR 0109 決定 3",
    "ADR 0109 決定 8",
    "ADR 0109 decision 3",
    "ADR 0109 decision 8",
)

#: The roots scanned for wrong pointers (`docs/` is left out because the ADRs themselves live
#: there).
POINTER_SCAN_ROOTS = ("packages", "tools")

#: The file suffixes scanned (both the TS readers and the Python writers — wrong pointers occur on
#: both sides).
POINTER_SCAN_SUFFIXES = (".py", ".ts")

#: Directory names left out of the scan (generated files, installed dependencies).
POINTER_SCAN_SKIP = frozenset({"__pycache__", ".venv", "node_modules"})


def _spec_section(document: Path, heading: str) -> str:
    """Return the text from the heading line `heading` (`## …` / `### …`) up to the next heading
    of the same or a shallower level.

    Fails if the heading is missing.
    """
    lines = document.read_text(encoding="utf-8").split("\n")
    start = next(index for index, line in enumerate(lines) if line == heading)
    level = len(heading) - len(heading.lstrip("#"))
    rest = lines[start + 1 :]
    stop = next(
        (
            index
            for index, line in enumerate(rest)
            if line.startswith("#") and 0 < len(line) - len(line.lstrip("#")) <= level
        ),
        len(rest),
    )
    return "\n".join(rest[:stop])


def _scanned_sources() -> list[Path]:
    """The sources scanned for wrong pointers (excluding this check itself)."""
    return sorted(
        path
        for root in POINTER_SCAN_ROOTS
        for suffix in POINTER_SCAN_SUFFIXES
        for path in (REPO_ROOT / root).rglob(f"*{suffix}")
        if POINTER_SCAN_SKIP.isdisjoint(path.parts) and path != Path(__file__).resolve()
    )


class TestThePointerNamesTheDocumentThatCarriesTheClaim:
    """The pointer for "graph name = key of weights" names the document that actually makes that
    claim.

    A wrong pointer **turns nothing red** (a reader just opens the ADR and runs into the
    mismatch), so it is constrained mechanically. In fact more than 25 places pointed at `ADR 0109
    decision 3`, but that decision is about the `container` field (the descriptor's expected values
    + the parts' FileRef list), not the graph name.
    """

    def test_the_container_spec_section_states_the_rule(self) -> None:
        """`container-v1 §2.1` holds "container graph name = key of `weights` MUST"."""
        section = _spec_section(DOCS_ROOT / "container-v1.md", CONTAINER_SPEC_SECTION)

        assert "**Graph-name rules**" in section
        assert (
            "**Container graph name = component name in the distribution = key of `weights` in\n"
            "  `karume.json`** MUST" in section
        )

    def test_the_migrate_spec_section_defers_to_the_rule(self) -> None:
        """`container-v1 §12` holds the CLI default (the parent directory name) and points at
        §2.1 for the rule."""
        section = _spec_section(DOCS_ROOT / "container-v1.md", MIGRATE_SPEC_SECTION)

        assert "The graph name defaults to the parent directory\n  name" in section
        assert "Per the rule of §2.1 it MUST be the key of weights" in section

    def test_the_old_pointers_do_not_state_the_rule(self) -> None:
        """Counterpart (not vacuous): the word "graph name" never appears in the section where
        the 2 wrong pointers live."""
        decision = _spec_section(
            DOCS_ROOT / "decisions" / "0109-manifest-v5-container.md", "## Decision"
        )

        assert "### 3. The `container` field" in decision
        assert "### 8. Step 2 writer — the whole-repository mode of the migration CLI" in decision
        assert "graph name" not in decision
        assert "グラフ名" not in decision

    def test_no_source_cites_an_old_pointer_for_the_naming_rule(self) -> None:
        """No line that spells the graph name / component name points at {@link WRONG_POINTERS}.

        The scan covers both the readers (TS) and the writers (Python) — wrong pointers occur not
        only in the recipes but also in the runtime / models / hub tests and in comments on the
        tools side (and they did).

        NOTE: this check's own source is left out of the scan — otherwise the text above that
        explains "why those pointers are wrong" would be caught in its own net (the check is a
        tool that looks at other files' spellings).
        """
        offenders = [
            f"{path.relative_to(REPO_ROOT)}:{number} ({pointer})"
            for path in _scanned_sources()
            for number, line in enumerate(path.read_text(encoding="utf-8").split("\n"), start=1)
            for pointer in WRONG_POINTERS
            if pointer in line
            and any(word in line for word in ("グラフ名", "部品名", "graph name", "component name"))
        ]

        assert offenders == [], (
            f"the graph name / component name points at {' / '.join(WRONG_POINTERS)}:"
            f" {offenders} — the source of truth is container-v1"
            f" §{CONTAINER_SPEC_SECTION.split()[1]}"
        )

    def test_the_scan_really_reaches_both_languages(self) -> None:
        """Counterpart (the scan has not collapsed to nothing or to one language only)."""
        scanned = _scanned_sources()
        suffixes = {path.suffix for path in scanned}
        roots = {path.relative_to(REPO_ROOT).parts[0] for path in scanned}

        assert suffixes == set(POINTER_SCAN_SUFFIXES), suffixes
        assert roots == set(POINTER_SCAN_ROOTS), roots
        assert Path(__file__).resolve() not in scanned

    def test_the_deberta_note_points_at_the_gate_that_really_runs(self) -> None:
        """The matching check that deberta's copy names **exists** (the mix-up of F-2)."""
        source = (RECIPES_ROOT / "deberta" / "export.py").read_text(encoding="utf-8")
        gate = TestTheNamesAreTheWeightsKeys.test_deberta_names_the_seat_its_consumer_declares

        assert f"tests/{Path(__file__).name}::{TestTheNamesAreTheWeightsKeys.__name__}" in source
        assert gate.__name__ in source


class TestTheGateItselfCanFail:
    """The check has not degenerated into "green without looking at anything" (rules out both a
    vacuous predicate and an empty scan)."""

    def test_it_names_a_graph_name_taken_from_a_directory(self) -> None:
        source = (
            "def export_series(out_dir):\n"
            "    export_to_file(module, args, out_dir / MODEL_FILE, graph_name=out_dir.name)\n"
        )
        node = next(iter(ast.parse(source).body))
        assert isinstance(node, ast.FunctionDef)
        arguments = _graph_name_arguments(node)

        assert [ast.unparse(a) for a in arguments] == ["out_dir.name"]
        assert _derived_from_a_path(arguments[0])

    def test_it_passes_a_plain_constant(self) -> None:
        """Counterpart (naming a constant does not fail) — without it the claim above could be
        vacuous."""
        source = "def export_series(out_dir):\n    f(path, graph_name=SIGLIP2_ROLE)\n"
        node = next(iter(ast.parse(source).body))
        assert isinstance(node, ast.FunctionDef)

        assert not _derived_from_a_path(_graph_name_arguments(node)[0])

    def test_a_staging_seat_name_is_also_caught(self) -> None:
        """A name derived from the staging directory (`<component>.staging`) is caught by the same
        predicate."""
        source = "def export_series(staged):\n    f(path, graph_name=staged.parent.name)\n"
        node = next(iter(ast.parse(source).body))
        assert isinstance(node, ast.FunctionDef)

        assert _derived_from_a_path(_graph_name_arguments(node)[0])
