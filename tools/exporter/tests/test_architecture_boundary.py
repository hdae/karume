"""The check that mechanically stops core → recipe backflow (ADR 0065 decision 3).

The exporter has been split step by step into "the generic exporter core (the PyPI distribution)"
and "per-model recipes (repository only)", and only **one dependency direction, recipe → core**,
is allowed. Writing it down as a convention would let backflow creep in with every step, so the
imports are checked mechanically here (same ADR: tests guard the boundary, not convention).

Two checks:

1. core does not import **recipe-side modules** — both the recipes' current namespace (top-level
   packages such as `wan` / `_shared`, top-level modules such as `dist` / `migrate_series`, and
   shared test helpers directly under `tests/` such as `container_series` / `gemma_synthetic`;
   derived from the layout of `tools/export-recipes/`) and the old names `karume.patch_*` /
   `karume.paths` (none of either exists today, but the check stays to prevent a regression)
2. core does not import **upstream model packages** (`style_bert_vits2` / `transformers` /
   `diffusers` / `torchvision`) — the check that keeps the provenance obligations of upstream
   implementations out of the core wheel (ADR 0065 decision 7)

The scan uses `ast`, not a string grep — so that lazy imports inside functions (shaped like
`from karume import dist` in `karume.cli`) do not become an escape hatch of the staged migration.
`ast` sees only static import statements (`importlib.import_module` and `__import__` are out of
scope).
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

#: Where the sources of the checked package live (`src/` layout — ADR 0065 decision 5).
KARUME_ROOT = Path(__file__).resolve().parents[1] / "src" / "karume"

#: The set kept as the generic exporter core (the list of ADR 0065 decision 1).
#:
#: MUST: **kept as an explicit list**. Deciding the targets by sweeping `src/karume/` would let a
#: new module silently enter the check (or stay out of it) without the decision "core or recipe".
#: NOTE: `dist` / `modelcard` / `cli` entered the check at stage 5 as well — after the family
#: knowledge (stages 3+4), the repo topology (`paths`) moved out too, so the whole content of the
#: core wheel is checked through this one list.
CORE_MODULES: tuple[str, ...] = (
    "__init__",
    "ir",
    "ops",
    "dims",
    "extents",
    "shapes",
    "convert",
    "aten_handlers",
    "normalize",
    "quantize",
    # Measurement-only rounding methods (no storage path — quant_methods module docstring).
    # Target selection is shared with quantize; no family knowledge.
    "quant_methods",
    # Calibrated rounding (GPTQ / AWQ). Target selection shared with quantize; no family
    # knowledge.
    "quant_calib",
    "act_quant",
    "emit",
    # Derivation of a quant's `requiredLimits` (ADR 0038 §7 addendum / ADR 0089 decision 3).
    # Knows only sizes and the WebGPU defaults.
    "limits",
    "verify",
    # The writer of the container format (krm / krg) and the reader for self-checks (ADR 0108).
    # Knows only bytes, declarations and the codec ledger.
    "container",
    # The 3 steps that place a distribution (write → read back and verify → swap). Knows only
    # paths and bytes.
    "publish",
    # Pure functions that build the PLE into container assets (ADR 0085 / 0109 decision 4). Only
    # the row byte count and the block limit.
    "ple",
    # The **read-only** layer for the old distribution (a safetensors dialect) (container-v1 §12).
    # Only for the migration CLI; it knows only bytes and declarations.
    "legacy",
    # The migration CLI old distribution → container format (container-v1 §12). The only place
    # that reads the old format; it knows declarations and raw bytes (no repo topology). Its only
    # family knowledge is the table of old PLE sidecar owners `PLE_OWNER` (pipeline name →
    # component name), an exception limited to the one-time migration of the old format (ADR 0065
    # addendum).
    "migrate",
    "pipeline",
    "goldens",
    "golden_models",
    "custom_ops",
    # Model-independent export verification helpers recovered from the Anima patch layer at
    # stage 2.
    "rope",
    # IR surgery into the states form (ADR 0067). Used by decode scripts, but holds no model
    # knowledge.
    "states",
    # The primitive of artifact publication (ADR 0052's staging → verify → swap). Knows only
    # paths.
    "artifacts",
    # The assembly engine, card rendering and CLI purified at stage 5 (neither family knowledge
    # nor repo topology).
    "dist",
    "modelcard",
    "cli",
)

#: Prefixes of recipe-side module names (core must not import them).
#:
#: NOTE: **not a single** `karume/patch_*.py` remains (at ADR 0065 stages 3+4 every family moved
#: to `tools/export-recipes/<family>/patch.py`). It stays on the forbidden list **to prevent a
#: regression** — this spelling is the shortest way to write a patch layer derived from an
#: upstream model back into core, and removing it would remove the check that stops "a
#: `karume/patch_foo.py` turned up unnoticed". The synthetic fault injection of
#: {@link TestTheBoundaryCheckItself} shows it is not tautological.
RECIPE_MODULE_PREFIXES = ("patch_",)

#: Recipe-side module names (core must not import them).
#:
#: NOTE: `karume/paths.py` is **gone too** (it moved to `tools/export-recipes/_shared/` at ADR
#: 0065 stage 5). It stays here as **a roster against regression**, like `patch_*` — this
#: spelling is the shortest way to write the repo topology (the repository's `models/` /
#: `outputs/` spellings) back into the core wheel, and removing it would remove the check that
#: stops "a `karume/paths.py` turned up unnoticed". The synthetic fault injection of
#: {@link TestTheBoundaryCheckItself} shows it is not tautological.
#: Conversely, `dist` / `modelcard` / `cli` left this list — they became part of core with
#: neither family knowledge nor repo topology, so they are **on the checked side, not the
#: forbidden side** ({@link CORE_MODULES}) (only the default distribution owner
#: `modelcard.HF_OWNER` remains — ADR 0065 addendum).
RECIPE_MODULES = frozenset({"paths"})

#: Where the recipes live (ADR 0065 decision 2 — outside the wheel, repository only).
RECIPES_ROOT = Path(__file__).resolve().parents[2] / "export-recipes"

#: Where the recipes' shared test helpers live (relative to the recipe root). The recipes'
#: conftest puts this directory on sys.path, so its contents are imported under top-level names
#: such as `container_series`.
RECIPE_TEST_HELPERS_DIR = "tests"

#: Entries directly under the recipe root and under {@link RECIPE_TEST_HELPERS_DIR} that are not
#: part of the recipe namespace — the test directory itself (`tests` — its contents go on
#: sys.path; nothing is imported under the name `tests`) and the pytest configuration
#: (`conftest.py` — read by pytest, not imported by name).
NON_RECIPE_NAMES = frozenset({"conftest", "tests"})

#: The prefix of the test modules pytest collects. They are collected, not imported by name from
#: elsewhere.
TEST_MODULE_PREFIX = "test_"


def recipe_packages(root: Path) -> frozenset[str]:
    """Derive the recipes' top-level names (packages and modules) from the entries directly under
    the recipe root.

    Recipes are imported not as `karume.*` but as top-level packages such as `wan` / `_shared`
    (`python -m <family>.<mod>`), so a roster of old names alone cannot stop backflow under the
    current spellings. A hand-written roster would open a hole every time a family is added and the
    update is forgotten, so it is derived from the layout. Of the names importable as identifiers,
    three kinds are taken:

    - directories with a `.py` directly inside — `__init__.py` is not required because a family
      placed as a namespace package imports too
    - `.py` modules directly under the root (`dist` / `migrate_series` and the like) — launch
      scripts, but the recipe-side tests import them under top-level names as `import dist` /
      `from dist import main`. Core writing the same spelling would be a dependency on the recipes
    - `.py` modules directly under `tests/` that are not test modules (`test_*`)
      (`container_series` / `gemma_synthetic` and the like) — the recipe-side tests import them
      under top-level names as `from container_series import write_component`. Same reason as the
      top-level modules

    MUST: fail loudly in an environment without the recipe root. Silently returning an empty set
    would leave this branch's check green and tautological (tests guard the boundary — ADR 0065
    decision 3). A missing `tests/` is allowed — with no shared helper there is no spelling that
    imports one either.
    """
    if not root.is_dir():
        raise FileNotFoundError(
            f"recipe root is missing: {root} (check 1 cannot run outside the repository)"
        )
    packages = (path.name for path in root.iterdir() if path.is_dir() and any(path.glob("*.py")))
    modules = (path.stem for path in root.glob("*.py"))
    test_helpers = (
        path.stem
        for path in (root / RECIPE_TEST_HELPERS_DIR).glob("*.py")
        if not path.stem.startswith(TEST_MODULE_PREFIX)
    )
    return frozenset(
        name
        for name in (*packages, *modules, *test_helpers)
        if name.isidentifier() and name not in NON_RECIPE_NAMES
    )


#: Upstream model packages kept out of the core wheel.
UPSTREAM_MODEL_PACKAGES = frozenset(
    {"diffusers", "style_bert_vits2", "torchvision", "transformers"}
)

#: Temporary exemptions from check 2 — a mapping **module → allowed upstream packages**.
#:
#: MUST: no whole-module exemption. The exemption is justified for one package only, so exempting
#: a set would mean "writing `transformers` in the 2 exempted modules makes the check report
#: nothing", opening the shortest way for the provenance obligations of upstream model
#: implementations into the core wheel.
#:
#: NOTE: `aten_handlers` imports torchvision plainly as a base dependency to look up the handler
#: key `torch.ops.torchvision.*` (deform_conv2d) (ADR 0055 decision 7). `golden_models` calls
#: `torchvision.ops.deform_conv2d` for the torch comparison of the same op. Neither is "a copy of
#: an upstream model implementation", and torchvision is a base dependency of the core wheel by
#: ADR 0055 decision 7 (ADR 0065 stage 6, packaging / provenance, is complete).
UPSTREAM_CHECK_EXEMPT: dict[str, frozenset[str]] = {
    "aten_handlers": frozenset({"torchvision"}),
    "golden_models": frozenset({"torchvision"}),
}

#: Temporary exemptions from check 1. **Empty** — every core module goes through check 1 (ADR
#: 0065 stage 5 complete).
#:
#: NOTE: its shape differs from check 2's {@link UPSTREAM_CHECK_EXEMPT} (a mapping holding "the
#: allowed packages") because check 1 has no unit of "allow only part" — the forbidden side is the
#: roster of `karume.<name>` itself, and a partial exemption "allows importing that module name" =
#: the same as a full exemption. An exemption is added only "to keep a known violation explicitly",
#: never as an escape hatch for getting green.
RECIPE_CHECK_EXEMPT: frozenset[str] = frozenset()

RECIPE_GATED_MODULES = tuple(name for name in CORE_MODULES if name not in RECIPE_CHECK_EXEMPT)


def core_module(module_name: str) -> Path:
    """core module name → the location of its source.

    The checks go **through paths** so that the fault injection below can run synthetic sources
    through **the same scanning code** (entrusting the subject to an existing module would make the
    check silently tautological the moment that family moved to the recipe side —
    {@link TestTheBoundaryCheckItself}).
    """
    return KARUME_ROOT / f"{module_name}.py"


def imported_modules(path: Path) -> list[tuple[int, str]]:
    """Collect every fully qualified module name that the source at `path` imports, with line
    numbers.

    `ast.walk` also picks up lazy imports inside functions and conditionals. The form `from karume
    import dist` returns both "`karume`" and "`karume.dist`" (the latter is the actual dependency;
    whether it is a symbol or a submodule cannot be told from the AST — a clash with a forbidden
    name is treated as a violation).
    """
    label = _label(path)
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=label)
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.extend((node.lineno, alias.name) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = _resolve_import_from_base(node, label)
            if base:
                found.append((node.lineno, base))
            found.extend(
                (node.lineno, f"{base}.{alias.name}" if base else alias.name)
                for alias in node.names
            )
    return found


def _label(path: Path) -> str:
    """The heading of a violation report. A scanned file is read as a module directly under
    `karume` (the origin of relative imports)."""
    return f"karume/{path.name}"


def _resolve_import_from_base(node: ast.ImportFrom, label: str) -> str:
    """Turn the base of `from ... import` into a fully qualified name (relative imports become
    absolute)."""
    if node.level == 0:
        return node.module or ""
    # Every core module is directly under `karume`, so the base of a one-level relative import is
    # `karume`. Two levels or more leave the package and never appear in core (fail loudly if they
    # do).
    if node.level > 1:
        raise ValueError(f"{label}:{node.lineno}: relative import out of the package")
    return f"karume.{node.module}" if node.module else "karume"


def recipe_imports(path: Path) -> list[str]:
    """Return the imports of recipe-side modules as a list of "line: import name" (empty if none).

    The forbidden top-level names are derived from the layout only here — deriving them at module
    load time would, in an environment without the recipe root, drag checks unrelated to check 1
    (upstream packages, the migration path) down with a collection error. The scope of the fail
    loudly is kept to check 1.
    """
    recipe_names = recipe_packages(RECIPES_ROOT)
    violations = []
    for lineno, dotted in imported_modules(path):
        parts = dotted.split(".")
        # Decide by the top-level name (`parts[0]`) — `karume.dist`, which shares its name with the
        # recipes' `dist.py`, is not a false positive.
        if parts[0] in recipe_names:
            violations.append(f"{_label(path)}:{lineno} -> {dotted}")
            continue
        if parts[0] != "karume" or len(parts) < 2:
            continue
        submodule = parts[1]
        if submodule.startswith(RECIPE_MODULE_PREFIXES) or submodule in RECIPE_MODULES:
            violations.append(f"{_label(path)}:{lineno} -> {dotted}")
    return violations


def upstream_imports(path: Path, allowed: frozenset[str] = frozenset()) -> list[str]:
    """Return the imports of upstream model packages as a list of "line: import name" (empty if
    none).

    `allowed` are the packages allowed for that module only ({@link UPSTREAM_CHECK_EXEMPT}). The
    default is empty, so for a module not in the exemption table every package is a violation.
    """
    gated = UPSTREAM_MODEL_PACKAGES - allowed
    return [
        f"{_label(path)}:{lineno} -> {dotted}"
        for lineno, dotted in imported_modules(path)
        if dotted.split(".")[0] in gated
    ]


class TestTheModuleRoster:
    """The roster ({@link CORE_MODULES}) names every module on disk.

    The roster stays hand-written (adding a new module MUST NOT skip the decision "core or
    recipe"), and this checks **only that they match**. A new module left off the roster would drop
    out of check 2's parametrize entirely and become a place that silently lets upstream model
    imports through.
    """

    def test_the_roster_names_every_module_on_disk(self) -> None:
        on_disk = {path.stem for path in KARUME_ROOT.glob("*.py")}
        roster = set(CORE_MODULES)

        assert on_disk == roster, (
            f"not in the roster: {sorted(on_disk - roster)}"
            f" / only in the roster: {sorted(roster - on_disk)}"
        )

    def test_the_package_has_no_subpackage_outside_the_scan(self) -> None:
        """The scan covers only the top level of `src/karume` — this reports the hole on the day a
        subpackage appears.

        As long as `core_module` returns `KARUME_ROOT / f"{name}.py"`, a `karume/xxx/yyy.py` is in
        neither the roster nor the checks (`__pycache__` is a build artifact and is excluded).
        """
        subdirectories = sorted(
            path.name
            for path in KARUME_ROOT.iterdir()
            if path.is_dir() and path.name != "__pycache__"
        )

        assert subdirectories == []


class TestThePublicSurface:
    """The package's public surface (`karume.__all__`) exists, and the single path to a
    distribution can be followed.

    The source of truth of the public surface is `__all__` alone (the list in the docstring
    explains it) — if it split in two, a writer who read only the docstring would spell out the 3
    steps "write → verify → swap" by hand.
    """

    def test_every_exported_name_resolves(self) -> None:
        import karume

        missing = [name for name in karume.__all__ if not hasattr(karume, name)]

        assert missing == []

    def test_the_one_way_to_publish_is_exported(self) -> None:
        """`publish_model` is on the public surface (the one path that makes a distribution — the
        MUST of pipeline)."""
        import karume

        assert "publish_model" in karume.__all__

    def test_the_export_list_is_sorted(self) -> None:
        """Adding an unsorted name turns it red (not a claim about the order, but a detector that
        lets the machine decide where an addition goes)."""
        import karume

        assert list(karume.__all__) == sorted(karume.__all__)


class TestCoreDoesNotImportRecipeModules:
    @pytest.mark.parametrize("module_name", RECIPE_GATED_MODULES)
    def test_a_core_module_imports_no_recipe_module(self, module_name: str) -> None:
        """core does not depend on `patch_*` or modules with family knowledge baked in (dependencies
        go recipe → core).

        Backflow would put family knowledge inside the wheel's physical boundary (the whole
        `karume/` directory).
        """
        assert recipe_imports(core_module(module_name)) == []


class TestCoreDoesNotImportUpstreamModelPackages:
    @pytest.mark.parametrize("module_name", CORE_MODULES)
    def test_a_core_module_imports_no_upstream_model_package(self, module_name: str) -> None:
        """The core wheel does not pull in upstream model implementations (or their provenance
        obligations).

        The 2 exempted modules **are checked too** (only torchvision is allowed) — dropping them
        entirely through an exclusion list would leave those 2 open as an entrance for upstream
        model implementations.
        """
        allowed = UPSTREAM_CHECK_EXEMPT.get(module_name, frozenset())

        assert upstream_imports(core_module(module_name), allowed) == []


class TestTheBoundaryCheckItself:
    """Fault injection showing that the checks really detect violations (the check against
    tautology).

    It shows with **synthetic sources** that the core set is green "because there is no violation",
    not "because the check sees nothing". The subject is not entrusted to an existing violating
    module because ADR 0065 stage 4 broke that arrangement in two consecutive batches — every time
    the targeted family moved to the recipe side the subject withered, and when the last one moved
    **no replacement was left** (with the final batch, no `karume/patch_*.py` remained). This closes
    the way the check could silently become tautological.

    The scan is the same {@link imported_modules} as the real one (a path not a byte different from
    the one applied to the core modules of `karume/`) — **only the subject** is synthetic; the
    check itself is shared.
    """

    @staticmethod
    def _violator(tmp_path: Path, name: str, source: str) -> Path:
        """Place a synthetic source meant as `karume/<name>.py` (the scan looks only at the
        path)."""
        path = tmp_path / f"{name}.py"
        path.write_text(source, encoding="utf-8")
        return path

    def test_it_catches_a_lazy_upstream_import(self, tmp_path: Path) -> None:
        """An upstream import hidden inside a function is seen too (no escape hatch for the staged
        migration)."""
        path = self._violator(
            tmp_path,
            "sneaky",
            "def build():\n    from transformers import AutoModel\n\n    return AutoModel\n",
        )

        assert upstream_imports(path) == [
            "karume/sneaky.py:2 -> transformers",
            "karume/sneaky.py:2 -> transformers.AutoModel",
        ]

    def test_an_exemption_covers_only_the_package_it_names(self, tmp_path: Path) -> None:
        """An exemption is "that module × that package" — even in an exempted module, any other
        upstream package is a violation.

        Turning the exemption back into a per-module set would let this pass while staying green
        (= opening the shortest way to bring upstream model implementations into the core wheel).
        """
        path = self._violator(
            tmp_path,
            "exempted",
            "import torchvision\n\nfrom transformers import AutoModel\n",
        )

        assert upstream_imports(path, frozenset({"torchvision"})) == [
            "karume/exempted.py:3 -> transformers",
            "karume/exempted.py:3 -> transformers.AutoModel",
        ]

    def test_it_stays_quiet_on_a_module_that_imports_nothing_forbidden(
        self, tmp_path: Path
    ) -> None:
        """The counterpart showing that the detector does not "react to anything" (the flip side of
        the claims above)."""
        path = self._violator(tmp_path, "clean", "import json\n\nfrom karume.ir import IrGraph\n")

        assert upstream_imports(path) == []
        assert recipe_imports(path) == []

    def test_it_catches_a_lazy_recipe_import(self, tmp_path: Path) -> None:
        """The form `from karume import paths` is seen too (symbol and submodule names are not told
        apart)."""
        path = self._violator(
            tmp_path, "leaky", "def run(argv):\n    from karume import paths\n\n    return paths\n"
        )

        assert recipe_imports(path) == ["karume/leaky.py:2 -> karume.paths"]

    def test_it_catches_a_relative_recipe_import(self, tmp_path: Path) -> None:
        """A one-level relative import is made absolute too (`from . import paths` does not slip
        past the check)."""
        path = self._violator(tmp_path, "relative", "from . import paths\n")

        assert recipe_imports(path) == ["karume/relative.py:1 -> karume.paths"]

    def test_it_catches_a_patch_module_by_prefix(self, tmp_path: Path) -> None:
        """The prefix check ({@link RECIPE_MODULE_PREFIXES}) works independently of the table of
        names.

        No `karume/patch_*.py` remains, so no existing module exercises this branch any more (=
        a place where only the roster stays and the check could become tautological). **The
        regression check keeps being exercised here** — this spelling is the shortest way for a
        patch layer derived from an upstream model to be written back into core.
        """
        path = self._violator(
            tmp_path, "regressed", "from karume.patch_newmodel import apply_all_patches\n"
        )

        assert recipe_imports(path) == [
            "karume/regressed.py:1 -> karume.patch_newmodel",
            "karume/regressed.py:1 -> karume.patch_newmodel.apply_all_patches",
        ]

    def test_it_catches_a_lazy_import_of_a_recipe_family(self, tmp_path: Path) -> None:
        """A lazy import of the recipes' current namespace (a top-level family package) is seen too.

        An import inside a function does not raise ImportError in the core-only pytest run, and
        breaks only when a wheel user calls that function — unless stopped here, the boundary
        violation reaches the distribution.
        """
        path = self._violator(tmp_path, "familiar", "def run():\n    from wan import umt5_patch\n")

        assert recipe_imports(path) == [
            "karume/familiar.py:2 -> wan",
            "karume/familiar.py:2 -> wan.umt5_patch",
        ]

    def test_it_catches_a_lazy_import_of_the_shared_recipe_layer(self, tmp_path: Path) -> None:
        """A lazy import of `_shared` (the shared recipe layer that holds the repo topology) is a
        violation too."""
        path = self._violator(
            tmp_path, "topology", "def run():\n    from _shared.paths import REPO_ROOT\n"
        )

        assert recipe_imports(path) == [
            "karume/topology.py:2 -> _shared.paths",
            "karume/topology.py:2 -> _shared.paths.REPO_ROOT",
        ]

    def test_it_catches_a_lazy_import_of_a_top_level_recipe_module(self, tmp_path: Path) -> None:
        """Importing a top-level script (`dist.py`) under its top-level name is a violation too."""
        path = self._violator(tmp_path, "launcher", "def run():\n    import dist\n")

        assert recipe_imports(path) == ["karume/launcher.py:2 -> dist"]

    def test_it_catches_a_lazy_import_of_a_shared_recipe_test_helper(self, tmp_path: Path) -> None:
        """Importing a shared recipe-side test helper (`tests/container_series.py`) under its
        top-level name is a violation too."""
        path = self._violator(
            tmp_path, "helper", "def run():\n    from container_series import write_component\n"
        )

        assert recipe_imports(path) == [
            "karume/helper.py:2 -> container_series",
            "karume/helper.py:2 -> container_series.write_component",
        ]

    def test_it_stays_quiet_on_a_core_module_sharing_a_recipe_script_name(
        self, tmp_path: Path
    ) -> None:
        """The decision is by top-level name — `karume.dist`, which shares its name with the
        recipes' `dist.py`, is core."""
        path = self._violator(tmp_path, "assembler", "from karume.dist import PIPELINES\n")

        assert recipe_imports(path) == []


class TestTheRecipeNamespaceIsDerivedFromDisk:
    """The forbidden top-level names are derived from the layout directly under
    `tools/export-recipes/`.

    A hand-written roster would open a hole every time a family is added and the update is
    forgotten. If the derivation misread the layout and came back empty, the new branch of check 1
    would silently become tautological.
    """

    def test_it_includes_the_recipe_families_and_the_shared_layer(self) -> None:
        assert {"wan", "_shared"} <= recipe_packages(RECIPES_ROOT)

    def test_it_includes_the_top_level_recipe_modules(self) -> None:
        """Top-level scripts are on the forbidden side too — the recipe-side tests use them spelled
        `import dist`."""
        assert {"dist", "migrate_series"} <= recipe_packages(RECIPES_ROOT)

    def test_it_includes_the_shared_recipe_test_helpers(self) -> None:
        """Shared helpers directly under `tests/` are on the forbidden side too — the recipe-side
        tests use them spelled `from container_series import write_component`."""
        assert {"container_series", "gemma_synthetic"} <= recipe_packages(RECIPES_ROOT)

    def test_it_excludes_the_entries_that_are_not_recipe_packages(self) -> None:
        """The deliberately excluded names (the recipe-side tests and the pytest configuration) are
        not in the forbidden set."""
        assert recipe_packages(RECIPES_ROOT).isdisjoint({"tests", "conftest"})

    def test_it_takes_only_the_helpers_from_the_test_directory(self, tmp_path: Path) -> None:
        """From directly under `tests/` only the shared helpers are taken, not the test modules or
        the pytest configuration.

        The real `tests/` has no `conftest.py` today, so both exclusions are exercised in a
        synthetic recipe root.
        """
        helpers = tmp_path / "export-recipes" / "tests"
        helpers.mkdir(parents=True)
        for name in ("shared_helper", "test_something", "conftest"):
            (helpers / f"{name}.py").write_text("")

        assert recipe_packages(tmp_path / "export-recipes") == {"shared_helper"}

    def test_it_fails_loudly_without_the_recipe_root(self, tmp_path: Path) -> None:
        """Returning an empty set without the recipe root would silently make the new branch of
        check 1 tautological."""
        with pytest.raises(FileNotFoundError):
            recipe_packages(tmp_path / "export-recipes")


#: The entry of the core modules that the read-only path for the old distribution (the migration
#: CLI) may reach.
MIGRATION_ENTRY = "migrate"

#: Heavy dependencies kept out of the migration CLI's import graph.
#:
#: MUST: no `torch`. `karume migrate` reads only the old shards' bytes and declarations and never
#: goes through the storage conversion (which depends on torch) — placing the binding-table mapping
#: (`container_bindings`) in a torch-side module would make the migration's import graph pull in a
#: dependency of about 1 GB wholesale (the same reason `dist` deferred its import of
#: `karume.verify` into a function).
MIGRATION_FORBIDDEN = frozenset({"torch"})


def karume_closure(entry: str) -> dict[str, list[str]]:
    """The core modules reachable from `karume.<entry>` → their imports (fully qualified names).

    The scan is **the same {@link imported_modules}** as checks 1 / 2 (`ast`, so lazy imports
    inside functions are picked up too). Only modules directly under `karume` are followed;
    external packages are recorded as leaves.
    """
    reached: dict[str, list[str]] = {}
    pending = [entry]
    while pending:
        name = pending.pop()
        if name in reached:
            continue
        dotted = [item for _, item in imported_modules(core_module(name))]
        reached[name] = dotted
        for item in dotted:
            parts = item.split(".")
            if parts[0] == "karume" and len(parts) >= 2 and parts[1] in CORE_MODULES:
                pending.append(parts[1])
    return reached


class TestTheMigrationPathStaysIndependentOfTorch:
    """torch does not appear in the migration CLI's import graph (prevents F-12 from recurring).

    NOTE: the run-time check (`sys.modules` after `import karume.migrate`) is held by
    `TORCH_FREE_MODULES` in `tests/test_package_init.py` — the package's `__init__` resolves the
    public surface on first access (PEP 562), so importing a submodule alone does not load torch.
    This one checks **the dependency direction of the modules** with `ast` (lazy imports inside
    functions included). The two fill each other's holes: `ast` does not see dynamic imports or
    torch reached through external packages, and the run-time check does not see imports inside
    functions that were not called.
    """

    def test_no_module_reachable_from_migrate_imports_torch(self) -> None:
        closure = karume_closure(MIGRATION_ENTRY)
        offenders = sorted(
            f"karume/{name}.py -> {item}"
            for name, imports in closure.items()
            for item in imports
            if item.split(".")[0] in MIGRATION_FORBIDDEN
        )

        assert offenders == []

    def test_the_closure_really_walks_the_graph(self) -> None:
        """The check against tautology — not only the entry but also what is reached from it is in
        the roster."""
        closure = karume_closure(MIGRATION_ENTRY)

        assert MIGRATION_ENTRY in closure
        # The migration always goes through the container, the old-format reader, the 3
        # publication steps and the merge.
        assert {"container", "legacy", "publish", "verify", "ple", "dist"} <= set(closure)

    def test_the_closure_catches_a_module_that_reaches_torch(self) -> None:
        """The detector really finds torch (starting from `emit` it always shows up)."""
        closure = karume_closure("emit")
        offenders = [
            item
            for imports in closure.values()
            for item in imports
            if item.split(".")[0] in MIGRATION_FORBIDDEN
        ]

        assert offenders
