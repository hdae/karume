"""Packages of the dependency groups are **not imported at module level** (a check of a
repository-wide convention).

The `[dependency-groups]` other than `dev` (anima / sbv2 / siglip2 / birefnet / …) **are not part
of the default `uv sync`** (the NOTE in the pyproject). Importing them at module level makes
**merely reading** that module raise `ModuleNotFoundError`, and in an environment that has not
synced the group, **pytest fails at collection as a whole**.

MUST: this check is needed because "breaking the rule is invisible in the development
environment". Everything passes on a machine that routinely syncs the groups, so the breakage is
noticed only when **someone on the default sync** runs it (this actually happened on 2026-08-22 —
`anima/single_file.py` imported `diffusers` and `huggingface_hub` at module level, and one test
became a collection error). The convention all 9 recipes followed, imports inside functions, is
turned into a mechanical check here.

The convention for users is "import inside the function that needs it" (`anima/export.py` /
`anima/patch.py` / `anima/lora.py` / `anima/pipeline_ref.py` and others are examples). A test that
touches them guards with `pytest.importorskip` (`anima/tests/test_patch.py` and others are
examples).
"""

from __future__ import annotations

import ast
import functools
import importlib.metadata
import re
import tomllib
from pathlib import Path

#: The parent of the recipes (`tools/export-recipes/`).
ROOT = Path(__file__).resolve().parent.parent

#: The dependencies that are always installed (the dependencies of the core `karume`) and the
#: groups this check leaves out. `dev` is the development tools, always installed when tests run.
ALWAYS_SYNCED_GROUPS = frozenset({"dev"})

#: The pyproject of the core (PyPI `karume`) (`[project] dependencies` = the always-synced
#: dependencies).
CORE_PYPROJECT = ROOT.parent / "exporter" / "pyproject.toml"

#: Import names that do not appear in any group's **declaration** but are not in the default sync
#: (transitive dependencies).
#:
#: `tokenizers` is installed only as a dependency of `transformers`, so a derivation that reads
#: only the group declarations misses it — writing it at module level would keep this check green
#: while the default-sync environment hits a collection error (the `importorskip` of
#: `tests/test_gemma_tokenizer.py` is the side that fills this hole by hand).
EXTRA_GATED = frozenset({"tokenizers"})


def _canonical_distribution(name: str) -> str:
    """The normalized distribution name of PEP 503 (`Pillow` / `huggingface_hub` →
    `pillow` / `huggingface-hub`), so a requirement and the installed metadata compare equal."""
    return re.sub(r"[-_.]+", "-", name).lower()


@functools.cache
def _installed_import_names() -> dict[str, frozenset[str]]:
    """Installed distribution (canonical name) → the top-level import names it provides.

    Taken from the package metadata (`importlib.metadata.packages_distributions`), not from a hand
    table — a table would be a second place to keep in step with the groups.
    """
    names: dict[str, set[str]] = {}
    for module, distributions in importlib.metadata.packages_distributions().items():
        for distribution in distributions:
            names.setdefault(_canonical_distribution(distribution), set()).add(module)
    return {distribution: frozenset(modules) for distribution, modules in names.items()}


def _import_names(requirement: str) -> frozenset[str]:
    """A requirement (a distribution name with a version) → its import names (`pillow` → `PIL`).

    An installed distribution answers from its metadata. A distribution that is not installed
    falls back to its name with `-` → `_` (`huggingface-hub` → `huggingface_hub`): in that
    environment a module-level import of it fails at collection anyway, so the gate matters where
    the group is synced — exactly where the metadata is present.
    """
    name = requirement
    for separator in ("==", ">=", "<=", "~=", "!=", ">", "<", "["):
        name = name.split(separator, 1)[0]
    name = name.strip()
    installed = _installed_import_names().get(_canonical_distribution(name))
    return installed if installed else frozenset({name.replace("-", "_")})


def _always_synced_modules() -> frozenset[str]:
    """The core's `[project] dependencies` (import names always synced, regardless of the
    groups).

    A name in the core's base dependencies is not gated even if a group declares it —
    `torchvision` is declared by the `anima` / `siglip2-preprocess` groups, but the core always
    pulls it in to register the `torchvision::deform_conv2d` op (`tools/exporter/pyproject.toml`).
    Without excluding it, this check would wrongly fail a legitimate module-level
    `import torchvision`.
    """
    core = tomllib.loads(CORE_PYPROJECT.read_text(encoding="utf-8"))
    return frozenset(
        module
        for requirement in core["project"]["dependencies"]
        for module in _import_names(requirement)
    )


def _gated_modules() -> frozenset[str]:
    """The set of import names brought in by the groups outside the default sync (derived from
    the pyproject).

    No table is kept in a second place — this check follows automatically on the day a group is
    added. What the declarations alone miss (transitive dependencies = {@link EXTRA_GATED}) is
    added, and what is always synced although declared (the core's base dependencies) is
    subtracted.
    """
    groups = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    gated: set[str] = set(EXTRA_GATED)
    for name, requirements in groups["dependency-groups"].items():
        if name in ALWAYS_SYNCED_GROUPS:
            continue
        for requirement in requirements:
            gated.update(_import_names(requirement))
    return frozenset(gated - _always_synced_modules())


def _recipe_sources() -> list[Path]:
    """The `.py` files of the recipes themselves (tests and `__pycache__` excluded).

    The files at the root (`dist.py` / `conftest.py`) are included — `conftest.py` in particular is
    always read by pytest, so a gated import there would cause exactly the collection error this
    check is meant to prevent.
    """
    return sorted(
        path
        for path in {*ROOT.glob("*.py"), *ROOT.glob("*/**/*.py")}
        if "tests" not in path.parts and "__pycache__" not in path.parts
    )


def _module_level_imports(source: Path) -> set[str]:
    """The top-level module names imported at module level (not inside a function or class)."""
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module is not None:
            names.add(node.module.split(".", 1)[0])
    return names


class TestOptionalGroupImports:
    def test_the_gated_set_is_not_empty(self) -> None:
        """If the derivation came out empty (the pyproject's shape changed), this check would be
        vacuous."""
        gated = _gated_modules()

        assert "diffusers" in gated
        assert "huggingface_hub" in gated
        assert "transformers" in gated

    def test_the_gated_set_covers_the_transitive_dependency(self) -> None:
        """A transitive dependency missing from the declarations (`tokenizers` through
        `transformers`) is gated too."""
        assert "tokenizers" in _gated_modules()

    def test_the_gated_set_holds_import_names_not_distribution_names(self) -> None:
        """`pillow` is imported as `PIL` — gating the distribution spelling would let a
        module-level `import PIL` through (pillow is installed as a torchvision dependency, so its
        metadata is present even in the default sync)."""
        gated = _gated_modules()

        assert "PIL" in gated
        assert "pillow" not in gated

    def test_a_core_dependency_is_never_gated(self) -> None:
        """`torchvision` is always pulled in by the core even though a group declares it (closes
        the false-positive side)."""
        assert "torchvision" not in _gated_modules()

    def test_it_finds_the_recipe_sources(self) -> None:
        """If the scan finds nothing, it is vacuous as well."""
        sources = _recipe_sources()

        assert len(sources) > 50
        assert any(path.name == "single_file.py" for path in sources)
        # The 2 files at the root are not missed (`conftest.py` is always read by pytest).
        assert any(path.name == "dist.py" for path in sources)
        assert any(path.name == "conftest.py" for path in sources)

    def test_no_recipe_imports_a_gated_module_at_module_level(self) -> None:
        gated = _gated_modules()
        offenders = {
            str(source.relative_to(ROOT)): sorted(_module_level_imports(source) & gated)
            for source in _recipe_sources()
            if _module_level_imports(source) & gated
        }

        assert offenders == {}, (
            "a dependency outside the default sync is imported at module level — move it into the"
            " function that needs it (in an environment without the group, merely reading it"
            f" fails): {offenders}"
        )
