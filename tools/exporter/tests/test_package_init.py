"""Importing `karume` loads no torch, and the public surface resolves as the eager import did.

`karume/__init__` resolves the public surface on first access (PEP 562). Every claim depends on
"which modules are already imported", so each is checked in a fresh interpreter rather than with
the test session's `sys.modules`.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

#: Public name → defining module (a copy of the `from <module> import <name>` of the eager-import
#: days). It specifies the public name set and where each name resolves; it is not a copy of
#: `karume._EXPORTS`.
EXPECTED_ORIGINS: dict[str, str] = {
    "FixedQuantizedWeight": "karume.emit",
    "IrGraph": "karume.ir",
    "Provenance": "karume.container",
    "convert": "karume.convert",
    "curated_decompositions": "karume.convert",
    "export_module": "karume.pipeline",
    "export_to_file": "karume.pipeline",
    "normalize_graph": "karume.normalize",
    "parse_ir_graph": "karume.verify",
    "publish_model": "karume.pipeline",
    "stored_model": "karume.emit",
    "verify_container": "karume.verify",
}

#: Modules that do not need torch. The distribution / card layer (the recipes' dist driver) is built
#: from this chain alone, and the migration CLI (`karume migrate`) from this chain plus the old-form
#: reader, the three publish steps and PLE / limits. The CLI entry point also defers its dispatch,
#: so importing it does not load torch.
TORCH_FREE_MODULES: tuple[str, ...] = (
    "karume",
    "karume.dist",
    "karume.modelcard",
    "karume.container",
    "karume.verify",
    "karume.migrate",
    "karume.publish",
    "karume.legacy",
    "karume.ple",
    "karume.limits",
    "karume.artifacts",
    "karume.cli",
)

#: Public names whose defining module needs no torch (the ir / container / verify part of
#: `EXPECTED_ORIGINS`).
TORCH_FREE_EXPORTS: tuple[str, ...] = tuple(
    name
    for name, module_name in EXPECTED_ORIGINS.items()
    if module_name in {"karume.ir", "karume.container", "karume.verify"}
)


def run_python(source: str) -> str:
    completed = subprocess.run(
        [sys.executable, "-c", source], capture_output=True, text=True, check=False
    )
    assert completed.returncode == 0, completed.stderr
    return completed.stdout


class TestImportingTorchFreeModules:
    @pytest.mark.parametrize("module_name", TORCH_FREE_MODULES)
    def test_torch_is_not_loaded(self, module_name: str) -> None:
        loaded = run_python(
            f"import {module_name}, sys\n"
            "print(sorted(m for m in ('torch', 'torchvision') if m in sys.modules))"
        )

        assert loaded.strip() == "[]"

    @pytest.mark.parametrize("name", TORCH_FREE_EXPORTS)
    def test_touching_a_torch_free_export_does_not_load_torch(self, name: str) -> None:
        """Resolved through a torch-dependent module, even the same object loads torch on access.

        Turns red when a row of `_EXPORTS` stops pointing at the defining module (the identity of
        the resolved object alone cannot tell the difference).
        """
        loaded = run_python(f"import karume, sys\nkarume.{name}\nprint('torch' in sys.modules)")

        assert loaded.strip() == "False"

    def test_touching_an_export_that_needs_torch_loads_it(self) -> None:
        """Deferral means "load when needed", not "never load" — the counterpart showing the check
        above is not tautological."""
        loaded = run_python("import karume, sys\nkarume.convert\nprint('torch' in sys.modules)")

        assert loaded.strip() == "True"


#: Script that returns, per name, "is the public value identical to the same-named attribute of
#: the defining module" as JSON. `{prelude}` swaps the modules imported first, changing the
#: resolution order. `dir` is taken before resolution (after it every name sits in the module
#: dictionary and cannot be told apart).
RESOLUTION_PROBE = """
import importlib, json
{prelude}
import karume
origins = json.loads({origins!r})
report = {{"dir": sorted(set(karume.__all__) - set(dir(karume)))}}
for name, module_name in origins.items():
    report[name] = getattr(karume, name) is getattr(importlib.import_module(module_name), name)
from karume import convert
report["from karume import convert"] = convert is importlib.import_module("karume.convert").convert
report["__all__"] = sorted(karume.__all__)
print(json.dumps(report))
"""


class TestThePublicSurfaceResolvesLikeTheEagerImport:
    @pytest.mark.parametrize(
        "prelude",
        [
            pytest.param("", id="package-first"),
            # `karume.convert` is a submodule with the same name as the public function `convert`.
            # When it is loaded first, the import system sets a same-named attribute on the
            # parent, so this checks that the public name does not turn into the module.
            pytest.param(
                "import " + ", ".join(sorted(set(EXPECTED_ORIGINS.values()))),
                id="submodules-first",
            ),
        ],
    )
    def test_every_name_is_the_object_its_origin_module_defines(self, prelude: str) -> None:
        report = json.loads(
            run_python(
                RESOLUTION_PROBE.format(prelude=prelude, origins=json.dumps(EXPECTED_ORIGINS))
            )
        )

        assert report.pop("__all__") == sorted(EXPECTED_ORIGINS)
        assert report.pop("dir") == []
        mismatched = [name for name, same in report.items() if not same]
        assert mismatched == []

    def test_a_non_exported_submodule_stays_reachable_as_an_attribute(self) -> None:
        """A submodule not named like a public name stays an attribute of the parent after import.

        Widening the guard (which drops only module attributes named like a public name) too far
        would make `karume.dist` raise AttributeError — this turns that mutation red.
        """
        reachable = run_python(
            "import sys, karume.dist, karume.modelcard\n"
            "print(karume.dist is sys.modules['karume.dist'],"
            " karume.modelcard is sys.modules['karume.modelcard'])"
        )

        assert reachable.split() == ["True", "True"]

    def test_an_unknown_name_raises_attribute_error(self) -> None:
        import karume

        with pytest.raises(AttributeError, match="no_such_export"):
            _ = karume.no_such_export
