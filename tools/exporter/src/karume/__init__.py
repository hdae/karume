"""Karume exporter — lowers a torch.export-ed model to the IR (docs/ir-v2.md) and `krm` containers.

The 12 entries of `__all__` are the source of truth for the public surface (the list and the
description here are 1:1 — the machine check is TestThePublicSurface in
`tests/test_architecture_boundary.py`):
IR type (`IrGraph`) · conversion (`convert` /
`curated_decompositions` / `normalize_graph`) · storage conversion (`FixedQuantizedWeight` /
`stored_model`) · provenance (`Provenance`) · verification (`parse_ir_graph` / `verify_container`) ·
the end-to-end path (`export_module` / `export_to_file` / `publish_model`).

Every distribution is produced through the single path of `publish_model` (storage conversion →
write → read back and verify → swap into place — the docstrings of `pipeline.publish_model` and
`publish.publish_container`).

The public surface is resolved on first access (PEP 562). Modules that do not need torch, such as
`karume.dist` / `karume.modelcard` / `karume.container`, do not load torch (or torchvision) when
the package initialises. An eager import would put the whole torch start-up cost on a bare
`import karume.dist` (pinned by `tests/test_package_init.py`).
"""

# The aliases carry `_` — keep implementation helpers out of the public namespace (attributes and
# `dir()`).
import sys as _sys
from importlib import import_module as _import_module
from types import ModuleType as _ModuleType

#: Public name → defining module.
_EXPORTS: dict[str, str] = {
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

__all__ = [
    "FixedQuantizedWeight",
    "IrGraph",
    "Provenance",
    "convert",
    "curated_decompositions",
    "export_module",
    "export_to_file",
    "normalize_graph",
    "parse_ir_graph",
    "publish_model",
    "stored_model",
    "verify_container",
]


def __getattr__(name: str) -> object:
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(_import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})


class _Package(_ModuleType):
    def __setattr__(self, name: str, value: object) -> None:
        # The first import of a submodule sets an attribute of the same name on the parent package.
        # The public function `convert` has the same name as the submodule `karume.convert`, so
        # letting that through turns `karume.convert` / `from karume import convert` into the
        # module (importing `karume.pipeline` first is enough). A public name MUST always point at
        # what `__getattr__` resolves.
        if name in _EXPORTS and isinstance(value, _ModuleType):
            return
        super().__setattr__(name, value)


_sys.modules[__name__].__class__ = _Package
