"""The version the manifest's `generator` names matches the source's `pyproject.toml` version.

`dist.generator_tag()` copies the version from the dist-info of the `karume` installed in the venv.
Forgetting to sync the venv after a bump silently writes a manifest that names the previous
version. This test stops that mix-up by going red (the fix is `(cd tools && uv sync --all-groups)`).
"""

from __future__ import annotations

import importlib.metadata
import tomllib
from pathlib import Path

from karume.dist import generator_tag

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


def _source_version() -> str:
    with PYPROJECT.open("rb") as file:
        return tomllib.load(file)["project"]["version"]


class TestGeneratorVersion:
    """Compares the venv's dist-info with `[project].version` in pyproject.toml."""

    def test_installed_distribution_matches_pyproject_version(self) -> None:
        assert importlib.metadata.version("karume") == _source_version()

    def test_generator_tag_names_the_pyproject_version(self) -> None:
        assert generator_tag() == f"karume/{_source_version()}"
