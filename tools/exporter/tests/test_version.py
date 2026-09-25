"""manifest の `generator` が名乗る版と、ソースの `pyproject.toml` の版が一致すること。

`dist.generator_tag()` は venv に入っている `karume` の dist-info の版を写す。bump の後に venv を
同期し忘れると、前の版を名乗る manifest が黙って焼ける。その取り違えをこのテストが赤で止める
（直し方は `(cd tools && uv sync --all-groups)`）。
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
    """venv の dist-info と pyproject.toml の `[project].version` の突き合わせ。"""

    def test_installed_distribution_matches_pyproject_version(self) -> None:
        assert importlib.metadata.version("karume") == _source_version()

    def test_generator_tag_names_the_pyproject_version(self) -> None:
        assert generator_tag() == f"karume/{_source_version()}"
