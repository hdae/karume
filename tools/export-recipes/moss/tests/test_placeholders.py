"""No ADR-number placeholder is left under `tools/export-recipes/moss/` once the ADR is merged.

The MOSS-Image ADR gets its number when it is merged; until then the recipe cites it with a
placeholder. This check is skipped until then — remove the mark in the commit that assigns the
number, and it fails for every placeholder still left.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

#: The recipe directory (`moss/`).
RECIPE = Path(__file__).resolve().parent.parent

#: The placeholder, also when a line break falls inside it (a pattern, so this file does not hold
#: the literal).
PLACEHOLDER = re.compile(r"ADR\s+N{4}\b")

#: The text files that can cite an ADR.
SUFFIXES = frozenset({".py", ".md", ".toml", ".json", ".txt"})


def placeholder_sites(root: Path) -> list[str]:
    """`<path>:<line>` of every placeholder under `root` (this file excluded)."""
    sites = []
    for path in sorted(root.rglob("*")):
        if path.suffix not in SUFFIXES or path.resolve() == Path(__file__).resolve():
            continue
        text = path.read_text(encoding="utf-8")
        for match in PLACEHOLDER.finditer(text):
            number = text.count("\n", 0, match.start()) + 1
            sites.append(f"{path.relative_to(root).as_posix()}:{number}")
    return sites


def test_the_scan_finds_a_placeholder(tmp_path):
    """The scan itself is live while the merge check below is skipped."""
    (tmp_path / "a.py").write_text("x = 1  # ADR " + "N" * 4 + " decision 3\n", encoding="utf-8")
    (tmp_path / "b.md").write_text("ADR 0065\n\nsee ADR\n" + "N" * 4 + "\n", encoding="utf-8")

    assert placeholder_sites(tmp_path) == ["a.py:1", "b.md:3"]


@pytest.mark.skip(
    reason="the MOSS-Image ADR has no number until it is merged; remove this mark in the commit"
    " that assigns it"
)
def test_no_adr_placeholder_is_left():
    assert placeholder_sites(RECIPE) == []
