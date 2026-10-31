"""No ADR-number placeholder and no reference to the untracked plan is left under
`tools/export-recipes/moss/`.

The MOSS-Image ADR got its number (0138) at the rebase onto main; until then the recipe cited it
with a placeholder, which the commit that assigned the number replaced. The recipe no longer
cites the working plan (an untracked file). The check fails for a placeholder or a plan reference
that comes back.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

#: The recipe directory (`moss/`).
RECIPE = Path(__file__).resolve().parent.parent

#: The ADR placeholder and the plan references ("plan item(s)", "plan step", "plan question",
#: "plan §"), also when a line break falls inside one (patterns, so this file does not hold the
#: literals).
PLACEHOLDERS = (
    re.compile(r"ADR\s+N{4}\b"),
    re.compile(r"\bplan\s+(?:(?:items?|step|question)\b|§)", re.IGNORECASE),
)

#: The text files that can cite an ADR.
SUFFIXES = frozenset({".py", ".md", ".toml", ".json", ".txt"})


def placeholder_sites(root: Path) -> list[str]:
    """`<path>:<line>` of every placeholder under `root` (this file excluded), in file order."""
    sites = []
    for path in sorted(root.rglob("*")):
        if path.suffix not in SUFFIXES or path.resolve() == Path(__file__).resolve():
            continue
        text = path.read_text(encoding="utf-8")
        starts = sorted(m.start() for pattern in PLACEHOLDERS for m in pattern.finditer(text))
        for start in starts:
            number = text.count("\n", 0, start) + 1
            sites.append(f"{path.relative_to(root).as_posix()}:{number}")
    return sites


def test_the_scan_finds_a_placeholder(tmp_path):
    """The scan itself finds a placeholder (the check below would pass on a scan that finds none)."""
    (tmp_path / "a.py").write_text("x = 1  # ADR " + "N" * 4 + " decision 3\n", encoding="utf-8")
    (tmp_path / "b.md").write_text("ADR 0065\n\nsee ADR\n" + "N" * 4 + "\n", encoding="utf-8")

    assert placeholder_sites(tmp_path) == ["a.py:1", "b.md:3"]


@pytest.mark.parametrize(
    "reference",
    ["plan item 15-5", "plan items 15-1 to 15-3", "plan step 1", "Plan\nquestion 18", "plan §1.4"],
)
def test_the_scan_finds_a_plan_reference(tmp_path, reference):
    (tmp_path / "a.py").write_text(f"x = 1\n# see the {reference}\n", encoding="utf-8")

    assert placeholder_sites(tmp_path) == ["a.py:2"]


def test_the_scan_leaves_other_uses_of_plan_alone(tmp_path):
    """A storage plan or the word "planned" is no reference to the working plan."""
    text = "plan = build_plan(steps)  # the planned items\n"
    (tmp_path / "a.py").write_text(text, encoding="utf-8")

    assert placeholder_sites(tmp_path) == []


def test_no_adr_placeholder_is_left():
    assert placeholder_sites(RECIPE) == []
