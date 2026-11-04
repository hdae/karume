"""The root of path resolution for all recipes (`_shared.paths`).

`REPO_ROOT = Path(__file__).resolve().parents[3]` is **a derived value that depends on the file's
depth**; one level off shifts all 7 constants. It breaks badly: the family tests that use
`INPUTS_ROOT` / `SERIES_ROOT` fall into the missing-asset `skipif` / `importorskip`, so they stay
**green with everything skipped**.

MUST: the **existence** of `models/` / `outputs/` is not required (both are outside git and do not
exist on a machine without assets). Only the number of levels and the spellings are checked here.
"""

from __future__ import annotations

from _shared.paths import (
    BENCH_ROOT,
    DIST_ROOT,
    EXAMPLES_ROOT,
    INPUTS_ROOT,
    MISC_ROOT,
    OUTPUTS_ROOT,
    REPO_ROOT,
    SERIES_ROOT,
)


class TestRepoRoot:
    def test_it_points_at_the_repository_root(self) -> None:
        """Check the root against a permanent file at the repository root (`deno.json` — the
        verification commands section of CLAUDE.md)."""
        assert (REPO_ROOT / "deno.json").is_file()

    def test_it_can_reach_back_to_this_module_source(self) -> None:
        """Reaching back to the module itself = direct proof of the level count of
        `parents[3]`."""
        assert (REPO_ROOT / "tools" / "export-recipes" / "_shared" / "paths.py").is_file()


class TestDerivedRoots:
    def test_the_three_asset_roots_hang_off_the_repository_root(self) -> None:
        """The 3 asset roots (distributions / generated files / input material) are at the
        repository root (docs/assets-layout.md)."""
        assert DIST_ROOT.parent == REPO_ROOT
        assert OUTPUTS_ROOT.parent == REPO_ROOT
        assert INPUTS_ROOT.parent == REPO_ROOT

    def test_the_generated_roots_hang_off_the_outputs_root(self) -> None:
        """MUST: generated files are not placed in `models/` (the DECIDED at the top of
        paths.py)."""
        for root in (SERIES_ROOT, BENCH_ROOT, MISC_ROOT, EXAMPLES_ROOT):
            assert root.parent == OUTPUTS_ROOT, root

    def test_the_spellings_are_the_ones_the_layout_document_names(self) -> None:
        """The spellings are knowledge shared by the writers (scripts) and the reader (the dist
        driver), so the values themselves are pinned."""
        assert DIST_ROOT.name == "models"
        assert OUTPUTS_ROOT.name == "outputs"
        assert INPUTS_ROOT.name == "inputs"
        assert SERIES_ROOT.name == "series"
