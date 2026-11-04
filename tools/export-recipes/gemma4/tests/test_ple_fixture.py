"""The packed PLE fixture committed to the repository is **byte-identical** to what the current
generator produces.

The reader side (`packages/models/tests/gemma4_ple_packed_test.ts`) checks the fixture's
**contents** bit-exactly against a torch reference, but the link "generator → committed file" was
only kept by running it by hand. If the production path ({@link karume.ple.ple_assets} /
{@link karume.pipeline.publish_model}) changes how blocks are cut or the container's byte layout,
nothing turns red and only the fixture is left in the old form.

Only the `krm` (container) files are compared. The neighbouring `oracle.safetensors` bakes
`torch.__version__` into its metadata, so its bytes move on a machine with a newer torch — pinning
that too would make "upgrading torch turns it red", a failure unrelated to what the check protects
(the container's form).
"""

from __future__ import annotations

from pathlib import Path

from gemma4.tests.ple_fixture import FIXTURE_ROOT, write_fixture

#: The storage dtypes compared (all that the generator writes).
STORAGES = ("i2", "i4")


def _parts(directory: Path) -> dict[str, bytes]:
    return {path.name: path.read_bytes() for path in sorted(directory.glob("*.krm"))}


class TestTheCommittedFixtureMatchesItsGenerator:
    """The output of `uv run python -m gemma4.tests.ple_fixture` matches the committed files."""

    def test_every_container_part_is_byte_identical(self, tmp_path: Path) -> None:
        write_fixture(tmp_path)

        for storage in STORAGES:
            committed = _parts(FIXTURE_ROOT / storage)
            regenerated = _parts(tmp_path / storage)

            # Non-vacuous: fail when the committed side is empty (= zero comparisons). The form is
            # split, so there are at least 2 parts.
            assert len(committed) >= 2, f"{storage}: {len(committed)} krm files in the repository"
            assert sorted(regenerated) == sorted(committed), storage
            for name, payload in committed.items():
                assert regenerated[name] == payload, (
                    f"{storage}/{name} differs from the generator's output —"
                    " regenerate with"
                    " `cd tools/export-recipes && uv run python -m gemma4.tests.ple_fixture`"
                )
