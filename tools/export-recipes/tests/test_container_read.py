"""The recipe-side reader that reads a placed container back as stored (`_shared.container_read`).

Only 2 points are pinned here:

- **An opened container can be reused** (`open_container` → each reader). Passing the
  representative path makes `verify_container` redo the descriptor parse and the merge every time,
  so a caller that reads block by block (the PLE acceptance check of gemma4) paid that repetition
  once per block.
- **The diagnostics use the repository's vocabulary** (`ContainerReadError`). A `KeyError` is hard
  to read because its `repr` wraps the message in double quotes, and the caller's `except` cannot
  tell it apart from "a dictionary lookup that missed".
"""

from __future__ import annotations

from pathlib import Path

import pytest
from container_series import write_component
from ir_fixtures import FIXTURE_PROVENANCE, ir_container

from _shared.container_read import (
    ContainerReadError,
    open_container,
    read_asset,
    read_asset_declarations,
    read_layouts,
    read_provenance,
    read_stored,
)
from karume.container import AssetInput

ASSET_NAME = "rope_base"
ASSET_ROLE = "rope-base"
ASSET_PAYLOAD = bytes(range(64))


@pytest.fixture
def component(tmp_path: Path) -> Path:
    """A valid component that bundles one asset (representative path)."""
    path = tmp_path / "series" / "dit" / "model.krm"
    write_component(
        path,
        ir_container(
            mark="reader",
            assets={ASSET_NAME: AssetInput(ASSET_ROLE, len(ASSET_PAYLOAD), ASSET_PAYLOAD)},
        ),
    )
    return path


class TestTheOpenedContainerIsReusable:
    """A read given the return value of `open_container` answers the same as a read given the
    representative path."""

    def test_the_asset_reads_the_same_bytes_from_either_entry(self, component: Path) -> None:
        opened = open_container(component)

        assert read_asset(opened, ASSET_NAME) == read_asset(component, ASSET_NAME)
        assert read_asset(opened, ASSET_NAME) == ASSET_PAYLOAD

    def test_the_declarations_and_layouts_agree_from_either_entry(self, component: Path) -> None:
        opened = open_container(component)

        assert read_asset_declarations(opened) == read_asset_declarations(component)
        assert read_layouts(opened) == read_layouts(component)
        assert read_stored(opened).keys() == read_stored(component).keys()

    def test_reusing_the_opened_container_opens_nothing_again(
        self, component: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Fault injection — reverting to an implementation that reopens calls `verify_container`
        once more here."""
        opened = open_container(component)
        from _shared import container_read

        def forbidden(*args: object, **kwargs: object) -> None:
            raise AssertionError("ran verify_container although an opened container was passed")

        monkeypatch.setattr(container_read, "verify_container", forbidden)

        assert read_asset(opened, ASSET_NAME) == ASSET_PAYLOAD
        assert read_asset_declarations(opened)[ASSET_NAME] == (ASSET_ROLE, len(ASSET_PAYLOAD))


class TestTheDiagnosticsUseTheRepositoryVocabulary:
    def test_a_missing_asset_fails_with_the_reader_error(self, component: Path) -> None:
        with pytest.raises(ContainerReadError, match="no asset 'absent'"):
            read_asset(component, "absent")

    def test_the_error_is_not_a_key_error(self, component: Path) -> None:
        """As a `KeyError`, the caller's `except KeyError` could not tell it apart from a
        dictionary lookup that missed."""
        assert not issubclass(ContainerReadError, KeyError)


class TestTheProvenanceIsTheOneTheWriterBaked:
    def test_it_returns_the_provenance_of_the_model_descriptor(self, component: Path) -> None:
        """The value the provenance match (`_shared.upstream`) reads is the very `provenance` the
        writer wrote."""
        assert read_provenance(component) == FIXTURE_PROVENANCE
