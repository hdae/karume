"""Reading the provenance of an upstream checkpoint from the local files (`_shared.upstream`).

Pinned here: "never claim a default value when it cannot be read", and how HF's spellings (the
first line of the download record, `license` / `license_name` of the front matter) are read. The
assembly-side match (`assert_upstream_provenance`) is checked in its real shape by each family's
`tests/test_distribution.py`.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from upstream_fixture import FIXTURE_REVISION, write_snapshot

from _shared.upstream import (
    SNAPSHOT_RECORD_DIR,
    UpstreamProvenanceError,
    card_license_identifier,
    snapshot_license,
    snapshot_provenance,
    snapshot_revision,
)
from karume.container import Provenance
from karume.modelcard import CardMetadata


class TestSnapshotRevision:
    def test_it_reads_the_commit_sha_of_the_download_record(self, tmp_path: Path) -> None:
        write_snapshot(tmp_path, license="mit")

        assert snapshot_revision(tmp_path) == FIXTURE_REVISION

    def test_a_checkpoint_without_the_record_fails_loudly(self, tmp_path: Path) -> None:
        """A checkpoint placed by hand (not through `--local-dir`) cannot claim a revision."""
        with pytest.raises(UpstreamProvenanceError, match="metadata is missing"):
            snapshot_revision(tmp_path)

    @pytest.mark.parametrize("first_line", ["", "main", "4e9de7a", FIXTURE_REVISION.upper()])
    def test_a_first_line_that_is_not_a_commit_sha_fails_loudly(
        self, tmp_path: Path, first_line: str
    ) -> None:
        """A branch name or a short SHA is not accepted as the revision (the same spelling can
        point at another commit)."""
        record = tmp_path / SNAPSHOT_RECORD_DIR / "config.json.metadata"
        record.parent.mkdir(parents=True)
        record.write_text(f"{first_line}\netag\n0\n", encoding="utf-8")

        with pytest.raises(UpstreamProvenanceError, match="commit SHA"):
            snapshot_revision(tmp_path)


class TestSnapshotLicense:
    def test_it_reads_the_license_of_the_front_matter(self, tmp_path: Path) -> None:
        write_snapshot(tmp_path, license="cc-by-nc-4.0")

        assert snapshot_license(tmp_path) == "cc-by-nc-4.0"

    def test_a_custom_license_is_named_by_its_license_name(self, tmp_path: Path) -> None:
        """`other` is not an identifier — the license could not be identified from a container
        taken away."""
        write_snapshot(tmp_path, license="other", extra="license_name: vendor-license\n")

        assert snapshot_license(tmp_path) == "vendor-license"

    def test_other_without_a_license_name_fails_loudly(self, tmp_path: Path) -> None:
        write_snapshot(tmp_path, license="other")

        with pytest.raises(UpstreamProvenanceError, match="license_name"):
            snapshot_license(tmp_path)

    def test_a_readme_without_a_license_fails_loudly(self, tmp_path: Path) -> None:
        (tmp_path / "README.md").write_text("---\ntags:\n- x\n---\n", encoding="utf-8")

        with pytest.raises(UpstreamProvenanceError, match="has no 'license'"):
            snapshot_license(tmp_path)

    def test_a_readme_without_front_matter_fails_loudly(self, tmp_path: Path) -> None:
        (tmp_path / "README.md").write_text("license: mit\n", encoding="utf-8")

        with pytest.raises(UpstreamProvenanceError, match="front matter"):
            snapshot_license(tmp_path)

    def test_a_license_mentioned_below_the_front_matter_is_not_read(self, tmp_path: Path) -> None:
        """A `license:` in the body is not a declaration (nothing outside the front matter is
        read)."""
        (tmp_path / "README.md").write_text(
            "---\ntags:\n- x\n---\nlicense: mit\n", encoding="utf-8"
        )

        with pytest.raises(UpstreamProvenanceError, match="has no 'license'"):
            snapshot_license(tmp_path)


class TestSnapshotProvenance:
    def test_it_carries_the_license_the_revision_and_the_notice(self, tmp_path: Path) -> None:
        write_snapshot(tmp_path, license="apache-2.0")

        assert snapshot_provenance(tmp_path, notice="NOTICE.md") == Provenance(
            license="apache-2.0", notice="NOTICE.md", upstream_revision=FIXTURE_REVISION
        )


class TestCardLicenseIdentifier:
    """The identifier written into the container from the card's frontmatter (for families with
    no upstream README at hand)."""

    def test_a_standard_license_is_the_identifier_itself(self) -> None:
        metadata = CardMetadata(pipeline_tag="t", base_model=("a/b",), license="mit", tags=())

        assert card_license_identifier(metadata, "card") == "mit"

    def test_other_is_named_by_its_license_name(self) -> None:
        metadata = CardMetadata(
            pipeline_tag="t", base_model=("a/b",), license="other", license_name="x", tags=()
        )

        assert card_license_identifier(metadata, "card") == "x"

    def test_other_without_a_license_name_fails_loudly(self) -> None:
        metadata = CardMetadata(pipeline_tag="t", base_model=("a/b",), license="other", tags=())

        with pytest.raises(UpstreamProvenanceError, match="card has 'license: other'"):
            card_license_identifier(metadata, "card")
