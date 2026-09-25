"""上流 checkpoint の出所を手元の実物から読む（`_shared.upstream`）。

固定するのは「読めないときに既定値を名乗らない」ことと、HF の綴り（取得記録の 1 行目・
front matter の `license` / `license_name`）の読み方。組み立て側の突合
（`assert_upstream_provenance`）は各 family の `tests/test_distribution.py` が実物の形で見る。
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
        """手で置いた checkpoint（`--local-dir` を通していない）は revision を名乗れない。"""
        with pytest.raises(UpstreamProvenanceError, match="metadata が無い"):
            snapshot_revision(tmp_path)

    @pytest.mark.parametrize("first_line", ["", "main", "4e9de7a", FIXTURE_REVISION.upper()])
    def test_a_first_line_that_is_not_a_commit_sha_fails_loudly(
        self, tmp_path: Path, first_line: str
    ) -> None:
        """枝名・短縮 SHA は revision の席に入れない（同じ綴りが別の commit を指しうる）。"""
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
        """`other` は識別子ではない — 持ち出した容器からライセンスが特定できなくなる。"""
        write_snapshot(tmp_path, license="other", extra="license_name: vendor-license\n")

        assert snapshot_license(tmp_path) == "vendor-license"

    def test_other_without_a_license_name_fails_loudly(self, tmp_path: Path) -> None:
        write_snapshot(tmp_path, license="other")

        with pytest.raises(UpstreamProvenanceError, match="license_name"):
            snapshot_license(tmp_path)

    def test_a_readme_without_a_license_fails_loudly(self, tmp_path: Path) -> None:
        (tmp_path / "README.md").write_text("---\ntags:\n- x\n---\n", encoding="utf-8")

        with pytest.raises(UpstreamProvenanceError, match="'license' が無い"):
            snapshot_license(tmp_path)

    def test_a_readme_without_front_matter_fails_loudly(self, tmp_path: Path) -> None:
        (tmp_path / "README.md").write_text("license: mit\n", encoding="utf-8")

        with pytest.raises(UpstreamProvenanceError, match="front matter"):
            snapshot_license(tmp_path)

    def test_a_license_mentioned_below_the_front_matter_is_not_read(self, tmp_path: Path) -> None:
        """本文の `license:` は宣言ではない（front matter の外は読まない）。"""
        (tmp_path / "README.md").write_text(
            "---\ntags:\n- x\n---\nlicense: mit\n", encoding="utf-8"
        )

        with pytest.raises(UpstreamProvenanceError, match="'license' が無い"):
            snapshot_license(tmp_path)


class TestSnapshotProvenance:
    def test_it_carries_the_license_the_revision_and_the_notice(self, tmp_path: Path) -> None:
        write_snapshot(tmp_path, license="apache-2.0")

        assert snapshot_provenance(tmp_path, notice="NOTICE.md") == Provenance(
            license="apache-2.0", notice="NOTICE.md", upstream_revision=FIXTURE_REVISION
        )


class TestCardLicenseIdentifier:
    """カードの frontmatter から容器へ焼く識別子（手元に上流 README が無い family の席）。"""

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

        with pytest.raises(UpstreamProvenanceError, match="card は 'license: other'"):
            card_license_identifier(metadata, "card")
