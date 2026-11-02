"""The fetched rule of the recipe tests' test assets (`real_assets`): an absent asset FAILs once the
root holds a complete fetch of the pin and SKIPs otherwise — the same split as the TS side's
`testAssetsFetched` (ADR 0139 decision 5)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from real_assets import TEST_ASSETS_PIN, assets_fetched, require_test_asset

PIN = "0123456789abcdef0123456789abcdef01234567"


@pytest.fixture
def pin_file(tmp_path: Path) -> Path:
    path = tmp_path / "pin.json"
    path.write_text(json.dumps({"repo": "hdae/karume-test-assets", "revision": PIN}))
    return path


@pytest.fixture
def root(tmp_path: Path) -> Path:
    path = tmp_path / "test-assets"
    path.mkdir()
    return path


def test_the_pin_is_the_file_the_ts_side_imports():
    pin = json.loads(TEST_ASSETS_PIN.read_text(encoding="utf-8"))
    assert set(pin) == {"repo", "revision"}
    assert len(pin["revision"]) == 40


class TestAssetsFetched:
    def test_is_true_only_when_the_record_is_the_pin(self, root: Path, pin_file: Path):
        (root / ".revision").write_text(f"{PIN}\n")
        assert assets_fetched(root, pin_file)
        (root / ".revision").write_text(f"{'f' * 40}\n")
        assert not assets_fetched(root, pin_file)

    def test_is_false_without_a_record(self, root: Path, pin_file: Path):
        assert not assets_fetched(root, pin_file)


class TestRequireTestAsset:
    def test_returns_a_present_file(self, root: Path, pin_file: Path):
        path = root / "goldens" / "a.bin"
        path.parent.mkdir()
        path.write_bytes(b"x")
        assert require_test_asset(path, root=root, pin_file=pin_file) == path

    def test_fails_on_an_absent_file_after_a_complete_fetch(self, root: Path, pin_file: Path):
        (root / ".revision").write_text(f"{PIN}\n")
        with pytest.raises(pytest.fail.Exception, match="the dataset at the pin lacks it"):
            require_test_asset(root / "missing.bin", root=root, pin_file=pin_file)

    def test_skips_on_an_absent_file_without_a_complete_fetch(self, root: Path, pin_file: Path):
        with pytest.raises(pytest.skip.Exception, match=r"node tools/test-assets/fetch\.ts"):
            require_test_asset(root / "missing.bin", root=root, pin_file=pin_file)
