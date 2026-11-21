"""`embeddinggemma2/sources.py`: the pin of the upstream snapshot and its check.

Stand-in snapshot directories exercise the check; the case that reads the real snapshot SKIPs
explicitly without it.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from _shared.paths import INPUTS_ROOT
from embeddinggemma2 import sources


def _snapshot(root: Path, files: dict[str, bytes]) -> sources.SourcePin:
    """Write `files` under `root` and return a pin naming their digests."""
    for name, payload in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    return sources.SourcePin(
        repo="test/model",
        revision="a" * 40,
        sha256={name: hashlib.sha256(payload).hexdigest() for name, payload in files.items()},
    )


class TestPin:
    def test_the_pin_names_the_commit_and_every_file_the_family_reads(self) -> None:
        assert sources.PIN.repo == "google/embeddinggemma-2"
        assert sources.PIN.revision == "f9972ac22e696a105c482202646979e33e4b5a1e"
        assert sorted(sources.PIN.sha256) == [
            "1_Pooling/config.json",
            "2_Normalize/config.json",
            "config.json",
            "config_sentence_transformers.json",
            "model.safetensors",
            "modules.json",
            "sentence_bert_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
        ]
        assert (
            sources.PIN.label == "google/embeddinggemma-2@f9972ac22e696a105c482202646979e33e4b5a1e"
        )

    def test_the_snapshot_is_hand_placed_under_the_family_s_inputs(self) -> None:
        assert sources.SNAPSHOT_DIR == INPUTS_ROOT / "embeddinggemma2" / "embeddinggemma-2"

    def test_the_real_snapshot_matches_the_pin(self) -> None:
        """Hashes the 1.49 GB checkpoint too (needs the snapshot)."""
        if not (sources.SNAPSHOT_DIR / "config.json").is_file():
            pytest.skip(f"the pinned snapshot is not placed at {sources.SNAPSHOT_DIR}")

        sources.assert_pinned(sources.SNAPSHOT_DIR, sources.PIN.sha256)


class TestAssertPinned:
    def test_the_pinned_bytes_pass(self, tmp_path) -> None:
        pin = _snapshot(tmp_path, {"tokenizer.json": b"{}", "1_Pooling/config.json": b"[]"})

        sources.assert_pinned(tmp_path, pin.sha256, pin)

    def test_other_bytes_fail_with_both_digests(self, tmp_path) -> None:
        pin = _snapshot(tmp_path, {"tokenizer.json": b"{}"})
        (tmp_path / "tokenizer.json").write_bytes(b"{ }")

        with pytest.raises(sources.SourcePinError, match=pin.sha256["tokenizer.json"]):
            sources.assert_pinned(tmp_path, ["tokenizer.json"], pin)

    def test_a_missing_file_fails_with_the_fetch_command(self, tmp_path) -> None:
        pin = _snapshot(tmp_path, {"tokenizer.json": b"{}"})
        (tmp_path / "tokenizer.json").unlink()

        with pytest.raises(sources.SourcePinError, match="hf download test/model --revision"):
            sources.assert_pinned(tmp_path, ["tokenizer.json"], pin)

    def test_a_file_the_pin_does_not_name_fails(self, tmp_path) -> None:
        """Reading an unpinned file is the slip the check exists for."""
        pin = _snapshot(tmp_path, {"tokenizer.json": b"{}", "chat_template.jinja": b""})
        unpinned = sources.SourcePin(
            repo=pin.repo,
            revision=pin.revision,
            sha256={"tokenizer.json": pin.sha256["tokenizer.json"]},
        )

        with pytest.raises(sources.SourcePinError, match="not pinned"):
            sources.assert_pinned(tmp_path, ["tokenizer.json", "chat_template.jinja"], unpinned)

    def test_only_the_named_files_are_read(self, tmp_path) -> None:
        """A script checks what it reads — a pinned file it does not name may be absent."""
        pin = _snapshot(tmp_path, {"tokenizer.json": b"{}", "model.safetensors": b"x"})
        (tmp_path / "model.safetensors").unlink()

        sources.assert_pinned(tmp_path, ["tokenizer.json"], pin)
