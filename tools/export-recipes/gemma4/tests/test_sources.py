"""`gemma4/sources.py`: where each model's upstream is read from, and the pin checks.

The real 12B snapshots are not needed: stand-in cache directories with symlinked LFS-style blobs
and plain small files exercise both branches of the digest (the blob name and the bytes). The two
cases that read the real hub cache SKIP explicitly without it.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from _shared.paths import INPUTS_ROOT
from gemma4 import sources

#: The QAT repository's `tokenizer.json` (E2B's — the packaging slip `gemma4.sources` describes).
E2B_TOKENIZER_SHA256 = "cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f"


def _snapshot(
    root: Path,
    *,
    repo: str = "test/model",
    lfs: tuple[str, ...] = ("model.safetensors", "tokenizer.json"),
    plain: tuple[str, ...] = ("config.json",),
) -> tuple[Path, sources.HubPin]:
    """A hub-cache-shaped snapshot: the `lfs` files link to blobs named by their SHA-256, the
    `plain` files are plain files; the pin names them all."""
    blobs = root / "blobs"
    snapshot = root / "snapshots" / ("a" * 40)
    blobs.mkdir(parents=True)
    snapshot.mkdir(parents=True)
    digests: dict[str, str] = {}
    for name in lfs:
        payload = f"{name} bytes".encode()
        digest = hashlib.sha256(payload).hexdigest()
        (blobs / digest).write_bytes(payload)
        (snapshot / name).symlink_to(Path("..") / ".." / "blobs" / digest)
        digests[name] = digest
    for name in plain:
        (snapshot / name).write_bytes(b"{}")
        digests[name] = hashlib.sha256(b"{}").hexdigest()
    return snapshot, sources.HubPin(repo=repo, revision="a" * 40, sha256=digests)


def _two_sources(root: Path) -> tuple[Path, Path, sources.ModelPins]:
    """A weights snapshot (`model.safetensors` + `config.json`) and a tokenizer snapshot
    (`tokenizer.json`), as 12B's pins split them."""
    weights_dir, weights = _snapshot(
        root / "weights", repo="test/weights", lfs=("model.safetensors",)
    )
    tokenizer_dir, tokenizer = _snapshot(
        root / "tokenizer", repo="test/tokenizer", lfs=("tokenizer.json",), plain=()
    )
    return weights_dir, tokenizer_dir, sources.ModelPins(weights=weights, tokenizer=tokenizer)


def _wrong(pin: sources.HubPin, name: str) -> sources.HubPin:
    return sources.HubPin(
        repo=pin.repo, revision=pin.revision, sha256={**pin.sha256, name: "0" * 64}
    )


class TestCheckpointDir:
    def test_e2b_and_e4b_stay_hand_placed_in_one_directory(self) -> None:
        assert sources.checkpoint_dir("e2b") == INPUTS_ROOT / "gemma4" / "gemma-4-E2B-it"
        assert sources.checkpoint_dir("e4b") == INPUTS_ROOT / "gemma4" / "gemma-4-E4B-it"
        assert sources.tokenizer_dir("e2b") == sources.checkpoint_dir("e2b")
        assert sources.tokenizer_dir("e4b") == sources.checkpoint_dir("e4b")
        assert sources.checkpoint_label("e2b") == "inputs/gemma4/gemma-4-E2B-it"
        assert sources.tokenizer_label("e2b") == "inputs/gemma4/gemma-4-E2B-it"

    def test_12b_reads_the_weights_and_the_tokenizer_from_two_pinned_snapshots(
        self, tmp_path, monkeypatch
    ) -> None:
        """The cache root is huggingface_hub's own (`constants.HF_HUB_CACHE`)."""
        constants = pytest.importorskip("huggingface_hub.constants")
        monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path))
        pins = sources.HUB_PINS["12b"]

        assert sources.checkpoint_dir("12b") == (
            tmp_path
            / "models--google--gemma-4-12B-it-qat-q4_0-unquantized"
            / "snapshots"
            / pins.weights.revision
        )
        assert sources.tokenizer_dir("12b") == (
            tmp_path / "models--google--gemma-4-12B-it" / "snapshots" / pins.tokenizer.revision
        )

    def test_12b_s_labels_name_their_sources(self) -> None:
        pins = sources.HUB_PINS["12b"]
        weights = f"google/gemma-4-12B-it-qat-q4_0-unquantized@{pins.weights.revision}"
        tokenizer = f"google/gemma-4-12B-it@{pins.tokenizer.revision}"

        assert sources.checkpoint_label("12b") == f"{weights} (weights) + {tokenizer} (tokenizer)"
        assert sources.tokenizer_label("12b") == tokenizer

    def test_an_unknown_model_fails_loudly(self) -> None:
        with pytest.raises(ValueError, match="no model"):
            sources.checkpoint_dir("27b")
        with pytest.raises(ValueError, match="no model"):
            sources.tokenizer_dir("27b")


class TestTwelveBPins:
    def test_the_weights_and_the_tokenizer_are_two_pinned_sources(self) -> None:
        pins = sources.HUB_PINS["12b"]

        assert pins.weights.repo == "google/gemma-4-12B-it-qat-q4_0-unquantized"
        assert pins.weights.revision == "b6ed86275a6a5735884e208bfed95b445a684ca2"
        assert sorted(pins.weights.sha256) == ["config.json", "model.safetensors"]
        assert pins.tokenizer.repo == "google/gemma-4-12B-it"
        assert pins.tokenizer.revision == "43c9d014b231474fa7212dbb48a762566c2b255d"
        assert sorted(pins.tokenizer.sha256) == ["tokenizer.json"]

    def test_the_tokenizer_pin_is_not_the_tokenizer_the_qat_repository_ships(self) -> None:
        assert sources.HUB_PINS["12b"].tokenizer.sha256["tokenizer.json"] != E2B_TOKENIZER_SHA256

    def test_the_cached_snapshots_match_the_pins(self) -> None:
        """The real hub cache (needs both snapshots)."""
        pins = sources.HUB_PINS["12b"]
        weights_dir, tokenizer_dir = sources.checkpoint_dir("12b"), sources.tokenizer_dir("12b")
        if not all((path / "config.json").is_file() for path in (weights_dir, tokenizer_dir)):
            pytest.skip(
                f"12B's snapshots are not in the hub cache ({weights_dir}, {tokenizer_dir})"
            )

        sources.assert_model_pins(weights_dir, tokenizer_dir, pins)

    def test_the_qat_snapshot_s_own_tokenizer_fails_the_tokenizer_pin(self) -> None:
        """Reading the tokenizer next to the weights is the slip the second source exists for
        (needs the QAT snapshot)."""
        weights_dir = sources.checkpoint_dir("12b")
        if not (weights_dir / "tokenizer.json").is_file():
            pytest.skip(f"12B's QAT snapshot is not in the hub cache ({weights_dir})")

        with pytest.raises(sources.SourcePinError, match=E2B_TOKENIZER_SHA256):
            sources.assert_snapshot_pin(weights_dir, sources.HUB_PINS["12b"].tokenizer)


class TestSnapshotPin:
    def test_the_pinned_snapshot_passes(self, tmp_path) -> None:
        snapshot, pin = _snapshot(tmp_path)

        sources.assert_snapshot_pin(snapshot, pin)

    def test_another_blob_fails_loudly(self, tmp_path) -> None:
        snapshot, pin = _snapshot(tmp_path)

        with pytest.raises(sources.SourcePinError, match=r"model\.safetensors"):
            sources.assert_snapshot_pin(snapshot, _wrong(pin, "model.safetensors"))

    def test_changed_small_file_bytes_fail_loudly(self, tmp_path) -> None:
        snapshot, pin = _snapshot(tmp_path)
        (snapshot / "config.json").write_bytes(b'{"edited": true}')

        with pytest.raises(sources.SourcePinError, match=r"config\.json"):
            sources.assert_snapshot_pin(snapshot, pin)

    def test_a_missing_file_fails_loudly(self, tmp_path) -> None:
        snapshot, pin = _snapshot(tmp_path)
        (snapshot / "tokenizer.json").unlink()

        with pytest.raises(sources.SourcePinError, match="missing"):
            sources.assert_snapshot_pin(snapshot, pin)


class TestModelPins:
    def test_both_pinned_snapshots_pass(self, tmp_path) -> None:
        weights_dir, tokenizer_dir, pins = _two_sources(tmp_path)

        sources.assert_model_pins(weights_dir, tokenizer_dir, pins)

    def test_a_wrong_weights_digest_fails_naming_the_weights_source(self, tmp_path) -> None:
        weights_dir, tokenizer_dir, pins = _two_sources(tmp_path)
        wrong = sources.ModelPins(
            weights=_wrong(pins.weights, "model.safetensors"), tokenizer=pins.tokenizer
        )

        with pytest.raises(sources.SourcePinError, match=r"model\.safetensors.*test/weights@"):
            sources.assert_model_pins(weights_dir, tokenizer_dir, wrong)

    def test_a_wrong_tokenizer_digest_fails_naming_the_tokenizer_source(self, tmp_path) -> None:
        weights_dir, tokenizer_dir, pins = _two_sources(tmp_path)
        wrong = sources.ModelPins(
            weights=pins.weights, tokenizer=_wrong(pins.tokenizer, "tokenizer.json")
        )

        with pytest.raises(sources.SourcePinError, match=r"tokenizer\.json.*test/tokenizer@"):
            sources.assert_model_pins(weights_dir, tokenizer_dir, wrong)

    def test_the_tokenizer_is_not_looked_for_next_to_the_weights(self, tmp_path) -> None:
        """Swapping the two directories fails: each pin is checked in its own source."""
        weights_dir, tokenizer_dir, pins = _two_sources(tmp_path)

        with pytest.raises(sources.SourcePinError, match="missing"):
            sources.assert_model_pins(tokenizer_dir, weights_dir, pins)
