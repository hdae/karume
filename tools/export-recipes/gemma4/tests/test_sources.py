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

#: The `tokenizer.json` every Gemma 4 repository shipped before Google's glitch-token fixes — what
#: the QAT repositories still ship (`gemma4.sources`).
PRE_FIX_TOKENIZER_SHA256 = "cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f"
#: The regular `-it` repositories' `tokenizer.json` after the fixes of 2026-11-03 (one per model).
FIXED_TOKENIZER_SHA256 = {
    "e2b": "ab26b2add729e6f645bd869f228f14190a06615538f3be6f7786f33d13d78ad2",
    "e4b": "91d9cf86906f78e197021fda88214d448e5c120665d0243a8e776e046986c6bc",
}


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
        assert (
            sources.HUB_PINS["12b"].tokenizer.sha256["tokenizer.json"] != PRE_FIX_TOKENIZER_SHA256
        )

    def test_the_cached_snapshots_match_the_pins(self) -> None:
        """The real hub cache (needs both snapshots)."""
        pytest.importorskip("huggingface_hub")
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
        pytest.importorskip("huggingface_hub")
        weights_dir = sources.checkpoint_dir("12b")
        if not (weights_dir / "tokenizer.json").is_file():
            pytest.skip(f"12B's QAT snapshot is not in the hub cache ({weights_dir})")

        with pytest.raises(sources.SourcePinError, match=PRE_FIX_TOKENIZER_SHA256):
            sources.assert_snapshot_pin(weights_dir, sources.HUB_PINS["12b"].tokenizer)


class TestSnapshotPin:
    def test_the_pinned_snapshot_passes(self, tmp_path) -> None:
        snapshot, pin = _snapshot(tmp_path)

        sources.assert_snapshot_pin(snapshot, pin)

    def test_another_blob_fails_loudly(self, tmp_path) -> None:
        snapshot, pin = _snapshot(tmp_path)

        with pytest.raises(sources.SourcePinError, match=r"model\.safetensors"):
            sources.assert_snapshot_pin(snapshot, _wrong(pin, "model.safetensors"))

    def test_a_blob_that_links_on_to_another_blob_passes(self, tmp_path) -> None:
        # huggingface_hub 1.33 links blobs/<sha256> on to blobs/<shard>/<another 64-hex hash>:
        # the snapshot's own link still names the SHA-256, the end of the chain does not.
        snapshot, pin = _snapshot(tmp_path, lfs=("model.safetensors",))
        blob = tmp_path / "blobs" / pin.sha256["model.safetensors"]
        shard = tmp_path / "blobs" / "5b"
        shard.mkdir()
        blob.rename(shard / ("5b" * 32))
        blob.symlink_to(Path("5b") / ("5b" * 32))

        sources.assert_snapshot_pin(snapshot, pin)

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


class TestQatPins:
    """The q4_0 QAT line of E2B / E4B (`gemma4.export_product_qat`): the QAT weights plus the
    regular tokenizer, the same two-source form as 12B."""

    @pytest.mark.parametrize(
        ("model", "weights", "tokenizer"),
        [
            (
                "e2b",
                "6befbaca7398925921802abd1f277b495b78b738",
                "4adf15748de9e1a0097b6274cee9c15836c41c37",
            ),
            (
                "e4b",
                "476025a01dbf99361c062bbeca3d6a76bb4c4566",
                "4ecc2ba63bcac23ba4a767fc87025b1450f84c9c",
            ),
        ],
    )
    def test_the_weights_are_the_qat_checkpoint_and_the_tokenizer_the_regular_one(
        self, model: str, weights: str, tokenizer: str
    ) -> None:
        pins = sources.qat_pins(model)
        name = model.upper()

        assert pins.weights.repo == f"google/gemma-4-{name}-it-qat-q4_0-unquantized"
        assert pins.weights.revision == weights
        assert sorted(pins.weights.sha256) == ["config.json", "model.safetensors"]
        assert pins.tokenizer.repo == f"google/gemma-4-{name}-it"
        assert pins.tokenizer.revision == tokenizer
        assert pins.tokenizer.sha256 == {"tokenizer.json": FIXED_TOKENIZER_SHA256[model]}

    @pytest.mark.parametrize("model", ["e2b", "e4b"])
    def test_the_tokenizer_revision_is_the_regular_line_s(self, model: str) -> None:
        """The regular line's hand-placed checkpoint records the commit it was fetched at (needs
        `inputs/gemma4/`)."""
        from _shared.upstream import UpstreamProvenanceError, snapshot_revision

        regular = sources.checkpoint_dir(model)
        try:
            revision = snapshot_revision(regular)
        except UpstreamProvenanceError as error:  # the record is a local asset (assets-layout)
            pytest.skip(f"no fetch record for {regular}: {error}")

        assert sources.qat_pins(model).tokenizer.revision == revision

    def test_the_directories_are_the_two_hub_cache_snapshots(self, tmp_path, monkeypatch) -> None:
        constants = pytest.importorskip("huggingface_hub.constants")
        monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path))
        pins = sources.qat_pins("e2b")

        assert sources.qat_checkpoint_dir("e2b") == (
            tmp_path
            / "models--google--gemma-4-E2B-it-qat-q4_0-unquantized"
            / "snapshots"
            / pins.weights.revision
        )
        assert sources.qat_tokenizer_dir("e2b") == (
            tmp_path / "models--google--gemma-4-E2B-it" / "snapshots" / pins.tokenizer.revision
        )

    def test_the_label_names_both_sources(self) -> None:
        pins = sources.qat_pins("e4b")

        assert pins.label == (
            f"google/gemma-4-E4B-it-qat-q4_0-unquantized@{pins.weights.revision} (weights)"
            f" + google/gemma-4-E4B-it@{pins.tokenizer.revision} (tokenizer)"
        )

    def test_the_regular_line_keeps_its_hand_placed_checkpoints(self) -> None:
        pytest.importorskip("huggingface_hub")
        assert "e2b" not in sources.HUB_PINS and "e4b" not in sources.HUB_PINS
        assert sources.checkpoint_dir("e2b") != sources.qat_checkpoint_dir("e2b")

    def test_a_model_without_a_qat_line_fails_loudly(self) -> None:
        with pytest.raises(ValueError, match="no q4_0 QAT line"):
            sources.qat_pins("12b")

    @pytest.mark.parametrize("model", ["e2b", "e4b"])
    def test_the_cached_snapshots_match_the_pins(self, model: str) -> None:
        """The real hub cache (needs both snapshots)."""
        pytest.importorskip("huggingface_hub")
        weights_dir, tokenizer_dir = (
            sources.qat_checkpoint_dir(model),
            sources.qat_tokenizer_dir(model),
        )
        if not all((path / "config.json").is_file() for path in (weights_dir, tokenizer_dir)):
            pytest.skip(f"{model}'s QAT snapshots are not in the hub cache")

        sources.assert_model_pins(weights_dir, tokenizer_dir, sources.qat_pins(model))


class TestQatDrafterPins:
    """12B's q4_0 QAT drafter (`gemma4.export_drafter_qat --model 12b`) and the target each QAT
    drafter pairs with (`QAT_TARGET_PINS`)."""

    def test_12b_pins_its_qat_assistant(self) -> None:
        pin = sources.qat_drafter_pin("12b")

        assert pin.repo == "google/gemma-4-12B-it-qat-q4_0-unquantized-assistant"
        assert pin.revision == "18934064dd4c5c6cc3621f6381e7d377fc8cb7bd"
        assert sorted(pin.sha256) == ["config.json", "model.safetensors"]
        assert pin.sha256["model.safetensors"] == (
            "67f1420cf24aa5065089aaed175223f7c245ccfda16111b6c56765afd7280db6"
        )

    def test_e2b_and_e4b_pair_with_their_qat_line(self) -> None:
        for model in ("e2b", "e4b"):
            assert sources.qat_target_pins(model) is sources.qat_pins(model)

    def test_12b_pairs_with_its_one_pinned_line_whose_weights_are_the_qat_checkpoint(self) -> None:
        pins = sources.qat_target_pins("12b")

        assert pins is sources.HUB_PINS["12b"]
        assert pins.weights.repo == "google/gemma-4-12B-it-qat-q4_0-unquantized"
        assert pins.tokenizer.repo == "google/gemma-4-12B-it"

    def test_every_qat_drafter_and_only_those_have_a_qat_target(self) -> None:
        assert set(sources.QAT_TARGET_PINS) == set(sources.QAT_DRAFTER_PINS)

    def test_the_qat_line_of_the_product_series_stays_e2b_and_e4b(self) -> None:
        """12B's QAT weights are its product series' (`gemma4.export_unified`), not a second line
        `gemma4.export_product_qat` could write."""
        assert sorted(sources.QAT_PINS) == ["e2b", "e4b"]

    def test_the_target_directories_are_the_pinned_snapshots(self, tmp_path, monkeypatch) -> None:
        constants = pytest.importorskip("huggingface_hub.constants")
        monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path))

        assert sources.qat_target_checkpoint_dir("e4b") == sources.qat_checkpoint_dir("e4b")
        assert sources.qat_target_tokenizer_dir("e4b") == sources.qat_tokenizer_dir("e4b")
        assert sources.qat_target_checkpoint_dir("12b") == sources.checkpoint_dir("12b")
        assert sources.qat_target_tokenizer_dir("12b") == sources.tokenizer_dir("12b")
        assert sources.qat_drafter_dir("12b") == (
            tmp_path
            / "models--google--gemma-4-12B-it-qat-q4_0-unquantized-assistant"
            / "snapshots"
            / "18934064dd4c5c6cc3621f6381e7d377fc8cb7bd"
        )

    def test_a_model_without_a_qat_target_fails_loudly(self) -> None:
        with pytest.raises(ValueError, match="no q4_0 QAT target"):
            sources.qat_target_pins("31b")

    def test_the_cached_12b_snapshot_matches_the_pin(self) -> None:
        """The real hub cache (needs the snapshot)."""
        pytest.importorskip("huggingface_hub")
        directory = sources.qat_drafter_dir("12b")
        if not (directory / "config.json").is_file():
            pytest.skip(f"12B's QAT drafter is not in the hub cache ({directory})")

        sources.assert_snapshot_pin(directory, sources.qat_drafter_pin("12b"))
