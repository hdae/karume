"""Which container `--weights stored` reads: the quant a profile names (`data.profiles`), never the
manifest's `defaultQuant`."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from data import profiles
from run import MANIFEST_FORMAT, container_paths

from gemma4 import sources


def _distribution(root: Path) -> Path:
    """A manifest whose default is the original-weights preset `bf16` (the regular Gemma 4 line's
    since 2026-11-04), next to the int4 presets that load the `i4` container."""
    root.mkdir()
    parts = {
        dtype: [{"path": f"e2b/model/model.{dtype}-0000{n}-of-00002.krm"} for n in (1, 2)]
        for dtype in ("i4", "bf16")
    }
    manifest = {
        "format": MANIFEST_FORMAT,
        "models": {
            "e2b": {
                "defaultQuant": "bf16",
                "quants": {
                    "i4": {"weights": {"model": "i4"}},
                    "i4-fast": {"weights": {"model": "i4"}},
                    "bf16": {"weights": {"model": "bf16"}},
                },
                "weights": {
                    "model": {dtype: {"container": {"parts": parts[dtype]}} for dtype in parts}
                },
            }
        },
    }
    (root / "karume.json").write_text(json.dumps(manifest))
    return root


def test_the_named_quant_wins_over_the_default(tmp_path: Path) -> None:
    root = _distribution(tmp_path / "dist")

    paths = container_paths({"distribution": root, "model": "e2b", "quant": "i4"})

    assert paths == [root / f"e2b/model/model.i4-0000{n}-of-00002.krm" for n in (1, 2)]


def test_a_quant_the_manifest_lacks_fails_loudly(tmp_path: Path) -> None:
    root = _distribution(tmp_path / "dist")

    with pytest.raises(ValueError, match="no quant 'i2'"):
        container_paths({"distribution": root, "model": "e2b", "quant": "i2"})


def test_every_profile_names_the_quant_its_line_ships() -> None:
    """The regular line ships `bf16` alone (since 2026-11-04); both QAT lines ship `i4`."""
    quants = {name: profile["quant"] for name, profile in profiles(Path("/nonexistent")).items()}

    assert quants == {
        "gemma4-e2b": "bf16",
        "gemma4-qat-e2b": "i4",
        "gemma4-qat-e4b": "i4",
        "gemma4-qat-q4_0-e2b": "i4",
        "gemma4-qat-q4_0-e4b": "i4",
    }


def test_the_q4_0_qat_profiles_read_the_regular_tokenizer_not_the_checkpoints() -> None:
    """The q4_0 QAT checkpoints ship the tokenizer from before the glitch-token fixes; the line
    (and so its score) uses the regular `-it` one."""
    for model in ("e2b", "e4b"):
        profile = profiles(Path("/nonexistent"))[f"gemma4-qat-q4_0-{model}"]

        assert profile["checkpoint"] == sources.qat_checkpoint_dir(model)
        assert profile["tokenizer"] == sources.qat_tokenizer_dir(model)
        assert profile["tokenizer"] != profile["checkpoint"]
