"""Model card rendering of the SigLIP2 distribution (`siglip2.card`).

The real distribution's `karume.json` is not used — the job here is to see the fake manifest's
values appear in the body as they are (= not a single hand-written number mixed in), so building
it with values **different** from the real ones detects more.

The derivation from the manifest (tables, numbers, usage) is checked over **one assembly pass** by
`TestSiglip2ModelCard` in `siglip2/tests/test_distribution.py`. This file holds the checks that
exist only on the card side — the template's pipeline specificity, not rendering a model missing
from the attribution table, and the load entry point it advertises.
"""

from __future__ import annotations

import copy
from typing import Any

import pytest

from siglip2.card import (
    SIGLIP2_MAP_HEAD_DIFF,
    SIGLIP2_MAP_HEAD_NORM,
    SIGLIP2_SUPPORTED_PIPELINE,
    render_siglip2_model_card,
)

#: Repository ID spelled in the usage snippet (dist passes it from the pipeline's declaration).
REPO = "hdae/fake-repo"

#: Another family's pipeline contract (**the spelling is kept as is** — to avoid code coupling
#: between recipes). `sbv2/tests/test_card.py` checks in both directions that two real templates
#: refuse each other's manifest. Here only one direction is needed: "refuse any contract other than
#: its own".
FOREIGN_PIPELINE = "birefnet/1"


def _ref(path: str, size: int, digit: str) -> dict[str, Any]:
    return {"path": path, "size": size, "sha256": digit * 64}


def _container(*refs: dict[str, Any]) -> dict[str, Any]:
    """One dtype of the weights (the `container` of `karume/5` — ADR 0109 decision 3).

    The card does not read `descriptor`, so only its shape follows the real thing (the lengths and
    sha256 of the two documents).
    """
    return {
        "container": {
            "descriptor": {
                "graph": {"length": 40, "sha256": "0" * 64},
                "model": {"length": 24, "sha256": "1" * 64},
            },
            "parts": list(refs),
        }
    }


def _siglip2_manifest(model: str = "base") -> dict[str, Any]:
    """Minimal SigLIP2 manifest (fake values that do not overlap the real ones)."""
    return {
        "format": "karume/5",
        "generator": "karume/9.9.9",
        "defaultModel": model,
        "models": {
            model: {
                "pipeline": SIGLIP2_SUPPORTED_PIPELINE,
                "weights": {
                    "vision": {"f32": _container(_ref("v/model.f32-00001-of-00003.krm", 11, "c"))}
                },
                "assets": {},
                "quants": {"f32": {"weights": {"vision": "f32"}, "session": {}}},
                "defaultQuant": "f32",
                "pipelineConfig": {
                    "imageWidth": 64,
                    "imageHeight": 96,
                    "imageMean": [0.1, 0.2, 0.3],
                    "imageStd": [0.4, 0.5, 0.6],
                    "hiddenDim": 7,
                    "interpolation": "bilinear",
                },
            }
        },
    }


class TestSiglip2CardGate:
    def test_it_refuses_a_pipeline_it_does_not_describe(self) -> None:
        manifest = _siglip2_manifest()
        manifest["models"]["base"]["pipeline"] = FOREIGN_PIPELINE
        with pytest.raises(ValueError, match=SIGLIP2_SUPPORTED_PIPELINE):
            render_siglip2_model_card(manifest, REPO)

    def test_it_refuses_a_model_it_cannot_attribute(self) -> None:
        """Rendering for a model name missing from the attribution table would silently produce a
        card with one `base_model` short (= a redistribution that does not name its source). It
        fails, listing the known names.
        """
        with pytest.raises(ValueError, match="large"):
            render_siglip2_model_card(_siglip2_manifest("large"), REPO)

    def test_it_renders_the_same_bytes_for_the_same_manifest(self) -> None:
        manifest = _siglip2_manifest()
        before = copy.deepcopy(manifest)
        assert render_siglip2_model_card(manifest, REPO) == render_siglip2_model_card(
            manifest, REPO
        )
        assert manifest == before


class TestSiglip2LegalPointers:
    """The legal texts at the repository root (ADR 0092 decision 7) can be reached from the card
    body."""

    def test_it_points_at_the_bundled_license_and_notice(self) -> None:
        card = render_siglip2_model_card(_siglip2_manifest(), REPO)
        assert "a verbatim copy is in `LICENSE.md`" in card
        assert "also listed in `NOTICE.md`, per Apache 2.0 §4(b)" in card

    def test_it_reports_the_measured_map_head_difference(self) -> None:
        """The difference the attribution section states is the single measured spelling (the
        source of truth is `siglip2.measurements`)."""
        card = render_siglip2_model_card(_siglip2_manifest(), REPO)
        assert SIGLIP2_MAP_HEAD_DIFF in card

    def test_it_reports_the_scale_that_gives_the_difference_its_meaning(self) -> None:
        """The size of the difference can only be read as a ratio to the scale (the L2 norm) —
        that number, too, has a single spelling.

        If the card and `NOTICE.md` each wrote it directly, the day only one moves would silently
        allow "only the notice claims a stale number" (the MUST of `siglip2.measurements`).
        """
        card = render_siglip2_model_card(_siglip2_manifest(), REPO)
        assert f"L2 norm is {SIGLIP2_MAP_HEAD_NORM}" in card


class TestSiglip2EntryPoint:
    """The load entry point the card advertises — `fromPretrained` only."""

    def test_it_does_not_advertise_the_local_asset_entry_point(self) -> None:
        """`fromAssets` is not advertised (ruling of 2026-08-29).

        The split distribution form became readable too (X2-101), but that surface is meant for
        local debugging and assumes the caller holds the bytes; the ordinary entry point for a
        reader using HF is `fromPretrained`. Listing both would leave the reader to decide "which
        one to use". `fromPretrained` is checked as well so that the check does not pass when the
        whole Usage section disappears.
        """
        card = render_siglip2_model_card(_siglip2_manifest(), REPO)
        assert "fromAssets" not in card
        assert "await Pipeline.fromPretrained" in card


class TestSiglip2HouseSentence:
    def test_the_title_is_followed_by_the_house_sentence(self) -> None:
        lines = render_siglip2_model_card(_siglip2_manifest(), REPO).splitlines()
        title = lines.index("# SigLIP2 Vision Tower — Karume")
        assert lines[title + 2] == (
            "This is an unofficial conversion of google/siglip2-base-patch16-224 for Karume."
            " It is not affiliated with or endorsed by Google."
        )
