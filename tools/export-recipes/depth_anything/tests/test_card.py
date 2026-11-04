"""Model card rendering of the Depth Anything V2 distribution (`depth_anything.card`).

The real distribution's `karume.json` is not used — the job here is to see that the fake
manifest's values reach the body as they are (= not a single hand-written number is mixed in), so
building it from values that **differ** from the real ones detects more.

The derivation from the manifest (tables, numbers, usage) is checked by
`TestDepthAnythingModelCard` in `depth_anything/tests/test_distribution.py` over **one full
assembly run**. What this file holds are the checks that exist only on the card side — the
template's pipeline specificity, not rendering a size missing from the attribution table, and the
loading entry point the card advertises.
"""

from __future__ import annotations

import copy
from typing import Any

import pytest

from depth_anything.card import (
    DEPTH_ANYTHING_LICENSE,
    DEPTH_ANYTHING_SUPPORTED_PIPELINE,
    DEPTH_ANYTHING_UPSTREAM,
    render_depth_anything_model_card,
)

#: The repository ID spelled in the usage snippet (dist passes it from the pipeline declaration).
REPO = "hdae/fake-repo"

#: Another family's pipeline contract (**the spelling is held as is** — so that no code coupling
#: between recipes is created). That two real templates reject each other's manifests is checked
#: in both directions by `sbv2/tests/test_card.py`. Only one direction is needed here: "reject
#: every contract other than its own".
FOREIGN_PIPELINE = "siglip2/1"


def _ref(path: str, size: int, digit: str) -> dict[str, Any]:
    return {"path": path, "size": size, "sha256": digit * 64}


def _container(*refs: dict[str, Any]) -> dict[str, Any]:
    """One dtype's worth of weights (the `container` of `karume/5` — ADR 0109 decision 3).

    The card does not read `descriptor`, so only its shape follows the real thing (the lengths
    and sha256 of the 2 documents).
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


def _depth_anything_manifest(model: str = "small") -> dict[str, Any]:
    """A minimal Depth Anything V2 manifest (fake values that do not overlap with the real
    ones)."""
    return {
        "format": "karume/5",
        "generator": "karume/9.9.9",
        "defaultModel": model,
        "models": {
            model: {
                "pipeline": DEPTH_ANYTHING_SUPPORTED_PIPELINE,
                "weights": {
                    "depth": {"f32": _container(_ref("d/model.f32-00001-of-00003.krm", 17, "e"))}
                },
                "assets": {},
                "quants": {"f32": {"weights": {"depth": "f32"}, "session": {}}},
                "defaultQuant": "f32",
                "pipelineConfig": {
                    "imageWidth": 64,
                    "imageHeight": 96,
                    "imageMean": [0.1, 0.2, 0.3],
                    "imageStd": [0.4, 0.5, 0.6],
                    "interpolation": "bicubic",
                },
            }
        },
    }


class TestDepthAnythingCardGate:
    def test_it_refuses_a_pipeline_it_does_not_describe(self) -> None:
        manifest = _depth_anything_manifest()
        manifest["models"]["small"]["pipeline"] = FOREIGN_PIPELINE
        with pytest.raises(ValueError, match=DEPTH_ANYTHING_SUPPORTED_PIPELINE):
            render_depth_anything_model_card(manifest, REPO)

    def test_it_refuses_a_size_it_cannot_attribute(self) -> None:
        """Rendering a size missing from the attribution table would put CC BY-NC 4.0 weights
        under an Apache-2.0 frontmatter (the MUST of `DEPTH_ANYTHING_UPSTREAM`). It fails with
        the list of known sizes.
        """
        for size in ("base", "large"):
            with pytest.raises(ValueError, match=size):
                render_depth_anything_model_card(_depth_anything_manifest(size), REPO)

    def test_it_renders_the_same_bytes_for_the_same_manifest(self) -> None:
        manifest = _depth_anything_manifest()
        before = copy.deepcopy(manifest)
        assert render_depth_anything_model_card(manifest, REPO) == render_depth_anything_model_card(
            manifest, REPO
        )
        assert manifest == before

    def test_the_attribution_table_only_holds_apache_licensed_sizes(self) -> None:
        """The table of shippable sizes = the licence check (the MUST at the top of
        `depth_anything.distribution`).

        Widening the table goes together with splitting `DEPTH_ANYTHING_LICENSE` (one value
        today) per model, so this is the place that detects that it was widened.
        """
        assert sorted(DEPTH_ANYTHING_UPSTREAM) == ["small"]
        assert DEPTH_ANYTHING_LICENSE == "apache-2.0"

    def test_the_title_comes_from_the_upstream_checkpoint_name(self) -> None:
        """The size in the title is derived from the one attribution table (with two tables a
        card could sell one size as another)."""
        card = render_depth_anything_model_card(_depth_anything_manifest(), REPO)
        assert "# Depth Anything V2 Small — Karume" in card


class TestDepthAnythingEntryPoint:
    """The loading entry point the card advertises — `fromPretrained` only."""

    def test_it_does_not_advertise_the_local_asset_entry_point(self) -> None:
        """`fromAssets` is not advertised (ruling of 2026-08-29).

        The split distribution can now be read as well (X2-101), but that is a surface for local
        debugging that assumes you hold the bytes yourself; the ordinary entry point for readers
        using HF is `fromPretrained`. Showing both would leave readers to decide which one to
        use. `fromPretrained` is checked too so that the check does not pass when the whole Usage
        section disappears.
        """
        card = render_depth_anything_model_card(_depth_anything_manifest(), REPO)
        assert "fromAssets" not in card
        assert "await Pipeline.fromPretrained" in card


class TestDepthAnythingHouseSentence:
    def test_the_title_is_followed_by_the_house_sentence(self) -> None:
        lines = render_depth_anything_model_card(_depth_anything_manifest(), REPO).splitlines()
        title = lines.index("# Depth Anything V2 Small — Karume")
        assert lines[title + 2] == (
            "This is an unofficial conversion of depth-anything/Depth-Anything-V2-Small-hf for"
            " Karume. It is not affiliated with or endorsed by depth-anything."
        )
