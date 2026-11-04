"""Model card rendering of the BiRefNet family distributions (`birefnet.card`).

The real distribution's `karume.json` is not used — the job here is to see that the fake manifest's
values appear in the text as they are (= that not a single hand-written number is mixed in), so
building it with values that **differ** from the real ones detects more. **The exceptions are the
resolution** (`imageWidth` / `imageHeight`) **and the presets**: the measured execution resources
(`BirefnetCheckpoint.resources` per checkpoint) are looked up by resolution, and the card is not
rendered when the measured preset is missing from the manifest, so the fake manifest also carries a
measured resolution and preset.

The derivations from the manifest (tables, numbers, usage) are checked by `TestBirefnetModelCard` /
`TestBirefnetResolutionFamily` in `birefnet/tests/test_distribution.py` over **one full assembly**.
What lives here are the checks only the card side has — that the template is specific to its
pipeline, that it does not render for a checkpoint missing from the attribution table or for an
unmeasured resolution, and the loading entry point it advertises.
"""

from __future__ import annotations

import copy
import re
from typing import Any

import pytest

from _shared.paths import REPO_ROOT
from birefnet.card import (
    BIREFNET_CHECKPOINTS,
    BIREFNET_MATTE_WEIGHTS,
    BIREFNET_POSTPROCESS_DESCRIPTIONS,
    BIREFNET_SUPPORTED_PIPELINE,
    BIREFNET_UPSTREAM,
    render_birefnet_model_card,
)

#: Repository ID spelled in the usage snippet (dist passes it from the pipeline's declaration).
REPO = "hdae/fake-repo"

#: The weights this repository ships (the axis the pipeline entry passes — not in the manifest).
CHECKPOINT = "hr"

#: Default model name = default resolution.
MODEL = "1024"

#: checkpoint → presets of the real distribution (default first — the values of
#: `birefnet.distribution.birefnet_seat_dtypes`; the rule itself is checked by
#: `birefnet/tests/test_distribution.py`). The fake manifest is built with these presets too — the
#: card
#: renders neither a model without the measured preset nor a preset wider than the upstream storage.
SEATS: dict[str, tuple[str, ...]] = {"hr": ("f16",), "lucida": ("f32", "f16")}

#: Another family's pipeline contract (**kept as a literal spelling** — so that no code coupling
#: between recipes is created). That two real templates refuse each other's manifests is checked in
#: both directions by `sbv2/tests/test_card.py`. Here only one direction is needed: "refuse any
#: contract other than its own".
FOREIGN_PIPELINE = "siglip2/1"

#: Declaration of the closed set of post-processing procedures the loader accepts (TS
#: `MATTE_PROCEDURES`).
MATTE_SOURCE = REPO_ROOT / "packages/models/src/birefnet/matte.ts"

#: The one-line declaration of `MATTE_PROCEDURES` (`export const MATTE_PROCEDURES = [...] as
#: const;`).
MATTE_PROCEDURES_DECLARATION = re.compile(
    r"^export const MATTE_PROCEDURES = \[(?P<body>[^\]]*)\] as const;$", re.MULTILINE
)


def _ts_matte_procedures(source: str) -> tuple[str, ...]:
    """Read the procedure names from the TS declaration (fail, rather than silently go empty, when
    its form changes)."""
    declarations = MATTE_PROCEDURES_DECLARATION.findall(source)
    assert len(declarations) == 1, f"cannot read the MATTE_PROCEDURES declaration in {MATTE_SOURCE}"
    (body,) = declarations
    names = tuple(re.findall(r'"([^"]+)"', body))
    rest = re.sub(r'"[^"]+"|,|\s', "", body)
    assert names and rest == "", f"MATTE_PROCEDURES has elements other than string names: {body!r}"
    return names


#: The values read from the TS declaration (a hand-written copy would stay green when only one side
#: moves). Neither a procedure the card cannot describe nor a card describing a procedure the loader
#: cannot read is created.
TS_MATTE_PROCEDURES = _ts_matte_procedures(MATTE_SOURCE.read_text(encoding="utf-8"))


def _ref(path: str, size: int, digit: str) -> dict[str, Any]:
    return {"path": path, "size": size, "sha256": digit * 64}


def _container(*refs: dict[str, Any]) -> dict[str, Any]:
    """One dtype of weights (the `container` of `karume/5` — ADR 0109 decision 3).

    The card does not read `descriptor`, so only its shape follows the real one (lengths and sha256
    of the 2 documents).
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


def _birefnet_manifest(
    model: str = MODEL, seats: tuple[str, ...] = SEATS[CHECKPOINT]
) -> dict[str, Any]:
    """Minimal BiRefNet family manifest (fake values that do not overlap the real ones, except the
    resolution and the presets).

    `seats` are the storage dtypes of the quant presets (default first). The default is the real
    presets of {@link CHECKPOINT} (HR).
    """
    side = int(model)
    return {
        "format": "karume/5",
        "generator": "karume/9.9.9",
        "defaultModel": model,
        "models": {
            model: {
                "pipeline": BIREFNET_SUPPORTED_PIPELINE,
                "weights": {
                    "matte": {
                        dtype: _container(_ref(f"m/model.{dtype}-00001-of-00003.krm", 13, "d"))
                        for dtype in seats
                    }
                },
                "assets": {},
                "quants": {dtype: {"weights": {"matte": dtype}, "session": {}} for dtype in seats},
                "defaultQuant": seats[0],
                "pipelineConfig": {
                    "imageWidth": side,
                    "imageHeight": side,
                    "imageMean": [0.1, 0.2, 0.3],
                    "imageStd": [0.4, 0.5, 0.6],
                    "interpolation": "bilinear",
                    "postprocess": "u8-pil-bicubic",
                },
            }
        },
    }


class TestTheSeatTable:
    def test_it_is_the_distributions_seats_with_the_default_first(self) -> None:
        """{@link SEATS} is a copy of the distribution's presets — no going green with the fake
        manifest built on old presets after the rule (`birefnet_seat_dtypes` /
        `birefnet_default_quant`) moved."""
        from birefnet.distribution import birefnet_default_quant, birefnet_seat_dtypes

        assert set(SEATS) == set(BIREFNET_CHECKPOINTS)
        for checkpoint, seats in SEATS.items():
            assert seats == birefnet_seat_dtypes(checkpoint)
            assert seats[0] == birefnet_default_quant(checkpoint)


class TestBirefnetPostprocess:
    """Description of the post-processing procedure (`pipelineConfig.postprocess`) — the name comes
    from the model's declaration."""

    def test_it_describes_exactly_the_procedures_the_loader_accepts(self) -> None:
        assert sorted(BIREFNET_POSTPROCESS_DESCRIPTIONS) == sorted(TS_MATTE_PROCEDURES)

    def test_every_checkpoint_declares_a_procedure_the_loader_accepts(self) -> None:
        """The procedure the assembly writes is inside the loader's closed set (otherwise the
        distribution cannot be read)."""
        from birefnet.distribution import BIREFNET_POSTPROCESS

        assert set(BIREFNET_POSTPROCESS.values()) <= set(TS_MATTE_PROCEDURES)

    @pytest.mark.parametrize("procedure", TS_MATTE_PROCEDURES)
    def test_it_describes_the_declared_procedure_only(self, procedure: str) -> None:
        manifest = _birefnet_manifest()
        manifest["models"][MODEL]["pipelineConfig"]["postprocess"] = procedure
        card = render_birefnet_model_card(manifest, REPO, CHECKPOINT)
        assert "\n".join(BIREFNET_POSTPROCESS_DESCRIPTIONS[procedure]) in card
        (other,) = (name for name in TS_MATTE_PROCEDURES if name != procedure)
        assert f"`{other}`" not in card

    def test_it_refuses_a_procedure_it_cannot_describe(self) -> None:
        """No extrapolated description — the card would be published disagreeing with the
        distribution's actual post-processing."""
        manifest = _birefnet_manifest()
        manifest["models"][MODEL]["pipelineConfig"]["postprocess"] = "nearest"
        with pytest.raises(ValueError, match="nearest"):
            render_birefnet_model_card(manifest, REPO, CHECKPOINT)

    def test_it_refuses_a_model_without_a_declared_procedure(self) -> None:
        """A missing field fails with the same wording as a procedure without a description (not a
        bare `KeyError`)."""
        manifest = _birefnet_manifest()
        del manifest["models"][MODEL]["pipelineConfig"]["postprocess"]
        with pytest.raises(ValueError, match="no description for post-processing procedure None"):
            render_birefnet_model_card(manifest, REPO, CHECKPOINT)


class TestBirefnetCardGate:
    def test_it_refuses_a_pipeline_it_does_not_describe(self) -> None:
        manifest = _birefnet_manifest()
        manifest["models"][MODEL]["pipeline"] = FOREIGN_PIPELINE
        with pytest.raises(ValueError, match=BIREFNET_SUPPORTED_PIPELINE):
            render_birefnet_model_card(manifest, REPO, CHECKPOINT)

    def test_it_refuses_a_checkpoint_it_cannot_attribute(self) -> None:
        """Rendering for a checkpoint missing from the attribution table would silently produce a
        card without `base_model` (= a redistribution that does not state its origin). It fails
        with the list of known checkpoints.
        """
        with pytest.raises(ValueError, match="tiny"):
            render_birefnet_model_card(_birefnet_manifest(), REPO, "tiny")

    def test_it_refuses_a_resolution_it_has_not_measured(self) -> None:
        """Every resource number the card states is tied to a measurement (no rendering at an
        unmeasured resolution)."""
        manifest = _birefnet_manifest()
        manifest["models"][MODEL]["pipelineConfig"]["imageWidth"] = 512
        with pytest.raises(ValueError, match="512"):
            render_birefnet_model_card(manifest, REPO, CHECKPOINT)

    def test_it_renders_the_same_bytes_for_the_same_manifest(self) -> None:
        manifest = _birefnet_manifest()
        before = copy.deepcopy(manifest)
        assert render_birefnet_model_card(manifest, REPO, CHECKPOINT) == (
            render_birefnet_model_card(manifest, REPO, CHECKPOINT)
        )
        assert manifest == before

    def test_the_title_follows_the_checkpoint(self) -> None:
        """A model upstream markets under a name is called by that name (the heading comes from the
        one attribution table)."""
        assert "# Lucida (BiRefNet) — Karume" in render_birefnet_model_card(
            _birefnet_manifest(seats=SEATS["lucida"]), REPO, "lucida"
        )
        assert "# BiRefNet HR — Karume" in render_birefnet_model_card(
            _birefnet_manifest(), REPO, CHECKPOINT
        )

    def test_the_attribution_does_not_follow_the_model_name(self) -> None:
        """The model name is the resolution, not the attribution axis — with the same model name,
        a different entry (checkpoint) states a different upstream."""
        for checkpoint, repo in BIREFNET_UPSTREAM.items():
            card = render_birefnet_model_card(
                _birefnet_manifest(seats=SEATS[checkpoint]), REPO, checkpoint
            )
            assert f"base_model: {repo}" in card

    def test_the_upstream_table_is_the_only_source_of_the_repository_ids(self) -> None:
        """`BIREFNET_UPSTREAM` is derived from the attribution table (with two tables, one could
        move alone)."""
        assert {
            name: entry.repo for name, entry in BIREFNET_CHECKPOINTS.items()
        } == BIREFNET_UPSTREAM


class TestBirefnetEntryPoint:
    """The loading entry point the card advertises — `fromPretrained` only."""

    def test_it_does_not_advertise_the_local_asset_entry_point(self) -> None:
        """`fromAssets` is not advertised (ruling of 2026-08-29).

        It can read split distributions now (X2-101), but it is a surface for local debugging that
        assumes you hold the bytes yourself; the ordinary entry point for a reader using HF is
        `fromPretrained`. Listing both would leave the reader to decide "which one do I use". The
        `fromPretrained` side is checked as well so that the check does not pass when the whole
        Usage section disappears.
        """
        card = render_birefnet_model_card(_birefnet_manifest(), REPO, CHECKPOINT)
        assert "fromAssets" not in card
        assert "await Pipeline.fromPretrained" in card


#: checkpoint × measured resolution (parameters of the execution-resource note).
_MEASURED = [
    (checkpoint, model)
    for checkpoint, entry in sorted(BIREFNET_CHECKPOINTS.items())
    for model in sorted(entry.resources.by_resolution)
]


class TestBirefnetResourceNote:
    """The execution-resource note (facts a user wants "before handing it over" — constants, since
    they are not in the manifest)."""

    @pytest.mark.parametrize(("checkpoint", "model"), _MEASURED)
    def test_the_card_names_the_gpu_memory_that_resolution_needs(
        self, checkpoint: str, model: str
    ) -> None:
        """MUST: state both the total allocation and the binding above the 128MiB default **from
        that model's measurement**.

        The spec default of WebGPU's `maxStorageBufferBindingSize` is 128MiB, so a larger binding is
        a constraint where "on some devices the request itself fails". If the card stays silent,
        the reader takes an empty `requiredLimits` (= the resident part fits the defaults) to mean
        "runs on default limits".
        """
        card = render_birefnet_model_card(
            _birefnet_manifest(model, SEATS[checkpoint]), REPO, checkpoint
        )
        measurement = BIREFNET_CHECKPOINTS[checkpoint].resources
        resources = measurement.by_resolution[model]

        assert resources.total in card
        assert resources.binding in card
        assert resources.run in card
        assert measurement.conditions in card
        assert "maxStorageBufferBindingSize" in card

    def test_each_checkpoint_measured_the_seat_it_names(self) -> None:
        """The measured preset is per checkpoint — HR on the f16 preset, Lucida on the f32 preset
        (both 2026-10-16)."""
        assert {name: entry.resources.quant for name, entry in BIREFNET_CHECKPOINTS.items()} == {
            "hr": "f16",
            "lucida": "f32",
        }

    @pytest.mark.parametrize("checkpoint", sorted(BIREFNET_CHECKPOINTS))
    def test_the_total_is_named_as_the_measured_seats_number(self, checkpoint: str) -> None:
        """Say which preset the total allocation was measured on (even on Lucida's card with two
        presets, the totals of the other preset are not extrapolated — 2026-09-26 review F2)."""
        card = render_birefnet_model_card(
            _birefnet_manifest(seats=SEATS[checkpoint]), REPO, checkpoint
        )
        prose = " ".join(card.split())
        measurement = BIREFNET_CHECKPOINTS[checkpoint].resources

        assert (
            f"{measurement.by_resolution[MODEL].total} allocated in total with the"
            f" `{measurement.quant}` quant" in prose
        )

    def test_it_refuses_a_model_without_the_measured_seat(self) -> None:
        """A distribution without the measured preset cannot state the total allocation (it would be
        read as another preset's number).

        A manifest with only the f16 preset for Lucida (measured on the f32 preset). HR has no
        renderable preset besides f16 (an f32 preset is wider than the upstream storage and the
        attribution section refuses it — {@link
        TestBirefnetStorageAttribution.test_a_seat_wider_than_the_checkpoint_is_refused}).
        """
        with pytest.raises(
            ValueError, match="the quant 'f32' the execution resources were measured with"
        ):
            render_birefnet_model_card(_birefnet_manifest(seats=("f16",)), REPO, "lucida")

    def test_it_does_not_carry_another_resolutions_measurement(self) -> None:
        """Copying one measurement to all models would make one of the cards state **numbers that
        were not measured**."""
        card = render_birefnet_model_card(_birefnet_manifest(MODEL), REPO, CHECKPOINT)

        assert BIREFNET_CHECKPOINTS[CHECKPOINT].resources.by_resolution["2048"].total not in card

    def test_it_does_not_carry_another_checkpoints_measurement(self) -> None:
        """Measurements are per checkpoint — HR's card states neither Lucida's total allocation nor
        the preset it was measured on.

        The measurement conditions (date and machine) can be the same for both checkpoints, so they
        are not used to tell them apart.
        """
        card = render_birefnet_model_card(_birefnet_manifest(MODEL), REPO, CHECKPOINT)
        prose = " ".join(card.split())
        other = BIREFNET_CHECKPOINTS["lucida"].resources

        assert other.by_resolution[MODEL].total not in card
        assert f"allocated in total with the `{other.quant}` quant" not in prose


def _with_seat(manifest: dict[str, Any], dtype: str, *, default: bool = False) -> dict[str, Any]:
    """Add `dtype` weights and a quant preset of the same name to the default model (made the
    default preset when `default`)."""
    model = manifest["models"][manifest["defaultModel"]]
    model["weights"]["matte"][dtype] = _container(
        _ref(f"m/model.{dtype}-00001-of-00002.krm", 7, "e")
    )
    model["quants"][dtype] = {"weights": {"matte": dtype}, "session": {}}
    if default:
        model["defaultQuant"] = dtype
    return manifest


class TestBirefnetStorageAttribution:
    """Per-preset storage description and `base_model_relation` (ADR 0113 — determined by the
    checkpoint's storage)."""

    def test_the_weights_key_is_the_distributions_role(self) -> None:
        """The card spells the weights key itself to avoid a circular import — it must match the
        distribution recipe."""
        from birefnet.distribution import BIREFNET_ROLE

        assert BIREFNET_MATTE_WEIGHTS == BIREFNET_ROLE

    @pytest.mark.parametrize("checkpoint", sorted(BIREFNET_CHECKPOINTS))
    def test_the_default_seat_declares_no_relation(self, checkpoint: str) -> None:
        """The default preset (the same type as the upstream storage — f16 for HR, f32 for Lucida)
        does not change the checkpoint's values — it is none of the 4 values, so it is left out."""
        card = render_birefnet_model_card(
            _birefnet_manifest(seats=SEATS[checkpoint]), REPO, checkpoint
        )
        assert "base_model_relation" not in card

    def test_an_f16_default_for_the_f32_checkpoint_is_quantized(self) -> None:
        card = render_birefnet_model_card(
            _with_seat(_birefnet_manifest(seats=("f32",)), "f16", default=True), REPO, "lucida"
        )
        assert "base_model_relation: quantized" in card

    def test_the_storage_lines_follow_the_seats_in_the_manifest(self) -> None:
        """dtype labels come from the manifest — an f32-only distribution does not claim an f16
        preset."""
        plain = render_birefnet_model_card(_birefnet_manifest(seats=("f32",)), REPO, "lucida")
        both = render_birefnet_model_card(_birefnet_manifest(seats=SEATS["lucida"]), REPO, "lucida")

        assert "  - `f32`:" in plain and "  - `f16`:" not in plain
        assert "  - `f32`:" in both and "  - `f16`:" in both

    def test_the_hr_card_names_its_f16_checkpoint(self) -> None:
        """HR's only preset (f16) is the upstream values themselves — with one preset, it is stated
        in the singular.

        The old wording "source checkpoint's own f32 values" was not true for HR (upstream is f16).
        """
        card = render_birefnet_model_card(_birefnet_manifest(), REPO, "hr")
        prose = " ".join(card.split())

        assert (
            "**No quantization**: the quant seat stores the source checkpoint's own values" in prose
        )
        assert "  - `f16`: the checkpoint's own f16 values" in card
        assert "`f32`:" not in card
        assert "widened exactly from f16 to f32" not in prose
        assert "own f32 values" not in card

    def test_a_seat_wider_than_the_checkpoint_is_refused(self) -> None:
        """There is no wording for a preset wider than the upstream storage (HR's f32 preset) — a
        preset that only widens the values exactly is not shipped (ADR 0113 addendum 2026-10-11),
        so a card claiming one is not rendered."""
        with pytest.raises(ValueError, match="for an f32 storage preset of an f16 checkpoint"):
            render_birefnet_model_card(_with_seat(_birefnet_manifest(), "f32"), REPO, "hr")

    def test_the_lucida_card_declares_its_f16_seat_quantized(self) -> None:
        card = render_birefnet_model_card(_birefnet_manifest(seats=SEATS["lucida"]), REPO, "lucida")

        assert "**Quantization**" in card
        assert "  - `f16`: **quantized**" in card
        assert "  - `f32`: the checkpoint's own f32 values, unchanged." in card

    def test_a_seat_it_has_no_wording_for_is_refused(self) -> None:
        """Phase 2's i8 is added together with its wording — no extrapolated wording goes into the
        attribution section."""
        with pytest.raises(ValueError, match="i8"):
            render_birefnet_model_card(_with_seat(_birefnet_manifest(), "i8"), REPO, "hr")


class TestBirefnetHouseSentence:
    def test_the_title_is_followed_by_the_house_sentence(self) -> None:
        lines = render_birefnet_model_card(_birefnet_manifest(), REPO, CHECKPOINT).splitlines()
        title = lines.index("# BiRefNet HR — Karume")
        assert lines[title + 2] == (
            "This is an unofficial conversion of ZhengPeng7/BiRefNet_HR for Karume."
            " It is not affiliated with or endorsed by ZhengPeng."
        )

    def test_the_lucida_card_names_the_fine_tune_it_converts(self) -> None:
        card = render_birefnet_model_card(_birefnet_manifest(seats=SEATS["lucida"]), REPO, "lucida")
        lines = card.splitlines()
        title = lines.index("# Lucida (BiRefNet) — Karume")
        assert lines[title + 2] == (
            "This is an unofficial conversion of egeorcun/lucida for Karume."
            " It is not affiliated with or endorsed by egeorcun."
        )
