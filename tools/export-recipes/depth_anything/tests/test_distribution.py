"""The Depth Anything V2 distribution recipe (`depth_anything.distribution`) — unit tests over
one full assembly run.

No real assets are used. The input reaching the assembly is a **valid minimal container** of a
few KB (`ir_fixtures`), and the cases that check that a check rejects something are built from
the same container ({@link _depth_anything_container}). The source of the preprocessing constants
(`preprocessor_config.json`) is a synthetic JSON as well.

Since manifest v2 (`karume/2` — ADR 0041) the layout inside the repository is uniformly "per-model
subtree + `shared/`", so every expected path starts with `<model name>/…`.

The layers observable from the core alone (size limits a synthetic plan can cover, the complete
quant mapping, the staging/swap invariants, the resolution rules of attribution profiles) live in
`tools/exporter/tests/test_dist.py` (the split of ADR 0065 steps 3+4). The per-template checks are
in `depth_anything/tests/test_card.py`.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest
from container_series import i2_container, placed_paths, replace_component, write_component
from ir_fixtures import Shape, ir_container
from upstream_fixture import (
    FIXTURE_REVISION,
    OTHER_REVISION,
    stamp_fixture_provenance,
    write_snapshot,
)

from _shared.licenses import APACHE_LICENSE_2_0_PATH
from _shared.upstream import UpstreamProvenanceError
from depth_anything.card import (
    DEPTH_ANYTHING_CONVT_DIFF,
    DEPTH_ANYTHING_LICENSE,
    DEPTH_ANYTHING_UPSTREAM,
)
from depth_anything.distribution import (
    DEPTH_ANYTHING_DEFAULT_MODEL,
    DEPTH_ANYTHING_OUTPUT_PATHS,
    DEPTH_ANYTHING_ROLE,
    DEPTH_ANYTHING_STORAGE_FORBIDDEN,
    DEPTH_ANYTHING_WEIGHTS,
    PIPELINE,
    DepthAnythingSources,
    depth_anything_checkpoint,
    depth_anything_plan,
    depth_anything_repo_name,
    depth_anything_series_name,
    depth_anything_sources,
)
from dist import default_out_dir, main
from karume.container import CODEC_LEDGER, Provenance
from karume.dist import (
    MANIFEST_FILENAME,
    MODEL_CARD_FILENAME,
    NOTICE_FILENAME,
    DistError,
    assemble_family,
    resolve_card_renderer,
    verify_dist,
)

#: The shape of the provenance the container states when the script (`depth_anything/export.py`)
#: bakes Small.
_SMALL_PROVENANCE = Provenance(
    license=DEPTH_ANYTHING_LICENSE, notice=NOTICE_FILENAME, upstream_revision=FIXTURE_REVISION
)


@pytest.fixture(autouse=True)
def _small_provenance(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the fixture containers state the provenance "the script baked Small" (the passing
    form of the provenance check)."""
    stamp_fixture_provenance(monkeypatch, _SMALL_PROVENANCE)


def _write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _in_subtree(model: str, paths: Iterable[str]) -> list[str]:
    """The expected paths inside the model subtree (the uniform layout of ADR 0041 §9)."""
    return [f"{model}/{rel}" for rel in paths]


def _placed_paths() -> list[str]:
    """The relative paths that appear in the distribution (weights slots expand into numbered
    parts — container-v1 §8)."""
    return placed_paths(DEPTH_ANYTHING_OUTPUT_PATHS, DEPTH_ANYTHING_WEIGHTS)


def _present(out_dir: Path) -> list[str]:
    return sorted(str(path.relative_to(out_dir)) for path in out_dir.rglob("*") if path.is_file())


# ---- Depth Anything V2（depth-estimation）------------------------------------
#
# The fake assets use **numbers that differ from the real ones** (non-square 96×64; mean/std are
# not the ImageNet statistics) — hard-coded preprocessing constants or dimensions would fail. It is
# non-square because a mix-up of `imageWidth` / `imageHeight` cannot in principle be detected on a
# square.

#: The field names of `pipelineConfig` (a copy of `ROOT_KEYS` in the TS-side
#: `packages/models/src/depth-anything/config.ts`). **The loader rejects both unknown and missing
#: keys at parse time**, so the field names of the baking side and the loading side must match
#: exactly.
_DEPTH_ANYTHING_CONFIG_KEYS = (
    "imageWidth",
    "imageHeight",
    "imageMean",
    "imageStd",
    "interpolation",
)

_DEPTH_ANYTHING_HEIGHT = 96
_DEPTH_ANYTHING_WIDTH = 64

#: A fake `preprocessor_config.json` (the same fields as the real one, different values.
#: `keep_aspect_ratio` / `ensure_multiple_of` are included as in the real one — to see that they
#: are not copied into the declaration).
_DEPTH_ANYTHING_PREPROCESSOR: Mapping[str, Any] = {
    "do_normalize": True,
    "do_pad": False,
    "do_rescale": True,
    "do_resize": True,
    "ensure_multiple_of": 14,
    "image_mean": [0.1, 0.2, 0.3],
    "image_processor_type": "DPTImageProcessor",
    "image_std": [0.4, 0.5, 0.6],
    "keep_aspect_ratio": True,
    "resample": 3,
    "rescale_factor": 1.0 / 255.0,
    "size": {"height": _DEPTH_ANYTHING_HEIGHT, "width": _DEPTH_ANYTHING_WIDTH},
}


def _depth_anything_container(
    *,
    storage: str = "f32",
    input_name: str = "pixel_values",
    input_shape: Sequence[Any] | None = None,
    depth_shape: Sequence[Any] | None = None,
    outputs: int = 1,
) -> list[bytes]:
    """The contents of the depth-estimation graph placed in the series (a part sequence — part 0
    first).

    What reaches the assembly must be a **valid container** (the assembly opens its input and
    checks it against every rule of the declaration —
    `karume.dist.assert_weight_components_verified`), so the cases a check must reject are built
    from the same container, with **only the declaration shifted from the real one**.

    `storage` is the storage of a mixed component as in the real thing (in f16 / i8 / i4 the
    ineligible weights and the biases stay f32, and i4 additionally mixes in i8) — the check of
    the forbidden table ({@link DEPTH_ANYTHING_STORAGE_FORBIDDEN}) can only be exercised in this
    form. The remaining arguments are the axes of the graph declaration, defaulting to the real
    values.
    """
    shape: Shape = (
        list(input_shape)
        if input_shape is not None
        else [1, 3, _DEPTH_ANYTHING_HEIGHT, _DEPTH_ANYTHING_WIDTH]
    )
    depth: Shape = list(depth_shape) if depth_shape is not None else [1, shape[2], shape[3]]
    return ir_container(
        mark="depth-anything",
        # The fake series also **names itself by the component name** (container graph name =
        # the manifest's weights key, a MUST of container-v1 §2.1 — the assembly's check looks at
        # this match).
        named=DEPTH_ANYTHING_ROLE,
        storage=storage,
        inputs=((input_name, shape),),
        outputs=[depth] * outputs,
    )


def _build_depth_anything_sources(
    root: Path,
    *,
    model: str = DEPTH_ANYTHING_DEFAULT_MODEL,
    preprocessor: Mapping[str, Any] | None = _DEPTH_ANYTHING_PREPROCESSOR,
    **container: Any,
) -> DepthAnythingSources:
    """Reproduce the series + the location of the real weights with fake assets (including the
    `io.*` files that must not be distributed).

    The arrangement follows the real layout of `_shared.paths` (`outputs/series/` and `inputs/`) —
    so the CLI-path tests can point at the same tree just by swapping the root.
    """
    sources = DepthAnythingSources(
        series=root / "outputs" / "series" / depth_anything_series_name(model),
        model=root / "inputs" / "depth-anything" / depth_anything_checkpoint(model),
    )
    write_component(sources.series / "model.krm", _depth_anything_container(**container))
    # An E2E fixture that must not enter the distribution (the series really does contain these).
    _write(sources.series / "io.ramp.safetensors", b"io-fixture")
    if preprocessor is not None:
        _write(
            sources.model / "preprocessor_config.json",
            json.dumps(preprocessor, ensure_ascii=False).encode("utf-8"),
        )
    # The download record of the checkpoint the preprocessing is read from (what the container's
    # revision is compared with).
    write_snapshot(sources.model, license=DEPTH_ANYTHING_LICENSE)
    return sources


@pytest.fixture
def depth_anything_assembled(tmp_path: Path) -> tuple[Path, dict]:
    sources = _build_depth_anything_sources(tmp_path)
    out_dir = tmp_path / "models" / depth_anything_repo_name(DEPTH_ANYTHING_DEFAULT_MODEL)
    manifest = assemble_family(
        [depth_anything_plan(sources, DEPTH_ANYTHING_DEFAULT_MODEL)],
        out_dir,
        DEPTH_ANYTHING_DEFAULT_MODEL,
        root_files=PIPELINE.root_files,
    )
    return out_dir, manifest


def _depth_anything_model(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    return manifest["models"][DEPTH_ANYTHING_DEFAULT_MODEL]


class TestDepthAnythingLayout:
    def test_it_places_the_single_graph_under_the_model_subtree(
        self, depth_anything_assembled
    ) -> None:
        out_dir, _ = depth_anything_assembled
        expected = _in_subtree(DEPTH_ANYTHING_DEFAULT_MODEL, _placed_paths())
        # The 2 legal texts (Apache 2.0 §4) are meta slots the manifest does not declare.
        assert _present(out_dir) == sorted(
            [*expected, MANIFEST_FILENAME, "LICENSE.md", "NOTICE.md"]
        )

    def test_it_never_carries_the_io_fixtures(self, depth_anything_assembled) -> None:
        out_dir, _ = depth_anything_assembled
        assert list(out_dir.rglob("io.*")) == []

    def test_it_declares_one_graph_and_no_assets(self, depth_anything_assembled) -> None:
        """Running needs only the one graph (no tokenizer and no table = `assets` is empty)."""
        _, manifest = depth_anything_assembled
        model = _depth_anything_model(manifest)
        assert model["pipeline"] == "depth-anything/1"
        assert list(model["weights"]) == [DEPTH_ANYTHING_ROLE]
        assert model["assets"] == {}
        assert list(model["quants"]) == ["f32"]
        assert model["defaultQuant"] == "f32"
        assert model["quants"]["f32"]["weights"] == {DEPTH_ANYTHING_ROLE: "f32"}
        assert model["quants"]["f32"]["session"] == {}

    def test_it_reassembles_over_a_previous_run(self, tmp_path: Path) -> None:
        sources = _build_depth_anything_sources(tmp_path)
        out_dir = tmp_path / "models" / depth_anything_repo_name(DEPTH_ANYTHING_DEFAULT_MODEL)
        first = assemble_family(
            [depth_anything_plan(sources)], out_dir, DEPTH_ANYTHING_DEFAULT_MODEL
        )
        assert first == assemble_family(
            [depth_anything_plan(sources)], out_dir, DEPTH_ANYTHING_DEFAULT_MODEL
        )
        assert verify_dist(out_dir)

    @pytest.mark.parametrize(("storage", "intruder"), [("f16", "f16"), ("i8", "i8"), ("i4", "i4")])
    def test_it_refuses_a_real_compressed_series_in_the_f32_seat(
        self, tmp_path: Path, storage: str, intruder: str
    ) -> None:
        """A compressed series shaped like the real one **satisfies the requirement check** —
        only the forbidden table catches a mix-up of the series root.

        A compressed series also keeps its ineligible weights and biases in F32, so "contains
        F32" holds, and unlike the single-dtype fake assets above the requirement check rejects
        nothing. The reason it must fail is "an asset of another series sits in the f32 slot",
        which a numeric check cannot in principle detect (ADR 0027 / 0029).
        """
        sources = _build_depth_anything_sources(tmp_path, storage=storage)
        with pytest.raises(
            DistError, match=rf"{DEPTH_ANYTHING_ROLE}: storage of .* {intruder} \(actual:"
        ):
            depth_anything_plan(sources)

    def test_the_plain_f32_series_passes_both_storage_gates(self, tmp_path: Path) -> None:
        """The passing counterpart — a plain f32 series passes both the requirement check and the
        forbidden table (the forbidden table has not collapsed into always failing)."""
        sources = _build_depth_anything_sources(tmp_path, storage="f32")

        assert depth_anything_plan(sources).name == DEPTH_ANYTHING_DEFAULT_MODEL

    def test_the_forbidden_table_names_every_compressed_layout_in_the_codec_ledger(self) -> None:
        """The forbidden table holds **every** layout of the storage check's vocabulary (the
        codec ledger's layouts) except f32 / i32.

        If even one is missing, a series root of that storage alone passes the f32 slot silently.
        i32 is plain storage (index tables), so it is not forbidden.
        """
        compressed = {entry.layout for entry in CODEC_LEDGER.values()} - {"f32", "i32"}

        assert set(DEPTH_ANYTHING_STORAGE_FORBIDDEN[DEPTH_ANYTHING_ROLE]) == compressed

    def test_it_refuses_an_i2_series_in_the_f32_seat(self, tmp_path: Path) -> None:
        """i2 (the fixed packed values of a QAT series), which `ir_container` cannot write, also
        fails on the actual binding table."""
        sources = _build_depth_anything_sources(tmp_path)
        replace_component(sources.series / "model.krm", i2_container(named=DEPTH_ANYTHING_ROLE))

        with pytest.raises(DistError, match=rf"{DEPTH_ANYTHING_ROLE}: storage of .* i2 \(actual:"):
            depth_anything_plan(sources)


class TestTheContainerProvenance:
    """The check closes on the provenance the container states (with the name table alone, a
    container baked from Base would pass as Small's series)."""

    def test_it_refuses_a_container_that_names_another_license(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A container baked from Base / Large (CC BY-NC 4.0) placed at Small's series path.

        Storage, input / output shapes and preprocessing have the same shape as Small, so every
        check before the provenance comparison passes.
        """
        stamp_fixture_provenance(
            monkeypatch,
            Provenance(
                license="cc-by-nc-4.0", notice=NOTICE_FILENAME, upstream_revision=FIXTURE_REVISION
            ),
        )
        sources = _build_depth_anything_sources(tmp_path)

        with pytest.raises(DistError, match=r"provenance\.license is 'cc-by-nc-4.0'"):
            depth_anything_plan(sources)

    def test_it_refuses_a_container_without_an_upstream_revision(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A container from the script before it baked the revision (provenance cannot be traced)
        is not shipped."""
        stamp_fixture_provenance(
            monkeypatch, Provenance(license=DEPTH_ANYTHING_LICENSE, notice=NOTICE_FILENAME)
        )
        sources = _build_depth_anything_sources(tmp_path)

        with pytest.raises(DistError, match="has no provenance.upstreamRevision"):
            depth_anything_plan(sources)

    def test_it_refuses_a_container_baked_from_another_revision(self, tmp_path: Path) -> None:
        """The checkpoint the preprocessing is read from and the checkpoint the container was
        baked from have different revisions."""
        sources = _build_depth_anything_sources(tmp_path)
        write_snapshot(sources.model, license=DEPTH_ANYTHING_LICENSE, revision=OTHER_REVISION)

        with pytest.raises(DistError, match="from a different revision"):
            depth_anything_plan(sources)

    def test_it_refuses_a_checkpoint_without_the_download_record(self, tmp_path: Path) -> None:
        """Without a download record the checkpoint offers nothing to compare with — fail rather
        than skip."""
        sources = _build_depth_anything_sources(tmp_path)
        for record in (sources.model / ".cache").rglob("*.metadata"):
            record.unlink()

        with pytest.raises(UpstreamProvenanceError, match=r"\.metadata is missing"):
            depth_anything_plan(sources)


class TestDepthAnythingPipelineConfig:
    """`pipelineConfig` matches the loader's schema (`src/depth-anything/config.ts`) down to the
    field names."""

    def test_it_declares_exactly_the_fields_the_loader_accepts(
        self, depth_anything_assembled
    ) -> None:
        _, manifest = depth_anything_assembled
        assert tuple(_depth_anything_model(manifest)["pipelineConfig"]) == (
            _DEPTH_ANYTHING_CONFIG_KEYS
        )

    def test_it_derives_the_preprocessing_constants_from_the_checkpoint(
        self, depth_anything_assembled
    ) -> None:
        """Hard-coded constants would show the real numbers (518² / the ImageNet statistics)."""
        _, manifest = depth_anything_assembled
        config = _depth_anything_model(manifest)["pipelineConfig"]
        assert config["imageWidth"] == _DEPTH_ANYTHING_WIDTH
        assert config["imageHeight"] == _DEPTH_ANYTHING_HEIGHT
        assert config["imageMean"] == _DEPTH_ANYTHING_PREPROCESSOR["image_mean"]
        assert config["imageStd"] == _DEPTH_ANYTHING_PREPROCESSOR["image_std"]

    def test_it_declares_bicubic_not_the_bilinear_the_other_towers_use(
        self, depth_anything_assembled
    ) -> None:
        """`resample: 3` = PIL's BICUBIC. Mixed up with the bilinear of SigLIP2 / BiRefNet,
        `pixel_values` would be off by 34 uint8 LSBs and still load and run."""
        _, manifest = depth_anything_assembled
        assert _depth_anything_model(manifest)["pipelineConfig"]["interpolation"] == "bicubic"

    def test_it_does_not_carry_the_aspect_ratio_knobs_into_the_declaration(
        self, depth_anything_assembled
    ) -> None:
        """`keep_aspect_ratio` / `ensure_multiple_of` of the upstream config are not copied into
        the declaration.

        The baked graph accepts only one square size, so a path that keeps the ratio has nowhere
        to go. Copied through, they would make the TS-side parse fail on unknown keys (= the
        distribution builds but cannot be loaded).
        """
        _, manifest = depth_anything_assembled
        config = _depth_anything_model(manifest)["pipelineConfig"]
        assert "keepAspectRatio" not in config
        assert "ensureMultipleOf" not in config

    def test_it_refuses_a_missing_preprocessor_config(self, tmp_path: Path) -> None:
        sources = _build_depth_anything_sources(tmp_path, preprocessor=None)
        with pytest.raises(DistError, match="missing assembly input"):
            depth_anything_plan(sources)

    def test_it_refuses_a_checkpoint_that_asks_for_another_interpolation(
        self, tmp_path: Path
    ) -> None:
        """`resample` 2 (BILINEAR) is not silently declared as bicubic."""
        sources = _build_depth_anything_sources(
            tmp_path, preprocessor={**_DEPTH_ANYTHING_PREPROCESSOR, "resample": 2}
        )
        with pytest.raises(DistError, match="resample"):
            depth_anything_plan(sources)

    def test_it_refuses_a_rescale_factor_the_host_cannot_express(self, tmp_path: Path) -> None:
        """The TS side is fixed to dividing 8-bit pixels by 255 (the divisor is not in the
        declaration)."""
        sources = _build_depth_anything_sources(
            tmp_path, preprocessor={**_DEPTH_ANYTHING_PREPROCESSOR, "rescale_factor": 1.0 / 127.5}
        )
        with pytest.raises(DistError, match="rescale_factor"):
            depth_anything_plan(sources)

    @pytest.mark.parametrize("flag", ["do_resize", "do_rescale", "do_normalize"])
    def test_it_refuses_a_checkpoint_that_skips_one_of_the_three_stages(
        self, tmp_path: Path, flag: str
    ) -> None:
        sources = _build_depth_anything_sources(
            tmp_path, preprocessor={**_DEPTH_ANYTHING_PREPROCESSOR, flag: False}
        )
        with pytest.raises(DistError, match=flag):
            depth_anything_plan(sources)

    @pytest.mark.parametrize("flag", ["do_center_crop", "do_pad"])
    def test_it_refuses_a_checkpoint_that_wants_a_crop_or_a_pad(
        self, tmp_path: Path, flag: str
    ) -> None:
        """This preprocessing layer has neither crop nor pad (a single resize)."""
        sources = _build_depth_anything_sources(
            tmp_path, preprocessor={**_DEPTH_ANYTHING_PREPROCESSOR, flag: True}
        )
        with pytest.raises(DistError, match=flag):
            depth_anything_plan(sources)

    @pytest.mark.parametrize(
        "patch",
        [
            {"image_mean": [0.1, 0.2]},
            {"image_std": [0.4, 0.0, 0.6]},
            {"image_std": [0.4, "0.5", 0.6]},
            {"size": {"height": 0, "width": _DEPTH_ANYTHING_WIDTH}},
        ],
    )
    def test_it_refuses_constants_outside_the_loader_value_range(
        self, tmp_path: Path, patch: Mapping[str, Any]
    ) -> None:
        """The value ranges equal the loader's (`config.ts`) — a std of 0 would quietly produce
        ±Infinity through division by zero."""
        sources = _build_depth_anything_sources(
            tmp_path, preprocessor={**_DEPTH_ANYTHING_PREPROCESSOR, **patch}
        )
        with pytest.raises(DistError):
            depth_anything_plan(sources)


class TestDepthAnythingGraphGate:
    """The assembly check — rejects, before placement, combinations that would still form a valid
    distribution despite the mismatch."""

    def test_it_refuses_a_graph_baked_for_another_resolution(self, tmp_path: Path) -> None:
        """A different size has a different pretraining resolution (Small 518² / a combination
        with another size)."""
        sources = _build_depth_anything_sources(tmp_path, input_shape=(1, 3, 518, 518))
        with pytest.raises(
            DistError,
            match="preprocessing dimensions and the baked resolution come from different versions",
        ):
            depth_anything_plan(sources)

    def test_it_refuses_a_graph_whose_axes_are_transposed(self, tmp_path: Path) -> None:
        """A mix-up only non-square dimensions can detect (`[1,3,W,H]`)."""
        sources = _build_depth_anything_sources(
            tmp_path, input_shape=(1, 3, _DEPTH_ANYTHING_WIDTH, _DEPTH_ANYTHING_HEIGHT)
        )
        with pytest.raises(
            DistError,
            match="preprocessing dimensions and the baked resolution come from different versions",
        ):
            depth_anything_plan(sources)

    def test_it_refuses_a_depth_map_with_a_channel_axis(self, tmp_path: Path) -> None:
        """`[1, 1, H, W]` cannot be told apart from `[1, H, W]` by element count (the rank is
        checked too)."""
        sources = _build_depth_anything_sources(
            tmp_path, depth_shape=(1, 1, _DEPTH_ANYTHING_HEIGHT, _DEPTH_ANYTHING_WIDTH)
        )
        with pytest.raises(DistError, match="no channel axis"):
            depth_anything_plan(sources)

    def test_it_refuses_a_graph_with_a_second_output(self, tmp_path: Path) -> None:
        """Another export that also emits intermediate stages would make the later stages that
        look up by position read a different value as the depth."""
        sources = _build_depth_anything_sources(tmp_path, outputs=2)
        with pytest.raises(DistError, match="exactly one depth map"):
            depth_anything_plan(sources)

    def test_it_refuses_a_graph_with_a_symbolic_axis(self, tmp_path: Path) -> None:
        """The resolution and the patch count are fixed, so no axis varies (here the height was
        exported as dynamic)."""
        sources = _build_depth_anything_sources(
            tmp_path, input_shape=(1, 3, "T", _DEPTH_ANYTHING_WIDTH)
        )
        with pytest.raises(DistError, match="symbolic dimensions"):
            depth_anything_plan(sources)

    def test_it_refuses_a_renamed_input(self, tmp_path: Path) -> None:
        """The runtime binds by name, so a changed spelling cannot be bound."""
        sources = _build_depth_anything_sources(tmp_path, input_name="px")
        with pytest.raises(DistError, match="graph inputs"):
            depth_anything_plan(sources)


class TestDepthAnythingModelCard:
    def _run(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        """Run the CLI once with fake assets only (the location of the real weights moves to tmp
        too)."""
        from depth_anything import distribution

        sources = _build_depth_anything_sources(tmp_path)
        # It is the recipe that reads `INPUTS_ROOT` (the real weights outside the series) — so
        # this module's binding is replaced, not the driver `dist`'s (`DIST_ROOT` is the opposite,
        # on the driver side).
        monkeypatch.setattr(distribution, "INPUTS_ROOT", tmp_path / "inputs")
        out_dir = tmp_path / "dist"
        main(
            [
                "--pipeline",
                "depth-anything",
                "--series",
                str(sources.series.parent),
                "--out",
                str(out_dir),
            ]
        )
        return out_dir

    def test_it_describes_the_relative_depth_distribution(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        out_dir = self._run(tmp_path, monkeypatch)
        card = (out_dir / MODEL_CARD_FILENAME).read_text(encoding="utf-8")
        assert "pipeline_tag: depth-estimation" in card
        assert f"license: {DEPTH_ANYTHING_LICENSE}" in card
        assert f"base_model: {DEPTH_ANYTHING_UPSTREAM['small']}" in card
        # A distribution that keeps the storage unchanged is none of the 4 values (left to the
        # Hub's inference).
        assert "base_model_relation" not in card
        # Being relative depth (not metric depth) is the very constraint users check first.
        assert "**Relative depth has no unit and no origin**" in card
        # Preprocessing is included (users pass raw RGB8). The numbers come from the manifest.
        assert f"resized to {_DEPTH_ANYTHING_WIDTH} × {_DEPTH_ANYTHING_HEIGHT} (bicubic" in card
        # The card is rendered from a distribution that **passed verification**.
        assert verify_dist(out_dir)

    def test_the_usage_snippet_enumerates_the_choices_from_the_manifest(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The form of the 2026-08-12 ruling: options appear as comments, and the selectable
        values are enumerated from the manifest.

        `estimate()` has no run-time knobs, so this pipeline's options are only model / quant.
        It is spelled "available:" even with a single slot so that the text follows automatically
        when slots are added (hard-coded text like "the only one this repository ships" becomes a
        lie the moment one is added).
        """
        card = (self._run(tmp_path, monkeypatch) / MODEL_CARD_FILENAME).read_text(encoding="utf-8")
        assert (
            f'  // model: "{DEPTH_ANYTHING_DEFAULT_MODEL}",'
            f" // default — available: {DEPTH_ANYTHING_DEFAULT_MODEL}" in card
        )
        assert '  // quant: "f32", // default — available: f32' in card
        assert "the only one this repository ships" not in card

    def test_it_states_that_only_the_apache_licensed_size_is_redistributed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Base / Large are CC BY-NC 4.0. "What is absent" is the fact readers look for first."""
        card = (self._run(tmp_path, monkeypatch) / MODEL_CARD_FILENAME).read_text(encoding="utf-8")
        assert "Only the Small checkpoint is Apache-2.0." in card
        assert "CC BY-NC 4.0" in card

    def test_the_title_comes_from_the_upstream_checkpoint_name(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The size is spelled from the one attribution table (with two tables a card could sell
        one size as another)."""
        card = (self._run(tmp_path, monkeypatch) / MODEL_CARD_FILENAME).read_text(encoding="utf-8")
        assert "# Depth Anything V2 Small — Karume" in card

    def test_it_points_at_the_bundled_license_and_notice(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        card = (self._run(tmp_path, monkeypatch) / MODEL_CARD_FILENAME).read_text(encoding="utf-8")
        assert "a verbatim copy is in `LICENSE.md`" in card
        assert "also listed in `NOTICE.md`, per Apache 2.0 §4(b)" in card


class TestDepthAnythingLegalText:
    """The Apache 2.0 text and the modification notice at the distribution repository's root
    (ADR 0092 decision 7)."""

    def test_it_ships_the_license_text_byte_identical(self, depth_anything_assembled) -> None:
        """§4(a) — what is provided is **a copy of this licence** (neither a summary nor a
        reformatted version).

        The original is `_shared/licenses/apache_license_2_0.txt`. Reformatting or a newline
        conversion anywhere on the assembly path would move a byte, but the prose would stay
        plausible, so every other check would pass it.
        """
        out_dir, _ = depth_anything_assembled
        assert (out_dir / "LICENSE.md").read_bytes() == APACHE_LICENSE_2_0_PATH.read_bytes()

    def test_it_ships_the_modification_notice(self, depth_anything_assembled) -> None:
        out_dir, _ = depth_anything_assembled
        notice = (out_dir / "NOTICE.md").read_text(encoding="utf-8")
        # The non-bit-identical rewrite and its measured width appear with the same number as in
        # the card.
        assert "pixel shuffle" in notice
        assert DEPTH_ANYTHING_CONVT_DIFF in notice
        assert "No quantization" in notice


class TestDepthAnythingCli:
    def test_the_default_output_directory_is_the_generation_repository(self) -> None:
        """The repository is per generation (ADR 0092 decision 2) — the size does not appear in
        the name."""
        assert (
            default_out_dir(PIPELINE, [DEPTH_ANYTHING_DEFAULT_MODEL]).name
            == "karume-depth-anything-v2"
        )

    def test_one_attribution_profile_needs_no_choice(self) -> None:
        profiles = PIPELINE.card_profiles
        assert len(profiles) == 1
        assert resolve_card_renderer(PIPELINE, None) is next(iter(profiles.values()))

    def test_the_series_name_is_the_lowercased_upstream_repository_name(
        self, tmp_path: Path
    ) -> None:
        """The series name and the input directory name are the upstream repository name itself
        (no second table)."""
        for model, repo in DEPTH_ANYTHING_UPSTREAM.items():
            checkpoint = repo.split("/", 1)[1]
            sources = depth_anything_sources(tmp_path, model)
            assert sources.series.name == checkpoint.lower()
            assert sources.model.name == checkpoint

    @pytest.mark.parametrize("model", ["base", "large"])
    def test_it_refuses_a_size_it_may_not_redistribute(self, tmp_path: Path, model: str) -> None:
        """Sizes that are CC BY-NC 4.0 upstream are not in the attribution table = they fail
        before the series is looked up.

        The script can bake any size via `--model-dir`, so this is the only check that keeps
        things from reading as "whatever was baked can be shipped" (the MUST of the Depth
        Anything section of `karume.dist`).
        """
        with pytest.raises(DistError, match="unknown Depth Anything V2 model"):
            depth_anything_sources(tmp_path, model)

    def test_it_assembles_into_the_pipeline_default_directory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import dist
        from depth_anything import distribution

        sources = _build_depth_anything_sources(tmp_path)
        # `DIST_ROOT` is on the driver side, which decides the default output location
        # (`dist.default_out_dir`); `INPUTS_ROOT` is on the recipe side, which reads the inputs
        # outside the series — the bindings of the two modules are replaced separately.
        monkeypatch.setattr(dist, "DIST_ROOT", tmp_path / "models")
        monkeypatch.setattr(distribution, "INPUTS_ROOT", tmp_path / "inputs")

        main(["--pipeline", "depth-anything", "--series", str(sources.series.parent)])

        out_dir = tmp_path / "models" / depth_anything_repo_name(DEPTH_ANYTHING_DEFAULT_MODEL)
        expected = _in_subtree(DEPTH_ANYTHING_DEFAULT_MODEL, _placed_paths())
        assert sorted(verify_dist(out_dir)) == sorted(expected)
