"""SigLIP2 distribution recipe (`siglip2.distribution`) — unit tests over one assembly pass.

No real assets are used. The input that reaches the assembly is a **valid minimal container** of a
few KB (`ir_fixtures`), and the cases that check a rejection by a check are built in the same
container ({@link _siglip2_container}). Synthetic JSON is enough for the source of the
preprocessing constants (`preprocessor_config.json`) too.

Since manifest v2 (`karume/2` — ADR 0041), the in-repository layout is uniformly "per-model
subtrees + `shared/`", so every expected path starts with `<model name>/…`.

The layers observable from the core alone (the size limits a synthetic plan can cover, the complete
quant mapping, the staging/swap invariants, the resolution rules of attribution profiles) live in
`tools/exporter/tests/test_dist.py` (the split of ADR 0065 steps 3+4). Per-template checks are in
`siglip2/tests/test_card.py`.
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
from siglip2.card import (
    SIGLIP2_LICENSE,
    SIGLIP2_MAP_HEAD_DIFF,
    SIGLIP2_MAP_HEAD_NORM,
    SIGLIP2_UPSTREAM,
)
from siglip2.distribution import (
    PIPELINE,
    SIGLIP2_DEFAULT_MODEL,
    SIGLIP2_OUTPUT_PATHS,
    SIGLIP2_ROLE,
    SIGLIP2_STORAGE_FORBIDDEN,
    SIGLIP2_WEIGHTS,
    Siglip2Sources,
    siglip2_checkpoint,
    siglip2_hidden_dim,
    siglip2_plan,
    siglip2_repo_name,
    siglip2_sources,
)


@pytest.fixture(autouse=True)
def _baked_provenance(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the fixture container claim the provenance "baked by the script" (the passing form of
    the provenance check)."""
    stamp_fixture_provenance(
        monkeypatch,
        Provenance(
            license=SIGLIP2_LICENSE, notice=NOTICE_FILENAME, upstream_revision=FIXTURE_REVISION
        ),
    )


def _write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _in_subtree(model: str, paths: Iterable[str]) -> list[str]:
    """Expected paths inside the model subtree (the uniform layout of ADR 0041 §9)."""
    return [f"{model}/{rel}" for rel in paths]


def _placed_paths() -> list[str]:
    """Relative paths that appear in the distribution (a weights slot expands into numbered parts
    — container-v1 §8)."""
    return placed_paths(SIGLIP2_OUTPUT_PATHS, SIGLIP2_WEIGHTS)


def _present(out_dir: Path) -> list[str]:
    return sorted(str(path.relative_to(out_dir)) for path in out_dir.rglob("*") if path.is_file())


# ---- SigLIP2（image-feature-extraction）--------------------------------------
#
# The fake assets use **numbers different from the real ones** (non-square 96×64, mean/std not
# 0.5, hidden 7) — hard-coded preprocessing constants or dimensions would fail. Non-square because
# a mix-up of `imageWidth` / `imageHeight` cannot in principle be detected with a square.

#: Field names of `pipelineConfig` (a copy of `ROOT_KEYS` in the TS-side
#: `packages/models/src/siglip2/config.ts`). **The loader rejects both unknown and missing keys at
#: parse time**, so the field names of the baking side and the loading side must match exactly.
_SIGLIP2_CONFIG_KEYS = (
    "imageWidth",
    "imageHeight",
    "imageMean",
    "imageStd",
    "hiddenDim",
    "interpolation",
)

_SIGLIP2_HEIGHT = 96
_SIGLIP2_WIDTH = 64
_SIGLIP2_HIDDEN = 7

#: Dimensions of the second model (so400m). **None overlaps base** — that `pipelineConfig` is
#: declared per model in a one-repository, two-model layout can only be observed when the values
#: differ.
_SIGLIP2_SO400M_DIMS = (48, 80, 5)

#: Fake `preprocessor_config.json` (the same fields as the real one, different values).
_SIGLIP2_PREPROCESSOR: Mapping[str, Any] = {
    "do_normalize": True,
    "do_rescale": True,
    "do_resize": True,
    "image_mean": [0.1, 0.2, 0.3],
    "image_std": [0.4, 0.5, 0.6],
    "image_processor_type": "SiglipImageProcessor",
    "resample": 2,
    "rescale_factor": 1.0 / 255.0,
    "size": {"height": _SIGLIP2_HEIGHT, "width": _SIGLIP2_WIDTH},
}


def _siglip2_preprocessor(height: int, width: int) -> dict[str, Any]:
    """`preprocessor_config.json` with the same fields and different dimensions (the second
    model's input)."""
    return {**_SIGLIP2_PREPROCESSOR, "size": {"height": height, "width": width}}


def _siglip2_graph(
    *,
    shape: Sequence[Any] = (1, 3, _SIGLIP2_HEIGHT, _SIGLIP2_WIDTH),
    hidden: int = _SIGLIP2_HIDDEN,
    name: str = "pixel_values",
    outputs: int = 1,
    symbols: Sequence[str] = (),
) -> str:
    """Minimal IR metadata the checks read (one input, the output declaration, symbolic
    dimensions)."""
    names = [f"out_{index}" for index in range(outputs)]
    return json.dumps(
        {
            "inputs": [{"name": name, "shape": list(shape)}],
            "outputs": names,
            "values": {output: {"dtype": "f32", "shape": [1, hidden]} for output in names},
            "symbols": list(symbols),
        }
    )


def _siglip2_container(
    *,
    storage: str = "f32",
    dims: tuple[int, int, int] = (_SIGLIP2_HEIGHT, _SIGLIP2_WIDTH, _SIGLIP2_HIDDEN),
    input_name: str = "pixel_values",
    input_shape: Sequence[Any] | None = None,
    outputs: int = 1,
) -> list[bytes]:
    """Contents of the vision tower placed in the series (a part sequence — part 0 first).

    What reaches the assembly must be a **valid container** (the assembly opens its input and
    checks it against every rule of the declaration —
    `karume.dist.assert_weight_components_verified`), so the cases meant to fail a check are built
    in the same container too and **only the declaration is shifted from the real thing**.

    `storage` is the storage form of a mixed component as in the real thing (for f16 / i8 / i4 the
    ineligible weights and the biases stay f32, and i4 additionally mixes in i8) — the check of the
    forbidden table ({@link SIGLIP2_STORAGE_FORBIDDEN}) can only be exercised in this form. The
    remaining arguments are the axes of the graph declaration, defaulting to the real values.
    """
    height, width, hidden = dims
    shape: Shape = list(input_shape) if input_shape is not None else [1, 3, height, width]
    return ir_container(
        mark="siglip2-vision",
        # The fake series also **names itself by the component name** (container graph name =
        # the manifest's weights key, a MUST of container-v1 §2.1 — the assembly check verifies
        # that they match).
        named=SIGLIP2_ROLE,
        storage=storage,
        inputs=((input_name, shape),),
        outputs=[[1, hidden]] * outputs,
    )


def _build_siglip2_sources(
    root: Path,
    *,
    model: str = SIGLIP2_DEFAULT_MODEL,
    preprocessor: Mapping[str, Any] | None = _SIGLIP2_PREPROCESSOR,
    dims: tuple[int, int, int] = (_SIGLIP2_HEIGHT, _SIGLIP2_WIDTH, _SIGLIP2_HIDDEN),
    **container: Any,
) -> Siglip2Sources:
    """Reproduce the series + the real-weight location with fake assets (including stray `io.*`
    files that are not distributed).

    The arrangement follows the real layout of `_shared.paths` (`outputs/series/` and `inputs/`) —
    so the CLI-path tests can point at the same tree just by swapping the root.
    """
    checkpoint = siglip2_checkpoint(model)
    sources = Siglip2Sources(
        series=root / "outputs" / "series" / checkpoint,
        model=root / "inputs" / "siglip2" / checkpoint,
    )
    write_component(sources.series / "model.krm", _siglip2_container(dims=dims, **container))
    # E2E fixture that must not enter the distribution (the series really does contain these).
    _write(sources.series / "io.ramp.safetensors", b"io-fixture")
    if preprocessor is not None:
        _write(
            sources.model / "preprocessor_config.json",
            json.dumps(preprocessor, ensure_ascii=False).encode("utf-8"),
        )
    # Download record of the checkpoint the preprocessing is read from (what the container's
    # revision is compared with).
    write_snapshot(sources.model, license=SIGLIP2_LICENSE)
    return sources


#: Spelling of the second model (the other one listed in the attribution table).
_SIGLIP2_SECOND_MODEL = "so400m"


def _build_siglip2_family(root: Path) -> None:
    """Lay out the inputs of one repository with two models (base + so400m) — no dimension
    overlaps."""
    _build_siglip2_sources(root)
    _build_siglip2_sources(
        root,
        model=_SIGLIP2_SECOND_MODEL,
        dims=_SIGLIP2_SO400M_DIMS,
        preprocessor=_siglip2_preprocessor(*_SIGLIP2_SO400M_DIMS[:2]),
    )


@pytest.fixture
def siglip2_assembled(tmp_path: Path) -> tuple[Path, dict]:
    sources = _build_siglip2_sources(tmp_path)
    out_dir = tmp_path / "models" / siglip2_repo_name(SIGLIP2_DEFAULT_MODEL)
    manifest = assemble_family(
        [siglip2_plan(sources, SIGLIP2_DEFAULT_MODEL)],
        out_dir,
        SIGLIP2_DEFAULT_MODEL,
        root_files=PIPELINE.root_files,
    )
    return out_dir, manifest


def _siglip2_model(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    return manifest["models"][SIGLIP2_DEFAULT_MODEL]


class TestSiglip2Layout:
    def test_it_places_the_single_graph_under_the_model_subtree(self, siglip2_assembled) -> None:
        out_dir, _ = siglip2_assembled
        expected = _in_subtree(SIGLIP2_DEFAULT_MODEL, _placed_paths())
        # The two legal texts (Apache 2.0 §4) are meta files the manifest does not declare.
        assert _present(out_dir) == sorted(
            [*expected, MANIFEST_FILENAME, "LICENSE.md", "NOTICE.md"]
        )

    def test_it_never_carries_the_io_fixtures(self, siglip2_assembled) -> None:
        out_dir, _ = siglip2_assembled
        assert list(out_dir.rglob("io.*")) == []

    def test_it_declares_one_graph_and_no_assets(self, siglip2_assembled) -> None:
        """Running needs only the one graph (no tokenizer, no table = `assets` is empty)."""
        _, manifest = siglip2_assembled
        model = _siglip2_model(manifest)
        assert model["pipeline"] == "siglip2/1"
        assert list(model["weights"]) == [SIGLIP2_ROLE]
        assert model["assets"] == {}
        # One slot (a single f32 series). The dtype labels are filled into a complete mapping.
        assert list(model["quants"]) == ["f32"]
        assert model["defaultQuant"] == "f32"
        assert model["quants"]["f32"]["weights"] == {SIGLIP2_ROLE: "f32"}
        assert model["quants"]["f32"]["session"] == {}

    def test_it_reassembles_over_a_previous_run(self, tmp_path: Path) -> None:
        sources = _build_siglip2_sources(tmp_path)
        out_dir = tmp_path / "models" / siglip2_repo_name(SIGLIP2_DEFAULT_MODEL)
        first = assemble_family([siglip2_plan(sources)], out_dir, SIGLIP2_DEFAULT_MODEL)
        assert first == assemble_family([siglip2_plan(sources)], out_dir, SIGLIP2_DEFAULT_MODEL)
        assert verify_dist(out_dir)

    @pytest.mark.parametrize(("storage", "intruder"), [("f16", "f16"), ("i8", "i8"), ("i4", "i4")])
    def test_it_refuses_a_real_compressed_series_in_the_f32_seat(
        self, tmp_path: Path, storage: str, intruder: str
    ) -> None:
        """A realistic compressed series **satisfies the requirement check** — only the forbidden
        table catches a mixed-up series root.

        A compressed series also keeps its ineligible weights and biases in F32, so "contains F32"
        is true and, unlike the single-dtype fake assets above, the requirement check rejects
        nothing. The reason it must fail is "an asset of another series sits in the f32 slot",
        which the numeric checks cannot detect in principle (ADR 0027 / 0029).
        """
        sources = _build_siglip2_sources(tmp_path, storage=storage)
        with pytest.raises(DistError, match=rf"{SIGLIP2_ROLE}: .* has .*{intruder} \(actual"):
            siglip2_plan(sources)

    def test_the_plain_f32_series_passes_both_storage_gates(self, tmp_path: Path) -> None:
        """Passing counterpart — a plain f32 series passes both the requirement check and the
        forbidden table (the forbidden table has not degenerated into always failing)."""
        sources = _build_siglip2_sources(tmp_path, storage="f32")

        assert siglip2_plan(sources).name == SIGLIP2_DEFAULT_MODEL

    def test_the_forbidden_table_names_every_compressed_layout_in_the_codec_ledger(self) -> None:
        """The forbidden table holds **all** of the storage check's vocabulary (the layouts of the
        codec ledger) except f32 / i32.

        If even one is missing, a series root in that storage form alone passes the f32 slot
        silently. i32 is plain storage (index tables), so it is not forbidden.
        """
        compressed = {entry.layout for entry in CODEC_LEDGER.values()} - {"f32", "i32"}

        assert set(SIGLIP2_STORAGE_FORBIDDEN[SIGLIP2_ROLE]) == compressed

    def test_it_refuses_an_i2_series_in_the_f32_seat(self, tmp_path: Path) -> None:
        """i2 (the fixed packed values of a QAT series), which `ir_container` cannot write, also
        fails on the actual binding table."""
        sources = _build_siglip2_sources(tmp_path)
        replace_component(sources.series / "model.krm", i2_container(named=SIGLIP2_ROLE))

        with pytest.raises(DistError, match=rf"{SIGLIP2_ROLE}: .* has .*i2 \(actual"):
            siglip2_plan(sources)


class TestSiglip2ContainerProvenance:
    """Compare the provenance the container claims with the attribution table and the local
    checkpoint (the same shape as depth_anything)."""

    def test_it_refuses_a_container_that_names_another_license(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        stamp_fixture_provenance(
            monkeypatch,
            Provenance(license="mit", notice=NOTICE_FILENAME, upstream_revision=FIXTURE_REVISION),
        )
        sources = _build_siglip2_sources(tmp_path)

        with pytest.raises(DistError, match=r"provenance\.license is 'mit'"):
            siglip2_plan(sources)

    def test_it_refuses_a_container_without_an_upstream_revision(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        stamp_fixture_provenance(
            monkeypatch, Provenance(license=SIGLIP2_LICENSE, notice=NOTICE_FILENAME)
        )
        sources = _build_siglip2_sources(tmp_path)

        with pytest.raises(DistError, match=r"has no provenance\.upstreamRevision"):
            siglip2_plan(sources)

    def test_it_refuses_a_container_baked_from_another_revision(self, tmp_path: Path) -> None:
        sources = _build_siglip2_sources(tmp_path)
        write_snapshot(sources.model, license=SIGLIP2_LICENSE, revision=OTHER_REVISION)

        with pytest.raises(DistError, match="built from a different revision"):
            siglip2_plan(sources)


class TestSiglip2PipelineConfig:
    """`pipelineConfig` matches the loader's schema (`src/siglip2/config.ts`) down to the field
    names."""

    def test_it_declares_exactly_the_fields_the_loader_accepts(self, siglip2_assembled) -> None:
        _, manifest = siglip2_assembled
        assert tuple(_siglip2_model(manifest)["pipelineConfig"]) == _SIGLIP2_CONFIG_KEYS

    def test_it_derives_the_preprocessing_constants_from_the_checkpoint(
        self, siglip2_assembled
    ) -> None:
        """If the values were hard-coded, the real numbers (224 / 384, mean = std = 0.5) would
        come out."""
        _, manifest = siglip2_assembled
        config = _siglip2_model(manifest)["pipelineConfig"]
        assert config["imageWidth"] == _SIGLIP2_WIDTH
        assert config["imageHeight"] == _SIGLIP2_HEIGHT
        assert config["imageMean"] == _SIGLIP2_PREPROCESSOR["image_mean"]
        assert config["imageStd"] == _SIGLIP2_PREPROCESSOR["image_std"]
        assert config["interpolation"] == "bilinear"

    def test_it_derives_the_hidden_width_from_the_exported_graph(self, siglip2_assembled) -> None:
        """The only source of `hiddenDim` is the graph's output declaration (config.json does not
        hold the width)."""
        _, manifest = siglip2_assembled
        assert _siglip2_model(manifest)["pipelineConfig"]["hiddenDim"] == _SIGLIP2_HIDDEN

    def test_it_refuses_a_missing_preprocessor_config(self, tmp_path: Path) -> None:
        sources = _build_siglip2_sources(tmp_path, preprocessor=None)
        with pytest.raises(DistError, match="missing assembly input"):
            siglip2_plan(sources)

    def test_it_refuses_a_checkpoint_that_asks_for_another_interpolation(
        self, tmp_path: Path
    ) -> None:
        """`resample` 3 (BICUBIC) has no implementation on the TS side — do not silently let it
        through as bilinear."""
        sources = _build_siglip2_sources(
            tmp_path, preprocessor={**_SIGLIP2_PREPROCESSOR, "resample": 3}
        )
        with pytest.raises(DistError, match="resample"):
            siglip2_plan(sources)

    def test_it_refuses_a_rescale_factor_the_host_cannot_express(self, tmp_path: Path) -> None:
        """The TS side is fixed to dividing 8-bit pixels by 255 (the divisor is not in the
        declaration)."""
        sources = _build_siglip2_sources(
            tmp_path, preprocessor={**_SIGLIP2_PREPROCESSOR, "rescale_factor": 1.0 / 127.5}
        )
        with pytest.raises(DistError, match="rescale_factor"):
            siglip2_plan(sources)

    @pytest.mark.parametrize("flag", ["do_resize", "do_rescale", "do_normalize"])
    def test_it_refuses_a_checkpoint_that_skips_one_of_the_three_stages(
        self, tmp_path: Path, flag: str
    ) -> None:
        sources = _build_siglip2_sources(
            tmp_path, preprocessor={**_SIGLIP2_PREPROCESSOR, flag: False}
        )
        with pytest.raises(DistError, match=flag):
            siglip2_plan(sources)

    @pytest.mark.parametrize("flag", ["do_center_crop", "do_pad"])
    def test_it_refuses_a_checkpoint_that_wants_a_crop_or_a_pad(
        self, tmp_path: Path, flag: str
    ) -> None:
        """This preprocessing layer has neither crop nor pad (a single stretch that does not keep
        the aspect ratio)."""
        sources = _build_siglip2_sources(
            tmp_path, preprocessor={**_SIGLIP2_PREPROCESSOR, flag: True}
        )
        with pytest.raises(DistError, match=flag):
            siglip2_plan(sources)

    @pytest.mark.parametrize(
        "patch",
        [
            {"image_mean": [0.1, 0.2]},
            {"image_std": [0.4, 0.0, 0.6]},
            {"image_std": [0.4, "0.5", 0.6]},
            {"size": {"height": 0, "width": _SIGLIP2_WIDTH}},
        ],
    )
    def test_it_refuses_constants_outside_the_loader_value_range(
        self, tmp_path: Path, patch: Mapping[str, Any]
    ) -> None:
        """The value range is the same as the loader's (`config.ts`) — std 0 silently produces
        ±Infinity through division by zero."""
        sources = _build_siglip2_sources(tmp_path, preprocessor={**_SIGLIP2_PREPROCESSOR, **patch})
        with pytest.raises(DistError):
            siglip2_plan(sources)


class TestSiglip2GraphGate:
    """Assembly check — rejects, before placement, combinations that would still form a valid
    distribution despite the mismatch."""

    def test_it_refuses_a_graph_baked_for_another_resolution(self, tmp_path: Path) -> None:
        """The preprocessing dimensions (preprocessor_config) and the baked resolution are decided
        separately.

        Without this check, pairing base's preprocessing config with so400m's graph would still
        form a distribution, and the Session's shape check would fail on the user's machine
        without saying "which one is right".
        """
        sources = _build_siglip2_sources(tmp_path, input_shape=(1, 3, 384, 384))
        with pytest.raises(
            DistError,
            match="preprocessing dimensions and the baked resolution come from different versions",
        ):
            siglip2_plan(sources)

    def test_it_refuses_a_graph_whose_axes_are_transposed(self, tmp_path: Path) -> None:
        """A mix-up only non-square dimensions can detect (`[1,3,W,H]`)."""
        sources = _build_siglip2_sources(
            tmp_path, input_shape=(1, 3, _SIGLIP2_WIDTH, _SIGLIP2_HEIGHT)
        )
        with pytest.raises(
            DistError,
            match="preprocessing dimensions and the baked resolution come from different versions",
        ):
            siglip2_plan(sources)

    def test_it_refuses_a_graph_with_a_second_output(self, tmp_path: Path) -> None:
        """Another export that includes `last_hidden_state` changes even the source of
        `hiddenDim`."""
        sources = _build_siglip2_sources(tmp_path, outputs=2)
        with pytest.raises(DistError, match="exactly one pooler_output"):
            siglip2_plan(sources)

    def test_it_refuses_a_graph_with_a_symbolic_axis(self, tmp_path: Path) -> None:
        """The resolution and the patch count are fixed, so no axis varies at all (the form of a
        height exported as dynamic)."""
        sources = _build_siglip2_sources(tmp_path, input_shape=(1, 3, "T", _SIGLIP2_WIDTH))
        with pytest.raises(DistError, match="symbolic dimensions"):
            siglip2_plan(sources)

    def test_it_refuses_a_renamed_input(self, tmp_path: Path) -> None:
        """The runtime binds by name, so a changed spelling cannot be bound."""
        sources = _build_siglip2_sources(tmp_path, input_name="pixels")
        with pytest.raises(DistError, match="graph inputs"):
            siglip2_plan(sources)


class TestSiglip2ModelCard:
    def _run(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, model: str) -> Path:
        """Run the CLI once on fake assets only (the real-weight location is moved to tmp as
        well)."""
        from siglip2 import distribution

        sources = _build_siglip2_sources(tmp_path, model=model)
        # The recipe side resolves `INPUTS_ROOT` (the real weights outside the series) — so this
        # module's binding is replaced, not the driver `dist`'s (`DIST_ROOT` is the reverse, on
        # the driver side).
        monkeypatch.setattr(distribution, "INPUTS_ROOT", tmp_path / "inputs")
        out_dir = tmp_path / "dist"
        main(
            [
                "--pipeline",
                "siglip2",
                "--model",
                model,
                "--series",
                str(sources.series.parent),
                "--out",
                str(out_dir),
            ]
        )
        return out_dir

    def test_it_describes_the_image_embedding_distribution(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        out_dir = self._run(tmp_path, monkeypatch, SIGLIP2_DEFAULT_MODEL)
        card = (out_dir / MODEL_CARD_FILENAME).read_text(encoding="utf-8")
        assert "pipeline_tag: image-feature-extraction" in card
        assert "license: apache-2.0" in card
        # A distribution that keeps the storage form is none of the 4 values (left to the Hub's
        # inference).
        assert "base_model_relation" not in card
        # Lacking the text tower is exactly the constraint a user wants to check first.
        assert "The text tower is not here." in card
        # Preprocessing is included (the user passes raw RGB8). The numbers come from the
        # manifest.
        assert f"resizes to {_SIGLIP2_WIDTH} × {_SIGLIP2_HEIGHT}" in card
        assert f"`pooler_output`, {_SIGLIP2_HIDDEN} f32 values, not L2-normalized" in card
        # The card is rendered from a distribution **that passed verification**.
        assert verify_dist(out_dir)

    def test_the_usage_snippet_enumerates_the_choices_from_the_manifest(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The form of the 2026-08-12 ruling: optionals as comments, with the selectable values
        enumerated from the manifest.

        `embed()` has no run-time knobs, so this pipeline's optionals are only model / quant. It
        says "available:" even with a single slot so the text follows automatically when slots
        are added (a hard-coded phrase like "the only one this repository ships" becomes false the
        moment one is added).
        """
        out_dir = self._run(tmp_path, monkeypatch, SIGLIP2_DEFAULT_MODEL)
        card = (out_dir / MODEL_CARD_FILENAME).read_text(encoding="utf-8")
        assert (
            f'  // model: "{SIGLIP2_DEFAULT_MODEL}",'
            f" // default — available: {SIGLIP2_DEFAULT_MODEL}" in card
        )
        assert '  // quant: "f32", // default — available: f32' in card
        assert "the only one this repository ships" not in card

    def test_the_attribution_follows_the_model_name(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Attribution is determined uniquely by the model name (not an axis to choose = it cannot
        be mixed up)."""
        card = (self._run(tmp_path, monkeypatch, "so400m") / MODEL_CARD_FILENAME).read_text(
            encoding="utf-8"
        )
        assert f"base_model: {SIGLIP2_UPSTREAM['so400m']}" in card
        assert SIGLIP2_UPSTREAM["base"] not in card


class TestSiglip2Family:
    """One repository, two models (ADR 0092 decision 8) — base and so400m live in the same
    repository."""

    def _run(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        from siglip2 import distribution

        _build_siglip2_family(tmp_path)
        monkeypatch.setattr(distribution, "INPUTS_ROOT", tmp_path / "inputs")
        out_dir = tmp_path / "models" / siglip2_repo_name(SIGLIP2_DEFAULT_MODEL)
        main(
            [
                "--pipeline",
                "siglip2",
                "--model",
                SIGLIP2_DEFAULT_MODEL,
                "--model",
                _SIGLIP2_SECOND_MODEL,
                "--series",
                str(tmp_path / "outputs" / "series"),
                "--out",
                str(out_dir),
            ]
        )
        return out_dir

    def test_both_models_live_in_one_repository_with_base_as_the_default(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The default is **the first `--model`** (`karume.dist.main`) — the order of the
        arguments decides the default."""
        out_dir = self._run(tmp_path, monkeypatch)
        manifest = json.loads((out_dir / MANIFEST_FILENAME).read_text(encoding="utf-8"))
        assert sorted(manifest["models"]) == sorted([SIGLIP2_DEFAULT_MODEL, _SIGLIP2_SECOND_MODEL])
        assert manifest["defaultModel"] == SIGLIP2_DEFAULT_MODEL
        assert out_dir.name == "karume-siglip2"

    def test_each_model_keeps_its_own_preprocessing_declaration(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Even living together the dimensions are per model — with one shared declaration, one
        of them would get the other's numbers."""
        out_dir = self._run(tmp_path, monkeypatch)
        manifest = json.loads((out_dir / MANIFEST_FILENAME).read_text(encoding="utf-8"))
        base = manifest["models"][SIGLIP2_DEFAULT_MODEL]["pipelineConfig"]
        so400m = manifest["models"][_SIGLIP2_SECOND_MODEL]["pipelineConfig"]
        assert (base["imageHeight"], base["imageWidth"], base["hiddenDim"]) == (
            _SIGLIP2_HEIGHT,
            _SIGLIP2_WIDTH,
            _SIGLIP2_HIDDEN,
        )
        assert (so400m["imageHeight"], so400m["imageWidth"], so400m["hiddenDim"]) == (
            _SIGLIP2_SO400M_DIMS
        )

    def test_the_card_attributes_both_upstream_checkpoints(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Dropping even one redistributed upstream gives a card that does not name its
        source."""
        card = (self._run(tmp_path, monkeypatch) / MODEL_CARD_FILENAME).read_text(encoding="utf-8")
        for repo in SIGLIP2_UPSTREAM.values():
            assert repo in card
        models = " / ".join(sorted([SIGLIP2_DEFAULT_MODEL, _SIGLIP2_SECOND_MODEL]))
        assert f'  // model: "{SIGLIP2_DEFAULT_MODEL}", // default — available: {models}' in card

    def test_the_layout_keeps_each_models_graph_in_its_own_subtree(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Part 0 (the part carrying the graph descriptor) declares the dimensions, so it **must
        not be folded into a shared slot**.

        Byte-identical components are folded into `shared/` (ADR 0041 §9), so the expectation is
        "the per-model subtrees + shared, nothing else" — with synthetic assets the weight parts
        happen to match, but if part 0, whose dimensions differ, were folded, that would be a bug
        on the container side.
        """
        out_dir = self._run(tmp_path, monkeypatch)
        placed = sorted(verify_dist(out_dir))
        graph_part = _placed_paths()[0]
        assert f"{SIGLIP2_DEFAULT_MODEL}/{graph_part}" in placed
        assert f"{_SIGLIP2_SECOND_MODEL}/{graph_part}" in placed
        assert {path.split("/", 1)[0] for path in placed} <= {
            SIGLIP2_DEFAULT_MODEL,
            _SIGLIP2_SECOND_MODEL,
            "shared",
        }


class TestSiglip2LegalText:
    """The Apache 2.0 full text and the modification notice at the distribution repository's root
    (ADR 0092 decision 7)."""

    def test_it_ships_the_license_text_byte_identical(self, siglip2_assembled) -> None:
        """§4(a) — what is provided is **a copy of this license** (neither a summary nor a
        reformatting).

        The original is `_shared/licenses/apache_license_2_0.txt`. Reformatting or newline
        conversion anywhere on the assembly path would move a byte, but it would still read as
        valid prose, so the other checks would let it through.
        """
        out_dir, _ = siglip2_assembled
        assert (out_dir / "LICENSE.md").read_bytes() == APACHE_LICENSE_2_0_PATH.read_bytes()

    def test_it_ships_the_modification_notice(self, siglip2_assembled) -> None:
        out_dir, _ = siglip2_assembled
        notice = (out_dir / "NOTICE.md").read_text(encoding="utf-8")
        # The notice states "the text tower is not included" and the measured range of the
        # rewrite that is not bit-identical.
        assert "vision tower only" in notice
        assert SIGLIP2_MAP_HEAD_DIFF in notice
        # The scale that gives the difference its meaning (the L2 norm) also states the same
        # single word as the card.
        assert f"L2 norm is {SIGLIP2_MAP_HEAD_NORM}" in notice
        assert "No quantization" in notice


class TestSiglip2Cli:
    def test_the_default_output_directory_is_the_family_repository(self) -> None:
        """One repository per family (ADR 0092 decision 1) — a single destination that ignores the
        model name."""
        assert default_out_dir(PIPELINE, [SIGLIP2_DEFAULT_MODEL]).name == "karume-siglip2"
        assert default_out_dir(PIPELINE, [_SIGLIP2_SECOND_MODEL]).name == "karume-siglip2"

    def test_one_attribution_profile_needs_no_choice(self) -> None:
        profiles = PIPELINE.card_profiles
        assert len(profiles) == 1
        assert resolve_card_renderer(PIPELINE, None) is next(iter(profiles.values()))

    def test_the_series_name_is_the_upstream_repository_name(self, tmp_path: Path) -> None:
        """The series name and the input directory name are the upstream repository name itself
        (no second table)."""
        for model, repo in SIGLIP2_UPSTREAM.items():
            sources = siglip2_sources(tmp_path, model)
            assert sources.series.name == repo.split("/", 1)[1]
            assert sources.model.name == sources.series.name

    def test_it_refuses_a_model_it_has_no_attribution_for(self, tmp_path: Path) -> None:
        """A model name missing from the attribution table "cannot name its source", so it fails
        before the series is looked up."""
        with pytest.raises(DistError, match="unknown SigLIP2 model 'large'"):
            siglip2_sources(tmp_path, "large")

    def test_it_assembles_into_the_pipeline_default_directory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import dist
        from siglip2 import distribution

        sources = _build_siglip2_sources(tmp_path)
        # `DIST_ROOT` is on the driver side, which decides the default output directory
        # (`dist.default_out_dir`); `INPUTS_ROOT` is on the recipe side, which resolves the inputs
        # outside the series — the bindings of the two modules are replaced separately.
        monkeypatch.setattr(dist, "DIST_ROOT", tmp_path / "models")
        monkeypatch.setattr(distribution, "INPUTS_ROOT", tmp_path / "inputs")

        main(["--pipeline", "siglip2", "--series", str(sources.series.parent)])

        out_dir = tmp_path / "models" / siglip2_repo_name(SIGLIP2_DEFAULT_MODEL)
        expected = _in_subtree(SIGLIP2_DEFAULT_MODEL, _placed_paths())
        assert sorted(verify_dist(out_dir)) == sorted(expected)


class TestSiglip2HiddenDim:
    """It stops with `DistError` when the **structure** of the output declaration is broken.

    The width affects both `hiddenSize` in `pipelineConfig` and the shape of the weights, so if
    this were loose, a distribution would be assembled in which "only the width silently became a
    number with another meaning".
    """

    @staticmethod
    def _graph(shape: object, outputs: int = 1) -> dict:
        names = [f"out_{index}" for index in range(outputs)]
        return {
            "outputs": names,
            "values": {name: {"dtype": "f32", "shape": shape} for name in names},
        }

    def test_the_reference_declaration_gives_the_width(self) -> None:
        assert siglip2_hidden_dim(self._graph([1, 768]), Path("g")) == 768

    @pytest.mark.parametrize("shape", [[768], [2, 768], [1, 768, 1], "1,768"])
    def test_a_shape_that_is_not_one_by_hidden_is_rejected(self, shape: object) -> None:
        with pytest.raises(DistError, match=r"is not \[1, hidden\]"):
            siglip2_hidden_dim(self._graph(shape), Path("g"))

    @pytest.mark.parametrize("hidden", [True, 0, -768, 768.0, "768"])
    def test_a_width_that_is_not_a_positive_integer_is_rejected(self, hidden: object) -> None:
        """`True` is a subtype of `int` in Python — with a loose type check a bool would pass as
        the width."""
        with pytest.raises(DistError, match="is not a positive integer"):
            siglip2_hidden_dim(self._graph([1, hidden]), Path("g"))

    def test_more_than_one_output_is_rejected(self) -> None:
        """If another export that includes `last_hidden_state` slips in, the width becomes a
        number with another meaning."""
        with pytest.raises(DistError, match="exactly one pooler_output"):
            siglip2_hidden_dim(self._graph([1, 768], outputs=2), Path("g"))
