"""The BiRefNet distribution recipe (`birefnet.distribution`) — unit tests for one round of
assembly.

No real assets are used. The input that reaches assembly is a **valid minimal container** of a few
KB (`ir_fixtures`), and the cases that check that a gate refuses something are built with the same
harness ({@link _birefnet_container}) — assembly opens its input as a container, so a hand-written
fake would fail for another reason before reaching the gate.

Since manifest v2 (`karume/2` — ADR 0041) the in-repository layout is uniformly "per-model subtree
+ `shared/`", so every expected path starts with `<model name>/…`.

The layers observable from the core alone (size limits a synthetic plan covers, the complete quant
mapping, the staging/swap invariants, the resolution rules of attribution profiles) live in
`tools/exporter/tests/test_dist.py` (the split of ADR 0065 steps 3+4). The per-template gates are
in `birefnet/tests/test_card.py`.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest
from container_series import i2_container, placed_paths, replace_component, write_component
from ir_fixtures import Shape, ir_container
from upstream_fixture import OTHER_REVISION, stamp_fixture_provenance

from _shared.licenses import mit_license
from birefnet.card import (
    BIREFNET_CHECKPOINTS,
    BIREFNET_LICENSE,
    BIREFNET_POSTPROCESS_DESCRIPTIONS,
    BIREFNET_UPSTREAM,
)
from birefnet.distribution import (
    BIREFNET_COPYRIGHTS,
    BIREFNET_DEFAULT_MODEL,
    BIREFNET_HR_CHECKPOINT,
    BIREFNET_IMAGE_MEAN,
    BIREFNET_IMAGE_STD,
    BIREFNET_LUCIDA_CHECKPOINT,
    BIREFNET_MODELS,
    BIREFNET_OUTPUT_PATHS,
    BIREFNET_PLAIN_DTYPE,
    BIREFNET_POSTPROCESS,
    BIREFNET_QUANT_SEATS,
    BIREFNET_ROLE,
    BIREFNET_STORAGE_FORBIDDEN,
    BIREFNET_STORAGE_REQUIREMENTS,
    BIREFNET_WEIGHT_DTYPES,
    LUCIDA_PIPELINE,
    PIPELINE,
    BirefnetSources,
    birefnet_default_quant,
    birefnet_plan,
    birefnet_postprocess,
    birefnet_quants,
    birefnet_repo_name,
    birefnet_role,
    birefnet_seat_dtypes,
    birefnet_series_name,
    birefnet_sources,
    birefnet_weights,
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


def _baked_provenance(checkpoint: str, revision: str | None = None) -> Provenance:
    """The provenance the script bakes from that checkpoint's download record (the passing form
    of the provenance gate).

    Without `revision` it names the attribution table's pinned value (= the value assembly checks
    against).
    """
    pinned = BIREFNET_CHECKPOINTS[checkpoint].revision
    return Provenance(
        license=BIREFNET_LICENSE,
        notice=NOTICE_FILENAME,
        upstream_revision=pinned if revision is None else revision,
    )


def _write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _in_subtree(model: str, paths: Iterable[str]) -> list[str]:
    """Expected paths inside the model subtree (the uniform layout of ADR 0041 §9)."""
    return [f"{model}/{rel}" for rel in paths]


def _placed_paths(checkpoint: str = BIREFNET_HR_CHECKPOINT) -> list[str]:
    """Relative paths that appear in the distribution (a weights seat expands into numbered
    parts — container-v1 §8).

    The output path table holds the roles of the whole dtype vocabulary, so it is narrowed to the
    roles of that checkpoint's seats.
    """
    roles = {birefnet_role(dtype) for dtype in birefnet_seat_dtypes(checkpoint)}
    paths = {role: path for role, path in BIREFNET_OUTPUT_PATHS.items() if role in roles}
    return placed_paths(paths, birefnet_weights(checkpoint))


def _present(out_dir: Path) -> list[str]:
    return sorted(str(path.relative_to(out_dir)) for path in out_dir.rglob("*") if path.is_file())


# ---- BiRefNet (image-segmentation) -------------------------------------------
#
# The fake assets have the real resolution (the model name is the resolution, so they cannot be
# shrunk to a small non-square like SigLIP2's). Instead, three cases sit on **the failing side**:
# "the model name and the size disagree", "non-square" and "a model name outside the accepted
# set".

#: The second model (`--model 2048`). It lives beside the default in the two-models-per-repository
#: layout (ADR 0092 decisions 1 / 8).
_BIREFNET_SECOND_MODEL = "2048"

#: The `pipelineConfig` field names (a copy of `ROOT_KEYS` in the TS side's
#: `packages/models/src/birefnet/config.ts`). **The loader refuses both unknown and missing keys
#: at parse time**, so the baking side and the loading side need exactly the same field names.
_BIREFNET_CONFIG_KEYS = (
    "imageWidth",
    "imageHeight",
    "imageMean",
    "imageStd",
    "interpolation",
    "postprocess",
)


#: The side of the default model (= the default model name itself). The fake assets' shapes are
#: built from it.
_DEFAULT_SIDE = int(BIREFNET_DEFAULT_MODEL)


def _birefnet_container(
    *,
    storage: str = "f32",
    side: int = _DEFAULT_SIDE,
    input_name: str = "pixel_values",
    input_shape: Sequence[Any] | None = None,
    matte_shape: Sequence[Any] | None = None,
    outputs: int = 1,
    symbols: Sequence[str] = (),
) -> list[bytes]:
    """The content of the matte-estimation graph placed in a series (a part list — part 0 first).

    What reaches assembly must be **a valid container** (assembly opens its input and checks it
    against every rule of the declaration — `karume.dist.assert_weight_components_verified`), so the
    side that a gate should refuse is built with the same harness too, and **only the declaration
    departs from the real thing**.

    `storage` is the storage form of a mixed component as in the real thing (f16 / i8 / i4 keep the
    ineligible weights and the biases in f32, and i4 additionally mixes in i8) — the gate of the
    forbidden table ({@link BIREFNET_STORAGE_FORBIDDEN}) can be tried only in this form. The other
    arguments are the axes of the graph declaration (input name / shape, matte shape, number of
    outputs, symbolic dimensions), defaulting to the real thing.
    """
    shape: Shape = list(input_shape) if input_shape is not None else [1, 3, side, side]
    matte: Shape = list(matte_shape) if matte_shape is not None else [1, 1, shape[2], shape[3]]
    return ir_container(
        mark="birefnet-matte",
        # The pseudo series also **names itself by the component name** (the container's graph
        # name = the manifest's weights key, MUST, container-v1 §2.1 — assembly's gate checks
        # this match).
        named=BIREFNET_ROLE,
        storage=storage,
        inputs=((input_name, shape),),
        outputs=[matte] * outputs,
        symbols=symbols,
    )


def _build_birefnet_sources(
    root: Path,
    *,
    checkpoint: str = BIREFNET_HR_CHECKPOINT,
    model: str = BIREFNET_DEFAULT_MODEL,
    side: int | None = None,
    storage: str = BIREFNET_PLAIN_DTYPE,
    f16_storage: str = "f16",
    f16_side: int | None = None,
    provenance: Provenance | None = None,
    **container: Any,
) -> BirefnetSources:
    """Recreate the series with fake assets (including the undistributed `io.*` / `quality.json`
    that sit among them).

    The arrangement follows the real layout of `_shared.paths` (`outputs/series/`) — so the CLI
    tests can point at the same tree just by swapping the root. There is one series per
    checkpoint × resolution × storage dtype (all of that checkpoint's seats — `birefnet_seat_dtypes`
    — are baked), so the fake assets' size is taken **from the model name** by default. `side` /
    `f16_side` exist to **deliberately break** that correspondence (reproducing a single series
    picked up by mistake).

    `storage` / `side` shape **the f32 seat's series** (the forbidden-table gate cases — Lucida is
    the one with an f32 seat), `f16_storage` / `f16_side` shape the f16 seat's series (the
    requirement-table gate cases). Specifying the shape of a seat the checkpoint lacks fails (a
    fault injection that silently does not get baked would go green without testing anything).
    The remaining graph-declaration axes (`**container`) apply to the series of every dtype.

    The provenance the container states is `provenance` (omitted: the checkpoint's pinned
    revision — {@link _baked_provenance}); only the provenance-gate cases replace it.
    """
    sources = birefnet_sources(root / "outputs" / "series", checkpoint, model)
    shapes = {BIREFNET_PLAIN_DTYPE: (storage, side), "f16": (f16_storage, f16_side)}
    for dtype, shape in shapes.items():
        if dtype not in sources.series_by_dtype and shape != (dtype, None):
            raise AssertionError(
                f"checkpoint '{checkpoint}' has no {dtype} seat — the series shape {shape} for"
                " that seat would not be baked"
            )
    with pytest.MonkeyPatch.context() as patch:
        stamp_fixture_provenance(
            patch, _baked_provenance(checkpoint) if provenance is None else provenance
        )
        for dtype, series in sources.series_by_dtype.items():
            seat_storage, seat_side = shapes[dtype]
            write_component(
                series / "model.krm",
                _birefnet_container(
                    storage=seat_storage,
                    side=seat_side if seat_side is not None else int(model),
                    **container,
                ),
            )
            # An E2E fixture that must not get into the distribution (real series hold these).
            _write(series / "io.ramp.safetensors", b"io-fixture")
            if dtype != BIREFNET_PLAIN_DTYPE:
                # The quality record of a compressed series (`birefnet/export.py`'s
                # QUALITY_FILE) stays out of the distribution too.
                _write(series / "quality.json", b"{}")
    return sources


#: checkpoint → the CLI name of the pipeline that assembles its distribution repository (one
#: Pipeline per repository — ADR 0092 decision 1).
_PIPELINE_NAMES = {BIREFNET_HR_CHECKPOINT: "birefnet", BIREFNET_LUCIDA_CHECKPOINT: "lucida"}


@pytest.fixture
def birefnet_assembled(tmp_path: Path) -> tuple[Path, dict]:
    sources = _build_birefnet_sources(tmp_path)
    out_dir = tmp_path / "models" / birefnet_repo_name(BIREFNET_HR_CHECKPOINT)
    manifest = assemble_family(
        [birefnet_plan(sources, BIREFNET_HR_CHECKPOINT)],
        out_dir,
        BIREFNET_DEFAULT_MODEL,
        root_files=PIPELINE.root_files,
    )
    return out_dir, manifest


def _birefnet_model(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    return manifest["models"][BIREFNET_DEFAULT_MODEL]


class TestBirefnetLayout:
    def test_it_places_the_single_graph_under_the_model_subtree(self, birefnet_assembled) -> None:
        out_dir, _ = birefnet_assembled
        expected = _in_subtree(BIREFNET_DEFAULT_MODEL, _placed_paths())
        # The two legal texts (MIT's copyright / permission notice and the modification notice)
        # are meta files the manifest does not declare.
        assert _present(out_dir) == sorted(
            [*expected, MANIFEST_FILENAME, "LICENSE.md", "NOTICE.md"]
        )

    def test_it_never_carries_the_io_fixtures(self, birefnet_assembled) -> None:
        out_dir, _ = birefnet_assembled
        assert list(out_dir.rglob("io.*")) == []

    def test_it_declares_one_graph_and_no_assets(self, birefnet_assembled) -> None:
        """Running needs one graph only (no tokenizer, no tables = `assets` is empty)."""
        _, manifest = birefnet_assembled
        model = _birefnet_model(manifest)
        assert model["pipeline"] == "birefnet/2"
        assert list(model["weights"]) == [BIREFNET_ROLE]
        assert model["assets"] == {}
        # HR's only seat is f16 — the upstream storage is f16 (ADR 0113 addendum 2026-10-11).
        assert list(model["weights"][BIREFNET_ROLE]) == ["f16"]
        assert list(model["quants"]) == ["f16"]
        assert model["defaultQuant"] == "f16"
        assert model["quants"]["f16"]["weights"] == {BIREFNET_ROLE: "f16"}
        assert model["quants"]["f16"]["session"] == {}

    def test_lucida_keeps_both_seats_with_f32_as_the_default(self, tmp_path: Path) -> None:
        """Lucida's upstream is f32 — two seats: f32 (the default) and a rounded f16."""
        sources = _build_birefnet_sources(tmp_path, checkpoint=BIREFNET_LUCIDA_CHECKPOINT)
        plan = birefnet_plan(sources, BIREFNET_LUCIDA_CHECKPOINT)

        assert list(plan.weights[BIREFNET_ROLE]) == ["f32", "f16"]
        assert list(plan.quants) == ["f32", "f16"]
        assert plan.default_quant == "f32"
        assert plan.quants["f32"]["weights"] == {BIREFNET_ROLE: "f32"}
        assert plan.quants["f16"]["weights"] == {BIREFNET_ROLE: "f16"}
        assert plan.quants["f32"]["session"] == {}
        assert plan.quants["f16"]["session"] == {}

    def test_the_order_of_the_given_series_does_not_matter(self, tmp_path: Path) -> None:
        """Series are matched as a set — the order (of the seats) comes from
        `birefnet_seat_dtypes`, so a different order of the given series yields the same plan
        (seats, weights, quants, default seat, artifacts)."""
        built = _build_birefnet_sources(tmp_path, checkpoint=BIREFNET_LUCIDA_CHECKPOINT)
        reordered = BirefnetSources(
            series_by_dtype=dict(reversed(list(built.series_by_dtype.items())))
        )
        assert list(reordered.series_by_dtype) == ["f16", "f32"]

        plan = birefnet_plan(reordered, BIREFNET_LUCIDA_CHECKPOINT)
        expected = birefnet_plan(built, BIREFNET_LUCIDA_CHECKPOINT)

        assert list(plan.weights[BIREFNET_ROLE]) == ["f32", "f16"]
        assert list(plan.quants) == ["f32", "f16"]
        assert plan.default_quant == "f32"
        assert set(plan.artifacts) == set(expected.artifacts)
        assert plan == expected

    def test_it_reassembles_over_a_previous_run(self, tmp_path: Path) -> None:
        sources = _build_birefnet_sources(tmp_path)
        out_dir = tmp_path / "models" / birefnet_repo_name(BIREFNET_HR_CHECKPOINT)
        plans = [birefnet_plan(sources, BIREFNET_HR_CHECKPOINT)]
        first = assemble_family(plans, out_dir, BIREFNET_DEFAULT_MODEL)
        assert first == assemble_family(plans, out_dir, BIREFNET_DEFAULT_MODEL)
        assert verify_dist(out_dir)

    @pytest.mark.parametrize(("storage", "intruder"), [("f16", "f16"), ("i8", "i8"), ("i4", "i4")])
    def test_it_refuses_a_real_compressed_series_in_the_f32_seat(
        self, tmp_path: Path, storage: str, intruder: str
    ) -> None:
        """A compressed series as in the real thing **satisfies the requirement check** — only the
        forbidden table catches a mixed-up series root.

        A compressed series also holds the ineligible weights and the biases in f32, so "contains
        f32" is true and the requirement check fails on nothing. The reason it should fail is
        "another series' asset sits in the f32 seat", which a numeric gate cannot detect in
        principle (ADR 0027 / 0029).
        """
        sources = _build_birefnet_sources(
            tmp_path, checkpoint=BIREFNET_LUCIDA_CHECKPOINT, storage=storage
        )
        with pytest.raises(
            DistError, match=rf"{birefnet_role('f32')}: .* has (\S+ / )*{intruder} \(actual"
        ):
            birefnet_plan(sources, BIREFNET_LUCIDA_CHECKPOINT)

    def test_the_plain_f32_series_passes_both_storage_gates(self, tmp_path: Path) -> None:
        """The passing counterpart — a plain f32 series passes both the requirement check and the
        forbidden table (the forbidden table has not collapsed into always failing)."""
        sources = _build_birefnet_sources(
            tmp_path, checkpoint=BIREFNET_LUCIDA_CHECKPOINT, storage="f32"
        )

        assert birefnet_plan(sources, BIREFNET_LUCIDA_CHECKPOINT).name == BIREFNET_DEFAULT_MODEL

    def test_the_forbidden_table_names_every_compressed_layout_in_the_codec_ledger(self) -> None:
        """The forbidden table holds **all** of the storage check's vocabulary (the codec
        ledger's layouts) except f32 / i32.

        If even one is missing, a series root of that storage form silently passes the f32 seat.
        i32 is plain storage (index tables), so it is not forbidden.
        """
        compressed = {entry.layout for entry in CODEC_LEDGER.values()} - {"f32", "i32"}

        assert set(BIREFNET_STORAGE_FORBIDDEN[birefnet_role("f32")]) == compressed

    def test_it_refuses_an_i2_series_in_the_f32_seat(self, tmp_path: Path) -> None:
        """i2 (the fixed packed values of QAT series), which `ir_container` cannot write, also
        fails on the actual binding table."""
        sources = _build_birefnet_sources(tmp_path, checkpoint=BIREFNET_LUCIDA_CHECKPOINT)
        replace_component(
            sources.series_by_dtype["f32"] / "model.krm", i2_container(named=BIREFNET_ROLE)
        )

        with pytest.raises(
            DistError, match=rf"{birefnet_role('f32')}: .* has (\S+ / )*i2 \(actual"
        ):
            birefnet_plan(sources, BIREFNET_LUCIDA_CHECKPOINT)


class TestBirefnetContainerProvenance:
    """The provenance a container states is checked against the attribution table's pinned values
    (the license and the upstream revision)."""

    def test_it_refuses_a_container_that_names_another_license(self, tmp_path: Path) -> None:
        sources = _build_birefnet_sources(
            tmp_path,
            provenance=Provenance(
                license="apache-2.0",
                notice=NOTICE_FILENAME,
                upstream_revision=BIREFNET_CHECKPOINTS[BIREFNET_HR_CHECKPOINT].revision,
            ),
        )

        with pytest.raises(DistError, match=r"provenance\.license is 'apache-2.0'"):
            birefnet_plan(sources, BIREFNET_HR_CHECKPOINT)

    def test_it_refuses_a_container_without_an_upstream_revision(self, tmp_path: Path) -> None:
        sources = _build_birefnet_sources(
            tmp_path, provenance=Provenance(license=BIREFNET_LICENSE, notice=NOTICE_FILENAME)
        )

        with pytest.raises(DistError, match=r"has no provenance\.upstreamRevision"):
            birefnet_plan(sources, BIREFNET_HR_CHECKPOINT)

    @pytest.mark.parametrize("checkpoint", [BIREFNET_HR_CHECKPOINT, BIREFNET_LUCIDA_CHECKPOINT])
    def test_every_series_naming_the_pinned_revision_is_assembled(
        self, tmp_path: Path, checkpoint: str
    ) -> None:
        """The passing form: when the series of every seat names the checkpoint's pinned
        revision, it assembles (the control for the fault injections below)."""
        sources = _build_birefnet_sources(tmp_path, checkpoint=checkpoint)

        birefnet_plan(sources, checkpoint)

    def test_the_pinned_revisions_tell_the_checkpoints_apart(self) -> None:
        """The pinned values tell the checkpoints apart — equal values would let the mix-ups
        below slip through."""
        revisions = [entry.revision for entry in BIREFNET_CHECKPOINTS.values()]

        assert len(set(revisions)) == len(revisions)

    def test_it_refuses_a_series_baked_from_another_revision(self, tmp_path: Path) -> None:
        """A series baked from a revision other than the pinned one is not assembled, even for
        the same checkpoint (to take an upstream update, move the pinned value — a series whose
        local download alone moved ahead is not shipped silently)."""
        sources = _build_birefnet_sources(
            tmp_path, provenance=_baked_provenance(BIREFNET_HR_CHECKPOINT, OTHER_REVISION)
        )

        with pytest.raises(DistError, match="upstreamRevision is") as raised:
            birefnet_plan(sources, BIREFNET_HR_CHECKPOINT)
        assert OTHER_REVISION in str(raised.value)

    def test_it_refuses_the_f16_series_of_another_checkpoint(self, tmp_path: Path) -> None:
        """Fault injection: Lucida's f32 series + HR's f16 series (both MIT, same size).

        If the provenance gate looked only at the license and at whether a revision exists,
        series of different checkpoints would assemble into two seats of one model (2026-09-26
        review A-1). Every container is checked against Lucida's pinned value, so the f16 series
        naming HR's revision fails it.
        """
        lucida = _build_birefnet_sources(tmp_path / "lucida", checkpoint=BIREFNET_LUCIDA_CHECKPOINT)
        hr = _build_birefnet_sources(tmp_path / "hr")
        mixed = BirefnetSources(
            series_by_dtype={
                "f32": lucida.series_by_dtype["f32"],
                "f16": hr.series_by_dtype["f16"],
            }
        )

        with pytest.raises(DistError, match="upstreamRevision is") as raised:
            birefnet_plan(mixed, BIREFNET_LUCIDA_CHECKPOINT)
        assert str(hr.series_by_dtype["f16"]) in str(raised.value)
        assert BIREFNET_CHECKPOINTS[BIREFNET_HR_CHECKPOINT].revision in str(raised.value)
        assert BIREFNET_CHECKPOINTS[BIREFNET_LUCIDA_CHECKPOINT].revision in str(raised.value)

    def test_the_single_hr_seat_refuses_another_checkpoints_series(self, tmp_path: Path) -> None:
        """Fault injection: Lucida's f16 series in HR's only seat (f16).

        With a single seat, comparing revisions between seats compares nothing. Checking against
        the pinned value keeps HR's repository to HR's weights (no distribution whose
        attribution, copyright line and post-processing say HR while the content is Lucida).
        """
        lucida = _build_birefnet_sources(tmp_path, checkpoint=BIREFNET_LUCIDA_CHECKPOINT)
        foreign = BirefnetSources(series_by_dtype={"f16": lucida.series_by_dtype["f16"]})

        with pytest.raises(DistError, match="upstreamRevision is") as raised:
            birefnet_plan(foreign, BIREFNET_HR_CHECKPOINT)
        assert BIREFNET_CHECKPOINTS[BIREFNET_LUCIDA_CHECKPOINT].revision in str(raised.value)
        assert BIREFNET_CHECKPOINTS[BIREFNET_HR_CHECKPOINT].revision in str(raised.value)


class TestBirefnetPipelineConfig:
    """`pipelineConfig` matches the loader's schema (`src/birefnet/config.ts`) down to the field
    names."""

    def test_it_declares_exactly_the_fields_the_loader_accepts(self, birefnet_assembled) -> None:
        _, manifest = birefnet_assembled
        assert tuple(_birefnet_model(manifest)["pipelineConfig"]) == _BIREFNET_CONFIG_KEYS

    def test_it_derives_the_resize_target_from_the_exported_graph(self, birefnet_assembled) -> None:
        """The baked graph is the only source of the resize target (the upstream has no
        preprocessing config)."""
        _, manifest = birefnet_assembled
        config = _birefnet_model(manifest)["pipelineConfig"]
        assert config["imageWidth"] == _DEFAULT_SIDE
        assert config["imageHeight"] == _DEFAULT_SIDE

    def test_it_declares_the_upstream_normalization_constants(self, birefnet_assembled) -> None:
        """The normalization constants have no machine-readable source, so dist holds them as a
        declaration (the head of the section)."""
        _, manifest = birefnet_assembled
        config = _birefnet_model(manifest)["pipelineConfig"]
        assert config["imageMean"] == list(BIREFNET_IMAGE_MEAN)
        assert config["imageStd"] == list(BIREFNET_IMAGE_STD)
        assert config["interpolation"] == "bilinear"

    def test_the_hr_repository_declares_the_upstream_postprocess(self, birefnet_assembled) -> None:
        """HR's upstream is σ → 8 bit → Pillow's BICUBIC (ADR 0125 — the upstream was chosen
        after an A/B)."""
        _, manifest = birefnet_assembled
        assert _birefnet_model(manifest)["pipelineConfig"]["postprocess"] == "u8-pil-bicubic"

    def test_the_lucida_repository_declares_its_own_upstream_postprocess(
        self, tmp_path: Path
    ) -> None:
        """Lucida's upstream is σ → floating-point antialiased bilinear → 8 bit. The procedure is
        decided by the checkpoint (with the same resolution and preprocessing, copying HR's
        procedure would take Lucida away from its own upstream).
        """
        sources = _build_birefnet_sources(tmp_path, checkpoint=BIREFNET_LUCIDA_CHECKPOINT)
        plan = birefnet_plan(sources, BIREFNET_LUCIDA_CHECKPOINT)
        assert plan.pipeline_config["postprocess"] == "float-bilinear"

    def test_every_checkpoint_has_a_postprocess_the_card_can_describe(self) -> None:
        """The procedure table has the same checkpoints as the attribution table, and its values
        are only procedures the card can describe."""
        assert sorted(BIREFNET_POSTPROCESS) == sorted(BIREFNET_UPSTREAM)
        assert set(BIREFNET_POSTPROCESS.values()) <= set(BIREFNET_POSTPROCESS_DESCRIPTIONS)

    def test_it_refuses_a_checkpoint_without_a_postprocess(self) -> None:
        """No default procedure fills the gap — that would silently produce a distribution
        declaring post-processing that differs from the upstream."""
        with pytest.raises(DistError, match="tiny"):
            birefnet_postprocess("tiny")


class TestBirefnetGraphGate:
    """The assembly gate — combinations that would still form a distribution despite the mismatch
    fail before placement."""

    def test_it_refuses_a_graph_baked_for_another_resolution(self, tmp_path: Path) -> None:
        """The model name is the very value users write as `model: "1024"` — if it departs from
        the content, it cannot ship."""
        sources = _build_birefnet_sources(tmp_path, input_shape=(1, 3, 512, 512))
        with pytest.raises(DistError, match=rf"what model '{BIREFNET_DEFAULT_MODEL}' claims"):
            birefnet_plan(sources, BIREFNET_HR_CHECKPOINT)

    def test_it_refuses_the_other_models_graph_in_this_models_seat(self, tmp_path: Path) -> None:
        """Mixing up the two co-located models — specific to this layout, it takes picking up
        just one wrong series.

        With the `1024` graph in the `2048` seat, `pipelineConfig` stays consistent at 1024², so
        both running and `verify_dist` pass, and **only the model name disagrees**.
        """
        sources = _build_birefnet_sources(
            tmp_path, model=_BIREFNET_SECOND_MODEL, f16_side=_DEFAULT_SIDE
        )
        with pytest.raises(DistError, match=rf"what model '{_BIREFNET_SECOND_MODEL}' claims"):
            birefnet_plan(sources, BIREFNET_HR_CHECKPOINT, _BIREFNET_SECOND_MODEL)

    def test_it_refuses_a_model_name_outside_the_published_set(self, tmp_path: Path) -> None:
        """{@link BIREFNET_MODELS} is the only accepted set — it fails before any series is
        searched for."""
        with pytest.raises(DistError, match="is not shipped"):
            birefnet_sources(tmp_path, BIREFNET_HR_CHECKPOINT, "512")

    def test_it_refuses_a_graph_whose_input_is_not_square(self, tmp_path: Path) -> None:
        sources = _build_birefnet_sources(tmp_path, input_shape=(1, 3, _DEFAULT_SIDE, 512))
        with pytest.raises(DistError, match=rf"what model '{BIREFNET_DEFAULT_MODEL}' claims"):
            birefnet_plan(sources, BIREFNET_HR_CHECKPOINT)

    def test_it_refuses_a_graph_with_another_channel_count(self, tmp_path: Path) -> None:
        sources = _build_birefnet_sources(
            tmp_path, input_shape=(1, 4, _DEFAULT_SIDE, _DEFAULT_SIDE)
        )
        with pytest.raises(DistError, match="both batch and channel count static"):
            birefnet_plan(sources, BIREFNET_HR_CHECKPOINT)

    def test_it_refuses_a_graph_with_a_second_output(self, tmp_path: Path) -> None:
        """With an export that includes multi-scale supervision, the downstream that reads by
        position would read another value as α."""
        sources = _build_birefnet_sources(tmp_path, outputs=2)
        with pytest.raises(DistError, match="exactly one matte"):
            birefnet_plan(sources, BIREFNET_HR_CHECKPOINT)

    def test_it_refuses_a_matte_that_is_not_one_channel(self, tmp_path: Path) -> None:
        """A shape an implementation looking only at the element count would pass
        (`[1, 3, S, S]` = 3 intermediate predictions)."""
        sources = _build_birefnet_sources(
            tmp_path, matte_shape=[1, 3, _DEFAULT_SIDE, _DEFAULT_SIDE]
        )
        with pytest.raises(DistError, match="1 channel with the same dimensions as the input"):
            birefnet_plan(sources, BIREFNET_HR_CHECKPOINT)

    def test_it_refuses_a_graph_with_a_symbolic_axis(self, tmp_path: Path) -> None:
        """The resolution and the window masks are baked as constants, so no axis may vary."""
        sources = _build_birefnet_sources(tmp_path, symbols=("T",))
        with pytest.raises(DistError, match="symbolic dimensions"):
            birefnet_plan(sources, BIREFNET_HR_CHECKPOINT)

    def test_it_refuses_a_renamed_input(self, tmp_path: Path) -> None:
        """The runtime binds by name, so a changed spelling cannot be bound."""
        sources = _build_birefnet_sources(tmp_path, input_name="pixels")
        with pytest.raises(DistError, match="graph inputs"):
            birefnet_plan(sources, BIREFNET_HR_CHECKPOINT)


class TestBirefnetModelCard:
    def _run(self, tmp_path: Path, checkpoint: str, *models: str) -> Path:
        """Run the CLI once with fake assets only (`--model` is the resolution; the first one is
        `defaultModel`)."""
        names = models or (BIREFNET_DEFAULT_MODEL,)
        series_dir = tmp_path / "outputs" / "series"
        for model in names:
            _build_birefnet_sources(tmp_path, checkpoint=checkpoint, model=model)
        out_dir = tmp_path / "dist"
        argv = ["--pipeline", _PIPELINE_NAMES[checkpoint]]
        for model in names:
            argv += ["--model", model]
        main([*argv, "--series", str(series_dir), "--out", str(out_dir)])
        return out_dir

    def test_it_describes_the_background_removal_distribution(self, tmp_path: Path) -> None:
        out_dir = self._run(tmp_path, BIREFNET_HR_CHECKPOINT)
        card = (out_dir / MODEL_CARD_FILENAME).read_text(encoding="utf-8")
        assert "pipeline_tag: image-segmentation" in card
        assert "license: mit" in card
        # A distribution that keeps the storage form is none of the 4 values (left to the Hub's
        # inference).
        assert "base_model_relation" not in card
        # That compositing is not done here is the very constraint users want to check first.
        assert "**Compositing is yours.**" in card
        assert f"resizes to {_DEFAULT_SIDE} × {_DEFAULT_SIDE}" in card
        assert "one alpha byte per pixel" in card
        # The post-processing procedure is rendered from the model's declaration
        # (HR = u8-pil-bicubic).
        assert "- **post-processing** (`u8-pil-bicubic`" in card
        assert "`float-bilinear`" not in card
        # The single-file distribution is retired (ADR 0081 decision 3) — the description of
        # the format does not contradict the real thing.
        assert "a single safetensors file" not in card
        assert "`.krm` part sequence" in card
        # The card is rendered from a distribution that **passed verification**.
        assert verify_dist(out_dir)

    def test_the_usage_snippet_enumerates_the_choices_from_the_manifest(
        self, tmp_path: Path
    ) -> None:
        """The form ruled on 2026-08-12: optionals are written as comments, and the selectable
        values are enumerated from the manifest.

        `segment()` has no run-time knobs, so this pipeline's optionals are model / quant only.
        It spells "available:" even with a single seat so the text follows automatically when
        seats are added (baked-in wording like "the only one this repository ships" turns false
        the moment one is added).
        """
        card = (self._run(tmp_path / "hr", BIREFNET_HR_CHECKPOINT) / MODEL_CARD_FILENAME).read_text(
            encoding="utf-8"
        )
        assert (
            f'  // model: "{BIREFNET_DEFAULT_MODEL}",'
            f" // default — available: {BIREFNET_DEFAULT_MODEL}" in card
        )
        # HR's only seat is f16 (spelled "available:" even for one seat); Lucida has f32
        # (default) / f16.
        assert '  // quant: "f16", // default — available: f16' in card
        assert "the only one this repository ships" not in card
        lucida = (
            self._run(tmp_path / "lucida", BIREFNET_LUCIDA_CHECKPOINT) / MODEL_CARD_FILENAME
        ).read_text(encoding="utf-8")
        assert '  // quant: "f32", // default — available: f16 / f32' in lucida

    def test_the_attribution_follows_the_checkpoint_of_the_repository(self, tmp_path: Path) -> None:
        """The attribution is decided uniquely by the pipeline's checkpoint (not an axis left to
        choose)."""
        card = (self._run(tmp_path, BIREFNET_LUCIDA_CHECKPOINT) / MODEL_CARD_FILENAME).read_text(
            encoding="utf-8"
        )
        assert f"base_model: {BIREFNET_UPSTREAM['lucida']}" in card
        # The fine-tune's base and the training data's license are facts of the lucida side only.
        assert BIREFNET_UPSTREAM["hr"] in card
        assert "ToonOut" in card
        # The card records that the variant with baked-in preprocessing is not shipped, and why.
        assert "lucida-m35-comfy.safetensors" in card
        # The post-processing procedure is also the checkpoint's upstream one (HR's is not
        # copied).
        assert "- **post-processing** (`float-bilinear`" in card
        assert "`u8-pil-bicubic`" not in card

    def test_the_default_model_card_does_not_carry_another_models_attribution(
        self, tmp_path: Path
    ) -> None:
        card = (self._run(tmp_path, BIREFNET_HR_CHECKPOINT) / MODEL_CARD_FILENAME).read_text(
            encoding="utf-8"
        )
        assert f"base_model: {BIREFNET_UPSTREAM['hr']}" in card
        assert "ToonOut" not in card


class TestBirefnetResolutionFamily:
    """Two models per repository (ADR 0092 decisions 1 / 8) — 1024² and 2048² live in the same
    repository."""

    def _run(self, tmp_path: Path) -> Path:
        for model in BIREFNET_MODELS:
            _build_birefnet_sources(tmp_path, model=model)
        out_dir = tmp_path / "models" / birefnet_repo_name(BIREFNET_HR_CHECKPOINT)
        main(
            [
                "--pipeline",
                "birefnet",
                "--model",
                BIREFNET_DEFAULT_MODEL,
                "--model",
                _BIREFNET_SECOND_MODEL,
                "--series",
                str(tmp_path / "outputs" / "series"),
                "--out",
                str(out_dir),
            ]
        )
        return out_dir

    def test_both_resolutions_live_in_one_repository_with_1024_as_the_default(
        self, tmp_path: Path
    ) -> None:
        """The default is **the first `--model`** (`karume.dist.main`) — the order on the command
        line decides the default."""
        out_dir = self._run(tmp_path)
        manifest = json.loads((out_dir / MANIFEST_FILENAME).read_text(encoding="utf-8"))
        assert sorted(manifest["models"]) == sorted(BIREFNET_MODELS)
        assert manifest["defaultModel"] == BIREFNET_DEFAULT_MODEL
        assert out_dir.name == "karume-birefnet-hr"

    def test_each_model_declares_its_own_resize_target(self, tmp_path: Path) -> None:
        """Co-located, the size is still per model — a shared declaration would give one of them
        the wrong number."""
        out_dir = self._run(tmp_path)
        manifest = json.loads((out_dir / MANIFEST_FILENAME).read_text(encoding="utf-8"))
        for model in BIREFNET_MODELS:
            config = manifest["models"][model]["pipelineConfig"]
            assert (config["imageWidth"], config["imageHeight"]) == (int(model), int(model))

    def test_the_layout_keeps_each_models_graph_in_its_own_subtree(self, tmp_path: Path) -> None:
        """Part 0 (the part holding the graph description) declares the size, so it must not be
        folded into the shared area."""
        placed = sorted(verify_dist(self._run(tmp_path)))
        graph_part = _placed_paths()[0]
        for model in BIREFNET_MODELS:
            assert f"{model}/{graph_part}" in placed
        assert {path.split("/", 1)[0] for path in placed} <= {*BIREFNET_MODELS, "shared"}

    def test_the_card_enumerates_both_models_and_their_own_resources(self, tmp_path: Path) -> None:
        """The Usage enumeration and the measured resources both follow the manifest's two
        models."""
        card = (self._run(tmp_path) / MODEL_CARD_FILENAME).read_text(encoding="utf-8")
        listing = " / ".join(sorted(BIREFNET_MODELS))
        assert f'  // model: "{BIREFNET_DEFAULT_MODEL}", // default — available: {listing}' in card
        # Resources are measured per model (copying one into both sections would make one of them
        # false). HR is measured on the f16 seat.
        assert "about 1.1 GiB allocated in total with the `f16` quant" in card
        assert "about 3.1 GiB allocated in total with the `f16` quant" in card
        # One upstream (what is co-located is the same checkpoint at different resolutions).
        assert card.count(f"base_model: {BIREFNET_UPSTREAM['hr']}") == 1
        assert BIREFNET_UPSTREAM["lucida"] not in card


class TestBirefnetCli:
    def test_the_default_output_directory_is_named_per_repository(self) -> None:
        """The repository name is not derived (`karume-lucida` cannot be derived from the model
        name)."""
        assert default_out_dir(PIPELINE, [BIREFNET_DEFAULT_MODEL]).name == "karume-birefnet-hr"
        assert default_out_dir(LUCIDA_PIPELINE, [BIREFNET_DEFAULT_MODEL]).name == "karume-lucida"

    def test_one_attribution_profile_needs_no_choice(self) -> None:
        profiles = PIPELINE.card_profiles
        assert len(profiles) == 1
        assert resolve_card_renderer(PIPELINE, None) is next(iter(profiles.values()))

    @pytest.mark.parametrize("model", BIREFNET_MODELS)
    def test_each_pipeline_looks_only_at_its_own_checkpoints_series(
        self, tmp_path: Path, model: str
    ) -> None:
        """Each pipeline holds its checkpoint fixed — a mix-up across pipelines cannot even be
        spelled.

        The copyright lines of `LICENSE.md` are one fixed set on the Pipeline, so HR's pipeline
        picking up Lucida's series would be "a distribution claiming a copyright that is not its
        own". Both remain valid prose and pass `verify_dist`, so **the series path being decided
        only by the pipeline's checkpoint** is the only safeguard (`--model` is the resolution
        and cannot choose the repository's content).
        """
        for pipeline, checkpoint in (
            (PIPELINE, BIREFNET_HR_CHECKPOINT),
            (LUCIDA_PIPELINE, BIREFNET_LUCIDA_CHECKPOINT),
        ):
            with pytest.raises(DistError) as raised:
                pipeline.plan(tmp_path, model)
            # The first one searched for is the default seat's series (HR = the f16 series,
            # Lucida = the f32 series).
            expected = birefnet_series_name(checkpoint, model, birefnet_default_quant(checkpoint))
            assert expected in str(raised.value)

    @pytest.mark.parametrize("pipeline", [PIPELINE, LUCIDA_PIPELINE])
    def test_each_pipeline_refuses_a_model_outside_the_published_set(self, tmp_path, pipeline):
        with pytest.raises(DistError, match="is not shipped"):
            pipeline.plan(tmp_path, "512")

    def test_the_series_name_carries_the_upstream_name_and_the_resolution(
        self, tmp_path: Path
    ) -> None:
        """The series name is the upstream repository name + the resolution (the same formula as
        `birefnet.export.default_out_dir`)."""
        for checkpoint, repo in BIREFNET_UPSTREAM.items():
            name = repo.split("/", 1)[1].lower().replace("_", "-")
            for model in BIREFNET_MODELS:
                expected = f"{name}-{model}"
                assert birefnet_series_name(checkpoint, model, BIREFNET_PLAIN_DTYPE) == expected
                sources = birefnet_sources(tmp_path, checkpoint, model)
                assert {dtype: path.name for dtype, path in sources.series_by_dtype.items()} == {
                    dtype: expected if dtype == BIREFNET_PLAIN_DTYPE else f"{expected}-{dtype}"
                    for dtype in birefnet_seat_dtypes(checkpoint)
                }

    def test_it_refuses_a_checkpoint_it_has_no_attribution_for(self, tmp_path: Path) -> None:
        """A checkpoint missing from the attribution table "cannot state its provenance", so it
        fails before any series is searched for."""
        with pytest.raises(DistError, match="is unknown"):
            birefnet_sources(tmp_path, "tiny")

    def test_it_assembles_into_the_pipeline_default_directory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import dist

        sources = _build_birefnet_sources(tmp_path, checkpoint=BIREFNET_LUCIDA_CHECKPOINT)
        monkeypatch.setattr(dist, "DIST_ROOT", tmp_path / "models")

        main(["--pipeline", "lucida", "--series", str(sources.series_by_dtype["f32"].parent)])

        out_dir = tmp_path / "models" / "karume-lucida"
        expected = _in_subtree(BIREFNET_DEFAULT_MODEL, _placed_paths(BIREFNET_LUCIDA_CHECKPOINT))
        assert sorted(verify_dist(out_dir)) == sorted(expected)


class TestBirefnetLegalText:
    """The MIT text and the modification notice at the root of the distribution repository
    (ADR 0092 decision 7)."""

    def _legal(self, tmp_path: Path, checkpoint: str) -> tuple[str, str]:
        _build_birefnet_sources(tmp_path, checkpoint=checkpoint)
        out_dir = tmp_path / "dist"
        main(
            [
                "--pipeline",
                _PIPELINE_NAMES[checkpoint],
                "--series",
                str(tmp_path / "outputs" / "series"),
                "--out",
                str(out_dir),
            ]
        )
        return (
            # LICENSE is returned by decoding its bytes (if read_text's newline translation
            # folded CRLF, the byte-identity gate would let it through — this keeps the same mesh
            # as the other families' read_bytes comparisons).
            (out_dir / "LICENSE.md").read_bytes().decode("utf-8"),
            (out_dir / "NOTICE.md").read_text(encoding="utf-8"),
        )

    @pytest.mark.parametrize("checkpoint", [BIREFNET_HR_CHECKPOINT, BIREFNET_LUCIDA_CHECKPOINT])
    def test_it_ships_the_license_text_byte_identical(
        self, tmp_path: Path, checkpoint: str
    ) -> None:
        """MIT's "include the copyright notice and the permission notice" — apart from the
        insertion point, it is the template text itself.

        The original is `_shared/licenses/mit.txt` with {@link BIREFNET_COPYRIGHTS} inserted. Any
        reformatting or newline translation somewhere along the assembly path moves a byte, but
        the prose stays valid, so the other gates let it through.
        """
        license_text, _ = self._legal(tmp_path, checkpoint)
        assert license_text == mit_license(BIREFNET_COPYRIGHTS[checkpoint])

    def test_the_hr_repository_carries_only_the_upstream_copyright(self, tmp_path: Path) -> None:
        license_text, notice = self._legal(tmp_path, BIREFNET_HR_CHECKPOINT)
        assert "MIT License" in license_text
        assert "Copyright (c) 2024 ZhengPeng" in license_text
        # It claims no copyright that is not its own (the fine-tune side is another repository's
        # fact).
        assert "egeorcun" not in license_text
        assert (
            "The above copyright notice and this permission notice shall be included in all"
            in license_text
        )
        assert BIREFNET_UPSTREAM[BIREFNET_HR_CHECKPOINT] in notice
        assert "no quantization" in notice.lower()

    def test_the_lucida_repository_carries_both_copyrights(self, tmp_path: Path) -> None:
        """Under MIT a derivative cannot drop the upstream copyright notice (the same direction as
        the upstream card's own statement)."""
        license_text, notice = self._legal(tmp_path, BIREFNET_LUCIDA_CHECKPOINT)
        assert "Copyright (c) 2026 egeorcun" in license_text
        assert "Copyright (c) 2024 ZhengPeng" in license_text
        assert BIREFNET_UPSTREAM[BIREFNET_LUCIDA_CHECKPOINT] in notice


class TestBirefnetWeightDtypes:
    """The storage-dtype axis (ADR 0113 — the seats are derived per checkpoint from the upstream
    storage; each dtype's series has its own root)."""

    def test_the_seats_follow_the_upstream_storage(self) -> None:
        """No seat wider than the upstream storage is offered, and the default is the storage
        type (ADR 0113 addendum 2026-10-11).

        HR's upstream is f16 only — an f32 seat would only hold the same values at twice the
        residency, so it is not offered. Lucida's upstream is f32 — an f32 seat (the default) and
        a rounded f16 seat.
        """
        assert birefnet_seat_dtypes(BIREFNET_HR_CHECKPOINT) == ("f16",)
        assert birefnet_default_quant(BIREFNET_HR_CHECKPOINT) == "f16"
        assert birefnet_seat_dtypes(BIREFNET_LUCIDA_CHECKPOINT) == ("f32", "f16")
        assert birefnet_default_quant(BIREFNET_LUCIDA_CHECKPOINT) == "f32"

    def test_the_default_seat_is_the_upstream_type_not_its_width(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """MUST: the default seat is the upstream storage dtype itself — another type of the same
        width (bf16 for an f16 checkpoint) is a rounding, so if it is the only seat, it fails
        instead of choosing a default.

        Fault injection: adding bf16 (16 bit) to the vocabulary leaves HR with only a bf16 seat.
        An implementation that chooses by width would return it as "the default that keeps the
        values".
        """
        import birefnet.card as card
        import birefnet.distribution as distribution

        monkeypatch.setattr(distribution, "BIREFNET_WEIGHT_DTYPES", ("f32", "bf16"))
        monkeypatch.setattr(card, "_STORAGE_BITS", {"f32": 32, "f16": 16, "bf16": 16})
        assert birefnet_seat_dtypes(BIREFNET_HR_CHECKPOINT) == ("bf16",)

        with pytest.raises(DistError, match="cannot choose the default preset"):
            birefnet_default_quant(BIREFNET_HR_CHECKPOINT)

    def test_each_dtype_has_its_own_output_path(self) -> None:
        """The storage dtype shows in the file name (a spelling that cannot be mixed up even with
        two series in one directory)."""
        assert {
            birefnet_role("f32"): f"{BIREFNET_ROLE}/model.f32.krm",
            birefnet_role("f16"): f"{BIREFNET_ROLE}/model.f16.krm",
        } == BIREFNET_OUTPUT_PATHS

    def test_every_seat_requires_its_own_storage(self) -> None:
        assert {
            birefnet_role(dtype): dtype for dtype in BIREFNET_WEIGHT_DTYPES
        } == BIREFNET_STORAGE_REQUIREMENTS

    def test_only_the_plain_seat_has_a_forbidden_table(self) -> None:
        """A mix-up in the f16 seat (an f32 series) fails the requirement check, so only the f32
        seat has a forbidden table."""
        assert set(BIREFNET_STORAGE_FORBIDDEN) == {birefnet_role(BIREFNET_PLAIN_DTYPE)}

    def test_the_series_root_carries_the_dtype_suffix(self, tmp_path: Path) -> None:
        """f32 has no suffix (the existing series' spelling); f16 has `-f16`
        (`birefnet.export.default_out_dir`)."""
        sources = birefnet_sources(tmp_path, BIREFNET_LUCIDA_CHECKPOINT, BIREFNET_DEFAULT_MODEL)
        assert {dtype: path.name for dtype, path in sources.series_by_dtype.items()} == {
            "f32": "lucida-1024",
            "f16": "lucida-1024-f16",
        }

    def test_hr_looks_only_at_its_f16_series(self, tmp_path: Path) -> None:
        """HR's only seat is f16 — an f32 series (`birefnet-hr-1024`) on disk is not searched
        for."""
        sources = birefnet_sources(tmp_path, BIREFNET_HR_CHECKPOINT, BIREFNET_DEFAULT_MODEL)
        assert {dtype: path.name for dtype, path in sources.series_by_dtype.items()} == {
            "f16": "birefnet-hr-1024-f16",
        }

    def test_hr_does_not_take_an_f32_series(self, tmp_path: Path) -> None:
        """An f32 series handed to HR's plan is not assembled (no seat that merely widens the
        values is shipped).

        Even with HR's provenance and size, a series of a dtype with no seat fails — otherwise a
        placement the weights declaration does not have slips in, or an f32 seat appears that
        bypasses the seat rules.
        """
        hr = _build_birefnet_sources(tmp_path / "hr")
        widened = _build_birefnet_sources(
            tmp_path / "widened",
            checkpoint=BIREFNET_LUCIDA_CHECKPOINT,
            provenance=_baked_provenance(BIREFNET_HR_CHECKPOINT),
        )
        sources = BirefnetSources(
            series_by_dtype={
                "f32": widened.series_by_dtype["f32"],
                "f16": hr.series_by_dtype["f16"],
            }
        )

        with pytest.raises(DistError, match=r"differ from checkpoint 'hr''s presets \['f16'\]"):
            birefnet_plan(sources, BIREFNET_HR_CHECKPOINT)

    def test_lucida_does_not_take_sources_without_its_f16_seat(self, tmp_path: Path) -> None:
        """A series dict missing a seat's key is not assembled either (the match is equality; a
        subset is not enough).

        The f32 series has Lucida's provenance and size, and only the f16 key is missing — if the
        match weakened to a subset, this plan would go looking for the declared f16 seat's series
        and fail outside the match.
        """
        built = _build_birefnet_sources(tmp_path, checkpoint=BIREFNET_LUCIDA_CHECKPOINT)
        sources = BirefnetSources(series_by_dtype={"f32": built.series_by_dtype["f32"]})

        with pytest.raises(
            DistError, match=r"differ from checkpoint 'lucida''s presets \['f32', 'f16'\]"
        ):
            birefnet_plan(sources, BIREFNET_LUCIDA_CHECKPOINT)

    def test_it_refuses_a_dtype_it_does_not_ship(self) -> None:
        with pytest.raises(DistError, match="is not shipped"):
            birefnet_series_name(BIREFNET_HR_CHECKPOINT, BIREFNET_DEFAULT_MODEL, "i8")

    def test_it_refuses_the_plain_series_in_the_f16_seat(self, tmp_path: Path) -> None:
        """An f32 series in the f16 seat (an export that forgot `--dtype f16`) — the requirement
        check fails it."""
        sources = _build_birefnet_sources(tmp_path, f16_storage="f32")
        with pytest.raises(DistError, match=rf"{birefnet_role('f16')}: .* has no f16"):
            birefnet_plan(sources, BIREFNET_HR_CHECKPOINT)

    def test_it_refuses_when_the_f16_series_is_missing(self, tmp_path: Path) -> None:
        """One series alone is not assembled (no distribution with an f16 seat that is declared
        but empty)."""
        sources = _build_birefnet_sources(tmp_path, checkpoint=BIREFNET_LUCIDA_CHECKPOINT)
        for path in sources.series_by_dtype["f16"].iterdir():
            path.unlink()
        sources.series_by_dtype["f16"].rmdir()

        with pytest.raises(DistError) as raised:
            birefnet_plan(sources, BIREFNET_LUCIDA_CHECKPOINT)
        assert sources.series_by_dtype["f16"].name in str(raised.value)

    def test_it_checks_the_graph_of_every_dtype_series(self, tmp_path: Path) -> None:
        """MUST: the f16 series' size is checked too (an f16 series baked at another resolution
        does not share the model name)."""
        sources = _build_birefnet_sources(
            tmp_path, checkpoint=BIREFNET_LUCIDA_CHECKPOINT, f16_side=512
        )
        with pytest.raises(DistError, match=rf"what model '{BIREFNET_DEFAULT_MODEL}' claims"):
            birefnet_plan(sources, BIREFNET_LUCIDA_CHECKPOINT)

    @pytest.mark.parametrize("checkpoint", [BIREFNET_HR_CHECKPOINT, BIREFNET_LUCIDA_CHECKPOINT])
    def test_every_seat_has_its_display_fields(self, checkpoint: str) -> None:
        """The quant seats' display fields are a separate table keyed by (seat, upstream
        storage) — if its keys drift from the set of seats, the distribution declares weights
        with no seat (or has an empty seat) (2026-09-26 review F8)."""
        stored = BIREFNET_CHECKPOINTS[checkpoint].stored_dtype
        seats = birefnet_seat_dtypes(checkpoint)

        assert all((dtype, stored) in BIREFNET_QUANT_SEATS for dtype in seats)
        assert tuple(birefnet_quants(checkpoint)) == seats

    def test_the_display_table_has_no_seat_that_no_repository_ships(self) -> None:
        """The display-field table keeps no pair that no checkpoint ships (such as HR's f32
        seat)."""
        shipped = {
            (dtype, entry.stored_dtype)
            for name, entry in BIREFNET_CHECKPOINTS.items()
            for dtype in birefnet_seat_dtypes(name)
        }

        assert set(BIREFNET_QUANT_SEATS) == shipped

    @pytest.mark.parametrize(
        ("checkpoint", "default"),
        [(BIREFNET_HR_CHECKPOINT, "f16"), (BIREFNET_LUCIDA_CHECKPOINT, "f32")],
    )
    def test_the_quant_seats_carry_display_fields_and_the_storage_width_is_the_default(
        self, tmp_path: Path, checkpoint: str, default: str
    ) -> None:
        """The display fields (ADR 0075) go from the seat table into the manifest as the same
        strings. The default is the upstream storage type."""
        sources = _build_birefnet_sources(tmp_path, checkpoint=checkpoint)
        out_dir = tmp_path / "models" / birefnet_repo_name(checkpoint)
        manifest = assemble_family(
            [birefnet_plan(sources, checkpoint)], out_dir, BIREFNET_DEFAULT_MODEL
        )
        model = _birefnet_model(manifest)
        stored = BIREFNET_CHECKPOINTS[checkpoint].stored_dtype

        for name, quant in model["quants"].items():
            seat = BIREFNET_QUANT_SEATS[(name, stored)]
            assert quant["label"] == seat.label
            assert quant["description"] == seat.description
        assert model["defaultQuant"] == default

    def test_a_single_seat_does_not_compare_itself_with_a_seat_that_is_not_there(self) -> None:
        """HR has one seat, f16 — if the display fields claimed a comparison with an f32 seat
        ("smaller download", "half-size"), the selection UI would show a sentence with nothing to
        compare against."""
        (quant,) = birefnet_quants(BIREFNET_HR_CHECKPOINT).values()
        text = f"{quant['label']} {quant['description']}".lower()

        for word in ("smaller", "larger", "largest", "less", "more", "most", "half"):
            assert re.search(rf"\b{word}\b", text) is None, word

    def test_the_card_measured_a_seat_each_repository_ships(self) -> None:
        """The card's run-time resources (stated as the numbers of the measured seat) were
        measured on a seat that repository ships."""
        for name, entry in BIREFNET_CHECKPOINTS.items():
            assert entry.resources.quant in birefnet_seat_dtypes(name)

    def test_the_quality_record_is_not_distributed(self, birefnet_assembled) -> None:
        out_dir, _ = birefnet_assembled
        assert list(out_dir.rglob("quality.json")) == []


class TestBirefnetStorageNotice:
    """NOTICE's storage description is built from the shipped seats and the checkpoint's storage
    (it does not contradict the content)."""

    def _notice(self, tmp_path: Path, checkpoint: str) -> str:
        return TestBirefnetLegalText()._legal(tmp_path, checkpoint)[1]

    @staticmethod
    def _prose(notice: str) -> str:
        """The text with line wrapping folded (the wording checks do not depend on where the lines
        break)."""
        return " ".join(notice.split())

    @pytest.mark.parametrize("checkpoint", [BIREFNET_HR_CHECKPOINT, BIREFNET_LUCIDA_CHECKPOINT])
    def test_the_storage_lines_are_wrapped_like_the_rest_of_the_template(
        self, tmp_path: Path, checkpoint: str
    ) -> None:
        notice = self._notice(tmp_path, checkpoint)
        assert max(len(line) for line in notice.splitlines()) <= 100

    @pytest.mark.parametrize("checkpoint", [BIREFNET_HR_CHECKPOINT, BIREFNET_LUCIDA_CHECKPOINT])
    def test_it_names_what_stays_f32_in_the_f16_seat(self, tmp_path: Path, checkpoint: str) -> None:
        """f16 storage covers linear / conv / embedding (the relative-position table). What stays
        f32 is the deformable sampling weights (`regular_conv` — the offset / modulator convs are
        f16) and bias / norm, and BN's α / β are values derived from the statistics (2026-09-26
        review A-3 / F4)."""
        prose = self._prose(self._notice(tmp_path, checkpoint))
        assert "linear, convolution and embedding (relative-position) weights" in prose
        assert "sampling weights (`regular_conv`)" in prose
        assert "derived per-channel affine in f32" in prose
        assert "the deformable convolutions' weights," not in prose

    @pytest.mark.parametrize("checkpoint", [BIREFNET_HR_CHECKPOINT, BIREFNET_LUCIDA_CHECKPOINT])
    def test_every_seat_is_described_and_no_other(self, tmp_path: Path, checkpoint: str) -> None:
        """NOTICE describes only that repository's seats (HR has f16 only — it does not claim an
        f32 seat it does not ship)."""
        notice = self._notice(tmp_path, checkpoint)
        seats = birefnet_seat_dtypes(checkpoint)
        for dtype in BIREFNET_WEIGHT_DTYPES:
            assert (f"  - `{dtype}`:" in notice) is (dtype in seats), dtype

    def test_the_hr_f16_seat_is_lossless_because_the_checkpoint_is_f16(
        self, tmp_path: Path
    ) -> None:
        """HR's checkpoint itself is f16 — the only seat (f16) keeps the values."""
        notice = self._prose(self._notice(tmp_path, BIREFNET_HR_CHECKPOINT))
        assert "the checkpoint itself is stored in f16" in notice
        # One seat, so it is stated in the singular ("every quant seat" would confuse readers of a
        # repository with one seat).
        assert (
            "**No quantization**: the quant seat stores the source checkpoint's own values"
            in notice
        )
        assert "`f16`: the checkpoint's own f16 values" in notice
        assert "**quantized**" not in notice
        # No f32 seat is shipped (a seat that only widens the values exactly — ADR 0113 addendum
        # 2026-10-11).
        assert "widened exactly from f16 to f32" not in notice
        # The old wording "source checkpoint's own f32 values" was untrue for HR (the upstream is
        # f16).
        assert "own f32 values" not in notice

    def test_the_lucida_f16_seat_is_declared_quantized(self, tmp_path: Path) -> None:
        """Lucida is an f32 checkpoint — the f16 seat holds rounded values and says so (it does not
        pretend to be lossless)."""
        notice = self._prose(self._notice(tmp_path, BIREFNET_LUCIDA_CHECKPOINT))
        assert "no quantization" not in notice.lower()
        assert "`f16`: **quantized**" in notice
        assert "`f32`: the checkpoint's own f32 values, unchanged." in notice
