"""Distribution recipe of vowel detection (`vowel_detector.distribution`) — unit tests over one
assembly pass.

No real assets are used. The input that reaches the assembly is a **valid minimal IR container** of
a few KB (`ir_fixtures`); only the cases that check that a check rejects something keep the
previous fake assets ({@link _vowel_detector_container}). A synthetic JSON is also enough for the
source of the feature contract and the mel basis (`feature_config.json`).

Since manifest v2 (`karume/2` — ADR 0041) the in-repository layout is uniformly "per-model subtree
+ `shared/`", so every expected path starts with `<model name>/…`.

The layers observable in the core alone (the size limit a synthetic plan suffices for, the complete
quant mapping, the staging/swap invariants, the resolution rule for attribution profiles) are held
by `tools/exporter/tests/test_dist.py` (the split of ADR 0065 steps 3+4). Observing the card
**over one assembly pass** is held here, and the per-template check (the advertised load entry
point) is in `vowel_detector/tests/test_card.py`.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from container_series import placed_paths, write_component
from ir_fixtures import Shape, ir_container
from safetensors.numpy import load_file

from _shared.licenses import MIT_COPYRIGHT_PLACEHOLDER, MIT_LICENSE_PATH
from dist import default_out_dir, main
from karume.dist import (
    MANIFEST_FILENAME,
    MODEL_CARD_FILENAME,
    DistError,
    assemble_family,
    resolve_card_renderer,
    verify_dist,
)
from karume.verify import verify_container
from vowel_detector.distribution import (
    PIPELINE,
    VOWEL_DETECTOR_COPYRIGHTS,
    VOWEL_DETECTOR_DEFAULT_MODEL,
    VOWEL_DETECTOR_GRAPH_ROLE,
    VOWEL_DETECTOR_MAX_FRAMES,
    VOWEL_DETECTOR_MIN_FRAMES,
    VOWEL_DETECTOR_OUTPUT_PATHS,
    VOWEL_DETECTOR_WEIGHTS,
    VowelDetectorSources,
    vowel_detector_plan,
    vowel_detector_repo_name,
    vowel_detector_series_name,
)


def _write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _in_subtree(model: str, paths: Iterable[str]) -> list[str]:
    """Expected paths inside the model subtree (the uniform layout of ADR 0041 §9)."""
    return [f"{model}/{rel}" for rel in paths]


def _placed_paths() -> list[str]:
    """Relative paths that appear in the distribution — **only weights entries** expand into the
    numbered parts (container-v1 §8).

    The mel basis is an assets entry (a one-file reference), so it is not split — in this family,
    which has both kinds of entry, the fact that the expansion applies to weights only shows up
    directly in the expected value.
    """
    return placed_paths(VOWEL_DETECTOR_OUTPUT_PATHS, VOWEL_DETECTOR_WEIGHTS)


def _present(out_dir: Path) -> list[str]:
    return sorted(str(path.relative_to(out_dir)) for path in out_dir.rglob("*") if path.is_file())


# ---- Vowel detection (audio → vowel sequence for lip-sync) ------------------------------------
#
# The fake assets are **one CRNN baked with a symbolic length** (input `2T` / output `T`) +
# upstream's `feature_config.json`. Three things are checked here: "only one graph is listed", "the
# time axis stays symbolic" and "the 4 fields of `pipelineConfig` are built from the upstream config
# and the script's declarations". The era of 4 length buckets ended with ADR 0056 / 0057.

#: Field names of `pipelineConfig` (a copy of `ROOT_KEYS` in the TS side's
#: `packages/models/src/vowel-detector/config.ts`). **The load side rejects both unknown and missing
#: keys at parse time**, so the field names of the baking side and the load side must match exactly.
_VOWEL_DETECTOR_CONFIG_KEYS = (
    "sampleRate",
    "featureDim",
    "classes",
    "minFrames",
    "maxFrames",
)

_VOWEL_DETECTOR_N_MELS = 80
_VOWEL_DETECTOR_N_FFT = 512
_VOWEL_DETECTOR_HOP = 160
_VOWEL_DETECTOR_CLASSES = ("a", "i", "u", "e", "o", "N", "pau", "cons")


def _vowel_detector_feature_config(**patch: Any) -> dict[str, Any]:
    """Skeleton of upstream's `assets/feature_config.json` (only the fields the checks read; the mel
    basis is a stand-in for uniform triangular windows)."""
    bins = _VOWEL_DETECTOR_N_FFT // 2 + 1
    basis = np.zeros((_VOWEL_DETECTOR_N_MELS, bins), dtype=np.float32)
    # Set exactly one element per row (the minimal shape that passes the empty-mel-channel check).
    for row in range(_VOWEL_DETECTOR_N_MELS):
        basis[row, row + 1] = 1.0
    return {
        "sample_rate": 16000,
        "n_fft": _VOWEL_DETECTOR_N_FFT,
        "hop": _VOWEL_DETECTOR_HOP,
        "n_mels": _VOWEL_DETECTOR_N_MELS,
        "feature_dim": _VOWEL_DETECTOR_N_MELS + 3,
        "classes": list(_VOWEL_DETECTOR_CLASSES),
        "mel_basis": basis.tolist(),
        **patch,
    }


def _vowel_detector_container(
    *,
    storage: str = "f32",
    feature_dim: int = _VOWEL_DETECTOR_N_MELS + 3,
    input_name: str = "features",
    in_shape: Sequence[Any] | None = None,
    out_shape: Sequence[Any] | None = None,
    outputs: int = 1,
) -> list[bytes]:
    """Contents of the vowel-detection graph placed in the series (the part sequence — part 0
    first).

    What reaches the assembly must be **a valid container** (the assembly opens its input and checks
    every rule of the declaration — `karume.dist.assert_weight_components_verified`), so the cases
    meant to fail a check are built in the same container too, **shifting only the declaration**
    away from the real one.

    `storage` is the storage of a mixed component as in the real thing (f16 / i8 / i4 keep their
    ineligible weights and biases in f32, and i4 additionally mixes in i8) — the check of the
    forbidden table ({@link VOWEL_DETECTOR_STORAGE_FORBIDDEN}) can be exercised only with this
    shape. The remaining arguments are the axes of the graph declaration, with defaults as in the
    real thing (the symbol `T` is bound by the `2T` of the input shape).
    """
    shape: Shape = list(in_shape) if in_shape is not None else [1, "2T", feature_dim]
    logits: Shape = (
        list(out_shape) if out_shape is not None else [1, "T", len(_VOWEL_DETECTOR_CLASSES)]
    )
    return ir_container(
        # The fake series also **names itself by the component name** (= the same convention as the
        # real thing — the container's graph name MUST be the manifest's weights key,
        # container-v1 §2.1).
        mark=VOWEL_DETECTOR_GRAPH_ROLE,
        storage=storage,
        inputs=((input_name, shape),),
        outputs=[logits] * outputs,
    )


def _build_vowel_detector_sources(
    root: Path,
    *,
    model: str = VOWEL_DETECTOR_DEFAULT_MODEL,
    feature_config: Mapping[str, Any] | None = None,
    omit_feature_config: bool = False,
    omit_graph: bool = False,
    **container: Any,
) -> VowelDetectorSources:
    """Reproduce one series and the upstream material with fake assets (including the `io.*` files
    that must not be shipped).

    The layout follows the real layout of `_shared.paths` (`outputs/series/` and `inputs/`) — so
    the CLI-path tests can point at the same tree just by swapping the root.
    """
    sources = VowelDetectorSources(
        series_dir=root / "outputs" / "series",
        model=root / "inputs" / "vowel-detector",
        model_name=model,
    )
    if not omit_graph:
        write_component(sources.series / "model.krm", _vowel_detector_container(**container))
        # E2E fixture that must not go into the distribution (the real series does contain it).
        _write(sources.series / "io.silence.safetensors", b"io-fixture")
    if not omit_feature_config:
        raw = feature_config if feature_config is not None else _vowel_detector_feature_config()
        _write(sources.model / "feature_config.json", json.dumps(raw).encode("utf-8"))
    return sources


@pytest.fixture
def vowel_detector_assembled(tmp_path: Path) -> tuple[Path, dict]:
    sources = _build_vowel_detector_sources(tmp_path)
    out_dir = tmp_path / "models" / vowel_detector_repo_name(VOWEL_DETECTOR_DEFAULT_MODEL)
    manifest = assemble_family(
        [vowel_detector_plan(sources, VOWEL_DETECTOR_DEFAULT_MODEL)],
        out_dir,
        VOWEL_DETECTOR_DEFAULT_MODEL,
        root_files=PIPELINE.root_files,
    )
    return out_dir, manifest


def _vowel_detector_model(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    return manifest["models"][VOWEL_DETECTOR_DEFAULT_MODEL]


class TestVowelDetectorLayout:
    def test_it_places_the_graph_under_the_model_subtree(self, vowel_detector_assembled) -> None:
        out_dir, _ = vowel_detector_assembled
        expected = _in_subtree(VOWEL_DETECTOR_DEFAULT_MODEL, _placed_paths())
        # The MIT license notice (ADR 0092 decision 7) is a meta file the manifest does not declare.
        assert _present(out_dir) == sorted([*expected, MANIFEST_FILENAME, "LICENSE.md"])

    def test_it_never_carries_the_io_fixtures(self, vowel_detector_assembled) -> None:
        out_dir, _ = vowel_detector_assembled
        assert list(out_dir.rglob("io.*")) == []

    def test_the_container_graph_is_named_after_the_weights_key(
        self, vowel_detector_assembled
    ) -> None:
        """The graph name that part 0 of the published container gives = the manifest's weights key.

        The runtime looks a graph up by name with `prepareContainer(opened, <weights key>)`
        (container-v1 §2.1). `karume dist` checks the two against each other before assembling
        (`DistError`), but here part 0 of the actually published file is read back to see, for one
        fake series, that the name is the key itself (so a spelling that moves outside the check is
        still noticed).
        """
        out_dir, manifest = vowel_detector_assembled
        weights = _vowel_detector_model(manifest)["weights"]
        assert weights, "with empty weights the assertions below would be vacuous"

        for key, labels in weights.items():
            for entry in labels.values():
                parts = [out_dir / ref["path"] for ref in entry["container"]["parts"]]
                assert list(verify_container(parts).read.graph.graphs) == [key]

    def test_the_graph_is_one_role_and_the_mel_basis_is_an_asset(
        self, vowel_detector_assembled
    ) -> None:
        """With a symbolic length the CRNN has exactly one role (the role axis of length buckets is
        gone)."""
        _, manifest = vowel_detector_assembled
        model = _vowel_detector_model(manifest)
        assert model["pipeline"] == "vowel-detector/1"
        assert list(model["weights"]) == ["crnn"]
        assert list(model["assets"]) == ["mel_basis"]
        assert list(model["quants"]) == ["f32"]
        assert model["defaultQuant"] == "f32"
        assert model["quants"]["f32"]["weights"] == {"crnn": "f32"}

    def test_it_writes_the_mel_basis_as_one_f32_tensor(self, vowel_detector_assembled) -> None:
        """Feature extraction is outside the graph, so the mel basis must ship as an asset or it
        cannot be reproduced."""
        out_dir, manifest = vowel_detector_assembled
        ref = _vowel_detector_model(manifest)["assets"]["mel_basis"]
        tensors = load_file(str(out_dir / ref["path"]))
        assert list(tensors) == ["mel_basis"]
        basis = tensors["mel_basis"]
        assert basis.dtype == np.float32
        assert basis.shape == (_VOWEL_DETECTOR_N_MELS, _VOWEL_DETECTOR_N_FFT // 2 + 1)

    def test_it_reassembles_over_a_previous_run(self, tmp_path: Path) -> None:
        sources = _build_vowel_detector_sources(tmp_path)
        out_dir = tmp_path / "models" / vowel_detector_repo_name(VOWEL_DETECTOR_DEFAULT_MODEL)
        first = assemble_family(
            [vowel_detector_plan(sources)], out_dir, VOWEL_DETECTOR_DEFAULT_MODEL
        )
        assert first == assemble_family(
            [vowel_detector_plan(sources)], out_dir, VOWEL_DETECTOR_DEFAULT_MODEL
        )
        assert verify_dist(out_dir)

    @pytest.mark.parametrize(("storage", "intruder"), [("f16", "f16"), ("i8", "i8"), ("i4", "i4")])
    def test_it_refuses_a_real_compressed_series_in_the_f32_seat(
        self, tmp_path: Path, storage: str, intruder: str
    ) -> None:
        """A compressed series as in the real thing **satisfies the requirement check** — only the
        forbidden table catches a mix-up of the series root.

        Compressed series also hold their ineligible weights and biases in F32, so "contains F32" is
        true, and unlike the single-dtype fake assets above not one byte fails the requirement
        check. The reason it should fail is "an asset of another series sits in the f32 slot",
        which the numeric checks cannot detect in principle (ADR 0027 / 0029).
        """
        sources = _build_vowel_detector_sources(tmp_path, storage=storage)
        with pytest.raises(
            DistError, match=rf"{VOWEL_DETECTOR_GRAPH_ROLE}: .* has (?:i8 / )?{intruder}"
        ):
            vowel_detector_plan(sources)

    def test_the_plain_f32_series_passes_both_storage_gates(self, tmp_path: Path) -> None:
        """The passing counterpart — a plain f32 series passes both the requirement check and the
        forbidden table (the forbidden table has not degenerated into rejecting everything)."""
        sources = _build_vowel_detector_sources(tmp_path, storage="f32")

        assert vowel_detector_plan(sources).name == VOWEL_DETECTOR_DEFAULT_MODEL


class TestVowelDetectorPipelineConfig:
    """`pipelineConfig` matches the load side's (`src/vowel-detector/config.ts`) schema down to the
    field names."""

    def test_it_declares_exactly_the_fields_the_loader_accepts(
        self, vowel_detector_assembled
    ) -> None:
        _, manifest = vowel_detector_assembled
        assert (
            tuple(_vowel_detector_model(manifest)["pipelineConfig"]) == _VOWEL_DETECTOR_CONFIG_KEYS
        )

    def test_it_takes_the_feature_contract_from_the_upstream_config(
        self, vowel_detector_assembled
    ) -> None:
        _, manifest = vowel_detector_assembled
        config = _vowel_detector_model(manifest)["pipelineConfig"]
        assert config["sampleRate"] == 16000
        assert config["featureDim"] == _VOWEL_DETECTOR_N_MELS + 3
        assert config["classes"] == list(_VOWEL_DETECTOR_CLASSES)

    def test_it_declares_the_operating_limit_the_loader_requires(
        self, vowel_detector_assembled
    ) -> None:
        """The upper bound exists only in the distribution (the IR holds no ranges for symbols and
        the load side holds no constant)."""
        _, manifest = vowel_detector_assembled
        config = _vowel_detector_model(manifest)["pipelineConfig"]
        assert config["maxFrames"] == VOWEL_DETECTOR_MAX_FRAMES

    def test_the_limit_is_the_symbolic_maximum_the_export_script_baked(self) -> None:
        """The shipped upper bound matches the upper bound of the baked symbolic dimension.

        `SYM_MAX` counts the 20 ms grid and the shipped `maxFrames` counts the 10 ms grid (the input
        is `2T`). If they drift, the user hits "a length the declaration allows but the graph does
        not accept" on their machine.
        """
        from vowel_detector import export

        assert VOWEL_DETECTOR_MAX_FRAMES == 2 * export.SYM_MAX

    def test_it_refuses_a_feature_dim_that_is_not_mel_plus_dsp(self, tmp_path: Path) -> None:
        """A config whose breakdown disagrees with upstream is refused even if its shape fits the
        graph."""
        sources = _build_vowel_detector_sources(
            tmp_path, feature_config=_vowel_detector_feature_config(feature_dim=90)
        )
        with pytest.raises(DistError, match="the feature breakdown disagrees with upstream"):
            vowel_detector_plan(sources)

    def test_it_needs_the_upstream_feature_config(self, tmp_path: Path) -> None:
        sources = _build_vowel_detector_sources(tmp_path, omit_feature_config=True)
        with pytest.raises(DistError, match=r"Assembly input is missing: .*feature_config\.json"):
            vowel_detector_plan(sources)


class TestVowelDetectorMelBasis:
    """The mel basis is part of the numeric path itself — its shape and contents are checked only
    here."""

    def test_it_refuses_a_basis_shaped_for_another_n_fft(self, tmp_path: Path) -> None:
        config = _vowel_detector_feature_config()
        config["mel_basis"] = [row[:-1] for row in config["mel_basis"]]
        sources = _build_vowel_detector_sources(tmp_path, feature_config=config)
        with pytest.raises(DistError, match=r"expected .* from n_mels / n_fft"):
            vowel_detector_plan(sources)

    def test_it_refuses_an_empty_mel_channel(self, tmp_path: Path) -> None:
        """A triangular window that fell outside the band pins that one column to the log floor
        forever (silent degradation)."""
        config = _vowel_detector_feature_config()
        config["mel_basis"][7] = [0.0] * len(config["mel_basis"][7])
        sources = _build_vowel_detector_sources(tmp_path, feature_config=config)
        with pytest.raises(DistError, match="empty mel channels"):
            vowel_detector_plan(sources)

    def test_it_refuses_a_non_finite_basis(self, tmp_path: Path) -> None:
        config = _vowel_detector_feature_config()
        config["mel_basis"][3][5] = float("nan")
        sources = _build_vowel_detector_sources(tmp_path, feature_config=config)
        with pytest.raises(DistError, match="non-finite elements"):
            vowel_detector_plan(sources)


class TestVowelDetectorGraphGate:
    """Assembly check — rejects, before placement, combinations that would still form a valid
    distribution despite the mismatch."""

    def test_it_refuses_a_graph_baked_for_a_fixed_length(self, tmp_path: Path) -> None:
        """A fixed-length graph has the same names and ranks = shipping it still forms a manifest.

        On the user's machine **every audio clip of any other length** fails at run time (it shows
        up only after distribution).
        """
        sources = _build_vowel_detector_sources(
            tmp_path,
            in_shape=[1, 500, _VOWEL_DETECTOR_N_MELS + 3],
            out_shape=[1, 250, len(_VOWEL_DETECTOR_CLASSES)],
        )
        with pytest.raises(DistError, match="symbolic dimensions"):
            vowel_detector_plan(sources)

    def test_it_refuses_a_graph_whose_input_is_not_twice_the_symbol(self, tmp_path: Path) -> None:
        """With the input left at `T`, the run-time binding is off by a factor of 2 and the `.lab`
        times stretch."""
        sources = _build_vowel_detector_sources(
            tmp_path, in_shape=[1, "T", _VOWEL_DETECTOR_N_MELS + 3]
        )
        with pytest.raises(DistError, match="the length on the 10 ms grid is twice the symbol T"):
            vowel_detector_plan(sources)

    def test_it_refuses_a_graph_with_another_feature_dim(self, tmp_path: Path) -> None:
        sources = _build_vowel_detector_sources(tmp_path, feature_dim=80)
        with pytest.raises(DistError, match="expected"):
            vowel_detector_plan(sources)

    def test_it_refuses_a_graph_whose_output_is_not_the_20ms_grid(self, tmp_path: Path) -> None:
        """An output baked on the 10 ms grid stretches the `.lab` times by a factor of 2."""
        sources = _build_vowel_detector_sources(
            tmp_path, out_shape=[1, "2T", len(_VOWEL_DETECTOR_CLASSES)]
        )
        with pytest.raises(DistError, match="20 ms grid"):
            vowel_detector_plan(sources)

    def test_it_refuses_a_graph_with_a_second_output(self, tmp_path: Path) -> None:
        sources = _build_vowel_detector_sources(tmp_path, outputs=2)
        with pytest.raises(DistError, match="exactly one logits output"):
            vowel_detector_plan(sources)

    def test_it_refuses_a_graph_with_a_second_symbol(self, tmp_path: Path) -> None:
        """The time axis is the only symbol (the host cannot pass a binding for a second one).

        The shape of an export that made the feature axis dynamic too — the second symbol is bound
        by the input shape.
        """
        sources = _build_vowel_detector_sources(tmp_path, in_shape=(1, "2T", "S"))
        with pytest.raises(DistError, match="symbolic dimensions"):
            vowel_detector_plan(sources)

    def test_it_refuses_a_renamed_input(self, tmp_path: Path) -> None:
        """The runtime binds by name, so a changed spelling cannot be bound."""
        sources = _build_vowel_detector_sources(tmp_path, input_name="mel")
        with pytest.raises(DistError, match="graph inputs"):
            vowel_detector_plan(sources)

    def test_it_refuses_a_missing_graph(self, tmp_path: Path) -> None:
        sources = _build_vowel_detector_sources(tmp_path, omit_graph=True)
        with pytest.raises(DistError, match="Assembly input is missing"):
            vowel_detector_plan(sources)


class TestVowelDetectorModelCard:
    def _run(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        """Run the CLI once with fake assets only (the upstream material's location is moved to tmp
        too)."""
        from vowel_detector import distribution

        sources = _build_vowel_detector_sources(tmp_path)
        # `INPUTS_ROOT` (the upstream material outside the series) is looked up on the recipe side —
        # patch this binding, not the driver's `dist` (`DIST_ROOT` is the reverse: the driver side).
        monkeypatch.setattr(distribution, "INPUTS_ROOT", tmp_path / "inputs")
        out_dir = tmp_path / "dist"
        main(
            [
                "--pipeline",
                "vowel-detector",
                "--series",
                str(sources.series_dir),
                "--out",
                str(out_dir),
            ]
        )
        return out_dir

    def test_it_describes_the_lip_sync_distribution(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        card = (self._run(tmp_path, monkeypatch) / MODEL_CARD_FILENAME).read_text(encoding="utf-8")
        assert "pipeline_tag: audio-classification" in card
        assert "license: mit" in card
        # A distribution that does not change the storage is none of the 4 values (left to the Hub's
        # inference).
        assert "base_model_relation" not in card
        # "One graph for any length" and the upper bound are exactly the constraints a user checks
        # first (the numbers come from the manifest).
        assert "**One graph for every length from 40 ms to 600.0 s.**" in card
        assert "longer than 600.0 s is rejected rather than silently truncated" in card
        assert f"**{VOWEL_DETECTOR_MAX_FRAMES} frames of 10 ms** (600.0 s)" in card

    def test_the_usage_snippet_enumerates_the_choices_from_the_manifest(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The form of the ruling of 2026-08-12: options are listed alongside as comments, and the
        selectable values are enumerated from the manifest.

        `detect()` has no run-time knobs, so this pipeline's only options are model and quant.
        Even with only one entry it is spelled "available:" so that the wording follows
        automatically when entries are added (baked-in text like "the only one this repository
        ships" becomes a lie the moment one is added).
        """
        card = (self._run(tmp_path, monkeypatch) / MODEL_CARD_FILENAME).read_text(encoding="utf-8")
        assert (
            f'  // model: "{VOWEL_DETECTOR_DEFAULT_MODEL}",'
            f" // default — available: {VOWEL_DETECTOR_DEFAULT_MODEL}" in card
        )
        assert '  // quant: "f32", // default — available: f32' in card
        assert "the only one this repository ships" not in card

    def test_it_carries_the_upstream_training_attributions(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The attributions upstream's NOTICE.txt explicitly asks to keep when the model is
        distributed."""
        card = (self._run(tmp_path, monkeypatch) / MODEL_CARD_FILENAME).read_text(encoding="utf-8")
        for source in (
            "reazon-research/japanese-hubert-base-k2",
            "ROHAN4600",
            "ITA corpus",
            "Common Voice ja",
            "Style-Bert-VITS2",
            "AivisHub",
            "ACML 1.0",
            "JSUT basic5000",
        ):
            assert source in card, source
        # It also states that the teacher model and the synthesis engine are **not bundled** (to
        # prevent misreading the AGPL-3.0).
        assert "the teacher is **not** part of these weights" in card
        assert "not distributed with, and not part of, these weights" in card

    def test_the_card_comes_from_a_verified_distribution(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        assert verify_dist(self._run(tmp_path, monkeypatch))


class TestVowelDetectorCli:
    def test_the_repository_name_does_not_carry_the_checkpoint_generation(self) -> None:
        """The generation is a fact spelled by the manifest keys; a new repository is not created
        each time the generation goes up."""
        assert (
            default_out_dir(PIPELINE, [VOWEL_DETECTOR_DEFAULT_MODEL]).name
            == "karume-vowel-detector"
        )

    def test_one_attribution_profile_needs_no_choice(self) -> None:
        profiles = PIPELINE.card_profiles
        assert len(profiles) == 1
        assert resolve_card_renderer(PIPELINE, None) is next(iter(profiles.values()))

    def test_the_series_name_follows_the_exporter(self) -> None:
        """The series name is the same expression as `vowel_detector.export.default_out_dir` (no
        second table)."""
        assert (
            vowel_detector_series_name(VOWEL_DETECTOR_DEFAULT_MODEL) == "vowel-detector-crnn-epoch3"
        )

    def test_it_assembles_into_the_pipeline_default_directory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import dist
        from vowel_detector import distribution

        sources = _build_vowel_detector_sources(tmp_path)
        # `DIST_ROOT` is on the driver side, which decides the default output location
        # (`dist.default_out_dir`); `INPUTS_ROOT` is on the recipe side, which looks up the inputs
        # outside the series — the bindings of the two modules are patched separately.
        monkeypatch.setattr(dist, "DIST_ROOT", tmp_path / "models")
        monkeypatch.setattr(distribution, "INPUTS_ROOT", tmp_path / "inputs")

        main(["--pipeline", "vowel-detector", "--series", str(sources.series_dir)])

        out_dir = tmp_path / "models" / "karume-vowel-detector"
        expected = _in_subtree(VOWEL_DETECTOR_DEFAULT_MODEL, _placed_paths())
        assert sorted(verify_dist(out_dir)) == sorted(expected)


class TestVowelDetectorLegalText:
    """The MIT license notice at the root of the distribution repository (ADR 0092 decision 7)."""

    def test_it_ships_the_license_text_byte_identical(self, vowel_detector_assembled) -> None:
        """The body is verbatim from the original — not one byte moves outside the insertion point
        (the copyright line).

        Reformatting or newline conversion anywhere on the assembly path would move a byte, but the
        prose would stay valid, so the other checks would let it through.
        """
        out_dir, _ = vowel_detector_assembled
        template = MIT_LICENSE_PATH.read_text(encoding="utf-8")
        rendered = (out_dir / "LICENSE.md").read_text(encoding="utf-8")

        assert rendered == template.replace(
            MIT_COPYRIGHT_PLACEHOLDER, "\n".join(VOWEL_DETECTOR_COPYRIGHTS)
        )

    def test_it_names_a_copyright_holder(self, vowel_detector_assembled) -> None:
        """An MIT text without a copyright line has nothing for "the above copyright notice" to
        point to."""
        out_dir, _ = vowel_detector_assembled
        rendered = (out_dir / "LICENSE.md").read_text(encoding="utf-8")

        assert MIT_COPYRIGHT_PLACEHOLDER not in rendered
        for line in VOWEL_DETECTOR_COPYRIGHTS:
            assert line in rendered


class TestVowelDetectorFeatureContract:
    """The feature-extraction dimensions are checked against **the values baked into the only
    reader** (W-P7-5).

    Without the check, the result is "assembly green, `verify_dist` green, a distribution refused
    at load time" (it fails on a real device after upload to HF).
    """

    @pytest.mark.parametrize(
        ("key", "value"),
        [
            ("sample_rate", 22050),
            ("n_fft", 1024),
            ("hop", 256),
            ("n_mels", 64),
        ],
    )
    def test_it_refuses_a_dimension_the_loader_cannot_read(
        self, tmp_path: Path, key: str, value: int
    ) -> None:
        config = _vowel_detector_feature_config(**{key: value})
        if key == "n_mels":
            # The point is to see it fail here before the breakdown check, so feature_dim is aligned
            # too.
            config["feature_dim"] = value + 3
        sources = _build_vowel_detector_sources(tmp_path, feature_config=config)

        with pytest.raises(DistError, match=f"{key} is {value}"):
            vowel_detector_plan(sources)

    def test_the_card_and_the_gate_use_the_same_hop(self) -> None:
        """The hop the card uses for seconds is the same number the assembly checks against
        upstream."""
        from vowel_detector.card import VOWEL_DETECTOR_HOP

        assert VOWEL_DETECTOR_HOP == 160

    @pytest.mark.parametrize(
        ("key", "value"),
        [("sample_rate", True), ("n_mels", 0), ("n_fft", "512"), ("hop", -160)],
    )
    def test_it_refuses_a_dimension_that_is_not_a_positive_integer(
        self, tmp_path: Path, key: str, value: object
    ) -> None:
        """`True` is a subtype of `int` in Python — a loose type check lets a bool through as a
        dimension."""
        sources = _build_vowel_detector_sources(
            tmp_path, feature_config=_vowel_detector_feature_config(**{key: value})
        )

        with pytest.raises(DistError, match=f"{key}"):
            vowel_detector_plan(sources)


class TestVowelDetectorClasses:
    """Structural check of the class vocabulary (the order is the class id, so a missing or
    duplicated entry is a label substitution)."""

    def test_it_refuses_a_vocabulary_of_another_length(self, tmp_path: Path) -> None:
        sources = _build_vowel_detector_sources(
            tmp_path,
            feature_config=_vowel_detector_feature_config(
                classes=list(_VOWEL_DETECTOR_CLASSES[:-1])
            ),
        )

        with pytest.raises(DistError, match="not an array of length 8"):
            vowel_detector_plan(sources)

    def test_it_refuses_a_duplicated_label(self, tmp_path: Path) -> None:
        """A duplicate means "one class silently takes another label"."""
        duplicated = ["a", *_VOWEL_DETECTOR_CLASSES[2:], "a"]
        sources = _build_vowel_detector_sources(
            tmp_path, feature_config=_vowel_detector_feature_config(classes=duplicated)
        )

        with pytest.raises(DistError, match="has duplicates"):
            vowel_detector_plan(sources)

    @pytest.mark.parametrize("bad", ["", 1, None])
    def test_it_refuses_a_label_that_is_not_a_non_empty_string(
        self, tmp_path: Path, bad: object
    ) -> None:
        classes = [bad, *_VOWEL_DETECTOR_CLASSES[1:]]
        sources = _build_vowel_detector_sources(
            tmp_path, feature_config=_vowel_detector_feature_config(classes=classes)
        )

        with pytest.raises(DistError, match="not a non-empty string"):
            vowel_detector_plan(sources)


class TestVowelDetectorFrameFloor:
    """The **lower end** of the input length is also a declaration of the distribution (E-M5-2) —
    the IR holds no ranges for symbols."""

    def test_it_declares_the_operating_floor_the_loader_requires(
        self, vowel_detector_assembled
    ) -> None:
        _, manifest = vowel_detector_assembled
        config = _vowel_detector_model(manifest)["pipelineConfig"]

        assert config["minFrames"] == VOWEL_DETECTOR_MIN_FRAMES

    def test_the_floor_is_the_symbolic_minimum_the_export_script_baked(self) -> None:
        """The shipped lower bound matches the lower bound of the baked symbolic dimension.

        `SYM_MIN` counts the 20 ms grid and the shipped `minFrames` counts the 10 ms grid (the input
        is `2T`). If they drift, audio shorter than declared binds `T = 1` and runs outside the
        traced range (no exception is raised).
        """
        from vowel_detector import export

        assert VOWEL_DETECTOR_MIN_FRAMES == 2 * export.SYM_MIN
        assert VOWEL_DETECTOR_MIN_FRAMES == export.MIN_LENGTH

    def test_the_floor_is_below_the_limit(self) -> None:
        """It forms a valid range (an inverted declaration would accept nothing)."""
        assert VOWEL_DETECTOR_MIN_FRAMES < VOWEL_DETECTOR_MAX_FRAMES
