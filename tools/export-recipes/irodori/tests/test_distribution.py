"""The Irodori distribution recipe (`irodori.distribution`) — unit tests of one assembly run.

No real assets. The inputs reaching the assembly are **valid minimal IR containers** of a few KB
(`ir_fixtures`); only the cases that check a refusal by a check keep the old fake assets
({@link _irodori_container}). Irodori also reads the checkpoint's `config_json` and the codec's
`metadata.json`, but synthetic JSON is enough for both (**the synthetic config differs from the
real v4-small weights in every number** — baked-in numbers would fail).

Since manifest v2 (`karume/2` — ADR 0041) the in-repository layout is uniformly "a subtree per
model + `shared/`", so every expected path starts with `<model name>/…`.

The layers observable with core alone (the size limit a synthetic plan is enough for, the
complete quant mapping, the staging/swap invariants, the resolution rules of the attribution
profiles) live in `tools/exporter/tests/test_dist.py` (the split of ADR 0065 stage 3+4). Card
rendering is checked here **for one assembly run**, and the per-template checks (the load entry
points it points to) are in `irodori/tests/test_card.py`.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from container_series import part_paths, placed_paths, replace_component, write_component
from ir_fixtures import ir_container
from safetensors.numpy import save

from _shared.licenses import apache_license_2_0, mit_license
from dist import default_out_dir, main
from irodori.card import (
    IRODORI_CODEC_MODEL,
    IRODORI_CODEC_ORIGIN_MODEL,
    IRODORI_CODEC_PARENT_MODEL,
    IRODORI_TEXT_BACKBONE_MODEL,
    IRODORI_UPSTREAMS,
)
from irodori.distribution import (
    CALIB_PROVENANCE_FILE,
    CALIB_SHIPPABLE_METHOD,
    IRODORI_CODEC_DIRS,
    IRODORI_CODEC_HALO_FRAMES,
    IRODORI_CODEC_NAME,
    IRODORI_COPYRIGHTS,
    IRODORI_DEFAULT_MODEL,
    IRODORI_DTYPE_ROLES,
    IRODORI_GRAPH_ROLES,
    IRODORI_NOTICE_MARKDOWN,
    IRODORI_OUTPUT_PATHS,
    IRODORI_QUANT_ABBREVIATIONS,
    IRODORI_QUANT_SEATS,
    IRODORI_SAMPLING_DEFAULTS,
    IRODORI_SERIES_DIRS,
    IRODORI_STORAGE_FORBIDDEN,
    IRODORI_STORAGE_REQUIREMENTS,
    IRODORI_V4_MODEL,
    IRODORI_V41_MODEL,
    IRODORI_V41_SHARED_ROLES,
    IRODORI_WEIGHT_DTYPES,
    IRODORI_WEIGHTS,
    PIPELINE,
    IrodoriSources,
    irodori_calib_floor,
    irodori_license_markdown,
    irodori_plan,
    irodori_references,
    irodori_repo_name,
    irodori_series_name,
    irodori_sources,
)
from karume.dist import (
    LEGAL_PATHS,
    MANIFEST_FILENAME,
    MODEL_CARD_FILENAME,
    DistError,
    assemble_family,
    assert_quant_presentation,
    resolve_card_renderer,
    verify_dist,
)


def _fake_checkpoint(metadata: Mapping[str, str] | None = None) -> bytes:
    """A stand-in for the upstream checkpoint (one tensor + `__metadata__`).

    The assembly reads only `config_json` from `__metadata__`, so the contents can be anything,
    but the file must be **a real safetensors** (the reader is the strict `safetensors` reader).
    """
    return save({"w": np.zeros(1, dtype=np.float32)}, metadata=dict(metadata or {}))


def _write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _in_subtree(model: str, paths: Iterable[str]) -> list[str]:
    """The expected paths inside the model subtree (the uniform layout of ADR 0041 §9)."""
    return [f"{model}/{rel}" for rel in paths]


def _placed_paths() -> list[str]:
    """The relative paths that appear in the distribution — **only the weights entries** expand
    into numbered parts (container-v1 §8).

    The tokenizer is an assets entry (a single-file reference), so it is not split.
    """
    return placed_paths(IRODORI_OUTPUT_PATHS, IRODORI_WEIGHTS)


def _stored_paths(model: str) -> list[str]:
    """The relative paths `model`'s own repository stores: {@link _placed_paths} without the roles
    it references in another repository ({@link irodori_references})."""
    borrowed = {role for reference in irodori_references(model) for role in reference.roles}
    stored = {role: path for role, path in IRODORI_OUTPUT_PATHS.items() if role not in borrowed}
    return placed_paths(stored, IRODORI_WEIGHTS)


def _present(out_dir: Path) -> list[str]:
    return sorted(str(path.relative_to(out_dir)) for path in out_dir.rglob("*") if path.is_file())


#: The synthetic checkpoint config. **Every number differs from the real v4-small weights**
#: (latent 32 / speaker 768 / reference 120s …) — if pipelineConfig were baked in, a manifest
#: built from this config would claim the real weights' numbers and fail. Only `latent_patch_size`
#: is 1 (the only condition under which the TS-side `latentDim` can serve both roles; the value
#: itself is a check).
_IRODORI_CONFIG: Mapping[str, Any] = {
    "latent_dim": 8,
    "latent_patch_size": 1,
    "speaker_patch_size": 2,
    "speaker_dim": 24,
    "text_dim": 12,
    "caption_dim": 16,
    "timestep_embed_dim": 6,
    "max_text_len": 10,
    "max_caption_len": 14,
    "ref_max_seconds": 8.0,
}

#: The numbers derived from the config above (one place, so the tests do not copy the formulas).
_IRODORI_SPEAKER_ROWS = 101  # int(8.0 × 25) // 2 + 1
_IRODORI_DIT_SYM_MAX = 750  # int(30.0 × 25) // 1
_IRODORI_MASK_TOTAL = 10 + _IRODORI_SPEAKER_ROWS + 14

#: The hidden width of the backbone (the projector's input — a number pipelineConfig does not
#: carry).
_IRODORI_HIDDEN = 32

#: The block count of the synthetic DiT and the shape of one condition-side K/V
#: (`[1, sum of the 3 condition segments, heads, head_dim]`). **Numbers different from the real
#: weights' 12 blocks / 20 heads × 64** — if the assembly baked in the block count, it would fail
#: on this synthetic one (the number of boundary tensors is derived from the outputs of
#: `dit_context` — ADR 0114).
_IRODORI_DIT_BLOCKS = 2
_IRODORI_CONTEXT_SHAPE = [1, _IRODORI_MASK_TOTAL, 2, 4]

#: The fake codec's `metadata.json` (the shape `irodori/dacvae/convert.py` writes). **Numbers
#: different from the real one** (not 48kHz / hop 1920), so baked-in `sampleRate` / `hopLength`
#: would fail. The combination is chosen to fit `frameRate` 25 (12,000 = 25 × 480).
_IRODORI_CODEC_METADATA: Mapping[str, Any] = {
    "kwargs": {"sample_rate": 12_000, "encoder_rates": [2, 4, 60]}
}
_IRODORI_HOP_LENGTH = 480

#: The record of the i4 series' calibration conditions (the shape
#: `irodori.export._write_calib_provenance` writes).
#:
#: The two budget fields have **the same values as the shipped series** (`calib_provenance.json`
#: of `outputs/series/irodori-v4-small-i4/` = 12 cases × 40 steps). Agreement with the floor the
#: check uses is checked against the source of truth by
#: {@link TestIrodoriCalibProvenance.test_the_floor_is_the_condition_the_export_defaults_to} — if
#: the values copied here go stale, that test fails.
_IRODORI_CALIB_PROVENANCE: Mapping[str, Any] = {
    "method": CALIB_SHIPPABLE_METHOD,
    "grid": "rtn",
    "group_size": 32,
    "cases": 12,
    "steps": 40,
}

#: The field names of `pipelineConfig` (a copy of `ROOT_KEYS` in the TS-side
#: `packages/models/src/irodori/config.ts`). **The loader rejects both unknown and missing keys at
#: parse time**, so the field names of the writer and the loader must match exactly. The test
#: holds a copy because this is the only place that fails when only one side moves.
_IRODORI_CONFIG_KEYS = (
    "maxTextLen",
    "maxCaptionLen",
    "speakerRows",
    "ditSymMax",
    "frameRate",
    "sampleRate",
    "hopLength",
    "codecHaloFrames",
    "latentDim",
    "speakerPatchSize",
    "speakerDim",
    "textDim",
    "captionDim",
    "timestepEmbedDim",
    "steps",
    "initScale",
    "cfgMinT",
    "cfgMaxT",
    "cfgScales",
    "minSeconds",
    "maxSeconds",
    "speakerUncondMode",
    "cfgGuidanceMode",
)


#: The shape of one graph (input names and shapes, outputs, symbol name). Outputs are a count
#: (each `[1]`, default names) or a list of named `(name, shape)` (for `dit_context`, whose outputs
#: the runtime binds by name).
_Spec = tuple[list[tuple[str, list[Any]]], int | list[tuple[str, list[Any]]], str]


def _context_kv(blocks: int = _IRODORI_DIT_BLOCKS) -> list[tuple[str, list[Any]]]:
    """Names and shapes of the condition-side K/V boundary (the outputs of `dit_context` = the
    inputs of `dit` from the fourth on)."""
    return [
        (name, list(_IRODORI_CONTEXT_SHAPE))
        for index in range(blocks)
        for name in (f"context_k_{index}", f"context_v_{index}")
    ]


def _irodori_specs() -> dict[str, _Spec]:
    """The shapes of the 9 graphs — consistent with {@link _IRODORI_CONFIG}.

    Both the cases a check must refuse and the valid containers ({@link _irodori_input}) are built
    from here — with two sources of truth for the shapes, the day only one moved a check would
    silently look at another shape.
    """
    latent_dim = _IRODORI_CONFIG["latent_dim"]
    speaker_dim = _IRODORI_CONFIG["speaker_dim"]
    text_dim = _IRODORI_CONFIG["text_dim"]
    caption_dim = _IRODORI_CONFIG["caption_dim"]
    return {
        "backbone": ([("input_ids", [1, "T"])], 1, "T"),
        "text_proj": ([("hidden", [1, "T", _IRODORI_HIDDEN])], 1, "T"),
        "caption_proj": ([("hidden", [1, "T", _IRODORI_HIDDEN])], 2, "T"),
        "speaker": (
            [("latent", [1, "S", latent_dim * _IRODORI_CONFIG["speaker_patch_size"]])],
            1,
            "S",
        ),
        "duration": (
            [
                ("text_state", [1, "T", text_dim]),
                ("speaker_vec", [1, speaker_dim]),
                ("has_speaker", [1, 1]),
                ("caption_vec", [1, caption_dim]),
                ("has_caption", [1, 1]),
            ],
            1,
            "T",
        ),
        # The DiT is split in two (ADR 0114) — the condition states are inputs of `dit_context`,
        # and `dit` takes the condition-side K/V as inputs.
        "dit": (
            [
                ("x_t", [1, "S", latent_dim]),
                ("t_embed", [1, _IRODORI_CONFIG["timestep_embed_dim"]]),
                ("mask", [1, 1, 1, f"S+{_IRODORI_MASK_TOTAL}"]),
                *_context_kv(),
            ],
            1,
            "S",
        ),
        "dit_context": (
            [
                ("text_state", [1, _IRODORI_CONFIG["max_text_len"], text_dim]),
                ("speaker_state", [1, _IRODORI_SPEAKER_ROWS, speaker_dim]),
                ("caption_state", [1, _IRODORI_CONFIG["max_caption_len"], caption_dim]),
            ],
            _context_kv(),
            "",
        ),
        # The two codec graphs (a separate series, pure convolution). Their input widths fit
        # latentDim / hopLength.
        "codec_decoder": ([("latent", [1, "S", latent_dim])], 1, "S"),
        "codec_encoder": ([("wav", [1, "T", _IRODORI_HOP_LENGTH])], 1, "T"),
    }


def _irodori_input(dtype: str, role: str, spec: _Spec | None = None) -> list[bytes]:
    """The input of one series that reaches the assembly (the parts of **a valid container**).

    The assembly opens the input containers and applies every rule of the declarations
    (`karume.dist.assert_weight_components_verified`), so the cases a check must refuse are built
    the same way and **only the declarations differ from the real ones** (`spec` is that axis).
    The storage vocabulary has the same shape as the real one (only eligible weights compressed,
    bias / constants / scales f32, i4 a mix of i4 + i8 + f32), so the check against preset
    mix-ups ({@link IRODORI_STORAGE_FORBIDDEN}) applies to this shape the same way.
    """
    inputs, outputs, _symbol = _irodori_specs()[role] if spec is None else spec
    named_outputs = [(None, [1]) for _ in range(outputs)] if isinstance(outputs, int) else outputs
    return ir_container(
        mark=f"irodori-{role}-{dtype}",
        # The pseudo series also **names itself after the component** (the container's graph
        # name = the manifest's weights key, MUST, container-v1 §2.1 — the assembly check looks
        # at this match).
        named=role,
        storage=dtype,
        inputs=tuple((name, shape) for name, shape in inputs),
        outputs=[shape for _name, shape in named_outputs],
        output_names=None if isinstance(outputs, int) else [name for name, _ in named_outputs],
    )


def _build_irodori_sources(
    root: Path,
    *,
    model: str = IRODORI_DEFAULT_MODEL,
    config: Mapping[str, Any] = _IRODORI_CONFIG,
    specs: Mapping[str, _Spec] | None = None,
    codec_metadata: Mapping[str, Any] | None = _IRODORI_CODEC_METADATA,
    calib_provenance: Mapping[str, Any] | None = _IRODORI_CALIB_PROVENANCE,
) -> IrodoriSources:
    """Recreate the series + checkpoint locations with fake assets (including things that must not
    be distributed).

    The layout follows the real one of `_shared.paths` (`outputs/series/` and `inputs/`) — so the
    tests of the CLI path can point at the same tree just by swapping the root. The codec is **a
    separate series with separate input material** (`dacvae-32dim`), so it goes into two
    directories apart from the Irodori body. One series per storage dtype (`IRODORI_DTYPE_ROLES`
    — only i4 covers the DiT's two roles, so its series has two directories as well).
    """
    series_root = root / "outputs" / "series"
    suffix = {dtype: "" if dtype == "f32" else f"-{dtype}" for dtype in IRODORI_WEIGHT_DTYPES}
    by_dtype = {
        dtype: series_root / f"{irodori_series_name(model)}{tail}" for dtype, tail in suffix.items()
    }
    codec_by_dtype = {
        dtype: series_root / f"{IRODORI_CODEC_NAME}{tail}"
        for dtype, tail in suffix.items()
        if not set(IRODORI_DTYPE_ROLES[dtype]).isdisjoint(IRODORI_CODEC_DIRS)
    }
    sources = IrodoriSources(
        model=root / "inputs" / "irodori" / model,
        codec_model=root / "inputs" / "irodori" / IRODORI_CODEC_NAME,
        series_by_dtype=by_dtype,
        codec_series_by_dtype=codec_by_dtype,
    )
    for dtype, roles in IRODORI_DTYPE_ROLES.items():
        for role in roles:
            in_codec = role in IRODORI_CODEC_DIRS
            series = (
                sources.codec_series_by_dtype[dtype] if in_codec else sources.series_by_dtype[dtype]
            )
            directory = (IRODORI_SERIES_DIRS | IRODORI_CODEC_DIRS)[role]
            # The forms that name synthetic metadata (cases that test a check) stop at the
            # planning stage and never reach the assembly, so the old fake containers are fine.
            payload = _irodori_input(dtype, role, (specs or {}).get(role))
            write_component(series / directory / "model.krm", payload)
            # E2E fixtures that must not enter the distribution (the series really holds them).
            _write(series / directory / "io.case0.safetensors", b"io-fixture")
    if calib_provenance is not None:
        # The record of the calibration conditions (`irodori.export._write_calib_provenance`
        # writes the same one directly under both i4 DiT graphs).
        for role in IRODORI_DTYPE_ROLES["i4"]:
            _write(
                sources.series_by_dtype["i4"] / IRODORI_SERIES_DIRS[role] / CALIB_PROVENANCE_FILE,
                json.dumps(calib_provenance, ensure_ascii=False).encode("utf-8"),
            )
    if codec_metadata is not None:
        _write(
            sources.codec_model / "metadata.json",
            json.dumps(codec_metadata, ensure_ascii=False).encode("utf-8"),
        )
    _write(sources.series / "tokenizer" / "tokenizer.json", b'{"vocabText": "a"}')
    # The three tokenizer goldens are for checks (not needed to run, so not in the distribution).
    for name in ("golden.encode.json", "golden.normalize.json", "nfkc-diff.json"):
        _write(sources.series / "tokenizer" / name, b'{"golden": true}')
    _write(
        sources.model / "model.safetensors",
        _fake_checkpoint({"config_json": json.dumps(config, ensure_ascii=False)}),
    )
    return sources


def _assemble_irodori(
    sources: IrodoriSources, out_dir: Path, model: str = IRODORI_DEFAULT_MODEL
) -> dict[str, Any]:
    return assemble_family([irodori_plan(sources, model)], out_dir, model)


def _v4_mirror_for(model: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """What a build of `model` through the driver needs for its cross-repository references.

    v4.1-small references v4-small's repository ({@link irodori_references}): this places
    v4-small's distribution where the driver reads referenced mirrors (`dist.DIST_ROOT`), built
    from the same synthetic series (so the shared roles are byte-identical), and returns the
    revision source — the placeholder (no Hub in the tests). Nothing for v4-small.
    """
    if model != IRODORI_V41_MODEL:
        return []
    import dist

    _assemble_irodori(
        _build_irodori_sources(tmp_path, model=IRODORI_V4_MODEL),
        tmp_path / "models" / irodori_repo_name(IRODORI_V4_MODEL),
        IRODORI_V4_MODEL,
    )
    monkeypatch.setattr(dist, "DIST_ROOT", tmp_path / "models")
    return [dist.ALLOW_PLACEHOLDER_FLAG]


@pytest.fixture
def irodori_assembled(tmp_path: Path) -> tuple[Path, dict]:
    sources = _build_irodori_sources(tmp_path)
    out_dir = tmp_path / "models" / irodori_repo_name(IRODORI_DEFAULT_MODEL)
    manifest = _assemble_irodori(sources, out_dir)
    return out_dir, manifest


def _irodori_model(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    return manifest["models"][IRODORI_DEFAULT_MODEL]


class TestIrodoriLayout:
    def test_it_places_every_declared_path_under_the_model_subtree(self, irodori_assembled) -> None:
        out_dir, _ = irodori_assembled
        expected = _in_subtree(IRODORI_DEFAULT_MODEL, _placed_paths())
        assert _present(out_dir) == sorted([*expected, MANIFEST_FILENAME])

    def test_it_never_carries_io_fixtures_or_tokenizer_goldens(self, irodori_assembled) -> None:
        """Only the 7 files needed to run go into the distribution (the goldens are assets for
        checks)."""
        out_dir, _ = irodori_assembled
        assert list(out_dir.rglob("io.*")) == []
        assert list(out_dir.rglob("golden.*")) == []
        assert list(out_dir.rglob("nfkc-diff.json")) == []

    def test_it_declares_the_nine_graphs_and_the_tokenizer(self, irodori_assembled) -> None:
        """9 = the 8 with the DiT split into `dit_context` / `dit` (the graph changed, so the
        expectation changed — ADR 0114)."""
        _, manifest = irodori_assembled
        model = _irodori_model(manifest)
        assert model["pipeline"] == "irodori/1"
        assert sorted(model["weights"]) == [
            "backbone",
            "caption_proj",
            "codec_decoder",
            "codec_encoder",
            "dit",
            "dit_context",
            "duration",
            "speaker",
            "text_proj",
        ]
        assert sorted(model["assets"]) == ["tokenizer"]

    def test_the_quants_are_the_five_seats_the_seat_table_spells(self, irodori_assembled) -> None:
        """There are 5 presets (3 storage series + the `-a8` execution knob + `i8+dit4` with only
        `dit` in i4). The default is `i8-a8`.

        `i8` / `i8-a8` **share the same i8 bytes** and differ only in `session` — the point of the
        preset table is that more presets do not mean more files in the distribution. `i8+dit4` is
        the exception: only `dit` points at the i4 series.
        """
        _, manifest = irodori_assembled
        model = _irodori_model(manifest)
        assert model["defaultQuant"] == "i8-a8"
        assert list(model["quants"]) == ["f32", "f16", "i8", "i8-a8", "i8+dit4"]
        for name, seat in IRODORI_QUANT_SEATS.items():
            assert model["quants"][name]["session"] == seat.session
            assert "gpuFeatures" not in model["quants"][name]
            # A complete mapping (hub's acceptance requirement) — each of the 8 roles points at
            # one dtype label.
            assert model["quants"][name]["weights"] == {
                role: seat.roles.get(role, seat.dtype) for role in model["weights"]
            }
        assert model["quants"]["i8"]["session"] == {}
        assert model["quants"]["i8-a8"]["session"] == {"linearCompute": "a8"}
        # MUST: `i8+dit4` declares no `linearCompute` (the w4a8 path is unmeasured for irodori —
        # the same MUST as in the preset table). If this loosens, a "fast but coarse" config ships
        # under the int4 preset's name.
        assert model["quants"]["i8+dit4"]["session"] == {}
        assert model["quants"]["i8+dit4"]["weights"]["dit"] == "i4"
        # The condition-side K/V projections stay i4 too (before the split they were stored as
        # i4 inside `dit` — ADR 0114).
        assert model["quants"]["i8+dit4"]["weights"]["dit_context"] == "i4"

    def test_every_quant_points_at_its_own_storage_series(self, irodori_assembled) -> None:
        """The mapping between presets and files (the assembly check verifies that the files of
        the compressed presets really are F16 / I8 / I4 storage, but this pins that **the
        declarations** do not cross series either — if only one side moved, the distribution would
        hold "f32 calling itself f16")."""
        out_dir, manifest = irodori_assembled
        model = _irodori_model(manifest)
        for name, seat in IRODORI_QUANT_SEATS.items():
            for role, label in model["quants"][name]["weights"].items():
                dtype = seat.roles.get(role, seat.dtype)
                parts = model["weights"][role][label]["container"]["parts"]
                # Split into parts, so the comparison is per list (container-v1 §8 — the preset's
                # spelling stays before the part number).
                expected = part_paths(f"model.{dtype}.krm", len(parts))
                for ref, tail in zip(parts, expected, strict=True):
                    assert ref["path"].endswith(tail), (role, name, ref["path"])
                    assert (out_dir / ref["path"]).is_file()

    def test_the_two_i8_seats_share_one_set_of_bytes(self, irodori_assembled) -> None:
        """`i8-a8` only adds one preset row (ADR 0050 step 2 — the distribution does not grow by a
        single byte)."""
        _, manifest = irodori_assembled
        model = _irodori_model(manifest)
        assert model["quants"]["i8"]["weights"] == model["quants"]["i8-a8"]["weights"]

    def test_the_dit4_seat_shares_the_i8_bytes_for_the_other_seven_roles(
        self, irodori_assembled
    ) -> None:
        """The only new files `i8+dit4` adds are the DiT's two (the other 7 roles share bytes with
        `i8`).

        Before the split it was `dit` alone — the DiT was split into `dit_context` / `dit`, so
        the set has two members (ADR 0114).
        """
        _, manifest = irodori_assembled
        model = _irodori_model(manifest)
        differing = {
            role
            for role, label in model["quants"]["i8+dit4"]["weights"].items()
            if label != model["quants"]["i8"]["weights"][role]
        }

        assert differing == {"dit", "dit_context"}

    def test_it_reassembles_over_a_previous_run(self, tmp_path: Path) -> None:
        sources = _build_irodori_sources(tmp_path)
        out_dir = tmp_path / "models" / irodori_repo_name(IRODORI_DEFAULT_MODEL)
        assert _assemble_irodori(sources, out_dir) == _assemble_irodori(sources, out_dir)
        assert verify_dist(out_dir)


class TestIrodoriPipelineConfig:
    """`pipelineConfig` matches the loader's schema (`src/irodori/config.ts`) down to the field
    names."""

    def test_it_declares_exactly_the_fields_the_loader_accepts(self, irodori_assembled) -> None:
        _, manifest = irodori_assembled
        assert tuple(_irodori_model(manifest)["pipelineConfig"]) == _IRODORI_CONFIG_KEYS

    def test_it_derives_the_model_specific_numbers_from_the_checkpoint(
        self, irodori_assembled
    ) -> None:
        """Baked-in values would show the real weights' numbers (latent 32 / speaker 768 /
        reference 120s)."""
        _, manifest = irodori_assembled
        config = _irodori_model(manifest)["pipelineConfig"]
        assert config["maxTextLen"] == _IRODORI_CONFIG["max_text_len"]
        assert config["maxCaptionLen"] == _IRODORI_CONFIG["max_caption_len"]
        assert config["latentDim"] == _IRODORI_CONFIG["latent_dim"]
        assert config["speakerPatchSize"] == _IRODORI_CONFIG["speaker_patch_size"]
        assert config["speakerDim"] == _IRODORI_CONFIG["speaker_dim"]
        assert config["textDim"] == _IRODORI_CONFIG["text_dim"]
        assert config["captionDim"] == _IRODORI_CONFIG["caption_dim"]
        assert config["timestepEmbedDim"] == _IRODORI_CONFIG["timestep_embed_dim"]
        # The limit of the reference latent after patching + one mean token / 30s × 25Hz ÷ latent
        # patch.
        assert config["speakerRows"] == _IRODORI_SPEAKER_ROWS
        assert config["ditSymMax"] == _IRODORI_DIT_SYM_MAX

    def test_it_derives_the_codec_numbers_from_the_codec_metadata(self, irodori_assembled) -> None:
        """Baked-in values would show the real numbers (48kHz / hop 1920). Only the halo is
        hard-coded."""
        _, manifest = irodori_assembled
        config = _irodori_model(manifest)["pipelineConfig"]
        assert config["sampleRate"] == _IRODORI_CODEC_METADATA["kwargs"]["sample_rate"]
        # hop_length = prod(encoder_rates) (as spelled in `DACVAE.__init__`).
        assert config["hopLength"] == _IRODORI_HOP_LENGTH
        assert config["codecHaloFrames"] == IRODORI_CODEC_HALO_FRAMES
        # The loader checks the consistency of the three (sampleRate == frameRate × hopLength) at
        # parse time.
        assert config["sampleRate"] == config["frameRate"] * config["hopLength"]

    def test_it_refuses_a_codec_whose_frame_rate_does_not_match(self, tmp_path: Path) -> None:
        """Do not let the two conversions, seconds → frames and seconds → samples → frames, move
        independently."""
        sources = _build_irodori_sources(
            tmp_path, codec_metadata={"kwargs": {"sample_rate": 12_000, "encoder_rates": [500]}}
        )
        with pytest.raises(DistError, match="hop_length"):
            irodori_plan(sources)

    def test_it_refuses_a_missing_codec_metadata(self, tmp_path: Path) -> None:
        sources = _build_irodori_sources(tmp_path, codec_metadata=None)
        with pytest.raises(DistError, match=r"metadata\.json"):
            irodori_plan(sources)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"sample_rate": 0, "encoder_rates": [2, 4, 60]},
            {"sample_rate": 12_000, "encoder_rates": []},
            {"sample_rate": 12_000, "encoder_rates": [2, "4", 60]},
        ],
    )
    def test_it_refuses_codec_numbers_that_are_not_positive_integers(
        self, tmp_path: Path, kwargs: Mapping[str, Any]
    ) -> None:
        sources = _build_irodori_sources(tmp_path, codec_metadata={"kwargs": kwargs})
        with pytest.raises(DistError):
            irodori_plan(sources)

    def test_it_carries_the_sampler_defaults_the_upstream_declares(self, irodori_assembled) -> None:
        _, manifest = irodori_assembled
        config = _irodori_model(manifest)["pipelineConfig"]
        for key, value in IRODORI_SAMPLING_DEFAULTS.items():
            assert config[key] == value
        # ADR 0047 decision 1 — the loader rejects any other two values at parse time.
        assert config["speakerUncondMode"] == "mask"
        assert config["cfgGuidanceMode"] == "independent"

    def test_it_refuses_a_checkpoint_without_the_config_metadata(self, tmp_path: Path) -> None:
        sources = _build_irodori_sources(tmp_path)
        _write(sources.model / "model.safetensors", _fake_checkpoint())
        with pytest.raises(DistError, match="config_json"):
            irodori_plan(sources)

    def test_it_refuses_a_checkpoint_whose_config_is_not_json(self, tmp_path: Path) -> None:
        sources = _build_irodori_sources(tmp_path)
        _write(
            sources.model / "model.safetensors",
            _fake_checkpoint({"config_json": "{"}),
        )
        with pytest.raises(DistError, match="cannot be parsed as JSON"):
            irodori_plan(sources)

    @pytest.mark.parametrize("value", [0, "32", True, None])
    def test_it_refuses_a_dimension_that_is_not_a_positive_integer(
        self, tmp_path: Path, value: Any
    ) -> None:
        sources = _build_irodori_sources(tmp_path, config={**_IRODORI_CONFIG, "latent_dim": value})
        with pytest.raises(DistError, match="latent_dim"):
            irodori_plan(sources)

    @pytest.mark.parametrize(
        "value", [float("nan"), float("inf"), float("-inf"), -1.0, 0.0, "8.0", True, None]
    )
    def test_it_refuses_a_reference_length_that_is_not_a_finite_positive_number(
        self, tmp_path: Path, value: Any
    ) -> None:
        """NaN compares False with everything and slips past `<= 0` — do not carry it into the
        seconds → frames conversion downstream."""
        sources = _build_irodori_sources(
            tmp_path, config={**_IRODORI_CONFIG, "ref_max_seconds": value}
        )
        with pytest.raises(DistError, match="ref_max_seconds"):
            irodori_plan(sources)

    def test_it_refuses_a_latent_patch_size_the_loader_schema_cannot_express(
        self, tmp_path: Path
    ) -> None:
        """`latentDim` is both the width of x_t and the frame width of the reference latent (only
        compatible at 1)."""
        sources = _build_irodori_sources(
            tmp_path, config={**_IRODORI_CONFIG, "latent_patch_size": 2}
        )
        with pytest.raises(DistError, match="latent_patch_size"):
            irodori_plan(sources)


class TestIrodoriGraphGate:
    """The assembly check — refuses, **before** placement, combinations that would pass with
    matching shapes despite a mismatch."""

    def _sources(self, tmp_path: Path, role: str, spec: _Spec) -> IrodoriSources:
        return _build_irodori_sources(tmp_path, specs={role: spec})

    @staticmethod
    def _bent(role: str, mutate) -> _Spec:
        """A spec with the real shape bent in exactly one place (only the input order / shapes
        move)."""
        inputs, outputs, symbol = _irodori_specs()[role]
        bent = [(name, list(shape)) for name, shape in inputs]
        mutate(bent)
        return bent, outputs, symbol

    def test_it_refuses_a_caption_projector_with_a_single_output(self, tmp_path: Path) -> None:
        """Without the second output (the sequence after `caption_norm`), `caption_vec` becomes a
        different vector."""
        sources = self._sources(
            tmp_path,
            "caption_proj",
            ([("hidden", [1, "T", _IRODORI_HIDDEN])], 1, "T"),
        )
        with pytest.raises(DistError, match="has 1 outputs"):
            irodori_plan(sources)

    def test_it_refuses_a_dit_that_lost_an_input(self, tmp_path: Path) -> None:
        sources = self._sources(tmp_path, "dit", self._bent("dit", lambda ins: ins.__delitem__(5)))
        with pytest.raises(DistError, match="graph inputs of"):
            irodori_plan(sources)

    def test_it_refuses_a_dit_whose_inputs_are_reordered(self, tmp_path: Path) -> None:
        def swap(ins: list[tuple[str, list[Any]]]) -> None:
            ins[3], ins[5] = ins[5], ins[3]

        sources = self._sources(tmp_path, "dit", self._bent("dit", swap))
        with pytest.raises(DistError, match="graph inputs of"):
            irodori_plan(sources)

    def test_it_refuses_a_graph_that_declares_another_conditioning_length(
        self, tmp_path: Path
    ) -> None:
        """Right padding passes even with a shifted declared length of the condition states (it
        silently reads the conditions at other positions).

        The condition states are inputs of `dit_context` (moved there from `dit` by ADR 0114), so
        that is the graph bent here.
        """

        def stretch(ins: list[tuple[str, list[Any]]]) -> None:
            ins[0][1][1] = _IRODORI_CONFIG["max_text_len"] + 1

        sources = self._sources(tmp_path, "dit_context", self._bent("dit_context", stretch))
        with pytest.raises(DistError, match="maxTextLen"):
            irodori_plan(sources)

    def test_it_refuses_a_speaker_encoder_with_another_patch_width(self, tmp_path: Path) -> None:
        sources = self._sources(tmp_path, "speaker", ([("latent", [1, "S", 999])], 1, "S"))
        with pytest.raises(DistError, match="speakerPatchSize"):
            irodori_plan(sources)

    def test_it_refuses_a_mask_whose_segments_do_not_add_up(self, tmp_path: Path) -> None:
        """If the segment sum shifts, only the mask's segment split silently points elsewhere."""

        def bend(ins: list[tuple[str, list[Any]]]) -> None:
            ins[2][1][3] = f"S+{_IRODORI_MASK_TOTAL + 1}"

        sources = self._sources(tmp_path, "dit", self._bent("dit", bend))
        with pytest.raises(DistError, match="mask"):
            irodori_plan(sources)

    def test_it_does_not_bake_the_dit_block_count(self, tmp_path: Path) -> None:
        """The number of boundary tensors is derived from the outputs of `dit_context` — a pair
        with another block count passes too (ADR 0114)."""
        inputs, _outputs, symbol = _irodori_specs()["dit"]
        sources = _build_irodori_sources(
            tmp_path,
            specs={
                "dit": ([*inputs[:3], *_context_kv(3)], 1, symbol),
                "dit_context": (_irodori_specs()["dit_context"][0], _context_kv(3), ""),
            },
        )

        assert irodori_plan(sources).weights["dit_context"]

    def test_it_refuses_context_outputs_out_of_the_boundary_order(self, tmp_path: Path) -> None:
        """The runtime binds the resident tensors by name — with K and V swapped it reads other
        values."""
        inputs, outputs, symbol = _irodori_specs()["dit_context"]
        assert isinstance(outputs, list)
        swapped = [outputs[1], outputs[0], *outputs[2:]]
        sources = self._sources(tmp_path, "dit_context", (inputs, swapped, symbol))
        with pytest.raises(DistError, match="boundary names"):
            irodori_plan(sources)

    def test_it_refuses_a_context_graph_with_an_unpaired_output(self, tmp_path: Path) -> None:
        inputs, outputs, symbol = _irodori_specs()["dit_context"]
        assert isinstance(outputs, list)
        sources = self._sources(tmp_path, "dit_context", (inputs, outputs[:-1], symbol))
        with pytest.raises(DistError, match="are not K / V pairs"):
            irodori_plan(sources)

    def test_it_refuses_a_context_graph_whose_inputs_are_reordered(self, tmp_path: Path) -> None:
        def swap(ins: list[tuple[str, list[Any]]]) -> None:
            ins[0], ins[2] = ins[2], ins[0]

        sources = self._sources(tmp_path, "dit_context", self._bent("dit_context", swap))
        with pytest.raises(DistError, match="graph inputs of"):
            irodori_plan(sources)

    def test_it_refuses_a_dit_built_for_another_block_count(self, tmp_path: Path) -> None:
        """The inputs of `dit` must match the order derived from the outputs of
        `dit_context`."""
        inputs, _outputs, symbol = _irodori_specs()["dit"]
        sources = self._sources(tmp_path, "dit", ([*inputs[:3], *_context_kv(3)], 1, symbol))
        with pytest.raises(DistError, match="graph inputs of"):
            irodori_plan(sources)

    def test_it_refuses_a_boundary_whose_shape_differs_between_the_two_graphs(
        self, tmp_path: Path
    ) -> None:
        def widen(ins: list[tuple[str, list[Any]]]) -> None:
            ins[3][1][2] += 1

        sources = self._sources(tmp_path, "dit", self._bent("dit", widen))
        with pytest.raises(DistError, match="disagrees"):
            irodori_plan(sources)

    def test_it_refuses_a_context_length_that_misses_the_mask_segments(
        self, tmp_path: Path
    ) -> None:
        """Even with matching shapes on both sides, a length different from the sum of the mask's 3
        condition segments shifts the segment split."""
        longer = [1, _IRODORI_MASK_TOTAL + 1, *_IRODORI_CONTEXT_SHAPE[2:]]
        context_inputs, outputs, _symbol = _irodori_specs()["dit_context"]
        dit_inputs, _dit_outputs, symbol = _irodori_specs()["dit"]
        assert isinstance(outputs, list)
        sources = _build_irodori_sources(
            tmp_path,
            specs={
                "dit_context": (context_inputs, [(name, longer) for name, _ in outputs], ""),
                "dit": (
                    [*dit_inputs[:3], *((name, list(longer)) for name, _ in dit_inputs[3:])],
                    1,
                    symbol,
                ),
            },
        )
        with pytest.raises(DistError, match="3 condition segments"):
            irodori_plan(sources)

    def test_a_series_holding_only_one_side_of_the_dit_pair_is_refused(self) -> None:
        """`dit` takes the outputs of `dit_context` as inputs — a series with only one side cannot
        run."""
        from irodori.distribution import _assert_irodori_dit_boundary

        with pytest.raises(DistError, match="One side of the DiT pair"):
            _assert_irodori_dit_boundary(
                {"dit": {"inputs": [], "outputs": ["v"]}}, {"dit": Path("dit")}, {}
            )

    def test_it_refuses_a_codec_decoder_for_another_latent_width(self, tmp_path: Path) -> None:
        """Mixing in a DACVAE of another dimension gives a different voice with matching
        shapes."""
        sources = self._sources(tmp_path, "codec_decoder", ([("latent", [1, "S", 999])], 1, "S"))
        with pytest.raises(DistError, match="latentDim"):
            irodori_plan(sources)

    def test_it_refuses_a_codec_encoder_with_another_hop(self, tmp_path: Path) -> None:
        """If the input width = hopLength shifts, only the waveform's frame split silently moves to
        another grid."""
        sources = self._sources(tmp_path, "codec_encoder", ([("wav", [1, "T", 999])], 1, "T"))
        with pytest.raises(DistError, match="hopLength"):
            irodori_plan(sources)

    def test_it_refuses_a_codec_decoder_that_lost_its_input_name(self, tmp_path: Path) -> None:
        sources = self._sources(tmp_path, "codec_decoder", ([("z", [1, "S", 8])], 1, "S"))
        with pytest.raises(DistError, match="graph inputs of"):
            irodori_plan(sources)

    def test_it_refuses_a_codec_decoder_with_two_outputs(self, tmp_path: Path) -> None:
        """Evidence that no separate target for checks (with intermediate values) slipped in."""
        sources = self._sources(tmp_path, "codec_decoder", ([("latent", [1, "S", 8])], 2, "S"))
        with pytest.raises(DistError, match="has 2 outputs"):
            irodori_plan(sources)

    def test_it_refuses_a_component_that_is_not_a_container(self, tmp_path: Path) -> None:
        sources = _build_irodori_sources(tmp_path)
        replace_component(sources.series / "dit" / "model.krm", b"not-a-container")
        with pytest.raises(DistError, match="cannot be read as a container"):
            irodori_plan(sources)


class TestIrodoriStorageSeries:
    """**Set equality** of series × storage dtype — the only detector of a mixed-up series root.

    MUST: a numeric check (the E2E tolerance) cannot detect it in principle (the detection limit
    of ADR 0027 / 0029 — the f32 and f16 series measure within the same order of magnitude, so
    each passes the other's threshold). Only presence and absence checks from both sides make any
    asset in any wrong preset fail.
    """

    def test_it_refuses_an_f16_series_asset_in_the_f32_seat(self, tmp_path: Path) -> None:
        """A compressed container also holds the ineligible weights as F32, so **a presence check
        alone lets it through**."""
        sources = _build_irodori_sources(tmp_path)
        replace_component(
            sources.series / "dit" / "model.krm",
            _irodori_input("f16", "dit"),
        )

        with pytest.raises(DistError, match="has f16"):
            irodori_plan(sources)

    def test_it_refuses_an_f32_series_asset_in_the_f16_seat(self, tmp_path: Path) -> None:
        """The opposite direction (forgotten rounding = plain f32 where `--dtype f16` was meant)
        fails on the absence of F16."""
        sources = _build_irodori_sources(tmp_path)
        replace_component(
            sources.series_by_dtype["f16"] / "dit" / "model.krm",
            _irodori_input("f32", "dit"),
        )

        with pytest.raises(DistError, match="has no f16"):
            irodori_plan(sources)

    def test_it_refuses_an_i8_series_asset_in_the_f32_seat(self, tmp_path: Path) -> None:
        """An i8 asset holds F32 (bias / norm / per-channel scale), so **the presence check is
        true**.

        MUST: adding the i8 series while the forbidden table still said "the f32 preset forbids
        only F16" would let this mix-up through (the same mechanism as the hole closed for f16 in
        step 1 — the forbidden storage is kept as a set).
        """
        sources = _build_irodori_sources(tmp_path)
        replace_component(
            sources.series / "dit" / "model.krm",
            _irodori_input("i8", "dit"),
        )

        with pytest.raises(DistError, match="has i8"):
            irodori_plan(sources)

    def test_it_refuses_an_f32_series_asset_in_the_i8_seat(self, tmp_path: Path) -> None:
        """The opposite direction (forgotten rounding = plain f32 where `--dtype i8` was meant)
        fails on the absence of I8."""
        sources = _build_irodori_sources(tmp_path)
        replace_component(
            sources.series_by_dtype["i8"] / "dit" / "model.krm",
            _irodori_input("f32", "dit"),
        )

        with pytest.raises(DistError, match="has no i8"):
            irodori_plan(sources)

    def test_it_refuses_an_i8_series_asset_in_the_f16_seat(self, tmp_path: Path) -> None:
        """A mix-up between compressed series fails on **the absence of the required dtype** (no
        forbidden table needed)."""
        sources = _build_irodori_sources(tmp_path)
        replace_component(
            sources.series_by_dtype["f16"] / "dit" / "model.krm",
            _irodori_input("i8", "dit"),
        )

        with pytest.raises(DistError, match="has no f16"):
            irodori_plan(sources)

    def test_it_refuses_an_f16_series_asset_in_the_i8_seat(self, tmp_path: Path) -> None:
        sources = _build_irodori_sources(tmp_path)
        replace_component(
            sources.series_by_dtype["i8"] / "dit" / "model.krm",
            _irodori_input("f16", "dit"),
        )

        with pytest.raises(DistError, match="has no i8"):
            irodori_plan(sources)

    def test_it_refuses_an_i4_series_asset_in_the_i8_seat(self, tmp_path: Path) -> None:
        """MUST: the i8 preset is closed off by **the absence of I4** (the requirement check alone
        cannot tell them apart).

        What is copied is **the real i4 series** (I4 inside the blocks except adaLN, I8 for adaLN
        and outside the blocks) — that container satisfies "contains I8", so without the forbidden
        table it could be plugged into the i8 preset. The damage would show in the default preset
        `i8-a8` (the predicate of the declared `linearCompute: "a8"` also accepts resident i4, so
        ADR 0076's w4a8 path would silently run under a preset that claims int8).
        """
        sources = _build_irodori_sources(tmp_path)
        replace_component(
            sources.series_by_dtype["i8"] / "dit" / "model.krm",
            _irodori_input("i4", "dit"),
        )

        with pytest.raises(DistError, match="has i4"):
            irodori_plan(sources)

    def test_it_refuses_an_i8_series_asset_in_the_i4_seat(self, tmp_path: Path) -> None:
        """The opposite direction (an i8 series where `--dtype i4` was meant) fails on the absence
        of I4."""
        sources = _build_irodori_sources(tmp_path)
        replace_component(
            sources.series_by_dtype["i4"] / "dit" / "model.krm",
            _irodori_input("i8", "dit"),
        )

        with pytest.raises(DistError, match="has no i4"):
            irodori_plan(sources)

    def test_it_refuses_an_i4_series_asset_in_the_f32_seat(self, tmp_path: Path) -> None:
        """An i4 asset also holds F32 (bias / norm / group scale), so **the presence check is
        true**."""
        sources = _build_irodori_sources(tmp_path)
        replace_component(
            sources.series / "dit" / "model.krm",
            _irodori_input("i4", "dit"),
        )

        with pytest.raises(DistError, match=r"has (?:i8 / )?i4"):
            irodori_plan(sources)

    def test_it_refuses_a_codec_series_mixup_too(self, tmp_path: Path) -> None:
        """The codec is a separate series (`dacvae-32dim{,-f16,-i8}`) — the same check applies to
        all 8 roles."""
        sources = _build_irodori_sources(tmp_path)
        replace_component(
            sources.codec_series_by_dtype["f16"] / "decoder" / "model.krm",
            _irodori_input("f32", "codec_decoder"),
        )

        with pytest.raises(DistError, match="has no f16"):
            irodori_plan(sources)

    def test_every_graph_role_carries_every_seat_its_series_declares(self) -> None:
        """The requirement table / forbidden table / declarations point at the same entries (9
        roles × f32/f16/i8 + i4 for the DiT's two).

        If only one side gained an entry, the new one would silently ship unchecked.
        """
        expected = {
            f"{role}_{dtype}" for dtype, roles in IRODORI_DTYPE_ROLES.items() for role in roles
        }

        assert set(IRODORI_STORAGE_REQUIREMENTS) == expected
        # I4 is forbidden in the i8 preset for the DiT's two graphs, which have an i4 series
        # (before the split, `dit` alone — ADR 0114).
        assert set(IRODORI_STORAGE_FORBIDDEN) == {
            *(f"{role}_f32" for role in IRODORI_GRAPH_ROLES),
            "dit_i8",
            "dit_context_i8",
        }
        # The forbidden set covers **every compressed series** (missing one would let that asset
        # through the f32 preset).
        assert set(IRODORI_STORAGE_FORBIDDEN[f"{IRODORI_GRAPH_ROLES[0]}_f32"]) == {
            "f16",
            "i8",
            "i4",
        }
        # The i8 preset is closed off by the absence of I4 (the i4 series contains I8 too, so the
        # requirement check alone does not close it).
        assert set(IRODORI_STORAGE_FORBIDDEN["dit_i8"]) == {"i4"}
        assert set(IRODORI_STORAGE_FORBIDDEN["dit_context_i8"]) == {"i4"}
        assert {
            files.file for labels in IRODORI_WEIGHTS.values() for files in labels.values()
        } == expected

    def test_the_dit_pair_moves_together_through_every_table(self) -> None:
        """The DiT's two graphs (ADR 0114) appear **as a pair** in the i4 series, the I4 ban of
        the i8 preset and the `i8+dit4` exception.

        With only one side, the 72 condition-side projections of `dit_context` would stay i8 in
        the i4 preset (the preset's numbers move), or an i4 asset would be plugged into the i8
        preset.
        """
        pair = {"dit", "dit_context"}

        assert set(IRODORI_DTYPE_ROLES["i4"]) == pair
        assert {role for role, dtype in IRODORI_QUANT_SEATS["i8+dit4"].roles.items()} == pair
        assert set(IRODORI_QUANT_SEATS["i8+dit4"].roles.values()) == {"i4"}
        assert {f"{role}_i8" for role in pair} <= set(IRODORI_STORAGE_FORBIDDEN)
        assert IRODORI_SERIES_DIRS["dit_context"] == "dit-context"

    def test_every_static_dim_names_an_input_of_its_graph(self) -> None:
        """Every (role, input) of the static dimension table exists among that role's inputs
        (consistency between the tables).

        The three condition states are inputs of `dit_context` (moved there from `dit` by ADR
        0114) — if only one table moved, the check would look up an axis of an input that does not
        exist and raise KeyError, or silently stop checking.
        """
        from irodori.distribution import (
            IRODORI_CONTEXT_INPUTS,
            IRODORI_DIT_HEAD_INPUTS,
            IRODORI_GRAPH_SHAPES,
            IRODORI_STATIC_DIMS,
        )

        inputs = {role: names for role, (names, _outputs) in IRODORI_GRAPH_SHAPES.items()}
        inputs["dit_context"] = IRODORI_CONTEXT_INPUTS
        inputs["dit"] = IRODORI_DIT_HEAD_INPUTS

        assert set(inputs) == set(IRODORI_GRAPH_ROLES)
        for role, name, _axis, _field in IRODORI_STATIC_DIMS:
            assert name in inputs[role], (role, name)
        assert {
            name for role, name, _axis, _field in IRODORI_STATIC_DIMS if role == "dit_context"
        } == set(IRODORI_CONTEXT_INPUTS)

    def test_every_seat_starts_from_a_storage_dtype_of_the_vocabulary(self) -> None:
        """Preset names follow the grammar of ADR 0074 — `<storage>` is the vocabulary of the asset
        header (decision 2). `w8` / `w4` are retired."""
        for name, seat in IRODORI_QUANT_SEATS.items():
            assert name.split("+")[0].split("-")[0] == seat.dtype, name

    def test_it_needs_no_abbreviation_legend(self) -> None:
        """The token `dit` of `i8+dit4` is the weights name itself (not an abbreviation — ADR 0074
        decision 4).

        A non-empty legend would put a line "`dit` means `dit`" on the card.
        """
        assert IRODORI_QUANT_ABBREVIATIONS == {}

    def test_every_seat_carries_a_label_and_a_description_within_the_limits(self) -> None:
        for name, seat in IRODORI_QUANT_SEATS.items():
            assert seat.label and seat.description, name
            assert_quant_presentation(
                f"irodori.quants.{name}",
                {"label": seat.label, "description": seat.description},
            )

    def test_every_quant_seat_names_a_storage_series_that_exists(self) -> None:
        """Every dtype the preset table (`IRODORI_QUANT_SEATS`) points at is one baked as a series.

        This is the only place that spells the mapping between preset names (`i8`) and series
        roots (`-i8`) — split into two places, one moving alone would produce "a preset pointing at
        a series that does not exist" or "a series nobody points at".
        MUST: the per-role exceptions (`dit` of `i8+dit4`) go through the same check.
        """
        named = {
            seat.roles.get(role, seat.dtype)
            for seat in IRODORI_QUANT_SEATS.values()
            for role in IRODORI_GRAPH_ROLES
        }

        assert named == set(IRODORI_WEIGHT_DTYPES)
        for seat in IRODORI_QUANT_SEATS.values():
            for role, dtype in seat.roles.items():
                assert role in IRODORI_DTYPE_ROLES[dtype], (role, dtype)

    def test_the_series_roots_match_the_exporter_spelling(self, tmp_path: Path) -> None:
        """The spelling of the series roots matches the writers (`export_*.default_out_root`) to
        the character."""
        import irodori.dacvae.export
        import irodori.export

        sources = irodori_sources(tmp_path)
        for dtype in IRODORI_WEIGHT_DTYPES:
            assert (
                sources.series_by_dtype[dtype].name
                == irodori.export.default_out_root(Path(IRODORI_DEFAULT_MODEL), dtype).name
            )
        for dtype in sources.codec_series_by_dtype:
            assert (
                sources.codec_series_by_dtype[dtype].name
                == irodori.dacvae.export.default_out_root(Path(IRODORI_CODEC_NAME), dtype).name
            )
        # The codec has no i4 series (i4 in `IRODORI_DTYPE_ROLES` is only the DiT's two).
        assert set(sources.codec_series_by_dtype) == {"f32", "f16", "i8"}


class TestIrodoriCalibProvenance:
    """Checking the rounding conditions of the i4 series — the place that checks mechanically the
    one fact **the assets cannot reveal**.

    MUST: neither the calibration method nor the budget changes the storage form by a single
    byte, so the required-dtype check, the graph check and `verify_dist` all pass. If the output
    of `--no-calib` or `--calib-steps 1` slipped into the distribution, the only symptom would be
    degraded audio.
    """

    def test_a_calibrated_series_passes(self, tmp_path: Path) -> None:
        sources = _build_irodori_sources(tmp_path)

        assert irodori_plan(sources).quants["i8+dit4"]["weights"]["dit"] == "i4"

    def test_a_missing_record_is_refused(self, tmp_path: Path) -> None:
        """A missing record means "an old export" — the message asks for a re-export by name."""
        sources = _build_irodori_sources(tmp_path, calib_provenance=None)

        with pytest.raises(DistError, match="has no record of its calibration conditions"):
            irodori_plan(sources)

    def test_a_missing_record_on_the_context_side_is_refused(self, tmp_path: Path) -> None:
        """Both DiT graphs need the record (checking only `dit` would let an old export of
        `dit_context` through)."""
        sources = _build_irodori_sources(tmp_path)
        (
            sources.series_by_dtype["i4"]
            / IRODORI_SERIES_DIRS["dit_context"]
            / CALIB_PROVENANCE_FILE
        ).unlink()

        with pytest.raises(DistError, match="has no record of its calibration conditions"):
            irodori_plan(sources)

    def test_records_that_differ_between_the_two_dit_graphs_are_refused(
        self, tmp_path: Path
    ) -> None:
        """The two are written from the same rounding by one export — different records mean series
        from separate runs.

        Each record meets the distribution conditions on its own (a budget at or above the floor),
        so judging them one at a time would let this through.
        """
        sources = _build_irodori_sources(tmp_path)
        _write(
            sources.series_by_dtype["i4"]
            / IRODORI_SERIES_DIRS["dit_context"]
            / CALIB_PROVENANCE_FILE,
            json.dumps({**_IRODORI_CALIB_PROVENANCE, "steps": 41}).encode("utf-8"),
        )

        with pytest.raises(DistError, match="differ per role"):
            irodori_plan(sources)

    def test_an_uncalibrated_series_is_refused(self, tmp_path: Path) -> None:
        """`--no-calib` is recorded as `rtn` — the assembly refuses it by name."""
        sources = _build_irodori_sources(
            tmp_path,
            calib_provenance={**_IRODORI_CALIB_PROVENANCE, "method": "rtn", "cases": 0, "steps": 0},
        )

        with pytest.raises(
            DistError, match="not built with a rounding method allowed for distribution"
        ):
            irodori_plan(sources)

    def test_an_unparsable_record_is_refused(self, tmp_path: Path) -> None:
        sources = _build_irodori_sources(tmp_path)
        _write(
            sources.series_by_dtype["i4"] / IRODORI_SERIES_DIRS["dit"] / CALIB_PROVENANCE_FILE,
            b"{ not json",
        )

        with pytest.raises(DistError, match="Cannot parse the record"):
            irodori_plan(sources)

    def test_a_smoke_budget_is_refused(self, tmp_path: Path) -> None:
        """`--calib-steps 1` leaves `method` at `gptq` — it passes unless the budget fields are
        checked too.

        The storage form does not change by a single byte (the grid stays RTN i4 g32), so both the
        header check and `verify_dist` read this asset as "a correct i8+dit4 preset".
        """
        sources = _build_irodori_sources(
            tmp_path, calib_provenance={**_IRODORI_CALIB_PROVENANCE, "steps": 1}
        )

        with pytest.raises(
            DistError, match="calibration budget 'steps' is below the distribution floor"
        ):
            irodori_plan(sources)

    def test_a_smaller_corpus_is_refused(self, tmp_path: Path) -> None:
        """A series baked from a reduced corpus fails at the same place (4 cases → 12 was the
        listening ruling of 2026-08-23)."""
        sources = _build_irodori_sources(
            tmp_path, calib_provenance={**_IRODORI_CALIB_PROVENANCE, "cases": 4}
        )

        with pytest.raises(
            DistError, match="calibration budget 'cases' is below the distribution floor"
        ):
            irodori_plan(sources)

    def test_a_record_written_before_the_budget_fields_is_still_accepted(
        self, tmp_path: Path
    ) -> None:
        """Backward compatibility MUST: the **absence** of the fields is accepted (rebuilding the
        record = a re-export costing the rounding time)."""
        legacy = {
            key: value
            for key, value in _IRODORI_CALIB_PROVENANCE.items()
            if key not in {"cases", "steps"}
        }
        sources = _build_irodori_sources(tmp_path, calib_provenance=legacy)

        assert irodori_plan(sources).quants["i8+dit4"]["weights"]["dit"] == "i4"

    def test_the_floor_is_the_condition_the_export_defaults_to(self) -> None:
        """The floor comes from the source of truth, not a copy (the corpus size and the full
        reference loop)."""
        from irodori.calib_cases import CALIB_CASES
        from irodori.pipeline_ref import NUM_STEPS

        assert irodori_calib_floor() == {"cases": len(CALIB_CASES), "steps": NUM_STEPS}
        # The shipped series (12 cases × 40 steps) is baked to meet exactly this floor.
        assert _IRODORI_CALIB_PROVENANCE["cases"] == len(CALIB_CASES)
        assert _IRODORI_CALIB_PROVENANCE["steps"] == NUM_STEPS


class TestIrodoriModelCard:
    def _run(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        """Run the CLI once on fake assets only (the checkpoint location moves to tmp too)."""
        from irodori import distribution

        sources = _build_irodori_sources(tmp_path)
        # `INPUTS_ROOT` (the checkpoint outside the series) is looked up by the recipe — so this
        # replaces the recipe's binding, not the driver's `dist` (`DIST_ROOT` is the opposite, on
        # the driver).
        monkeypatch.setattr(distribution, "INPUTS_ROOT", tmp_path / "inputs")
        out_dir = tmp_path / "dist"
        main(
            ["--pipeline", "irodori", "--series", str(sources.series.parent), "--out", str(out_dir)]
        )
        return out_dir

    def test_it_describes_the_text_to_audio_distribution(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        card = (self._run(tmp_path, monkeypatch) / MODEL_CARD_FILENAME).read_text(encoding="utf-8")
        assert "pipeline_tag: text-to-speech" in card
        assert "license: mit" in card
        # What the default quant (`i8-a8`) downloads is the int8 series, so `quantized` is
        # declared (the same as sbv2 / anima. The old "stays f32, not declared" was a stale
        # premise from before the i8 series was bundled).
        assert "base_model_relation: quantized" in card
        assert "Text in, waveform out" in card
        # The bundled codec appears both in the attribution and in `base_model` (it is
        # redistributed).
        assert "Semantic-DACVAE-Japanese-32dim" in card
        # The reference speaker can be passed as "audio" or as "latent" (voice cloning is wired
        # up).
        assert "Voice cloning is wired up both ways" in card
        # The sample rate must match the distribution (there is no resampling = a mismatch fails
        # loudly). This number comes down from the codec's metadata through the manifest.
        rate = _IRODORI_CODEC_METADATA["kwargs"]["sample_rate"]
        assert f"distribution's own {rate} Hz" in card
        assert "there is no resampler" in card
        # The untiled encoder can fail on a long reference (a by-design constraint filed in
        # limitations).
        assert "`codec_encoder` is not tiled" in card
        # The repo is spelled from the pipeline's declaration, not from the output directory
        # (`tmp/dist`).
        assert '  repo: "hdae/karume-irodori-v4-small",' in card
        # The revision sits commented out inside the object ref (uncommenting it is enough to
        # pin).
        assert '  // revision: "<full commit sha>",' in card
        # Usage takes the form "uncomment to go one step further" (ruling of 2026-08-12): both
        # forms of voice cloning (audio / latent) and the optional knobs are listed commented out,
        # and the selectable values are derived mechanically from the manifest (more presets
        # update both the list and the default — here 5 presets, default `i8-a8`).
        assert '// speaker: { audio: decodeWav(await Deno.readFile("reference.wav")) },' in card
        assert "// speaker: { latent: savedLatent }," in card
        assert '// quant: "i8-a8", // default — available: f16 / f32 / i8 / i8+dit4 / i8-a8' in card
        assert "// durationSeconds: 5," in card

    def test_it_derives_the_shape_section_from_the_manifest(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        out_dir = self._run(tmp_path, monkeypatch)
        card = (out_dir / MODEL_CARD_FILENAME).read_text(encoding="utf-8")
        assert f"up to {_IRODORI_CONFIG['max_text_len']} tokens" in card
        assert f"up to {_IRODORI_DIT_SYM_MAX} frames" in card
        # The card is rendered from a distribution that **passed verification** (no description
        # appears while the table and the files disagree).
        assert verify_dist(out_dir)


class TestIrodoriLegalText:
    """The legal texts at the root of the distribution repository (ADR 0092 decision 7) — the
    same two files in both the v4 and v4.1 repositories.

    There are two upstream lines: MIT (the Irodori body, the text backbone and the codec's direct
    upstream) and Apache 2.0 of the codec's original weights `facebook/dacvae-watermarked`.
    `LICENSE.md` carries both texts, and `NOTICE.md` attributes both by name.
    """

    @staticmethod
    def _assemble(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, model: str) -> Path:
        from irodori import distribution

        sources = _build_irodori_sources(tmp_path, model=model)
        monkeypatch.setattr(distribution, "INPUTS_ROOT", tmp_path / "inputs")
        out_dir = tmp_path / "dist"
        main(
            [
                "--pipeline",
                "irodori",
                "--model",
                model,
                "--series",
                str(sources.series.parent),
                "--out",
                str(out_dir),
                *_v4_mirror_for(model, tmp_path, monkeypatch),
            ]
        )
        return out_dir

    @pytest.mark.parametrize("model", sorted(IRODORI_UPSTREAMS))
    def test_it_ships_the_license_text_byte_identical(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, model: str
    ) -> None:
        """The assembly path does not move `LICENSE.md` by a single byte (the two tests below check
        that the texts are verbatim).

        Reformatting or newline conversion anywhere in the assembly path would move a byte, but the
        prose would still be valid, so the other checks would pass (read as bytes, so that the
        newline conversion of `read_text` does not fold CRLF).
        """
        out_dir = self._assemble(tmp_path, monkeypatch, model)
        license_text = (out_dir / "LICENSE.md").read_bytes().decode("utf-8")

        assert license_text == irodori_license_markdown()

    def test_the_license_opens_with_the_mit_text_for_every_component(self) -> None:
        """MIT § "include the copyright notice and the permission notice" — everything except the
        insertion point is the template text itself."""
        license_text = PIPELINE.root_files["LICENSE.md"]

        assert license_text.startswith(mit_license(IRODORI_COPYRIGHTS))
        assert license_text.count("Permission is hereby granted, free of charge") == 1

    def test_the_license_carries_the_apache_text_verbatim_for_the_codec(self) -> None:
        """Apache 2.0 §4(a) — a verbatim copy of the license of the codec's original weights,
        after a heading naming its scope.

        The heading names only the codec's two components (spelled as the manifest's weights
        keys). It goes after the MIT text (the leading MIT stays intact).
        """
        license_text = PIPELINE.root_files["LICENSE.md"]
        apache = apache_license_2_0()
        mit_end = len(mit_license(IRODORI_COPYRIGHTS))

        assert license_text.count(apache) == 1
        assert license_text.index(apache) > mit_end
        heading = next(
            line
            for line in license_text[mit_end:].splitlines()
            if line.startswith("## Apache License 2.0")
        )
        for role in IRODORI_GRAPH_ROLES:
            assert (f"`{role}`" in heading) == (role in IRODORI_CODEC_DIRS), role
        assert license_text.index(heading) < license_text.index(apache)
        assert IRODORI_CODEC_ORIGIN_MODEL in license_text[mit_end:]

    def test_it_keeps_the_text_backbones_copyright_next_to_the_authors(self) -> None:
        """The backbone is a fine-tune of `modernbert-ja-310m` — under MIT a derivative cannot drop
        the upstream notice."""
        license_text = PIPELINE.root_files["LICENSE.md"]

        assert "Copyright (c) 2026 Aratako\nCopyright (c) 2025 SB Intuitions\n" in license_text
        assert (
            "The above copyright notice and this permission notice shall be included in all"
            in license_text
        )

    @pytest.mark.parametrize("model", sorted(IRODORI_UPSTREAMS))
    def test_the_legal_text_passes_the_undeclared_file_gate(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, model: str
    ) -> None:
        """The two files the manifest does not declare pass `verify_dist` as `LEGAL_PATHS`
        entries."""
        out_dir = self._assemble(tmp_path, monkeypatch, model)

        assert sorted(LEGAL_PATHS) == ["LICENSE.md", "NOTICE.md"]
        assert all((out_dir / name).is_file() for name in LEGAL_PATHS)
        assert sorted(verify_dist(out_dir)) == sorted(_in_subtree(model, _stored_paths(model)))

    def test_the_notice_names_the_bundled_upstreams_and_every_graph(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The components the notice names are the same set as the manifest's weights (one side
        cannot grow alone)."""
        out_dir = self._assemble(tmp_path, monkeypatch, IRODORI_DEFAULT_MODEL)
        notice = (out_dir / "NOTICE.md").read_text(encoding="utf-8")
        manifest = json.loads((out_dir / MANIFEST_FILENAME).read_text(encoding="utf-8"))

        assert notice == IRODORI_NOTICE_MARKDOWN
        assert IRODORI_CODEC_MODEL in notice
        assert IRODORI_TEXT_BACKBONE_MODEL in notice
        for component in _irodori_model(manifest)["weights"]:
            assert f"`{component}`" in notice, component
        assert "quantized" in notice

    def test_the_notice_attributes_both_upstream_licenses(self) -> None:
        """NOTICE names both upstream lines: MIT (the Irodori side) and Apache 2.0 (the codec's
        original weights).

        Apache 2.0 §4(c) requires the attribution to be kept — naming only the codec's direct
        upstream (MIT) would leave no trace anywhere in the distribution repository that the
        original weights are Apache.
        """
        notice = IRODORI_NOTICE_MARKDOWN

        assert "licensed under the MIT License" in notice
        assert IRODORI_CODEC_MODEL in notice
        assert f"https://huggingface.co/{IRODORI_CODEC_ORIGIN_MODEL}" in notice
        assert f"https://huggingface.co/{IRODORI_CODEC_PARENT_MODEL}" in notice
        assert "licensed under the Apache License, Version 2.0" in notice
        for role in IRODORI_CODEC_DIRS:
            assert f"`{role}`" in notice, role

    def test_the_notice_holds_for_either_version_repository(self) -> None:
        """MUST: the body's upstream repository is not named — one set of `root_files` goes into
        both the v4 and v4.1 repositories, so naming a version would make one repository's notice
        disagree with its contents.
        """
        for upstream in IRODORI_UPSTREAMS.values():
            assert upstream.repo not in IRODORI_NOTICE_MARKDOWN
            assert upstream.display not in IRODORI_NOTICE_MARKDOWN
        assert "the Irodori-TTS checkpoint listed in `README.md`" in IRODORI_NOTICE_MARKDOWN


class TestIrodoriCli:
    @staticmethod
    def _reroot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        import dist
        from irodori import distribution

        # `DIST_ROOT` is on the driver, which decides the default output directory
        # (`dist.default_out_dir`), `INPUTS_ROOT` on the recipe, which looks up the inputs outside
        # the series — the bindings of the two modules are replaced separately.
        monkeypatch.setattr(dist, "DIST_ROOT", tmp_path / "models")
        monkeypatch.setattr(distribution, "INPUTS_ROOT", tmp_path / "inputs")

    def test_the_default_output_directory_follows_the_single_model(self) -> None:
        assert default_out_dir(PIPELINE, [IRODORI_DEFAULT_MODEL]).name == "karume-irodori-v4-small"

    def test_one_attribution_profile_needs_no_choice(self) -> None:
        """Only the weights of one upstream repository are carried over, so there is one
        attribution (nothing to choose)."""
        profiles = PIPELINE.card_profiles
        assert len(profiles) == 1
        assert resolve_card_renderer(PIPELINE, None) is next(iter(profiles.values()))

    def test_it_assembles_into_the_pipeline_default_directory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sources = _build_irodori_sources(tmp_path)
        self._reroot(tmp_path, monkeypatch)

        main(["--pipeline", "irodori", "--series", str(sources.series.parent)])

        out_dir = tmp_path / "models" / irodori_repo_name(IRODORI_DEFAULT_MODEL)
        expected = _in_subtree(IRODORI_DEFAULT_MODEL, _placed_paths())
        assert sorted(verify_dist(out_dir)) == sorted(expected)

    def test_the_model_flag_moves_the_series_and_the_default_directory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # NOTE: the card's attribution is looked up from the model name in `IRODORI_UPSTREAMS`
        # (2026-09-01), so a fake name like the old `v9-large` makes the assembly fail loudly. With
        # a real key other than the default, the main point, "--model moves the series and the
        # output directory", is still observable.
        sources = _build_irodori_sources(tmp_path, model="v4.1-small")
        self._reroot(tmp_path, monkeypatch)

        main(
            [
                "--pipeline",
                "irodori",
                "--model",
                "v4.1-small",
                "--series",
                str(sources.series.parent),
                *_v4_mirror_for("v4.1-small", tmp_path, monkeypatch),
            ]
        )

        out_dir = tmp_path / "models" / "karume-irodori-v4.1-small"
        manifest = json.loads((out_dir / MANIFEST_FILENAME).read_text(encoding="utf-8"))
        assert list(manifest["models"]) == ["v4.1-small"]
        assert verify_dist(out_dir)


class TestIrodoriCrossRepoReferences:
    """v4.1-small references the components it shares with v4-small in v4-small's repository
    instead of storing them a second time (ADR 0038 §7 addendum 2026-10-28)."""

    #: The 26 roles, spelled out (not derived like the recipe's table — so a change of the
    #: dtype tables cannot move the declaration silently).
    _SHARED = frozenset(
        {
            *(
                f"{role}_{dtype}"
                for role in (
                    "backbone",
                    "text_proj",
                    "caption_proj",
                    "speaker",
                    "dit",
                    "dit_context",
                    "codec_decoder",
                    "codec_encoder",
                )
                for dtype in ("f32", "f16", "i8")
            ),
            "dit_context_i4",
            "tokenizer",
        }
    )

    _REVISION = "485bbee5f0331db555d465955fef48a183bc46e0"

    def test_v41_declares_exactly_the_shared_roles_at_v4_small(self) -> None:
        (reference,) = irodori_references(IRODORI_V41_MODEL)

        assert reference.repo == "hdae/karume-irodori-v4-small"
        assert reference.model == IRODORI_V4_MODEL
        assert len(reference.roles) == 26
        assert set(reference.roles) == self._SHARED

    def test_the_containers_whose_bytes_differ_stay_local(self) -> None:
        """References are per container: `duration_*` (retrained upstream) and `dit_i4`
        (calibrated per model) are stored in v4.1-small's own repository."""
        local = set(IRODORI_OUTPUT_PATHS) - set(IRODORI_V41_SHARED_ROLES)

        assert local == {"duration_f32", "duration_f16", "duration_i8", "dit_i4"}

    def test_v4_small_is_self_contained(self) -> None:
        assert irodori_references(IRODORI_V4_MODEL) == ()

    def test_the_pipeline_carries_the_declaration(self) -> None:
        assert PIPELINE.references is irodori_references

    def test_v41_stores_only_what_differs_and_borrows_every_part_of_the_rest(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from irodori import distribution

        sources = _build_irodori_sources(tmp_path, model=IRODORI_V41_MODEL)
        _v4_mirror_for(IRODORI_V41_MODEL, tmp_path, monkeypatch)
        monkeypatch.setattr(distribution, "INPUTS_ROOT", tmp_path / "inputs")
        out_dir = tmp_path / "dist"

        main(
            [
                "--pipeline",
                "irodori",
                "--model",
                IRODORI_V41_MODEL,
                "--series",
                str(sources.series.parent),
                "--out",
                str(out_dir),
                "--ref-revision",
                f"{irodori_repo_name(IRODORI_V4_MODEL)}={self._REVISION}",
            ]
        )

        manifest = json.loads((out_dir / MANIFEST_FILENAME).read_text(encoding="utf-8"))
        model = manifest["models"][IRODORI_V41_MODEL]
        borrowed = {
            (name, label): {
                (ref.get("repo"), ref.get("revision")) for ref in entry["container"]["parts"]
            }
            for name, labels in model["weights"].items()
            for label, entry in labels.items()
        }
        local = {("duration", "f32"), ("duration", "f16"), ("duration", "i8"), ("dit", "i4")}
        pinned = {("hdae/karume-irodori-v4-small", self._REVISION)}
        for key, sources_of_parts in borrowed.items():
            assert sources_of_parts == ({(None, None)} if key in local else pinned), key
        tokenizer = model["assets"]["tokenizer"]
        assert (tokenizer["repo"], tokenizer["revision"]) == next(iter(pinned))
        # Only the four differing containers are files of this repository.
        stored = {path.split("/")[1] for path in verify_dist(out_dir)}
        assert stored == {"duration", "dit"}
        assert all(
            path.startswith((f"{IRODORI_V41_MODEL}/duration/", f"{IRODORI_V41_MODEL}/dit/model.i4"))
            for path in verify_dist(out_dir)
        )
