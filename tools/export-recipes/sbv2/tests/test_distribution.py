"""SBV2's distribution recipe (`sbv2.distribution`) — unit tests of one assembly pass.

No real assets are used. What goes into the weights slots are **valid minimal IR containers** of a
few KB (`ir_fixtures` — {@link _sbv2_container}); only the cases that check that a gate rejects
something keep the older fake assets. SBV2 also reads `config.json` and `style_vectors.npy`, and a
synthetic one of a few lines / a few KB is enough for each (**the synthetic config's style2id is in
a different order from the real FN4 weights** — baking a table in would fail).

Since manifest v2 (`karume/2` — ADR 0041) the in-repository layout is uniformly "a subtree per
model + `shared/`", so every expected path starts with `<model name>/…`.

The layers observable with core alone (the size limit a synthetic plan is enough for, the complete
quant mapping, the staging/swap invariants) live in `tools/exporter/tests/test_dist.py` (the split
of ADR 0065 steps 3+4).
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from container_series import (
    part_paths,
    placed_paths,
    read_component,
    replace_component,
    write_component,
)
from ir_fixtures import ir_container
from safetensors.numpy import load_file, save_file

from _shared.licenses import CC_BY_SA_4_0_PATH
from dist import main
from karume.dist import (
    LEGAL_PATHS,
    MANIFEST_FILENAME,
    MANIFEST_FORMAT,
    MODEL_CARD_FILENAME,
    SHARED_DIRNAME,
    DistError,
    assemble_family,
    assert_quant_presentation,
    resolve_card_renderer,
    verify_dist,
)
from sbv2.card import SBV2_JVNV_CORPUS_PAGE, SBV2_JVNV_PROFILE, SBV2_TEXT_ENCODER_MODEL
from sbv2.distribution import (
    EXPORT_PROVENANCE_FILE,
    FN_PIPELINE,
    PIPELINE,
    SBV2_DEFAULT_MODEL,
    SBV2_DEFAULT_QUANT,
    SBV2_FAMILY_DIRS,
    SBV2_JVNV_NOTICE_MARKDOWN,
    SBV2_KNOB_KEYS,
    SBV2_MAX_FRAMES,
    SBV2_MAX_TOKENS,
    SBV2_OUTPUT_PATHS,
    SBV2_QUANT_ABBREVIATIONS,
    SBV2_QUANTS,
    SBV2_SPEAKER_KEY,
    SBV2_SPEAKER_TENSOR,
    SBV2_STORAGE_FORBIDDEN,
    SBV2_STORAGE_REQUIREMENTS,
    SBV2_STYLE_KEY,
    SBV2_SYM_EXPECTATIONS,
    SBV2_TEXT_ENCODER_INPUTS,
    SBV2_TEXT_ENCODER_LAYERS,
    SBV2_TEXT_ENCODER_VARIANT,
    SBV2_WEIGHTS,
    Sbv2Sources,
    sbv2_knob_defaults,
    sbv2_pipeline_config,
    sbv2_placements,
    sbv2_plan,
    sbv2_repo_name,
    sbv2_series_name,
    sbv2_sources,
    sbv2_speaker_embeddings,
    sbv2_style_vectors,
)


def _write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _in_subtree(model: str, paths: Iterable[str]) -> list[str]:
    """The expected paths inside a model subtree (the uniform layout of ADR 0041 §9)."""
    return [f"{model}/{rel}" for rel in paths]


def _placed_paths() -> list[str]:
    """The relative paths that appear in the distribution — **only the weights slots** expand into
    numbered parts (container-v1 §8).

    tokenizer / symbols / the 2 tables are assets slots (one file reference each), so they are not
    split.
    """
    return placed_paths(SBV2_OUTPUT_PATHS, SBV2_WEIGHTS)


def _present(out_dir: Path) -> list[str]:
    return sorted(str(path.relative_to(out_dir)) for path in out_dir.rglob("*") if path.is_file())


#: `style_bert_vits2` is the optional `sbv2` dependency group. Only the layer that takes the default
#: knobs from its constants needs it (the assembly itself receives the knobs as arguments, so every
#: path can be observed even without it).
requires_sbv2_package = pytest.mark.skipif(
    importlib.util.find_spec("style_bert_vits2") is None,
    reason="the sbv2 dependency group is missing (`uv sync --group sbv2` is required)",
)

#: The IR shape of the fake text_encoder and the extraction position of `symbols.json`. The gate
#: `assert_bert_hidden` passes only the combination **22 layers × 1 output × position 1** (the same
#: as the real assets).
#: The symbol name used when a baked constant is **deliberately** grown in the text_encoder slot
#: (the real thing has none — the slot of the vacuous-pass gate).
_SBV2_TEXT_ENCODER_SYMBOL = "T"

_SBV2_GRAPH_LAYERS = 22
_SBV2_GRAPH_OUTPUTS = 1
_SBV2_BERT_FROM_END = 1


#: Role (a slot with a dtype label) → **component name** = the key of the manifest's weights = the
#: container's graph name (container-v1 §2.1). `text_encoder` and `text_encoder_i4` are 2 dtypes of
#: the same single component, so they go by one spelling. Derived from the declaration (a copy would
#: let one of them move alone).
_SBV2_COMPONENTS: Mapping[str, str] = {
    files.file: component for component, labels in SBV2_WEIGHTS.items() for files in labels.values()
}

#: Role → the storage dtype of that series (the IR vocabulary).
_SBV2_STORAGES: Mapping[str, str] = {
    "text_encoder": "i8",
    "text_encoder_i4": "i4",
    **{
        f"{target}_{label}": label for target in ("front", "voice") for label in ("f16", "i8", "i4")
    },
}


def _sbv2_container(
    role: str,
    *,
    storage: str | None = None,
    model: str = "",
    layers: int = _SBV2_GRAPH_LAYERS,
    outputs: int = _SBV2_GRAPH_OUTPUTS,
    inputs: Iterable[str] = SBV2_TEXT_ENCODER_INPUTS,
    sym_max: int | None = None,
    baked: bool = True,
) -> list[bytes]:
    """**Valid containers** for the weights slots (a different byte sequence per role).

    The assembly opens the input containers and checks them against every rule of the declaration
    (`karume.dist.assert_weight_components_verified`), so the weights slots must hold the real
    thing. The side that makes a gate fail uses the same container and **shifts only the
    declaration from the real one**. This also holds the shapes the family-specific gates read —
    for text_encoder, 22 initializer names with layer indices × 1 output × the order of the inputs
    ({@link assert_bert_hidden}); for front / voice, the static dimension of the baked constant
    ({@link assert_baked_sym_max}).

    The storage vocabulary has the same shape as the real thing (only the eligible weights are
    compressed, bias / constants / scales are f32, i4 is a mix of i4 + i8 + f32 —
    {@link _SBV2_SERIES_HEADERS}), so the gates that look for slot mix-ups also apply to this
    shape. Passing `storage` makes **the shape stay the slot's while only the storage is of another
    series** (to run the 3×3 mix-ups through the real gate). `model` is the axis that shifts the
    bytes per model, and the rest are axes for making gates fail (layer count / output count / input
    order / baked limit / whether it is baked at all).
    """
    mark = f"{role}-{model}" if model else role
    dtype = storage if storage is not None else _SBV2_STORAGES[role]
    if role.startswith("text_encoder"):
        # A slot that adds a baked constant (when `sym_max` is passed) also grows the input
        # dimension that binds the symbol — a symbol without a binding point never appears in a
        # real graph.
        axis: Any = 4 if sym_max is None else _SBV2_TEXT_ENCODER_SYMBOL
        return ir_container(
            mark=mark,
            # The pseudo series also **go by the component name** (the assembly's gate checks
            # that the graph name matches the key).
            named=_SBV2_COMPONENTS[role],
            storage=dtype,
            inputs=tuple((name, (1, axis)) for name in inputs),
            outputs=[[1, axis]] * outputs,
            weights=[
                f"p_model_encoder_layer_{index}_attention_self_query_proj_weight"
                for index in range(layers)
            ],
            baked=None if sym_max is None else (_SBV2_TEXT_ENCODER_SYMBOL, sym_max),
        )
    expectation = SBV2_SYM_EXPECTATIONS[role]
    ceiling = expectation.sym_max if sym_max is None else sym_max
    return ir_container(
        mark=mark,
        named=_SBV2_COMPONENTS[role],
        storage=dtype,
        inputs=(("x", (expectation.symbol,)),),
        baked=(expectation.symbol, ceiling) if baked else None,
    )


#: Fake assets (a different byte sequence per role — a mix-up shows in the hash). The 8 model
#: roles carry in their headers the dtypes `SBV2_STORAGE_REQUIREMENTS` requires and are also read
#: as IR containers.
_SBV2_PAYLOADS = {
    **{role: _sbv2_container(role) for role in _SBV2_STORAGES},
    "tokenizer": b'{"deberta": true}',
}

#: Slot → the storage vocabulary the binding table of that series **always contains** (measured on
#: the real distributed assets — i4 is a mix, and the weights not eligible for i4 stay i8, so it
#: contains i8 too). To reproduce a mix-up, this whole set is swapped.
_SBV2_SERIES_HEADERS: Mapping[str, tuple[str, ...]] = {
    "text_encoder": ("f32", "i8"),
    "text_encoder_i4": ("f32", "i8", "i4"),
    "front_f16": ("f32", "f16"),
    "front_i8": ("f32", "i8"),
    "front_i4": ("f32", "i8", "i4"),
    "voice_f16": ("f32", "f16"),
    "voice_i8": ("f32", "i8"),
    "voice_i4": ("f32", "i8", "i4"),
}

#: Slots of the same graph (the range that the storage dtype alone can tell apart). A mix-up across
#: **different graphs**, such as `front_i8` → `voice_i8`, has the same storage form, so the dtype
#: gate cannot see it in principle — that belongs to the source path ({@link sbv2_placements}) and
#: the shape gates, and is not handled here.
_SBV2_SERIES_GROUPS: tuple[tuple[str, ...], ...] = (
    ("text_encoder", "text_encoder_i4"),
    ("front_f16", "front_i8", "front_i4"),
    ("voice_f16", "voice_i8", "voice_i4"),
)

#: The runtime knob defaults. **Different numbers from the real `style_bert_vits2` values** — so
#: the values themselves tell whether the assembly ships the knobs from its arguments or its own
#: table.
_SBV2_KNOBS: Mapping[str, Any] = {
    "style": "Neutral",
    "styleWeight": 0.5,
    "sdpRatio": 0.25,
    "noiseScale": 0.55,
    "noiseScaleW": 0.75,
    "lengthScale": 1.25,
}

#: The model-specific fields of `symbols.json`. **Different numbers from the real weights (44100 /
#: 512)** — so the values tell whether the assembly matches them against config.json or bakes in
#: the real numbers.
_SBV2_SAMPLING_RATE = 22050
_SBV2_HOP_LENGTH = 256

#: The `symbols.json` of the shared slot (only the rules common to every model — it holds neither
#: knob defaults nor style / speaker; the MUST of `sbv2.demo.jp_extra_rules`).
_SBV2_SYMBOLS = json.dumps(
    {
        "bertHiddenFromEnd": _SBV2_BERT_FROM_END,
        "samplingRate": _SBV2_SAMPLING_RATE,
        "hopLength": _SBV2_HOP_LENGTH,
    },
    ensure_ascii=False,
).encode("utf-8")

#: The width of the speaker embedding (the synthetic config's `model.gin_channels`. The real FN4
#: weights use 512, but for the fake assets it is enough to see that the width is constrained — the
#: match with the real value is derived from the config, so it is not pinned).
_SBV2_GIN_CHANNELS = 8

#: The synthetic config. **In a different order from the real FN4 weights' style2id (Neutral / high
#: / low / NSFW)**, with 2 speakers. Shipping a baked-in table fails here.
_SBV2_CONFIG: Mapping[str, Any] = {
    "version": "2.6.1-JP-Extra",
    "data": {
        "spk2id": {"kappa": 0, "sigma": 1},
        "n_speakers": 2,
        "num_styles": 3,
        "style2id": {"Neutral": 0, "Sleepy": 1, "Shout": 2},
        "sampling_rate": _SBV2_SAMPLING_RATE,
        "hop_length": _SBV2_HOP_LENGTH,
    },
    "model": {"gin_channels": _SBV2_GIN_CHANNELS},
}


def _style_table(rows: int, offset: float = 0.0) -> np.ndarray:
    """A `[rows, 256]` f32 with a different value per row (to check identity after moving it into
    the distribution)."""
    return np.arange(rows * 256, dtype=np.float32).reshape(rows, 256) + offset


def _speaker_table(rows: int, cols: int = _SBV2_GIN_CHANNELS, offset: float = 0.0) -> np.ndarray:
    """A `[rows, cols]` f32 corresponding to the ckpt's `emb_g.weight` (its values are a different
    series from the style table)."""
    return (np.arange(rows * cols, dtype=np.float32) * 0.5 - 1.0).reshape(rows, cols) + offset


def _write_ckpt(model_dir: Path, tensors: Mapping[str, np.ndarray]) -> None:
    """Puts a fake real-weights ckpt in place (dist requires exactly one `*.safetensors`)."""
    model_dir.mkdir(parents=True, exist_ok=True)
    save_file(dict(tensors), str(model_dir / "ckpt.safetensors"))


def _build_sbv2_sources(
    root: Path,
    *,
    model: str = SBV2_DEFAULT_MODEL,
    config: Mapping[str, Any] = _SBV2_CONFIG,
    style_rows: int = 3,
    speaker_rows: int = 2,
    speaker_cols: int = _SBV2_GIN_CHANNELS,
    symbols: bytes = _SBV2_SYMBOLS,
    offset: float = 0.0,
) -> Sbv2Sources:
    """Reproduces the locations of the series + demo assets + real weights with fake assets
    (including things that are not distributed mixed in).

    The 3 locations follow the real layout of `_shared.paths` (`outputs/series/` / directly under
    `outputs/` / `inputs/`) — so a test of the CLI path can point at the same tree just by swapping
    the root. `offset` is the axis that puts a per-model difference into the tables and the weights
    (to check the family assembly's sharing decision).
    """
    series = root / "outputs" / "series"
    stem = sbv2_series_name(model)
    sources = Sbv2Sources(
        series_f16=series / f"{stem}-f16",
        series_i8=series / f"{stem}-i8",
        series_i4=series / f"{stem}-i4",
        text_encoder=series / "deberta-i8" / "sbv2-22layer",
        text_encoder_i4=series / "deberta-i4" / "sbv2-22layer",
        demo=root / "outputs" / "misc" / "sbv2-demo",
        model=root / "inputs" / "sbv2" / model,
    )
    write_component(sources.text_encoder / "model.krm", _SBV2_PAYLOADS["text_encoder"])
    _write(sources.text_encoder / "io.case0.safetensors", b"io-fixture")
    write_component(sources.text_encoder_i4 / "model.krm", _SBV2_PAYLOADS["text_encoder_i4"])
    _write(sources.text_encoder_i4 / "io.case0.safetensors", b"io-fixture")
    for series_dir, label in (
        (sources.series_f16, "f16"),
        (sources.series_i8, "i8"),
        (sources.series_i4, "i4"),
    ):
        for target in ("front", "voice"):
            role = f"{target}_{label}"
            payload = _SBV2_PAYLOADS[role] if not offset else _sbv2_container(role, model=model)
            write_component(series_dir / target / "model.krm", payload)
            _write(series_dir / target / "io.p2.safetensors", b"io-fixture")
        # The standalone graphs that are not distributed (for golden verification only) also sit
        # in the series.
        for target in ("dp", "flow", "dec"):
            _write(series_dir / target / "model.safetensors", b"not-distributed")
    _write(sources.demo / "deberta-tokenizer.json", _SBV2_PAYLOADS["tokenizer"])
    _write(sources.demo / "symbols.json", symbols)
    # Demo-only assets (not part of the distribution).
    _write(sources.demo / "assets.safetensors", b"demo-only")
    _write(sources.model / "config.json", json.dumps(config, ensure_ascii=False).encode("utf-8"))
    sources.model.mkdir(parents=True, exist_ok=True)
    np.save(sources.model / "style_vectors.npy", _style_table(style_rows, offset))
    # The ckpt holds more than the speaker embedding (a shape that shows exactly one is picked by
    # name).
    _write_ckpt(
        sources.model,
        {
            "emb_g.weight": _speaker_table(speaker_rows, speaker_cols, offset),
            "dec.conv_pre.weight": np.zeros((2, 2), dtype=np.float32),
        },
    )
    return sources


def _assemble_sbv2(
    sources: Sbv2Sources, out_dir: Path, model: str = SBV2_DEFAULT_MODEL
) -> dict[str, Any]:
    return assemble_family([sbv2_plan(sources, _SBV2_KNOBS, model)], out_dir, model)


@pytest.fixture
def sbv2_assembled(tmp_path: Path) -> tuple[Path, dict]:
    sources = _build_sbv2_sources(tmp_path)
    out_dir = tmp_path / "models" / sbv2_repo_name("jvnv")
    manifest = _assemble_sbv2(sources, out_dir)
    return out_dir, manifest


def _sbv2_model(manifest: Mapping[str, Any], name: str = SBV2_DEFAULT_MODEL) -> Mapping[str, Any]:
    return manifest["models"][name]


class TestSbv2Layout:
    def test_it_places_every_declared_path_under_the_model_subtree(self, sbv2_assembled) -> None:
        out_dir, _ = sbv2_assembled
        expected = _in_subtree(SBV2_DEFAULT_MODEL, _placed_paths())
        assert _present(out_dir) == sorted([*expected, MANIFEST_FILENAME])

    def test_it_never_carries_io_fixtures_into_the_distribution(self, sbv2_assembled) -> None:
        out_dir, _ = sbv2_assembled
        assert list(out_dir.rglob("io.*")) == []

    def test_it_leaves_the_single_graphs_out_of_the_distribution(self, sbv2_assembled) -> None:
        """dp / flow / dec are for golden verification only (the fused front / voice are the
        whole execution path)."""
        out_dir, manifest = sbv2_assembled
        model = _sbv2_model(manifest)
        assert sorted(model["weights"]) == ["front", "text_encoder", "voice"]
        assert sorted(model["assets"]) == [
            "speaker_embeddings",
            "style_vectors",
            "symbols",
            "tokenizer",
        ]
        for target in ("dp", "flow", "dec", "decoder", "duration_predictor"):
            assert not (out_dir / SBV2_DEFAULT_MODEL / target).exists()

    def test_it_renames_the_two_series_into_dtype_files(self, sbv2_assembled) -> None:
        out_dir, _ = sbv2_assembled
        for role in ("front_f16", "front_i8", "voice_f16", "voice_i8"):
            placed = out_dir / SBV2_DEFAULT_MODEL / SBV2_OUTPUT_PATHS[role]
            assert read_component(placed) == _SBV2_PAYLOADS[role]

    def test_it_reassembles_over_a_previous_run(self, tmp_path: Path) -> None:
        sources = _build_sbv2_sources(tmp_path)
        out_dir = tmp_path / "models" / sbv2_repo_name("jvnv")
        _assemble_sbv2(sources, out_dir)
        _assemble_sbv2(sources, out_dir)  # does not fail when links already exist
        assert verify_dist(out_dir)

    def test_it_rewrites_a_generated_table_without_touching_the_source(
        self, tmp_path: Path
    ) -> None:
        """A generated product (a table) is written after unlinking the previous link — opening it
        and writing would corrupt the source asset."""
        sources = _build_sbv2_sources(tmp_path)
        out_dir = tmp_path / "models" / sbv2_repo_name("jvnv")
        _assemble_sbv2(sources, out_dir)
        _assemble_sbv2(sources, out_dir)
        table = load_file(str(out_dir / SBV2_DEFAULT_MODEL / SBV2_OUTPUT_PATHS["style_vectors"]))[
            SBV2_STYLE_KEY
        ]
        assert np.array_equal(table, _style_table(3))
        assert np.array_equal(np.load(sources.model / "style_vectors.npy"), _style_table(3))


class TestSbv2StyleVectors:
    def test_it_converts_the_npy_into_a_single_f32_tensor(self, sbv2_assembled) -> None:
        out_dir, manifest = sbv2_assembled
        path = _sbv2_model(manifest)["assets"]["style_vectors"]["path"]
        tensors = load_file(str(out_dir / path))
        assert list(tensors) == [SBV2_STYLE_KEY]
        table = tensors[SBV2_STYLE_KEY]
        assert table.dtype == np.float32
        assert table.shape == (3, 256)
        assert np.array_equal(table, _style_table(3))

    def test_it_stops_when_the_row_count_disagrees_with_num_styles(self, tmp_path: Path) -> None:
        sources = _build_sbv2_sources(tmp_path, style_rows=2)
        out_dir = tmp_path / "models" / sbv2_repo_name("jvnv")
        with pytest.raises(
            DistError, match="the row count 2 does not match the config's num_styles 3"
        ):
            _assemble_sbv2(sources, out_dir)
        # The check comes before placement — not a single file of a partial distribution is left.
        assert not out_dir.exists()

    def test_it_stops_when_the_row_count_disagrees_with_the_style_names(
        self, tmp_path: Path
    ) -> None:
        """A form that matches `num_styles` but lacks names (the row-to-name mapping is broken)."""
        data = {**_SBV2_CONFIG["data"], "style2id": {"Neutral": 0, "Sleepy": 1}}
        config = {**_SBV2_CONFIG, "data": data}
        sources = _build_sbv2_sources(tmp_path, config=config)
        with pytest.raises(DistError, match=r"style2id \(2 entries\) — a style ID"):
            sbv2_style_vectors(sources.model, config)

    def test_it_stops_when_the_table_is_not_two_dimensional(self, tmp_path: Path) -> None:
        sources = _build_sbv2_sources(tmp_path)
        np.save(sources.model / "style_vectors.npy", np.zeros(256, dtype=np.float32))
        with pytest.raises(DistError, match=r"is not \[number of styles, 256\]"):
            sbv2_style_vectors(sources.model, _SBV2_CONFIG)

    def test_it_stops_when_the_table_is_not_256_wide(self, tmp_path: Path) -> None:
        """A table of the wrong width with only the row count matching is a form where the
        distribution assembles green and fails on the user's machine.

        `sbv2.export.style_vector`, which reads the same `.npy`, rejects the width, so if this
        slipped through, the accepted set for one asset would differ between scripts.
        """
        sources = _build_sbv2_sources(tmp_path)
        np.save(sources.model / "style_vectors.npy", np.zeros((3, 128), dtype=np.float32))
        with pytest.raises(DistError, match=r"\(3, 128\) is not \[number of styles, 256\]"):
            sbv2_style_vectors(sources.model, _SBV2_CONFIG)


class TestSbv2SpeakerEmbeddings:
    """`front` / `voice` both take `g` as a graph input — they cannot run without this table."""

    def test_it_converts_the_ckpt_tensor_into_a_single_f32_tensor(self, sbv2_assembled) -> None:
        out_dir, manifest = sbv2_assembled
        path = _sbv2_model(manifest)["assets"]["speaker_embeddings"]["path"]
        tensors = load_file(str(out_dir / path))
        assert list(tensors) == [SBV2_SPEAKER_KEY]
        table = tensors[SBV2_SPEAKER_KEY]
        assert table.dtype == np.float32
        assert table.shape == (2, _SBV2_GIN_CHANNELS)
        assert np.array_equal(table, _speaker_table(2))

    def test_it_picks_the_speaker_tensor_by_name(self, tmp_path: Path) -> None:
        """The ckpt holds other tensors too (if the lookup by name failed, other weights would be
        shipped)."""
        sources = _build_sbv2_sources(tmp_path)
        table = sbv2_speaker_embeddings(sources.model, _SBV2_CONFIG)
        assert np.array_equal(table, _speaker_table(2))

    def test_it_stops_when_the_row_count_disagrees_with_n_speakers(self, tmp_path: Path) -> None:
        sources = _build_sbv2_sources(tmp_path, speaker_rows=3)
        out_dir = tmp_path / "models" / sbv2_repo_name("jvnv")
        with pytest.raises(DistError, match="row count 3 does not match the config's n_speakers 2"):
            _assemble_sbv2(sources, out_dir)
        # The check comes before placement — not a single file of a partial distribution is left.
        assert not out_dir.exists()

    def test_it_stops_when_the_row_count_disagrees_with_the_speaker_names(
        self, tmp_path: Path
    ) -> None:
        """A form that matches `n_speakers` but lacks names (the row-to-name mapping is broken)."""
        data = {**_SBV2_CONFIG["data"], "spk2id": {"kappa": 0}}
        config = {**_SBV2_CONFIG, "data": data}
        sources = _build_sbv2_sources(tmp_path, config=config)
        with pytest.raises(DistError, match=r"spk2id \(1 entries\) — a speaker ID"):
            sbv2_speaker_embeddings(sources.model, config)

    def test_it_stops_when_the_column_count_disagrees_with_gin_channels(
        self, tmp_path: Path
    ) -> None:
        """The column count can be derived from the config (it is the width of the graph input g
        itself), so the whole shape is constrained."""
        sources = _build_sbv2_sources(tmp_path, speaker_cols=_SBV2_GIN_CHANNELS + 1)
        with pytest.raises(
            DistError, match="column count 9 does not match the config's gin_channels 8"
        ):
            sbv2_speaker_embeddings(sources.model, _SBV2_CONFIG)

    def test_it_stops_when_the_ckpt_has_no_speaker_embedding(self, tmp_path: Path) -> None:
        sources = _build_sbv2_sources(tmp_path)
        _write_ckpt(sources.model, {"dec.conv_pre.weight": np.zeros((2, 2), dtype=np.float32)})
        with pytest.raises(DistError, match=SBV2_SPEAKER_TENSOR.replace(".", r"\.")):
            sbv2_speaker_embeddings(sources.model, _SBV2_CONFIG)

    def test_it_stops_when_the_ckpt_is_not_unique(self, tmp_path: Path) -> None:
        """Closes the form where which ckpt it was taken from changes silently (the same
        requirement as `load_net_g`)."""
        sources = _build_sbv2_sources(tmp_path)
        save_file({"emb_g.weight": _speaker_table(2)}, str(sources.model / "another.safetensors"))
        with pytest.raises(DistError, match="the ckpt is not unique"):
            sbv2_speaker_embeddings(sources.model, _SBV2_CONFIG)

    def test_it_stops_when_the_config_has_no_model_section(self, tmp_path: Path) -> None:
        sources = _build_sbv2_sources(tmp_path)
        with pytest.raises(DistError, match="has no 'model' section"):
            sbv2_speaker_embeddings(sources.model, {"data": _SBV2_CONFIG["data"]})

    def test_the_speaker_map_resolves_into_the_shipped_table(self, sbv2_assembled) -> None:
        """The values of `pipelineConfig.speakers` are closed as row numbers of the speaker
        table."""
        out_dir, manifest = sbv2_assembled
        model = _sbv2_model(manifest)
        speakers = model["pipelineConfig"]["speakers"]
        table = load_file(str(out_dir / model["assets"]["speaker_embeddings"]["path"]))
        assert sorted(speakers.values()) == list(range(table[SBV2_SPEAKER_KEY].shape[0]))
        styles = model["pipelineConfig"]["styles"]
        style_table = load_file(str(out_dir / model["assets"]["style_vectors"]["path"]))
        assert sorted(styles.values()) == list(range(style_table[SBV2_STYLE_KEY].shape[0]))


class TestSbv2PipelineConfig:
    def test_it_reads_the_tables_from_the_config_instead_of_baking_them(self) -> None:
        """**The same path returning a different table** is the observable form of "not baked
        in"."""
        first = sbv2_pipeline_config(
            {"data": {"style2id": {"Neutral": 0, "Sleepy": 1}, "spk2id": {"kappa": 0}}},
            _SBV2_KNOBS,
        )
        second = sbv2_pipeline_config(
            {
                "data": {
                    "style2id": {"Neutral": 0, "Angry": 1, "Sad": 2},
                    "spk2id": {"omega": 0, "psi": 1},
                }
            },
            _SBV2_KNOBS,
        )
        assert first["styles"] == {"Neutral": 0, "Sleepy": 1}
        assert second["styles"] == {"Neutral": 0, "Angry": 1, "Sad": 2}
        assert first["speakers"] == {"kappa": 0}
        assert second["speakers"] == {"omega": 0, "psi": 1}

    def test_the_assembled_manifest_carries_the_config_tables(self, sbv2_assembled) -> None:
        _, manifest = sbv2_assembled
        config = _sbv2_model(manifest)["pipelineConfig"]
        assert config["styles"] == _SBV2_CONFIG["data"]["style2id"]
        assert config["speakers"] == _SBV2_CONFIG["data"]["spk2id"]

    def test_it_defaults_to_the_first_speaker_of_the_config(self, sbv2_assembled) -> None:
        _, manifest = sbv2_assembled
        assert _sbv2_model(manifest)["pipelineConfig"]["defaults"]["speaker"] == "kappa"

    def test_it_fills_the_knob_defaults_from_the_given_values(self, sbv2_assembled) -> None:
        _, manifest = sbv2_assembled
        defaults = _sbv2_model(manifest)["pipelineConfig"]["defaults"]
        assert {key: defaults[key] for key in SBV2_KNOB_KEYS} == dict(_SBV2_KNOBS)

    def test_it_stops_when_the_default_style_is_absent_from_the_config(self) -> None:
        knobs = {**_SBV2_KNOBS, "style": "Whisper"}
        with pytest.raises(DistError, match="the default style 'Whisper' is not in"):
            sbv2_pipeline_config(_SBV2_CONFIG, knobs)

    def test_it_stops_when_a_knob_default_is_missing(self) -> None:
        knobs = {key: value for key, value in _SBV2_KNOBS.items() if key != "noiseScaleW"}
        with pytest.raises(DistError, match="runtime knob defaults are missing"):
            sbv2_pipeline_config(_SBV2_CONFIG, knobs)

    def test_it_stops_when_the_config_has_no_style_table(self) -> None:
        with pytest.raises(DistError, match=r"data\.style2id"):
            sbv2_pipeline_config({"data": {"spk2id": {"kappa": 0}}}, _SBV2_KNOBS)

    def test_it_stops_when_the_config_has_no_data_section(self) -> None:
        with pytest.raises(DistError, match="has no 'data' section"):
            sbv2_pipeline_config({"model": {"gin_channels": 512}}, _SBV2_KNOBS)

    def test_it_declares_the_operating_limits_the_loader_requires(self, sbv2_assembled) -> None:
        """The 2 limit fields exist only in the distribution (the loader has no constants and
        cannot read it if they are missing)."""
        _, manifest = sbv2_assembled
        config = _sbv2_model(manifest)["pipelineConfig"]
        assert config["maxTokens"] == SBV2_MAX_TOKENS
        assert config["maxFrames"] == SBV2_MAX_FRAMES

    def test_the_limits_are_the_symbolic_maxima_the_export_scripts_baked(self) -> None:
        """The shipped limits and the limits of the baked symbolic dimensions are the same single
        pair.

        If they drift, it becomes "the declaration passes but the graph's relative-position table
        is too short", which **shows up only on the user's machine after shipping** (front / voice
        do not report the limit of their symbolic dimension at run time). This also pins that the
        token sequence T, DeBERTa's symbolic dimension, equals the limit of the phoneme dimension P
        (front) — if only one of them moved, the meaning of the limit would split.
        """
        from deberta import export as export_deberta
        from sbv2 import export as export_sbv2

        assert SBV2_MAX_TOKENS == export_deberta.SYM_MAX
        assert SBV2_MAX_TOKENS == export_sbv2.SYM_MAX
        assert SBV2_MAX_FRAMES == export_sbv2.FLOW_SYM_MAX


class TestSbv2StorageGate:
    """The storage dtype gate (grounded in the same real accident as Anima — raw F32 stayed silent
    all the way to the reference-match gate)."""

    def test_it_stops_when_an_f16_series_holds_raw_f32(self, tmp_path: Path) -> None:
        sources = _build_sbv2_sources(tmp_path)
        replace_component(
            sources.series_f16 / "front" / "model.krm",
            _sbv2_container("front_f16", storage="f32"),
        )
        out_dir = tmp_path / "models" / sbv2_repo_name("jvnv")
        with pytest.raises(DistError, match=r"front_f16: .* has no f16"):
            _assemble_sbv2(sources, out_dir)
        # The check comes before placement — not a single file of a partial distribution is left.
        assert not out_dir.exists()

    def test_it_stops_when_an_i8_series_lacks_i8_storage(self, tmp_path: Path) -> None:
        sources = _build_sbv2_sources(tmp_path)
        replace_component(
            sources.series_i8 / "voice" / "model.krm",
            _sbv2_container("voice_i8", storage="f16"),
        )
        with pytest.raises(DistError, match=r"voice_i8: .* has no i8"):
            _assemble_sbv2(sources, tmp_path / "out")

    def test_it_stops_when_the_text_encoder_is_not_i8(self, tmp_path: Path) -> None:
        """DeBERTa ships only one i8 series (1.32GB of f32 is unrealistic for distribution)."""
        sources = _build_sbv2_sources(tmp_path)
        replace_component(
            sources.text_encoder / "model.krm",
            _sbv2_container("text_encoder", storage="f32"),
        )
        with pytest.raises(DistError, match=r"text_encoder: .* has no i8"):
            _assemble_sbv2(sources, tmp_path / "out")

    def test_it_stops_when_the_i4_seat_holds_the_i8_series(self, tmp_path: Path) -> None:
        """Putting the i8 series into the i4 slot silently assembles a distribution whose size
        alone went back to the original.

        The 2 slots are the same 22-layer DeBERTa, so the layer count, output and input gates
        **both pass** — only the storage dtype requirement can tell the slots apart.
        """
        sources = _build_sbv2_sources(tmp_path)
        replace_component(sources.text_encoder_i4 / "model.krm", _SBV2_PAYLOADS["text_encoder"])
        with pytest.raises(DistError, match=r"text_encoder_i4: .* has no i4"):
            _assemble_sbv2(sources, tmp_path / "out")

    def test_it_stops_when_a_voice_i4_seat_holds_the_i8_series(self, tmp_path: Path) -> None:
        """The same for net_g's i4 slot — the gain of the mix is small, so a mix-up is not visible
        from the size difference either.

        The i4 mix of `front` / `voice` is another series of the same graph (only the 6 eligible
        linears are i4), so putting in the i8 series matches the layer count, the inputs and the
        outputs alike. Only the storage dtype requirement can tell the slots apart.
        """
        sources = _build_sbv2_sources(tmp_path)
        replace_component(sources.series_i4 / "voice" / "model.krm", _SBV2_PAYLOADS["voice_i8"])
        with pytest.raises(DistError, match=r"voice_i4: .* has no i4"):
            _assemble_sbv2(sources, tmp_path / "out")

    def test_it_stops_when_the_i4_series_lands_in_the_voice_i8_seat(self, tmp_path: Path) -> None:
        """The mix-up in the other direction (i4 series → i8 slot) — an existence check alone
        **lets it through**.

        The i4 series is a mix, and the weights not eligible for i4 stay i8, so it always contains
        i8 and satisfies "contains i8". The i8 slot computes in f32, so execution raises nothing
        either, and with the slot name and the path still `model.i8.krm` only the audio comes out
        at i4 quality. The forbidden table (`SBV2_STORAGE_FORBIDDEN`) is the only detector.
        """
        sources = _build_sbv2_sources(tmp_path)
        replace_component(sources.series_i8 / "voice" / "model.krm", _SBV2_PAYLOADS["voice_i4"])
        out_dir = tmp_path / "models" / sbv2_repo_name("jvnv")
        with pytest.raises(DistError, match=r"voice_i8: .* has i4"):
            _assemble_sbv2(sources, out_dir)
        # The check comes before placement — not a single file of a partial distribution is left.
        assert not out_dir.exists()

    def test_it_stops_when_the_i4_series_lands_in_the_text_encoder_seat(
        self, tmp_path: Path
    ) -> None:
        """The same for DeBERTa's i8 slot — the 2 are the same 22 layers, so the shape gates both
        pass."""
        sources = _build_sbv2_sources(tmp_path)
        replace_component(sources.text_encoder / "model.krm", _SBV2_PAYLOADS["text_encoder_i4"])
        with pytest.raises(DistError, match=r"text_encoder: .* has i4"):
            _assemble_sbv2(sources, tmp_path / "out")

    def test_no_series_slips_into_another_seat_of_the_same_graph(self) -> None:
        """**Every** mix-up of a slot of the same graph × another series fails on either a
        requirement or a prohibition.

        If only one of the tables is updated on the day a slot is added, the combinations that slip
        through the net silently line up in the distribution (a mix-up of series roots cannot in
        principle be detected by the numeric gates — ADR 0027 / 0029).
        """
        # MUST: tie the enumeration to the production tables. With a dict inside the test, when a
        # 4th series appeared in `SBV2_STORAGE_REQUIREMENTS` and was not added here, it would stay
        # green without ever seeing the very failure mode the docstring names.
        assert set(_SBV2_SERIES_HEADERS) == set(SBV2_STORAGE_REQUIREMENTS)
        assert {role for group in _SBV2_SERIES_GROUPS for role in group} == set(
            SBV2_STORAGE_REQUIREMENTS
        )

        for group in _SBV2_SERIES_GROUPS:
            for seat in group:
                for series in group:
                    found = _SBV2_SERIES_HEADERS[series]
                    caught = SBV2_STORAGE_REQUIREMENTS[seat] not in found or any(
                        dtype in found for dtype in SBV2_STORAGE_FORBIDDEN.get(seat, ())
                    )
                    assert caught is (series != seat), f"{series} → the {seat} slot"

    def test_every_same_graph_seat_mix_up_is_refused_by_the_real_gates(
        self, tmp_path: Path
    ) -> None:
        """Runs through the **real gates** (`assert_storage` / `assert_storage_absent`) instead
        of the table above.

        The test above reimplements the predicate, so it would not fail even if the call to
        `assert_storage_absent` disappeared from `sbv2_plan`. This one actually goes through the
        assembly, so the moment the call is dropped the off-diagonal turns green and it fails.
        """
        for group in _SBV2_SERIES_GROUPS:
            for seat in group:
                for series in group:
                    sources = _build_sbv2_sources(tmp_path / f"{seat}-{series}")
                    # The graph's shape claims to be **the slot's** (to run only the storage
                    # dtype gate, not the symbolic dimension gate <CG4-3> — moving both the slot
                    # and the series would make it unclear which gate failed). Only the storage
                    # dtype moves, and its set is the real shape named by
                    # {@link _SBV2_SERIES_HEADERS}.
                    replace_component(
                        sbv2_placements(sources)[seat],
                        _sbv2_container(
                            seat, storage=_SBV2_STORAGES[series], model="swapped-series"
                        ),
                    )
                    out_dir = tmp_path / "out" / f"{seat}-{series}"

                    if series == seat:
                        # The diagonal passes (the same series into the same slot).
                        _assemble_sbv2(sources, out_dir)
                    else:
                        with pytest.raises(DistError):
                            _assemble_sbv2(sources, out_dir)


class TestSbv2SymGate:
    """The gate on the limit of the symbolic dimension (CG4-3) — closes the disconnect between a
    non-default `--sym-max` and the manifest's constants.

    `sbv2.export --target voice --sym-max 1024` passes on the official CLI, and the golden only goes
    up to 512, so the export holds. The distribution side bakes `maxFrames` as 4096, so **the
    distribution is green too**, and it first fails with `sym_prefix_slice` exceeding Tmax when a
    user asks for an utterance of 1025–4096 frames.

    Only the text_encoder slot points the other way: it checks that there is **no** baked constant
    carrying the limit (the very premise that the record is the only means of matching) — if one is
    baked back in, it is the signal to move the slot's classification.
    """

    @staticmethod
    def _rebake(sources: Sbv2Sources, role: str, symbol: str, sym_max: int) -> None:
        """Makes only the slot's asset look re-baked at another symbolic dimension (the storage
        dtype stays correct)."""
        del symbol  # The slot decides the symbol (only the limit is re-baked).
        replace_component(sbv2_placements(sources)[role], _sbv2_container(role, sym_max=sym_max))

    def test_it_stops_when_the_voice_graph_is_baked_at_another_frame_max(
        self, tmp_path: Path
    ) -> None:
        sources = _build_sbv2_sources(tmp_path)
        self._rebake(sources, "voice_i8", "T", 1024)
        out_dir = tmp_path / "models" / sbv2_repo_name("jvnv")

        with pytest.raises(DistError, match=r"voice_i8: .*the limit 1024, .*declares 4096"):
            _assemble_sbv2(sources, out_dir)
        # The check comes before placement — not a single file of a partial distribution is left.
        assert not out_dir.exists()

    def test_it_stops_when_the_front_graph_is_baked_at_another_token_max(
        self, tmp_path: Path
    ) -> None:
        """The front / maxTokens side has the same hole (`--target front --sym-max` passes too)."""
        sources = _build_sbv2_sources(tmp_path)
        self._rebake(sources, "front_f16", "P", 256)

        with pytest.raises(DistError, match=r"front_f16: .*the limit 256, .*declares 512"):
            _assemble_sbv2(sources, tmp_path / "out")

    def test_it_stops_when_the_text_encoder_grew_a_baked_symbol_slice(self, tmp_path: Path) -> None:
        """The vacuous-pass gate of the text_encoder slot — "cannot be read" is a fact about the
        graph's shape, not a permanent guarantee.

        DeBERTa promoted the relative-position index table to a graph input in ADR 0045 wave 3, so
        it has no baked constant carrying the limit, and the matching relies on the record alone (a
        series without a record is accepted). If the table were baked back in, a `sym_prefix_slice`
        would appear and become readable from the artifact, and then the right move is to put the
        slot on the `baked=True` side — it must not silently stay with the weaker record-only gate.
        """
        sources = _build_sbv2_sources(tmp_path)
        self._rebake(sources, "text_encoder", "T", SBV2_MAX_TOKENS)

        with pytest.raises(
            DistError, match=r"text_encoder: .* has sym_prefix_slice nodes .*: 1 of them"
        ):
            _assemble_sbv2(sources, tmp_path / "out")

    def test_it_stops_when_the_graph_has_no_baked_symbol_slice(self, tmp_path: Path) -> None:
        """The vacuous-pass gate — when the target node disappears, e.g. because the table was
        promoted to an input, it does not silently go green."""
        sources = _build_sbv2_sources(tmp_path)
        replace_component(
            sbv2_placements(sources)["front_i4"], _sbv2_container("front_i4", baked=False)
        )

        with pytest.raises(DistError, match=r"front_i4: .* has no sym_prefix_slice .* at all"):
            _assemble_sbv2(sources, tmp_path / "out")

    def test_the_expectations_cover_every_weights_seat(self) -> None:
        """On the day one more dtype slot is added, only the slots not named must not silently
        slip through.

        text_encoder is covered too (all 8 roles of the slots) — front and DeBERTa are separate
        scripts that share `SBV2_MAX_TOKENS`, so a pair where only one was re-taken with a
        non-default value assembles normally.
        """
        expected = {files.file for seats in SBV2_WEIGHTS.values() for files in seats.values()}

        assert set(SBV2_SYM_EXPECTATIONS) == expected
        assert {entry.sym_max for entry in SBV2_SYM_EXPECTATIONS.values()} == {
            SBV2_MAX_TOKENS,
            SBV2_MAX_FRAMES,
        }

    def test_only_the_text_encoder_seats_are_unreadable_from_the_artifact(self) -> None:
        """Only the text_encoder slots have `baked` false (the bake gate of front / voice is not
        dropped)."""
        unreadable = {role for role, entry in SBV2_SYM_EXPECTATIONS.items() if not entry.baked}

        assert unreadable == {files.file for files in SBV2_WEIGHTS["text_encoder"].values()}


class TestSbv2SymProvenance:
    """Matching the writer's record (`export_provenance.json`) against the slot's declaration."""

    @staticmethod
    def _record(sources: Sbv2Sources, role: str, record: Mapping[str, Any]) -> Path:
        path = sbv2_placements(sources)[role].parent / EXPORT_PROVENANCE_FILE
        path.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
        return path

    def test_a_matching_record_changes_nothing(self, tmp_path: Path) -> None:
        sources = _build_sbv2_sources(tmp_path)
        self._record(sources, "voice_i8", {"target": "voice", "sym_max": SBV2_MAX_FRAMES})

        _assemble_sbv2(sources, tmp_path / "out")

    def test_a_series_without_a_record_is_accepted(self, tmp_path: Path) -> None:
        """A series baked before records existed is not required to be re-exported (the baked
        dimension holds the same fact)."""
        sources = _build_sbv2_sources(tmp_path)

        _assemble_sbv2(sources, tmp_path / "out")

        assert not (sbv2_placements(sources)["voice_i8"].parent / EXPORT_PROVENANCE_FILE).exists()

    def test_it_stops_when_the_record_names_another_sym_max(self, tmp_path: Path) -> None:
        sources = _build_sbv2_sources(tmp_path)
        self._record(sources, "voice_f16", {"target": "voice", "sym_max": 1024})

        with pytest.raises(
            DistError, match=r"voice_f16: the sym_max in the provenance record is 1024"
        ):
            _assemble_sbv2(sources, tmp_path / "out")

    def test_it_stops_when_the_record_names_another_target(self, tmp_path: Path) -> None:
        """The mix-up of putting what was baked as `front` into the voice slot (only the record can
        name it)."""
        sources = _build_sbv2_sources(tmp_path)
        self._record(sources, "voice_i4", {"target": "front", "sym_max": SBV2_MAX_FRAMES})

        with pytest.raises(
            DistError, match=r"voice_i4: the target in the provenance record is 'front'"
        ):
            _assemble_sbv2(sources, tmp_path / "out")

    def test_a_matching_text_encoder_record_changes_nothing(self, tmp_path: Path) -> None:
        """The normal pair — the record of the 22-layer variant baked with the default
        `--sym-max` passes."""
        sources = _build_sbv2_sources(tmp_path)
        self._record(
            sources,
            "text_encoder",
            {"target": SBV2_TEXT_ENCODER_VARIANT, "sym_max": SBV2_MAX_TOKENS},
        )

        _assemble_sbv2(sources, tmp_path / "out")

    def test_it_stops_when_the_text_encoder_was_baked_at_another_token_max(
        self, tmp_path: Path
    ) -> None:
        """For the text_encoder slot **only the record** carries the limit (a deviation of
        `deberta.export --sym-max`).

        DeBERTa's graph has no baked constant carrying the limit (the relative-position tables are
        graph inputs — ADR 0045 wave 3), so there is no artifact-side detector like
        `assert_baked_sym_max`. `pipelineConfig.maxTokens` is baked as a constant, so unless this
        fails the distribution holds as "a text_encoder baked at 256 while declaring that it accepts
        up to 512".
        """
        sources = _build_sbv2_sources(tmp_path)
        self._record(
            sources, "text_encoder_i4", {"target": SBV2_TEXT_ENCODER_VARIANT, "sym_max": 256}
        )

        with pytest.raises(
            DistError, match=r"text_encoder_i4: the sym_max in the provenance record is 256"
        ):
            _assemble_sbv2(sources, tmp_path / "out")

    def test_it_stops_when_the_text_encoder_record_names_another_variant(
        self, tmp_path: Path
    ) -> None:
        """The mix-up of copying a verification variant (all layers out / dev) into the 22-layer
        slot.

        {@link assert_bert_hidden} checks the layer count and the output count, but that passes
        for **a 22-layer graph whose contents are correct** — a series re-baked from `--layers 22`
        with another output shape slips through the shape gates. Only the record can say which
        variant it was baked as.
        """
        sources = _build_sbv2_sources(tmp_path)
        self._record(sources, "text_encoder", {"target": "full-24layer", "sym_max": 512})

        with pytest.raises(
            DistError, match=r"text_encoder: the target in the provenance record is 'full-24layer'"
        ):
            _assemble_sbv2(sources, tmp_path / "out")

    def test_it_stops_when_the_record_is_not_json(self, tmp_path: Path) -> None:
        sources = _build_sbv2_sources(tmp_path)
        (sbv2_placements(sources)["front_i8"].parent / EXPORT_PROVENANCE_FILE).write_text(
            "not-json", encoding="utf-8"
        )

        with pytest.raises(DistError, match=r"front_i8: cannot parse the provenance record"):
            _assemble_sbv2(sources, tmp_path / "out")


class TestSbv2BertHiddenGate:
    """The gate that passes only the combination 22 layers × 1 output × position 1.

    The layer count and the output shape are held by the variant in `deberta/export.py`, and the
    position by a constant in `sbv2/demo.py` — separate scripts, so a distribution where one forgot
    to move them as a pair assembles normally. Even when they drift the shapes still match and it
    runs; it stays silent, only **producing audio from the BERT features of another layer**.
    """

    def test_it_stops_when_the_encoder_was_not_truncated(self, tmp_path: Path) -> None:
        """A 24-layer asset whose truncation was forgotten (1 output) has its final output at layer
        23, so it is another layer."""
        sources = _build_sbv2_sources(tmp_path)
        replace_component(
            sources.text_encoder / "model.krm", _sbv2_container("text_encoder", layers=24)
        )
        out_dir = tmp_path / "out"
        with pytest.raises(DistError, match=r"has 24 layers, not the expected 22"):
            _assemble_sbv2(sources, out_dir)
        # The check comes before placement — not a single file of a partial distribution is left.
        assert not out_dir.exists()

    def test_it_stops_when_the_verification_variant_slipped_in(self, tmp_path: Path) -> None:
        """If the all-layers-out (verification) asset gets into the distribution path, both the
        readback and the extraction position change."""
        sources = _build_sbv2_sources(tmp_path)
        replace_component(
            sources.text_encoder / "model.krm", _sbv2_container("text_encoder", outputs=23)
        )
        with pytest.raises(
            DistError, match=r"has 23 graph outputs, not the 1 the distribution requires"
        ):
            _assemble_sbv2(sources, tmp_path / "out")

    def test_it_stops_when_only_the_symbols_are_stale(self, tmp_path: Path) -> None:
        """The asset has one output but `symbols.json` still claims the 3 meant for the old
        all-layers-out graph."""
        symbols = json.dumps({"bertHiddenFromEnd": 3})
        sources = _build_sbv2_sources(tmp_path, symbols=symbols.encode("utf-8"))
        with pytest.raises(
            DistError, match=r"bertHiddenFromEnd=3 is not 1, the only value meaningful"
        ):
            _assemble_sbv2(sources, tmp_path / "out")

    def test_it_stops_when_the_position_tables_were_baked_back_in(self, tmp_path: Path) -> None:
        """Stops, on the distribution path, an asset whose index table left the inputs (= the
        form where 2MiB of constants were baked back in)."""
        sources = _build_sbv2_sources(tmp_path)
        replace_component(
            sources.text_encoder / "model.krm",
            _sbv2_container("text_encoder", inputs=("input_ids", "attention_mask")),
        )
        with pytest.raises(
            DistError, match=r"graph inputs of .* are \['input_ids', 'attention_mask'\]"
        ):
            _assemble_sbv2(sources, tmp_path / "out")

    def test_it_checks_every_text_encoder_seat(self, tmp_path: Path) -> None:
        """A gate applied per slot — looking only at the i8 slot lets a forgotten truncation in
        the i4 slot through.

        The 2 are products of separate `deberta/export.py` runs, so forgetting to move them as a
        pair happens in the ordinary course of things.
        """
        sources = _build_sbv2_sources(tmp_path)
        replace_component(
            sources.text_encoder_i4 / "model.krm", _sbv2_container("text_encoder_i4", layers=24)
        )
        with pytest.raises(DistError, match=r"has 24 layers, not the expected 22"):
            _assemble_sbv2(sources, tmp_path / "out")


class TestSbv2SymbolsModelFields:
    """The **model-specific fields** of `symbols.json` match the config of the model being
    shipped.

    The demo assets are written to one fixed place without the model name, so assembling in a tree
    where another model's assets are left over ships `samplingRate` alone with a different value
    (the TS side puts it as is into the `sampleRate` returned by `generate()`, so it stays
    **silent**).
    """

    @staticmethod
    def _symbols(**overrides: Any) -> bytes:
        payload = {
            "bertHiddenFromEnd": _SBV2_BERT_FROM_END,
            "samplingRate": _SBV2_SAMPLING_RATE,
            "hopLength": _SBV2_HOP_LENGTH,
            **overrides,
        }
        return json.dumps(payload, ensure_ascii=False).encode("utf-8")

    def test_the_matching_pair_passes(self, tmp_path: Path) -> None:
        sources = _build_sbv2_sources(tmp_path, symbols=self._symbols())

        assert _assemble_sbv2(sources, tmp_path / "out")["models"][SBV2_DEFAULT_MODEL]

    def test_it_stops_when_the_sampling_rate_disagrees_with_the_config(
        self, tmp_path: Path
    ) -> None:
        sources = _build_sbv2_sources(tmp_path, symbols=self._symbols(samplingRate=44100))
        out_dir = tmp_path / "out"

        with pytest.raises(DistError, match=r"samplingRate: symbols.json=44100"):
            _assemble_sbv2(sources, out_dir)
        # The check comes before placement — not a single file of a partial distribution is left.
        assert not out_dir.exists()

    def test_it_stops_when_the_hop_length_disagrees_with_the_config(self, tmp_path: Path) -> None:
        sources = _build_sbv2_sources(tmp_path, symbols=self._symbols(hopLength=512))

        with pytest.raises(DistError, match=r"hopLength: symbols.json=512"):
            _assemble_sbv2(sources, tmp_path / "out")

    def test_it_stops_when_the_field_is_missing_entirely(self, tmp_path: Path) -> None:
        """A missing field means "an old `sbv2.demo assets`" — a form that cannot be matched is not
        let through."""
        payload = json.loads(self._symbols())
        del payload["samplingRate"]
        sources = _build_sbv2_sources(
            tmp_path, symbols=json.dumps(payload, ensure_ascii=False).encode("utf-8")
        )

        with pytest.raises(DistError, match=r"samplingRate: symbols.json=None"):
            _assemble_sbv2(sources, tmp_path / "out")


class TestSbv2Manifest:
    def test_it_writes_the_envelope_of_adr_0041(self, sbv2_assembled) -> None:
        out_dir, manifest = sbv2_assembled
        on_disk = json.loads((out_dir / MANIFEST_FILENAME).read_text(encoding="utf-8"))
        assert on_disk == manifest
        assert manifest["format"] == MANIFEST_FORMAT
        assert manifest["defaultModel"] == SBV2_DEFAULT_MODEL
        assert _sbv2_model(manifest)["pipeline"] == "sbv2/1"
        assert manifest["generator"].startswith("karume/")

    def test_it_derives_size_and_sha256_from_the_placed_files(self, sbv2_assembled) -> None:
        out_dir, manifest = sbv2_assembled
        shards = _sbv2_model(manifest)["weights"]["front"]["i8"]["container"]["parts"]
        # The 3-piece set applies per shard (no way to make just some part of the list stale).
        for ref, payload in zip(shards, _SBV2_PAYLOADS["front_i8"], strict=True):
            assert ref["size"] == len(payload)
            assert ref["sha256"] == hashlib.sha256(payload).hexdigest()
            assert (out_dir / ref["path"]).read_bytes() == payload

    def test_the_text_encoder_declares_both_storage_forms(self, sbv2_assembled) -> None:
        """The dtype key is always required, a uniform form (ADR 0041 §3 — the 2-form parsing was
        removed).

        The order is exactly the declaration order of {@link SBV2_WEIGHTS} (the same role is not
        listed in different orders in the manifest's weights section and quant section — the
        counterpart of the MUST of `complete_quant_weights`).
        """
        _, manifest = sbv2_assembled
        entry = _sbv2_model(manifest)["weights"]["text_encoder"]
        assert list(entry) == ["i8", "i4"]
        for label in ("i8", "i4"):
            paths = [ref["path"] for ref in entry[label]["container"]["parts"]]
            expected = part_paths(f"model.{label}.krm", len(paths))
            assert [path.rsplit("/", 1)[-1] for path in paths] == expected

    def test_every_dtype_seat_declares_an_ordered_part_list(self, sbv2_assembled) -> None:
        """The part list of `karume/5` **always has 2 or more entries** (part 0 of the graph
        description + the following parts — the always-split form of container-v1 §8).

        This used to pin "always 1 entry" (an observation point from the time when the split rule
        applied only to slots and one container was shipped as one file). Now that splitting is
        always on, that invariant of the distribution itself is inverted, so the claim is inverted
        too — a declaration with 1 entry is a form where the writer did not split off the graph
        description's part, an asset that the part-list gate of `karume verify` should reject.

        The order is checked too (numbering starts at 1 with no gaps) because the loader reads **in
        list order** (the first is part 0) — even if the set matches, a broken order cannot be
        read.
        """
        _, manifest = sbv2_assembled
        for name, labels in _sbv2_model(manifest)["weights"].items():
            for label, entry in labels.items():
                paths = [ref["path"] for ref in entry["container"]["parts"]]
                assert len(paths) >= 2, (name, label)
                stem = paths[0].rsplit("/", 1)[0]
                base = f"model.{label}.krm"
                assert paths == [f"{stem}/{rel}" for rel in part_paths(base, len(paths))], (
                    name,
                    label,
                )

    def test_it_carries_the_quant_table(self, sbv2_assembled) -> None:
        _, manifest = sbv2_assembled
        model = _sbv2_model(manifest)
        assert sorted(model["quants"]) == sorted(SBV2_QUANTS)
        assert model["defaultQuant"] == SBV2_DEFAULT_QUANT
        assert model["defaultQuant"] in model["quants"]

    def test_every_quant_names_the_text_encoder_too(self, sbv2_assembled) -> None:
        """The single-file role that could not be written in `weights` in v1 is part of the
        complete mapping in v2 (§3).

        The quants that bake BERT as the i4 mix are **named** independently of the table to catch
        the case where an edit of `SBV2_QUANTS` silently moves an existing quant's text_encoder to
        another storage form (deriving it from the table would be vacuous).
        """
        bert_i4_quants = {"i8+bert4", "i4"}
        _, manifest = sbv2_assembled
        model = _sbv2_model(manifest)
        assert bert_i4_quants <= set(model["quants"])
        for name, quant in model["quants"].items():
            assert set(quant["weights"]) == set(SBV2_WEIGHTS), name
            assert quant["weights"]["text_encoder"] == ("i4" if name in bert_i4_quants else "i8"), (
                name
            )

    def test_the_bert4_quant_is_i8_with_only_the_text_encoder_swapped(self, sbv2_assembled) -> None:
        """`i8+bert4` means "the same configuration as `i8` with only BERT in i4" — pins that the
        difference is 1 slot.

        The session knobs are checked as well because if a change on **another axis**, such as
        `i8-a8`, slipped in while it still claims to be "a storage-only slot", what the
        listening-based ruling (perf-ledger Q-1) applies to would silently change.
        """
        _, manifest = sbv2_assembled
        quants = _sbv2_model(manifest)["quants"]
        base, variant = quants["i8"], quants["i8+bert4"]

        assert variant["session"] == base["session"]
        differing = {
            role for role in base["weights"] if base["weights"][role] != variant["weights"][role]
        }
        assert differing == {"text_encoder"}
        assert variant["weights"]["text_encoder"] == "i4"

    def test_the_voices_declare_all_three_storage_forms(self, sbv2_assembled) -> None:
        """`front` / `voice` have 3 slots, f16 / i8 / the i4 mix (in the declaration order of
        {@link SBV2_WEIGHTS})."""
        _, manifest = sbv2_assembled
        for role in ("front", "voice"):
            entry = _sbv2_model(manifest)["weights"][role]
            assert list(entry) == ["f16", "i8", "i4"], role
            for label in entry:
                paths = [ref["path"] for ref in entry[label]["container"]["parts"]]
                names = part_paths(f"model.{label}.krm", len(paths))
                assert [path.rsplit("/", 1)[-1] for path in paths] == names, role

    def test_the_i4_quant_takes_the_mixed_form_in_every_role(self, sbv2_assembled) -> None:
        """`i4` means "all 3 slots as the i4 mix, session stays `i8`" — pins that there is a single
        axis.

        The session is checked for the same reason as the `i8+bert4` gate — if activation
        quantization (the `-a8` axis) slipped in, what the listening-based ruling (perf-ledger Q-1 /
        research 2026-08-19 §6) applies to would silently change.
        """
        _, manifest = sbv2_assembled
        quants = _sbv2_model(manifest)["quants"]
        base, variant = quants["i8"], quants["i4"]

        assert variant["session"] == base["session"]
        assert set(variant["weights"].values()) == {"i4"}
        assert set(variant["weights"]) == set(base["weights"])


class TestSbv2QuantNaming:
    """Quant names follow ADR 0074's grammar (`<storage>[+<component><bits>]…[-<knob>]…`) and the
    display fields (ADR 0075)."""

    def test_every_seat_starts_from_a_storage_dtype_of_the_vocabulary(self) -> None:
        """`<storage>` is the vocabulary of the asset headers (ADR 0074 decision 2) — the second
        vocabulary `w8` / `w4` was retired."""
        for name in SBV2_QUANTS:
            assert name.split("+")[0].split("-")[0] in ("f32", "f16", "i8", "i4"), name

    def test_the_component_override_token_spells_the_text_encoder(self) -> None:
        """The recipe defines the abbreviations (ADR 0074 decision 4) — the mapping also appears
        in the card."""
        assert SBV2_QUANT_ABBREVIATIONS == {"bert": "text_encoder"}
        assert set(SBV2_QUANT_ABBREVIATIONS.values()) <= set(SBV2_WEIGHTS)

    def test_the_f16_seat_names_the_int8_text_encoder_it_actually_ships(self) -> None:
        """Matches what is actually shipped (ADR 0074 decision 6) — the f16 slot's text_encoder is
        i8 (there is no f16 series)."""
        assert SBV2_QUANTS["f16+bert8"]["weights"]["text_encoder"] == "i8"
        assert "f16" not in SBV2_QUANTS

    def test_every_seat_carries_a_label_and_a_description_within_the_limits(self) -> None:
        for name, quant in SBV2_QUANTS.items():
            assert quant["label"] and quant["description"], name
            assert_quant_presentation(f"sbv2.quants.{name}", quant)


class TestSbv2VerifyDist:
    def test_it_passes_on_a_freshly_assembled_tree(self, sbv2_assembled) -> None:
        out_dir, _ = sbv2_assembled
        expected = _in_subtree(SBV2_DEFAULT_MODEL, _placed_paths())
        assert sorted(verify_dist(out_dir)) == sorted(expected)

    def test_it_passes_without_a_model_card(self, sbv2_assembled) -> None:
        """Verification runs **before the card is written**. Its absence does not trip the check
        for undeclared files."""
        out_dir, _ = sbv2_assembled
        assert not (out_dir / MODEL_CARD_FILENAME).exists()
        assert verify_dist(out_dir)

    def test_it_catches_an_undeclared_file(self, sbv2_assembled) -> None:
        out_dir, _ = sbv2_assembled
        (out_dir / SBV2_DEFAULT_MODEL / "voice" / "io.p2.safetensors").write_bytes(b"stale")
        with pytest.raises(DistError, match="Files the manifest does not declare"):
            verify_dist(out_dir)


class TestSbv2Family:
    """ "Several speakers in one repository" like the 4 JVNV models — text_encoder / tokenizer
    become shared."""

    @pytest.fixture
    def family(self, tmp_path: Path) -> tuple[Path, dict]:
        first = _build_sbv2_sources(tmp_path, model="F1")
        second = _build_sbv2_sources(tmp_path, model="F2", offset=7.0)
        out_dir = tmp_path / "models" / "karume-sbv2-family"
        manifest = assemble_family(
            [sbv2_plan(first, _SBV2_KNOBS, "F1"), sbv2_plan(second, _SBV2_KNOBS, "F2")],
            out_dir,
            "F1",
        )
        return out_dir, manifest

    def test_the_text_encoder_is_fetched_once_for_the_whole_family(self, family) -> None:
        """The 319MB DeBERTa duplicated per model is the real harm of v1 (ADR 0041 Context ②)."""
        _, manifest = family
        shared = [
            f"{SHARED_DIRNAME}/{rel}" for rel in part_paths(SBV2_OUTPUT_PATHS["text_encoder"])
        ]
        # What is folded is **the whole component** (the shard list moves under shared/ as is).
        for name, model in manifest["models"].items():
            refs = model["weights"]["text_encoder"]["i8"]["container"]["parts"]
            assert [ref["path"] for ref in refs] == shared, name

    def test_the_tables_that_differ_stay_per_model(self, family) -> None:
        out_dir, manifest = family
        first = manifest["models"]["F1"]["assets"]["style_vectors"]["path"]
        second = manifest["models"]["F2"]["assets"]["style_vectors"]["path"]
        assert first == f"F1/{SBV2_OUTPUT_PATHS['style_vectors']}"
        assert second == f"F2/{SBV2_OUTPUT_PATHS['style_vectors']}"
        assert np.array_equal(load_file(str(out_dir / first))[SBV2_STYLE_KEY], _style_table(3))
        assert np.array_equal(
            load_file(str(out_dir / second))[SBV2_STYLE_KEY], _style_table(3, 7.0)
        )

    def test_the_shared_assets_are_the_ones_both_models_produce_identically(self, family) -> None:
        _, manifest = family
        shared = {
            name
            for name, ref in manifest["models"]["F1"]["assets"].items()
            if ref["path"].startswith(f"{SHARED_DIRNAME}/")
        }
        assert shared == {"tokenizer", "symbols"}

    def test_the_assembled_family_verifies(self, family) -> None:
        out_dir, _ = family
        assert verify_dist(out_dir)


class TestSbv2Sources:
    def test_it_derives_the_two_series_from_the_model_name(self, tmp_path: Path) -> None:
        sources = sbv2_sources(tmp_path, "FN7")
        assert sources.series_f16 == tmp_path / "sbv2-FN7-f16"
        assert sources.series_i8 == tmp_path / "sbv2-FN7-i8"
        assert sources.model.name == "FN7"
        # DeBERTa does not depend on the model name (a family keeps a single one in shared/).
        assert sources.text_encoder == tmp_path / "deberta-i8" / "sbv2-22layer"
        # A **separate series** per storage form (ADR 0019) — only the variant name is the same.
        assert sources.text_encoder_i4 == tmp_path / "deberta-i4" / "sbv2-22layer"

    def test_it_looks_outside_the_series_root_for_the_host_assets(self, tmp_path: Path) -> None:
        """The demo assets (directly under `outputs/`) and the real weights (`inputs/`) are not
        series — the spelling is in paths.py."""
        sources = sbv2_sources(tmp_path)
        assert tmp_path not in sources.demo.parents
        assert tmp_path not in sources.model.parents
        assert sources.demo.name == "sbv2-demo"
        assert sources.model.name == SBV2_DEFAULT_MODEL


@requires_sbv2_package
class TestSbv2KnobDefaults:
    """Where the default knobs come from (`style_bert_vits2.constants`) — in the distribution only
    the manifest holds knobs."""

    @staticmethod
    def _package_knobs() -> dict[str, Any]:
        from style_bert_vits2.constants import (
            DEFAULT_LENGTH,
            DEFAULT_NOISE,
            DEFAULT_NOISEW,
            DEFAULT_SDP_RATIO,
            DEFAULT_STYLE,
            DEFAULT_STYLE_WEIGHT,
        )

        return {
            "style": DEFAULT_STYLE,
            "styleWeight": DEFAULT_STYLE_WEIGHT,
            "sdpRatio": DEFAULT_SDP_RATIO,
            "noiseScale": DEFAULT_NOISE,
            "noiseScaleW": DEFAULT_NOISEW,
            "lengthScale": DEFAULT_LENGTH,
        }

    def test_it_pulls_every_knob_from_the_package(self) -> None:
        assert sbv2_knob_defaults() == self._package_knobs()


@requires_sbv2_package
class TestSbv2Cli:
    """The path of `karume dist --pipeline sbv2` (the default output location, the `--model` axis,
    families)."""

    @staticmethod
    def _sources(tmp_path: Path, model: str, offset: float = 0.0):
        knobs = TestSbv2KnobDefaults._package_knobs()
        symbols = json.dumps(
            {
                "bertHiddenFromEnd": _SBV2_BERT_FROM_END,
                "samplingRate": _SBV2_SAMPLING_RATE,
                "hopLength": _SBV2_HOP_LENGTH,
            },
            ensure_ascii=False,
        ).encode("utf-8")
        return knobs, _build_sbv2_sources(tmp_path, model=model, symbols=symbols, offset=offset)

    @staticmethod
    def _reroot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        import dist
        from sbv2 import distribution

        # `DIST_ROOT` is on the driver side that decides the default output location
        # (`dist.default_out_dir`), `MISC_ROOT` / `INPUTS_ROOT` on the recipe side that takes the
        # inputs from outside the series — bindings of different modules are swapped separately.
        monkeypatch.setattr(dist, "DIST_ROOT", tmp_path / "models")
        monkeypatch.setattr(distribution, "MISC_ROOT", tmp_path / "outputs" / "misc")
        monkeypatch.setattr(distribution, "INPUTS_ROOT", tmp_path / "inputs")

    def test_it_assembles_into_the_pipeline_default_directory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        knobs, sources = self._sources(tmp_path, SBV2_DEFAULT_MODEL)
        self._reroot(tmp_path, monkeypatch)

        main(["--pipeline", "sbv2", "--series", str(sources.series_f16.parent)])

        out_dir = tmp_path / "models" / sbv2_repo_name("jvnv")
        expected = _in_subtree(SBV2_DEFAULT_MODEL, _placed_paths())
        assert sorted(verify_dist(out_dir)) == sorted(expected)
        manifest = json.loads((out_dir / MANIFEST_FILENAME).read_text(encoding="utf-8"))
        defaults = _sbv2_model(manifest)["pipelineConfig"]["defaults"]
        assert defaults["style"] == knobs["style"]
        # The card is drawn from the manifest that passed verification (the style table appears in
        # the text as is).
        card = (out_dir / MODEL_CARD_FILENAME).read_text(encoding="utf-8")
        assert "pipeline_tag: text-to-speech" in card
        for style in _sbv2_model(manifest)["pipelineConfig"]["styles"]:
            assert f"| `{style}` |" in card

    def test_the_model_flag_moves_the_series_and_the_default_directory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _, sources = self._sources(tmp_path, "FN7")
        self._reroot(tmp_path, monkeypatch)

        main(
            [
                "--pipeline",
                "sbv2-fn",
                "--model",
                "FN7",
                "--series",
                str(sources.series_f16.parent),
            ]
        )

        # The default output location is the family's repository (no per-speaker repository is
        # created — W-RC5-2).
        out_dir = tmp_path / "models" / "karume-sbv2-fn"
        manifest = json.loads((out_dir / MANIFEST_FILENAME).read_text(encoding="utf-8"))
        assert list(manifest["models"]) == ["FN7"]
        assert verify_dist(out_dir)

    def test_repeating_the_model_flag_assembles_one_family_repository(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _, first = self._sources(tmp_path, "F1")
        _, second = self._sources(tmp_path, "F2", offset=3.0)
        assert second.demo == first.demo  # the demo assets share one location
        self._reroot(tmp_path, monkeypatch)
        out_dir = tmp_path / "models" / "karume-sbv2-jvnv"

        main(
            [
                "--pipeline",
                "sbv2",
                "--card-profile",
                "jvnv",
                "--model",
                "F1",
                "--model",
                "F2",
                "--series",
                str(first.series_f16.parent),
                "--out",
                str(out_dir),
                "--repo",
                "hdae/karume-sbv2-jvnv",
            ]
        )

        manifest = json.loads((out_dir / MANIFEST_FILENAME).read_text(encoding="utf-8"))
        assert list(manifest["models"]) == ["F1", "F2"]
        assert manifest["defaultModel"] == "F1"
        assert verify_dist(out_dir)
        card = (out_dir / MODEL_CARD_FILENAME).read_text(encoding="utf-8")
        assert "## Model: F1" in card
        assert "## Model: F2" in card
        assert '  repo: "hdae/karume-sbv2-jvnv",' in card
        # The attribution is the chosen family's (not a single word of FN's source is mixed in).
        assert "base_model:\n  - litagin/style_bert_vits2_jvnv\n" in card
        assert "license: cc-by-sa-4.0" in card
        assert "rufflet17" not in card

    @pytest.mark.parametrize(("pipeline", "model"), [("sbv2", "FN4"), ("sbv2-fn", "F1")])
    def test_it_refuses_a_voice_of_the_other_family_before_assembling(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pipeline: str, model: str
    ) -> None:
        """The license text and the attribution at the repository root are decided by the family —
        a voice of another family fails **before assembling** (no leftovers). Letting it through
        would produce a distribution in which an FN voice claims the CC BY-SA license text.
        """
        _, sources = self._sources(tmp_path, model)
        self._reroot(tmp_path, monkeypatch)

        with pytest.raises(DistError, match="does not belong in the repository of voice family"):
            main(
                [
                    "--pipeline",
                    pipeline,
                    "--model",
                    model,
                    "--series",
                    str(sources.series_f16.parent),
                ]
            )

        assert not (tmp_path / "models").exists()


@requires_sbv2_package
class TestSbv2LegalText:
    """The CC BY-SA 4.0 license text and the notice of attribution and changes at the distribution
    repository's root (ADR 0092 decision 7)."""

    @staticmethod
    def _assemble(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pipeline: str, model: str):
        _, sources = TestSbv2Cli._sources(tmp_path, model)
        TestSbv2Cli._reroot(tmp_path, monkeypatch)
        out_dir = tmp_path / "dist"
        main(
            [
                "--pipeline",
                pipeline,
                "--model",
                model,
                "--series",
                str(sources.series_f16.parent),
                "--out",
                str(out_dir),
            ]
        )
        return out_dir

    def test_the_jvnv_repository_ships_the_license_text_byte_identical(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """§3(a)(1)(C) — what is included is **a copy of this license text** (neither a summary nor
        a reformatting).

        Reformatting or a newline conversion anywhere on the assembly path moves a byte, but it
        stays valid as prose, so the other gates let it through.
        """
        out_dir = self._assemble(tmp_path, monkeypatch, "sbv2", "F1")
        assert (out_dir / "LICENSE.md").read_bytes() == CC_BY_SA_4_0_PATH.read_bytes()

    def test_the_legal_text_passes_the_undeclared_file_gate(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The 2 files the manifest does not declare pass `verify_dist` as the `LEGAL_PATHS`
        slots."""
        out_dir = self._assemble(tmp_path, monkeypatch, "sbv2", "F1")

        assert sorted(LEGAL_PATHS) == ["LICENSE.md", "NOTICE.md"]
        assert all((out_dir / name).is_file() for name in LEGAL_PATHS)
        assert sorted(verify_dist(out_dir)) == sorted(_in_subtree("F1", _placed_paths()))

    def test_the_notice_credits_both_works_and_states_the_changes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """§3(a)(1)(A)(B) and §3(b) — the 2 sources, the notice of changes and the same-license
        inheritance."""
        out_dir = self._assemble(tmp_path, monkeypatch, "sbv2", "F1")
        notice = (out_dir / "NOTICE.md").read_text(encoding="utf-8")

        assert notice == SBV2_JVNV_NOTICE_MARKDOWN
        assert "https://huggingface.co/litagin/style_bert_vits2_jvnv" in notice
        assert f"https://huggingface.co/{SBV2_TEXT_ENCODER_MODEL}" in notice
        assert SBV2_JVNV_CORPUS_PAGE in notice
        # The notice of changes: the truncation to 22 layers (the only change that alters the text
        # encoder's structure) and quantization.
        assert f"first {SBV2_TEXT_ENCODER_LAYERS} encoder layers" in notice
        assert "quantized" in notice
        assert "licensed under CC BY-SA 4.0 as well" in notice

    def test_the_notice_holds_for_any_subset_of_the_voices(self) -> None:
        """MUST: the voices are not named one by one — one set of `root_files` also goes with any
        subset given by `--model`, so naming them would claim voices that were not assembled (the
        card names the breakdown from the manifest).
        """
        for source_dir in SBV2_JVNV_PROFILE.source_dirs:
            assert source_dir not in SBV2_JVNV_NOTICE_MARKDOWN, source_dir
        assert "(`README.md`) lists which ones this repository holds" in SBV2_JVNV_NOTICE_MARKDOWN

    def test_the_fn_repository_carries_no_legal_text(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The FN line has no license text that can be bundled (only Booth's distribution terms) —
        it must not claim CC BY-SA."""
        out_dir = self._assemble(tmp_path, monkeypatch, "sbv2-fn", "FN4")

        assert not (out_dir / "LICENSE.md").exists()
        assert not (out_dir / "NOTICE.md").exists()
        assert verify_dist(out_dir)


class TestSbv2CardProfile:
    """Attribution profiles — each family's Pipeline holds **exactly one**.

    The engine-side rule (with one it passes when omitted / with 2 or more it must be explicit) is
    held by `TestCardProfile` in `tools/exporter/tests/test_dist.py` with a synthetic pipeline.
    This checks that **each slot's attribution belongs to its own family** and that the names of
    other families are not accepted.
    """

    @pytest.mark.parametrize(("pipeline", "family"), [(PIPELINE, "jvnv"), (FN_PIPELINE, "fn")])
    def test_each_family_resolves_to_its_own_attribution_without_a_choice(
        self, pipeline, family: str
    ) -> None:
        assert sorted(pipeline.card_profiles) == [family]
        assert resolve_card_renderer(pipeline, None) is pipeline.card_profiles[family]
        assert resolve_card_renderer(pipeline, family) is pipeline.card_profiles[family]

    @pytest.mark.parametrize(("pipeline", "other"), [(PIPELINE, "fn"), (FN_PIPELINE, "jvnv")])
    def test_it_refuses_the_other_familys_attribution(self, pipeline, other: str) -> None:
        """`--pipeline sbv2 --card-profile fn` would draw FN's attribution on JVNV voices — it
        fails."""
        with pytest.raises(DistError, match=f"'{other}' does not exist"):
            resolve_card_renderer(pipeline, other)

    def test_the_two_families_render_different_cards(self) -> None:
        """Bundled the wrong way, the 2 families would draw the same card."""
        assert PIPELINE.card_profiles["jvnv"] is not FN_PIPELINE.card_profiles["fn"]


class TestSbv2PipelineEntry:
    """What the one line of `--pipeline sbv2` / `sbv2-fn` points to (the slot the driver merges into
    core's table)."""

    def test_it_carries_the_default_model_and_the_repo_name(self) -> None:
        assert PIPELINE.default_model == SBV2_DEFAULT_MODEL
        assert PIPELINE.repo_name(SBV2_DEFAULT_MODEL) == "karume-sbv2-jvnv"

    @pytest.mark.parametrize(
        ("pipeline", "models", "repo"),
        [
            (PIPELINE, ("F1", "F2", "M1", "M2"), "karume-sbv2-jvnv"),
            (FN_PIPELINE, ("FN4", "FN7"), "karume-sbv2-fn"),
        ],
    )
    def test_every_voice_of_a_family_names_the_family_repository(
        self, pipeline, models: tuple[str, ...], repo: str
    ) -> None:
        """MUST: repositories are per family (W-RC5-2) — deriving one from the speaker name would
        make a pass without `--repo` produce a distribution whose card Usage names a repository
        that does not exist (`karume-sbv2-F1`).
        """
        assert {pipeline.repo_name(model) for model in models} == {repo}

    def test_each_default_model_belongs_to_its_own_family(self) -> None:
        """If the default model were a voice of another family, a pass that omits `--model` would
        always fail at the gate."""
        assert PIPELINE.default_model.startswith(SBV2_FAMILY_DIRS["jvnv"])
        assert FN_PIPELINE.default_model.startswith(SBV2_FAMILY_DIRS["fn"])

    def test_its_card_refuses_another_pipelines_manifest(self) -> None:
        manifest = {"models": {"m": {"pipeline": "sbv2/0"}}}
        for render_card in PIPELINE.card_profiles.values():
            with pytest.raises(ValueError):
                render_card(manifest, "hdae/x")
