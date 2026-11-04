"""Anima's distribution recipe (`anima.distribution`) — assembling a series into the
distribution layout.

No real assets are used. What goes into the weights slots is a **valid minimal IR container** of
a few KB (`ir_fixtures` — {@link _weights_container}); only the extras slots such as the rope
base table, and the cases that check a rejection by a check, keep the old fake assets.

Since manifest v2 (`karume/2` — ADR 0041) the layout inside a repository is uniformly "a subtree
per model + `shared/`", so every expected path starts with `<model name>/…`.

The generic engine side (the layer observable from a synthetic plan alone — the size limits, the
complete quant mapping, the staging/swap invariants) stays in `tools/exporter/tests/test_dist.py`.
Only **the cases that cannot be built without going through Anima's recipe** live here.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from container_series import (
    part_paths,
    placed_paths,
    read_component,
    replace_component,
    write_component,
)
from ir_fixtures import ir_container

import dist
from _shared.container_read import read_asset
from anima.card import ATTRIBUTION_NOTICE, NVIDIA_NOTICE, RIGHTS_GRANT_STATEMENT
from anima.distribution import (
    ANIMA_AESTHETIC_MODEL_NAME,
    ANIMA_BASE_MODEL_NAME,
    ANIMA_DEFAULT_QUANT,
    ANIMA_MODELS,
    ANIMA_QUANT_ABBREVIATIONS,
    ANIMA_QUANTS,
    ANIMA_STORAGE_FORBIDDEN,
    ANIMA_TURBO_MODEL_NAME,
    ANIMA_TURBO_PIPELINE_CONFIG,
    ANIMA_WEIGHTS,
    CALIB_PROVENANCE_FILE,
    EXTRA_MODELS,
    EXTRA_NOTICE_MARKDOWN,
    EXTRA_PIPELINE,
    LICENSE_SOURCE_PATH,
    LORA_PROVENANCE_FILE,
    NVIDIA_LICENSE_SOURCE_PATH,
    OFFICIAL_MODELS,
    OFFICIAL_NOTICE_MARKDOWN,
    OFFICIAL_PIPELINE,
    OFFICIAL_REPO_NAME,
    OUTPUT_PATHS,
    ROPE_BASE_ASSET,
    ROPE_BASE_ROLE,
    STORAGE_REQUIREMENTS,
    AnimaModel,
    AnimaSources,
    anima_dist_plan,
    anima_model,
    anima_plan,
    anima_quants,
    anima_sources,
    anima_weights,
    notice_markdown,
    seat_base_storage,
    text_stack_role,
)
from dist import main
from karume.artifacts import STAGING_SUFFIX
from karume.container import AssetInput, container_parts
from karume.dist import (
    MANIFEST_FILENAME,
    MANIFEST_FORMAT,
    MODEL_CARD_FILENAME,
    SHARED_DIRNAME,
    Artifact,
    DistError,
    assemble_family,
    assert_model_name,
    assert_quant_presentation,
    complete_quant_weights,
    materialize,
    resolve_card_renderer,
    verify_dist,
)

#: The rope base table bundled into the transformer container (asset `rope_base` — ADR 0109
#: decision 4). Like the real one its content is the bytes of one safetensors file, but the
#: container does not interpret it, so **any bytes will do**.
_ROPE_BASE = b"rope-base-table"


def _rope_asset(payload: bytes = _ROPE_BASE) -> dict[str, AssetInput]:
    """The asset declaration of one `rope_base` (the source of truth for the name and role is
    `anima.distribution`)."""
    return {ROPE_BASE_ASSET: AssetInput(ROPE_BASE_ROLE, len(payload), payload)}


def _weights_container(
    role: str, storage: str, *, named: str, rope: bytes | None = None
) -> list[bytes]:
    """**A valid container** for a weights slot (different bytes per role).

    The assembly opens the input containers and checks them against every rule of the
    declaration (`karume.dist.assert_weight_components_verified`), so a weights slot must hold
    the real thing. The storage vocabulary has the same shape as the real one (only eligible
    weights are compressed; bias / constants / scales are f32; i4 is a mix of i4 + i8 + f32), so
    the absence check of {@link ANIMA_STORAGE_FORBIDDEN} applies to this shape too.

    `rope` is an asset only the transformer slot holds (passing it adds one dedicated part).
    """
    return ir_container(
        mark=role,
        # The fake series is **named by its component** too (container graph name = the
        # manifest's weights key, MUST — container-v1 §2.1). `mark` makes the bytes differ; its
        # spelling does not match the component name.
        named=named,
        storage=storage,
        assets={} if rope is None else _rope_asset(rope),
    )


#: The contents of the fake assets (different bytes per role — a mix-up shows in the hashes).
#: The 5 model roles carry in their binding tables the storage vocabulary that
#: `STORAGE_REQUIREMENTS` requires. The 3 transformer roles bundle the rope base table as an
#: asset, so they have one more part ({@link _TRANSFORMER_PARTS}).
_PAYLOADS = {
    "text_encoder": _weights_container("text-encoder", "f16", named="text_encoder"),
    "text_conditioner": _weights_container("text-conditioner", "f16", named="text_conditioner"),
    "transformer_f16": _weights_container(
        "transformer-f16", "f16", named="transformer", rope=_ROPE_BASE
    ),
    "transformer_i8": _weights_container(
        "transformer-i8", "i8", named="transformer", rope=_ROPE_BASE
    ),
    "transformer_i4": _weights_container(
        "transformer-i4", "i4", named="transformer", rope=_ROPE_BASE
    ),
    "vae_decoder": _weights_container("vae-decoder", "f16", named="vae_decoder"),
    "tokenizer": b'{"qwen2": true}',
    "tokenizer_2": b'{"t5": true}',
    # The 4 roles of the faithful preset `bf16` (ADR 0131 — only the base model has them).
    "text_encoder_bf16": _weights_container("text-encoder-bf16", "bf16", named="text_encoder"),
    "text_conditioner_bf16": _weights_container(
        "text-conditioner-bf16", "bf16", named="text_conditioner"
    ),
    "transformer_bf16": _weights_container(
        "transformer-bf16", "bf16", named="transformer", rope=_ROPE_BASE
    ),
    "vae_decoder_bf16": _weights_container("vae-decoder-bf16", "bf16", named="vae_decoder"),
}

#: The part count of a transformer slot (the asset `rope_base` adds one dedicated part).
_TRANSFORMER_PARTS = len(_PAYLOADS["transformer_f16"])

#: Role name → part count (overrides for {@link placed_paths} / {@link part_paths}).
_PART_TOTALS: Mapping[str, int] = {
    f"transformer_{storage}": _TRANSFORMER_PARTS for storage in ("f16", "i8", "i4", "bf16")
}

#: The 3 components of the text stack ({@link text_stack_role} turns them into role names per
#: storage).
_TEXT_STACK_COMPONENTS = ("text_encoder", "text_conditioner", "vae_decoder")


def _write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


#: A fake "baked record" (a sha256 that matches no real LoRA). **No model bakes in a LoRA any
#: more** (2026-09-01 — baking went away when Turbo became an official checkpoint), so this
#: record is written only to build the "something present that must not be" side.
FAKE_LORA_SHA256 = "9" * 64


def _bake_lora_record(
    series: Path, sha256: str = FAKE_LORA_SHA256, component: str = "transformer"
) -> Path:
    """Put the attribution record the old fused series had into one series (the same shape as the
    real one).

    The series of the old `anima-turbo` (LoRA baked in) still exist in `outputs/series/`, so
    plugging one into a new slot by mistake **can really happen**. Fused and plain assets do not
    differ in shape by a single byte, so only the check for the record's **absence** catches it.
    `component` is which of the writer's 2 `LORA_PREFIXES` targets (transformer /
    text_conditioner) it was baked into.
    """
    path = series / component / LORA_PROVENANCE_FILE
    _write(
        path,
        json.dumps({"file": "anima-turbo-lora-v0.2.safetensors", "sha256": sha256}).encode("utf-8"),
    )
    return path


def _calib_record(method: str, model: str) -> bytes:
    """The record of calibration conditions that `anima/export.py` leaves in an i4 series (the same
    shape as the real one).

    step / CFG are **per model** (`anima.calib.calib_conditions` derives them from
    `pipeline_config`), so the stand-in takes them from the same single place — copying the
    values would make the check of "which model's conditions it was baked with" untestable (a
    series baked with the wrong `--model` could no longer be built here either).
    """
    defaults = anima_model(model).pipeline_config["defaults"]
    return json.dumps(
        {
            "method": method,
            "group_size": 32,
            "grid": "rtn",
            "prompts": 4,
            "resolution": 512,
            "steps": int(defaults["steps"]),
            "guidance": float(defaults["guidanceScale"]),
            "text_dtype": "f16",
        }
    ).encode("utf-8")


def _build_series(
    series_dir: Path,
    *,
    model: str = ANIMA_TURBO_MODEL_NAME,
    with_i4: bool = False,
    i8_rope: bytes | None = None,
    i4_rope: bytes | None = None,
    bf16_rope: bytes | None = None,
    mark: bytes = b"",
    calib_method: str = "gptq",
    calib_model: str | None = None,
) -> AnimaSources:
    """Reproduce the series layout (as in `outputs/series/`) with fake assets (including stray
    `io.*` files).

    `mark` is a difference mixed into the transformer series only — the axis for building
    "weights that differ per model" and "base assets identical across models" in a family
    assembly. `i8_rope` / `i4_rope` / `bf16_rope` are axes that move the rope base table away
    from the f16 series (each series can be varied independently — to check that the net covers
    every series). The text stack is written **once per storage label the model declares** (the
    base model has f16 + bf16). `calib_method` is the axis that changes the i4 series' rounding
    method (the default is the shippable `gptq`). `calib_model` is the axis that moves **only
    the calibration conditions** to another model's (a series baked with the wrong `--model` —
    weights and storage form correct, only the conditions differ).

    `with_i4` is the axis that adds **a 3rd storage series not in the accepted set** (the
    partner of {@link _i4_seat} — only for the i4 mechanism tests). Its spelling is
    `<model>-i4-dyn`, the same as `anima_sources`; this function builds the series and the spec
    declares the slot.

    **No model** gets a LoRA attribution record (the same state as the real ones since baking
    went away on 2026-09-01) — tests build the side with a record explicitly with
    {@link _bake_lora_record}.
    """
    sources = anima_sources(series_dir, model)
    if with_i4:
        sources = replace(
            sources,
            transformer={**sources.transformer, "i4": series_dir / f"{model}-i4-dyn"},
        )
    for storage, base in sources.base.items():
        text_series = {
            "text_encoder": base,
            "text_conditioner": sources.text_conditioner[storage],
            "vae_decoder": base,
        }
        for component, series in text_series.items():
            write_component(
                series / component / "model.krm", _PAYLOADS[text_stack_role(component, storage)]
            )
        # E2E fixtures that must not get into the distribution (the series really hold
        # these).
        _write(base / "text_encoder" / "io.t005.safetensors", b"io-fixture")
        _write(base / "vae_decoder" / "io.case0.safetensors", b"io-fixture")
    ropes = {
        "f16": _ROPE_BASE,
        "i8": i8_rope or _ROPE_BASE,
        "i4": i4_rope or _ROPE_BASE,
        "bf16": bf16_rope or _ROPE_BASE,
    }
    for storage, series in sources.transformer.items():
        role = f"transformer_{storage}"
        rope = ropes[storage]
        payload = (
            _PAYLOADS[role]
            if not mark and rope == _ROPE_BASE
            else _weights_container(
                f"{role}{mark.decode('utf-8')}" if mark else role,
                storage,
                named="transformer",
                rope=rope,
            )
        )
        write_component(series / "transformer" / "model.krm", payload)
        _write(series / "transformer" / "io.s01024t0699.safetensors", b"io-fixture")
    # Only the i4 series has calibration conditions (f16 / i8 are not calibrated).
    if "i4" in sources.transformer:
        _write(
            sources.transformer["i4"] / "transformer" / CALIB_PROVENANCE_FILE,
            _calib_record(calib_method, model if calib_model is None else calib_model),
        )
    _write(sources.tokenizers / "qwen2-tokenizer.json", _PAYLOADS["tokenizer"])
    _write(sources.tokenizers / "t5-tokenizer.json", _PAYLOADS["tokenizer_2"])
    return sources


#: The 2 models the family-assembly fixture lists (the official repository's base model +
#: Aesthetic). **A pair that really lives in the same repository** is used — assembling a
#: combination the accepted sets put in different repositories would make the fixture and
#: {@link TestPipelineMembership} say different things.
FAMILY_MODELS = (ANIMA_BASE_MODEL_NAME, ANIMA_AESTHETIC_MODEL_NAME)


#: The shipping i4 slot **is gone from every model** (the user's ruling of 2026-09-01 — the last
#: slot, held by the old fused turbo, is not carried over into the official checkpoint either).
#: The i4 **mechanism** still remains in the code as a lever to bring it back (placing a 3rd
#: storage series, the cross-series comparison of the rope base table, the calibration-condition
#: check, the storage dtype requirements and prohibitions, deriving the quant entries), so a
#: spec with an injected slot guards just the mechanism.
#:
#: MUST: the injection is confined to the single entry `anima_plan(..., spec=…)` — not one line
#: of {@link ANIMA_MODELS} moves (moving it would make "the distribution's accepted set" sway for
#: an experiment, and the check of whether the slot exists (no i4 on any model) would only check
#: its own fixture). It is **the same entry** `anima.eval_dist` uses for visual evaluation, so if
#: the mechanism breaks, that side fails too.
def _i4_seat(model: str = ANIMA_TURBO_MODEL_NAME) -> AnimaModel:
    spec = anima_model(model)
    assert "i4" not in spec.storages, (
        "the i4 slot is back in the distribution's accepted set (no injection needed)"
    )
    return replace(spec, storages=(*spec.storages, "i4"))


def _assemble_anima(
    sources: AnimaSources,
    out_dir: Path,
    model: str = ANIMA_TURBO_MODEL_NAME,
    *,
    spec: AnimaModel | None = None,
) -> dict[str, Any]:
    """Test glue that runs a single-model assembly (plan → materialize) in one line.

    `spec` is the entry for replacing slots (when omitted it is looked up by model name = the
    shipping path) — only the tests of the i4 mechanism pass it, with "the spec with an i4 slot
    added" made by {@link _i4_seat}.
    """
    return assemble_family([anima_plan(sources, model, spec=spec)], out_dir, model)


def _in_subtree(
    model: str, paths: Iterable[str] | None = None, *, storages: tuple[str, ...] | None = None
) -> list[str]:
    """The expected paths inside a model subtree (the uniform layout of ADR 0041 §9).

    When omitted, it expands **only the storage forms the model declared** (a model without an
    i4 slot gets no i4 files) into the form they take in the distribution — the 5 weights roles
    become numbered parts (container-v1 §8), and the tokenizer (assets) stays one file. Pass
    `storages` only when looking at a tree assembled with an injected slot
    ({@link _i4_seat}).
    """
    if paths is None:
        storages = anima_model(model).storages if storages is None else storages
        # Role → its dtype label (weights slots only — assets slots always appear, whatever
        # the storage form).
        labels = {
            files.file: label for entry in ANIMA_WEIGHTS.values() for label, files in entry.items()
        }
        declared = {
            role: rel
            for role, rel in OUTPUT_PATHS.items()
            if role not in labels or labels[role] in storages
        }
        paths = placed_paths(declared, ANIMA_WEIGHTS, _PART_TOTALS)
    return [f"{model}/{rel}" for rel in paths]


def _present(out_dir: Path) -> list[str]:
    return sorted(str(path.relative_to(out_dir)) for path in out_dir.rglob("*") if path.is_file())


@pytest.fixture
def assembled(tmp_path: Path) -> tuple[Path, dict]:
    sources = _build_series(tmp_path / "series")
    out_dir = tmp_path / "models" / ANIMA_TURBO_MODEL_NAME
    manifest = _assemble_anima(sources, out_dir)
    return out_dir, manifest


#: The storage forms when assembled with an injected slot (the same derivation as
#: {@link _i4_seat} — the declared storage forms + i4).
I4_SEAT_STORAGES = (*anima_model(ANIMA_TURBO_MODEL_NAME).storages, "i4")


@pytest.fixture
def assembled_with_i4(tmp_path: Path) -> tuple[Path, dict]:
    """A distribution assembled with an injected i4 slot **that is not in the accepted set**
    (keeping the mechanism — {@link _i4_seat})."""
    sources = _build_series(tmp_path / "series", with_i4=True)
    out_dir = tmp_path / "models" / "i4-seat"
    manifest = _assemble_anima(sources, out_dir, spec=_i4_seat())
    return out_dir, manifest


class TestLayout:
    def test_it_places_every_declared_path_under_the_model_subtree(self, assembled) -> None:
        out_dir, _ = assembled
        assert _present(out_dir) == sorted(
            [*_in_subtree(ANIMA_TURBO_MODEL_NAME), MANIFEST_FILENAME]
        )

    def test_it_never_carries_io_fixtures_into_the_distribution(self, assembled) -> None:
        out_dir, _ = assembled
        assert list(out_dir.rglob("io.*")) == []

    def test_it_renames_the_two_transformer_series_into_dtype_files(self, assembled) -> None:
        out_dir, _ = assembled
        subtree = out_dir / ANIMA_TURBO_MODEL_NAME / "transformer"
        assert read_component(subtree / "model.f16.krm") == _PAYLOADS["transformer_f16"]
        assert read_component(subtree / "model.i8.krm") == _PAYLOADS["transformer_i8"]

    def test_no_shipped_model_places_an_i4_file(self, assembled) -> None:
        """The shipping i4 slot is gone from every model (ruling of 2026-09-01) — not a single file
        appears either."""
        out_dir, _ = assembled
        assert not (out_dir / ANIMA_TURBO_MODEL_NAME / OUTPUT_PATHS["transformer_i4"]).exists()
        assert not any("i4" in name for name in _present(out_dir))

    def test_an_injected_i4_seat_gets_its_own_dtype_file(self, assembled_with_i4) -> None:
        """i4 is the 3rd storage slot next to f16 / i8 (putting it on the same path would lose a
        slot).

        The slot itself left the distribution, but the **mechanism** that puts the 3rd one on
        its own path remains as a lever to bring it back (it is also the path `anima.eval_dist`
        uses right now), so it is guarded with an injected slot.
        """
        out_dir, _ = assembled_with_i4
        placed = out_dir / ANIMA_TURBO_MODEL_NAME / OUTPUT_PATHS["transformer_i4"]
        assert placed.name == "model.i4.krm"
        assert read_component(placed) == _PAYLOADS["transformer_i4"]
        assert _present(out_dir) == sorted(
            [
                *_in_subtree(ANIMA_TURBO_MODEL_NAME, storages=I4_SEAT_STORAGES),
                MANIFEST_FILENAME,
            ]
        )

    def test_a_single_model_repository_has_no_shared_directory(self, assembled) -> None:
        """`shared/` is a location that appears only when 2 or more models hold the same contents
        (ADR 0041 §5)."""
        out_dir, _ = assembled
        assert not (out_dir / SHARED_DIRNAME).exists()

    def test_it_reassembles_over_a_previous_run(self, tmp_path: Path) -> None:
        sources = _build_series(tmp_path / "series")
        out_dir = tmp_path / "models" / ANIMA_TURBO_MODEL_NAME
        _assemble_anima(sources, out_dir)
        _assemble_anima(sources, out_dir)  # does not fail even with existing links
        assert verify_dist(out_dir)

    def test_the_model_name_moves_the_whole_subtree(self, tmp_path: Path) -> None:
        """`--model` moves the subtree name, the series name and the manifest key with one word."""
        sources = _build_series(tmp_path / "series", model="anima-wai-v1.0")
        out_dir = tmp_path / "models" / "anima-wai-v1.0"
        manifest = _assemble_anima(sources, out_dir, "anima-wai-v1.0")
        assert list(manifest["models"]) == ["anima-wai-v1.0"]
        assert manifest["defaultModel"] == "anima-wai-v1.0"
        assert _present(out_dir) == sorted([*_in_subtree("anima-wai-v1.0"), MANIFEST_FILENAME])


class TestPlacementStrategy:
    def test_it_places_independent_copies(self, assembled) -> None:
        """The distribution holds no hard links (a self-contained snapshot independent of the
        series)."""
        out_dir, _ = assembled
        placed = out_dir / ANIMA_TURBO_MODEL_NAME / OUTPUT_PATHS["text_encoder"]
        assert read_component(placed) == _PAYLOADS["text_encoder"]
        # Being an independent copy applies to each part (no slot slips through by being
        # split).
        assert [part.stat().st_nlink for part in container_parts(placed)] == [1, 1, 1]

    def test_a_series_rewrite_does_not_reach_the_dist(self, tmp_path: Path) -> None:
        """Re-exporting a series (a truncating overwrite) does not reach an assembled distribution.

        The linking approach broke here — sharing the same inode, rewriting the series silently
        made the manifest's sha256 disagree with the files.
        """
        sources = _build_series(tmp_path / "series")
        out_dir = tmp_path / "models" / ANIMA_TURBO_MODEL_NAME
        _assemble_anima(sources, out_dir)
        # What is rewritten is the series' **part 0** (a re-export truncates and overwrites
        # each part).
        source = container_parts(sources.base["f16"] / "text_encoder" / "model.krm")[0]
        with source.open("wb") as handle:
            handle.write(b"rewritten-after-assembly")
        placed = out_dir / ANIMA_TURBO_MODEL_NAME / OUTPUT_PATHS["text_encoder"]
        assert read_component(placed) == _PAYLOADS["text_encoder"]

    def test_it_stops_when_an_input_is_missing(self, tmp_path: Path) -> None:
        sources = _build_series(tmp_path / "series")
        for part in container_parts(sources.base["f16"] / "vae_decoder" / "model.krm"):
            part.unlink()
        with pytest.raises(DistError, match="Assembly input is missing"):
            _assemble_anima(sources, tmp_path / "models" / ANIMA_TURBO_MODEL_NAME)


class TestRopeBase:
    def test_every_series_carries_the_same_table_inside_its_own_container(self, assembled) -> None:
        """The rope base table is a container asset (ADR 0109 decision 4) — one per series, with
        identical contents."""
        out_dir, _ = assembled
        placed = [
            out_dir / ANIMA_TURBO_MODEL_NAME / OUTPUT_PATHS[f"transformer_{storage}"]
            for storage in ("f16", "i8")
        ]
        assert [read_asset(path, ROPE_BASE_ASSET) for path in placed] == [_ROPE_BASE] * 2

    def test_a_series_without_the_asset_fails_in_the_assembly_vocabulary(
        self, tmp_path: Path
    ) -> None:
        """The operational accident of pointing the transformer slot at a series without `rope_base`
        (a mix-up with the static series).

        MUST: the diagnostic is a `DistError` (the assembly's vocabulary) — raising the reader's
        exception as is would make an assembly failure look like "a container-reading bug" (the
        same practice as `assert_storage` translating `ContainerFormatError`).
        """
        sources = _build_series(tmp_path / "series")
        replace_component(
            sources.transformer["i8"] / "transformer" / "model.krm",
            _weights_container("transformer-i8", "i8", named="transformer"),
        )

        with pytest.raises(DistError, match=r"no asset 'rope_base'"):
            _assemble_anima(sources, tmp_path / "models" / ANIMA_TURBO_MODEL_NAME)

    def test_it_refuses_to_pick_a_side_when_the_series_disagree(self, tmp_path: Path) -> None:
        sources = _build_series(tmp_path / "series", i8_rope=b"rope-base-table-DIFFERENT")
        out_dir = tmp_path / "models" / ANIMA_TURBO_MODEL_NAME
        with pytest.raises(DistError, match="is not byte-identical across the series"):
            _assemble_anima(sources, out_dir)
        # Having stopped, leave no partial distribution behind (later steps must not see an
        # output with only one side in it).
        assert not (out_dir / MANIFEST_FILENAME).exists()

    def test_the_check_reaches_a_third_series_too(self, tmp_path: Path) -> None:
        """The net covers **every declared series** (stopping at the 2nd would let the 3rd slip
        through).

        A quant of a series that slipped through would run with "a rope table of another
        geometry", pass loading and execution, and only the images would break, silently. The
        3rd is exercised by injecting the i4 slot that left the distribution
        ({@link _i4_seat}).
        """
        sources = _build_series(
            tmp_path / "series", with_i4=True, i4_rope=b"rope-base-table-DIFFERENT"
        )
        out_dir = tmp_path / "models" / "i4-seat"
        with pytest.raises(DistError, match="is not byte-identical across the series"):
            _assemble_anima(sources, out_dir, spec=_i4_seat())
        assert not (out_dir / MANIFEST_FILENAME).exists()


class TestLoraProvenance:
    """Confirms that **no model has a LoRA baked in** by the absence of a record in the series.

    With the restructuring of 2026-09-01 the official Turbo shipped as a checkpoint, and baking
    left the accepted set (every `lora_sha256` of `ANIMA_MODELS` is `None`). Fused and plain
    assets do not differ in shape by a single byte, so plugging an old fused series (still
    present in `outputs/series/`) into a new slot by mistake is caught by no other check — the
    check of "no record" is the only net.
    """

    def test_no_model_declares_a_baked_lora_any_more(self) -> None:
        """The fact on the accepted-set side (`None` = every model is on the side that actively
        checks for the absence of a record)."""
        assert [spec.lora_sha256 for spec in ANIMA_MODELS.values()] == [None] * len(ANIMA_MODELS)

    def test_it_stops_when_the_official_turbo_gets_a_series_with_a_baked_lora(
        self, tmp_path: Path
    ) -> None:
        """The mix-up of plugging a series of the old `anima-turbo` (LoRA baked in) into the
        official Turbo's slot."""
        sources = _build_series(tmp_path / "series")
        _bake_lora_record(sources.transformer["f16"])
        out_dir = tmp_path / "models" / ANIMA_TURBO_MODEL_NAME

        with pytest.raises(DistError, match="received a series that carries a"):
            _assemble_anima(sources, out_dir)

        # It is a check at the planning step, so not a single distribution file appears.
        assert not out_dir.exists()

    def test_it_stops_when_a_plain_model_gets_a_series_with_a_baked_lora(
        self, tmp_path: Path
    ) -> None:
        """The same single implementation applies to the base model's slot too (the net covers
        series, not models)."""
        sources = _build_series(tmp_path / "series", model=ANIMA_BASE_MODEL_NAME)
        _bake_lora_record(sources.transformer["f16"])
        out_dir = tmp_path / "models" / ANIMA_BASE_MODEL_NAME

        with pytest.raises(DistError, match="received a series that carries a"):
            _assemble_anima(sources, out_dir, ANIMA_BASE_MODEL_NAME)

        assert not out_dir.exists()

    def test_the_check_reaches_every_transformer_series(self, tmp_path: Path) -> None:
        """If the net stayed on the one f16 series, a record mixed into another storage slot would
        slip through.

        The old fused series come as 3 (f16 / i8 / i4), so swapping just one of them into the
        distribution's storage slots (now 2: f16 / i8) is a mix-up that can really happen. The
        enumeration is tied to the declarations so that the net widens automatically the day a
        slot is added.
        """
        for storage in ANIMA_MODELS[ANIMA_TURBO_MODEL_NAME].storages:
            sources = _build_series(tmp_path / f"series-{storage}")
            _bake_lora_record(sources.transformer[storage])

            with pytest.raises(DistError, match="received a series that carries a"):
                _assemble_anima(sources, tmp_path / "models" / storage)

    @pytest.mark.parametrize(
        "model",
        [ANIMA_TURBO_MODEL_NAME, ANIMA_AESTHETIC_MODEL_NAME],
        ids=["shared-anima-f16", "own-text-conditioner"],
    )
    def test_the_check_reaches_the_text_conditioner_series_too(
        self, tmp_path: Path, model: str
    ) -> None:
        """The writer also bakes the LoRA into text_conditioner and leaves a record
        (`LORA_PREFIXES`).

        With the net on transformer series only, plugging a baked conditioner series into a
        distribution slot would slip through, and nobody would check the MUST "no LoRA baked
        in" of `anima-f16` (shared). Checks that the same check applies to the slots of both the
        shared series and the model's own `<model>-f16`.
        """
        sources = _build_series(tmp_path / "series", model=model)
        _bake_lora_record(sources.text_conditioner["f16"], component="text_conditioner")
        out_dir = tmp_path / "models" / model

        with pytest.raises(DistError, match="received a series that carries a"):
            _assemble_anima(sources, out_dir, model)

        assert not out_dir.exists()

    # NOTE: the old test "the record is not carried into the distribution" was removed in this
    # iteration. A series with a record fails at the planning step through the check above, so
    # a record being absent from an assembled tree became **vacuously true** (that series-side
    # records do not reach the distribution is checked with real files by the calibration
    # record's test of the same name and the stray `io.*` test).


class TestModelVariants:
    """Facts that differ per model (the range of slots, where text_conditioner comes from, the
    default step / CFG).

    The 5 official variants (the versions of Turbo / base / Aesthetic) and the 2 third-party
    fine-tunes go through the same single implementation (ADR 0087) — the only differences are
    the declarations in {@link ANIMA_MODELS}; there is one assembly path.
    """

    def test_a_plain_model_excludes_the_i4_seat(self, tmp_path: Path) -> None:
        """The base model has **neither** the i4 slot nor its files (the exclusion ruling of 0.5.0 —
        the NOTE of `ANIMA_MODELS`).

        The reversal of the visual ruling (2026-08-24 skip shipping → 2026-08-25 exclude).
        "Has neither" is checked in **both** the quant table and the file list — with only one,
        either "the slot is gone but the files still ship (wasted GB)" or "the files are gone but
        the slot remains (selecting it gives 404)" would be green. That turbo keeps i4 is
        checked by the existing TestQuant checks (published under a separate ruling). The
        calibration check for the rtn method also keeps its coverage through the existing turbo
        check (the acceptance check of the rounding method) — the base model's i4 can no longer
        be configured, so the base-side calibration-check test was removed with this ruling.
        """
        sources = _build_series(tmp_path / "series", model=ANIMA_BASE_MODEL_NAME)
        out_dir = tmp_path / "models" / ANIMA_BASE_MODEL_NAME
        manifest = _assemble_anima(sources, out_dir, ANIMA_BASE_MODEL_NAME)

        entry = manifest["models"][ANIMA_BASE_MODEL_NAME]
        # bf16 is the faithful preset (ADR 0131 decision 3 — every model since the addendum of
        # 2026-10-17).
        assert sorted(entry["weights"]["transformer"]) == ["bf16", "f16", "i8"]
        assert "f16+dit4" not in entry["quants"]
        assert "f16+dit4-attn8-s16" not in entry["quants"]
        assert not (out_dir / ANIMA_BASE_MODEL_NAME / OUTPUT_PATHS["transformer_i4"]).exists()

    def test_the_official_aesthetic_carries_the_same_three_storages(self, tmp_path: Path) -> None:
        """Aesthetic also has the 3 slots f16 / i8 / bf16 (its default is the visual ruling's 30
        steps / CFG 4)."""
        sources = _build_series(tmp_path / "series", model=ANIMA_AESTHETIC_MODEL_NAME)
        out_dir = tmp_path / "models" / ANIMA_AESTHETIC_MODEL_NAME
        manifest = _assemble_anima(sources, out_dir, ANIMA_AESTHETIC_MODEL_NAME)

        entry = manifest["models"][ANIMA_AESTHETIC_MODEL_NAME]
        assert sorted(entry["weights"]["transformer"]) == ["bf16", "f16", "i8"]
        defaults = entry["pipelineConfig"]["defaults"]
        assert (defaults["steps"], defaults["guidanceScale"]) == (30, 4)

    def test_no_shipped_model_declares_an_i4_seat(self) -> None:
        """MUST: the shipping i4 slot is gone **from every model** (the user's ruling of 2026-09-01
        — ADR 0087).

        The ruling also does not carry over the last slot, held only by the old fused turbo,
        into the official checkpoint. Which slots exist is derived from the declared storage
        forms (`anima_quants` / `anima_weights`), so restoring even one model here moves the
        quant table, the files and the int4 mention in NOTICE together — conversely, this one
        line on the declaration side is the only source of truth for the slot.
        """
        assert [name for name, spec in ANIMA_MODELS.items() if "i4" in spec.storages] == []
        for name, spec in ANIMA_MODELS.items():
            # The faithful preset `bf16` is on every model (ADR 0131 decision 3, addendum
            # 2026-10-17).
            assert spec.storages == ("f16", "i8", "bf16"), name

    def test_the_official_variants_split_on_the_text_conditioner(self, tmp_path: Path) -> None:
        """Only the f16 of the variants measured bit-identical shares the text_conditioner
        (2026-09-01).

        Turbo is bit-identical to base after f16 rounding, so its f16 reads the shared series;
        Aesthetic was measured to keep differences in 64 tensors, so it has its own — mixing them
        up would run with **another model's text conditioning**, pass loading and execution, and
        only the images would drift, silently. bf16 keeps upstream's bits, so turbo has its own
        too (60 / 118 tensors differ from base — measured 2026-10-17).
        """
        turbo = anima_sources(tmp_path / "series", ANIMA_TURBO_MODEL_NAME)
        aesthetic = anima_sources(tmp_path / "series", ANIMA_AESTHETIC_MODEL_NAME)

        assert turbo.text_conditioner["f16"] == turbo.base["f16"]
        assert turbo.text_conditioner["bf16"] != turbo.base["bf16"]
        assert turbo.text_conditioner["bf16"].name == f"{ANIMA_TURBO_MODEL_NAME}-bf16"
        assert aesthetic.text_conditioner != aesthetic.base
        assert aesthetic.text_conditioner["f16"].name == f"{ANIMA_AESTHETIC_MODEL_NAME}-f16"
        assert aesthetic.text_conditioner["bf16"].name == f"{ANIMA_AESTHETIC_MODEL_NAME}-bf16"

    def test_a_fine_tune_reads_its_own_text_conditioner(self, tmp_path: Path) -> None:
        """A third-party fine-tune has retrained llm_adapter too, so it must not read the shared
        series."""
        base = anima_sources(tmp_path / "series", ANIMA_BASE_MODEL_NAME)
        wai = anima_sources(tmp_path / "series", "anima-wai-v1.0")

        assert base.text_conditioner == base.base
        assert wai.text_conditioner != wai.base
        assert wai.text_conditioner["f16"].name == "anima-wai-v1.0-f16"
        assert wai.text_conditioner["bf16"].name == "anima-wai-v1.0-bf16"

    def test_the_default_quant_stays_the_same_seat(self, tmp_path: Path) -> None:
        """The default entry matches turbo's (changing the model does not change what the default
        means for users)."""
        sources = _build_series(tmp_path / "series", model=ANIMA_BASE_MODEL_NAME)
        manifest = _assemble_anima(sources, tmp_path / "out", ANIMA_BASE_MODEL_NAME)

        assert manifest["models"][ANIMA_BASE_MODEL_NAME]["defaultQuant"] == ANIMA_DEFAULT_QUANT

    def test_a_plain_model_defaults_to_many_steps_with_guidance(self, tmp_path: Path) -> None:
        """The default uses CFG — the negative prompt takes effect only on this path."""
        sources = _build_series(tmp_path / "series", model=ANIMA_BASE_MODEL_NAME)
        manifest = _assemble_anima(sources, tmp_path / "out", ANIMA_BASE_MODEL_NAME)

        defaults = manifest["models"][ANIMA_BASE_MODEL_NAME]["pipelineConfig"]["defaults"]
        assert defaults["guidanceScale"] != 1
        assert defaults["steps"] > 8
        assert defaults["negativePrompt"]

    def test_the_default_model_is_the_official_turbo(self, tmp_path: Path) -> None:
        """The official repository's default is Turbo (the 2026-09-01 ruling, following upstream's
        README "start with Turbo").

        The default moves together with `pipelineConfig` (8 steps / CFG 1 = the negative prompt
        has no effect), so the default model and the default contents are pinned in **one
        test**.
        """
        assert OFFICIAL_PIPELINE.default_model == ANIMA_TURBO_MODEL_NAME
        assert OFFICIAL_MODELS[0] == ANIMA_TURBO_MODEL_NAME

        sources = _build_series(tmp_path / "series")
        manifest = _assemble_anima(sources, tmp_path / "out")

        defaults = manifest["models"][ANIMA_TURBO_MODEL_NAME]["pipelineConfig"]["defaults"]
        assert (defaults["steps"], defaults["guidanceScale"]) == (8, 1)


class TestPipelineMembership:
    """The modification notice at the repository root is one set fixed on the Pipeline — it must
    not be possible to assemble with the wrong one.

    The split axis is **official / fine-tune** (ruling of 2026-09-01 — ADR 0087). The official
    repository has only CircleStone's single license; the fine-tune repository layers the
    permissions of each source page on top, so both the notice and the origins section differ.
    """

    def test_the_official_pipeline_refuses_a_community_fine_tune(self, tmp_path: Path) -> None:
        with pytest.raises(DistError, match="does not belong in this pipeline's repository"):
            anima_dist_plan(tmp_path / "series", "anima-wai-v1.0", OFFICIAL_MODELS)

    def test_the_extra_pipeline_refuses_an_official_model(self, tmp_path: Path) -> None:
        with pytest.raises(DistError, match="does not belong in this pipeline's repository"):
            anima_dist_plan(tmp_path / "series", ANIMA_TURBO_MODEL_NAME, EXTRA_MODELS)

    def test_the_two_lists_partition_the_accepted_models(self) -> None:
        """MUST: every model is in **exactly one** of the two.

        An overlap would ship the same model under 2 notices; a gap would silently create a
        model that is in the accepted set yet cannot be assembled from any repository (not
        readable from the choices of `--pipeline`).
        """
        official, extra = set(OFFICIAL_MODELS), set(EXTRA_MODELS)

        assert official | extra == set(ANIMA_MODELS)
        assert official & extra == set()

    def test_the_two_repositories_declare_different_modifications(self) -> None:
        """If the notices were folded into one (= the same text), one of them would be a lie."""
        assert OFFICIAL_NOTICE_MARKDOWN != EXTRA_NOTICE_MARKDOWN
        assert "community fine-tune" in EXTRA_NOTICE_MARKDOWN
        assert "community fine-tune" not in OFFICIAL_NOTICE_MARKDOWN

    def test_neither_notice_claims_a_baked_lora(self) -> None:
        """Baking went away on 2026-09-01 — if it remained in the notice, the stated modifications
        would contradict the facts.

        §3(d)(i) asks for "a notice of the modifications", so listing a modification that was
        not made is not extra wording but **a wrong notice** (plausible prose as values go, so
        it would surface only after shipping).
        """
        for notice in (OFFICIAL_NOTICE_MARKDOWN, EXTRA_NOTICE_MARKDOWN):
            assert "LoRA" not in notice

    def test_neither_notice_claims_an_int4_series(self) -> None:
        """With no i4 slot in the distribution, "an int4 series was added too" is **a notice of a
        modification that was not made**.

        §3(d)(i) asks for a notice of the modifications, so this is an error, not one extra line
        (the old base notice was really in this state — corrected 2026-09-01). The notice is
        fixed on the Pipeline and cannot see the manifest, so the slots and the text move
        **independently** = they need to be cross-checked.
        """
        for notice in (OFFICIAL_NOTICE_MARKDOWN, EXTRA_NOTICE_MARKDOWN):
            assert "int4" not in notice
            assert "int8-quantized series of the transformer was added" in notice
        assert not any("i4" in spec.storages for spec in ANIMA_MODELS.values())

    def test_both_notices_state_the_bf16_series_every_model_ships(self) -> None:
        """Every model ships the original-weights preset `bf16` (ADR 0131 decision 3), so both
        repositories carry a bf16 series and both notices must say so (§3(d)(i)).

        The notice is fixed on the Pipeline and cannot read the manifest, so the storages and the
        text move independently — the same cross-check as the int4 case above, in the other
        direction. The line must not call the series quantized: bf16 keeps the upstream values.
        """
        assert all("bf16" in spec.storages for spec in ANIMA_MODELS.values())
        for notice in (OFFICIAL_NOTICE_MARKDOWN, EXTRA_NOTICE_MARKDOWN):
            flat = " ".join(notice.split())
            assert "A bf16 series was added alongside the f16 one." in flat
            assert (
                "It keeps the values of the upstream BF16 weights unchanged"
                " (no rounding or quantization)" in flat
            )
            assert "bf16-quantized" not in flat

    def test_an_unknown_model_name_is_refused_with_the_choices(self, tmp_path: Path) -> None:
        with pytest.raises(DistError, match="Unknown Anima model name"):
            anima_sources(tmp_path / "series", "anima-nope")


class TestCalibProvenance:
    """Cross-checks at assembly time that the i4 series was made with a shippable rounding (GPTQ
    calibrated × a budget at or above the minimum × this model's conditions).

    Neither the calibration method, nor the budget, nor the conditions **change a single byte of
    the storage form** (research 2026-08-21 §6 — even the file size is identical to the byte),
    so both the header dtype check and the structural check of `verify_dist` let it through.
    `--no-calib` / `--calib-prompts 1` are opt-outs for smoke runs and `--model` is just a knob
    that picks the conditions, yet if their outputs got into the distribution, the assets could
    not tell, and all that would show is a "blurry overall" image — the same discipline as the
    LoRA attribution applies here too.

    The ruling of 2026-09-01 removed the shipping i4 slot from every model, so no real spec ever
    reaches this check. The check is a lever to bring it back (the safety net that applies first
    the day the slot returns) and also the path `anima.eval_dist` uses right now, so **a spec
    with an injected slot** ({@link _i4_seat}) guards the mechanism.
    """

    def _plan(self, tmp_path: Path, **kwargs) -> tuple[AnimaSources, Path]:
        """Series with an injected i4 slot and the output directory (every test in this class takes
        the same single path)."""
        return _build_series(tmp_path / "series", with_i4=True, **kwargs), (
            tmp_path / "models" / "i4-seat"
        )

    def _assemble(self, sources: AnimaSources, out_dir: Path) -> dict[str, Any]:
        return _assemble_anima(sources, out_dir, spec=_i4_seat())

    def test_it_stops_when_the_i4_series_was_rounded_without_calibration(
        self, tmp_path: Path
    ) -> None:
        """Refuses a `--no-calib` output (method = rtn) by name."""
        sources, out_dir = self._plan(tmp_path, calib_method="rtn")

        with pytest.raises(
            DistError, match="not built with a rounding method allowed for distribution"
        ):
            self._assemble(sources, out_dir)

        # It is a check at the planning step, so not a single distribution file appears.
        assert not out_dir.exists()

    def test_it_stops_when_the_i4_series_carries_no_record_at_all(self, tmp_path: Path) -> None:
        """A series without a record (its calibration conditions cannot be cross-checked) is not
        green either."""
        sources, out_dir = self._plan(tmp_path)
        (sources.transformer["i4"] / "transformer" / CALIB_PROVENANCE_FILE).unlink()

        with pytest.raises(DistError, match="has no record of its calibration conditions"):
            self._assemble(sources, out_dir)

    def test_it_stops_when_the_record_is_not_readable_json(self, tmp_path: Path) -> None:
        sources, out_dir = self._plan(tmp_path)
        (sources.transformer["i4"] / "transformer" / CALIB_PROVENANCE_FILE).write_bytes(b"{oops")

        with pytest.raises(
            DistError, match="Cannot parse the record of the calibration conditions"
        ):
            self._assemble(sources, out_dir)

    def test_a_record_written_before_the_guidance_field_is_still_accepted(
        self, tmp_path: Path
    ) -> None:
        """Backward-compatibility MUST: adding a field **does not force rebuilding existing
        series**.

        `guidance` was added on 2026-08-23, when the calibration conditions became per model,
        and the turbo i4 series taken before that (the files uploaded to HF back then) do not
        have it. If the reader required the field, this one-line addition would impose "a
        re-export costing the rounding time" on the existing series — only **the fields present**
        are checked (the MUST of the same name in `_shared.calib_provenance`).
        """
        sources, out_dir = self._plan(tmp_path)
        path = sources.transformer["i4"] / "transformer" / CALIB_PROVENANCE_FILE
        legacy = json.loads(path.read_text(encoding="utf-8"))
        del legacy["guidance"]
        path.write_text(json.dumps(legacy), encoding="utf-8")

        manifest = self._assemble(sources, out_dir)

        assert "f16+dit4" in manifest["models"][ANIMA_TURBO_MODEL_NAME]["quants"]

    def test_it_stops_when_the_calibration_ran_on_a_smaller_budget(self, tmp_path: Path) -> None:
        """`--calib-prompts 1` leaves `method` as `gptq` — it passes unless the budget field is
        checked too.

        The rounding grid does not move, so the file size does not change by a byte; the image
        just gets blurry.
        """
        sources, out_dir = self._plan(tmp_path)
        path = sources.transformer["i4"] / "transformer" / CALIB_PROVENANCE_FILE
        record = json.loads(path.read_text(encoding="utf-8"))
        path.write_text(json.dumps({**record, "prompts": 1}), encoding="utf-8")

        with pytest.raises(
            DistError, match="calibration budget 'prompts' is below the distribution floor"
        ):
            self._assemble(sources, out_dir)

    def test_it_stops_when_the_series_was_calibrated_under_another_models_conditions(
        self, tmp_path: Path
    ) -> None:
        """Refuses by name an i4 baked with the wrong `--model`.

        `--model` of `anima.export` is a knob **only for picking the calibration conditions**,
        so an asset baked from turbo's weights with `--model anima-v1.0` looks like "a correct
        turbo i4" — the storage form, the counts and the LoRA record are all correct; the only
        difference is calibration with the base model's many steps and CFG. (Even if the owners
        of the slots swapped, the conditions are derived from the single place
        `pipeline_config`, so the mix-up in the turbo→base direction goes through the same
        check.)
        """
        sources, out_dir = self._plan(tmp_path, calib_model=ANIMA_BASE_MODEL_NAME)

        with pytest.raises(
            DistError, match="calibration condition 'steps' differs from this model's default"
        ):
            self._assemble(sources, out_dir)

    def test_it_stops_when_only_the_guidance_came_from_another_model(self, tmp_path: Path) -> None:
        """Even with matching steps, a different CFG fails (both fields are in the check)."""
        sources, out_dir = self._plan(tmp_path)
        path = sources.transformer["i4"] / "transformer" / CALIB_PROVENANCE_FILE
        record = json.loads(path.read_text(encoding="utf-8"))
        path.write_text(json.dumps({**record, "guidance": 4.0}), encoding="utf-8")

        with pytest.raises(
            DistError, match="calibration condition 'guidance' differs from this model's default"
        ):
            self._assemble(sources, out_dir)

    def test_the_record_never_reaches_the_distribution(self, assembled_with_i4) -> None:
        """The record is a fact of the series side — it is not carried into the distribution (the HF
        repository).

        Checked with series where the record really exists (= the assembly with an injected i4
        slot). The shipping path has no i4 series at all, so checking there would be vacuous,
        merely confirming that "what does not exist does not appear".
        """
        out_dir, _ = assembled_with_i4

        assert not any(name.endswith(CALIB_PROVENANCE_FILE) for name in _present(out_dir))


class TestStorageGate:
    """The storage checks (based on a real accident — a raw f32 from a forgotten `--dtype` stayed
    silent until the PNG check)."""

    def test_it_stops_when_an_f16_component_is_stored_as_raw_f32(self, tmp_path: Path) -> None:
        sources = _build_series(tmp_path / "series")
        replace_component(
            sources.base["f16"] / "text_encoder" / "model.krm",
            _weights_container("text-encoder", "f32", named="text_encoder"),
        )
        out_dir = tmp_path / "models" / ANIMA_TURBO_MODEL_NAME
        with pytest.raises(DistError, match=r"text_encoder: .* has no f16"):
            _assemble_anima(sources, out_dir)
        # The check runs before placement — not a single file of a partial distribution is left
        # (the same discipline as the rope mismatch).
        assert not out_dir.exists()

    def test_it_stops_when_the_i8_transformer_lacks_i8_storage(self, tmp_path: Path) -> None:
        sources = _build_series(tmp_path / "series")
        replace_component(
            sources.transformer["i8"] / "transformer" / "model.krm",
            _weights_container("transformer-i8", "f16", named="transformer", rope=_ROPE_BASE),
        )
        with pytest.raises(DistError, match=r"transformer_i8: .* has no i8"):
            _assemble_anima(sources, tmp_path / "models" / ANIMA_TURBO_MODEL_NAME)

    def test_it_stops_when_the_i4_transformer_lacks_i4_storage(self, tmp_path: Path) -> None:
        """The mix-up of an i8 series in the i4 slot — if the requirement stayed i8, it would slip
        through silently.

        The slot left the distribution but remains in the requirement table
        (`STORAGE_REQUIREMENTS`), so an injected slot guards the check ({@link _i4_seat}).
        """
        sources = _build_series(tmp_path / "series", with_i4=True)
        replace_component(
            sources.transformer["i4"] / "transformer" / "model.krm",
            _weights_container("transformer-i4", "i8", named="transformer", rope=_ROPE_BASE),
        )
        with pytest.raises(DistError, match=r"transformer_i4: .* has no i4"):
            _assemble_anima(sources, tmp_path / "models" / "i4-seat", spec=_i4_seat())

    def test_it_stops_when_the_i4_series_lands_in_the_i8_seat(self, tmp_path: Path) -> None:
        """The mix-up in the other direction (i4 series → i8 slot) — a presence check alone **lets
        it through**.

        An i4 series is a mix whose default storage is i8, so it always contains i8 and
        satisfies "contains i8". If the default quant `f16+dit8-a8-attn8-s16` picked up a
        resident i4, the `a8` predicate since `c285f97` accepts i4 too
        (ADR 0076), so instead of failing loudly it would run under the w4a8 numeric contract —
        a configuration that ADR 0076 decision 6 decided not to offer would come out silently
        in the default entry. The prohibition table (`ANIMA_STORAGE_FORBIDDEN`) is the only
        detector.
        """
        sources = _build_series(tmp_path / "series")
        replace_component(
            sources.transformer["i8"] / "transformer" / "model.krm",
            _weights_container("transformer-i4", "i4", named="transformer", rope=_ROPE_BASE),
        )
        with pytest.raises(DistError, match=r"transformer_i8: .* has i4"):
            _assemble_anima(sources, tmp_path / "models" / ANIMA_TURBO_MODEL_NAME)

    def test_no_transformer_series_slips_into_another_series_seat(self) -> None:
        """**Every** mix-up of 4 slots × the other 3 series fails through either a requirement or a
        prohibition.

        If only one of the tables were updated the day a slot is added, the combinations that
        slip through the net would silently line up in the distribution (a mix-up of series
        roots cannot be detected by the numeric checks in principle — ADR 0027 / 0029).
        """
        #: Series → the storage vocabulary its binding table **always contains** (i4 is a mix
        #: whose default storage is i8, so it contains i8 too).
        headers = {
            "transformer_f16": {"f32", "f16"},
            "transformer_i8": {"f32", "i8"},
            "transformer_i4": {"f32", "i8", "i4"},
            "transformer_bf16": {"f32", "bf16"},
        }
        # MUST: tie the enumeration to the production table. Leaving it as a dict inside the
        # test would, when a 4th series appears in `STORAGE_REQUIREMENTS` but is not added to
        # `headers`, stay green without ever seeing the very failure mode the docstring
        # names.
        assert set(headers) == {
            role for role in STORAGE_REQUIREMENTS if role.startswith("transformer")
        }

        for seat in headers:
            for series, found in headers.items():
                caught = STORAGE_REQUIREMENTS[seat] not in found or any(
                    dtype in found for dtype in ANIMA_STORAGE_FORBIDDEN.get(seat, ())
                )
                assert caught is (series != seat), f"{series} → {seat} slot"

    def test_every_series_seat_mix_up_is_refused_by_the_real_gates(self, tmp_path: Path) -> None:
        """Runs the 3×3 through **the real gates** (`assert_storage` / `assert_storage_absent`), not
        the table above.

        The test above reimplements the predicates, so it does not fail if the one-line call of
        `assert_storage_absent` disappears from `anima_plan`. This one really goes through the
        assembly, so the moment the call is gone the off-diagonal turns green and it fails. The
        i4 slot left the distribution, so a slot is injected to build the board with all 3
        slots ({@link _i4_seat} — the tables still have 3 slots).
        """
        # What is plugged in is **a valid container** (the diagonal goes through the whole
        # assembly, so it has to be real). The storage vocabulary is the series' own: f16 =
        # {f32, f16} / i8 = {f32, i8} / i4 = {f32, i8, i4}.
        seats = [f"transformer_{storage}" for storage in I4_SEAT_STORAGES]
        for seat in seats:
            for series in seats:
                sources = _build_series(tmp_path / f"series-{seat}-{series}", with_i4=True)
                storage = seat.removeprefix("transformer_")
                target = sources.transformer[storage] / "transformer" / "model.krm"
                replace_component(
                    target,
                    _weights_container(
                        "swapped-series",
                        series.removeprefix("transformer_"),
                        named="transformer",
                        rope=_ROPE_BASE,
                    ),
                )
                out_dir = tmp_path / "models" / f"{seat}-{series}"

                if series == seat:
                    # The diagonal passes (the same series into the same slot)
                    _assemble_anima(sources, out_dir, spec=_i4_seat())
                else:
                    with pytest.raises(DistError):
                        _assemble_anima(sources, out_dir, spec=_i4_seat())

    def test_the_bf16_seats_refuse_every_other_series_through_the_real_gates(
        self, tmp_path: Path
    ) -> None:
        """Runs the mix-ups between the 4 roles of the faithful preset (ADR 0131) and the same
        components of the other storages through **the real gates**.

        The base model has 3 slots for the transformer (f16 / i8 / bf16) and 2 slots for each of
        the 3 text-stack components (f16 / bf16). Plugging a bf16 series into an f16 slot (or
        the reverse) keeps both the shapes and the part count, so the storage checks are the
        only detector — checks that only the diagonal passes and every off-diagonal fails.
        """
        model = ANIMA_BASE_MODEL_NAME
        seats = [
            (component, storage, other)
            for component, storages in (
                ("transformer", ("f16", "i8", "bf16")),
                *((component, ("f16", "bf16")) for component in _TEXT_STACK_COMPONENTS),
            )
            for storage in storages
            for other in storages
        ]
        for component, storage, series in seats:
            sources = _build_series(
                tmp_path / f"series-{component}-{storage}-{series}", model=model
            )
            if component == "transformer":
                root = sources.transformer[storage]
            elif component == "text_conditioner":
                root = sources.text_conditioner[storage]
            else:
                root = sources.base[storage]
            replace_component(
                root / component / "model.krm",
                _weights_container(
                    "swapped-series",
                    series,
                    named=component,
                    rope=_ROPE_BASE if component == "transformer" else None,
                ),
            )
            out_dir = tmp_path / "models" / f"{component}-{storage}-{series}"

            if series == storage:
                _assemble_anima(sources, out_dir, model)
            else:
                with pytest.raises(DistError):
                    _assemble_anima(sources, out_dir, model)

    def test_it_stops_when_the_component_is_not_a_container(self, tmp_path: Path) -> None:
        sources = _build_series(tmp_path / "series")
        replace_component(sources.base["f16"] / "vae_decoder" / "model.krm", b"not-a-container")
        with pytest.raises(DistError, match="cannot be read as a container"):
            _assemble_anima(sources, tmp_path / "models" / ANIMA_TURBO_MODEL_NAME)


class TestPlanGates:
    """Checks decided from `plans` alone run **before the first byte is written** (the MUST at the
    top of dist.py)."""

    def test_it_refuses_a_missing_input_the_storage_gate_never_looks_at(
        self, tmp_path: Path
    ) -> None:
        """A missing input of a role that requires no storage dtype (tokenizer) also fails **before
        placement**.

        For the roles in the requirement table, {@link assert_storage} at the planning step
        checks existence too, so only the roles it lets through remained as a path that "fails
        in the middle of placement".
        """
        sources = _build_series(tmp_path / "series")
        (sources.tokenizers / "qwen2-tokenizer.json").unlink()
        out_dir = tmp_path / "models" / ANIMA_TURBO_MODEL_NAME
        with pytest.raises(DistError, match="assembly input is missing"):
            _assemble_anima(sources, out_dir)
        assert not out_dir.exists()


class TestManifest:
    def test_it_writes_the_envelope_of_adr_0041(self, assembled) -> None:
        out_dir, manifest = assembled
        on_disk = json.loads((out_dir / MANIFEST_FILENAME).read_text(encoding="utf-8"))
        assert on_disk == manifest
        assert manifest["format"] == MANIFEST_FORMAT
        assert manifest["generator"].startswith("karume/")
        assert manifest["defaultModel"] == ANIMA_TURBO_MODEL_NAME
        assert list(manifest["models"]) == [ANIMA_TURBO_MODEL_NAME]
        assert manifest["models"][ANIMA_TURBO_MODEL_NAME]["pipeline"] == "anima/1"

    def test_every_weights_entry_is_keyed_by_dtype(self, assembled) -> None:
        """The two v1 forms `{file}` / `{variants}` are gone — a lone i8 has a dtype key too (§3).

        The expected values are derived from **the storage forms the model declared**
        (`anima_weights`) — expecting the whole vocabulary ({@link ANIMA_WEIGHTS}) as is would
        fail for a model that dropped a slot. The link to the vocabulary is kept as "all roles,
        a subset of labels".
        """
        _, manifest = assembled
        weights = manifest["models"][ANIMA_TURBO_MODEL_NAME]["weights"]
        declared = anima_weights(anima_model(ANIMA_TURBO_MODEL_NAME))

        assert set(declared) == set(ANIMA_WEIGHTS)
        assert set(declared["transformer"]) < set(ANIMA_WEIGHTS["transformer"]), (
            "the i4 slot came back"
        )
        assert sorted(weights) == sorted(declared)
        for name, entry in weights.items():
            assert sorted(entry) == sorted(declared[name]), name
            for files in entry.values():
                assert sorted(files) == ["container"]
                assert sorted(files["container"]) == ["descriptor", "parts"]

    def test_the_unconditional_files_live_in_assets(self, assembled) -> None:
        _, manifest = assembled
        assets = manifest["models"][ANIMA_TURBO_MODEL_NAME]["assets"]
        assert sorted(assets) == ["tokenizer", "tokenizer_2"]
        for ref in assets.values():
            assert sorted(ref) == ["path", "sha256", "size"]

    def test_it_derives_size_and_sha256_from_the_placed_files(self, assembled) -> None:
        out_dir, manifest = assembled
        parts = manifest["models"][ANIMA_TURBO_MODEL_NAME]["weights"]["text_encoder"]["f16"][
            "container"
        ]["parts"]
        # The triple applies to each part (no single entry of the list can be stale).
        for ref, payload in zip(parts, _PAYLOADS["text_encoder"], strict=True):
            assert ref["size"] == len(payload)
            assert ref["sha256"] == hashlib.sha256(payload).hexdigest()
            assert (out_dir / ref["path"]).read_bytes() == payload

    def test_it_carries_the_quant_table_and_pipeline_config(self, assembled) -> None:
        """The quant table holds **only the entries the declared storage forms support** (derived by
        `anima_quants`).

        Keeping the choice of entries as a hard-coded list of quant names would leave the entries
        behind the day a storage form is dropped (selecting one gives 404) — the expected values
        come from the same derivation, so the table and the files move from one place.
        """
        _, manifest = assembled
        model = manifest["models"][ANIMA_TURBO_MODEL_NAME]
        declared = anima_quants(anima_model(ANIMA_TURBO_MODEL_NAME))

        assert sorted(model["quants"]) == sorted(declared)
        assert set(declared) < set(ANIMA_QUANTS), (
            "the i4 entries did not drop out of the quant table"
        )
        assert model["defaultQuant"] == ANIMA_DEFAULT_QUANT
        assert model["defaultQuant"] in model["quants"]
        assert model["pipelineConfig"] == dict(ANIMA_TURBO_PIPELINE_CONFIG)

    def test_every_quant_maps_every_weights_entry(self, assembled) -> None:
        """The hub also checks the completeness of the mapping at run time — a gap would fail only
        after shipping."""
        _, manifest = assembled
        model = manifest["models"][ANIMA_TURBO_MODEL_NAME]
        for name, quant in model["quants"].items():
            assert set(quant["weights"]) == set(model["weights"]), name
            for weight, label in quant["weights"].items():
                assert label in model["weights"][weight]


class TestQuantNaming:
    """Entry names follow the grammar of ADR 0074 (`<storage>[+<component><bits>]…[-<knob>]…`) and
    the presentation fields (ADR 0075)."""

    def test_every_seat_starts_from_the_shared_base_storage(self) -> None:
        """Anima's base storage is `f16` (only the faithful preset is `bf16`) — compression is
        spelled with `+dit…`.

        If the rule that takes `<storage>` literally as the base shared by all roles (ADR 0074
        decisions 1 and 6) broke, entry names would revert to the old spelling that names "only
        the transformer's storage". The base is one of the text stack's storage labels (f16 /
        bf16), and when the table names the text stack, it matches that base.
        """
        bases = {name: seat_base_storage(name) for name in ANIMA_QUANTS}
        assert bases == {name: "bf16" if name == "bf16" else "f16" for name in ANIMA_QUANTS}
        for name, quant in ANIMA_QUANTS.items():
            for role, label in quant["weights"].items():
                if role != "transformer":
                    assert label == bases[name], (name, role)

    def test_the_component_override_token_spells_the_transformer(self) -> None:
        """The recipe defines the abbreviations (ADR 0074 decision 4) — the mapping appears on the
        card too."""
        assert ANIMA_QUANT_ABBREVIATIONS == {"dit": "transformer"}
        assert set(ANIMA_QUANT_ABBREVIATIONS.values()) <= set(ANIMA_WEIGHTS)

    def test_every_seat_carries_a_label_and_a_description_within_the_limits(self) -> None:
        """The presentation fields are optional, but Anima fills them for all 9 entries (a family
        shown in the selection UI)."""
        for name, quant in ANIMA_QUANTS.items():
            assert quant["label"] and quant["description"], name
            assert_quant_presentation(f"anima.quants.{name}", quant)

    def test_no_description_repeats_which_seat_is_the_default(self) -> None:
        """The default is what `defaultQuant` points to (ADR 0075 decision 3) — writing it in prose
        would hold it twice."""
        for name, quant in ANIMA_QUANTS.items():
            assert "default" not in quant["description"].lower(), name


#: The complete mappings of the entries that existed before the faithful preset was added
#: (ADR 0131) (the 3 text-stack roles fixed at f16). Switching the rule that fills the roles a
#: table does not name to "the entry name's `<storage>`" moves none of the existing mappings —
#: that fact is pinned as literals (building the expected values from the derivation would copy
#: a mistake in the rule straight into them and make the test vacuous).
_PRE_BF16_MAPPINGS: Mapping[str, Mapping[str, str]] = {
    name: {
        "text_encoder": "f16",
        "text_conditioner": "f16",
        "transformer": transformer,
        "vae_decoder": "f16",
    }
    for name, transformer in {
        "f16": "f16",
        "f16+dit8": "i8",
        "f16+dit8-a8": "i8",
        "f16+dit8-a8-attn8": "i8",
        "f16+dit8-a8-attn8-s16": "i8",
        "f16+dit4": "i4",
        "f16+dit4-attn8-s16": "i4",
        "f16-c16": "f16",
    }.items()
}


class TestFaithfulPreset:
    """The faithful preset `bf16` (ADR 0131 decision 3 — bf16 for all 4 roles; every model since
    the addendum of 2026-10-17)."""

    @pytest.fixture
    def base_assembled(self, tmp_path: Path) -> tuple[Path, dict]:
        sources = _build_series(tmp_path / "series", model=ANIMA_BASE_MODEL_NAME)
        out_dir = tmp_path / "models" / ANIMA_BASE_MODEL_NAME
        return out_dir, _assemble_anima(sources, out_dir, ANIMA_BASE_MODEL_NAME)

    def test_the_seat_name_yields_its_base_storage(self) -> None:
        """`<storage>` is what comes before the first `+` / `-` (the grammar of ADR 0074)."""
        assert seat_base_storage("bf16") == "bf16"
        assert seat_base_storage("f16") == "f16"
        assert seat_base_storage("f16+dit8-a8-attn8-s16") == "f16"
        assert seat_base_storage("f16-c16") == "f16"

    @pytest.mark.parametrize("model", list(ANIMA_MODELS))
    def test_the_existing_seats_keep_their_mapping_on_every_model(self, model: str) -> None:
        """The complete mappings of the existing entries do not change when the filling rule changes
        (the i4 entries are checked with an injected spec)."""
        spec = anima_model(model)
        spec = replace(spec, storages=(*spec.storages, "i4"))
        weights = anima_weights(spec)
        completed = complete_quant_weights(weights, anima_quants(spec))

        assert {
            name: quant["weights"] for name, quant in completed.items() if name != "bf16"
        } == _PRE_BF16_MAPPINGS

    def test_every_model_declares_the_seat(self) -> None:
        """Entries and files are derived from storages — every model has bf16 for all 4 roles
        (2026-10-17)."""
        holders = [name for name, spec in ANIMA_MODELS.items() if "bf16" in anima_quants(spec)]
        assert holders == list(ANIMA_MODELS)
        for name, spec in ANIMA_MODELS.items():
            for role, labels in anima_weights(spec).items():
                assert "bf16" in labels, (name, role)

    def test_the_seat_maps_all_four_components_to_bf16(self, base_assembled) -> None:
        _, manifest = base_assembled
        entry = manifest["models"][ANIMA_BASE_MODEL_NAME]

        assert entry["quants"]["bf16"]["weights"] == {
            "text_encoder": "bf16",
            "text_conditioner": "bf16",
            "transformer": "bf16",
            "vae_decoder": "bf16",
        }
        assert entry["quants"]["bf16"]["label"] == "Original weights (bf16)"
        assert entry["quants"]["bf16"]["session"] == {}
        # The existing entries keep the f16 text stack (the entry name's `<storage>` fills
        # it).
        for name, mapping in _PRE_BF16_MAPPINGS.items():
            if name in entry["quants"]:
                assert entry["quants"][name]["weights"] == mapping, name
        assert entry["defaultQuant"] == ANIMA_DEFAULT_QUANT

    def test_the_bf16_files_sit_next_to_the_unchanged_f16_ones(self, base_assembled) -> None:
        """bf16 lives in separate files named `.bf16` — not one path of the f16 text stack moves
        (the fine-tune repository refers to those paths across repositories)."""
        out_dir, _ = base_assembled
        assert {role: OUTPUT_PATHS[role] for role in _TEXT_STACK_COMPONENTS} == {
            "text_encoder": "text_encoder/model.krm",
            "text_conditioner": "text_conditioner/model.krm",
            "vae_decoder": "vae_decoder/model.krm",
        }
        for component in (*_TEXT_STACK_COMPONENTS, "transformer"):
            role = f"{component}_bf16"
            assert OUTPUT_PATHS[role] == f"{component}/model.bf16.krm"
            placed = out_dir / ANIMA_BASE_MODEL_NAME / OUTPUT_PATHS[role]
            assert read_component(placed) == _PAYLOADS[role]
        assert _present(out_dir) == sorted([*_in_subtree(ANIMA_BASE_MODEL_NAME), MANIFEST_FILENAME])
        assert sorted(verify_dist(out_dir)) == sorted(_in_subtree(ANIMA_BASE_MODEL_NAME))

    def test_the_series_are_read_from_the_bf16_directories(self, tmp_path: Path) -> None:
        """The text stack + VAE come from the shared `anima-bf16`, the DiT from
        `<model>-bf16-dyn`."""
        sources = anima_sources(tmp_path / "series", ANIMA_BASE_MODEL_NAME)

        assert sources.base == {
            "f16": tmp_path / "series" / "anima-f16",
            "bf16": tmp_path / "series" / "anima-bf16",
        }
        assert sources.text_conditioner == sources.base
        assert sources.transformer["bf16"] == tmp_path / "series" / "anima-v1.0-bf16-dyn"

    def test_the_card_lists_the_seat_with_its_presentation(self, tmp_path: Path) -> None:
        """The card's quant table comes from the manifest — the entry's row shows the presentation
        fields (ADR 0075) and the storage of the 4 roles."""
        _build_series(tmp_path / "series", model=ANIMA_BASE_MODEL_NAME)
        out_dir = tmp_path / "dist"
        main(
            [
                "--pipeline",
                "anima",
                "--series",
                str(tmp_path / "series"),
                "--model",
                ANIMA_BASE_MODEL_NAME,
                "--out",
                str(out_dir),
            ]
        )
        card = (out_dir / MODEL_CARD_FILENAME).read_text(encoding="utf-8")

        row = next(line for line in card.splitlines() if line.startswith("| `bf16` |"))
        assert ANIMA_QUANTS["bf16"]["label"] in row
        assert ANIMA_QUANTS["bf16"]["description"] in row
        for component in (*_TEXT_STACK_COMPONENTS, "transformer"):
            assert f"`{component}` = `bf16`" in row

    def test_the_rope_check_reaches_the_bf16_series(self, tmp_path: Path) -> None:
        sources = _build_series(
            tmp_path / "series", model=ANIMA_BASE_MODEL_NAME, bf16_rope=b"rope-base-DIFFERENT"
        )
        with pytest.raises(DistError, match="is not byte-identical across the series"):
            _assemble_anima(sources, tmp_path / "models" / "out", ANIMA_BASE_MODEL_NAME)

    @pytest.mark.parametrize("component", ["transformer", "text_conditioner"])
    def test_the_lora_check_reaches_the_bf16_series(self, tmp_path: Path, component: str) -> None:
        """The "no baked LoRA" check covers the bf16 series too (the net is per series)."""
        sources = _build_series(tmp_path / "series", model=ANIMA_BASE_MODEL_NAME)
        series = (
            sources.transformer["bf16"]
            if component == "transformer"
            else sources.text_conditioner["bf16"]
        )
        _bake_lora_record(series, component=component)

        with pytest.raises(DistError, match="received a series that carries a"):
            _assemble_anima(sources, tmp_path / "models" / "out", ANIMA_BASE_MODEL_NAME)


class TestI4Quants:
    """The 2 entries with resident i4 (`f16+dit4` = storage only / `f16+dit4-attn8-s16` = **the
    low-VRAM entry**).

    Since the ruling of 2026-09-01 **no shipped model has these 2 entries** (i4 left `storages`,
    and the derivation of `anima_quants` drops the entries with it). The entry definitions remain
    in the quant vocabulary ({@link ANIMA_QUANTS}) as a lever to bring them back — adding i4 to
    `storages` makes them appear as they are. This checks two things: "the vocabulary-side
    definitions are not broken" and "not a single entry reaches the distribution".
    """

    @staticmethod
    def _i4_seats() -> dict[str, Any]:
        return {
            name: quant
            for name, quant in ANIMA_QUANTS.items()
            if quant["weights"].get("transformer") == "i4"
        }

    def test_it_declares_exactly_the_two_i4_seats(self) -> None:
        assert sorted(self._i4_seats()) == ["f16+dit4", "f16+dit4-attn8-s16"]

    def test_the_plain_seat_leaves_the_session_untouched(self) -> None:
        """`f16+dit4` is an entry that changes only the storage (the compute path stays f32)."""
        assert ANIMA_QUANTS["f16+dit4"]["session"] == {}

    def test_the_attention_seat_declares_only_the_attention_knobs(self) -> None:
        assert ANIMA_QUANTS["f16+dit4-attn8-s16"]["session"] == {
            "attentionCompute": "a8",
            "attentionScoreStorage": "f16",
        }

    def test_no_i4_seat_declares_linear_compute(self) -> None:
        """MUST: do not declare `linearCompute` on an i4 entry. **The reason for this invariant
        changed on 2026-08-21** — old: "the `a8` predicate requires resident i8, so declaring it has
        no effect" / new: it does take effect with w4a8 (ADR 0076), but the visual ruling is that
        **applying it roughens the image's fine detail** (research 2026-08-21 §6). Speed comes
        back (1,640 → 955 ms/step), but this entry exists for size and VRAM; if speed is needed,
        the default `f16+dit8-a8-attn8-s16` is better.
        """
        for name, quant in self._i4_seats().items():
            assert "linearCompute" not in quant["session"], name

    def test_the_default_quant_stays_on_the_i8_seat(self) -> None:
        """Adding entries does not move the default (changing the default is a separate decision
        that needs a quality ruling)."""
        assert ANIMA_DEFAULT_QUANT == "f16+dit8-a8-attn8-s16"

    def test_no_shipped_model_carries_an_i4_seat(self, assembled) -> None:
        """MUST: not a single i4 entry appears in the distribution's quant table (ruling of
        2026-09-01).

        "The entry is gone but the files still ship (wasted GB)" and "the files are gone but the
        entry remains (selecting it gives 404)" are different breakages, so **both** the quant
        table and the files are checked.
        """
        _, manifest = assembled
        quants = manifest["models"][ANIMA_TURBO_MODEL_NAME]["quants"]

        assert set(quants) & set(self._i4_seats()) == set()
        for entry in quants.values():
            assert entry["weights"]["transformer"] != "i4"
        for spec in ANIMA_MODELS.values():
            assert set(anima_quants(spec)) & set(self._i4_seats()) == set()

    def test_the_seats_reach_the_manifest_with_their_session_knobs(self, assembled_with_i4) -> None:
        """If the slot comes back, just adding it to the table makes it reach the distribution (the
        quant table comes from the manifest — §3).

        If the derivation (`anima_quants`) broke, the lever to bring it back could not be
        pulled, so a spec with an injected slot runs the one path "declaration → quant table →
        manifest".
        """
        _, manifest = assembled_with_i4
        quants = manifest["models"][ANIMA_TURBO_MODEL_NAME]["quants"]

        assert set(self._i4_seats()) <= set(quants)
        for name, quant in self._i4_seats().items():
            assert quants[name]["weights"]["transformer"] == "i4"
            assert quants[name]["session"] == dict(quant["session"])


class TestVerifyDist:
    def test_it_passes_on_a_freshly_assembled_tree(self, assembled) -> None:
        out_dir, _ = assembled
        assert sorted(verify_dist(out_dir)) == sorted(_in_subtree(ANIMA_TURBO_MODEL_NAME))

    def test_it_catches_a_file_that_no_longer_matches_its_declared_size(self, assembled) -> None:
        out_dir, _ = assembled
        # A split role fails on damage to one part (every part in the list is compared).
        target = container_parts(out_dir / ANIMA_TURBO_MODEL_NAME / OUTPUT_PATHS["vae_decoder"])[1]
        target.unlink()  # unlink the hard link before writing (do not damage the source series)
        target.write_bytes(b"shorter")
        with pytest.raises(DistError, match="size differs from the manifest"):
            verify_dist(out_dir)

    def test_it_catches_a_missing_file(self, assembled) -> None:
        out_dir, _ = assembled
        (out_dir / ANIMA_TURBO_MODEL_NAME / OUTPUT_PATHS["tokenizer"]).unlink()
        with pytest.raises(DistError, match="file the manifest references is missing"):
            verify_dist(out_dir)

    def test_it_catches_an_undeclared_file(self, assembled) -> None:
        out_dir, _ = assembled
        (
            out_dir / ANIMA_TURBO_MODEL_NAME / "transformer" / "io.s01024t0699.safetensors"
        ).write_bytes(b"stale")
        with pytest.raises(DistError, match="Files the manifest does not declare"):
            verify_dist(out_dir)

    def test_it_admits_the_model_card_as_a_meta_file(self, assembled) -> None:
        """`README.md` is a meta file on a par with karume.json (accepted even as a leftover of the
        previous assembly)."""
        out_dir, _ = assembled
        (out_dir / MODEL_CARD_FILENAME).write_text("previous model card", encoding="utf-8")
        assert sorted(verify_dist(out_dir)) == sorted(_in_subtree(ANIMA_TURBO_MODEL_NAME))

    def test_it_still_refuses_a_meta_name_in_a_subdirectory(self, assembled) -> None:
        """The exception covers only the 2 files at the root — the same name in a subdirectory stays
        undeclared."""
        out_dir, _ = assembled
        (out_dir / ANIMA_TURBO_MODEL_NAME / MODEL_CARD_FILENAME).write_text(
            "stray file", encoding="utf-8"
        )
        with pytest.raises(DistError, match="Files the manifest does not declare"):
            verify_dist(out_dir)


def _rewrite(out_dir: Path, mutate) -> None:
    """Rewrite `karume.json` (the only means of building counterexamples for verify_dist)."""
    path = out_dir / MANIFEST_FILENAME
    manifest = json.loads(path.read_text(encoding="utf-8"))
    mutate(manifest)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


class TestVerifyDistStructure:
    """Checks that the v2 manifest **is self-contained** (the producing side also checks that it
    has a shape the hub accepts)."""

    def test_it_refuses_a_manifest_that_is_not_karume_2(self, assembled) -> None:
        out_dir, _ = assembled
        _rewrite(out_dir, lambda m: m.update(format="karume/1"))
        with pytest.raises(DistError, match=MANIFEST_FORMAT):
            verify_dist(out_dir)

    def test_it_refuses_a_default_model_that_is_not_declared(self, assembled) -> None:
        out_dir, _ = assembled
        _rewrite(out_dir, lambda m: m.update(defaultModel="anima-lite"))
        with pytest.raises(DistError, match="defaultModel 'anima-lite'"):
            verify_dist(out_dir)

    def test_it_refuses_a_default_quant_that_is_not_declared(self, assembled) -> None:
        out_dir, _ = assembled

        def drop_default(manifest: dict) -> None:
            manifest["models"][ANIMA_TURBO_MODEL_NAME]["defaultQuant"] = "nonexistent"

        _rewrite(out_dir, drop_default)
        with pytest.raises(DistError, match="defaultQuant 'nonexistent'"):
            verify_dist(out_dir)

    def test_it_refuses_a_quant_that_leaves_a_weights_entry_unmapped(self, assembled) -> None:
        out_dir, _ = assembled

        def unmap(manifest: dict) -> None:
            del manifest["models"][ANIMA_TURBO_MODEL_NAME]["quants"]["f16"]["weights"][
                "vae_decoder"
            ]

        _rewrite(out_dir, unmap)
        with pytest.raises(DistError, match="is not a total mapping"):
            verify_dist(out_dir)

    def test_it_refuses_a_quant_that_names_a_dtype_the_weights_lack(self, assembled) -> None:
        out_dir, _ = assembled

        def retype(manifest: dict) -> None:
            manifest["models"][ANIMA_TURBO_MODEL_NAME]["quants"]["f16"]["weights"][
                "vae_decoder"
            ] = "i8"

        _rewrite(out_dir, retype)
        with pytest.raises(DistError, match="has no dtype 'i8'"):
            verify_dist(out_dir)

    def test_it_refuses_a_path_outside_the_model_subtree_and_shared(self, assembled) -> None:
        """The layout is uniformly "a subtree per model + shared/" (ADR 0041 §9)."""
        out_dir, _ = assembled

        def flatten(manifest: dict) -> None:
            ref = manifest["models"][ANIMA_TURBO_MODEL_NAME]["assets"]["tokenizer"]
            ref["path"] = OUTPUT_PATHS["tokenizer"]

        _rewrite(out_dir, flatten)
        with pytest.raises(DistError, match="the layout is per-model subtrees"):
            verify_dist(out_dir)


class TestFamilyAssembly:
    """Several models in one repository (ADR 0041 §2) + shared files placed once under `shared/`
    (§5).

    The models listed are **a pair that really lives together** (the official repository's base
    model + Aesthetic) — building a combination the accepted sets put in different repositories
    in the fixture would make the tests and {@link TestPipelineMembership} say different things.
    """

    @pytest.fixture
    def family(self, tmp_path: Path) -> tuple[Path, dict]:
        series = tmp_path / "series"
        # base (text path / VAE / tokenizer) is common; only the transformer differs per
        # model.
        first = _build_series(series, model=FAMILY_MODELS[0], mark=b"-base")
        second = _build_series(series, model=FAMILY_MODELS[1], mark=b"-aesthetic")
        out_dir = tmp_path / "models" / "anima-family"
        manifest = assemble_family(
            [anima_plan(first, FAMILY_MODELS[0]), anima_plan(second, FAMILY_MODELS[1])],
            out_dir,
            FAMILY_MODELS[0],
        )
        return out_dir, manifest

    def test_it_declares_every_model_with_the_first_as_default(self, family) -> None:
        _, manifest = family
        assert list(manifest["models"]) == list(FAMILY_MODELS)
        assert manifest["defaultModel"] == FAMILY_MODELS[0]

    def test_it_places_a_byte_identical_file_once_under_shared(self, family) -> None:
        out_dir, manifest = family
        # What gets folded is **the whole component** (the shard list moves under shared/ as
        # is).
        shared = [f"{SHARED_DIRNAME}/{rel}" for rel in part_paths(OUTPUT_PATHS["text_encoder"])]
        for name in manifest["models"]:
            refs = manifest["models"][name]["weights"]["text_encoder"]["f16"]["container"]["parts"]
            assert [ref["path"] for ref in refs] == shared
        for rel, payload in zip(shared, _PAYLOADS["text_encoder"], strict=True):
            assert (out_dir / rel).read_bytes() == payload
        # Nothing remains in each model's subtree (placed once = no duplicates shipped).
        for name in manifest["models"]:
            for rel in part_paths(OUTPUT_PATHS["text_encoder"]):
                assert not (out_dir / name / rel).exists()

    def test_it_keeps_the_files_that_differ_inside_each_model_subtree(self, family) -> None:
        """The weight parts differ per model, so they stay in the subtrees.

        Only the last part is **the dedicated part of the asset `rope_base`**; the rope base
        table depends only on the geometry, so it is byte-identical across models — folding is
        per part, so only that part moves to `shared/` (the same fact as the folding of
        `karume/4`'s `extras`, now at part granularity).
        """
        out_dir, manifest = family
        paths = {
            name: [
                ref["path"] for ref in entry["weights"]["transformer"]["i8"]["container"]["parts"]
            ]
            for name, entry in manifest["models"].items()
        }
        numbered = part_paths(OUTPUT_PATHS["transformer_i8"], _TRANSFORMER_PARTS)
        assert paths == {
            name: [
                *(f"{name}/{rel}" for rel in numbered[:-1]),
                f"{SHARED_DIRNAME}/{numbered[-1]}",
            ]
            for name in FAMILY_MODELS
        }
        assert [(out_dir / rel).read_bytes() for rel in paths[FAMILY_MODELS[0]][:-1]] != [
            (out_dir / rel).read_bytes() for rel in paths[FAMILY_MODELS[1]][:-1]
        ]

    def test_the_shared_and_the_private_files_together_cover_the_tree(self, family) -> None:
        out_dir, _ = family
        # The rope base table is inside the transformer's container, so the folding unit stays
        # the whole component. Every model also has the bf16 text stack (2026-10-17), so it is
        # folded — the fake assets make the conditioner's contents equal across models, so even
        # when read from the model's own series it folds into `shared/` (in the real ones
        # upstream's llm_adapter differs, so it stays in each model's subtree).
        shared_roles = (
            "text_encoder",
            "text_conditioner",
            "vae_decoder",
            "text_encoder_bf16",
            "text_conditioner_bf16",
            "vae_decoder_bf16",
            "tokenizer",
            "tokenizer_2",
        )
        # weights slots expand into numbered parts (container-v1 §8).
        expected = [
            f"{SHARED_DIRNAME}/{rel}"
            for rel in placed_paths(
                {role: OUTPUT_PATHS[role] for role in shared_roles}, ANIMA_WEIGHTS, _PART_TOTALS
            )
        ]
        # The weight parts are per model; only the last asset part (the rope base table) is
        # folded across models. A storage held by only one model (the base model's faithful
        # preset bf16) has nothing to fold with, so both its text stack and its rope part stay
        # whole in that model's subtree (ADR 0041 §5).
        common = set.intersection(*(set(anima_model(model).storages) for model in FAMILY_MODELS))
        for model in FAMILY_MODELS:
            for storage in anima_model(model).storages:
                numbered = part_paths(OUTPUT_PATHS[f"transformer_{storage}"], _TRANSFORMER_PARTS)
                expected += [f"{model}/{rel}" for rel in numbered[:-1]]
                tail = SHARED_DIRNAME if storage in common else model
                expected.append(f"{tail}/{numbered[-1]}")
                if storage != "f16" and storage not in common:
                    private_text = {
                        text_stack_role(component, storage): OUTPUT_PATHS[
                            text_stack_role(component, storage)
                        ]
                        for component in _TEXT_STACK_COMPONENTS
                    }
                    expected += [
                        f"{model}/{rel}" for rel in placed_paths(private_text, ANIMA_WEIGHTS)
                    ]
        assert _present(out_dir) == sorted(set(expected) | {MANIFEST_FILENAME})

    def test_it_leaves_no_empty_directory_behind_after_folding(self, family) -> None:
        out_dir, _ = family
        empty = [
            str(path.relative_to(out_dir))
            for path in out_dir.rglob("*")
            if path.is_dir() and not any(path.iterdir())
        ]
        assert empty == []

    def test_it_refuses_two_references_to_one_path_that_disagree(self, family) -> None:
        """Sharing one path is legal, but disagreeing {size, sha256} would make the fetch layer
        oscillate.

        Sharing happens on the family side (where the dedicated part of the rope base table is
        folded across 2 models), and what gets bent is **a part other than part 0** — part 0's
        size is determined by "header + 2 documents", so
        {@link karume.dist.assert_container_limits} would fail first and the path comparison
        would never be reached.
        """
        out_dir, _ = family

        def bend(manifest: dict) -> None:
            entry = manifest["models"][FAMILY_MODELS[0]]["weights"]["transformer"]["i8"]
            entry["container"]["parts"][-1]["size"] += 1

        _rewrite(out_dir, bend)
        with pytest.raises(DistError, match="disagree"):
            verify_dist(out_dir)

    def test_the_assembled_family_verifies(self, family) -> None:
        out_dir, manifest = family
        declared = verify_dist(out_dir)
        # The shared ones fold into one, so the declared paths are fewer than "number of models
        # × the part count of one model".
        per_model = len(placed_paths(OUTPUT_PATHS, ANIMA_WEIGHTS, _PART_TOTALS))
        assert len(declared) < per_model * len(manifest["models"])

    def test_it_reassembles_a_family_over_a_previous_run(self, tmp_path: Path) -> None:
        """Assembling again over a folded tree does not fail (the path that goes over the existing
        files of `shared/` again)."""
        series = tmp_path / "series"
        first = _build_series(series, model=FAMILY_MODELS[0], mark=b"-base")
        second = _build_series(series, model=FAMILY_MODELS[1], mark=b"-aesthetic")
        out_dir = tmp_path / "models" / "anima-family"
        plans = [anima_plan(first, FAMILY_MODELS[0]), anima_plan(second, FAMILY_MODELS[1])]
        before = assemble_family(plans, out_dir, FAMILY_MODELS[0])
        after = assemble_family(plans, out_dir, FAMILY_MODELS[0])
        assert before == after
        assert verify_dist(out_dir)

    def test_it_refuses_a_duplicated_model_name(self, tmp_path: Path) -> None:
        sources = _build_series(tmp_path / "series")
        plan = anima_plan(sources, ANIMA_TURBO_MODEL_NAME)
        with pytest.raises(DistError, match="Duplicate model names"):
            assemble_family([plan, plan], tmp_path / "out", ANIMA_TURBO_MODEL_NAME)

    def test_it_refuses_a_default_model_it_is_not_assembling(self, tmp_path: Path) -> None:
        sources = _build_series(tmp_path / "series")
        with pytest.raises(DistError, match="Default model 'anima-lite'"):
            assemble_family(
                [anima_plan(sources, ANIMA_TURBO_MODEL_NAME)], tmp_path / "out", "anima-lite"
            )


class TestAtomicReplacement:
    """The assembly builds into staging and puts it in place with a rename — an existing
    distribution is never exposed in a half-built state."""

    def _snapshot(self, out_dir: Path) -> dict[str, bytes]:
        return {rel_path: (out_dir / rel_path).read_bytes() for rel_path in _present(out_dir)}

    def _siblings(self, out_dir: Path) -> list[str]:
        """Working directories left next to the output (staging / the set-aside copy) — empty on
        both success and failure."""
        return sorted(path.name for path in out_dir.parent.iterdir() if path.name != out_dir.name)

    def _fail_at(self, monkeypatch: pytest.MonkeyPatch, nth: int) -> None:
        """Make only the `nth` placement an I/O failure (injects a failure in the middle of several
        GB)."""
        calls = 0

        def failing(artifact: Artifact, dest: Path) -> None:
            nonlocal calls
            calls += 1
            if calls == nth:
                raise OSError("failed in the middle of placement")
            materialize(artifact, dest)

        monkeypatch.setattr("karume.dist.materialize", failing)

    def test_a_failure_midway_leaves_no_distribution_on_a_fresh_target(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sources = _build_series(tmp_path / "series")
        out_dir = tmp_path / "models" / ANIMA_TURBO_MODEL_NAME
        self._fail_at(monkeypatch, nth=3)
        with pytest.raises(OSError, match="failed in the middle of placement"):
            _assemble_anima(sources, out_dir)

        assert not out_dir.exists()
        assert list(out_dir.parent.iterdir()) == []

    def test_it_writes_the_card_into_the_tree_it_swaps_in(self, tmp_path: Path) -> None:
        """`render_card` receives the assembled manifest, and the card is written inside the tree
        that is put in place."""
        sources = _build_series(tmp_path / "series")
        out_dir = tmp_path / "models" / ANIMA_TURBO_MODEL_NAME
        assemble_family(
            [anima_plan(sources, ANIMA_TURBO_MODEL_NAME)],
            out_dir,
            ANIMA_TURBO_MODEL_NAME,
            render_card=lambda manifest, host_assets: f"{manifest['defaultModel']}\n",
        )
        card = (out_dir / MODEL_CARD_FILENAME).read_text(encoding="utf-8")
        assert card == f"{ANIMA_TURBO_MODEL_NAME}\n"
        assert sorted(verify_dist(out_dir)) == sorted(_in_subtree(ANIMA_TURBO_MODEL_NAME))
        assert self._siblings(out_dir) == []

    def test_reassembling_a_subset_replaces_the_whole_repository(self, tmp_path: Path) -> None:
        """Reassembling replaces everything, including models not in the plan (no remains of the
        previous run survive)."""
        series = tmp_path / "series"
        first = _build_series(series, model=FAMILY_MODELS[0], mark=b"-base")
        second = _build_series(series, model=FAMILY_MODELS[1], mark=b"-aesthetic")
        out_dir = tmp_path / "models" / "anima-family"
        assemble_family(
            [anima_plan(first, FAMILY_MODELS[0]), anima_plan(second, FAMILY_MODELS[1])],
            out_dir,
            FAMILY_MODELS[0],
        )

        manifest = assemble_family([anima_plan(first, FAMILY_MODELS[0])], out_dir, FAMILY_MODELS[0])

        assert list(manifest["models"]) == [FAMILY_MODELS[0]]
        assert not (out_dir / FAMILY_MODELS[1]).exists()
        # With a single model there is nothing to fold with = no `shared/` remains either
        # (ADR 0041 §5).
        assert not (out_dir / SHARED_DIRNAME).exists()
        assert _present(out_dir) == sorted([*_in_subtree(FAMILY_MODELS[0]), MANIFEST_FILENAME])
        assert sorted(verify_dist(out_dir)) == sorted(_in_subtree(FAMILY_MODELS[0]))

    def test_it_discards_a_working_directory_left_by_an_interrupted_run(
        self, tmp_path: Path
    ) -> None:
        """A staging directory left by an interruption is discarded, not reused (no files outside
        the plan get mixed in)."""
        sources = _build_series(tmp_path / "series")
        out_dir = tmp_path / "models" / ANIMA_TURBO_MODEL_NAME
        leftover = out_dir.with_name(out_dir.name + STAGING_SUFFIX)
        _write(
            leftover / "anima-wai-v1.0" / "transformer" / "model.f16.safetensors", b"interrupted"
        )

        _assemble_anima(sources, out_dir)

        assert _present(out_dir) == sorted(
            [*_in_subtree(ANIMA_TURBO_MODEL_NAME), MANIFEST_FILENAME]
        )
        assert self._siblings(out_dir) == []

    def test_a_successful_reassembly_lands_the_same_tree_without_leftovers(
        self, tmp_path: Path
    ) -> None:
        """The success path is idempotent across replacements — the manifest and the files equal the
        first run's, and no working traces remain."""
        sources = _build_series(tmp_path / "series")
        out_dir = tmp_path / "models" / ANIMA_TURBO_MODEL_NAME
        first = _assemble_anima(sources, out_dir)
        snapshot = self._snapshot(out_dir)

        again = _assemble_anima(sources, out_dir)

        assert again == first
        assert self._snapshot(out_dir) == snapshot
        assert self._siblings(out_dir) == []


class TestModelCard:
    """`karume dist` writes the model card **after** assembly + verification."""

    def _run(self, tmp_path: Path, *argv: str) -> Path:
        _build_series(tmp_path / "series")
        out_dir = tmp_path / "dist"
        main(
            [
                "--pipeline",
                "anima",
                "--series",
                str(tmp_path / "series"),
                "--out",
                str(out_dir),
                *argv,
            ]
        )
        return out_dir

    def test_it_writes_a_model_card_next_to_the_manifest(self, tmp_path: Path) -> None:
        card = (self._run(tmp_path) / MODEL_CARD_FILENAME).read_text(encoding="utf-8")
        assert card.startswith("---\n")
        assert "base_model: circlestone-labs/Anima-Base-v1.0-Diffusers" in card

    def test_it_derives_the_download_size_from_the_assembled_tree(self, tmp_path: Path) -> None:
        """The number on the card is **the actual byte count of the assembled tree** (table → a
        single total cell since the ruling of 2026-09-03).

        Turns the default quant's Download cell back into its magnitude and compares it with the
        total of `stat()` over the files that entry selects in the tree (allowing only the
        rounding to 3 significant digits). A mismatch between the declaration and the files is
        `verify_dist`'s domain, so this checks "whether the card counts the way the files are".
        """
        out_dir = self._run(tmp_path)
        card = (out_dir / MODEL_CARD_FILENAME).read_text(encoding="utf-8")
        model = json.loads((out_dir / MANIFEST_FILENAME).read_text(encoding="utf-8"))["models"][
            ANIMA_TURBO_MODEL_NAME
        ]
        selected = {ref["path"] for ref in model["assets"].values()}
        for role, label in model["quants"][ANIMA_DEFAULT_QUANT]["weights"].items():
            entry = model["weights"][role][label]
            selected |= {ref["path"] for ref in entry["container"]["parts"]}
        total = sum((out_dir / path).stat().st_size for path in selected)

        # With the table dropped, neither per-part paths nor sizes appear any more (readers see
        # the total).
        for rel_path in _in_subtree(ANIMA_TURBO_MODEL_NAME):
            assert f"`{rel_path}`" not in card
        row = next(
            line for line in card.splitlines() if line.startswith(f"| `{ANIMA_DEFAULT_QUANT}`")
        )
        value, unit = row.split("|")[3].split("(")[0].strip().split(" ")
        scale = {"B": 1, "KiB": 1 << 10, "MiB": 1 << 20, "GiB": 1 << 30}[unit]
        assert abs(float(value.replace(",", "")) * scale - total) <= total * 0.005

    def test_it_names_the_repository_the_pipeline_declares(self, tmp_path: Path) -> None:
        """The repository ID comes from the pipeline's declaration — the name of the build
        directory (`tmp/dist`) does not leak in."""
        card = (self._run(tmp_path) / MODEL_CARD_FILENAME).read_text(encoding="utf-8")
        assert '  repo: "hdae/karume-anima",' in card

    def test_it_leaves_the_tree_verifiable_after_writing_the_card(self, tmp_path: Path) -> None:
        out_dir = self._run(tmp_path)
        assert sorted(verify_dist(out_dir)) == sorted(_in_subtree(ANIMA_TURBO_MODEL_NAME))

    def test_it_reassembles_over_a_previous_card(self, tmp_path: Path) -> None:
        out_dir = self._run(tmp_path)
        first = (out_dir / MODEL_CARD_FILENAME).read_bytes()
        main(
            [
                "--pipeline",
                "anima",
                "--series",
                str(tmp_path / "series"),
                "--out",
                str(out_dir),
            ]
        )
        assert (out_dir / MODEL_CARD_FILENAME).read_bytes() == first


class TestLegalTexts:
    """The distribution repository meets the redistribution conditions of §3 of the upstream
    license (CircleStone Non-Commercial License).

    There are 4 conditions — "provide a copy of the license text" (a), "display the Attribution
    Notice prominently" (b), "include in the Notice that it was modified" (d)(i) and "do not let
    it be mistaken for an official product" (d)(iii) — and each applies to **one distribution
    repository** (not to a model's assets), so the 2 files at the root handle them.
    """

    #: sha256 of the original text fetched from upstream (measured 2026-08-20). It is pinned
    #: because if formatting or newline conversion moved even one byte, it would no longer be "a
    #: copy of this license" (§3(a)).
    LICENSE_SHA256 = "ee956174133d7c2cdcf220440c7726187eaf4b50e8e48ee32194353a22164d15"

    #: sha256 of the vendored NVIDIA Open Model License Agreement — a verbatim plain-text
    #: transcription of the web version "Version Release Date: October 24, 2025"
    #: (<https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/>,
    #: transcribed 2026-10-29 and checked word for word against the page). MUST: a commit that
    #: changes this digest records a deliberate re-transcription and says from which version.
    NVIDIA_LICENSE_SHA256 = "5d375d57dbe72dced9996fc467b6956012574d802ce7daf99345073a2043b402"

    #: What sits between the two licence texts in `LICENSE.md`: the separator and the heading
    #: that states the scope of the second text.
    NVIDIA_PART_OPENING = (
        "\n\n---\n\n## NVIDIA Open Model License Agreement — applies in addition to the"
        " CircleStone license above\n\n"
    )

    def _assert_license_layout(self, shipped: bytes) -> None:
        """§3(a) of both licences — `LICENSE.md` is a copy of each text, not a summary.

        The CircleStone text opens the file byte for byte, then the separator and the scope
        heading, and the vendored NVIDIA text closes it byte for byte inside its fence.
        """
        circlestone = LICENSE_SOURCE_PATH.read_bytes()
        nvidia = NVIDIA_LICENSE_SOURCE_PATH.read_bytes()
        assert shipped.startswith(circlestone)
        rest = shipped[len(circlestone) :].decode("utf-8")
        assert rest.startswith(self.NVIDIA_PART_OPENING)
        assert shipped.endswith(b"```text\n" + nvidia + b"```\n")
        # The fence holds only if the agreement itself never contains one.
        assert b"```" not in nvidia

    def _run(
        self,
        tmp_path: Path,
        *,
        pipeline: str = "anima",
        model: str = ANIMA_TURBO_MODEL_NAME,
        extra: Iterable[str] = (),
    ) -> Path:
        """Run the CLI of `dist.py` once (by default the official repository = `--pipeline anima`).

        There are 2 repositories, and the `NOTICE.md` at the root is a set that differs per
        Pipeline, so the legal-text checks also cover **which repository it was assembled for**.
        """
        _build_series(tmp_path / "series", model=model)
        out_dir = tmp_path / "dist"
        main(
            [
                "--pipeline",
                pipeline,
                "--model",
                model,
                "--series",
                str(tmp_path / "series"),
                "--out",
                str(out_dir),
                *extra,
            ]
        )
        return out_dir

    @staticmethod
    def _official_mirror(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[str]:
        """What an extra-repository build needs: the official repository's mirror it references
        (where the driver reads referenced mirrors, from the same synthetic text stack) and a
        revision source — the placeholder (no Hub in the tests)."""
        sources = _build_series(tmp_path / "official-series", model=ANIMA_BASE_MODEL_NAME)
        assemble_family(
            [anima_plan(sources, ANIMA_BASE_MODEL_NAME)],
            tmp_path / "models" / OFFICIAL_REPO_NAME,
            ANIMA_BASE_MODEL_NAME,
        )
        monkeypatch.setattr(dist, "DIST_ROOT", tmp_path / "models")
        return [dist.ALLOW_PLACEHOLDER_FLAG]

    @staticmethod
    def _flat(text: str) -> str:
        """Flattening so that **sentences** are checked regardless of where lines wrap (legal
        wording works in sentences)."""
        return " ".join(text.split())

    def test_the_recipe_carries_the_license_verbatim(self) -> None:
        source = LICENSE_SOURCE_PATH.read_bytes()
        assert hashlib.sha256(source).hexdigest() == self.LICENSE_SHA256

    def test_the_recipe_carries_the_nvidia_agreement_verbatim(self) -> None:
        source = NVIDIA_LICENSE_SOURCE_PATH.read_bytes()
        assert hashlib.sha256(source).hexdigest() == self.NVIDIA_LICENSE_SHA256
        text = source.decode("utf-8")
        assert text.startswith("NVIDIA Open Model License Agreement\n")
        assert text.endswith("\nVersion Release Date: October 24, 2025\n")

    def test_it_ships_the_license_text_byte_identical(self, tmp_path: Path) -> None:
        """§3(a) — what is provided is **a copy of this license** (neither a summary nor a
        rewrite)."""
        out_dir = self._run(tmp_path)
        self._assert_license_layout((out_dir / "LICENSE.md").read_bytes())

    def test_the_notice_displays_the_attribution_verbatim(self, tmp_path: Path) -> None:
        """§3(b) — the displayed wording is verbatim (both sentences)."""
        notice = (self._run(tmp_path) / "NOTICE.md").read_text(encoding="utf-8")
        assert notice == OFFICIAL_NOTICE_MARKDOWN
        for sentence in ATTRIBUTION_NOTICE.split("\n"):
            assert sentence in notice

    def test_the_extra_repository_ships_its_own_notice(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The root of the fine-tune repository gets **that repository's** modification notice
        (mixing them up would make the notice a lie)."""
        out_dir = self._run(
            tmp_path,
            pipeline="anima-extra",
            model="anima-wai-v1.0",
            extra=self._official_mirror(tmp_path, monkeypatch),
        )

        notice = (out_dir / "NOTICE.md").read_text(encoding="utf-8")
        assert notice == EXTRA_NOTICE_MARKDOWN
        self._assert_license_layout((out_dir / "LICENSE.md").read_bytes())
        assert ATTRIBUTION_NOTICE in notice

    def test_the_notice_specifies_that_the_rights_are_granted_directly_by_the_company(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The latter half of §3(a) — states explicitly that the rights of use are granted directly
        by the Company to the recipient (one verbatim constant).

        Checked both in the notice builder itself (with an arbitrary list of modifications) and
        in the `NOTICE.md` placed in both repositories — if only one path forgot the constant,
        that repository alone would lack §3(a).
        """
        assert RIGHTS_GRANT_STATEMENT in notice_markdown(("- A modification.",))
        official = (self._run(tmp_path / "official") / "NOTICE.md").read_text(encoding="utf-8")
        extra_out = self._run(
            tmp_path / "extra",
            pipeline="anima-extra",
            model="anima-wai-v1.0",
            extra=self._official_mirror(tmp_path / "extra", monkeypatch),
        )
        extra = (extra_out / "NOTICE.md").read_text(encoding="utf-8")
        assert RIGHTS_GRANT_STATEMENT in official
        assert RIGHTS_GRANT_STATEMENT in extra

    def test_the_notice_carries_the_nvidia_attribution_in_both_repositories(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """NVIDIA Open Model License §3.1 / §3.2 — the attribution notice, verbatim on its own
        line, and "Built on NVIDIA Cosmos", from the one shared constant, in its own section
        before the license statement.

        Checked in both repositories' `NOTICE.md`, like the rights-grant statement above — a
        repository whose path forgot the constant would lack the §3.1 notice.
        """
        official = (self._run(tmp_path / "official") / "NOTICE.md").read_text(encoding="utf-8")
        extra_out = self._run(
            tmp_path / "extra",
            pipeline="anima-extra",
            model="anima-wai-v1.0",
            extra=self._official_mirror(tmp_path / "extra", monkeypatch),
        )
        extra = (extra_out / "NOTICE.md").read_text(encoding="utf-8")
        for notice in (official, extra):
            lines = notice.split("\n")
            assert NVIDIA_NOTICE in notice
            assert "Licensed by NVIDIA Corporation under the NVIDIA Open Model License" in lines
            assert "Built on NVIDIA Cosmos." in lines
            _, _, after_disclaimer = notice.partition("## Not an official product")
            nvidia_section, _, _ = after_disclaimer.partition("## License")
            assert "## NVIDIA Open Model License\n" in nvidia_section
            assert NVIDIA_NOTICE in nvidia_section

    def test_the_notice_states_the_modifications_inside_the_attribution_notice(
        self, tmp_path: Path
    ) -> None:
        """§3(d)(i) — states that it was modified **inside the Attribution Notice**."""
        text = (self._run(tmp_path) / "NOTICE.md").read_text(encoding="utf-8")
        # The modifications are not a separate section but inside the Attribution Notice
        # section (before the next heading).
        notice_section = text.split("## Not an official product")[0]
        assert "## Attribution Notice" in notice_section
        flat = self._flat(notice_section)
        assert flat.count("this Attribution Notice also states that") == 1
        assert flat.count("modified as follows:") == 1
        # The modifications actually made (conversion to the container format and the added
        # quantized series) are listed.
        assert "converted into the container format" in flat
        assert (
            "a `.krm` part sequence whose first part carries the graph and model descriptors"
            in flat
        )
        assert "An int8-quantized series of the transformer was added alongside the f16 one" in flat
        assert "A bf16 series was added alongside the f16 one." in flat

    def test_the_notice_disclaims_any_official_standing(self, tmp_path: Path) -> None:
        """§3(d)(iii) — must not be mistaken for an official or endorsed product."""
        text = (self._run(tmp_path) / "NOTICE.md").read_text(encoding="utf-8")
        assert (
            "This is not an official product of CircleStone Labs LLC, and it is not endorsed,"
            " approved or validated by CircleStone Labs LLC." in self._flat(text)
        )
        assert text.endswith(
            "The full license text is distributed alongside this repository as LICENSE.md.\n"
        )

    def test_the_assembled_card_displays_the_attribution_too(self, tmp_path: Path) -> None:
        """The display of §3(b) must hold in **the distribution as placed** (test_card checks the
        section's contents)."""
        card = (self._run(tmp_path) / MODEL_CARD_FILENAME).read_text(encoding="utf-8")
        assert ATTRIBUTION_NOTICE in card
        assert NVIDIA_NOTICE in card

    def test_the_tree_still_verifies_with_the_legal_texts_in_place(self, tmp_path: Path) -> None:
        """The manifest does not declare the 2 root files — they are on the exception side of the
        undeclared-file check."""
        out_dir = self._run(tmp_path)
        assert sorted(verify_dist(out_dir)) == sorted(_in_subtree(ANIMA_TURBO_MODEL_NAME))
        assert (out_dir / "LICENSE.md").is_file()
        assert (out_dir / "NOTICE.md").is_file()

    def test_both_pipelines_declare_exactly_the_two_legal_seats(self) -> None:
        for pipeline in (OFFICIAL_PIPELINE, EXTRA_PIPELINE):
            assert sorted(pipeline.root_files) == ["LICENSE.md", "NOTICE.md"]

    def test_both_pipelines_ship_the_same_license_text(self) -> None:
        """Only one upstream license applies — what differs per repository is only the modification
        notice (§3(a) is the same text)."""
        assert OFFICIAL_PIPELINE.root_files["LICENSE.md"] == EXTRA_PIPELINE.root_files["LICENSE.md"]


class TestPipelineEntry:
    """The 2 lines plugged into the driver ({@link OFFICIAL_PIPELINE} / {@link EXTRA_PIPELINE}).

    The name, the repository name and the attribution are checked independently per repository
    (why they cannot be folded into one: {@link TestPipelineMembership}).
    """

    def test_the_model_name_is_a_legal_path_segment(self) -> None:
        """A model name is both a manifest key and a directory name in the repository (ADR 0041 §6 /
        §9)."""
        for model in (*OFFICIAL_MODELS, *EXTRA_MODELS):
            assert assert_model_name(model) == model

    def test_it_names_the_default_model_and_its_repository(self) -> None:
        assert OFFICIAL_PIPELINE.default_model == ANIMA_TURBO_MODEL_NAME
        assert EXTRA_PIPELINE.default_model == EXTRA_MODELS[0]
        assert OFFICIAL_PIPELINE.repo_name(OFFICIAL_PIPELINE.default_model) == "karume-anima"
        assert EXTRA_PIPELINE.repo_name(EXTRA_PIPELINE.default_model) == "karume-anima-extra"

    def test_the_repository_name_does_not_depend_on_the_model(self) -> None:
        """Several models per repository (ADR 0087) — whichever model is assembled, there is one
        destination.

        Under the old convention, which returned `karume-<model name>`, a single-model assembly
        without `--out` went to a directory named after a repository that does not exist, and
        that name was printed in the card's Usage example too.
        """
        assert {OFFICIAL_PIPELINE.repo_name(model) for model in OFFICIAL_MODELS} == {"karume-anima"}
        assert {EXTRA_PIPELINE.repo_name(model) for model in EXTRA_MODELS} == {"karume-anima-extra"}

    def test_the_extra_repository_references_the_official_text_stack(self) -> None:
        """The six roles the release runbook used to pass as options (ADR 0038 §7 addendum
        2026-10-28); the official repository is self-contained."""
        for model in EXTRA_MODELS:
            (reference,) = EXTRA_PIPELINE.references(model)
            assert reference.repo == "hdae/karume-anima"
            assert reference.model == ANIMA_BASE_MODEL_NAME
            assert sorted(reference.roles) == [
                "text_encoder",
                "text_encoder_bf16",
                "tokenizer",
                "tokenizer_2",
                "vae_decoder",
                "vae_decoder_bf16",
            ]
        for model in OFFICIAL_MODELS:
            assert OFFICIAL_PIPELINE.references(model) == ()

    def test_one_attribution_needs_no_choice(self) -> None:
        """There is only one attribution per repository (nothing is asked when there is no
        choice)."""
        for pipeline, profile in ((OFFICIAL_PIPELINE, "anima"), (EXTRA_PIPELINE, "anima-extra")):
            assert len(pipeline.card_profiles) == 1
            assert resolve_card_renderer(pipeline, None) is pipeline.card_profiles[profile]

    def test_the_two_repositories_render_different_cards(self) -> None:
        """Official / fine-tune differ in both the title and the introduction of the origins section
        — mixing up the renderers would swap the attributions."""
        assert (
            OFFICIAL_PIPELINE.card_profiles["anima"]
            is not EXTRA_PIPELINE.card_profiles["anima-extra"]
        )

    def test_their_cards_refuse_another_pipelines_manifest(self) -> None:
        manifest = {"models": {"m": {"pipeline": "anima/0"}}}
        for pipeline in (OFFICIAL_PIPELINE, EXTRA_PIPELINE):
            for render_card in pipeline.card_profiles.values():
                with pytest.raises(ValueError):
                    render_card(manifest, "hdae/x")
