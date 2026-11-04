"""The Wan distribution recipe (`wan.distribution`) and card (`wan.card`) — unit tests of one
assembly round.

Covers both Wan2.1 (`--pipeline wan`) and Wan2.2 (`--pipeline wan-ti2v` — the same plan function
built with the generation table `WAN22`). In the Wan2.2 set, the DiT / VAE containers claim the
Wan2.2 pin and the two text assets claim the Wan2.1 pin (ADR 0121 decision 9 — the split of the
provenance check).

The inputs that reach the assembly are made of **valid minimal containers** of a few KB
(`ir_fixtures`, and `umt5_fixture` for umT5) and synthetic assets in the same form as the writers
(`wan.text_embeds` / `wan.umt5_tokenizer`). The cases that check a rejection by a check are made
with the same vessels, with **only the declaration shifted from the real thing**. `text_encoder` is
built in the published form (the umT5 distribution is assembled first, and referenced across
repositories).

With `KARUME_SERIES=1` on a machine that holds the real series (`outputs/series/`), the plan is
also built once over them (an implementation-time check — ADR 0139 decision 6; SKIP otherwise).
"""

from __future__ import annotations

import hashlib
import json
import re
import tomllib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import ir_fixtures
import numpy as np
import pytest
from container_series import (
    part_paths,
    placed_paths,
    read_component,
    write_component,
)
from ir_fixtures import fixture_spec, ir_container
from real_assets import SERIES_CHECKS, requires_series
from safetensors.numpy import save
from upstream_fixture import FIXTURE_REVISION, OTHER_REVISION, stamp_fixture_provenance

import dist
from _shared.licenses import APACHE_LICENSE_2_0_PATH
from _shared.paths import REPO_ROOT, SERIES_ROOT
from karume.container import CODEC_LEDGER, AssetInput, Provenance, container_parts
from karume.dist import (
    MANIFEST_FILENAME,
    MODEL_CARD_FILENAME,
    NOTICE_FILENAME,
    CrossRepoReference,
    DistError,
    ExternalComponents,
    assemble_family,
    resolve_card_renderer,
    verify_dist,
)
from karume.emit import stored_model
from karume.ir import IrInitializer, IrShared, IrStorage
from karume.modelcard import FileRefKey
from karume.pipeline import publish_model
from wan import umt5_tokenizer
from wan.card import (
    WAN21_CARD,
    WAN22_ACCEPTED_SIZES,
    WAN22_CARD,
    WAN22_FRAMES,
    WAN22_RESOURCES,
    WAN22_SUPPORTED_PIPELINE,
    WAN_ACCEPTED_SIZES,
    WAN_FRAMES,
    WAN_QUANT_TRANSFORMER,
    WAN_RESOURCE_QUANT,
    WAN_SUPPORTED_PIPELINE,
    WAN_TEXT_ENCODER_OPTION,
    WAN_TEXT_ENCODER_PATHS,
    render_wan_model_card,
)
from wan.distribution import (
    PIPELINE,
    TI2V_PIPELINE,
    WAN21,
    WAN22,
    WAN22_I8_SERIES,
    WAN22_PIPELINE,
    WAN22_REPO_NAME,
    WAN22_SERIES,
    WAN_CONTAINER_ROLES,
    WAN_DIT_CONTEXT_INPUT,
    WAN_GRAPH_ROLES,
    WAN_I8_SERIES,
    WAN_MODEL_FILE,
    WAN_OUTPUT_PATHS,
    WAN_PIPELINE,
    WAN_PIPELINE_CONFIG,
    WAN_QUANT_ABBREVIATIONS,
    WAN_REPO_NAME,
    WAN_ROPE_BASE_ASSET,
    WAN_ROPE_BASE_ROLE,
    WAN_SERIES,
    WAN_STORAGE_FORBIDDEN,
    WAN_STORAGE_REQUIREMENTS,
    WAN_TEXT_EMBEDS_FILE,
    WAN_TEXT_EMBEDS_METADATA_KEY,
    WAN_TEXT_EMBEDS_ROLE,
    WAN_TEXT_EMBEDS_SERIES,
    WAN_TEXT_EMBEDS_VERSIONS,
    WAN_TEXT_ENCODER_DTYPE,
    WAN_TEXT_ENCODER_ROLE,
    WAN_TOKENIZER_ROLE,
    WAN_TOKENIZER_SUBFOLDER,
    WAN_TOKENIZER_VERSIONS,
    WAN_TRANSFORMER_F16_ROLE,
    WAN_TRANSFORMER_I8_ROLE,
    WAN_TRANSFORMER_ROLE,
    WAN_VAE_ENCODER_ATTN_ROLE,
    WAN_VAE_ENCODER_POST_ROLE,
    WAN_VAE_ENCODER_PRE_ROLE,
    WAN_VAE_ENCODER_ROLES,
    WAN_VAE_FIRST_32_I8_ROLE,
    WAN_VAE_FIRST_32_ROLE,
    WAN_VAE_FIRST_I8_ROLE,
    WAN_VAE_FIRST_ROLE,
    WAN_VAE_LATENT_INPUT,
    WAN_VAE_LENDER_ROLES,
    WAN_VAE_NEXT_32_I8_ROLE,
    WAN_VAE_NEXT_32_ROLE,
    WAN_VAE_NEXT_I8_ROLE,
    WAN_VAE_NEXT_ROLE,
    WAN_WEIGHTS,
    WanGeneration,
    WanSources,
    assert_shared_decoder,
    wan_placements,
    wan_plan,
    wan_references,
    wan_sources,
)
from wan.prompts import FIXED_PROMPTS
from wan.sources import DEFAULT_MODEL, SOURCES, UMT5_SOURCES
from wan.tests.umt5_fixture import umt5_container, write_tokenizer_asset
from wan.umt5_distribution import (
    UMT5_DEFAULT_MODEL,
    UMT5_INPUTS,
    UMT5_OUTPUT_PATHS,
    UMT5_REPO_NAME,
    UMT5_ROLE,
    umt5_plan,
)

#: The synthetic DiT context input `[1, rows, width]` (numbers **different** from the real
#: 512 × 4096 — it fails if the width were baked in).
_ROWS = 6
_WIDTH = 8

#: The synthetic base RoPE table (the assembly only checks the declaration and the byte identity
#: between the transformer containers).
_ROPE_BASE = b"rope-base-table!"

#: The valid provenance the writers (`wan.export_dit` / `wan.export_vae`) bake.
_PINNED = Provenance(
    license=SOURCES[DEFAULT_MODEL].license,
    notice=NOTICE_FILENAME,
    upstream_revision=SOURCES[DEFAULT_MODEL].revision,
)

#: The valid provenance the umT5 writer (`wan.umt5_export`) bakes — the umT5 upstream is the pin of
#: the original `google/umt5-xxl`, a different row from the Wan pin of the DiT / VAE (ADR 0122
#: decision 1).
_UMT5_PINNED = Provenance(
    license=UMT5_SOURCES[UMT5_DEFAULT_MODEL].source.license,
    notice=NOTICE_FILENAME,
    upstream_revision=UMT5_SOURCES[UMT5_DEFAULT_MODEL].source.revision,
)

#: The fields of `pipelineConfig` (a copy of `ROOT_KEYS` / `SCHEDULER_KEYS` / `DEFAULTS_KEYS` of
#: the TS side's `packages/models/src/wan/config.ts`). The loading side rejects unknown and missing
#: keys at parse time, so the field names of the baking side and the loading side must match
#: exactly.
_CONFIG_KEYS = {"scheduler": ("shift",), "defaults": ("steps", "guidance")}

#: A copy of the TS side's accepted set (`packages/models/tests/wan_pipeline_test.ts` checks that it
#: equals the values of `descriptor.ts`). The card's table is compared with this fixture, so
#: updating only one of the card and the TS side turns red.
_CARD_LIMITS_FIXTURE = REPO_ROOT / "packages/models/tests/fixtures/wan-card-limits.json"

#: The target of the cross-repository reference (the umT5 distribution repository) and its
#: synthetic commit SHA.
_UMT5_REPO = f"hdae/{UMT5_REPO_NAME}"
_UMT5_REVISION = FIXTURE_REVISION


@pytest.fixture(autouse=True)
def _pinned_provenance(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the fixture containers claim the provenance "baked by the scripts from the pinned
    revision"."""
    stamp_fixture_provenance(monkeypatch, _PINNED)


def _graph_container(
    role: str,
    *,
    storage: str = "f16",
    context: Sequence[int] = (1, _ROWS, _WIDTH),
    rope: tuple[str, str] | None = (WAN_ROPE_BASE_ASSET, WAN_ROPE_BASE_ROLE),
    rope_payload: bytes = _ROPE_BASE,
) -> list[bytes]:
    """One component placed in the series (a list of parts). Only the transformer has the context
    input and the base RoPE table asset."""
    if role != WAN_TRANSFORMER_ROLE:
        return ir_container(mark=role, named=role, storage=storage, inputs=(("latent", [1, 2]),))
    assets = {} if rope is None else {rope[0]: AssetInput(rope[1], len(rope_payload), rope_payload)}
    return ir_container(
        mark=role,
        named=role,
        storage=storage,
        inputs=((WAN_DIT_CONTEXT_INPUT, list(context)),),
        assets=assets,
    )


def _vae_container(role: str, inputs: Sequence[tuple[str, list[int]]]) -> list[bytes]:
    """One VAE chunk graph (only the declarations of the latent + cache inputs have the real
    shapes)."""
    return ir_container(mark=role, named=role, storage="f16", inputs=inputs)


def _prompt_rows() -> list[dict[str, Any]]:
    """The metadata prompts in the same form as the writer (`wan.text_embeds.asset_metadata`) (the
    token counts are synthetic)."""
    return [
        {
            "name": prompt.name,
            "role": prompt.role,
            "prompt": prompt.text,
            "normalized": prompt.text.strip(),
            "tokens": index + 1,
            "source": {"url": prompt.url, "locator": prompt.locator},
        }
        for index, prompt in enumerate(FIXED_PROMPTS)
    ]


def _text_embeds(
    *,
    source: Mapping[str, str] | None = None,
    prompts: list[dict[str, Any]] | None = None,
    width: int = _WIDTH,
    encoder_dtype: str = WAN_TEXT_ENCODER_DTYPE,
    versions: Mapping[str, str] = {},
    extra_metadata: Mapping[str, str] = {},
    shapes: Mapping[str, Sequence[int]] = {},
) -> bytes:
    """A synthetic embedding asset (`F32 [tokens, width]` + one metadata key — the same form as the
    writer).

    `versions` is a diff that overrides the versions the writer records (fixed values).
    """
    upstream = SOURCES[DEFAULT_MODEL]
    rows = _prompt_rows() if prompts is None else prompts
    meta = {
        "source": dict(source)
        if source is not None
        else {"repo": upstream.repo, "revision": upstream.revision},
        "text_encoder": {"class": "UMT5EncoderModel", "dtype": encoder_dtype},
        "versions": {**WAN_TEXT_EMBEDS_VERSIONS, "torch": "2.13.0+cpu", **versions},
        "prompts": rows,
    }
    tensors = {
        row["name"]: np.zeros(shapes.get(row["name"], (row["tokens"], width)), dtype=np.float32)
        for row in rows
    }
    payload = json.dumps(meta, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return save(tensors, metadata={WAN_TEXT_EMBEDS_METADATA_KEY: payload, **extra_metadata})


def _tokenizer_asset(**overrides: Any) -> bytes:
    """A synthetic tokenizer asset (the same top-level fields as the writer
    `wan.umt5_tokenizer.build_asset` — the vocabulary and the tables are not read by the checks, so
    they are absent). `overrides` overrides fields."""
    upstream = SOURCES[DEFAULT_MODEL]
    asset: dict[str, Any] = {
        "format": umt5_tokenizer.ASSET_FORMAT,
        "source": {
            "repo": upstream.repo,
            "revision": upstream.revision,
            "subfolder": WAN_TOKENIZER_SUBFOLDER,
        },
        "versions": {
            **{name: WAN_TEXT_EMBEDS_VERSIONS[name] for name in WAN_TOKENIZER_VERSIONS},
            "python": "3.14.6",
        },
        "maxLength": _ROWS,
        "unkId": 2,
        "eosId": 1,
        **overrides,
    }
    return json.dumps(asset, ensure_ascii=False).encode("utf-8")


#: The placement role of the VAE decoder i8 → the graph name of the container (ADR 0133 decision 7
#: — the Wan2.2 i8 series).
_VAE_I8_GRAPHS = {
    WAN_VAE_FIRST_I8_ROLE: WAN_VAE_FIRST_ROLE,
    WAN_VAE_NEXT_I8_ROLE: WAN_VAE_NEXT_ROLE,
}


def _default_container(role: str) -> list[bytes]:
    """The valid form per placement role (the transformer per storage label — f16 series / i8
    series; the text_encoder is the umT5 i8 series). The provenance is the value `ir_fixtures` bakes
    for the transformer and the VAE (swapped to the Wan pin `_PINNED` only during the tests), and
    the original's pin the umT5 writer bakes for the text_encoder (`_UMT5_PINNED`)."""
    if role == WAN_TEXT_ENCODER_ROLE:
        return umt5_container(provenance=_UMT5_PINNED, width=_WIDTH)
    if role == WAN_TRANSFORMER_F16_ROLE:
        return _graph_container(WAN_TRANSFORMER_ROLE)
    if role == WAN_TRANSFORMER_I8_ROLE:
        return _graph_container(WAN_TRANSFORMER_ROLE, storage="i8")
    if role in _VAE_I8_GRAPHS:
        return _graph_container(_VAE_I8_GRAPHS[role], storage="i8")
    if role in WAN_VAE_LENDER_ROLES:
        return _shared_container(role)
    return _graph_container(role)


#: The placement role of a weight-borrowing set → the graph name of the container (ADR 0134 — the
#: f16 / i8 roles have the same graph name).
_VAE_SHARED_GRAPHS = {
    WAN_VAE_FIRST_32_ROLE: WAN_VAE_FIRST_32_ROLE,
    WAN_VAE_FIRST_32_I8_ROLE: WAN_VAE_FIRST_32_ROLE,
    WAN_VAE_NEXT_32_ROLE: WAN_VAE_NEXT_32_ROLE,
    WAN_VAE_NEXT_32_I8_ROLE: WAN_VAE_NEXT_32_ROLE,
}


def _shared_container(role: str, *, mark: str | None = None) -> list[bytes]:
    """The container of a weight-borrowing set (the same form as the writer
    `wan.export_vae --shared`: every initializer is a shared declaration pointing at the same tensor
    key of the lender, declares the lender's storage, and holds no weight bytes).

    The lender is the default-tile container of the same label ({@link _default_container} — the
    same material `fixture_spec`). Giving `mark` a different graph name from the lender makes a
    borrower that names tensor keys the lender does not have (fault injection).
    """
    lender = WAN_VAE_LENDER_ROLES[role]
    storage = "i8" if lender in _VAE_I8_GRAPHS else "f16"
    graph, tensors, scales, overrides = fixture_spec(
        mark=mark or _VAE_I8_GRAPHS.get(lender, lender),
        storage=storage,
        inputs=((WAN_VAE_LATENT_INPUT, [1, 2]),),
    )
    committed = stored_model(
        graph, tensors, weight_dtype=storage, weight_scales=scales, weight_dtype_overrides=overrides
    ).graph
    shared = replace(
        committed,
        initializers={
            name: IrInitializer(
                shared=IrShared(tensor=init.tensor), storage=IrStorage(dtype=init.storage.dtype)
            )
            for name, init in committed.initializers.items()
        },
    )
    with TemporaryDirectory() as staging:
        path = Path(staging) / "model.krm"
        publish_model(
            path,
            shared,
            {},
            provenance=ir_fixtures.FIXTURE_PROVENANCE,
            graph_name=_VAE_SHARED_GRAPHS[role],
        )
        return [part.read_bytes() for part in container_parts(path)]


def _build_sources(
    root: Path,
    *,
    containers: Mapping[str, list[bytes]] = {},
    embeds: bytes | None = None,
    tokenizer: bytes | None = None,
    generation: WanGeneration = WAN21,
) -> WanSources:
    """Reproduce the 5 series with fake assets (including goldens that must not be distributed).
    The keys of `containers` are placement roles (`text_encoder` / `transformer_f16` /
    `transformer_i8` / the 2 of the VAE — Wan2.2 has no `transformer_f16`). The provenance of the
    containers is whatever `ir_fixtures` bakes at that moment (for the Wan2.2 set, call this after
    switching to the Wan2.2 pin with {@link wan22_provenance})."""
    sources = wan_sources(root / "outputs" / "series", generation)
    placements = wan_placements(sources, generation)
    for role in generation.container_roles:
        write_component(placements[role], containers.get(role) or _default_container(role))
    # Goldens that must not enter the distribution (the series really holds these).
    for role in generation.transformer_roles:
        (placements[role].parent / "io.band-s00192-t0999.safetensors").write_bytes(b"io")
    # pipeline_steps sit at the root of the main DiT series (for real: the f16 series for 2.1, the
    # i8 series for 2.2).
    steps_series = placements[generation.transformer_roles[0]].parent.parent
    (steps_series / "pipeline_steps.band-boxing-cats.safetensors").write_bytes(b"steps")
    # The goldens of the encoder GPU check sit at the root of the f16 series (for real: only the
    # 2.2 series).
    if generation.vae_encoder:
        (sources.series / "vae_encoder.boxing-cats-1280x704.safetensors").write_bytes(b"golden")
    (sources.text_encoder.parent / "reference.band-l0008.safetensors").write_bytes(b"golden")
    sources.text_embeds.parent.mkdir(parents=True, exist_ok=True)
    sources.text_embeds.write_bytes(_text_embeds() if embeds is None else embeds)
    sources.tokenizer.parent.mkdir(parents=True, exist_ok=True)
    sources.tokenizer.write_bytes(_tokenizer_asset() if tokenizer is None else tokenizer)
    return sources


def _assemble_umt5(root: Path, sources: WanSources) -> Path:
    """Assemble the source of the cross-repository reference (the umT5 distribution) from the same
    series (the publication order — umT5 first)."""
    out_dir = root / "models" / UMT5_REPO_NAME
    # The required asset of the umT5 distribution (the tokenizer — the Wan distribution does not
    # reference it) is placed in the parent of the same series.
    tokenizer = write_tokenizer_asset(sources.text_encoder.parents[2])
    assemble_family([umt5_plan(sources.text_encoder, tokenizer)], out_dir, UMT5_DEFAULT_MODEL)
    return out_dir


def _reference(umt5_dir: Path, revision: str = _UMT5_REVISION) -> ExternalComponents:
    """The reference the recipe declares ({@link wan_references} — the text encoder in the umT5
    repository), resolved against a local umT5 distribution."""
    (declared,) = wan_references(DEFAULT_MODEL)
    return ExternalComponents(
        repo=declared.repo,
        revision=revision,
        dist=umt5_dir,
        model=declared.model,
        roles=declared.roles,
    )


def _render(manifest: Mapping[str, Any], host_assets: Mapping[FileRefKey, int]) -> str:
    return PIPELINE.card_profiles["wan"](
        manifest, repo=f"hdae/{WAN_REPO_NAME}", host_assets=host_assets
    )


@pytest.fixture
def assembled(tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    """A distribution in the published form (text_encoder is a cross-repository reference to the
    umT5 repository)."""
    sources = _build_sources(tmp_path)
    out_dir = tmp_path / "models" / WAN_REPO_NAME
    manifest = assemble_family(
        [wan_plan(sources)],
        out_dir,
        DEFAULT_MODEL,
        render_card=_render,
        root_files=PIPELINE.root_files,
        external=(_reference(_assemble_umt5(tmp_path, sources)),),
    )
    return out_dir, manifest


def _card(manifest: Mapping[str, Any], repo: str = "hdae/x") -> str:
    """Rebuild only the card (the abbreviation table passed is the source of truth of the
    distribution recipe)."""
    return render_wan_model_card(manifest, repo, WAN_QUANT_ABBREVIATIONS)


def _model(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    return manifest["models"][DEFAULT_MODEL]


def _present(out_dir: Path) -> list[str]:
    return sorted(str(path.relative_to(out_dir)) for path in out_dir.rglob("*") if path.is_file())


def _overview_names_every_graph(out_dir: Path, manifest: Mapping[str, Any]) -> None:
    """The card's "What is this" section names every component of the manifest's weights (= the
    graphs of the distribution) (catches the shape where, on the day a component is added, only the
    graph list of the overview goes stale)."""
    card = (out_dir / MODEL_CARD_FILENAME).read_text(encoding="utf-8")
    overview = card.split("## What is this", 1)[1].split("\n## ", 1)[0]
    for model in manifest["models"].values():
        missing = [name for name in model["weights"] if f"`{name}`" not in overview]
        assert missing == [], missing


#: The Wan2.2 model name, and the valid provenance its writers (`wan.ti2v_export_dit` /
#: `wan.export_vae --model ti2v-5b`) bake (the Wan2.2 pin).
_TI2V = "ti2v-5b"
_PINNED22 = Provenance(
    license=SOURCES[_TI2V].license,
    notice=NOTICE_FILENAME,
    upstream_revision=SOURCES[_TI2V].revision,
)

#: A copy of the Wan2.2 TS side's accepted set (`WAN22_TI2V_GENERATION` of `descriptor.ts` — the TS
#: test matches the same fixture).
_TI2V_CARD_LIMITS_FIXTURE = REPO_ROOT / "packages/models/tests/fixtures/wan-ti2v-card-limits.json"

#: The Wan2.2 reference preset and practical preset.
_REFERENCE = "f16+dit8"
_PRACTICAL = "f16+dit8-a8-attn8-s16"
#: Wan2.2's online-attention preset (ADR 0132 decision 6 — the default on 2026-10-23 until conv8).
_ONLINE = "f16+dit8-a8-attn8-online"
#: Wan2.2's int8 VAE decoder presets (ADR 0133 decision 6): the reference and the practical one.
_I8 = "i8"
_CONV8 = "i8-a8-attn8-online-conv8"


def _pin(model: str) -> dict[str, str]:
    """The upstream pin in the form of the `source` field of the text assets."""
    return {"repo": SOURCES[model].repo, "revision": SOURCES[model].revision}


@pytest.fixture
def wan22_provenance(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the fixture containers claim the Wan2.2 pin (overriding the autouse Wan2.1 pin)."""
    stamp_fixture_provenance(monkeypatch, _PINNED22)


def _stamped(
    monkeypatch: pytest.MonkeyPatch, provenance: Provenance, make: Callable[[], list[bytes]]
) -> list[bytes]:
    """Make one container that claims `provenance` (afterwards the tests' provenance is
    restored)."""
    with monkeypatch.context() as scoped:
        stamp_fixture_provenance(scoped, provenance)
        return make()


def _render22(manifest: Mapping[str, Any], host_assets: Mapping[FileRefKey, int]) -> str:
    return TI2V_PIPELINE.card_profiles["wan-ti2v"](
        manifest, repo=f"hdae/{WAN22_REPO_NAME}", host_assets=host_assets
    )


def _card22(manifest: Mapping[str, Any], repo: str = "hdae/x") -> str:
    """Rebuild only the Wan2.2 card."""
    return render_wan_model_card(manifest, repo, WAN_QUANT_ABBREVIATIONS, card=WAN22_CARD)


def _build_sources22(root: Path, **overrides: Any) -> WanSources:
    """Reproduce the Wan2.2 series (i8 DiT, f16 VAE, the Wan2.1 text assets, umT5) with fake
    assets."""
    return _build_sources(root, generation=WAN22, **overrides)


@pytest.fixture
def assembled22(tmp_path: Path, wan22_provenance: None) -> tuple[Path, dict[str, Any]]:
    """A Wan2.2 distribution in the published form (text_encoder is a cross-repository reference to
    the umT5 repository)."""
    sources = _build_sources22(tmp_path)
    out_dir = tmp_path / "models" / WAN22_REPO_NAME
    manifest = assemble_family(
        [wan_plan(sources, generation=WAN22)],
        out_dir,
        _TI2V,
        render_card=_render22,
        root_files=TI2V_PIPELINE.root_files,
        external=(_reference(_assemble_umt5(tmp_path, sources)),),
    )
    return out_dir, manifest


class TestLayout:
    def test_it_places_the_graphs_and_the_assets_under_the_model_subtree(self, assembled) -> None:
        """The transformer has 2 per storage label (`model.f16.krm` / `model.i8.krm`), the VAE
        decoder 1 each, and 2 assets. The text_encoder is a cross-repository reference, so not a
        single byte is placed. The VAE encoder is a Wan2.2-only component, so it is not placed (even
        though it is in the all-generation table, the 2.1 distribution does not change)."""
        out_dir, _ = assembled
        own = {
            role: WAN_OUTPUT_PATHS[role]
            for role in (*WAN21.container_roles, WAN_TEXT_EMBEDS_ROLE, WAN_TOKENIZER_ROLE)
            if role != UMT5_ROLE
        }
        expected = [
            f"{DEFAULT_MODEL}/{rel}"
            for rel in placed_paths(
                own,
                {name: labels for name, labels in WAN21.weights.items() if name != UMT5_ROLE},
                # The transformer containers have one more part, dedicated to the `rope_base` asset.
                {WAN_TRANSFORMER_F16_ROLE: 4, WAN_TRANSFORMER_I8_ROLE: 4},
            )
        ]
        assert _present(out_dir) == sorted(
            [*expected, MANIFEST_FILENAME, MODEL_CARD_FILENAME, "LICENSE.md", NOTICE_FILENAME]
        )
        assert list(out_dir.rglob(f"{WAN_TEXT_ENCODER_ROLE}/*")) == []
        assert list(out_dir.rglob("vae_encoder*")) == []
        assert WAN_OUTPUT_PATHS[WAN_TRANSFORMER_F16_ROLE] == "transformer/model.f16.krm"
        assert WAN_OUTPUT_PATHS[WAN_TRANSFORMER_I8_ROLE] == "transformer/model.i8.krm"
        assert WAN_OUTPUT_PATHS[WAN_TOKENIZER_ROLE] == "umt5_tokenizer/tokenizer.json"

    def test_it_never_carries_the_series_goldens(self, assembled) -> None:
        out_dir, _ = assembled
        assert list(out_dir.rglob("io.*")) == []
        assert list(out_dir.rglob("pipeline_steps.*")) == []
        assert list(out_dir.rglob("reference.*")) == []

    def test_a_vae_encoder_beside_the_wan21_vae_is_never_carried(self, tmp_path: Path) -> None:
        """Even with an encoder container in the 2.1 f16 series, the 2.1 plan does not pick it up
        (the encoder is a Wan2.2-only component — the all-generation weights table reaches 2.1 too,
        so the generation field must narrow it)."""
        sources = _build_sources(tmp_path)
        for role in WAN_VAE_ENCODER_ROLES:
            write_component(sources.series / role / WAN_MODEL_FILE, _graph_container(role))
        plan = wan_plan(sources)
        assert not set(WAN_VAE_ENCODER_ROLES) & set(plan.artifacts)
        assert not set(WAN_VAE_ENCODER_ROLES) & set(plan.weights)
        for quant in plan.quants.values():
            assert not set(WAN_VAE_ENCODER_ROLES) & set(quant["weights"])

    def test_the_manifest_declares_the_three_seats_and_the_model_assets(self, assembled) -> None:
        """There are 3 quant presets: `f16`, the reference preset `f16+dit8` and the practical
        preset `f16+dit8-a8-attn8-s16` (the default) (ADR 0120 decision 1, ruling 4 of 2026-10-04),
        and every preset selects the text_encoder i8 (weights is a complete mapping — ADR 0119
        addendum B). The assets are the per-model `text_embeds` and `umt5_tokenizer` (independent of
        the quant — addendum C)."""
        _, manifest = assembled
        model = _model(manifest)
        assert manifest["defaultModel"] == DEFAULT_MODEL
        assert model["pipeline"] == WAN_PIPELINE
        # The VAE encoder (Wan2.2 only) is not in the 2.1 weights — the 2.1 manifest is unchanged.
        assert list(model["weights"]) == [
            WAN_TEXT_ENCODER_ROLE,
            WAN_TRANSFORMER_ROLE,
            WAN_VAE_FIRST_ROLE,
            WAN_VAE_NEXT_ROLE,
        ]
        assert list(model["weights"][WAN_TEXT_ENCODER_ROLE]) == ["i8"]
        assert list(model["weights"][WAN_TRANSFORMER_ROLE]) == ["f16", "i8"]
        # The VAE decoder i8 (Wan2.2 only — ADR 0133 decision 7) is not in the 2.1 weights.
        assert list(model["weights"][WAN_VAE_FIRST_ROLE]) == ["f16"]
        assert list(model["weights"][WAN_VAE_NEXT_ROLE]) == ["f16"]
        assert list(model["assets"]) == [WAN_TEXT_EMBEDS_ROLE, WAN_TOKENIZER_ROLE]
        assert model["assets"][WAN_TEXT_EMBEDS_ROLE]["path"] == (
            f"{DEFAULT_MODEL}/{WAN_TEXT_EMBEDS_ROLE}/{WAN_TEXT_EMBEDS_FILE}"
        )
        assert model["assets"][WAN_TOKENIZER_ROLE]["path"] == (
            f"{DEFAULT_MODEL}/{WAN_TOKENIZER_ROLE}/{umt5_tokenizer.ASSET_FILE}"
        )
        assert list(model["quants"]) == ["f16", "f16+dit8", "f16+dit8-a8-attn8-s16"]
        assert model["defaultQuant"] == "f16+dit8-a8-attn8-s16"
        vae = {WAN_VAE_FIRST_ROLE: "f16", WAN_VAE_NEXT_ROLE: "f16"}
        text = {WAN_TEXT_ENCODER_ROLE: "i8"}
        quants = model["quants"]
        assert quants["f16"]["weights"] == {**text, WAN_TRANSFORMER_ROLE: "f16", **vae}
        assert quants["f16+dit8"]["weights"] == {**text, WAN_TRANSFORMER_ROLE: "i8", **vae}
        assert quants["f16+dit8-a8-attn8-s16"]["weights"] == quants["f16+dit8"]["weights"]
        assert quants["f16"]["session"] == {}
        # The reference preset has the same i8 weights as the practical preset and an empty
        # session (the counterpart of the local A/B check — ADR 0110 decision 5 ②).
        assert quants["f16+dit8"]["session"] == {}
        assert quants["f16+dit8-a8-attn8-s16"]["session"] == {
            "linearCompute": "a8",
            "attentionCompute": "a8",
            "attentionScoreStorage": "f16",
        }

    def test_the_pipeline_config_has_exactly_the_fields_the_loader_accepts(self, assembled) -> None:
        _, manifest = assembled
        config = _model(manifest)["pipelineConfig"]
        assert {key: tuple(value) for key, value in config.items()} == _CONFIG_KEYS
        assert config == WAN_PIPELINE_CONFIG

    def test_it_reassembles_to_the_same_bytes(self, tmp_path: Path) -> None:
        """Assembling twice from the same series gives the same bytes for the manifest and every
        file (deterministic)."""
        sources = _build_sources(tmp_path)
        out_dir = tmp_path / "models" / WAN_REPO_NAME

        def digest() -> dict[str, bytes]:
            return {path: (out_dir / path).read_bytes() for path in _present(out_dir)}

        first = assemble_family([wan_plan(sources)], out_dir, DEFAULT_MODEL)
        before = digest()
        assert first == assemble_family([wan_plan(sources)], out_dir, DEFAULT_MODEL)
        assert digest() == before
        assert verify_dist(out_dir)

    def test_the_repository_ships_the_apache_license_and_the_change_notice(self, assembled) -> None:
        out_dir, _ = assembled
        assert (out_dir / "LICENSE.md").read_bytes() == APACHE_LICENSE_2_0_PATH.read_bytes()
        notice = (out_dir / NOTICE_FILENAME).read_text(encoding="utf-8")
        assert "Apache License, Version 2.0" in notice
        # The weights of the i8 presets are modifications of the distribution, so they are in the
        # notice (MUST: the text corresponds to the distribution).
        prose = " ".join(notice.split())
        assert "**int8 transformer**" in prose
        assert "keep the source float32 values" in prose

    def test_the_notice_says_where_the_text_encoder_comes_from(self, assembled) -> None:
        """The text_encoder is a cross-repository reference (the umT5 repository), the embedding
        assets stay for the precomputed path, and the tokenizer is bundled with the preprocessing
        tables (ADR 0119 addenda A / C). It never says "not distributed"."""
        out_dir, _ = assembled
        prose = " ".join((out_dir / NOTICE_FILENAME).read_text(encoding="utf-8").split())
        assert "**The text encoder is referenced, not stored here.**" in prose
        assert f"`{UMT5_REPO_NAME}` at a pinned commit" in prose
        assert "for use without the text encoder" in prose
        assert "ftfy's mojibake-detection pattern" in prose
        assert "text encoder is not distributed" not in prose


class TestTheTextEncoderReference:
    """The text_encoder is declared as a cross-repository reference to the umT5 distribution (ADR
    0119 addendum A — ADR 0109 decision 3)."""

    def test_every_part_is_pinned_to_the_umt5_repository(self, assembled) -> None:
        out_dir, manifest = assembled
        umt5 = json.loads(
            (out_dir.parent / UMT5_REPO_NAME / MANIFEST_FILENAME).read_text(encoding="utf-8")
        )
        source = umt5["models"][UMT5_DEFAULT_MODEL]["weights"][UMT5_ROLE]["i8"]["container"]
        entry = _model(manifest)["weights"][WAN_TEXT_ENCODER_ROLE]["i8"]["container"]

        assert entry["parts"] == [
            {"repo": _UMT5_REPO, "revision": _UMT5_REVISION, **ref} for ref in source["parts"]
        ]
        assert [ref["path"] for ref in entry["parts"]] == [
            f"{UMT5_DEFAULT_MODEL}/{rel}"
            for rel in part_paths(UMT5_OUTPUT_PATHS[UMT5_ROLE], len(source["parts"]))
        ]
        assert entry["descriptor"] == source["descriptor"]

    def test_the_referenced_bytes_are_the_series_container(self, tmp_path: Path) -> None:
        """The referenced parts are the umT5 series container itself (the Wan and umT5 repositories
        point at the same bytes)."""
        sources = _build_sources(tmp_path)
        umt5_dir = _assemble_umt5(tmp_path, sources)
        placed = umt5_dir / UMT5_DEFAULT_MODEL / UMT5_OUTPUT_PATHS[UMT5_ROLE]

        assert read_component(placed) == read_component(sources.text_encoder)

    def test_it_refuses_a_reference_whose_bytes_differ_from_its_own(self, tmp_path: Path) -> None:
        """If the source is a different umT5 (here a container with a different output width), it
        is not assembled (the core cross-check — a reference with different contents would
        distribute "another model's weights" as its own)."""
        other = _build_sources(
            tmp_path / "other",
            containers={
                WAN_TEXT_ENCODER_ROLE: umt5_container(provenance=_UMT5_PINNED, width=_WIDTH + 1)
            },
        )
        umt5_dir = _assemble_umt5(tmp_path / "other", other)
        sources = _build_sources(tmp_path)

        with pytest.raises(DistError, match="differs from the bytes this assembly builds"):
            assemble_family(
                [wan_plan(sources)],
                tmp_path / "models" / WAN_REPO_NAME,
                DEFAULT_MODEL,
                external=(_reference(umt5_dir),),
            )

    def test_the_output_paths_match_the_umt5_repository(self) -> None:
        """A cross-repository reference looks up `<model name>/<this path>` declared by the source,
        so there is one spelling."""
        assert WAN_OUTPUT_PATHS[WAN_TEXT_ENCODER_ROLE] == UMT5_OUTPUT_PATHS[UMT5_ROLE]
        assert WAN_TEXT_ENCODER_ROLE == UMT5_ROLE


class TestTheStorageGates:
    @pytest.mark.parametrize("storage", ["f32", "i8"])
    def test_it_refuses_a_series_without_f16_storage(self, tmp_path: Path, storage: str) -> None:
        """The mix-up of pointing at a plain f32 / other-storage series while meaning the f16
        series."""
        sources = _build_sources(
            tmp_path,
            containers={"vae_decoder_next": _graph_container("vae_decoder_next", storage=storage)},
        )
        with pytest.raises(DistError, match=r"vae_decoder_next: .* has no f16"):
            wan_plan(sources)

    def test_it_refuses_the_i8_series_in_the_f16_transformer_seat(self, tmp_path: Path) -> None:
        sources = _build_sources(
            tmp_path,
            containers={
                WAN_TRANSFORMER_F16_ROLE: _graph_container(WAN_TRANSFORMER_ROLE, storage="i8")
            },
        )
        with pytest.raises(DistError, match=r"transformer_f16: .* has no f16"):
            wan_plan(sources)

    def test_it_refuses_the_f16_series_in_the_i8_transformer_seat(self, tmp_path: Path) -> None:
        sources = _build_sources(
            tmp_path,
            containers={WAN_TRANSFORMER_I8_ROLE: _graph_container(WAN_TRANSFORMER_ROLE)},
        )
        with pytest.raises(DistError, match=r"transformer_i8: .* has no i8"):
            wan_plan(sources)

    def test_it_refuses_a_mixed_i4_series_in_the_i8_transformer_seat(self, tmp_path: Path) -> None:
        """A mixed i4 series has i8 as its default storage, so it satisfies "contains i8" — only the
        forbidden table rejects it."""
        sources = _build_sources(
            tmp_path,
            containers={
                WAN_TRANSFORMER_I8_ROLE: _graph_container(WAN_TRANSFORMER_ROLE, storage="i4")
            },
        )
        with pytest.raises(DistError, match=r"transformer_i8: .* has i4"):
            wan_plan(sources)

    def test_it_refuses_an_f16_graph_in_the_text_encoder_seat(self, tmp_path: Path) -> None:
        """A series that is not i8 (a mixed-up f16 DiT series) plugged into the text_encoder
        preset."""
        sources = _build_sources(
            tmp_path,
            containers={WAN_TEXT_ENCODER_ROLE: _graph_container(WAN_TEXT_ENCODER_ROLE)},
        )
        with pytest.raises(DistError, match=r"text_encoder: .* has no i8"):
            wan_plan(sources)

    def test_it_refuses_a_text_encoder_whose_tables_were_quantized(self, tmp_path: Path) -> None:
        """A umT5 whose relative-position tables were rounded to i8 (it satisfies "contains i8" —
        only the binding-table check rejects it)."""
        sources = _build_sources(
            tmp_path,
            containers={
                WAN_TEXT_ENCODER_ROLE: umt5_container(provenance=_UMT5_PINNED, table_layout="i8")
            },
        )
        with pytest.raises(DistError, match="differs from the per-kind requirements"):
            wan_plan(sources)

    def test_the_forbidden_table_names_every_other_compressed_layout(self) -> None:
        """Per role, the forbidden table holds **all** layouts of the codec ledger except f32 / i32
        and the required storage."""
        compressed = {entry.layout for entry in CODEC_LEDGER.values()} - {"f32", "i32"}
        assert set(WAN_STORAGE_FORBIDDEN) == set(WAN_STORAGE_REQUIREMENTS)
        assert set(WAN_STORAGE_REQUIREMENTS) == set(WAN_CONTAINER_ROLES)
        for role, required in WAN_STORAGE_REQUIREMENTS.items():
            assert set(WAN_STORAGE_FORBIDDEN[role]) == compressed - {required}, role


class TestTheTextEncoderContract:
    """The inputs and outputs of the umT5 container mesh with Wan's text step and the DiT context
    (ADR 0119 decisions 3 / 4)."""

    def test_it_refuses_an_encoder_whose_width_differs_from_the_dit_context(
        self, tmp_path: Path
    ) -> None:
        sources = _build_sources(
            tmp_path,
            containers={
                WAN_TEXT_ENCODER_ROLE: umt5_container(provenance=_UMT5_PINNED, width=_WIDTH + 1)
            },
        )
        with pytest.raises(DistError, match=f"output width {_WIDTH + 1}"):
            wan_plan(sources)

    def test_it_refuses_an_encoder_with_other_input_names(self, tmp_path: Path) -> None:
        renamed = [
            ("ids", dtype, shape) if index == 0 else (name, dtype, shape)
            for index, (name, dtype, shape) in enumerate(UMT5_INPUTS)
        ]
        sources = _build_sources(
            tmp_path,
            containers={
                WAN_TEXT_ENCODER_ROLE: umt5_container(provenance=_UMT5_PINNED, inputs=renamed)
            },
        )
        with pytest.raises(DistError, match="the graph inputs"):
            wan_plan(sources)

    def test_it_refuses_a_missing_encoder(self, tmp_path: Path) -> None:
        sources = _build_sources(tmp_path)
        for part in sources.text_encoder.parent.glob("model-*.krm"):
            part.unlink()
        with pytest.raises(DistError):
            wan_plan(sources)


class TestTheTokenizerAsset:
    """The checks of the tokenizer asset (format, provenance, versions, maximum length —
    `wan.distribution.assert_umt5_tokenizer`)."""

    def test_the_writer_and_the_gate_spell_the_same_format(self) -> None:
        """The synthetic asset is made with the format version of the writer (the form the check
        accepts has not drifted from the writer)."""
        assert json.loads(_tokenizer_asset())["format"] == umt5_tokenizer.ASSET_FORMAT

    def test_it_refuses_another_format(self, tmp_path: Path) -> None:
        sources = _build_sources(
            tmp_path, tokenizer=_tokenizer_asset(format="karume-wan-umt5-tokenizer/2")
        )
        with pytest.raises(DistError, match="parseWanTokenizerAsset"):
            wan_plan(sources)

    @pytest.mark.parametrize(
        "field", ["revision", "subfolder"], ids=["other-revision", "other-subfolder"]
    )
    def test_it_refuses_a_tokenizer_from_elsewhere(self, tmp_path: Path, field: str) -> None:
        source = json.loads(_tokenizer_asset())["source"]
        source[field] = OTHER_REVISION if field == "revision" else "text_encoder"
        sources = _build_sources(tmp_path, tokenizer=_tokenizer_asset(source=source))
        with pytest.raises(DistError, match="differs from the upstream pin"):
            wan_plan(sources)

    @pytest.mark.parametrize("package", WAN_TOKENIZER_VERSIONS)
    def test_it_refuses_a_tokenizer_made_with_another_pinned_version(
        self, tmp_path: Path, package: str
    ) -> None:
        versions = {**json.loads(_tokenizer_asset())["versions"], package: "0.0.1"}
        sources = _build_sources(tmp_path, tokenizer=_tokenizer_asset(versions=versions))
        with pytest.raises(DistError, match="group's pins"):
            wan_plan(sources)

    @pytest.mark.parametrize("max_length", [_ROWS + 1, 0, "512", True])
    def test_it_refuses_a_max_length_the_dit_context_cannot_hold(
        self, tmp_path: Path, max_length: object
    ) -> None:
        sources = _build_sources(tmp_path, tokenizer=_tokenizer_asset(maxLength=max_length))
        with pytest.raises(DistError, match="maxLength"):
            wan_plan(sources)

    def test_a_max_length_equal_to_the_context_rows_is_accepted(self, tmp_path: Path) -> None:
        """The pair (boundary): exactly the row count passes (the synthetic default has this
        form)."""
        assert json.loads(_tokenizer_asset())["maxLength"] == _ROWS
        assert wan_plan(_build_sources(tmp_path))

    def test_it_refuses_an_asset_that_is_not_json(self, tmp_path: Path) -> None:
        sources = _build_sources(tmp_path, tokenizer=b"\xff not json")
        with pytest.raises(DistError, match="JSON"):
            wan_plan(sources)

    def test_it_refuses_a_missing_asset(self, tmp_path: Path) -> None:
        sources = _build_sources(tmp_path)
        sources.tokenizer.unlink()
        with pytest.raises(DistError):
            wan_plan(sources)


class TestTheContainerProvenance:
    """Matches the provenance the containers claim against the upstream pin
    (`wan.sources.SOURCES`)."""

    def test_it_refuses_a_container_baked_from_another_revision(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        stamp_fixture_provenance(
            monkeypatch,
            Provenance(
                license=_PINNED.license, notice=NOTICE_FILENAME, upstream_revision=OTHER_REVISION
            ),
        )
        with pytest.raises(DistError, match="a different revision"):
            wan_plan(_build_sources(tmp_path))

    def test_it_refuses_a_container_that_names_another_license(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        stamp_fixture_provenance(
            monkeypatch,
            Provenance(
                license="cc-by-nc-4.0",
                notice=NOTICE_FILENAME,
                upstream_revision=_PINNED.upstream_revision,
            ),
        )
        with pytest.raises(DistError, match=r"provenance\.license is 'cc-by-nc-4.0'"):
            wan_plan(_build_sources(tmp_path))

    def test_it_refuses_a_model_missing_from_the_source_table(self, tmp_path: Path) -> None:
        with pytest.raises(DistError, match="unknown"):
            wan_plan(_build_sources(tmp_path), "t2v-14b")

    def test_it_refuses_a_text_encoder_that_names_the_wan_checkpoint(self, tmp_path: Path) -> None:
        """The umT5 container claims the pin of the original `google/umt5-xxl` (ADR 0122 decision
        1) — a container claiming the Wan pin (the form before the provenance was switched) is
        rejected by the umT5 check even inside the Wan plan."""
        encoder = umt5_container(provenance=_PINNED, width=_WIDTH)
        sources = _build_sources(tmp_path, containers={WAN_TEXT_ENCODER_ROLE: encoder})
        with pytest.raises(DistError, match="a different revision"):
            wan_plan(sources)

    def test_it_refuses_a_wan22_model_of_the_source_table(self, tmp_path: Path) -> None:
        """Even a Wan2.2 model in the source table is not assembled into the Wan2.1 distribution
        (`karume-wan2.1`)."""
        assert "ti2v-5b" in SOURCES
        with pytest.raises(DistError, match=r"unknown Wan2\.1 model 'ti2v-5b'"):
            wan_plan(_build_sources(tmp_path), "ti2v-5b")


class TestTheRopeBaseAsset:
    @pytest.mark.parametrize(
        ("role", "storage"),
        [(WAN_TRANSFORMER_F16_ROLE, "f16"), (WAN_TRANSFORMER_I8_ROLE, "i8")],
        ids=["f16", "i8"],
    )
    def test_it_refuses_a_transformer_without_the_rope_base_tables(
        self, tmp_path: Path, role: str, storage: str
    ) -> None:
        """A DiT without the base table would fail only after the user's fromPretrained has
        downloaded the weights."""
        sources = _build_sources(
            tmp_path,
            containers={role: _graph_container(WAN_TRANSFORMER_ROLE, storage=storage, rope=None)},
        )
        with pytest.raises(DistError, match=r"asset 'rope_base' is missing"):
            wan_plan(sources)

    def test_it_refuses_a_rope_base_asset_with_another_role(self, tmp_path: Path) -> None:
        sources = _build_sources(
            tmp_path,
            containers={
                WAN_TRANSFORMER_F16_ROLE: _graph_container(
                    "transformer", rope=(WAN_ROPE_BASE_ASSET, "ple-index")
                )
            },
        )
        with pytest.raises(DistError, match="has role 'ple-index'"):
            wan_plan(sources)

    def test_it_refuses_rope_base_tables_that_differ_between_the_seats(
        self, tmp_path: Path
    ) -> None:
        """The f16 / i8 containers hold base tables of the same geometry — if only one had a
        different table, RoPE would go wrong for that preset alone."""
        sources = _build_sources(
            tmp_path,
            containers={
                WAN_TRANSFORMER_I8_ROLE: _graph_container(
                    WAN_TRANSFORMER_ROLE, storage="i8", rope_payload=b"other-rope-table"
                )
            },
        )
        with pytest.raises(DistError, match="is not byte-identical"):
            wan_plan(sources)


class TestTheVaeChunkPair:
    """first / next are chunk graphs of the same set (the same rules as wanVaeChunkLayout on the
    TS side)."""

    LATENT = (WAN_VAE_LATENT_INPUT, [16, 1, 4, 4])

    def _pair(self, tmp_path: Path, first, following) -> WanSources:
        return _build_sources(
            tmp_path,
            containers={
                WAN_VAE_FIRST_ROLE: _vae_container(WAN_VAE_FIRST_ROLE, first),
                WAN_VAE_NEXT_ROLE: _vae_container(WAN_VAE_NEXT_ROLE, following),
            },
        )

    def test_a_first_whose_caches_are_a_subsequence_of_next_is_accepted(
        self, tmp_path: Path
    ) -> None:
        """The same form as the real thing: next has caches between that first does not (the 2 of
        time_conv)."""
        first = [self.LATENT, ("cache_00", [4, 2, 4, 4]), ("cache_02", [8, 2, 4, 4])]
        following = [*first[:2], ("cache_01", [3, 2, 4, 4]), first[2]]

        assert wan_plan(self._pair(tmp_path, first, following))

    def test_it_refuses_graphs_baked_with_different_tiles(self, tmp_path: Path) -> None:
        """A set where only one was rebaked with another tile edge (a partial update with --target,
        an old real run that failed midway)."""
        first = [self.LATENT, ("cache_00", [4, 2, 4, 4])]
        following = [(WAN_VAE_LATENT_INPUT, [16, 1, 8, 8]), ("cache_00", [4, 2, 8, 8])]
        with pytest.raises(DistError, match="the latent input shape differs"):
            wan_plan(self._pair(tmp_path, first, following))

    def test_it_refuses_a_first_cache_that_next_lacks_or_shapes_differently(
        self, tmp_path: Path
    ) -> None:
        first = [self.LATENT, ("cache_00", [4, 2, 4, 4])]
        following = [self.LATENT, ("cache_00", [5, 2, 4, 4])]
        with pytest.raises(DistError, match=r"'cache_00' .* is not in next with the same shape"):
            wan_plan(self._pair(tmp_path, first, following))

    def test_it_refuses_caches_in_another_order(self, tmp_path: Path) -> None:
        first = [self.LATENT, ("cache_00", [4, 2, 4, 4]), ("cache_01", [4, 2, 4, 4])]
        following = [self.LATENT, first[2], first[1]]
        with pytest.raises(DistError, match="order of the cache inputs differs from next"):
            wan_plan(self._pair(tmp_path, first, following))

    def test_it_refuses_a_graph_whose_first_input_is_not_the_latent(self, tmp_path: Path) -> None:
        first = [("cache_00", [4, 2, 4, 4]), self.LATENT]
        with pytest.raises(DistError, match="the first graph input"):
            wan_plan(self._pair(tmp_path, first, [self.LATENT, first[0]]))


class TestTheTextEmbeddingAsset:
    def test_it_refuses_a_second_metadata_key(self, tmp_path: Path) -> None:
        sources = _build_sources(tmp_path, embeds=_text_embeds(extra_metadata={"other": "1"}))
        with pytest.raises(DistError, match="the metadata keys"):
            wan_plan(sources)

    def test_it_refuses_embeddings_made_from_another_revision(self, tmp_path: Path) -> None:
        upstream = SOURCES[DEFAULT_MODEL]
        sources = _build_sources(
            tmp_path,
            embeds=_text_embeds(source={"repo": upstream.repo, "revision": OTHER_REVISION}),
        )
        with pytest.raises(DistError, match="differs from the upstream pin"):
            wan_plan(sources)

    def test_it_refuses_prompts_that_differ_from_the_fixed_table(self, tmp_path: Path) -> None:
        """The card draws its text from the fixed-prompt table, so if the asset metadata differs it
        is not distributed."""
        rows = _prompt_rows()
        rows[0] = {**rows[0], "prompt": rows[0]["prompt"] + " Extra."}
        sources = _build_sources(tmp_path, embeds=_text_embeds(prompts=rows))
        with pytest.raises(DistError, match="the fixed prompt table"):
            wan_plan(sources)

    def test_it_refuses_a_width_that_differs_from_the_dit_context(self, tmp_path: Path) -> None:
        sources = _build_sources(tmp_path, embeds=_text_embeds(width=_WIDTH + 1))
        with pytest.raises(DistError, match="the DiT's context input"):
            wan_plan(sources)

    def test_it_refuses_an_i8_transformer_whose_context_differs(self, tmp_path: Path) -> None:
        """The embedding asset is one quant-independent file — it must also mesh with the context
        input of the DiT of the i8 preset."""
        sources = _build_sources(
            tmp_path,
            containers={
                WAN_TRANSFORMER_I8_ROLE: _graph_container(
                    WAN_TRANSFORMER_ROLE, storage="i8", context=(1, _ROWS, _WIDTH + 1)
                )
            },
        )
        with pytest.raises(DistError, match="the DiT's context input"):
            wan_plan(sources)

    def test_it_refuses_more_valid_rows_than_the_dit_context_holds(self, tmp_path: Path) -> None:
        rows = _prompt_rows()
        rows[1] = {**rows[1], "tokens": _ROWS + 1}
        sources = _build_sources(tmp_path, embeds=_text_embeds(prompts=rows))
        with pytest.raises(DistError, match=f"valid length 1 to {_ROWS}"):
            wan_plan(sources)

    def test_it_refuses_a_tensor_whose_rows_differ_from_the_token_count(
        self, tmp_path: Path
    ) -> None:
        name = FIXED_PROMPTS[2].name
        sources = _build_sources(tmp_path, embeds=_text_embeds(shapes={name: (5, _WIDTH)}))
        with pytest.raises(DistError, match=f"'{name}' is not F32"):
            wan_plan(sources)

    def test_it_refuses_a_missing_asset(self, tmp_path: Path) -> None:
        sources = _build_sources(tmp_path)
        sources.text_embeds.unlink()
        with pytest.raises(DistError):
            wan_plan(sources)

    def test_it_refuses_embeddings_from_an_encoder_in_another_dtype(self, tmp_path: Path) -> None:
        """NOTICE and the card claim "the bfloat16 upstream encoder" — an asset made in f32 is not
        distributed."""
        sources = _build_sources(tmp_path, embeds=_text_embeds(encoder_dtype="float32"))
        with pytest.raises(DistError, match="the dtype 'float32' of the umT5"):
            wan_plan(sources)

    @pytest.mark.parametrize("package", sorted(WAN_TEXT_EMBEDS_VERSIONS))
    def test_it_refuses_embeddings_made_with_another_pinned_version(
        self, tmp_path: Path, package: str
    ) -> None:
        sources = _build_sources(tmp_path, embeds=_text_embeds(versions={package: "0.0.1"}))
        with pytest.raises(DistError, match="decision 4's fixed"):
            wan_plan(sources)

    @pytest.mark.parametrize(
        "normalized", [None, 123, ""], ids=["missing", "not-a-string", "empty"]
    )
    def test_it_refuses_a_row_without_a_normalized_text(
        self, tmp_path: Path, normalized: object
    ) -> None:
        """The TS parseWanTextEmbeds rejects a row without normalized.

        The user would fail after the download, so the check before distribution rejects the same
        rows.
        """
        rows = _prompt_rows()
        if normalized is None:
            del rows[1]["normalized"]
        else:
            rows[1] = {**rows[1], "normalized": normalized}
        sources = _build_sources(tmp_path, embeds=_text_embeds(prompts=rows))
        with pytest.raises(DistError, match=f"'{rows[1]['name']}'s normalized"):
            wan_plan(sources)

    @pytest.mark.parametrize("field", ["normalized", "prompt"])
    def test_it_refuses_a_normalized_text_that_another_row_owns(
        self, tmp_path: Path, field: str
    ) -> None:
        """If the normalized string of row 1 equals the string of row 0, which embedding to use is
        undecided."""
        rows = _prompt_rows()
        rows[1] = {**rows[1], "normalized": rows[0][field]}
        sources = _build_sources(tmp_path, embeds=_text_embeds(prompts=rows))
        with pytest.raises(
            DistError, match=f"the string of '{rows[1]['name']}' is the same as '{rows[0]['name']}'"
        ):
            wan_plan(sources)

    def test_a_row_whose_normalized_text_is_its_own_prompt_is_accepted(
        self, tmp_path: Path
    ) -> None:
        """A row whose original text is unchanged by normalization (the real boxing-cats) passes.

        A match within the same row is not a duplicate.
        """
        rows = _prompt_rows()
        rows[0] = {**rows[0], "normalized": rows[0]["prompt"]}
        assert wan_plan(_build_sources(tmp_path, embeds=_text_embeds(prompts=rows)))


class TestTheModelCard:
    def test_it_is_the_only_profile_and_is_resolved_without_a_choice(self) -> None:
        profiles = PIPELINE.card_profiles
        assert list(profiles) == ["wan"]
        assert resolve_card_renderer(PIPELINE, None) is profiles["wan"]

    def test_the_quant_table_explains_the_abbreviation_in_the_seat_names(self, assembled) -> None:
        """The abbreviation mapping is **always** printed (ADR 0074 decision 4) — so that it is
        readable which component `dit8` refers to."""
        out_dir, _ = assembled
        card = (out_dir / MODEL_CARD_FILENAME).read_text(encoding="utf-8")
        assert "In a quant name, `dit` is the `transformer` component." in card
        quants = card.split("### Quants")[1]
        assert "| `f16+dit8-a8-attn8-s16` (default) |" in quants
        assert "| `f16` |" in quants
        assert "(default)" not in quants.replace("| `f16+dit8-a8-attn8-s16` (default) |", "")

    def test_the_overview_names_every_graph_of_the_manifest(self, assembled) -> None:
        _overview_names_every_graph(*assembled)

    def test_it_refuses_a_pipeline_it_does_not_describe(self, assembled) -> None:
        _, manifest = assembled
        foreign = json.loads(json.dumps(manifest))
        foreign["models"][DEFAULT_MODEL]["pipeline"] = "siglip2/1"
        with pytest.raises(ValueError, match=WAN_SUPPORTED_PIPELINE):
            _card(foreign)

    def test_the_title_is_followed_by_the_house_sentence(self, assembled) -> None:
        _, manifest = assembled
        lines = _card(manifest).splitlines()
        title = lines.index("# Wan2.1 T2V 1.3B — Karume")
        assert lines[title + 2] == (
            "This is an unofficial conversion of Wan-AI/Wan2.1-T2V-1.3B-Diffusers for Karume."
            " It is not affiliated with or endorsed by Wan-AI."
        )

    def test_it_refuses_to_attribute_a_wan22_model(self, assembled) -> None:
        """Even for a Wan2.2 model in the source table, the Wan2.1 card does not write its
        attribution."""
        _, manifest = assembled
        foreign = json.loads(json.dumps(manifest))
        foreign["models"] = {"ti2v-5b": foreign["models"][DEFAULT_MODEL]}
        assert "ti2v-5b" in SOURCES
        with pytest.raises(ValueError, match=r"'ti2v-5b' is not a Wan2\.1 model"):
            _card(foreign)

    def test_it_attributes_the_pinned_upstream_revision(self, assembled) -> None:
        out_dir, _ = assembled
        card = (out_dir / MODEL_CARD_FILENAME).read_text(encoding="utf-8")
        upstream = SOURCES[DEFAULT_MODEL]
        assert f"base_model: {upstream.repo}" in card
        assert f"license: {upstream.license}" in card
        assert f"at commit `{upstream.revision}`" in card

    def test_it_lists_every_fixed_prompt_with_its_role(self, assembled) -> None:
        out_dir, _ = assembled
        card = (out_dir / MODEL_CARD_FILENAME).read_text(encoding="utf-8")
        for prompt in FIXED_PROMPTS:
            assert f"| `{prompt.name}` | {prompt.role} |" in card
            assert prompt.text.strip("\n") in card

    def test_the_usage_names_the_declared_repository(self, assembled) -> None:
        out_dir, _ = assembled
        card = (out_dir / MODEL_CARD_FILENAME).read_text(encoding="utf-8")
        assert f'repo: "hdae/{WAN_REPO_NAME}"' in card
        assert "await Pipeline.fromPretrained(" in card
        assert "fromAssets" not in card

    def test_the_defaults_come_from_the_manifest(self, assembled) -> None:
        """The defaults are not baked in but drawn from pipelineConfig (assembling with values that
        differ from the real thing gives different values)."""
        _, manifest = assembled
        changed = json.loads(json.dumps(manifest))
        changed["models"][DEFAULT_MODEL]["pipelineConfig"] = {
            "scheduler": {"shift": 7.5},
            "defaults": {"steps": 23, "guidance": 4.25},
        }
        card = _card(changed)
        assert "- **steps**: 23" in card
        assert "- **guidance**: 4.25" in card
        assert "- **shift** (flow-matching shift): 7.5" in card

    def test_it_renders_the_same_bytes_for_the_same_manifest(self, assembled) -> None:
        _, manifest = assembled
        assert _card(manifest) == _card(manifest)

    def test_the_accepted_inputs_match_the_typescript_side(self) -> None:
        """The card's accepted set equals the TS accepted set (the other side is
        wan_pipeline_test.ts)."""
        fixture = json.loads(_CARD_LIMITS_FIXTURE.read_text(encoding="utf-8"))
        assert fixture == {
            "acceptedSizes": [
                {"width": width, "height": height} for width, height in WAN_ACCEPTED_SIZES
            ],
            "minFrames": WAN_FRAMES[0],
            "maxFrames": WAN_FRAMES[1],
        }, (
            "after changing WAN_ACCEPTED_SIZES / WAN_FRAMES in card.py, align the fixture and"
            " descriptor.ts too"
        )

    def test_it_names_the_measured_resources_with_their_conditions(self, assembled) -> None:
        out_dir, _ = assembled
        card = (out_dir / MODEL_CARD_FILENAME).read_text(encoding="utf-8")
        assert "## Resources" in card
        assert f"with the `{WAN_RESOURCE_QUANT}` quant" in card
        assert "6.19 GiB" in card
        assert "3.32 GiB" in card
        prose = " ".join(card.split())
        assert "took 10.4 s and peaked at 6.30 GiB" in prose
        assert "A 50-step run through the text encoder has not been done yet." in prose
        assert "`maxStorageBufferBindingSize` (128 MiB)" in card

    def test_it_names_the_browser_run_and_what_has_not_run_in_a_browser(self, assembled) -> None:
        """The browser claims only the conditions it actually ran, and says the paths and presets
        it did not run have not run (ADR 0118 step 9)."""
        out_dir, _ = assembled
        prose = " ".join((out_dir / MODEL_CARD_FILENAME).read_text(encoding="utf-8").split())
        assert "Chrome on an NVIDIA GeForce RTX 5070 Ti finished one 50-step run" in prose
        assert "81 frames) in 46.1 minutes" in prose
        assert (
            "The text encoder on the GPU and the int8 quants have not been run in a browser yet."
            in prose
        )
        assert "Browsers are not verified yet" not in prose
        assert "Browsers have not been checked yet" not in prose

    def test_it_names_the_measured_figures_of_every_seat(self, assembled) -> None:
        """The per-preset transformer rows stay as measured; fields that were not measured are not
        guessed and say not measured."""
        out_dir, _ = assembled
        card = (out_dir / MODEL_CARD_FILENAME).read_text(encoding="utf-8")
        assert "| `f16` | 33 | 17.1 s | 5.15 GiB | ~30 minutes |" in card
        assert "| `f16+dit8` | 81 | not measured | not measured | not run |" in card
        assert "| `f16+dit8-a8-attn8-s16` | 33 | 8.5 s | 4.02 GiB | 952 s (~16 minutes) |" in card
        assert (
            "| `f16+dit8-a8-attn8-s16` | 81 | 34.0 s | 5.46 GiB | 3,703 s (~62 minutes) |" in card
        )
        assert "have not been measured" not in card
        prose = " ".join(card.split())
        assert "a relative RMS error of 0.107 at 33 frames and 0.210 at 81 frames" in prose
        assert "no clear degradation was seen" in prose
        assert "the default quant stays" not in prose

    def test_it_says_a_seat_without_figures_has_not_been_measured(self, assembled) -> None:
        """A preset not in the table does not guess numbers and says not measured (the order of the
        presets is as in the manifest)."""
        _, manifest = assembled
        changed = json.loads(json.dumps(manifest))
        quants = changed["models"][DEFAULT_MODEL]["quants"]
        quants["f16+other"] = quants[WAN_RESOURCE_QUANT]
        assert "f16+other" not in WAN_QUANT_TRANSFORMER
        card = _card(changed)
        assert "The other quants (`f16+other`) have not been measured yet." in card
        assert "| `f16+other` |" not in card.split("### Quants")[0]

    def test_it_refuses_a_manifest_without_the_measured_quant(self, assembled) -> None:
        """A distribution without the measured preset does not claim resource numbers (no guessed
        numbers are printed)."""
        _, manifest = assembled
        changed = json.loads(json.dumps(manifest))
        quants = changed["models"][DEFAULT_MODEL]["quants"]
        quants["i8"] = quants.pop(WAN_RESOURCE_QUANT)
        changed["models"][DEFAULT_MODEL]["defaultQuant"] = "i8"
        with pytest.raises(ValueError, match=f"quant '{WAN_RESOURCE_QUANT}'"):
            _card(changed)


class TestTheModelCardOnTheTextEncoder:
    """The card's text_encoder description (where it comes from, the path choice, the download size,
    the declared limits) comes from the manifest."""

    def test_it_names_the_repository_and_the_commit_it_borrows_from(self, assembled) -> None:
        out_dir, _ = assembled
        card = " ".join((out_dir / MODEL_CARD_FILENAME).read_text(encoding="utf-8").split())
        link = f"[`{_UMT5_REPO}`](https://huggingface.co/{_UMT5_REPO})"
        assert f"referenced from {link}" in card
        assert f"at commit `{_UMT5_REVISION[:16]}…` of {link}" in card
        assert "stored in this repository" not in card

    def test_a_self_contained_build_says_the_encoder_is_stored_here(self, tmp_path: Path) -> None:
        """Even assembled without a cross-repository reference (one repository of 8 GiB), the card
        describes the facts."""
        manifest = assemble_family([wan_plan(_build_sources(tmp_path))], tmp_path / "x", "t2v-1.3b")
        card = " ".join(_card(manifest).split())
        assert "stored in this repository" in card
        assert UMT5_REPO_NAME not in card

    def test_the_usage_offers_free_text_and_the_path_choice(self, assembled) -> None:
        out_dir, _ = assembled
        card = (out_dir / MODEL_CARD_FILENAME).read_text(encoding="utf-8")
        gpu, precomputed = WAN_TEXT_ENCODER_PATHS
        assert f'// {WAN_TEXT_ENCODER_OPTION}: "{gpu}", // default — "{precomputed}"' in card
        assert 'prompt: "A cat walks on the grass, realistic style.",' in card
        assert "pipeline.prompts.find" not in card
        prose = " ".join(card.split())
        assert "accepts free text" in prose
        assert "the text encoder session is disposed before the transformer session" in prose
        assert (
            f'With `{WAN_TEXT_ENCODER_OPTION}: "{precomputed}"`, the text encoder is not fetched'
            in prose
        )

    def test_the_download_of_the_text_encoder_comes_from_the_manifest(self, assembled) -> None:
        _, manifest = assembled
        changed = json.loads(json.dumps(manifest))
        parts = _model(changed)["weights"][WAN_TEXT_ENCODER_ROLE]["i8"]["container"]["parts"]
        parts[-1]["size"] += 3 << 30
        size = sum(ref["size"] for ref in parts)
        prose = " ".join(_card(changed).split())
        assert (
            f"includes the text encoder ({size / (1 << 30):.2f} GiB from `{_UMT5_REPO}`)" in prose
        )

    def test_it_names_the_declared_limits_when_every_quant_declares_them(self, assembled) -> None:
        _, manifest = assembled
        assert "Declared limits" not in _card(manifest)
        changed = json.loads(json.dumps(manifest))
        limits = {"maxBufferSize": 1_050_148_864, "maxStorageBufferBindingSize": 1_050_148_864}
        for quant in _model(changed)["quants"].values():
            quant["requiredLimits"] = dict(limits)
        prose = " ".join(_card(changed).split())
        assert (
            "every quant declares `maxBufferSize` ≥ 1,050,148,864 bytes and"
            " `maxStorageBufferBindingSize` ≥ 1,050,148,864 bytes" in prose
        )
        _model(changed)["quants"]["f16"]["requiredLimits"]["maxBufferSize"] += 1
        with pytest.raises(ValueError, match="requiredLimits differ"):
            _card(changed)

    def test_it_refuses_a_manifest_without_the_text_encoder(self, assembled) -> None:
        _, manifest = assembled
        changed = json.loads(json.dumps(manifest))
        del _model(changed)["weights"][WAN_TEXT_ENCODER_ROLE]
        with pytest.raises(ValueError, match="text_encoder"):
            _card(changed)


class TestReferenceRevisionSources:
    """Where the driver `dist.py` takes the umT5 reference's revision from (module doc of
    `dist.py`, "Cross-repository references") — exactly one source per build, settled before a
    byte is written.

    The placeholder (40 zeros) is reachable only through its explicit flag (ADR 0119 addendum E):
    the publication rebuild (release-runbook §0) never passes it, so it cannot ship a commit that
    does not exist. The Hub is mocked — the tests read no network.
    """

    _HUB_REVISION = "fedcba9876543210fedcba9876543210fedcba98"

    @staticmethod
    def _argv(root: Path, *extra: str) -> list[str]:
        return [
            "--pipeline",
            "wan",
            "--series",
            str(root / "outputs" / "series"),
            "--out",
            str(root / "models" / WAN_REPO_NAME),
            *extra,
        ]

    @staticmethod
    def _ready(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        """The umT5 distribution where the driver reads referenced mirrors (`models/<name>/`)."""
        monkeypatch.setattr(dist, "DIST_ROOT", tmp_path / "models")
        return _assemble_umt5(tmp_path, _build_sources(tmp_path))

    def _hub(
        self, umt5_dir: Path, *, oids: Mapping[str, str] | None = None
    ) -> Callable[[str], tuple[Any, str | None]]:
        """A mocked HF API whose main of the umT5 repository holds the local mirror's files (every
        file as LFS, its oid = the sha256), with `oids` overriding some."""
        tree = [
            {
                "type": "file",
                "path": str(path.relative_to(umt5_dir)),
                "oid": "0" * 40,
                "lfs": {
                    "oid": (oids or {}).get(
                        str(path.relative_to(umt5_dir)),
                        hashlib.sha256(path.read_bytes()).hexdigest(),
                    ),
                    "size": path.stat().st_size,
                },
            }
            for path in sorted(umt5_dir.rglob("*"))
            if path.is_file()
        ]
        api = f"https://huggingface.co/api/models/{_UMT5_REPO}"

        def get(url: str) -> tuple[Any, str | None]:
            if url == f"{api}/revision/main":
                return {"sha": self._HUB_REVISION}, None
            assert url == f"{api}/tree/{self._HUB_REVISION}?recursive=true", url
            return tree, None

        return get

    def _text_encoder_refs(self, root: Path) -> list[Mapping[str, Any]]:
        manifest = json.loads(
            (root / "models" / WAN_REPO_NAME / MANIFEST_FILENAME).read_text(encoding="utf-8")
        )
        return _model(manifest)["weights"][WAN_TEXT_ENCODER_ROLE]["i8"]["container"]["parts"]

    def test_both_generations_declare_the_umt5_text_encoder(self) -> None:
        """The reference the runbook used to spell as options: `hdae/karume-umt5-xxl`, model
        `xxl`, role `text_encoder`."""
        expected = (CrossRepoReference("karume-umt5-xxl", "xxl", ("text_encoder",)),)

        assert PIPELINE.references(DEFAULT_MODEL) == expected
        assert TI2V_PIPELINE.references(WAN22.default_model) == expected

    def test_publishing_with_the_placeholder_is_refused_before_anything_is_written(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._ready(tmp_path, monkeypatch)
        argv = self._argv(
            tmp_path, "--ref-revision", f"{UMT5_REPO_NAME}={dist.PLACEHOLDER_REVISION}"
        )

        with pytest.raises(DistError, match=dist.ALLOW_PLACEHOLDER_FLAG):
            dist.main(argv)
        assert not (tmp_path / "models" / WAN_REPO_NAME).exists()

    def test_a_development_mirror_declares_the_placeholder_explicitly(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._ready(tmp_path, monkeypatch)

        dist.main(self._argv(tmp_path, dist.ALLOW_PLACEHOLDER_FLAG))

        parts = self._text_encoder_refs(tmp_path)
        assert {ref["revision"] for ref in parts} == {dist.PLACEHOLDER_REVISION}

    def test_a_real_revision_needs_no_declaration(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._ready(tmp_path, monkeypatch)

        dist.main(self._argv(tmp_path, "--ref-revision", f"{UMT5_REPO_NAME}={_UMT5_REVISION}"))

        assert verify_dist(tmp_path / "models" / WAN_REPO_NAME)
        assert {ref["revision"] for ref in self._text_encoder_refs(tmp_path)} == {_UMT5_REVISION}

    def test_a_build_without_a_revision_source_is_refused(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No silent self-contained build of a pipeline that declares a reference."""
        self._ready(tmp_path, monkeypatch)

        with pytest.raises(DistError, match=r"reference \['karume-umt5-xxl'\] across"):
            dist.main(self._argv(tmp_path))
        assert not (tmp_path / "models" / WAN_REPO_NAME).exists()

    def test_the_hub_pins_the_referenced_main_and_confirms_the_borrowed_files(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        umt5_dir = self._ready(tmp_path, monkeypatch)

        dist.main(self._argv(tmp_path, "--resolve-refs", "hf"), get=self._hub(umt5_dir))

        assert {ref["revision"] for ref in self._text_encoder_refs(tmp_path)} == {
            self._HUB_REVISION
        }

    def test_a_borrowed_file_with_other_bytes_on_the_hub_is_refused_before_anything_is_written(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A referenced mirror rebuilt locally but not uploaded: one part differs on the Hub."""
        umt5_dir = self._ready(tmp_path, monkeypatch)
        parts = sorted((umt5_dir / UMT5_DEFAULT_MODEL / WAN_TEXT_ENCODER_ROLE).glob("*.krm"))
        stale = str(parts[1].relative_to(umt5_dir))

        with pytest.raises(DistError, match=rf"1 of \d+ files\):\n  {re.escape(stale)}: LFS oid"):
            dist.main(
                self._argv(tmp_path, "--resolve-refs", "hf"),
                get=self._hub(umt5_dir, oids={stale: "f" * 64}),
            )
        assert not (tmp_path / "models" / WAN_REPO_NAME).exists()

    def test_the_sources_exclude_each_other(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        umt5_dir = self._ready(tmp_path, monkeypatch)

        with pytest.raises(DistError, match="exclude each other"):
            dist.main(
                self._argv(tmp_path, "--resolve-refs", "hf", dist.ALLOW_PLACEHOLDER_FLAG),
                get=self._hub(umt5_dir),
            )
        with pytest.raises(DistError, match="exclude each other"):
            dist.main(
                self._argv(
                    tmp_path,
                    "--resolve-refs=hf",
                    "--ref-revision",
                    f"{UMT5_REPO_NAME}={_UMT5_REVISION}",
                ),
                get=self._hub(umt5_dir),
            )
        assert not (tmp_path / "models" / WAN_REPO_NAME).exists()

    def test_an_abbreviated_declaration_is_not_read(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Neither the check nor the core accepts an abbreviated form (explicit means spelled out in
        full — the doc of `build_driver_parser`)."""
        self._ready(tmp_path, monkeypatch)

        with pytest.raises(SystemExit):
            dist.main(self._argv(tmp_path, "--allow-placeholder"))
        assert not (tmp_path / "models" / WAN_REPO_NAME).exists()

    def test_the_placeholder_is_forty_zeros(self) -> None:
        """The form passes hub's parse (40 lowercase hex digits) — which is why the shape check does
        not stop it."""
        assert dist.PLACEHOLDER_REVISION == "0" * 40
        declared = wan_references(DEFAULT_MODEL)
        pins = partial(dist.reference_pins, resolve=None, get=self._unreachable)
        assert pins(declared, {}, allow_placeholder=True) == (
            {UMT5_REPO_NAME: dist.PLACEHOLDER_REVISION},
            None,
        )
        assert pins((), {}, allow_placeholder=False) == ({}, None)
        assert pins(declared, {UMT5_REPO_NAME: _UMT5_REVISION}, allow_placeholder=False) == (
            {},
            None,
        )

    @staticmethod
    def _unreachable(url: str) -> tuple[Any, str | None]:
        raise AssertionError(f"no Hub access expected: {url}")


class TestTheWritersSpellTheSameNames:
    """The distribution recipe does not import torch, so it holds the spellings itself — they match
    the writers (which import torch)."""

    def test_the_series_names(self) -> None:
        from wan import export_dit, export_vae, text_embeds

        assert export_dit.SERIES.name == WAN_SERIES
        assert export_dit.I8_SERIES.name == WAN_I8_SERIES
        assert export_vae.SERIES_NAME == WAN_SERIES
        assert text_embeds.SERIES_NAME == WAN_TEXT_EMBEDS_SERIES

    def test_the_embedding_asset_format(self) -> None:
        from wan import text_embeds

        assert text_embeds.ASSET_NAME == WAN_TEXT_EMBEDS_FILE
        assert text_embeds.METADATA_KEY == WAN_TEXT_EMBEDS_METADATA_KEY

    def test_the_encoder_dtype_the_writer_records(self) -> None:
        from wan import text_embeds

        metadata = text_embeds.asset_metadata(
            (), {}, {}, model=DEFAULT_MODEL, versions=WAN_TEXT_EMBEDS_VERSIONS
        )
        assert metadata["text_encoder"]["dtype"] == WAN_TEXT_ENCODER_DTYPE

    def test_the_pinned_versions_match_the_wan_dependency_group(self) -> None:
        """The pinned values of the check equal the `==` pins of the `wan` group in pyproject.

        The pins are the source of the versions the writers record (the versions
        `uv run --group wan` installs).
        """
        project = tomllib.loads((REPO_ROOT / "tools/export-recipes/pyproject.toml").read_text())
        pins = dict(
            requirement.split("==", 1)
            for requirement in project["dependency-groups"]["wan"]
            if "==" in requirement
        )
        assert {name: pins.get(name) for name in WAN_TEXT_EMBEDS_VERSIONS} == dict(
            WAN_TEXT_EMBEDS_VERSIONS
        )

    def test_the_versions_the_writer_records_in_this_environment(self) -> None:
        for package in ("diffusers", "ftfy", "transformers"):
            pytest.importorskip(package)
        from wan import text_embeds

        recorded = text_embeds._versions()
        assert {name: recorded[name] for name in WAN_TEXT_EMBEDS_VERSIONS} == dict(
            WAN_TEXT_EMBEDS_VERSIONS
        )

    def test_the_vae_latent_input(self) -> None:
        from wan import export_vae

        assert export_vae.LATENT_INPUT == WAN_VAE_LATENT_INPUT

    def test_the_vae_tiles_and_the_32_tile_graph_names(self) -> None:
        """The default-tile set uses the tiles of the writer's table, and the graph names of the 32
        set are the spelling of the writer's `--shared` (ADR 0134)."""
        from wan import export_vae

        assert WAN21.vae_tiles[0] == export_vae.VAE_SERIES[DEFAULT_MODEL].tile
        assert WAN22.vae_tiles[0] == export_vae.VAE_SERIES[_TI2V].tile
        assert export_vae.shared_target(WAN_VAE_FIRST_ROLE, 32) == WAN_VAE_FIRST_32_ROLE
        assert export_vae.shared_target(WAN_VAE_NEXT_ROLE, 32) == WAN_VAE_NEXT_32_ROLE

    def test_the_rope_base_asset_and_the_context_input(self) -> None:
        from wan import export_dit

        assert (export_dit.ROPE_BASE_ASSET, export_dit.ROPE_BASE_ROLE) == (
            WAN_ROPE_BASE_ASSET,
            WAN_ROPE_BASE_ROLE,
        )
        assert export_dit.MODEL_FILE == WAN_MODEL_FILE
        assert WAN_DIT_CONTEXT_INPUT in export_dit.INPUT_NAMES

    def test_the_wan22_series_names(self) -> None:
        """The i8 series of the Wan2.2 DiT and the f16 series of the VAE (writers
        `wan.ti2v_export_dit` / `wan.export_vae --model ti2v-5b`)."""
        from wan import export_vae, ti2v_export_dit

        assert ti2v_export_dit.SERIES_NAME == WAN22_I8_SERIES
        assert ti2v_export_dit.SERIES.name == WAN22_I8_SERIES
        assert export_vae.TI2V_SERIES_NAME == WAN22_SERIES
        assert export_vae.VAE_SERIES[_TI2V].series == WAN22_SERIES
        assert ti2v_export_dit.MODEL_FILE == WAN_MODEL_FILE
        assert WAN_DIT_CONTEXT_INPUT in ti2v_export_dit.INPUT_NAMES

    def test_the_wan22_vae_encoder_graphs(self) -> None:
        """The 3 graphs of the VAE encoder (writer `wan.export_vae_encoder` — the same f16 series as
        the decoder)."""
        from wan import export_vae_encoder

        assert export_vae_encoder.TARGETS == WAN_VAE_ENCODER_ROLES
        assert export_vae_encoder.SERIES_NAME == WAN22_SERIES == WAN22.series
        assert export_vae_encoder.MODEL_FILE == WAN_MODEL_FILE


class TestTheGenerationTables:
    """The placements, storage and weights tables derived from the generation tables (Wan2.1 is the
    all-generation table without the Wan2.2-only VAE encoder, VAE decoder i8 and 32-latent-tile set
    — the 2.1 distribution is unchanged)."""

    def test_the_wan21_tables_are_the_full_tables_without_the_wan22_only_roles(self) -> None:
        encoder = {
            *WAN_VAE_ENCODER_ROLES,
            WAN_VAE_FIRST_I8_ROLE,
            WAN_VAE_NEXT_I8_ROLE,
            *WAN_VAE_LENDER_ROLES,
        }
        assert WAN21.vae_encoder_roles == ()
        # The Wan2.1 decoder has only the default-tile (32) set with weights (the tile of the
        # writer's table).
        assert WAN21.vae_tiles == (32,)
        assert WAN21.container_roles == (
            WAN_TEXT_ENCODER_ROLE,
            WAN_TRANSFORMER_F16_ROLE,
            WAN_TRANSFORMER_I8_ROLE,
            WAN_VAE_FIRST_ROLE,
            WAN_VAE_NEXT_ROLE,
        )
        assert WAN21.container_roles == tuple(
            role for role in WAN_CONTAINER_ROLES if role not in encoder
        )
        assert WAN21.transformer_roles == (WAN_TRANSFORMER_F16_ROLE, WAN_TRANSFORMER_I8_ROLE)
        assert WAN21.storage_requirements == {
            role: required
            for role, required in WAN_STORAGE_REQUIREMENTS.items()
            if role not in encoder
        }
        assert WAN21.storage_forbidden == {
            role: forbidden
            for role, forbidden in WAN_STORAGE_FORBIDDEN.items()
            if role not in encoder
        }
        assert WAN21.weights == {
            name: {label: files for label, files in labels.items() if files.file not in encoder}
            for name, labels in WAN_WEIGHTS.items()
            if name not in encoder
        }
        assert (WAN21.vae_labels, WAN21.vae_decoder_roles) == (
            ("f16",),
            (WAN_VAE_FIRST_ROLE, WAN_VAE_NEXT_ROLE),
        )
        assert (WAN21.repo_name, WAN21.pipeline, WAN21.series, WAN21.i8_series) == (
            WAN_REPO_NAME,
            WAN_PIPELINE,
            WAN_SERIES,
            WAN_I8_SERIES,
        )
        assert WAN21.pipeline_config == WAN_PIPELINE_CONFIG
        assert PIPELINE.root_files["NOTICE.md"] == WAN21.notice

    def test_the_wan22_tables_hold_only_the_int8_transformer(self) -> None:
        assert WAN22.container_roles == (
            *WAN_VAE_ENCODER_ROLES,
            WAN_TEXT_ENCODER_ROLE,
            WAN_TRANSFORMER_I8_ROLE,
            WAN_VAE_FIRST_ROLE,
            WAN_VAE_FIRST_I8_ROLE,
            WAN_VAE_NEXT_ROLE,
            WAN_VAE_NEXT_I8_ROLE,
            WAN_VAE_FIRST_32_ROLE,
            WAN_VAE_FIRST_32_I8_ROLE,
            WAN_VAE_NEXT_32_ROLE,
            WAN_VAE_NEXT_32_I8_ROLE,
        )
        assert list(WAN22.weights[WAN_TRANSFORMER_ROLE]) == ["i8"]
        assert set(WAN22.storage_requirements) == set(WAN22.container_roles)
        # The mix-up of plugging the f16 DiT into the i8 preset fails on both the required and the
        # forbidden side.
        assert "f16" in WAN22.storage_forbidden[WAN_TRANSFORMER_I8_ROLE]
        assert "f16" not in WAN22.quants

    def test_only_wan22_carries_the_vae_encoder_in_f16(self) -> None:
        """Only Wan2.2 distributes the 3 encoder graphs (ADR 0121 decision 11), treated like the
        decoder: the f16 series, one file with storage label f16, `<component>/model.f16.krm`, f16
        required and other compressed storage forbidden."""
        assert (WAN21.vae_encoder, WAN22.vae_encoder) == (False, True)
        assert WAN_VAE_ENCODER_ROLES == (
            WAN_VAE_ENCODER_PRE_ROLE,
            WAN_VAE_ENCODER_ATTN_ROLE,
            WAN_VAE_ENCODER_POST_ROLE,
        )
        assert WAN22.vae_encoder_roles == WAN_VAE_ENCODER_ROLES
        for role in WAN_VAE_ENCODER_ROLES:
            assert {label: files.file for label, files in WAN22.weights[role].items()} == {
                "f16": role
            }
            assert WAN_OUTPUT_PATHS[role] == f"{role}/model.f16.krm"
            assert WAN22.storage_requirements[role] == "f16"
            assert WAN22.storage_forbidden[role] == WAN_STORAGE_FORBIDDEN[WAN_VAE_FIRST_ROLE]
            assert role not in WAN21.weights
        # The order of a generation's weights is the order of the all-generation table (the order of
        # the I2V steps — the encoder first).
        assert list(WAN22.weights) == list(WAN_GRAPH_ROLES)

    def test_only_wan22_carries_the_vae_decoder_in_int8(self) -> None:
        """ADR 0133 decision 7: only in Wan2.2 do the 2 decoder graphs come as 2 each, f16 / i8. i8
        is `<graph name>/model.i8.krm`, with i8 required and other compressed storage (f16
        included) forbidden."""
        assert WAN22.vae_labels == ("f16", "i8")
        assert WAN22.vae_decoder_roles == (
            WAN_VAE_FIRST_ROLE,
            WAN_VAE_FIRST_I8_ROLE,
            WAN_VAE_NEXT_ROLE,
            WAN_VAE_NEXT_I8_ROLE,
            # The 32-latent-tile set (ADR 0134) follows the default-tile set in the same order.
            WAN_VAE_FIRST_32_ROLE,
            WAN_VAE_FIRST_32_I8_ROLE,
            WAN_VAE_NEXT_32_ROLE,
            WAN_VAE_NEXT_32_I8_ROLE,
        )
        assert WAN_OUTPUT_PATHS[WAN_VAE_FIRST_I8_ROLE] == "vae_decoder_first/model.i8.krm"
        assert WAN_OUTPUT_PATHS[WAN_VAE_NEXT_I8_ROLE] == "vae_decoder_next/model.i8.krm"
        for graph, i8_role in (
            (WAN_VAE_FIRST_ROLE, WAN_VAE_FIRST_I8_ROLE),
            (WAN_VAE_NEXT_ROLE, WAN_VAE_NEXT_I8_ROLE),
        ):
            assert {label: files.file for label, files in WAN22.weights[graph].items()} == {
                "f16": graph,
                "i8": i8_role,
            }
            assert WAN22.storage_requirements[i8_role] == "i8"
            assert "f16" in WAN22.storage_forbidden[i8_role]
            assert i8_role not in WAN21.container_roles

    def test_the_int8_vae_decoder_comes_from_the_i8_series(self, tmp_path: Path) -> None:
        """The i8 decoder is `<graph name>/model.krm` at the same series root as the DiT i8
        container (writer `wan.export_vae --model ti2v-5b --dtype i8`)."""
        sources = wan_sources(tmp_path, WAN22)
        placements = wan_placements(sources, WAN22)
        assert placements[WAN_VAE_FIRST_I8_ROLE] == (
            tmp_path / WAN22_I8_SERIES / WAN_VAE_FIRST_ROLE / WAN_MODEL_FILE
        )
        assert placements[WAN_VAE_NEXT_I8_ROLE] == (
            tmp_path / WAN22_I8_SERIES / WAN_VAE_NEXT_ROLE / WAN_MODEL_FILE
        )
        assert placements[WAN_VAE_FIRST_ROLE] == (
            tmp_path / WAN22_SERIES / WAN_VAE_FIRST_ROLE / WAN_MODEL_FILE
        )
        assert not {WAN_VAE_FIRST_I8_ROLE, WAN_VAE_NEXT_I8_ROLE} & set(
            wan_placements(wan_sources(tmp_path), WAN21)
        )

    def test_only_wan22_carries_the_32_tile_pair_per_storage_label(self, tmp_path: Path) -> None:
        """ADR 0134 decision 5: the 32-latent-tile set exists only in Wan2.2, one set per storage
        label, with roles under the same rules as the decoder (f16 is the graph name, i8 has the
        `_i8` suffix), `<graph name>/model.<label>.krm`, and `<graph name>/model.krm` at the same
        series root as the lender (the 16 set of the same label)."""
        assert WAN22.vae_tiles == (16, 32)
        assert (WAN_VAE_FIRST_32_ROLE, WAN_VAE_NEXT_32_ROLE) == (
            "vae_decoder_first_32",
            "vae_decoder_next_32",
        )
        assert (WAN_VAE_FIRST_32_I8_ROLE, WAN_VAE_NEXT_32_I8_ROLE) == (
            "vae_decoder_first_32_i8",
            "vae_decoder_next_32_i8",
        )
        assert WAN_VAE_LENDER_ROLES == {
            WAN_VAE_FIRST_32_ROLE: WAN_VAE_FIRST_ROLE,
            WAN_VAE_FIRST_32_I8_ROLE: WAN_VAE_FIRST_I8_ROLE,
            WAN_VAE_NEXT_32_ROLE: WAN_VAE_NEXT_ROLE,
            WAN_VAE_NEXT_32_I8_ROLE: WAN_VAE_NEXT_I8_ROLE,
        }
        placements = wan_placements(wan_sources(tmp_path, WAN22), WAN22)
        for graph, f16_role, i8_role in (
            (WAN_VAE_FIRST_32_ROLE, WAN_VAE_FIRST_32_ROLE, WAN_VAE_FIRST_32_I8_ROLE),
            (WAN_VAE_NEXT_32_ROLE, WAN_VAE_NEXT_32_ROLE, WAN_VAE_NEXT_32_I8_ROLE),
        ):
            assert {label: files.file for label, files in WAN22.weights[graph].items()} == {
                "f16": f16_role,
                "i8": i8_role,
            }
            assert WAN_OUTPUT_PATHS[f16_role] == f"{graph}/model.f16.krm"
            assert WAN_OUTPUT_PATHS[i8_role] == f"{graph}/model.i8.krm"
            assert placements[f16_role] == tmp_path / WAN22_SERIES / graph / WAN_MODEL_FILE
            assert placements[i8_role] == tmp_path / WAN22_I8_SERIES / graph / WAN_MODEL_FILE
            assert WAN22.storage_requirements[f16_role] == "f16"
            assert WAN22.storage_requirements[i8_role] == "i8"
            assert graph not in WAN21.weights
        assert not set(WAN_VAE_LENDER_ROLES) & set(WAN21.container_roles)
        assert not set(WAN_VAE_LENDER_ROLES) & set(wan_placements(wan_sources(tmp_path), WAN21))

    def test_the_driver_offers_wan_ti2v_right_after_wan(self) -> None:
        names = list(dist.PIPELINES)
        assert names[names.index("wan") + 1] == "wan-ti2v"
        assert dist.PIPELINES["wan-ti2v"] is TI2V_PIPELINE


class TestTheWan22Distribution:
    """The distribution of `--pipeline wan-ti2v` (ADR 0121 step 8 — `karume-wan2.2`, model
    `ti2v-5b`)."""

    def test_it_places_the_int8_transformer_the_vae_and_the_assets(self, assembled22) -> None:
        """The transformer is a single i8 file, the VAE decoder 2 each of f16 / i8 (ADR 0133), the
        32-latent-tile set 2 each of f16 / i8 (ADR 0134 — only the 2-document part 0 and an empty
        const region), the encoder 1 each, and 2 assets. The text_encoder is a cross-repository
        reference. The encoder goldens (`vae_encoder.*`) are not distributed."""
        out_dir, _ = assembled22
        own = {
            role: WAN_OUTPUT_PATHS[role]
            for role in (
                *WAN_VAE_ENCODER_ROLES,
                WAN_TRANSFORMER_I8_ROLE,
                WAN_VAE_FIRST_ROLE,
                WAN_VAE_FIRST_I8_ROLE,
                WAN_VAE_NEXT_ROLE,
                WAN_VAE_NEXT_I8_ROLE,
                *WAN_VAE_LENDER_ROLES,
                WAN_TEXT_EMBEDS_ROLE,
                WAN_TOKENIZER_ROLE,
            )
        }
        expected = [
            f"{_TI2V}/{rel}"
            for rel in placed_paths(
                own,
                {name: labels for name, labels in WAN22.weights.items() if name != UMT5_ROLE},
                {WAN_TRANSFORMER_I8_ROLE: 4, **dict.fromkeys(WAN_VAE_LENDER_ROLES, 2)},
            )
        ]
        assert _present(out_dir) == sorted(
            [*expected, MANIFEST_FILENAME, MODEL_CARD_FILENAME, "LICENSE.md", NOTICE_FILENAME]
        )
        assert list((out_dir / _TI2V / WAN_TRANSFORMER_ROLE).glob("model.f16*")) == []
        for graph in (WAN_VAE_FIRST_ROLE, WAN_VAE_NEXT_ROLE):
            assert list((out_dir / _TI2V / graph).glob("model.i8*"))
        assert list(out_dir.rglob(f"{WAN_TEXT_ENCODER_ROLE}/*")) == []
        assert list(out_dir.rglob("io.*")) == []
        assert list(out_dir.rglob("pipeline_steps.*")) == []
        assert list(out_dir.rglob("vae_encoder.*")) == []

    def test_the_manifest_declares_the_five_seats(self, assembled22) -> None:
        """Five presets: the reference, the practical (s16) and the online-attention preset (ADR
        0132 decision 6) on the f16 VAE decoder, then the int8 VAE decoder's reference `i8` and
        practical `i8-a8-attn8-online-conv8` (ADR 0133 decision 6); no f16 preset; the default is
        the conv8 preset since 2026-10-23 (ADR 0133 decision 8 — the decoder A/B, the timing and the
        visual check; the online and s16 presets stay as siblings). Every preset takes the f16 VAE
        encoder (even for T2V alone — the ruling in the addendum on step 9's plan and step 9a's
        results). The weights are ordered as the I2V stages."""
        _, manifest = assembled22
        model = manifest["models"][_TI2V]
        assert manifest["defaultModel"] == _TI2V
        assert list(manifest["models"]) == [_TI2V]
        assert model["pipeline"] == "wan-ti2v/1" == WAN22_PIPELINE
        assert list(model["weights"]) == [
            WAN_VAE_ENCODER_PRE_ROLE,
            WAN_VAE_ENCODER_ATTN_ROLE,
            WAN_VAE_ENCODER_POST_ROLE,
            WAN_TEXT_ENCODER_ROLE,
            WAN_TRANSFORMER_ROLE,
            WAN_VAE_FIRST_ROLE,
            WAN_VAE_NEXT_ROLE,
            # The 32-tile pair (ADR 0134 decision 5) after the 16-tile pair it borrows from.
            WAN_VAE_FIRST_32_ROLE,
            WAN_VAE_NEXT_32_ROLE,
        ]
        for role in WAN_VAE_ENCODER_ROLES:
            assert list(model["weights"][role]) == ["f16"]
        assert list(model["weights"][WAN_TEXT_ENCODER_ROLE]) == ["i8"]
        assert list(model["weights"][WAN_TRANSFORMER_ROLE]) == ["i8"]
        for graph in (
            WAN_VAE_FIRST_ROLE,
            WAN_VAE_NEXT_ROLE,
            WAN_VAE_FIRST_32_ROLE,
            WAN_VAE_NEXT_32_ROLE,
        ):
            assert list(model["weights"][graph]) == ["f16", "i8"]
        assert model["assets"][WAN_TEXT_EMBEDS_ROLE]["path"] == (
            f"{_TI2V}/{WAN_TEXT_EMBEDS_ROLE}/{WAN_TEXT_EMBEDS_FILE}"
        )
        assert model["assets"][WAN_TOKENIZER_ROLE]["path"] == (
            f"{_TI2V}/{WAN_TOKENIZER_ROLE}/{umt5_tokenizer.ASSET_FILE}"
        )
        assert list(model["quants"]) == [_REFERENCE, _PRACTICAL, _ONLINE, _I8, _CONV8]
        assert model["defaultQuant"] == _CONV8
        seat = {
            **dict.fromkeys(WAN_VAE_ENCODER_ROLES, "f16"),
            WAN_TEXT_ENCODER_ROLE: "i8",
            WAN_TRANSFORMER_ROLE: "i8",
            WAN_VAE_FIRST_ROLE: "f16",
            WAN_VAE_NEXT_ROLE: "f16",
            # Every preset maps the 32-tile pair with the decoder's label (ADR 0134 decision 5).
            WAN_VAE_FIRST_32_ROLE: "f16",
            WAN_VAE_NEXT_32_ROLE: "f16",
        }
        assert model["quants"][_REFERENCE]["weights"] == seat
        assert model["quants"][_PRACTICAL]["weights"] == seat
        assert model["quants"][_ONLINE]["weights"] == seat
        assert model["quants"][_REFERENCE]["session"] == {}
        assert model["quants"][_PRACTICAL]["session"] == {
            "linearCompute": "a8",
            "attentionCompute": "a8",
            "attentionScoreStorage": "f16",
        }
        assert model["quants"][_ONLINE]["session"] == {
            "linearCompute": "a8",
            "attentionCompute": "a8",
            "attentionScoreStorage": "none",
        }
        assert model["quants"][_ONLINE]["label"] == (
            "int8 transformer, int8 activations, online attention"
        )
        assert model["quants"][_ONLINE]["description"] == (
            "The int8 transformer with per-token int8 activations and attention computed online:"
            " the scores are never stored, so the long clips run faster on the same weights as the"
            " s16 preset."
        )
        int8_vae = {
            **seat,
            WAN_VAE_FIRST_ROLE: "i8",
            WAN_VAE_NEXT_ROLE: "i8",
            WAN_VAE_FIRST_32_ROLE: "i8",
            WAN_VAE_NEXT_32_ROLE: "i8",
        }
        assert model["quants"][_I8]["weights"] == int8_vae
        assert model["quants"][_CONV8]["weights"] == int8_vae
        assert model["quants"][_I8]["session"] == {}
        assert model["quants"][_CONV8]["session"] == {
            "linearCompute": "a8",
            "attentionCompute": "a8",
            "attentionScoreStorage": "none",
            "convCompute": "a8",
        }
        assert model["quants"][_I8]["label"] == "int8 transformer and VAE decoder, f32 compute"
        assert model["quants"][_CONV8]["label"] == (
            "int8 transformer and VAE, int8 activations, online attention"
        )

    def test_the_pipeline_config_is_the_official_720p_setting(self, assembled22) -> None:
        _, manifest = assembled22
        config = manifest["models"][_TI2V]["pipelineConfig"]
        assert {key: tuple(value) for key, value in config.items()} == _CONFIG_KEYS
        assert config == {"scheduler": {"shift": 5.0}, "defaults": {"steps": 50, "guidance": 5.0}}

    def test_the_repository_and_the_default_model(self) -> None:
        assert TI2V_PIPELINE.default_model == _TI2V
        assert TI2V_PIPELINE.repo_name(_TI2V) == "karume-wan2.2" == WAN22_REPO_NAME
        assert PIPELINE.repo_name(DEFAULT_MODEL) == WAN_REPO_NAME

    def test_it_reassembles_to_the_same_bytes(self, tmp_path: Path, wan22_provenance) -> None:
        sources = _build_sources22(tmp_path)
        out_dir = tmp_path / "models" / WAN22_REPO_NAME

        def digest() -> dict[str, bytes]:
            return {path: (out_dir / path).read_bytes() for path in _present(out_dir)}

        first = assemble_family([wan_plan(sources, generation=WAN22)], out_dir, _TI2V)
        before = digest()
        assert first == assemble_family([wan_plan(sources, generation=WAN22)], out_dir, _TI2V)
        assert digest() == before
        assert verify_dist(out_dir)

    def test_the_text_assets_are_the_wan21_series_files(self, tmp_path: Path, assembled22) -> None:
        """The 2 assets are the very files of the Wan2.1 series (ADR 0121 decision 9 —
        byte-identical in the distribution too)."""
        out_dir, _ = assembled22
        series = tmp_path / "outputs" / "series"
        wan21, wan22 = wan_sources(series, WAN21), wan_sources(series, WAN22)
        assert (wan22.text_embeds, wan22.tokenizer) == (wan21.text_embeds, wan21.tokenizer)
        placed = out_dir / _TI2V
        assert (placed / WAN_OUTPUT_PATHS[WAN_TEXT_EMBEDS_ROLE]).read_bytes() == (
            wan21.text_embeds.read_bytes()
        )
        assert (placed / WAN_OUTPUT_PATHS[WAN_TOKENIZER_ROLE]).read_bytes() == (
            wan21.tokenizer.read_bytes()
        )

    def test_the_repository_ships_the_apache_license_and_the_wan22_notice(
        self, assembled22
    ) -> None:
        out_dir, _ = assembled22
        assert (out_dir / "LICENSE.md").read_bytes() == APACHE_LICENSE_2_0_PATH.read_bytes()
        prose = " ".join((out_dir / NOTICE_FILENAME).read_text(encoding="utf-8").split())
        assert "modified form of the Wan2.2 TI2V 5B checkpoint" in prose
        assert "Apache License, Version 2.0" in prose
        assert "**int8 transformer**: the transformer is distributed only in this form." in prose
        assert "No float16 or float32 copy of the transformer is distributed." in prose
        assert "**f16 VAE decoder**" in prose
        # The 2 condition inputs explain both T2V and I2V (ADR 0121 step 9c — I2V became usable):
        # for I2V the tokens of the first latent frame are true and their time is 0
        # (`wanConditionMask` / `timestepsProj(0)` of dit-loop.ts).
        assert (
            "In text-to-video the mask is all false and the second timestep equals the first."
            in prose
        )
        assert (
            "In image-to-video the mask is true for the tokens of the first latent frame, which the"
            " host replaces with the encoded conditioning image, and the second timestep is 0."
            in prose
        )
        assert "the only mode this distribution runs" not in prose
        assert "(`patch_size` 2, 12 channels)" in prose
        # The VAE encoder is in the distribution — the modifications (3 graphs, the last time
        # slice, f16 rounding) are announced.
        assert "**f16 VAE encoder for image-to-video**" in prose
        assert "re-expressed as three graphs" in prose
        assert "so only that slice is stored" in prose
        assert "The VAE encoder is not included." not in prose
        assert "**The text encoder is referenced, not stored here.**" in prose
        assert f"`{UMT5_REPO_NAME}` at a pinned commit" in prose
        assert "umT5-XXL encoder of the Wan2.1 T2V 1.3B checkpoint" in prose
        # A copy of the 2.1 sentences with facts that differ for 2.2 does not remain.
        assert "float16 transformer" not in prose
        assert "second copy of the transformer" not in prose
        assert "Wan2.1 T2V 1.3B checkpoint listed" not in prose


class TestTheWan22ProvenanceSplit:
    """The split of the provenance check (ADR 0121 decision 9): the DiT / VAE containers are checked
    against the Wan2.2 pin, the 2 text assets against the Wan2.1 pin. A mix-up fails in either
    direction."""

    def test_the_split_pins_pass(self, tmp_path: Path, wan22_provenance) -> None:
        plan = wan_plan(_build_sources22(tmp_path), generation=WAN22)
        assert plan.name == _TI2V
        assert plan.pipeline == WAN22_PIPELINE

    def test_it_refuses_embeddings_that_name_the_wan22_pin(
        self, tmp_path: Path, wan22_provenance
    ) -> None:
        sources = _build_sources22(tmp_path, embeds=_text_embeds(source=_pin(_TI2V)))
        with pytest.raises(
            DistError, match=r"the embeddings' provenance .* differs from the upstream pin"
        ):
            wan_plan(sources, generation=WAN22)

    def test_it_refuses_a_tokenizer_that_names_the_wan22_pin(
        self, tmp_path: Path, wan22_provenance
    ) -> None:
        source = {**_pin(_TI2V), "subfolder": WAN_TOKENIZER_SUBFOLDER}
        sources = _build_sources22(tmp_path, tokenizer=_tokenizer_asset(source=source))
        with pytest.raises(
            DistError, match=r"the tokenizer's provenance .* differs from the upstream pin"
        ):
            wan_plan(sources, generation=WAN22)

    @pytest.mark.parametrize(
        "role",
        [
            WAN_TRANSFORMER_I8_ROLE,
            WAN_VAE_FIRST_ROLE,
            WAN_VAE_NEXT_ROLE,
            WAN_VAE_FIRST_I8_ROLE,
            WAN_VAE_NEXT_I8_ROLE,
            *WAN_VAE_ENCODER_ROLES,
        ],
    )
    def test_it_refuses_a_container_baked_from_the_wan21_pin(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, wan22_provenance, role: str
    ) -> None:
        container = _stamped(monkeypatch, _PINNED, lambda: _default_container(role))
        sources = _build_sources22(tmp_path, containers={role: container})
        injected = re.escape(str(wan_placements(sources, WAN22)[role].parent))
        with pytest.raises(DistError, match=rf"{injected}.*a different revision"):
            wan_plan(sources, generation=WAN22)

    def test_the_wan21_plan_still_reads_both_from_the_wan21_pin(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The pair: the Wan2.1 plan checks the DiT / VAE and the text assets against the Wan2.1 pin
        (a DiT with the Wan2.2 pin fails)."""
        assert WAN21.text_model == DEFAULT_MODEL
        dit = _stamped(
            monkeypatch,
            _PINNED22,
            lambda: _graph_container(WAN_TRANSFORMER_ROLE, storage="i8"),
        )
        sources = _build_sources(tmp_path, containers={WAN_TRANSFORMER_I8_ROLE: dit})
        with pytest.raises(DistError, match="a different revision"):
            wan_plan(sources)


class TestTheWan22Gates:
    def test_it_refuses_an_f16_transformer_in_the_int8_seat(
        self, tmp_path: Path, wan22_provenance
    ) -> None:
        sources = _build_sources22(
            tmp_path,
            containers={WAN_TRANSFORMER_I8_ROLE: _graph_container(WAN_TRANSFORMER_ROLE)},
        )
        with pytest.raises(DistError, match=r"transformer_i8: .* has no i8"):
            wan_plan(sources, generation=WAN22)

    def test_it_refuses_a_mixed_i4_series_in_the_int8_seat(
        self, tmp_path: Path, wan22_provenance
    ) -> None:
        """A mixed series that satisfies "contains i8" — only the forbidden table rejects it (the
        same check as 2.1)."""
        sources = _build_sources22(
            tmp_path,
            containers={
                WAN_TRANSFORMER_I8_ROLE: _graph_container(WAN_TRANSFORMER_ROLE, storage="i4")
            },
        )
        with pytest.raises(DistError, match=r"transformer_i8: .* has i4"):
            wan_plan(sources, generation=WAN22)

    def test_an_f16_transformer_beside_the_vae_is_never_carried(
        self, tmp_path: Path, wan22_provenance
    ) -> None:
        """Even with a DiT in the f16 series, the Wan2.2 placement table does not pick it up (the
        f16 DiT is not distributed)."""
        sources = _build_sources22(tmp_path)
        write_component(
            sources.series / WAN_TRANSFORMER_ROLE / WAN_MODEL_FILE,
            _graph_container(WAN_TRANSFORMER_ROLE),
        )
        plan = wan_plan(sources, generation=WAN22)
        assert set(plan.artifacts) == {
            *WAN22.container_roles,
            WAN_TEXT_EMBEDS_ROLE,
            WAN_TOKENIZER_ROLE,
        }
        assert WAN_TRANSFORMER_F16_ROLE not in plan.artifacts

    @pytest.mark.parametrize("role", [WAN_VAE_FIRST_I8_ROLE, WAN_VAE_NEXT_I8_ROLE])
    def test_it_refuses_an_f16_vae_decoder_in_the_int8_seat(
        self, tmp_path: Path, wan22_provenance, role: str
    ) -> None:
        """The mix-up of plugging an f16-series decoder into the i8 preset (fails because the
        requirement is absent — ADR 0133 decision 7)."""
        sources = _build_sources22(
            tmp_path, containers={role: _graph_container(_VAE_I8_GRAPHS[role])}
        )
        with pytest.raises(DistError, match=rf"{role}: .* has no i8"):
            wan_plan(sources, generation=WAN22)

    @pytest.mark.parametrize("role", WAN_VAE_ENCODER_ROLES)
    @pytest.mark.parametrize("storage", ["f32", "i8"])
    def test_it_refuses_a_vae_encoder_without_f16_storage(
        self, tmp_path: Path, wan22_provenance, role: str, storage: str
    ) -> None:
        """The mix-up of plugging a plain f32 / other-storage series into the encoder preset (the
        same check as the decoder)."""
        sources = _build_sources22(
            tmp_path, containers={role: _graph_container(role, storage=storage)}
        )
        with pytest.raises(DistError, match=rf"{role}: .* has no f16"):
            wan_plan(sources, generation=WAN22)

    @pytest.mark.parametrize("role", WAN_VAE_ENCODER_ROLES)
    def test_it_refuses_a_series_without_one_of_the_vae_encoder_graphs(
        self, tmp_path: Path, wan22_provenance, role: str
    ) -> None:
        """The encoder is a component fetched even for T2V-only use — a series missing even one is
        not assembled."""
        sources = _build_sources22(tmp_path)
        missing = wan_placements(sources, WAN22)[role]
        for part in missing.parent.iterdir():
            part.unlink()
        missing.parent.rmdir()
        with pytest.raises(DistError, match=rf"Assembly input is missing: .*{role}"):
            wan_plan(sources, generation=WAN22)

    @pytest.mark.parametrize("role", sorted(WAN_VAE_LENDER_ROLES))
    def test_every_32_tile_container_borrows_everything_from_its_lender(
        self, tmp_path: Path, wan22_provenance, role: str
    ) -> None:
        """ADR 0134 decision 1: the borrower holds not a single byte outside part 0, every
        initializer is a shared declaration, and the lender (the 16 set of the same label) holds
        the named keys as data with the same dtype and shape."""
        placements = wan_placements(_build_sources22(tmp_path), WAN22)
        borrower = placements[role]

        assert_shared_decoder(borrower, placements[WAN_VAE_LENDER_ROLES[role]])
        parts = container_parts(borrower)
        assert [part.stat().st_size == 0 for part in parts] == [False, *[True] * (len(parts) - 1)]

    @pytest.mark.parametrize("role", [WAN_VAE_FIRST_32_ROLE, WAN_VAE_NEXT_32_I8_ROLE])
    def test_it_refuses_a_32_tile_container_that_carries_weight_bytes(
        self, tmp_path: Path, wan22_provenance, role: str
    ) -> None:
        """The mix-up of plugging a container that holds weights into the preset of a borrowing set
        (the storage check looks at the lender, so it passes — only the sharing check rejects
        it)."""
        storage = "i8" if role.endswith("_i8") else "f16"
        graph = _VAE_SHARED_GRAPHS[role]
        sources = _build_sources22(
            tmp_path, containers={role: _graph_container(graph, storage=storage)}
        )
        with pytest.raises(
            DistError, match=rf"{graph}/model.krm: holds \d+ bytes outside the descriptors"
        ):
            wan_plan(sources, generation=WAN22)

    @pytest.mark.parametrize("role", [WAN_VAE_FIRST_32_I8_ROLE, WAN_VAE_NEXT_32_ROLE])
    def test_it_refuses_a_32_tile_container_naming_a_tensor_its_lender_lacks(
        self, tmp_path: Path, wan22_provenance, role: str
    ) -> None:
        """The form of a series where only the lender was rebaked (keys the borrower names are
        absent from the lender)."""
        sources = _build_sources22(
            tmp_path, containers={role: _shared_container(role, mark="vae_decoder_other")}
        )
        with pytest.raises(
            DistError,
            match=r"the shared initializer '[^']*vae_decoder_other[^']*'"
            " has no real entry in the lender",
        ):
            wan_plan(sources, generation=WAN22)

    def test_it_refuses_a_wan21_model(self, tmp_path: Path, wan22_provenance) -> None:
        with pytest.raises(DistError, match=r"unknown Wan2\.2 model 't2v-1\.3b'"):
            wan_plan(_build_sources22(tmp_path), DEFAULT_MODEL, WAN22)

    def test_it_refuses_rope_base_without_its_asset(self, tmp_path: Path, wan22_provenance) -> None:
        sources = _build_sources22(
            tmp_path,
            containers={
                WAN_TRANSFORMER_I8_ROLE: _graph_container(
                    WAN_TRANSFORMER_ROLE, storage="i8", rope=None
                )
            },
        )
        with pytest.raises(
            DistError, match=r"asset 'rope_base' is missing.*`python -m wan\.ti2v_export_dit write`"
        ):
            wan_plan(sources, generation=WAN22)

    def test_a_vae_pair_of_different_tiles_points_to_the_wan22_writer(
        self, tmp_path: Path, wan22_provenance
    ) -> None:
        """The rebake hint of the sharing check names the generation's writer (the default of 2.1's
        `export_vae` writes the 2.1 series, so it has no effect on the 2.2 set)."""
        latent = (WAN_VAE_LATENT_INPUT, [48, 1, 4, 4])
        sources = _build_sources22(
            tmp_path,
            containers={
                WAN_VAE_FIRST_ROLE: _vae_container(WAN_VAE_FIRST_ROLE, [latent]),
                WAN_VAE_NEXT_ROLE: _vae_container(
                    WAN_VAE_NEXT_ROLE, [(WAN_VAE_LATENT_INPUT, [48, 1, 8, 8])]
                ),
            },
        )
        with pytest.raises(
            DistError,
            match=r"the latent input shape differs.*`python -m wan\.export_vae --model ti2v-5b`",
        ):
            wan_plan(sources, generation=WAN22)


class TestTheWan22ModelCard:
    def test_it_is_the_only_profile_and_is_resolved_without_a_choice(self) -> None:
        profiles = TI2V_PIPELINE.card_profiles
        assert list(profiles) == ["wan-ti2v"]
        assert resolve_card_renderer(TI2V_PIPELINE, None) is profiles["wan-ti2v"]

    def test_the_accepted_inputs_match_the_typescript_side(self) -> None:
        """The card's accepted set equals the TS accepted set (`WAN22_TI2V_GENERATION`) through the
        same fixture."""
        fixture = json.loads(_TI2V_CARD_LIMITS_FIXTURE.read_text(encoding="utf-8"))
        assert fixture == {
            "acceptedSizes": [
                {"width": width, "height": height} for width, height in WAN22_ACCEPTED_SIZES
            ],
            "minFrames": WAN22_FRAMES[0],
            "maxFrames": WAN22_FRAMES[1],
        }, (
            "after changing WAN22_ACCEPTED_SIZES / WAN22_FRAMES in card.py, align the fixture and"
            " the TS side too"
        )

    def test_it_describes_wan22(self, assembled22) -> None:
        out_dir, _ = assembled22
        card = (out_dir / MODEL_CARD_FILENAME).read_text(encoding="utf-8")
        upstream = SOURCES[_TI2V]
        assert "# Wan2.2 TI2V 5B — Karume" in card
        assert f"base_model: {upstream.repo}" in card
        assert f"at commit `{upstream.revision}`" in card
        assert f'repo: "hdae/{WAN22_REPO_NAME}"' in card
        assert "await Ti2vPipeline.fromPretrained(" in card
        assert 'import { encodePng } from "jsr:@karume/models";' in card
        assert 'import { frameToRgba, Ti2vPipeline } from "jsr:@karume/models/wan";' in card
        assert "- **size**: 1280 × 704 or 704 × 1280." in card
        assert "- **frames**: 4n+1 from 5 to 121." in card
        assert "  // width: 1280, height: 704, // or 704 × 1280" in card
        quants = card.split("### Quants")[1].split("\n\n")[1]
        rows = [line for line in quants.splitlines() if line.startswith("| `")]
        assert [row.split("`")[1] for row in rows] == [
            _REFERENCE,
            _PRACTICAL,
            _ONLINE,
            _I8,
            _CONV8,
        ]
        assert [row.split(" |")[0] for row in rows if "(default)" in row] == [
            f"| `{_CONV8}` (default)"
        ]
        assert f"If no quant is given, it runs as `{_CONV8}`" in card
        assert f'// quant: "{_CONV8}", // default' in card
        assert f"implements `{WAN22_SUPPORTED_PIPELINE}`" in card
        prose = " ".join(card.split())
        # Claims both T2V and I2V (ADR 0121 step 9c — I2V reached the public API).
        assert "A **text-to-video** and **image-to-video** distribution of Wan2.2 TI2V 5B" in prose
        assert "Text to video by default; image to video when `generate()` is given an `image`" in (
            prose
        )
        assert "Text to video only." not in prose
        assert "not available yet" not in prose
        assert "at 24 fps" in prose
        assert "- Nine graphs:" in prose
        # The 32-tile pair (ADR 0134): borrowed weights, an explicit tile, 16 by default.
        assert (
            "a second decoder pair at a 32-latent tile that borrows the 16-tile pair's weights, so"
            " there is no extra weight download; the application chooses the tile explicitly, 16 by"
            " default, and the 10 GB class keeps 16" in prose
        )

    def test_the_overview_names_every_graph_of_the_manifest(self, assembled22) -> None:
        _overview_names_every_graph(*assembled22)

    def test_it_says_49_frames_did_not_run_through_the_pipeline_class(self, assembled22) -> None:
        out_dir, _ = assembled22
        prose = " ".join((out_dir / MODEL_CARD_FILENAME).read_text(encoding="utf-8").split())
        assert "and 33 frames at 1280 × 704 in 50-step runs." in prose
        assert (
            "49 frames at 1280 × 704 ran 50 steps through the same pipeline stages, driven by a"
            " development script rather than the pipeline class." in prose
        )
        assert "full 50-step runs of both quants at 1280 × 704 with 33 frames" in prose
        assert "33 and 49 frames" not in prose

    def test_the_usage_suggests_the_typescript_defaults(self) -> None:
        """The default comment of Usage is drawn from the generation's defaults (a copy of
        `defaults` of `descriptor.ts`), and the defaults are within the accepted set."""
        for card in (WAN21_CARD, WAN22_CARD):
            assert card.default_size in card.accepted_sizes
            low, high = card.frames
            assert low <= card.default_frames <= high
            assert (card.default_frames - 1) % 4 == 0
        assert (WAN22_CARD.default_size, WAN22_CARD.default_frames) == ((1280, 704), 33)

    def test_it_carries_none_of_the_wan21_sizes_or_frame_counts(self, assembled22) -> None:
        out_dir, _ = assembled22
        card = (out_dir / MODEL_CARD_FILENAME).read_text(encoding="utf-8")
        for figure in ("832", "480", "81"):
            assert re.search(rf"\b{figure}\b", card) is None, figure
        # 2.2 also has a full Chrome run (121 frames), but the sentence about the 2.1 browser run
        # is not copied.
        assert "46.1 minutes" not in card
        assert "`f16` quant with the precomputed embeddings" not in " ".join(card.split())
        assert re.search(r"(?<!\w)Pipeline\.fromPretrained\(", card) is None

    def test_it_attributes_the_text_assets_to_the_wan21_checkpoint(self, assembled22) -> None:
        out_dir, _ = assembled22
        prose = " ".join((out_dir / MODEL_CARD_FILENAME).read_text(encoding="utf-8").split())
        text = SOURCES[DEFAULT_MODEL]
        assert f"[{text.repo}](https://huggingface.co/{text.repo}) at commit `{text.revision}`" in (
            prose
        )
        relation = "this checkpoint's `text_encoder` holds the same encoder rounded to bfloat16"
        assert relation in prose
        assert "bit-identical in float32" not in prose
        # The copy of the 2.1 modification summary (the f16 transformer and its second, int8
        # version) does not remain.
        assert "float16 transformer" not in prose
        assert "second, int8 copy" not in prose
        assert "the transformer shipped only in int8" in prose

    def test_it_names_the_measured_resources_and_what_was_not_measured(self, assembled22) -> None:
        out_dir, _ = assembled22
        card = (out_dir / MODEL_CARD_FILENAME).read_text(encoding="utf-8")
        assert "| `f16+dit8` | 33 | 21.8 s | 7.86 GiB | 2,143 s | 439 s | 43 min 3 s |" in card
        assert (
            "| `f16+dit8` | 49 | not measured | 8.45 GiB | 3,360 s | 647 s | 66 min 53 s |" in card
        )
        assert (
            "| `f16+dit8-a8-attn8-s16` | 33 | 9.9 s | 7.84 GiB | 960 s | 439 s | 23 min 19 s |"
            in card
        )
        assert (
            "| `f16+dit8-a8-attn8-s16` | 49 | not measured | 8.49 GiB | 1,504 s | 655 s |"
            " 36 min 5 s |" in card
        )
        assert f"| `{_ONLINE}` | 121 | 21.7 s | 9.01 GiB | not run | not run | not run |" in card
        assert f"| `{_ONLINE}` | 33 |" not in card
        assert f"| `{_ONLINE}` | 49 |" not in card
        assert set(WAN22_RESOURCES) == {_REFERENCE, _PRACTICAL, _ONLINE}
        prose = " ".join(card.split())
        assert (
            "- **Quality of `f16+dit8-a8-attn8-s16`**: after the first step its latent differs from"
            " `f16+dit8`'s (the same int8 weights, computed in float32) by a relative RMS error of"
            " 0.044 at 33 frames and 0.049 at 121 frames. Side by side with `f16+dit8` on twelve"
            " 50-step clips at 1280 × 704 with 33 frames (seeds 42 to 45 with the three fixed"
            " prompts), no clear degradation was seen."
        ) in prose
        # One forward is from step 2 (2026-10-04), the full run from 2026-10-05. 49 frames is not a
        # product class.
        assert "timed on its own on 2026-10-04" in prose
        assert "on 2026-10-05, at 1280 × 704 with 50 steps" in prose
        assert "The 33-frame runs went through `Ti2vPipeline`" in prose
        assert "the 49-frame runs drove the same pipeline stages from a development script" in prose
        assert "4.36 GiB at 33 frames and 4.51 GiB at 49 frames with either quant" in prose
        assert "4.52" not in prose
        # The default preset's quality (ADR 0132 "Results"): the 33-frame A/B only; the 121-frame
        # A/B and the visual check are not claimed (the visual check was the s16 preset's).
        assert (
            f"- **Quality of `{_ONLINE}`**: after the first step its latent differs from"
            " `f16+dit8`'s (the same int8 weights, computed in float32) by a relative RMS error of"
            " 0.042 at 33 frames (text to video, on the NVIDIA GeForce RTX 3080 Ti under Node with"
            " Dawn on 2026-10-23); at 121 frames it has not been measured, because `f16+dit8`'s"
            " 121-frame step does not fit that 12 GiB GPU under Node with Dawn. The visual check of"
            f" twelve 50-step clips was made with `{_PRACTICAL}` only — the same check of"
            f" `{_ONLINE}` is still pending — and its image-to-video error has not been measured."
        ) in prose
        # The int8 VAE decoder's presets (ADR 0133 decision 6) have no row in the Resources table
        # (their measurements are the 33-frame clips of the visual check, in their bullet); the
        # default-quant cell of the Models table names conv8, so the check is on the table itself.
        assert f"The other quants (`{_I8}` / `{_CONV8}`) have not been measured yet." in prose
        resources = card.split("## Resources", 1)[1].split("\n## ", 1)[0]
        assert f"| `{_I8}` |" not in resources
        assert f"| `{_CONV8}` |" not in resources
        assert (
            f"- **`{_I8}` and `{_CONV8}`**: both store the VAE decoder's weights as int8 (one scale"
            " per output channel) as well as the transformer's; `conv8` computes the decoder's 3-D"
            " convolutions on per-pixel int8 activations (integer dot products). On the NVIDIA"
            " GeForce RTX 3080 Ti under Node with Dawn (2026-10-23, 1280 × 704 with 33 frames, 50"
            f" steps, 8 clips) `{_CONV8}` decoded the clip in 87 s against 168 s for `{_ONLINE}`"
            " (the whole clip 533 s against 611 s) with a GPU-wide peak of 6.2 GiB (nvidia-smi)."
        ) in prose
        assert "has been measured with this distribution only in the 121-frame run above" in prose
        assert "`maxStorageBufferBindingSize` (128 MiB)" in card

    def test_it_names_the_121_frame_row_with_its_own_gpu_and_reading(self, assembled22) -> None:
        """The 121-frame row is only for the practical preset — a full run on the RTX 3080 Ti
        development machine, with a different GPU and a different reading (nvidia-smi, whole GPU)
        from the B570 rows, and it says thermal limiting is included. The reference preset was not
        run at 121."""
        out_dir, _ = assembled22
        card = (out_dir / MODEL_CARD_FILENAME).read_text(encoding="utf-8")
        assert (
            "| `f16+dit8-a8-attn8-s16` | 121 | not measured | 11.09 GiB | 3,068 s | 710 s |"
            " 63 min 8 s |" in card
        )
        assert "| `f16+dit8` | 121 |" not in card
        prose = " ".join(card.split())
        assert (
            "The 33- and 49-frame rows were measured on an Intel Arc B570 in Deno 2.9.6 on"
            " 2026-10-05" in prose
        )
        # The 121 row shares only the size, steps, shift and guidance (the prompt and the text path
        # differ).
        assert (
            f"the `{_PRACTICAL}` 121-frame row at the same size, steps, shift and guidance on an"
            " NVIDIA GeForce RTX 3080 Ti (12 GiB) in Deno 2.9.6 on 2026-10-06; the"
            f" `{_ONLINE}` row is described below the table:" in prose
        )
        assert "same settings" not in prose
        assert "another prompt (`cat-dog-baking`, where the B570 rows used `boxing-cats`)" in prose
        assert "it is not comparable with the B570 rows" in prose
        assert (
            f"The `{_PRACTICAL}` 121-frame row comes from a different GPU and a different reading"
            in prose
        )
        # The online row is one DiT pass under Node + Dawn (ADR 0132 "Results") with its own
        # reading, and its 50-step cells are "not run".
        assert (
            "Pass is one transformer pass (batch 1, 7,920 tokens) timed on its own on 2026-10-04,"
            f" except in the `{_ONLINE}` row." in prose
        )
        assert (
            f"The `{_ONLINE}` row is one transformer pass at 121 frames (batch 1, 27,280 tokens)"
            " timed on its own on the same NVIDIA GeForce RTX 3080 Ti under Node with Dawn on"
            " 2026-10-23 (the second run of the same input); its Transformer peak is the runtime's"
            " own count of what the transformer allocated in that run (neither nvidia-smi nor the"
            " driver's fdinfo). That quant has not been run for 50 steps, so its stage, decode and"
            " clip times are not given, and it has not been measured at 33 or 49 frames." in prose
        )
        assert "the memory in use on the whole GPU as read by nvidia-smi, not the driver's" in prose
        assert "its times include thermal throttling" in prose
        # There may also be sha rows of the reference preset for 2 steps × 121, so the claim is
        # limited to 50 steps.
        assert (
            "The 121-frame clip has not been run for 50 steps with the `f16+dit8` quant." in prose
        )
        # The definition of how it was measured is limited to the B570 rows (the 121 row is
        # nvidia-smi).
        assert (
            "For the B570 rows, Transformer peak is the total allocation during the transformer"
            " stage (the driver's fdinfo)." in prose
        )
        assert (
            "- **GPU memory**: the B570 rows' peaks are of the total allocation (the driver's"
            f" fdinfo), the `{_PRACTICAL}` 121-frame row's is the memory in use on the whole GPU"
            f" (nvidia-smi), and the `{_ONLINE}` row's is the runtime's own count." in prose
        )
        assert "- **GPU memory**: the peaks are of the total allocation" not in prose
        assert "about 1.46 GiB at 121 frames (S = 27,280)" in prose
        # The binding closest to the 2 GiB binding limit (a row block of the scores), and the actual
        # limit Chrome granted.
        assert "the largest binding at 121 frames is about 1.96 GiB" in prose
        assert "bindings of up to 2,147,483,644 bytes" in prose
        assert "2 GiB buffers" not in prose

    def test_it_says_121_frames_fit_12_gib_and_the_b570_is_checked_only_up_to_57_frames(
        self, assembled22
    ) -> None:
        """The accepted upper bound of 121 is claimed by the fact that it completed in Chrome on the
        development machine (RTX 3080 Ti, 12 GiB) and the RTX 5070 Ti, with the caveat that the peak
        is nvidia-smi's whole GPU and Deno's headroom can be thinner. On the B570 only up to 57
        frames was confirmed, and longer clips were not run, so they are unsupported. The "not
        expected to fit" estimate (a guess) covers only the 2 points with estimates (81, 121); for
        the other lengths it says unknown. Saying only 121 is unsupported would read as 61–117
        running, so it is written as a range. It says that if it does not fit, it fails in the
        middle of execution rather than at admission."""
        out_dir, _ = assembled22
        prose = " ".join((out_dir / MODEL_CARD_FILENAME).read_text(encoding="utf-8").split())
        assert (
            "121 frames at 1280 × 704 (the `f16+dit8-a8-attn8-s16` quant) ran 50 steps the same way"
            " on an NVIDIA GeForce RTX 3080 Ti (12 GiB) in Deno, and in Chrome on an NVIDIA GeForce"
            " RTX 5070 Ti. 704 × 1280 has been run only in the 2-step runs at 17 frames." in prose
        )
        assert (
            "On the RTX 3080 Ti, the memory in use on the whole GPU (read with nvidia-smi) peaked"
            " at 11.09 GiB during the transformer stage, about 0.9 GiB below the card's 12 GiB"
            in prose
        )
        assert "to spare" not in prose
        assert (
            "The headroom Deno actually has can be thinner, since Deno stops allocating at a"
            " ceiling below the card's size that varies over time, and on a 12 GiB GPU shared with"
            " other programs 121 frames may not fit (an estimate — not run)." in prose
        )
        assert (
            "On a GPU with about 10 GB, such as the Intel Arc B570 (where Deno can allocate about"
            " 9.4 GiB in total), clips up to 57 frames have been checked: the longest 50-step runs"
            " there were 49 frames with both quants and 57 frames with the"
            " `f16+dit8-a8-attn8-s16` quant, driven by the development script (its rows under"
            " Resources stop at 49 frames). Longer clips are not supported there, because none of"
            " them has been run on the B570." in prose
        )
        # Saying only 121 is unsupported would read as 61–117 running on 10 GB-class GPUs — write a
        # range.
        assert "121 frames are not supported" not in prose
        # Estimates exist only for 81 and 121 — 61–77 / 85–117 are not said to be "not expected to
        # fit".
        assert "not expected to fit" not in prose
        assert (
            "A memory estimate made before the runs covers only two of those lengths and puts both"
            " beyond that ceiling: the next length after 77 frames with the"
            " `f16+dit8-a8-attn8-s16` quant, and 121 frames with both quants (the 10.64 GiB the"
            " transformer stage allocated at 121 frames on the RTX 3080 Ti is above it too). The"
            " other lengths were not estimated, so whether they fit is not known." in prose
        )
        # The accepted set is not per machine — it must not read as if a 10 GB-class GPU were
        # rejected up front with ModelInputError.
        assert (
            "The accepted set does not depend on the GPU, so such a request is still accepted, and"
            " if it does not fit it fails during the run — with an out-of-memory error"
            " (`GpuOutOfMemoryError`), or a lost device near the limit — rather than with"
            " `ModelInputError`, and the time spent on the stages before it is lost." in prose
        )
        # The browser full run is claimed as values from another GPU and runtime, and the
        # combinations that did not run are said not to have run.
        assert (
            "Chrome on an NVIDIA GeForce RTX 5070 Ti finished one 50-step run (the"
            " `f16+dit8-a8-attn8-s16` quant with the text encoder on the GPU, 1280 × 704, 121"
            " frames) in 40.3 minutes." in prose
        )
        assert "the precomputed embeddings have not been run in a browser yet" in prose
        assert "2,417.5 s (40.3 minutes) — a different GPU and runtime from the rows above" in prose
        assert "Not run in a browser yet." not in prose
        assert "which has not been checked yet" not in prose

    def test_it_says_a_seat_without_figures_has_not_been_measured(self, assembled22) -> None:
        _, manifest = assembled22
        changed = json.loads(json.dumps(manifest))
        quants = changed["models"][_TI2V]["quants"]
        quants["f16+other"] = quants[_REFERENCE]
        card = _card22(changed)
        assert (
            f"The other quants (`{_I8}` / `{_CONV8}` / `f16+other`) have not been measured yet."
            in card
        )
        assert "| `f16+other` |" not in card.split("### Quants")[0]

    def test_it_refuses_a_manifest_without_a_measured_seat(self, assembled22) -> None:
        _, manifest = assembled22
        changed = json.loads(json.dumps(manifest))
        model = changed["models"][_TI2V]
        model["quants"] = {"i8": model["quants"][_REFERENCE]}
        model["defaultQuant"] = "i8"
        with pytest.raises(ValueError, match="no quant whose execution resources were measured"):
            _card22(changed)

    def test_the_defaults_come_from_the_manifest(self, assembled22) -> None:
        _, manifest = assembled22
        changed = json.loads(json.dumps(manifest))
        changed["models"][_TI2V]["pipelineConfig"] = {
            "scheduler": {"shift": 7.5},
            "defaults": {"steps": 23, "guidance": 4.25},
        }
        card = _card22(changed)
        assert "- **steps**: 23" in card
        assert "- **shift** (flow-matching shift): 7.5" in card

    def test_it_refuses_a_wan21_model_or_pipeline(self, assembled22) -> None:
        _, manifest = assembled22
        foreign_model = json.loads(json.dumps(manifest))
        foreign_model["models"] = {DEFAULT_MODEL: foreign_model["models"][_TI2V]}
        with pytest.raises(ValueError, match=r"'t2v-1\.3b' is not a Wan2\.2 model"):
            _card22(foreign_model)
        foreign_pipeline = json.loads(json.dumps(manifest))
        foreign_pipeline["models"][_TI2V]["pipeline"] = WAN_PIPELINE
        with pytest.raises(ValueError, match=WAN22_SUPPORTED_PIPELINE):
            _card22(foreign_pipeline)

    def test_the_wan21_card_refuses_a_wan22_manifest(self, assembled22) -> None:
        _, manifest = assembled22
        with pytest.raises(ValueError, match=WAN_SUPPORTED_PIPELINE):
            _card(manifest)


def _frontmatter(card: str) -> str:
    """The YAML at the top of the card (the part between `---` lines)."""
    return card.split("---\n", 2)[1]


class TestTheWan22ImageToVideoCard:
    """The I2V description in the Wan2.2 card (ADR 0121 step 9c): the inputs (`image` / `fit`), the
    usage, the verification figures (copies of the addenda "results of step 9a" and "results of step
    9b" and of the test constants), and the scope of the quality figures (measured on T2V). The
    Wan2.1 card has not a single I2V line (`wan.Pipeline` of 2.1 rejects `image`)."""

    def test_it_tags_image_to_video_and_keeps_the_single_pipeline_tag(self, assembled22) -> None:
        out_dir, _ = assembled22
        head = _frontmatter((out_dir / MODEL_CARD_FILENAME).read_text(encoding="utf-8"))
        # HF has one pipeline tag — it stays text-to-video, and I2V is claimed with tags.
        assert "pipeline_tag: text-to-video\n" in head
        assert "pipeline_tag: image-to-video" not in head
        assert "tags:\n  - text-to-video\n  - image-to-video\n  - video-generation\n" in head

    def test_the_wan21_card_carries_no_image_to_video_lines(self, assembled) -> None:
        out_dir, _ = assembled
        card = (out_dir / MODEL_CARD_FILENAME).read_text(encoding="utf-8")
        assert "image-to-video" not in card
        assert "image to video" not in card
        assert "- **image**" not in card
        assert "- **fit**" not in card

    def test_it_describes_the_image_and_fit_inputs(self, assembled22) -> None:
        out_dir, _ = assembled22
        card = (out_dir / MODEL_CARD_FILENAME).read_text(encoding="utf-8")
        inputs = card.split("## Accepted inputs", 1)[1].split("\n## ", 1)[0]
        prose = " ".join(inputs.split())
        assert (
            "- **image**: the first frame for image to video, as decoded RGB8 pixels"
            " `{ data, width, height }` (`data` a `Uint8Array` with 3 bytes per pixel" in prose
        )
        assert "used as given (no color-space conversion)" in prose
        # The size selection (`selectWanI2vSize` — with 2 sizes, "width ≥ height → 1280×704").
        assert (
            "Without `width` / `height`, the accepted size closest to the image's aspect ratio is"
            " used — 1280 × 704 when the image is at least as wide as it is tall, 704 × 1280"
            " otherwise." in prose
        )
        assert '- **fit** (only with `image`): `"crop"` (the default' in prose
        assert (
            "scales the image to cover the output size, keeping its aspect ratio, and cuts out"
            in prose
        )
        assert (
            '`"stretch"` (as in diffusers\' `WanImageToVideoPipeline`) resizes the width and'
            in prose
        )
        assert "a port of Pillow's LANCZOS filter that gives the same bytes as Pillow 12.3.0" in (
            prose
        )
        # A misspelling is a plain Error; fit without image is a ModelInputError (the check of
        # plan.ts).
        assert "`fit` without `image` throws `ModelInputError`" in prose
        assert "a value other than these two throws a plain `Error`" in prose
        # The plain Error for a misspelling is named as the exception to the closing sentence right
        # after it (outside the accepted set is ModelInputError).
        assert "(the one exception to the sentence below)" in prose
        # The I2V item comes after the seed item and before the rejection sentence (within the list
        # of accepted inputs).
        assert (
            inputs.index("- **seed**")
            < inputs.index("- **image**")
            < inputs.index("Anything outside these sets")
        )

    def test_the_usage_shows_an_image_to_video_call(self, assembled22) -> None:
        out_dir, _ = assembled22
        card = (out_dir / MODEL_CARD_FILENAME).read_text(encoding="utf-8")
        usage = card.split("## Usage", 1)[1].split("\n## ", 1)[0]
        assert "  image: { data: rgb, width: imageWidth, height: imageHeight }," in usage
        assert '  // fit: "crop", // default' in usage
        prose = " ".join(usage.split())
        # Decoding the image is the caller's job (fast-png / jpeg-js on Deno, createImageBitmap and
        # a canvas in the browser).
        assert "Decoding the image file is up to the caller" in prose
        assert "an image decoder such as fast-png and jpeg-js works" in prose
        assert "examples/wan" not in prose
        assert "`createImageBitmap` and a canvas" in prose
        # jpeg-js defaults to RGBA; the example takes RGB directly with `formatAsRGBA: false`
        # (`examples/wan/decode-image.ts`).
        assert (
            "a canvas gives RGBA, and so does jpeg-js by default (`formatAsRGBA: false` gives RGB)"
            " — drop the alpha channel." in prose
        )
        # Differences from the reference in the decoding step (EXIF orientation, alpha, the JPEG
        # decoder) — the same facts as the README of the example / gpu-lab.
        assert "does not apply the EXIF orientation" in prose
        assert "`createImageBitmap` applies the orientation by default" in prose
        assert "drops the alpha channel without compositing it over a background" in prose
        assert (
            "For PNG, fast-png gives the same pixels as Pillow; JPEG decoders differ slightly"
            in prose
        )
        # The precomputed path has no text step — the transformer comes after the encoder.
        assert (
            "With an `image`, the VAE encoder stage runs first and is disposed before the next"
            " stage (the text encoder, or the transformer with the precomputed embeddings) is"
            " opened." in prose
        )
        # I2V on Deno ran only on the precomputed path (the combination with the GPU umT5 completed
        # only in Chrome).
        assert (
            "In Deno, image to video has so far been run only with the precomputed embeddings;"
            " with the text encoder on the GPU (the default in this example) it has been run only"
            " in Chrome (see What is this)." in prose
        )
        # The second code block is closed too (an even number of ```).
        assert usage.count("```") == 4

    def test_it_explains_the_conditioning_in_the_overview_and_the_attribution(
        self, assembled22
    ) -> None:
        out_dir, _ = assembled22
        prose = " ".join((out_dir / MODEL_CARD_FILENAME).read_text(encoding="utf-8").split())
        assert (
            "at every step the first latent frame of the transformer's input is replaced with it"
            " (the diffusers form: the tokens of that frame are marked by the condition mask and"
            " take timestep 0)." in prose
        )
        assert "for image to video the resize, normalization and patchify of the image and the" in (
            prose
        )
        assert (
            "in image to video the tokens of the first latent frame are masked in and take"
            " timestep 0, in text to video the mask is all false" in prose
        )
        assert "that text-to-video leaves inactive" not in prose
        # The modification summary includes the VAE encoder (a component in the distribution).
        assert "the part of the VAE encoder that encodes one conditioning image" in prose
        # The browser I2V is written as a combination that completed, and it says it was not
        # compared with the reference (no quality claim).
        assert (
            "Image to video finished there with the same quant and the text encoder on the GPU,"
            " from an 896 × 1152 image cropped to 704 × 1280, in 20-step runs with shift 7:"
            " 390.0 s for 33 frames and 1,346.5 s for 121 frames. These browser runs were not"
            " compared with a reference." in prose
        )
        assert "neither has image to video" not in prose

    def test_it_copies_the_image_to_video_verification_figures(self, assembled22) -> None:
        """The figures are copies of the addenda "results of step 9a / 9b" and the test constants
        (`WAN_I2V_ENCODER_BAND` 7.8e-5, `LATENT_RATIO_BANDS` 2.9e-3 / 3.5e-3, `FINAL_RATIO_BAND`
        3.9e-3, `FRAME_RATIO_BAND` 8.9e-3, the T2V latents.1 band 9.5e-4, fault injection 138–358
        times)."""
        out_dir, _ = assembled22
        card = (out_dir / MODEL_CARD_FILENAME).read_text(encoding="utf-8")
        section = card.split("## Determinism and verification", 1)[1].split("\n## ", 1)[0]
        prose = " ".join(section.split())
        assert "- **Image to video against the upstream reference** (the `f16+dit8` quant" in prose
        assert "the largest absolute difference of the normalized latent stays within 7.8e-5" in (
            prose
        )
        # The third size of the encoder comparison (256×160) is claimed to be outside the accepted
        # set.
        assert (
            "for three images, each at the two accepted sizes and at a small 256 × 160 size"
            " outside them" in prose
        )
        assert (
            "stays within 2.9e-3 for the latent after the first step, 3.5e-3 after the second,"
            " 3.9e-3 for the latent passed to the VAE and 8.9e-3 for the frames" in prose
        )
        # Why the band is wide (the DiT amplifies the difference at the encoder exit through the
        # condition tokens).
        assert "(3.5e-3 against 9.5e-4 after the second step)" in prose
        assert "the transformer amplifies that about 250 times on those tokens" in prose
        assert "the difference falls back to the text-to-video size" in prose
        assert "138 to 358 times its tolerance away" in prose
        assert "Five faults in the encoder land 15,000 times its tolerance away or more." in prose
        assert (
            "- **Pinned image-to-video runs**: the same request with the same `image` and `fit`"
            " repeats bit for bit" in prose
        )
        assert "for one image at 1280 × 704 with 9 frames (2 steps) with both quants" in prose
        assert "for the other two images and for 704 × 1280 with the `f16+dit8` quant" in prose
        # The 50-step I2V has sha rows for both presets (RTX 3080 Ti, opt-in; the practical preset
        # has a different floor from the reference preset).
        assert (
            "In the same opt-in set, 50-step image-to-video runs at 1280 × 704 with 33 frames are"
            " pinned with both quants on the NVIDIA GeForce RTX 3080 Ti, the"
            " `f16+dit8-a8-attn8-s16` result again required to differ from the `f16+dit8` one."
            in prose
        )
        assert "are not pinned" not in prose
        # The T2V comparison item says T2V.
        assert "- **Text to video against the upstream reference**" in prose

    def test_the_quality_figures_say_which_task_they_were_measured_on(self, assembled22) -> None:
        """The T2V figures and the I2V figures (the A/B 0.046 of step 9d and the 12 visual pairs)
        are kept apart within the same sentence."""
        out_dir, _ = assembled22
        prose = " ".join((out_dir / MODEL_CARD_FILENAME).read_text(encoding="utf-8").split())
        assert (
            "On image to video the same first-step check holds at 1280 × 704 with 33 frames (one"
            " image, `crop`, an opt-in case: relative RMS error 0.046, within twice the measured"
            " error), and 12 clips (3 images × seeds 42–45, 1280 × 704 with 33 frames, 50 steps)"
            " side by side with `f16+dit8` showed no clear degradation on 2026-10-10 either."
            in prose
        )
        assert (
            "no clear degradation was seen. Both were measured with text to video; on image to"
            " video the first-step error is 0.046 at 33 frames (one image, `crop`) and twelve"
            " 50-step clips (3 images × seeds 42–45) side by side showed no clear degradation"
            " either." in prose
        )
        assert "have not been compared on image to video" not in prose
        assert "has not been compared between the quants" not in prose
        # The s16 preset's visual check is kept as its own (it was the default until 2026-10-23);
        # the online preset's comparison is the 33-frame text-to-video A/B only.
        assert (
            "It was the default until 2026-10-23, chosen after a visual check of 12 clips" in prose
        )
        assert "It became the default" not in prose
        assert (
            f"`{_ONLINE}`, the default from 2026-10-23 until `{_CONV8}` took over the same day, was"
            " compared the same way on text to video at 1280 × 704 with 33 frames, under Node with"
            " Dawn on the same GPU: relative RMS error 0.042, within the same tolerance. Its"
            " 121-frame comparison, its visual check and its image-to-video comparison have not"
            " been done yet." in prose
        )
        # The conv8 preset (the default since 2026-10-23 — ADR 0133 decision 8): the decoder's A/B
        # numbers and the maintainer's visual check against the online preset.
        assert (
            f"`{_CONV8}`, the default since 2026-10-23, keeps that transformer and changes only the"
            " VAE decoder (int8 weights, int8 activations in its 3-D convolutions): at the first"
            " decoded frame its relative RMS error is 0.013 against the same int8 weights computed"
            f" in f32 (`{_I8}`; the gate is 0.026) and 0.032 against the f16 decoder (`{_I8}`"
            " itself 0.030), and 8 clips (2 prompts × seeds 42–45, 1280 × 704 with 33 frames, 50"
            " steps)"
            f" side by side with `{_ONLINE}` showed no visible difference on 2026-10-23." in prose
        )
        assert "Text to video has been checked end to end on the GPU: 17 frames" in prose

    def test_the_resources_name_what_was_measured_for_image_to_video(self, assembled22) -> None:
        """The table is all T2V. For I2V it writes only the peak of the encoder alone (addendum
        "results of step 9a") and the 50-step wall time (the result file of the step 9b full run),
        and says the VRAM peak of one clip has not been measured."""
        out_dir, _ = assembled22
        card = (out_dir / MODEL_CARD_FILENAME).read_text(encoding="utf-8")
        section = card.split("## Resources", 1)[1].split("\n## ", 1)[0]
        prose = " ".join(section.split())
        assert "All of the above is text to video. Image to video adds the VAE encoder stage" in (
            prose
        )
        assert "the encoder kept the whole process at a peak of 1,421 MiB (nvidia-smi)" in prose
        assert (
            "took 1,395.3 s with `f16+dit8` and 723.5 s with `f16+dit8-a8-attn8-s16` in all,"
            " the encoder stage under 1 s of it" in prose
        )
        assert "an image-to-video clip's peak has not been measured" in prose
        # I2V rows are not in the table (the table rows are T2V measurements only).
        table = [line for line in section.splitlines() if line.startswith("| `")]
        assert table and all("image" not in line for line in table)


_REAL = wan_sources(SERIES_ROOT)
# MUST: `SERIES_CHECKS and …` — the default lane does not touch `outputs/series/` (ADR 0139
# decision 6).
_REAL_PRESENT = SERIES_CHECKS and (
    (_REAL.series / WAN_TRANSFORMER_ROLE).is_dir()
    and (_REAL.i8_series / WAN_TRANSFORMER_ROLE).is_dir()
    and _REAL.text_embeds.is_file()
    and _REAL.text_encoder.parent.is_dir()
    and _REAL.tokenizer.is_file()
)


@pytest.mark.skipif(
    SERIES_CHECKS and not _REAL_PRESENT,
    reason=f"no real series: {_REAL.series} / {_REAL.i8_series} / {_REAL.text_encoder.parent}",
)
@requires_series
class TestTheRealSeries:
    """The plan can be built from the real series (the outputs of `wan.export_dit` (f16 / i8) /
    `wan.export_vae` / `wan.text_embeds` / `wan.umt5_export` / `wan.umt5_tokenizer`).

    It reads only the 2 documents of the containers, the binding tables, the assets (declarations
    and the base RoPE table), the header of the embedding asset and the tokenizer asset (about
    8 MB) (no weight payloads are read).
    """

    def test_the_plan_passes_every_gate(self) -> None:
        plan = wan_plan(_REAL)
        assert plan.pipeline == WAN_PIPELINE
        assert set(plan.artifacts) == {
            *WAN21.container_roles,
            WAN_TEXT_EMBEDS_ROLE,
            WAN_TOKENIZER_ROLE,
        }
        assert not set(WAN_VAE_ENCODER_ROLES) & set(plan.artifacts)


_REAL22 = wan_sources(SERIES_ROOT, WAN22)
_REAL22_PRESENT = SERIES_CHECKS and (
    (_REAL22.i8_series / WAN_TRANSFORMER_ROLE).is_dir()
    and (_REAL22.series / WAN_VAE_FIRST_ROLE).is_dir()
    and (_REAL22.series / WAN_VAE_NEXT_ROLE).is_dir()
    # The VAE decoder i8 (ADR 0133 decision 7 — `wan.export_vae --model ti2v-5b --dtype i8`).
    and (_REAL22.i8_series / WAN_VAE_FIRST_ROLE).is_dir()
    and (_REAL22.i8_series / WAN_VAE_NEXT_ROLE).is_dir()
    # The 32-tile pairs (ADR 0134 — `wan.export_vae --model ti2v-5b [--dtype i8] --tile 32
    # --shared`).
    and all(
        (root / graph).is_dir()
        for root in (_REAL22.series, _REAL22.i8_series)
        for graph in (WAN_VAE_FIRST_32_ROLE, WAN_VAE_NEXT_32_ROLE)
    )
    and _REAL22.text_embeds.is_file()
    and _REAL22.text_encoder.parent.is_dir()
    and _REAL22.tokenizer.is_file()
)


@pytest.mark.skipif(
    SERIES_CHECKS and not _REAL22_PRESENT,
    reason=f"no real Wan2.2 series: {_REAL22.series} / {_REAL22.i8_series} /"
    f" {_REAL22.text_encoder.parent}",
)
@requires_series
class TestTheRealWan22Series:
    """The plan can be built from the real Wan2.2 series (`wan.ti2v_export_dit` /
    `wan.export_vae --model ti2v-5b`), the Wan2.1 text assets and the umT5 mirror (it reads only the
    declarations, the binding tables, the base RoPE table and the asset headers)."""

    def test_the_plan_passes_every_gate(self) -> None:
        plan = wan_plan(_REAL22, generation=WAN22)
        assert plan.name == _TI2V
        assert plan.pipeline == WAN22_PIPELINE
        assert set(plan.artifacts) == {
            *WAN22.container_roles,
            WAN_TEXT_EMBEDS_ROLE,
            WAN_TOKENIZER_ROLE,
        }
        assert set(WAN_VAE_ENCODER_ROLES) <= set(plan.artifacts)
        assert plan.default_quant == _CONV8

    def test_the_text_assets_are_the_wan21_series_files(self) -> None:
        assert (_REAL22.text_embeds, _REAL22.tokenizer) == (_REAL.text_embeds, _REAL.tokenizer)
