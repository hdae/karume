"""Wan の配布 recipe（`wan.distribution`）とカード（`wan.card`）— 組み立て 1 周ぶんの単体テスト。

Wan2.1（`--pipeline wan`）と Wan2.2（`--pipeline wan-ti2v` — 同じ計画関数を世代の表 `WAN22` で
組む）の両方を見る。Wan2.2 の組では、DiT / VAE の容器は Wan2.2 の pin を、テキスト資産 2 本は
Wan2.1 の pin を名乗る（ADR 0121 決定 9 — 出所の門の分割）。

組み立てへ届く入力は数 KB の**正当な最小コンテナ**（`ir_fixtures`・umT5 は `umt5_fixture`）と、
書き手（`wan.text_embeds` / `wan.umt5_tokenizer`）と同じ形の合成の資産で作る。門に落とされることを
見るケースも同じ器で作り、**宣言だけを実物とずらす**。`text_encoder` は公開と同じ形（umT5 の配布形を
先に組み、そこへの越境参照）で組む。

実物の系列（`outputs/series/`）がある機では、実物で計画を 1 周組む門も回す（無ければ SKIP）。
"""

from __future__ import annotations

import json
import re
import tomllib
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from container_series import part_paths, placed_paths, read_component, write_component
from ir_fixtures import ir_container
from safetensors.numpy import save
from upstream_fixture import FIXTURE_REVISION, OTHER_REVISION, stamp_fixture_provenance

import dist
from _shared.licenses import APACHE_LICENSE_2_0_PATH
from _shared.paths import REPO_ROOT, SERIES_ROOT
from karume.container import CODEC_LEDGER, AssetInput, Provenance
from karume.dist import (
    MANIFEST_FILENAME,
    MODEL_CARD_FILENAME,
    NOTICE_FILENAME,
    DistError,
    ExternalComponents,
    assemble_family,
    resolve_card_renderer,
    verify_dist,
)
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
    WAN_VAE_FIRST_ROLE,
    WAN_VAE_LATENT_INPUT,
    WAN_VAE_NEXT_ROLE,
    WAN_WEIGHTS,
    WanGeneration,
    WanSources,
    wan_placements,
    wan_plan,
    wan_sources,
)
from wan.prompts import FIXED_PROMPTS
from wan.sources import DEFAULT_MODEL, SOURCES, UMT5_SOURCES
from wan.tests.umt5_fixture import umt5_container
from wan.umt5_distribution import (
    UMT5_DEFAULT_MODEL,
    UMT5_INPUTS,
    UMT5_OUTPUT_PATHS,
    UMT5_REPO_NAME,
    UMT5_ROLE,
    umt5_plan,
)

#: 合成の DiT の文脈入力 `[1, rows, width]`（実物の 512 × 4096 と**違う**数 — 幅を焼き込んでいれば
#: 落ちる）。
_ROWS = 6
_WIDTH = 8

#: 合成の RoPE 素表（組み立てが見るのは宣言と、transformer の容器どうしのバイト同一だけ）。
_ROPE_BASE = b"rope-base-table!"

#: 書き手（`wan.export_dit` / `wan.export_vae`）が焼く出所の正常形。
_PINNED = Provenance(
    license=SOURCES[DEFAULT_MODEL].license,
    notice=NOTICE_FILENAME,
    upstream_revision=SOURCES[DEFAULT_MODEL].revision,
)

#: umT5 の書き手（`wan.umt5_export`）が焼く出所の正常形 — umT5 の上流は本家 `google/umt5-xxl` の
#: pin で、DiT / VAE の Wan の pin とは別の行（ADR 0122 決定 1）。
_UMT5_PINNED = Provenance(
    license=UMT5_SOURCES[UMT5_DEFAULT_MODEL].source.license,
    notice=NOTICE_FILENAME,
    upstream_revision=UMT5_SOURCES[UMT5_DEFAULT_MODEL].source.revision,
)

#: `pipelineConfig` の欄（TS 側 `packages/models/src/wan/config.ts` の `ROOT_KEYS` /
#: `SCHEDULER_KEYS` / `DEFAULTS_KEYS` の写し）。ロード側は未知キーも欠落も parse 時に落とすので、
#: 焼く側とロード側の欄名は完全一致が要る。
_CONFIG_KEYS = {"scheduler": ("shift",), "defaults": ("steps", "guidance")}

#: TS 側の受理集合の写し（`packages/models/tests/wan_pipeline_test.ts` が `descriptor.ts` の値と
#: 同じことを見る）。カードの表とこの fixture を比べるので、カードと TS のどちらか片方だけの
#: 更新は赤になる。
_CARD_LIMITS_FIXTURE = REPO_ROOT / "packages/models/tests/fixtures/wan-card-limits.json"

#: 越境参照の先（umT5 の配布リポ）と、その合成の commit SHA。
_UMT5_REPO = f"hdae/{UMT5_REPO_NAME}"
_UMT5_REVISION = FIXTURE_REVISION


@pytest.fixture(autouse=True)
def _pinned_provenance(monkeypatch: pytest.MonkeyPatch) -> None:
    """フィクスチャ容器に「台本が pin した revision から焼いた」出所を名乗らせる。"""
    stamp_fixture_provenance(monkeypatch, _PINNED)


def _graph_container(
    role: str,
    *,
    storage: str = "f16",
    context: Sequence[int] = (1, _ROWS, _WIDTH),
    rope: tuple[str, str] | None = (WAN_ROPE_BASE_ASSET, WAN_ROPE_BASE_ROLE),
    rope_payload: bytes = _ROPE_BASE,
) -> list[bytes]:
    """系列に置く部品 1 本（part 列）。transformer だけが文脈入力と RoPE 素表の資産を持つ。"""
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
    """VAE の chunk グラフ 1 本（潜在 + cache の入力の宣言だけを実物の形にしたもの）。"""
    return ir_container(mark=role, named=role, storage="f16", inputs=inputs)


def _prompt_rows() -> list[dict[str, Any]]:
    """書き手（`wan.text_embeds.asset_metadata`）と同じ形のメタの prompts（トークン数は合成）。"""
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
    """合成の埋め込み資産（`F32 [tokens, width]` + メタのキー 1 つ — 書き手と同じ形）。

    `versions` は書き手が記録する版（固定値）へ上書きする差分。
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
    """合成のトークナイザ資産（書き手 `wan.umt5_tokenizer.build_asset` と同じ最上位の欄 — 語彙と表は
    門が読まないので持たない）。`overrides` は欄の上書き。"""
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


def _default_container(role: str) -> list[bytes]:
    """配置の役割ごとの正常形（transformer は格納ラベルごと — f16 系列 / i8 系列・text_encoder は
    umT5 の i8 系列）。出所は、transformer と VAE が `ir_fixtures` の焼く値（テストの間だけ Wan の
    pin `_PINNED` に差し替わる）・text_encoder が umT5 の書き手の焼く本家の pin
    （`_UMT5_PINNED`）。"""
    if role == WAN_TEXT_ENCODER_ROLE:
        return umt5_container(provenance=_UMT5_PINNED, width=_WIDTH)
    if role == WAN_TRANSFORMER_F16_ROLE:
        return _graph_container(WAN_TRANSFORMER_ROLE)
    if role == WAN_TRANSFORMER_I8_ROLE:
        return _graph_container(WAN_TRANSFORMER_ROLE, storage="i8")
    return _graph_container(role)


def _build_sources(
    root: Path,
    *,
    containers: Mapping[str, list[bytes]] = {},
    embeds: bytes | None = None,
    tokenizer: bytes | None = None,
    generation: WanGeneration = WAN21,
) -> WanSources:
    """系列 5 本を偽資産で再現する（配布しない golden の混入込み）。`containers` の鍵は配置の
    役割（`text_encoder` / `transformer_f16` / `transformer_i8` / VAE の 2 本 — Wan2.2 は
    `transformer_f16` を持たない）。容器の出所はその時点で `ir_fixtures` が焼く値（Wan2.2 の組は
    {@link wan22_provenance} で Wan2.2 の pin にしてから呼ぶ）。"""
    sources = wan_sources(root / "outputs" / "series", generation)
    placements = wan_placements(sources, generation)
    for role in generation.container_roles:
        write_component(placements[role], containers.get(role) or _default_container(role))
    # 配布に入ってはいけない golden（系列には実際にこれらが並んでいる）。
    for role in generation.transformer_roles:
        (placements[role].parent / "io.band-s00192-t0999.safetensors").write_bytes(b"io")
    # pipeline_steps は主の DiT の系列の根に並ぶ（実物: 2.1 は f16 系列・2.2 は i8 系列）。
    steps_series = placements[generation.transformer_roles[0]].parent.parent
    (steps_series / "pipeline_steps.band-boxing-cats.safetensors").write_bytes(b"steps")
    # encoder の GPU の門の golden は f16 系列の根に並ぶ（実物: 2.2 の系列だけ）。
    if generation.vae_encoder:
        (sources.series / "vae_encoder.boxing-cats-1280x704.safetensors").write_bytes(b"golden")
    (sources.text_encoder.parent / "reference.band-l0008.safetensors").write_bytes(b"golden")
    sources.text_embeds.parent.mkdir(parents=True, exist_ok=True)
    sources.text_embeds.write_bytes(_text_embeds() if embeds is None else embeds)
    sources.tokenizer.parent.mkdir(parents=True, exist_ok=True)
    sources.tokenizer.write_bytes(_tokenizer_asset() if tokenizer is None else tokenizer)
    return sources


def _assemble_umt5(root: Path, sources: WanSources) -> Path:
    """越境参照の参照元（umT5 の配布形）を同じ系列から組む（公開の順序 — umT5 が先）。"""
    out_dir = root / "models" / UMT5_REPO_NAME
    assemble_family([umt5_plan(sources.text_encoder)], out_dir, UMT5_DEFAULT_MODEL)
    return out_dir


def _reference(umt5_dir: Path, revision: str = _UMT5_REVISION) -> ExternalComponents:
    """`--ref-*` 5 指定と同じ越境参照（text_encoder を umT5 のリポへ向ける）。"""
    return ExternalComponents(
        repo=_UMT5_REPO,
        revision=revision,
        dist=umt5_dir,
        model=UMT5_DEFAULT_MODEL,
        roles=(WAN_TEXT_ENCODER_ROLE,),
    )


def _render(manifest: Mapping[str, Any], host_assets: Mapping[str, int]) -> str:
    return PIPELINE.card_profiles["wan"](
        manifest, repo=f"hdae/{WAN_REPO_NAME}", host_assets=host_assets
    )


@pytest.fixture
def assembled(tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    """公開と同じ形の配布形（text_encoder は umT5 のリポへの越境参照）。"""
    sources = _build_sources(tmp_path)
    out_dir = tmp_path / "models" / WAN_REPO_NAME
    manifest = assemble_family(
        [wan_plan(sources)],
        out_dir,
        DEFAULT_MODEL,
        render_card=_render,
        root_files=PIPELINE.root_files,
        external=_reference(_assemble_umt5(tmp_path, sources)),
    )
    return out_dir, manifest


def _card(manifest: Mapping[str, Any], repo: str = "hdae/x") -> str:
    """カードだけを組み直す（略称の対応表は配布 recipe の正本を渡す）。"""
    return render_wan_model_card(manifest, repo, WAN_QUANT_ABBREVIATIONS)


def _model(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    return manifest["models"][DEFAULT_MODEL]


def _present(out_dir: Path) -> list[str]:
    return sorted(str(path.relative_to(out_dir)) for path in out_dir.rglob("*") if path.is_file())


def _overview_names_every_graph(out_dir: Path, manifest: Mapping[str, Any]) -> None:
    """カードの「What is this」節が、manifest の weights の部品（= 配布形のグラフ）を 1 つ残らず
    名指しすること（部品を足した日に概要のグラフの列挙だけが古びる形を落とす）。"""
    card = (out_dir / MODEL_CARD_FILENAME).read_text(encoding="utf-8")
    overview = card.split("## What is this", 1)[1].split("\n## ", 1)[0]
    for model in manifest["models"].values():
        missing = [name for name in model["weights"] if f"`{name}`" not in overview]
        assert missing == [], missing


#: Wan2.2 のモデル名と、その書き手（`wan.ti2v_export_dit` / `wan.export_vae --model ti2v-5b`）が焼く
#: 出所の正常形（Wan2.2 の pin）。
_TI2V = "ti2v-5b"
_PINNED22 = Provenance(
    license=SOURCES[_TI2V].license,
    notice=NOTICE_FILENAME,
    upstream_revision=SOURCES[_TI2V].revision,
)

#: Wan2.2 の TS 側の受理集合の写し（`descriptor.ts` の `WAN22_TI2V_GENERATION` — TS 側のテストが同じ
#: fixture を突き合わせる）。
_TI2V_CARD_LIMITS_FIXTURE = REPO_ROOT / "packages/models/tests/fixtures/wan-ti2v-card-limits.json"

#: Wan2.2 の参照席・実用席。
_REFERENCE = "f16+dit8"
_PRACTICAL = "f16+dit8-a8-attn8-s16"


def _pin(model: str) -> dict[str, str]:
    """上流の pin をテキスト資産の `source` 欄の形で返す。"""
    return {"repo": SOURCES[model].repo, "revision": SOURCES[model].revision}


@pytest.fixture
def wan22_provenance(monkeypatch: pytest.MonkeyPatch) -> None:
    """フィクスチャ容器に Wan2.2 の pin を名乗らせる（autouse の Wan2.1 の pin を上書き）。"""
    stamp_fixture_provenance(monkeypatch, _PINNED22)


def _stamped(
    monkeypatch: pytest.MonkeyPatch, provenance: Provenance, make: Callable[[], list[bytes]]
) -> list[bytes]:
    """`provenance` を名乗る容器を 1 本だけ作る（作った後はテストの出所へ戻る）。"""
    with monkeypatch.context() as scoped:
        stamp_fixture_provenance(scoped, provenance)
        return make()


def _render22(manifest: Mapping[str, Any], host_assets: Mapping[str, int]) -> str:
    return TI2V_PIPELINE.card_profiles["wan-ti2v"](
        manifest, repo=f"hdae/{WAN22_REPO_NAME}", host_assets=host_assets
    )


def _card22(manifest: Mapping[str, Any], repo: str = "hdae/x") -> str:
    """Wan2.2 のカードだけを組み直す。"""
    return render_wan_model_card(manifest, repo, WAN_QUANT_ABBREVIATIONS, card=WAN22_CARD)


def _build_sources22(root: Path, **overrides: Any) -> WanSources:
    """Wan2.2 の系列（i8 の DiT・f16 の VAE・Wan2.1 のテキスト資産・umT5）を偽資産で再現する。"""
    return _build_sources(root, generation=WAN22, **overrides)


@pytest.fixture
def assembled22(tmp_path: Path, wan22_provenance: None) -> tuple[Path, dict[str, Any]]:
    """公開と同じ形の Wan2.2 の配布形（text_encoder は umT5 のリポへの越境参照）。"""
    sources = _build_sources22(tmp_path)
    out_dir = tmp_path / "models" / WAN22_REPO_NAME
    manifest = assemble_family(
        [wan_plan(sources, generation=WAN22)],
        out_dir,
        _TI2V,
        render_card=_render22,
        root_files=TI2V_PIPELINE.root_files,
        external=_reference(_assemble_umt5(tmp_path, sources)),
    )
    return out_dir, manifest


class TestLayout:
    def test_it_places_the_graphs_and_the_assets_under_the_model_subtree(self, assembled) -> None:
        """transformer は格納ラベルごとに 2 本（`model.f16.krm` / `model.i8.krm`）・VAE の
        decoder は 1 本ずつ・資産 2 本。text_encoder は越境参照なので 1 バイトも置かない。VAE の
        encoder は Wan2.2 だけの部品なので置かない（全世代の表に在っても 2.1 の配布形は
        変わらない）。"""
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
                # transformer の容器は資産 `rope_base` の専用 part が 1 本増える。
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
        """2.1 の f16 系列に encoder の容器が置かれていても、2.1 の計画は拾わない（encoder は
        Wan2.2 だけの部品 — 全世代の weights の表が 2.1 にも回るので、世代の欄で絞れている
        こと）。"""
        sources = _build_sources(tmp_path)
        for role in WAN_VAE_ENCODER_ROLES:
            write_component(sources.series / role / WAN_MODEL_FILE, _graph_container(role))
        plan = wan_plan(sources)
        assert not set(WAN_VAE_ENCODER_ROLES) & set(plan.artifacts)
        assert not set(WAN_VAE_ENCODER_ROLES) & set(plan.weights)
        for quant in plan.quants.values():
            assert not set(WAN_VAE_ENCODER_ROLES) & set(quant["weights"])

    def test_the_manifest_declares_the_three_seats_and_the_model_assets(self, assembled) -> None:
        """quant 席は `f16`・参照席 `f16+dit8`・実用席 `f16+dit8-a8-attn8-s16`（既定）の 3 つ
        （ADR 0120 決定 1・裁定 2026-10-04 の 4）で、どの席も text_encoder の i8 を選ぶ（weights は
        完全写像 — ADR 0119 追記 B）。資産はモデル単位の `text_embeds` と `umt5_tokenizer`
        （quant 非依存 — 同 C）。"""
        _, manifest = assembled
        model = _model(manifest)
        assert manifest["defaultModel"] == DEFAULT_MODEL
        assert model["pipeline"] == WAN_PIPELINE
        # VAE の encoder（Wan2.2 だけ）は 2.1 の weights に無い — 2.1 の manifest は不変。
        assert list(model["weights"]) == [
            WAN_TEXT_ENCODER_ROLE,
            WAN_TRANSFORMER_ROLE,
            WAN_VAE_FIRST_ROLE,
            WAN_VAE_NEXT_ROLE,
        ]
        assert list(model["weights"][WAN_TEXT_ENCODER_ROLE]) == ["i8"]
        assert list(model["weights"][WAN_TRANSFORMER_ROLE]) == ["f16", "i8"]
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
        # 参照席は実用席と同じ i8 の重みで session が空（自機 A/B 門の比較相手 — ADR 0110
        # 決定 5 ②）。
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
        """同じ系列から 2 度組むと、manifest も全ファイルも同じバイトになる（決定的）。"""
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
        # i8 の席の重みは配布形の改変なので告知に載る（文面は配布形と対応 MUST）。
        prose = " ".join(notice.split())
        assert "**int8 transformer**" in prose
        assert "keep the source float32 values" in prose

    def test_the_notice_says_where_the_text_encoder_comes_from(self, assembled) -> None:
        """text_encoder は越境参照（umT5 のリポ）・埋め込み資産は precomputed の経路のために残る・
        トークナイザは前処理の表と束ねた形（ADR 0119 追記 A / C）。「配っていない」とは書かない。"""
        out_dir, _ = assembled
        prose = " ".join((out_dir / NOTICE_FILENAME).read_text(encoding="utf-8").split())
        assert "**The text encoder is referenced, not stored here.**" in prose
        assert f"`{UMT5_REPO_NAME}` at a pinned commit" in prose
        assert "for use without the text encoder" in prose
        assert "ftfy's mojibake-detection pattern" in prose
        assert "text encoder is not distributed" not in prose


class TestTheTextEncoderReference:
    """text_encoder は umT5 の配布形への越境参照として宣言する（ADR 0119 追記 A —
    ADR 0109 決定 3）。"""

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
        """参照先の part は umT5 の系列の容器そのもの（Wan と umT5 のリポが同じバイト列を指す）。"""
        sources = _build_sources(tmp_path)
        umt5_dir = _assemble_umt5(tmp_path, sources)
        placed = umt5_dir / UMT5_DEFAULT_MODEL / UMT5_OUTPUT_PATHS[UMT5_ROLE]

        assert read_component(placed) == read_component(sources.text_encoder)

    def test_it_refuses_a_reference_whose_bytes_differ_from_its_own(self, tmp_path: Path) -> None:
        """参照元が別の umT5（ここでは出力の幅が違う容器）なら組まない（core の突合 — 中身の違う
        参照は「別のモデルの重み」を自分のものとして配る形になる）。"""
        other = _build_sources(
            tmp_path / "other",
            containers={
                WAN_TEXT_ENCODER_ROLE: umt5_container(provenance=_UMT5_PINNED, width=_WIDTH + 1)
            },
        )
        umt5_dir = _assemble_umt5(tmp_path / "other", other)
        sources = _build_sources(tmp_path)

        with pytest.raises(DistError, match="自分で組むバイト列と違う"):
            assemble_family(
                [wan_plan(sources)],
                tmp_path / "models" / WAN_REPO_NAME,
                DEFAULT_MODEL,
                external=_reference(umt5_dir),
            )

    def test_the_output_paths_match_the_umt5_repository(self) -> None:
        """越境参照は参照元が宣言する `<モデル名>/<この path>` を引くので、綴りは 1 つ。"""
        assert WAN_OUTPUT_PATHS[WAN_TEXT_ENCODER_ROLE] == UMT5_OUTPUT_PATHS[UMT5_ROLE]
        assert WAN_TEXT_ENCODER_ROLE == UMT5_ROLE


class TestTheStorageGates:
    @pytest.mark.parametrize("storage", ["f32", "i8"])
    def test_it_refuses_a_series_without_f16_storage(self, tmp_path: Path, storage: str) -> None:
        """f16 系列のつもりで素の f32 / 別格納の系列を指した取り違え。"""
        sources = _build_sources(
            tmp_path,
            containers={"vae_decoder_next": _graph_container("vae_decoder_next", storage=storage)},
        )
        with pytest.raises(DistError, match=r"vae_decoder_next: .* f16 が無い"):
            wan_plan(sources)

    def test_it_refuses_the_i8_series_in_the_f16_transformer_seat(self, tmp_path: Path) -> None:
        sources = _build_sources(
            tmp_path,
            containers={
                WAN_TRANSFORMER_F16_ROLE: _graph_container(WAN_TRANSFORMER_ROLE, storage="i8")
            },
        )
        with pytest.raises(DistError, match=r"transformer_f16: .* f16 が無い"):
            wan_plan(sources)

    def test_it_refuses_the_f16_series_in_the_i8_transformer_seat(self, tmp_path: Path) -> None:
        sources = _build_sources(
            tmp_path,
            containers={WAN_TRANSFORMER_I8_ROLE: _graph_container(WAN_TRANSFORMER_ROLE)},
        )
        with pytest.raises(DistError, match=r"transformer_i8: .* i8 が無い"):
            wan_plan(sources)

    def test_it_refuses_a_mixed_i4_series_in_the_i8_transformer_seat(self, tmp_path: Path) -> None:
        """i4 の混成系列は既定の格納が i8 なので「i8 を含む」を満たす — 禁止表だけが落とす。"""
        sources = _build_sources(
            tmp_path,
            containers={
                WAN_TRANSFORMER_I8_ROLE: _graph_container(WAN_TRANSFORMER_ROLE, storage="i4")
            },
        )
        with pytest.raises(DistError, match=r"transformer_i8: .* i4 がある"):
            wan_plan(sources)

    def test_it_refuses_an_f16_graph_in_the_text_encoder_seat(self, tmp_path: Path) -> None:
        """text_encoder の席に i8 でない系列（f16 の DiT 系列の取り違え）を挿した形。"""
        sources = _build_sources(
            tmp_path,
            containers={WAN_TEXT_ENCODER_ROLE: _graph_container(WAN_TEXT_ENCODER_ROLE)},
        )
        with pytest.raises(DistError, match=r"text_encoder: .* i8 が無い"):
            wan_plan(sources)

    def test_it_refuses_a_text_encoder_whose_tables_were_quantized(self, tmp_path: Path) -> None:
        """相対位置の表が i8 に丸まった umT5（「i8 を含む」は満たす — 束縛表の門だけが落とす）。"""
        sources = _build_sources(
            tmp_path,
            containers={
                WAN_TEXT_ENCODER_ROLE: umt5_container(provenance=_UMT5_PINNED, table_layout="i8")
            },
        )
        with pytest.raises(DistError, match="種類ごとの要求と違う"):
            wan_plan(sources)

    def test_the_forbidden_table_names_every_other_compressed_layout(self) -> None:
        """禁止表は役割ごとに、codec 台帳の layout から f32 / i32 と要求する格納を除いた**全部**を
        持つ。"""
        compressed = {entry.layout for entry in CODEC_LEDGER.values()} - {"f32", "i32"}
        assert set(WAN_STORAGE_FORBIDDEN) == set(WAN_STORAGE_REQUIREMENTS)
        assert set(WAN_STORAGE_REQUIREMENTS) == set(WAN_CONTAINER_ROLES)
        for role, required in WAN_STORAGE_REQUIREMENTS.items():
            assert set(WAN_STORAGE_FORBIDDEN[role]) == compressed - {required}, role


class TestTheTextEncoderContract:
    """umT5 の容器の入出力が Wan の text 段と DiT の文脈に噛み合うこと（ADR 0119 決定 3 / 4）。"""

    def test_it_refuses_an_encoder_whose_width_differs_from_the_dit_context(
        self, tmp_path: Path
    ) -> None:
        sources = _build_sources(
            tmp_path,
            containers={
                WAN_TEXT_ENCODER_ROLE: umt5_container(provenance=_UMT5_PINNED, width=_WIDTH + 1)
            },
        )
        with pytest.raises(DistError, match=f"出力の幅 {_WIDTH + 1}"):
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
        with pytest.raises(DistError, match="グラフ入力"):
            wan_plan(sources)

    def test_it_refuses_a_missing_encoder(self, tmp_path: Path) -> None:
        sources = _build_sources(tmp_path)
        for part in sources.text_encoder.parent.glob("model-*.krm"):
            part.unlink()
        with pytest.raises(DistError):
            wan_plan(sources)


class TestTheTokenizerAsset:
    """トークナイザ資産の門（形式・出所・版・最大長 —
    `wan.distribution.assert_umt5_tokenizer`）。"""

    def test_the_writer_and_the_gate_spell_the_same_format(self) -> None:
        """合成の資産は書き手の形式の版で作っている（門の受理形が書き手から外れていない）。"""
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
        with pytest.raises(DistError, match="上流の pin"):
            wan_plan(sources)

    @pytest.mark.parametrize("package", WAN_TOKENIZER_VERSIONS)
    def test_it_refuses_a_tokenizer_made_with_another_pinned_version(
        self, tmp_path: Path, package: str
    ) -> None:
        versions = {**json.loads(_tokenizer_asset())["versions"], package: "0.0.1"}
        sources = _build_sources(tmp_path, tokenizer=_tokenizer_asset(versions=versions))
        with pytest.raises(DistError, match="ピン"):
            wan_plan(sources)

    @pytest.mark.parametrize("max_length", [_ROWS + 1, 0, "512", True])
    def test_it_refuses_a_max_length_the_dit_context_cannot_hold(
        self, tmp_path: Path, max_length: object
    ) -> None:
        sources = _build_sources(tmp_path, tokenizer=_tokenizer_asset(maxLength=max_length))
        with pytest.raises(DistError, match="maxLength"):
            wan_plan(sources)

    def test_a_max_length_equal_to_the_context_rows_is_accepted(self, tmp_path: Path) -> None:
        """対（境界）: 行数ちょうどは通る（合成の既定がこの形）。"""
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
    """容器が名乗る出所を上流の pin（`wan.sources.SOURCES`）へ突き合わせる。"""

    def test_it_refuses_a_container_baked_from_another_revision(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        stamp_fixture_provenance(
            monkeypatch,
            Provenance(
                license=_PINNED.license, notice=NOTICE_FILENAME, upstream_revision=OTHER_REVISION
            ),
        )
        with pytest.raises(DistError, match="別の revision"):
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
        with pytest.raises(DistError, match=r"provenance\.license が 'cc-by-nc-4.0'"):
            wan_plan(_build_sources(tmp_path))

    def test_it_refuses_a_model_missing_from_the_source_table(self, tmp_path: Path) -> None:
        with pytest.raises(DistError, match="知らない"):
            wan_plan(_build_sources(tmp_path), "t2v-14b")

    def test_it_refuses_a_text_encoder_that_names_the_wan_checkpoint(self, tmp_path: Path) -> None:
        """umT5 の容器は本家 `google/umt5-xxl` の pin を名乗る（ADR 0122 決定 1）— Wan の pin を
        名乗る容器（出所を切り替える前の形）は、Wan の計画の中でも umT5 の門が落とす。"""
        encoder = umt5_container(provenance=_PINNED, width=_WIDTH)
        sources = _build_sources(tmp_path, containers={WAN_TEXT_ENCODER_ROLE: encoder})
        with pytest.raises(DistError, match="別の revision"):
            wan_plan(sources)

    def test_it_refuses_a_wan22_model_of_the_source_table(self, tmp_path: Path) -> None:
        """取得元の表に載った Wan2.2 のモデルでも、Wan2.1 の配布（`karume-wan2.1`）は組まない。"""
        assert "ti2v-5b" in SOURCES
        with pytest.raises(DistError, match=r"Wan2\.1 のモデル 'ti2v-5b' は知らない"):
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
        """素表を持たない DiT は、利用者の fromPretrained が重みを落とした後で落ちる形になる。"""
        sources = _build_sources(
            tmp_path,
            containers={role: _graph_container(WAN_TRANSFORMER_ROLE, storage=storage, rope=None)},
        )
        with pytest.raises(DistError, match=r"資産 'rope_base' が無い"):
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
        with pytest.raises(DistError, match="役割が 'ple-index'"):
            wan_plan(sources)

    def test_it_refuses_rope_base_tables_that_differ_between_the_seats(
        self, tmp_path: Path
    ) -> None:
        """f16 / i8 の容器は同じ幾何の素表を持つ — 片方だけ別の表なら、その席だけ RoPE が狂う。"""
        sources = _build_sources(
            tmp_path,
            containers={
                WAN_TRANSFORMER_I8_ROLE: _graph_container(
                    WAN_TRANSFORMER_ROLE, storage="i8", rope_payload=b"other-rope-table"
                )
            },
        )
        with pytest.raises(DistError, match="バイト同一でない"):
            wan_plan(sources)


class TestTheVaeChunkPair:
    """first / next は同じ組の chunk グラフ（TS の wanVaeChunkLayout と同じ規則）。"""

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
        """実物と同じ形: next は first に無い cache（time_conv の 2 本）を間に持つ。"""
        first = [self.LATENT, ("cache_00", [4, 2, 4, 4]), ("cache_02", [8, 2, 4, 4])]
        following = [*first[:2], ("cache_01", [3, 2, 4, 4]), first[2]]

        assert wan_plan(self._pair(tmp_path, first, following))

    def test_it_refuses_graphs_baked_with_different_tiles(self, tmp_path: Path) -> None:
        """片方だけ別のタイル辺で焼き直した組（--target の部分更新・途中で落ちた旧版の実走）。"""
        first = [self.LATENT, ("cache_00", [4, 2, 4, 4])]
        following = [(WAN_VAE_LATENT_INPUT, [16, 1, 8, 8]), ("cache_00", [4, 2, 8, 8])]
        with pytest.raises(DistError, match="潜在入力の形"):
            wan_plan(self._pair(tmp_path, first, following))

    def test_it_refuses_a_first_cache_that_next_lacks_or_shapes_differently(
        self, tmp_path: Path
    ) -> None:
        first = [self.LATENT, ("cache_00", [4, 2, 4, 4])]
        following = [self.LATENT, ("cache_00", [5, 2, 4, 4])]
        with pytest.raises(DistError, match=r"'cache_00' .* が next に同じ形で無い"):
            wan_plan(self._pair(tmp_path, first, following))

    def test_it_refuses_caches_in_another_order(self, tmp_path: Path) -> None:
        first = [self.LATENT, ("cache_00", [4, 2, 4, 4]), ("cache_01", [4, 2, 4, 4])]
        following = [self.LATENT, first[2], first[1]]
        with pytest.raises(DistError, match="順が next と違う"):
            wan_plan(self._pair(tmp_path, first, following))

    def test_it_refuses_a_graph_whose_first_input_is_not_the_latent(self, tmp_path: Path) -> None:
        first = [("cache_00", [4, 2, 4, 4]), self.LATENT]
        with pytest.raises(DistError, match="先頭のグラフ入力"):
            wan_plan(self._pair(tmp_path, first, [self.LATENT, first[0]]))


class TestTheTextEmbeddingAsset:
    def test_it_refuses_a_second_metadata_key(self, tmp_path: Path) -> None:
        sources = _build_sources(tmp_path, embeds=_text_embeds(extra_metadata={"other": "1"}))
        with pytest.raises(DistError, match="メタのキー"):
            wan_plan(sources)

    def test_it_refuses_embeddings_made_from_another_revision(self, tmp_path: Path) -> None:
        upstream = SOURCES[DEFAULT_MODEL]
        sources = _build_sources(
            tmp_path,
            embeds=_text_embeds(source={"repo": upstream.repo, "revision": OTHER_REVISION}),
        )
        with pytest.raises(DistError, match="上流の pin"):
            wan_plan(sources)

    def test_it_refuses_prompts_that_differ_from_the_fixed_table(self, tmp_path: Path) -> None:
        """カードは固定プロンプトの表から本文を描くので、資産のメタが違えば配らない。"""
        rows = _prompt_rows()
        rows[0] = {**rows[0], "prompt": rows[0]["prompt"] + " Extra."}
        sources = _build_sources(tmp_path, embeds=_text_embeds(prompts=rows))
        with pytest.raises(DistError, match="固定プロンプトの表"):
            wan_plan(sources)

    def test_it_refuses_a_width_that_differs_from_the_dit_context(self, tmp_path: Path) -> None:
        sources = _build_sources(tmp_path, embeds=_text_embeds(width=_WIDTH + 1))
        with pytest.raises(DistError, match="DiT の文脈入力"):
            wan_plan(sources)

    def test_it_refuses_an_i8_transformer_whose_context_differs(self, tmp_path: Path) -> None:
        """埋め込み資産は quant 非依存の 1 本 — i8 の席の DiT の文脈入力とも噛み合う必要がある。"""
        sources = _build_sources(
            tmp_path,
            containers={
                WAN_TRANSFORMER_I8_ROLE: _graph_container(
                    WAN_TRANSFORMER_ROLE, storage="i8", context=(1, _ROWS, _WIDTH + 1)
                )
            },
        )
        with pytest.raises(DistError, match="DiT の文脈入力"):
            wan_plan(sources)

    def test_it_refuses_more_valid_rows_than_the_dit_context_holds(self, tmp_path: Path) -> None:
        rows = _prompt_rows()
        rows[1] = {**rows[1], "tokens": _ROWS + 1}
        sources = _build_sources(tmp_path, embeds=_text_embeds(prompts=rows))
        with pytest.raises(DistError, match=f"有効長 1〜{_ROWS}"):
            wan_plan(sources)

    def test_it_refuses_a_tensor_whose_rows_differ_from_the_token_count(
        self, tmp_path: Path
    ) -> None:
        name = FIXED_PROMPTS[2].name
        sources = _build_sources(tmp_path, embeds=_text_embeds(shapes={name: (5, _WIDTH)}))
        with pytest.raises(DistError, match=f"'{name}' が F32"):
            wan_plan(sources)

    def test_it_refuses_a_missing_asset(self, tmp_path: Path) -> None:
        sources = _build_sources(tmp_path)
        sources.text_embeds.unlink()
        with pytest.raises(DistError):
            wan_plan(sources)

    def test_it_refuses_embeddings_from_an_encoder_in_another_dtype(self, tmp_path: Path) -> None:
        """NOTICE とカードは「bfloat16 の上流の encoder」と名乗る — f32 で作った資産は配らない。"""
        sources = _build_sources(tmp_path, embeds=_text_embeds(encoder_dtype="float32"))
        with pytest.raises(DistError, match="umT5 の dtype 'float32'"):
            wan_plan(sources)

    @pytest.mark.parametrize("package", sorted(WAN_TEXT_EMBEDS_VERSIONS))
    def test_it_refuses_embeddings_made_with_another_pinned_version(
        self, tmp_path: Path, package: str
    ) -> None:
        sources = _build_sources(tmp_path, embeds=_text_embeds(versions={package: "0.0.1"}))
        with pytest.raises(DistError, match="決定 4 の固定"):
            wan_plan(sources)

    @pytest.mark.parametrize(
        "normalized", [None, 123, ""], ids=["missing", "not-a-string", "empty"]
    )
    def test_it_refuses_a_row_without_a_normalized_text(
        self, tmp_path: Path, normalized: object
    ) -> None:
        """TS の parseWanTextEmbeds は normalized の無い行を拒む。

        利用者がダウンロードの後で落ちるので、配る前の門で同じ行を落とす。
        """
        rows = _prompt_rows()
        if normalized is None:
            del rows[1]["normalized"]
        else:
            rows[1] = {**rows[1], "normalized": normalized}
        sources = _build_sources(tmp_path, embeds=_text_embeds(prompts=rows))
        with pytest.raises(DistError, match=f"'{rows[1]['name']}' の normalized"):
            wan_plan(sources)

    @pytest.mark.parametrize("field", ["normalized", "prompt"])
    def test_it_refuses_a_normalized_text_that_another_row_owns(
        self, tmp_path: Path, field: str
    ) -> None:
        """行 1 の正規化後の文字列が行 0 の文字列と同じだと、どちらの埋め込みを使うかが
        決まらない。"""
        rows = _prompt_rows()
        rows[1] = {**rows[1], "normalized": rows[0][field]}
        sources = _build_sources(tmp_path, embeds=_text_embeds(prompts=rows))
        with pytest.raises(DistError, match=f"'{rows[1]['name']}' の文字列が '{rows[0]['name']}'"):
            wan_plan(sources)

    def test_a_row_whose_normalized_text_is_its_own_prompt_is_accepted(
        self, tmp_path: Path
    ) -> None:
        """原文が正規化で変わらない行（実物の boxing-cats）は通る。

        同じ行の中の一致は重複でない。
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
        """略称の対応は**必ず**出す（ADR 0074 決定 4）— `dit8` がどの部品の話か読めるように。"""
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

    def test_it_refuses_to_attribute_a_wan22_model(self, assembled) -> None:
        """取得元の表に有る Wan2.2 のモデルでも、Wan2.1 のカードには出所を書かない。"""
        _, manifest = assembled
        foreign = json.loads(json.dumps(manifest))
        foreign["models"] = {"ti2v-5b": foreign["models"][DEFAULT_MODEL]}
        assert "ti2v-5b" in SOURCES
        with pytest.raises(ValueError, match=r"'ti2v-5b' は Wan2\.1 のモデル"):
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
        assert "WanPipeline.fromPretrained(" in card
        assert "fromAssets" not in card

    def test_the_defaults_come_from_the_manifest(self, assembled) -> None:
        """既定値は焼き込まず pipelineConfig から描く（実物と違う値で組むと違う値が出る）。"""
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
        """カードの受理集合は TS の受理集合と同じ（反対側は wan_pipeline_test.ts）。"""
        fixture = json.loads(_CARD_LIMITS_FIXTURE.read_text(encoding="utf-8"))
        assert fixture == {
            "acceptedSizes": [
                {"width": width, "height": height} for width, height in WAN_ACCEPTED_SIZES
            ],
            "minFrames": WAN_FRAMES[0],
            "maxFrames": WAN_FRAMES[1],
        }, "card.py の WAN_ACCEPTED_SIZES / WAN_FRAMES を変えたら fixture と descriptor.ts も揃える"

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
        """ブラウザは実走した条件だけを名乗り、走らせていない経路と席は未実走と書く
        （ADR 0118 段 9）。"""
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
        """席ごとの transformer の行は実測のまま・計測していない欄は推し量らず未計測と名乗る。"""
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
        """表に無い席は数を推し量らず未計測と名乗る（席の並びは manifest のまま）。"""
        _, manifest = assembled
        changed = json.loads(json.dumps(manifest))
        quants = changed["models"][DEFAULT_MODEL]["quants"]
        quants["f16+other"] = quants[WAN_RESOURCE_QUANT]
        assert "f16+other" not in WAN_QUANT_TRANSFORMER
        card = _card(changed)
        assert "The other quants (`f16+other`) have not been measured yet." in card
        assert "| `f16+other` |" not in card.split("### Quants")[0]

    def test_it_refuses_a_manifest_without_the_measured_quant(self, assembled) -> None:
        """実測した席が無い配布形では資源の数を名乗らない（推し量った数を出さない）。"""
        _, manifest = assembled
        changed = json.loads(json.dumps(manifest))
        quants = changed["models"][DEFAULT_MODEL]["quants"]
        quants["i8"] = quants.pop(WAN_RESOURCE_QUANT)
        changed["models"][DEFAULT_MODEL]["defaultQuant"] = "i8"
        with pytest.raises(ValueError, match=f"quant '{WAN_RESOURCE_QUANT}'"):
            _card(changed)


class TestTheModelCardOnTheTextEncoder:
    """カードの text_encoder の記述（どこから取るか・経路の選択・取得量・宣言 limit）は
    manifest から。"""

    def test_it_names_the_repository_and_the_commit_it_borrows_from(self, assembled) -> None:
        out_dir, _ = assembled
        card = " ".join((out_dir / MODEL_CARD_FILENAME).read_text(encoding="utf-8").split())
        link = f"[`{_UMT5_REPO}`](https://huggingface.co/{_UMT5_REPO})"
        assert f"referenced from {link}" in card
        assert f"at commit `{_UMT5_REVISION[:16]}…` of {link}" in card
        assert "stored in this repository" not in card

    def test_a_self_contained_build_says_the_encoder_is_stored_here(self, tmp_path: Path) -> None:
        """越境参照なしで組んだ形（8 GiB の 1 リポ）でも、カードは事実どおりに描く。"""
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
        with pytest.raises(ValueError, match="requiredLimits が違う"):
            _card(changed)

    def test_it_refuses_a_manifest_without_the_text_encoder(self, assembled) -> None:
        _, manifest = assembled
        changed = json.loads(json.dumps(manifest))
        del _model(changed)["weights"][WAN_TEXT_ENCODER_ROLE]
        with pytest.raises(ValueError, match="text_encoder"):
            _card(changed)


class TestThePlaceholderRevisionGate:
    """仮の SHA（40 桁の 0）の門 — ドライバ `dist.py` が明示なしの焼きを書く前に拒む
    （ADR 0119 追記 E）。

    公開の焼き直し（release-runbook §0 — bump の後に必ず焼き直す）は明示を付けないので、参照先の
    実 SHA を渡さない限り通らない。
    """

    @staticmethod
    def _argv(root: Path, umt5_dir: Path, revision: str) -> list[str]:
        return [
            "--pipeline",
            "wan",
            "--series",
            str(root / "outputs" / "series"),
            "--out",
            str(root / "models" / WAN_REPO_NAME),
            "--ref-repo",
            _UMT5_REPO,
            "--ref-revision",
            revision,
            "--ref-dist",
            str(umt5_dir),
            "--ref-model",
            UMT5_DEFAULT_MODEL,
            "--ref-role",
            WAN_TEXT_ENCODER_ROLE,
        ]

    def _ready(self, tmp_path: Path) -> Path:
        return _assemble_umt5(tmp_path, _build_sources(tmp_path))

    def test_publishing_with_the_placeholder_is_refused_before_anything_is_written(
        self, tmp_path: Path
    ) -> None:
        umt5_dir = self._ready(tmp_path)
        argv = self._argv(tmp_path, umt5_dir, dist.PLACEHOLDER_REVISION)

        with pytest.raises(DistError, match=dist.ALLOW_PLACEHOLDER_FLAG):
            dist.main(argv)
        assert not (tmp_path / "models" / WAN_REPO_NAME).exists()

    def test_a_development_mirror_declares_the_placeholder_explicitly(self, tmp_path: Path) -> None:
        umt5_dir = self._ready(tmp_path)
        argv = self._argv(tmp_path, umt5_dir, dist.PLACEHOLDER_REVISION)

        dist.main([*argv, dist.ALLOW_PLACEHOLDER_FLAG])

        manifest = json.loads(
            (tmp_path / "models" / WAN_REPO_NAME / MANIFEST_FILENAME).read_text(encoding="utf-8")
        )
        parts = _model(manifest)["weights"][WAN_TEXT_ENCODER_ROLE]["i8"]["container"]["parts"]
        assert {ref["revision"] for ref in parts} == {dist.PLACEHOLDER_REVISION}

    def test_a_real_revision_needs_no_declaration(self, tmp_path: Path) -> None:
        umt5_dir = self._ready(tmp_path)

        dist.main(self._argv(tmp_path, umt5_dir, _UMT5_REVISION))

        assert verify_dist(tmp_path / "models" / WAN_REPO_NAME)

    def test_an_abbreviated_declaration_is_not_read(self, tmp_path: Path) -> None:
        """省略形は門も core も受けない（明示は綴り切る — `build_driver_parser` の doc）。"""
        umt5_dir = self._ready(tmp_path)
        argv = self._argv(tmp_path, umt5_dir, dist.PLACEHOLDER_REVISION)

        with pytest.raises(SystemExit):
            dist.main([*argv, "--allow-placeholder"])
        assert not (tmp_path / "models" / WAN_REPO_NAME).exists()

    def test_the_placeholder_is_forty_zeros(self) -> None:
        """hub の parse（40 桁の小文字 hex）を通る形 — だから形の門では止まらない。"""
        assert dist.PLACEHOLDER_REVISION == "0" * 40
        dist.assert_placeholder_intent(dist.PLACEHOLDER_REVISION, allowed=True)
        dist.assert_placeholder_intent(None, allowed=False)
        dist.assert_placeholder_intent(_UMT5_REVISION, allowed=False)


class TestTheWritersSpellTheSameNames:
    """配布 recipe は torch を読まないので綴りを自前で持つ — 書き手（torch を読む）と一致する。"""

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
        """門の固定値は pyproject の `wan` グループの `==` ピンと同じ。

        ピンは書き手が記録する版の出所（`uv run --group wan` が入れる版）。
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

    def test_the_rope_base_asset_and_the_context_input(self) -> None:
        from wan import export_dit

        assert (export_dit.ROPE_BASE_ASSET, export_dit.ROPE_BASE_ROLE) == (
            WAN_ROPE_BASE_ASSET,
            WAN_ROPE_BASE_ROLE,
        )
        assert export_dit.MODEL_FILE == WAN_MODEL_FILE
        assert WAN_DIT_CONTEXT_INPUT in export_dit.INPUT_NAMES

    def test_the_wan22_series_names(self) -> None:
        """Wan2.2 の DiT の i8 系列と VAE の f16 系列（書き手 `wan.ti2v_export_dit` /
        `wan.export_vae --model ti2v-5b`）。"""
        from wan import export_vae, ti2v_export_dit

        assert ti2v_export_dit.SERIES_NAME == WAN22_I8_SERIES
        assert ti2v_export_dit.SERIES.name == WAN22_I8_SERIES
        assert export_vae.TI2V_SERIES_NAME == WAN22_SERIES
        assert export_vae.VAE_SERIES[_TI2V].series == WAN22_SERIES
        assert ti2v_export_dit.MODEL_FILE == WAN_MODEL_FILE
        assert WAN_DIT_CONTEXT_INPUT in ti2v_export_dit.INPUT_NAMES

    def test_the_wan22_vae_encoder_graphs(self) -> None:
        """VAE の encoder の 3 グラフ（書き手 `wan.export_vae_encoder` — decoder と同じ f16
        系列）。"""
        from wan import export_vae_encoder

        assert export_vae_encoder.TARGETS == WAN_VAE_ENCODER_ROLES
        assert export_vae_encoder.SERIES_NAME == WAN22_SERIES == WAN22.series
        assert export_vae_encoder.MODEL_FILE == WAN_MODEL_FILE


class TestTheGenerationTables:
    """世代の表から導く配置・格納・weights の表（Wan2.1 は全世代の表から Wan2.2 だけの VAE
    encoder を除いたもの — 2.1 の配布形が不変）。"""

    def test_the_wan21_tables_are_the_full_tables_without_the_vae_encoder(self) -> None:
        encoder = set(WAN_VAE_ENCODER_ROLES)
        assert WAN21.vae_encoder_roles == ()
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
            name: labels for name, labels in WAN_WEIGHTS.items() if name not in encoder
        }
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
            WAN_VAE_NEXT_ROLE,
        )
        assert list(WAN22.weights[WAN_TRANSFORMER_ROLE]) == ["i8"]
        assert set(WAN22.storage_requirements) == set(WAN22.container_roles)
        # f16 の DiT を i8 の席へ挿す取り違えは、要求と禁止の両側で落ちる。
        assert "f16" in WAN22.storage_forbidden[WAN_TRANSFORMER_I8_ROLE]
        assert "f16" not in WAN22.quants

    def test_only_wan22_carries_the_vae_encoder_in_f16(self) -> None:
        """encoder の 3 グラフは Wan2.2 だけが配り（ADR 0121 決定 11）、decoder と同じ扱い:
        f16 系列・格納ラベル f16 の 1 本・`<部品>/model.f16.krm`・f16 を要求し他の圧縮格納を
        禁ずる。"""
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
        # 世代の weights の並びは全世代の表の並び（I2V の段の順 — encoder が先頭）。
        assert list(WAN22.weights) == list(WAN_GRAPH_ROLES)

    def test_the_driver_offers_wan_ti2v_right_after_wan(self) -> None:
        names = list(dist.PIPELINES)
        assert names[names.index("wan") + 1] == "wan-ti2v"
        assert dist.PIPELINES["wan-ti2v"] is TI2V_PIPELINE


class TestTheWan22Distribution:
    """`--pipeline wan-ti2v` の配布形（ADR 0121 段 8 — `karume-wan2.2`・モデル `ti2v-5b`）。"""

    def test_it_places_the_int8_transformer_the_vae_and_the_assets(self, assembled22) -> None:
        """transformer は i8 の 1 本だけ・VAE の decoder と encoder は 1 本ずつ・資産 2 本。
        text_encoder は越境参照。encoder の golden（`vae_encoder.*`）は配らない。"""
        out_dir, _ = assembled22
        own = {
            role: WAN_OUTPUT_PATHS[role]
            for role in (
                *WAN_VAE_ENCODER_ROLES,
                WAN_TRANSFORMER_I8_ROLE,
                WAN_VAE_FIRST_ROLE,
                WAN_VAE_NEXT_ROLE,
                WAN_TEXT_EMBEDS_ROLE,
                WAN_TOKENIZER_ROLE,
            )
        }
        expected = [
            f"{_TI2V}/{rel}"
            for rel in placed_paths(
                own,
                {name: labels for name, labels in WAN22.weights.items() if name != UMT5_ROLE},
                {WAN_TRANSFORMER_I8_ROLE: 4},
            )
        ]
        assert _present(out_dir) == sorted(
            [*expected, MANIFEST_FILENAME, MODEL_CARD_FILENAME, "LICENSE.md", NOTICE_FILENAME]
        )
        assert list((out_dir / _TI2V / WAN_TRANSFORMER_ROLE).glob("model.f16*")) == []
        assert list(out_dir.rglob(f"{WAN_TEXT_ENCODER_ROLE}/*")) == []
        assert list(out_dir.rglob("io.*")) == []
        assert list(out_dir.rglob("pipeline_steps.*")) == []
        assert list(out_dir.rglob("vae_encoder.*")) == []

    def test_the_manifest_declares_the_two_seats(self, assembled22) -> None:
        """席は参照席と実用席の 2 つ・f16 席は無い・既定は実用席（視認の裁定 2026-10-06 —
        ADR 0121 追記「段 8a の結果」）。どの席も VAE の encoder の f16 を取る（T2V だけの利用でも
        取る — 追記「段 9 の計画の裁定と段 9a の結果」の裁定）。weights の並びは I2V の段の順。"""
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
        ]
        for role in WAN_VAE_ENCODER_ROLES:
            assert list(model["weights"][role]) == ["f16"]
        assert list(model["weights"][WAN_TEXT_ENCODER_ROLE]) == ["i8"]
        assert list(model["weights"][WAN_TRANSFORMER_ROLE]) == ["i8"]
        assert list(model["weights"][WAN_VAE_FIRST_ROLE]) == ["f16"]
        assert list(model["weights"][WAN_VAE_NEXT_ROLE]) == ["f16"]
        assert model["assets"][WAN_TEXT_EMBEDS_ROLE]["path"] == (
            f"{_TI2V}/{WAN_TEXT_EMBEDS_ROLE}/{WAN_TEXT_EMBEDS_FILE}"
        )
        assert model["assets"][WAN_TOKENIZER_ROLE]["path"] == (
            f"{_TI2V}/{WAN_TOKENIZER_ROLE}/{umt5_tokenizer.ASSET_FILE}"
        )
        assert list(model["quants"]) == [_REFERENCE, _PRACTICAL]
        assert model["defaultQuant"] == _PRACTICAL
        seat = {
            **dict.fromkeys(WAN_VAE_ENCODER_ROLES, "f16"),
            WAN_TEXT_ENCODER_ROLE: "i8",
            WAN_TRANSFORMER_ROLE: "i8",
            WAN_VAE_FIRST_ROLE: "f16",
            WAN_VAE_NEXT_ROLE: "f16",
        }
        assert model["quants"][_REFERENCE]["weights"] == seat
        assert model["quants"][_PRACTICAL]["weights"] == seat
        assert model["quants"][_REFERENCE]["session"] == {}
        assert model["quants"][_PRACTICAL]["session"] == {
            "linearCompute": "a8",
            "attentionCompute": "a8",
            "attentionScoreStorage": "f16",
        }

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
        """資産 2 本は Wan2.1 の系列のファイルそのもの（ADR 0121 決定 9 — 配布形でもバイト
        同一）。"""
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
        assert "In text-to-video, the only mode this distribution runs, the mask is all false" in (
            prose
        )
        assert "(`patch_size` 2, 12 channels)" in prose
        # VAE の encoder は配布形に載る — 改変（3 グラフ・最後の時間スライス・f16 の丸め）を告げる。
        assert "**f16 VAE encoder for image-to-video**" in prose
        assert "re-expressed as three graphs" in prose
        assert "so only that slice is stored" in prose
        assert "The VAE encoder is not included." not in prose
        assert "**The text encoder is referenced, not stored here.**" in prose
        assert f"`{UMT5_REPO_NAME}` at a pinned commit" in prose
        assert "umT5-XXL encoder of the Wan2.1 T2V 1.3B checkpoint" in prose
        # 2.1 の文の写しで、2.2 では事実が違う箇所が残っていない。
        assert "float16 transformer" not in prose
        assert "second copy of the transformer" not in prose
        assert "Wan2.1 T2V 1.3B checkpoint listed" not in prose


class TestTheWan22ProvenanceSplit:
    """出所の門の分割（ADR 0121 決定 9）: DiT / VAE の容器は Wan2.2 の pin、テキスト資産 2 本は
    Wan2.1 の pin で見る。取り違えはどちら向きでも落ちる。"""

    def test_the_split_pins_pass(self, tmp_path: Path, wan22_provenance) -> None:
        plan = wan_plan(_build_sources22(tmp_path), generation=WAN22)
        assert plan.name == _TI2V
        assert plan.pipeline == WAN22_PIPELINE

    def test_it_refuses_embeddings_that_name_the_wan22_pin(
        self, tmp_path: Path, wan22_provenance
    ) -> None:
        sources = _build_sources22(tmp_path, embeds=_text_embeds(source=_pin(_TI2V)))
        with pytest.raises(DistError, match=r"埋め込みの出所 .* が上流の pin"):
            wan_plan(sources, generation=WAN22)

    def test_it_refuses_a_tokenizer_that_names_the_wan22_pin(
        self, tmp_path: Path, wan22_provenance
    ) -> None:
        source = {**_pin(_TI2V), "subfolder": WAN_TOKENIZER_SUBFOLDER}
        sources = _build_sources22(tmp_path, tokenizer=_tokenizer_asset(source=source))
        with pytest.raises(DistError, match=r"トークナイザの出所 .* が上流の pin"):
            wan_plan(sources, generation=WAN22)

    @pytest.mark.parametrize(
        "role",
        [WAN_TRANSFORMER_I8_ROLE, WAN_VAE_FIRST_ROLE, WAN_VAE_NEXT_ROLE, *WAN_VAE_ENCODER_ROLES],
    )
    def test_it_refuses_a_container_baked_from_the_wan21_pin(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, wan22_provenance, role: str
    ) -> None:
        container = _stamped(monkeypatch, _PINNED, lambda: _default_container(role))
        sources = _build_sources22(tmp_path, containers={role: container})
        injected = re.escape(str(wan_placements(sources, WAN22)[role].parent))
        with pytest.raises(DistError, match=rf"{injected}.*別の revision"):
            wan_plan(sources, generation=WAN22)

    def test_the_wan21_plan_still_reads_both_from_the_wan21_pin(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """対: Wan2.1 の計画は DiT / VAE もテキスト資産も Wan2.1 の pin で見る（Wan2.2 の pin の
        DiT は落ちる）。"""
        assert WAN21.text_model == DEFAULT_MODEL
        dit = _stamped(
            monkeypatch,
            _PINNED22,
            lambda: _graph_container(WAN_TRANSFORMER_ROLE, storage="i8"),
        )
        sources = _build_sources(tmp_path, containers={WAN_TRANSFORMER_I8_ROLE: dit})
        with pytest.raises(DistError, match="別の revision"):
            wan_plan(sources)


class TestTheWan22Gates:
    def test_it_refuses_an_f16_transformer_in_the_int8_seat(
        self, tmp_path: Path, wan22_provenance
    ) -> None:
        sources = _build_sources22(
            tmp_path,
            containers={WAN_TRANSFORMER_I8_ROLE: _graph_container(WAN_TRANSFORMER_ROLE)},
        )
        with pytest.raises(DistError, match=r"transformer_i8: .* i8 が無い"):
            wan_plan(sources, generation=WAN22)

    def test_it_refuses_a_mixed_i4_series_in_the_int8_seat(
        self, tmp_path: Path, wan22_provenance
    ) -> None:
        """「i8 を含む」は満たす混成系列 — 禁止表だけが落とす（2.1 と同じ門）。"""
        sources = _build_sources22(
            tmp_path,
            containers={
                WAN_TRANSFORMER_I8_ROLE: _graph_container(WAN_TRANSFORMER_ROLE, storage="i4")
            },
        )
        with pytest.raises(DistError, match=r"transformer_i8: .* i4 がある"):
            wan_plan(sources, generation=WAN22)

    def test_an_f16_transformer_beside_the_vae_is_never_carried(
        self, tmp_path: Path, wan22_provenance
    ) -> None:
        """f16 系列に DiT が置かれていても、Wan2.2 の配置表は拾わない（f16 の DiT は配らない）。"""
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

    @pytest.mark.parametrize("role", WAN_VAE_ENCODER_ROLES)
    @pytest.mark.parametrize("storage", ["f32", "i8"])
    def test_it_refuses_a_vae_encoder_without_f16_storage(
        self, tmp_path: Path, wan22_provenance, role: str, storage: str
    ) -> None:
        """encoder の席に素の f32 / 別格納の系列を挿した取り違え（decoder と同じ門）。"""
        sources = _build_sources22(
            tmp_path, containers={role: _graph_container(role, storage=storage)}
        )
        with pytest.raises(DistError, match=rf"{role}: .* f16 が無い"):
            wan_plan(sources, generation=WAN22)

    @pytest.mark.parametrize("role", WAN_VAE_ENCODER_ROLES)
    def test_it_refuses_a_series_without_one_of_the_vae_encoder_graphs(
        self, tmp_path: Path, wan22_provenance, role: str
    ) -> None:
        """encoder は T2V だけの利用でも取る部品 — 1 本でも欠けた系列からは組まない。"""
        sources = _build_sources22(tmp_path)
        missing = wan_placements(sources, WAN22)[role]
        for part in missing.parent.iterdir():
            part.unlink()
        missing.parent.rmdir()
        with pytest.raises(DistError, match=rf"組み立ての入力が無い: .*{role}"):
            wan_plan(sources, generation=WAN22)

    def test_it_refuses_a_wan21_model(self, tmp_path: Path, wan22_provenance) -> None:
        with pytest.raises(DistError, match=r"Wan2\.2 のモデル 't2v-1\.3b' は知らない"):
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
            DistError, match=r"資産 'rope_base' が無い.*`python -m wan\.ti2v_export_dit write`"
        ):
            wan_plan(sources, generation=WAN22)

    def test_a_vae_pair_of_different_tiles_points_to_the_wan22_writer(
        self, tmp_path: Path, wan22_provenance
    ) -> None:
        """共有の門の焼き直しの案内は世代の書き手を名指しする（2.1 の `export_vae` の既定は 2.1 の
        系列を書くので、2.2 の組には効かない）。"""
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
            DistError, match=r"潜在入力の形.*`python -m wan\.export_vae --model ti2v-5b`"
        ):
            wan_plan(sources, generation=WAN22)


class TestTheWan22ModelCard:
    def test_it_is_the_only_profile_and_is_resolved_without_a_choice(self) -> None:
        profiles = TI2V_PIPELINE.card_profiles
        assert list(profiles) == ["wan-ti2v"]
        assert resolve_card_renderer(TI2V_PIPELINE, None) is profiles["wan-ti2v"]

    def test_the_accepted_inputs_match_the_typescript_side(self) -> None:
        """カードの受理集合は TS の受理集合（`WAN22_TI2V_GENERATION`）と同じ fixture を挟んで
        一致。"""
        fixture = json.loads(_TI2V_CARD_LIMITS_FIXTURE.read_text(encoding="utf-8"))
        assert fixture == {
            "acceptedSizes": [
                {"width": width, "height": height} for width, height in WAN22_ACCEPTED_SIZES
            ],
            "minFrames": WAN22_FRAMES[0],
            "maxFrames": WAN22_FRAMES[1],
        }, "card.py の WAN22_ACCEPTED_SIZES / WAN22_FRAMES を変えたら fixture と TS 側も揃える"

    def test_it_describes_wan22(self, assembled22) -> None:
        out_dir, _ = assembled22
        card = (out_dir / MODEL_CARD_FILENAME).read_text(encoding="utf-8")
        upstream = SOURCES[_TI2V]
        assert "# Wan2.2 TI2V 5B — Karume" in card
        assert f"base_model: {upstream.repo}" in card
        assert f"at commit `{upstream.revision}`" in card
        assert f'repo: "hdae/{WAN22_REPO_NAME}"' in card
        assert "WanTi2vPipeline.fromPretrained(" in card
        assert "import { encodePng, wanFrameToRgba, WanTi2vPipeline }" in card
        assert "- **size**: 1280 × 704 or 704 × 1280." in card
        assert "- **frames**: 4n+1 from 5 to 121." in card
        assert "  // width: 1280, height: 704, // or 704 × 1280" in card
        assert f"| `{_PRACTICAL}` (default) |" in card.split("### Quants")[1]
        assert f"implements `{WAN22_SUPPORTED_PIPELINE}`" in card
        prose = " ".join(card.split())
        assert "Text to video only." in prose
        assert "at 24 fps" in prose
        assert "- Seven graphs:" in prose
        assert "and the VAE encoder is included, but image-to-video is not available yet." in prose

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
        """Usage の既定のコメントは世代の既定（`descriptor.ts` の `defaults` の写し）から描き、
        既定は受理集合の内。"""
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
        # 2.2 にも Chrome の通しはある（121 フレーム）が、2.1 のブラウザの通しの文は写さない。
        assert "46.1 minutes" not in card
        assert "`f16` quant with the precomputed embeddings" not in " ".join(card.split())
        assert "WanPipeline.fromPretrained(" not in card

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
        # 2.1 の改変の要約の写し（f16 の transformer と、その 2 本目の int8 版）が残っていない。
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
        assert set(WAN22_RESOURCES) == {_REFERENCE, _PRACTICAL}
        prose = " ".join(card.split())
        assert (
            "- **Quality of `f16+dit8-a8-attn8-s16`**: after the first step its latent differs from"
            " `f16+dit8`'s (the same int8 weights, computed in float32) by a relative RMS error of"
            " 0.044 at 33 frames and 0.049 at 121 frames. Side by side with `f16+dit8` on twelve"
            " 50-step clips at 1280 × 704 with 33 frames (seeds 42 to 45 with the three fixed"
            " prompts), no clear degradation was seen."
        ) in prose
        # 1 forward は段 2（2026-10-04）・通しは 2026-10-05。49 フレームは製品の class でない。
        assert "timed on its own on 2026-10-04" in prose
        assert "on 2026-10-05, at 1280 × 704 with 50 steps" in prose
        assert "The 33-frame runs went through `WanTi2vPipeline`" in prose
        assert "the 49-frame runs drove the same pipeline stages from a development script" in prose
        assert "4.36 GiB at 33 frames and 4.51 GiB at 49 frames with either quant" in prose
        assert "4.52" not in prose
        assert "have not been measured yet" not in prose
        assert "has been measured with this distribution only in the 121-frame run above" in prose
        assert "`maxStorageBufferBindingSize` (128 MiB)" in card

    def test_it_names_the_121_frame_row_with_its_own_gpu_and_reading(self, assembled22) -> None:
        """121 フレームの行は実用席だけ・RTX 3080 Ti の開発機の通しで、B570 の行と GPU も測り方
        （nvidia-smi の GPU 全体）も違い、熱制限込みだと名乗る。参照席の 121 は回していない。"""
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
        # 121 の行は寸法・ステップ・shift・guidance だけが同じ（プロンプトと text の経路は違う）。
        assert (
            "the 121-frame row at the same size, steps, shift and guidance on an NVIDIA GeForce"
            " RTX 3080 Ti (12 GiB) in Deno 2.9.6 on 2026-10-06" in prose
        )
        assert "same settings" not in prose
        assert "another prompt (`cat-dog-baking`, where the B570 rows used `boxing-cats`)" in prose
        assert "it is not comparable with the B570 rows" in prose
        assert "the memory in use on the whole GPU as read by nvidia-smi, not the driver's" in prose
        assert "its times include thermal throttling" in prose
        # 2 ステップ × 121 の参照席の sha 行は別にあり得るので、50 ステップに限って名乗る。
        assert (
            "The 121-frame clip has not been run for 50 steps with the `f16+dit8` quant." in prose
        )
        # 測り方の定義文は B570 の行に限る（121 の行は nvidia-smi）。
        assert (
            "For the B570 rows, Transformer peak is the total allocation during the transformer"
            " stage (the driver's fdinfo)." in prose
        )
        assert (
            "- **GPU memory**: the B570 rows' peaks are of the total allocation (the driver's"
            " fdinfo), and the 121-frame row's is the memory in use on the whole GPU (nvidia-smi)."
            in prose
        )
        assert "- **GPU memory**: the peaks are of the total allocation" not in prose
        assert "about 1.46 GiB at 121 frames (S = 27,280)" in prose
        # 2 GiB の束縛上限に最も近い束縛（スコアの行ブロック）と、Chrome が与えた上限の実数。
        assert "the largest binding at 121 frames is about 1.96 GiB" in prose
        assert "bindings of up to 2,147,483,644 bytes" in prose
        assert "2 GiB buffers" not in prose

    def test_it_says_121_frames_fit_12_gib_and_the_b570_is_checked_only_up_to_57_frames(
        self, assembled22
    ) -> None:
        """受理の上限 121 は開発機（RTX 3080 Ti・12 GiB）と RTX 5070 Ti の Chrome で完走した事実で
        名乗り、山は nvidia-smi の GPU 全体で Deno の余裕はもっと薄いことがあると限る。B570 で
        確かめたのは 57 フレームまでで、それより長いクリップは回していないので非対応。入らない
        見込み（推測）は見積りのある 2 点（81・121）だけで、残りの長さは分からないと書く。121 だけを
        非対応と書くと 61〜117 が回ると読めるので、範囲で書く。入らなければ admission ではなく
        実行の途中で落ちると書く。"""
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
        # 121 だけを非対応と書くと 61〜117 が 10 GB 級で回ると読める — 範囲で書く。
        assert "121 frames are not supported" not in prose
        # 見積りは 81 と 121 の 2 点だけ — 61〜77 / 85〜117 を「入らない見込み」とは書かない。
        assert "not expected to fit" not in prose
        assert (
            "A memory estimate made before the runs covers only two of those lengths and puts both"
            " beyond that ceiling: the next length after 77 frames with the"
            " `f16+dit8-a8-attn8-s16` quant, and 121 frames with both quants (the 10.64 GiB the"
            " transformer stage allocated at 121 frames on the RTX 3080 Ti is above it too). The"
            " other lengths were not estimated, so whether they fit is not known." in prose
        )
        # 受理集合は機ごとではない — 10 GB 級でも ModelInputError で先に拒まれるとは読ませない。
        assert (
            "The accepted set does not depend on the GPU, so such a request is still accepted, and"
            " if it does not fit it fails during the run — with an out-of-memory error"
            " (`GpuOutOfMemoryError`), or a lost device near the limit — rather than with"
            " `ModelInputError`, and the time spent on the stages before it is lost." in prose
        )
        # ブラウザの通しは別の GPU・runtime の値と名乗り、走っていない組は未実走と書く。
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
        assert "The other quants (`f16+other`) have not been measured yet." in card
        assert "| `f16+other` |" not in card.split("### Quants")[0]

    def test_it_refuses_a_manifest_without_a_measured_seat(self, assembled22) -> None:
        _, manifest = assembled22
        changed = json.loads(json.dumps(manifest))
        model = changed["models"][_TI2V]
        model["quants"] = {"i8": model["quants"][_REFERENCE]}
        model["defaultQuant"] = "i8"
        with pytest.raises(ValueError, match="1 つも無い"):
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
        with pytest.raises(ValueError, match=r"'t2v-1\.3b' は Wan2\.2 のモデル"):
            _card22(foreign_model)
        foreign_pipeline = json.loads(json.dumps(manifest))
        foreign_pipeline["models"][_TI2V]["pipeline"] = WAN_PIPELINE
        with pytest.raises(ValueError, match=WAN22_SUPPORTED_PIPELINE):
            _card22(foreign_pipeline)

    def test_the_wan21_card_refuses_a_wan22_manifest(self, assembled22) -> None:
        _, manifest = assembled22
        with pytest.raises(ValueError, match=WAN_SUPPORTED_PIPELINE):
            _card(manifest)


_REAL = wan_sources(SERIES_ROOT)
_REAL_PRESENT = (
    (_REAL.series / WAN_TRANSFORMER_ROLE).is_dir()
    and (_REAL.i8_series / WAN_TRANSFORMER_ROLE).is_dir()
    and _REAL.text_embeds.is_file()
    and _REAL.text_encoder.parent.is_dir()
    and _REAL.tokenizer.is_file()
)


@pytest.mark.skipif(
    not _REAL_PRESENT,
    reason=f"実物の系列が無い: {_REAL.series} / {_REAL.i8_series} / {_REAL.text_encoder.parent}",
)
class TestTheRealSeries:
    """実物の系列（`wan.export_dit`〈f16 / i8〉/ `wan.export_vae` / `wan.text_embeds` /
    `wan.umt5_export` / `wan.umt5_tokenizer` の出力）で計画が組める。

    読むのは容器の 2 文書・束縛表・資産（宣言と RoPE の素表）と、埋め込み資産のヘッダ・トークナイザ
    資産（約 8 MB）だけ（重みの payload は読まない）。
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
_REAL22_PRESENT = (
    (_REAL22.i8_series / WAN_TRANSFORMER_ROLE).is_dir()
    and (_REAL22.series / WAN_VAE_FIRST_ROLE).is_dir()
    and (_REAL22.series / WAN_VAE_NEXT_ROLE).is_dir()
    and _REAL22.text_embeds.is_file()
    and _REAL22.text_encoder.parent.is_dir()
    and _REAL22.tokenizer.is_file()
)


@pytest.mark.skipif(
    not _REAL22_PRESENT,
    reason=f"Wan2.2 の実物の系列が無い: {_REAL22.series} / {_REAL22.i8_series} /"
    f" {_REAL22.text_encoder.parent}",
)
class TestTheRealWan22Series:
    """実物の Wan2.2 の系列（`wan.ti2v_export_dit` / `wan.export_vae --model ti2v-5b`）と、Wan2.1 の
    テキスト資産・umT5 のミラーで計画が組める（読むのは宣言・束縛表・RoPE の素表・資産のヘッダ
    だけ）。"""

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
        assert plan.default_quant == _PRACTICAL

    def test_the_text_assets_are_the_wan21_series_files(self) -> None:
        assert (_REAL22.text_embeds, _REAL22.tokenizer) == (_REAL.text_embeds, _REAL.tokenizer)
