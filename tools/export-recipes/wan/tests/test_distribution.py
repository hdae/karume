"""Wan2.1 の配布 recipe（`wan.distribution`）とカード（`wan.card`）— 組み立て 1 周ぶんの単体テスト。

組み立てへ届く入力は数 KB の**正当な最小コンテナ**（`ir_fixtures`・umT5 は `umt5_fixture`）と、
書き手（`wan.text_embeds` / `wan.umt5_tokenizer`）と同じ形の合成の資産で作る。門に落とされることを
見るケースも同じ器で作り、**宣言だけを実物とずらす**。`text_encoder` は公開と同じ形（umT5 の配布形を
先に組み、そこへの越境参照）で組む。

実物の系列（`outputs/series/`）がある機では、実物で計画を 1 周組む門も回す（無ければ SKIP）。
"""

from __future__ import annotations

import json
import tomllib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import ir_fixtures
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
    WAN_VAE_FIRST_ROLE,
    WAN_VAE_LATENT_INPUT,
    WAN_VAE_NEXT_ROLE,
    WAN_WEIGHTS,
    WanSources,
    wan_placements,
    wan_plan,
    wan_sources,
)
from wan.prompts import FIXED_PROMPTS
from wan.sources import DEFAULT_MODEL, SOURCES
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

#: `pipelineConfig` の欄（TS 側 `packages/models/src/wan/config.ts` の `ROOT_KEYS` /
#: `SCHEDULER_KEYS` / `DEFAULTS_KEYS` の写し）。ロード側は未知キーも欠落も parse 時に落とすので、
#: 焼く側とロード側の欄名は完全一致が要る。
_CONFIG_KEYS = {"scheduler": ("shift",), "defaults": ("steps", "guidance")}

#: TS 側の受理集合の写し（`packages/models/tests/wan_pipeline_test.ts` が `pipeline.ts` の値と
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
    umT5 の i8 系列）。出所は `ir_fixtures` が焼く値（テストの間だけ差し替わる）に揃える。"""
    if role == WAN_TEXT_ENCODER_ROLE:
        return umt5_container(provenance=ir_fixtures.FIXTURE_PROVENANCE, width=_WIDTH)
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
) -> WanSources:
    """系列 5 本を偽資産で再現する（配布しない golden の混入込み）。`containers` の鍵は配置の
    役割（`text_encoder` / `transformer_f16` / `transformer_i8` / VAE の 2 本）。"""
    sources = wan_sources(root / "outputs" / "series")
    placements = wan_placements(sources)
    for role in WAN_CONTAINER_ROLES:
        write_component(placements[role], containers.get(role) or _default_container(role))
    # 配布に入ってはいけない golden（系列には実際にこれらが並んでいる）。
    for series in (sources.series, sources.i8_series):
        (series / WAN_TRANSFORMER_ROLE / "io.band-s00192-t0999.safetensors").write_bytes(b"io")
    (sources.series / "pipeline_steps.band-boxing-cats.safetensors").write_bytes(b"steps")
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


class TestLayout:
    def test_it_places_the_graphs_and_the_assets_under_the_model_subtree(self, assembled) -> None:
        """transformer は格納ラベルごとに 2 本（`model.f16.krm` / `model.i8.krm`）・VAE は
        1 本ずつ・資産 2 本。text_encoder は越境参照なので 1 バイトも置かない。"""
        out_dir, _ = assembled
        own = {role: path for role, path in WAN_OUTPUT_PATHS.items() if role != UMT5_ROLE}
        expected = [
            f"{DEFAULT_MODEL}/{rel}"
            for rel in placed_paths(
                own,
                {name: labels for name, labels in WAN_WEIGHTS.items() if name != UMT5_ROLE},
                # transformer の容器は資産 `rope_base` の専用 part が 1 本増える。
                {WAN_TRANSFORMER_F16_ROLE: 4, WAN_TRANSFORMER_I8_ROLE: 4},
            )
        ]
        assert _present(out_dir) == sorted(
            [*expected, MANIFEST_FILENAME, MODEL_CARD_FILENAME, "LICENSE.md", NOTICE_FILENAME]
        )
        assert list(out_dir.rglob(f"{WAN_TEXT_ENCODER_ROLE}/*")) == []
        assert WAN_OUTPUT_PATHS[WAN_TRANSFORMER_F16_ROLE] == "transformer/model.f16.krm"
        assert WAN_OUTPUT_PATHS[WAN_TRANSFORMER_I8_ROLE] == "transformer/model.i8.krm"
        assert WAN_OUTPUT_PATHS[WAN_TOKENIZER_ROLE] == "umt5_tokenizer/tokenizer.json"

    def test_it_never_carries_the_series_goldens(self, assembled) -> None:
        out_dir, _ = assembled
        assert list(out_dir.rglob("io.*")) == []
        assert list(out_dir.rglob("pipeline_steps.*")) == []
        assert list(out_dir.rglob("reference.*")) == []

    def test_the_manifest_declares_the_three_seats_and_the_model_assets(self, assembled) -> None:
        """quant 席は `f16`・参照席 `f16+dit8`・実用席 `f16+dit8-a8-attn8-s16`（既定）の 3 つ
        （ADR 0120 決定 1・裁定 2026-10-04 の 4）で、どの席も text_encoder の i8 を選ぶ（weights は
        完全写像 — ADR 0119 追記 B）。資産はモデル単位の `text_embeds` と `umt5_tokenizer`
        （quant 非依存 — 同 C）。"""
        _, manifest = assembled
        model = _model(manifest)
        assert manifest["defaultModel"] == DEFAULT_MODEL
        assert model["pipeline"] == WAN_PIPELINE
        assert list(model["weights"]) == list(WAN_GRAPH_ROLES)
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
                WAN_TEXT_ENCODER_ROLE: umt5_container(
                    provenance=ir_fixtures.FIXTURE_PROVENANCE, width=_WIDTH + 1
                )
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
                WAN_TEXT_ENCODER_ROLE: umt5_container(
                    provenance=ir_fixtures.FIXTURE_PROVENANCE, table_layout="i8"
                )
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
                WAN_TEXT_ENCODER_ROLE: umt5_container(
                    provenance=ir_fixtures.FIXTURE_PROVENANCE, width=_WIDTH + 1
                )
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
                WAN_TEXT_ENCODER_ROLE: umt5_container(
                    provenance=ir_fixtures.FIXTURE_PROVENANCE, inputs=renamed
                )
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

    def test_it_refuses_a_pipeline_it_does_not_describe(self, assembled) -> None:
        _, manifest = assembled
        foreign = json.loads(json.dumps(manifest))
        foreign["models"][DEFAULT_MODEL]["pipeline"] = "siglip2/1"
        with pytest.raises(ValueError, match=WAN_SUPPORTED_PIPELINE):
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
        }, "card.py の WAN_ACCEPTED_SIZES / WAN_FRAMES を変えたら fixture と pipeline.ts も揃える"

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

    def test_it_names_the_measured_figures_of_every_seat(self, assembled) -> None:
        """席ごとの transformer の行は実測のまま・計測していない欄は推し量らず未計測と名乗る。"""
        out_dir, _ = assembled
        card = (out_dir / MODEL_CARD_FILENAME).read_text(encoding="utf-8")
        assert "| `f16` | 33 | 17.1 s | 5.15 GiB | ~30 minutes |" in card
        assert "| `f16+dit8` | 81 | not measured | not measured | not run |" in card
        assert "| `f16+dit8-a8-attn8-s16` | 33 | 8.5 s | 4.02 GiB | 952 s (~16 minutes) |" in card
        assert "| `f16+dit8-a8-attn8-s16` | 81 | 34.0 s | 5.46 GiB | not run |" in card
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
            *WAN_CONTAINER_ROLES,
            WAN_TEXT_EMBEDS_ROLE,
            WAN_TOKENIZER_ROLE,
        }
