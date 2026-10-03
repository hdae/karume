"""Wan2.1 の配布 recipe（`wan.distribution`）とカード（`wan.card`）— 組み立て 1 周ぶんの単体テスト。

組み立てへ届く入力は数 KB の**正当な最小コンテナ**（`ir_fixtures`）と、書き手
（`wan.text_embeds`）と同じ形の合成の埋め込み資産で作る。門に落とされることを見るケースも
同じ器で作り、**宣言だけを実物とずらす**。

実物の系列（`outputs/series/`）がある機では、実物で計画を 1 周組む門も回す（無ければ SKIP）。
"""

from __future__ import annotations

import json
import tomllib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from container_series import placed_paths, write_component
from ir_fixtures import ir_container
from safetensors.numpy import save
from upstream_fixture import OTHER_REVISION, stamp_fixture_provenance

from _shared.licenses import APACHE_LICENSE_2_0_PATH
from _shared.paths import REPO_ROOT, SERIES_ROOT
from karume.container import CODEC_LEDGER, AssetInput, Provenance
from karume.dist import (
    MANIFEST_FILENAME,
    MODEL_CARD_FILENAME,
    NOTICE_FILENAME,
    DistError,
    assemble_family,
    resolve_card_renderer,
    verify_dist,
)
from wan.card import (
    WAN_ACCEPTED_SIZES,
    WAN_FRAMES,
    WAN_RESOURCE_QUANT,
    WAN_SUPPORTED_PIPELINE,
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


def _default_container(role: str) -> list[bytes]:
    """配置の役割ごとの正常形（transformer は格納ラベルごと — f16 系列 / i8 系列）。"""
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
) -> WanSources:
    """系列 3 本を偽資産で再現する（配布しない golden の混入込み）。`containers` の鍵は配置の
    役割（`transformer_f16` / `transformer_i8` / VAE の 2 本）。"""
    sources = wan_sources(root / "outputs" / "series")
    placements = wan_placements(sources)
    for role in WAN_CONTAINER_ROLES:
        write_component(placements[role], containers.get(role) or _default_container(role))
    # 配布に入ってはいけない golden（系列には実際にこれらが並んでいる）。
    for series in (sources.series, sources.i8_series):
        (series / WAN_TRANSFORMER_ROLE / "io.band-s00192-t0999.safetensors").write_bytes(b"io")
    (sources.series / "pipeline_steps.band-boxing-cats.safetensors").write_bytes(b"steps")
    sources.text_embeds.parent.mkdir(parents=True, exist_ok=True)
    sources.text_embeds.write_bytes(_text_embeds() if embeds is None else embeds)
    return sources


@pytest.fixture
def assembled(tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    sources = _build_sources(tmp_path)
    out_dir = tmp_path / "models" / WAN_REPO_NAME
    manifest = assemble_family(
        [wan_plan(sources)],
        out_dir,
        DEFAULT_MODEL,
        render_card=lambda manifest, host_assets: PIPELINE.card_profiles["wan"](
            manifest, repo=f"hdae/{WAN_REPO_NAME}", host_assets=host_assets
        ),
        root_files=PIPELINE.root_files,
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
    def test_it_places_three_graphs_and_the_embedding_asset_under_the_model_subtree(
        self, assembled
    ) -> None:
        """transformer は格納ラベルごとに 2 本（`model.f16.krm` / `model.i8.krm`）・VAE は
        1 本ずつ。"""
        out_dir, _ = assembled
        expected = [
            f"{DEFAULT_MODEL}/{rel}"
            for rel in placed_paths(
                WAN_OUTPUT_PATHS,
                WAN_WEIGHTS,
                # transformer の容器は資産 `rope_base` の専用 part が 1 本増える。
                {WAN_TRANSFORMER_F16_ROLE: 4, WAN_TRANSFORMER_I8_ROLE: 4},
            )
        ]
        assert _present(out_dir) == sorted(
            [*expected, MANIFEST_FILENAME, MODEL_CARD_FILENAME, "LICENSE.md", NOTICE_FILENAME]
        )
        assert WAN_OUTPUT_PATHS[WAN_TRANSFORMER_F16_ROLE] == "transformer/model.f16.krm"
        assert WAN_OUTPUT_PATHS[WAN_TRANSFORMER_I8_ROLE] == "transformer/model.i8.krm"

    def test_it_never_carries_the_series_goldens(self, assembled) -> None:
        out_dir, _ = assembled
        assert list(out_dir.rglob("io.*")) == []
        assert list(out_dir.rglob("pipeline_steps.*")) == []

    def test_the_manifest_declares_the_three_seats_and_the_model_asset(self, assembled) -> None:
        """quant 席は `f16`（既定）・参照席 `f16+dit8`・実用席 `f16+dit8-a8-attn8-s16` の 3 つ
        （ADR 0120 決定 1 / 6）。資産はモデル単位の `text_embeds`（quant 非依存 — 決定 4）。"""
        _, manifest = assembled
        model = _model(manifest)
        assert manifest["defaultModel"] == DEFAULT_MODEL
        assert model["pipeline"] == WAN_PIPELINE
        assert list(model["weights"]) == list(WAN_GRAPH_ROLES)
        assert list(model["weights"][WAN_TRANSFORMER_ROLE]) == ["f16", "i8"]
        assert list(model["assets"]) == [WAN_TEXT_EMBEDS_ROLE]
        assert model["assets"][WAN_TEXT_EMBEDS_ROLE]["path"] == (
            f"{DEFAULT_MODEL}/{WAN_TEXT_EMBEDS_ROLE}/{WAN_TEXT_EMBEDS_FILE}"
        )
        assert list(model["quants"]) == ["f16", "f16+dit8", "f16+dit8-a8-attn8-s16"]
        assert model["defaultQuant"] == "f16"
        vae = {WAN_VAE_FIRST_ROLE: "f16", WAN_VAE_NEXT_ROLE: "f16"}
        quants = model["quants"]
        assert quants["f16"]["weights"] == {WAN_TRANSFORMER_ROLE: "f16", **vae}
        assert quants["f16+dit8"]["weights"] == {WAN_TRANSFORMER_ROLE: "i8", **vae}
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
        assert "text encoder is not distributed" in notice
        # i8 の席の重みは配布形の改変なので告知に載る（文面は配布形と対応 MUST）。
        prose = " ".join(notice.split())
        assert "**int8 transformer**" in prose
        assert "keep the source float32 values" in prose


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

    def test_the_forbidden_table_names_every_other_compressed_layout(self) -> None:
        """禁止表は役割ごとに、codec 台帳の layout から f32 / i32 と要求する格納を除いた**全部**を
        持つ。"""
        compressed = {entry.layout for entry in CODEC_LEDGER.values()} - {"f32", "i32"}
        assert set(WAN_STORAGE_FORBIDDEN) == set(WAN_STORAGE_REQUIREMENTS)
        assert set(WAN_STORAGE_REQUIREMENTS) == set(WAN_CONTAINER_ROLES)
        for role, required in WAN_STORAGE_REQUIREMENTS.items():
            assert set(WAN_STORAGE_FORBIDDEN[role]) == compressed - {required}, role


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
        assert "| `f16+dit8-a8-attn8-s16` |" in card
        assert "| `f16` (default) |" in card

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
        assert "`maxStorageBufferBindingSize` (128 MiB)" in card

    def test_it_says_the_other_seats_have_not_been_measured(self, assembled) -> None:
        """実測した席（`f16`）以外は数を推し量らず未計測と名乗る（席の並びは manifest のまま）。"""
        out_dir, manifest = assembled
        card = (out_dir / MODEL_CARD_FILENAME).read_text(encoding="utf-8")
        assert (
            "The other quants (`f16+dit8` / `f16+dit8-a8-attn8-s16`) have not been measured yet."
            in card
        )
        only_measured = json.loads(json.dumps(manifest))
        quants = only_measured["models"][DEFAULT_MODEL]["quants"]
        only_measured["models"][DEFAULT_MODEL]["quants"] = {
            WAN_RESOURCE_QUANT: quants[WAN_RESOURCE_QUANT]
        }
        assert "have not been measured" not in _card(only_measured)

    def test_it_refuses_a_manifest_without_the_measured_quant(self, assembled) -> None:
        """実測した席が無い配布形では資源の数を名乗らない（推し量った数を出さない）。"""
        _, manifest = assembled
        changed = json.loads(json.dumps(manifest))
        quants = changed["models"][DEFAULT_MODEL]["quants"]
        quants["i8"] = quants.pop(WAN_RESOURCE_QUANT)
        changed["models"][DEFAULT_MODEL]["defaultQuant"] = "i8"
        with pytest.raises(ValueError, match=f"quant '{WAN_RESOURCE_QUANT}'"):
            _card(changed)


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
)


@pytest.mark.skipif(
    not _REAL_PRESENT, reason=f"実物の系列が無い: {_REAL.series} / {_REAL.i8_series}"
)
class TestTheRealSeries:
    """実物の系列（`wan.export_dit`〈f16 / i8〉/ `wan.export_vae` / `wan.text_embeds` の出力）で
    計画が組める。

    読むのは容器の 2 文書・束縛表・資産（宣言と RoPE の素表）と埋め込み資産のヘッダだけ（重みの
    payload は読まない）。
    """

    def test_the_plan_passes_every_gate(self) -> None:
        plan = wan_plan(_REAL)
        assert plan.pipeline == WAN_PIPELINE
        assert set(plan.artifacts) == {*WAN_CONTAINER_ROLES, WAN_TEXT_EMBEDS_ROLE}
