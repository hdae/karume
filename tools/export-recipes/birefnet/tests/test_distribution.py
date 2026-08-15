"""BiRefNet 系の配布 recipe（`birefnet.distribution`）— 組み立て 1 周ぶんの単体テスト。

実資産は使わない — dist が safetensors から読むのは**ヘッダの dtype 集合**と IR メタデータ
（`__metadata__`）だけなので、数十バイトの正規ヘッダ付き偽資産で層の振る舞いは全て観測できる。

manifest v2（`karume/2` — ADR 0041）以降、リポ内レイアウトは一律「モデル別サブツリー +
`shared/`」なので、期待 path は全て `<モデル名>/…` を頭に持つ。

core だけで観測できる層（合成計画で足りる規模上限・quant 完全写像・staging/swap の不変条件・
帰属プロファイルの解決規則）は `tools/exporter/tests/test_dist.py` が持つ（ADR 0065 段 3+4 の
分割）。テンプレート単位の門は `birefnet/tests/test_card.py`。
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from birefnet.card import BIREFNET_UPSTREAM
from birefnet.distribution import (
    BIREFNET_DEFAULT_MODEL,
    BIREFNET_IMAGE_MEAN,
    BIREFNET_IMAGE_STD,
    BIREFNET_OUTPUT_PATHS,
    BIREFNET_RESOLUTION,
    BIREFNET_ROLE,
    PIPELINE,
    BirefnetSources,
    birefnet_plan,
    birefnet_repo_name,
    birefnet_series_name,
    birefnet_sources,
)
from dist import default_out_dir, main
from karume.dist import (
    MANIFEST_FILENAME,
    MODEL_CARD_FILENAME,
    DistError,
    assemble_family,
    resolve_card_renderer,
    verify_dist,
)
from karume.ir import IR_METADATA_KEY


def _fake_safetensors(
    dtype: str, payload: bytes, metadata: Mapping[str, str] | None = None
) -> bytes:
    """格納 dtype の門を通る最小の safetensors（8 バイト長 + ヘッダ JSON + データ節）。

    `metadata` を渡すと `__metadata__` 節が付く（IR コンテナを要求する門のため）。
    """
    header: dict[str, Any] = {
        "w": {"dtype": dtype, "shape": [len(payload)], "data_offsets": [0, len(payload)]}
    }
    if metadata is not None:
        header["__metadata__"] = dict(metadata)
    encoded = json.dumps(header).encode("utf-8")
    return len(encoded).to_bytes(8, "little") + encoded + payload


def _write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _in_subtree(model: str, paths: Iterable[str]) -> list[str]:
    """モデルサブツリー内の期待 path（ADR 0041 §9 の一様レイアウト）。"""
    return [f"{model}/{rel}" for rel in paths]


def _present(out_dir: Path) -> list[str]:
    return sorted(str(path.relative_to(out_dir)) for path in out_dir.rglob("*") if path.is_file())


# ---- BiRefNet 系（image-segmentation）----------------------------------------
#
# 偽資産は実物と同じ 1024²（{@link BIREFNET_RESOLUTION} が他の解像度を配布から締め出すので、
# SigLIP2 のように小さな非正方へ寄せられない）。代わりに「寸法が違う」「非正方」「軸が
# 転置された」の 3 つを**落ちる側**のケースとして持つ。

#: `pipelineConfig` の欄名（TS 側 `packages/models/src/birefnet/config.ts` の `ROOT_KEYS` の
#: 写し）。**ロード側は未知キーも欠落も parse 時に落とす**ので、焼く側とロード側の欄名は
#: 完全一致が要る。
_BIREFNET_CONFIG_KEYS = (
    "imageWidth",
    "imageHeight",
    "imageMean",
    "imageStd",
    "interpolation",
)


def _birefnet_graph(
    *,
    shape: Sequence[Any] = (1, 3, BIREFNET_RESOLUTION, BIREFNET_RESOLUTION),
    out_shape: Sequence[Any] | None = None,
    name: str = "pixel_values",
    outputs: int = 1,
    symbols: Sequence[str] = (),
) -> str:
    """門が読む最小の IR メタデータ（入力 1 本・出力の宣言・記号次元）。"""
    names = [f"out_{index}" for index in range(outputs)]
    matte = list(out_shape) if out_shape is not None else [1, 1, shape[2], shape[3]]
    return json.dumps(
        {
            "inputs": [{"name": name, "shape": list(shape)}],
            "outputs": names,
            "values": {output: {"dtype": "f32", "shape": matte} for output in names},
            "symbols": list(symbols),
        }
    )


def _build_birefnet_sources(
    root: Path,
    *,
    model: str = BIREFNET_DEFAULT_MODEL,
    graph: str | None = None,
    dtype: str = "F32",
) -> BirefnetSources:
    """系列を偽資産で再現する（配布しない `io.*` の混入込み）。

    並びは `_shared.paths` の実レイアウト（`outputs/series/`）に揃える — CLI 経路のテストが
    root を差し替えるだけで同じ木を指せる形。
    """
    sources = BirefnetSources(series=root / "outputs" / "series" / birefnet_series_name(model))
    _write(
        sources.series / "model.safetensors",
        _fake_safetensors(
            dtype,
            b"birefnet-matte-weights",
            {IR_METADATA_KEY: graph if graph is not None else _birefnet_graph()},
        ),
    )
    # 配布に入ってはいけない E2E フィクスチャ（系列には実際にこれが並んでいる）。
    _write(sources.series / "io.ramp.safetensors", b"io-fixture")
    return sources


@pytest.fixture
def birefnet_assembled(tmp_path: Path) -> tuple[Path, dict]:
    sources = _build_birefnet_sources(tmp_path)
    out_dir = tmp_path / "models" / birefnet_repo_name(BIREFNET_DEFAULT_MODEL)
    manifest = assemble_family(
        [birefnet_plan(sources, BIREFNET_DEFAULT_MODEL)], out_dir, BIREFNET_DEFAULT_MODEL
    )
    return out_dir, manifest


def _birefnet_model(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    return manifest["models"][BIREFNET_DEFAULT_MODEL]


class TestBirefnetLayout:
    def test_it_places_the_single_graph_under_the_model_subtree(self, birefnet_assembled) -> None:
        out_dir, _ = birefnet_assembled
        expected = _in_subtree(BIREFNET_DEFAULT_MODEL, BIREFNET_OUTPUT_PATHS.values())
        assert _present(out_dir) == sorted([*expected, MANIFEST_FILENAME])

    def test_it_never_carries_the_io_fixtures(self, birefnet_assembled) -> None:
        out_dir, _ = birefnet_assembled
        assert list(out_dir.rglob("io.*")) == []

    def test_it_declares_one_graph_and_no_assets(self, birefnet_assembled) -> None:
        """実行に要るのはグラフ 1 本だけ（tokenizer も表も無い = `assets` は空）。"""
        _, manifest = birefnet_assembled
        model = _birefnet_model(manifest)
        assert model["pipeline"] == "birefnet/1"
        assert list(model["weights"]) == [BIREFNET_ROLE]
        assert model["assets"] == {}
        assert list(model["quants"]) == ["f32"]
        assert model["defaultQuant"] == "f32"
        assert model["quants"]["f32"]["weights"] == {BIREFNET_ROLE: "f32"}
        assert model["quants"]["f32"]["session"] == {}

    def test_it_reassembles_over_a_previous_run(self, tmp_path: Path) -> None:
        sources = _build_birefnet_sources(tmp_path)
        out_dir = tmp_path / "models" / birefnet_repo_name(BIREFNET_DEFAULT_MODEL)
        first = assemble_family([birefnet_plan(sources)], out_dir, BIREFNET_DEFAULT_MODEL)
        assert first == assemble_family([birefnet_plan(sources)], out_dir, BIREFNET_DEFAULT_MODEL)
        assert verify_dist(out_dir)

    def test_it_refuses_a_compressed_asset_in_the_f32_seat(self, tmp_path: Path) -> None:
        """格納形は系列ディレクトリ名でなくヘッダが正（`--dtype` 付け忘れの逆向き）。"""
        sources = _build_birefnet_sources(tmp_path, dtype="F16")
        with pytest.raises(DistError, match="F32 が無い"):
            birefnet_plan(sources)


class TestBirefnetPipelineConfig:
    """`pipelineConfig` はロード側（`src/birefnet/config.ts`）のスキーマと欄名まで一致する。"""

    def test_it_declares_exactly_the_fields_the_loader_accepts(self, birefnet_assembled) -> None:
        _, manifest = birefnet_assembled
        assert tuple(_birefnet_model(manifest)["pipelineConfig"]) == _BIREFNET_CONFIG_KEYS

    def test_it_derives_the_resize_target_from_the_exported_graph(self, birefnet_assembled) -> None:
        """resize 先の出どころは焼かれたグラフ 1 つきり（上流に前処理 config が無い）。"""
        _, manifest = birefnet_assembled
        config = _birefnet_model(manifest)["pipelineConfig"]
        assert config["imageWidth"] == BIREFNET_RESOLUTION
        assert config["imageHeight"] == BIREFNET_RESOLUTION

    def test_it_declares_the_upstream_normalization_constants(self, birefnet_assembled) -> None:
        """正規化定数は機械可読な出どころが無いので dist が宣言として持つ（節の冒頭）。"""
        _, manifest = birefnet_assembled
        config = _birefnet_model(manifest)["pipelineConfig"]
        assert config["imageMean"] == list(BIREFNET_IMAGE_MEAN)
        assert config["imageStd"] == list(BIREFNET_IMAGE_STD)
        assert config["interpolation"] == "bilinear"


class TestBirefnetGraphGate:
    """組み立て門 — ずれても配布形としては成立してしまう組み合わせを、配置の前に落とす。"""

    def test_it_refuses_a_graph_baked_for_another_resolution(self, tmp_path: Path) -> None:
        """配るのは 1024² だけ（2048² は実行段が未実測 — docs/limitations.md）。"""
        sources = _build_birefnet_sources(tmp_path, graph=_birefnet_graph(shape=(1, 3, 512, 512)))
        with pytest.raises(DistError, match="配るのは"):
            birefnet_plan(sources)

    def test_it_refuses_a_graph_whose_input_is_not_square(self, tmp_path: Path) -> None:
        sources = _build_birefnet_sources(
            tmp_path, graph=_birefnet_graph(shape=(1, 3, BIREFNET_RESOLUTION, 512))
        )
        with pytest.raises(DistError, match="配るのは"):
            birefnet_plan(sources)

    def test_it_refuses_a_graph_with_another_channel_count(self, tmp_path: Path) -> None:
        sources = _build_birefnet_sources(
            tmp_path,
            graph=_birefnet_graph(shape=(1, 4, BIREFNET_RESOLUTION, BIREFNET_RESOLUTION)),
        )
        with pytest.raises(DistError, match="batch もチャネル数も静的"):
            birefnet_plan(sources)

    def test_it_refuses_a_graph_with_a_second_output(self, tmp_path: Path) -> None:
        """multi-scale supervision 込みの export は、位置で引く後段が別の値を α として読む。"""
        sources = _build_birefnet_sources(tmp_path, graph=_birefnet_graph(outputs=2))
        with pytest.raises(DistError, match="マット 1 本"):
            birefnet_plan(sources)

    def test_it_refuses_a_matte_that_is_not_one_channel(self, tmp_path: Path) -> None:
        """要素数だけ見る実装なら通ってしまう形（`[1, 3, S, S]` = 中間予測 3 枚）。"""
        sources = _build_birefnet_sources(
            tmp_path,
            graph=_birefnet_graph(out_shape=[1, 3, BIREFNET_RESOLUTION, BIREFNET_RESOLUTION]),
        )
        with pytest.raises(DistError, match="入力と同じ寸法の 1 チャネル"):
            birefnet_plan(sources)

    def test_it_refuses_a_graph_with_a_symbolic_axis(self, tmp_path: Path) -> None:
        """解像度も窓マスクも定数として焼かれているので、動かす軸は 1 本も無い。"""
        sources = _build_birefnet_sources(tmp_path, graph=_birefnet_graph(symbols=("T",)))
        with pytest.raises(DistError, match="記号次元"):
            birefnet_plan(sources)

    def test_it_refuses_a_renamed_input(self, tmp_path: Path) -> None:
        """実行側は名前で束ねるので、綴りが変われば束ねられない。"""
        sources = _build_birefnet_sources(tmp_path, graph=_birefnet_graph(name="pixels"))
        with pytest.raises(DistError, match="グラフ入力"):
            birefnet_plan(sources)


class TestBirefnetModelCard:
    def _run(self, tmp_path: Path, model: str) -> Path:
        """偽資産だけで CLI を 1 周回す。"""
        sources = _build_birefnet_sources(tmp_path, model=model)
        out_dir = tmp_path / "dist"
        main(
            [
                "--pipeline",
                "birefnet",
                "--model",
                model,
                "--series",
                str(sources.series.parent),
                "--out",
                str(out_dir),
            ]
        )
        return out_dir

    def test_it_describes_the_background_removal_distribution(self, tmp_path: Path) -> None:
        out_dir = self._run(tmp_path, BIREFNET_DEFAULT_MODEL)
        card = (out_dir / MODEL_CARD_FILENAME).read_text(encoding="utf-8")
        assert "pipeline_tag: image-segmentation" in card
        assert "license: mit" in card
        # 格納形を変えない配布形は 4 値のどれでもない（Hub の推論に任せる）。
        assert "base_model_relation" not in card
        # 合成をこちらでしない事実は、利用者が最初に確かめたい制約そのもの。
        assert "**Compositing is yours.**" in card
        assert f"resizes to {BIREFNET_RESOLUTION} × {BIREFNET_RESOLUTION}" in card
        assert "one alpha byte per pixel" in card
        # カードは**検証を通った**配布形から描かれる。
        assert verify_dist(out_dir)

    def test_the_attribution_follows_the_model_name(self, tmp_path: Path) -> None:
        """帰属はモデル名から一意に決まる（選ばせる軸にしない = 取り違えようがない）。"""
        card = (self._run(tmp_path, "lucida") / MODEL_CARD_FILENAME).read_text(encoding="utf-8")
        assert f"base_model: {BIREFNET_UPSTREAM['lucida']}" in card
        # fine-tune の元と学習データのライセンスは lucida 側だけの事実。
        assert BIREFNET_UPSTREAM["hr"] in card
        assert "ToonOut" in card
        # 前処理を焼き込んだ変種を配らないことと、その理由がカードに残る。
        assert "lucida-m35-comfy.safetensors" in card

    def test_the_default_model_card_does_not_carry_another_models_attribution(
        self, tmp_path: Path
    ) -> None:
        card = (self._run(tmp_path, BIREFNET_DEFAULT_MODEL) / MODEL_CARD_FILENAME).read_text(
            encoding="utf-8"
        )
        assert f"base_model: {BIREFNET_UPSTREAM['hr']}" in card
        assert "ToonOut" not in card


class TestBirefnetCli:
    def test_the_default_output_directory_is_named_per_model(self) -> None:
        """リポ名は導出しない（`karume-lucida` はモデル名からは決まらない綴り）。"""
        assert default_out_dir(PIPELINE, [BIREFNET_DEFAULT_MODEL]).name == "karume-birefnet-hr"
        assert default_out_dir(PIPELINE, ["lucida"]).name == "karume-lucida"

    def test_one_attribution_profile_needs_no_choice(self) -> None:
        profiles = PIPELINE.card_profiles
        assert len(profiles) == 1
        assert resolve_card_renderer(PIPELINE, None) is next(iter(profiles.values()))

    def test_the_series_name_carries_the_upstream_name_and_the_resolution(
        self, tmp_path: Path
    ) -> None:
        """系列名は上流リポ名 + 解像度（`birefnet.export.default_out_dir` と同じ式）。"""
        for model, repo in BIREFNET_UPSTREAM.items():
            checkpoint = repo.split("/", 1)[1].lower().replace("_", "-")
            expected = f"{checkpoint}-{BIREFNET_RESOLUTION}"
            assert birefnet_series_name(model) == expected
            assert birefnet_sources(tmp_path, model).series.name == expected

    def test_it_refuses_a_model_it_has_no_attribution_for(self, tmp_path: Path) -> None:
        """帰属表に無いモデル名は「出所を名乗れない」ので、系列を探す前に落とす。"""
        with pytest.raises(DistError, match="知らない"):
            birefnet_sources(tmp_path, "tiny")

    def test_it_assembles_into_the_pipeline_default_directory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import dist

        sources = _build_birefnet_sources(tmp_path, model="lucida")
        monkeypatch.setattr(dist, "DIST_ROOT", tmp_path / "models")

        main(
            ["--pipeline", "birefnet", "--model", "lucida", "--series", str(sources.series.parent)]
        )

        out_dir = tmp_path / "models" / "karume-lucida"
        expected = _in_subtree("lucida", BIREFNET_OUTPUT_PATHS.values())
        assert sorted(verify_dist(out_dir)) == sorted(expected)
