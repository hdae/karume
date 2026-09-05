"""BiRefNet 系の配布 recipe（`birefnet.distribution`）— 組み立て 1 周ぶんの単体テスト。

実資産は使わない。組み立てへ届く入力は数 KB の**正当な最小 IR コンテナ**（`ir_fixtures`）で、
門に落とされることを見るケースだけが従来の偽資産のまま（{@link _birefnet_container}）。

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
from ir_fixtures import ir_container
from shard_series import placed_paths, write_component

from _shared.licenses import mit_license
from birefnet.card import BIREFNET_UPSTREAM
from birefnet.distribution import (
    BIREFNET_COPYRIGHTS,
    BIREFNET_DEFAULT_MODEL,
    BIREFNET_IMAGE_MEAN,
    BIREFNET_IMAGE_STD,
    BIREFNET_LUCIDA_MODEL,
    BIREFNET_OUTPUT_PATHS,
    BIREFNET_RESOLUTION,
    BIREFNET_ROLE,
    BIREFNET_WEIGHTS,
    LUCIDA_PIPELINE,
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


def _placed_paths() -> list[str]:
    """配布形に現れる相対 path（weights の席は shard 連番に展開される — ADR 0081）。"""
    return placed_paths(BIREFNET_OUTPUT_PATHS, BIREFNET_WEIGHTS)


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


def _birefnet_container(dtype: str, graph: str | None, storage: str | None) -> bytes | list[bytes]:
    """系列に置くマット推定グラフの中身。

    組み立てへ届く既定の形は**正当な IR コンポーネント**でなければならない（組み立ては入力を
    IR v1 の全規則で見る — `karume.dist.assert_weight_components_verified`）。正当な側は
    shard 列（`list[bytes]`・先頭がグラフ shard — ADR 0081）で、偽コンテナは代表 path 1 本の
    `bytes` のまま（計画の門で止まるので分割の側まで届かない）。

    軸は 2 本ある:

    - `dtype` は**単一 dtype の偽コンテナ**が名乗るヘッダ dtype。要求検査（「F32 が無い」）を
      見るケースはこちらでしか作れない — 実物どおりの f16 系列は F32 も含むので、要求検査は
      通ってしまう。
    - `storage` は**実物どおりの混成コンポーネント**の格納形（f16 / i8 / i4 は適格外の重みと
      bias が F32 で残り、i4 はさらに I8 が混ざる）。禁止表
      （{@link BIREFNET_STORAGE_FORBIDDEN}）の門はこの形でしか試せない。
    """
    if storage is not None or (graph is None and dtype == "F32"):
        return ir_container(
            mark="birefnet-matte",
            storage=storage if storage is not None else dtype.lower(),
            inputs=(("pixel_values", (1, 3, BIREFNET_RESOLUTION, BIREFNET_RESOLUTION)),),
            outputs=([1, 1, BIREFNET_RESOLUTION, BIREFNET_RESOLUTION],),
        )
    return _fake_safetensors(
        dtype, b"birefnet-matte-weights", {IR_METADATA_KEY: graph or _birefnet_graph()}
    )


def _build_birefnet_sources(
    root: Path,
    *,
    model: str = BIREFNET_DEFAULT_MODEL,
    graph: str | None = None,
    dtype: str = "F32",
    storage: str | None = None,
) -> BirefnetSources:
    """系列を偽資産で再現する（配布しない `io.*` の混入込み）。

    並びは `_shared.paths` の実レイアウト（`outputs/series/`）に揃える — CLI 経路のテストが
    root を差し替えるだけで同じ木を指せる形。
    """
    sources = BirefnetSources(series=root / "outputs" / "series" / birefnet_series_name(model))
    write_component(
        sources.series / "model.safetensors", _birefnet_container(dtype, graph, storage)
    )
    # 配布に入ってはいけない E2E フィクスチャ（系列には実際にこれが並んでいる）。
    _write(sources.series / "io.ramp.safetensors", b"io-fixture")
    return sources


#: モデル名 → その配布リポを組む pipeline の CLI 名（1 リポ 1 Pipeline — ADR 0092 決定 1）。
_PIPELINE_NAMES = {BIREFNET_DEFAULT_MODEL: "birefnet", BIREFNET_LUCIDA_MODEL: "lucida"}


@pytest.fixture
def birefnet_assembled(tmp_path: Path) -> tuple[Path, dict]:
    sources = _build_birefnet_sources(tmp_path)
    out_dir = tmp_path / "models" / birefnet_repo_name(BIREFNET_DEFAULT_MODEL)
    manifest = assemble_family(
        [birefnet_plan(sources, BIREFNET_DEFAULT_MODEL)],
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
        # 法的テキスト 2 本（MIT の著作権 / 許諾表示と改変告知）は manifest が宣言しないメタ席。
        assert _present(out_dir) == sorted(
            [*expected, MANIFEST_FILENAME, "LICENSE.md", "NOTICE.md"]
        )

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

    @pytest.mark.parametrize(("storage", "intruder"), [("f16", "F16"), ("i8", "I8"), ("i4", "I4")])
    def test_it_refuses_a_real_compressed_series_in_the_f32_seat(
        self, tmp_path: Path, storage: str, intruder: str
    ) -> None:
        """実物どおりの圧縮系列は**要求検査を満たす** — 禁止表だけが系列 root の取り違えを見る。

        圧縮系列も適格外の重みと bias を F32 で持つので「F32 を含む」は真になり、上の
        単一 dtype の偽資産と違って要求検査では 1 バイトも落ちない。落ちるべき理由は
        「f32 席に別系列の資産が居る」で、数値の門では原理的に検出できない
        （ADR 0027 / 0029）。
        """
        sources = _build_birefnet_sources(tmp_path, storage=storage)
        with pytest.raises(DistError, match=rf"{BIREFNET_ROLE}: .* {intruder} がある"):
            birefnet_plan(sources)

    def test_the_plain_f32_series_passes_both_storage_gates(self, tmp_path: Path) -> None:
        """正常対 — 素の f32 系列は要求検査も禁止表も通る（禁止表が恒真に倒れていない）。"""
        sources = _build_birefnet_sources(tmp_path, storage="f32")

        assert birefnet_plan(sources).name == BIREFNET_DEFAULT_MODEL


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
        """配るのは 1024² だけ（2048² は実行できるが公開裁定前 — docs/limitations.md）。"""
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
                _PIPELINE_NAMES[model],
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

    def test_the_usage_snippet_enumerates_the_choices_from_the_manifest(
        self, tmp_path: Path
    ) -> None:
        """裁定 2026-08-12 の形: optional はコメント併記で、選べる値は manifest 由来の列挙。

        `segment()` に実行時ノブは無いので、この pipeline の optional は model / quant だけ。
        1 席しか無くても「available:」で綴るのは、席が増えたときに文面が自動で追従するため
        （「the only one this repository ships」のような焼き込みは、増えた瞬間に嘘になる）。
        """
        card = (self._run(tmp_path, BIREFNET_DEFAULT_MODEL) / MODEL_CARD_FILENAME).read_text(
            encoding="utf-8"
        )
        assert (
            f'  // model: "{BIREFNET_DEFAULT_MODEL}",'
            f" // default — available: {BIREFNET_DEFAULT_MODEL}" in card
        )
        assert '  // quant: "f32", // default — available: f32' in card
        assert "the only one this repository ships" not in card

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
        assert default_out_dir(LUCIDA_PIPELINE, [BIREFNET_LUCIDA_MODEL]).name == "karume-lucida"

    def test_one_attribution_profile_needs_no_choice(self) -> None:
        profiles = PIPELINE.card_profiles
        assert len(profiles) == 1
        assert resolve_card_renderer(PIPELINE, None) is next(iter(profiles.values()))

    @pytest.mark.parametrize(
        ("pipeline", "model"),
        [(PIPELINE, BIREFNET_LUCIDA_MODEL), (LUCIDA_PIPELINE, BIREFNET_DEFAULT_MODEL)],
    )
    def test_each_pipeline_refuses_the_other_repositorys_model(
        self, tmp_path: Path, pipeline, model: str
    ) -> None:
        """席を跨いで組むと、リポ直下の著作権表示が中身と食い違ったまま配布形が成立する。

        `LICENSE.md` の著作権行は Pipeline に固定で載る 1 組なので、HR の席で Lucida を
        組むと「fine-tune の著作権を名乗らない配布」、逆向きだと「自分のものでない著作権を
        名乗る配布」になる。どちらも散文としては妥当なまま `verify_dist` を通る。
        """
        with pytest.raises(DistError, match="この pipeline のリポに入らない"):
            pipeline.plan(tmp_path, model)

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

        sources = _build_birefnet_sources(tmp_path, model=BIREFNET_LUCIDA_MODEL)
        monkeypatch.setattr(dist, "DIST_ROOT", tmp_path / "models")

        main(["--pipeline", "lucida", "--series", str(sources.series.parent)])

        out_dir = tmp_path / "models" / "karume-lucida"
        expected = _in_subtree(BIREFNET_LUCIDA_MODEL, _placed_paths())
        assert sorted(verify_dist(out_dir)) == sorted(expected)


class TestBirefnetLegalText:
    """配布リポ直下の MIT 原文と改変告知（ADR 0092 決定 7）。"""

    def _legal(self, tmp_path: Path, model: str) -> tuple[str, str]:
        sources = _build_birefnet_sources(tmp_path, model=model)
        out_dir = tmp_path / "dist"
        main(
            [
                "--pipeline",
                _PIPELINE_NAMES[model],
                "--series",
                str(sources.series.parent),
                "--out",
                str(out_dir),
            ]
        )
        return (
            # LICENSE はバイト列を decode して返す（read_text の改行変換で CRLF が畳まれると
            # バイト同一の門が素通りする — 他家族の read_bytes 比較と同じ網の粗さにする）。
            (out_dir / "LICENSE.md").read_bytes().decode("utf-8"),
            (out_dir / "NOTICE.md").read_text(encoding="utf-8"),
        )

    @pytest.mark.parametrize("model", [BIREFNET_DEFAULT_MODEL, BIREFNET_LUCIDA_MODEL])
    def test_it_ships_the_license_text_byte_identical(self, tmp_path: Path, model: str) -> None:
        """MIT §「著作権表示と許諾表示を含めること」— 差し込み口以外は本文テンプレそのもの。

        原本は `_shared/licenses/mit.txt` に {@link BIREFNET_COPYRIGHTS} を差し込んだ結果。
        組み立ての経路のどこかで整形や改行変換が入ると 1 バイト動くが、散文としては妥当な
        ままなので他の門は素通りする。
        """
        license_text, _ = self._legal(tmp_path, model)
        assert license_text == mit_license(BIREFNET_COPYRIGHTS[model])

    def test_the_hr_repository_carries_only_the_upstream_copyright(self, tmp_path: Path) -> None:
        license_text, notice = self._legal(tmp_path, BIREFNET_DEFAULT_MODEL)
        assert "MIT License" in license_text
        assert "Copyright (c) 2024 ZhengPeng" in license_text
        # 自分のものでない著作権を名乗らない（fine-tune 側は別リポの事実）。
        assert "egeorcun" not in license_text
        assert (
            "The above copyright notice and this permission notice shall be included in all"
            in license_text
        )
        assert BIREFNET_UPSTREAM[BIREFNET_DEFAULT_MODEL] in notice
        assert "no quantization" in notice.lower()

    def test_the_lucida_repository_carries_both_copyrights(self, tmp_path: Path) -> None:
        """MIT は派生でも上流の著作権表示を落とせない（上流カードの自己申告と同じ向き）。"""
        license_text, notice = self._legal(tmp_path, BIREFNET_LUCIDA_MODEL)
        assert "Copyright (c) 2026 egeorcun" in license_text
        assert "Copyright (c) 2024 ZhengPeng" in license_text
        assert BIREFNET_UPSTREAM[BIREFNET_LUCIDA_MODEL] in notice
