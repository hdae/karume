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
    BIREFNET_HR_CHECKPOINT,
    BIREFNET_IMAGE_MEAN,
    BIREFNET_IMAGE_STD,
    BIREFNET_LUCIDA_CHECKPOINT,
    BIREFNET_MODELS,
    BIREFNET_OUTPUT_PATHS,
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
# 偽資産は実物と同じ解像度（モデル名 = 解像度なので、SigLIP2 のように小さな非正方へは
# 寄せられない）。代わりに「モデル名と寸法が食い違う」「非正方」「受理集合外のモデル名」の
# 3 つを**落ちる側**のケースとして持つ。

#: 2 モデル目（`--model 2048`）。1 リポ 2 モデルの器（ADR 0092 決定 1 / 8）で既定と同居する。
_BIREFNET_SECOND_MODEL = "2048"

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


#: 既定モデルの一辺（= 既定のモデル名そのもの）。偽資産の shape はここから組む。
_DEFAULT_SIDE = int(BIREFNET_DEFAULT_MODEL)


def _birefnet_graph(
    *,
    shape: Sequence[Any] = (1, 3, _DEFAULT_SIDE, _DEFAULT_SIDE),
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


def _birefnet_container(
    dtype: str, graph: str | None, storage: str | None, side: int = _DEFAULT_SIDE
) -> bytes | list[bytes]:
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
            inputs=(("pixel_values", (1, 3, side, side)),),
            outputs=([1, 1, side, side],),
        )
    return _fake_safetensors(
        dtype, b"birefnet-matte-weights", {IR_METADATA_KEY: graph or _birefnet_graph()}
    )


def _build_birefnet_sources(
    root: Path,
    *,
    checkpoint: str = BIREFNET_HR_CHECKPOINT,
    model: str = BIREFNET_DEFAULT_MODEL,
    side: int | None = None,
    graph: str | None = None,
    dtype: str = "F32",
    storage: str | None = None,
) -> BirefnetSources:
    """系列を偽資産で再現する（配布しない `io.*` の混入込み）。

    並びは `_shared.paths` の実レイアウト（`outputs/series/`）に揃える — CLI 経路のテストが
    root を差し替えるだけで同じ木を指せる形。系列は checkpoint × 解像度で 1 本ずつなので、
    偽資産の寸法は既定で**モデル名から**引く。`side` はその対応を**故意にずらす**ための席
    （系列を 1 本掴み違えた形をそのまま再現する）。
    """
    sources = BirefnetSources(
        series=root / "outputs" / "series" / birefnet_series_name(checkpoint, model)
    )
    write_component(
        sources.series / "model.safetensors",
        _birefnet_container(dtype, graph, storage, side=side if side is not None else int(model)),
    )
    # 配布に入ってはいけない E2E フィクスチャ（系列には実際にこれが並んでいる）。
    _write(sources.series / "io.ramp.safetensors", b"io-fixture")
    return sources


#: checkpoint → その配布リポを組む pipeline の CLI 名（1 リポ 1 Pipeline — ADR 0092 決定 1）。
_PIPELINE_NAMES = {BIREFNET_HR_CHECKPOINT: "birefnet", BIREFNET_LUCIDA_CHECKPOINT: "lucida"}


@pytest.fixture
def birefnet_assembled(tmp_path: Path) -> tuple[Path, dict]:
    sources = _build_birefnet_sources(tmp_path)
    out_dir = tmp_path / "models" / birefnet_repo_name(BIREFNET_HR_CHECKPOINT)
    manifest = assemble_family(
        [birefnet_plan(sources, BIREFNET_HR_CHECKPOINT)],
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
        out_dir = tmp_path / "models" / birefnet_repo_name(BIREFNET_HR_CHECKPOINT)
        plans = [birefnet_plan(sources, BIREFNET_HR_CHECKPOINT)]
        first = assemble_family(plans, out_dir, BIREFNET_DEFAULT_MODEL)
        assert first == assemble_family(plans, out_dir, BIREFNET_DEFAULT_MODEL)
        assert verify_dist(out_dir)

    def test_it_refuses_a_compressed_asset_in_the_f32_seat(self, tmp_path: Path) -> None:
        """格納形は系列ディレクトリ名でなくヘッダが正（`--dtype` 付け忘れの逆向き）。"""
        sources = _build_birefnet_sources(tmp_path, dtype="F16")
        with pytest.raises(DistError, match="F32 が無い"):
            birefnet_plan(sources, BIREFNET_HR_CHECKPOINT)

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
            birefnet_plan(sources, BIREFNET_HR_CHECKPOINT)

    def test_the_plain_f32_series_passes_both_storage_gates(self, tmp_path: Path) -> None:
        """正常対 — 素の f32 系列は要求検査も禁止表も通る（禁止表が恒真に倒れていない）。"""
        sources = _build_birefnet_sources(tmp_path, storage="f32")

        assert birefnet_plan(sources, BIREFNET_HR_CHECKPOINT).name == BIREFNET_DEFAULT_MODEL


class TestBirefnetPipelineConfig:
    """`pipelineConfig` はロード側（`src/birefnet/config.ts`）のスキーマと欄名まで一致する。"""

    def test_it_declares_exactly_the_fields_the_loader_accepts(self, birefnet_assembled) -> None:
        _, manifest = birefnet_assembled
        assert tuple(_birefnet_model(manifest)["pipelineConfig"]) == _BIREFNET_CONFIG_KEYS

    def test_it_derives_the_resize_target_from_the_exported_graph(self, birefnet_assembled) -> None:
        """resize 先の出どころは焼かれたグラフ 1 つきり（上流に前処理 config が無い）。"""
        _, manifest = birefnet_assembled
        config = _birefnet_model(manifest)["pipelineConfig"]
        assert config["imageWidth"] == _DEFAULT_SIDE
        assert config["imageHeight"] == _DEFAULT_SIDE

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
        """モデル名は利用者が `model: "1024"` と綴る値そのもの — 中身とずれたら配れない。"""
        sources = _build_birefnet_sources(tmp_path, graph=_birefnet_graph(shape=(1, 3, 512, 512)))
        with pytest.raises(DistError, match=rf"モデル '{BIREFNET_DEFAULT_MODEL}' が名乗る"):
            birefnet_plan(sources, BIREFNET_HR_CHECKPOINT)

    def test_it_refuses_the_other_models_graph_in_this_models_seat(self, tmp_path: Path) -> None:
        """同居する 2 モデルの取り違え — 系列を 1 本掴み違えるだけで起こる、この器の固有形。

        `2048` の席へ `1024` のグラフが入ると、`pipelineConfig` は 1024² で整合したままなので
        実行も `verify_dist` も通り、**食い違うのはモデル名だけ**になる。
        """
        sources = _build_birefnet_sources(
            tmp_path, model=_BIREFNET_SECOND_MODEL, side=_DEFAULT_SIDE
        )
        with pytest.raises(DistError, match=rf"モデル '{_BIREFNET_SECOND_MODEL}' が名乗る"):
            birefnet_plan(sources, BIREFNET_HR_CHECKPOINT, _BIREFNET_SECOND_MODEL)

    def test_it_refuses_a_model_name_outside_the_published_set(self, tmp_path: Path) -> None:
        """受理集合は {@link BIREFNET_MODELS} 1 つきり — 系列を探す前に落とす。"""
        with pytest.raises(DistError, match="は配らない"):
            birefnet_sources(tmp_path, BIREFNET_HR_CHECKPOINT, "512")

    def test_it_refuses_a_graph_whose_input_is_not_square(self, tmp_path: Path) -> None:
        sources = _build_birefnet_sources(
            tmp_path, graph=_birefnet_graph(shape=(1, 3, _DEFAULT_SIDE, 512))
        )
        with pytest.raises(DistError, match=rf"モデル '{BIREFNET_DEFAULT_MODEL}' が名乗る"):
            birefnet_plan(sources, BIREFNET_HR_CHECKPOINT)

    def test_it_refuses_a_graph_with_another_channel_count(self, tmp_path: Path) -> None:
        sources = _build_birefnet_sources(
            tmp_path,
            graph=_birefnet_graph(shape=(1, 4, _DEFAULT_SIDE, _DEFAULT_SIDE)),
        )
        with pytest.raises(DistError, match="batch もチャネル数も静的"):
            birefnet_plan(sources, BIREFNET_HR_CHECKPOINT)

    def test_it_refuses_a_graph_with_a_second_output(self, tmp_path: Path) -> None:
        """multi-scale supervision 込みの export は、位置で引く後段が別の値を α として読む。"""
        sources = _build_birefnet_sources(tmp_path, graph=_birefnet_graph(outputs=2))
        with pytest.raises(DistError, match="マット 1 本"):
            birefnet_plan(sources, BIREFNET_HR_CHECKPOINT)

    def test_it_refuses_a_matte_that_is_not_one_channel(self, tmp_path: Path) -> None:
        """要素数だけ見る実装なら通ってしまう形（`[1, 3, S, S]` = 中間予測 3 枚）。"""
        sources = _build_birefnet_sources(
            tmp_path,
            graph=_birefnet_graph(out_shape=[1, 3, _DEFAULT_SIDE, _DEFAULT_SIDE]),
        )
        with pytest.raises(DistError, match="入力と同じ寸法の 1 チャネル"):
            birefnet_plan(sources, BIREFNET_HR_CHECKPOINT)

    def test_it_refuses_a_graph_with_a_symbolic_axis(self, tmp_path: Path) -> None:
        """解像度も窓マスクも定数として焼かれているので、動かす軸は 1 本も無い。"""
        sources = _build_birefnet_sources(tmp_path, graph=_birefnet_graph(symbols=("T",)))
        with pytest.raises(DistError, match="記号次元"):
            birefnet_plan(sources, BIREFNET_HR_CHECKPOINT)

    def test_it_refuses_a_renamed_input(self, tmp_path: Path) -> None:
        """実行側は名前で束ねるので、綴りが変われば束ねられない。"""
        sources = _build_birefnet_sources(tmp_path, graph=_birefnet_graph(name="pixels"))
        with pytest.raises(DistError, match="グラフ入力"):
            birefnet_plan(sources, BIREFNET_HR_CHECKPOINT)


class TestBirefnetModelCard:
    def _run(self, tmp_path: Path, checkpoint: str, *models: str) -> Path:
        """偽資産だけで CLI を 1 周回す（`--model` は解像度・先頭が `defaultModel`）。"""
        names = models or (BIREFNET_DEFAULT_MODEL,)
        series_dir = tmp_path / "outputs" / "series"
        for model in names:
            _build_birefnet_sources(tmp_path, checkpoint=checkpoint, model=model)
        out_dir = tmp_path / "dist"
        argv = ["--pipeline", _PIPELINE_NAMES[checkpoint]]
        for model in names:
            argv += ["--model", model]
        main([*argv, "--series", str(series_dir), "--out", str(out_dir)])
        return out_dir

    def test_it_describes_the_background_removal_distribution(self, tmp_path: Path) -> None:
        out_dir = self._run(tmp_path, BIREFNET_HR_CHECKPOINT)
        card = (out_dir / MODEL_CARD_FILENAME).read_text(encoding="utf-8")
        assert "pipeline_tag: image-segmentation" in card
        assert "license: mit" in card
        # 格納形を変えない配布形は 4 値のどれでもない（Hub の推論に任せる）。
        assert "base_model_relation" not in card
        # 合成をこちらでしない事実は、利用者が最初に確かめたい制約そのもの。
        assert "**Compositing is yours.**" in card
        assert f"resizes to {_DEFAULT_SIDE} × {_DEFAULT_SIDE}" in card
        assert "one alpha byte per pixel" in card
        # 単一ファイル配布形は廃止済み（ADR 0081 決定 3）— 器の説明が現物と食い違わない。
        assert "a single safetensors file" not in card
        assert "graph shard" in card
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
        card = (self._run(tmp_path, BIREFNET_HR_CHECKPOINT) / MODEL_CARD_FILENAME).read_text(
            encoding="utf-8"
        )
        assert (
            f'  // model: "{BIREFNET_DEFAULT_MODEL}",'
            f" // default — available: {BIREFNET_DEFAULT_MODEL}" in card
        )
        assert '  // quant: "f32", // default — available: f32' in card
        assert "the only one this repository ships" not in card

    def test_the_attribution_follows_the_checkpoint_of_the_repository(self, tmp_path: Path) -> None:
        """帰属は pipeline 席の checkpoint から一意に決まる（選ばせる軸にしない）。"""
        card = (self._run(tmp_path, BIREFNET_LUCIDA_CHECKPOINT) / MODEL_CARD_FILENAME).read_text(
            encoding="utf-8"
        )
        assert f"base_model: {BIREFNET_UPSTREAM['lucida']}" in card
        # fine-tune の元と学習データのライセンスは lucida 側だけの事実。
        assert BIREFNET_UPSTREAM["hr"] in card
        assert "ToonOut" in card
        # 前処理を焼き込んだ変種を配らないことと、その理由がカードに残る。
        assert "lucida-m35-comfy.safetensors" in card

    def test_the_default_model_card_does_not_carry_another_models_attribution(
        self, tmp_path: Path
    ) -> None:
        card = (self._run(tmp_path, BIREFNET_HR_CHECKPOINT) / MODEL_CARD_FILENAME).read_text(
            encoding="utf-8"
        )
        assert f"base_model: {BIREFNET_UPSTREAM['hr']}" in card
        assert "ToonOut" not in card


class TestBirefnetResolutionFamily:
    """1 リポ 2 モデル（ADR 0092 決定 1 / 8）— 1024² と 2048² が同じリポに同居する。"""

    def _run(self, tmp_path: Path) -> Path:
        for model in BIREFNET_MODELS:
            _build_birefnet_sources(tmp_path, model=model)
        out_dir = tmp_path / "models" / birefnet_repo_name(BIREFNET_HR_CHECKPOINT)
        main(
            [
                "--pipeline",
                "birefnet",
                "--model",
                BIREFNET_DEFAULT_MODEL,
                "--model",
                _BIREFNET_SECOND_MODEL,
                "--series",
                str(tmp_path / "outputs" / "series"),
                "--out",
                str(out_dir),
            ]
        )
        return out_dir

    def test_both_resolutions_live_in_one_repository_with_1024_as_the_default(
        self, tmp_path: Path
    ) -> None:
        """既定は**先頭の `--model`**（`karume.dist.main`）— 綴りの順序が既定を決める。"""
        out_dir = self._run(tmp_path)
        manifest = json.loads((out_dir / MANIFEST_FILENAME).read_text(encoding="utf-8"))
        assert sorted(manifest["models"]) == sorted(BIREFNET_MODELS)
        assert manifest["defaultModel"] == BIREFNET_DEFAULT_MODEL
        assert out_dir.name == "karume-birefnet-hr"

    def test_each_model_declares_its_own_resize_target(self, tmp_path: Path) -> None:
        """同居しても寸法はモデルごと — 1 つの宣言を共有していれば片方が別の数になる。"""
        out_dir = self._run(tmp_path)
        manifest = json.loads((out_dir / MANIFEST_FILENAME).read_text(encoding="utf-8"))
        for model in BIREFNET_MODELS:
            config = manifest["models"][model]["pipelineConfig"]
            assert (config["imageWidth"], config["imageHeight"]) == (int(model), int(model))

    def test_the_layout_keeps_each_models_graph_in_its_own_subtree(self, tmp_path: Path) -> None:
        """グラフ shard は寸法の宣言そのものなので、**共有席へ畳まれてはいけない**。"""
        placed = sorted(verify_dist(self._run(tmp_path)))
        graph_shard = _placed_paths()[0]
        for model in BIREFNET_MODELS:
            assert f"{model}/{graph_shard}" in placed
        assert {path.split("/", 1)[0] for path in placed} <= {*BIREFNET_MODELS, "shared"}

    def test_the_card_enumerates_both_models_and_their_own_resources(self, tmp_path: Path) -> None:
        """Usage の列挙も資源の実測も manifest のモデル 2 つに追従する。"""
        card = (self._run(tmp_path) / MODEL_CARD_FILENAME).read_text(encoding="utf-8")
        listing = " / ".join(sorted(BIREFNET_MODELS))
        assert f'  // model: "{BIREFNET_DEFAULT_MODEL}", // default — available: {listing}' in card
        # 資源はモデルごとの実測（1 つを両方の節へ写していれば片方が嘘になる）。
        assert "about 1.7 GiB" in card
        assert "about 4.1 GiB" in card
        # 上流は 1 本（同居しているのは同じ checkpoint の解像度違い）。
        assert card.count(f"base_model: {BIREFNET_UPSTREAM['hr']}") == 1
        assert BIREFNET_UPSTREAM["lucida"] not in card


class TestBirefnetCli:
    def test_the_default_output_directory_is_named_per_repository(self) -> None:
        """リポ名は導出しない（`karume-lucida` はモデル名からは決まらない綴り）。"""
        assert default_out_dir(PIPELINE, [BIREFNET_DEFAULT_MODEL]).name == "karume-birefnet-hr"
        assert default_out_dir(LUCIDA_PIPELINE, [BIREFNET_DEFAULT_MODEL]).name == "karume-lucida"

    def test_one_attribution_profile_needs_no_choice(self) -> None:
        profiles = PIPELINE.card_profiles
        assert len(profiles) == 1
        assert resolve_card_renderer(PIPELINE, None) is next(iter(profiles.values()))

    @pytest.mark.parametrize("model", BIREFNET_MODELS)
    def test_each_pipeline_looks_only_at_its_own_checkpoints_series(
        self, tmp_path: Path, model: str
    ) -> None:
        """checkpoint は席が固定で持つ — 席を跨いだ取り違えは綴りとして作れない。

        `LICENSE.md` の著作権行は Pipeline に固定で載る 1 組なので、HR の席が Lucida の系列を
        掴むと「自分のものでない著作権を名乗る配布」になる。どちらも散文としては妥当なまま
        `verify_dist` を通るため、**系列 path が席の checkpoint からしか決まらない**ことが
        唯一の防波堤になる（`--model` は解像度で、リポの中身を選べない）。
        """
        for pipeline, checkpoint in (
            (PIPELINE, BIREFNET_HR_CHECKPOINT),
            (LUCIDA_PIPELINE, BIREFNET_LUCIDA_CHECKPOINT),
        ):
            with pytest.raises(DistError) as raised:
                pipeline.plan(tmp_path, model)
            assert birefnet_series_name(checkpoint, model) in str(raised.value)

    @pytest.mark.parametrize("pipeline", [PIPELINE, LUCIDA_PIPELINE])
    def test_each_pipeline_refuses_a_model_outside_the_published_set(self, tmp_path, pipeline):
        with pytest.raises(DistError, match="は配らない"):
            pipeline.plan(tmp_path, "512")

    def test_the_series_name_carries_the_upstream_name_and_the_resolution(
        self, tmp_path: Path
    ) -> None:
        """系列名は上流リポ名 + 解像度（`birefnet.export.default_out_dir` と同じ式）。"""
        for checkpoint, repo in BIREFNET_UPSTREAM.items():
            name = repo.split("/", 1)[1].lower().replace("_", "-")
            for model in BIREFNET_MODELS:
                expected = f"{name}-{model}"
                assert birefnet_series_name(checkpoint, model) == expected
                assert birefnet_sources(tmp_path, checkpoint, model).series.name == expected

    def test_it_refuses_a_checkpoint_it_has_no_attribution_for(self, tmp_path: Path) -> None:
        """帰属表に無い checkpoint は「出所を名乗れない」ので、系列を探す前に落とす。"""
        with pytest.raises(DistError, match="知らない"):
            birefnet_sources(tmp_path, "tiny")

    def test_it_assembles_into_the_pipeline_default_directory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import dist

        sources = _build_birefnet_sources(tmp_path, checkpoint=BIREFNET_LUCIDA_CHECKPOINT)
        monkeypatch.setattr(dist, "DIST_ROOT", tmp_path / "models")

        main(["--pipeline", "lucida", "--series", str(sources.series.parent)])

        out_dir = tmp_path / "models" / "karume-lucida"
        expected = _in_subtree(BIREFNET_DEFAULT_MODEL, _placed_paths())
        assert sorted(verify_dist(out_dir)) == sorted(expected)


class TestBirefnetLegalText:
    """配布リポ直下の MIT 原文と改変告知（ADR 0092 決定 7）。"""

    def _legal(self, tmp_path: Path, checkpoint: str) -> tuple[str, str]:
        sources = _build_birefnet_sources(tmp_path, checkpoint=checkpoint)
        out_dir = tmp_path / "dist"
        main(
            [
                "--pipeline",
                _PIPELINE_NAMES[checkpoint],
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

    @pytest.mark.parametrize("checkpoint", [BIREFNET_HR_CHECKPOINT, BIREFNET_LUCIDA_CHECKPOINT])
    def test_it_ships_the_license_text_byte_identical(
        self, tmp_path: Path, checkpoint: str
    ) -> None:
        """MIT §「著作権表示と許諾表示を含めること」— 差し込み口以外は本文テンプレそのもの。

        原本は `_shared/licenses/mit.txt` に {@link BIREFNET_COPYRIGHTS} を差し込んだ結果。
        組み立ての経路のどこかで整形や改行変換が入ると 1 バイト動くが、散文としては妥当な
        ままなので他の門は素通りする。
        """
        license_text, _ = self._legal(tmp_path, checkpoint)
        assert license_text == mit_license(BIREFNET_COPYRIGHTS[checkpoint])

    def test_the_hr_repository_carries_only_the_upstream_copyright(self, tmp_path: Path) -> None:
        license_text, notice = self._legal(tmp_path, BIREFNET_HR_CHECKPOINT)
        assert "MIT License" in license_text
        assert "Copyright (c) 2024 ZhengPeng" in license_text
        # 自分のものでない著作権を名乗らない（fine-tune 側は別リポの事実）。
        assert "egeorcun" not in license_text
        assert (
            "The above copyright notice and this permission notice shall be included in all"
            in license_text
        )
        assert BIREFNET_UPSTREAM[BIREFNET_HR_CHECKPOINT] in notice
        assert "no quantization" in notice.lower()

    def test_the_lucida_repository_carries_both_copyrights(self, tmp_path: Path) -> None:
        """MIT は派生でも上流の著作権表示を落とせない（上流カードの自己申告と同じ向き）。"""
        license_text, notice = self._legal(tmp_path, BIREFNET_LUCIDA_CHECKPOINT)
        assert "Copyright (c) 2026 egeorcun" in license_text
        assert "Copyright (c) 2024 ZhengPeng" in license_text
        assert BIREFNET_UPSTREAM[BIREFNET_LUCIDA_CHECKPOINT] in notice
