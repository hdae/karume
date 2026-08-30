"""SigLIP2 の配布 recipe（`siglip2.distribution`）— 組み立て 1 周ぶんの単体テスト。

実資産は使わない。組み立てへ届く入力は数 KB の**正当な最小 IR コンテナ**（`ir_fixtures`）で、
門に落とされることを見るケースだけが従来の偽資産のまま（{@link _siglip2_container}）。
前処理定数の出どころ（`preprocessor_config.json`）も合成 JSON で足りる。

manifest v2（`karume/2` — ADR 0041）以降、リポ内レイアウトは一律「モデル別サブツリー +
`shared/`」なので、期待 path は全て `<モデル名>/…` を頭に持つ。

core だけで観測できる層（合成計画で足りる規模上限・quant 完全写像・staging/swap の不変条件・
帰属プロファイルの解決規則）は `tools/exporter/tests/test_dist.py` が持つ（ADR 0065 段 3+4 の
分割）。テンプレート単位の門は `siglip2/tests/test_card.py`。
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest
from ir_fixtures import ir_container
from shard_series import placed_paths, write_component

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
from siglip2.card import SIGLIP2_UPSTREAM
from siglip2.distribution import (
    PIPELINE,
    SIGLIP2_DEFAULT_MODEL,
    SIGLIP2_OUTPUT_PATHS,
    SIGLIP2_ROLE,
    SIGLIP2_WEIGHTS,
    Siglip2Sources,
    siglip2_checkpoint,
    siglip2_plan,
    siglip2_repo_name,
    siglip2_sources,
)


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
    return placed_paths(SIGLIP2_OUTPUT_PATHS, SIGLIP2_WEIGHTS)


def _present(out_dir: Path) -> list[str]:
    return sorted(str(path.relative_to(out_dir)) for path in out_dir.rglob("*") if path.is_file())


# ---- SigLIP2（image-feature-extraction）--------------------------------------
#
# 偽資産は**実物と違う数**にする（96×64 の非正方・mean/std は 0.5 でない・hidden 7）— 前処理
# 定数や寸法を焼き込んでいれば落ちる。非正方なのは、`imageWidth` / `imageHeight` の取り違えが
# 正方形では原理的に検出できないため。

#: `pipelineConfig` の欄名（TS 側 `packages/models/src/siglip2/config.ts` の `ROOT_KEYS` の写し）。
#: **ロード側は未知キーも欠落も parse 時に落とす**ので、焼く側とロード側の欄名は完全一致が要る。
_SIGLIP2_CONFIG_KEYS = (
    "imageWidth",
    "imageHeight",
    "imageMean",
    "imageStd",
    "hiddenDim",
    "interpolation",
)

_SIGLIP2_HEIGHT = 96
_SIGLIP2_WIDTH = 64
_SIGLIP2_HIDDEN = 7

#: 偽の `preprocessor_config.json`（実物と同じ欄・違う値）。
_SIGLIP2_PREPROCESSOR: Mapping[str, Any] = {
    "do_normalize": True,
    "do_rescale": True,
    "do_resize": True,
    "image_mean": [0.1, 0.2, 0.3],
    "image_std": [0.4, 0.5, 0.6],
    "image_processor_type": "SiglipImageProcessor",
    "resample": 2,
    "rescale_factor": 1.0 / 255.0,
    "size": {"height": _SIGLIP2_HEIGHT, "width": _SIGLIP2_WIDTH},
}


def _siglip2_graph(
    *,
    shape: Sequence[Any] = (1, 3, _SIGLIP2_HEIGHT, _SIGLIP2_WIDTH),
    hidden: int = _SIGLIP2_HIDDEN,
    name: str = "pixel_values",
    outputs: int = 1,
    symbols: Sequence[str] = (),
) -> str:
    """門が読む最小の IR メタデータ（入力 1 本・出力の宣言・記号次元）。"""
    names = [f"out_{index}" for index in range(outputs)]
    return json.dumps(
        {
            "inputs": [{"name": name, "shape": list(shape)}],
            "outputs": names,
            "values": {output: {"dtype": "f32", "shape": [1, hidden]} for output in names},
            "symbols": list(symbols),
        }
    )


def _siglip2_container(dtype: str, graph: str | None, storage: str | None) -> bytes | list[bytes]:
    """系列に置く vision tower の中身。

    組み立てへ届く既定の形は**正当な IR コンポーネント**でなければならない（組み立ては入力を
    IR v1 の全規則で見る — `karume.dist.assert_weight_components_verified`）。正当な側は
    shard 列（`list[bytes]`・先頭がグラフ shard — ADR 0081）で、偽コンテナは代表 path 1 本の
    `bytes` のまま（計画の門で止まるので分割の側まで届かない）。

    軸は 2 本ある:

    - `dtype` は**単一 dtype の偽コンテナ**が名乗るヘッダ dtype。要求検査（「F32 が無い」）を
      見るケースはこちらでしか作れない — 実物どおりの f16 系列は F32 も含むので、要求検査は
      通ってしまう。
    - `storage` は**実物どおりの混成コンテナ**の格納形（f16 / i8 / i4 は適格外の重みと bias が
      F32 で残り、i4 はさらに I8 が混ざる）。禁止表（{@link SIGLIP2_STORAGE_FORBIDDEN}）の門は
      この形でしか試せない。
    """
    if storage is not None or (graph is None and dtype == "F32"):
        return ir_container(
            mark="siglip2-vision",
            storage=storage if storage is not None else dtype.lower(),
            inputs=(("pixel_values", (1, 3, _SIGLIP2_HEIGHT, _SIGLIP2_WIDTH)),),
            outputs=([1, _SIGLIP2_HIDDEN],),
        )
    return _fake_safetensors(
        dtype, b"siglip2-vision-weights", {IR_METADATA_KEY: graph or _siglip2_graph()}
    )


def _build_siglip2_sources(
    root: Path,
    *,
    model: str = SIGLIP2_DEFAULT_MODEL,
    graph: str | None = None,
    dtype: str = "F32",
    storage: str | None = None,
    preprocessor: Mapping[str, Any] | None = _SIGLIP2_PREPROCESSOR,
) -> Siglip2Sources:
    """系列 + 実重みの置き場を偽資産で再現する（配布しない `io.*` の混入込み）。

    並びは `_shared.paths` の実レイアウト（`outputs/series/` と `inputs/`）に揃える — CLI 経路の
    テストが root を差し替えるだけで同じ木を指せる形。
    """
    checkpoint = siglip2_checkpoint(model)
    sources = Siglip2Sources(
        series=root / "outputs" / "series" / checkpoint,
        model=root / "inputs" / "siglip2" / checkpoint,
    )
    write_component(sources.series / "model.safetensors", _siglip2_container(dtype, graph, storage))
    # 配布に入ってはいけない E2E フィクスチャ（系列には実際にこれが並んでいる）。
    _write(sources.series / "io.ramp.safetensors", b"io-fixture")
    if preprocessor is not None:
        _write(
            sources.model / "preprocessor_config.json",
            json.dumps(preprocessor, ensure_ascii=False).encode("utf-8"),
        )
    return sources


@pytest.fixture
def siglip2_assembled(tmp_path: Path) -> tuple[Path, dict]:
    sources = _build_siglip2_sources(tmp_path)
    out_dir = tmp_path / "models" / siglip2_repo_name(SIGLIP2_DEFAULT_MODEL)
    manifest = assemble_family(
        [siglip2_plan(sources, SIGLIP2_DEFAULT_MODEL)], out_dir, SIGLIP2_DEFAULT_MODEL
    )
    return out_dir, manifest


def _siglip2_model(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    return manifest["models"][SIGLIP2_DEFAULT_MODEL]


class TestSiglip2Layout:
    def test_it_places_the_single_graph_under_the_model_subtree(self, siglip2_assembled) -> None:
        out_dir, _ = siglip2_assembled
        expected = _in_subtree(SIGLIP2_DEFAULT_MODEL, _placed_paths())
        assert _present(out_dir) == sorted([*expected, MANIFEST_FILENAME])

    def test_it_never_carries_the_io_fixtures(self, siglip2_assembled) -> None:
        out_dir, _ = siglip2_assembled
        assert list(out_dir.rglob("io.*")) == []

    def test_it_declares_one_graph_and_no_assets(self, siglip2_assembled) -> None:
        """実行に要るのはグラフ 1 本だけ（tokenizer も表も無い = `assets` は空）。"""
        _, manifest = siglip2_assembled
        model = _siglip2_model(manifest)
        assert model["pipeline"] == "siglip2/1"
        assert list(model["weights"]) == [SIGLIP2_ROLE]
        assert model["assets"] == {}
        # 席は 1 つ（f32 系列 1 本きり）。dtype ラベルは自動補完で完全写像になる。
        assert list(model["quants"]) == ["f32"]
        assert model["defaultQuant"] == "f32"
        assert model["quants"]["f32"]["weights"] == {SIGLIP2_ROLE: "f32"}
        assert model["quants"]["f32"]["session"] == {}

    def test_it_reassembles_over_a_previous_run(self, tmp_path: Path) -> None:
        sources = _build_siglip2_sources(tmp_path)
        out_dir = tmp_path / "models" / siglip2_repo_name(SIGLIP2_DEFAULT_MODEL)
        first = assemble_family([siglip2_plan(sources)], out_dir, SIGLIP2_DEFAULT_MODEL)
        assert first == assemble_family([siglip2_plan(sources)], out_dir, SIGLIP2_DEFAULT_MODEL)
        assert verify_dist(out_dir)

    def test_it_refuses_a_compressed_asset_in_the_f32_seat(self, tmp_path: Path) -> None:
        """格納形は系列ディレクトリ名でなくヘッダが正（`--dtype` 付け忘れの逆向き）。"""
        sources = _build_siglip2_sources(tmp_path, dtype="F16")
        with pytest.raises(DistError, match="F32 が無い"):
            siglip2_plan(sources)

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
        sources = _build_siglip2_sources(tmp_path, storage=storage)
        with pytest.raises(DistError, match=rf"{SIGLIP2_ROLE}: .* {intruder} がある"):
            siglip2_plan(sources)

    def test_the_plain_f32_series_passes_both_storage_gates(self, tmp_path: Path) -> None:
        """正常対 — 素の f32 系列は要求検査も禁止表も通る（禁止表が恒真に倒れていない）。"""
        sources = _build_siglip2_sources(tmp_path, storage="f32")

        assert siglip2_plan(sources).name == SIGLIP2_DEFAULT_MODEL


class TestSiglip2PipelineConfig:
    """`pipelineConfig` はロード側（`src/siglip2/config.ts`）のスキーマと欄名まで一致する。"""

    def test_it_declares_exactly_the_fields_the_loader_accepts(self, siglip2_assembled) -> None:
        _, manifest = siglip2_assembled
        assert tuple(_siglip2_model(manifest)["pipelineConfig"]) == _SIGLIP2_CONFIG_KEYS

    def test_it_derives_the_preprocessing_constants_from_the_checkpoint(
        self, siglip2_assembled
    ) -> None:
        """焼き込んでいれば実物（224 / 384・mean = std = 0.5）の数が出てくる。"""
        _, manifest = siglip2_assembled
        config = _siglip2_model(manifest)["pipelineConfig"]
        assert config["imageWidth"] == _SIGLIP2_WIDTH
        assert config["imageHeight"] == _SIGLIP2_HEIGHT
        assert config["imageMean"] == _SIGLIP2_PREPROCESSOR["image_mean"]
        assert config["imageStd"] == _SIGLIP2_PREPROCESSOR["image_std"]
        assert config["interpolation"] == "bilinear"

    def test_it_derives_the_hidden_width_from_the_exported_graph(self, siglip2_assembled) -> None:
        """`hiddenDim` の出どころはグラフの出力宣言 1 つきり（config.json は幅を持たない）。"""
        _, manifest = siglip2_assembled
        assert _siglip2_model(manifest)["pipelineConfig"]["hiddenDim"] == _SIGLIP2_HIDDEN

    def test_it_refuses_a_missing_preprocessor_config(self, tmp_path: Path) -> None:
        sources = _build_siglip2_sources(tmp_path, preprocessor=None)
        with pytest.raises(DistError, match="組み立ての入力が無い"):
            siglip2_plan(sources)

    def test_it_refuses_a_checkpoint_that_asks_for_another_interpolation(
        self, tmp_path: Path
    ) -> None:
        """`resample` 3（BICUBIC）は TS 側に実装が無い — 黙って bilinear で通さない。"""
        sources = _build_siglip2_sources(
            tmp_path, preprocessor={**_SIGLIP2_PREPROCESSOR, "resample": 3}
        )
        with pytest.raises(DistError, match="resample"):
            siglip2_plan(sources)

    def test_it_refuses_a_rescale_factor_the_host_cannot_express(self, tmp_path: Path) -> None:
        """TS 側は 8bit 画素を 255 で割る形で閉じている（除数は宣言に無い）。"""
        sources = _build_siglip2_sources(
            tmp_path, preprocessor={**_SIGLIP2_PREPROCESSOR, "rescale_factor": 1.0 / 127.5}
        )
        with pytest.raises(DistError, match="rescale_factor"):
            siglip2_plan(sources)

    @pytest.mark.parametrize("flag", ["do_resize", "do_rescale", "do_normalize"])
    def test_it_refuses_a_checkpoint_that_skips_one_of_the_three_stages(
        self, tmp_path: Path, flag: str
    ) -> None:
        sources = _build_siglip2_sources(
            tmp_path, preprocessor={**_SIGLIP2_PREPROCESSOR, flag: False}
        )
        with pytest.raises(DistError, match=flag):
            siglip2_plan(sources)

    @pytest.mark.parametrize("flag", ["do_center_crop", "do_pad"])
    def test_it_refuses_a_checkpoint_that_wants_a_crop_or_a_pad(
        self, tmp_path: Path, flag: str
    ) -> None:
        """この前処理層は crop も pad も持たない（アスペクト比を保たない伸縮 1 本）。"""
        sources = _build_siglip2_sources(
            tmp_path, preprocessor={**_SIGLIP2_PREPROCESSOR, flag: True}
        )
        with pytest.raises(DistError, match=flag):
            siglip2_plan(sources)

    @pytest.mark.parametrize(
        "patch",
        [
            {"image_mean": [0.1, 0.2]},
            {"image_std": [0.4, 0.0, 0.6]},
            {"image_std": [0.4, "0.5", 0.6]},
            {"size": {"height": 0, "width": _SIGLIP2_WIDTH}},
        ],
    )
    def test_it_refuses_constants_outside_the_loader_value_range(
        self, tmp_path: Path, patch: Mapping[str, Any]
    ) -> None:
        """値域はロード側（`config.ts`）と同じ — std 0 は 0 除算で ±Infinity を静かに作る。"""
        sources = _build_siglip2_sources(tmp_path, preprocessor={**_SIGLIP2_PREPROCESSOR, **patch})
        with pytest.raises(DistError):
            siglip2_plan(sources)


class TestSiglip2GraphGate:
    """組み立て門 — ずれても配布形としては成立してしまう組み合わせを、配置の前に落とす。"""

    def test_it_refuses_a_graph_baked_for_another_resolution(self, tmp_path: Path) -> None:
        """前処理の寸法（preprocessor_config）と焼かれた解像度は別々に決まる。

        base の前処理 config と so400m のグラフを組み合わせても、ここが無ければ配布形は
        成立し、利用者の手元で Session の shape 検査が「どちらが正か」を伝えないまま落ちる。
        """
        sources = _build_siglip2_sources(tmp_path, graph=_siglip2_graph(shape=(1, 3, 384, 384)))
        with pytest.raises(DistError, match="前処理の寸法と焼かれた解像度が別の版"):
            siglip2_plan(sources)

    def test_it_refuses_a_graph_whose_axes_are_transposed(self, tmp_path: Path) -> None:
        """非正方の寸法だけが検出できる取り違え（`[1,3,W,H]`）。"""
        sources = _build_siglip2_sources(
            tmp_path, graph=_siglip2_graph(shape=(1, 3, _SIGLIP2_WIDTH, _SIGLIP2_HEIGHT))
        )
        with pytest.raises(DistError, match="前処理の寸法と焼かれた解像度が別の版"):
            siglip2_plan(sources)

    def test_it_refuses_a_graph_with_a_second_output(self, tmp_path: Path) -> None:
        """`last_hidden_state` 込みの別 export は `hiddenDim` の出どころごと別物になる。"""
        sources = _build_siglip2_sources(tmp_path, graph=_siglip2_graph(outputs=2))
        with pytest.raises(DistError, match="pooler_output 1 本だけ"):
            siglip2_plan(sources)

    def test_it_refuses_a_graph_with_a_symbolic_axis(self, tmp_path: Path) -> None:
        """解像度もパッチ数も固定なので、動かす軸は 1 本も無い。"""
        sources = _build_siglip2_sources(tmp_path, graph=_siglip2_graph(symbols=("T",)))
        with pytest.raises(DistError, match="記号次元"):
            siglip2_plan(sources)

    def test_it_refuses_a_renamed_input(self, tmp_path: Path) -> None:
        """実行側は名前で束ねるので、綴りが変われば束ねられない。"""
        sources = _build_siglip2_sources(tmp_path, graph=_siglip2_graph(name="pixels"))
        with pytest.raises(DistError, match="グラフ入力"):
            siglip2_plan(sources)


class TestSiglip2ModelCard:
    def _run(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, model: str) -> Path:
        """偽資産だけで CLI を 1 周回す（実重みの置き場も tmp へ寄せる）。"""
        from siglip2 import distribution

        sources = _build_siglip2_sources(tmp_path, model=model)
        # `INPUTS_ROOT`（系列の外にある実重み）を引くのは recipe 側 — ドライバの `dist`
        # ではなくこちらの束縛を外す（`DIST_ROOT` は逆でドライバ側）。
        monkeypatch.setattr(distribution, "INPUTS_ROOT", tmp_path / "inputs")
        out_dir = tmp_path / "dist"
        main(
            [
                "--pipeline",
                "siglip2",
                "--model",
                model,
                "--series",
                str(sources.series.parent),
                "--out",
                str(out_dir),
            ]
        )
        return out_dir

    def test_it_describes_the_image_embedding_distribution(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        out_dir = self._run(tmp_path, monkeypatch, SIGLIP2_DEFAULT_MODEL)
        card = (out_dir / MODEL_CARD_FILENAME).read_text(encoding="utf-8")
        assert "pipeline_tag: image-feature-extraction" in card
        assert "license: apache-2.0" in card
        # 格納形を変えない配布形は 4 値のどれでもない（Hub の推論に任せる）。
        assert "base_model_relation" not in card
        # text tower を持たない事実は、利用者が最初に確かめたい制約そのもの。
        assert "The text tower is not here." in card
        # 前処理は同梱（利用者が渡すのは生の RGB8）。数は manifest から降りてくる。
        assert f"resizes to {_SIGLIP2_WIDTH} × {_SIGLIP2_HEIGHT}" in card
        assert f"`pooler_output`, {_SIGLIP2_HIDDEN} f32 values, not L2-normalized" in card
        # カードは**検証を通った**配布形から描かれる。
        assert verify_dist(out_dir)

    def test_the_usage_snippet_enumerates_the_choices_from_the_manifest(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """裁定 2026-08-12 の形: optional はコメント併記で、選べる値は manifest 由来の列挙。

        `embed()` に実行時ノブは無いので、この pipeline の optional は model / quant だけ。
        1 席しか無くても「available:」で綴るのは、席が増えたときに文面が自動で追従するため
        （「the only one this repository ships」のような焼き込みは、増えた瞬間に嘘になる）。
        """
        out_dir = self._run(tmp_path, monkeypatch, SIGLIP2_DEFAULT_MODEL)
        card = (out_dir / MODEL_CARD_FILENAME).read_text(encoding="utf-8")
        assert (
            f'  // model: "{SIGLIP2_DEFAULT_MODEL}",'
            f" // default — available: {SIGLIP2_DEFAULT_MODEL}" in card
        )
        assert '  // quant: "f32", // default — available: f32' in card
        assert "the only one this repository ships" not in card

    def test_the_attribution_follows_the_model_name(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """帰属はモデル名から一意に決まる（選ばせる軸にしない = 取り違えようがない）。"""
        card = (self._run(tmp_path, monkeypatch, "so400m") / MODEL_CARD_FILENAME).read_text(
            encoding="utf-8"
        )
        assert f"base_model: {SIGLIP2_UPSTREAM['so400m']}" in card
        assert SIGLIP2_UPSTREAM["base"] not in card


class TestSiglip2Cli:
    def test_the_default_output_directory_follows_the_single_model(self) -> None:
        assert default_out_dir(PIPELINE, [SIGLIP2_DEFAULT_MODEL]).name == "karume-siglip2-base"
        assert default_out_dir(PIPELINE, ["so400m"]).name == "karume-siglip2-so400m"

    def test_one_attribution_profile_needs_no_choice(self) -> None:
        profiles = PIPELINE.card_profiles
        assert len(profiles) == 1
        assert resolve_card_renderer(PIPELINE, None) is next(iter(profiles.values()))

    def test_the_series_name_is_the_upstream_repository_name(self, tmp_path: Path) -> None:
        """系列名 / 入力素材のディレクトリ名は上流リポ名そのもの（表を 2 つ持たない）。"""
        for model, repo in SIGLIP2_UPSTREAM.items():
            sources = siglip2_sources(tmp_path, model)
            assert sources.series.name == repo.split("/", 1)[1]
            assert sources.model.name == sources.series.name

    def test_it_refuses_a_model_it_has_no_attribution_for(self, tmp_path: Path) -> None:
        """帰属表に無いモデル名は「出所を名乗れない」ので、系列を探す前に落とす。"""
        with pytest.raises(DistError, match="知らない"):
            siglip2_sources(tmp_path, "large")

    def test_it_assembles_into_the_pipeline_default_directory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import dist
        from siglip2 import distribution

        sources = _build_siglip2_sources(tmp_path)
        # `DIST_ROOT` は既定の出力先を決めるドライバ側（`dist.default_out_dir`）、`INPUTS_ROOT` は
        # 系列の外の入力を引く recipe 側 — 別モジュールの束縛を別々に外す。
        monkeypatch.setattr(dist, "DIST_ROOT", tmp_path / "models")
        monkeypatch.setattr(distribution, "INPUTS_ROOT", tmp_path / "inputs")

        main(["--pipeline", "siglip2", "--series", str(sources.series.parent)])

        out_dir = tmp_path / "models" / siglip2_repo_name(SIGLIP2_DEFAULT_MODEL)
        expected = _in_subtree(SIGLIP2_DEFAULT_MODEL, _placed_paths())
        assert sorted(verify_dist(out_dir)) == sorted(expected)
