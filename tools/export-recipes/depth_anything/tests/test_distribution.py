"""Depth Anything V2 の配布 recipe（`depth_anything.distribution`）— 組み立て 1 周ぶんの単体テスト。

実資産は使わない — dist が safetensors から読むのは**ヘッダの dtype 集合**と IR メタデータ
（`__metadata__`）だけなので、数十バイトの正規ヘッダ付き偽資産で層の振る舞いは全て観測できる。
前処理定数の出どころ（`preprocessor_config.json`）も合成 JSON で足りる。

manifest v2（`karume/2` — ADR 0041）以降、リポ内レイアウトは一律「モデル別サブツリー +
`shared/`」なので、期待 path は全て `<モデル名>/…` を頭に持つ。

core だけで観測できる層（合成計画で足りる規模上限・quant 完全写像・staging/swap の不変条件・
帰属プロファイルの解決規則）は `tools/exporter/tests/test_dist.py` が持つ（ADR 0065 段 3+4 の
分割）。テンプレート単位の門は `depth_anything/tests/test_card.py`。
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from depth_anything.card import DEPTH_ANYTHING_LICENSE, DEPTH_ANYTHING_UPSTREAM
from depth_anything.distribution import (
    DEPTH_ANYTHING_DEFAULT_MODEL,
    DEPTH_ANYTHING_OUTPUT_PATHS,
    DEPTH_ANYTHING_ROLE,
    PIPELINE,
    DepthAnythingSources,
    depth_anything_checkpoint,
    depth_anything_plan,
    depth_anything_repo_name,
    depth_anything_series_name,
    depth_anything_sources,
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


# ---- Depth Anything V2（depth-estimation）------------------------------------
#
# 偽資産は**実物と違う数**にする（96×64 の非正方・mean/std は ImageNet 統計でない）— 前処理
# 定数や寸法を焼き込んでいれば落ちる。非正方なのは、`imageWidth` / `imageHeight` の取り違えが
# 正方形では原理的に検出できないため。

#: `pipelineConfig` の欄名（TS 側 `packages/models/src/depth-anything/config.ts` の `ROOT_KEYS` の
#: 写し）。**ロード側は未知キーも欠落も parse 時に落とす**ので、焼く側とロード側の欄名は
#: 完全一致が要る。
_DEPTH_ANYTHING_CONFIG_KEYS = (
    "imageWidth",
    "imageHeight",
    "imageMean",
    "imageStd",
    "interpolation",
)

_DEPTH_ANYTHING_HEIGHT = 96
_DEPTH_ANYTHING_WIDTH = 64

#: 偽の `preprocessor_config.json`（実物と同じ欄・違う値。`keep_aspect_ratio` /
#: `ensure_multiple_of` は実物どおり載せておく — 宣言へ写していないことを見るため）。
_DEPTH_ANYTHING_PREPROCESSOR: Mapping[str, Any] = {
    "do_normalize": True,
    "do_pad": False,
    "do_rescale": True,
    "do_resize": True,
    "ensure_multiple_of": 14,
    "image_mean": [0.1, 0.2, 0.3],
    "image_processor_type": "DPTImageProcessor",
    "image_std": [0.4, 0.5, 0.6],
    "keep_aspect_ratio": True,
    "resample": 3,
    "rescale_factor": 1.0 / 255.0,
    "size": {"height": _DEPTH_ANYTHING_HEIGHT, "width": _DEPTH_ANYTHING_WIDTH},
}


def _depth_anything_graph(
    *,
    shape: Sequence[Any] = (1, 3, _DEPTH_ANYTHING_HEIGHT, _DEPTH_ANYTHING_WIDTH),
    out_shape: Sequence[Any] | None = None,
    name: str = "pixel_values",
    outputs: int = 1,
    symbols: Sequence[str] = (),
) -> str:
    """門が読む最小の IR メタデータ（入力 1 本・出力の宣言・記号次元）。"""
    names = [f"out_{index}" for index in range(outputs)]
    depth = list(out_shape) if out_shape is not None else [1, shape[2], shape[3]]
    return json.dumps(
        {
            "inputs": [{"name": name, "shape": list(shape)}],
            "outputs": names,
            "values": {output: {"dtype": "f32", "shape": depth} for output in names},
            "symbols": list(symbols),
        }
    )


def _build_depth_anything_sources(
    root: Path,
    *,
    model: str = DEPTH_ANYTHING_DEFAULT_MODEL,
    graph: str | None = None,
    dtype: str = "F32",
    preprocessor: Mapping[str, Any] | None = _DEPTH_ANYTHING_PREPROCESSOR,
) -> DepthAnythingSources:
    """系列 + 実重みの置き場を偽資産で再現する（配布しない `io.*` の混入込み）。

    並びは `_shared.paths` の実レイアウト（`outputs/series/` と `inputs/`）に揃える — CLI 経路の
    テストが root を差し替えるだけで同じ木を指せる形。
    """
    sources = DepthAnythingSources(
        series=root / "outputs" / "series" / depth_anything_series_name(model),
        model=root / "inputs" / "depth-anything" / depth_anything_checkpoint(model),
    )
    _write(
        sources.series / "model.safetensors",
        _fake_safetensors(
            dtype,
            b"depth-anything-weights",
            {IR_METADATA_KEY: graph if graph is not None else _depth_anything_graph()},
        ),
    )
    # 配布に入ってはいけない E2E フィクスチャ（系列には実際にこれが並んでいる）。
    _write(sources.series / "io.ramp.safetensors", b"io-fixture")
    if preprocessor is not None:
        _write(
            sources.model / "preprocessor_config.json",
            json.dumps(preprocessor, ensure_ascii=False).encode("utf-8"),
        )
    return sources


@pytest.fixture
def depth_anything_assembled(tmp_path: Path) -> tuple[Path, dict]:
    sources = _build_depth_anything_sources(tmp_path)
    out_dir = tmp_path / "models" / depth_anything_repo_name(DEPTH_ANYTHING_DEFAULT_MODEL)
    manifest = assemble_family(
        [depth_anything_plan(sources, DEPTH_ANYTHING_DEFAULT_MODEL)],
        out_dir,
        DEPTH_ANYTHING_DEFAULT_MODEL,
    )
    return out_dir, manifest


def _depth_anything_model(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    return manifest["models"][DEPTH_ANYTHING_DEFAULT_MODEL]


class TestDepthAnythingLayout:
    def test_it_places_the_single_graph_under_the_model_subtree(
        self, depth_anything_assembled
    ) -> None:
        out_dir, _ = depth_anything_assembled
        expected = _in_subtree(DEPTH_ANYTHING_DEFAULT_MODEL, DEPTH_ANYTHING_OUTPUT_PATHS.values())
        assert _present(out_dir) == sorted([*expected, MANIFEST_FILENAME])

    def test_it_never_carries_the_io_fixtures(self, depth_anything_assembled) -> None:
        out_dir, _ = depth_anything_assembled
        assert list(out_dir.rglob("io.*")) == []

    def test_it_declares_one_graph_and_no_assets(self, depth_anything_assembled) -> None:
        """実行に要るのはグラフ 1 本だけ（tokenizer も表も無い = `assets` は空）。"""
        _, manifest = depth_anything_assembled
        model = _depth_anything_model(manifest)
        assert model["pipeline"] == "depth-anything/1"
        assert list(model["weights"]) == [DEPTH_ANYTHING_ROLE]
        assert model["assets"] == {}
        assert list(model["quants"]) == ["f32"]
        assert model["defaultQuant"] == "f32"
        assert model["quants"]["f32"]["weights"] == {DEPTH_ANYTHING_ROLE: "f32"}
        assert model["quants"]["f32"]["session"] == {}

    def test_it_reassembles_over_a_previous_run(self, tmp_path: Path) -> None:
        sources = _build_depth_anything_sources(tmp_path)
        out_dir = tmp_path / "models" / depth_anything_repo_name(DEPTH_ANYTHING_DEFAULT_MODEL)
        first = assemble_family(
            [depth_anything_plan(sources)], out_dir, DEPTH_ANYTHING_DEFAULT_MODEL
        )
        assert first == assemble_family(
            [depth_anything_plan(sources)], out_dir, DEPTH_ANYTHING_DEFAULT_MODEL
        )
        assert verify_dist(out_dir)

    def test_it_refuses_a_compressed_asset_in_the_f32_seat(self, tmp_path: Path) -> None:
        """格納形は系列ディレクトリ名でなくヘッダが正（`--dtype` 付け忘れの逆向き）。"""
        sources = _build_depth_anything_sources(tmp_path, dtype="F16")
        with pytest.raises(DistError, match="F32 が無い"):
            depth_anything_plan(sources)


class TestDepthAnythingPipelineConfig:
    """`pipelineConfig` はロード側（`src/depth-anything/config.ts`）のスキーマと欄名まで一致。"""

    def test_it_declares_exactly_the_fields_the_loader_accepts(
        self, depth_anything_assembled
    ) -> None:
        _, manifest = depth_anything_assembled
        assert tuple(_depth_anything_model(manifest)["pipelineConfig"]) == (
            _DEPTH_ANYTHING_CONFIG_KEYS
        )

    def test_it_derives_the_preprocessing_constants_from_the_checkpoint(
        self, depth_anything_assembled
    ) -> None:
        """焼き込んでいれば実物（518² / ImageNet 統計）の数が出てくる。"""
        _, manifest = depth_anything_assembled
        config = _depth_anything_model(manifest)["pipelineConfig"]
        assert config["imageWidth"] == _DEPTH_ANYTHING_WIDTH
        assert config["imageHeight"] == _DEPTH_ANYTHING_HEIGHT
        assert config["imageMean"] == _DEPTH_ANYTHING_PREPROCESSOR["image_mean"]
        assert config["imageStd"] == _DEPTH_ANYTHING_PREPROCESSOR["image_std"]

    def test_it_declares_bicubic_not_the_bilinear_the_other_towers_use(
        self, depth_anything_assembled
    ) -> None:
        """`resample: 3` = PIL の BICUBIC。SigLIP2 / BiRefNet の bilinear と取り違えると、
        `pixel_values` が uint8 1 LSB の 34 倍ずれたままロードも実行も通る。"""
        _, manifest = depth_anything_assembled
        assert _depth_anything_model(manifest)["pipelineConfig"]["interpolation"] == "bicubic"

    def test_it_does_not_carry_the_aspect_ratio_knobs_into_the_declaration(
        self, depth_anything_assembled
    ) -> None:
        """上流 config の `keep_aspect_ratio` / `ensure_multiple_of` は宣言へ写さない。

        焼かれたグラフは正方 1 点でしか受け取らないので、保つ経路には行き先が無い。素通しで
        写すと TS 側の parse が未知キーで落ちる（= 配布形はできるのにロードできない）。
        """
        _, manifest = depth_anything_assembled
        config = _depth_anything_model(manifest)["pipelineConfig"]
        assert "keepAspectRatio" not in config
        assert "ensureMultipleOf" not in config

    def test_it_refuses_a_missing_preprocessor_config(self, tmp_path: Path) -> None:
        sources = _build_depth_anything_sources(tmp_path, preprocessor=None)
        with pytest.raises(DistError, match="組み立ての入力が無い"):
            depth_anything_plan(sources)

    def test_it_refuses_a_checkpoint_that_asks_for_another_interpolation(
        self, tmp_path: Path
    ) -> None:
        """`resample` 2（BILINEAR）を黙って bicubic として宣言しない。"""
        sources = _build_depth_anything_sources(
            tmp_path, preprocessor={**_DEPTH_ANYTHING_PREPROCESSOR, "resample": 2}
        )
        with pytest.raises(DistError, match="resample"):
            depth_anything_plan(sources)

    def test_it_refuses_a_rescale_factor_the_host_cannot_express(self, tmp_path: Path) -> None:
        """TS 側は 8bit 画素を 255 で割る形で閉じている（除数は宣言に無い）。"""
        sources = _build_depth_anything_sources(
            tmp_path, preprocessor={**_DEPTH_ANYTHING_PREPROCESSOR, "rescale_factor": 1.0 / 127.5}
        )
        with pytest.raises(DistError, match="rescale_factor"):
            depth_anything_plan(sources)

    @pytest.mark.parametrize("flag", ["do_resize", "do_rescale", "do_normalize"])
    def test_it_refuses_a_checkpoint_that_skips_one_of_the_three_stages(
        self, tmp_path: Path, flag: str
    ) -> None:
        sources = _build_depth_anything_sources(
            tmp_path, preprocessor={**_DEPTH_ANYTHING_PREPROCESSOR, flag: False}
        )
        with pytest.raises(DistError, match=flag):
            depth_anything_plan(sources)

    @pytest.mark.parametrize("flag", ["do_center_crop", "do_pad"])
    def test_it_refuses_a_checkpoint_that_wants_a_crop_or_a_pad(
        self, tmp_path: Path, flag: str
    ) -> None:
        """この前処理層は crop も pad も持たない（伸縮 1 本）。"""
        sources = _build_depth_anything_sources(
            tmp_path, preprocessor={**_DEPTH_ANYTHING_PREPROCESSOR, flag: True}
        )
        with pytest.raises(DistError, match=flag):
            depth_anything_plan(sources)

    @pytest.mark.parametrize(
        "patch",
        [
            {"image_mean": [0.1, 0.2]},
            {"image_std": [0.4, 0.0, 0.6]},
            {"image_std": [0.4, "0.5", 0.6]},
            {"size": {"height": 0, "width": _DEPTH_ANYTHING_WIDTH}},
        ],
    )
    def test_it_refuses_constants_outside_the_loader_value_range(
        self, tmp_path: Path, patch: Mapping[str, Any]
    ) -> None:
        """値域はロード側（`config.ts`）と同じ — std 0 は 0 除算で ±Infinity を静かに作る。"""
        sources = _build_depth_anything_sources(
            tmp_path, preprocessor={**_DEPTH_ANYTHING_PREPROCESSOR, **patch}
        )
        with pytest.raises(DistError):
            depth_anything_plan(sources)


class TestDepthAnythingGraphGate:
    """組み立て門 — ずれても配布形としては成立してしまう組み合わせを、配置の前に落とす。"""

    def test_it_refuses_a_graph_baked_for_another_resolution(self, tmp_path: Path) -> None:
        """サイズが違えば事前学習解像度も違う（Small 518² / 別サイズの組み合わせ）。"""
        sources = _build_depth_anything_sources(
            tmp_path, graph=_depth_anything_graph(shape=(1, 3, 518, 518))
        )
        with pytest.raises(DistError, match="前処理の寸法と焼かれた解像度が別の版"):
            depth_anything_plan(sources)

    def test_it_refuses_a_graph_whose_axes_are_transposed(self, tmp_path: Path) -> None:
        """非正方の寸法だけが検出できる取り違え（`[1,3,W,H]`）。"""
        sources = _build_depth_anything_sources(
            tmp_path,
            graph=_depth_anything_graph(
                shape=(1, 3, _DEPTH_ANYTHING_WIDTH, _DEPTH_ANYTHING_HEIGHT)
            ),
        )
        with pytest.raises(DistError, match="前処理の寸法と焼かれた解像度が別の版"):
            depth_anything_plan(sources)

    def test_it_refuses_a_depth_map_with_a_channel_axis(self, tmp_path: Path) -> None:
        """`[1, 1, H, W]` は要素数では `[1, H, W]` と区別できない（階数まで見る）。"""
        sources = _build_depth_anything_sources(
            tmp_path,
            graph=_depth_anything_graph(
                out_shape=(1, 1, _DEPTH_ANYTHING_HEIGHT, _DEPTH_ANYTHING_WIDTH)
            ),
        )
        with pytest.raises(DistError, match="チャネル軸なし"):
            depth_anything_plan(sources)

    def test_it_refuses_a_graph_with_a_second_output(self, tmp_path: Path) -> None:
        """中間段まで出す別 export は、位置で引く後段が別の値を深度として読む。"""
        sources = _build_depth_anything_sources(tmp_path, graph=_depth_anything_graph(outputs=2))
        with pytest.raises(DistError, match="深度マップ 1 本だけ"):
            depth_anything_plan(sources)

    def test_it_refuses_a_graph_with_a_symbolic_axis(self, tmp_path: Path) -> None:
        """解像度もパッチ数も固定なので、動かす軸は 1 本も無い。"""
        sources = _build_depth_anything_sources(
            tmp_path, graph=_depth_anything_graph(symbols=("T",))
        )
        with pytest.raises(DistError, match="記号次元"):
            depth_anything_plan(sources)

    def test_it_refuses_a_renamed_input(self, tmp_path: Path) -> None:
        """実行側は名前で束ねるので、綴りが変われば束ねられない。"""
        sources = _build_depth_anything_sources(tmp_path, graph=_depth_anything_graph(name="px"))
        with pytest.raises(DistError, match="グラフ入力"):
            depth_anything_plan(sources)


class TestDepthAnythingModelCard:
    def _run(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        """偽資産だけで CLI を 1 周回す（実重みの置き場も tmp へ寄せる）。"""
        from depth_anything import distribution

        sources = _build_depth_anything_sources(tmp_path)
        # `INPUTS_ROOT`（系列の外にある実重み）を引くのは recipe 側 — ドライバの `dist`
        # ではなくこちらの束縛を外す（`DIST_ROOT` は逆でドライバ側）。
        monkeypatch.setattr(distribution, "INPUTS_ROOT", tmp_path / "inputs")
        out_dir = tmp_path / "dist"
        main(
            [
                "--pipeline",
                "depth-anything",
                "--series",
                str(sources.series.parent),
                "--out",
                str(out_dir),
            ]
        )
        return out_dir

    def test_it_describes_the_relative_depth_distribution(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        out_dir = self._run(tmp_path, monkeypatch)
        card = (out_dir / MODEL_CARD_FILENAME).read_text(encoding="utf-8")
        assert "pipeline_tag: depth-estimation" in card
        assert f"license: {DEPTH_ANYTHING_LICENSE}" in card
        assert f"base_model: {DEPTH_ANYTHING_UPSTREAM['small']}" in card
        # 格納形を変えない配布形は 4 値のどれでもない（Hub の推論に任せる）。
        assert "base_model_relation" not in card
        # 相対深度であること（metric depth でないこと）は、利用者が最初に確かめる制約そのもの。
        assert "**Relative depth has no unit and no origin**" in card
        # 前処理は同梱（利用者が渡すのは生の RGB8）。数は manifest から降りてくる。
        assert f"resized to {_DEPTH_ANYTHING_WIDTH} × {_DEPTH_ANYTHING_HEIGHT} (bicubic" in card
        # カードは**検証を通った**配布形から描かれる。
        assert verify_dist(out_dir)

    def test_it_states_that_only_the_apache_licensed_size_is_redistributed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Base / Large は CC BY-NC 4.0。「無いこと」は読み手が最初に探す事実。"""
        card = (self._run(tmp_path, monkeypatch) / MODEL_CARD_FILENAME).read_text(encoding="utf-8")
        assert "Only the Small checkpoint is Apache-2.0." in card
        assert "CC BY-NC 4.0" in card

    def test_the_title_comes_from_the_upstream_checkpoint_name(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """サイズの綴りは帰属表 1 本から導く（2 表にすると別サイズとして売れてしまう）。"""
        card = (self._run(tmp_path, monkeypatch) / MODEL_CARD_FILENAME).read_text(encoding="utf-8")
        assert "# Depth Anything V2 Small — Karume" in card


class TestDepthAnythingCli:
    def test_the_default_output_directory_follows_the_single_model(self) -> None:
        assert (
            default_out_dir(PIPELINE, [DEPTH_ANYTHING_DEFAULT_MODEL]).name
            == "karume-depth-anything-v2-small"
        )

    def test_one_attribution_profile_needs_no_choice(self) -> None:
        profiles = PIPELINE.card_profiles
        assert len(profiles) == 1
        assert resolve_card_renderer(PIPELINE, None) is next(iter(profiles.values()))

    def test_the_series_name_is_the_lowercased_upstream_repository_name(
        self, tmp_path: Path
    ) -> None:
        """系列名 / 入力素材のディレクトリ名は上流リポ名そのもの（表を 2 つ持たない）。"""
        for model, repo in DEPTH_ANYTHING_UPSTREAM.items():
            checkpoint = repo.split("/", 1)[1]
            sources = depth_anything_sources(tmp_path, model)
            assert sources.series.name == checkpoint.lower()
            assert sources.model.name == checkpoint

    @pytest.mark.parametrize("model", ["base", "large"])
    def test_it_refuses_a_size_it_may_not_redistribute(self, tmp_path: Path, model: str) -> None:
        """上流で CC BY-NC 4.0 のサイズは帰属表に載っていない = 系列を探す前に落ちる。

        台本は `--model-dir` でどのサイズでも焼けるので、「焼けたものは配れる」と読める形に
        しないための唯一の門（`karume.dist` の Depth Anything 節の MUST）。
        """
        with pytest.raises(DistError, match="知らない"):
            depth_anything_sources(tmp_path, model)

    def test_it_assembles_into_the_pipeline_default_directory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import dist
        from depth_anything import distribution

        sources = _build_depth_anything_sources(tmp_path)
        # `DIST_ROOT` は既定の出力先を決めるドライバ側（`dist.default_out_dir`）、`INPUTS_ROOT` は
        # 系列の外の入力を引く recipe 側 — 別モジュールの束縛を別々に外す。
        monkeypatch.setattr(dist, "DIST_ROOT", tmp_path / "models")
        monkeypatch.setattr(distribution, "INPUTS_ROOT", tmp_path / "inputs")

        main(["--pipeline", "depth-anything", "--series", str(sources.series.parent)])

        out_dir = tmp_path / "models" / depth_anything_repo_name(DEPTH_ANYTHING_DEFAULT_MODEL)
        expected = _in_subtree(DEPTH_ANYTHING_DEFAULT_MODEL, DEPTH_ANYTHING_OUTPUT_PATHS.values())
        assert sorted(verify_dist(out_dir)) == sorted(expected)
