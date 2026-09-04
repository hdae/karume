"""gemma4 の配布 recipe（`gemma4.distribution`）— 組み立て 1 周ぶんの単体テスト。

実資産（3.7GiB）は使わない。組み立てへ届く入力は数 KB の**正当な最小の製品系列**
（{@link gemma4.tests.product_fixture}）で、門に落とされることを見るケースだけがその変種になる。

核は「別々の台本が持つ同じ事実を、組み立て時に突き合わせる」— 製品グラフ（`export_product.py`）/
PLE sidecar（同）/ トークナイザ資産（`tokenizer.py`）/ 上流の生成既定（チェックポイント）は
4 つとも独立に動けるので、噛み合っていないことは**配布形を並べる前**にしか落とせない。

core だけで観測できる層（規模上限・quant 完全写像・staging/swap の不変条件・帰属プロファイルの
解決規則）は `tools/exporter/tests/test_dist.py` が持つ（ADR 0065 段 3+4 の分割）。
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import pytest
from ir_fixtures import ir_container
from shard_series import placed_paths, replace_component, write_component

from _shared.licenses import APACHE_LICENSE_2_0_PATH
from gemma4 import distribution as gemma4_distribution
from gemma4.card import GEMMA4_UPSTREAM, render_gemma4_model_card
from gemma4.distribution import (
    GEMMA4_CAPACITY,
    GEMMA4_CHUNK_LENGTH,
    GEMMA4_DEFAULT_MODEL,
    GEMMA4_MAX_CHUNK_LENGTH,
    GEMMA4_OUTPUT_PATHS,
    GEMMA4_PLE_INDEX_ROLE,
    GEMMA4_ROLE,
    GEMMA4_TOKENIZER_ROLE,
    GEMMA4_WEIGHTS,
    Gemma4Sources,
    gemma4_plan,
    gemma4_repo_name,
    gemma4_rope_input_name,
    gemma4_series_name,
    gemma4_sources,
)
from gemma4.rope import FULL_ATTENTION, SLIDING_ATTENTION
from gemma4.tests import product_fixture as fixture
from karume.dist import (
    MANIFEST_FILENAME,
    DistError,
    assemble_family,
    resolve_card_renderer,
    verify_dist,
)

#: 合成の寸法で成立する実行時ノブ（実物は 768 / 4096・合成の位置上限は 37）。
SMALL_CHUNK = 2
SMALL_CAPACITY = 4


@pytest.fixture(autouse=True)
def _small_runtime_knobs(monkeypatch: pytest.MonkeyPatch) -> None:
    """実行時ノブを合成の寸法へ寄せる（`capacity ≤ maxPosition` の門を素通りさせるため）。

    実物の値そのままだと合成の位置上限（37）に対して容量 4096 が外れる — その組み合わせ自体は
    {@link TestGemma4Config.test_it_refuses_a_capacity_beyond_the_model_limit} が門として使う。
    """
    monkeypatch.setattr(gemma4_distribution, "GEMMA4_CHUNK_LENGTH", SMALL_CHUNK)
    monkeypatch.setattr(gemma4_distribution, "GEMMA4_CAPACITY", SMALL_CAPACITY)


def _sources(root: Path) -> Gemma4Sources:
    return Gemma4Sources(
        product=root / "series" / "gemma4-e2b-product",
        tokenizer=root / "series" / "gemma4-e2b-tokenizer",
        model=root / "inputs" / "gemma-4-E2B-it",
    )


def _build(root: Path, **overrides: Any) -> Gemma4Sources:
    sources = _sources(root)
    fixture.write_series(sources.product, sources.tokenizer, sources.model, **overrides)
    return sources


def _present(out_dir: Path) -> list[str]:
    return sorted(str(path.relative_to(out_dir)) for path in out_dir.rglob("*") if path.is_file())


def _in_subtree(model: str, paths: Iterable[str]) -> list[str]:
    return [f"{model}/{rel}" for rel in paths]


def _assemble(root: Path, model: str = GEMMA4_DEFAULT_MODEL) -> tuple[Path, dict[str, Any]]:
    sources = _build(root)
    out_dir = root / "models" / gemma4_repo_name(model)
    manifest = assemble_family(
        [gemma4_plan(sources, model)],
        out_dir,
        model,
        root_files=gemma4_distribution.PIPELINE.root_files,
    )
    return out_dir, manifest


@pytest.fixture
def gemma4_assembled(tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    return _assemble(tmp_path)


def _model(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    return manifest["models"][GEMMA4_DEFAULT_MODEL]


class TestGemma4Layout:
    def test_it_places_the_graph_sidecar_and_tokenizer_under_the_model_subtree(
        self, gemma4_assembled
    ) -> None:
        out_dir, _ = gemma4_assembled
        index = fixture.ple_index([(0, 4), (4, fixture.VOCAB)])
        expected = _in_subtree(
            GEMMA4_DEFAULT_MODEL,
            [
                *placed_paths(GEMMA4_OUTPUT_PATHS, GEMMA4_WEIGHTS),
                *(f"ple/{shard['file']}" for shard in index["shards"]),
            ],
        )
        # 法的テキスト 2 本（Apache 2.0 §4）とカードは manifest が宣言しないメタ席。
        assert _present(out_dir) == sorted(
            [*expected, MANIFEST_FILENAME, "LICENSE.md", "NOTICE.md"]
        )

    def test_it_never_carries_the_acceptance_only_files(self, gemma4_assembled) -> None:
        """`ple.probe.safetensors` / `reference.json` は系列に同居するが配布へは入らない。"""
        out_dir, _ = gemma4_assembled
        assert list(out_dir.rglob("ple.probe.*")) == []
        assert list(out_dir.rglob("reference.json")) == []

    def test_it_declares_one_graph_and_the_sidecar_as_assets(self, gemma4_assembled) -> None:
        _, manifest = gemma4_assembled
        model = _model(manifest)
        assert model["pipeline"] == "gemma4/1"
        assert list(model["weights"]) == [GEMMA4_ROLE]
        assert list(model["quants"]) == ["i4"]
        assert model["defaultQuant"] == "i4"
        assert model["quants"]["i4"]["weights"] == {GEMMA4_ROLE: "i4"}
        assert model["quants"]["i4"]["session"] == {}

    def test_the_sidecar_asset_names_are_the_index_file_names(self, gemma4_assembled) -> None:
        """MUST: 取得キー = `ple.json` の `shards[].file`（読み手はそれ 1 本で引く）。"""
        out_dir, manifest = gemma4_assembled
        model = _model(manifest)
        index = json.loads(
            (out_dir / model["assets"][GEMMA4_PLE_INDEX_ROLE]["path"]).read_text(encoding="utf-8")
        )
        declared = [shard["file"] for shard in index["shards"]]
        assert declared, "索引が shard を 1 本も持たない"
        for file in declared:
            assert file in model["assets"]
        sidecars = sorted(
            name
            for name in model["assets"]
            if name not in (GEMMA4_TOKENIZER_ROLE, GEMMA4_PLE_INDEX_ROLE)
        )
        assert sidecars == sorted(declared)

    def test_it_reassembles_over_a_previous_run(self, tmp_path: Path) -> None:
        first_dir, first = _assemble(tmp_path)
        second_dir, second = _assemble(tmp_path)
        assert first_dir == second_dir
        assert first == second
        assert verify_dist(first_dir)


class TestGemma4Config:
    """`pipelineConfig` の 6 欄 — 導出（`maxPosition` / `rope` / `sampler`）と実行時ノブの関係。"""

    def test_it_derives_max_position_from_the_upstream_declaration(self, gemma4_assembled) -> None:
        """MUST: 写経しない — 出どころは上流 `text_config.max_position_embeddings` だけ。"""
        _, manifest = gemma4_assembled
        assert _model(manifest)["pipelineConfig"]["maxPosition"] == fixture.MAX_POSITION

    def test_it_derives_the_rope_parameters_from_the_upstream_config(
        self, gemma4_assembled
    ) -> None:
        """層種別ごとの theta / headDim / rotaryDim（full だけ `global_head_dim` を読む）。"""
        _, manifest = gemma4_assembled

        assert _model(manifest)["pipelineConfig"]["rope"] == {
            SLIDING_ATTENTION: {
                "theta": fixture.SLIDING_THETA,
                "headDim": fixture.SLIDING_HEAD_DIM,
                # `default` は全周波数が回る
                "rotaryDim": fixture.SLIDING_HEAD_DIM,
            },
            FULL_ATTENTION: {
                "theta": fixture.FULL_THETA,
                "headDim": fixture.FULL_HEAD_DIM,
                # `proportional`: 2 × int(0.5 × 8 // 2) = 4
                "rotaryDim": 4,
            },
        }

    def test_it_refuses_a_rope_type_it_cannot_mirror(self, tmp_path: Path) -> None:
        """MUST: 式が別物なら落とす（ホストが宣言どおりに組めない表を配らない）。"""
        text_config = json.loads(json.dumps(dict(fixture.TEXT_CONFIG)))
        text_config["rope_parameters"][SLIDING_ATTENTION]["rope_type"] = "yarn"
        sources = _sources(tmp_path)
        fixture.write_series(
            sources.product, sources.tokenizer, sources.model, text_config=text_config
        )

        with pytest.raises(DistError, match="rope_type"):
            gemma4_plan(sources)

    @pytest.mark.parametrize(
        ("dropped", "message"),
        [("max_position_embeddings", "max_position_embeddings"), ("layer_types", "layer_types")],
    )
    def test_it_refuses_a_checkpoint_config_without_the_declaration(
        self, tmp_path: Path, dropped: str, message: str
    ) -> None:
        """MUST: 導出元が欠けたら落とす（既定へ落とすと配布形が勝手な数を名乗る）。"""
        text_config = {key: value for key, value in fixture.TEXT_CONFIG.items() if key != dropped}
        sources = _sources(tmp_path)
        fixture.write_series(
            sources.product, sources.tokenizer, sources.model, text_config=text_config
        )

        with pytest.raises(DistError, match=message):
            gemma4_plan(sources)

    def test_it_refuses_a_checkpoint_without_a_text_config(self, tmp_path: Path) -> None:
        sources = _build(tmp_path)
        (sources.model / "config.json").write_text(json.dumps({"model_type": "gemma4"}))

        with pytest.raises(DistError, match="text_config"):
            gemma4_plan(sources)

    def test_it_copies_the_upstream_sampler_recommendation(self, gemma4_assembled) -> None:
        """MUST: 値を写経しない（ADR 0083 決定 7 — 出どころは上流の宣言そのもの）。"""
        _, manifest = gemma4_assembled
        assert _model(manifest)["pipelineConfig"]["sampler"] == {
            "temperature": fixture.GENERATION_CONFIG["temperature"],
            "topK": fixture.GENERATION_CONFIG["top_k"],
            "topP": fixture.GENERATION_CONFIG["top_p"],
        }

    def test_it_declares_the_runtime_knobs(self, gemma4_assembled) -> None:
        _, manifest = gemma4_assembled
        config = _model(manifest)["pipelineConfig"]
        assert config["chunkLength"] == SMALL_CHUNK
        assert config["capacity"] == SMALL_CAPACITY

    def test_it_refuses_a_capacity_beyond_the_model_limit(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """実物のノブ（容量 4096）× 合成の位置上限（37）— 長い会話でだけ落ちる形を焼かない。"""
        monkeypatch.setattr(gemma4_distribution, "GEMMA4_CHUNK_LENGTH", GEMMA4_CHUNK_LENGTH)
        monkeypatch.setattr(gemma4_distribution, "GEMMA4_CAPACITY", GEMMA4_CAPACITY)
        sources = _build(tmp_path)
        with pytest.raises(DistError, match="モデルの位置上限"):
            gemma4_plan(sources)

    def test_it_refuses_a_capacity_below_one_chunk(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(gemma4_distribution, "GEMMA4_CHUNK_LENGTH", 4)
        monkeypatch.setattr(gemma4_distribution, "GEMMA4_CAPACITY", 2)
        sources = _build(tmp_path)
        with pytest.raises(DistError, match="1 chunk すら入らない"):
            gemma4_plan(sources)

    @pytest.mark.parametrize("chunk", [1, GEMMA4_MAX_CHUNK_LENGTH + 1])
    def test_it_refuses_a_chunk_length_outside_the_traced_range(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, chunk: int
    ) -> None:
        """記号 `M` は trace 時に `[2, SYM_MAX]` で宣言される — その外は宣言できない。"""
        monkeypatch.setattr(gemma4_distribution, "GEMMA4_CHUNK_LENGTH", chunk)
        monkeypatch.setattr(gemma4_distribution, "GEMMA4_CAPACITY", 2048)
        sources = _build(tmp_path)
        with pytest.raises(DistError, match="chunkLength"):
            gemma4_plan(sources)

    def test_the_declared_knobs_are_inside_the_shipped_bounds(self) -> None:
        """実物の宣言（768 / 4096）そのものが両方の上限の内側にあること。"""
        assert 2 <= GEMMA4_CHUNK_LENGTH <= GEMMA4_MAX_CHUNK_LENGTH
        assert GEMMA4_CHUNK_LENGTH <= GEMMA4_CAPACITY

    def test_it_declares_the_traced_chunk_bound(self, gemma4_assembled) -> None:
        """MUST: 記号 `M` の trace 上限を配布形が宣言する（読み手は資産から導けない）。

        IR の `symbols` は名前の列だけで上限を持たないので、`chunkLength` を上書きした呼び手が
        trace 範囲の外へ出たことは資産側では検出できない。焼く側が知っている唯一の数を宣言へ
        載せることで、TS 側（`parseGemma4PipelineConfig` / `assertChunkLength`）が門にできる。

        既定（{@link SMALL_CHUNK} へ寄せてある）とは**別の事実**なので、上限は monkeypatch の
        影響を受けず実物の値のまま出る。
        """
        _, manifest = gemma4_assembled
        assert _model(manifest)["pipelineConfig"]["maxChunkLength"] == GEMMA4_MAX_CHUNK_LENGTH

    def test_the_chunk_bound_mirrors_the_export_script(self) -> None:
        """MUST: 記号 `M` の上限は焼く側（`gemma4.export.SYM_MAX`）と同じ数。

        配布 recipe は torch を読まない（既定 sync の CI job で collection ごと落とさない）
        ので写しを持つ。写しが古びると「trace の外の chunk 長を宣言した配布形」が通る。

        NOTE: 見えるのは 2 つの**既定値**が一致することだけ。系列を `--sym-max 640` などで
        組み直しても（513 未満は export 自体が拒む）、使った値はどこにも残らないのでこの門は
        緑のままになる。
        """
        pytest.importorskip("torch")
        from gemma4.export import SYM_MAX

        assert GEMMA4_MAX_CHUNK_LENGTH == SYM_MAX


class TestGemma4Graph:
    """製品グラフの形 — 入力の並びと、PLE 索引との噛み合い。"""

    def test_it_refuses_a_graph_with_other_inputs(self, tmp_path: Path) -> None:
        # 入力の綴りが違うコンテナ（`export_decode.py` の token-only 形などの取り違え）。
        wrong = ir_container(
            mark="other",
            storage="i4",
            inputs=(("input_ids", [1, "M"]),),
            outputs=([1, 1, fixture.VOCAB],),
        )
        sources = _build(tmp_path)
        replace_component(sources.product / "model.safetensors", wrong)
        with pytest.raises(DistError, match="グラフ入力が"):
            gemma4_plan(sources)

    def test_it_refuses_a_graph_whose_exit_is_not_the_last_row(self, tmp_path: Path) -> None:
        sources = _sources(tmp_path)
        fixture.write_series(
            sources.product,
            sources.tokenizer,
            sources.model,
            # 全語彙 logits（`[1, M, V]` 相当の 3 軸だが先頭 2 軸が [1, 1] でない）。
            container=fixture.product_container(vocab=fixture.VOCAB),
        )
        write_component(
            sources.product / "model.safetensors",
            ir_container(
                mark="rows",
                storage="i4",
                inputs=(("input_ids", [1, "M"]),),
                outputs=([1, 4, fixture.VOCAB],),
            ),
        )
        with pytest.raises(DistError, match=r"\[1, 1, V\] でない"):
            gemma4_plan(sources)

    def test_it_refuses_rope_inputs_whose_width_is_not_the_declared_head_dim(
        self, tmp_path: Path
    ) -> None:
        """宣言（config 由来）とグラフ（コンテナ由来）は別々に動く — 噛み合わせはここだけ。"""
        sources = _sources(tmp_path)
        fixture.write_series(
            sources.product,
            sources.tokenizer,
            sources.model,
            container=fixture.product_container(
                head_dims={FULL_ATTENTION: fixture.FULL_HEAD_DIM + 2}
            ),
        )
        with pytest.raises(DistError, match=gemma4_rope_input_name(FULL_ATTENTION, "cos")):
            gemma4_plan(sources)

    def test_it_refuses_a_graph_that_still_bakes_the_rope_tables(self, tmp_path: Path) -> None:
        """派生入力も表も両方持つ形（外に出し切れていない世代）を落とす。"""
        sources = _sources(tmp_path)
        fixture.write_series(
            sources.product,
            sources.tokenizer,
            sources.model,
            container=fixture.product_container(baked_rope=True),
        )
        with pytest.raises(DistError, match="焼き込んだ RoPE 表"):
            gemma4_plan(sources)

    def test_it_refuses_a_graph_without_a_free_capacity_symbol(self, tmp_path: Path) -> None:
        """容量記号は states にだけ現れる 1 本（TS 側 `capacitySymbolOf` の鏡像）。"""
        sources = _sources(tmp_path)
        fixture.write_series(
            sources.product,
            sources.tokenizer,
            sources.model,
            container=fixture.product_container(free_symbol=False),
        )
        with pytest.raises(DistError, match="入力 shape から決まらない記号"):
            gemma4_plan(sources)

    @pytest.mark.parametrize(("field", "axis"), [("layers", 2), ("dim", 3)])
    def test_it_refuses_a_sidecar_shaped_for_another_graph(
        self, tmp_path: Path, field: str, axis: int
    ) -> None:
        index = fixture.ple_index([(0, 4), (4, fixture.VOCAB)])
        index[field] = int(index[field]) + 1
        sources = _sources(tmp_path)
        fixture.write_series(sources.product, sources.tokenizer, sources.model, index=index)
        with pytest.raises(DistError, match=f"軸 {axis}"):
            gemma4_plan(sources)


class TestGemma4Sidecar:
    """PLE sidecar — 索引の形と、shard の現物が名乗る世代。"""

    def test_it_refuses_a_sidecar_whose_rows_are_not_the_vocabulary(self, tmp_path: Path) -> None:
        """MUST: 行数が語彙数と違えば**別 token の有効な行**を引く（ADR 0085 決定 5）。"""
        sources = _sources(tmp_path)
        fixture.write_series(
            sources.product,
            sources.tokenizer,
            sources.model,
            index=fixture.ple_index([(0, 4)], tokens=4),
        )
        with pytest.raises(DistError, match="製品グラフの語彙数"):
            gemma4_plan(sources)

    @pytest.mark.parametrize(
        ("ranges", "message"),
        [
            ([(0, 3), (4, 6)], "連続しない"),
            ([(0, 4), (4, 4)], "が空"),
            ([(0, 4)], "shard の合計"),
        ],
    )
    def test_it_refuses_an_index_that_is_not_a_partition(
        self, tmp_path: Path, ranges: list[tuple[int, int]], message: str
    ) -> None:
        sources = _sources(tmp_path)
        index = fixture.ple_index(ranges)
        with pytest.raises(DistError, match=message):
            fixture.write_series(sources.product, sources.tokenizer, sources.model, index=index)
            gemma4_plan(sources)

    def test_it_refuses_an_index_with_unknown_keys(self, tmp_path: Path) -> None:
        index = fixture.ple_index([(0, 4), (4, fixture.VOCAB)])
        index["strategy"] = "token-major"
        sources = _sources(tmp_path)
        fixture.write_series(sources.product, sources.tokenizer, sources.model, index=index)
        with pytest.raises(DistError, match="未知キー"):
            gemma4_plan(sources)

    def test_it_refuses_a_shard_that_names_another_generation(self, tmp_path: Path) -> None:
        """索引だけ差し替えた組み合わせは**形も dtype も合う**まま別 token の行を引く。"""
        index = fixture.ple_index([(0, 4), (4, fixture.VOCAB)])
        sources = _sources(tmp_path)
        fixture.write_series(
            sources.product,
            sources.tokenizer,
            sources.model,
            index=index,
            shard_metadata={
                0: {
                    "schema": fixture.PLE_SCHEMA,
                    "tokens": index["tokens"],
                    "layers": index["layers"],
                    "dim": index["dim"],
                    "embedScale": index["embedScale"],
                    "start": 0,
                    # 範囲だけがずれた写し（テンソルの形は索引どおり）。
                    "stop": 3,
                }
            },
        )
        with pytest.raises(DistError, match="索引と食い違う"):
            gemma4_plan(sources)

    def test_it_refuses_a_missing_shard(self, tmp_path: Path) -> None:
        sources = _build(tmp_path)
        (sources.product / "ple-00001-of-00002.safetensors").unlink()
        with pytest.raises(DistError):
            gemma4_plan(sources)


class TestGemma4Tokenizer:
    def test_it_refuses_a_tokenizer_for_another_vocabulary(self, tmp_path: Path) -> None:
        sources = _sources(tmp_path)
        fixture.write_series(
            sources.product,
            sources.tokenizer,
            sources.model,
            tokenizer=fixture.tokenizer_asset(vocab=fixture.VOCAB + 1),
        )
        with pytest.raises(DistError, match="製品グラフの語彙数"):
            gemma4_plan(sources)

    def test_it_refuses_a_raw_upstream_tokenizer_json(self, tmp_path: Path) -> None:
        """上流の 32MB の `tokenizer.json` を置いた取り違え（compile 台本を通していない）。"""
        sources = _sources(tmp_path)
        fixture.write_series(
            sources.product,
            sources.tokenizer,
            sources.model,
            tokenizer={"version": "1.0", "model": {"type": "BPE"}},
        )
        with pytest.raises(DistError, match="format が"):
            gemma4_plan(sources)


class TestGemma4Sampler:
    @pytest.mark.parametrize("dropped", ["temperature", "top_k", "top_p"])
    def test_it_refuses_a_checkpoint_without_the_recommendation(
        self, tmp_path: Path, dropped: str
    ) -> None:
        config = {key: value for key, value in fixture.GENERATION_CONFIG.items() if key != dropped}
        sources = _sources(tmp_path)
        fixture.write_series(
            sources.product, sources.tokenizer, sources.model, generation_config=config
        )
        with pytest.raises(DistError, match=f"{dropped} が無い"):
            gemma4_plan(sources)

    @pytest.mark.parametrize(
        ("key", "value", "message"),
        [
            ("temperature", -1.0, "temperature"),
            ("top_k", 0, "top_k"),
            ("top_p", 1.5, "top_p"),
        ],
    )
    def test_it_refuses_a_recommendation_outside_the_accepted_range(
        self, tmp_path: Path, key: str, value: Any, message: str
    ) -> None:
        """受理集合は TS 側 `parseGemma4PipelineConfig` と同じ — 配ってから落ちる形にしない。"""
        config = {**fixture.GENERATION_CONFIG, key: value}
        sources = _sources(tmp_path)
        fixture.write_series(
            sources.product, sources.tokenizer, sources.model, generation_config=config
        )
        with pytest.raises(DistError, match=message):
            gemma4_plan(sources)


class TestGemma4Storage:
    """系列 root の取り違え — 数値の門では原理的に検出できないので、ここが唯一の検出器。"""

    @pytest.mark.parametrize(("storage", "message"), [("f32", "I4 が無い"), ("f16", "I4 が無い")])
    def test_it_refuses_a_container_without_the_packed_int4_weights(
        self, tmp_path: Path, storage: str, message: str
    ) -> None:
        sources = _build(tmp_path)
        replace_component(
            sources.product / "model.safetensors", ir_container(mark="plain", storage=storage)
        )
        with pytest.raises(DistError, match=message):
            gemma4_plan(sources)

    def test_it_refuses_a_container_carrying_half_precision(self, tmp_path: Path) -> None:
        """F16 の混入は「別 family の系列 root を指した」印にしかならない。"""
        sources = _build(tmp_path)
        replace_component(
            sources.product / "model.safetensors", ir_container(mark="half", storage="f16")
        )
        with pytest.raises(DistError):
            gemma4_plan(sources)


class TestGemma4Naming:
    def test_the_series_and_repo_names_come_from_one_word(self) -> None:
        assert gemma4_series_name("e2b", "product") == "gemma4-e2b-product"
        assert gemma4_series_name("e2b", "tokenizer") == "gemma4-e2b-tokenizer"
        assert gemma4_repo_name("e2b") == "karume-gemma4"
        # 家族 1 リポ（ADR 0092 決定 1）— どのモデルを組んでも行き先は 1 つ。
        assert gemma4_repo_name("e4b") == gemma4_repo_name("e2b")

    def test_the_sources_follow_the_repo_topology(self, tmp_path: Path) -> None:
        sources = gemma4_sources(tmp_path, GEMMA4_DEFAULT_MODEL)
        assert sources.product == tmp_path / "gemma4-e2b-product"
        assert sources.tokenizer == tmp_path / "gemma4-e2b-tokenizer"
        # チェックポイントは帰属表から導く（2 つ目の表を持たない）。
        assert sources.model.name == GEMMA4_UPSTREAM[GEMMA4_DEFAULT_MODEL].split("/", 1)[1]

    def test_it_refuses_a_model_outside_the_attribution_table(self, tmp_path: Path) -> None:
        with pytest.raises(DistError, match="知らない"):
            gemma4_sources(tmp_path, "e4b")


class TestGemma4Card:
    def test_the_profile_is_the_only_one(self) -> None:
        profiles = gemma4_distribution.PIPELINE.card_profiles
        assert list(profiles) == ["gemma4"]
        assert resolve_card_renderer(gemma4_distribution.PIPELINE, None) is profiles["gemma4"]

    def test_it_renders_the_attribution_and_the_declared_defaults(self, gemma4_assembled) -> None:
        _, manifest = gemma4_assembled
        card = render_gemma4_model_card(manifest, "hdae/karume-gemma4")
        assert "license: apache-2.0" in card
        assert GEMMA4_UPSTREAM[GEMMA4_DEFAULT_MODEL] in card
        # 使われ方は上流カードへ誘導する（2026-09-01 のライセンス方針）。
        assert "upstream model card" in card
        assert "LICENSE.md" in card and "NOTICE.md" in card
        # 数は manifest から導出する（推奨サンプラも位置上限も本文に出る）。
        assert str(fixture.MAX_POSITION) in card
        assert str(fixture.GENERATION_CONFIG["top_k"]) in card

    def test_it_refuses_to_describe_another_pipeline(self, gemma4_assembled) -> None:
        _, manifest = gemma4_assembled
        other = json.loads(json.dumps(manifest))
        other["models"][GEMMA4_DEFAULT_MODEL]["pipeline"] = "anima/1"
        with pytest.raises(ValueError, match="gemma4/1"):
            render_gemma4_model_card(other, "hdae/karume-gemma4")

    def test_it_refuses_a_model_outside_the_attribution_table(self, gemma4_assembled) -> None:
        _, manifest = gemma4_assembled
        other = json.loads(json.dumps(manifest))
        other["models"]["e4b"] = other["models"].pop(GEMMA4_DEFAULT_MODEL)
        other["defaultModel"] = "e4b"
        with pytest.raises(ValueError, match="帰属表に無い"):
            render_gemma4_model_card(other, "hdae/karume-gemma4")


class TestGemma4LegalText:
    def test_it_ships_the_license_text_byte_identical(self, gemma4_assembled) -> None:
        """§4(a) — 提供するのは**このライセンスのコピー**（要約でも整形でもない）。

        原本は `_shared/licenses/apache_license_2_0.txt`。組み立ての経路のどこかで整形や
        改行変換が入ると 1 バイト動くが、散文としては妥当なままなので他の門は素通りする。
        """
        out_dir, _ = gemma4_assembled
        assert (out_dir / "LICENSE.md").read_bytes() == APACHE_LICENSE_2_0_PATH.read_bytes()

    def test_it_ships_the_modification_notice(self, gemma4_assembled) -> None:
        """Apache 2.0 §4(b) の改変告知。"""
        out_dir, _ = gemma4_assembled
        notice = (out_dir / "NOTICE.md").read_text(encoding="utf-8")
        assert "gemma-4-E2B-it" in notice
        assert "int4" in notice
