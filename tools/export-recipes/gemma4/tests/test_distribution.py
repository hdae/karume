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
from collections.abc import Callable, Iterable, Mapping
from functools import partial
from pathlib import Path
from typing import Any

import pytest
from container_series import placed_paths, replace_component
from ir_fixtures import ir_container

from _shared.container_read import read_asset, read_asset_declarations
from _shared.licenses import APACHE_LICENSE_2_0_PATH
from gemma4 import distribution as gemma4_distribution
from gemma4.card import GEMMA4_UPSTREAM, render_gemma4_model_card
from gemma4.chat import PLAIN_ROLES
from gemma4.distribution import (
    GEMMA4_CAPACITY,
    GEMMA4_CHUNK_LENGTH,
    GEMMA4_DEFAULT_MODEL,
    GEMMA4_DRAFTER_ROLE,
    GEMMA4_MAX_CHUNK_LENGTH,
    GEMMA4_OUTPUT_PATHS,
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
from karume.container import AssetInput, container_parts
from karume.dist import (
    MANIFEST_FILENAME,
    DistError,
    assemble_family,
    resolve_card_renderer,
    verify_dist,
)
from karume.ple import PLE_INDEX_ASSET, PLE_INDEX_ROLE

#: 役割名 → part 本数（PLE の資産は専用 part を取るので `model` だけ多い）。
_PART_TOTALS: Mapping[str, int] = {GEMMA4_ROLE: len(fixture.product_container())}

#: PLE の寸法（索引を曲げるときの元値）。
_PLE_DIMS: Mapping[str, int] = {"layers": fixture.LAYERS, "dim": fixture.DIM}


def _shift_first_block(index: dict[str, Any]) -> None:
    """先頭 block の末尾を 1 行縮める（範囲が連続しない索引）。"""
    for key in ("values", "scales"):
        index[key]["blocks"][0]["stop"] -= 1


def _rename_first_block(index: dict[str, Any]) -> None:
    """先頭 block の資産名だけを容器に無い綴りへ差し替える。"""
    index["values"]["blocks"][0]["asset"] = "ple.values.absent"


def _container_with_ple(
    *,
    tokens: int = fixture.VOCAB,
    layers: int = fixture.LAYERS,
    dim: int = fixture.DIM,
    bend: Callable[[dict[str, Any]], None] | None = None,
) -> list[bytes]:
    """製品コンテナ（PLE の索引だけを 1 箇所曲げられる形）。

    索引は資産 `ple_index` の payload なので、曲げるのは**その JSON だけ** — 資産の block 列は
    そのままにしておくと「索引だけ古い組み合わせ」がそのまま作れる。
    """
    assets = dict(
        fixture.ple_container_assets(
            tokens=tokens, layers=layers, dim=dim, block_bytes=_PLE_BLOCK_BYTES
        )
    )
    if bend is not None:
        index = json.loads(bytes(assets[PLE_INDEX_ASSET].payload))
        bend(index)
        payload = json.dumps(index, ensure_ascii=False).encode("utf-8")
        assets[PLE_INDEX_ASSET] = AssetInput(PLE_INDEX_ROLE, len(payload), payload)
    return fixture.product_container(assets=assets)


#: tiny な PLE を 2 block 以上へ割る block 上限（block 跨ぎを 1 度は踏む）。
_PLE_BLOCK_BYTES = fixture.LAYERS * fixture.DIM * (fixture.VOCAB // 2)

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
        drafter=root / "series" / "gemma4-e2b-drafter",
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


def _assert_notice_names(notice: str, repos: Iterable[str]) -> None:
    """改変告知が上流チェックポイントを名指ししていること（org を落とした綴りで見る）。"""
    for repo in repos:
        checkpoint = repo.split("/", 1)[1]
        assert checkpoint in notice, f"改変告知が '{checkpoint}' を名指ししていない"


class TestGemma4Layout:
    def test_it_places_the_graph_sidecar_and_tokenizer_under_the_model_subtree(
        self, gemma4_assembled
    ) -> None:
        out_dir, _ = gemma4_assembled
        # PLE は `model` 容器の資産なので、配布形に独立したファイルとしては現れない。
        expected = _in_subtree(
            GEMMA4_DEFAULT_MODEL,
            placed_paths(GEMMA4_OUTPUT_PATHS, GEMMA4_WEIGHTS, _PART_TOTALS),
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

    def test_it_declares_two_graphs_and_the_sidecar_as_assets(self, gemma4_assembled) -> None:
        """weights は**製品グラフと drafter の 2 本**（ADR 0096 段 2）— drafter は assets ではない。

        並びまで見るのは、quant の `weights` 写像が weights の宣言順で埋まるため
        （`karume.dist.complete_quant_weights` の MUST）— 入れ替わると manifest の 2 節が
        別の順に並ぶ。
        """
        _, manifest = gemma4_assembled
        model = _model(manifest)
        assert model["pipeline"] == "gemma4/1"
        assert list(model["weights"]) == [GEMMA4_ROLE, GEMMA4_DRAFTER_ROLE]
        assert list(model["quants"]) == ["i4", "i4-gemvpar", "i4-fast"]
        assert model["defaultQuant"] == "i4-fast"
        assert model["quants"]["i4-gemvpar"]["weights"] == model["quants"]["i4"]["weights"]
        assert model["quants"]["i4-gemvpar"]["session"] == {"linearGemvReduce": "parallel"}
        assert model["quants"]["i4-fast"]["weights"] == model["quants"]["i4"]["weights"]
        assert model["quants"]["i4-fast"]["session"] == {
            "linearGemvReduce": "parallel",
            "fuseRmsNormAdd": True,
        }
        # 役割ごとに基底格納が違う（drafter は linear まで i8）ので、自動補完が 2 席とも
        # それぞれの唯一の dtype ラベルで埋める。
        assert model["quants"]["i4"]["weights"] == {GEMMA4_ROLE: "i4", GEMMA4_DRAFTER_ROLE: "i8"}
        assert model["quants"]["i4"]["session"] == {}

    def test_it_never_carries_the_drafter_goldens(self, gemma4_assembled) -> None:
        """drafter 系列の golden は検収専用 — `ple.probe` と同じく配布へは入らない。"""
        out_dir, _ = gemma4_assembled
        assert list(out_dir.rglob("drafter-golden.*")) == []

    def test_the_ple_blocks_live_inside_the_model_container(self, gemma4_assembled) -> None:
        """MUST: PLE は容器の資産（ADR 0109 決定 4）— manifest の `assets` には出ない。"""
        out_dir, manifest = gemma4_assembled
        model = _model(manifest)
        assert list(model["assets"]) == [GEMMA4_TOKENIZER_ROLE]

        placed = out_dir / model["weights"][GEMMA4_ROLE]["i4"]["container"]["parts"][0]["path"]
        index = json.loads(read_asset(placed, PLE_INDEX_ASSET))
        declared = read_asset_declarations(placed)
        blocks = [block["asset"] for block in index["values"]["blocks"]]
        assert blocks, "索引が block を 1 本も持たない"
        for name in blocks:
            assert name in declared

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

    def test_the_row_symbol_mirrors_the_export_script(self) -> None:
        """MUST: 出口の行数記号の綴りは焼く側（`export_product.ROW_SYMBOL`）と同じ。

        配布 recipe は torch を読まないので写しを持つ（`M` の上限と同じ理由）。綴りが割れると
        「行軸が記号でない資産」を落とす門（{@link gemma4_vocab_size}）が空振りする。
        """
        pytest.importorskip("torch")
        from gemma4.export_product import ROW_SYMBOL

        assert gemma4_distribution.GEMMA4_ROW_SYMBOL == ROW_SYMBOL


class TestGemma4Graph:
    """製品グラフの形 — 入力の並びと、PLE 索引との噛み合い。"""

    def test_it_refuses_a_graph_with_other_inputs(self, tmp_path: Path) -> None:
        # 入力の綴りが違うコンテナ（`export_decode.py` の token-only 形などの取り違え）。
        wrong = ir_container(
            mark="other",
            storage="i4",
            inputs=(("input_ids", [1, "M"]), ("last_row", [fixture.ROW_SYMBOL])),
            outputs=(
                [1, fixture.ROW_SYMBOL, fixture.VOCAB],
                [1, fixture.ROW_SYMBOL, fixture.HIDDEN],
            ),
            # PLE は同じ容器の資産なので、差し替えた容器にも載せる（索引の門で先に落ちない）。
            assets=fixture.ple_container_assets(block_bytes=_PLE_BLOCK_BYTES),
        )
        sources = _build(tmp_path)
        replace_component(sources.product / "model.krm", wrong)
        with pytest.raises(DistError, match="グラフ入力が"):
            gemma4_plan(sources)

    def test_it_refuses_a_graph_whose_exit_is_not_the_last_row(self, tmp_path: Path) -> None:
        sources = _sources(tmp_path)
        fixture.write_series(
            sources.product,
            sources.tokenizer,
            sources.model,
            # 行軸が記号でない logits（`[1, 4, V]` — 焼いた行数の資産）。
            container=fixture.product_container(vocab=fixture.VOCAB),
        )
        replace_component(
            sources.product / "model.krm",
            ir_container(
                mark="rows",
                storage="i4",
                inputs=(("input_ids", [1, "M"]),),
                outputs=([1, 4, fixture.VOCAB], [1, 4, fixture.HIDDEN]),
                assets=fixture.ple_container_assets(block_bytes=_PLE_BLOCK_BYTES),
            ),
        )
        with pytest.raises(DistError, match=r"\[1, R, \*\] でない"):
            gemma4_plan(sources)

    def test_it_refuses_a_graph_whose_outputs_are_swapped(self, tmp_path: Path) -> None:
        """MUST: logits と hidden は行軸まで同型 — 入れ替えは幅でしか捕まらない。

        取り違えたまま通すと、語彙数のつもりで hidden_size を読んだ manifest が組み上がる
        （PLE 索引・トークナイザとの相互照合がその数で回る）。
        """
        sources = _sources(tmp_path)
        fixture.write_series(
            sources.product,
            sources.tokenizer,
            sources.model,
            container=fixture.product_container(swap_outputs=True),
        )
        with pytest.raises(DistError, match="出力 1（hidden）の幅"):
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
    def test_it_refuses_a_ple_index_shaped_for_another_graph(
        self, tmp_path: Path, field: str, axis: int
    ) -> None:
        sources = _sources(tmp_path)
        fixture.write_series(
            sources.product,
            sources.tokenizer,
            sources.model,
            # 2 ずつ動かすのは、PLE の 1 行が 4 の倍数でなければ block を切れないため。
            container=_container_with_ple(**{field: int(_PLE_DIMS[field]) + 2}),
        )
        with pytest.raises(DistError, match=f"軸 {axis}"):
            gemma4_plan(sources)


class TestGemma4Ple:
    """PLE の索引（容器の資産 `ple_index`）の形と、資産宣言との噛み合わせ。"""

    def test_it_refuses_an_index_whose_rows_are_not_the_vocabulary(self, tmp_path: Path) -> None:
        """MUST: 行数が語彙数と違えば**別 token の有効な行**を引く（ADR 0085 決定 5）。"""
        sources = _sources(tmp_path)
        fixture.write_series(
            sources.product,
            sources.tokenizer,
            sources.model,
            container=_container_with_ple(tokens=fixture.VOCAB - 2),
        )
        with pytest.raises(DistError, match="製品グラフの語彙数"):
            gemma4_plan(sources)

    def test_it_refuses_an_index_that_is_not_a_partition(self, tmp_path: Path) -> None:
        """範囲が連続しない索引は「引けない id」か「2 本が同じ id」を作る（沈黙誤値）。"""
        sources = _sources(tmp_path)
        fixture.write_series(
            sources.product,
            sources.tokenizer,
            sources.model,
            container=_container_with_ple(bend=_shift_first_block),
        )
        with pytest.raises(DistError, match="連続しない"):
            gemma4_plan(sources)

    def test_it_refuses_an_index_with_unknown_keys(self, tmp_path: Path) -> None:
        sources = _sources(tmp_path)
        fixture.write_series(
            sources.product,
            sources.tokenizer,
            sources.model,
            container=_container_with_ple(
                bend=lambda index: index.__setitem__("strategy", "token-major")
            ),
        )
        with pytest.raises(DistError, match="未知キー"):
            gemma4_plan(sources)

    def test_it_refuses_an_index_that_names_an_asset_the_container_lacks(
        self, tmp_path: Path
    ) -> None:
        """索引だけ差し替えた組み合わせは**形も dtype も合う**まま別 token の行を引く。"""
        sources = _sources(tmp_path)
        fixture.write_series(
            sources.product,
            sources.tokenizer,
            sources.model,
            container=_container_with_ple(bend=_rename_first_block),
        )
        with pytest.raises(DistError, match="容器に無い"):
            gemma4_plan(sources)

    def test_it_refuses_a_missing_part(self, tmp_path: Path) -> None:
        """part 列の 1 本でも欠ければ落とす（容器として開けない）。"""
        sources = _build(tmp_path)
        container_parts(sources.product / "model.krm")[-1].unlink()
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

    @pytest.mark.parametrize(("storage", "message"), [("f32", "i4 が無い"), ("f16", "i4 が無い")])
    def test_it_refuses_a_container_without_the_packed_int4_weights(
        self, tmp_path: Path, storage: str, message: str
    ) -> None:
        sources = _build(tmp_path)
        replace_component(
            sources.product / "model.krm", ir_container(mark="plain", storage=storage)
        )
        with pytest.raises(DistError, match=message):
            gemma4_plan(sources)

    def test_it_refuses_a_container_carrying_half_precision(self, tmp_path: Path) -> None:
        """f16 の混入は「別 family の系列 root を指した」印にしかならない。"""
        sources = _build(tmp_path)
        replace_component(sources.product / "model.krm", ir_container(mark="half", storage="f16"))
        with pytest.raises(DistError):
            gemma4_plan(sources)

    def test_it_refuses_a_drafter_whose_linear_weights_fell_to_int4(self, tmp_path: Path) -> None:
        """drafter に i4 が在れば落とす — 受理率が 1 〜 3 割落ちるだけの資産の唯一の検出器。

        出力ヘッドは i8 のままなので存在検査（i8 が在る）では素通りし、shape も manifest も
        正しいまま配れてしまう（`gemma4/export_drafter.py` の 2026-09-08 実測）。
        """
        sources = _build(tmp_path, drafter_bytes=fixture.drafter_container(storage="i4"))

        with pytest.raises(DistError, match="i4 がある"):
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

    def test_the_card_counts_the_container_assets_as_host_read(self, tmp_path: Path) -> None:
        """PLE は容器の資産へ移った（ADR 0109 決定 4）— Download の内訳注記がそれを数える。

        `karume/4` では PLE が独立したファイル（manifest の `assets`）だったので、path 集合で
        数えれば足りた。容器の中へ入った今は宣言が `container` 側にしか無いので、組み立てが
        引いて渡す（{@link karume.dist.container_asset_bytes}）— 数え落とすと「Download の
        大半が host 読みの表」という、この注記が存在する唯一の実例で注記が黙って消える。
        """
        sources = _build(tmp_path)
        out_dir = tmp_path / "models" / gemma4_repo_name(GEMMA4_DEFAULT_MODEL)
        assemble_family(
            [gemma4_plan(sources, GEMMA4_DEFAULT_MODEL)],
            out_dir,
            GEMMA4_DEFAULT_MODEL,
            render_card=partial(render_gemma4_model_card, repo="hdae/karume-gemma4"),
            root_files=gemma4_distribution.PIPELINE.root_files,
        )

        card = (out_dir / "README.md").read_text(encoding="utf-8")

        assert "of assets, read on the host" in card

    def test_the_note_is_absent_when_the_container_assets_are_not_counted(
        self, gemma4_assembled
    ) -> None:
        """恒真化の門 — 独立ファイルの assets（tokenizer）だけでは注記の条件を満たさない。"""
        _, manifest = gemma4_assembled

        card = render_gemma4_model_card(manifest, "hdae/karume-gemma4")

        assert "of assets, read on the host" not in card

    def test_it_names_every_accepted_chat_role(self, gemma4_assembled) -> None:
        """カードが語る role は受理集合（`gemma4.chat.PLAIN_ROLES` — TS 側と同じ射程）と一致する。

        1 つでも落ちると、公開カードの読み手はその role を使えないと誤読する。
        """
        _, manifest = gemma4_assembled
        card = render_gemma4_model_card(manifest, "hdae/karume-gemma4")

        roles = " / ".join(f"`{role}`" for role in PLAIN_ROLES)
        assert f"Messages are plain {roles} turns" in card

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
        # session の boolean は manifest と同じ JSON 表記で綴る — Python の `True` が出ると
        # 読み手がそのまま貼った TypeScript と綴りが食い違う。
        fast = next(line for line in card.splitlines() if line.startswith("| `i4-fast`"))
        assert "`fuseRmsNormAdd` = `true`" in fast
        assert "True" not in fast

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

    def test_neither_the_title_nor_the_overview_names_a_model(self, gemma4_assembled) -> None:
        """家族 1 リポ（ADR 0092 決定 1）— 名乗る綴りは manifest から引く。

        モデルの綴りを題や概要へ焼くと、2 つ目のモデルを足した日に「E2B と名乗りながら
        E4B も配る」カードが黙って出る（`_gemma4_metadata` の帰属表の門はモデルが**既知**なら
        素通りする）。
        """
        _, manifest = gemma4_assembled
        card = render_gemma4_model_card(manifest, "hdae/karume-gemma4")
        title = next(line for line in card.splitlines() if line.startswith("# "))
        overview = card.split("## What is this", 1)[1].split("\n## ", 1)[0]

        assert "E2B" not in title
        # 概要が名乗る綴りは manifest から引いたもの（焼き込みではない）。
        assert GEMMA4_UPSTREAM[GEMMA4_DEFAULT_MODEL] in overview

    def test_it_names_the_device_limits_a_quant_declares(self, gemma4_assembled) -> None:
        """`requiredLimits` は「載らない device を先に知る」ための欄なので本文に出す。

        合成の系列は寸法が小さく core が欄を焼かないので、欄そのものを差し込んで見る。
        """
        _, manifest = gemma4_assembled
        other = json.loads(json.dumps(manifest))
        other["models"][GEMMA4_DEFAULT_MODEL]["quants"]["i4"]["requiredLimits"] = {
            "maxStorageBufferBindingSize": 134_217_728,
        }
        card = render_gemma4_model_card(other, "hdae/karume-gemma4")

        assert "device limits" in card
        assert "maxStorageBufferBindingSize" in card
        assert "134,217,728" in card

    @staticmethod
    def _device_limits_line(manifest: Mapping[str, Any], limits_by_quant: Callable) -> str:
        other = json.loads(json.dumps(manifest))
        quants = other["models"][GEMMA4_DEFAULT_MODEL]["quants"]
        for quant_name, quant in quants.items():
            quant["requiredLimits"] = limits_by_quant(quant_name)
        card = render_gemma4_model_card(other, "hdae/karume-gemma4")
        return next(line for line in card.splitlines() if "**device limits**" in line)

    def test_a_limit_every_quant_shares_is_named_once(self, gemma4_assembled) -> None:
        """同じ重みの席が同じ上限を持つとき、上限名は 1 回ずつ（quant ごとに繰り返さない）。"""
        _, manifest = gemma4_assembled
        assert len(_model(manifest)["quants"]) >= 2
        line = self._device_limits_line(
            manifest,
            lambda _quant: {
                "maxBufferSize": 402_653_184,
                "maxStorageBufferBindingSize": 402_653_184,
            },
        )

        assert line.count("`maxBufferSize`") == 1
        assert line.count("`maxStorageBufferBindingSize`") == 1
        assert line.count("402,653,184") == 2
        assert " for `" not in line

    def test_a_limit_that_differs_between_quants_names_each_quant(self, gemma4_assembled) -> None:
        """値が割れたら値ごとに quant 名を添える（畳んだ 1 値を全席の上限と読ませない）。"""
        _, manifest = gemma4_assembled
        line = self._device_limits_line(
            manifest,
            lambda quant: {"maxBufferSize": 536_870_912 if quant == "i4" else 402_653_184},
        )
        others = [name for name in _model(manifest)["quants"] if name != "i4"]

        assert line.count("`maxBufferSize`") == 1
        assert "≥ 536,870,912 B for `i4`" in line
        assert f"≥ 402,653,184 B for {', '.join(f'`{name}`' for name in others)}" in line

    def test_it_omits_the_device_limits_line_when_no_quant_declares_one(
        self, gemma4_assembled
    ) -> None:
        """逆枝 — 欄が無いときは節ごと消える（情報が黙って落ちる形の記録）。"""
        _, manifest = gemma4_assembled
        assert not _model(manifest)["quants"]["i4"].get("requiredLimits")

        card = render_gemma4_model_card(manifest, "hdae/karume-gemma4")

        assert "device limits" not in card


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

    def test_the_notice_names_every_model_the_manifest_carries(self, gemma4_assembled) -> None:
        """§4(b) の告知は「この配布形が実際に再配布した上流」を 1 つ残らず名指しすること。

        告知は `Pipeline.root_files` の席（core の型は `Mapping[str, str]`）なので manifest を
        見て組めない — 焼き込んだ散文と実際に配ったモデルの対応は、ここでしか見られない。
        """
        out_dir, manifest = gemma4_assembled
        notice = (out_dir / "NOTICE.md").read_text(encoding="utf-8")

        _assert_notice_names(notice, [GEMMA4_UPSTREAM[name] for name in manifest["models"]])

    def test_the_notice_covers_the_whole_attribution_table(self) -> None:
        """帰属表へモデルを足した瞬間に赤くする門（`GEMMA4_NOTICE_MARKDOWN` の MUST）。

        散文は手で書くほかない（法的テキストの文面レビューは人が読む前提）ので、忘れを
        組み立て時ではなくここで受ける。
        """
        _assert_notice_names(gemma4_distribution.GEMMA4_NOTICE_MARKDOWN, GEMMA4_UPSTREAM.values())

    def test_a_model_the_notice_does_not_mention_is_caught(self, gemma4_assembled) -> None:
        """恒真でないことの裏取り — 2 件目を足した組で上の 2 本が落ちる。"""
        out_dir, _ = gemma4_assembled
        notice = (out_dir / "NOTICE.md").read_text(encoding="utf-8")

        with pytest.raises(AssertionError, match="gemma-4-E4B-it"):
            _assert_notice_names(notice, [*GEMMA4_UPSTREAM.values(), "google/gemma-4-E4B-it"])
