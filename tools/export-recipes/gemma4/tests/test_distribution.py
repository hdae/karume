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

import hashlib
import json
from collections.abc import Callable, Iterable, Mapping
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import Any

import pytest
from container_series import placed_paths, replace_component
from ir_fixtures import ir_container

from _shared.container_read import read_asset, read_asset_declarations
from _shared.licenses import APACHE_LICENSE_2_0_PATH
from gemma4 import card as gemma4_card
from gemma4 import chat
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
from karume.container import AssetInput, Provenance, container_parts
from karume.dist import (
    MANIFEST_FILENAME,
    DistError,
    assemble_family,
    resolve_card_renderer,
    verify_dist,
)
from karume.ple import PLE_INDEX_ASSET, PLE_INDEX_ROLE, PLE_ROLES

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

#: 帰属表に**無い**モデル名とその上流の綴り（未知のモデルを落とす門の入力）。E4B が帰属表へ
#: 入ったので、門の入力は実在しない綴りにしてある。
UNLISTED_MODEL = "unlisted"
UNLISTED_CHECKPOINT = "gemma-4-unlisted-it"

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
        upstream_tokenizer=root / "inputs" / "gemma-4-E2B-it",
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
        assert model["quants"]["i4-gemvpar"]["session"] == {
            "linearGemvReduce": "parallel",
            "stateAttentionReduce": "parallel",
        }
        assert model["quants"]["i4-fast"]["weights"] == model["quants"]["i4"]["weights"]
        assert model["quants"]["i4-fast"]["session"] == {
            "linearGemvReduce": "parallel",
            "fuseRmsNormAdd": True,
            "stateAttentionReduce": "parallel-fused",
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
        載せることで、TS 側（`gemma.parsePipelineConfig` / `assertChunkLength`）が門にできる。

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
        """受理集合は TS 側 `gemma.parsePipelineConfig` と同じ — 配ってから落ちる形にしない。"""
        config = {**fixture.GENERATION_CONFIG, key: value}
        sources = _sources(tmp_path)
        fixture.write_series(
            sources.product, sources.tokenizer, sources.model, generation_config=config
        )
        with pytest.raises(DistError, match=message):
            gemma4_plan(sources)

    def test_it_refuses_a_generation_default_it_neither_carries_nor_lists(
        self, tmp_path: Path
    ) -> None:
        """MUST: a key upstream adds surfaces as a decision instead of being dropped."""
        config = {**fixture.GENERATION_CONFIG, "repetition_penalty": 1.1}
        sources = _sources(tmp_path)
        fixture.write_series(
            sources.product, sources.tokenizer, sources.model, generation_config=config
        )

        with pytest.raises(DistError, match="repetition_penalty"):
            gemma4_plan(sources)

    def test_every_upstream_key_it_does_not_carry_has_a_reason(self) -> None:
        upstream = {
            "bos_token_id",
            "do_sample",
            "eos_token_id",
            "pad_token_id",
            "suppress_tokens",
            "temperature",
            "top_k",
            "top_p",
            "transformers_version",
        }
        carried = {source for source, _ in gemma4_distribution.GEMMA4_SAMPLER_FIELDS}

        assert (
            carried
            | {gemma4_distribution.GEMMA4_SUPPRESS_TOKENS_KEY}
            | set(gemma4_distribution.GEMMA4_GENERATION_NOT_CARRIED)
            == upstream
        )
        assert all(gemma4_distribution.GEMMA4_GENERATION_NOT_CARRIED.values())

    def test_an_e2b_manifest_has_no_suppressed_ids(self, gemma4_assembled) -> None:
        _, manifest = gemma4_assembled

        assert "suppressTokens" not in _model(manifest)["pipelineConfig"]

    @pytest.mark.parametrize(
        ("value", "message"),
        [
            ([], "non-empty list of integers"),
            ([1, True], "non-empty list of integers"),
            ("1", "non-empty list of integers"),
            ([1, 1], "repeats an id"),
            ([-1], "negative id"),
        ],
    )
    def test_it_refuses_malformed_suppressed_ids(
        self, tmp_path: Path, value: Any, message: str
    ) -> None:
        sources = _sources(tmp_path)
        fixture.write_series(
            sources.product,
            sources.tokenizer,
            sources.model,
            generation_config={**fixture.GENERATION_CONFIG, "suppress_tokens": value},
        )

        with pytest.raises(DistError, match=message):
            gemma4_plan(sources)

    def test_the_sampler_alone_refuses_a_checkpoint_with_suppressed_ids(
        self, tmp_path: Path
    ) -> None:
        """The QAT family's reader: its configuration has no `suppressTokens`, so it must not drop
        them silently."""
        (tmp_path / "generation_config.json").write_text(
            json.dumps({**fixture.GENERATION_CONFIG, "suppress_tokens": [1]}), encoding="utf-8"
        )

        with pytest.raises(DistError, match="cannot carry"):
            gemma4_distribution.gemma4_sampler(tmp_path)


class TestGemma4Storage:
    """系列 root の取り違え — 数値の門では原理的に検出できないので、ここが唯一の検出器。"""

    @pytest.mark.parametrize(("storage", "message"), [("f32", "has no i4"), ("f16", "has no i4")])
    def test_it_refuses_a_container_without_the_packed_int4_weights(
        self, tmp_path: Path, storage: str, message: str
    ) -> None:
        sources = _build(tmp_path)
        replace_component(
            sources.product / "model.krm", ir_container(mark="plain", storage=storage)
        )
        with pytest.raises(DistError, match=message):
            gemma4_plan(sources)

    def test_it_refuses_an_e2b_container_whose_embedding_is_not_int8(self, tmp_path: Path) -> None:
        """E2B / E4B keep the int8 embedding: an all-int4 container (12B's storage) is refused."""
        sources = _build(tmp_path, container=fixture.product_container(embedding_storage="i4"))

        with pytest.raises(DistError, match="has no i8"):
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

        with pytest.raises(DistError, match=r"has (\S+ / )*i4 \(actual"):
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
            gemma4_sources(tmp_path, UNLISTED_MODEL)


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
        other["models"][UNLISTED_MODEL] = other["models"].pop(GEMMA4_DEFAULT_MODEL)
        other["defaultModel"] = UNLISTED_MODEL
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

    def test_the_notice_states_the_int8_embedding_only_for_e2b_and_e4b(self) -> None:
        """12B's embedding keeps its 4-bit grid; a notice saying every embedding table went to
        int8 misstates the change made to it (Apache 2.0 §4(b))."""
        notice = " ".join(gemma4_distribution.GEMMA4_NOTICE_MARKDOWN.split())

        assert "and the embedding tables to int8." not in notice
        assert "for `gemma-4-E2B-it` and `gemma-4-E4B-it` the embedding tables to int8" in notice
        assert "the linear weights and the embedding table keep its own 4-bit grid" in notice

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

        with pytest.raises(AssertionError, match=UNLISTED_CHECKPOINT):
            _assert_notice_names(
                notice, [*GEMMA4_UPSTREAM.values(), f"google/{UNLISTED_CHECKPOINT}"]
            )


#: E4B の上流 `tokenizer.json` の代役（門は中身を読まず sha256 だけを突き合わせる）。
_UPSTREAM_TOKENIZER = b'{"model": "upstream tokenizer stand-in"}'


def _e4b_sources(root: Path) -> Gemma4Sources:
    """E4B の入力 — drafter 無し・トークナイザは E2B の系列を借りる（共有の表の既定どおり）。"""
    return Gemma4Sources(
        product=root / "series" / "gemma4-e4b-product",
        drafter=None,
        tokenizer=root / "series" / "gemma4-e2b-tokenizer",
        model=root / "inputs" / "gemma-4-E4B-it",
        upstream_tokenizer=root / "inputs" / "gemma-4-E4B-it",
    )


def _build_e4b(
    root: Path, *, upstream: bytes = _UPSTREAM_TOKENIZER, recorded: bytes | None = None
) -> Gemma4Sources:
    """E4B の最小系列（`recorded` = トークナイザ資産が名乗る出所のバイト列・既定は上流と同一）。"""
    sources = _e4b_sources(root)
    source_bytes = upstream if recorded is None else recorded
    fixture.write_series(
        sources.product,
        sources.tokenizer,
        sources.model,
        # 借り手 drafter 系列は fixture が既定で product の隣へ書く（E4B の計画は読まない）。
        tokenizer={
            **fixture.tokenizer_asset(),
            "source": {"sha256": hashlib.sha256(source_bytes).hexdigest()},
        },
    )
    (sources.model / "tokenizer.json").write_bytes(upstream)
    return sources


def _assemble_e4b(root: Path) -> tuple[Path, dict[str, Any]]:
    sources = _build_e4b(root)
    out_dir = root / "models" / "e4b-only"
    manifest = assemble_family(
        [gemma4_plan(sources, "e4b")],
        out_dir,
        "e4b",
        root_files=gemma4_distribution.PIPELINE.root_files,
    )
    return out_dir, manifest


class TestGemma4E4B:
    """E4B の model entry — drafter 無し・quant は E2B と同じ 3 席・既定は暫定 `i4`。"""

    def test_the_upstream_and_the_series_names_follow_the_model(self, tmp_path: Path) -> None:
        assert GEMMA4_UPSTREAM["e4b"] == "google/gemma-4-E4B-it"
        sources = gemma4_sources(tmp_path, "e4b")

        assert sources.product == tmp_path / "gemma4-e4b-product"
        assert sources.model.name == "gemma-4-E4B-it"
        # drafter は配らない・トークナイザは E2B の系列を共有する。
        assert sources.drafter is None
        assert sources.tokenizer == tmp_path / "gemma4-e2b-tokenizer"

    def test_every_model_declares_a_default_quant_and_a_tokenizer_series(self) -> None:
        """キー集合が帰属表と一致しないと、未宣言のモデルが組み立て時に KeyError で落ちる。"""
        assert set(gemma4_distribution.GEMMA4_DEFAULT_QUANT) == set(GEMMA4_UPSTREAM)
        assert set(gemma4_distribution.GEMMA4_TOKENIZER_SERIES) == set(GEMMA4_UPSTREAM)
        assert set(gemma4_distribution.GEMMA4_SPEED_NOTE) == set(GEMMA4_UPSTREAM)
        for model, quant in gemma4_distribution.GEMMA4_DEFAULT_QUANT.items():
            assert quant in gemma4_distribution.gemma4_quants(model)

    def test_the_entry_declares_no_drafter(self, tmp_path: Path) -> None:
        """drafter 役が weights に無いことが「投機を張れない」の唯一の表現。

        読み手は `speculative` 指定のロードを weights の解決で落とす（部品を 1 つも取る前）。
        """
        out_dir, manifest = _assemble_e4b(tmp_path)
        entry = manifest["models"]["e4b"]

        assert list(entry["weights"]) == [GEMMA4_ROLE]
        assert not (out_dir / "e4b" / GEMMA4_DRAFTER_ROLE).exists()
        for quant in entry["quants"].values():
            assert quant["weights"] == {GEMMA4_ROLE: "i4"}
            assert "drafter" not in quant["description"].lower()

    def test_the_entry_carries_the_three_quants_with_i4_as_the_default(
        self, tmp_path: Path
    ) -> None:
        _, manifest = _assemble_e4b(tmp_path)
        entry = manifest["models"]["e4b"]

        assert sorted(entry["quants"]) == ["i4", "i4-fast", "i4-gemvpar"]
        assert entry["defaultQuant"] == "i4"
        # E2B の既定は据え置き（E4B の暫定は E2B の束に影響しない）。
        assert gemma4_distribution.GEMMA4_DEFAULT_QUANT["e2b"] == "i4-fast"

    def test_the_unmeasured_seats_carry_the_e2b_declarations_without_a_speed_claim(
        self, tmp_path: Path
    ) -> None:
        """E4B の並列・融合の席は E2B と同じ宣言のまま、速いとは名乗らない（E4B は未計測）。"""
        _, manifest = _assemble_e4b(tmp_path)
        entry = manifest["models"]["e4b"]
        e2b = gemma4_distribution.gemma4_quants("e2b")

        for name in ("i4-gemvpar", "i4-fast"):
            description = entry["quants"][name]["description"]
            assert entry["quants"][name]["session"] == e2b[name]["session"]
            assert "not yet measured" in description.lower()
            assert "faster" not in description.lower()

    def test_the_card_names_no_drafter_for_an_e4b_only_manifest(self, tmp_path: Path) -> None:
        """配っていない drafter の上流を出所に書かない（base_model は manifest から導く）。"""
        _, manifest = _assemble_e4b(tmp_path)

        card = render_gemma4_model_card(manifest, "hdae/karume-gemma4")

        assert "google/gemma-4-E4B-it" in card
        assert "gemma-4-E2B-it-assistant" not in card
        assert "drafter head" not in card

    def test_the_card_still_names_the_e2b_drafter_when_e2b_ships_it(self, gemma4_assembled) -> None:
        """恒真化の門 — drafter を運ぶモデルが居れば drafter の上流は base_model に残る。"""
        _, manifest = gemma4_assembled

        card = render_gemma4_model_card(manifest, "hdae/karume-gemma4")

        assert "google/gemma-4-E2B-it-assistant" in card

    def test_both_models_share_one_tokenizer_file(self, tmp_path: Path) -> None:
        """同じ系列を指す 2 モデルの資産は `shared/` へ 1 回だけ置かれる（core の共有席）。"""
        e4b = _build_e4b(tmp_path)
        e2b = _sources(tmp_path)
        fixture.write_series(
            e2b.product,
            e2b.tokenizer,
            e2b.model,
            # E2B は同じ資産を読む（E4B が書いた出所つきの資産をそのまま使う）。
            tokenizer=json.loads((e4b.tokenizer / "tokenizer.json").read_text(encoding="utf-8")),
        )
        out_dir = tmp_path / "models" / "both"

        manifest = assemble_family(
            [gemma4_plan(e2b, "e2b"), gemma4_plan(e4b, "e4b")],
            out_dir,
            "e2b",
            root_files=gemma4_distribution.PIPELINE.root_files,
        )

        paths = {
            name: model["assets"][GEMMA4_TOKENIZER_ROLE]["path"]
            for name, model in manifest["models"].items()
        }
        assert paths["e2b"] == paths["e4b"]
        assert paths["e4b"].startswith("shared/")
        # E2B は drafter を運び続ける（E4B を足しても E2B の entry は変わらない）。
        assert GEMMA4_DRAFTER_ROLE in manifest["models"]["e2b"]["weights"]

    def test_it_refuses_a_borrowed_tokenizer_compiled_from_another_upstream(
        self, tmp_path: Path
    ) -> None:
        """共有の前提（上流が同一バイト）が崩れた組は落とす — 語彙数が同じでも id 列は割れうる。"""
        sources = _build_e4b(tmp_path, recorded=b"a different upstream tokenizer")

        with pytest.raises(DistError, match="共有の前提"):
            gemma4_plan(sources, "e4b")

    def test_it_refuses_a_borrowed_tokenizer_without_a_recorded_source(
        self, tmp_path: Path
    ) -> None:
        sources = _build_e4b(tmp_path)
        (sources.tokenizer / "tokenizer.json").write_text(
            json.dumps(fixture.tokenizer_asset()), encoding="utf-8"
        )

        with pytest.raises(DistError, match=r"source\.sha256"):
            gemma4_plan(sources, "e4b")

    def test_it_refuses_a_drafter_series_for_a_model_without_a_drafter(
        self, tmp_path: Path
    ) -> None:
        """drafter を配らないモデルに席があれば、別モデル向けの drafter が紛れ込んだ組。"""
        sources = _build_e4b(tmp_path)
        smuggled = replace(sources, drafter=tmp_path / "series" / "gemma4-e2b-drafter")

        with pytest.raises(DistError, match="drafter 系列"):
            gemma4_plan(smuggled, "e4b")

    def test_it_refuses_an_e2b_plan_whose_drafter_series_is_missing(self, tmp_path: Path) -> None:
        """drafter を配るモデルの席が空なら、黙って投機なしの配布形が出来る形を落とす。"""
        sources = replace(_build(tmp_path), drafter=None)

        with pytest.raises(DistError, match="drafter 系列"):
            gemma4_plan(sources, "e2b")


#: The ids the 12B stand-in suppresses (upstream writes `[258883, 258882]` — descending).
_TWELVE_B_SUPPRESSED = (fixture.VOCAB - 1, fixture.VOCAB - 2)

#: 12B's upstream `tokenizer.json` stand-in (the gate compares SHA-256 only) and its pins.
_TWELVE_B_TOKENIZER = b'{"model": "12b tokenizer stand-in"}'
_TWELVE_B_PINS = gemma4_distribution.HUB_PINS["12b"]

#: The `tokenizer.json` the QAT weights' repository ships (E2B's — `gemma4.sources`), placed next
#: to the stand-in weights' config so a plan that read the tokenizer from there would fail.
_QAT_REPOSITORY_TOKENIZER = b'{"model": "E2B tokenizer shipped with the QAT weights"}'

#: The provenance a pinned 12B container carries (`gemma4/export_unified.py` bakes the weights'
#: revision).
_TWELVE_B_PROVENANCE = Provenance(
    license="apache-2.0",
    writer="karume-fixture",
    upstream_revision=_TWELVE_B_PINS.weights.revision,
)


def _twelve_b_container(**overrides: Any) -> list[bytes]:
    """A PLE-free product container (no `per_layer_inputs`, no PLE assets) naming the pin, its
    embedding packed int4 like its linears (no int8 tensor)."""
    return fixture.product_container(
        **{
            "ple": False,
            "provenance": _TWELVE_B_PROVENANCE,
            "embedding_storage": "i4",
            **overrides,
        }
    )


def _build_twelve_b(
    root: Path,
    *,
    recorded: bytes = _TWELVE_B_TOKENIZER,
    container: list[bytes] | None = None,
    ple_dim: int = 0,
    suppress: list[int] | None = None,
    max_position: int = fixture.MAX_POSITION,
) -> Gemma4Sources:
    """12B's minimal series: its own tokenizer series (recording `recorded` as its source), no
    drafter, a config without per-layer embeddings, and a generation config that suppresses
    {@link _TWELVE_B_SUPPRESSED} (upstream order — unsorted).

    The upstream side is two directories, as the pins make it: the weights' config (next to the
    QAT repository's own `tokenizer.json` — {@link _QAT_REPOSITORY_TOKENIZER}), and the tokenizer
    source holding `tokenizer.json` and `generation_config.json`."""
    sources = Gemma4Sources(
        product=root / "series" / "gemma4-12b-product",
        drafter=None,
        tokenizer=root / "series" / "gemma4-12b-tokenizer",
        model=root / "inputs" / "gemma-4-12B-it-qat-q4_0-unquantized",
        upstream_tokenizer=root / "inputs" / "gemma-4-12B-it",
    )
    fixture.write_series(
        sources.product,
        sources.tokenizer,
        sources.model,
        container=container if container is not None else _twelve_b_container(),
        tokenizer={
            **fixture.tokenizer_asset(),
            "source": {"sha256": hashlib.sha256(recorded).hexdigest()},
        },
        text_config={
            **fixture.TEXT_CONFIG,
            "hidden_size_per_layer_input": ple_dim,
            "max_position_embeddings": max_position,
        },
        generation_config={
            **fixture.GENERATION_CONFIG,
            "suppress_tokens": list(_TWELVE_B_SUPPRESSED if suppress is None else suppress),
        },
    )
    sources.upstream_tokenizer.mkdir(parents=True)
    (sources.model / "generation_config.json").rename(
        sources.upstream_tokenizer / "generation_config.json"
    )
    (sources.upstream_tokenizer / "tokenizer.json").write_bytes(_TWELVE_B_TOKENIZER)
    (sources.model / "tokenizer.json").write_bytes(_QAT_REPOSITORY_TOKENIZER)
    return sources


def _assemble_models(root: Path, plans: list[Any], default: str) -> tuple[Path, dict[str, Any]]:
    out_dir = root / "models" / "family"
    manifest = assemble_family(
        plans,
        out_dir,
        default,
        render_card=partial(render_gemma4_model_card, repo="hdae/karume-gemma4"),
        root_files=gemma4_distribution.PIPELINE.root_files,
    )
    return out_dir, manifest


class TestGemma4TwelveB:
    """The 12B entry — no per-layer embeddings, its own tokenizer, `i4` alone, its own dialect."""

    def test_the_tables_cover_every_model(self) -> None:
        assert set(gemma4_card.GEMMA4_FRONT_ENDS) == set(GEMMA4_UPSTREAM)
        assert set(gemma4_card.GEMMA4_EMBEDDING_STORAGE) == set(GEMMA4_UPSTREAM)
        assert set(chat.CHAT_DIALECTS) == set(GEMMA4_UPSTREAM)
        assert set(chat.CHAT_DIALECTS.values()) <= set(chat.GENERATION_PROMPT_SUFFIX)
        assert {"e2b", "e4b"} == gemma4_card.GEMMA4_PLE_MODELS

    def test_the_sources_are_the_two_pinned_hub_snapshots(self, tmp_path, monkeypatch) -> None:
        """The weights' config from the QAT checkpoint, the tokenizer and the generation defaults
        from 12B-it."""
        constants = pytest.importorskip("huggingface_hub.constants")
        monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path / "hub"))

        sources = gemma4_sources(tmp_path, "12b")

        assert sources.model == (
            tmp_path
            / "hub"
            / "models--google--gemma-4-12B-it-qat-q4_0-unquantized"
            / "snapshots"
            / _TWELVE_B_PINS.weights.revision
        )
        assert sources.upstream_tokenizer == (
            tmp_path
            / "hub"
            / "models--google--gemma-4-12B-it"
            / "snapshots"
            / _TWELVE_B_PINS.tokenizer.revision
        )
        assert sources.tokenizer == tmp_path / "gemma4-12b-tokenizer"
        assert sources.drafter is None

    def test_e2b_and_e4b_read_one_checkpoint(self, tmp_path) -> None:
        for model in ("e2b", "e4b"):
            sources = gemma4_sources(tmp_path, model)

            assert sources.upstream_tokenizer == sources.model

    def test_the_tokenizer_and_the_generation_defaults_come_from_the_tokenizer_source(
        self, tmp_path
    ) -> None:
        """The plan passes with the QAT repository's `tokenizer.json` next to the weights' config
        (the fixture places it there) and reads the suppressed ids from the tokenizer source."""
        sources = _build_twelve_b(tmp_path)

        plan = gemma4_plan(sources, "12b")

        assert not (sources.model / "generation_config.json").exists()
        assert plan.pipeline_config["suppressTokens"] == sorted(_TWELVE_B_SUPPRESSED)

    def test_a_tokenizer_asset_compiled_from_the_qat_repository_s_file_is_refused(
        self, tmp_path
    ) -> None:
        sources = _build_twelve_b(tmp_path, recorded=_QAT_REPOSITORY_TOKENIZER)

        with pytest.raises(DistError, match="共有の前提"):
            gemma4_plan(sources, "12b")

    def test_the_entry_is_the_ple_free_int4_model(self, tmp_path) -> None:
        out_dir, manifest = _assemble_models(
            tmp_path, [gemma4_plan(_build_twelve_b(tmp_path), "12b")], "12b"
        )
        entry = manifest["models"]["12b"]

        assert list(entry["weights"]) == [GEMMA4_ROLE]
        assert list(entry["quants"]) == ["i4"]
        assert entry["defaultQuant"] == "i4"
        assert entry["quants"]["i4"]["session"] == {}
        assert entry["pipelineConfig"]["chatDialect"] == chat.GEMMA4_UNIFIED_CHAT_DIALECT
        assert entry["pipelineConfig"]["suppressTokens"] == sorted(_TWELVE_B_SUPPRESSED)
        first_part = sorted((out_dir / "12b" / GEMMA4_ROLE).iterdir())[0]
        roles = {role for role, _ in read_asset_declarations(first_part).values()}
        assert not roles & {PLE_INDEX_ROLE, *PLE_ROLES.values()}

    def test_only_the_qat_weights_carry_an_int4_embedding(self) -> None:
        """The card says an int4 embedding is "on the same 4-bit grid" as the QAT linears; a
        model with an int4 embedding outside the QAT table would make that clause false."""
        int4 = {
            name
            for name, storage in gemma4_card.GEMMA4_EMBEDDING_STORAGE.items()
            if storage == "i4"
        }

        assert int4 == {"12b"}
        assert int4 <= set(gemma4_card.GEMMA4_WEIGHTS_UPSTREAM)

    def test_it_refuses_a_12b_container_whose_embedding_is_int8(self, tmp_path) -> None:
        """The first 12B export rounded the embedding to int8 again; shapes and manifest are the
        same, only the numbers differ, so the storage gate is what refuses that series."""
        sources = _build_twelve_b(tmp_path, container=_twelve_b_container(embedding_storage="i8"))

        with pytest.raises(DistError, match=r"has i8 \(actual"):
            gemma4_plan(sources, "12b")

    def test_the_i4_seat_names_the_int4_embedding(self, tmp_path) -> None:
        _, manifest = _assemble_models(
            tmp_path,
            [gemma4_plan(_build(tmp_path), "e2b"), gemma4_plan(_build_twelve_b(tmp_path), "12b")],
            "e2b",
        )
        twelve_b = manifest["models"]["12b"]["quants"]["i4"]
        e2b = manifest["models"]["e2b"]["quants"]["i4"]

        assert twelve_b["label"] == "Packed int4 linear and embeddings"
        assert "int8" not in twelve_b["description"]
        assert e2b["label"] == "Packed int4 linear, int8 embeddings"
        assert "int8 embedding tables" in e2b["description"]

    def test_adding_12b_leaves_the_e2b_entry_byte_identical(self, tmp_path) -> None:
        _, alone = _assemble_models(
            tmp_path / "alone", [gemma4_plan(_build(tmp_path / "alone"), "e2b")], "e2b"
        )
        both_root = tmp_path / "both"
        _, both = _assemble_models(
            both_root,
            [gemma4_plan(_build(both_root), "e2b"), gemma4_plan(_build_twelve_b(both_root), "12b")],
            "e2b",
        )

        assert json.dumps(both["models"]["e2b"], sort_keys=True) == json.dumps(
            alone["models"]["e2b"], sort_keys=True
        )
        assert "chatDialect" not in both["models"]["e2b"]["pipelineConfig"]
        assert "suppressTokens" not in both["models"]["e2b"]["pipelineConfig"]

    def test_it_refuses_a_suppressed_id_outside_the_vocabulary(self, tmp_path) -> None:
        sources = _build_twelve_b(tmp_path, suppress=[fixture.VOCAB])

        with pytest.raises(DistError, match="outside the graph's vocabulary"):
            gemma4_plan(sources, "12b")

    def test_it_refuses_e2b_s_tokenizer_asset_for_12b(self, tmp_path) -> None:
        """12B's tokenizer drops 239 of E2B's merges — E2B's asset in its series is refused."""
        sources = _build_twelve_b(tmp_path, recorded=b"E2B's tokenizer.json")

        with pytest.raises(DistError, match="共有の前提"):
            gemma4_plan(sources, "12b")

    def test_it_refuses_a_container_carrying_ple_assets(self, tmp_path) -> None:
        sources = _build_twelve_b(
            tmp_path, container=_twelve_b_container(assets=fixture.ple_container_assets())
        )

        with pytest.raises(DistError, match="PLE assets"):
            gemma4_plan(sources, "12b")

    def test_it_refuses_a_graph_that_takes_per_layer_inputs(self, tmp_path) -> None:
        sources = _build_twelve_b(tmp_path, container=_twelve_b_container(ple=True, assets={}))

        with pytest.raises(DistError, match="per_layer_inputs"):
            gemma4_plan(sources, "12b")

    def test_it_refuses_a_config_that_disagrees_with_the_model_table(self, tmp_path) -> None:
        sources = _build_twelve_b(tmp_path, ple_dim=fixture.DIM)

        with pytest.raises(DistError, match="GEMMA4_PLE_MODELS"):
            gemma4_plan(sources, "12b")

    def test_it_refuses_a_container_from_another_revision(self, tmp_path) -> None:
        other = Provenance(
            license="apache-2.0", writer="karume-fixture", upstream_revision="1" * 40
        )
        sources = _build_twelve_b(tmp_path, container=_twelve_b_container(provenance=other))

        with pytest.raises(DistError, match="revision"):
            gemma4_plan(sources, "12b")

    def test_the_dist_plan_checks_the_snapshot_against_the_pin(self, tmp_path, monkeypatch) -> None:
        constants = pytest.importorskip("huggingface_hub.constants")
        monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path / "empty-hub"))

        with pytest.raises(DistError, match="missing"):
            gemma4_distribution.gemma4_dist_plan(tmp_path / "series", "12b")

    def test_the_card_names_the_qat_weights_and_the_base_model(self, tmp_path) -> None:
        out_dir, _ = _assemble_models(
            tmp_path,
            [gemma4_plan(_build(tmp_path), "e2b"), gemma4_plan(_build_twelve_b(tmp_path), "12b")],
            "e2b",
        )
        card = (out_dir / "README.md").read_text(encoding="utf-8")
        front_matter = card.split("---", 2)[1]

        assert (
            "- **`12b`**: the weights are [google/gemma-4-12B-it-qat-q4_0-unquantized]"
            "(https://huggingface.co/google/gemma-4-12B-it-qat-q4_0-unquantized)"
        ) in card
        assert "of the base model [google/gemma-4-12B-it]" in card
        assert "the source of the tokenizer" in card
        assert "for `12b`, the QAT checkpoint's own 4-bit grid" in card
        # Both redistributed upstreams, the base model first (the QAT checkpoint's own
        # `base_model`).
        listed = [line.strip() for line in front_matter.splitlines() if "google/" in line]
        assert listed == [
            "- google/gemma-4-E2B-it",
            "- google/gemma-4-12B-it",
            "- google/gemma-4-12B-it-qat-q4_0-unquantized",
            "- google/gemma-4-E2B-it-assistant",
        ]

    def test_the_card_names_the_ple_and_the_front_ends_per_model(self, tmp_path) -> None:
        out_dir, _ = _assemble_models(
            tmp_path,
            [gemma4_plan(_build(tmp_path), "e2b"), gemma4_plan(_build_twelve_b(tmp_path), "12b")],
            "e2b",
        )
        card = (out_dir / "README.md").read_text(encoding="utf-8")

        assert "google/gemma-4-12B-it" in card
        assert "**Per-layer embeddings** (`e2b`) **are container assets" in card
        assert "`12b` has no per-layer embeddings." in card
        assert "encoder-free vision and audio embedders for `12b`" in card
        assert "the per-layer embeddings of `e2b` were moved out of the graph" in card
        assert (
            "and the embedding tables of `e2b` to int8 and the embedding table of `12b`, on the"
            " same 4-bit grid, to packed int4 (group 32);"
        ) in card.replace("\n  ", " ")

    def test_a_12b_only_card_states_no_int8_embedding(self, tmp_path) -> None:
        out_dir, _ = _assemble_models(
            tmp_path, [gemma4_plan(_build_twelve_b(tmp_path), "12b")], "12b"
        )
        card = (out_dir / "README.md").read_text(encoding="utf-8").replace("\n  ", " ")

        assert "the embedding table of `12b`, on the same 4-bit grid, to packed int4" in card
        assert "int8 embedding" not in card
        assert "embedding tables to int8" not in card

    def test_the_card_states_the_position_limit_per_model(self, tmp_path) -> None:
        """12B declares twice E2B / E4B's positions — the overview must not state the default
        model's limit for both."""
        twelve_b = 2 * fixture.MAX_POSITION
        out_dir, _ = _assemble_models(
            tmp_path,
            [
                gemma4_plan(_build(tmp_path), "e2b"),
                gemma4_plan(_build_twelve_b(tmp_path, max_position=twelve_b), "12b"),
            ],
            "e2b",
        )
        card = (out_dir / "README.md").read_text(encoding="utf-8")

        assert (
            f"the model's declared position limit ({fixture.MAX_POSITION} for `e2b`,"
            f" {twelve_b} for `12b`)"
        ) in card

    def test_an_e2b_only_card_keeps_its_text(self, gemma4_assembled) -> None:
        _, manifest = gemma4_assembled

        card = render_gemma4_model_card(manifest, "hdae/karume-gemma4")

        assert "- **Per-layer embeddings are container assets, not graph weights.**" in card
        assert f"{fixture.MAX_POSITION} (the model's declared position limit)" in card
        assert "also carries vision and audio towers;\n  this distribution contains neither" in card
        assert (
            "  were quantized to packed int4 (group 32) and the embedding tables to int8; the\n"
            in card
        )
