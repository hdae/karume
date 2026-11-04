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
import re
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
    GEMMA4_REGULAR_MODELS,
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
    """先頭 block の末尾を 1 行縮める（範囲が連続しない索引）— in every table the index has."""
    for key in ("values", "scales"):
        if key in index:
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


#: tiny な PLE を 2 block 以上へ割る block 上限（block 跨ぎを 1 度は踏む）— half the vocabulary of
#: the bf16 table's rows.
_PLE_BLOCK_BYTES = fixture.LAYERS * fixture.DIM * 2 * (fixture.VOCAB // 2)

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
        product=root / "series" / "gemma4-e2b-product-bf16",
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
        """`ple.probe.safetensors` / the greedy goldens / `reference.json` は系列に同居するが
        配布へは入らない。"""
        out_dir, _ = gemma4_assembled
        assert list(out_dir.rglob("ple.probe.*")) == []
        assert list(out_dir.rglob("greedy.*")) == []
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
        # The one storage of the main model: the checkpoint's bf16 values.
        assert list(model["weights"][GEMMA4_ROLE]) == ["bf16"]
        assert list(model["weights"][GEMMA4_DRAFTER_ROLE]) == ["i8"]
        assert list(model["quants"]) == ["bf16"]
        assert model["defaultQuant"] == "bf16"
        # 役割ごとに基底格納が違う（drafter は linear まで i8）ので、自動補完が 2 席とも
        # それぞれの唯一の dtype ラベルで埋める（in the declaration order of the weights）。
        assert list(model["quants"]["bf16"]["weights"].items()) == [
            (GEMMA4_ROLE, "bf16"),
            (GEMMA4_DRAFTER_ROLE, "i8"),
        ]
        assert model["quants"]["bf16"]["session"] == {}

    def test_it_never_carries_the_drafter_goldens(self, gemma4_assembled) -> None:
        """drafter 系列の golden は検収専用 — `ple.probe` と同じく配布へは入らない。"""
        out_dir, _ = gemma4_assembled
        assert list(out_dir.rglob("drafter-golden.*")) == []

    def test_the_ple_blocks_live_inside_the_model_container(self, gemma4_assembled) -> None:
        """MUST: PLE は容器の資産（ADR 0109 決定 4）— manifest の `assets` には出ない。"""
        out_dir, manifest = gemma4_assembled
        model = _model(manifest)
        assert list(model["assets"]) == [GEMMA4_TOKENIZER_ROLE]

        placed = out_dir / model["weights"][GEMMA4_ROLE]["bf16"]["container"]["parts"][0]["path"]
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
            storage="bf16",
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
                storage="bf16",
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

    @pytest.mark.parametrize("model", GEMMA4_REGULAR_MODELS)
    def test_it_refuses_a_series_compiled_before_its_upstream_changed(
        self, tmp_path: Path, model: str
    ) -> None:
        """A model's own series is held to its current upstream `tokenizer.json` too: a series
        compiled before the file was replaced (Google's glitch-token fixes of 2026-11-03 changed
        E2B's and E4B's) has the right vocabulary and tokenizes some inputs differently."""
        sources = _build(tmp_path) if model == "e2b" else _build_e4b(tmp_path)
        gemma4_plan(sources, model)  # the series as compiled from its upstream passes

        (sources.upstream_tokenizer / "tokenizer.json").write_bytes(b'{"model": "after a fix"}')

        with pytest.raises(DistError, match=r"was compiled from a tokenizer\.json"):
            gemma4_plan(sources, model)


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

    @pytest.mark.parametrize(
        ("storage", "message"), [("f32", "has no bf16"), ("f16", "has no bf16")]
    )
    def test_it_refuses_a_container_without_the_bf16_weights(
        self, tmp_path: Path, storage: str, message: str
    ) -> None:
        sources = _build(tmp_path)
        replace_component(
            sources.product / "model.krm", ir_container(mark="plain", storage=storage)
        )
        with pytest.raises(DistError, match=message):
            gemma4_plan(sources)

    def test_it_refuses_an_int4_product_series(self, tmp_path: Path) -> None:
        """An int4 product series (the q4_0 QAT line's storage) has the same graph contract and PLE
        input, so only the storage tells it apart."""
        sources = _build(
            tmp_path, container=fixture.product_container(storage="i4", embedding_storage="i4")
        )

        with pytest.raises(DistError, match=f"{GEMMA4_ROLE}: storage .* has no bf16"):
            gemma4_plan(sources)

    @pytest.mark.parametrize("rounded", ["i8", "i4"])
    def test_it_refuses_a_rounded_weight_next_to_the_bf16_ones(
        self, tmp_path: Path, rounded: str
    ) -> None:
        """The presence check (bf16 is there) lets a container with one rounded table through;
        the product forbids every rounded storage."""
        sources = _build(tmp_path, container=fixture.product_container(embedding_storage=rounded))

        with pytest.raises(DistError, match=f"{GEMMA4_ROLE}: storage .* has {rounded}"):
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
        # The product series is the bf16 one (`gemma4/export_product.py` writes it).
        assert sources.product == tmp_path / "gemma4-e2b-product-bf16"
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
        """The tautology guard of the case above: rendered from the manifest alone (the container
        assets not counted), no note counts the PLE — at most the separate-file assets (the
        tokenizer) are counted.

        (The stand-in tokenizer asset records its source's SHA-256, which puts it above 1% of the
        stand-in's few-KiB download — the note's threshold — so the note may show with the
        tokenizer's bytes alone; the real asset is far below it.)"""
        _, manifest = gemma4_assembled
        tokenizer = _model(manifest)["assets"][GEMMA4_TOKENIZER_ROLE]["size"]
        assert tokenizer < 1024  # spelled in bytes below (`karume.modelcard._download_size`)

        card = render_gemma4_model_card(manifest, "hdae/karume-gemma4")

        notes = set(re.findall(r"\(([^()]*) of assets, read on the host\)", card))
        assert notes <= {f"{tokenizer} B"}, notes

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

    def test_the_relation_is_quantized_while_a_weight_is_stored_rounded(
        self, gemma4_assembled, tmp_path: Path
    ) -> None:
        """`base_model_relation` comes from the manifest: E2B's int8 drafter is a quantized form,
        while an E4B-only manifest stores nothing but the checkpoint's bf16 values — none of the
        Hub's relations, so the field is left out."""
        _, e2b = gemma4_assembled
        _, e4b = _assemble_e4b(tmp_path)

        def front_matter(manifest: Mapping[str, Any]) -> str:
            return render_gemma4_model_card(manifest, "hdae/karume-gemma4").split("---")[1]

        assert "base_model_relation: quantized\n" in front_matter(e2b)
        assert "base_model_relation" not in front_matter(e4b)

    def test_it_refuses_a_manifest_whose_main_model_is_not_bf16(self, gemma4_assembled) -> None:
        """The attribution states the checkpoint's bf16 values unchanged; a manifest whose `model`
        role stores something else is not drawn."""
        _, manifest = gemma4_assembled
        other = json.loads(json.dumps(manifest))
        weights = other["models"][GEMMA4_DEFAULT_MODEL]["weights"][GEMMA4_ROLE]
        weights["i4"] = weights.pop("bf16")

        with pytest.raises(ValueError, match=r"stored as \['i4'\]"):
            render_gemma4_model_card(other, "hdae/karume-gemma4")

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
        other["models"][GEMMA4_DEFAULT_MODEL]["quants"]["bf16"]["requiredLimits"] = {
            "maxStorageBufferBindingSize": 134_217_728,
        }
        card = render_gemma4_model_card(other, "hdae/karume-gemma4")

        assert "device limits" in card
        assert "maxStorageBufferBindingSize" in card
        assert "134,217,728" in card

    #: A second seat beside the line's one (`bf16`), so the folding across seats has two to fold —
    #: the q4_0 QAT line's card renders the same section over three.
    _TWIN = "bf16-twin"

    @classmethod
    def _device_limits_line(cls, manifest: Mapping[str, Any], limits_by_quant: Callable) -> str:
        other = json.loads(json.dumps(manifest))
        quants = other["models"][GEMMA4_DEFAULT_MODEL]["quants"]
        quants[cls._TWIN] = json.loads(json.dumps(quants["bf16"]))
        assert len(quants) >= 2
        for quant_name, quant in quants.items():
            quant["requiredLimits"] = limits_by_quant(quant_name)
        card = render_gemma4_model_card(other, "hdae/karume-gemma4")
        return next(line for line in card.splitlines() if "**device limits**" in line)

    def test_a_limit_every_quant_shares_is_named_once(self, gemma4_assembled) -> None:
        """同じ重みの席が同じ上限を持つとき、上限名は 1 回ずつ（quant ごとに繰り返さない）。"""
        _, manifest = gemma4_assembled
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
            lambda quant: {"maxBufferSize": 536_870_912 if quant == "bf16" else 402_653_184},
        )

        assert line.count("`maxBufferSize`") == 1
        assert "≥ 536,870,912 B for `bf16`" in line
        assert f"≥ 402,653,184 B for `{self._TWIN}`" in line

    def test_it_omits_the_device_limits_line_when_no_quant_declares_one(
        self, gemma4_assembled
    ) -> None:
        """逆枝 — 欄が無いときは節ごと消える（情報が黙って落ちる形の記録）。"""
        _, manifest = gemma4_assembled
        assert not _model(manifest)["quants"]["bf16"].get("requiredLimits")

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
        assert "keep the checkpoints' BF16 values unchanged" in notice
        # The drafter is the one rounded form; no int4 form of the main model ships.
        assert "int8 throughout" in notice
        assert "int4" not in notice

    def test_the_notice_names_no_model_outside_the_line(self) -> None:
        """12B ships in the q4_0 QAT line only (`karume-gemma4-qat`): a notice naming its
        checkpoint would claim a redistribution this repository does not make (§4(b))."""
        notice = gemma4_distribution.GEMMA4_NOTICE_MARKDOWN

        for model, upstream in GEMMA4_UPSTREAM.items():
            if model not in GEMMA4_REGULAR_MODELS:
                assert upstream not in notice, model
        assert "12B" not in notice

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
        _assert_notice_names(
            gemma4_distribution.GEMMA4_NOTICE_MARKDOWN,
            [GEMMA4_UPSTREAM[model] for model in GEMMA4_REGULAR_MODELS],
        )

    def test_a_model_the_notice_does_not_mention_is_caught(self, gemma4_assembled) -> None:
        """恒真でないことの裏取り — 2 件目を足した組で上の 2 本が落ちる。"""
        out_dir, _ = gemma4_assembled
        notice = (out_dir / "NOTICE.md").read_text(encoding="utf-8")

        with pytest.raises(AssertionError, match=UNLISTED_CHECKPOINT):
            _assert_notice_names(
                notice,
                [
                    *(GEMMA4_UPSTREAM[model] for model in GEMMA4_REGULAR_MODELS),
                    f"google/{UNLISTED_CHECKPOINT}",
                ],
            )


#: E4B の上流 `tokenizer.json` の代役（門は中身を読まず sha256 だけを突き合わせる）。
_UPSTREAM_TOKENIZER = b'{"model": "upstream tokenizer stand-in"}'


def _e4b_sources(root: Path) -> Gemma4Sources:
    """E4B's inputs — no drafter, its own tokenizer series (`GEMMA4_TOKENIZER_SERIES`)."""
    return Gemma4Sources(
        product=root / "series" / "gemma4-e4b-product-bf16",
        drafter=None,
        tokenizer=root / "series" / "gemma4-e4b-tokenizer",
        model=root / "inputs" / "gemma-4-E4B-it",
        upstream_tokenizer=root / "inputs" / "gemma-4-E4B-it",
    )


def _borrow_e2b_series(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make E4B borrow E2B's tokenizer series — no model borrows today, but the borrowing check
    (`assert_gemma4_tokenizer_source`) stays the rule for the day two upstreams are the same bytes
    again."""
    monkeypatch.setattr(
        gemma4_distribution,
        "GEMMA4_TOKENIZER_SERIES",
        {**gemma4_distribution.GEMMA4_TOKENIZER_SERIES, "e4b": "e2b"},
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
    """E4B の model entry — drafter 無し・quant は E2B と同じ 1 席 `bf16`（既定）。"""

    def test_the_upstream_and_the_series_names_follow_the_model(self, tmp_path: Path) -> None:
        assert GEMMA4_UPSTREAM["e4b"] == "google/gemma-4-E4B-it"
        sources = gemma4_sources(tmp_path, "e4b")

        assert sources.product == tmp_path / "gemma4-e4b-product-bf16"
        assert sources.model.name == "gemma-4-E4B-it"
        # No drafter; its own tokenizer series (E2B's and E4B's upstream tokenizers differ since
        # Google's fixes of 2026-11-03).
        assert sources.drafter is None
        assert sources.tokenizer == tmp_path / "gemma4-e4b-tokenizer"

    def test_every_model_declares_a_default_quant_and_a_tokenizer_series(self) -> None:
        """キー集合が the line's models と一致しないと、未宣言のモデルが組み立て時に KeyError で
        落ちる。"""
        assert set(gemma4_distribution.GEMMA4_DEFAULT_QUANT) == set(GEMMA4_REGULAR_MODELS)
        assert set(gemma4_distribution.GEMMA4_TOKENIZER_SERIES) == set(GEMMA4_UPSTREAM)
        assert set(GEMMA4_REGULAR_MODELS) == {"e2b", "e4b"}
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
        for name, quant in entry["quants"].items():
            # The preset loads the one storage of the main model alone.
            assert quant["weights"] == {GEMMA4_ROLE: "bf16"}, name
            assert "drafter" not in quant["description"].lower()

    def test_the_entry_carries_the_one_quant_bf16_as_the_default(self, tmp_path: Path) -> None:
        _, manifest = _assemble_e4b(tmp_path)
        entry = manifest["models"]["e4b"]

        assert list(entry["quants"]) == ["bf16"]
        assert entry["defaultQuant"] == "bf16"
        assert list(entry["weights"][GEMMA4_ROLE]) == ["bf16"]
        # Both models default to the original weights (the maintainer's ruling of 2026-11-04).
        assert gemma4_distribution.GEMMA4_DEFAULT_QUANT == {"e2b": "bf16", "e4b": "bf16"}

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
        """Two models whose tokenizer assets are the same bytes get one file in `shared/` (core's
        shared seat)."""
        e4b = _build_e4b(tmp_path)
        e2b = _sources(tmp_path)
        fixture.write_series(
            e2b.product,
            e2b.tokenizer,
            e2b.model,
            # E2B は同じ資産を読む（E4B が書いた出所つきの資産をそのまま使う）。
            tokenizer=json.loads((e4b.tokenizer / "tokenizer.json").read_text(encoding="utf-8")),
        )
        # Sharing holds only while both upstream files are the same bytes (the asset is held to
        # each model's own upstream `tokenizer.json`).
        (e2b.model / "tokenizer.json").write_bytes(_UPSTREAM_TOKENIZER)
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
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """共有の前提（上流が同一バイト）が崩れた組は落とす — 語彙数が同じでも id 列は割れうる。"""
        _borrow_e2b_series(monkeypatch)
        sources = _build_e4b(tmp_path, recorded=b"a different upstream tokenizer")

        with pytest.raises(DistError, match=r"was compiled from a tokenizer\.json"):
            gemma4_plan(sources, "e4b")

    def test_it_refuses_a_borrowed_tokenizer_without_a_recorded_source(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _borrow_e2b_series(monkeypatch)
        sources = _build_e4b(tmp_path)
        (sources.tokenizer / "tokenizer.json").write_text(
            json.dumps(fixture.tokenizer_asset(source=None)), encoding="utf-8"
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


class TestGemma4RegularModels:
    """The regular line ships E2B / E4B; 12B ships in the q4_0 QAT line only (its plan checks —
    PLE-free graph, its own tokenizer, dialect, suppressed ids — are in `test_distribution_qat.py`).
    """

    def test_the_tables_cover_every_model(self) -> None:
        assert set(gemma4_card.GEMMA4_FRONT_ENDS) == set(GEMMA4_UPSTREAM)
        assert set(chat.CHAT_DIALECTS) == set(GEMMA4_UPSTREAM)
        assert set(chat.CHAT_DIALECTS.values()) <= set(chat.GENERATION_PROMPT_SUFFIX)
        assert {"e2b", "e4b"} == gemma4_card.GEMMA4_PLE_MODELS

    def test_e2b_and_e4b_read_one_checkpoint(self, tmp_path) -> None:
        for model in ("e2b", "e4b"):
            sources = gemma4_sources(tmp_path, model)

            assert sources.upstream_tokenizer == sources.model

    def test_the_sources_refuse_12b(self, tmp_path) -> None:
        with pytest.raises(DistError, match=r"no model '12b'.*--pipeline gemma4-qat"):
            gemma4_sources(tmp_path, "12b")

    def test_the_plan_refuses_a_model_outside_the_line(self, tmp_path) -> None:
        """The plan holds the line's model set itself — a caller that builds the sources by hand
        does not bypass it."""
        sources = _build(tmp_path)

        with pytest.raises(DistError, match="not one of this line's models"):
            gemma4_plan(sources, "12b")

    def test_an_e2b_only_card_keeps_its_text(self, gemma4_assembled) -> None:
        _, manifest = gemma4_assembled

        card = render_gemma4_model_card(manifest, "hdae/karume-gemma4")

        assert "- **Per-layer embeddings are container assets, not graph weights.**" in card
        assert f"{fixture.MAX_POSITION} (the model's declared position limit)" in card
        assert "also carries vision and audio towers;\n  this distribution contains neither" in card
        assert (
            "  decoder was extracted and re-expressed in the Karume container format; the weights"
            " keep\n  the checkpoint's bf16 values unchanged; the per-layer embeddings were moved"
            " out of the\n" in card
        )
        assert "int4" not in card


class TestGemma4Bf16Product:
    """The line's one `model` container, the preset `bf16` (the default — the maintainer's ruling
    of 2026-11-04): the checkpoint's bf16 weights, the PLE included (index schema 5), with the int8
    drafter borrowing its table."""

    def test_the_sources_name_the_bf16_series(self, tmp_path: Path) -> None:
        for model in GEMMA4_REGULAR_MODELS:
            sources = gemma4_sources(tmp_path, model)

            assert sources.product == tmp_path / f"gemma4-{model}-product-bf16"

    def test_it_places_the_bf16_container_under_the_model_path(self, gemma4_assembled) -> None:
        out_dir, manifest = gemma4_assembled
        container = _model(manifest)["weights"][GEMMA4_ROLE]["bf16"]["container"]

        assert all(part["path"].startswith("e2b/model/model.bf16") for part in container["parts"])
        assert all((out_dir / part["path"]).is_file() for part in container["parts"])

    def test_the_bf16_container_carries_a_schema_5_ple(self, gemma4_assembled) -> None:
        out_dir, manifest = gemma4_assembled
        container = _model(manifest)["weights"][GEMMA4_ROLE]["bf16"]["container"]
        placed = out_dir / container["parts"][0]["path"]

        index = json.loads(read_asset(placed, PLE_INDEX_ASSET))
        read = gemma4_distribution.gemma4_ple_index(placed, storage="bf16")

        assert index["schema"] == 5
        assert "scales" not in index
        assert "scales" not in read
        assert read["values"]["rowBytes"] == fixture.LAYERS * fixture.DIM * 2

    def test_the_q4_0_qat_line_refuses_the_bf16_series(self, tmp_path: Path) -> None:
        """The other line's `model` role holds int4 alone — the regular bf16 series given to it is a
        caller's mix-up."""
        sources = _build(tmp_path)

        with pytest.raises(DistError, match=f"{GEMMA4_ROLE}: storage .* has no i4"):
            gemma4_plan(sources, "e2b", gemma4_distribution.GEMMA4_QAT_Q4_0_LINE)

    def test_it_refuses_a_drafter_that_cannot_borrow_from_the_product(self, tmp_path: Path) -> None:
        """The drafter borrows the product's table: one naming another key fails."""
        sources = _build(
            tmp_path, drafter_bytes=fixture.drafter_container(shared_tensor="gemma4.embed_other")
        )

        with pytest.raises(DistError, match=r"指し先.*product-bf16"):
            gemma4_plan(sources, "e2b")

    def test_it_refuses_a_schema_5_index_with_a_scales_table(self, tmp_path: Path) -> None:
        assets = dict(fixture.ple_container_assets(storage="bf16"))
        index = json.loads(bytes(assets[PLE_INDEX_ASSET].payload))
        index["scales"] = index["values"]
        payload = json.dumps(index).encode("utf-8")
        assets[PLE_INDEX_ASSET] = AssetInput(PLE_INDEX_ROLE, len(payload), payload)
        sources = _build(tmp_path, container=fixture.product_container(assets=assets))

        with pytest.raises(DistError, match=r"未知キー \['scales'\]"):
            gemma4_plan(sources, "e2b")

    def test_it_refuses_an_int8_index_in_the_bf16_container(self, tmp_path: Path) -> None:
        sources = _build(
            tmp_path,
            container=fixture.product_container(assets=fixture.ple_container_assets(storage="i8")),
        )

        # The schema 3 index carries a `scales` table, which a bf16 index does not have.
        with pytest.raises(DistError, match=r"product-bf16.*未知キー \['scales'\]"):
            gemma4_plan(sources, "e2b")

    def test_the_card_and_the_notice_state_the_bf16_values_unchanged(
        self, gemma4_assembled
    ) -> None:
        out_dir, manifest = gemma4_assembled

        card = render_gemma4_model_card(manifest, "hdae/karume-gemma4")
        notice = (out_dir / "NOTICE.md").read_text(encoding="utf-8")

        assert "its weights shipped unchanged (bf16)" in card
        assert "the checkpoint's bf16 values unchanged" in card
        assert "Original weights (bf16)" in card
        assert "keep the checkpoints' BF16 values unchanged" in notice
