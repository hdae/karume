"""gemma4's distribution recipe (`gemma4.distribution`) — unit tests of one assembly pass.

The real assets (3.7GiB) are not used. The input reaching the assembly is a **valid minimal product
series** of a few KB ({@link gemma4.tests.product_fixture}), and only the cases that check a gate
rejects something use variants of it.

The core is "compare, at assembly time, the same facts held by separate scripts" — the product
graph (`export_product.py`) / the PLE sidecar (same) / the tokenizer asset (`tokenizer.py`) / the
upstream generation defaults (the checkpoint) can all four move independently, so a mismatch can
only be rejected **before the distribution is laid out**.

The layers observable with the core alone (scale limits, the complete quant mapping, the
staging/swap invariants, the resolution rules of the attribution profiles) are covered by
`tools/exporter/tests/test_dist.py` (the split of ADR 0065 steps 3+4).
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

#: Role name → part count (the PLE assets take parts of their own, so only `model` has more).
_PART_TOTALS: Mapping[str, int] = {GEMMA4_ROLE: len(fixture.product_container())}

#: PLE dimensions (the original values when bending the index).
_PLE_DIMS: Mapping[str, int] = {"layers": fixture.LAYERS, "dim": fixture.DIM}


def _shift_first_block(index: dict[str, Any]) -> None:
    """Shrink the end of the first block by one row (an index whose ranges are not contiguous) — in
    every table the index has."""
    for key in ("values", "scales"):
        if key in index:
            index[key]["blocks"][0]["stop"] -= 1


def _rename_first_block(index: dict[str, Any]) -> None:
    """Replace only the first block's asset name with a spelling that is not in the container."""
    index["values"]["blocks"][0]["asset"] = "ple.values.absent"


def _container_with_ple(
    *,
    tokens: int = fixture.VOCAB,
    layers: int = fixture.LAYERS,
    dim: int = fixture.DIM,
    bend: Callable[[dict[str, Any]], None] | None = None,
) -> list[bytes]:
    """A product container (in a form where one place of the PLE index can be bent).

    The index is the payload of the asset `ple_index`, so only **that JSON** is bent — leaving the
    asset's block sequence as it is directly builds "a combination where only the index is stale".
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


#: Block limit that splits the tiny PLE into 2 or more blocks (so a block crossing is exercised at
#: least once) — half the vocabulary of the bf16 table's rows.
_PLE_BLOCK_BYTES = fixture.LAYERS * fixture.DIM * 2 * (fixture.VOCAB // 2)

#: A model name **not in** the attribution table and its upstream spelling (the input of the gate
#: that rejects unknown models). E4B entered the attribution table, so the gate's input is a
#: spelling that does not exist.
UNLISTED_MODEL = "unlisted"
UNLISTED_CHECKPOINT = "gemma-4-unlisted-it"

#: Run-time knobs that hold for the synthetic dimensions (the real ones are 768 / 4096; the
#: synthetic position limit is 37).
SMALL_CHUNK = 2
SMALL_CAPACITY = 4


@pytest.fixture(autouse=True)
def _small_runtime_knobs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bring the run-time knobs to the synthetic dimensions (to get past the
    `capacity ≤ maxPosition` gate).

    With the real values the capacity 4096 falls outside the synthetic position limit (37) — that
    combination itself is used as a gate by
    {@link TestGemma4Config.test_it_refuses_a_capacity_beyond_the_model_limit}.
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
    """The modification notice names the upstream checkpoint (checked by the spelling without the
    org)."""
    for repo in repos:
        checkpoint = repo.split("/", 1)[1]
        assert checkpoint in notice, f"the modification notice does not name '{checkpoint}'"


class TestGemma4Layout:
    def test_it_places_the_graph_sidecar_and_tokenizer_under_the_model_subtree(
        self, gemma4_assembled
    ) -> None:
        out_dir, _ = gemma4_assembled
        # PLE is an asset of the `model` container, so it does not appear as a separate file of the
        # distribution.
        expected = _in_subtree(
            GEMMA4_DEFAULT_MODEL,
            placed_paths(GEMMA4_OUTPUT_PATHS, GEMMA4_WEIGHTS, _PART_TOTALS),
        )
        # The 2 legal texts (Apache 2.0 §4) and the card are metadata files the manifest does not
        # declare.
        assert _present(out_dir) == sorted(
            [*expected, MANIFEST_FILENAME, "LICENSE.md", "NOTICE.md"]
        )

    def test_it_never_carries_the_acceptance_only_files(self, gemma4_assembled) -> None:
        """`ple.probe.safetensors` / the greedy goldens / `reference.json` live in the series but
        do not go into the distribution."""
        out_dir, _ = gemma4_assembled
        assert list(out_dir.rglob("ple.probe.*")) == []
        assert list(out_dir.rglob("greedy.*")) == []
        assert list(out_dir.rglob("reference.json")) == []

    def test_it_declares_two_graphs_and_the_sidecar_as_assets(self, gemma4_assembled) -> None:
        """weights are **the product graph and the drafter** (ADR 0096 step 2) — the drafter is not
        an asset.

        The order is checked too because the quant's `weights` mapping is filled in the declaration
        order of weights (the MUST of `karume.dist.complete_quant_weights`) — swapped, the 2
        sections of the manifest would be in another order.
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
        # The base storage differs per role (the drafter is i8 down to the linears), so the
        # auto-completion fills both entries with their own single dtype label (in the declaration
        # order of the weights).
        assert list(model["quants"]["bf16"]["weights"].items()) == [
            (GEMMA4_ROLE, "bf16"),
            (GEMMA4_DRAFTER_ROLE, "i8"),
        ]
        assert model["quants"]["bf16"]["session"] == {}

    def test_it_never_carries_the_drafter_goldens(self, gemma4_assembled) -> None:
        """The drafter series' goldens are for acceptance only — like `ple.probe`, they do not go
        into the distribution."""
        out_dir, _ = gemma4_assembled
        assert list(out_dir.rglob("drafter-golden.*")) == []

    def test_the_ple_blocks_live_inside_the_model_container(self, gemma4_assembled) -> None:
        """MUST: PLE is an asset of the container (ADR 0109 decision 4) — it does not appear in the
        manifest's `assets`."""
        out_dir, manifest = gemma4_assembled
        model = _model(manifest)
        assert list(model["assets"]) == [GEMMA4_TOKENIZER_ROLE]

        placed = out_dir / model["weights"][GEMMA4_ROLE]["bf16"]["container"]["parts"][0]["path"]
        index = json.loads(read_asset(placed, PLE_INDEX_ASSET))
        declared = read_asset_declarations(placed)
        blocks = [block["asset"] for block in index["values"]["blocks"]]
        assert blocks, "the index holds no block"
        for name in blocks:
            assert name in declared

    def test_it_reassembles_over_a_previous_run(self, tmp_path: Path) -> None:
        first_dir, first = _assemble(tmp_path)
        second_dir, second = _assemble(tmp_path)
        assert first_dir == second_dir
        assert first == second
        assert verify_dist(first_dir)


class TestGemma4Config:
    """The 6 fields of `pipelineConfig` — the derived ones (`maxPosition` / `rope` / `sampler`) and
    their relation to the run-time knobs."""

    def test_it_derives_max_position_from_the_upstream_declaration(self, gemma4_assembled) -> None:
        """MUST: no hand copy — the only source is upstream
        `text_config.max_position_embeddings`."""
        _, manifest = gemma4_assembled
        assert _model(manifest)["pipelineConfig"]["maxPosition"] == fixture.MAX_POSITION

    def test_it_derives_the_rope_parameters_from_the_upstream_config(
        self, gemma4_assembled
    ) -> None:
        """theta / headDim / rotaryDim per layer type (only full reads `global_head_dim`)."""
        _, manifest = gemma4_assembled

        assert _model(manifest)["pipelineConfig"]["rope"] == {
            SLIDING_ATTENTION: {
                "theta": fixture.SLIDING_THETA,
                "headDim": fixture.SLIDING_HEAD_DIM,
                # `default` rotates every frequency
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
        """MUST: fail if the formula is different (do not distribute a table the host cannot build
        as declared)."""
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
        """MUST: fail if a derivation source is missing (falling back to a default would make the
        distribution claim an arbitrary number)."""
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
        """MUST: no hand copy of the values (ADR 0083 decision 7 — the source is the upstream
        declaration itself)."""
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
        """The real knob (capacity 4096) × the synthetic position limit (37) — do not bake a form
        that fails only in long conversations."""
        monkeypatch.setattr(gemma4_distribution, "GEMMA4_CHUNK_LENGTH", GEMMA4_CHUNK_LENGTH)
        monkeypatch.setattr(gemma4_distribution, "GEMMA4_CAPACITY", GEMMA4_CAPACITY)
        sources = _build(tmp_path)
        with pytest.raises(DistError, match="exceeds the model's position limit"):
            gemma4_plan(sources)

    def test_it_refuses_a_capacity_below_one_chunk(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(gemma4_distribution, "GEMMA4_CHUNK_LENGTH", 4)
        monkeypatch.setattr(gemma4_distribution, "GEMMA4_CAPACITY", 2)
        sources = _build(tmp_path)
        with pytest.raises(DistError, match="not even one chunk fits"):
            gemma4_plan(sources)

    @pytest.mark.parametrize("chunk", [1, GEMMA4_MAX_CHUNK_LENGTH + 1])
    def test_it_refuses_a_chunk_length_outside_the_traced_range(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, chunk: int
    ) -> None:
        """The symbol `M` is declared as `[2, SYM_MAX]` at trace time — nothing outside it can be
        declared."""
        monkeypatch.setattr(gemma4_distribution, "GEMMA4_CHUNK_LENGTH", chunk)
        monkeypatch.setattr(gemma4_distribution, "GEMMA4_CAPACITY", 2048)
        sources = _build(tmp_path)
        with pytest.raises(DistError, match="chunkLength"):
            gemma4_plan(sources)

    def test_the_declared_knobs_are_inside_the_shipped_bounds(self) -> None:
        """The real declaration (768 / 4096) itself lies inside both limits."""
        assert 2 <= GEMMA4_CHUNK_LENGTH <= GEMMA4_MAX_CHUNK_LENGTH
        assert GEMMA4_CHUNK_LENGTH <= GEMMA4_CAPACITY

    def test_it_declares_the_traced_chunk_bound(self, gemma4_assembled) -> None:
        """MUST: the distribution declares the trace upper bound of the symbol `M` (the reader
        cannot derive it from the asset).

        The IR's `symbols` is only a list of names with no bounds, so a caller that overrode
        `chunkLength` going outside the trace range cannot be detected on the asset side.
        Putting the one number the baking side knows into the declaration lets the TS side
        (`gemma.parsePipelineConfig` / `assertChunkLength`) make it a gate.

        It is **a different fact** from the default (brought to {@link SMALL_CHUNK}), so the upper
        bound is not affected by monkeypatch and comes out at the real value.
        """
        _, manifest = gemma4_assembled
        assert _model(manifest)["pipelineConfig"]["maxChunkLength"] == GEMMA4_MAX_CHUNK_LENGTH

    def test_the_chunk_bound_mirrors_the_export_script(self) -> None:
        """MUST: the upper bound of the symbol `M` is the same number as the baking side's
        (`gemma4.export.SYM_MAX`).

        The distribution recipe does not import torch (so the default-sync CI job does not lose
        the whole collection), so it holds a copy. A stale copy would pass "a distribution that
        declares a chunk length outside the trace".

        NOTE: what is visible is only that the 2 **default values** agree. Even if the series is
        rebuilt with `--sym-max 640` or the like (below 513 the export itself refuses), the value
        used remains nowhere, so this gate stays green.
        """
        pytest.importorskip("torch")
        from gemma4.export import SYM_MAX

        assert GEMMA4_MAX_CHUNK_LENGTH == SYM_MAX

    def test_the_row_symbol_mirrors_the_export_script(self) -> None:
        """MUST: the spelling of the exit's row-count symbol is the same as the baking side's
        (`export_product.ROW_SYMBOL`).

        The distribution recipe does not import torch, so it holds a copy (the same reason as the
        upper bound of `M`). If the spellings split, the gate that rejects "an asset whose row axis
        is not a symbol" ({@link gemma4_vocab_size}) checks nothing.
        """
        pytest.importorskip("torch")
        from gemma4.export_product import ROW_SYMBOL

        assert gemma4_distribution.GEMMA4_ROW_SYMBOL == ROW_SYMBOL


class TestGemma4Graph:
    """The product graph's form — the order of the inputs, and the fit with the PLE index."""

    def test_it_refuses_a_graph_with_other_inputs(self, tmp_path: Path) -> None:
        # A container with differently spelled inputs (a mix-up with e.g. the token-only form of
        # `export_decode.py`).
        wrong = ir_container(
            mark="other",
            storage="bf16",
            inputs=(("input_ids", [1, "M"]), ("last_row", [fixture.ROW_SYMBOL])),
            outputs=(
                [1, fixture.ROW_SYMBOL, fixture.VOCAB],
                [1, fixture.ROW_SYMBOL, fixture.HIDDEN],
            ),
            # PLE is an asset of the same container, so it is placed on the replacement container
            # too (so the index gate does not fail first).
            assets=fixture.ple_container_assets(block_bytes=_PLE_BLOCK_BYTES),
        )
        sources = _build(tmp_path)
        replace_component(sources.product / "model.krm", wrong)
        with pytest.raises(DistError, match="graph inputs are"):
            gemma4_plan(sources)

    def test_it_refuses_a_graph_whose_exit_is_not_the_last_row(self, tmp_path: Path) -> None:
        sources = _sources(tmp_path)
        fixture.write_series(
            sources.product,
            sources.tokenizer,
            sources.model,
            # Logits whose row axis is not a symbol (`[1, 4, V]` — an asset with a baked row count).
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
        with pytest.raises(DistError, match=r"is not \[1, R, \*\]"):
            gemma4_plan(sources)

    def test_it_refuses_a_graph_whose_outputs_are_swapped(self, tmp_path: Path) -> None:
        """MUST: logits and hidden have the same shape down to the row axis — a swap is caught only
        by the width.

        Passed through mixed up, a manifest that read hidden_size as the vocabulary size would be
        assembled (the cross-checks with the PLE index and the tokenizer run with that number).
        """
        sources = _sources(tmp_path)
        fixture.write_series(
            sources.product,
            sources.tokenizer,
            sources.model,
            container=fixture.product_container(swap_outputs=True),
        )
        with pytest.raises(DistError, match=r"the width of output 1 \(hidden\)"):
            gemma4_plan(sources)

    def test_it_refuses_rope_inputs_whose_width_is_not_the_declared_head_dim(
        self, tmp_path: Path
    ) -> None:
        """The declaration (from the config) and the graph (from the container) move separately —
        this is the only place they are fitted together."""
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
        """Rejects a form that has both the derived inputs and the tables (a generation that did not
        move them out completely)."""
        sources = _sources(tmp_path)
        fixture.write_series(
            sources.product,
            sources.tokenizer,
            sources.model,
            container=fixture.product_container(baked_rope=True),
        )
        with pytest.raises(DistError, match="baked RoPE table initializers remain"):
            gemma4_plan(sources)

    def test_it_refuses_a_graph_without_a_free_capacity_symbol(self, tmp_path: Path) -> None:
        """The capacity symbol is the one that appears only in states (a mirror of
        `capacitySymbolOf` on the TS side)."""
        sources = _sources(tmp_path)
        fixture.write_series(
            sources.product,
            sources.tokenizer,
            sources.model,
            container=fixture.product_container(free_symbol=False),
        )
        with pytest.raises(DistError, match="symbols not determined by the input shapes"):
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
            # Moved by 2 because blocks cannot be cut unless one PLE row is a multiple of 4.
            container=_container_with_ple(**{field: int(_PLE_DIMS[field]) + 2}),
        )
        with pytest.raises(DistError, match=f"axis {axis}"):
            gemma4_plan(sources)


class TestGemma4Ple:
    """The form of the PLE index (the container's asset `ple_index`) and its fit with the asset
    declarations."""

    def test_it_refuses_an_index_whose_rows_are_not_the_vocabulary(self, tmp_path: Path) -> None:
        """MUST: if the row count differs from the vocabulary size, **a valid row of another token**
        is read (ADR 0085 decision 5)."""
        sources = _sources(tmp_path)
        fixture.write_series(
            sources.product,
            sources.tokenizer,
            sources.model,
            container=_container_with_ple(tokens=fixture.VOCAB - 2),
        )
        with pytest.raises(DistError, match="the product graph's vocabulary size"):
            gemma4_plan(sources)

    def test_it_refuses_an_index_that_is_not_a_partition(self, tmp_path: Path) -> None:
        """An index whose ranges are not contiguous makes "ids that cannot be looked up" or "two
        entries for the same id" (a silent wrong value)."""
        sources = _sources(tmp_path)
        fixture.write_series(
            sources.product,
            sources.tokenizer,
            sources.model,
            container=_container_with_ple(bend=_shift_first_block),
        )
        with pytest.raises(DistError, match="does not continue from the previous block's end"):
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
        with pytest.raises(DistError, match="unknown keys"):
            gemma4_plan(sources)

    def test_it_refuses_an_index_that_names_an_asset_the_container_lacks(
        self, tmp_path: Path
    ) -> None:
        """A combination where only the index was replaced reads another token's row **with shape
        and dtype still matching**."""
        sources = _sources(tmp_path)
        fixture.write_series(
            sources.product,
            sources.tokenizer,
            sources.model,
            container=_container_with_ple(bend=_rename_first_block),
        )
        with pytest.raises(DistError, match="is not in the container"):
            gemma4_plan(sources)

    def test_it_refuses_a_missing_part(self, tmp_path: Path) -> None:
        """Fail if even one of the part list is missing (it cannot be opened as a container)."""
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
        with pytest.raises(DistError, match="the product graph's vocabulary size"):
            gemma4_plan(sources)

    def test_it_refuses_a_raw_upstream_tokenizer_json(self, tmp_path: Path) -> None:
        """A mix-up that placed upstream's 32MB `tokenizer.json` (not run through the compile
        script)."""
        sources = _sources(tmp_path)
        fixture.write_series(
            sources.product,
            sources.tokenizer,
            sources.model,
            tokenizer={"version": "1.0", "model": {"type": "BPE"}},
        )
        with pytest.raises(DistError, match="format is"):
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
        with pytest.raises(DistError, match=f"{dropped} is missing"):
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
        """The accepted set is the same as `gemma.parsePipelineConfig` on the TS side — no form that
        fails after distribution."""
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
    """A mix-up of the series root — numeric gates cannot detect it in principle, so this is the
    only detector."""

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
        """An f16 creeping in can only mean "pointed at the series root of another family"."""
        sources = _build(tmp_path)
        replace_component(sources.product / "model.krm", ir_container(mark="half", storage="f16"))
        with pytest.raises(DistError):
            gemma4_plan(sources)

    def test_it_refuses_a_drafter_whose_linear_weights_fell_to_int4(self, tmp_path: Path) -> None:
        """Fail if the drafter has i4 — the only detector for an asset that merely loses 10 to 30
        percent of its acceptance rate.

        The output head stays i8, so a presence check (i8 is there) lets it through, and it could be
        distributed with shape and manifest still correct (measured on 2026-09-08 in
        `gemma4/export_drafter.py`).
        """
        sources = _build(tmp_path, drafter_bytes=fixture.drafter_container(storage="i4"))

        with pytest.raises(DistError, match=r"has (\S+ / )*i4 \(actual"):
            gemma4_plan(sources)


class TestGemma4Naming:
    def test_the_series_and_repo_names_come_from_one_word(self) -> None:
        assert gemma4_series_name("e2b", "product") == "gemma4-e2b-product"
        assert gemma4_series_name("e2b", "tokenizer") == "gemma4-e2b-tokenizer"
        assert gemma4_repo_name("e2b") == "karume-gemma4"
        # One repository per family (ADR 0092 decision 1) — whichever model is assembled, there is
        # one destination.
        assert gemma4_repo_name("e4b") == gemma4_repo_name("e2b")

    def test_the_sources_follow_the_repo_topology(self, tmp_path: Path) -> None:
        sources = gemma4_sources(tmp_path, GEMMA4_DEFAULT_MODEL)
        # The product series is the bf16 one (`gemma4/export_product.py` writes it).
        assert sources.product == tmp_path / "gemma4-e2b-product-bf16"
        assert sources.tokenizer == tmp_path / "gemma4-e2b-tokenizer"
        # The checkpoint is derived from the attribution table (no second table is held).
        assert sources.model.name == GEMMA4_UPSTREAM[GEMMA4_DEFAULT_MODEL].split("/", 1)[1]

    def test_it_refuses_a_model_outside_the_attribution_table(self, tmp_path: Path) -> None:
        with pytest.raises(DistError, match="unknown gemma4 model"):
            gemma4_sources(tmp_path, UNLISTED_MODEL)


class TestGemma4Card:
    def test_the_profile_is_the_only_one(self) -> None:
        profiles = gemma4_distribution.PIPELINE.card_profiles
        assert list(profiles) == ["gemma4"]
        assert resolve_card_renderer(gemma4_distribution.PIPELINE, None) is profiles["gemma4"]

    def test_the_card_counts_the_container_assets_as_host_read(self, tmp_path: Path) -> None:
        """PLE moved into the container's assets (ADR 0109 decision 4) — the Download breakdown note
        counts it.

        In `karume/4` PLE was a separate file (the manifest's `assets`), so counting the path set
        was enough. Now that it is inside the container, the declaration exists only on the
        `container` side, so the assembly looks it up and passes it on
        ({@link karume.dist.container_asset_bytes}) — missing it in the count would silently drop
        the note in the one real case it exists for: "most of the Download is a table read by the
        host".
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
        """The roles the card talks about match the accepted set (`gemma4.chat.PLAIN_ROLES` — the
        same scope as the TS side).

        If even one is missing, readers of the public card misread that they cannot use that role.
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
        # Usage points to the upstream card (the license policy of 2026-09-01).
        assert "upstream model card" in card
        assert "LICENSE.md" in card and "NOTICE.md" in card
        # The numbers are derived from the manifest (the recommended sampler and the position limit
        # also appear in the text).
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
        with pytest.raises(ValueError, match="has no upstream in the attribution table"):
            render_gemma4_model_card(other, "hdae/karume-gemma4")

    def test_neither_the_title_nor_the_overview_names_a_model(self, gemma4_assembled) -> None:
        """One repository per family (ADR 0092 decision 1) — the name it states is taken from the
        manifest.

        Baking the model's spelling into the title or the summary would, on the day a second model
        is added, silently produce a card that "claims to be E2B while also distributing E4B" (the
        attribution-table gate of `_gemma4_metadata` lets it through when the model is **known**).
        """
        _, manifest = gemma4_assembled
        card = render_gemma4_model_card(manifest, "hdae/karume-gemma4")
        title = next(line for line in card.splitlines() if line.startswith("# "))
        overview = card.split("## What is this", 1)[1].split("\n## ", 1)[0]

        assert "E2B" not in title
        # The spelling the summary states is the one taken from the manifest (not baked in).
        assert GEMMA4_UPSTREAM[GEMMA4_DEFAULT_MODEL] in overview

    def test_it_names_the_device_limits_a_quant_declares(self, gemma4_assembled) -> None:
        """`requiredLimits` is a field for "knowing in advance which devices it does not fit", so
        it appears in the text.

        The synthetic series has small dimensions and the core does not bake the field, so the
        field itself is injected for the check.
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
        """When entries of the same weights have the same limit, each limit name appears once (not
        repeated per quant)."""
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
        """If the values split, each value gets its quant names (a single folded value is not to be
        read as the limit of every entry)."""
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
        """The reverse branch — without the field the section disappears (a record of the form
        where information is silently dropped)."""
        _, manifest = gemma4_assembled
        assert not _model(manifest)["quants"]["bf16"].get("requiredLimits")

        card = render_gemma4_model_card(manifest, "hdae/karume-gemma4")

        assert "device limits" not in card


class TestGemma4LegalText:
    def test_it_ships_the_license_text_byte_identical(self, gemma4_assembled) -> None:
        """§4(a) — what is provided is **a copy of this license** (not a summary, not reformatted).

        The original is `_shared/licenses/apache_license_2_0.txt`. A reformat or a line-ending
        conversion anywhere on the assembly path moves a byte while it stays valid prose, so the
        other gates let it through.
        """
        out_dir, _ = gemma4_assembled
        assert (out_dir / "LICENSE.md").read_bytes() == APACHE_LICENSE_2_0_PATH.read_bytes()

    def test_it_ships_the_modification_notice(self, gemma4_assembled) -> None:
        """The modification notice of Apache 2.0 §4(b)."""
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
        """The §4(b) notice names every upstream "this distribution actually redistributed",
        without exception.

        The notice is a `Pipeline.root_files` file (the core's type is `Mapping[str, str]`), so it
        cannot be built from the manifest — the correspondence between the baked prose and the
        models actually distributed can be checked only here.
        """
        out_dir, manifest = gemma4_assembled
        notice = (out_dir / "NOTICE.md").read_text(encoding="utf-8")

        _assert_notice_names(notice, [GEMMA4_UPSTREAM[name] for name in manifest["models"]])

    def test_the_notice_covers_the_whole_attribution_table(self) -> None:
        """The gate that turns red the moment a model is added to the attribution table (the MUST
        of `GEMMA4_NOTICE_MARKDOWN`).

        The prose can only be written by hand (the wording review of legal text assumes a person
        reads it), so forgetting it is caught here rather than at assembly time.
        """
        _assert_notice_names(
            gemma4_distribution.GEMMA4_NOTICE_MARKDOWN,
            [GEMMA4_UPSTREAM[model] for model in GEMMA4_REGULAR_MODELS],
        )

    def test_a_model_the_notice_does_not_mention_is_caught(self, gemma4_assembled) -> None:
        """Evidence that it is not a tautology — the 2 above fail on a combination with a second
        entry added."""
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


#: Stand-in for E4B's upstream `tokenizer.json` (the gate does not read the contents and compares
#: only the sha256).
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
    """E4B's minimal series (`recorded` = the bytes of the source the tokenizer asset states;
    by default identical to upstream)."""
    sources = _e4b_sources(root)
    source_bytes = upstream if recorded is None else recorded
    fixture.write_series(
        sources.product,
        sources.tokenizer,
        sources.model,
        # The fixture writes the borrower drafter series next to product by default (E4B's plan
        # does not read it).
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
    """E4B's model entry — no drafter, and the same single quant `bf16` as E2B (default)."""

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
        """If the key set does not match the line's models, an undeclared model fails with a
        KeyError at assembly time."""
        assert set(gemma4_distribution.GEMMA4_DEFAULT_QUANT) == set(GEMMA4_REGULAR_MODELS)
        assert set(gemma4_distribution.GEMMA4_TOKENIZER_SERIES) == set(GEMMA4_UPSTREAM)
        assert set(GEMMA4_REGULAR_MODELS) == {"e2b", "e4b"}
        for model, quant in gemma4_distribution.GEMMA4_DEFAULT_QUANT.items():
            assert quant in gemma4_distribution.gemma4_quants(model)

    def test_the_entry_declares_no_drafter(self, tmp_path: Path) -> None:
        """The drafter role being absent from weights is the only expression of "no speculation
        possible".

        The reader rejects a load that specifies `speculative` at the resolution of weights (before
        fetching a single component).
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
        """The upstream of a drafter that is not distributed is not written as a source (base_model
        is derived from the manifest)."""
        _, manifest = _assemble_e4b(tmp_path)

        card = render_gemma4_model_card(manifest, "hdae/karume-gemma4")

        assert "google/gemma-4-E4B-it" in card
        assert "gemma-4-E2B-it-assistant" not in card
        assert "drafter head" not in card

    def test_the_card_still_names_the_e2b_drafter_when_e2b_ships_it(self, gemma4_assembled) -> None:
        """Gate against tautology — if some model carries a drafter, the drafter's upstream remains
        in base_model."""
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
            # E2B reads the same asset (it uses the asset with the source E4B wrote as it is).
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
        # E2B keeps carrying the drafter (adding E4B does not change E2B's entry).
        assert GEMMA4_DRAFTER_ROLE in manifest["models"]["e2b"]["weights"]

    def test_it_refuses_a_borrowed_tokenizer_compiled_from_another_upstream(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A combination whose shared premise (byte-identical upstreams) broke is rejected — even
        with the same vocabulary size the id sequences can split."""
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
        """An entry for a model that distributes no drafter means a drafter for another model
        slipped in."""
        sources = _build_e4b(tmp_path)
        smuggled = replace(sources, drafter=tmp_path / "series" / "gemma4-e2b-drafter")

        with pytest.raises(DistError, match="the drafter series is"):
            gemma4_plan(smuggled, "e4b")

    def test_it_refuses_an_e2b_plan_whose_drafter_series_is_missing(self, tmp_path: Path) -> None:
        """An empty entry for a model that distributes a drafter is rejected, since it would
        silently make a distribution without speculation."""
        sources = replace(_build(tmp_path), drafter=None)

        with pytest.raises(DistError, match="the drafter series is"):
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

        with pytest.raises(
            DistError, match=r"of the shared initializer is not in the lender .*product-bf16"
        ):
            gemma4_plan(sources, "e2b")

    def test_it_refuses_a_schema_5_index_with_a_scales_table(self, tmp_path: Path) -> None:
        assets = dict(fixture.ple_container_assets(storage="bf16"))
        index = json.loads(bytes(assets[PLE_INDEX_ASSET].payload))
        index["scales"] = index["values"]
        payload = json.dumps(index).encode("utf-8")
        assets[PLE_INDEX_ASSET] = AssetInput(PLE_INDEX_ROLE, len(payload), payload)
        sources = _build(tmp_path, container=fixture.product_container(assets=assets))

        with pytest.raises(DistError, match=r"unknown keys \['scales'\]"):
            gemma4_plan(sources, "e2b")

    def test_it_refuses_an_int8_index_in_the_bf16_container(self, tmp_path: Path) -> None:
        sources = _build(
            tmp_path,
            container=fixture.product_container(assets=fixture.ple_container_assets(storage="i8")),
        )

        # The schema 3 index carries a `scales` table, which a bf16 index does not have.
        with pytest.raises(DistError, match=r"product-bf16.*unknown keys \['scales'\]"):
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
