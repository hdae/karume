"""Script-level contract of the product graph script (`gemma4/export_product.py`) (the part that
needs no real weights).

The real-weight emit is manual (the same rule as the tests of the two chunk series). Only the rules
that turn into a **false PASS** when broken are pinned here:

- the PLE sidecar's split rule (fewest blocks within the limit → rows evened out), and rejection
  of a form where one token exceeds the limit
- the relayout (table-major → token-major) is **bit-identical** to the 35-table path, and
  {@link product.assert_ple_sidecar} **actually detects** a layer-shifted scale and an off-by-one
  range
- the dequantization order (`f32(i8) * per-row scale` → `* embed_scale`) matches the fake-quantized
  tables
- the product wrapper's eager equivalence (`argmax(logits)` at R = 1 matches the token-only form's
  token on every row), and R > 1 equals exactly "R one-row runs" stacked (the premise of
  speculative verify)
- {@link product.assert_ir_form_product} actually checks the product form's rules (PLE moved out,
  no argmax, the two selected-row outputs logits / hidden and their order) and the rules shared
  with the two existing series (states / layer types / residues / storage)
- one round of the series driver (sidecar → index → reference → container → provenance record,
  placed by one swap)

Only the cases that need transformers SKIP through `importorskip` (the two-job setup of
ADR 0065).
"""

from __future__ import annotations

import functools
import json
from dataclasses import replace
from itertools import pairwise
from pathlib import Path

import pytest
import torch
from container_series import write_component
from ir_fixtures import ir_container
from safetensors import safe_open
from torch import nn
from torch.export import Dim

from _shared import decode_series as shared
from _shared.container_read import read_asset, read_asset_declarations, read_stored
from gemma4 import export as gx
from gemma4 import export_decode as decode
from gemma4 import export_product as product
from gemma4 import export_token as token_only
from gemma4 import ple
from gemma4.tests.test_export import HIDDEN, PLE_DIM, TINY_SYM_MAX, VOCAB, WINDOW
from gemma4.tests.test_export_decode import (
    DECODE_LAYER_TYPES,
    OWNER_LAYERS,
    _StubSeriesTokenizer,
    _tiny_decode_config,
)
from karume.container import container_parts
from karume.convert import PRESERVED_OP_PREFIXES_WITH_ATTENTION
from karume.pipeline import export_module
from karume.ple import (
    PLE_FLOAT_INDEX_SCHEMA,
    PLE_INDEX_ASSET,
    PleError,
    ple_block_ranges,
    ple_row_bytes,
)
from karume.quantize import quantize_to_int8, round_weights_to_bf16
from karume.states import to_states_form

#: The schema-3 int8 storage: the generic PLE asset path (`ple_container_assets` /
#: `assert_ple_assets`) these tests cut, write and read back with per-row scales. The series itself
#: stores bf16 (`product.BF16_STORAGE`).
INT8 = "i8"

#: The block limit that makes the tiny series produce **2 or more** blocks (the real one gives 9 at
#: the default 32 MiB — running the driver with a limit that yields only one block never exercises
#: the block list, the index or probes that cross a block boundary).
TINY_PLE_BLOCK = ple_row_bytes(INT8, len(DECODE_LAYER_TYPES), PLE_DIM)[product.PLE_VALUES_KEY] * (
    VOCAB // 2
)

#: The fields of the series summary (order included). Changing them changes the form of real-run
#: records.
PRODUCT_SUMMARY_KEYS = [
    "dir",
    "nodes",
    "outputs",
    "initializers",
    "model_bytes",
    "ple_bytes",
    "ple_blocks",
    "ple_probe_tokens",
    "ops",
    "symbols",
    "case_lengths",
    "quantized",
    "form",
    "reference",
    "sanity",
    "greedy",
    "greedy_steps",
    "margin_min",
    "continuation",
]


# ---- the sidecar split rule -------------------------------------------------


class TestPleTokenBytes:
    def test_it_counts_the_int8_values_and_the_per_row_scales(self):
        assert product.ple_token_bytes(35, 256, INT8) == 35 * 256 + 35 * 4

    def test_the_real_bf16_table_has_two_bytes_per_value_and_no_scales(self):
        """The real one (35 layers × 256): 17,920 bytes/token — the length of ADR 0085's single
        contiguous read."""
        assert product.ple_token_bytes(35, 256, product.BF16_STORAGE) == 35 * 256 * 2 == 17_920


class TestPleBlockRanges:
    """How blocks are cut belongs to core ({@link karume.ple.ple_block_ranges}) — this checks how it
    is used."""

    @staticmethod
    def _values(tokens: int, layers: int, dim: int, block_bytes: int):
        return ple_block_ranges(
            storage=INT8,
            tokens=tokens,
            layers=layers,
            dim=dim,
            block_bytes=block_bytes,
        )[product.PLE_VALUES_KEY]

    def test_it_covers_the_vocabulary_without_gaps_or_overlaps(self):
        ranges = self._values(1000, 1, 8, 4000)

        assert ranges[0][0] == 0
        assert ranges[-1][1] == 1000
        for previous, following in pairwise(ranges):
            assert previous[1] == following[0]

    def test_it_fills_every_block_to_the_limit(self):
        """A block is filled up to the limit **in whole rows** (8 bytes a row, limit 4000 → 500 rows
        a block)."""
        rows = [stop - start for start, stop in self._values(1000, 1, 8, 4000)]

        assert rows == [500, 500]

    def test_every_block_stays_under_the_limit(self):
        row = ple_row_bytes(INT8, 35, 256)[product.PLE_VALUES_KEY]
        for start, stop in self._values(262144, 35, 256, 32 * 1024 * 1024):
            assert (stop - start) * row <= 32 * 1024 * 1024

    def test_a_single_row_over_the_limit_fails_loudly(self):
        """The split cannot get any finer (the limit is not broken silently)."""
        with pytest.raises(PleError, match="block limit"):
            self._values(4, 1, 4096, 1024)


class TestPleTableRows:
    def test_it_reads_the_row_count_from_the_split_tables(self, tiny_tables):
        """MUST: the source of truth for the row count is the split tables (the config field has
        been replaced with the probe model's 8 rows)."""
        assert product.ple_table_rows(tiny_tables, VOCAB) == VOCAB

    def test_a_probe_sized_config_field_does_not_leak_in(self, tiny_tables):
        """A regression that actually happened: reading `vocab_size_per_layer_input` writes an 8-row
        sidecar."""
        assert product.ple_table_rows(tiny_tables, VOCAB) != ple.PLE_PROBE_ROWS

    def test_a_mismatch_with_the_main_vocabulary_fails_loudly(self, tiny_tables):
        """The writer-side half of ADR 0085 decision 5 (the two counts the host loader matches)."""
        with pytest.raises(AssertionError, match="vocab rows"):
            product.ple_table_rows(tiny_tables, VOCAB + 1)

    def test_uneven_tables_fail_loudly(self, tiny_tables):
        tiny_tables[0].weight.data = torch.zeros(VOCAB - 1, PLE_DIM)

        with pytest.raises(AssertionError, match="do not share one row count"):
            product.ple_table_rows(tiny_tables, VOCAB)


class TestPleProbeTokens:
    @staticmethod
    def _ranges():
        return ple_block_ranges(storage=INT8, tokens=1000, layers=1, dim=8, block_bytes=4000)[
            product.PLE_VALUES_KEY
        ]

    def test_it_touches_both_sides_of_every_block_boundary(self):
        ranges = self._ranges()

        probe = product.ple_probe_tokens(1000, ranges)

        for start, stop in ranges:
            assert start in probe
            assert stop - 1 in probe

    def test_it_is_sorted_unique_and_inside_the_vocabulary(self):
        ranges = self._ranges()

        probe = product.ple_probe_tokens(1000, ranges)

        assert list(probe) == sorted(set(probe))
        assert all(0 <= token < 1000 for token in probe)


# ---- bit identity of the relayout -------------------------------------------


@pytest.fixture
def tiny_tables() -> nn.ModuleList:
    """A tiny fake-quantized stand-in for the 35-way PLE split (as many layers as
    {@link DECODE_LAYER_TYPES})."""
    torch.manual_seed(3)
    return nn.ModuleList(
        [
            nn.Embedding.from_pretrained(torch.randn(VOCAB, PLE_DIM), freeze=True)
            for _ in DECODE_LAYER_TYPES
        ]
    )


def _quantize_tables(tables: nn.ModuleList) -> dict[str, torch.Tensor]:
    """Fake-quantizes the same targets as {@link gx.is_int8_module} and returns the scale table."""
    from karume.quantize import fake_quant_int8

    holder = nn.Module()
    holder.per_layer = tables
    return dict(fake_quant_int8(holder, include=gx.is_int8_module).scales)


def _int8_tables(tables: nn.ModuleList) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """The fake-quantized tables as int8 values `[V, D]` and per-row scales `[V]` — the layered
    input of an int8 `product.ple_container_assets`. The scales are the fake-quant's own (an amax
    taken again could move one ulp), so `f32(i8) * scale` restores the rounded weights exactly."""
    scales = _quantize_tables(tables)
    values: list[torch.Tensor] = []
    row_scales: list[torch.Tensor] = []
    for index, table in enumerate(tables):
        scale = scales[f"{gx.PER_LAYER_PREFIX}{index}.weight"]
        values.append(quantize_to_int8(table.weight.data, scale))
        row_scales.append(scale.reshape(-1).clone())
    return values, row_scales


@pytest.fixture
def tiny_ple(tiny_tables, tmp_path):
    """Splits a tiny int8 PLE into 2 or more blocks, places it in a container and returns
    `(container path, index, probe, reference)`."""
    values, row_scales = _int8_tables(tiny_tables)
    layers = len(DECODE_LAYER_TYPES)
    embed_scale = float(PLE_DIM) ** 0.5
    ranges = ple_block_ranges(
        storage=INT8,
        tokens=VOCAB,
        layers=layers,
        dim=PLE_DIM,
        block_bytes=TINY_PLE_BLOCK,
    )[product.PLE_VALUES_KEY]
    assert len(ranges) > 1, "the tiny series must cross a block boundary too"
    probe = product.ple_probe_tokens(VOCAB, ranges)
    with torch.no_grad():
        reference = ple.per_layer_inputs(
            tiny_tables, torch.tensor([list(probe)], dtype=torch.int64), embed_scale
        )
    spill_dir = tmp_path / "staging"
    spill_dir.mkdir()
    built = product.ple_container_assets(
        values,
        row_scales,
        tokens=VOCAB,
        layers=layers,
        dim=PLE_DIM,
        embed_scale=embed_scale,
        spill_dir=spill_dir,
        storage=INT8,
        block_bytes=TINY_PLE_BLOCK,
    )
    # The same order as the real run — release the i8 data right after spilling it to the temporary
    # files (not held during export).
    values.clear()
    row_scales.clear()
    container = tmp_path / "model.krm"
    write_component(container, ir_container(mark="ple-probe", assets=built.assets))
    index = json.loads(bytes(built.assets[PLE_INDEX_ASSET].payload))
    return container, index, probe, reference, built


class TestThePleSpill:
    """The PLE data is spilled once to **temporary files in the staging directory** and removed
    before the swap.

    If the lazy reader kept holding the 35 tables in memory, the i8 data (about 2.35 GB for E2B)
    would stay resident through `torch.export` and the conversion (the script's RAM budget is the
    README's 24GB). Once spilled, the caller can move `values.clear()` back to **before** export.
    """

    def test_the_payload_lands_in_the_working_seat(self, tiny_ple) -> None:
        *_, built = tiny_ple
        layers = len(DECODE_LAYER_TYPES)

        assert len(built.spills) == 2
        sizes = {path.name: path.stat().st_size for path in built.spills}
        assert sizes == {
            ".ple.values.spill": VOCAB * layers * PLE_DIM,
            ".ple.scales.spill": VOCAB * layers * 4,
        }

    def test_the_assets_read_from_the_spill_after_the_tables_are_released(self, tiny_ple) -> None:
        """The asset payload can still be read **after** the 35 tables are released (= the data
        lives in the file, not the tables)."""
        _, index, _, _, built = tiny_ple
        block = index[product.PLE_VALUES_KEY]["blocks"][0]

        payload = built.assets[block["asset"]].payload

        assert len(bytes(payload())) == built.assets[block["asset"]].length

    def test_discard_removes_the_temporary_files(self, tiny_ple) -> None:
        """MUST: remove them before the swap (the whole staging directory is put in place, so
        leftovers would end up in the distribution)."""
        _, _, _, _, built = tiny_ple

        built.discard()

        assert [path for path in built.spills if path.exists()] == []


class TestAssertPleAssets:
    def test_the_written_bytes_rebuild_the_table_major_path_bit_for_bit(self, tiny_ple):
        container, index, probe, reference, _ = tiny_ple

        product.assert_ple_assets(container, index, probe, reference)

    def test_the_blocks_are_token_major_with_per_row_scales(self, tiny_ple):
        """The distributed form itself (ADR 0085 decision 1) — one row = `layer × dim` i8 + `layer`
        f32."""
        container, index, _, _, _ = tiny_ple
        layers = len(DECODE_LAYER_TYPES)

        assert index["storage"] == INT8
        assert index[product.PLE_VALUES_KEY]["rowBytes"] == layers * PLE_DIM
        assert index[product.PLE_SCALES_KEY]["rowBytes"] == layers * 4
        declared = read_asset_declarations(container)
        for key in (product.PLE_VALUES_KEY, product.PLE_SCALES_KEY):
            table = index[key]
            for block in table["blocks"]:
                rows = block["stop"] - block["start"]
                assert declared[block["asset"]][1] == rows * table["rowBytes"]

    def test_every_block_lands_in_a_part_of_its_own(self, tiny_ple):
        """A block that needs a range read sits **alone in a part of its own** (container-v1
        §4.2)."""
        container, index, _, _, _ = tiny_ple
        blocks = sum(len(index[key]["blocks"]) for key in ("values", "scales"))

        # On top of the weights part and the index part, each block adds one part.
        assert len(container_parts(container)) > blocks

    def test_each_block_is_decoded_only_once(self, monkeypatch, tiny_ple):
        """MUST: never read the same block twice (the probe touches both ends and the middle of a
        block).

        Re-reading per row (a cache keyed by `(asset, token)`) decodes the same asset up to 3
        times — a real block is tens of MB, so it costs acceptance wall time, yet nothing fails
        and no number drifts. Only the call count can pin it.
        """
        container, index, probe, reference, _ = tiny_ple
        # Control (not vacuous): a block holding 2 or more probe rows exists on **both the values
        # and the scales side**. Taking the maximum over both together would be satisfied by the
        # scales side alone (one block holds many rows), so the day the values path became "one row
        # per block" this would pass green without pinning duplicate reads.
        crowded = {
            key: max(
                sum(block["start"] <= token < block["stop"] for token in probe)
                for block in index[key]["blocks"]
            )
            for key in (product.PLE_VALUES_KEY, product.PLE_SCALES_KEY)
        }
        assert min(crowded.values()) >= 2, crowded
        read: list[str] = []
        original = product.read_asset

        def _counted(opened, name: str):
            read.append(name)
            return original(opened, name)

        monkeypatch.setattr(product, "read_asset", _counted)

        product.assert_ple_assets(container, index, probe, reference)

        assert sorted(read) == sorted(set(read))

    def test_a_layer_shifted_scale_is_detected(self, tiny_ple):
        """MUST: a layer-shifted scale matches in shape, type and dtype (only `torch.equal` catches
        it)."""
        container, index, probe, reference, _ = tiny_ple
        shifted = reference.clone()
        shifted[0, :, 0] = reference[0, :, 1]

        with pytest.raises(AssertionError, match="not bit-identical"):
            product.assert_ple_assets(container, index, probe, shifted)

    def test_an_off_by_one_token_range_is_detected(self, tiny_ple):
        """Shifting the range by one row yields **a valid row of another token** (the silent wrong
        value of ADR 0085 decision 5)."""
        container, index, probe, reference, _ = tiny_ple
        moved = {
            **index,
            **{
                key: {
                    **index[key],
                    "blocks": [
                        {**block, "start": block["start"] + 1} for block in index[key]["blocks"]
                    ],
                }
                for key in ("values", "scales")
            },
        }

        with pytest.raises(AssertionError):
            product.assert_ple_assets(container, moved, probe, reference)

    def test_a_probe_outside_every_range_is_detected(self, tiny_ple):
        """An index that does not cover the vocab (a probe does not slip through unchecked)."""
        container, index, probe, reference, _ = tiny_ple
        truncated = {
            **index,
            **{
                key: {**index[key], "blocks": index[key]["blocks"][:-1]}
                for key in ("values", "scales")
            },
        }

        with pytest.raises(AssertionError, match="does not cover the vocab"):
            product.assert_ple_assets(container, truncated, probe, reference)


# ---- eager equivalence ------------------------------------------------------


@pytest.fixture
def tiny_model_trio():
    """(token-only wrapper, product wrapper, 35-way PLE split) sharing the same weights.

    MUST: **share the objects** of model / tables — building them separately splits the random
    weights, and the eager-equivalence comparison loses its premise that the two should agree.
    """
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(0)
    gx.register_attention()
    config = _tiny_decode_config()
    config._attn_implementation = gx.ATTENTION_NAME
    model = transformers.Gemma4ForCausalLM(config).to(torch.float32).eval()
    tables = nn.ModuleList(
        [
            nn.Embedding.from_pretrained(torch.randn(VOCAB, PLE_DIM), freeze=True)
            for _ in DECODE_LAYER_TYPES
        ]
    )
    del model.model.embed_tokens_per_layer
    decode.swap_rope_inputs(model)
    return (
        token_only.TokenOnlyChunkWrapper(model, tables).eval(),
        product.ProductChunkWrapper(model, tables).eval(),
        tables,
    )


class TestEagerEquivalence:
    def test_the_argmax_of_the_product_logits_matches_the_token_only_exit(self, tiny_model_trio):
        """The exits differ only in the argmax (ADR 0083 decision 6) — every row is exercised.

        Being bit-identical to the token-only form at R = 1 (the binding of ordinary decode /
        prefill) shows that widening the exit for speculative verify keeps the existing path
        working.
        """
        token_form, product_form, tables = tiny_model_trio
        torch.manual_seed(1)
        ids = torch.randint(0, VOCAB, (1, 7), dtype=torch.int64)
        specs = decode.rope_specs(product_form.model.config)
        rope = decode.rope_args(specs, decode.positions_for(ids))
        stacked = ple.per_layer_inputs(tables, ids, product_form.per_layer_scale)

        with torch.no_grad():
            for row in range(int(ids.shape[1])):
                last_row = torch.tensor([row], dtype=torch.int64)
                logits, hidden = product_form(ids, *rope, stacked, last_row)
                expected = token_form(ids, *rope, last_row)
                assert tuple(logits.shape) == (1, 1, VOCAB), f"row {row}: output shape"
                assert tuple(hidden.shape) == (1, 1, HIDDEN), f"row {row}: hidden shape"
                assert int(logits[0, 0].argmax()) == int(expected[0, 0, 0]), f"row {row}: token"

    def test_multiple_rows_are_the_per_row_single_row_results_stacked(self, tiny_model_trio):
        """R > 1 equals "R one-row runs" stacked **exactly** (the premise of verify).

        If rows were not independent, the rows speculative verify scores would differ from the
        rows it actually accepts.
        """
        _, product_form, tables = tiny_model_trio
        torch.manual_seed(5)
        ids = torch.randint(0, VOCAB, (1, 7), dtype=torch.int64)
        specs = decode.rope_specs(product_form.model.config)
        rope = decode.rope_args(specs, decode.positions_for(ids))
        stacked = ple.per_layer_inputs(tables, ids, product_form.per_layer_scale)
        rows = decode.last_rows_for(ids, 3)

        with torch.no_grad():
            logits, hidden = product_form(ids, *rope, stacked, rows)
            singles = [product_form(ids, *rope, stacked, row.reshape(1)) for row in rows]

        assert tuple(logits.shape) == (1, 3, VOCAB)
        assert tuple(hidden.shape) == (1, 3, HIDDEN)
        for index, (one_logits, one_hidden) in enumerate(singles):
            # hidden is a gather of the same body forward, so it is bit-identical. For logits the M
            # of lm_head's GEMM changes between 3 and 1 and the reduction order may move, so they
            # are compared within a tolerance (the top-1 matches exactly).
            assert torch.equal(hidden[:, index : index + 1], one_hidden), f"row {index}: hidden"
            assert torch.allclose(logits[:, index : index + 1], one_logits, atol=1e-5, rtol=1e-5), (
                f"row {index}: logits"
            )
            assert int(logits[0, index].argmax()) == int(one_logits[0, 0].argmax()), (
                f"row {index}: top-1"
            )

    def test_the_second_output_is_the_hidden_the_lm_head_consumed(self, tiny_model_trio):
        """MUST: output 1 is **lm_head's input itself** (another intermediate value would also fit
        [1,R,H])."""
        _, product_form, tables = tiny_model_trio
        torch.manual_seed(6)
        ids = torch.randint(0, VOCAB, (1, 5), dtype=torch.int64)
        specs = decode.rope_specs(product_form.model.config)
        rope = decode.rope_args(specs, decode.positions_for(ids))
        stacked = ple.per_layer_inputs(tables, ids, product_form.per_layer_scale)

        with torch.no_grad():
            logits, hidden = product_form(ids, *rope, stacked, decode.last_rows_for(ids, 2))
            cap = float(product_form.model.config.final_logit_softcapping)
            replayed = torch.tanh(product_form.model.lm_head(hidden) / cap) * cap

        assert torch.equal(replayed, logits)

    def test_the_host_supplied_ple_is_what_the_graph_used_to_compute(self, tiny_model_trio):
        """When the host-supplied PLE has the same values as the 35-table path, the logits match the
        token-only form too.

        MUST: not vacuous — passing `per_layer_inputs` in another order breaks it (checked by the
        control below).
        """
        token_form, product_form, tables = tiny_model_trio
        torch.manual_seed(2)
        ids = torch.randint(0, VOCAB, (1, 5), dtype=torch.int64)
        specs = decode.rope_specs(product_form.model.config)
        rope = decode.rope_args(specs, decode.positions_for(ids))
        last_row = torch.tensor([4], dtype=torch.int64)
        stacked = ple.per_layer_inputs(tables, ids, product_form.per_layer_scale)

        with torch.no_grad():
            correct, _ = product_form(ids, *rope, stacked, last_row)
            swapped, _ = product_form(ids, *rope, stacked.flip(2), last_row)
            expected = token_form(ids, *rope, last_row)

        assert int(correct[0, 0].argmax()) == int(expected[0, 0, 0])
        assert not torch.equal(correct, swapped), (
            "the same logits come out with the layer order broken"
        )


# ---- one round with a tiny real model ---------------------------------------


class TestRowSymbolBinding:
    """The product graph exported with 2 example rows **runs at both R = 1 and R = kmax** (needs
    transformers).

    The pass criterion of step 1 itself: ordinary prefill / decode runs at R = 1, so if
    torch.export specialized `Dim(min=1)` to 1 the existing path would not run. The IR's `symbols`
    is only a list of names with no upper bound (`docs/ir-v2.md`), so this property can only be
    seen by actually running the exported ExportedProgram at other row counts.
    """

    def test_the_exported_program_runs_at_one_row_and_at_the_maximum(self, tiny_model_trio):
        _, wrapper, tables = tiny_model_trio
        torch.manual_seed(7)
        ids = torch.randint(0, VOCAB, (1, 2 * product.ROW_SYM_MAX), dtype=torch.int64)
        stacked = ple.per_layer_inputs(tables, ids, wrapper.per_layer_scale)
        specs = decode.rope_specs(wrapper.model.config)
        rope = decode.rope_args(specs, decode.positions_for(ids))
        del wrapper.per_layer
        seq = Dim(decode.SEQ_SYMBOL, min=2, max=TINY_SYM_MAX)
        rows = Dim(product.ROW_SYMBOL, min=1, max=product.ROW_SYM_MAX)

        program = torch.export.export(
            wrapper,
            (ids, *rope, stacked, decode.last_rows_for(ids, product.EXAMPLE_ROWS)),
            dynamic_shapes=(*({1: seq} for _ in range(2 + len(rope))), {0: rows}),
            strict=False,
        )

        module = program.module()
        for count in (1, product.ROW_SYM_MAX):
            with torch.no_grad():
                logits, hidden = module(ids, *rope, stacked, decode.last_rows_for(ids, count))
            assert tuple(logits.shape) == (1, count, VOCAB), f"R = {count}: logits shape"
            assert tuple(hidden.shape) == (1, count, HIDDEN), f"R = {count}: hidden shape"

    def test_the_slack_is_what_bounds_the_row_symbol(self):
        """The upper bound of R is the sliding ring's slack itself (not two numbers that can move
        separately)."""
        # verify is k draft rows + 1 bonus row, and the upper bound of k is the ring's slack (the
        # number of rows that may be rejected).
        assert product.ROW_SYM_MAX == decode.SLIDING_SLACK_ROWS + 1
        assert 1 < product.EXAMPLE_ROWS <= product.ROW_SYM_MAX


class TestExportedProductForm:
    """Runs a tiny real model through export → surgery → write with mixed quantization (needs
    transformers)."""

    @pytest.fixture
    def tiny_container(self, tiny_model_trio, tmp_path):
        _, wrapper, tables = tiny_model_trio
        int8, int4, scales = gx.quantize_wrapper(wrapper)
        torch.manual_seed(4)
        ids = torch.randint(0, VOCAB, (1, WINDOW + 3), dtype=torch.int64)
        stacked = ple.per_layer_inputs(tables, ids, wrapper.per_layer_scale)
        last_row = decode.last_rows_for(ids, product.EXAMPLE_ROWS)
        specs = decode.rope_specs(wrapper.model.config)
        rope = decode.rope_args(specs, decode.positions_for(ids))
        # PLE is now an input the host supplies, so drop the module before export (the same order as
        # the script).
        del wrapper.per_layer
        seq = Dim(decode.SEQ_SYMBOL, min=2, max=TINY_SYM_MAX)
        rows = Dim(product.ROW_SYMBOL, min=1, max=product.ROW_SYM_MAX)
        graph, tensors = export_module(
            wrapper,
            (ids, *rope, stacked, last_row),
            dynamic_shapes=(*({1: seq} for _ in range(2 + len(rope))), {0: rows}),
            symbol_names=(decode.SEQ_SYMBOL, product.ROW_SYMBOL),
            preserved=PRESERVED_OP_PREFIXES_WITH_ATTENTION,
        )
        config = wrapper.model.config
        surgical = to_states_form(graph, decode.states_plan(graph, config))
        verified = decode._write_container(
            surgical,
            tensors,
            tmp_path / gx.MODEL_FILE,
            graph_name="tiny",
            weight_dtype="i8",
            weight_scales=scales,
            weight_dtype_overrides=dict.fromkeys(int4.scales, "i4"),
        )
        storage = {
            "i8": len(int8.scales) - len(DECODE_LAYER_TYPES),
            "i4": len(int4.scales),
        }
        return verified, config, storage

    def test_the_container_is_a_verified_product_graph(self, tiny_container):
        verified, config, storage = tiny_container

        form = product.assert_ir_form_product(verified, config, storage, VOCAB)

        assert [spec.name for spec in verified.inputs] == [
            decode.INPUT_IDS,
            *decode.ROPE_INPUTS,
            product.PER_LAYER_INPUTS,
            decode.TOKEN_ONLY_LAST_ROW,
        ]
        assert len(verified.outputs) == 2
        assert "argmax" not in verified.required_ops
        assert form["attention_nodes"] == len(DECODE_LAYER_TYPES)
        assert form["state_append_nodes"] == 2 * OWNER_LAYERS
        assert form["logits"] == [1, product.ROW_SYMBOL, VOCAB]
        assert form["hidden"] == [1, product.ROW_SYMBOL, HIDDEN]

    def test_the_row_count_stays_a_symbol_the_runtime_can_bind_to_one(self, tiny_container):
        """MUST: R stays a symbol (exporting with 2 example rows does not make the row count a
        constant).

        If R were baked in, ordinary decode (R = 1) could not run. The only binding point is
        `last_row[R]`, and the IR's `symbols` holds only a list of names (no upper bound is
        recorded), so the runtime can bind it to 1 or to 8.
        """
        verified, _, _ = tiny_container

        last_row = next(spec for spec in verified.inputs if spec.name == decode.TOKEN_ONLY_LAST_ROW)
        assert list(last_row.shape) == [product.ROW_SYMBOL]
        assert sorted(verified.symbols) == sorted(
            {decode.SEQ_SYMBOL, decode.CAPACITY_SYMBOL, product.ROW_SYMBOL}
        )

    def test_the_sliding_slots_carry_the_speculative_slack(self, tiny_container):
        """The physical row count of a sliding slot is `window + slack` (the condition for rejected
        rows not to overwrite the live window)."""
        verified, config, _ = tiny_container

        capacities = [list(slot.shape)[2] for slot in verified.states.values()]
        baked = [value for value in capacities if not isinstance(value, str)]

        assert baked, "no sliding slot (the one whose capacity is a number) found"
        assert set(baked) == {WINDOW + decode.SLIDING_SLACK_ROWS}
        assert decode.sliding_capacity(config) == WINDOW + decode.SLIDING_SLACK_ROWS

    def test_swapped_outputs_are_detected(self, tiny_container):
        """Swapped outputs are rejected.

        The tiny subject has `VOCAB == HIDDEN`, so **the widths cannot tell them apart** (the real
        model has 262,144 and 2,048, and the width check fails first). It still fails because a
        structural check walks output 0's ancestors back to lm_head — the hidden side dead-ends at
        the row-selection `embedding`.
        """
        verified, config, storage = tiny_container
        swapped = replace(verified, outputs=list(reversed(verified.outputs)))

        with pytest.raises(AssertionError, match=r"no lm_head \(linear\)"):
            product.assert_ir_form_product(swapped, config, storage, VOCAB)

    def test_the_ple_tables_are_gone_from_the_container(self, tiny_container):
        """The residency reduction itself (ADR 0085) — no embedding that looks up a PLE table
        remains."""
        verified, _, _ = tiny_container

        weights = [
            list(verified.values[node.ins[0]].shape)
            for node in verified.nodes
            if node.op == "embedding"
        ]

        assert [VOCAB, PLE_DIM] not in weights
        # embedding: 1 main embedding + 1 row selection (the RoPE table lookup is gone).
        assert len(weights) == 2

    def test_the_decode_form_check_rejects_the_product_graph(self, tiny_container):
        """The form checks of the two existing series reject the product form (pins that the three
        forms do not mix)."""
        verified, config, storage = tiny_container

        with pytest.raises(AssertionError, match="graph input"):
            decode.assert_ir_form_decode(verified, config, storage, token_only=True)

    def test_a_residual_ple_table_is_detected(self, tiny_container):
        """Rejects a form where an embedding looking up a PLE table remains (wiring that passes both
        the input and the table)."""
        from karume.ir import IrInitializer, IrNode, IrStorage, IrValue

        verified, config, storage = tiny_container
        polluted = replace(
            verified,
            initializers={
                **verified.initializers,
                "per_layer.0.weight": IrInitializer(
                    tensor="per_layer.0.weight", storage=IrStorage(dtype="i8")
                ),
            },
            values={
                **verified.values,
                "per_layer.0.weight": IrValue(dtype="f32", shape=[VOCAB, PLE_DIM]),
                "per_layer.0.out": IrValue(dtype="f32", shape=[1, decode.SEQ_SYMBOL, PLE_DIM]),
            },
            nodes=[
                *verified.nodes,
                IrNode(
                    op="embedding",
                    ins=["per_layer.0.weight", decode.INPUT_IDS],
                    outs=["per_layer.0.out"],
                    attrs={},
                ),
            ],
        )

        with pytest.raises(AssertionError, match="PLE table shape"):
            product.assert_ir_form_product(polluted, config, storage, VOCAB)

    def test_an_argmax_exit_is_detected(self, tiny_container):
        """An exit that regressed to argmax (no room left for sampling — ADR 0083 decision 6)."""
        from karume.ir import IrNode

        verified, config, storage = tiny_container
        with_argmax = replace(
            verified,
            nodes=[
                *verified.nodes,
                IrNode(op="argmax", ins=[verified.outputs[0]], outs=["token"], attrs={"dim": -1}),
            ],
        )

        with pytest.raises(AssertionError, match="argmax"):
            product.assert_ir_form_product(with_argmax, config, storage, VOCAB)

    def test_a_full_row_lm_head_is_detected(self, tiny_container):
        """The row selection moved **after** lm_head (the values match, so only form and structure
        show it).

        lm_head's input is the hidden exit itself, so widening the row axis to the chunk rows `M`
        makes output 1's declaration disagree first (it is no longer `[1, R, H]`). A row selection
        moved after lm_head always passes through here, so the regression stops at this first
        check — a form whose declaration is right but whose wiring is broken is caught by the
        structural check of {@link TestExportedProductForm.test_swapped_outputs_are_detected}.
        """
        verified, config, storage = tiny_container
        producer = {out: node for node in verified.nodes for out in node.outs}
        linear = next(
            node
            for node in verified.nodes
            if node.op == "linear"
            and list(verified.values[node.outs[0]].shape)[:2] == [1, product.ROW_SYMBOL]
        )
        widened = replace(
            verified,
            values={
                **verified.values,
                linear.ins[0]: replace(
                    verified.values[linear.ins[0]],
                    shape=[1, decode.SEQ_SYMBOL, verified.values[linear.ins[0]].shape[2]],
                ),
            },
        )
        assert producer.get(linear.ins[0]) is not None

        with pytest.raises(AssertionError, match=r"declared shape of output 1"):
            product.assert_ir_form_product(widened, config, storage, VOCAB)


# ---- one round of the series driver -----------------------------------------


@pytest.fixture
def tiny_product_series(monkeypatch, tmp_path_factory):
    """The set of substitutions that runs `export_product.export_series` once without real assets.

    Only **where the materials come from** (model, cases, tokenizer), the expectation tiny random
    weights cannot meet (the top-1 at the last position) and the sidecar limit that a tiny vocab
    would turn into a single block are replaced. The path sidecar → check → export → surgery →
    write → form check → publish runs for real (the same substitution policy as
    `test_export_decode.tiny_series`).

    The checkpoint is bf16 (the tiny model and tables rounded to bf16 as the loader returns them —
    the real checkpoint is BF16); the loaded 2-D weights and PLE tables (`ple.<layer>`) are
    recorded, generation defaults sit in the checkpoint directory and the margin floor is 0.
    Returns `(cases, checkpoint, weights, seen)`.
    """
    transformers = pytest.importorskip("transformers")

    def materials(model_dir):
        torch.manual_seed(0)
        gx.register_attention()
        config = _tiny_decode_config()
        config._attn_implementation = gx.ATTENTION_NAME
        model = transformers.Gemma4ForCausalLM(config).to(torch.float32).eval()
        tables = nn.ModuleList(
            [
                nn.Embedding.from_pretrained(torch.randn(VOCAB, PLE_DIM), freeze=True)
                for _ in DECODE_LAYER_TYPES
            ]
        )
        # MUST: return it, as the real loader does, with `vocab_size_per_layer_input` replaced by
        # **the probe model's row count** (the implementation of `gx.load_model_and_tables` — the
        # 35-way split is the PLE's only source of truth). Returning the plain VOCAB would hide from
        # the driver a regression where the script reads this field from the config.
        model.config.vocab_size_per_layer_input = ple.PLE_PROBE_ROWS
        return model, tables

    monkeypatch.setattr(gx, "load_model_and_tables", materials)

    from gemma4 import provenance
    from gemma4.tests.test_export_decode import TINY_CASE_NAMES, _StubSeriesTokenizer

    torch.manual_seed(1)
    cases = tuple(
        (name, torch.randint(0, VOCAB, (1, WINDOW + 3 - index), dtype=torch.int64))
        for index, name in enumerate(TINY_CASE_NAMES)
    )
    checkpoint = tmp_path_factory.mktemp("checkpoint")
    for name in provenance.FINGERPRINT_FILES:
        (checkpoint / name).write_bytes(name.encode())
    monkeypatch.setattr(gx, "build_cases", lambda model_dir, sym_max, window: cases)
    monkeypatch.setattr(gx, "load_tokenizer", lambda model_dir: _StubSeriesTokenizer())
    # Make the tiny series split the PLE into 2 or more blocks too (crossing a block boundary at
    # least once).
    monkeypatch.setattr(
        product,
        "ple_block_ranges",
        functools.partial(ple_block_ranges, block_bytes=TINY_PLE_BLOCK),
    )
    monkeypatch.setattr(
        product,
        "ple_container_assets",
        functools.partial(product.ple_container_assets, block_bytes=TINY_PLE_BLOCK),
    )
    seen: dict[str, dict[str, int]] = {}

    def record(greedy, expected, labels):
        seen["greedy"] = dict(greedy)
        return {"stub": "ok"}

    monkeypatch.setattr(gx, "_sanity", record)

    load = gx.load_model_and_tables
    weights: dict[str, torch.Tensor] = {}

    def bf16_materials(model_dir, **options):
        model, tables = load(model_dir, **options)
        round_weights_to_bf16(model)
        round_weights_to_bf16(tables)
        weights.update(
            {
                f"model.{name}": parameter.detach().clone()
                for name, parameter in model.named_parameters()
                if parameter.dim() == 2
            }
        )
        weights.update(
            {f"ple.{layer}": table.weight.detach().clone() for layer, table in enumerate(tables)}
        )
        return model, tables

    monkeypatch.setattr(gx, "load_model_and_tables", bf16_materials)
    monkeypatch.setattr(decode, "MARGIN_FLOOR", 0.0)
    (checkpoint / "generation_config.json").write_text(
        json.dumps({"temperature": 1.0, "top_k": 64, "top_p": 0.95}), encoding="utf-8"
    )
    return cases, checkpoint, weights, seen


#: The tiny series' greedy steps.
STEPS = 2


class TestExportSeries:
    def test_it_publishes_the_container_the_probe_the_goldens_and_the_record(
        self, tiny_product_series, tmp_path
    ):
        cases, checkpoint, _weights, seen = tiny_product_series
        out_dir = tmp_path / "gemma4-e2b-product-bf16"

        summary = product.export_series(checkpoint, out_dir, sym_max=TINY_SYM_MAX, steps=STEPS)

        assert summary["ple_blocks"][product.PLE_VALUES_KEY] > 1, "must cross a block boundary"
        # The PLE is an asset of the container, so it does not appear as a separate file in the
        # series.
        greedy = [f"{shared.GREEDY_PREFIX}{name}{shared.GREEDY_SUFFIX}" for name, _ in cases]
        assert sorted(path.name for path in out_dir.iterdir()) == sorted(
            [product.PLE_PROBE_FILE, provenance_file(), *greedy, *_container_files(out_dir)]
        )
        assert list(summary) == PRODUCT_SUMMARY_KEYS
        assert summary["outputs"] == 2
        assert summary["greedy"] == greedy
        assert summary["greedy_steps"] == STEPS
        assert set(seen["greedy"]) == {name for name, _ in cases}
        # Neither the staging nor the backup directory remains (cleaning up after the swap is the
        # job of the core primitive).
        assert list(tmp_path.iterdir()) == [out_dir]

    def test_every_linear_and_the_embedding_hold_the_loaded_bits_as_bf16(
        self, tiny_product_series, tmp_path
    ):
        """The checkpoint's values are stored as they are: no i8 / i4 weight, and every bf16
        tensor decodes to the loaded weight bit for bit."""
        _, checkpoint, weights, _ = tiny_product_series
        out_dir = tmp_path / "series"

        summary = product.export_series(checkpoint, out_dir, sym_max=TINY_SYM_MAX, steps=STEPS)

        stored = read_stored(out_dir / gx.MODEL_FILE)
        compressed = {key: entry for key, entry in stored.items() if entry.layout != "f32"}
        assert {entry.layout for entry in compressed.values()} == {product.BF16_STORAGE}
        assert len(compressed) == summary["quantized"][product.BF16_STORAGE]
        assert summary["form"]["storage"].get("i8", 0) == 0
        assert summary["form"]["storage"].get("i4", 0) == 0
        # The tied pair is one tensor under whichever of its two names the trace kept.
        embedding, head = f"{gx.EMBED_TOKENS_MODULE}.weight", f"{gx.LM_HEAD_MODULE}.weight"
        assert any(key in compressed for key in (embedding, head))
        for key, entry in compressed.items():
            source = embedding if key == head else key
            decoded = (
                torch.frombuffer(bytearray(entry.payload), dtype=torch.int16)
                .view(torch.bfloat16)
                .to(torch.float32)
                .reshape(entry.shape)
            )
            assert torch.equal(decoded, weights[source]), key

    def test_the_ple_is_schema_5_bf16_and_holds_the_loaded_tables_bit_for_bit(
        self, tiny_product_series, tmp_path
    ):
        """Every token row of the stored table (not only the probe) is the loaded tables' bf16 bits,
        and the probe check passes on the published container."""
        _, checkpoint, weights, _ = tiny_product_series
        out_dir = tmp_path / "series"

        product.export_series(checkpoint, out_dir, sym_max=TINY_SYM_MAX, steps=STEPS)

        index = _published_index(out_dir)
        assert index["schema"] == PLE_FLOAT_INDEX_SCHEMA == 5
        assert index["storage"] == product.BF16_STORAGE
        assert "scales" not in index
        assert index["tokens"] == VOCAB
        assert index["layers"] == len(DECODE_LAYER_TYPES)
        assert index["dim"] == PLE_DIM
        assert index["embedScale"] == pytest.approx(float(PLE_DIM) ** 0.5)
        blocks = index[product.PLE_VALUES_KEY]["blocks"]
        assert blocks[0]["start"] == 0
        assert blocks[-1]["stop"] == VOCAB
        roles = {
            role for role, _length in read_asset_declarations(out_dir / gx.MODEL_FILE).values()
        }
        assert "ple-scales" not in roles
        layers, dim = int(index["layers"]), int(index["dim"])
        stored = torch.cat(
            [
                torch.frombuffer(
                    bytearray(read_asset(out_dir / gx.MODEL_FILE, block["asset"])),
                    dtype=torch.int16,
                )
                for block in index["values"]["blocks"]
            ]
        ).view(torch.bfloat16)
        loaded = torch.stack([weights[f"ple.{layer}"] for layer in range(layers)], dim=1)
        assert torch.equal(stored.to(torch.float32).reshape(-1, layers, dim), loaded)
        with safe_open(str(out_dir / product.PLE_PROBE_FILE), framework="pt") as handle:
            probe = [int(token) for token in handle.get_tensor(product.PROBE_TOKENS_KEY)]
            expected = handle.get_tensor(product.PROBE_INPUTS_KEY)
        product.assert_ple_assets(out_dir / gx.MODEL_FILE, index, probe, expected)

    def test_the_record_names_the_series_own_goldens(self, tiny_product_series, tmp_path):
        cases, checkpoint, _, _ = tiny_product_series
        out_dir = tmp_path / "gemma4-e4b-product-bf16"

        product.export_series(checkpoint, out_dir, sym_max=TINY_SYM_MAX, steps=STEPS)

        record = json.loads((out_dir / provenance_file()).read_text(encoding="utf-8"))
        assert record["series"] == out_dir.name
        assert record["reference"]["series"] == out_dir.name
        assert record["reference"]["suppressTokens"] == []
        assert sorted(record["reference"]["goldens"]) == sorted(
            f"{shared.GREEDY_PREFIX}{name}{shared.GREEDY_SUFFIX}" for name, _ in cases
        )

    def test_the_goldens_are_the_stored_model_s(self, tiny_product_series, tmp_path):
        """Each golden is the greedy continuation its file records."""
        cases, checkpoint, _, _ = tiny_product_series
        out_dir = tmp_path / "series"

        summary = product.export_series(checkpoint, out_dir, sym_max=TINY_SYM_MAX, steps=STEPS)

        for name, ids in cases:
            path = out_dir / f"{shared.GREEDY_PREFIX}{name}{shared.GREEDY_SUFFIX}"
            with safe_open(str(path), framework="pt") as handle:
                assert torch.equal(handle.get_tensor(shared.PROMPT_KEY).to(torch.int64), ids[0])
                expected = handle.get_tensor(shared.EXPECTED_KEY).tolist()
            assert len(expected) == STEPS
            assert summary["continuation"][name] == _StubSeriesTokenizer().decode(expected)


class TestBf16PleTables:
    def test_it_keeps_the_bits_and_releases_each_f32_table(self):
        source = torch.tensor([[1.5, -0.25], [3.0e-3, 0.0]]).to(torch.bfloat16).to(torch.float32)
        tables = [torch.nn.Embedding.from_pretrained(source.clone()) for _ in range(2)]

        values = product.bf16_ple_tables(tables)

        assert [value.dtype for value in values] == [torch.int16, torch.int16]
        assert all(
            torch.equal(value.view(torch.bfloat16).to(torch.float32), source) for value in values
        )
        assert all(table.weight.numel() == 0 for table in tables)

    def test_a_value_bf16_cannot_hold_fails(self):
        # 1 + 2^-10 needs 10 mantissa bits; bf16 has 7.
        table = torch.nn.Embedding.from_pretrained(torch.tensor([[1.0 + 2**-10, 1.0]]))

        with pytest.raises(AssertionError, match="PLE table 0 is not exact in bf16"):
            product.bf16_ple_tables([table])


def provenance_file() -> str:
    """The provenance record's file name (the spelling's source of truth is
    {@link gemma4.provenance})."""
    from gemma4 import provenance

    return provenance.REFERENCE_FILE


def _container_files(out_dir: Path) -> list[str]:
    """The part file names of the placed container (their count is set by the bytes written)."""
    return [path.name for path in container_parts(out_dir / gx.MODEL_FILE)]


def _published_index(out_dir: Path) -> dict:
    """Reads the asset `ple_index` of the placed container (taken from the distributed form
    itself)."""
    return json.loads(read_asset(out_dir / gx.MODEL_FILE, PLE_INDEX_ASSET))


def test_the_quantized_values_survive_a_round_trip_through_the_blocks(tiny_tables, tmp_path):
    """The relayout of every token and layer is kept as i8 bytes (beyond the probe too)."""
    layers = len(DECODE_LAYER_TYPES)
    values, row_scales = _int8_tables(tiny_tables)
    expected = [value.clone() for value in values]
    spill_dir = tmp_path / "staging"
    spill_dir.mkdir()
    built = product.ple_container_assets(
        values,
        row_scales,
        tokens=VOCAB,
        layers=layers,
        dim=PLE_DIM,
        embed_scale=float(PLE_DIM) ** 0.5,
        spill_dir=spill_dir,
        storage=INT8,
        block_bytes=TINY_PLE_BLOCK,
    )
    values.clear()
    row_scales.clear()
    container = tmp_path / "model.krm"
    write_component(container, ir_container(mark="ple-roundtrip", assets=built.assets))
    index = json.loads(bytes(built.assets[PLE_INDEX_ASSET].payload))

    for block in index[product.PLE_VALUES_KEY]["blocks"]:
        start, stop = block["start"], block["stop"]
        raw = read_asset(container, str(block["asset"]))
        rows = torch.frombuffer(bytearray(raw), dtype=torch.int8).reshape(
            stop - start, layers, PLE_DIM
        )
        for layer in range(layers):
            assert torch.equal(rows[:, layer], expected[layer][start:stop]), (
                f"block {block['asset']}: layer {layer}"
            )
