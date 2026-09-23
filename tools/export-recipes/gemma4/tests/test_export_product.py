"""製品グラフ台本（`gemma4/export_product.py`）の台本レベルの約束事（実重み不要分）。

実重みの emit は手動（chunk 系列 2 本のテストと同じ規律）。ここで固定するのは、壊れると
**偽 PASS** になる側の規律だけ:

- PLE sidecar の分割規則（上限内の最小本数 → 行数の均し）と、1 token が上限を超える形の拒否
- 再配置（table-major → token-major）が 35 表経路と**ビット一致**すること、および
  {@link product.assert_ple_sidecar} が scale の層ずれ・範囲の off-by-one を**実際に検出**すること
- 逆量子化の順序（`f32(i8) * per-row scale` → `* embed_scale`）が fake-quant 済みの表と一致すること
- 製品ラッパの eager 同値（R = 1 の `argmax(logits)` が token-only 形の token と全行で一致）と、
  R > 1 が「R 回の 1 行 run」を積んだものと厳密一致すること（投機 verify の前提）
- {@link product.assert_ir_form_product} が製品形の門（PLE 外出し・argmax 不在・選択行の
  logits / hidden 2 本と順序）と、既存 2 系列と共有の規律（states / 層種別 / 残骸 / 格納）を
  実際に見ること
- 系列 driver の一周（sidecar → 索引 → 参照 → コンテナ → 出所記録を 1 回の据え替えで置く）

transformers を要するケースだけ `importorskip` で SKIP する（ADR 0065 の 2 job 構成）。
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

from _shared.container_read import read_asset, read_asset_declarations
from gemma4 import export as gx
from gemma4 import export_decode as decode
from gemma4 import export_product as product
from gemma4 import export_token as token_only
from gemma4 import ple
from gemma4.tests.test_export import HIDDEN, PLE_DIM, TINY_SYM_MAX, VOCAB, WINDOW
from gemma4.tests.test_export_decode import (
    DECODE_LAYER_TYPES,
    OWNER_LAYERS,
    _tiny_decode_config,
)
from karume.container import container_parts
from karume.convert import PRESERVED_OP_PREFIXES_WITH_ATTENTION
from karume.pipeline import export_module
from karume.ple import PLE_INDEX_ASSET, PleError, ple_block_ranges, ple_row_bytes
from karume.quantize import quantize_to_int8
from karume.states import to_states_form

#: tiny 系列で **2 本以上**の block を作らせる block 上限（実物は既定の 32 MiB で 9 本 —
#: 1 本しか出ない上限で driver を回すと、block 列・索引・probe の block 跨ぎが 1 度も踏まれない）。
TINY_PLE_BLOCK = ple_row_bytes(product.PLE_STORAGE, len(DECODE_LAYER_TYPES), PLE_DIM)[
    product.PLE_VALUES_KEY
] * (VOCAB // 2)

#: 系列の要約の欄（順序込み）。ここが変わると実走の記録の形が変わる。
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
]


# ---- sidecar の分割規則 ----------------------------------------------------


class TestPleTokenBytes:
    def test_it_counts_the_int8_values_and_the_per_row_scales(self):
        assert product.ple_token_bytes(35, 256) == 35 * 256 + 35 * 4

    def test_the_real_model_lands_on_the_recorded_figure(self):
        """実物（35 層 × 256）の 9,100 バイト/token — ADR 0085 の連続 1 読みの長さ。"""
        assert product.ple_token_bytes(35, 256) == 9100


class TestPleBlockRanges:
    """block の切り方は core（{@link karume.ple.ple_block_ranges}）が持つ — ここは使い方の門。"""

    @staticmethod
    def _values(tokens: int, layers: int, dim: int, block_bytes: int):
        return ple_block_ranges(
            storage=product.PLE_STORAGE,
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
        """block は**行の倍数**で上限まで詰める（1 行 8 バイト・上限 4000 → 500 行/本）。"""
        rows = [stop - start for start, stop in self._values(1000, 1, 8, 4000)]

        assert rows == [500, 500]

    def test_every_block_stays_under_the_limit(self):
        row = ple_row_bytes(product.PLE_STORAGE, 35, 256)[product.PLE_VALUES_KEY]
        for start, stop in self._values(262144, 35, 256, 32 * 1024 * 1024):
            assert (stop - start) * row <= 32 * 1024 * 1024

    def test_a_single_row_over_the_limit_fails_loudly(self):
        """分割の粒度がこれ以上細かくできない形（黙って上限を破らない）。"""
        with pytest.raises(PleError, match="block 上限"):
            self._values(4, 1, 4096, 1024)


class TestPleTableRows:
    def test_it_reads_the_row_count_from_the_split_tables(self, tiny_tables):
        """MUST: 行数の正本は分割表（config の欄は検査席の 8 行へ差し替え済み）。"""
        assert product.ple_table_rows(tiny_tables, VOCAB) == VOCAB

    def test_a_probe_sized_config_field_does_not_leak_in(self, tiny_tables):
        """実際に踏んだ回帰: `vocab_size_per_layer_input` を読むと 8 行の sidecar が書かれる。"""
        assert product.ple_table_rows(tiny_tables, VOCAB) != ple.PLE_PROBE_ROWS

    def test_a_mismatch_with_the_main_vocabulary_fails_loudly(self, tiny_tables):
        """ADR 0085 決定 5 の書き手側の半分（ホスト loader が突き合わせる 2 つの一致）。"""
        with pytest.raises(AssertionError, match="vocab 行数"):
            product.ple_table_rows(tiny_tables, VOCAB + 1)

    def test_uneven_tables_fail_loudly(self, tiny_tables):
        tiny_tables[0].weight.data = torch.zeros(VOCAB - 1, PLE_DIM)

        with pytest.raises(AssertionError, match="行数が揃っていない"):
            product.ple_table_rows(tiny_tables, VOCAB)


class TestPleProbeTokens:
    @staticmethod
    def _ranges():
        return ple_block_ranges(
            storage=product.PLE_STORAGE, tokens=1000, layers=1, dim=8, block_bytes=4000
        )[product.PLE_VALUES_KEY]

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


# ---- 再配置のビット一致 ----------------------------------------------------


@pytest.fixture
def tiny_tables() -> nn.ModuleList:
    """fake-quant 済みの tiny な PLE 35 分割相当（層数は {@link DECODE_LAYER_TYPES} と同数）。"""
    torch.manual_seed(3)
    return nn.ModuleList(
        [
            nn.Embedding.from_pretrained(torch.randn(VOCAB, PLE_DIM), freeze=True)
            for _ in DECODE_LAYER_TYPES
        ]
    )


def _quantize_tables(tables: nn.ModuleList) -> dict[str, torch.Tensor]:
    """{@link gx.is_int8_module} と同じ対象で fake-quant を掛け、scale 台帳を返す。"""
    from karume.quantize import fake_quant_int8

    holder = nn.Module()
    holder.per_layer = tables
    return dict(fake_quant_int8(holder, include=gx.is_int8_module).scales)


class TestQuantizedPleTables:
    def test_the_int8_values_and_scales_reproduce_the_fake_quant_weights(self, tiny_tables):
        """`f32(i8) * scale` が丸め済みの重みを**厳密に**復元する（ADR 0019 の ±127 論証）。"""
        scales = _quantize_tables(tiny_tables)
        rounded = [table.weight.data.clone() for table in tiny_tables]

        values, row_scales = product.quantized_ple_tables(tiny_tables, scales)

        for index, weight in enumerate(rounded):
            restored = values[index].to(torch.float32) * row_scales[index].unsqueeze(-1)
            assert torch.equal(restored, weight), f"表 {index}"

    def test_it_releases_the_float32_bodies(self, tiny_tables):
        """MUST: 1 表ずつ f32 を手放す（ピーク RAM の根拠 — 実物では 9.4GB）。"""
        scales = _quantize_tables(tiny_tables)

        product.quantized_ple_tables(tiny_tables, scales)

        assert all(table.weight.data.numel() == 0 for table in tiny_tables)

    def test_a_missing_scale_fails_loudly(self, tiny_tables):
        with pytest.raises(AssertionError, match="scale 台帳キー"):
            product.quantized_ple_tables(tiny_tables, {})


@pytest.fixture
def tiny_ple(tiny_tables, tmp_path):
    """tiny な PLE を 2 block 以上へ割って容器へ据え、`(容器 path, 索引, probe, 参照)` を返す。"""
    scales = _quantize_tables(tiny_tables)
    layers = len(DECODE_LAYER_TYPES)
    embed_scale = float(PLE_DIM) ** 0.5
    ranges = ple_block_ranges(
        storage=product.PLE_STORAGE,
        tokens=VOCAB,
        layers=layers,
        dim=PLE_DIM,
        block_bytes=TINY_PLE_BLOCK,
    )[product.PLE_VALUES_KEY]
    assert len(ranges) > 1, "tiny 系列でも block 跨ぎを踏ませる"
    probe = product.ple_probe_tokens(VOCAB, ranges)
    with torch.no_grad():
        reference = ple.per_layer_inputs(
            tiny_tables, torch.tensor([list(probe)], dtype=torch.int64), embed_scale
        )
    values, row_scales = product.quantized_ple_tables(tiny_tables, scales)
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
        block_bytes=TINY_PLE_BLOCK,
    )
    # 実物と同じ順序 — 一時ファイルへ落とした直後に i8 実体を手放す（export 中に持たない）。
    values.clear()
    row_scales.clear()
    container = tmp_path / "model.krm"
    write_component(container, ir_container(mark="ple-probe", assets=built.assets))
    index = json.loads(bytes(built.assets[PLE_INDEX_ASSET].payload))
    return container, index, probe, reference, built


class TestThePleSpill:
    """PLE の実体は**作業席の一時ファイル**へ 1 度だけ落ち、据え替えの前に消える。

    遅延読み口がメモリ上の 35 表を掴んだままだと、i8 の実体（E2B で約 2.35 GB）が
    `torch.export` と変換の間ずっと常駐する（台本の RAM 予算は README の 24GB）。落として
    しまえば呼び手は `values.clear()` を export の**前**へ戻せる。
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
        """35 表を手放した**後**でも資産の payload が引ける（= 実体は表ではなくファイル）。"""
        _, index, _, _, built = tiny_ple
        block = index[product.PLE_VALUES_KEY]["blocks"][0]

        payload = built.assets[block["asset"]].payload

        assert len(bytes(payload())) == built.assets[block["asset"]].length

    def test_discard_removes_the_temporary_files(self, tiny_ple) -> None:
        """MUST: 据え替えの前に消す（作業席ごと据わるので、残すと配布物に混ざる）。"""
        _, _, _, _, built = tiny_ple

        built.discard()

        assert [path for path in built.spills if path.exists()] == []


class TestAssertPleAssets:
    def test_the_written_bytes_rebuild_the_table_major_path_bit_for_bit(self, tiny_ple):
        container, index, probe, reference, _ = tiny_ple

        product.assert_ple_assets(container, index, probe, reference)

    def test_the_blocks_are_token_major_with_per_row_scales(self, tiny_ple):
        """配布形そのもの（ADR 0085 決定 1）— 1 行 = `layer × dim` の i8 + `layer` 本の f32。"""
        container, index, _, _, _ = tiny_ple
        layers = len(DECODE_LAYER_TYPES)

        assert index["storage"] == product.PLE_STORAGE
        assert index[product.PLE_VALUES_KEY]["rowBytes"] == layers * PLE_DIM
        assert index[product.PLE_SCALES_KEY]["rowBytes"] == layers * 4
        declared = read_asset_declarations(container)
        for key in (product.PLE_VALUES_KEY, product.PLE_SCALES_KEY):
            table = index[key]
            for block in table["blocks"]:
                rows = block["stop"] - block["start"]
                assert declared[block["asset"]][1] == rows * table["rowBytes"]

    def test_every_block_lands_in_a_part_of_its_own(self, tiny_ple):
        """区間読みを要する block は**専用 part に単独**（container-v1 §4.2）。"""
        container, index, _, _, _ = tiny_ple
        blocks = sum(len(index[key]["blocks"]) for key in ("values", "scales"))

        # 重みの part + 索引の part に加えて、block 1 本につき part が 1 つ増える。
        assert len(container_parts(container)) > blocks

    def test_a_layer_shifted_scale_is_detected(self, tiny_ple):
        """MUST: scale の層ずれは形も型も dtype も合う（`torch.equal` でしか捕まらない）。"""
        container, index, probe, reference, _ = tiny_ple
        shifted = reference.clone()
        shifted[0, :, 0] = reference[0, :, 1]

        with pytest.raises(AssertionError, match="ビット一致しない"):
            product.assert_ple_assets(container, index, probe, shifted)

    def test_an_off_by_one_token_range_is_detected(self, tiny_ple):
        """範囲を 1 行ずらすと**別 token の有効な行**が出る（ADR 0085 決定 5 の沈黙誤値）。"""
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
        """索引が vocab を覆っていない形（probe が無検査で素通りしない）。"""
        container, index, probe, reference, _ = tiny_ple
        truncated = {
            **index,
            **{
                key: {**index[key], "blocks": index[key]["blocks"][:-1]}
                for key in ("values", "scales")
            },
        }

        with pytest.raises(AssertionError, match="覆っていない"):
            product.assert_ple_assets(container, truncated, probe, reference)


# ---- eager 同値 ------------------------------------------------------------


@pytest.fixture
def tiny_model_trio():
    """同じ重みを共有する (token-only ラッパ, 製品ラッパ, PLE 35 分割)。

    MUST: model / tables の**実体を共有**する — 別々に組むと乱数重みが割れて eager 同値の
    突合が「一致すべき前提」を失う。
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
        """出口の差は argmax の有無だけ（ADR 0083 決定 6）— 全行を踏む。

        R = 1（通常の decode / prefill の束縛）で token-only 形とビット同一であることが、
        投機 verify のために出口を広げても既存経路が動くことの実証。
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
                assert tuple(logits.shape) == (1, 1, VOCAB), f"row {row} の出力形"
                assert tuple(hidden.shape) == (1, 1, HIDDEN), f"row {row} の hidden 形"
                assert int(logits[0, 0].argmax()) == int(expected[0, 0, 0]), f"row {row} の token"

    def test_multiple_rows_are_the_per_row_single_row_results_stacked(self, tiny_model_trio):
        """R > 1 は「R 回の 1 行 run」を積んだものと**厳密に**一致する（verify の前提）。

        行ごとに独立でなければ、投機 verify が採点する行と実際に採用する行が別物になる。
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
            # hidden は同じ本体 forward の gather なのでビット同一。logits は lm_head の GEMM の
            # M が 3 と 1 で変わり、縮約順が動きうるので近傍で見る（1 位は厳密に一致する）。
            assert torch.equal(hidden[:, index : index + 1], one_hidden), f"行 {index} の hidden"
            assert torch.allclose(logits[:, index : index + 1], one_logits, atol=1e-5, rtol=1e-5), (
                f"行 {index} の logits"
            )
            assert int(logits[0, index].argmax()) == int(one_logits[0, 0].argmax()), (
                f"行 {index} の 1 位"
            )

    def test_the_second_output_is_the_hidden_the_lm_head_consumed(self, tiny_model_trio):
        """MUST: 出力 1 は lm_head の**入力そのもの**（別の中間値でも [1,R,H] は合う）。"""
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
        """ホスト供給の PLE が 35 表経路と同じ値なら、logits も token-only 形と一致する。

        MUST: 恒真でない — `per_layer_inputs` を別の並びで渡せばここが割れる（下の対で確認）。
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
        assert not torch.equal(correct, swapped), "層の並びを崩しても同じ logits が出ている"


# ---- tiny な実モデルでの一周 -----------------------------------------------


class TestRowSymbolBinding:
    """例示 2 行で焼いた製品グラフが **R = 1 と R = kmax の両方で走る**こと（transformers 要）。

    段 1 の合格線そのもの: 通常の prefill / decode は R = 1 で走るので、torch.export が
    `Dim(min=1)` を 1 へ特殊化していれば既存経路が動かない。IR の `symbols` は名前の列だけで
    上限を持たない（`docs/ir-v2.md`）ので、この性質を見られるのは焼いた ExportedProgram を
    実際に別の行数で回す形だけ。
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
            assert tuple(logits.shape) == (1, count, VOCAB), f"R = {count} の logits 形"
            assert tuple(hidden.shape) == (1, count, HIDDEN), f"R = {count} の hidden 形"

    def test_the_slack_is_what_bounds_the_row_symbol(self):
        """R の上限は sliding ring の余裕そのもの（別々に動かせる 2 つの数にしない）。"""
        # verify は draft k 行 + bonus 1 行で、k の上限が ring の余裕（棄却されうる行数）。
        assert product.ROW_SYM_MAX == decode.SLIDING_SLACK_ROWS + 1
        assert 1 < product.EXAMPLE_ROWS <= product.ROW_SYM_MAX


class TestExportedProductForm:
    """tiny な実モデルを export → 手術 → 混成量子化つき書き出しまで通す（transformers が要る）。"""

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
        # PLE はホストが供給する入力になったので、export の前に席ごと落とす（台本と同じ順序）。
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
        """MUST: R が記号のまま残ること（例示 2 行で焼いても行数が定数化しない）。

        R が焼かれると通常の decode（R = 1）が走れない。束縛点は `last_row[R]` 1 本だけで、
        IR の `symbols` は名前の列しか持たない（上限は記録されない）ので、ランタイムは 1 でも
        8 でも束縛できる。
        """
        verified, _, _ = tiny_container

        last_row = next(spec for spec in verified.inputs if spec.name == decode.TOKEN_ONLY_LAST_ROW)
        assert list(last_row.shape) == [product.ROW_SYMBOL]
        assert sorted(verified.symbols) == sorted(
            {decode.SEQ_SYMBOL, decode.CAPACITY_SYMBOL, product.ROW_SYMBOL}
        )

    def test_the_sliding_slots_carry_the_speculative_slack(self, tiny_container):
        """sliding スロットの物理行数が `window + 余裕`（棄却行が live 窓を潰さない条件）。"""
        verified, config, _ = tiny_container

        capacities = [list(slot.shape)[2] for slot in verified.states.values()]
        baked = [value for value in capacities if not isinstance(value, str)]

        assert baked, "sliding スロット（容量が実数の側）が見つからない"
        assert set(baked) == {WINDOW + decode.SLIDING_SLACK_ROWS}
        assert decode.sliding_capacity(config) == WINDOW + decode.SLIDING_SLACK_ROWS

    def test_swapped_outputs_are_detected(self, tiny_container):
        """出力の入れ替えを落とす。

        tiny な被験体は `VOCAB == HIDDEN` なので**幅では見分けがつかない**（実物は 262,144 と
        2,048 で幅の門が先に落ちる）。それでも落ちるのは、出力 0 の祖先を遡って lm_head に
        当たることを見る構造検査があるから — hidden 側は行選択の `embedding` で行き止まる。
        """
        verified, config, storage = tiny_container
        swapped = replace(verified, outputs=list(reversed(verified.outputs)))

        with pytest.raises(AssertionError, match="lm_head（linear）が無い"):
            product.assert_ir_form_product(swapped, config, storage, VOCAB)

    def test_the_ple_tables_are_gone_from_the_container(self, tiny_container):
        """常駐削減そのもの（ADR 0085）— PLE 表を引く embedding が 1 本も残らない。"""
        verified, _, _ = tiny_container

        weights = [
            list(verified.values[node.ins[0]].shape)
            for node in verified.nodes
            if node.op == "embedding"
        ]

        assert [VOCAB, PLE_DIM] not in weights
        # embedding は 主 embedding 1 + 行選択 1（RoPE の表引きはもう居ない）。
        assert len(weights) == 2

    def test_the_decode_form_check_rejects_the_product_graph(self, tiny_container):
        """既存 2 系列の形検査は製品形を通さない（3 つの形が混ざらないことの固定）。"""
        verified, config, storage = tiny_container

        with pytest.raises(AssertionError, match="グラフ入力"):
            decode.assert_ir_form_decode(verified, config, storage, token_only=True)

    def test_a_residual_ple_table_is_detected(self, tiny_container):
        """PLE 表を引く embedding が残った形（入力も表も両方通る配線）を落とす。"""
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

        with pytest.raises(AssertionError, match="PLE 表の形"):
            product.assert_ir_form_product(polluted, config, storage, VOCAB)

    def test_an_argmax_exit_is_detected(self, tiny_container):
        """出口が argmax へ退行した形（sampling の余地が消える — ADR 0083 決定 6）。"""
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
        """行選択が lm_head の**後ろ**へ回った形（値は一致するので形と構造でしか見えない）。

        lm_head の入力は hidden 出口そのものなので、行軸を chunk 行 `M` へ広げると出力 1 の
        宣言が先に食い違う（`[1, R, H]` でなくなる）。行選択が後ろへ回れば必ずここを通るので、
        退行はこの 1 本目の門で止まる — 宣言だけ正しく配線が壊れた形は
        {@link TestExportedProductForm.test_swapped_outputs_are_detected} の構造検査が受ける。
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

        with pytest.raises(AssertionError, match=r"出力 1 の宣言 shape"):
            product.assert_ir_form_product(widened, config, storage, VOCAB)


# ---- 系列 driver の一周 ----------------------------------------------------


@pytest.fixture
def tiny_product_series(monkeypatch, tmp_path_factory):
    """`export_product.export_series` を実資産なしで 1 周させるための差し替え一式。

    差し替えるのは**素材の出どころ**（模型・ケース・トークナイザ）と、tiny な乱数重みでは
    立てられない期待（最終位置の 1 位）と、tiny な vocab では 1 本にしかならない sidecar の
    上限だけ。量子化 → sidecar → 検査 → export → 手術 → 書き出し → 形検査 → 公開の経路は
    本物を通す（差し替えの方針は `test_export_decode.tiny_series` と同じ）。
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
        # MUST: 本物と同じく `vocab_size_per_layer_input` を**検査席の行数**へ差し替えた状態で
        # 返す（`gx.load_model_and_tables` の実装 — 分割 35 本を PLE の唯一の正本にする形）。
        # 素の VOCAB のまま返すと、台本が config からこの欄を読む退行が driver に映らない。
        model.config.vocab_size_per_layer_input = ple.PLE_PROBE_ROWS
        return model, tables

    monkeypatch.setattr(gx, "load_model_and_tables", materials)

    from _shared import decode_series as shared
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
    reference = tmp_path_factory.mktemp("reference-series")
    from safetensors.torch import save_file

    for name, ids in cases:
        save_file(
            {shared.PROMPT_KEY: ids[0].to(torch.int32).contiguous()},
            str(reference / f"{shared.GREEDY_PREFIX}{name}{shared.GREEDY_SUFFIX}"),
        )
    monkeypatch.setattr(gx, "build_cases", lambda model_dir, sym_max, window: cases)
    monkeypatch.setattr(gx, "load_tokenizer", lambda model_dir: _StubSeriesTokenizer())
    # tiny 系列でも PLE を 2 block 以上へ割らせる（block 跨ぎを 1 度は踏む）。
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
    return cases, checkpoint, reference, seen


class TestExportSeries:
    def test_it_publishes_the_container_the_probe_and_the_provenance(
        self, tiny_product_series, tmp_path
    ):
        cases, checkpoint, reference, seen = tiny_product_series
        out_dir = tmp_path / "series"

        summary = product.export_series(
            checkpoint,
            out_dir,
            sym_max=TINY_SYM_MAX,
            reference=reference,
        )

        assert summary["ple_blocks"][product.PLE_VALUES_KEY] > 1, "block 跨ぎを踏ませる"
        # PLE は容器の資産なので、系列には独立したファイルとして現れない。
        assert sorted(path.name for path in out_dir.iterdir()) == sorted(
            [
                product.PLE_PROBE_FILE,
                provenance_file(),
                *_container_files(out_dir),
            ]
        )
        assert list(summary) == PRODUCT_SUMMARY_KEYS
        assert summary["outputs"] == 2
        assert set(seen["greedy"]) == {name for name, _ in cases}
        # 作業席も退避席も残らない（据え替えの後片付けは core の原語の担当）。
        assert list(tmp_path.iterdir()) == [out_dir]

    def test_the_index_describes_every_block_and_the_dequantization(
        self, tiny_product_series, tmp_path
    ):
        _, checkpoint, reference, _ = tiny_product_series
        out_dir = tmp_path / "series"

        product.export_series(
            checkpoint,
            out_dir,
            sym_max=TINY_SYM_MAX,
            reference=reference,
        )

        index = _published_index(out_dir)
        assert index["storage"] == product.PLE_STORAGE
        assert index["tokens"] == VOCAB
        assert index["layers"] == len(DECODE_LAYER_TYPES)
        assert index["dim"] == PLE_DIM
        assert index["embedScale"] == pytest.approx(float(PLE_DIM) ** 0.5)
        blocks = index[product.PLE_VALUES_KEY]["blocks"]
        assert blocks[0]["start"] == 0
        assert blocks[-1]["stop"] == VOCAB

    def test_the_probe_reference_matches_the_published_assets(self, tiny_product_series, tmp_path):
        """据えた資産だけで逆量子化ビット一致が言えること（TS 側の門が読む 2 本の対）。"""
        _, checkpoint, reference, _ = tiny_product_series
        out_dir = tmp_path / "series"

        product.export_series(
            checkpoint,
            out_dir,
            sym_max=TINY_SYM_MAX,
            reference=reference,
        )

        index = _published_index(out_dir)
        with safe_open(str(out_dir / product.PLE_PROBE_FILE), framework="pt") as handle:
            probe = handle.get_tensor(product.PROBE_TOKENS_KEY)
            expected = handle.get_tensor(product.PROBE_INPUTS_KEY)

        product.assert_ple_assets(
            out_dir / gx.MODEL_FILE, index, [int(token) for token in probe], expected
        )

    def test_a_stale_reference_series_is_rejected_before_the_export(
        self, tiny_product_series, tmp_path
    ):
        """流用する golden の検めは席へ入る前（数十分の export を始める前に落とす）。"""
        _, checkpoint, _unused, _ = tiny_product_series
        out_dir = tmp_path / "series"

        with pytest.raises(AssertionError, match="参照 golden 系列"):
            product.export_series(
                checkpoint,
                out_dir,
                sym_max=TINY_SYM_MAX,
                reference=Path(tmp_path / "missing"),
            )
        assert not out_dir.exists()


def provenance_file() -> str:
    """出所記録のファイル名（綴りの正本は {@link gemma4.provenance}）。"""
    from gemma4 import provenance

    return provenance.REFERENCE_FILE


def _container_files(out_dir: Path) -> list[str]:
    """据わったコンテナの part ファイル名（本数は書いたバイト数が決める）。"""
    return [path.name for path in container_parts(out_dir / gx.MODEL_FILE)]


def _published_index(out_dir: Path) -> dict:
    """据えた容器の資産 `ple_index` を読む（配布形そのものから引く）。"""
    return json.loads(read_asset(out_dir / gx.MODEL_FILE, PLE_INDEX_ASSET))


def test_the_quantized_values_survive_a_round_trip_through_the_blocks(tiny_tables, tmp_path):
    """全 token・全層の再配置が i8 のバイト列として保たれること（probe の外側も見る）。"""
    scales = _quantize_tables(tiny_tables)
    layers = len(DECODE_LAYER_TYPES)
    expected = [
        quantize_to_int8(table.weight.data, scales[f"{gx.PER_LAYER_PREFIX}{index}.weight"])
        for index, table in enumerate(tiny_tables)
    ]
    values, row_scales = product.quantized_ple_tables(tiny_tables, scales)
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
                f"block {block['asset']} の層 {layer}"
            )
