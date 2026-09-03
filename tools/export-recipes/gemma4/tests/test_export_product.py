"""製品グラフ台本（`gemma4/export_product.py`）の台本レベルの約束事（実重み不要分）。

実重みの emit は手動（chunk 系列 2 本のテストと同じ規律）。ここで固定するのは、壊れると
**偽 PASS** になる側の規律だけ:

- PLE sidecar の分割規則（上限内の最小本数 → 行数の均し）と、1 token が上限を超える形の拒否
- 再配置（table-major → token-major）が 35 表経路と**ビット一致**すること、および
  {@link product.assert_ple_sidecar} が scale の層ずれ・範囲の off-by-one を**実際に検出**すること
- 逆量子化の順序（`f32(i8) * per-row scale` → `* embed_scale`）が fake-quant 済みの表と一致すること
- 製品ラッパの eager 同値（`argmax(logits)` が token-only 形の token と全行で一致）
- {@link product.assert_ir_form_product} が製品形の 3 本（PLE 外出し・argmax 不在・最終行
  logits）と、既存 2 系列と共有の規律（states / 層種別 / 残骸 / 格納）を実際に見ること
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
from safetensors import safe_open
from torch import nn
from torch.export import Dim

from gemma4 import export as gx
from gemma4 import export_decode as decode
from gemma4 import export_product as product
from gemma4 import export_token as token_only
from gemma4 import ple
from gemma4.tests.test_export import PLE_DIM, TINY_SYM_MAX, VOCAB, WINDOW
from gemma4.tests.test_export_decode import (
    DECODE_LAYER_TYPES,
    OWNER_LAYERS,
    TINY_CONTAINER_FILES,
    _tiny_decode_config,
)
from karume.convert import PRESERVED_OP_PREFIXES_WITH_ATTENTION
from karume.pipeline import export_module
from karume.quantize import quantize_to_int8
from karume.shards import SHARD_DATA_CAPACITY
from karume.states import to_states_form

#: tiny 系列で **2 本以上**の sidecar を作らせる容量（実物は既定の容量で 9 本 — 1 本しか
#: 出ない容量で driver を回すと、連番・索引・probe の shard 跨ぎが 1 度も踏まれない）。
TINY_PLE_LIMIT = product.ple_token_bytes(len(DECODE_LAYER_TYPES), PLE_DIM) * (VOCAB // 2)

#: 系列の要約の欄（順序込み）。ここが変わると実走の記録の形が変わる。
PRODUCT_SUMMARY_KEYS = [
    "dir",
    "nodes",
    "outputs",
    "initializers",
    "model_bytes",
    "ple_bytes",
    "ple_shards",
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


class TestPlanPleShards:
    def test_it_covers_the_vocabulary_without_gaps_or_overlaps(self):
        ranges = product.plan_ple_shards(1000, 10, limit=4000)

        assert ranges[0][0] == 0
        assert ranges[-1][1] == 1000
        for previous, following in pairwise(ranges):
            assert previous[1] == following[0]

    def test_it_takes_the_minimum_count_under_the_limit(self):
        """上限 4000 バイト = 400 token/本 なので 1000 token は 3 本ちょうど。"""
        ranges = product.plan_ple_shards(1000, 10, limit=4000)

        assert len(ranges) == 3

    def test_it_evens_the_rows_instead_of_leaving_a_remainder_shard(self):
        """k を先に決めてから均す（ADR 0081 の書き手ポリシーと同型 — 端数 shard を作らない）。"""
        rows = [stop - start for start, stop in product.plan_ple_shards(1000, 10, limit=4000)]

        assert rows == [334, 333, 333]

    def test_every_shard_stays_under_the_limit(self):
        for start, stop in product.plan_ple_shards(262144, 9100):
            assert (stop - start) * 9100 <= SHARD_DATA_CAPACITY

    def test_the_real_model_lands_on_nine_shards(self):
        """実物の見込み（262,144 token × 9,100B = 2,275MiB → 既定の容量で 9 本）。

        既定は書き手の容量（{@link karume.shards.SHARD_DATA_CAPACITY}）— 受理上限が
        ファイル長で測る値になったので、sidecar もヘッダぶんを空けた容量で切る。
        """
        ranges = product.plan_ple_shards(262144, 9100)

        assert len(ranges) == 9
        assert ranges[-1][1] == 262144

    def test_a_single_token_over_the_limit_fails_loudly(self):
        """分割の粒度がこれ以上細かくできない形（黙って上限を破らない）。"""
        with pytest.raises(ValueError, match="shard 上限"):
            product.plan_ple_shards(4, 4096, limit=1024)


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
    def test_it_touches_both_sides_of_every_shard_boundary(self):
        ranges = product.plan_ple_shards(1000, 10, limit=4000)

        probe = product.ple_probe_tokens(1000, ranges)

        for start, stop in ranges:
            assert start in probe
            assert stop - 1 in probe

    def test_it_is_sorted_unique_and_inside_the_vocabulary(self):
        ranges = product.plan_ple_shards(1000, 10, limit=4000)

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
def tiny_sidecar(tiny_tables, tmp_path):
    """tiny な sidecar を 2 本以上へ割って書き、`(索引, probe, 参照)` を返す。"""
    scales = _quantize_tables(tiny_tables)
    layers = len(DECODE_LAYER_TYPES)
    embed_scale = float(PLE_DIM) ** 0.5
    ranges = product.plan_ple_shards(
        VOCAB, product.ple_token_bytes(layers, PLE_DIM), limit=TINY_PLE_LIMIT
    )
    assert len(ranges) > 1, "tiny 系列でも shard 跨ぎを踏ませる"
    probe = product.ple_probe_tokens(VOCAB, ranges)
    with torch.no_grad():
        reference = ple.per_layer_inputs(
            tiny_tables, torch.tensor([list(probe)], dtype=torch.int64), embed_scale
        )
    values, row_scales = product.quantized_ple_tables(tiny_tables, scales)
    index = product.ple_index(VOCAB, layers, PLE_DIM, embed_scale)
    index["shards"] = product.write_ple_shards(values, row_scales, ranges, tmp_path, index)
    return index, probe, reference


class TestAssertPleSidecar:
    def test_the_written_bytes_rebuild_the_table_major_path_bit_for_bit(
        self, tiny_sidecar, tmp_path
    ):
        index, probe, reference = tiny_sidecar

        product.assert_ple_sidecar(tmp_path, index, probe, reference)

    def test_the_shards_are_token_major_with_per_row_scales(self, tiny_sidecar, tmp_path):
        """配布形そのもの（ADR 0085 決定 1）— `[token][layer][dim]` i8 + `[token][layer]` f32。"""
        index, _, _ = tiny_sidecar

        shard = index["shards"][0]
        with safe_open(str(tmp_path / shard["file"]), framework="pt") as handle:
            rows = shard["stop"] - shard["start"]
            assert handle.get_slice(product.PLE_VALUES_KEY).get_shape() == [
                rows,
                len(DECODE_LAYER_TYPES),
                PLE_DIM,
            ]
            assert handle.get_slice(product.PLE_SCALES_KEY).get_shape() == [
                rows,
                len(DECODE_LAYER_TYPES),
            ]
            assert handle.get_slice(product.PLE_VALUES_KEY).get_dtype() == "I8"

    def test_each_shard_declares_its_own_token_range(self, tiny_sidecar, tmp_path):
        """索引と shard の自己申告が食い違う組を読み手が落とせる形（沈黙誤値の防波堤）。"""
        index, _, _ = tiny_sidecar

        for shard in index["shards"]:
            with safe_open(str(tmp_path / shard["file"]), framework="pt") as handle:
                declared = json.loads(handle.metadata()[product.PLE_METADATA_KEY])
            assert declared["start"] == shard["start"]
            assert declared["stop"] == shard["stop"]
            assert declared["schema"] == product.PLE_SCHEMA
            assert declared["tokens"] == VOCAB

    def test_a_layer_shifted_scale_is_detected(self, tiny_sidecar, tmp_path):
        """MUST: scale の層ずれは形も型も dtype も合う（`torch.equal` でしか捕まらない）。"""
        index, probe, reference = tiny_sidecar
        shifted = reference.clone()
        shifted[0, :, 0] = reference[0, :, 1]

        with pytest.raises(AssertionError, match="ビット一致しない"):
            product.assert_ple_sidecar(tmp_path, index, probe, shifted)

    def test_an_off_by_one_token_range_is_detected(self, tiny_sidecar, tmp_path):
        """範囲を 1 行ずらすと**別 token の有効な行**が出る（ADR 0085 決定 5 の沈黙誤値）。"""
        index, probe, reference = tiny_sidecar
        moved = dict(index)
        moved["shards"] = [{**shard, "start": shard["start"] + 1} for shard in index["shards"]]

        with pytest.raises(AssertionError):
            product.assert_ple_sidecar(tmp_path, moved, probe, reference)

    def test_a_probe_outside_every_range_is_detected(self, tiny_sidecar, tmp_path):
        """索引が vocab を覆っていない形（probe が無検査で素通りしない）。"""
        index, probe, reference = tiny_sidecar
        truncated = dict(index)
        truncated["shards"] = index["shards"][:-1]

        with pytest.raises(AssertionError, match="覆っていない"):
            product.assert_ple_sidecar(tmp_path, truncated, probe, reference)


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
        """出口の差は argmax の有無だけ（ADR 0083 決定 6）— 全行を踏む。"""
        token_form, product_form, tables = tiny_model_trio
        torch.manual_seed(1)
        ids = torch.randint(0, VOCAB, (1, 7), dtype=torch.int64)
        specs = decode.rope_specs(product_form.model.config)
        rope = decode.rope_args(specs, decode.positions_for(ids))
        stacked = ple.per_layer_inputs(tables, ids, product_form.per_layer_scale)

        with torch.no_grad():
            for row in range(int(ids.shape[1])):
                last_row = torch.tensor([row], dtype=torch.int64)
                logits = product_form(ids, *rope, stacked, last_row)
                expected = token_form(ids, *rope, last_row)
                assert tuple(logits.shape) == (1, 1, VOCAB), f"row {row} の出力形"
                assert int(logits[0, 0].argmax()) == int(expected[0, 0, 0]), f"row {row} の token"

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
            correct = product_form(ids, *rope, stacked, last_row)
            swapped = product_form(ids, *rope, stacked.flip(2), last_row)
            expected = token_form(ids, *rope, last_row)

        assert int(correct[0, 0].argmax()) == int(expected[0, 0, 0])
        assert not torch.equal(correct, swapped), "層の並びを崩しても同じ logits が出ている"


# ---- tiny な実モデルでの一周 -----------------------------------------------


class TestExportedProductForm:
    """tiny な実モデルを export → 手術 → 混成量子化つき書き出しまで通す（transformers が要る）。"""

    @pytest.fixture
    def tiny_container(self, tiny_model_trio, tmp_path):
        _, wrapper, tables = tiny_model_trio
        int8, int4, scales = gx.quantize_wrapper(wrapper)
        torch.manual_seed(4)
        ids = torch.randint(0, VOCAB, (1, WINDOW + 3), dtype=torch.int64)
        stacked = ple.per_layer_inputs(tables, ids, wrapper.per_layer_scale)
        last_row = decode.last_row_for(ids)
        specs = decode.rope_specs(wrapper.model.config)
        rope = decode.rope_args(specs, decode.positions_for(ids))
        # PLE はホストが供給する入力になったので、export の前に席ごと落とす（台本と同じ順序）。
        del wrapper.per_layer
        seq = Dim(decode.SEQ_SYMBOL, min=2, max=TINY_SYM_MAX)
        graph, tensors = export_module(
            wrapper,
            (ids, *rope, stacked, last_row),
            dynamic_shapes=(*({1: seq} for _ in range(2 + len(rope))), None),
            symbol_names=(decode.SEQ_SYMBOL,),
            preserved=PRESERVED_OP_PREFIXES_WITH_ATTENTION,
        )
        config = wrapper.model.config
        surgical = to_states_form(graph, decode.states_plan(graph, config))
        verified = decode._write_container(
            surgical,
            tensors,
            tmp_path / gx.MODEL_FILE,
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
        assert len(verified.outputs) == 1
        assert "argmax" not in verified.required_ops
        assert form["attention_nodes"] == len(DECODE_LAYER_TYPES)
        assert form["state_append_nodes"] == 2 * OWNER_LAYERS
        assert form["logits"] == [1, 1, VOCAB]

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
        """行選択が lm_head の**後ろ**へ回った形（値は一致するので構造でしか見えない）。"""
        verified, config, storage = tiny_container
        producer = {out: node for node in verified.nodes for out in node.outs}
        linear = next(
            node
            for node in verified.nodes
            if node.op == "linear" and list(verified.values[node.outs[0]].shape)[:2] == [1, 1]
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

        with pytest.raises(AssertionError, match="選択済みの 1 行"):
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
    monkeypatch.setattr(
        product,
        "plan_ple_shards",
        functools.partial(product.plan_ple_shards, limit=TINY_PLE_LIMIT),
    )
    seen: dict[str, dict[str, int]] = {}

    def record(greedy, expected, labels):
        seen["greedy"] = dict(greedy)
        return {"stub": "ok"}

    monkeypatch.setattr(gx, "_sanity", record)
    return cases, checkpoint, reference, seen


class TestExportSeries:
    def test_it_publishes_the_container_the_sidecar_and_the_provenance(
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

        shards = [str(shard["file"]) for shard in summary["ple_shards"]]
        assert len(shards) > 1, "tiny 系列でも sidecar の連番を踏ませる"
        assert sorted(path.name for path in out_dir.iterdir()) == sorted(
            [
                product.PLE_INDEX_FILE,
                product.PLE_PROBE_FILE,
                provenance_file(),
                *shards,
                *TINY_CONTAINER_FILES,
            ]
        )
        assert list(summary) == PRODUCT_SUMMARY_KEYS
        assert summary["outputs"] == 1
        assert set(seen["greedy"]) == {name for name, _ in cases}
        # 作業席も退避席も残らない（据え替えの後片付けは core の原語の担当）。
        assert list(tmp_path.iterdir()) == [out_dir]

    def test_the_index_describes_every_shard_and_the_dequantization(
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

        index = json.loads((out_dir / product.PLE_INDEX_FILE).read_text(encoding="utf-8"))
        assert index["schema"] == product.PLE_SCHEMA
        assert index["tokens"] == VOCAB
        assert index["layers"] == len(DECODE_LAYER_TYPES)
        assert index["dim"] == PLE_DIM
        assert index["embedScale"] == pytest.approx(float(PLE_DIM) ** 0.5)
        assert index["shards"][0]["start"] == 0
        assert index["shards"][-1]["stop"] == VOCAB

    def test_the_probe_reference_matches_the_written_sidecar(self, tiny_product_series, tmp_path):
        """据えた資産だけで逆量子化ビット一致が言えること（TS 側の門が読む 2 本の対）。"""
        _, checkpoint, reference, _ = tiny_product_series
        out_dir = tmp_path / "series"

        product.export_series(
            checkpoint,
            out_dir,
            sym_max=TINY_SYM_MAX,
            reference=reference,
        )

        index = json.loads((out_dir / product.PLE_INDEX_FILE).read_text(encoding="utf-8"))
        with safe_open(str(out_dir / product.PLE_PROBE_FILE), framework="pt") as handle:
            probe = handle.get_tensor(product.PROBE_TOKENS_KEY)
            expected = handle.get_tensor(product.PROBE_INPUTS_KEY)

        product.assert_ple_sidecar(out_dir, index, [int(token) for token in probe], expected)

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


def test_the_quantized_values_survive_a_round_trip_through_the_shards(tiny_tables, tmp_path):
    """全 token・全層の再配置が i8 のバイト列として保たれること（probe の外側も見る）。"""
    scales = _quantize_tables(tiny_tables)
    layers = len(DECODE_LAYER_TYPES)
    expected = [
        quantize_to_int8(table.weight.data, scales[f"{gx.PER_LAYER_PREFIX}{index}.weight"])
        for index, table in enumerate(tiny_tables)
    ]
    values, row_scales = product.quantized_ple_tables(tiny_tables, scales)
    ranges = product.plan_ple_shards(
        VOCAB, product.ple_token_bytes(layers, PLE_DIM), limit=TINY_PLE_LIMIT
    )
    index = product.ple_index(VOCAB, layers, PLE_DIM, float(PLE_DIM) ** 0.5)
    index["shards"] = product.write_ple_shards(values, row_scales, ranges, tmp_path, index)

    for shard in index["shards"]:
        with safe_open(str(tmp_path / shard["file"]), framework="pt") as handle:
            block = handle.get_tensor(product.PLE_VALUES_KEY)
        start, stop = shard["start"], shard["stop"]
        for layer in range(layers):
            assert torch.equal(block[:, layer], expected[layer][start:stop]), (
                f"shard {shard['file']} の層 {layer}"
            )
