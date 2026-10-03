"""`wan/umt5_patch.py`（umT5 の有効長の S 形ラッパ・バケット表の Python 側の正本）と
`wan/umt5_probe.py` の約束事（ADR 0119 決定 3〜5・段 10c の下見）。

固定するのは、壊れても例外が出ず**数だけが静かにずれる**側:

- 入力で受けたバケット表を層の表で引いたバイアスが、上流の `compute_bias` と**ビット一致**する
  （表の向き `table[i][j] = bucket(j − i)` と式）
- 活性を上流のままにしたラッパが上流の forward と f32 で**ビット一致**する（書き下しの同値）。
  `gelu_new` → `nn.GELU(approximate="tanh")` の差し替えは丸めの差だけ
- ラッパは入力の表を実際に使う（表を転置・1 ずらすと出力が大きく動く — 上の一致が恒真でない対）
- exporter の i8 の既定の対象は相対位置の表を含む（調査 §1.2 の推測の確認）。ラッパの選択はそれを
  外し、emit は F32 の明示が無いと落ちる（決定 5 の「F32 のままにする指定の形」）
- S 形の export: 入力 2 本（i32）・記号 `L` 1 つ・RMSNorm が `rms_norm` に畳まれ・活性が
  `gelu_tanh`・`pow` が残らない・表の格納は F32・golden が CPU で採れる
- 「有効長だけ」と「512 + マスク」の一致（調査 §2.3 の再現 — 実モデルでの一致は段 10c の検収）

乱数初期化の小さな umT5（`wan.umt5_probe.TINY_CONFIG`）だけで回す — 実重みは読まない。
transformers は `wan` グループにしか無いので、モデルを作る fixture の中で `importorskip` する。
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest
import torch
from safetensors.torch import load_file, save_file

from karume.pipeline import publish_model
from wan import umt5_patch as up
from wan import umt5_probe as probe

#: 層数（`TINY_CONFIG` — op の本数の期待はこれから導く）。
LAYERS = int(probe.TINY_CONFIG["num_layers"])
D_MODEL = int(probe.TINY_CONFIG["d_model"])

#: `gelu_new` を tanh 近似の `aten.gelu` へ差し替えた出力と上流の差の上限（比 = 最大絶対差 ÷
#: 参照の最大絶対値）。実測（2026-10-03・`LENGTHS` の 8 本）の最悪は 3.93e-7（L = 511）で、その
#: 約 5 倍。表の取り違え（下の故障注入）は 2.6e-2 以上で、5 桁離れている。
TANH_RATIO_BOUND = 2e-6

#: 「有効長だけ」と「512 + マスク」の f32 の差の上限（同じ比）。実測の最悪は 7.02e-7（L = 2 — 8 本中
#: 6 本はビット一致・差は GEMM の縮約順）で、その約 5 倍。マスクを落とすと 1.1 以上。
MASK_F32_RATIO_BOUND = 3.5e-6

#: 表の故障注入が少なくとも動かす比（実測の最小は転置の 2.6e-2〈L = 2〉）。
FAULT_RATIO_FLOOR = 1e-2

#: 計測の対象の長さ（下限・固定プロンプト・max_distance 越え・上限）。
LENGTHS = (2, 28, 126, 300, 512)


@pytest.fixture(scope="module")
def model() -> Any:
    pytest.importorskip("transformers")
    return probe.tiny_model()


def attentions(model: Any) -> list[Any]:
    return [block.layer[0].SelfAttention for block in model.encoder.block]


class TestRelativePositionBuckets:
    @pytest.mark.parametrize("length", [2, 9, 28, 130, 512])
    def test_gathered_table_is_upstream_compute_bias(self, model, length):
        """全層で: 表を層の表で引いて `[1, heads, L, L]` へ回すと上流の `compute_bias` そのもの。"""
        for attention in attentions(model):
            table = up.relative_position_buckets(length, attention)
            with torch.no_grad():
                gathered = attention.relative_attention_bias(table).permute([2, 0, 1]).unsqueeze(0)
                expected = attention.compute_bias(length, length)

            assert table.shape == (length, length)
            assert torch.equal(gathered, expected)

    def test_meta_attention_builds_the_same_table_as_the_model(self, model):
        """重みを持たない上流のモジュール（fixture の生成に使う口）でも同じ表。"""
        meta = up.bucket_attention(model.config)

        assert torch.equal(
            up.relative_position_buckets(300, meta),
            up.relative_position_buckets(300, attentions(model)[0]),
        )

    @pytest.mark.parametrize("length", [0, 1, up.MAX_LENGTH + 1])
    def test_lengths_outside_the_graph_domain_are_rejected(self, model, length):
        with pytest.raises(ValueError, match="の外"):
            up.relative_position_buckets(length, attentions(model)[0])

    @pytest.mark.parametrize("length", [True, 28.0])
    def test_non_integer_lengths_are_rejected(self, model, length):
        with pytest.raises(TypeError):
            up.relative_position_buckets(length, attentions(model)[0])

    def test_decoder_config_is_rejected(self, model):
        """片方向（デコーダ）のバケットは別の式 — 双方向だけを扱う。"""
        decoder = copy.deepcopy(model.config)
        decoder.is_decoder = True

        with pytest.raises(up.Umt5ConfigError, match="is_decoder"):
            up.bucket_attention(decoder)


class TestWrapper:
    @pytest.mark.parametrize("length", LENGTHS)
    def test_upstream_activation_is_bit_exact_with_the_upstream_forward(self, model, length):
        wrapper = up.Umt5EncoderTokens(model, activation="upstream")
        inputs = probe.case_inputs(wrapper, length)

        with torch.no_grad():
            actual = wrapper(*inputs)

        assert torch.equal(actual, up.valid_output(model, inputs[0]))

    @pytest.mark.parametrize("length", LENGTHS)
    def test_tanh_activation_moves_only_the_rounding(self, model, length):
        wrapper = up.Umt5EncoderTokens(model, activation="tanh")
        inputs = probe.case_inputs(wrapper, length)

        with torch.no_grad():
            report = up.compare(wrapper(*inputs), up.valid_output(model, inputs[0]))

        assert report["ratio"] <= TANH_RATIO_BOUND
        # 差し替えが実際に効いている（上流の活性のままなら 0 — 上の上限が恒真でないことの対）。
        assert report["differing"] > 0

    @pytest.mark.parametrize(
        "fault",
        [
            pytest.param(lambda table: table.T.contiguous(), id="transposed"),
            pytest.param(lambda table: (table + 1).clamp(max=31), id="shifted-by-one"),
        ],
    )
    @pytest.mark.parametrize("length", [2, 28, 512])
    def test_a_wrong_table_moves_the_output(self, model, length, fault):
        """表は入力から実際に読まれる — 取り違えた表は丸めの桁を大きく越えて出力を動かす。"""
        wrapper = up.Umt5EncoderTokens(model, activation="upstream")
        ids, table = probe.case_inputs(wrapper, length)

        with torch.no_grad():
            ratio = up.max_ratio(wrapper(ids, fault(table)), up.valid_output(model, ids))

        assert ratio >= FAULT_RATIO_FLOOR

    def test_training_mode_is_rejected(self, model):
        training = copy.deepcopy(model).train()

        with pytest.raises(up.Umt5ConfigError, match="学習モード"):
            up.Umt5EncoderTokens(training)

    def test_non_f32_weights_are_rejected(self, model):
        with pytest.raises(up.Umt5ConfigError, match="f32"):
            up.Umt5EncoderTokens(copy.deepcopy(model).to(torch.bfloat16))

    def test_another_activation_is_rejected(self, model):
        relu = copy.deepcopy(model.config)
        relu.dense_act_fn = "relu"

        with pytest.raises(up.Umt5ConfigError, match="relu"):
            up.check_supported(relu)


class TestQuantSelection:
    def test_default_targets_include_the_relative_bias_tables(self, model):
        """exporter の i8 の既定（`QUANT_CHANNEL_AXES` の `nn.Embedding`）は表まで丸める。"""
        report = probe.quant_targets(model)

        assert report["relative_bias_tables"] == LAYERS
        assert report["default_includes_relative_bias"]
        assert report["default"][up.RELATIVE_BIAS_ATTRIBUTE] == LAYERS

    def test_the_wrapper_selection_rounds_everything_but_the_tables(self, model):
        report = probe.quant_targets(model)

        assert not report["selected_includes_relative_bias"]
        assert report["selected"] == {
            "embed_tokens": 1,
            **{name: LAYERS for name in ("k", "o", "q", "v", "wi_0", "wi_1", "wo")},
        }

    def test_fake_quant_leaves_the_tables_and_rounds_the_linears(self, model):
        wrapper = up.Umt5EncoderTokens(copy.deepcopy(model))
        original = dict(model.encoder.named_parameters())

        up.fake_quant_i8(wrapper)

        for name, parameter in wrapper.encoder.named_parameters():
            before = original[name]
            if up.RELATIVE_BIAS_ATTRIBUTE in name or "layer_norm" in name:
                assert torch.equal(parameter, before), name
            else:
                assert not torch.equal(parameter, before), name

    def test_overrides_name_every_table_weight(self, model):
        wrapper = up.Umt5EncoderTokens(model)

        assert up.storage_overrides(wrapper) == {
            f"encoder.block.{index}.layer.0.SelfAttention.relative_attention_bias.weight": "f32"
            for index in range(LAYERS)
        }


@pytest.fixture(scope="module")
def exported(model, tmp_path_factory) -> tuple[dict[str, Any], Path]:
    """下見の export を容器と golden のファイルまで書く（読み直しの検証は `publish_model` の中）。

    容器を書く口はここだけ — 台本側が `graph_name=` を名乗らない理由は `probe.export_probe`。
    """
    export = probe.export_probe(model)
    out_dir = tmp_path_factory.mktemp("umt5") / probe.TARGET
    out_dir.mkdir()
    publish_model(
        out_dir / probe.MODEL_FILE,
        export.graph,
        dict(export.tensors),
        provenance=probe.provenance(),
        graph_name=probe.TARGET,
        weight_dtype="i8",
        weight_scales=export.scales,
        weight_dtype_overrides=export.overrides,
    )
    for name, io in export.golden.items():
        save_file(dict(io), str(out_dir / name))
    return probe.summarize(export), out_dir


class TestExport:
    def test_inputs_outputs_and_symbol(self, exported):
        summary, _ = exported

        assert summary["inputs"] == [
            ["input_ids", [1, "L"], "i32"],
            ["relative_position_buckets", ["L", "L"], "i32"],
        ]
        assert summary["symbols"] == ["L"]
        assert summary["outputs"] == [[1, "L", D_MODEL]]

    def test_ops_take_the_decomposed_attention_path(self, exported):
        """RMSNorm は `rms_norm` に畳まれ、活性は `gelu_tanh`。

        attention は融合 op に乗らず bmm → add → softmax → bmm の分解経路。
        """
        ops = exported[0]["ops"]

        assert ops["rms_norm"] == 2 * LAYERS + 1
        assert ops["gelu_tanh"] == LAYERS
        assert ops["embedding"] == 1 + LAYERS
        assert ops["linear"] == 7 * LAYERS
        assert ops["softmax"] == LAYERS
        assert ops["bmm"] == 2 * LAYERS
        assert "pow" not in ops
        assert "attention" not in ops

    def test_relative_bias_tables_stay_f32(self, exported):
        assert exported[0]["storage"] == {
            "embed_tokens": ["i8"],
            "linear": ["i8"],
            "other": ["f32"],
            up.RELATIVE_BIAS_ATTRIBUTE: ["f32"],
        }

    def test_emit_without_the_override_fails_on_a_table(self, exported):
        message = exported[0]["emit_without_overrides"]

        assert "per-channel scale が無い" in message
        assert "relative_attention_bias" in message

    def test_golden_is_written_for_each_length(self, exported):
        summary, out_dir = exported

        names = [f"io.l{length:03d}.safetensors" for length in probe.EXPORT_LENGTHS]
        assert list(summary["golden"]) == names
        for length, name in zip(probe.EXPORT_LENGTHS, names, strict=True):
            io = load_file(str(out_dir / name))
            assert {key: (value.dtype, tuple(value.shape)) for key, value in io.items()} == {
                "input.input_ids": (torch.int32, (1, length)),
                "input.relative_position_buckets": (torch.int32, (length, length)),
                "output.0": (torch.float32, (1, length, D_MODEL)),
            }
        assert sorted(path.name for path in out_dir.glob("model-*.krm"))


class TestMaskEquivalence:
    def test_f32_differs_only_by_the_reduction_order(self, model):
        rows = probe.mask_equivalence(model, torch.float32, LENGTHS)

        assert max(row["ratio"] for row in rows) <= MASK_F32_RATIO_BOUND

    def test_bf16_is_bit_exact(self, model):
        """調査 §2.3 の観察（4 層・乱数初期化で 7 本ともビット一致）の再現。"""
        rows = probe.mask_equivalence(model, torch.bfloat16, LENGTHS)

        assert [row["length"] for row in rows if not row["bit_exact"]] == []

    @pytest.mark.parametrize("length", [2, 28, 300])
    def test_padding_without_the_mask_is_far_off(self, model, length):
        """マスクを落とした詰め方は大きく割れる（上の一致が恒真でないことの対）。"""
        ids = probe.tiny_ids(length)
        padded = torch.zeros((1, up.MAX_LENGTH), dtype=torch.long)
        padded[:, :length] = ids

        with torch.no_grad():
            unmasked = model(input_ids=padded).last_hidden_state[:, :length]

        assert up.max_ratio(unmasked, up.valid_output(model, ids)) >= FAULT_RATIO_FLOOR
