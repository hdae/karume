"""The contract of `wan/umt5_patch.py` (the S-form wrapper of umT5 over the valid length and the
Python-side source of truth of the bucket table) and `wan/umt5_probe.py` (ADR 0119 decisions 3–5,
the preview of step 10c).

What is pinned is the side where a break raises no exception and **only the numbers drift
quietly**:

- the bias looked up from the input bucket table through the layer's table is **bit-identical** to
  the upstream `compute_bias` (the table orientation `table[i][j] = bucket(j − i)` and the formula)
- the wrapper with the upstream activation left in place is **bit-identical** in f32 to the
  upstream forward (the rewrite is equivalent). Replacing `gelu_new` → `nn.GELU(approximate="tanh")`
  differs only by rounding
- the wrapper really uses the input table (transposing the table or shifting it by 1 moves the
  output a lot — the pair that keeps the match above from being tautological)
- the exporter's default i8 targets include the relative-position tables (confirming the guess of
  the investigation §1.2). The wrapper's selection removes them, and emit fails without the explicit
  F32 (decision 5's "form that keeps them F32")
- the S-form export: two inputs (i32), one symbol `L`, RMSNorm folded into `rms_norm`, the
  activation `gelu_tanh`, no `pow` left, the tables stored as F32, the golden can be taken on CPU
- "valid length only" matches "512 + mask" (reproducing the investigation §2.3 — the match on the
  real model is the acceptance of step 10c)

It runs only on a small randomly initialized umT5 (`wan.umt5_probe.TINY_CONFIG`) — no real weights
are read. transformers exists only in the `wan` group, so the fixture that builds the model calls
`importorskip`.
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

#: The number of layers (`TINY_CONFIG` — the expected op counts derive from it).
LAYERS = int(probe.TINY_CONFIG["num_layers"])
D_MODEL = int(probe.TINY_CONFIG["d_model"])

#: The upper bound of the difference between the output with `gelu_new` replaced by the tanh
#: approximation `aten.gelu` and the upstream (ratio = maximum absolute difference ÷ maximum
#: absolute value of the reference). The worst measured (2026-10-03, the 8 lengths of `LENGTHS`)
#: is 3.93e-7 (L = 511), and this is about 5 times that. A mixed-up table (the fault injection
#: below) gives 2.6e-2 or more, five orders of magnitude away.
TANH_RATIO_BOUND = 2e-6

#: The upper bound of the f32 difference between "valid length only" and "512 + mask" (the same
#: ratio). The worst measured is 7.02e-7 (L = 2 — 6 of the 8 are bit-identical; the difference is
#: the GEMM reduction order), and this is about 5 times that. Dropping the mask gives 1.1 or more.
MASK_F32_RATIO_BOUND = 3.5e-6

#: The ratio the table fault injections move at least (the smallest measured is the transpose's
#: 2.6e-2 at L = 2).
FAULT_RATIO_FLOOR = 1e-2

#: The lengths measured (the lower bound, the fixed prompt, beyond max_distance, the upper bound).
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
        """In every layer: looking the table up through the layer's table and rotating it to
        `[1, heads, L, L]` gives exactly the upstream `compute_bias`."""
        for attention in attentions(model):
            table = up.relative_position_buckets(length, attention)
            with torch.no_grad():
                gathered = attention.relative_attention_bias(table).permute([2, 0, 1]).unsqueeze(0)
                expected = attention.compute_bias(length, length)

            assert table.shape == (length, length)
            assert torch.equal(gathered, expected)

    def test_meta_attention_builds_the_same_table_as_the_model(self, model):
        """The same table from the upstream module without weights (the entry point used to
        generate the fixture)."""
        meta = up.bucket_attention(model.config)

        assert torch.equal(
            up.relative_position_buckets(300, meta),
            up.relative_position_buckets(300, attentions(model)[0]),
        )

    @pytest.mark.parametrize("length", [0, 1, up.MAX_LENGTH + 1])
    def test_lengths_outside_the_graph_domain_are_rejected(self, model, length):
        with pytest.raises(ValueError, match="is outside"):
            up.relative_position_buckets(length, attentions(model)[0])

    @pytest.mark.parametrize("length", [True, 28.0])
    def test_non_integer_lengths_are_rejected(self, model, length):
        with pytest.raises(TypeError):
            up.relative_position_buckets(length, attentions(model)[0])

    def test_decoder_config_is_rejected(self, model):
        """Unidirectional (decoder) buckets use another formula — only bidirectional ones are
        handled."""
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
        # The replacement really takes effect (0 with the upstream activation left in place — the
        # pair that keeps the upper bound above from being tautological).
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
        """The table is really read from the input — a mixed-up table moves the output far beyond
        the rounding digits."""
        wrapper = up.Umt5EncoderTokens(model, activation="upstream")
        ids, table = probe.case_inputs(wrapper, length)

        with torch.no_grad():
            ratio = up.max_ratio(wrapper(ids, fault(table)), up.valid_output(model, ids))

        assert ratio >= FAULT_RATIO_FLOOR

    def test_training_mode_is_rejected(self, model):
        training = copy.deepcopy(model).train()

        with pytest.raises(up.Umt5ConfigError, match="training mode"):
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
        """The exporter's default i8 (`nn.Embedding` in `QUANT_CHANNEL_AXES`) rounds the tables
        too."""
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
    """Write the preview export through to the container and golden files (the read-back
    verification is inside `publish_model`).

    This is the only place that writes a container — why the script side does not state
    `graph_name=` is explained in `probe.export_probe`.
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
        """RMSNorm is folded into `rms_norm`, and the activation is `gelu_tanh`.

        attention does not land on a fused op and takes the decomposed path bmm → add → softmax →
        bmm.
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

        assert "has no per-channel scale" in message
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
        """Reproduces the observation of the investigation §2.3 (4 layers, random initialization,
        all 7 bit-identical)."""
        rows = probe.mask_equivalence(model, torch.bfloat16, LENGTHS)

        assert [row["length"] for row in rows if not row["bit_exact"]] == []

    @pytest.mark.parametrize("length", [2, 28, 300])
    def test_padding_without_the_mask_is_far_off(self, model, length):
        """Packing with the mask dropped splits a lot (the pair that keeps the match above from
        being tautological)."""
        ids = probe.tiny_ids(length)
        padded = torch.zeros((1, up.MAX_LENGTH), dtype=torch.long)
        padded[:, :length] = ids

        with torch.no_grad():
            unmasked = model(input_ids=padded).last_hidden_state[:, :length]

        assert up.max_ratio(unmasked, up.valid_output(model, ids)) >= FAULT_RATIO_FLOOR
