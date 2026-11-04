"""Behaviour tests of the script for the default token-only exit (`gemma4/export_token.py`).

Pins one round trip of a tiny real model (export → surgery → write with mixed quantization → form
check) and **eager equivalence with the logits opt-in form** (two wrappers sharing the same weights
produce the same token for the selected row). No real weights or GPU are used.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn
from torch.export import Dim

from gemma4 import export as gx
from gemma4 import export_decode as decode
from gemma4 import export_token as token
from gemma4.tests.test_export import PLE_DIM, TINY_SYM_MAX, VOCAB, WINDOW
from gemma4.tests.test_export_decode import (
    DECODE_LAYER_TYPES,
    OWNER_LAYERS,
    _tiny_decode_config,
)
from karume.convert import PRESERVED_OP_PREFIXES_WITH_ATTENTION
from karume.pipeline import export_module
from karume.states import to_states_form


@pytest.fixture
def tiny_model_pair():
    """(logits opt-in wrapper, token-only wrapper) sharing the same weights.

    MUST: **share the objects** of model / tables — building them separately splits the random
    weights, and the eager-equivalence comparison loses its premise that the two should agree.
    Sharing works because both wrappers only hold the modules and write no state.
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
        decode.DecodeChunkWrapper(model, tables).eval(),
        token.TokenOnlyChunkWrapper(model, tables).eval(),
    )


class TestEagerEquivalence:
    def test_the_selected_row_token_matches_the_logits_form(self, tiny_model_pair):
        """Row selection + one-row lm_head + softcap + argmax matches the opt-in form's token for
        the same row.

        Every row is exercised (with only the last row, an index slip in the row selection — for
        example always row 0 — would pass unnoticed).
        """
        logits_form, token_form = tiny_model_pair
        torch.manual_seed(1)
        ids = torch.randint(0, VOCAB, (1, 7), dtype=torch.int64)
        specs = decode.rope_specs(logits_form.model.config)
        rope = decode.rope_args(specs, decode.positions_for(ids))
        with torch.no_grad():
            _, reference = logits_form(ids, *rope)
            for row in range(int(ids.shape[1])):
                selected = token_form(ids, *rope, torch.tensor([row], dtype=torch.int64))
                assert tuple(selected.shape) == (1, 1, 1), f"row {row}: output shape"
                assert int(selected[0, 0, 0]) == int(reference[0, row, 0]), f"row {row}: token"


class TestExportedTokenForm:
    """Runs a tiny real model through export → surgery → write with mixed quantization (needs
    transformers)."""

    @pytest.fixture
    def tiny_container(self, tiny_model_pair, tmp_path):
        _, wrapper = tiny_model_pair
        int8, int4, scales = gx.quantize_wrapper(wrapper)
        ids = torch.randint(0, VOCAB, (1, WINDOW + 3), dtype=torch.int64)
        seq = Dim(decode.SEQ_SYMBOL, min=2, max=TINY_SYM_MAX)
        specs = decode.rope_specs(wrapper.model.config)
        args, shapes = decode._export_args(token.VARIANT, ids, seq, specs)
        graph, tensors = export_module(
            wrapper,
            args,
            dynamic_shapes=shapes,
            symbol_names=(decode.SEQ_SYMBOL,),
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
        storage = {"i8": len(int8.scales), "i4": len(int4.scales)}
        return verified, config, storage

    def test_the_container_is_a_verified_token_only_graph(self, tiny_container):
        verified, config, storage = tiny_container

        form = decode.assert_ir_form_decode(verified, config, storage, token_only=True)

        assert form["attention_nodes"] == len(DECODE_LAYER_TYPES)
        assert form["state_append_nodes"] == 2 * OWNER_LAYERS
        assert [spec.name for spec in verified.inputs] == [
            decode.INPUT_IDS,
            *decode.ROPE_INPUTS,
            decode.TOKEN_ONLY_LAST_ROW,
        ]
        assert len(verified.outputs) == 1
        assert "argmax" in verified.required_ops

    def test_the_logits_form_check_rejects_the_token_only_graph(self, tiny_container):
        """Pins that forgetting `token_only` (or the reverse) does not pass silently."""
        verified, config, storage = tiny_container

        with pytest.raises(AssertionError, match="graph input"):
            decode.assert_ir_form_decode(verified, config, storage)
