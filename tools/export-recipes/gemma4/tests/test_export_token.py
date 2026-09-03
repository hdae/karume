"""token-only 既定出口の台本（`gemma4/export_token.py`）の挙動テスト。

tiny な実モデルの一周（export → 手術 → 混成量子化つき書き出し → 形検査）と、
**logits opt-in 形との eager 同値**（同じ重みを共有する 2 ラッパで、選んだ行の token が
一致する）を縛る。実重み・GPU は使わない。
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
    """同じ重みを共有する (logits opt-in ラッパ, token-only ラッパ)。

    MUST: model / tables の**実体を共有**する — 別々に組むと乱数重みが割れて eager 同値の
    突合が「一致すべき前提」を失う。共有できるのは両ラッパがモジュールを持つだけで
    状態を書かないから。
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
        """行選択 + 1 行 lm_head + softcap + argmax が opt-in 形の同じ行の token と一致する。

        全行を踏む（最終行だけだと行選択の添字ずれ — 例えば常に 0 行 — が素通りする）。
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
                assert tuple(selected.shape) == (1, 1, 1), f"row {row} の出力形"
                assert int(selected[0, 0, 0]) == int(reference[0, row, 0]), f"row {row} の token"


class TestExportedTokenForm:
    """tiny な実モデルを export → 手術 → 混成量子化つき書き出しまで通す（transformers が要る）。"""

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
        """`token_only` の指定し忘れ（またはその逆）が黙って通らないことの固定。"""
        verified, config, storage = tiny_container

        with pytest.raises(AssertionError, match="グラフ入力"):
            decode.assert_ir_form_decode(verified, config, storage)
