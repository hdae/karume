"""The recipe-side rewrites of the Qwen3.5 text tower keep upstream's numbers (`qwen3_5.patch`).

Each rewrite is compared with the upstream code it replaces, on random inputs: bit for bit where
the rewrite claims it (the `1 + w` norms, the causal conv), within the measured ULP band where it
does not (the l2-normalisation — research 2026-10-31 §6.1). The whole tiny tower is compared with
upstream's stock `Qwen3_5TextModel`, and the fault injections show the comparison can fail (a
non-causal mask, rows in the wrong order).
"""

from __future__ import annotations

import copy

import pytest

pytest.importorskip("transformers")

import torch
from torch.nn import functional

from qwen3_5 import patch
from qwen3_5.tests.text_tower_fixture import stock_model, text_config

#: Ratio bound (max |Δ| / max |reference|) of the whole tiny tower against the stock model. The
#: measured value is 2e-7 (the l2-normalisation's ≤ 4 ULP, SDPA with an explicit mask against
#: is_causal, the f32 silu path); 1e-5 leaves a 50× margin and stays 4 orders below what a
#: non-causal mask or swapped RoPE rows cause (the fault injections below).
TOWER_RATIO = 1e-5


def ratio(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return float((actual.double() - expected.double()).abs().max() / expected.double().abs().max())


def ulp_distance(a: torch.Tensor, b: torch.Tensor) -> int:
    """Largest distance in units of the last place between two f32 tensors of one sign pattern."""
    ai, bi = a.contiguous().view(torch.int32).long(), b.contiguous().view(torch.int32).long()
    return int((ai - bi).abs().max())


@pytest.fixture(scope="module")
def stock():
    return stock_model()


@pytest.fixture(scope="module")
def tower(stock):
    model = copy.deepcopy(stock)
    patch.register_attention()
    model.config._attn_implementation = patch.ATTENTION_NAME
    return patch.prepare_text_tower(model)


class TestTheZeroCentredNorm:
    def test_the_offset_weight_reproduces_upstream_bit_for_bit(self):
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5RMSNorm

        torch.manual_seed(1)
        norm = Qwen3_5RMSNorm(64, eps=1e-6)
        with torch.no_grad():
            norm.weight.normal_(0.0, 0.3)
        offset = patch.UnitOffsetRmsNorm(patch.unit_offset(norm.weight.detach()), norm.eps)
        x = torch.randn(3, 17, 64) * 5

        assert torch.equal(offset(x), norm(x))

    def test_an_un_offset_weight_does_not(self):
        """Fault injection: storing `w` itself (forgetting the offset) is caught."""
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5RMSNorm

        torch.manual_seed(1)
        norm = Qwen3_5RMSNorm(64, eps=1e-6)
        with torch.no_grad():
            norm.weight.normal_(0.0, 0.3)
        raw = patch.UnitOffsetRmsNorm(norm.weight.detach().clone(), norm.eps)
        x = torch.randn(3, 17, 64)

        assert not torch.allclose(raw(x), norm(x), atol=1e-3)

    def test_exactly_the_zero_centred_norms_are_offset(self, tower):
        """2 per layer + q/k of the full-attention layer + the final norm; never the GDN norm."""
        keys = tower.unit_offset_keys

        assert len(keys) == 2 * 4 + 2 * 1 + 1
        assert "norm.weight" in keys
        assert "layers.3.self_attn.q_norm.weight" in keys
        assert not [key for key in keys if "linear_attn" in key]


class TestTheCausalConv:
    @pytest.mark.parametrize("length", [1, 2, 5, 64])
    def test_it_is_upstreams_conv_and_silu_bit_for_bit(self, length):
        torch.manual_seed(length)
        conv = torch.nn.Conv1d(48, 48, kernel_size=4, groups=48, bias=False, padding=3)
        x = torch.randn(1, 48, length)
        expected = functional.silu(conv(x)[:, :, :length].contiguous())

        actual = patch.causal_conv1d(x, conv.weight.squeeze(1), None, activation="silu")

        assert torch.equal(actual, expected)

    def test_it_refuses_another_activation_or_packed_sequences(self):
        x, weight = torch.zeros(1, 8, 5), torch.zeros(8, 4)

        with pytest.raises(NotImplementedError):
            patch.causal_conv1d(x, weight, activation=None)
        with pytest.raises(NotImplementedError):
            patch.causal_conv1d(x, weight, activation="silu", seq_idx=torch.zeros(1))


class TestTheRecurrence:
    def test_the_l2_normalisation_is_within_4_ulp_of_upstream(self):
        from transformers.models.qwen3_5.modeling_qwen3_5 import l2norm

        torch.manual_seed(2)
        x = torch.randn(1, 512, 4, 128) * 3

        assert ulp_distance(patch.l2_normalize(x), l2norm(x, dim=-1, eps=1e-6)) <= 4

    def test_the_swap_matches_upstreams_chunked_function(self):
        """The wrapper (l2norm rewrite + the custom op's eager body = upstream's function) against
        upstream's call with `use_qk_l2norm_in_kernel=True`: only the normalisation differs."""
        from transformers.models.qwen3_5.modeling_qwen3_5 import torch_chunk_gated_delta_rule

        torch.manual_seed(3)
        q, k = torch.randn(1, 70, 4, 16), torch.randn(1, 70, 4, 16)
        v = torch.randn(1, 70, 4, 16)
        g = -torch.rand(1, 70, 4) * 2
        beta = torch.rand(1, 70, 4)
        expected, _ = torch_chunk_gated_delta_rule(
            q, k, v, g=g, beta=beta, use_qk_l2norm_in_kernel=True
        )

        actual, state = patch.gated_delta_rule(
            q, k, v, g=g, beta=beta, use_qk_l2norm_in_kernel=True, cu_seqlens=None
        )

        assert state is None
        assert ratio(actual, expected) < 1e-6

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"initial_state": torch.zeros(1, 4, 16, 16), "use_qk_l2norm_in_kernel": True},
            {"output_final_state": True, "use_qk_l2norm_in_kernel": True},
            {"use_qk_l2norm_in_kernel": False},
            {"use_qk_l2norm_in_kernel": True, "cu_seqlens": torch.tensor([0, 3])},
        ],
        ids=["initial-state", "final-state", "no-l2norm", "packed"],
    )
    def test_it_refuses_anything_but_the_stateless_prefill(self, kwargs):
        x = torch.zeros(1, 3, 4, 16)

        with pytest.raises(NotImplementedError):
            patch.gated_delta_rule(x, x, x, torch.zeros(1, 3, 4), torch.zeros(1, 3, 4), **kwargs)

    def test_the_decode_path_is_unreachable(self):
        with pytest.raises(NotImplementedError):
            patch.recurrent_gated_delta_rule()


class TestTheAttention:
    def test_a_missing_mask_fails_loudly(self):
        q = torch.zeros(1, 4, 3, 8)

        with pytest.raises(ValueError, match="without a mask"):
            patch.gqa_attention(torch.nn.Identity(), q, q[:, :2], q[:, :2], None)

    def test_the_mask_is_causal_over_the_positions(self):
        mask = patch.causal_mask(torch.arange(5, dtype=torch.float32))

        assert torch.equal(mask[0, 0], torch.ones(5, 5, dtype=torch.bool).tril())


class TestTheTower:
    @pytest.mark.parametrize("length", [1, 10, 23])
    def test_it_follows_the_stock_text_model(self, stock, tower, length):
        torch.manual_seed(length)
        ids = torch.randint(0, 320, (1, length))
        with torch.no_grad():
            expected = stock(input_ids=ids).last_hidden_state
            actual = tower(*patch.inputs_for(ids, stock.config))

        assert ratio(actual, expected) < TOWER_RATIO

    def test_a_non_causal_mask_is_caught(self, stock, tower, monkeypatch):
        """Fault injection: an all-true mask (attention sees the future)."""
        monkeypatch.setattr(
            patch,
            "causal_mask",
            lambda p: torch.ones(1, 1, p.shape[0], p.shape[0], dtype=torch.bool),
        )
        ids = torch.randint(0, 320, (1, 23), generator=torch.Generator().manual_seed(4))
        with torch.no_grad():
            expected = stock(input_ids=ids).last_hidden_state
            actual = tower(*patch.inputs_for(ids, stock.config))

        assert ratio(actual, expected) > 100 * TOWER_RATIO

    def test_rope_rows_in_the_wrong_order_are_caught(self, stock, tower):
        """Fault injection: the RoPE rows in reverse order. (A shift of every position by the same
        amount is not a fault: RoPE scores depend on the distance only.)"""
        ids = torch.randint(0, 320, (1, 23), generator=torch.Generator().manual_seed(5))
        ids_, positions, _, _ = patch.inputs_for(ids, stock.config)
        cos, sin = patch.rope_rows(range(22, -1, -1), stock.config)
        with torch.no_grad():
            expected = stock(input_ids=ids).last_hidden_state
            actual = tower(ids_, positions, cos, sin)

        assert ratio(actual, expected) > 100 * TOWER_RATIO

    def test_it_refuses_a_bf16_upstream(self, stock):
        model = copy.deepcopy(stock).to(torch.bfloat16)
        model.config._attn_implementation = patch.ATTENTION_NAME

        with pytest.raises(patch.Qwen35ConfigError, match="f32 only"):
            patch.Qwen35TextTower(model)

    def test_it_refuses_another_activation(self):
        with pytest.raises(patch.Qwen35ConfigError, match="hidden_act"):
            patch.check_supported(text_config(hidden_act="gelu"))


class TestTheHostRopeRows:
    def test_they_follow_upstreams_rotary_module_within_the_position_bound(self):
        """ADR 0091 decision 2's bound, over the whole range of `T`."""
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding

        config = text_config(
            head_dim=256,
            rope_parameters={
                "rope_type": "default",
                "rope_theta": 1e7,
                "partial_rotary_factor": 0.25,
                "mrope_section": [11, 11, 10],
                "mrope_interleaved": True,
            },
        )
        length = patch.MAX_TOKENS
        position = torch.arange(length)[None, None, :].expand(3, 1, -1)
        cos, sin = Qwen3_5TextRotaryEmbedding(config)(torch.zeros(1, length, 1), position)
        host_cos, host_sin = patch.rope_rows(range(length), config)
        bound = patch.rope_bound(range(length))[None, :, None]

        assert host_cos.shape == cos.shape == (1, length, 64)
        assert ((host_cos.double() - cos.double()).abs() <= bound).all()
        assert ((host_sin.double() - sin.double()).abs() <= bound).all()

    def test_a_shifted_position_breaks_the_bound(self):
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding

        config = text_config()
        position = torch.arange(64)[None, None, :].expand(3, 1, -1)
        cos, _ = Qwen3_5TextRotaryEmbedding(config)(torch.zeros(1, 64, 1), position)
        shifted, _ = patch.rope_rows(range(1, 65), config)

        assert (
            (shifted.double() - cos.double()).abs() > patch.rope_bound(range(64))[None, :, None]
        ).any()
